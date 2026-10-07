"""`tools/verity/lean/upstream.py`: the no-build scan on a small git repository standing in for a dependency, the audit step at
the pinned revision, `--ack`, and the exact pass on the controls' fixture when `lake` is installed."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_upstream", HERE.parent / "upstream.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)

LIB = """namespace Lib

/-- A docstring may name mcaError and code. -/
theorem doc_only : True := trivial

theorem proved_one (x : Nat) : mcaError x ≤ RS.code x := by simp [mcaError]

theorem open_one : mcaError 1 = code 1 := sorry

theorem proof_only : True := by have := mcaError; exact code -- the names only in the proof

theorem primed : mcaError' 1 = code 1 := rfl

theorem by_cases : ∀ n : Nat, mcaError n = code n
  | 0 => rfl
  | n + 1 => rfl

end Lib
"""
WATCH = {"mca": {"about": "our assumption A", "mentions": ["Defs.mcaError", "RS.code"], "ack": []}}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def dependency(root: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = root / "dep"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "dep")
    return repo, git(repo, "rev-parse", "HEAD")


def package(tmp_path: Path, watch: dict | None = None) -> tuple[Path, Path, str]:
    """A package pinning a `Lib` dependency whose checkout is the repository above, with an `upstream` section."""
    repo, rev = dependency(tmp_path, {"Lib/A.lean": LIB, "scripts/axiom_baseline.json": json.dumps({"sorry": ["Lib.open_one"]})})
    pkg = tmp_path / "pkg"
    (pkg / ".lake" / "packages").mkdir(parents=True)
    shutil.move(str(repo), pkg / ".lake" / "packages" / "Lib")
    (pkg / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "Lib", "type": "git", "rev": rev}]}))
    cfg = {"scan": {"Lib": {"baseline": "scripts/axiom_baseline.json", "sources": ["Lib"]}}, "imports": ["Lib"],
           "watch": json.loads(json.dumps(watch or WATCH))}
    (pkg / "lean-audit.json").write_text(json.dumps({"roots": ["P"], "upstream": cfg}))
    return pkg, pkg / ".lake" / "packages" / "Lib", rev


@pytest.mark.slow
def test_the_scan_matches_statements_not_docstrings_proofs_or_other_names(tmp_path):
    pkg, repo, rev = package(tmp_path)
    scanned = {"baseline": "scripts/axiom_baseline.json", "sources": ["Lib"]}
    hits = {h["name"]: h["proved"] for h in U.scan(repo, rev, scanned, WATCH)["mca"]}
    assert hits == {"Lib.proved_one": True, "Lib.open_one": False, "Lib.by_cases": True}


def test_the_scan_reads_a_file_whose_name_git_quotes(tmp_path):
    """`git ls-tree --name-only` quotes a name with bytes outside ASCII (`"Lib/D\\303\\251fs.lean"`), which doesn't end in `.lean`,
    so the scan never read such a file, and a theorem there proving a watched assumption went unseen and the audit passed."""
    repo, rev = dependency(tmp_path, {"Lib/Défs.lean": LIB, "scripts/axiom_baseline.json": json.dumps({"sorry": ["Lib.open_one"]})})
    scanned = {"baseline": "scripts/axiom_baseline.json", "sources": ["Lib"]}
    assert {h["name"] for h in U.scan(repo, rev, scanned, WATCH)["mca"]} == {"Lib.proved_one", "Lib.open_one", "Lib.by_cases"}


def test_new_is_proved_and_not_acknowledged():
    hits = {"mca": [{"name": "a", "proved": True}, {"name": "b", "proved": True}, {"name": "c", "proved": False}]}
    assert U.new(hits, {"mca": {"ack": ["a"]}}) == {"mca": ["b"]}


def test_the_audit_step_fails_until_a_new_proof_is_acknowledged(tmp_path):
    pkg, repo, rev = package(tmp_path)
    cfg = json.loads((pkg / "lean-audit.json").read_text())["upstream"]
    status, fails = U.pinned(pkg, cfg)
    assert [f.split(",")[0] for f in fails] == [f"upstream: Lib at {rev[:12]} proves Lib.by_cases",
                                                f"upstream: Lib at {rev[:12]} proves Lib.proved_one"]
    assert status["Lib"]["entries"]["mca"] == {"hits": 3, "proved": 2, "new": ["Lib.by_cases", "Lib.proved_one"]}
    cfg["watch"]["mca"]["ack"] = ["Lib.by_cases", "Lib.proved_one"]
    assert U.pinned(pkg, cfg)[1] == []


def test_the_audit_step_reads_the_pinned_revision_not_the_checkout(tmp_path):
    pkg, repo, rev = package(tmp_path, {"mca": {**WATCH["mca"], "ack": ["Lib.by_cases", "Lib.proved_one"]}})
    (repo / "Lib" / "B.lean").write_text("namespace Lib\ntheorem later : mcaError 2 = code 2 := rfl\nend Lib\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "upstream moves on")
    cfg = json.loads((pkg / "lean-audit.json").read_text())["upstream"]
    assert U.pinned(pkg, cfg)[1] == []
    assert U.pinned(pkg, cfg, rev="HEAD")[1][0].split(",")[0].endswith("proves Lib.later")


def test_a_missing_checkout_is_named(tmp_path):
    pkg, repo, rev = package(tmp_path)
    shutil.rmtree(repo)
    cfg = json.loads((pkg / "lean-audit.json").read_text())["upstream"]
    assert U.pinned(pkg, cfg)[1][0].startswith(f"upstream: {repo} is not a git checkout")


@pytest.mark.slow
def test_fetch_clones_a_missing_checkout_from_the_manifest_at_the_pinned_revision(tmp_path, capsys):
    """The weekly look at upstream's head runs on a machine that never built the package: `--fetch` needs git, not `setup.sh`."""
    pkg, repo, rev = package(tmp_path, {"mca": {**WATCH["mca"], "ack": ["Lib.by_cases", "Lib.proved_one"]}})
    origin = tmp_path / "origin"
    shutil.move(str(repo), origin)
    (origin / "Lib" / "B.lean").write_text("namespace Lib\ntheorem later : mcaError 2 = code 2 := rfl\nend Lib\n")
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "upstream moves on")
    manifest = json.loads((pkg / "lake-manifest.json").read_text())
    manifest["packages"][0]["url"] = str(origin)
    (pkg / "lake-manifest.json").write_text(json.dumps(manifest))
    assert U.main([str(pkg), "--fetch", "--rev", "origin/HEAD"]) == 1
    assert "NEW    Lib.later" in capsys.readouterr().out
    assert git(repo, "rev-parse", "HEAD") == rev                    # checked out where the manifest pins it, as lake would
    assert U.main([str(pkg), "--fetch"]) == 0                       # the pinned revision, from the clone


@pytest.mark.slow
def test_the_command_exits_1_on_news_and_ack_records_them(tmp_path, capsys):
    pkg, repo, rev = package(tmp_path)
    assert U.main([str(pkg)]) == 1
    assert "NEW    Lib.proved_one" in capsys.readouterr().out
    assert U.main([str(pkg), "--ack"]) == 0
    assert json.loads((pkg / "lean-audit.json").read_text())["upstream"]["watch"]["mca"]["ack"] == \
        ["Lib.by_cases", "Lib.proved_one"]
    assert U.main([str(pkg)]) == 0


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_lean_pass_matches_elaborated_types_and_finds_sorry(tmp_path):
    pkg = tmp_path / "control"
    shutil.copytree(HERE / "fixtures" / "control", pkg, ignore=shutil.ignore_patterns(".lake"))
    with (pkg / "Control" / "Thms.lean").open("a") as f:
        f.write("\nnamespace Control\ntheorem bad : double 1 = 3 := sorry\nend Control\n")
    cfg = {"imports": ["Control"], "watch": {"doubling": {"about": "x", "mentions": ["Control.double"], "ack": []},
                                             "absent": {"about": "y", "mentions": ["Control.nosuch"], "ack": []}}}
    hits = U.lean_pass(pkg, cfg, tmp_path / "out")
    assert {h["name"]: (h["proved"], h["axioms"]) for h in hits["doubling"]} == {
        "Control.bad": (False, ["sorryAx"]), "Control.helper": (True, []), "Control.main_thm": (True, [])}
    assert hits["absent"] == []
