"""`tools/check/lean_audit.py`: audit verdicts reused only while every input is unchanged, the hermetic scratch tree as the read
guard, and dependencies kept warm across runs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("check_lean_audit", ROOT / "tools" / "check" / "lean_audit.py")
LA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LA)

#: a stand-in for tools/lean/audit.py --all --build --root R --out O: each package fails when a file its policy names under
#: "reads" is missing (an input the key would not cover), fetches its git dependencies into .lake/packages, and notes what ran
FAKE_AUDIT = r'''
import json, sys
from pathlib import Path
a = sys.argv[1:]
root, out = Path(a[a.index("--root") + 1]), Path(a[a.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
reps, failed = [], []
for pol in sorted(root.rglob("lean-audit.json")):
    pkg = pol.parent
    if "tools/lean" in str(pkg):
        continue
    missing = [r for r in json.loads(pol.read_text()).get("reads", []) if not (root / r).exists()]
    deps = pkg / ".lake" / "packages" / "dep"
    warm = (deps / "built").exists()
    if json.loads((pkg / "lake-manifest.json").read_text())["packages"]:
        deps.mkdir(parents=True, exist_ok=True)
        (deps / "built").write_text("x")
    with open(out / "ran.txt", "a") as f:
        f.write(f"{pkg.relative_to(root)} {'warm' if warm else 'cold'}\n")
    reps.append({"package": str(pkg), "failures": [f"reads {m}" for m in missing]})
    if missing:
        failed.append(str(pkg))
(out / "audit.json").write_text(json.dumps({"passed": not failed, "failed": failed, "packages": reps}))
sys.exit(1 if failed else 0)
'''


def _tree(tmp: Path, monkeypatch) -> Path:
    root = tmp / "repo"
    files = {
        "tools/lean/audit.py": FAKE_AUDIT,
        "tools/check/lean_audit.py": "", "tools/check/check.py": "",
        "pkg/a/lakefile.toml": "", "pkg/a/lean-toolchain": "v1\n", "pkg/a/A.lean": "theorem a : True := trivial\n",
        "pkg/a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"}]}),
        "pkg/a/lean-audit.json": "{}",
        "pkg/b/lakefile.toml": "", "pkg/b/lean-toolchain": "v1\n", "pkg/b/B.lean": "",
        "pkg/b/lake-manifest.json": json.dumps({"packages": [{"name": "a", "type": "path", "dir": "../a"}]}),
        "pkg/b/lean-audit.json": json.dumps({"reads": ["docs/outside.txt"]}),
        "docs/outside.txt": "read by b's audit, but no input of it",
    }
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    monkeypatch.setattr(LA, "ROOT", root)
    monkeypatch.setattr(LA.CHECK, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LA.CHECK, "cores", lambda: 2)
    return root


def _run(tmp: Path, n: int) -> tuple[int, list[str], dict]:
    out = tmp / f"out{n}"
    rc = LA.run(out, tmp / "cache" / "lean-audit", tmp / "cache" / "lean-deps", log=lambda s: None)
    ran = (out / "ran.txt").read_text().split("\n")[:-1] if (out / "ran.txt").exists() else []
    return rc, ran, json.loads((out / "audit.json").read_text())


def test_verdicts_are_reused_only_while_every_input_is_unchanged(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    # b reads a file outside its inputs: in the scratch tree it is not there, so b fails and nothing is stored for it
    rc, ran, doc = _run(tmp_path, 1)
    assert rc == 1 and sorted(ran) == ["pkg/a cold", "pkg/b cold"] and doc["failed"] == ["pkg/b"] and doc["cached"] == {}
    (root / "pkg/b/lean-audit.json").write_text("{}")
    rc, ran, doc = _run(tmp_path, 2)                   # a reused; b (and a, its path dependency) audited, a's dependencies warm
    assert rc == 0 and sorted(ran) == ["pkg/a warm", "pkg/b cold"] and list(doc["cached"]) == ["pkg/a"]
    rc, ran, doc = _run(tmp_path, 3)                   # nothing changed: nothing runs
    assert rc == 0 and ran == [] and sorted(doc["cached"]) == ["pkg/a", "pkg/b"] and doc["passed"]
    (root / "pkg/a/A.lean").write_text("theorem a : True := by trivial\n")
    rc, ran, doc = _run(tmp_path, 4)                   # a changed: a, and b which requires it by path, both run again
    assert rc == 0 and sorted(ran) == ["pkg/a warm", "pkg/b cold"] and doc["cached"] == {}
    (root / "tools/lean/audit.py").write_text(FAKE_AUDIT + "\n# the auditor changed\n")
    assert sorted(_run(tmp_path, 5)[1]) == ["pkg/a warm", "pkg/b cold"]     # the auditor is an input of every package


def test_the_dependencies_are_kept_by_manifest_and_toolchain(tmp_path, monkeypatch):
    root = _tree(tmp_path, monkeypatch)
    k = LA.deps_key(root / "pkg/a")
    assert k and LA.deps_key(root / "pkg/b") is None                         # b fetches nothing (a path dependency only)
    (root / "pkg/b/lean-audit.json").write_text("{}")
    _run(tmp_path, 1)
    assert (tmp_path / "cache" / "lean-deps" / k / "dep" / "built").exists()   # moved back after the run
    (root / "pkg/a/lean-toolchain").write_text("v2\n")
    assert LA.deps_key(root / "pkg/a") != k


def test_inputs_are_the_package_its_path_dependencies_and_the_auditor(tmp_path, monkeypatch):
    _tree(tmp_path, monkeypatch)
    files = LA.CHECK.tracked_files()
    assert LA.packages(files) == ["pkg/a", "pkg/b"]
    assert LA.path_deps("pkg/b") == ["pkg/a"]
    got = LA.inputs("pkg/b", files)
    assert "pkg/a/A.lean" in got and "pkg/b/B.lean" in got and "tools/lean/audit.py" in got and "docs/outside.txt" not in got
    nested = [*files, "pkg/a/inner/lakefile.toml", "pkg/a/inner/X.lean"]
    assert "pkg/a/inner/X.lean" not in LA.own_files("pkg/a", nested)


def test_the_real_packages_and_their_inputs():
    files = LA.CHECK.tracked_files()
    pkgs = LA.packages(files)
    assert "backends/flock/verifier/lean/soundness" in pkgs and not [p for p in pkgs if p.startswith("tools/lean")]
    got = LA.inputs("backends/flock/verifier/lean/soundness", files)
    assert "backends/flock/verifier/lean/level3/lean-audit.json" in got and "backends/flock/verifier/lean/Main.lean" in got
    assert "tools/lean/audit.py" in got and not [f for f in got if f.startswith("backends/flock/verifier/lean/soundness/.lake")]
