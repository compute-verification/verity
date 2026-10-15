"""`tools/verity/lean/merge.py`: two branches' `lean-audit.json` merged as `check.py --update` would write the merged tree's record,
and refused where a record can't say what that is; as a function, and as git's merge driver."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_merge", HERE.parent / "merge.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
A = M.A

STD = ["Classical.choice", "Quot.sound", "propext"]
FILES = {"P": "P.lean", "P.Defs": "P/Defs.lean", "P.More": "P/More.lean", "P.Thms": "P/Thms.lean"}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def read(name: str, module: str, h: str) -> dict:
    return {"name": name, "module": module, "kind": "def", "hash": h, "sha256": sha(h)}


F, G, H, K = read("P.f", "P.Defs", "1" * 16), read("P.g", "P.Defs", "2" * 16), read("P.h", "P.Defs", "3" * 16), \
    read("P.k", "P.More", "4" * 16)


def thm(reads: list[dict], statement: str = "P.f n = n", h: str = "00000000000000aa") -> dict:
    return {"kind": "theorem", "module": "P.Thms", "signature": f"(n : Nat) : {statement}", "axioms": STD, "type_hash": h,
            "type_sha256": sha(h), "hypotheses": [], "reads": reads}


def record(deps: dict | None = None, **guarantees: dict) -> dict:
    """The record `check.py --update` writes for a package whose guarantees are ``guarantees`` (name suffix: its facts)."""
    facts = {"lean": "4.34.0", "modules": list(FILES), "imports": {}, "declarations": {}, "axioms": STD, "offenders": [],
             "declared_axioms": [], "escapes": [], "pins": {f"P.{n}": f for n, f in guarantees.items()}}
    fails, pol = A.judge({"roots": ["P"], "meaning": ["P"], "guarantees": {f"P.{n}": {} for n in guarantees}}, facts, FILES,
                         update=True, owner="@lean")
    assert fails == []
    return pol | {"dependencies": deps or {"mathlib": "a" * 64, "toolchain": "c" * 64}}


def text_conflicts(base: dict, ours: dict, theirs: dict, tmp: Path) -> bool:
    for n, d in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp / n).write_text(A.text(d))
    return subprocess.run(["git", "merge-file", "-p", *(str(tmp / n) for n in ("ours", "base", "theirs"))],
                          capture_output=True).returncode != 0


def test_two_branches_that_each_add_guarantees_merge_to_the_record_of_the_merged_tree(tmp_path):
    base = record(t=thm([F]))
    ours = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), deps={"mathlib": "b" * 64, "toolchain": "c" * 64})
    theirs = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs)
    assert why == []
    tree = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"),
                  deps={"mathlib": "b" * 64, "toolchain": "c" * 64})
    assert A.text(merged) == A.text(tree)
    assert merged["reads"]["P.Defs"]["guarantees"] == ["P.t", "P.u", "P.v"] and list(merged["reads"]["P.Defs"]["definitions"]) == [
        "P.f", "P.g", "P.h"]
    assert text_conflicts(base, ours, theirs, tmp_path)
    assert M.merge(base, ours, ours) == (ours, [])


def test_a_hand_written_section_one_side_extends_is_kept():
    base = record(t=thm([F]))
    ours = base | {"meaning": ["P", "P.More"]}
    theirs = record(t=thm([F]), v=thm([H], "P.h n = n", "00000000000000cc"))
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and merged["meaning"] == ["P", "P.More"] and "P.v" in merged["guarantees"]
    assert M.merge(base, ours, base | {"meaning": ["P", "P.Other"]})[1] == ["meaning: changed on both sides, differently"]


def test_a_key_both_sides_changed_refuses():
    base = record(t=thm([F]))
    u, v = thm([F, G], "P.g n = n", "00000000000000bb"), thm([H], "P.h n = n", "00000000000000cc")
    restated = M.merge(base, record(t=thm([F], "P.f n ≤ n", "00000000000000dd"), u=u),
                       record(t=thm([F], "P.f n < n + 1", "00000000000000ee"), v=v))[1]
    assert restated == ["guarantees[P.t]: changed on both sides, differently"]
    moved = M.merge(base, record(t=thm([read("P.f", "P.Defs", "9" * 16)]), u=u),
                    record(t=thm([read("P.f", "P.Defs", "8" * 16)]), v=v))[1]
    assert moved == ["reads[P.Defs].definitions[P.f]: changed on both sides, differently"]
    bumped = M.merge(base, base | {"dependencies": {"mathlib": "b" * 64, "toolchain": "c" * 64}},
                     base | {"dependencies": {"mathlib": "d" * 64, "toolchain": "c" * 64}})[1]
    assert bumped == ["dependencies[mathlib]: changed on both sides, differently"]
    added = M.merge(base, record(t=thm([F]), u=u), record(t=thm([F]), u=thm([F], "P.f n = 0 + n", "00000000000000ff")))[1]
    assert added == ["guarantees[P.u]: added on both sides, differently"]


def test_an_owner_merges_apart_from_the_statement():
    base = record(t=thm([F]))
    unowned = {**base, "guarantees": {"P.t": A.unowned(base["guarantees"]["P.t"])}}
    restated = record(t=thm([F], "P.f n ≤ n", "00000000000000dd"))
    merged, why = M.merge(unowned, base, {**restated, "guarantees": {"P.t": A.unowned(restated["guarantees"]["P.t"])}})
    assert why == [] and merged["guarantees"] == restated["guarantees"], "one side's owner, the other's restatement"
    other = {**base, "guarantees": {"P.t": base["guarantees"]["P.t"] | {"owner": "@proofs"}}}
    assert M.merge(unowned, base, other)[1] == ["guarantees[P.t].owner: added on both sides, differently"]


def test_a_definition_one_side_stops_reading_refuses_when_the_other_side_changed_its_module():
    base = record(t=thm([F, G]))
    ours = record(t=thm([F], "P.f n + 0 = n", "00000000000000dd"))
    theirs = record(t=thm([F, G]), v=thm([H], "P.h n = n", "00000000000000cc"))
    assert M.merge(base, ours, theirs)[1] == [
        "reads[P.Defs].definitions[P.g]: no longer read on one side, while the other side changed what the guarantees read "
        "in P.Defs; the record does not say which guarantees read P.g"]
    assert M.merge(base, ours, record(t=thm([F, G]), v=thm([K], "P.k n = n", "00000000000000cc")))[1] == []


def test_a_32_bit_group_is_not_merged_with_a_rehashed_one():
    base = record(t=thm([F]))
    old = base | {"reads": A.read_groups(A.guarantee_reads({"pins": {"P.t": thm([F])}}), old=True)}
    ours = old | {"reads": A.read_groups(A.guarantee_reads({"pins": {"P.t": thm([F]), "P.u": thm([F, G])}}), old=True),
                  "guarantees": record(t=thm([F]), u=thm([F, G]))["guarantees"]}
    why = M.merge(old, ours, base)[1]
    assert why and all(w.startswith("reads[P.Defs]") for w in why)
    assert "reads[P.Defs]: its definitions would mix 32-bit and SHA-256 hashes (one side's record was rehashed)" in why


def before_the_rename(rec: dict) -> dict:
    """``rec`` as a branch from before guarantees were called that writes it: `pins`, and each module's readers as `pins`."""
    out = {("pins" if k == "guarantees" else k): v for k, v in rec.items()}
    out["reads"] = {m: {("pins" if k == "guarantees" else k): v for k, v in g.items()} for m, g in rec["reads"].items()}
    return out


def test_a_side_from_before_the_rename_merges_under_today_s_names(tmp_path):
    base = record(t=thm([F]))
    ours = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"))
    theirs = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    tree = record(t=thm([F]), u=thm([F, G], "P.g n = n", "00000000000000bb"), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    for n, d in (("base", before_the_rename(base)), ("ours", ours), ("theirs", before_the_rename(theirs))):
        (tmp_path / n).write_text(A.text(d))
    assert M.main([str(tmp_path / n) for n in ("base", "ours", "theirs")]) == 0
    assert (tmp_path / "ours").read_text() == A.text(tree)
    (tmp_path / "theirs").write_text(A.text(theirs | {"pins": {"P.w": {}}}))
    assert M.main([str(tmp_path / n) for n in ("base", "ours", "theirs")]) == 1
    assert (tmp_path / "ours").read_text() == A.text(tree)


def test_normalize_recomputes_each_digest_from_its_definitions(tmp_path):
    rec = record(t=thm([F]), v=thm([H, K], "P.h n = P.k n", "00000000000000cc"))
    stale = json.loads(json.dumps(rec))
    for g in stale["reads"].values():
        g["digest"] = "0" * 32
    path = tmp_path / "lean-audit.json"
    path.write_text(A.text(stale))
    assert M.main(["--normalize", str(path)]) == 0
    assert path.read_text() == A.text(rec)
    assert M.normalize(path) == 0


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert not check or r.returncode == 0, r.stderr
    return r


def test_git_merges_two_branches_that_add_guarantees_and_refuses_a_real_conflict(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text("lean-audit.json merge=lean-audit\n")
    rec = repo / "pkg" / "lean-audit.json"
    rec.parent.mkdir()
    git(repo, "init", "-q", "-b", "main")
    u, v = thm([F, G], "P.g n = n", "00000000000000bb"), thm([H, K], "P.h n = P.k n", "00000000000000cc")
    branches = {"main": record(t=thm([F])), "a": record(t=thm([F]), u=u), "b": record(t=thm([F]), v=v),
                "c": record(t=thm([F], "P.f n ≤ n", "00000000000000dd")),
                "d": record(t=thm([F], "P.f n < n + 1", "00000000000000ee"))}
    for name, doc in branches.items():
        if name != "main":
            git(repo, "checkout", "-q", "-b", name, "main")
        rec.write_text(A.text(doc))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "a")

    plain = git(repo, "merge", "--no-edit", "b", check=False)
    assert plain.returncode and "CONFLICT (content): Merge conflict in pkg/lean-audit.json" in plain.stdout
    git(repo, "merge", "--abort")

    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    assert git(repo, "config", "merge.lean-audit.driver").stdout.strip() == "python3 tools/verity/lean/merge.py %O %A %B %P"
    git(repo, "merge", "--no-edit", "b")
    assert rec.read_text() == A.text(record(t=thm([F]), u=u, v=v))

    git(repo, "checkout", "-q", "c")
    refused = git(repo, "merge", "--no-edit", "d", check=False)
    assert refused.returncode and "merge.py: pkg/lean-audit.json: refused" in refused.stderr
    assert "guarantees[P.t]: changed on both sides, differently" in refused.stderr
    assert git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split() == ["pkg/lean-audit.json"]
    assert rec.read_text() == A.text(branches["c"])


def test_shape_entries_and_frozen_names_merge_one_by_one():
    """Two branches that each freeze or enter a different guarantee merge; one entry set two ways is refused."""
    base = {"shape": {"premises": [], "properties": {}, "frozen": {"P.a": "RESTATE"}}}
    ours = {"shape": {"premises": [], "properties": {"P.b": {"function": "P.f", "outcome": "accept"}}, "frozen": {"P.a": "RESTATE"}}}
    theirs = {"shape": {"premises": [], "properties": {}, "frozen": {"P.c": "lemma"}}}
    merged, why = M.merge(base, ours, theirs)
    assert why == [] and merged["shape"] == {"premises": [], "properties": {"P.b": {"function": "P.f", "outcome": "accept"}},
                                             "frozen": {"P.c": "lemma"}}
    _, why = M.merge(base, {"shape": {"frozen": {"P.a": "x"}}}, {"shape": {"frozen": {"P.a": "y"}}})
    assert why == ["shape[frozen][P.a]: changed on both sides, differently"]


def test_two_branches_that_each_widen_a_layer_merge_its_imports_and_reasons():
    """#1860 against #1842 (12 Oct): each side added imports to `Definitions.Accounting` and a clause to its `why`."""
    base = {"layers": {"D": {"imports": ["Mathlib", "D"], "except": [], "why": "trusted text; they import no proof"},
                       "E": {"imports": ["E"], "except": [], "why": "e"}}}
    ours = {"layers": {"D": {"imports": ["Mathlib", "P.Audit", "D"], "except": [], "why": "trusted text; M reads P; they import no proof"},
                       "E": {"imports": ["E"], "except": [], "why": "e"}}}
    theirs = {"layers": {"D": {"imports": ["Mathlib", "Core", "D"], "except": [], "why": "trusted text; C reads Core; they import no proof"},
                         "E": {"imports": ["E", "F"], "except": [], "why": "e"}}}
    merged, why = M.merge(base, ours, theirs)
    assert why == []
    assert merged["layers"] == {"D": {"imports": ["Mathlib", "Core", "P.Audit", "D"], "except": [],
                                      "why": "trusted text; C reads Core; M reads P; they import no proof"},
                                "E": {"imports": ["E", "F"], "except": [], "why": "e"}}
    gone = {"layers": {"D": {"imports": ["D"], "except": [], "why": "trusted text"}, "E": base["layers"]["E"]}}
    merged, why = M.merge(base, gone, theirs)
    assert why == [] and merged["layers"]["D"] == {"imports": ["Core", "D"], "except": [], "why": "trusted text; C reads Core"}, \
        "what one side removed stays removed"
    _, why = M.merge(base, {"layers": {"D": {**ours["layers"]["D"], "mode": "x"}}}, {"layers": {"D": {**ours["layers"]["D"], "mode": "y"}}})
    assert why == ["layers[D].mode: added on both sides, differently"]


def test_two_branches_that_each_add_a_root_merge_the_roots():
    """c3-spec-2 against c3-n11-readcol (13 Oct): each side added a test module to Flock's roots."""
    base = {"roots": ["Flock", "Flock.Test.RecOpening", "VStar"]}
    ours = {"roots": ["Flock", "Flock.Test.RecOpen", "Flock.Test.RecOpening", "VStar"]}
    theirs = {"roots": ["Flock", "Flock.Test.RecOpening", "Flock.Test.ReadWhole", "VStar"]}
    merged, why = M.merge(base, ours, theirs)
    assert why == []
    assert merged["roots"] == ["Flock", "Flock.Test.RecOpen", "Flock.Test.RecOpening", "Flock.Test.ReadWhole", "VStar"]
    merged, why = M.merge(base, {"roots": ["Flock", "VStar"]}, theirs)
    assert why == [] and merged["roots"] == ["Flock", "Flock.Test.ReadWhole", "VStar"], "what one side removed stays removed"


def _lean(tmp: Path, base: str, ours: str, theirs: str) -> tuple[str, int]:
    for n, t in (("base", base), ("ours", ours), ("theirs", theirs)):
        (tmp / n).write_text(t)
    return M.lean(tmp / "base", tmp / "ours", tmp / "theirs")


def test_a_lean_file_whose_sides_only_add_imports_merges_and_any_other_conflict_stays_git_s(tmp_path):
    """#1860 against #1850 (12 Oct): each side added a module to Guarantees.lean's import list at the same place."""
    base = "import G.Bandwidth\nimport G.Joint\n\ntheorem t : True := trivial\n"
    ours = "import G.Memory\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := trivial\n"
    theirs = "import G.Actual\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := by trivial\n"
    assert _lean(tmp_path, base, ours, theirs) == (
        "import G.Actual\nimport G.Memory\nimport G.Bandwidth\nimport G.Joint\n\ntheorem t : True := by trivial\n", 0)
    text, left = _lean(tmp_path, base, ours.replace(":= trivial", ":= rfl"), theirs)
    assert left == 1 and "<<<<<<< ours" in text and "import G.Actual\nimport G.Memory\n" in text


def test_git_merges_two_branches_that_each_import_a_new_module_with_the_installed_driver(tmp_path, monkeypatch):
    for k, v in {"HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}.items():
        monkeypatch.setenv(k, v)
    repo = tmp_path / "repo"
    (repo / "tools" / "verity" / "lean").mkdir(parents=True)
    for f in ("merge.py", "check.py"):
        shutil.copy(HERE.parent / f, repo / "tools" / "verity" / "lean" / f)
    (repo / ".gitattributes").write_text(f"*.lean merge={M.LEAN}\n")
    g = repo / "G.lean"
    git(repo, "init", "-q", "-b", "main")
    for name, first in (("main", ""), ("a", "import G.Memory\n"), ("b", "import G.Actual\n")):
        if name != "main":
            git(repo, "checkout", "-q", "-b", name, "main")
        g.write_text(first + "import G.Joint\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
    git(repo, "checkout", "-q", "a")
    plain = git(repo, "merge", "--no-edit", "b", check=False)
    assert plain.returncode and "CONFLICT (content): Merge conflict in G.lean" in plain.stdout
    git(repo, "merge", "--abort")
    subprocess.run([sys.executable, "tools/verity/lean/merge.py", "--install"], cwd=repo, check=True, capture_output=True)
    git(repo, "merge", "--no-edit", "b")
    assert g.read_text() == "import G.Actual\nimport G.Memory\nimport G.Joint\n"
