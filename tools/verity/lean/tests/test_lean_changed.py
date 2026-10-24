"""`lean_changed.py --records --update`: the regenerated record survives where the run keeps it (LEAN-10), and a node
whose Lean slots grant no `build` slot takes the `--dev-cache` path.

In a recorded run the fast path never writes ROOT (the commit's shared source tree), and the slot tree's copy of the
updated `lean-audit.json` is restored to the committed bytes by the next requirer iteration's `sync`, so without an
explicit keep the record a lane must commit survives nowhere the run keeps. These tests stub the slot, the tree sync
and check.py itself, and check the keep under `out/record/` (recorded and not), the ROOT write-back (unrecorded
only), and which path a refused `build` slot takes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_changed", HERE.parent / "lean_changed.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)

OLD = json.dumps({"guarantees": {"G": {"owner": "@lean"}}}) + "\n"
NEW = json.dumps({"guarantees": {"G": {"owner": "@lean", "type_hash": "new"}}}) + "\n"


@pytest.fixture()
def fast(tmp_path, monkeypatch):
    """`audit_records` against a stand-in package `pkg`, a held-open slot tree and a fake check.py whose `--update`
    writes `NEW` over the tree's record; `sync` restores the committed record, as the real one does from ROOT's files."""
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "lean-audit.json").write_text(OLD)
    cache = tmp_path / "cache"
    tree = cache / LC.RECORDS / "local"
    (tree / "pkg").mkdir(parents=True)
    monkeypatch.setattr(LC, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "cache_dir", lambda: cache)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: ["pkg/lean-audit.json"])
    monkeypatch.setattr(LC.LA, "packages", lambda _: ["pkg"])
    monkeypatch.setattr(LC.LA, "deps_key", lambda _: "k")
    monkeypatch.setattr(LC, "requirers", lambda *_: ["pkg/Req"])
    monkeypatch.setattr(LC, "records_tree", lambda: tree)
    monkeypatch.setattr(LC, "link", lambda *a, **k: True)
    monkeypatch.setattr(LC, "unlink", lambda *a, **k: None)
    monkeypatch.setattr(LC, "label", lambda _: None)
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: None)

    def sync(_closure, _files, t):
        (t / "pkg" / "lean-audit.json").write_text(OLD)
        return 0

    monkeypatch.setattr(LC, "sync", sync)

    calls = []

    def run(argv, **kw):
        calls.append((argv, kw.get("cwd")))
        if "--update" in argv:
            (Path(argv[-1]) / "lean-audit.json").write_text(NEW)
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    return root, tree, calls


def test_a_recorded_update_keeps_the_regenerated_record_in_the_run_outputs(fast, tmp_path, monkeypatch):
    root, _, _ = fast
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    assert LC.audit_records("pkg", tmp_path / "deps", update=True) == 0
    kept = run_dir / "lean-audit" / "record" / "lean-audit.json"
    assert kept.read_text() == NEW, "the regenerated record is kept under the run's outputs"
    assert (root / "pkg" / "lean-audit.json").read_text() == OLD, "a recorded run never writes the shared source tree"


def test_an_unrecorded_update_writes_the_record_back_into_the_checkout(fast, tmp_path, monkeypatch):
    root, _, _ = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    out = tmp_path / "out"
    assert LC.audit_records("pkg", tmp_path / "deps", update=True, out=out) == 0
    assert (root / "pkg" / "lean-audit.json").read_text() == NEW, "the unrecorded copy-back still writes the checkout"
    assert (out / "record" / "lean-audit.json").read_text() == NEW, "and the keep is written either way"


def test_no_record_change_keeps_nothing(fast, tmp_path, monkeypatch):
    _, tree, _ = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)

    def run(argv, **_):
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    out = tmp_path / "out"
    assert LC.audit_records("pkg", tmp_path / "deps", update=True, out=out) == 0
    assert not (out / "record").exists(), "an unchanged record keeps nothing"


def refuse(*_a, **_k):
    raise LC.LEAN_SLOT.Refused("lean-slots lists no pool 'build', the only one this request takes")


def test_a_refused_build_slot_takes_the_dev_cache_path_on_the_checkout_and_its_requirers(fast, tmp_path, monkeypatch):
    root, tree, calls = fast
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", refuse)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", update=True, log=said.append) == 0
    assert [(a[a.index("--out") + 2:], cwd) for a, cwd in calls] == [([str(root / "pkg")], root),
                                                                       ([str(root / "pkg/Req")], root)]
    assert all("--dev-cache" in a and "--no-replay" in a for a, _ in calls), "both steps take the --dev-cache path"
    assert "--update" in calls[0][0] and "--update" not in calls[1][0], "a requirer is checked, never updated"
    assert calls[1][0][calls[1][0].index("--out") + 1] == str(run_dir / "lean-audit" / "requirer-pkg_Req")
    assert any("taking the --dev-cache path" in s for s in said)
    assert not (tree / ".lock").exists(), "the slot tree is never touched"
    assert (run_dir / "lean-audit" / "record" / "lean-audit.json").read_text() == NEW, "the same keep as the slot path"
    result = json.loads((run_dir / "result.json").read_text())
    assert result["kind"] == LC.FAST_KIND and result["workload_fingerprint"]["path"] == "dev-cache"


def test_a_build_slot_takes_the_slot_path(fast, tmp_path, monkeypatch):
    _, tree, calls = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 0
    assert [a[-1] for a, _ in calls] == [str(tree / "pkg"), str(tree / "pkg/Req")]
    assert not any("--dev-cache" in a for a, _ in calls)
    assert any("taking the slot path" in s for s in said)


def test_dev_cache_takes_that_path_even_where_a_build_slot_is_granted(fast, tmp_path, monkeypatch):
    root, tree, calls = fast
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: pytest.fail("--dev-cache takes no build slot"))
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append, dev_cache=True) == 0
    assert [a[-1] for a, _ in calls] == [str(root / "pkg"), str(root / "pkg/Req")]
    assert all("--dev-cache" in a for a, _ in calls) and any("asks for it; taking the --dev-cache path" in s for s in said)
    for argv in (["--dev-cache"], ["--records", "pkg", "--dev-cache", "--tree", str(tree)]):
        with pytest.raises(SystemExit):
            LC.main(argv)


def test_the_dev_cache_path_never_builds_in_a_shipped_tree(fast, tmp_path, monkeypatch):
    root, _, calls = fast
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", refuse)
    (root / "READY.json").write_text("{}\n")
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 2
    assert not calls and any("--cwd clone" in s for s in said)
