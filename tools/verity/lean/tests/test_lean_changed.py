"""`lean_changed.py --records --update`: the regenerated record survives where the run keeps it (LEAN-10), and a node
whose Lean slots grant no `build` slot takes the `--dev-cache` path.

In a recorded run the fast path never writes ROOT (the commit's shared source tree), and the slot tree's copy of the
updated `lean-audit.json` is restored to the committed bytes by the next requirer iteration's `sync`, so without an
explicit keep the record a lane must commit survives nowhere the run keeps. These tests stub the slot, the tree sync
and check.py itself, and check the keep under `out/record/` (recorded and not), the ROOT write-back (unrecorded
only), which path a refused `build` slot takes, and how a run finds and repairs a tree another run left mid-way."""
from __future__ import annotations

import importlib.util
import json
import os
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
    monkeypatch.setattr(LC, "reused", lambda *a, **k: {})
    monkeypatch.setattr(LC, "passed", lambda *a, **k: None)

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


def module(lib: Path, name: str, olean: bytes = b"olean\x02rest", trace: str | None = None, hashed: str = "aa") -> Path:
    """A module's outputs under ``lib`` as Lake leaves them: `.olean`, `.ilean`, their `.hash` files and the trace."""
    base = lib / name
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".olean").write_bytes(olean)
    base.with_suffix(".ilean").write_text("{}")
    base.with_name(f"{base.name}.olean.hash").write_text(hashed)
    base.with_name(f"{base.name}.ilean.hash").write_text("bb")
    t = base.with_suffix(".trace")
    t.write_text(trace if trace is not None else json.dumps({"outputs": {"o": ["aa.olean"], "i": "bb.ilean", "c": "cc.c"}}))
    return t


def test_damaged_finds_each_module_an_interrupted_build_left_half_written(tmp_path):
    lib = tmp_path / "lib"
    module(lib, "P/Good")
    bad = [module(lib, "P/NoOlean"), module(lib, "P/NotOlean", olean=b"\0\0\0"), module(lib, "P/Stale", hashed="zz"),
           module(lib, "P/Cut", trace='{"outputs": {"o": ["aa.ol')]
    (lib / "P" / "NoOlean.olean").unlink()
    assert LC.damaged(lib) == sorted(bad)
    assert LC.damaged(tmp_path / "none") == []


def torn(lib: Path, name: str) -> Path:
    """A module whose `.olean` was written after its `.hash`, the trace and the `.hash` still agreeing: what a build killed
    after `lean` wrote its outputs leaves, and a `.ltar` restore from Lake's cache over a stale `.hash` (TOP-90 row 54)."""
    t = module(lib, name)
    h = lib / f"{name}.olean.hash"
    os.utime(h, ns=(10**18, 10**18))
    os.utime(lib / f"{name}.olean", ns=(10**18 + 1, 10**18 + 1))
    return t


def test_an_output_newer_than_its_hash_is_half_written_and_loses_every_hash_with_its_trace(tmp_path):
    """Lake reads an import's hash from its `.hash` without hashing the `.olean`, so a module rebuilt against the torn one
    records the old hash for the new bytes; the trace alone going would leave the `.hash` that a `.ltar` restore keeps."""
    tree = tmp_path / "tree"
    lib = tree / "pkg" / ".lake" / "build" / "lib" / "lean"
    good = module(lib, "P/Good")
    os.utime(lib / "P" / "Good.olean", ns=(10**18, 10**18))
    os.utime(lib / "P" / "Good.olean.hash", ns=(10**18, 10**18))
    bad = torn(lib, "P/Torn")
    ir = tree / "pkg" / ".lake" / "build" / "ir" / "P"
    ir.mkdir(parents=True)
    (ir / "Torn.c").write_text("c")
    (ir / "Torn.c.hash").write_text("cc")
    (ir / "Good.c.hash").write_text("cc")
    assert LC.damaged(lib) == [bad], "an output written after its `.hash`; one written at the same time is not"
    said = []
    LC.repair(tree, ["pkg"], {}, None, said.append)
    assert not bad.exists() and not list(lib.glob("P/Torn.*.hash")) and not (ir / "Torn.c.hash").exists()
    assert (lib / "P" / "Torn.olean").exists() and (ir / "Torn.c").exists(), "the outputs stay for Lake to replace"
    assert good.exists() and (lib / "P" / "Good.olean.hash").exists() and (ir / "Good.c.hash").exists()
    assert any("1 module(s) half-written by an earlier build, rebuilt: P/Torn" in s for s in said)


def test_a_module_without_a_trace_loses_the_hash_files_it_kept(tmp_path):
    """compute-accounting, 26 Oct 09:49Z: cpu-5's `build-0` held `W1.olean` rewritten at 09:08, its `.hash` from 07:47 and
    no `W1.trace`, as a `repair` that dropped only the trace left it; `damaged` reads only traces, so it went unflagged,
    and the next `.ltar` restore would have kept the stale `.hash` beside a new trace."""
    tree = tmp_path / "tree"
    lib = tree / "pkg" / ".lake" / "build" / "lib" / "lean"
    good = module(lib, "P/Good")
    module(lib, "P/W1").unlink()
    ir = tree / "pkg" / ".lake" / "build" / "ir" / "P"
    ir.mkdir(parents=True)
    for name in ("W1", "Good"):
        (ir / f"{name}.c").write_text("c")
        (ir / f"{name}.c.hash").write_text("cc")
    assert LC.damaged(lib) == [], "no trace to read"
    said = []
    LC.repair(tree, ["pkg"], {}, None, said.append)
    assert not list(lib.glob("P/W1.*.hash")) and not (ir / "W1.c.hash").exists()
    assert (lib / "P" / "W1.olean").exists() and (ir / "W1.c").exists(), "the outputs stay for Lake to replace"
    assert good.exists() and (lib / "P" / "Good.olean.hash").exists() and (ir / "Good.c.hash").exists()
    assert any("1 module(s) without a trace kept their .hash files, dropped: P/W1" in s for s in said)


def test_a_tree_left_mid_way_is_repaired_and_said_so_and_a_clean_run_clears_the_mark(fast, tmp_path, monkeypatch):
    _, tree, _ = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    (tree / LC.DIRTY).write_text(json.dumps({"run": "r-old", "pid": 7, "package": "pkg", "phase": "deps", "at": "T"}))
    lib = tree / "pkg" / ".lake" / "build" / "lib" / "lean"
    module(lib, "P/Good")
    half = module(lib, "P/Half")
    (lib / "P" / "Half.olean").unlink()
    (tree / "pkg" / ".lake" / "packages" / "dep").mkdir(parents=True)
    dropped, said = [], []
    monkeypatch.setattr(LC, "unlink", lambda p, t, links: dropped.append(p))
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 0
    assert any("left mid-way by r-old (pid 7, pkg in phase deps, T)" in s for s in said)
    assert any("1 module(s) half-written by an earlier build, rebuilt: P/Half" in s for s in said)
    assert not half.exists() and (lib / "P" / "Good.trace").exists(), "only the half-written module loses its trace"
    assert dropped == ["pkg"], "the dependencies the run was taking are taken again"
    assert not (tree / LC.DIRTY).exists(), "a run that ends without a killed check.py clears the mark"


def test_a_killed_check_leaves_the_mark_for_the_next_run(fast, tmp_path, monkeypatch):
    _, tree, calls = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-killed")

    def run(argv, **_):
        return type("P", (), {"returncode": -9 if argv[-1].endswith("Req") else 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == -9
    left = json.loads((tree / LC.DIRTY).read_text())
    assert (left["run"], left["package"], left["phase"], left["killed"]) == ("r-killed", "pkg/Req", "check", -9)
    monkeypatch.setattr(LC.subprocess, "run", lambda argv, **_: type("P", (), {"returncode": 0})())
    said.clear()
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 0
    assert any("left mid-way by r-killed" in s and "check.py exit -9" in s for s in said)
    assert not (tree / LC.DIRTY).exists()


def test_after_a_run_left_mid_way_dependencies_off_their_pinned_checkout_are_taken_again(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    (tree / "pkg" / ".lake" / "packages" / "dep" / ".git").mkdir(parents=True)
    (tree / "pkg" / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "git", "name": "dep", "rev": "abc"}]}))
    monkeypatch.setattr(LC, "git_head", lambda d: "def")
    links, said = {"pkg": "k"}, []
    LC.repair(tree, ["pkg"], links, None, said.append)
    assert links == {"pkg": "k"} and not said, "a tree no run left mid-way keeps its dependencies"
    LC.repair(tree, ["pkg"], links, {"run": "r-old", "package": "other", "phase": "check"}, said.append)
    assert links == {} and not (tree / "pkg" / ".lake" / "packages").exists()
    assert any("dep is not the checkout its manifest pins (abc)" in s for s in said)


@pytest.fixture()
def trying(tmp_path, monkeypatch):
    """`try_file` against nested stand-in packages `Sec` and `Sec/Proofs` (whose modules sit under `srcDir = ".."`, as
    Security/Proofs' do), a slot tree, no sandbox, and `build` and check.py recorded instead of run."""
    root = tmp_path / "root"
    (root / "Sec" / "Proofs" / "Flock").mkdir(parents=True)
    (root / "Sec" / "Proofs" / "lakefile.toml").write_text('name = "p"\nsrcDir = ".."\n')
    (root / "Sec" / "Proofs" / "Flock" / "X.lean").write_text("import Mathlib\n")
    lake = tmp_path / "lake"
    lake.write_text("")
    cache = tmp_path / "cache"
    tree = cache / LC.RECORDS / "local"
    monkeypatch.setattr(LC, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "cache_dir", lambda: cache)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: ["Sec/Proofs/Flock/X.lean"])
    monkeypatch.setattr(LC.LA, "packages", lambda _: ["Sec", "Sec/Proofs"])
    monkeypatch.setattr(LC.LA, "deps_key", lambda _: "k")
    monkeypatch.setattr(LC, "closure", lambda p: [p, "Sec"])
    monkeypatch.setattr(LC, "records_tree", lambda: tree)
    monkeypatch.setattr(LC, "sync", lambda *_: 1)
    monkeypatch.setattr(LC, "sandboxed", lambda: False)
    monkeypatch.setattr(LC.shutil, "which", lambda _: str(lake))
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: None)
    calls = {"build": [], "run": [], "link": [], "lean_rc": 0, "deps_rc": 0, "lean_text": ""}

    def link(pkg, t, deps, links, log=print, copy=False):
        calls["link"].append((pkg, copy))
        links[pkg] = "k"
        return True

    def build(pkg, t, lake_, targets=None, sandbox=True, cmd=None):
        calls["build"].append((pkg, t, targets, sandbox, cmd))
        return (calls["lean_rc"], calls["lean_text"]) if cmd else (calls["deps_rc"], "")

    monkeypatch.setattr(LC, "link", link)
    monkeypatch.setattr(LC, "build", build)
    monkeypatch.setattr(LC.subprocess, "run", lambda argv, **kw: calls["run"].append(argv))
    return tree, calls


def test_try_builds_the_files_imports_then_runs_lean_on_it_in_its_owning_package_and_never_check(trying, tmp_path):
    tree, calls = trying
    calls["lean_rc"] = 1
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 1, "lean's exit status"
    assert calls["build"] == [("Sec/Proofs", tree, ["+Proofs.Flock.X:deps"], False, None),
                              ("Sec/Proofs", tree, None, False, ["env", "lean", "Flock/X.lean"])]
    assert calls["link"] == [("Sec/Proofs", True)], "the owning package's dependencies, a copy as --records takes"
    assert calls["run"] == [], "no check.py, audit or --update"
    assert not (tree / LC.DIRTY).exists() and any("lean exit 1" in s for s in said)
    assert LC.main(["--try", "Sec/Proofs/Flock/X.lean", "--tree", str(tree)]) == 1
    with pytest.raises(SystemExit):
        LC.main(["--try", "Sec/Proofs/Flock/X.lean", "--records", "Sec"])


def test_try_forgets_both_modules_an_import_clash_names_so_the_next_try_rebuilds_them(trying, tmp_path):
    """cpu-5's `build-0`, 26 Oct: `Proofs.Pouw` failed there with `W1BudgetAbort` replayed beside a fresh
    `TTOutW1RowSeed`, each built against another `W1`, while it elaborated from source everywhere else."""
    tree, calls = trying
    proofs = tree / "Sec" / "Proofs" / ".lake" / "build" / "lib" / "lean"
    sec = tree / "Sec" / ".lake" / "build" / "lib" / "lean"
    stale, fresh, other = module(proofs, "Proofs/Pouw/W1BudgetAbort"), module(sec, "Defs/RowSeed"), module(proofs, "P/Other")
    calls["lean_rc"] = 1
    calls["lean_text"] = ("Flock/X.lean:1:0: error: import Proofs.Pouw.W1BudgetAbort failed, environment already contains "
                          "'Pouw.PearlC.W1.Instr.exec.eq_10' from Defs.RowSeed\n")
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 1, "lean's status, unchanged"
    assert not stale.exists() and not fresh.exists() and other.exists()
    assert not list(proofs.glob("Proofs/Pouw/W1BudgetAbort.*.hash")) and not list(sec.glob("Defs/RowSeed.*.hash"))
    assert any("import clash between Proofs.Pouw.W1BudgetAbort and Defs.RowSeed" in s for s in said)
    calls["lean_text"] = "Flock/X.lean:3:0: error: unknown identifier 'x'\n"
    said.clear()
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 1
    assert other.exists() and not any("clash" in s for s in said), "any other error forgets nothing"


def test_try_runs_no_lean_when_the_imports_fail_to_build(trying, tmp_path):
    _, calls = trying
    calls["deps_rc"] = 3
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=lambda _: None) == 3
    assert [c[4] for c in calls["build"]] == [None], "lean never runs on a file whose imports don't build"


def test_try_keeps_dependencies_no_records_run_took_and_leaves_them_out_of_links(trying, tmp_path):
    tree, calls = trying
    (tree / "Sec" / "Proofs" / ".lake" / "packages" / "mathlib").mkdir(parents=True)
    (tree / LC.LINKS).write_text(json.dumps({"Sec": "k"}))
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=lambda _: None) == 0
    assert calls["link"] == [] and LC.read_links(tree) == {"Sec": "k"}, "a later --records run never drops them"


def test_try_refuses_without_a_build_slot_naming_the_nodes_that_have_them(trying, tmp_path, monkeypatch):
    _, calls = trying
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", refuse)
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 2
    assert calls["build"] == [] and any("vy-nebius-cpu-1" in s and "nothing ran" in s for s in said)


def test_try_refuses_a_file_in_no_package_and_a_sandbox_without_dependencies(trying, tmp_path, monkeypatch):
    tree, calls = trying
    said = []
    assert LC.try_file("Elsewhere/X.lean", tmp_path / "deps", log=said.append) == 2
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", tree=tmp_path / "other", log=said.append) == 2
    monkeypatch.setattr(LC, "sandboxed", lambda: True)
    monkeypatch.setattr(LC, "link", lambda *a, **k: False)
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 2
    assert calls["build"] == [] and any("--records Sec/Proofs" in s for s in said)


def test_build_builds_the_files_modules_on_the_slot_tree_and_reports_each_files_first_error_as_json(trying, tmp_path,
                                                                                                     monkeypatch, capsys):
    """The ready mark's preflight reads `--build`'s last line: a Lean error at a position in a tracked file fails that file,
    named from the repository's root, whichever package of the build's closure holds it."""
    tree, calls = trying
    (slots := tmp_path / "lean-slots").write_text("build 1\n")
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", slots)
    (LC.ROOT / "Sec" / "Proofs" / "Flock" / "Y.lean").write_text("import Proofs.Flock.X\n")
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: ["Sec/Proofs/Flock/X.lean", "Sec/Proofs/Flock/Y.lean",
                                                                   "Sec/Defs/A.lean"])
    calls["deps_rc"] = 1
    monkeypatch.setattr(LC, "build", lambda pkg, t, lake_, targets=None, sandbox=True, cmd=None: (
        calls["build"].append((pkg, targets)) or 1,
        "✖ [3/9] Building Proofs.Flock.Y\nerror: ././../Proofs/Flock/Y.lean:2:4: unknown identifier 'foo'\n  more\n"
        "error: ./../Defs/A.lean:1:0: unexpected token\nerror: ././../Proofs/Flock/Y.lean:5:0: a later one\n"
        "error: Lean exited with code 1\n"))
    assert LC.main(["--build", "Sec/Proofs/Flock/X.lean", "Sec/Proofs/Flock/Y.lean", "Elsewhere/Z.lean"]) == 1
    assert calls["build"] == [("Sec/Proofs", ["+Proofs.Flock.X", "+Proofs.Flock.Y"])] and calls["run"] == []
    doc = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(LC.BUILD_LINE))
    assert doc["built"] == {"Sec/Proofs": 1} and doc["failed"] == {
        "Sec/Proofs/Flock/Y.lean": "error: ././../Proofs/Flock/Y.lean:2:4: unknown identifier 'foo'",
        "Sec/Defs/A.lean": "error: ./../Defs/A.lean:1:0: unexpected token"}
    assert list(doc["skipped"]) == ["Elsewhere/Z.lean"] and not (tree / LC.DIRTY).exists()


def test_build_builds_nothing_off_a_node_or_where_the_node_keeps_no_dependencies(trying, tmp_path, monkeypatch, capsys):
    """An agent VM has no Lean slots, and a node without the package's dependencies would fetch Mathlib cold: both are
    skipped with why, so the preflight notes them instead of refusing."""
    _, calls = trying
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", tmp_path / "absent")
    assert LC.main(["--build", "Sec/Proofs/Flock/X.lean"]) == 2
    doc = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(LC.BUILD_LINE))
    assert "not a node" in doc["skipped"]["*"] and calls["build"] == [] and calls["link"] == []
    (slots := tmp_path / "lean-slots").write_text("build 1\n")
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", slots)
    monkeypatch.setattr(LC, "link", lambda *a, **k: False)
    assert LC.main(["--build", "Sec/Proofs/Flock/X.lean"]) == 2
    doc = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(LC.BUILD_LINE))
    assert "Mathlib" in doc["skipped"]["Sec/Proofs"] and doc["built"] == {} and calls["build"] == []
    with pytest.raises(SystemExit):
        LC.main(["--build", "Sec/Proofs/Flock/X.lean", "--try", "Sec/Proofs/Flock/X.lean"])


def test_module_of_reads_the_packages_src_dir(trying):
    root = LC.ROOT
    assert LC.module_of("Sec/Proofs", "Sec/Proofs/Flock/X.lean") == "Proofs.Flock.X"
    (root / "Sec" / "lakefile.lean").write_text('package sec where\n  srcDir := "src"\n')
    assert LC.module_of("Sec", "Sec/src/A/B.lean") == "A.B"
    (root / "Plain").mkdir()
    assert LC.module_of("Plain", "Plain/A/B.lean") == "A.B"
