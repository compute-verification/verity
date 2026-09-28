"""`tools/verity/lean/lean_changed.py --records`: a lane's records audited without the kernel replay in a Lean slot of pool `build`, on
that slot's tree, with its own copy of the audit's dependencies; its result is of kind `lean-fast/v1`, which check and research
merge refuse."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_lean_changed_records", ROOT / "tools" / "verity" / "lean" / "lean_changed.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)
spec = importlib.util.spec_from_file_location("check_for_lean_records", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

#: `audit.py` as far as `--records` sees it: its arguments, working directory, threads, Lean slot and the lock files it holds
#: open logged; reports under `--out` as `--no-replay` leaves them (no replay time); `--update` writing a record
FAKE_AUDIT = """\
import json, os, sys
held = []
for fd in os.listdir("/proc/self/fd"):
    try:
        held.append(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError:
        pass
with open(os.environ["AUDIT_LOG"], "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "threads": os.environ.get("LEAN_NUM_THREADS"),
                        "locks": sorted(p for p in held if p.endswith(".lock")),
                        **({"lean_slot": os.environ["LEAN_SLOT"]} if "LEAN_SLOT" in os.environ else {})}) + "\\n")
rc = int(os.environ.get("AUDIT_RC", "0"))
out, pkg = sys.argv[sys.argv.index("--out") + 1], sys.argv[-1]
rep = {"package": pkg, "seconds": {"build": 1.0, "audit": 2.0}, "failures": ["audit: no"] if rc else []}
os.makedirs(os.path.join(out, "audit-x"), exist_ok=True)
with open(os.path.join(out, "audit-x", "report.json"), "w") as f:
    json.dump(rep, f)
with open(os.path.join(out, "audit.json"), "w") as f:
    json.dump({"passed": not rc, "failed": [pkg] if rc else [], "packages": [rep]}, f)
if "--update" in sys.argv:
    with open(os.path.join(pkg, "lean-audit.json"), "w") as f:
        f.write('{"guarantees": {"A.t": {}}}')
sys.exit(rc)
"""


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A checkout with package `a` (a git dependency, and `p` by path), `p` and `c`; `audit.py` as `FAKE_AUDIT`; no Lean slots
    file and no run, whatever machine runs the tests."""
    root = tmp_path / "repo"
    files = {
        "a/lakefile.toml": "", "a/lean-toolchain": "v1\n", "a/lean-audit.json": "{}", "a/Aa.lean": "def a := 1\n",
        "a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"},
                                                         {"name": "p", "type": "path", "dir": "../p"}]}),
        "p/lakefile.toml": "", "p/lean-toolchain": "v1\n", "p/lean-audit.json": "{}", "p/Pp.lean": "def p := 1\n",
        "c/lakefile.toml": "", "c/lean-toolchain": "v1\n", "c/lean-audit.json": "{}", "c/C.lean": "def c := 1\n",
        "c/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "def"}]}),
    }
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    for m in (LC, LC.LA):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LC.LA.COMMON, "cores", lambda: 2)
    (tmp_path / "audit.py").write_text(FAKE_AUDIT)
    monkeypatch.setattr(LC, "AUDIT_PY", tmp_path / "audit.py")
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache"))
    for k in (LC.LEAN_SLOT.HELD, "LEAN_SLOT_POOL", "RESEARCH_LANE", "RESEARCH_RUN_DIR", "RESEARCH_RUN_ID", "CHECK_SLOT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", tmp_path / "no-lean-slots")
    return root


def _calls(tmp: Path) -> list[dict]:
    log = tmp / "audit.log"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    log.unlink(missing_ok=True)
    return calls


def _argv(call: dict) -> list[str]:
    """A call's arguments with its `--out` directory, a temporary one outside a run, as OUT."""
    a = call["argv"]
    return [*a[:a.index("--out") + 1], "OUT", *a[a.index("--out") + 2:]]


def test_the_audit_runs_without_the_replay_on_the_trees_own_copy_of_the_audits_dependencies(repo, tmp_path, monkeypatch, capsys):
    tree, deps = tmp_path / "cache" / "lean-records" / "tree", tmp_path / "deps"
    k = LC.LA.deps_key(repo / "a")
    (deps / k / "dep").mkdir(parents=True)
    (deps / k / "dep" / "Dep.olean").write_text("olean")
    assert LC.audit_records("a", deps, tree) == 0
    (call,) = _calls(tmp_path)
    assert _argv(call) == ["--build", "--no-replay", "--no-runs", "--out", "OUT", str(tree / "a")]
    assert (call["cwd"], call["threads"]) == (str(tree), "2")
    assert (tree / "a" / "Aa.lean").is_file() and (tree / "p" / "Pp.lean").is_file() and not (tree / "c").exists()
    copied, kept = tree / "a" / ".lake" / "packages" / "dep" / "Dep.olean", deps / k / "dep" / "Dep.olean"
    assert copied.read_text() == "olean" and copied.stat().st_ino != kept.stat().st_ino, "a copy, never a link to the audit's"
    assert json.loads((tree / LC.LINKS).read_text()) == {"a": k}
    out = capsys.readouterr().out
    assert f"dependencies copied ({k[:12]})" in out and "lean-records: a: audit.py exit 0" in out

    assert LC.audit_records("a", deps, tree, update=True) == 0
    assert _argv(_calls(tmp_path)[0]) == ["--build", "--no-replay", "--no-runs", "--update", "--out", "OUT", str(tree / "a")]
    assert (repo / "a" / "lean-audit.json").read_text() == '{"guarantees": {"A.t": {}}}', "the record comes back to the checkout"
    out = capsys.readouterr().out
    assert f"lean-records: wrote {repo / 'a' / 'lean-audit.json'}" in out
    assert "dependencies copied" not in out, "the tree keeps its copy while the manifest and toolchain stay"

    (repo / "a" / "lean-audit.json").write_text("{}")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    moved = tmp_path / "moved.json"
    assert LC.audit_records("a", deps, tree, update=True, moved=moved, out=tmp_path / "reports") == 0
    assert _calls(tmp_path)[0]["argv"][3:] == ["--update", "--moved", str(moved), "--out", str(tmp_path / "reports"),
                                               str(tree / "a")]
    assert (repo / "a" / "lean-audit.json").read_text() == "{}", "a recorded run's source tree is shared by its commit's runs"

    monkeypatch.setenv("AUDIT_RC", "1")
    assert LC.audit_records("a", deps, tree) == 1, "the audit's verdict on the records is the exit status"


def test_its_result_is_of_its_own_kind_which_check_and_research_merge_refuse(repo, tmp_path, monkeypatch):
    """In a recorded run the reports go to the run's `lean-audit/` and the result is of kind `lean-fast/v1`, written into the
    reports and into `result.json`, the Attempt's result: check's lean-audit step fails on exactly this `audit.json`, and
    research merge refuses that result kind."""
    from research import result
    from research.merge import REFUSED_KINDS

    run, tree = tmp_path / "run", tmp_path / "cache" / "lean-records" / "tree"
    run.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    assert LC.audit_records("p", tmp_path / "deps", tree) == 0
    assert _calls(tmp_path)[0]["argv"][3:5] == ["--out", str(run / "lean-audit")]
    doc = json.loads((run / "lean-audit" / "audit.json").read_text())
    assert doc["kind"] == LC.FAST_KIND == "lean-fast/v1" and [r["kind"] for r in doc["packages"]] == [LC.FAST_KIND]
    assert json.loads((run / "lean-audit" / "audit-x" / "report.json").read_text())["kind"] == LC.FAST_KIND
    res = result.check(run, "r1")
    assert res.valid and res.doc["kind"] == LC.FAST_KIND and res.validation_status == "passed"
    assert res.doc["artifacts"] == ["lean-audit"] and res.doc["workload_fingerprint"]["replay"] is False
    assert LC.FAST_KIND in REFUSED_KINDS
    assert CHECK.lean_fast(run).startswith(f"refused: {tree / 'p'}: a lean-fast/v1 result")

    monkeypatch.setenv("AUDIT_RC", "1")
    assert LC.audit_records("p", tmp_path / "deps", tree) == 1
    res = result.check(run, "r1")
    assert res.valid and res.doc["kind"] == LC.FAST_KIND and res.validation_status == "failed", "a failure has the kind too"


def test_the_copy_is_taken_from_hard_links_after_the_dependencies_lock_is_released(repo, tmp_path, monkeypatch):
    """A copy of some 8 GB under `lean-deps/<k>.lock` held check's lean-audit, which moves the dependencies in under that lock,
    for as long as it took (red team, #966). Under the lock only hard links are made; the copy is taken from them after, and
    they are gone once it's done, or at the next run in that tree when one died mid-copy."""
    tree, deps = tmp_path / "cache" / "lean-records" / "tree", tmp_path / "deps"
    k = LC.LA.deps_key(repo / "a")
    kept = deps / k / "dep" / "Dep.olean"
    kept.parent.mkdir(parents=True)
    kept.write_text("olean")
    snap = tree / "a" / ".lake" / "packages.snap"
    (snap / "left-by-a-killed-run").mkdir(parents=True)
    real, locked = shutil.copy2, []

    def copy2(src, dst, **kw):
        with open(deps / f"{k}.lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked.append(False)
            except BlockingIOError:
                locked.append(True)
        return real(src, dst, **kw)

    monkeypatch.setattr(LC.shutil, "copy2", copy2)
    assert LC.audit_records("a", deps, tree) == 0
    copied = tree / "a" / ".lake" / "packages" / "dep" / "Dep.olean"
    assert locked == [False] and copied.read_text() == "olean" and copied.stat().st_ino != kept.stat().st_ino
    assert not snap.exists() and kept.stat().st_nlink == 1

    def no_link(*_a, **_kw):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(LC.os, "link", no_link)                 # the dependencies on another filesystem: copied under the lock
    locked.clear()
    other = tree.with_name("other")
    assert LC.audit_records("a", deps, other) == 0
    assert locked == [True] and (other / "a" / ".lake" / "packages" / "dep" / "Dep.olean").read_text() == "olean"


def test_without_the_audits_dependencies_the_tree_fetches_its_own_once(repo, tmp_path, capsys):
    tree, deps = tmp_path / "cache" / "lean-records" / "tree", tmp_path / "deps"
    k = LC.LA.deps_key(repo / "a")
    assert LC.audit_records("a", deps, tree) == 0
    assert json.loads((tree / LC.LINKS).read_text()) == {"a": k}
    assert "lean-records: a: its dependencies are fetched cold, once for this tree" in capsys.readouterr().out
    (tree / "a" / ".lake" / "packages" / "dep").mkdir(parents=True)          # what the audit's `setup.sh` fetched there
    (deps / k / "dep").mkdir(parents=True)
    assert LC.audit_records("a", deps, tree) == 0
    out = capsys.readouterr().out
    assert "fetched cold" not in out and "copied" not in out, "the tree keeps what it fetched"
    assert LC.audit_records("p", deps, tree) == 0 and json.loads((tree / LC.LINKS).read_text()) == {"a": k}, \
        "a package that fetches nothing needs none"


def test_a_tree_keeps_one_packages_dependencies(repo, tmp_path):
    tree, deps = tmp_path / "cache" / "lean-records" / "tree", tmp_path / "deps"
    ka, kc = LC.LA.deps_key(repo / "a"), LC.LA.deps_key(repo / "c")
    assert ka != kc
    for k in (ka, kc):
        (deps / k / "dep").mkdir(parents=True)
    assert LC.audit_records("a", deps, tree) == 0 and (tree / "a" / ".lake" / "packages" / "dep").is_dir()
    assert LC.audit_records("c", deps, tree) == 0
    assert json.loads((tree / LC.LINKS).read_text()) == {"c": kc}
    assert not (tree / "a" / ".lake" / "packages").exists(), "each copy costs inodes the node is short of"
    assert (tree / "c" / ".lake" / "packages" / "dep").is_dir() and (deps / ka / "dep").is_dir()


def test_it_holds_a_lean_slot_of_pool_build_and_works_on_that_slots_tree(repo, tmp_path, monkeypatch, capsys):
    """Never the `audit` pool, where hour-long audits hold every slot, nor the one LEAN_SLOT_POOL names, nor the one LEAN_SLOT
    says its caller holds; audit.py holds the slot's lock itself. A node whose SLOTS lists no `build` refuses it, even with an
    `any` pool, and nothing runs."""
    deps, slots = tmp_path / "deps", tmp_path / "locks" / "lean-slots"
    slots.parent.mkdir()
    slots.write_text("check 1\naudit 1\nbuild 2\n")
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", slots)
    monkeypatch.setenv("LEAN_SLOT_POOL", "audit")
    held = os.open(slots.parent / "lean-build-0.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(held, fcntl.LOCK_EX)
    assert LC.audit_records("p", deps) == 0
    os.close(held)
    (call,) = _calls(tmp_path)
    assert call["lean_slot"] == "build/1" and LC.LEAN_SLOT.HELD not in os.environ
    assert call["locks"] == [str(slots.parent / "lean-build-1.lock")], "the slot is held until audit.py exits, whoever dies first"
    tree = tmp_path / "cache" / "lean-records" / "build-1"
    assert (tree / "p" / "Pp.lean").is_file() and not (tmp_path / "cache" / "lean-records" / "build-0").exists()
    fd = os.open(slots.parent / "lean-build-1.lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)

    assert LC.audit_records("p", deps) == 0 and _calls(tmp_path)[0]["lean_slot"] == "build/0"
    assert (tmp_path / "cache" / "lean-records" / "build-0" / "p" / "Pp.lean").is_file()

    monkeypatch.setenv(LC.LEAN_SLOT.HELD, "check/0")                  # run from inside check's lean-audit step, say
    assert LC.audit_records("p", deps) == 0
    (call,) = _calls(tmp_path)
    assert call["lean_slot"] == "build/0" and call["argv"][-1] == str(tmp_path / "cache" / "lean-records" / "build-0" / "p")

    capsys.readouterr()
    for text in ("check 1\naudit 1\n", "check 1\nany 4\n"):
        slots.write_text(text)
        assert LC.audit_records("p", deps) == 2 and _calls(tmp_path) == []
        assert "lists no pool 'build', the only one this request takes" in capsys.readouterr().out, "a guard never fails open"


def test_without_lean_slots_its_tree_is_local(repo, tmp_path):
    assert LC.records_tree() == tmp_path / "cache" / "lean-records" / "local"
    assert LC.audit_records("p", tmp_path / "deps") == 0
    assert _calls(tmp_path)[0]["argv"][-1] == str(tmp_path / "cache" / "lean-records" / "local" / "p")


def test_it_names_one_package_and_one_run_holds_a_tree(repo, tmp_path, capsys):
    tree, deps = tmp_path / "cache" / "lean-records" / "tree", tmp_path / "deps"
    assert LC.audit_records("q", deps, tree) == 2
    assert "lean-records: 'q' matches 0 packages: a, c, p" in capsys.readouterr().out
    assert LC.audit_records("p/", deps, tree) == 0 and _calls(tmp_path)[0]["argv"][-1] == str(tree / "p")
    with open(tree / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert LC.audit_records("a", deps, tree) == 2
    assert f"lean-records: another run holds {tree}" in capsys.readouterr().out and _calls(tmp_path) == []


def test_it_refuses_a_tree_outside_its_own(repo, tmp_path, capsys):
    """`--tree` naming check's lean-changed tree, whose links.json says its dependencies are linked: the fast path kept those
    hard links to check's lean-deps and built on them unsandboxed (red team, #966). Only a tree under lean-records/ is its own."""
    deps, cache = tmp_path / "deps", tmp_path / "cache"
    k = LC.LA.deps_key(repo / "a")
    (deps / k / "dep").mkdir(parents=True)
    (deps / k / "dep" / "Dep.olean").write_text("olean")
    checks = cache / "lean-changed"
    LC.sync(LC.closure("a"), LC.LA.COMMON.tracked_files(), checks)
    links = {}
    assert LC.link("a", checks, deps, links, print)
    (checks / LC.LINKS).write_text(json.dumps(links))
    (cache / "lean-records").mkdir()
    (cache / "lean-records" / "elsewhere").symlink_to(checks)
    for tree in (checks, cache / "lean-records", cache / "lean-records" / "elsewhere", tmp_path / "tree"):
        assert LC.audit_records("a", deps, tree) == 2
        assert f"lean-records: {tree} is not under {(cache / 'lean-records').resolve()}" in capsys.readouterr().out
    assert _calls(tmp_path) == [] and LC.main(["--records", "a", "--tree", str(checks)]) == 2
    assert LC.audit_records("a", deps, cache / "lean-records" / "mine") == 0


def test_it_takes_no_check_slot(repo, tmp_path, monkeypatch):
    """The fast path waits only for its `build` slot, never in the check slots' line behind merge checks."""
    def execv(*args):
        raise AssertionError(f"re-executed: {args}")

    seen = []
    monkeypatch.setattr(LC.os, "execv", execv)
    monkeypatch.setattr(LC, "audit_records", lambda name, deps, tree, *a, **kw: seen.append((name, tree, a)) or 0)
    (tmp_path / "slots").write_text("c 8-31\n")
    monkeypatch.setenv("CHECK_SLOTS", str(tmp_path / "slots"))
    assert LC.main(["--records", "a", "--update"]) == 0
    assert seen == [("a", None, (True, None, None))]
    with pytest.raises(SystemExit):
        LC.main(["--update"])
    assert not hasattr(LC, "SLOT") and '"slot.py"' not in (ROOT / "tools" / "verity" / "lean" / "lean_changed.py").read_text()
