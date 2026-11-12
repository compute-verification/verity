"""`research data evict`'s passes over what nothing else removes (`store/leftovers.py`): a uv cache's idle editable shards and
build directories, the store's lock files, request directories whose run dir is gone, and the clones of runs that are over.
Each group shows the removal and everything that must stay: in use by a live process, too young, a lock held, a run that has not
ended, a name or layout the pass does not know, a `research keep` record.  Every path is under `tmp_path`, /proc is a directory there, and time is the
`now` and `clock` each pass is given."""
import fcntl
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import clone, remote, retention
from research.store import evict as EV
from research.store import fetchset
from research.store import leftovers as L

NOW = 2_000_000_000.0
H = 3600.0
RID = "r20330518-033320-0a1b"
RIDS = [f"r20330518-0333{n:02d}-0a1b" for n in range(20)]


def age(path: Path, hours: float) -> None:
    """Set every mtime in the tree at `path` to `hours` before NOW."""
    when = NOW - hours * H
    for p in [*sorted(path.rglob("*"), reverse=True), path]:
        os.utime(p, (when, when), follow_symlinks=False)


def process(proc: Path, pid: int, *, pgid: int | None = None, start: int = 100, state: str = "S", cwd: str = "/",
            cmdline: str = "python3", environ: str = "", fds: tuple[str, ...] = ()) -> None:
    """A process in the fake /proc, with every field of `stat` the readers take (its group, state and start time among them)."""
    d = proc / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "stat").write_text(f"{pid} (p{pid}) {state} 1 {pgid or pid} {pgid or pid} 0 -1 4194560 0 0 0 0 0 0 0 0 20 0 1 0 {start} 1000 10\n")
    (d / "comm").write_text(f"p{pid}\n")
    os.symlink(cwd, d / "cwd")
    os.symlink("/", d / "root")
    os.symlink("/usr/bin/python3", d / "exe")
    for i, f in enumerate(fds):
        os.symlink(f, d / "fd" / str(i))
    (d / "maps").write_text("")
    (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())
    (d / "environ").write_text(environ)


@pytest.fixture
def proc(tmp_path):
    p = tmp_path / "proc"
    process(p, 1)
    return p


def snapshot(top: Path) -> dict[str, tuple[int, float]]:
    """Every path under `top` with its mode and mtime: equal before and after a pass that changed nothing."""
    return {str(p): (os.lstat(p).st_mode, os.lstat(p).st_mtime) for p in [top, *top.rglob("*")]}


def held(path: Path) -> int:
    """A descriptor of another opener that holds the flock on `path`."""
    fd = os.open(path, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def is_free(path: Path) -> bool:
    """Whether another opener can take the flock on `path` now."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


def after_first(prefix: str, t: list[float], to: float):
    """A log that moves the clock `t` to `to` at the pass's first line starting with `prefix`."""
    return lambda s: t.__setitem__(0, to) if s.startswith(prefix) else None


# ---- uv's editable shards and temporary build directories ------------------------------------------------------------------------------

D1, D2, D3 = "0123456789abcdef", "123456789abcdef0", "23456789abcdef01"


@pytest.fixture
def cache(tmp_path):
    """A uv cache: its tag, and an unpacked wheel in the archive that shards and build directories link."""
    c = tmp_path / "cache" / "uv"
    (c / "archive-v0" / "wheel").mkdir(parents=True)
    (c / "archive-v0" / "wheel" / "mod.py").write_text("x = 1\n")
    (c / L.UV_TAG).write_text(L.UV_SIGNATURE + "\n# This file is a cache directory tag created by uv.\n")
    return c


def shard(c: Path, digest: str, hours: float, bucket: str = "sdists-v9", pointer: bool = True) -> Path:
    """An editable shard as uv leaves it: its lock, its revision pointer, and the revision's wheel, metadata and link into the
    archive.  7 inodes, 6 without the pointer."""
    s = c / bucket / "editable" / digest
    rev = s / "rev0"
    rev.mkdir(parents=True)
    (s / ".lock").write_text("")
    if pointer:
        (s / "revision.rev").write_bytes(b"rev0")
    (rev / "pkg-0.1-py3-none-any.whl").write_bytes(b"w" * 1000)
    (rev / "metadata.msgpack").write_bytes(b"m" * 24)
    os.symlink(c / "archive-v0" / "wheel", rev / "pkg-0.1-py3-none-any")
    age(s, hours)
    return s


def build(c: Path, name: str, hours: float) -> Path:
    """A temporary build directory a killed uv left: a venv whose module is a hard link into the archive.  3 inodes of its own."""
    b = c / "builds-v0" / name
    (b / "lib").mkdir(parents=True)
    (b / "pyvenv.cfg").write_text("home = /usr/bin\n")
    os.link(c / "archive-v0" / "wheel" / "mod.py", b / "lib" / "mod.py")
    age(b, hours)
    return b


def uv(c: Path, proc: Path, **kw):
    return L.evict_uv(c, proc=proc, now=kw.pop("now", NOW), clock=kw.pop("clock", lambda: 0.0), **kw)


def test_an_idle_shard_loses_its_revision_pointer_first_and_goes_once_idle_two_days_more(cache, proc):
    """uv reads a cached wheel after it has released the shard's lock, so no pass removes a wheel a uv could still have found:
    the first step unlinks the pointer alone and stamps the shard, the second removes the shard two days on."""
    s = shard(cache, D1, 49)
    rep = uv(cache, proc)
    assert rep["error"] is None and rep["found"] == 1 and rep["deleted"] == 0 and rep["kept_by"] == {"retired": 1}
    assert (rep["deleted_inodes"], rep["deleted_bytes"]) == (1, 4)
    assert sorted(p.name for p in s.iterdir()) == [".lock", "rev0"] and (s / "rev0" / "pkg-0.1-py3-none-any.whl").stat().st_size == 1000
    assert os.lstat(s).st_mtime == NOW
    rep = uv(cache, proc, now=NOW + 47 * H)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"recent": 1} and (s / "rev0").is_dir()
    rep = uv(cache, proc, now=NOW + 49 * H)
    assert rep["error"] is None and rep["deleted"] == 1 and rep["deleted_first"] == [str(s)] and rep["kept"] == 0
    assert rep["deleted_inodes"] == 6 and rep["deleted_bytes"] >= 1024 and not s.exists()
    assert (cache / "archive-v0" / "wheel" / "mod.py").read_text() == "x = 1\n", "the archive is uv's own to prune"


def test_a_shard_uv_built_again_after_its_pointer_went_starts_over(cache, proc):
    s = shard(cache, D1, 49)
    uv(cache, proc)
    (s / "rev1").mkdir()
    (s / "rev1" / "pkg-0.1-py3-none-any.whl").write_bytes(b"w" * 1000)
    (s / "revision.rev").write_bytes(b"rev1")
    age(s, -10)                                                    # uv built it again 10 h after the first step
    assert uv(cache, proc, now=NOW + 49 * H)["kept_by"] == {"recent": 1}
    rep = uv(cache, proc, now=NOW + 59 * H)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"retired": 1}, "a shard with a pointer is never removed outright"
    assert sorted(p.name for p in s.iterdir()) == [".lock", "rev0", "rev1"]


def test_a_shard_written_within_two_days_stays_wherever_the_write_was(cache, proc):
    s = shard(cache, D1, 100)
    os.utime(s / "rev0" / "metadata.msgpack", (NOW - 47 * H, NOW - 47 * H))
    before = snapshot(cache)
    rep = uv(cache, proc)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"recent": 1} and rep["deleted_inodes"] == 0 and snapshot(cache) == before


def test_a_shard_whose_lock_another_process_holds_stays_and_is_named(cache, proc):
    s = shard(cache, D1, 100)
    before = snapshot(cache)
    fd = held(s / ".lock")
    try:
        rep = uv(cache, proc)
    finally:
        os.close(fd)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"2": 1} and snapshot(cache) == before
    assert rep["kept_first"] == [{"path": str(s), "why": "2: its lock is held (.lock)"}]
    assert uv(cache, proc)["kept_by"] == {"retired": 1}, "and goes on once the lock is free"


@pytest.mark.parametrize("how", ["open file", "cwd", "relative in a command line", "environment"])
def test_a_shard_a_live_process_names_stays_and_is_named(cache, proc, how):
    s = shard(cache, D1, 100, pointer=False)
    process(proc, 7, **{"open file": {"fds": (str(s / ".lock"),)}, "cwd": {"cwd": str(s / "rev0")},
                        "relative in a command line": {"cwd": str(cache), "cmdline": f"unzip sdists-v9//editable/{D1}/rev0/x.whl"},
                        "environment": {"environ": f"A=1\0WHEEL={s}/rev0\0"}}[how])
    before = snapshot(cache)
    rep = uv(cache, proc)
    assert rep["deleted"] == 0 and snapshot(cache) == before
    assert rep["kept_first"] == [{"path": str(s), "why": "2: process 7 (p7) names it"}]


def test_what_the_pass_does_not_know_stays_and_is_said(cache, proc, tmp_path):
    """An entry that is neither the pointer nor a revision, a bucket of a uv version whose layout was not read, a shard that is
    a symbolic link: none is guessed at."""
    odd, other = shard(cache, D2, 100), shard(cache, D3, 100, bucket="sdists-v11")
    (odd / "notes.txt").write_text("?")
    outside = shard(tmp_path / "elsewhere", D1, 100)
    os.symlink(outside, cache / "sdists-v9" / "editable" / "fedcba9876543210")
    age(odd, 100)
    before = snapshot(cache), snapshot(outside)
    rep = uv(cache, proc)
    assert rep["error"] is None and rep["found"] == 2 and rep["deleted"] == 0 and rep["kept_by"] == {"unknown": 2}
    assert (snapshot(cache), snapshot(outside)) == before
    assert {k["path"]: k["why"] for k in rep["kept_first"]} == {
        str(odd): "unknown: its notes.txt is neither the revision pointer nor a revision's directory",
        str(other.parent): "unknown: sdists-v11 is not a bucket whose layout this pass knows (sdists-v9, sdists-v10): its 1 shards stay"}


def test_a_shard_without_a_lock_file_gets_one_and_goes_like_any_other_from_then(cache, proc):
    """A uv that waited on the `.lock` a pass unlinked builds the shard again under it, and leaves a shard without one.  Nothing
    of it goes without a lock taken at its path, where the next uv waits: the pass makes one, which counts as a write."""
    s = shard(cache, D1, 100)
    (s / ".lock").unlink()
    age(s, 100)
    before = snapshot(cache)
    rep = uv(cache, proc, dry_run=True)
    assert rep["kept_by"] == {"recent": 1} and rep["deleted_inodes"] == 0 and snapshot(cache) == before
    rep = uv(cache, proc)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"recent": 1} and rep["deleted_inodes"] == 0
    assert sorted(p.name for p in s.iterdir()) == [".lock", "rev0", "revision.rev"] and (s / ".lock").stat().st_size == 0
    assert os.lstat(s).st_mtime == NOW and is_free(s / ".lock")
    assert uv(cache, proc, now=NOW + 47 * H)["kept_by"] == {"recent": 1}
    assert uv(cache, proc, now=NOW + 49 * H)["kept_by"] == {"retired": 1} and not (s / "revision.rev").exists()
    assert uv(cache, proc, now=NOW + 98 * H)["deleted_first"] == [str(s)] and not s.exists()


def test_an_idle_build_directory_goes_and_a_recent_or_named_one_stays(cache, proc):
    gone, recent, used = build(cache, ".tmpAbC123", 49), build(cache, ".tmpAbC124", 47), build(cache, ".tmpAbC125", 100)
    (cache / "builds-v0" / "kept-by-name").mkdir()
    process(proc, 8, cwd=str(used / "lib"))
    rep = uv(cache, proc)
    assert rep["error"] is None and rep["found"] == 3 and rep["deleted_first"] == [str(gone)] and not gone.exists()
    assert rep["deleted_inodes"] == 3, "its own three: the module is a second link of the archive's, which frees nothing"
    assert rep["kept_by"] == {"2": 1, "recent": 1} and recent.is_dir() and (used / "lib" / "mod.py").exists()
    assert rep["kept_first"] == [{"path": str(used), "why": "2: process 8 (p8) names it"}]
    assert (cache / "archive-v0" / "wheel" / "mod.py").read_text() == "x = 1\n" and (cache / "builds-v0" / "kept-by-name").is_dir()


def test_a_uv_dry_run_removes_nothing_and_says_what_would_go(cache, proc):
    retire, remove, b = shard(cache, D1, 100), shard(cache, D2, 100, pointer=False), build(cache, ".tmpAbC123", 100)
    before, lines = snapshot(cache), []
    rep = uv(cache, proc, dry_run=True, log=lines.append)
    assert snapshot(cache) == before
    assert rep["dry_run"] is True and rep["deleted"] == 2 and sorted(rep["deleted_first"]) == sorted([str(remove), str(b)])
    assert rep["kept_by"] == {"retired": 1} and rep["deleted_inodes"] == 6 + 3 + 1
    assert f"uv: would delete {b} (3 inodes, 0.0 MB)" in lines and any(ln.startswith(f"uv: would delete {remove} (6 inodes") for ln in lines)
    assert any(ln.startswith(f"uv: kept {retire}: retired: its revision pointer would be removed") for ln in lines)
    assert lines[-1].startswith("uv: would delete 2 of 3 (0.0 GB, 10 inodes); kept 1 retired:1")


def test_no_uv_removal_starts_after_the_time_limit_and_the_oldest_goes_first(cache, proc):
    young, oldest, mid = build(cache, ".tmpAbC123", 60), build(cache, ".tmpAbC124", 100), build(cache, ".tmpAbC125", 80)
    t = [0.0]
    rep = uv(cache, proc, clock=lambda: t[0], budget_s=240, log=after_first("uv: deleted ", t, 241.0))
    assert rep["deleted_first"] == [str(oldest)] and rep["kept_by"] == {"budget": 2} and young.is_dir() and mid.is_dir()


def test_a_directory_without_a_cache_tag_is_not_taken_for_a_uv_cache(cache, proc, tmp_path):
    s = shard(cache, D1, 100)
    (cache / L.UV_TAG).unlink()
    before = snapshot(cache)
    rep = uv(cache, proc)
    assert rep["error"] == f"{cache} has no CACHEDIR.TAG of a cache directory, so it is not taken for a uv cache"
    assert rep["deleted"] == 0 and snapshot(cache) == before and s.is_dir()
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert uv(fresh, proc)["error"] is None and uv(tmp_path / "absent", proc)["error"] is None, "a node uv has not run on yet"


def test_a_uv_pass_that_cannot_read_the_processes_removes_nothing(cache, tmp_path):
    shard(cache, D1, 100), build(cache, ".tmpAbC123", 100)
    before = snapshot(cache)
    rep = uv(cache, tmp_path / "no-proc")
    assert rep["error"].startswith(f"2: cannot list {tmp_path / 'no-proc'}") and rep["deleted"] == 0
    assert rep["kept_by"] == {"unestablished": 2} and snapshot(cache) == before


# ---- the store's lock files ------------------------------------------------------------------------------------------------------------

def key(n: int) -> str:
    return f"blob-{n:064x}"


def lock_file(store: Path, name: str, hours: float = 2) -> Path:
    (store / "locks").mkdir(parents=True, exist_ok=True)
    p = store / "locks" / f"{name}.lock"
    p.write_text("")
    age(p, hours)
    return p


def locks(store: Path, **kw):
    return L.evict_locks(store, now=kw.pop("now", NOW), clock=kw.pop("clock", lambda: 0.0), **kw)


def test_a_free_lock_file_over_an_hour_old_goes_and_the_rest_stay(tmp_path):
    store = tmp_path / "store"
    gone, recent, busy = lock_file(store, key(1)), lock_file(store, key(2), 0.9), lock_file(store, key(3))
    data, theirs = lock_file(store, key(4)), lock_file(store, "check-d")
    data.write_text("held by 7")
    age(data, 2)
    (store / "locks" / f"{key(5)}.lock").mkdir()
    (store / "locks" / "slots").write_text("a\n")
    fd = held(busy)
    try:
        rep = locks(store)
    finally:
        os.close(fd)
    assert rep["error"] is None and rep["found"] == 6 and rep["deleted_first"] == [str(gone)] and rep["deleted_inodes"] == 1
    assert rep["kept_by"] == {"2": 1, "recent": 1, "unknown": 3}
    assert sorted(p.name for p in (store / "locks").iterdir()) == sorted(
        [recent.name, busy.name, data.name, theirs.name, f"{key(5)}.lock", "slots"])
    why = {k["path"]: k["why"] for k in rep["kept_first"]}
    assert why[str(busy)] == f"2: its lock is held ({busy.name})"
    assert why[str(data)] == "unknown: it is not empty, and the store's lock files hold no data"
    assert why[str(theirs)] == "unknown: it is not a regular file named like one of the store's lock files", \
        "another program's lock file, whose takers do not check what they hold"


def test_a_lock_dry_run_removes_nothing_and_the_time_limit_stops_the_pass(tmp_path):
    store = tmp_path / "store"
    files = [lock_file(store, key(n)) for n in range(3)]
    rep = locks(store, dry_run=True)
    assert rep["deleted"] == 3 and rep["deleted_first"] == [str(f) for f in files] and all(f.exists() for f in files)
    t = [0.0]
    rep = locks(store, clock=lambda: t[0], budget_s=240, log=after_first("locks: deleted ", t, 241.0))
    assert rep["deleted"] == 1 and rep["kept_by"] == {"budget": 2} and [f.exists() for f in files] == [False, True, True]
    assert locks(tmp_path / "no-store")["found"] == 0


def test_a_lock_file_replaced_while_the_pass_took_it_stays(tmp_path, monkeypatch):
    """The pass unlinks only the file it holds the lock of: one that another pass unlinked, and a fetch made anew, between this
    pass's open and its lock is not the file it holds."""
    store = tmp_path / "store"
    path = lock_file(store, key(1))

    def replaced(fd, op):
        path.unlink()
        path.write_text("")
        age(path, 2)
        return fcntl.flock(fd, op)

    monkeypatch.setattr(L, "fcntl", SimpleNamespace(flock=replaced, LOCK_EX=fcntl.LOCK_EX, LOCK_NB=fcntl.LOCK_NB))
    rep = locks(store)
    assert rep["deleted"] == 0 and path.exists()
    assert rep["kept_first"] == [{"path": str(path), "why": f"2: its lock file was replaced while this pass took it ({path.name})"}]


def test_a_fetch_that_waited_on_a_lock_file_unlinked_meanwhile_takes_the_lock_on_the_file_the_path_names(tmp_path, monkeypatch):
    """The race an unlink opens: a taker that waited on the old file would hold its lock beside the holder of a new file of the
    same name.  The pass holds the lock, a fetch opens the file and waits, the pass unlinks the file, another opener makes and
    locks a new one, and the pass lets go: the fetch must not enter its block until it holds the lock on the new file."""
    store = SimpleNamespace(root=tmp_path / "store")
    path = lock_file(store.root, key(1))
    waiting, moved, entered = threading.Event(), threading.Event(), threading.Event()
    calls, seen = [], {}

    def flock(f, op):
        calls.append(os.fstat(f.fileno()).st_ino)
        (waiting if len(calls) == 1 else moved).set()
        return fcntl.flock(f, op)

    def fetch():
        try:
            with fetchset.lock(store, key(1)):
                entered.set()
                moved.set()
                seen.update(ino=os.stat(path).st_ino, locked=not is_free(path))
        except BaseException as e:      # noqa: BLE001  shown by the assertion below
            seen["error"] = e
            moved.set()

    monkeypatch.setattr(fetchset, "fcntl", SimpleNamespace(flock=flock, LOCK_EX=fcntl.LOCK_EX))
    evictor = L._take(path)
    old = os.fstat(evictor).st_ino
    t = threading.Thread(target=fetch)
    t.start()
    waiting.wait()                      # the fetch has the old file open and asks for its lock
    os.unlink(path)
    other = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(other, fcntl.LOCK_EX)
    new = os.fstat(other).st_ino
    os.close(evictor)                   # the fetch now holds the lock of the unlinked file
    moved.wait()                        # it asks again, or (without the check) it is inside its block
    in_block_beside_the_other_holder = entered.is_set()
    os.close(other)
    t.join()
    assert "error" not in seen, seen
    assert not in_block_beside_the_other_holder, "the fetch ran its block on an unlinked lock file while another opener held the new one"
    assert calls == [old, new] and seen == {"ino": new, "locked": True}


def test_a_fetch_takes_a_lock_the_pass_removed_as_one_it_never_had(tmp_path):
    store = SimpleNamespace(root=tmp_path / "store")
    path = lock_file(store.root, key(1))
    assert locks(store.root)["deleted"] == 1 and not path.exists()
    with fetchset.lock(store, key(1)):
        assert not is_free(path)
    assert is_free(path) and fetchset.LOCK_NAME.fullmatch(key(1)) and fetchset.LOCK_NAME.fullmatch("entry-" + "ab" * 32)


# ---- request directories whose run directory is gone -----------------------------------------------------------------------------------

@pytest.fixture
def root(tmp_path):
    r = tmp_path / "research"
    (r / "runs").mkdir(parents=True)
    return r


def request(root: Path, rid: str, hours: float = 2, *, runs_root: Path | None = None, mem_wait_s: float | None = None,
            launched: bool = True) -> Path:
    """A staged request: request.json, an input, its custody key and, once a launch has delivered it, its lock.  6 inodes."""
    d = root / "requests" / rid
    (d / "inputs").mkdir(parents=True)
    (d / "request.json").write_text(json.dumps({"run_id": rid, "runs_root": str(runs_root or root / "runs"),
                                                **({} if mem_wait_s is None else {"mem_wait_s": mem_wait_s})}))
    (d / "inputs" / "job.sh").write_text("true\n")
    (d / "custody.key").write_text("k")
    if launched:
        (d / remote.LAUNCH_LOCK).write_text("")
    age(d, hours)
    return d


def requests(root: Path, proc: Path, **kw):
    return L.evict_requests(root / "requests", proc=proc, now=kw.pop("now", NOW), clock=kw.pop("clock", lambda: 0.0), **kw)


def test_a_request_whose_run_dir_is_gone_goes_and_one_whose_run_dir_is_anywhere_stays(root, proc, tmp_path):
    gone, here, aside, own, lost = (request(root, RIDS[0]), request(root, RIDS[1]), request(root, RIDS[2]),
                                    request(root, RIDS[3], runs_root=tmp_path / "other-runs"),
                                    request(root, RIDS[4], runs_root=tmp_path / "unmounted"))
    (root / "runs" / RIDS[1]).mkdir()
    (root / "runs-aside" / RIDS[2]).mkdir(parents=True)
    (tmp_path / "other-runs" / RIDS[3]).mkdir(parents=True)
    (root / "requests" / "notes").mkdir()
    rep = requests(root, proc)
    assert rep["error"] is None and rep["found"] == 6 and rep["deleted_first"] == [str(gone)] and not gone.exists()
    assert rep["deleted_inodes"] == 6 and rep["kept_by"] == {"run": 3, "unknown": 2}
    assert all((d / "request.json").exists() and (d / "inputs" / "job.sh").exists() for d in (here, aside, own, lost))
    assert {k["path"]: k["why"] for k in rep["kept_first"]} == {
        str(lost): f"unknown: {tmp_path / 'unmounted'}, where its run dir would be, is not a directory",
        str(root / "requests" / "notes"): "unknown: it is not named like a run id"}


def test_a_request_stays_for_as_long_as_its_launch_may_still_deliver_it(root, proc):
    """A launch delivers a staged request three times, each for two minutes, the preflight's limit and the memory wait it was
    given, and claims the run dir as its first act: a request younger than that, or than an hour, may be on its way."""
    assert remote.launch_limit(None) == 3 * (120 + remote.PREFLIGHT_TIMEOUT_S) + 18 == 918
    assert remote.launch_limit(7200) == 3 * (120 + remote.PREFLIGHT_TIMEOUT_S + 7200) + 18 == 22518
    young, old = request(root, RIDS[0], 0.9), request(root, RIDS[1], 1.1)
    waits, waited = request(root, RIDS[2], 6, mem_wait_s=7200), request(root, RIDS[3], 6.5, mem_wait_s=7200)
    touched = request(root, RIDS[4], 5)
    os.utime(touched / remote.LAUNCH_LOCK, (NOW - 600, NOW - 600))      # a delivery took its lock ten minutes ago
    rep = requests(root, proc)
    assert sorted(rep["deleted_first"]) == sorted([str(old), str(waited)]) and rep["kept_by"] == {"recent": 3}
    assert young.is_dir() and waits.is_dir() and touched.is_dir() and not old.exists() and not waited.exists()


def test_a_request_a_delivery_holds_or_a_live_process_names_stays_and_is_named(root, proc):
    locked, named, cmd = request(root, RIDS[0]), request(root, RIDS[1]), request(root, RIDS[2])
    process(proc, 9, environ=f"HOME=/x\0RESEARCH_RUN_ID={RIDS[1]}\0")
    process(proc, 10, cmdline=f"python3 -m research launch-request {cmd}/request.json")
    before = snapshot(root)
    fd = held(locked / remote.LAUNCH_LOCK)
    try:
        rep = requests(root, proc)
    finally:
        os.close(fd)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"2": 3} and snapshot(root) == before
    assert rep["kept_first"] == [{"path": str(locked), "why": "2: its lock is held (.launch.lock)"},
                                 {"path": str(named), "why": "2: process 9 (p9) names its run"},
                                 {"path": str(cmd), "why": "2: process 10 (p10) names its run"}]


def test_a_request_goes_under_its_launch_lock_request_json_first(root, proc, monkeypatch):
    """A delivery that arrives while the pass removes a request waits on the lock and then finds no request to claim a run dir
    for; one that never had a lock file gets one for the removal."""
    d, never = request(root, RIDS[0]), request(root, RIDS[1], launched=False)
    seen = []
    real = EV._remove_tree

    def remove(path):
        seen.append(((path / "request.json").exists(), not is_free(path / remote.LAUNCH_LOCK)))
        return real(path)

    monkeypatch.setattr(EV, "_remove_tree", remove)
    rep = requests(root, proc)
    assert seen == [(False, True), (False, True)] and rep["deleted"] == 2 and not d.exists() and not never.exists()


def test_a_run_dir_claimed_while_the_pass_took_the_lock_keeps_its_request(root, proc, monkeypatch):
    d = request(root, RIDS[0])
    real = L._take

    def take(path, create=False):
        (root / "runs" / RIDS[0]).mkdir()           # a delivery claimed the run dir and let go of the lock just before
        return real(path, create)

    monkeypatch.setattr(L, "_take", take)
    rep = requests(root, proc)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"run": 1} and (d / "request.json").exists() and (d / "inputs" / "job.sh").exists()


def test_a_request_dry_run_removes_and_creates_nothing(root, proc):
    d, never = request(root, RIDS[0]), request(root, RIDS[1], launched=False)
    before, lines = snapshot(root), []
    rep = requests(root, proc, dry_run=True, log=lines.append)
    assert snapshot(root) == before and rep["deleted"] == 2 and rep["deleted_inodes"] == 6 + 5
    assert f"requests: would delete {d} (6 inodes, 0.0 MB)" in lines and f"requests: would delete {never} (5 inodes, 0.0 MB)" in lines


def test_a_request_pass_fails_closed(root, proc, tmp_path):
    """No runs directory, a request that names none, a /proc that cannot be read, the time limit: nothing goes."""
    d = request(root, RIDS[0])
    (d / "request.json").write_text("not json")
    age(d, 2)
    (root / "runs").rmdir()
    rep = requests(root, proc)
    assert rep["deleted"] == 0 and rep["kept_first"] == [
        {"path": str(d), "why": f"unknown: {root / 'runs'}, where its run dir would be, is not a directory"}]
    (root / "runs").mkdir()
    rep = requests(root, tmp_path / "no-proc")
    assert rep["error"].startswith("2: cannot list") and rep["deleted"] == 0 and rep["kept_by"] == {"unestablished": 1} and d.is_dir()
    more = request(root, RIDS[1])
    t = [0.0]
    rep = requests(root, proc, clock=lambda: t[0], budget_s=240, log=after_first("requests: deleted ", t, 241.0))
    assert rep["deleted_first"] == [str(d)] and rep["kept_by"] == {"budget": 1} and more.is_dir()


# ---- the clones of runs that are over --------------------------------------------------------------------------------------------------

def a_clone(root: Path, rid: str, *, runner: int | None = 50, status: dict | None = None, refused: bool = False) -> Path:
    """A run's clone under scratch/ (4 inodes) and, with a runner, its run dir: launch.json and status.json or refused.json."""
    c = root / "scratch" / rid
    (c / "tree" / "pkg").mkdir(parents=True)
    (c / "tree" / "pkg" / "m.py").write_text("x = 1\n")
    if runner is not None:
        rd = root / "runs" / rid
        rd.mkdir(parents=True)
        (rd / "launch.json").write_text(json.dumps({"runner_pid": runner}))
        if refused:
            (rd / "refused.json").write_text(json.dumps({"fault": "CloneRefused"}))
        if status is not None:
            (rd / "status.json").write_text(json.dumps(status))
    return c


def clones(root: Path, proc: Path, **kw):
    return L.evict_clones(root / "scratch", proc=proc, clock=kw.pop("clock", lambda: 0.0), **kw)


RUNNING = {"state": "running", "pgid": 60, "pid": 70, "starttime_ticks": 777}
OVER = {
    "its status is final and nothing is left of it": dict(status={"state": "done", "pgid": 60}),
    "it refused": dict(refused=True),
    "its runner is a zombie": dict(status={"state": "failed", "pgid": 60}, live=[dict(pid=50, state="Z")]),
    "its runner died with it: the process its status records has no /proc entry": dict(status=RUNNING),
    "its runner died with it: the pid its status records is another process now": dict(status=RUNNING, live=[dict(pid=70, start=999)]),
    "the last transition's process is gone": dict(
        status={"state": "running", "pgid": 60, "transitions": [{"state": "running", "pid": 70, "starttime_ticks": 777}]}),
}
NOT_OVER = {
    "2: its runner (pid 50) runs": dict(status={"state": "done", "pgid": 60}, live=[dict(pid=50)]),
    "2: 1 process(es) of its workload's group 60 run": dict(status={"state": "done", "pgid": 60}, live=[dict(pid=61, pgid=60)]),
    "2: its status is running and the process it records (pid 70) is not known to have exited": dict(
        status=RUNNING, live=[dict(pid=70, start=777)]),
    "2: its status is launched and the process it records (pid 70) is not known to have exited": dict(
        status={"state": "launched", "pgid": 60, "pid": 70}, live=[dict(pid=70, start=999)]),
    "2: its status is running and it records no process": dict(status={"state": "running", "pgid": 60}),
    "unknown: no launch.json names its runner": dict(runner=None),
    "unknown: no status.json names its workload's process group": dict(),
    "2: process 80 (p80) names its run": dict(status={"state": "done", "pgid": 60}, live=[dict(pid=80, environ=f"RESEARCH_RUN_ID={RID}\0")]),
}


@pytest.mark.parametrize("why", OVER)
def test_the_clone_of_a_run_that_is_over_goes(root, proc, why):
    """Over: the runner has exited and nothing of the workload is left, by a final status, a refusal, or a recorded process that is
    gone by its pid and its start time (a runner killed with its workload writes no final status)."""
    case = dict(OVER[why])
    for p in case.pop("live", []):
        process(proc, **p)
    c = a_clone(root, RID, **case)
    assert clone.not_over(root / "runs" / RID, proc) is None
    rep = clones(root, proc)
    assert rep["error"] is None and rep["deleted_first"] == [str(c)] and rep["deleted_inodes"] == 4 and not c.exists()
    assert (root / "runs" / RID / "launch.json").exists(), "the run's record is not the clone's"


@pytest.mark.parametrize("why", NOT_OVER)
def test_the_clone_of_a_run_that_may_not_be_over_stays_and_is_named(root, proc, why):
    """A live runner, a live member of the workload's group, a recorded process that still runs or that only a pid speaks for,
    a missing record, a live process that names the run: each keeps the clone."""
    case = dict(NOT_OVER[why])
    for p in case.pop("live", []):
        process(proc, **p)
    c = a_clone(root, RID, **case)
    before = snapshot(root)
    rep = clones(root, proc)
    assert rep["deleted"] == 0 and snapshot(root) == before
    assert rep["kept_first"] == [{"path": str(c), "why": why}] and rep["kept_by"] == {why.split(":")[0]: 1}
    if not why.startswith("2: process"):
        assert clone.not_over(root / "runs" / RID, proc) == why


def test_only_a_directory_named_like_a_run_id_is_a_clone(root, proc, tmp_path):
    victim = tmp_path / "victim"
    (victim / "keep").mkdir(parents=True)
    (root / "scratch" / "ca-rowseg").mkdir(parents=True)
    (root / "scratch" / RIDS[1]).symlink_to(victim)
    (root / "runs" / RIDS[1]).mkdir()
    (root / "runs" / RIDS[1] / "launch.json").write_text('{"runner_pid": 50}')
    (root / "runs" / RIDS[1] / "refused.json").write_text("{}")
    rep = clones(root, proc)
    assert rep["found"] == 1 and rep["deleted"] == 0 and (victim / "keep").is_dir() and (root / "scratch" / RIDS[1]).is_symlink()
    assert rep["kept_first"] == [{"path": str(root / "scratch" / "ca-rowseg"), "why": "unknown: it is not named like a run id"}]


def test_a_clone_dry_run_a_failed_removal_and_the_time_limit(root, proc, monkeypatch):
    a, b = (a_clone(root, r, status={"state": "done", "pgid": 60}) for r in RIDS[:2])
    before, lines = snapshot(root), []
    rep = clones(root, proc, dry_run=True, log=lines.append)
    assert snapshot(root) == before and rep["deleted"] == 2 and f"clones: would delete {a} (4 inodes, 0.0 MB)" in lines
    t = [0.0]
    rep = clones(root, proc, clock=lambda: t[0], budget_s=240, log=after_first("clones: deleted ", t, 241.0))
    assert rep["deleted_first"] == [str(a)] and rep["kept_by"] == {"budget": 1} and b.is_dir()
    monkeypatch.setattr(clone, "remove", lambda path: "OSError: [Errno 39] Directory not empty after 3 attempts")
    rep = clones(root, proc)
    assert rep["deleted"] == 0 and rep["kept_first"] == [
        {"path": str(b), "why": "error: OSError: [Errno 39] Directory not empty after 3 attempts"}]
    assert clones(root, root / "no-proc")["error"].startswith("2: cannot list"), "a /proc it cannot read removes nothing"
    assert b.is_dir()


# ---- every pass: `research keep` records, and how much it says -------------------------------------------------------------------------

def keep(path: Path, top: Path) -> Path:
    """Record `path` with `research keep`; the record's file."""
    retention.keep(path, owner="@infra", expires=None, priority="keep", root=top, node="node")
    return retention.sidecar(path)


def test_nothing_a_keep_record_covers_goes(cache, root, proc, tmp_path):
    """A record of its own, inside the directory or beside it, one on a directory above, one anywhere beneath, and one that
    cannot be read: each keeps the path whole, for `research retention` to decide."""
    covers = "record: a `research keep` record covers it; `research retention` decides"
    own, beside, holds = shard(cache, D1, 100), build(cache, ".tmpAbC123", 100), build(cache, ".tmpAbC124", 100)
    keep(own, tmp_path)
    os.rename(keep(beside, tmp_path), retention.beside(beside))
    nested = keep(holds / "lib", tmp_path)
    before = snapshot(cache)
    rep = uv(cache, proc)
    assert rep["deleted"] == 0 and rep["deleted_inodes"] == 0 and rep["kept_by"] == {"record": 3} and snapshot(cache) == before
    assert {k["path"]: k["why"] for k in rep["kept_first"]} == {
        str(own): covers, str(beside): covers,
        str(holds): f"record: it holds a `research keep` record ({nested}); `research retention` decides"}

    unread = request(root, RIDS[0])
    (root / "requests" / f"{RIDS[0]}.retention.json").write_text("not json")
    before = snapshot(root)
    rep = requests(root, proc)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"record": 1} and snapshot(root) == before
    assert rep["kept_first"][0]["path"] == str(unread)
    assert rep["kept_first"][0]["why"].startswith(f"record: {root / 'requests' / RIDS[0]}.retention.json: unreadable")

    for rid in RIDS[1:3]:
        a_clone(root, rid, status={"state": "done", "pgid": 60})
    keep(root / "scratch", tmp_path)
    store = tmp_path / "store"
    lock_file(store, key(1))
    keep(store / "locks", tmp_path)
    before = snapshot(root), snapshot(store)
    above = "record: a `research keep` record on {} covers it; `research retention` decides"
    rep = clones(root, proc)
    assert rep["deleted"] == 0 and rep["kept_by"] == {"record": 2}
    assert {k["why"] for k in rep["kept_first"]} == {above.format(os.path.realpath(root / "scratch"))}
    rep = locks(store)
    assert rep["deleted"] == 0 and rep["found"] == 1 and rep["kept_by"] == {"record": 1}
    assert rep["kept_first"][0]["why"] == above.format(os.path.realpath(store / "locks"))
    assert (snapshot(root), snapshot(store)) == before


def test_a_pass_names_the_first_twenty_of_each_kind_and_counts_them_all(tmp_path):
    store = tmp_path / "store"
    for n in range(25):
        lock_file(store, key(n))
        lock_file(store, f"another-programs-{n}")
    lines = []
    rep = locks(store, log=lines.append)
    assert rep["deleted"] == 25 and len(rep["deleted_first"]) == 20 and rep["kept_by"] == {"unknown": 25} and len(rep["kept_first"]) == 20
    assert len([ln for ln in lines if ln.startswith("locks: deleted /")]) == 20
    assert len([ln for ln in lines if ln.startswith("locks: kept /")]) == 20
    assert len(lines) == 41 and lines[-1] == "locks: deleted 25 of 50 (0.0 GB, 25 inodes); kept 25 unknown:25"
