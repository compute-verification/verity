"""`research data evict`'s passes over what nothing else removes (`store/leftovers.py`): a uv cache's idle editable shards and
build directories, and the store's lock files.
Each group shows the removal and everything that must stay: in use by a live process, too young, a lock held, a run that has not
ended, a name or layout the pass does not know.  Every path is under `tmp_path`, /proc is a directory there, and time is the
`now` and `clock` each pass is given."""
import fcntl
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.store import fetchset
from research.store import leftovers as L

NOW = 2_000_000_000.0
H = 3600.0


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
