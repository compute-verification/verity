"""What the slow Lean tests build, kept on the machine across runs under a key of all the build reads: a directory that
`once(name, make, key)` fills (conftest.py, `made`) and a Lean executable's run (`run`).

A key is a sha256 over `FORMAT` and its parts: an absolute path by its content (a directory's: each file's relative path and
content), any other string as itself, and with `code`, the static import closure of those modules (a repository module by its
content and the environment variables it names, the rest by `uv.lock` and the interpreter). `run` keys an executable that
reads only its argv's paths and what is under them (`flock-verify`, `flock-witness`). A directory's build draws its OS bytes
from a stream of its name, so two builds under one key are the same bytes, and is kept with the repository files it opened,
which a hit must find unchanged. What can't be keyed (a relative or absent path, a read outside the repository, a process or
thread started, a function run from a repository file outside the closure, a key that changed as it ran) is built and not
kept. Entries are written whole (a temporary name, then a rename); each name keeps its newest `KEEP`, in `VERITY_FLOCK_CACHE`
(by default `~/.cache/verity/flock-once`)."""
from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path

FORMAT = "verity/flock-once/v1"
ROOT = Path(__file__).resolve().parents[4]
CACHE = Path(os.environ.get("VERITY_FLOCK_CACHE") or Path.home() / ".cache" / "verity" / "flock-once")
KEEP = {"dir": 2, "run": 256}
#: a build that starts another process or thread is not kept: what that one reads or runs is not seen
SPAWN = {"subprocess.Popen", "_posixsubprocess.fork_exec", "os.fork", "os.forkpty", "os.system", "os.posix_spawn", "os.exec",
         "_thread.start_new_thread", "_thread.start_joinable_thread"}
_ENV = re.compile(r"""(?:environ(?:\.get)?\s*[\[(]|getenv\s*\()\s*["']([A-Za-z0-9_]+)["']""")
_EVENTS = (sys.monitoring.events.PY_START, sys.monitoring.events.PY_RESUME)
_TOOL, _digests, _closure, _local, _urandom, _building = 4, {}, {}, threading.local(), os.urandom, threading.Lock()


def _os_bytes(size: int) -> bytes:
    """`os.urandom`, but in a thread building a directory, the next bytes of its name's stream."""
    if (s := getattr(_local, "stream", None)) is None:
        return _urandom(size)
    s[1] += 1
    return hashlib.shake_256(f"{FORMAT}/os-bytes\0{s[0]}\0{s[1]}".encode()).digest(size)


def _audit(event: str, args: tuple) -> None:
    if (opened := getattr(_local, "opened", None)) is not None and (event in SPAWN or event == "open" and args):
        with contextlib.suppress(TypeError):  # an open of a file descriptor
            opened.add(f"<{event}>" if event in SPAWN else os.path.realpath(os.fsdecode(args[0])))


def _ran(code, offset: int):
    """In a thread building a directory, the file of each function it starts (each once: then not until the next build).
    Audit hooks, such as suites.py's guard and tracer, run with no events."""
    if (ran := getattr(_local, "ran", None)) is not None:
        ran.add(code.co_filename)
        return sys.monitoring.DISABLE


os.urandom = _os_bytes
sys.addaudithook(_audit)
sys.monitoring.use_tool_id(_TOOL, "verity-flock-once")
_ = [sys.monitoring.register_callback(_TOOL, e, _ran) for e in _EVENTS]


def _raise(e: OSError):
    raise e


#: below this size a file is hashed at every read: rewritten in place to the same length within one timestamp tick, it keeps
#: its stat, so a digest remembered by stat would key a run on bytes it no longer has
SMALL = 1 << 20


def _file(p) -> str:
    st = os.stat(p)
    k = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if st.st_size < SMALL or k not in _digests:
        with open(p, "rb") as f:
            _digests[k] = hashlib.file_digest(f, "sha256").hexdigest()
    return _digests[k]


def _part(x) -> str | None:
    s = os.fspath(x)
    if not os.path.isabs(s):
        return None if os.path.exists(s) else f"s {s}"
    if not os.path.isdir(s):
        return f"f {_file(s)}" if os.path.isfile(s) else None
    h = hashlib.sha256()
    for d, dirs, files in os.walk(s, onerror=_raise, followlinks=True):
        dirs.sort()
        for f in sorted(files):
            h.update(f"{os.path.relpath(os.path.join(d, f), s)}\0{_file(os.path.join(d, f))}\n".encode())
    return f"d {h.hexdigest()}"


@functools.cache
def _code(roots: tuple[str, ...]) -> tuple[str, frozenset]:
    """The static import closure of the modules `roots` (imports in function bodies too) as a key's part, and its files. A
    module it misses can't make a build kept: what the build runs outside the closure refuses it (`made`)."""
    seen, env, visited, todo = {}, set(), set(), list(roots)
    while todo:
        if (name := todo.pop()) in visited:
            continue
        visited.add(name)
        todo += [name.rsplit(".", i)[0] for i in range(1, name.count(".") + 1)]
        with contextlib.suppress(Exception):
            f = importlib.util.find_spec(name).origin
            f = os.path.realpath(f) if isinstance(f, str) and os.path.isabs(f) else ""
            if f.startswith(f"{ROOT}/") and not f.startswith(f"{os.path.realpath(sys.prefix)}/"):
                seen[name] = f
        if not seen.get(name, "").endswith(".py"):
            continue
        env |= set(_ENV.findall(src := Path(seen[name]).read_text()))
        package = (name if seen[name].endswith("__init__.py") else name.rpartition(".")[0]).split(".")
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                todo += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = ".".join(package[:len(package) + 1 - n.level] + [n.module or ""]).strip(".") if n.level else n.module
                todo += [base, *(f"{base}.{a.name}" for a in n.names)]
    # where entries are kept, and which run is asking: never what is built. A build that reads the run identity for more
    # than scratch paths spawns a process (the binaries record RESEARCH_RUN_ID), which is never kept.
    env -= {"VERITY_FLOCK_CACHE", "RESEARCH_RUN_DIR", "RESEARCH_RUN_ID"}
    return json.dumps([sorted((n, _file(f)) for n, f in seen.items()), _file(ROOT / "uv.lock"), sys.version, sys.platform,
                       [(v, os.environ.get(v)) for v in sorted(env)]]), frozenset(seen.values())


def key(*parts, code=()) -> str | None:
    """The key of `parts` (paths by content, other strings as themselves) and of the modules `code`; None if one can't be."""
    try:
        ps, (c, files) = [_part(x) for x in parts], _code(tuple(code)) if code else ("", frozenset())
    except OSError:
        return None
    k = hashlib.sha256(json.dumps([FORMAT, ps, c]).encode()).hexdigest() if None not in ps else None
    _closure[k] = files
    return k


def _keep(top: Path, keep: int, write) -> None:
    """`write(tmp)`, which renames `tmp` into `top`; then only the newest `keep` entries of `top` stay."""
    tmp = top / f".tmp-{os.getpid()}-{threading.get_ident()}"
    with contextlib.suppress(OSError):  # a rename refused: another run kept it first
        top.mkdir(parents=True, exist_ok=True)
        write(tmp)
    shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() else tmp.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        for p in top.glob(".tmp-*"):
            if not Path(f"/proc/{p.name.split('-')[1]}").exists():
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        for p in sorted((p for p in top.iterdir() if not p.name.startswith(".")), key=lambda p: p.stat().st_mtime_ns)[:-keep]:
            shutil.rmtree(p) if p.is_dir() else p.unlink()


def run(argv: list, check: bool = False) -> subprocess.CompletedProcess:
    """`subprocess.run(argv, capture_output=True, text=True)` of a Lean executable (`argv[0]` absolute), or its kept exit code
    and stdout (stderr empty); kept when it exits 0 or 1 with output."""
    argv = [os.fspath(a) for a in argv]
    k, paths = key("run", *argv), sorted(((a, i) for i, a in enumerate(argv) if os.path.isabs(a)), key=lambda t: -len(t[0]))
    f, p = CACHE / f"run-{Path(argv[0]).name}" / f"{k}.json", None
    if k:
        with contextlib.suppress(OSError, ValueError, KeyError):
            got = json.loads(f.read_text())
            os.utime(f)
            out = functools.reduce(lambda s, t: s.replace(f"\0{t[1]}\0", t[0]), paths, got["stdout"])
            p = subprocess.CompletedProcess(argv, got["returncode"], out, "")
    if p is None:
        p = subprocess.run(argv, capture_output=True, text=True, check=False)
        out = functools.reduce(lambda s, t: s.replace(t[0], f"\0{t[1]}\0"), paths, p.stdout)
        kept = k and p.returncode in (0, 1) and p.stdout and key("run", *argv) == k
        if kept and (str(ROOT) in out or f"{tempfile.gettempdir()}/" in out):
            warnings.warn(f"{argv[:2]}: not kept: its output names a path that is not its argv's")
        elif kept:
            _keep(f.parent, KEEP["run"], lambda tmp: (tmp.write_text(json.dumps({"returncode": p.returncode, "stdout": out})),
                                                      os.replace(tmp, f)))
    if check and p.returncode:
        raise subprocess.CalledProcessError(p.returncode, argv, p.stdout, p.stderr)
    return p


def made(name: str, key, d: Path, make) -> None:
    """`make(d)`, or the build kept under `key()` (None: `make(d)` alone) copied into `d`, which is empty."""
    if not (k := key() if key else None):
        return make(d)
    e = CACHE / name / k
    with contextlib.suppress(OSError, ValueError):
        if all(_file(ROOT / r) == h for r, h in json.loads((e / "reads.json").read_text()).items()):
            os.utime(e)
            shutil.copytree(e / "tree", d, dirs_exist_ok=True)
            return
    shutil.rmtree(d)
    d.mkdir()
    if not _building.acquire(blocking=False):  # what a build runs is seen process-wide: one build at a time
        return make(d)
    _local.stream, _local.opened, _local.ran = [name, 0], set(), set()
    sys.monitoring.restart_events()
    sys.monitoring.set_events(_TOOL, _EVENTS[0] | _EVENTS[1])
    try:
        make(d)
        opened, ran = _local.opened, _local.ran
    finally:
        sys.monitoring.set_events(_TOOL, 0)
        _local.stream = _local.opened = _local.ran = None
        _building.release()
    skip = ("/proc/", "/sys/", "/dev/", *(f"{os.path.realpath(q)}/" for q in (d, sys.prefix, sys.base_prefix)))
    opened = {p for p in opened if not p.startswith(skip) and not p.endswith(".pyc")}
    ran = {p for p in (os.path.realpath(f) for f in ran if not f.startswith("<")) if not p.startswith(skip)} - _closure[k]
    outside = sorted(p for p in opened if not p.startswith(f"{ROOT}/")) + sorted(ran)
    if outside or key() != k:
        return warnings.warn(f"{name}: not kept: {'it read or ran ' + outside[0] if outside else 'an input changed as it ran'}")

    def write(tmp: Path) -> None:
        shutil.copytree(d, tmp / "tree")
        reads = {os.path.relpath(p, ROOT): _file(p) for p in sorted(opened) if os.path.isfile(p)}
        (tmp / "reads.json").write_text(json.dumps(reads))
        os.rename(tmp, e)

    _keep(CACHE / name, KEEP["dir"], write)
