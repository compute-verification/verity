"""What the slow Lean tests build, kept on the machine across runs under a key of everything the build reads: a directory
`once(name, make, key)` fills (conftest.py, `made`), and a Lean executable's run (`run`).

A key is a sha256 over `FORMAT` and its parts: an absolute path by its content (a directory's: each file under it, by its
relative path), any other string as itself, and with `code`, the import closure of those modules (each repository module
by its content and the environment variables it names; the rest by `uv.lock` and the interpreter). `run` keys the argv of
an executable that reads only its argv's files and what is under its argv's directories (`flock-verify`, `flock-witness`),
the executable by its bytes, and keeps the exit code and stdout, each argv path in it written as its index. A directory's
build draws its OS bytes from a stream of its name, so two builds under one key are the same bytes, and keeps the
repository files it opened, which a hit must find unchanged. What can't be keyed (a relative or absent path, an import of a
computed name, a read outside the repository or by another process, a key that changed while it ran) is built and not kept. Entries are written
whole (a temporary name, then a rename), and each name keeps its newest `KEEP`. `VERITY_FLOCK_CACHE` is the directory,
`~/.cache/verity/flock-once` by default."""
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
#: the closure's imports of a computed name, each in a command-line entry no build here runs: (module, call)
COMPUTED = {("verity_flock.class_statement", "importlib.util.spec_from_file_location('stress_program', mod)"),
            ("verity_flock.class_statement", "importlib.import_module(mod)")}
_LOADS = {"import_module", "__import__", "spec_from_file_location", "run_path", "run_module", "entry_points"}
_ENV = re.compile(r"""(?:environ(?:\.get)?\s*[\[(]|getenv\s*\()\s*["']([A-Za-z0-9_]+)["']""")
_digests: dict = {}
_local = threading.local()
_urandom = os.urandom


def _os_bytes(size: int) -> bytes:
    """`os.urandom`, but in a thread building a directory, the next bytes of its name's stream."""
    s = getattr(_local, "stream", None)
    if s is None:
        return _urandom(size)
    s[1] += 1
    return hashlib.shake_256(f"{FORMAT}/os-bytes\0{s[0]}\0{s[1]}".encode()).digest(size)


def _audit(event: str, args: tuple) -> None:
    opened = getattr(_local, "opened", None)
    if opened is not None and event in ("subprocess.Popen", "os.fork", "os.system", "os.posix_spawn", "os.exec"):
        opened.add(f"<{event}>")  # another process's reads are not seen
    elif opened is not None and event == "open" and args and not isinstance(args[0], int):
        with contextlib.suppress(TypeError):
            opened.add(os.path.abspath(os.fsdecode(args[0])))


os.urandom = _os_bytes
sys.addaudithook(_audit)


def _raise(e: OSError):
    raise e


def _file(p) -> str:
    st = os.stat(p)
    k = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if k not in _digests:
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
def _code(roots: tuple[str, ...]) -> str | None:
    """The import closure of the modules `roots` as a key's part; None at an import of a computed name."""
    seen, env, visited, todo = {}, set(), set(), list(roots)
    while todo:
        name = todo.pop()
        if name in visited:
            continue
        visited.add(name)
        todo += [name.rsplit(".", i)[0] for i in range(1, name.count(".") + 1)]
        with contextlib.suppress(ImportError, ValueError, AttributeError):
            f = importlib.util.find_spec(name).origin
            f = os.path.realpath(f) if isinstance(f, str) and os.path.isabs(f) else ""
            if f.startswith(f"{ROOT}/") and not f.startswith(f"{os.path.realpath(sys.prefix)}/"):
                seen[name] = f
        if not seen.get(name, "").endswith(".py"):
            continue
        src = Path(seen[name]).read_text()
        env |= set(_ENV.findall(src))
        package = (name if seen[name].endswith("__init__.py") else name.rpartition(".")[0]).split(".")
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) in _LOADS:
                if (name, ast.unparse(n)) not in COMPUTED:
                    return None
            elif isinstance(n, ast.Import):
                todo += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = ".".join(package[:len(package) + 1 - n.level] + [n.module or ""]).strip(".") if n.level else n.module
                todo += [base, *(f"{base}.{a.name}" for a in n.names)]
    return json.dumps([sorted((n, _file(f)) for n, f in seen.items()), _file(ROOT / "uv.lock"), sys.version, sys.platform,
                       [(v, os.environ.get(v)) for v in sorted(env)]])


def key(*parts, code=()) -> str | None:
    """The key of `parts` (paths by content, strings as themselves) and of the import closure of the modules `code`
    (dotted names); None if any of them can't be hashed."""
    try:
        ps, c = [_part(x) for x in parts], _code(tuple(code)) if code else ""
    except OSError:
        return None
    return None if None in ps or c is None else hashlib.sha256(json.dumps([FORMAT, ps, c]).encode()).hexdigest()


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

        def write(tmp: Path) -> None:
            tmp.write_text(json.dumps({"returncode": p.returncode, "stdout": out}))
            os.replace(tmp, f)

        kept = k and p.returncode in (0, 1) and p.stdout and key("run", *argv) == k
        if kept and (str(ROOT) in out or f"{tempfile.gettempdir()}/" in out):
            warnings.warn(f"{argv[:2]}: not kept: its output names a path that is not its argv's")
        elif kept:
            _keep(f.parent, KEEP["run"], write)
    if check and p.returncode:
        raise subprocess.CalledProcessError(p.returncode, argv, p.stdout, p.stderr)
    return p


def made(name: str, key, d: Path, make) -> None:
    """`make(d)`, or the build kept under `key()` (None: `make(d)` alone) copied into `d`, which is empty."""
    k = key() if key else None
    if not k:
        return make(d)
    e = CACHE / name / k
    with contextlib.suppress(OSError, ValueError):
        if all(_file(ROOT / r) == h for r, h in json.loads((e / "reads.json").read_text()).items()):
            os.utime(e)
            shutil.copytree(e / "tree", d, dirs_exist_ok=True)
            return
    shutil.rmtree(d)
    d.mkdir()
    _local.stream, _local.opened = [name, 0], set()
    try:
        make(d)
        opened = _local.opened
    finally:
        _local.stream = _local.opened = None
    skip = (f"{d}/", f"{sys.prefix}/", f"{sys.base_prefix}/", "/proc/", "/sys/", "/dev/")
    opened = {p for p in opened if not p.startswith(skip) and not p.endswith(".pyc")}
    outside = sorted(p for p in opened if not p.startswith(f"{ROOT}/"))
    if outside or key() != k:
        return warnings.warn(f"{name}: not kept: {'it read ' + outside[0] if outside else 'an input changed while it ran'}")

    def write(tmp: Path) -> None:
        shutil.copytree(d, tmp / "tree")
        reads = {os.path.relpath(p, ROOT): _file(p) for p in sorted(opened) if os.path.isfile(p)}
        (tmp / "reads.json").write_text(json.dumps(reads))
        os.rename(tmp, e)

    _keep(CACHE / name, KEEP["dir"], write)
