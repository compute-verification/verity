"""What a suite's tests read outside the test process itself, for `suites.py`'s per-test verdicts. Standard library only: the guard
loads it for the children it forks and for its own process, and `sitecustomize` beside it for every Python a test starts.

Each open of one of the suite's test modules (a `test_*.py` under `VERITY_SUITE_DIR`), or of a repository file outside the suite's
inputs (`VERITY_SUITE_ALLOW`, relative to `VERITY_SUITE_ROOT`), is appended at once to `VERITY_SUITE_READS` as a line
`R|O<tab>module<tab>path`. The module is the test module `VERITY_SUITE_MODULE` names: the guard sets it while one is collected or
its tests run, and a child inherits it. One write per open, so a forked child that ends through `os._exit` loses nothing.

It also traces what a per-test key narrowed to the files the tests use (`suites.py`, `Traced`) must hold, each once per process:
- `F` a repository file opened or loaded as a shared library, outside the interpreter's environment and this tracer; a cached
  module (`__pycache__/*.pyc`) counts as its source, which the import read it for;
- `L` a repository directory listed (`os.listdir`, `os.scandir`: imports, `glob`, `pkgutil` list what they search);
- `X` a program started that this tracer won't see into, so what it reads is unknown: anything but a machine probe (`PROBES`)
  or a Python that loads this tracer (the suite's PYTHONPATH, `VERITY_SUITE_READS` and a test module in `VERITY_SUITE_MODULE`,
  and no `-I`, `-E` or `-S`).

And what no test may reach, a machine's own state (`HOST`), opened, listed, loaded or looked up (`os.stat`, `os.lstat`,
`os.access`, which `os.path` and `pathlib` go through): `H` with the path, which fails the suite (`suites.py`).
"""
from __future__ import annotations

import json
import os
import re
import sys

#: a module's cached bytecode, by its source's directory and name
_PYC = re.compile(r"^(.*?)(?:^|/)__pycache__/([^/]+?)\.[^/]+\.pyc$")
_PYTHON = re.compile(r"^python(\d+(\.\d+)?)?$")
#: A machine's own state, which no test may reach: the pod layout (`/workspace/cp`, and the model cache, venvs, jobs, caches and RAM
#: reservations beside it), the research machine root, the lease and its deadline, the GPU and CPU splits. `suites.py` points the
#: variables that name them into an empty directory for the tests (`CLEAN_HOST`), so a test that still reaches one reads a default
#: no variable moved, and passes or fails with the machine it runs on.
HOST = tuple(os.path.join(p, "") for p in ("/workspace/cp", "/workspace/hf", "/workspace/jobs", "/workspace/ramlock", "/workspace/cache",
                                           "/workspace/usage", "/workspace/research", "/workspace/venv312", "/workspace/src",
                                           "/workspace/pouw", "/etc/research", "/etc/vy", "/root/dm", "/root/.research/lease",
                                           "/root/.research/lease.log", "/root/.research/lease.pid", "/root/.research/lease.sh",
                                           "/root/.research/pod_guard.sh", "/root/.research/pods", "/run/gpu-lease"))
#: what a check reads there by design: the machine's content-addressed store and its bytecode cache
HOST_NOT = tuple(os.path.join(p, "") for p in ("/workspace/research/store", "/workspace/research/cache/pycache"))
#: the calls that look a path up without opening it
LOOKUPS = ("stat", "lstat", "access")
#: programs that read no repository file, whatever they're given: they report on the machine (`platform` runs `uname`)
PROBES = frozenset({"uname", "true", "false", "nproc", "getconf", "hostname", "id", "whoami", "ldconfig", "lscpu", "nvidia-smi"})


class Reads:
    def __init__(self, env=os.environ):
        self.root = os.path.realpath(env.get("VERITY_SUITE_ROOT", ""))
        self.suite = os.path.join(os.path.realpath(env.get("VERITY_SUITE_DIR") or os.getcwd()), "")
        paths = [os.path.normpath(os.path.join(self.root, p)) for p in json.loads(env.get("VERITY_SUITE_ALLOW", "[]"))]
        self.here = os.path.join(os.path.dirname(os.path.realpath(__file__)), "")
        self.prefix = tuple({os.path.join(os.path.realpath(sys.prefix), ""), os.path.join(os.path.abspath(sys.prefix), "")})
        self.allowed = tuple(os.path.join(p, "") for p in paths if os.path.isdir(p))
        self.allowed += (*self.prefix, self.here)                                       # the tracer itself
        self.allowed_files = frozenset(p for p in paths if not os.path.isdir(p))
        self.inside = os.path.join(self.root, "")
        self.own = tuple(os.path.join(os.path.dirname(os.path.abspath(env[k])), "") for k in
                         ("VERITY_SUITE_REPORT", "VERITY_SUITE_READS", "VERITY_SUITE_REUSE", "VERITY_SUITE_HEAVY") if env.get(k))
        self.own += tuple(os.path.join(os.path.abspath(env[k]), "") for k in ("VERITY_SUITE_HOST",) if env.get(k))   # the runner's files
        # ... and what a recorded check hands its tests: the run's custody, the store, the bytecode cache
        self.own += tuple(os.path.join(os.path.abspath(p), "") for p in (env.get("CHECK_STORE_CUSTODY"), env.get("RESEARCH_STORE"),
                                                                         getattr(sys, "pycache_prefix", None)) if p)
        self.own += HOST_NOT
        self.own += tuple(os.path.join(os.path.abspath(p), "") for p in sys.path if p)          # where code is imported from
        self.log = env.get("VERITY_SUITE_READS")
        self.busy = False
        self.sent: set[tuple[str, str, str]] = set()

    def _abs(self, path: str) -> str:
        return os.path.normpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))

    def classify(self, path: str) -> tuple[str, str] | None:
        """("R", path) for one of the suite's test modules, ("O", path) for a repository file outside the inputs, else None."""
        path = self._abs(path)
        name = os.path.basename(path)
        if path.startswith(self.suite) and name.startswith("test_") and name.endswith(".py"):
            return "R", os.path.relpath(path, self.root)
        if path.startswith(self.inside) and not path.startswith(self.allowed) and path not in self.allowed_files:
            return "O", os.path.relpath(path, self.root)
        return None

    def traced(self, path: str) -> str | None:
        """The repository path a key narrowed to what the tests use holds for an open of ``path``; None outside the repository,
        the interpreter's environment and this tracer."""
        path = self._abs(path)
        if not path.startswith(self.inside) or path.startswith((*self.prefix, self.here)):
            return None
        rel = os.path.relpath(path, self.root)
        m = _PYC.match(rel)
        return f"{m.group(1)}/{m.group(2)}.py".lstrip("/") if m else rel

    def host(self, path) -> str | None:
        """``path`` when it is a machine's own state (`HOST`) outside the repository, the interpreter's environment and the
        runner's files; else None."""
        if path is None or isinstance(path, int):
            return None
        try:
            p = self._abs(os.fsdecode(path))
        except (TypeError, ValueError):
            return None
        q = os.path.join(p, "")
        keep = (self.inside, *self.prefix, self.here, *self.own)
        if not q.startswith(HOST) or q.startswith(keep) or any(k.startswith(q) for k in keep):   # a parent of those: `realpath` walks it
            return None
        return p

    def watch(self, note) -> None:
        """Call ``note(path)`` before each lookup (`LOOKUPS`) this process makes through `os`, which has no audit event."""
        for name in LOOKUPS:
            real = getattr(os, name)
            if getattr(real, "_verity_suite_watched", False):
                continue

            def looked(path, *args, _real=real, **kwargs):
                note(path)
                return _real(path, *args, **kwargs)
            looked._verity_suite_watched = True
            for support in (os.supports_dir_fd, os.supports_fd, os.supports_follow_symlinks, os.supports_effective_ids):
                if real in support:                  # `shutil` and `os` ask these sets which calls take a descriptor
                    support.add(looked)
            setattr(os, name, looked)

    def seeable(self, argv, env) -> bool:
        """Whether a program started with ``argv`` (and ``env``, None when inherited) is a Python that traces itself here."""
        if not argv:
            return False
        exe = os.fsdecode(argv[0])
        if os.path.basename(exe) in PROBES:
            return True
        flags = [os.fsdecode(a) for a in argv[1:] if isinstance(a, (str, bytes))]
        if not (_PYTHON.match(os.path.basename(exe)) or os.path.realpath(exe) == os.path.realpath(sys.executable)):
            return False
        for a in flags:
            if a in ("-c", "-m") or not a.startswith("-") or a.startswith("--"):
                break
            if set(a[1:]) & set("IES"):
                return False
        env = os.environ if env is None else {os.fsdecode(k): os.fsdecode(v) for k, v in env.items()}
        return (all(env.get(k) for k in ("VERITY_SUITE_READS", "VERITY_SUITE_MODULE", "VERITY_SUITE_ROOT"))
                and self.here.rstrip("/") in [os.path.realpath(p) for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])

    def trace(self, event: str, args: tuple) -> tuple[str, str] | None:
        """(`F`, path), (`L`, directory) or (`X`, program) for an audit event, else None."""
        if event in ("open", "ctypes.dlopen"):
            if args and isinstance(args[0], (str, bytes, os.PathLike)) and (p := self.traced(os.fsdecode(args[0]))):
                return "F", p
        elif event in ("os.listdir", "os.scandir"):
            if args and not isinstance(args[0], int) and (p := self.traced(os.fsdecode(args[0]) if args[0] is not None else ".")):
                return "L", p
        elif event == "subprocess.Popen":
            exe, argv, _cwd, env = (list(args) + [None] * 4)[:4]
            argv = [argv] if isinstance(argv, (str, bytes)) else list(argv or [])
            if exe is not None and argv:
                argv = [exe, *argv[1:]]
            if not self.seeable(argv, env):
                return "X", os.fsdecode(argv[0]) if argv else "?"
        elif event in ("os.exec", "os.posix_spawn"):
            path, argv, env = (list(args) + [None] * 3)[:3]
            if not self.seeable([path, *list(argv or [])[1:]], env):
                return "X", os.fsdecode(path)
        elif event in ("os.system", "os.spawn"):
            return "X", event
        return None

    def write(self, kind: str, module: str, rel: str) -> None:
        if not self.log:
            return
        self.busy = True
        try:
            fd = os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, f"{kind}\t{module}\t{rel}\n".encode())
            finally:
                os.close(fd)
        finally:
            self.busy = False

    def hit(self, kind: str, module: str, rel: str) -> None:
        if (kind, module, rel) not in self.sent:
            self.sent.add((kind, module, rel))
            self.write(kind, module, rel)

    def looked(self, path, module: str) -> None:
        if not self.busy and (h := self.host(path)) is not None:
            self.hit("H", module, h)

    def seen(self, event: str, args: tuple, module: str) -> None:
        if self.busy:
            return
        if event in ("open", "os.listdir", "os.scandir", "ctypes.dlopen") and args:
            self.looked(args[0], module)
        if event == "open" and args and not isinstance(args[0], int):
            try:
                path = os.fsdecode(args[0])
            except TypeError:
                return
            if (hit := self.classify(path)) is not None:
                self.write(hit[0], module, hit[1])
        try:
            hit = self.trace(event, args)
        except (TypeError, ValueError):
            hit = None
        if hit is not None:
            self.hit(hit[0], module, hit[1])

    def install(self) -> None:
        """Trace this process (a child a test started): every open, attributed to the module it inherited."""
        module = os.environ.get("VERITY_SUITE_MODULE", "*")
        sys.addaudithook(lambda event, args: self.seen(event, args, module))
        self.watch(lambda path: self.looked(path, module))
