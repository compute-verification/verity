"""What a suite's tests read outside the test process itself, for `suites.py`'s per-test verdicts. Standard library only: the guard
loads it for the children it forks and for its own process, and `sitecustomize` beside it for every Python a test starts.

Each open of one of the suite's test modules (a `test_*.py` under `VERITY_SUITE_DIR`), or of a repository file outside the suite's
inputs (`VERITY_SUITE_ALLOW`, relative to `VERITY_SUITE_ROOT`), is appended at once to `VERITY_SUITE_READS` as a line
`R|O<tab>module<tab>path[<tab>test]`. The module is the test module `VERITY_SUITE_MODULE` names: the guard sets it while one is
collected or its tests run, and a child inherits it; the test is the node id `VERITY_SUITE_TEST` names while one runs. One write
per open, so a forked child that ends through `os._exit` loses nothing.

It also traces what a per-test key narrowed to the files the tests use (`suites.py`, `Traced`) must hold, each once per process:
- `F` a repository file opened or loaded as a shared library, outside the interpreter's environment and this tracer; a cached
  module (`__pycache__/*.pyc`) counts as its source, which the import read it for;
- `L` a repository directory listed (`os.listdir`, `os.scandir`: imports, `glob`, `pkgutil` list what they search);
- `X` a program started that this tracer won't see into, so what it reads is unknown: anything but a machine probe (`PROBES`),
  a Python that loads this tracer (the suite's PYTHONPATH, `VERITY_SUITE_READS` and a test module in `VERITY_SUITE_MODULE`,
  and no `-I`, `-E` or `-S`), or one of the suite's declared build outputs (`VERITY_SUITE_PROGRAMS`, `given`).

A declared build output (a Lean executable) reads what its arguments name and what it derives from them, and nothing else of the
repository, so its start is traced as `F` reads instead: each argument (or `--flag=` value) that names a repository path, a file
with every file beside it and a directory with every file under it, and `O` for those outside the inputs. `lake build` of
declared outputs in their own package is traced as reading nothing: what it builds is the output, which the key holds.
"""
from __future__ import annotations

import json
import os
import re
import sys

#: a module's cached bytecode, or the temporary file an import writes it through, by its source's directory and name
_PYC = re.compile(r"^(.*?)(?:^|/)__pycache__/([^/]+?)\.[^/]+\.pyc(?:\.\d+)?$")
_PYTHON = re.compile(r"^python(\d+(\.\d+)?)?$")
#: programs that read no repository file, whatever they're given: they report on the machine (`platform` runs `uname`)
PROBES = frozenset({"uname", "true", "false", "nproc", "getconf", "hostname", "id", "whoami", "ldconfig", "lscpu", "nvidia-smi"})
#: more files than this under a declared output's argument: traced as a program no tracer sees into, not file by file
GIVEN_MAX = 5000


def _source(path: str) -> str:
    m = _PYC.match(path)
    return os.path.join(m.group(1), f"{m.group(2)}.py") if m else path

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
        #: "0" for a suite whose keys leave Markdown out (`suites.markdown`): every `.md` it opens is outside its inputs
        self.markdown = env.get("VERITY_SUITE_MARKDOWN", "1") != "0"
        self.log = env.get("VERITY_SUITE_READS")
        self.programs = frozenset(os.path.realpath(os.path.join(self.root, p)) for p in json.loads(env.get("VERITY_SUITE_PROGRAMS", "[]")))
        self.busy = False
        self.sent: set[tuple[str, str, str]] = set()

    def _abs(self, path: str) -> str:
        return os.path.normpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))

    def classify(self, path: str) -> tuple[str, str] | None:
        """("R", path) for one of the suite's test modules, ("O", path) for a repository file outside the inputs (a cached
        module reported as its source, which the import read it for), else None."""
        path = self._abs(path)
        name = os.path.basename(path)
        if path.startswith(self.suite) and name.startswith("test_") and name.endswith(".py"):
            return "R", os.path.relpath(path, self.root)
        if self.outside(path):
            return "O", os.path.relpath(_source(path), self.root)
        return None

    def outside(self, path: str) -> bool:
        """Whether ``path`` (absolute and normalized) is a repository file outside the inputs; a cached module is inside when its
        source is, and Markdown is outside them in a suite whose keys leave it out."""
        if not path.startswith(self.inside):
            return False
        if not self.markdown and path.endswith(".md") and not path.startswith((*self.prefix, self.here)):
            return True
        return not path.startswith(self.allowed) and path not in self.allowed_files and _source(path) not in self.allowed_files

    def traced(self, path: str) -> str | None:
        """The repository path a key narrowed to what the tests use holds for an open of ``path``; None outside the repository,
        the interpreter's environment and this tracer."""
        path = self._abs(path)
        if not path.startswith(self.inside) or path.startswith((*self.prefix, self.here)):
            return None
        return os.path.relpath(_source(path), self.root)

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

    def given(self, argv, cwd) -> list[str] | None:
        """The repository files (from the root) a declared build output started with ``argv`` in ``cwd`` reads, [] for `lake build`
        of declared outputs in their package, None for any other program."""
        if not argv or not self.programs:
            return None
        cwd = os.fsdecode(os.fspath(cwd)) if cwd is not None else os.getcwd()
        exe = os.fsdecode(os.fspath(argv[0]))
        args = [os.fsdecode(os.fspath(a)) for a in argv[1:] if isinstance(a, (str, bytes, os.PathLike))]
        if os.path.basename(exe) == "lake":
            built = [os.path.realpath(os.path.join(cwd, ".lake", "build", "bin", t)) for t in args[1:]]
            return [] if args[:1] == ["build"] and built and all(b in self.programs for b in built) else None
        if os.sep not in exe or os.path.realpath(os.path.join(cwd, exe)) not in self.programs:
            return None
        files: set[str] = set()
        for a in args:
            value = a.split("=", 1)[1] if a.startswith("-") and "=" in a else a
            if not value or value.startswith("-"):
                continue
            p = os.path.realpath(os.path.join(cwd, value))
            if not p.startswith(self.inside) or p.startswith((*self.prefix, self.here)):
                continue
            if os.path.isdir(p):
                found = [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
            elif os.path.isfile(p):
                d = os.path.dirname(p)
                found = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            else:
                continue
            files.update(os.path.relpath(f, self.root) for f in found)
            if len(files) > GIVEN_MAX:
                return None
        return sorted(files)

    def trace(self, event: str, args: tuple) -> list[tuple[str, str]]:
        """(`F`, path), (`L`, directory), (`O`, path) or (`X`, program) for an audit event."""
        if event in ("open", "ctypes.dlopen"):
            if args and isinstance(args[0], (str, bytes, os.PathLike)) and (p := self.traced(os.fsdecode(args[0]))):
                return [("F", p)]
        elif event in ("os.listdir", "os.scandir"):
            if args and not isinstance(args[0], int) and (p := self.traced(os.fsdecode(args[0]) if args[0] is not None else ".")):
                return [("L", p)]
        elif event in ("subprocess.Popen", "os.exec", "os.posix_spawn"):
            if event == "subprocess.Popen":
                exe, argv, cwd, env = (list(args) + [None] * 4)[:4]
                argv = [argv] if isinstance(argv, (str, bytes)) else list(argv or [])
                if exe is not None and argv:
                    argv = [exe, *argv[1:]]
            else:
                path, argv, env = (list(args) + [None] * 3)[:3]
                argv, cwd = [path, *list(argv or [])[1:]], None
            if self.seeable(argv, env):
                return []
            if (files := self.given(argv, cwd)) is None:
                return [("X", os.fsdecode(argv[0]) if argv else "?")]
            return [("F", f) for f in files] + [("O", f) for f in files if self.outside(os.path.join(self.root, f))]
        elif event in ("os.system", "os.spawn"):
            return [("X", event)]
        return []

    def write(self, kind: str, module: str, rel: str, test: str = "") -> None:
        if not self.log:
            return
        self.busy = True
        try:
            fd = os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, (f"{kind}\t{module}\t{rel}" + (f"\t{test}" if test else "") + "\n").encode())
            finally:
                os.close(fd)
        finally:
            self.busy = False

    def seen(self, event: str, args: tuple, module: str, test: str = "") -> None:
        if self.busy:
            return
        if event == "open" and args and not isinstance(args[0], int):
            try:
                path = os.fsdecode(args[0])
            except TypeError:
                return
            if (hit := self.classify(path)) is not None:
                self.write(hit[0], module, hit[1], test if hit[0] == "O" else "")
        try:
            hits = self.trace(event, args)
        except (TypeError, ValueError):
            hits = []
        for kind, rel in hits:
            if (kind, module, rel) not in self.sent:
                self.sent.add((kind, module, rel))
                self.write(kind, module, rel, test if kind == "O" else "")

    def install(self) -> None:
        """Trace this process (a child a test started): every open, attributed to the module it inherited."""
        module, test = os.environ.get("VERITY_SUITE_MODULE", "*"), os.environ.get("VERITY_SUITE_TEST", "")
        sys.addaudithook(lambda event, args: self.seen(event, args, module, test))
