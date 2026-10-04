"""The result cache: a target's report is reused only while every file its check read or resolved is unchanged.

A cached run records, per target, under the repository root, in a recording that starts when that target's check starts (never
the task list's build, nor an earlier target's check):

- every file the check opened for reading and every directory it listed (an audit hook on ``open``, ``os.listdir`` and
  ``os.scandir``), every path it only probed (``os.stat``: whether it was a file, a directory or absent), and every library built
  in the tree that it loaded (``ctypes.dlopen``: by its bytes);
- the source file of every function that ran (``sys.monitoring``: the first start or resume of each code object in the recording);
- what the Definitions the check reached were built from: each specialization whose body it read (and every one reachable from
  its target through body nodes) brings the recording of its body's build (`verity.ir.defs._build_body`: the code that ran and
  the files read while it was traced, wherever in the process that happened, its nested builds included), and each primitive
  the file of its evaluator;
- what the target's own bindings produced: the recordings of the catalog's root functions, `_model_roots` entries and
  `TEMPLATE_PARAMS` entries whose roots reach it (`targets.BINDINGS`), while they ran;
- each module among those files with its import closure: the in-tree modules its top-level ``import`` statements name (and its
  parent packages), recursively, and what was read and run while it was first imported in this process (a loader hook);
- what the process read outside every recording (the registries' imports listing their packages, ``pkgutil`` enumerating
  them): every target holds that.

`targets.py` and `pins.json` are not files of any target: a target holds the parts of `targets.py` it depends on (its binding
units and the rest of the module with every binding unit cut out, ``common``, each by the digest of its AST, so comments and
formatting are no input; any `targets.py` code its check ran, by the unit that holds it) and the entries of `pins.json` its check
looked up (`recorded_pins`), each by its value, or absent.  The checker's own version is the digest of this package's sources less
those two.  Beside them the key holds the interpreter, the platform, the installed distributions, the request (the target, its
options, and whether the partition invariant applies to it, as the catalog resolved it) and the ``VERITY_*`` / ``CIRCUIT_CHECK_*``
environment.  Each target's check runs with a memo of its own, so no result an earlier target computed stands in for its reads.

A hit means every file, listing, probe, library, `targets.py` unit and pin this target's check read or resolved is unchanged
(the tree as git has it, `visible`: bytecode or a build inside the tree counts as absent).  Not covered: a file a C extension
reads with its own I/O (none here); anything outside the repository, which the interpreter, platform and distribution list stand
for; a file a module read while it was imported before tracing began (the checker's own imports, ``verity.ir`` and
``verity.evaluation`` among them: their sources are inputs, data they opened at import is not); the modules a function imports
inside its body when it did not run in the recording (a value a process-wide cache holds from a function another target ran,
reached only that way); and an edit to a binding that does not reach the target, even if it changes which object of the same id
the catalog builds (the registry refuses two Definitions of one id).  A new file in a package the registries enumerate changes
the listing every target holds, so it reruns everything.  What the tracer cannot see (a program it starts: ``subprocess``,
``os.system``, ``exec``, ``posix_spawn``) makes the process's later results uncacheable (`untraced`).

Only a verdict is stored, of the tree its check read: not a check that raised (an ImportError says the environment was broken,
not what the target is), nor one whose reads, digested again by the process that ran it right after, differ from the digests
the entry would record (the tree changed during the run: a concurrent run rewrote it), nor one after a module was imported from
outside the tree (another tree's copy, which no recorded read covers).

Entries are files ``<dir>/<version>/<sha256 of the key>/<sha256 of the entry>.json``; there is no expiry and no path or time
shortcut.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = "circuit-check-cache/v2"
_ROOT: list[Path] = []
#: what the tracer cannot see: a program started from here, or a repository library loaded by the dynamic loader
UNTRACEABLE = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn")
_UNTRACED: list[str] = []
_INTERNAL: list[bool] = []
_HERE = Path(__file__).resolve().parent
#: the checker's files that are no input of every target: each target holds its own parts of them
TARGETS, PINS = _HERE / "targets.py", _HERE / "pins.json"


class Frame:
    """What one recording saw: reads (relative to the root), the files of the code that ran, the first lines of `targets.py`'s
    code that ran, the specializations whose body it read, and the pins it looked up."""

    __slots__ = ("files", "dirs", "probes", "libs", "code", "lines", "defs", "pins")

    def __init__(self) -> None:
        self.files: set[str] = set()
        self.dirs: set[str] = set()
        self.probes: set[str] = set()
        self.libs: set[str] = set()
        self.code: set[str] = set()
        self.lines: set[int] = set()
        self.defs: dict[int, Any] = {}
        self.pins: set[str] = set()

    def merge(self, o: "Frame", defs: bool = True) -> None:
        self.files |= o.files
        self.dirs |= o.dirs
        self.probes |= o.probes
        self.libs |= o.libs
        self.code |= o.code
        self.lines |= o.lines
        if defs:
            self.defs.update(o.defs)
        self.pins |= o.pins


#: the open recordings, innermost last; the first holds what the process read outside every recording
_FRAMES: list[Frame] = [Frame()]
#: what was read and run while each module was first imported (after `trace`), by module name
_MODULES: dict[str, Frame] = {}
#: what each binding unit read and ran while it made its roots
_PRODUCED: dict[str, Frame] = {}


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_VISIBLE: dict[Path, tuple[frozenset[str], frozenset[str]] | None] = {}


def visible(root: Path) -> tuple[frozenset[str], frozenset[str]] | None:
    """(files, directories) of the tree as git has it: tracked and untracked-unignored in a checkout; in a shipped tree (no
    `.git`), the commit's tree from the machine's bare repository (`READY.json` names it).  None when neither is available."""
    root = Path(root).resolve()
    if root not in _VISIBLE:
        import subprocess

        if (root / ".git").exists():
            argv = ["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard"]
        elif (root / "READY.json").exists() and (root.parent.parent / "git" / "verity.git").exists():
            tree = json.loads((root / "READY.json").read_text())["tree"]
            argv = ["git", "--git-dir", str(root.parent.parent / "git" / "verity.git"), "ls-tree", "-r", "-z", "--name-only", tree]
        else:
            argv = None
        _INTERNAL.append(True)
        try:
            r = subprocess.run(argv, capture_output=True) if argv else None
        finally:
            _INTERNAL.pop()
        if r is None or r.returncode:
            _VISIBLE[root] = None
        else:
            files = frozenset(x.decode() for x in r.stdout.split(b"\0") if x)
            dirs = frozenset({"."} | {str(Path(f).parents[i]) for f in files for i in range(len(Path(f).parents) - 1)})
            _VISIBLE[root] = (files, dirs)
    return _VISIBLE[root]


def _listing(root: Path, rel: str) -> str:
    names = sorted(os.listdir(root / rel))
    if (v := visible(root)) is not None:                # what a build or a run adds inside the tree is not an input
        names = [n for n in names if os.path.normpath(os.path.join(rel, n)) in v[0] | v[1]]
    return hashlib.sha256("\0".join(names).encode()).hexdigest()


def _rel(path: Any) -> str | None:
    if isinstance(path, int) or not _ROOT:
        return None
    try:
        p = Path(os.fsdecode(path))
        p = p if p.is_absolute() else Path.cwd() / p
        rel = os.path.relpath(os.path.normpath(p), _ROOT[0])
    except (TypeError, ValueError):
        return None
    if rel.startswith("..") or "__pycache__" in rel.split(os.sep) or rel.split(os.sep)[0] == ".venv":
        return None
    return rel


_TARGETS_REL: list[str | None] = [None]


def _for_linecache() -> bool:
    """Whether this open is `linecache` reading source lines for a stack it formats (torch's import formats one): text no
    report holds, so `targets.py` read that way is no input."""
    import linecache

    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename == linecache.__file__:
            return True
        f = f.f_back
    return False


def _hook(event: str, args: tuple) -> None:
    if _INTERNAL and event in ("open", "os.listdir", "os.scandir"):
        return                                         # the tracer's own reads (digesting, parsing): no input of a check
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else "r"
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return
        if isinstance(mode, str) or mode is None:
            if (rel := _rel(path)) is not None and not (rel == _TARGETS_REL[0] and _for_linecache()):
                _FRAMES[-1].files.add(rel)
    elif event in ("os.listdir", "os.scandir"):
        if (rel := _rel(args[0] if args and args[0] is not None else ".")) is not None:
            _FRAMES[-1].dirs.add(rel)
    elif event in UNTRACEABLE and not _INTERNAL:
        argv = args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else args[:1]
        _UNTRACED.append(f"{event} {' '.join(map(str, argv))[:240]}")
    elif event == "ctypes.dlopen" and args and args[0] and (rel := _rel(args[0])) is not None:
        _FRAMES[-1].libs.add(rel)                      # a library built in the tree: an input by its bytes (`_digests`)


def _probing(fn):
    def stat(path, *a, **k):
        if not _INTERNAL and (rel := _rel(path)) is not None:
            _FRAMES[-1].probes.add(rel)
        return fn(path, *a, **k)
    return stat


_TOOL: list[int] = []
_TARGETS_FILE = str(TARGETS)


def _started(code, offset):
    f = code.co_filename
    if f == _TARGETS_FILE:
        _FRAMES[-1].lines.add(code.co_firstlineno)
    else:
        _FRAMES[-1].code.add(f)
    return sys.monitoring.DISABLE


@contextmanager
def recording(merge: bool = True) -> Iterator[Frame]:
    """Record what runs inside: a new frame on top, which its enclosing recording (not the process's own) also holds when it
    ends, with ``merge``.  Every code object starts afresh in it (`sys.monitoring.restart_events`), so a function an earlier
    recording ran is seen again."""
    fr = Frame()
    if not _ROOT:
        yield fr
        return
    _FRAMES.append(fr)
    if _TOOL:
        sys.monitoring.restart_events()
    try:
        yield fr
    finally:
        _FRAMES.pop()
        if merge and len(_FRAMES) > 1:
            _FRAMES[-1].merge(fr)


def _instrument() -> None:
    """The loader hook (what each module's import read and ran, `_MODULES`), and the Definition hooks: each body's build
    recorded on its specialization, each body read noted in the recording that read it."""
    import importlib.machinery as M

    exec0 = M.SourceFileLoader.exec_module

    def exec_module(self, module):
        with recording() as fr:
            exec0(self, module)
        _MODULES[module.__name__] = fr

    M.SourceFileLoader.exec_module = exec_module
    try:
        from verity.ir import defs as D
    except ImportError:
        return
    build0 = D._build_body

    def _build_body(sp):
        # not merged: a recording that reads the body holds the specialization (`body`), and with it this build
        with recording(merge=False) as fr:
            body = build0(sp)
        sp._circuit_check_built = fr
        return body

    D._build_body = _build_body
    get = D.SpecializedDefinition.body.fget

    def body(self):
        if len(_FRAMES) > 1:
            _FRAMES[-1].defs.setdefault(id(self), self)
        return get(self)

    D.SpecializedDefinition.body = property(body)
    sig = D.SpecializedDefinition.signature

    def signature(self):                               # a built body's signature is read from the body
        if len(_FRAMES) > 1:
            _FRAMES[-1].defs.setdefault(id(self), self)
        return sig(self)

    D.SpecializedDefinition.signature = signature


def trace(root: Path) -> None:
    """Record what this process (and every process forked from it) reads under ``root`` from now on: opens and listings through
    the audit hook, existence probes (`Path.exists` and the like) through `os.stat` / `os.lstat`, which raise no audit event,
    the code that runs through `sys.monitoring`, and the Definitions' builds and reads (`_instrument`)."""
    if not _ROOT:
        _ROOT.append(Path(root).resolve())
        _TARGETS_REL[0] = _rel(TARGETS)
        sys.addaudithook(_hook)
        os.stat, os.lstat = _probing(os.stat), _probing(os.lstat)
        mon = sys.monitoring
        for tool in (4, 3, 5):
            if mon.get_tool(tool) is None:
                mon.use_tool_id(tool, "circuit-check-cache")
                mon.register_callback(tool, mon.events.PY_START, _started)
                mon.register_callback(tool, mon.events.PY_RESUME, _started)
                mon.set_events(tool, mon.events.PY_START | mon.events.PY_RESUME)
                _TOOL.append(tool)
                break
        else:
            _UNTRACED.append("sys.monitoring: no tool id is free, so the code a check runs cannot be recorded")
        _instrument()


def tracing() -> bool:
    return bool(_ROOT)


def untraced() -> list[str]:
    """What this process did that the tracer cannot see (a verdict after it is not stored)."""
    return list(_UNTRACED)


def produced(units: Iterable[str], fr: Frame) -> None:
    """Hold ``fr`` (what a binding read and ran while it made its roots) under each of its ``units``."""
    for u in units:
        _PRODUCED.setdefault(u, Frame()).merge(fr)


# --- the pins a check looked up ---------------------------------------------------------------------------------------------

class _PinKind:
    def __init__(self, kind: str, d: dict) -> None:
        self._kind, self._d = kind, d

    def get(self, key: str, default: Any = None) -> Any:
        _FRAMES[-1].pins.add(f"{self._kind}#{key}")
        return self._d.get(key, default)

    def __getattr__(self, name: str) -> Any:
        _FRAMES[-1].pins.add(f"{self._kind}#*")
        return getattr(self._d, name)

    def __getitem__(self, key: str) -> Any:
        _FRAMES[-1].pins.add(f"{self._kind}#{key}")
        return self._d[key]

    def __contains__(self, key: str) -> bool:
        _FRAMES[-1].pins.add(f"{self._kind}#{key}")
        return key in self._d

    def __iter__(self):
        _FRAMES[-1].pins.add(f"{self._kind}#*")
        return iter(self._d)

    def __len__(self) -> int:
        _FRAMES[-1].pins.add(f"{self._kind}#*")
        return len(self._d)


class recorded_pins:
    """The pins file as the checks read it, noting in the open recording each entry looked up (``get(kind).get(key)``) and the
    whole file on any other use."""

    def __init__(self, pins: dict) -> None:
        self._d = pins

    def get(self, kind: str, default: Any = None) -> Any:
        if kind not in self._d:
            _FRAMES[-1].pins.add(f"{kind}#*")
            return default
        return _PinKind(kind, self._d[kind])

    def __getitem__(self, kind: str) -> Any:
        return _PinKind(kind, self._d[kind])

    def __getattr__(self, name: str) -> Any:
        _FRAMES[-1].pins.add("*")
        return getattr(self._d, name)

    def __contains__(self, kind: str) -> bool:
        _FRAMES[-1].pins.add(f"{kind}#*")
        return kind in self._d

    def __iter__(self):
        _FRAMES[-1].pins.add("*")
        return iter(self._d)

    def __len__(self) -> int:
        _FRAMES[-1].pins.add("*")
        return len(self._d)


# --- targets.py's units -----------------------------------------------------------------------------------------------------

def _is_binding(name: str) -> bool:
    return name.startswith("_") and name.endswith(("_roots", "_bindings"))


def _digest_ast(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _model_list(fn: ast.FunctionDef) -> list[ast.expr] | None:
    """The entries of `_model_roots`' returned list, when every one is a ``(label, make)`` tuple."""
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if len(rets) != 1 or not isinstance(rets[0].value, ast.List):
        return None
    elts = rets[0].value.elts
    if all(isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant) and isinstance(e.elts[0].value, str)
           for e in elts):
        return elts
    return None


def _with_list(fn: ast.FunctionDef, keep: list[ast.expr]) -> ast.FunctionDef:
    import copy

    fn = copy.deepcopy(fn)
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    ret.value.elts = copy.deepcopy(keep)
    return fn


def _units_of(source: str) -> tuple[dict[str, str], list[tuple[int, int, str | list[tuple[int, int, str]]]]]:
    """``targets.py``'s units by name, each the digest of its AST: every binding function (`_is_binding`), each entry of
    `_model_roots` (the function with its list cut to that entry) and of ``TEMPLATE_PARAMS``, and ``common``: the module with
    every one of those cut out (a binding function to its name, the list and the dict emptied).  Beside them the line span of
    each top-level statement and the unit that holds it (for `_model_roots`, the spans of its entries)."""
    tree = ast.parse(source)
    units: dict[str, str] = {}
    spans: list[tuple[int, int, Any]] = []
    body: list[ast.stmt] = []
    for node in tree.body:
        first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        span = (first, node.end_lineno or node.lineno)
        if isinstance(node, ast.FunctionDef) and _is_binding(node.name):
            elts = _model_list(node) if node.name == "_model_roots" else None
            if elts is None:
                units[node.name] = _digest_ast(node)
                spans.append((*span, node.name))
                body.append(ast.Expr(ast.Name(node.name)))
            else:
                inner = []
                for e in elts:
                    u = f"{node.name}:{e.elts[0].value}"
                    units[u] = _digest_ast(_with_list(node, [e]))
                    inner.append((e.lineno, e.end_lineno or e.lineno, u))
                spans.append((*span, inner))
                body.append(_with_list(node, []))
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else \
            node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == "TEMPLATE_PARAMS" and isinstance(node.value, ast.Dict) \
                and all(isinstance(k, ast.Constant) for k in node.value.keys):
            for k, v in zip(node.value.keys, node.value.values):
                units[f"TEMPLATE_PARAMS:{k.value}"] = _digest_ast(v)
            spans.append((*span, "*"))
            body.append(ast.Expr(ast.Name("TEMPLATE_PARAMS")))
            continue
        spans.append((*span, "common"))
        body.append(node)
    units["common"] = _digest_ast(ast.Module(body=body, type_ignores=[]))
    return units, spans


_UNITS: dict[str, tuple[dict[str, str], list]] = {}


def _units_cached(text: str) -> tuple[dict[str, str], list]:
    h = hashlib.sha256(text.encode()).hexdigest()
    if h not in _UNITS:
        _UNITS[h] = _units_of(text)
    return _UNITS[h]


def _unit_at(spans: list, line: int) -> list[str]:
    for lo, hi, u in spans:
        if lo <= line <= hi:
            if isinstance(u, list):
                inner = [v for a, b, v in u if a <= line <= b]
                return inner or [v for _a, _b, v in u] or ["*"]
            return [u]
    return ["common"]


# --- a target's reads -------------------------------------------------------------------------------------------------------

_OUTSIDE: set[str] = set()


def _interpreter_dirs() -> tuple[str, ...]:
    import sysconfig

    dirs = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
            *(sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib"))}
    return tuple({os.path.join(f(d), "") for d in dirs if d for f in (os.path.abspath, os.path.realpath)})


def _check_outside() -> None:
    """A module imported from outside the tree and the interpreter's own directories (another tree's copy of a package, say) is a
    read no key can hold: the process's results after it are not stored (`untraced`)."""
    inside = (os.path.join(str(_ROOT[0]), ""), *_interpreter_dirs()) if _ROOT else ()
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or _rel(f) is not None:
            continue
        if inside and name not in _OUTSIDE and not any(p.startswith(inside) for p in (os.path.abspath(f), os.path.realpath(f))):
            _OUTSIDE.add(name)
            _UNTRACED.append(f"imported {name} from {f}, outside the tree")


_IMPORTS: dict[str, list[str]] = {}


def _top_imports(path: str, name: str) -> list[str]:
    """The modules ``name``'s top-level ``import`` statements name (inside ``if``, ``try`` and ``with`` too), and the submodules a
    ``from`` import may name, absolute."""
    if path in _IMPORTS:
        return _IMPORTS[path]
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError, ValueError):
        _IMPORTS[path] = []
        return []
    pkg = name if os.path.basename(path) == "__init__.py" else name.rpartition(".")[0]
    out: list[str] = []

    def walk(body):
        for n in body:
            if isinstance(n, ast.Import):
                out.extend(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    parts = pkg.split(".")
                    base = ".".join(parts[:len(parts) - (n.level - 1)] + ([n.module] if n.module else []))
                else:
                    base = n.module or ""
                out.append(base)
                out.extend(f"{base}.{a.name}" for a in n.names)
            elif isinstance(n, ast.If):
                walk(n.body)
                walk(n.orelse)
            elif isinstance(n, (ast.Try, ast.TryStar)):
                walk(n.body)
                for h in n.handlers:
                    walk(h.body)
                walk(n.orelse)
                walk(n.finalbody)
            elif isinstance(n, ast.With):
                walk(n.body)
    walk(tree.body)
    _IMPORTS[path] = out
    return out


def _evaluators(d: Any) -> list[str]:
    """The files of the functions a Definition object carries (a primitive's evaluator, a composite's signature and body, a
    reference's function)."""
    d = getattr(d, "defn", d)
    fields = getattr(d, "__dict__", {})
    return [f.__code__.co_filename for a in ("evaluate", "fn", "signature", "body") if hasattr(f := fields.get(a), "__code__")]


def attributed(fr: Frame, units: Iterable[str] = (), roots: Iterable[Any] = ()) -> dict[str, list[str]]:
    """What a target's check read or resolved: its recording ``fr``, the recordings of its binding ``units`` and of the builds
    of every specialization it read or reaches from ``roots``, each module among them with its import closure, and what the
    process read outside every recording."""
    _INTERNAL.append(True)
    try:
        return _attributed(fr, units, roots)
    finally:
        _INTERNAL.pop()


def _attributed(fr: Frame, units: Iterable[str], roots: Iterable[Any]) -> dict[str, list[str]]:
    _check_outside()
    by_file = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if f:
            by_file.setdefault(os.path.abspath(f), name)
    acc = Frame()
    acc.merge(fr)
    for u in units:
        if u in _PRODUCED:                             # what the binding ran, not every body its roots reach
            acc.merge(_PRODUCED[u], defs=False)
    outside = _FRAMES[0]
    # an import's own reads count with its module's closure, and the task list's reads of the pins and the catalog with what
    # each target looks up in them
    sources = {r for r in (_rel(f) for f in [*by_file, PINS, TARGETS]) if r is not None}
    acc.files |= outside.files - sources
    acc.dirs |= outside.dirs
    acc.probes |= outside.probes
    acc.libs |= outside.libs
    seen_defs: set[int] = set()
    stack = list(acc.defs.values()) + list(roots)
    code: set[str] = set(acc.code)
    done: set[str] = set()
    work = list(code)
    unit_names = set(units) | {"common"}

    def expand() -> None:
        while stack:
            d = stack.pop()
            if id(d) in seen_defs:
                continue
            seen_defs.add(id(d))
            built = getattr(d, "_circuit_check_built", None)
            if built is not None:
                take(built)
            work.extend(_evaluators(d))
            body = getattr(d, "_body", None)
            if body is not None:
                stack.extend(n.fn for n in body.nodes)

    def take(f: Frame) -> None:
        acc.files |= f.files
        acc.dirs |= f.dirs
        acc.probes |= f.probes
        acc.libs |= f.libs
        acc.lines |= f.lines
        acc.pins |= f.pins
        work.extend(f.code)
        stack.extend(f.defs.values())

    while True:
        expand()
        if not work:
            break
        f = os.path.abspath(work.pop())
        if f in done:
            continue
        done.add(f)
        if f == _TARGETS_FILE or f.startswith(os.path.join(str(_HERE), "")):
            continue                                   # the checker's version, and `targets.py` by its units
        if (rel := _rel(f)) is None:
            continue
        acc.files.add(rel)
        name = by_file.get(f)
        if name is None:
            continue
        if name in _MODULES:
            take(_MODULES[name])
        parts = name.split(".")
        for imp in _top_imports(f, name) + [".".join(parts[:i]) for i in range(1, len(parts))]:
            m = sys.modules.get(imp)
            if (mf := getattr(m, "__file__", None)) is not None:
                work.append(mf)
    trel = _rel(TARGETS)
    if trel is not None:
        if acc.lines:
            _units, spans = _units_cached(TARGETS.read_text())
            for line in acc.lines:
                unit_names.update(_unit_at(spans, line))
        if trel in acc.files:                          # read as a file (a traceback's source line, say): all of it
            acc.files.discard(trel)
            unit_names.add("*")
    prel = _rel(PINS)
    if prel is not None and prel in acc.files:
        acc.files.discard(prel)
        acc.pins.add("*")
    return {"files": sorted(acc.files), "dirs": sorted(acc.dirs), "probes": sorted(acc.probes - acc.files - acc.dirs),
            "libs": sorted(acc.libs), "units": sorted(f"{trel}#{u}" for u in unit_names) if trel is not None else [],
            "pins": sorted(f"{prel}#{p}" for p in acc.pins) if prel is not None else []}


def version() -> str:
    """The checker's own version: the digest of this package's sources, less `targets.py` and `pins.json` (each target holds
    its own parts of them)."""
    h = hashlib.sha256()
    _INTERNAL.append(True)
    try:
        for p in sorted(_HERE.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p not in (TARGETS, PINS):
                h.update(p.relative_to(_HERE).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    finally:
        _INTERNAL.pop()
    return h.hexdigest()


def key(request: dict) -> dict:
    from importlib import metadata

    _INTERNAL.append(True)                             # the tracer's own work (`platform` runs `uname`): no input of a check
    try:
        # the system, the CPU and the C library, not `platform.platform()`: that names the host's kernel release, which differs
        # between pods and would keep one pod's reports from ever counting on another (`tools/check/verdicts.py`)
        plat = "-".join([platform.system(), platform.machine(), *platform.libc_ver()])
    finally:
        _INTERNAL.pop()
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    return {"schema": SCHEMA, "version": version(), "request": request, "python": sys.version, "platform": plat,
            "distributions": hashlib.sha256("\n".join(dists).encode()).hexdigest(),
            "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("VERITY_", "CIRCUIT_CHECK_"))}}


def _kind(p: Path) -> str | None:
    return "dir" if p.is_dir() else "file" if p.exists() else None


def _unit(root: Path, r: str, seen) -> str | None:
    path, _, name = r.partition("#")
    if not seen(path) or not (root / path).is_file():
        return None
    if name == "*":
        return _sha_file(root / path)
    return _units_cached((root / path).read_text())[0].get(name)


def _pin(root: Path, r: str, seen) -> str | None:
    path, _, rest = r.partition("#")
    if not seen(path) or not (root / path).is_file():
        return None
    if rest == "*":
        return _sha_file(root / path)
    kind, _, k = rest.partition("#")
    pins = json.loads((root / path).read_text())
    if kind not in pins:
        return None
    v = pins[kind] if k == "*" else pins[kind].get(k)
    return None if v is None else hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def _digests(root: Path, got: dict[str, list[str]], fresh: bool = False) -> dict[str, dict[str, str | None]]:
    """Each recorded read as it is now.  With git's view of the tree (`visible`), a file or directory git does not have (bytecode,
    a build inside the tree) counts as absent: derived from what git has, it is not an input of its own.  Each read is digested
    once per process (a run changes none of its inputs) unless ``fresh``."""
    _INTERNAL.append(True)
    try:
        return _digests_of(root, got, fresh)
    finally:
        _INTERNAL.pop()


def _digests_of(root: Path, got: dict[str, list[str]], fresh: bool) -> dict[str, dict[str, str | None]]:
    v = visible(root)
    seen = (lambda r: True) if v is None else (lambda r: r in v[0] or r in v[1])

    def memo(kind: str, r: str, fn):
        if fresh:
            return fn()
        k = (str(root), kind, r)
        if k not in _MEMO:
            _MEMO[k] = fn()
        return _MEMO[k]

    files = {r: memo("f", r, lambda r=r: _sha_file(root / r) if seen(r) and (root / r).is_file() else None) for r in got["files"]}
    dirs = {r: memo("d", r, lambda r=r: _listing(root, r) if seen(r) and (root / r).is_dir() else None) for r in got["dirs"]}
    probes = {r: memo("p", r, lambda r=r: _kind(root / r) if seen(r) else None) for r in got.get("probes", [])}
    # a library the dynamic loader mapped is git-ignored (built from its source here), so it counts by its bytes
    libs = {r: memo("l", r, lambda r=r: _sha_file(root / r) if (root / r).is_file() else None) for r in got.get("libs", [])}
    units = {r: memo("u", r, lambda r=r: _unit(root, r, seen)) for r in got.get("units", [])}
    pins = {r: memo("n", r, lambda r=r: _pin(root, r, seen)) for r in got.get("pins", [])}
    return {"files": files, "dirs": dirs, "probes": probes, "libs": libs, "units": units, "pins": pins}


_MEMO: dict[tuple[str, str, str], str | None] = {}


def _dir(cache: Path, k: dict) -> Path:
    return Path(cache) / k["version"] / hashlib.sha256(json.dumps(k, sort_keys=True, default=str).encode()).hexdigest()[:32]


def lookup(cache: Path, k: dict, root: Path) -> tuple[dict, str] | None:
    """(report, entry id) of an entry with key ``k`` whose every recorded read is unchanged under ``root``, else None."""
    d = _dir(cache, k)
    if not d.is_dir():
        return None
    for p in sorted(d.glob("*.json")):
        _INTERNAL.append(True)
        try:
            e = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        finally:
            _INTERNAL.pop()
        if e.get("key") != k:
            continue
        rec = e["reads"]
        now = _digests(root, {k: list(v) for k, v in rec.items()})
        if now == rec:
            return e["report"], p.stem
    return None


def seen(root: Path, got: dict[str, list[str]]) -> dict[str, dict[str, str | None]]:
    """The reads as the process that just ran a check sees them: digested now, not remembered from earlier in the run."""
    return _digests(root, got, fresh=True)


def store(cache: Path, k: dict, root: Path, got: dict[str, list[str]], report: dict,
          worker: dict[str, dict[str, str | None]] | None = None) -> str | None:
    """Record ``report`` under ``k`` with its reads' digests; None, and nothing stored, when ``worker`` (`seen` in the process that
    ran the check) differs from them: the tree changed while the run went on, and the report may be of a tree no digest names."""
    now = _digests(root, got)
    if worker is not None and worker != now:
        return None
    e = {"key": k, "reads": now, "report": report}
    body = json.dumps(e, sort_keys=True, separators=(",", ":"), default=str)
    eid = hashlib.sha256(body.encode()).hexdigest()
    d = _dir(cache, k)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{eid}.tmp"
    tmp.write_text(body)
    os.replace(tmp, d / f"{eid}.json")
    return eid


__all__ = ["PINS", "SCHEMA", "TARGETS", "UNTRACEABLE", "Frame", "attributed", "key", "lookup", "produced", "recorded_pins",
           "recording", "seen", "store", "trace", "tracing", "untraced", "version"]
