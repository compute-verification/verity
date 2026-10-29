"""The result cache: a target's report is reused only while everything its check read or resolved is unchanged, wherever a
pure move (files renamed, their import lines rewritten) put the modules among it.

A cached run records, per target, under the repository root, in a recording that starts when that target's check starts (never
the task list's build, nor an earlier target's check):

- every file the check opened for reading and every directory it listed (an audit hook on ``open``, ``os.listdir`` and
  ``os.scandir``; not the import system reading a module's source or searching a directory for a module, which the module's
  name and content stand for), every path it only probed (``os.stat``: whether it was a file, a directory or absent; when
  `inspect` or `linecache` probes a loaded module's source file, the module, by its name), and every library built in the tree
  that it loaded (``ctypes.dlopen``: by its bytes);
- the module of every function that ran (``sys.monitoring``: the first start or resume of each code object in the recording),
  and every module it imported, already loaded or not (``__import__`` and ``importlib.import_module``: what a module registered
  at import decides a lookup that runs none of its code); a module imported by a name computed at run time (``import_module``,
  ``__import__`` called rather than an ``import`` statement), and every name an import of which failed;
- what the Definitions the check reached were built from: each specialization whose body it read (and every one reachable from
  its target through body nodes) brings the recording of its body's build (`verity.ml.circuits.defs._build_body`: the code that ran and
  the files read while it was traced, wherever in the process that happened, its nested builds included), and each primitive
  the module of its evaluator;
- what made the Definitions it reached: the recording of each catalog root function, `_model_roots` entry or `TEMPLATE_PARAMS`
  entry while it ran (`producing`), held by every Definition made while it ran (`made_by`: a primitive, composite or
  specialization constructed then, and each specialization of a composite constructed then) and by every target that reaches
  one; a target holds the binding units of every binding whose roots reach it (`targets.BINDINGS`) by their digests, but
  not what a binding ran when it made none of what the target reaches (a registry module a root function imports is no input
  of a primitive it returns);
- each of those modules with its import closure: the in-tree modules its top-level ``import`` statements name (and its parent
  packages), recursively, and what was read and run while it was first imported in this process (a loader hook, which compiles
  each from its source);
- what the process read outside every recording (``pkgutil`` enumerating the registries, the files their imports read): every
  target holds that.

A module is held by its name and its content, not its path (`_modrec`): the digest of its AST with every ``from`` import and
every ``import M as N`` cut to the names it binds (`_Norm`: comments, formatting, how import statements are split and how a run
of them binding different names is ordered are no input, `_commuted`, but not inside a ``try`` or ``with`` body nor across a
``__future__`` import), and, for each import site (each dotted prefix of a plain ``import a.b`` among them), the module it
resolves to (by the path-based finder over the in-tree ``sys.path`` entries, `_resolve`).  A module that reads its own path or
name (``__file__``, ``__name__``, ``__spec__`` and the like) is held at its path too.  The binding files (`targets.py` and each
Definition module's sidecar, `sidecars`) and the pins files (``pins/``'s, one per pin, and each sidecar's) are not files of any
target: a target holds the parts of each binding file it depends on (its binding units and the rest of that file with every
binding unit cut out, ``common``, each its AST digested as a module's is, with its import sites, a sidecar's resolved from its own package as
Python imports it; any binding code its check ran, by the unit that holds it) and the pins its check looked up
(`recorded_pins`), each in the file that holds it (one no file holds, where it would be added, `sidecars.pin_reads`), by its
value, or absent.  `targets.py` and ``pins/``'s files are named by their paths, a sidecar by its module's name (``<module>``
and `SUFFIX`, `_side`), so it moves with its module.  Likewise each Definition whose kernels its check looked up in
`verity.ml.circuits.evaluation.batch._KERNELS` (`_RecordedKernels`), by what is registered for it now (each kernel's name, domain, version
and which function and sampler it is, `_where`: by its module's content and its own definition, not its path), or none, and by
each in-tree module whose import registered it, as a module it ran (so a wrapper's `_where` hides no change of what it wraps):
the module that registers a kernel is imported with the registries, and no check of a target without one runs any of it.  A
package an enumerator listed (`targets.load_registries`, the vLLM registry's ``catalog.load`` and ``boolean._load``,
`_ENUMERATORS`: each imports every module it lists and names any that fails, nothing else) is held by its listing, the modules
it lists by their names, each module's content used only to follow a move (`_children`), and the modules it imported are not
pinned to their names.  A module whose source `inspect` looked for is held by its name and content (``present``), unless it is
the checker's own: those, like the checker code that ran, are held by the checker's version alone.  The checker's
own version is the digest of this package's code (each module's AST without its docstrings, `_code`, so an edit to its prose or
comments reruns no target) less `targets.py`, ``pins/``, `heavy.json` (`HEAVY`: the scheduler's
peaks, which decide which targets run beside which, never what a check computes, and which no recording holds either,
`cli.heavy`), the known failures (`KNOWN`, which decide only the run's summary) and `SHARD` (which targets a shard checks and how
shards' records are joined).  Beside them the key holds the interpreter and its optimization level, the platform, the request (the target, its options, whether the partition invariant applies to it, as the catalog resolved it, and for
a Definition the digest of every primitive and specialization it reaches, each by its descriptor id and canonical descriptor
entry, `descriptor`), the modules the registries import for having a sidecar (`registries`: a new or removed sidecar, or a
moved module that has one, reruns every target, as a new import line in `targets.py` would) and the ``VERITY_*`` /
``CIRCUIT_CHECK_*`` environment. Each target's check runs with a memo of its own, so
no result an earlier target computed stands in for its reads.

An installed distribution is a read, not part of the key (``dists``, `_outside_tops`): each top-level module the process loaded
from outside the tree, held by the versions of the distributions that provide it and by no file in the tree resolving its
name, so a new workspace member or a lock entry no check imports reuses every target, and a version a check loaded reruns it.
A module in the tree is held by its name and resolved again at lookup (``modules``), so a new ``sys.path`` entry or editable
redirect that shadows it reruns what holds it, without ``sys.path`` in the key.

A hit means every file, listing, probe, library, ``pins/`` pin and kernel registration this target's check read is unchanged
(the tree as git has it, `visible`: a file git does not track, bytecode, a build or a fetched file inside the tree, counts as
absent, so a read of one is never keyed on its content), and every module, binding unit and sidecar pin it held is unchanged but
for where its modules are (`_moved`): following the import sites from its units and from each module still at its name, every
module it held is found, its content unchanged, at the name the sites now resolve to, and a sidecar's units and pins where its
module is found (one no site leads to: the one module of that content this process imported before any check); a module imported
by a computed name, that reads its own path or name, or an import of which failed, is still at its name; each package a moved
module now imports through is one it held, or empty or a docstring alone; each enumerated package lists the same modules, or
each it lists no longer is imported anyway and each new one it lists is a renamed one or empty; and each module whose source
`inspect` looked for still resolves by its name (its content may differ: only where it is was asked), or is the one imported
module of its content.  Not covered: a file a C extension reads with its own I/O (none here); anything outside the repository,
which the interpreter, platform, distribution list and ``sys.path`` stand for; a file a module read while it was imported before
tracing began (the checker's own imports, ``verity.ml.circuits`` and ``verity.ml.circuits.evaluation`` among them: their sources are inputs, data
they opened at import is not); what a function read or imported in its body when it did not run in the recording (a value a
process-wide cache, ``functools.cache`` and the like, holds from a call another target's check made, beyond the function's own
module and its closure); an edit to a binding that does not reach the target, even if it changes which object of the same id
the catalog builds (the registry refuses two Definitions of one id); what a binding ran that changes, keeping its
id, what a Definition it did not make does at check time without changing its descriptor: a value it computed (from a module
only it imported) and left in a module's state, or closed over, where an evaluator reads it; a value a module only the
registries import leaves in another module's state at import, other than a kernel registration or a Definition (an enumerated
module is held by its name in the listing, not by its content); a module edited after this process imported it and before the
check ran (the process runs the old code, while the reads digested after the check see the new source); behaviour that depends
on a module's name or path through another object (an object's ``__module__``, a code object's ``co_filename``,
``inspect.getfile``: a moved module's objects report their new name; `targets._authored` reads ``__module__`` for the task list
only), and on the order in which modules are first imported (a moved registry module is imported by its importer rather than by
the enumeration, and consecutive imports are taken in the order of the names they bind); and report text that names a module or
a path is the text of the run that stored it.  What runs of a module is what its source compiles to, not its cached bytecode,
which Python takes when its header matches the source's mtime in whole seconds and its size: a module imported while tracing is
compiled from its source, and one imported before from bytecode its source does not compile to makes the process uncacheable
(`_stale_bytecode`).  A new non-empty module in a package the registries enumerate, or one moved where nothing imports it,
reruns everything.  What the tracer cannot see (a program it starts: ``subprocess``, ``os.system``, ``exec``, ``posix_spawn``, a
``fork``) makes the process's later results uncacheable (`untraced`).

Only a verdict is stored, of the tree its check read: not a check that raised (an ImportError says the environment was broken,
not what the target is), nor one whose reads, digested again by the process that ran it right after, differ from the digests
the entry would record (the tree changed during the run: a concurrent run rewrote it), nor one after a module was imported from
outside the tree (another tree's copy, which no recorded read covers).

The task list of ``--all`` (each target and its key) is an entry too, of the request ``task-list``: it holds what any target
could (every binding's recording, every Definition the catalog reached with what its build read, `task_list_reads`), every in-tree
module the process imported but the checker's own (its version) and every binding file whole, and counts only when each is
unchanged where it was, never through a move; a run reuses it only when every target it lists is reused that way too, and then
builds no Definition.

Entries are files ``<dir>/<version>/<sha512 of the key>/<sha512 of the entry>.json``; there is no expiry and no path or time
shortcut.  An entry reused only where its modules moved is stored again, its reads renamed to where they are now (`lookup`).
"""
from __future__ import annotations

import ast
import dis
import hashlib
import importlib
import json
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import entries as E

SCHEMA = "circuit-check-cache/v3"
_ROOT: list[Path] = []
#: each link in an import path directory outside the tree that leads into it, with where it leads (`_links`): a remote run
#: puts such a shim first on the path (``pyshim/<key>/verity -> <tree>/verity``), so the tree's modules have their files there
_LINKS: dict[str, str] = {}
#: what the tracer cannot see: a program started from here, or a repository library loaded by the dynamic loader
UNTRACEABLE = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork")
_UNTRACED: list[str] = []
_INTERNAL: list[bool] = []
_HERE = Path(__file__).resolve().parent
#: the checker's files that are no input of every target: each target holds its own parts of them
TARGETS, PINS = _HERE / "targets.py", _HERE / "pins"
#: the targets' measured peaks `cli.run` schedules by, and beside them the catalog bindings' (by binding name) the circuit_check
#: suite's build schedules by (`units.built_apart`): no input of any target, since they decide only which targets run beside
#: which, and a target whose process the OOM killer took is never stored
HEAVY = _HERE / "heavy.json"
#: the failures `main` already has (`circuit_check.known`): no input of any target, since they decide only the run's summary
#: (`checks.summarize`, over every target's stored report, recomputed each run) and no check imports them
KNOWN = (_HERE / "known.py", _HERE / "known")
#: which targets a ``--shard`` checks and how ``--combine`` joins the shards' records (`circuit_check.shard`): no input of any
#: target, since no check imports it
SHARD = _HERE / "shard.py"
#: a sidecar's name beside its module's (`sidecars`): ``fp8.circuit-check.py`` and ``.json`` of ``fp8.py``
SUFFIX = ".circuit-check"
#: the binding files beside `targets.py` (each Definition module's sidecar, `sidecars`), absolute; the pins files (``pins/``'s
#: and the sidecars'), and where a pin is or would be (`sidecars.pin_reads`)
_SIDECARS: set[str] = set()
_UNIT_RELS: set[str] = set()
_PIN_FILES: set[str] = set()
_PIN_FILE: list = []


class Frame:
    """What one recording saw: reads (relative to the root), the files of the code that ran, the binding files' code that ran
    (file, first line), the specializations whose body it read, the pins it looked up, and the Definitions whose kernels it
    looked up."""

    __slots__ = ("files", "dirs", "probes", "libs", "code", "lines", "defs", "pins", "enums", "named", "missing", "sources",
                 "kernels")

    def __init__(self) -> None:
        self.files: set[str] = set()
        self.dirs: set[str] = set()
        self.probes: set[str] = set()
        self.libs: set[str] = set()
        self.code: set[str] = set()
        self.lines: set[tuple[str, int]] = set()
        self.defs: dict[int, Any] = {}
        self.pins: set[str] = set()
        #: package directories an enumerator (`_ENUMERATORS`) listed, modules imported by a computed name, imports that failed,
        #: and the files `inspect` or `linecache` looked for a module's source at
        self.enums: set[str] = set()
        self.named: set[str] = set()
        self.missing: set[str] = set()
        self.sources: set[str] = set()
        self.kernels: set[str] = set()

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
        self.enums |= o.enums
        self.named |= o.named
        self.missing |= o.missing
        self.sources |= o.sources
        self.kernels |= o.kernels


#: the open recordings, innermost last; the first holds what the process read outside every recording
_FRAMES: list[Frame] = [Frame()]
#: bumped each time a recording opens or closes: a specialization whose body was last read at another count is noted again
_EPOCH = [0]
#: what was read and run while each module was first imported (after `trace`), by module name
_MODULES: dict[str, Frame] = {}
#: what each binding unit read and ran while it made its roots
_PRODUCED: dict[str, Frame] = {}
#: the binding units running now (`producing`), and each Definition made while one ran, with those units
_MAKING: list[tuple[str, ...]] = []
_MADE: dict[int, tuple[Any, frozenset[str]]] = {}


def _sha_file(p: Path) -> str:
    h = hashlib.sha512()
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
    return hashlib.sha512("\0".join(names).encode()).hexdigest()


def _rel(path: Any) -> str | None:
    if isinstance(path, int) or not _ROOT:
        return None
    try:
        p = Path(os.fsdecode(path))
        p = os.path.normpath(p if p.is_absolute() else Path.cwd() / p)
        rel = os.path.relpath(p, _ROOT[0])
        if rel.startswith("..") and _LINKS:            # a file through a link into the tree is the tree's (`_LINKS`)
            rel = next((os.path.relpath(t + p[len(k):], _ROOT[0]) for k, t in _LINKS.items()
                        if p == k or p.startswith(k + os.sep)), rel)
    except (TypeError, ValueError):
        return None
    if rel.startswith("..") or "__pycache__" in rel.split(os.sep) or rel.split(os.sep)[0] == ".venv":
        return None
    return rel


def _links(root: Path) -> dict[str, str]:
    """`_LINKS`: the links in the import path's directories outside ``root`` that lead into it, each with where it leads.  Read
    once, when tracing starts: neither its listings nor its ``lstat``s are a check's reads, and `_rel`, which every ``stat``
    calls, resolves no path itself."""
    out: dict[str, str] = {}
    for e in sys.path:
        d = os.path.abspath(e or os.getcwd())
        if not os.path.relpath(d, root).startswith(".."):
            continue
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            p = os.path.join(d, n)
            if os.path.islink(p) and not os.path.relpath(t := os.path.realpath(p), root).startswith(".."):
                out[p] = t
    return out


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


_SELF = os.path.abspath(__file__)
_IMPORTLIB = ("<frozen importlib._bootstrap>", "<frozen importlib._bootstrap_external>", os.path.abspath(importlib.__file__))
#: the import hooks (`_instrument`) the import system resolves and loads a module for
_HOOKS = ("__import__", "import_module", "exec_module")


#: the read `importlib.metadata` makes of every `sys.path` entry, held by what it yields: each installed distribution's
#: name and version (`_distributions`), not each directory's listing (every workspace member is installed, so
#: a listing would put every member's root in every target's reads, and a file added to one would make every target miss)
DISTRIBUTIONS = "<distributions>"
_METADATA: list[str] = []


def _metadata_listing() -> bool:
    """Whether the audited listing is `importlib.metadata` searching `sys.path` for distributions."""
    if not _METADATA:
        import importlib.metadata as md

        _METADATA.append(os.path.dirname(md.__file__))
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename.startswith(_METADATA[0]):
            return True
        f = f.f_back
    return False


def _distributions() -> str:
    """The installed distributions as one digest: each one's name and version, sorted."""
    import importlib.metadata as md

    _INTERNAL.append(True)
    try:
        names = sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata['Name']})
    finally:
        _INTERNAL.pop()
    return hashlib.sha512("\0".join(names).encode()).hexdigest()


def _importing() -> bool:
    """Whether the audited call is the import system finding or loading a module for one of the import hooks: a listing it
    searched or a source it read, which the module's name and content stand for (`_modules`)."""
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename in _IMPORTLIB:
        f = f.f_back
    return f is not None and f.f_code.co_filename == _SELF and f.f_code.co_name in _HOOKS


#: the functions (module, qualified name) that list a package with ``pkgutil`` only to import each module it lists, naming any
#: that fails: what they list and import is held by the package's modules (`_children`); one moved is no longer one of them
_ENUMERATORS = frozenset({("verity.ml.definitions.registry.catalog", "load"), ("verity.ml.definitions.boolean", "_load")})


def _enumerator(f) -> bool:
    return f is not None and ((f.f_code.co_filename == _TARGETS_FILE and f.f_code.co_name == "load_registries")
                              or (f.f_globals.get("__name__"), f.f_code.co_qualname) in _ENUMERATORS)


def _enumerating() -> bool:
    """Whether the audited listing is ``pkgutil`` enumerating a package for `targets.load_registries` or another of the
    `_ENUMERATORS`."""
    import pkgutil

    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename == pkgutil.__file__:
        f = f.f_back
    return _enumerator(f)


_PATHLIB: list[tuple[str, ...]] = []


def _path_frames() -> list[tuple[str, ...]]:
    """The files of the path functions a probe passes through (`os.path`, `pathlib`), and of `inspect` and `linecache`."""
    if not _PATHLIB:
        import genericpath
        import inspect
        import linecache
        import pathlib
        import posixpath

        _PATHLIB.append(tuple({m.__file__ for m in (genericpath, posixpath, os, pathlib)}
                              | {"<frozen genericpath>", "<frozen posixpath>", "<frozen os>"}))
        _PATHLIB.append((inspect.__file__, linecache.__file__))
    return _PATHLIB


def _sourcing() -> bool:
    """Whether the probe is `inspect` or `linecache` looking for a module's source file (torch's import asks `inspect` for the
    module of a frame, which tries the file of every module loaded): a question about the module, held by its name."""
    paths, f = _path_frames(), sys._getframe(2)
    while f is not None and f.f_code.co_filename in paths[0]:
        f = f.f_back
    return f is not None and f.f_code.co_filename in paths[1]


def _listing_probe() -> bool:
    """Whether the probe is ``pkgutil`` telling a package directory from another entry of a package an enumerator lists
    (`_enumerating`; a sidecar's pins file among them): no input, since what the listing yields is held by the package's
    modules (`_children`)."""
    import pkgutil

    paths, f = _path_frames(), sys._getframe(2)
    while f is not None and f.f_code.co_filename in paths[0]:
        f = f.f_back
    if f is None or f.f_code.co_filename != pkgutil.__file__:
        return False
    while f is not None and f.f_code.co_filename == pkgutil.__file__:
        f = f.f_back
    return _enumerator(f)


def _hook(event: str, args: tuple) -> None:
    if _INTERNAL and event in ("open", "os.listdir", "os.scandir"):
        return                                         # the tracer's own reads (digesting, parsing): no input of a check
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else "r"
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return
        if isinstance(mode, str) or mode is None:
            if (rel := _rel(path)) is not None and not ((rel == _TARGETS_REL[0] or rel in _UNIT_RELS) and _for_linecache()) \
                    and not _importing():
                _FRAMES[-1].files.add(rel)
    elif event in ("os.listdir", "os.scandir"):
        if (rel := _rel(args[0] if args and args[0] is not None else ".")) is not None and not _importing():
            if _metadata_listing():
                _FRAMES[-1].dirs.add(DISTRIBUTIONS)
            else:
                (_FRAMES[-1].enums if _enumerating() else _FRAMES[-1].dirs).add(rel)
    elif event in UNTRACEABLE and not _INTERNAL:
        argv = args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else args[:1]
        _UNTRACED.append(f"{event} {' '.join(map(str, argv))[:240]}")
    elif event == "ctypes.dlopen" and args and args[0] and (rel := _rel(args[0])) is not None:
        _FRAMES[-1].libs.add(rel)                      # a library built in the tree: an input by its bytes (`_digests`)


def _probing(fn):
    def stat(path, *a, **k):
        if not _INTERNAL and (rel := _rel(path)) is not None and not _listing_probe():
            (_FRAMES[-1].sources if rel.endswith(".py") and _sourcing() else _FRAMES[-1].probes).add(rel)
        return fn(path, *a, **k)
    return stat


_TOOL: list[int] = []
_TARGETS_FILE = str(TARGETS)


def _started(code, offset):
    f = code.co_filename
    if f == _TARGETS_FILE or f in _SIDECARS:
        _FRAMES[-1].lines.add((f, code.co_firstlineno))
    else:
        _FRAMES[-1].code.add(f)
    return sys.monitoring.DISABLE


def _unit_files() -> set[str]:
    return {_TARGETS_FILE} | _SIDECARS


def unit_files(paths: Iterable[str]) -> None:
    """Hold these binding files (the sidecars) as `targets.py` is held: by the units of theirs a target depends on."""
    _SIDECARS.update(os.path.abspath(p) for p in paths)
    _UNIT_RELS.clear()
    _UNIT_RELS.update(r for f in _SIDECARS if (r := _rel(f)) is not None)


_REGISTRIES: list[str] = []


def registries(modules: Iterable[str]) -> None:
    """The modules the registries import for having a sidecar (`targets.load_registries`), in every target's key."""
    _REGISTRIES[:] = sorted(modules)


def pin_files(paths: Iterable[str], where: Any) -> None:
    """The pins files (``pins/``'s and the sidecars'), and ``where(kind, key)``: the files a lookup of a pin read
    (`sidecars.pin_reads`)."""
    _PIN_FILES.clear()
    _PIN_FILES.update(os.path.abspath(p) for p in paths)
    _PIN_FILE[:] = [where]


@contextmanager
def internal() -> Iterator[None]:
    """The checker's own reads (finding and reading the sidecars): no input of a check."""
    _INTERNAL.append(True)
    try:
        yield
    finally:
        _INTERNAL.pop()


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
    _EPOCH[0] += 1
    if _TOOL:
        sys.monitoring.restart_events()
    try:
        yield fr
    finally:
        _FRAMES.pop()
        _EPOCH[0] += 1
        if merge and len(_FRAMES) > 1:
            _FRAMES[-1].merge(fr)


def _imported(mods: list) -> None:
    for m in mods:
        if (f := getattr(m, "__file__", None)) and isinstance(f, str) and isinstance(m, type(sys)):
            _FRAMES[-1].code.add(f)


_IMPORT_NAME = dis.opmap["IMPORT_NAME"]


def _statement(f) -> bool:
    """Whether frame ``f`` is executing an ``import`` statement (rather than calling ``__import__`` with a computed name)."""
    try:
        return f.f_code.co_code[f.f_lasti] == _IMPORT_NAME
    except (IndexError, AttributeError):
        return False


def _absolute(name: str, globals: dict | None, level: int) -> str:
    if not level:
        return name
    pkg = (globals or {}).get("__package__")
    if pkg is None and globals and "__name__" in globals:
        pkg = globals["__name__"] if "__path__" in globals else globals["__name__"].rpartition(".")[0]
    return _relative(pkg or "", level, name) or name


def _instrument() -> None:
    """The loader hook (what each module's import read and ran, `_MODULES`), and the Definition hooks: each body's build
    recorded on its specialization, each body read noted in the recording that read it."""
    import importlib.machinery as M

    exec0 = M.SourceFileLoader.exec_module

    def exec_module(self, module):
        with recording() as fr:
            path = getattr(module, "__file__", None)
            if isinstance(path, str) and _rel(path) is not None:
                # compiled from the source the key digests, not its cached bytecode (`_stale_bytecode`)
                exec(self.source_to_code(self.get_data(path), path), module.__dict__)
            else:
                exec0(self, module)
        _MODULES[module.__name__] = fr

    M.SourceFileLoader.exec_module = exec_module
    import builtins
    import importlib

    imp0, by_name0 = builtins.__import__, importlib.import_module

    def __import__(name, globals=None, locals=None, fromlist=(), level=0):
        f = sys._getframe(1)
        try:
            m = imp0(name, globals, locals, fromlist, level)
        except ImportError:
            if not _INTERNAL:
                base = _absolute(name, globals, level)
                _FRAMES[-1].missing.update([base, *(f"{base}.{a}" for a in fromlist or () if a != "*")])
            raise
        if not _INTERNAL:
            if not _statement(f):                      # a name computed at run time: it does not follow a move
                _FRAMES[-1].named.add(_absolute(name, globals, level))
            if len(_FRAMES) > 1:
                # an import of a module already loaded runs nothing of it, yet what it registered may decide the check
                _imported([m, *(getattr(m, a, None) for a in fromlist or ())] if fromlist or level else [sys.modules.get(name)])
        return m

    def import_module(name, package=None):
        f = sys._getframe(1)
        try:
            m = by_name0(name, package)
        except ImportError:
            if not _INTERNAL:
                bare = name.lstrip(".")
                _FRAMES[-1].missing.add(_absolute(bare, {"__package__": package}, len(name) - len(bare)))
            raise
        if not _INTERNAL:
            if not _enumerator(f):
                _FRAMES[-1].named.add(m.__name__)
            if len(_FRAMES) > 1:
                _imported([m])
        return m

    builtins.__import__, importlib.import_module = __import__, import_module
    try:
        from verity.ml.circuits import defs as D
    except ImportError:
        return
    try:
        from verity.ml.circuits.evaluation import batch
    except ImportError:
        batch = None
    if batch is not None and not isinstance(batch._KERNELS, _RecordedKernels):
        batch._KERNELS = _RecordedKernels(batch._KERNELS)
    build0 = D._build_body

    def _build_body(sp):
        # not merged: a recording that reads the body holds the specialization (`body`), and with it this build
        with recording(merge=False) as fr:
            body = build0(sp)
        sp._circuit_check_built = fr
        return body

    D._build_body = _build_body

    def marking(cls) -> None:
        init0 = cls.__init__

        def __init__(self, *a, **k):
            init0(self, *a, **k)
            units = frozenset(u for us in _MAKING for u in us) | made_by(getattr(self, "defn", None))
            if units:
                _MADE[id(self)] = (self, units)

        cls.__init__ = __init__

    for cls in (D.PrimitiveDefinition, D.CompositeDefinition, D.SpecializedDefinition):
        marking(cls)
    get = D.SpecializedDefinition.body.fget
    D.SpecializedDefinition._circuit_check_read = -1

    def body(self):
        # the layout's walks read a body millions of times a check: noted once under each innermost recording
        if self._circuit_check_read == _EPOCH[0] and (b := self._body) is not None:
            return b
        if len(_FRAMES) > 1:
            _FRAMES[-1].defs.setdefault(id(self), self)
        b = get(self)
        self._circuit_check_read = _EPOCH[0]
        return b

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
        _LINKS.update(_links(_ROOT[0]))
        _TARGETS_REL[0] = _rel(TARGETS)
        unit_files(())
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
        _UNTRACED.extend(f"{rel}: imported before tracing from bytecode that is not its source's" for rel in _stale_bytecode())
        _instrument()


def _stale_bytecode() -> list[str]:
    """The tree's modules imported before tracing from cached bytecode that is not what their source compiles to.  Python takes
    a module's bytecode when its header matches the source's mtime, in whole seconds, and its size (`_taken`), so an edit of the
    same size within the second the bytecode was written runs the old code while the key digests the new source; bytecode whose
    header does not match was not run, and a module imported while tracing is compiled from its source (`_instrument`)."""
    import marshal

    out = []
    _INTERNAL.append(True)
    try:
        for m in list(sys.modules.values()):
            spec = getattr(m, "__spec__", None)
            path, cached = getattr(spec, "origin", None), getattr(spec, "cached", None)
            if not (isinstance(path, str) and path.endswith(".py") and cached and (rel := _rel(path)) is not None):
                continue
            try:
                with open(cached, "rb") as fh:
                    head, body = fh.read(16), fh.read()
                with open(path, "rb") as fh:
                    st, source = os.fstat(fh.fileno()), fh.read()
                if not _taken(head, st, source):
                    continue                           # Python compiled this one from its source
                code = marshal.loads(body)
            except (OSError, EOFError, ValueError, TypeError):
                continue                               # none, or one Python would not take either
            if code != compile(source, path, "exec", dont_inherit=True):
                out.append(rel)
    finally:
        _INTERNAL.pop()
    return out


def _taken(head: bytes, st: os.stat_result, source: bytes) -> bool:
    """Whether the import system takes bytecode with this header for this source (`importlib._bootstrap_external`'s rule)."""
    import _imp
    import importlib.util

    if len(head) < 16 or head[:4] != importlib.util.MAGIC_NUMBER:
        return False
    flags = int.from_bytes(head[4:8], "little")
    if not flags & 1:
        return (head[8:12] == (int(st.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
                and head[12:16] == (st.st_size & 0xFFFFFFFF).to_bytes(4, "little"))
    mode = _imp.check_hash_based_pycs
    if mode == "never" or (mode == "default" and not flags & 2):
        return True
    return head[8:16] == importlib.util.source_hash(source)


def tracing() -> bool:
    return bool(_ROOT)


def untraced() -> list[str]:
    """What this process did that the tracer cannot see (a verdict after it is not stored)."""
    return list(_UNTRACED)


def produced(units: Iterable[str], fr: Frame) -> None:
    """Hold ``fr`` (what a binding read and ran while it made its roots) under each of its ``units``."""
    for u in units:
        _PRODUCED.setdefault(u, Frame()).merge(fr)


@contextmanager
def producing(units: tuple[str, ...]) -> Iterator[Frame]:
    """Run a binding's ``units``: what it reads and runs is held under them (`produced`), and every Definition made meanwhile
    (a primitive, a composite, a specialization, or one of a composite made so) is marked as theirs (`made_by`)."""
    with recording() as fr:
        _MAKING.append(tuple(units))
        try:
            yield fr
        finally:
            _MAKING.pop()
    produced(units, fr)


def made_by(d: Any) -> frozenset[str]:
    """The binding units that were running when ``d`` was made (empty when none was)."""
    e = _MADE.get(id(d))
    return e[1] if e is not None and e[0] is d else frozenset()


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


# --- modules by name and content --------------------------------------------------------------------------------------------

#: names whose value is the module's own path or name: a module that reads one is held at its path
_ANCHORS = frozenset({"__file__", "__name__", "__path__", "__spec__", "__loader__", "__cached__", "__package__"})


class _Norm(ast.NodeTransformer):
    """A module's AST with every ``from`` import and every ``import M as N`` cut to the names it binds (one statement per name,
    so splitting or joining import statements changes nothing), and those imports' sites in order.  ``import a.b`` binds ``a``
    and its uses spell ``a.b``, so it stays as it is."""

    def __init__(self) -> None:
        self.sites: list[tuple[int, str, str | None]] = []
        self.anchored = False

    def visit_ImportFrom(self, node: ast.ImportFrom) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for a in node.names:
            self.sites.append((node.level, node.module or "", a.name))
            out.append(ast.Expr(ast.Constant(f"<from-import {a.name} as {a.asname}>")))
        return out

    def visit_Import(self, node: ast.Import) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for a in node.names:
            if a.asname:
                self.sites.append((-1, a.name, None))
                out.append(ast.Expr(ast.Constant(f"<import as {a.asname}>")))
            else:
                parts = a.name.split(".")
                self.sites += [(-1, ".".join(parts[:i]), None) for i in range(1, len(parts) + 1)]
                out.append(ast.Import(names=[ast.alias(name=a.name)]))
        return out

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if (isinstance(node.left, ast.Name) and node.left.id == "__name__" and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq) and isinstance(c := node.comparators[0], ast.Constant)
                and c.value == "__main__"):
            return node                                # false under every name a module is imported by
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _ANCHORS:
            self.anchored = True
        return node


def _bound(s: ast.Import | ast.ImportFrom) -> str:
    a = s.names[0]
    return a.asname or (a.name if isinstance(s, ast.ImportFrom) else a.name.partition(".")[0])


def _commuted(tree: ast.AST) -> ast.AST:
    """``tree`` with each run of consecutive import statements split one name each and ordered by the name each binds, stably
    (of two that bind one name the later still wins; ``import *`` ends a run): imports of different names commute but for the
    order modules are first imported in, which no key holds."""
    for node in ast.walk(tree):
        for field, body in ast.iter_fields(node):
            if not (isinstance(body, list) and body and all(isinstance(s, ast.stmt) for s in body)):
                continue
            if field == "body" and isinstance(node, (ast.Try, ast.TryStar, ast.With, ast.AsyncWith)):
                continue
            out: list[ast.stmt] = []
            run: list[ast.Import | ast.ImportFrom] = []
            for s in [*body, None]:
                if (isinstance(s, (ast.Import, ast.ImportFrom)) and all(a.name != "*" for a in s.names)
                        and not (isinstance(s, ast.ImportFrom) and s.module == "__future__")):
                    run += [ast.ImportFrom(module=s.module, names=[a], level=s.level) if isinstance(s, ast.ImportFrom)
                            else ast.Import(names=[a]) for a in s.names]
                    continue
                out += sorted(run, key=_bound)
                run = []
                if s is not None:
                    out.append(s)
            setattr(node, field, out)
    return tree


def _normalized(node: ast.AST) -> tuple[str, list[tuple[int, str, str | None]], bool]:
    """(digest of ``node`` with its imports cut to the names they bind and each run of them ordered by those names, `_commuted`,
    the imports' sites, whether it reads its own path or name).  ``node`` is copied, not changed."""
    import copy

    n = _Norm()
    out = n.visit(_commuted(copy.deepcopy(node)))
    return hashlib.sha512(ast.dump(out, include_attributes=False).encode()).hexdigest(), n.sites, n.anchored


_PARSED: dict[str, tuple[str, list, bool, bool]] = {}
#: each source file as read since the process started, or since the last digest made afresh (`_digests`)
_PATHS: dict[str, tuple[str, list, bool, bool]] = {}


def _parsed(p: Path) -> tuple[str, list, bool, bool]:
    """A source file's (content digest, import sites, anchored, trivial), by its bytes: `_normalized` of its module, and
    whether it is empty or a docstring alone.  A file that does not parse is held by its bytes, at its path."""
    if (k := str(p)) not in _PATHS:
        _PATHS[k] = _parsed_bytes(p.read_bytes())
    return _PATHS[k]


def _parsed_bytes(data: bytes) -> tuple[str, list, bool, bool]:
    h = hashlib.sha512(data).hexdigest()
    if h not in _PARSED:
        try:
            tree = ast.parse(data)
        except (SyntaxError, ValueError):
            _PARSED[h] = (f"bytes:{h}", [], True, False)
        else:
            body = tree.body
            trivial = not body or (len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                                   and isinstance(body[0].value.value, str))
            _PARSED[h] = (*_normalized(tree), trivial)
    return _PARSED[h]


def _roots(root: Path) -> list[str]:
    """The import path's entries inside the tree (``sys.path`` order, relative to ``root``), where module names resolve."""
    out: list[str] = []
    for e in sys.path:
        rel = os.path.relpath(os.path.realpath(e or os.getcwd()), root)
        if not rel.startswith("..") and rel.split(os.sep)[0] != ".venv" and rel not in out:
            out.append(rel)
    return out


def _redirected(root: Path) -> dict[str, str]:
    """The top-level packages an exact-mode editable install serves from the tree, each by its source file relative to ``root``:
    `editables`' finder, which hatch's `dev-mode-exact` puts on ``sys.meta_path`` after the path-based one, maps each name to a
    file, so no ``sys.path`` entry finds them (the tree's root, where core's `verity` is, never is one)."""
    out: dict[str, str] = {}
    for finder in sys.meta_path:
        if isinstance(red := getattr(finder, "_redirections", None), dict):
            for name, p in red.items():
                rel = os.path.relpath(os.path.realpath(p), root) if isinstance(p, str) else ".."
                if not rel.startswith("..") and rel.split(os.sep)[0] != ".venv":
                    out.setdefault(name, rel)
    return out


def _resolve(root: Path, name: str) -> Path | None:
    """The tree's source file module ``name`` imports from (a package's ``__init__.py``), as the path-based finder searches
    the in-tree import path: a regular package or a module in the first entry that has one, else a namespace package's
    portions, else, for a top-level name, the file an editable redirect (`_redirected`) maps it to; None when it resolves to
    no visible file (absent, a namespace package, or outside the tree)."""
    v = visible(root)
    isfile = (lambda r: r in v[0] and (root / r).is_file()) if v is not None else (lambda r: (root / r).is_file())
    isdir = (lambda r: r in v[1] and (root / r).is_dir()) if v is not None else (lambda r: (root / r).is_dir())
    paths, found, top = _roots(root), None, True
    for part in name.split("."):
        if not part.isidentifier():
            return None
        found, portions, nxt = None, [], []
        for d in paths:
            pk = os.path.normpath(os.path.join(d, part))
            if isfile(os.path.join(pk, "__init__.py")):
                found, nxt = os.path.join(pk, "__init__.py"), [pk]
                break
            if isfile(pk + ".py"):
                found = pk + ".py"
                break
            if isdir(pk):
                portions.append(pk)
        if found is None and not portions and top and isfile(r := _redirected(root).get(part, "")):
            found, nxt = r, [os.path.dirname(r)] if os.path.basename(r) == "__init__.py" else []
        if found is None:
            if not portions:
                return None
            nxt = portions
        paths, top = nxt, False
    return root / found if found is not None and (root / found).is_file() else None


def _relative(pkg: str, level: int, module: str) -> str | None:
    parts = pkg.split(".") if pkg else []
    if level - 1 > len(parts) or (not parts and level):
        return None
    base = parts[:len(parts) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def _sites(root: Path, package: str, raw: list[tuple[int, str, str | None]]) -> list[list]:
    """Each import site's target module, absolute, as it resolves now from ``package`` (``from P import m`` names ``P.m`` when
    that is a module, else ``P``), with the target's content digest (None outside the tree)."""
    out = []
    for level, module, alias in raw:
        if level < 0:
            t = module
        else:
            base = module if level == 0 else _relative(package, level, module)
            if base is None:
                t = "." * level + module
            elif alias == "*":
                t = base
            else:
                sub = f"{base}.{alias}" if base else alias
                t = sub if _resolve(root, sub) is not None else base
        f = _resolve(root, t) if t and not t.startswith(".") else None
        out.append([t, _parsed(f)[0] if f is not None else None])
    return out


def _modrec(root: Path, name: str) -> dict | None:
    """Module ``name`` as it is now: its content digest (`_parsed`: comments, formatting and the spelling of its imports aside),
    what each of its import sites resolves to (`_sites`), and, when it reads its own path or name, its path.  None when the
    name resolves to no file in the tree."""
    f = _resolve(root, name)
    if f is None:
        return None
    d, raw, anchored, _trivial = _parsed(f)
    rec: dict[str, Any] = {"d": d, "sites": _sites(root, name if f.name == "__init__.py" else name.rpartition(".")[0], raw)}
    if anchored:
        rec["at"] = os.path.relpath(f, root)
    return rec


def _local(root: Path, name: str) -> str | None:
    """Module ``name``'s content digest, or a sidecar's (``<module>.circuit-check``, `_side`) where its module is now; None when
    there is no such file in the tree."""
    if (m := _side(name)) is None:
        f = _resolve(root, name)
    else:
        f = root / r if (r := _sidecar(root, m, ".py")) is not None and _seen(root)(r) and (root / r).is_file() else None
    return None if f is None else _parsed(f)[0]


def _children(root: Path, pkg: str) -> dict[str, str] | None:
    """The modules ``pkgutil`` lists in package ``pkg`` (``x.py`` or ``x/__init__.py``, visible), each by its content digest."""
    f = _resolve(root, pkg)
    if f is None or f.name != "__init__.py":
        return None
    d = f.parent
    v = visible(root)
    rel = os.path.relpath(d, root)
    ok = (lambda r: r in v[0] and (root / r).is_file()) if v is not None else (lambda r: (root / r).is_file())
    out = {}
    for n in sorted(os.listdir(d)):
        if n.endswith(".py") and n[:-3].isidentifier() and n != "__init__.py" and ok(os.path.join(rel, n)):
            out[n[:-3]] = _parsed(d / n)[0]
        elif n.isidentifier() and ok(os.path.join(rel, n, "__init__.py")):
            out[n] = _parsed(d / n / "__init__.py")[0]
    return out


_LOADED: list[dict[str, list[str]]] = []


def _loaded(root: Path) -> dict[str, list[str]]:
    """The tree's modules this process has imported, by content digest: in the process that looks entries up, before any
    check, what the task list's build imported, which every process that checks a target imports in the same way (the
    sidecars' binding files among them, by their names, `_side`)."""
    if not _LOADED:
        ix: dict[str, list[str]] = {}
        for name, m in list(sys.modules.items()):
            f = getattr(m, "__file__", None)
            if not (isinstance(f, str) and _rel(f) is not None):
                continue
            side = _side(name)
            p = _resolve(root, name) if side is None else root / r if (r := _sidecar(root, side, ".py")) is not None else None
            if p is not None and os.path.realpath(p) == os.path.realpath(f):
                ix.setdefault(_parsed(p)[0], []).append(name)
        _LOADED.append(ix)
    return _LOADED[0]

# --- the kernels a check looked up ------------------------------------------------------------------------------------------

class _RecordedKernels(dict):
    """`verity.ml.circuits.evaluation.batch._KERNELS`, noting in the open recording each Definition whose kernels are looked up (a lookup
    that finds none included), and all of them on any enumeration.  Registering (``setdefault``) is no lookup; it notes the
    in-tree module whose import registered each key (`by`), which a lookup of that key adds to the code the recording ran."""

    def __init__(self, *a: Any) -> None:
        super().__init__(*a)
        self.by: dict[str, set[str]] = {}

    def setdefault(self, key, default=None):
        f = sys._getframe(1)
        while f is not None and not (f.f_code.co_name == "<module>" and _rel(f.f_code.co_filename) is not None):
            f = f.f_back
        if f is not None:
            self.by.setdefault(str(key), set()).add(f.f_code.co_filename)
        return dict.setdefault(self, key, default)

    def _note(self, key: Any) -> None:
        if len(_FRAMES) > 1 and not _INTERNAL:
            _FRAMES[-1].kernels.add(str(key))
            for k, files in self.by.items():
                if key in ("*", k):
                    _FRAMES[-1].code.update(files)

    def get(self, key, default=None):
        self._note(key)
        return dict.get(self, key, default)

    def __getitem__(self, key):
        self._note(key)
        return dict.__getitem__(self, key)

    def __contains__(self, key) -> bool:
        self._note(key)
        return dict.__contains__(self, key)

    def __iter__(self):
        self._note("*")
        return dict.__iter__(self)

    def keys(self):
        self._note("*")
        return dict.keys(self)

    def values(self):
        self._note("*")
        return dict.values(self)

    def items(self):
        self._note("*")
        return dict.items(self)


def _where(fn: Any) -> str | None:
    """Which function ``fn`` is, wherever its module is: for one defined in a tracked file of the tree, that module's content
    digest (`_parsed`), its qualified name and the digest of its own definition (`_defined_at`: its imports cut as a module's
    are, so a pure move changes none of them); for one defined in an untracked file, its path and first line; outside the tree,
    its file and first line; for a callable that is no function, its type's name."""
    import functools
    import inspect

    if fn is None:
        return None
    while isinstance(fn, functools.partial):
        fn = fn.func
    fn = inspect.unwrap(fn)
    code = getattr(fn, "__code__", None) or getattr(getattr(fn, "__call__", None), "__code__", None)
    if code is None:
        return f"{type(fn).__module__}.{type(fn).__qualname__}"
    rel = _rel(code.co_filename)
    if rel is None or not _ROOT:
        return f"{code.co_filename}:{code.co_firstlineno}"
    p, v = _ROOT[0] / rel, visible(_ROOT[0])
    if (v is not None and rel not in v[0]) or not p.is_file():
        return f"untracked:{rel}:{code.co_firstlineno}"
    return f"{_parsed(p)[0]}:{code.co_qualname}:{_defined_at(p, code.co_firstlineno)}"


_TREES: dict[str, ast.Module | None] = {}


def _defined_at(p: Path, line: int) -> str:
    """The digest (`_normalized`) of the function, lambda or class definitions in source file ``p`` whose code starts at
    ``line`` (a decorated one at its first decorator), or the line itself when none does."""
    data = p.read_bytes()
    h = hashlib.sha512(data).hexdigest()
    if h not in _TREES:
        try:
            _TREES[h] = ast.parse(data)
        except (SyntaxError, ValueError):
            _TREES[h] = None
    tree = _TREES[h]
    defs = ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda
    nodes = [] if tree is None else [
        n for n in ast.walk(tree) if isinstance(n, defs)
        and min([n.lineno, *(d.lineno for d in getattr(n, "decorator_list", ()))]) == line]
    if not nodes:
        return f"line:{line}"
    return hashlib.sha512("".join(_normalized(n)[0] for n in nodes).encode()).hexdigest()


def _kernels_now(key: str) -> str | None:
    """What is registered now for ``key`` (``*``: for every Definition), digested; None when nothing is."""
    try:
        from verity.ml.circuits.evaluation import batch
    except ImportError:
        return None
    reg = batch._KERNELS
    keys = sorted(dict.keys(reg)) if key == "*" else [key]
    got = {k: {n: [kn.domain, kn.version, _where(kn.fn), _where(kn.sample)] for n, kn in sorted(dict.get(reg, k, {}).items())}
           for k in keys}
    got = {k: v for k, v in got.items() if v}
    return hashlib.sha512(json.dumps(got, sort_keys=True).encode()).hexdigest() if got else None


# --- a binding file's units -------------------------------------------------------------------------------------------------

def _side(read: str) -> str | None:
    """The module whose sidecar a unit or pin read (``<module>.circuit-check#...``) is in; None for one of `targets.py` or
    ``pins/``, named by its path."""
    head = read.partition("#")[0]
    return head[:-len(SUFFIX)] if head.endswith(SUFFIX) and "/" not in head else None


def _sidecar(root: Path, module: str, ext: str) -> str | None:
    """Where module ``module``'s sidecar of ``ext`` (``.py``, ``.json``) is now, or would be, relative to ``root`` (as
    `sidecars.discover` finds it); None when the name resolves to no file of the tree."""
    f = _resolve(root, module)
    if f is None:
        return None
    if f.name != "__init__.py":
        return os.path.relpath(f.with_name(f"{f.stem}{SUFFIX}{ext}"), root)
    inside, beside = f.parent / f"__init__{SUFFIX}{ext}", f.parent.with_name(f"{f.parent.name}{SUFFIX}{ext}")
    return os.path.relpath(beside if beside.is_file() and not inside.is_file() else inside, root)


def _named_file(f: str, ext: str, by_file: dict[str, str]) -> str | None:
    """How a read of binding or pins file ``f`` names it: `targets.py` and ``pins/``'s by their paths; a sidecar by its module's
    name (`SUFFIX`), when the tree resolves that name to the module beside it, else by its path."""
    if (rel := _rel(f)) is None:
        return None
    p = Path(f)
    if p == TARGETS or PINS in p.parents or not p.name.endswith(SUFFIX + ext):
        return rel
    stem = p.name[:-len(SUFFIX + ext)]
    beside = [p.parent / "__init__.py"] if stem == "__init__" else [p.with_name(f"{stem}.py"), p.parent / stem / "__init__.py"]
    for m in beside:
        name = by_file.get(os.path.abspath(m))
        if name is not None and (r := _resolve(_ROOT[0], name)) is not None and os.path.realpath(r) == os.path.realpath(m):
            return f"{name}{SUFFIX}"
    return rel


def _is_binding(name: str) -> bool:
    return name.startswith("_") and name.endswith(("_roots", "_bindings"))


def _digest_ast(node: ast.AST, sites: dict[str, list] | None = None, unit: str = "") -> str:
    """A unit's digest (`_normalized`: its imports cut to the names they bind), its import sites noted in ``sites``."""
    d, raw, _anchored = _normalized(node)
    if sites is not None:
        sites[unit] = raw
    return d


def _model_list(fn: ast.FunctionDef) -> list[ast.expr] | None:
    """The entries of `_model_roots`' returned list, when every one is a ``(label, make)`` tuple and no two share a label (each
    entry's unit is named by its label, so two of one label are one unit, and the whole function is that unit then)."""
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if len(rets) != 1 or not isinstance(rets[0].value, ast.List):
        return None
    elts = rets[0].value.elts
    if all(isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant) and isinstance(e.elts[0].value, str)
           for e in elts) and len({e.elts[0].value for e in elts}) == len(elts):
        return elts
    return None


def _with_list(fn: ast.FunctionDef, keep: list[ast.expr]) -> ast.FunctionDef:
    import copy

    fn = copy.deepcopy(fn)
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    ret.value.elts = copy.deepcopy(keep)
    return fn


def _units_of(source: str) -> tuple[dict[str, str], list[tuple[int, int, str | list[tuple[int, int, str]]]], dict[str, list]]:
    """A binding file's units (``targets.py``'s, a sidecar's) by name, each the digest of its AST (its imports cut to the names
    they bind, `_normalized`):
    every binding function (`_is_binding`), each entry of `_model_roots` (the function with its list cut to that entry) and of
    ``TEMPLATE_PARAMS``, and ``common``: the module with every one of those cut out (a binding function to its name, the list
    and the dict emptied).  Beside them the line span of each top-level statement and the unit that holds it (for
    `_model_roots`, the spans of its entries), and each unit's import sites."""
    tree = ast.parse(source)
    units: dict[str, str] = {}
    sites: dict[str, list] = {}
    spans: list[tuple[int, int, Any]] = []
    body: list[ast.stmt] = []
    for node in tree.body:
        first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        span = (first, node.end_lineno or node.lineno)
        if isinstance(node, ast.FunctionDef) and _is_binding(node.name):
            elts = _model_list(node) if node.name == "_model_roots" else None
            if elts is None:
                units[node.name] = _digest_ast(node, sites, node.name)
                spans.append((*span, node.name))
                body.append(ast.Expr(ast.Name(node.name)))
            else:
                inner = []
                for e in elts:
                    u = f"{node.name}:{e.elts[0].value}"
                    units[u] = _digest_ast(_with_list(node, [e]), sites, u)
                    inner.append((e.lineno, e.end_lineno or e.lineno, u))
                spans.append((*span, inner))
                body.append(_with_list(node, []))
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else \
            node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == "TEMPLATE_PARAMS" and isinstance(node.value, ast.Dict) \
                and all(isinstance(k, ast.Constant) for k in node.value.keys):
            for k, v in zip(node.value.keys, node.value.values):
                units[f"TEMPLATE_PARAMS:{k.value}"] = _digest_ast(v, sites, f"TEMPLATE_PARAMS:{k.value}")
            spans.append((*span, "*"))
            body.append(ast.Expr(ast.Name("TEMPLATE_PARAMS")))
            continue
        spans.append((*span, "common"))
        body.append(node)
    units["common"] = _digest_ast(ast.Module(body=body, type_ignores=[]), sites, "common")
    return units, spans, sites


_UNITS: dict[str, tuple[dict[str, str], list, dict[str, list]]] = {}


def _units_cached(text: str) -> tuple[dict[str, str], list, dict[str, list]]:
    h = hashlib.sha512(text.encode()).hexdigest()
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
    """What a target's check read or resolved: its recording ``fr``; for every specialization it read or reaches from
    ``roots``, its build's recording and the recording of each binding that made it (`made_by`; without ``roots``, those of
    all its binding ``units``); each module among them with its import closure; its ``units`` by digest; and what the process
    read outside every recording."""
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
    if not roots:                                      # a template, or a named target that doesn't resolve: its whole bindings
        for u in units:
            if u in _PRODUCED:
                acc.merge(_PRODUCED[u], defs=False)
    outside = _FRAMES[0]
    # an import's own reads count with its module's closure, and the task list's reads of the pins and the catalog with what
    # each target looks up in them
    sources = {r for r in (_rel(f) for f in [*by_file, *_PIN_FILES, *_unit_files()]) if r is not None}
    acc.files |= outside.files - sources
    acc.dirs |= outside.dirs
    acc.probes |= outside.probes
    acc.libs |= outside.libs
    acc.enums |= outside.enums
    acc.named |= outside.named
    acc.missing |= outside.missing
    acc.sources |= outside.sources
    acc.kernels |= outside.kernels
    root = _ROOT[0]
    mods: set[str] = set()

    def module(f: str, rel: str) -> str | None:
        """The name file ``f`` is imported by, when the tree resolves that name to it; else it is held by its path."""
        name = by_file.get(f)
        if name is not None and (p := _resolve(root, name)) is not None and os.path.realpath(p) == os.path.realpath(f):
            mods.add(name)
        else:
            acc.files.add(rel)
        return name

    seen_defs: set[int] = set()
    seen_units: set[str] = set()
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
            for u in made_by(d):                       # what made it ran: not every Definition its binding's roots reach
                if u in _PRODUCED and u not in seen_units:
                    seen_units.add(u)
                    take(_PRODUCED[u], defs=False)
            built = getattr(d, "_circuit_check_built", None)
            if built is not None:
                take(built)
            work.extend(_evaluators(d))
            body = getattr(d, "_body", None)
            if body is not None:
                stack.extend(n.fn for n in body.nodes)

    def take(f: Frame, defs: bool = True) -> None:
        acc.files |= f.files
        acc.dirs |= f.dirs
        acc.probes |= f.probes
        acc.libs |= f.libs
        acc.lines |= f.lines
        acc.pins |= f.pins
        acc.enums |= f.enums
        acc.named |= f.named
        acc.missing |= f.missing
        acc.sources |= f.sources
        acc.kernels |= f.kernels
        work.extend(f.code)
        if defs:
            stack.extend(f.defs.values())

    while True:
        expand()
        if not work:
            break
        f = os.path.abspath(work.pop())
        if f in done:
            continue
        done.add(f)
        if f in _unit_files() or f.startswith(os.path.join(str(_HERE), "")):
            continue                                   # the checker's version, and the binding files by their units
        if (rel := _rel(f)) is None:
            continue
        name = module(f, rel)
        if name is None:
            continue
        if name in _MODULES:
            take(_MODULES[name])
        parts = name.split(".")
        for imp in _top_imports(f, name) + [".".join(parts[:i]) for i in range(1, len(parts))]:
            m = sys.modules.get(imp)
            if (mf := getattr(m, "__file__", None)) is not None:
                work.append(mf)
    # a unit is named by its file (`_named_file`): `targets.py`'s bare, a sidecar's as ``<module>.circuit-check#<unit>``; each
    # file's ``common`` comes with it
    trel = _rel(TARGETS)
    held = {u if "#" in u else f"{trel}#{u}" for u in unit_names if "#" in u or trel is not None}
    for f, line in acc.lines:
        if (n := _named_file(f, ".py", by_file)) is not None:
            held.update(f"{n}#{u}" for u in _unit_at(_units_cached(Path(f).read_text())[1], line))
    held |= {f"{u.partition('#')[0]}#common" for u in held}
    for f in _unit_files():
        if (r := _rel(f)) is not None and r in acc.files and (n := _named_file(f, ".py", by_file)) is not None:
            acc.files.discard(r)                       # read as a file (a traceback's source line, say): all of it
            held.add(f"{n}#*")
    pin_names = {r: n for f in _PIN_FILES
                 if (r := _rel(f)) is not None and (n := _named_file(f, ".json", by_file)) is not None}
    pins = {f"{pin_names[r]}#*" for r in pin_names.keys() & acc.files}
    acc.files -= pin_names.keys()
    where = _PIN_FILE[0] if _PIN_FILE else (lambda kind, key: [E.path(PINS / kind, key, ".json")])
    for p in acc.pins:
        kind, _, k = p.partition("#")
        if p == "*" or k == "*":
            pins |= {f"{n}#{p}" for n in pin_names.values()}
        else:
            pins |= {f"{n}#{p}" for f in where(kind, k) if (n := _named_file(os.path.abspath(f), ".json", by_file)) is not None}
    enums: set[str] = set()
    for rel in acc.enums:                              # a package `load_registries` enumerated, by its name
        f = os.path.abspath(os.path.join(root, rel, "__init__.py"))
        name = by_file.get(f)
        if name is not None and (p := _resolve(root, name)) is not None and os.path.realpath(p) == os.path.realpath(f):
            mods.add(name)
            enums.add(name)
        else:
            acc.dirs.add(rel)
    present: set[str] = set()
    for rel in acc.sources:                            # a loaded module's source looked for, by the module's name
        f = os.path.abspath(os.path.join(root, rel))
        name = by_file.get(f)
        if f.startswith(os.path.join(str(_HERE), "")):
            continue                                   # the checker's version, and `targets.py` by its units
        if name is not None and (p := _resolve(root, name)) is not None and os.path.realpath(p) == os.path.realpath(f):
            present.add(name)
        elif name is not None and (m := _side(name)) is not None and (r := _sidecar(root, m, ".py")) is not None \
                and os.path.realpath(root / r) == os.path.realpath(f):
            present.add(name)                          # a sidecar's, by its module's name: it moves with its module
        else:
            acc.probes.add(rel)
    return {"files": sorted(acc.files), "dirs": sorted(acc.dirs), "probes": sorted(acc.probes - acc.files - acc.dirs),
            "libs": sorted(acc.libs), "units": sorted(held), "pins": sorted(pins),
            "modules": sorted(mods), "named": sorted(acc.named & mods), "missing": sorted(acc.missing), "enums": sorted(enums),
            "present": sorted(present - mods), "kernels": sorted(acc.kernels), "dists": _outside_tops()}


_PACKAGES: list[dict[str, list[str]]] = []


def _packages() -> dict[str, list[str]]:
    """Each importable top-level name an installed distribution provides, with those distributions (read once a process)."""
    if not _PACKAGES:
        from importlib import metadata

        _PACKAGES.append(metadata.packages_distributions())
    return _PACKAGES[0]


def _outside_tops() -> list[str]:
    """The top-level modules loaded from an installed distribution outside the tree: every one in the process, a superset of
    what this target's closure reached (an import from outside the tree isn't followed into its own imports), each held by
    the versions of the distributions that provide it (`_dist`). The interpreter's own modules are the key's ``python``."""
    out = set()
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if "." not in name and isinstance(f, str) and _rel(f) is None and _packages().get(name):
            out.add(name)
    return sorted(out)


def _dist(root: Path, top: str) -> str | None:
    """Top-level module ``top`` as an installed distribution now: ``tree`` when a file in the tree now resolves the name (a
    package that shadows it), else each providing distribution's ``name==version``; None when none provides it."""
    if _resolve(root, top) is not None:
        return "tree"
    from importlib import metadata

    names = _packages().get(top)
    if not names:
        return None
    try:
        return ",".join(sorted(f"{d}=={metadata.version(d)}" for d in set(names)))
    except metadata.PackageNotFoundError:
        return None


class _Prose(ast.NodeTransformer):
    """A module's AST without its statements that are a string alone (its docstrings among them)."""

    def visit_Expr(self, node: ast.Expr) -> ast.AST | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


_CODE: dict[bytes, bytes] = {}


def _code(p: Path) -> bytes:
    """A file of the checker as its version holds it: a module by the digest of its AST with every string-alone statement cut
    (`_Prose`; the AST holds no comment, formatting or line number), so an edit to its prose keeps the version; any other file
    by its bytes.  Memoized by the bytes, since `key` reads every file again for each entry."""
    data = p.read_bytes()
    if p.suffix != ".py":
        return data
    if data not in _CODE:
        _CODE[data] = b"ast:" + hashlib.sha512(ast.dump(_Prose().visit(ast.parse(data))).encode()).digest()
    return _CODE[data]


def version() -> str:
    """The checker's own version: the digest of this package's code (`_code`: no docstring or comment), less `targets.py` and
    ``pins/`` (each target holds its own parts of them), `HEAVY`, `KNOWN` and `SHARD` (no target's input)."""
    h = hashlib.sha512()
    _INTERNAL.append(True)
    try:
        for p in sorted(_HERE.rglob("*")):
            if (p.is_file() and "__pycache__" not in p.parts and p not in (TARGETS, HEAVY, KNOWN[0], SHARD)
                    and PINS not in p.parents and KNOWN[1] not in p.parents):
                h.update(p.relative_to(_HERE).as_posix().encode() + b"\0" + _code(p) + b"\0")
    finally:
        _INTERNAL.pop()
    return h.hexdigest()


#: variables of the keyed prefixes that say where test results are kept, not what a check computes: a check slot sets its own
#: `VERITY_TEST_CACHE` (`tools/verity/check/slot.py`), which would keep one slot's reports from counting in another
UNKEYED = frozenset({"VERITY_TEST_CACHE", "VERITY_TEST_CACHE_MIN_S"})


def key(request: dict) -> dict:
    _INTERNAL.append(True)                             # the tracer's own work (`platform` runs `uname`): no input of a check
    try:
        # the system, the CPU and the C library, not `platform.platform()`: that names the host's kernel release, which differs
        # between pods and would keep one pod's reports from ever counting on another (`tools/verity/check/verdicts.py`)
        plat = "-".join([platform.system(), platform.machine(), *platform.libc_ver()])
    finally:
        _INTERNAL.pop()
    return {"schema": SCHEMA, "version": version(), "request": request, "python": sys.version, "optimize": sys.flags.optimize,
            "platform": plat,
            "registries": hashlib.sha256("\n".join(_REGISTRIES).encode()).hexdigest(),
            "environment": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith(("VERITY_", "CIRCUIT_CHECK_")) and k not in UNKEYED}}


_OWN: dict[int, tuple[Any, str]] = {}


def descriptor(fn: Any) -> str:
    """The digest of every primitive and specialization ``fn`` reaches (through body nodes and function statics), each by its
    descriptor id and the digest of its canonical descriptor entry (`verity.ml.circuits.codec`): what the target is, whatever module or
    path it was written in.  A by-calls root (`targets.is_by_calls`) is its own entry (`by_calls.entry`), without what its body
    calls.  A body that cannot be built gives the error's type, so no stored entry matches."""
    from verity.ml.circuits import codec as C
    from verity.ml.circuits.defs import FunctionDefinition, PrimitiveDefinition, SpecializedDefinition

    from . import by_calls as BC
    from . import targets as TG

    def statics(v: Any) -> Iterator[Any]:
        if isinstance(v, FunctionDefinition):
            yield v
        elif isinstance(v, (list, tuple, dict)):
            for x in v.values() if isinstance(v, dict) else v:
                yield from statics(x)

    parts: list[list[str]] = []
    seen: set[int] = set()
    stack = [fn]
    with recording(merge=False):                       # a body built here records its own build (`_build_body`)
        try:
            while stack:
                f = stack.pop()
                if id(f) in seen:
                    continue
                seen.add(id(f))
                by_calls = TG.is_by_calls(f)
                if isinstance(f, (PrimitiveDefinition, SpecializedDefinition)):
                    e = _OWN.get(id(f))
                    if e is None or e[0] is not f:
                        own = BC.entry(f) if by_calls else C.definition_entry(f)
                        e = _OWN[id(f)] = (f, hashlib.sha512(C.canonical_json(own)).hexdigest())
                    parts.append([C.descriptor_id(f), e[1]])
                if isinstance(f, SpecializedDefinition):
                    if not by_calls:
                        stack.extend(n.fn for n in f.body.nodes)
                    stack.extend(statics(f.bindings))
        except Exception as e:  # noqa: BLE001  (its check raises too, and an errored check is never stored)
            return f"error:{type(e).__name__}"
    return hashlib.sha512(json.dumps(sorted(parts)).encode()).hexdigest()


def _kind(p: Path) -> str | None:
    return "dir" if p.is_dir() else "file" if p.exists() else None


def _unit(root: Path, r: str, seen) -> str | dict | None:
    """A binding unit as it is now: its digest and what its import sites resolve to, from the package Python imports the file
    in (a sidecar's, ``<module>.circuit-check``'s: its module's name); all of the file, by its bytes.  A sidecar's is read
    where its module is now (`_sidecar`)."""
    head, _, name = r.partition("#")
    module = _side(r)
    path = head if module is None else _sidecar(root, module, ".py")
    if path is None or not seen(path) or not (root / path).is_file():
        return None
    if name == "*":
        return _sha_file(root / path)
    units, _spans, sites = _units_cached((root / path).read_text())
    if name not in units:
        return None
    return {"d": units[name], "sites": _sites(root, module or __package__ or "circuit_check", sites[name])}


def _pin(root: Path, r: str, seen) -> str | None:
    """A pin as it is now, by its value, in its file of ``pins/`` or where its sidecar's module is now; all of a file, by its
    bytes."""
    head, _, rest = r.partition("#")
    module = _side(r)
    path = head if module is None else _sidecar(root, module, ".json")
    if path is None or not seen(path) or not (root / path).is_file():
        return None
    if rest == "*":
        return _sha_file(root / path)
    kind, _, k = rest.partition("#")
    pins = json.loads((root / path).read_text())
    if kind not in pins:
        return None
    v = pins[kind] if k == "*" else pins[kind].get(k)
    return None if v is None else hashlib.sha512(json.dumps(v, sort_keys=True).encode()).hexdigest()


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
    seen = _seen(root)
    if fresh:
        _PATHS.clear()

    def memo(kind: str, r: str, fn):
        if fresh:
            return fn()
        k = (str(root), kind, r)
        if k not in _MEMO:
            _MEMO[k] = fn()
        return _MEMO[k]

    files = {r: memo("f", r, lambda r=r: _sha_file(root / r) if seen(r) and (root / r).is_file() else None) for r in got["files"]}
    dirs = {r: memo("d", r, lambda r=r: _distributions() if r == DISTRIBUTIONS else
                    _listing(root, r) if seen(r) and (root / r).is_dir() else None) for r in got["dirs"]}
    probes = {r: memo("p", r, lambda r=r: _kind(root / r) if seen(r) else None) for r in got.get("probes", [])}
    # a library the dynamic loader mapped is git-ignored (built from its source here), so it counts by its bytes
    libs = {r: memo("l", r, lambda r=r: _sha_file(root / r) if (root / r).is_file() else None) for r in got.get("libs", [])}
    units = {r: memo("u", r, lambda r=r: _unit(root, r, seen)) for r in got.get("units", [])}
    pins = {r: memo("n", r, lambda r=r: _pin(root, r, seen)) for r in got.get("pins", [])}
    modules = {r: memo("m", r, lambda r=r: _modrec(root, r)) for r in got.get("modules", [])}
    named = {r: memo("c", r, lambda r=r: _local(root, r)) for r in got.get("named", [])}
    missing = {r: memo("c", r, lambda r=r: _local(root, r)) for r in got.get("missing", [])}
    enums = {r: memo("e", r, lambda r=r: _children(root, r)) for r in got.get("enums", [])}
    present = {r: memo("c", r, lambda r=r: _local(root, r)) for r in got.get("present", [])}
    kernels = {r: memo("k", r, lambda r=r: _kernels_now(r)) for r in got.get("kernels", [])}
    dists = {r: memo("x", r, lambda r=r: _dist(root, r)) for r in got.get("dists", [])}
    return {"files": files, "dirs": dirs, "probes": probes, "libs": libs, "units": units, "pins": pins, "modules": modules,
            "named": named, "missing": missing, "enums": enums, "present": present, "kernels": kernels,
            "dists": dists}


#: what matches by equality alone: paths and their contents, the names imports looked up by name, kernel registrations (and
#: ``pins/``'s pins; a sidecar's are read where its module is, `_moved`)
_EXACT = ("files", "dirs", "probes", "libs", "named", "missing", "kernels", "dists")


def _seen(root: Path):
    v = visible(root)
    return (lambda r: True) if v is None else (lambda r: r in v[0] or r in v[1])


def _at(read: str, module: str) -> str:
    """Sidecar read ``read`` (`_side`) named for its module at ``module``."""
    return f"{module}{SUFFIX}#{read.partition('#')[2]}"


def _moved(root: Path, rec: dict, now: dict, renamed: dict[str, str] | None = None) -> bool:
    """Whether the tree differs from what entry ``rec`` recorded only by where its modules are (each module's name now in
    ``renamed``, when one is given and it holds): each recorded module found, with
    the same content, at the name the recorded import sites resolve to now, starting from the binding units and the
    modules that still resolve by name, and propagating along each found module's own sites.  A sidecar's units and pins are
    read where its module was found (one not held that left its name, where its units' import sites of it lead; else at its
    name), its units' sites propagating in turn; ``pins/``'s are equal.

    A module stays at its name when it reads its own path or name, was imported by a name computed at run time, or an import
    of it failed (``named``, ``missing``).  One no site leads to is found among what this process imported before any check,
    when exactly one has its content.  Every package a found module now imports through is one recorded, or empty or a
    docstring alone, or, for a sidecar's module the entry does not hold, one it was imported through at its name as well.  A package an enumerator listed lists the same modules, or each it no longer lists is imported anyway and
    each it lists anew was listed under another name or is empty.  A module whose source `inspect` looked for (``present``)
    still resolves by its name, or was found where it moved, or is the one imported module with its content."""
    if any(now.get(k) != rec.get(k) for k in _EXACT):
        return False
    mods: dict[str, dict | None] = rec.get("modules", {})
    phi: dict[str, str] = {}
    todo: list[tuple[list, list]] = []

    def pair(was: list, is_: list) -> bool:
        if len(was) != len(is_):
            return False
        for (t, d), (t2, _d2) in zip(was, is_):
            if mods.get(t) is not None:
                if t in phi:
                    if phi[t] != t2:
                        return False
                elif not locate(t, t2):
                    return False
            elif t2 != t and not (d is not None and _resolve(root, t) is None and _local(root, t2) == d):
                return False
        return True

    def locate(n: str, n2: str) -> bool:
        r, cur = mods[n], _modrec(root, n2)
        if cur is None or r is None or cur["d"] != r["d"] or cur.get("at") != r.get("at"):
            return False
        phi[n] = n2
        todo.append((r["sites"], cur["sites"]))
        return True

    seen = _seen(root)

    def where(read: str, kind: str, fn) -> Any:
        """A sidecar's read as it is where its module is now: found (``phi``), else, one not held, at its name."""
        m = _side(read)
        return now[kind].get(read) if phi.get(m, m) == m else fn(root, _at(read, phi[m]), seen)

    def unit(r: Any, c: Any) -> bool:
        if isinstance(r, dict) != isinstance(c, dict) or (not isinstance(r, dict) and r != c):
            return False
        if isinstance(r, dict):
            if r["d"] != c["d"]:
                return False
            todo.append((r["sites"], c["sites"]))
        return True

    def follow(m: str) -> None:
        """Where module ``m``, not held, went when it left its name: where its sidecar's units' import sites of it lead, the one
        imported module with the content they recorded."""
        if _resolve(root, m) is not None:
            return
        ds = {d for u, r in side.items() if _side(u) == m and isinstance(r, dict) for t, d in r["sites"] if t == m}
        if len(ds) == 1 and (d := ds.pop()) is not None and len(names := _loaded(root).get(d, [])) == 1:
            if _side(names[0]) is None and names[0] not in phi.values():
                phi[m] = names[0]

    side: dict[str, Any] = {}                          # a sidecar's units, read once its module is found
    for u, r in rec.get("units", {}).items():
        if _side(u) is not None:
            side[u] = r
        elif not unit(r, now["units"].get(u)):
            return False
    for p, r in rec.get("pins", {}).items():
        if _side(p) is None and now["pins"].get(p) != r:
            return False
    for n, r in mods.items():
        c = now["modules"].get(n)
        if r is None:
            if c is not None:
                return False
        elif c is not None and c["d"] == r["d"] and c.get("at") == r.get("at"):
            phi[n] = n
            todo.append((r["sites"], c["sites"]))
        elif n in rec.get("named", {}) or "at" in r:
            return False
    while True:
        while todo:
            if not pair(*todo.pop()):
                return False
        for m in {_side(u) for u in side} - phi.keys() - mods.keys():
            follow(m)
        if ready := [u for u in side if (m := _side(u)) in phi or mods.get(m) is None]:
            for u in ready:
                if not unit(side.pop(u), where(u, "units", _unit)):
                    return False
            continue
        left = [n for n, r in mods.items() if r is not None and n not in phi]
        if not left:
            break
        n = left[0]
        if n in rec.get("named", {}) or "at" in mods[n]:
            return False
        names = _loaded(root).get(mods[n]["d"], [])
        if len(names) != 1 or names[0] in phi.values() or not locate(n, names[0]):
            return False
    image = set(phi.values())
    if len(image) != len(phi):
        return False
    for n, n2 in phi.items():
        if n2 == n:
            continue
        parts = n2.split(".")
        # a followed module's packages it was imported through at its name as well: held before the move no more than after
        through = set() if n in mods else {".".join(n.split(".")[:i]) for i in range(1, n.count(".") + 1)}
        for i in range(1, len(parts)):
            p = ".".join(parts[:i])
            if p not in image and p not in through and (f := _resolve(root, p)) is not None and not _parsed(f)[3]:
                return False
    found: dict[str, str] = {}
    for n, d in rec.get("present", {}).items():
        if (m := _side(n)) is not None and phi.get(m, m) != m:
            if (d is None) != (_local(root, n2 := f"{phi[m]}{SUFFIX}") is None):
                return False
            found[n] = n2
        elif (d is None) != (now["present"].get(n) is None):
            if d is None or len(names := _loaded(root).get(d, [])) != 1:
                return False
            found[n] = names[0]
    for pkg, kids in rec.get("enums", {}).items():
        cur = _children(root, phi.get(pkg, ""))
        if cur is None or kids is None:
            return False
        if cur == kids:
            continue
        loaded, was = _loaded(root), set(kids.values())
        for c, d in kids.items():
            if c not in cur and d not in loaded:
                return False
        for c, d in cur.items():
            if c not in kids and d not in was and ((f := _resolve(root, f"{phi[pkg]}.{c}")) is None or not _parsed(f)[3]):
                return False
    for p, r in rec.get("pins", {}).items():
        if _side(p) is not None and where(p, "pins", _pin) != r:
            return False
    if renamed is not None:
        renamed.update(phi)
        renamed.update(found)
    return True


_MEMO: dict[tuple[str, str, str], str | None] = {}


def _dir(cache: Path, k: dict) -> Path:
    return Path(cache) / k["version"] / hashlib.sha512(json.dumps(k, sort_keys=True, default=str).encode()).hexdigest()[:32]


#: the reads that name modules, which a move renames, and those that name a sidecar by its module's
_RENAMED = ("modules", "enums", "present")
_SIDE_RENAMED = ("units", "pins")


def lookup(cache: Path, k: dict, root: Path, moved: bool = True) -> tuple[dict, str] | None:
    """(report, entry id) of an entry with key ``k`` whose every recorded read is unchanged under ``root``, else None.  An entry
    found only where its modules moved (`_moved`) is stored again with its reads renamed to where they are, digested now (what
    a run would record), so that a later edit of a moved module is told from a move; not with ``moved`` false, which `_moved`
    needs for the modules this process imported (`_loaded`)."""
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
        if not moved:
            continue
        renamed: dict[str, str] = {}
        _INTERNAL.append(True)
        try:
            hit = _moved(root, rec, now, renamed)
        finally:
            _INTERNAL.pop()
        if hit:
            got = {r: [renamed.get(n, n) for n in v] if r in _RENAMED
                   else [_at(n, renamed[m]) if (m := _side(n)) in renamed else n for n in v] if r in _SIDE_RENAMED
                   else list(v) for r, v in rec.items()}
            got["modules"] = sorted({*got.get("modules", ()), *(n2.rsplit(".", i)[0] for n, n2 in renamed.items() if n2 != n
                                                                for i in range(1, n2.count(".") + 1))})
            return e["report"], store(cache, k, root, got, e["report"]) or p.stem
    return None


def seen(root: Path, got: dict[str, list[str]]) -> dict[str, dict[str, str | None]]:
    """The reads as the process that just ran a check sees them: digested now, not remembered from earlier in the run."""
    return _digests(root, got, fresh=True)


def seen_all(root: Path, gots: list[dict[str, list[str]]]) -> list[dict[str, dict[str, str | None]]]:
    """`seen` of each read set in ``gots``, every read among them digested once, now: what each gives on its own while the tree
    is unchanged (`circuit_check.units` stores hundreds of records whose reads are mostly the same modules)."""
    if not gots:
        return []
    every = {k: sorted({r for g in gots for r in g.get(k, ())}) for k in {"files", "dirs", *(k for g in gots for k in g)}}
    now = seen(root, every)
    return [{k: {r: v[r] for r in g.get(k, ())} for k, v in now.items()} for g in gots]


def unit_reads(units: Iterable[str], roots: Iterable[Any]) -> dict[str, list[str]]:
    """What a catalog binding's build read (`circuit_check.units`): what its ``units`` read and ran while they made their roots
    (`producing`), and `attributed` over its ``roots``: the build of every Definition they reach, the bindings that made those,
    those modules' import closures, its own units by digest, and what the process read outside every recording."""
    units = tuple(units)
    fr = Frame()
    for u in units:
        if u in _PRODUCED:
            fr.merge(_PRODUCED[u], defs=False)
    return attributed(fr, units, roots)


def task_list_reads(roots: Iterable[Any]) -> dict[str, list[str]]:
    """What the task list's entry holds (`cli.run`): what any target could hold (`attributed` over every binding's recording and
    every root), every in-tree module this process imported but the checker's own (its key's `version`), and every binding file
    whole.  More than the union of the targets' reads: a new binding entry, a new Definition in a registry module or a new module
    an enumerated package lists changes the list without changing what any target read."""
    fr = Frame()
    for f in _PRODUCED.values():
        fr.merge(f)
    got = attributed(fr, (), roots)
    _INTERNAL.append(True)
    try:
        root = _ROOT[0]
        by_file: dict[str, str] = {}
        for name, m in list(sys.modules.items()):
            if isinstance(f := getattr(m, "__file__", None), str):
                by_file.setdefault(os.path.abspath(f), name)
        mods = {name for f, name in by_file.items() if _rel(f) is not None and not f.startswith(os.path.join(str(_HERE), ""))
                and (p := _resolve(root, name)) is not None and os.path.realpath(p) == os.path.realpath(f)}
        units = {f"{n}#*" for f in _unit_files() if (n := _named_file(f, ".py", by_file)) is not None}
    finally:
        _INTERNAL.pop()
    return {**got, "modules": sorted(set(got["modules"]) | mods), "units": sorted(set(got["units"]) | units)}


def store(cache: Path, k: dict, root: Path, got: dict[str, list[str]], report: dict,
          worker: dict[str, dict[str, str | None]] | None = None) -> str | None:
    """Record ``report`` under ``k`` with its reads' digests; None, and nothing stored, when ``worker`` (`seen` in the process that
    ran the check) differs from them: the tree changed while the run went on, and the report may be of a tree no digest names."""
    now = _digests(root, got)
    if worker is not None and worker != now:
        return None
    e = {"key": k, "reads": now, "report": report}
    body = json.dumps(e, sort_keys=True, separators=(",", ":"), default=str)
    eid = hashlib.sha512(body.encode()).hexdigest()
    d = _dir(cache, k)
    d.mkdir(parents=True, exist_ok=True)
    # a temporary name of this writer's own: two checks may store the same entry at once
    tmp = d / f".{eid}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    tmp.write_text(body)
    os.replace(tmp, d / f"{eid}.json")
    return eid


__all__ = ["PINS", "SCHEMA", "SUFFIX", "TARGETS", "UNTRACEABLE", "Frame", "attributed", "internal", "key", "lookup", "pin_files",
           "produced", "recorded_pins", "recording", "registries", "seen", "seen_all", "store", "task_list_reads", "trace", "tracing",
           "unit_files", "unit_reads", "untraced", "version"]
