"""The dependency rules (AGENTS.md "Map and commands") as import-boundary checks over the tree.

The import order is one listing, ``tests/import_order.toml``, and the checks at the end of this file read only it:

    a file loads only what its own row or an earlier row holds, in Python and in Lean, but for the ties the listing
    names by file (``against``), which only shrink;  no file under a ``verifier/`` loads one under a ``prover/``;
    ``verity``'s own Python is in places (``place``: a place is the directories and files its ``paths`` name), and a
    place's Python names only files of its own place and of the places its ``loads`` lists;  a place of the
    verifiers' side is trusted (one under a ``verifier/`` or a ``device/``, or one the listing marks), and until a
    protocol's verifier is its Lean, a trusted place loads and reaches only the places the listing froze for it, the
    soundness path, which only shrinks;  under ``verity/``, outside the trusted places, only the files a trusted
    place's ``loaded_by`` lists name a file of it, tests included, and that only shrinks too;  Python of ``verity``
    in no place only groups packages and imports the standard library alone;  a directory under ``verity/`` with an
    import root of its own is a ``distribution`` the listing names with its owner, which no file of ``verity`` loads
    by name;  ``infra/`` is the last row, which no file of another row loads, a test included, but the ties it lists.

The listing's tables (``against`` and each place's lists) are generated from the tree by
``tools/verity/check/import_order.py``, and the listing is byte for byte what it writes.

Beside the order:

    ``verity`` imports the standard library, its declared dependencies and itself, never another distribution of the
    workspace (Flock, an integration, a benchmark, a tool) or an external runtime, and a verifier, a device and the
    scalar references (``STDLIB_ONLY``) no dependency;
    nothing outside ``benchmarks/`` imports ``benchmarks/``, by name or by path (``BENCHMARKS_IMPORTERS``);  circuit-check
    imports ``verity`` and no other distribution of the workspace (``CIRCUIT_CHECK_IMPORTS``);  only tests name
    a function past a guard (``TEST_ONLY``);  SPOT[Flock] imports none of the network certifier, in its Python, its Lean
    packages or its analysis (``SPOT_NOT_CERTIFIER``).

An import is an import statement (a relative one resolved against its package), a string handed to
``importlib.import_module``, ``__import__`` or ``importorskip``, or a ``"module:attr"`` string naming what to load at run
time: a module named as data is still a dependency.  The order reads a load by name; a load by file (a path handed to
``spec_from_file_location``, ``SourceFileLoader`` or ``run_path``) is read only into ``benchmarks/``.  Tests (``tests/``)
are suites of their own and declare what they read;
a ``lean/`` directory holds a Lean package and the scripts that write its vectors, run by path.  The directories of the
listing's ``distribution`` have import roots of their own, and only ``loaded_by`` reads them as loaders.  No other file
under ``verity/``, tests included, reads another distribution at run time: its package data or a module named in a
string (`RUN_TIME_READ`, matched in the text, f-strings too).

numpy is a core dependency (architecture decision: `verity.circuits.evaluation` batches over numpy arrays).  The
plain-integer references stay numpy-free on purpose, because they are the specification the Rust guest, the Lean vectors and
the probes are checked against: the rest of `verity.circuits` (the tensor-core step in `verity.definitions.silicon`
among them).  `verity.commitments.randomness` is exact stdlib arithmetic too.
"""

from __future__ import annotations

import ast
import builtins
import fnmatch
import functools
import importlib.util
import posixpath
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Set
from pathlib import Path
from typing import Any, NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "verity"

#: Import names ``verity`` never imports or names as a string: the other workspace distributions, external runtimes, and the
#: repository's own directories.
FORBIDDEN_EXTERNAL = ("verity_sp1", "verity_vllm", "verity_numerical", "verity_flock", "verity_torch", "verity_cli", "verity_difftest",
                      "circuit_check", "research", "cluster", "sp1", "vllm", "torch", "jax", "onnx", "archive", "backends",
                      "benchmarks", "examples", "frontends", "infra", "integrations", "tools", "veritor", "adapters", "demos")
#: ``verity``'s third-party dependencies (the root's ``pyproject.toml``); anything else it imports is the stdlib or
#: ``verity``.
DEPENDENCIES = {"numpy"}
#: Where ``verity``'s Python imports none of them either, only the standard library and ``verity``: every verifier's and
#: device's directory (a protocol composes no third-party package into a verdict), and these: the scalar references,
#: the hardware's cards, record and registration log, SPOT's planning beside its verifier, and the reporting
#: grid.  Within them, what may use numpy.
STDLIB_ONLY = ("verity/circuits", "verity/definitions/circuits", "verity/definitions/silicon", "verity/commitments/randomness",
               "verity/claims", "verity/hardware", "verity/protocols/spot", "verity/protocols/reporting.py")
NUMPY_USERS = ("verity/circuits/evaluation", "verity/circuits/boolean/forms.py")
#: The listing's distributions that name neither the date they become part of ``verity`` nor the reason they stay their
#: own: each owner owes one of the two.  It only shrinks.
NO_DATE_OR_REASON = ("verity/configuration/config", "verity/protocols/inference_only/nci",
                     "verity/composite/device/live_challenger", "verity/protocols/poub/device/live_verifier",
                     "verity/protocols/pouw/clear", "verity/protocols/pouw/receipts")
#: SPOT[Flock] stands on its own, and the network certifier is a layer beside it: SPOT reads the certifier's stamp as a
#: certified input and imports none of the certifier, and the two compose only above both.  SPOT's row of the
#: listing and the certifier's, the trees each of whose Python files (tests included) is checked, SPOT's Lean packages
#: and its analysis (Security's Flock and Spot areas and the sampling claims); then the certifier's modules and Lean
#: (its packages and its Security areas).
SPOT_ROW = "verity/protocols/spot"
CERTIFIER_ROW = "verity/protocols/certifier"
SPOT_TREES = ("verity/protocols/spot", "verity/protocols/spot/verifier/lean", "verity/protocols/flock/verifier",
              "verity/protocols/flock/prover", "verity/protocols/flock/pod", "verity/protocols/flock/agreement", "verity/protocols/flock/tool.py")
SPOT_LEAN = ("verity/protocols/spot/verifier/lean", "verity/protocols/flock/verifier")
SPOT_SECURITY = ("security/*/Flock", "security/*/Flock.lean", "security/*/Spot", "security/*/Spot.lean",
                 "security/Definitions/Claims/Sampling*")
CERTIFIER_MODULES = ("verity.protocols.certifier.verifier", "verity.protocols.certifier.prover", "verity.protocols.certifier.device")
CERTIFIER_LEAN = ("verity/protocols/certifier/device/program/lean", "verity/protocols/certifier/device/lean",
                  "verity/protocols/certifier/verifier/lean", "security/*/NetworkCertifier", "security/*/NetworkCertifier.lean")
SPOT_NOT_CERTIFIER = ("SPOT[Flock] never imports the network certifier, which composes with it only above both and whose "
                      "stamp SPOT reads as a certified input")
#: Flock's Lean verifier, by its Lake package's name, and the Lake packages that may require it: those whose Lean reads the
#: proof system's verifier itself.  The list only shrinks.  A package that reads only the circuit format, the commitments or
#: the bytes, hashes and parsing below them requires a base package (``BASE_LEAN``) instead.
FLOCK_VERIFIER = "flock_verifier"
REQUIRES_FLOCK_VERIFIER = {
    "core_firewall", "core_memory_challenger", "core_onsite", "core_remote", "core_sigma", "core_spot", "core_vstar",
    "core_vstar_recursive", "network_check", "pous_check", "pouw_commit", "remote_audit", "security_proofs",
    "security_properties",
}
#: The base packages below Flock's verifier, each with every package it may require: nothing above it.
BASE_LEAN = {"core_base": (), "core_circuits": ("core_base",), "core_commitments": ("core_base",)}
#: Files outside ``benchmarks/`` that import a module of it, by name or by path, each with its owner and what it imports.
#: ``benchmarks/`` is measurement only: a number a protocol uses reaches it
#: through a card file citing an ``art:`` id, never as an import, and code a protocol runs moves out to the package that
#: uses it.  It only shrinks.
BENCHMARKS_IMPORTERS = {
    # the served path: verity_vllm runs benchmarks' code
    "adapters/vllm/verity_vllm/protocols/pous.py": "memory-accounting: P2's store and its load "
                                                       "(`benchmarks/pous/p2_v1`), through its `P2_V1`",
    "adapters/vllm/verity_vllm/protocols/remote_bandwidth/remote.py": "circuits (vllm): the remote session's processes "
                                                                          "(`benchmarks/remote_session`, "
                                                                          "its `REMOTE_SESSION`) and `paged.py` "
                                                                          "(`benchmarks/vllm_encoded`, its `VLLM_ENCODED`)",
    # Flock's measurement campaign, its tests, and the frozen instance sets `instances` reads
    **dict.fromkeys((
        "verity/protocols/flock/pod/gemm_hill.py", "verity/protocols/flock/pod/read_gadget.py",
        "verity/protocols/flock/prover/python/verity_flock/bench.py", "verity/protocols/flock/prover/python/verity_flock/circuit_bench.py",
        "verity/protocols/flock/prover/python/verity_flock/instances.py", "verity/protocols/flock/prover/python/verity_flock/ir_bench.py",
        "verity/protocols/flock/prover/python/verity_flock/register.py",
        "verity/protocols/flock/prover/tests/test_backend_bench.py", "verity/protocols/flock/prover/tests/test_class_statement.py",
        "verity/protocols/flock/prover/tests/test_gemm_hill.py", "verity/protocols/flock/prover/tests/test_ir_lowering_bench.py",
    ), "proofs: `verity_numerical.bench` (`benchmarks/numerical`)'s contract, cell, generator suites and frozen instances"),
    # research's registry of measuring tools
    **dict.fromkeys(("tools/research/src/research/store/tools_registry.py", "tools/research/tests/test_store_prov.py"),
                    "infra: the store's tool registry names `benchmarks.<probe>.tool` modules (tc_probe, tc_probe_fp4, "
                    "native_peak, commitments, pouw)"),
    # tests and vector scripts that check against a benchmark's reference
    "verity/protocols/pouw/verifier/lean/window/tests/test_known_rows.py": "proofs: H1's known rows "
                                                                       "(`benchmarks/pouw/h1_lanes/run.py`)",
    **dict.fromkeys(("verity/protocols/pous/verifier/check/scripts/difftest_vectors.py",
                     "verity/protocols/pous/verifier/check/tests/test_timed_audit.py"),
                    "memory-accounting: P2's reference and wire (`benchmarks/pous/p2_v1`) and the harness's control "
                    "responder (`benchmarks/pous/band_gpu/harness.py`)"),
    "verity/definitions/ops/tests/test_tc_probe_families.py": "circuits: the tensor-core probe's families "
                                                              "(`benchmarks/tc_probe`)",
    **dict.fromkeys(("adapters/vllm/tests/protocols/pouw_h1_circuits.py", "adapters/vllm/tests/protocols/pouw_h1_gpu.py"),
                    "compute-accounting: H1's prover, producer and the live challenger's auditor "
                    "(`benchmarks/pouw/h1_prove`, `served_h1`, `verity/composite/device/live_challenger`)"),
    **dict.fromkeys(("adapters/vllm/tests/engine/ppl_oracle.py", "adapters/vllm/tests/engine/fattn_oracle.py"),
                    "circuits (vllm): the int8 step's oracles (`benchmarks/ppl/steps.py`, `benchmarks/pouw/edited_gemm`)"),
    "adapters/vllm/tests/engine/int8_attention_sass.py": "kernels: the PoUW kernels' SASS gate "
                                                             "(`benchmarks/pouw/harness/sass_gate.py`)",
    # brought out of benchmarks/ by the layout move, each with a load of benchmarks/ by path
    "verity/composite/device/live_challenger/auditor.py":
        "proofs: the relay's load (`benchmarks/network_traces/load_relay.py`), "
        "loaded by path since before the move; goes when the session's code moves under `verity/`",
    "verity/protocols/poub/device/live_verifier/tests/stub_verifier.py":
        "memory-accounting: the remote session (`benchmarks/remote_session`), "
        "loaded by path since before the move; goes when the session's code moves under `verity/`",
    "verity/protocols/poub/device/live_verifier/tests/test_host.py":
        "memory-accounting: the remote session (`benchmarks/remote_session`), "
        "loaded by path since before the move; goes when the session's code moves under `verity/`",
}
#: circuit-check's own code, a developer tool of ``verity``: it imports ``verity`` and itself, and no other distribution of
#: the workspace.  What a check needs of one, that distribution hands it in a sidecar it owns (`circuit_check.sidecars`).
CIRCUIT_CHECK = "tools/verity/circuit_check/src"
#: The files of circuit-check that still import another distribution of the workspace, each with the packages it imports:
#: Flock's lowerings and templates, which its lowering and template checks hold each Definition's Boolean circuit to.  It
#: only shrinks.
CIRCUIT_CHECK_IMPORTS = {
    f"{CIRCUIT_CHECK}/circuit_check/{name}.py": {"verity_flock"}
    for name in ("ambient", "checks", "sidecars", "structural", "targets")
}
#: Functions past a guard, which tests alone may name, since a guard never fails open (AGENTS.md "Working principles"): each
#: with the file that defines it and why tests need it.
TEST_ONLY = {
    "_build_unchecked": ("verity/circuits/partition_v2.py",
                         "`partition_v2.build` without its refusal of units of several size classes, for the tests "
                         "of `check` on advice past the builder's scope, as a prover may send it"),
}

#: Modules that only a prover's or developer's build turns on, which nothing under ``verity/`` outside their own package
#: names (an import, an attribute or a string), each with the module and why a verifier must not read it.
PROVER_ONLY = {
    "trace_cache": ("verity/definitions/ops/boolean/trace_cache.py",
                    "traced circuits reused across a prover's builds: a verifier reading them would take a stale or tampered "
                    "entry for a Definition's gates"),
}

MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}


def _sources(root: Path = SRC) -> list[Path]:
    """``verity``'s own Python files: every ``.py`` under ``root`` outside a ``tests/`` or ``lean/`` directory, a Lake build
    directory (a Lean package builds in its ``.lake``) or a distribution of its own (the listing's ``distribution``), and
    no circuit-check sidecar (``<module>.circuit-check.py``: circuit-check's bindings, which it loads by path and no
    module imports)."""
    listing = _listing()
    return sorted(p for p in root.rglob("*.py")
                  if not {"tests", "lean", ".lake", "__pycache__"} & set(p.relative_to(SRC).parts)
                  and _distribution(p.relative_to(ROOT).as_posix(), listing) is None and not p.name.endswith(SIDECAR))


def _imports(path: Path, src: Path = SRC, package: tuple[str, ...] = ("verity",),
             nodes: Iterable[ast.AST] | None = None) -> list[tuple[str, int, bool]]:
    """``(module, line, imported)`` for every module ``path`` (or its parsed ``nodes``) names.  Statements and import-call
    strings are ``imported``; a ``"module:attr"`` string is not, since any ``word:word`` constant matches: only the
    forbidden and direction rules apply to it."""
    package = [*package, *path.relative_to(src).parts[:-1]]
    out = []
    for node in nodes if nodes is not None else ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            out += [(a.name, node.lineno, True) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append((node.module, node.lineno, True))
        elif isinstance(node, ast.ImportFrom) and node.level > len(package):
            out.append(("." * node.level + (node.module or ""), node.lineno, True))
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1]
            out += [(".".join([*base, head]), node.lineno, True) for head in ([node.module] if node.module else [a.name for a in node.names])]
        elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in IMPORT_CALLS):
            out.append((node.args[0].value, node.lineno, True))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_ATTR.fullmatch(node.value):
            out.append((node.value.partition(":")[0], node.lineno, False))
    return out


def _external(name: str, imported: bool, allowed: set[str]) -> bool:
    """Whether a non-``verity`` ``name`` breaks the rules: it is forbidden and not ``allowed``, or it is imported and is
    neither stdlib nor in ``allowed``."""
    top = name.split(".")[0]
    return (top in FORBIDDEN_EXTERNAL and top not in allowed) or (
        imported and top != "verity" and top not in sys.stdlib_module_names and top not in allowed)


#: A read of another package at run time, matched in the text: its package data (``importlib.resources``) or a module an import
#: call names by a string, an f-string included (`_imports` sees only a whole constant handed to an import call).
RUN_TIME_READ = re.compile(r"""\b(?:files|read_text|read_binary|open_text|open_binary|import_module|__import__|find_spec)\("""
                           r"""\s*[rRbBuUfF]{0,2}["'](?P<name>[A-Za-z_]\w*)""")


def _run_time_reads(text: str) -> list[tuple[int, str]]:
    """``(line, name)`` for each read in ``text`` that `RUN_TIME_READ` matches and whose top-level name is ``FORBIDDEN_EXTERNAL``."""
    return [(text.count("\n", 0, m.start()) + 1, m["name"]) for m in RUN_TIME_READ.finditer(text) if m["name"] in FORBIDDEN_EXTERNAL]


def test_verity_reads_no_other_distribution_at_run_time() -> None:
    """No Python file under ``verity/`` (tests, Lean scripts and sidecars included; a directory of the listing's
    ``distribution`` is a distribution of its own) reads another distribution's package data or names one of its modules
    for an import call, so ``verity`` alone evaluates its Definitions and no suite of it takes another distribution as an
    input."""
    offenders, listing = [], _listing()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if {".lake", "__pycache__"} & set(path.parts) or _distribution(rel, listing) is not None:
            continue
        offenders += [f"{rel}:{line}: {name}" for line, name in _run_time_reads(path.read_text(encoding="utf-8"))]
    assert not offenders, ("verity reads another distribution at run time, and an integration may import verity but never the "
                           "reverse: move the data or code into verity beside what reads it (as `bf16_gemm_check`'s device pins "
                           "sit in its `device_pins/`), or take it as a parameter the caller supplies\n" + "\n".join(offenders))


def test_a_run_time_read_is_found() -> None:
    """The check above isn't vacuous: package data, and a module named by a string or an f-string, are each a read of their
    distribution; ``verity``'s own package data is not."""
    src = ('P = resources.files("verity_vllm") / "pins"\nm = importlib.import_module("verity_vllm.pins")\n'
           "n = import_module(f'verity_vllm.{x}')\no = __import__('research')\n"
           'q = resources.files("verity.definitions.ops") / "tables"\nr = resources.files(__package__)\n')
    assert _run_time_reads(src) == [(1, "verity_vllm"), (2, "verity_vllm"), (3, "verity_vllm"), (4, "research")]


def test_every_source_directory_is_a_package() -> None:
    """``verity`` is one regular package: every directory of its own Python has an ``__init__.py``, so no part of it resolves
    as a namespace package from whatever else is on the path.  Tests and Lean scripts are run by path and stay out."""
    missing = sorted({str(p.parent.relative_to(SRC.parent)) for p in _sources() for d in [p.parent, *p.parent.parents]
                      if d.is_relative_to(SRC) and not (d / "__init__.py").exists()})
    assert not missing, missing


def _under(rel: str, paths: Iterable[str]) -> bool:
    """Whether ``rel`` is one of ``paths`` or lies under one."""
    return any(rel == p or rel.startswith(p + "/") for p in paths)


def test_verity_imports_the_standard_library_its_dependencies_and_itself() -> None:
    """Python of ``verity`` (``_sources``) names no other distribution of the workspace, no external runtime and no
    directory of the repository (``FORBIDDEN_EXTERNAL``), in an import or in a string, and imports only the standard
    library, ``verity`` and its declared dependencies (``DEPENDENCIES``).  A verifier's and a device's Python and the
    directories of ``STDLIB_ONLY`` import no dependency either, but for ``NUMPY_USERS``; each listed path still holds
    Python of ``verity``."""
    offenders, held = [], set()
    for path in _sources():
        rel = path.relative_to(ROOT).as_posix()
        held |= {p for p in (*STDLIB_ONLY, *NUMPY_USERS) if _under(rel, [p])}
        alone = (_trusted(rel) is not None or _under(rel, STDLIB_ONLY)) and not _under(rel, NUMPY_USERS)
        offenders += [f"{rel}:{line}: {name}" for name, line, imported in _imports(path)
                      if _external(name, imported, set() if alone else DEPENDENCIES)]
    assert not offenders, ("verity imports the standard library, its declared dependencies and itself, and a verifier, a "
                           "device and the scalar references (STDLIB_ONLY) the standard library and verity alone:\n"
                           + "\n".join(offenders))
    gone = sorted({*STDLIB_ONLY, *NUMPY_USERS} - held)
    assert not gone, f"paths of STDLIB_ONLY or NUMPY_USERS that hold no Python of verity's own; remove them: {gone}"


def _covered(rel: str, globs: Iterable[str]) -> bool:
    """Whether ``rel`` is a path ``globs`` name or lies under one (``*`` crosses directories, as ``fnmatch``'s does)."""
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel, g + "/*") for g in globs)


def _spot_python_offenders() -> list[str]:
    """Each import of a certifier module (``CERTIFIER_MODULES``) in a Python file of ``SPOT_TREES`` (a tree, or one file), its
    tests included."""
    out = []
    files = {p for tree in SPOT_TREES for p in ([ROOT / tree] if (ROOT / tree).is_file() else (ROOT / tree).rglob("*.py"))}
    for path in sorted(files):
        if {".lake", "__pycache__"} & set(path.relative_to(ROOT).parts):
            continue
        out += [f"{path.relative_to(ROOT)}:{line}: {name}" for name, line, _ in _imports(path)
                if any(name == m or name.startswith(m + ".") for m in CERTIFIER_MODULES)]
    return out


def test_spot_never_imports_the_network_certifier_in_python() -> None:
    """SPOT[Flock]'s Python imports none of the network certifier.  The listing puts the certifier's row
    (``CERTIFIER_ROW``) after SPOT's (``SPOT_ROW``) and lists no tie out of the rows up to SPOT's into
    a later one, so nothing SPOT's Python loads, directly or through another file, is the certifier's; and no
    file of ``SPOT_TREES``, Flock's and their tests among them, imports one of ``CERTIFIER_MODULES``."""
    listing = _listing()
    spot, certifier = listing.steps[SPOT_ROW], listing.steps[CERTIFIER_ROW]
    assert spot < certifier, f"{SPOT_NOT_CERTIFIER}: tests/import_order.toml lists {CERTIFIER_ROW} no later than {SPOT_ROW}"
    ties = sorted(f"{a} -> {b}" for a, b in listing.against if listing.steps[a] <= spot < listing.steps[b])
    assert not ties, (f"{SPOT_NOT_CERTIFIER}: a tie of tests/import_order.toml leaves the rows up to {SPOT_ROW} for a later "
                      f"row, which may load the certifier: {ties}")
    offenders = _spot_python_offenders()
    assert not offenders, f"{SPOT_NOT_CERTIFIER}:\n" + "\n".join(offenders)


def _lean_modules(files: Iterable[str], lakefiles: Iterable[str]) -> dict[str, list[str]]:
    """The tracked Lean ``files`` by module name: each one's path from its Lake package's ``srcDir``, its package the
    deepest directory of ``lakefiles`` above it.  Two files of one name are both kept; fixtures are left out."""
    roots = {}
    for lake in lakefiles:
        pkg = Path(lake).parent
        roots[pkg] = Path(posixpath.normpath(pkg / tomllib.loads((ROOT / lake).read_text()).get("srcDir", ".")))
    out: dict[str, list[str]] = {}
    for f in files:
        p = Path(f)
        owner = max((d for d in roots if d in p.parents), key=lambda d: len(d.parts), default=None)
        if owner is not None and "fixtures" not in p.parts:
            out.setdefault(".".join(p.relative_to(roots[owner]).with_suffix("").parts), []).append(f)
    return out


def _lean_reach(roots: Iterable[str], imports_of: Callable[[str], Iterable[str]],
                files_of: dict[str, list[str]]) -> dict[str, str | None]:
    """Each file ``roots`` reach through their imports (``imports_of``: a file's modules, ``files_of``: a module's files),
    with the file that first imported it, None for a root."""
    parent: dict[str, str | None] = dict.fromkeys(roots)
    todo = list(parent)
    while todo:
        f = todo.pop()
        for m in imports_of(f):
            for g in files_of.get(m, ()):
                if g not in parent:
                    parent[g] = f
                    todo.append(g)
    return parent


def _chain(parent: dict[str, str | None], f: str) -> str:
    out = [f]
    while (p := parent[out[-1]]) is not None:
        out.append(p)
    return " <- ".join(out)


def _lake_closure(pkgs: Iterable[str]) -> set[str]:
    """``pkgs`` and every package they require by path, directly or through one another (paths from the root)."""
    seen, todo = set(), list(pkgs)
    while todo:
        pkg = todo.pop()
        lake = ROOT / pkg / "lakefile.toml"
        if pkg in seen or not lake.is_file():
            continue
        seen.add(pkg)
        todo += [(ROOT / pkg / r["path"]).resolve().relative_to(ROOT).as_posix()
                 for r in tomllib.loads(lake.read_text()).get("require", []) if "path" in r]
    return seen


def _load_tool(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spot_never_imports_the_network_certifier_in_lean() -> None:
    """SPOT[Flock]'s Lean imports none of the network certifier's (``CERTIFIER_LEAN``): its packages (``SPOT_LEAN``)
    require no certifier package by path, directly or through another, and no module of those packages or of its analysis
    (``SPOT_SECURITY``) imports a certifier module, directly or through any other module."""
    pkgs = sorted(p for p in _lake_closure(SPOT_LEAN) if _covered(p, CERTIFIER_LEAN))
    assert not pkgs, f"{SPOT_NOT_CERTIFIER}: {SPOT_LEAN} require {pkgs}"
    headers = _load_tool("lean_headers", ROOT / "tools/verity/lean/headers.py")
    files = _tracked("*.lean")
    roots = [f for f in files if _covered(f, (*SPOT_LEAN, *SPOT_SECURITY))]
    assert roots, f"SPOT_LEAN and SPOT_SECURITY name no Lean file: {SPOT_LEAN + SPOT_SECURITY}"
    parent = _lean_reach(roots, lambda f: headers.imports((ROOT / f).read_text(encoding="utf-8", errors="replace")) or (),
                         _lean_modules(files, _tracked("*lakefile.toml")))
    offenders = [_chain(parent, f) for f in sorted(parent) if _covered(f, CERTIFIER_LEAN)]
    assert not offenders, (f"{SPOT_NOT_CERTIFIER}: take what you need from SPOT's own modules or the shared vocabulary "
                           "(Core, Hash), never from the certifier's:\n" + "\n".join(offenders))


def test_a_spot_import_of_the_certifier_is_found() -> None:
    """The check above isn't vacuous: a certifier module that a SPOT root reaches through another module is found with the
    chain that reaches it, a module no file declares (Mathlib's) is skipped, and a package's ``srcDir`` names its modules."""
    files_of = {"B": ["s/B.lean"], "C": ["verity/protocols/certifier/device/program/lean/C.lean"]}
    imports = {"s/A.lean": ["Mathlib.X", "B"], "s/B.lean": ["C"]}
    parent = _lean_reach(["s/A.lean"], lambda f: imports.get(f, ()), files_of)
    found = [_chain(parent, f) for f in sorted(parent) if _covered(f, CERTIFIER_LEAN)]
    assert found == ["verity/protocols/certifier/device/program/lean/C.lean <- s/B.lean <- s/A.lean"], found
    assert _covered("security/Proofs/Properties/NetworkCertifier/Check/Bytes.lean", CERTIFIER_LEAN)
    assert not _covered("security/Proofs/Flock/Bytes.lean", CERTIFIER_LEAN)
    modules = _lean_modules(["security/Proofs/Flock/Bytes.lean", "verity/protocols/flock/verifier/Flock/Bytes.lean"],
                            ["security/Proofs/lakefile.toml", "verity/protocols/flock/verifier/lakefile.toml"])
    assert modules == {"Proofs.Flock.Bytes": ["security/Proofs/Flock/Bytes.lean"],
                       "Flock.Bytes": ["verity/protocols/flock/verifier/Flock/Bytes.lean"]}, modules


def _lake_requires() -> dict[str, list[str]]:
    """Each tracked lakefile's package with the packages it requires, by name; fixtures are left out."""
    out: dict[str, list[str]] = {}
    for lake in _tracked("*lakefile.toml"):
        if "fixtures" not in Path(lake).parts:
            doc = tomllib.loads((ROOT / lake).read_text())
            out[doc["name"]] = [r["name"] for r in doc.get("require", [])]
    return out


def _base_line_offenders(requires: dict[str, list[str]]) -> list[str]:
    """What crosses the base packages' line in ``requires`` ({package: the packages it requires}): a package outside
    ``REQUIRES_FLOCK_VERIFIER`` that requires Flock's verifier, an entry of that list that no longer does, and a base package
    (``BASE_LEAN``) that no lakefile declares or that requires more than its entry allows."""
    out = [f"{pkg} requires {FLOCK_VERIFIER} and REQUIRES_FLOCK_VERIFIER, which only shrinks, doesn't list it: require the "
           f"base package that holds what it reads ({', '.join(BASE_LEAN)}), or move what it reads into one"
           for pkg, needs in sorted(requires.items()) if FLOCK_VERIFIER in needs and pkg not in REQUIRES_FLOCK_VERIFIER]
    out += [f"{pkg} no longer requires {FLOCK_VERIFIER}: remove it from REQUIRES_FLOCK_VERIFIER"
            for pkg in sorted(REQUIRES_FLOCK_VERIFIER) if FLOCK_VERIFIER not in requires.get(pkg, ())]
    for base, allowed in BASE_LEAN.items():
        if base not in requires:
            out.append(f"BASE_LEAN names {base}, which no lakefile declares")
        elif extra := sorted(set(requires[base]) - set(allowed)):
            out.append(f"{base} is a base package and requires {', '.join(extra)}: it may require "
                       f"{', '.join(allowed) or 'nothing'} alone, so what needs more goes in a package above it")
    return out


def test_only_the_listed_lean_packages_require_flocks_verifier() -> None:
    """Over every lakefile's requirements: Flock's verifier is required only by the packages whose Lean reads the verifier
    itself (``REQUIRES_FLOCK_VERIFIER``, which only shrinks), and the base packages below it (``BASE_LEAN``) require nothing
    above them, so the circuit format, the commitments and the bytes below them reach a package without the proof system."""
    requires = _lake_requires()
    assert FLOCK_VERIFIER in requires, f"no lakefile declares {FLOCK_VERIFIER}: FLOCK_VERIFIER names the wrong package"
    offenders = _base_line_offenders(requires)
    assert not offenders, "\n".join(offenders)


def test_a_requirement_across_the_base_line_is_found() -> None:
    """The check above isn't vacuous: a new requirer of Flock's verifier, a listed one that stopped, and a base package
    that requires a package above it or one outside the tree are each found, and the lists as they stand cross nothing."""
    held = {FLOCK_VERIFIER: list(BASE_LEAN), **{pkg: [FLOCK_VERIFIER] for pkg in REQUIRES_FLOCK_VERIFIER},
            **{base: list(allowed) for base, allowed in BASE_LEAN.items()}}
    assert _base_line_offenders(held) == []
    listed = min(REQUIRES_FLOCK_VERIFIER)
    crossed = {**held, "core_identity": ["core_base", FLOCK_VERIFIER], listed: ["core_base"], "core_base": ["mathlib"],
               "core_circuits": ["core_base", FLOCK_VERIFIER]}
    found = _base_line_offenders(crossed)
    assert [line.split(" ", 1)[0] for line in found] == ["core_circuits", "core_identity", listed, "core_base",
                                                         "core_circuits"], found
    assert _base_line_offenders({pkg: needs for pkg, needs in held.items() if pkg != "core_commitments"}) == [
        "BASE_LEAN names core_commitments, which no lakefile declares"]


def _workspace_packages() -> set[str]:
    """The import packages of the workspace's distributions other than ``verity``, from each member's wheel configuration."""
    members = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]["workspace"]["members"]
    return {Path(p).name for member in members
            for p in tomllib.loads((ROOT / member / "pyproject.toml").read_text(encoding="utf-8")).get("tool", {})
            .get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", [])}


def _circuit_check_imports(texts: dict[str, str], packages: Set[str]) -> dict[str, set[str]]:
    """{file: the packages among ``packages`` it names} for circuit-check's files ``texts`` that name one: in an import, an
    import call's string or a ``"module:attr"`` string."""
    found: dict[str, set[str]] = {}
    for rel, text in texts.items():
        names = _imports(ROOT / rel, ROOT / CIRCUIT_CHECK, (), ast.walk(ast.parse(text, filename=rel)))
        if hits := {name.split(".")[0] for name, _line, _imported in names} & packages:
            found[rel] = hits
    return found


def test_circuit_check_imports_no_workspace_distribution_but_verity() -> None:
    """A check that needs another distribution's code would make ``verity``'s tool depend on what attaches to it: an
    integration, a benchmark, the team's tooling.  The dependency points the other way, through a sidecar."""
    packages = _workspace_packages() - {"circuit_check"}
    assert {"verity_vllm", "verity_numerical", "verity_flock", "research"} <= packages, packages
    texts = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in _tracked(f"{CIRCUIT_CHECK}/*.py")}
    assert len(texts) > 10, sorted(texts)
    found = _circuit_check_imports(texts, packages)
    offenders = [f"{f}: {pkg}" for f, pkgs in sorted(found.items()) for pkg in sorted(pkgs - CIRCUIT_CHECK_IMPORTS.get(f, set()))]
    assert not offenders, ("circuit-check imports `verity` and itself, and no other distribution of the workspace. Have the "
                           "distribution hand circuit-check what the check needs, in a sidecar it owns beside the module that "
                           "defines it (`<module>.circuit-check.py`: a binding in `circuit_check.targets`' forms, or a reader "
                           "as `request_program`; `circuit_check.sidecars` finds it without naming the package), or, if the "
                           "check is the distribution's own, run it from that distribution's command and suite. "
                           "CIRCUIT_CHECK_IMPORTS only shrinks.\n" + "\n".join(offenders))
    stale = sorted(f"{f}: {pkg}" for f, pkgs in CIRCUIT_CHECK_IMPORTS.items() for pkg in pkgs - found.get(f, set()))
    assert not stale, f"CIRCUIT_CHECK_IMPORTS entries circuit-check no longer imports; remove them: {stale}"


def test_a_circuit_check_import_of_another_distribution_is_found() -> None:
    """The check above isn't vacuous: a statement inside a function, an import call's string and a ``"module:attr"`` string
    are each found; ``verity``, circuit-check's own modules and a third-party package are not."""
    texts = {
        f"{CIRCUIT_CHECK}/circuit_check/a.py": ("def check():\n    from verity_vllm.query import required\n"
                                                "    import verity_numerical.models as M\n"),
        f"{CIRCUIT_CHECK}/circuit_check/b.py": ("import importlib\nimportlib.import_module('verity_flock.ir_lower')\n"
                                                "LOAD = 'research.cli:main'\n"),
        f"{CIRCUIT_CHECK}/circuit_check/c.py": ("import numpy\nfrom verity.circuits import defs\nfrom . import cache\n"
                                                "from .targets import family\nNOTE = 'verity_vllm binds the vocabulary'\n"),
    }
    found = _circuit_check_imports(texts, {"verity_vllm", "verity_numerical", "verity_flock", "research"})
    assert found == {f"{CIRCUIT_CHECK}/circuit_check/a.py": {"verity_vllm", "verity_numerical"},
                     f"{CIRCUIT_CHECK}/circuit_check/b.py": {"verity_flock", "research"}}, found
def _benchmarks_packages() -> set[str]:
    """The import names ``benchmarks/`` provides: its distributions' wheel packages, the packages (directories with an
    ``__init__.py``) on its suites' ``pythonpath`` roots inside it, and ``benchmarks`` itself, a namespace from the
    repository root.  A bare module on such a root (``run``, ``inputs``, ``pous``) is too common a name to match by name;
    reaching one takes a path into ``benchmarks/``, which ``_path_loads`` finds."""
    out = {"benchmarks"}
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "benchmarks/*pyproject.toml"], cwd=ROOT, check=True,
                             capture_output=True).stdout
    for rel in (p for p in tracked.decode().split("\0") if p):
        base = (ROOT / rel).parent
        tool = tomllib.loads((ROOT / rel).read_text(encoding="utf-8")).get("tool", {})
        out |= {Path(p).name for p in tool.get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", [])}
        for entry in tool.get("pytest", {}).get("ini_options", {}).get("pythonpath", []):
            if (root := (base / entry).resolve()).is_relative_to(ROOT / "benchmarks"):
                out |= {init.parent.name for init in root.glob("*/__init__.py")}
    return out


#: What reaches a module by its path: a call that loads one from a file or puts a directory on the import path.  ``sys.path``
#: itself (its ``insert``, ``append`` and ``extend``, or an assignment to it) is matched apart.
PATH_LOADERS = {"spec_from_file_location", "SourceFileLoader", "run_path", "addsitedir"}
#: Calls whose result is a path built from their arguments.
PATH_CALLS = {"Path", "PurePath", "str", "fspath", "join", "joinpath", "abspath", "realpath", "normpath", "expanduser"}
BENCHMARKS_PATH = re.compile(r"(?:^|[/\\])benchmarks(?:[/\\]|$)")


def _into_benchmarks(e: ast.AST, names: Set[str] = frozenset(), constants: Set[str] = frozenset()) -> bool:
    """Whether ``e`` is a path into ``benchmarks/``: a string naming the directory, one of ``names``, an attribute read of
    one of ``constants`` (another module's), or a path built from one (``/``, ``+``, an f-string, ``Path``, ``str``,
    ``os.path.join``, ``.parent``, ``.parents[2]``, ``.resolve()``).  A container is not a path."""
    if isinstance(e, ast.Constant):
        return isinstance(e.value, str) and bool(BENCHMARKS_PATH.search(e.value))
    if isinstance(e, ast.Name):
        return e.id in names
    if isinstance(e, ast.Attribute):
        return e.attr in constants or _into_benchmarks(e.value, names, constants)
    parts = (e.value,) if isinstance(e, ast.Subscript) else (e.left, e.right) if isinstance(e, ast.BinOp) else (
        (e.body, e.orelse) if isinstance(e, ast.IfExp) else ())
    if isinstance(e, ast.JoinedStr):
        parts = tuple(v.value if isinstance(v, ast.FormattedValue) else v for v in e.values)
    elif isinstance(e, ast.Call):
        f = e.func
        parts = ((f.value,) if isinstance(f, ast.Attribute) else ()) + (
            tuple(e.args) if (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)) in PATH_CALLS else ())
    return any(_into_benchmarks(p, names, constants) for p in parts)


def _assigned(nodes: Iterable[ast.AST]) -> list[tuple[str, ast.AST]]:
    """``(name, value)`` for each name ``nodes`` bind from an expression: an assignment, a walrus, a parameter's default."""
    out: list[tuple[str, ast.AST]] = []
    for n in nodes:
        if isinstance(n, ast.Assign):
            out += [(t.id, n.value) for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and n.value is not None and isinstance(n.target, ast.Name):
            out.append((n.target.id, n.value))
        elif isinstance(n, ast.arguments):
            positional = [*n.posonlyargs, *n.args]
            out += [(a.arg, d) for a, d in zip(positional[len(positional) - len(n.defaults):], n.defaults)]
            out += [(a.arg, d) for a, d in zip(n.kwonlyargs, n.kw_defaults) if d is not None]
    return out


def _bound_from(assigned: list[tuple[str, ast.AST]], seeds: Set[str], source: Callable[[ast.AST, Set[str]], bool]) -> set[str]:
    """``seeds`` and the names of ``assigned`` (``_assigned``) bound to an expression ``source`` accepts given the names
    found so far.  Scopes are not told apart: a name bound so anywhere in a file counts everywhere in it."""
    names = set(seeds)
    while grown := {name for name, value in assigned if name not in names and source(value, names)}:
        names |= grown
    return names


def _sys_path(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "path" and isinstance(node.value, ast.Name) and node.value.id == "sys"


def _path_loads(nodes: Iterable[ast.AST], reaches: Callable[[ast.AST], bool], loaders: Set[str]) -> list[tuple[int, str]]:
    """``(line, code)`` for each of ``nodes`` that puts a path ``reaches`` accepts on ``sys.path`` or loads a module from
    it, directly or through one of ``loaders``, the functions that hand a parameter to such a load."""
    out = []
    for n in nodes:
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            on_path = name in {"insert", "append", "extend"} and isinstance(f, ast.Attribute) and _sys_path(f.value)
            hit = on_path or name in PATH_LOADERS or name in loaders
            args = [*n.args, *(k.value for k in n.keywords)]
        elif isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            on_path = hit = any(_sys_path(t.value if isinstance(t, ast.Subscript) else t) for t in targets)
            args = [n.value]
        else:
            continue
        if on_path:
            args = [x for a in args for x in (a.elts if isinstance(a, (ast.List, ast.Tuple, ast.Set)) else [a])]
        if hit and any(reaches(a) for a in args):
            out.append((n.lineno, ast.unparse(n).splitlines()[0][:120]))
    return out


def _benchmarks_importers(texts: dict[str, str], packages: Set[str]) -> dict[str, list[str]]:
    """Each file of ``texts`` (repository path -> source) outside ``benchmarks/`` that imports a module of
    ``benchmarks/``, with where: an import of one of ``packages`` (a statement, an
    import-call string or a ``"module:attr"`` string), or a path into ``benchmarks/`` (``_into_benchmarks``) put on
    ``sys.path`` or loaded by file.  A path is traced through the file's own names and parameter defaults, through other
    files' upper-case module constants read as attributes (``remote.REMOTE_SESSION``), and through any function, wherever it
    is defined, that hands a parameter to such a load (``remote.session_module``).  Nothing is executed (AST only), and
    only files a text search can't rule out are parsed."""
    alt = "|".join(map(re.escape, sorted(packages)))
    named = [re.compile(rf"""(?:import|from)\s+(?:{alt})\b|(?:{"|".join(IMPORT_CALLS)})\(\s*["'](?:{alt})\b"""),
             re.compile(rf"""["'](?:{alt})(?:\.\w+)*[.:]\w+["']""")]
    quoted = re.compile(r"""["'][^"'\n]*\bbenchmarks\b[^"'\n]*["']""")
    loads = ("sys.path", *PATH_LOADERS)
    ours = {rel: text for rel, text in texts.items() if not rel.startswith("benchmarks/")}
    trees: dict[str, ast.AST] = {}
    constants: set[str] = set()

    def parse(rel: str) -> ast.AST:
        try:
            return ast.parse(texts[rel], filename=rel)
        except SyntaxError:
            return ast.Module(body=[], type_ignores=[])

    def into(e: ast.AST, names: Set[str] = frozenset()) -> bool:
        return _into_benchmarks(e, names, constants)

    by_name = {rel for rel, text in ours.items() if any(p in text for p in packages) and any(r.search(text) for r in named)}
    new = sorted(by_name | {rel for rel, text in ours.items() if "benchmarks" in text and quoted.search(text)})
    nodes: dict[str, list[ast.AST]] = {}
    assigned: dict[str, list[tuple[str, ast.AST]]] = {}
    while True:
        for rel in new:
            trees[rel] = parse(rel)
            nodes[rel] = list(ast.walk(trees[rel]))
            assigned[rel] = _assigned(nodes[rel])
        bound = {rel: frozenset(_bound_from(assigned[rel], set(), into)) for rel in trees}
        grown = {t.id for rel, tree in trees.items() for s in tree.body if isinstance(s, (ast.Assign, ast.AnnAssign))
                 for t in (s.targets if isinstance(s, ast.Assign) else [s.target])
                 if isinstance(t, ast.Name) and t.id.isupper() and t.id in bound[rel]} - constants
        if not grown:
            break
        constants |= grown
        attr = re.compile(r"\.(?:" + "|".join(sorted(grown)) + r")\b")
        new = [rel for rel, text in ours.items() if rel not in trees and attr.search(text)]
    handed = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
              for rel in trees for n in nodes[rel]
              if isinstance(n, ast.Call) and any(into(a, bound[rel]) for a in [*n.args, *(k.value for k in n.keywords)])}
    handed -= {None, *PATH_LOADERS, *dir(builtins)}
    defines = re.compile(r"^([ \t]*)(?:async[ \t]+)?def (?:" + "|".join(map(re.escape, sorted(handed))) + r")\(", re.M)

    def loads_in_body(text: str, m: re.Match[str]) -> bool:
        end = re.compile(rf"^[ \t]{{0,{len(m.group(1))}}}[^\s)#]", re.M).search(text, text.find("\n", m.end()) + 1)
        return any(w in text[m.end():end.start() if end else None] for w in loads)

    loaders = set()
    for rel, text in texts.items() if handed else ():
        if not (any(w in text for w in loads) and any(loads_in_body(text, m) for m in defines.finditer(text))):
            continue
        for fn in ast.walk(trees.get(rel) or parse(rel)):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name in handed - loaders:
                body = list(ast.walk(fn))
                params = _bound_from(_assigned(body), {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
                                     - {"self", "cls"}, _into_benchmarks)
                if _path_loads(body, lambda e, p=params: _into_benchmarks(e, p), set()):
                    loaders.add(fn.name)
    found = {}
    for rel in trees:
        hits = [(line, name) for name, line, _ in (_imports(ROOT / rel, ROOT, (), nodes[rel]) if rel in by_name else ())
                if name.split(".")[0] in packages]
        hits += _path_loads(nodes[rel], lambda e, n=bound[rel]: into(e, n), loaders)
        if hits:
            found[rel] = [f"{line}: {what}" for line, what in sorted(hits)]
    return found


def _tracked(*globs: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *globs], cwd=ROOT, check=True, capture_output=True).stdout
    return [rel for rel in out.decode().split("\0") if rel]


def _tracked_python() -> dict[str, str]:
    return {rel: (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for rel in _tracked("*.py") if (ROOT / rel).is_file()}


def test_nothing_outside_benchmarks_imports_it() -> None:
    """``benchmarks/`` is measurement only: no file outside it imports a module
    of it, by name or by path, apart from ``BENCHMARKS_IMPORTERS``, which only shrinks."""
    packages = _benchmarks_packages()
    assert {"verity_bench", "verity_numerical"} <= packages, packages
    found = _benchmarks_importers(_tracked_python(), packages)
    offenders = [f"{f}:{hit}" for f, hits in found.items() if f not in BENCHMARKS_IMPORTERS for hit in hits]
    assert not offenders, ("benchmarks/ is measurement only, and nothing outside it imports it: move the code you need out "
                           "of benchmarks/ into the package that uses it, or take the number through a card file citing its "
                           "art: id. BENCHMARKS_IMPORTERS only shrinks.\n" + "\n".join(offenders))
    stale = sorted(set(BENCHMARKS_IMPORTERS) - set(found))
    assert not stale, f"BENCHMARKS_IMPORTERS entries that no longer import benchmarks/; remove them: {stale}"


def test_a_benchmarks_import_is_found() -> None:
    """The check above isn't vacuous: imports by name, a path put on ``sys.path`` through a name, a parameter's default
    from another file's constant, and a call to another file's loader (whose load follows a nested function) are each
    found; a path into another directory, a mere mention of ``benchmarks/``, and ``benchmarks/``'s own files are not."""
    texts = {
        "x/names.py": ("import verity_bench.vllm\nfrom benchmarks.tc_probe import tool\nimport importlib\n"
                       "importlib.import_module('datacenter.graph')\nTOOL = 'benchmarks.commitments.tool:COMMIT_COST'\n"),
        "x/syspath.py": ("import sys\nfrom pathlib import Path\nPPL = Path(__file__).parents[2] / 'benchmarks' / 'ppl'\n"
                         "sys.path.insert(0, str(PPL))\nsys.path[:0] = ['../../benchmarks/rho']\n"),
        "x/config.py": "from pathlib import Path\nSESSION = Path('/r') / 'benchmarks' / 'remote_session'\n",
        "x/loader.py": ("import importlib.util\nfrom x import config\ndef load(name, directory=config.SESSION):\n"
                        "    def named(n):\n        return n\n"
                        "    return importlib.util.spec_from_file_location(named(name), directory / f'{name}.py')\n"),
        "x/caller.py": "from x import config, loader\nloader.load('paged', config.SESSION.parent / 'vllm_encoded')\n",
        "x/clean.py": ("import sys\nsys.path.insert(0, 'tools/verity/check')\nimport suites\n"
                       "NOTE = 'the numbers come from benchmarks/ppl'\nimport verity_vllm\n"),
        "benchmarks/y/run.py": "import verity_bench\nimport sys\nsys.path.insert(0, '../benchmarks/z')\n",
    }
    found = _benchmarks_importers(texts, {"benchmarks", "verity_bench", "datacenter"})
    assert found == {
        "x/names.py": ["1: verity_bench.vllm", "2: benchmarks.tc_probe", "4: datacenter.graph", "5: benchmarks.commitments.tool"],
        "x/syspath.py": ["4: sys.path.insert(0, str(PPL))", "5: sys.path[:0] = ['../../benchmarks/rho']"],
        "x/loader.py": ["6: importlib.util.spec_from_file_location(named(name), directory / f'{name}.py')"],
        "x/caller.py": ["2: loader.load('paged', config.SESSION.parent / 'vllm_encoded')"],
    }, found


def _is_test(rel: str) -> bool:
    parts = Path(rel).parts
    return "tests" in parts[:-1] or parts[-1].startswith("test_") or parts[-1] == "conftest.py"


def _uses(tree: ast.AST, names: set[str]) -> list[tuple[str, int]]:
    """``(name, line)`` for each use of one of ``names``: a name, an attribute, an imported name, or a string equal to one
    (what ``getattr`` reads).  Defining it is not a use."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            out.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in names:
            out.append((node.attr, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            out += [(a.name, node.lineno) for a in node.names if a.name in names]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in names:
            out.append((node.value, node.lineno))
    return out


def test_only_tests_call_a_function_past_a_guard() -> None:
    """No tracked Python file but a test (under a ``tests/`` directory, a ``test_*.py`` or a ``conftest.py``) uses a
    function of ``TEST_ONLY``, its own file included; each is still defined there, and some test still uses it."""
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT, check=True, capture_output=True).stdout
    word = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(TEST_ONLY))) + r")\b")
    offenders, used, defined = [], set(), set()
    for rel in (p for p in tracked.decode().split("\0") if p):
        path = ROOT / rel
        if not path.is_file() or not word.search(text := path.read_text(encoding="utf-8", errors="replace")):
            continue
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError:
            continue
        hits = _uses(tree, set(TEST_ONLY))
        if _is_test(rel):
            used |= {name for name, _ in hits}
            continue
        offenders += [f"{rel}:{line}: {name}" for name, line in hits]
        defined |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and TEST_ONLY.get(n.name, ("",))[0] == rel}
    assert not offenders, ("only tests may use these (TEST_ONLY), since a guard never fails open: call the guarded function\n"
                           + "\n".join(offenders))
    assert defined == set(TEST_ONLY), f"TEST_ONLY entries their file no longer defines; remove them: {sorted(set(TEST_ONLY) - defined)}"
    assert used == set(TEST_ONLY), f"TEST_ONLY functions no test uses; remove them and the entry: {sorted(set(TEST_ONLY) - used)}"


def test_a_use_past_a_guard_is_found() -> None:
    """The check above isn't vacuous: an import, an attribute call and a ``getattr`` string are each a use, and the
    definition is not."""
    src = ("from verity.circuits.partition_v2 import _build_unchecked as b\nV._build_unchecked(P)\n"
           "getattr(V, '_build_unchecked')(P)\ndef _build_unchecked(p):\n    return p\n")
    assert sorted(line for _name, line in _uses(ast.parse(src), {"_build_unchecked"})) == [1, 2, 3]
    assert _is_test("verity/circuits/tests/test_partition_v2.py") and not _is_test("verity/circuits/partition_v2.py")


def test_only_its_own_package_names_a_prover_only_module() -> None:
    """No Python file under ``verity/`` (tests and distributions of its own included) names a module of ``PROVER_ONLY``
    from outside the directory one level below its row of the listing (``_below``: its package, with its tests), and each
    still exists.  So
    no verifier and no device names one, and neither does anything they load."""
    steps = _listing().steps
    homes = {name: _below(f, steps) for name, (f, _why) in PROVER_ONLY.items()}
    word = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(PROVER_ONLY))) + r")\b")
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if {".lake", "__pycache__"} & set(path.parts) or not word.search(text := path.read_text(encoding="utf-8")):
            continue
        rel = path.relative_to(ROOT).as_posix()
        hits = _uses(ast.parse(text, filename=rel), set(PROVER_ONLY))
        hits += [(part, line) for name, line, _ in _imports(path) for part in set(name.split(".")) & set(PROVER_ONLY)]
        offenders += [f"{rel}:{line}: {name}" for name, line in hits if _below(rel, steps) != homes[name]]
    assert not offenders, ("only its own package names a prover-only module (PROVER_ONLY), so that no verifier reads "
                           "one:\n" + "\n".join(offenders))
    missing = [f for f, _why in PROVER_ONLY.values() if not (ROOT / f).is_file()]
    assert not missing, f"PROVER_ONLY entries whose module is gone; remove them: {missing}"


def _docstrings(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


CENSUS_PATH = re.compile(r"(?<!verity/)census[/\\]|\b(hardware|networks|workloads|instance_sets)(\.schema)?\.json\b")


def test_core_never_reads_the_census() -> None:
    """The census (``benchmarks/census/``, data only) is read by the renderer, benchmarks, tools and the website, never by
    ``verity``: it takes such numbers as parameters.  No string in its code (docstrings aside) names the directory or one of
    its files; a Program's census schema (``verity/census/v1``, ``ml.circuits.census``) is not the directory."""
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docs = _docstrings(tree)
        offenders += [f"{path.relative_to(SRC)}:{n.lineno}" for n in ast.walk(tree)
                      if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and CENSUS_PATH.search(n.value)]
    assert not offenders, offenders


#: The listing: the tree's directories, bottom first; the directories under ``verity/`` that are distributions of their own;
#: the ties against the order; and the places of ``verity``'s own Python, each with the places it loads and, for the
#: verifiers' side, what it reaches and what loads it.
ORDER = ROOT / "tests" / "import_order.toml"
#: The command that writes the listing's generated tables from the tree, and the line of the listing they start at: what
#: is above it (the order, the distributions) is kept by hand, as are each place's ``paths`` and its ``trusted`` mark.
REGENERATE = "`uv run tools/verity/check/import_order.py`"
GENERATED = "# The ties against the order"
#: The files the order reads as loaders: Python, Lean, and a Lake package's requirements.
LOADERS = (".py", ".lean", "lakefile.toml")
#: What it leaves out as loaders: circuit-check's bindings, which it loads by path beside their module, as it leaves the tests.
SIDECAR = ".circuit-check.py"


class Load(NamedTuple):
    """One file loading another."""
    loader: str
    loaded: str
    line: int
    #: ``python import`` (a statement or an import-call string), ``python string`` (a ``"module:attr"`` string), ``python
    #: package`` (the ``__init__.py`` of a package above an imported module), ``lean import`` or ``lake require``
    kind: str
    name: str

    def __str__(self) -> str:
        return f"{self.loader}:{self.line}: {self.kind} {self.name} loads {self.loaded}"


#: The lists of a place's entry in the listing: every place's ``loads``, and a trusted place's ``reaches`` and ``loaded_by``.
LISTS = ("loads", "reaches", "loaded_by")
#: The kinds of a load that name the loaded file itself, an import of the module or a string of it; the third kind is
#: the package above a named module.
NAMING = ("python import", "python string")


class Listing(NamedTuple):
    """``ORDER`` as the tests read it."""

    steps: dict[str, int]                               # each listed directory's step: the bottom row's 0, one a row
    distributions: dict[str, dict[str, str]]            # each distribution of its own under ``verity/``, by directory
    against: dict[tuple[str, str], set[str]]            # each excepted pair of rows: the first's files that load the second
    #: each place of ``verity``'s own Python, by its first path: its ``paths``, whether the listing marks it ``trusted``,
    #: the places it ``loads`` and, for a trusted one, ``reaches``, and the files it is ``loaded_by``
    places: dict[str, dict[str, Any]]


def _places(entries: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The listing's ``place`` entries by name, a place's name being its first path."""
    return {e["paths"][0]: {"paths": tuple(e["paths"]), "trusted": e.get("trusted", False), **{k: set(e.get(k, ())) for k in LISTS}}
            for e in entries}


def _listing(text: str | None = None) -> Listing:
    """The listing (``ORDER``, or ``text``)."""
    doc = tomllib.loads(ORDER.read_text(encoding="utf-8") if text is None else text)
    steps = {d: n for n, row in enumerate(doc["order"]) for d in row}
    assert "." in steps and len(steps) == sum(map(len, doc["order"])), "the order lists `.` and each directory once"
    return Listing(steps, {e["directory"]: {k: v for k, v in e.items() if k != "directory"} for e in doc.get("distribution", [])},
                   {(e["loader"], e["loaded"]): set(e["files"]) for e in doc.get("against", [])}, _places(doc.get("place", [])))


def _row(path: str, steps: dict[str, int]) -> str:
    """The listed directory ``path`` belongs to: the deepest one above it, ``.`` for the rest."""
    d = posixpath.dirname(path)
    while d and d not in steps:
        d = posixpath.dirname(d)
    return d or "."


def _below(path: str, steps: dict[str, int]) -> str:
    """Where a file in no place is, as a place's lists name it: the directory one level below its row that holds it, or
    its row for a file directly in it."""
    row = _row(path, steps)
    below = posixpath.dirname(path)[0 if row == "." else len(row) + 1:].split("/")[0]
    return row if not below else below if row == "." else f"{row}/{below}"


def _place(path: str, places: dict[str, dict[str, Any]]) -> str | None:
    """The place ``path`` is in: the one a path of which is the longest at or above it, so that a file listed by itself
    in another place's directory is its own place's; None for a file in no place."""
    owner = {p: name for name, e in places.items() for p in e["paths"]}
    parts = path.split("/")
    return next((owner[p] for n in range(len(parts), 0, -1) if (p := "/".join(parts[:n])) in owner), None)


def _part(path: str, name: str) -> str | None:
    """The outermost directory above ``path`` named ``name`` (``verifier``, ``prover``), or None."""
    parts = path.split("/")[:-1]
    return "/".join(parts[:parts.index(name) + 1]) if name in parts else None


def _trusted(path: str) -> str | None:
    """The verifier's or the device's directory ``path`` is in: the outermost ``verifier/`` or ``device/`` directory above
    it under ``verity/``, or None."""
    parts = path.split("/")
    at = next((i for i, part in enumerate(parts[:-1]) if part in ("verifier", "device")), None)
    return None if at is None or parts[0] != "verity" else "/".join(parts[:at + 1])


def _trusted_place(entry: dict[str, Any]) -> bool:
    """Whether a place is the verifiers' side: its name, its first path, is in a ``verifier/`` or a ``device/``
    directory, or the listing marks it ``trusted`` (what the verifiers and the devices stand on outside those
    directories, and the Lean of their side with its tests and scripts).  A path a place holds besides is its own
    place's side: a Definition the provers' side keeps in a verifier's directory is the provers'."""
    return entry["trusted"] or _trusted(entry["paths"][0] + "/_") is not None


def _distribution(path: str, listing: Listing) -> str | None:
    """The distribution of its own that ``path`` is in: a directory of the listing's ``distribution`` above it and in its
    row, so that a row listed below a distribution (Flock's verifier, below Flock) is no part of it."""
    row = _row(path, listing.steps)
    return next((d for d in listing.distributions if path.startswith(d + "/") and _row(d + "/_", listing.steps) == row), None)


def _lean_script(path: str) -> bool:
    """Whether ``path`` is a script beside a Lean package: Python under a ``lean/`` directory, which writes the Lean's test
    vectors from the Python it is held to, is run by path and is no part of what decides a verdict."""
    return path.endswith(".py") and "lean" in path.split("/")[:-1]


def _source(path: str, listing: Listing) -> bool:
    """Whether ``path`` is Python of the distribution ``verity``: under ``verity/``, in no distribution of its own, and no
    script beside a Lean package.  (Tests and circuit-check's bindings are no loaders at all.)"""
    return path.startswith("verity/") and path.endswith(".py") and not _lean_script(path) and _distribution(path, listing) is None


@functools.cache
def _public_boundary() -> Any:
    """``tools/verity/check/public_boundary.py``, whose ``Tree`` resolves a Python import and a Lake package as the public
    boundary reads them.  Loaded on first use, since it loads this file for ``_imports``."""
    return _load_tool("public_boundary_of_the_order", ROOT / "tools/verity/check/public_boundary.py")


def _path_roots(tree: Any, f: str, parsed: ast.AST, text: str) -> tuple[str, ...]:
    """Where a bare import of ``f`` resolves beyond its package's and its suite's roots: its own directory (a script's), and,
    when it changes ``sys.path``, each tracked directory a path anchored on ``__file__`` reaches in it
    (``tools/verity/check/anchors.py``), whether or not that path is the one put there."""
    roots = {posixpath.dirname(f)}
    if "sys.path" in text and "__file__" in text:
        anchors = _public_boundary().ANCHORS
        names = anchors.bindings(parsed, f)
        roots |= set(names.values())
        todo = [parsed]
        while todo:
            node = todo.pop()
            if not isinstance(node, ast.Name) and (reached := anchors.anchor(node, f, names)) is not None:
                roots.add(reached)
            else:
                todo += ast.iter_child_nodes(node)
    return tuple(sorted(r for r in roots if r in tree.dirs))


@functools.cache
def _named(tree: Any, name: str, directory: str, roots: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The tracked files that importing ``name`` from a file of ``directory`` runs: its module, and the ``__init__.py`` of
    each package above it that the name spells."""
    modules = tuple(t for t in tree._module(name, posixpath.join(directory, "_.py"), roots) if t in tree.files)
    above = []
    for hit in modules:
        parts, at = name.split("."), hit.removesuffix("/__init__.py").removesuffix(".py").split("/")
        spelled = next((k for k in range(min(len(parts), len(at)), 0, -1) if at[-k:] == parts[:k]), 0)
        above += [init for j in range(1, spelled) if (init := "/".join(at[:len(at) - spelled + j]) + "/__init__.py") in tree.files]
    return modules, tuple(above)


def _naming_nodes(parsed: ast.AST) -> list[ast.AST]:
    """The nodes of ``parsed`` that can name a module (the four kinds ``_imports`` reads), in the order of their lines.
    One pass with no generator, since the order reads every Python file of the tree."""
    out, todo = [], [parsed]
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call, ast.Constant)):
            out.append(node)
        for field in node._fields:
            value = getattr(node, field, None)
            if type(value) is list:
                todo += [v for v in value if isinstance(v, ast.AST)]
            elif isinstance(value, ast.AST):
                todo.append(value)
    return sorted(out, key=lambda n: (n.lineno, n.col_offset))


def _python_loads(tree: Any, f: str) -> list[Load]:
    """What the Python file ``f`` loads, at load or inside a function: for each module it names (``_imports``; a relative
    name resolved from the root), the files ``_named`` gives, and for ``from M import a`` the module ``M.a`` when it is
    one.  Where every name a ``from M import ...`` takes is a module of ``M``, the file names those modules and ``M`` is
    the package above them."""
    text = tree.text(f)
    parsed = ast.parse(text, filename=f)
    path, directory, roots = tree.root / f, posixpath.dirname(f), _path_roots(tree, f, parsed, text)
    nodes = _naming_nodes(parsed)
    froms = [n for n in nodes if isinstance(n, ast.ImportFrom) and n.module]
    rest = [n for n in nodes if not (isinstance(n, ast.ImportFrom) and n.module)]
    names = [(name, line, "python import" if imported else "python string")
             for name, line, imported in _imports(path, tree.root, (), rest)]
    for n, (name, line, _) in zip(froms, _imports(path, tree.root, (), froms)):
        modules = [f"{name}.{a.name}" for a in n.names]
        whole = all(_named(tree, module, directory, roots)[0] for module in modules)
        names += [(name, line, "python package" if whole else "python import"), *((module, line, "python import") for module in modules)]
    out: dict[tuple[str, str], Load] = {}
    for name, line, named in sorted(names, key=lambda n: n[1]):
        modules, above = _named(tree, name, directory, roots)
        for kind, files in ((named, modules), ("python package", above)):
            for t in files:
                if t != f:
                    out.setdefault((t, kind), Load(f, t, line, kind, name))
    return list(out.values())


@functools.cache
def _lake_sources(tree: Any) -> dict[str, tuple[str, ...]]:
    """By lakefile, where a Lean file of its package finds a module: the package's own source directory, then those of the
    packages it requires by path, directly or through one another."""
    pb = _public_boundary()
    _, _, pkgs = tree.lake
    out = {}
    for lf in pkgs:
        reached, todo = [lf], [lf]
        while todo:
            for e in tree._lakefile(todo.pop()):
                if e.kind == "lake require" and e.target in pkgs and e.target not in reached:
                    reached.append(e.target)
                    todo.append(e.target)
        out[lf] = tuple(dict.fromkeys(pb._rel(posixpath.relpath(pkgs[r].src, tree.root)) for r in reached))
    return out


@functools.cache
def _lean_module(tree: Any, lakefile: str | None, module: str) -> tuple[str, ...]:
    """The tracked files a Lean file of ``lakefile``'s package finds ``module`` in: under its sources (``_lake_sources``),
    else under any tracked package's own source directory, test fixtures aside, since a file may be built against a
    package its own doesn't require (a benchmark's ``Main.lean``, a submission checked against the analysis)."""
    sources = _lake_sources(tree)
    path = posixpath.join(*_public_boundary().HEADERS.parts(module)) + ".lean"
    found = [t for src in sources.get(lakefile, ()) if (t := posixpath.join(src, path)) in tree.files]
    found = found or [t for lf, srcs in sources.items() if not _is_test(lf) and (t := posixpath.join(srcs[0], path)) in tree.files]
    return tuple(dict.fromkeys(found))


def _lean_loads(tree: Any, f: str) -> list[Load]:
    """What the Lean file ``f`` imports among the tracked files: each module of its header that its own package or a
    package it requires by path holds; Mathlib's and the toolchain's are not tracked."""
    headers = _public_boundary().HEADERS
    text = tree.text(f)
    header, starts = headers.header(text), headers.starts(text)
    assert header is not None and starts is not None, f"{f}: its header is cut off, so what it imports can't be read"
    out: dict[str, Load] = {}
    for imp, at in zip(header.imports, starts):
        for t in _lean_module(tree, tree.lake[1].get(f), imp.module):
            if t != f:
                out.setdefault(t, Load(f, t, text.count("\n", 0, at) + 1, "lean import", imp.module))
    return list(out.values())


def _loads(tree: Any, f: str) -> list[Load]:
    """What the tracked file ``f`` loads among the tracked files: a Python file's imports, a Lean file's, and for a
    ``lakefile.toml`` the lakefile of each package it requires by path."""
    if f.endswith(".py"):
        return _python_loads(tree, f)
    if f.endswith(".lean"):
        return _lean_loads(tree, f)
    return [Load(f, e.target, e.line, e.kind, e.what) for e in tree._lakefile(f) if e.kind == "lake require"]


def _all_loads(tree: Any) -> tuple[Load, ...]:
    """Every load of ``tree``'s Python, Lean and Lake files; of its tests, the Python ones'."""
    return tuple(x for f in sorted(tree.files) if f.endswith(LOADERS) and (f.endswith(".py") or not _is_test(f))
                 for x in _loads(tree, f))


def _held(x: Load) -> bool:
    """Whether the order reads ``x``: its loader is no test (``_is_test``: a suite of its own, which declares what it
    reads) and no circuit-check binding (``SIDECAR``)."""
    return not _is_test(x.loader) and not x.loader.endswith(SIDECAR)


@functools.cache
def _tree_loads() -> tuple[frozenset[str], tuple[Load, ...], frozenset[str]]:
    """The tracked tree's directories, its loads (``_all_loads``) and its files, read once for the tests below."""
    tree = _public_boundary().Tree(ROOT)
    return frozenset(tree.dirs), _all_loads(tree), frozenset(tree.files)


def _against(loads: Iterable[Load], steps: dict[str, int]) -> dict[tuple[str, str], dict[str, Load]]:
    """The ties against the order among ``loads``, by pair of rows, the first listed before the second: each file of the
    first that loads a file of the second, with one such load (a module's before a package's above one).  A test's and a
    binding's loads count only into the last row, which nothing outside it loads."""
    last = max(steps.values())
    out: dict[tuple[str, str], dict[str, Load]] = {}
    for x in sorted(loads, key=lambda x: x.kind == "python package"):
        a, b = _row(x.loader, steps), _row(x.loaded, steps)
        if steps[a] < steps[b] and (_held(x) or steps[b] == last):
            out.setdefault((a, b), {}).setdefault(x.loader, x)
    return out


def _hold_order(loads: Iterable[Load], listing: Listing) -> None:
    found = _against(loads, listing.steps)
    new = [f"{x} ({a} -> {b})" for (a, b), files in sorted(found.items())
           for f, x in sorted(files.items()) if f not in listing.against.get((a, b), ())]
    assert not new, ("a file loads only what its own row or an earlier row of tests/import_order.toml holds: move what the "
                     "loader needs down to its row or an earlier one, or pass it in.  `against` only shrinks, so a new tie "
                     "is cut, never listed:\n" + "\n".join(new))
    done = [f"{a} -> {b}" for (a, b), files in sorted(listing.against.items()) if not files & set(found.get((a, b), ()))]
    assert not done, f"pairs of `against` (tests/import_order.toml) no file needs any more; delete them: {done}"
    gone = [f"{a} -> {b}: {f}" for (a, b), files in sorted(listing.against.items()) for f in sorted(files - set(found[a, b]))]
    assert not gone, ("files of `against` (tests/import_order.toml) that no longer load a file of the later row; delete "
                      "them:\n" + "\n".join(gone))


def _to_provers(loads: Iterable[Load]) -> list[Load]:
    """The loads of a file under a ``prover/`` by a file under a ``verifier/``, scripts beside a verifier's Lean aside."""
    return [x for x in loads if _held(x) and _part(x.loader, "verifier") and _part(x.loaded, "prover") and not _lean_script(x.loader)]


def _into_distributions(loads: Iterable[Load], listing: Listing) -> list[Load]:
    """The loads of a file of a distribution of its own by Python of ``verity`` (``_source``)."""
    return [x for x in loads if _held(x) and _source(x.loader, listing) and _distribution(x.loaded, listing) is not None]


def _place_loads(loads: Iterable[Load], listing: Listing) -> dict[str, tuple[dict[str, Load], dict[str, list[str]], dict[str, Load]]]:
    """For each place of the listing: the places its Python (``_source``) names, each with one such load; for a trusted
    place (``_trusted_place``), the places it reaches, each with the chain to it, and the Python files under
    ``verity/`` that name a file of it from an untrusted place or from outside every place, verifier and device, a
    test or a binding included, each with one such load.  A file names what it imports or spells in a string
    (``NAMING``), not the package above it.  A place reaches what the files it loads name in turn, and so on, through
    every load, the packages above a named module included (a chain of files); and what the places it loads may load
    in turn, by their own ``loads``, and so on (a chain of places), so that a place on a verifier's path gains a load
    only as the verifier's ``reaches`` does.  A named file in no place is named by the directory one level below its
    row (``_below``), but ``verity``'s own Python in no place, which holds no load, is not named at all.  A load by
    file is read only into ``benchmarks/``."""
    owner = {p: name for name, e in listing.places.items() for p in e["paths"]}

    @functools.cache
    def place(f: str) -> str | None:
        parts = f.split("/")
        return next((owner[p] for n in range(len(parts), 0, -1) if (p := "/".join(parts[:n])) in owner), None)

    source = functools.cache(lambda f: _source(f, listing))
    trusted = {name for name, e in listing.places.items() if _trusted_place(e)}
    by: dict[str, list[Load]] = {}
    loaders: dict[str, dict[str, Load]] = {}
    for x in loads:
        if x.loader.endswith(".py"):
            if _held(x):
                by.setdefault(x.loader, []).append(x)
            if (x.kind in NAMING and x.loader.startswith("verity/") and place(x.loaded) in trusted
                    and place(x.loader) not in trusted and (place(x.loader) is not None or _trusted(x.loader) is None)):
                loaders.setdefault(place(x.loaded), {}).setdefault(x.loader, x)
    own: dict[str, list[str]] = {}
    for f in sorted(by):
        if source(f) and place(f) is not None:
            own.setdefault(place(f), []).append(f)
    def named(d: str, x: Load) -> str | None:
        """The place a load by a file of ``d`` names outside it, or None."""
        inside = place(x.loaded) == d and _distribution(x.loaded, listing) is None
        return None if x.kind not in NAMING or inside or (place(x.loaded) is None and source(x.loaded)) else (
            place(x.loaded) or _below(x.loaded, listing.steps))

    direct: dict[str, dict[str, Load]] = {d: {} for d in sorted(own.keys() | loaders.keys())}
    for d, files in own.items():
        for x in (x for f in files for x in by[f]):
            if (at := named(d, x)) is not None:
                direct[d].setdefault(at, x)
    out = {}
    for d in direct:
        reach: dict[str, list[str]] = {}
        if d in trusted:
            files = own.get(d, [])
            via: dict[str, str | None] = dict.fromkeys(files)
            todo = files[::-1]
            while todo:
                f = todo.pop()
                for x in by.get(f, ()):
                    if x.loaded not in via:
                        via[x.loaded] = f
                        todo.append(x.loaded)
            for f in via:
                for x in by.get(f, ()):
                    if (at := named(d, x)) is not None and at not in reach:
                        chain = [f]
                        while (before := via[chain[-1]]) is not None:
                            chain.append(before)
                        reach[at] = [*chain[::-1], x.loaded]
            through = {d: [d]}
            todo = [d]
            while todo:
                at = todo.pop(0)
                for after in direct.get(at, ()):
                    if after not in through:
                        through[after] = [*through[at], after]
                        todo.append(after)
                        reach.setdefault(after, through[after])
        out[d] = (direct[d], reach, loaders.get(d, {}) if d in trusted else {})
    return out


def _hold_places(loads: Iterable[Load], files: Set[str], listing: Listing) -> None:
    paths = [p for e in listing.places.values() for p in e["paths"]]
    twice = sorted({p for p in paths if paths.count(p) > 1})
    assert not twice, f"paths of `place` (tests/import_order.toml) that two places name; a path is one place's: {twice}"
    found = _place_loads(loads, listing)
    listed = lambda d, kind: listing.places.get(d, {}).get(kind, ())  # noqa: E731
    trusted = {d for d, e in listing.places.items() if _trusted_place(e)}
    grown = [f"{d} loads {at}: {x}" for d, (direct, _, _) in found.items() if d in trusted
             for at, x in sorted(direct.items()) if at not in listed(d, "loads")]
    grown += [f"{d} reaches {at}: {' -> '.join(chain)}" for d, (_, reach, _) in found.items()
              for at, chain in sorted(reach.items()) if at not in listed(d, "reaches")]
    assert not grown, ("a verifier, a device or what they stand on gains no load, and what it reaches gains no place (a "
                       "trusted `place` of tests/import_order.toml only shrinks): state it in Lean, or take the value as "
                       "an argument:\n" + "\n".join(grown))
    new = [f"{d} is loaded by {x}" for d, (_, _, loaders) in found.items()
           for f, x in sorted(loaders.items()) if f not in listed(d, "loaded_by")]
    assert not new, ("under verity/, outside the trusted places, only the files a trusted place's `loaded_by` lists name "
                     "a file of it (`place` in tests/import_order.toml), and the list only shrinks: move what the file "
                     "needs to a place both load, or pass it in:\n" + "\n".join(new))
    added = [f"{d} loads {at}: {x}" for d, (direct, _, _) in found.items() if d not in trusted
             for at, x in sorted(direct.items()) if at not in listed(d, "loads")]
    assert not added, ("a place of verity loads only the places its `loads` lists (`place` in tests/import_order.toml): "
                       "pass the value in, or, where the place is meant to stand on the other, add the line and say "
                       f"why in the pull request ({REGENERATE} writes it):\n" + "\n".join(added))
    stale = [f"{d}: {kind} {at}" for d, lists in sorted(listing.places.items())
             for kind, now in zip(LISTS, found.get(d, ({}, {}, {}))) for at in sorted(lists[kind] - set(now))]
    stale += [f"{d}: no Python is at {p}" for d, lists in sorted(listing.places.items()) for p in lists["paths"]
              if not any(f.endswith(".py") and (f == p or f.startswith(p + "/")) for f in files)]
    assert not stale, (f"lines of `place` (tests/import_order.toml) the tree no longer has; delete them ({REGENERATE} "
                       f"does): {stale}")


#: What the generated tables say of themselves in the listing.
AGAINST_HEADER = """\
# The ties against the order: for a pair of rows, the first listed before the second, each file of the first that loads a
# file of the second.  It only shrinks: the test refuses a file not listed here (cut the tie instead: move what the
# loader needs down to its row or an earlier one, or pass it in), a listed file that no longer loads the later row, and
# a pair no file needs any more, which it names to delete."""
PLACE_HEADER = """\
# Which folder may import which: the places of `verity`'s own Python.  A place is the directories and files its `paths`
# name, and its first path is its name; a file belongs to the place whose path is the longest above it, so a file listed
# by itself in another place's directory is its own place's.  A place's Python names only files of its own place and of
# the places its `loads` lists: an import, at load or inside a function, or a module named in a string.  To let a place
# import another, add the line to its `loads` with the import; a line no file needs any more is deleted.  Python of
# `verity` in no place only groups packages and imports the standard library alone, and every place may load it.
#
# A place is the verifiers' side, trusted, when its name is under a `verifier/` or a `device/` directory, or it is
# marked `trusted = true`: what the verifiers and the devices stand on outside those directories, and the Lean of
# their side with its tests and scripts.  The rest is the provers' side.  A trusted place's `loads` only shrinks, and it
# has two lists more that only shrink: `reaches`, what the files it loads name in turn and what the places it loads
# may load in turn, and so on, the soundness path of its Python until its Lean decides its verdicts; and `loaded_by`,
# the Python files under `verity/` that name a file of it from an untrusted place or from outside every place,
# verifier and device, tests included.  A file outside every place is named by the directory one level below its row."""


def _generated(text: str, loads: Iterable[Load], grow: bool = True) -> str:
    """The listing ``text`` with its generated tables written from ``loads``: ``against``, and each place's ``loads``,
    ``reaches`` and ``loaded_by``.  What is above the tables is kept, and so are each place's ``paths`` and ``trusted``
    mark.  Without ``grow`` the lists that only shrink gain nothing: ``against`` and a trusted place's lists keep the
    lines the tree still has."""
    listing = _listing(text)
    array = lambda items: "[\n" + "".join(f'    "{x}",\n' for x in items) + "]"  # noqa: E731
    out = [text[:text.index(GENERATED)].rstrip("\n"), "", AGAINST_HEADER]
    found = _against(loads, listing.steps)
    for (a, b), files in sorted(found.items(), key=lambda pair: (listing.steps[pair[0][0]], listing.steps[pair[0][1]])):
        if lines := sorted(f for f in files if grow or f in listing.against.get((a, b), ())):
            out += ["[[against]]", f'loader = "{a}"', f'loaded = "{b}"', f"files = {array(lines)}", ""]
    out.append(PLACE_HEADER)
    placed = _place_loads(loads, listing)
    for d, entry in sorted(listing.places.items()):
        keep = None if grow or not _trusted_place(entry) else entry
        out += ["[[place]]", f"paths = {array(entry['paths'])}", *(["trusted = true"] if entry["trusted"] else [])]
        out += [f"{kind} = {array(lines)}" for kind, items in zip(LISTS, placed.get(d, ({}, {}, {})))
                if (lines := sorted(x for x in items if keep is None or x in keep[kind]))]
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _hold_distributions(dirs: Set[str], loads: Iterable[Load], listing: Listing,
                        neither: Iterable[str] = NO_DATE_OR_REASON) -> None:
    has = {d: e.keys() & {"date", "reason"} for d, e in listing.distributions.items()}
    form = [d for d, e in sorted(listing.distributions.items())
            if not (d.startswith("verity/") and e.get("owner") and len(has[d]) < 2
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", e.get("date", "2000-01-01")))]
    assert not form, ("each `distribution` of tests/import_order.toml is a directory under verity/ with its owner and "
                      "exactly one of the date it becomes part of `verity` (YYYY-MM-DD) and the reason it stays its "
                      f"own: {form}")
    owes = sorted(d for d in has if not has[d] and d not in set(neither))
    assert not owes, ("a `distribution` of tests/import_order.toml names the date it becomes part of `verity` or the "
                      "reason it stays its own; NO_DATE_OR_REASON (tests/test_boundaries.py) holds the ones that name "
                      f"neither, and only shrinks: give each its date or its reason: {owes}")
    named = sorted(d for d in neither if has.get(d, True))
    assert not named, ("names of NO_DATE_OR_REASON (tests/test_boundaries.py) whose `distribution` has its date or its "
                       f"reason, or is no longer listed; delete them: {named}")
    gone = sorted(set(listing.distributions) - dirs)
    assert not gone, f"directories of `distribution` (tests/import_order.toml) that hold no tracked file; delete them: {gone}"
    loaded = _into_distributions(loads, listing)
    assert not loaded, ("`verity` loads no other distribution, one under verity/ included (`distribution` in "
                        "tests/import_order.toml): move what both read into `verity`, or pass it in:\n" + "\n".join(map(str, loaded)))


def test_a_file_loads_only_its_own_row_or_an_earlier_one() -> None:
    """The import order.  ``tests/import_order.toml`` lists the tree's directories bottom first; a file belongs to the
    deepest listed directory above it, and loads only files of its own row or an earlier one: in Python (an import at
    load or inside a function, a module named in a string, the packages above the imported module, a bare import under a
    directory the file puts on ``sys.path``), in Lean (an import) and in Lake (a package required by path).  The ties
    against it are the listing's ``against``: for a pair of rows, each file of the first that loads the second.  It only
    shrinks: a file it doesn't list fails, and so does a listed file that no longer loads the later row, naming the pair to
    delete once its last file is gone.  Reading every tracked Python and Lean file is most of this file's time, which
    nothing cheaper replaces: an order over the whole tree is only held by reading the whole tree."""
    listing = _listing()
    dirs, loads, _ = _tree_loads()
    empty = sorted(d for d in listing.steps if d != "." and d not in dirs)
    assert not empty, f"directories tests/import_order.toml lists that hold no tracked file; delete their rows: {empty}"
    _hold_order(loads, listing)


def test_a_verifier_loads_no_prover_code() -> None:
    """No file under a ``verifier/`` directory loads a file under a ``prover/`` directory, with no exception.  It is an
    assertion of its own and no consequence of the order, which lists a protocol's verifier and its prover in one row.
    The form: a verifier is what the tree has under ``verifier/``, Lean and Python alike, so a protocol's Python verifier
    is held as its Lean is until its Lean decides its verdicts; a file changes sides by moving between ``verifier/`` and
    ``prover/``, and what both sides read is the verifier's, which the prover loads.  Every load by name counts: at load,
    inside a function, a module named in a string, a Lean import, a Lake requirement; a load by file is read only into
    ``benchmarks/``.  Tests are no loaders here, and neither are the scripts beside a verifier's Lean (``_lean_script``)."""
    found = _to_provers(_tree_loads()[1])
    assert not found, ("a verifier loads no prover code: move what both sides read into the verifier's directory, where "
                       "the prover loads it, or state it in Lean:\n" + "\n".join(map(str, found)))


def test_a_place_loads_only_the_places_the_listing_names() -> None:
    """Which folder may import which.  The listing's ``place`` entries are the places of ``verity``'s own Python, each
    the directories and files its ``paths`` name (a file belongs to the place whose path is the longest above it), and a
    place's Python names only files of its own place and of the places its ``loads`` lists: an import at load or inside
    a function, a module named in a string.  The lists are the tree's own: a load they don't name fails, and so does a
    listed place no file names any more.

    A place is the verifiers' side, trusted, when its name is under a ``verifier/`` or a ``device/`` directory or the
    listing marks it: the wire, the deployment layer, the Site and what else the verifiers and the devices stand on,
    and the Lean of their side with its tests and scripts.  Until a protocol's Lean verifier decides its verdicts, its
    Python verifier does, and that Python and everything it loads is on the soundness path.  A trusted place's
    ``loads`` only shrinks, and it has two lists more that only shrink: ``reaches``, what the files it loads name in
    turn, through every load, the packages above a named module included, and what the places it loads may load in
    turn, so that a place on a verifier's path gains a load only as the verifier's ``reaches`` does; and
    ``loaded_by``, the Python files under ``verity/`` that name a file of it from an untrusted place or from outside
    every place, verifier and device, a prover's and a test's alike.  A place or a file the tree adds fails, a place
    with the chain that reaches it.  What the trusted places reach outside themselves is the soundness path, and
    AGENTS.md names each place of it."""
    listing = _listing()
    _, loads, files = _tree_loads()
    _hold_places(loads, files, listing)
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    trusted = {d for d, e in listing.places.items() if _trusted_place(e)}
    path = {at for d in trusted for at in listing.places[d]["reaches"]} - trusted
    missing = sorted(at for at in path if f"`{at}`" not in agents and f"`{at}/`" not in agents)
    assert not missing, f"AGENTS.md doesn't name {missing}, which the verifiers' side reaches: the soundness path"


def test_a_file_in_no_place_imports_the_standard_library_alone() -> None:
    """Python of ``verity`` in no place of the listing is a file directly under ``verity/`` (``verity/__init__.py``
    re-exports nothing; ``errors`` and ``settings`` are shared by every place) or an ``__init__.py`` that only groups
    packages, and imports the standard library and nothing else, ``verity`` included.  So a new package is given its
    place before it loads anything, and every place may load these files.  The ``__init__.py`` that makes a
    distribution of its own a package of ``verity`` (one in every directory from ``verity/`` down to it) is held as
    they are."""
    listing = _listing()
    packages = [p for p in SRC.rglob("__init__.py")
                if not {"tests", "lean", ".lake", "__pycache__"} & set(p.relative_to(SRC).parts)
                and _distribution(p.relative_to(ROOT).as_posix(), listing) is not None
                and all((d / "__init__.py").exists() for d in p.parents if d.is_relative_to(SRC))]
    offenders = []
    for path in sorted({*_sources(), *packages}):
        rel = path.relative_to(ROOT).as_posix()
        if _place(rel, listing.places) is not None and path not in packages:
            continue
        if path.name != "__init__.py" and path.parent != SRC:
            offenders.append(f"{rel}: in no place")
        offenders += [f"{rel}:{line}: {name}" for name, line, imported in _imports(path)
                      if _external(name, imported, set()) or name.split(".")[0] == "verity"]
    assert not offenders, ("Python of verity in no `place` of tests/import_order.toml only groups packages and imports the "
                           "standard library alone: give a new package its place, with what it loads, or move what the "
                           "file needs out of it:\n" + "\n".join(offenders))


def test_the_listing_is_what_its_generator_writes() -> None:
    """The listing's tables are generated: ``against`` and each place's lists are byte for byte what the generator
    writes from the tree, so a change to what a file loads is made in the listing by running it, and the listing's diff
    shows what the change opens."""
    text = ORDER.read_text(encoding="utf-8")
    assert _generated(text, _tree_loads()[1]) == text, (
        f"tests/import_order.toml is not what its generator writes from the tree: run {REGENERATE} and read its diff")


def test_verity_loads_no_distribution_of_its_own() -> None:
    """A directory under ``verity/`` with an import root of its own is a distribution of its own (the listing's
    ``distribution``, each with its owner and exactly one of the date it becomes part of ``verity`` and the reason it
    stays its own; ``NO_DATE_OR_REASON`` holds the entries that still name neither, and only shrinks), and ``verity``
    loads none of them by name, a verifier's Python least of all: no Python of ``verity`` (``_source``) loads a file of
    one, by its own name or as a module of ``verity``.  A load by file is read only into ``benchmarks/``.  The list only
    shrinks: a listed directory with no tracked file fails, and a directory not listed is ``verity``'s, held to its
    rules."""
    dirs, loads, _ = _tree_loads()
    _hold_distributions(dirs, loads, _listing())


#: A tree with one load of each kind the order reads, for the test below.
ORDER_TREE = {
    "verity/__init__.py": "",
    "verity/circuits/__init__.py": "",
    "verity/circuits/at_load.py": "from verity.definitions import ops\n",
    "verity/circuits/in_function.py": "def f():\n    import verity.definitions.ops\n",
    "verity/circuits/in_string.py": "TOOL = 'verity.definitions.ops:run'\n",
    "verity/circuits/tests/test_a.py": "import verity.definitions.ops\n",
    "verity/circuits/at_load.circuit-check.py": "import verity.definitions.ops\n",
    "verity/commitments/__init__.py": "",
    "verity/commitments/crypto/__init__.py": "",
    "verity/commitments/crypto/lms.py": "",
    "verity/definitions/__init__.py": "",
    "verity/definitions/ops.py": "import verity.circuits.at_load\n",
    "verity/protocols/flock/__init__.py": "",
    "verity/protocols/flock/prover/stage.py": "import verity.protocols.pouw.verifier.check\n",
    "verity/protocols/flock/verifier/__init__.py": "",
    "verity/protocols/flock/verifier/interface.py": "",
    "verity/protocols/flock/verifier/lakefile.toml": ('name = "flock"\n[[lean_lib]]\nname = "Flock"\n'
                                            '[[require]]\nname = "spot"\npath = "../../spot/lean"\n'),
    "verity/protocols/flock/verifier/Flock/V.lean": "import Spot.X\nimport Mathlib.Y\nimport Flock.W\n",
    "verity/protocols/flock/verifier/Flock/W.lean": "",
    "verity/configuration/package_above.py": "import verity.protocols.flock.verifier.interface\n",
    "verity/protocols/spot/lean/lakefile.toml": 'name = "spot"\n[[lean_lib]]\nname = "Spot"\n',
    "verity/protocols/spot/lean/Spot/X.lean": "",
    "verity/protocols/spot/by_path.py": ("import sys\nfrom pathlib import Path\n"
                               "sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'demos' / 'e2e'))\nimport prof\n"),
    "demos/e2e/prof.py": "",
    "tools/lazy.py": "def f():\n    from benchmarks.probe import run\n",
    "benchmarks/probe/run.py": "import tools.lazy\n",
    "benchmarks/probe/tests/test_run.py": "import infra.cluster.usage\n",
    "infra/cluster/usage.py": "import benchmarks.probe.run\n",
    "infra/ops/pyproject.toml": '[tool.hatch.build.targets.wheel]\npackages = ["src/verity_ops"]\n',
    "infra/ops/src/verity_ops/__init__.py": "",
    "tools/tests/test_console.py": "import verity_ops.no_such_module\n",
    "verity/protocols/__init__.py": "",
    "verity/protocols/pouw/__init__.py": "",
    "verity/protocols/pouw/verifier/__init__.py": "",
    "verity/protocols/pouw/verifier/check.py": ("from verity.circuits import at_load\nfrom verity.commitments.crypto import lms\n\n\n"
                                                "def f():\n    from verity.protocols.pouw.prover import build\n"),
    "verity/protocols/pouw/verifier/lean/scripts/vectors.py": "import verity.protocols.pouw.prover.build\n",
    "verity/protocols/pouw/verifier/tests/test_check.py": "import verity.protocols.pouw.prover.build\n",
    "verity/protocols/pouw/verifier/check/interop.py": "import verity.protocols.flock.prover.stage\n",
    "verity/protocols/pouw/device/__init__.py": "",
    "verity/protocols/pouw/device/serve.py": "from verity.protocols.pouw.verifier import check\n",
    "verity/protocols/pouw/device/party.py": "from verity.protocols.pouw.device import serve\n",
    "verity/protocols/pouw/prover/__init__.py": "",
    "verity/protocols/pouw/prover/build.py": "from verity.protocols.pouw.verifier import check\n",
    "verity/protocols/pouw/prover/tests/test_build.py": "from verity.protocols.pouw.device import serve\n",
}


def test_the_listing_checks_find_what_they_refuse(tmp_path: Path) -> None:
    """The checks above aren't vacuous.  Against the listing's order: an import at load, inside a function and as a
    module named in a string, the package above a module of an earlier row, a bare import under a directory put on
    ``sys.path``, a Lean import from a package required by path and that requirement, and a test's import of the last
    row, by its path and by the name of a package one of its distributions ships, whatever module of it is named; not a
    load with the order, a test's of another row, a sidecar's, or a module no tracked file holds.  From a place: the
    places its files name, and not the package above a named module, a file of its own place or what a distribution
    inside it loads.  From a verifier besides: its import of its prover inside a function and that of a distribution
    inside it, and the place it reaches only through another file; not its prover's import of it, its test's or its
    Lean script's of the prover; and what a place it loads gains, from a file no load of the verifier's reaches.  Into
    a trusted place: a prover's file, a prover's test, for a place the listing marks, a binding, and a file in a
    device's directory that the listing gives to an untrusted place; not a device's file.  Where the listing names a
    file of a device as a place of its own: that file's load of the rest of its directory.  And the generator writes
    a listing the checks hold, and without ``--grow`` holds back each line of a list that only shrinks."""
    for f, text in ORDER_TREE.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    tree = _public_boundary().Tree(tmp_path, files=ORDER_TREE)
    loads = _all_loads(tree)
    verifier, prover, device, check = (f"verity/protocols/pouw/{d}" for d in ("verifier", "prover", "device", "verifier/check"))
    own = {"verity/protocols/flock": {"owner": "proofs", "reason": "Flock's distribution"}, check: {"owner": "memory-accounting"}}
    steps = _listing().steps
    places = _places([{"paths": ["verity/circuits"]}, {"paths": ["verity/definitions"]}, {"paths": ["verity/commitments/crypto"]},
                      {"paths": [verifier]}, {"paths": [device]}, {"paths": [prover]}, {"paths": ["verity/protocols/flock/verifier"]},
                      {"paths": ["verity/configuration"]}, {"paths": ["verity/protocols/spot"]}])
    listing = Listing(steps, own, {}, places)
    found = {pair: sorted(files) for pair, files in _against(loads, steps).items()}
    assert found == {
        ("verity/circuits", "verity/definitions"): [f"verity/circuits/{f}.py" for f in ("at_load", "in_function", "in_string")],
        ("verity/configuration", "verity/protocols/flock"): ["verity/configuration/package_above.py"],
        ("verity/protocols/flock/verifier", "verity/protocols/spot"): ["verity/protocols/flock/verifier/Flock/V.lean", "verity/protocols/flock/verifier/lakefile.toml"],
        ("verity/protocols/flock", "verity/protocols/pouw"): ["verity/protocols/flock/prover/stage.py"],
        ("verity/protocols/spot", "demos"): ["verity/protocols/spot/by_path.py"],
        ("tools", "benchmarks"): ["tools/lazy.py"],
        ("tools", "infra"): ["tools/tests/test_console.py"],
        ("benchmarks", "infra"): ["benchmarks/probe/tests/test_run.py"],
    }, found
    assert _row("verity/settings.py", steps) == "verity" and _row("pyproject.toml", steps) == "." \
        and _row("verity/protocols/flock/verifier/Flock/V.lean", steps) == "verity/protocols/flock/verifier"
    assert [_below(f, steps) for f in ("verity/commitments/crypto/lms.py", "verity/commitments/merkle.py", "verity/errors.py",
                                       f"{prover}/circuit/words.py", "examples/a/b.py", "conftest.py")] == [
        "verity/commitments/crypto", "verity/commitments", "verity", prover, "examples", "."]
    party = f"{device}/party.py"
    split = places | _places([{"paths": [party]}])
    assert [_place(f, split) for f in ("verity/circuits/tests/test_a.py", f"{check}/interop.py", party, f"{device}/serve.py",
                                       "verity/commitments/merkle.py", "verity/errors.py", "tools/lazy.py")] == [
        "verity/circuits", verifier, party, device, None, None, None]
    assert [_distribution(f, listing) for f in ("verity/protocols/flock/prover/stage.py", "verity/protocols/flock/verifier/interface.py",
                                                f"{check}/interop.py", f"{verifier}/check.py")] == ["verity/protocols/flock", None, check, None]
    assert [_trusted(f) for f in (f"{check}/interop.py", f"{device}/serve.py", f"{prover}/build.py", "tools/verifier/run.py")] == [
        verifier, device, None, None]
    assert [_trusted_place(e) for e in _places([{"paths": ["verity/commitments", party]}, {"paths": [party, prover]},
                                                {"paths": ["verity/definitions"], "trusted": True}]).values()] == [False, True, True]
    assert sorted({(x.loader, x.line, x.loaded) for x in _to_provers(loads)}) == [
        (f"{verifier}/check.py", 6, f"{prover}/__init__.py"), (f"{verifier}/check.py", 6, f"{prover}/build.py"),
        (f"{check}/interop.py", 1, "verity/protocols/flock/prover/stage.py")]
    assert [str(x) for x in _into_distributions(loads, listing)] == [
        "verity/configuration/package_above.py:1: python package verity.protocols.flock.verifier.interface loads "
        "verity/protocols/flock/__init__.py"]
    found = _place_loads(loads, listing)
    assert list(found) == ["verity/circuits", "verity/configuration", "verity/definitions", "verity/protocols/flock/verifier",
                           device, prover, verifier, "verity/protocols/spot"], list(found)
    assert [sorted(x) for x in found["verity/circuits"]] == [["verity/definitions"], [], []], found["verity/circuits"]
    assert [sorted(x) for x in found["verity/definitions"]] == [["verity/circuits"], [], []], found["verity/definitions"]
    assert [sorted(x) for x in found["verity/protocols/spot"]] == [["demos/e2e"], [], []], found["verity/protocols/spot"]
    assert [sorted(x) for x in found[prover]] == [[verifier], [], []], found[prover]
    direct, reach, loaders = found[verifier]
    assert sorted(direct) == ["verity/circuits", "verity/commitments/crypto", prover], direct
    assert sorted(reach) == ["verity/circuits", "verity/commitments/crypto", "verity/definitions", prover], reach
    assert reach["verity/definitions"] == [f"{verifier}/check.py", "verity/circuits/at_load.py", "verity/definitions/ops.py"]
    assert sorted(loaders) == ["verity/protocols/flock/prover/stage.py", f"{prover}/build.py"], loaders
    assert [sorted(x) for x in found[device]] == [[verifier], sorted([*reach, verifier]), [f"{prover}/tests/test_build.py"]], found[device]
    assert [sorted(x) for x in found["verity/protocols/flock/verifier"]] == [[], [], ["verity/configuration/package_above.py"]]
    marked = _place_loads(loads, listing._replace(places=places | _places([{"paths": ["verity/definitions"], "trusted": True}])))
    assert [sorted(x) for x in marked["verity/definitions"]] == [["verity/circuits"], ["verity/circuits"], [
        "verity/circuits/at_load.circuit-check.py", "verity/circuits/at_load.py", "verity/circuits/in_function.py",
        "verity/circuits/in_string.py", "verity/circuits/tests/test_a.py"]], marked["verity/definitions"]
    parted = _place_loads(loads, listing._replace(places=split))
    assert list(parted) == [*list(found)[:4], device, party, *list(found)[5:]], list(parted)
    assert [sorted(x) for x in parted[device]] == [sorted(x) for x in found[device]], parted[device]
    assert [sorted(x) for x in parted[party]] == [[device], sorted([*reach, device, verifier]), []], parted[party]
    kept = _place_loads(loads, listing._replace(places=places | _places([{"paths": ["verity/circuits", party]}])))
    assert sorted(kept["verity/circuits"][0]) == ["verity/definitions", device] and sorted(kept[device][2]) == [
        party, f"{prover}/tests/test_build.py"], (kept["verity/circuits"], kept[device])

    def lists(places: dict[str, dict[str, Any]], found: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {d: e | dict(zip(LISTS, map(set, found.get(d, ({}, {}, {}))))) for d, e in places.items()}

    files = set(ORDER_TREE)
    held = Listing(steps, own, {pair: set(files) for pair, files in _against(loads, steps).items()}, lists(places, found))
    _hold_order(loads, held)
    _hold_places(loads, files, held)
    hold_distributions = functools.partial(_hold_distributions, neither=(check,))
    hold_distributions(tree.dirs, [x for x in loads if "package_above" not in x.loader], held)
    pair = ("verity/circuits", "verity/definitions")
    with pytest.raises(AssertionError, match=r"no file needs any more; delete them: \['verity/commitments/windows -> tools'\]"):
        _hold_order(loads, held._replace(against=held.against | {("verity/commitments/windows", "tools"): {"verity/commitments/windows/rows.py"}}))
    with pytest.raises(AssertionError, match="only shrinks.*\n.*in_string.py:1: python string verity.definitions.ops loads"):
        _hold_order(loads, held._replace(against=held.against | {pair: held.against[pair] - {"verity/circuits/in_string.py"}}))
    with pytest.raises(AssertionError, match="only shrinks.*\n.*test_run.py:1: python import infra.cluster.usage loads"):
        _hold_order(loads, held._replace(against={p: files for p, files in held.against.items() if p[1] != "infra"}))
    with pytest.raises(AssertionError, match="no longer load a file of the later row; delete them:\n.*verity/circuits/gone.py"):
        _hold_order(loads, held._replace(against=held.against | {pair: held.against[pair] | {"verity/circuits/gone.py"}}))

    def without(d: str, **gone: str) -> Listing:
        return held._replace(places=held.places | {d: held.places[d] | {kind: held.places[d][kind] - {at} for kind, at in gone.items()}})

    def also(d: str, **more: str) -> Listing:
        return held._replace(places=held.places | {d: held.places[d] | {kind: held.places[d][kind] | {at} for kind, at in more.items()}})

    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{verifier} loads {prover}: .*\n\\s*{verifier} reaches {prover}: "):
        _hold_places(loads, files, without(verifier, loads=prover, reaches=prover))
    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{verifier} loads verity/commitments/crypto: "):
        _hold_places(loads, files, without(verifier, loads="verity/commitments/crypto"))
    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{verifier} reaches verity/definitions: {verifier}/check.py -> "):
        _hold_places(loads, files, without(verifier, reaches="verity/definitions"))
    aside = Load("verity/circuits/in_function.py", "verity/protocols/spot/by_path.py", 3, "python import", "verity.protocols.spot.by_path")
    with pytest.raises(AssertionError, match=f"gains no load(.|\n)*{verifier} reaches demos/e2e: {verifier} -> verity/circuits -> "
                                             f"verity/protocols/spot -> demos/e2e\n\\s*{verifier} reaches verity/protocols/spot: "):
        _hold_places((*loads, aside), files, also("verity/circuits", loads="verity/protocols/spot"))
    with pytest.raises(AssertionError, match=f"only shrinks.*\n\\s*{device} is loaded by {prover}/tests/test_build.py:1: "):
        _hold_places(loads, files, without(device, loaded_by=f"{prover}/tests/test_build.py"))
    with pytest.raises(AssertionError, match=f"only shrinks.*\n\\s*{verifier} is loaded by {prover}/build.py:1: "):
        _hold_places(loads, files, without(verifier, loaded_by=f"{prover}/build.py"))
    with pytest.raises(AssertionError, match="only the places its `loads` lists.*\n\\s*verity/circuits loads verity/definitions: .*at_load.py:1: "):
        _hold_places(loads, files, without("verity/circuits", loads="verity/definitions"))
    with pytest.raises(AssertionError, match=f"no longer has; delete them .*: \\['{verifier}: reaches tools'"):
        _hold_places(loads, files, also(verifier, reaches="tools"))
    with pytest.raises(AssertionError, match="no longer has; delete them .*: \\['verity/circuits: loads verity/commitments/crypto'"):
        _hold_places(loads, files, also("verity/circuits", loads="verity/commitments/crypto"))
    with pytest.raises(AssertionError, match="no longer has; delete them .*: \\['verity/protocols/spot: loaded_by verity/protocols/spot/plan.py'"):
        _hold_places(loads, files, also("verity/protocols/spot", loaded_by="verity/protocols/spot/plan.py"))
    with pytest.raises(AssertionError, match=f"no longer has; delete them .*: \\['{device}/gone.py: no Python is at {device}/gone.py'"):
        _hold_places(loads, files, held._replace(places=held.places | _places([{"paths": [f"{device}/gone.py"]}])))
    with pytest.raises(AssertionError, match=f"two places name; a path is one place's: \\['{prover}'\\]"):
        _hold_places(loads, files, held._replace(places=held.places | _places([{"paths": ["verity/commitments", prover]}])))
    parts = held._replace(places=lists(split, parted))
    _hold_places(loads, files, parts)
    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{party} loads {device}: {party}:1: "):
        _hold_places(loads, files, parts._replace(places=parts.places | {party: parts.places[party] | {"loads": set()}}))
    stub = ORDER.read_text(encoding="utf-8")
    stub = stub[:stub.index(GENERATED)] + GENERATED + "".join(
        f"\n[[place]]\npaths = {list(e['paths'])!r}\n".replace("'", '"') for e in split.values())
    tool = _load_tool("import_order", ROOT / "tools/verity/check/import_order.py")
    text, back = tool.tables(sys.modules[__name__], stub, loads, grow=True)
    assert not back and _generated(text, loads) == text and tool.tables(sys.modules[__name__], text, loads) == (text, [])
    bare, back = tool.tables(sys.modules[__name__], stub, loads)
    assert "[[against]]" not in bare and bare.count("reaches") == PLACE_HEADER.count("reaches") and bare.count("loads = [") == 5, bare
    assert {"against tools -> infra: tools/tests/test_console.py", f"{verifier}: loads {prover}", f"{party}: reaches {device}",
            f"{device}: loaded_by {prover}/tests/test_build.py"} <= set(back) and not [x for x in back if x.startswith("verity/circuits")], back
    written = _listing(text)
    _hold_order(loads, written)
    _hold_places(loads, files, written)
    assert sorted(written.places) == sorted(split) and written.places[party]["loads"] == {device} \
        and written.places[party]["reaches"] >= parts.places[party]["reaches"]
    with pytest.raises(AssertionError, match="loads no other distribution.*\n.*package_above.py:1: python package"):
        hold_distributions(tree.dirs, loads, held)
    for entry in ({"reason": "no owner"}, {"owner": "proofs", "date": "2026-11-18", "reason": "both"}, {"owner": "proofs", "date": "soon"}):
        with pytest.raises(AssertionError, match=r"its owner and exactly one of.*: \['verity/protocols/flock'\]"):
            hold_distributions(tree.dirs, (), held._replace(distributions=own | {"verity/protocols/flock": entry}))
    with pytest.raises(AssertionError, match=f"give each its date or its reason: \\['{check}'\\]"):
        _hold_distributions(tree.dirs, (), held, neither=())
    for name in ("verity/protocols/flock", "verity/rho"):
        with pytest.raises(AssertionError, match=f"or is no longer listed; delete them: \\['{name}'\\]"):
            _hold_distributions(tree.dirs, (), held, neither=(check, name))
    with pytest.raises(AssertionError, match=r"hold no tracked file; delete them: \['verity/rho'\]"):
        hold_distributions(tree.dirs, (), held._replace(distributions=own | {"verity/rho": {"owner": "training", "date": "2026-11-18"}}))
