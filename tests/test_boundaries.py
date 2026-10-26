"""The dependency rules (AGENTS.md "Map and commands") as import-boundary checks over ``verity/``, and the one over ``infra/``.

    ml.circuits <- ml.commitments -> ml.crypto;  ml.randomness, ml.crypto and core.claims import no unit;  the
    catalog (ml.circuits.ops, .silicon, .targets) and the kernels (ml.kernels) build on ml.circuits, and the served
    workloads' Definitions (ml.definitions) on the catalog and the kernels;
    ml.circuits <- core.protocols.profile;  each protocol imports ml, the profile and the protocols it names in
    ``ALLOWED``, and no third-party package;  a verifier imports no ml (Daniel, 18 Oct): until a protocol's verifier of
    record is its Lean, its Python keeps only the ml edges ``CORE_ML_EDGES`` froze, what of ml core reaches is
    ``SOUNDNESS_PATH``, and both only shrink;  ml never imports core;  no unit imports
    another distribution of the workspace (C-Flock, an integration, a benchmark, a tool) or an external runtime: ``verity``
    imports the standard library, its declared dependencies and itself;  nothing outside ``infra/`` imports ``infra/``;  only
    tests name a function past a guard (``TEST_ONLY``).

An import is an import statement (a relative one resolved against its package), a string handed to
``importlib.import_module``, ``__import__`` or ``importorskip``, or a ``"module:attr"`` string naming what to load at run
time: a module named as data is still a dependency.  Tests (``tests/``) are suites of their own and declare what they read;
a ``lean/`` directory holds a Lean package and the scripts that write its vectors, run by path.  The trees in ``SEPARATE``
have import roots of their own, and only the rule that ml never imports core reads them; it reads ml's tests too.

numpy is a core dependency (architecture decision: `verity.ml.circuits.evaluation` batches over numpy arrays).  The
plain-integer references stay numpy-free on purpose, because they are the specification the Rust guest, the Lean vectors and
the probes are checked against: the rest of `verity.ml.circuits` (the tensor-core step in `verity.ml.circuits.silicon`
among them).  `verity.ml.randomness` is exact stdlib arithmetic too.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "verity"

#: For each unit (a package, or a module, under ``verity``), the other units it may import.  ``errors`` is shared by all.
ALLOWED = {
    "ml.circuits": set(),
    "ml.randomness": set(),
    "ml.crypto": set(),
    "ml.commitments": {"ml.circuits", "ml.crypto"},
    # the catalog and the kernels, distributions of their own before the platform layout
    "ml.circuits.silicon": {"ml.circuits"},
    "ml.circuits.targets": {"ml.circuits", "ml.circuits.silicon"},
    "ml.circuits.ops": {"ml.circuits", "ml.circuits.silicon", "ml.kernels"},
    "ml.kernels": {"ml.circuits", "ml.circuits.silicon"},
    "ml.private_circuits": {"ml.circuits"},
    # the served workloads' kernel Definitions and their Boolean versions, replay and the proof-unit cut, which no verifier reads
    "ml.definitions": {"ml.circuits", "ml.circuits.ops", "ml.circuits.silicon", "ml.kernels"},
    # 2026-10-24 (Daniel): integrations are frontends. An integration hands Verity a window: its circuit, the values it
    # commits in a `verity.ml.commitments` scheme and registers before any coin, and the drawn units' openings; core's
    # records come from the caller. Its catalog of templates is Verity's Definitions (`ml.circuits.ops`).
    "ml.windows": {"ml.circuits", "ml.circuits.ops", "ml.commitments"},
    "core.claims": set(),
    "core.protocols.profile": {"ml.circuits"},
    # The verifier's coins sample its own bytes as the Lean verifier does (`coins`), and the proof service's challenger
    # draws with them (`service.Stream.select`); every coin source is named by a claim instance (`challenger.coin_claim`).
    "core.protocols.sampled_proofs": {"core.claims", "ml.circuits", "ml.commitments", "ml.randomness",
                                      "core.protocols.profile"},
    "core.protocols.pous": {"core.claims", "ml.circuits", "ml.commitments", "ml.crypto", "ml.randomness"},
    # 2026-10-11 (Daniel, decision 7 of the resource-accounting agenda): the bandwidth audit's store is PoUS's `C*`,
    # under its setup and `vk`, reused as they are.
    "core.protocols.bandwidth": {"ml.crypto", "ml.randomness", "core.protocols.pous"},
    # 2026-10-12 (memory accounting's remote bandwidth program): the remote audit commits receipts in frame-v3-sha512
    # trees, draws from the verifier's secret, and records its openings as an integrity profile.  Its coins are the
    # proof service's one challenger's (`sampled_proofs.challenger`, MEM-43).
    "core.protocols.remote_bandwidth": {"ml.commitments", "ml.randomness", "core.protocols.profile",
                                        "core.protocols.sampled_proofs"},
    # 2026-10-03 (Daniel, `.agents/skills/friction/SKILL.md`): PoUW's tile check is a sampled proof.  Its schemes and
    # circuits are stated over the catalog's silicon models and Definitions (the tensor-core steps, FP32 and the casts its
    # verifier recomputes): the one edge into the catalog.
    "core.protocols.pouw": {"ml.circuits", "ml.circuits.ops", "ml.circuits.silicon", "ml.commitments", "ml.crypto",
                            "ml.randomness", "core.protocols.profile", "core.protocols.sampled_proofs"},
    # N2 of the proof-service plan: a link-window's network certificates are registered through the proof service; they
    # are LMS signatures (`ml.crypto.lms`).
    "core.protocols.certifier": {"ml.circuits", "ml.commitments", "ml.crypto", "core.protocols.sampled_proofs"},
}
#: Import names ``verity`` never imports or names as a string: the other workspace distributions, external runtimes, and the
#: repository's own directories.
FORBIDDEN_EXTERNAL = ("verity_sp1", "verity_vllm", "verity_numerical", "verity_flock", "verity_torch", "verity_cli", "verity_difftest",
                      "circuit_check", "research", "cluster", "sp1", "vllm", "torch", "jax", "onnx", "archive", "backends",
                      "benchmarks", "examples", "frontends", "infra", "integrations", "tools", "veritor",
                      "verity_torch_loop")
#: ``verity``'s third-party dependencies (the root's ``pyproject.toml``); anything else it imports is the stdlib or
#: ``verity``.  A protocol imports none: it is the standard library and ``verity``, and the application composes protocols.
DEPENDENCIES = {"numpy"}
#: Trees under ``verity/`` with import roots of their own: C-Flock's distribution (``verity_flock``) and its verifier's
#: scripts, the GPU hashes (``hash_gpu``, run as ``python -m hash_gpu.bench``), the differential test of the Lean against
#: its Python (``verity_difftest``, which their tests import, on Hypothesis), and the PoUS Lean check (``pous-check``), whose Python is
#: run by path beside ``benchmarks/pous``: its difftest generator, which its ``lean-audit.json`` runs, and ``interop.py``,
#: which holds it to the wire's Python reference through ``verity_difftest``.
SEPARATE = ("ml/flock", "ml/kernels/hash_gpu", "core/difftest", "core/protocols/pous/check")
#: Source subtrees that must stay stdlib-only, the scalar references, and within them what may use numpy.
NUMPY_FREE = ("ml/circuits", "ml/randomness", "core/claims")
NUMPY_USERS = ("ml/circuits/evaluation", "ml/circuits/boolean/forms.py", "ml/circuits/ops")
#: ml builds on no verifier: core's frozen edges into ml (``CORE_ML_EDGES``) never run the other way.  The files of
#: ``verity/ml/`` that import core anyway
#: (globs from ``verity/``), each with why.
ML_IMPORTS_CORE = {
    "ml/flock/tests/*": "C-Flock's tests check its prover and the Lean verifier against core's Python verifiers",
    "ml/commitments/tests/test_hm96.py": "checks that the claims hm96's PROTOCOL.md cites are ids of `verity.core.claims`",
    "ml/commitments/tests/test_block_check.py": "holds the opening unit's reference to the remote bandwidth audit's own "
                                                "decode, receipts and trees (`remote_bandwidth.codec`, `receipts`, `audit`)",
    "ml/flock/python/verity_flock/rec_*.py": "V*'s builders (rec_outer, rec_residuals, rec_vstage) state the outer statement "
                                             "over sampled proofs' `one_stage.partition` and `one_stage.registered`, which "
                                             "use only verity.ml; the follow-up to T2 moves them into ml "
                                             "(`verity/ml/circuits/population.py`, `verity/ml/commitments/registered.py`), "
                                             "core imports them from there, and this entry goes",
    "ml/flock/pod/rec_inner.py": "stages the inner statement over sampled proofs' `one_stage.partition`, as V*'s builders "
                                  "do (the rec_* entry), and moves with them in the follow-up to T2",
    "ml/flock/pod/rec_reprice.py": "builds the inner statement's population program and partition with sampled proofs' "
                                    "`one_stage.partition`, as rec_inner does, and moves with it in the follow-up to T2",
    "ml/flock/python/verity_flock/inner_registered.py": "registers a staged inner statement's description rows with sampled "
                                                        "proofs' `one_stage.registered`, as V*'s builders do (the rec_* "
                                                        "entry), and moves with them in the follow-up to T2",
}
#: A verifier imports no ``verity.ml`` code (Daniel, 18 Oct; AGENTS.md "Design rules"): Lean is the definition and the
#: verifier of record, and ml's Python mirrors it.  Until a protocol's verifier of record is its Lean (the 8 Oct procedure,
#: ``.agents/skills/friction/SKILL.md``), its Python verifier keeps the ml edges ``ALLOWED`` gave it on 18 Oct, frozen here,
#: and gains none; a core unit not listed has none.  When ``ALLOWED`` drops an edge, drop it here too.
CORE_ML_EDGES = {
    "core.protocols.profile": {"ml.circuits"},
    "core.protocols.sampled_proofs": {"ml.circuits", "ml.commitments", "ml.randomness"},
    "core.protocols.pous": {"ml.circuits", "ml.commitments", "ml.crypto", "ml.randomness"},
    "core.protocols.bandwidth": {"ml.crypto", "ml.randomness"},
    "core.protocols.remote_bandwidth": {"ml.commitments", "ml.randomness"},
    "core.protocols.pouw": {"ml.circuits", "ml.circuits.ops", "ml.circuits.silicon", "ml.commitments", "ml.crypto",
                            "ml.randomness"},
    "core.protocols.certifier": {"ml.circuits", "ml.commitments", "ml.crypto"},
}
#: The units of ``verity.ml`` core's Python verifiers read, directly or through one another (what ``ALLOWED`` lets core
#: reach): their verdicts rest on them, so they are on the soundness path, and AGENTS.md names each.  Frozen on 18 Oct
#: with ``CORE_ML_EDGES``: it only shrinks, as each protocol's verifier of record becomes its Lean.  The rest of ml
#: (C-Flock's prover, the private circuits) is not on it.
SOUNDNESS_PATH = {"ml.circuits", "ml.circuits.ops", "ml.circuits.silicon", "ml.commitments", "ml.crypto", "ml.kernels",
                  "ml.randomness"}
#: What a test that refuses a new core -> ml edge says.
LEAN_OF_RECORD = ("a verifier imports no verity.ml code (Daniel, 18 Oct; AGENTS.md \"Design rules\", \"Lean is the definition "
                  "and the verifier of record\"): state it in Lean and import nothing from verity.ml")
#: Files outside ``infra/`` that import a package of it, each with why.
INFRA_IMPORTERS = {
    "tools/research/console/verity_console.py": "the console reads `cluster.usage`, and moves to infra/ with research's pods, "
                                                "deploy and Slack code in the follow-up to the platform layout move (#1450)",
}
#: Functions past a guard, which tests alone may name, since a guard never fails open (AGENTS.md "Working principles"): each
#: with the file that defines it and why tests need it.
TEST_ONLY = {
    "_build_unchecked": ("verity/ml/circuits/partition_v2.py",
                         "`partition_v2.build` without its refusal of units of several size classes (CIR-64), for the tests "
                         "of `check` on advice past the builder's scope, as a prover may send it"),
}

#: Modules of ``verity.ml`` that only a prover's or developer's build turns on, which nothing in ``verity/core`` names (an
#: import, an attribute or a string), each with the module and why a verifier must not read it.
PROVER_ONLY = {
    "trace_cache": ("verity/ml/circuits/ops/boolean/trace_cache.py",
                    "traced circuits reused across a prover's builds: a verifier reading them would take a stale or tampered "
                    "entry for a Definition's gates"),
}

MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}


def _separate(rel: Path) -> bool:
    return any(rel.as_posix() == s or rel.as_posix().startswith(s + "/") for s in SEPARATE)


def _sources(root: Path = SRC) -> list[Path]:
    """``verity``'s own Python files: every ``.py`` under ``root`` outside a ``tests/`` or ``lean/`` directory, a Lake build
    directory (core's Lean builds in ``verity/core/service/flock/.lake``) or a ``SEPARATE`` tree, and no circuit-check
    sidecar (``<module>.circuit-check.py``: circuit-check's bindings, which it loads by path and no module imports)."""
    return sorted(p for p in root.rglob("*.py")
                  if not {"tests", "lean", ".lake", "__pycache__"} & set(p.relative_to(SRC).parts)
                  and not _separate(p.relative_to(SRC)) and not p.name.endswith(".circuit-check.py"))


def _imports(path: Path, src: Path = SRC, package: tuple[str, ...] = ("verity",)) -> list[tuple[str, int, bool]]:
    """``(module, line, imported)`` for every module ``path`` names.  Statements and import-call strings are ``imported``; a
    ``"module:attr"`` string is not, since any ``word:word`` constant matches: only the forbidden and direction rules
    apply to it."""
    package = [*package, *path.relative_to(src).parts[:-1]]
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
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


def _unit(dotted: str) -> str | None:
    """The unit a module under ``verity`` (``ml.circuits.cut``) belongs to, or None for a grouping or a top-level file."""
    parts = dotted.split(".")
    return next((".".join(parts[:n]) for n in range(len(parts), 0, -1) if ".".join(parts[:n]) in ALLOWED), None)


def _dotted(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _external(name: str, imported: bool, allowed: set[str]) -> bool:
    """Whether a non-``verity`` ``name`` breaks the rules: it is forbidden and not ``allowed``, or it is imported and is
    neither stdlib nor in ``allowed``."""
    top = name.split(".")[0]
    return (top in FORBIDDEN_EXTERNAL and top not in allowed) or (
        imported and top != "verity" and top not in sys.stdlib_module_names and top not in allowed)


@pytest.mark.parametrize("unit", sorted(ALLOWED))
def test_core_unit_imports_only_inward(unit: str) -> None:
    files = [p for p in _sources() if _unit(_dotted(p)) == unit]
    assert files, f"unit {unit!r} has no source: remove it from ALLOWED"
    offenders = []
    for path in files:
        for name, line, imported in _imports(path):
            parts = name.split(".")
            target = _unit(".".join(parts[1:])) if parts[0] == "verity" and len(parts) > 1 else None
            third_party = set() if unit.startswith("core.protocols.") else DEPENDENCIES
            if (_external(name, imported, third_party)
                    or (parts[0] == "verity" and parts[1:2] != ["errors"] and target not in (None, unit, *ALLOWED[unit]))):
                offenders.append(f"{path.relative_to(SRC)}:{line}: {name}")
    assert not offenders, "\n".join(offenders)


def test_every_source_belongs_to_a_unit_or_is_a_grouping() -> None:
    """A new package under ``verity`` is a new unit, listed in ``ALLOWED``; what is outside every unit is a grouping's
    ``__init__.py`` (``ml``, ``core``, ``core.protocols``) or a file directly under ``verity/``, and imports the standard
    library and nothing else: ``verity/__init__.py`` re-exports nothing and ``verity/errors.py`` is shared by every unit, the
    stdlib-only references included."""
    offenders = []
    for path in _sources():
        if _unit(_dotted(path)) is not None:
            continue
        if path.name != "__init__.py" and path.parent != SRC:
            offenders.append(f"{path.relative_to(SRC)}: outside every unit; add its package to ALLOWED")
        offenders += [f"{path.relative_to(SRC)}:{line}: {name}" for name, line, imported in _imports(path)
                      if _external(name, imported, set()) or name.split(".")[0] == "verity"]
    assert not offenders, "\n".join(offenders)


def test_every_source_directory_is_a_package() -> None:
    """``verity`` is one regular package: every directory of its own Python has an ``__init__.py``, so no part of it resolves
    as a namespace package from whatever else is on the path.  Tests and Lean scripts are run by path and stay out."""
    missing = sorted({str(p.parent.relative_to(SRC.parent)) for p in _sources() for d in [p.parent, *p.parent.parents]
                      if d.is_relative_to(SRC) and not (d / "__init__.py").exists()})
    assert not missing, missing


@pytest.mark.parametrize("subtree", NUMPY_FREE)
def test_scalar_references_are_stdlib_only(subtree: str) -> None:
    offenders = [f"{path.relative_to(SRC)}:{line}: {name}"
                 for path in _sources(SRC / subtree)
                 if not any(path.relative_to(SRC).as_posix().startswith(u) for u in NUMPY_USERS)
                 for name, line, _ in _imports(path)
                 if name.split(".")[0] == "numpy"]
    assert not offenders, "\n".join(offenders)


def _ml_importers_of_core() -> dict[str, list[str]]:
    """Each file under ``verity/ml/`` (its tests and ``SEPARATE`` trees included) that names a module of ``verity.core``,
    with where."""
    out: dict[str, list[str]] = {}
    for path in sorted((SRC / "ml").rglob("*.py")):
        rel = path.relative_to(SRC)
        if {".lake", "__pycache__"} & set(rel.parts):
            continue
        hits = [f"{line}: {name}" for name, line, _ in _imports(path) if name.split(".")[:2] == ["verity", "core"]]
        if hits:
            out[rel.as_posix()] = hits
    return out


def test_ml_never_imports_core() -> None:
    found = _ml_importers_of_core()
    offenders = [f"{f}:{hit}" for f, hits in found.items() if not any(fnmatch.fnmatch(f, g) for g in ML_IMPORTS_CORE)
                 for hit in hits]
    assert not offenders, "\n".join(offenders)
    stale = [g for g in ML_IMPORTS_CORE if not any(fnmatch.fnmatch(f, g) for f in found)]
    assert not stale, f"ML_IMPORTS_CORE entries that match no importer; remove them: {stale}"


def _core_reach_in_ml(allowed: dict[str, set[str]]) -> set[str]:
    """The ml units the core units of ``allowed`` reach, directly or through one another."""
    reached, todo = set(), [u for u in allowed if u.startswith("core.")]
    while todo:
        for v in allowed.get(todo.pop(), set()) - reached:
            reached.add(v)
            todo.append(v)
    return {u for u in reached if u.startswith("ml.")}


def _core_ml_growth(allowed: dict[str, set[str]]) -> list[str]:
    """What ``allowed`` adds to core's frozen reach into ml: each core unit's ml edges beyond ``CORE_ML_EDGES``, and ml
    units reached off ``SOUNDNESS_PATH`` (through a new core edge, or a new edge out of a unit on the path)."""
    grown = [f"{u} -> {v}" for u in sorted(allowed) if u.startswith("core.")
             for v in sorted(allowed[u]) if v.startswith("ml.") and v not in CORE_ML_EDGES.get(u, set())]
    return grown + [f"core reaches {v}, off SOUNDNESS_PATH" for v in sorted(_core_reach_in_ml(allowed) - SOUNDNESS_PATH)]


def test_core_gains_no_edge_into_ml() -> None:
    """No core unit gains an ml edge, and core's reach into ml doesn't grow (Daniel, 18 Oct): the edges only shrink, each
    with its protocol's switch to Lean."""
    grown = _core_ml_growth(ALLOWED)
    assert not grown, f"{grown}: {LEAN_OF_RECORD}"


def test_core_ml_growth_is_refused() -> None:
    """The check above isn't vacuous: an added edge from a core unit, or from an ml unit on the soundness path, is growth."""
    assert _core_ml_growth({**ALLOWED, "core.protocols.pous": ALLOWED["core.protocols.pous"] | {"ml.private_circuits"}}) == [
        "core.protocols.pous -> ml.private_circuits", "core reaches ml.private_circuits, off SOUNDNESS_PATH"]
    assert _core_ml_growth({**ALLOWED, "core.claims": {"ml.crypto"}}) == ["core.claims -> ml.crypto"]
    assert _core_ml_growth({**ALLOWED, "ml.crypto": {"ml.private_circuits"}}) == [
        "core reaches ml.private_circuits, off SOUNDNESS_PATH"]


def test_core_ml_edges_shrink_with_allowed() -> None:
    """``CORE_ML_EDGES`` and ``SOUNDNESS_PATH`` shrink with ``ALLOWED``, so an edge it drops can't come back.  With the
    test above, ``SOUNDNESS_PATH`` is exactly what ``ALLOWED`` lets core reach in ml."""
    dropped = [f"{u} -> {v}" for u, vs in sorted(CORE_ML_EDGES.items()) for v in sorted(vs - ALLOWED.get(u, set()))]
    assert not dropped, f"ALLOWED no longer has {dropped}: remove them from CORE_ML_EDGES too"
    ml = _core_reach_in_ml(ALLOWED)
    assert ml >= SOUNDNESS_PATH, (f"core no longer reaches {sorted(SOUNDNESS_PATH - ml)} in ml: remove them from "
                                  "SOUNDNESS_PATH and from AGENTS.md's list of the soundness path")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    missing = [u for u in sorted(SOUNDNESS_PATH) if f"`{u}`" not in agents]
    assert not missing, f"AGENTS.md doesn't name {missing}, which core's Python verifiers read"


def _infra_packages() -> set[str]:
    """The import packages of ``infra/``'s distributions, from each one's wheel configuration.  A project with no wheel (the
    Slack events service's suite, whose code Vercel deploys) has no package to import."""
    return {Path(p).name for pyproject in sorted((ROOT / "infra").glob("*/pyproject.toml"))
            for p in tomllib.loads(pyproject.read_text()).get("tool", {}).get("hatch", {}).get("build", {})
            .get("targets", {}).get("wheel", {}).get("packages", [])}


def test_nothing_outside_infra_imports_it() -> None:
    packages = _infra_packages()
    assert "cluster" in packages
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT, check=True, capture_output=True).stdout
    word = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(packages))) + r")\b")
    found: dict[str, list[str]] = {}
    for rel in (p for p in tracked.decode().split("\0") if p and not p.startswith("infra/")):
        path = ROOT / rel
        if not path.is_file() or not word.search(text := path.read_text(encoding="utf-8", errors="replace")):
            continue
        try:
            names = _imports(path, ROOT, ())
        except SyntaxError:
            continue
        hits = [f"{line}: {name}" for name, line, imported in names if imported and name.split(".")[0] in packages]
        if hits:
            found[rel] = hits
    offenders = [f"{f}:{hit}" for f, hits in found.items() if f not in INFRA_IMPORTERS for hit in hits]
    assert not offenders, "\n".join(offenders)
    stale = sorted(set(INFRA_IMPORTERS) - set(found))
    assert not stale, f"INFRA_IMPORTERS entries that no longer import infra/; remove them: {stale}"


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
    src = ("from verity.ml.circuits.partition_v2 import _build_unchecked as b\nV._build_unchecked(P)\n"
           "getattr(V, '_build_unchecked')(P)\ndef _build_unchecked(p):\n    return p\n")
    assert sorted(line for _name, line in _uses(ast.parse(src), {"_build_unchecked"})) == [1, 2, 3]
    assert _is_test("verity/ml/circuits/tests/test_partition_v2.py") and not _is_test("verity/ml/circuits/partition_v2.py")


def test_core_never_names_a_prover_only_module() -> None:
    """No file of ``verity/core`` (its tests included) names a module of ``PROVER_ONLY``, and each still exists."""
    offenders = []
    for path in sorted((SRC / "core").rglob("*.py")):
        if {".lake", "__pycache__"} & set(path.relative_to(SRC).parts):
            continue
        hits = _uses(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), set(PROVER_ONLY))
        hits += [(name, line) for name, line, _ in _imports(path) if set(name.split(".")) & set(PROVER_ONLY)]
        offenders += [f"{path.relative_to(SRC)}:{line}: {name}" for name, line in hits]
    assert not offenders, "a verifier reads no prover-only module (PROVER_ONLY):\n" + "\n".join(offenders)
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
