"""The dependency rules (AGENTS.md "Map and commands") as import-boundary checks over ``verity/``, and the one over ``infra/``.

    ml.circuits <- ml.commitments -> ml.crypto;  ml.randomness, ml.crypto and core.claims import no unit;  the
    catalog (ml.circuits.ops, .silicon, .targets) and the kernels (ml.kernels) build on ml.circuits, and the served
    workloads' Definitions (ml.definitions) on the catalog and the kernels;
    ml.circuits <- core.protocols.profile;  each protocol imports ml, the profile and the protocols it names in
    ``ALLOWED``, and no third-party package;  a verifier imports no ml: until a protocol's verifier of
    record is its Lean, its Python keeps only the ml edges ``CORE_ML_EDGES`` froze, what of ml core reaches is
    ``SOUNDNESS_PATH``, and both only shrink;  ml never imports core;  no unit imports
    another distribution of the workspace (C-Flock, an integration, a benchmark, a tool) or an external runtime: ``verity``
    imports the standard library, its declared dependencies and itself;  nothing outside ``infra/`` imports ``infra/``;
    nothing outside ``benchmarks/`` imports ``benchmarks/``, by name or by path (``BENCHMARKS_IMPORTERS``);  only tests name
    a function past a guard (``TEST_ONLY``).

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
import builtins
import fnmatch
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Set
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "verity"

#: For each unit (a package, or a module, under ``verity``), the other units it may import.  ``errors`` and ``settings`` are shared by all.
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
    # the served workloads' kernel Definitions and their Boolean versions, replay and the proof-block cut, which no verifier reads
    "ml.definitions": {"ml.circuits", "ml.circuits.ops", "ml.circuits.silicon", "ml.kernels"},
    # Integrations are frontends. An integration hands Verity a window: its circuit, the values it
    # commits in a `verity.ml.commitments` scheme and registers before any coin, and the drawn units' openings; core's
    # records come from the caller. Its catalog of templates is Verity's Definitions (`ml.circuits.ops`).
    "ml.windows": {"ml.circuits", "ml.circuits.ops", "ml.commitments"},
    "core.claims": set(),
    # 2026-10-27 (memory accounting, the deployment API v3, note:memory-accounting/20261026T2305Z-draft-deployment-api-v3):
    # the wire is what both sides parse with, stdlib only; the deployment layer is protocol-agnostic and reads frames with
    # it, signs and verifies only through keys its caller hands it, and checks a configuration against what a protocol and
    # a workload declare to it, so neither gains an ml edge.
    "core.wire": set(),
    "core.deploy": {"core.wire"},
    "core.protocols.profile": {"ml.circuits"},
    # The verifier's coins sample its own bytes as the Lean verifier does (`coins`), and the proof service's challenger
    # draws with them (`service.Stream.select`); every coin source is named by a claim instance (`challenger.coin_claim`).
    # Its challenger, firewall, workers and verifier are
    # parties of the deployment layer (`parties`), their messages the wire's.
    "core.protocols.sampled_proofs": {"core.claims", "core.wire", "core.deploy", "ml.circuits", "ml.commitments",
                                      "ml.randomness", "core.protocols.profile"},
    # 2026-10-27 (memory accounting, the deployment API v3): the wipe's memory challenger and prover are parties of the
    # deployment layer (`wipe_parties`), their messages the wire's.
    "core.protocols.pous": {"core.claims", "core.wire", "core.deploy", "ml.circuits", "ml.commitments", "ml.crypto",
                            "ml.randomness"},
    # 2026-10-11 (Daniel, decision 7 of the resource-accounting agenda): the bandwidth audit's store is PoUS's `C*`,
    # under its setup and `vk`, reused as they are.
    "core.protocols.bandwidth": {"ml.crypto", "ml.randomness", "core.protocols.pous"},
    # The remote bandwidth audit commits receipts in frame-v3-sha512
    # trees, draws from the verifier's secret, and records its openings as an integrity profile.  Its coins are the
    # proof service's one challenger's (`sampled_proofs.challenger`).
    # The remote bandwidth audit is over the P2-encoded weights store, and a P2 store is PoUS's `C*`, under its setup and `vk`, reused as they are (`p2.py`).
    # The session's verifier and prover are parties of the deployment layer (`session_parties`), their messages the wire's.
    "core.protocols.remote_bandwidth": {"ml.commitments", "ml.randomness", "core.protocols.profile",
                                        "core.protocols.sampled_proofs", "core.protocols.pous", "core.wire",
                                        "core.deploy"},
    # PoUW's tile check is a sampled proof (`.agents/skills/friction/SKILL.md`).  Its schemes and
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
                      "benchmarks", "examples", "frontends", "infra", "integrations", "tools", "veritor")
#: ``verity``'s third-party dependencies (the root's ``pyproject.toml``); anything else it imports is the stdlib or
#: ``verity``.  A protocol imports none: it is the standard library and ``verity``, and the application composes protocols.
DEPENDENCIES = {"numpy"}
#: Trees under ``verity/`` with import roots of their own: C-Flock's distribution (``verity_flock``) and its verifier's
#: scripts, the differential test of the Lean against its Python (``verity_difftest``, which their tests import, on
#: Hypothesis), and the PoUS Lean check (``pous-check``), whose Python is run by path beside ``benchmarks/pous``: its difftest generator, which its ``lean-audit.json`` runs, and ``interop.py``,
#: which holds it to the wire's Python reference through ``verity_difftest``.
SEPARATE = ("ml/flock", "core/difftest", "core/protocols/pous/check")
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
    "ml/windows/tests/test_blocks.py": "holds CMP-92's class circuit to network accounting's check of its servers "
                                       "(`certifier.blocks.server_blocks`), which refuses its two-server split (NET-92)",
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
#: A verifier imports no ``verity.ml`` code (AGENTS.md "Design rules"): Lean is the definition and the
#: verifier of record, and ml's Python mirrors it.  Until a protocol's verifier of record is its Lean (the switch procedure,
#: ``.agents/skills/friction/SKILL.md``), its Python verifier keeps the ml edges ``ALLOWED`` gave it when they were frozen here,
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
#: reach): their verdicts rest on them, so they are on the soundness path, and AGENTS.md names each.  Frozen
#: with ``CORE_ML_EDGES``: it only shrinks, as each protocol's verifier of record becomes its Lean.  The rest of ml
#: (C-Flock's prover, the private circuits) is not on it.
SOUNDNESS_PATH = {"ml.circuits", "ml.circuits.ops", "ml.circuits.silicon", "ml.commitments", "ml.crypto", "ml.kernels",
                  "ml.randomness"}
#: What a test that refuses a new core -> ml edge says.
LEAN_OF_RECORD = ("a verifier imports no verity.ml code (AGENTS.md \"Design rules\", \"Lean is the definition "
                  "and the verifier of record\"): state it in Lean and import nothing from verity.ml")
#: Files outside ``infra/`` that import a package of it, each with why.
INFRA_IMPORTERS = {
    "tools/research/console/verity_console.py": "the console reads `cluster.usage`, and moves to infra/ with research's pods, "
                                                "deploy and Slack code in the follow-up to the platform layout move (#1450)",
}
#: Files outside ``benchmarks/`` that import a module of it, by name or by path, each with its owner and what it imports.
#: ``benchmarks/`` is measurement only: a number a protocol uses reaches it
#: through a card file citing an ``art:`` id, never as an import, and code a protocol runs moves out to the package that
#: uses it.  It only shrinks.
BENCHMARKS_IMPORTERS = {
    # the H200 example: its runner loads benchmark code by path
    "examples/h200/e2e_h200/attention.py": "compute-accounting: kernels' attn_ref (`benchmarks/torch_gpu/attn_ref`)",
    "examples/h200/e2e_h200/commit.py": "compute-accounting: kernels' commit_slots (`benchmarks/torch_gpu/commit_slots`)",
    "examples/h200/e2e_h200/percall.py": "compute-accounting: J10-R's sm_90 build (`benchmarks/torch_gpu/j10r_sm90`)",
    "examples/h200/e2e_h200/store.py": "compute-accounting: P2's store (`benchmarks/pous`) and the s8 decode linear "
                                       "(`benchmarks/torch_gpu/s8_dec_linear`)",
    "examples/h200/interfaces/verity_e2e_demo/replay_bundle.py": "compute-accounting: kernels' commit_slots "
                                                                 "(`benchmarks/torch_gpu/commit_slots`)",
    # the served path: verity_vllm runs benchmarks' code
    "integrations/vllm/verity_vllm/protocols/pous.py": "memory-accounting: P2's store (`benchmarks/pous/p2_v1`) and the "
                                                       "staging decode (`benchmarks/torch_gpu/p2_stage`), through "
                                                       "`config.P2_POUS` and `config.P2_STAGE`",
    "integrations/vllm/verity_vllm/protocols/remote_bandwidth/remote.py": "circuits (vllm): the remote session's processes "
                                                                          "(`benchmarks/remote_session`, "
                                                                          "`config.REMOTE_SESSION`) and `paged.py` "
                                                                          "(`benchmarks/vllm_encoded`)",
    "integrations/vllm/verity_vllm/protocols/remote_bandwidth/in_place.py": "proofs: the remote session's `device` and "
                                                                            "`inplace` (`benchmarks/remote_session`)",
    "integrations/vllm/verity_vllm/engine/int8_credited_env.py": "circuits (vllm): `benchmarks/vllm_encoded/paged.py`",
    "verity/core/protocols/pous/check/timed_audit.py": "memory-accounting: the memory challenger, the strict timed audit's "
                                                       "reference (`benchmarks/pous/band_gpu/memory_challenger.py`)",
    "verity/ml/flock/python/verity_flock/struct_lower.py": "proofs: H1's unit (`benchmarks/pouw/h1_prove`, `hidden_zk`)",
    # C-Flock and circuit-check build statements from the numerical benchmarks' templates and lowerings
    **dict.fromkeys((
        "verity/ml/flock/pod/gemm_hill.py", "verity/ml/flock/pod/read_gadget.py", "verity/ml/flock/pod/rec_inner.py",
        "verity/ml/flock/python/verity_flock/bench.py", "verity/ml/flock/python/verity_flock/boolean_export.py",
        "verity/ml/flock/python/verity_flock/circuit.py", "verity/ml/flock/python/verity_flock/circuit_bench.py",
        "verity/ml/flock/python/verity_flock/instances.py", "verity/ml/flock/python/verity_flock/ir_bench.py",
        "verity/ml/flock/python/verity_flock/register.py",
        "verity/ml/flock/tests/plain_hidden.py", "verity/ml/flock/tests/served_registered.py",
        "verity/ml/flock/tests/test_backend.py", "verity/ml/flock/tests/test_circuit.py",
        "verity/ml/flock/tests/test_class_statement.py", "verity/ml/flock/tests/test_derive.py",
        "verity/ml/flock/tests/test_flock_rows.py", "verity/ml/flock/tests/test_gemm_hill.py",
        "verity/ml/flock/tests/test_inner_registered.py", "verity/ml/flock/tests/test_ir_lowering.py",
        "verity/ml/flock/tests/test_ir_sampling.py", "verity/ml/flock/tests/test_lean_hidden_outputs.py",
        "verity/ml/flock/tests/test_lean_plain_hidden.py", "verity/ml/flock/tests/test_lean_plain_hidden_tables.py",
        "verity/ml/flock/tests/test_lean_registered_bit_rows.py", "verity/ml/flock/tests/test_lean_registered_reads.py",
        "verity/ml/flock/tests/test_lean_served_registered.py", "verity/ml/flock/tests/test_lean_session_tables.py",
        "verity/ml/flock/tests/test_lean_typed_statement.py", "verity/ml/flock/tests/test_lean_verifier.py",
        "verity/ml/flock/tests/test_stage_typed.py", "verity/ml/flock/tests/test_typed_statement.py",
        "verity/ml/flock/tests/tmpl_statement.py",
    ), "proofs: `verity_numerical.bench` (`benchmarks/numerical`)'s templates, generator, input sets and lowerings"),
    **dict.fromkeys((
        "tools/verity/circuit_check/src/circuit_check/checks.py", "tools/verity/circuit_check/src/circuit_check/sidecars.py",
        "tools/verity/circuit_check/src/circuit_check/targets.py", "tools/verity/circuit_check/tests/test_circuit_check.py",
    ), "circuits: `verity_numerical.bench` (`benchmarks/numerical`)'s templates and lowerings, as circuit-check targets"),
    # research's registry of measuring tools
    **dict.fromkeys(("tools/research/src/research/store/tools_registry.py", "tools/research/tests/test_store_prov.py"),
                    "infra: the store's tool registry names `benchmarks.<probe>.tool` modules (tc_probe, tc_probe_fp4, "
                    "native_peak, commitments, pouw)"),
    # tests and vector scripts that check against a benchmark's reference
    **dict.fromkeys((
        "verity/core/protocols/pouw/tests/epoch_coins_vectors.py", "verity/core/protocols/pouw/tests/test_pouw_audit.py",
        "verity/core/protocols/pouw/tests/test_pouw_epoch_coins.py", "verity/core/protocols/pouw/tests/test_pouw_pearl_c.py",
        "verity/core/protocols/pouw/tests/test_pouw_pearl_c4.py", "verity/core/protocols/pouw/tests/test_pouw_pearl_c_sm120.py",
        "verity/core/protocols/pouw/tests/test_pouw_pearl_c_work.py", "verity/core/protocols/pouw/tests/test_pouw_pearl_kw.py",
        "verity/core/protocols/pouw/lean/commit/tests/difftest_vectors.py",
    ), "compute-accounting: PoUW's clear lifecycle (`benchmarks/pouw/clear`, and `pearl_c4`)"),
    "verity/core/protocols/pouw/lean/window/tests/test_known_rows.py": "proofs: H1's known rows "
                                                                       "(`benchmarks/pouw/h1_lanes/run.py`)",
    **dict.fromkeys(("verity/core/protocols/pous/check/scripts/difftest_vectors.py",
                     "verity/core/protocols/pous/check/tests/test_timed_audit.py"),
                    "memory-accounting: P2's reference and wire (`benchmarks/pous/p2_v1`) and the memory challenger "
                    "(`benchmarks/pous/band_gpu`)"),
    "verity/ml/circuits/ops/tests/test_tc_probe_families.py": "circuits: the tensor-core probe's families "
                                                              "(`benchmarks/tc_probe`)",
    **dict.fromkeys(("integrations/vllm/tests/protocols/pouw_h1_circuits.py",
                     "integrations/vllm/tests/protocols/pouw_h1_gpu.py",
                     "integrations/vllm/tests/protocols/test_pouw_h1_circuits.py"),
                    "compute-accounting: H1's prover, producer and the live challenger's auditor "
                    "(`benchmarks/pouw/h1_prove`, `served_h1`, `benchmarks/live_challenger`)"),
    **dict.fromkeys(("integrations/vllm/tests/protocols/challenger_host.py", "integrations/vllm/tests/protocols/local_gate.py"),
                    "proofs: the live challenger's gate and model registry (`benchmarks/live_challenger`)"),
    **dict.fromkeys(("integrations/vllm/tests/engine/ppl_oracle.py", "integrations/vllm/tests/engine/fattn_oracle.py",
                     "integrations/vllm/tests/engine/int8_store_gpu.py"),
                    "circuits (vllm): the int8 step's oracles (`benchmarks/ppl/steps.py`, `benchmarks/pouw/edited_gemm`) "
                    "and `benchmarks/vllm_encoded/paged.py`"),
    "integrations/vllm/tests/engine/int8_attention_sass.py": "kernels: the PoUW kernels' SASS gate "
                                                             "(`benchmarks/pouw/harness/sass_gate.py`)",
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
    library and nothing else: ``verity/__init__.py`` re-exports nothing, and ``verity/errors.py`` and the settings registry
    ``verity/settings.py`` are shared by every unit, the stdlib-only references included."""
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
    """No core unit gains an ml edge, and core's reach into ml doesn't grow: the edges only shrink, each
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
    """Each file of ``texts`` (repository path -> source) outside ``benchmarks/`` (and ``archive/``, which has rules of its
    own) that imports a module of ``benchmarks/``, with where: an import of one of ``packages`` (a statement, an
    import-call string or a ``"module:attr"`` string), or a path into ``benchmarks/`` (``_into_benchmarks``) put on
    ``sys.path`` or loaded by file.  A path is traced through the file's own names and parameter defaults, through other
    files' upper-case module constants read as attributes (``config.REMOTE_SESSION``), and through any function, wherever it
    is defined, that hands a parameter to such a load (``remote.session_module``).  Nothing is executed (AST only), and
    only files a text search can't rule out are parsed."""
    alt = "|".join(map(re.escape, sorted(packages)))
    named = [re.compile(rf"""(?:import|from)\s+(?:{alt})\b|(?:{"|".join(IMPORT_CALLS)})\(\s*["'](?:{alt})\b"""),
             re.compile(rf"""["'](?:{alt})(?:\.\w+)*[.:]\w+["']""")]
    quoted = re.compile(r"""["'][^"'\n]*\bbenchmarks\b[^"'\n]*["']""")
    loads = ("sys.path", *PATH_LOADERS)
    ours = {rel: text for rel, text in texts.items() if not rel.startswith(("benchmarks/", "archive/"))}
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


def _tracked_python() -> dict[str, str]:
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT, check=True, capture_output=True).stdout
    return {rel: (ROOT / rel).read_text(encoding="utf-8", errors="replace")
            for rel in tracked.decode().split("\0") if rel and (ROOT / rel).is_file()}


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
                       "importlib.import_module('datacenter.graph')\nTOOL = 'benchmarks.pouw.tool:POUW_VLLM'\n"),
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
        "x/names.py": ["1: verity_bench.vllm", "2: benchmarks.tc_probe", "4: datacenter.graph", "5: benchmarks.pouw.tool"],
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
