"""The dependency rules (AGENTS.md "Map and commands") as import-boundary checks over ``verity/``, and the one over ``infra/``.

    ml.circuits <- ml.commitments -> ml.crypto;  ml.randomness, ml.crypto, core.claims and hardware import no unit;  the
    catalog (ml.circuits.ops, .silicon, .targets) and the kernels (ml.kernels) build on ml.circuits, the served
    workloads' Definitions (ml.definitions) on the catalog and the kernels, and the protocols' circuit builders
    (ml.protocols) on the catalog and their protocols;
    ml.circuits <- core.protocols.profile;  each protocol imports ml, the profile and the protocols it names in
    ``ALLOWED``, and no third-party package;  a verifier imports no ml: until a protocol's verifier is its Lean, its Python keeps only the ml edges ``CORE_ML_EDGES`` froze, what of ml core reaches is
    ``SOUNDNESS_PATH``, and both only shrink;  ml never imports core;  no unit imports
    another distribution of the workspace (Flock, an integration, a benchmark, a tool) or an external runtime: ``verity``
    imports the standard library, its declared dependencies and itself;  nothing outside ``infra/`` imports ``infra/``;
    nothing outside ``benchmarks/`` imports ``benchmarks/``, by name or by path (``BENCHMARKS_IMPORTERS``);  only tests name
    a function past a guard (``TEST_ONLY``);  SPOT[Flock] imports none of the network certifier, in its Python, its Lean
    packages or its analysis (``SPOT_NOT_CERTIFIER``).

An import is an import statement (a relative one resolved against its package), a string handed to
``importlib.import_module``, ``__import__`` or ``importorskip``, or a ``"module:attr"`` string naming what to load at run
time: a module named as data is still a dependency.  Tests (``tests/``) are suites of their own and declare what they read;
a ``lean/`` directory holds a Lean package and the scripts that write its vectors, run by path.  The trees in ``SEPARATE``
have import roots of their own, and only the rule that ml never imports core reads them; it reads ml's tests too.  No other
file under ``verity/``, tests included, reads another distribution at run time: its package data or a module named in a
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

#: For each unit (a package, or a module, under ``verity``), the other units it may import.  ``errors`` and ``settings`` are shared by all.
ALLOWED = {
    "ml.circuits": set(),
    "ml.randomness": set(),
    "ml.crypto": set(),
    # the card files, one TOML a SKU, and their loader: the one source of each card's numbers
    "hardware": set(),
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
    # commits in a `verity.commitments` scheme and registers before any coin, and the sampled proof blocks' openings; core's
    # records come from the caller. Its catalog of templates is Verity's Definitions (`ml.circuits.ops`).
    "ml.windows": {"ml.circuits", "ml.circuits.ops", "ml.commitments"},
    "core.claims": set(),
    # An auditor party's signed period roots and liveness records,
    # stdlib only (its frame-v3-sha512 tree held to `ml.commitments.merkle` by its test), the signer injected.
    "core.attest": set(),
    # The deployment API: the wire is what both sides parse with, stdlib only; the deployment layer is protocol-agnostic and
    # reads frames with it, signs and verifies only through keys its caller hands it, and checks a configuration against what
    # a protocol and a workload declare to it, so neither gains an ml edge.
    "core.wire": set(),
    # The Site, the registered hardware: its locations, links and registration log, the check each entry passes and the
    # views a protocol reads. Its digests are canonical JSON's, the wire's.
    "core.site": {"core.wire"},
    "core.deploy": {"core.wire"},
    # The registration log every subprotocol and both composites read, the Python of `Core.Registration`: the Site's
    # entries and check, canonical JSON and SHA-512 only, so it reaches no ml.
    "core.protocols.registration": {"core.wire", "core.site"},
    "core.protocols.profile": {"ml.circuits"},
    # The verifier's coins sample its own bytes as the Lean verifier does (`coins`), and the proof system's challenger
    # samples with them (`service.Stream.select`); every coin source is named by a claim instance (`challenger.coin_claim`).
    # SPOT[Flock]'s inference server, proof server and verifier, the core deployment's ZK server and workers, and the
    # challenger both run are parties of the deployment layer (`parties`), their messages the wire's.
    "core.protocols.spot": {"core.claims", "core.wire", "core.deploy", "ml.circuits", "ml.commitments",
                                      "ml.randomness", "core.protocols.profile", "core.protocols.record"},
    # The wipe's memory challenger and prover are parties of the
    # deployment layer (`wipe_parties`), their messages the wire's. The
    # memory challenger reports its sessions under its party's per-period root (`core.attest`).
    "core.protocols.pous": {"core.claims", "core.attest", "core.wire", "core.deploy", "ml.commitments", "ml.crypto",
                            "ml.randomness", "core.protocols.reporting"},
    # The bandwidth audit's store is PoUS's `C*`,
    # under its setup and `vk`, reused as they are.
    "core.protocols.bandwidth": {"ml.crypto", "ml.randomness", "core.protocols.pous"},
    # The memory challenger, an auditor-trusted device, over PoUS's protocol and verifier; its party is a party of the
    # deployment layer, its messages the wire's.
    "core.devices.memory_challenger": {"core.protocols.pous", "core.wire", "core.deploy"},
    # The verifier's side runs where `period.json` puts it, a side of the deployment layer (`verity.configuration.deploy.stream`).
    # The remote bandwidth audit commits receipts in frame-v3-sha512
    # trees, samples from the verifier's secret, and records its openings as an integrity profile.  Its coins are the
    # proof system's one challenger's (`spot.challenger`).
    # The remote bandwidth audit is over the P2-encoded weights store, and a P2 store is PoUS's `C*`, under its setup and `vk`, reused as they are (`p2.py`).
    # The session's verifier and prover are parties of the deployment layer (`session_parties`), their messages the wire's.
    "core.protocols.remote_bandwidth": {"ml.commitments", "ml.randomness", "core.protocols.profile",
                                        "core.protocols.spot", "core.protocols.pous", "core.wire",
                                        "core.deploy", "core.protocols.reporting"},
    # PoUW's tile check is a sampled proof (`.agents/skills/friction/SKILL.md`).
    "core.protocols.pouw": {"ml.circuits", "ml.commitments", "ml.crypto", "ml.randomness", "core.protocols.profile",
                            "core.protocols.spot"},
    # N2 of the proof-system plan: a link-window's network certificates are registered through the proof system; they
    # are LMS signatures (`ml.crypto.lms`).
    "core.protocols.certifier": {"ml.circuits", "ml.commitments", "ml.crypto", "core.protocols.spot"},
    # A record entry (`Core.Record`): the form every protocol's signed, chained record entries take.
    "core.protocols.record": set(),
    # The reporting grid every monitoring-period report states, its periods and δs: stdlib only.
    "core.protocols.reporting": set(),
    # What the protocols' provers build and their verifiers don't read, which no core unit imports.  PoUW's and the
    # certifier's are stated over their protocol's parameters and records in core (`ML_IMPORTS_CORE`), and PoUW's schemes
    # and circuits over the catalog's silicon models and Definitions (the tensor-core steps, FP32 and the casts).
    "ml.protocols.pouw": {"ml.circuits", "ml.circuits.ops", "ml.circuits.silicon", "ml.commitments", "ml.crypto",
                          "ml.randomness", "core.protocols.pouw", "core.protocols.profile", "core.protocols.spot"},
    "ml.protocols.pous": {"ml.circuits", "ml.commitments"},
    "ml.protocols.certifier": {"ml.circuits", "ml.commitments", "core.protocols.certifier", "core.protocols.spot"},
}
#: Import names ``verity`` never imports or names as a string: the other workspace distributions, external runtimes, and the
#: repository's own directories.
FORBIDDEN_EXTERNAL = ("verity_sp1", "verity_vllm", "verity_numerical", "verity_flock", "verity_torch", "verity_cli", "verity_difftest",
                      "circuit_check", "research", "cluster", "sp1", "vllm", "torch", "jax", "onnx", "archive", "backends",
                      "benchmarks", "examples", "frontends", "infra", "integrations", "tools", "veritor")
#: ``verity``'s third-party dependencies (the root's ``pyproject.toml``); anything else it imports is the stdlib or
#: ``verity``.  A protocol imports none: it is the standard library and ``verity``, and the application composes protocols.
DEPENDENCIES = {"numpy"}
#: Trees under ``verity/`` with import roots of their own: Flock's distribution (``verity_flock``) and its verifier's
#: scripts, the differential test of the Lean against its Python (``verity_difftest``, which their tests import, on
#: Hypothesis), and the PoUS Lean check (``pous-check``), whose Python is run by path beside ``benchmarks/pous``: its difftest generator, which its ``lean-audit.json`` runs, and ``interop.py``,
#: which holds it to the wire's Python reference through ``verity_difftest``.
SEPARATE = ("ml/flock", "core/difftest", "core/protocols/pous/check")
#: Source subtrees that must stay stdlib-only, the scalar references, and within them what may use numpy.
NUMPY_FREE = ("ml/circuits", "ml/randomness", "core/claims", "hardware")
NUMPY_USERS = ("ml/circuits/evaluation", "ml/circuits/boolean/forms.py", "ml/circuits/ops")
#: ml builds on no verifier: core's frozen edges into ml (``CORE_ML_EDGES``) never run the other way.  The files of
#: ``verity/ml/`` that import core anyway
#: (globs from ``verity/``), each with why.
ML_IMPORTS_CORE = {
    "ml/flock/tests/*": "Flock's tests check its prover and the Lean verifier against core's Python verifiers",
    "ml/commitments/tests/test_hm96.py": "checks that the claims hm96's PROTOCOL.md cites are ids of `verity.claims`",
    "ml/commitments/tests/test_block_check.py": "holds the opening unit's reference to the remote bandwidth audit's own "
                                                "decode, receipts and trees (`remote_bandwidth.codec`, `receipts`, `audit`)",
    "ml/flock/python/verity_flock/rec_*.py": "V*'s builders (rec_outer, rec_residuals, rec_vstage) state the outer statement "
                                             "over sampled proofs' `one_stage.partition` and `one_stage.registered`, which "
                                             "use only verity.ml; the follow-up to T2 moves them into ml "
                                             "(`verity/circuits/population.py`, `verity/ml/commitments/registered.py`), "
                                             "core imports them from there, and this entry then covers only rec_live's coin "
                                             "server, which takes the inner session's coin tree from the challenger's device "
                                             "through sampled proofs' `remote.RemoteChallenger` (`--tree-from`) and holds "
                                             "its sessions to the device's release log (`challenger.coin_trees_of`)",
    "ml/flock/pod/rec_inner.py": "stages the inner statement over sampled proofs' `one_stage.partition`, as V*'s builders "
                                  "do (the rec_* entry), and moves with them in the follow-up to T2",
    "ml/flock/pod/rec_reprice.py": "builds the inner statement's population program and partition with sampled proofs' "
                                    "`one_stage.partition`, as rec_inner does, and moves with it in the follow-up to T2",
    "ml/flock/python/verity_flock/inner_registered.py": "registers a staged inner statement's description rows with sampled "
                                                        "proofs' `one_stage.registered`, as V*'s builders do (the rec_* "
                                                        "entry), and moves with them in the follow-up to T2",
    "ml/protocols/pouw/*": "PoUW's circuits and schemes, which its verifier doesn't read, are stated over the protocol's "
                           "parameters, records and Python references in core (`pouw.protocol`, `registration`, `window`, "
                           "`spot.one_stage`, ...); they shrink as those references leave core",
    "ml/protocols/certifier/*": "the certifier's Boolean Definitions that core doesn't read extend the ones it does "
                                "(`head_check`'s gates, `cache_step`, `compile_slice`, `replay_chain`, `arith_encode`, "
                                "`io_tie`, ...) and are stated over its advice and link-window layout in core (`advice`, "
                                "`active`, `served_rows`, ...)",
}
#: A verifier imports no ``verity.ml`` code (AGENTS.md "Design rules"): Lean is the definition and the
#: verifier, and ml's Python mirrors it.  Until a protocol's verifier is its Lean (the switch procedure,
#: ``.agents/skills/friction/SKILL.md``), its Python verifier keeps the ml edges ``ALLOWED`` gave it when they were frozen here,
#: and gains none; a core unit not listed has none.  When ``ALLOWED`` drops an edge, drop it here too.
CORE_ML_EDGES = {
    "core.protocols.profile": {"ml.circuits"},
    "core.protocols.spot": {"ml.circuits", "ml.commitments", "ml.randomness"},
    "core.protocols.pous": {"ml.commitments", "ml.crypto", "ml.randomness"},
    "core.protocols.bandwidth": {"ml.crypto", "ml.randomness"},
    "core.protocols.remote_bandwidth": {"ml.commitments", "ml.randomness"},
    "core.protocols.pouw": {"ml.circuits", "ml.commitments", "ml.crypto", "ml.randomness"},
    "core.protocols.certifier": {"ml.circuits", "ml.commitments", "ml.crypto"},
}
#: The units of ``verity.ml`` core's Python verifiers read, directly or through one another (what ``ALLOWED`` lets core
#: reach): their verdicts rest on them, so they are on the soundness path, and AGENTS.md names each.  Frozen
#: with ``CORE_ML_EDGES``: it only shrinks, as each protocol's verifier becomes its Lean.  The rest of ml (the
#: catalog, the kernels, the protocols' circuit builders, Flock's prover, the private circuits) is not on it.
SOUNDNESS_PATH = {"ml.circuits", "ml.commitments", "ml.crypto", "ml.randomness"}
#: What a test that refuses a new core -> ml edge says.
LEAN_OF_RECORD = ("a verifier imports no verity.ml code (AGENTS.md \"Design rules\", \"Lean is the definition "
                  "and the verifier of record\"): state it in Lean and import nothing from verity.ml")
#: SPOT[Flock] stands on its own, and the network certifier is a layer beside it: SPOT reads the certifier's stamp as a
#: certified input and imports none of the certifier, and the two compose only above both.  SPOT's Python units, the
#: trees each of whose Python files (tests included) is checked, its Lean packages and its analysis (Security's Flock and
#: Spot areas and the sampling claims); then the certifier's Python units, modules and Lean (its packages and its
#: Security areas).
SPOT_PYTHON = ("core.protocols.spot",)
SPOT_TREES = ("verity/spot", "verity/spot/verifier/lean", "verity/flock/verifier",
              "verity/flock/prover")
SPOT_LEAN = ("verity/spot/verifier/lean", "verity/flock/verifier")
SPOT_SECURITY = ("Security/*/Flock", "Security/*/Flock.lean", "Security/*/Spot", "Security/*/Spot.lean",
                 "Security/Definitions/Claims/Sampling*")
CERTIFIER_PYTHON = ("core.protocols.certifier", "ml.protocols.certifier")
CERTIFIER_MODULES = ("verity.protocols.certifier.verifier", "verity.protocols.certifier.prover", "verity.protocols.certifier.device")
CERTIFIER_LEAN = ("verity/protocols/certifier/device/program/lean", "verity/protocols/certifier/device/lean",
                  "verity/protocols/certifier/verifier/lean", "Security/*/NetworkCertifier", "Security/*/NetworkCertifier.lean")
SPOT_NOT_CERTIFIER = ("SPOT[Flock] never imports the network certifier, which composes with it only above both and whose "
                      "stamp SPOT reads as a certified input")
#: Files outside ``infra/`` that import a package of it, each with why.
INFRA_IMPORTERS = {
    "tools/research/console/verity_console.py": "the console reads `cluster.usage`, and moves to infra/ with research's pods, "
                                                "deploy and Slack code",
}
#: Files outside ``benchmarks/`` that import a module of it, by name or by path, each with its owner and what it imports.
#: ``benchmarks/`` is measurement only: a number a protocol uses reaches it
#: through a card file citing an ``art:`` id, never as an import, and code a protocol runs moves out to the package that
#: uses it.  It only shrinks.
BENCHMARKS_IMPORTERS = {
    # the H200 example: its runner loads benchmark code by path
    "archive/examples/h200/e2e_h200/attention.py": "compute-accounting: attn_bf16's reference kernels (`benchmarks/torch_gpu/attn_bf16`)",
    "archive/examples/h200/e2e_h200/commit.py": "compute-accounting: kernels' commit_slots (`benchmarks/torch_gpu/commit_slots`)",
    "archive/examples/h200/e2e_h200/percall.py": "compute-accounting: J10-R's sm_90 build (`benchmarks/torch_gpu/j10r_sm90`)",
    "archive/examples/h200/e2e_h200/store.py": "compute-accounting: P2's store (`benchmarks/pous`) and the staging decode "
                                       "(`benchmarks/torch_gpu/p2_stage`)",
    "benchmarks/one_stage/interfaces/verity_e2e_demo/replay_bundle.py": "compute-accounting: kernels' commit_slots "
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
    # Flock's measurement campaign, its tests, and the frozen instance sets `instances` reads
    **dict.fromkeys((
        "verity/flock/pod/gemm_hill.py", "verity/flock/pod/read_gadget.py",
        "verity/flock/prover/python/verity_flock/bench.py", "verity/flock/prover/python/verity_flock/circuit_bench.py",
        "verity/flock/prover/python/verity_flock/instances.py", "verity/flock/prover/python/verity_flock/ir_bench.py",
        "verity/flock/prover/python/verity_flock/register.py",
        "verity/flock/prover/tests/test_backend_bench.py", "verity/flock/prover/tests/test_class_statement.py",
        "verity/flock/prover/tests/test_gemm_hill.py", "verity/flock/prover/tests/test_ir_lowering_bench.py",
    ), "proofs: `verity_numerical.bench` (`benchmarks/numerical`)'s contract, cell, generator suites and frozen instances"),
    # research's registry of measuring tools
    **dict.fromkeys(("tools/research/src/research/store/tools_registry.py", "tools/research/tests/test_store_prov.py"),
                    "infra: the store's tool registry names `benchmarks.<probe>.tool` modules (tc_probe, tc_probe_fp4, "
                    "native_peak, commitments, pouw)"),
    # tests and vector scripts that check against a benchmark's reference
    **dict.fromkeys((
        "verity/protocols/pouw/verifier/tests/epoch_coins_vectors.py", "verity/protocols/pouw/verifier/tests/test_pouw_audit.py",
        "verity/protocols/pouw/verifier/tests/test_pouw_epoch_coins.py", "verity/protocols/pouw/verifier/tests/test_pouw_pearl_c.py",
        "verity/protocols/pouw/verifier/tests/test_pouw_pearl_c4.py", "verity/protocols/pouw/verifier/tests/test_pouw_pearl_c_sm120.py",
        "verity/protocols/pouw/verifier/tests/test_pouw_pearl_c_work.py", "verity/protocols/pouw/verifier/tests/test_pouw_pearl_kw.py",
        "verity/protocols/pouw/verifier/lean/commit/tests/difftest_vectors.py",
    ), "compute-accounting: PoUW's clear lifecycle (`verity/protocols/pouw/clear`, and `pearl_c4`)"),
    "verity/protocols/pouw/verifier/lean/window/tests/test_known_rows.py": "proofs: H1's known rows "
                                                                       "(`benchmarks/pouw/h1_lanes/run.py`)",
    **dict.fromkeys(("verity/protocols/pous/verifier/check/scripts/difftest_vectors.py",
                     "verity/protocols/pous/verifier/check/tests/test_timed_audit.py"),
                    "memory-accounting: P2's reference and wire (`benchmarks/pous/p2_v1`) and the harness's control "
                    "responder (`benchmarks/pous/band_gpu/harness.py`)"),
    "verity/definitions/ops/tests/test_tc_probe_families.py": "circuits: the tensor-core probe's families "
                                                              "(`benchmarks/tc_probe`)",
    **dict.fromkeys(("integrations/vllm/tests/protocols/pouw_h1_circuits.py",
                     "integrations/vllm/tests/protocols/pouw_h1_gpu.py",
                     "integrations/vllm/tests/protocols/test_pouw_h1_circuits.py"),
                    "compute-accounting: H1's prover, producer and the live challenger's auditor "
                    "(`benchmarks/pouw/h1_prove`, `served_h1`, `verity/composite/device/live_challenger`)"),
    **dict.fromkeys(("integrations/vllm/tests/protocols/challenger_host.py", "integrations/vllm/tests/protocols/local_gate.py"),
                    "proofs: the live challenger's gate and model registry (`verity/composite/device/live_challenger`)"),
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
    "_build_unchecked": ("verity/circuits/partition_v2.py",
                         "`partition_v2.build` without its refusal of units of several size classes, for the tests "
                         "of `check` on advice past the builder's scope, as a prover may send it"),
}

#: Modules of ``verity.ml`` that only a prover's or developer's build turns on, which nothing in ``verity/core`` names (an
#: import, an attribute or a string), each with the module and why a verifier must not read it.
PROVER_ONLY = {
    "trace_cache": ("verity/definitions/ops/boolean/trace_cache.py",
                    "traced circuits reused across a prover's builds: a verifier reading them would take a stale or tampered "
                    "entry for a Definition's gates"),
}

MODULE_ATTR = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*")
IMPORT_CALLS = {"import_module", "__import__", "importorskip"}


def _separate(rel: Path) -> bool:
    return any(rel.as_posix() == s or rel.as_posix().startswith(s + "/") for s in SEPARATE)


def _sources(root: Path = SRC) -> list[Path]:
    """``verity``'s own Python files: every ``.py`` under ``root`` outside a ``tests/`` or ``lean/`` directory, a Lake build
    directory (core's Lean builds in ``verity/flock/verifier/.lake``) or a ``SEPARATE`` tree, and no circuit-check
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


#: A read of another package at run time, matched in the text: its package data (``importlib.resources``) or a module an import
#: call names by a string, an f-string included (`_imports` sees only a whole constant handed to an import call).
RUN_TIME_READ = re.compile(r"""\b(?:files|read_text|read_binary|open_text|open_binary|import_module|__import__|find_spec)\("""
                           r"""\s*[rRbBuUfF]{0,2}["'](?P<name>[A-Za-z_]\w*)""")


def _run_time_reads(text: str) -> list[tuple[int, str]]:
    """``(line, name)`` for each read in ``text`` that `RUN_TIME_READ` matches and whose top-level name is ``FORBIDDEN_EXTERNAL``."""
    return [(text.count("\n", 0, m.start()) + 1, m["name"]) for m in RUN_TIME_READ.finditer(text) if m["name"] in FORBIDDEN_EXTERNAL]


def test_verity_reads_no_other_distribution_at_run_time() -> None:
    """No Python file under ``verity/`` (tests, Lean scripts and sidecars included; a ``SEPARATE`` tree is a distribution of its own)
    reads another distribution's package data or names one of its modules for an import call, so ``verity`` alone evaluates its
    Definitions and no suite of it takes another distribution as an input."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if {".lake", "__pycache__"} & set(rel.parts) or _separate(rel):
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


def _reached(allowed: dict[str, set[str]], units: Iterable[str]) -> set[str]:
    """The units ``units`` reach in ``allowed``, directly or through one another."""
    reached, todo = set(), list(units)
    while todo:
        for v in allowed.get(todo.pop(), set()) - reached:
            reached.add(v)
            todo.append(v)
    return reached


def _core_reach_in_ml(allowed: dict[str, set[str]]) -> set[str]:
    """The ml units the core units of ``allowed`` reach, directly or through one another."""
    return {u for u in _reached(allowed, [u for u in allowed if u.startswith("core.")]) if u.startswith("ml.")}


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


def _covered(rel: str, globs: Iterable[str]) -> bool:
    """Whether ``rel`` is a path ``globs`` name or lies under one (``*`` crosses directories, as ``fnmatch``'s does)."""
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(rel, g + "/*") for g in globs)


def _spot_python_offenders() -> list[str]:
    """Each import of a certifier module (``CERTIFIER_MODULES``) in a Python file of ``SPOT_TREES``, its tests included."""
    out = []
    for tree in SPOT_TREES:
        for path in sorted((ROOT / tree).rglob("*.py")):
            if {".lake", "__pycache__"} & set(path.relative_to(ROOT).parts):
                continue
            out += [f"{path.relative_to(ROOT)}:{line}: {name}" for name, line, _ in _imports(path)
                    if any(name == m or name.startswith(m + ".") for m in CERTIFIER_MODULES)]
    return out


def test_spot_never_imports_the_network_certifier_in_python() -> None:
    """SPOT[Flock]'s Python imports no unit of the network certifier (``SPOT_PYTHON``): ``ALLOWED`` lets sampled proofs
    reach none of ``CERTIFIER_PYTHON``, and no file of its trees, Flock's and their tests among them, imports one."""
    assert set(CERTIFIER_PYTHON) <= set(ALLOWED), f"CERTIFIER_PYTHON names a unit ALLOWED doesn't: {CERTIFIER_PYTHON}"
    reached = sorted(_reached(ALLOWED, SPOT_PYTHON) & set(CERTIFIER_PYTHON))
    assert not reached, f"{SPOT_NOT_CERTIFIER}: ALLOWED lets {SPOT_PYTHON} reach {reached}"
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
    assert _covered("Security/Proofs/Properties/NetworkCertifier/Check/Bytes.lean", CERTIFIER_LEAN)
    assert not _covered("Security/Proofs/Flock/Bytes.lean", CERTIFIER_LEAN)
    modules = _lean_modules(["Security/Proofs/Flock/Bytes.lean", "verity/flock/verifier/Flock/Bytes.lean"],
                            ["Security/Proofs/lakefile.toml", "verity/flock/verifier/lakefile.toml"])
    assert modules == {"Proofs.Flock.Bytes": ["Security/Proofs/Flock/Bytes.lean"],
                       "Flock.Bytes": ["verity/flock/verifier/Flock/Bytes.lean"]}, modules


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


#: The listing: the tree's directories, bottom first; the directories under ``verity/`` that are distributions of their own;
#: the ties against the order; and for each verifier and device, what its Python loads and reaches and what loads it.
ORDER = ROOT / "tests" / "import_order.toml"
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


#: The lists of a verifier's or a device's entry in the listing.
TRUSTED = ("loads", "reaches", "loaded_by")


class Listing(NamedTuple):
    """``ORDER`` as the tests read it."""

    steps: dict[str, int]                               # each listed directory's step: the bottom row's 0, one a row
    distributions: dict[str, dict[str, str]]            # each distribution of its own under ``verity/``, by directory
    against: dict[tuple[str, str], set[str]]            # each excepted pair of rows: the first's files that load the second
    #: each verifier's and device's entry, by its path: the places its Python ``loads`` and ``reaches``, and the files it is ``loaded_by``
    trusted: dict[str, dict[str, set[str]]]


def _listing(text: str | None = None) -> Listing:
    """The listing (``ORDER``, or ``text``)."""
    doc = tomllib.loads(ORDER.read_text(encoding="utf-8") if text is None else text)
    steps = {d: n for n, row in enumerate(doc["order"]) for d in row}
    assert "." in steps and len(steps) == sum(map(len, doc["order"])), "the order lists `.` and each directory once"
    return Listing(steps, {e["directory"]: {k: v for k, v in e.items() if k != "directory"} for e in doc.get("distribution", [])},
                   {(e["loader"], e["loaded"]): set(e["files"]) for e in doc.get("against", [])},
                   {e["path"]: {k: set(e.get(k, ())) for k in TRUSTED} for e in doc.get("trusted", [])})


def _row(path: str, steps: dict[str, int]) -> str:
    """The listed directory ``path`` belongs to: the deepest one above it, ``.`` for the rest."""
    d = posixpath.dirname(path)
    while d and d not in steps:
        d = posixpath.dirname(d)
    return d or "."


def _place(path: str, steps: dict[str, int]) -> str:
    """Where ``path`` is, as a verifier's lists name it: the directory one level below its row that holds it (a protocol's
    ``verifier/``, ``prover/`` or ``device/``, the directory of one package of a row), or its row for a file directly in it."""
    row = _row(path, steps)
    below = posixpath.dirname(path)[0 if row == "." else len(row) + 1:].split("/")[0]
    return row if not below else below if row == "." else f"{row}/{below}"


def _part(path: str, name: str) -> str | None:
    """The outermost directory above ``path`` named ``name`` (``verifier``, ``prover``), or None."""
    parts = path.split("/")[:-1]
    return "/".join(parts[:parts.index(name) + 1]) if name in parts else None


def _trusted(path: str, entries: Iterable[str] = ()) -> str | None:
    """The verifier's or the device's directory ``path`` is in: the outermost ``verifier/`` or ``device/`` directory above
    it under ``verity/``, or None.  Where ``entries`` (the listing's ``trusted``) name a file or a sub-directory of it, the
    deepest of them at or above ``path`` instead: the directory's own entry holds what no other names."""
    parts = path.split("/")
    at = next((i for i, part in enumerate(parts[:-1]) if part in ("verifier", "device")), None)
    if at is None or parts[0] != "verity":
        return None
    entries = set(entries)
    return next((e for n in range(len(parts), at + 1, -1) if (e := "/".join(parts[:n])) in entries), "/".join(parts[:at + 1]))


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


def _python_loads(tree: Any, f: str) -> list[Load]:
    """What the Python file ``f`` loads, at load or inside a function: for each module it names (``_imports``; a relative
    name resolved from the root), the files ``_named`` gives, and for ``from M import a`` the module ``M.a`` when it is one."""
    text = tree.text(f)
    parsed = ast.parse(text, filename=f)
    path, directory, roots = tree.root / f, posixpath.dirname(f), _path_roots(tree, f, parsed, text)
    nodes = list(ast.walk(parsed))
    froms = [n for n in nodes if isinstance(n, ast.ImportFrom) and n.module]
    names = _imports(path, tree.root, (), nodes)
    names += [(f"{name}.{a.name}", line, True) for n, (name, line, _) in zip(froms, _imports(path, tree.root, (), froms))
              for a in n.names]
    out: dict[tuple[str, str], Load] = {}
    for name, line, imported in sorted(names, key=lambda n: n[1]):
        modules, above = _named(tree, name, directory, roots)
        for kind, files in (("python import" if imported else "python string", modules), ("python package", above)):
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
def _tree_loads() -> tuple[frozenset[str], tuple[Load, ...]]:
    """The tracked tree's directories and its loads (``_all_loads``), read once for the tests below."""
    tree = _public_boundary().Tree(ROOT)
    return tree.dirs, _all_loads(tree)


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


def _trusted_loads(loads: Iterable[Load], listing: Listing) -> dict[str, tuple[dict[str, Load], dict[str, list[str]], dict[str, Load]]]:
    """For each verifier's and each device's entry (``_trusted``, split as the listing's ``trusted`` splits a directory):
    the places outside it that its Python (``_source``) loads, each with one such load; the places it reaches through
    those files' own loads and so on, each with the chain of files to it; and the Python files under ``verity/`` outside
    every verifier and device that load a file of it, a test or a binding included, each with one such load.  A place is
    another entry, or outside them the directory one level below a row (``_place``).  Every load counts, at load, inside
    a function or named in a string."""
    entries = set(listing.trusted)
    at = functools.cache(lambda f: _trusted(f, entries) or _place(f, listing.steps))
    by: dict[str, list[Load]] = {}
    loaders: dict[str, dict[str, Load]] = {}
    for x in loads:
        if x.loader.endswith(".py"):
            if _held(x):
                by.setdefault(x.loader, []).append(x)
            if x.loader.startswith("verity/") and _trusted(x.loader) is None and _trusted(x.loaded) is not None:
                loaders.setdefault(at(x.loaded), {}).setdefault(x.loader, x)
    own: dict[str, list[str]] = {}
    for f in sorted(by):
        if _source(f, listing) and _trusted(f) is not None:
            own.setdefault(at(f), []).append(f)
    out = {}
    for d in sorted(own.keys() | loaders.keys()):
        files = own.get(d, [])
        inside = lambda f: at(f) == d and _distribution(f, listing) is None  # noqa: E731, B023
        direct: dict[str, Load] = {}
        for x in (x for f in files for x in by[f]):
            if not inside(x.loaded):
                direct.setdefault(at(x.loaded), x)
        via: dict[str, str | None] = dict.fromkeys(files)
        todo = files[::-1]
        while todo:
            f = todo.pop()
            for x in by.get(f, ()):
                if x.loaded not in via:
                    via[x.loaded] = f
                    todo.append(x.loaded)
        reach: dict[str, list[str]] = {}
        for f in sorted(via):
            if not inside(f) and at(f) not in reach:
                chain = [f]
                while (before := via[chain[-1]]) is not None:
                    chain.append(before)
                reach[at(f)] = chain[::-1]
        out[d] = (direct, reach, loaders.get(d, {}))
    return out


def _hold_trusted(loads: Iterable[Load], listing: Listing) -> None:
    found = _trusted_loads(loads, listing)
    listed = lambda d, kind: listing.trusted.get(d, {}).get(kind, ())  # noqa: E731
    grown = [f"{d} loads {at}: {x}" for d, (direct, _, _) in found.items()
             for at, x in sorted(direct.items()) if at not in listed(d, "loads")]
    grown += [f"{d} reaches {at}: {' -> '.join(chain)}" for d, (_, reach, _) in found.items()
              for at, chain in sorted(reach.items()) if at not in listed(d, "reaches")]
    assert not grown, ("a verifier or a device gains no load, and what it reaches gains no place (`trusted` in "
                       "tests/import_order.toml only shrinks): state it in Lean, or take the value as an argument:\n"
                       + "\n".join(grown))
    new = [f"{d} is loaded by {x}" for d, (_, _, loaders) in found.items()
           for f, x in sorted(loaders.items()) if f not in listed(d, "loaded_by")]
    assert not new, ("under verity/, outside the verifiers and the devices, only the files a verifier's or a device's "
                     "`loaded_by` lists load it (`trusted` in tests/import_order.toml), and the list only shrinks: move "
                     "what the file needs to a row both load, or pass it in:\n" + "\n".join(new))
    stale = [f"{d}: {kind} {at}" for d, lists in sorted(listing.trusted.items())
             for kind, now in zip(TRUSTED, found.get(d, ({}, {}, {}))) for at in sorted(lists[kind] - set(now))]
    stale += [f"{d}: no load and no loader" for d, lists in sorted(listing.trusted.items()) if d not in found and not any(lists.values())]
    assert not stale, ("entries of `trusted` (tests/import_order.toml) the tree no longer has; delete them, and an "
                       f"entry left with none: {stale}")


def _hold_distributions(dirs: Set[str], loads: Iterable[Load], listing: Listing) -> None:
    form = [d for d, e in sorted(listing.distributions.items())
            if not (d.startswith("verity/") and e.get("owner") and len(e.keys() & {"date", "reason"}) < 2
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2}", e.get("date", "2000-01-01")))]
    assert not form, ("each `distribution` of tests/import_order.toml is a directory under verity/ with its owner and, of "
                      f"the date it becomes part of `verity` (YYYY-MM-DD) and the reason it stays its own, at most one: {form}")
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
    delete once its last file is gone.  Reading every tracked Python and Lean file takes some twenty seconds, which
    nothing cheaper replaces: an order over the whole tree is only held by reading the whole tree."""
    listing = _listing()
    dirs, loads = _tree_loads()
    empty = sorted(d for d in listing.steps if d != "." and d not in dirs)
    assert not empty, f"directories tests/import_order.toml lists that hold no tracked file; delete their rows: {empty}"
    _hold_order(loads, listing)


def test_a_verifier_loads_no_prover_code() -> None:
    """No file under a ``verifier/`` directory loads a file under a ``prover/`` directory, with no exception.  It is an
    assertion of its own and no consequence of the order, which lists a protocol's verifier and its prover in one row.
    The form: a verifier is what the tree has under ``verifier/``, Lean and Python alike, so a protocol's Python verifier
    is held as its Lean is until its Lean decides its verdicts; a file changes sides by moving between ``verifier/`` and
    ``prover/``, and what both sides read is the verifier's, which the prover loads.  Every load counts: at load, inside a
    function, a module named in a string, a Lean import, a Lake requirement.  Tests are no loaders here, and neither are
    the scripts beside a verifier's Lean (``_lean_script``)."""
    found = _to_provers(_tree_loads()[1])
    assert not found, ("a verifier loads no prover code: move what both sides read into the verifier's directory, where "
                       "the prover loads it, or state it in Lean:\n" + "\n".join(map(str, found)))


def test_a_verifier_or_a_device_gains_no_load_and_no_loader() -> None:
    """Until a protocol's Lean verifier decides its verdicts, its Python verifier does, and that Python and everything it
    loads is on the soundness path; a device's Python is trusted the same way.  The listing's ``trusted`` entries freeze
    it for each ``verifier/`` and ``device/`` directory under ``verity/``: ``loads``, every place outside it that its
    Python loads, other verifiers and devices among them; ``reaches``, the places those files load in turn, and so on;
    and ``loaded_by``, the Python files under ``verity/`` outside every verifier and device that load it, a prover's
    and a test's alike.  A place is another entry, or outside them the directory one level below a row (``_place``).
    Where one directory holds two things that are held apart, the listing names a file or a sub-directory of it as an
    entry of its own, the directory's entry is the rest, and a load between the two is held as any other.  All three
    lists only shrink, as each verifier becomes its Lean and what a prover needs leaves its verifier: a place or a file
    the tree adds fails, a place with the chain of files that reaches it, and so does a listed one the tree no longer
    has.  AGENTS.md names each row reached."""
    listing = _listing()
    _hold_trusted(_tree_loads()[1], listing)
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    rows = {_row(at + "/_", listing.steps) for lists in listing.trusted.values() for at in lists["reaches"]}
    missing = sorted(r for r in rows if f"`{r}`" not in agents and f"`{r}/`" not in agents)
    assert not missing, f"AGENTS.md doesn't name {missing}, which the verifiers' and the devices' Python reaches"


def test_verity_loads_no_distribution_of_its_own() -> None:
    """A directory under ``verity/`` with an import root of its own is a distribution of its own (the listing's
    ``distribution``, each with its owner and the date it becomes part of ``verity`` or the reason it stays its own), and
    ``verity`` loads none of them, a verifier's Python least of all: no Python of ``verity`` (``_source``) loads a file of
    one, by its own name or as a module of ``verity``.  The list only shrinks: a listed directory with no tracked file
    fails, and a directory not listed is ``verity``'s, held to its rules."""
    dirs, loads = _tree_loads()
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
    "verity/flock/__init__.py": "",
    "verity/flock/prover/stage.py": "import verity.protocols.pouw.verifier.check\n",
    "verity/flock/verifier/__init__.py": "",
    "verity/flock/verifier/interface.py": "",
    "verity/flock/verifier/lakefile.toml": ('name = "flock"\n[[lean_lib]]\nname = "Flock"\n'
                                            '[[require]]\nname = "spot"\npath = "../../spot/lean"\n'),
    "verity/flock/verifier/Flock/V.lean": "import Spot.X\nimport Mathlib.Y\nimport Flock.W\n",
    "verity/flock/verifier/Flock/W.lean": "",
    "verity/difftest/package_above.py": "import verity.flock.verifier.interface\n",
    "verity/spot/lean/lakefile.toml": 'name = "spot"\n[[lean_lib]]\nname = "Spot"\n',
    "verity/spot/lean/Spot/X.lean": "",
    "verity/spot/by_path.py": ("import sys\nfrom pathlib import Path\n"
                               "sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'workloads' / 'e2e'))\nimport prof\n"),
    "workloads/e2e/prof.py": "",
    "tools/lazy.py": "def f():\n    from benchmarks.probe import run\n",
    "benchmarks/probe/run.py": "import tools.lazy\n",
    "benchmarks/probe/tests/test_run.py": "import infra.cluster.usage\n",
    "infra/cluster/usage.py": "import benchmarks.probe.run\n",
    "verity/protocols/__init__.py": "",
    "verity/protocols/pouw/__init__.py": "",
    "verity/protocols/pouw/verifier/__init__.py": "",
    "verity/protocols/pouw/verifier/check.py": ("from verity.circuits import at_load\nfrom verity.commitments.crypto import lms\n\n\n"
                                                "def f():\n    from verity.protocols.pouw.prover import build\n"),
    "verity/protocols/pouw/verifier/lean/scripts/vectors.py": "import verity.protocols.pouw.prover.build\n",
    "verity/protocols/pouw/verifier/tests/test_check.py": "import verity.protocols.pouw.prover.build\n",
    "verity/protocols/pouw/verifier/check/interop.py": "import verity.flock.prover.stage\n",
    "verity/protocols/pouw/device/__init__.py": "",
    "verity/protocols/pouw/device/serve.py": "from verity.protocols.pouw.verifier import check\n",
    "verity/protocols/pouw/device/party.py": "from verity.protocols.pouw.device import serve\n",
    "verity/protocols/pouw/prover/__init__.py": "",
    "verity/protocols/pouw/prover/build.py": "from verity.protocols.pouw.verifier import check\n",
    "verity/protocols/pouw/prover/tests/test_build.py": "from verity.protocols.pouw.device import serve\n",
}


def test_the_listing_checks_find_what_they_refuse(tmp_path: Path) -> None:
    """The four checks above aren't vacuous.  Against the listing's order: an import at load, inside a function and as a
    module named in a string, the package above a module of an earlier row, a bare import under a directory put on
    ``sys.path``, a Lean import from a package required by path and that requirement, and a test's import of the last
    row; not a load with the order, a test's of another row, a sidecar's, or a module no tracked file holds.  From a
    verifier: its import of its prover inside a function and that of a distribution inside it, the places it loads, at
    the grain of a row's packages, and the place it reaches only through another file; not its prover's import of it,
    its test's or its Lean script's of the prover, nor, among its places, what a distribution inside it loads.  Into a
    verifier or a device: a prover's file and a prover's test; not a device's file.  Where the listing names a file of a
    device as an entry of its own: that file's load of the rest of its directory."""
    for f, text in ORDER_TREE.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text(text, encoding="utf-8")
    tree = _public_boundary().Tree(tmp_path, files=ORDER_TREE)
    loads = _all_loads(tree)
    verifier, prover, device, check = (f"verity/protocols/pouw/{d}" for d in ("verifier", "prover", "device", "verifier/check"))
    own = {"verity/flock": {"owner": "proofs", "reason": "Flock's distribution"}, check: {"owner": "memory-accounting"}}
    steps = _listing().steps
    listing = Listing(steps, own, {}, {})
    found = {pair: sorted(files) for pair, files in _against(loads, steps).items()}
    assert found == {
        ("verity/circuits", "verity/definitions"): [f"verity/circuits/{f}.py" for f in ("at_load", "in_function", "in_string")],
        ("verity/difftest", "verity/flock"): ["verity/difftest/package_above.py"],
        ("verity/flock/verifier", "verity/spot"): ["verity/flock/verifier/Flock/V.lean", "verity/flock/verifier/lakefile.toml"],
        ("verity/flock", "verity/protocols/pouw"): ["verity/flock/prover/stage.py"],
        ("verity/spot", "workloads"): ["verity/spot/by_path.py"],
        ("tools", "benchmarks"): ["tools/lazy.py"],
        ("benchmarks", "infra"): ["benchmarks/probe/tests/test_run.py"],
    }, found
    assert _row("verity/settings.py", steps) == "verity" and _row("pyproject.toml", steps) == "." \
        and _row("verity/flock/verifier/Flock/V.lean", steps) == "verity/flock/verifier"
    assert [_place(f, steps) for f in ("verity/commitments/crypto/lms.py", "verity/commitments/merkle.py", "verity/errors.py",
                                       f"{prover}/circuit/words.py", "examples/a/b.py", "conftest.py")] == [
        "verity/commitments/crypto", "verity/commitments", "verity", prover, "examples", "."]
    assert [_distribution(f, listing) for f in ("verity/flock/prover/stage.py", "verity/flock/verifier/interface.py",
                                                f"{check}/interop.py", f"{verifier}/check.py")] == ["verity/flock", None, check, None]
    assert [_trusted(f) for f in (f"{check}/interop.py", f"{device}/serve.py", f"{prover}/build.py", "tools/verifier/run.py")] == [
        verifier, device, None, None]
    party = f"{device}/party.py"
    assert [_trusted(f, {party, check}) for f in (party, f"{device}/serve.py", f"{check}/interop.py", f"{prover}/build.py")] == [
        party, device, check, None]
    assert sorted({(x.loader, x.line, x.loaded) for x in _to_provers(loads)}) == [
        (f"{verifier}/check.py", 6, f"{prover}/__init__.py"), (f"{verifier}/check.py", 6, f"{prover}/build.py"),
        (f"{check}/interop.py", 1, "verity/flock/prover/stage.py")]
    assert [str(x) for x in _into_distributions(loads, listing)] == [
        "verity/difftest/package_above.py:1: python package verity.flock.verifier.interface loads verity/flock/__init__.py"]
    found = _trusted_loads(loads, listing)
    assert list(found) == ["verity/flock/verifier", device, verifier], list(found)
    direct, reach, loaders = found[verifier]
    assert sorted(direct) == ["verity", "verity/circuits", "verity/commitments", "verity/commitments/crypto",
                              "verity/protocols", "verity/protocols/pouw", prover], direct
    assert sorted(reach) == [*sorted(direct)[:4], "verity/definitions", "verity/protocols", "verity/protocols/pouw", prover], reach
    assert reach["verity/definitions"] == [f"{verifier}/check.py", "verity/circuits/at_load.py", "verity/definitions/__init__.py"]
    assert sorted(loaders) == ["verity/flock/prover/stage.py", f"{prover}/build.py"], loaders
    assert [sorted(x) for x in found[device]] == [["verity", "verity/protocols", "verity/protocols/pouw", verifier],
                                                  sorted([*reach, verifier]), [f"{prover}/tests/test_build.py"]], found[device]
    assert [sorted(x) for x in found["verity/flock/verifier"]] == [[], [], ["verity/difftest/package_above.py"]]
    split = _trusted_loads(loads, listing._replace(trusted={party: {}}))
    assert list(split) == ["verity/flock/verifier", device, party, verifier], list(split)
    assert [sorted(x) for x in split[device]] == [sorted(x) for x in found[device]], split[device]
    assert [sorted(x) for x in split[party]] == [["verity", "verity/protocols", "verity/protocols/pouw", device],
                                                 sorted([*reach, device, verifier]), []], split[party]

    held = Listing(steps, own, {pair: set(files) for pair, files in _against(loads, steps).items()},
                   {d: dict(zip(TRUSTED, map(set, lists))) for d, lists in found.items()})
    _hold_order(loads, held)
    _hold_trusted(loads, held)
    _hold_distributions(tree.dirs, [x for x in loads if "package_above" not in x.loader], held)
    pair = ("verity/circuits", "verity/definitions")
    with pytest.raises(AssertionError, match=r"no file needs any more; delete them: \['verity/windows -> tools'\]"):
        _hold_order(loads, held._replace(against=held.against | {("verity/windows", "tools"): {"verity/windows/rows.py"}}))
    with pytest.raises(AssertionError, match="only shrinks.*\n.*in_string.py:1: python string verity.definitions.ops loads"):
        _hold_order(loads, held._replace(against=held.against | {pair: held.against[pair] - {"verity/circuits/in_string.py"}}))
    with pytest.raises(AssertionError, match="only shrinks.*\n.*test_run.py:1: python import infra.cluster.usage loads"):
        _hold_order(loads, held._replace(against={p: files for p, files in held.against.items() if p[1] != "infra"}))
    with pytest.raises(AssertionError, match="no longer load a file of the later row; delete them:\n.*verity/circuits/gone.py"):
        _hold_order(loads, held._replace(against=held.against | {pair: held.against[pair] | {"verity/circuits/gone.py"}}))

    def without(d: str, **gone: str) -> Listing:
        return held._replace(trusted=held.trusted | {d: {kind: places - {gone.get(kind)} for kind, places in held.trusted[d].items()}})

    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{verifier} loads {prover}: .*\n\\s*{verifier} reaches {prover}: "):
        _hold_trusted(loads, without(verifier, loads=prover, reaches=prover))
    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{verifier} loads verity/commitments/crypto: "):
        _hold_trusted(loads, without(verifier, loads="verity/commitments/crypto"))
    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{verifier} reaches verity/definitions: {verifier}/check.py -> "):
        _hold_trusted(loads, without(verifier, reaches="verity/definitions"))
    with pytest.raises(AssertionError, match=f"only shrinks.*\n\\s*{device} is loaded by {prover}/tests/test_build.py:1: "):
        _hold_trusted(loads, without(device, loaded_by=f"{prover}/tests/test_build.py"))
    with pytest.raises(AssertionError, match=f"only shrinks.*\n\\s*{verifier} is loaded by {prover}/build.py:1: "):
        _hold_trusted(loads, without(verifier, loaded_by=f"{prover}/build.py"))
    with pytest.raises(AssertionError, match=f"delete them, and an entry left with none: \\['{verifier}: reaches tools'"):
        _hold_trusted(loads, held._replace(trusted=held.trusted | {
            verifier: held.trusted[verifier] | {"reaches": held.trusted[verifier]["reaches"] | {"tools"}}}))
    with pytest.raises(AssertionError, match="left with none: \\['verity/spot/verifier: loaded_by verity/spot/plan.py'"):
        _hold_trusted(loads, held._replace(trusted=held.trusted | {
            "verity/spot/verifier": {"loads": set(), "reaches": set(), "loaded_by": {"verity/spot/plan.py"}}}))
    with pytest.raises(AssertionError, match=f"left with none: \\['{device}/gone.py: no load and no loader'"):
        _hold_trusted(loads, held._replace(trusted=held.trusted | {f"{device}/gone.py": dict.fromkeys(TRUSTED, set())}))
    parts = held._replace(trusted={d: dict(zip(TRUSTED, map(set, lists))) for d, lists in split.items()})
    _hold_trusted(loads, parts)
    with pytest.raises(AssertionError, match=f"gains no load.*\n\\s*{party} loads {device}: {party}:1: "):
        _hold_trusted(loads, parts._replace(trusted=parts.trusted | {
            party: parts.trusted[party] | {"loads": parts.trusted[party]["loads"] - {device}}}))
    with pytest.raises(AssertionError, match="loads no other distribution.*\n.*package_above.py:1: python package"):
        _hold_distributions(tree.dirs, loads, held)
    for entry in ({"reason": "no owner"}, {"owner": "proofs", "date": "2026-11-18", "reason": "both"}, {"owner": "proofs", "date": "soon"}):
        with pytest.raises(AssertionError, match=r"its owner and.*at most one: \['verity/flock'\]"):
            _hold_distributions(tree.dirs, (), held._replace(distributions=own | {"verity/flock": entry}))
    with pytest.raises(AssertionError, match=r"hold no tracked file; delete them: \['verity/rho'\]"):
        _hold_distributions(tree.dirs, (), held._replace(distributions=own | {"verity/rho": {"owner": "training"}}))
