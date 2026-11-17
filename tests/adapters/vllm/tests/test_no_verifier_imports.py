"""No library file of the adapter imports a protocol's `verifier` or `attester` module, but those `KNOWN` lists.

The adapter is the provers' side: what it needs of a protocol is a constant, a type or a pure helper the protocol also offers
from a module outside its trusted code. The scan reads every import with `ast`, function-local ones and literal `import_module`
names included, and fails on one `KNOWN` doesn't list and on an entry nothing imports any more, so the list only shrinks. An
entry is the imported name with `verity.protocols.` dropped, under the lane that owns the protocol.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[4] / "adapters" / "vllm")
PREFIX = "verity.protocols."
TRUSTED = ("verifier", "attester")
LANES = {
    "proofs.": "proofs",
    "registration.": "proofs",
    "resource_accounting.compute.": "compute-accounting",
    "resource_accounting.network.": "network-accounting",
    "resource_accounting.memory.poub.": "network-accounting",
    "resource_accounting.memory.pous.": "memory-accounting",
}

KNOWN: dict[str, dict[str, tuple[str, ...]]] = {
    "proofs": {
        # Needs, outside spot's verifier/ and device/: the registered rows' record (`Registered`, its schemas, domains and
        # widths), the docket's and partition's digests and template ids, the sampling design's parsers, the second stage's
        # row links, registration's `Declaration`, and the challenger's client types (`CoinError`, `LiveOpening`, `LiveRecord`).
        "verity_vllm/commit/registered_rows.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/compiler/replay_chain.py": (
            "proofs.spot.verifier.one_stage.produced", "proofs.spot.verifier.one_stage.registered",
        ),
        "verity_vllm/pipeline/advice.py": ("registration.verifier.registration",),
        "verity_vllm/pipeline/boundary_rows.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.partition",
            "proofs.spot.verifier.second_stage",
        ),
        "verity_vllm/pipeline/boundary_stop.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.served",
        ),
        "verity_vllm/pipeline/commit_coins.py": (
            "proofs.spot.verifier.challenger.challenger", "proofs.spot.verifier.challenger.challenger.CoinError",
            "proofs.spot.verifier.challenger.challenger.LiveOpening", "proofs.spot.verifier.challenger.challenger.live_reason",
            "proofs.spot.verifier.challenger.remote",
        ),
        "verity_vllm/pipeline/edge.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/pipeline/expert_rows.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/pipeline/expert_tables.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/pipeline/interior_links.py": (
            "proofs.spot.verifier.one_stage.registered", "proofs.spot.verifier.second_stage",
        ),
        "verity_vllm/pipeline/interior_rows.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.partition",
        ),
        "verity_vllm/pipeline/registration_log.py": ("proofs.spot.verifier.one_stage.docket",),
        "verity_vllm/pipeline/serve_day.py": ("registration.verifier.registration",),
        "verity_vllm/pipeline/serve_tie.py": ("proofs.spot.verifier.one_stage.docket",),
        "verity_vllm/pipeline/serving_rows.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.partition",
            "proofs.spot.verifier.one_stage.registered", "proofs.spot.verifier.service",
        ),
        "verity_vllm/pipeline/window_sampling_design.py": ("proofs.spot.verifier.one_stage.draw",),
        "verity_vllm/program/predict/node_window_values.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/program/predict/weights_tie.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.draw",
            "proofs.spot.verifier.one_stage.registered",
        ),
        "verity_vllm/program/predict/window_rows.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/program/predict/window_weights.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.registered",
        ),
        # The compiler's served rows (`verity_vllm/compiler/served_rows.py`): the Lean verifier's `--registered` port form
        # (`produced.port`, `registered.READS_FROM`), until SPOT's workload side offers the registered-port format.
        "verity_vllm/compiler/served_rows.py": (
            "proofs.spot.verifier.one_stage.produced", "proofs.spot.verifier.one_stage.registered",
        ),
        "verity_vllm/protocols/pouw/h1_ranks.py": ("proofs.spot.verifier.challenger.challenger.LiveRecord",),
        "verity_vllm/protocols/pouw/h1_settings.py": ("proofs.spot.verifier.challenger.challenger.LiveRecord",),
    },
    "network-accounting": {
        # Needs: network certification's advice format (`Advice`, `RequestAdvice`, `Row`, `Source`, `SCHEMA`), which moves out of
        # `verity/` to the adapter by the integrations rule, its release and io-tie headers and ids, the replay step's
        # reference; PoUB's receipt plan (`Plan`, `PER_LEAF`, salts), codec and memory-map format outside poub's verifier/.
        "verity_vllm/pipeline/advice.py": ("resource_accounting.network.verifier.advice",),
        "verity_vllm/pipeline/advice_code/codec.py": ("resource_accounting.network.verifier.advice",),
        "verity_vllm/pipeline/advice_code/derived.py": (
            "resource_accounting.network.verifier.advice.RequestAdvice", "resource_accounting.network.verifier.advice.Row",
            "resource_accounting.network.verifier.advice.Source", "resource_accounting.network.verifier.advice.StepAdvice",
        ),
        "verity_vllm/pipeline/advice_code/lengths.py": ("resource_accounting.network.verifier.advice",),
        "verity_vllm/pipeline/advice_code/window_code.py": ("resource_accounting.network.verifier.arith_encode.CW",),
        "verity_vllm/pipeline/ids_front.py": ("resource_accounting.network.verifier.release",),
        "verity_vllm/pipeline/serve_admit.py": ("resource_accounting.network.verifier.release.read_watermark",),
        "verity_vllm/pipeline/serve_commit.py": ("resource_accounting.network.verifier.charge_chain",),
        "verity_vllm/pipeline/serve_head.py": ("resource_accounting.network.verifier.release",),
        "verity_vllm/pipeline/serve_protocol.py": ("resource_accounting.network.verifier.release",),
        "verity_vllm/pipeline/window_advice.py": (
            "resource_accounting.network.verifier.advice", "resource_accounting.network.verifier.advice.KINDS",
            "resource_accounting.network.verifier.advice.MalformedAdvice", "resource_accounting.network.verifier.advice.Output",
            "resource_accounting.network.verifier.advice.Row", "resource_accounting.network.verifier.advice.Source",
            "resource_accounting.network.verifier.advice.StepAdvice", "resource_accounting.network.verifier.advice.WIDTH_BITS",
        ),
        "verity_vllm/program/predict/hits.py": ("resource_accounting.network.verifier.advice",),
        "verity_vllm/program/predict/scan_state.py": ("resource_accounting.network.verifier.blocks",),
        "verity_vllm/program/predict/steps.py": ("resource_accounting.network.verifier.advice",),
        # The compiler's (`verity_vllm/compiler/`): the replay step's and chain's circuits and row forms, the charge's
        # book and code (`charge_chain`, `arith_encode`), the IO tie's routing, the advice's form (its reader, constants
        # and error) and head_check's circuit helpers (`_Consts`, `_coll`, `ands`), until network certification's workload side
        # offers them.
        "verity_vllm/compiler/advice_check.py": (
            "resource_accounting.network.verifier.advice.FINISHES", "resource_accounting.network.verifier.advice.MalformedAdvice",
            "resource_accounting.network.verifier.advice.VERSION", "resource_accounting.network.verifier.advice.WIDTH_BITS",
            "resource_accounting.network.verifier.advice._Reader", "resource_accounting.network.verifier.head_check._Consts",
            "resource_accounting.network.verifier.head_check._coll", "resource_accounting.network.verifier.head_check.ands",
        ),
        "verity_vllm/compiler/cache_step.py": (
            "resource_accounting.network.verifier.head_check._Consts", "resource_accounting.network.verifier.head_check._coll",
            "resource_accounting.network.verifier.head_check.ands",
        ),
        "verity_vllm/compiler/cache_step_v3.py": (
            "resource_accounting.network.verifier.head_check._Consts", "resource_accounting.network.verifier.head_check._coll",
            "resource_accounting.network.verifier.head_check.ands",
        ),
        "verity_vllm/compiler/compile_slice.py": (
            "resource_accounting.network.verifier.arith_encode", "resource_accounting.network.verifier.charge_chain",
        ),
        "verity_vllm/compiler/compile_slice_v4.py": (
            "resource_accounting.network.verifier.head_check._Consts", "resource_accounting.network.verifier.head_check._coll",
            "resource_accounting.network.verifier.head_check.ands",
        ),
        "verity_vllm/compiler/compile_slice_v5.py": (
            "resource_accounting.network.verifier.head_check._Consts", "resource_accounting.network.verifier.head_check._coll",
            "resource_accounting.network.verifier.head_check.ands",
        ),
        "verity_vllm/compiler/compile_slice_v6.py": (
            "resource_accounting.network.verifier.head_check._Consts", "resource_accounting.network.verifier.head_check._coll",
            "resource_accounting.network.verifier.head_check.ands",
        ),
        "verity_vllm/compiler/compile_slice_v7.py": ("resource_accounting.network.verifier.charge_chain",),
        "verity_vllm/compiler/served_statics.py": ("resource_accounting.network.verifier.head_check",),
        "verity_vllm/compiler/served_tie.py": (
            "resource_accounting.network.verifier.head_check", "resource_accounting.network.verifier.io_tie",
        ),
        "verity_vllm/protocols/poub/binder.py": ("resource_accounting.memory.poub.verifier.audit.Tiles",),
        "verity_vllm/protocols/poub/remote.py": (
            "resource_accounting.memory.poub.verifier.audit", "resource_accounting.memory.poub.verifier.codec",
        ),
    },
    "compute-accounting": {
        # Needs, outside pouw's verifier/: the matmul domain's error (`DomainError`).
        "verity_vllm/check/replay/pouw_circuit.py": ("resource_accounting.compute.pouw.verifier.protocol.DomainError",),
    },
    "memory-accounting": {
        # `protocols/pous.py` reads the schemes for its refusals, setup's salt draw, and the period's certificate and gaps
        # for its details; the store, its encode and decode and the period are pous's prover's (`weight_store`).
        "verity_vllm/protocols/pous.py": (
            "resource_accounting.memory.pous.verifier.period", "resource_accounting.memory.pous.verifier.schemes",
            "resource_accounting.memory.pous.verifier.schemes.p2", "resource_accounting.memory.pous.verifier.setup",
        ),
    },
}


def _imported(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names += [f"{node.module}.{a.name}" for a in node.names]
        elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
              and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names.append(node.args[0].value)
    return [n[len(PREFIX):] for n in names if n.startswith(PREFIX) and set(n.split(".")) & set(TRUSTED)]


def _lane(name: str) -> str | None:
    return next((lane for p, lane in LANES.items() if name.startswith(p)), None)


def scan() -> dict[str, dict[str, set[str]]]:
    found: dict[str, dict[str, set[str]]] = {}
    for f in sorted(ROOT.rglob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        if rel.startswith("tests/") or any(part.startswith(".") for part in f.relative_to(ROOT).parts):
            continue
        for name in _imported(ast.parse(f.read_text())):
            found.setdefault(_lane(name) or "(no lane)", {}).setdefault(rel, set()).add(name)
    return found


def test_no_new_verifier_or_device_import():
    new = [f"{lane}: {f}: {n}" for lane, files in scan().items() for f, names in files.items()
           for n in sorted(names - set(KNOWN.get(lane, {}).get(f, ())))]
    assert not new, ("library files import a protocol's verifier or attester module: import the name from the protocol's "
                     "module outside its verifier/ and attester/ (ask its lane to offer it there), never by adding it to "
                     "KNOWN:\n" + "\n".join(new))


def test_known_lists_only_live_imports():
    found = scan()
    stale = [f"{lane}: {f}: {n}" for lane, files in KNOWN.items() for f, names in files.items()
             for n in names if n not in found.get(lane, {}).get(f, set())]
    assert not stale, "KNOWN entries nothing imports any more: delete them:\n" + "\n".join(stale)
