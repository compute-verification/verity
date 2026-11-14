"""No library file of the adapter imports a protocol's `verifier` or `device` module, but those `KNOWN` lists.

The adapter is the provers' side: what it needs of a protocol is a constant, a type or a pure helper the protocol also offers
from a module outside its trusted code. The scan reads every import with `ast`, function-local ones and literal `import_module`
names included, and fails on one `KNOWN` doesn't list and on an entry nothing imports any more, so the list only shrinks. An
entry is the imported name with `verity.protocols.` dropped, under the lane that owns the protocol.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "verity.protocols."
TRUSTED = ("verifier", "device")
LANES = {
    "proofs.": "proofs",
    "accounting.registration.": "proofs",
    "accounting.pouw.": "compute-accounting",
    "accounting.certifier.": "network-accounting",
    "accounting.poub.": "network-accounting",
    "accounting.pous.": "memory-accounting",
}

KNOWN: dict[str, dict[str, tuple[str, ...]]] = {
    "proofs": {
        # Needs, outside spot's verifier/ and device/: the registered rows' record (`Registered`, its schemas, domains and
        # widths), the docket's and partition's digests and template ids, the draw design's parsers, the second stage's
        # row links, registration's `Declaration`, and the challenger's client types (`CoinError`, `LiveOpening`, `LiveRecord`).
        "verity_vllm/commit/registered_rows.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/pipeline/advice.py": ("accounting.registration.verifier.registration",),
        "verity_vllm/pipeline/boundary_rows.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.partition",
            "proofs.spot.verifier.second_stage",
        ),
        "verity_vllm/pipeline/boundary_stop.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.served",
        ),
        "verity_vllm/pipeline/commit_coins.py": (
            "proofs.spot.device.challenger", "proofs.spot.device.challenger.CoinError",
            "proofs.spot.device.challenger.LiveOpening", "proofs.spot.device.challenger.live_reason",
            "proofs.spot.device.remote",
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
        "verity_vllm/pipeline/serve_day.py": ("accounting.registration.verifier.registration",),
        "verity_vllm/pipeline/serve_tie.py": ("proofs.spot.verifier.one_stage.docket",),
        "verity_vllm/pipeline/serving_rows.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.partition",
            "proofs.spot.verifier.one_stage.registered", "proofs.spot.verifier.service",
        ),
        "verity_vllm/pipeline/window_law.py": ("proofs.spot.verifier.one_stage.draw",),
        "verity_vllm/program/predict/node_window_values.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/program/predict/weights_tie.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.draw",
            "proofs.spot.verifier.one_stage.registered",
        ),
        "verity_vllm/program/predict/window_rows.py": ("proofs.spot.verifier.one_stage.registered",),
        "verity_vllm/program/predict/window_weights.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.registered",
        ),
        "verity_vllm/protocols/pouw/h1_ranks.py": ("proofs.spot.device.challenger.LiveRecord",),
        "verity_vllm/protocols/pouw/h1_settings.py": ("proofs.spot.device.challenger.LiveRecord",),
    },
    "network-accounting": {
        # Needs: the certifier's advice format (`Advice`, `RequestAdvice`, `Row`, `Source`, `SCHEMA`), which moves out of
        # `verity/` to the adapter by the integrations rule, its release and io-tie headers and ids, the replay step's
        # reference; PoUB's receipt plan (`Plan`, `PER_LEAF`, salts), codec and memory-map format outside poub's verifier/.
        "verity_vllm/pipeline/advice.py": ("accounting.certifier.verifier.advice",),
        "verity_vllm/pipeline/advice_code/codec.py": ("accounting.certifier.verifier.advice",),
        "verity_vllm/pipeline/advice_code/derived.py": (
            "accounting.certifier.verifier.advice.RequestAdvice", "accounting.certifier.verifier.advice.Row",
            "accounting.certifier.verifier.advice.Source", "accounting.certifier.verifier.advice.StepAdvice",
        ),
        "verity_vllm/pipeline/advice_code/lengths.py": ("accounting.certifier.verifier.advice",),
        "verity_vllm/pipeline/advice_code/tick_step.py": ("accounting.certifier.verifier.replay_step",),
        "verity_vllm/pipeline/advice_code/window_code.py": ("accounting.certifier.verifier.arith_encode.CW",),
        "verity_vllm/pipeline/ids_front.py": (
            "accounting.certifier.verifier.io_tie", "accounting.certifier.verifier.release",
        ),
        "verity_vllm/pipeline/serve_admit.py": ("accounting.certifier.verifier.release.read_watermark",),
        "verity_vllm/pipeline/serve_commit.py": (
            "accounting.certifier.verifier.charge_chain", "accounting.certifier.verifier.compile_slice",
            "accounting.certifier.verifier.served_rows",
        ),
        "verity_vllm/pipeline/serve_head.py": ("accounting.certifier.verifier.release",),
        "verity_vllm/pipeline/serve_protocol.py": (
            "accounting.certifier.verifier.active", "accounting.certifier.verifier.io_tie",
            "accounting.certifier.verifier.release",
        ),
        "verity_vllm/pipeline/window_advice.py": (
            "accounting.certifier.verifier.advice", "accounting.certifier.verifier.advice.KINDS",
            "accounting.certifier.verifier.advice.MalformedAdvice", "accounting.certifier.verifier.advice.Output",
            "accounting.certifier.verifier.advice.Row", "accounting.certifier.verifier.advice.Source",
            "accounting.certifier.verifier.advice.StepAdvice", "accounting.certifier.verifier.advice.WIDTH_BITS",
        ),
        "verity_vllm/program/predict/hits.py": ("accounting.certifier.verifier.advice",),
        "verity_vllm/program/predict/scan_state.py": ("accounting.certifier.verifier.blocks",),
        "verity_vllm/program/predict/steps.py": ("accounting.certifier.verifier.advice",),
        "verity_vllm/protocols/remote_bandwidth/binder.py": ("accounting.poub.verifier.audit.Tiles",),
        "verity_vllm/protocols/remote_bandwidth/memory_map.py": (
            "accounting.poub.verifier.audit", "accounting.poub.verifier.codec",
            "accounting.poub.verifier.memory_map.Change", "accounting.poub.verifier.memory_map.Header",
            "accounting.poub.verifier.memory_map.MemoryMapError", "accounting.poub.verifier.memory_map.Replay",
            "accounting.poub.verifier.memory_map.Step", "accounting.poub.verifier.memory_map.Written",
            "accounting.poub.verifier.memory_map.dumps", "accounting.poub.verifier.memory_map.parse",
        ),
        "verity_vllm/protocols/remote_bandwidth/plan.py": (
            "accounting.poub.verifier.receipts", "accounting.poub.verifier.receipts.PER_LEAF",
            "accounting.poub.verifier.receipts.Plan",
        ),
        "verity_vllm/protocols/remote_bandwidth/registry.py": (
            "accounting.poub.verifier.receipts", "accounting.poub.verifier.receipts.PER_LEAF",
            "accounting.poub.verifier.receipts.Plan", "accounting.poub.verifier.store.store_salt",
        ),
        "verity_vllm/protocols/remote_bandwidth/remote.py": (
            "accounting.poub.verifier.audit", "accounting.poub.verifier.codec", "accounting.poub.verifier.receipts",
        ),
        "verity_vllm/protocols/remote_bandwidth/verify.py": (
            "accounting.poub.verifier.receipts", "accounting.poub.verifier.receipts.PER_LEAF",
        ),
    },
    "compute-accounting": {
        # Needs, outside pouw's verifier/: the matmul domain's error (`DomainError`).
        "verity_vllm/check/replay/pouw_circuit.py": ("accounting.pouw.verifier.protocol.DomainError",),
    },
    "memory-accounting": {
        # `protocols/pous.py` runs a whole period (setup, serve, audit, certify, the memory challenger's session): it
        # needs a prover entry in pous's prover/ that takes the layout and serves, the period's audit left to the verifier.
        "verity_vllm/protocols/pous.py": (
            "accounting.pous.device.challenger_session", "accounting.pous.device.memory_challenger",
            "accounting.pous.verifier.layout.TensorLayout", "accounting.pous.verifier.period",
            "accounting.pous.verifier.run", "accounting.pous.verifier.schemes", "accounting.pous.verifier.schemes.p2",
            "accounting.pous.verifier.schemes.p2.decode_segment", "accounting.pous.verifier.schemes.p2.expand_prefix",
            "accounting.pous.verifier.setup", "accounting.pous.verifier.verifier.DIGEST_BYTES",
            "accounting.pous.verifier.verifier.VerifierKey",
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
    assert not new, ("library files import a protocol's verifier or device module: import the name from the protocol's "
                     "module outside its verifier/ and device/ (ask its lane to offer it there), never by adding it to "
                     "KNOWN:\n" + "\n".join(new))


def test_known_lists_only_live_imports():
    found = scan()
    stale = [f"{lane}: {f}: {n}" for lane, files in KNOWN.items() for f, names in files.items()
             for n in names if n not in found.get(lane, {}).get(f, set())]
    assert not stale, "KNOWN entries nothing imports any more: delete them:\n" + "\n".join(stale)
