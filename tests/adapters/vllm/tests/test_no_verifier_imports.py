"""No library file of the adapter imports a protocol's `verifier` or `attester` module, but those `KNOWN` lists.

The adapter is the provers' side: what it needs of a protocol is a constant, a type or a pure helper the protocol also offers
from a module outside its trusted code. The scan reads every import with `ast`, function-local ones and literal `import_module`
names included, and fails on one `KNOWN` doesn't list and on an entry nothing imports any more, so the list only shrinks. An
entry is the imported name with `verity.protocols.` dropped, under the lane that owns the protocol. No library file
imports the network attester's package at all, which the adapter reaches through its scheme.
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
        # Needs, outside spot's verifier/ and device/: the docket's and partition's digests and template ids, the sampling
        # design's rate table, the second stage's row links, registration's `Declaration`, and the challenger's client
        # types (`CoinError`, `LiveOpening`, `LiveRecord`).
        "verity_vllm/pipeline/advice.py": ("registration.verifier.registration",),
        "verity_vllm/pipeline/boundary_rows.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.second_stage",
        ),
        "verity_vllm/pipeline/boundary_stop.py": ("proofs.spot.verifier.one_stage.docket",),
        "verity_vllm/pipeline/commit_coins.py": (
            "proofs.spot.verifier.challenger.challenger", "proofs.spot.verifier.challenger.challenger.CoinError",
            "proofs.spot.verifier.challenger.challenger.LiveOpening", "proofs.spot.verifier.challenger.challenger.live_reason",
            "proofs.spot.verifier.challenger.remote",
        ),
        "verity_vllm/pipeline/interior_links.py": ("proofs.spot.verifier.second_stage",),
        "verity_vllm/pipeline/interior_rows.py": ("proofs.spot.verifier.one_stage.docket",),
        "verity_vllm/pipeline/serve_day.py": ("registration.verifier.registration",),
        "verity_vllm/pipeline/serving_rows.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.service",
        ),
        "verity_vllm/program/predict/weights_tie.py": (
            "proofs.spot.verifier.one_stage.docket", "proofs.spot.verifier.one_stage.draw",
        ),
        "verity_vllm/protocols/pouw/h1_ranks.py": ("proofs.spot.verifier.challenger.challenger.LiveRecord",),
        "verity_vllm/protocols/pouw/h1_settings.py": ("proofs.spot.verifier.challenger.challenger.LiveRecord",),
    },
    "network-accounting": {},
    "memory-accounting": {
        # `protocols/pous.py` reads the schemes for its refusals, setup's salt draw, and the period's certificate and gaps
        # for its details; the store, its encode and decode and the period are pous's prover's (`weight_store`).
        "verity_vllm/protocols/pous.py": (
            "resource_accounting.memory.pous.verifier.period", "resource_accounting.memory.pous.verifier.schemes",
            "resource_accounting.memory.pous.verifier.schemes.p2", "resource_accounting.memory.pous.verifier.setup",
        ),
    },
}


def _names(tree: ast.AST) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names += [f"{node.module}.{a.name}" for a in node.names]
        elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
              and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names.append(node.args[0].value)
    return names


def _imported(tree: ast.AST) -> list[str]:
    return [n[len(PREFIX):] for n in _names(tree) if n.startswith(PREFIX) and set(n.split(".")) & set(TRUSTED)]


def _lane(name: str) -> str | None:
    return next((lane for p, lane in LANES.items() if name.startswith(p)), None)


def _library() -> list[tuple[str, ast.AST]]:
    return [(f.relative_to(ROOT).as_posix(), ast.parse(f.read_text())) for f in sorted(ROOT.rglob("*.py"))
            if not f.relative_to(ROOT).as_posix().startswith("tests/")
            and not any(part.startswith(".") for part in f.relative_to(ROOT).parts)]


def scan() -> dict[str, dict[str, set[str]]]:
    found: dict[str, dict[str, set[str]]] = {}
    for rel, tree in _library():
        for name in _imported(tree):
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


def test_no_import_of_the_network_attesters_package():
    network = PREFIX + "resource_accounting.network"
    found = [f"{rel}: {n}" for rel, tree in _library() for n in _names(tree) if n == network or n.startswith(network + ".")]
    assert not found, ("library files import the network attester's package: reach the network attester through its "
                       'scheme, `compose.schemes.get("network-attester")`:\n' + "\n".join(found))
