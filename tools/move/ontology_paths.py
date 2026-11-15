"""Writes `ontology_paths_map.toml`, the layout map of the protocols' ontology, in one of two ways for the network
protocol's directory.

    python3 tools/move/ontology_paths.py --way B     # resource_accounting/network/ is the protocol (the default)
    python3 tools/move/ontology_paths.py --way A     # resource_accounting/network/attester/ is the protocol

Way A: `resource_accounting/network/attester/` holds `attester/`, `prover/` and `verifier/`. Way B:
`resource_accounting/network/` holds them. Every other row is the same in both ways. [modules] is what `layout.py
--modules` derives from the rows, so the map passes `layout.py`'s own check of it."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

MOVE = Path(__file__).resolve().parent
P = "verity/protocols"
NETWORK = {"A": f"{P}/resource_accounting/network/attester", "B": f"{P}/resource_accounting/network"}

ROOTS = {"verity": "verity", "verity_difftest": "tools/difftest/python/verity_difftest",
         "verity_flock": f"{P}/proofs/flock/prover/python/verity_flock",
         "verity_numerical": "benchmarks/numerical/python/verity_numerical", "verity_vllm": "adapters/vllm/verity_vllm",
         "research": "tools/research/src/research", "circuit_check": "tools/verity/circuit_check/src/circuit_check",
         "cluster": "infra/cluster/src/cluster", "verity_cli": "tools/verity/cli/verity_cli",
         "verity_config": "tools/config/verity_config",
         "verity_machine_model": "benchmarks/machine_model/verity_machine_model",
         "datacenter": "benchmarks/datacenter/datacenter", "verity_torch": "adapters/torch/verity_torch"}

EXCLUDE = ["fixtures/**", "**/fixtures/**", "**/*vectors*.json", "**/*vectors*.txt.gz", "**/CHANGELOG.md", "*.patch",
           "**/TRUSTED.sha256", "**/lake-manifest.json.lock", "tools/move/**", "benchmarks/numerical/tests/bench/data/**",
           "**/price_twins.json", "verity/circuits/silicon/captures/**", "verity/circuits/silicon/hawkeye/**",
           "tools/verity/check/tests/test_suites.py", "tools/verity/check/tests/test_move_check.py",
           "tools/research/src/research/store/moves.json", "benchmarks/network_e2e/tests/data/**",
           "adapters/vllm/manifests/**", "adapters/vllm/workloads/**", "adapters/vllm/data/**",
           "adapters/vllm/verity_vllm/pins/**", "**/*.lean", "**/*.cu", "**/*.cuh", "**/*.cpp", "**/*.h", "**/*.c"]
INCLUDE = ["**/fixtures/README.md", "**/fixtures/**/README.md", "**/lake-manifest.json",
           "adapters/vllm/manifests/definitions.json", "adapters/vllm/manifests/README.md"]


SPLIT = {
    "tests/tree.toml: verity/protocols/accounting":
        "the layer's own row; registration, which leaves it, has a row of its own",
    "benchmarks/demo_minutes/tests/test_no_pin_literal.py: verity/protocols/accounting":
        "the scan follows `resource_accounting/`; `verity/protocols/registration` joins its list by hand",
    "verity/protocols/accounting/certifier/verifier/README.md: verity/protocols/proofs/spot":
        "the link names SPOT whole, which stays; only its `device/` moves, inside it",
}


def rows(way: str) -> dict[str, str]:
    net = NETWORK[way]
    return {
        f"{P}/accounting/": f"{P}/resource_accounting/",
        f"{P}/accounting/pouw/": f"{P}/resource_accounting/compute/pouw/",
        f"{P}/accounting/pous/": f"{P}/resource_accounting/memory/pous/",
        f"{P}/accounting/pous/device/": f"{P}/resource_accounting/memory/pous/attester/",
        f"{P}/accounting/poub/": f"{P}/resource_accounting/memory/poub/",
        f"{P}/accounting/poub/device/": f"{P}/resource_accounting/memory/poub/attester/",
        f"{P}/accounting/certifier/": f"{net}/",
        f"{P}/accounting/certifier/device/": f"{net}/attester/",
        f"{P}/accounting/registration/": f"{P}/registration/",
        f"{P}/policy/": f"{P}/workload_classification/",
        f"{P}/policy/inference_only/": f"{P}/workload_classification/training_detection/inference_only/",
        f"{P}/proofs/spot/device/": f"{P}/proofs/spot/verifier/challenger/",
        f"{P}/compose/device/": f"{P}/compose/verifier/",
    }


def derived(paths: dict[str, str]) -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("layout_for_ontology", MOVE / "layout.py")
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    return L.derived_modules(L.Layout({"paths": paths, "roots": {"old": ROOTS, "new": ROOTS}}))


def toml(way: str) -> str:
    paths = rows(way)
    q = json.dumps
    inline = "{ " + ", ".join(f"{k} = {q(v)}" for k, v in ROOTS.items()) + " }"
    out = [f"# The protocols' ontology, way {way}: written by `python3 tools/move/ontology_paths.py --way {way}`, run by",
           "# `python3 tools/move/layout.py --map tools/move/ontology_paths_map.toml`. Edit the script, not this file.",
           "#",
           "# - `policy/` becomes `workload_classification/`, with `inference_only` under `training_detection/`;",
           "# - `accounting/` becomes `resource_accounting/`, grouped by resource: `compute/pouw`, `memory/pous`,",
           f"#   `memory/poub`, and the network protocol at `{NETWORK[way]}/`;",
           "# - an attester's `device/` (the network certifier's, PoUS's and PoUB's) becomes `attester/`;",
           "# - the challenger's two `device/` directories go under their protocol's `verifier/`: SPOT's to `challenger/`,",
           "#   compose's `live_challenger/` straight under `verifier/`;",
           "# - `registration/` leaves `accounting/` and sits beside the layers.",
           "# Lean module names follow their package's root, so no Lean module, declaration or lock record changes; a Lake",
           "# path between packages does. Lean sources are not rewritten (a comment's path rebuilds the module). The format",
           "# is `layout_map.toml`'s.",
           "#",
           "# Names that follow a path, which the mover rewrites: workspace members and suite inputs, `uv.lock` (by",
           "# `uv lock`), Lake requirement paths (`lakefile.toml`, `lake-manifest.json`) and the check's pins of those",
           "# manifests, the listing's paths and distributions (`tests/import_order.toml`). No distribution or Lake package",
           "# name follows a path (`pous-check`, `memory-challenger-lean`, `certifier_program`, `core_memory_challenger`), so",
           "# none needs a line. The hand lines the move's change adds, which name a directory by its last part:",
           "# `tests/test_boundaries.py`'s `_trusted` and `tests/test_lowrisk_excludes.py`'s `TRUSTED_DIRS` gain `attester`;",
           "# `tests/tree.toml`'s `NAME/device/` becomes `NAME/attester/`, with a row for each resource group;",
           "# `benchmarks/demo_minutes/tests/test_no_pin_literal.py`'s `SCANNED` gains `verity/protocols/registration`.",
           "# Then two regenerations: `uv run tools/check/import_order.py` (the listing's lines re-sort) and",
           "# `tools/pous_grader/TRUSTED.sha256` by the command `tests/test_pins.py` prints (it pins a rewritten lakefile).",
           "", "[roots]", f"old = {inline}", f"new = {inline}", "", "[paths]"]
    out += [f"{q(k)} = {q(v)}" for k, v in paths.items()]
    retired = [f"{P}/accounting/", f"{P}/policy/"]
    groups = [f"{P}/resource_accounting/compute", f"{P}/resource_accounting/memory",
              f"{P}/workload_classification/training_detection"] + ([f"{P}/resource_accounting/network"] if way == "A" else [])
    out += ["", "[retired]", "# the directories this map empties", f"paths = {q(retired)}", "",
            "[create]", "# the new groups' packages, empty as every group's is"]
    out += [f"{q(g + '/__init__.py')} = \"\"" for g in groups]
    out += ["", "[rewrite]",
            "# Never rewritten: records of what was (the earlier moves and their maps, changelogs, research's recorded",
            "# moves), files whose bytes a digest pins, every Lean file and every kernel source.",
            "exclude = [" + ",\n           ".join(q(e) for e in EXCLUDE) + "]",
            "include = [" + ",\n           ".join(q(e) for e in INCLUDE) + "]",
            "# the auditor devices' install directories, each a checkout of the tree (`pods/nebius/challenger/`,",
            "# `pods/nebius/verifier/`): their units name the challenger's service by its path there",
            f"root_prefixes = {q(['/opt/vy-challenger/current/', '/opt/vy-verifier/current/'])}",
            "", "[split]",
            "# `accounting/` splits: registration leaves it for `verity/protocols/registration/`, and a reference to the",
            "# whole directory follows the larger part, `resource_accounting/`. SPOT's directory has no row of its own, so",
            "# a reference to it reads as split though only its `device/` moves, into its own `verifier/`."]
    out += [f"{q(k)} = {q(v)}" for k, v in SPLIT.items()]
    out += ["", "[modules]", "# what `layout.py --modules` derives from [paths] and [roots]"]
    out += [f"{q(k)} = {q(v)}" for k, v in derived(paths).items()]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--way", choices=sorted(NETWORK), default="B")
    ap.add_argument("--out", type=Path, default=MOVE / "ontology_paths_map.toml")
    a = ap.parse_args()
    a.out.write_text(toml(a.way))
    print(f"wrote {a.out} (way {a.way})")


if __name__ == "__main__":
    main()
