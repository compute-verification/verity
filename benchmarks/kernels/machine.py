"""machine.py toolkit VERSION | arch [NAME] | sku NAME | session-node NAME SLOT: the kit's readers of the machine's record and the card's (`verity_bench.kit`),
printed for a shell script, or refused by name before the script takes a card.

    toolkit VERSION  `kit.toolkit`: $CUDA_HOME where it is set and not another version, else the directory the run's
                     machine's `[properties]` list at that version (the copy `research run --on` puts in
                     $RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml).
                     Exit 3 where the properties are read and list no toolkit at that version.
    arch NAME        `kit.card_arch`: the `arch` of the card record whose `driver` holds NAME, the name nvidia-smi reports.
    arch             `kit.machine_arch`: the `arch` of the run's machine's card, with no card needed.
    sku NAME         `kit.card_sku`: the `sku` of the card record whose `driver` holds NAME.
    session-node NAME SLOT  `kit.session_node`: the node a PoUB session on that card registers (NAME `cpu`: no card).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOT_LISTED = 3


def main(argv: list[str]) -> int:
    from verity_bench import kit

    match argv:
        case ["toolkit", version]:
            try:
                print(kit.toolkit(version))
            except kit.NotListed as e:
                print(f"machine.py: {e}", file=sys.stderr)
                return NOT_LISTED
        case ["arch", name]:
            print(kit.card_arch(name))
        case ["arch"]:
            print(kit.machine_arch())
        case ["sku", name]:
            print(kit.card_sku(name))
        case ["session-node", name, slot]:
            print(kit.session_node(None if name == "cpu" else name, slot))
        case _:
            raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / "benchmarks" / "harness"), str(ROOT / "tools" / "config"), str(ROOT)]
    sys.exit(main(sys.argv[1:]))
