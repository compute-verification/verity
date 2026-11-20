"""machine.py toolkit [VERSION] | arch [NAME] | card-arch SHORT | gencode ARCH | sku [NAME] | session-node NAME SLOT: the kit's readers of the machine's record and the card's (`verity_bench.kit`),
printed for a shell script, or refused by name before the script takes a card.

    toolkit VERSION  `kit.toolkit`: $CUDA_HOME where it is set and not another version, else the directory the run's
                     machine's `[properties]` list at that version (the copy `research run --on` puts in
                     $RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml).
                     Exit 3 where the properties are read and list no toolkit at that version.
    toolkit          `kit.machine_toolkit`: the toolkit the run's machine's `[properties]` list first.
    arch NAME        `kit.card_arch`: the `arch` of the card record whose `driver` holds NAME, the name nvidia-smi reports.
    arch             `kit.machine_arch`: the `arch` of the run's machine's card, with no card needed.
    card-arch SHORT  The `arch` of the one GPU card file whose `short` is SHORT (`verity.device.gpu_card_called`), for a
                     build whose caller names its card.
    gencode ARCH     `kit.arch_gencode`: nvcc's -gencode value for the target ARCH (sm_90a), from its architecture file.
    sku NAME         `kit.card_sku`: the `sku` of the card record whose `driver` holds NAME.
    sku              `kit.run_card_sku`: the `sku` of the card the run is on, `kit.card_sku` of the name
                     nvidia-smi reports for the lease's GPUs, else the host's (vLLM's CARD_SKU).
    session-node NAME SLOT  `kit.session_node`: the node a PoUB session on that card registers (NAME `cpu`: no card).

A reader of the run's machine refuses where its record is missing, naming the file it looked for
($RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml): a build that sets RESEARCH_RUN_DIR to a private run directory must
carry the record there.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOT_LISTED = 3


def missing_record(refusal: str) -> str | None:
    """What to add to ``refusal`` where it is the kit's of the run's machine's record, whose copy is not in the run
    directory: the file looked for, and that a private run directory must carry it; None for any other refusal."""
    from verity_bench import settings

    if not settings.is_set("RESEARCH_MACHINE"):
        return None
    path = Path(settings.RESEARCH_RUN_DIR) / "inputs" / f"{settings.RESEARCH_MACHINE}.toml"
    if path.is_file() or str(path) not in refusal:
        return None
    return (f"machine.py: looked for the run's machine record at $RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml, here {path}, "
            "and it is not there. `research run --on` puts it in the run directory it makes; a build that sets "
            "RESEARCH_RUN_DIR to a private run directory must carry the record: copy the launching run's "
            f"inputs/{settings.RESEARCH_MACHINE}.toml into that directory's inputs/ before it calls machine.py")


def main(argv: list[str]) -> int:
    try:
        return read(argv)
    except SystemExit as e:
        if isinstance(e.code, str) and (hint := missing_record(e.code)):
            raise SystemExit(f"{e.code}\n{hint}") from None
        raise


def read(argv: list[str]) -> int:
    from verity_bench import kit

    match argv:
        case ["toolkit", version]:
            try:
                print(kit.toolkit(version))
            except kit.NotListed as e:
                print(f"machine.py: {e}", file=sys.stderr)
                return NOT_LISTED
        case ["toolkit"]:
            print(kit.machine_toolkit())
        case ["arch", name]:
            print(kit.card_arch(name))
        case ["arch"]:
            print(kit.machine_arch())
        case ["card-arch", short]:
            from verity.device import gpu_card_called

            try:
                print(gpu_card_called(short, directory=kit.cards_directory())["arch"])
            except KeyError as e:
                raise SystemExit(e.args[0]) from None
        case ["gencode", arch]:
            print(kit.arch_gencode(arch))
        case ["sku", name]:
            print(kit.card_sku(name))
        case ["session-node", name, slot]:
            print(kit.session_node(None if name == "cpu" else name, slot))
        case ["sku"]:
            print(kit.run_card_sku())
        case _:
            raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / "benchmarks" / "harness"), str(ROOT / "tools" / "config"), str(ROOT)]
    sys.exit(main(sys.argv[1:]))
