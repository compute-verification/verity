"""machine.py toolkit VERSION | arch NAME: what kernels' runners read from their machine's record and their card's, printed
for the shell, or refused by name before the runner takes a card.

    toolkit VERSION  the directory of the CUDA toolkit VERSION: $CUDA where it is set, else the one the run's machine's
                     `[properties]` list at that version under `toolkit` (`verity_bench.kit.machine_properties()`: the
                     copy `research run --on` puts in $RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml).
                     Exit 3 where the properties are read and list no toolkit at that version.
    arch NAME        the `arch` of the card record whose `driver` holds NAME, the name nvidia-smi reports (`kit.card_short`).
"""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOT_LISTED = 3


class NotListed(Exception):
    """The machine's record is read and lists no toolkit at the version asked for."""


def toolkit(version: str, env: Mapping[str, str]) -> str:
    if env.get("CUDA"):
        return env["CUDA"]
    from verity_bench import kit, settings

    try:
        found = kit.machine_properties()["toolkit"]["value"]
    except SystemExit as e:
        raise SystemExit(f"no CUDA toolkit {version}: set CUDA to its directory, or run on a machine `research run --on` "
                         f"names: {e}") from None
    name = settings.RESEARCH_MACHINE
    if found == "unread":
        raise SystemExit(f"machine {name!r}: its [properties] have toolkit unread: read it there, or set CUDA")
    listed = [] if found == "none" else [found, *found.get("others", [])]
    at = [t["directory"] for t in listed if t["version"] == version]
    if not at:
        raise NotListed(f"machine {name!r}: its [properties] list no CUDA toolkit {version} under toolkit (they list "
                        f"{', '.join(t['version'] for t in listed) or 'none'}): set CUDA")
    return at[0]


def arch(name: str) -> str:
    from verity.device import load_card, skus
    from verity_bench import kit
    from verity_bench.overhead import CARD_FILES

    kit.card_short(name, directory=CARD_FILES)
    [card] = [c for c in (load_card(s, directory=CARD_FILES) for s in skus("gpu", directory=CARD_FILES))
              if name in {d["value"] for d in c.get("driver", ())}]
    return card["arch"]


def main(argv: list[str]) -> int:
    match argv:
        case ["toolkit", version]:
            try:
                print(toolkit(version, os.environ))
            except NotListed as e:
                print(f"machine.py: {e}", file=sys.stderr)
                return NOT_LISTED
        case ["arch", name]:
            print(arch(name))
        case _:
            raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT / "benchmarks" / "harness"), str(ROOT / "tools" / "config"), str(ROOT)]
    sys.exit(main(sys.argv[1:]))
