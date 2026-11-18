"""machine.py toolkit VERSION | arch NAME: what kernels' runners read from their machine's record and their card's, printed
for the shell, or refused by name before the runner takes a card.

    toolkit VERSION  the directory of the CUDA toolkit VERSION: $CUDA where it is set, else the one machine $MACHINE's
                     record lists at that version under `toolkit` (`verity_bench.kit.machine_properties`), the record
                     read from $MACHINES_D, by default $RESEARCH_RUN_DIR/inputs, where `research run --send` puts it.
                     Exit 3 where the record is read and lists no toolkit at that version.
    arch NAME        the `arch` of the card record whose `driver` holds NAME, the name nvidia-smi reports (`kit.card_short`).

    research run --on MACHINE --send <notes>/machines.d/MACHINE.toml --env MACHINE=MACHINE ... -- bash RUNNER
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
    name = env.get("MACHINE")
    records = env.get("MACHINES_D") or (f"{env['RESEARCH_RUN_DIR']}/inputs" if env.get("RESEARCH_RUN_DIR") else "")
    if not name or not records:
        raise SystemExit(f"no CUDA toolkit {version}: set CUDA to its directory, or MACHINE to the machine's name and "
                         "MACHINES_D to the directory of its record (default $RESEARCH_RUN_DIR/inputs, `research run --send`)")
    from verity_bench import kit

    found = kit.machine_properties(name, directory=Path(records))["toolkit"]["value"]
    if found == "unread":
        raise SystemExit(f"machine {name!r}: its record ({records}/{name}.toml) has toolkit unread: read it there, or set CUDA")
    listed = [] if found == "none" else [found, *found.get("others", [])]
    at = [t["directory"] for t in listed if t["version"] == version]
    if not at:
        raise NotListed(f"machine {name!r}: its record ({records}/{name}.toml) lists no CUDA toolkit {version} under "
                        f"toolkit (it lists {', '.join(t['version'] for t in listed) or 'none'}): set CUDA")
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
