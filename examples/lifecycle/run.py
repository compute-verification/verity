"""One monitoring period of each composite, onsite and remote, simulated on CPU: each phase's public outputs and the period's
verdict. `python run.py [--cheat NAME]` runs both, the developer deviating by NAME (`lifecycle.CHEATS`) if given."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lifecycle
import onsite
import remote


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cheat", choices=lifecycle.CHEATS + lifecycle.ONSITE_CHEATS)
    ap.add_argument("--only", choices=("onsite", "remote"))
    a = ap.parse_args(argv)
    for name, mod in (("onsite", onsite), ("remote", remote)):
        if a.only and a.only != name:
            continue
        o = mod.run(a.cheat)
        print(f"== {name} ({lifecycle.products()[name][0]}), developer {a.cheat or 'honest'}: run {o['status']}")
        print("\n".join(lifecycle.summary(o)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
