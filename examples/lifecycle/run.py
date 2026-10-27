"""One monitoring period of the remote composite, simulated on CPU: each phase's public outputs and the period's verdict.
`python run.py [--cheat NAME]` runs it, the developer deviating by NAME (`lifecycle.CHEATS`) if given."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lifecycle
import remote


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cheat", choices=lifecycle.CHEATS)
    a = ap.parse_args(argv)
    o = remote.run(a.cheat)
    print(f"== remote ({lifecycle.products()['remote'][0]}), developer {a.cheat or 'honest'}: run {o['status']}")
    print("\n".join(lifecycle.summary(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
