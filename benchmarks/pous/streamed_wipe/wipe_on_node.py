"""`python benchmarks/pous/streamed_wipe/wipe_on_node.py --period DIR --out DIR [--node NAME] [--expect VERDICT]`: the wipe
on this node (`wipe_period.main`), its challenger's side started where `period.json` puts it (`stream.Side.of`)."""

from __future__ import annotations

import sys

from verity.runtime.stream import Side
from verity.protocols.accounting.pous.verifier import wipe_period

if __name__ == "__main__":
    sys.exit(wipe_period.main(side_of=Side.of))
