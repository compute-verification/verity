"""`python benchmarks/pous/streamed_wipe/wipe_on_node.py --period DIR --out DIR [--node NAME] [--expect VERDICT]`: the wipe
on this node (`wipe_run.main`), its challenger's side started where `period.json` puts it (`stream.Side.of`)."""

from __future__ import annotations

import sys

from verity.protocols.accounting.pous.prover import wipe_run
from verity.runtime.stream import Side

if __name__ == "__main__":
    sys.exit(wipe_run.main(side_of=Side.of))
