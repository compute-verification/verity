"""The evidence the Definitions' conformance texts and the row kernels' provenance records cite, read from `evidence.json` and
keyed by the capture or probe they identify: evidence-store art ids under the role each plays (``run_record``, the record a
run published; ``art``, its output), record digests, the harness commits that ran them and capture dates; and the GPU card
files (`card_files`) a binding reads a device's figures from."""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE: dict[str, dict] = json.loads(Path(__file__).with_name("evidence.json").read_text())

CARDS = Path("calibration/hardware/cards")


def card_files() -> Path:
    """The source tree's GPU card files (`verity.device`), the first `CARDS` above this file, wherever the Definitions sit in
    the tree; FileNotFoundError where none is (an install outside a source tree)."""
    here = Path(__file__).resolve()
    for root in here.parents:
        if (root / CARDS).is_dir():
            return root / CARDS
    raise FileNotFoundError(f"no {CARDS} above {here}: the Definitions read the card files of the source tree they sit in")
