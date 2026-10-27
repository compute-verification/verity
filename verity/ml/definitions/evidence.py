"""The ids and dates the Definitions' conformance texts and the row kernels' provenance records cite (evidence-store runs,
records and art ids, capture dates, the ask that fixed the window slots), read from `evidence.json`, keyed by the capture or
probe they identify."""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE: dict[str, dict] = json.loads(Path(__file__).with_name("evidence.json").read_text())
