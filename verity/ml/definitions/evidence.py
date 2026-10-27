"""The ids and dates the Definitions' conformance texts, the row kernels' provenance records and the lifted Programs' labels
and refusals cite (evidence-store runs, records and art ids, capture dates, who ran a quarantine's difftests, the rulings and
findings the lifted Programs follow, the ask that fixed the window slots), read from `evidence.json`, keyed by the capture,
probe or rule they identify."""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE: dict[str, dict] = json.loads(Path(__file__).with_name("evidence.json").read_text())
