"""The evidence the Definitions' conformance texts and the row kernels' provenance records cite, read from `evidence.json` and
keyed by the capture or probe they identify: evidence-store art ids under the role each plays (``run_record``, the record a
run published; ``art``, its output), record digests, the harness commits that ran them and capture dates."""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE: dict[str, dict] = json.loads(Path(__file__).with_name("evidence.json").read_text())
