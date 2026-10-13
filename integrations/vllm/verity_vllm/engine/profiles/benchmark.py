"""Model and device facts for the shared overhead-benchmark presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PRESET_FACTS: dict[str, dict[str, Any]] = json.loads(
    Path(__file__).with_suffix(".json").read_text()
)
