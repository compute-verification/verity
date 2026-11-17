from __future__ import annotations

import json
from functools import cache
from pathlib import Path

#: each registered metric's `version` and `unit` by its id, which tests/test_metrics_registry.py holds equal to the rows
#: of the tree's metrics.toml files; refreshed by `uv run python tests/test_metrics_registry.py --pin`
PINS = Path(__file__).with_name("metric_versions.json")


@cache
def pins() -> dict[str, dict]:
    return json.loads(PINS.read_text(encoding="utf-8"))


def stamp(name: str) -> dict[str, int]:
    pin = pins().get(name)
    return {} if pin is None else {"version": pin["version"]}
