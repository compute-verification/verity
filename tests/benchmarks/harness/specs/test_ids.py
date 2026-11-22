"""A spec's or sweep's `id` is its file's name."""
from __future__ import annotations

import tomllib
from pathlib import Path

SPECS = (Path(__file__).resolve().parents[4] / "benchmarks" / "harness" / "specs")
SCHEMAS = ("verity-bench/vllm-spec/v0", "verity-bench/vllm-sweep/v0", "verity-bench/steady-spec/v0")


def test_every_spec_and_sweep_is_named_after_its_id() -> None:
    for path in SPECS.glob("*.toml"):
        doc = tomllib.loads(path.read_text())
        if doc.get("schema") in SCHEMAS:
            assert doc["id"] == path.stem, f"{path.name}: its id is {doc['id']!r}; name the file after its id"
