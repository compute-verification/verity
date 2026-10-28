"""The README lists every spec and sweep here, and names none that isn't."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

SPECS = Path(__file__).resolve().parents[1]
SCHEMAS = ("verity-bench/vllm-spec/v0", "verity-bench/vllm-sweep/v0")


def test_the_readme_has_a_line_for_every_spec_and_sweep() -> None:
    ids = set()
    for path in SPECS.glob("*.toml"):
        doc = tomllib.loads(path.read_text())
        if doc.get("schema") in SCHEMAS:
            assert doc["id"] == path.stem, f"{path.name}: its id is {doc['id']!r}; name the file after its id"
            ids.add(doc["id"])
    listed = re.findall(r"^- `([a-z0-9-]+)`: ", (SPECS / "README.md").read_text(), re.M)
    missing, stale = sorted(ids - set(listed)), sorted(set(listed) - ids)
    assert not (missing or stale), ("benchmarks/specs/README.md lists each spec and sweep once, as `- `ID`: what it "
                                    f"measures` in its group: add {missing}, remove {stale}")
    assert len(listed) == len(set(listed)), "a spec or sweep is listed twice in benchmarks/specs/README.md"
