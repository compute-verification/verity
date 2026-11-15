"""No key a committed configuration pins is a literal in code: each `[[pin]]` of each configuration
`tools/verity/cli/verity_cli/configurations.toml` commits is stated in that configuration alone, and the code that reads `trusted-device`
(`memory_challenger.trusted_device`) takes it from there. A literal split across lines (each 64-digit half) counts."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "tools" / "verity" / "cli" / "verity_cli" / "configurations.toml"
#: the places whose code reads or passes a device's pin
SCANNED = ("verity/protocols/resource_accounting", "verity/protocols/registration", "benchmarks/pous", "benchmarks/one_stage", "demos",
           "adapters/vllm/verity_vllm/protocols", "tools/verity/cli/verity_cli")
CODE = (".py", ".sh", ".lean")
SKIPPED = {".lake", "__pycache__", ".venv", "target", "node_modules"}


def code() -> list[Path]:
    """The code files under `SCANNED`, build trees left out."""
    out = []
    for d in SCANNED:
        for top, dirs, files in os.walk(ROOT / d):
            dirs[:] = sorted(x for x in dirs if x not in SKIPPED)
            out += [Path(top) / f for f in sorted(files) if f.endswith(CODE)]
    return out


def pins() -> dict[str, str]:
    """Each pinned key's SHA-512, by the configuration that pins it."""
    out = {}
    for path in tomllib.loads(COMMITTED.read_text()):
        for p in tomllib.loads((ROOT / path).read_text()).get("pin", []):
            out[p["key-sha512"]] = path
    return out


def test_the_committed_configurations_pin_keys() -> None:
    assert pins(), "no committed configuration pins a key: this test would hold nothing"


def test_no_pinned_key_is_a_literal_in_code() -> None:
    needles = {part: key for key in pins() for part in (key, key[:64], key[64:])}
    by = pins()
    found = [f"{f.relative_to(ROOT)}: {needles[n][:16]}… (pinned by {by[needles[n]]})"
             for f in code() for n in needles if n in f.read_text(errors="replace")]
    assert not found, ("a pinned key is a literal in code; read it from the configuration's [[pin]] entries "
                       f"(VERITY_PINS, period.json's deployment) instead: {found}")
