"""`data/hf_configs` holds every configuration the profiles read.

The configuration-derived profiles (`profiles.generic` since P6; before it gen_moe_olmoe, gen_dense2_*, gen_ln_*, dense_generic) and their tests read
`data/hf_configs/<ROLE>.config.json` from the tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HF_DIR = REPO / "data" / "hf_configs"
ROLES = ("OLMOE", "LLAMA32_1B", "GEMMA2_2B", "PHI3_MINI", "PYTHIA160M", "B7")


def test_tracked_hf_configs_are_the_roles_the_profiles_read():
    if not HF_DIR.is_dir():
        pytest.skip("data/hf_configs not in this checkout (pre-fix sparse worktree or pod tree shipped before the int-9 roots)")
    have = {p.name for p in HF_DIR.glob("*.config.json")}
    for role in ROLES:
        assert f"{role}.config.json" in have, f"{role}.config.json missing from data/hf_configs"
    for p in HF_DIR.glob("*.config.json"):
        json.loads(p.read_text())     # every shipped config is a JSON document (the profiles json.load them at import)
