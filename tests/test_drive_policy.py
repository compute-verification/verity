"""policy.toml, drive's policy over what the runners report, parses strictly with this tree's research, each allowance's owner a
handle of the registry, and takes only jobs ci.toml declares off the gate: drive reads it at the tip it judges and a bad file
counts as no policy, so only this test catches one before it lands."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from research import policy
from research.queue import REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def test_policy_toml_parses_and_takes_only_declared_jobs_off_the_gate():
    handles = {h["handle"] for h in json.loads((ROOT / REGISTRY).read_text())["handles"]}
    pol = policy.parse((ROOT / policy.PATH).read_text(), policy.PATH, handles)
    jobs = {t["name"] for t in tomllib.loads((ROOT / "tools/verity/check/ci.toml").read_text())["job"]}
    assert set(pol.off_gate) <= jobs, f"{policy.PATH}'s off_gate names jobs ci.toml doesn't declare: {set(pol.off_gate) - jobs}"
