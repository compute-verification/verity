"""Lane MAND: capture pre-change pinned values for the bit-identical regression.

Run BEFORE and AFTER the schema/fold change; the JSON must be identical.
Usage: PYTHONPATH=. python scripts/mand_baseline_pin.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

from veritor.analysis import bound
from veritor.core import VerificationPolicy

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "sec_conftest", ROOT / "tests" / "veritor" / "security" / "conftest.py"
)
sec = importlib.util.module_from_spec(spec)
sys.modules["sec_conftest"] = sec
spec.loader.exec_module(sec)

POLICIES = [
    (VerificationPolicy(Fraction(1, 2), Fraction(1, 2)), Fraction(1, 4)),
    (VerificationPolicy(Fraction(1, 3), Fraction(1, 5)), Fraction(1, 100)),
    (VerificationPolicy(Fraction(1, 2), 1), Fraction(1, 4)),
    (VerificationPolicy(1, Fraction(1, 2)), Fraction(1, 8)),
    (VerificationPolicy(0, 1), Fraction(1, 2)),
    (VerificationPolicy(1, 1), Fraction(0)),
]

record: dict = {"policy_digests": {}, "bounds": {}}
for policy, _ in POLICIES:
    record["policy_digests"][f"q={policy.q},s={policy.s}"] = str(policy.digest)

targets = {
    "chain-2x2": sec.chain_compiled(stages=2, cells=2, width=8),
    "chain-1x2-w2": sec.chain_compiled(stages=1, cells=2, width=2),
    "chain-2x2-wide": sec.chain_compiled(stages=2, cells=2, width=8, wide_units=True),
    "random-0-1": sec.random_marked_compiled(0, 1),
    "random-3-2": sec.random_marked_compiled(3, 2),
}
for name, compiled in targets.items():
    for policy, eta in POLICIES:
        result = bound(compiled, policy, eta)
        key = f"{name}|q={policy.q},s={policy.s}|eta={eta}"
        record["bounds"][key] = {
            "bits": result.bits.hex() if isinstance(result.bits, float) else result.bits,
            "capped": result.capped,
            "out_bits": result.out_bits,
            "knapsack_bits": float(result.knapsack_bits).hex(),
            "laplace_bits": float(result.laplace_bits).hex(),
            "cost_step": float(result.cost_step).hex(),
            "buckets": result.buckets,
            "errors_limit": result.errors_limit,
            "rho": float(result.rho).hex(),
            "policy_digest": str(result.policy.digest),
        }

out = ROOT / "fixtures" / "mand" / "bound-baseline-pin.json"
out.write_text(json.dumps(record, indent=1, sort_keys=True))
print(f"wrote {out} ({len(record['bounds'])} bound records)")
