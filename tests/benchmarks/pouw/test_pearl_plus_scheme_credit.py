"""Pearl+'s resource scheme (`verity.protocols.resource_accounting.compute.pouw.scheme`) against the run verifier's credit
(``benchmarks/pouw/pearl_c_vllm/verify_run.py``, `Credit.lean`'s Python twin): a block's cost is a whole tile's credited
MACs, and each charge's bound is a key of the work the verifier publishes."""
from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c import PearlC
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c_work import UnitRecord
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import Matmul, Shape

VERIFY_RUN = Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "pearl_c_vllm" / "verify_run.py"


def _verify_run():
    name = "pearl_c_vllm_verify_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, VERIFY_RUN)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_the_verifiers_protocol_and_assumption() -> None:
    vr = _verify_run()
    assert PS.PEARL_PLUS.protocol == vr.PROTOCOL and PS.PEARL_PLUS.assumptions == (vr.ASSUMPTION_H100,)


def test_block_cost_is_the_credits_macs() -> None:
    vr = _verify_run()
    scheme = PearlC()
    tm, tn = scheme.tile
    record = UnitRecord(Matmul(0, Shape(tm, 1024, tn), 0), real_rows=tm, k_kept=1020)
    credit = vr.credit_at_assumption([SimpleNamespace(scheme=scheme, records=[record])], 3, Fraction(1, 100))
    for block in PS.PEARL_PLUS.blocks:
        assert block.cost({"TM": tm, "TN": tn, "K_KEPT": record.k_kept}) == credit["macs"]
    assert Fraction(credit["rho"]) == scheme.cap
    assert {c.bound for c in PS.PEARL_PLUS.charges} <= {*credit, *vr.grid_work(credit, credit)}
