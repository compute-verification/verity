"""`one_linear.py`'s circuit: one template's units, each proof block over the zero-knowledge floor and under 2^27 bits,
and its partition at two levels refused by name at the door, before any coin."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage"))

import one_linear as OL  # noqa: E402
from verity.circuits import laying  # noqa: E402
from verity.commitments.periods import interface as I  # noqa: E402
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC  # noqa: E402


def test_one_template_its_blocks_between_the_floor_and_the_cap():
    c = OL.circuit()
    assert [n for _t, n in c.templates] == [OL.UNITS] and not c.wires
    k_log = I.check_proof_block(OL.gemm())
    assert laying.zk_min_units(k_log, 1) <= OL.CAP and k_log + (OL.CAP - 1).bit_length() <= I.K_MAX


def test_two_levels_are_refused_at_the_door_by_name():
    c, part = OL.two_levels()
    with pytest.raises(OC.Refused, match="partition: blocks-levels: the object has 2 levels of blocks; this reader takes one"):
        OC.block(c, part, 0)
