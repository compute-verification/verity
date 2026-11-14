"""PoUS's setup as a sampled proof (`setup_proof`, both sides): the sampling design, the claim, and a block's place and
inputs."""

from __future__ import annotations

import pytest

from verity.protocols.accounting.pous.prover import setup_circuit as SC
from verity.protocols.accounting.pous.prover import setup_proof as SP
from verity.protocols.accounting.pous.verifier import setup_proof as V


def test_law_samples_8828_at_any_size():
    assert V.law(1) is None and V.law(V.SAMPLE) is None
    assert V.law(V.SAMPLE + 1) == f"rate:8828/{V.SAMPLE + 1}" and V.law(1 << 19) == "rate:8828/524288"
    with pytest.raises(ValueError):
        V.law(0)


def test_claim_round_trips():
    c = V.Claim({"w": 64}, bytes(88), bytes(range(64)), "ab" * 64, V.law(1 << 19), b"\x01\x02")
    assert V.Claim.from_json(c.to_json()) == c


def test_padded_is_each_blocks_slot_payload():
    """`padded`'s slice at block u is `slot_payload` at `place(u)`: `SetupGives`' `slice` of the W it reads."""
    p = SP.payload_bytes(64)
    rows = [bytes(range(i, i + 15)) for i in range(3)]
    w = SP.padded(rows, 64)
    for u in range(len(w) // p):
        r, k = SP.place(64, 15, u)
        assert w[u * p:(u + 1) * p] == SC.slot_payload(rows[r], k, p)


def test_unit_inputs_base_and_slot():
    prefix = bytes(128)
    block, base, row, slot = SP.unit_inputs(64, prefix, 4, b"x" * 8, bytes(15), 5, 15)
    assert base == prefix + (1).to_bytes(8, "big") + (1).to_bytes(8, "big") and slot == 2 and block == b"x" * 8
    with pytest.raises(ValueError):
        SP.unit_inputs(64, prefix, 4, b"x" * 8, bytes(14), 5, 15)
