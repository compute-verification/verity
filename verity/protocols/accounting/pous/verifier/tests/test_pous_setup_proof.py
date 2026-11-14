"""PoUS's setup as a sampled proof (`setup_proof`, both sides, and `window.SetupIntegration`): the sampling design, the
claim, a block's place and inputs, and the setup's window."""

from __future__ import annotations

import random

import pytest

from verity.circuits.codec import descriptor_id
from verity.commitments.periods import interface as I
from verity.protocols.accounting.pous.prover import setup_circuit as SC
from verity.protocols.accounting.pous.prover import setup_proof as SP
from verity.protocols.accounting.pous.prover import window as WIN
from verity.protocols.accounting.pous.verifier import setup_proof as V
from verity.protocols.accounting.pous.verifier.schemes import p2 as P2


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


def test_setup_window_reads_each_sampled_block_and_its_row():
    """Five sampled blocks of a store of four 16-byte rows (three slots each, the last slot's tail zero), over segments of
    four blocks: each unit reads its block of ``store`` and its row of ``weights`` at their leaves, and records its base,
    its slot and a 1, and a 0 where its row of W is not the store's; a sample past the store is refused."""
    rng = random.Random(5)
    params = P2.P2Params(64, 189, 4, 3)
    salt = bytes(rng.getrandbits(8) for _ in range(24))
    prefix = P2.expand_prefix(params, salt)
    rows = [bytes(rng.getrandbits(8) for _ in range(16)) for _ in range(4)]
    pay = SP.padded(rows, 64)
    store = [b for s in range(3) for b in P2.encode_blocks(params, salt, s, 0, pay[24 * s:24 * s + 24])]
    sample = [0, 4, 8, 5, 11]
    integ = WIN.SetupIntegration(64, 189, 16, 4, 4, sample, prefix, store, rows)
    c = integ.circuit()
    assert dict(c.registered.params) == {"store": 12 * 4, "weights": 4 * 8}
    assert [(r.port, r.param, r.at, r.base) for r in c.registered.reads] == [
        x for i, u in enumerate(sample) for x in (("block", "store", i, 4 * u), ("row", "weights", i, 8 * (u // 3)))]
    win = I.Window(c, "00" * 64, {"law": "all"}, 8)
    integ.run(win)
    (rec,) = win._recorded[descriptor_id(SP.setup_unit(64, 189, 16))]
    assert rec.outputs[0].tolist() == [1] * len(sample) and rec.inputs[3].tolist() == [u % 3 for u in sample]
    assert rec.inputs[1][2].tobytes() == prefix + (2).to_bytes(8, "big") + (0).to_bytes(8, "big")
    rows[1] = bytes(16)
    wrong = I.Window(c, "00" * 64, {"law": "all"}, 8)
    integ.run(wrong)
    assert wrong._recorded[descriptor_id(SP.setup_unit(64, 189, 16))][0].outputs[0].tolist() == [1, 0, 1, 0, 1]
    with pytest.raises(ValueError, match="one of the store's 12"):
        WIN.SetupIntegration(64, 189, 16, 4, 4, [12]).circuit()
