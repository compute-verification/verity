"""PoUS's setup as one circuit (`window.Setup`, `setup_proof` on both sides): the sampling design, the claim, a block's
place and inputs, the setup's C over every store block through the door, the prover's commitments, and the two places
the verifier stops: C's coin-bound inputs (`setup_proof.bound`) and the door's statement (`one_circuit.lower`)."""

from __future__ import annotations

import functools
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.codec import descriptor_chunks, program_digest
from verity.commitments.periods import interface as I
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
from verity.protocols.resource_accounting.memory.pous.prover import setup_circuit as SC
from verity.protocols.resource_accounting.memory.pous.prover import setup_proof as SP
from verity.protocols.resource_accounting.memory.pous.prover import window as WIN
from verity.protocols.resource_accounting.memory.pous.verifier import setup_proof as V
from verity.protocols.resource_accounting.memory.pous.verifier.schemes import p2 as P2

RUN = "ab" * 64
#: the proof system's sample of the twelve blocks (each costs the door a parse of C)
SAMPLE = [0, 11]


def test_sampling_design_samples_8828_at_any_size():
    assert V.sampling_design(1) is None and V.sampling_design(V.SAMPLE) is None
    assert V.sampling_design(V.SAMPLE + 1) == f"rate:8828/{V.SAMPLE + 1}"
    assert V.sampling_design(1 << 19) == "rate:8828/524288"
    with pytest.raises(ValueError):
        V.sampling_design(0)


def test_claim_round_trips():
    c = V.Claim({"w": 64}, bytes(88), bytes(range(64)), V.sampling_design(1 << 19), b"\x01\x02")
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


def setup_case(rows_of_w=None):
    """A store of four 16-byte rows of W (three slots each, the last slot's tail zero) over segments of four blocks, its
    twelve blocks encoded under a salt, and the setup over them: (params, salt, the setup)."""
    rng = random.Random(5)
    params = P2.P2Params(64, 189, 4, 3)
    salt = bytes(rng.getrandbits(8) for _ in range(24))
    rows = [bytes(rng.getrandbits(8) for _ in range(16)) for _ in range(4)]
    pay = SP.padded(rows, 64)
    store = [b for s in range(3) for b in P2.encode_blocks(params, salt, s, 0, pay[24 * s:24 * s + 24])]
    weights = rows if rows_of_w is None else rows_of_w(rows)
    return params, salt, WIN.Setup(64, 189, 16, 4, 4, P2.expand_prefix(params, salt), store, weights)


@functools.cache
def registered():
    """C (the setup's Program's descriptor bytes), the setup's window with the store and W committed, and the partition
    object with its blocks (`interface.partition_blocks`)."""
    params, salt, setup = setup_case()
    circuit = setup.circuit()
    program, _query = circuit.program()
    win = I.Window(circuit, RUN, {"law": "all"}, WIN.CAP, registered=setup.registered(program_digest(program)))
    part = {**win.window.partition, "blocks": I.partition_blocks(circuit, WIN.CAP, (program, _query))}
    return b"".join(descriptor_chunks(program)), win, part, circuit.registered.doc()


@pytest.mark.slow
def test_the_setup_is_one_circuit_over_every_store_block():
    """One template, the setup unit, a unit a store block; each reads its block of the store and its row of W as
    registered reads, takes its base and slot as C's inputs, and is a proof block of its own: no sampled list, no row the
    verifier makes. A setup over another count of blocks or rows than it reads is refused."""
    _, _, setup = setup_case()
    circuit = setup.circuit()
    assert setup.blocks == 12 and circuit.templates == ((setup.unit, 12),)
    assert circuit.registered.supplied == () and circuit.registered.params == {"store": 48, "weights": 32}
    (block, row) = circuit.reads
    units = np.arange(12)
    assert block.source(units).tolist() == (4 * units).tolist()
    assert row.source(units).tolist() == (8 * (units // 3)).tolist()
    assert sorted(p.split("/")[1] for p in circuit.external) == ["base", "slot"]
    _c, _win, part, _reg = registered()
    assert part["blocks"] == [[u, u + 1] for u in range(12)]
    assert OC.block(_c, part, 5) == [(5, 6)]
    with pytest.raises(ValueError, match="12 blocks and 4 rows"):
        WIN.Setup(64, 189, 16, 4, 4, None, [b""] * 11, [b""] * 4).registered("00" * 64)


def test_the_prover_commits_the_store_then_the_setups_values():
    """The store and W are committed before the run; `run` records every unit, its block and row by reference, its base and
    slot as words, and its bit: a 1 for each block, and a 0 where its row of W is not the one the store encodes. The window
    then closes."""
    _c, win, _part, _reg = registered()
    _, _, setup = setup_case()
    got = []
    setup.run(type("W", (), {"record": lambda self, *a: got.append(a)})())
    ((unit, inputs, outputs),) = got
    assert unit is setup.unit and inputs[0] is None and inputs[2] is None
    assert inputs[1].shape == (12, 72) and inputs[3][:, 0].tolist() == [u % 3 for u in range(12)]
    assert outputs[0].tolist() == [1] * 12
    _, _, wrong = setup_case(lambda rows: [rows[0], bytes(16), *rows[2:]])
    got.clear()
    wrong.run(type("W", (), {"record": lambda self, *a: got.append(a)})())
    assert got[0][2][0].tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    fresh = I.Window(win.circuit, RUN, {"law": "all"}, WIN.CAP, registered=win.registered)
    setup.run(fresh)
    assert fresh.close().roots


@pytest.mark.slow
def test_the_verifier_refuses_at_cs_coin_bound_inputs():
    """The verifier reads the partition's blocks for the sample, then stops where C's base and slot need their binding to
    the salt and to the unit's position (circuits' input kind, Core's side lean's): `NotBound`, by name. A partition that
    leaves a block out is refused before, by the door (E_BLOCKS)."""
    params, salt, _ = setup_case()
    c, _win, part, reg = registered()
    with pytest.raises(V.NotBound, match=r"^pous setup: no coin-bound input kind yet: base\(u\) = expand_prefix\(params, "
                                         r"salt\) .*; slot\(u\) = u mod n"):
        V.check({}, c, part, reg, SAMPLE, params, salt, 16)
    with pytest.raises(OC.Refused, match="E_BLOCKS"):
        V.check({}, c, {**part, "blocks": part["blocks"][:-1]}, reg, SAMPLE, params, salt, 16)


@pytest.mark.slow
def test_the_door_stops_at_lower_on_a_setup_block(monkeypatch):
    """Past the binding, the door's verdict on a sampled block: it restricts C to the block itself (C_S, never the
    prover's circuit file) and stops at its statement, `one_circuit.lower` (proofs'). Slow: restricting the setup's C
    takes over a minute, and no smaller circuit is the setup's."""
    params, salt, _ = setup_case()
    c, _win, part, reg = registered()
    monkeypatch.setattr(V, "bound", lambda *a: None)
    with tempfile.TemporaryDirectory() as d:
        cf, pub, s = Path(d, "circuit.txt"), Path(d, "pub.bin"), Path(d, "s0")
        cf.write_bytes(b"")
        pub.write_bytes(b"")
        s.mkdir()
        (s / "session.json").write_text("{}")
        with pytest.raises(ValueError, match=r"^setup block 5: the door refused \(circuit\): .*" + OC.PREFIX + "lower"):
            V.check({5: PF.Proof(cf, pub, (s,))}, c, part, reg, [5], params, salt, 16)
