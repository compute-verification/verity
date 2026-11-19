"""PoUS's setup as one circuit (`window.Setup`, `setup_proof` on both sides): the claim and the proof layer's draw, a
block's place and inputs, the setup's C over every store block through the door, the prover's commitments, the two
places the verifier stops: the units' constants and held ok (`setup_proof.bound`) and the door's restriction of a block
that reads the supplied salt, and the order of the weights' commitment before the salt."""

from __future__ import annotations

import functools
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.codec import descriptor_chunks, program_digest
from verity.commitments import rowleaf as RL
from verity.commitments.crypto import lms as R
from verity.commitments.periods import interface as I
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
from verity.protocols.resource_accounting.memory.pous.prover import setup_circuit as SC
from verity.protocols.resource_accounting.memory.pous.prover import setup_proof as SP
from verity.protocols.resource_accounting.memory.pous.prover import window as WIN
from verity.protocols.resource_accounting.memory.pous.verifier import setup_proof as V
from verity.protocols.resource_accounting.memory.pous.verifier.schemes import p2 as P2

RUN = "ab" * 64


def draw(units):
    """The proof layer's draw of `units` among the setup's twelve blocks (each costs the door a parse of C)."""
    law = {"law": "subset", "k": len(units)}
    return S.Draw(S.WindowCommitmentRecord({}, {}, "", 0), law, {**law, "population": 12, "units": list(units)})


#: the proof layer's draw of two of the twelve blocks
DRAW = draw([0, 11])


def test_the_claim_is_the_draws_and_the_stores():
    """The claim's law is the proof layer's draw's and its store the registered store's root; it round-trips."""
    store = {"value": "store", "root": bytes(range(64)).hex(), "leaves": 12}
    c = V.Claim.of({"w": 64}, bytes(88), store, DRAW, b"\x01\x02")
    assert c.law == DRAW.law and c.store == bytes(range(64))
    assert V.Claim.from_json(c.to_json()) == c


def test_proofs_of_other_blocks_than_the_draws_are_refused():
    """The setup's check takes no sample of its own: proofs of a block the proof layer did not draw, or missing a drawn
    block, are refused by name before the door reads anything; and a finding naming another block than the drawn one it
    was offered for is refused too."""
    params, salt, _ = setup_case()
    for proofs in ({0: None, 5: None}, {0: None}, {0: None, 11: None, 5: None}):
        with pytest.raises(V.Undrawn, match=r"^the proofs are of blocks .*, the proof layer drew \[0, 11\]$"):
            V.check(None, None, proofs, b"", {}, {}, DRAW, params, salt, 16)
    finding = type("F", (), {"block": 11, "session": "s0", "accepted": True, "why": ""})()
    seen = []
    verify = lambda proof, **kw: (seen.append(kw["draw"]) or ([finding], None))  # noqa: E731
    with pytest.raises(V.Undrawn, match=r"^setup block 0: session s0 is of block 11$"):
        with pytest.MonkeyPatch.context() as m:
            m.setattr(V, "bound", lambda *a: None)
            V.check(lambda *a: None, verify, {0: None, 11: None}, b"", {}, {"supplied": {V.SALT: V.SALT_MARK}}, DRAW,
                    params, salt, 16)
    assert seen == [DRAW.draw]


def test_padded_is_each_blocks_slot_payload():
    """`padded`'s slice at block u is `slot_payload` at `place(u)`: `SetupGives`' `slice` of the W it reads."""
    p = SP.payload_bytes(64)
    rows = [bytes(range(i, i + 15)) for i in range(3)]
    w = SP.padded(rows, 64)
    for u in range(len(w) // p):
        r, k = SP.place(64, 15, u)
        assert w[u * p:(u + 1) * p] == SC.slot_payload(rows[r], k, p)


def test_unit_inputs_salt_place_and_slot():
    salt = bytes(range(24))
    block, got, at, row, slot = SP.unit_inputs(64, salt, 4, b"x" * 8, bytes(15), 5, 15)
    assert got == salt and at == (1).to_bytes(8, "big") + (1).to_bytes(8, "big") and slot == 2 and block == b"x" * 8
    with pytest.raises(ValueError):
        SP.unit_inputs(64, salt, 4, b"x" * 8, bytes(14), 5, 15)


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
    return params, salt, WIN.Setup(64, 189, 16, 4, 4, salt, store, weights)


@functools.cache
def registered():
    """C (the setup's Program's descriptor bytes), the setup's window with the store and W committed, and the partition
    object with its blocks (`interface.partition_blocks`)."""
    params, salt, setup = setup_case()
    circuit = setup.circuit()
    program, _query = circuit.program()
    win = I.Window(circuit, RUN, {"law": "all"}, WIN.CAP, registered=setup.registered(program_digest(program)))
    part = {**win.window.partition, "blocks": [I.partition_blocks(circuit, WIN.CAP, (program, _query))]}
    return b"".join(descriptor_chunks(program)), win, part, circuit.registered.doc()


@pytest.mark.slow
def test_the_setup_is_one_circuit_over_every_store_block():
    """One template, the setup unit, a unit a store block; each reads its block of the store, its row of W and the
    supplied salt as registered reads, takes its place and its slot as C's inputs, and is a proof block of its own: no sampled
    list, no row the verifier makes. A setup over another count of blocks or rows than it reads is refused."""
    _, _, setup = setup_case()
    circuit = setup.circuit()
    assert setup.blocks == 12 and circuit.templates == ((setup.unit, 12),)
    assert circuit.registered.supplied == {V.SALT: V.SALT_MARK}
    assert circuit.registered.params == {"store": 48, "weights": 32, V.SALT: 12}
    assert circuit.registered.tables == (WIN.store_table(12, 4),)
    (block, row, salt) = circuit.reads
    units = np.arange(12)
    assert block.source(units).tolist() == (4 * units).tolist()
    assert row.source(units).tolist() == (8 * (units // 3)).tolist()
    assert salt.source(units).tolist() == [0] * 12
    assert sorted(p.split("/")[1] for p in circuit.external) == ["at", "slot"]
    _c, _win, part, _reg = registered()
    assert part["blocks"] == [[[u, u + 1] for u in range(12)]]
    assert OC.block(_c, part, 5) == [(5, 6)]
    with pytest.raises(ValueError, match="12 blocks and 4 rows"):
        WIN.Setup(64, 189, 16, 4, 4, None, [b""] * 11, [b""] * 4).registered("00" * 64)


def test_the_setup_and_the_session_read_one_store():
    """The store is one registered value in one form: the table ``store/blocks`` of one block a row in row-seg/v1
    rows, which the setup's units read and the session's openings open, its root the setup's; past 2^31 rows it is
    refused."""
    _, _, setup = setup_case()
    reg = setup.registered("00" * 64)[WIN.STORE]
    assert reg.schema == RL.SCHEMA_HM96_SHA512_ROW_SEG and reg.rows == WIN.commit_store("00" * 64, setup.store).rows
    session = WIN.session_units(3, 8, 1000, 8, 12, R.Params(R.TOY_N, 1, 1, toy=True))
    assert setup.circuit().registered.tables == session.registered.tables
    with pytest.raises(ValueError, match="1 to 2\\^31 blocks"):
        WIN.store_table((1 << 31) + 1, 4)


def test_the_prover_commits_the_store_then_the_setups_values():
    """The store and W are committed before the run; `run` gives the window the salt as the verifier computes it and
    records every unit, its block, row and salt by reference, its place and its slot as words, and its ok word: a 1 for
    each block, and a 0 where its row of W is not the one the store encodes. The window then closes."""
    _c, win, _part, _reg = registered()
    _, salt, setup = setup_case()
    got, given = [], []
    fake = type("W", (), {"record": lambda self, *a: got.append(a), "give": lambda self, *a: given.append(a)})
    setup.run(fake())
    ((unit, inputs, outputs),) = got
    assert given == [(V.SALT, salt)]
    assert unit is setup.unit and inputs[0] is None and inputs[1] is None and inputs[3] is None
    assert inputs[2].shape == (12, 8)
    assert inputs[4][:, 0].tolist() == [u % 3 for u in range(12)]
    assert outputs[0][:, 0].tolist() == [1] * 12
    _, _, wrong = setup_case(lambda rows: [rows[0], bytes(16), *rows[2:]])
    got.clear()
    wrong.run(fake())
    assert got[0][2][0][:, 0].tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    fresh = I.Window(win.circuit, RUN, {"law": "all"}, WIN.CAP, registered=win.registered)
    setup.run(fresh)
    assert fresh.registered[V.SALT].rows == ((*[int.from_bytes(salt[i:i + 2], "little") for i in range(0, 24, 2)],),)
    assert fresh.close().roots


@pytest.mark.slow
def test_the_verifier_refuses_at_the_units_constants():
    """The verifier reads the partition's blocks for the proof layer's draw and holds C's salt to the supplied
    ``pous.setup.salt/v1``, then stops where each unit's place and slot are constants of C and its ok held to 1, which
    C's format has no form for yet: `NotBound`, by name. A registration whose salt is not so marked is refused before,
    and a partition that leaves a block out before that, by the door (E_BLOCKS)."""
    params, salt, _ = setup_case()
    c, _win, part, reg = registered()
    proofs = dict.fromkeys(DRAW.units)
    with pytest.raises(V.NotBound, match=r"^pous setup: not bound: its block's .* and its slot u mod n constants of the "
                                         r"unit in C, and its ok a public output held to 1$"):
        V.check(OC.block, PF.verify, proofs, c, part, reg, DRAW, params, salt, 16)
    with pytest.raises(V.NotBound, match=r"^pous setup: not bound: C's setup/salt is marked None, not "):
        V.check(OC.block, PF.verify, proofs, c, part, {**reg, "supplied": {}}, DRAW, params, salt, 16)
    with pytest.raises(OC.Refused, match="E_BLOCKS"):
        V.check(OC.block, PF.verify, proofs, c, {**part, "blocks": [part["blocks"][0][:-1]]}, reg, DRAW, params, salt, 16)


@pytest.mark.slow
def test_the_door_stops_at_the_supplied_salt_on_a_setup_block(monkeypatch):
    """Past the binding, the door's verdict on a drawn block: it restricts C to the block itself (C_S, never the
    prover's circuit file) and stops there, by name, where the block reads the supplied salt bit by bit: restrict keeps a
    registered input's leaves (`restrict.py`'s ``registered``) and no supplied input's part (`supplied.part`) yet, so
    the salt's 16-bit leaves read as bits are refused (recut-boundary, circuits'). Slow: restricting the setup's C takes
    about half a minute, and no smaller circuit is the setup's."""
    params, salt, _ = setup_case()
    c, _win, part, reg = registered()
    monkeypatch.setattr(V, "bound", lambda *a: None)
    with tempfile.TemporaryDirectory() as d:
        cf, pub, s = Path(d, "circuit.txt"), Path(d, "pub.bin"), Path(d, "s0")
        cf.write_bytes(b"")
        pub.write_text('{"units": {"indices": [5]}}\n')
        s.mkdir()
        (s / "session.json").write_text("{}")
        with pytest.raises(ValueError, match=r"^recut-boundary: .* reads an input node of 16-bit leaves in 1-bit leaves "):
            V.check(OC.block, PF.verify, {5: PF.Proof(cf, pub, (s,))}, c, part, reg, draw([5]), params, salt, 16)


def test_a_weights_commitment_registered_after_the_salt_is_refused():
    """The salt (`pous.setup.salt/v1`) is the memory attester's coin, drawn after W's commitment: an evaluation of it
    whose W was registered after the coin is refused by name (E_SUPPLIED_ORDER, naming the weights), and one whose W was
    registered before passes. The place that holds the order is the record's: W's registration entry before the coin's
    entry, which the supplied input's check reads when the verifier computes the salt."""
    from verity.circuits import supplied as S

    S.check_order({WIN.WEIGHTS: 4}, 5, V.SALT_MARK["supplied"])
    with pytest.raises(S.SuppliedError, match=r"^E_SUPPLIED_ORDER: \['weights'\] registered after pous\.setup\.salt/v1"):
        S.check_order({WIN.WEIGHTS: 7}, 5, V.SALT_MARK["supplied"])
