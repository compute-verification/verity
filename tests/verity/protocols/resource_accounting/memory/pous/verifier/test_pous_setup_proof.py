"""PoUS's setup as a sampled proof (`setup_proof`, both sides, and `window.SetupIntegration`): the sampling design, the
claim, a block's place and inputs, and the setup's window."""

from __future__ import annotations

import random

import numpy as np
import pytest

from verity.commitments.periods import interface as I
from verity.protocols.resource_accounting.memory.pous.prover import setup_circuit as SC
from verity.protocols.resource_accounting.memory.pous.prover import setup_proof as SP
from verity.protocols.resource_accounting.memory.pous.prover import window as WIN
from verity.protocols.resource_accounting.memory.pous.verifier import setup_proof as V
from verity.protocols.resource_accounting.memory.pous.verifier.schemes import p2 as P2


def test_sampling_design_samples_8828_at_any_size():
    assert V.sampling_design(1) is None and V.sampling_design(V.SAMPLE) is None
    assert V.sampling_design(V.SAMPLE + 1) == f"rate:8828/{V.SAMPLE + 1}"
    assert V.sampling_design(1 << 19) == "rate:8828/524288"
    with pytest.raises(ValueError):
        V.sampling_design(0)


def test_claim_round_trips():
    c = V.Claim({"w": 64}, bytes(88), bytes(range(64)), "ab" * 64, V.sampling_design(1 << 19), b"\x01\x02")
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


SAMPLE = [0, 4, 8, 5, 11]


def setup_case(rows_of_w=None):
    """A store of four 16-byte rows of W (three slots each, the last slot's tail zero) over segments of four blocks, and
    the setup of five of its blocks: (params, salt, the integration)."""
    rng = random.Random(5)
    params = P2.P2Params(64, 189, 4, 3)
    salt = bytes(rng.getrandbits(8) for _ in range(24))
    rows = [bytes(rng.getrandbits(8) for _ in range(16)) for _ in range(4)]
    pay = SP.padded(rows, 64)
    store = [b for s in range(3) for b in P2.encode_blocks(params, salt, s, 0, pay[24 * s:24 * s + 24])]
    weights = rows if rows_of_w is None else rows_of_w(rows)
    return params, salt, WIN.SetupIntegration(64, 189, 16, 4, 4, SAMPLE, P2.expand_prefix(params, salt), store, weights,
                                              module="one_stage.registered")


class Given:
    """A window's stand-in that keeps what `run` gives it and records (`Window.give` is proofs' and not written)."""

    def __init__(self) -> None:
        self.given, self.recorded = {}, []

    def give(self, param, rows, module):
        self.given[param] = (list(rows), module)

    def record(self, template, inputs, outputs, label=None):
        self.recorded.append((template, inputs, outputs))


def test_setup_window_reads_every_input_of_each_sampled_proof_block():
    """Each unit reads its block of ``store`` and its row of ``weights`` at their leaves, and its own row of the supplied
    base (72 leaves) and slot (one leaf), and every unit's ok output the supplied ok's one row; the circuit names the
    three as supplied, which proofs' `Registered` refuses until the window gives them (`proofs/zero-knowledge`); a
    sample past the store is refused first."""
    _, _, integ = setup_case()
    assert [(r.port, r.param, r.at, r.base) for r in integ.reads()] == [
        x for i, u in enumerate(SAMPLE) for x in (("block", "store", i, 4 * u), ("row", "weights", i, 8 * (u // 3)),
                                                  ("base", V.BASE, i, 72 * i), ("slot", V.SLOT, i, i))] + [
        ("o0", V.OK, 0, 0)]
    assert (integ.reads()[-1].radix, integ.reads()[-1].step) == ((len(SAMPLE),), (0,))
    with pytest.raises(NotImplementedError, match=r"^proofs/zero-knowledge: .*Registered.supplied \['setup/base', "
                                                  r"'setup/slot', 'setup/ok'\]"):
        integ.circuit()
    with pytest.raises(ValueError, match="one of the store's 12"):
        WIN.SetupIntegration(64, 189, 16, 4, 4, [12]).circuit()


def test_setup_run_gives_the_verifiers_rows_and_records_each_units_bit():
    """`run` gives the window the base, slot and ok rows the verifier makes (`setup_proof.supplied`'s bytes, under the
    caller's module), records every input by reference, and each unit's bit: a 1, and a 0 where its row of W is not the
    store's. On a real window it raises through proofs' `Window.give`."""
    params, salt, integ = setup_case()
    win = Given()
    integ.run(win)
    want = V.supplied(params, salt, 16, SAMPLE)
    assert win.given == {v: ([bytes.fromhex(r[0]["bytes"]) for r in want[v]], "one_stage.registered")
                         for v in (V.BASE, V.SLOT, V.OK)}
    assert win.given[V.OK][0] == [b"\x00\x01"]
    ((_, inputs, outputs),) = win.recorded
    assert inputs == [None] * 4 and outputs[0].tolist() == [1] * len(SAMPLE)
    _, _, wrong = setup_case(lambda rows: [rows[0], bytes(16), *rows[2:]])
    win = Given()
    wrong.run(win)
    assert win.recorded[0][2][0].tolist() == [1, 0, 1, 0, 1]
    unit = SP.setup_unit(64, 189, 16)
    plain = I.Circuit("pous-setup", ((unit, len(SAMPLE)),), registered=I.Registered({"store": 48, "weights": 32}))
    with pytest.raises(NotImplementedError, match=r"^proofs/zero-knowledge: .*Window.give"):
        integ.run(I.Window(plain, "00" * 64, {"law": "all"}, 8))


def test_supplied_rows_read_the_verifiers_bytes_as_the_words_the_units_ports_hold():
    """SPOT reads a supplied row's bytes as the parameter's leaves of big-endian 32-bit halves (a 16-bit half for two
    bytes), least significant first (`one_stage.registered.supplied_rows`), at the unit's ports' leaf width, one bit;
    the unit's base port holds the base's bits in byte order, the little-endian 16-bit words its run recorded, and its
    slot port the slot's bits. Over the verifier's hex, `supplied_rows` gives each sampled block's base and slot as the
    unit reads them, and packed into 16-bit words (`bit_words`) the words its port records."""
    from verity.protocols.proofs.spot.verifier.one_stage import registered as RG

    params, salt, integ = setup_case()
    params_of, out = SP.setup_unit(64, 189, 16).signature()
    ports = dict(params_of)

    class Unit:
        fn = type("Fn", (), {"signature": staticmethod(
            lambda: (((V.BASE, ports["base"]), (V.SLOT, ports["slot"])), out))})

    got = RG.sources(V.supplied(params, salt, 16, SAMPLE))
    prefix = P2.expand_prefix(params, salt)
    bases = [SP.unit_inputs(64, prefix, 4, b"", bytes(16), u, 16)[1] for u in SAMPLE]
    base_leaves = RG.supplied_rows(Unit, V.BASE, got[V.BASE], {})
    slot_leaves = RG.supplied_rows(Unit, V.SLOT, got[V.SLOT], {})
    assert base_leaves == WIN._bit_rows(bases).tolist()
    assert [RG.bit_words(r, 1, 16) for r in base_leaves] == [np.frombuffer(b, dtype="<u2").tolist() for b in bases]
    slots = [SP.place(64, 16, u)[1] for u in SAMPLE]
    assert slot_leaves == [[s >> i & 1 for i in range(SC.SLOT_BITS)] for s in slots]
    assert [RG.bit_words(r, 1, 16) for r in slot_leaves] == [[s] for s in slots]
    assert all(len(b) == 144 for b in bases) and integ.supplied_rows()[V.BASE] == [V.leaf_bytes(b) for b in bases]


def test_a_unit_whose_ok_is_0_opens_no_root_the_verifier_pins():
    """Every unit's ``o0`` reads the supplied ok, which SPOT reads from the verifier's hex as the word 1 (`supplied_rows`,
    at the bit leaves of the output's zero-padded word or a 16-bit leaf), and which the period commits each unit's ``o0``
    under (`Window.close`: an output row a registered read names stays the unit's own, under the value's salt). The
    stand-in for the Lean verifier's opening of a registered read, which `Window.give` keeps from running: a unit row's
    root under the supplied salts, against the root of the verifier's own row. The units whose ok is 1 open it; the unit
    whose row of W is wrong records ok 0, whose row opens another root, so its proof is refused."""
    from verity.protocols.proofs.spot.verifier.one_stage import registered as RG

    params, salt, _ = setup_case()
    _, _, wrong = setup_case(lambda rows: [rows[0], bytes(16), *rows[2:]])
    win = Given()
    wrong.run(win)
    ok = win.recorded[0][2][0].tolist()
    assert ok == [1, 0, 1, 0, 1]
    params_of, out = SP.setup_unit(64, 189, 16).signature()
    word = dict(params_of)["slot"]

    class Unit:
        fn = type("Fn", (), {"signature": staticmethod(lambda: (((V.OK, word),), out))})

    (row,) = RG.sources(V.supplied(params, salt, 16, SAMPLE))[V.OK]
    (leaves,) = RG.supplied_rows(Unit, V.OK, [row], {})
    assert RG.bit_words(leaves, 1, 16) == [1] and RG.leaves_of(bytes.fromhex(row[0][1]), 16) == [1]
    prog = "ab" * 64
    salts = RG.supplied_salts(prog, V.OK, 1)
    pinned = RG.Registered.commit(prog, V.OK, [[1]], salts).root
    opens = [RG.Registered.commit(prog, V.OK, [[o]], salts).root == pinned for o in ok]
    assert opens == [True, False, True, False, True]


def test_register_hands_the_challenger_the_verifiers_supplied_rows():
    """`register` hands the proof system's challenger (`Stream.register`'s form) the docket, the layout's ports, the
    registered values and the rows the verifier makes from the salt and the sample."""
    params, salt, _ = setup_case()

    class Stream:
        def register(self, record, ports, **copies):
            self.got = (record, ports, copies)
            return "receipt"

    s = Stream()
    assert V.register(s, {"format": "x"}, {"block": {}}, {"store": 48, "weights": 32}, params, salt, 16,
                      SAMPLE) == "receipt"
    assert s.got == ({"format": "x"}, {"block": {}}, {"registered": {"store": 48, "weights": 32},
                                                       "supplied": V.supplied(params, salt, 16, SAMPLE)})
