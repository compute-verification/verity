"""PoUS's prover's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.periods.interface`): its circuit in proof blocks of PoUS's Definitions (`decode_circuit`,
`answer_circuit`, `verdict_circuit`), its values recorded into a `Window`, and the window's four records written for the
workers, which stage from them as they do from an adapter's.

`SetupIntegration` is the setup's statements (each a sampled block's decode against the store's registered row), an
`Integration`: its `circuit()` and its `run(window)`. `session_units` is a session's units in the registered circuit, its
circuit alone: no values are recorded for it yet.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import bits as EB
from verity.circuits.defs import bind
from verity.commitments.crypto import lms as R
from verity.commitments.gates.embedding import read as table_read
from verity.commitments.periods.interface import Circuit, Read, Registered, Table, TableOpening, TableRead, Window, Wire
from verity.protocols.resource_accounting.memory.pous.verifier.setup_proof import BASE, OK, SLOT, base, leaf_bytes

from . import answer_circuit, decode_circuit, session_circuit, verdict_circuit
from . import setup_circuit as SP
from .decode_circuit import P2_BASE_BYTES
from .setup_proof import payload_bytes, place, setup_unit, unit_inputs

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit, session_circuit)
#: the setup's registered parameters: the store's blocks and W's rows
STORE, WEIGHTS = "store", "weights"
#: the session's: the store's blocks as a table of one block a row, and the memory attester's public key
STORE_TABLE, ATTESTER_KEY = "store/blocks", "attester-key"


class SetupIntegration:
    """PoUS's setup as an integration: one proof block per sampled block, the setup unit (`setup_proof.setup_unit`) over
    the block of the store, its base, the registered row of W that holds its payload and its slot (`setup_proof.place`).
    Every input is a registered read: the block and the row of the store's and W's registered values at public leaves,
    and the base and the slot of the rows the verifier makes from its own expansion and the sample (`BASE`, `SLOT`,
    PoUS's verifier's `setup_proof.supplied`), so a sampled block is tied to its index. Each unit's ok output reads the
    verifier's supplied 1 (`OK`), so a unit whose ok is 0 opens no root the verifier pins.

    `w`, `c0` are P2's point, `row_bytes` W's rows' bytes N, `rows` how many, `blocks_per_segment` the store's segment
    length; `sample` is the challenger's sampled block indices, sampled after the store's commitment is registered;
    `prefix` is the salt's 128-byte expansion, `store` and `weights` the store's blocks and W's rows, and `module` core's
    ``one_stage.registered`` as the caller passes it, which only `run` reads."""

    def __init__(self, w: int, c0: int, row_bytes: int, rows: int, blocks_per_segment: int, sample: Sequence[int],
                 prefix: bytes | None = None, store: Sequence[bytes] | None = None,
                 weights: Sequence[bytes] | None = None, module: Any = None) -> None:
        self.w, self.c0, self.row_bytes, self.rows = w, c0, row_bytes, rows
        self.blocks_per_segment, self.sample = blocks_per_segment, tuple(sample)
        self.prefix, self.store, self.weights, self.module = prefix, store, weights, module

    def reads(self) -> tuple[Read, ...]:
        """Each sampled block's four reads (16-bit leaves): its block of ``store`` at u·W/16, its row of ``weights`` at
        ⌊u/n⌋·N/2, and its own row of the supplied ``BASE`` (L/2 leaves) and ``SLOT`` (one leaf); and every unit's ok
        output ``o0`` the one row of the supplied ``OK``."""
        tid, bw, rw = descriptor_id(setup_unit(self.w, self.c0, self.row_bytes)), self.w // 16, self.row_bytes // 2
        lw = P2_BASE_BYTES // 2
        return (*(r for i, u in enumerate(self.sample) for r in (
            Read(tid, "block", STORE, i, (1,), u * bw, (0,)),
            Read(tid, "row", WEIGHTS, i, (1,), place(self.w, self.row_bytes, u)[0] * rw, (0,)),
            Read(tid, "base", BASE, i, (1,), i * lw, (0,)),
            Read(tid, "slot", SLOT, i, (1,), i, (0,)))),
            Read(tid, "o0", OK, 0, (len(self.sample),), 0, (0,)))

    def circuit(self) -> Circuit:
        """The sampled blocks' units over their registered reads (`reads`), the base, the slot and the ok of the
        registered values the verifier supplies (`Registered.supplied`)."""
        unit = setup_unit(self.w, self.c0, self.row_bytes)
        n, k = SP.slots(self.row_bytes, payload_bytes(self.w)), len(self.sample)
        if any(not 0 <= u < self.rows * n for u in self.sample):
            raise ValueError(f"a sampled block is one of the store's {self.rows * n}")
        params = {STORE: self.rows * n * (self.w // 16), WEIGHTS: self.rows * (self.row_bytes // 2),
                  BASE: k * (P2_BASE_BYTES // 2), SLOT: k, OK: 1}
        return Circuit("pous-setup", ((unit, k),),
                       registered=Registered(params, reads=self.reads(), supplied=(BASE, SLOT, OK)))

    def supplied_rows(self) -> dict[str, list[bytes]]:
        """The supplied values' rows as the verifier makes them (`setup_proof.supplied`'s bytes): one a sampled block of
        the base's port words (`leaf_bytes`) and of the slot as one big-endian 16-bit leaf, and the ok's one row, 1."""
        if self.prefix is None:
            raise ValueError("SetupIntegration.supplied_rows reads the salt's expansion")
        n = SP.slots(self.row_bytes, payload_bytes(self.w))
        return {BASE: [leaf_bytes(base(self.prefix, self.blocks_per_segment, u)) for u in self.sample],
                SLOT: [(u % n).to_bytes(2, "big") for u in self.sample], OK: [(1).to_bytes(2, "big")]}

    def run(self, window: Window) -> None:
        """The supplied base, slot and ok rows (`supplied_rows`, given to the window), and each unit's bit, every input
        recorded by reference."""
        if self.prefix is None or self.store is None or self.weights is None:
            raise ValueError("SetupIntegration.run reads the salt's expansion, the store's blocks and W's rows")
        unit = setup_unit(self.w, self.c0, self.row_bytes)
        cases = [unit_inputs(self.w, self.prefix, self.blocks_per_segment, self.store[u],
                             self.weights[place(self.w, self.row_bytes, u)[0]], u, self.row_bytes) for u in self.sample]
        block, base_bits, row = (_bit_rows([c[k] for c in cases]) for k in range(3))
        slot = np.array([[c[3] >> i & 1 for i in range(SP.SLOT_BITS)] for c in cases], dtype=np.uint8)
        ok = EB.evaluate_bits(unit, block, base_bits, row, slot)
        for param, rows in self.supplied_rows().items():
            window.give(param, rows, self.module)
        window.record(unit, [None, None, None, None], [ok[:, 0].astype(np.uint16)])


def _bit_rows(rows: Sequence[bytes]) -> np.ndarray:
    """Byte strings as bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋, one row each."""
    return np.unpackbits(np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), -1), axis=1, bitorder="little")


def coin_bound(rounds: dict[str, int]) -> tuple:
    """STAND-IN, the one place the session's coin-bound inputs are declared: ``ji`` (``pous.session.round_coin(j)``, 96
    bits) of each round's unit and ``tok`` (``pous.session.round_key(j)``, 32 bits) of each opening, ``head``
    (``pous.session.header()``) and ``ctx`` (``pous.session.context()``) of the signed digest's unit, each to be
    circuits' ``CoinInput<w>_v1`` entry, which the verifier computes from the released commitment and coin openings.
    Until that input exists they are declared nowhere, so they stay rows the prover fills: nothing binds them, and a
    circuit built here proves nothing about the coins. ``rounds`` is each round template's units by descriptor id."""
    return ()


def session_units(k: int, head: int, limit_ns: int, block_bytes: int, blocks: int, params: R.Params = R.DEPLOYED) -> Circuit:
    """A session of ``k`` rounds over a store of ``blocks`` blocks of ``block_bytes`` bytes, as units of the registered
    circuit, in the order its wires run (each writer before its readers):

        PousSessionDigest_v1{k, head, limit_ns}  1 unit    (head, ctx, the records rows) -> (ok, m, the records rows)
        PousSessionSig_v1{N, W, H}               1 unit    key: the registered key; m: the digest's m by wire
        EmbeddingRead_v2{blocks, block_bytes/2}  k units   round j's opening of the store's table at its index
        PousRound_v2{block_bytes, 0}             ⌈k/2⌉     round j = 2u: records row u by wire, row from opening j
        PousRound_v2{block_bytes, 1}             ⌊k/2⌋     round j = 2u + 1: the same, at the record's second slot

    The store's table and the key are registered values; the coin-bound inputs are `coin_bound`'s."""
    dig = bind(session_circuit.PousSessionDigest, K=k, HEAD=head, LIMIT=limit_ns)
    sig = bind(session_circuit.PousSessionSig, N=params.N, W=params.W, H=params.H)
    opener = table_read(blocks, block_bytes // 2)
    slots = [(bind(session_circuit.PousRound, N=block_bytes, SLOT=s), (k - s + 1) // 2) for s in (0, 1)]
    slots = [(t, n) for t, n in slots if n]
    d, g, o = (descriptor_id(t) for t in (dig, sig, opener))
    wires = [Wire(g, "m", d, "o1", 0, (1,), 0, (0,))]
    for s, (t, n) in enumerate(slots):
        r = descriptor_id(t)
        wires += [Wire(r, "records", d, f"o{2 + u}", u, (1,), 0, (0,)) for u in range(n)]
        wires.append(Wire(r, "row", o, "o0", 0, (n,), s, (2,)))
    key_words = (16 + params.N) // 2
    registered = Registered(
        {STORE: blocks * block_bytes // 2, ATTESTER_KEY: key_words},
        tables=(Table(STORE_TABLE, STORE, 0, block_bytes // 2, blocks),),
        reads=(Read(g, "key", ATTESTER_KEY, 0, (1,), 0, (0,)),),
        roots=(TableRead(o, "root", STORE_TABLE, "root", 0, k), TableRead(o, "domain", STORE_TABLE, "domain", 0, k)),
        openings=(TableOpening(o, STORE_TABLE, "tok", ("row", "salt", "path"), "o1", 0, k, row="o0"),))
    coin_bound({descriptor_id(t): n for t, n in slots})
    return Circuit("pous-session", ((dig, 1), (sig, 1), (opener, k), *slots), wires=tuple(wires), registered=registered)
