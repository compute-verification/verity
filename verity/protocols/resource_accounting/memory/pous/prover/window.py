"""PoUS's prover's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.periods.interface`): its circuit in proof blocks of PoUS's Definitions (`decode_circuit`,
`answer_circuit`, `verdict_circuit`), its values recorded into a `Window`, and the window's four records written for the
workers, which stage from them as they do from an adapter's.

`Setup` is the setup's one circuit C (the setup unit over every block of the store), an `Integration`: its `circuit()`
and its `run(window)`. `session_units` is a session's units in the registered circuit, its circuit alone: no values are
recorded for it yet. Both read one store: the registered value ``store``, the table ``store/blocks`` of one block a row
in ``hm96-sha512/row-seg/v1`` rows (`store_table`, `commit_store`), whose root the setup publishes and the session's
openings pin.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import bits as EB
from verity.circuits.defs import bind
from verity.commitments import registered_value as RV
from verity.commitments.crypto import lms as R
from verity.commitments.gates.embedding import MAX_V, read as table_read
from verity.commitments.periods.interface import Circuit, Read, Registered, Table, TableOpening, TableRead, Window, Wire
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2

from . import answer_circuit, decode_circuit, session_circuit, verdict_circuit
from . import setup_circuit as SC
from .setup_proof import payload_bytes, place, setup_unit, unit_inputs

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit, session_circuit)
#: the store, one registered value, and its table of one block a row; W's rows; the memory attester's public key
STORE, STORE_TABLE, WEIGHTS, ATTESTER_KEY = "store", "store/blocks", "weights", "attester-key"
#: the units of a proof block of the setup's C: one store block each
CAP = 1
#: the session's supplied inputs (`coin_bound`): round j's coin value is ``ROUND_COIN/j``
ROUND_COIN, SESSION_HEADER, SESSION_CONTEXT = "round-coin", "session-header", "session-context"


def store_table(blocks: int, words: int) -> Table:
    """The store as the table ``store/blocks`` of the registered ``store``: ``blocks`` rows of ``words`` 16-bit leaves,
    one block a row, at most 2^31 rows (`gates.embedding`'s V), so a round's index is its opening's 32-bit key."""
    if isinstance(blocks, bool) or not isinstance(blocks, int) or not 1 <= blocks <= MAX_V:
        raise ValueError(f"the store is 1 to 2^31 blocks, not {blocks!r}")
    return Table(STORE_TABLE, STORE, 0, words, blocks)


def commit_store(program: str, blocks: Sequence[bytes]) -> RV.Registered:
    """The store committed once, before any coin, under `program`: its `store_table`'s rows, each block's 16-bit words
    in ``hm96-sha512/row-seg/v1``. Its root is the setup's published root, which the memory attester signs as the
    session's ``binding``."""
    store_table(len(blocks), len(blocks[0]) // 2 if blocks else 0)
    return RV.Registered.commit(program, STORE, [np.frombuffer(b, "<u2").tolist() for b in blocks], schema=SEG)


class Setup:
    """PoUS's setup as one circuit C: the setup unit (`setup_proof.setup_unit`, ``PousP2SetupCheck_v2``) over every block
    of the store in order, unit u block u, one proof block a store block (`CAP`, `interface.partition_blocks`). Each unit
    reads its block of the registered ``store`` and the registered row of ``weights`` that holds its payload (`reads`),
    and takes the setup's salt, its place ``u64be(s) ‖ u64be(v)`` and its slot (`setup_proof.at`, `setup_proof.place`)
    as C's input nodes; its ok word is C's output. Which blocks are proved is the proof system's sample on its live coins
    after the store is committed, and which units a block holds the partition's: the setup lists neither.

    The salt is the one value the verifier supplies (`pous.setup.salt/v1`), and each unit's place and slot are constants
    of the unit in C; nothing in C binds them yet, and its ok word is held to 1 by nothing yet: the verifier refuses
    there (`setup_proof.bound`) until the supplied input does.

    `w`, `c0` are P2's point, `row_bytes` W's rows' bytes N, `rows` how many, `blocks_per_segment` the store's segment
    length; `salt` is the setup's salt and `store` and `weights` the store's blocks and W's rows, which only
    `registered` and `run` read."""

    def __init__(self, w: int, c0: int, row_bytes: int, rows: int, blocks_per_segment: int, salt: bytes | None = None,
                 store: Sequence[bytes] | None = None, weights: Sequence[bytes] | None = None) -> None:
        self.w, self.c0, self.row_bytes, self.rows = w, c0, row_bytes, rows
        self.blocks_per_segment, self.salt, self.store, self.weights = blocks_per_segment, salt, store, weights
        self.unit = setup_unit(w, c0, row_bytes)
        self.slots = SC.slots(row_bytes, payload_bytes(w))

    @property
    def blocks(self) -> int:
        """B, the store's blocks: W's rows' slots."""
        return self.rows * self.slots

    def reads(self) -> tuple[Read, ...]:
        """Unit u's two registered reads (16-bit leaves): its block of ``store`` at u·W/16, and its row of ``weights`` at
        ⌊u/n⌋·N/2, n a row's slots."""
        tid = descriptor_id(self.unit)
        return (Read(tid, "block", STORE, 0, (self.blocks,), 0, (self.w // 16,)),
                Read(tid, "row", WEIGHTS, 0, (self.rows, self.slots), 0, (self.row_bytes // 2, 0)))

    def circuit(self) -> Circuit:
        tid = descriptor_id(self.unit)
        params = {STORE: self.blocks * (self.w // 16), WEIGHTS: self.rows * (self.row_bytes // 2)}
        return Circuit("pous-setup", ((self.unit, self.blocks),), external=(f"{tid}/salt", f"{tid}/at", f"{tid}/slot"),
                       registered=Registered(params, tables=(store_table(self.blocks, self.w // 16),),
                                             reads=self.reads()))

    def registered(self, program: str) -> dict[str, RV.Registered]:
        """The store (`commit_store`) and W's rows, each committed once before the run under C's Program digest
        `program`, as the rows of 16-bit words its reading port holds."""
        if self.store is None or self.weights is None:
            raise ValueError("Setup.registered reads the store's blocks and W's rows")
        if len(self.store) != self.blocks or len(self.weights) != self.rows:
            raise ValueError(f"the setup reads {self.blocks} blocks and {self.rows} rows, not {len(self.store)} and "
                             f"{len(self.weights)}")
        return {STORE: commit_store(program, self.store),
                WEIGHTS: RV.Registered.commit(program, WEIGHTS, [np.frombuffer(b, "<u2").tolist() for b in self.weights],
                                              schema=V2)}

    def run(self, window: Window) -> None:
        """Every unit recorded: its block and row by reference, the salt, its place and its slot as 16-bit words, and its
        ok word."""
        if self.salt is None or self.store is None or self.weights is None:
            raise ValueError("Setup.run reads the salt, the store's blocks and W's rows")
        cases = [unit_inputs(self.w, self.salt, self.blocks_per_segment, self.store[u],
                             self.weights[place(self.w, self.row_bytes, u)[0]], u, self.row_bytes)
                 for u in range(self.blocks)]
        block, salt, at, row = (_bit_rows([c[k] for c in cases]) for k in range(4))
        slot = np.array([[c[4] >> i & 1 for i in range(SC.SLOT_BITS)] for c in cases], dtype=np.uint8)
        ok = EB.evaluate_bits(self.unit, block, salt, at, row, slot)
        words = lambda k: np.frombuffer(b"".join(c[k] for c in cases), dtype="<u2").reshape(self.blocks, -1)  # noqa: E731
        slots = np.array([[c[4]] for c in cases], dtype=np.uint16)
        oks = (ok.astype(np.uint16) << np.arange(16, dtype=np.uint16)).sum(axis=1, dtype=np.uint16)[:, None]
        window.record(self.unit, [None, words(1), words(2), None, slots], [oks])


def _bit_rows(rows: Sequence[bytes]) -> np.ndarray:
    """Byte strings as bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋, one row each."""
    return np.unpackbits(np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), -1), axis=1, bitorder="little")


def coin_bound(k: int, head: int, rounds: Sequence[str], opener: str, digest: str
               ) -> tuple[dict[str, int], tuple[Read, ...], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """The session's supplied inputs (PROTOCOL.md §5.4), as their parameters in 16-bit leaves, the reads that put them
    in the units' ports, their marks and their formats: round j's coin value ``pous.session.round_coin/v1`` (``round``
    j, 96 bits, ``u32be(j) ‖ u64be(i_j)``) under ``halves``, three 32-bit leaves (``j``, ``ihi``, ``ilo``), since the
    opening's key is a 32-bit leaf: round j's unit reads them whole as its ``j``, ``ihi`` and ``ilo`` and round j's
    opening reads ``ilo`` (16-bit leaves 4 and 5) as its key; and the signed header (``pous.session.header/v1`` at
    its ``head`` bytes) and context (``pous.session.context/v1``, 848 bits) under ``le``, which the digest's unit reads.
    ``rounds`` is the round templates' ids by slot.

    STAND-IN until `Registered.supplied` takes marks: `session_units` declares the parameters and reads and holds the
    marks back, so the values are registered rows the prover fills and the circuit proves nothing about the coins.
    Passing the marks as ``supplied`` is the switch."""
    params = {SESSION_HEADER: session_circuit.head_bytes(head) // 2, SESSION_CONTEXT: session_circuit.CTX_BYTES // 2}
    reads = [Read(digest, "head", SESSION_HEADER, 0, (1,), 0, (0,)), Read(digest, "ctx", SESSION_CONTEXT, 0, (1,), 0, (0,))]
    marks = {SESSION_HEADER: {"supplied": "pous.session.header/v1", "args": {"head": head}, "layout": "le",
                              "bits": 8 * session_circuit.head_bytes(head)},
             SESSION_CONTEXT: {"supplied": "pous.session.context/v1", "args": {}, "layout": "le",
                               "bits": 8 * session_circuit.CTX_BYTES}}
    for j in range(k):
        p = f"{ROUND_COIN}/{j}"
        params[p] = 6
        reads += [Read(rounds[j % 2], h, p, j // 2, (1,), 2 * t, (0,)) for t, h in enumerate(("j", "ihi", "ilo"))]
        reads.append(Read(opener, "tok", p, j, (1,), 4, (0,)))
        marks[p] = {"supplied": "pous.session.round_coin/v1", "args": {"round": j}, "layout": "halves", "bits": 96}
    formats = {f"{ROUND_COIN}/{j}": {"row_schema": V2, "word_bits": 32} for j in range(k)}
    return params, tuple(reads), marks, formats


def session_units(k: int, head: int, limit_ns: int, block_bytes: int, blocks: int, params: R.Params = R.DEPLOYED) -> Circuit:
    """A session of ``k`` rounds over a store of ``blocks`` blocks of ``block_bytes`` bytes, as units of the registered
    circuit, in the order its wires run (each writer before its readers):

        PousSessionDigest_v1{k, head, limit_ns}  1 unit    (head, ctx, the records rows) -> (ok, m, the records rows)
        PousSessionSig_v1{N, W, H}               1 unit    key: the registered key; m: the digest's m by wire
        EmbeddingRead_v2{blocks, block_bytes/2}  k units   round j's opening of the store's table at its index
        PousRound_v2{block_bytes, 0}             ⌈k/2⌉     round j = 2u: records row u by wire, row from opening j
        PousRound_v2{block_bytes, 1}             ⌊k/2⌋     round j = 2u + 1: the same, at the record's second slot

    The store (`store_table`, the setup's one) and the key are registered values; the coins' are `coin_bound`'s. An
    index past the store's blocks opens a row of 0x7FFF words with a verdict of 1 (`gates.embedding`), so the verifier
    holds every round's index below the store's blocks in the clear (`session_proofs.clear_reasons`)."""
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
    coins, reads, _marks, formats = coin_bound(k, head, [descriptor_id(t) for t, _ in slots], o, d)
    registered = Registered(
        {STORE: blocks * block_bytes // 2, ATTESTER_KEY: (16 + params.N) // 2, **coins},
        tables=(store_table(blocks, block_bytes // 2),),
        reads=(Read(g, "key", ATTESTER_KEY, 0, (1,), 0, (0,)), *reads),
        roots=(TableRead(o, "root", STORE_TABLE, "root", 0, k), TableRead(o, "domain", STORE_TABLE, "domain", 0, k)),
        openings=(TableOpening(o, STORE_TABLE, "tok", ("row", "salt", "path"), "o1", 0, k, row="o0"),), formats=formats)
    return Circuit("pous-session", ((dig, 1), (sig, 1), (opener, k), *slots), wires=tuple(wires), registered=registered)
