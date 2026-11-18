"""PoUS's prover's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.periods.interface`): its circuit in proof blocks of PoUS's Definitions (`decode_circuit`,
`answer_circuit`, `verdict_circuit`), its values recorded into a `Window`, and the window's four records written for the
workers, which stage from them as they do from an adapter's.

`Setup` is the setup's one circuit C (the setup unit over every block of the store) and `SessionIntegration` a session's
statements (the signature check, the message's hash and each round's opened block); each is an `Integration`, its
`circuit()` and its `run(window)`. `write_window` closes a window an integration ran and writes its records. None is
written yet.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import bits as EB
from verity.commitments import registered_value as RV
from verity.commitments.periods.interface import Circuit, Integration, Read, Registered, Window, Wire

from . import answer_circuit, decode_circuit, session_circuit, verdict_circuit
from . import setup_circuit as SC
from .session import SessionStatements, SessionUnit
from .setup_proof import payload_bytes, place, setup_unit, unit_inputs

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit, session_circuit)
#: the setup's registered parameters: the store's blocks and W's rows
STORE, WEIGHTS = "store", "weights"
#: the units of a proof block of the setup's C: one store block each
CAP = 1


class Setup:
    """PoUS's setup as one circuit C: the setup unit (`setup_proof.setup_unit`, ``PousP2SetupCheck_v1``) over every block
    of the store in order, unit u block u, one proof block a store block (`CAP`, `interface.partition_blocks`). Each unit
    reads its block of the registered ``store`` and the registered row of ``weights`` that holds its payload (`reads`),
    and takes its base and its slot (`setup_proof.place`) as C's input nodes, committed once at C's input, their writer;
    its ok output is C's. Which blocks are proved is the proof system's sample on its live coins after the store is
    committed, and which units a block holds the partition's: the setup lists neither.

    The base is a function of the verifier's salt and the slot of the unit's position, and nothing in C binds them yet:
    the verifier refuses there (`setup_proof.bound`) until circuits' coin-bound input kind does.

    `w`, `c0` are P2's point, `row_bytes` W's rows' bytes N, `rows` how many, `blocks_per_segment` the store's segment
    length; `prefix` is the salt's 128-byte expansion and `store` and `weights` the store's blocks and W's rows, which
    only `registered` and `run` read."""

    def __init__(self, w: int, c0: int, row_bytes: int, rows: int, blocks_per_segment: int, prefix: bytes | None = None,
                 store: Sequence[bytes] | None = None, weights: Sequence[bytes] | None = None) -> None:
        self.w, self.c0, self.row_bytes, self.rows = w, c0, row_bytes, rows
        self.blocks_per_segment, self.prefix, self.store, self.weights = blocks_per_segment, prefix, store, weights
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
        return Circuit("pous-setup", ((self.unit, self.blocks),), external=(f"{tid}/base", f"{tid}/slot"),
                       registered=Registered(params, reads=self.reads()))

    def registered(self, program: str) -> dict[str, RV.Registered]:
        """The store's blocks and W's rows, each committed once before the run under C's Program digest `program`, as
        the rows of 16-bit words its reading port holds."""
        if self.store is None or self.weights is None:
            raise ValueError("Setup.registered reads the store's blocks and W's rows")
        if len(self.store) != self.blocks or len(self.weights) != self.rows:
            raise ValueError(f"the setup reads {self.blocks} blocks and {self.rows} rows, not {len(self.store)} and "
                             f"{len(self.weights)}")
        return {p: RV.Registered.commit(program, p, [np.frombuffer(b, "<u2").tolist() for b in rows],
                                        schema=RV.SCHEMA_HM96_SHA512_ROW_V2)
                for p, rows in ((STORE, self.store), (WEIGHTS, self.weights))}

    def run(self, window: Window) -> None:
        """Every unit recorded: its block and row by reference, its base and slot as 16-bit words, and its bit."""
        if self.prefix is None or self.store is None or self.weights is None:
            raise ValueError("Setup.run reads the salt's expansion, the store's blocks and W's rows")
        cases = [unit_inputs(self.w, self.prefix, self.blocks_per_segment, self.store[u],
                             self.weights[place(self.w, self.row_bytes, u)[0]], u, self.row_bytes)
                 for u in range(self.blocks)]
        block, base, row = (_bit_rows([c[k] for c in cases]) for k in range(3))
        slot = np.array([[c[3] >> i & 1 for i in range(SC.SLOT_BITS)] for c in cases], dtype=np.uint8)
        ok = EB.evaluate_bits(self.unit, block, base, row, slot)
        bases = np.frombuffer(b"".join(c[1] for c in cases), dtype="<u2").reshape(self.blocks, -1)
        slots = np.array([[c[3]] for c in cases], dtype=np.uint16)
        window.record(self.unit, [None, bases, None, slots], [ok[:, 0].astype(np.uint16)])


def _bit_rows(rows: Sequence[bytes]) -> np.ndarray:
    """Byte strings as bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋, one row each."""
    return np.unpackbits(np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), -1), axis=1, bitorder="little")


class SessionIntegration:
    """PoUS's session as an integration (`session.session_statements`): the signature check (`PousSessionSig_v1`, one
    unit), the message's hash (`PousSessionDigest_v1`, one unit) and one opened block a round (`PousRound_v1` at slot 0
    for the even rounds and at slot 1 for the odd), each its own proof block. Its wires are the session's links: the
    signature's ``m`` reads the digest's output m (``o1``), and round j's ``records`` the digest's records output ⌊j/2⌋
    (``o{2 + ⌊j/2⌋}``). The rows the verifier knows (the key, the header, the signed context, j and i) and
    each round's registered row are free rows here, which the session's tables are to fill."""

    def __init__(self, statements: SessionStatements) -> None:
        self.statements = statements
        rounds = statements.rounds
        self._parity = tuple(tuple(u for j, u in enumerate(rounds) if j % 2 == s) for s in (0, 1))

    def _templates(self) -> list[tuple[SessionUnit, ...]]:
        st = self.statements
        return [g for g in ((st.signature,), (st.digest,), *self._parity) if g]

    def circuit(self) -> Circuit:
        groups = self._templates()
        sig, dig = (descriptor_id(g[0].definition) for g in groups[:2])
        wires = [Wire(sig, "m", dig, "o1", 0, (1,), 0, (0,))]
        for group in self._parity:
            wires += [Wire(descriptor_id(group[0].definition), "records", dig, f"o{2 + u}", u, (1,), 0, (0,))
                      for u in range(len(group))]
        return Circuit("pous-session", tuple((g[0].definition, len(g)) for g in groups), wires=tuple(wires))

    def run(self, window: Window) -> None:
        """Each unit's rows as bytes (a port's words, `octets`) and its outputs: the one-bit ok as a 16-bit integer, m and
        the records rows as bytes."""
        for group in self._templates():
            def rows(values, group=group):
                return np.frombuffer(b"".join(values), dtype=np.uint8).reshape(len(group), -1)
            inputs = [rows([u.rows[p] for u in group]) for p in range(len(group[0].rows))]
            outputs = [np.array([u.outputs[0][0] for u in group], dtype=np.uint16),
                       *(rows([u.outputs[o] for u in group]) for o in range(1, len(group[0].outputs)))]
            window.record(group[0].definition, inputs, outputs)


def write_window(integration: Integration, out_dir: Path, registrar: Any) -> Path:
    """The window `integration` runs, closed and written under `out_dir` (`Window.close`, `Window.write`, its
    registration record made by `registrar`): the four records the workers stage from. Returns the window's
    directory."""
    raise NotImplementedError("proofs/zero-knowledge: PoUS's window records for the workers (write_window)")
