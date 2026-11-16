"""PoUS's prover's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.periods.interface`): its circuit in proof blocks of PoUS's Definitions (`decode_circuit`,
`answer_circuit`, `verdict_circuit`), its values recorded into a `Window`, and the window's four records written for the
workers, which stage from them as they do from an adapter's.

`SetupIntegration` is the setup's statements (each a sampled block's decode against the store's registered row) and
`SessionIntegration` a session's (the signature check, the message's hash and each round's opened block); each is an
`Integration`, its `circuit()` and its `run(window)`. `write_window` closes a window an integration ran and writes its
records. None is written yet.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import bits as EB
from verity.commitments.periods.interface import Circuit, Integration, Read, Registered, Window, Wire
from verity.protocols.resource_accounting.memory.pous.verifier.setup_proof import BASE, OK, SLOT, base, leaf_bytes

from . import answer_circuit, decode_circuit, session_circuit, verdict_circuit
from . import setup_circuit as SP
from .decode_circuit import P2_BASE_BYTES
from .session import SessionStatements, SessionUnit
from .setup_proof import payload_bytes, place, setup_unit, unit_inputs

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit, session_circuit)
#: the setup's registered parameters: the store's blocks and W's rows
STORE, WEIGHTS = "store", "weights"


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
            def rows(values):
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
