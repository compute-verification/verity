"""PoUS's prover's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.windows.interface`): its circuit in proof blocks of PoUS's Definitions (`decode_circuit`,
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
from verity.commitments.windows.interface import Circuit, Integration, Read, Registered, Window

from . import answer_circuit, decode_circuit, verdict_circuit
from . import setup_circuit as SP
from .setup_proof import payload_bytes, place, setup_unit, unit_inputs

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit)
#: the setup's registered parameters: the store's blocks and W's rows
STORE, WEIGHTS = "store", "weights"


class SetupIntegration:
    """PoUS's setup as an integration: one proof block per sampled block, the setup unit (`setup_proof.setup_unit`) over
    the block of the store, its base, the registered row of W that holds its payload and its slot (`setup_proof.place`).

    `w`, `c0` are P2's point, `row_bytes` W's rows' bytes N, `rows` how many, `blocks_per_segment` the store's segment
    length; `sample` is the challenger's sampled block indices, drawn after the store's commitment is registered;
    `prefix` is the salt's 128-byte expansion, and `store` and `weights` the store's blocks and W's rows, which only
    `run` reads."""

    def __init__(self, w: int, c0: int, row_bytes: int, rows: int, blocks_per_segment: int, sample: Sequence[int],
                 prefix: bytes | None = None, store: Sequence[bytes] | None = None,
                 weights: Sequence[bytes] | None = None) -> None:
        self.w, self.c0, self.row_bytes, self.rows = w, c0, row_bytes, rows
        self.blocks_per_segment, self.sample = blocks_per_segment, tuple(sample)
        self.prefix, self.store, self.weights = prefix, store, weights

    def circuit(self) -> Circuit:
        """The sampled blocks' units, each reading its block of the store's registered ``store`` and its row of the
        registered ``weights`` at public leaves (16-bit leaves: a block's W/16, a row's N/2). Its base and its slot are
        recorded inputs: the window has no row the verifier supplies a unit, so until it has one they are free rows."""
        unit = setup_unit(self.w, self.c0, self.row_bytes)
        tid, bw, rw = descriptor_id(unit), self.w // 16, self.row_bytes // 2
        n = SP.slots(self.row_bytes, payload_bytes(self.w))
        if any(not 0 <= u < self.rows * n for u in self.sample):
            raise ValueError(f"a sampled block is one of the store's {self.rows * n}")
        reads = tuple(r for i, u in enumerate(self.sample) for r in (
            Read(tid, "block", STORE, i, (1,), u * bw, (0,)),
            Read(tid, "row", WEIGHTS, i, (1,), place(self.w, self.row_bytes, u)[0] * rw, (0,))))
        return Circuit("pous-setup", ((unit, len(self.sample)),),
                       registered=Registered({STORE: self.rows * n * bw, WEIGHTS: self.rows * rw}, reads=reads))

    def run(self, window: Window) -> None:
        """Each sampled block's base and slot, and its unit's bit."""
        if self.prefix is None or self.store is None or self.weights is None:
            raise ValueError("SetupIntegration.run reads the salt's expansion, the store's blocks and W's rows")
        unit = setup_unit(self.w, self.c0, self.row_bytes)
        cases = [unit_inputs(self.w, self.prefix, self.blocks_per_segment, self.store[u],
                             self.weights[place(self.w, self.row_bytes, u)[0]], u, self.row_bytes) for u in self.sample]
        block, base, row = (_bit_rows([c[k] for c in cases]) for k in range(3))
        slot = np.array([[c[3] >> i & 1 for i in range(SP.SLOT_BITS)] for c in cases], dtype=np.uint8)
        ok = EB.evaluate_bits(unit, block, base, row, slot)
        window.record(unit, [None, np.frombuffer(b"".join(c[1] for c in cases), dtype="<u2").reshape(len(cases), -1),
                             None, np.array([c[3] for c in cases], dtype=np.uint16)], [ok[:, 0].astype(np.uint16)])


def _bit_rows(rows: Sequence[bytes]) -> np.ndarray:
    """Byte strings as bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋, one row each."""
    return np.unpackbits(np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), -1), axis=1, bitorder="little")


class SessionIntegration:
    """PoUS's session as an integration: the signature check, the message's hash and one opened block a round, each
    its own proof block."""

    def circuit(self) -> Circuit:
        raise NotImplementedError("proofs/zero-knowledge: PoUS's session as a window's circuit "
                                  "(SessionIntegration.circuit)")

    def run(self, window: Window) -> None:
        raise NotImplementedError("proofs/zero-knowledge: PoUS's session's values recorded into a window "
                                  "(SessionIntegration.run)")


def write_window(integration: Integration, out_dir: Path, registrar: Any) -> Path:
    """The window `integration` runs, closed and written under `out_dir` (`Window.close`, `Window.write`, its
    registration record made by `registrar`): the four records the workers stage from. Returns the window's
    directory."""
    raise NotImplementedError("proofs/zero-knowledge: PoUS's window records for the workers (write_window)")
