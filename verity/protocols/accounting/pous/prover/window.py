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

from verity.commitments.windows.interface import Circuit, Integration, Window

from . import answer_circuit, decode_circuit, verdict_circuit
from .setup_proof import GOAL

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit)


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
        raise NotImplementedError(f"{GOAL}: PoUS's setup as a window's circuit (SetupIntegration.circuit)")

    def run(self, window: Window) -> None:
        raise NotImplementedError(f"{GOAL}: PoUS's setup's values recorded into a window (SetupIntegration.run)")


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
