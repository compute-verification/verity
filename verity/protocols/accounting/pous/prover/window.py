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

from pathlib import Path
from typing import Any

from verity.commitments.windows.interface import Circuit, Integration, Window

from . import answer_circuit, decode_circuit, verdict_circuit

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit)


class SetupIntegration:
    """PoUS's setup as an integration: one proof block per sampled block's decode, compared with the store's registered
    row at its slot."""

    def circuit(self) -> Circuit:
        raise NotImplementedError("proofs/zero-knowledge: PoUS's setup as a window's circuit "
                                  "(SetupIntegration.circuit)")

    def run(self, window: Window) -> None:
        raise NotImplementedError("proofs/zero-knowledge: PoUS's setup's values recorded into a window "
                                  "(SetupIntegration.run)")


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
