"""PoUS's prover's statements, handed to the proof system as any integration hands a window over
(`verity.commitments.periods.interface`): its circuit in proof blocks of PoUS's Definitions (`decode_circuit`,
`answer_circuit`, `verdict_circuit`), its values recorded into a `Window`, and the window's four records written for the
workers, which stage from them as they do from an adapter's.

`SessionIntegration` is a session's statements (the signature check, the message's hash and each round's opened block),
an `Integration`, its `circuit()` and its `run(window)`. `write_window` closes a window an integration ran and writes its
records. None is written yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.commitments.periods.interface import Circuit, Integration, Window, Wire

from . import answer_circuit, decode_circuit, session_circuit, verdict_circuit
from .session import SessionStatements, SessionUnit

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit, session_circuit)


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
