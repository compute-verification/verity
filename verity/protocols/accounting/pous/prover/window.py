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

import numpy as np

from verity.circuits.codec import descriptor_id
from verity.commitments.windows.interface import Circuit, Integration, Window, Wire

from . import answer_circuit, decode_circuit, session_circuit, verdict_circuit
from .session import SessionStatements, SessionUnit

DEFINITIONS = (decode_circuit, answer_circuit, verdict_circuit, session_circuit)


class SetupIntegration:
    """PoUS's setup as an integration: one proof block per sampled block's decode, compared with the store's registered
    row at its slot."""

    def circuit(self) -> Circuit:
        raise NotImplementedError("proofs/zero-knowledge: PoUS's setup as a window's circuit "
                                  "(SetupIntegration.circuit)")

    def run(self, window: Window) -> None:
        raise NotImplementedError("proofs/zero-knowledge: PoUS's setup's values recorded into a window "
                                  "(SetupIntegration.run)")


def _bits(data: bytes, n: int) -> list[int]:
    return [(data[t // 8] >> (t % 8)) & 1 for t in range(n)]


class SessionIntegration:
    """PoUS's session as an integration (`session.session_statements`): the signature check (`PousSessionSig_v1`, one
    unit), the message's hash (`PousSessionDigest_v1`, one unit) and one opened block a round (`PousRound_v1` at slot 0
    for the even rounds and at slot 1 for the odd), each its own proof block. Its wires are the session's links: the
    signature's ``m`` reads the digest's output m (``o1``), and round j's ``records`` the digest's records output ⌊j/2⌋
    (``o{2 + ⌊j/2⌋}``). The rows the verifier knows (the key, the header, the domain id, the signed context, j and i) and
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
        for group in self._templates():
            params, ret = group[0].definition.signature()
            fields = getattr(ret, "fields", None) or (ret,)
            inputs = [np.array([_bits(u.rows[p], ty.leaves) for u in group], dtype=np.uint8)
                      for p, (_, ty) in enumerate(params)]
            outputs = [np.array([_bits(u.outputs[o], ty.leaves) for u in group], dtype=np.uint8)
                       for o, ty in enumerate(fields)]
            window.record(group[0].definition, inputs, outputs)


def write_window(integration: Integration, out_dir: Path, registrar: Any) -> Path:
    """The window `integration` runs, closed and written under `out_dir` (`Window.close`, `Window.write`, its
    registration record made by `registrar`): the four records the workers stage from. Returns the window's
    directory."""
    raise NotImplementedError("proofs/zero-knowledge: PoUS's window records for the workers (write_window)")
