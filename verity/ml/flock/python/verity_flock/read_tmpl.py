"""The table read's unit template as the verifier builds it: ``flock-read-tmpl V H``'s text (`Flock.ReadNet.readTmpl V H`, the
one template `Flock.ReadNet.check` accepts), parsed (`tmpl.parse`). A structural read unit is staged on it in place of
`struct_lower`'s lowering of the Python Definition, whose Definitions and columns the check refuses
(`class_statement.unit_lowering`).

``FLOCK_READ_TMPL`` names the executable (``lake build flock-read-tmpl`` in ``verity/core/service/flock``), else
``flock-read-tmpl`` on the path."""
from __future__ import annotations

import functools
import os
import subprocess
from typing import Any

from . import tmpl as TM

#: the read's Definition, `verity.ml.commitments.gates.embedding.EmbeddingRead_v1{V, H}`, by its base id
READ = "EmbeddingRead_v1"


@functools.cache
def template(V: int, H: int) -> TM.Tmpl:
    """The read of a table of V rows of H 16-bit words, as ``flock-read-tmpl V H`` writes it."""
    exe = os.environ.get("FLOCK_READ_TMPL") or "flock-read-tmpl"
    try:
        r = subprocess.run([exe, str(V), str(H)], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"{exe}: no such executable: build it with `lake build flock-read-tmpl` in "
                                "verity/core/service/flock and name it in FLOCK_READ_TMPL") from None
    if r.returncode:
        raise ValueError(f"{exe} {V} {H}: {r.stderr.strip() or f'exit {r.returncode}'}")
    return TM.parse(r.stdout)


def of(definition: Any) -> TM.Tmpl | None:
    """The verifier's template of ``definition`` where it builds one, a bound `READ` (`template`); else None."""
    if getattr(definition, "base_id", None) != READ:
        return None
    S = definition.statics
    return template(int(S["V"]), int(S["H"]))
