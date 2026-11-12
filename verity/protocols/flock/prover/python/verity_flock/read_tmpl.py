"""The table read's unit template as the verifier builds it: ``flock-read-tmpl V H``'s text (`Flock.ReadNet.readTmpl V H`, the
one template `Flock.ReadNet.check` accepts), parsed (`tmpl.parse`). A structural read unit is staged on it in place of
`struct_lower`'s lowering of the Python Definition, whose Definitions and columns the check refuses
(`class_statement.unit_lowering`).

`check` asks the same executable whether a circuit file holds that template (``flock-read-tmpl check V H CIRCUIT``).

The executable is `EXE` where a caller sets it, else this tree's build (`BUILT`, where ``lake build flock-read-tmpl`` in
``verity/protocols/flock/verifier`` puts it), else ``flock-read-tmpl`` on the path (`executable`)."""
from __future__ import annotations

import functools
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import tmpl as TM

#: the read's Definition, `verity.commitments.gates.embedding.EmbeddingRead_v1{V, H}`, by its base id
READ = "EmbeddingRead_v1"
#: the executable a caller names (a test's stand-in, a job's own build); None: `BUILT`, else the path's
EXE: str | None = None
#: where ``lake build flock-read-tmpl`` puts the executable in this tree's Flock package
BUILT = Path(__file__).resolve().parents[6] / "verity/protocols/flock/verifier/.lake/build/bin/flock-read-tmpl"


def executable() -> str:
    """The verifier's ``flock-read-tmpl``: `EXE`, else `BUILT` where it is built, else the one on the path."""
    exe = EXE or (str(BUILT) if BUILT.is_file() else shutil.which("flock-read-tmpl"))
    if not exe or not Path(exe).is_file():
        raise FileNotFoundError(f"{exe or 'flock-read-tmpl'}: no such executable: build it with `lake build flock-read-tmpl` in "
                                "verity/protocols/flock/verifier")
    return exe


def digest() -> str:
    """SHA-512 of `executable`'s bytes, which a cache of statements staged on its templates keys on."""
    return hashlib.sha512(Path(executable()).read_bytes()).hexdigest()


def is_read(definition: Any) -> bool:
    """``definition`` is a bound `READ`, the Definition the verifier builds a template of."""
    return getattr(definition, "base_id", None) == READ


@functools.cache
def template(V: int, H: int) -> TM.Tmpl:
    """The read of a table of V rows of H 16-bit words, as ``flock-read-tmpl V H`` writes it."""
    exe = executable()
    r = subprocess.run([exe, str(V), str(H)], capture_output=True, text=True, check=False)
    if r.returncode:
        raise ValueError(f"{exe} {V} {H}: {r.stderr.strip() or f'exit {r.returncode}'}")
    return TM.parse(r.stdout)


def check(V: int, H: int, circuit: Path) -> str | None:
    """What ``flock-read-tmpl check V H CIRCUIT`` decides of the circuit file ``circuit`` (`Flock.ReadTmplCmd.premises`):
    None where it accepts the file as a templated statement whose unit's template passes `Flock.ReadNet.check V H` and
    whose outputs are its unit's; else its reason, the first of those that fails. A usage error or an unreadable file
    raises."""
    exe = executable()
    r = subprocess.run([exe, "check", str(V), str(H), str(circuit)], capture_output=True, text=True, check=False)
    if r.returncode == 0:
        return None
    why = r.stderr.strip() or f"exit {r.returncode}"
    if r.returncode != 1:
        raise ValueError(f"{exe} check {V} {H} {circuit}: {why}")
    return why


def of(definition: Any) -> TM.Tmpl | None:
    """The verifier's template of ``definition`` where it builds one, a bound `READ` (`template`); else None."""
    if not is_read(definition):
        return None
    S = definition.statics
    return template(int(S["V"]), int(S["H"]))
