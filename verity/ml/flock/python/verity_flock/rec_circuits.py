"""V*'s circuit texts kept per shape, so a statement's circuit is built once per inner shape, not once per session.

A V* statement's circuit text, partition and program are decided by its template, its shape and its instance count alone
(`rec_residuals.statement`: "which only n, S and holo decide"; a level's by LANES, H, n and its program's version); only
its instance files carry the session (rows, fresh salts). With ``$REC_CIRCUIT_CACHE`` set to a directory, a statement's
circuit text is kept there under a key of what decides it and of this code (`code_digest`: every source the text is built
from), and read back instead of lowering, laying out and composing it again. Unset: nothing is kept, as before.

On a hit the statement has no lowering, so its stager skips its own replay of the laid-out unit (``evaluate``): that replay
checks the stager against itself once per shape, when the text is built; every session's witness is still computed and
checked by the prover, and its proof by the verifier, from the same text.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path

ENV = "REC_CIRCUIT_CACHE"


@cache
def code_digest() -> str:
    """SHA-256 over every source a V* circuit text is built from: this package's modules, `verity.ml.circuits` and the
    partition and program code of sampled proofs. Any change to them is a new key, so no text outlives the code that wrote it."""
    import verity.ml.circuits as C
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    h = hashlib.sha256()
    for root in (Path(__file__).parent, Path(C.__file__).parent):
        for p in sorted(root.rglob("*.py")):
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    h.update(Path(P.__file__).read_bytes())
    return h.hexdigest()


def _dir(kind: str, key: dict) -> Path | None:
    root = os.environ.get(ENV)
    if not root:
        return None
    k = hashlib.sha256(json.dumps({"kind": kind, "key": key, "code": code_digest()}, sort_keys=True).encode()).hexdigest()
    return Path(root) / f"{kind.replace('/', '-')}-{k[:32]}"


def load(kind: str, key: dict) -> tuple[str, dict] | None:
    """The kept circuit text of ``kind`` at ``key`` and what was kept beside it, or None (no cache, or not kept yet)."""
    d = _dir(kind, key)
    if d is None or not (d / "info.json").is_file():
        return None
    return (d / "circuit.txt").read_text(), json.loads((d / "info.json").read_text())


def store(kind: str, key: dict, text: str, info: dict) -> None:
    """Keep ``text`` and ``info`` for ``kind`` at ``key``. Written aside and renamed into place, so a reader sees all of it
    or nothing, and two stagers of one shape at once each write a whole copy and one of them wins."""
    d = _dir(kind, key)
    if d is None or (d / "info.json").is_file():
        return
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = d.parent / f".{d.name}.{os.getpid()}"
    tmp.mkdir()
    (tmp / "circuit.txt").write_text(text)
    (tmp / "info.json").write_text(json.dumps({**info, "kind": kind, "key": key}, sort_keys=True))
    try:
        tmp.rename(d)
    except OSError:
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
