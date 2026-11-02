"""A served window's binding, which is public: the SHA-512, under `BINDING_TAG`, of the served model, its revision, the
window and the SHA-512 of the serve dir's weights registration (None without one). PoUS's piece (`served_piece.py`)
binds its hiding commitment to it and the memory challenger logs the window's salt under it, so it is the audit the
bandwidth party holds that window's PoUS store by (the gate's ``bw_party --pous-audit AUDIT``, `p2_party.pous_record`).
`of_serve_dir` derives it from the serve dir as `served_piece.run` does, which
`benchmarks/one_stage/served_bandwidth.py collect` does too, to credit a bandwidth answer only by the party's own answer
of a session under the window's own audit, as the auditor's host kept it. It imports no torch, so the collector loads it
by its path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.ml.commitments.identity import canonical_json_bytes

BINDING_TAG = b"verity/pous/served-window/v1"


def binding(model: str, revision: str, window: int, registration: str | None) -> bytes:
    """What the answer's hiding commitment is bound to: the served model, the window and its weights registration."""
    doc = {"model": model, "registration_sha512": registration, "revision": revision, "window": window}
    return hashlib.sha512(BINDING_TAG + canonical_json_bytes(doc)).digest()


def registration_sha512(serve_dir: Path) -> str | None:
    """The SHA-512 of the serve dir's ``weights_registration.json``, None when it has none."""
    path = serve_dir / "weights_registration.json"
    return hashlib.sha512(path.read_bytes()).hexdigest() if path.is_file() else None


def of_serve_dir(serve_dir: Path, window: int) -> bytes:
    """Window `window`'s binding: ``serve.json``'s model and revision and the weights registration's SHA-512;
    ValueError when ``serve.json`` can't be read or names no model and revision."""
    try:
        serve = json.loads((serve_dir / "serve.json").read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"{serve_dir} has no serve.json to read ({type(e).__name__})") from None
    model, revision = (serve.get(k) if isinstance(serve, dict) else None for k in ("model", "revision"))
    if not (isinstance(model, str) and isinstance(revision, str)):
        raise ValueError(f"{serve_dir}'s serve.json names no model and revision")
    return binding(model, revision, window, registration_sha512(serve_dir))


__all__ = ["BINDING_TAG", "binding", "of_serve_dir", "registration_sha512"]
