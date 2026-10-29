"""The proof's identifier v1 (``PROTOCOL.md``, "The proof's identifier (v1)"): the JSON object in a C-Flock proof's header
that names the oracle, the salt and the workload the proof is for.

``{"format": "verity/pouw/identifier/v1", "scheme": <the scheme's registered name>, "salt": <the epoch's salt, lowercase
hex>, "workload": <lowercase hex SHA-512 of the declared workload's canonical bytes>}``.  The scheme names the oracle H, so
there is no seed digest.  The proof binds the object through ``pub.sha``, which seeds its Fiat-Shamir transcript.  The
verifier recomputes the bytes from the epoch it audits (`identifier_bytes`), and C-Flock's verifier compares the proof's
object with them byte for byte (``--identifier F``).

Canonical bytes are RFC 8785 (`verity.ml.circuits.codec.canonical_json`), on the domain where that function is RFC 8785: ASCII
strings and integers of magnitude at most 2^53 − 1.  Anything outside it is refused.
"""

from __future__ import annotations

import hashlib
from typing import Any

from verity.ml.circuits.codec import canonical_json, jcs_domain_error
from verity.ml.commitments import domains

from .audit import Epoch
from .protocol import DomainError, PoUWScheme
from .schemes import SCHEMES

FORMAT = domains.POUW_IDENTIFIER
SALT_BYTES = 32
#: a certified epoch's salt (`audit.Epoch.certified`)
CERTIFIED_SALT_BYTES = 64


def workload_json(epoch: Epoch) -> dict[str, Any]:
    """The declared workload: every matmul in declaration order, then the weights root, which binds every weight's row root.
    The index and the weight id are 64-bit, so they are decimal strings: RFC 8785 writes a number as an IEEE 754 double,
    which is exact only up to 2^53."""
    return {"matmuls": [{"index": str(u.index), "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": str(u.weight)}
                        for u in epoch.workload.matmuls],
            "weights_root": epoch.weights_root.hex()}


def _canonical(obj: Any) -> bytes:
    err = jcs_domain_error(obj)
    if err is not None:
        raise DomainError(f"the identifier's JSON is outside RFC 8785's domain: {err}")
    return canonical_json(obj)


def workload_bytes(epoch: Epoch) -> bytes:
    """The declared workload's canonical bytes."""
    return _canonical(workload_json(epoch))


def workload_digest(epoch: Epoch) -> bytes:
    """SHA-512 of the declared workload's canonical bytes."""
    return hashlib.sha512(workload_bytes(epoch)).digest()


def _name(scheme: PoUWScheme | str) -> str:
    name = scheme if isinstance(scheme, str) else getattr(scheme, "name", None)
    if not isinstance(name, str) or name not in SCHEMES:
        raise DomainError(f"no PoUW scheme {name!r}; registered: {', '.join(SCHEMES)}")
    return name


def identifier_v1(scheme: PoUWScheme | str, epoch: Epoch) -> dict[str, str]:
    """The identifier of a proof for `epoch` under `scheme` (a registered name, or a scheme whose ``name`` is one)."""
    name = _name(scheme)
    if not isinstance(epoch.salt, bytes) or len(epoch.salt) not in (SALT_BYTES, CERTIFIED_SALT_BYTES):
        raise DomainError(f"the salt is {SALT_BYTES} bytes, or a certified epoch's {CERTIFIED_SALT_BYTES}")
    return {"format": FORMAT, "scheme": name, "salt": epoch.salt.hex(), "workload": workload_digest(epoch).hex()}


def identifier_bytes(scheme: PoUWScheme | str, epoch: Epoch) -> bytes:
    """The identifier's canonical bytes: what the verifier hands C-Flock's verifier as ``--identifier F``."""
    return _canonical(identifier_v1(scheme, epoch))


__all__ = ["CERTIFIED_SALT_BYTES", "FORMAT", "SALT_BYTES", "identifier_bytes", "identifier_v1", "workload_bytes", "workload_digest", "workload_json"]
