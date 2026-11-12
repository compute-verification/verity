"""The scan's ok bits at public content: each scan member's ``o0`` root is a constant of the schedule's T.

A proof block of the network certifier's scan outputs one public word, ``o0``, its ok bit: 1 when every check of its
bucket held. The bit is worth what binds it. A member's ``o0`` rows are committed under public salts (`salt`), each
row's by its index in the member's tree, so the tree of a member whose every proof block is ok is one anyone computes:
its n rows `OK_ROW` under salts 0 … n − 1, at the member's ``o0`` frame domain (`binding`), which is its template's
and its proof blocks' count's and names no run, round or execution phase. One execution phase's scan has the members
`members` gives, a function of the schedule's T alone, so the roots are constants of T (`constants`).

The period check (its rows form's `okChecks`; Lean `NetworkCheck.ScanOk`, which this mirrors) computes them
once a period, from T, and holds every execution phase's scan's registration to them by comparison alone (`held`): its
``roots`` name each member's ``<member>/o0`` at the constant's root, leaves and domain, and no other port ending
``/o0``. A member's statement reads its ``o0`` rows at the root its registration names, so a proof block's ok word
opens a row `OK_ROW` or SHA-512 collides: the bit is the verifier's own computation from public parameters, never a
row the prover's registered file states.

The salts hide nothing and need not: an ok row's content is the one the verifier accepts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from verity.commitments import advice as ADV
from verity.commitments import domains, hm96
from verity.commitments.identity import identity_digest_sha512
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.commitments.rowleaf import bit_row

__all__ = ["OK_ROW", "PORT", "SALT_BYTES", "SALT_TAG", "WORD_BITS", "Constant", "binding", "committed", "constants",
           "held", "known", "members", "ok_port", "salt", "set_of"]

#: the ok row the verifier accepts: a proof block's one output bit, 1, in its one word
OK_ROW = (1,)
WORD_BITS = 16
#: a scan proof block's public word, its first output
PORT = "o0"
SALT_BYTES = hm96.SHA512.salt_bytes
#: the ok rows' hm96 salt tag
SALT_TAG = domains.CERTIFIER_SCAN_OK_ROW_SALT.encode() + b"\0"


def salt(position: int) -> bytes:
    """Ok row `position`'s hm96 salt, its index in its member's tree and nothing of the link, the execution phase or
    the registration: `SALT_BYTES` bytes of SHA-512 in counter mode (``SHA-512(m ‖ u32be c)`` for c < 3, each output
    whole) over m = `SALT_TAG` ‖ u64be(position)."""
    head = SALT_TAG + int(position).to_bytes(8, "big")
    return b"".join(hashlib.sha512(head + c.to_bytes(4, "big")).digest() for c in range(SALT_BYTES // 64))


def known(n: int) -> dict[str, list]:
    """A member's ``o0`` of `n` proof blocks as known content (Flock's registered reads' ``known``): `n` rows `OK_ROW`,
    row i under `salt` i."""
    return {"rows": [list(OK_ROW)] * int(n), "salts": [salt(i).hex() for i in range(int(n))]}


def set_of(template: str) -> str:
    """A scan template's statements' input set, which their frame domains bind (`binding`): the template's alone,
    naming no round and no execution phase."""
    return f"unit-scan-{template}"


def binding(template: str, n: int) -> bytes:
    """The binding of the ``o0`` frame domain of a statement over `n` proof blocks of `template` (Flock's statement's,
    at instances 0 … n of the input set `set_of`): a tree under it of the rows' leaves is the statement's ``o0``."""
    return identity_digest_sha512(domains.FLOCK_CIRCUIT_BINDING, {"set": set_of(template), "lo": 0, "hi": int(n),
                                                                  "port": PORT, "schema": SCHEMA})


def members(T: int) -> tuple[tuple[str, str, int], ...]:
    """One execution phase's scan's members, (name, template, proof blocks), in its Program's order: bucket 0's
    ``first``; ``ingress-0``, buckets 1 … T − 2, and ``last``, bucket T − 1, whose next state no state leaf holds,
    so it is a member of its own; and ``egress-0``, the T egress buckets. 2T proof blocks."""
    T = int(T)
    return (("first", "first", 1), *((("ingress-0", "ingress", T - 2),) if T >= 3 else ()),
            *((("last", "ingress", 1),) if T >= 2 else ()), ("egress-0", "egress", T))


def committed(template: str, n: int) -> ADV.Committed:
    """The ``o0`` tree of a member of `n` proof blocks of `template`, every one ok."""
    row = bit_row((OK_ROW, WORD_BITS))[0]
    return ADV.commit([row] * int(n), WORD_BITS, binding(template, n), salts=[salt(i) for i in range(int(n))])


@dataclass(frozen=True)
class Constant:
    """A member's ``o0`` as a registration's ``roots`` names it when every proof block of the member is ok."""

    port: str
    leaves: int
    domain: str
    root: str


def constants(T: int) -> tuple[Constant, ...]:
    """Each member's constant at the schedule's `T` (`members`), in order: what one period check computes once."""
    out = []
    for name, template, n in members(T):
        tree = committed(template, n)
        out.append(Constant(f"{name}/{PORT}", n, ADV.domain(tree.binding, n).domain_id.hex(), tree.root.hex()))
    return tuple(out)


def ok_port(port: object) -> bool:
    """Whether a ``roots`` entry's port is a proof block's public word: it ends ``/o0``."""
    return isinstance(port, str) and port.endswith("/" + PORT)


def held(consts: Sequence[Constant], registration: dict) -> bool:
    """Whether a scan's registration names its ok bits at the constants: every constant's port is in its ``roots``,
    and every ``roots`` entry whose port ends ``/o0`` is at its port's constant, in leaves, domain and root. It
    compares and hashes nothing."""
    roots = registration.get("roots")
    entries = [e for e in roots if isinstance(e, dict)] if isinstance(roots, list) else []
    ports = [e.get("port") for e in entries]
    return all(c.port in ports for c in consts) and all(
        any(c.port == e["port"] and type(e.get("leaves")) is int and e["leaves"] == c.leaves
            and e.get("domain") == c.domain and e.get("root") == c.root for c in consts)
        for e in entries if ok_port(e.get("port")))
