"""The Python side of the memory challenger's difftests: `memory_challenger.py` (the reference, unchanged) behind the Lean
port's JSON formats, and the sessions the tests draw.

The decoders mirror `MemoryChallenger.Protocol`'s: byte strings lowercase hex, integers naturals, and where the protocol
decodes its inputs (`params`, `commitment`, `evidence`) each input exactly the canonical JSON of what it decodes to. Past
decoding every reference is the Python's own function on the decoded objects.

Sessions are run by `memory_challenger.MemoryChallenger` itself, on an injected clock, and signed by a test signer's key
(`pous_test_signer`'s derivation) at a leaf the test picks (`AtLeaf`), or by a deployed-parameter key whose tree is sparse
(`deployed_key`, as `test_pous_memory_challenger._SparseTree`). No real key is read: every key here is derived from a
constant in this file."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import protocol as P
from verity.core.protocols.pous import memory_challenger as mc
from verity.ml.crypto import lms

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = HERE.parents[5]


def _test_signer():
    spec = importlib.util.spec_from_file_location(
        "pous_test_signer", ROOT / "verity/core/protocols/pous/tests/pous_test_signer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts = _test_signer()
TEST_ID = mc.TEST_PREFIX + mc.PROTOCOL
SCHEMES_ALL = {**mc.SCHEMES, **ts.SCHEMES}


# --- decoding, as `MemoryChallenger.Protocol` and `.Difftest` decode -------------------------------------------------------

def _get(j: Any, k: str) -> Any:
    if not isinstance(j, dict) or k not in j:
        raise D.Reject(f"{k} missing")
    return j[k]


def _str(j: Any, k: str) -> str:
    v = _get(j, k)
    if not isinstance(v, str):
        raise D.Reject(f"{k}: not a string")
    return v


def hex_(j: Any, k: str) -> bytes:
    return P.unhex(_str(j, k))


def nat(j: Any, k: str) -> int:
    _get(j, k)
    return P.nat(j, k)


def list_(j: Any, k: str) -> list:
    v = _get(j, k)
    if not isinstance(v, list):
        raise D.Reject(f"{k}: not an array")
    return v


def digests(j: Any, k: str) -> dict[int, bytes]:
    """`MemoryChallenger.Digests.ofJson`: an object of hex digests keyed by label index, decimal with no leading zero."""
    v = _get(j, k)
    if not isinstance(v, dict):
        raise D.Reject(f"{k}: not an object")
    out = {}
    for key, h in v.items():
        if not (key.isascii() and key.isdigit() and str(int(key)) == key):
            raise D.Reject(f"{k}: the key {key} is not a block index")
        if not isinstance(h, str):
            raise D.Reject(f"{k}: not a string")
        out[int(key)] = P.unhex(h)
    return out


def digests_json(d: dict[int, bytes]) -> dict[str, str]:
    return {str(i): h.hex() for i, h in d.items()}


def hexes(j: Any, k: str) -> list[bytes]:
    out = []
    for x in list_(j, k):
        if not isinstance(x, str):
            raise D.Reject(f"{k}: not a string")
        out.append(P.unhex(x))
    return out


def commitment_of(j: Any) -> mc.SessionCommitment:
    return mc.SessionCommitment(hex_(j, "session"), hex_(j, "coin_root"), hex_(j, "coin_key"), nat(j, "blocks"), nat(j, "k"),
                                nat(j, "limit_ns"), nat(j, "block_bytes"), hex_(j, "binding"))


def round_of(j: Any) -> mc.Round:
    return mc.Round(nat(j, "j"), nat(j, "index"), nat(j, "send_ns"), nat(j, "receipt_ns"), hex_(j, "digest"))


def opening_of(j: Any) -> mc.CoinOpening:
    return mc.CoinOpening(nat(j, "j"), nat(j, "index"), hex_(j, "salt"), tuple(hexes(j, "path")))


def signed_of(j: Any) -> mc.SignedSession:
    return mc.SignedSession(commitment_of(_get(j, "commitment")), _str(j, "ended"), tuple(round_of(r) for r in list_(j, "rounds")),
                            _str(j, "scheme"), hex_(j, "public_key"), hex_(j, "signature"))


@dataclass(frozen=True)
class Params:
    expected: dict[int, bytes]
    trusted: tuple[bytes, ...]

    def to_json(self) -> dict:
        return {"expected": digests_json(self.expected), "trusted": [t.hex() for t in self.trusted]}


@dataclass(frozen=True)
class Evidence:
    session: mc.SignedSession
    openings: tuple[mc.CoinOpening, ...]

    def to_json(self) -> dict:
        return {"openings": [o.to_json() for o in self.openings], "session": self.session.to_json()}


def _exactly(decode, j: Any):
    a = decode(j)
    if D.canonical(a.to_json()) != D.canonical(j):
        raise D.Reject("not the canonical JSON of what it holds")
    return a


def decode_params(j: Any) -> Params:
    return _exactly(lambda j: Params(digests(j, "expected"), tuple(hexes(j, "trusted"))), j)


def decode_commitment(b: bytes) -> mc.SessionCommitment:
    return _exactly(commitment_of, P.canonical_json(b))


def decode_evidence(b: bytes) -> Evidence:
    return _exactly(lambda j: Evidence(signed_of(_get(j, "session")), tuple(opening_of(o) for o in list_(j, "openings"))),
                    P.canonical_json(b))


def commitment_digest(c: mc.SessionCommitment) -> bytes:
    return hashlib.sha512(D.canonical(c.to_json()).encode()).digest()


def judge(p: Params, c: mc.SessionCommitment, e: Evidence, schemes) -> mc.Verdict:
    return mc.verdict(e.session, c, e.openings, p.expected.get, trusted=set(p.trusted), schemes=schemes)


def protocol_reference(protocol_id: str, schemes):
    """`MemoryChallenger.mkProtocol protocol_id schemes` in Python: `P.answer` over the decoders and `verdict`."""

    def check(p, c, e):
        v = judge(p, c, e, schemes)
        return ("accept", v.public(), None) if v.accepted else ("reject", "; ".join(v.reasons))

    return lambda case: P.answer(protocol_id, case, params=decode_params, commitment=decode_commitment,
                                 evidence=decode_evidence, check=check, digest=commitment_digest)


# --- keys and signers ------------------------------------------------------------------------------------------------------

#: The sessions' hm96 keys: few, since a key's tables are built once and cached.
COIN_KEYS = [mc.HM96.key_from_label(b"memory-challenger-lean/coin key %d" % i) for i in range(2)]
SETUP_ROOTS = [hashlib.sha512(b"memory-challenger-lean/setup root %d" % i).digest() for i in range(2)]
"""Roots a session is bound to: a setup's root of `C*` is 64 bytes."""
SECRETS = [b"challenger a", b"challenger b"]
STRANGER = b"someone else"


@cache
def test_key(secret: bytes, height: int) -> lms.PrivateKey:
    """`pous_test_signer.TestSigner(secret, height)`'s key."""
    params = lms.Params(64, 1, height)
    return lms.keygen(params, hashlib.sha512(b"verity-pous/test-signer/v0" + secret).digest()[:params.N])


class _SparseTree:
    """A deployed key's tree for leaf 0 alone (`test_pous_memory_challenger._SparseTree`)."""

    def __init__(self, I: bytes, seed: bytes) -> None:
        self.root = b""
        probe = lms.sign_at(lms.PrivateKey(lms.DEPLOYED, I, seed, self), 0, b"probe")  # type: ignore[arg-type]
        self.root = lms.root_of(lms.DEPLOYED, I, b"probe", probe)

    def __getitem__(self, r: int) -> bytes:
        return self.root if r == 1 else hashlib.sha512(b"sibling%d" % r).digest()


@cache
def deployed_key() -> lms.PrivateKey:
    I, seed = hashlib.sha512(b"memory-challenger-lean/I").digest()[:16], hashlib.sha512(b"memory-challenger-lean/seed").digest()
    return lms.PrivateKey(lms.DEPLOYED, I, seed, _SparseTree(I, seed))  # type: ignore[arg-type]


@dataclass(frozen=True)
class AtLeaf:
    """`LmsSigner` with its leaf state at `q`, with no state file: `sign` signs at `q` every time."""

    key: lms.PrivateKey
    q: int
    scheme: str

    @property
    def public_key(self) -> bytes:
        return lms.public_key(self.key)

    def sign(self, session: bytes, record: bytes) -> bytes:
        return lms.sign_at(self.key, self.q, lms.memory_challenger_message(self.key.I, session, record)).encode()


def test_signer(secret: bytes, height: int, q: int) -> AtLeaf:
    return AtLeaf(test_key(secret, height), q, ts.scheme(height))


def deployed_signer() -> AtLeaf:
    return AtLeaf(deployed_key(), 0, mc.LMS)


# --- sessions --------------------------------------------------------------------------------------------------------------

def block(i: int, n: int) -> bytes:
    return hashlib.shake_256(b"C%d" % i).digest(n)


class Clock:
    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


@dataclass(frozen=True)
class Session:
    """A session as a protocol case holds it, with the signer that signed it (to re-sign an edit)."""

    signed: mc.SignedSession
    commitment: mc.SessionCommitment
    openings: tuple[mc.CoinOpening, ...]
    expected: tuple[bytes, ...]
    trusted: tuple[bytes, ...]
    signer: AtLeaf

    def case(self) -> dict:
        return {"params": Params(dict(enumerate(self.expected)), self.trusted).to_json(),
                "commitment": D.canonical(self.commitment.to_json()).encode().hex(),
                "evidence": D.canonical(Evidence(self.signed, self.openings).to_json()).encode().hex()}

    def resign(self, **edits) -> Session:
        s = dataclasses.replace(self.signed, **edits)
        try:
            sig = self.signer.sign(s.commitment.session, s.message())
        except (ValueError, OverflowError):  # an edit past the record's widths has no message; the old signature stays
            sig = s.signature
        return dataclasses.replace(self, signed=dataclasses.replace(s, signature=sig))


@st.composite
def sessions(draw, signer: AtLeaf | None = None) -> Session:
    """An honest run of `MemoryChallenger` (some rounds late, some answers tampered), signed by `signer` or a drawn test
    signer at a drawn leaf."""
    if signer is None:
        height = draw(st.integers(1, 2))
        signer = test_signer(draw(st.sampled_from(SECRETS)), height, draw(st.integers(0, 2 ** height - 1)))
    blocks, k = draw(st.integers(1, 24)), draw(st.integers(1, 4))
    block_bytes, limit = draw(st.integers(1, 40)), draw(st.integers(1, 10 ** 6))
    seed = draw(st.integers(0, 3))
    session_id = hashlib.sha512(b"session %d" % draw(st.integers(0, 3))).digest()[:32]
    tree = mc.CoinTree.draw(b"source %d" % seed, session_id, blocks, k, key=draw(st.sampled_from(COIN_KEYS)),
                            salts=[hashlib.sha512(b"salt %d %d" % (seed, j)).digest() * 3 for j in range(k)])
    spends = draw(st.lists(st.one_of(st.integers(0, limit), st.just(limit), st.just(limit + 1)), min_size=k, max_size=k))
    tampered = draw(st.sets(st.integers(0, k - 1), max_size=1))
    clock = Clock(draw(st.integers(0, 2 ** 40)))
    c = mc.MemoryChallenger(tree, signer, log=ts.session_log(), limit_ns=limit, block_bytes=block_bytes,
                            binding=draw(st.sampled_from(SETUP_ROOTS)), clock=clock)
    while not c.done:
        j = len(c.rounds)
        i = c.challenge()
        clock.now += spends[j]
        b = block(i, block_bytes)
        c.receive(b[:-1] + bytes([b[-1] ^ 1]) if j in tampered else b)
    signed = c.close()
    expected = tuple(mc.block_digest(block(i, block_bytes)) for i in range(blocks))
    return Session(signed, c.commitment, c.openings(), expected, (signer.public_key,), signer)


def _round(s: Session, x: int, **edits) -> Session:
    rounds = list(s.signed.rounds)
    r = x % len(rounds)
    rounds[r] = dataclasses.replace(rounds[r], **{k: v(rounds[r]) for k, v in edits.items()})
    return s.resign(rounds=tuple(rounds))


def _unsigned(s: Session, **edits) -> Session:
    return dataclasses.replace(s, signed=dataclasses.replace(s.signed, **edits))


def _signature(s: Session, sig: bytes) -> Session:
    return _unsigned(s, signature=sig)


def _flip(b: bytes, x: int) -> bytes:
    return b[:x % len(b)] + bytes([b[x % len(b)] ^ (1 << (x % 8))]) + b[x % len(b) + 1:] if b else b"\x00"


def _openings(s: Session, f) -> Session:
    return dataclasses.replace(s, openings=tuple(f(list(s.openings))))


def _stranger(s: Session, x: int) -> Session:
    other = test_signer(STRANGER, 1, x % 2)
    return dataclasses.replace(s, signed=dataclasses.replace(s.signed, public_key=other.public_key,
                                                             signature=other.sign(s.signed.commitment.session, s.signed.message())))


def _replayed(s: Session, x: int) -> Session:
    """The signer's signature at the same leaf over another session's record: a replayed leaf."""
    other = dataclasses.replace(s.signed, rounds=s.signed.rounds[:-1] if x % 2 else s.signed.rounds + s.signed.rounds[-1:])
    return _signature(s, s.signer.sign(s.signed.commitment.session, other.message()))


#: Schemes a session may name, accepted or not; the last is non-printable past ASCII, which Python's `repr` escapes and
#: `MemoryChallenger.pyRepr` doesn't.
SCHEME_NAMES = ["none", mc.LMS, *(ts.scheme(h) for h in (1, 2, 3, 10, 11)), "x'y", 'a"b', "a'\"b\\", "\t\n\r\x00\x7f",
                "é ünï ☃ 𝄞", "\x85\u2028"]
PRINTABLE_SCHEMES = SCHEME_NAMES[:-1]


def _scheme(s: Session, x: int) -> Session:
    return _unsigned(s, scheme=SCHEME_NAMES[x % len(SCHEME_NAMES)])


def _printable_scheme(s: Session, x: int) -> Session:
    return _unsigned(s, scheme=PRINTABLE_SCHEMES[x % len(PRINTABLE_SCHEMES)])

#: One adversarial edit of a session, `(session, x) -> session`, `x` a drawn natural.
MUTATIONS = [
    lambda s, x: _signature(s, _flip(s.signed.signature, x)),
    lambda s, x: _signature(s, s.signed.signature[:-1 - x % 3]),
    lambda s, x: _signature(s, b""),
    lambda s, x: _signature(s, s.signed.signature[:4] + (x % 2 ** 32).to_bytes(4, "big") + s.signed.signature[8:]),
    lambda s, x: dataclasses.replace(s, trusted=()),
    lambda s, x: dataclasses.replace(s, trusted=(test_key(STRANGER, 1).I + bytes(64),)),
    lambda s, x: dataclasses.replace(s, trusted=s.trusted + (_flip(s.signed.public_key, x),)),
    _stranger,
    lambda s, x: dataclasses.replace(_stranger(s, x), trusted=s.trusted + (test_signer(STRANGER, 1, 0).public_key,)),
    _scheme,
    _replayed,
    lambda s, x: _signature(s, test_signer(SECRETS[x % 2], 1 + x % 2, 0).sign(b"another session", s.signed.message())),
    lambda s, x: _unsigned(s, commitment=dataclasses.replace(s.commitment, limit_ns=s.commitment.limit_ns + 1)),
    lambda s, x: s.resign(commitment=dataclasses.replace(s.commitment, blocks=s.commitment.blocks + 1 + x % 3)),
    lambda s, x: s.resign(commitment=dataclasses.replace(s.commitment, binding=s.commitment.binding + b"x")),
    lambda s, x: dataclasses.replace(s, commitment=dataclasses.replace(s.commitment, k=s.commitment.k + 1)),
    lambda s, x: dataclasses.replace(s, commitment=dataclasses.replace(s.commitment, coin_root=_flip(s.commitment.coin_root, x))),
    lambda s, x: _unsigned(s, rounds=_round(s, x, receipt_ns=lambda r: r.receipt_ns + 1).signed.rounds),
    lambda s, x: _openings(s, lambda o: o[::-1]),
    lambda s, x: _openings(s, lambda o: o[:-1]),
    lambda s, x: _openings(s, lambda o: o + [dataclasses.replace(o[x % len(o)], index=o[x % len(o)].index + 1)]),
    lambda s, x: _openings(s, lambda o: [dataclasses.replace(o[x % len(o)], index=o[x % len(o)].index + 1)] + o),
    lambda s, x: _openings(s, lambda o: [dataclasses.replace(p, salt=_flip(p.salt, x)) if i == x % len(o) else p
                                         for i, p in enumerate(o)]),
    lambda s, x: _openings(s, lambda o: [dataclasses.replace(p, path=p.path[:-1]) if i == x % len(o) else p
                                         for i, p in enumerate(o)]),
    lambda s, x: _openings(s, lambda o: [dataclasses.replace(p, j=p.j + 2 ** 32) for p in o]),
    lambda s, x: dataclasses.replace(s, expected=s.expected[:x % (len(s.expected) + 1)]),
    lambda s, x: dataclasses.replace(s, expected=tuple(_flip(e, x) for e in s.expected)),
    lambda s, x: _round(s, x, index=lambda r: (r.index + 1) % s.commitment.blocks),
    lambda s, x: _round(s, x, index=lambda r: r.index + s.commitment.blocks),
    lambda s, x: _round(s, x, index=lambda r: 2 ** 64 + r.index),
    lambda s, x: _round(s, x, j=lambda r: r.j + 1),
    lambda s, x: _round(s, x, j=lambda r: 2 ** 32 + r.j),
    lambda s, x: _round(s, x, digest=lambda r: _flip(r.digest, x)),
    lambda s, x: _round(s, x, digest=lambda r: r.digest[:-1]),
    lambda s, x: _round(s, x, receipt_ns=lambda r: r.send_ns + s.commitment.limit_ns + 1 + x),
    lambda s, x: _round(s, x, receipt_ns=lambda r: r.send_ns - 1 if r.send_ns else 0),
    lambda s, x: _round(s, x, send_ns=lambda r: r.receipt_ns + 1),
    lambda s, x: _round(s, x, send_ns=lambda r: 2 ** 64),
    lambda s, x: s.resign(rounds=s.signed.rounds + s.signed.rounds[-1:]),
    lambda s, x: s.resign(rounds=s.signed.rounds[:-1]),
    lambda s, x: s.resign(rounds=()),
    lambda s, x: s.resign(ended=[mc.LATE, mc.COMPLETE, "", "ünïcode\n\x01", "x" * (x % 5)][x % 5]),
    lambda s, x: dataclasses.replace(s.resign(commitment=dataclasses.replace(s.commitment, k=len(s.signed.rounds))),
                                     commitment=dataclasses.replace(s.commitment, k=len(s.signed.rounds))),
]


def _json_edit(field: str, f):
    """An edit of the case's `field` (`params`, or the JSON inside `commitment` or `evidence`) at the JSON level."""

    def edit(case: dict, x: int) -> dict:
        if field == "params":
            return case | {"params": f(case["params"], x)}
        j = P.canonical_json(bytes.fromhex(case[field]))
        out = f(j, x)
        return case | {field: (out if isinstance(out, bytes) else D.canonical(out).encode()).hex()}

    return edit


def _drop(j: dict, x: int) -> dict:
    keys = sorted(j)
    return {k: v for k, v in j.items() if k != keys[x % len(keys)]}


def _at(path: tuple[str, ...], f):
    """`f` applied at `path` of nested objects."""
    def go(j, x):
        return f(j, x) if not path else j | {path[0]: _at(path[1:], f)(j[path[0]], x)}
    return go


#: One edit of a valid case's JSON: an extra or missing field, a constant not the protocol's, a hash that doesn't match,
#: uppercase or non-hex, a negative or non-integer number, not canonical, not UTF-8, the wrong type.
JSON_MUTATIONS = [
    _json_edit("commitment", lambda j, x: j | {"extra": x}),
    _json_edit("commitment", _drop),
    _json_edit("commitment", lambda j, x: j | {"protocol": "verity-pous/memory-challenger/v0"}),
    _json_edit("commitment", lambda j, x: j | {"clock": "CLOCK_REALTIME ns"}),
    _json_edit("commitment", lambda j, x: j | {"coin_tree": mc.RECORDS_TREE}),
    _json_edit("commitment", lambda j, x: j | {"coin_key_sha512": "00" * 64}),
    _json_edit("commitment", lambda j, x: j | {"session": j["session"].upper()}),
    _json_edit("commitment", lambda j, x: j | {"binding": j["binding"] + "0"}),
    _json_edit("commitment", lambda j, x: j | {"blocks": -1 - x}),
    _json_edit("commitment", lambda j, x: j | {"k": str(j["k"])}),
    _json_edit("commitment", lambda j, x: j | {"limit_ns": True}),
    _json_edit("commitment", lambda j, x: D.canonical(j).replace(",", ", ", 1).encode()),
    _json_edit("commitment", lambda j, x: b"\xff" + D.canonical(j).encode()),
    _json_edit("commitment", lambda j, x: [j]),
    _json_edit("evidence", lambda j, x: j | {"extra": None}),
    _json_edit("evidence", _drop),
    _json_edit("evidence", _at(("session",), _drop)),
    _json_edit("evidence", _at(("session",), lambda j, x: j | {"note": "x"})),
    _json_edit("evidence", _at(("session", "commitment"), lambda j, x: j | {"coin_key_sha512": "ab" * 64})),
    _json_edit("evidence", _at(("session", "rounds"), lambda r, x: r + [r[-1] | {"extra": 1}] if r else [{}])),
    _json_edit("evidence", _at(("session", "rounds"), lambda r, x: [{k: v for k, v in r[0].items() if k != "digest"}] if r else [1])),
    _json_edit("evidence", _at(("session", "signature"), lambda h, x: h.upper() if h else "0")),
    _json_edit("evidence", _at(("session", "ended"), lambda e, x: 1)),
    _json_edit("evidence", _at(("openings",), lambda o, x: o + [o[0] | {"path": [1]}] if o else [{}])),
    _json_edit("evidence", _at(("openings",), lambda o, x: o + [o[0] | {"index": -1}] if o else [None])),
    _json_edit("evidence", _at(("openings",), lambda o, x: {"0": o})),
    _json_edit("evidence", lambda j, x: D.canonical(j).encode() + b" "),
    _json_edit("params", lambda j, x: j | {"extra": []}),
    _json_edit("params", _drop),
    _json_edit("params", lambda j, x: j | {"trusted": [t.upper() for t in j["trusted"]] or ["A"]}),
    _json_edit("params", lambda j, x: j | {"expected": "00"}),
    _json_edit("params", lambda j, x: j | {"expected": j["expected"] | {["01", "-1", "x", "1_0", "١"][x % 5]: "00"}}),
    _json_edit("params", lambda j, x: j | {"expected": j["expected"] | {str(x): x}}),
    _json_edit("params", lambda j, x: j | {"expected": list(j["expected"].values())}),
    _json_edit("params", lambda j, x: [j]),
    lambda case, x: case | {"commitment": case["commitment"].upper()},
    lambda case, x: case | {"evidence": case["evidence"][:-1]},
    lambda case, x: {k: v for k, v in case.items() if k != sorted(case)[x % 3]},
    lambda case, x: case | {"evidence": 0},
]


def cases(signer: AtLeaf | None = None, json_mutations: bool = True, printable: bool = False) -> st.SearchStrategy:
    """Protocol cases: honest sessions, sessions under one adversarial edit, and (with `json_mutations`) cases under one JSON
    edit. `printable` names only printable schemes, whose reason `MemoryChallenger.pyRepr` writes as Python's `repr`."""
    mutations = [_printable_scheme if printable and m is _scheme else m for m in MUTATIONS]
    edited = st.builds(lambda s, m, x: m(s, x).case(), sessions(signer), st.sampled_from(mutations), st.integers(0, 2 ** 16))
    honest = sessions(signer).map(Session.case)
    if not json_mutations:
        return st.one_of(honest, edited)
    malformed = st.builds(lambda s, m, x: m(s.case(), x), sessions(signer), st.sampled_from(JSON_MUTATIONS), st.integers(0, 2 ** 16))
    return st.one_of(honest, edited, malformed)
