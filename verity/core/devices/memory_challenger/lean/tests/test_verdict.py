"""The verdict on a record (`memory-challenger-verdict`, `MemoryChallenger.Judge`) against `memory_challenger.verdict`: the
whole `Verdict`, its reasons word for word, on records as `verdict.py` writes them from honest sessions, sessions under one
adversarial edit (`mc_cases.MUTATIONS`), records under one JSON edit, and each accepted-schemes mapping a caller passes;
then the named negative controls, through `verdict.verdict`, the call a call site switches to."""

from __future__ import annotations

import dataclasses
import importlib.util
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import mc_cases as M
import verity_difftest as D
from verity.core.protocols.pous import memory_challenger as mc

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

_spec = importlib.util.spec_from_file_location("memory_challenger_lean_verdict", M.PACKAGE / "verdict.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

#: Every scheme the port verifies, by id (`MemoryChallenger.KNOWN_SCHEMES`).
KNOWN = {**mc.SCHEMES, **M.ts.SCHEMES}
#: The mappings a caller passes: the accepted schemes, the test signers', both, none, one test height alone.
SCHEME_SETS = [mc.SCHEMES, M.ts.SCHEMES, KNOWN, {}, {M.ts.scheme(1): M.ts.SCHEMES[M.ts.scheme(1)]}]


def expected_of(s: M.Session):
    return lambda i: s.expected[i] if i < len(s.expected) else None


def record(s: M.Session, schemes=M.ts.SCHEMES) -> dict:
    return V.record(s.signed, s.commitment, s.openings, expected_of(s), trusted=set(s.trusted), schemes=schemes)


def reference(j) -> dict:
    """`MemoryChallenger.Judge.verdictFn` in Python: decode as the Lean decodes, then `memory_challenger.verdict`."""
    signed = M.signed_of(M._get(j, "signed"))
    commitment = M.commitment_of(M._get(j, "commitment"))
    openings = [M.opening_of(o) for o in M.list_(j, "openings")]
    expected = M.digests(j, "expected")
    trusted = M.hexes(j, "trusted")
    ids = M.list_(j, "schemes")
    if not all(isinstance(i, str) for i in ids):
        raise D.Reject("record.schemes: not a string")
    again = {"signed": signed.to_json(), "commitment": commitment.to_json(), "openings": [o.to_json() for o in openings],
             "expected": M.digests_json(expected), "trusted": [t.hex() for t in trusted], "schemes": ids}
    if D.canonical(again) != D.canonical(j):
        raise D.Reject("not the canonical JSON of what it holds")
    if any(i not in KNOWN for i in ids):
        raise D.Reject("a scheme this port doesn't verify")
    v = mc.verdict(signed, commitment, openings, expected.get, trusted=set(trusted), schemes={i: KNOWN[i] for i in ids})
    return {"accepted": v.accepted, "reasons": list(v.reasons), "late": v.late, "wrong": v.wrong}


def port(inputs, **kw) -> D.Port:
    return D.Port(V.EXE, "verdict", reference, inputs, package=M.PACKAGE, **kw)


server = D.server_fixture(port(st.none()))


def _edit(path: tuple[str, ...], f):
    def go(j, x):
        return f(j, x) if not path else j | {path[0]: _edit(path[1:], f)(j[path[0]], x)}
    return go


#: One edit of a valid record's JSON: a field extra or missing, a header constant or derived field not the protocol's,
#: hex not lowercase, a negative or non-integer number, an index key not decimal, a scheme id the port doesn't verify.
JSON_MUTATIONS = [
    lambda j, x: j | {"extra": x},
    lambda j, x: {k: v for k, v in j.items() if k != sorted(j)[x % len(j)]},
    _edit(("commitment",), lambda c, x: c | {"protocol": "verity-pous/memory-challenger/v0"}),
    _edit(("commitment",), lambda c, x: c | {"coin_key_sha512": "00" * 64}),
    _edit(("commitment",), lambda c, x: c | {"session": c["session"].upper()}),
    _edit(("commitment",), lambda c, x: c | {"blocks": -1 - x}),
    _edit(("commitment",), lambda c, x: c | {"k": str(c["k"])}),
    _edit(("signed",), lambda s, x: s | {"note": x}),
    _edit(("signed", "commitment"), lambda c, x: c | {"clock": "CLOCK_REALTIME ns"}),
    _edit(("signed", "rounds"), lambda r, x: [r[0] | {"receipt_ns": -1}] + r[1:] if r else [{}]),
    _edit(("signed", "signature"), lambda h, x: h.upper() if h else "0"),
    _edit(("openings",), lambda o, x: o + [o[0] | {"j": -1}] if o else [None]),
    _edit(("expected",), lambda e, x: e | {["01", "-1", "x", "1_0", "١"][x % 5]: "00"}),
    _edit(("expected",), lambda e, x: list(e.values())),
    _edit(("trusted",), lambda t, x: [k.upper() for k in t] or ["A"]),
    _edit(("schemes",), lambda s, x: s + [["none", mc.LMS + " ", M.ts.scheme(11), "test/"][x % 4]]),
    _edit(("schemes",), lambda s, x: s + [x]),
]


def records(signer=None, malformed: bool = True) -> st.SearchStrategy:
    """Records of honest sessions, of sessions under one edit, and (with `malformed`) under one JSON edit, each under a
    drawn mapping."""
    mutations = [M._printable_scheme if m is M._scheme else m for m in M.MUTATIONS]
    schemes = st.sampled_from(SCHEME_SETS)
    honest = st.builds(record, M.sessions(signer), schemes)
    edited = st.builds(lambda s, m, x, sc: record(m(s, x), sc), M.sessions(signer), st.sampled_from(mutations),
                       st.integers(0, 2 ** 16), schemes)
    if not malformed:
        return st.one_of(honest, edited)
    edited_json = st.builds(lambda s, m, x: m(record(s), x), M.sessions(signer), st.sampled_from(JSON_MUTATIONS),
                            st.integers(0, 2 ** 16))
    return st.one_of(honest, edited, edited, edited_json)


@pytest.mark.slow
def test_verdict_agrees(server):
    """The whole verdict on drawn records under the test signers' keys: accepted ones, every rejection reason, refusals."""
    stats = D.run(port(records()), examples=500, server=server)
    assert stats.outputs > 250 and stats.rejects > 20, stats


@pytest.mark.slow
def test_verdict_agrees_at_the_deployed_parameters(server):
    """Sessions signed by a key at S1's deployed parameters (`lms.DEPLOYED`, a sparse tree): accepted under `SCHEMES`."""
    stats = D.run(port(records(M.deployed_signer(), malformed=False), need_rejects=False), examples=60, server=server)
    assert stats.outputs == 60, stats
    s, signer = _honest(), M.deployed_signer()
    s = dataclasses.replace(s, trusted=(signer.public_key,), signed=dataclasses.replace(
        s.signed, public_key=signer.public_key, scheme=mc.LMS, signature=signer.sign(s.signed.commitment.session,
                                                                                   s.signed.message())))
    args = (s.signed, s.commitment, s.openings, expected_of(s))
    python = mc.verdict(*args, trusted=set(s.trusted))
    assert python.accepted and V.verdict(*args, trusted=set(s.trusted)) == python


def _honest(spends=(1, 1, 1)) -> M.Session:
    """A three-round session over 8 blocks of 16 bytes, its limit 10 ns, signed at leaf 1 of a height-1 test key."""
    signer = M.test_signer(M.SECRETS[0], 1, 1)
    tree = mc.CoinTree.draw(b"source", bytes(32), 8, 3, key=M.COIN_KEYS[0], salts=[bytes([j]) * 192 for j in range(3)])
    clock = M.Clock(1_000)
    c = mc.MemoryChallenger(tree, signer, log=M.ts.session_log(), limit_ns=10, block_bytes=16, binding=M.SETUP_ROOTS[0],
                            clock=clock)
    while not c.done:
        j = len(c.rounds)
        i = c.challenge()
        clock.now += spends[j]
        c.receive(M.block(i, 16))
    expected = tuple(mc.block_digest(M.block(i, 16)) for i in range(8))
    return M.Session(c.close(), c.commitment, c.openings(), expected, (signer.public_key,), signer)


def _salt(s: M.Session, j: int) -> M.Session:
    return dataclasses.replace(s, openings=tuple(dataclasses.replace(o, salt=M._flip(o.salt, 7)) if o.j == j else o
                                                 for o in s.openings))


def controls() -> dict[str, tuple[M.Session, tuple[str, ...]]]:
    """The negative controls, each a session and the reasons its verdict gives, beside the honest session."""
    s = _honest()
    limit = s.commitment.limit_ns
    blocks = s.commitment.blocks
    return {
        "honest": (s, ()),
        "bad signature": (M._signature(s, M._flip(s.signed.signature, 100)), ("the signature does not verify",)),
        "untrusted key": (dataclasses.replace(s, trusted=()),
                          ("the session is signed by a key that is not a trusted memory challenger's",)),
        "unknown scheme": (M._unsigned(s, scheme="none"), ("signature scheme 'none' is not accepted",)),
        "wrong header": (s.resign(commitment=dataclasses.replace(s.commitment, binding=M.SETUP_ROOTS[1])),
                         ("the signed header is not the session's commitment",)),
        "incomplete session": (s.resign(rounds=s.signed.rounds[:-1]), ("the session ended complete after 2 of 3 rounds",)),
        "wrong coin": (_salt(s, 1), ("round 1's index is not its committed coin",)),
        "wrong digest": (M._round(s, 2, digest=lambda r: M._flip(r.digest, 3)), ("1 answers are not their blocks of C",)),
        "late": (_honest(spends=[1, limit + 1, 1]),
                 ("the session ended late after 2 of 3 rounds", f"1 answers are past Δ + RTT = {limit} ns")),
        "negative elapsed": (_honest(spends=[1, -1, 1]),
                             ("the session ended late after 2 of 3 rounds", f"1 answers are past Δ + RTT = {limit} ns")),
        "out-of-range index": (M._round(s, 0, index=lambda r: r.index + blocks),
                               ("round 0's index is not its committed coin", "1 answers are not their blocks of C")),
    }


def test_negative_controls(server):
    """Each negative control by name, through `verdict.verdict` (the call a call site switches to) and the difftest port:
    the Lean's verdict is the Python's, reasons word for word, and the controls are rejected for their own reason."""
    port_ = port(st.none())
    for name, (s, reasons) in controls().items():
        args = (s.signed, s.commitment, s.openings, expected_of(s))
        python = mc.verdict(*args, trusted=set(s.trusted), schemes=M.ts.SCHEMES)
        lean = V.verdict(*args, trusted=set(s.trusted), schemes=M.ts.SCHEMES)
        assert lean == python, (name, lean, python)
        assert python.reasons == reasons and python.accepted == (name == "honest"), (name, python)
        rec = record(s)
        assert port_.key(D.replay(port_, rec, server)) == port_.key(port_.python(rec)), name


def test_refusals():
    """The Lean refuses, and `verdict.verdict` raises, where the record names a scheme the port doesn't verify or holds a
    field `to_json` can't have written: it fails closed, never to an answer."""
    s = _honest()
    args = (s.signed, s.commitment, s.openings, expected_of(s))
    with pytest.raises(ValueError, match="not a scheme this port verifies"):
        V.verdict(*args, trusted=set(s.trusted), schemes={**M.ts.SCHEMES, "an unknown scheme": None})
    bad = dataclasses.replace(s.signed, rounds=(dataclasses.replace(s.signed.rounds[0], receipt_ns=-1),) + s.signed.rounds[1:])
    with pytest.raises(ValueError, match="refuses the record"):
        V.verdict(bad, s.commitment, s.openings, expected_of(s), trusted=set(s.trusted), schemes=M.ts.SCHEMES)
