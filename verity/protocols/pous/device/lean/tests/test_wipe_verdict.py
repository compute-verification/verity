"""The streamed wipe's verdict on a record (`streamed-wipe-verdict`, `MemoryChallenger.StreamedWipe`) against
`streamed_wipe.verdict`: the whole verdict (accepted, and the reasons word for word) on records as
`wipe_verdict.py` writes them from honest sessions at every batch size (each above 1 refused as uncovered), sessions
under one adversarial edit (`wipe_cases.MUTATIONS`), records under one JSON edit, and each accepted-schemes mapping a
caller passes; then the named negative controls through `wipe_verdict.verdict` and `judge`, the calls a caller makes, the
format (a v1 session, a session that names an egress, a commitment that names the slewed clock and a start mark that is
not one refused, an empty mark and a digest read), and the switch itself."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import shutil

import pytest
from hypothesis import strategies as st

import mc_cases as M
import verity_difftest as D
import wipe_cases as W
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import streamed_wipe as sw

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

_spec = importlib.util.spec_from_file_location("streamed_wipe_lean_verdict", M.PACKAGE / "wipe_verdict.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

#: The mappings a caller passes: the accepted schemes, the test signers', both, none, one test height alone.
SCHEME_SETS = [mc.SCHEMES, M.ts.SCHEMES, W.KNOWN, {}, {M.ts.scheme(1): M.ts.SCHEMES[M.ts.scheme(1)]}]


def record(s: W.Session, schemes=M.ts.SCHEMES) -> dict:
    return V.record(s.signed, s.commitment, trusted=set(s.trusted), schemes=schemes)


def reference(j) -> dict:
    """`MemoryChallenger.StreamedWipe.verdictFn` in Python: decode as the Lean decodes, then `streamed_wipe.verdict`."""
    signed = W.signed_of(M._get(j, "signed"))
    commitment = W.commitment_of(M._get(j, "commitment"))
    trusted = M.hexes(j, "trusted")
    ids = M.list_(j, "schemes")
    if not all(isinstance(i, str) for i in ids):
        raise D.Reject("record.schemes: not a string")
    again = {"signed": signed.to_json(), "commitment": commitment.to_json(), "trusted": [t.hex() for t in trusted],
             "schemes": ids}
    if D.canonical(again) != D.canonical(j):
        raise D.Reject("not the canonical JSON of what it holds")
    if any(i not in W.KNOWN for i in ids):
        raise D.Reject("a scheme this port doesn't verify")
    return V.python(signed, commitment, trusted=set(trusted), schemes={i: W.KNOWN[i] for i in ids})


def port(inputs, **kw) -> D.Port:
    return D.Port(V.EXE, "verdict", reference, inputs, package=M.PACKAGE,
                  of_record="python" if V.VERDICT == "python" else "lean", **kw)


server = D.server_fixture(port(st.none()))


def records(signer=None, malformed: bool = True) -> st.SearchStrategy:
    """Records of honest sessions, of sessions under one edit, and (with `malformed`) under one JSON edit, each under a
    sampled mapping."""
    schemes = st.sampled_from(SCHEME_SETS)
    honest = st.builds(record, W.sessions(signer), schemes)
    edited = st.builds(lambda s, m, x, sc: record(m(s, x), sc), W.sessions(signer), st.sampled_from(W.MUTATIONS),
                       st.integers(0, 2 ** 16), schemes)
    if not malformed:
        return st.one_of(honest, edited)
    edited_json = st.builds(lambda s, m, x: m(record(s), x), W.sessions(signer), st.sampled_from(W.JSON_MUTATIONS),
                            st.integers(0, 2 ** 16))
    return st.one_of(honest, edited, edited, edited_json)


@pytest.mark.slow
def test_verdict_agrees(server):
    """The whole verdict on sampled records under the test signers' keys: accepted ones at batch size 1, rejected ones
    above it, every rejection reason, refusals."""
    stats = D.run(port(records()), examples=600, server=server)
    assert stats.outputs > 300 and stats.rejects > 30, stats


@pytest.mark.slow
def test_verdict_agrees_at_the_deployed_parameters(server):
    """Sessions signed by a key at S1's deployed parameters (`lms.DEPLOYED`, a sparse tree): accepted under `SCHEMES`."""
    stats = D.run(port(records(M.deployed_signer(), malformed=False), need_rejects=False), examples=30, server=server)
    assert stats.outputs == 30, stats
    s = W.run(_params(), M.deployed_signer())
    python = sw.verdict(s.signed, s.commitment, trusted=set(s.trusted))
    assert python.accepted and V.verdict(s.signed, s.commitment, trusted=set(s.trusted)) == python


LIMIT = 1_000


def _params(batch: int = 1, k: int = 3, labels: int = 8) -> sw.WipeParams:
    return sw.WipeParams(labels, k, batch, LIMIT, (("gpu", 3 * sw.LABEL_BYTES), ("host-dram", (labels - 3) * sw.LABEL_BYTES)))


CERTIFICATE = b"a start certificate, as the network certifier returned it"


def _honest(batch: int = 1, spends=(1, 1, 1), tampered=frozenset(),
            certificate: bytes | None = None) -> W.Session:
    """A three-challenge session over 8 labels in two tiers, its deadline 1000 ns, signed at leaf 1 of a height-1 key;
    with `certificate`, its start mark is that certificate's SHA-512."""
    return W.run(_params(batch), M.test_signer(M.SECRETS[0], 1, 1), spends=lambda b: spends[b], tampered=tampered,
                 certificate=certificate)


def _uncovered(batch: int) -> str:
    return f"batch size {batch} is uncovered: the guarantee covers the sequential reveal, batch size 1"


def controls() -> dict[str, tuple[W.Session, tuple[str, ...]]]:
    """The negative controls, each a session and the reasons its verdict gives, beside the honest
    session at batch size 1, with no start mark and with one; an otherwise honest session at batch size 2 is one,
    rejected as uncovered."""
    s = _honest()
    marked = _honest(certificate=CERTIFICATE)
    other = hashlib.sha512(CERTIFICATE + b"x").digest()
    return {
        "honest": (s, ()),
        "honest, with a start mark": (marked, ()),
        "another mark, signed": (marked.resign(start_mark=other), ()),
        "mark edited": (W._unsigned(marked, start_mark=other), ("the signature does not verify",)),
        "mark dropped": (W._unsigned(marked, start_mark=b""), ("the signature does not verify",)),
        "mark added": (W._unsigned(s, start_mark=other), ("the signature does not verify",)),
        "honest, batch 2": (_honest(batch=2), (_uncovered(2),)),
        "late answer, batch 2": (_honest(batch=2, spends=(1, LIMIT + 1)),
                                 (_uncovered(2), "the session ended late after 2 of 2 batches",
                                  f"batch 1 is past its deadline of {LIMIT} ns")),
        "bad signature": (W._signature(s, M._flip(s.signed.signature, 100)), ("the signature does not verify",)),
        "untrusted signer": (dataclasses.replace(s, trusted=()),
                             ("the session is signed by a key that is not a trusted memory challenger's",)),
        "stranger's signature": (W._stranger(s, 0),
                                 ("the session is signed by a key that is not a trusted memory challenger's",)),
        "unknown scheme": (W._unsigned(s, scheme="none"), ("signature scheme 'none' is not accepted",)),
        "wrong header": (s.resign(commitment=W._params(s, limit_ns=LIMIT + 1)),
                         ("the signed header is not the session's commitment",)),
        "wrong label": (_honest(tampered={1}), ("the session ended wrong after 2 of 3 batches",
                                                "batch 1 has 1 wrong answers")),
        "late answer": (_honest(spends=(1, LIMIT + 1, 1)), ("the session ended late after 2 of 3 batches",
                                                            f"batch 1 is past its deadline of {LIMIT} ns")),
        "answer before its challenge": (W._batch(s, 2, receipt_ns=lambda r: r.send_ns - 1),
                                        (f"batch 2 is past its deadline of {LIMIT} ns",)),
        "answer at the deadline": (W._batch(s, 2, receipt_ns=lambda r: r.send_ns + LIMIT), ()),
        "wrong count": (W._batch(s, 1, count=lambda r: 2), ("batch 1 is not the session's batch 1",)),
        "wrong batch number": (W._batch(s, 0, b=lambda r: 1), ("batch 0 is not the session's batch 0",)),
        "wrong first index": (W._batch(s, 2, first=lambda r: 1), ("batch 2 is not the session's batch 2",)),
        "incomplete session": (s.resign(batches=s.signed.batches[:-1]),
                               ("the session ended complete after 2 of 3 batches",)),
        "extra batch": (s.resign(batches=s.signed.batches + s.signed.batches[-1:]),
                        ("the session ended complete after 4 of 3 batches", "batch 3 is not the session's batch 3")),
        "sent before the fill completed": (W._first_before_fill(s, 0),
                                           ("the first batch was sent before the fill completed",)),
        "record past its width": (W._batch(s, 0, b=lambda r: 2 ** 32),
                                  ("the signature does not verify", "batch 0 is not the session's batch 0")),
    }


def test_negative_controls(server):
    """Each negative control by name, through `wipe_verdict.verdict`, `judge` under each mode and the difftest port: the
    Lean's verdict is the Python's, reasons word for word, and each control is rejected for its own reason."""
    port_ = port(st.none())
    for name, (s, reasons) in controls().items():
        kw = {"trusted": set(s.trusted), "schemes": M.ts.SCHEMES}
        python = sw.verdict(s.signed, s.commitment, **kw)
        assert python == sw.Verdict(not reasons, reasons), (name, python)
        assert python.public() == {"accepted": not reasons}, name
        assert V.verdict(s.signed, s.commitment, **kw) == python, name
        lean = V.lean(s.signed, s.commitment, **kw)
        assert s.commitment.params.covered == ("batch 2" not in name), name
        for decide in V.VERDICTS:
            v, sides = V.judge(s.signed, s.commitment, decide=decide, **kw)
            assert v == python and set(sides) == {"lean": {"lean"}, "both": {"lean", "python"},
                                                  "python": {"python"}}[decide], (name, decide, v, sides)
            assert all(side == lean for side in sides.values()), (name, decide, sides)
        rec = record(s)
        assert port_.key(D.replay(port_, rec, server)) == port_.key(port_.python(rec)), name


def test_the_format_is_v2(server):
    """A record of an honest session, with a start mark or none, is read, its signed session names no egress and its
    header names the raw clock; the same session as the v1 format wrote it (no start mark, a signed egress, the v1
    protocol in its commitments), a signed session that names an egress, a commitment that names `CLOCK_MONOTONIC ns`
    (the published one, the signed session's, or both), and a start mark in upper case or of another length, are
    refused by the Lean and by the reference."""
    port_ = port(st.none())
    for s in (_honest(), _honest(certificate=CERTIFICATE)):
        rec = record(s)
        mark = rec["signed"]["start_mark"]
        assert mark == s.signed.start_mark.hex() and len(mark) in (0, 128)
        assert rec["commitment"]["clock"] == rec["signed"]["commitment"]["clock"] == sw.CLOCK == "CLOCK_MONOTONIC_RAW ns"
        assert sorted(rec["signed"]) == ["batches", "commitment", "ended", "fill_end_ns", "fill_start_ns", "public_key",
                                         "scheme", "signature", "start_mark"]
        assert D.replay(port_, rec, server)[0] == "output"
        v1 = {"protocol": "verity-pous/streamed-wipe/v1"}
        unmarked = {k: v for k, v in rec["signed"].items() if k != "start_mark"}
        digest = mark or hashlib.sha512(CERTIFICATE).hexdigest()
        refused = [W._v1(rec, 0),
                   rec | {"signed": unmarked},
                   rec | {"signed": rec["signed"] | {"egress_bits": 0}},
                   rec | {"signed": rec["signed"] | {"egress_bits": 1234}},
                   rec | {"commitment": rec["commitment"] | v1},
                   rec | {"signed": rec["signed"] | {"commitment": rec["signed"]["commitment"] | v1}},
                   rec | {"commitment": rec["commitment"] | v1,
                          "signed": rec["signed"] | {"commitment": rec["signed"]["commitment"] | v1}}]
        slewed = {"clock": "CLOCK_MONOTONIC ns"}
        refused += [rec | {"commitment": rec["commitment"] | slewed},
                    rec | {"signed": rec["signed"] | {"commitment": rec["signed"]["commitment"] | slewed}},
                    W._slewed_clock(rec, 0)]
        refused += [rec | {"signed": rec["signed"] | {"start_mark": bad}}
                    for bad in (digest.upper(), digest[:-2], digest + "00", digest[:64], "00", digest[:-1], None)]
        assert digest.upper() != digest
        for bad in refused:
            assert D.replay(port_, bad, server)[0] == "reject", bad


def test_a_disagreement_is_refused(monkeypatch):
    """Under `both`, a Python verdict that differs from the Lean's in any field rejects the session, naming both; under
    `lean` the Python isn't consulted."""
    s = _honest()
    kw = {"trusted": set(s.trusted), "schemes": M.ts.SCHEMES}
    real = V.python(s.signed, s.commitment, **kw)
    for edit in ({"accepted": False}, {"reasons": ["x"]}):
        monkeypatch.setattr(V, "python", lambda *a, edit=edit, **k: real | edit)
        v, sides = V.judge(s.signed, s.commitment, decide="both", **kw)
        assert not v.accepted and v.reasons[0].startswith(V.DISAGREE), (edit, v)
        assert sides == {"lean": real, "python": real | edit}
        v, sides = V.judge(s.signed, s.commitment, decide="lean", **kw)
        assert v.accepted and set(sides) == {"lean"}


def test_refusals():
    """The Lean refuses, `verdict` raises and `judge` rejects, where the record names a scheme the port
    doesn't verify or holds a field `to_json` can't have written: it fails closed, never to an answer."""
    s = _honest()
    unknown = {**M.ts.SCHEMES, "an unknown scheme": None}
    with pytest.raises(ValueError, match="not a scheme this port verifies"):
        V.verdict(s.signed, s.commitment, trusted=set(s.trusted), schemes=unknown)
    for decide in ("lean", "both"):
        v, sides = V.judge(s.signed, s.commitment, trusted=set(s.trusted), schemes=unknown, decide=decide)
        assert not v.accepted and "refuses the record" in v.reasons[-1], v
    bad = dataclasses.replace(s.signed, ended=None)
    with pytest.raises(ValueError, match="refuses the record"):
        V.verdict(bad, s.commitment, trusted=set(s.trusted), schemes=M.ts.SCHEMES)
    with pytest.raises(ValueError, match="one of"):
        V.judge(s.signed, s.commitment, trusted=set(s.trusted), decide="lean-only")
