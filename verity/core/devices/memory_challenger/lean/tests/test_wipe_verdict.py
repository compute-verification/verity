"""The streamed wipe's verdict on a record (`streamed-wipe-verdict`, `MemoryChallenger.StreamedWipe`) against
`streamed_wipe.verdict`: the whole verdict (accepted, reasons word for word, and the egress in MiB) on records as
`wipe_verdict.py` writes them from honest sessions at every batch size (each above 1 refused as uncovered), sessions
under one adversarial edit (`wipe_cases.MUTATIONS`), records under one JSON edit, and each accepted-schemes mapping a
caller passes; then the named negative controls through `wipe_verdict.verdict` and `judge`, the calls a caller makes, and
the switch itself."""

from __future__ import annotations

import dataclasses
import importlib.util
import shutil

import pytest
from hypothesis import strategies as st

import mc_cases as M
import verity_difftest as D
import wipe_cases as W
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw

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
    drawn mapping."""
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
    """The whole verdict on drawn records under the test signers' keys: accepted ones at batch size 1, rejected ones
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


def _honest(batch: int = 1, spends=(1, 1, 1), tampered=frozenset(), egress_bits: int = 1234) -> W.Session:
    """A three-challenge session over 8 labels in two tiers, its deadline 1000 ns, signed at leaf 1 of a height-1 key."""
    return W.run(_params(batch), M.test_signer(M.SECRETS[0], 1, 1), spends=lambda b: spends[b], tampered=tampered,
                 egress_bits=egress_bits)


def _uncovered(batch: int) -> str:
    return f"batch size {batch} is uncovered: the guarantee covers the sequential reveal, batch size 1"


def controls() -> dict[str, tuple[W.Session, tuple[str, ...], int]]:
    """The negative controls, each a session, the reasons its verdict gives and its egress in MiB, beside the honest
    session at batch size 1; an otherwise honest session at batch size 2 is one, rejected as uncovered."""
    s = _honest()
    mib = 1
    return {
        "honest": (s, (), mib),
        "honest, batch 2": (_honest(batch=2), (_uncovered(2),), mib),
        "late answer, batch 2": (_honest(batch=2, spends=(1, LIMIT + 1)),
                                 (_uncovered(2), "the session ended late after 2 of 2 batches",
                                  f"batch 1 is past its deadline of {LIMIT} ns"), mib),
        "bad signature": (W._signature(s, M._flip(s.signed.signature, 100)), ("the signature does not verify",), mib),
        "untrusted signer": (dataclasses.replace(s, trusted=()),
                             ("the session is signed by a key that is not a trusted memory challenger's",), mib),
        "stranger's signature": (W._stranger(s, 0),
                                 ("the session is signed by a key that is not a trusted memory challenger's",), mib),
        "unknown scheme": (W._unsigned(s, scheme="none"), ("signature scheme 'none' is not accepted",), mib),
        "wrong header": (s.resign(commitment=W._params(s, limit_ns=LIMIT + 1)),
                         ("the signed header is not the session's commitment",), mib),
        "wrong label": (_honest(tampered={1}), ("the session ended wrong after 2 of 3 batches",
                                                "batch 1 has 1 wrong answers"), mib),
        "late answer": (_honest(spends=(1, LIMIT + 1, 1)), ("the session ended late after 2 of 3 batches",
                                                            f"batch 1 is past its deadline of {LIMIT} ns"), mib),
        "answer before its challenge": (W._batch(s, 2, receipt_ns=lambda r: r.send_ns - 1),
                                        (f"batch 2 is past its deadline of {LIMIT} ns",), mib),
        "answer at the deadline": (W._batch(s, 2, receipt_ns=lambda r: r.send_ns + LIMIT), (), mib),
        "wrong count": (W._batch(s, 1, count=lambda r: 2), ("batch 1 is not the session's batch 1",), mib),
        "wrong batch number": (W._batch(s, 0, b=lambda r: 1), ("batch 0 is not the session's batch 0",), mib),
        "wrong first index": (W._batch(s, 2, first=lambda r: 1), ("batch 2 is not the session's batch 2",), mib),
        "incomplete session": (s.resign(batches=s.signed.batches[:-1]),
                               ("the session ended complete after 2 of 3 batches",), mib),
        "extra batch": (s.resign(batches=s.signed.batches + s.signed.batches[-1:]),
                        ("the session ended complete after 4 of 3 batches", "batch 3 is not the session's batch 3"), mib),
        "sent before the fill completed": (W._first_before_fill(s, 0),
                                           ("the first batch was sent before the fill completed",), mib),
        "record past its width": (W._batch(s, 0, b=lambda r: 2 ** 32),
                                  ("the signature does not verify", "batch 0 is not the session's batch 0"), mib),
        "egress edited": (W._unsigned(s, egress_bits=2 ** 20 * 8 + 1), ("the signature does not verify",), 2),
        "egress saturated": (_honest(egress_bits=10 ** 15), (), sw.EGRESS_MAX_UNITS),
    }


def test_negative_controls(server):
    """Each negative control by name, through `wipe_verdict.verdict`, `judge` under each mode and the difftest port: the
    Lean's verdict is the Python's, reasons word for word, and each control is rejected for its own reason."""
    port_ = port(st.none())
    for name, (s, reasons, mib) in controls().items():
        kw = {"trusted": set(s.trusted), "schemes": M.ts.SCHEMES}
        python = sw.verdict(s.signed, s.commitment, **kw)
        assert python == sw.Verdict(not reasons, reasons, mib), (name, python)
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


def test_a_disagreement_is_refused(monkeypatch):
    """Under `both`, a Python verdict that differs from the Lean's in any field rejects the session, naming both, with the
    egress saturated; under `lean` the Python isn't consulted."""
    s = _honest()
    kw = {"trusted": set(s.trusted), "schemes": M.ts.SCHEMES}
    real = V.python(s.signed, s.commitment, **kw)
    for edit in ({"accepted": False}, {"reasons": ["x"]}, {"egress_mib": 2}):
        monkeypatch.setattr(V, "python", lambda *a, edit=edit, **k: real | edit)
        v, sides = V.judge(s.signed, s.commitment, decide="both", **kw)
        assert not v.accepted and v.egress_mib == sw.EGRESS_MAX_UNITS and v.reasons[0].startswith(V.DISAGREE), (edit, v)
        assert sides == {"lean": real, "python": real | edit}
        v, sides = V.judge(s.signed, s.commitment, decide="lean", **kw)
        assert v.accepted and set(sides) == {"lean"}


def test_refusals():
    """The Lean refuses, `verdict` raises and `judge` rejects (egress saturated), where the record names a scheme the port
    doesn't verify or holds a field `to_json` can't have written: it fails closed, never to an answer."""
    s = _honest()
    unknown = {**M.ts.SCHEMES, "an unknown scheme": None}
    with pytest.raises(ValueError, match="not a scheme this port verifies"):
        V.verdict(s.signed, s.commitment, trusted=set(s.trusted), schemes=unknown)
    for decide in ("lean", "both"):
        v, sides = V.judge(s.signed, s.commitment, trusted=set(s.trusted), schemes=unknown, decide=decide)
        assert not v.accepted and v.egress_mib == sw.EGRESS_MAX_UNITS and "refuses the record" in v.reasons[-1], v
    bad = dataclasses.replace(s.signed, ended=None)
    with pytest.raises(ValueError, match="refuses the record"):
        V.verdict(bad, s.commitment, trusted=set(s.trusted), schemes=M.ts.SCHEMES)
    with pytest.raises(ValueError, match="one of"):
        V.judge(s.signed, s.commitment, trusted=set(s.trusted), decide="lean-only")
