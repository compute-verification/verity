"""The wipe's half of the check that holds its count to the network certifier's (`MemoryChallenger.WipeCount`):
`wipe-count-test serve` against `wipe_protocol_case.counted_test_answer` (`wipe_period.outcome_counted` behind the
Lean's decoders) on sampled sessions, with a start mark and with none, as they ran, under one adversarial edit, unsigned
or under params of another server, each with a sampled egress in its evidence; one wipe with a start mark and a count,
accepted and certified at that count where `wipe-protocol-test` certifies nothing on the same case, and the same wipe
with no count, with no start mark, with a saturated count, unsigned, tampered, with a count out of range and under
another server, each with its reason; and `named`, what a composition's tie reads off the case (`wipe_named` of
`memory-challenger-difftest`), against `wipe_period.named`."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil

import mc_cases as M
import pytest
import verity_difftest as D
import wipe_cases as WC
import wipe_protocol_case as C
from hypothesis import strategies as st

from verity.protocols.accounting.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.pous.verifier import wipe_period as WP

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

#: What the Python raises where the Lean refuses (`test_formats.REFUSALS`).
REFUSALS = (D.Reject, ValueError, TypeError, OverflowError)
CERTIFICATE = b"a start certificate"


def counted_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("wipe-count-test", C.counted_test_answer, cases, package=M.PACKAGE)


def uncounted_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("wipe-protocol-test", C.test_answer, cases, package=M.PACKAGE)


def named(v) -> dict | None:
    """`wipe_period.named` on the port's input, byte strings as hex."""
    signed, egress = M._get(v, "signed"), M._get(v, "egress_mib")
    n = WP.named(WC.commitment_of(M._get(v, "commitment")), None if signed is None else WC.signed_of(signed),
                 None if egress is None else WC.int_(v, "egress_mib"))
    return None if n is None else {k: x.hex() if isinstance(x, bytes) else x for k, x in n.items()}


def named_port(inputs: st.SearchStrategy) -> D.Port:
    return D.Port("memory-challenger-difftest", "wipe_named", named, inputs, package=M.PACKAGE, rejects=REFUSALS)


counted_server = D.server_fixture(counted_port(st.none()))
uncounted_server = D.server_fixture(uncounted_port(st.none()))
named_server = D.server_fixture(named_port(st.none()))


@st.composite
def sampled(draw) -> dict:
    """A sampled session's case: as it ran, under one adversarial edit, unsigned, or under params of another server."""
    s = draw(WC.sessions())
    x = draw(st.integers(0, 2 ** 16))
    how = draw(st.sampled_from(["as-is", "as-is", "as-is", "as-is", "mutated", "unsigned", "unsigned", "other"]))
    if how == "mutated":
        s = draw(st.sampled_from(WC.MUTATIONS))(s, x)
    signed = None if how == "unsigned" else s.signed
    server = s.commitment.server + b"x" if how == "other" else None
    egress = draw(st.integers(0, sw.EGRESS_MAX_UNITS) if draw(st.booleans()) else WC.EGRESS)
    return C.wipe_case(s.commitment, signed, s.trusted, server=server, egress_mib=egress)


@pytest.mark.slow
def test_the_counted_check_agrees(counted_server):
    stats = D.run(counted_port(sampled()), examples=400, server=counted_server)
    assert stats.counts.get("accept", 0) > 5 and stats.counts.get("undecided", 0) > 3 and stats.rejects > 100, stats


def _both(port: D.ProtocolPort, server, case: dict) -> dict:
    """The Lean's answer on `case`, which is the Python's."""
    lean = D.replay(port, case, server)
    assert port.key(lean) == port.key(port.python(case)), (lean, port.python(case))
    return json.loads(lean[1])


def test_a_marked_wipe_is_certified_on_its_count_and_nothing_else_is(counted_server, uncounted_server):
    """One wipe with a start mark and a count: the counted check certifies at that count where the wipe's own check
    certifies nothing, and without the count, without the mark or without the signature it certifies nothing either."""
    params = sw.WipeParams(8, 3, 1, 1_000, (("gpu", 3 * sw.LABEL_BYTES), ("host-dram", 5 * sw.LABEL_BYTES)))
    marked = WC.run(params, M.test_signer(M.SECRETS[0], 1, 1), certificate=CERTIFICATE)
    unmarked = WC.run(params, M.test_signer(M.SECRETS[0], 1, 0))
    assert marked.signed.start_mark == hashlib.sha512(CERTIFICATE).digest() and unmarked.signed.start_mark == b""
    counted, uncounted = counted_port(st.none()), uncounted_port(st.none())

    def case(s: WC.Session, egress: int | None, **kw) -> dict:
        return C.wipe_case(s.commitment, kw.pop("signed", s.signed), s.trusted, egress_mib=egress, **kw)

    a = _both(counted, counted_server, case(marked, 2))
    assert a["verdict"] == "accept" and a["protocol"] == C.COUNTED_TEST_ID, a
    assert a["public"] == WP.certify(params, 2) and a["public"]["certified"] and a["public"]["egress_mib"] == 2, a
    assert a["profile"]["gaps"] == [] and a["profile"] == WP.counted_profile(params, 2), a
    b = _both(uncounted, uncounted_server, case(marked, 2))
    assert b["verdict"] == "accept" and b["public"] == WP.uncounted(params) and not b["public"]["certified"], b
    assert b["profile"]["gaps"] == WP.writer_gap(2) != [] and b["commitment"] == a["commitment"], b

    saturated = _both(counted, counted_server, case(marked, sw.EGRESS_MAX_UNITS))
    assert saturated["verdict"] == "accept" and not saturated["public"]["certified"], saturated
    assert saturated["profile"]["gaps"] == WP.gaps(params, sw.EGRESS_MAX_UNITS) != [], saturated

    tampered = dataclasses.replace(marked.signed, fill_start_ns=marked.signed.fill_start_ns - 1)
    for name, c, verdict, reason in [
            ("no count", case(marked, None), "reject", WP.NO_COUNT),
            ("no mark", case(unmarked, 2), "reject", WP.NO_MARK),
            ("unsigned", case(marked, 2, signed=None), "undecided", WP.UNSIGNED),
            ("tampered", case(marked, 2, signed=tampered), "reject", "the signature does not verify"),
            ("out of range", case(marked, -1), "reject", WP.EGRESS_RANGE.format(-1, sw.EGRESS_MAX_UNITS)),
            ("another server", case(marked, 2, server=b"another node"), "reject", WP.OTHER)]:
        got = _both(counted, counted_server, c)
        assert (got["verdict"], got["reason"], got["public"]) == (verdict, reason, None), (name, got)
    # the wipe's own check takes the two the counted check refuses for their count
    for c in (case(marked, None), case(unmarked, 2)):
        assert _both(uncounted, uncounted_server, c)["verdict"] == "accept"


@st.composite
def named_inputs(draw) -> dict:
    """A sampled session's commitment, its signed session or none, and a sampled egress; now and then a field dropped,
    an egress that is no integer, a commitment `WipeParams` refuses, a session that is no object, or a batch record past
    a field's width, which leaves the session no message."""
    s = draw(WC.sessions())
    j = {"commitment": s.commitment.to_json(),
         "signed": draw(st.sampled_from([s.signed.to_json(), s.signed.to_json(), None])), "egress_mib": draw(WC.EGRESS)}
    how = draw(st.sampled_from(["as-is"] * 8 + ["dropped", "egress", "labels", "signed", "wide"]))
    if how == "dropped":
        del j[draw(st.sampled_from(sorted(j)))]
    elif how == "egress":
        j["egress_mib"] = draw(st.sampled_from([True, "1", [1], {"mib": 1}]))
    elif how == "labels":
        j["commitment"] = j["commitment"] | {"labels": 0}
    elif how == "signed":
        j["signed"] = draw(st.sampled_from([1, "signed", []]))
    elif how == "wide" and j["signed"] is not None and j["signed"]["batches"]:
        first, *rest = j["signed"]["batches"]
        j["signed"] = j["signed"] | {"batches": [first | {"b": 2 ** 32}, *rest]}
    return j


def test_named_agrees(named_server):
    stats = D.run(named_port(named_inputs()), examples=200, server=named_server)
    assert stats.outputs > 100 and stats.rejects > 10, stats


def test_named_is_the_sessions_marks_and_the_evidences_count(named_server):
    """On a session with a start mark, `named` is the certificate's SHA-512, the session's start and end messages and
    the count as given, in range or not; it is none while the session is unsigned or the evidence gives no count, and
    refused where a batch record is past its field's width."""
    params = sw.WipeParams(8, 3, 1, 1_000, (("gpu", 8 * sw.LABEL_BYTES),))
    s = WC.run(params, M.test_signer(M.SECRETS[0], 1, 1), certificate=CERTIFICATE)
    port = named_port(st.none())
    c, signed = s.commitment.to_json(), s.signed.to_json()
    for egress in (0, 7, -1, 10 ** 15):
        want = {"egress_mib": egress, "end_message": sw.end_message(s.signed).hex(),
                "start_mark": hashlib.sha512(CERTIFICATE).hexdigest(), "start_message": sw.start_message(s.commitment).hex()}
        value = {"commitment": c, "signed": signed, "egress_mib": egress}
        assert D.replay(port, value, named_server) == ("output", D.canonical(want)), egress
        assert named(value) == want
    for value in ({"commitment": c, "signed": None, "egress_mib": 7}, {"commitment": c, "signed": signed, "egress_mib": None}):
        assert D.replay(port, value, named_server) == ("output", D.canonical(None)), value
        assert named(value) is None
    wide = signed | {"batches": [signed["batches"][0] | {"b": 2 ** 32}, *signed["batches"][1:]]}
    assert D.replay(port, {"commitment": c, "signed": wide, "egress_mib": 7}, named_server)[0] == "reject"
    with pytest.raises(OverflowError):
        named({"commitment": c, "signed": wide, "egress_mib": 7})
    # with no count nothing reads the session's message, and both give none
    assert D.replay(port, {"commitment": c, "signed": wide, "egress_mib": None}, named_server) == ("output", D.canonical(None))
