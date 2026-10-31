"""The wipe as a core protocol (`MemoryChallenger.WipeProtocol`, `verity/wipe/v0`): `wipe-protocol-test serve` against
`wipe_protocol_case.test_answer` (the period hook's `outcome` behind the Lean's decoders) on drawn sessions at every batch
size, unsigned, under one adversarial edit (`wipe_cases.MUTATIONS`), one JSON edit or params of another server or
session; and `wipe-protocol serve` under S1's LMS on each named wipe (`wipe_protocol_case.EXPECT`): an honest one, a
segment certified at `StreamedWipeMeetsReads`' threshold at `k = 2152` and one at `k = 5231` with memory past the fill, a
prover keeping one label past the threshold of foreign content, a late read and a session never signed, each with the
verdict it is aimed at; and a commitment of no reads, refused by both as they decode it."""

from __future__ import annotations

import json
import shutil

import pytest
from hypothesis import assume
from hypothesis import strategies as st

import mc_cases as M
import verity_difftest as D
import wipe_cases as WC
import wipe_protocol_case as C
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_period as WP

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def signers_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("wipe-protocol-test", C.test_answer, cases, package=M.PACKAGE)


def deployed_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("wipe-protocol", C.answer, cases, package=M.PACKAGE)


signers_server = D.server_fixture(signers_port(st.none()))
deployed_server = D.server_fixture(deployed_port(st.none()))


def _json_edit(c: dict, f, x: int) -> dict:
    """`wipe_cases.JSON_MUTATIONS`' edit `f` of the record a case holds, back in the case: its commitment, its signed
    session and its params' trusted keys (an edit elsewhere in the record leaves the case as it was)."""
    j = {"commitment": json.loads(bytes.fromhex(c["commitment"])),
         "signed": json.loads(bytes.fromhex(c["evidence"]))["signed"], "trusted": c["params"]["trusted"], "schemes": []}
    e = f(j, x)
    return {"commitment": D.canonical(e.get("commitment", j["commitment"])).encode().hex(),
            "evidence": D.canonical({"signed": e.get("signed", j["signed"])}).encode().hex(),
            "params": c["params"] | {"trusted": e.get("trusted", j["trusted"])}}


#: One edit of a case's own framing: params extra, missing or not hex, evidence that isn't `{"signed": …}`.
CASE_EDITS = [
    lambda c, x: c | {"params": c["params"] | {"extra": x}},
    lambda c, x: c | {"params": {k: v for k, v in c["params"].items() if k != sorted(c["params"])[x % 3]}},
    lambda c, x: c | {"params": c["params"] | {"session": c["params"]["session"].upper()}},
    lambda c, x: c | {"params": c["params"] | {"trusted": "00"}},
    lambda c, x: c | {"evidence": D.canonical({"signed": None, "more": x}).encode().hex()},
    lambda c, x: c | {"evidence": D.canonical([None]).encode().hex()},
    lambda c, x: c | {"evidence": b'{"signed": null}'.hex()},
    lambda c, x: c | {"commitment": c["commitment"][:-1]},
    lambda c, x: {k: v for k, v in c.items() if k != "evidence"},
]


def _in_a_case(s: str) -> bool:
    """Whether a case's canonical JSON (`codec.canonical_json`) can hold the string: ASCII without DEL."""
    return s.isascii() and "\x7f" not in s


@st.composite
def drawn(draw) -> dict:
    s = draw(WC.sessions(names=WC.CASE_TIER_NAMES))
    x = draw(st.integers(0, 2 ** 16))
    how = draw(st.sampled_from(["as-is", "as-is", "mutated", "mutated", "unsigned", "json", "case", "other"]))
    if how == "mutated":
        s = draw(st.sampled_from(WC.MUTATIONS))(s, x)
        assume(_in_a_case(s.signed.scheme) and _in_a_case(s.signed.ended))
    signed = None if how == "unsigned" else s.signed
    if how == "other":
        server, session = draw(st.sampled_from([(s.commitment.server + b"x", None), (None, M._flip(s.commitment.session, x))]))
        return C.wipe_case(s.commitment, signed, s.trusted, server=server, session=session)
    c = C.wipe_case(s.commitment, signed, s.trusted)
    if how == "json":
        return _json_edit(c, draw(st.sampled_from(WC.JSON_MUTATIONS)), x)
    if how == "case":
        return draw(st.sampled_from(CASE_EDITS))(c, x)
    return c


@pytest.mark.slow
def test_the_test_protocol_agrees(signers_server):
    stats = D.run(signers_port(drawn()), examples=400, server=signers_server)
    assert stats.counts.get("accept", 0) > 20 and stats.counts.get("undecided", 0) > 10 and stats.rejects > 150, stats


def test_the_named_wipes(deployed_server):
    """Each named wipe once, under S1's LMS: the Lean's answer is the Python's, with the verdict it is aimed at."""
    port = deployed_port(st.none())
    seen = {}
    for m, want in C.EXPECT.items():
        c = C.build(m)
        lean = D.replay(port, c, deployed_server)
        python = port.python(c)
        assert port.key(lean) == port.key(python), (m, lean, python)
        seen[m] = json.loads(lean[1])
        assert seen[m]["verdict"] == want, (m, seen[m])
        if m not in ("params", "commitment", "evidence"):
            assert seen[m]["reason"] == C.answer(c)["reason"], (m, seen[m])
    m = 1 << 28
    for name, k, r, threshold, memory in [("segment", 2152, 24, 11_709_069, m),
                                          ("reads", 5231, 59, 5_074_062 + (8 << 20), m + (8 << 20))]:
        p = seen[name]["public"]
        assert (p["certified"], p["k"], p["r"], p["memory_bits"], p["eps"]) == (True, k, r, memory, f"1/{2 ** 128}"), p
        assert p["foreign_bits"] == threshold + p["egress_mib"] * sw.EGRESS_UNIT_BITS, p
        assert seen[name]["profile"]["gaps"] == [] and seen[name]["profile"]["certificate"]["reads"] == k
    small = seen["honest"]["public"]
    assert small["certified"] and small["k"] == WP.READS
    assert small["foreign_bits"] - small["egress_mib"] * sw.EGRESS_UNIT_BITS < small["memory_bits"]
    assert "wrong answers" in seen["foreign"]["reason"] and "past its deadline" in seen["late"]["reason"]
    assert seen["unsigned"]["reason"] == WP.UNSIGNED and seen["other-server"]["reason"] == WP.OTHER
    assert seen["untrusted"]["reason"] == "the session is signed by a key that is not a trusted memory challenger's"
    assert seen["tampered"]["reason"] == "the signature does not verify"
    assert seen["commitment"]["commitment"] is None and seen["evidence"]["commitment"] is not None


def test_a_wipe_of_no_reads_is_refused_as_it_decodes(deployed_server):
    """A commitment of `k = 0` timed reads, which `gaps` would leave uncertified, never reaches the check: the Lean and
    the Python refuse it as they decode it, before its digest."""
    port = deployed_port(st.none())
    c = C.build("honest")
    c["commitment"] = D.canonical(json.loads(bytes.fromhex(c["commitment"])) | {"k": 0}).encode().hex()
    lean = D.replay(port, c, deployed_server)
    assert port.key(lean) == port.key(port.python(c)), lean
    for a in (json.loads(lean[1]), C.answer(c)):
        assert (a["verdict"], a["commitment"]) == ("reject", None) and "1 <= batch <= k" in a["reason"], a


def test_each_kind_gets_its_verdict(deployed_server):
    """`case(kind)` for the composition's difftest: each kind's answer, Lean and Python, is the verdict it names."""
    port = deployed_port(st.none())
    for kind, (c, a) in C.cases(1).items():
        lean = D.replay(port, c, deployed_server)
        assert port.key(lean) == port.key(port.python(c)), (kind, lean)
        assert a["verdict"] == json.loads(lean[1])["verdict"] == C.verdict_of(kind), (kind, a)


def test_a_test_signed_wipe_is_never_the_deployed_protocols(deployed_server):
    s = WC.run(sw.WipeParams(8, 2, 1, 100, (("gpu", 512),)), M.test_signer(M.SECRETS[0], 1, 0))
    c = C.wipe_case(s.commitment, s.signed, s.trusted)
    port = deployed_port(st.none())
    lean = D.replay(port, c, deployed_server)
    assert port.key(lean) == port.key(port.python(c))
    assert json.loads(lean[1])["reason"].startswith("signature scheme 'test/lms-sha512/n64-w1-h1/v0' is not accepted")
