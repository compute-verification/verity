"""The wipe as a core protocol (`MemoryChallenger.WipeProtocol`, `verity/wipe/v0`): `wipe-protocol-test serve` against
`wipe_protocol_case.test_answer` (the period hook's `outcome` behind the Lean's decoders) on sampled sessions at every batch
size, unsigned, under one adversarial edit (`wipe_cases.MUTATIONS`), one JSON edit or params of another server or
session, each with a sampled egress in its evidence (`wipe_cases.EGRESS`: none, in range, or outside it); and
`wipe-protocol --server server-4 --fill 0 serve` (the deployed command's one form of `serve`) under S1's LMS on each named wipe (`wipe_protocol_case.EXPECT`): an honest one, a
segment at `k = 2152` (`r = 24`) and one at `k = 5231` (`r = 59`) with memory past the fill, one whose evidence gives an
egress of 0 MiB, each accepted and not certified, its count named as the evidence writer's, one whose evidence gives no
egress (accepted, not certified) and one whose egress is out of range (rejected), a
prover keeping one label past the threshold of foreign content, a late read and a session never signed, each with the
verdict it is aimed at; a commitment of no reads, refused by both as they decode it; and `--server NAME --fill BYTES
serve`, the served road's, which refuses by name a case of another server than the served node NAME, and one whose
labels fill less than the fill floor BYTES, and answers the served node's as `serve` does; and the deployed command's
refusal by name of `serve` or `check` with no served node and fill."""

from __future__ import annotations

import json
import shutil
import subprocess

import mc_cases as M
import pytest
import verity_difftest as D
import wipe_cases as WC
import wipe_protocol_case as C
from hypothesis import strategies as st

from verity.protocols.accounting.memory.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.memory.pous.verifier import wipe_period as WP

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def signers_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("wipe-protocol-test", C.test_answer, cases, package=M.PACKAGE)


def deployed_port(cases: st.SearchStrategy) -> D.ProtocolPort:
    return D.ProtocolPort("wipe-protocol", C.served_answer, cases, package=M.PACKAGE,
                          serve=("--server", C.SERVER.decode(), "--fill", "0", "serve"))


signers_server = D.server_fixture(signers_port(st.none()))
deployed_server = D.server_fixture(deployed_port(st.none()))


def _json_edit(c: dict, f, x: int) -> dict:
    """`wipe_cases.JSON_MUTATIONS`' edit `f` of the record a case holds, back in the case: its commitment, its signed
    session and its params' trusted keys (an edit elsewhere in the record leaves the case as it was)."""
    ev = json.loads(bytes.fromhex(c["evidence"]))
    j = {"commitment": json.loads(bytes.fromhex(c["commitment"])),
         "signed": ev["signed"], "trusted": c["params"]["trusted"], "schemes": []}
    e = f(j, x)
    return {"commitment": D.canonical(e.get("commitment", j["commitment"])).encode().hex(),
            "evidence": D.canonical(ev | {"signed": e.get("signed", j["signed"])}).encode().hex(),
            "params": c["params"] | {"trusted": e.get("trusted", j["trusted"])}}


#: One edit of a case's own framing: params extra, missing or not hex, evidence that isn't
#: `{"egress_mib": …, "signed": …}`, or whose egress is no integer.
CASE_EDITS = [
    lambda c, x: c | {"params": c["params"] | {"extra": x}},
    lambda c, x: c | {"params": {k: v for k, v in c["params"].items() if k != sorted(c["params"])[x % 3]}},
    lambda c, x: c | {"params": c["params"] | {"session": c["params"]["session"].upper()}},
    lambda c, x: c | {"params": c["params"] | {"trusted": "00"}},
    lambda c, x: c | {"evidence": D.canonical({"egress_mib": None, "signed": None, "more": x}).encode().hex()},
    lambda c, x: c | {"evidence": D.canonical([None]).encode().hex()},
    lambda c, x: c | {"evidence": b'{"egress_mib": null, "signed": null}'.hex()},
    lambda c, x: c | {"evidence": D.canonical({"signed": json.loads(bytes.fromhex(c["evidence"]))["signed"]}).encode().hex()},
    lambda c, x: c | {"evidence": D.canonical(json.loads(bytes.fromhex(c["evidence"]))
                                              | {"egress_mib": [True, str(x), [x], {"mib": x}][x % 4]}).encode().hex()},
    lambda c, x: c | {"commitment": c["commitment"][:-1]},
    lambda c, x: {k: v for k, v in c.items() if k != "evidence"},
]


@st.composite
def sampled(sample) -> dict:
    s = sample(WC.sessions())
    x = sample(st.integers(0, 2 ** 16))
    how = sample(st.sampled_from(["as-is", "as-is", "mutated", "mutated", "unsigned", "unsigned", "json", "case", "other"]))
    if how == "mutated":
        s = sample(st.sampled_from(WC.MUTATIONS))(s, x)
    signed = None if how == "unsigned" else s.signed
    egress = sample(WC.EGRESS)
    if how == "other":
        server, session = sample(st.sampled_from([(s.commitment.server + b"x", None), (None, M._flip(s.commitment.session, x))]))
        return C.wipe_case(s.commitment, signed, s.trusted, server=server, session=session, egress_mib=egress)
    c = C.wipe_case(s.commitment, signed, s.trusted, egress_mib=egress)
    if how == "json":
        return _json_edit(c, sample(st.sampled_from(WC.JSON_MUTATIONS)), x)
    if how == "case":
        return sample(st.sampled_from(CASE_EDITS))(c, x)
    return c


@pytest.mark.slow
def test_the_test_protocol_agrees(signers_server):
    stats = D.run(signers_port(sampled()), examples=400, server=signers_server)
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
            assert seen[m]["reason"] == C.served_answer(c)["reason"], (m, seen[m])
    m = 1 << 28
    for name, k, r, memory in [("segment", 2152, 24, m), ("reads", 5231, 59, m + (8 << 20))]:
        p = seen[name]["public"]
        assert p == {"certified": False, "egress_mib": sw.EGRESS_MAX_UNITS, "eps": "0", "foreign_bits": memory, "k": k,
                     "memory_bits": memory, "r": r}, p
        assert seen[name]["profile"]["certificate"]["reads"] == k and seen[name]["profile"]["certificate"]["r"] == r
    assert seen["honest"]["public"]["k"] == WP.READS
    for name in ("honest", "segment", "reads", "zero-egress"):
        evidence = json.loads(bytes.fromhex(C.build(name)["evidence"]))
        assert sorted(evidence) == ["egress_mib", "signed"] and "egress_bits" not in evidence["signed"], name
        params = WC.commitment_of(json.loads(bytes.fromhex(C.build(name)["commitment"]))).params
        assert seen[name]["public"] == WP.uncounted(params), name
        assert seen[name]["profile"]["gaps"] == WP.writer_gap(evidence["egress_mib"]) == [
            (f"the egress count is {evidence['egress_mib']} MiB, the evidence writer's, which no network certifier "
             "signed: no wipe is certified on it until the check that composes the wipe with the network certifier")], name
    assert json.loads(bytes.fromhex(C.build("zero-egress")["evidence"]))["egress_mib"] == 0
    assert not seen["zero-egress"]["public"]["certified"]
    none = seen["uncounted"]
    assert json.loads(bytes.fromhex(C.build("uncounted")["evidence"]))["egress_mib"] is None
    assert none["public"] == WP.certify(WC.commitment_of(json.loads(bytes.fromhex(C.build("uncounted")["commitment"]))).params,
                                        sw.EGRESS_MAX_UNITS)
    assert not none["public"]["certified"] and none["public"]["egress_mib"] == sw.EGRESS_MAX_UNITS
    assert none["public"]["foreign_bits"] == none["public"]["memory_bits"] and none["public"]["eps"] == "0"
    assert none["profile"]["gaps"] == [(f"the egress count is {sw.EGRESS_MAX_UNITS} MiB, outside 0 to "
                                        f"{sw.EGRESS_MAX_UNITS - 1}: it bounds nothing")]
    outside = json.loads(bytes.fromhex(C.build("egress-range")["evidence"]))["egress_mib"]
    assert not 0 <= outside <= sw.EGRESS_MAX_UNITS
    assert seen["egress-range"]["reason"] == WP.EGRESS_RANGE.format(outside, sw.EGRESS_MAX_UNITS)
    assert "wrong answers" in seen["foreign"]["reason"] and "past its deadline" in seen["late"]["reason"]
    assert seen["unsigned"]["reason"] == WP.UNSIGNED and seen["other-server"]["reason"] == C.OTHER_SERVER.format(C.SERVER.decode())
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
    for a in (json.loads(lean[1]), C.served_answer(c)):
        assert (a["verdict"], a["commitment"]) == ("reject", None) and "1 <= batch <= k" in a["reason"], a


def test_each_kind_gets_its_verdict(deployed_server):
    """`case(kind)` for the composition's difftest: each kind's answer, Lean and Python, is the verdict it names."""
    port = deployed_port(st.none())
    for kind, (c, a) in C.cases(1).items():
        lean = D.replay(port, c, deployed_server)
        assert port.key(lean) == port.key(port.python(c)), (kind, lean)
        assert a["verdict"] == json.loads(lean[1])["verdict"] == C.verdict_of(kind), (kind, a)


def test_a_test_signed_wipe_is_never_the_deployed_protocols(deployed_server):
    s = WC.run(sw.WipeParams(8, 2, 1, 100, (("gpu", 512),)), M.test_signer(M.SECRETS[0], 1, 0), server=C.SERVER)
    c = C.wipe_case(s.commitment, s.signed, s.trusted)
    port = deployed_port(st.none())
    lean = D.replay(port, c, deployed_server)
    assert port.key(lean) == port.key(port.python(c))
    assert json.loads(lean[1])["reason"].startswith("signature scheme 'test/lms-sha512/n64-w1-h1/v0' is not accepted")


OTHER_SERVER = C.OTHER_SERVER
#: `MemoryAttester.WipeProtocol.unheld`'s refusal, the deployed command's for `serve` or `check` with no served node
#: and fill floor
UNHELD = ("verity/wipe/v0: serve and check are held to the served node and the fill floor the card file gives: "
          "--server NAME --fill BYTES serve, or --server NAME --fill BYTES check PARAMS COMMITMENT EVIDENCE; there is "
          "no form without them")


def _served(exe: str, served: str, cases: list[dict], fill: int = 0) -> list[dict]:
    lines = "".join(D.canonical({k: c[k] for k in ("commitment", "evidence", "params")}) + "\n" for c in cases)
    p = subprocess.run([str(D.build(M.PACKAGE, exe)), "--server", served, "--fill", str(fill), "serve"], input=lines,
                       capture_output=True, text=True, check=True)
    return [json.loads(x) for x in p.stdout.splitlines()]


@pytest.mark.parametrize("exe", ["wipe-protocol", "wipe-protocol-test"])
def test_the_served_road_refuses_a_wipe_of_another_server_by_name(exe):
    """Held to the served node, a case whose params name another server is refused by name, before its signature or its
    reads; the served node's own case gets `serve`'s answer, accepted or undecided. Under `wipe-protocol` the honest
    wipe's case, under `wipe-protocol-test` a test signer's."""
    if exe == "wipe-protocol":
        honest, unsigned = C.build("honest"), C.build("unsigned")
    else:
        s = WC.run(sw.WipeParams(8, 2, 1, 100, (("gpu", 512),)), M.test_signer(M.SECRETS[0], 1, 0))
        honest, unsigned = C.wipe_case(s.commitment, s.signed, s.trusted), C.wipe_case(s.commitment, None, s.trusted)
    server = bytes.fromhex(honest["params"]["server"]).decode()
    other = server + "x"
    own = _served(exe, server, [honest, unsigned])
    assert [a["verdict"] for a in own] == ["accept", "undecided"], own
    assert own[1]["reason"] == WP.UNSIGNED
    refused = _served(exe, other, [honest, unsigned])
    for a, mine in zip(refused, own, strict=True):
        assert (a["verdict"], a["reason"]) == ("reject", OTHER_SERVER.format(other)), a
        assert a["commitment"] == mine["commitment"] and a["public"] is None


def test_the_served_road_refuses_a_wipe_below_the_fill_floor():
    """Held to a fill floor one byte above the honest wipe's, the honest wipe is refused by name before its
    signature or its reads; at its own fill it is accepted. The Lean and its Python twin answer alike at each."""
    honest = C.build("honest")
    filled = json.loads(bytes.fromhex(honest["commitment"]))["labels"] * sw.LABEL_BYTES
    for fill, verdict in ((filled + 1, "reject"), (filled, "accept")):
        [lean] = _served("wipe-protocol", C.SERVER.decode(), [honest], fill)
        assert lean == C.served_reference(C.ID, C.mc.SCHEMES, C.SERVER, fill)(honest), (fill, lean)
        assert lean["verdict"] == verdict, (fill, lean)
    [refused] = _served("wipe-protocol", C.SERVER.decode(), [honest], filled + 1)
    assert refused["reason"] == C.UNDER_FILLED.format(filled, filled + 1) and refused["public"] is None


def test_the_deployed_command_has_no_form_without_a_served_node(tmp_path):
    """`wipe-protocol serve` and `wipe-protocol check` on three files, without both `--server` and `--fill`, are
    refused by name (exit 64) before any case is read; `--server NAME --fill BYTES check` on the three files answers as
    `--server NAME --fill BYTES serve` does."""
    exe = str(D.build(M.PACKAGE, "wipe-protocol"))
    c = C.build("honest")
    files = []
    for k in ("params", "commitment", "evidence"):
        f = tmp_path / k
        f.write_bytes(D.canonical(c[k]).encode() if k == "params" else bytes.fromhex(c[k]))
        files.append(str(f))
    line = D.canonical({k: c[k] for k in ("commitment", "evidence", "params")}) + "\n"
    served = C.SERVER.decode()
    for args in (["serve"], ["check", *files], ["--server", "serve"], ["--server", served],
                 ["--server", served, "serve"], ["--server", served, "check", *files],
                 ["--server", served, "--fill", "-1", "serve"]):
        p = subprocess.run([exe, *args], input=line, capture_output=True, text=True)
        assert (p.returncode, p.stdout, p.stderr.strip()) == (64, "", UNHELD), (args, p)
    held = subprocess.run([exe, "--server", served, "--fill", "0", "check", *files], capture_output=True, text=True)
    assert held.returncode == 0 and json.loads(held.stdout) == _served("wipe-protocol", served, [c])[0]
