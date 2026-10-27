"""`pous-run-protocol serve` (`PousCheck.RunProtocol`, onsite Verity's PoUS component) against the Python verifier of the
same case (`answer_of`: the case decoded as the Lean decodes it, then `pous.period`'s `tie` and `run.check_at`, and its
`certify` and `profile`): every field of the answer but its reason, the certified `s_bits`, `eps` and the profile's gaps
among them.

Under `--test` (`testProtocol`, the test signers' schemes alone) the cases are `test_run`'s drawn runs over honest setups,
honest and under one edit of the run, of its session or of its JSON, each as the protocol's case with its setup's own
commitment and evidence, those under one edit of the case's framing, and those under one edit of the setup's evidence
(another setup's, a bit of its commitment, salt or `C*` off, `C*` cut short, the setup missing). Under the deployed
protocol (S1's LMS alone) they are a small period of `pous.period.run`, signed by a key at S1's deployed parameters
(`mc_cases.deployed_signer`), honest and under a prover that answers every other round with zeros; each protocol refuses
the other's signer; and the same two at the certified point (`period.certified_point`) on a store set up there, where
the honest one certifies its store (slow: a 256-block encode). The period's setup case is the setup protocol's
(`pous-setup`), which accepts it; the period's case with another store's setup (a run committed to a different setup) is
refused, for the tie's reason."""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from test_run import LEAN, WIDTHS, M, Run, _decode_session, _decode_setup, _point_of, documents, runs
from test_setup_difftest import _commitment as _setup_commitment
from test_setup_difftest import _evidence as _setup_evidence
from test_setup_difftest import _evidence_bytes, honest, setup_port, setup_protocol
from verity_difftest import protocol as P
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous import setup as pous_setup

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EXE = "pous-run-protocol"
TEST_ID = mc.TEST_PREFIX + period.PROTOCOL


@pytest.fixture(autouse=True, scope="module")
def _test_signer_dir(tmp_path_factory):
    """The test signer's session logs under the suite's basetemp (`pous_test_signer.DIR`)."""
    M.ts.DIR = tmp_path_factory.mktemp("test-signer")
    yield
    M.ts.DIR = None


def _canonical(j, holds: dict) -> None:
    if D.canonical(holds) != D.canonical(j):
        raise D.Reject("not the canonical JSON of what it holds (a field extra, or a constant not the protocol's)")


def _params(j) -> tuple:
    pt = _point_of(M._get(j, "point"))
    sid, trusted = M.hex_(j, "session_id"), tuple(M.hexes(j, "trusted"))
    _canonical(j, {"point": period.point_json(pt), "session_id": sid.hex(), "trusted": [t.hex() for t in trusted]})
    return pt, sid, trusted


def _setup(b: bytes):
    return M._exactly(_decode_setup, P.canonical_json(b))


def _evidence(b: bytes) -> tuple:
    j = P.canonical_json(b)
    digests, session, s = M.hex_(j, "digests"), _decode_session(M._get(j, "session")), M._get(j, "setup")
    commitment, salt = _setup_commitment(M.hex_(s, "commitment"))
    store = _setup_evidence(M.hex_(s, "evidence"))
    _canonical(j, {"digests": digests.hex(), "session": session.to_json(),
                   "setup": {"commitment": (commitment + salt).hex(), "evidence": _evidence_bytes(store).hex()}})
    return digests, session, (commitment, salt, store)


def answer_of(protocol: str, schemes: dict):
    """`PousCheck.RunProtocol.mkProtocol protocol schemes`'s answer in Python."""
    def check(p, setup, e) -> tuple:
        (point, sid, trusted), (digests, session, inputs) = p, e
        a = period.answer(point, schemes, trusted, setup, digests, session, sid, protocol, setup_inputs=inputs)
        return (a["verdict"], a["public"], a["profile"]) if a["verdict"] == "accept" else (a["verdict"], a["reason"])

    def answer(case) -> dict:
        return P.answer(protocol, case, params=_params, commitment=_setup, evidence=_evidence, check=check,
                        digest=lambda s: hashlib.sha512(D.canonical(s.to_json()).encode()).digest())
    return answer


def case_of(doc) -> dict:
    """The protocol's case of a `pous-run` difftest line: the setup's public outputs, `{"digests", "session", "setup"}`
    with the setup's own commitment and evidence from its case (`setup_of`, when the run has one), and the rest but the
    schemes, which are the protocol's own."""
    d = doc if isinstance(doc, dict) else {}
    of = d.get("setup_of") if isinstance(d.get("setup_of"), dict) else None
    evidence = {k: d[k] for k in ("digests", "session") if k in d}
    if of is not None:
        evidence["setup"] = {"commitment": of["commitment"], "evidence": of["evidence"]}
    return {"commitment": D.canonical(d.get("setup")).encode().hex(),
            "evidence": D.canonical(evidence).encode().hex(),
            "params": {k: d[k] for k in ("point", "session_id", "trusted") if k in d}}


def _json(h: str):
    return json.loads(bytes.fromhex(h))


#: One edit of a case's framing: its params, its evidence's keys or bytes, its commitment's bytes, a field missing.
FRAMING = [
    lambda c: c | {"params": c["params"] | {"schemes": [mc.LMS]}},
    lambda c: c | {"params": [c["params"]]},
    lambda c: c | {"evidence": json.dumps(_json(c["evidence"])).encode().hex()},
    lambda c: c | {"evidence": D.canonical(_json(c["evidence"]) | {"trusted": []}).encode().hex()},
    lambda c: c | {"commitment": json.dumps(_json(c["commitment"])).encode().hex()},
    lambda c: c | {"commitment": c["commitment"].upper()},
    lambda c: {k: v for k, v in c.items() if k != "commitment"},
]


def _setup_edit(f):
    """An edit of a case's setup evidence, `f(setup, x) -> setup` over `{"commitment", "evidence"}` as hex."""
    def edit(c, x):
        ev = _json(c["evidence"])
        if not isinstance(ev.get("setup"), dict):
            return c
        return c | {"evidence": D.canonical(ev | {"setup": f(ev["setup"], x)}).encode().hex()}
    return edit


def _flip_hex(h: str, x: int) -> str:
    b = bytearray(bytes.fromhex(h))
    if b:
        b[x % len(b)] ^= 1 << (x % 8)
    return b.hex()


def _another_setup(s: dict, x: int) -> dict:
    """Another honest setup's commitment and evidence, at the first reduced width."""
    w, c0 = WIDTHS[0]
    of = period.setup_case(*_store_case(honest(w, c0, 1 + x % 2, 1, 4 + x % 3)))
    return {"commitment": of["commitment"], "evidence": of["evidence"]}


def _store_case(stored: tuple) -> tuple:
    q, w, salt, encoded = stored
    return q, pous_setup.commit(q, w), salt, encoded


#: One edit of a case's setup evidence: another setup's, its commitment to `W` or salt or `C*` changed in a bit, `C*`
#: cut short, the setup missing.
SETUP_EDITS = [
    _setup_edit(_another_setup),
    _setup_edit(lambda s, x: s | {"commitment": _flip_hex(s["commitment"][:128], x) + s["commitment"][128:]}),
    _setup_edit(lambda s, x: s | {"commitment": s["commitment"][:128] + _flip_hex(s["commitment"][128:], x)}),
    _setup_edit(lambda s, x: s | {"evidence": _flip_hex(s["evidence"], x)}),
    _setup_edit(lambda s, x: s | {"evidence": s["evidence"][:-2 - 2 * (x % 4)]}),
    lambda c, x: c | {"evidence": D.canonical({k: v for k, v in _json(c["evidence"]).items() if k != "setup"})
                      .encode().hex()},
]
#: Honest setups `(q, W, salt, C*)` at the reduced widths, whose encodings the setup protocol accepts.
SETUPS = st.sampled_from(WIDTHS).flatmap(lambda wc: st.builds(honest, st.just(wc[0]), st.just(wc[1]), st.integers(1, 3),
                                                              st.integers(1, 2), st.integers(0, 3)))

TEST = D.ProtocolPort(EXE, answer_of(TEST_ID, M.ts.SCHEMES), st.nothing(), package=LEAN, serve=("--test", "serve"))
DEPLOYED = D.ProtocolPort(EXE, answer_of(period.PROTOCOL, mc.SCHEMES), st.nothing(), package=LEAN)
test_server = D.server_fixture(TEST)
deployed_server = D.server_fixture(DEPLOYED)
setup_server = D.server_fixture(setup_port())

STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(2000)
OTHER_STORE = hashlib.shake_256(b"another tiny MLP's weights").digest(2000)
COINS = hashlib.sha512(b"the period's live coins").digest()


def _zeros_on_odd_rounds(blocks):
    return lambda i: bytes(len(blocks[i])) if i % 2 else blocks[i]


def periods(signer, store: bytes = STORE) -> dict[str, period.Period]:
    """A small period of `store` on node `n0` signed by `signer`: honest, and under a prover that answers every other
    block with zeros."""
    kw = dict(signer=signer, schemes=mc.SCHEMES if signer.scheme == mc.LMS else M.ts.SCHEMES,
              protocol=period.PROTOCOL if signer.scheme == mc.LMS else TEST_ID)
    return {"honest": period.run("n0", store, COINS, log=M.ts.session_log(), clock=M.Clock(10 ** 9), **kw),
            "zeros": period.run("n0", store, COINS, log=M.ts.session_log(), clock=M.Clock(10 ** 9),
                                prover=_zeros_on_odd_rounds, **kw)}


def with_setup_of(case: dict, of: dict) -> dict:
    """`case` with its evidence's setup replaced by the setup case `of`'s commitment and evidence."""
    ev = _json(case["evidence"]) | {"setup": {"commitment": of["commitment"], "evidence": of["evidence"]}}
    return case | {"evidence": D.canonical(ev).encode().hex()}


def _part(answer: dict) -> dict:
    return {k: v for k, v in answer.items() if k != "reason"}


def test_a_small_periods_answers_agree(deployed_server, test_server) -> None:
    """The deployed protocol on the hook's small period signed at S1's deployed parameters: the hook's answer is the
    Python verifier's of its case and the Lean's, accepted with nothing certified (P2's certificate is at its deployed
    point alone, and the profile says so) and rejected for the answers that are not their blocks. The test protocol
    refuses the deployed signer, and the deployed one a test signer."""
    deployed, test = periods(M.deployed_signer()), periods(M.test_signer(M.SECRETS[0], 2, 1))
    got = {}
    for name, p in deployed.items():
        assert _part(DEPLOYED.reference(p.case)) == _part(p.answer), name
        got[name] = json.loads(D.replay(DEPLOYED, p.case, deployed_server)[1])
    assert {n: a["verdict"] for n, a in got.items()} == {"honest": "accept", "zeros": "reject"}, got
    assert got["honest"]["public"] == {"certified": False, "eps": "0", "s_bits": 0}, got
    assert got["honest"]["profile"]["gaps"] == [f"P2 has no certificate at (w, c0, blocks) = "
                                                f"({period.W}, {period.C0}, {got['honest']['profile']['blocks']})"]
    for name, p in test.items():
        assert _part(TEST.reference(p.case)) == _part(p.answer), name
        assert json.loads(D.replay(TEST, p.case, test_server)[1])["verdict"] == {"honest": "accept", "zeros": "reject"}[name]
        assert json.loads(D.replay(DEPLOYED, p.case, deployed_server)[1])["verdict"] == "reject"
    assert json.loads(D.replay(TEST, deployed["honest"].case, test_server)[1])["verdict"] == "reject"


@pytest.fixture(scope="module")
def certified_store() -> period.Store:
    """`STORE` set up at the certified point (`period.certified_point`, 256 blocks), its `C*` encoded in up to
    `period.SETUP_WORKERS` processes: about 16 core-minutes, since no `C*` that setup accepts there is cheaper than
    encoding one (a random block decodes to a payload with probability 2^-16)."""
    workers = min(period.SETUP_WORKERS, len(os.sched_getaffinity(0)))
    return period.setup_store(STORE, COINS, period.certified_point(len(STORE)), workers=workers)


@pytest.mark.slow
def test_a_certified_periods_answers_agree(deployed_server, certified_store) -> None:
    """The deployed protocol at the certified point (`period.certified_point`, P2's proved family at 256 blocks), on a
    store set up there: the Lean and the Python accept the honest period, its setup tied, with `s_bits` = 3,989,072 at
    `1% + 2^-128` and no gap, under `P2SlackFamilyUncond`, and reject the prover that answers every other block with
    zeros. The one test of an accepted, certified run through the tie: the quick tier's certified-point checks hold the
    run's check and `certify` there on a `C*` no setup accepts."""
    held = certified_store
    kw = dict(signer=M.deployed_signer(), schemes=mc.SCHEMES, protocol=period.PROTOCOL, clock=M.Clock(10 ** 9))
    got = {}
    for name, prover in (("honest", period.honest), ("zeros", _zeros_on_odd_rounds)):
        p = period.audit("n0", held, COINS, log=M.ts.session_log(), prover=prover, **kw)
        assert _part(DEPLOYED.reference(p.case)) == _part(p.answer), name
        got[name] = json.loads(D.replay(DEPLOYED, p.case, deployed_server)[1])
        assert _part(got[name]) == _part(p.answer), name
    assert {n: a["verdict"] for n, a in got.items()} == {"honest": "accept", "zeros": "reject"}, got
    assert got["honest"]["public"] == {"certified": True, "eps": f"{2 ** 126 + 25}/{25 * 2 ** 128}", "s_bits": 3_989_072}
    assert got["honest"]["profile"]["gaps"] == [] and got["honest"]["profile"]["certificate"]["segments"] is None


def test_a_small_periods_setup_is_accepted(setup_server) -> None:
    p = periods(M.deployed_signer())["honest"]
    kind, text = D.replay(setup_port(), p.setup_case, setup_server)
    answer = json.loads(text)
    assert answer["verdict"] == "accept" and answer["public"]["root"] == _json(p.case["commitment"])["root"], answer
    assert _part(setup_protocol(p.setup_case)) == _part(answer)


def test_a_run_committed_to_another_setup_is_refused(deployed_server) -> None:
    """The setup tie, under the deployed protocol: the honest small period's case with another store's honest setup in
    its evidence (setup accepts it, at the same point, and publishes another root) is refused because the run's
    commitment is not what that setup published; with that setup's commitment as well, the run's table and session
    (bound to its own root) are refused by the run's check; with its own setup's `C*` one bit off, by setup."""
    mine, other = periods(M.deployed_signer())["honest"], periods(M.deployed_signer(), OTHER_STORE)["honest"]
    assert mine.case["params"]["point"] == other.case["params"]["point"]
    assert mine.case["commitment"] != other.case["commitment"]
    w, salt, store = period.setup_inputs(mine.setup_case)
    flipped = {"commitment": (w + salt).hex(),
               "evidence": _evidence_bytes([[M._flip(b, 3) if (s, j) == (0, 0) else b for j, b in enumerate(seg)]
                                            for s, seg in enumerate(store)]).hex()}
    cases = {"other-setup": with_setup_of(mine.case, other.setup_case),
             "other-commitment": with_setup_of(mine.case | {"commitment": other.case["commitment"]}, other.setup_case),
             "setup-flipped": with_setup_of(mine.case, flipped)}
    got = {}
    for name, c in cases.items():
        python = DEPLOYED.reference(c)
        lean = json.loads(D.replay(DEPLOYED, c, deployed_server)[1])
        assert _part(lean) == _part(python), name
        assert lean["verdict"] == "reject", (name, lean)
        got[name] = lean["reason"], python["reason"]
    tie = "the run's commitment is not what its setup published"
    assert got["other-setup"] == (tie, tie), got
    assert all(r != tie and not r.startswith("setup: ") for r in got["other-commitment"]), got
    assert all(r.startswith("setup: ") for r in got["setup-flipped"]), got


@pytest.mark.slow
def test_drawn_runs_agree(test_server) -> None:
    """`test_run`'s drawn runs over honest setups (`SETUPS`) as cases under `--test`, and under one edit of the case's
    framing or of its setup's commitment and evidence."""
    rs = runs(stores=SETUPS)
    cases = documents(rs).map(case_of)
    framed = st.builds(lambda c, m: m(c), st.builds(case_of, documents(rs)), st.sampled_from(FRAMING))
    setup_edited = st.builds(lambda c, m, x: m(c, x), st.builds(case_of, st.builds(Run.doc, rs)),
                             st.sampled_from(SETUP_EDITS), st.integers(0, 2 ** 16))
    port = D.ProtocolPort(EXE, TEST.reference, st.one_of(cases, cases, cases, framed, setup_edited), package=LEAN,
                          serve=("--test", "serve"))
    stats = D.run(port, examples=400, server=test_server)
    assert stats.counts.get("accept", 0) > 20 and stats.counts.get("reject", 0) > 100, stats
