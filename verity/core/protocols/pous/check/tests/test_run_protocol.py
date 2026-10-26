"""`pous-run-protocol serve` (`PousCheck.RunProtocol`, onsite Verity's PoUS component) against the Python verifier of the
same case (`answer_of`: the case decoded as the Lean decodes it, then `run.check_at` and `pous.period`'s `certify` and
`profile`): every field of the answer but its reason, the certified `s_bits`, `eps` and the profile's gaps among them.

Under `--test` (`testProtocol`, the test signers' schemes alone) the cases are `test_run`'s drawn runs, honest and under
one edit of the run, of its session or of its JSON, each as the protocol's case, and those under one edit of the case's
framing. Under the deployed protocol (S1's LMS alone) they are a small period of `pous.period.run`, signed by a key at
S1's deployed parameters (`mc_cases.deployed_signer`), honest and under a prover that answers every other round with
zeros; each protocol refuses the other's signer. The period's setup case is the setup protocol's (`pous-setup`), which
accepts it."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from test_run import LEAN, M, _decode_session, _decode_setup, _point_of, documents
from test_setup_difftest import setup_port, setup_protocol
from verity_difftest import protocol as P
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period

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
    digests, session = M.hex_(j, "digests"), _decode_session(M._get(j, "session"))
    _canonical(j, {"digests": digests.hex(), "session": session.to_json()})
    return digests, session


def answer_of(protocol: str, schemes: dict):
    """`PousCheck.RunProtocol.mkProtocol protocol schemes`'s answer in Python."""
    def check(p, setup, e) -> tuple:
        (point, sid, trusted), (digests, session) = p, e
        a = period.answer(point, schemes, trusted, setup, digests, session, sid, protocol)
        return (a["verdict"], a["public"], a["profile"]) if a["verdict"] == "accept" else (a["verdict"], a["reason"])

    def answer(case) -> dict:
        return P.answer(protocol, case, params=_params, commitment=_setup, evidence=_evidence, check=check,
                        digest=lambda s: hashlib.sha512(D.canonical(s.to_json()).encode()).digest())
    return answer


def case_of(doc) -> dict:
    """The protocol's case of a `pous-run` difftest line: the setup, `{"digests", "session"}`, and the rest but the
    schemes, which are the protocol's own."""
    d = doc if isinstance(doc, dict) else {}
    return {"commitment": D.canonical(d.get("setup")).encode().hex(),
            "evidence": D.canonical({k: d[k] for k in ("digests", "session") if k in d}).encode().hex(),
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

TEST = D.ProtocolPort(EXE, answer_of(TEST_ID, M.ts.SCHEMES), st.nothing(), package=LEAN, serve=("--test", "serve"))
DEPLOYED = D.ProtocolPort(EXE, answer_of(period.PROTOCOL, mc.SCHEMES), st.nothing(), package=LEAN)
test_server = D.server_fixture(TEST)
deployed_server = D.server_fixture(DEPLOYED)
setup_server = D.server_fixture(setup_port())

STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(2000)
COINS = hashlib.sha512(b"the period's live coins").digest()


def _zeros_on_odd_rounds(blocks):
    return lambda i: bytes(len(blocks[i])) if i % 2 else blocks[i]


def periods(signer) -> dict[str, period.Period]:
    """A small period on node `n0` signed by `signer`: honest, and under a prover that answers every other block with
    zeros."""
    kw = dict(signer=signer, schemes=mc.SCHEMES if signer.scheme == mc.LMS else M.ts.SCHEMES,
              protocol=period.PROTOCOL if signer.scheme == mc.LMS else TEST_ID)
    return {"honest": period.run("n0", STORE, COINS, log=M.ts.session_log(), clock=M.Clock(10 ** 9), **kw),
            "zeros": period.run("n0", STORE, COINS, log=M.ts.session_log(), clock=M.Clock(10 ** 9),
                                prover=_zeros_on_odd_rounds, **kw)}


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
    assert got["honest"]["profile"]["gaps"] == [f"P2 has no certificate at (w, c0, blocks per segment) = "
                                                f"({period.W}, {period.C0}, {got['honest']['profile']['blocks']})"]
    for name, p in test.items():
        assert _part(TEST.reference(p.case)) == _part(p.answer), name
        assert json.loads(D.replay(TEST, p.case, test_server)[1])["verdict"] == {"honest": "accept", "zeros": "reject"}[name]
        assert json.loads(D.replay(DEPLOYED, p.case, deployed_server)[1])["verdict"] == "reject"
    assert json.loads(D.replay(TEST, deployed["honest"].case, test_server)[1])["verdict"] == "reject"


def test_a_small_periods_setup_is_accepted(setup_server) -> None:
    p = periods(M.deployed_signer())["honest"]
    kind, text = D.replay(setup_port(), p.setup_case, setup_server)
    answer = json.loads(text)
    assert answer["verdict"] == "accept" and answer["public"]["root"] == _json(p.case["commitment"])["root"], answer
    assert _part(setup_protocol(p.setup_case)) == _part(answer)


@pytest.mark.slow
def test_drawn_runs_agree(test_server) -> None:
    """`test_run`'s drawn runs as cases under `--test`, and under one edit of the case's framing."""
    cases = documents().map(case_of)
    framed = st.builds(lambda c, m: m(c), st.builds(case_of, documents()), st.sampled_from(FRAMING))
    port = D.ProtocolPort(EXE, TEST.reference, st.one_of(cases, cases, framed), package=LEAN, serve=("--test", "serve"))
    stats = D.run(port, examples=300, server=test_server)
    assert stats.counts.get("accept", 0) > 20 and stats.counts.get("reject", 0) > 100, stats
