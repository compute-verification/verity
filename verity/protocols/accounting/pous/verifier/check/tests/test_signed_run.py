"""`pous-signed-run serve` (`PousCheck.SignedRun`) against the Python verifier of the same case (`answer_of`: the case
decoded as the Lean decodes it, then `period.answer_signed`, which is `run.check_signed_at` and the signed run's profile):
every field of the answer, and the reason of each refusal of the check.

The cases are a small store set up at the small point (`period.point_for(size, W, C0)`), its session run by the memory
challenger, the setup record signed by the session's key under the session's id for the published outputs the test hands
it, and one opening per round made from the prover's table (`period.openings_of`). Under `--test` the session's key is a
test signer's (its setup record at the next leaf); under the deployed protocol it is a key at S1's deployed parameters
(`mc_cases.deployed_signer`, which signs at leaf 0 alone, so both records are signed there). Each case is honest or under
one edit that one of the check's refusals names: the point's, the session's or the setup's published outputs; the
openings (one missing or extra, a row, a path, the opening of a round whose index an earlier round drew); a prover that
answers every other round with zeros, or no trusted key; the setup record signed under another id, by another key, with
its signature off, or holding other published outputs; and the case's JSON (a setup record with a `passed` key beside
its fields, an opening's hex in capitals)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from dataclasses import dataclass
from functools import cache

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from test_run import LEAN, M, _decode_session, _decode_setup, _point_of
from verity_difftest import protocol as P
from verity.protocols.accounting.pous.device import challenger_session
from verity.protocols.accounting.pous.device import memory_challenger as mc
from verity.protocols.accounting.pous.verifier import period
from verity.protocols.accounting.pous.verifier import run as R

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EXE = "pous-signed-run"
TEST_ID = mc.TEST_PREFIX + period.SIGNED_PROTOCOL
STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(2000)
COINS = hashlib.sha512(b"the period's live coins").digest()


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


def _setup(b: bytes) -> R.Setup:
    return M._exactly(_decode_setup, P.canonical_json(b))


def _signed_setup(j) -> mc.SignedSetupRecord:
    try:
        return mc.SignedSetupRecord.from_json(j)
    except (ValueError, TypeError, KeyError) as e:
        raise D.Reject(str(e)) from None


def _evidence(b: bytes) -> tuple:
    j = P.canonical_json(b)
    openings = []
    for o in M.list_(j, "paths"):
        openings.append(R.Opening(M.hex_(o, "row"), tuple(M.hexes(o, "path"))))
    session, setup = _decode_session(M._get(j, "session")), _signed_setup(M._get(j, "setup"))
    _canonical(j, {"paths": [o.to_json() for o in openings], "session": session.to_json(), "setup": setup.to_json()})
    return session, setup, tuple(openings)


def answer_of(protocol: str, schemes: dict):
    """`PousCheck.SignedRun.mkProtocol protocol schemes`'s answer in Python."""
    def check(p, published, e) -> tuple:
        (point, sid, trusted), (session, setup, openings) = p, e
        a = period.answer_signed(point, schemes, trusted, published, session, setup, openings, sid, protocol)
        return (a["verdict"], a["public"], a["profile"]) if a["verdict"] == "accept" else (a["verdict"], a["reason"])

    def answer(case) -> dict:
        return P.answer(protocol, case, params=_params, commitment=_setup, evidence=_evidence, check=check,
                        digest=lambda s: hashlib.sha512(D.canonical(s.to_json()).encode()).digest())
    return answer


TEST = D.ProtocolPort(EXE, answer_of(TEST_ID, M.ts.SCHEMES), st.nothing(), package=LEAN, serve=("--test", "serve"))
DEPLOYED = D.ProtocolPort(EXE, answer_of(period.SIGNED_PROTOCOL, mc.SCHEMES), st.nothing(), package=LEAN)
test_server = D.server_fixture(TEST)
deployed_server = D.server_fixture(DEPLOYED)


@cache
def held() -> period.EncodedStore:
    return period.setup_store(STORE, COINS, period.point_for(len(STORE), period.W, period.C0))


@dataclass(frozen=True)
class Signed:
    """A signed run's parts: `case` is its protocol's case."""

    point: R.Point
    session_id: bytes
    trusted: tuple[bytes, ...]
    published: R.Setup
    session: R.ChallengePhaseRecord
    setup: mc.SignedSetupRecord
    openings: tuple[R.Opening, ...]

    def case(self) -> dict:
        return period.case_signed(self.point, self.session_id, self.trusted, self.published, self.session, self.setup,
                                  self.openings)


def _zeros_on_odd_rounds(blocks):
    return lambda i: bytes(len(blocks[i])) if i % 2 else blocks[i]


def signed_run(signer, coins: bytes = COINS, prover=period.honest, setup_signer=None, setup_session: bytes | None = None,
               **record) -> Signed:
    """The memory challenger's session on `held()` under `coins` (as `period.audit` runs it), signed by `signer`, whose
    key the verifier trusts; the setup record signed by
    `setup_signer` (by default the session's key, at its next leaf for a test key) under `setup_session` (by default the
    session's id) for the published outputs, each overridden by `record`; and the openings from the prover's table."""
    h, pub = held(), held().published
    q = h.point.params
    sid = challenger_session.coin_session_id(coins, period.SESSION_DOMAIN, "n0", pub.root)
    challenger = mc.MemoryChallenger(mc.CoinTree.draw(coins, sid, q.blocks, h.point.k), signer, log=M.ts.session_log(),
                                     limit_ns=h.point.limit_ns, block_bytes=q.block_bytes, binding=pub.root,
                                     clock=M.Clock(10 ** 9))
    signed = challenger.run(prover(h.blocks))
    session = R.ChallengePhaseRecord(signed, challenger.commitment, challenger.openings())
    if setup_signer is None:
        setup_signer = dataclasses.replace(signer, q=signer.q + 1) if signer.scheme != mc.LMS else signer
    fields = dict(blocks=pub.blocks, commitment=pub.commitment, salt=pub.salt, root=pub.root) | record
    setup = mc.sign_setup_record(setup_signer, sid if setup_session is None else setup_session, **fields)
    openings = tuple(period.openings_of(h.point, pub.salt, h.digests, session.signed.rounds))
    return Signed(h.point, sid, (signer.public_key,), pub, session, setup, openings)


def _repeat(s: Signed) -> int:
    """The first round whose index an earlier round drew."""
    seen = set()
    for j, r in enumerate(s.session.signed.rounds):
        if r.index in seen:
            return j
        seen.add(r.index)
    raise AssertionError("no round draws an index twice")


def _opening(s: Signed, j: int, f) -> Signed:
    o = list(s.openings)
    o[j] = f(o[j])
    return dataclasses.replace(s, openings=tuple(o))


def _commitment(s: Signed, **kw) -> Signed:
    return dataclasses.replace(s, session=dataclasses.replace(s.session, commitment=dataclasses.replace(
        s.session.commitment, **kw)))


def _setup_json(case: dict, **kw) -> dict:
    ev = json.loads(bytes.fromhex(case["evidence"]))
    ev["setup"] |= kw
    return case | {"evidence": D.canonical(ev).encode().hex()}


DECODER = "evidence: "
"""The reason prefix of a case the decoders refuse: the Lean's and the Python's words for it differ."""


def edited(signer) -> dict[str, tuple[dict, str | None]]:
    """The cases of `signer`'s run, honest and under each edit, with the reason each is refused for (`None`: accepted;
    `""`: the verdict's, not checked here; `DECODER`: the decoders')."""
    s = signed_run(signer)
    q, pub, k, j = s.point.params, s.published, len(s.openings), _repeat(s)
    flip = M._flip
    stranger = M.test_signer(M.STRANGER, 2, 0)
    other_id = hashlib.sha512(b"another session").digest()[:len(s.session_id)]
    point = s.point
    cases: dict[str, tuple[Signed | dict, str | None]] = {
        "honest": (s, None),
        "point-blocks": (dataclasses.replace(s, point=dataclasses.replace(
            point, params=dataclasses.replace(q, segments=q.segments + 1))),
            f"the setup published {pub.blocks} blocks, not the point's {q.blocks_per_segment * (q.segments + 1)}"),
        "salt-short": (dataclasses.replace(s, published=dataclasses.replace(pub, salt=pub.salt[:-1])),
                       f"the setup's salt is {len(pub.salt) - 1} bytes, not {R.SALT_BYTES}"),
        "session-blocks": (_commitment(s, blocks=q.blocks + 1),
                           f"the session audits {q.blocks + 1} blocks, not the setup's {q.blocks}"),
        "point-k": (dataclasses.replace(s, point=dataclasses.replace(point, k=point.k + 1)),
                    f"the session runs {point.k} rounds, not the point's {point.k + 1}"),
        "point-limit": (dataclasses.replace(s, point=dataclasses.replace(point, limit_ns=s.session.commitment.limit_ns - 1)),
                        f"the session's limit of {s.session.commitment.limit_ns} ns is past the point's Δ + RTT = "
                        f"{s.session.commitment.limit_ns - 1} ns"),
        "session-block-bytes": (_commitment(s, block_bytes=q.block_bytes + 1),
                                f"the session's answers are {q.block_bytes + 1} bytes, not the point's {q.block_bytes}"),
        "session-binding": (_commitment(s, binding=flip(pub.root, 5)),
                            "the session is not bound to the root the setup published"),
        "session-id": (dataclasses.replace(s, session_id=other_id),
                       "the session does not run under the id the auditor designated"),
        "opening-missing": (dataclasses.replace(s, openings=s.openings[:-1]),
                            f"the case holds {k - 1} openings, not one for each of the session's {k} rounds"),
        "opening-extra": (dataclasses.replace(s, openings=s.openings + s.openings[:1]),
                          f"the case holds {k + 1} openings, not one for each of the session's {k} rounds"),
        "row": (_opening(s, 3, lambda o: R.Opening(flip(o.row, 7), o.path)),
                "round 3's opening is not a row under the root the setup published"),
        "path": (_opening(s, 4, lambda o: R.Opening(o.row, (flip(o.path[0], 9), *o.path[1:]))),
                 "round 4's opening is not a row under the root the setup published"),
        "path-short": (_opening(s, 5, lambda o: R.Opening(o.row, o.path[:-1])),
                       "round 5's opening is not a row under the root the setup published"),
        "path-long": (_opening(s, 6, lambda o: R.Opening(o.row, (*o.path, o.path[-1]))),
                      "round 6's opening is not a row under the root the setup published"),
        "repeat-row": (_opening(s, j, lambda o: R.Opening(flip(o.row, 11), o.path)),
                       f"round {j}'s opening is not a row under the root the setup published"),
        "zeros": (signed_run(signer, prover=_zeros_on_odd_rounds), ""),
        "untrusted": (dataclasses.replace(s, trusted=()), ""),
        "setup-other-session": (signed_run(signer, setup_session=other_id),
                                "the setup record is not signed by the session's key under the session's id"),
        "setup-stranger": (signed_run(signer, setup_signer=stranger),
                           "the setup record is not signed by the session's key under the session's id"),
        "setup-signature": (dataclasses.replace(s, setup=dataclasses.replace(s.setup, signature=flip(s.setup.signature, 300))),
                            "the setup record is not signed by the session's key under the session's id"),
        "setup-blocks": (signed_run(signer, blocks=pub.blocks + 1),
                         f"the setup record holds {pub.blocks + 1} blocks, not the published {pub.blocks}"),
        "setup-commitment": (signed_run(signer, commitment=flip(pub.commitment, 1)),
                             "the setup record's commitment to W is not the published one"),
        "setup-salt": (signed_run(signer, salt=flip(pub.salt, 2)), "the setup record's salt is not the published one"),
        "setup-root": (signed_run(signer, root=flip(pub.root, 3)), "the setup record's root is not the published one"),
    }
    out = {n: (c.case(), r) for n, (c, r) in cases.items()}
    honest_case = out["honest"][0]
    out["setup-passed"] = (_setup_json(honest_case, passed=1), DECODER)
    ev = json.loads(bytes.fromhex(honest_case["evidence"]))
    ev["paths"][0]["row"] = ev["paths"][0]["row"].upper()
    out["row-capitals"] = (honest_case | {"evidence": D.canonical(ev).encode().hex()}, DECODER)
    return out


def _agree(port, server, cases: dict[str, tuple[dict, str | None]]) -> dict[str, dict]:
    got = {}
    for name, (case, reason) in cases.items():
        python = port.reference(case)
        lean = json.loads(D.replay(port, case, server)[1])
        if reason == DECODER:
            assert lean["reason"].startswith(DECODER) and python["reason"].startswith(DECODER), (name, lean, python)
            lean, python = (dict(a, reason=None) for a in (lean, python))
        assert lean == python, (name, lean["reason"], python["reason"])
        assert lean["verdict"] == ("accept" if reason is None else "reject"), (name, lean)
        if reason and reason != DECODER:
            assert lean["reason"] == reason, (name, lean["reason"])
        got[name] = lean
    return got


def test_each_refusal_agrees(test_server) -> None:
    """Under `--test`: the honest run is accepted with nothing certified and the signed run's two gaps after the point's;
    each edit is refused, by the Lean and the Python alike and in the same words, for the refusal it names. The honest
    session samples one block in two rounds, and the opening of the later round is checked as its own."""
    cases = edited(M.test_signer(M.SECRETS[0], 2, 1))
    got = _agree(TEST, test_server, cases)
    honest = got["honest"]
    assert honest["protocol"] == TEST_ID and honest["public"] == {"certified": False, "eps": "0", "s_bits": 0}, honest
    assert honest["profile"]["gaps"][-2:] == list(period.SIGNED_GAPS) and len(period.SIGNED_GAPS) == 2, honest


def test_the_deployed_signer_agrees(deployed_server, test_server) -> None:
    """Under the deployed protocol, signed at S1's deployed parameters: the honest run is accepted, its profile v0's
    with the point's gap and the signed run's two, and a setup record holding another root is refused. The test protocol
    refuses the deployed signer, and the deployed one a test signer."""
    signer = M.deployed_signer()
    pub = held().published
    cases = {"honest": (signed_run(signer).case(), None),
             "setup-root": (signed_run(signer, root=M._flip(pub.root, 3)).case(),
                            "the setup record's root is not the published one")}
    got = _agree(DEPLOYED, deployed_server, cases)
    assert got["honest"]["profile"]["gaps"] == [
        f"P2 has no certificate at (w, c0, blocks) = ({period.W}, {period.C0}, {got['honest']['profile']['blocks']})",
        *period.SIGNED_GAPS]
    v0 = json.loads(json.dumps(period.profile(held().point)))
    assert got["honest"]["profile"] | {"gaps": None} == v0 | {"gaps": None}
    assert json.loads(D.replay(TEST, cases["honest"][0], test_server)[1])["verdict"] == "reject"
    test_case = signed_run(M.test_signer(M.SECRETS[0], 2, 1)).case()
    assert json.loads(D.replay(DEPLOYED, test_case, deployed_server)[1])["verdict"] == "reject"


def test_setup_accepted_is_the_certificates_hypothesis() -> None:
    """At the deployed point, where P2 has a certificate, the signed run's profile is v0's but for its gaps and its
    certificate's hypotheses, which end with the open premise `SETUP_ACCEPTED` (the Lean's `#guard`s pin its own)."""
    signed, v0 = period.profile_signed(R.DEPLOYED), period.profile(R.DEPLOYED)
    assert signed["certificate"]["hypotheses"] == [*v0["certificate"]["hypotheses"], period.SETUP_ACCEPTED], signed
    assert signed["certificate"] | {"hypotheses": None} == v0["certificate"] | {"hypotheses": None}
    assert signed["gaps"] == [*v0["gaps"], *period.SIGNED_GAPS]


@pytest.mark.slow
def test_drawn_sessions_agree(test_server) -> None:
    """Sessions under flipped coins on the one setup, honest or under one bit off in one opening's row or path, or in the
    setup record's signature."""
    signer = M.test_signer(M.SECRETS[0], 2, 1)

    def draw(seed: int, edit: int, x: int) -> dict:
        s = signed_run(signer, coins=hashlib.sha512(b"coins %d" % seed).digest())
        j = x % len(s.openings)
        if edit == 1:
            s = _opening(s, j, lambda o: R.Opening(M._flip(o.row, x), o.path))
        elif edit == 2:
            s = _opening(s, j, lambda o: R.Opening(o.row, tuple(M._flip(p, x) if i == x % len(o.path) else p
                                                                 for i, p in enumerate(o.path))))
        elif edit == 3:
            s = dataclasses.replace(s, setup=dataclasses.replace(s.setup, signature=M._flip(s.setup.signature, x)))
        return s.case()

    port = D.ProtocolPort(EXE, TEST.reference, st.builds(draw, st.integers(0, 2 ** 16), st.integers(0, 3),
                                                         st.integers(0, 2 ** 16)), package=LEAN, serve=("--test", "serve"))
    stats = D.run(port, examples=40, server=test_server)
    assert stats.counts.get("accept", 0) > 3 and stats.counts.get("reject", 0) > 10, stats
