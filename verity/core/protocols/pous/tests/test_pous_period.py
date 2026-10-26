"""`pous.period`, the period hook, in Python: which points it certifies at (`certificate`, `certified_point`), its
encode split over processes, a node's store kept and reloaded, and what an audit at the certified point publishes. Its
answers against the Lean's are `check/tests/test_run_protocol.py`'s."""

from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction

import pytest
from pous_test_signer import SCHEMES, TestSigner, session_log
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous import run as R
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.pous.verifier import VerifierKey

TEST_ID = mc.TEST_PREFIX + period.PROTOCOL
SALT = hashlib.sha512(b"the setup's salt").digest()[:R.SALT_BYTES]
EPS = Fraction(1, 100) + Fraction(1, 2 ** 128)


class Clock:
    def __call__(self) -> int:
        return 10 ** 9


def test_the_certified_point_is_the_smallest_the_family_certifies():
    """`certified_point` holds the store at P2's registered width and prime, at least `2^8` blocks, the deployed `k` and
    `Δ + RTT`: `P2SlackFamilyUncond`'s range, where the run certifies `⌊ρ·blocks·w⌋` at `1% + 2^-128`. The registered
    point keeps its own certificate (8 segments), and the small point certifies nothing."""
    pb = 2054
    for size, blocks, segments in ((0, 256, 1), (256 * pb, 256, 1), (256 * pb + 1, 257, 1), (3 * 65_536 * pb, 196_608, 3)):
        pt = period.certified_point(size)
        q = pt.params
        assert (q.w, q.c0, q.blocks, q.segments, pt.k, pt.limit_ns) == (16448, 21065, blocks, segments, 105, 854_000)
        assert period.gaps(pt)[1] == [] and period.certificate(q).segments is None
        assert period.certify(pt) == {"certified": True, "eps": str(EPS), "s_bits": blocks * 16448 * 18 // 19}
    assert period.certify(period.certified_point(0))["s_bits"] == 3_989_072
    assert period.certificate(R.DEPLOYED.params) == P2.P2(R.DEPLOYED.params).certificate
    assert period.certificate(R.DEPLOYED.params).segments == 8
    small = period.point_for(2000)
    assert period.certify(small) == {"certified": False, "eps": "0", "s_bits": 0}
    assert period.gaps(small)[1] == [f"P2 has no certificate at (w, c0, blocks) = (1024, 105, {small.params.blocks})"]
    assert period.gaps(R.Point(R.DEPLOYED.params, 104, R.DEPLOYED.limit_ns))[1] == ["k = 104 is below the certificate's 105"]


def test_encode_in_processes_is_the_serial_encode():
    q = P2Params(128, 173, 8, 2)
    w = hashlib.shake_256(b"W").digest(q.w_bytes)
    serial = period.encode(q, SALT, w)
    assert serial == [P2.encode_segment(q, None, SALT, s, w[s * 112:(s + 1) * 112]) for s in range(2)]
    assert period.encode(q, SALT, w, workers=3) == serial


def test_a_kept_store_is_audited_again(tmp_path):
    """A store set up once and saved (`Store.save`) loads as it was, and each audit of it is a fresh session: two
    periods, each accepted, under session ids of their own."""
    held = period.setup_store(b"a period's records", hashlib.sha512(b"setup coins").digest())
    held.save(tmp_path / "pous-store")
    again = period.Store.load(tmp_path / "pous-store")
    assert again == held and len(again.blocks) == held.point.params.blocks
    signer, log = TestSigner(b"period", 3), session_log()
    got = [period.audit("n0", again, hashlib.sha512(b"coins %d" % i).digest(), signer=signer, log=log, schemes=SCHEMES,
                        protocol=TEST_ID, clock=Clock()) for i in range(2)]
    assert [p.verdict for p in got] == ["accept", "accept"] and got[0].case["params"] != got[1].case["params"]
    assert {p.s_bits for p in got} == {0}


def _certified_store() -> period.Store:
    """A `C*` of uniformly random canonical blocks at the certified point, with a setup case of its own: what the run
    checks of it (the table, its root and the session) doesn't read how it was encoded, and a 256-block encode is 16
    core-minutes. No setup accepts it (a random block is no codeword), so the tie refuses its audits."""
    pt = period.certified_point(0)
    q = pt.params
    blocks = tuple(b"\x00" + os.urandom(q.block_bytes - 1) for _ in range(q.blocks))
    digests = b"".join(mc.block_digest(b) for b in blocks)
    rows = [digests[64 * i:64 * (i + 1)] for i in range(q.blocks)]
    root, commitment = VerifierKey.from_row_digests(q, SALT, rows).root, hashlib.sha512(b"W").digest()
    return period.Store(pt, R.Setup(q.blocks, commitment, root, SALT), blocks, digests,
                        period.setup_case(q, commitment, SALT, [list(blocks)]))


def _run_of(case: dict) -> tuple:
    """`run.check_at`'s inputs from the run protocol's case, under the test signers' schemes."""
    ev, p = json.loads(bytes.fromhex(case["evidence"])), case["params"]
    return (period.point_of(p["point"]), SCHEMES, [bytes.fromhex(t) for t in p["trusted"]],
            R.Setup.from_json(json.loads(bytes.fromhex(case["commitment"]))), bytes.fromhex(ev["digests"]),
            R.Session.from_json(ev["session"]), bytes.fromhex(p["session_id"]))


def test_an_audit_at_the_certified_point_checks_its_run():
    """At the certified point the run's check accepts an honest session and refuses a prover that answers every other
    block with zeros, and what an accepted run there publishes is `s_bits = ⌊ρ·256·16448⌋` = 3,989,072 at
    `1% + 2^-128`, with no gap, under `P2SlackFamilyUncond`. This `C*` is no setup's, so the answer is the tie's refusal;
    an accepted, certified audit through the tie is `check/tests/test_run_protocol.py`'s (slow: a 256-block encode)."""
    held, signer = _certified_store(), TestSigner(b"certified", 3)
    kw = dict(signer=signer, log=session_log(), schemes=SCHEMES, protocol=TEST_ID, clock=Clock())
    honest = period.audit("n0", held, hashlib.sha512(b"coins").digest(), **kw)
    assert honest.verdict == "reject" and honest.answer["reason"].startswith("setup: "), honest.answer
    R.check_at(*_run_of(honest.case))
    assert period.certify(held.point) == {"certified": True, "eps": str(EPS), "s_bits": 3_989_072}
    prof = period.profile(held.point)
    assert prof["gaps"] == [] and prof["certificate"]["theorem"] == "Pous.SecurityProofs.P2SlackFamilyUncond"
    zeros = period.audit("n0", held, hashlib.sha512(b"coins 2").digest(),
                         prover=lambda blocks: lambda i: bytes(len(blocks[i])) if i % 2 else blocks[i], **kw)
    assert zeros.verdict == "reject" and (zeros.s_bits, zeros.eps) == (0, 0)
    with pytest.raises(R.Refused):
        R.check_at(*_run_of(zeros.case))
