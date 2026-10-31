"""`pous.period`, the period hook, in Python: which points it certifies at (`certificate`, `point_for`), its encode split
over processes, a node's store kept and reloaded, what an audit at the certified point publishes, and the records it
stores, every copy of the auditor's ledger in them cut to the entries their window cites (`cite`). Its answers
against the Lean's are `check/tests/test_run_protocol.py`'s."""

from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction

import pytest
from pous_fakes import Clock
from pous_test_signer import SCHEMES, TestSigner, session_log
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period
from verity.core.protocols.pous import run as R
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.pous.verifier import VerifierKey
from verity.ml.circuits import requirement

TEST_ID = mc.TEST_PREFIX + period.PROTOCOL
SALT = hashlib.sha512(b"the setup's salt").digest()[:R.SALT_BYTES]
EPS = Fraction(1, 100) + Fraction(1, 2 ** 128)


def test_the_default_point_is_the_smallest_the_family_certifies():
    """`point_for`'s default holds the store at P2's registered width and prime, at least `2^8` blocks, the deployed `k`
    and `Δ + RTT`: `P2SlackFamilyUncond`'s range, where the run certifies `⌊ρ·blocks·w⌋` at `1% + 2^-128`. The toy's
    6,240,150-byte store is 3,039 blocks there. The registered point keeps its own certificate (8 segments), and the
    small point (`point_for(size, W, C0)`) certifies nothing."""
    pb = 2054
    for size, blocks, segments in ((0, 256, 1), (256 * pb, 256, 1), (256 * pb + 1, 257, 1), (6_240_150, 3_039, 1),
                                   (3 * 65_536 * pb, 196_608, 3)):
        pt = period.point_for(size)
        q = pt.params
        assert (q.w, q.c0, q.blocks, q.segments, pt.k, pt.limit_ns) == (16448, 21065, blocks, segments, 105, 854_000)
        assert period.gaps(pt)[1] == [] and period.certificate(q).segments is None
        assert period.certify(pt) == {"certified": True, "eps": str(EPS), "s_bits": blocks * 16448 * 18 // 19}
    assert period.certify(period.point_for(0))["s_bits"] == 3_989_072
    assert period.certificate(R.DEPLOYED.params) == P2.P2(R.DEPLOYED.params).certificate
    assert period.certificate(R.DEPLOYED.params).segments == 8
    small = period.point_for(2000, period.W, period.C0)
    assert (small.params.w, small.params.c0, small.params.blocks) == (1024, 105, 16)
    assert period.certify(small) == {"certified": False, "eps": "0", "s_bits": 0}
    assert period.gaps(small)[1] == [f"P2 has no certificate at (w, c0, blocks) = (1024, 105, {small.params.blocks})"]
    assert period.gaps(R.Point(R.DEPLOYED.params, 104, R.DEPLOYED.limit_ns))[1] == ["k = 104 is below the certificate's 105"]


def test_demo_bs_store_is_certified_by_period_held_at():
    """Above the family's `2^23` blocks the run takes `periodHeldAt` (`p2.wide_certificate`) to `2^27`, in any number of
    segments: at Demo B's store, 1038 segments of `2^16` blocks, it certifies `PeriodHeldDemo`'s 1,060,008,348,186 bits
    at `1% + 2^-128` under `WIDE_CLAIMS`, refuses a `k` below 105 as the family does, and past `2^27` names the gap."""
    def at(bps: int, segments: int, k: int = 105) -> R.Point:
        return R.Point(P2Params(16448, 21065, bps, segments), k, R.DEPLOYED.limit_ns)

    demo = at(1 << 16, 1038)
    assert demo.params.blocks == 68_026_368
    assert period.certify(demo) == {"certified": True, "eps": str(EPS), "s_bits": 1_060_008_348_186}
    prof = period.profile(demo)
    assert prof["gaps"] == [] and prof["certificate"]["theorem"] == "Pous.Period.SecurityProofs.periodHeldAt"
    assert prof["certificate"]["claims"] == list(P2.WIDE_CLAIMS) and prof["certificate"]["segments"] is None
    assert period.point_for(1038 * 65_536 * 2054).params == demo.params
    assert period.gaps(at(1 << 16, 1038, 104))[1] == ["k = 104 is below the certificate's 105"]
    assert period.certificate(at(1 << 16, 128).params).theorem == "Pous.SecurityProofs.P2SlackFamilyUncond"
    for bps, segments in ((1 << 16, 129), ((1 << 16) - 1, 129), (1 << 16, 1 << 11)):
        assert period.certificate(at(bps, segments).params).theorem == "Pous.Period.SecurityProofs.periodHeldAt"
        assert period.certify(at(bps, segments))["s_bits"] == bps * segments * 16448 * 18 // 19
    past = at(1 << 16, (1 << 11) + 1)
    assert period.certify(past) == {"certified": False, "eps": "0", "s_bits": 0}
    assert period.gaps(past)[1] == [f"P2 has no certificate at (w, c0, blocks) = (16448, 21065, {(1 << 27) + (1 << 16)})"]


def test_encode_in_processes_is_the_serial_encode():
    q = P2Params(128, 173, 8, 2)
    w = hashlib.shake_256(b"W").digest(q.w_bytes)
    serial = period.encode(q, SALT, w)
    assert serial == [P2.encode_segment(q, None, SALT, s, w[s * 112:(s + 1) * 112]) for s in range(2)]
    assert period.encode(q, SALT, w, workers=3) == serial


def test_a_kept_store_is_audited_again(tmp_path):
    """A store set up once and saved (`EncodedStore.save`) loads as it was, and each audit of it is a fresh session: two
    periods, each accepted, under session ids of their own."""
    held = period.setup_store(b"a period's records", hashlib.sha512(b"setup coins").digest(),
                              period.point_for(18, period.W, period.C0))
    held.save(tmp_path / "pous-store")
    again = period.EncodedStore.load(tmp_path / "pous-store")
    assert again == held and len(again.blocks) == held.point.params.blocks
    signer, log = TestSigner(b"period", 3), session_log()
    got = [period.audit("n0", again, hashlib.sha512(b"coins %d" % i).digest(), signer=signer, log=log, schemes=SCHEMES,
                        protocol=TEST_ID, clock=Clock(10 ** 9)) for i in range(2)]
    assert [p.verdict for p in got] == ["accept", "accept"] and got[0].case["params"] != got[1].case["params"]
    assert {p.public("s_bits") for p in got} == {0}


def _certified_store() -> period.EncodedStore:
    """A `C*` of uniformly random canonical blocks at the certified point, with a setup case of its own: what the run
    checks of it (the table, its root and the session) doesn't read how it was encoded, and a 256-block encode is 16
    core-minutes. No setup accepts it (a random block is no codeword), so the tie refuses its audits."""
    pt = period.point_for(0)
    q = pt.params
    blocks = tuple(b"\x00" + os.urandom(q.block_bytes - 1) for _ in range(q.blocks))
    digests = b"".join(mc.block_digest(b) for b in blocks)
    rows = [digests[64 * i:64 * (i + 1)] for i in range(q.blocks)]
    root, commitment = VerifierKey.from_row_digests(q, SALT, rows).root, hashlib.sha512(b"W").digest()
    return period.EncodedStore(pt, R.Setup(q.blocks, commitment, root, SALT), blocks, digests,
                               period.setup_case(q, commitment, SALT, [list(blocks)]))


def test_an_encoded_store_reads_as_the_audits_store():
    """`EncodedStore.store` is `C*` as `requirement.Store`: every block registered and resident at every step, since
    nothing is written or freed after setup, and no payloads."""
    held = _certified_store()
    n = held.point.params.blocks
    assert held.store == requirement.Store(n, {}, {}, {})
    assert held.store.resident(0) == held.store.resident(n) == frozenset(range(len(held.blocks)))


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
    kw = dict(signer=signer, log=session_log(), schemes=SCHEMES, protocol=TEST_ID, clock=Clock(10 ** 9))
    honest = period.audit("n0", held, hashlib.sha512(b"coins").digest(), **kw)
    assert honest.verdict == "reject" and honest.answer["reason"].startswith("setup: "), honest.answer
    R.check_at(*_run_of(honest.case))
    assert period.certify(held.point) == {"certified": True, "eps": str(EPS), "s_bits": 3_989_072}
    prof = period.profile(held.point)
    assert prof["gaps"] == [] and prof["certificate"]["theorem"] == "Pous.SecurityProofs.P2SlackFamilyUncond"
    zeros = period.audit("n0", held, hashlib.sha512(b"coins 2").digest(),
                         prover=lambda blocks: lambda i: bytes(len(blocks[i])) if i % 2 else blocks[i], **kw)
    assert zeros.verdict == "reject" and (zeros.public("s_bits"), zeros.eps) == (0, 0)
    with pytest.raises(R.Refused):
        R.check_at(*_run_of(zeros.case))


LANE, OTHER = "compute-accounting", "memory-accounting"


def _line(lines: list[str], kind: str, session: str | None, body: dict, sig_bytes: int) -> str:
    """The auditor's next ledger line in its form (`sampled_proofs.challenger.ledger_line`), under a stand-in signature of
    `sig_bytes` bytes at leaf `seq`: `cite` reads the form, never the signature."""
    e = {"body": body, "kind": kind, "prev": hashlib.sha512(lines[-1].encode()).hexdigest() if lines else "0" * 128,
         "schema": period.LEDGER_SCHEMAS[1], "seq": len(lines), "session": session}
    return json.dumps({**e, "sig": (len(lines).to_bytes(4, "big") + bytes(sig_bytes - 4)).hex()}, sort_keys=True,
                      separators=(",", ":"))


def _ledger(n: int, sig_bytes: int = 2048) -> tuple[list[str], str, list[int]]:
    """A ledger of `n` sessions in each of two lanes' periods, then a second period of `LANE`'s with `n` more: the ledger,
    the session a window cites (`LANE`'s middle one in its first period) and the indices of the entries it cites."""
    lines: list[str] = []
    at: dict[str, int] = {}

    def add(kind: str, session: str | None, body: dict) -> None:
        at[f"{kind} {session or body['period']}"] = len(lines)
        lines.append(_line(lines, kind, session, body, sig_bytes))

    for name, lanes, first in (("p-1", (LANE, OTHER), 0), ("p-2", (LANE,), n)):
        for lane in lanes:
            add("period-open", None, {"period": f"{name}-{lane}", "lane": lane, "key_sha512": "ab" * 64, "cap": 2 * n,
                                      "circuit": "cd" * 64})
        held: dict[str, list[str]] = {lane: [] for lane in lanes}
        for i in range(first, first + n):
            for j, lane in enumerate(lanes):
                s = f"s20261026T{i:06d}Z-{j:04x}"
                held[lane].append(s)
                add("reserve", s, {"by": lane, "minutes": 30})
                add("open", s, {"by": lane, "reserve": at[f"reserve {s}"], "window": {"root": f"{i:0128x}"}})
                add("end", s, {"closed": True, "complete": True, "files": {}})
        for lane in lanes:
            add("period-close", None, {"period": f"{name}-{lane}", "lane": lane, "complete": True,
                                       "windows": [{"root": None, "session": s, "complete": True} for s in held[lane]]})
    s = f"s20261026T{n // 2:06d}Z-0000"
    return lines, s, [at[k] for k in (f"period-open p-1-{LANE}", f"reserve {s}", f"open {s}", f"end {s}",
                                      f"period-close p-1-{LANE}")]


def _window(d, lines: list[str], s: str):
    """A live window's records as the torch loop writes them: the salt session's files beside infra's ledger head
    (`window_salt.issue`'s `dest`) and ``salt.json``'s ``live`` params, each with the whole ledger."""
    close = {"chain": "11" * 64, "ended": None, "lines": 3, "session": s, "signature": "22" * 40}
    files = {"period.json": json.dumps({"index": 0}).encode(), "pouw/registration.json": b'{"window": "w0"}',
             "pouw/salt-session/windows.jsonl": "".join(x + "\n" for x in lines).encode(),
             "pouw/salt-session/pins.jsonl": b'{"commit":"c","main":"m","pinned_utc":"t"}\n',
             f"pouw/salt-session/{s}/challenger.json": json.dumps(close).encode(),
             f"pouw/salt-session/{s}/releases.jsonl": b'{"event":"head"}\n',
             "pouw/salt.json": json.dumps({"answer": {"session": {"session": s}}, "live": {
                 "ledger": lines, "main": ["m"], "pins": ["p"],
                 "session": {"close": close, "hellos": [], "log": ['{"event":"head"}'], "record": {}}}},
                 sort_keys=True).encode(),
             "answers/pouw.json": b'{"verdict": "accept"}'}
    for name, data in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    return files


def _unframe(b: bytes) -> dict[str, bytes]:
    out, pos = {}, 0
    while pos < len(b):
        n = int.from_bytes(b[pos:pos + 4], "big")
        name, pos = b[pos + 4:pos + 4 + n].decode(), pos + 4 + n
        m = int.from_bytes(b[pos:pos + 8], "big")
        out[name], pos = b[pos + 8:pos + 8 + m], pos + 8 + m
    return out


def _whole(files: dict[str, bytes]) -> bytes:
    """The records framed whole, as `records` framed them before it cut the ledger."""
    return b"".join(len(k.encode()).to_bytes(4, "big") + k.encode() + len(v).to_bytes(8, "big") + v
                    for k, v in sorted(files.items()) if not k.startswith("answers/"))


def test_the_records_keep_only_the_ledger_entries_their_window_cites(tmp_path):
    """A live window's records hold the auditor's whole ledger twice, in the salt session's ``windows.jsonl`` and in
    ``salt.json``'s ``live``: the store keeps, in each, the cited session's reserve, open and end and its period's open
    and close, and every other file as it is, while the window's files on disk keep the whole ledger PoUW reads."""
    lines, s, kept = _ledger(200)
    files = _window(tmp_path, lines, s)
    store = period.records(tmp_path)
    got = _unframe(store)
    assert sorted(got) == sorted(k for k in files if not k.startswith("answers/"))
    want = [lines[i] for i in kept]
    assert got["pouw/salt-session/windows.jsonl"].decode().split("\n") == [*want, ""]
    salt, whole = json.loads(got["pouw/salt.json"]), json.loads(files["pouw/salt.json"])
    assert salt["live"]["ledger"] == want and salt == {**whole, "live": {**whole["live"], "ledger": want}}
    assert {k: v for k, v in got.items() if k not in ("pouw/salt-session/windows.jsonl", "pouw/salt.json")} == {
        k: v for k, v in files.items() if k in got and k not in ("pouw/salt-session/windows.jsonl", "pouw/salt.json")}
    assert {k: (tmp_path / k).read_bytes() for k in files} == files
    assert len(lines) == 1806 and len(store) * 100 < len(_whole(files))


def test_a_ledger_that_opens_no_cited_session_is_refused(tmp_path):
    """A copy of the ledger that doesn't open the session a close record beside it cites (a head older than the
    window's open) is refused, naming the file and the session: the store never holds a ledger cut short of what its
    window cites. A copy with no close record beside it cites nothing, and is kept whole."""
    lines, s, kept = _ledger(8)
    files = _window(tmp_path, lines, s)
    (tmp_path / "pouw/salt-session/windows.jsonl").write_text("".join(x + "\n" for x in lines[:kept[1] + 1]))
    with pytest.raises(ValueError, match=f"^pouw/salt-session/windows.jsonl: the auditor's ledger opens no session {s}"):
        period.records(tmp_path)
    (tmp_path / "pouw/salt-session/windows.jsonl").write_bytes(files["pouw/salt-session/windows.jsonl"])
    (tmp_path / "pouw/elsewhere").mkdir()
    (tmp_path / "pouw/elsewhere/windows.jsonl").write_bytes(files["pouw/salt-session/windows.jsonl"])
    got = _unframe(period.records(tmp_path))
    assert got["pouw/elsewhere/windows.jsonl"] == files["pouw/salt-session/windows.jsonl"]
    assert got["pouw/salt-session/windows.jsonl"].decode().split("\n") == [*(lines[i] for i in kept), ""]
    assert period.cited([json.loads(x) for x in lines], {s}) == kept
    with pytest.raises(ValueError, match="^no challenger session is cited beside the auditor's ledger"):
        period.cited([json.loads(x) for x in lines], set())


def test_pous_accepts_the_cut_records_as_it_does_the_whole(tmp_path):
    """The store's verdict doesn't read what it holds: PoUS's run at the small point accepts the cut records, as it
    accepts the whole, on fewer blocks."""
    lines, s, _ = _ledger(8, 64)
    files = _window(tmp_path, lines, s)
    signer, log = TestSigner(b"records", 4), session_log()
    point = {}
    for name, store in (("whole", _whole(files)), ("cut", period.records(tmp_path))):
        got = period.run("n0", store, hashlib.sha512(b"coins " + name.encode()).digest(), signer=signer, log=log,
                         point=period.point_for(len(store), period.W, period.C0), schemes=SCHEMES, protocol=TEST_ID,
                         clock=Clock(10 ** 9))
        assert got.verdict == "accept", got.answer
        point[name] = period.point_of(got.case["params"]["point"]).params.blocks
    assert point["cut"] < point["whole"]
