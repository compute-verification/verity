"""A lane's days in the auditor ledger: its `declaration` of a UTC day's program, signed at the challenger's own clock
before the day's first window opens; the gate refusing a second declaration of a day and, once a lane has declared, an
open on a day it hasn't; `verify_ledger`'s check of both; and `declared_day`, what a per-day check reads of a window."""

from __future__ import annotations

import hashlib
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from verity.protocols.spot.device import challenger as C
from verity.commitments.crypto import lms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

P = lms.Params(64, 4, 5)
T0 = datetime(2026, 10, 20, 9, 30, 0, tzinfo=UTC)
DAY0 = 20746
PIN = "c" * 40
FAMILIES = [["pouw/salt", 1], ["pouw/round", 3]]
SECRET = bytes(range(32))
PROGRAM, OTHER = "ab" * 64, "cd" * 64


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("vy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
    (tmp_path / "coin-ports").write_text("vy-nebius-1 7811-7899\n")
    (tmp_path / "pins.jsonl").write_text(A.pin_line(PIN, "d" * 40, "2026-10-20T09:00:00Z") + "\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")


@pytest.fixture
def signer(host):
    return A.key_signer(host, P, workers=1)


def hooks(host, signer, token="ab12", now=T0):
    active = lambda s: False  # noqa: E731
    return dict(now=now, token=token, own=lambda d: None, start=lambda s: None, active=active, secret=SECRET,
                ready=lambda s: None,
                reserve=lambda s: A.reserve(host, s, signer=signer, pinned=PIN, now=now),
                admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=now, active=active),
                declare=lambda args: A.declare(host, ["declare", *args], signer=signer, pinned=PIN, now=now))


def call(host, signer, words, key, **kw):
    """A lane key's request (`key`), or vy-admin's (`key` None)."""
    rc, out = (A.gate(["--ssh", words, "--key", key], host, **hooks(host, signer, **kw)) if key
               else A.gate(shlex.split(words), host, by="vy-admin", **hooks(host, signer, **kw)))
    return rc, json.loads(out)


RESERVE = f"reserve --peer vy-nebius-1 --scheme verity/test/coins/v1 --families '{json.dumps(FAMILIES)}'"


def reserve(host, signer, key, *, for_=None, **kw) -> str:
    return ok(call(host, signer, RESERVE + (f" --for {for_}" if for_ else ""), key, **kw))["session"]


def open_(host, signer, session, root, key, **kw):
    return call(host, signer, f"open --session {session} --root {root}", key, **kw)


def window(host, signer, root, key=None, *, token="ab12", for_=None, **kw) -> dict:
    """Reserve (vy-admin's `--for for_` when `key` is None) and open at `root`: the open's answer."""
    return ok(open_(host, signer, reserve(host, signer, key, for_=for_, token=token, **kw), root, key, **kw))


def declare(host, signer, key, args="", program=PROGRAM, **kw):
    return call(host, signer, f"declare --program {program} {args}".strip(), key, **kw)


def ok(answer):
    rc, doc = answer
    assert rc == 0, doc
    return doc


def refused(answer, why):
    rc, doc = answer
    assert rc == 3 and why in doc["refused"], doc


def key_of(host) -> C.ChallengerKey:
    return C.ChallengerKey(P.W, P.H, (host.key / "public").read_bytes())


def signed_at(host) -> dict:
    """What `_ledger_entry` reads of a v2 open beside its window and prover: its host, pinned commit and key."""
    return {"host": "vy-challenger-1.x", "pinned": PIN,
            "key_sha512": hashlib.sha512((host.key / "public").read_bytes()).hexdigest()}


def lms_verify(key: C.ChallengerKey, message: bytes, signature: bytes) -> bool:
    p = lms.Params(64, key.W, key.H)
    return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))


def checked(host, ledger=None):
    public = (host.key / "public").read_bytes()
    lines = A.read_ledger((ledger or host).ledger)
    return A.verify_ledger(lines, public[:16], lambda m, s: lms.verify(P, public, m, lms.Signature.decode(P, s)))


def day_read(host, root, ledger=None):
    lines = [ln.decode() for ln in A.read_ledger((ledger or host).ledger)]
    return C.declared_day(lines, {"root": root}, key_of(host), lms_verify)


def test_a_day_is_the_days_from_1970_to_a_times_utc_date_and_none_before_1970_or_off_form():
    assert A.day_of("1970-01-01T00:00:00Z") == 0 and A.day_of("1970-01-01T23:59:59Z") == 0
    assert A.day_of(A.utc(T0)) == DAY0 and A.day_of("2026-10-21T00:00:00Z") == DAY0 + 1
    assert A.day_of("2000-02-29T12:00:00Z") == 11016 and A.day_of("9999-12-31T23:59:59Z") == 2932896
    for t in ("1969-12-31T23:59:59Z", "2026-10-20 09:30:00", "2026-02-30T00:00:00Z", "2026-10-20T09:30:00", None, 20746):
        assert A.day_of(t) is None


def test_a_lane_declares_a_day_once_at_the_challengers_clock_and_the_same_request_again_gets_its_entry_back(host,
                                                                                                         signer):
    first = ok(declare(host, signer, "hash", "--blocks b0,b1"))
    assert first["kind"] == "declaration" and "earlier" not in first
    assert first["body"] == {"by": "hash", "blocks": ["b0", "b1"], "day": DAY0, "declared_utc": "2026-10-20T09:30:00Z",
                             "pinned": PIN, "program": PROGRAM}
    again = ok(declare(host, signer, "hash", "--blocks b0,b1"))
    assert again == {**first, "earlier": True} and len(A.read_ledger(host.ledger)) == 1
    refused(declare(host, signer, "hash", "--blocks b0,b1", program=OTHER),
            f"lane hash has declared day {DAY0} already (entry 0, program {PROGRAM[:16]}...): a lane declares a day once")
    refused(declare(host, signer, "hash", "--blocks b0"), "a lane declares a day once")
    refused(declare(host, signer, "hash", f"--day {DAY0} --blocks b1,b0"), "a lane declares a day once")
    assert ok(declare(host, signer, "vllm", program=OTHER))["body"]["blocks"] == []
    later = ok(declare(host, signer, "hash", f"--day {DAY0 + 2}", program=OTHER))
    assert later["body"]["day"] == DAY0 + 2 and later["body"]["declared_utc"] == "2026-10-20T09:30:00Z"
    refused(declare(host, signer, "hash", f"--day {DAY0 - 1}"),
            f"day {DAY0 - 1} is before today, day {DAY0} (2026-10-20): a day is declared before it ends")
    assert [e["seq"] for e in A.entries(A.read_ledger(host.ledger))] == [0, 1, 2]
    checked(host)


def test_once_a_lane_has_declared_it_opens_only_on_a_day_it_declared_and_a_lane_that_never_declares_opens_as_before(
        host, signer):
    window(host, signer, "01" * 64, "hash", token="0001")
    ok(declare(host, signer, "hash", f"--day {DAY0 + 1}"))
    s = reserve(host, signer, "hash", token="0002")
    n = len(A.read_ledger(host.ledger))
    refused(open_(host, signer, s, "02" * 64, "hash"),
            f"lane hash declares its days and hasn't declared day {DAY0} (2026-10-20), the day this open registers its "
            "window on: declare it first")
    assert len(A.read_ledger(host.ledger)) == n
    ok(declare(host, signer, "hash"))
    assert ok(open_(host, signer, s, "02" * 64, "hash"))["session"] == s
    window(host, signer, "03" * 64, "vllm", token="0003")
    checked(host)
    assert day_read(host, "02" * 64)["day"] == DAY0


def test_an_open_at_the_last_second_of_a_declared_day_is_admitted_and_one_at_the_next_midnight_refused(host, signer):
    eve = datetime(2026, 10, 20, 23, 59, 30, tzinfo=UTC)
    last = datetime(2026, 10, 20, 23, 59, 59, tzinfo=UTC)
    midnight = datetime(2026, 10, 21, 0, 0, 0, tzinfo=UTC)
    ok(declare(host, signer, "hash", now=eve))
    a = reserve(host, signer, "hash", token="0001", now=eve)
    b = reserve(host, signer, "hash", token="0002", now=eve)
    ok(open_(host, signer, a, "01" * 64, "hash", now=last))
    refused(open_(host, signer, b, "02" * 64, "hash", now=midnight),
            f"hasn't declared day {DAY0 + 1} (2026-10-21)")
    ok(declare(host, signer, "hash", "--blocks b0", program=OTHER, now=midnight))
    ok(open_(host, signer, b, "02" * 64, "hash", now=midnight))
    checked(host)
    assert day_read(host, "01" * 64) == {"day": DAY0, "declarations": [{"blocks": [], "day": DAY0, "program": PROGRAM}]}
    assert day_read(host, "02" * 64) == {"day": DAY0 + 1, "declarations": [
        {"blocks": [], "day": DAY0, "program": PROGRAM}, {"blocks": ["b0"], "day": DAY0 + 1, "program": OTHER}]}


def test_vy_admin_declares_for_a_lane_whose_windows_it_opens_and_no_lane_key_declares_for_another(host, signer):
    refused(declare(host, signer, "hash", "--for kernels"), "unrecognized arguments: --for kernels")
    rc, out = A.gate(["--ssh", f"declare --program {PROGRAM}"], host, **hooks(host, signer))
    assert rc == 3 and "this key reads only: a day is declared" in json.loads(out)["refused"]
    for lane in ("auditor", "vy-admin", "'a b'"):
        refused(declare(host, signer, None, f"--for {lane}"), "--for ")
    assert A.read_ledger(host.ledger) == []
    body = ok(declare(host, signer, None, f"--for kernels --day {DAY0 + 1}"))["body"]
    assert body["by"] == "vy-admin" and body["for"] == "kernels"
    s = reserve(host, signer, None, for_="kernels", token="0001")
    refused(open_(host, signer, s, "01" * 64, None), f"lane kernels declares its days and hasn't declared day {DAY0}")
    window(host, signer, "02" * 64, token="0002")
    assert ok(declare(host, signer, "kernels"))["body"]["by"] == "kernels"
    ok(open_(host, signer, s, "01" * 64, None))
    checked(host)
    assert [d["day"] for d in day_read(host, "01" * 64)["declarations"]] == [DAY0 + 1, DAY0]
    assert day_read(host, "02" * 64)["declarations"] == []


@pytest.mark.parametrize("args, why", [
    ("", "the following arguments are required: --program"),
    ("--program nope", "--program: the day's program's SHA-512"),
    (f"--program {PROGRAM.upper()}", "--program: the day's program's SHA-512"),
    (f"--program {PROGRAM[:-2]}", "--program: the day's program's SHA-512"),
    (f"--program {PROGRAM} --blocks b0,,b1", "--blocks: descriptor keys"),
    (f"--program {PROGRAM} --blocks ,", "--blocks: descriptor keys"),
    (f"--program {PROGRAM} --blocks 'b0 b1'", "--blocks: descriptor keys"),
    (f"--program {PROGRAM} --blocks {'b' * 257}", "--blocks: descriptor keys"),
    (f"--program {PROGRAM} --day 01", "--day '01'"),
    (f"--program {PROGRAM} --day -1", "--day '-1'"),
    (f"--program {PROGRAM} --day 12345678", "--day '12345678'"),
    (f"--program {PROGRAM} --day today", "--day 'today'"),
    (f"--program {PROGRAM} --lane kernels", "unrecognized arguments: --lane kernels"),
])
def test_a_declaration_the_gate_cant_check_is_refused_and_writes_nothing(host, signer, args, why):
    rc, doc = call(host, signer, f"declare {args}".strip(), "hash")
    refused((rc, doc), why)
    assert "\n" not in doc["refused"]
    assert A.read_ledger(host.ledger) == []


def test_vy_admins_blocks_are_capped_at_what_a_lanes_request_carries_and_the_longest_are_declared_whole(host, signer):
    """A lane's request over ssh is cut at 8192 bytes before its blocks are read; vy-admin's has no such cut."""
    refused(declare(host, signer, None, f"--blocks {','.join(['b' * 255] * 33)}"), "--blocks: descriptor keys")
    blocks = ",".join(["b" * 255] * 31 + ["b" * 256])
    assert len(blocks) == A.MAX_BLOCKS
    assert ok(declare(host, signer, None, f"--blocks {blocks}"))["body"]["blocks"] == blocks.split(",")
    checked(host)


@pytest.mark.parametrize("args", [
    ["declare", "hash", "hash", PROGRAM, "[]"],
    ["declare", "hash", "hash", PROGRAM, "[]", "-", "-"],
    ["reserve", "hash", "hash", PROGRAM, "[]", "-"],
    ["declare", "a b", "hash", PROGRAM, "[]", "-"],
    ["declare", "hash", "hash", PROGRAM.upper(), "[]", "-"],
    ["declare", "hash", "hash", PROGRAM, "nope", "-"],
    ["declare", "hash", "hash", PROGRAM, '{"b0": 1}', "-"],
    ["declare", "hash", "hash", PROGRAM, "[1]", "-"],
    ["declare", "hash", "hash", PROGRAM, '["b 0"]', "-"],
    ["declare", "hash", "hash", PROGRAM, "[]", "01"],
    ["declare", "hash", "hash", PROGRAM, "[]", "today"],
])
def test_the_auditors_declare_step_refuses_arguments_the_gate_would_not_pass(host, signer, args):
    with pytest.raises(A.Refused, match="declare: "):
        A.declare(host, args, signer=signer, pinned=PIN, now=T0)
    assert A.read_ledger(host.ledger) == []


def forge(host, signer, kind, session, body):
    I, sign, _ = signer
    A.append(host, kind, session, lambda lines: body, I, sign)


def declaration(**kw):
    return {"by": "hash", "blocks": ["b0"], "day": DAY0, "declared_utc": "2026-10-20T09:30:00Z", "pinned": PIN,
            "program": PROGRAM, **kw}


def reserve_body(**kw):
    return {"by": "hash", "peer": "vy-nebius-1", "scheme": "verity/test/coins/v1", "epoch": 0, "families": FAMILIES,
            "minutes": 10, "started_utc": "2026-10-20T09:30:00Z", "expires_utc": "2026-10-20T09:40:00Z", **kw}


def open_body(reserve_seq, **kw):
    return {"window": {"root": "01" * 64}, "reserve": reserve_seq,
            **{k: v for k, v in reserve_body().items() if k in A.RESERVED}, "opened_by": "hash",
            "started_utc": "2026-10-20T09:31:00Z", **kw}


S1 = "s20261020T093000Z-0001"
FORM = "a declaration is v2 and names its prover, its program"
FORGED = {
    "a day declared twice": ([declaration(), declaration(program=OTHER)], f"lane hash declares day {DAY0} twice"),
    "vy-admin's for a lane its lane declared": ([declaration(by="kernels"), declaration(by="vy-admin", **{"for": "kernels"})],
                                                f"lane kernels declares day {DAY0} twice"),
    "a day before its declared_utc's": ([declaration(day=DAY0 - 1)], FORM),
    "a declared_utc before 1970": ([declaration(day=0, declared_utc="1969-12-31T23:59:59Z")], FORM),
    "a declared_utc off form": ([declaration(declared_utc="2026-10-20 09:30:00")], FORM),
    "no declared_utc": ([{k: v for k, v in declaration().items() if k != "declared_utc"}], FORM),
    "a day that isn't natural": ([declaration(day=True)], FORM),
    "a negative day": ([declaration(day=-1, declared_utc="1970-01-01T00:00:00Z")], FORM),
    "no program": ([{k: v for k, v in declaration().items() if k != "program"}], FORM),
    "a program in uppercase": ([declaration(program=PROGRAM.upper())], FORM),
    "blocks as a string": ([declaration(blocks="b0")], FORM),
    "blocks of numbers": ([declaration(blocks=[0])], FORM),
    "no pinned commit": ([declaration(pinned=None)], FORM),
    "no prover": ([{k: v for k, v in declaration().items() if k != "by"}], FORM),
    "a for that isn't a name": ([declaration(**{"for": "a b"})], FORM),
}


@pytest.mark.parametrize("case", list(FORGED))
def test_the_check_refuses_a_signed_declaration_the_auditor_would_not_write(host, signer, case):
    bodies, why = FORGED[case]
    for body in bodies:
        forge(host, signer, "declaration", None, body)
    with pytest.raises(ValueError, match=why):
        checked(host)


def test_the_check_refuses_a_declaration_with_a_session(host, signer):
    forge(host, signer, "declaration", S1, declaration())
    with pytest.raises(ValueError, match="a session's entry has a session, a period's none"):
        checked(host)


def test_the_check_refuses_a_declared_lanes_open_on_a_day_it_hasnt_declared_or_on_no_day(host, signer):
    forge(host, signer, "declaration", None, declaration(day=DAY0 + 1))
    forge(host, signer, "declaration", None, declaration(by="vllm"))
    forge(host, signer, "reserve", S1, reserve_body())
    forge(host, signer, "open", S1, open_body(2))
    with pytest.raises(ValueError, match=f"entry 3: session {S1} opens on day {DAY0}, which its lane hash hasn't declared"):
        checked(host)

    other = A.Host(root=host.root.parent / "other", peers=host.peers, pins=host.pins)
    other.root.mkdir()
    forge(other, signer, "declaration", None, declaration())
    forge(other, signer, "open", S1, {"window": {"root": "01" * 64}, "by": "hash"})
    with pytest.raises(ValueError, match=f"entry 1: session {S1}'s open names no day \\(started_utc\\), and its lane hash "
                                         "declares its days"):
        checked(host, other)


def test_the_check_takes_a_declaration_after_the_lanes_open_and_another_lanes_declarations_bind_it_not(host, signer):
    """A late declaration leaves the day's earlier windows standing, and `declared_day` reads them as undeclared."""
    forge(host, signer, "reserve", S1, reserve_body())
    forge(host, signer, "open", S1, open_body(0, **signed_at(host)))
    forge(host, signer, "declaration", None, declaration())
    forge(host, signer, "declaration", None, declaration(by="vllm", day=DAY0 + 1))
    forge(host, signer, "reserve", "s20261020T093000Z-0002", reserve_body(by="vllm", **{"for": "kernels"}))
    forge(host, signer, "open", "s20261020T093000Z-0002",
          open_body(4, window={"root": "02" * 64}, by="vllm", opened_by="vllm", **{"for": "kernels"}, **signed_at(host)))
    assert set(checked(host)) == {"01" * 64, "02" * 64}
    assert day_read(host, "01" * 64) == {"day": DAY0, "declarations": []}
    assert day_read(host, "02" * 64) == {"day": DAY0, "declarations": []}


def test_declared_day_refuses_a_window_its_ledger_doesnt_open_on_a_day(host, signer):
    ok(declare(host, signer, "hash"))
    window(host, signer, "01" * 64, "hash", token="0001")
    with pytest.raises(C.CoinError, match="opens no challenger session for this window"):
        day_read(host, "02" * 64)
    with pytest.raises(C.CoinError, match="the window's identity is malformed"):
        C.declared_day([ln.decode() for ln in A.read_ledger(host.ledger)], {"root": "01"}, key_of(host), lms_verify)
    with pytest.raises(C.CoinError, match="the auditor's ledger is refused"):
        C.declared_day([ln.decode() for ln in A.read_ledger(host.ledger)][1:], {"root": "01" * 64}, key_of(host),
                       lms_verify)

    other = A.Host(root=host.root.parent / "other", peers=host.peers, pins=host.pins)
    other.root.mkdir()
    forge(other, signer, "open", S1, {"window": {"root": "01" * 64}, "by": "hash", **signed_at(host)})
    with pytest.raises(C.CoinError, match=f"opens this window's session {S1} on no day"):
        day_read(host, "01" * 64, other)
