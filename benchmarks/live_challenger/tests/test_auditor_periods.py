"""Periods in the auditor ledger: `period-open` and `period-close`, who writes them, which windows a period holds, and
`verify_ledger`'s checks of their nesting and of each close's list."""

from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from verity.ml.crypto import lms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import auditor as A  # noqa: E402

P = lms.Params(64, 4, 4)
T0 = datetime(2026, 10, 20, 9, 30, 0, tzinfo=timezone.utc)
PIN = "c" * 40
FAMILIES = [["pouw/salt", 1], ["pouw/round", 3]]
SECRET = bytes(range(32))


@pytest.fixture
def host(tmp_path) -> A.Host:
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("vy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n")
    return A.Host(root=tmp_path / "var", peers=tmp_path / "peers")


@pytest.fixture
def signer(host):
    return A.key_signer(host, P, workers=1)


def hooks(host, signer, token="ab12", active=lambda s: False):
    return dict(now=T0, token=token, own=lambda d: None, start=lambda s: None, active=active, secret=SECRET,
                ready=lambda s: None,
                admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0, active=active),
                period=lambda args: A.period(host, args, signer=signer, pinned=PIN, now=T0, active=active))


def ssh(host, signer, words, key, **kw):
    rc, out = A.gate(["--ssh", words, "--key", key], host, **hooks(host, signer, **kw))
    return rc, json.loads(out)


def admin(host, signer, words, **kw):
    rc, out = A.gate(shlex.split(words), host, by="vy-admin", **hooks(host, signer, **kw))
    return rc, json.loads(out)


def open_words(root: str, *extra: str) -> str:
    return (f"open --commit {root} --peer vy-nebius-1 --scheme verity/test/coins/v1 --families '{json.dumps(FAMILIES)}' "
            + " ".join(extra))


def ok(answer):
    rc, doc = answer
    assert rc == 0, doc
    return doc


def refused(answer, why):
    rc, doc = answer
    assert rc == 3 and why in doc["refused"], doc


def end(host, signer, session, complete):
    I, sign, _ = signer
    A.append(host, "end", session, lambda lines: {"rc": 0, "closed": True, "complete": complete, "record_leaf": None},
             I, sign)


def checked(host):
    public = (host.key / "public").read_bytes()
    lines = A.read_ledger(host.ledger)
    verify = lambda m, s: lms.verify(P, public, m, lms.Signature.decode(P, s))  # noqa: E731
    A.verify_ledger(lines, public[:16], verify)
    return A.verify_periods(lines, public[:16], verify)


def test_a_lanes_period_holds_exactly_its_windows_between_open_and_close_each_with_its_outcome(host, signer):
    before = ok(ssh(host, signer, open_words("01" * 64), "hash", token="0001"))["session"]
    opened = ok(ssh(host, signer, "period-open --name hash-p1", "hash"))
    assert opened["kind"] == "period-open" and opened["body"]["lane"] == "hash" and opened["body"]["by"] == "hash"
    done = ok(ssh(host, signer, open_words("02" * 64), "hash", token="0002"))["session"]
    other = ok(ssh(host, signer, open_words("03" * 64), "vllm", token="0003"))["session"]
    short = ok(ssh(host, signer, open_words("04" * 64), "hash", token="0004"))["session"]
    end(host, signer, done, True)
    end(host, signer, short, False)
    end(host, signer, other, True)
    closed = ok(ssh(host, signer, "period-close --name hash-p1", "hash"))
    assert closed["body"]["windows"] == [{"root": "02" * 64, "session": done, "complete": True},
                                         {"root": "04" * 64, "session": short, "complete": False}]
    assert closed["body"]["complete"] is False
    ok(ssh(host, signer, open_words("05" * 64), "hash", token="0005"))

    periods = checked(host)
    assert list(periods) == ["hash-p1"] and periods["hash-p1"]["windows"] == ["02" * 64, "04" * 64]
    assert before not in {w["session"] for w in closed["body"]["windows"]}

    ok(ssh(host, signer, "period-open --name hash-p2", "hash"))
    full = ok(ssh(host, signer, open_words("06" * 64), "hash", token="0006"))["session"]
    end(host, signer, full, True)
    assert ok(ssh(host, signer, "period-close --name hash-p2", "hash"))["body"]["complete"] is True
    assert checked(host)["hash-p2"]["windows"] == ["06" * 64]


def test_periods_never_overlap_in_a_lane_a_name_is_used_once_and_only_its_lane_or_vy_admin_closes_one(host, signer):
    ok(ssh(host, signer, "period-open --name p1", "hash"))
    refused(ssh(host, signer, "period-open --name p2", "hash"), "lane hash has period p1 open")
    refused(ssh(host, signer, "period-open --name p1", "vllm"), "a period's name is used once")
    refused(ssh(host, signer, "period-close --name p1", "vllm"), "period p1 is lane hash's")
    refused(ssh(host, signer, "period-close --name nope", "hash"), "no open period nope")
    ok(ssh(host, signer, "period-open --name v1", "vllm"))
    ok(admin(host, signer, "period-close --name p1"))
    refused(ssh(host, signer, "period-close --name p1", "hash"), "no open period p1")
    refused(ssh(host, signer, "period-open --name p1", "hash"), "a period's name is used once")
    assert [p["close"] is None for p in checked(host).values()] == [False, True]


def test_who_writes_a_period(host, signer):
    rc, out = A.gate(["--ssh", "period-open --name p1"], host, **hooks(host, signer))
    assert rc == 3 and "reads only" in json.loads(out)["refused"]
    refused(ssh(host, signer, "period-open --name p1 --lane vllm", "hash"), "unrecognized arguments: --lane vllm")
    refused(admin(host, signer, "period-open --name p1"), "--lane")
    refused(admin(host, signer, "period-open --name p1 --lane vy-admin"), "--lane 'vy-admin'")
    refused(admin(host, signer, "period-open --name 'p 1' --lane kernels"), "--name 'p 1'")
    assert A.read_ledger(host.ledger) == []
    assert ok(admin(host, signer, "period-open --name k1 --lane kernels"))["body"]["by"] == "vy-admin"


def test_vy_admins_window_for_a_lane_counts_in_that_lanes_period_and_no_lane_key_opens_one_for_another(host, signer):
    ok(admin(host, signer, "period-open --name k1 --lane kernels"))
    refused(ssh(host, signer, open_words("01" * 64, "--for kernels"), "hash"), "unrecognized arguments: --for kernels")
    refused(admin(host, signer, open_words("01" * 64, "--for auditor")), "--for 'auditor'")
    mine = ok(admin(host, signer, open_words("02" * 64), token="0002"))
    assert "for" not in mine
    dev = ok(admin(host, signer, open_words("01" * 64, "--for kernels"), token="0001"))
    assert dev["by"] == "vy-admin" and dev["for"] == "kernels" and dev["secret"] == SECRET.hex()
    body = A.sessions_of(A.read_ledger(host.ledger))[dev["session"]]["open"]["body"]
    assert body["by"] == "vy-admin" and body["for"] == "kernels"
    end(host, signer, dev["session"], True)
    closed = ok(admin(host, signer, "period-close --name k1"))
    assert closed["body"]["windows"] == [{"root": "01" * 64, "session": dev["session"], "complete": True}]
    assert closed["body"]["complete"] is True and closed["body"]["lane"] == "kernels"
    assert checked(host)["k1"]["windows"] == ["01" * 64]


def test_a_period_closes_only_once_its_windows_have_ended_and_an_empty_one_is_incomplete(host, signer):
    ok(ssh(host, signer, "period-open --name p1", "hash"))
    s = ok(ssh(host, signer, open_words("01" * 64), "hash", token="0001"))["session"]
    refused(ssh(host, signer, "period-close --name p1", "hash", active=lambda _: True), f"window {s} still runs")
    stopped = ok(ssh(host, signer, "period-close --name p1", "hash"))
    assert stopped["body"]["windows"] == [{"root": "01" * 64, "session": s, "complete": False}]

    ok(ssh(host, signer, "period-open --name p2", "hash"))
    empty = ok(ssh(host, signer, "period-close --name p2", "hash"))
    assert empty["body"]["windows"] == [] and empty["body"]["complete"] is False
    checked(host)


def forge(host, signer, kind, session, body):
    I, sign, _ = signer
    A.append(host, kind, session, lambda lines: body, I, sign)


LISTS = "doesn't list exactly its windows and their outcomes"
WHY = {"drops a window": LISTS, "claims complete": LISTS, "lists another lane's": LISTS, "empty but complete": LISTS,
       "overlaps": "lane hash has a period open", "name reused": "p1 is named twice", "closes nothing": "for p9, which isn't",
       "has a session": "a period's none"}


@pytest.mark.parametrize("case", list(WHY))
def test_the_check_refuses_a_signed_period_entry_the_gate_would_not_write(host, signer, case):
    ok(ssh(host, signer, "period-open --name p1", "hash"))
    s = ok(ssh(host, signer, open_words("01" * 64), "hash", token="0001"))["session"]
    o = ok(ssh(host, signer, open_words("02" * 64), "vllm", token="0002"))["session"]
    end(host, signer, s, False)
    w = {"root": "01" * 64, "session": s, "complete": False}
    close = {"period": "p1", "lane": "hash", "by": "hash"}
    if case == "drops a window":
        forge(host, signer, "period-close", None, {**close, "windows": [], "complete": False})
    elif case == "claims complete":
        forge(host, signer, "period-close", None, {**close, "windows": [{**w, "complete": True}], "complete": True})
    elif case == "lists another lane's":
        forge(host, signer, "period-close", None,
              {**close, "windows": [w, {"root": "02" * 64, "session": o, "complete": False}], "complete": False})
    elif case == "empty but complete":
        ok(ssh(host, signer, "period-close --name p1", "hash"))
        ok(ssh(host, signer, "period-open --name p2", "hash"))
        forge(host, signer, "period-close", None, {**close, "period": "p2", "windows": [], "complete": True})
    elif case == "overlaps":
        forge(host, signer, "period-open", None, {"period": "p2", "lane": "hash", "by": "hash"})
    elif case == "name reused":
        forge(host, signer, "period-open", None, {"period": "p1", "lane": "vllm", "by": "vllm"})
    elif case == "closes nothing":
        forge(host, signer, "period-close", None, {**close, "period": "p9", "windows": [], "complete": False})
    elif case == "has a session":
        forge(host, signer, "period-close", s, {**close, "windows": [w], "complete": False})
    with pytest.raises(ValueError, match=WHY[case]):
        checked(host)


def test_an_end_after_its_periods_close_leaves_the_close_and_the_ledger_standing(host, signer):
    ok(ssh(host, signer, "period-open --name p1", "hash"))
    s = ok(ssh(host, signer, open_words("01" * 64), "hash", token="0001"))["session"]
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    end(host, signer, s, True)
    assert checked(host)["p1"]["close"]["body"]["windows"] == [{"root": "01" * 64, "session": s, "complete": False}]
