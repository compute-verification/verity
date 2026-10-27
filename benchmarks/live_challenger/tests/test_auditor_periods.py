"""Periods in the auditor ledger: `period-open` and `period-close`, who writes them, which windows a period holds (each
counted at its reserve, an expired one as a refused window), and `verify_ledger`'s checks of their nesting, of each
reserve, open and end in order, and of each close's list."""

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

P = lms.Params(64, 4, 5)
T0 = datetime(2026, 10, 20, 9, 30, 0, tzinfo=timezone.utc)
PIN = "c" * 40
FAMILIES = [["pouw/salt", 1], ["pouw/round", 3]]
SECRET = bytes(range(32))
CIRCUIT = "9f" * 64


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


def hooks(host, signer, token="ab12", active=lambda s: False, now=T0):
    return dict(now=now, token=token, own=lambda d: None, start=lambda s: None, active=active, secret=SECRET,
                ready=lambda s: None,
                reserve=lambda s: A.reserve(host, s, signer=signer, pinned=PIN, now=now),
                admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=now, active=active),
                period=lambda args: A.period(host, args, signer=signer, pinned=PIN, now=now, active=active))


def ssh(host, signer, words, key, **kw):
    rc, out = A.gate(["--ssh", words, "--key", key], host, **hooks(host, signer, **kw))
    return rc, json.loads(out)


def admin(host, signer, words, **kw):
    rc, out = A.gate(shlex.split(words), host, by="vy-admin", **hooks(host, signer, **kw))
    return rc, json.loads(out)


RESERVE = f"reserve --peer vy-nebius-1 --scheme verity/test/coins/v1 --families '{json.dumps(FAMILIES)}'"


def call(host, signer, words, key, **kw):
    """A lane key's request (`key`), or vy-admin's (`key` None)."""
    return ssh(host, signer, words, key, **kw) if key else admin(host, signer, words, **kw)


def window(host, signer, root, key=None, *, token="ab12", for_=None, **kw) -> dict:
    """Reserve (vy-admin's `--for for_` when `key` is None) and open at `root`: the open's answer."""
    s = ok(call(host, signer, RESERVE + (f" --for {for_}" if for_ else ""), key, token=token, **kw))["session"]
    return ok(call(host, signer, f"open --session {s} --root {root}", key, **kw))


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


def checked(host, ledger=None):
    public = (host.key / "public").read_bytes()
    lines = A.read_ledger((ledger or host).ledger)
    verify = lambda m, s: lms.verify(P, public, m, lms.Signature.decode(P, s))  # noqa: E731
    A.verify_ledger(lines, public[:16], verify)
    return A.verify_periods(lines, public[:16], verify)


def test_a_lanes_period_holds_exactly_its_windows_between_open_and_close_each_with_its_outcome(host, signer):
    before = window(host, signer, "01" * 64, "hash", token="0001")["session"]
    opened = ok(ssh(host, signer, f"period-open --name hash-p1 --cap 8 --circuit {CIRCUIT}", "hash"))
    assert opened["kind"] == "period-open" and opened["body"]["lane"] == "hash" and opened["body"]["by"] == "hash"
    done = window(host, signer, "02" * 64, "hash", token="0002")["session"]
    other = window(host, signer, "03" * 64, "vllm", token="0003")["session"]
    short = window(host, signer, "04" * 64, "hash", token="0004")["session"]
    end(host, signer, done, True)
    end(host, signer, short, False)
    end(host, signer, other, True)
    closed = ok(ssh(host, signer, "period-close --name hash-p1", "hash"))
    assert closed["body"]["windows"] == [{"root": "02" * 64, "session": done, "complete": True},
                                         {"root": "04" * 64, "session": short, "complete": False}]
    assert closed["body"]["complete"] is False
    window(host, signer, "05" * 64, "hash", token="0005")

    periods = checked(host)
    assert list(periods) == ["hash-p1"] and periods["hash-p1"]["windows"] == [done, short]
    assert before not in {w["session"] for w in closed["body"]["windows"]}

    ok(ssh(host, signer, f"period-open --name hash-p2 --cap 8 --circuit {CIRCUIT}", "hash"))
    full = window(host, signer, "06" * 64, "hash", token="0006")["session"]
    end(host, signer, full, True)
    assert ok(ssh(host, signer, "period-close --name hash-p2", "hash"))["body"]["complete"] is True
    assert checked(host)["hash-p2"]["windows"] == [full]


def test_a_window_joins_its_period_at_its_reserve_and_one_that_expires_unopened_is_a_refused_window(host, signer):
    early = ok(ssh(host, signer, RESERVE + " --minutes 5", "hash", token="0001"))["session"]
    ok(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}", "hash"))
    rc, out = ssh(host, signer, f"open --session {early} --root {'01' * 64}", "hash")
    assert rc == 0 and out["session"] == early
    answer = ok(ssh(host, signer, RESERVE + " --minutes 5", "hash", token="0002"))
    lapsed = answer["session"]
    assert answer["period"] == "p1" and answer["expires_utc"] == "2026-10-20T09:35:00Z"
    refused(ssh(host, signer, "period-close --name p1", "hash"),
            f"window {lapsed} is reserved until 2026-10-20T09:35:00Z")
    late = datetime(2026, 10, 20, 9, 35, 1, tzinfo=timezone.utc)
    refused(ssh(host, signer, f"open --session {lapsed} --root {'02' * 64}", "hash", now=late),
            "reserve expired at 2026-10-20T09:35:00Z: it is a refused window of its period")
    rc, out = ssh(host, signer, f"status {lapsed}", "hash", now=late)
    assert rc == 0 and out["expired"] is True and out["open"] is None
    closed = ok(ssh(host, signer, "period-close --name p1", "hash", now=late))
    assert closed["body"]["windows"] == [{"root": None, "session": lapsed, "complete": False}]
    assert closed["body"]["complete"] is False
    assert checked(host)["p1"]["windows"] == [lapsed]
    refused(ssh(host, signer, f"open --session {lapsed} --root {'02' * 64}", "hash"),
            f"period p1 has closed, listing it unopened")


def test_periods_never_overlap_in_a_lane_a_name_is_used_once_and_only_its_lane_or_vy_admin_closes_one(host, signer):
    ok(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}", "hash"))
    refused(ssh(host, signer, f"period-open --name p2 --cap 8 --circuit {CIRCUIT}", "hash"), "lane hash has period p1 open")
    refused(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}", "vllm"), "a period's name is used once")
    refused(ssh(host, signer, "period-close --name p1", "vllm"), "period p1 is lane hash's")
    refused(ssh(host, signer, "period-close --name nope", "hash"), "no open period nope")
    ok(ssh(host, signer, f"period-open --name v1 --cap 8 --circuit {CIRCUIT}", "vllm"))
    ok(admin(host, signer, "period-close --name p1"))
    refused(ssh(host, signer, "period-close --name p1", "hash"), "no open period p1")
    refused(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}", "hash"), "a period's name is used once")
    assert [p["close"] is None for p in checked(host).values()] == [False, True]


def test_who_writes_a_period(host, signer):
    rc, out = A.gate(["--ssh", f"period-open --name p1 --cap 8 --circuit {CIRCUIT}"], host, **hooks(host, signer))
    assert rc == 3 and "reads only" in json.loads(out)["refused"]
    refused(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT} --lane vllm", "hash"),
            "unrecognized arguments: --lane vllm")
    refused(admin(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}"), "--lane")
    refused(admin(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT} --lane vy-admin"), "--lane 'vy-admin'")
    refused(admin(host, signer, f"period-open --name 'p 1' --cap 8 --circuit {CIRCUIT} --lane kernels"), "--name 'p 1'")
    assert A.read_ledger(host.ledger) == []
    opened = ok(admin(host, signer, f"period-open --name k1 --cap 8 --circuit {CIRCUIT} --lane kernels"))
    assert opened["body"]["by"] == "vy-admin"


def test_vy_admins_window_for_a_lane_counts_in_that_lanes_period_and_no_lane_key_reserves_one_for_another(host, signer):
    ok(admin(host, signer, f"period-open --name k1 --cap 8 --circuit {CIRCUIT} --lane kernels"))
    refused(ssh(host, signer, RESERVE + " --for kernels", "hash"), "unrecognized arguments: --for kernels")
    refused(admin(host, signer, RESERVE + " --for auditor"), "--for 'auditor'")
    mine = window(host, signer, "02" * 64, token="0002")
    assert "for" not in mine
    dev = window(host, signer, "01" * 64, token="0001", for_="kernels")
    assert dev["by"] == "vy-admin" and dev["for"] == "kernels" and dev["secret"] == SECRET.hex()
    body = A.sessions_of(A.read_ledger(host.ledger))[dev["session"]]["open"]["body"]
    assert body["by"] == "vy-admin" and body["for"] == "kernels" and body["opened_by"] == "vy-admin"
    end(host, signer, dev["session"], True)
    closed = ok(admin(host, signer, "period-close --name k1"))
    assert closed["body"]["windows"] == [{"root": "01" * 64, "session": dev["session"], "complete": True}]
    assert closed["body"]["complete"] is True and closed["body"]["lane"] == "kernels"
    assert checked(host)["k1"]["windows"] == [dev["session"]]


def test_a_period_closes_only_once_its_windows_have_ended_and_an_empty_one_is_incomplete(host, signer):
    ok(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}", "hash"))
    s = window(host, signer, "01" * 64, "hash", token="0001")["session"]
    refused(ssh(host, signer, "period-close --name p1", "hash", active=lambda _: True), f"window {s} still runs")
    stopped = ok(ssh(host, signer, "period-close --name p1", "hash"))
    assert stopped["body"]["windows"] == [{"root": "01" * 64, "session": s, "complete": False}]

    ok(ssh(host, signer, f"period-open --name p2 --cap 8 --circuit {CIRCUIT}", "hash"))
    empty = ok(ssh(host, signer, "period-close --name p2", "hash"))
    assert empty["body"]["windows"] == [] and empty["body"]["complete"] is False
    checked(host)


def forge(host, signer, kind, session, body):
    I, sign, _ = signer
    A.append(host, kind, session, lambda lines: body, I, sign)


LISTS = "doesn't list exactly its windows and their outcomes"
WHY = {"drops a window": LISTS, "claims complete": LISTS, "lists another lane's": LISTS, "empty but complete": LISTS,
       "omits an expired reserve": LISTS, "gives an unopened reserve a root": LISTS,
       "overlaps": "lane hash has a period open", "name reused": "p1 is named twice", "closes nothing": "for p9, which isn't",
       "has a session": "a period's none"}


@pytest.mark.parametrize("case", list(WHY))
def test_the_check_refuses_a_signed_period_entry_the_gate_would_not_write(host, signer, case):
    ok(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}", "hash"))
    s = window(host, signer, "01" * 64, "hash", token="0001")["session"]
    o = window(host, signer, "02" * 64, "vllm", token="0002")["session"]
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
    elif case in ("omits an expired reserve", "gives an unopened reserve a root"):
        r = ok(ssh(host, signer, RESERVE + " --minutes 1", "hash", token="0003"))["session"]
        listed = [w] + ([{"root": "03" * 64, "session": r, "complete": False}] if "root" in case else [])
        forge(host, signer, "period-close", None, {**close, "windows": listed, "complete": False})
    elif case == "empty but complete":
        ok(ssh(host, signer, "period-close --name p1", "hash"))
        ok(ssh(host, signer, f"period-open --name p2 --cap 8 --circuit {CIRCUIT}", "hash"))
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
    ok(ssh(host, signer, f"period-open --name p1 --cap 8 --circuit {CIRCUIT}", "hash"))
    s = window(host, signer, "01" * 64, "hash", token="0001")["session"]
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    end(host, signer, s, True)
    assert checked(host)["p1"]["close"]["body"]["windows"] == [{"root": "01" * 64, "session": s, "complete": False}]


def test_a_period_open_commits_its_cap_and_circuit_and_the_gate_reserves_no_window_past_the_cap(host, signer):
    """`CircuitWorkPeriod` reads both, committed before window 0's first coin; a reserve
    counts toward the cap from its entry on, opened or not."""
    opened = ok(admin(host, signer, f"period-open --name k1 --lane kernels --cap 2 --circuit {'ab' * 64}"))
    assert (opened["body"]["cap"], opened["body"]["circuit"]) == (2, "ab" * 64)
    first = window(host, signer, "01" * 64, token="0001", for_="kernels")["session"]
    window(host, signer, "02" * 64, "hash", token="0002")
    held = ok(admin(host, signer, RESERVE + " --for kernels", token="0003"))["session"]
    n = len(A.read_ledger(host.ledger))
    refused(admin(host, signer, RESERVE + " --for kernels", token="0004"), "period k1 holds its cap of 2")
    assert len(A.read_ledger(host.ledger)) == n and not (host.sessions / "s20261020T093000Z-0004").exists()
    ok(admin(host, signer, f"open --session {held} --root {'03' * 64}"))
    ok(admin(host, signer, "period-close --name k1"))
    ok(admin(host, signer, f"period-open --name k2 --lane kernels --cap 1 --circuit {CIRCUIT.upper()}"))
    last = window(host, signer, "04" * 64, token="0005", for_="kernels")["session"]
    periods = checked(host)
    assert {k: (p["cap"], p["circuit"], p["windows"]) for k, p in periods.items()} == {
        "k1": (2, "ab" * 64, [first, held]), "k2": (1, CIRCUIT, [last])}


def test_a_period_open_commits_its_draw_k_when_given_and_none_otherwise(host, signer):
    """PoUW's credit reads the period's draw K from its open (`draw_k`), signed before window 0's first coin."""
    opened = ok(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --draw-k 27713", "hash"))
    assert opened["body"]["draw_k"] == 27713
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    assert "draw_k" not in ok(admin(host, signer, f"period-open --name k1 --lane kernels --cap 1 --circuit {CIRCUIT}"))["body"]
    assert ok(admin(host, signer, f"period-open --name v1 --lane vllm --cap 1 --circuit {CIRCUIT} --draw-k {2 ** 24}"))[
        "body"]["draw_k"] == 2 ** 24
    lines = A.read_ledger(host.ledger)
    assert [json.loads(e)["body"].get("draw_k") for e in lines if json.loads(e)["kind"] == "period-open"] == [27713, None,
                                                                                                              2 ** 24]
    assert set(checked(host)) == {"p1", "k1", "v1"}


def test_a_period_open_commits_its_table_rows_when_given_and_none_otherwise(host, signer):
    """PoUW's credit bounds every step's served rows m_s by the table rows T its period's open commits (`rows`, credit's
    `committed_rows`), signed before window 0's first coin beside its draw K or alone."""
    opened = ok(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --draw-k 27 --rows 12", "hash"))
    assert (opened["body"]["draw_k"], opened["body"]["rows"]) == (27, 12)
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    body = ok(admin(host, signer, f"period-open --name k1 --lane kernels --cap 1 --circuit {CIRCUIT} --rows {2 ** 24}"))[
        "body"]
    assert body["rows"] == 2 ** 24 and "draw_k" not in body
    assert "rows" not in ok(admin(host, signer, f"period-open --name v1 --lane vllm --cap 1 --circuit {CIRCUIT}"))["body"]
    lines = [json.loads(e) for e in A.read_ledger(host.ledger)]
    assert [e["body"].get("rows") for e in lines if e["kind"] == "period-open"] == [12, 2 ** 24, None]
    assert set(checked(host)) == {"p1", "k1", "v1"}


def test_a_period_open_commits_its_weights_root_when_given_and_none_otherwise(host, signer):
    """PoUW's credit holds every window of a period to the served model's registered weights' root its open commits
    (`weights_root`, `credit.weights_root` of the B rows the auditor registered before the period's first salt, with
    its model), signed before window 0's first coin, beside its draw K or alone."""
    w = "5e" * 64
    host.models.write_text(json.dumps({"model": "M8", "circuit": CIRCUIT, "weights_root": w,
                                       "checkpoint_sha512": "c4" * 64}) + "\n")
    opened = ok(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --draw-k 27 --weights-root {w}",
                    "hash"))
    assert (opened["body"]["draw_k"], opened["body"]["weights_root"]) == (27, w)
    assert opened["body"]["model"] == {"name": "M8", "checkpoint_sha512": "c4" * 64}
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    body = ok(admin(host, signer, f"period-open --name k1 --lane kernels --cap 1 --circuit {CIRCUIT} --weights-root {w}"))[
        "body"]
    assert body["weights_root"] == w and "draw_k" not in body
    assert "weights_root" not in ok(admin(host, signer, f"period-open --name v1 --lane vllm --cap 1 --circuit {CIRCUIT}"))[
        "body"]
    lines = [json.loads(e) for e in A.read_ledger(host.ledger)]
    assert [e["body"].get("weights_root") for e in lines if e["kind"] == "period-open"] == [w, w, None]
    assert set(checked(host)) == {"p1", "k1", "v1"}


def test_a_period_open_commits_each_ks_circuit_file_when_given_and_none_otherwise(host, signer):
    """PoUW's credit verifies each window's statements on the circuit file its period's open committed for the call's K
    (`circuit_files`, ``{k<K>: SHA-512}``, credit's `committed_files`), signed before window 0's first coin beside its
    draw K, whatever order or spacing the JSON came in."""
    files = {"k4096": "1a" * 64, "k64": "2b" * 64}
    given = shlex.quote(json.dumps({"k64": "2b" * 64, "k4096": "1a" * 64}, indent=1))
    opened = ok(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --draw-k 27 "
                                  f"--circuit-files {given}", "hash"))
    assert opened["body"]["circuit_files"] == files and opened["body"]["draw_k"] == 27
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    body = ok(admin(host, signer, f"period-open --name k1 --lane kernels --cap 1 --circuit {CIRCUIT} "
                                  f"--circuit-files {shlex.quote(json.dumps(files))}"))["body"]
    assert body["circuit_files"] == files and "draw_k" not in body and "weights_root" not in body
    lines = [json.loads(e) for e in A.read_ledger(host.ledger)]
    assert [e["body"].get("circuit_files") for e in lines if e["kind"] == "period-open"] == [files, files]
    assert set(checked(host)) == {"p1", "k1"}


@pytest.mark.parametrize("files", [
    "{}", "[]", "nope", '{"k64": "2b"}', '{"k0": "' + "2b" * 64 + '"}', '{"k064": "' + "2b" * 64 + '"}',
    '{"64": "' + "2b" * 64 + '"}', '{"k64": "' + "2B" * 64 + '"}', '{"k64": 1}', '{"lives": "' + "2b" * 64 + '"}',
    '{"k64": "' + "2b" * 64 + '", "k64": "' + "1a" * 64 + '"}',
])
def test_a_period_open_refuses_circuit_files_it_cant_check(host, signer, files):
    rc, out = ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --circuit-files {shlex.quote(files)}",
                  "hash")
    assert rc == 3 and "--circuit-files" in out["refused"]
    assert A.read_ledger(host.ledger) == []
    assert len(A.circuit_files_of(json.dumps({f"k{i}": "2b" * 64 for i in range(1, 65)}))) == 64
    assert A.circuit_files_of(json.dumps({"k64": "2b" * 64, "live": "1a" * 64})) == {"k64": "2b" * 64, "live": "1a" * 64}
    with pytest.raises(A.Refused, match="1 to 64 distinct K"):
        A.circuit_files_of(json.dumps({f"k{i}": "2b" * 64 for i in range(1, 66)}))


def test_a_period_open_commits_its_fixed_length_when_given_and_none_otherwise(host, signer):
    """PoUW's credit holds each window's served rows to the fixed length its period's open committed (`length`,
    ``{max_tokens, stop}``, credit's `committed_length`), signed before window 0's first coin."""
    length = {"max_tokens": [3, 1, 2], "stop": "fixed"}
    opened = ok(ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --draw-k 27 "
                                  f"--length {shlex.quote(json.dumps(length, indent=1))}", "hash"))
    assert opened["body"]["length"] == length and opened["body"]["draw_k"] == 27
    ok(ssh(host, signer, "period-close --name p1", "hash"))
    assert "length" not in ok(admin(host, signer, f"period-open --name v1 --lane vllm --cap 1 --circuit {CIRCUIT}"))[
        "body"]
    assert set(checked(host)) == {"p1", "v1"}


@pytest.mark.parametrize("length", [
    "{}", "[]", "nope", '{"max_tokens": []}', '{"max_tokens": [], "stop": "fixed"}',
    '{"max_tokens": [0], "stop": "fixed"}', '{"max_tokens": [true], "stop": "fixed"}',
    '{"max_tokens": [1.0], "stop": "fixed"}', '{"max_tokens": [4], "stop": "eos"}',
    '{"max_tokens": [4], "stop": "fixed", "served": [4]}', '{"max_tokens": [16777217], "stop": "fixed"}',
])
def test_a_period_open_refuses_a_length_it_cant_check(host, signer, length):
    rc, out = ssh(host, signer, f"period-open --name p1 --cap 2 --circuit {CIRCUIT} --length {shlex.quote(length)}",
                  "hash")
    assert rc == 3 and "--length" in out["refused"]
    assert A.read_ledger(host.ledger) == []


@pytest.mark.parametrize("args", [
    ["period-open", "p1", "hash", "hash", "2", "ab" * 64, "weights_root=" + "5e" * 64, "weights_root=" + "5e" * 64],
    ["period-open", "p1", "hash", "hash", "2", "ab" * 64, "pinned=" + "5e" * 20],
    ["period-open", "p1", "hash", "hash", "2", "ab" * 64, "27"],
    ["period-open", "p1", "hash", "hash", "2", "ab" * 64, "draw_k=27", "weights_root=" + "5e" * 64, "draw_k=28"],
])
def test_a_period_open_with_a_field_twice_or_unknown_is_refused(host, signer, args):
    with pytest.raises(A.Refused, match="period: "):
        A.period(host, args, signer=signer, pinned=PIN, now=T0, active=lambda s: False)
    assert A.read_ledger(host.ledger) == []


@pytest.mark.parametrize("args, why", [
    ("--name p1", "the following arguments are required: --cap, --circuit"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --weights-root {'5E' * 64}", "--weights-root '5E5E"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --weights-root {'5e' * 63}", "--weights-root '5e5e"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --weights-root {'5e' * 65}", "--weights-root '5e5e"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --weights-root {'5e' * 63}zz", "--weights-root '5e5e"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --draw-k 3 --weights-root nope", "--weights-root 'nope'"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --draw-k 0", "--draw-k '0'"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --draw-k -3", "--draw-k"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --draw-k 07", "--draw-k '07'"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --draw-k {2 ** 24 + 1}", f"--draw-k '{2 ** 24 + 1}'"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --draw-k many", "--draw-k 'many'"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --rows 0", "--rows '0'"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --rows 012", "--rows '012'"),
    (f"--name p1 --cap 2 --circuit {CIRCUIT} --rows {2 ** 24 + 1}", f"--rows '{2 ** 24 + 1}'"),
    (f"--name p1 --circuit {CIRCUIT}", "required: --cap"),
    ("--name p1 --cap 2", "required: --circuit"),
    (f"--name p1 --cap 0 --circuit {CIRCUIT}", "--cap '0'"),
    (f"--name p1 --cap -1 --circuit {CIRCUIT}", "--cap '-1'"),
    (f"--name p1 --cap 02 --circuit {CIRCUIT}", "--cap '02'"),
    (f"--name p1 --cap two --circuit {CIRCUIT}", "--cap 'two'"),
    (f"--name p1 --cap {2 ** 24 + 1} --circuit {CIRCUIT}", f"--cap {2 ** 24 + 1}"),
    ("--name p1 --cap 2 --circuit ''", "--circuit ''"),
    ("--name p1 --cap 2 --circuit 'a b'", "--circuit 'a b'"),
    ("--name p1 --cap 2 --circuit pouw/pearl-c@v3", "not a Program's id"),
    (f"--name p1 --cap 2 --circuit {'a' * 127}", f"--circuit '{'a' * 127}'"),
    (f"--name p1 --cap 2 --circuit {'a' * 129}", f"--circuit '{'a' * 129}'"),
])
def test_a_period_open_without_a_valid_cap_and_circuit_is_refused(host, signer, args, why):
    rc, doc = ssh(host, signer, f"period-open {args}", "hash")
    refused((rc, doc), why)
    assert "\n" not in doc["refused"] and doc["refused"].count("usage:") <= 1
    assert A.read_ledger(host.ledger) == []


@pytest.mark.parametrize("body, why", [
    ({}, None),
    ({"cap": 1, "circuit": CIRCUIT}, "past its cap 1"),
    ({"cap": 0, "circuit": CIRCUIT}, "cap is a positive integer"),
    ({"cap": True, "circuit": CIRCUIT}, "cap is a positive integer"),
    ({"cap": 3}, "cap is a positive integer"),
    ({"circuit": CIRCUIT}, "cap is a positive integer"),
    ({"cap": 3, "circuit": CIRCUIT.upper()}, "cap is a positive integer"),
    ({"cap": 3, "circuit": "pouw/pearl-c@v3"}, "its circuit a SHA-512 in lowercase"),
])
def test_the_check_reads_a_periods_cap_and_keeps_one_step_opens_before_the_first_reserve_verifiable(host, signer, body,
                                                                                                    why):
    forge(host, signer, "period-open", None, {"period": "p1", "lane": "hash", "by": "hash", **body})
    for i in (1, 2):
        forge(host, signer, "open", f"s20261020T093000Z-000{i}", {"window": {"root": f"0{i}" * 64}, "by": "hash"})
    if why is None:
        p = checked(host)["p1"]
        assert p["cap"] is None and p["windows"] == ["s20261020T093000Z-0001", "s20261020T093000Z-0002"]
    else:
        with pytest.raises(ValueError, match=why):
            checked(host)


def reserve_body(**kw):
    return {"by": "hash", "peer": "vy-nebius-1", "scheme": "verity/test/coins/v1", "epoch": 0, "families": FAMILIES,
            "minutes": 10, "started_utc": "2026-10-20T09:30:00Z", "expires_utc": "2026-10-20T09:40:00Z", **kw}


def open_body(reserve_seq=0, **kw):
    return {"window": {"root": "01" * 64}, "reserve": reserve_seq,
            **{k: v for k, v in reserve_body().items() if k in A.RESERVED}, "opened_by": "hash",
            "started_utc": "2026-10-20T09:31:00Z", **kw}


S1, S2 = "s20261020T093000Z-0001", "s20261020T093000Z-0002"
ORDER = {
    "a one-step open after a reserve": ([("reserve", S1, reserve_body()), ("open", S2, {"window": {"root": "02" * 64},
                                                                                         "by": "hash"})],
                                        "without a reserve after the ledger's first reserve"),
    "an open binding another reserve": ([("reserve", S1, reserve_body()), ("reserve", S2, reserve_body()),
                                         ("open", S2, open_body(0))], "binds no reserve of it"),
    "an open of a session never reserved": ([("reserve", S1, reserve_body()), ("open", S2, open_body(0))],
                                            "binds no reserve of it"),
    "an open with another request": ([("reserve", S1, reserve_body()), ("open", S1, open_body(families=[["x", 1]]))],
                                     "doesn't carry its reserve's request"),
    "an open dropping the reserve's for": ([("reserve", S1, reserve_body(by="vy-admin", **{"for": "kernels"})),
                                            ("open", S1, open_body(by="vy-admin", opened_by="vy-admin"))],
                                           "doesn't carry its reserve's request"),
    "an open by another lane": ([("reserve", S1, reserve_body()), ("open", S1, open_body(opened_by="vllm"))],
                                "opened by 'vllm', not its reserve's prover"),
    "an open after its expiry": ([("reserve", S1, reserve_body()),
                                  ("open", S1, open_body(started_utc="2026-10-20T09:40:01Z"))],
                                 "opened outside its reserve's minutes"),
    "an open before its reserve's start": ([("reserve", S1, reserve_body()),
                                            ("open", S1, open_body(started_utc="2026-10-20T09:29:59Z"))],
                                           "opened outside its reserve's minutes"),
    "an expiry that isn't start + minutes": ([("reserve", S1, reserve_body(expires_utc="2026-10-20T10:30:00Z"))],
                                             "its expiry"),
    "a reserve without minutes": ([("reserve", S1, reserve_body(minutes=None))], "its expiry"),
    "a second reserve of a session": ([("reserve", S1, reserve_body()), ("reserve", S1, reserve_body())],
                                      "first entry of its session"),
    "a reserve after its session's open": ([("reserve", S1, reserve_body()), ("open", S1, open_body()),
                                            ("reserve", S1, reserve_body())], "first entry of its session"),
    "an end of a reserve never opened": ([("reserve", S1, reserve_body()), ("end", S1, {"complete": True})],
                                         "has no open"),
    "a second open of a reserve": ([("reserve", S1, reserve_body()), ("open", S1, open_body()),
                                    ("open", S1, open_body(window={"root": "02" * 64}))], "a second open"),
}


@pytest.mark.parametrize("case", list(ORDER))
def test_the_check_refuses_a_reserve_open_or_end_out_of_order(host, signer, case):
    entries, why = ORDER[case]
    for kind, session, body in entries:
        forge(host, signer, kind, session, body)
    with pytest.raises(ValueError, match=why):
        checked(host)


def test_the_check_refuses_an_open_after_its_periods_close_and_a_reserve_past_the_cap(host, signer):
    ok(ssh(host, signer, f"period-open --name p1 --cap 1 --circuit {CIRCUIT}", "hash"))
    forge(host, signer, "reserve", S1, reserve_body())
    forge(host, signer, "period-close", None, {"period": "p1", "lane": "hash", "by": "hash",
                                               "windows": [{"root": None, "session": S1, "complete": False}],
                                               "complete": False})
    checked(host)
    forge(host, signer, "open", S1, open_body(1))
    with pytest.raises(ValueError, match="opened after its period p1 closed"):
        checked(host)

    other = A.Host(root=host.root.parent / "other", peers=host.peers, pins=host.pins)
    other.root.mkdir()
    ok(ssh(other, signer, f"period-open --name p1 --cap 1 --circuit {CIRCUIT}", "hash"))
    forge(other, signer, "reserve", S1, reserve_body())
    forge(other, signer, "reserve", S2, reserve_body())
    with pytest.raises(ValueError, match="period p1's window 1 is past its cap 1"):
        checked(host, other)
