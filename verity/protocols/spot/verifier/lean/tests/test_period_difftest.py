"""Sampled proofs' period, the ledger's half (`Spot.Period`: `sessionsOf`, `windowsOf`), held to the reference below
through the difftest's `period_windows` piece. The reference lives here alone: there is no Python verifier of the
period, and it reads the ledger through the challenger's mirror (`challenger._checked`) as the Lean reads it through
`Challenger.checked`.

The cases are ledgers a small key signs, written one entry at a time as the auditor's gate writes them (`Book`): a
lane's periods, each with windows that hold a sample's family, reserved, opened and ended, other lanes' sessions among
them, the lane's sessions of other layouts between its periods, and an entry stamped after the monitoring period's
end. Then one change each: a ledger cut before that entry, a signature edited, another family asked for, a window in
no period, a period left open, a period holding a session of another layout or one opened before the monitoring
period, a reserve never opened, a window never ended, an entry before the period stamped past its end; and ledgers
that must still read the same: an end stamped before the entry written ahead of it, a window stamped before the
monitoring period's start by a clock set back, a ledger of one-step opens. A malformed request is the request's fault
on both sides.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import shutil
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import strategies as st

from test_challenger_difftest import PACKAGE, PIECES, _key, _sign_ledger, lms_verify

import verity_difftest as D
from verity.commitments.crypto import lms
from verity.protocols.spot.device import challenger as Ch

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

W, H = 1, 6
KEY = _key(W, H, 0)
PUBLIC = lms.public_key(KEY)
KEY_JSON = Ch.ChallengerKey(W, H, PUBLIC).to_json()
KEY_SHA = hashlib.sha512(PUBLIC).hexdigest()
#: the most entries a `Book` holds: one leaf of the key each
LEAVES = 2 ** H

LANE, OTHER = "dev", "other-lane"
LABEL = "draw"
DRAW = [[LABEL, 17]]
SALT = [["pouw/salt", 1]]
BASE = datetime(2026, 3, 2, tzinfo=UTC)
DAY = 86400


def at(seconds: int) -> str:
    """The time `seconds` after the monitoring period's start, as the ledger writes one."""
    return (BASE + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


START, STOP = at(0), at(7 * DAY)
KEYS = ["key", "label", "lane", "ledger", "start", "stop"]
STAMP = {"reserve": "started_utc", "open": "started_utc", "end": "ended_utc", "period-open": "opened_utc",
         "period-close": "closed_utc", "declaration": "declared_utc"}
PARAMS_FORM = "a period's parameters are malformed"


# ---- the reference --------------------------------------------------------------------------------------------------

def stamp_of(e: dict) -> str | None:
    t = e["body"].get(STAMP[e["kind"]])
    return t if type(t) is str and Ch.later(t, 0) == t else None


def reaches(es: list[dict], t: str) -> int | None:
    return next((i for i, e in enumerate(es) if (s := stamp_of(e)) is not None and s >= t), None)


def holds(label: str, families: Any) -> bool:
    return type(families) is list and any(type(f) is list and len(f) > 0 and type(f[0]) is str and f[0] == label
                                          for f in families)


def _refused(code: str, why: str, **more: Any) -> dict:
    return {"code": code, "refused": why, **more}


def reference(j: Any) -> Any:
    """`Spot.Period.windowsPiece`: the lane's sessions of the monitoring period, then its windows and the others, or
    the refusal's code and text."""
    if not isinstance(j, dict) or sorted(j) != KEYS:
        raise D.Reject(f"a request's keys are {', '.join(KEYS)}")
    label, lane, start, stop = (j[k] for k in ("label", "lane", "start", "stop"))
    if not (all(type(x) is str for x in (label, lane, start, stop)) and label and lane
            and Ch.later(start, 0) == start and Ch.later(stop, 0) == stop and start < stop):
        raise D.Reject(PARAMS_FORM)
    try:
        key = Ch.ChallengerKey.from_json(j["key"])
    except Ch.CoinError as e:
        raise D.Reject(str(e)) from None
    try:
        wins, periods = Ch._checked(j["ledger"], key, lms_verify)
    except Ch.CoinError as e:
        return _refused("coin", str(e))
    es = [json.loads(line) for line in j["ledger"]]
    stop_at = reaches(es, stop)
    if stop_at is None:
        return _refused("early", f"the auditor's ledger holds no entry stamped at or after {stop}: a ledger that does "
                                 "not reach the monitoring period's end could hide a later session")
    start_at = reaches(es, start)
    v2 = Ch.LEDGER_SCHEMA
    reserves = [e["session"] for e in es
                if e["schema"] == v2 and (e["kind"] == "reserve" or e["kind"] == "open" and "reserve" not in e["body"])
                and Ch.lane_of(e) == lane]
    sessions = []
    for root, w in wins.items():
        e = w["open"]
        if e["schema"] == v2 and Ch.lane_of(e) == lane and start_at <= e["seq"] < stop_at:
            sid = e["session"]
            sessions.append({"families": e["body"].get("families"),
                             "period": next((n for n, p in periods.items() if sid in p["windows"]), None),
                             "place": reserves.index(sid) if sid in reserves else len(reserves), "root": root,
                             "session": sid, "withheld": e["body"].get("withheld") is True})
    sampled = [s for s in sessions if holds(label, s["families"])]
    others = [s for s in sessions if not holds(label, s["families"])]
    if not sampled:
        return _refused("missing", f"lane {lane} opens no session holding a {label} coin in the monitoring period",
                        sessions=sessions)
    for s in sampled:
        if s["period"] is None:
            return _refused("outside", f"session {s['session']} is in no period of lane {lane}", sessions=sessions)
    held = [(n, p) for n, p in periods.items() if any(s["period"] == n for s in sampled)]
    opened = {w["open"]["session"]: w for w in wins.values()}
    windows = {s["session"] for s in sampled}
    for n, p in held:
        if p["close"] is None:
            return _refused("period", f"period {n} is not closed", sessions=sessions)
    for n, p in held:
        for sid in p["windows"]:
            if sid in opened and sid not in windows:
                return _refused("period", f"period {n}'s session {sid} is not a window: it was opened outside the "
                                          f"monitoring period, or it holds no {label} coin", sessions=sessions)
    for n, p in held:
        for sid in p["windows"]:
            if sid not in opened:
                return _refused("missing", f"period {n}'s session {sid} was reserved and never opened",
                                sessions=sessions)
    for s in sampled:
        if opened[s["session"]]["end"] is None:
            return _refused("missing", f"session {s['session']} has not ended", sessions=sessions)
    return {"others": others, "sessions": sessions,
            "windows": [{"period": s["period"], "phase": i, "root": s["root"], "session": s["session"]}
                        for i, s in enumerate(sampled)]}


# ---- ledgers --------------------------------------------------------------------------------------------------------

class Book:
    """A ledger under the test's key, written one entry at a time as the gate writes it: a reserve joins its lane's open
    period, an open binds its reserve and carries its request, a period's close lists its sessions with the outcomes
    the ledger shows then. `events` are its entries' (kind, session, body); `lines` chains and signs them, entry `i` at
    leaf `i`."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.events: list[tuple[str, str | None, dict]] = []
        self.sessions: dict[str, dict] = {}
        self.periods: dict[str, dict] = {}
        self.current: dict[str, str] = {}
        self._ids = itertools.count()

    def _add(self, kind: str, session: str | None, body: dict) -> int:
        self.events.append((kind, session, body))
        return len(self.events) - 1

    def _join(self, lane: str, sid: str) -> None:
        if lane in self.current:
            self.periods[self.current[lane]]["sessions"].append(sid)

    def reserve(self, lane: str, t: int, families: Any = DRAW, withheld: bool = True, minutes: int = 600) -> str:
        sid = f"s20260302T{next(self._ids):06d}Z-{self.rng.randbytes(2).hex()}"
        body = self.rng.choice([{"by": lane}, {"by": "operator", "for": lane}])
        body.update(peer="10.0.0.7", scheme="verity/test/coins/v1", epoch=0, families=families, minutes=minutes,
                    started_utc=at(t), expires_utc=Ch.later(at(t), minutes), **({"withheld": True} if withheld else {}))
        self.sessions[sid] = {"lane": lane, "reserve": self._add("reserve", sid, body), "request": body, "t": t}
        self._join(lane, sid)
        return sid

    def open(self, sid: str, t: int) -> str:
        s = self.sessions[sid]
        r = s["request"]
        s["root"] = self.rng.randbytes(64).hex()
        body = {"window": {"root": s["root"]}, "reserve": s["reserve"], **{k: r[k] for k in Ch.RESERVED if k in r},
                "opened_by": r["by"], "pinned": self.rng.randbytes(20).hex(), "key_sha512": KEY_SHA, "host": f"h.{sid}",
                "started_utc": at(t), **({"withheld": True} if r.get("withheld") is True else {})}
        s["open"] = self._add("open", sid, body)
        return s["root"]

    def open_one_step(self, lane: str, t: int | None, families: Any = DRAW, withheld: bool = True) -> str:
        """A v2 open that binds no reserve, as a ledger with no reserve yet holds them: counted in its lane's period at
        the open."""
        sid = f"one-{next(self._ids)}"
        root = self.rng.randbytes(64).hex()
        body = {"window": {"root": root}, "by": lane, "pinned": self.rng.randbytes(20).hex(), "key_sha512": KEY_SHA,
                "host": f"h.{sid}", "families": families, **({} if t is None else {"started_utc": at(t)}),
                **({"withheld": True} if withheld else {})}
        self.sessions[sid] = {"lane": lane, "root": root, "open": self._add("open", sid, body)}
        self._join(lane, sid)
        return sid

    def end(self, sid: str, t: int, complete: bool = False) -> None:
        s = self.sessions[sid]
        s["complete"] = complete
        s["end"] = self._add("end", sid, {"closed": True, "complete": complete, "files": {}, "rc": 0,
                                          "record_leaf": None, "ended_utc": at(t)})

    def window(self, lane: str, t: int, families: Any = DRAW, withheld: bool = True) -> str:
        """A session of `lane` reserved at `t`, opened a minute later and ended a minute after that."""
        sid = self.reserve(lane, t, families, withheld)
        self.open(sid, t + 60)
        self.end(sid, t + 120)
        return sid

    def period_open(self, name: str, lane: str, t: int, cap: int = 8) -> None:
        self._add("period-open", None, {"period": name, "lane": lane, "by": "operator", "cap": cap,
                                        "circuit": "ab" * 64, "key_sha512": KEY_SHA, "opened_utc": at(t)})
        self.periods[name] = {"lane": lane, "sessions": []}
        self.current[lane] = name

    def period_close(self, name: str, t: int) -> None:
        p = self.periods[name]
        outcomes = [{"complete": "open" in s and s.get("complete") is True, "root": s.get("root") if "open" in s else None,
                     "session": sid} for sid in p["sessions"] for s in [self.sessions[sid]]]
        self._add("period-close", None, {"period": name, "lane": p["lane"], "by": "operator", "windows": outcomes,
                                         "complete": bool(outcomes) and all(w["complete"] for w in outcomes),
                                         "closed_utc": at(t)})
        del self.current[p["lane"]]

    def lines(self, upto: int | None = None) -> list[str]:
        out: list[str] = []
        assert len(self.events) <= LEAVES, len(self.events)
        for i, (kind, sid, body) in enumerate(self.events[:upto]):
            entry = {"body": body, "kind": kind, "prev": hashlib.sha512(out[-1].encode()).hexdigest() if out else "0" * 128,
                     "schema": Ch.LEDGER_SCHEMA, "seq": i, "session": sid}
            out.append(Ch.ledger_text({**entry, "sig": _sign_ledger(KEY, i, entry)}))
        return out


def request(ledger: list[str], label: str = LABEL, lane: str = LANE, start: str = START, stop: str = STOP) -> dict:
    return {"key": KEY_JSON, "label": label, "lane": lane, "ledger": ledger, "start": start, "stop": stop}


#: each named case's refusal code, None where the ledger's half accepts
NAMES = {"honest": None, "stamped-early": None, "set-back": None, "one-step": None, "early": "early", "coin": "coin",
         "no-window": "missing", "outside": "outside", "open-period": "period", "foreign": "period",
         "straddle": "period", "unopened": "missing", "unended": "missing", "ahead": "missing"}
REQUESTS = ["request"]


def case(rng: random.Random, name: str) -> Any:
    """The request of case `name` (module docstring): a week's ledger of lane `LANE`, changed as `name` says. Its
    generator is seeded from `rng`, so Hypothesis records one draw a case and not each sampled byte."""
    rng = random.Random(rng.getrandbits(64))
    b = Book(rng)
    one_step = name == "one-step"

    def session(lane: str, t: int, families: Any = DRAW, ended: bool = True) -> str:
        if one_step:
            sid = b.open_one_step(lane, rng.choice([t, None]), families)
        else:
            sid = b.reserve(lane, t, families)
            b.open(sid, t + 60)
        if ended:
            b.end(sid, t + 120 - (rng.randint(60, 600) if name == "stamped-early" else 0), complete=rng.random() < 0.5)
        return sid

    t = -2 * DAY
    if name == "ahead":
        session(OTHER, 8 * DAY, SALT)
    if rng.random() < 0.5:
        b.period_open("before", LANE, t)
        session(LANE, t + 600)
        b.period_close("before", t + 1200)
    if name == "straddle":
        b.period_open("p-0", LANE, -DAY)
        session(LANE, -DAY + 600)
    t = 600
    periods = rng.randint(1, 2)
    for k in range(periods):
        last = k == periods - 1
        if not (name == "straddle" and k == 0):
            b.period_open(f"p-{k}", LANE, t)
        for w in range(rng.randint(1, 2)):
            t += 3600
            if rng.random() < 0.5:
                session(OTHER, t - 1800, rng.choice([DRAW, SALT]))
            if name == "set-back" and last and w == 0:
                session(LANE, -DAY + t)
            else:
                session(LANE, t, ended=not (name == "unended" and last and w == 0))
        if name == "foreign" and last:
            session(LANE, t + 300, SALT)
        if name == "unopened" and last and not one_step:
            b.reserve(LANE, t + 400)
        t += 3600
        if not (name == "open-period" and last):
            b.period_close(f"p-{k}", t)
        if not one_step and (name == "outside" and last or name != "open-period" and rng.random() < 0.5):
            t += 3600
            session(LANE, t, DRAW if name == "outside" else rng.choice([SALT, [["pouw/draw", 1]], [], None]))
    if one_step and rng.random() < 0.5:
        b.period_open("after", OTHER, 7 * DAY + 5)
    else:
        b.reserve(OTHER, 7 * DAY + rng.randint(0, 3600), SALT)
    lines = b.lines()
    j = request(lines, label="nope" if name == "no-window" else LABEL)
    if name == "early":
        j["ledger"] = lines[:-1]
    elif name == "coin":
        k = rng.randrange(len(lines))
        e = json.loads(lines[k])
        e["sig"] = e["sig"][:-1] + ("0" if e["sig"][-1] != "0" else "1")
        j["ledger"] = lines[:k] + [Ch.ledger_text(e)] + lines[k + 1:]
    elif name == "request":
        what = rng.choice(["keys", "label", "lane", "start", "stop", "order", "key"])
        if what == "keys":
            j = rng.choice([{k: v for k, v in j.items() if k != "lane"}, {**j, "x": 1}, None, []])
        elif what == "order":
            j["start"], j["stop"] = rng.choice([(STOP, START), (START, START)])
        elif what == "key":
            j["key"] = rng.choice([None, {"W": W, "H": H}, {**KEY_JSON, "public": "zz"}])
        else:
            j[what] = rng.choice([None, 5, "", ["x"]] + (["2026-03-02", "2026-02-30T00:00:00Z"] if what in ("start", "stop")
                                                         else []))
    return j


INPUTS = st.builds(case, st.randoms(use_true_random=False),
                   st.sampled_from(["honest"] * 4 + list(NAMES) + REQUESTS * 2))
PERIOD = D.Port(PIECES, "period_windows", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(PERIOD)


def _answer(pieces, j: Any) -> dict:
    kind, text = D.replay(PERIOD, j, pieces)
    assert kind == "output", (kind, text)
    return json.loads(text)


@pytest.mark.slow
def test_the_reader_agrees_and_each_case_is_answered_as_named(pieces):
    """Lean and the reference agree on sampled cases, and each named case is accepted or refused under its own code
    over a few seeds."""
    stats = D.run(PERIOD, examples=60, server=pieces)
    assert stats.outputs > 30 and stats.rejects > 0, stats
    for name, code in NAMES.items():
        for seed in range(3):
            got = _answer(pieces, case(random.Random(f"{name}/{seed}"), name))
            assert got.get("code") == code, (name, seed, got.get("code"), got.get("refused"))
            assert code is not None or got["windows"], (name, seed)


def test_a_small_ledger_reads_as_written(pieces):
    """A ledger written by hand reads as its entries say: the windows in the ledger's order with their periods, and the
    lane's other sessions with their layout, place among the lane's reserves and period, a session before the
    monitoring period and another lane's left out."""
    b = Book(random.Random("small"))
    b.period_open("old", LANE, -DAY)
    before = b.window(LANE, -DAY + 600)
    b.period_close("old", -DAY + 1200)
    b.period_open("p", LANE, 100)
    first = b.window(LANE, 200)
    b.window(OTHER, 300)
    second = b.window(LANE, 400, withheld=False)
    b.period_close("p", 900)
    third = b.window(LANE, 1000, [["pouw/draw", 1]])
    b.period_open("q", LANE, 2000)
    dead = b.reserve(LANE, 2100, SALT)
    b.period_close("q", 2200)
    fourth = b.window(LANE, 3000, None, withheld=False)
    b.reserve(OTHER, 7 * DAY, SALT)

    def root(sid: str) -> str:
        return b.sessions[sid]["root"]

    def session(sid: str, families: Any, place: int, period: str | None, withheld: bool) -> dict:
        return {"families": families, "period": period, "place": place, "root": root(sid), "session": sid,
                "withheld": withheld}

    want = {"others": [session(third, [["pouw/draw", 1]], 3, None, True), session(fourth, None, 5, None, False)],
            "sessions": [session(first, DRAW, 1, "p", True), session(second, DRAW, 2, "p", False),
                         session(third, [["pouw/draw", 1]], 3, None, True), session(fourth, None, 5, None, False)],
            "windows": [{"period": "p", "phase": 0, "root": root(first), "session": first},
                        {"period": "p", "phase": 1, "root": root(second), "session": second}]}
    assert before not in json.dumps(want) and dead not in json.dumps(want)
    assert reference(request(b.lines())) == want
    assert _answer(pieces, request(b.lines())) == want


@pytest.mark.slow
def test_monitoring_periods_one_after_another_split_the_sessions(pieces):
    """Whatever the stamps, monitoring periods that follow one another list each root the lane's opens name exactly
    once between them: ledgers whose entries carry stamps in no order, read at three periods end to end, give the
    opens from the first entry stamped at or after the first start up to the first stamped at or after the last stop,
    in the ledger's order, none twice and none left out."""
    ends = [at(k * DAY) for k in range(4)]
    for seed in range(12):
        rng = random.Random(f"split/{seed}")
        b = Book(rng)
        for _ in range(rng.randint(8, 14)):
            lane = rng.choice([LANE, LANE, OTHER])
            sid = b.reserve(lane, rng.randint(-DAY, 3 * DAY), rng.choice([DRAW, SALT]))
            if rng.random() < 0.85:
                b.open(sid, b.sessions[sid]["t"] + rng.randint(0, 3600))
                if rng.random() < 0.7:
                    b.end(sid, rng.randint(-DAY, 4 * DAY))
        b.reserve(OTHER, 3 * DAY + 1, SALT)
        lines = b.lines()
        es = [json.loads(line) for line in lines]
        lo, hi = reaches(es, ends[0]), reaches(es, ends[3])
        want = [e["body"]["window"]["root"] for e in es[lo:hi] if e["kind"] == "open" and Ch.lane_of(e) == LANE]
        got: list[str] = []
        for start, stop in zip(ends, ends[1:]):
            answer = _answer(pieces, request(lines, start=start, stop=stop))
            assert answer.get("code") in (None, "missing", "outside", "period"), answer
            got += [s["root"] for s in answer["sessions"]]
        assert got == want and len(set(got)) == len(got), seed


@pytest.mark.parametrize("j", [None, [], {}, request([]) | {"x": 1}, request([], label=""), request([], lane=""),
                               request([], start=STOP, stop=START), request([], stop="2026-03-09"),
                               request([]) | {"key": None}], ids=lambda j: D.canonical(j)[:40])
def test_a_malformed_request_is_rejected_by_both(pieces, j):
    with pytest.raises(D.Reject):
        reference(j)
    kind, _why = pieces.call("period_windows", j)
    assert kind == "reject"
