"""The auditor host's challenger service (vy-challenger-1; RT4-INT-17, -18, -24, -25, -27, -29, -30): `serve.py` runs only as
the `auditor` user, from a tree infra pinned, one session a window, each recorded in a signed append-only ledger before its
first coin.

    vy-challenger-gate reserve --peer NAME --scheme S --families JSON [--epoch E] [--minutes M] [--for LANE]
    vy-challenger-gate open --session S --root R
    vy-challenger-gate period-open --name P --cap N --circuit SHA512 [--lane LANE] | period-close --name P
    vy-challenger-gate status SESSION | ledger | pubkey | fetch SESSION

- **Who reaches it.** Every key in `research`'s authorized_keys on the host is forced to `vy-challenger-gate` (`command=`),
  which is this file's `gate` as root through one sudoers line. A lane's own key is forced with the lane's name
  (`vy-challenger-gate LANE`, from `/etc/vy/challenger-gate-keys`, `vy-challenger-setup`), and only such a key reserves and
  opens a window: the reserve records that name as `by`, its prover. A key forced with no name (the shared lane key, which
  the control pod's publisher holds) reads only: `status`, `ledger`, `pubkey`, `fetch`. A lane names a window and gets a
  session; it never names code. Infra's `vy-admin` runs the same gate (`sudo vy-challenger-gate VERB ...`, `by` vy-admin),
  and it alone may reserve a window for a lane that has no key of its own (`--for LANE`: the reserve records `for` and
  keeps `by` vy-admin, and the window's prover is LANE). Nothing else runs as `research` there, and nothing runs as
  `auditor` but the session unit, `reserve` and `admit`.
- **What runs.** `/opt/vy-challenger/current` -> `/opt/vy-challenger/<commit>`, a root-owned `git archive` of that commit
  (`pin.sh`); the commit is the directory's name, and every entry names it (`pinned`). The key (`/var/lib/vy-challenger/key`,
  serve.py's layout) and every session's `out/` are `auditor`'s, in a directory only `auditor` and root can enter.
- **A window** is opened in two steps, each a signed entry appended under the ledger's lock (or refused) and fsynced.
  `reserve` takes the request (peer, scheme, families, epoch, minutes) and appends the session's `reserve` entry with its
  `started_utc` and `expires_utc` (start + minutes); nothing serves yet. From that entry on the window is one of its
  lane's open period, counted toward the period's cap. `open --session S --root R` (R: 128 lowercase hex digits, for
  SALT-WIRE's `window_open_root(weights' root, round-0 request table's root, S)`, which the verifier recomputes) is
  refused unless S is reserved, not yet opened, unexpired, the opener's own (its reserve's `by` or `for`) and its
  period still open, and R is no open's root yet (v1's `commit` and one-step v2 opens included): one session a root,
  ever. `admit` appends the `open` entry, which binds S (`session`), R (`window.root`) and the reserve (`reserve`, its
  `seq`), carries the reserve's request, names its opener (`opened_by`) and takes a port, before `serve.py` starts, so
  before its first coin. A reserve not opened by its expiry is its period's refused window. Its `host` is the string the
  challenger's own record signs (`<host>.<session>`), which ties `record.json` to it. A window's time is the auditor's,
  never the requester's: its entries' `seq` and times. `serve.py` gets the root (`--root`: a salt opens only after it)
  and the session's secret (`--secret-file`), which the gate returns to the opener alone and which the client's hello
  must carry; the open entry keeps its SHA-512, `fetch` never ships it, and the session deletes it when it ends. The
  open's answer also carries the session's coin commitment root (`commitment`, from the record serve.py signs before it
  listens; `status` gives it too), so the serving side checks the hello's commitment against it before its first coin.
  Before the first reserve the ledger held one-step v2 opens (`--commit`), which still verify; none follows a reserve.
- **An `end` entry** follows when `serve.py` exits: its exit code, whether its client closed it, the SHA-512 of each file of
  `out/` (which seals the release log), how many coins each family released (`released`, counted from the release log under
  the signed record's commitment; `complete` when every family released all its coins, so a period can refuse a short
  release), the leaf the challenger's record was signed at (`record_leaf`), its commitment root (`commitment`) and how
  many hellos serve.py refused before it accepted one (`refused_hellos`, each in `out/hellos.jsonl`, which `files` seals). A session with an `open` and no `end` never
  ended (its host stopped): a refused window to the verifier, like one whose `closed` or `complete` is false.
- **A period** is a lane's span of windows between two signed entries, `period-open` and `period-close` (session null),
  which the lane's own gate key writes for its lane (`--name P`), or vy-admin for any lane (`period-open --name P --lane
  LANE`, `period-close --name P`). A period's name is used once, and a lane has one period open at a time. Its open
  commits, before window 0's first coin, the most windows it may hold (`--cap`, a positive integer) and its circuit
  (`--circuit`: the declared circuit's SHA-512, never a Program's id, which the gate records and CHAL1 checks each window
  against). Its windows are exactly the reserves (and one-step v2 opens) whose prover (`for`, else `by`) is its
  lane between its open and its close, at most `cap` (the next reserve is refused); every one counts. The close is
  refused while one of them is reserved and unexpired or still runs, and lists each, in order, as {root, session,
  complete} (root null for a reserve never opened; complete when its end is in the ledger and complete), with `complete` true only when it lists at least one window and all are: an
  incomplete window is the period's visible failure. Period entries are taken under the gate's lock, so no window is
  admitted between a close's reading of the ledger and its entry.
- **The ledger** (`/var/lib/vy-challenger/windows.jsonl`, `chattr +a`) is one JSON object a line: `schema`, `seq` (from 0),
  `prev` (SHA-512 of the previous line's bytes, 128 zeros first), `kind`, `session`, `body` and `sig`, an LMS signature under
  the challenger key over `entry_message(I, entry)`. Its entries' LMS leaves strictly increase (`append` refuses an entry
  whose leaf isn't after the last one, as a restored key state would sign), and each record's leaf lies between its
  session's open and end and is no entry's. v2 entries follow the host's v1 entries in one chain, and `verify_ledger`, the
  reference check, reads both, and checks each window's reserve, open and end in that order, its expiry, and the periods
  (`verify_periods` returns them). Infra publishes the ledger and each ended session to the store (`publish.sh`, the control
  pod's watchers loop `challenger-publish`), each put an extension of the last (`publish.sh` refuses any other head, and
  `extends` is the check), labelled `challenger-ledger head` by infra.

Every signature of the key (`reserve`'s, `admit`'s, `serve.py`'s record, `end`'s, a period's) is taken under `sign.lock`, so `lms.LeafState`, which
refuses a second signer, never sees two.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]
sys.path.insert(0, str(TREE))

SCHEMA_V1 = "verity/challenger-auditor/v1"
SCHEMA = "verity/challenger-auditor/v2"
ROLE = "verity/challenger-auditor-record/v1"
TAGS = {SCHEMA_V1: "verity/challenger-auditor/entry/v1", SCHEMA: "verity/challenger-auditor/entry/v2"}
ZERO = "0" * 128
PORTS = range(7811, 7900)
MINUTES = (1, 720)
ROOT = re.compile(r"[0-9a-f]{128}")
HEX = re.compile(r"(?:[0-9a-f]{2})+")
PINNED = re.compile(r"[0-9a-f]{40}")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
SCHEME = re.compile(r"[a-z0-9][a-z0-9/._-]{0,127}")
SESSION = re.compile(r"s\d{8}T\d{6}Z-[0-9a-f]{4}")
READ_ONLY = "-"
PERIOD_KINDS = ("period-open", "period-close")
SESSION_KINDS = ("reserve", "open", "end")
RESERVED = ("by", "for", "peer", "scheme", "epoch", "families", "minutes")
UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
CAP = (1, 2 ** 24)
CIRCUIT = re.compile(r"[0-9a-fA-F]{128}")
HOST_ACCOUNTS = ("root", "vy-admin", "research", "auditor")
SECRET = "secret"
OPEN_REQUEST = "open.json"
OUT_FILES = ("record.json", "key.json", "releases.jsonl", "hellos.jsonl", "challenger.json", "address.json", "tunnel.log",
             "serve.log")


class Refused(Exception):
    """A request the gate or the session refuses, with why."""


@dataclass(frozen=True)
class Host:
    """Where the service keeps its state, and the name its records give the host."""

    root: Path = Path("/var/lib/vy-challenger")
    peers: Path = Path("/etc/vy/peers")
    name: str = "vy-challenger-1"

    @property
    def ledger(self) -> Path:
        return self.root / "windows.jsonl"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def key(self) -> Path:
        return self.root / "key"


def utc(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def later(t: str, minutes: int) -> str:
    """`t` (`utc`'s form) plus `minutes`: a reserve's expiry."""
    return utc(datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) + timedelta(minutes=minutes))


def pinned_commit(tree: Path = TREE) -> str:
    """The commit this tree was pinned at: its directory's name under /opt/vy-challenger."""
    if not PINNED.fullmatch(tree.name):
        raise Refused(f"{tree} is not a pinned tree (/opt/vy-challenger/<commit>, pin.sh): no session runs from it")
    return tree.name


# --- the ledger ---------------------------------------------------------------------------------------------------------------

def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def entry_message(I: bytes, entry: dict) -> bytes:
    """What an entry's `sig` signs: SHA-512([ROLE] || I || [identity_digest_sha512(TAG, entry without sig)]), each [x] x
    prefixed with its length (u64), as `challenger.record_message` frames the challenger's record under its own role; TAG is
    the entry's schema's (`TAGS`)."""
    from verity.ml.commitments.identity import identity_digest_sha512
    if len(I) != 16:
        raise ValueError(f"an LMS identifier is 16 bytes, not {len(I)}")
    tag = TAGS.get(entry.get("schema"))
    if tag is None:
        raise ValueError(f"schema {entry.get('schema')!r} is none of {sorted(TAGS)}")
    body = {k: v for k, v in entry.items() if k != "sig"}
    return hashlib.sha512(_field(ROLE.encode()) + I + _field(identity_digest_sha512(tag, body))).digest()


def leaf_of(sig: Any) -> int:
    """The LMS leaf a hex signature was made at (its first 4 bytes, `lms.Signature.encode`)."""
    if type(sig) is not str or len(sig) < 8 or not HEX.fullmatch(sig):
        raise ValueError("a signature is lowercase hex")
    return int(sig[:8], 16)


def read_ledger(path: Path) -> list[bytes]:
    try:
        return [ln for ln in path.read_bytes().split(b"\n") if ln]
    except FileNotFoundError:
        return []


def entries(lines: list[bytes]) -> list[dict]:
    return [json.loads(ln) for ln in lines]


def sessions_of(lines: list[bytes]) -> dict[str, dict[str, dict]]:
    """session -> {"reserve": entry, "open": entry, "end": entry} as far as the ledger has them."""
    out: dict[str, dict[str, dict]] = {}
    for e in entries(lines):
        if e["kind"] in SESSION_KINDS:
            out.setdefault(e["session"], {})[e["kind"]] = e
    return out


def lane_of(e: dict) -> str:
    """A reserve's or an open's prover: the lane vy-admin reserved it for (`for`), else its requester (`by`)."""
    return e["body"].get("for") or e["body"]["by"]


def circuit_of(s: str) -> str:
    """A period's circuit as its open records it: the declared circuit's SHA-512 (128 hex digits, kept in lowercase).
    The verifier of record refuses a Program's id there."""
    if type(s) is not str or not CIRCUIT.fullmatch(s):
        raise Refused(f"--circuit {s!r}: the declared circuit's SHA-512 (128 hex digits), not a Program's id")
    return s.lower()


def cap_of(v: Any) -> int:
    """A period's cap: the most windows it may hold, a positive integer."""
    if type(v) is not int or not CAP[0] <= v <= CAP[1]:
        raise Refused(f"--cap {v!r}: the most windows the period may hold, {CAP[0]} to {CAP[1]}")
    return v


def counts_at(e: dict) -> bool:
    """Whether a window joins its lane's open period at this entry: at its reserve, or at a v2 open with no reserve (a
    one-step open, before the first reserve)."""
    return e["kind"] == "reserve" or (e["kind"] == "open" and e["schema"] == SCHEMA and "reserve" not in e["body"])


def periods_of(lines: list[bytes]) -> dict[str, dict]:
    """period -> {"lane", "open", "close" (None while open), "windows": [session, ...], "cap", "circuit"}: a period's
    windows are exactly its lane's reserves (and one-step v2 opens) between its period-open and its period-close."""
    periods: dict[str, dict] = {}
    current: dict[str, dict] = {}
    for e in entries(lines):
        if e["kind"] == "period-open":
            p = current[e["body"]["lane"]] = periods[e["body"]["period"]] = {
                "lane": e["body"]["lane"], "open": e, "close": None, "windows": [], "cap": e["body"].get("cap"),
                "circuit": e["body"].get("circuit")}
        elif e["kind"] == "period-close":
            p = periods[e["body"]["period"]]
            p["close"] = e
            current.pop(p["lane"], None)
        elif counts_at(e) and (p := current.get(lane_of(e))) is not None:
            p["windows"].append(e["session"])
    return periods


def window_outcomes(lines: list[bytes], windows: list[str]) -> list[dict]:
    """Each window's outcome as a period-close records it: its root (None when it was never opened, as an expired reserve
    isn't), complete only when its end is in the ledger and complete."""
    sessions = sessions_of(lines)
    return [{"root": root_of(sessions[s]["open"]) if "open" in sessions[s] else None, "session": s,
             "complete": "end" in sessions[s] and sessions[s]["end"]["body"].get("complete") is True} for s in windows]


def root_of(e: dict) -> str:
    """An open entry's window root: v2's `window.root`, v1's `window.commit` (v1's period label names nothing)."""
    w = e["body"]["window"]
    return w["root"] if e["schema"] == SCHEMA else w["commit"]


def roots_of(lines: list[bytes]) -> dict[str, str]:
    """root -> the session its open names, for every open in the ledger."""
    return {root_of(s["open"]): sid for sid, s in sessions_of(lines).items() if "open" in s}


def extends(published: list[bytes], ledger: list[bytes]) -> bool:
    """Whether `ledger` is `published` with entries appended: a head that isn't (a fork, a cut, an edit) is refused."""
    return ledger[:len(published)] == published


@contextmanager
def locked(path: Path) -> Iterator[None]:
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append(host: Host, kind: str, session: str, body_of: Callable[[list[bytes]], dict], I: bytes,
           sign: Callable[[bytes], bytes]) -> dict:
    """Append one signed entry, its body `body_of` the ledger as it stands (which may refuse), under the ledger's lock;
    fsynced. Refused, unwritten, when the signature's leaf isn't after the ledger's last entry's (a restored key state)."""
    with locked(host.root / "ledger.lock"):
        lines = read_ledger(host.ledger)
        entry = {"schema": SCHEMA, "seq": len(lines), "prev": hashlib.sha512(lines[-1]).hexdigest() if lines else ZERO,
                 "kind": kind, "session": session, "body": body_of(lines)}
        with locked(host.root / "sign.lock"):
            entry["sig"] = sign(entry_message(I, entry)).hex()
        if lines and leaf_of(entry["sig"]) <= leaf_of(json.loads(lines[-1])["sig"]):
            raise Refused(f"the key signed at leaf {leaf_of(entry['sig'])}, not after the ledger's last entry's: its key state "
                          "was restored or forked, and no entry is written under it")
        fd = os.open(host.ledger, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, json.dumps(entry, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
    return entry


def verify_ledger(lines: list[bytes], I: bytes, verify: Callable[[bytes, bytes], bool]) -> dict[str, dict]:
    """The reference check of a published ledger under the challenger key: every entry v1 or v2 (no v1 after a v2) and signed,
    `seq` from 0, `prev` chained, its leaves strictly increasing; a `reserve` first in its session, its expiry its start
    plus its minutes; one `open` a root, whatever its period label, in canonical lowercase, and after the first reserve
    each open binds its session's reserve (`reserve`, its seq), carries its request, is by its prover and falls inside its
    minutes and before its period's close; one `end` a session, after its `open`, and a v2 end's `record_leaf` between its
    session's open and end and no entry's or other record's; periods as `verify_periods` checks them. Returns root ->
    {"reserve" (None for a one-step open), "open", "end" or None}: S is the open's `session`, R its root; raises
    ValueError at the first entry that fails. It reads one head: a verifier takes infra's newest published head, and refuses
    two that don't `extend` one another."""
    return _verify(lines, I, verify)[0]


def verify_periods(lines: list[bytes], I: bytes, verify: Callable[[bytes, bytes], bool]) -> dict[str, dict]:
    """The ledger's periods once `verify_ledger` passes: period -> {"lane", "open", "close" (None while open), "windows":
    [session, ...], "cap", "circuit"}. A period's name is used once; a lane has one period open at a time; its windows are
    exactly the reserves (and one-step v2 opens) whose prover is its lane (`lane_of`) between its period-open and its
    period-close, counted at the reserve, so a reserve that expired unopened is one, at most its
    `cap` (a positive integer, with `circuit` committed beside it before window 0's first coin; both None only for a
    period opened before they were required, which a reader of the cap refuses), and the close lists each of them,
    in order, with its root (null when never opened) and outcome (complete when its end precedes the close and is
    complete) and is complete only when it
    has a window and every one is. An end after its period's close (the gate's lock leaves none) changes nothing: the
    close recorded that window incomplete."""
    return _verify(lines, I, verify)[1]


def _verify(lines: list[bytes], I: bytes, verify: Callable[[bytes, bytes], bool]) -> tuple[dict[str, dict], dict[str, dict]]:
    windows: dict[str, dict] = {}
    by_session: dict[str, str] = {}
    leaves: set[int] = set()
    records: dict[int, int] = {}
    periods: dict[str, dict] = {}
    current: dict[str, str] = {}
    reserves: dict[str, dict] = {}
    prev, last, v2, two_step = ZERO, -1, False, False

    def join(i: int, lane: str | None, session: str) -> str | None:
        if (period := current.get(lane) if lane is not None else None) is not None:
            p = periods[period]
            if p["cap"] is not None and len(p["windows"]) >= p["cap"]:
                raise ValueError(f"entry {i}: period {period}'s window {len(p['windows'])} is past its cap {p['cap']}")
            p["windows"].append(session)
        return period

    for i, ln in enumerate(lines):
        try:
            e = json.loads(ln)
        except ValueError:
            raise ValueError(f"entry {i}: not JSON") from None
        if (type(e) is not dict or e.get("schema") not in TAGS or (v2 and e["schema"] == SCHEMA_V1) or e.get("seq") != i
                or e.get("prev") != prev):
            raise ValueError(f"entry {i}: not entry {i} of a {SCHEMA} ledger chained to its predecessor")
        v2 = v2 or e["schema"] == SCHEMA
        kind = e.get("kind")
        period_kind = kind in PERIOD_KINDS
        if (e.get("session") is not None if period_kind else type(e.get("session")) is not str) or type(e.get("body")) is not dict:
            raise ValueError(f"entry {i}: a session's entry has a session, a period's none, and each a body")
        q = leaf_of(e.get("sig"))
        if not verify(entry_message(I, e), bytes.fromhex(e["sig"])):
            raise ValueError(f"entry {i}: its signature doesn't verify under the challenger key")
        if q <= last:
            raise ValueError(f"entry {i}: signed at leaf {q}, not after the last entry's {last}")
        if kind == "reserve":
            b, session = e["body"], e["session"]
            if e["schema"] != SCHEMA or not SESSION.fullmatch(session) or session in reserves or session in by_session:
                raise ValueError(f"entry {i}: a reserve is v2 and the first entry of its session {session}")
            if (type(b.get("by")) is not str or not NAME.fullmatch(b["by"])
                    or ("for" in b and (type(b["for"]) is not str or not NAME.fullmatch(b["for"])))
                    or type(b.get("minutes")) is not int or not MINUTES[0] <= b["minutes"] <= MINUTES[1]
                    or type(b.get("started_utc")) is not str or not UTC.fullmatch(b["started_utc"])
                    or b.get("expires_utc") != later(b["started_utc"], b["minutes"])):
                raise ValueError(f"entry {i}: a reserve names its prover, its minutes, its start and its expiry (start + "
                                 "minutes)")
            two_step = True
            reserves[session] = {"entry": e, "period": join(i, lane_of(e), session), "opened": False}
        elif kind == "open":
            try:
                root = root_of(e)
                lane = lane_of(e) if e["schema"] == SCHEMA else None
            except (KeyError, TypeError):
                raise ValueError(f"entry {i}: an open names its window and its prover") from None
            if type(root) is not str or not ROOT.fullmatch(root):
                raise ValueError(f"entry {i}: a window root is 128 lowercase hex digits")
            if root in windows or e["session"] in by_session:
                raise ValueError(f"entry {i}: a second open for root {root[:16]}… or session {e['session']}")
            b = e["body"]
            if e["schema"] == SCHEMA and "reserve" in b:
                r = reserves.get(e["session"])
                if r is None or r["entry"]["seq"] != b["reserve"]:
                    raise ValueError(f"entry {i}: an open for session {e['session']} binds no reserve of it")
                rb = r["entry"]["body"]
                if any(b.get(k) != rb.get(k) for k in RESERVED):
                    raise ValueError(f"entry {i}: session {e['session']}'s open doesn't carry its reserve's request")
                if b.get("opened_by") not in {rb["by"], rb.get("for")} - {None}:
                    raise ValueError(f"entry {i}: session {e['session']} was opened by {b.get('opened_by')!r}, not its "
                                     "reserve's prover")
                if type(b.get("started_utc")) is not str or not UTC.fullmatch(b["started_utc"]) or not (
                        rb["started_utc"] <= b["started_utc"] <= rb["expires_utc"]):
                    raise ValueError(f"entry {i}: session {e['session']} was opened outside its reserve's minutes")
                if r["period"] is not None and periods[r["period"]]["close"] is not None:
                    raise ValueError(f"entry {i}: session {e['session']} was opened after its period {r['period']} closed")
                r["opened"] = True
            elif e["schema"] == SCHEMA and two_step:
                raise ValueError(f"entry {i}: a v2 open without a reserve after the ledger's first reserve")
            else:
                join(i, lane, e["session"])
            windows[root], by_session[e["session"]] = {"open": e, "end": None, "q": q}, root
        elif kind == "end":
            root = by_session.get(e["session"])
            if root is None or windows[root]["end"] is not None:
                raise ValueError(f"entry {i}: an end for session {e['session']}, which has no open or has ended")
            r = e["body"].get("record_leaf") if e["schema"] == SCHEMA else None
            if r is not None:
                if type(r) is not int or not windows[root]["q"] < r < q or r in records:
                    raise ValueError(f"entry {i}: record leaf {r!r} isn't between its open's and its end's, once")
                records[r] = i
            windows[root]["end"] = e
        elif period_kind:
            name, lane = e["body"].get("period"), e["body"].get("lane")
            if (e["schema"] != SCHEMA or type(name) is not str or not NAME.fullmatch(name) or type(lane) is not str
                    or not NAME.fullmatch(lane)):
                raise ValueError(f"entry {i}: a {kind} is v2 and names its period and its lane")
            if kind == "period-open":
                if name in periods or lane in current:
                    raise ValueError(f"entry {i}: period {name} is named twice, or lane {lane} has a period open")
                cap, circuit = e["body"].get("cap"), e["body"].get("circuit")
                if cap is not None or circuit is not None:
                    try:
                        if cap_of(cap) != cap or circuit_of(circuit) != circuit:
                            raise Refused("not canonical")
                    except Refused:
                        raise ValueError(f"entry {i}: period {name}'s cap is a positive integer and its circuit a "
                                         "SHA-512 in lowercase") from None
                periods[name], current[lane] = {"lane": lane, "open": e, "close": None, "windows": [], "cap": cap,
                                                "circuit": circuit}, name
            else:
                p = periods.get(name)
                if p is None or p["close"] is not None or p["lane"] != lane:
                    raise ValueError(f"entry {i}: a period-close for {name}, which isn't lane {lane}'s open period")
                outcomes = [{"root": by_session.get(s), "session": s,
                             "complete": s in by_session and windows[by_session[s]]["end"] is not None
                             and windows[by_session[s]]["end"]["body"].get("complete") is True} for s in p["windows"]]
                if (e["body"].get("windows") != outcomes
                        or e["body"].get("complete") is not (bool(outcomes) and all(w["complete"] for w in outcomes))):
                    raise ValueError(f"entry {i}: period {name}'s close doesn't list exactly its windows and their outcomes")
                p["close"] = e
                del current[lane]
        else:
            raise ValueError(f"entry {i}: kind {kind!r}")
        leaves.add(q)
        prev, last = hashlib.sha512(ln).hexdigest(), q
    if reused := sorted(leaves & set(records)):
        raise ValueError(f"entry {records[reused[0]]}: record leaf {reused[0]} is also an entry's")
    return ({root: {"reserve": (reserves.get(w["open"]["session"]) or {}).get("entry"), "open": w["open"], "end": w["end"]}
             for root, w in windows.items()},
            {name: {k: p[k] for k in ("lane", "open", "close", "windows", "cap", "circuit")} for name, p in periods.items()})


# --- the key (serve.py's) -----------------------------------------------------------------------------------------------------

def load_key(host: Host, params=None, workers: int | None = None):
    """The challenger's key and leaf state in `host.key`, made there when absent (serve.py's `load_key`)."""
    sys.path.insert(0, str(HERE))
    import serve
    from verity.ml.crypto import lms
    return serve.load_key(host.key, params or lms.DEPLOYED, workers or len(os.sched_getaffinity(0)))


def key_signer(host: Host, params=None, workers: int | None = None):
    """(I, sign, key_sha512) for `append`; each signature opens the key's leaf state and closes it after."""
    from verity.ml.crypto import lms
    key, state = load_key(host, params, workers)
    state.close()
    public = lms.public_key(key)

    def sign(message: bytes) -> bytes:
        with lms.LeafState(host.key / "state.json", public, key.params.H) as st:
            return lms.sign(key, st, message).encode()
    return public[:16], sign, hashlib.sha512(public).hexdigest()


# --- the gate (root) ----------------------------------------------------------------------------------------------------------

def peers(host: Host) -> list[str]:
    names = [ln.split()[0] for ln in host.peers.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [n for n in names if n != host.name]


def parse_reserve(argv: list[str], host: Host, *, admin: bool = False) -> dict:
    """A reserve's request; `--for LANE` (vy-admin's alone: a development window reserved for a lane without its own key)
    names the window's prover."""
    p = argparse.ArgumentParser(prog="vy-challenger-gate reserve", exit_on_error=False, add_help=False, allow_abbrev=False)
    for flag in ("--peer", "--scheme", "--families"):
        p.add_argument(flag, required=True)
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--minutes", type=int, default=120)
    if admin:
        p.add_argument("--for", dest="lane", default=None)

    def refuse(why: object) -> None:
        raise Refused(f"reserve: {why}; {' '.join(p.format_usage().split())} (no --commit, --period or --lane: the "
                      "window's root comes at its open, its time is the auditor's, its prover your gate key's lane)")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    if a.peer not in peers(host):
        raise Refused(f"--peer {a.peer!r}: one of {peers(host)} (/etc/vy/peers)")
    if not SCHEME.fullmatch(a.scheme):
        raise Refused(f"--scheme {a.scheme!r}: [a-z0-9/._-], up to 128")
    try:
        families = json.loads(a.families)
    except json.JSONDecodeError:
        families = None
    if (not isinstance(families, list) or not families or len(a.families) > 4096
            or not all(isinstance(f, list) and len(f) == 2 and isinstance(f[0], str) and SCHEME.fullmatch(f[0])
                       and type(f[1]) is int and f[1] > 0 for f in families)
            or len({f[0] for f in families}) != len(families)):
        raise Refused("--families: a JSON list of [label, count], labels distinct, count >= 1, up to 4096 bytes")
    if not 0 <= a.epoch < 2 ** 63:
        raise Refused("--epoch: 0 <= E < 2^63")
    if not MINUTES[0] <= a.minutes <= MINUTES[1]:
        raise Refused(f"--minutes: {MINUTES[0]} to {MINUTES[1]}")
    lane = getattr(a, "lane", None)
    if lane is not None and (not NAME.fullmatch(lane) or lane in HOST_ACCOUNTS):
        raise Refused(f"--for {lane!r}: a lane's name")
    return {"peer": a.peer, "scheme": a.scheme, "families": families, "epoch": a.epoch, "minutes": a.minutes,
            **({"for": lane} if lane else {})}


def parse_open(argv: list[str]) -> tuple[str, str]:
    """An open's (session, root): the reserved session and the window's root, 128 lowercase hex digits."""
    p = argparse.ArgumentParser(prog="vy-challenger-gate open", exit_on_error=False, add_help=False, allow_abbrev=False)
    p.add_argument("--session", required=True)
    p.add_argument("--root", required=True)

    def refuse(why: object) -> None:
        raise Refused(f"open: {why}; {' '.join(p.format_usage().split())} (reserve first: its answer names the session)")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    if not SESSION.fullmatch(a.session):
        raise Refused(f"--session {a.session!r}: a reserve's session (sYYYYMMDDTHHMMSSZ-xxxx)")
    if not ROOT.fullmatch(a.root):
        raise Refused("--root: the window's root, 128 lowercase hex digits (a SHA-512)")
    return a.session, a.root


def free_port(lines: list[bytes], active: Callable[[str], bool]) -> int:
    busy = {s["open"]["body"]["port"] for sid, s in sessions_of(lines).items()
            if "open" in s and "end" not in s and active(sid)}
    for port in PORTS:
        if port not in busy:
            return port
    raise Refused(f"every port of {PORTS.start}..{PORTS.stop - 1} has a live session")


def period_holding(lines: list[bytes], session: str) -> tuple[str, dict] | None:
    return next(((n, p) for n, p in periods_of(lines).items() if session in p["windows"]), None)


def cap_refusal(lines: list[bytes], lane: str) -> str | None:
    """Why the lane's next window would be past its open period's cap, or None."""
    for name, p in periods_of(lines).items():
        if p["lane"] == lane and p["close"] is None and p["cap"] is not None and len(p["windows"]) >= p["cap"]:
            return (f"period {name} holds its cap of {p['cap']} windows: close it (period-close --name {name}) and open "
                    "another for more")
    return None


def open_refusal(lines: list[bytes], session: str, opener: str, root: str, now: datetime) -> str | None:
    """Why `opener` may not open `session` at `root` now, or None: the session is reserved, not yet opened, unexpired,
    the opener's own (its reserve's `by` or `for`), its period still open, and the root no open's yet (v1 and v2 too)."""
    s = sessions_of(lines).get(session, {})
    if "reserve" not in s:
        return f"session {session} has no reserve in the ledger: reserve first"
    if "open" in s:
        return f"session {session} is opened already: a reserve opens once"
    rb = s["reserve"]["body"]
    if opener not in {rb["by"], rb.get("for")} - {None}:
        return f"session {session} is {lane_of(s['reserve'])}'s reserve, not {opener}'s"
    if utc(now) > rb["expires_utc"]:
        return f"session {session}'s reserve expired at {rb['expires_utc']}: it is a refused window of its period"
    if (held := period_holding(lines, session)) is not None and held[1]["close"] is not None:
        return f"session {session}'s period {held[0]} has closed, listing it unopened"
    if (sid := roots_of(lines).get(root)) is not None:
        return f"root {root[:16]}… already has its session ({sid}) in the ledger: one session a root, ever"
    return None


def gate_reserve(argv: list[str], host: Host, *, by: str, admin: bool, now: datetime, token: str,
                 reserve: Callable[[str], None], own: Callable[[Path], None]) -> dict:
    """Validate a request and hand it to `reserve` (as `auditor`), which signs the session's reserve entry: the session
    id, its expiry and the period it is a window of. Nothing serves until its open."""
    req = parse_reserve(argv, host, admin=admin)
    lines = read_ledger(host.ledger)
    if (why := cap_refusal(lines, req.get("for") or by)) is not None:
        raise Refused(why)
    session = f"s{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{token}"
    d = host.sessions / session
    d.mkdir(mode=0o755)
    req.update(session=session, by=by, requested_utc=utc(now))
    (d / "request.json").write_text(json.dumps(req, indent=1, sort_keys=True) + "\n")
    own(d)
    with locked(host.root / "gate.lock"):
        reserve(session)
    lines = read_ledger(host.ledger)
    entry = sessions_of(lines)[session]["reserve"]
    held = period_holding(lines, session)
    return {"session": session, "seq": entry["seq"], "peer": req["peer"], "by": by,
            **({"for": req["for"]} if "for" in req else {}), "expires_utc": entry["body"]["expires_utc"],
            "period": held and held[0]}


def gate_open(argv: list[str], host: Host, *, by: str, now: datetime, admit: Callable[[str], None],
              start: Callable[[str], None], own: Callable[[Path], None], secret: bytes,
              ready: Callable[[str], str | None]) -> dict:
    """Open a reserved session at its window's root: hand it to `admit` (as `auditor`), which signs the open entry
    binding the session and the root, and `start` its unit (serve.py with --root, so the salt opens only after it): the
    session's port on the peer, its secret, which only this answer carries, and its coin commitment's root once `ready`
    has it (serve.py signs its record before it listens), so the serving side checks the hello's commitment against it
    before its first coin."""
    session, root = parse_open(argv)
    d = host.sessions / session
    with locked(host.root / "gate.lock"):
        if (why := open_refusal(read_ledger(host.ledger), session, by, root, now)) is not None:
            raise Refused(why)
        fd = os.open(d / SECRET, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o400)
        try:
            os.write(fd, secret.hex().encode() + b"\n")
        finally:
            os.close(fd)
        req = {"session": session, "root": root, "by": by, "requested_utc": utc(now),
               "secret_sha512": hashlib.sha512(secret).hexdigest()}
        (d / OPEN_REQUEST).write_text(json.dumps(req, indent=1, sort_keys=True) + "\n")
        own(d / SECRET)
        try:
            admit(session)
        except Refused:
            for f in (SECRET, OPEN_REQUEST):
                (d / f).unlink(missing_ok=True)
            raise
        start(session)
    entry = sessions_of(read_ledger(host.ledger))[session]["open"]
    b = entry["body"]
    return {"session": session, "port": b["port"], "peer": b["peer"], "host": b["host"], "seq": entry["seq"],
            "reserve": b["reserve"], "root": root, "by": b["by"], **({"for": b["for"]} if "for" in b else {}),
            "opened_by": by, "secret": secret.hex(), "commitment": ready(session)}


def gate_status(session: str, host: Host, active: Callable[[str], bool], now: datetime) -> dict:
    if not SESSION.fullmatch(session):
        raise Refused(f"no session {session!r}")
    s = sessions_of(read_ledger(host.ledger)).get(session)
    if s is None:
        raise Refused(f"no session {session} in the ledger")
    expired = "open" not in s and "reserve" in s and utc(now) > s["reserve"]["body"]["expires_utc"]
    return {"session": session, "reserve": s.get("reserve"), "open": s.get("open"), "end": s.get("end"),
            "active": "open" in s and "end" not in s and active(session), "expired": expired,
            "commitment": commitment_of(host.sessions / session / "out")}


def gate_fetch(session: str, host: Host) -> bytes:
    """The session's directory (its request and out/, never its secret) as a gzipped tar."""
    gate_status(session, host, lambda _: False, datetime.now(timezone.utc))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(host.sessions / session, arcname=session,
                filter=lambda ti: None if ti.name == f"{session}/{SECRET}" else ti)
    return buf.getvalue()


USAGE = ("verbs: reserve --peer NAME --scheme S --families JSON [--epoch E] [--minutes M] [--for LANE: vy-admin's] "
         "| open --session S --root R "
         "| period-open --name P --cap N --circuit SHA512 [--lane LANE: vy-admin's] | period-close --name P "
         "(a lane's own gate key, or vy-admin) "
         "| status SESSION | ledger | pubkey | fetch SESSION")


def parse_period(verb: str, argv: list[str], *, lane: str | None) -> tuple[str, str | None, list[str]]:
    """(period, lane, [cap, circuit] for a period-open) of a period-open or period-close: a lane key's own lane, or
    vy-admin's `--lane` (period-open)."""
    p = argparse.ArgumentParser(prog=f"vy-challenger-gate {verb}", exit_on_error=False, add_help=False, allow_abbrev=False)
    p.add_argument("--name", required=True)
    if lane is None and verb == "period-open":
        p.add_argument("--lane", required=True)
    if verb == "period-open":
        p.add_argument("--cap", required=True)
        p.add_argument("--circuit", required=True)

    def refuse(why: object) -> None:
        raise Refused(f"{verb}: {why}; {' '.join(p.format_usage().split())}")

    p.error = refuse
    try:
        a = p.parse_args(argv)
    except argparse.ArgumentError as e:
        refuse(e)
    named = getattr(a, "lane", None) or lane
    for what, v in (("--name", a.name), ("--lane", named)):
        if v is not None and (not NAME.fullmatch(v) or v in HOST_ACCOUNTS):
            raise Refused(f"{verb}: {what} {v!r}: 1 to 64 of [A-Za-z0-9._:-], starting with a letter or digit")
    if verb != "period-open":
        return a.name, named, []
    cap = cap_of(int(a.cap)) if re.fullmatch(r"[1-9][0-9]{0,8}", a.cap) else cap_of(a.cap)
    return a.name, named, [str(cap), circuit_of(a.circuit)]


def gate(argv: list[str], host: Host = Host(), **hooks: Any) -> tuple[int, bytes]:
    """The gate's verbs; (exit code, stdout). A refusal is exit 3 with {"refused": why}. Over ssh (`--ssh WORDS --key NAME`,
    the forced command's) `by` is NAME, the lane the key was registered for, and a key registered with no lane (`-`) reads
    only; run by root (vy-admin's sudo) `by` is the sudoer."""
    def refused(why: str) -> tuple[int, bytes]:
        return 3, json.dumps({"refused": why}).encode() + b"\n"

    by = hooks.get("by") or os.environ.get("SUDO_USER") or "root"
    admin = argv[:1] != ["--ssh"]
    if argv[:1] == ["--ssh"]:
        if len(argv) == 4 and argv[2] == "--key":
            key = argv[3]
        elif len(argv) == 2:
            key = READ_ONLY
        else:
            return refused("the gate's forced command is --ssh WORDS --key NAME")
        if key != READ_ONLY and not NAME.fullmatch(key):
            return refused(f"gate key name {key!r}")
        by = None if key == READ_ONLY else key
        if len(argv[1]) > 8192:
            return refused("the request is over 8192 bytes")
        try:
            argv = shlex.split(argv[1])
        except ValueError as e:
            return refused(f"the request doesn't parse as shell words: {e}")
    verb, rest = (argv[0], argv[1:]) if argv else ("", [])
    active = hooks.get("active") or unit_active
    now = hooks.get("now") or datetime.now(timezone.utc)
    try:
        if verb in ("reserve", "open") and by is None:
            raise Refused("this key reads only: a window is reserved and opened with its prover's own gate key, which "
                          "infra registers for its lane (send infra the public half)")
        if verb == "reserve":
            doc = gate_reserve(rest, host, by=by, admin=admin, now=now, token=hooks.get("token") or secrets.token_hex(2),
                               reserve=hooks.get("reserve") or (lambda s: as_auditor_step("reserve", s)),
                               own=hooks.get("own") or own_by_auditor)
        elif verb == "open":
            doc = gate_open(rest, host, by=by, now=now, admit=hooks.get("admit") or (lambda s: as_auditor_step("admit", s)),
                            start=hooks.get("start") or start_unit, own=hooks.get("own") or own_by_auditor,
                            secret=hooks.get("secret") or secrets.token_bytes(32),
                            ready=hooks.get("ready") or (lambda s: await_commitment(host, s, active)))
        elif verb in PERIOD_KINDS:
            if by is None:
                raise Refused("this key reads only: a period is opened and closed with its lane's own gate key, or by vy-admin")
            name, lane, extra = parse_period(verb, rest, lane=None if admin else by)
            with locked(host.root / "gate.lock"):
                e = (hooks.get("period") or as_auditor_period)([verb, name, lane or READ_ONLY, by, *extra])
            doc = {"seq": e["seq"], "kind": e["kind"], "body": e["body"]}
        elif verb == "status" and len(rest) == 1:
            doc = gate_status(rest[0], host, active, now)
        elif verb == "ledger" and not rest:
            return 0, host.ledger.read_bytes() if host.ledger.exists() else b""
        elif verb == "pubkey" and not rest:
            doc = json.loads((host.root / "pubkey.json").read_text())
        elif verb == "fetch" and len(rest) == 1:
            return 0, gate_fetch(rest[0], host)
        else:
            raise Refused(USAGE)
    except Refused as e:
        return refused(str(e))
    return 0, json.dumps(doc, sort_keys=True).encode() + b"\n"


def _auditor(args: list[str]) -> list[str]:
    return ["runuser", "-u", "auditor", "--", "/usr/bin/env", "-i", "PYTHONDONTWRITEBYTECODE=1", "HOME=/var/lib/vy-challenger",
            "PATH=/usr/bin:/bin", "/usr/bin/python3", "-I", str(Path(__file__).resolve()), *args]


def as_auditor_step(verb: str, session: str) -> None:
    p = subprocess.run(_auditor([verb, session]), capture_output=True, text=True)
    if p.returncode:
        raise Refused(p.stderr.strip().splitlines()[-1].removeprefix("refused: ") if p.stderr.strip()
                      else f"{verb} exited {p.returncode}")


def as_auditor_period(args: list[str]) -> dict:
    p = subprocess.run(_auditor(args), capture_output=True, text=True)
    if p.returncode:
        raise Refused(p.stderr.strip().splitlines()[-1].removeprefix("refused: ") if p.stderr.strip()
                      else f"{args[0]} exited {p.returncode}")
    return json.loads(p.stdout)


def own_by_auditor(path: Path) -> None:
    import pwd
    pw = pwd.getpwnam("auditor")
    os.chown(path, pw.pw_uid, pw.pw_gid)


def start_unit(session: str) -> None:
    """Blocks until the unit runs (Type=exec), so the next `admit`, under the gate's lock, sees it active."""
    p = subprocess.run(["systemctl", "start", f"vy-challenger@{session}.service"], capture_output=True, text=True)
    if p.returncode:
        raise Refused(f"session {session} was admitted but its unit didn't start ({p.stderr.strip()}): its window is spent, "
                      f"an open entry with no end")


def unit_active(session: str) -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", f"vy-challenger@{session}.service"]).returncode == 0


# --- the session (auditor) ----------------------------------------------------------------------------------------------------

def request_of(host: Host, session: str, name: str = "request.json") -> dict:
    if not SESSION.fullmatch(session):
        raise Refused(f"no session {session!r}")
    return json.loads((host.sessions / session / name).read_text())


def reserve(host: Host, session: str, *, signer, pinned: str, now: datetime) -> dict:
    """The window's `reserve` entry, or Refused when the session is in the ledger or the lane's open period holds its
    cap. From this entry on the window is one of that period's; it expires unopened `minutes` after `started_utc`."""
    req = request_of(host, session)
    I, sign, key_sha512 = signer

    def body_of(lines: list[bytes]) -> dict:
        if session in sessions_of(lines):
            raise Refused(f"session {session} is in the ledger already")
        if (why := cap_refusal(lines, req.get("for") or req["by"])) is not None:
            raise Refused(why)
        return {"by": req["by"], **({"for": req["for"]} if "for" in req else {}), "peer": req["peer"],
                "scheme": req["scheme"], "epoch": req["epoch"], "families": req["families"], "minutes": req["minutes"],
                "pinned": pinned, "key_sha512": key_sha512, "host": f"{host.name}.{session}", "started_utc": utc(now),
                "expires_utc": later(utc(now), req["minutes"])}
    return append(host, "reserve", session, body_of, I, sign)


def admit(host: Host, session: str, *, signer, pinned: str, now: datetime, active: Callable[[str], bool]) -> dict:
    """The window's `open` entry, binding the reserved session and its root, carrying its reserve's request and a port no
    live session holds; or Refused (`open_refusal`, read under the ledger's lock)."""
    req = request_of(host, session, OPEN_REQUEST)
    I, sign, key_sha512 = signer

    def body_of(lines: list[bytes]) -> dict:
        if (why := open_refusal(lines, session, req["by"], req["root"], now)) is not None:
            raise Refused(why)
        r = sessions_of(lines)[session]["reserve"]
        return {"window": {"root": req["root"]}, "reserve": r["seq"],
                **{k: r["body"][k] for k in RESERVED if k in r["body"]}, "opened_by": req["by"], "pinned": pinned, "key_sha512": key_sha512, "host": r["body"]["host"],
                "port": free_port(lines, active), "secret_sha512": req["secret_sha512"], "started_utc": utc(now)}
    return append(host, "open", session, body_of, I, sign)


def period(host: Host, args: list[str], *, signer, pinned: str, now: datetime, active: Callable[[str], bool]) -> dict:
    """`period-open NAME LANE BY CAP CIRCUIT` or `period-close NAME LANE|- BY` (LANE - for vy-admin's close of any lane's
    period): the signed entry, or Refused. A period's name is used once and a lane has one period open at a time; its
    open commits the most windows it may hold (`cap`) and its circuit; it closes once none of its windows still runs, its
    close listing each window with its outcome."""
    if len(args) != (6 if args[:1] == ["period-open"] else 4):
        raise Refused(f"period: {args}")
    verb, name, lane, by = args[:4]
    if verb == "period-open":
        cap = cap_of(int(args[4])) if re.fullmatch(r"[1-9][0-9]{0,8}", args[4]) else cap_of(args[4])
        circuit = circuit_of(args[5])
    I, sign, key_sha512 = signer

    def body_of(lines: list[bytes]) -> dict:
        periods = periods_of(lines)
        if verb == "period-open":
            if name in periods:
                raise Refused(f"period {name} is in the ledger already: a period's name is used once")
            if (o := next((n for n, p in periods.items() if p["lane"] == lane and p["close"] is None), None)) is not None:
                raise Refused(f"lane {lane} has period {o} open: close it first (periods of a lane never overlap)")
            return {"period": name, "lane": lane, "by": by, "cap": cap, "circuit": circuit, "pinned": pinned,
                    "key_sha512": key_sha512, "opened_utc": utc(now)}
        p = periods.get(name)
        if p is None or p["close"] is not None:
            raise Refused(f"no open period {name}")
        if lane != READ_ONLY and p["lane"] != lane:
            raise Refused(f"period {name} is lane {p['lane']}'s: only its lane's key or vy-admin closes it")
        sessions = sessions_of(lines)
        for w in p["windows"]:
            s = sessions[w]
            if "open" not in s and utc(now) <= s["reserve"]["body"]["expires_utc"]:
                raise Refused(f"period {name}'s window {w} is reserved until {s['reserve']['body']['expires_utc']}: a "
                              "period closes once each of its windows has ended or expired")
            if "open" in s and "end" not in s and active(w):
                raise Refused(f"period {name}'s window {w} still runs: a period closes once each of its windows has "
                              "ended or expired")
        outcomes = window_outcomes(lines, p["windows"])
        return {"period": name, "lane": p["lane"], "by": by, "windows": outcomes,
                "complete": bool(outcomes) and all(w["complete"] for w in outcomes), "closed_utc": utc(now)}

    if verb not in PERIOD_KINDS or not NAME.fullmatch(name) or not (lane == READ_ONLY or NAME.fullmatch(lane)):
        raise Refused(f"period: {args}")
    return append(host, verb, None, body_of, I, sign)


def serve_argv(host: Host, session: str, open_body: dict, tree: Path = TREE) -> list[str]:
    return [sys.executable, "-I", str(tree / "benchmarks" / "live_challenger" / "serve.py"),
            "--out", str(host.sessions / session / "out"), "--port", str(open_body["port"]), "--scheme", open_body["scheme"],
            "--families", json.dumps(open_body["families"]), "--epoch", str(open_body["epoch"]), "--peer", open_body["peer"],
            "--peers", str(host.peers), "--key", str(host.key), "--host", open_body["host"],
            "--root", open_body["window"]["root"], "--secret-file", str(host.sessions / session / SECRET)]


def record_of(out: Path):
    """The challenger record the session signed in `out` (`remote.RECORD`), with its JSON; (None, None) until it has one."""
    from verity.core.protocols.sampled_proofs import challenger as C
    from verity.core.protocols.sampled_proofs import remote
    try:
        doc = json.loads((out / remote.RECORD).read_text())
        return C.ChallengerRecord.from_json(doc), doc
    except (OSError, ValueError, KeyError, TypeError, C.CoinError):
        return None, None


def commitment_of(out: Path) -> str | None:
    """The coin commitment's root (hex) of the record the session signed, None until it has one."""
    record, _ = record_of(out)
    return None if record is None else record.commitment.root.hex()


def await_commitment(host: Host, session: str, active: Callable[[str], bool], seconds: float = 60.0) -> str | None:
    """The session's commitment root once serve.py has signed its record (before it listens), or None if its unit stopped
    first or `seconds` passed."""
    deadline = time.monotonic() + seconds
    while (root := commitment_of(host.sessions / session / "out")) is None and active(session) and time.monotonic() < deadline:
        time.sleep(0.2)
    return root


def released_of(out: Path) -> tuple[dict[str, int] | None, int | None]:
    """(coins released by family, counted from the release log under the commitment the record signs; the record's leaf),
    each None when the session left no such file."""
    from verity.core.protocols.sampled_proofs import challenger as C
    from verity.core.protocols.sampled_proofs import remote
    record, doc = record_of(out)
    if record is None:
        return None, None
    try:
        released = {label: len(r) for label, r in remote.released_roots(out / remote.RELEASES, record.commitment).items()}
    except C.CoinError:
        released = None
    return released, leaf_of(doc["signature"])


def run(host: Host, session: str, *, signer, pinned: str, now: Callable[[], datetime], argv: list[str] | None = None,
        wait: Callable[[subprocess.Popen, float], int] | None = None) -> dict:
    """Serve the admitted session (serve.py, holding `sign.lock` until it listens or exits), then append its `end` entry and
    delete its secret."""
    I, sign, _ = signer
    s = sessions_of(read_ledger(host.ledger)).get(session, {})
    if "open" not in s or "end" in s:
        raise Refused(f"session {session} has no open entry, or has ended: nothing to serve")
    body = s["open"]["body"]
    out = host.sessions / session / "out"
    rc, why = 2, None
    if body["pinned"] != pinned:
        why = f"admitted at {body['pinned']}, but this tree is {pinned}: the pin moved, so the session doesn't run"
    else:
        out.mkdir(mode=0o755, exist_ok=True)
        with open(out / "serve.log", "wb") as log, locked(host.root / "sign.lock"):
            proc = subprocess.Popen(argv or serve_argv(host, session, body), stdout=subprocess.PIPE, stderr=log,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            first = proc.stdout.readline()
        with open(out / "stdout.log", "wb") as f:
            f.write(first)
        rc = (wait or _wait)(proc, body["minutes"] * 60)
        proc.stdout.close()
    (host.sessions / session / SECRET).unlink(missing_ok=True)
    summary = {}
    try:
        summary = json.loads((out / "challenger.json").read_text())
    except (OSError, ValueError):
        pass
    files = {n: hashlib.sha512((out / n).read_bytes()).hexdigest() for n in OUT_FILES if (out / n).is_file()}
    released, record_leaf = released_of(out)
    hellos = (out / "hellos.jsonl").read_text().count("\n") if (out / "hellos.jsonl").is_file() else 0
    complete = released is not None and all(released.get(label) == n for label, n in body["families"])
    end = {"rc": rc, "closed": rc == 0 and summary.get("ended") == "closed by the client",
           "ended": why or summary.get("ended") or f"serve.py exited {rc}", "files": files, "released": released,
           "complete": complete, "record_leaf": record_leaf, "commitment": commitment_of(out), "refused_hellos": hellos,
           "ended_utc": utc(now())}
    return append(host, "end", session, lambda lines: end, I, sign)


def _wait(proc: subprocess.Popen, seconds: float) -> int:
    try:
        return proc.wait(seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        return proc.wait()


# --- main ---------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    host = Host()
    if argv[:1] == ["gate"]:
        rc, out = gate(argv[1:], host)
        sys.stdout.buffer.write(out)
        return rc
    try:
        if argv[:1] == ["reserve"] and len(argv) == 2:
            e = reserve(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=datetime.now(timezone.utc))
            print(json.dumps({"seq": e["seq"], "session": e["session"]}))
        elif argv[:1] == ["admit"] and len(argv) == 2:
            e = admit(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=datetime.now(timezone.utc),
                      active=unit_active)
            print(json.dumps({"seq": e["seq"], "session": e["session"]}))
        elif argv[:1] in (["period-open"], ["period-close"]):
            e = period(host, argv, signer=key_signer(host), pinned=pinned_commit(), now=datetime.now(timezone.utc),
                       active=unit_active)
            print(json.dumps(e))
        elif argv[:1] == ["run"] and len(argv) == 2:
            e = run(host, argv[1], signer=key_signer(host), pinned=pinned_commit(), now=lambda: datetime.now(timezone.utc))
            print(json.dumps({"seq": e["seq"], "session": e["session"], "rc": e["body"]["rc"]}))
        elif argv[:1] == ["keygen"] and len(argv) == 1:
            from verity.ml.crypto import lms
            key, state = load_key(host)
            state.close()
            public = lms.public_key(key)
            doc = {"W": key.params.W, "H": key.params.H, "public": public.hex(),
                   "public_sha512": hashlib.sha512(public).hexdigest()}
            (host.root / "pubkey.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            print(json.dumps(doc))
        else:
            print("usage: auditor.py gate VERB ... | reserve SESSION | admit SESSION | run SESSION | period-open NAME LANE BY CAP CIRCUIT "
                  "| period-close NAME LANE|- BY | keygen", file=sys.stderr)
            return 2
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
