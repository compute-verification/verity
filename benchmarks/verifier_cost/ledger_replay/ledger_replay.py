"""`ledger_replay.sh`'s inputs and its table.

  python ledger_replay.py DIR S... [--period N] [--lms W,H]
  python ledger_replay.py --table OUT

Per S, DIR/ledger-S.json is the `challenger_ledger` piece's request, ``{"key", "ledger", "window"}``: an auditor's
ledger of S sessions as the challenger's writer signs one (`challenger.ledger_line`), under a key at the deployed LMS
parameters (`lms.DEPLOYED`, or `--lms`), and the window its last session opens. DIR/expect-S.json is what the piece's
reference answers (`challenger.audited_session`, or ``{"refused": why}``). One line of JSON a size on stdout: the
ledger's sessions, entries, periods and declarations, its bytes and longest line, and the reference's seconds on it,
one replay (`python_s`), with the signatures it verified (`verifies`).

The ledger is one lane's, two-step, in the bodies the auditor's challenger service writes: each session a reserve, an
open binding it and an end naming its record's leaf, one after another, half an hour apart; a declaration before each
UTC day's first session; and the sessions in periods of `--period` sessions (by default one period holding them all),
each a period-open before its first reserve and a period-close after its last end. A period's close lists each of its
sessions with its root, so one period's sessions are bounded by a ledger line's most characters
(`challenger.MAX_LEDGER_LINE`): past it the reference refuses the ledger and the row says so.

The key signs at the ledger's leaves alone: its tree holds the real one-time keys there and, in place of each subtree's
root beside them, a digest standing in for keys never made, which a verifier can't tell from a whole key (a whole key
at the deployed parameters is 2^20 one-time keys, made before it signs once).

`--table OUT` reads OUT's inputs.jsonl and ledger_replay.jsonl (`ledger_replay.sh`) and writes one row a size on stdout:
the compiled replay's seconds, the reference's, and how the compiled replay's time splits: each entry's reading, its
line's SHA-512, its message, its LMS verification, and the rest (`rest_s`), which holds the replay's searches and each
entry's checks of its body. The parts are made apart from the replay, so they need not sum to it, and the rest can be
below zero. `searches_s` is the searches made apart from the replay (`Main.lean`'s `searches`), `share` each part's
share of the replay, and `reading_s` the parts of an entry's reading (`entries_s`). With three sizes or more it fits
the rest to ``c + a S + b S²`` by least squares and gives ``b S²``'s share of the replay at each size: a fit, not a
measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from verity.commitments.crypto import lms
from verity.protocols.proofs.spot.device import challenger as Ch

LANE = "lane-a"
HOST = "auditor-1"
PINNED = "5e" * 20
SCHEME = "verity/pouw/epoch-coins/live/v1"
FAMILIES = [["pouw/salt", 1], ["pouw/draw", 8]]
MINUTES = 30
START = datetime(2026, 1, 5, tzinfo=UTC)
#: a session's entries (reserve, open, end) and the leaves it takes: theirs and its record's, between its open and end
SESSION_ENTRIES, SESSION_LEAVES = 3, 4
PHASES = ("entries", "chain", "messages", "signatures")
#: the parts of an entry's reading that Main.lean times apart
READING = ("parse", "canonical", "canon", "record")


def _utc(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(what: str) -> str:
    return hashlib.sha512(what.encode()).hexdigest()


def periods_of(sessions: int, period: int) -> int:
    return -(-sessions // period)


def days_of(sessions: int) -> int:
    return -(-sessions * MINUTES // (24 * 60))


def entries_of(sessions: int, period: int) -> int:
    """The entries of a ledger of `sessions` sessions in periods of `period`."""
    return SESSION_ENTRIES * sessions + 2 * periods_of(sessions, period) + days_of(sessions)


def leaves_of(sessions: int, period: int) -> int:
    """The leaves such a ledger's key signs at, its sessions' records' leaves among them."""
    return SESSION_LEAVES * sessions + 2 * periods_of(sessions, period) + days_of(sessions)


def key_of(params: lms.Params, leaves: int, seed: bytes = bytes(64)) -> lms.PrivateKey:
    """The key of `seed` at `params`, signing at its first `leaves` leaves (the module docstring's stand-in tree)."""
    top = 2 ** params.H
    if leaves > top:
        raise SystemExit(f"a key at H = {params.H} has {top} leaves, and the ledger takes {leaves}")
    if leaves == top:
        return lms.keygen(params, seed)
    I = hashlib.sha512(b"verity/lms/v0/I" + seed).digest()[:16]
    tree = [b""] * (2 * top)
    tree[top:top + leaves] = lms._leaves(params, I, seed, 0, leaves)
    lo, hi = top, top + leaves
    while lo > 1:
        if hi % 2:
            tree[hi] = hashlib.sha512(b"stand-in" + I + hi.to_bytes(4, "big")).digest()
            hi += 1
        for r in range(lo // 2, hi // 2):
            tree[r] = lms._interior(params, I, r, tree[2 * r], tree[2 * r + 1])
        lo, hi = lo // 2, hi // 2
    return lms.PrivateKey(params, I, seed, tuple(tree))


def ledger(sessions: int, period: int, key: lms.PrivateKey) -> tuple[list[str], str]:
    """The ledger's lines and its last session's root."""
    public = lms.public_key(key)
    I, key_sha512 = public[:16], hashlib.sha512(public).hexdigest()
    pins = Ch.pins_seal([Ch.pin_line(PINNED, PINNED, _utc(START - timedelta(days=1)))])
    lines: list[str] = []
    leaf = 0

    def append(kind: str, session: str | None, body: dict) -> None:
        nonlocal leaf
        lines.append(Ch.ledger_line(lines, kind, session, body, I, lambda m: lms.sign_at(key, leaf, m).encode()))
        leaf += 1

    outcomes: list[dict] = []
    name, root = "", ""
    for i in range(sessions):
        at = START + timedelta(minutes=MINUTES * i)
        if i % (24 * 60 // MINUTES) == 0:
            append(Ch.DECLARATION, None, {"by": LANE, "blocks": ["b0"], "day": Ch.day_of(_utc(at)), "declared_utc": _utc(at),
                                          "pinned": PINNED, "program": _digest("program")})
        if i % period == 0:
            name = f"p-{i // period:06d}"
            append("period-open", None, {"period": name, "lane": LANE, "by": LANE, "cap": min(period, sessions - i),
                                         "circuit": _digest("circuit"), "pinned": PINNED, "key_sha512": key_sha512,
                                         "opened_utc": _utc(at)})
        session, root = f"s{at:%Y%m%dT%H%M%S}Z-{i % 65536:04x}", _digest(f"window {i}")
        reserve = {"by": LANE, "peer": "10.0.0.7", "scheme": SCHEME, "epoch": i, "families": FAMILIES, "minutes": MINUTES,
                   "pinned": PINNED, "key_sha512": key_sha512, "host": f"{HOST}.{session}", "started_utc": _utc(at),
                   "expires_utc": Ch.later(_utc(at), MINUTES)}
        append("reserve", session, reserve)
        append("open", session, {"window": {"root": root}, "reserve": len(lines) - 1,
                                 **{k: reserve[k] for k in Ch.RESERVED if k in reserve}, "opened_by": LANE, "pinned": PINNED,
                                 "pins": pins, "key_sha512": key_sha512, "host": reserve["host"], "port": 7000,
                                 "secret_sha512": _digest(f"secret {i}"), "started_utc": _utc(at + timedelta(minutes=1))})
        leaf += 1               # the session's record, signed between its open and its end
        append("end", session, {"rc": 0, "closed": True, "ended": "closed by the client",
                                "files": {f: _digest(f"{f} {i}") for f in Ch.SEALED_FILES},
                                "released": {label: n for label, n in FAMILIES}, "complete": True, "record_leaf": leaf - 1,
                                "commitment": _digest(f"commitment {i}"), "refused_hellos": 0,
                                "ended_utc": _utc(at + timedelta(minutes=MINUTES - 5))})
        outcomes.append({"root": root, "session": session, "complete": True})
        if (i + 1) % period == 0 or i + 1 == sessions:
            append("period-close", None, {"period": name, "lane": LANE, "by": LANE, "windows": outcomes, "complete": True,
                                          "closed_utc": _utc(at + timedelta(minutes=MINUTES - 1))})
            outcomes = []
    return lines, root


def write(out: Path, sessions: int, period: int, key: lms.PrivateKey) -> dict:
    """DIR/ledger-S.json and DIR/expect-S.json, and the size's row."""
    params = key.params
    lines, root = ledger(sessions, period, key)
    k = Ch.ChallengerKey(params.W, params.H, lms.public_key(key))
    request = {"key": k.to_json(), "ledger": lines, "window": {"root": root}}
    verifies = 0

    def verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
        nonlocal verifies
        verifies += 1
        return lms.verify(params, key.public, message, lms.Signature.decode(params, signature))

    t0 = time.perf_counter()
    try:
        answer = Ch.audited_session(lines, request["window"], k, verify)._asdict()
    except Ch.CoinError as e:
        answer = {"refused": str(e)}
    seconds = time.perf_counter() - t0
    text = json.dumps(request, sort_keys=True, separators=(",", ":"))
    (out / f"ledger-{sessions}.json").write_text(text)
    (out / f"expect-{sessions}.json").write_text(json.dumps(answer, sort_keys=True, separators=(",", ":")))
    return {"sessions": sessions, "entries": len(lines), "periods": periods_of(sessions, period),
            "declarations": days_of(sessions), "lms": [params.W, params.H], "bytes": len(text),
            "longest_line": max(map(len, lines)), "line_cap": Ch.MAX_LEDGER_LINE, "python_s": round(seconds, 6),
            "verifies": verifies, "refused": answer.get("refused")}


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares' (c, a, b) of ``y = c + a x + b x²``, by Cramer's rule on the normal equations."""
    s = [sum(x ** k for x in xs) for k in range(5)]
    t = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]

    def det(m: list[list[float]]) -> float:
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    rows = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]
    d = det(rows)
    return tuple(det([[t[i] if j == k else rows[i][j] for j in range(3)] for i in range(3)]) / d for k in range(3))


def table(out: Path) -> list[dict]:
    """One row a size the verifier accepted, from OUT's inputs.jsonl and ledger_replay.jsonl."""
    inputs = {r["sessions"]: r for r in map(json.loads, (out / "inputs.jsonl").read_text().splitlines())}
    phases: dict[int, dict[str, dict]] = {}
    for r in map(json.loads, (out / "ledger_replay.jsonl").read_text().splitlines()):
        phases.setdefault(r["sessions"], {})[r["phase"]] = r
    rows = []
    for n in sorted(phases):
        p, i = phases[n], inputs[n]
        if not p["checked"]["ok"]:
            rows.append({"sessions": n, "entries": i["entries"], "longest_line": i["longest_line"], "refused": i["refused"]})
            continue
        split = {k: p[k]["us"] / 1e6 for k in PHASES}
        replay = p["checked"]["us"] / 1e6
        split["rest"] = replay - sum(split.values())
        split["searches"] = p["searches"]["us"] / 1e6
        reading = {k: p[k]["us"] / 1e6 for k in READING}
        reading["other"] = split["entries"] - sum(reading.values())
        rows.append({"sessions": n, "entries": i["entries"], "periods": i["periods"], "bytes": i["bytes"],
                     "longest_line": i["longest_line"], "replay_s": replay, "piece_s": p["piece"]["us"] / 1e6,
                     "python_s": i["python_s"], **{f"{k}_s": round(v, 6) for k, v in split.items()},
                     "share": {k: round(v / replay, 6) for k, v in split.items()},
                     "reading_s": {k: round(v, 6) for k, v in reading.items()}, "hwm_mb": p["checked"]["hwm_mb"]})
    fitted = [r for r in rows if "rest_s" in r]
    if len(fitted) >= 3:
        c, a, b = _fit([float(r["sessions"]) for r in fitted], [r["rest_s"] for r in fitted])
        for r in fitted:
            square = b * r["sessions"] ** 2
            r["fit"] = {"c": c, "a": a, "b": b, "square_s": round(square, 6), "square_share": round(square / r["replay_s"], 6)}
    return rows


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("sessions", type=int, nargs="*")
    ap.add_argument("--period", type=int, default=0, help="a period's sessions (default: one period holds them all)")
    ap.add_argument("--lms", default=f"{lms.DEPLOYED.W},{lms.DEPLOYED.H}", help="the key's W,H (default: the deployed)")
    ap.add_argument("--table", type=Path, help="a run's $OUT: print its table and write nothing")
    args = ap.parse_args(argv)
    if args.table is not None:
        for row in table(args.table):
            print(json.dumps(row))
        return
    if args.out is None or not args.sessions or min(args.sessions) < 1 or args.period < 0:
        ap.error("DIR and the sessions of each ledger, one or more each")
    w, h = map(int, args.lms.split(","))
    params = lms.Params(64, w, h)
    args.out.mkdir(parents=True, exist_ok=True)
    period = {n: args.period or n for n in args.sessions}
    key = key_of(params, max(leaves_of(n, period[n]) for n in args.sessions))
    for n in args.sessions:
        print(json.dumps(write(args.out, n, period[n], key)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
