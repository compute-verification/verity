"""ASK-64's per-step round, measured from the serving node against the proof service's challenger off the node: one live
round a step (PERIOD-PATH ask 11).

The challenger is one session on its auditor host from infra's gate (`served_h1/coins.py open`: reserved in the lane's
open development period and opened on the session's stand-in root, `coins.stand_in_root`, never a window's), its layout
a PoUW live window of ROUNDS rounds (`window.LIVE_SCHEME`, epoch 1); the gate gives the session's id, the port its
challenger listens on at this machine's 127.0.0.1, the serving node, the secret its hello carries and the root of the
coin commitment its challenger signed (`coins.Session`, ``--secret-file``, ``--commitment``).  This client is served_h1's
coin source (`coins.open_coins` over `remote.RemoteChallenger`): it refuses another layout or another commitment than
the gate's open answered, opens the salt coin after the session's stand-in root (the challenger opens it after no
other), then round s's coin after step s's stand-in root, sha512(s || coin s-1), so no step's root exists before the
previous step's coin, and times each round from the root's send to its opening checked against the commitment
(`live_reason`, inside `RemoteChallenger.open`); then the draw coin, and the close.  Rounds go back to back unless
--pace-us holds each step, up to the layout's ROUNDS or --max-s seconds, whichever comes first, so one session gives as
many rounds as its timeout allows and still closes; a session that fails is written up to its failure (``ended``).

`run` writes OUT (`percall-rtt.json`): per-round quantiles in ms (p50, p90, p99, max) over every round, over the first
`WINDOW` (one Dither window, `dither_fused.ROUNDS`) and per `WINDOW` block (the challenger's release cost grows with the
round's index), the client's check alone (`live_reason` replayed on every opening), the SHA-512 over the round roots in
order, and every sample (ns); beside it `percall-wait-ns.txt`, one sample a line, which sbench's dither-1s-percall-rtt
arm replays, and `percall-coins.json`, the openings the client received (`COINS`, about 1.9 kB a round).  `check` is the
auditor's side, from the session as the gate publishes it (``ssh GATE fetch SESSION``) and every head of the auditor's
ledger it holds (``ssh GATE ledger``, one ``--ledger`` each): core's check of both takes the session the ledger opens on
the client's stand-in root, under the challenger's key record (``--key-record``, else key.json's own key, said so; no
key is pinned in code; `coins.audited_window`, `remote.trusted_session`); it is the client's session, opened for its
layout, in the verifier's period when it names one (`coins.window_reasons`); the challenger's log holds this client's
commitment, its salt after that root and its round roots in order; and served_h1's `coins.check` passes on the openings
the client received (`coins_check`).  It adds the challenger's own per-round times from that log (integer nanoseconds
on its monotonic clock).

    python3 percall_rtt.py run --challenger 127.0.0.1:PORT --session ID --secret-file F --commitment HEX --rounds 32768 \\
        --max-s 1800 --out out/percall-rtt.json
    python3 percall_rtt.py check --challenger-out SESSION_DIR --ledger LEDGER [--ledger LEDGER ...] \\
        --client ~/.research/runs/CLIENT/out/percall-rtt.json --key-record KEY_RECORD [--period P]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks" / "pouw" / "served_h1")]
import coins as CO

EPOCH = 1
#: one Dither window's rounds (`dither_fused.ROUNDS`)
WINDOW = 1 << 14
PROGRESS = 1024
#: beside OUT: the openings the client received, as served_h1's manifest carries its window's (`coins.check`'s `coins`)
COINS = "percall-coins.json"


def quantiles(ns: list[int]) -> dict:
    """p50, p90, p99 and the max of `ns` (nanoseconds), in ms."""
    if not ns:
        return {"n": 0}
    s = sorted(ns)
    at = lambda q: round(s[min(len(s) - 1, int(q * len(s)))] / 1e6, 4)
    return {"n": len(s), "min_ms": round(s[0] / 1e6, 4), "p50_ms": at(0.5), "p90_ms": at(0.9), "p99_ms": at(0.99),
            "max_ms": round(s[-1] / 1e6, 4), "mean_ms": round(sum(s) / len(s) / 1e6, 4)}


def step_root(s: int, prev: bytes) -> bytes:
    return hashlib.sha512(b"percall-rtt/step" + s.to_bytes(8, "big") + prev).digest()


def draw_root(chain: list[bytes]) -> bytes:
    """The root the draw coin opens after: over the round roots in order."""
    return hashlib.sha512(b"percall-rtt/receipts" + hashlib.sha512(b"".join(chain)).digest()).digest()


def run(a: argparse.Namespace) -> int:
    cs = CO.open_coins(a.challenger, EPOCH, a.rounds)
    stand_in = CO.stand_in_root(a.challenger.id)
    ns: list[int] = []
    chain: list[bytes] = []
    salt = draw = None
    ended, stop = None, (time.monotonic() + a.max_s) if a.max_s else None
    try:
        salt = cs.salt(stand_in)
        prev, pace = salt.coin, a.pace_us / 1e6
        for s in range(a.rounds):
            if stop is not None and time.monotonic() >= stop:
                break
            root = step_root(s, prev)
            t0 = time.perf_counter_ns()
            prev = cs.rounds.record(root).coin
            ns.append(time.perf_counter_ns() - t0)
            chain.append(root)
            if len(ns) % PROGRESS == 0:
                print(f"percall_rtt: {len(ns)} rounds, the last {PROGRESS}: {quantiles(ns[-PROGRESS:])}", flush=True)
            if pace:
                time.sleep(pace)
        draw = cs.draw(draw_root(chain))
        cs.close()
    except CO.Ch.CoinError as e:
        ended = str(e)
    commitment = cs.commitment
    check_ns = []
    for s, o in enumerate(cs.rounds.openings):
        t0 = time.perf_counter_ns()
        why = CO.Ch.live_reason(commitment, o, CO.Wd.ROUND, s, o.after)
        check_ns.append(time.perf_counter_ns() - t0)
        if why is not None:
            raise CO.Ch.CoinError(f"round {s}: {why}")
    blocks = [quantiles(ns[i:i + WINDOW]) for i in range(0, len(ns), WINDOW)]
    doc = {
        "what": "ASK-64 / PERIOD-PATH 11: one live round a step, the serving node to the challenger off the node",
        "client_host": socket.gethostname(), "challenger": a.challenger.address, "session": a.challenger.id,
        "coins": cs.label,
        "commitment": commitment.to_json(), "layout_rounds": a.rounds, "rounds": len(ns), "max_s": a.max_s,
        "pace_us": a.pace_us, "ended": ended or "closed by the client",
        "round_trip": quantiles(ns), "first_window": quantiles(ns[:WINDOW]), "window": WINDOW,
        "by_window_p50_ms": [b["p50_ms"] for b in blocks], "by_window": blocks,
        "client_check": quantiles(check_ns), "record_reason": cs.rounds.reason() if ns else None,
        "opening_bytes": len(json.dumps(cs.rounds.openings[0].to_json())) if ns else None,
        "root": stand_in.hex(), "root_is": CO.STAND_IN, "salt_after": salt.after.hex() if salt else None,
        "draw_after": draw.after.hex() if draw else None, "roots_sha512": hashlib.sha512(b"".join(chain)).hexdigest(),
        "samples_ns": ns,
    }
    received = {"session": a.challenger.id, "commitment": commitment.to_json(), "window": {"root": stand_in.hex()},
                "salt_coin": salt.to_json() if salt else None,
                "rounds": [{"opening": o.to_json()} for o in cs.rounds.openings],
                "draw_coin": draw.to_json() if draw else None}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc) + "\n")
    (out.parent / "percall-wait-ns.txt").write_text("\n".join(map(str, ns)) + "\n")
    (out.parent / COINS).write_text(json.dumps(received) + "\n")
    print(json.dumps({k: doc[k] for k in ("coins", "rounds", "ended", "round_trip", "first_window", "by_window_p50_ms",
                                          "client_check", "record_reason")}, indent=1), flush=True)
    return 0 if ended is None and doc["record_reason"] is None else 1


def coins_check(a: argparse.Namespace, stand_in: bytes) -> dict:
    """served_h1's `coins.check` of the openings the client received (`COINS` beside ``--client``, or ``--coins``)
    against the published session and the gate's ledger, on the session's stand-in root: round s's root recomputed from
    them (`step_root` over the salt's coin, then round s - 1's) and the draw's root over those (`draw_root`)."""
    path = Path(a.coins) if a.coins else Path(a.client).with_name(COINS)
    try:
        received = json.loads(path.read_text())
        prev = CO.Ch.LiveOpening.from_json(received["salt_coin"]).coin
        chain = []
        for s, r in enumerate(received["rounds"]):
            chain.append(step_root(s, prev))
            prev = CO.Ch.LiveOpening.from_json(r["opening"]).coin
    except (OSError, ValueError, KeyError, TypeError, CO.Ch.CoinError) as e:
        return {"ok": False, "reason": f"the client's openings ({path}) are unreadable: {type(e).__name__}: {e}"}
    got = CO.check(a.challenger_out, received, a.ledger, chain, None, draw_root(chain) if received.get("draw_coin") else None,
                   key=CO.key_record(a.key_record) if a.key_record else None, period=a.period, root=stand_in)
    return {k: got[k] for k in ("ok", "coins", "reason", "reasons", "ledger", "rounds", "draws") if k in got}


def check(a: argparse.Namespace) -> int:
    d, client = CO.published(a.challenger_out), json.loads(Path(a.client).read_text())
    theirs = CO.Ch.ChallengerKey.from_json(json.loads((d / CO.R.KEY).read_text()))
    key = CO.key_record(a.key_record) if a.key_record else theirs
    host = CO.Ch.ChallengerRecord.from_json(json.loads((d / CO.R.RECORD).read_text())).host
    sid = client["session"]
    root = CO.stand_in_root(sid)
    reasons: dict[str, str | None] = {
        "key": None if key == theirs else "the session's key.json is not the challenger's key record"}
    if reasons["key"] is not None:
        print(json.dumps({"ok": False, "host": host, "reasons": reasons}, indent=1), flush=True)
        return 1
    try:
        commitment, roots, audited, entries = CO.audited_window(d, a.ledger, root, key)
    except CO.Ch.CoinError as e:
        print(json.dumps({"ok": False, "host": host, "reasons": {**reasons, "ledger": str(e)}}, indent=1), flush=True)
        return 1
    reasons.update(CO.window_reasons(audited, entries, commitment, sid, {"root": root.hex()}, root, a.period))
    ledger = CO.ledger_info(audited, entries)
    n = client["rounds"]
    got = hashlib.sha512(b"".join(roots[CO.Wd.ROUND][:n])).hexdigest()
    reasons["commitment"] = (None if commitment.to_json() == client["commitment"]
                             else "the challenger signed another commitment than the client's")
    reasons["roots"] = (None if got == client["roots_sha512"] and len(roots[CO.Wd.ROUND]) >= n
                        else "the challenger's round roots are not the client's")
    reasons["salt"] = (None if roots[CO.Wd.SALT] == (root,) and client.get("salt_after") == root.hex()
                       else "the salt coin didn't open after the session's stand-in root, here and at the challenger")
    reasons["draw"] = (None if roots[CO.Wd.DRAW] == ((bytes.fromhex(client["draw_after"]),) if client["draw_after"] else ())
                       else "the draw root differs")
    coins = coins_check(a, root)
    reasons["coins"] = None if coins.get("ok") else "coins.check refused the client's openings: " + (
        coins.get("reason") or ", ".join(k for k, v in (coins.get("reasons") or {}).items() if v))
    events = [json.loads(x) for x in (d / CO.R.RELEASES).read_text().splitlines()[1:]]
    rounds = [(r, e) for r, e in zip(events[::2], events[1::2]) if r["label"] == CO.Wd.ROUND]
    serve_ns = [e["released_at"] - r["received_at"] for r, e in rounds]
    gap_ns = [b["received_at"] - x["received_at"] for (x, _), (b, _) in itertools.pairwise(rounds)]
    doc = {"ok": all(v is None for v in reasons.values()), "host": host, "key_sha512": hashlib.sha512(key.public).hexdigest(),
           "key": "the key record's" if a.key_record else "key.json's own (unanchored)", "session": sid,
           "root": root.hex(), "root_is": CO.STAND_IN, "ledger": ledger, "period_named": a.period,
           "reasons": reasons, "coins_check": coins, "rounds_logged": len(roots[CO.Wd.ROUND]),
           "challenger_open": quantiles(serve_ns), "challenger_period": quantiles(gap_ns),
           "challenger_period_by_window_p50_ms": [quantiles(gap_ns[i:i + WINDOW])["p50_ms"]
                                                  for i in range(0, len(gap_ns), WINDOW)]}
    print(json.dumps(doc, indent=1), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--challenger", default=None, metavar="HOST:PORT", help=f"default ${CO.CHALLENGER_ENV}")
    r.add_argument("--session", default=None, metavar="ID", help=f"the session's id from infra (default ${CO.SESSION_ENV})")
    r.add_argument("--secret-file", default=None, metavar="F",
                   help=f"the session's secret, which the hello carries (default ${CO.SECRET_ENV})")
    r.add_argument("--commitment", default=None, metavar="HEX",
                   help=f"the coin commitment's root the gate's open answered, the hello's (default ${CO.COMMITMENT_ENV})")
    r.add_argument("--rounds", type=int, default=WINDOW, help="the session's layout: at most this many rounds")
    r.add_argument("--max-s", type=float, default=0.0,
                   help="stop opening rounds after this many seconds, then draw and close (inside the session's timeout)")
    r.add_argument("--pace-us", type=float, default=0.0)
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--challenger-out", required=True,
                   help="the session as the gate publishes it (out/ beside request.json and open.json), or its out/")
    c.add_argument("--ledger", action="append", required=True,
                   help="a head of the auditor's ledger as infra published it (ssh GATE ledger); once for each head")
    c.add_argument("--client", required=True)
    c.add_argument("--coins", default=None, help=f"the client's received openings (default {COINS} beside --client)")
    c.add_argument("--key-record", default=None, help="the challenger's key record (challenger-key/v1)")
    c.add_argument("--period", default=None, help="the development period the session's reserve is a window of")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        a.challenger = CO.session(a.challenger, a.session, a.secret_file, a.commitment)
        if a.challenger is None:
            ap.error("run measures the challenger off the node: --challenger HOST:PORT and --session ID (with "
                     f"--secret-file), or ${CO.CHALLENGER_ENV} and ${CO.SESSION_ENV} (with ${CO.SECRET_ENV})")
        return run(a)
    return check(a)


if __name__ == "__main__":
    raise SystemExit(main())
