"""ASK-64's per-step round, measured from the serving node against the proof service's challenger off the node: one live
round a step (PERIOD-PATH ask 11).

The challenger is one session on its auditor host from infra's gate (`served_h1/coins.py open`: reserved in the lane's
open development period and opened on the session's stand-in root, `coins.stand_in_root`, never a window's), its layout
a PoUW live window of ROUNDS rounds (`window.LIVE_SCHEME`, epoch 1); the gate gives the session's id, the port its
challenger listens on at this machine's 127.0.0.1, the serving node, and the secret its hello carries (`coins.Session`,
``--secret-file``).  This client is served_h1's coin source (`coins.open_coins` over `remote.RemoteChallenger`): it
refuses another layout, opens the salt coin after the session's stand-in root (the challenger opens it after no
other), then round s's coin after step s's stand-in root, sha512(s || coin s-1), so no step's root exists before the
previous step's coin, and times each round from the root's send to its opening checked against the commitment
(`live_reason`, inside `RemoteChallenger.open`); then the draw coin, and the close.  Rounds go back to back unless
--pace-us holds each step, up to the layout's ROUNDS or --max-s seconds, whichever comes first, so one session gives as
many rounds as its timeout allows and still closes; a session that fails is written up to its failure (``ended``).

`run` writes OUT (`percall-rtt.json`): per-round quantiles in ms (p50, p90, p99, max) over every round, over the first
`WINDOW` (one Dither window, `dither_fused.ROUNDS`) and per `WINDOW` block (the challenger's release cost grows with the
round's index), the client's check alone (`live_reason` replayed on every opening), the SHA-512 over the round roots in
order, and every sample (ns); beside it `percall-wait-ns.txt`, one sample a line, which sbench's dither-1s-percall-rtt
arm replays.  `check` is the auditor's side, from the session as the gate publishes it (``ssh GATE fetch SESSION``) and
the gate's ledger (``ssh GATE ledger``): the record verifies under the challenger's key record (``--key-record``, else
key.json's own key, said so; no key is pinned in code; `remote.trusted_session`); the published session and its ledger
entries are the client's session, its open binding its reserve, on its stand-in root (`coins.gate_reasons`), in the
verifier's period when it names one; and the challenger's log holds this client's commitment, its salt after that
root and its round roots in order; it adds the challenger's own per-round times from that log (its monotonic clock).

    python3 percall_rtt.py run --challenger 127.0.0.1:PORT --session ID --secret-file F --rounds 32768 --max-s 1800 \\
        --out out/percall-rtt.json
    python3 percall_rtt.py check --challenger-out SESSION_DIR --ledger LEDGER \\
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


def run(a: argparse.Namespace) -> int:
    cs = CO.open_coins(a.challenger, EPOCH, a.rounds)
    root = CO.stand_in_root(a.challenger.id)
    ns: list[int] = []
    roots = hashlib.sha512()
    salt = draw = None
    ended, stop = None, (time.monotonic() + a.max_s) if a.max_s else None
    try:
        salt = cs.salt(root)
        prev, pace = salt.coin, a.pace_us / 1e6
        for s in range(a.rounds):
            if stop is not None and time.monotonic() >= stop:
                break
            root = step_root(s, prev)
            t0 = time.perf_counter_ns()
            prev = cs.rounds.record(root).coin
            ns.append(time.perf_counter_ns() - t0)
            roots.update(root)
            if len(ns) % PROGRESS == 0:
                print(f"percall_rtt: {len(ns)} rounds, the last {PROGRESS}: {quantiles(ns[-PROGRESS:])}", flush=True)
            if pace:
                time.sleep(pace)
        draw = cs.draw(hashlib.sha512(b"percall-rtt/receipts" + roots.digest()).digest())
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
        "root": root.hex(), "root_is": CO.STAND_IN, "salt_after": salt.after.hex() if salt else None,
        "draw_after": draw.after.hex() if draw else None, "roots_sha512": roots.hexdigest(), "samples_ns": ns,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc) + "\n")
    (out.parent / "percall-wait-ns.txt").write_text("\n".join(map(str, ns)) + "\n")
    print(json.dumps({k: doc[k] for k in ("coins", "rounds", "ended", "round_trip", "first_window", "by_window_p50_ms",
                                          "client_check", "record_reason")}, indent=1), flush=True)
    return 0 if ended is None and doc["record_reason"] is None else 1


def check(a: argparse.Namespace) -> int:
    (d, _, _), client = CO.published(a.challenger_out), json.loads(Path(a.client).read_text())
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
    commitment, roots = CO.R.trusted_session(d, key, CO.lms_verify)
    gate, ledger = CO.gate_reasons(a.challenger_out, a.ledger, sid, root, commitment, a.period)
    reasons.update(gate)
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
    events = [json.loads(x) for x in (d / CO.R.RELEASES).read_text().splitlines()[1:]]
    rounds = [(r, e) for r, e in zip(events[::2], events[1::2]) if r["label"] == CO.Wd.ROUND]
    serve_ns = [round((e["released_at"] - r["received_at"]) * 1e9) for r, e in rounds]
    gap_ns = [round((b["received_at"] - x["received_at"]) * 1e9) for (x, _), (b, _) in itertools.pairwise(rounds)]
    doc = {"ok": all(v is None for v in reasons.values()), "host": host, "key_sha512": hashlib.sha512(key.public).hexdigest(),
           "key": "the key record's" if a.key_record else "key.json's own (unanchored)", "session": sid,
           "root": root.hex(), "root_is": CO.STAND_IN, "ledger": ledger, "period_named": a.period,
           "reasons": reasons, "rounds_logged": len(roots[CO.Wd.ROUND]),
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
    r.add_argument("--rounds", type=int, default=WINDOW, help="the session's layout: at most this many rounds")
    r.add_argument("--max-s", type=float, default=0.0,
                   help="stop opening rounds after this many seconds, then draw and close (inside the session's timeout)")
    r.add_argument("--pace-us", type=float, default=0.0)
    r.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--challenger-out", required=True,
                   help="the session as the gate publishes it (request.json, open.json, out/)")
    c.add_argument("--ledger", required=True, help="the gate's ledger (ssh GATE ledger)")
    c.add_argument("--client", required=True)
    c.add_argument("--key-record", default=None, help="the challenger's key record (challenger-key/v1)")
    c.add_argument("--period", default=None, help="the development period the session's reserve is a window of")
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        a.challenger = CO.session(a.challenger, a.session, a.secret_file)
        if a.challenger is None:
            ap.error("run measures the challenger off the node: --challenger HOST:PORT and --session ID (with "
                     f"--secret-file), or ${CO.CHALLENGER_ENV} and ${CO.SESSION_ENV} (with ${CO.SECRET_ENV})")
        return run(a)
    return check(a)


if __name__ == "__main__":
    raise SystemExit(main())
