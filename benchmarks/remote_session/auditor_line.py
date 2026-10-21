"""The memory line on the auditor's clock, the analysis of record for a gate run: each accepted paced session's line
(`verifier.paced_line`, time-held and snapshot, at each ingress charge) with its groups' times read from what the
auditor's host signed rather than from the verifier's stamps.

Group g's coin σ_g left the challenger at its release log's `released_at` for challenge coin g, and window g's last
root reached it at the first `received_at` of a request naming that root (`auditor_seen`). Openings coin g's request
names it (`audit.Session.close` asks for the coin after the window's last root), and a paced window is drawn at its
k-th root, so that request leaves as the root arrives; for k = 1, σ_(g+2)'s request names the same root too. Both
times are the challenger's monotonic clock in its release log, whose lines its close signs (`challenger.session_of`),
and whose close, record and refused hellos the auditor's ledger sealed at the session's end (`remote.trusted_session`).
So L_g = max(A_g, s_(g+1)) − s_g on the challenger's clock holds group g's reads: they come after its coin, whose seed
they take, and before its last root, which chains them.

Beside the times the line reads the verifier's record of the session (`verifier.json`): its verdict, each group's
passes and held bytes (`ledger.memory`) and its pooled wrong receipts. The verifier runs on the audit's machine, not
the auditor's host, so these rest on its verdict, which the session's transcript (`verifier.py --transcript`) lets the
auditor replay offline (the remote audit's `lean/tests/replay.py transcript`). The verifier's stamps are read only to
cross-check the challenger's: each coin interval on both clocks, their rates, and each group's L on both.

The ledger check: its longest head verified under the pinned key (`ledger_head`, `verify_ledger`, `verify_periods`);
the period's lane, cap and circuit, and that it closed; each session reserved and opened by the lane for the peer under
the pin at epoch k, opened at its window root R_k, ended closed with no refused hello, inside the period; σ_0 asked
after R_k, and each coin released after its root's receipt. A session served by `--challenger serve` (the verifier's
machine) has no ledger: its log is read under the key its run wrote (`session_of`) and its line is not the auditor's.

    python auditor_line.py OUT --host-bytes B [--pin SHA512 --period P] [--ledger HEAD ...] [--ingress B/S ...]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import challengers as CH  # noqa: E402
import verifier as V  # noqa: E402
from verity.core.protocols.remote_bandwidth import coins  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as Ch  # noqa: E402
from verity.core.protocols.sampled_proofs import remote  # noqa: E402

INGRESS = (4.4e9, 54.4e9)
"""The ingress charges a gate run's line is read at: the node's NICs into memory (time-held) and every link into it
(link-charged), bytes a second."""
LANE, PEER = "memory-accounting", "vy-nebius-1"


def release_log(lines: list[str]) -> dict[str, dict[str, list]]:
    """A release log's lines after its head, by family: each coin's root (`after`), its request's receipt and the
    coin's release, on the challenger's clock, in index order (`challenger.released_roots_of` checks the form)."""
    fam: dict[str, dict[str, list]] = {}
    for i in range(1, len(lines) - 1, 2):
        root, release = json.loads(lines[i]), json.loads(lines[i + 1])
        f = fam.setdefault(root["label"], {"after": [], "received": [], "released": []})
        f["after"].append(root["root"])
        f["received"].append(root["received_at"])
        f["released"].append(release["released_at"])
    return fam


def auditor_seen(lines: list[str], seen: list[dict]) -> tuple[list[dict], dict]:
    """`seen` (a session's `windows_seen`) with each group's coin and last root on the challenger's clock: `released_ns`
    challenge coin g's release, `arrivals_ns` the first receipt of a request naming openings coin g's root, the
    window's last; and which family's request named each last root first."""
    fam = release_log(lines)
    ch, op = fam.get("challenge"), fam.get("openings")
    n = len(seen)
    if ch is None or op is None or len(ch["released"]) < n or len(op["after"]) < n:
        raise ValueError(f"the release log holds {len((ch or {}).get('released', ()))} challenge and "
                         f"{len((op or {}).get('after', ()))} openings coins, not one of each for each of {n} groups")
    first: dict[str, tuple[int, str]] = {}
    for label, f in fam.items():
        for root, t in zip(f["after"], f["received"]):
            if root not in first or t < first[root][0]:
                first[root] = (t, label)
    out, by = [], {}
    for g, w in enumerate(seen):
        t, label = first[op["after"][g]]
        by[label] = by.get(label, 0) + 1
        out.append({**w, "released_ns": ch["released"][g], "arrivals_ns": [t] if w["arrivals_ns"] else []})
    return out, {"last_root_first_named_by": by}


def q(xs: list[float], scale: float = 1e6) -> dict | None:
    xs = sorted(x / scale for x in xs)
    if not xs:
        return None
    at = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"n": len(xs), "min": round(xs[0], 3), "median": round(at(0.5), 3), "p90": round(at(0.9), 3),
            "p99": round(at(0.99), 3), "max": round(xs[-1], 3)}


def group_l(seen: list[dict]) -> list[int]:
    """Each group's L but the last's (`paced_line`'s L_g)."""
    s, n = [w["released_ns"] for w in seen], len(seen)
    return [max(seen[g]["arrivals_ns"][-1], s[g + 1]) - s[g] for g in range(n - 1)]


def shares(session: dict, registration: dict, host_bytes: int, ingress: tuple[float, ...]) -> dict:
    row = {}
    for b in ingress:
        p = V.recorded_paced(session, registration, host_bytes=host_bytes, ingress_bytes_per_s=b)
        worst = p["stretches"][p["time"]["stretch"]]
        row[f"{b:.1e}"] = {"time": p["time"]["share"], "snapshot": p["snapshot"]["share"],
                           "worst_stretch": p["time"]["stretch"], "worst_l_max_ms": worst["l_max_ns"] / 1e6,
                           "worst_span_s": worst["span_ns"] / 1e9}
    return row


def cross_check(verifier_seen: list[dict], auditor: list[dict]) -> dict:
    """The challenger's clock held to the verifier's stamps: coin intervals, rate and each group's L on both."""
    Rv, Ra = [w["released_ns"] for w in verifier_seen], [w["released_ns"] for w in auditor]
    n = len(Rv)
    Lv, La = group_l(verifier_seen), group_l(auditor)
    return {"coin_interval_auditor_minus_verifier_ms": q([(Ra[g + 1] - Ra[g]) - (Rv[g + 1] - Rv[g])
                                                          for g in range(n - 1)]),
            "rate_auditor_over_verifier_minus_1_ppm": ((Ra[-1] - Ra[0]) / (Rv[-1] - Rv[0]) - 1) * 1e6
            if n > 1 and Rv[-1] != Rv[0] else None,
            "L_auditor_ms": q(La), "L_verifier_ms": q(Lv), "L_auditor_minus_verifier_ms": q(
                [a - v for a, v in zip(La, Lv)])}


def log_checks(lines: list[str], window_root: str) -> dict:
    """What the release log shows about the order of coins and roots: each coin released after its request's receipt,
    the clock never running back, and σ_0 asked after the session's window root."""
    fam = release_log(lines)
    times = [t for i in range(1, len(lines) - 1, 2)
             for t in (json.loads(lines[i])["received_at"], json.loads(lines[i + 1])["released_at"])]
    return {"coins_released": {label: len(f["released"]) for label, f in fam.items()},
            "coin_before_its_root": [(label, i) for label, f in fam.items()
                                     for i, (r, s) in enumerate(zip(f["received"], f["released"])) if s < r],
            "clock_monotone": all(a <= b for a, b in zip(times, times[1:])),
            "sigma0_after_window_root": bool(fam.get("challenge", {}).get("after"))
            and fam["challenge"]["after"][0] == window_root}


def ledger_check(heads: list[Path], key: Ch.ChallengerKey, period: str, lane: str) -> tuple[dict, dict, dict]:
    """The auditor's ledger: its longest head under `key`, verified, with its windows by root and its periods."""
    lines = Ch.ledger_head([remote.ledger_lines(p) for p in heads], key, CH.lms_verify)
    raw = [line.encode() for line in lines]
    verify = lambda m, sig: CH.lms_verify(key, m, sig)  # noqa: E731
    windows = Ch.verify_ledger(raw, key.public[:16], verify)
    periods = Ch.verify_periods(raw, key.public[:16], verify)
    p = periods.get(period)
    doc = {"entries": len(lines), "sha512": hashlib.sha512("".join(x + "\n" for x in lines).encode()).hexdigest(),
           "verify_ledger": "passed", "heads": len(heads)}
    doc["period"] = None if p is None else {
        "name": period, "lane": p["lane"], "cap": p["cap"], "circuit": p["circuit"], "windows": len(p["windows"]),
        "opened_by": p["open"]["body"].get("by") if p.get("open") else None,
        "opened_utc": p["open"]["body"].get("opened_utc") if p.get("open") else None,
        "closed": p.get("close") is not None,
        "closed_utc": p["close"]["body"].get("closed_utc") if p.get("close") else None,
        "ok": p["lane"] == lane and p.get("close") is not None}
    return windows, periods, doc


def session_check(s: dict, k: int, R: str, windows: dict, periods: dict, *, period: str, lane: str, peer: str,
                  pin: str) -> dict:
    """Session k's entries on the ledger: its reserve, open and end, held to the lane, the peer, the pin, its epoch and
    the period."""
    sid = (s.get("challenger") or {}).get("session")
    w = windows.get(R)
    if w is None:
        return {"window_root": R, "ok": False, "problem": "the ledger opens no session for this session's window root"}
    rs, op, end = w.get("reserve"), w["open"], w.get("end")
    b, ob = (rs or {}).get("body", {}), op["body"]
    chk = {"window_root": R, "session": op["session"], "is_the_verifiers": op["session"] == sid,
           "reserve_seq": (rs or {}).get("seq"), "open_seq": op["seq"], "end_seq": (end or {}).get("seq"),
           "lane": Ch.lane_of(op), "reserve_by": b.get("by"), "open_by": ob.get("by"), "peer": b.get("peer"),
           "open_peer": ob.get("peer"), "key_is_pin": ob.get("key_sha512") == pin and b.get("key_sha512") == pin,
           "scheme": b.get("scheme"), "epoch": b.get("epoch"), "families": b.get("families"),
           "in_period": op["session"] in ((periods.get(period) or {}).get("windows") or []),
           "end": None if end is None else {f: end["body"].get(f) for f in
                                            ("closed", "complete", "rc", "refused_hellos", "released", "ended_utc")}}
    chk["ok"] = bool(chk["is_the_verifiers"] and chk["lane"] == lane and chk["open_by"] == lane and chk["peer"] == peer
                     and chk["open_peer"] == peer and chk["key_is_pin"] and chk["scheme"] == coins.SCHEME
                     and chk["epoch"] == k and chk["in_period"] and end is not None
                     and end["body"].get("closed") is True and end["body"].get("refused_hellos") == 0)
    return chk


def signed_lines(d: Path, kind: str, R: str, heads: list[Path], key: Ch.ChallengerKey | None,
                 period: str | None) -> tuple[list[str], dict]:
    """The release log's lines from the session's directory `d`, each line the one the challenger's close signs, read
    through the ledger (`trusted_session`, a gate's session) or under the key the run wrote (`session_of`, a served
    one)."""
    files = remote.session_files(d)
    if kind == "gate":
        remote.trusted_session(d, heads, {"root": R}, key, CH.lms_verify)
        got = Ch.audited_session(Ch.ledger_head([remote.ledger_lines(p) for p in heads], key, CH.lms_verify),
                                 {"root": R}, key, CH.lms_verify)
        if got.period != period:
            raise Ch.CoinError(f"the ledger holds the session's window in period {got.period}, not {period}")
        trust = {"read": "trusted_session", "period": got.period, "cap": got.cap, "circuit": got.circuit}
    else:
        key = Ch.ChallengerKey.from_json(json.loads((d / remote.KEY).read_text()))
        Ch.session_of(key, files["record"], files["log"], files["close"], CH.lms_verify)
        trust = {"read": "session_of", "key_sha512": hashlib.sha512(key.public).hexdigest(),
                 "note": "a served challenger on the verifier's machine: no ledger, not the auditor's clock"}
    close = Ch.ChallengerClose.from_json(files["close"])
    if len(files["log"]) != close.lines or Ch.log_chain(files["log"]) != close.chain:
        raise Ch.CoinError("the release log read is not the one the close signs")
    return files["log"], trust


def analyse(out: Path, *, host_bytes: int, pin: str | None, period: str | None, heads: list[Path] | None,
            lane: str = LANE, peer: str = PEER, ingress: tuple[float, ...] = INGRESS) -> dict:
    v = json.loads((out / "verifier.json").read_text())
    reg = v["registration"]
    D = bytes.fromhex(reg["digest"])
    ch_dir = out / "challenger"
    kinds = {(s.get("challenger") or {}).get("challenger") for s in v["sessions"]}
    key = windows = periods = None
    doc: dict = {"registration": reg["digest"], "host_bytes": host_bytes, "ingress": list(ingress),
                 "verifier_rtt_ms": {k: x / 1e6 for k, x in v.get("rtt", {}).items() if k.endswith("_ns")}}
    if "gate" in kinds:
        if pin is None or period is None:
            raise SystemExit("auditor_line: a gate run's line needs --pin and --period")
        heads = [*(heads or []), *sorted(ch_dir.glob("*.ledger.jsonl"))]
        key = next((k for k in (Ch.ChallengerKey.from_json(json.loads(p.read_text()))
                                for p in sorted(ch_dir.glob("*/out/key.json")))
                    if hashlib.sha512(k.public).hexdigest() == pin), None)
        if key is None:
            raise SystemExit(f"auditor_line: no session's key.json under {ch_dir} is the pinned key {pin[:16]}…")
        windows, periods, doc["ledger"] = ledger_check(heads, key, period, lane)
        doc["ledger"]["heads_read"] = [str(p) for p in heads]
    rows, line = [], {}
    for s in v["sessions"]:
        k, c = s["k"], s.get("challenger") or {}
        R = coins.window(D, k)["root"]
        row = {"k": k, "accepted": s["accepted"], "failure": s.get("failure"), "pace": s.get("pace"),
               "challenger": c.get("challenger"), "challenger_session": c.get("session"),
               "verifier_coin_check": c.get("reason")}
        if c.get("challenger") == "gate":
            row["ledger"] = session_check(s, k, R, windows, periods, period=period, lane=lane, peer=peer, pin=pin)
            d = ch_dir / c["session"] / "out"
        elif c.get("challenger") == "serve":
            d = Path(c["dir"]) if Path(c["dir"]).is_dir() else ch_dir / c["session"]
        else:
            rows.append({**row, "skipped": "no challenger log: coins drawn in the verifier's process"})
            continue
        try:
            lines, row["trust"] = signed_lines(d, c["challenger"], R, heads or [], key, period)
        except (Ch.CoinError, OSError, ValueError, KeyError) as e:
            rows.append({**row, "refused": f"the challenger's session is not readable as signed: {e}"})
            continue
        row["log"] = log_checks(lines, R)
        if not (s.get("pace") and s["accepted"]):
            rows.append(row)
            continue
        seen, row["named"] = auditor_seen(lines, s["windows_seen"])
        a = copy.deepcopy(s)
        a["windows_seen"] = seen
        row["groups"] = len(seen)
        row["shares_auditor_clock"] = shares(a, reg, host_bytes, ingress)
        row["shares_verifier_clock"] = shares(s, reg, host_bytes, ingress)
        row["cross_check"] = cross_check(s["windows_seen"], seen)
        good = row.get("ledger", {"ok": True})["ok"] and not row["log"]["coin_before_its_root"] \
            and row["log"]["clock_monotone"] and row["log"]["sigma0_after_window_root"] and c.get("reason") is None
        row["of_record"] = bool(good and c.get("challenger") == "gate")
        rows.append(row)
        t = f"T={s['pace']['t_ms'] / 1000:g}s"
        for clock in ("auditor", "verifier"):
            for b, x in row[f"shares_{clock}_clock"].items():
                cur = line.setdefault(t, {}).setdefault(clock, {}).get(b)
                if cur is None or x["time"] < cur["time"]:
                    line[t][clock][b] = {"k": k, "time": x["time"], "snapshot": x["snapshot"],
                                         "of_record": row["of_record"]}
    doc["sessions"], doc["line"] = rows, line
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the verifier's out/ (verifier.json, challenger/)")
    ap.add_argument("--host-bytes", type=int, required=True, help="the node's host DRAM, bytes (verifier.py's)")
    ap.add_argument("--pin", help="SHA-512 of the challenger's public key (a gate run)")
    ap.add_argument("--period", help="the audit's period on the ledger (a gate run)")
    ap.add_argument("--ledger", type=Path, action="append",
                    help="a published head of the auditor's ledger, read with each session's (challenger/*.ledger.jsonl): "
                         "all are checked to extend one another, and the longest is read")
    ap.add_argument("--lane", default=LANE)
    ap.add_argument("--peer", default=PEER)
    ap.add_argument("--ingress", type=float, nargs="+", default=list(INGRESS), metavar="B/S")
    a = ap.parse_args(argv)
    doc = analyse(a.out, host_bytes=a.host_bytes, pin=a.pin, period=a.period, heads=a.ledger, lane=a.lane, peer=a.peer,
                  ingress=tuple(a.ingress))
    print(json.dumps(doc, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
