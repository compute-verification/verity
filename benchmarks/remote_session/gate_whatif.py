"""The what-if line beside a gate run's line of record: each accepted paced session replayed with every exchange between
the verifier and the challenger a round trip r instead of the session's own, every other time the record holds kept,
and its line (`auditor_line.shares`) on the auditor's clock and on the verifier's at each r. Replayed at the session's
own round trip it is the record: every group's coin, root and draws on the verifier's clock and every coin's release
and root's receipt on the challenger's come back to the nanosecond, and so the line `auditor_line.analyse` reads
(`identity`, checked; the script exits 1 when it fails).

The loop replayed is `verifier.py`'s at k = 1, as a session's record shows it, and a record of another form is refused:
at root g the verifier asks the challenger for openings coin g and σ_(g+2) in one round trip (`coins.SessionCoins.ahead`),
sends window g's draws (its `draws_sent_ns` D_g), then σ_(g+2) (its `released_ns` R_(g+2)); σ_0 and σ_1 each go alone,
stamped before they are asked (`audit.AuditRun._release`), and the last two groups' openings coins go alone. Group g's
root arrives at A_g. The prover's record (`prover.json`) has each pass from its coin's arrival to its root, ph_g, and
from its root to its send, gp_g, and τ, half the verifier↔prover round trip (verifier.json's `rtt` median), puts them on
the verifier's clock: root g seen at S_g = A_g − τ − gp_g, coin g arrived at C_g = S_g − ph_g. So:

- *The prover.* Pass g runs from max(C_g, root g − 1 sent) for S_g − that, its recorded time; coin g takes C_g − R_g
  from its stamp, which for σ_0 and σ_1 holds their fetch.
- *The verifier.* It takes root g at Q_g = max(A_g, b_g), once it has sent window g's boundary b_g (`boundary_ns`:
  σ_(g+1), or the close after the last window), and its exchange X_g = D_g − Q_g is the challenger's own time P_g
  (first receipt to last release of the round trip, on its log) and net_g, the rest: the network both ways and the
  verifier's time around it. The session's round trip r0 is the least net_g (or σ_0's fetch, R_1 − R_0 less its P, if
  less). Round trip r takes r0 − r off every exchange and off σ_0's and σ_1's fetches, and keeps each exchange's excess
  over r0 (a queue behind other work, compute, jitter); σ_1 leaves no sooner than its window allows. Window g + 1's
  boundary leaves its recorded step after D_g. Root 0 also waits on σ_1's fetch, which the loop's record doesn't
  stamp: it is taken as coin 1's arrival at the prover less τ, C_1 − τ, which moves with the fetch.
- *The challenger.* Its receipt of root g's request and its release of each coin move with the verifier's times, each
  crossing half of r0 − r shorter. The auditor's-clock line reads a release and a later receipt, one crossing each way,
  so the split doesn't move it.

r0 is read from both clocks, so a host run that takes no `ss -ti` samples needs no stand-in for it. What a row at r
claims is the line this session's own passes, the verifier's own steps and the challenger's own times would have given
with every exchange r0 − r faster: what else would move with the challenger nearer (the prover's passes beside the
openings, the verifier's other work, a later session's pace) is the record's, unreplayed.

A row may also take the passes faster: each pass's own time (S_g − max(C_g, root g − 1 sent)) times F (`--pass-scale`),
or every pass MS (`--pass-ms`), each coin's trip, root's send and verifier's step kept. Rows are every pass model at the
session's own round trip and at each r. The replay keeps the session's groups, so a stretch of them spans less time at
faster passes; at the stated T the verifier would pace more groups into a stretch (`--pace-pass-ms`), which this
doesn't model. Each row counts the groups whose pass waited on its coin and the roots that waited on the verifier's
boundary, so it says which side the line is waiting on.

    python gate_whatif.py OUT --line AUDITOR_LINE_JSON --prover PROVER_JSON [--rtt-ms MS ...] [--pass-scale F ...]
        [--pass-ms MS ...] [--table FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import auditor_line as al  # noqa: E402
from verity.core.protocols.sampled_proofs import remote  # noqa: E402

RTT_MS = (0.3, 1.15, 10.0, 25.0, 50.0)
"""The round trips a row is replayed at by default, ms: in-region and a few farther."""
FAMILIES = ("challenge", "openings")


class Refused(ValueError):
    """A session whose record isn't the loop this replays."""


def log_coins(lines: list[str]) -> list[tuple[str, int, str, int, int]]:
    """A release log's coins in its order after its head: (label, index, root asked after, received_at, released_at),
    on the challenger's clock."""
    out = []
    for i in range(1, len(lines) - 1, 2):
        root, release = json.loads(lines[i]), json.loads(lines[i + 1])
        out.append((root["label"], root["index"], root["root"], root["received_at"], release["released_at"]))
    return out


@dataclass(frozen=True)
class SessionTimes:
    """A session's times as the replay reads them, ns: the verifier's series R, A, D and the auditor's Ra (σ_g's
    release) and Aa (root g's first receipt), with the residuals that rebuild them."""
    n: int
    tau: int
    window: int
    """The window's stated length: σ_1 leaves no sooner than this after σ_0."""
    R: list[int]
    A: list[int]
    D: list[int]
    Ra: list[int]
    Aa: list[int]
    coin: list[int]
    """C_g − R_g: coin g's time from its stamp to the prover."""
    pure: list[int]
    """S_g − max(C_g, root g − 1 sent): pass g's own time."""
    gp: list[int]
    X: list[int]
    """D_g − Q_g: the verifier's exchange at root g."""
    P: list[int]
    """The challenger's own time in that exchange."""
    step: list[int]
    """b_(g+1) − D_g: window g + 1's boundary (σ_(g+2), or the close after the last) after window g's draws."""
    to_aa: list[int]
    """Aa_g − Q_g."""
    from_ra: list[int]
    """For σ_0 and σ_1 Ra_g − R_g; for the rest D_(g−2) − Ra_g."""
    fetch0: int
    """σ_0's fetch less the challenger's time: R_1 − R_0 − its P."""

    @property
    def net(self) -> list[int]:
        return [x - p for x, p in zip(self.X, self.P)]

    @property
    def r0(self) -> int:
        return min(*self.net, self.fetch0)

    def queue(self) -> list[int]:
        """How long each root waited for the verifier to send its window's boundary."""
        b = self.R[1:] + [self.D[-2] + self.step[-1]]
        return [max(0, x - a) for x, a in zip(b, self.A)]


def record(s: dict, lines: list[str], auditor: list[dict], prover: dict, tau: int) -> SessionTimes:
    """Session `s`'s record (`verifier.json`'s), its signed release log `lines`, its groups on the challenger's clock
    (`auditor_line.auditor_seen`), its prover's session (`prover.json`'s), and τ; refused unless it is the loop the
    replay models."""
    seen, n = s["windows_seen"], len(s["windows_seen"])
    if (s.get("pace") or {}).get("k") != 1:
        raise Refused(f"its pace is k = {(s.get('pace') or {}).get('k')}: the replay models k = 1")
    if n < 4 or any(w["passes"] != 1 or len(w["arrivals_ns"]) != 1 or w["draws_sent_ns"] is None
                    or w["boundary_ns"] is None for w in seen):
        raise Refused(f"its {n} groups aren't each one root with its draws and its boundary sent")
    if any(seen[g]["boundary_ns"] != seen[g + 1]["released_ns"] for g in range(n - 1)):
        raise Refused("a window's boundary isn't the next window's coin's release")
    ph, gp = prover.get("pass_host_ns") or [], prover.get("gap_host_ns") or []
    if len(ph) != n or len(gp) != n:
        raise Refused(f"the prover's record has {len(ph)} passes for its {n} groups")
    log = log_coins(lines)
    if {label for label, *_ in log} - set(FAMILIES):
        raise Refused(f"its release log has {sorted({x[0] for x in log} - set(FAMILIES))} coins: the replay models "
                      "challenge and openings coins only")
    at = {(label, i): j for j, (label, i, *_) in enumerate(log)}
    if any((f, g) not in at for f in FAMILIES for g in range(n)):
        raise Refused(f"its release log hasn't a challenge and an openings coin for each of its {n} groups")
    if [x[:2] for x in log[:2]] != [("challenge", 0), ("challenge", 1)]:
        raise Refused("its release log doesn't start with σ_0 and σ_1, each asked alone")
    alone = [g for g in range(n - 2) if (j := at["openings", g]) + 1 >= len(log)
             or log[j + 1][:3] != ("challenge", g + 2, log[j][2])]
    if alone:
        raise Refused(f"σ_(g+2) wasn't asked with openings coin g, after its root, in one round trip at g = {alone[:8]} "
                      "(coins.SessionCoins.ahead): the replay models a k = 1 loop whose every root asks its two coins at once")
    R = [w["released_ns"] for w in seen]
    A = [w["arrivals_ns"][-1] for w in seen]
    D = [w["draws_sent_ns"] for w in seen]
    b = [w["boundary_ns"] for w in seen]
    step = [b[g + 1] - D[g] for g in range(n - 1)]
    if min(step) < 0:
        raise Refused(f"window g + 1's boundary left before window g's draws at g = "
                      f"{[g for g, x in enumerate(step) if x < 0][:8]}")
    Ra = [w["released_ns"] for w in auditor]
    Aa = [w["arrivals_ns"][-1] for w in auditor]
    S = [a - tau - x for a, x in zip(A, gp)]
    C = [x - p for x, p in zip(S, ph)]
    pure = [S[0] - C[0]] + [S[g] - max(C[g], A[g - 1] - tau) for g in range(1, n)]
    Q = [max(A[0], b[0], C[1] - tau)] + [max(a, x) for a, x in zip(A[1:], b[1:])]
    end = [log[at["challenge", g + 2]][4] if g + 2 < n else log[at["openings", g]][4] for g in range(n)]
    P = [end[g] - log[at["openings", g]][3] for g in range(n)]
    sigma0 = log[at["challenge", 0]]
    return SessionTimes(n=n, tau=tau, window=s["window_ns"], R=R, A=A, D=D, Ra=Ra, Aa=Aa,
                        coin=[c - r for c, r in zip(C, R)], pure=pure, gp=gp,
                        X=[d - q for d, q in zip(D, Q)], P=P, step=step, to_aa=[a - q for a, q in zip(Aa, Q)],
                        from_ra=[Ra[0] - R[0], Ra[1] - R[1]] + [D[g - 2] - Ra[g] for g in range(2, n)],
                        fetch0=R[1] - R[0] - (sigma0[4] - sigma0[3]))


def passes_of(rec: SessionTimes, scale: float | None = None, ms: float | None = None) -> list[int]:
    """Each group's pass time under a pass model: its own times `scale`, or `ms` for every pass, or (neither) its
    own."""
    if ms is not None:
        return [round(ms * 1e6)] * rec.n
    return list(rec.pure) if scale is None else [round(x * scale) for x in rec.pure]


def replay(rec: SessionTimes, rtt_ns: int, passes: list[int] | None = None) -> dict[str, list[int]]:
    """The session's times with every exchange `rtt_ns` instead of `rec.r0` and each pass `passes` (its own by
    default, `passes_of`): the verifier's series R, A, D and the auditor's Ra, Aa; `rec`'s own at `rtt_ns` =
    `rec.r0`."""
    d = rec.r0 - rtt_ns
    out_h = d // 2
    in_h = d - out_h
    n = rec.n
    pure = rec.pure if passes is None else passes
    R, A, D, Aa = [0] * n, [0] * n, [0] * n, [0] * n
    R[0] = rec.R[0]
    R[1] = max(R[0] + rec.window, rec.R[1] - d)
    free = None
    for g in range(n):
        c = R[g] + rec.coin[g] - (d if g < 2 else 0)
        A[g] = (c if free is None else max(c, free)) + pure[g] + rec.gp[g] + rec.tau
        free = A[g] - rec.tau
        q = max(A[g], R[g + 1] if g + 1 < n else D[g - 1] + rec.step[g - 1])
        if g == 0:
            q = max(q, R[1] + rec.coin[1] - d - rec.tau)
        D[g] = q + rec.X[g] - d
        Aa[g] = q + rec.to_aa[g] - out_h
        if g + 2 < n:
            R[g + 2] = D[g] + rec.step[g]
    Ra = [R[0] + rec.from_ra[0] - out_h, R[1] + rec.from_ra[1] - out_h] + [
        D[g - 2] - rec.from_ra[g] + in_h for g in range(2, n)]
    return {"R": R, "A": A, "D": D, "Ra": Ra, "Aa": Aa}


def waits(rec: SessionTimes, t: dict[str, list[int]], rtt_ns: int) -> dict[str, int]:
    """In `replay`'s series `t` at `rtt_ns`, the groups after the first whose pass waited on its coin (it reached the
    prover after root g − 1 was sent) and the roots that waited for the verifier to send their window's boundary."""
    d, n, R, A, D = rec.r0 - rtt_ns, rec.n, t["R"], t["A"], t["D"]
    b = R[1:] + [D[-2] + rec.step[-1]]
    return {"coin": sum(R[g] + rec.coin[g] - (d if g < 2 else 0) > A[g - 1] - rec.tau for g in range(1, n)),
            "verifier": sum(a < x for a, x in zip(A, b))}


def seen_at(s: dict, released: list[int], arrived: list[int]) -> dict:
    return {**s, "windows_seen": [{**w, "released_ns": released[g], "arrivals_ns": [arrived[g]]}
                                  for g, w in enumerate(s["windows_seen"])]}


def row_at(s: dict, reg: dict, rec: SessionTimes, rtt_ns: int, host_bytes: int, rates: dict,
           passes: list[int] | None = None) -> dict:
    t = replay(rec, rtt_ns, passes)
    v, a = seen_at(s, t["R"], t["A"]), seen_at(s, t["Ra"], t["Aa"])
    return {"rtt_ms": rtt_ns / 1e6, "series": t, "L_verifier_ms": al.q(al.group_l(v["windows_seen"])),
            "L_auditor_ms": al.q(al.group_l(a["windows_seen"])), "passes_s": (t["A"][-1] - t["R"][0]) / 1e9,
            "pass_ms": al.q(rec.pure if passes is None else passes), "waits": waits(rec, t, rtt_ns),
            "shares_verifier_clock": al.shares(v, reg, host_bytes, rates, clock="replay"),
            "shares_auditor_clock": al.shares(a, reg, host_bytes, rates, clock="replay")}


def identity(rec: SessionTimes, at_r0: dict, row: dict) -> dict:
    """Whether the replay at the session's own round trip is its record: each series to the ns, and both clocks'
    shares as `auditor_line.analyse` read them (`row`)."""
    own = {"R": rec.R, "A": rec.A, "D": rec.D, "Ra": rec.Ra, "Aa": rec.Aa}
    off = {k: sum(x != y for x, y in zip(at_r0["series"][k], own[k])) for k in own}
    shares = {c: at_r0[f"shares_{c}_clock"] == row[f"shares_{c}_clock"] for c in ("auditor", "verifier")}
    return {"holds": not any(off.values()) and all(shares.values()), "groups_off": off, "shares_equal": shares}


def pass_models(pass_scale: tuple[float, ...], pass_ms: tuple[float, ...]) -> list[dict]:
    """The pass models a grid takes: {"scale": F} for each of `pass_scale`, then {"ms": MS} for each of `pass_ms`."""
    return [{"scale": f} for f in pass_scale] + [{"ms": m} for m in pass_ms]


def whatif(out: Path, prover: dict, doc: dict, *, rtt_ms: tuple[float, ...] = RTT_MS,
           pass_scale: tuple[float, ...] = (1.0,), pass_ms: tuple[float, ...] = ()) -> dict:
    """The line of record `doc` (`auditor_line.analyse(out, ...)`'s, `auditor_line.py`'s output) with each of its
    accepted paced sessions replayed under each pass model (`pass_models`) at its own round trip and at each of
    `rtt_ms`, with `prover` the prover's record."""
    grid = [(m, ms) for m in pass_models(pass_scale, pass_ms) for ms in (None, *rtt_ms)]
    if "rates" not in doc:
        raise SystemExit("gate_whatif: the line of record was written before its forms' rates (its `ingress`): run "
                         "this tree's auditor_line.py on the run again and pass its output as --line")
    host_bytes, rates = doc["host_bytes"], doc["rates"]
    v = json.loads((out / "verifier.json").read_text())
    reg, tau = v["registration"], v["rtt"]["median_ns"] // 2
    by_k = {p["k"]: p for p in prover["sessions"]}
    rows = []
    for s, row in zip(v["sessions"], doc["sessions"]):
        if "shares_auditor_clock" not in row:
            continue
        got: dict = {"k": s["k"], "T": row["T"], "credited": row["credit"]["ok"], "of_record": row.get("of_record")}
        rows.append(got)
        c = s["challenger"]
        d = out / "challenger" / c["session"] / "out" if c.get("challenger") == "gate" else \
            Path(c["dir"]) if Path(c["dir"]).is_dir() else out / "challenger" / c["session"]
        try:
            if s["k"] not in by_k:
                raise Refused("the prover's record has no session k")
            lines = remote.session_files(d)["log"]
            rec = record(s, lines, al.auditor_seen(lines, s["windows_seen"])[0], by_k[s["k"]], tau)
        except Refused as e:
            got["refused"] = str(e)
            continue
        own = row_at(s, reg, rec, rec.r0, host_bytes, rates)
        net = rec.net
        got.update(tau_ms=tau / 1e6, r0_ms=rec.r0 / 1e6, net_ms=al.q(net),
                   excess_ms=al.q([x - rec.r0 for x in net]), challenger_ms=al.q(rec.P), step_ms=al.q(rec.step),
                   pass_ms=al.q(rec.pure), queued_groups=sum(x > 0 for x in rec.queue()),
                   identity=identity(rec, own, row),
                   rows=[{"pass": m, "own_rtt": ms is None, **{k: x for k, x in r.items() if k != "series"}}
                         for m, ms, r in ((m, ms, row_at(s, reg, rec, rec.r0 if ms is None else round(ms * 1e6),
                                                         host_bytes, rates, passes_of(rec, m.get("scale"),
                                                                                        m.get("ms"))))
                                          for m, ms in grid)])
        got["own"] = {k: x for k, x in own.items() if k != "series"}
    doc_line = {}
    for t in sorted({r["T"] for r in rows}):
        ok = [r for r in rows if r["T"] == t and r["credited"] and "rows" in r]

        def worst(i: int, form: str) -> dict:
            return min(({"k": r["k"], **{f: r["rows"][i]["shares_auditor_clock"][form][f] for f in al.SHARES},
                         "l_max_ms": r["rows"][i]["shares_auditor_clock"][form]["worst_l_max_ms"],
                         "waits": r["rows"][i]["waits"]} for r in ok),
                       key=lambda x: x["time"], default=dict.fromkeys(al.SHARES, 0.0))

        doc_line[t] = {"credited_sessions": [r["k"] for r in ok], "rows": [
            {"pass": m, "rtt_ms": "own" if ms is None else ms, **{form: worst(i, form) for form in al.line.FORMS}}
            for i, (m, ms) in enumerate(grid)]}
    return {"registration": reg["digest"], "peer": doc.get("peer"), "line_of_record": doc["line"],
            "credit": doc["credit"], "cross_check": {r["k"]: r.get("cross_check") for r in doc["sessions"]},
            "sessions": rows, "whatif_line": doc_line,
            "identity_holds": all(r["identity"]["holds"] for r in rows if "identity" in r)}


def table(doc: dict) -> str:
    """`whatif`'s doc as Markdown: each session's round trip, identity and line at it, then the what-if line."""
    out = ["| session | T | credited | r0 (ms) | identity | own line, auditor's clock |", "|---|---|---|---|---|---|"]
    for r in doc["sessions"]:
        if "identity" not in r:
            out.append(f"| {r['k']} | {r['T']} | {r['credited']} | - | refused: {r.get('refused')} | - |")
            continue
        own = " / ".join(f"{x['time']:.4%}" for x in r["own"]["shares_auditor_clock"].values())
        out.append(f"| {r['k']} | {r['T']} | {r['credited']} | {r['r0_ms']:.3f} | "
                   f"{'holds' if r['identity']['holds'] else 'FAILS: ' + json.dumps(r['identity'])} | {own} |")
    out.append("")
    for t, line in doc["whatif_line"].items():
        out += ["", f"what-if line {t} (credited {line['credited_sessions']}), time-held / link-charged weights, the "
                "worst stretch's L_max and the groups that waited on their coin / on the verifier:", "",
                "| passes | r (ms) | line | L_max (ms) | waits: coin / verifier |", "|---|---|---|---|---|"]
        for x in line["rows"]:
            v = [y for k, y in x.items() if k not in ("pass", "rtt_ms")]
            p = f"{x['pass']['scale']:g}× own" if "scale" in x["pass"] else f"{x['pass']['ms']:g} ms each"
            r = x["rtt_ms"] if x["rtt_ms"] == "own" else f"{x['rtt_ms']:g}"
            w = v[0].get("waits") or {}
            out.append(f"| {p} | {r} | " + " / ".join(f"{y['time']:.4%}" for y in v) +
                       f" | {v[0].get('l_max_ms', 0):.1f} | {w.get('coin', '-')} / {w.get('verifier', '-')} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the verifier's out/ (verifier.json, challenger/)")
    ap.add_argument("--line", type=Path, required=True, help="auditor_line.py's output over OUT: the line of record, "
                    "its sessions' credit and the gate sessions' peer")
    ap.add_argument("--prover", type=Path, required=True, help="the prover run's prover.json")
    ap.add_argument("--rtt-ms", type=float, nargs="+", default=list(RTT_MS), metavar="MS")
    ap.add_argument("--pass-scale", type=float, nargs="+", default=[1.0], metavar="F",
                    help="replay with each pass's own time times F")
    ap.add_argument("--pass-ms", type=float, nargs="+", default=[], metavar="MS", help="replay with every pass MS")
    ap.add_argument("--table", type=Path, help="write each session's identity and the what-if line here, as Markdown")
    a = ap.parse_args(argv)
    doc = whatif(a.out, json.loads(a.prover.read_text()), json.loads(a.line.read_text()), rtt_ms=tuple(a.rtt_ms),
                 pass_scale=tuple(a.pass_scale), pass_ms=tuple(a.pass_ms))
    print(json.dumps(doc, indent=1, default=str))
    if a.table is not None:
        a.table.write_text(table(doc))
    said = {r["k"]: r["identity"]["holds"] if "identity" in r else r["refused"] for r in doc["sessions"]}
    print(f"gate_whatif: peer {doc['peer']}; the replay at each session's own round trip is its record: {said}",
          file=sys.stderr)
    return 0 if doc["identity_holds"] else 1


if __name__ == "__main__":
    sys.exit(main())
