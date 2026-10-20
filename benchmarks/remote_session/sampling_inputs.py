"""The remote bandwidth audit's inputs to a joint sampling plan (PoUW and the memory claim under one partition): the
memory line's ε against the openings proved, the pass rate and pacing, and the window; the rebuilds the plan leaves in
the line; and the serving cost of passes and of proved openings, at three served operating points.

    python sampling_inputs.py --verifier VERIFIER_JSON --cost-model COST_MODEL_JSON --out DIR

`VERIFIER_JSON` is a run's `verifier.json` with accepted paced sessions (#2612's `ledger.paced`). The script recomputes
every recorded stretch from the session's groups and refuses to go on unless the ledger's figures come back, then checks
the closed form below against the same arithmetic on uniform sessions, and only then tabulates the closed form.
`COST_MODEL_JSON` is a `sampled_proofs.costfit` record whose observations include `BlockOpeningCheck_v1`'s sessions.

With o openings proved a second, d of them a group of k passes at r passes a second, and β = B/d, where
B = −S·expm1(ln δ/S) ≈ ln(1/δ) is `audit.pooled`'s budget over S draws, the line loses this many blocks over a span
Θ, as a fraction of the store's N blocks:

    sampling, snapshot        B / S_Θ                                     S_Θ the openings drawn over Θ
    sampling, held over time  (B/o + min(1, β)·(L_max − k/r)) / Θ
    rebuilds, held over time  2·L_max·(X_r + I)·(Θ + 2·L_max) / (Θ·N)

Θ is a stretch, as the recorded line reads each (`paced_line` charges every stretch the session's whole pooled wrong
count and cites the worst), or the whole session, with the wrong count charged once (the pooled reading, the same
arithmetic over one stretch of every credited group). The sampling terms place the wrong receipts `audit.pooled` allows
where they cost most: one group's worth in the group after Θ at L_max/k a receipt, the rest at 1/r. The rebuild term
ends an absence of up to 2·L_max at every rebuild and ingress the ceilings allow.

L_max = f·k/r + c with f groups in flight. The recorded sessions run their passes back to back with two in flight (each
group's coin leaves before the root of the group before it arrives), so f = 2 there, with c measured. f = 1 is a coin
sent on a timer at r/k a second once the root before it is in, which no session has run: modeled, with the same c.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, timing  # noqa: E402
from verity.core.protocols.sampled_proofs import costfit  # noqa: E402

import wire  # noqa: E402

LN = -math.log(audit.DELTA)
BLOCK = 4096
STATEMENT = 512
"""Openings a statement: C-Flock's ceiling, m = k_log + nbl ≤ 35, at the unit's k_log 26."""
FLOOR_GPU_S = 0.0799
"""An opening's GPU device phase at a full statement, firewall on: note:20261018T1937Z-report-zk-openings
(r20261018-211523-93ec), measured."""
DEPLOYED_EXTRA_ANDS = (45.4e6 - 20.3e6) + 5e6
"""p1024/k3/masked's decode over p512/k2's, and the deployed trees' depths, per opening: the same note's caveats,
modeled."""
MEMSAT = {"run": "r20261018-222953-fbf1", "plain": 4260.0, "codec": 3271.0,
          "arms": [(65.84, 5.91e-3, 0.9689, 1999.0), (29.81, 6.18e-3, 0.9862, 2669.0),
                   (14.90, 3.87e-3, 0.9934, 3083.0)]}
"""int8 Llama-3.1-8B on one GPU at saturation (note:hash/20261018T1935Z-draft-memsat): output tokens/s plain and with
the codec, and per receipts arm (passes/s, serving s a pass, γ at spec R, tokens/s)."""


def groups_of(seen: list[dict]) -> SimpleNamespace:
    s, n = [w["released_ns"] for w in seen], [w["passes"] for w in seen]
    ends = [max(w["arrivals_ns"][-1] if w["arrivals_ns"] else s[g], s[g + 1] if g + 1 < len(s) else 0)
            for g, w in enumerate(seen)]
    L = [e - x for e, x in zip(ends, s)]
    gap = [s[g + 1] - s[g] if g + 1 < len(s) else L[g] for g in range(len(seen))]
    return SimpleNamespace(s=s, n=n, ends=ends, L=L, gap=gap)


def span_line(g: SimpleNamespace, lo: int, hi: int, *, wrong: float, blocks: int, per_ns: float, chip: float) -> dict:
    """`paced_line`'s arithmetic (#2612's verifier.py) over groups [lo, hi), group hi the one after them: the span (from
    the first group's end when lo is 0), L_max, and the blocks lost in both forms, `per_ns` the rebuild and ingress
    ceilings in blocks a ns and `chip` the chip's blocks."""
    span = g.s[hi] - (g.ends[0] if lo == 0 else g.s[lo])
    P = sum(g.n[lo:hi])
    l_max = max(g.L[max(0, lo - 1):hi + 1])
    rebuilds = per_ns * (span + 2 * l_max)
    rates = sorted((((l_max if x == hi else g.gap[x]) / g.n[x], g.n[x] * blocks) for x in range(lo, hi + 1)),
                   key=lambda t: t[0], reverse=True)
    left, fill = wrong, 0.0
    for v, cap in rates:
        took = min(left, cap)
        fill, left = fill + took * v, left - took
        if left <= 0:
            break
    return {"span_ns": span, "passes": P, "groups": hi - lo, "l_max_ns": l_max, "rebuild_blocks": rebuilds,
            "rebuild_absent_blocks": 2 * l_max * rebuilds / span, "wrong_absent_blocks": fill / span,
            "snapshot_lost_blocks": wrong / P + rebuilds / P + chip,
            "held_lost_blocks": chip + (2 * l_max * rebuilds + fill) / span}


def record_of(windows: list[tuple[int, int]]) -> SimpleNamespace:
    """What `audit.pooled` reads of a record: each window's receipts and draws."""
    return SimpleNamespace(windows=[SimpleNamespace(receipts=u, profile=SimpleNamespace(
        delta=audit.DELTA, drawn={"receipt": d})) for u, d in windows])


def budget(S: int) -> float:
    return -math.expm1(math.log(audit.DELTA) / S) * S


def closed(*, o: float, d: float, k: float, r: float, l_max: float, theta: float, B: float = LN) -> tuple[float, float]:
    """The sampling terms (snapshot, held over time) as fractions of the store, the module's closed form."""
    return B / (o * theta), (B / o + min(1.0, B / d) * (l_max - k / r)) / theta


def agree(name: str, got: float, want: float, rel: float = 1e-9) -> None:
    if not math.isclose(got, want, rel_tol=rel, abs_tol=1e-6):
        raise SystemExit(f"{name}: recomputed {got!r}, recorded {want!r}: the inputs below don't read this session")


def reproduce(v: dict, ingress_bytes_per_s: float) -> list[dict]:
    """Each accepted paced session's recorded line, recomputed; with its pooled (session) reading and its c."""
    reg = v["registration"]
    N, enc = reg["blocks"], wire.codec_named(reg["codec"]).encode_squarings
    out = []
    for i, sess in enumerate(v["sessions"]):
        paced = sess.get("ledger", {}).get("paced")
        if not sess.get("accepted") or not paced or "stretches" not in paced:
            continue
        pace, seen = sess["pace"], sess["windows_seen"]
        rec = record_of([(u, w["profile"]["drawn"]["receipt"]) for u, w in
                         zip(sess["record"]["public"]["receipts"], sess["record"]["profile"]["windows"])])
        wrong = audit.pooled(rec)[0]
        agree(f"session {i} wrong", wrong, paced["wrong"])
        x_r = sess["ledger"]["inputs"]["squarings_per_s"] / enc
        per_ns = (x_r + ingress_bytes_per_s / BLOCK) / 1e9
        chip = paced["chip_blocks"]
        g, G = groups_of(seen), pace["groups"]
        stretches = [span_line(g, m * G, (m + 1) * G, wrong=wrong, blocks=N, per_ns=per_ns, chip=chip)
                     for m in range(len(seen) // G - 1)]
        if len(stretches) != len(paced["stretches"]):
            raise SystemExit(f"session {i}: {len(stretches)} stretches against the ledger's {len(paced['stretches'])}")
        for m, (a, b) in enumerate(zip(stretches, paced["stretches"])):
            for key in ("span_ns", "passes", "l_max_ns", "rebuild_blocks", "rebuild_absent_blocks",
                        "wrong_absent_blocks"):
                agree(f"session {i} stretch {m} {key}", a[key], b[key])
            agree(f"session {i} stretch {m} held", a["held_lost_blocks"], b["held over time"]["lost_blocks"])
            agree(f"session {i} stretch {m} snapshot", a["snapshot_lost_blocks"], b["held on average"]["lost_blocks"])
        d, k, S = pace["draws"], pace["k"], sum(w.profile.drawn["receipt"] for w in rec.windows)
        B = budget(S)
        agree(f"session {i} wrong as B·kN/d", B * k * N / d, wrong)
        for m, a in enumerate(stretches):
            theta, kr = a["span_ns"] / 1e9, a["span_ns"] / 1e9 / a["groups"]
            snap, held = closed(o=d / kr, d=d, k=k, r=k / kr, l_max=a["l_max_ns"] / 1e9, theta=theta, B=B)
            agree(f"session {i} stretch {m} closed form, held", held * N, a["wrong_absent_blocks"])
            a.update(eps_snapshot=wrong / a["passes"] / N, eps_held=a["wrong_absent_blocks"] / N,
                     rebuild_held=a["rebuild_absent_blocks"] / N)
        whole = span_line(g, 0, pace["windows"] * G, wrong=wrong, blocks=N, per_ns=per_ns, chip=chip)
        whole.update(eps_snapshot=wrong / whole["passes"] / N, eps_held=whole["wrong_absent_blocks"] / N,
                     rebuild_held=whole["rebuild_absent_blocks"] / N)
        mean_gap = (g.s[-1] - g.s[0]) / (len(g.s) - 1) / 1e9
        worst = max(stretches, key=lambda a: a["held_lost_blocks"])
        out.append({"session": i, "t_ms": pace["t_ms"], "k": k, "groups": G, "draws": d, "windows": pace["windows"],
                    "draws_total": S, "budget": B, "wrong": wrong, "x_r": x_r, "ingress": ingress_bytes_per_s / BLOCK,
                    "chip_blocks": chip, "blocks": N, "passes_per_s": k / mean_gap, "mean_gap_s": mean_gap,
                    "l_max_s": max(g.L) / 1e9, "c_s": max(g.L) / 1e9 - 2 * k * mean_gap,
                    "gamma_spec": sess["ledger"]["spec"]["gamma"], "stretches": stretches, "worst": worst,
                    "pooled": whole})
    if not out:
        raise SystemExit("no accepted paced session with a recorded line in the verifier's record")
    return out


def check_closed_form() -> list[dict]:
    """The closed form against `span_line` on uniform sessions, a group spilling into the rest (β > 1) and not, and
    one openings coin drawing over m groups (d = s/m). It holds while the wrong receipts fit in the stretch (β < G)."""
    out = []
    for s, m, G, W, k, r, c, N in ((1, 1, 40, 3, 1, 8.0, 0.057, 10_000), (2, 1, 40, 3, 2, 12.0, 0.03, 10_000),
                                   (64, 1, 12, 4, 1, 11.89, 0.057, 10_000), (232, 1, 12, 7, 1, 15.0, 0.06, 130_023),
                                   (1, 4, 400, 3, 1, 8.0, 0.057, 10_000)):
        d = s / m
        gap, L = k / r * 1e9, (2 * k / r + c) * 1e9
        seen = [{"released_ns": x * gap, "passes": k, "arrivals_ns": [x * gap + L]} for x in range((W + 1) * G)]
        wrong = audit.pooled(record_of([(m * k * N, s)] * ((W + 1) * G // m)))[0]
        B = budget(s * (W + 1) * G // m)
        g = groups_of(seen)
        for lo, hi in [(m * G, (m + 1) * G) for m in range(W)] + [(0, W * G)]:
            a = span_line(g, lo, hi, wrong=wrong, blocks=N, per_ns=0.0, chip=0.0)
            theta = a["span_ns"] / 1e9
            snap, held = closed(o=d * r / k, d=d, k=k, r=r, l_max=L / 1e9, theta=theta, B=B)
            agree(f"uniform d={d} [{lo},{hi}) held", held * N, a["wrong_absent_blocks"])
            agree(f"uniform d={d} [{lo},{hi}) snapshot", B / (d * (hi - lo)) * N, wrong / a["passes"])
            out.append({"d": d, "k": k, "r": r, "c": c, "groups": [lo, hi], "beta": B / d, "eps_held": held,
                        "eps_snapshot": B / (d * (hi - lo))})
    return out


def opening_prices(cost_model: dict) -> dict:
    """GPU-s an opening at a full statement: the one fit's (`costfit`'s rates) at the measured unit and the deployed
    one, and the device floor at both."""
    fitted = costfit.from_record(cost_model)
    m = fitted.model()
    ands = [o["ands"] for o in cost_model["observations"] if o["template"].startswith("BlockOpeningCheck_v1")]
    if not ands:
        raise SystemExit("the cost model observed no BlockOpeningCheck_v1 session")
    A = ands[0]

    def fit(a: float) -> float:
        return (m.statement_fixed + m.statement_per_and * a) / STATEMENT + (m.staged_per_and + m.per_and) * a

    dep = A + DEPLOYED_EXTRA_ANDS
    return {"ands": A, "deployed_ands": dep, "fit": fit(A), "fit_deployed": fit(dep), "floor": FLOOR_GPU_S,
            "floor_deployed": FLOOR_GPU_S * dep / A, "rates": {"statement_fixed": m.statement_fixed,
            "statement_per_and": m.statement_per_and, "staged_per_and": m.staged_per_and, "per_and": m.per_and},
            "sources": sorted({s for f in cost_model["fits"].values() for s in f["sources"]})}


def points(recorded: list[dict]) -> dict:
    """The operating points: what each measured, and what is modeled."""
    rec = recorded[0]
    xi = rec["x_r"] + rec["ingress"]
    gib62 = 62 * 2 ** 30
    return {
        "32B x8": {"label": "Qwen2.5-32B bf16, 8 TP1 engines", "blocks": rec["blocks"], "gpus": 8, "shard_bytes": gib62,
                   "weights_bytes": 8 * 65_525_514_240, "c_s": max(x["c_s"] for x in recorded), "xi": xi,
                   "chip": rec["chip_blocks"],
                   "tokens_per_s": 1000.0, "codec_loss": None,
                   "source": "W2 round 5 (note:20261018T0720Z-report-north-star-node; r20261018-200634-12c0): store, "
                             "weights, c measured; tokens/s the offered load, not saturation"},
        "8B x8": {"label": "Llama-3.1-8B int8, an engine a GPU, 8 GPUs", "blocks": 8 * 2_359_296, "gpus": 8,
                  "shard_bytes": 9_663_676_416, "weights_bytes": 8 * (6.98e9 + 2 * 1.05e9),
                  "c_s": 0.072 - 2 / 51.7, "xi": xi, "chip": rec["chip_blocks"],
                  "tokens_per_s": 8 * MEMSAT["codec"], "codec_loss": 1 - MEMSAT["codec"] / MEMSAT["plain"],
                  "source": "memsat's GPU x8 (modeled); c from round 3's 8B node (L_max 72 ms at k = 1, 51.7/s)"},
        "8B x1": {"label": "Llama-3.1-8B int8 on 1 GPU of the node", "blocks": 2_359_296, "gpus": 1,
                  "shard_bytes": 9_663_676_416, "weights_bytes": 6.98e9 + 2 * 1.05e9, "c_s": 0.072 - 2 / 51.7,
                  "xi": xi, "chip": rec["chip_blocks"], "tokens_per_s": MEMSAT["codec"],
                  "codec_loss": 1 - MEMSAT["codec"] / MEMSAT["plain"],
                  "source": "memsat (note:hash/20261018T1935Z-draft-memsat; r20261018-222953-fbf1), measured"},
    }


def line_at(p: dict, *, o: float, d: float, k: float, r: float, theta: float, flight: int = 2) -> dict:
    """The module's closed form at an operating point: the line's terms as fractions of the store, and the line."""
    l_max = flight * k / r + p["c_s"]
    snap, held = closed(o=o, d=d, k=k, r=r, l_max=l_max, theta=theta)
    N = p["blocks"]
    rebuild = 2 * l_max * p["xi"] * (theta + 2 * l_max) / (theta * N)
    excess = 2 * (l_max - p["c_s"]) * p["xi"] * (theta + 2 * l_max) / (theta * N)
    chip = p["chip"] / N
    lost = (held + rebuild + chip) * N * BLOCK
    return {"l_max_s": l_max, "eps_snapshot": snap, "eps_held": held, "rebuild_held": rebuild,
            "rebuild_excess": excess, "chip": chip, "credited_bytes": max(0.0, p["weights_bytes"] - lost)}


def costs(p: dict, prices: dict, *, o: float, r: float) -> dict:
    """GPUs (GPU-s a second) the node spends: passes' serving at r·(shard / a GPU's read) on each GPU, and proving."""
    t_s = p["shard_bytes"] / timing.GPU_READ_BYTES_PER_S
    return {"pass_s": t_s, "serving_share": min(1.0, r * t_s), "passes_gpus": min(1.0, r * t_s) * p["gpus"],
            **{f"proving_gpus_{name}": o * prices[name] for name in ("fit", "floor", "fit_deployed", "floor_deployed")}}


def solve(p: dict, prices: dict, eps: float, *, theta: float, line_rule: bool, flight: int = 2,
          price: str = "fit", shared: bool = False) -> dict | None:
    """The cheapest (r, k, d) whose held-over-time sampling term over Θ is at most ε, and, under `line_rule`, whose
    rebuild excess over L_max's floor c, 2·f·(X_r + I)·k / (r·N), is at most ε too. For each (k, d), r is the least that
    meets them: k/r ≤ ε·N/(2·f·(X_r + I)), and k/r·(β + min(1, β)·(f − 1)) + min(1, β)·c ≤ ε·Θ.

    A group is a window of `audit.pooled`'s, and a window it draws nothing from may be wrong throughout, so d ≥ 1.
    `shared` lets one openings coin draw over m groups' receipts after all their roots (d = 1/m, m ≤ 512): no session
    has run it, so it is modeled, with the same closed form."""
    tau = eps * p["blocks"] / (2 * flight * p["xi"]) if line_rule else math.inf
    t_s = p["shard_bytes"] / timing.GPU_READ_BYTES_PER_S
    best = None
    for k in range(1, 9):
        for d in ([1 / m for m in range(512, 1, -1)] if shared else []) + list(range(1, 4097)):
            beta = LN / d
            room = eps * theta - min(1.0, beta) * p["c_s"]
            if room <= 0:
                continue
            r = k * max(1 / tau, (beta + min(1.0, beta) * (flight - 1)) / room)
            if r * t_s > 1:
                continue
            o = d * r / k
            cost = r * t_s * p["gpus"] + o * prices[price]
            if best is None or cost < best[0]:
                best = (cost, k, d, r, o)
    if best is None:
        return None
    _, k, d, r, o = best
    return {"eps": eps, "theta_s": theta, "line_rule": line_rule, "flight": flight, "price": price, "shared": shared,
            "k": k, "d": d, "r": r, "o": o, **line_at(p, o=o, d=d, k=k, r=r, theta=theta, flight=flight),
            **costs(p, prices, o=o, r=r)}


def pct(x: float, digits: int = 3) -> str:
    return f"{100 * x:.{digits}g}%"


def table(head: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                     + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verifier", type=Path, required=True)
    ap.add_argument("--cost-model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ingress-bytes-per-s", type=float, default=2.8e9,
                    help="the node's disks, as the record charged them")
    ap.add_argument("--window", type=float, default=60.0, help="T, s")
    ap.add_argument("--horizon", type=float, default=600.0, help="the pooled reading's session, s")
    a = ap.parse_args()
    v = json.loads(a.verifier.read_text())
    T, H = a.window, a.horizon

    recorded = reproduce(v, a.ingress_bytes_per_s)
    uniform = check_closed_form()
    prices = opening_prices(json.loads(a.cost_model.read_text()))
    pts = points(recorded)
    md = []

    md.append("## Recorded sessions, recomputed (each stretch matches the ledger to 1e-9)\n")
    md.append(table(["session", "T", "groups x draws", "passes/s", "L_max", "c", "wrong (pooled)",
                     "worst stretch: ε snapshot / held", "rebuild held", "pooled over the session: ε snapshot / held",
                     "session span"],
                    [[x["session"], f"{x['t_ms'] / 1e3:g} s", f"{x['groups']} x {x['draws']}",
                      f"{x['passes_per_s']:.2f}",
                      f"{x['l_max_s'] * 1e3:.1f} ms", f"{x['c_s'] * 1e3:.1f} ms", f"{x['wrong']:.4g}",
                      f"{pct(x['worst']['eps_snapshot'])} / {pct(x['worst']['eps_held'])}",
                      pct(x["worst"]["rebuild_held"]),
                      f"{pct(x['pooled']['eps_snapshot'])} / {pct(x['pooled']['eps_held'])}",
                      f"{x['pooled']['span_ns'] / 1e9:.1f} s"] for x in recorded]))

    p32 = pts["32B x8"]
    r32 = next((x for x in recorded if x["t_ms"] == T * 1e3), recorded[0])["passes_per_s"]
    rows = []
    for s_T in (206, 759, 2773, 6072, 11091, 33273):
        k = max(1, math.ceil(r32 * T / s_T))
        d = s_T * k / (r32 * T)
        o = s_T / T
        per = line_at(p32, o=o, d=d, k=k, r=r32, theta=T)
        pool = line_at(p32, o=o, d=d, k=k, r=r32, theta=10 * T)
        rows.append([f"{s_T:,}", k, f"{d:.3g}", pct(LN / s_T), pct(per["eps_held"]), pct(LN / (10 * s_T)),
                     pct(pool["eps_held"]), pct(per["rebuild_held"])])
    md.append(f"\n## ε against openings proved a {T:g}-s window (δ = 2^-40), at 32B x8's {r32:.2f} passes/s\n")
    md.append(table(["openings a window", "k", "draws a group", "ε a window, snapshot", "ε a window, held",
                     "pooled over 10 windows, snapshot", "pooled over 10, held", "rebuilds held (a window)"], rows))

    chosen = {}
    md.append(f"\n## The region: ε held over time pooled over {H:g} s, cheapest at the one fit's price\n")
    region = []
    for name, p in pts.items():
        for flight in (2, 1):
            for eps in (0.005, 0.01, 0.02) if name != "8B x1" else (0.01,):
                for line_rule, shared in ((True, False), (True, True), (False, False)):
                    s = solve(p, prices, eps, theta=H, line_rule=line_rule, flight=flight, shared=shared)
                    if s is not None:
                        s.update(point=name, window_s=T, per_window=closed(o=s["o"], d=s["d"], k=s["k"], r=s["r"],
                                                                           l_max=s["l_max_s"], theta=T)[1])
                        if eps == 0.01 and flight == 2 and not shared and (line_rule or name not in chosen):
                            chosen[name] = s
                    region.append(s or {"point": name, "eps": eps, "line_rule": line_rule, "flight": flight,
                                        "shared": shared, "infeasible": True})
            s = solve(p, prices, 0.01, theta=T, line_rule=True, flight=flight)
            region.append(dict(s, point=name, window_s=T, per_window=s["eps_held"], reading="a window") if s else
                          {"point": name, "eps": 0.01, "line_rule": True, "flight": flight, "reading": "a window",
                           "infeasible": True})
    rows = []
    for s in region:
        rule = ("line" if s["line_rule"] else "ε only") + (", shared draws (modeled)" if s.get("shared") else "")
        head = [s["point"], s["flight"], pct(s["eps"]), rule, s.get("reading", "pooled")]
        if s.get("infeasible"):
            rows.append(head + ["infeasible"] + [""] * 11)
            continue
        d = s["d"] if s["d"] >= 1 else f"1/{round(1 / s['d'])}"
        rows.append(head + [f"{s['r']:.3g}", s["k"], d, f"{s['o']:.3g}", f"{s['o'] * T:,.0f}",
                     f"{s['l_max_s'] * 1e3:.0f} ms",
                     pct(s["eps_held"]), pct(s["per_window"]), pct(s["rebuild_held"]),
                     f"{pct(s['serving_share'])} ({s['passes_gpus']:.2f})",
                     f"{s['proving_gpus_fit']:.2f} / {s['proving_gpus_floor']:.2f}",
                     f"{s['proving_gpus_fit_deployed']:.2f} / {s['proving_gpus_floor_deployed']:.2f}"])
    md.append(table(["point", "groups in flight", "ε*", "rule", "reading", "passes/s r", "k", "draws a group d",
                     "openings/s o",
                     f"openings a {T:g}-s window", "L_max", "ε held (reading)", "ε held a window", "rebuilds held",
                     "pass serving a GPU (GPUs)", "proving GPUs, fit / floor", "deployed unit, fit / floor"], rows))

    md.append(f"\n## Pass rate at each point's 1% openings rate (the line rule's, else ε only's), pooled over "
              f"{H:g} s\n")
    sweep = []
    for name, rates in (("32B x8", (1.0, 2.0, 4.0, 8.0, r32, 16.0)), ("8B x8", (8.0, 14.90, 29.81, 65.84, 100.0)),
                        ("8B x1", (14.90, 29.81, 65.84))):
        p, o = pts[name], chosen[name]["o"]
        for r in rates:
            k = max(1, math.ceil(r / o))
            d = o * k / r
            two, one = (line_at(p, o=o, d=d, k=k, r=r, theta=H, flight=f) for f in (2, 1))
            meas = [arm for arm in MEMSAT["arms"] if arm[0] == r] if name.startswith("8B") else []
            sweep.append({"point": name, "r": r, "o": o, "k": k, "d": d, **two, "one_in_flight": one,
                          **costs(p, prices, o=o, r=r),
                          "reads_over_R": r * p["blocks"] * BLOCK / timing.SPEC_BYTES_PER_S,
                          "measured": {"serving_share": meas[0][0] * meas[0][1], "gamma_spec": meas[0][2],
                                       "tokens_per_s": meas[0][3]} if meas else None})
    rows = [[x["point"], f"{x['r']:.4g}", x["k"], f"{x['d']:.3g}",
             f"{x['l_max_s'] * 1e3:.0f} / {x['one_in_flight']['l_max_s'] * 1e3:.0f} ms",
             pct(x["serving_share"]) + (f" ({pct(x['measured']['serving_share'])} measured)" if x["measured"] else ""),
             pct(x["reads_over_R"]) + (f" (1 − γ {pct(1 - x['measured']['gamma_spec'])})" if x["measured"] else ""),
             f"{pct(x['eps_held'])} / {pct(x['one_in_flight']['eps_held'])}",
             f"{pct(x['rebuild_held'])} / {pct(x['one_in_flight']['rebuild_held'])}", pct(x["chip"]),
             f"{x['credited_bytes'] / 1e9:.1f} / {x['one_in_flight']['credited_bytes'] / 1e9:.1f} GB"] for x in sweep]
    md.append(table(["point", "passes/s", "k", "draws a group", "L_max, 2 / 1 in flight", "pass serving a GPU",
                     "reads / R (≥ 1 − γ)", "ε held, pooled, 2 / 1", "rebuilds held, 2 / 1", "chip",
                     "weights credited, held over time, 2 / 1"], rows))

    md.append("\n## Serve-time commitments at each point's 1% region (the line rule's, else ε only's), modeled\n")
    serve = []
    for name, p in pts.items():
        s = chosen[name]
        tokens = p["tokens_per_s"] * (1 - s["serving_share"]) if p["codec_loss"] is not None else p["tokens_per_s"]
        row = {"point": name, "r": s["r"], "k": s["k"], "o": s["o"], "tokens_per_s": tokens,
               "store_bytes": p["blocks"] * BLOCK, "payload_leaves": p["blocks"],
               "receipt_bytes_per_s": s["r"] * p["blocks"] * 64, "receipt_leaves_per_s": s["r"] * p["blocks"] / 16,
               "pass_roots_per_s": s["r"], "coins_per_s": 2 * s["r"] / s["k"], "openings_per_s": s["o"]}
        serve.append(row)
    md.append(table(["point", "r", "k", "tokens/s (32B: offered; 8B: codec less passes)", "store, once",
                     "receipts a s (a token)",
                     "receipt leaves a s", "pass roots a s", "coins a s (32 B)", "openings a s (a token)"],
                    [[x["point"], f"{x['r']:.3g}", x["k"], f"{x['tokens_per_s']:,.0f}",
                      f"{x['store_bytes'] / 1e9:.1f} GB, {x['payload_leaves']:,} leaves",
                      f"{x['receipt_bytes_per_s'] / 1e9:.2f} GB "
                      f"({x['receipt_bytes_per_s'] / x['tokens_per_s'] / 1e6:.2f} MB)",
                      f"{x['receipt_leaves_per_s']:,.0f}", f"{x['pass_roots_per_s']:.3g}", f"{x['coins_per_s']:.3g}",
                      f"{x['openings_per_s']:.3g} ({x['openings_per_s'] / x['tokens_per_s']:.2g})"] for x in serve]))

    md.append("\n## Prices of an opening (GPU-s, a 512-unit statement)\n")
    md.append(table(["", "measured unit", "deployed unit (modeled)"],
                    [["ANDs", f"{prices['ands']:,}", f"{prices['deployed_ands']:,.0f}"],
                     ["one fit", f"{prices['fit']:.4f}", f"{prices['fit_deployed']:.4f}"],
                     ["device floor (firewall)", f"{prices['floor']:.4f}", f"{prices['floor_deployed']:.4f}"]]))

    out = {"recorded": recorded, "uniform_checks": uniform, "prices": prices, "points": pts, "region": region,
           "sweep": sweep, "serve": serve, "window_s": T, "horizon_s": H, "delta": audit.DELTA, "ln_inv_delta": LN}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sampling-inputs.json").write_text(json.dumps(out, indent=1, default=str))
    (a.out / "sampling-inputs.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
