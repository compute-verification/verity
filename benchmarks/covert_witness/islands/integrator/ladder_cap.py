"""The scoring cap's ladder over the ladder fit's alpha range: at each held-out loss, the greatest forward gates per
token of the compute-optimal model under any alpha in the fit's 95% profile interval (coordinator 19:16Z: no attack is
dropped by a loose exponent).

    python islands/integrator/ladder_cap.py FIT.json --out CAP.json [--source ART] [--flips LEDGER_ROOT]

FIT.json is ladder-v3's (te-ladder/fit/v2, with its fit's residuals and alpha_95, its costs and bpb_per_nat). At each
alpha of ladder_report's grid inside alpha_95, E, A, B and beta are fitted again over the fit's points with alpha held
(ladder_report's least squares), and that fit's compute-optimal ladder is built as FIT.json's is (the same sizes,
forward power law and points a decade). CAP.json's points (`envelope`) are every alpha's points' losses, each at the
greatest forward any alpha's ladder has there, read log-linearly as te_core.policy's `ladder_fit` reads a ladder, and
the losses between them where the greatest passes from one ladder to another. Between two neighbouring points every
ladder is log-linear, so `ladder_fit`'s reading of CAP.json is never under any alpha's.
`--flips` lists the ledger's rows the cap reads (valid at the pin or the pilot, with a shape) whose eligibility under
some alpha differs from the fit's own.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

CW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CW))
_spec = importlib.util.spec_from_file_location("ladder_report", CW / "islands/ladder-v3/ladder_report.py")
LR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LR)

from te_core import frontier, policy  # noqa: E402

SCHEMA = "te-ladder/cap/v1"
#: bits per byte between a ladder's end and the point past it where the greatest drops
EDGE = 1e-9


def profile(points: list[dict], alpha: float) -> dict:
    """The best E, A, B >= 0 and beta on ladder_report's grid with alpha held, and its relative error."""
    n, d, y = (np.array([p[k] for p in points], dtype=float)
               for k in ("non_embedding_params", "tokens", "nats_per_token"))
    best = None
    for be in LR.GRID:
        cols = np.stack([np.ones_like(n), n ** -alpha, d ** -be], 1)
        coef = LR.nnls3(cols, y, 1 / y)
        if coef is None:
            continue
        err = float(((cols @ coef - y) / y) ** 2 @ np.ones_like(y))
        if best is None or err < best[0]:
            best = (err, be, coef)
    err, be, (e, a, b) = best
    return {"E": float(e), "A": float(a), "alpha": float(alpha), "B": float(b), "beta": float(be),
            "rms_rel_pct": 100 * math.sqrt(err / len(y))}


def ladder(fit: dict, f: dict) -> list[dict]:
    """f's compute-optimal points over FIT.json's sizes, as ladder_report's `compute_optimal` places them."""
    ns = sorted(fit["costs"]["sizes"].values())
    w = fit["costs"]["forward_macs_per_token"]
    out = []
    for x in np.linspace(math.log10(ns[0]), math.log10(ns[-1]),
                         max(2, round(LR.FIT_PER_DECADE * math.log10(ns[-1] / ns[0])) + 1)):
        n = 10 ** x
        nats, _ = LR.optimum(f, n)
        out.append({"loss": nats * fit["bpb_per_nat"],
                    "forward_gates_per_token": w["k"] * n ** w["g"] * fit["costs"]["gates_per_mac"]})
    return out


def read(points: list[dict]):
    """A ladder as te_core.policy's `ladder_fit` reads one, from its points."""
    pts = sorted((p["loss"], math.log(p["forward_gates_per_token"])) for p in points)

    def at(loss: float) -> float | None:
        for (l1, g1), (l2, g2) in zip(pts, pts[1:]):
            if l1 <= loss <= l2:
                return math.exp(g1 + (g2 - g1) * (loss - l1) / (l2 - l1)) if l2 > l1 else math.exp(max(g1, g2))
        return None
    return at


def envelope(ladders: list[list[dict]]) -> list[dict]:
    """The greatest of the ladders at every loss of any of them and wherever the greatest changes between two such
    losses (each ladder log-linear there). Where a ladder ends and the greatest drops, the end keeps the greater value
    and a point `EDGE` bpb past it the lesser, so a reading errs high only within `EDGE` of it, never low."""
    readers = [read(lad) for lad in ladders]
    xs = sorted({p["loss"] for lad in ladders for p in lad})

    def top(x: float) -> float:
        return max(v for r in readers if (v := r(x)) is not None)
    out = [(xs[0], top(xs[0]))]
    for x0, x1 in zip(xs, xs[1:]):
        lines = [(math.log(v0), math.log(v1)) for r in readers
                 if (v0 := r(x0)) is not None and (v1 := r(x1)) is not None]
        if not lines:
            raise SystemExit(f"ladder_cap: no alpha's ladder spans {x0:.6f} to {x1:.6f} bpb, so the cap has a gap")
        span, edge = x1 - x0, min(EDGE, (x1 - x0) / 4)

        def at(t: float) -> float:
            return max(c0 + (c1 - c0) * t for c0, c1 in lines)
        inner = [t for (a0, a1), (b0, b1) in itertools.combinations(lines, 2) if (a1 - a0) != (b1 - b0)
                 and 0 < (t := (b0 - a0) / ((a1 - a0) - (b1 - b0))) < 1 and a0 + (a1 - a0) * t >= at(t) - 1e-12]
        if at(0) < math.log(out[-1][1]) - 1e-12:
            inner.append(edge / span)
        if at(1) < math.log(top(x1)) - 1e-12:
            inner.append(1 - edge / span)
        out += [(x0 + span * t, math.exp(at(t))) for t in sorted(set(inner))] + [(x1, top(x1))]
    return [{"loss": x, "forward_gates_per_token": g} for x, g in out]


def cap(fit: dict, source: str | None = None) -> dict:
    f = fit["fit"]
    points = [r for r in f["residuals"] if r["in_fit"]]
    lo, hi = f["alpha_95"]
    alphas = [float(a) for a in LR.GRID if lo - 1e-9 <= a <= hi + 1e-9]
    profiles = {f"{a:.2f}": profile(points, a) for a in alphas}
    ladders = {k: ladder(fit, p) for k, p in profiles.items()}
    own = ladder(fit, f)
    off = max(abs(a["forward_gates_per_token"] / b["forward_gates_per_token"] - 1) + abs(a["loss"] - b["loss"])
              for a, b in zip(own, fit["points"]))
    if off > 1e-9:
        raise SystemExit(f"ladder_cap: the fit's own ladder built here is {off:.2e} off FIT.json's points; "
                         "ladder_report's compute_optimal changed, so build the ladders as it does")
    keys = ("E", "A", "alpha", "B", "beta", "rms_rel_pct", "points", "sizes", "alpha_95", "beta_95")
    return {"schema": SCHEMA, "points": envelope(list(ladders.values())), "source": source,
            "fit": {k: f[k] for k in keys}, "profiles": profiles, "ladders": ladders,
            "rule": "points: the greatest forward gates per token of the compute-optimal model under any alpha of the "
                    "fit's 95% profile interval (alpha_95, on ladder_report's grid; E, A, B and beta fitted again with "
                    "alpha held: profiles; each alpha's ladder built as FIT.json's is: ladders), at each loss (bits per "
                    "byte on the search pool) of any alpha's ladder and where the greatest passes from one to another, "
                    f"and {EDGE:g} bpb past a ladder's end where it drops. te_core.policy's ladder_fit reads points; "
                    "between two of them every alpha's ladder is log-linear, so that reading is never under any's"}


def flips(cap_doc: dict, fit: dict, root: Path) -> dict:
    """The rows the cap reads whose eligibility under some alpha's ladder differs from the fit's own; the counts."""
    with open(root / "index.jsonl") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    v3 = set(frontier.pinned())
    shapes = policy.starts(policy.MANIFEST)
    own = read(fit["points"])
    readers = {k: read(lad) for k, lad in cap_doc["ladders"].items()}
    out, n, over, uncounted = [], 0, 0, 0
    for r in frontier.valid(rows, frontier.pinned(pilot=True)):
        shape = shapes.get(r["start"])
        if shape is None or (r.get("data") == "v3" and shape.get("tokenizer") != policy.V3_TOKENIZER):
            continue
        n += 1
        w = (r.get("model") or {}).get("forward_gates_per_token")
        sw = r.get("start_forward_gates_per_token") or policy.w_fwd(shape)
        uncounted += w is None
        over += w is not None and w > policy.CAP_FALLBACK * sw
        base = policy.eligible(w, sw, r.get("val_bpb"), own)
        at = {k: policy.eligible(w, sw, r.get("val_bpb"), lad) for k, lad in readers.items()}
        if any(v != base for v in at.values()):
            out.append({"id": r["id"], "run_id": r.get("run_id"), "step": r.get("checkpoint_step"), "start": r["start"],
                        "pin": "v3" if r.get("measurement_commit") in v3 else "pilot", "val_bpb": r.get("val_bpb"),
                        "forward_over_start": w / sw, "under_fit": base,
                        "eligible_at_alpha": sorted(k for k, v in at.items() if v)})
    return {"rows": n, "over_twice_the_start": over, "uncounted": uncounted, "flips": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fit", type=Path, help="ladder-v3's FIT.json (te-ladder/fit/v2)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default=None, help="where FIT.json comes from (an art id), recorded in CAP.json")
    ap.add_argument("--flips", type=Path, default=None, metavar="LEDGER_ROOT",
                    help="te_core's ledger root, whose index.jsonl's rows are checked (writes FLIPS.json beside --out)")
    a = ap.parse_args(argv)
    fit = json.loads(a.fit.read_text())
    doc = cap(fit, a.source)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    env, own = doc["points"], read(fit["points"])
    print(f"{len(doc['profiles'])} alphas {min(doc['profiles'])} to {max(doc['profiles'])}; cap ladder {len(env)} "
          f"points, {env[0]['loss']:.4f} to {env[-1]['loss']:.4f} bpb (the fit's own {fit['points'][-1]['loss']:.4f} "
          f"to {fit['points'][0]['loss']:.4f})")
    for x in (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3):
        c, o = read(env)(x), own(x)
        print(f"  at {x:.2f} bpb: cap {c:.3e}" if c else f"  at {x:.2f} bpb: cap -", f"fit {o:.3e}" if o else "fit -")
    if a.flips:
        got = flips(doc, fit, a.flips)
        (a.out.parent / "FLIPS.json").write_text(json.dumps(got, indent=1) + "\n")
        print(f"rows the cap reads: {got['rows']}, forward over twice the start's {got['over_twice_the_start']}, "
              f"uncounted {got['uncounted']}; eligibility flips across alpha: {len(got['flips'])}")
        for t in got["flips"]:
            print(f"  {t['id']} ({t['pin']}, {t['start']}, {t['run_id']}@{t['step']}): {t['val_bpb']:.5f} bpb, "
                  f"{t['forward_over_start']:.2f}x the start's forward, under the fit {t['under_fit']}, "
                  f"eligible at alpha {', '.join(t['eligible_at_alpha']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
