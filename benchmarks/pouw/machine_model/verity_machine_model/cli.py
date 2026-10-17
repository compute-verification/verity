"""`machine-model`: re-solve a named price vector from the data and check every attack against it.

    machine-model list                        the named vectors: epoch, rows, credits and published τ
    machine-model solve [NAME] [--json]       re-solve NAME (default: model.toml's `current`): τ, τ at 600 W, U/F_low and
                                              margin against what was published, θ, the cheapest attacks, the check,
                                              then whether (π₃₂), PRICES-32's load half and π_c ≥ 16 hold
                                              (data/floors.csv), and every copy route's price and measured cost a word
    machine-model census [EPOCH] [--kind K]   each instruction class's u on each row at EPOCH (default: the last)
    machine-model attacks [NAME] [--design D] [-n N]
                                              the N cheapest attacks of each design (or of D) at NAME's prices
    machine-model adopt CENSUS EPOCH          merge a measured census (an opcode probe's census.csv) into data/ at EPOCH,
                                              the last epoch: each (kind, row) keeps the faster of its old and new line
    machine-model eps [DESIGN] [--q Q] [--fold-cost R] [--upfill] [--vector V] [--json]
                                              a design's ε budget (data/designs, default dither-duo) at q* = Q (default
                                              8): every term with its source, ε_machine's LP (honest's and the
                                              cheater's mix and binding rows), the ε(q*) curve for q* in [1, 16] and
                                              its q* threshold; --upfill adds filler keeping the uniform datapath busy
    machine-model eps --table [DESIGN ...] [--fold-sweep R,R,...] [--upfill]
                                              one line a design (every design by default; with --fold-sweep, each at
                                              each fold cost; with --upfill, each without and with it): γ, ε_machine
                                              at q* = 2, 4, 8, 16, ε at q* = 2 and 8, and the thresholds

`--data DIR` (before the command) reads another data directory, such as a copy with a candidate row. `solve` exits 1 when
the vector fails its check or doesn't reproduce a published number.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from verity_machine_model import attacks as atk
from verity_machine_model import eps as E
from verity_machine_model.adopt import adopt
from verity_machine_model.model import CLAIMS, DATA, census, model, vector, vectors
from verity_machine_model.resolve import resolve


def _list(args) -> int:
    print(f"{'vector':16s} {'epoch':16s} {'rows':>4s} {'credits':>7s} {'τ':>7s} {'margin':>6s}  summary")
    for n in vectors(args.data):
        v = vector(n, args.data)
        tag = f" (superseded by {v.superseded_by})" if v.superseded_by else ""
        print(f"{n:16s} {v.epoch:16s} {len(v.rows):4d} {len(v.credits):7d} {v.published.get('tau', float('nan')):7.4f} "
              f"{v.published.get('margin', float('nan')):6.3f}  {v.summary}{tag}")
    return 0


def _solve(args) -> int:
    name = args.name or model(args.data).current
    r = resolve(name, args.data)
    v, s = r.vector, r.solution
    bad = r.unreproduced()
    if args.json:
        print(json.dumps({"vector": name, "epoch": v.epoch, **r.numbers(), "tau_max": s.tau_max, "theta": s.theta,
                          "cheapest": r.cheapest, "three_piece": r.three_piece, "attacks": len(r.attacks),
                          "published": v.published, "unreproduced": bad, "failures": r.failures,
                          "floors": {k: {"kinds": n, "below": b, "absent": ab} for k, (n, b, ab) in r.floors.items()},
                          "copies": [{"kind": k, "words": w, "price": p, "measured": mc, "what": what}
                                     for k, w, p, mc, what in r.copies]},
                         ensure_ascii=False,
                         indent=1))
        return 1 if bad or r.failures else 0
    print(f"{name} (epoch {v.epoch}, {len(v.rows)} rows, {len(v.credits)} credits): {v.summary}")
    if v.superseded_by:
        print(f"  superseded by {v.superseded_by}")
    print(f"  {v.note}")
    print(f"  τ {r.tau:.4f}   τ at 600 W {r.tau_600w:.4f}   U/F_low {r.uf_low:.4f}   margin {r.margin:.3f}")
    if v.published:
        print("  published: " + "   ".join(f"{k} {x}" for k, x in v.published.items())
              + ("   (reproduced)" if not bad else ""))
    else:
        print("  published: nothing yet (record these under [published] in its vector file)")
    for b in bad:
        print(f"  NOT REPRODUCED: {b}")
    print(f"  τ-max {s.tau_max:.4f} at U/F_low {1 + s.infl_at_tau_max:.4f}")
    print("  θ: " + ", ".join(f"{k} {x:.3f}" for k, x in sorted(s.theta.items(), key=lambda t: -t[1]) if x > 5e-4))
    print(f"  cheapest in {r.model.adopted}: " + "; ".join(f"X {x:.3f} {fam}: {nm}" for x, fam, nm in r.cheapest)
          + (f"; three-piece (iv) X {r.three_piece:.3f}" if r.three_piece is not None else ""))
    fams = atk.families(r.attacks)
    print(f"  check: {len(r.census)} kinds at or below measured cost, ρ ≥ {r.model.rho} on the CUDA kinds, "
          f"{len(v.credits)} credits paid, {len(r.attacks)} attacks in {len(fams)} families at X ≥ {r.model.x_min}: "
          + ("pass" if not r.failures else f"FAIL ({len(r.failures)})"))
    for f in r.failures[:20]:
        print(f"    {f}")
    for claim, (n, below, absent) in r.floors.items():
        print(f"  {CLAIMS[claim]}: {n} kinds, " + ("holds" if not below else "fails: " + "; ".join(below))
              + (f" (not split out at {v.epoch}: {', '.join(absent)})" if absent else ""))
    print(textwrap.fill("copy routes, MACs a word (price/measured), cheapest first: "
                        + ", ".join(f"{k} {p:.2f}/{mc:.4g}" for k, _w, p, mc, _ in r.copies),
                        width=120, initial_indent="  ", subsequent_indent="    "))
    return 1 if bad or r.failures else 0


def _census(args) -> int:
    m = model(args.data)
    epoch = args.epoch or m.epochs[-1]
    if epoch not in m.epochs:
        raise SystemExit(f"no epoch {epoch!r}: one of {', '.join(m.epochs)}")
    c = census(m, epoch, args.data)
    for k in sorted(c):
        if args.kind and k != args.kind:
            continue
        print(f"{k}: " + ", ".join(f"{r} {u / m.cycle_u:.4g}" for r, u in c[k].items()) + "  (SMSP-cycles)")
    return 0


def _attacks(args) -> int:
    r = resolve(args.name or model(args.data).current, args.data)
    designs = [args.design] if args.design else sorted({a.design for a in r.attacks})
    for d in designs:
        xs, tp = atk.cheapest(r.census, r.solution, r.attacks, d, args.n)
        print(f"{d}:" + (f"  (three-piece (iv) X {tp:.3f})" if tp is not None else ""))
        for x, fam, nm in xs:
            print(f"  X {x:.3f}  {fam}: {nm}")
    return 0


def _adopt(args) -> int:
    out = adopt(args.census, args.epoch, args.data)
    m = model(args.data)
    prev = m.epochs[-2]
    print(f"{'kind':10s} {'row':14s} {'u at ' + prev:>14s} {'measured':>10s}")
    for a in out:
        before = f"{a.before:.1f}" if a.before is not None else "–"
        verdict = f"adopted from {args.epoch}" if a.adopted else "kept (the measurement is slower)"
        print(f"{a.kind:10s} {a.row:14s} {before:>14s} {a.measured:10.1f}  {verdict}")
    n = sum(a.adopted for a in out)
    print(f"{n} of {len(out)} lines adopted at {args.epoch}, {len(out) - n} kept")
    return 0


def _pct(x: float | None) -> str:
    return "unmeasured" if x is None else f"{100 * x:.3f}%"


def _thr(t: float | None) -> str:
    return "none ≤ 64" if t is None else f"{t:.3f}"


def _mix(m: E.Mix, per_mma: float) -> str:
    w = ", ".join(f"{r} {100 * max(m.words[r], 0.0):.1f}%" for r in E.ROUTES)
    return (f"{m.T * per_mma:.3f} SMSP-cycles a MMA ({w}; filler on UP {100 * max(m.filler_uniform, 0.0):.0f}%), "
            f"binding {', '.join(m.binding)}")


def _design(args, name: str) -> E.Design:
    d = E.design(name, args.data)
    d = d.with_fold_cost(args.fold_cost) if args.fold_cost is not None else d
    return d.with_upfill() if args.upfill else d


def _eps_record(d: E.Design, pi: E.Machine, q: float, data: Path) -> dict:
    e = E.eps(d, pi, q, data)
    per_mma = e.honest.words * e.honest.depth / d.L
    return {
        "design": d.name, "summary": d.summary, "vector": pi.vector, "honest": e.honest.name,
        "honest_status": e.honest.status, "L": d.L, "fold": d.fold_name, "fold_cost": d.fold_cost(),
        "filler_per_word": e.filler_per_word, "filler_route": d.filler.route, "q": q, "gamma": e.gamma,
        "as_built": {"cycles_per_mma": e.as_built.T * per_mma, "binding": e.as_built.binding},
        "baseline": {"cycles_per_mma": e.baseline.T * per_mma, "binding": e.baseline.binding,
                     "words": e.baseline.words, "program": d.baseline_name},
        "cheat": {"cycles_per_mma": e.cheat.T * per_mma, "binding": e.cheat.binding, "words": e.cheat.words,
                  "filler_uniform": e.cheat.filler_uniform},
        "measured_cycles_per_mma": e.T_honest_measured * per_mma if e.T_honest_measured else None,
        "terms": e.terms(), "machine_as_built": e.machine_as_built, "total": e.total(),
        "total_certified": e.total(sched=False),
        "sources": {"skip": {k: p.source for k, p in d.skip.items()}, "mand": {k: p.source for k, p in d.mand.items()},
                    "clock": d.clock.source, "luck": d.luck_source, "baseline": d.baseline_source},
        "curve": [[x, E.eps(d, pi, x, data).machine_eps] for x in E.QS],
        "threshold": E.threshold(d, pi, data), "threshold_lanes": E.threshold(d, pi, data, uniform=False),
        "uniform_at_16": E.eps(d, pi, 16, data, lanes=False).machine_eps,
    }


def _eps_table(args, pi: E.Machine) -> int:
    names = args.designs or E.designs(args.data)
    sweep = [float(x) for x in args.fold_sweep.split(",")] if args.fold_sweep else [None]
    qs = (2, 4, 8, 16)
    print(f"Π {pi.vector}; ε_machine at q* = {', '.join(map(str, qs))}; ε at q* = 2 and 8 (ε_sched 0: honest "
          f"certified at the model bound); q*₀ the least q* with ε_machine = 0 (all routes; lanes alone); the uniform "
          f"route alone at q* = 16")
    print(f"{'design':26s} {'fold':>6s} {'L':>3s} {'γ':>6s} {'T_H/MMA':>8s} {'binds':8s} {'skip':>6s} "
          + " ".join(f"{'m@' + str(q):>7s}" for q in qs) + f" {'ε@2':>7s} {'ε@8':>7s} {'q*₀':>9s} {'lanes':>6s} "
          f"{'UP@16':>6s}")
    rows = []
    for n in names:
        for r in sweep:
            d = E.design(n, args.data)
            d = d.with_fold_cost(r) if r is not None else d
            rows += [d, d.with_upfill()] if args.upfill else [d]
    for d in rows:
        es = {q: E.eps(d, pi, q, args.data) for q in qs}
        e = es[8]
        per_mma = e.honest.words * e.honest.depth / d.L
        t, tl = E.threshold(d, pi, args.data), E.threshold(d, pi, args.data, uniform=False)
        up = E.eps(d, pi, 16, args.data, lanes=False).machine_eps
        print(f"{d.name:26s} {d.fold_cost():6.3g} {d.L:3d} {e.gamma:6.3f} {e.baseline.T * per_mma:8.3f} "
              f"{e.baseline.binding[0]:8s} {100 * e.skip:5.2f}% "
              + " ".join(f"{100 * es[q].machine_eps:6.2f}%" for q in qs)
              + f" {100 * es[2].total(sched=False):6.2f}% {100 * e.total(sched=False):6.2f}% {_thr(t):>9s} "
              f"{_thr(tl):>6s} {100 * up:5.2f}%")
    return 0


def _eps(args) -> int:
    pi = E.machine(args.vector or "", args.data)
    if args.table:
        return _eps_table(args, pi)
    if len(args.designs) > 1:
        raise SystemExit("eps: one design, or --table for several")
    d = _design(args, args.designs[0] if args.designs else "dither-duo")
    rec = _eps_record(d, pi, args.q, args.data)
    if args.json:
        print(json.dumps(rec, ensure_ascii=False, indent=1))
        return 0
    e = E.eps(d, pi, args.q, args.data)
    h, per_mma = e.honest, e.honest.words * e.honest.depth / d.L
    filler = (f"{e.filler_per_word:.4g} a word on {d.filler.route}" if e.filler_per_word else "none")
    print(f"{d.name}: {d.summary}")
    print(f"  Π: {pi.vector}'s {len(pi.rows)} rows; honest {h.name} ({h.status}): {h.mma}, {h.words} words a MMA; "
          f"L {d.L}; fold {d.fold_name} {d.fold_cost():g} a word ({', '.join(f'{k} {v:g}' for k, v in d.fold.items())});"
          f" filler {filler}")
    print(f"  honest as built: {_mix(e.as_built, per_mma)}")
    prog = ", ".join(f"{k} {v:g}" for k, v in d.baseline.items())
    print(f"  baseline (tensor and {d.baseline_name or 'nothing'}{f' at {prog} a word' if prog else ''}): "
          f"{_mix(e.baseline, per_mma)}; γ {e.gamma:.3f}")
    if e.T_honest_measured:
        print(f"  measured: {e.T_honest_measured * per_mma:.3f} SMSP-cycles a MMA ({h.measured_source})")
    print(f"  cheater at q* = {args.q:g}: {_mix(e.cheat, per_mma)}")
    t = e.terms()
    print(f"  ε_sched    {_pct(t['sched'])}" + ("" if t["sched"] is not None else
                                                " (0: honest certified at the model bound, moving it into γ)"))
    for k, parts in (("skip", d.skip), ("mand", d.mand)):
        print(f"  ε_{k:9s}{_pct(t[k])}")
        for name, p in parts.items():
            print(f"      {name} {_pct(p.eps)}: {p.source}")
    print(f"  ε_machine  {_pct(t['machine'])} (against honest as built {_pct(e.machine_as_built)})")
    print(f"  ε_luck     {_pct(t['luck'])}: δ {d.delta:.3g}, p {d.p}, S {d.checks:g} ({d.luck_source})")
    print(f"  ε_clock    {_pct(t['clock'])}: {d.clock.source}")
    print(f"  ε          {_pct(e.total())} (first order {_pct(sum(v or 0.0 for v in t.values()))})")
    print(f"  q* threshold (ε_machine = 0): {_thr(rec['threshold'])}; lanes alone {_thr(rec['threshold_lanes'])}; "
          f"the uniform route alone at q* = 16: {_pct(rec['uniform_at_16'])}")
    print(textwrap.fill("ε_machine(q*): " + ", ".join(f"{x:g} {100 * y:.2f}%" for x, y in rec["curve"]), width=120,
                        initial_indent="  ", subsequent_indent="    "))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="machine-model", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA, help="the data directory (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list")
    s = sub.add_parser("solve")
    s.add_argument("name", nargs="?")
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("census")
    c.add_argument("epoch", nargs="?")
    c.add_argument("--kind")
    a = sub.add_parser("attacks")
    a.add_argument("name", nargs="?")
    a.add_argument("--design")
    a.add_argument("-n", type=int, default=3)
    d = sub.add_parser("adopt")
    d.add_argument("census", type=Path)
    d.add_argument("epoch")
    e = sub.add_parser("eps")
    e.add_argument("designs", nargs="*")
    e.add_argument("--q", type=float, default=8.0, help="q*, lane instructions a fresh word (default %(default)s)")
    e.add_argument("--fold-cost", type=float, help="scale the design's fold to R lane instructions a word")
    e.add_argument("--vector", help="the vector whose rows are Π (default: model.toml's current)")
    e.add_argument("--json", action="store_true")
    e.add_argument("--table", action="store_true")
    e.add_argument("--fold-sweep", help="with --table: each design at each of these fold costs")
    e.add_argument("--upfill", action="store_true",
                   help="add verified filler keeping the uniform datapath busy (with --table: a row beside each)")
    args = p.parse_args(argv)
    if args.cmd is None:
        print(__doc__)
        return 0
    return {"list": _list, "solve": _solve, "census": _census, "attacks": _attacks, "adopt": _adopt,
            "eps": _eps}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
