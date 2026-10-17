"""`machine-model`: re-solve a named price vector from the data and check every attack against it.

    machine-model list                        the named vectors: epoch, rows, credits and published τ
    machine-model solve [NAME] [--json]       re-solve NAME (default: model.toml's `current`): τ, τ at 600 W, U/F_low and
                                              margin against what was published, θ, the cheapest attacks, the check,
                                              then whether (π₃₂), PRICES-32's load half and π_c ≥ 16 hold
                                              (data/floors.csv), and every copy route's price and measured cost a word
    machine-model census [EPOCH] [--kind K]   each instruction class's u on each row at EPOCH (default: the last)
    machine-model attacks [NAME] [--design D] [-n N]
                                              the N cheapest attacks of each design (or of D) at NAME's prices
    machine-model adopt CENSUS EPOCH          merge a measured census (an opcode probe's census.csv) at EPOCH, the last
                                              epoch, into data/census/EPOCH.csv: each (kind, row) keeps its faster line
    machine-model designs [--json]            the design registry (data/designs): each design's name, its aliases, the
                                              noise on each product, fold, filler, proof unit, coins and perplexity arm
    machine-model eps [DESIGN] [--device sm_NN] [--q Q] [--q-uniform QU] [--operands O] [--credit C]
                      [--fold-cost R] [--fill X [--fill-mix K=V,...]] [--lookups [--lookup-q QT] [--lookup-bytes B]
                      [--l2 B] [--index-bits B]] [--uni-issue C] [--checking full|sampled [--adapt UNIT]] [--vector V]
                      [--json]
                                              a design's ε budget (a name or alias in data/designs, default
                                              dither-1s) at q* = Q (default 8): every term with its source,
                                              ε_machine's LP (the credited and the cheater's mix, binding rows and
                                              certificate π*), the ε(q*) curve for q* in [0.5, 16] (and for q*_U at
                                              q* = Q) and the q* at which ε_machine reaches its floor; q*_U is q*
                                              unless given; the cheater's operand delivery is free (default), at the
                                              noise's floor, or honest's; ε is a fraction of honest's kernel (--credit
                                              H, default) or of the floor program ♭ (--credit floor); --fill adds X
                                              lane instructions a word of verified filler on lanes (LOP3 and IMAD
                                              alike, or --fill-mix's class weights); --lookups opens the table route (data/table.toml: lane instructions and L2 bytes a word
                                              against an L2 facet; --lookup-q, --lookup-bytes, --l2; --index-bits B
                                              for B-bit noise entries on the operand it indexes); --uni-issue C also
                                              charges a uniform instruction C times an ALU's cost on the issue rows
                                              (a no-op where the census has it there already); --checking full
                                              (default) is the protocol with every word and operand verified, the
                                              core ε alone, and --checking sampled adds ε_sample = ε_luck + ε_adapt
                                              on its own line, for the design's proof unit (data/adapt.toml) or
                                              --adapt UNIT's
    machine-model eps --table [DESIGN ...] [--fold-sweep R,R,...] [--fill X [--fill-mix K=V,...]] [--operands O]
                      [--credit C] [--q Q] [--lookups ...] [--uni-issue C] [--checking full|sampled [--adapt UNIT]]
                                              one line a design (every design by default; with --fold-sweep, each at
                                              each fold cost), in two column groups: the core (γ and γ_f, γ with the
                                              clock gap f_max/f_H, the credited time, ε_skip, ε_sched, ε_machine at
                                              q* = 2, 4, 8, 16 and its floor, the core ε at q* = 2, 8 and Q, and the
                                              thresholds), then with --checking sampled the sample (ε_luck, ε_adapt
                                              and ε_sample at Q)
    machine-model redteam [DESIGN ...] [--device sm_NN] [--q Q] [--operands O] [--credit C]
                                              the red-team audit (redteam.py): each design's binding and near-binding
                                              facets, the decision-76 move in the cheater's favour at which the core
                                              first rises (per row), and, where Π prices no uniform datapath, the ε a
                                              sibling-rate uniform route would open (default q* = 1.22, credit floor)
    machine-model margin [DESIGN ...] [--device sm_NN] [--m M] [--least [--mix FFMA]] [--q Q] [--operands O] [--credit C]
                                              decision 76's margin (redteam.margin_audit): each design's core as built,
                                              its worst core with every near measured facet's lane coefficient moved M
                                              in the cheater's favour at once (default 2%), the single facet that drives
                                              it, and each near facet's own move; with --least, the cheapest filler a
                                              word (in --mix, default the design's) that brings the worst back to the
                                              floor (default q* = 1.22, credit floor)
    machine-model capacity GROUP [GROUP ...] [--vector V] [--json]
                                              the certified share σ of a window's capacity (capacity.py): each GROUP
                                              is [WEIGHT/]TERM+TERM..., its share of the cell's SMSP-time (default 1)
                                              and its certified load, TERM one of FOLD:SHARE[*R] (a noise-GEMM filler
                                              at SHARE of the tensor peak, its words folded R times; FOLD duo, xin13,
                                              li, arx (a cost probe) or none), lane:RATE (verified lane chains, warp instructions a
                                              SMSP-cycle), served:SHARE[:FOLD] or CLASS=RATE; σ is read on the pooled
                                              load, with each group's placement-attested σ beside it

`--device sm_NN` (before the command, or on `eps`) reads that device's directory, `data/devices/sm_NN/` (README
"Devices"); sm_120, `data/` itself, is the default. `--data DIR` (before the command) reads another data directory, such
as a copy with a candidate row. `solve` exits 1 when the vector fails its check or doesn't reproduce a published number.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

from verity_machine_model import attacks as atk
from verity_machine_model import capacity as C
from verity_machine_model import eps as E
from verity_machine_model.adopt import adopt
from verity_machine_model.model import CLAIMS, DATA, census, device, devices, model, vector, vectors
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
    w = ", ".join(f"{r} {100 * max(m.words[r], 0.0) + 0.0:.1f}%" for r in E.ROUTES if r != "table" or m.words[r])
    if m.forms:
        w += "; tensor on " + ", ".join(f"{k} {100 * x:.1f}%" for k, x in sorted(m.forms.items(), key=lambda t: -t[1]))
    up = "".join(f"; filler {j + 1} on UP {100 * max(x, 0.0) + 0.0:.0f}%" for j, x in enumerate(m.filler_uniform))
    return (f"{m.T * per_mma:.3f} SMSP-cycles a MMA ({w}{up}), "
            f"binding {', '.join(m.binding)}")


def _pi(m: E.Mix) -> str:
    return ", ".join(f"{r} {w:.3f}" for r, w in sorted(m.pi.items(), key=lambda t: -t[1]))


def _design(args, name: str) -> E.Design:
    d = E.design(name, args.data)
    d = d.with_fold_cost(args.fold_cost) if args.fold_cost is not None else d
    d = d.with_fill(args.fill, _fill_mix(args)) if args.fill else d
    d = d.with_index_bits(args.index_bits) if args.index_bits else d
    return d if args.q_uniform is None else replace(d, name=f"{d.name}@qu{args.q_uniform:g}", q_uniform=args.q_uniform)


def _l2(tb: E.Table) -> str:
    return "L2 unbounded: unmeasured on this device" if tb.l2 is None else f"L2 {tb.l2 / 1e12:g} TB/s"


def _fill_mix(args) -> dict[str, float] | None:
    if args.fill_mix is None:
        return None
    if not args.fill:
        raise SystemExit("eps: --fill-mix is the --fill filler's mix: add --fill")
    try:
        mix = {k.strip(): float(v) for k, v in (kv.split("=") for kv in args.fill_mix.split(","))}
    except ValueError:
        raise SystemExit(f"eps: --fill-mix {args.fill_mix!r}: write CLASS=WEIGHT,... (ALU=1,IMAD=1)") from None
    if not mix or any(v < 0 for v in mix.values()) or not sum(mix.values()):
        raise SystemExit(f"eps: --fill-mix {args.fill_mix!r}: weights ≥ 0, not all 0")
    return mix


def _lookups(args) -> E.Table | None:
    if not args.lookups and (args.lookup_q is not None or args.l2 is not None or args.lookup_bytes is not None):
        raise SystemExit("eps: --lookup-q, --lookup-bytes and --l2 price the table route: add --lookups")
    return E.table(args.data, args.lookup_q, args.l2, args.lookup_bytes) if args.lookups else None


def _adapt(args, d: E.Design) -> E.Adapt | None:
    if args.checking == "full":
        if args.adapt:
            raise SystemExit("eps: --adapt picks the sampled protocol's proof unit: add --checking sampled")
        return None
    return E.adapt(args.adapt or d.unit, args.data)


def _registry(d: E.Design) -> dict:
    return {"name": d.name, "title": d.title, "aliases": list(d.aliases), "base": d.base or None,
            "summary": d.summary, "noise": d.noise, "fold": {"name": d.fold_name, **d.fold},
            "filler": [{"per_word": f.per_word, "mix": f.mix, "route": f.route} for f in d.filler], "L": d.L,
            "unit": d.unit, "coins": {"bits": d.coin_bits, "source": d.coins_source}, "ppl": d.ppl or None}


def _designs(args) -> int:
    ds = [E.design(n, args.data) for n in E.designs(args.data)]
    if args.json:
        print(json.dumps([_registry(d) for d in ds], ensure_ascii=False, indent=1))
        return 0
    print(f"{'design':15s} {'title':22s} {'linears':>9s} {'lm_head':>9s} {'QKᵀ':>9s} {'P·V':>5s} {'fold':>9s} "
          f"{'filler':>6s} {'unit':>6s} {'coins':>5s} {'ppl':14s} aliases")
    for d in ds:
        n = d.noise
        print(f"{d.name:15s} {d.title:22s} {n['linears']:>9s} {n['lm_head']:>9s} {n['qk']:>9s} {n['pv']:>5s} "
              f"{d.fold_name + ' ' + f'{d.fold_cost():g}':>9s} {sum(f.per_word for f in d.filler):6g} {d.unit:>6s} "
              f"{d.coin_bits:5d} {d.ppl or '–':14s} {', '.join(d.aliases)}")
    print(textwrap.fill("noise: " + "; ".join(f"{k}: {v}" for k, v in E.NOISE.items()), width=120,
                        subsequent_indent="  "))
    return 0


def _eps_record(d: E.Design, pi: E.Machine, q: float, data: Path, operands: str, credit: str = "H",
                table: E.Table | None = None, adapt: E.Adapt | None = None, checking: str = "full") -> dict:
    e = E.eps(d, pi, q, data, operands=operands, credit=credit, table=table, adapt=adapt)
    sampled, dt = checking == "sampled", d.table(table)
    s = E.setup(d, pi, data, operands, credit, table)
    per_mma = e.honest.words * e.honest.depth / d.L
    return {
        "design": d.name, "summary": d.summary, "registry": _registry(d), "vector": pi.vector, "honest": e.honest.name,
        "honest_status": e.honest.status, "operands": operands, "credit": credit, "L": d.L, "fold": d.fold_name,
        "fold_cost": d.fold_cost(), "fillers": [{"per_word": pw, "route": f.route, "mix": f.mix, "source": f.source,
                                                 "p": f.p, "luck": f.luck} for f, pw in e.fillers], "q": q,
        "operand_floor": d.floor, "gamma": e.gamma, "gamma_clock": e.gamma_clock,
        "honest_cycles_per_mma": e.T_honest * per_mma,
        "honest_time": e.T_honest_source, "credit_ratio": e.credit_ratio,
        "as_built": {"cycles_per_mma": e.as_built.T * per_mma, "binding": e.as_built.binding},
        "baseline": {"cycles_per_mma": e.baseline.T * per_mma, "binding": e.baseline.binding,
                     "words": e.baseline.words, "program": d.baseline_name},
        "cheat": {"cycles_per_mma": e.cheat.T * per_mma, "binding": e.cheat.binding, "words": e.cheat.words,
                  "forms": e.cheat.forms, "filler_uniform": e.cheat.filler_uniform, "pi": e.cheat.pi},
        "honest_form": e.honest.form,
        "lucky_cycles_per_mma": e.lucky.T * per_mma, "deficit": e.deficit, "lucky_short": e.lucky.short,
        "measured_cycles_per_mma": e.T_honest_measured * per_mma if e.T_honest_measured else None,
        "checking": checking, "terms": e.terms(), "cert": e.cert, "route": e.route, "floor": e.floor,
        "machine_as_built": e.machine_as_built, "core": e.core(), "core_certified": e.core(sched=False),
        "sample": None if not sampled else {
            "eps": e.sample(), "luck": e.luck, "adapt": adapt.eps if adapt else 0.0,
            "unit": adapt.name if adapt else None, "selection": adapt.selection if adapt else None,
            "budget": {"checks": d.checks, "delta": d.delta, "p": d.p}},
        "sources": {"skip": {k: p.source for k, p in d.skip.items()}, "mand": {k: p.source for k, p in d.mand.items()},
                    "clock": d.clock.source, "luck": d.luck_source, "baseline": d.baseline_source,
                    "operands": d.floor_source if operands == "floor" else operands},
        "index_bits": d.index_bits,
        "lookups": None if dt is None else {"q": dt.q, "bytes": dt.bytes, "l2": dt.l2,
                                            "bytes_per_smsp_cycle": dt.per_cycle(), "source": dt.source},
        "data": str(data),
        "adapt": None if adapt is None else {"unit": adapt.name, "eps": adapt.eps, "source": adapt.source,
                                             "selection": adapt.selection},
        "curve": [[x, s.machine_eps(x)] for x in E.QS],
        "curve_uniform": [[x, s.machine_eps(q, q_uniform=x)] for x in E.QS],
        "threshold": E.threshold(s), "threshold_lanes": E.threshold(s, uniform=False),
        "uniform_at_16": s.machine_eps(16, lanes=False),
    }


def _eps_table(args, pi: E.Machine) -> int:
    names = args.designs or E.designs(args.data)
    sweep = [float(x) for x in args.fold_sweep.split(",")] if args.fold_sweep else [None]
    qs = (2, 4, 8, 16)
    qe = sorted({2.0, 8.0, args.q})
    cred = {"H": "honest's kernel (H)", "model": "honest's model bound B(H)",
            "floor": "the floor program ♭ (SPEC §0.7a)"}[args.credit]
    tb = _lookups(args)
    sampled = args.checking == "sampled"
    look = "" if tb is None else (f", the table route open at {tb.q:g} lane instructions and {tb.bytes:g} B of L2 a "
                                  f"word ({_l2(tb)})")
    unit = "" if not sampled else (f" Sample (the sampling compiler, never in core): the proof unit ("
                                   f"{f'{args.adapt} for every design' if args.adapt else 'each design its own'}), "
                                   f"ε_luck at q* = {args.q:g}, ε_adapt, ε_sample = their sum.")
    print(f"Π {pi.vector}, operands {args.operands}, credit {cred}{look}, checking {args.checking}. Core (every word "
          f"and operand verified): γ_f γ with the clock gap f_max/f_H; T/MMA the credited time and its binding rows, "
          f"H/cred honest's time over it; ε_skip; "
          f"ε_sched; ε_machine at q* = {', '.join(map(str, qs))} and its floor (the lane and uniform routes closed); "
          f"core ε at q* = {', '.join(f'{q:g}' for q in qe)}; q*₀ the least q* at which ε_machine reaches its floor "
          f"(all routes; lanes alone); the uniform route alone at q* = 16.{unit}")
    print(f"{'design':26s} {'fold':>6s} {'L':>3s} {'γ':>6s} {'γ_f':>6s} {'T/MMA':>7s} {'binds':12s} {'H/cred':>6s} "
          f"{'skip':>7s} "
          f"{'sched':>6s} " + " ".join(f"{'m@' + str(q):>7s}" for q in qs)
          + f" {'floor':>7s} " + " ".join(f"{f'core@{q:g}':>9s}" for q in qe)
          + f" {'q*₀':>9s} {'lanes':>6s} {'UP@16':>6s}"
          + (f" | {'unit':>15s} {'luck':>7s} {'adapt':>7s} {'sample':>7s}" if sampled else ""))
    rows = []
    for n in names:
        for r in sweep:
            d = E.design(n, args.data)
            d = d.with_fold_cost(r) if r is not None else d
            d = d.with_fill(args.fill, _fill_mix(args)) if args.fill else d
            rows.append(d.with_index_bits(args.index_bits) if args.index_bits else d)
    for d in rows:
        ad = _adapt(args, d)
        es = {q: E.eps(d, pi, q, args.data, operands=args.operands, credit=args.credit, table=tb, adapt=ad)
              for q in {*qs, *qe}}
        e = es[8]
        s = E.setup(d, pi, args.data, args.operands, args.credit, tb)
        per_mma = e.honest.words * e.honest.depth / d.L
        t, tl = E.threshold(s), E.threshold(s, uniform=False)
        up = s.machine_eps(16, lanes=False)
        sched = "–" if e.sched is None else f"{100 * e.sched:5.1f}%"
        eq = es[args.q]
        print(f"{d.name:26s} {d.fold_cost():6.3g} {d.L:3d} {e.gamma:6.3f} {e.gamma_clock:6.3f} "
              f"{e.baseline.T * per_mma:7.3f} "
              f"{','.join(e.baseline.binding)[:12]:12s} {e.credit_ratio:6.3f} {100 * e.skip:6.3f}% {sched:>6s} "
              + " ".join(f"{100 * es[q].machine_eps:6.2f}%" for q in qs)
              + f" {100 * e.floor:6.2f}% "
              + " ".join(f"{100 * es[q].core():8.3f}%" for q in qe)
              + f" {_thr(t):>9s} {_thr(tl):>6s} {100 * up:5.2f}%"
              + (f" | {ad.name:>15s} {100 * eq.luck:6.3f}% {100 * ad.eps:6.3f}% {100 * eq.sample():6.3f}%"
                 if ad else ""))
    return 0


def _eps(args) -> int:
    pi = E.machine(args.vector or "", args.data)
    if args.uni_issue is not None and E.uniform_issue_charged(pi):
        print(f"eps: {pi.vector}'s census charges a uniform instruction on the issue rows already; --uni-issue "
              f"{args.uni_issue:g} changes nothing", file=sys.stderr)
    pi = pi if args.uni_issue is None else E.uniform_on_issue(pi, args.uni_issue)
    _fill_mix(args)
    if args.table:
        return _eps_table(args, pi)
    if len(args.designs) > 1:
        raise SystemExit("eps: one design, or --table for several")
    names = E.designs(args.data)
    d = _design(args, args.designs[0] if args.designs else "dither-1s" if "dither-1s" in names else names[0])
    tb = _lookups(args)
    ad = _adapt(args, d)
    rec = _eps_record(d, pi, args.q, args.data, args.operands, args.credit, tb, ad, args.checking)
    if args.json:
        print(json.dumps(rec, ensure_ascii=False, indent=1))
        return 0
    e = E.eps(d, pi, args.q, args.data, operands=args.operands, credit=args.credit, table=tb, adapt=ad)
    h, per_mma = e.honest, e.honest.words * e.honest.depth / d.L
    filler = ", ".join(f"{pw:.4g} a word on {f.route}" for f, pw in e.fillers) or "none"
    floor = ", ".join(f"{k} {v:g}" for k, v in d.floor.items())
    ops = {"free": "free (SPEC's conservative default)", "honest": "at honest's",
           "floor": f"at the noise's floor φ_del, {floor} a MMA ({d.floor_source})"}
    built = {"H": "honest as built", "model": "honest as built",
             "floor": f"the floor program ♭ (the tensor route at φ_del = {floor} a MMA)"}
    print(f"{d.name}: {d.summary}")
    print(f"  {d.title}" + (f" (also {', '.join(d.aliases)})" if d.aliases else "") + ": noise "
          + ", ".join(f"{k} {v}" for k, v in d.noise.items()) + f"; proof unit {d.unit}; {d.coin_bits}-bit coins; "
          + (f"perplexity arm {d.ppl}" if d.ppl else "no perplexity arm on main"))
    print(f"  Π: {pi.vector}'s {len(pi.rows)} rows; honest {h.name} ({h.status}): {h.mma} ({h.form}), {h.words} words a "
          f"MMA; the cheater's tensor route on any of {max(1, sum(f.kind in pi.c for f in E.forms(args.data)))} MMA "
          "forms; "
          f"L {d.L}; fold {d.fold_name} {d.fold_cost():g} a word ({', '.join(f'{k} {v:g}' for k, v in d.fold.items())});"
          f" filler {filler}")
    print(f"  {built[args.credit]}: {_mix(e.as_built, per_mma)}")
    prog = ", ".join(f"{k} {v:g}" for k, v in d.baseline.items())
    print(f"  credited (tensor and {d.baseline_name or 'nothing'}{f' at {prog} a word' if prog else ''}): "
          f"{_mix(e.baseline, per_mma)}")
    print(f"  honest: {e.T_honest * per_mma:.3f} SMSP-cycles a MMA ({e.T_honest_source}"
          + (f", {h.measured_source}" if e.T_honest_source != "model" else "") + f"); γ {e.gamma:.3f}, "
          f"{e.gamma_clock:.3f} with the clock gap"
          + (f"; honest over the credit at f_max {e.credit_ratio:.3f}, in γ" if args.credit != "H" else ""))
    print(f"  cheater's operand delivery: {ops[args.operands]}")
    if tb is not None:
        dt = d.table(tb)
        per = "" if dt.l2 is None else f" = {dt.per_cycle():.3f} B a SMSP-cycle"
        print(f"  table route: {dt.q:g} lane instructions and {dt.bytes:g} B of L2 a word ({d.index_bits}-bit entries "
              f"on the indexed operand), {_l2(dt)}{per} ({dt.source})")
    print(f"  cheater at q* = {args.q:g}: {_mix(e.cheat, per_mma)}")
    print(f"  certificate π*: {_pi(e.cheat)}")
    t = e.terms()
    print(f"  ε_sched    {_pct(t['sched'])}" + (
        "" if t["sched"] is not None else " (0: credited at the floor program, moving it into γ)"
        if args.credit == "floor" else " (0: honest credited at the model bound, moving it into γ)"))
    for k, parts in (("skip", d.skip), ("mand", d.mand)):
        print(f"  ε_{k:9s}{_pct(t[k])}")
        for name, p in parts.items():
            print(f"      {name} {_pct(p.eps)}: {p.source}")
    print(f"  ε_machine  {_pct(t['machine'])} = ε_cert(π*) {_pct(e.cert)} + ε_route(π*) {_pct(e.route)}; floor "
          f"{_pct(e.floor)} (the lane and uniform routes closed); against honest as built {_pct(e.machine_as_built)}")
    print(f"  ε_clock    {_pct(t['clock'])}: " + (d.clock.source if args.credit == "H" else "credited at f_max, in γ"))
    core = ("skip", "mand", "machine", "sched", "clock")
    print(f"  ε core     {_pct(e.core())} (100% checking, every word and operand verified; first order "
          f"{_pct(sum(t[k] or 0.0 for k in core))})")
    if args.checking == "sampled":
        short = ", ".join(f"{k} {100 * v:.4f}%" for k, v in e.lucky.short.items())
        print(f"  ε_sample   {_pct(e.sample())} (the sampling compiler's, never in the core ε) = ε_luck + ε_adapt")
        print(f"      ε_luck  {_pct(t['luck'])}: the lucky cheater leaves undone {short} (words alone: d̄ "
              f"{100 * e.deficit:.4f}%; budget δ {d.delta:.3g}, p {d.p} a word"
              + "".join(f", {f.p} filler {j + 1}" for j, (f, _) in enumerate(e.fillers)) + f", S {d.checks:g}: "
              f"{d.luck_source})")
        print(f"      ε_adapt {_pct(t['adapt'])}: proof unit {ad.name}, {ad.selection}: {ad.source}")
    print(f"  q* threshold (ε_machine = its floor): {_thr(rec['threshold'])}; lanes alone "
          f"{_thr(rec['threshold_lanes'])}; the uniform route alone at q* = 16: {_pct(rec['uniform_at_16'])}")
    print(textwrap.fill("ε_machine(q*): " + ", ".join(f"{x:g} {100 * y:.2f}%" for x, y in rec["curve"]), width=120,
                        initial_indent="  ", subsequent_indent="    "))
    print(textwrap.fill(f"ε_machine(q*_U) at q* = {args.q:g}: "
                        + ", ".join(f"{x:g} {100 * y:.2f}%" for x, y in rec["curve_uniform"]), width=120,
                        initial_indent="  ", subsequent_indent="    "))
    return 0


def _capacity(args) -> int:
    pi = E.machine(args.vector or "", args.data)
    groups = []
    for g in args.groups:
        w, _, spec = g.rpartition("/")
        groups.append((float(w) if w else 1.0, C.parse(spec)))
    x = C.window(pi, groups)
    if args.json:
        print(json.dumps(x.record(), ensure_ascii=False, indent=1))
        return 0
    p = x.pooled
    print(f"Π: {pi.vector}; the empty GPU's useful capacity {x.empty:.5f} tensor slots a SMSP-cycle "
          f"({x.empty * C.TENSOR_CYCLES:.3f}× the tensor peak)")
    for (w, g), h in zip(x.groups, x.parts):
        print(f"  {w:.4g} of the cell: {g.name}: σ {1 - h.hidden / x.empty:.4f} placed, B {h.credit():.4f}, tensor "
              f"{g.tensor_share():.3f}" + (f", OVER on {', '.join(h.over)}" if h.over else ""))
    print(f"  pooled load: " + ", ".join(f"{k} {v:.4g}" for k, v in sorted(p.work.n.items())) + " warp instructions a "
          f"SMSP-cycle; rows " + ", ".join(f"{r} {v:.3f}" for r, v in sorted(p.load.items(), key=lambda kv: -kv[1])[:5]))
    if p.over:
        print(f"  OVER on {', '.join(p.over)}: the certified load alone exceeds Π (a miscount or a model violation)")
    print(f"  hidden: {p.hidden:.5f} slots a SMSP-cycle, " + (", ".join(f"{k} {v:.4g}" for k, v in p.mix.items())
                                                             or "nothing") +
          f"; binding {', '.join(p.binding) or 'none'}")
    print(f"  σ {x.sigma():.4f}  (ε_cap {x.eps_cap():.4f}; placement-attested {x.sigma_placed():.4f}; time credit B "
          f"{x.credit():.4f}; tensor share {x.tensor():.4f})")
    return 0


def _redteam(args) -> int:
    from verity_machine_model import redteam as RT

    names = args.designs or E.designs(args.data)
    for name in names:
        a = RT.audit(name, args.data, q=args.q, operands=args.operands, credit=args.credit)
        print(f"{a.design} on {a.device}: core {_pct(a.core)}  ε_machine {_pct(a.machine_eps)}  floor {_pct(a.floor)}  "
              f"γ {a.gamma:.3f}  binds {', '.join(a.binding)}" + ("" if a.uniform_route else "  [Π has no uniform route]"))
        for r in a.near:
            if r.ceiling:
                how = "the issue ceiling, silicon cannot beat it"
            elif r.flip_pct is None:
                how = "no flip ≤ 10%"
            else:
                how = f"flips at {r.flip_pct}% → {_pct(r.core_at_flip)} (at 2%: {_pct(r.core_at_2pct)})"
            print(f"    {r.name:12s} load {r.load:.3f}  dual {r.dual:.3f}: {how}")
        if not a.uniform_route:
            base, opened, binding = RT.uniform_gap(name, args.data, q=args.q, operands=args.operands, credit=args.credit)
            print(f"    uniform-datapath gap: core {_pct(base)} → {_pct(opened)} with a sibling-rate uniform route "
                  f"(binds {', '.join(binding) or 'none'})")
    return 0


def _margin(args) -> int:
    from verity_machine_model import redteam as RT

    mix = {c: 1.0 for c in args.mix.split(",")} if args.mix else None
    for name in args.designs or E.designs(args.data):
        mg = RT.margin_audit(name, args.data, m=args.m, q=args.q, operands=args.operands, credit=args.credit)
        drive = f"driven by {mg.driver}" if mg.driver else "no near facet drives it"
        print(f"{mg.design} on {mg.device}: core {_pct(mg.base)}  worst at {mg.m*100:g}% {_pct(mg.worst)}  ({drive})")
        for r, s in mg.rows:
            print(f"    {r:12s} alone → {_pct(s)}")
        if args.least:
            lf = RT.least_filler(name, args.data, mix=mix, m=args.m, q=args.q, operands=args.operands,
                                 credit=args.credit)
            how = (f"{lf.per_word:g} a word (worst {_pct(lf.core)}, γ {lf.gamma:.3f})" if lf.per_word is not None
                   else f"none ≤ cap holds it (worst {_pct(lf.core)})")
            print(f"    least {'+'.join(lf.mix)} filler that holds {mg.m*100:g}%: {how}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="machine-model", description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA, help="the data directory (default: %(default)s)")
    dev = f"the device: {', '.join(devices())} (default sm_120, data/ itself; the others data/devices/sm_NN/)"
    p.add_argument("--device", help=dev)
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
    r = sub.add_parser("designs")
    r.add_argument("--json", action="store_true")
    e = sub.add_parser("eps")
    e.add_argument("designs", nargs="*")
    e.add_argument("--device", default=argparse.SUPPRESS, help=dev)
    e.add_argument("--q", type=float, default=8.0, help="q*, lane instructions a fresh word (default %(default)s)")
    e.add_argument("--q-uniform", type=float, help="q*_U, uniform instructions a fresh word (default: q*)")
    e.add_argument("--operands", choices=E.OPERANDS, default="free",
                   help="the cheater's operand delivery on the tensor route (default %(default)s)")
    e.add_argument("--credit", choices=E.CREDITS, default="H",
                   help="what ε is a fraction of: honest's kernel (H, default), its model bound B(H) (model) or the "
                        "floor program ♭ (floor, SPEC §0.7a)")
    e.add_argument("--fold-cost", type=float, help="scale the design's fold to R lane instructions a word")
    e.add_argument("--vector", help="the vector whose rows are Π (default: model.toml's current)")
    e.add_argument("--json", action="store_true")
    e.add_argument("--table", action="store_true")
    e.add_argument("--fold-sweep", help="with --table: each design at each of these fold costs")
    e.add_argument("--fill", type=float, help="add X lane instructions a word of verified filler on lanes")
    e.add_argument("--fill-mix", metavar="K=V,...",
                   help="with --fill: the filler's class mix on lanes (default ALU=1,IMAD=1, LOP3 and IMAD alternating)")
    e.add_argument("--lookups", action="store_true",
                   help="open the cheater's table route: words from register-indexed L2 lookups (data/table.toml)")
    e.add_argument("--lookup-q", type=float, help="with --lookups: lane instructions a table word (default: the file's)")
    e.add_argument("--l2", type=float, help="with --lookups: L2 read bytes a second (default: the file's)")
    e.add_argument("--uni-issue", type=float, metavar="C",
                   help="put the uniform classes on the issue rows (issue, the joint JI…_0) at C times an ALU's "
                        "cost there (PREDICT's u16s_l7qs measured 0.92, r20261017-045133-67bf)")
    e.add_argument("--checking", choices=("full", "sampled"), default="full",
                   help="full: every word and operand verified, the core ε alone (default); sampled: also ε_sample = "
                        "ε_luck + ε_adapt, on its own line and columns")
    e.add_argument("--adapt", metavar="UNIT",
                   help="with --checking sampled: the proof unit (data/adapt.toml; default: the design's own)")
    e.add_argument("--lookup-bytes", type=float, metavar="B",
                   help="with --lookups: L2 bytes a table word (default: the file's)")
    e.add_argument("--index-bits", type=int, metavar="B",
                   help="B-bit noise entries on the operand the table route indexes, which costs B times the 1-bit "
                        "route's lane instructions and bytes (default: the design's [table] bits, else 1)")
    k = sub.add_parser("capacity")
    k.add_argument("groups", nargs="+", metavar="GROUP")
    k.add_argument("--vector", help="the vector whose rows are Π (default: model.toml's current)")
    k.add_argument("--json", action="store_true")
    rt = sub.add_parser("redteam")
    rt.add_argument("designs", nargs="*", help="the designs to audit (default: every design in the data directory)")
    rt.add_argument("--device", default=argparse.SUPPRESS, help=dev)
    rt.add_argument("--q", type=float, default=1.22, help="q*, lane instructions a fresh word (default %(default)s)")
    rt.add_argument("--operands", choices=E.OPERANDS, default="free",
                    help="the cheater's operand delivery on the tensor route (default %(default)s)")
    rt.add_argument("--credit", choices=E.CREDITS, default="floor",
                    help="what ε is a fraction of (default %(default)s: the floor program ♭)")
    mg = sub.add_parser("margin")
    mg.add_argument("designs", nargs="*", help="the designs to audit (default: every design in the data directory)")
    mg.add_argument("--device", default=argparse.SUPPRESS, help=dev)
    mg.add_argument("--m", type=float, default=0.02, help="the margin each near facet moves in the cheater's favour "
                    "(default %(default)s: decision 76's 2%%)")
    mg.add_argument("--least", action="store_true", help="also find the cheapest filler a word that holds the margin")
    mg.add_argument("--mix", help="with --least: the filler's class mix, comma-separated (default: the design's)")
    mg.add_argument("--q", type=float, default=1.22, help="q*, lane instructions a fresh word (default %(default)s)")
    mg.add_argument("--operands", choices=E.OPERANDS, default="free",
                    help="the cheater's operand delivery on the tensor route (default %(default)s)")
    mg.add_argument("--credit", choices=E.CREDITS, default="floor",
                    help="what ε is a fraction of (default %(default)s: the floor program ♭)")
    args = p.parse_args(argv)
    if args.device is not None:
        if args.data != DATA:
            raise SystemExit("--device or --data, not both")
        args.data = device(args.device)
    if args.cmd is None:
        print(__doc__)
        return 0
    return {"list": _list, "solve": _solve, "census": _census, "attacks": _attacks, "adopt": _adopt,
            "designs": _designs, "eps": _eps, "capacity": _capacity, "redteam": _redteam,
            "margin": _margin}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
