"""machine-model eps [DESIGN] [--device sm_NN] [--q Q] [--q-uniform QU] [--operands O] [--credit C]
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
                                          core ε alone (ε_luck in it, Corollary D3), and --checking sampled adds
                                          ε_sample = ε_adapt on its own line, for the design's proof block
                                          (data/adapt.toml) or --adapt UNIT's
machine-model eps --table [DESIGN ...] [--fold-sweep R,R,...] [--fill X [--fill-mix K=V,...]] [--operands O]
                  [--credit C] [--q Q] [--lookups ...] [--uni-issue C] [--checking full|sampled [--adapt UNIT]]
                                          one line a design (every design by default; with --fold-sweep, each at
                                          each fold cost), in two column groups: the core (γ and γ_f, γ with the
                                          clock gap f_max/f_H, the credited time, ε_skip, ε_sched, ε_machine at
                                          q* = 2, 4, 8, 16 and its floor, ε_luck at Q, the core ε at q* = 2, 8
                                          and Q, and the thresholds), then with --checking sampled the sample
                                          (ε_adapt and ε_sample at Q)"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

from verity_machine_model import eps as E
from verity_machine_model.commands._shared import device_help, pct as _pct, registry as _registry


def add(sub) -> None:
    e = sub.add_parser("eps")
    e.add_argument("designs", nargs="*")
    e.add_argument("--device", default=argparse.SUPPRESS, help=device_help())
    e.add_argument("--q", type=float, default=8.0, help="q*, lane instructions a fresh word (default %(default)s)")
    e.add_argument("--q-uniform", type=float, help="q*_U, uniform instructions a fresh word (default: q*)")
    e.add_argument("--operands", choices=E.OPERANDS, default="free",
                   help="the cheater's operand delivery on the tensor route (default %(default)s)")
    e.add_argument("--credit", choices=E.CREDITS, default="H",
                   help="what ε is a fraction of: honest's kernel (H, default), its model bound B(H) (model) or the "
                        "floor program ♭ (floor, Corollary D3)")
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
                        "cost there (u16s_l7qs measures 0.92)")
    e.add_argument("--checking", choices=("full", "sampled"), default="full",
                   help="full: every word and operand verified, the core ε alone, ε_luck in it (default); sampled: "
                        "also ε_sample = ε_adapt, on its own line and columns")
    e.add_argument("--adapt", metavar="UNIT",
                   help="with --checking sampled: the proof block (data/adapt.toml; default: the design's own)")
    e.add_argument("--lookup-bytes", type=float, metavar="B",
                   help="with --lookups: L2 bytes a table word (default: the file's)")
    e.add_argument("--index-bits", type=int, metavar="B",
                   help="B-bit noise entries on the operand the table route indexes, which costs B times the 1-bit "
                        "route's lane instructions and bytes (default: the design's [table] bits, else 1)")


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
            raise SystemExit("eps: --adapt picks the sampled protocol's proof block: add --checking sampled")
        return None
    return E.adapt(args.adapt or d.unit, args.data)


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
        "luck_budget": {"checks": d.checks, "delta": d.delta, "p": d.p},
        "sample": None if not sampled else {
            "eps": e.sample(), "adapt": adapt.eps if adapt else 0.0,
            "unit": adapt.name if adapt else None, "selection": adapt.selection if adapt else None},
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
            "floor": "the floor program ♭ (Corollary D3)"}[args.credit]
    tb = _lookups(args)
    sampled = args.checking == "sampled"
    look = "" if tb is None else (f", the table route open at {tb.q:g} lane instructions and {tb.bytes:g} B of L2 a "
                                  f"word ({_l2(tb)})")
    unit = "" if not sampled else (f" Sample (the sampling compiler, never in core): the proof block ("
                                   f"{f'{args.adapt} for every design' if args.adapt else 'each design its own'}), "
                                   f"ε_adapt, ε_sample = ε_adapt.")
    print(f"Π {pi.vector}, operands {args.operands}, credit {cred}{look}, checking {args.checking}. Core (every word "
          f"and operand verified): γ_f γ with the clock gap f_max/f_H; T/MMA the credited time and its binding rows, "
          f"H/cred honest's time over it; ε_skip; "
          f"ε_sched; ε_machine at q* = {', '.join(map(str, qs))} and its floor (the lane and uniform routes closed); "
          f"ε_luck at q* = {args.q:g}; core ε at q* = {', '.join(f'{q:g}' for q in qe)}; q*₀ the least q* at which "
          f"ε_machine reaches its floor (all routes; lanes alone); the uniform route alone at q* = 16.{unit}")
    print(f"{'design':26s} {'fold':>6s} {'L':>3s} {'γ':>6s} {'γ_f':>6s} {'T/MMA':>7s} {'binds':12s} {'H/cred':>6s} "
          f"{'skip':>7s} "
          f"{'sched':>6s} " + " ".join(f"{'m@' + str(q):>7s}" for q in qs)
          + f" {'floor':>7s} {'luck':>7s} " + " ".join(f"{f'core@{q:g}':>9s}" for q in qe)
          + f" {'q*₀':>9s} {'lanes':>6s} {'UP@16':>6s}"
          + (f" | {'unit':>15s} {'adapt':>7s} {'sample':>7s}" if sampled else ""))
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
              + f" {100 * e.floor:6.2f}% {100 * eq.luck:6.3f}% "
              + " ".join(f"{100 * es[q].core():8.3f}%" for q in qe)
              + f" {_thr(t):>9s} {_thr(tl):>6s} {100 * up:5.2f}%"
              + (f" | {ad.name:>15s} {100 * ad.eps:6.3f}% {100 * eq.sample():6.3f}%" if ad else ""))
    return 0


def run(args) -> int:
    pi = E.machine(args.vector or "", args.data)
    if args.uni_issue is not None and E.uniform_issue_charged(pi):
        sheet = "DS_issue" in pi.c.get(E.UNI, {})
        print(f"eps: {pi.vector}'s census charges a uniform instruction on the issue rows already; --uni-issue "
              f"{args.uni_issue:g} changes nothing" + ("; on a data sheet, `machine-model sheet --eps --uni C` prices "
                                                       "it at C of a slot" if sheet else ""), file=sys.stderr)
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
    ops = {"free": "free (the conservative default)", "honest": "at honest's",
           "floor": f"at the noise's floor φ_del, {floor} a MMA ({d.floor_source})"}
    built = {"H": "honest as built", "model": "honest as built",
             "floor": f"the floor program ♭ (the tensor route at φ_del = {floor} a MMA)"}
    print(f"{d.name}: {d.summary}")
    print(f"  {d.title}" + (f" (also {', '.join(d.aliases)})" if d.aliases else "") + ": noise "
          + ", ".join(f"{k} {v}" for k, v in d.noise.items()) + f"; proof block {d.unit}; {d.coin_bits}-bit coins; "
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
    short = ", ".join(f"{k} {100 * v:.4f}%" for k, v in e.lucky.short.items())
    print(f"  ε_luck     {_pct(t['luck'])}: the lucky cheater leaves undone {short} (words alone: d̄ "
          f"{100 * e.deficit:.4f}%; budget δ {d.delta:.3g}, p {d.p} a word"
          + "".join(f", {f.p} filler {j + 1}" for j, (f, _) in enumerate(e.fillers)) + f", S {d.checks:g}: "
          f"{d.luck_source})")
    core = ("skip", "mand", "machine", "luck", "sched", "clock")
    print(f"  ε core     {_pct(e.core())} (100% checking, every word and operand verified, Corollary D3; first order "
          f"{_pct(sum(t[k] or 0.0 for k in core))})")
    if args.checking == "sampled":
        print(f"  ε_sample   {_pct(e.sample())} (the sampling compiler's, never in the core ε) = ε_adapt")
        print(f"      ε_adapt {_pct(t['adapt'])}: proof block {ad.name}, {ad.selection}: {ad.source}")
    print(f"  q* threshold (ε_machine = its floor): {_thr(rec['threshold'])}; lanes alone "
          f"{_thr(rec['threshold_lanes'])}; the uniform route alone at q* = 16: {_pct(rec['uniform_at_16'])}")
    print(textwrap.fill("ε_machine(q*): " + ", ".join(f"{x:g} {100 * y:.2f}%" for x, y in rec["curve"]), width=120,
                        initial_indent="  ", subsequent_indent="    "))
    print(textwrap.fill(f"ε_machine(q*_U) at q* = {args.q:g}: "
                        + ", ".join(f"{x:g} {100 * y:.2f}%" for x, y in rec["curve_uniform"]), width=120,
                        initial_indent="  ", subsequent_indent="    "))
    return 0
