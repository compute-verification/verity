"""machine-model render [SECTION ...]       every table of the data directory as Markdown, read from data/ alone (no
                                          solve): vectors, epochs, rows, floors, designs, honest, units, forms, table,
                                          headline, attacks, devices, theta (default: all of them); the README holds
                                          none"""
from __future__ import annotations

import tomllib
from collections import Counter
from pathlib import Path

from verity_machine_model import eps as E
from verity_machine_model import theta as T
from verity_machine_model.model import (CLAIMS, DATA, all_attacks, census_files, census_lines, device, devices, floors,
                                        model, rows, vector, vectors)


def _cell(x) -> str:
    s = "–" if x is None or x == "" else str(x)
    return s.replace("\n", " ").replace("|", "\\|")


def _table(head: list[str], body: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_cell(x) for x in r) + " |" for r in body]
    return "\n".join(out)


def _code(xs) -> str:
    return ", ".join(f"`{x}`" for x in xs)


def _num(d: dict, k: str, places: int) -> str:
    return f"{d[k]:.{places}f}" if k in d else "–"


def _vectors(data: Path) -> str | None:
    m = model(data)
    body = []
    for n in vectors(data):
        v = vector(n, data)
        status = "current" if n == m.current else f"superseded by {v.superseded_by}" if v.superseded_by else ""
        p = v.published
        body.append([f"`{n}`", v.epoch, status, len(v.rows), len(v.credits), _num(p, "tau", 4), _num(p, "uf_low", 4),
                     _num(p, "margin", 3), v.note, v.summary])
    return _table(["vector", "epoch", "status", "rows", "credits", "τ", "U/F_low", "margin", "note", "summary"],
                  body) if body else None


def _epochs(data: Path) -> str | None:
    m = model(data)
    first = m.epochs[0]
    lines: dict[str, list[dict]] = {}
    for d in census_lines(m, data):
        lines.setdefault(d["since"] or first, []).append(d)
    closed = Counter(d["until"] for d in census_lines(m, data) if d["until"])
    since = Counter(a.since or first for a in all_attacks(data))
    named = set(vectors(data))
    body = []
    for i, e in enumerate(m.epochs):
        ls = lines.get(e, [])
        body.append([f"`{e}`", "yes" if e in named else "", len(ls), len({d["kind"] for d in ls}),
                     len({d["row"] for d in ls}), closed[m.epochs[i - 1]] if i else 0, since[e]])
    return _table(["epoch", "vector", "census lines from it", "kinds", "rows",
                   "lines it closes (until the epoch before)", "attacks from it"], body)


def _rows(data: Path) -> str | None:
    m = model(data)
    first: dict[str, str] = {}
    for n in vectors(data):
        for r in vector(n, data).rows:
            first.setdefault(r, n)
    cur = set(vector(m.current, data).rows) if m.current in vectors(data) else set()
    body = [[f"`{r.name}`", r.status, first.get(r.name, ""), "yes" if r.name in cur else "", " ".join(r.runs),
             r.description] for r in rows(data).values()]
    return _table(["row", "status", "first vector", f"in {m.current}", "runs", "what"], body)


def _floors(data: Path) -> str | None:
    if not (data / "floors.csv").is_file():
        return None
    body = [[f"`{f.kind}`", f.words, f"{f.floor:g}", f.claim, f.what] for f in floors(data)]
    claims = "\n".join(f"- `{k}`: {v}" for k, v in CLAIMS.items())
    return claims + "\n\n" + _table(["kind", "words a warp instruction", "floor (MACs a word)", "claim", "what"], body)


def _designs(data: Path) -> str | None:
    names = E.designs(data)
    if not names:
        return None
    body = []
    for n in names:
        d = E.design(n, data)
        filler = "; ".join(f"{f.per_word:g} a word ({', '.join(f'{k} {v:g}' for k, v in (f.mix or {}).items())}), "
                           f"p {f.p:g}" for f in d.filler)
        body.append([f"`{d.name}`", d.title, _code(d.aliases), d.base and f"`{d.base}`",
                     ", ".join(d.noise[p] for p in E.PRODUCTS), d.sums, d.honest, d.L,
                     f"{d.fold_name} {d.fold_cost():g}", filler, d.unit, d.coin_bits, d.baseline_name,
                     d.ppl and f"`{d.ppl}`"])
    legend = ("Noise leaves (`NOISE` in `eps.py`):\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in E.NOISE.items())
              + "\n\nPeel sums kernels (`SUMS`):\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in E.SUMS.items()))
    return _table(["design", "title", "aliases", "base", "noise: linears, lm_head, QKᵀ, P·V", "sums", "honest", "L",
                   "fold", "filler", "unit", "coin bits", "baseline", "perplexity arm"], body) + "\n\n" + legend


def _honest(data: Path) -> str | None:
    body = []
    for n in E.profiles(data):
        h = E.honest(n, data)
        body.append([f"`{h.name}`", h.status, h.L, h.mma, h.form, h.words, f"{h.fold_name} {sum(h.fold.values()):g}",
                     "–" if h.measured is None else f"{h.measured:g}", h.source])
    return _table(["kernel", "status", "L", "MMA", "form", "words a MMA", "fold", "measured SMSP-cycles a MMA",
                   "source"], body) if body else None


def _units(data: Path) -> str | None:
    body = [[f"`{u.name}`", f"{u.eps:g}", u.selection, u.source] for u in (E.adapt(n, data) for n in E.adapts(data))]
    return _table(["unit", "ε_adapt", "selection", "source"], body)


def _forms(data: Path) -> str | None:
    p = data / "forms.toml"
    if not p.is_file():
        return None
    t = tomllib.loads(p.read_text())
    body = [[f"`{k}`", "×".join(map(str, f["shape"])), f["split"], "yes" if f.get("sparse") else "", f["exact"],
             _code(f.get("sass", ()))] for k, f in t.items() if isinstance(f, dict)]
    return _table(["form", "shape m×n×k", "split", "2:4 sparse", "exact", "SASS"], body)


def _route(data: Path) -> str | None:
    if not (data / "table.toml").is_file():
        return None
    t = tomllib.loads((data / "table.toml").read_text())
    l2 = f"{t['l2']:.4g}" if "l2" in t else "unmeasured"
    return _table(["q (lane instructions a word)", "L2 bytes a word", "L2 bytes a second", "SMs", "MHz", "status",
                   "source"], [[t["q"], t["bytes"], l2, t["sms"], t["mhz"], t.get("status"), t["source"]]])


def _headline(data: Path) -> str | None:
    p = data / "headline.toml"
    if not p.is_file():
        return None
    t = tomllib.loads(p.read_text())
    readings = list(t.get("reading", {}))
    body = [[f, x.get("gpu"), f"`{x['design']}`", x.get("p1") and f"`{x['p1']}`",
             *(x.get(r, {}).get("vector") for r in readings), "; ".join(x.get("conditions", ()))]
            for f, x in t.get("family", {}).items()]
    head = ["family", "GPU", "design", "p = 1 sibling",
            *(f"{t['reading'][r].get('title', r)} (q* = {t['reading'][r]['q']:g})" for r in readings), "conditions"]
    return _table(head, body)


def _attacks(data: Path) -> str | None:
    fams: dict[str, list] = {}
    for a in all_attacks(data):
        fams.setdefault(a.family, []).append(a)
    m = model(data)
    order = lambda e: m.epochs.index(e) if e in m.epochs else -1
    span = lambda es: ", ".join(sorted(es, key=order))
    body = [[f, len(xs), "; ".join(sorted({a.design for a in xs})), span({a.since or m.epochs[0] for a in xs}),
             span({a.until or "open" for a in xs})] for f, xs in fams.items()]
    return _table(["family", "attacks", "designs", "since", "until"], body) if body else None


def _devices(_data: Path) -> str | None:
    body = []
    for name in devices():
        d = device(name)
        m = model(d)
        lines = sum(len(p.read_text().splitlines()) - 1 for p in census_files(m, d))
        body.append([name, f"`{_where(d)}`", len(m.epochs), len(rows(d)), lines, len(vectors(d)),
                     len(E.designs(d)), len(E.profiles(d)), m.current])
    return _table(["device", "directory", "epochs", "rows", "census lines", "vectors", "designs", "honest kernels",
                   "current vector"], body)


def _measured(x) -> str:
    return "–" if x is None else f"{x:g}"


def _theta(data: Path) -> str | None:
    if not T.has(data):
        return None
    t = T.theta(data)
    figs = {f.key: f for f in T.figures(t)}
    out = [f"{t.gpu}, {t.sms} SMs ({t.sms_source}). Documents: " + "; ".join(t.documents) + ".",
           "### Clocks",
           _table(["clock", "MHz", "grade", "source", "measured"],
                  [[f"`{c.name}`", f"{c.mhz:g}", c.grade, c.source, c.measured] for c in t.clocks.values()]),
           "### Rates (a SM-clock's, in the unit named; B/s for the whole GPU)",
           _table(["rate", "what", "unit", "sheet", "figure at clock (10¹²/s)", "measured", "grade", "source",
                   "runs, or the gap"],
                  [[f"`{r.key}`", r.what, r.unit, _measured(r.sheet),
                    f"{figs[r.key].figure:g} at `{r.clock}` (2 · sheet · SMs · clock {figs[r.key].value:.2f})"
                    if r.key in figs else "", _measured(r.measured), r.grade, r.source, r.runs or f"gap: {r.gap}"]
                   for r in t.rates.values()]),
           "### Capacities",
           _table(["capacity", "what", "value", "unit", "grade", "source", "measured, or the gap"],
                  [[f"`{c.key}`", c.what, f"{c.value:g}", c.unit, c.grade, c.source,
                    c.measured or (c.gap and f"gap: {c.gap}")] for c in t.capacities.values()]),
           "### Exact paths",
           _table(["format", "native", "exact", "ISA", "measured"],
                  [[p["format"], p["native"], p["exact"], p["isa"], p.get("measured")] for p in t.paths]),
           "### FP8 atoms",
           _table(["form", "products", "group", "alignment", "accumulation", "promotion", "rate", "source"],
                  [[x[k] for k in ("form", "products", "group", "alignment", "accumulation", "promotion", "rate",
                                   "source")] for x in t.fp8])]
    hs = T.folds(t, data)
    L = hs[0].L
    body = []
    rs = {rd: {r.name: r for r in T.routes(t, rd)} for rd in T.READINGS}
    for name in rs["sheet"]:
        body.append([f"d: {name}", *(f"{rs[rd][name].d:.2f}" if name in rs[rd] else "–" for rd in T.READINGS),
                     "ρ₈ over deliveries a SM-clock"])
    body.append(["d: mma.sync D ≠ C seed, charged its issue slot", *(f"{T.seed(t, rd):.2f}" for rd in T.READINGS),
                 "ρ₈ / (issue · 128 words); 0 while the seed MMA's products are needed"])
    body.append(["per second over per SM-clock, a tensor-over-lane ratio", *(f"{T.tick(t, rd):.4f}" for rd in T.READINGS),
                 "the tensor figures' clock over the lanes'"])
    ss = {rd: T.strassen(t, rd, L) for rd in T.READINGS}
    body.append([f"Strassen at L = {L}: tensor time over naive", *(f"{ss[rd].tensor:.4f}" for rd in T.READINGS),
                 f"7/8 · ρ_k{L} / ρ_k{L // 2}"])
    for name in (n for n in T.ADDING if n in rs["sheet"]):
        body.append([f"exchange: {name}", *(f"{ss[rd].rates[name]:.3f}" if name in ss[rd].rates else "none"
                                            for rd in T.READINGS),
                     "5d/(4L) / (1 − tensor time); none: the level saves no tensor time"])
    for big in T.NOFOLD_L:
        cheapest = [min(s.rates.values(), default=None) for s in (T.strassen(t, rd, big) for rd in T.READINGS)]
        body.append([f"exchange at L = {big}, the cheapest route", *(f"{x:.4g}" if x is not None else "none"
                                                                     for x in cheapest),
                     "k32 products at full rate: 10d/L"])
    bs = {rd: T.bound(t, rd) for rd in (*T.READINGS, "measured only")}
    mo = bs["measured only"]
    body.append(["B_RF, words a SM-clock", f"{bs['sheet'].words:.1f}", f"{bs['measured'].words:.1f}",
                 f"LP over the writers; measured only: {mo.words:.1f}"])
    body.append(["c ≈ ρ₈/B_RF (bits)", f"{bs['sheet'].c:.4f}", f"{bs['measured'].c:.4f}",
                 f"measured only: {mo.c:.4f}"])
    for f in hs:
        body.append([f"r_H, honest `{f.profile}` ({f.status}, {f.fold})", f"{f.r_h:.4f}", f"{f.r_h:.4f}",
                     ", ".join(f"{c} {v:.4f}" for c, v in f.per_word.items())])
        body.append([f"r_min, {f.fold}'s own", f"{f.r_min:g}", f"{f.r_min:g}",
                     ", ".join(f"{c} {v:g}" for c, v in f.minimal.items())])
        body.append([f"slack over the cheapest exchange, `{f.profile}`",
                     *(f"{100 * f.eps(rd):.2f}%" for rd in T.READINGS),
                     "(r_H − r_min) in tensor MAC-times a MAC, over the cheapest adding route's exchange"])
        body.append([f"γ, `{f.profile}`", "–", f"{f.gamma:.3f}" if f.gamma is not None else "not measured",
                     f"measured SMSP-cycles a MMA over the lone INT8 MMA's {f.lone:.2f}"])
    out += ["### Derived (`machine-model theta` prints the arithmetic)",
            _table(["derived", "sheet", "measured", "how"], body)]
    share = lambda u, rd: "" if (s := T.share(t, u, rd)) is None else f"{100 * s:.2f}%"
    out += ["### Units, with their verdicts",
            _table(["unit", "what", "verdict", "rates", "bound a SM-clock", "share, sheet", "share, measured", "why",
                    "settles it"],
                   [[f"`{u.key}`", u.what, u.verdict, _code(u.rates),
                     f"{u.bound:g} against `{u.against}`" if u.bound is not None else "", share(u, "sheet"),
                     share(u, "measured"), u.why, u.gap] for u in t.units.values()])]
    if "sha512" in t.rates:
        ours = t.rates["sha512"].measured
        out += ["### SHA-512 against the public figures (c_sha)",
                _table(["form", "implementation", "card", "SMs", "G compressions/s", "at most MHz",
                        "a SM-clock, at least", "ours over it", "c_sha at most", "source"],
                       [[s.public.form, s.public.impl, s.public.card, f"{s.public.sms}", f"{s.public.gps:g}",
                         f"{s.public.mhz:g}", f"{s.per_clk:.6f}", f"{ours / s.per_clk:.3f}", f"{s.c:.3f}",
                         s.public.source]
                        for s in T.sha(t)])]
    if t.sha_mix is not None:
        cs = [T.sha_ceiling(t, rd, hi) for hi in (True, False) for rd in T.READINGS]
        out += ["### SHA-512's ceiling on the card's pipes (`[sha512_mix]`, an LP over ALU, IMAD and issue)",
                _table(["IMAD.HI", "reading", "SM-clocks a compression", "as built", "rotations by IMAD.HI",
                        "rotations by SHF", "adds to IMAD", "a SM-clock", "ours over it"],
                       [["yes" if c.hi else "no", c.reading, f"{c.clocks:.2f}", f"{c.built:.2f}", f"{c.hi_rotates:.0f}",
                         f"{c.shf_rotates:.0f}", f"{c.adds:.0f}", f"{c.rate:.6f}", f"{ours / c.rate:.3f}"]
                        for c in cs]),
                f"Mix: {t.sha_mix['source']}. ALU and IMAD on it: {t.sha_mix['overlap']}."]
    return "\n\n".join(out)


SECTIONS = {"vectors": ("Vectors", _vectors), "epochs": ("Epochs and their census", _epochs),
            "rows": ("Rows", _rows), "floors": ("Floors", _floors), "designs": ("Designs", _designs),
            "honest": ("Honest kernels", _honest), "units": ("Proof blocks", _units), "forms": ("MMA forms", _forms),
            "table": ("The table route", _route), "headline": ("The per-family headline's designs", _headline),
            "attacks": ("Attack families", _attacks), "devices": ("Devices", _devices),
            "theta": ("θ: rates, capacities, clocks and exact paths, with sources", _theta)}


def _where(data: Path) -> str:
    return f"{data.resolve().relative_to(DATA.parent)}/" if data.resolve().is_relative_to(DATA.parent) else str(data)


def render(data: Path = DATA, sections: list[str] | None = None) -> str:
    out = [f"# `machine-model render`: {_where(data)}"]
    for key in sections or SECTIONS:
        title, f = SECTIONS[key]
        if (body := f(data)) is not None:
            out.append(f"## {title}\n\n{body}")
    return "\n\n".join(out) + "\n"


def add(sub) -> None:
    r = sub.add_parser("render")
    r.add_argument("sections", nargs="*", metavar="SECTION",
                   help=f"the sections to print, of {', '.join(SECTIONS)} (default: all)")


def run(args) -> int:
    if bad := [s for s in args.sections if s not in SECTIONS]:
        raise SystemExit(f"render: no section {', '.join(bad)}; the sections are {', '.join(SECTIONS)}")
    print(render(args.data, args.sections), end="")
    return 0
