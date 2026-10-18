"""machine-model render [SECTION ...]       every table of the data directory as Markdown on stdout, read from data/ alone
                                          (no solve): vectors, epochs, rows, floors, designs, honest, units, forms,
                                          table, attacks, devices (default: all of them); the README holds none"""
from __future__ import annotations

import tomllib
from collections import Counter
from pathlib import Path

from verity_machine_model import eps as E
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
        body.append([f"`{n}`", v.epoch, status, len(v.rows), len(v.credits), _num(p, "tau", 4), _num(p, "tau_600w", 4),
                     _num(p, "uf_low", 4), _num(p, "margin", 3), v.note, v.summary])
    return _table(["vector", "epoch", "status", "rows", "credits", "τ", "τ at 600 W", "U/F_low", "margin", "note",
                   "summary"], body) if body else None


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
    return _table(["epoch", "vector", "census lines from it", "kinds", "rows", "lines it closes (until the epoch before)",
                   "attacks from it"], body)


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
    return _table(["q (lane instructions a word)", "L2 bytes a word", "L2 bytes a second", "SMs", "MHz", "status",
                   "source"], [[t["q"], t["bytes"], f"{t['l2']:.4g}" if "l2" in t else "unmeasured", t["sms"], t["mhz"], t.get("status"),
                                t["source"]]])


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


SECTIONS = {"vectors": ("Vectors", _vectors), "epochs": ("Epochs and their census", _epochs),
            "rows": ("Rows", _rows), "floors": ("Floors", _floors), "designs": ("Designs", _designs),
            "honest": ("Honest kernels", _honest), "units": ("Proof units", _units), "forms": ("MMA forms", _forms),
            "table": ("The table route", _route), "attacks": ("Attack families", _attacks),
            "devices": ("Devices", _devices)}


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
