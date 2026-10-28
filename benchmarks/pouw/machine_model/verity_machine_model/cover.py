"""The coverage table (data/cover/): the evidence for MODEL-MIN's coverage premise, that every instruction a card
executes is dominated by an envelope op (envelope.toml: MODEL-MIN's `Accel.Op`, its pipes, formats and card numbers, and
the entries MODEL-COVER proposes) at no fewer units of the same resource, one row per opcode (rows.toml) on the H200
(sm_90) and the RTX PRO 6000 (sm_120).

An opcode a card lists is in NVIDIA's family table (CUDA Binary Utilities 12.9 §4.3/§4.4 and CUDA 13's, in
benchmarks/pouw/sass_census/data/), in the SASS corpus of the CUDA 13 libraries (corpus.json), or both; a census kind is
one the machine model's census measured (data/census/*.csv for sm_120, data/devices/sm_90/census*). `problems` lists
every way the table fails that: an opcode, a form or a kind with no row; a row naming an op, pipe, format, store or θ key
that doesn't exist; a lane or collective reading or writing more words than its op allows; and a covered or coarse row
whose rate, in its op's units (`demands`), exceeds its resource's ρ. A row whose op, pipe or format is proposed rests on
it (`rests_on`), and `counts(..., proposed=False)` counts those rows as holes: the table on `Accel.Op` alone.
"""
from __future__ import annotations

import csv
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from verity_machine_model import theta as T
from verity_machine_model.model import DATA, device, kind_lists

COVER = DATA / "cover"
TABLES = Path(__file__).resolve().parents[2] / "sass_census" / "data"
#: corpus.json and the ISA tables are in fixtures/artifacts.json and gitignored; this brings them into a checkout
FETCH = "uv run research data fetch-fixtures benchmarks/pouw/machine_model/data/cover benchmarks/pouw/sass_census/data"
DEVICES = ("sm_90", "sm_120")
FAMILY = {"sm_90": "hopper", "sm_120": "blackwell"}
MARKS = ("covered", "coarse", "hole", "absent")
DOMINATED = ("covered", "coarse")
SLACK = 1.001   # a measured rate a hair over its sheet (2047.6 · 1/2 against 1024) is the sheet's rounding
LANES = 32
WORD = 4        # bytes a word
TENSOR = ("mma", "mmaSp")
PIPED = ("lane", "collective")
STORED = ("move", "atom", "red")
RATED = (*TENSOR, *PIPED, "tex", *STORED)


@dataclass(frozen=True)
class Row:
    n: int
    ops: tuple[str, ...]
    devices: tuple[str, ...]
    forms: tuple[str, ...]
    mark: str
    envelope: str
    raw: dict = field(compare=False, repr=False)

    def get(self, k: str, dev: str | None = None):
        v = self.raw.get(k)
        return v.get(dev) if isinstance(v, dict) and dev is not None else v

    def matches(self, op: str) -> bool:
        return any(re.fullmatch(f, op) for f in self.forms)


@dataclass(frozen=True)
class Cover:
    envelope: dict
    rows: list[Row]
    units: list[dict]
    opcodes: dict[str, dict[str, set[str]]]   # device → opcode → the sources listing it
    forms: dict[str, dict[str, int]]          # device → dotted corpus opcode → instructions in the corpus
    kinds: dict[str, set[str]]                # device → census kinds

    def table(self, name: str) -> dict[str, dict]:
        return {e["key"]: e for e in self.envelope[name]}


@dataclass(frozen=True)
class Demand:
    """One of a row's rates read in its op's units a SM-clock against its resource's ρ."""
    rate: str                # where the rate is from: "θ KEY" or the row's rate_value
    amount: float | None     # the op's units a SM-clock at that rate
    resource: str            # a pipe, a store, tensor or tex
    rho: float | None
    problem: str | None = None


def _base(op: str) -> str:
    return op if op in ("UIADD3.64", "UMOV.64") else op.split(".")[0]


def as_list(v) -> list:
    return [] if v is None else list(v) if isinstance(v, list) else [v]


def _kinds(dev: str) -> set[str]:
    """The census kinds a device's census files name, a kind list (`@NAME`) read as its kinds."""
    d = device(dev)
    files = [*sorted((d / "census").glob("*.csv"))] + ([d / "census.csv"] if (d / "census.csv").is_file() else [])
    lists = kind_lists(d)
    out: set[str] = set()
    for p in files:
        for r in csv.DictReader(p.open()):
            k = r["kind"]
            out |= set(lists[k[1:]]) if k.startswith("@") else {k}
    return out


def fetched(p: Path) -> dict:
    if not p.is_file():
        raise FileNotFoundError(f"{p} is a fetched fixture and isn't here: `{FETCH}`")
    return json.loads(p.read_text())


def load(cover: Path = COVER, tables: Path = TABLES) -> Cover:
    env = tomllib.loads((cover / "envelope.toml").read_text())
    data = tomllib.loads((cover / "rows.toml").read_text())
    rows = [Row(i, tuple(r["ops"]), tuple(r.get("devices", DEVICES)), tuple(r.get("forms", ())), r["mark"],
                r["envelope"], r) for i, r in enumerate(data["row"])]
    bu = fetched(tables / "bu_12_9.json")["tables"]
    c13 = fetched(tables / "sass_isa_tables.json")["tables"]
    corpus = fetched(cover / "corpus.json")["opcodes"]
    opcodes: dict[str, dict[str, set[str]]] = {}
    for dev in DEVICES:
        o: dict[str, set[str]] = {}
        for src, table in (("BU 12.9", bu), ("CUDA 13", c13)):
            for op in table[FAMILY[dev]]:
                o.setdefault(op, set()).add(src)
        for op in corpus[dev]:
            o.setdefault(_base(op), set()).add("corpus")
        opcodes[dev] = o
    return Cover(env, rows, data["unit"], opcodes, corpus, {d: _kinds(d) for d in DEVICES})


def plain(c: Cover, dev: str, op: str) -> list[Row]:
    return [r for r in c.rows if op in r.ops and dev in r.devices and not r.forms]


def row_of(c: Cover, dev: str, form: str) -> Row | None:
    """The row a dotted opcode resolves to: its one matching forms row, else its opcode's plain row."""
    base = _base(form)
    hit = [r for r in c.rows if base in r.ops and dev in r.devices and r.forms and r.matches(form)]
    if hit:
        return hit[0]
    p = plain(c, dev, base)
    return p[0] if p else None


def rows_of(c: Cover, dev: str, op: str) -> list[Row]:
    return [r for r in c.rows if op in r.ops and dev in r.devices]


def rests_on(c: Cover, r: Row) -> list[str]:
    """The proposed entries a row's argument needs: its op, its pipe and its formats, each one envelope.toml marks
    `proposed`."""
    out = []
    if "proposed" in c.table("op").get(r.envelope, {}):
        out.append(f"op {r.envelope}")
    if "proposed" in c.table("pipe").get(r.get("pipe"), {}):
        out.append(f"pipe {r.get('pipe')}")
    fmts = c.table("fmt")
    out += [f"fmt {f}" for f in as_list(r.get("fmt")) if "proposed" in fmts.get(f, {})]
    return out


def mark(c: Cover, r: Row, proposed: bool = True) -> str:
    """A row's mark, or hole when `proposed` is off and it rests on a proposed entry."""
    return "hole" if not proposed and r.mark in DOMINATED and rests_on(c, r) else r.mark


def mark_of(c: Cover, dev: str, op: str, proposed: bool = True) -> str:
    """An opcode's mark on a device: its worst row's (hole, then coarse, then covered), absent only if every row is."""
    ms = {mark(c, r, proposed) for r in rows_of(c, dev, op)}
    for m in ("hole", "coarse", "covered"):
        if m in ms:
            return m
    return "absent" if ms else "missing"


def rho(c: Cover, dev: str, resource: str) -> float | None:
    """A resource's ρ a SM-clock on a device: a pipe's (its resource's ρ times its lanes), a store's in words (dram's
    from words a second), tensor's in s8 MACs; the card's numbers, else the proposed ones."""
    d = c.envelope["device"][dev]
    pipe = c.table("pipe").get(resource)
    lanes = 1
    if pipe is not None:
        resource, lanes = pipe["resource"], pipe.get("lanes", 1)
    if resource == "dram":
        return d["dram"] / (d["sms"] * d["clock_mhz"] * 1e6)
    v = d["rho"].get(resource, d.get("proposed_rho", {}).get(resource))
    return None if v is None else v * lanes


def resources(c: Cover, r: Row, dev: str) -> list[str]:
    """What a row's rates are read against: tensor, its pipe, tex, or its stores (one a rate; one rate or one store
    is read against all of the other)."""
    if r.envelope in TENSOR:
        return ["tensor"]
    if r.envelope in PIPED:
        return as_list(r.get("pipe"))
    if r.envelope == "tex":
        return ["tex"]
    return as_list(r.get("resource", dev))


def _amount(x: float, unit: str, units: float, d: dict) -> float | None:
    """A rate a SM-clock (or a second) in the op's units a SM-clock, at `units` a warp instruction."""
    if unit == "warp instructions/SM/clk":
        return x * units
    if unit in ("lanes/SM/clk", "texels/SM/clk"):   # θ tex's texel is one filtered fetch, a lane's
        return x * units / LANES
    if unit == "FP16 FMAs/SM/clk":
        return x / 2 * units / LANES
    if unit == "B/SM/clk":
        return x / WORD
    if unit == "B/s":
        return x / WORD / (d["sms"] * d["clock_mhz"] * 1e6)
    if unit in ("adds/SM/clk", "words/SM/clk"):
        return x
    return None


def demands(c: Cover, r: Row, dev: str, theta: T.Theta) -> list[Demand]:
    """Each of a row's rates (the larger of a θ key's sheet and measured values, or its rate_value) in its op's units a
    SM-clock: tensor units (MACs × the largest weight of its formats, halved for mmaSp's kept products), a pipe's lanes
    (`units` a warp instruction, 32 by default), tex's fetches, or a store's words."""
    d = c.envelope["device"][dev]
    v = r.get("rate_value", dev)
    if v is not None:
        unit = "MACs/SM/clk" if r.envelope in TENSOR else r.get("rate_unit", dev)
        rates = [(f"{v:g} {unit or '(no rate_unit)'}", float(v), unit)]
    else:
        rates = []
        for k in as_list(r.get("rate", dev)):
            x = theta.rates.get(k)
            vals = [y for y in (x.sheet, x.measured) if y is not None] if x else []
            rates.append((f"θ {k}", max(vals) if vals else None, x.unit if x else None))
    res = resources(c, r, dev)
    if len(res) == 1:
        res = res * len(rates)
    elif len(rates) == 1:
        rates = rates * len(res)
    if len(res) != len(rates):
        return [Demand(k, None, "", None, f"{len(rates)} rates against {len(res)} resources") for k, _, _ in rates]
    out = []
    for (k, x, unit), s in zip(rates, res):
        p = rho(c, dev, s)
        problem = None
        amount = None
        if x is None:
            problem = f"{k} has no value"
        elif r.envelope in TENSOR:
            ws = [d["weight"].get(f) for f in as_list(r.get("fmt"))]
            if not ws or None in ws:
                problem = f"a format of {as_list(r.get('fmt'))} has no weight on {dev}"
            else:
                amount = x * max(ws) / (2 if r.envelope == "mmaSp" else 1)
        else:
            amount = _amount(x, unit or "", r.get("units", dev) or LANES, d)
            if amount is None:
                problem = f"{k}'s unit {unit!r} has no conversion to {r.envelope}'s units"
        if problem is None and p is None:
            problem = f"{s!r} has no ρ on {dev}"
        out.append(Demand(k, amount, s, p, problem))
    return out


def _words(c: Cover, r: Row) -> str | None:
    """Why a lane's or collective's words exceed its op's, or None."""
    w = r.get("words")
    if not isinstance(w, list) or len(w) != 2:
        return "a lane or collective states its register words a lane, words = [in, out]"
    op = c.table("op")[r.envelope]
    n_in = op.get("n_in", c.table("pipe").get(r.get("pipe"), {}).get("n_in"))
    n_out = op["n_out"]
    if n_in is None or w[0] > n_in or w[1] > n_out:
        return f"{w[0]} words in and {w[1]} out a lane, past {r.envelope}'s {n_in} in and {n_out} out on {r.get('pipe')}"
    return None


def problems(c: Cover) -> list[str]:
    out: list[str] = []
    ops, pipes, fmts = c.table("op"), c.table("pipe"), c.table("fmt")
    thetas = {d: T.theta(device(d)) for d in DEVICES}
    for r in c.rows:
        where = f"rows.toml row {r.n} ({', '.join(r.ops)})"
        if r.mark not in MARKS:
            out.append(f"{where}: mark {r.mark!r} isn't one of {', '.join(MARKS)}")
        if r.envelope not in ops:
            out.append(f"{where}: envelope op {r.envelope!r} isn't in envelope.toml")
            continue
        if r.mark == "hole" and "severity" not in r.raw:
            out.append(f"{where}: a hole says its severity")
        if r.mark != "absent" and r.envelope != "-" and "argument" not in r.raw:
            out.append(f"{where}: a dominated row states its argument")
        if r.envelope in PIPED and r.mark != "absent" and r.get("pipe") not in pipes:
            out.append(f"{where}: a {r.envelope} names its pipe, one of {', '.join(pipes)}")
        if r.envelope not in PIPED and "pipe" in r.raw:
            out.append(f"{where}: only a lane or a collective has a pipe")
        if r.envelope in TENSOR and r.mark != "absent":
            for f in as_list(r.get("fmt")) or [None]:
                if f not in fmts:
                    out.append(f"{where}: format {f!r} isn't in envelope.toml")
        if r.envelope in PIPED and r.mark in DOMINATED and (w := _words(c, r)):
            out.append(f"{where}: {w}")
        for dev in r.devices:
            if dev not in DEVICES:
                out.append(f"{where}: device {dev!r} isn't one of {', '.join(DEVICES)}")
                continue
            for k in as_list(r.get("rate", dev)):
                if k not in thetas[dev].rates:
                    out.append(f"{where}: θ key {k!r} isn't in {dev}'s theta.toml")
            if r.envelope not in RATED or r.mark not in DOMINATED:
                continue
            ds = demands(c, r, dev, thetas[dev])
            if not ds:
                out.append(f"{where}: a dominated {r.envelope} row on {dev} needs a rate (θ key or rate_value)")
            for x in ds:
                if x.problem:
                    out.append(f"{where}: on {dev} {x.problem}")
                elif x.amount > x.rho * SLACK:
                    out.append(f"{where}: on {dev} {x.rate} is {x.amount:g} {r.envelope} units of {x.resource} a "
                               f"SM-clock, over ρ {x.resource} = {x.rho:g}")
    for u in c.units:
        if u["envelope"] not in ops:
            out.append(f"rows.toml unit {u['key']}: envelope op {u['envelope']!r} isn't in envelope.toml")
    for dev in DEVICES:
        for op in sorted(c.opcodes[dev]):
            if not plain(c, dev, op):
                srcs = ", ".join(sorted(c.opcodes[dev][op]))
                out.append(f"{dev} {op} ({srcs}) has no plain row in rows.toml: add one, or add {op} to a row's ops")
        for form in sorted(c.forms[dev]):
            base = _base(form)
            hit = [r.n for r in c.rows if base in r.ops and dev in r.devices and r.forms and r.matches(form)]
            if len(hit) > 1:
                out.append(f"{dev} {form} matches the forms of rows {hit}: narrow them to one")
        for k in sorted(c.kinds[dev]):
            if not any(dev in r.devices and any(re.fullmatch(p, k) for p in r.get("kinds") or ()) for r in c.rows):
                out.append(f"{dev} census kind {k} is claimed by no row's kinds")
        units = {u["key"] for u in c.units if dev in u.get("devices", DEVICES)}
        for k in sorted(set(T.UNITS) - units):
            out.append(f"{dev} θ unit {k} has no [[unit]] in rows.toml")
    return out


@dataclass(frozen=True)
class Counts:
    opcodes: dict[str, int]
    forms: dict[str, int]
    instructions: dict[str, int]


def counts(c: Cover, dev: str, proposed: bool = True) -> Counts:
    """Opcodes by mark, corpus forms by their row's mark, and the corpus's instructions by mark; with `proposed` off,
    a row resting on a proposed entry is a hole."""
    ops = {m: 0 for m in (*MARKS, "missing")}
    for op in c.opcodes[dev]:
        ops[mark_of(c, dev, op, proposed)] += 1
    forms = {m: 0 for m in (*MARKS, "missing")}
    insts = dict(forms)
    for f, n in c.forms[dev].items():
        r = row_of(c, dev, f)
        m = mark(c, r, proposed) if r else "missing"
        forms[m] += 1
        insts[m] += n
    return Counts(ops, forms, insts)


def holes(c: Cover) -> list[tuple[str, str, str, str]]:
    """Every hole: (devices, what, severity, argument), rows then units."""
    out = []
    for r in c.rows:
        if r.mark == "hole":
            ops = ", ".join(r.ops) + (f" [{', '.join(r.forms)}]" if r.forms else "")
            out.append(("/".join(r.devices), f"{ops}: {r.raw.get('what', '')}", r.raw["severity"], r.raw["argument"]))
    for u in c.units:
        if u["mark"] == "hole":
            out.append(("/".join(u.get("devices", DEVICES)), f"unit {u['key']}: {u['what']}", u["severity"],
                        u["argument"]))
    return out
