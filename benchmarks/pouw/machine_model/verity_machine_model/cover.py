"""The coverage table (data/cover/): the evidence for MODEL-MIN's coverage premise (CMP-165), that every instruction a card
executes is dominated by an envelope op (envelope.toml) at no fewer units of the same resource, one row per opcode
(rows.toml) on the H200 (sm_90) and the RTX PRO 6000 (sm_120).

An opcode a card lists is in NVIDIA's family table (CUDA Binary Utilities 12.9 §4.3/§4.4 and CUDA 13's, in
benchmarks/pouw/sass_census/data/), in the SASS corpus of the CUDA 13 libraries (corpus.json), or both; a census kind is
one the machine model's census measured (data/census/*.csv for sm_120, data/devices/sm_90/census*). `problems` lists
every way the table fails that: an opcode, a form or a kind with no row, a row naming an op or a θ key that doesn't
exist, and a tensor row whose rate times its weight exceeds the card's tensor rate.
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
DEVICES = ("sm_90", "sm_120")
FAMILY = {"sm_90": "hopper", "sm_120": "blackwell"}
MARKS = ("covered", "coarse", "hole", "absent")
SLACK = 1.001   # a measured rate a hair over its sheet (2047.6 · 1/2 against 1024) is the sheet's rounding


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


def _base(op: str) -> str:
    return op if op in ("UIADD3.64", "UMOV.64") else op.split(".")[0]


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


def load(cover: Path = COVER, tables: Path = TABLES) -> Cover:
    env = tomllib.loads((cover / "envelope.toml").read_text())
    data = tomllib.loads((cover / "rows.toml").read_text())
    rows = [Row(i, tuple(r["ops"]), tuple(r.get("devices", DEVICES)), tuple(r.get("forms", ())), r["mark"],
                r["envelope"], r) for i, r in enumerate(data["row"])]
    bu = json.loads((tables / "bu_12_9.json").read_text())["tables"]
    c13 = json.loads((tables / "sass_isa_tables.json").read_text())["tables"]
    corpus = json.loads((cover / "corpus.json").read_text())["opcodes"]
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


def mark_of(c: Cover, dev: str, op: str) -> str:
    """An opcode's mark on a device: its worst row's (hole, then coarse, then covered), absent only if every row is."""
    ms = {r.mark for r in rows_of(c, dev, op)}
    for m in ("hole", "coarse", "covered"):
        if m in ms:
            return m
    return "absent" if ms else "missing"


def rate(c: Cover, r: Row, dev: str, theta: T.Theta) -> float | None:
    """A tensor row's rate, MACs a SM-clock (logical for a sparse form): the larger of its θ key's sheet and measured
    values, or its `rate_value`."""
    v = r.get("rate_value", dev)
    if v is not None:
        return float(v)
    k = r.get("rate", dev)
    if k is None or k not in theta.rates:
        return None
    x = theta.rates[k]
    vals = [y for y in (x.sheet, x.measured) if y is not None]
    return max(vals) if vals else None


def tensor_units(c: Cover, r: Row, dev: str, theta: T.Theta) -> float | None:
    """T units a SM-clock the row's instruction can take: rate × weight, halved for a sparse form's kept products."""
    x = rate(c, r, dev, theta)
    if x is None:
        return None
    w = c.envelope["device"][dev]["weight"][r.get("fmt")]
    return x * w / (2 if r.envelope == "T.sp" else 1)


def problems(c: Cover) -> list[str]:
    out: list[str] = []
    ops = {o["key"] for o in c.envelope["op"]}
    thetas = {d: T.theta(device(d)) for d in DEVICES}
    for r in c.rows:
        where = f"rows.toml row {r.n} ({', '.join(r.ops)})"
        if r.mark not in MARKS:
            out.append(f"{where}: mark {r.mark!r} isn't one of {', '.join(MARKS)}")
        if r.envelope not in ops:
            out.append(f"{where}: envelope op {r.envelope!r} isn't in envelope.toml")
        if r.mark == "hole" and "severity" not in r.raw:
            out.append(f"{where}: a hole says its severity")
        if r.mark != "absent" and r.envelope != "-" and "argument" not in r.raw:
            out.append(f"{where}: a dominated row states its argument")
        for dev in r.devices:
            if dev not in DEVICES:
                out.append(f"{where}: device {dev!r} isn't one of {', '.join(DEVICES)}")
                continue
            k = r.get("rate", dev)
            if k is not None and k not in thetas[dev].rates:
                out.append(f"{where}: θ key {k!r} isn't in {dev}'s theta.toml")
            if r.envelope in ("T", "T.sp") and r.mark in ("covered", "coarse"):
                u = tensor_units(c, r, dev, thetas[dev])
                rho = c.envelope["device"][dev]["rho"]["tensor"]
                if u is None:
                    out.append(f"{where}: a dominated tensor row on {dev} needs a rate (θ key or rate_value)")
                elif u > rho * SLACK:
                    out.append(f"{where}: on {dev} rate × weight = {u:g} T units a SM-clock, over ρ_T = {rho:g}")
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


def counts(c: Cover, dev: str) -> Counts:
    """Opcodes by mark, corpus forms by their row's mark, and the corpus's instructions by mark."""
    ops = {m: 0 for m in (*MARKS, "missing")}
    for op in c.opcodes[dev]:
        ops[mark_of(c, dev, op)] += 1
    forms = {m: 0 for m in (*MARKS, "missing")}
    insts = dict(forms)
    for f, n in c.forms[dev].items():
        r = row_of(c, dev, f)
        m = r.mark if r else "missing"
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
