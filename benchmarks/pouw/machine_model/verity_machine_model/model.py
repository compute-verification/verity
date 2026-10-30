"""The machine model's data (`data/`): the constants, the rows, the census of instruction classes, the attacks and the named
vectors, read into what the LP takes.

An *epoch* is a state of the census and the attack list, named after the vector that introduced it (`epochs` in
`model.toml`, in order). A census line or an attack holds from its `since` epoch through its `until` epoch (empty: every
epoch from the first, or through the last), so a refit or a new measurement is a new line and every older vector still
re-solves on the data it was published on. The census is a file for each epoch (`data/census/EPOCH.csv`, the lines
from that epoch), so a new epoch adds a file and closes older lines in theirs.
"""
from __future__ import annotations

import csv
import json
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DEVICES = DATA / "devices"
DEVICE = "sm_120"   # data/ itself, the RTX PRO 6000: the default device

Census = dict[str, dict[str, float]]  # instruction class: {row: u one warp instruction holds the row alone}
STATUSES = ("measured", "ASSUMED", "provisional", "joint", "datasheet")


@dataclass(frozen=True)
class Model:
    cycle_u: float                      # u in one SMSP-cycle (R*)
    honest_real: float                  # u of real time a MAC for the honest kernel
    cuda: tuple[str, ...]               # the CUDA lane kinds ρ binds, in constraint order
    honest: dict[str, float]            # the honest GEMM's program a MMA
    adopted: str                        # the design whose cheapest attack is the margin
    rho: float
    tie: float
    x_min: float
    epochs: tuple[str, ...]
    current: str = ""                   # the vector `solve` re-solves by default
    uniform_forms: tuple[str, ...] | None = None   # the lane classes with a uniform-datapath form; None: every one


@dataclass(frozen=True)
class Row:
    name: str
    status: str
    runs: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class Entry:
    kind: str
    row: str
    u: float
    since: str
    until: str
    runs: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class Attack:
    design: str
    family: str
    name: str
    program: dict[str, float]           # warp instructions of each kind a tensor MAC replaced, in pricing order
    on: bool = True
    since: str = ""
    until: str = ""


@dataclass(frozen=True)
class Floor:
    kind: str
    words: int          # words a warp instruction of the kind writes (at most)
    floor: float        # MACs a word its price must reach
    claim: str          # one of CLAIMS
    what: str


CLAIMS = {"pi32": "(π₃₂): every root-writing kind at ≥ 32 MACs a word",
          "loads16": "(π₃₂)'s load half: every load into registers at ≥ 16 MACs a word",
          "copy16": "π_c ≥ 16: every copy route (every kind that writes registers, a UR word at the 32 lanes it serves) "
                    "at ≥ 16 MACs a word"}
COPIES = "copy16"   # the claim whose kinds `solve` lists with their prices: the copy routes


@dataclass(frozen=True)
class Vector:
    name: str
    epoch: str
    rows: tuple[str, ...]
    credits: tuple[tuple[str, float], ...]
    summary: str
    note: str
    superseded_by: str
    published: dict[str, float]
    legacy: dict[str, float] = field(default_factory=dict)


def u_of(cycle_u: float, rate: str, noise: str, cycles: str, u: str) -> float:
    """u one warp instruction holds a row, from exactly one of: its measured `rate` (warp instructions/SM/clk, 4 SMSPs an SM,
    derated by `noise` when given), `cycles` (SMSP-cycles it holds the row) or `u` itself (a derived value: a fit or a joint
    vertex)."""
    given = [c for c in (rate, cycles, u) if c != ""]
    if len(given) != 1:
        raise ValueError(f"a census line gives exactly one of rate, cycles and u, not {given}")
    if rate != "":
        r = float(rate)
        return cycle_u * 4 / (r * (1 + float(noise))) if noise != "" else cycle_u * 4 / r
    if noise != "":
        raise ValueError("noise derates a rate only")
    return float(cycles) * cycle_u if cycles != "" else float(u)


def model(data: Path = DATA) -> Model:
    """A device directory (`data/devices/sm_NN/`) that no vector is solved on leaves out what only `solve` reads: the
    honest kernel's u a MAC, the price LP's `[lp]`, `[honest]` and `adopted` (NaN and empty here)."""
    t = tomllib.loads((data / "model.toml").read_text())
    c, lp = t["constants"], t.get("lp", {})
    return Model(cycle_u=c["cycle_u"], honest_real=c.get("honest_real", math.nan), cuda=tuple(t["cuda"]),
                 honest=dict(t.get("honest", {})), adopted=t.get("adopted", ""), rho=lp.get("rho", math.nan),
                 tie=lp.get("tie", math.nan), x_min=lp.get("x_min", math.nan), epochs=tuple(t["epochs"]),
                 current=t["current"],
                 uniform_forms=tuple(t["uniform_forms"]) if "uniform_forms" in t else None)


def device(name: str = DEVICE) -> Path:
    """A device's data directory (README "Devices"): `data/` for sm_120, else `data/devices/sm_NN/`, which a command can
    price once it holds a `model.toml` beside its census."""
    m = re.fullmatch(r"sm_?(\d+)", name)
    key = f"sm_{m.group(1)}" if m else name
    if key == DEVICE:
        return DATA
    p = DEVICES / key
    if not (p / "model.toml").is_file():
        why = "has a census and no model.toml yet" if (p / "census.csv").is_file() else "isn't a device directory"
        raise SystemExit(f"device {name}: {p} {why}; priced: {', '.join(devices())}")
    return p


def devices() -> list[str]:
    """The devices a command can price: sm_120 and each device directory with a `model.toml`."""
    return [DEVICE, *sorted((p.name for p in DEVICES.glob("sm_*") if (p / "model.toml").is_file()),
                            key=lambda n: int(n[3:]))]


def _runs(s: str) -> tuple[str, ...]:
    return tuple(s.split())


def cites(s: str, named: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """A `runs` field's runs and notes, each `@NAME` the list `named` (`run_lists`) gives it."""
    return tuple(x for c in _runs(s) for x in (named[c[1:]] if c.startswith("@") else (c,)))


def rows(data: Path = DATA) -> dict[str, Row]:
    named = run_lists(data)
    with open(data / "rows.csv", newline="") as f:
        return {d["row"]: Row(d["row"], d["status"], cites(d["runs"], named), d["description"])
                for d in csv.DictReader(f)}


def census_files(m: Model, data: Path = DATA) -> list[Path]:
    """`data/census/EPOCH.csv`: a file for each epoch that brought census lines, in model.toml's order, each line's `since`
    its file's epoch. A device directory (`data/devices/sm_NN/`) keeps its measured census in one `census.csv`, which
    its writer rewrites whole, and any other epoch's lines (sm_90's and sm_100's data sheet, `census/DS-NN.csv`) after
    it."""
    out = [p] if (p := data / "census.csv").is_file() else []
    return out + [p for e in m.epochs if (p := data / "census" / f"{e}.csv").is_file()]


def citations() -> dict[str, str]:
    """`data/citations.toml`: the notes many lines cite, in `data/` and every device directory, each cited in a line's
    `runs` as `@NAME`."""
    return tomllib.loads((DATA / "citations.toml").read_text())


def run_lists(data: Path = DATA) -> dict[str, tuple[str, ...]]:
    """`data/census/runs.toml`: the run lists many census lines share (a refit's every line cites every run its fit read),
    each cited in a line's `runs` as `@NAME`; and `citations`, each a list of its one note."""
    p = data / "census" / "runs.toml"
    own = {k: tuple(v) for k, v in tomllib.loads(p.read_text()).items()} if p.is_file() else {}
    shared = {k: (v,) for k, v in citations().items()}
    if both := set(own) & set(shared):
        raise SystemExit(f"{p}: {sorted(both)} also name a note in {DATA / 'citations.toml'}; give the run list its "
                         f"own name")
    return own | shared


def kind_lists(data: Path = DATA) -> dict[str, tuple[str, ...]]:
    """`data/census/kinds.toml`: the kind lists a row prices at one value (the joint issue rows' vector classes), each
    written as one census line of kind `@NAME`, which stands for a line of each kind it lists."""
    p = data / "census" / "kinds.toml"
    return {k: tuple(v) for k, v in tomllib.loads(p.read_text()).items()} if p.is_file() else {}


def census_lines(m: Model, data: Path = DATA) -> list[dict[str, str]]:
    """Every census line as written, the files in epoch order."""
    out: list[dict[str, str]] = []
    for p in census_files(m, data):
        with open(p, newline="") as f:
            out += csv.DictReader(f)
    return out


def entries(m: Model, data: Path = DATA) -> list[Entry]:
    named, kinds = run_lists(data), kind_lists(data)
    return [Entry(k, d["row"], u_of(m.cycle_u, d["rate"], d["noise"], d["cycles"], d["u"]), d["since"],
                  d["until"], cites(d["runs"], named)) for d in census_lines(m, data)
            for k in (kinds[d["kind"][1:]] if d["kind"].startswith("@") else (d["kind"],))]


def holds(m: Model, since: str, until: str, epoch: str) -> bool:
    i = m.epochs.index(epoch)
    return (since == "" or m.epochs.index(since) <= i) and (until == "" or i <= m.epochs.index(until))


def census(m: Model, epoch: str, data: Path = DATA) -> Census:
    """Each instruction class's u on each row at `epoch`."""
    out: Census = {}
    for e in entries(m, data):
        if holds(m, e.since, e.until, epoch):
            if e.row in out.setdefault(e.kind, {}):
                raise ValueError(f"census: two lines for {e.kind} on {e.row} at {epoch}")
            out[e.kind][e.row] = e.u
    return out


def measured_cost(c: Census) -> dict[str, float]:
    """An instruction class's measured cost: its time alone, the largest of its rows."""
    return {k: max(v.values()) for k, v in c.items()}


def floors(data: Path = DATA) -> list[Floor]:
    """The per-word price floors the claims read (`data/floors.csv`): (π₃₂) for root-writing kinds, its load half
    for loads into registers, and π_c ≥ 16 (per-row keys' premise) for every copy route."""
    with open(data / "floors.csv", newline="") as f:
        return [Floor(d["kind"], int(d["words"]), float(d["floor"]), d["claim"], d["what"]) for d in csv.DictReader(f)]


def all_attacks(data: Path = DATA) -> list[Attack]:
    """Every attack in `data/attacks/*.jsonl`, a file per family, one program a line."""
    out = []
    for p in sorted((data / "attacks").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(Attack(d["design"], d["family"], d["name"], d["program"], d.get("on", True),
                                  d.get("since", ""), d.get("until", "")))
    return out


def attacks(m: Model, epoch: str, data: Path = DATA) -> list[Attack]:
    return [a for a in all_attacks(data) if holds(m, a.since, a.until, epoch)]


def vector(name: str, data: Path = DATA) -> Vector:
    p = data / "vectors" / f"{name}.toml"
    if not p.is_file():
        raise SystemExit(f"no vector {name!r}: one of {', '.join(vectors(data))}")
    t = tomllib.loads(p.read_text())
    return Vector(name=t["name"], epoch=t["epoch"], rows=tuple(t["rows"]), credits=tuple((k, c) for k, c in t["credits"]),
                  summary=t.get("summary", ""), note=t.get("note", ""), superseded_by=t.get("superseded_by", ""),
                  published=dict(t.get("published", {})), legacy=dict(t.get("legacy", {})))


def vectors(data: Path = DATA) -> list[str]:
    m = model(data)
    names = [p.stem for p in (data / "vectors").glob("*.toml")]
    return sorted(names, key=lambda n: (m.epochs.index(vector(n, data).epoch),
                                        [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)]))
