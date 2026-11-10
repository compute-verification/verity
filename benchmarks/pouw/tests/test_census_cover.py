"""Census cover: no instruction of a bit-exact program the tree carries is free at a headline statement's certificate.

Bit-exact folds on the H200, the B200 and sm_120 broke FOLD-OPT at π* the same way: a bit-exact fold issued an
instruction the statement's machine has no class for, or one whose class has no coefficient on π*'s row, so the Lean
priced it at 0 and the fold came in under ♭'s. Here that is a failure. For every headline statement the exporters write
(`lean_headline.py`, and `lean_headline_dc.py` where it exists), on the statement's vector and at its certificate π*,
every instruction of every bit-exact program the tree carries for that device maps to one of the vector's classes priced
above 0 at π* (Σ_r π*_r·c_kr > 0). The programs:

- every table of census-class counts in the device's data: the honest kernels' profiles, the designs (folds,
  `[fold_routes]`, fillers, baselines, operand floors) and `headline.toml`;
- sm_120's attack families that hold at the vector's epoch (`data/attacks/`);
- `test_lean_headline.py`'s `FOLDS` and `UNIFORM_FOLD`, where it has them;
- every program `facet_break/generate.py` emits, each family's targeted part, by the SASS its gate pins (`expect_of`);
- the red-team witnesses' timed loops (`fixtures/census_cover_sass.json`: cpw_routes' modes, rt3_dc's lane loops).

A SASS opcode's class is the one the census reads off a timed program (`loads.rates_by_class`: `class_of`, a side unit's
or TLD's by the program's name), on every device: the H200's and B200's data have no map of their own. An opcode it
leaves unclassed, or whose class the statement's vector has no line for, is a warp instruction no class of the
statement names, so it counts in the exporter's declared catch-all (`CATCH_ALL`: by device key, A2's on that device's
data sheet; by vector name, that vector's), which must then be priced at π* itself; a class the vector lists but π*
prices at 0 is a gap whatever the catch-all. NOP is left out: it changes no state, so the program without its NOPs is
the same program.

A gap someone is closing goes on `CLOSING` by device and instruction, so it holds while a statement moves to a new vector,
with its owner and reason. An entry fails once its instruction has its own class priced at π* on every statement of its
device (a catch-all covering it doesn't count: the entry waits for the class), so the list only shrinks.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest
from verity_machine_model.model import DATA, attacks, devices, model, vector, vectors

POUW = Path(__file__).resolve().parents[1]
SASS = Path(__file__).resolve().parent / "fixtures" / "census_cover_sass.json"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("pouw_lean_headline", POUW / "lean_headline.py")
loads = _load("facet_break_loads", POUW / "facet_break" / "loads.py")
generate = _load("facet_break_generate", POUW / "facet_break" / "generate.py")

#: instructions that change no state: a program priced without them is the same program
NO_WORK = frozenset({"NOP"})
#: a census class, as a table's key (a vector's name has a hyphen, a setting is lower case)
CLASS = re.compile(r"[A-Z][A-Z0-9_]*")


@dataclass(frozen=True)
class Statement:
    label: str
    key: str                       # the device, sm_NN
    data: Path
    vector: str
    lam: dict[str, Fraction]       # π*, its rows' weights
    price: dict[str, Fraction]     # each class of the statement's machine at π*
    catch_all: str | None
    exporter: str


@dataclass(frozen=True)
class Program:
    key: str
    name: str
    n: dict[str, float]            # instructions a unit, by SASS opcode (`sass`) or census class
    sass: bool
    holds: tuple[str, str] | None = None   # an attack's (since, until) epochs
    probe: str | None = None       # a generated program's name, which classes its side units and TLDs


def class_of(p: Program, op: str) -> str | None:
    """A SASS opcode's census class as the census reads it off a timed program (`loads.rates_by_class`)."""
    return next(iter(loads.rates_by_class({"probe": p.probe, "rates": {f"loop:{op}": 1.0}})), None)


@dataclass(frozen=True)
class Gap:
    st: Statement
    item: str
    cls: str | None
    why: str
    where: tuple[str, ...]


def _at(lam: dict[str, Fraction], c: dict[str, Fraction]) -> Fraction:
    return sum((w * c.get(r, Fraction(0)) for r, w in lam.items()), Fraction(0))


def statements() -> list[Statement]:
    """Every headline statement the exporters write: its vector, its π* and each class's price there."""
    i = H.instance()
    m = H.measured(i)
    fam, cat = H.SM120, getattr(H, "CATCH_ALL", {})
    out = [Statement(f"{fam.ns}SheetHeadline", fam.key, fam.data, fam.device, {i.pistar: Fraction(1)},
                     {k: H.coef(i, i.pistar, k) for k in i.classes}, cat.get(fam.key) or cat.get(fam.device),
                     "lean_headline.py"),
           Statement(f"{fam.ns}MeasuredHeadline", fam.key, fam.data, H.VECTOR, {m.pistar: Fraction(1)},
                     {k: m.c[k].get(m.pistar, Fraction(0)) for k in i.classes}, cat.get(H.VECTOR), "lean_headline.py")]
    dc = POUW / "lean_headline_dc.py"
    if dc.is_file():
        D = _load("pouw_lean_headline_dc", dc)
        cat = getattr(D, "CATCH_ALL", {})
        for f in D.FAMILIES:
            for v, kind in ((f.sheet, "Sheet"), (f.measured, "Measured")):
                x = D.reading(f, v).data
                out.append(Statement(f"{f.ns}{kind}Headline", f.key, f.data, v, dict(x.lam),
                                     {k: _at(x.lam, x.c[k]) for k in x.classes},
                                     (cat.get(f.key) if kind == "Sheet" else None) or cat.get(v), "lean_headline_dc.py"))
    return out


def _tables(t: dict, path: str):
    """Each table of census-class counts in a TOML document, by its dotted path; the top level holds settings only."""
    for k, v in t.items():
        p = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            n = {c: x for c, x in v.items() if CLASS.fullmatch(c) and isinstance(x, (int, float))
                 and not isinstance(x, bool) and x}
            if n:
                yield p, n
            yield from _tables(v, p)
        elif isinstance(v, list):
            for j, e in enumerate(v):
                if isinstance(e, dict):
                    yield from _tables({f"{k}[{j}]": e}, path)


def data_programs(key: str, data: Path) -> list[Program]:
    """The device's honest profiles, designs and (sm_120's) headline.toml and attack families, in census classes."""
    files = sorted((data / "honest").glob("*.toml")) + [data / "designs.toml"] + sorted((data / "designs").glob("*.toml"))
    if data == DATA:
        files.append(data / "headline.toml")
    out = []
    for f in files:
        if f.is_file():
            rel = f.relative_to(POUW).as_posix()
            out += [Program(key, f"{rel} [{p}]", n, False) for p, n in _tables(tomllib.loads(f.read_text()), "")]
    if (data / "attacks").is_dir():
        from verity_machine_model.model import all_attacks
        out += [Program(key, f"attack {a.family}: {a.name}", dict(a.program), False, (a.since, a.until))
                for a in all_attacks(data) if a.on]
    return out


def fold_programs() -> list[Program]:
    """`test_lean_headline.py`'s bit-exact fx1 folds, lane instructions a word, and its uniform route."""
    T = _load("pouw_test_lean_headline", POUW / "tests" / "test_lean_headline.py")
    out = [Program("sm_120", f"test_lean_headline.py FOLDS[{n!r}]", dict(f), False)
           for n, f in getattr(T, "FOLDS", {}).items()]
    if hasattr(T, "UNIFORM_FOLD"):
        out.append(Program("sm_120", "test_lean_headline.py UNIFORM_FOLD", dict(T.UNIFORM_FOLD), False))
    return out


def facet_break_programs() -> list[Program]:
    """Every family's targeted programs from `facet_break/generate.py`, by the SASS counts its gate pins a trip."""
    out = []
    for fam in re.findall(r'"(\w+)": \(targeted', inspect.getsource(generate.grid)):
        for g in generate.grid(0, 0, True, fam):
            n: dict[str, float] = {}
            for item in filter(None, re.split(r"[,;]", generate.expect_of(g)[0])):
                op, c = item.rsplit("=", 1)
                n[op] = n.get(op, 0) + int(c)
            out.append(Program("sm_120", f"facet_break {fam} {g['name']}", n, True, probe=g["name"]))
    return out


def witness_programs() -> list[Program]:
    return [Program(w["device"], f"{w['program']} ({w['run']})", w["counts"], True)
            for w in json.loads(SASS.read_text())["programs"]]


def programs() -> list[Program]:
    out = [p for k in devices() if k in {"sm_120", "sm_90", "sm_100"}
           for p in data_programs(k, DATA if k == "sm_120" else DATA / "devices" / k)]
    return out + fold_programs() + facet_break_programs() + witness_programs()


def _holds(st: Statement, p: Program) -> bool:
    if p.holds is None:
        return True
    from verity_machine_model.model import holds
    return holds(model(st.data), *p.holds, vector(st.vector, st.data).epoch)


def gaps(sts: list[Statement], progs: list[Program]) -> dict[tuple[str, str], Gap]:
    """Each (vector, instruction) a program issues that its statement prices at 0 at π*, with the programs issuing it."""
    found: dict[tuple[str, str], tuple[Statement, str | None, str, list[str]]] = {}
    for st in sts:
        for p in progs:
            if p.key != st.key or not _holds(st, p):
                continue
            for item, x in p.n.items():
                if not x or (p.sass and item in NO_WORK):
                    continue
                k = class_of(p, item) if p.sass else item
                if st.catch_all and (k is None or k not in st.price):
                    k = st.catch_all
                if k is None:
                    why = "has no class (loads.class_of)"
                elif k not in st.price:
                    why = f"is {k}, which {st.vector} has no line for" if p.sass else f"has no line on {st.vector}"
                elif st.price[k] <= 0:
                    why = (f"is {k}, " if p.sass else "") + f"which has no coefficient on {_rows(st)}"
                else:
                    continue
                found.setdefault((st.vector, item), (st, k, why, []))[3].append(p.name)
    return {key: Gap(st, key[1], k, why, tuple(ws)) for key, (st, k, why, ws) in found.items()}


def _rows(st: Statement) -> str:
    return " + ".join(f"{r} {w}" if w != 1 else r for r, w in st.lam.items())


def _census_file(st: Statement) -> str:
    epoch = vector(st.vector, st.data).epoch
    if st.data == DATA:
        return f"machine_model/data/census/{epoch}.csv (since {epoch})"
    return f"machine_model/data/devices/{st.key}/census.csv"


def fix(g: Gap) -> str:
    st, row = g.st, _rows(g.st)
    where = ", ".join(g.where[:3]) + (f" and {len(g.where) - 3} more" if len(g.where) > 3 else "")
    head = f"{st.vector} ({st.label}, π* {row}): {g.item} {g.why}; issued by {where}."
    line = f"a census line for {g.cls or 'its class'} on {' and '.join(st.lam)} in {_census_file(st)}"
    if g.cls is None:
        add = (f"Add a CLASS_RULES rule for it in facet_break/loads.py and {line}, or declare a catch-all priced on "
               f"{' and '.join(st.lam)} (CATCH_ALL[{st.vector!r}] in {st.exporter})")
    elif g.cls not in st.price:
        add = f"Add {line}" + (f", or map {g.item} in loads.class_of to a class {st.vector} has" if g.cls != g.item
                               else "")
    else:
        add = f"Add {line}"
    return (f"{head} {add}; a gap another PR is closing goes on CLOSING as ({st.key!r}, {g.item!r}) with its owner "
            f"and reason.")


def own_price(st: Statement, item: str) -> Fraction:
    """An instruction's price at π* by its own class, a catch-all aside: what a CLOSING entry waits for."""
    k = item if item in st.price else loads.class_of(item)
    return st.price.get(k, Fraction(0)) if k else Fraction(0)


def stale(sts: list[Statement], found: dict[tuple[str, str], Gap],
          closing: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """The CLOSING entries whose device has statements here, none with the gap, each pricing it by its own class."""
    out = []
    for key, item in closing:
        on = [st for st in sts if st.key == key]
        if on and not any(g.st.key == key and i == item for (_, i), g in found.items()) \
                and all(own_price(st, item) > 0 for st in on):
            out.append((key, item))
    return out


RT3_120 = "RT3-120 (cursor/ca-sm120-unclassed-ea24)"
UNOWNED = "none yet: compute-accounting's coordinator"

#: Gaps being closed: (device, instruction) → (owner, reason). Each is an escape of its device's headline FOLD-OPT
#: until its owner lands the class (a `loads.CLASS_RULES` rule and its census lines on π*'s rows) or a catch-all priced
#: on them. The witnesses are census_cover_sass.json's runs: cpw_routes (RTX PRO 6000), rt3_dc (H200 and B200), and
#: red_generic (its H200 and B200 SASS).
CLOSING: dict[tuple[str, str], tuple[str, str]] = {
    # sm_120, unclassed. On DS-120, the sheet's CATCH_ALL["sm_120"] (A2's NOROW, 1 at DS_issue) covers each; the
    # measured reading (VF3r-16 at JI3s_0, VF3s-16 at JI3t_0) has neither a class nor a catch-all for any of them.
    ("sm_120", "STG.E.64"): (RT3_120, "cpw_routes modes 2 and 7's state-word stores; facet_break uc's stg64"),
    ("sm_120", "STS.64"): (RT3_120, "facet_break uc's sts64 probe"),
    ("sm_120", "ST.E.128"): (RT3_120, "facet_break uc's st128 probe, a generic store"),
    ("sm_120", "LD.E.128"): (RT3_120, "facet_break uc's ld128 probe, a generic load"),
    ("sm_120", "BAR.SYNC.DEFER_BLOCKING"): (RT3_120, "cpw_routes modes 9 and 10's block barrier (RT3-120-6); uc's bar"),
    ("sm_120", "MEMBAR.ALL.GPU"): (RT3_120, "fence.proxy.async's first half in cpw_routes modes 8-10; uc's fpa"),
    ("sm_120", "FENCE.VIEW.ASYNC.S"): (RT3_120, "fence.proxy.async's second half in cpw_routes modes 8-10; uc's fpa"),
    ("sm_120", "TLD.LZ"): (RT3_120, "facet_break rt9's _tld1x8_ programs: loads._TLD_NAME reads no width from that name"),
    ("sm_120", "BRA"): (UNOWNED, "cpw_routes modes 8-10's branches"),
    ("sm_120", "BRA.U"): (UNOWNED, "the loop branch of every cpw_routes mode (0-10)"),
    ("sm_120", "BRA.U.ANY"): (UNOWNED, "cpw_routes modes 8-10's warp-uniform branch"),
    ("sm_120", "BSSY.RECONVERGENT"): (UNOWNED, "cpw_routes modes 8-10's reconvergence point"),
    ("sm_120", "BSYNC.RECONVERGENT"): (UNOWNED, "cpw_routes modes 8-10's reconvergence wait"),
    ("sm_120", "CCTL.IVALL"): (UNOWNED, "cpw_routes modes 8-10's L1 invalidate around the bulk reduce-xor"),
    ("sm_120", "DEPBAR.LE"): (UNOWNED, "cpw_routes modes 8-10's wait on the bulk operations"),
    ("sm_120", "ELECT"): (UNOWNED, "cpw_routes modes 8-10's elected issuing lane"),
    ("sm_120", "LDC"): (UNOWNED, "the constant-bank loads of cpw_routes modes 0-8 and 10"),
    ("sm_120", "LDC.64"): (UNOWNED, "cpw_routes modes 2, 3, 5 and 7's 64-bit constant-bank loads"),
    # sm_120, classed (loads.class_of: the H200's and B200's checkpoint loops issue them), with no sm_120
    # line: DS-120's catch-all covers each, and the measured reading has neither a line nor a catch-all for them; a line
    # on DS_issue and the measured π* row closes each
    ("sm_120", "STG.E.128"): (RT3_120, "cpw_routes modes 1 and 8-10's checkpoint egress (RT3-120-6); facet_break uc's stg128: STG128"),
    ("sm_120", "STG.E"): (RT3_120, "facet_break uc's stg32 probe (on RT3-120's branch): STG32"),
    ("sm_120", "STS"): (RT3_120, "facet_break uc's sts32 probe: STS32"),
    ("sm_120", "ISETP.NE.AND"): (UNOWNED, "cpw_routes mode 9's thread-0 test: ISETP"),
    ("sm_120", "REDUX.XOR"): (UNOWNED, "cpw_routes modes 2 and 7's warp XOR (LI-CP-W's honest route): REDUX"),
    # sm_120, classed, no line on the measured reading's π* row (JI3s_0 on main, JI3t_0 on); 1 at DS_issue
    ("sm_120", "UNI64"): (UNOWNED, "228 bit-slice attack variants' XU on UNI64, which is on UP alone (#2791's OFF_MENU)"),
    ("sm_120", "TMA512"): (UNOWNED, "attack (ii)'s TMA gather and dither-duo-l32's [per_mma]"),
    ("sm_120", "TMAred512"): (UNOWNED, "attack (ii)'s TMA reduce-add"),
    ("sm_120", "UBLKRED.G.S.XOR"): (UNOWNED, "cpw_routes modes 8-10's bulk reduce-xor (1 to 16 KB): TMAred512"),
    ("sm_120", "UTMA16K"): (UNOWNED, "the TMA loads of twelve honest profiles' [per_mma] (c-fx1, the l7qs kernels, fattn)"),
}


#: an instruction no class can ever name, for the tests of the check itself: every `loads.class_of` rule matches an
#: upper-case SASS mnemonic, and no SASS opcode is lower case, so a class added later can't take it
UNCLASSED = "toy.unclassed"


def test_the_check_finds_an_unclassed_instruction():
    """An instruction with no class (UNCLASSED), a classed instruction free at π* (SHFL.BFLY) and a class the vector has
    no line for (IMAD) are each a gap; a priced instruction and a NOP are not; a catch-all covers the unclassed
    instruction and the class with no line, never a listed class π* prices at 0."""
    st = Statement("toy", "sm_120", DATA, "toy", {"R": Fraction(1)}, {"ALU": Fraction(1), "SHFL": Fraction(0)}, None,
                   "lean_headline.py")
    p = Program("sm_120", "toy", {"LOP3.LUT": 2, UNCLASSED: 1, "SHFL.BFLY": 1, "NOP": 3}, True)
    q = Program("sm_120", "toy classes", {"ALU": 1, "IMAD": 1}, False)
    found = gaps([st], [p, q])
    assert {i for _, i in found} == {UNCLASSED, "SHFL.BFLY", "IMAD"}
    assert found[("toy", UNCLASSED)].cls is None and found[("toy", "SHFL.BFLY")].cls == "SHFL"
    caught = Statement(**{**st.__dict__, "catch_all": "ALU"})
    assert set(gaps([caught], [p, q])) == {("toy", "SHFL.BFLY")}


def test_a_closing_entry_waits_for_its_own_class():
    """An entry is stale once its instruction's own class is priced at π* on each of its device's statements; a
    catch-all covering it, a statement still free at π*, or a device with no statement here keeps it."""
    st = Statement("toy", "sm_120", DATA, "toy", {"R": Fraction(1)}, {"ALU": Fraction(1), "SHFL": Fraction(0)}, "ALU",
                   "lean_headline.py")
    other = Statement(**{**st.__dict__, "vector": "toy2", "price": {"ALU": Fraction(1), "SHFL": Fraction(2)}})
    p = Program("sm_120", "toy", {"LOP3.LUT": 2, UNCLASSED: 1, "SHFL.BFLY": 1}, True)
    closing = {("sm_120", UNCLASSED), ("sm_120", "SHFL.BFLY"), ("sm_120", "LOP3.LUT"), ("sm_90", "LOP3.LUT")}
    assert set(stale([st, other], gaps([st, other], [p]), closing)) == {("sm_120", "LOP3.LUT")}
    assert set(stale([other], gaps([other], [p]), closing)) == {("sm_120", "SHFL.BFLY"), ("sm_120", "LOP3.LUT")}


def test_the_cover_reads_every_source():
    """Each kind of program is read (so a moved file can't silently drop one), each declared catch-all is priced at its
    statement's π*, and each CLOSING entry names a device the model has."""
    progs = programs()
    names = [p.name for p in progs]
    assert any("/honest/" in n for n in names) and any("/designs/" in n for n in names)
    assert any(n.startswith("attack ") for n in names) and any(n.startswith("facet_break ") for n in names)
    assert any(n.startswith("cpw_routes ") for n in names) and any(n.startswith("rt3_dc ") for n in names)
    assert {p.key for p in progs if p.name.startswith("red_generic ")} == {"sm_90", "sm_100"}
    for st in statements():
        assert st.catch_all is None or st.price.get(st.catch_all, 0) > 0, (st.label, st.catch_all)
    assert {k for k, _ in CLOSING} <= set(devices())


@pytest.mark.slow
def test_every_instruction_is_priced_at_each_statements_certificate():
    """No instruction of a bit-exact program is free at a headline statement's π*, past CLOSING's entries."""
    open_ = [g for (_, item), g in gaps(statements(), programs()).items() if (g.st.key, item) not in CLOSING]
    assert not open_, "\n".join(fix(g) for g in sorted(open_, key=lambda g: (g.st.vector, g.item)))


@pytest.mark.slow
def test_each_closing_entry_is_still_a_gap():
    """A CLOSING entry whose instruction now has its own class priced at every statement's π* is closed: delete it."""
    sts = statements()
    done = stale(sts, gaps(sts, programs()), CLOSING)
    assert not done, f"closed, delete from CLOSING in tests/test_census_cover.py: {done}"
