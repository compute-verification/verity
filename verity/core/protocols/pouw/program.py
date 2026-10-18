"""PoUW's audit window as one static Program: its units, the positions each reads and writes, the commit phase each
position is cut at, and the shape check that refuses a layout the padded-credit rule doesn't cover (`problems`).  Every
unit and position is a function of the public shape (`window.Call`) and the tile template's depth, so the check runs
once per shape, before any value exists, and reads no value.

The rule is PADCREDIT's (`note:compute-accounting/20261018T2200Z-draft-padded-credit-rule`, §8 and §9), with W5's
round-3 findings (`note:compute-accounting/20261018T2310Z-finding-padded-credit-r3`):

- **The commit order** (`PHASES`).  The registration holds each weight row B_j with ok_B,j, and each call's
  S_B = Σ_j ok_B,j; then the salt; then the epoch's B formings, whose anchors derive from seed_b = H(salt ‖ root_B);
  then the activation commitment, one leaf per slot holding its raw row x_i, ok_A,i and admit_i, which the A side's
  noise seed hashes; then that noise; then the A formings; then the tiles and the tally; then the draw.  A gate is cut
  before the noise of the operand it gates (`late-selection`).
- **The formings** are rowk's pieces at the template's depth (`rowk.PORTS8`, cut every `rowk.ROW_K`): each piece reads
  only its raw row, anchors and its row's earlier pieces' outputs, and the tail writes the formed row a tile reads
  (`unclosed-forming`, `operand-after-certificate`).  An anchor is a public input region the verifier recomputes under
  its rule's id (`rowk.DERIVATIONS8`).  Under h3, E_A is no anchor, since its seed hashes the row's unsalted digest, so
  the A head derives it in its body (`Pc8RowAHeadSeeded`; `anchored-ea`).
- **The tiles** read the formed rows, A's from the window's side tree and B's from the epoch's B̃ tree, whose domain
  binds the salt and root_W (`stale-b-rows`).  A tile's credit is gated by ok_A, admit_A and ok_B, which the tally reads
  through `one`.  The filler's rows and gates are constant zeros, never committed rows (`filler`).
- **The always-proved set** (`missing-proved`): every ok and admission unit, every A forming piece and every tally unit,
  in every window; the B formings once per epoch, either always proved or drawn in proportion to their public fan-out,
  the static count of credited tiles their output reaches plus one (`B_PLANS`; Daniel, 22:42Z).  Tiles are drawn
  uniformly over the window's N; no draw reads a value (`draw`, `fan-out`).
- **The tally** (`tally`) is a Program of limb units, each output at most 16 bits: per call S_A in one limb, S_A·S_B in
  two, a running sum in three (real < 2^40), then the level.  No unit of the window takes advice (`advice`), and a
  tally unit reads only gates, registered sums and earlier tally units' limbs (`tally-reads`).

`DEFINITIONS` names every Definition a layout may use, by id, and `TO_BUILD` the ones not on main yet.  The window on
main is `current`: its tile is `Pc8TileSide` over side rows no proved unit writes, with ok and admit in the side row,
after the noise, and `problems` refuses it.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import pc8, rowk

PHASES = ("registration", "salt", "epoch", "activation", "noise", "forming", "tiles", "draw")
_AT = {p: i for i, p in enumerate(PHASES)}
HASHINGS = ("h0", "h3")
B_PLANS = ("proved", "weighted")
PLANS = ("proved", "weighted", "drawn")
KINDS = ("ok", "admit", "form", "tile", "tally")
LIMB = 16
#: the trees a formed row may sit in: A's in the window's side tree, B's in the epoch's B̃ tree (domain: salt, root_W)
WINDOW_A, EPOCH_B = "window-a", "epoch-b"

_PIECES = {f"row_{s}_{p}": d for s, ds in (("a", (rowk.Pc8RowAHeadDef, rowk.Pc8RowAFirstDef, rowk.Pc8RowAFormDef,
                                                   rowk.Pc8RowATailDef)),
                                           ("b", (rowk.Pc8RowBHeadDef, rowk.Pc8RowBFirstDef, rowk.Pc8RowBFormDef,
                                                  rowk.Pc8RowBTailDef)))
           for p, d in zip(("head", "first", "form", "tail"), ds)}
SEEDED_HEAD = "Pc8RowAHeadSeeded"
OK_UNIT, ADMIT_UNIT = "Pc8RowAOk", "PouwQueueAdmit"
TALLY = ("PouwTallySum", "PouwTallyProduct", "PouwTallyAdd", "PouwTallyLevel")
TILES = (pc8.Pc8TileHiddenDef.name, pc8.Pc8TileHiddenH3Def.name, pc8.Pc8TileSideDef.name)
#: every Definition a layout may name, by id, with its kind; a forming's noise is its anchors or, in-body, its own
DEFINITIONS = ({d.name: "form" for d in _PIECES.values()} | {SEEDED_HEAD: "form", OK_UNIT: "ok", ADMIT_UNIT: "admit"}
               | {t: "tally" for t in TALLY} | {t: "tile" for t in TILES})
#: the Definitions that derive their noise in their own body
IN_BODY = frozenset({SEEDED_HEAD})
#: the Definitions this layout names that main doesn't have yet
TO_BUILD = (SEEDED_HEAD, OK_UNIT, ADMIT_UNIT) + TALLY


@dataclass(frozen=True)
class Pos:
    """A committed position: its commit phase, its writer (a unit's name; None for a raw row, a registered value or a
    constant), whether the registration holds it, its constant value (the filler's), the tree it sits in, the raw row a
    gate certifies, and the registered gates a registered sum covers."""
    phase: str
    writer: str | None = None
    registered: bool = False
    constant: int | None = None
    tree: str = ""
    certifies: str | None = None
    sums: tuple[str, ...] = ()


@dataclass(frozen=True)
class Unit:
    """One unit of the window's Program: its Definition's id and kind, the positions it reads and writes, its anchor
    regions (region, rule id), its advice ports, its plan (always proved, drawn by its public fan-out, or in the uniform
    draw over the tiles) and, for a weighted unit, that fan-out; a tile's gates (gate, the operand it gates or None) and
    its floor operands."""
    definition: str
    kind: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    anchors: tuple[tuple[str, str], ...] = ()
    advice: tuple[str, ...] = ()
    plan: str = "proved"
    fan_out: int = 0
    gates: tuple[tuple[str, str | None], ...] = ()
    floor: tuple[str, ...] = ()


@dataclass(frozen=True)
class Layout:
    """A window's Program: its hashing, its positions and units by name, the law of the draw over its tiles, and which
    tiles credit (every call's; not the filler)."""
    hashing: str
    positions: Mapping[str, Pos]
    units: Mapping[str, Unit]
    draw: str = "uniform"
    filler: str = "filler"


# ---- the layout ----------------------------------------------------------------------------------------------------------

_DERIVATION = {d.rule: d for d in rowk.DERIVATIONS8.values()}
#: each anchor seed's phase: seed_a hashes the activation commitment's root, seed_b = H(salt ‖ root_B) the epoch's
_SEED_PHASE = {"seed_a": "noise", "seed_b": "epoch"}


def _rule(region: str) -> str:
    return rowk.DERIVATIONS8[region].rule


def _row(pos: dict, units: dict, side: str, row: str, k: int, ks: int, hashing: str, plan: str, fan_out: int) -> str:
    """Row `row`'s forming pieces on `side`: their units and positions, after its raw row `{row}.x`; the formed row's
    position."""
    phase = "epoch" if side == "b" else "forming"
    tree = EPOCH_B if side == "b" else WINDOW_A
    segs = k // ks
    head = f"{row}.head"
    for j in range(segs):
        pos[f"{row}.codes{j}"] = Pos(phase, f"{row}/form{j}")
        pos[f"{row}.carry{j}"] = Pos(phase, f"{row}/form{j}")
    pos[head] = Pos(phase, f"{row}/head")
    pos[f"{row}.p"] = Pos(phase, f"{row}/tail", tree=tree)
    pos[f"{row}.formed-ok"] = Pos(phase, f"{row}/tail")
    seeded = side == "a" and hashing == "h3"
    e = () if seeded else (("e", _rule("ea" if side == "a" else "eb")),)
    own, other = ("fa", "fb") if side == "a" else ("fb", "fa")
    piece = lambda role, reads, writes, anchors, name=None: Unit(
        name or _PIECES[f"row_{side}_{role}"].name, "form", reads, writes, anchors, plan=plan,
        fan_out=fan_out if plan == "weighted" else 0)
    units[f"{row}/head"] = piece("head", (f"{row}.x",), (head,), e, SEEDED_HEAD if seeded else None)
    slices = ((own, _rule(f"{own}-slice")), (other, _rule(f"{other}-slice")))
    for j in range(segs):
        carry = (f"{row}.carry{j - 1}",) if j else ()
        units[f"{row}/form{j}"] = piece("first" if j == 0 else "form", (f"{row}.x", head) + carry,
                                        (f"{row}.codes{j}", f"{row}.carry{j}"), slices)
    g = (("g", _rule("g")),) if side == "b" else ()
    units[f"{row}/tail"] = piece("tail", (head, *(f"{row}.codes{j}" for j in range(segs)), f"{row}.carry{segs - 1}"),
                                 (f"{row}.p", f"{row}.formed-ok"), g)
    return f"{row}.p"


def layout(shape: Sequence[Wd.Call], k: int, row_k: int = rowk.ROW_K, hashing: str = "h3", b_plan: str = "proved",
           tile: str | None = None) -> Layout:
    """The window's Program over `shape` at depth `k` (row cut `row_k`) under `hashing`, its B formings `b_plan`: the
    layout `problems` holds to the rule.  `tile` is the tile's Definition (by default the hiding tile of `hashing`)."""
    if hashing not in HASHINGS or b_plan not in B_PLANS or not (row_k > 0 and k % row_k == 0):
        raise ValueError(f"a layout takes hashing in {HASHINGS}, b_plan in {B_PLANS} and row_k dividing k, not "
                         f"{hashing!r}, {b_plan!r}, {row_k!r} at k = {k!r}")
    tile = tile or (pc8.Pc8TileHiddenDef.name if hashing == "h0" else pc8.Pc8TileHiddenH3Def.name)
    pos: dict[str, Pos] = {}
    units: dict[str, Unit] = {}
    for c, call in enumerate(shape):
        sb = f"sb{c}"
        for j in range(call.n):
            b = f"b{c}.{j}"
            pos[f"{b}.x"] = Pos("registration", registered=True, tree="W")
            pos[f"{b}.ok"] = Pos("registration", registered=True, tree="W", certifies=f"{b}.x")
            _row(pos, units, "b", b, k, row_k, hashing, b_plan, call.m + 1)
        pos[sb] = Pos("registration", registered=True, tree="W", sums=tuple(f"b{c}.{j}.ok" for j in range(call.n)))
        gates_a = []
        for i in range(call.m):
            a = f"a{c}.{i}"
            pos[f"{a}.x"] = Pos("activation", tree="A")
            pos[f"{a}.ok"] = Pos("activation", f"{a}/ok", tree="A", certifies=f"{a}.x")
            pos[f"{a}.admit"] = Pos("activation", f"{a}/admit", tree="A")
            units[f"{a}/ok"] = Unit(OK_UNIT, "ok", (f"{a}.x",), (f"{a}.ok",))
            units[f"{a}/admit"] = Unit(ADMIT_UNIT, "admit", (), (f"{a}.admit",))
            _row(pos, units, "a", a, k, row_k, hashing, "proved", 0)
            gates_a += [f"{a}.ok", f"{a}.admit"]
        for i in range(call.m):
            for j in range(call.n):
                a, b, t = f"a{c}.{i}", f"b{c}.{j}", f"t{c}.{i}.{j}"
                pos[f"{t}.leaf"] = Pos("tiles", t)
                units[t] = Unit(tile, "tile", (f"{a}.p", f"{b}.p"), (f"{t}.leaf",), plan="drawn",
                                gates=((f"{a}.ok", f"{a}.p"), (f"{a}.admit", f"{a}.p"), (f"{b}.ok", f"{b}.p")),
                                floor=(f"{a}.p", f"{b}.p"))
        units[f"sum{c}"] = Unit(TALLY[0], "tally", tuple(gates_a), (f"sa{c}",))
        pos[f"sa{c}"] = Pos("tiles", f"sum{c}")
        units[f"prod{c}"] = Unit(TALLY[1], "tally", (f"sa{c}", sb), (f"real{c}.0", f"real{c}.1"))
        pos[f"real{c}.0"] = pos[f"real{c}.1"] = Pos("tiles", f"prod{c}")
        prev = tuple(f"acc{c - 1}.{x}" for x in range(3)) if c else ()
        units[f"add{c}"] = Unit(TALLY[2], "tally", prev + (f"real{c}.0", f"real{c}.1"),
                                tuple(f"acc{c}.{x}" for x in range(3)))
        for x in range(3):
            pos[f"acc{c}.{x}"] = Pos("tiles", f"add{c}")
    last = len(shape) - 1
    units["level"] = Unit(TALLY[3], "tally", tuple(f"acc{last}.{x}" for x in range(3)), ("level",))
    pos["level"] = Pos("tiles", "level")
    for s in ("a", "b"):
        pos[f"filler.{s}.p"] = Pos("registration", constant=0)
        pos[f"filler.{s}.ok"] = Pos("registration", constant=0)
    pos["filler.leaf"] = Pos("tiles", "filler")
    units["filler"] = Unit(tile, "tile", ("filler.a.p", "filler.b.p"), ("filler.leaf",), plan="drawn",
                           gates=(("filler.a.ok", "filler.a.p"), ("filler.b.ok", "filler.b.p")),
                           floor=("filler.a.p", "filler.b.p"))
    return Layout(hashing, pos, units)


def current(shape: Sequence[Wd.Call], k: int) -> Layout:
    """The window as main has it: each call's side rows (fp8-pi: codes ‖ P ‖ the index field with ok and admit) are the
    committer's, cut with the tile's noise and written by no unit; the tile is `Pc8TileSide`; nothing tallies."""
    pos: dict[str, Pos] = {}
    units: dict[str, Unit] = {}
    for c, call in enumerate(shape):
        for s, count in (("a", call.m), ("b", call.n)):
            for r in range(count):
                row = f"{s}{c}.{r}"
                pos[f"{row}.p"] = Pos("forming", tree=f"window-{s}")
                pos[f"{row}.ok"] = Pos("forming", tree=f"window-{s}", certifies=f"{row}.p")
                pos[f"{row}.admit"] = Pos("forming", tree=f"window-{s}")
        for i in range(call.m):
            for j in range(call.n):
                a, b, t = f"a{c}.{i}", f"b{c}.{j}", f"t{c}.{i}.{j}"
                pos[f"{t}.leaf"] = Pos("tiles", t)
                units[t] = Unit(pc8.Pc8TileSideDef.name, "tile", (f"{a}.p", f"{b}.p"), (f"{t}.leaf",), plan="drawn",
                                gates=((f"{a}.ok", f"{a}.p"), (f"{a}.admit", f"{a}.p"), (f"{b}.ok", f"{b}.p"),
                                       (f"{b}.admit", f"{b}.p")),
                                floor=(f"{a}.p", f"{b}.p"))
    for s in ("a", "b"):
        pos[f"filler.{s}.p"] = Pos("forming", tree=f"window-{s}")
        pos[f"filler.{s}.ok"] = Pos("forming", tree=f"window-{s}")
    pos["filler.leaf"] = Pos("tiles", "filler")
    units["filler"] = Unit(pc8.Pc8TileSideDef.name, "tile", ("filler.a.p", "filler.b.p"), ("filler.leaf",),
                           plan="drawn", gates=(("filler.a.ok", "filler.a.p"), ("filler.b.ok", "filler.b.p")),
                           floor=("filler.a.p", "filler.b.p"))
    return Layout("h3", pos, units)


# ---- the shape check -----------------------------------------------------------------------------------------------------

def _readers(lay: Layout) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, u in lay.units.items():
        for p in u.reads:
            out.setdefault(p, []).append(name)
    return out


def _is(lay: Layout, p: str, *kinds: str) -> bool:
    q = lay.positions.get(p)
    return q is not None and q.writer is not None and q.writer in lay.units and lay.units[q.writer].kind in kinds


def _forming_chain(lay: Layout, operand: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The forming units that reach `operand` along writer edges inside the formings, and the positions they read that
    are no forming's output."""
    seen: list[str] = []
    outside: list[str] = []
    stack = [operand]
    while stack:
        q = lay.positions.get(stack.pop())
        w = q.writer if q is not None else None
        if w is None or w not in lay.units or lay.units[w].kind != "form" or w in seen:
            continue
        seen.append(w)
        for p in lay.units[w].reads:
            (stack if _is(lay, p, "form") else outside).append(p)
    return tuple(seen), tuple(dict.fromkeys(outside))


def _noise_phase(lay: Layout, chain: Iterable[str]) -> int:
    """The earliest phase of the row's own noise a forming chain reads: its per-row anchors' (E_A from seed_a, E_B from
    seed_b) and an in-body derivation's, at the noise.  The shared factors F_A, F_B and G, known from the epoch on, are
    no row's own."""
    at = [_AT[_SEED_PHASE[_DERIVATION[rule].seed]] for u in chain for _r, rule in lay.units[u].anchors
          if rule in _DERIVATION and _DERIVATION[rule].per_row]
    at += [_AT["noise"] for u in chain if lay.units[u].definition in IN_BODY]
    return min(at, default=len(PHASES))


def _cones(lay: Layout, readers: Mapping[str, list[str]]) -> dict[str, int]:
    """Each weighted unit's static cone: the credited tiles its output reaches along forming edges, plus itself."""
    out = {}
    for name, u in lay.units.items():
        if u.plan != "weighted":
            continue
        tiles, stack, seen = set(), list(u.writes), set()
        while stack:
            for r in readers.get(stack.pop(), ()):
                if r in seen:
                    continue
                seen.add(r)
                if lay.units[r].kind == "tile" and r != lay.filler:
                    tiles.add(r)
                elif lay.units[r].kind == "form":
                    stack += lay.units[r].writes
        out[name] = len(tiles) + 1
    return out


def problems(lay: Layout) -> list[str]:
    """Every reason the shape check refuses `lay`, each ``window-form: <kind>: <where>``; empty when it holds the rule.
    It reads the layout only: Definition ids, positions, phases and plans."""
    out: list[str] = []
    bad = lambda kind, where: out.append(f"window-form: {kind}: {where}")
    if lay.hashing not in HASHINGS:
        bad("hashing", f"{lay.hashing!r} is none of {HASHINGS}")
    if lay.draw != "uniform":
        bad("draw", f"the tiles' draw is {lay.draw!r}, not uniform over the N tiles")
    readers = _readers(lay)
    for name, u in lay.units.items():
        if DEFINITIONS.get(u.definition) != u.kind:
            bad("unknown-definition", f"{name}: {u.definition} is no {u.kind} Definition of the table")
            continue
        if u.advice:
            bad("advice", f"{name} takes advice {list(u.advice)}")
        want = {"tile": ("drawn",), "form": B_PLANS}.get(u.kind, ("proved",))
        if u.plan not in want:
            bad("missing-proved" if u.kind != "tile" else "plan", f"{name} ({u.definition}) is {u.plan}, not {' or '.join(want)}")
        for r, rule in u.anchors:
            if rule not in _DERIVATION:
                bad("unproved-noise", f"{name}: region {r} names no anchor rule ({rule})")
            elif lay.hashing == "h3" and rule == _rule("ea"):
                bad("anchored-ea", f"{name}: under h3 E_A's seed hashes the row's unsalted digest, so it is no anchor")
        if u.kind == "tally":
            for p in u.reads:
                q = lay.positions.get(p)
                if not (q is not None and (q.registered and (q.sums or q.certifies)) or _is(lay, p, "ok", "admit", "tally")):
                    bad("tally-reads", f"{name} reads {p}, neither a gate, a registered sum nor a tally limb")
    for name, cone in _cones(lay, readers).items():
        if lay.units[name].fan_out != cone:
            bad("fan-out", f"{name}: its draw weight {lay.units[name].fan_out} is not its static cone {cone}")
    tallied = {p for u in lay.units.values() if u.kind == "tally" for p in u.reads}
    covered = {g for p in tallied if p in lay.positions for g in lay.positions[p].sums}
    chains: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for name, u in lay.units.items():
        if u.kind != "tile":
            continue
        filler = name == lay.filler
        for x, operand in enumerate(u.floor):
            q = lay.positions.get(operand)
            if filler:
                if q is None or q.constant != 0:
                    bad("filler", f"the filler's operand {operand} is not the constant zero row")
            elif x == 1 and (q is None or q.tree != EPOCH_B):
                bad("stale-b-rows", f"{name}: its B operand {operand} is not in the epoch's B̃ tree")
        for g, operand in u.gates:
            q = lay.positions.get(g)
            if filler:
                if q is None or q.constant != 0:
                    bad("filler", f"the filler's gate {g} is not the constant 0")
                continue
            if q is None or not (q.registered or (_is(lay, g, "ok", "admit") and lay.units[q.writer].plan == "proved")):
                bad("unsourced-gate", f"{name}: gate {g} is neither registered nor an always-proved unit's output")
                if q is None:
                    continue
            if g not in tallied and g not in covered:
                bad("untallied-gate", f"{name}: no tally unit reads gate {g}")
            if operand not in chains:
                chains[operand] = _forming_chain(lay, operand)
            chain, outside = chains[operand]
            if not chain:
                bad("operand-after-certificate", f"{name}: its operand {operand} is written by no forming")
                continue
            if q.certifies is not None:
                if q.certifies not in outside:
                    bad("operand-after-certificate", f"{name}: no forming of {operand} reads {q.certifies}, the row "
                                                     f"{g} certifies")
                stray = [p for p in outside if p != q.certifies]
                if stray:
                    bad("unclosed-forming", f"{name}: a forming of {operand} reads {stray}, neither its raw row, an "
                                            "anchor nor a forming's output")
            noise = _noise_phase(lay, chain)
            if noise == len(PHASES):
                bad("unnoised", f"{name}: the forming of {operand} reads no noise of its row")
            elif _AT[q.phase] >= noise:
                bad("late-selection", f"{name}: gate {g} is cut at {q.phase}, not before its operand's noise "
                                      f"({PHASES[noise]})")
    return list(dict.fromkeys(out))


def problem(lay: Layout) -> str | None:
    """The shape check's first refusal of `lay`, or None."""
    p = problems(lay)
    return p[0] if p else None


# ---- the tally's limb Program --------------------------------------------------------------------------------------------

def limbs(x: int, count: int) -> tuple[int, ...]:
    """x in `count` 16-bit limbs, low first; raises ValueError when it doesn't fit."""
    if not (Wd._int(x) and 0 <= x < 1 << (LIMB * count)):
        raise ValueError(f"{x!r} is not {count} {LIMB}-bit limbs")
    return tuple(x >> (LIMB * i) & ((1 << LIMB) - 1) for i in range(count))


def _value(ls: Sequence[int]) -> int:
    return sum(v << (LIMB * i) for i, v in enumerate(ls))


def one(word: int) -> int:
    """A gate word read as a bit: 1 exactly when the word is 1, so a wrong writer's 2 counts as 0."""
    return int(word == 1)


def tally(calls: Sequence[Mapping[str, object]], bits: int = Wd.GRID_BITS) -> list[dict[str, object]]:
    """The tally's limb units on the gates, in order: per call ``sum`` (S_A = Σ_i one(ok_A,i)·one(admit_i), one limb),
    ``product`` (S_A·S_B in two limbs, S_B the registered sum's two) and ``add`` (the running sum in three), then
    ``level`` (the level of the masked sum, 0 refused).  Each call is ``{"a": [(ok, admit), ...], "sb": S_B}``; each
    unit is ``{"unit", "reads", "writes"}`` of 16-bit words.  Raises ValueError for a call whose counts don't fit."""
    out: list[dict[str, object]] = []
    acc: tuple[int, ...] = (0, 0, 0)
    for c, call in enumerate(calls):
        pairs = list(call["a"])  # type: ignore[arg-type]
        sa = sum(one(ok) * one(ad) for ok, ad in pairs)
        sb = limbs(int(call["sb"]), 2)  # type: ignore[arg-type]
        out.append({"unit": TALLY[0], "reads": [w for p in pairs for w in p], "writes": list(limbs(sa, 1))})
        out.append({"unit": TALLY[1], "reads": [sa, *sb], "writes": list(limbs(sa * _value(sb), 2))})
        new = limbs(_value(acc) + sa * _value(sb), 3)
        out.append({"unit": TALLY[2], "reads": [*(acc if c else ()), *limbs(sa * _value(sb), 2)], "writes": list(new)})
        acc = new
    real = _value(acc)
    out.append({"unit": TALLY[3], "reads": list(acc),
                "writes": [Wd.level(Wd.floor_point(real, bits), bits) if real else 0, int(real >= 1)]})
    return out


def tally_problem(units: Sequence[Mapping[str, object]], calls: int, bits: int = Wd.GRID_BITS) -> str | None:
    """Why `units` is not the tally's limb Program over `calls` calls, or None: each unit's writes are 16-bit words and
    its Definition's function of its reads, and each unit reads the limbs the one before it wrote."""
    if len(units) != 3 * calls + 1:
        return f"{len(units)} tally units, not 3 per call and the level"
    acc: list[int] = [0, 0, 0]
    for c in range(calls):
        s, p, a = units[3 * c:3 * c + 3]
        if (s["unit"], p["unit"], a["unit"]) != TALLY[:3]:
            return f"call {c}: its units are not {TALLY[:3]}"
        words = [w for u in (s, p, a) for w in list(u["reads"]) + list(u["writes"])]  # type: ignore[arg-type]
        if any(not (Wd._int(w) and 0 <= w < 1 << LIMB) for w in words):
            return f"call {c}: a word is not 16 bits"
        reads = list(s["reads"])  # type: ignore[arg-type]
        if len(reads) % 2 or list(s["writes"]) != [sum(one(reads[x]) * one(reads[x + 1]) for x in range(0, len(reads), 2))]:
            return f"call {c}: its sum is not Σ one(ok)·one(admit) over its slots"
        sa, sb = list(p["reads"])[0], _value(list(p["reads"])[1:])  # type: ignore[arg-type]
        if sa != list(s["writes"])[0] or list(p["writes"]) != list(limbs(sa * sb, 2)):  # type: ignore[arg-type]
            return f"call {c}: its product is not S_A·S_B in two limbs"
        want = (acc if c else []) + list(p["writes"])  # type: ignore[arg-type]
        if list(a["reads"]) != want:  # type: ignore[arg-type]
            return f"call {c}: its add doesn't read the running sum's limbs and the product's"
        total = _value(acc) + _value(list(p["writes"]))  # type: ignore[arg-type]
        if total >= 1 << (3 * LIMB) or list(a["writes"]) != list(limbs(total, 3)):  # type: ignore[arg-type]
            return f"call {c}: its add is not the running sum in three limbs"
        acc = list(a["writes"])  # type: ignore[arg-type]
    lv = units[-1]
    real = _value(acc)
    want_level = [Wd.level(Wd.floor_point(real, bits), bits) if real else 0, int(real >= 1)]
    if lv["unit"] != TALLY[3] or list(lv["reads"]) != acc or list(lv["writes"]) != want_level:  # type: ignore[arg-type]
        return "its level unit is not the level of the running sum"
    return None


# ---- the cost of each B plan ---------------------------------------------------------------------------------------------

def cost(shape: Sequence[Wd.Call], k: int, eps: Fraction, delta: Fraction, row_k: int = rowk.ROW_K,
         windows: int = 1) -> dict[str, object]:
    """The proofs a window and its salt take, for each B plan, at full credit (every slot and weight row selected),
    `windows` windows a salt: each forming is 2 + k/row_k piece proofs.  Proved: every B forming once a salt.  Weighted:
    the B formings join the draw at their static cone (m_c + 1), so each window draws K″ over N + Σ n_c·(m_c + 1)."""
    pieces = 1 + k // row_k + 1
    n_tiles = Wd.shape_tiles(shape)
    real = sum(c.m * c.n for c in shape)
    credited = Wd.credit(real, n_tiles).credited
    k_tiles = Wd.k_prime(eps, delta, credited, n_tiles, 1)
    rows_a, rows_b = sum(c.m for c in shape), sum(c.n for c in shape)
    cone = sum(c.n * (c.m + 1) for c in shape)
    k_joint = Wd.k_prime(eps, delta, credited, n_tiles + cone, 1)
    b_draws = k_joint * Fraction(cone, n_tiles + cone)
    window_fixed = {"ok": rows_a, "admit": rows_a, "a_pieces": rows_a * pieces, "tally": 3 * len(shape) + 1}
    return {
        "n": n_tiles, "credited": credited, "k_tiles": k_tiles, "pieces_per_row": pieces, "rows_b": rows_b,
        "window_fixed": window_fixed,
        "proved": {"tiles_per_window": k_tiles, "b_pieces_per_salt": rows_b * pieces,
                   "b_pieces_per_window": Fraction(rows_b * pieces, windows)},
        "weighted": {"draws_per_window": k_joint, "tiles_per_window": k_joint - b_draws,
                     "b_formings_per_window": b_draws, "b_pieces_per_window": b_draws * pieces},
    }
