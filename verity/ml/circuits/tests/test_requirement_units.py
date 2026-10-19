"""`requirement_units`: the plan unit and the memory-map unit on a toy decoder's requirement (eight weight blocks, the last read
only through step 1, a gathered table of two rows a block, three requests, one paused), under `requirement.pack`'s layout and a
layout of one value a block, held to their references and to `requirement.plan_matches` and `store_matches`; and each clause's
negative refused."""

from __future__ import annotations

import dataclasses
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits import requirement as R
from verity.ml.circuits import requirement_units as RU
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch

PLAN = SimpleNamespace(P=16, NR=16, V=16, G=4, TB=8, RPB=2)
MAP = SimpleNamespace(K=2, C=2, V=16, M=8, N=16, SLOTS=2)
VW = RU.value_bits(MAP.SLOTS)
STEPS = 4
WEIGHTS = list(range(8))
#: a weight block only the first steps read (an expert's): its registered block is freed at its last read
READ_UNTIL = {7: 1}
TABLE = list(range(8, 24))
#: request: (its steps, the table row it gathers at each); C is paused at step 2
REQUESTS = {"A": ({0: 3, 1: 5, 2: 0, 3: 7}), "B": ({0: 9, 1: 2}), "C": ({1: 4, 3: 11})}


def _value(label: str, bits: int) -> int:
    return int.from_bytes(hashlib.shake_256(label.encode()).digest(bits // 8), "little")


def toy():
    """The requirement, each written value's id, and the table rows each step gathers."""
    width, born, last, uses, values = {}, {}, {}, {s: set() for s in range(STEPS)}, {}
    for v in WEIGHTS:
        width[v], born[v], last[v], values[v] = R.BLOCK_BITS, -1, READ_UNTIL.get(v, STEPS - 1), _value(f"w{v}", R.BLOCK_BITS)
    for r, v in enumerate(TABLE):
        width[v], born[v], last[v], values[v] = VW, -1, STEPS - 1, _value(f"row{r}", VW)
    kv, gathers, nxt = {}, {s: [] for s in range(STEPS)}, 100
    for name, steps in REQUESTS.items():
        mine = []
        for s in sorted(steps):
            uses[s] |= {w for w in WEIGHTS if s <= last[w]} | {TABLE[steps[s]]} | set(mine)
            gathers[s].append(steps[s])
            for v in mine:
                last[v] = s
            if s < max(steps):
                kv[nxt] = (name, s)
                width[nxt], born[nxt], values[nxt] = VW, s, _value(f"{name}{s}", VW)
                mine.append(nxt)
                nxt += 1
    req = R.Requirement(width, frozenset(WEIGHTS + TABLE), born, last, {s: frozenset(u) for s, u in uses.items()}, STEPS, values)
    positions = sorted(kv, key=lambda v: (born[v], v))
    return req, positions, {s: sorted(g) for s, g in gathers.items()}


def spread(req: R.Requirement, packed: R.Layout, positions):
    """Each written value alone in a fresh block of its step, slot 0, freed at its last read."""
    block, offset = dict(packed.block), dict(packed.offset)
    written, freed, payload, j = {}, {}, {k: v for k, v in R.pack(req)[1].payload.items() if k < MAP.N}, MAP.N
    for v in positions:
        block[v], offset[v] = j, 0
        written.setdefault(req.born[v], set()).add(j)
        freed.setdefault(req.last[v], set()).add(j)
        payload[j] = req.values[v]
        j += 1
    for v in req.registered:
        freed.setdefault(req.last[v], set()).add(block[v])
    return R.Layout(block, offset), R.Store(MAP.N, {s: frozenset(b) for s, b in written.items()},
                                           {s: frozenset(b) for s, b in freed.items()}, payload)


def _bits(words, shape) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64).reshape(shape[:-1])
    return ((w[..., None] >> np.arange(16, dtype=np.uint64)) & 1).astype(np.uint8)


def evaluate(defn, S, batch: list[dict]) -> list[int]:
    """The Definition's ok bit on each instance's arguments, bit-sliced, with bits 1 .. 15 checked zero."""
    params, _ret = defn.signature()
    vww = RU.value_bits(S.SLOTS) // 16 if hasattr(S, "SLOTS") else 0
    arrays = []
    for name, t in params:
        rows = []
        for a in batch:
            v = a[name]
            if name == "val":
                v = [RU._words(x, vww) for x in v]
            elif name == "pay":
                v = [RU._words(x, RU.BLOCK_WORDS) for x in v]
            rows.append(_bits(v, (*np.shape(v), 16) if np.ndim(v) else (16,)))
        arrays.append(np.stack(rows))
    out = np.asarray(evaluate_batch(defn, *arrays)).reshape(len(batch), 16)
    assert not out[:, 1:].any()
    return [int(x) for x in out[:, 0]]


@pytest.fixture(scope="module")
def case():
    req, positions, gathers = toy()
    packed, packed_store = R.pack(req)
    layouts = {"pack": (packed, packed_store), "spread": spread(req, packed, positions)}
    reg_block = {v: packed.block[v] for v in req.registered}
    return SimpleNamespace(req=req, positions=positions, gathers=gathers, layouts=layouts, reg_block=reg_block,
                           plan=bind(RU.PlanMatch, **vars(PLAN)), map=bind(RU.MemoryMapStep, **vars(MAP)))


def plan_args(c, layout, t, plan):
    lam = RU.layout_words(MAP, c.req, layout, c.positions)
    gathered = frozenset(TABLE[r] for r in c.gathers[t])
    return RU.plan_inputs(PLAN, c.req, t, plan, c.reg_block, c.gathers[t], gathered, c.positions, lam)


def map_args(c, layout, store, t):
    lam, phi = RU.layout_words(MAP, c.req, layout, c.positions), RU.map_words(MAP, store)
    return RU.map_inputs(MAP, c.req, store, t, c.positions, lam, phi, c.reg_block)


def ref_plan(a) -> int:
    return RU.plan_match(PLAN, a["count"], a["regmask"], a["kvmask"], a["gmask"], a["lam"], a["plan"], a["tok"])


def ref_map(a) -> int:
    return RU.memory_map(MAP, a["t"], a["first"], a["count"], a["present"], a["pos"], a["last"], a["rfree"], a["lam"], a["phi"],
                         a["val"], a["pay"])


def test_the_toy_is_what_the_units_cover(case) -> None:
    """Both layouts are honest (`store_matches` and `plan_matches` at every step), the spread one's blocks are half empty, and
    each step's plan is its weights, its gathered rows' blocks and its written values' blocks: C's block is resident at step
    2, when C is paused, and no plan holds it. Weight block 7 is freed at step 1, its last read, in both stores and in the map
    row, the only registered free before the last step."""
    req = case.req
    for layout, store in case.layouts.values():
        for t in range(STEPS):
            assert R.store_matches(req, layout, store, R.BLOCK_BITS, t)
            assert R.plan_matches(req, layout, R.read_blocks(req, layout, t), R.BLOCK_BITS, t)
        assert 7 in store.freed[1] and 7 not in store.resident(2) and 7 in R.read_blocks(req, layout, 1)
        assert RU.map_words(MAP, store)[:MAP.N] == [1 if j == 7 else STEPS - 1 for j in range(MAP.N)]
        assert RU.registered_frees(req, case.reg_block) == {j: 1 if j == 7 else STEPS - 1 for j in range(MAP.N)}
    layout, store = case.layouts["spread"]
    c1 = next(v for v in case.positions if req.born[v] == 1 and req.last[v] == 3 and layout.block[v] not in
              R.read_blocks(req, layout, 2))
    assert layout.block[c1] in store.resident(2)
    assert max(len(R.read_blocks(req, layout, t)) for t in range(STEPS)) <= PLAN.P


def test_each_step_s_honest_plan_is_accepted(case) -> None:
    """Under either layout, every step's plan unit (its plan the blocks the step reads) returns 1, as its reference does."""
    plans = [plan_args(case, layout, t, R.read_blocks(case.req, layout, t)) for layout, _store in case.layouts.values()
             for t in range(STEPS)]
    assert evaluate(case.plan, PLAN, plans) == [ref_plan(a) for a in plans] == [1] * len(plans)


@pytest.mark.slow
def test_each_step_s_honest_memory_map_is_accepted(case) -> None:
    """Under either layout, every step's memory-map unit returns 1, as its reference does."""
    maps = [map_args(case, layout, store, t) for layout, store in case.layouts.values() for t in range(STEPS)]
    assert evaluate(case.map, MAP, maps) == [ref_map(a) for a in maps] == [1] * len(maps)


def test_a_plan_with_a_wrong_block_is_refused(case) -> None:
    """A plan holding a block the step doesn't read (a paused request's, a table block no gather selects, a registered block
    past its last read), missing one it reads, out of order, or past its count returns 0, and so does a gather outside the
    table. The unit asks for exactly the blocks the step reads, which implies `plan_matches`: that one-sided link refuses
    the plans missing a block the step reads and allows the extra blocks as filler."""
    req, (layout, _store) = case.req, case.layouts["spread"]
    t = 2
    honest = sorted(R.read_blocks(req, layout, t))
    paused = next(layout.block[v] for v in case.positions if req.born[v] == 1 and layout.block[v] not in honest)
    unselected = next(PLAN.TB + r // PLAN.RPB for r in range(16) if PLAN.TB + r // PLAN.RPB not in honest)
    plans = {"paused-request-block": sorted(honest + [paused]), "unselected-table-block": sorted(honest + [unselected]),
             "freed-registered-block": sorted(honest + [7]), "swapped": sorted(honest[:-1] + [paused]),
             "missing": honest[:-1]}
    batch = []
    for name, plan in plans.items():
        assert R.plan_matches(req, layout, plan, R.BLOCK_BITS, t) == (name not in ("swapped", "missing")), name
        batch.append(plan_args(case, layout, t, plan))
    a = plan_args(case, layout, t, honest)
    batch.append({**a, "plan": [a["plan"][1], a["plan"][0]] + a["plan"][2:]})
    batch.append({**a, "plan": a["plan"][:a["count"]] + [5] + a["plan"][a["count"] + 1:]})
    batch.append({**a, "tok": [16 * PLAN.RPB] + a["tok"][1:]})
    batch.append({**a, "count": PLAN.P + 1})
    assert evaluate(case.plan, PLAN, batch) == [ref_plan(x) for x in batch] == [0] * len(batch)


def _phi(a, j, write=None, free=None):
    """``a`` with block j's write or free step in the map row replaced (a registered block has only its free)."""
    phi = list(a["phi"])
    at = j if j < MAP.N else MAP.N + 2 * (j - MAP.N) + 1
    phi[at] = phi[at] if free is None else free
    if write is not None:
        phi[at - 1] = write
    return {**a, "phi": phi}


@pytest.mark.slow
def test_each_memory_map_clause_s_negative_is_refused(case) -> None:
    """A payload that doesn't carry its value, a free after its values' last read (late) or before it (early), of a written
    block or of a registered one (weight block 7, whose last read is step 1), a write at another step, a block written empty,
    nonzero bits in an empty slot, two values in one slot, a value outside the step's blocks: each returns 0 as the reference
    does. The early frees and the mismatch each break `store_matches` at a step, and the late frees don't: the one-sided
    link allows a block resident past its values' last read."""
    req = case.req
    layout, store = case.layouts["spread"]
    t = 0
    a = map_args(case, layout, store, t)
    j0, j1 = a["first"], a["first"] + 1
    k = min(range(2), key=lambda i: a["last"][i])
    jk, lastk = a["lam"][2 * a["pos"][k]], a["last"][k]
    assert lastk < STEPS - 1
    pay = list(a["pay"])
    pay[0] ^= 1 << 77
    pad = list(a["pay"])
    pad[1] |= 1 << (VW + 3)
    lam = list(a["lam"])
    lam[2 * a["pos"][1]], lam[2 * a["pos"][1] + 1] = j0, 0
    out = list(a["lam"])
    out[2 * a["pos"][1]] = j1 + 1
    batch = {"payload-mismatch": {**a, "pay": pay}, "late-free": _phi(a, jk, free=lastk + 1),
             "early-free": _phi(a, jk, free=lastk - 1), "write-at-another-step": _phi(a, j0, write=t + 1),
             "empty-block": {**a, "lam": lam}, "nonzero-pad": {**a, "pay": pad}, "value-outside-the-step": {**a, "lam": out},
             "present-dropped": {**a, "present": 1}}
    shared = list(a["lam"])
    shared[2 * a["pos"][1]], shared[2 * a["pos"][1] + 1] = j0, 0
    batch["two-in-one-slot"] = {**a, "lam": shared, "pay": [a["pay"][0], 0]}
    a1 = map_args(case, layout, store, 1)
    assert RU._bits(a1["rfree"], MAP.N) == [int(j == 7) for j in range(MAP.N)]
    batch["registered-late-free"] = _phi(a1, 7, free=2)
    batch["registered-early-free"] = _phi(a1, 7, free=0)
    got = evaluate(case.map, MAP, list(batch.values()))
    assert got == [ref_map(x) for x in batch.values()] == [0] * len(batch)

    def broken(fn) -> bool:
        s2 = fn(store)
        return not all(R.store_matches(req, layout, s2, R.BLOCK_BITS, s) for s in range(STEPS))

    def refree(j, at):
        def f(s):
            freed = {at_: frozenset(js - {j}) for at_, js in s.freed.items()}
            freed[at] = freed.get(at, frozenset()) | {j}
            return dataclasses.replace(s, freed=freed)
        return f

    assert not broken(refree(jk, lastk + 1)) and broken(refree(jk, lastk - 1))
    assert not broken(refree(7, 2)) and broken(refree(7, 0))
    assert broken(lambda s: dataclasses.replace(s, payload={**s.payload, j0: s.payload[j0] ^ (1 << 77)}))


def test_the_value_leaves_are_the_values_each_map_row_carries(case) -> None:
    """`map_value_leaves`, from the public row alone, names leaf i of the value commitments (written value positions[i]) for
    each value row, whose words are that value's, and block first + c for each payload row, whose words are its payload.
    Over the steps, every written value a later step reads is one value leaf, once; a public row past C blocks, past K
    values or with a position past V is refused."""
    req, rows = case.req, RU.memory_map_rows(MAP)
    for layout, store in case.layouts.values():
        seen = []
        for t in range(STEPS):
            a = map_args(case, layout, store, t)
            words = RU.row_words(MAP, rows, a)
            leaves = RU.map_value_leaves(MAP, {x: a[x] for x in ("first", "count", "present", "pos")})
            for row, (kind, at) in leaves.items():
                want = (RU._words(req.values[case.positions[at]], VW // 16) if kind == "value"
                        else RU._words(store.payload[at], RU.BLOCK_WORDS))
                assert words[row] == want, (t, row)
            assert sorted(at for kind, at in leaves.values() if kind == "block") == sorted(store.written.get(t, ()))
            seen += [at for kind, at in leaves.values() if kind == "value"]
        assert sorted(seen) == [i for i, v in enumerate(case.positions) if req.last.get(v, -1) > req.born[v]]
    a = map_args(case, *case.layouts["spread"], 0)
    for bad in ({**a, "count": MAP.C + 1}, {**a, "present": 1 << MAP.K}, {**a, "pos": [MAP.V] + a["pos"][1:]}):
        with pytest.raises(ValueError, match="a public row of at most"):
            RU.map_value_leaves(MAP, bad)


def test_the_rows_are_the_signature_s_params() -> None:
    """Each unit's rows cover its parameters once each, the public row first, in signature order."""
    for defn, S, rows in ((RU.PlanMatch, PLAN, RU.plan_rows(PLAN)), (RU.MemoryMapStep, MAP, RU.memory_map_rows(MAP))):
        params, _ = bind(defn, **vars(S)).signature()
        flat = [p.partition("[")[0] for _r, _b, ps in rows for p in ps]
        assert list(dict.fromkeys(flat)) == [n for n, _ in params]
        bits = {n: t.leaves for n, t in params}
        assert sum(b for _r, b, _ps in rows) == sum(bits.values())
        assert rows[0][0] == "pub"
