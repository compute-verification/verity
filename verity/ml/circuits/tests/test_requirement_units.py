"""`requirement_units`: the plan unit and the memory-map unit on a toy decoder's requirement (eight weight blocks, the last
read only through step 1, a gathered table of two rows a block, three requests, one paused), under `requirement.pack`'s layout and a
layout of one value a block, held to their references and to `requirement.plan_matches` and `store_matches`; the sharded plan
unit on a store of weights, a gathered table and filler past one shard; each clause's negative refused; and a re-key at the
lifetime cap, one pass a window, accepted, and one window early refused."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits import requirement as R
from verity.ml.circuits import requirement_units as RU
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch

STEPS = 4
MAP = SimpleNamespace(K=2, C=2, V=16, E=2, M=8, N=16, SLOTS=2, T=STEPS)
#: the plan unit reads the map's layout row as it is: an entry a value's epoch
PLAN = SimpleNamespace(P=16, NR=16, V=MAP.V * MAP.E, G=4, TB=8, RPB=2)
VW = RU.value_bits(MAP.SLOTS)
#: one pass a step, each its own window: a block written at step s joins window s + 1
PASSES = [(s, s) for s in range(STEPS)]
PI, LIFE = RU.pass_words(MAP, PASSES), 2
WEIGHTS = list(range(8))
#: a weight block only the first steps read (an expert's): its registered block is freed at its last structural reader
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
    """Each written value alone in a fresh block of its step, slot 0, freed at its last structural reader."""
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
    return RU.map_inputs(MAP, c.req, store, t, c.positions, lam, phi, c.reg_block, PI, LIFE)


def ref_plan(a) -> int:
    return RU.plan_match(PLAN, a["count"], a["regmask"], a["kvmask"], a["gmask"], a["lam"], a["plan"], a["tok"])


def ref_map(a) -> int:
    return RU.memory_map(MAP, a["t"], a["first"], a["count"], a["present"], a["pos"], a["ep"], a["last"], a["rfree"], a["life"],
                         a["lam"], a["phi"], a["pi"], a["val"], a["pay"])


def test_the_toy_is_what_the_units_cover(case) -> None:
    """Both layouts are honest (`store_matches` and `plan_matches` at every step), the spread one's blocks are half empty, and
    each step's plan is its weights, its gathered rows' blocks and its written values' blocks: C's block is resident at step
    2, when C is paused, and no plan holds it. Weight block 7 is freed at step 1, its last structural reader, in both stores and in the map
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
    past its values' last structural reader), missing one it reads, out of order, or past its count returns 0, and so does a gather outside the
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
    """A payload that doesn't carry its value, a free after its values' last structural reader (late) or before it (early), of a written
    block or of a registered one (weight block 7, whose last structural reader is step 1), a write at another step, a block written empty,
    nonzero bits in an empty slot, two values in one slot, a value outside the step's blocks: each returns 0 as the reference
    does. The early frees and the mismatch each break `store_matches` at a step, and the late frees don't: the one-sided
    link allows a block resident past its values' last structural reader."""
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


SHARD = SimpleNamespace(S=32, GAPS=2)
#: a store past one shard: weight blocks 0 .. 19 and 28 .. 36 (36 an expert's, read only at step 0), a gathered table of
#: two rows a block at blocks 20 .. 27, filler from 37
STORE_N, FILLER = 100, 37
TABLE_BLOCKS = range(20, 28)


def shard_toy():
    """The requirement over the sharded store, its layout, and the blocks each step's plan units name (``need``: the
    weights the step reads and the gathered table whole)."""
    weights = [*range(20), *range(28, FILLER)]
    rows = [100 + r for r in range(16)]
    width = {**{v: R.BLOCK_BITS for v in weights}, **{v: VW for v in rows}}
    last = {**{v: 0 if v == 36 else 1 for v in weights}, **{v: 1 for v in rows}}
    uses = {0: frozenset(weights) | {rows[3]}, 1: frozenset(set(weights) - {36}) | {rows[10]}}
    req = R.Requirement(width, frozenset(width), {v: -1 for v in width}, last, uses, 2, {})
    layout = R.Layout({**{v: v for v in weights}, **{v: 20 + (v - 100) // 2 for v in rows}},
                      {**{v: 0 for v in weights}, **{v: ((v - 100) % 2) * VW for v in rows}})
    need = {t: sorted({layout.block[v] for v in req.reads(t) if v in weights} | set(TABLE_BLOCKS)) for t in range(2)}
    return req, layout, need


def shards(plan, need) -> list[dict]:
    return [RU.shard_inputs(SHARD, lo, size, plan, need) for lo, size in RU.plan_shards(STORE_N, SHARD.S)]


def ref_shard(a) -> int:
    return RU.plan_shard(SHARD, a["count"], a["size"], a["regmask"], a["plan"])


@pytest.fixture(scope="module")
def shard_unit():
    return bind(RU.PlanShard, **vars(SHARD))


def test_a_sharded_plan_is_one_plan_of_the_store(shard_unit) -> None:
    """The store's shards are S blocks from 0 but the last (4 here), and a pass's plan, cut into them, is its shards'
    plans in order. A plan of the whole store, or one missing at most GAPS filler blocks a shard (and, at step 1, the
    expert's block it no longer reads), is accepted by every shard's unit, as its reference does, and is `plan_matches`
    at its step; a ``range`` is cut as its list is."""
    req, layout, need = shard_toy()
    assert RU.plan_shards(STORE_N, SHARD.S) == [(0, 32), (32, 32), (64, 32), (96, 4)]
    whole = range(STORE_N)
    assert shards(whole, need[0]) == shards(list(whole), need[0])
    plans = {0: [whole, [j for j in whole if j not in (40, 41, 70, 99)]], 1: [[j for j in whole if j not in (36, 40)]]}
    batch = []
    for t, ps in plans.items():
        for plan in ps:
            assert R.plan_matches(req, layout, plan, R.BLOCK_BITS, t)
            cut = shards(plan, need[t])
            joined = [j + lo for a, (lo, _) in zip(cut, RU.plan_shards(STORE_N, SHARD.S)) for j in a["plan"][:a["count"]]]
            assert joined == list(plan)
            batch += cut
    assert evaluate(shard_unit, SHARD, batch) == [ref_shard(a) for a in batch] == [1] * len(batch)


def test_a_shard_missing_a_block_its_step_reads_is_refused(shard_unit) -> None:
    """A plan missing a weight block the step reads, or the expert's at step 0, is refused by that block's shard and is
    not `plan_matches`. So is one missing a table block no gather selects, which `plan_matches` allows: the unit counts a
    gathered table whole, since which rows a step gathers is the circuit's values. More than GAPS blocks missing from a
    shard, all filler, is refused too: the unit's search is GAPS + 1 entries a block."""
    req, layout, need = shard_toy()
    cases = {"weight": (0, 5, False), "expert": (0, 36, False), "unselected-table-block": (0, 26, True)}
    batch = []
    for name, (t, j, lean) in cases.items():
        plan = [b for b in range(STORE_N) if b != j]
        assert R.plan_matches(req, layout, plan, R.BLOCK_BITS, t) == lean, name
        batch.append(shards(plan, need[t])[j // SHARD.S])
    plan = [b for b in range(STORE_N) if b not in (40, 41, 42)]
    assert R.plan_matches(req, layout, plan, R.BLOCK_BITS, 0)
    batch.append(shards(plan, need[0])[1])
    assert evaluate(shard_unit, SHARD, batch) == [ref_shard(a) for a in batch] == [0] * len(batch)


def test_each_shard_clause_s_negative_is_refused(shard_unit) -> None:
    """Out of order, a repeated block, a block past the shard's size, a nonzero entry past the count, a count past the
    size or a size past S, and a block named past the size: each returns 0, as the reference does."""
    _req, _layout, need = shard_toy()
    full, last = shards(range(STORE_N), need[0])[0], shards(range(STORE_N), need[0])[3]
    gap = shards([b for b in range(STORE_N) if b != 99], need[0])[3]
    p = full["plan"]
    batch = [{**full, "plan": [p[1], p[0]] + p[2:]}, {**full, "plan": p[:5] + [p[4]] + p[6:]},
             {**last, "plan": last["plan"][:3] + [4] + last["plan"][4:]}, {**gap, "plan": gap["plan"][:3] + [9] + gap["plan"][4:]},
             {**last, "count": 5}, {**full, "size": SHARD.S + 1}, {**last, "regmask": [1 << 4, 0]}]
    assert evaluate(shard_unit, SHARD, batch) == [ref_shard(a) for a in batch] == [0] * len(batch)


def test_the_shard_s_search_covers_its_band(shard_unit) -> None:
    """With every block named, a shard's plan missing GAPS blocks at its start, middle or end (each entry p one of blocks
    p .. p + GAPS, its band's edges) is accepted when the missing blocks aren't named and refused when one is; seeded
    random plans and masks agree with the reference."""
    rng = np.random.default_rng(48)
    batch = []
    for missing in ((0, 1), (15, 16), (30, 31)):
        plan = [j for j in range(SHARD.S) if j not in missing]
        a = {"count": len(plan), "size": SHARD.S, "plan": plan + [0] * len(missing)}
        batch.append({**a, "regmask": RU._mask(set(range(SHARD.S)) - set(missing), SHARD.S)})
        batch.append({**a, "regmask": RU._mask(set(range(SHARD.S)) - {missing[0]}, SHARD.S)})
    for _ in range(64):
        size = int(rng.integers(SHARD.S - 4, SHARD.S + 1))
        keep = sorted(rng.choice(size, size - int(rng.integers(0, 4)), replace=False).tolist())
        named = set(rng.choice(size, int(rng.integers(0, size)), replace=False).tolist()) & set(keep)
        if rng.random() < 0.5:
            named.add(int(rng.integers(0, size)))
        batch.append({"count": len(keep), "size": size, "plan": keep + [0] * (SHARD.S - len(keep)),
                      "regmask": RU._mask(named, SHARD.S)})
    got = evaluate(shard_unit, SHARD, batch)
    assert got == [ref_shard(a) for a in batch]
    assert got[:6] == [1, 0] * 3 and 0 < sum(got[6:]) < 64


#: the re-key: written value 0 (A's step-0 value, block N since step 0, last structural reader 3) moves at step 2
MOVED, AT = 0, 2


def rekeyed(case, life, moved=MOVED):
    """The spread layout with written value ``moved`` re-keyed at step 2 into a fresh block N + 5 of step 2 at its slot: its
    epoch-1 layout entry, its old block freed at 2 and the new one at its last structural reader. Each step's map arguments under the
    lifetime cap ``life`` and the toy's pass row, the layout row and the new block."""
    req, (layout, store) = case.req, case.layouts["spread"]
    v = case.positions[moved]
    old, new = layout.block[v], MAP.N + len(case.positions)
    freed = {s: js - {old} for s, js in store.freed.items()}
    freed[AT] = freed.get(AT, frozenset()) | {old}
    freed[req.last[v]] |= {new}
    store = dataclasses.replace(store, written={**store.written, AT: store.written[AT] | {new}}, freed=freed,
                                payload={**store.payload, new: req.values[v]})
    lam, phi = RU.layout_words(MAP, req, layout, case.positions), RU.map_words(MAP, store)
    x = MAP.V + moved
    lam[2 * x], lam[2 * x + 1] = new, layout.offset[v] // VW
    maps = [RU.map_inputs(MAP, req, store, t, case.positions, lam, phi, case.reg_block, PI, life) for t in range(STEPS)]
    born = maps[req.born[v]]
    born["last"][born["pos"].index(moved)] = AT
    a = maps[AT]
    k = bin(a["present"]).count("1")
    a["present"] |= 1 << k
    a["pos"][k], a["ep"][k], a["last"][k], a["val"][k] = moved, 1, req.last[v], req.values[v] % (1 << VW)
    return maps, lam, new


def test_the_cap_is_the_stale_bound_in_step_form() -> None:
    """Over every schedule of passes at steps 0 .. 5 in consecutive windows, with windows to spare (T = 6), appends of 1, 0,
    2, 1, 1 and 0 blocks a step and L of 1 .. 3: a block written at step s is below the stale bound of the first pass after
    a re-key's step r (N plus the counts of the appends that joined a window L or more before it; none if no pass follows
    r) iff s is below the cap the pass row gives at r. An append joins the window of the first pass after its step."""
    steps, counts = 6, [1, 0, 2, 1, 1, 0]
    S = SimpleNamespace(**{**vars(MAP), "T": steps})
    blocks = [s for s in range(steps) for _ in range(counts[s])]
    for chosen in itertools.product((0, 1), repeat=steps):
        at = [s for s in range(steps) if chosen[s]]
        for gaps in itertools.product((0, 1), repeat=max(len(at) - 1, 0)):
            wins = [0] + list(itertools.accumulate(gaps))
            passes = list(zip(wins, at))
            pi = RU.pass_words(S, passes)
            joined = [next((w for w, p in passes if p > s), None) for s in range(steps)]
            for L in (1, 2, 3):
                for r in range(steps):
                    stale = [s for s in range(steps) if joined[s] is not None and joined[r] is not None
                             and joined[s] <= joined[r] - L]
                    bound = -1 if joined[r] is None else MAP.N + sum(counts[s] for s in stale)
                    cap = RU.rekey_cap(S, pi, L, r)
                    assert [MAP.N + o < bound for o in range(len(blocks))] == [s < cap for s in blocks], (passes, L, r)


@pytest.mark.slow
def test_a_rekey_at_the_cap_is_accepted_and_one_window_early_is_refused(case) -> None:
    """Value 0's block, written at step 0, joined window 1; re-keyed at step 2, whose first pass after is window 3, it is 2
    windows old. At L = 2 (cap 1 at step 2) every step's map unit accepts, and every step's plan unit with step 3 reading
    the value at epoch 1, in the new block; a step-3 plan reading the old block, or the new one at epoch 0, is refused.
    At L = 3 (cap 0) the re-key is one window early, and step 2's map unit refuses it; a pass row dating window 0's last
    pass a step later (cap 1) accepts it, which only the row's binding to the record's passes refuses. The references
    agree."""
    req, (layout, _store) = case.req, case.layouts["spread"]
    assert PI == [0, 1, 2, 3] and [RU.rekey_cap(MAP, PI, L, AT) for L in (2, 3)] == [1, 0]
    maps, lam, new = rekeyed(case, 2)
    early = rekeyed(case, 3)[0][AT]
    batch = maps + [early, {**early, "pi": [1] + PI[1:]}]
    assert evaluate(case.map, MAP, batch) == [ref_map(a) for a in batch] == [1] * STEPS + [0, 1]
    v = case.positions[MOVED]
    plans = []
    for t in range(STEPS):
        blocks, entries = set(R.read_blocks(req, layout, t)), {}
        if t > AT:
            blocks, entries = blocks - {layout.block[v]} | {new}, {v: MAP.V + MOVED}
        gathered = frozenset(TABLE[r] for r in case.gathers[t])
        plans.append(RU.plan_inputs(PLAN, req, t, blocks, case.reg_block, case.gathers[t], gathered, case.positions, lam,
                                    entries))
    bad = [RU.plan_inputs(PLAN, req, 3, R.read_blocks(req, layout, 3), case.reg_block, case.gathers[3],
                          frozenset(TABLE[r] for r in case.gathers[3]), case.positions, lam, {v: MAP.V + MOVED}),
           {**plans[3], "kvmask": RU.plan_inputs(PLAN, req, 3, [], case.reg_block, case.gathers[3],
                                                 frozenset(TABLE[r] for r in case.gathers[3]), case.positions, lam)["kvmask"]}]
    assert evaluate(case.plan, PLAN, plans + bad) == [ref_plan(a) for a in plans + bad] == [1] * STEPS + [0, 0]


@pytest.mark.slow
def test_a_rekey_s_old_block_is_held_to_its_slot_its_write_and_the_cap(case) -> None:
    """Step 2's re-key of value 0, accepted at L = 2 (cap 1), returns 0 when the new block holds the value in another slot
    than the old one, when its old block is a registered one, or one step 2 itself writes (at L = 0, cap 3, which that
    block's write step is below), and when no pass follows its step (the pass row's last window's pass at step 2). Value 2
    (A's step-1 value, written at step 1) re-keyed at step 2 is refused at cap 1: its block is one window too young. Its
    write step forged to 0 in the map row passes step 2's unit, and its writer, step 1's, refuses it. The reference agrees
    on each."""
    maps, _lam, new = rekeyed(case, 2)
    a, x = maps[AT], MAP.V + MOVED
    slot, reg, own = list(a["lam"]), list(a["lam"]), list(a["lam"])
    slot[2 * x + 1] = 1
    pay = list(a["pay"])
    pay[new - a["first"]] <<= VW
    reg[2 * MOVED] = 7
    own[2 * MOVED] = a["first"]
    young, _, _ = rekeyed(case, 2, moved=2)
    phi = list(young[AT]["phi"])
    phi[MAP.N + 2 * (case.layouts["spread"][0].block[case.positions[2]] - MAP.N)] = 0
    assert RU.rekey_cap(MAP, PI, 0, AT) == 3 and RU.rekey_cap(MAP, [0, 1, 2, 2], 2, AT) == 0
    batch = {"another-slot": {**a, "lam": slot, "pay": pay}, "registered-old-block": {**a, "lam": reg},
             "old-block-the-step-writes": {**a, "lam": own, "life": 0}, "no-pass-after-the-step": {**a, "pi": [0, 1, 2, 2]},
             "one-window-too-young": young[AT], "forged-write-step (re-key)": {**young[AT], "phi": phi},
             "forged-write-step (writer)": {**young[1], "phi": phi}}
    want = [0, 0, 0, 0, 0, 1, 0]
    assert evaluate(case.map, MAP, list(batch.values())) == [ref_map(b) for b in batch.values()] == want, list(batch)


def test_the_rows_are_the_signature_s_params() -> None:
    """Each unit's rows cover its parameters once each, the public row first, in signature order."""
    for defn, S, rows in ((RU.PlanMatch, PLAN, RU.plan_rows(PLAN)), (RU.MemoryMapStep, MAP, RU.memory_map_rows(MAP)),
                          (RU.PlanShard, SHARD, RU.plan_shard_rows(SHARD))):
        params, _ = bind(defn, **vars(S)).signature()
        flat = [p.partition("[")[0] for _r, _b, ps in rows for p in ps]
        assert list(dict.fromkeys(flat)) == [n for n, _ in params]
        bits = {n: t.leaves for n, t in params}
        assert sum(b for _r, b, _ps in rows) == sum(bits.values())
        assert rows[0][0] == "pub"
