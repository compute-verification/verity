"""`beta.py`, the min-cut bound β(C) on the stacked share of dynamic memory, on toys whose answer is known: Dinic's flow and the
bound's network held to every cut by enumeration, writes a cheap function of a 1-bit input (β ≈ 0), writes of encodings the
circuit computes (β = 1), an attention whose work grows with its context behind a fixed-width output (the bound charges the
output, not the work), data products as work beside a weight's free MACs, a hyperedge paid once, unknown work kept out of D,
a window, the least E below M_dyn, and coarse graining, which never lowers the bound. Every number is exact: E, R_in and L are
rationals and the bound a `Fraction`."""
from __future__ import annotations

import itertools
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.ops import prims
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, coll_concat
from verity.ml.circuits.types import Array, Value

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import beta as B  # noqa: E402

S32, S8, BIT = Value(32), Value(8), Value(1)
E40 = B.E_WORD["2^-40"]


def stand_in(name: str, params, ret=S32) -> PrimitiveDefinition:
    """A registry primitive's id with a dummy evaluator: property/v1's table keys on ids."""
    return PrimitiveDefinition(name, 1, tuple(params), ret, lambda *v: 0, register=False)


MUL = stand_in("F32Mul", (("a", S32), ("b", S32)))  # roundings
SEL = stand_in("SelectI32", (("c", BIT), ("a", S32), ("b", S32)))  # compare_select
UNLISTED = stand_in("TestBetaUnlisted", (("c", BIT),))
IMMA = stand_in("BlackwellS8ImmaDot32", (("acc", S32), ("a", Array(32, S8)), ("b", Array(32, S8))))  # an s8-family step


def comp(name: str, params, ret, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda _S: (tuple(params), ret), body, register=False)


def chain(name: str, n_in: int, gates: int, n_out: int = 1) -> CompositeDefinition:
    """`gates` roundings in a chain through its `n_in` words, then `n_out` words of the chain's end times each input word."""
    def body(Bd, _S, xs):
        a = xs[0]
        for j in range(gates):
            a = Bd.call(MUL, a, xs[j % n_in])
        return a if n_out == 1 else array_of([Bd.call(MUL, a, xs[i % n_in]) for i in range(n_out)])
    return comp(name, (("xs", Array(n_in, S32)),), S32 if n_out == 1 else Array(n_out, S32), body)


def graph_of(program: Program, ends: list[int], **kw) -> tuple[B.Graph, dict]:
    return B.from_program(program, doc=census(program, ends), **kw)


# -- the toys ------------------------------------------------------------------------------------------------------------


def cheap_writes(words: int = 32) -> tuple[Program, list[int]]:
    """Step 0: a 1-bit run input selects each of `words` words between two registered ones (one compare_select a word);
    step 1 reads them all, so they are held: 32·words bits of writes."""
    ends: list[int] = []
    fold = chain("TestBetaFold", words, words)

    def body(Bd, _S, x, w):
        ends.clear()
        ks = [Bd.call(SEL, x[0], w[2 * i], w[2 * i + 1]) for i in range(words)]
        ends.append(len(Bd.nodes))
        out = Bd.call(fold, array_of(ks))
        ends.append(len(Bd.nodes))
        return out

    fn = comp("TestBetaCheap", (("x", Array(1, BIT)), ("w", Array(2 * words, S32))), S32, body)
    return Program(fn, registered=("w",)), ends


def encoder(words: int = 32, depth: int = 4) -> tuple[Program, list[int]]:
    """Step 0: each of `words` words is its own Call, `depth` roundings of a 32-bit run input; step 1 reads them all."""
    ends: list[int] = []
    enc = chain("TestBetaEnc", 1, depth)
    fold = chain("TestBetaEncFold", words, words)

    def body(Bd, _S, x):
        ends.clear()
        ks = [Bd.call(enc, x) for _ in range(words)]
        ends.append(len(Bd.nodes))
        out = Bd.call(fold, array_of(ks))
        ends.append(len(Bd.nodes))
        return out

    fn = comp("TestBetaEncoder", (("x", Array(1, S32)),), S32, body)
    return Program(fn), ends


def attention(n: int = 6, m: int = 8, kv: int = 8) -> tuple[Program, list[int]]:
    """Two layers over n steps, a step a position. At step t, from a run input word x_t:

    - ``proj``: layer 1's KV, kv words (kv roundings), read by every later attention: a write;
    - ``attn``: reads layer 1's KV of positions 0..t with (t + 1)·m·kv roundings, its context's work, and returns one word;
    - ``up``: layer 2's KV, kv words of attn's word (kv roundings): a write, behind a 32-bit output whatever attn's work;
    - ``attn2``: reads layer 2's KV of positions 0..t."""
    ends: list[int] = []
    proj = chain("TestBetaProj", 1, 0, kv)
    up = chain("TestBetaUp", 1, 0, kv)

    def body(Bd, _S, x):
        ends.clear()
        k1, k2, outs = [], [], []
        for t in range(n):
            k1.append(Bd.call(proj, x[t:t + 1]))
            ctx = Array((t + 1) * kv, S32)
            o = Bd.call(chain(f"TestBetaAttn{t + 1}x{m}", (t + 1) * kv, (t + 1) * m * kv), coll_concat(ctx, k1))
            k2.append(Bd.call(up, array_of([o])))
            outs.append(Bd.call(chain(f"TestBetaAttnTwo{t + 1}", (t + 1) * kv, (t + 1) * kv), coll_concat(ctx, k2)))
            ends.append(len(Bd.nodes))
        return array_of(outs)

    fn = comp(f"TestBetaAttention{n}x{m}", (("x", Array(n, S32)),), Array(n, S32), body)
    return Program(fn), ends


def test_cheap_writes_give_beta_near_zero():
    """Each write is one compare_select of the 1-bit input: D holds every select at 1024/E bits each, against 32 bits a
    write at the cut on the writes. Under inputs="window" the input's 1 bit crosses once, however many selects read it."""
    g, info = graph_of(*cheap_writes())
    assert (info["writes"], g.m_dyn, info["unknown_work_nodes"]) == (32, 1024, 0)
    b = B.bound(g, E=E40)
    assert b.crossing == 0 and b.work == 32 and b.beta == Fraction(32, E40)  # 0.0028%
    w = B.bound(g, E=E40, inputs="window")
    assert w.crossing == 1 and w.beta == (1 + Fraction(32 * 1024, E40)) / 1024
    r = B.bound(g, E=E40, R_in=1, L=2)
    assert r.s_stack == b.cut + 16 and r.input_bits == 16  # R_in·L is bytes: 8 bits a byte


def test_computed_encodings_give_beta_one():
    """Each write is a Call of 4 roundings a 32-bit word: at E = 64 units a 1024-bit word, putting it in D costs 64 bits
    against its 32, so every write is cut where it's written and β = 1. At a large E the work is cheap again."""
    g, _ = graph_of(*encoder())
    assert g.m_dyn == 1024
    b = B.bound(g, E=64)
    assert b.beta == 1 and b.crossing == 1024 and b.work == 0
    big = B.bound(g, E=1 << 20)
    assert big.beta == Fraction(32 * 4, 1 << 20) and big.crossing == 0


def test_attention_work_is_discounted():
    """Layer 2's KV comes from attention's one-word output: the min cut puts attention in U and pays 32 bits for its output
    plus up's work, whatever attention's work, which grows with the context (m and t). At E = 64 (16 bits a unit): proj in D
    costs 8·16 = 128 of its 256 bits, up in D 128 + the 32-bit output, so 288 bits a position against 512 written; the last
    position's KV is read by no later step and is no write. A bound charging all the work grows with m and n²."""
    rows = {}
    for n, m in ((4, 8), (4, 32), (6, 8), (6, 64)):
        g, info = graph_of(*attention(n, m))
        b = B.bound(g, E=64)
        assert info["writes"] == 2 * (n - 1) and g.m_dyn == (n - 1) * 2 * 8 * 32
        assert b.cut == 288 * (n - 1) and b.beta == Fraction(288, 512), (n, m, b.cut)
        rows[n, m] = Fraction(1024) * sum(w for w in g.work if w) / 64
    assert rows[4, 32] > rows[4, 8] * 3 and rows[6, 64] > rows[6, 8] * 6  # what charging all the work would say
    assert all(r > 288 * (n - 1) for (n, _m), r in rows.items())


def test_node_work_is_property_v1s():
    """A node's work is property/v1's tally of it: the graph's total is `stacking.analyze`'s, step by step."""
    for program, ends in (cheap_writes(), encoder(), attention(5, 3)):
        doc = census(program, ends)
        g, _ = B.from_program(program, doc=doc)
        a = P.analyze(program, doc)
        assert sum(g.work) == sum(sum(s["units"].values()) for s in a.steps)


def products(registered: tuple[str, ...] = ("w",)) -> tuple[Program, list[int]]:
    """Step 0: two 32-lane s8 dots, x·y of two run inputs and x·w by `registered` w; step 1 multiplies their results."""
    ends: list[int] = []

    def body(Bd, _S, x, y, w):
        ends.clear()
        z = Bd.call(prims.ZERO32)
        a, b = Bd.call(IMMA, z, x, y), Bd.call(IMMA, z, x, w)
        ends.append(len(Bd.nodes))
        out = Bd.call(MUL, a, b)
        ends.append(len(Bd.nodes))
        return out

    fn = comp("TestBetaProducts", (("x", Array(32, S8)), ("y", Array(32, S8)), ("w", Array(32, S8))), S32, body)
    return Program(fn, registered=registered), ends


def test_data_products_are_work_and_a_weights_macs_are_not():
    """Nonlinear units: the dot of two run inputs is 32 data products, each a unit of its node's work; the dot by a
    registered w is a linear map and free. A root input `registered` takes (as a Build that binds its weights as run inputs
    needs) carries no bits and makes its products by it free; one that isn't a root input is refused."""
    program, ends = products()
    g, info = graph_of(program, ends)
    assert (info["products"], info["budgeted_units"], info["work_units"]) == (32, 1, 33)
    assert info["products_by_callee"] == {"BlackwellS8ImmaDot32_v1": 32} and list(info["run_inputs"].values()) == [256, 256]
    assert sorted(w for w in g.work if w) == [1, 32] and g.m_dyn == 64
    g2, info2 = graph_of(program, ends, registered=[0])
    assert (info2["products"], info2["work_units"], info2["registered_inputs"]) == (0, 1, 1) and len(info2["run_inputs"]) == 1
    assert g2.inputs == frozenset({1}) and all(not x.dynamic for x in g2.values if x.producer == 0)
    assert graph_of(*products(("w", "y")))[1]["products"] == 0
    with pytest.raises(ValueError, match="registered names root inputs"):
        graph_of(program, ends, registered=[3])


def test_leaves_one_is_the_least_e_below_m_dyn():
    """The encoder's writes cost 4 units a 32-bit word in D: the cut drops below their 1024 bits once 1024·128/E < 1024, at
    E = 129, and at E = 128 it is M_dyn still. A graph whose writes are free leaves 1 at once."""
    g, _ = graph_of(*encoder())
    E = B.leaves_one(g)
    assert E == 129 and B.bound(g, E=E, R_in=0).cut < g.m_dyn == B.bound(g, E=E - 1, R_in=0).cut
    assert B.leaves_one(B.Graph([0, 0], [B.Value(0, 8, (1,)), B.Value(1, 8, write=True)], ["a", "b"])) == 1
    assert B.leaves_one(B.Graph([0], [], ["a"])) is None


def test_window_holds_earlier_calls_in_u():
    """Over the window of steps 2..n-1, steps 0 and 1 are held in U: their values reach the window as advice at their bits,
    and their KV is not the window's to write."""
    n = 6
    program, ends = attention(n, 8)
    g, info = graph_of(program, ends, window=range(2, n))
    assert info["writes"] == 2 * (n - 3) and len(g.fixed) == sum(1 for k in range(len(g.work)) if g.names[k].endswith(("@0", "@1")))
    assert B.bound(g, E=64).cut == 288 * (n - 3)
    with pytest.raises(ValueError, match="non-empty range"):
        graph_of(program, ends, window=range(3, 3))


def test_unknown_work_never_in_d():
    """A write made by a gate property/v1 refuses (unlisted) has unknown work: it can't sit in D, so its bits are paid."""
    ends: list[int] = []
    fold = chain("TestBetaFoldOne", 1, 1)

    def body(Bd, _S, x):
        ends.clear()
        k = Bd.call(UNLISTED, x[0])
        ends.append(len(Bd.nodes))
        out = Bd.call(fold, array_of([k]))
        ends.append(len(Bd.nodes))
        return out

    program = Program(comp("TestBetaUnknown", (("x", Array(1, BIT)),), S32, body))
    g, info = graph_of(program, ends)
    assert info["unknown_work_nodes"] == 1 and info["refused"] == {"unlisted:TestBetaUnlisted_v1": 1}
    assert B.bound(g, E=E40).beta == 1


def test_a_value_read_twice_is_paid_once():
    """A value read by two nodes of D crosses the cut once: the hyperedge node, not one edge a reader."""
    g = B.Graph([0, 0, 0], [B.Value(0, 10, (1, 2)), B.Value(1, 100, write=True), B.Value(2, 100, write=True)],
                ["a", "b", "c"], fixed=frozenset({0}))
    b = B.bound(g, E=1)
    assert (b.cut, b.crossing, b.d) == (10, 10, (1, 2))


def test_exact_parameters():
    g, _ = graph_of(*cheap_writes(4))
    assert B.bound(g, E="24504").beta == B.bound(g, E=Fraction(24504)).beta == Fraction(32, 24504)
    with pytest.raises(ValueError, match="binary value"):
        B.bound(g, E=24504.0)
    with pytest.raises(ValueError, match="inputs is one of"):
        B.bound(g, inputs="free")


# -- the flow and the network against every cut ---------------------------------------------------------------------------


def random_graph(rng: random.Random, n: int) -> B.Graph:
    work = [rng.choice([0, 1, 2, 5, 40, Fraction(3, 2), None if rng.random() < 0.1 else 7]) for _ in range(n)]
    values = []
    for _ in range(rng.randint(n, 3 * n)):
        p = rng.randrange(n)
        readers = tuple(rng.sample(range(n), rng.randint(0, min(3, n))))
        values.append(B.Value(p, rng.randint(0, 64), readers, rng.random() < 0.85, rng.random() < 0.3, rng.random() < 0.1))
    inputs = frozenset(v for v in range(n) if rng.random() < 0.15)
    fixed = frozenset(v for v in range(n) if rng.random() < 0.1)
    return B.Graph(work, values, [f"v{i}" for i in range(n)], inputs, fixed)


def brute(g: B.Graph, E: Fraction, c: int, inputs: str) -> Fraction:
    """The least cost over every assignment of the nodes to U and D (held nodes and unknown work in U)."""
    n = len(g.work)
    held = set(g.fixed) | (set(g.inputs) if inputs == "window" else set()) | {v for v, w in enumerate(g.work) if w is None}
    free = [v for v in range(n) if v not in held]
    best = None
    for bits in itertools.product((False, True), repeat=len(free)):
        D = {v for v, d in zip(free, bits) if d}
        cost = Fraction(c) * sum((Fraction(g.work[v]) for v in D), Fraction(0)) / E
        for x in g.values:
            b = x.bits if x.dynamic and not (inputs == "rate" and x.input) else 0
            if b and x.producer not in D and (x.write or any(r in D for r in x.readers if r != x.producer)):
                cost += b
        best = cost if best is None else min(best, cost)
    return best


def test_bound_is_the_least_cut():
    """The flow network's min cut is the least cost of b(U→D) + |c|·W(D)/E over every cut with the writes in D, on 120
    seeded random graphs (fractional work, unknown work, hyperedges, writes, inputs and held nodes) under both input modes."""
    for seed in range(120):
        rng = random.Random(seed)
        g = random_graph(rng, rng.randint(1, 9))
        for E, inputs in ((Fraction(E40), "rate"), (Fraction(64), "window"), (Fraction(7, 3), "rate")):
            assert B.bound(g, E=E, c=1024, inputs=inputs).cut == brute(g, E, 1024, inputs), (seed, E, inputs)


def test_max_flow_against_every_cut():
    """Dinic's flow equals the least s-t cut by enumeration on 200 seeded random networks."""
    for seed in range(200):
        rng = random.Random(1000 + seed)
        n = rng.randint(2, 8)
        edges = [(rng.randrange(n), rng.randrange(n), rng.randint(0, 20)) for _ in range(rng.randint(0, 4 * n))]
        flow, side = B.max_flow(n, edges, 0, n - 1) if n > 1 else (0, [True])
        best = min(sum(c for u, v, c in edges if u in S and v not in S)
                   for r in range(n - 1) for mid in itertools.combinations(range(1, n - 1), r)
                   for S in [{0, *mid}]) if n > 2 else sum(c for u, v, c in edges if u == 0 and v == 1)
        assert flow == best, seed
        S = {v for v in range(n) if side[v]}
        assert 0 in S and n - 1 not in S and sum(c for u, v, c in edges if u in S and v not in S) == flow


# -- coarse graining ------------------------------------------------------------------------------------------------------


def test_coarse_graining_never_lowers_the_bound():
    """A coarse cut is a cut of the fine graph: on 150 seeded random graphs under random groupings, and on the attention toy
    grouped by step and by callee, the coarse bound is at least the fine one. By step, attention shares a node with the
    writes, so its work is charged and the bound is strictly looser."""
    for seed in range(150):
        rng = random.Random(5000 + seed)
        g = random_graph(rng, rng.randint(1, 9))
        groups = [rng.randrange(max(1, len(g.work) // 2)) for _ in g.work]
        coarse = B.coarsen(g, groups.__getitem__)
        for E, inputs in ((Fraction(64), "rate"), (Fraction(E40), "window")):
            assert B.bound(coarse, E=E, inputs=inputs).cut >= B.bound(g, E=E, inputs=inputs).cut, seed
    g, _ = graph_of(*attention(6, 8))
    fine = B.bound(g, E=64)
    by_step = B.coarsen(g, lambda v: g.names[v].rsplit("@", 1)[-1] if "@" in g.names[v] else g.names[v])
    by_callee = B.coarsen(g, lambda v: g.names[v].split(":", 1)[1].split("@")[0])
    assert by_step.m_dyn == by_callee.m_dyn == g.m_dyn
    assert B.bound(by_step, E=64).cut > fine.cut and B.bound(by_callee, E=64).cut >= fine.cut
    assert B.bound(B.coarsen(g, lambda v: v), E=64).cut == fine.cut


def test_report_writes_each_bound_and_coarse_ones_are_looser():
    """The CLI's document on the attention toy: written after each bound, every coarse bound at least the fine one, by step
    strictly looser, and JSON throughout."""
    g, info = graph_of(*attention(5, 8))
    seen: list[list[str]] = []
    doc = B.report(g, info, chunks=(3, 1), first=info["inputs"], Es=(64, E40), Ls=(0, 600), m_node=1 << 30,
                   write=lambda d: seen.append(sorted(d["coarse"]) + ["fine"] * ("bounds" in d)))
    assert seen[-1] == ["by_1_calls", "by_3_calls", "by_step", "fine"] and len(seen) == 4
    fine = [Fraction(r["cut_bits"]) for r in doc["bounds"]]
    assert fine[0] == 288 * 4
    for name, c in doc["coarse"].items():
        assert all(Fraction(r["cut_bits"]) >= f for r, f in zip(c["bounds"], fine)), name
    assert Fraction(doc["coarse"]["by_step"]["bounds"][0]["cut_bits"]) > fine[0]
    assert [r["cut_bits"] for r in doc["coarse"]["by_1_calls"]["bounds"]] == [r["cut_bits"] for r in doc["bounds"]]
    assert json.loads(json.dumps(doc, default=str))["bounds"][1]["m_node_bits"] == 1 << 30


# -- modeled decoders -----------------------------------------------------------------------------------------------------


def spec(name: str) -> dict:
    return json.loads((Path(B.__file__).parent / name).read_text())


def test_the_8b_decoder_adds_up_to_the_census_request():
    """The 8B spec's blocks are the census's request Program's budgeted units, 9,404,379,802 (its ``check``), and its data
    products the Program's 11,418,992,640 (``check_products``): 32 heads · 128 a (query row, key) pair a layer for Q′K′ᵀ over
    the request's 256·257/2 + (257 + .. + 287) = 41,328 pairs, and as many for P V′ over those rows' keys padded to 32, 45,792:
    32 steps, the KV of steps 0..30 (256 + 30 rows a layer) and 31 tokens written. A term one unit off is refused."""
    s = spec("decoder-llama31-8b-int8.json")
    g, info = B.decoder(s)
    assert (info["budgeted_units"], info["steps"], info["writes"]) == (9_404_379_802, 32, 32 * 31 + 31)
    assert info["products"] == 32 * 4096 * (41_328 + 45_792) == 11_418_992_640 and info["products_checked"]
    assert info["work_units"] == 9_404_379_802 + info["products"] == sum(g.work)
    assert g.m_dyn == 32 * (256 + 30) * 32768 + 31 * 32
    with pytest.raises(ValueError, match="budgeted units .* not the measured"):
        B.decoder({**s, "sample": s["sample"] + 1})
    with pytest.raises(ValueError, match="data products .* not the measured"):
        B.decoder({**s, "check_products": s["check_products"] + 1})


def test_decoder_prefill_rows_and_the_specs_build():
    """A prompt run a row a step reads its prompt token at each of its steps; every spec here builds."""
    s = {"prompt": 3, "decode": 2, "layers": 2, "prefill": "rows", "row": {"embed": 1, "pre": 2, "post": 3, "final": 4},
         "sample": 5, "attention": {"r0": 1, "b_q": 1, "b_kv": 1, "w0": 1, "w_start": 0},
         "bits": {"hidden": 8, "q": 8, "kv": 4, "o": 8, "token": 2}}
    g, info = B.decoder(s)
    assert info["steps"] == 5 and info["writes"] == 2 * 4 + 2 and g.m_dyn == 2 * 4 * 4 + 2 * 2
    assert sum(1 for x in g.values if x.input) == 3
    for name in ("decoder-llama31-8b-int8.json", "decoder-llama31-70b-tp8.json", "decoder-e2e-h200-mqa.json"):
        assert B.decoder(spec(name))[1]["work_units"] > 0
    with pytest.raises(ValueError, match="prefill"):
        B.decoder({**s, "prefill": "chunked"})


def test_decoder_attention_is_charged_its_output_not_its_work():
    """At E = |c| (a unit a bit), with free blocks and costly tokens and residuals, the writes come from D at the cost of
    attention's outputs crossing (2 + 1 rows of 5 bits; the last step writes nothing), however much attention's keys
    cost; free attention costs nothing."""
    s = {"prompt": 2, "decode": 2, "layers": 1, "row": {"embed": 0, "pre": 0, "post": 0, "final": 0}, "sample": 0,
         "attention": {"r0": 0, "b_q": 0, "b_kv": 100, "w0": 0, "w_start": 0},
         "bits": {"hidden": 1000, "q": 1000, "kv": 10, "o": 5, "token": 1000}}
    for b_kv in (100, 10_000):
        g, _ = B.decoder({**s, "attention": {**s["attention"], "b_kv": b_kv}})
        b = B.bound(g, E=1024)
        assert (b.cut, b.crossing, b.work) == (15, 15, 0) and g.m_dyn == 30 + 2000
    g, _ = B.decoder({**s, "attention": {**s["attention"], "b_kv": 0}})
    assert B.bound(g, E=1024).cut == 0


def test_table_moves_only_the_input_term():
    g, _ = graph_of(*cheap_writes(4))
    rows = B.table(g, Es=(E40,), Ls=(0, 600, 3600))
    assert [r["input_bits"] for r in rows] == ["0", str(8 * B.R_IN * 600), str(8 * B.R_IN * 3600)]
    assert len({r["cut_bits"] for r in rows}) == 1


def test_the_input_term_is_shared_over_the_node():
    """One request's graph on a node holding M_node bits of dynamic memory: β = cut/M_dyn + R_in·L/M_node."""
    g, _ = graph_of(*cheap_writes(4))
    b = B.bound(g, E=E40, R_in=1, L=2, m_node=10 * g.m_dyn)
    assert b.beta == Fraction(32, E40) + Fraction(16, 10 * g.m_dyn) and b.s_stack == b.beta * g.m_dyn
    assert B.table(g, Es=(E40,), Ls=(2,), R_in=1, m_node=10 * g.m_dyn)[0]["beta"] == str(b.beta)
    with pytest.raises(ValueError, match="m_node"):
        B.bound(g, m_node=g.m_dyn - 1)
