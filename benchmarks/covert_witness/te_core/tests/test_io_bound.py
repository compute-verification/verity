"""te_core.io_bound: the cap's closed forms, each inequality the bound rests on checked by brute force on small products,
the test vectors (tests/io_bound_vectors.json), and the kinds `rows_of` reads off captured steps of the tiny model."""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from te_core import io_bound as iob, policy
from te_core.capture import capture_step

#: the tiny model's steps: d 64, d_ff 192, 256 tokens a micro-step, head dim 32, and a vocabulary no other dim equals
VOCAB = 320
TINY = {"size": "tiny", "vocab": VOCAB, "batch": 8, "ctx": 64, "micro_steps": 2}
#: the console's k scaled to the tiny model: a block holds 1/64 of a d × d weight, as at the s shape
K_TINY = 1024
WIDTHS = (1, 2, 4, 8, 16, 32)


def close(a, b, rel: float = 1e-12) -> bool:
    """`a` and `b` equal as JSON trees, numbers to `rel`."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[x], b[x], rel) for x in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or math.isclose(a, b, rel_tol=rel)
    return a == b


def faces(S) -> tuple[int, int, int]:
    """|π_IL S|, |π_LJ S|, |π_IJ S| of a set of MACs (i, l, j)."""
    return len({(i, l) for i, l, _ in S}), len({(l, j) for _, l, j in S}), len({(i, j) for i, _, j in S})


# -- the vectors ------------------------------------------------------------------------------------------------------

def test_the_vectors_are_the_bound_on_their_inputs():
    data = json.loads(iob.VECTORS.read_text())
    assert close(data, json.loads(json.dumps(iob.vectors()))), "stale: python -m te_core.io_bound vectors"


FIGURES = ("blocks_root", "blocks_linear", "blocks", "rho", "rho_main", "blocks_no_cell", "rho_no_cell")


def test_each_vectors_term_solves_its_cap():
    """What a port of the bound checks without this module: N·Γ + B·√N = G and N·Γ = Σ_P (G_P − Λ·L_P)⁺ at each Λ on
    the products it keeps (every one for the numerical Λ, the best subset's for the proved one, `proved_subset`, and
    `proved_subset_no_cell` for the figure without P-cell), the blocks the larger, ρ = N·c/|C| with every product in
    |C|; with nothing counted every figure is 0, and where no subset has a proved Λ the proved ones are null."""
    forms, subsets, dropped = Counter(), 0, 0
    for v in json.loads(iob.VECTORS.read_text())["vectors"]:
        i, cap, term = v["inputs"], v["cap"], v["term"]
        ps, c, k, C = i["products"], i["c"], i["k"], i["C_gates"]
        assert 0 <= term["share"] < 1, v["name"]
        lin = [(p["gates_per_mac"] * p["macs"], p["gates_per_mac"] * m * p["b_c"] * p["outputs"])
               for p, m in zip(ps, cap["mu"])]
        assert sum(x for _, x in lin) == pytest.approx(cap["L"], rel=1e-12)
        if not ps:
            assert all(term[f"{lam}_{x}"] == 0 for lam in ("proved", "numerical") for x in FIGURES)
            continue
        if cap["lam_proved"] is None:
            assert all(term[f"proved_{x}"] is None for x in FIGURES), v["name"]
            assert term["proved_subset"] is None and term["proved_subset_no_cell"] is None
        for lam, sub in (("numerical", None), ("proved", "proved_subset"), ("proved", "proved_subset_no_cell")):
            if lam == "proved" and cap["lam_proved"] is None:
                continue
            kept = term[sub]["kept"] if sub else list(range(len(ps)))
            lv = term[sub]["lam"] if sub else cap["lam_numerical"]
            if sub:
                dropped += bool(term[sub]["dropped"])
                assert term[sub]["dropped"] == len(ps) - len(kept)
                assert term[sub]["dropped_macs_share"] == pytest.approx(
                    1 - sum(ps[j]["macs"] for j in kept) / sum(p["macs"] for p in ps), abs=1e-15)
            G = sum(ps[j]["gates_per_mac"] * ps[j]["macs"] for j in kept)
            gamma = lv * max(ps[j]["gates_per_mac"] * cap["alpha"][j] for j in kept)
            B = lv * k / 2 * math.sqrt(sum(ps[j]["gates_per_mac"] ** 2 * ps[j]["outputs"] /
                                           (ps[j]["b_a"] * ps[j]["b_b"]) for j in kept))
            net = sum(max(0.0, lin[j][0] - lv * lin[j][1]) for j in kept)
            if sub != "proved_subset_no_cell":
                root, linear, n = (term[f"{lam}_{x}"] for x in ("blocks_root", "blocks_linear", "blocks"))
                assert root * gamma + B * math.sqrt(root) == pytest.approx(G, rel=1e-9), v["name"]
                assert linear == pytest.approx(net / gamma, rel=1e-12), v["name"]
                subsets += net > max(0.0, G - lv * sum(lin[j][1] for j in kept)) * (1 + 1e-9)
                assert n == max(root, linear)
                forms["linear" if linear > root else "root"] += 1
                assert term[f"{lam}_rho"] == pytest.approx(n * c / C, rel=1e-12)
                assert term[f"{lam}_rho_main"] == pytest.approx(G / gamma * c / C, rel=1e-12)
            if sub != "proved_subset":
                g_cell = gamma + k * max(ps[j]["cell_gates"] for j in kept)
                n = term[f"{lam}_blocks_no_cell"]
                assert n == pytest.approx(max(iob.blocks(G, g_cell, B), net / g_cell), rel=1e-12), v["name"]
                assert term[f"{lam}_rho_no_cell"] == pytest.approx(n * c / C, rel=1e-12)
            assert term[f"{lam}_rho_no_cell"] <= term[f"{lam}_rho"] <= term[f"{lam}_rho_main"]
        if cap["lam_proved"] is not None:
            assert term["proved_rho"] <= term["numerical_rho"]
    # both forms, a linear form that drops a product, and a best subset that drops one
    assert forms["linear"] and forms["root"] and subsets and dropped
    void = [v for v in json.loads(iob.VECTORS.read_text())["vectors"] if v["cap"]["lam_proved"] is None]
    assert [v["cap"]["whole_chain"] for v in void] == [None]


def test_a_short_chain_product_drops_from_the_proved_bound_and_never_voids_it():
    """A product whose whole-making chain is too short for Λ (a rank-1 adapter's, r = k/(3K²b) past 1/4) beside
    products whose chains are long enough, or none: the proved bound is the others' alone at the same |C|, and the
    subset names what it dropped; with only short chains no subset has a proved Λ, and the figures are None, never 0
    (nothing counted is 0)."""
    c, k = iob.CONSOLE
    bf16 = {"b_a": 16, "b_b": 16, "b_c": 32, "gates_per_mac": 23682 / 16, "cell_gates": 23682.0}
    square = {"macs": 5e14, "outputs": 5e14 / 512, "k_dim": 512, "k_whole": 512} | bf16
    free = {"macs": 1e14, "outputs": 1e14 / 1024, "k_dim": 1024, "k_whole": None} | bf16
    adapter = {"macs": 5e11, "outputs": 5e11, "k_dim": 1, "k_whole": 1} | bf16
    C = 1e18
    alone = iob.bound([square, free], c, k, C)["proved"]
    got = iob.bound([square, free, adapter], c, k, C)["proved"]
    share = adapter["macs"] / (square["macs"] + free["macs"] + adapter["macs"])
    for fig, sub in (("rho", "subset"), ("rho_no_cell", "subset_no_cell")):
        assert got[fig] == pytest.approx(alone[fig], rel=1e-12) and got[fig] > 0
        assert (got[sub]["k_whole_min"], got[sub]["kept"], got[sub]["dropped"]) == (512, [0, 1], 1)
        assert got[sub]["dropped_macs_share"] == pytest.approx(share, rel=1e-12)
    assert alone["subset"]["dropped"] == 0
    void = iob.bound([adapter, square | {"k_whole": 64}], c, k, C)["proved"]
    assert all(void[x] is None for x in FIGURES) and void["subset"] is None
    assert iob.bound([adapter, free], c, k, C)["proved"]["subset"]["kept"] == [1]
    assert iob.bound([], c, k, C)["proved"]["rho"] == 0


# -- the closed forms ---------------------------------------------------------------------------------------------------

def test_the_cap_at_the_consoles_policy():
    # k = 65,536: bf16 operands into 32-bit sums, 2/2/8 and one bit; c, te_core.policy's session of the 45M reference,
    # is 3.626e11 bf16 MACs
    assert iob.CONSOLE is policy.CONSOLE
    assert iob.alpha(65536, 16, 16, 32) == pytest.approx(35673.28, rel=1e-6)
    assert iob.alpha(65536, 2, 2, 8) == pytest.approx(570772.5, rel=1e-6)
    assert iob.alpha(65536, 1, 1, 1) == pytest.approx(3228776.7, rel=1e-6)
    assert iob.beta(65536, 16, 16) == 2048
    assert iob.mu(65536, 16, 16, 32) == pytest.approx(0.918559, rel=1e-6)
    assert iob.CONSOLE[0] / (23682 / 16) / iob.alpha(65536, 16, 16, 32) == pytest.approx(1.0164e7, rel=1e-4)


def test_lambdas_factors():
    assert iob.SHARE == pytest.approx(2 / math.sqrt(3))
    assert iob.feed(0) == 1 and iob.feed(16) == pytest.approx(math.sqrt(2)) and iob.feed(32) == pytest.approx(2)
    assert iob.feed(8) == pytest.approx(math.sqrt(4 / 3))
    assert iob.feed(16 - 1e-9) == pytest.approx(iob.feed(16 + 1e-9))   # continuous where the two forms meet
    for K in (128, 512, 1408, 65536):
        r = 65536 / (3 * K * K * 16)
        u = 0.0   # nested whole-making from nothing: u ← r·(1 + u)², which climbs to the least root
        for _ in range(200):
            u = r * (1 + u) ** 2
        assert iob.whole_chain(65536, K, 16) == pytest.approx((1 + u) ** 1.5, rel=1e-12)
    assert iob.whole_chain(65536, 512, 16) == pytest.approx(1.0079, abs=1e-4)
    assert iob.whole_chain(65536, 64, 16) == math.inf   # r = 1/3: no root, the proved bound is void


def test_blocks_is_the_least_n():
    rng = random.Random(0)
    for _ in range(2000):
        G, gamma, B = 10 ** rng.uniform(0, 20), 10 ** rng.uniform(0, 10), rng.choice((0.0, 10 ** rng.uniform(0, 12)))
        n = iob.blocks(G, gamma, B)
        assert n * gamma + B * math.sqrt(n) == pytest.approx(G, rel=1e-9)
    assert iob.blocks(0, 1, 1) == 0 and iob.blocks(6, 2, 0) == pytest.approx(3)


# -- the inequalities, by brute force ---------------------------------------------------------------------------------

def test_loomis_whitney():
    cube = list(itertools.product(range(2), repeat=3))
    for mask in range(1, 1 << len(cube)):
        S = [p for b, p in enumerate(cube) if mask >> b & 1]
        x, y, z = faces(S)
        assert len(S) ** 2 <= x * y * z
    rng = random.Random(1)
    for _ in range(3000):
        pts = list(itertools.product(*(range(rng.randint(1, 6)) for _ in range(3))))
        S = rng.sample(pts, rng.randint(1, len(pts)))
        x, y, z = faces(S)
        assert len(S) ** 2 <= x * y * z
    p, q, r = 3, 4, 5   # a box meets it with equality
    x, y, z = faces(list(itertools.product(range(p), range(q), range(r))))
    assert (p * q * r) ** 2 == x * y * z


def test_one_blocks_cap():
    """√(x·y·(t/b_C + s)) ≤ α + β√s and ≤ α + μ·s·b_C for every split of k into x·b_A + y·b_B + t; α is reached, and
    the line meets the most over the split at s·b_C = 16k/27, its tangent there."""
    rng = random.Random(2)
    for _ in range(20000):
        k, ba, bb, bc = rng.uniform(1, 1e6), rng.choice(WIDTHS), rng.choice(WIDTHS), rng.choice((8, 16, 32))
        w = [rng.random() for _ in range(3)]
        x, y, t = (k * v / sum(w) for v in w)
        s = rng.choice((0.0, rng.uniform(0, 1e6), rng.uniform(0, 2 * k / bc)))
        lhs = math.sqrt(x / ba * (y / bb) * (t / bc + s))
        assert lhs <= (iob.alpha(k, ba, bb, bc) + iob.beta(k, ba, bb) * math.sqrt(s)) * (1 + 1e-12)
        assert lhs <= (iob.alpha(k, ba, bb, bc) + iob.mu(k, ba, bb, bc) * s * bc) * (1 + 1e-12)
    k = 3e4
    assert math.sqrt((k / 3 / 16) ** 2 * (k / 3 / 32)) == pytest.approx(iob.alpha(k, 16, 16, 32))
    E = 16 * k / 27   # past k/2 the best split reads no partial sums: x·b_A = y·b_B = k/2
    line = iob.alpha(k, 16, 16, 32) + iob.mu(k, 16, 16, 32) * E
    assert (k / 2) * math.sqrt(E / (16 * 16 * 32)) == pytest.approx(line)


def _chain_partition(I, L, J, block_of, ba, bb, bc) -> None:
    """A product's MACs split into blocks by `block_of(i, l, j)`, each output's chain running l = 0, 1, ...: LW on each
    block at the bits it reads (its operand entries, and a partial sum for each chain it touches but doesn't start), and
    the amortized caps M ≤ N·α + β·√(N·|O|) and M ≤ N·α + μ·b_C·|O| at the largest block's bits."""
    blocks = defaultdict(list)
    for p in itertools.product(range(I), range(L), range(J)):
        blocks[block_of(*p)].append(p)
    bits = []
    for S in blocks.values():
        x, y, z = faces(S)
        starts = len({(i, j) for i, l, j in S if l == 0})
        assert len(S) ** 2 <= x * y * z
        bits.append(x * ba + y * bb + (z - starts) * bc)
    k, N = max(bits), len(blocks)
    assert I * L * J <= (N * iob.alpha(k, ba, bb, bc) + iob.beta(k, ba, bb) * math.sqrt(N * I * J)) * (1 + 1e-12)
    assert I * L * J <= (N * iob.alpha(k, ba, bb, bc) + iob.mu(k, ba, bb, bc) * bc * I * J) * (1 + 1e-12)


def test_the_amortized_cap_on_partitions_of_a_chain_product():
    rng = random.Random(3)
    for _ in range(300):
        I, L, J = rng.randint(1, 6), rng.randint(1, 8), rng.randint(1, 6)
        ba, bb, bc = rng.choice(WIDTHS), rng.choice(WIDTHS), rng.choice((8, 16, 32))
        n = rng.randint(1, I * L * J)
        _chain_partition(I, L, J, lambda i, l, j: rng.randrange(n), ba, bb, bc)
        p, q, r = rng.randint(1, I), rng.randint(1, L), rng.randint(1, J)
        _chain_partition(I, L, J, lambda i, l, j: (i // p, l // q, j // r), ba, bb, bc)
        _chain_partition(I, L, J, lambda i, l, j: (i // p, j // r), ba, bb, bc)   # whole chains: the β term's case


def test_products_splitting_one_blocks_bits():
    """Σ_P g_P·(α_P(k_P) + β_P(k_P)·√s_P) ≤ Γ₀(k) + (k/2)·√(Σ_P g_P²·s_P/(b_A·b_B)) and Σ_P g_P·(α_P(k_P) +
    μ_P(k_P)·b_C·s_P) ≤ Γ₀(k) + Σ_P g_P·μ_P(k)·b_C·s_P for Σ k_P = k (arrays disjoint)."""
    rng = random.Random(4)
    for _ in range(5000):
        k = rng.uniform(1e3, 1e6)
        ps = [(rng.choice(WIDTHS), rng.choice(WIDTHS), rng.uniform(1, 2000)) for _ in range(rng.randint(1, 5))]
        w = [rng.random() for _ in ps]
        ks, ss = [k * v / sum(w) for v in w], [rng.choice((0.0, rng.uniform(0, 1e5))) for _ in ps]
        lhs = sum(g * (iob.alpha(kp, a, b) + iob.beta(kp, a, b) * math.sqrt(s)) for (a, b, g), kp, s in zip(ps, ks, ss))
        rhs = max(g * iob.alpha(k, a, b) for a, b, g in ps) + k / 2 * math.sqrt(
            sum(g * g * s / (a * b) for (a, b, g), s in zip(ps, ss)))
        assert lhs <= rhs * (1 + 1e-12)
        lhs = sum(g * (iob.alpha(kp, a, b) + iob.mu(kp, a, b) * s * 32) for (a, b, g), kp, s in zip(ps, ks, ss))
        rhs = max(g * iob.alpha(k, a, b) for a, b, g in ps) + sum(
            g * iob.mu(k, a, b) * s * 32 for (a, b, g), s in zip(ps, ss))
        assert lhs <= rhs * (1 + 1e-12)


def test_the_linear_form_is_its_best_subsets():
    """Products of one width, none made whole (Γ₀ and Λ the same on every subset): the linear form on the set is at
    least on any subset, so chains shorter than Λ·μ·b_C (48 at the console's k and Λ = SHARE·feed) don't lower it."""
    rng = random.Random(7)
    bf16 = {"b_a": 16, "b_b": 16, "b_c": 32, "gates_per_mac": 23682 / 16, "cell_gates": 23682.0, "k_whole": None}
    for _ in range(300):
        ps = []
        for _ in range(rng.randint(1, 5)):
            macs, K = 10 ** rng.uniform(10, 15), rng.choice((4, 16, 48, 64, 512, 65536))
            ps.append({"macs": macs, "outputs": macs / K, "k_dim": K} | bf16)
        whole = iob.bound(ps, *iob.CONSOLE, 1e18)
        for r in range(1, len(ps)):
            for sub in itertools.combinations(ps, r):
                part = iob.bound(list(sub), *iob.CONSOLE, 1e18)
                for lam in ("proved", "numerical"):
                    assert part[lam]["blocks_linear"] <= whole[lam]["blocks_linear"] * (1 + 1e-12)
    assert iob.SHARE * iob.feed(16) * iob.mu(65536, 16, 16) * 32 == pytest.approx(48)


def test_a_linear_layers_three_products_do_at_most_share_of_one():
    """Forward X·W, dX = dY·W and dW = dY·X sharing a block's bits: at most 2/√3 of one product's α, reached."""
    rng = random.Random(5)
    b, bc, k = 16, 32, 1.0
    cap = iob.alpha(k, b, b, bc)
    for _ in range(20000):
        w = [rng.random() ** 2 for _ in range(6)]
        x, wt, y, tf, td, tw = (k * v / sum(w) for v in w)
        macs = sum(math.sqrt(p / b * (q / b) * (t / bc)) for p, q, t in ((x, wt, tf), (y, wt, td), (y, x, tw)))
        assert macs <= iob.SHARE * cap * (1 + 1e-12)
        ss = [rng.choice((0.0, rng.uniform(0, 2 * k / bc))) for _ in range(3)]   # chains each product starts
        macs = sum(math.sqrt(p / b * (q / b) * (t / bc + s))
                   for (p, q, t), s in zip(((x, wt, tf), (y, wt, td), (y, x, tw)), ss))
        assert macs <= iob.SHARE * (cap + iob.mu(k, b, b, bc) * sum(ss) * bc) * (1 + 1e-12)
    u, t = 2 * k / 9, k / 9
    assert 3 * math.sqrt(u / b * (u / b) * (t / bc)) == pytest.approx(iob.SHARE * cap)


def test_a_fed_pair_does_at_most_feed_of_one():
    """A block reading n partial sums of P1 (b_C bits each) and finishing them into P2's b-bit operand entries."""
    rng = random.Random(6)
    bc, k = 32, 1.0
    for b in (4, 8, 16, 24, 32):
        cap = iob.alpha(k, b, b, bc)
        for _ in range(5000):
            w = [rng.random() ** 3 for _ in range(5)]
            x1, y1, nb, y2, z2 = (k * v / sum(w) for v in w)
            n = nb / bc
            macs = math.sqrt(x1 / b * (y1 / b) * n) + math.sqrt(n * (y2 / b) * (z2 / bc))
            assert macs <= iob.feed(b, bc) * cap * (1 + 1e-12)
            s1, s2 = (rng.choice((0.0, rng.uniform(0, 2 * k / bc))) for _ in range(2))   # chains P1 and P2 start
            macs = math.sqrt(x1 / b * (y1 / b) * (n + s1)) + math.sqrt(n * (y2 / b) * (z2 / bc + s2))
            assert macs <= iob.feed(b, bc) * (cap + iob.mu(k, b, b, bc) * (s1 + s2) * bc) * (1 + 1e-12)


# -- captured steps -----------------------------------------------------------------------------------------------------

def _rows(name: str, k: float = K_TINY, **kw):
    g = iob.capture(name, **(TINY | kw))
    return g, iob.rows_of(g, "bfloat16", k)


def _dims(r) -> set:
    return {r.dims["m"], r.dims["k"], r.dims["n"]}


def test_dense_counts_its_block_products_but_not_the_first_layers_or_newton_schulzs():
    g, rows = _rows("dense")
    kinds = Counter(r.kind for r in rows)
    assert set(kinds) == {"fresh", "expansion", "attention", "self"}
    head = [r for r in rows if VOCAB in _dims(r)]   # the tied head's logits, dX and dE
    assert len(head) == 3 * TINY["micro_steps"] and all(r.kind == "fresh" for r in head)
    exp = [r for r in rows if r.kind == "expansion"]   # layer 0's q, k, v: the embedding's rows through a gather, a gain
    assert Counter(r.phase for r in exp) == {"forward": 3 * TINY["micro_steps"], "backward": 3 * TINY["micro_steps"]}
    assert all(min(r.rates) < min(r.widths) for r in exp)
    assert {r.phase for r in rows if r.kind == "self"} == {"optimizer"}
    C = iob.step_gates(g, "bfloat16")
    assert sum(r.macs * r.gates_per_mac for r in rows) < C
    assert min(r.k_whole for r in rows if r.kind == "fresh") == 64   # d: the q/k/v/o and gate/up chains
    m = iob.measure(g, {"tiny": (1e9, K_TINY), "console": iob.CONSOLE})["policies"]
    assert 0.5 < m["tiny"]["kinds"]["fresh"]["share"] < 1 and m["tiny"]["bound"]["proved"]["rho"] > 0
    # at the console's k a block holds a whole 64 × 64 weight: layer 0's q, k, v come whole from the embedding's rows
    # (64 < 3^{3/2}·√(k/32)), so later layers' operands are cheap too, and 64-long chains make operands whole faster
    # than a block reads them (r = 1/3): Λ is void on every product, so the proved bound is the products whose
    # chains are at least d_ff = 192 long (and those with none), the rest still in |C|
    assert m["console"]["kinds"]["expansion"]["share"] > m["tiny"]["kinds"]["expansion"]["share"]
    proved = m["console"]["bound"]["proved"]
    assert m["console"]["bound"]["whole_chain"] is None and proved["lam"] is not None and proved["rho_no_cell"] > 0
    for sub in ("subset", "subset_no_cell"):
        assert proved[sub]["k_whole_min"] == 192 and 0 < proved[sub]["dropped_macs_share"] < 1


def test_lora_counts_only_its_adapters():
    _, rows = _rows("lora")
    assert set(Counter(r.kind for r in rows)) == {"fresh", "theta", "attention", "unaffected"}
    assert all(16 in _dims(r) for r in rows if r.kind == "fresh")
    assert all(r.kind in ("theta", "unaffected") for r in rows if r.kind != "attention" and 16 not in _dims(r))
    first = [r for r in rows if r.affected == (False, True)]   # layer 0's x·aᵀ: x is the frozen embedding's rows
    assert first and all(r.kind == "theta" and r.dims["n"] == 16 for r in first)
    assert all(r.phase == "forward" for r in rows if r.kind == "unaffected")   # layer 0's base q, k, v
    # the rank-16 chains are summed with the frozen base's output, which is paid: no counted operand is made whole by them
    assert min(r.k_whole for r in rows if r.kind == "fresh") > 16


def test_a_frozen_embedding_puts_the_head_at_theta():
    _, rows = _rows("frozen_embed")
    head = [r for r in rows if VOCAB in _dims(r)]
    assert len(head) == 2 * TINY["micro_steps"] and all(r.kind == "theta" for r in head)   # logits and dX; no dE


def test_partial_depth_leaves_the_blocks_below_unaffected(monkeypatch):
    import model

    monkeypatch.setitem(model.SIZES, "tiny3", model.Size("tiny3", 64, 3, 2, 192))
    _, rows = _rows("partial", size="tiny3")
    kinds = Counter(r.kind for r in rows)
    # block 0's seven products and its attention, forward only
    assert kinds["unaffected"] == 9 * TINY["micro_steps"]
    assert {r.phase for r in rows if r.kind == "unaffected"} == {"forward"}
    # block 1's q, k, v read block 0's output through block 1's trained gain: forward and dW
    assert kinds["expansion"] == 6 * TINY["micro_steps"]
    head = [r for r in rows if VOCAB in _dims(r)]
    assert head and all(r.kind == "theta" for r in head)


class _Widening(nn.Module):
    def __init__(self, d: int, wide: int) -> None:
        super().__init__()
        self.first = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, wide, bias=False)
        self.last = nn.Linear(wide, d, bias=False)

    def forward(self, x):
        return self.last(self.up(self.first(x)))


@pytest.mark.parametrize("wide, kind", [(64, "expansion"), (16, "fresh"), (4, "fresh")])
def test_a_frozen_widening_passes_fewer_bits_an_entry_than_its_width(wide, kind):
    """`last` reads up(h) with `up` frozen: an entry costs a block 32·16/wide bits, below its 32 when wide > 16 (at a k
    of 64 bits, a fraction of one weight column, as the console's k is of the s shape's)."""
    torch.manual_seed(0)
    m = _Widening(16, wide)
    m.up.requires_grad_(False)
    g = capture_step(m, torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1),
                     lambda mm, b: F.mse_loss(mm(b[0]), b[1]), (torch.empty(6, 16), torch.empty(6, 16)))
    first, up, last = [r for r in iob.rows_of(g, k=64) if r.phase == "forward"]
    assert (first.kind, up.kind, last.kind) == ("theta", "theta", kind)
    assert last.rates[0] == pytest.approx(min(32, 32 * 16 / wide)) and last.widths == (32, 32)


class _LowRank(nn.Module):
    def __init__(self, d: int, r: int) -> None:
        super().__init__()
        self.first = nn.Linear(d, d, bias=False)
        self.a = nn.Linear(d, r, bias=False)
        self.b = nn.Linear(r, d, bias=False)
        self.last = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.last(self.b(self.a(self.first(x))))


@pytest.mark.parametrize("r, kind", [(4, "expansion"), (64, "fresh")])
def test_a_short_chain_through_a_trained_weight_makes_its_outputs_cheap(r, kind):
    """b's output: x rows of a(·) and y rows of b's weight, r entries each at 32 bits, make x·y entries whole, so within
    the console's k an entry costs a block 4·r²·32·32/k bits: 1 at r = 4, its width 32 at r = 64."""
    torch.manual_seed(0)
    m = _LowRank(256, r)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: F.mse_loss(mm(b[0]), b[1]),
                     (torch.empty(6, 256), torch.empty(6, 256)))
    first, a, b, last = [x for x in iob.rows_of(g) if x.phase == "forward"]
    assert (first.kind, a.kind, b.kind, last.kind) == ("theta", "fresh", "fresh", kind)
    assert last.rates[0] == pytest.approx(min(32, 4 * r * r * 32 * 32 / iob.CONSOLE[1])) and last.k_whole == r


class _Cast(nn.Module):
    """Two trained layers whose second product reads its operands through `cast` (None: as they are) and casts its
    result back to the layers' dtype."""

    def __init__(self, d: int, cast) -> None:
        super().__init__()
        self.first = nn.Linear(d, d, bias=False)
        self.last = nn.Linear(d, d, bias=False)
        self.cast = cast

    def forward(self, x):
        h, w = self.first(x), self.last.weight
        if self.cast is None:
            return F.linear(h, w)
        return F.linear(h.to(self.cast), w.to(self.cast)).to(x.dtype)


def _cast_rows(dtype, cast, k: float = 64):
    from te_core.capture import products

    torch.manual_seed(0)
    m = _Cast(16, cast).to(dtype)
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: F.mse_loss(mm(b[0]), b[1]),
                     (torch.empty(6, 16, dtype=dtype), torch.empty(6, 16, dtype=dtype)))
    return iob.rows_of(g, "bfloat16", k), products.rows(g, k, "bfloat16")


def test_a_widening_cast_keeps_its_products_count_and_a_narrowing_one_pays_its_narrower_width():
    """An operand widened by a cast (bf16 → fp32, `x.float()`) keeps its argument's rate, and a block can cut before
    the cast, so it is paid at the width it passed: every product of the step counts exactly as the uncast step's,
    through io_bound.rows_of and through the capture's rows (`product_kind` reads their rates and widths, so a row a
    capture wrote with `paid` false at the cast's width counts too). An operand narrowed by a cast (fp32 → bf16) pays
    its narrower width."""
    from te_core import formalism

    def counted(rows):
        return sorted(json.dumps({"phase": r.phase, "kind": r.kind} | r.product(), sort_keys=True) for r in rows)

    plain, plain_rows = _cast_rows(torch.bfloat16, None)
    wide, wide_rows = _cast_rows(torch.bfloat16, torch.float32)
    assert counted(wide) == counted(plain)
    last = [r for r in wide if r.kind == "fresh" and r.phase == "forward"]
    assert len(last) == 1 and last[0].widths == (16, 16) and last[0].rates == (16.0, 16.0)
    assert sum(r.kind == "fresh" for r in wide) == sum(r.kind == "fresh" for r in plain) > 1
    for got, rows in ((wide_rows, wide), (plain_rows, plain)):
        assert sorted(formalism.product_kind(r) for r in got["rows"] for _ in range(r["count"])) == \
            sorted(r.kind for r in rows)
        assert all(r["paid"] == [iob.is_paid(x, w) for x, w in zip(r["rates"], r["widths"])] for r in got["rows"])
        old = [r | {"paid": [False, False]} for r in got["rows"]]
        assert [formalism.product_kind(r) for r in old] == [formalism.product_kind(r) for r in got["rows"]]
    narrow, _ = _cast_rows(torch.float32, torch.bfloat16)
    last = [r for r in narrow if r.kind == "fresh" and r.phase == "forward" and r.dims["m"] * r.dims["n"] == 6 * 16]
    assert [(r.widths, r.rates) for r in last] == [((16, 16), (16.0, 16.0))]
    assert [r.product()["b_a"] for r in narrow if r.kind == "fresh"] == [16] * sum(r.kind == "fresh" for r in narrow)


FIXTURE_45M = Path(__file__).resolve().parents[1] / "capture" / "tests" / "fixtures" / "dense_45m.products.json"


def _corrected(name: str, batch: int, ctx: int, micro_steps: int, products=None, C=None) -> tuple[dict, dict]:
    c, k = policy.CONSOLE
    g = iob.capture(name, "s", 32000, batch, ctx, micro_steps)
    if products is None:
        products = [r.product() for r in iob.rows_of(g, "bfloat16", k) if r.kind == "fresh"]
        C = iob.step_gates(g, "bfloat16")
    w = iob.whole_chain_of(iob.chain_classes(g, k), k)
    return w, iob.bound(products, c, k, C, w["whole_chain"])["proved"]


def test_the_45m_steps_whole_making_system():
    """The 45M dense step (the fixture's products, batch 2 at ctx 256): attention's 64-long chains sit below every
    counted operand, so its system's least fixed point gives whole_chain 1.0887 (the one-class rule at its least k_whole,
    256, gave 1.033) and 163,708 without P-cell, 2,870,251 with it. Its LoRA r16 step has no fixed point: void. P-match
    fails only at the embedding's gather, whose rows the system prices at their width."""
    from te_core import formalism

    rows = json.loads(FIXTURE_45M.read_text())
    w, p = _corrected("dense", 2, 256, 1, formalism.products_of_trace(rows)["products"], rows["gates"])
    assert w["e"] == pytest.approx(0.058322155, rel=1e-6) and w["whole_chain"] == pytest.approx(1.08874665, rel=1e-6)
    assert w["r"][64] == pytest.approx(1 / 3) and w["mu"][64] == pytest.approx(0.3451, abs=1e-4)
    assert round(p["rho_no_cell"]) == 163_708 and round(p["rho"]) == 2_870_251
    w, p = _corrected("lora", 2, 256, 1, [{"macs": 1, "outputs": 1, "k_dim": 16, "k_whole": None, "b_a": 16, "b_b": 16,
                                            "b_c": 32, "gates_per_mac": 1.0, "cell_gates": 1.0}], 1.0)
    assert w["mu"] is None and math.isinf(w["whole_chain"]) and p["rho"] is None and p["rho_no_cell"] is None
    c, k = policy.CONSOLE
    assert iob.p_match(iob.capture("dense", "s", 32000, 2, 256, 1), k) == ["embedding"]
    assert iob.p_match(iob.capture("lora", "s", 32000, 2, 256, 1), k) == []


@pytest.mark.slow
def test_the_s_t20_steps_whole_making_system():
    """The s-t20 steps at full size (256 sequences of 1,024 tokens in four micro-batches) at the console: dense 257,670
    without P-cell and 4.75e6 with it (4.10's one-class rule gave 258,250 and 4.87e6), LoRA r16 void. Only the full
    capture has the weight gradients' 65,536-long chains and the products the figures sum; the tiny steps can't."""
    w, p = _corrected("dense", 256, 1024, 4)
    assert w["whole_chain"] == pytest.approx(1.0314051, rel=1e-6)
    assert round(p["rho_no_cell"]) == 257_670 and p["rho"] == pytest.approx(4.7545e6, rel=1e-4)
    w, p = _corrected("lora", 256, 1024, 4)
    assert math.isinf(w["whole_chain"]) and p["rho"] is None and p["rho_no_cell"] is None
