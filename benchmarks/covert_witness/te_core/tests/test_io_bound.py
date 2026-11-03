"""te_core.io_bound: the cap's closed forms, each inequality the bound rests on checked by brute force on small products,
the test vectors (tests/io_bound_vectors.json), and the kinds `rows_of` reads off captured steps of the tiny model."""
from __future__ import annotations

import itertools
import json
import math
import random
from collections import Counter, defaultdict

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


def test_each_vectors_term_solves_its_cap():
    """What a port of the bound checks without this module: N·Γ + B·√N = G at each Λ, ρ = N·c/|C|."""
    for v in json.loads(iob.VECTORS.read_text())["vectors"]:
        i, cap, term = v["inputs"], v["cap"], v["term"]
        G = term["share"] * i["C_gates"]
        assert 0 <= term["share"] < 1, v["name"]
        for lam in ("proved", "numerical"):
            n, L = term[f"{lam}_blocks"], cap[f"lam_{lam}"]
            if not i["products"] or L is None:   # nothing counted, or Λ void
                assert n == 0 and term[f"{lam}_rho"] == 0
                continue
            gamma, B = L * cap["gamma0"], L * i["k"] / 2 * math.sqrt(cap["Q"])
            assert n * gamma + B * math.sqrt(n) == pytest.approx(G, rel=1e-9), v["name"]
            assert term[f"{lam}_rho"] == pytest.approx(n * i["c"] / i["C_gates"], rel=1e-12)
            assert term[f"{lam}_rho_main"] == pytest.approx(G / gamma * i["c"] / i["C_gates"], rel=1e-12)
            assert term[f"{lam}_rho_no_cell"] <= term[f"{lam}_rho"] <= term[f"{lam}_rho_main"]
        assert term["proved_rho"] <= term["numerical_rho"]
    void = [v for v in json.loads(iob.VECTORS.read_text())["vectors"] if v["cap"]["lam_proved"] is None]
    assert [v["cap"]["whole_chain"] for v in void] == [None]


# -- the closed forms ---------------------------------------------------------------------------------------------------

def test_the_cap_at_the_consoles_policy():
    # k = 65,536: bf16 operands into 32-bit sums, 2/2/8 and one bit; c, te_core.policy's session of the 45M reference,
    # is 3.626e11 bf16 MACs
    assert iob.CONSOLE is policy.CONSOLE
    assert iob.alpha(65536, 16, 16, 32) == pytest.approx(35673.28, rel=1e-6)
    assert iob.alpha(65536, 2, 2, 8) == pytest.approx(570772.5, rel=1e-6)
    assert iob.alpha(65536, 1, 1, 1) == pytest.approx(3228776.7, rel=1e-6)
    assert iob.beta(65536, 16, 16) == 2048
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
    """√(x·y·(t/b_C + s)) ≤ α + β√s for every split of k into x·b_A + y·b_B + t, and α is reached."""
    rng = random.Random(2)
    for _ in range(20000):
        k, ba, bb, bc = rng.uniform(1, 1e6), rng.choice(WIDTHS), rng.choice(WIDTHS), rng.choice((8, 16, 32))
        w = [rng.random() for _ in range(3)]
        x, y, t = (k * v / sum(w) for v in w)
        s = rng.choice((0.0, rng.uniform(0, 1e6)))
        lhs = math.sqrt(x / ba * (y / bb) * (t / bc + s))
        assert lhs <= (iob.alpha(k, ba, bb, bc) + iob.beta(k, ba, bb) * math.sqrt(s)) * (1 + 1e-12)
    k = 3e4
    assert math.sqrt((k / 3 / 16) ** 2 * (k / 3 / 32)) == pytest.approx(iob.alpha(k, 16, 16, 32))


def _chain_partition(I, L, J, block_of, ba, bb, bc) -> None:
    """A product's MACs split into blocks by `block_of(i, l, j)`, each output's chain running l = 0, 1, ...: LW on each
    block at the bits it reads (its operand entries, and a partial sum for each chain it touches but doesn't start), and
    the amortized cap M ≤ N·α + β·√(N·|O|) at the largest block's bits."""
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
    """Σ_P g_P·(α_P(k_P) + β_P(k_P)·√s_P) ≤ Γ₀(k) + (k/2)·√(Σ_P g_P²·s_P/(b_A·b_B)) for Σ k_P = k (arrays disjoint)."""
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
    # than a block reads them (r = 1/3): the proved bound is void
    assert m["console"]["kinds"]["expansion"]["share"] > m["tiny"]["kinds"]["expansion"]["share"]
    assert m["console"]["bound"]["proved"]["lam"] is None and m["console"]["bound"]["proved"]["rho"] == 0


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
