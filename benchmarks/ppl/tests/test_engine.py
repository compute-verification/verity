import math
import types

import numpy as np
import pytest
import torch

import engine
import matmul
import methodology
import model as M
import report
import run
import spec
import steps


def evaluate(snap, ids, arms, seeds=(0,), calib=None, windows=None, **kw):
    windows = list(range(len(ids))) if windows is None else windows
    return engine.evaluate(snap, ids, windows, [spec.load(a) if isinstance(a, str) else a for a in arms], seeds, calib,
                           device="cpu", log=lambda s: None, hf_check=kw.pop("hf_check", False), **kw)


def ppl(nll: np.ndarray) -> float:
    return float(np.exp(nll.mean(dtype=np.float64)))


def float64_nll(snap, ids) -> np.ndarray:
    """The model in float64 with explicit GQA, causal softmax and RoPE: an independent reference for the bf16 stream."""
    cfg = M.Config.read(snap)
    W = M.Weights(snap)
    t = {k: v.double() for k, v in W.read(list(W.where)).items()}
    x = t["model.embed_tokens.weight"][torch.from_numpy(ids)]
    T = ids.shape[1]
    inv = M.inv_freq(cfg).double()
    f = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
    cos, sin = torch.cat([f, f], -1).cos(), torch.cat([f, f], -1).sin()

    def norm(h, w):
        return w * h / torch.sqrt((h * h).mean(-1, keepdim=True) + cfg.eps)

    def lin(p, n, h):
        y = h @ t[f"{p}{n}.weight"].T
        return y + t[f"{p}{n}.bias"] if f"{p}{n}.bias" in t else y

    for li in range(cfg.layers):
        p = f"model.layers.{li}."
        h = norm(x, t[p + "input_layernorm.weight"])
        q, k, v = (lin(p, f"self_attn.{n}_proj", h).view(len(ids), T, -1, cfg.head_dim).transpose(1, 2) for n in "qkv")
        q, k = q * cos + M.rotate_half(q) * sin, k * cos + M.rotate_half(k) * sin
        k, v = k.repeat_interleave(cfg.rep, 1), v.repeat_interleave(cfg.rep, 1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        a = (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(len(ids), T, -1)
        x = x + lin(p, "self_attn.o_proj", a)
        h = norm(x, t[p + "post_attention_layernorm.weight"])
        x = x + lin(p, "mlp.down_proj", torch.nn.functional.silu(lin(p, "mlp.gate_proj", h)) * lin(p, "mlp.up_proj", h))
    logits = norm(x, t["model.norm.weight"]) @ t["lm_head.weight"].T
    return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), torch.from_numpy(ids)[:, 1:].reshape(-1),
                                             reduction="none").view(len(ids), -1).numpy()


@pytest.mark.parametrize("which", ["llama", "qwen"])
def test_the_bf16_stream_is_the_model(which, request, ids):
    snap = request.getfixturevalue(which)
    r = evaluate(snap, ids[:4], [])
    want = float64_nll(snap, ids[:4])
    assert r["nll"]["bf16/s0"].shape == (4, 63)
    assert abs(ppl(r["nll"]["bf16/s0"]) / ppl(want) - 1) < 1e-2
    assert 100 < ppl(want) < 1000


def passed(r):
    return {n: g["failed"] for n, g in r["gates"].items() if g["failed"]} == {}


@pytest.mark.parametrize("which", ["llama", "qwen"])
def test_the_adopted_design_passes_every_gate(which, request, ids, calib):
    snap = request.getfixturevalue(which)
    arms = ["int8-all", "adopted", "adopted-linears", "adopted-attention"]
    r = evaluate(snap, ids[:4], arms, calib=calib)
    assert passed(r), r["gates"]
    g = r["gates"]
    cfg = M.Config.read(snap)
    # per (arm with the adopted attention, layer, KV group): Q's peel and P's two blocks (key 0, then the rest)
    assert g["peel"]["checked"] == 2 * cfg.layers * cfg.kv_heads * 3
    assert g["int8_range"]["checked"] == g["accumulator"]["checked"] > 0 and g["product_exact"]["checked"] > 0
    bf = ppl(r["nll"]["bf16/s0"])
    for k in ("int8-all/s0", "adopted/s0", "adopted-linears/s0", "adopted-attention/s0"):
        assert k in r["nll"] and abs(ppl(r["nll"][k]) / bf - 1) < 0.05, k
        assert r["divergence"][k][-1] > 0
    assert 0 < r["attn_rel_err"]["adopted/s0"] < 0.1


def test_q_dithered_attention_passes_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["adopted-qkd"], calib=calib)
    assert passed(r), r["gates"]
    cfg = M.Config.read(llama)
    # P's two blocks per (layer, KV group); Q carries no peel
    assert r["gates"]["peel"]["checked"] == 2 * cfg.layers * cfg.kv_heads
    assert abs(ppl(r["nll"]["adopted-qkd/s0"]) / ppl(r["nll"]["bf16/s0"]) - 1) < 0.05


def test_dithered_weights_pass_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["adopted-bd", "adopted-qkd"], calib=calib)
    assert passed(r), r["gates"]
    assert abs(ppl(r["nll"]["adopted-bd/s0"]) / ppl(r["nll"]["bf16/s0"]) - 1) < 0.05
    assert not np.array_equal(r["nll"]["adopted-bd/s0"], r["nll"]["adopted-qkd/s0"])


def test_a_noised_head_passes_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["adopted-bd-lm", "adopted-bd", "co", "ncp", "cb", "co32c", "co32c-pv", "co32c-pv256"], calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in ("adopted-bd-lm", "co", "ncp", "cb", "co32c", "co32c-pv", "co32c-pv256"):
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.05, a
    assert not np.array_equal(r["nll"]["adopted-bd-lm/s0"], r["nll"]["adopted-bd/s0"])


def test_wide_dithers_pass_every_gate(llama, ids, calib):
    arms = ["adopted-bd-lm", "adopted-bd-lm-d2", "adopted-bd-lm-d4", "co32c-d4"]
    r = evaluate(llama, ids[:4], arms, calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in arms:
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.1, a
    assert not np.array_equal(r["nll"]["adopted-bd-lm/s0"], r["nll"]["adopted-bd-lm-d2/s0"])


def test_linear_variants_pass_their_gates(llama, ids, calib):
    arms = [spec.parse({"linear": [{"step": "rotate", "kind": "h"}, {"step": "int8", "group": 32}]}, "grouped"),
            spec.parse({"linear": [{"step": "smooth", "alpha": 0.5}, {"step": "rotate"},
                                   {"step": "int8", "weight_top": 63, "gptq": True}]}, "gptq"),
            spec.parse({"linear": [{"step": "int8", "act": "dither", "act_top": 64}],
                        "attention": [{"step": "int8", "p_block": 32, "p_top": 119, "v_scale": "head", "k": "dither"},
                                      {"step": "peel", "operand": "p", "width": 8}]}, "mixed"),
            spec.parse({"linear": [{"step": "int8", "act": "dither", "weight": "dither", "weight_top": 63},
                                   {"step": "peel", "alphabet": "bit"}],
                        "attention": [{"step": "int8", "q": "dither", "k": "dither", "v": "dither"}]}, "dither-peel"),
            spec.parse({"linear": [{"step": "int8", "act": "dither", "density": 0.5}],
                        "attention": [{"step": "int8", "q": "dither", "q_top": 63, "k": "dither", "p": "dither",
                                       "p_sink": True, "v": "dither", "v_top": 63}]}, "thin-q-pv")]
    r = evaluate(llama, ids[:4], arms, calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in arms:
        assert abs(ppl(r["nll"][f"{a.name}/s0"]) / bf - 1) < 0.1, a.name


def test_noise_corrected_whole_is_the_plain_product(llama, ids):
    lin = {"step": "int8", "act_top": 95, "weight_top": 95}
    plain = spec.parse({"linear": [lin]}, "t95")
    arms = [plain] + [spec.parse({"linear": [lin, {"step": "noise", "x_width": 32, "w_width": 32, "period": p,
                                                   "correct": c}]}, f"n-{c}-{p}")
                      for c, p in (("all", 0), ("all", 32), ("ew+xf", 0), ("none", 32))]
    r = evaluate(llama, ids[:4], arms)
    assert passed(r), r["gates"]
    nll = {a.name: r["nll"][f"{a.name}/s0"] for a in arms}
    assert np.array_equal(nll["n-all-0"], nll["t95"]) and np.array_equal(nll["n-all-32"], nll["t95"])
    assert ppl(nll["t95"]) < ppl(nll["n-ew+xf-0"]) < ppl(nll["n-none-32"])
    wide = spec.parse({"linear": [{"step": "int8", "act_top": 127}, {"step": "noise", "x_width": 1}]}, "wide")
    assert not passed(evaluate(llama, ids[:4], [wide]))


def test_ssn_a_w0_is_the_formats_pattern():
    def literal(n, k):
        s = -(-k // 28)
        w0 = np.zeros((s, 4, n), dtype=np.int64)
        zero = sum(6 * 13 ** e for e in range(4))
        for j in range(n):
            c = j + 1 if j + 1 < zero else j + 2
            for e in range(4):
                w0[0, e, j] = c % 13 - 6
                c //= 13
            for st in range(1, s):
                w0[st, (j + st) % 4, j] = 1
        return w0

    for n, k, off in ((40, 100, 0), (24, 2048, 2048), (16, 64, 14270)):
        assert np.array_equal(steps.ssn_a_w0(n, k, off).numpy(), literal(off + n, k)[:, :, off:]), (n, k, off)
    codes = steps.ssn_a_w0(28559, 28)[0].T
    assert len({tuple(c) for c in codes.tolist()}) == 28559 and not bool((codes == 0).all(1).any())
    with pytest.raises(ValueError, match="outrun"):
        steps.ssn_a_w0(28560, 28)


def test_ssn_code5_is_the_formats_code():
    def balanced(n, lanes=5):
        j = np.arange(n)
        r = np.where(j % 2 == 0, j // 2 + 1, -(j // 2 + 1))
        out = []
        for _ in range(lanes):
            d = (r + 6) % 13 - 6
            out.append(d)
            r = (r - d) // 13
        assert (r == 0).all()
        return np.stack(out)

    for n, off in ((40, 0), (512, 2048), (16, 128240)):
        assert np.array_equal(steps.ssn_code5(n, off).numpy(), balanced(off + n)[:, off:]), (n, off)
    codes = steps.ssn_code5(371292).T
    assert len({tuple(c) for c in codes.tolist()}) == 371292 and not bool((codes == 0).all(1).any())
    assert float(codes.abs().max()) == 6
    with pytest.raises(ValueError, match="outrun"):
        steps.ssn_code5(371293)


def test_column_signs_are_one_public_pattern():
    whole = steps.column_signs(66, 3072)
    assert set(whole.unique().tolist()) == {-1.0, 1.0} and abs(float(whole.mean())) < 0.01
    assert torch.equal(steps.column_signs(66, 512, 2048), whole[:, 2048:2560])
    assert not torch.equal(whole[0], whole[1])


def test_fused_projections_draw_one_calls_coins():
    """With `fused`, q, k and v (gate and up) each serve their own columns of the one fused call's noise; `denoise`
    serves the product."""
    rows = {"self_attn.q_proj": 96, "self_attn.k_proj": 32, "self_attn.v_proj": 32}
    run = types.SimpleNamespace(seed=1, li=3, window=5, rotations={}, gates=matmul.G.Gates(),
                                arm=types.SimpleNamespace(name="t"), checking=False, cache={},
                                layer=types.SimpleNamespace(weight=lambda name: torch.empty(rows[name], 0)))
    m, k = 16, 96
    x = torch.ones(m, k)

    def noise(name, n, rule, fused, denoise=False):
        mm = matmul.MM("linear", name, run, x, torch.ones(n, k)).integer(x, torch.ones(n, k))
        return steps.served_linear(mm, lambda _m: torch.zeros(m, n, dtype=torch.int64), rule=rule, fused=fused,
                                   denoise=denoise)

    for rule in ("ssn_1l", "ssn_1l5", "ssn_1l_hw", "ssn_1l_tag", "ssn_1l_tag_pd", "aorb_sv4", "ssn_pd", "ssn_a",
                 "raw_byte"):
        whole = noise("self_attn.qkv", k + 2 * 32, rule, False)
        q, kk, v = (noise(f"self_attn.{p}_proj", n, rule, True) for p, n in (("q", k), ("k", 32), ("v", 32)))
        assert torch.equal(torch.cat([q, kk, v], 1), whole), rule
        apart = noise("self_attn.k_proj", 32, rule, False)
        assert not torch.equal(apart, kk), rule
        assert not bool((kk == 0).all()) and bool((noise("self_attn.k_proj", 32, rule, True, True) == 0).all()), rule
    assert run.gates.log["accumulator"]["failed"] == 0
    lead, pd = (noise("mlp.down_proj", 40, r, False) for r in ("ssn_1l", "ssn_pd"))
    rows = (lead - pd) - (lead - pd)[:1]
    assert torch.equal(rows, rows[:, :1].expand_as(rows)) and not bool((rows == 0).all())
    tag = noise("mlp.down_proj", 40, "ssn_1l_tag", False)
    s = 1 + -(-(k - steps.SSN_TAG_DATA) // steps.SSN_1L_DATA)
    assert s == 1 + -(-(k - steps.SSN_DATA) // steps.SSN_1L_DATA)
    c = steps.ssn_a_w0(40, steps.SSN_DATA)[0].sum(0)
    mm = matmul.MM("linear", "mlp.down_proj", run, x, torch.ones(40, k))
    lanes, _ = mm.coins((m, s - 1)).double().sum(1, keepdim=True), mm.coins((m, 4))
    g = steps.tag_matrix(mm.coins((steps.TAG_LANES,), 2 ** 16), 40)
    want = steps.row_tags(m) @ g + (s - 1) - torch.round(c / 2 + 1.5 * (s - 1)) + torch.round(c / 2 + (s - 1) / 2)
    assert torch.equal(tag - lead, want) and len({tuple(r) for r in (tag - lead).tolist()}) == m
    tag_pd = noise("mlp.down_proj", 40, "ssn_1l_tag_pd", False)
    assert torch.equal(tag - tag_pd, lanes + (s - 1) - torch.round(c / 2 + 1.5 * (s - 1)) + torch.round(c / 2))
    mm2 = matmul.MM("linear", "mlp.down_proj", run, x, torch.ones(40, k)).integer(x, torch.ones(40, k))
    nz2 = steps.served_linear(mm2, lambda _m: torch.zeros(m, 40, dtype=torch.int64), rule="ssn_1l_tag", nz=2)
    assert torch.equal(nz2 - tag, lanes - torch.round(c / 2 + 2 * (s - 1)) + torch.round(c / 2 + 1.5 * (s - 1)))


def test_row_tags_and_g():
    tags = steps.row_tags(28560)
    assert tags.shape == (28560, 4) and len({tuple(r) for r in tags.tolist()}) == 28560
    assert not bool((tags == 0).all(1).any()) and float(tags.abs().max()) == 6
    with pytest.raises(ValueError, match="outrun"):
        steps.row_tags(28561)
    key = torch.tensor([3.0, 70.0, 65535.0, 0.0])
    g = steps.tag_matrix(key, 3072)
    assert set(g.unique().tolist()) == set(range(-3, 4)) and abs(float(g.mean())) < 0.1
    assert torch.equal(steps.tag_matrix(key, 512, 2048), g[:, 2048:2560])
    assert not torch.equal(steps.tag_matrix(key + 1, 3072), g)


def test_sv4_code_and_aorb_sides():
    c = steps.sv_code(4, 88)
    norms = (c * c).sum(0)
    assert c.shape == (4, 88) and len({tuple(v) for v in c.T.tolist()}) == 88
    assert [int((norms == r).sum()) for r in (1, 2, 3, 4)] == [8, 24, 32, 24]
    assert bool((norms[1:] >= norms[:-1]).all()) and torch.equal(steps.sv_code(4, 300)[:, :88], c)
    # Llama-3.2-1B's calls: k 2048 gives S = 67, k 8192 (down) S = 265
    assert [steps.aorb_side_b(2048, n, 67) for n in (3072, 2048, 16384, 512)] == [False, False, False, True]
    assert [steps.aorb_side_b(8192, n, 67) for n in (16384, 8192, 3072, 2048)] == [False, False, True, True]
    assert not steps.aorb_side_b(2048, 2048, 265) and steps.aorb_side_b(8192, 2048, 265)


@pytest.mark.parametrize("m, n, side_b", [(16, 40, False), (64, 20, True)])
def test_aorb_noise_is_its_formula(m, n, side_b):
    """`aorb_sv4` on each side against its literal formula: side A a row's noise against the columns' sv4 vectors and
    row tags against G, side B the same with rows and columns exchanged."""
    k = 96
    s = 1 + -(-(k - steps.SSN_TAG_DATA) // steps.SSN_1L_DATA)
    assert steps.aorb_side_b(m, n, s) == side_b
    run = types.SimpleNamespace(seed=2, li=1, window=4, rotations={}, gates=matmul.G.Gates(),
                                arm=types.SimpleNamespace(name="t"), checking=False, cache={})
    x = torch.ones(m, k)
    mm = matmul.MM("linear", "mlp.down_proj", run, x, torch.ones(n, k)).integer(x, torch.ones(n, k))
    got = steps.served_linear(mm, lambda _m: torch.zeros(m, n, dtype=torch.int64), rule="aorb_sv4")
    mm = matmul.MM("linear", "mlp.down_proj", run, x, torch.ones(n, k))
    a, b = (n, m) if side_b else (m, n)
    lanes = (1 + 2 * mm.coins((a, s - 1))).double().sum(1, keepdim=True)
    a1 = mm.coins((a, 4)).double()
    key = mm.coins((steps.TAG_LANES,), 2 ** 16)
    code = steps.sv_code(4, b)
    want = (a1 @ code + lanes + steps.row_tags(a) @ steps.tag_matrix(key, b)
            - torch.round(code.sum(0) / 2 + 2 * (s - 1)))
    assert torch.equal(got, want.T if side_b else want) and run.gates.log["accumulator"]["failed"] == 0


@pytest.mark.parametrize("coin_a", [1, 127])
def test_blc_coin_lanes_are_a_tile_column_term(coin_a):
    """`ssn_1l_tag` with blc4a's coin lanes against its k32 layout written out: step 0 holds 24 data lanes, the row tags
    against G and a1 against C; each later step 28 data lanes, three coin lanes against A' = (1, coin_a, 1) (odd coins
    on +-127 a 64-row tile, step and column) and the row lane 1 + l_t against 1. The accumulator less the public bias
    and each tile's weighted column sum of coins is what the step serves, and de-noised the product; at k 86 it takes
    one step more than the lead's 31 data lanes."""
    g = torch.Generator().manual_seed(3)
    m, k, n = 128, 86, 40
    s = 1 + -(-(k - steps.SSN_TAG_DATA) // 28)
    assert s == 2 + -(-(k - steps.SSN_TAG_DATA) // steps.SSN_1L_DATA)
    x = torch.randint(-127, 128, (m, k), generator=g).float()
    w = torch.randint(-127, 128, (n, k), generator=g).float()
    run = types.SimpleNamespace(seed=4, li=2, window=1, rotations={}, gates=matmul.G.Gates(),
                                arm=types.SimpleNamespace(name="t"), checking=False, cache={})

    def served(denoise, rows=m):
        mm = matmul.MM("linear", "mlp.down_proj", run, x[:rows], w).integer(x[:rows], w)
        return steps.served_linear(mm, lambda _m: x[:rows].long() @ w.long().T, rule="ssn_1l_tag", coin_lanes=3,
                                   coin_a=coin_a, denoise=denoise)

    mm = matmul.MM("linear", "mlp.down_proj", run, x, w)
    lanes, a1, key = mm.coins((m, s - 1)).double(), mm.coins((m, 4)).double(), mm.coins((steps.TAG_LANES,), 2 ** 16)
    code = steps.ssn_a_w0(n, steps.SSN_DATA)[0]
    coins = 2 * torch.randint(0, 128, (m // 64, s - 1, 3, n), generator=g).double() - 127
    xa, wa = torch.zeros(m, 32 * s, dtype=torch.float64), torch.zeros(n, 32 * s, dtype=torch.float64)
    xa[:, :24], wa[:, :24] = x[:, :24], w[:, :24]
    xa[:, 24:28], wa[:, 24:28] = steps.row_tags(m), steps.tag_matrix(key, n).T
    xa[:, 28:32], wa[:, 28:32] = a1, code.T
    for t in range(1, s):
        lo = 24 + 28 * (t - 1)
        d = min(28, k - lo)
        xa[:, 32 * t:32 * t + d], wa[:, 32 * t:32 * t + d] = x[:, lo:lo + d], w[:, lo:lo + d]
        xa[:, 32 * t + 28:32 * t + 31] = torch.tensor([1.0, coin_a, 1.0], dtype=torch.float64)
        xa[:, 32 * t + 31], wa[:, 32 * t + 31] = 1 + lanes[:, t - 1], 1
    folded = torch.empty(m, n, dtype=torch.float64)
    weights = torch.tensor([1.0, coin_a, 1.0], dtype=torch.float64)[None, :, None]
    for tile in range(m // 64):
        wa[:, [32 * t + c for t in range(1, s) for c in range(28, 31)]] = coins[tile].reshape(-1, n).T
        r = slice(64 * tile, 64 * (tile + 1))
        folded[r] = xa[r] @ wa.T - (weights * coins[tile]).sum((0, 1))
    assert torch.equal(served(False), folded - torch.round(code.sum(0) / 2 + 1.5 * (s - 1)))
    assert torch.equal(served(True), (x.long() @ w.long().T).double())
    assert run.gates.log["accumulator"]["failed"] == 0
    with pytest.raises(ValueError, match="leave a tile under 64"):
        served(False, rows=96)


def test_round_significand():
    s = torch.rand(1000, generator=torch.Generator().manual_seed(0)) * 1e-2 + 1e-6
    assert torch.equal(steps.round_significand(s, 24), s)
    r = steps.round_significand(s, 22)
    assert float(((r.double() - s.double()) / s.double()).abs().max()) <= 2 ** -22 and not torch.equal(r, s)
    assert torch.equal(steps.round_significand(r, 22), r)
    assert bool(((-steps.MAGIC * r).double() == -steps.MAGIC * r.double()).all())
    assert not bool(((-steps.MAGIC * s).double() == -steps.MAGIC * s.double()).all())


def test_magic_epilogue_is_vllms_where_its_constant_is_exact():
    """The magic conversion reads MAGIC + P; under 22-bit scales every column's FFMA constant is exact and the output
    is vLLM's s_a (s_w float(P)) bit for bit; raw scales leave some constants inexact, and only those columns differ.
    Row 0 (all 127) passes the Cauchy-Schwarz bound and falls back; its product with column 0 (all 127) is wide."""
    small = torch.tensor([-2 ** 22, -1, 0, 1, 2 ** 22 - 1])
    assert torch.equal((small + steps.MAGIC_BITS).to(torch.int32).view(torch.float32).double(),
                       steps.MAGIC + small.double())
    g = torch.Generator().manual_seed(0)
    m, k, n = 40, 1024, 48
    xq = torch.randint(-15, 16, (m, k), generator=g, dtype=torch.int8)
    wq = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    xq[0], wq[0] = 127, 127
    sa, sw = torch.rand(m, generator=g) * 1e-2 + 1e-4, torch.rand(n, generator=g) * 1e-3 + 1e-5
    p = xq.long() @ wq.long().T

    def epilogue(sw):
        run = types.SimpleNamespace(seed=0, li=0, window=0, rotations={}, gates=matmul.G.Gates(),
                                    arm=types.SimpleNamespace(name="t"), checking=False, cache={})
        mm = matmul.MM("linear", "mlp.down_proj", run, xq.float(), wq.float())
        return steps.magic_epilogue(mm, p, sa, sw, xq, wq), run.gates.log

    def vllm(sw):
        return (p.float() * sw[None, :]) * sa[:, None]

    s22 = steps.round_significand(sw, 22)
    y, log = epilogue(s22)
    assert torch.equal(y, vllm(s22)) and log["epilogue"]["failed"] == 0 and log["epilogue_differs"]["failed"] == 0
    assert log["epilogue_fallback"]["by"] == {"t": [m, 1]} and log["epilogue_wide"]["by"] == {"t": [m * n, 1]}
    tallied = matmul.G.Gates()
    tallied.tally("epilogue_differs", 4, 2)
    assert tallied.passed and tallied.log["epilogue_differs"]["failed"] == 2
    y, log = epilogue(sw)
    exact = (-steps.MAGIC * sw).double() == -steps.MAGIC * sw.double()
    assert 0 < int(exact.sum()) < n and log["epilogue"]["failed"] == 0
    assert torch.equal(y[:, exact], vllm(sw)[:, exact]) and not torch.equal(y, vllm(sw))
    assert torch.equal(y[0], vllm(sw)[0])


def test_u8pv_is_the_online_softmax_kernel():
    """u8pv's closed form on the softmax's P (P / R for exp(S - m)) is the tiled kernel's loop on S: blocks in order,
    the running max over keys 1.., O and l rescaled at every new max, the sink added in float at the end."""
    g = torch.Generator().manual_seed(0)
    rep, T, d, block = 2, 300, 16, 112
    S = torch.randn(rep, T, T, generator=g, dtype=torch.float64) * 3
    S[..., 0] += 6
    V = torch.randn(T, d, generator=g)
    causal = torch.arange(T)[None, :] <= torch.arange(T)[:, None]
    P = torch.softmax(S.masked_fill(~causal, float("-inf")), -1).float()
    run = types.SimpleNamespace(seed=0, li=0, window=0, rotations={}, gates=matmul.G.Gates(),
                                arm=types.SimpleNamespace(name="t"), checking=False, cache={})
    extra = {"rep": rep, "rows": (0, T), "causal": causal, "s": T, "d": d}
    got = steps.u8_pv(matmul.MM("pv", "pv", run, P, V, extra=extra), matmul.terminal, block=block).double()
    sv = V.double().abs().amax(1) / 127
    v8 = torch.round(V.double() / sv[:, None]).clamp(-127, 127)
    want = torch.empty(rep, T, d, dtype=torch.float64)
    for h in range(rep):
        for i in range(T):
            s, m, o, l = S[h, i], None, torch.zeros(d, dtype=torch.float64), 0.0
            for a in range(0, T, block):
                lo, hi = max(a, 1), min(a + block, i + 1)
                if lo >= hi:
                    continue
                new = float(s[lo:hi].max()) if m is None else max(m, float(s[lo:hi].max()))
                if m is not None:
                    o, l = o * math.exp(m - new), l * math.exp(m - new)
                m = new
                p = torch.exp(s[lo:hi] - m)
                top = float(sv[a:a + block].max())
                pt = torch.round(255 * p * sv[lo:hi] / top)
                o, l = o + (pt @ v8[lo:hi]) * top / 255, l + float(p.sum())
            m = float(s[0]) if m is None else m
            p0 = math.exp(float(s[0]) - m)
            want[h, i] = (o + p0 * V[0].double()) / (l + p0)
    err = (got - want).abs()
    assert float(err.max()) < 2 * float(V.abs().max()) / 255 and float(err.median()) < 1e-6
    assert run.gates.log["grid_range"]["failed"] == 0 and run.gates.log["grid_range"]["checked"] == 3


def test_attention_formats_pass_their_gates(llama, ids):
    names = ("int8-qk-had-v1", "int8-pv-u8-v1", "int8-attn-v1", "dither-int8-v1", "dither-int8-pv-v1", "ssn-all-v1")
    small = spec.parse({"linear": [{"step": "int8"}], "attention": [{"step": "u8pv", "block": 16}]}, "u8pv-16")
    noised = spec.parse({"linear": [{"step": "int8"}], "attention": [{"step": "u8pv", "block": 16},
                                                                          {"step": "served", "matmuls": ["pv"]}]},
                        "u8pv-16-served")
    arms = ["int8-channel127-v1", *names, small, noised]
    r = evaluate(llama, ids[:4], arms)
    assert passed(r), r["gates"]
    nll = {k.split("/")[0]: v for k, v in r["nll"].items()}
    assert all(not np.array_equal(nll[n], nll["int8-channel127-v1"]) for n in (*names, "u8pv-16", "u8pv-16-served"))
    assert not np.array_equal(nll["u8pv-16"], nll["u8pv-16-served"])
    assert abs(ppl(nll["u8pv-16"]) / ppl(nll["int8-channel127-v1"]) - 1) < 0.05


def test_served_noise_stays_in_the_product(llama, ids):
    stock = spec.parse({"linear": [{"step": "int8"}]}, "stock")
    rules = (("ssn_a", 1), ("ssn_a", 4), ("ssn_1l", 1), ("ssn_1l", 4), ("ssn_1l5", 1), ("ssn_1l_hw", 1),
             ("ssn_1l_tag", 1), ("ssn_pd", 1), ("ssn2", 1), ("raw_byte", 1))
    arms = [stock] + [spec.parse({"linear": [{"step": "int8"}, {"step": "served", "rule": r, "bits": b}]}, f"{r}-{b}")
                      for r, b in rules]
    arms.append(spec.parse({"linear": [{"step": "int8"}, {"step": "served", "rule": "ssn_1l", "fused": True}]},
                           "ssn_1l-fused"))
    head = [{"step": "int8"}, {"step": "int8", "linears": ["lm_head"]}]
    arms.append(spec.parse({"linear": head}, "head"))
    arms.append(spec.parse({"linear": head + [{"step": "served", "rule": "ssn_1l5", "linears": ["lm_head"]}]},
                           "head-ssn_1l5"))
    r = evaluate(llama, ids[:4], arms)
    assert passed(r), r["gates"]
    nll = {a.name: r["nll"][f"{a.name}/s0"] for a in arms}
    assert all(not np.array_equal(nll[a.name], nll["stock"]) for a in arms[1:])
    assert not np.array_equal(nll["ssn_1l-fused"], nll["ssn_1l-1"])
    assert not np.array_equal(nll["head-ssn_1l5"], nll["head"])
    rise = {a.name: abs(ppl(nll[a.name]) / ppl(nll["stock"]) - 1) for a in arms[1:]}
    assert max(rise["ssn_a-1"], rise["ssn_1l-1"], rise["ssn_1l5-1"]) < rise["raw_byte-1"]


def test_a_denoised_lead_serves_stock(llama, ids):
    """De-noised, the lead serves stock bit for bit; under the magic epilogue it serves vLLM's epilogue bit for bit at
    22-bit scales, with blc4a's coin lanes (either weighting) too, and the tallies count each arm's own outputs."""
    xu = ("ssn-1l-tag-magic-s22-v1", "ssn-1l-tag-blc4a-xu-v1", "ssn-1l-tag-blc4a-w127-xu-v1")
    arms = ["int8-channel127-v1", "ssn-1l-tag-dn-v1", "int8-vllm-v1", "int8-vllm-s22-v1", "ssn-1l-tag-magic-v1", *xu,
            "aorb-sv4-v1", "ssn-1l-tag-v1", "ssn-1l-tag-blc4a-v1"]
    r = evaluate(llama, ids[:4], arms)
    g = r["gates"]
    assert passed({"gates": {n: v for n, v in g.items() if n not in matmul.G.TALLIES}}), g
    nll = {k.split("/")[0]: v for k, v in r["nll"].items()}
    assert np.array_equal(nll["ssn-1l-tag-dn-v1"], nll["int8-channel127-v1"])
    assert all(np.array_equal(nll[a], nll["int8-vllm-s22-v1"]) for a in xu)
    for noisy in ("ssn-1l-tag-v1", "aorb-sv4-v1", "ssn-1l-tag-blc4a-v1"):
        assert not np.array_equal(nll[noisy], nll["int8-channel127-v1"]), noisy
    assert not np.array_equal(nll["ssn-1l-tag-blc4a-v1"], nll["ssn-1l-tag-v1"])
    by = g["epilogue_differs"]["by"]
    assert set(by) == {"ssn-1l-tag-magic-v1", *xu} and all(by[a][1] == 0 for a in xu)
    assert by["ssn-1l-tag-magic-v1"][0] == by["ssn-1l-tag-magic-s22-v1"][0] > 0
    assert g["epilogue"]["checked"] > 0 and g["epilogue"]["failed"] == 0


def test_seeds(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["int8-linears", "adopted-linears"], seeds=(0, 1), calib=calib)
    assert set(r["nll"]) == {"bf16/s0", "int8-linears/s0", "adopted-linears/s0", "adopted-linears/s1"}
    assert not np.array_equal(r["nll"]["adopted-linears/s0"], r["nll"]["adopted-linears/s1"])
    again = evaluate(llama, ids[:4], ["adopted-linears"], seeds=(0,), calib=calib)
    assert np.array_equal(again["nll"]["adopted-linears/s0"], r["nll"]["adopted-linears/s0"])
    assert np.array_equal(again["nll"]["bf16/s0"], r["nll"]["bf16/s0"])


def test_shards_are_one_run(llama, ids, calib):
    """Two GPUs' shards (contiguous whole groups of REF_BATCH windows) give each window the NLL one run gives it."""
    assert engine.shards(16, 2) == [list(range(8)), list(range(8, 16))]
    assert engine.shards(4, 2) == [[0, 1, 2, 3]] and engine.shards(10, 2) == [list(range(8)), [8, 9]]
    arms = ["adopted"]
    whole = evaluate(llama, ids, arms, calib=calib)
    parts = [evaluate(llama, ids[s], arms, calib=calib, windows=s) for s in engine.shards(len(ids), 2)]
    for k in whole["nll"]:
        assert np.array_equal(whole["nll"][k], np.concatenate([p["nll"][k] for p in parts])), k
    assert parts[1]["gates"]["peel"]["checked"] == 0 and parts[0]["gates"]["peel"]["checked"] > 0


def test_parts_merge_to_one_run(llama, ids, monkeypatch, tmp_path):
    """`merge` of the parts `eval --part I/N` writes gives the results.json one run over every window gives."""
    import json
    monkeypatch.setitem(methodology.CONTEXTS, ids.shape[1], len(ids))
    arms, wrec = run.arms_for(["int8-linears"]), {"sha256": "0" * 64}
    a = types.SimpleNamespace(model="llama-3.2-1b", set="wikitext2", ctx=ids.shape[1])

    def scored(w):  # HF's layers stand compared on the check window, so the gates pass without transformers
        r = evaluate(llama, ids[w], arms, windows=w)
        r["gates"]["hf_reference"] = {"checked": int(methodology.CHECK_WINDOW in w), "failed": 0, "first_failure": None}
        return r

    whole = scored(list(range(len(ids))))
    one = run.merge(a, arms, (0,), wrec, [whole], whole.pop("nll"), 0.0, 1)
    dirs = []
    for i, w in enumerate(engine.shards(len(ids), 2)):
        r, d = scored(w), tmp_path / f"part{i}"
        d.mkdir()
        np.savez(d / "part.npz", **{k.replace("/", "|"): v for k, v in r.pop("nll").items()})
        (d / "part.json").write_text(json.dumps(r))
        (d / "part-args.json").write_text(json.dumps({**vars(a), "spec": ["int8-linears"], "seeds": "0", "plugin": None,
                                                       "part": f"{i}/2", "windows": w, "wrec": wrec, "seconds": 0.0,
                                                       "run": f"r{i}"}))
        dirs.append(str(d))
    assert run.main(["merge", "--out", str(tmp_path / "merged"), *reversed(dirs)]) == 0
    got = json.loads((tmp_path / "merged" / "results.json").read_text())
    assert got["passed"] and got["env"]["run"] == "r0,r1"
    assert {k: v["seeds"] for k, v in got["arms"].items()} == {k: v["seeds"] for k, v in one["arms"].items()}
    with pytest.raises(SystemExit, match="every window once"):
        run.main(["merge", "--out", str(tmp_path / "half"), dirs[0]])


def _sabotage(monkeypatch, kind):
    reg = {k: dict(v) for k, v in spec.REGISTRY.items()}
    monkeypatch.setattr(spec, "REGISTRY", reg)
    if kind == "range":
        @spec.linear_step("overflow", phase="integer")
        def overflow(mm, nxt):
            return nxt(mm.replace(x=mm.x * 2))
        return spec.parse({"linear": [{"step": "int8"}, {"step": "overflow"}]}, "overflow")
    if kind == "peel":
        monkeypatch.setattr(steps, "fold_peel", lambda u, k: u @ k.reshape(*k.shape[:-1], -1, 32).sum(-2).transpose(-1, -2) + 1)
        return spec.load("adopted-attention")
    real = matmul.int_mm
    monkeypatch.setattr(matmul, "int_mm", lambda a, b: real(a, b) + (torch.arange(a.shape[0])[:, None] == 0).long())
    return spec.load("int8-linears")


@pytest.mark.parametrize("kind, gate", [("range", "int8_range"), ("peel", "peel"), ("product", "product_exact")])
def test_a_failed_gate_reports_nothing(kind, gate, monkeypatch, llama, ids, tmp_path):
    arm = _sabotage(monkeypatch, kind)
    r = evaluate(llama, ids[:4], [arm])
    assert r["gates"][gate]["failed"] > 0 and r["gates"][gate]["first_failure"]["where"]
    a = types.SimpleNamespace(model="llama-3.1-8b", set="c4", ctx=2048)
    nll = {k: np.zeros((16, 2047)) for k in r["nll"]}
    shard = {k: r[k] for k in ("gates", "divergence", "attn_rel_err")}
    res = run.merge(a, [arm], (0,), {"sha256": "x"}, [shard], nll, 1.0, 1)
    assert res["passed"] is False
    assert all("seeds" not in e for e in res["arms"].values())
    assert "GATES FAILED" in report.table(res) and gate in report.table(res)


def test_the_hf_reference_gate_must_run(llama, ids):
    """`run.py eval` fails a run whose check window's layers were never compared with HF's."""
    r = evaluate(llama, ids[:4], ["int8-linears"])
    a = types.SimpleNamespace(model="llama-3.1-8b", set="c4", ctx=2048)
    nll = {k: np.zeros((16, 2047)) for k in r["nll"]}
    res = run.merge(a, [spec.load("int8-linears")], (0,), {"sha256": "x"},
                    [{k: r[k] for k in ("gates", "divergence", "attn_rel_err")}], nll, 1.0, 1)
    assert res["passed"] is False and res["gates"]["hf_reference"]["failed"] == 1


def test_hf_parity(llama, qwen, ids):
    """The engine's bf16 layers are HF transformers' LlamaDecoderLayer / Qwen2DecoderLayer with SDPA, bit for bit."""
    pytest.importorskip("transformers")
    for snap in (llama, qwen):
        r = evaluate(snap, ids[:4], [], hf_check=True)
        g = r["gates"]["hf_reference"]
        assert g["checked"] == M.Config.read(snap).layers and g["failed"] == 0, g


def test_row_chunks_change_no_bit(llama, ids, calib, monkeypatch):
    """At 32K the s x s attention work and the loss run in chunks of query rows; the chunks change no bit, since every
    coin and scale is drawn per row or for the whole window, and the products are exact."""
    arms = ["adopted", "int8-all"]
    whole = evaluate(llama, ids[:4], arms, calib=calib)
    monkeypatch.setattr(M, "ROWS_WHOLE", 0)
    monkeypatch.setattr(M, "ROWS_BYTES", 16 * 2 * 64 * 8)
    monkeypatch.setattr(M, "LOSS_WHOLE", 16)
    monkeypatch.setattr(M, "LOSS_ROWS", 24)
    assert M.row_chunk(2, 64) == 16
    chunked = evaluate(llama, ids[:4], arms, calib=calib)
    assert passed(chunked), chunked["gates"]
    for k in whole["nll"]:
        assert np.array_equal(whole["nll"][k], chunked["nll"][k]), k


def test_the_readme_plugin_example_runs(llama, ids, monkeypatch, tmp_path):
    """README's "Adding an intervention" example, as written, is a step a spec names and the engine runs and gates."""
    monkeypatch.setattr(spec, "REGISTRY", {k: dict(v) for k, v in spec.REGISTRY.items()})
    code = run.README.read_text().split("## Adding an intervention")[1].split("~~~python\n")[1].split("~~~")[0]
    (tmp_path / "clip.py").write_text(code)
    run.load_plugins([str(tmp_path / "clip.py")])
    arm = spec.parse({"linear": [{"step": "clip", "sigmas": 1}, {"step": "int8"}]}, "clip")
    assert arm.linear[0].kw == {"sigmas": 1.0} and arm.scope == "linears"
    plain = spec.parse({"linear": [{"step": "int8"}]}, "unclipped")
    r = evaluate(llama, ids[:4], [arm, plain])
    assert passed(r) and r["gates"]["int8_range"]["checked"] > 0
    assert not np.array_equal(r["nll"]["clip/s0"], r["nll"]["unclipped/s0"])


def test_check_window_runs_the_costly_gates_only(llama, ids):
    r = evaluate(llama, ids[4:8], ["int8-attention"], windows=[4, 5, 6, 7])
    assert methodology.CHECK_WINDOW not in [4, 5, 6, 7]
    assert r["gates"]["product_exact"]["checked"] == 0 and r["gates"]["int8_range"]["checked"] > 0
