import math
import types

import numpy as np
import pytest
import torch

import engine
import gates as G
import matmul
import spec
import steps


def fake_mm(rows: range, s: int, d: int = 128):
    g = G.Gates()
    pos = torch.tensor(list(rows))
    causal = torch.arange(s)[None, :] <= pos[:, None]
    mm = types.SimpleNamespace(extra={"causal": causal, "d": d}, where="t", gates=g)
    mm.gate = lambda name, ok, detail=None: g.record(name, ok, "t", detail)
    return mm


def fattn_ref(S: np.ndarray, pos: int, A: int) -> tuple:
    """fattn_ref.py's check, as written there: e, L, s_q and z_q of one query column."""
    lut = np.rint(127.0 * np.exp2(-np.arange(256) / 16.0)).astype(np.int64)
    m = int(S[: pos + 1].max())
    dd = (m - S[: pos + 1]).astype(np.uint64)
    t = np.minimum((dd * np.uint64(A)) >> np.uint64(32), 255).astype(np.int64)
    e = lut[t]
    L = int(e.sum())
    f32 = np.float32
    return e, L, f32(1) / f32(L), f32(e[0]) / f32(L)


def test_the_table_is_fattn_cus():
    lut = steps.lut_table(16, 256)
    assert int(lut[0]) == 127 and int(lut[255]) == 0 and bool((lut[:-1] >= lut[1:]).all())
    assert lut.tolist() == [round(127 * 2 ** (-t / 16)) for t in range(256)]
    assert int(steps.lut_table(32, 512)[-1]) == 0 and int(steps.lut_table(32, 256)[-1]) == 1
    assert steps.lut_up(16).tolist()[:2] == [127, 133]


def test_lut_softmax_is_fattn_refs():
    """The kernel's form (key 0 in the max, one K scale) gives fattn_ref.py's e, L, s_q and z_q bit for bit."""
    g = torch.Generator().manual_seed(0)
    rep, s, r0 = 2, 48, 16
    rows = range(r0, s)
    S = torch.randint(-20_000, 20_000, (rep, len(rows), s), generator=g).double()
    S[0, :, 0] += 40_000  # a sink that leads the max on head 0
    sq = torch.rand(rep, len(rows), 1, generator=g) * 0.05 + 0.01
    sk = torch.full((s, 1), 0.02)
    mm = fake_mm(rows, s)
    out = steps.lut_softmax(mm, S, sq, sk, res=16, size=256, sink=False)
    assert isinstance(out, matmul.Softmaxed) and mm.gates.passed and mm.gates.log["lut"]["checked"] == 1
    e, s_q, z_q = out.extra["lut"]
    c = math.log2(math.e) * 16 / math.sqrt(128) * 2.0 ** 32
    for h in range(rep):
        for i, pos in enumerate(rows):
            A = int(np.rint(np.float64(sq[h, i, 0]) * np.float64(sk[0, 0]) * c))
            want_e, L, sp, sz = fattn_ref(S[h, i].long().numpy(), pos, A)
            assert e[h, i, : pos + 1].tolist() == want_e.tolist() and int(e[h, i, pos + 1:].abs().sum()) == 0
            assert int(e[h, i].sum()) == L
            assert s_q[h, i, 0].numpy().tobytes() == sp.tobytes() and z_q[h, i, 0].numpy().tobytes() == sz.tobytes()
    assert torch.equal(out.p[..., 0:1], z_q) and torch.allclose(out.p.sum(-1), torch.ones(rep, len(rows)), atol=1e-5)


def test_the_sink_is_its_own_block():
    """`sink`: the max over keys 1..pos, so a dominant key 0 leaves the rest their 7 bits; key 0's weight above the
    max is H[u mod res] 2^(u div res)."""
    s = 4
    S = torch.tensor([[[0.0, 0, 0, 0], [5000, 1000, 0, 0], [5000, 1000, 900, 0], [5000, 1000, 900, 1000]]]).double()
    sq, sk = torch.full((1, 4, 1), 1e-3), torch.full((s, 1), 1.0)
    A = round(float(np.float32(1e-3)) * math.log2(math.e) * 16 / math.sqrt(128) * 2 ** 32)
    shared = steps.lut_softmax(fake_mm(range(4), s), S, sq, sk, res=16, size=256, sink=False).extra["lut"][0]
    mm = fake_mm(range(4), s)
    own = steps.lut_softmax(mm, S, sq, sk, res=16, size=256, sink=True).extra["lut"][0]
    assert mm.gates.passed
    assert shared[0, 1].tolist() == [127, int(steps.lut_table(16, 256)[(4000 * A) >> 32]), 0, 0]
    u = -((-4000 * A) >> 32)
    assert own[0, 1].tolist() == [int(steps.lut_up(16)[u % 16]) << (u // 16), 127, 0, 0]
    assert own[0, 0].tolist() == [127, 0, 0, 0] and own[0, 3, 1] == own[0, 3, 3] == 127
    assert own[0, 3, 2] == steps.lut_table(16, 256)[(100 * A) >> 32]


def test_round_takes_t_to_nearest():
    """`rnd`: t = floor((m - lam + 2^31) / 2^32), on every key, the sink's negative t included."""
    s = 3
    S = torch.tensor([[[0.0, 0, 0], [0, 0, 0], [3000, 1000, 657]]]).double()
    sq, sk = torch.full((1, 3, 1), 1e-3), torch.full((s, 1), 1.0)
    A = round(float(np.float32(1e-3)) * math.log2(math.e) * 16 / math.sqrt(128) * 2 ** 32)
    lut, up = steps.lut_table(16, 256), steps.lut_up(16)
    mm = fake_mm(range(3), s)
    e = steps.lut_softmax(mm, S, sq, sk, res=16, size=256, sink=True, rnd=True).extra["lut"][0]
    assert mm.gates.passed
    t2, t0 = (343 * A + 2 ** 31) >> 32, (-2000 * A + 2 ** 31) >> 32
    assert t2 != (343 * A) >> 32 and t0 != (-2000 * A) >> 32
    assert e[0, 2].tolist() == [int(up[(-t0) % 16]) << ((-t0) // 16), 127, int(lut[t2])]


def test_key_scales():
    g = torch.Generator().manual_seed(1)
    k = torch.randn(16, 8, generator=g)
    k[0] *= 50
    q, sk = steps.key_quant(k, "round", 127, 1, "head", False, None)
    assert torch.allclose(sk, torch.full((16, 1), float(k.abs().max()) / 127)) and q.abs().max() == 127
    q, sk = steps.key_quant(k, "round", 127, 1, "head", True, None)
    assert torch.isclose(sk[0, 0], k[0].abs().max() / 127) and torch.allclose(sk[1:], k[1:].abs().max() / 127)
    qd, sd = steps.key_quant(k, "dither", 111, 1, "head", True, None)
    assert qd.min() >= -112 and qd.max() <= 110 and torch.equal(qd[1:] + 0, torch.floor(k[1:] / sd[1:]))
    hi = torch.tensor(1.0)
    qc, sc = steps.key_quant(k, "round", 127, 1, "calib", False, (hi, hi))
    assert torch.allclose(sc, torch.full((16, 1), 1 / 127)) and qc.abs().max() == 127
    assert torch.equal(steps.key_quant(k, "round", 127, 1, "token", False, None)[0], steps.quant(k, 127, -1)[0])


@pytest.mark.parametrize("text, message", [
    ('softmax = "lut"', "k_scale = head or calib"),
    ('k_scale = "head"\nsoftmax = "lut"\np = "dither"', "p = round"),
    ('k_scale = "head"\nsoftmax = "lut"\nlut_res = 32', "raise lut_size"),
    ('k_sink = true', "k_scale = head or calib"),
    ('k_scale = "head"\nlut_sink = true', "softmax = lut"),
    ('k_scale = "head"\nlut_round = true', "softmax = lut"),
    ('k_scale = "row"', "k_scale is one of"),
])
def test_rejections(text, message):
    import tomllib
    with pytest.raises(ValueError, match=message):
        spec.parse(tomllib.loads('[[attention]]\nstep = "int8"\n' + text), "t")


def test_lut_arms_pass_every_gate(llama, ids, calib):
    arms = ["vllm-int8", "vllm-int8-lut", "vllm-int8-lut-sink-cal", "co32c-u16-cohalf-lut32-sink"]
    r = engine.evaluate(llama, ids[:4], list(range(4)), [spec.load(a) for a in arms], (0,), calib, device="cpu",
                        log=lambda s: None, hf_check=False)
    assert all(g["failed"] == 0 for g in r["gates"].values()), r["gates"]
    cfg = __import__("model").Config.read(llama)
    # one lut gate per (LUT arm, layer, KV group, window); H1's P peel on its two blocks under the LUT
    assert r["gates"]["lut"]["checked"] == 3 * cfg.layers * cfg.kv_heads * 4
    assert r["gates"]["peel"]["checked"] == 2 * cfg.layers * cfg.kv_heads
    bf = np.exp(r["nll"]["bf16/s0"].mean())
    for a in arms:
        assert abs(np.exp(r["nll"][f"{a}/s0"].mean()) / bf - 1) < 0.05, a
    assert not np.array_equal(r["nll"]["vllm-int8/s0"], r["nll"]["vllm-int8-lut/s0"])
