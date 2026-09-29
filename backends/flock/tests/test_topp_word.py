"""The top-p keep word's circuit (``verity_flock.topp_word``) against ``sampling.topp_keep``, bit for bit."""
import numpy as np
import pytest
from verity_flock import topp_word as TW

f32 = np.float32
SPECIALS = [0, 0x80000000, 1, 0x807FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000,
            0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FFFFFFF, 0x2EDBE6FF, 0x0DA24260, 0x3F800001, 0xC2FC0000, 0xC2FE0000, 0xC3000000]


def _words(rng, n):
    w = rng.integers(0, 2 ** 32, n, dtype=np.uint64)
    w[: n // 4] = rng.choice(SPECIALS, n // 4)
    w[n // 4: n // 2] = (rng.standard_normal(n // 2 - n // 4).astype(f32) * f32(4)).view(np.uint32)
    return w


def _piece(name, *ops):
    E = TW.Engine(1)
    args = [E.free(len(o), TW.WIDTH[k], np.asarray(o, np.uint64)[None, :]) for o, k in zip(ops, TW.OPS[name][1])]
    return E.op(name, *args).d[0]


def test_pieces_match_the_pipelines_operations():
    from verity_vllm.program.registry import sampling as SM
    from verity_vllm.program.registry import topp_split as TS

    rng = np.random.default_rng(7)
    n = 3000
    a, b = _words(rng, n), _words(rng, n)
    af, bf = a.astype(np.uint32).view(f32), b.astype(np.uint32).view(f32)
    cfg = TS.Cfg()

    def same(got, want):
        """Equal words, any NaN equal to any NaN: only compares read the pipeline's values, and they see no payload."""
        want = np.asarray(want).astype(np.uint32).astype(np.uint64)
        nan = np.isnan(want.astype(np.uint32).view(f32)) & np.isnan(got.astype(np.uint32).view(f32))
        assert ((got == want) | nan).all()

    same(_piece("exp", a), TS.tl_exp(af, cfg).view(np.uint32))
    r = _piece("rcp", b)
    same(_piece("div", a, b, r), TS.tl_div(af, bf, cfg).view(np.uint32))
    same(_piece("log", a), np.array([SM.nv_logf(int(v)) for v in a], np.uint64))
    with np.errstate(invalid="ignore", over="ignore"):
        assert (_piece("near", a, b) == (np.abs((af - bf).astype(f32)) < f32(1e-9))).all()
        assert (_piece("ge", a, b) == (af >= bf)).all()
        same(_piece("sub", a, b), (af - bf).view(np.uint32))
        same(_piece("fma", a, b, a[::-1].copy()), TS.fma32(af, bf, af[::-1].copy()).view(np.uint32))
    py_max = np.array([f32(max(x, y)) for x, y in zip(af, bf)], f32).view(np.uint32)
    same(_piece("max", a, b), py_max)
    counts = rng.integers(0, 1 << 18, n, dtype=np.uint64)
    q = np.concatenate([rng.integers(0, 2 ** 32, n // 2, dtype=np.uint64),
                        (rng.random(n - n // 2) * 3e5).astype(f32).view(np.uint32).astype(np.uint64)])
    qf = q.astype(np.uint32).view(f32)
    with np.errstate(invalid="ignore"):
        qi = np.where(~np.isfinite(qf) | (qf <= 0), 0, np.minimum(np.trunc(np.nan_to_num(qf, posinf=0)), 2 ** 32 - 1))
    want = np.minimum(np.maximum(counts.astype(np.float64) - qi, 1), counts)
    assert (_piece("numkeep", counts, q) == want.astype(np.uint64)).all()


def _rows(V, T, rng):
    xs, ps, ss = [], [], []
    for i in range(T):
        x = (rng.standard_normal(V) * rng.choice([0.5, 3.0, 10.0])).astype(f32)
        x = (x.view(np.uint32) & 0xFFFF0000).view(f32)
        k = i % 7
        if k == 1:
            x = (np.round(x * 2) / 2).astype(f32)
        elif k == 2:
            x[rng.choice(V, 3)] = np.nan
        elif k == 3:
            x[rng.random(V) < 0.5] = -np.inf
        elif k == 4:
            x[:] = -np.inf
            x[rng.choice(V, 4)] = rng.standard_normal(4).astype(f32)
        elif k == 5:
            x = rng.integers(0, 2 ** 32, V, dtype=np.uint64).astype(np.uint32).view(f32)
        xs.append(x)
        ps.append(rng.choice([0.9, 0.5, 0.95, 0.3, 0.999999, 1e-8]) if i % 9 else rng.choice([1.0, np.nan, 0.0]))
        ss.append(TW.SPLIT_COUNTS[i % 6] if i % 11 else rng.choice([0, 3, 64, 0xFFFFFFFF]))
    return np.array(xs), np.array(ps, f32), np.array(ss, np.uint64)


@pytest.mark.parametrize("V", [160, 200])
@pytest.mark.slow
def test_the_keep_word_is_topp_keep(V):
    from verity_vllm.program.registry import sampling as SM

    rng = np.random.default_rng(V)
    x, p, s = _rows(V, 33, rng)
    keep, _E = TW.evaluate(x.view(np.uint32), p.view(np.uint32), s)
    for i in range(len(x)):
        assert (keep[i] == SM.topp_keep(x[i], float(p[i]), int(s[i]))).all(), i


@pytest.mark.slow
def test_the_count_is_the_evaluated_circuit():
    rng = np.random.default_rng(1)
    x, p, s = _rows(160, 2, rng)
    _keep, E = TW.evaluate(x.view(np.uint32), p.view(np.uint32), s)
    assert E.counts == TW.count(160)
    one = TW.count(160, splits=8)
    assert all(k[0][0] != "pipeline S = 16" for k in one) and any(k[0][0] == "pipeline S = 8" for k in one)
    assert not TW.count(160, splits=3) or all(k[0][0] in ("p >= 1", "split count") for k in TW.count(160, splits=3))


@pytest.mark.slow
def test_each_lane_follows_its_own_half_where_the_halves_disagree(monkeypatch):
    """``topp_split``'s own divergent rows (verity PR #169, ``test_topp_split_halves``): ordinary logits, the step kernels'
    partials constructed so that each half's own scalars decide ``nomask``, in both orders. The circuit, given the same partials
    for its step kernels, is ``sampling.topp_keep`` bit for bit."""
    from verity_vllm.program.registry import sampling as SM
    from verity_vllm.program.registry import topp_split as TS

    V, S, P = 3000, 16, f32(0.5)
    rows = {35: [False, True], 50: [False, True], 17: [True, False], 47: [True, False]}

    def narrowing():
        parts = np.zeros((TW.R, S, TW.F, 3), f32)
        parts[..., 0], parts[..., 1], parts[..., 2] = f32(1.0 / S), f32(1e-12), f32(1.0)
        return parts

    def step(x, stats, parts, p, ROUND, S_, F_, R_, BLOCK, cfg):
        parts[...] = narrowing()

    def parts_of(E, r, S_):
        n = TW.F * S_
        return [E.const([int(f32(1.0 / S_).view(np.uint32))] * n), E.const([int(f32(1e-12).view(np.uint32))] * n), E.const([1] * n, TW.CW)]

    x = np.array([(-np.random.default_rng(seed).random(V) * f32(2.5e-5)).astype(f32) for seed in rows])
    monkeypatch.setattr(TS, "step_kernel", step)
    keep, _E = TW.evaluate(x.view(np.uint32), np.full(len(x), P).view(np.uint32), np.full(len(x), S, np.uint64), parts_of)
    for i, stopped in enumerate(rows.values()):
        assert TS.topp_split_row(x[i], float(P), S)[3]["stopped"] == stopped
        ref = SM.topp_keep(x[i], float(P), S)
        assert ref.any() and not ref.all()
        assert (keep[i] == ref).all(), i
