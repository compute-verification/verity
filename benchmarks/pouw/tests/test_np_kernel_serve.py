"""The served lead's host side off the device (benchmarks/pouw/np_kernel/serve_lead.py): its y, `reference` and `fold`,
against verity_int8's (`int8_exact`'s product and epilogue) bit for bit, and folds that leave a coin term in or fold
through fl(−1.5·2²³·s_j) or the weight's scale first against it; the served y end to end from a word model of the formed
A' and B' (layout.h's lane_a and lane_b under the lead's layout, the lane words over B' lanes 28-31 as ssn_k's cv puts
them), fed through the flat strided feed serve.cu copies under fresh coins and pooled ones: its final word read back
through the row's fold words and the column sources as the epilogue reads it (layout.h's fold_word), then `fold`, against
verity_int8's y; the pooled certifier's order, refusals and prefixes; and the report's steady steps and output
comparison."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.protocols.pouw.lead import feed as F  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "np_kernel"
spec = importlib.util.spec_from_file_location("np_kernel_serve_lead", HERE / "serve_lead.py")
SL = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SL)


def tag_of(i: int, e: int) -> int:
    """layout.h's tag_of, with C's truncating division and remainder."""
    z, d = (i // 2 + 1) * (-1 if i & 1 else 1), 0
    for _ in range(e + 1):
        d = (int(np.fmod(z, 13)) + 19) % 13 - 6
        z = int((z - d) / 13)
    return d


def code_of(j: int, e: int) -> int:
    return (j + 14281) // 13**e % 13 - 6


def kidx(t: int, lane: int, k: int) -> int:
    i = (lane if lane < 21 else -1) if t == 0 else (21 + 28 * (t - 1) + lane if lane < 28 else -1)
    return i if 0 <= i < k else -1


def signed(word: int, byte: int) -> int:
    return int(np.int8(np.uint8((word >> (8 * byte)) & 0xFF)))


def u32(a) -> np.ndarray:
    return (a.numpy() if hasattr(a, "numpy") else np.asarray(a)).view(np.uint32).astype(np.int64)


def final_word(xq: np.ndarray, w8: np.ndarray, feed, n: int) -> np.ndarray:
    """The GEMM's final word [m, n] from A' and B' formed lane by lane: Σ A'·B' − β_j (int64, unwrapped)."""
    m, k = xq.shape
    S = F.steps(k)
    rows, g = u32(feed.rows), u32(feed.g)
    lanes = u32(feed.lanes(n)).reshape(-1, S, n)
    p = np.zeros((m, n), dtype=np.int64)
    for i in range(m):
        bit = lambda q: int(rows[i, q >> 5] >> (q & 31)) & 1  # noqa: E731
        for j in range(n):
            acc = 0
            for t in range(S):
                for lane in range(32):
                    kk = kidx(t, lane, k)
                    a = int(xq[i, kk]) if kk >= 0 else 0
                    b = int(w8[j, kk]) if kk >= 0 else 0
                    if t == 0 and 24 <= lane < 28:
                        a, b = tag_of(i, lane - 24), signed(int(g[j]), lane - 24)
                    if t == 0 and (21 <= lane < 24 or lane == 31):
                        a, b = bit(3 if lane == 31 else lane - 21), code_of(j, 3 if lane == 31 else lane - 21)
                    if t > 0 and lane == 31:
                        a = 1 + bit(t + 3)
                    if 28 <= lane < 31:
                        a = int(F.ROT_W[i & 1, lane - 28])
                    if lane >= 28:
                        b = signed(int(lanes[i // 64, t, j]), lane - 28)
                    acc += a * b
            beta = (sum(code_of(j, e) for e in range(4)) + 3 * (S - 1) + 1) // 2
            p[i, j] = acc - beta
    return p


def row_terms(m: int, k: int, n: int, feed) -> tuple[np.ndarray, np.ndarray]:
    """The walk w_i = Σ_{t ≥ 1} (1 + b_t[i]) [m] and the cross terms Σ_e a1[i, e] C[e, j] + Σ_e T(i)[e] G[e, j] [m, n],
    int64, from the call's row and G words."""
    S, rows, g = F.steps(k), u32(feed.rows)[:m], u32(feed.g)[:n]
    a1 = np.stack([(rows[:, 0] >> e) & 1 for e in range(4)], 1)
    walk = (S - 1) + np.array([sum(int(r[q >> 5] >> (q & 31)) & 1 for q in range(4, S + 3)) for r in rows])
    G = np.array([[signed(int(g[j]), e) for j in range(n)] for e in range(4)])
    T = np.array([[tag_of(i, e) for e in range(4)] for i in range(m)])
    return walk, a1 @ F.codes(n) + T @ G


def coin_terms(m: int, k: int, n: int, feed) -> np.ndarray:
    """n_ij = −β_j + Σ_e a1[i, e] C[e, j] + Σ_e T(i)[e] G[e, j] + Σ_{t ≥ 1} (1 + b_t[i]), int64 [m, n]."""
    walk, cross = row_terms(m, k, n, feed)
    beta = (F.codes(n).sum(0) + 3 * (F.steps(k) - 1) + 1) // 2
    return -beta[None, :] + cross + walk[:, None]


def epilogue_read(word: np.ndarray, k: int, feed) -> np.ndarray:
    """What the epilogue reads back from final words ``word`` [m, n] (layout.h's fold_word), mod 2³²: one IADD3 of the
    row source 0x4B400000 − w_i (fold_row) and the column source the feed gives the row's tile and parity, then two IDP4A
    that take out a1·C and T·G."""
    m, n = word.shape
    walk, cross = row_terms(m, k, n, feed)
    src = feed.sources(n)
    src = (src.numpy() if hasattr(src, "numpy") else src).astype(np.int64).reshape(-1, 2, n)
    return (word + (SL.MAGIC - walk)[:, None] + src[np.arange(m) // 64, np.arange(m) & 1] - cross) % 2**32


def stock(ab, s_a, s_w):
    """verity_int8's y on the exact product ``ab``: `int8_exact.dequantize`, its epilogue."""
    return E.dequantize(torch.as_tensor(ab).to(torch.int32), torch.as_tensor(s_a).reshape(-1), torch.as_tensor(s_w))


def bits(y) -> np.ndarray:
    return y.view(torch.int16).numpy()


def call_of(m: int, k: int, n: int, seed: int):
    """A call as the served arm makes one: rows quantized by verity_int8 (heavy-tailed, so s_a spans decades) and a
    weight quantized per row, both scales with 24-bit significands."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((m, k), generator=gen) * (1 + 30 * (torch.rand((1, k), generator=gen) < 0.01))
    xq, s_a = E.quantize_rows(x.to(torch.bfloat16))
    w8, s_w = E.quantize_rows(torch.randn((n, k), generator=gen) * 0.02)
    return xq, s_a, w8, s_w


@pytest.mark.parametrize("m,k,n", [(1, 2048, 256), (17, 2048, 512), (64, 8192, 256), (100, 60, 48)])
def test_the_leads_y_is_verity_int8s(m, k, n):
    """The lead's served y is the plain arm's bit for bit, whatever the coins: `reference` against `int8_exact`'s own
    product and epilogue, on rows and weights quantized as the arm quantizes them."""
    xq, s_a, w8, s_w = call_of(m, k, n, m * 1000 + n)
    y, ab = SL.reference(xq, s_a, w8, s_w)
    assert np.array_equal(bits(y), bits(E.dequantize(E.int_mm(xq, w8), s_a, s_w)))
    assert torch.equal(ab, E.int_mm(xq, w8).to(torch.int64)) and int(ab.abs().max()) <= SL.FOLD_BOUND
    assert ((s_w.view(torch.int32) & 3) != 0).float().mean() > 0.5


def test_the_fold_is_stock_on_p_up_to_the_bound():
    rng = np.random.default_rng(3)
    p = np.concatenate([[0, 1, -1, SL.FOLD_BOUND, -SL.FOLD_BOUND, SL.FOLD_BOUND - 1, 1 - SL.FOLD_BOUND],
                        rng.integers(-SL.FOLD_BOUND, SL.FOLD_BOUND + 1, 249)]).reshape(16, 16)
    s_a = (rng.random(16) * 0.1 + 1e-6).astype(np.float32)
    s_w = (rng.random(16) * 4e-3 + 1e-7).astype(np.float32)
    got = SL.fold(torch.from_numpy(p + SL.MAGIC), torch.from_numpy(s_a), torch.from_numpy(s_w))
    assert np.array_equal(bits(got), bits(stock(p, s_a, s_w)))


#: (p, s_a's bits, s_w's bits) where the scales' order alone moves the bf16 output
ORDER = [(-162281, 1024863733, 983206857), (77507, 1023883001, 969546748), (163583, 1027132447, 980511694),
         (-132060, 1026526746, 984294742)]


def test_a_fold_that_leaves_a_coin_term_in_or_folds_otherwise_isnt_verity_int8s():
    """A fold that departs from `fold` isn't verity_int8's y, each way: the word less σ alone (the coin terms n_ij left
    in), the weight's scale folded through m_j = fl(−1.5·2²³·s_j) by an FMA, and the weight's scale first."""
    m, k, n = 64, 2048, 256
    xq, s_a, w8, s_w = call_of(m, k, n, 11)
    link = F.InProcessCertifier(np.random.default_rng(11).bytes)
    link.open(0, bytes(64), bytes(64), 1)
    feed = link.release(F.Call(0, 0, 0, m, k, n), bytes(64))
    ab = E.int_mm(xq, w8).to(torch.int64)
    want = bits(stock(ab, s_a, s_w))
    noisy = SL.fold(ab + torch.from_numpy(coin_terms(m, k, n, feed)) + SL.MAGIC, s_a, s_w)
    assert (bits(noisy) != want).mean() > 0.05
    sj, si = s_w.reshape(1, -1).to(torch.float64), s_a.reshape(-1, 1)
    v = (ab + SL.MAGIC).to(torch.int32).view(torch.float32).to(torch.float64)
    ffma = ((v * sj + (s_w.reshape(1, -1) * -12582912.0).to(torch.float64)).to(torch.float32) * si).to(torch.bfloat16)
    assert (bits(ffma) != want).sum() > 0
    p = torch.tensor([[q] for q, _, _ in ORDER], dtype=torch.int64)
    sa = torch.tensor([a for _, a, _ in ORDER], dtype=torch.int32).view(torch.float32)
    sw = torch.tensor([w for _, _, w in ORDER], dtype=torch.int32).view(torch.float32)
    for i in range(len(ORDER)):
        one = (p[i:i + 1], sa[i:i + 1], sw[i:i + 1])
        weight_first = ((one[0].to(torch.float32) * one[2]) * one[1]).to(torch.bfloat16)
        assert bits(SL.fold(one[0] + SL.MAGIC, one[1], one[2])).item() == bits(stock(*one)).item()
        assert bits(weight_first).item() != bits(stock(*one)).item()


@pytest.mark.parametrize("coins", ["fresh", "pooled"])
@pytest.mark.parametrize("m,k", [(5, 60), (70, 200)])
def test_the_fold_of_the_formed_word_is_verity_int8s_y(coins, m, k):
    """The served y from the formed operands, end to end: the final word (A·B + σ + n_ij, σ the lane coins' sum and n_ij
    `coin_terms`, both nonzero) reads back through the row's fold words and the feed's column source β_j − σ as
    0x4B400000 + A·B, and `fold` on it is verity_int8's y bit for bit, on rows and a weight quantized as the arm
    quantizes them."""
    n, rng = 16, np.random.default_rng(m * 1000 + k)
    xq, s_a, w8, s_w = call_of(m, k, n, m * 1000 + k)
    link = F.InProcessCertifier(rng.bytes) if coins == "fresh" else SL.PooledCertifier(128, rng.bytes, pin=False)
    link.open(0, bytes(64), bytes(64), 1)
    feed = link.release(F.Call(0, 0, 0, m, k, n), bytes(64))
    ab = E.int_mm(xq, w8).to(torch.int64).numpy()
    word = final_word(xq.numpy(), w8.numpy(), feed, n)
    terms = coin_terms(m, k, n, feed)
    assert ((word - ab - terms) != 0).any() and (terms != 0).any()
    read = epilogue_read(word, k, feed)
    assert np.array_equal(read, (ab + SL.MAGIC) % 2**32)
    y = SL.fold(torch.from_numpy(read), s_a, s_w)
    assert np.array_equal(bits(y), bits(E.dequantize(E.int_mm(xq, w8), s_a, s_w)))


def test_a_pool_holds_each_tiles_coins_as_a_64_row_calls_feed():
    k, n, seed = 200, 16, 7
    pool = SL.Pool.sample(k, n, 100, np.random.default_rng(seed).bytes, pin=False)
    rand, tile = np.random.default_rng(seed).bytes, F.Shape(SL.ROW_TILE, k, n)
    for r in range(2):
        f = F.Feed.of(F.Call(0, r, 0, SL.ROW_TILE, k, n), F.CallCoins.sample(tile, rand))
        assert np.array_equal(pool.rows[64 * r:64 * r + 64].numpy().view(np.uint32), f.rows)
        assert np.array_equal(pool.lanes[r].numpy().view(np.uint32), f.lanes_c[0])
        assert np.array_equal(pool.sources[r].numpy(), (F.betas(tile.C, tile.S) - f.sigma[0]).astype(np.int32))
        if r == 0:
            assert np.array_equal(pool.g.numpy().view(np.uint32), f.g)
    assert tuple(pool.rows.shape) == (128, tile.H) and pool.bytes() == 4 * (128 * tile.H + tile.C + 2 * tile.S * tile.C
                                                                              + 2 * 2 * tile.C)


def test_the_pooled_certifier_keeps_the_certifiers_order_and_refusals():
    link = SL.PooledCertifier(64, np.random.default_rng(1).bytes, pin=False)
    call = F.Call(0, 0, 0, 17, 200, 16)
    with pytest.raises(F.FeedError, match="before its window"):
        link.release(call, bytes(64))
    link.open(0, bytes(64), bytes(64), 2)
    feed = link.release(call, bytes(64))
    assert isinstance(feed, SL.PooledFeed) and tuple(feed.rows.shape) == (32, call.shape.H)
    assert feed.lanes(16).numel() == call.shape.T * call.shape.S * 16 and feed.sources(16).numel() == 2 * 16
    with pytest.raises(F.FeedError, match="stride 32"):
        feed.lanes(32)
    with pytest.raises(F.FeedError, match="released already"):
        link.release(call, bytes(64))
    with pytest.raises(F.FeedError, match="past the pool's 64"):
        link.release(F.Call(0, 1, 0, 65, 200, 16), bytes(64))
    again = SL.PooledCertifier(64, np.random.default_rng(2).bytes, pin=False)
    again.open(0, bytes(64), bytes(64), 2)
    first, second = (again.release(F.Call(0, i, 0, m, 200, 16), bytes(64)) for i, m in enumerate((64, 3)))
    assert first.pool is second.pool and torch.equal(second.rows, first.rows[:16])


def write_arm(d: Path, steps: list[tuple[int, float]], start: float, tokens: dict[str, list[int]], lead=None) -> Path:
    d.mkdir(parents=True)
    recs = [{"kind": "header"}] + [{"kind": "step", "step": s, "T": T, "t": t} for s, (T, t) in enumerate(steps)]
    (d / "advice_trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "main.timings.json").write_text(json.dumps({"start_unix": start}))
    (d / "main.jsonl").write_text("".join(json.dumps({"user": u, "response": {"token_ids": ids}}) + "\n"
                                          for u, ids in tokens.items()))
    if lead is not None:
        (d / "lead.json").write_text(json.dumps(lead))
    return d


def test_steady_is_the_longest_run_of_full_steps_from_the_loads_start():
    st = [{"step": s, "T": T, "t": t} for s, (T, t) in
          enumerate([(8192, 0.0), (8192, 1.0), (8192, 2.0), (100, 3.0), (8192, 4.0), (8192, 5.0), (8192, 6.0), (8192, 7.0)])]
    assert SL.steady(st, 8192, 0.0) == [4, 5, 6, 7]
    assert SL.steady(st, 8192, 4.5) == [6, 7]


def test_the_report_compares_step_walls_and_tokens_with_the_base(tmp_path, capsys):
    audit = {k: 1 for k in ("outputs", "differ_from_reference", "differ_from_stock", "sign_differs_from_stock",
                             "ab_past_bound")} | {"max_steps_from_stock": 3}
    stock = write_arm(tmp_path / "stock", [(8192, 10.0 + 0.03 * s) for s in range(5)], 9.0, {"a": [1], "b": [2]})
    lead = write_arm(tmp_path / "lead2", [(8192, 10.0 + 0.5 * s) for s in range(5)], 9.0, {"a": [1], "b": [3]},
                     lead={"mode": 2, "audits": [audit, audit]})
    out = tmp_path / "report.json"
    assert SL.main(["report", "--arm", f"stock={stock}", "--arm", f"lead2={lead}", "--T", "8192", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["base"] == "stock" and doc["arms"]["stock"]["step_ms"]["median"] == pytest.approx(30.0)
    got = doc["arms"]["lead2"]
    assert got["step_ms"]["median"] == pytest.approx(500.0) and got["full_steps"] == 4
    assert (got["tokens_equal_to_base"], got["requests_differing"]) == (1, ["b"])
    assert got["lead"]["audited_calls"] == 2 and got["lead"]["audit_differ_from_reference"] == 2
    assert got["lead"]["audit_max_steps_from_stock"] == 3 and "audits" not in got["lead"]
    assert "ARM lead2: 4 full steps, step median 500.0 ms" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["serve", "--", "--port", "1"], ["serve", "--stock", "--lib", "x.so"]])
def test_serve_takes_stock_or_a_library(argv):
    with pytest.raises(SystemExit) as ex:
        SL.main(argv)
    assert ex.value.code == 2
