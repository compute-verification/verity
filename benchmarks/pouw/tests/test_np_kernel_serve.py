"""The served lead's host side off the device (benchmarks/pouw/np_kernel/serve_lead.py): `reference` against a word model of
the formed A' and B' (layout.h's lane_a and lane_b under the lead's layout, the lane words over B' lanes 28-31 as ssn_k's
cv puts them, the column sources at each row's parity) fed through the flat strided feed serve.cu copies, under fresh
coins and pooled ones; the pooled certifier's order, refusals and prefixes; the tags; and the report's steady steps and
output comparison."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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


def kernel_model(xq: np.ndarray, w8: np.ndarray, s_a: np.ndarray, s_w: np.ndarray, feed, n: int):
    """The served p and y [m, n] from A' and B' formed lane by lane, the GEMM's Σ A'·B' − β_j, then the column source of
    the row's tile and parity added mod 2³² and the fold (fma exact in float64, one rounding to float32)."""
    m, k = xq.shape
    S = F.steps(k)
    rows, g = u32(feed.rows), u32(feed.g)
    lanes, src = u32(feed.lanes(n)).reshape(-1, S, n), u32(feed.sources(n)).reshape(-1, 2, n)
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
    word = p + src[np.arange(m) // 64, np.arange(m) & 1]
    v = (word & 0xFFFFFFFF).astype(np.uint32).view(np.float32).astype(np.float64)
    sj = s_w.astype(np.float32)
    f = (v * sj.astype(np.float64) + (sj * np.float32(-12582912.0)).astype(np.float64)).astype(np.float32)
    y = torch.from_numpy(f * s_a.astype(np.float32)[:, None]).to(torch.bfloat16)
    return word - F.MAGIC, y


@pytest.mark.parametrize("coins", ["fresh", "pooled"])
@pytest.mark.parametrize("m,k", [(5, 60), (70, 200)])
def test_reference_is_the_formed_operands_word_model(coins, m, k):
    n, rng = 16, np.random.default_rng(m * 1000 + k)
    xq = rng.integers(-127, 128, (m, k)).astype(np.int8)
    w8 = rng.integers(-127, 128, (n, k)).astype(np.int8)
    s_a = (rng.random(m) * 1e-2 + 1e-4).astype(np.float32)
    s_w = (rng.random(n) * 2e-3 + 1e-5).astype(np.float32)
    link = F.InProcessCertifier(rng.bytes) if coins == "fresh" else SL.PooledCertifier(128, rng.bytes, pin=False)
    link.open(0, bytes(64), bytes(64), 1)
    feed = link.release(F.Call(0, 0, 0, m, k, n), bytes(64))
    p_model, y_model = kernel_model(xq, w8, s_a, s_w, feed, n)
    y, p, ab = SL.reference(torch.from_numpy(xq), torch.from_numpy(s_a), torch.from_numpy(w8), torch.from_numpy(s_w),
                            feed.rows, feed.g)
    assert np.array_equal(p.numpy(), p_model)
    assert torch.equal(y.view(torch.int16), y_model.view(torch.int16))
    assert np.array_equal(ab.numpy(), xq.astype(np.int64) @ w8.astype(np.int64).T)
    assert (p != ab).any()


def test_a_pool_holds_each_tiles_coins_as_a_64_row_calls_feed():
    k, n, seed = 200, 16, 7
    pool = SL.Pool.sample(k, n, 100, np.random.default_rng(seed).bytes, pin=False)
    rand, tile = np.random.default_rng(seed).bytes, F.Shape(SL.ROW_TILE, k, n)
    for r in range(2):
        f = F.Feed.of(F.Call(0, r, 0, SL.ROW_TILE, k, n), F.CallCoins.sample(tile, rand))
        assert np.array_equal(pool.rows[64 * r:64 * r + 64].numpy().view(np.uint32), f.rows)
        assert np.array_equal(pool.lanes[r].numpy().view(np.uint32), f.lanes_c[0])
        assert np.array_equal(pool.sources[r].numpy(), (F.MAGIC - f.sigma[0]).astype(np.int32))
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


def test_tags_are_layouts():
    got = SL.tags(400)
    assert got.tolist() == [[tag_of(i, e) for e in range(4)] for i in range(400)]
    z = (np.arange(400) // 2 + 1) * np.where(np.arange(400) & 1, -1, 1)
    assert np.array_equal(got @ (13 ** np.arange(4)), z) and np.abs(got).max() <= 6


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
                             "p_past_bound", "ab_past_bound")} | {"max_steps_from_stock": 3}
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
