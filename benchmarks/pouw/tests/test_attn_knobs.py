"""gemm_rate/attn_knobs.py's CPU side: the paged caches it lays the step's keys into (vLLM's interleaved FlashAttention
cache and attn_rate.py's separate one), the byte comparison and ulp distance its checks report, the table's verdicts, and
fa3_tile_build.py's rewrite of the pinned hdim-64 tile."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemm_rate"))
import attn_knobs as ak  # noqa: E402  (gemm_rate/ is a directory of scripts, put on sys.path above)
import fa3_tile_build as tb  # noqa: E402


def _kv(t: int, hkv: int = 2, d: int = 4):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(t, hkv, d, generator=g).to(torch.bfloat16), torch.randn(t, hkv, d, generator=g).to(torch.bfloat16))


@pytest.mark.parametrize("interleaved", [True, False])
def test_each_sequence_s_rows_sit_at_its_block_table_s_slots(interleaved):
    lens, block = [5, 3, 9], 4
    k, v = _kv(sum(lens))
    kc, vc, bt = ak.paged(k, v, lens, block, interleaved)
    assert bt.dtype == torch.int32 and bt.shape == (3, 3)
    o = 0
    for i, n in enumerate(lens):
        for j in range(n):
            b = int(bt[i, j // block])
            assert torch.equal(kc[b, j % block], k[o + j]) and torch.equal(vc[b, j % block], v[o + j])
        o += n
    assert len({int(x) for x in bt[:, 0]}) == 3


def test_the_interleaved_cache_is_vllm_s_flash_attention_layout():
    k, v = _kv(6)
    kc, vc, _ = ak.paged(k, v, [6], 4, True)
    base = kc._base if kc._base is not None else kc
    assert base.shape == (2, 2, 4, 8)
    assert torch.equal(base.transpose(1, 2)[..., :4], kc) and torch.equal(base.transpose(1, 2)[..., 4:], vc)


def _bf(xs):
    return torch.tensor(xs, dtype=torch.float32).to(torch.bfloat16)


def test_ulps_count_bf16_steps_across_zero():
    one = _bf([1.0])
    nxt = (one.view(torch.int16) + 1).view(torch.bfloat16)
    assert ak.ulps(one, nxt) == 1
    assert ak.ulps(_bf([0.0]), _bf([-0.0])) == 0
    tiny = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)
    assert ak.ulps(tiny, -tiny) == 2


def test_compare_reports_equal_bytes_or_the_distance():
    o, lse = _bf([[1.0, 2.0]]), torch.tensor([[0.5]])
    assert ak.compare((o, lse), (o.clone(), lse.clone())) == {"eq": True}
    o2 = o.clone()
    o2[0, 1] = (o2[0, 1:].view(torch.int16) + 3).view(torch.bfloat16)
    got = ak.compare((o2, lse), (o, lse))
    assert got["eq"] is False and got["out_eq"] is False and got["max_ulps"] == 3 and got["lse_max_abs"] == 0
    assert got["max_abs"] == pytest.approx(3 * 2.0 ** -6)
    assert ak.compare((o, lse + 1e-7), (o, lse))["out_eq"] is True


def _w(prompt, arm, of_peak, r=0):
    return {"kind": "window", "sms": 132, "prompt": prompt, "arm": arm, "round": r, "of_peak": of_peak,
            "theta_1830": of_peak * 0.98, "sm_mhz": 1800.0, "us_call": 120.0}


def test_the_table_gives_each_arm_s_median_and_its_check():
    eq = {"eq": True}
    rows = [_w("8x1024", "served", x, r) for r, x in enumerate((0.28, 0.30, 0.29))]
    rows += [_w("8x1024", "served-eager", 0.27), _w("8x1024", "so:m128", 0.31)]
    rows += [{"kind": "check", "prompt": "8x1024", "arm": "served", "vs_served": eq, "repeat": eq, "batch": eq},
             {"kind": "check", "prompt": "8x1024", "arm": "so:m128", "vs_served": {"eq": False, "max_ulps": 2, "max_abs": 0.01},
              "repeat": eq, "batch": eq}]
    text = ak.table(rows)
    assert "| 8x1024 | served | 0.290 | 0.284 | 1800 | 120.0 | bytes equal | 3 |" in text
    assert "| 8x1024 | served-eager | 0.270 | 0.265 | 1800 | 120.0 | bytes equal | 1 |" in text
    assert "| 8x1024 | so:m128 | 0.310 | 0.304 | 1800 | 120.0 | differs: 2 ulp, 0.01 abs | 1 |" in text


def test_retile_rewrites_only_the_causal_hdim64_tile(tmp_path):
    h = tmp_path / "tile_size.h"
    h.write_text(f"if (headdim <= 64) {{\n    {tb.PINNED_HD64}\n}}\n")
    tb.retile(h, 128)
    assert "return {is_causal ? 128 : 192, is_causal ? 128 : is_local || paged_kv_non_TMA ? 160 : 192," in h.read_text()
    with pytest.raises(SystemExit, match="exactly once"):
        tb.retile(h, 128)


def test_checked_lazy_max_biases_exp2_and_tests_signs_before_it(tmp_path):
    made = {new for _, new in tb.LAZY_PATCHES}
    own = [old for old, _ in tb.CHECKED_PATCHES if not any(old in new for new in made)]
    h = tmp_path / "softmax.h"
    h.write_text("".join(old for old, _ in tb.LAZY_PATCHES) + "".join(own) + 2 * f"    {tb.SUM_SCALE}\n")
    (tmp_path / "flash_fwd_kernel_sm90.h").write_text(f"    {tb.SOFTMAX_PINNED}\n")
    tb.lazy_max(h, checked=True)
    s = h.read_text()
    assert "lazy_exp2(scores);" in s and "neg >> 31" in s and "float(1ull << -Max_offset)" in s
    assert f"!Is_FP8 ? -{tb.LAZY_BIAS} : 8" in (tmp_path / "flash_fwd_kernel_sm90.h").read_text()
    with pytest.raises(SystemExit, match="exactly once"):
        tb.lazy_max(h, checked=True)


@pytest.mark.parametrize("redo", [False, True])
def test_held_lazy_max_moves_k_s_release_after_the_redo_test(tmp_path, redo):
    made = {new for _, new in tb.LAZY_PATCHES}
    own = [old for old, _ in tb.HELD_PATCHES if not any(old in new for new in made)]
    h = tmp_path / "softmax.h"
    h.write_text("".join(old for old, _ in tb.LAZY_PATCHES) + "".join(own))
    m = tmp_path / "mainloop_fwd_sm90_tma_gmma_ws.hpp"
    m.write_text("".join(old for old, _ in tb.HELD_MAINLOOP))
    tb.lazy_max(h, held=True, redo=redo)
    s, ml = h.read_text(), m.read_text()
    assert "bar.red.or.pred" in s and ("< -1.f" in s) == redo
    assert ml.index("softmax.lazy_redo(") < ml.index("pipeline_k.consumer_release(") < ml.index("VERITY_FA3_O")


@pytest.mark.parametrize("probe", tb.HELD_PROBES)
def test_held_probes_each_undo_one_of_helds_costs(tmp_path, probe):
    made = {new for _, new in tb.LAZY_PATCHES}
    own = [old for old, _ in tb.HELD_PATCHES if not any(old in new for new in made)]
    h = tmp_path / "softmax.h"
    h.write_text("".join(old for old, _ in tb.LAZY_PATCHES) + "".join(own))
    m = tmp_path / "mainloop_fwd_sm90_tma_gmma_ws.hpp"
    m.write_text("".join(old for old, _ in tb.HELD_MAINLOOP))
    tb.lazy_max(h, held=True, probe=probe)
    s, ml = h.read_text(), m.read_text()
    assert ml.count(tb.K_RELEASE) == 1 and ("bar.red.or.pred" in s) == (probe != "held-warp")
    early = ml.index(tb.K_RELEASE) < ml.index("softmax.lazy_redo(")
    assert early == (probe == "held-early-k")


def test_thresh_keeps_the_max_only_within_tau(tmp_path):
    h = tmp_path / "softmax.h"
    h.write_text("".join(old for old, _ in tb.THRESH_PATCHES))
    tb.lazy_max(h, thresh=True)
    s = h.read_text()
    assert f"<= {tb.THRESH_TAU}.f" in s and "skip_rescale = __all_sync(0xffffffffu, kept);" in s
