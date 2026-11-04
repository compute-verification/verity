"""The lead's kernel off the device (benchmarks/pouw/np_kernel): cpu_check.cc's ragged calls, whose padding rows carry their
tags, lane-coin A' and coins (and a control whose padding rows have none is rejected), the fold's window, exact at
`serving.LEAD_FOLD_BOUND` and at no other bound, the lead serving verity_int8's own y at Llama-3.2-1B's linears (and
83d82109a8's fold caught there), G's rejection draw, and ssn.cu's check holding served outputs to verity_int8's epilogue
within the same bound."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from verity.core.protocols.pouw.serving import LEAD_FOLD_BOUND

HERE = Path(__file__).resolve().parents[1] / "np_kernel"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Llama-3.2-1B's fused linears, (n, K): qkv, o, gate_up, down (ssn.cu's time_all and check)
SERVED = ((3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192))


@pytest.fixture(scope="module")
def cpu_check(tmp_path_factory) -> Path:
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for cpu_check.cc")
    exe = tmp_path_factory.mktemp("np-kernel") / "cpu_check"
    subprocess.run([gxx, "-O2", "-std=c++17", "-ffp-contract=off", "-o", str(exe), str(HERE / "cpu_check.cc")], check=True)
    return exe


def run(exe: Path, *args: str) -> tuple[int, list[dict]]:
    p = subprocess.run([str(exe), *args], capture_output=True, text=True)
    return p.returncode, [json.loads(s) for s in p.stdout.splitlines()]


def test_a_ragged_calls_padding_rows_carry_their_tags_lane_coin_a_and_coins(cpu_check):
    rc, lines = run(cpu_check, "--bound", str(LEAD_FOLD_BOUND), "ragged")
    calls = [x for x in lines if x.get("level") == "ragged"]
    honest = [x for x in calls if not x["control"]]
    assert rc == 0 and lines[-1]["pass"] and all(x["pass"] for x in calls)
    assert {x["layout"] for x in calls} == {"blc4a_t64_e7_s1_rot_fold"}
    decode = next(x for x in honest if x["call_m"] == 1)
    assert (decode["committed_m"], decode["computed_m"], decode["padding_rows"]) == (16, 64, 15)
    assert all(x["padding_rows_wrong"] == x["padding_rows_tagless"] == x["word_mismatches"] == x["served_mismatches"] == 0
               and x["served_outputs"] > 0 for x in honest)
    controls = [x for x in calls if x["control"]]
    assert controls and all(x["padding_rows_wrong"] == x["padding_rows"] > 0 and x["word_mismatches"] > 0 for x in controls)


def test_the_fold_is_exact_within_the_leads_bound_and_misreads_past_it(cpu_check):
    rc, lines = run(cpu_check, "--bound", str(LEAD_FOLD_BOUND), "window")
    w = lines[0]
    assert rc == 0 and w["pass"] and w["read_back_not_p"] == w["served_differ_from_int8"] == 0 and w["max_abs_n_ij"] > 0
    assert w["read_back_at_bound_plus_1"] == LEAD_FOLD_BOUND + 2
    assert w["read_back_at_minus_bound_minus_1"] == -LEAD_FOLD_BOUND - 0.5


@pytest.mark.parametrize("bound", [LEAD_FOLD_BOUND // 2, LEAD_FOLD_BOUND * 2])
def test_no_other_bound_is_the_folds(cpu_check, bound):
    """Half the bound reads back past itself (so it isn't the fold's edge); twice the bound misreads within itself."""
    rc, lines = run(cpu_check, "--bound", str(bound), "window")
    assert rc == 1 and not lines[0]["pass"] and not lines[-1]["pass"]


@pytest.fixture(scope="module")
def served_calls(tmp_path_factory) -> list[tuple[Path, object, object]]:
    """verity_int8's calls at Llama-3.2-1B's fused linears on 64 rows, and decode's one row at qkv, as cpu_check's served
    level reads them: bf16 activations and weights, each quantized a row by `int8_exact.quantize_rows` (so the scales
    are real ones, with 24-bit significands), their `int_mm` product and the y `plain_linear` serves."""
    torch = pytest.importorskip("torch")
    from verity_vllm.engine import int8_exact as E

    FV = _module("np_kernel_fold_vectors", HERE / "fold_vectors.py")
    d = tmp_path_factory.mktemp("np-kernel-served")
    rng = np.random.default_rng(20261030)
    calls = []
    for m, (n, k) in [(64, s) for s in SERVED] + [(1, SERVED[0])]:
        x = torch.from_numpy(rng.standard_normal((m, k), dtype=np.float32)).to(torch.bfloat16)
        w8, s_w = E.quantize_rows(torch.from_numpy(rng.standard_normal((n, k), dtype=np.float32) * 0.02).to(torch.bfloat16))
        xq, s_a = E.quantize_rows(x)
        y = E.plain_linear(x, w8, s_w)
        path = d / f"m{m}_n{n}_k{k}.bin"
        FV.write(str(path), {"x": xq.numpy(), "s_a": s_a.numpy(), "w": w8.numpy(), "s_w": s_w.numpy(),
                             "z": E.int_mm(xq, w8).numpy(), "y": y.view(torch.int16).numpy().view(np.uint16),
                             "seed": np.array([1000 * m + n + k], np.int64)})
        calls.append((path, s_a.numpy(), s_w.numpy()))
    return calls


def served_lines(cpu_check, calls) -> tuple[int, list[dict]]:
    rc, lines = run(cpu_check, "--bound", str(LEAD_FOLD_BOUND), *(f"served:{p}" for p, _, _ in calls))
    return rc, [x for x in lines if x.get("level") == "served"]


def test_the_lead_serves_verity_int8s_y_at_the_served_shapes(cpu_check, served_calls):
    """Every output of the lead's fold on its own words (A' and B' formed with the certifier's coins, the fold's row and
    column words) is verity_int8's y bit for bit, under the per-row scales int8_exact gives."""
    scales = np.concatenate([np.concatenate([a, w]) for _, a, w in served_calls])
    assert np.mean(scales.view(np.uint32) & 3 != 0) > 0.5   # 24-bit significands, not the 22 the old checks used
    rc, calls = served_lines(cpu_check, served_calls)
    assert rc == 0 and len(calls) == 5 and all(x["pass"] for x in calls)
    assert {(x["n"], x["K"]) for x in calls} == set(SERVED) and {x["m"] for x in calls} == {64, 1}
    assert all(x["served_mismatches"] == x["read_back_not_z"] == x["past_bound"] == 0 and x["max_abs_n_ij"] > 0
               for x in calls)
    assert sum(x["outputs"] for x in calls) == 64 * sum(n for n, _ in SERVED) + SERVED[0][0]


def test_83d82109a8s_fold_doesnt_serve_verity_int8s_y(cpu_check, served_calls):
    """A fold that takes sigma alone out of the word (83d82109a8's FOLD=1 fold) doesn't serve verity_int8's y: on the
    same words it leaves n_ij = the word less sigma less A . B in (row source 0, column source 0x4B400000 - sigma), rounds
    through FFMA(v, s_j, fl(-1.5 2^23 s_j)) and applies the weight's scale first. The served level tells that fold, and
    each of its departures alone, from verity_int8's y, so a check against verity_int8's y fails on it."""
    _, calls = served_lines(cpu_check, served_calls)
    out = sum(x["outputs"] for x in calls)
    share = {k: sum(x[f"control_{k}_mismatches"] for x in calls) / out
             for k in ("83d82109a8", "n_ij_left", "ffma", "weight_scale_first")}
    assert share["83d82109a8"] > 0.05 and share["n_ij_left"] > 0.05 and share["ffma"] > 0.002, share
    assert share["weight_scale_first"] > 0, share


@pytest.mark.slow
def test_gs_draw_keeps_each_g_from_equally_many_words(cpu_check):
    """All 2^32 words through g_word_of (about a minute): a bias of a few words in one G, which no sample shows."""
    rc, lines = run(cpu_check, "draw")
    assert rc == 0 and lines[0]["pass"] and lines[0]["kept_per_g_min"] == lines[0]["kept_per_g_max"]


def test_ssn_checks_served_outputs_to_verity_int8s_epilogue_within_the_leads_bound():
    src = (HERE / "ssn.cu").read_text()
    m = re.search(r"^constexpr long long FOLD_BOUND = 1ll << (\d+);$", src, re.M)
    assert m and 1 << int(m.group(1)) == LEAD_FOLD_BOUND
    assert "pv < -FOLD_BOUND || pv > FOLD_BOUND" in src and '\\"fold_bound\\": %lld' in src
    assert "const long long pv = ab;" in src and "bf16_rn(int8_epilogue((int)pv, si[i], sj[j]))" in src
    assert '"\\"int8_epilogue(A.B)\\""' in src and "stock_fold" not in src
