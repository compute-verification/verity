"""`verity.ml.definitions.registry.sampling_rows` against the GPU's own words (`test_sampling_rows`): on the GPU fixture's rows
(`fstoch01_rows.npz`, over a tracked file's limit) the vectorised row and the lane walk equal the device's served token and the
device kernel's masked row, and the `_v2` logs are the GPUs' words on every input of their probe, fetched from the evidence
store.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.program.test_sampling_rows import _gpu_paths, f32b, needs_tables
from verity.ml.definitions.registry import sampling_rows as SR

FIX = os.path.join(os.path.dirname(__file__), "data", "topp_split_fixture")
#: the `_v2` logs' probe on the L40S and the H100: each FTZ build's output on every probed input word
PROBE = "art:e5dd6ce9fa74e7ca4538b848943d8bb555f2adac528277848395dbf4cdff66a5"


@pytest.mark.slow
def test_the_v2_logs_are_the_gpus_words_on_every_probed_input(tmp_path):
    """Every input word of the probe (20,007 for logf, 20,005 for log1pf: the edges, then random negatives and NaNs), against each FTZ
    build's output on the L40S and the H100.  Skipped where the evidence store is unreachable."""
    import glob

    from tests.regression import store_io
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    root = store_io.fetch(PROBE, to=tmp_path / "probe")
    runs = sorted(glob.glob(str(root / "*" / "out-*-ftz-true-*.json")))
    assert len(runs) == 4
    for path in runs:
        d = json.load(open(path))
        for name, key in (("nv_logf", "logf"), ("nv_log1pf", "log1pf")):
            pairs = [(int(x, 16), int(w, 16)) for x, w in zip(d[key]["in"], d[key]["out"])]
            for f in _gpu_paths(name):
                bad = [(hex(x), hex(f(x)), hex(w)) for x, w in pairs if f(x) != w]
                assert not bad, (path, name, len(bad), bad[:4])


@needs_tables
@pytest.mark.slow
def test_the_r16_gpu_fixture_rows_device_token_masked_row_and_lane_walk_all_agree_with_the_vectorised_row():
    """Three real rows of the llama B8 stochastic record (`fstoch01_rows.npz`, an L40S, vLLM d9105ea80): the device's served token,
    the kernel-consumed masked row, seed / pos / temperature / top_p as the kernel read them.  The vectorised row reproduces the
    served token AND every lane of the masked row; the lane walk is run on one of them (10 s) as the reference beside it."""
    from verity_vllm.properties.admission import evaluate_spec
    from verity.ml.definitions.registry import sampling_topp_difftest as D
    z = np.load(os.path.join(FIX, "fstoch01_rows.npz"))
    meta = json.load(open(os.path.join(FIX, "fstoch01_rows.npz.json")))
    first = True
    for k, v in meta.items():
        bf = z[f"{k}_bf16_logits_bits"]
        masked_dev = z[f"{k}_masked_f32_bits"]
        d = v["device"]
        out = SR.gumbel_topp_token_select_row(bf, f32b(d["top_p"]), d["seed"], d["pos"], f32b(d["temperature"]), v["S"])
        assert out["token"] == v["served"], (k, out["token"], v["served"])
        assert int((out["masked"] == masked_dev).sum()) == v["V"], k
        if first:
            words = [bf.astype(np.uint64), np.array([f32b(d["top_p"])], dtype=np.uint64), np.array([d["seed"]], dtype=np.uint64),
                     np.array([d["pos"]], dtype=np.uint64), np.array([f32b(d["temperature"])], dtype=np.uint64)]
            assert evaluate_spec(D.spec({"V": v["V"], "S": v["S"]}), words)[0] == v["served"]
            first = False
