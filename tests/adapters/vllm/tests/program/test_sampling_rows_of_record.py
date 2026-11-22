"""`verity_vllm.definitions.registry.sampling_rows` against the GPU's own words (`test_sampling_rows`): on the GPU fixture's rows
(`fstoch01_rows.npz`, over a tracked file's limit) the vectorised row and the lane walk equal the device's served token and the
device kernel's masked row, and the `_v2` logs are the GPUs' words on every input of their probe, fetched from the evidence
store.
"""
from __future__ import annotations

import json
import os

import pytest

from tests.program.test_sampling_rows import _gpu_paths

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
        with open(path) as fh:
            d = json.load(fh)
        for name, key in (("nv_logf", "logf"), ("nv_log1pf", "log1pf")):
            pairs = [(int(x, 16), int(w, 16)) for x, w in zip(d[key]["in"], d[key]["out"])]
            for f in _gpu_paths(name):
                bad = [(hex(x), hex(f(x)), hex(w)) for x, w in pairs if f(x) != w]
                assert not bad, (path, name, len(bad), bad[:4])


