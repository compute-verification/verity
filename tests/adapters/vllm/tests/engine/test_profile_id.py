"""`engine_profile.profile_id_of`: the profile id records carry is the first 12 hex digits of the sha256 of the manifest's
canonical JSON, pinned here on fixed manifests (non-ASCII text, NaN and infinities included)."""

import pytest

from verity_vllm.engine import engine_profile as prof

CASES = {
    "engine": ({"engine_kwargs": {"model": "m", "dtype": "bfloat16", "enforce_eager": True, "max_num_seqs": 16, "seed": 0},
                "versions": {"vllm": "0.11.0", "torch": "2.8.0"}, "gpu": {"name": "dev", "cc": [8, 9]},
                "sampling": {"temperature": 0.0, "top_p": 1.0}}, "54c674310f6f"),
    "unicode": ({"note": "é ≥ 🙂", "z": [3, 1, 2], "a": None}, "82f8b57ec644"),
    "nan": ({"x": float("nan"), "y": float("inf"), "w": -0.0}, "45b00c92caf7"),
    "empty": ({}, "44136fa355b3"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_profile_id_of_is_pinned(name):
    manifest, want = CASES[name]
    assert prof.profile_id_of(manifest) == want
