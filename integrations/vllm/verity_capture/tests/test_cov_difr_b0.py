"""[COV cov-difr] The B0 sampled-row profile `gen_ov_sampling_b0` (SmolLM2 pinned profile + the TokenSampleGumbel recognizer):
torch-free, no run artifacts.  The default B0 profile stays the pinned one (PROTECTED); the `_t1` cells select this module with
`--profile gen_ov_sampling_b0`."""

from __future__ import annotations

import json
from pathlib import Path

from verity_capture.profiles import gen_ov_sampling_b0 as OVB0, gen_ov_sampling_b1 as OVB1, profile_module_for_case, vllm_d9105ea80_sm89_eager as BASE
from verity_capture.profiles.gen_ov_sampling_patterns import TokenSampleGumbel
from verity_capture.tests.test_gen_sampling import _adapter_fn

PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def test_b0_profile_is_the_pinned_profile_plus_the_sampler_recognizer():
    p = OVB0.PROFILE
    assert p.id == BASE.PROFILE_ID + "+ov-sampling" and p.id != BASE.PROFILE.id and p.id != OVB1.PROFILE.id
    assert p.accepted is False
    assert p.config is BASE.PROFILE.config and p.notes["case"] == "B0" and p.weight_field is BASE.PROFILE.weight_field
    assert isinstance(p.patterns[0], TokenSampleGumbel) and p.patterns[0].vocab == BASE.C["V"] == 49152
    assert [type(x) for x in p.patterns[1:]] == [type(x) for x in BASE.PROFILE.patterns]
    assert p.kernel_allowlist is BASE.PROFILE.kernel_allowlist
    assert "ov_sampling" in p.notes and "GumbelArgmaxBf16" in p.notes["ov_sampling"]["definition"]


def test_default_b0_profile_is_unchanged():
    assert profile_module_for_case("B0") == "vllm_d9105ea80_sm89_eager"


def test_t1_workloads_load_as_seeded_temperature_one_headers():
    """The D87 `_t1` workloads (sampling lives in the workload): temperature 1.0, top_p 1, top_k 0, one seed per request,
    `greedy: False` -- the header the recognizer classifies as GumbelArgmaxBf16 rows (no `_temperature_kernel` at T = 1)."""
    load = _adapter_fn("load_workload")
    for name, profile, n_req in (("workload_cov_b0_b1_c256_t1", "B1", 1), ("workload_cov_b0_b8-mixed_c256_t1", "B8-mixed", 8)):
        p = PROFILES / f"{name}.json"
        wl = load(p)
        s = wl["sampling"]
        assert s["greedy"] is False and s["temperature"] == 1.0 and s["top_p"] == 1.0 and s["top_k"] == 0
        assert s["variant"] == "t1" and s["seed"] == 20260917 and s["n"] == 1 and "per_request" not in s
        assert len(wl["requests"]) == n_req and all(r["seed"] == 20260917 for r in wl["requests"])
        doc = json.loads(p.read_text())
        assert doc["serving_profile"] == profile and doc["context_class"] == "C256"
