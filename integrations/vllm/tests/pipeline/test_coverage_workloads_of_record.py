"""`test_coverage_workloads`' generated B0 workloads (`verity_vllm/engine/profiles/workload_cov_b0_*.json`, generated on a pod and
not tracked) against the draws they were made from. Skipped where they are absent.
"""
from __future__ import annotations

import json

import pytest

from tests.pipeline.test_coverage_workloads import REPO
from verity_vllm.pipeline import coverage_workloads as gw


def test_generated_b0_workloads_match_their_draws():
    for prof in gw.PROFILES:
        for ctx in sorted(gw.CTX):
            p = REPO / "verity_vllm" / "engine" / "profiles" / f"workload_cov_b0_{prof.lower()}_{ctx.lower()}.json"
            if not p.exists():
                pytest.skip(f"{p.name} not generated in this checkout")
            doc = json.loads(p.read_text())
            draws = gw.draw_lengths(prof, ctx, doc["length_seed"])
            assert [(r["prompt_len"], r["max_tokens"], r["arrive_step"]) for r in doc["requests"]] == \
                   [(d["prompt_len"], d["max_tokens"], d["arrive_step"]) for d in draws]
            for r in doc["requests"]:
                assert len(r["prompt_token_ids"]) == r["prompt_len"]
            assert doc["expected_engine_steps"] == gw.simulate_schedule(draws)["expected_engine_steps"]
            assert doc["serving_profile"] == prof and doc["context_class"] == ctx
            if prof != "B1":
                assert doc["engine"]["args"]["max_num_batched_tokens"] >= 8 * gw.CTX[ctx]["cap"]
