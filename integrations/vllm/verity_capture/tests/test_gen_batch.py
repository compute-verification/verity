"""GEN batch lane: the concurrency / chunked-prefill / long-prompt / prefix-caching workloads are well-formed, and the torch-free
step-attribution diagnostic (`bench/gen_batch_attribution.py`) reads the M1 log correctly and flags schedule departures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_capture.bench import gen_batch_attribution as ga
from verity_capture.bench import gen_batch_provenance as gp
from verity_capture.card import workload_summary

REPO = Path(__file__).resolve().parents[2]
PROF = REPO / "workloads"
M1_LOG = REPO / "data" / "logs" / "m1.jsonl.gz"
VOCAB = {"B0": 49152, "B1": 151936}
WORKLOADS = {
    "workload_b0_conc2_small.json": ("B0", [(7, 16), (19, 6)], {}),
    "workload_b0_conc4_mixed.json": ("B0", [(7, 16), (32, 8), (61, 12), (128, 4)], {}),
    "workload_b0_conc8_mixed.json": ("B0", [(7, 16), (32, 8), (61, 12), (128, 4), (16, 16), (48, 6), (96, 10), (24, 14)], {}),
    "workload_b0_chunked_200p.json": ("B0", [(7, 8), (200, 12)], {"max_num_batched_tokens": 64}),
    "workload_b0_long512_1req.json": ("B0", [(512, 8)], {}),
    "workload_b0_prefix64_2req.json": ("B0", [(88, 8), (88, 8)], {"enable_prefix_caching": True}),
    "workload_qwen15_conc4_mixed.json": ("B1", [(7, 16), (32, 8), (61, 12), (128, 4)], {}),
}


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_workload_schema(name: str) -> None:
    case, spec, engine_args = WORKLOADS[name]
    doc = json.loads((PROF / name).read_text())
    assert doc["schema"] == "verity-vllm/workload/v1" and doc["case"] == case
    assert doc["variants"]["greedy"] == {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
    assert doc.get("engine_args_required", {}) == engine_args
    reqs = doc["requests"]
    assert [(r["prompt_len"], r["max_tokens"]) for r in reqs] == spec
    assert [r["index"] for r in reqs] == list(range(len(reqs)))
    assert len({r["request_id"] for r in reqs}) == len(reqs)
    for r in reqs:
        assert len(r["prompt_token_ids"]) == r["prompt_len"] and r["max_tokens"] > 0
        assert all(0 <= t < VOCAB[case] for t in r["prompt_token_ids"])
        assert "accepted_token_ids" not in r and "expected_token_ids" not in r         # the clean CONTROL run is the reference
    assert "accepted_token_ids" not in doc
    s = workload_summary(PROF / name)
    assert s["n_requests"] == len(reqs) and s["case"] == case and s["prompt_lens"] == sorted({p for p, _ in spec})


def test_prefix_workload_shares_exactly_64_tokens() -> None:
    doc = json.loads((PROF / "workload_b0_prefix64_2req.json").read_text())
    a, b = (r["prompt_token_ids"] for r in doc["requests"])
    assert a[:64] == b[:64] and a[64] != b[64] and len(a) == len(b) == 88
    assert all(r["shared_prefix_len"] == 64 for r in doc["requests"])
    assert 64 % 16 == 0                                                              # four full KV blocks


def test_chunked_workload_exceeds_the_chunk() -> None:
    doc = json.loads((PROF / "workload_b0_chunked_200p.json").read_text())
    budget = doc["engine_args_required"]["max_num_batched_tokens"]
    assert budget == 64 and max(r["prompt_len"] for r in doc["requests"]) > 3 * budget
    assert min(r["prompt_len"] for r in doc["requests"]) < budget                     # the decode request fits beside a chunk


def test_concurrency_workloads_match_the_frozen_harness_assumption() -> None:
    """`gates.expected_steps` / `noninterference.compare` assume max(max_tokens) steps: the concurrency workloads are designed for it
    (all requests admitted at step 0 under the default token budget, some finishing early)."""
    for name in ("workload_b0_conc2_small.json", "workload_b0_conc4_mixed.json", "workload_b0_conc8_mixed.json", "workload_qwen15_conc4_mixed.json"):
        doc = json.loads((PROF / name).read_text())
        total_prompt = sum(r["prompt_len"] for r in doc["requests"])
        assert total_prompt <= 2048 and len(doc["requests"]) <= 8                    # default max_num_batched_tokens / --max-num-seqs 8
        assert len({r["max_tokens"] for r in doc["requests"]}) > 1                   # the batch shrinks mid-run
        assert max(r["max_tokens"] for r in doc["requests"]) == 16


@pytest.mark.skipif(not M1_LOG.exists(), reason="corpus log not present")
def test_attribution_m1_on_schedule(tmp_path: Path) -> None:
    rep = ga.analyze(M1_LOG, PROF / "workload_32x16_1req.json", control_tokens=REPO / "data" / "logs" / "m1_ctl_tokens.json")
    assert rep["n_steps"] == 16 == rep["expected_steps_harness"] and rep["on_schedule"] and rep["deviations"] == []
    assert rep["steps"][0]["rows"] == 32 and all(s["rows"] == 1 for s in rep["steps"][1:])
    r0 = rep["requests"][0]
    assert r0["n_sampled"] == 16 and r0["prompt_positions_observed_as_rows"] == 32
    assert all(rep["checks"]["log_sampled_equals_control"].values())
    md = ga.markdown(rep)
    assert "| 0 | prefill | 32 |" in md and "none" in md.split("## Deviations")[1]
    assert ga.main(["--log", str(M1_LOG), "--workload", str(PROF / "workload_32x16_1req.json"), "--out", str(tmp_path / "attr")]) == 0
    assert (tmp_path / "attr.json").exists() and (tmp_path / "attr.md").exists()


@pytest.mark.skipif(not M1_LOG.exists(), reason="corpus log not present")
def test_attribution_flags_schedule_departures(tmp_path: Path) -> None:
    doc = json.loads((PROF / "workload_32x16_1req.json").read_text())
    doc["requests"][0]["max_tokens"] = 12                                          # the log ran 16 steps: 4 unexpected steps
    doc["requests"][0]["prompt_token_ids"][3] ^= 1                                 # one prompt token differs from the log's row
    wl = tmp_path / "wl.json"
    wl.write_text(json.dumps(doc))
    rep = ga.analyze(M1_LOG, wl)
    kinds = rep["deviation_kinds"]
    assert not rep["on_schedule"] and kinds["unexpected_request_present"] == 4 and kinds["step_count"] == 1
    assert kinds["prompt_token_mismatch"] == 1 and kinds["sampled_count_differs_from_max_tokens"] == 1


@pytest.mark.skipif(not M1_LOG.exists(), reason="corpus log not present")
def test_provenance_m1_single_request_closes_over_its_own_prompt(tmp_path: Path) -> None:
    """The control case of the prefix-caching note: one request, every declared prompt position is read, the sampled token
    closes over exactly prompt[0..31], and the first attention row's K/V come from the request's own instances."""
    rep = gp.analyze(M1_LOG, "vllm_d9105ea80_sm89_eager", 0)
    assert rep["fold"]["unresolved"] == 0 and rep["fold"]["errors"] == 0
    assert rep["num_computed"] == {"req0": 0}
    (cov,) = rep["prompt_consumption"]
    assert cov["param"] == "prompt" and cov["consumed"] == cov["declared"] == 32 and cov["never_consumed"] == "none"   # 1 request: `prompt`
    (tok,) = rep["sampled_token_upstream"]
    assert tok["request"] == "req0@31" and tok["prompt_positions"] == {"prompt": "0..31 (32)"}
    (att,) = rep["attention_layer0"]
    assert att["request"] == "req0" and att["position"] == 0
    assert all(k.startswith("req0:") for o in att["operands"] for k in o["from"])
    md = gp.render(rep)
    assert "never consumed: none" in md and "req0:RoPE_v1" in md
    out = tmp_path / "prov.md"
    assert gp.main(["--run-dir", str(M1_LOG), "--profile", "vllm_d9105ea80_sm89_eager", "--out", str(out)]) == 0
    assert out.exists() and out.with_suffix(".json").exists()
