"""prefix_cache's workload pair: the pinned warm/cold files are the declared experiment, and `gen` with its defaults writes
them (torch-free)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity_vllm.pipeline import prefix_cache as PC

PROFILES = Path(__file__).resolve().parents[2] / "workloads"
BS = 16


def test_generated_pair_is_the_declared_experiment():
    warm = json.loads((PROFILES / "workload_cov_b0_prefix_c256.json").read_text())
    cold = json.loads((PROFILES / "workload_cov_b0_prefix_c256_cold.json").read_text())
    assert warm["engine"]["args"] == {"enable_prefix_caching": True}
    assert cold["engine"]["args"] == {"enable_prefix_caching": False}
    # identical requests (ids, tokens, limits, arrivals) -- only the engine arg differs
    strip = lambda d: [{k: v for k, v in r.items() if k != "prefix_role"} for r in d["requests"]]  # noqa: E731
    assert strip(warm) == strip(cold)
    a, b = warm["requests"]
    assert a["arrive_step"] == 0 and b["arrive_step"] == PC.WARM_ARRIVE_STEP > 0
    assert a["prompt_len"] == b["prompt_len"] == PC.PREFIX_TOKENS + PC.CONT_TOKENS
    assert a["prompt_token_ids"][: PC.PREFIX_TOKENS] == b["prompt_token_ids"][: PC.PREFIX_TOKENS]
    assert a["prompt_token_ids"][PC.PREFIX_TOKENS:] != b["prompt_token_ids"][PC.PREFIX_TOKENS:]
    assert PC.PREFIX_TOKENS % BS == 0 and warm["feature"]["expect"]["cached_blocks_if_hit"] == PC.PREFIX_TOKENS // BS
    assert warm["feature"]["expect"]["num_computed_at_first_step"][b["request_id"]] == PC.PREFIX_TOKENS
    assert cold["feature"]["expect"]["num_computed_at_first_step"][b["request_id"]] == 0
    assert warm["expected_engine_steps"] == PC.WARM_ARRIVE_STEP + PC.MAX_TOKENS
    assert warm["serving_profile"] == PC.PROFILE_LABEL and warm["context_class"] == "C256"


class _ByteTok:
    """`coverage_workloads.Tok`'s interface over UTF-8 bytes, standing in for the served model's tokenizer."""

    path = Path("bytes.json")
    sha256 = "0" * 64

    def __init__(self, path: str) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", "replace")


def test_gen_with_its_defaults_writes_the_pinned_pair(tmp_path: Path, monkeypatch):
    from verity_vllm.pipeline import coverage_workloads as gw

    monkeypatch.setattr(gw, "Tok", _ByteTok)
    assert PC.main(PC.GenOptions(tokenizer="unused", out_dir=str(tmp_path))) == 0
    got = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tmp_path.glob("workload_*.json"))}
    assert got == {"workload_cov_b0_prefix_c256.json": "aa3bc587588897f046959836d19abb5f3d1701b980e08ba941bdee0a45401b59",
                   "workload_cov_b0_prefix_c256_cold.json": "95cf0f3c99dd50cd9af7616974cf05fd5ca1abd4c541d0b304311bb76fc77cee"}
