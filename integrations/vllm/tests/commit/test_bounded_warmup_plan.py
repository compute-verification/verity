"""The bounded warm-up learns every step's plan from `throwaway_requests`; the record run must stage the same layout.

SmolLM2-135M Gumbel (top-p 1) sm_120, B1 256/32: every decode step failed closed with
"ran 2005760 bytes against a plan of 2005504".  The warm-up requests dropped the per-request `seed`, so the warm-up batch was unseeded
and the sampler hook bound no `runner.sampler/seed`; every seeded record step then staged 8 more bytes past the plan's end.  The worker
compares 256 B-padded lengths, so the tail shows only where the plan ends within 8 bytes of a chunk boundary -- 135M's decode step did,
SmolLM2-360M's (the same vocabulary) and 135M top-p 0.95's (4 B of top-p splits before the seed) did not.

The adapter imports torch, so `throwaway_requests` is exec'd from source; the collector's layout rule is restated from
`native_collect.cpp` (`stage`: 4-byte alignment, gaps zeroed) and the worker's padded comparison from `native_collect.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

C = 256                                              # the chunk-leaf size the worker pads a step to
DECODE_PLAN_BYTES = 2_005_504                        # the learned 135M decode plan of the failing runs (a multiple of C)


def _adapter(name: str):
    src = (Path(__file__).resolve().parents[2] / "verity_vllm" / "engine" / "vllm_adapter.py").read_text()
    ns: dict = {"np": np, "Engine": object}
    for node in ast.parse(src).body:
        if (isinstance(node, ast.FunctionDef) and node.name == name) or \
           (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "THROWAWAY_SAMPLING_FIELDS" for t in node.targets)):
            exec(compile(ast.Module([node], []), "vllm_adapter", "exec"), ns)
    return ns[name]


throwaway_requests = _adapter("throwaway_requests")
ENGINE = SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
    model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=49152))))))


def _packed(nbytes: list[int]) -> int:
    """The collector's step length: each tensor at the next 4-byte offset (native_collect.cpp `stage`)."""
    cur = 0
    for nb in nbytes:
        cur = ((cur + 3) & ~3) + nb
    return cur


def _padded(used: int) -> int:
    return -(-used // C) * C


def _decode_step(requests: list[dict], body: list[int]) -> list[int]:
    """One decode step's staged tensors: the model body, the sampled ids (i32 per row), and -- as the sampler hook binds it, only for an
    explicitly seeded batch (`any_explicit_seed`) -- the seed (i64 per row)."""
    rows = len(requests)
    out = body + [4 * rows]
    if any(r.get("seed") is not None for r in requests):
        out.append(8 * rows)
    return out


def _body_ending_at(total: int) -> list[int]:
    """A synthetic model body (bf16 activations of hidden 576) whose step, with the sampled id, is exactly `total` bytes."""
    act = 576 * 2
    n, rest = divmod(total - 4, act)
    return [act] * n + ([rest] if rest else [])


RECORD = [{"request_id": "31b8c1c4", "prompt_token_ids": list(range(256)), "max_tokens": 32, "arrive_step": 0, "seed": 931850444, "n": 1}]


def _unseeded(reqs: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in ("seed", "n")} for r in reqs]


def test_an_unseeded_warmup_plan_is_one_chunk_short_of_the_seeded_record_step():
    body = _body_ending_at(DECODE_PLAN_BYTES)
    plan = _packed(_decode_step(_unseeded(RECORD), body))
    step = _packed(_decode_step(RECORD, body))
    assert plan == DECODE_PLAN_BYTES and step == plan + 8
    assert (_padded(step), _padded(plan)) == (2_005_760, 2_005_504)      # the failing runs' message, byte for byte


def test_the_same_tail_fits_the_padding_away_from_a_chunk_boundary():
    for total in (DECODE_PLAN_BYTES - 64, DECODE_PLAN_BYTES + 4):         # e.g. 360M, or 135M top-p 0.95 (4 B of splits first)
        body = _body_ending_at(total)
        assert _padded(_packed(_decode_step(RECORD, body))) == _padded(_packed(_decode_step(_unseeded(RECORD), body)))


def test_warmup_requests_keep_the_sampling_fields_so_the_plan_equals_the_record_layout():
    warm = throwaway_requests(ENGINE, RECORD, seed=200)
    assert [w["seed"] for w in warm] == [931850444] and [w["n"] for w in warm] == [1]
    assert len(warm[0]["prompt_token_ids"]) == 256 and warm[0]["prompt_token_ids"] != RECORD[0]["prompt_token_ids"]
    body = _body_ending_at(DECODE_PLAN_BYTES)
    assert _packed(_decode_step(warm, body)) == _packed(_decode_step(RECORD, body))


def test_warmup_requests_keep_a_per_request_variant_and_leave_unseeded_requests_unseeded():
    mixed = [{"request_id": "a", "prompt_token_ids": [5] * 8, "max_tokens": 4, "seed": None, "n": 1},
             {"request_id": "b", "prompt_token_ids": [6] * 3, "max_tokens": 2, "arrive_step": 1, "seed": 7, "n": 2,
              "variant": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 0}]
    warm = throwaway_requests(ENGINE, mixed, seed=200)
    assert warm[0]["seed"] is None and "variant" not in warm[0]
    assert {k: warm[1][k] for k in ("seed", "n", "variant", "temperature", "top_p", "top_k", "arrive_step", "max_tokens")} == \
        {"seed": 7, "n": 2, "variant": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "arrive_step": 1, "max_tokens": 2}
    assert [w["request_id"] for w in warm] == ["warmup-0", "warmup-1"]
