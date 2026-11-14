"""`admission_bound.scheduled_step_bytes` and `predicted_pool_bytes` on schedules and paths no Commit record of
`test_admission_commit.py` holds: a schedule of no request, a step at which no request is live, the hidden stream an FA3 launch
binds, and the pool of a run staged in per-step blocks.  The sizing is a small one written here, so each figure can be read off."""
from __future__ import annotations

from typing import Any

import pytest

import verity_vllm.pipeline.admission_bound as AB
from verity_vllm.acquire.sources.hidden_source import CHUNK_WORDS, mat_total_words

Record = dict[str, Any]
PER_TOKEN, PER_SEQ, PER_STEP, LAYERS = 100, 10, 1000, 2
MODEL: Record = {"layers": LAYERS, "heads": 4, "kv_heads": 2, "head_dim": 64}


def sizing(**model: Any) -> Record:
    return {"model": MODEL | model, "per_token_bytes": PER_TOKEN, "per_seq_bytes": PER_SEQ, "per_step_bytes": PER_STEP, "rope_per_layer": True}


def request(prompt: int, max_tokens: int, arrive: int = 0) -> Record:
    return {"prompt_len": prompt, "max_tokens": max_tokens, "arrive_step": arrive}


def stream_bytes(slabs: int, rows: int, key_blocks: int, block_n: int, head_dim: int = 64) -> int:
    """The bytes of one launch's M1 stream in every layer: its layout's words, to a whole chunk, 4 B a word."""
    return LAYERS * 4 * -(-mat_total_words(slabs, rows, key_blocks, block_n, head_dim) // CHUNK_WORDS) * CHUNK_WORDS


def test_a_schedule_of_no_request_is_refused() -> None:
    with pytest.raises(ValueError, match="^no requests$"):
        AB.scheduled_step_bytes(sizing(), [])


def test_a_step_at_which_no_request_is_live_binds_nothing_and_is_no_step_of_the_schedule() -> None:
    """A request is live from its arrival to its last row.  Where the next arrives after the one before it ended, the steps between
    bind nothing: the horizon counts them, the schedule's steps do not, and the run's bytes are the two requests' own."""
    first, second = request(3, 1), request(5, 2, arrive=4)
    both = AB.scheduled_step_bytes(sizing(), [first, second], lag=0)
    alone = [AB.scheduled_step_bytes(sizing(), [r | {"arrive_step": 0}], lag=0) for r in (first, second)]
    assert [x["step"] for x in both["per_step"]] == [0, 4, 5] and (both["steps"], both["horizon"]) == (3, 6)
    assert [(x["prefills"], x["decodes"], x["tokens"]) for x in both["per_step"]] == [(1, 0, 3), (1, 0, 5), (0, 1, 1)]
    assert [x["bytes"] for x in both["per_step"]] == [x["bytes"] for a in alone for x in a["per_step"]]
    assert both["total_bytes"] == alone[0]["total_bytes"] + alone[1]["total_bytes"] == sum(x["bytes"] for x in both["per_step"])


def test_an_fa3_launch_binds_a_stream_of_its_kv_heads_slabs_over_the_key_blocks_of_its_tile() -> None:
    """An FA3 launch's M1 stream holds a slab for each KV head of each sequence, of a row for each query of each of the KV head's
    query heads: as many rows in all as an FA2 launch's slab for each query head.  It differs in its key blocks, which are those of
    the FA3 kernel's tile: 192 keys a block while a slab's rows are at most 64 (one warpgroup) and 128 past that, where FA2's are
    128 at this head dimension.  The keys are those of the longest context a live request can reach."""
    def hidden(prompt: int, max_tokens: int, **model: Any) -> list[int]:
        doc = AB.scheduled_step_bytes(sizing(**model), [request(prompt, max_tokens)], lag=0)
        assert doc["attn_backend"] == (model.get("attn_backend") or "fa2")
        assert doc["composition"]["hidden_bytes"] == sum(x["hidden_bytes"] for x in doc["per_step"])
        return [x["hidden_bytes"] for x in doc["per_step"]]

    assert hidden(8, 1, attn_backend="fa3") == [stream_bytes(2, 16, 1, 192)]
    assert hidden(8, 2, attn_backend="fa3") == [stream_bytes(2, 16, 1, 192), stream_bytes(2, 2, 1, 192)]
    assert hidden(40, 1, attn_backend="fa3") == [stream_bytes(2, 80, 1, 128)]
    assert hidden(128, 1, attn_backend="fa3") == [stream_bytes(2, 256, 2, 128)], "a context of 129 is two key blocks, its prompt one"
    assert hidden(200, 2, attn_backend="fa3") == [stream_bytes(2, 400, 2, 128), stream_bytes(2, 2, 2, 192)]
    assert hidden(8, 1, attn_backend="fa3", head_dim=128) == [stream_bytes(2, 16, 1, 128, 128)]
    assert hidden(8, 1) == hidden(8, 1, attn_backend=None) == [stream_bytes(4, 8, 1, 128)]


@pytest.mark.parametrize("head_dim", [96, 256])
def test_an_fa3_launch_of_a_head_dimension_its_tap_has_no_kernel_for_binds_no_hidden_stream(head_dim: int) -> None:
    doc = AB.scheduled_step_bytes(sizing(attn_backend="fa3", head_dim=head_dim), [request(8, 2)], lag=0)
    assert doc["composition"]["hidden_bytes"] == 0 and doc["hidden_m1"] is True
    assert [x["bytes"] for x in doc["per_step"]] == [8 * PER_TOKEN + PER_SEQ + PER_STEP, PER_TOKEN + PER_SEQ + PER_STEP]
    assert AB.scheduled_step_bytes(sizing(head_dim=head_dim), [request(8, 2)], lag=0)["composition"]["hidden_bytes"] > 0


def test_a_run_staged_in_per_step_blocks_pins_its_raw_bytes_and_their_levels_only_when_it_retains_them_on_the_host() -> None:
    raw = 8 << 20
    assert AB.predicted_pool_bytes(raw, 0, retain="host", levels_factor=1.25, windowed=False, bounded=False, retain_exclude=None) == (
        10 << 20, "per-step blocks, retain host: raw 8 MiB x 1.25")
    assert AB.predicted_pool_bytes(raw, 0, retain="none", levels_factor=1.25, windowed=False, bounded=False, retain_exclude=None) == (
        0, "per-step blocks, retain none: no raw bytes copied to host")
