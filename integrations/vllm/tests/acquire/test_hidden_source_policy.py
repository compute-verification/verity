"""The hidden source against a manifest built under the guarded-max policy, which no manifest is built under since the M1 stream left the
manifest (`query.required`) but older manifests still carry: it refuses a tap build whose ROW word 3 or MS class is not the manifest's,
and the MS class lengthens the stream layout and the arena bound."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_vllm.acquire.sources.hidden_source import HiddenM1Source, guarded_max_policy, mat_total_words, tap_ms, tap_row3
from verity_vllm.commit.hidden_stream import StreamLayout


def test_the_policy_is_read_off_the_query_header():
    assert guarded_max_policy({"query": {"guarded_max": {"words": 5}}})
    assert not guarded_max_policy({"query": {"query_id": "Q"}}) and not guarded_max_policy({}) and not guarded_max_policy(None)


def test_the_layout_without_ms_is_the_record_s():
    off, on = StreamLayout(32, 256, 2), StreamLayout(32, 256, 2, ms=True)
    assert "MS" not in off.offsets and on.offsets["MS"] == off.total_words and on.digest() != off.digest()
    assert off.digest() == StreamLayout(32, 256, 2, 128, 64).digest() and off.total_words == mat_total_words(32, 256, 2)
    assert on.word_index("MS", 3, 17, block=1) == off.total_words + (3 * 256 + 17) * 2 + 1
    assert on.unrank(on.word_index("MS", 3, 17, block=1)) == {"cls": "MS", "slab": 3, "row": 17, "block": 1}
    assert on.unrank(off.offsets["FIN"] + 5)["cls"] == "FIN"
    with pytest.raises(ValueError):
        off.word_index("MS", 0, 0, block=0)


def _source(row3, src=None):
    s = object.__new__(HiddenM1Source)
    s.src = (123 if row3 else 59) if src is None else src
    s.ns_name, s.stats = "verity_fa2_matReqG" if row3 else "verity_fa2_matReq", {"row3": row3, "ms": tap_ms(s.src)}
    s.ms = s.stats["ms"]
    return s


def test_the_source_refuses_a_build_whose_word_3_or_ms_class_is_not_the_manifests():
    assert tap_row3(SimpleNamespace()) == 0 and tap_row3(SimpleNamespace(verity_tap_row3=lambda: 1)) == 1
    assert tap_ms(123) and not tap_ms(59) and not tap_ms(63)
    _source(1).require_row3(True)
    _source(0).require_row3(False)
    with pytest.raises(RuntimeError, match="mode 0 .*guarded-max policy and needs mode 1"):
        _source(0).require_row3(True)
    with pytest.raises(RuntimeError, match="mode 1 .*no guarded-max policy and needs mode 0"):
        _source(1).require_row3(False)
    with pytest.raises(RuntimeError, match="source mask 59 lacks the MS class"):
        _source(1, src=59).require_row3(True)                       # the guarded build: word 3 without the MS class
    with pytest.raises(RuntimeError, match="source mask 123 writes the MS class"):
        _source(0, src=123).require_row3(False)


def test_a_build_with_the_ms_class_widens_the_arena_bound():
    on, off = _source(1), _source(0)
    model = SimpleNamespace(config=SimpleNamespace(num_attention_heads=32, num_hidden_layers=16))
    env = {"batch": 1, "max_prompt": 256, "max_new": 32}
    assert off.extra_step_bytes(env, model) == {"per_token_bytes": 0, "per_seq_bytes": 0, "tensors_per_step": 0}
    assert on.extra_step_bytes(env, model) == {"per_token_bytes": 4 * 16 * 32 * 5, "per_seq_bytes": 0, "tensors_per_step": 16}
