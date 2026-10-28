"""The guarded-max policy (`verity_vllm.query.guarded_max`): the guarded maxes and `max_scaled` values an attention Call commits, the
query header that states them, the MS class in the hidden-stream identities, manifest-verify telling the policies apart, and the hidden
source refusing a tap build whose ROW word 3 or MS class is not the manifest's."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_vllm.acquire.sources.hidden_source import HiddenM1Source, mat_total_words, tap_ms, tap_row3
from verity_vllm.commit.hidden_stream import StreamLayout
from verity_vllm.query import guarded_max as GM
from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.manifest import verify as V


def _call(family, **statics):
    return SimpleNamespace(family=family, statics=statics, definition=f"{family}{statics}")


def _program():
    """A Llama-shaped attention: one Call per query row, T keys, 32 heads, kBlockN 128 -- rows up to 128 keys visit one key block, 129..256 two,
    257..384 three -- plus a legacy Call without BN (64) and a non-attention Call."""
    calls = [_call("Attention_v3", T=t, NH=32, KVH=8, D=64, BN=128) for t in (1, 128, 129, 256, 287)]
    return SimpleNamespace(calls=calls + [_call("Attention_v1", T=130, NH=2, KVH=2, D=128), _call("Gemm_v1", K=64, N=8)])


def test_attention_words_count_the_blocks_after_each_rows_first():
    assert GM.attention_words(_program()) == {"Attention_v1": 2 * 2, "Attention_v3": 32 * (0 + 0 + 1 + 1 + 2)}


def test_max_scaled_words_count_the_blocks_with_more_than_one_visible_key():
    # T = 1: one key; 128: one full block; 129: a full block and a one-key block; 256: two full; 287: two full and 31 keys
    assert GM.max_scaled_words(_program()) == {"Attention_v1": 2 * 3, "Attention_v3": 32 * (0 + 1 + 1 + 2 + 3)}


def _row_101():
    layers = 16
    prefill = [_call("Attention_v3", T=r + 1, NH=32, KVH=8, D=64, BN=128) for r in range(256)] * layers
    decode = [_call("Attention_v3", T=256 + s, NH=32, KVH=8, D=64, BN=128) for s in range(1, 32)] * layers
    return SimpleNamespace(calls=prefill + decode)


def test_101_counts_the_tap_lists_97_280_guards_and_242_688_max_scaled():
    assert GM.attention_words(_row_101()) == {"Attention_v3": 97_280}
    assert GM.max_scaled_words(_row_101()) == {"Attention_v3": 242_688}


def test_under_fa3_per_iteration_check_inf_only_the_guarded_blocks_commit_a_guard():
    # Attention_v4 blocks below MASKED_FROM read the running max itself (ROW word 0); max * scale is committed at every block
    calls = [_call("Attention_v4", T=287, NH=32, KVH=8, D=128, BN=128, MASKED_FROM=mf) for mf in (2, 1, 0)]
    P = SimpleNamespace(calls=calls)
    assert GM.attention_words(P) == {"Attention_v4": 32 * (0 + 1 + 2)} and GM.max_scaled_words(P) == {"Attention_v4": 32 * 3 * 3}


def test_tensor_parallel_attention_is_refused_by_name():
    with pytest.raises(ValueError, match="AttentionTP2_v1.*tensor-parallel"):
        GM.attention_words(SimpleNamespace(calls=[_call("AttentionTP2_v1", T=5, C={})]))


def _query():
    return {"query_id": "Q_module_body_v1", "query_text": "q", "query_id_digest": "d", "attention_hidden_family": "fa2_hidden_m1_stream"}


def test_the_header_states_the_policy_and_a_global_header_sums_its_components():
    q = GM.header(_query(), {"Attention_v3": 5}, {"Attention_v3": 6})
    assert q["guarded_max"] == {"stream_family": "fa2_hidden_m1_stream", "row_field": 3, "tap_row3": 1, "words": 5,
                                "words_by_family": {"Attention_v3": 5},
                                "max_scaled": {"stream_class": "MS", "src_bit": 64, "words": 6, "words_by_family": {"Attention_v3": 6}}}
    assert q["query_text"].endswith(GM.POLICY_TEXT) and q["query_id_digest"] != "d"
    parts = [q, GM.header(_query(), {"Attention_v3": 2, "AttentionSoftcap_v1": 4}, {"Attention_v3": 3, "AttentionSoftcap_v1": 5})]
    g = GM.merged(q, parts)["guarded_max"]
    assert g["words_by_family"] == {"Attention_v3": 7, "AttentionSoftcap_v1": 4} and g["words"] == 11
    assert g["max_scaled"] == {"stream_class": "MS", "src_bit": 64, "words": 14, "words_by_family": {"Attention_v3": 9, "AttentionSoftcap_v1": 5}}
    assert GM.policy_of({"query": q}) and not GM.policy_of({"query": _query()}) and not GM.policy_of(None)


def _stream_row(HB, M, NB, BN=128, D=64, **extra):
    return {"family": "fa2_hidden_m1_stream", "program_digest": "p", "request_id": "r0", "step": 0, "invocation": 0, "op_path": "fa2.m1/L0",
            "output_member": f"g{HB}x{M}x{NB}", "element_range": [0, fmt.mat_total_words(HB, M, NB, BN, D)], "rank": 0, "dtype": "u32",
            "geometry": {"HB": HB, "M": M, "NB": NB, "BN": BN, "D": D, **extra}}


def test_the_ms_class_lengthens_every_stream_identity_and_moves_the_digest():
    other = {"family": "input_token_ids", "program_digest": "p", "request_id": "r0", "step": 0, "invocation": 0, "op_path": "m",
             "output_member": "in", "element_range": [0, 256], "rank": 0}
    ids = [_stream_row(32, 256, 2), _stream_row(32, 1, 3), other]
    man = {"query": _query(), "identities": ids, "manifest_digest": fmt.manifest_digest(ids)}
    on = GM.with_ms(man)
    assert on["manifest_digest"] != man["manifest_digest"] == fmt.manifest_digest(ids)       # the record's manifest is untouched
    (a, b, c) = on["identities"]
    assert a["element_range"] == [0, StreamLayout(32, 256, 2, ms=True).total_words] == [0, mat_total_words(32, 256, 2, ms=True)]
    assert a["element_range"][1] - ids[0]["element_range"][1] == 32 * 256 * 2 and a["geometry"]["ms"] is True
    assert b["element_range"][1] - ids[1]["element_range"][1] == 32 * 1 * 3 and c == other
    assert GM.with_ms(on) == on                                                                # applied once
    with pytest.raises(ValueError, match="not its geometry's stream"):
        GM.with_ms(dict(man, identities=[dict(ids[0], element_range=[0, 5])]))


def test_the_layout_without_ms_is_the_record_s():
    off, on = StreamLayout(32, 256, 2), StreamLayout(32, 256, 2, ms=True)
    assert "MS" not in off.offsets and on.offsets["MS"] == off.total_words and on.digest() != off.digest()
    assert off.digest() == StreamLayout(32, 256, 2, 128, 64).digest() and off.total_words == mat_total_words(32, 256, 2)
    assert on.word_index("MS", 3, 17, block=1) == off.total_words + (3 * 256 + 17) * 2 + 1
    assert on.unrank(on.word_index("MS", 3, 17, block=1)) == {"cls": "MS", "slab": 3, "row": 17, "block": 1}
    assert on.unrank(off.offsets["FIN"] + 5)["cls"] == "FIN"
    with pytest.raises(ValueError):
        off.word_index("MS", 0, 0, block=0)


def _manifest(query):
    return {"manifest_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "identities": [], "program_digest": "p",
            "populations": {"identities": 0}, "complete": True, "query": query}


def test_manifest_verify_tells_the_policies_apart_and_records_them_only_when_one_side_has_it():
    on, off = _manifest(GM.header(_query(), {"Attention_v3": 5}, {"Attention_v3": 6})), _manifest(_query())
    assert V.verify(off, off)["ok"] and "used_guarded_max" not in V.verify(off, off)
    assert V.verify(on, on)["ok"] and V.verify(on, on)["rebuilt_guarded_max"]["words"] == 5
    for used, rebuilt in ((on, off), (off, on)):
        r = V.verify(used, rebuilt)
        assert not r["ok"] and r["why"].startswith("guarded-max policy")


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
