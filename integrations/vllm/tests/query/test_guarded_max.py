"""The guarded-max policy (`verity_vllm.query.guarded_max`): the guarded maxes an attention Call commits, the query header that states
them (identities and manifest digest unchanged), manifest-verify telling the policies apart, and the hidden source refusing a tap build
whose ROW word 3 is not the manifest's."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_vllm.acquire.sources.hidden_source import HiddenM1Source, tap_row3
from verity_vllm.query import guarded_max as GM
from verity_vllm.query.manifest import verify as V


def _call(family, **statics):
    return SimpleNamespace(family=family, statics=statics, definition=f"{family}{statics}")


def _program():
    """#101's attention shape: one Call per query row, T keys, 32 heads, kBlockN 128 -- rows up to 128 keys visit one key block, 129..256 two,
    257..384 three -- plus a legacy Call without BN (64) and a non-attention Call."""
    calls = [_call("Attention_v3", T=t, NH=32, KVH=8, D=64, BN=128) for t in (1, 128, 129, 256, 287)]
    return SimpleNamespace(calls=calls + [_call("Attention_v1", T=130, NH=2, KVH=2, D=128), _call("Gemm_v1", K=64, N=8)])


def test_attention_words_count_the_blocks_after_each_rows_first():
    assert GM.attention_words(_program()) == {"Attention_v1": 2 * 2, "Attention_v3": 32 * (0 + 0 + 1 + 1 + 2)}


def test_101_counts_the_tap_lists_97_280():
    layers = 16
    prefill = [_call("Attention_v3", T=r + 1, NH=32, KVH=8, D=64, BN=128) for r in range(256)] * layers
    decode = [_call("Attention_v3", T=256 + s, NH=32, KVH=8, D=64, BN=128) for s in range(1, 32)] * layers
    assert GM.attention_words(SimpleNamespace(calls=prefill + decode)) == {"Attention_v3": 97_280}


def test_tensor_parallel_attention_is_refused_by_name():
    with pytest.raises(ValueError, match="AttentionTP2_v1.*tensor-parallel"):
        GM.attention_words(SimpleNamespace(calls=[_call("AttentionTP2_v1", T=5, C={})]))


def _query():
    return {"query_id": "Q_module_body_v1", "query_text": "q", "query_id_digest": "d", "attention_hidden_family": "fa2_hidden_m1_stream"}


def test_the_header_states_the_policy_and_a_global_header_sums_its_components():
    q = GM.header(_query(), {"Attention_v3": 5})
    assert q["guarded_max"] == {"stream_family": "fa2_hidden_m1_stream", "row_field": 3, "tap_row3": 1, "words": 5,
                                "words_by_family": {"Attention_v3": 5}}
    assert q["query_text"].endswith(GM.POLICY_TEXT) and q["query_id_digest"] != "d"
    parts = [q, GM.header(_query(), {"Attention_v3": 2, "AttentionSoftcap_v1": 4})]
    assert GM.merged(q, parts)["guarded_max"]["words_by_family"] == {"Attention_v3": 7, "AttentionSoftcap_v1": 4}
    assert GM.policy_of({"query": q}) and not GM.policy_of({"query": _query()}) and not GM.policy_of(None)


def _manifest(query):
    return {"manifest_digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "identities": [], "program_digest": "p",
            "populations": {"identities": 0}, "complete": True, "query": query}


def test_manifest_verify_tells_the_policies_apart_and_records_them_only_when_one_side_has_it():
    on, off = _manifest(GM.header(_query(), {"Attention_v3": 5})), _manifest(_query())
    assert V.verify(off, off)["ok"] and "used_guarded_max" not in V.verify(off, off)
    assert V.verify(on, on)["ok"] and V.verify(on, on)["rebuilt_guarded_max"]["words"] == 5
    for used, rebuilt in ((on, off), (off, on)):
        r = V.verify(used, rebuilt)
        assert not r["ok"] and r["why"].startswith("guarded-max policy")


def _source(row3):
    src = object.__new__(HiddenM1Source)
    src.ns_name, src.stats = "verity_fa2_matReqG" if row3 else "verity_fa2_matReq", {"row3": row3}
    return src


def test_the_source_refuses_a_build_whose_word_3_is_not_the_manifests():
    assert tap_row3(SimpleNamespace()) == 0 and tap_row3(SimpleNamespace(verity_tap_row3=lambda: 1)) == 1
    _source(1).require_row3(True)
    _source(0).require_row3(False)
    with pytest.raises(RuntimeError, match="mode 0 .*guarded-max policy and needs mode 1"):
        _source(0).require_row3(True)
    with pytest.raises(RuntimeError, match="mode 1 .*no guarded-max policy and needs mode 0"):
        _source(1).require_row3(False)
