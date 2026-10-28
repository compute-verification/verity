"""X-03 per-layer attention plane classes: the expected hidden-plane inventory of a step is DERIVED per attention layer from
the Program's attention Call statics (NH, KVH, D, window) x the step's active-sequence set; uniformity across layers is a consequence for
one-class Programs, not a rule.  Synthetic declarations and inventories only (class labels A/B; no model is named).  Offline (no CUDA)."""
from __future__ import annotations

import pytest

from verity_vllm.acquire import fa2_plane_classes as FP
from verity_vllm.acquire.fa2_plane_classes import (
    AttnLayerDecl, LayoutPolicy, StepSequences, check_step, counts_by_layer, declarations_from_attention_impls, plane_class, uniform_rule,
)

NH, KVH, D = 8, 4, 256                      # a GQA shape with a non-legacy tile: names carry `x<BN>x<D>` = x64x256
TILE = f"x{64}x{D}"
NB = 17                                      # ceil(seqlen_k / 64) of the synthetic step; the check does not read it


def decl(k: int, window: int | None) -> AttnLayerDecl:
    return AttnLayerDecl(k, NH, KVH, D, window, label=f"class-{'B' if window is not None else 'A'}")


def two_class(n_layers: int = 26, window: int = 4096) -> list[AttnLayerDecl]:
    """Class A (no window: swaps on decode) on odd layers, class B (window: never swaps) on even layers."""
    return [decl(k, window if k % 2 == 0 else None) for k in range(n_layers)]


def one_class(n_layers: int = 26, window: int | None = None) -> list[AttnLayerDecl]:
    return [decl(k, window) for k in range(n_layers)]


def whole(k: int, HB: int, M: int) -> list[str]:
    return [f"fa2.m1/L{k}/g{HB}x{M}x{NB}{TILE}"]


def per_seq(k: int, seqs: list[int], sl: int = 1, h: int = NH, planes=FP.PLANES) -> list[str]:
    return [f"fa2.m1/L{k}/s{i}/g{h}x{sl}x{NB}{TILE}/{p}" for i in seqs for p in planes]


def decode(n_seq: int) -> StepSequences:
    return StepSequences((1,) * n_seq)


def observed_pattern(decls: list[AttnLayerDecl], n_seq: int) -> list[str]:
    """The inventory the collector commits at a decode step of `n_seq` sequences for these declarations: a class-B layer (window) runs the
    un-swapped kernel -> per-sequence compaction (6 planes x n_seq, h = NH); a class-A GQA layer runs the swapped kernel -> one whole row
    (HB = n_seq*KVH, M = NH/KVH).  n_seq == 1: both classes stage one whole row (compaction is a multi-sequence rule)."""
    names: list[str] = []
    for d in decls:
        if d.window is None:
            names += whole(d.layer, n_seq * KVH, NH // KVH)
        elif n_seq >= 2:
            names += per_seq(d.layer, list(range(n_seq)))
        else:
            names += whole(d.layer, NH, 1)
    return names


# ---------------------------------------------------------------------------------------------------------------------------- (a) one class

def _both(decls, seqs, names):
    new = check_step(decls, seqs, names)["fa2_layers_bad"]
    old = uniform_rule(counts_by_layer(names), [d.layer for d in decls])
    return new, old


def test_one_class_uniform_step_passes_under_both_rules():
    decls = one_class(4)
    for n_seq in (1, 3, 8):
        names = observed_pattern(decls, n_seq)
        new, old = _both(decls, decode(n_seq), names)
        assert new == old == {}
    v = check_step(decls, decode(8), observed_pattern(decls, 8))
    assert v["ok"] and list(v["classes"]) == ["wholex1"]                     # uniformity: derived, one class


def test_one_class_failure_cases_give_the_old_rule_verdict():
    """The uniformity rule's test vectors (test_x03_identities): a layer's row dropped + another doubled, a layer outside the attention set, a zero count."""
    decls = one_class(2)
    good = observed_pattern(decls, 1)                                           # g4x2 whole rows, one per layer
    dropped_doubled = [n for n in good if "/L1/" not in n] + whole(0, KVH, NH // KVH)
    assert _both(decls, decode(1), dropped_doubled) == ({1: 0, 0: 2}, {1: 0, 0: 2})
    outside = good + whole(7, KVH, NH // KVH)
    assert _both(decls, decode(1), outside) == ({7: 1}, {7: 1})
    decls4 = one_class(4)
    nothing_for_1 = [n for n in observed_pattern(decls4, 1) if "/L1/" not in n]
    assert _both(decls4, decode(1), nothing_for_1) == ({1: 0}, {1: 0})
    # the uniformity rule's tie artifact (2 layers, counts {1, 0}: the modal count is ambiguous, the smaller wins, the GOOD layer is named too);
    # the derived rule needs no mode and names only the layer that disagrees with its own expectation
    assert _both(decls, decode(1), [n for n in good if "/L1/" not in n]) == ({1: 0}, {0: 1, 1: 0})
    # per-sequence class, one plane of one sequence dropped in layer 1
    decls_b = one_class(4, window=4096)
    good_b = observed_pattern(decls_b, 4)                                       # 24 planes per layer
    short = [n for n in good_b if n != f"fa2.m1/L1/s2/g{NH}x1x{NB}{TILE}/ROW"]
    assert _both(decls_b, decode(4), short) == ({1: 23}, {1: 23})


def test_every_layer_uniformly_wrong_is_where_the_derived_rule_is_stronger():
    """The one divergence from the uniformity rule, by design: an inventory that is uniform AND wrong (every layer 5 planes per sequence) is invisible
    to a uniformity rule and is named by the derived one."""
    decls = one_class(3, window=4096)
    names = sum((per_seq(k, [0, 1], planes=FP.PLANES[:5]) for k in range(3)), [])
    new, old = _both(decls, decode(2), names)
    assert old == {} and set(new) == {0, 1, 2}
    v = check_step(decls, decode(2), names)
    assert any("count 10 observed, 12 expected" in s for s in v["layers_bad"][0])
    assert any("carries planes" in s for s in v["layers_bad"][0])


# ------------------------------------------------------------------------------------------------------------- (b) interleaved two classes

@pytest.mark.parametrize("n_seq", [8, 7, 3])
def test_two_class_pattern_passes_when_declared_and_fails_when_one_class_is_declared(n_seq):
    """The observed pattern: even layers 6 x n_seq per-sequence planes (`s<i>/g8x1x..`), odd layers one whole row (`g32x2x..`)."""
    decls = two_class()
    names = observed_pattern(decls, n_seq)
    even = [d.layer for d in decls if d.window is not None]
    odd = [d.layer for d in decls if d.window is None]
    assert counts_by_layer(names)[0] == 6 * n_seq and counts_by_layer(names)[1] == 1
    v = check_step(decls, decode(n_seq), names)
    assert v["ok"], v["layers_bad"]
    assert sorted(v["classes"]) == [f"per_seqx{6 * n_seq}", "wholex1"]
    assert v["classes"][f"per_seqx{6 * n_seq}"] == even and v["classes"]["wholex1"] == odd
    # the uniformity rule refuses exactly this step: the tie between 13 and 13 resolves to the smaller count, the even layers are named
    assert uniform_rule(counts_by_layer(names), [d.layer for d in decls]) == {k: 6 * n_seq for k in even}
    # one class declared (all A): every even layer is named, with its class contradiction
    va = check_step(one_class(), decode(n_seq), names)
    assert not va["ok"] and sorted(va["layers_bad"]) == even and not va["undeclared_layers"]
    assert any("class per_seq observed, whole declared" in s for s in va["layers_bad"][0])
    # one class declared (all B): every odd layer is named
    vb = check_step(one_class(window=4096), decode(n_seq), names)
    assert not vb["ok"] and sorted(vb["layers_bad"]) == odd
    assert any("class whole observed, per_seq declared" in s for s in vb["layers_bad"][1])


def test_uniform_steps_of_a_two_class_program_are_a_consequence_not_a_rule():
    """Prefill (max_seqlen_q > 1: no layer swaps) and a one-sequence decode (no compaction) make both classes coincide in count."""
    decls = two_class()
    # prefill: 8 sequences x 1024 rows -> every layer per-sequence, 48 planes, h = NH, sl = 1024
    pre = StepSequences((1024,) * 8)
    names = sum((per_seq(d.layer, list(range(8)), sl=1024) for d in decls), [])
    v = check_step(decls, pre, names)
    assert v["ok"] and list(v["classes"]) == ["per_seqx48"]
    # one sequence left: both classes stage one whole row -- with DIFFERENT geometry (g8x1 for B, g4x2 for A), equal count
    one = observed_pattern(decls, 1)
    assert f"fa2.m1/L0/g{NH}x1x{NB}{TILE}" in one and f"fa2.m1/L1/g{KVH}x{NH // KVH}x{NB}{TILE}" in one
    v1 = check_step(decls, decode(1), one)
    assert v1["ok"] and list(v1["classes"]) == ["wholex1"]
    # a one-sequence step whose class-B layer committed the swapped geometry would be a class contradiction, count notwithstanding
    swapped_b = [n for n in one if "/L0/" not in n] + whole(0, KVH, NH // KVH)
    v2 = check_step(decls, decode(1), swapped_b)
    assert not v2["ok"] and list(v2["layers_bad"]) == [0] and any(f"g{KVH}x{NH // KVH} observed, g{NH}x1 expected" in s for s in v2["layers_bad"][0])


# ------------------------------------------------------------------------------------------ (c) a layer contradicting its declared class

def test_a_layer_whose_observed_class_contradicts_its_declaration_is_named():
    decls = two_class(6)
    n_seq = 4
    names = observed_pattern(decls, n_seq)
    # layer 3 is class A (whole); it committed per-sequence planes instead
    bad3 = [n for n in names if "/L3/" not in n] + per_seq(3, list(range(n_seq)))
    v = check_step(decls, decode(n_seq), bad3)
    assert not v["ok"] and list(v["layers_bad"]) == [3]
    assert any("class per_seq observed, whole declared" in s for s in v["layers_bad"][3])
    assert v["fa2_layers_bad"] == {3: 6 * n_seq}
    # layer 4 is class B (per-sequence); it committed the swapped whole row instead
    bad4 = [n for n in names if "/L4/" not in n] + whole(4, n_seq * KVH, NH // KVH)
    v = check_step(decls, decode(n_seq), bad4)
    assert list(v["layers_bad"]) == [4] and any("class whole observed, per_seq declared" in s for s in v["layers_bad"][4])
    # both at once: both named, nothing else
    v = check_step(decls, decode(n_seq), [n for n in bad3 if "/L4/" not in n] + whole(4, n_seq * KVH, NH // KVH))
    assert sorted(v["layers_bad"]) == [3, 4]


def test_geometry_within_a_class_is_checked_against_the_declaration_and_the_active_set():
    decls = two_class(2)
    n_seq = 3
    good = observed_pattern(decls, n_seq)
    # whole row with the wrong slab count (HB = 2*KVH instead of 3*KVH)
    v = check_step(decls, decode(n_seq), [n for n in good if "/L1/" not in n] + whole(1, 2 * KVH, NH // KVH))
    assert any(f"g{2 * KVH}x2 observed, g{3 * KVH}x2 expected" in s for s in v["layers_bad"][1])
    # per-sequence planes with h = KVH slabs instead of NH
    v = check_step(decls, decode(n_seq), [n for n in good if "/L0/" not in n] + per_seq(0, [0, 1, 2], h=KVH))
    assert any(f"{KVH} slabs observed, NH = {NH} expected" in s for s in v["layers_bad"][0])
    # a sequence index outside the active set, and one active sequence missing
    v = check_step(decls, decode(n_seq), [n for n in good if "/L0/" not in n] + per_seq(0, [0, 1, 5]))
    assert any("sequences [0, 1, 5] observed, [0, 1, 2] active" in s for s in v["layers_bad"][0])
    # rows per sequence vs the scheduled seqlen_q (exact set) -- a prefill of (5, 7, 2) rows
    pre = StepSequences((5, 7, 2))
    v = check_step(decls, pre, sum((per_seq(k, [0, 1, 2], sl=5) for k in range(2)), []))
    assert any("s1: 5 rows observed, seqlen_q 7 scheduled" in s for s in v["layers_bad"][0])
    # the same inventory under an inexact set (only counts known) is not compared per sequence
    assert check_step(decls, StepSequences.from_counts(3, 14), sum((per_seq(k, [0, 1, 2], sl=5) for k in range(2)), []))["ok"]
    # a head_dim in the name that is not the declared one
    v = check_step(decls, decode(n_seq), [n for n in good if "/L1/" not in n] + [f"fa2.m1/L1/g{n_seq * KVH}x{NH // KVH}x{NB}x64x128"])
    assert any("head_dim 128 in the name, 256 declared" in s for s in v["layers_bad"][1])


# ----------------------------------------------------------------------------------------------------------------- padding, fa3, inputs

def test_padding_planes_are_counted_apart_and_a_padding_only_step_expects_nothing():
    decls = two_class(2)
    active = observed_pattern(decls, 3)
    pad = [f"fa2.m1/L{k}/r{r}/g{NH}x1x{NB}{TILE}/bot" for k in range(2) for r in (5, 6, 7)]
    v = check_step(decls, decode(3), active + pad)
    assert v["ok"] and v["padding_planes"] == {0: 3, 1: 3}
    v0 = check_step(decls, StepSequences(()), pad)
    assert v0["ok"] and v0["expected"][0]["class"] == FP.NONE and v0["padding_planes"] == {0: 3, 1: 3}
    # executed planes at a step with no active sequence are a violation (something launched that the schedule did not)
    assert not check_step(decls, StepSequences(()), pad + whole(1, KVH, 2))["ok"]


def test_fa3_policy_stages_one_whole_launch_per_layer_whatever_the_window():
    decls = two_class(4)
    pol = LayoutPolicy(backend="fa3", compact=False)
    for d in decls:
        c = plane_class(d, decode(8), pol)
        assert (c.kind, c.count, c.slabs, c.rows) == (FP.WHOLE, 1, 8 * KVH, NH // KVH)
    names = [f"fa3.m1/L{k}/g{8 * KVH}x{NH // KVH}x9x128x{D}" for k in range(4)]
    assert check_step(decls, decode(8), names, pol)["ok"]
    assert not check_step(decls, decode(8), names[:-1], pol)["ok"]


def test_compaction_off_stages_whole_padded_launches():
    decls = two_class(2)
    pol = LayoutPolicy(backend="fa2", compact=False)
    a, b = plane_class(decls[1], decode(4), pol), plane_class(decls[0], decode(4), pol)
    assert (a.kind, a.count, a.slabs, a.rows) == (FP.WHOLE, 1, 4 * KVH, NH // KVH)            # class A still swaps
    assert (b.kind, b.count, b.slabs, b.rows) == (FP.WHOLE, 1, 4 * NH, 1)                      # class B: b*NH slabs of one row, whole


def test_mha_layer_never_swaps_and_a_windowed_layer_never_swaps():
    mha = AttnLayerDecl(0, 8, 8, 64, None)
    assert not mha.swaps_on_decode() and plane_class(mha, decode(2)).kind == FP.PER_SEQ and plane_class(mha, decode(1)).kind == FP.WHOLE
    assert plane_class(mha, decode(1)).slabs == 8 and plane_class(mha, decode(1)).rows == 1
    assert not decl(0, 4096).swaps_on_decode() and decl(1, None).swaps_on_decode()
    assert not plane_class(decl(1, None), StepSequences((1, 3))).swapped                       # a mixed step: max_seqlen_q > 1, no fold


def test_step_sequences_from_rows_and_from_counts():
    s = StepSequences.from_rows([("r0", 0, 5), ("r1", 5, 6), ("r2", 6, 6), ("r3", 6, 7)])
    assert s.seqlen_q == (5, 1, 0, 1) and s.active == (0, 1, 3) and s.n_seq == 4 and s.max_seqlen_q == 5 and s.exact
    d = StepSequences.from_counts(8, 8)
    assert d.seqlen_q == (1,) * 8 and d.exact
    p = StepSequences.from_counts(3, 100)
    assert p.n_seq == 3 and p.max_seqlen_q == 98 and not p.exact
    assert StepSequences.from_counts(0, 0).n_seq == 0


def test_declarations_from_the_build_record_use_only_the_layer_statics():
    impls = {
        "model.layers.0.self_attn.attn": {"num_heads": 8, "num_kv_heads": 4, "head_size": 256, "sliding_window": [4095, 0]},
        "model.layers.1.self_attn.attn": {"num_heads": 8, "num_kv_heads": 4, "head_size": 256, "sliding_window": None},
        "model.layers.2.self_attn.attn": {"num_heads": 8, "num_kv_heads": 4, "head_size": 256, "sliding_window": [-1, -1]},
        "model.layers.3.self_attn.attn": {"num_heads": 8, "num_kv_heads": 4, "head_size": 256, "sliding_window": 2047},
        "model.layers.4.self_attn.attn": {"num_heads": 8, "num_kv_heads": 4, "head_size": 256, "sliding_window": -1},
        "model.layers.5.self_attn.attn": {"num_heads": 8, "num_kv_heads": None, "head_size": 256},        # incomplete: skipped, not guessed
        "model.norm": {"num_heads": 8, "num_kv_heads": 4, "head_size": 256},                            # not a layer: skipped
    }
    ds = declarations_from_attention_impls(impls)
    assert [d.layer for d in ds] == [0, 1, 2, 3, 4]
    assert [d.window for d in ds] == [4095, None, None, 2047, None]
    assert all((d.NH, d.KVH, d.D) == (8, 4, 256) for d in ds)
    assert [d.swaps_on_decode() for d in ds] == [False, True, True, False, True]


def test_attention_impls_of_record_reads_the_build_result(tmp_path):
    import json

    (tmp_path / "build_request_LP4_T2").mkdir()
    (tmp_path / "build_request_LP4_T2" / "result.json").write_text(json.dumps({"attention_impls_observed": {
        "model.layers.0.self_attn.attn": {"num_heads": 4, "num_kv_heads": 2, "head_size": 64, "sliding_window": None}}}))
    obs, p = FP.attention_impls_of_record(programs_root=str(tmp_path))
    assert obs and p.endswith("result.json")
    assert FP.attention_impls_of_record(program_dir=str(tmp_path / "nope")) == (None, None)
    (tmp_path / "result.json").write_text("{}")
    assert FP.attention_impls_of_record(program_dir=str(tmp_path)) == (None, None)


# ------------------------------------------------------------------------------------------------------------------ committer wiring

torch = pytest.importorskip("torch")

from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, NativeHostCommitter, TensorMeta  # noqa: E402


class Attention:  # class names are what `_mro_names` sees
    pass


class LinearBase:
    pass


class Sampler:
    pass


class _W:
    compact = True


class _Src:
    short_name = "hidden_m1"
    _w = _W()


def _com(n_layers: int) -> NativeHostCommitter:
    com = NativeHostCommitter(None, run_id="x03-classes", gpu_tree=True)
    mods = []
    for k in range(n_layers):
        mods += [(f"model.layers.{k}.self_attn.qkv_proj", LinearBase()), (f"model.layers.{k}.self_attn.attn", Attention())]
    mods += [("model.embed_tokens", LinearBase()), ("runner.sampler", Sampler())]
    com._modules = mods
    com.sources = [_Src()]
    return com


def _layout(names: list[str], n_seq: int, tokens: int) -> list[TensorMeta]:
    out, off = [], 0
    for i, n in enumerate(names):
        m = TensorMeta(i, n, "bfloat16", (4, 8), 64, GPU_CHUNK_BYTES, i, 1)
        m.dev_off = m.stream_off = off
        off += 64
        out.append(m)
    ids = TensorMeta(len(out), "model.embed_tokens/in", "int64", (tokens,), 8 * tokens, GPU_CHUNK_BYTES, len(out), 1)
    ids.dev_off = ids.stream_off = off
    out.append(ids)
    smp = TensorMeta(len(out), "runner.sampler/sampled_token_ids", "int64", (n_seq, 1), 8 * n_seq, GPU_CHUNK_BYTES, len(out), 1)
    smp.dev_off = smp.stream_off = off + 8 * tokens
    out.append(smp)
    return out


def _step(decls, n_seq: int) -> list[str]:
    names = []
    for d in decls:
        names += [f"model.layers.{d.layer}.self_attn.qkv_proj", f"model.layers.{d.layer}.self_attn.attn"]
    return names + observed_pattern(decls, n_seq)


def test_committer_derives_the_per_layer_expectation_when_declarations_are_handed_over():
    decls = two_class(4)
    com = _com(4)
    for s, n in enumerate((8, 7, 3, 1)):
        com._layouts[s] = _layout(_step(decls, n), n_seq=n, tokens=n)
    r0 = com.identity_check()                                             # no declarations: the uniformity rule fails the two-class steps
    assert r0["ok"] is False and r0["violations_n"] == 3 and r0["fa2_rule"]["steps"] == {"uniform": 4}
    assert r0["violations"][0]["fa2_layers_bad"] == {0: 48, 2: 48}
    com.attention_layer_decls = decls
    r1 = com.identity_check()
    assert r1["ok"] is True, r1["violations"]
    assert r1["fa2_rule"]["steps"] == {"per-layer": 4} and r1["fa2_rule"]["attention_decls"] == 4
    assert r1["fa2_rule"]["classes_declared"] == [("global-gqa", 2), ("windowed-gqa", 2)]
    assert r1["fa2_rule"]["policy"] == {"backend": "fa2", "compact": True}


def test_committer_reads_the_active_set_from_step_rows_first_and_names_the_layer():
    decls = two_class(4)
    com = _com(4)
    com.attention_layer_decls = decls
    com.step_rows = {0: [(f"r{i}", i, i + 1) for i in range(8)]}         # the binding record: 8 active sequences, one row each
    names = _step(decls, 8)
    names = [n for n in names if "/L2/" not in n] + whole(2, 8 * KVH, NH // KVH)   # layer 2 (class B) committed class A's row
    com._layouts[0] = _layout(names, n_seq=8, tokens=8)
    r = com.identity_check()
    assert r["ok"] is False and r["violations"][0]["fa2_layers_bad"] == {2: 1}
    det = r["violations"][0]["fa2_detail"]
    assert det["n_seq"] == 8 and det["max_seqlen_q"] == 1 and det["exact_seqlens"] is True
    assert any("class whole observed, per_seq declared" in s for s in det["layers_bad"][2])
    # step_rows decide the active set (the record, not the layout): 9 rows declared -> class B layers miss s8, class A rows read
    # g32x2 where 9 sequences imply g36x2 -- every layer is named, each with its own reason
    com.step_rows = {0: [(f"r{i}", i, i + 1) for i in range(9)]}
    r = com.identity_check()
    assert set(r["violations"][0]["fa2_layers_bad"]) == {0, 1, 2, 3}
    lb = r["violations"][0]["fa2_detail"]["layers_bad"]
    assert any("[0, 1, 2, 3, 4, 5, 6, 7] observed, [0, 1, 2, 3, 4, 5, 6, 7, 8] active" in s for s in lb[0])
    assert any(f"g{8 * KVH}x2 observed, g{9 * KVH}x2 expected" in s for s in lb[1])


def test_committer_falls_back_to_the_uniform_rule_for_a_step_without_an_active_set():
    decls = two_class(2)
    com = _com(2)
    com.attention_layer_decls = decls
    metas = _layout(_step(decls, 3), n_seq=3, tokens=3)
    com._layouts[0] = [m for m in metas if m.name != "runner.sampler/sampled_token_ids"]   # no sampler ids: n_seq unknown
    r = com.identity_check()
    assert r["fa2_rule"]["steps"] == {"uniform": 1}
    assert r["ok"] is False and r["violations"][0]["fa2_layers_bad"] == {0: 18}           # the uniformity rule's verdict (and the sampler-ids miss)


def test_existing_single_class_vectors_are_unchanged_under_declarations():
    """test_x03_identities' vectors (one whole row per layer, `g1x64x1`) under one-class MHA declarations: same verdicts."""
    decls = [AttnLayerDecl(k, 1, 1, 64, None) for k in range(2)]
    com = _com(2)
    com.attention_layer_decls = decls

    def good():
        names = []
        for k in range(2):
            names += [f"model.layers.{k}.self_attn.qkv_proj", f"model.layers.{k}.self_attn.attn", f"fa2.m1/L{k}/g1x64x1"]
        return names

    com._layouts[0] = _layout(good(), n_seq=1, tokens=64)
    com._layouts[1] = _layout([n for n in good() if n != "fa2.m1/L1/g1x64x1"] + ["fa2.m1/L0/g1x64x1"], n_seq=1, tokens=64)
    r = com.identity_check()
    assert r["ok"] is False and r["violations"][0]["step"] == 1 and r["violations"][0]["fa2_layers_bad"] == {1: 0, 0: 2}
    assert r["fa2_rule"]["steps"] == {"per-layer": 2}
