"""R15 tp TP-04: the required set of a tensor-parallel rank Program (`derive_step --tp W --rank r`), interface ACKed by manbind (board
M-0016 / M-0033 / M-0052) -- the CONSERVATIVE rule, per rank r:

  (a) every collective's rank-local INPUT is required: family `tp_rank_partials`, op_path = the issuing module, output_member
      `part_rank<r>` (row-parallel Gemm partial on the K/W shard; masked EmbeddingShard lookup) or `shard_rank<r>` (the lm_head logits
      shard before its all-gather), range [0, rows*N_r), rank = r;
  (b) the collective OUTPUT (the AllReduce2 / AllGather2 row) is required as the module's ordinary output member (`0` for a linear, `out`
      otherwise), width N (W*N for the gather), family instance_outputs, rank = r on EVERY rank's manifest (each rank commits its copy);
  (c) EmbeddingShard -> `part_rank<r>`[H] + the `in` ids row;
  (d) build_manifest(rank=r, ranks=W) refuses a Program whose result.tp != (W, r); a TP1 Program's manifest is unchanged;
  (e) the row manifest = merge_manifests([rank0, rank1]) with program_digest_per_rank; ranks are never folded (a missing or duplicated
      rank part is refused), and the coverage check of rank r is over rank r's rows only.

Negatives (brief §4 TP items): omit rank 1's partials -> coverage FAIL naming rank 1 with the whole family missing; relabel rank 0's
partials as rank 1's -> rank 0 FAIL (its part_rank0 rows unbound; the mislabelled copies do not count); rank 1's rows against rank 0's
layouts -> FAIL; no tp partial source at all (the R13 collective-outputs-only collector) -> family tp_rank_partials missing entirely.
Synthetic 2-layer rank Programs in the derive_step vocabulary (no torch, no pod)."""
from __future__ import annotations

import copy

import pytest

from verity_capture.commit.required_manifest import build_manifest, coverage_check, merge_manifests
from verity_capture.commit.tests.test_required_manifest import EPS, _layouts_from, _program

H, V, W = 128, 64, 2


def _tp_program(rank: int, *, LP: int = 4, T: int = 2, layers: int = 2, NH: int = 2, KVH: int = 1, D: int = 64) -> dict:
    """A TP2 rank Program: the TP1 synthetic Program with the row-parallel linears on K/W shards followed by AllReduce2 rows, the embedding
    as a masked EmbeddingShard + AllReduce2, the logits Gemm on V/W columns + AllGather2 (the derive_step --tp 2 --rank r vocabulary:
    `<module>/all_reduce2_k[i]/all_reduce`, `<module>/embedding_shard_k[i]/shard`, `<module>/all_gather2_k[i]/all_gather`)."""
    prog = _program(LP=LP, T=T, layers=layers, NH=NH, KVH=KVH, D=D, H=H, V=V)
    rows_in = prog["instances"]["rows"]
    rows: list[dict] = []
    k = [0]

    def add(spec: str, name: str) -> None:
        rows.append({"i": len(rows), "spec": spec, "form": "call", "name": name, "args": []})

    # walk the TP1 rows; rewrite the collective-issuing modules; after the last row of a module call insert its collective rows -- ONE
    # AllReduce2 row per token row (derive_step emits the collective per row, like every other per-row instance)
    pending: tuple[str, str, int] | None = None          # (spec, name prefix, rows)
    prev_mod = None

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            spec, pfx, n = pending
            for i in range(n):
                add(spec, f"{pfx}[{i}]/all_reduce")
            pending = None

    for r in rows_in:
        spec, name = r["spec"], r["name"]
        fam = spec.split("{")[0]
        mod = name.split("/")[0]
        if pending is not None and mod != prev_mod:
            flush()
            k[0] += 1
        prev_mod = mod
        if fam == "Embedding_v1":
            add(f"EmbeddingShard_v1{{VS={V // W},H={H},START={rank * (V // W)}}}", f"{mod}/embedding_shard_{k[0]}[{name.split('[')[1]}/shard")
            pending = (f"AllReduce2_v1{{N={H}}}", f"{mod}/all_reduce2_{k[0]}", (pending[2] + 1) if pending else 1)
            continue
        if fam == "Gemm_v1" and (mod.endswith(".o_proj") or mod.endswith(".down_proj")):
            K = int(spec.split("K=")[1].split(",")[0])
            N = int(spec.split("N=")[1].split("}")[0])
            add(f"Gemm_v1{{K={K // W},N={N}}}", name)
            pending = (f"AllReduce2_v1{{N={N}}}", f"{mod}/all_reduce2_{k[0]}", (pending[2] + 1) if pending else 1)
            continue
        if fam == "Gemm_v1" and mod.endswith("logits_processor"):
            K = int(spec.split("K=")[1].split(",")[0])
            add(f"Gemm_v1{{K={K},N={V // W}}}", name)
            add(f"AllGather2_v1{{N={V // W}}}", f"{mod}/all_gather2_{k[0]}[0]/all_gather")
            k[0] += 1
            continue
        add(spec, name)
    flush()
    prog["instances"]["rows"] = rows
    prog["result"]["tp"] = {"world": W, "rank": rank, "engine_args_required": {"tensor_parallel_size": W, "disable_custom_all_reduce": True}}
    prog["result"]["program"]["digest"] = (f"{rank}" * 64)
    prog["artifact"]["program_digest"] = f"{rank}" * 64
    return prog


def _man(rank: int) -> dict:
    return build_manifest(_tp_program(rank), {"sweep": {"row_id": "tp2-test", "batch": 1, "arrive_steps": [0]}}, rank=rank, ranks=W)


def _layouts(man: dict, rank: int) -> dict[int, list[dict]]:
    """The layouts of rank r's collector (its own rows only: every rank commits its own tree)."""
    sub = dict(man, identities=[r for r in man["identities"] if int(r["rank"]) == rank])
    return _layouts_from(sub)


def test_tp2_rank_manifest_requires_partials_and_collective_outputs():
    m0 = _man(0)
    assert m0["complete"] is True and m0["unmodelled"] == {}
    assert m0["query"]["tp"] == {"world": 2, "rank": 0, "rule": "TP-04: partials AND collective outputs both required, per rank"}
    assert "tp_rank_partials" in m0["query"]["required_families"]
    assert all(int(r["rank"]) == 0 for r in m0["identities"])
    fam = m0["populations"]["by_family"]
    steps = len(m0["steps"])
    layers = 2
    # (a) partials: per step o_proj + down_proj per layer, the embedding shard, the logits shard
    assert fam["tp_rank_partials"] == (2 * layers + 2) * steps
    s0 = {(r["op_path"], r["output_member"]): r for r in m0["identities"] if r["step"] == 0 and r["family"] == "tp_rank_partials"}
    assert s0[("model.layers.0.self_attn.o_proj", "part_rank0")]["element_range"] == [0, 4 * H] and s0[("model.layers.0.self_attn.o_proj", "part_rank0")]["spec"] == "Gemm_v1"
    assert s0[("model.embed_tokens", "part_rank0")]["element_range"] == [0, 4 * H] and s0[("model.embed_tokens", "part_rank0")]["spec"] == "EmbeddingShard_v1"
    assert s0[("logits_processor", "shard_rank0")]["element_range"] == [0, V // W]
    # (b) the collective outputs are the module's ordinary output member, full width, family instance_outputs
    outs = {(r["op_path"], r["output_member"]): r for r in m0["identities"] if r["step"] == 0 and r["family"] == "instance_outputs" and r["spec"].startswith("All")}
    assert outs[("model.layers.0.self_attn.o_proj", "0")]["element_range"] == [0, 4 * H] and outs[("model.layers.0.self_attn.o_proj", "0")]["spec"] == "AllReduce2_v1"
    assert outs[("model.layers.1.mlp.down_proj", "0")]["spec"] == "AllReduce2_v1"
    assert outs[("model.embed_tokens", "out")]["element_range"] == [0, 4 * H]
    assert outs[("logits_processor", "out")]["element_range"] == [0, V] and outs[("logits_processor", "out")]["spec"] == "AllGather2_v1"
    # (c) the embedding ids row stays
    assert any(r["family"] == "input_token_ids" and r["op_path"] == "model.embed_tokens" and r["step"] == 0 and r["element_range"] == [0, 4] for r in m0["identities"])
    # the non-collective modules are unchanged relative to the TP1 rule (qkv_proj, gate_up_proj, norms, rope, attention hidden, sampler)
    assert fam["sampled_token_ids"] == steps and fam["fa2_hidden_m1_stream"] == layers * steps
    # rank 1: same population shape, rank=1 everywhere, its own Program digest, different manifest digest
    m1 = _man(1)
    assert all(int(r["rank"]) == 1 for r in m1["identities"]) and m1["populations"]["by_family"] == fam
    assert m1["program_digest"] != m0["program_digest"] and m1["manifest_digest"] != m0["manifest_digest"]
    assert {r["output_member"] for r in m1["identities"] if r["family"] == "tp_rank_partials"} == {"part_rank1", "shard_rank1"}


def test_tp2_collective_without_a_width_is_unmodelled_never_a_zero_width_row():
    """manbind M-0247: AllReduce2 / AllGather2 widths come from the spec statics (N); a collective row whose spec carries no N must make
    the manifest INCOMPLETE (named 'width 0' unmodelled entry -> coverage_check refuses), never a silent [0, 0) identity."""
    for fam, old in (("AllReduce2_v1", "AllReduce2_v1{"), ("AllGather2_v1", "AllGather2_v1{")):
        prog = _tp_program(1)
        hit = 0
        for r in prog["instances"]["rows"]:
            if r["spec"].startswith(old):
                r["spec"] = fam + "{}"                                                 # statics without N
                hit += 1
        assert hit > 0
        man = build_manifest(prog, {"sweep": {"row_id": "tp2-test", "batch": 1, "arrive_steps": [0]}}, rank=1, ranks=W)
        assert man["complete"] is False
        assert any(k.startswith(fam) and "width 0" in k for k in man["unmodelled"]), man["unmodelled"]
        assert all(int(i["element_range"][1]) > int(i["element_range"][0]) for i in man["identities"]
                   if i["element_range"][1] is not None)                              # no zero-width row was written
        assert not any(str(i.get("spec") or "").startswith(fam) for i in man["identities"])
        res = coverage_check(man, _layouts(man, 1), rank=1)
        assert res["ok"] is False and "manifest incomplete" in res["error"] and fam in res["error"]


def test_tp2_moe_block_partial_is_required_under_the_experts_module():
    """TP-06 (moe M-0249): a tensor-parallel fused-MoE layer -- the rank Program's MoE rows sit under `model.layers.<L>.mlp.experts` and the
    block's all-reduce is issued by that module (MoERunner).  Required per rank: the block stream planes (the rank's own interior), the
    rank-local partial `part_rank<r>` = the MoeSum output (width H per routed row) as the all-reduce INPUT, and the received all-reduced
    module output `out` (width H).  Nothing else changes for the dense layers."""
    prog = _tp_program(1, layers=2)
    rows = prog["instances"]["rows"]
    I, E, TOPK = 96, 8, 2
    out: list[dict] = []
    seen: dict[str, int] = {}
    for r in rows:
        name = r["name"]
        mod = name.split("/")[0]
        if mod.startswith("model.model.layers.1.mlp."):
            # drop the dense MLP of layer 1 (gate_up / act / down + its all-reduce rows); emit the MoE block once per token row (keyed by the
            # gate_up row so the block sits at the MLP's position in the stream)
            if mod.endswith("gate_up_proj"):
                sfx = name.split("[")[1].rstrip("]")
                step_tag = name.split("/")[1]
                mx = "model.model.layers.1.mlp.experts"
                k = seen.get(step_tag, 0)
                seen[step_tag] = k + 1
                out.append({"i": 0, "spec": f"MoeRouterTopK_v1{{E={E},TOPK={TOPK}}}", "form": "call", "name": f"{mx}/router_{step_tag}[{sfx}]", "args": []})
                for t in range(TOPK):
                    out.append({"i": 0, "spec": f"MoeExpertGemm_v1{{K={H},N={2 * I}}}", "form": "call", "name": f"{mx}/up_{step_tag}_{t}[{sfx}]", "args": []})
                    out.append({"i": 0, "spec": f"SiluMul_v1{{I={I}}}", "form": "call", "name": f"{mx}/act_{step_tag}_{t}[{sfx}]", "args": []})
                    out.append({"i": 0, "spec": f"MoeExpertGemmW_v1{{K={I},N={H}}}", "form": "call", "name": f"{mx}/down_{step_tag}_{t}[{sfx}]", "args": []})
                out.append({"i": 0, "spec": f"MoeSum_v1{{H={H},TOPK={TOPK}}}", "form": "call", "name": f"{mx}/moe_sum_{step_tag}[{sfx}]", "args": []})
                out.append({"i": 0, "spec": f"AllReduce2_v1{{N={H}}}", "form": "call", "name": f"{mx}/all_reduce2_{step_tag}[{sfx}]/all_reduce", "args": []})
            continue
        out.append(r)
    for i, r in enumerate(out):
        r["i"] = i
    prog["instances"]["rows"] = out
    man = build_manifest(prog, {"sweep": {"row_id": "tp2-moe-test", "batch": 1, "arrive_steps": [0]}}, rank=1, ranks=W)
    assert man["complete"] is True, man["unmodelled"]
    ex = [i for i in man["identities"] if i["op_path"] == "model.layers.1.mlp.experts"]
    assert ex, "no identities under the experts module"
    parts = [i for i in ex if i["output_member"] == "part_rank1"]
    outs = [i for i in ex if i["output_member"] == "out"]
    assert parts and outs
    assert all(i["family"] == "tp_rank_partials" and i["rank"] == 1 and i["width"] == H for i in parts)
    assert all(i["family"] == "instance_outputs" and i["rank"] == 1 and i["width"] == H and i["spec"] == "AllReduce2_v1" for i in outs)
    # one partial identity and one received output per step, both over that step's rows
    assert {i["step"] for i in parts} == {i["step"] for i in outs} == set(range(len(man["steps"])))
    for p in parts:
        o = next(i for i in outs if i["step"] == p["step"])
        assert p["element_range"] == o["element_range"] == [0, p["rows"] * H]
    # the block stream planes of layer 1 are still required (the rank's own expert shards), the dense layer 0 partial is unchanged
    assert any(i["family"] == "moe_block_stream" and i["op_path"] == "moe/L1" for i in man["identities"])
    assert any(i["op_path"] == "model.layers.0.mlp.down_proj" and i["output_member"] == "part_rank1" for i in man["identities"])
    # no other rank's partial, and never at TP1 (the MoeSum rule fires only under an all-reducing module)
    assert not any(i["output_member"] == "part_rank0" for i in man["identities"])


def test_tp2_moe_partial_with_h_absent_fails_closed_not_zero_width():
    """rev-cb M-0359 T6-2: a MoeSum row whose statics lack H (a Program whose fused-MoE spec carries no hidden width) must NOT yield a
    zero-width `part_rank<r>` identity: the manifest is incomplete (MAN-02) and coverage_check refuses it."""
    from verity_capture.commit.required_manifest import coverage_check

    prog = _tp_program(1, layers=2)
    mx = "model.model.layers.1.mlp.experts"
    rows = prog["instances"]["rows"]
    out = []
    for r in rows:
        if r["name"].split("/")[0] == "model.model.layers.1.mlp.down_proj":
            sfx, step_tag = r["name"].split("[")[1].rstrip("]"), r["name"].split("/")[1]
            out.append({"i": 0, "spec": "MoeSum_v1{TOPK=2}", "form": "call", "name": f"{mx}/moe_sum_{step_tag}[{sfx}]", "args": []})
            out.append({"i": 0, "spec": f"AllReduce2_v1{{N={H}}}", "form": "call", "name": f"{mx}/all_reduce2_{step_tag}[{sfx}]/all_reduce", "args": []})
            continue
        out.append(r)
    for i, r in enumerate(out):
        r["i"] = i
    prog["instances"]["rows"] = out
    man = build_manifest(prog, {"sweep": {"row_id": "tp2-moe-noH", "batch": 1, "arrive_steps": [0]}}, rank=1, ranks=W)
    assert man["complete"] is False
    assert any(k.startswith("MoeSum_v1 under model.layers.1.mlp.experts (part_rank1: width 0)") for k in man["unmodelled"]), man["unmodelled"]
    assert not any(i["op_path"] == "model.layers.1.mlp.experts" and i["output_member"] == "part_rank1" for i in man["identities"])
    assert not any(i["element_range"] == [0, 0] for i in man["identities"])
    chk = coverage_check(man, {}, rank=1)
    assert chk["ok"] is False and "manifest incomplete" in chk["error"]


def test_tp2_rank_or_world_mismatch_is_refused_and_tp1_unchanged():
    p0 = _tp_program(0)
    with pytest.raises(ValueError, match="rank Program mismatch"):
        build_manifest(p0, rank=1, ranks=W)                  # a rank-0 Program labelled rank 1
    with pytest.raises(ValueError, match="rank Program mismatch"):
        build_manifest(p0, rank=0, ranks=1)                  # a TP2 Program read as TP1 (world folded)
    with pytest.raises(ValueError, match="rank Program mismatch"):
        build_manifest(_program(), rank=0, ranks=W)          # a TP1 Program asked for a TP2 manifest
    # TP1 Programs: no tp family, no tp query field, the same digest as before this rule
    t1 = build_manifest(_program(), {"sweep": {"row_id": "t", "batch": 1, "arrive_steps": [0]}})
    assert "tp" not in t1["query"] and "tp_rank_partials" not in t1["query"]["required_families"]
    assert t1["populations"]["by_family"].get("tp_rank_partials", 0) == 0
    assert all(int(r["rank"]) == 0 for r in t1["identities"])


def test_tp2_merge_never_folds_ranks():
    m0, m1 = _man(0), _man(1)
    row = merge_manifests([m0, m1])
    assert row["ranks"] == 2 and row["program_digest_per_rank"] == {"0": m0["program_digest"], "1": m1["program_digest"]}
    assert row["populations"]["identities"] == m0["populations"]["identities"] + m1["populations"]["identities"]
    assert row["populations"]["by_family"]["weights"] == 2 * m0["populations"]["by_family"]["weights"]        # each rank registers its shards
    assert row["requests"] == ["r0"] and row["complete"] is True
    assert sorted({int(r["rank"]) for r in row["identities"]}) == [0, 1]
    # header program_digest is ONE string (manbind's loader contract), labelled as the fold of the per-rank Programs; rows keep their rank's
    assert isinstance(row["program_digest"], str) and len(row["program_digest"]) == 64 and row["program_digest_kind"] == "tp-ranks-fold"
    assert row["component_digests"] == sorted({m0["program_digest"], m1["program_digest"]}) and row["program_digest"] not in row["component_digests"]
    assert {r["program_digest"] for r in row["identities"] if int(r["rank"]) == 1} == {m1["program_digest"]}
    # [revb F-r17b-38] the ranks' shared-module alias table TRAVELS with the merge: every invocation >= 1 identity of the row is the image
    # of an alias entry (the TP2 B1 row of record on r17-int-2 lost the table here -> 960 + 960 unreachable on every rank)
    assert m0["op_path_aliases"] and row["op_path_aliases"] == m0["op_path_aliases"] == m1["op_path_aliases"]
    images = {(c, int(k)) for c, k in row["op_path_aliases"].values()}
    shared = [r for r in row["identities"] if int(r.get("invocation", 0) or 0) >= 1]
    assert shared and all((r["op_path"], int(r["invocation"])) in images for r in shared)
    with pytest.raises(ValueError, match="every rank"):
        merge_manifests([m0])                                # rank 1 missing
    with pytest.raises(ValueError, match="every rank"):
        merge_manifests([m0, copy.deepcopy(m0)])             # rank 0 twice, rank 1 missing
    with pytest.raises(ValueError, match="appears twice"):   # [rev-cb M-0214 T1] every rank present, rank 0 twice: one part per rank, refused
        merge_manifests([m0, copy.deepcopy(m0), m1])
    with pytest.raises(ValueError, match="appears twice"):   # ... also when the duplicate is a DIFFERENT rank-0 Program (its digest must not enter the fold)
        other0 = dict(copy.deepcopy(m0), program_digest="f" * 64)
        merge_manifests([m0, other0, m1])
    # an incomplete part makes the row manifest incomplete
    bad = copy.deepcopy(m1)
    bad["complete"], bad["unmodelled"] = False, {"Foo_v1 under x": 3}
    assert merge_manifests([m0, bad])["complete"] is False
    # [rev-pm M-0339 T1] every part must declare the same world: a TP1 part hand-labelled `rank 1` (ranks=1) is not rank 1 of a world-2 row
    with pytest.raises(ValueError, match="different worlds"):
        merge_manifests([m0, dict(copy.deepcopy(m1), ranks=1)])
    # [rev-pm M-0339 T2] the rank is a structural input of a rank Program: two ranks with ONE digest = one rank Program labelled twice
    m1c = copy.deepcopy(m1)
    m1c["program_digest"] = m0["program_digest"]
    for r in m1c["identities"]:
        r["program_digest"] = m0["program_digest"]
    with pytest.raises(ValueError, match="same Program digest"):
        merge_manifests([m0, m1c])


def test_tp2_coverage_per_rank_positive_and_negatives():
    m0, m1 = _man(0), _man(1)
    row = merge_manifests([m0, m1])
    lay0, lay1 = _layouts(row, 0), _layouts(row, 1)
    c0, c1 = coverage_check(row, lay0, rank=0), coverage_check(row, lay1, rank=1)
    assert c0["ok"] and c1["ok"], (c0["missing"][:2], c1["missing"][:2])
    assert c0["checked"] == m0["populations"]["identities"] - m0["populations"]["by_family"]["weights"]
    assert c0["checked_by_family"]["tp_rank_partials"] == m0["populations"]["by_family"]["tp_rank_partials"]

    # NEG 1: omit rank 1's partials (collective outputs only on rank 1) -> FAIL naming rank 1, family missing entirely; rank 0 unaffected
    lay1_np = {s: [m for m in ms if "_rank1" not in m["name"]] for s, ms in lay1.items()}
    c = coverage_check(row, lay1_np, rank=1)
    assert not c["ok"] and c["families_missing_entirely"] == ["tp_rank_partials"]
    assert c["missing_n"] == m1["populations"]["by_family"]["tp_rank_partials"] and all(m["output_member"].endswith("_rank1") for m in c["missing"])
    assert coverage_check(row, lay0, rank=0)["ok"]

    # NEG 2: relabel rank 0's partials as rank 1's -> rank 0 FAIL (part_rank0 unbound; the part_rank1-named copies are unmatched extras)
    lay0_rl = {s: [dict(m, name=m["name"].replace("_rank0", "_rank1")) for m in ms] for s, ms in lay0.items()}
    c = coverage_check(row, lay0_rl, rank=0)
    assert not c["ok"] and c["families_missing_entirely"] == ["tp_rank_partials"]
    assert c["committed_unmatched_n"] == m0["populations"]["by_family"]["tp_rank_partials"]

    # NEG 3: rank 1's rows against rank 0's layouts (ranks folded / swapped) -> FAIL on every partial
    c = coverage_check(row, lay0, rank=1)
    assert not c["ok"] and c["missing_by_family"]["tp_rank_partials"] == m1["populations"]["by_family"]["tp_rank_partials"]

    # NEG 4: drop one collective OUTPUT (the all-reduced o_proj of layer 1, step 1) -> FAIL naming that identity (both are required)
    lay0_no = {s: [m for m in ms if not (s == 1 and m["name"] == "model.layers.1.self_attn.o_proj/0")] for s, ms in lay0.items()}
    c = coverage_check(row, lay0_no, rank=0)
    assert not c["ok"] and c["missing_n"] == 1 and c["missing"][0]["op_path"] == "model.layers.1.self_attn.o_proj" and c["missing"][0]["output_member"] == "0"

    # NEG 5: a partial of the wrong width (the OTHER rank's shard width would still be N; a half-width copy) -> FAIL
    lay0_hw = {s: [dict(m, numel=m["numel"] // 2) if m["name"] == "model.layers.0.mlp.down_proj/part_rank0" else m for m in ms] for s, ms in lay0.items()}
    c = coverage_check(row, lay0_hw, rank=0)
    assert not c["ok"] and all(m["op_path"] == "model.layers.0.mlp.down_proj" for m in c["missing"])


def _olmoe_tp_program(rank: int, *, site_names: bool, Q: int = 64, **kw) -> dict:
    """[R19 moe, row #70] the OLMoE TP2 attention: `OlmoeAttention._apply_qk_norm` all-gathers q AND k (two `all_gather2` calls, N = Q/W
    columns each) in the MIDDLE of `self_attn` -- between the qkv_proj Gemm (whose output is the module's ordinary `0`, not a partial: the
    gathered value is a q/k SLICE of it) and the q_norm/k_norm rows.  `site_names=True` is the frontend vocabulary of the fix (the row's
    module is the call SITE `<self_attn>.<fx node>`); False is the R17 vocabulary (both calls under the bare `self_attn`), which the
    manifest folded into ONE `self_attn/out` identity of width 2 x W x N per token with no partial on either rank (6,944 unbound per rank)."""
    prog = _tp_program(rank, **kw)
    rows_in = prog["instances"]["rows"]
    # after the last qkv_proj row of every layer-0 call, one all_gather2 row per token row for q (node 5) and for k (node 6)
    rows: list[dict] = []
    pending: list[dict] = []
    for r in rows_in:
        mod = r["name"].split("/", 1)[0]
        if pending and mod != "model.model.layers.0.self_attn.qkv_proj":
            n = len(pending)
            for node in ("all_gather2_5", "all_gather2_6"):
                site = f"model.model.layers.0.self_attn.{node}" if site_names else "model.model.layers.0.self_attn"
                for i in range(n):
                    rows.append({"i": len(rows), "spec": f"AllGather2_v1{{N={Q // W}}}", "form": "call", "name": f"{site}/{node}[{i}]/all_gather", "args": []})
            pending = []
        if mod == "model.model.layers.0.self_attn.qkv_proj":
            pending.append(r)
        rows.append(dict(r, i=len(rows)))
    prog["instances"]["rows"] = rows
    return prog


def test_tp2_mid_module_collective_is_its_own_identity_with_a_partial():
    """[R19 moe F-r19-moe, row #70] a collective issued MID-module (no Gemm/EmbeddingShard partial under the same module) is required as
    ONE identity per call SITE -- op_path = the module, invocation = the site's index among the module's collective sites in Program
    order, the site path recorded in `op_path_aliases` (the binder's k-th occurrence = invocation k; sampled_replay resolves rows through
    the same table) -- and the collective's rank-local INPUT is required on the SAME row as `shard_rank<r>` / `part_rank<r>` (there is no
    partial-producing row to hang it on: the shard is a slice of the qkv Gemm's `0`), so `tp_peer_binding` binds every gathered output
    to the peer's shard.  The qkv_proj Gemm stays the module's ordinary `0` (it feeds attention, not the collective)."""
    Q = 64
    m = {rank: build_manifest(_olmoe_tp_program(rank, site_names=True, Q=Q), {"sweep": {"row_id": "olmoe-tp2-test", "batch": 1, "arrive_steps": [0]}},
                              rank=rank, ranks=W) for rank in (0, 1)}
    for rank in (0, 1):
        man = m[rank]
        assert man["complete"] is True and man["unmodelled"] == {}, man["unmodelled"]
        s0 = [r for r in man["identities"] if r["step"] == 0 and r["op_path"] == "model.layers.0.self_attn"]
        outs = sorted(((r["invocation"], r["output_member"], r["element_range"], r["family"], r["spec"]) for r in s0))
        LP = 4
        assert outs == [(0, "out", [0, LP * Q], "instance_outputs", "AllGather2_v1"), (0, f"shard_rank{rank}", [0, LP * Q // W], "tp_rank_partials", "AllGather2_v1"),
                        (1, "out", [0, LP * Q], "instance_outputs", "AllGather2_v1"), (1, f"shard_rank{rank}", [0, LP * Q // W], "tp_rank_partials", "AllGather2_v1")], outs
        # the site paths alias to (module, k) -- the collector commits under the module; the k-th occurrence is invocation k
        assert man["op_path_aliases"]["model.layers.0.self_attn.all_gather2_5"] == ["model.layers.0.self_attn", 0]
        assert man["op_path_aliases"]["model.layers.0.self_attn.all_gather2_6"] == ["model.layers.0.self_attn", 1]
        # the qkv Gemm is NOT a partial: the module's ordinary output
        qkv = [r for r in man["identities"] if r["step"] == 0 and r["op_path"] == "model.layers.0.self_attn.qkv_proj"]
        assert [(r["output_member"], r["family"]) for r in qkv] == [("0", "instance_outputs")]
        # the dense TP2 vocabulary (o_proj / down_proj / embed / logits) is unchanged
        assert [r for r in man["identities"] if r["step"] == 0 and r["op_path"] == "model.layers.0.self_attn.o_proj" and r["family"] == "tp_rank_partials"]
    row = merge_manifests([m[0], m[1]])
    assert row["complete"] is True
    pb = row["tp_peer_binding"]
    assert pb["n_unbound"] == 0, pb["unbound"]
    assert pb["collective_outputs"] == m[0]["populations"]["by_family"]["tp_rank_partials"] + m[1]["populations"]["by_family"]["tp_rank_partials"]
    # coverage: the collector's committed names (module/out x2 in order, module/shard_rank<r> x2) bind exactly
    for rank in (0, 1):
        c = coverage_check(row, _layouts(row, rank), rank=rank)
        assert c["ok"], (c["missing"][:3], c["committed_unmatched"][:3] if "committed_unmatched" in c else None)
    # every invocation >= 1 identity is the image of an alias entry (sampled_replay's identities_unreachable_by_alias)
    images = {(c, int(k)) for c, k in row["op_path_aliases"].values()}
    assert all((r["op_path"], int(r["invocation"])) in images for r in row["identities"] if int(r.get("invocation", 0) or 0) >= 1)
    # the three enumerators (inventory / streaming Traversal / reference) agree on the site rule
    from verity_capture.commit import required_values as RV
    wl = {"sweep": {"row_id": "olmoe-tp2-test", "batch": 1, "arrive_steps": [0]}}
    for rank in (0, 1):
        trav = build_manifest(_olmoe_tp_program(rank, site_names=True, Q=Q), wl, rank=rank, ranks=W, engine="traversal")
        assert RV.manifest_equal(m[rank], trav)["equal"], RV.manifest_equal(m[rank], trav)
        ref, _summary = RV.reference_required_values(_olmoe_tp_program(rank, site_names=True, Q=Q), RV.Query(rank=rank, ranks=W))
        assert {str(i.key()) for i in ref} == {str((r["program_digest"], r["request_id"], r["step"], r["invocation"], r["op_path"], r["output_member"],
                                                      tuple(r["element_range"]), r["rank"])) for r in m[rank]["identities"]}


def test_tp2_mid_module_collective_under_the_bare_module_is_named_not_folded():
    """The R17 vocabulary (two gathers under the bare `self_attn`, no site): the manifest must NOT emit one folded 2xWxN `out` identity
    with no partial (the silent shape of row #70); the merge names it unbound (fail closed, by name)."""
    m = {rank: build_manifest(_olmoe_tp_program(rank, site_names=False), {"sweep": {"row_id": "olmoe-tp2-bare", "batch": 1, "arrive_steps": [0]}},
                              rank=rank, ranks=W) for rank in (0, 1)}
    row = merge_manifests([m[0], m[1]])
    assert row["complete"] is False or row["tp_peer_binding"]["n_unbound"] > 0


def test_tp2_global_b2_per_rank_then_merged():
    """[R15 tp TP-04, task 3] a B>=2 TP row: per rank, manbind's global manifest over the declared requests (request-local addressing,
    engine-step composition) built from that rank's request Programs; then the rank parts merged with tp-ranks-fold.  The partials
    (part_rank<r> / shard_rank<r>) are packed over the co-scheduled requests like the module output they precede (same block `rows`);
    the two ranks declare the identical engine-step composition; the loader keys rank r's rows by rank r's Program digest."""
    from verity_capture.commit.required_manifest import build_global_manifest
    fixture = {"fixture": "tp2_b2", "requests": [{"request_id": "a", "index": 0, "prompt_len": 4, "max_tokens": 2, "arrive_step": 0},
                                                  {"request_id": "b", "index": 1, "prompt_len": 3, "max_tokens": 3, "arrive_step": 1}]}
    g = {}
    for rank in (0, 1):
        g[rank] = build_global_manifest(fixture, {"r0": _tp_program(rank, LP=4, T=1), "r1": _tp_program(rank, LP=3, T=2)}, rank=rank, ranks=W)
        assert g[rank]["rank"] == rank and g[rank]["ranks"] == W and g[rank]["global"] is True and g[rank]["complete"] is True
        parts = [r for r in g[rank]["identities"] if r["family"] == "tp_rank_partials"]
        assert parts and all(int(r["rank"]) == rank and r["output_member"].endswith(f"rank{rank}") for r in parts)
        assert all("engine_step" in r and r.get("rows") is not None for r in parts)              # packed like instance outputs: block height named
        # co-scheduled engine step 1 (r0 decode + r1 prefill): a down_proj partial for each request with its own block height
        e1 = [r for r in parts if r["engine_step"] == 1 and r["op_path"].endswith("down_proj")]
        assert {(r["request_id"], r["rows"]) for r in e1} == {("r0", 1), ("r1", 3)}
    row = merge_manifests([g[0], g[1]])
    assert row["program_digest_kind"] == "tp-ranks-fold" and row["global"] is True and row["engine_steps"] == g[0]["engine_steps"]
    assert row["program_digest_per_rank"] == {"0": g[0]["program_digest"], "1": g[1]["program_digest"]}
    assert row["populations"]["identities"] == g[0]["populations"]["identities"] + g[1]["populations"]["identities"]
    assert {r["program_digest"] for r in row["identities"] if int(r["rank"]) == 1} == {g[1]["program_digest"]}
    # a rank part of a different declaration is refused (not the same workload on every rank)
    other = build_global_manifest({"fixture": "tp2_b2x", "requests": fixture["requests"][:1]}, {"r0": _tp_program(1, LP=4, T=1)}, rank=1, ranks=W)
    with pytest.raises(ValueError, match="engine-step composition"):
        merge_manifests([g[0], other])
