"""prefix_cache: the linkage record on synthetic StepStart logs (torch-free)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.observe import prefix_cache as PC
from verity_vllm.observe.events import RunHeader, StepEnd, StepStart
from verity_vllm.observe.log import LogWriter, read_log

BS = 16


# ------------------------------------------------------------------------------------------------------------ synthetic logs


def _step(step: int, reqs: list[tuple[str, int, int, list[int], list[int]]]) -> StepStart:
    """reqs: (request_id, num_computed, n_rows, block_table, token_ids); slots follow the block table (position p -> block
    table[p // BS] * BS + p % BS)."""
    rows, ids, pos, slots, seq, ncomp, bt = [], [], [], [], {}, {}, {}
    for rid, nc, n, table, toks in reqs:
        for k in range(n):
            p = nc + k
            rows.append((rid, p))
            pos.append(p)
            ids.append(toks[k])
            slots.append(table[p // BS] * BS + p % BS)
        seq[rid] = nc + n
        ncomp[rid] = nc
        bt[rid] = list(table)
    lr = [(rid, seq[rid] - 1) for rid, *_ in reqs]
    return StepStart(step, rows, ids, pos, seq, ncomp, bt, BS, slots, lr, [len(rows) - 1] * len(lr), len(reqs), len(rows), len(rows))


def _header(prompts: dict[str, list[int]], pc: bool = True) -> RunHeader:
    wl = {"requests": [{"request_id": r, "prompt_token_ids": t, "max_tokens": 4} for r, t in prompts.items()], "sampling": {}}
    return RunHeader("test", "now", "pid", {"engine_kwargs": {"enable_prefix_caching": pc}}, {}, wl)


A, B = "req-A", "req-B"
P = list(range(1000, 1000 + BS))            # one full shared block
X = [7, 8]
Y = [9, 10]


def _warm_log() -> list:
    """A prefills P+X (18 rows, blocks [0, 1]) at step 0, decodes at 1; B arrives at step 2 with block 0 cached: num_computed 16,
    2 prompt rows into its own block 5, then decodes at 3."""
    ev = [_header({A: P + X, B: P + Y})]
    ev += [_step(0, [(A, 0, 18, [0, 1], P + X)]), StepEnd(0)]
    ev += [_step(1, [(A, 18, 1, [0, 1, 2], [500])]), StepEnd(1)]
    ev += [_step(2, [(A, 19, 1, [0, 1, 2], [501]), (B, 16, 2, [0, 5], Y)]), StepEnd(2)]
    ev += [_step(3, [(A, 20, 1, [0, 1, 2], [502]), (B, 18, 1, [0, 5, 6], [600])]), StepEnd(3)]
    return ev


def test_warm_request_links_every_reused_slot_to_the_producer_rows():
    rec = PC.extract_linkage(_warm_log())
    assert rec["block_size"] == BS and rec["engine_enable_prefix_caching"] is True and rec["steps"] == 4
    ra, rb = rec["requests"][A], rec["requests"][B]
    assert ra["status"] == "cold" and ra["warm"] is False and ra["num_computed_at_first_step"] == 0 and ra["reused_blocks"] == []
    assert rb["warm"] is True and rb["first_step"] == 2 and rb["num_computed_at_first_step"] == 16 and rb["prompt_rows_at_first_step"] == 2
    assert rb["declared_prompt_len"] == 18 and rb["scheduled_steps"] == [2, 3]
    assert len(rb["reused_blocks"]) == 1
    blk = rb["reused_blocks"][0]
    assert blk["physical_block"] == 0 and blk["token_range"] == [0, 16] and blk["status"] == "linked"
    assert len(blk["slots"]) == 16
    for off, s in enumerate(blk["slots"]):
        assert s["slot"] == off and s["position"] == off
        assert s["writer"] == {"step": 0, "row": off, "request_id": A, "position": off, "token": P[off]}
        assert s["token_identical"] is True and s["position_identical"] is True
    assert rb["producers"] == [{"request_id": A, "step": 0}]
    assert rb["linked_slots"] == 16 and rb["unlinked_slots"] == 0 and rb["identity_failures"] == 0 and rb["status"] == "linked"
    assert rec["closure"] == {"warm_requests": 1, "linked_slots": 16, "unlinked_slots": 0, "identity_failures": 0, "status": "linked"}


def test_cold_control_has_no_reuse():
    ev = [_header({A: P + X, B: P + Y}, pc=False)]
    ev += [_step(0, [(A, 0, 18, [0, 1], P + X)]), StepEnd(0)]
    ev += [_step(1, [(A, 18, 1, [0, 1, 2], [500])]), StepEnd(1)]
    ev += [_step(2, [(A, 19, 1, [0, 1, 2], [501]), (B, 0, 18, [3, 4], P + Y)]), StepEnd(2)]
    rec = PC.extract_linkage(ev)
    assert rec["engine_enable_prefix_caching"] is False
    assert rec["requests"][B]["status"] == "cold" and rec["requests"][B]["prompt_rows_at_first_step"] == 18
    assert rec["closure"]["status"] == "no-reuse" and rec["closure"]["warm_requests"] == 0


def test_state_from_outside_the_log_is_unlinked_not_closed():
    """B reuses a block nobody in this log wrote (a warm server / a previous run): matching tokens do not make it linked."""
    ev = [_header({A: P + X, B: P + Y})]
    ev += [_step(0, [(A, 0, 18, [0, 1], P + X)]), StepEnd(0)]
    ev += [_step(1, [(A, 18, 1, [0, 1, 2], [500]), (B, 16, 2, [9, 5], Y)]), StepEnd(1)]   # block 9: no writer
    rec = PC.extract_linkage(ev)
    rb = rec["requests"][B]
    assert rb["status"] == "unlinked" and rb["unlinked_slots"] == 16 and rb["linked_slots"] == 0
    assert rb["reused_blocks"][0]["status"] == "unlinked" and all(s["writer"] is None for s in rb["reused_blocks"][0]["slots"])
    assert rec["closure"]["status"] == "unlinked"


def test_linked_but_not_identical_is_reported_separately():
    """The physical block was written in the log, but by a different token sequence than B declares: the linkage exists, the
    consumer's claim (prompt == producer's prompt) fails -> identity-mismatch, not linked."""
    P2 = [t + 1 for t in P]
    ev = [_header({A: P2 + X, B: P + Y})]
    ev += [_step(0, [(A, 0, 18, [0, 1], P2 + X)]), StepEnd(0)]
    ev += [_step(1, [(A, 18, 1, [0, 1, 2], [500]), (B, 16, 2, [0, 5], Y)]), StepEnd(1)]
    rec = PC.extract_linkage(ev)
    rb = rec["requests"][B]
    assert rb["linked_slots"] == 16 and rb["identity_failures"] == 16 and rb["status"] == "identity-mismatch"
    assert all(s["token_identical"] is False for s in rb["reused_blocks"][0]["slots"])


def test_last_writer_wins_after_block_reuse():
    """Block 0 is written by A at step 0, freed, rewritten by C at step 3 with other tokens; B at step 4 reads C's write."""
    C = "req-C"
    Q = [t + 50 for t in P]
    ev = [_header({A: P + X, B: Q + Y, C: Q + X})]
    ev += [_step(0, [(A, 0, 18, [0, 1], P + X)]), StepEnd(0)]
    ev += [_step(3, [(C, 0, 18, [0, 7], Q + X)]), StepEnd(3)]
    ev += [_step(4, [(C, 18, 1, [0, 7, 8], [1]), (B, 16, 2, [0, 5], Y)]), StepEnd(4)]
    rec = PC.extract_linkage(ev)
    rb = rec["requests"][B]
    assert rb["producers"] == [{"request_id": C, "step": 3}] and rb["status"] == "linked"
    assert all(s["writer"]["request_id"] == C and s["token_identical"] for s in rb["reused_blocks"][0]["slots"])


def test_producer_instances_from_instances_jsonl(tmp_path: Path):
    inst = tmp_path / "instances.jsonl"
    # fold.py's format: one record per instance, `row` = row index inside the compute, `rows` = the compute's row count
    lines = []
    node = 0
    for r in range(18):
        for hint in ("model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.rotary_emb/rope.k",
                     "model.layers.0.self_attn.rotary_emb/rope.q", "model.layers.0.mlp.gate_proj"):
            lines.append({"node": node, "step": 0, "row": r, "rows": 18, "spec": "Gemm_v1{...}", "hint": hint})
            node += 1
    lines.append({"node": node, "step": 2, "row": 0, "rows": 3, "spec": "Gemm_v1{...}", "hint": "model.layers.0.self_attn.k_proj"})
    inst.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rec = PC.extract_linkage(_warm_log(), instances=inst)
    blk = rec["requests"][B]["reused_blocks"][0]
    pi = blk["producer_instances"]
    assert pi["count"] == 16 * 3 and len(pi["nodes"]) == 48          # 16 slots x (k_proj, v_proj, rope.k) of step 0; rope.q/mlp/step-2 excluded
    assert pi["hints"] == ["model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.rotary_emb/rope.k", "model.layers.0.self_attn.v_proj"]
    # the pinned profile's fused-qkv hint form is matched too
    assert PC.KV_PRODUCER_RE.search("model.layers.3.self_attn.qkv_proj/model.layers.3.self_attn.qkv_proj.weight")
    assert PC.KV_PRODUCER_RE.search("model.layers.3.self_attn.rotary_emb/rope.k") and not PC.KV_PRODUCER_RE.search("model.layers.3.self_attn.rotary_emb/rope.q")
    assert all(n < 18 * 5 for n in pi["nodes"])
    from verity_vllm.observe.fold import patterns_prefix as PP

    e = PP.obligations(rec)["requests"][B]["state_edges"][3]
    assert e["producer_instances"] == [15, 16, 17]                     # row 3's k_proj, v_proj, rotary nodes (5 per row, q/mlp skipped)
    assert blk["slots"][3]["producer_instances"] == [15, 16, 17]


def test_extract_run_writes_beside_tokens_json(tmp_path: Path):
    cap = tmp_path / "run" / "capture"
    cap.mkdir(parents=True)
    with LogWriter(cap / "log.jsonl.gz") as w:
        for ev in _warm_log():
            w.write(ev)
    (cap / "tokens.json").write_text("{}")
    rec, dst = PC.extract_run(tmp_path / "run")
    assert dst == cap / "prefix_linkage.json" and dst.exists()
    on_disk = json.loads(dst.read_text())
    assert on_disk["schema"] == PC.SCHEMA_LINKAGE and on_disk["closure"]["status"] == "linked"
    assert on_disk["requests"][B]["reused_blocks"][0]["slots"][3]["writer"]["request_id"] == A
    # the log round-trips the tuple-list fields the extractor relies on
    sts = [e for e in read_log(cap / "log.jsonl.gz") if isinstance(e, StepStart)]
    assert sts[2].rows[0] == (A, 19) and sts[2].rows[1] == (B, 16) and sts[2].slot_mapping[1] == 5 * BS
    assert "req-A"[:8] in PC.summarize(rec) and "linked" in PC.summarize(rec)


def test_workload_override_supplies_declared_prompts_when_header_lacks_them():
    ev = _warm_log()
    ev[0] = RunHeader("test", "now", "pid", {}, {}, {"requests": [], "sampling": {}})
    rec = PC.extract_linkage(ev)
    assert rec["requests"][B]["declared_prompt_len"] is None
    assert all(s["token_identical"] is None for s in rec["requests"][B]["reused_blocks"][0]["slots"])
    assert rec["requests"][B]["status"] == "linked"          # linkage does not depend on the declaration; identity does
    rec2 = PC.extract_linkage(ev, workload={"requests": [{"request_id": B, "prompt_token_ids": P + Y}]})
    assert rec2["requests"][B]["declared_prompt_len"] == 18
    assert all(s["token_identical"] is True for s in rec2["requests"][B]["reused_blocks"][0]["slots"])


def _run_dir(root: Path, events: list, *, gen: dict[str, list[int]], census_steps: list[dict] | None = None, pc: bool = True) -> Path:
    cap = root / "capture"
    cap.mkdir(parents=True)
    with LogWriter(cap / "log.jsonl.gz") as w:
        for ev in events:
            w.write(ev)
    reqs = [{"request_id": r, "generated_token_ids": g, "arrive_step": 0, "first_step": 0, "finish_step": 3, "finish_reason": "length", "n_generated": len(g)} for r, g in gen.items()]
    (cap / "tokens.json").write_text(json.dumps({"requests": reqs}))
    (root / "control").mkdir()
    (root / "control" / "tokens.json").write_text(json.dumps({"requests": reqs}))
    (cap / "versions.json").write_text(json.dumps({"engine_kwargs": {"enable_prefix_caching": pc}}))
    if census_steps is not None:
        (root / "census").mkdir()
        (root / "census" / "census.json").write_text(json.dumps({"steps": census_steps}))
    (root / "gates.json").write_text(json.dumps({"verdict": "ACCEPTED", "gates": {"G1": {"status": "pass", "reason": None}}}))
    return root


def test_evidence_compares_warm_and_cold_run_dirs(tmp_path: Path):
    census = [{"step": 0, "n_kernels": 10, "kernels": {"void vllm::reshape_and_cache_flash_kernel<...>": {"count": 30, "grid": [18, 1, 1], "block": [128, 1, 1]}}},
              {"step": 2, "n_kernels": 10, "kernels": {"void vllm::reshape_and_cache_flash_kernel<...>": {"count": 30, "grid": [3, 1, 1], "block": [128, 1, 1]},
                                                       "cutlass_gemm_xyz": {"count": 5, "grid": [1, 2, 1], "block": [64, 1, 1]}}}]
    warm = _run_dir(tmp_path / "warm", _warm_log(), gen={A: [1, 2], B: [3, 4]}, census_steps=census)
    cold_ev = [_header({A: P + X, B: P + Y}, pc=False), _step(0, [(A, 0, 18, [0, 1], P + X)]), StepEnd(0),
               _step(1, [(A, 18, 1, [0, 1, 2], [500])]), StepEnd(1),
               _step(2, [(A, 19, 1, [0, 1, 2], [501]), (B, 0, 18, [3, 4], P + Y)]), StepEnd(2)]
    cold = _run_dir(tmp_path / "cold", cold_ev, gen={A: [1, 2], B: [3, 4]}, pc=False)
    doc = PC.evidence(warm, cold)
    assert doc["warm"]["reuse_observed"] is True and doc["cold"]["reuse_observed"] is False
    assert doc["warm"]["first_step_per_request"][B] == {"step": 2, "num_computed": 16, "rows": 2, "seq_len": 18, "block_table": [0, 5], "step_num_tokens": 3}
    assert doc["cold"]["first_step_per_request"][B]["num_computed"] == 0 and doc["cold"]["first_step_per_request"][B]["rows"] == 18
    assert doc["served_identity"] == {A: True, B: True}
    cw = doc["warm"]["census_at_warm_admission_step"]
    assert list(cw["row_kernels"].values())[0]["grid"] == [3, 1, 1] and "cutlass_gemm_xyz" in cw["gemm_kernels"]
    assert doc["linkage_closure"]["status"] == "linked" and doc["obligations_run"]["status"] == "closed"
    assert doc["warm"]["gates"]["verdict"] == "ACCEPTED" and doc["warm"]["gates"]["G1"]["status"] == "pass"
    assert (warm / "prefix_evidence.json").exists() and (warm / "capture" / "prefix_linkage.json").exists()
    txt = PC.evidence_text(doc)
    assert "reuse_observed=True" in txt and "served identity" in txt


def test_census_witness_sorts_row_count_and_gemm_kernels():
    names = ["void vllm::rotary_embedding_kernel<...>", "sm90_tile_kernel", "sm80_tile_kernel", "ampere_tile_kernel",
             "cublasLt_splitKreduce", "xmma_tile", "Kernel2_gemm", "void at::vectorized_elementwise_kernel<...>"]
    census = {"steps": [{"step": 0, "kernels": {n: {"count": 1, "grid": [1, 1, 1], "block": [32, 1, 1]} for n in names}}]}
    w = PC.census_row_witness(census, 0)
    assert list(w["row_kernels"]) == names[:1] and list(w["gemm_kernels"]) == names[1:7]


# ----------------------------------------------------------------------------------------------------- obligations (patterns_prefix)


def test_obligations_closed_warm_run():
    from verity_vllm.observe.fold import patterns_prefix as PP

    ob = PP.obligations(PC.extract_linkage(_warm_log()))
    assert ob["run"]["status"] == "closed" and PP.closure(PC.extract_linkage(_warm_log())) == "closed"
    rb = ob["requests"][B]
    assert rb["status"] == "closed" and rb["declared_leaves_without_gate"] == 16 and rb["prompt_rows_with_gates"] == 2
    assert rb["counts"] == {"state_edges": 16, "alias_obligations": 16, "alias_failed": 0, "alias_undecided": 0, "external_state": 0}
    e = rb["state_edges"][5]
    assert e == {"consumer_request": B, "consumer_first_step": 2, "position": 5, "slot": 5, "physical_block": 0,
                 "producer_request": A, "producer_step": 0, "producer_row": 5, "producer_instances": []}
    assert all(a["holds"] is True and a["producer_request"] == A for a in rb["alias_obligations"])
    assert ob["requests"][A]["status"] == "cold"
    iface = PP.interface_of(PC.extract_linkage(_warm_log()), B)
    assert iface["own_prompt_positions"] == [16, 17] and iface["aliased_positions"]["0"] == [A, 0] and iface["external_positions"] == []
    assert "closed" in PP.blocker_text(PC.extract_linkage(_warm_log()))


def test_obligations_external_state_and_alias_violation():
    from verity_vllm.observe.fold import patterns_prefix as PP

    ev = [_header({A: P + X, B: P + Y})]
    ev += [_step(0, [(A, 0, 18, [0, 1], P + X)]), StepEnd(0)]
    ev += [_step(1, [(A, 18, 1, [0, 1, 2], [500]), (B, 16, 2, [9, 5], Y)]), StepEnd(1)]
    ob = PP.obligations(PC.extract_linkage(ev))
    assert ob["run"]["status"] == "external-state" and ob["requests"][B]["counts"]["external_state"] == 16
    assert ob["requests"][B]["state_edges"] == [] and "external" in PP.blocker_text(PC.extract_linkage(ev))
    P2 = [t + 1 for t in P]
    ev2 = [_header({A: P2 + X, B: P + Y}), _step(0, [(A, 0, 18, [0, 1], P2 + X)]), StepEnd(0),
           _step(1, [(A, 18, 1, [0, 1, 2], [500]), (B, 16, 2, [0, 5], Y)]), StepEnd(1)]
    ob2 = PP.obligations(PC.extract_linkage(ev2))
    assert ob2["run"]["status"] == "alias-violated" and ob2["run"]["state_edges"] == 16 and ob2["run"]["alias_failed"] == 16
    # the linkage exists (edges) even though the claim fails: identity and linkage are separate facts
    assert ob2["requests"][B]["counts"]["state_edges"] == 16


def test_obligations_cold_run_has_nothing_to_close():
    from verity_vllm.observe.fold import patterns_prefix as PP

    ev = [_header({A: P + X}, pc=False), _step(0, [(A, 0, 18, [0, 1], P + X)]), StepEnd(0)]
    lk = PC.extract_linkage(ev)
    assert PP.closure(lk) == "cold" and PP.blocker_text(lk).startswith("no prefix reuse")


@pytest.mark.parametrize("ncomp,expect_blocks", [(16, 1), (24, 2), (32, 2)])
def test_partial_last_block_ranges(ncomp, expect_blocks):
    """num_computed need not be a block multiple (vLLM recomputes a partial tail; the range is what was cached)."""
    P32 = list(range(2000, 2032))
    ev = [_header({A: P32 + X, B: P32 + Y})]
    ev += [_step(0, [(A, 0, 34, [0, 1, 2], P32 + X)]), StepEnd(0)]
    ev += [_step(2, [(B, ncomp, 34 - ncomp, [0, 1, 2, 3], (P32 + Y)[ncomp:])]), StepEnd(2)]
    rb = PC.extract_linkage(ev)["requests"][B]
    assert len(rb["reused_blocks"]) == expect_blocks
    assert rb["reused_blocks"][-1]["token_range"] == [BS * (expect_blocks - 1), ncomp]
    assert rb["linked_slots"] == ncomp and rb["status"] == "linked"
