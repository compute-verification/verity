"""[R15 tp, coord M-0481 C2 form (B), TP family] TPPartialSource.compare_match_oracle: every committed TP partial / MoE received output of a
rank == the tp2_match capture recorder's stored collective input / output at the same (step, k), bit-for-bit, exhaustively over the family.
Needs torch (torch.save/load of the oracle steps); CPU only.

  * positive: 3 committed members (o_proj partial, MoE partial, MoE received /out) == the stored part/out tensors -> PASS 3/3
  * NEGATIVE late read: the committed MoE /out holds another tensor's bytes -> FAIL naming (step, k, member), 1 mismatch
  * NEGATIVE relabel: the rank's members are tagged part_rank1 -> every oracle collective is missing_in_commit -> FAIL
  * NEGATIVE extra: a committed partial the oracle never recorded -> committed_not_in_oracle -> FAIL (exhaustive or not a record)
  * NEGATIVE no oracle for a committed step -> steps_without_oracle -> FAIL
  * NEGATIVE retained copy changed after the commit (the oracle agrees with the change) -> not_verified -> FAIL
  * witness-conditioned: equal MoE align witnesses leave every tensor compared; a differing one exempts its cone (this rank's partials
    from its position on, every rank's received outputs there, every later step) and leaves upstream compared; a tampered value under
    an equal witness, or a witness on one side only, FAILS

The rank's members are committed on the GPU-tree step block path and read through `oracle_compare.OpenedReader` (opened, verified against the
rank's run root), as `worker.tp2_commit_finalize` reads them.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tests.check.opened import commit_steps, meta, retained_block  # noqa: E402
from verity_vllm.check import oracle_compare as OC  # noqa: E402
from verity_vllm.acquire.sources import oracle_witness as OW  # noqa: E402
from verity_vllm.acquire.sources import partial_source as tps  # noqa: E402

OPROJ, MOE = "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.experts"


class RowParallelLinear:
    reduce_results, tp_size = True, 2

    def parameters(self):
        return iter(())


class FusedMoE:
    reduce_results, tp_size, moe_config = True, 2, None

    def parameters(self):
        return iter(())


def _model():
    return types.SimpleNamespace(named_modules=lambda: [(OPROJ, RowParallelLinear()), (MOE, FusedMoE())])


def _meta(name: str, t: torch.Tensor):
    return name, tps._bytes(t)


def _oracle(cap: Path, rank: int, steps: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]]) -> None:
    cap.mkdir(parents=True, exist_ok=True)
    with (cap / f"coll.rank{rank}.jsonl").open("w") as f:
        for s, (parts, _outs) in steps.items():
            for k, mod in enumerate([OPROJ, MOE][: len(parts)]):
                f.write(json.dumps({"rank": rank, "step": s, "k": k, "kind": "all_reduce", "module": mod, "shape": list(parts[k].shape)}) + "\n")
    for s, (parts, outs) in steps.items():
        torch.save({"step": s, "part": parts, "out": outs}, cap / f"coll.rank{rank}.step{s}.pt")


def _src(steps: dict, rank: int = 0, relabel: int | None = None):
    """The rank's committed members {step: [(name, bytes)]} -> (a TP source over their committer, the reader of record over its run)."""
    com, run = commit_steps(steps)
    return tps.TPPartialSource(com, _model(), rank=rank, world=2, relabel_rank=relabel), OC.OpenedReader(com, run)


def _tensors(rows: int = 4, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(rows, 8, generator=g).to(torch.bfloat16) for _ in range(4)]


def test_positive_three_members_equal_the_stored_collectives(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1)]})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "PASS" and r["tensors"] == 3 and r["equal"] == 3 and r["steps_compared"] == 1
    assert r["by_member"] == {"partial": {"tensors": 2, "equal": 2, "exempt": 0}, "out": {"tensors": 1, "equal": 1, "exempt": 0}}
    assert r["compared"] == 3 and r["exempt"] == 0 and r["witness_counts"] == {"equal": 0, "differs": 0, "unpaired": 0}
    assert r["mismatch_n"] == r["missing_n"] == r["not_in_oracle_n"] == r["not_verified_n"] == 0 and r["families_covered"] == ["tp_rank_partials"]
    assert r["value_source"]["source"] == "opened" and r["value_source"]["reads"] == 3 and r["value_source"]["failed"] == 0


def test_late_read_of_the_moe_received_output_is_a_named_mismatch(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o0)]})   # o0 != o1: the next norm's bytes
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["mismatch_n"] == 1 and r["equal"] == 2
    m = r["mismatches"][0]
    assert (m["step"], m["k"], m["member"]) == (0, 1, f"{MOE}/out") and m["n_diff_bytes"] > 0


def test_relabelled_rank_members_are_missing_in_commit(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank1", p0), _meta(f"{MOE}/part_rank1", p1), _meta(f"{MOE}/out", o1)]}, relabel=1)
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["missing_n"] == 2 and r["tensors"] == 1
    assert [m[2] for m in r["missing_in_commit"]] == [f"{OPROJ}/part_rank0", f"{MOE}/part_rank0"]


def test_committed_partial_the_oracle_never_recorded_fails_exhaustiveness(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1),
                    _meta("model.layers.9.self_attn.o_proj/part_rank0", p0)]})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["equal"] == 3 and r["not_in_oracle_n"] == 1
    assert r["committed_not_in_oracle"] == [(0, "model.layers.9.self_attn.o_proj/part_rank0")]


def test_committed_step_without_a_stored_oracle_fails(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1)],
                1: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1)]})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["steps_without_oracle"] == [1] and r["steps_compared"] == 1 and r["equal"] == 3


# ---------------------------------------------------------------------------------------------- rev-pm M-0585 TIP1 / TIP2 / TIP3

def test_tip1_oracle_step_the_committer_never_reached_fails_named(tmp_path):
    """N-A: the oracle stored steps 0 and 1 but the committer only reached step 0 -> exhaustive in both directions, FAIL naming step 1."""
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1]), 1: ([p1, p0], [o1, o0])})
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1)]})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["oracle_steps_not_committed"] == [1] and r["steps_compared"] == 1 and r["equal"] == 3
    assert "never committed" in r["why"] and "first 1" in r["why"]


def test_tip2_truncated_oracle_pt_is_a_named_verdict_not_a_crash(tmp_path):
    """N-B: the .pt holds fewer tensors than the jsonl names -> named 'oracle inconsistent', no IndexError."""
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    torch.save({"step": 0, "part": [p0], "out": [o0, o1]}, tmp_path / "coll.rank0.step0.pt")       # jsonl names 2, .pt part holds 1
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1)]})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["oracle_inconsistent"] == [{"step": 0, "jsonl_names": 2, "pt_part": 1, "pt_out": 2}]
    assert "jsonl names 2" in r["why"] and ".pt holds part 1" in r["why"] and "error" not in r


def test_tip3_module_recorded_twice_in_a_step_is_refused(tmp_path):
    """N-C: the same module twice at one step with equal bytes must not PASS 2/2 against ONE committed member."""
    p0, p1, o0, o1 = _tensors()
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "coll.rank0.jsonl").open("w") as f:
        for k in range(2):
            f.write(json.dumps({"rank": 0, "step": 0, "k": k, "kind": "all_reduce", "module": OPROJ, "shape": list(p0.shape)}) + "\n")
    torch.save({"step": 0, "part": [p0, p0], "out": [o0, o0]}, tmp_path / "coll.rank0.step0.pt")
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0)]})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["duplicate_module_steps"] == [{"step": 0, "modules": [OPROJ]}] and r["tensors"] == 0
    assert "recorded more than once" in r["why"]


def test_a_member_changed_in_the_retained_copy_after_the_commit_is_not_verified_and_fails_named(tmp_path):
    """On the TP path: after the commit the rank's retained copy of the MoE received output is overwritten with another tensor's bytes,
    and the oracle agrees with the overwrite -- a compare of the committer's memory would PASS 3/3.  The opening of that member does not
    verify against the rank's run root: not_verified, FAIL by name; the members in other leaves still compare equal."""
    p0, p1, o0, o1 = _tensors(rows=16)                                         # 256 B each: one leaf per member
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1)]})
    m = meta(src.c, 0, f"{MOE}/out")
    retained_block(src.c, 0)[m.dev_off:m.dev_off + m.nbytes] = torch.frombuffer(bytearray(tps._bytes(o0)), dtype=torch.uint8)
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o0])})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["not_verified_n"] == 1 and r["equal"] == 2 and r["mismatch_n"] == 0 and r["not_in_oracle_n"] == 0
    assert r["not_verified"][0][:3] == (0, 1, f"{MOE}/out") and "does not verify against the run root" in r["why"]
    assert r["value_source"]["failed"] == 1 and r["value_source"]["failures"][0]["name"] == f"{MOE}/out"


LAYER = "model.layers.1"
NORMS = [f"{LAYER}.input_layernorm", f"{LAYER}.post_attention_layernorm"]


def _layer_oracle(cap: Path, parts: list[torch.Tensor], kind: str = "all_reduce") -> None:
    cap.mkdir(parents=True, exist_ok=True)
    with (cap / "coll.rank0.jsonl").open("w") as f:
        for k, p in enumerate(parts):
            f.write(json.dumps({"rank": 0, "step": 0, "k": k, "kind": kind, "module": LAYER, "shape": list(p.shape)}) + "\n")
    torch.save({"step": 0, "part": parts, "out": []}, cap / "coll.rank0.step0.pt")


def test_a_fused_norm_record_compares_its_partial_not_the_stacked_residual(tmp_path):
    """The recorder stores a `fused_allreduce_rms_norm` call's part as stack([partial, residual]); the Commit's member is the partial."""
    p0, p1, r0, r1 = _tensors()
    _layer_oracle(tmp_path, [torch.stack([p0, r0]), torch.stack([p1, r1])], kind="fused_allreduce_rms_norm")
    src, rd = _norm_src([(NORMS[0], p0), (NORMS[1], p1)])
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "PASS" and r["tensors"] == r["equal"] == 2
    src, rd = _norm_src([(NORMS[0], r0), (NORMS[1], p1)])
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["mismatch_n"] == 1


def _norm_src(committed: list[tuple[str, torch.Tensor]]):
    src, rd = _src({0: [_meta(f"{n}/part_rank0", t) for n, t in committed]})
    src.far.site_names = lambda: list(NORMS)
    return src, rd


def test_a_layer_recorded_twice_pairs_with_its_norm_site_partials_in_commit_order(tmp_path):
    """GLM: the recorder names both of a decoder layer's all-reduces by the layer, the Commit names them under its two norms."""
    p0, p1, _o0, _o1 = _tensors()
    _layer_oracle(tmp_path, [p0, p1])
    src, rd = _norm_src([(NORMS[0], p0), (NORMS[1], p1)])
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "PASS" and r["tensors"] == r["equal"] == 2 and r["duplicate_module_steps"] == []


def test_norm_site_partials_committed_out_of_call_order_are_mismatches(tmp_path):
    p0, p1, _o0, _o1 = _tensors()
    _layer_oracle(tmp_path, [p0, p1])
    src, rd = _norm_src([(NORMS[1], p1), (NORMS[0], p0)])
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["mismatch_n"] == 2 and r["equal"] == 0


def test_a_layer_recorded_more_often_than_its_norm_site_partials_is_still_refused(tmp_path):
    p0, p1, _o0, _o1 = _tensors()
    _layer_oracle(tmp_path, [p0, p1])
    src, rd = _norm_src([(NORMS[0], p0)])
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["duplicate_module_steps"] == [{"step": 0, "modules": [LAYER]}]


def _value_check(model, committed: list[tuple[str, torch.Tensor]]) -> dict:
    com, run = commit_steps({0: [_meta(n, t) for n, t in committed]})
    src = tps.TPPartialSource(com, model, rank=0, world=2)
    src.ref = {0: {n: tps._bytes(t) for n, t in committed}}
    return src.compare(OC.OpenedReader(com, run))


def test_a_rank_with_only_norm_site_partials_passes_the_value_check_on_the_tap_alone():
    """GLM-5.3: o_proj and the MoE leave their output un-reduced for fused_allreduce_rms_norm, so no RowParallelLinear site is there
    to re-execute; every partial still equals its hook-time clone, opened against the run root."""
    p0, p1, _o0, _o1 = _tensors()
    r = _value_check(types.SimpleNamespace(named_modules=lambda: []), [(f"{NORMS[0]}/part_rank0", p0), (f"{NORMS[1]}/part_rank0", p1)])
    assert r["verdict"] == "PASS" and r["tap_equal"] == 2 and r["reexec_tensors"] == 0 and r["reexec_due"] is False


def test_a_rank_with_a_row_parallel_site_and_no_re_execution_still_fails():
    p0, _p1, _o0, _o1 = _tensors()
    r = _value_check(_model(), [(f"{OPROJ}/part_rank0", p0)])
    assert r["verdict"] == "FAIL" and r["tap_equal"] == 1 and r["reexec_tensors"] == 0 and r["reexec_due"] is True


# ---------------------------------------------------------------------------------------------- the witness-conditioned oracle

def _schedule(seed: int) -> list[torch.Tensor]:
    """One MoE align call's committed witness: its rank, start, blk and par words as the float32 view the source commits."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 64, (6,), generator=g, dtype=torch.int32).view(torch.float32) for _ in OW.MEMBERS]


def _record_schedule(cap: Path, rank: int, step: int, ws: list[torch.Tensor], k: int = 1) -> None:
    """The capture's witness row of the MoE's align call at (step, k): it runs after o_proj's all-reduce, before the MoE's."""
    with (cap / OW.witness_file(rank)).open("a") as f:
        for m, t in zip(OW.MEMBERS, ws):
            f.write(json.dumps({"rank": rank, "step": step, "k": k, "module": MOE, "member": m, "sha256": OW.digest(tps._bytes(t))}) + "\n")


def _step(rank: int, ws: list[torch.Tensor], part_o: torch.Tensor, part_m: torch.Tensor, out_m: torch.Tensor) -> list:
    return [_meta(f"{OPROJ}/part_rank{rank}", part_o), *(_meta(f"{MOE}/{m}", t) for m, t in zip(OW.MEMBERS, ws)),
            _meta(f"{MOE}/part_rank{rank}", part_m), _meta(f"{MOE}/out", out_m)]


def test_equal_witnesses_leave_every_tensor_compared(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1]), 1: ([p1, p0], [o1, o0])})
    for s in (0, 1):
        _record_schedule(tmp_path, 0, s, _schedule(s))
    src, rd = _src({0: _step(0, _schedule(0), p0, p1, o1), 1: _step(0, _schedule(1), p1, p0, o0)})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "PASS" and r["compared"] == r["equal"] == 6 and r["exempt"] == 0
    assert r["witness_counts"] == {"equal": 2, "differs": 0, "unpaired": 0} and r["exempt_by"] == []


def test_a_differing_schedule_exempts_its_cone_and_upstream_is_still_compared(tmp_path):
    """Step 0's MoE ran another schedule: its partial and received output differ, and so does all of step 1 (its attention reads the
    KV that output wrote).  o_proj's partial of step 0 is upstream of the align call and is still compared, bit for bit."""
    p0, p1, o0, o1 = _tensors()
    q0, q1, u0, u1 = _tensors(seed=8)
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1]), 1: ([p1, p0], [o1, o0])})
    for s in (0, 1):
        _record_schedule(tmp_path, 0, s, _schedule(s))
    committed = {0: _step(0, _schedule(10), p0, q1, u1), 1: _step(0, _schedule(1), q0, u0, q1)}
    src, rd = _src(committed)
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "PASS", r.get("why")
    assert (r["compared"], r["equal"], r["exempt"], r["mismatch_n"]) == (1, 1, 5, 0)
    assert r["witness_counts"] == {"equal": 1, "differs": 1, "unpaired": 0}
    assert r["exempt_by"] == [{"rank": 0, "step": 0, "k": 1, "module": MOE, "members": list(OW.MEMBERS)}]
    assert r["tensor_runs"] == [[0, "out", 1, 1, "exempt", 0, 1], [0, "partial", 0, 0, "equal", None, 1], [0, "partial", 1, 1, "exempt", 0, 1],
                                [1, "out", 1, 1, "exempt", 0, 1], [1, "partial", 0, 1, "exempt", 0, 2]]
    committed[0][0] = _meta(f"{OPROJ}/part_rank0", q0)
    src, rd = _src(committed)
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["mismatch_n"] == 1 and r["exempt"] == 5
    assert (r["mismatches"][0]["step"], r["mismatches"][0]["k"], r["mismatches"][0]["member"]) == (0, 0, f"{OPROJ}/part_rank0")
    assert "1 of 1 compared tensor(s) differ" in r["why"]


def test_a_tampered_value_under_an_equal_witness_fails(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    _record_schedule(tmp_path, 0, 0, _schedule(0))
    src, rd = _src({0: _step(0, _schedule(0), p0, p1, o0)})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["exempt"] == 0 and r["mismatch_n"] == 1 and r["mismatches"][0]["member"] == f"{MOE}/out"


def test_a_peer_ranks_differing_witness_exempts_this_ranks_received_output_not_its_partial(tmp_path):
    """Rank 1's experts ran another schedule at step 0: every rank's received MoE output differs, and only rank 1's MoE partial.  Judged
    with every rank's witnesses, rank 0's received output is exempt and its own MoE partial, upstream of that collective, still compared."""
    p0, p1, o0, o1 = _tensors()
    q0, q1, u0, _u1 = _tensors(seed=8)
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    _oracle(tmp_path, 1, {0: ([q0, q1], [o0, o1])})
    _record_schedule(tmp_path, 0, 0, _schedule(0))
    _record_schedule(tmp_path, 1, 0, _schedule(1))
    rank1 = _src({0: _step(1, _schedule(11), q0, p0, u0)}, rank=1)
    for part_m, verdict in ((p1, "PASS"), (o1, "FAIL")):
        rank0 = _src({0: _step(0, _schedule(0), p0, part_m, u0)})
        blocks = {r_: s_.compare_match_oracle(str(tmp_path), rd_) for r_, (s_, rd_) in enumerate((rank0, rank1))}
        assert blocks[0]["verdict"] == "FAIL" and blocks[0]["exempt"] == 0          # alone, rank 0 knows only its own witness
        differs = OW.condition_on_witnesses(blocks)
        assert differs == [{"rank": 1, "step": 0, "k": 1, "module": MOE, "members": list(OW.MEMBERS)}]
        assert blocks[0]["verdict"] == verdict and blocks[0]["exempt"] == 1 and blocks[0]["compared"] == 2, blocks[0].get("why")
        assert blocks[0]["tensor_runs"][0] == [0, "out", 1, 1, "exempt", 0, 1] and blocks[0]["exempt_by"] == differs
        assert blocks[1]["verdict"] == "PASS" and blocks[1]["exempt"] == 2 and blocks[1]["compared"] == 1
        assert all("per_tensor" not in b and b["conditioned_on_ranks"] == [0, 1] for b in blocks.values())


def test_a_witness_on_one_side_only_fails_named(tmp_path):
    """A capture that recorded no align witness: the committed one has no counterpart, so no tensor can be exempt by it, and the oracle
    fails by name even though every value is equal."""
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: _step(0, _schedule(0), p0, p1, o1)})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["equal"] == 3 and r["witness_counts"]["unpaired"] == 1
    assert r["witnesses"]["unpaired"] == [{"step": 0, "module": MOE, "in_commit": True, "in_capture": False}] and "on one side only" in r["why"]


@pytest.mark.parametrize("rank,step,k,out,reached", [
    (0, 0, 0, False, False), (0, 0, 0, True, False),     # before the call
    (1, 0, 1, False, True), (0, 0, 1, False, False),     # at its position: the call's own rank's partial, not a peer's
    (0, 0, 1, True, True), (0, 0, 2, False, True),       # the collective that carries it, and every partial after it
    (0, 1, 0, False, True)])                             # every later step
def test_the_cone_of_a_witness(rank, step, k, out, reached):
    assert OW.in_cone({"rank": 1, "step": 0, "k": 1}, rank, step, k, out) is reached
