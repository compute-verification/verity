"""TPPartialSource.compare_match_oracle: every committed TP partial / MoE received output of a
rank == the tp2_match capture recorder's stored collective input / output at the same (step, k), bit-for-bit, exhaustively over the family.
Needs torch (torch.save/load of the oracle steps); CPU only.

  * positive: 3 committed members (o_proj partial, MoE partial, MoE received /out) == the stored part/out tensors -> PASS 3/3
  * NEGATIVE late read: the committed MoE /out holds another tensor's bytes -> FAIL naming (step, k, member), 1 mismatch
  * NEGATIVE relabel: the rank's members are tagged part_rank1 -> every oracle collective is missing_in_commit -> FAIL
  * NEGATIVE extra: a committed partial the oracle never recorded -> committed_not_in_oracle -> FAIL (exhaustive or not a record)
  * NEGATIVE no oracle for a committed step -> steps_without_oracle -> FAIL
  * NEGATIVE retained copy changed after the commit (the oracle agrees with the change) -> not_verified -> FAIL
  * MoE align witnesses: nothing is exempt, so every tensor is compared whatever the witnesses say; a differing witness FAILS by name
    with what it implies (a routing difference, or routing or schedule), even when every value is equal; a witness on one side only
    FAILS; a schedule difference alone never reads as routing

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
from verity_vllm.commit.step_rows import ROW_BYTES  # noqa: E402
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
    assert r["by_member"] == {"partial": {"tensors": 2, "equal": 2}, "out": {"tensors": 1, "equal": 1}}
    assert r["compared"] == 3 and "exempt" not in r and r["witness_counts"] == {"equal": 0, "differs": 0, "unpaired": 0}
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


# ---------------------------------------------------------------------------------------------- the oracle comparison's three refusals

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
    p0, p1, o0, o1 = _tensors(rows=ROW_BYTES // 16)                            # a row each: one leaf per member
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


# ---------------------------------------------------------------------------------------------- the MoE align witnesses

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
    assert r["verdict"] == "PASS" and r["compared"] == r["equal"] == 6
    assert r["witness_counts"] == {"equal": 2, "differs": 0, "unpaired": 0} and r["witness_differs"] == {"n": 0, "routing": 0, "first": []}


def test_a_differing_witness_fails_by_name_and_every_tensor_is_still_compared(tmp_path):
    """Step 0's MoE align call differs from the capture's in every member, par among them (a routing difference: par is the per-expert
    counts' alone), and the MoE partial and received output it produced differ, as does all of step 1.  Nothing is exempt: every tensor
    is compared, and the witness is the first fault named."""
    p0, p1, o0, o1 = _tensors()
    q0, q1, u0, u1 = _tensors(seed=8)
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1]), 1: ([p1, p0], [o1, o0])})
    for s in (0, 1):
        _record_schedule(tmp_path, 0, s, _schedule(s))
    src, rd = _src({0: _step(0, _schedule(10), p0, q1, u1), 1: _step(0, _schedule(1), q0, u0, q1)})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and (r["compared"], r["equal"], r["mismatch_n"]) == (6, 1, 5)
    w = {"rank": 0, "step": 0, "k": 1, "module": MOE, "members": list(OW.MEMBERS), "implies": OW.ROUTING}
    assert r["witnesses"]["differs"] == [w] and r["witness_differs"] == {"n": 1, "routing": 1, "first": [w]}
    assert r["tensor_runs"] == [[0, "out", 1, 1, "mismatch", 1], [0, "partial", 0, 0, "equal", 1], [0, "partial", 1, 1, "mismatch", 1],
                                [1, "out", 1, 1, "mismatch", 1], [1, "partial", 0, 1, "mismatch", 2]]
    assert r["why"].startswith("1 MoE align witness(es) differ from the capture's, 1 implying a routing difference") and "implies routing)" in r["why"]
    assert "also 5 of 6 compared tensor(s) differ" in r["why"]


def test_a_schedule_difference_with_every_value_equal_still_fails_named(tmp_path):
    """Only align.rank differs (a pair's order within its expert, which the schedule or the routing can change) and every value is
    equal: where the Commit did not replay the capture's schedule (no schedule file, or a routing difference) the oracle cannot say
    which, and FAILS naming it."""
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    ws = _schedule(0)
    _record_schedule(tmp_path, 0, 0, ws)
    src, rd = _src({0: _step(0, [_schedule(9)[0], *ws[1:]], p0, p1, o1)})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["compared"] == r["equal"] == 3 and r["mismatch_n"] == 0
    assert r["witnesses"]["differs"][0]["members"] == ["align.rank"] and r["witness_differs"]["routing"] == 0
    assert "implies routing or schedule" in r["why"] and "also" not in r["why"]


def test_a_tampered_value_under_an_equal_witness_fails(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    _record_schedule(tmp_path, 0, 0, _schedule(0))
    src, rd = _src({0: _step(0, _schedule(0), p0, p1, o0)})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["mismatch_n"] == 1 and r["mismatches"][0]["member"] == f"{MOE}/out"


def test_a_peer_ranks_differing_witness_fails_that_rank_and_the_tensor_it_changed_here(tmp_path):
    """Rank 1's experts ran under another align at step 0: rank 1 FAILS by name, and rank 0, whose own witness is equal, compares the
    received MoE output that align changed and FAILS on it; its own MoE partial, upstream of that collective, is equal."""
    p0, p1, o0, o1 = _tensors()
    q0, q1, u0, _u1 = _tensors(seed=8)
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    _oracle(tmp_path, 1, {0: ([q0, q1], [o0, o1])})
    _record_schedule(tmp_path, 0, 0, _schedule(0))
    _record_schedule(tmp_path, 1, 0, _schedule(1))
    (s1, rd1), (s0, rd0) = _src({0: _step(1, _schedule(11), q0, p0, u0)}, rank=1), _src({0: _step(0, _schedule(0), p0, p1, u0)})
    r1, r0 = s1.compare_match_oracle(str(tmp_path), rd1), s0.compare_match_oracle(str(tmp_path), rd0)
    assert r1["verdict"] == "FAIL" and r1["witness_counts"]["differs"] == 1 and r1["why"].startswith("1 MoE align witness(es) differ")
    assert r0["verdict"] == "FAIL" and r0["witness_counts"] == {"equal": 1, "differs": 0, "unpaired": 0}
    assert r0["mismatch_n"] == 1 and r0["mismatches"][0]["member"] == f"{MOE}/out" and r0["equal"] == 2


def test_a_witness_on_one_side_only_fails_named(tmp_path):
    """A capture that recorded no align witness: the committed one has no counterpart, and the oracle fails by name even though every
    value is equal."""
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: _step(0, _schedule(0), p0, p1, o1)})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "FAIL" and r["equal"] == 3 and r["witness_counts"]["unpaired"] == 1
    assert r["witnesses"]["unpaired"] == [{"step": 0, "module": MOE, "in_commit": True, "in_capture": False}] and "on one side only" in r["why"]


@pytest.mark.parametrize("members,implies", [
    (["align.par"], OW.ROUTING), (["align.blk"], OW.ROUTING), (["align.start"], OW.ROUTING), (["align.start", "align.blk"], OW.ROUTING),
    (list(OW.MEMBERS), OW.ROUTING), (["align.rank"], OW.ROUTING_OR_SCHEDULE), (["align.rank", "align.start", "align.blk"], OW.ROUTING_OR_SCHEDULE)])
def test_what_a_differing_witness_implies(members, implies):
    assert OW.implied(members) == implies


def test_a_schedule_difference_alone_never_reads_as_routing():
    """Two atomic orders of one routing, the kernel's own freedom (`moe_align_source.align_values` over the emulated kernel): whatever
    members differ, `implied` says routing or schedule, never routing."""
    from tests.acquire.test_dsv4_moe_align_source import kernel
    from verity_vllm.acquire.sources import moe_align_source as MA

    g, E, BM, seen = torch.Generator().manual_seed(5), 6, 4, set()
    for P in range(2, 30):
        ids = torch.randint(0, E, (P,), generator=g, dtype=torch.int32)
        ws = []
        for _ in range(2):
            *vals, ok = MA.align_values(ids.reshape(-1, 1), *kernel(ids.tolist(), E, BM, torch.randperm(P, generator=g).tolist()), E, BM)
            assert bool(ok)
            ws.append([tps._bytes(v.view(torch.float32)) for v in vals])
        differs = [m for m, a, b in zip(OW.MEMBERS, *ws) if a != b]
        if differs:
            assert OW.implied(differs) == OW.ROUTING_OR_SCHEDULE, (ids.tolist(), differs)
            seen.add(tuple(differs))
    assert seen
