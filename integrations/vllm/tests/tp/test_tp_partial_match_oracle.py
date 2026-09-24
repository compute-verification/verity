"""[R15 tp, coord M-0481 C2 form (B), TP family] TPPartialSource.compare_match_oracle: every committed TP partial / MoE received output of a
rank == the tp2_match capture recorder's stored collective input / output at the same (step, k), bit-for-bit, exhaustively over the family.
Needs torch (torch.save/load of the oracle steps); CPU only.

  * positive: 3 committed members (o_proj partial, MoE partial, MoE received /out) == the stored part/out tensors -> PASS 3/3
  * NEGATIVE late read: the committed MoE /out holds another tensor's bytes -> FAIL naming (step, k, member), 1 mismatch
  * NEGATIVE relabel: the rank's members are tagged part_rank1 -> every oracle collective is missing_in_commit -> FAIL
  * NEGATIVE extra: a committed partial the oracle never recorded -> committed_not_in_oracle -> FAIL (exhaustive or not a record)
  * NEGATIVE no oracle for a committed step -> steps_without_oracle -> FAIL
  * NEGATIVE retained copy changed after the commit (the oracle agrees with the change) -> not_verified -> FAIL

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
from verity_vllm.tp import partial_source as tps  # noqa: E402

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


def _tensors(rows: int = 4):
    g = torch.Generator().manual_seed(7)
    return [torch.randn(rows, 8, generator=g).to(torch.bfloat16) for _ in range(4)]


def test_positive_three_members_equal_the_stored_collectives(tmp_path):
    p0, p1, o0, o1 = _tensors()
    _oracle(tmp_path, 0, {0: ([p0, p1], [o0, o1])})
    src, rd = _src({0: [_meta(f"{OPROJ}/part_rank0", p0), _meta(f"{MOE}/part_rank0", p1), _meta(f"{MOE}/out", o1)]})
    r = src.compare_match_oracle(str(tmp_path), rd)
    assert r["verdict"] == "PASS" and r["tensors"] == 3 and r["equal"] == 3 and r["steps_compared"] == 1
    assert r["by_member"] == {"partial": {"tensors": 2, "equal": 2}, "out": {"tensors": 1, "equal": 1}}
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
