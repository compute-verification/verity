"""A TP rank Program's `schedule` witness binds, in the Commit's replay, to the reading `MoeAlignRanks_v1` row's own committed rank
(`format.SCHEDULE_PARAM`).  The replay read every parameter but the weights and the sampler's as the prompt, so DeepSeek-V4's align rows
were resolved against `input_token_ids`.  Keyed by its committed rank, the row reproduces that rank exactly when it ranks each expert's
pairs: any order the kernel's atomics can hand out passes, and a rank that is no ranking fails by name."""
from __future__ import annotations

import numpy as np

from verity_vllm.check.replay import evaluate as EV, opening as OP
from verity.ml.definitions.numerics.moe_align_rows import moe_align_ranks

E, P, BM, ROW = 3, 8, 2, 1
EXPERTS = "model.layers.0.ffn.experts"
ALIGN, ROUTER = f"{EXPERTS}/marlin_moe_align", f"{EXPERTS}/router"
IDS = np.array([2, 0, 2, 1, 0, 2, 5, 0], dtype=np.int32)          # one pair routed outside [0, E)
GPU_POS = np.array([7, 1, 5, 3, 0, 6, 9, 2], dtype=np.int32)      # the order sorted_token_ids placed each expert's pairs in
STATICS = {"E": E, "P": P, "BM": BM}
RANK_I, PNAMES = 9, ["prompt", "schedule", "weights"]


class _Index:
    """The rows `_evaluate_row` reads: router row 0 (`out`, the ids), constant rows 10 + p (the pair indices) and the align row."""

    def __init__(self, family="MoeAlignRanks_v1", sched=(0, P)):
        args = [[["n", 0, 0, P]], [["p", 1, *sched]], [["n", 10 + p, 0, 1] for p in range(P)]]
        self.by_i = {RANK_I: {"i": RANK_I, "spec": f"{family}{{E={E},P={P},BM={BM}}}", "args": args},
                     **{10 + p: {"i": 10 + p, "spec": f"Const32[0x{p:08x}]_v1"} for p in range(P)}}
        self.pnames, self.supplied, self.kv_prefix = PNAMES, {}, None
        self._d = {"family": family, "statics": STATICS, "step": 0, "op_path": ALIGN, "row": ROW,
                   "members": [("rank.0", P, "i32"), ("rank.1", P, "i32")]}
        self._router = {"family": "Router_v1", "statics": {}, "step": 0, "op_path": ROUTER, "row": ROW, "members": [("out", P, "i32")]}

    def describe(self, r):
        return self._d if int(r["i"]) == RANK_I else None

    def member_of_port(self, i):
        return {"0": "rank.0", "1": "rank.1"}

    def member_at(self, j, lo, hi):
        return (self._router, "out", lo, hi, "i32") if j == 0 else None


def _store(rank: np.ndarray, start: np.ndarray) -> OP.CommittedStore:
    """ROW 1 of three committed rows of the router's ids and the align's rank / start, as the committer writes them."""
    planes = {(ROUTER, "out"): IDS, (ALIGN, "rank.0"): rank, (ALIGN, "rank.1"): start}
    entries, blobs = [], {}
    for (op, member), row in planes.items():
        words = np.zeros((3, P), dtype=np.uint32)
        words[ROW] = np.asarray(row).astype(np.int64) & 0xFFFFFFFF
        blobs[f"{op}/{member}"] = words.tobytes()
        entries.append({"name": f"{op}/{member}", "element_offset": 0, "encoding": {"dtype": "int32"},
                        "identity": {"request_id": "r0", "step": 0, "op_path": op, "output_member": member, "element_range": [0, 3 * P]}})
    return OP.CommittedStore({"entries": entries}, lambda step, name, b0, b1: blobs[name][b0:b1])


def _gpu() -> tuple[np.ndarray, np.ndarray]:
    got = moe_align_ranks(STATICS, [IDS, GPU_POS, np.arange(P)])
    return got["0"].view(np.int32), got["1"].view(np.int32)


def _replay(pi, rank, start):
    return EV._replay_vu(pi, _store(rank, start), lambda *a: None, {"i": RANK_I, "request_id": "r0", "phase": "decode"}, 0, {})


def test_the_gpus_ranks_replay_with_the_schedule_bound_to_the_rows_own_committed_rank():
    rank, start = _gpu()
    assert rank.tolist() == [2, 1, 0, 0, 0, 1, 0, 2] and start.tolist() == [1, 0, 1, 1, 1, 0, 0, 1]
    rec = _replay(_Index(), rank, start)
    assert rec["result"] is True, rec.get("why")
    wit = [x for x in rec["inputs"] if x["kind"] == "witness"]
    assert wit == [{"kind": "witness", "param": "schedule", "range": [0, P], "identity": ["r0", 0, ALIGN, "rank.0", [ROW * P, (ROW + 1) * P]]}]
    assert not [x for x in rec["inputs"] if x["kind"] == "prompt"]


def test_any_ranking_the_kernel_can_produce_passes():
    rank, start = _gpu()
    other = rank.copy()
    other[[0, 2]] = other[[2, 0]]                       # expert 2's pairs 0 and 2 placed the other way round
    assert _replay(_Index(), other, start)["result"] is True


def test_a_rank_that_ranks_no_order_fails_by_name():
    rank, start = _gpu()
    dup = rank.copy()
    dup[0] = dup[5]                                     # two of expert 2's pairs at one rank
    rec = _replay(_Index(), dup, start)
    assert rec["result"] is False and f"{ALIGN}/rank.0" in rec["why"]


def test_the_schedule_binds_only_a_readers_whole_own_rank():
    rank, start = _gpu()
    rec = _replay(_Index(family="MoeAlignBlocks_v1"), rank, start)
    assert rec["result"] is None and "only a ['MoeAlignRanks_v1'] row's own committed rank" in rec["why"]
    rec = _replay(_Index(sched=(0, P - 1)), rank, start)
    assert rec["result"] is None and "schedule slice [0, 7)" in rec["why"]
