"""Red-team repro (circuits W6, claim 2), kept as a regression test: the match oracle must not read a routing change as a schedule
difference.

The witness-conditioned oracle exempted every tensor in the cone of an align call whose committed witness (rank, start, blk, par)
differed from the capture's.  Those four members are functions of the router's top-k ids as well as of atomicAdd's order, so a
committed routing that is not the capture's also made the witness differ, and exempted the MoE output it changed and everything after
it.  At a B1 decode (6 pairs on 6 distinct experts) the schedule has no freedom at all: rank is 0, start 1 and par 6 for every routing,
and blk differs exactly when the ids do.  A DeepSeek-V4-Pro Commit showed it on rank 0: 928 decode-step
witnesses differ in align.blk alone, and 52 of step 0's in align.par, which only the per-expert counts decide.  The oracle now exempts
nothing and fails a differing witness by name with what it implies (`oracle_witness.implied`)."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tests.check.opened import commit_steps  # noqa: E402
from verity_vllm.acquire.sources import moe_align_source as MA  # noqa: E402
from verity_vllm.acquire.sources import oracle_witness as OW  # noqa: E402
from verity_vllm.acquire.sources import partial_source as tps  # noqa: E402
from verity_vllm.check import oracle_compare as OC  # noqa: E402

OPROJ, MOE = "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.experts"
E, BM = 384, 8
HONEST = [3, 17, 40, 41, 200, 301]
ROUTED = [3, 17, 40, 41, 200, 10]           # one expert swapped for a lower-numbered one: a different routing, no other change


class RowParallelLinear:
    reduce_results, tp_size = True, 2

    def parameters(self):
        return iter(())


class FusedMoE:
    reduce_results, tp_size, moe_config = True, 2, None

    def parameters(self):
        return iter(())


def _kernel_align(ids: list[int]) -> tuple[torch.Tensor, ...]:
    """`moe_align_block_size`'s outputs for one token's pairs in index order within each expert (one schedule the kernel can give)."""
    P = len(ids)
    slots, experts = [], []
    for e in sorted(set(ids)):
        mine = [p for p, x in enumerate(ids) if x == e]
        for b in range(0, len(mine), BM):
            blk = mine[b:b + BM]
            slots += blk + [P] * (BM - len(blk))
            experts.append(e)
    return (torch.tensor(slots, dtype=torch.int32), torch.tensor(experts, dtype=torch.int32), torch.tensor([len(slots)], dtype=torch.int32))


def _witness(ids: list[int]) -> list[torch.Tensor]:
    sorted_ids, expert_ids, ntpp = _kernel_align(ids)
    rank, start, blk, par, ok = MA.align_values(torch.tensor([ids], dtype=torch.int32), sorted_ids, expert_ids, ntpp, E, BM)
    assert bool(ok)
    return [t.view(torch.float32) for t in (rank, start, blk, par)]


def _tensors(seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 8, generator=g).to(torch.bfloat16) for _ in range(4)]


def _oracle(cap: Path, steps: dict) -> None:
    cap.mkdir(parents=True, exist_ok=True)
    with (cap / "coll.rank0.jsonl").open("w") as f:
        for s, (parts, _outs) in steps.items():
            for k, mod in enumerate([OPROJ, MOE]):
                f.write(json.dumps({"rank": 0, "step": s, "k": k, "kind": "all_reduce", "module": mod, "shape": list(parts[k].shape)}) + "\n")
    for s, (parts, outs) in steps.items():
        torch.save({"step": s, "part": parts, "out": outs}, cap / f"coll.rank0.step{s}.pt")


def _record_witness(cap: Path, step: int, ws: list[torch.Tensor]) -> None:
    with (cap / OW.witness_file(0)).open("a") as f:
        for m, t in zip(OW.MEMBERS, ws):
            f.write(json.dumps({"rank": 0, "step": step, "k": 1, "module": MOE, "member": m, "sha256": OW.digest(tps._bytes(t))}) + "\n")


def _step(ws, part_o, part_m, out_m) -> list:
    return [(f"{OPROJ}/part_rank0", tps._bytes(part_o)), *((f"{MOE}/{m}", tps._bytes(t)) for m, t in zip(OW.MEMBERS, ws)),
            (f"{MOE}/part_rank0", tps._bytes(part_m)), (f"{MOE}/out", tps._bytes(out_m))]


def test_a_decode_routing_change_differs_in_blk_alone():
    """At 6 pairs on 6 distinct experts the schedule has no freedom: only blk can differ, and it differs exactly when the routing does."""
    honest, routed = _witness(HONEST), _witness(ROUTED)
    same = [tps._bytes(a) == tps._bytes(b) for a, b in zip(honest, routed)]
    assert dict(zip(OW.MEMBERS, same)) == {"align.rank": True, "align.start": True, "align.blk": False, "align.par": True}


def test_a_routing_change_under_an_equal_upstream_is_a_mismatch_not_an_exemption(tmp_path):
    """Step 0 is equal in both runs, schedule included.  At step 1 the Commit routed the token to another expert and committed the MoE
    partial and received output that routing gives; o_proj's partial before the align call is equal.  Nothing upstream differs, so the
    changed MoE output has no schedule to blame: it is a mismatch, and the witness FAILS by name as a routing difference."""
    p0, p1, o0, o1 = _tensors(7)
    q0, q1, u0, u1 = _tensors(8)
    r1, v1 = _tensors(9)[:2]
    _oracle(tmp_path, {0: ([p0, p1], [o0, o1]), 1: ([q0, q1], [u0, u1])})
    for s in (0, 1):
        _record_witness(tmp_path, s, _witness(HONEST))
    com, run = commit_steps({0: _step(_witness(HONEST), p0, p1, o1), 1: _step(_witness(ROUTED), q0, r1, v1)})
    src = tps.TPPartialSource(com, types.SimpleNamespace(named_modules=lambda: [(OPROJ, RowParallelLinear()), (MOE, FusedMoE())]), rank=0, world=2)
    r = src.compare_match_oracle(str(tmp_path), OC.OpenedReader(com, run))
    assert r["witness_counts"] == {"equal": 1, "differs": 1, "unpaired": 0}
    assert r["witnesses"]["differs"][0]["members"] == ["align.blk"]
    assert r["verdict"] == "FAIL" and r["mismatch_n"] >= 1, (r["verdict"], r.get("compared"), r.get("exempt"), r.get("exempt_by"))
    assert r["witnesses"]["differs"][0]["implies"] == OW.ROUTING and "implying a routing difference" in r["why"]
