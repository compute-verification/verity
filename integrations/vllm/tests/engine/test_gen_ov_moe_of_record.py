"""`test_gen_ov_moe` on what the pod recorded, which no tracked file holds: the router Program against the recorded
`topk_softmax` kernel cases, and the end-to-end resolve / fold / VU-population facts on the wave-1 OLMoE captures. Skipped
where they are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program

from tests.engine.test_gen_ov_moe import OV, TOPK, C, E, H, I, _eval
from verity_vllm.observe.fold.resolver import Compute, Unsupported
from verity_vllm.program.quarantine.ov_moe import moe as Q

REPO = Path(__file__).resolve().parents[2]
W1_RUNS = REPO / "out" / "gen" / "lanes" / "moe" / "runs"
REC_ROUTER = REPO / "out" / "gen" / "lanes" / "ov-moe" / "difftest" / "moe_router_topk"


@pytest.mark.skipif(not (REC_ROUTER / "manifest.json").exists(), reason="the recorded topk_softmax cases are not in this checkout")
def test_router_program_is_bit_exact_on_the_recorded_kernel_cases():
    """The 24 `_moe_C::topk_softmax` cases recorded on the pod (bf16 logits in, f32 weights + i32 ids out): weights AND ids
    bit-exact with `NvExpf` (the correctly rounded stand-in matched 160/192 weight words, REPORT.md §4)."""
    import json
    man = json.loads((REC_ROUTER / "manifest.json").read_text())
    prog = Program(bind(Q.MoeRouterTopK, E=64, TOPK=8, VPT=8))
    for rec in man["cases"]:
        cdir = REC_ROUTER / f"case_{rec['i']}"
        xin = [int(b) for b in np.load(cdir / "in_0.npy").reshape(-1).astype(np.uint16)]
        w = [int(b) for b in np.load(cdir / "out_0.npy").reshape(-1).view(np.uint32)]
        ids = [int(b) for b in np.load(cdir / "out_1.npy").reshape(-1).view(np.uint32)]
        out = _eval(prog, [xin])
        assert out[:8] == w and out[8:] == ids, (rec["i"], rec["inputs"][0].get("kind"))


def _w1_log(name: str) -> Path:
    p = W1_RUNS / name / "capture" / "log.jsonl.gz"
    if not p.is_file():
        pytest.skip(f"wave-1 capture {p} not present")
    return p


@pytest.mark.parametrize("name,steps", [("olmoe_3x1_1req", 1), ("olmoe_1x1_1req", 1)])
def test_real_capture_resolves_without_unsupported(name, steps):
    from verity_vllm.observe.fold.resolve_log import resolve_log
    from verity_vllm.observe.fold.resolver import parts_of
    _, results, _, _ = resolve_log(_w1_log(name), OV.PROFILE)
    assert len(results) == steps
    for sr in results:
        assert sr.error is None
        parts = [p for r in sr.resolutions for p in parts_of(r)]
        assert not [p for p in parts if isinstance(p, Unsupported)]
        specs = [p.spec.id for p in parts if isinstance(p, Compute)]
        assert specs.count(f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}") == C["NL"] and specs.count(f"MoeSum_v1{{TOPK={TOPK},H={H}}}") == C["NL"]


def test_real_capture_folds_the_whole_prefill():
    """The fold binds the whole 3-token prefill (no unresolved read)."""
    from verity_vllm.pipeline.match import fold_summary_result
    summ, _ = fold_summary_result(_w1_log("olmoe_3x1_1req"), "gen_ov_moe_olmoe", with_program=True)
    assert summ["unsupported"] == 0 and summ["unresolved"] == 0 and summ["errors"] == 0 and summ["n_instances"] == 1736
