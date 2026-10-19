"""kernels-dither-rtx's sbench.py on the H200 and the B200, unchanged but for four more arms:

    stock-cutlass  sbench's stock arm without its disable, so vLLM keeps its first int8 choice, CutlassInt8ScaledMMLinearKernel.
                   It runs on sm_90a only (it refuses int8 on sm_100a and sm_120). It applies the weight scale before the
                   activation scale, the reverse of Triton and of the int8-exact path (CA's H1-HB, r20261019-064629-9e21),
                   so Dither's served row pairs with `stock` (Triton) and this arm is reported beside it.
    dither-hb      dither-side with this lane's two-launch unit (hb_side.py, h1u.cu's kh_unit_ws from the .so HB_H1U names)
                   in place of h1r's version 2 on the side stream; gated like every dither arm (its tokens against exact's,
                   `dither_fused.check` on its receipts, the no-write control refused).
    dither-1s-hb   dither-1s (one stream: the served y from the receipted MMAs) with this lane's one-stream unit (hb_one.py,
                   h1u.cu's kh_one from the .so HB_H1U names) in place of h1o_units; gated as dither-1s is
                   (`dither_one.check` with h1o.so's references on its gate, control, fault and fault_wb steps), its
                   tokens reported against exact's, not gated (it serves the dithered product).
    dither-1s-hbp  dither-1s-hb peeled (CA's peel, 11:02Z): hb_one's kh_one_peel (from the .so HB_H1U_PEEL names, else
                   HB_H1U's), whose z is today's A B^T from the same stream, so y is today's; gated by hb_peel.py
                   (`dither_one.check` with z held to A B^T, y equal to today's on every call) on the same four captures,
                   and its tokens must equal exact's: the gate's, and every decode rep's (else exit 4 after the reps).
                   The gate's tokens come from the same generation run first with the window disarmed: in one stream
                   the control serves y from the poisoned slot and both faults move y by design, so the armed
                   generation's tokens can't be exact's; the checks read the armed generation's captures.

sbench's arms run in spawned worker processes, which import this file as their main module before unpickling
`sbench.worker`, so the change below applies in every worker too. Run by hb_served.sh with HB_DITHER_DIR, the source tree's
benchmarks/hash/dither.

    python hb_sbench.py --out OUT [sbench.py's flags]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["HB_DITHER_DIR"])
import sbench  # noqa: E402

_build = sbench.build


def build(arm: str, scratch, gpu_mem: float):
    if arm == "dither-hb":
        import hb_side
        llm = _build("dither-side", scratch, gpu_mem)
        hb_side.install(llm.dither_win, os.environ["HB_H1U"])
        return llm
    if arm == "dither-1s-hb":
        import hb_one
        hb_one.patch(os.environ["HB_H1U"])
        return _build("dither-1s", scratch, gpu_mem)
    if arm == "dither-1s-hbp":
        import hb_one
        import hb_peel
        hb_one.patch(os.environ.get("HB_H1U_PEEL") or os.environ["HB_H1U"], peel=True)
        llm = _build("dither-1s", scratch, gpu_mem)
        llm.dither_mod = hb_peel
        clean_gate_tokens(llm)
        return llm
    if arm != "stock-cutlass":
        return _build(arm, scratch, gpu_mem)
    from vllm import LLM
    os.environ.pop("VLLM_DISABLED_KERNELS", None)
    return LLM(model=str(sbench.stock_checkpoint(sbench.PIN, scratch)), **sbench.engine_args(gpu_mem))


def clean_gate_tokens(llm) -> None:
    """While the window is armed (the gate's generation), `llm.generate` runs the same generation disarmed first and
    returns its outputs, then the armed one, whose captures the checks read."""
    gen, win = llm.generate, llm.dither_win

    def generate(*args, **kwargs):
        if not win.arms:
            return gen(*args, **kwargs)
        arms, win.arms = win.arms, []
        clean = gen(*args, **kwargs)
        win.arms = arms
        gen(*args, **kwargs)
        if win.arms:
            raise RuntimeError(f"the gate's generation left captures {win.arms} unarmed")
        return clean

    llm.generate = generate


def main() -> int:
    rc = sbench.main()
    summary = Path(sys.argv[sys.argv.index("--out") + 1]) / "summary.json"
    if rc or not summary.exists():
        return rc
    eq = json.loads(summary.read_text())["gate_tokens_equal_ref_arm"]
    failed = {arm: eq[arm] for arm in TOKENS_EVERY_REP if arm in eq and not all(eq[arm])}
    if failed:
        print(json.dumps({"reps_tokens_failed": failed}), flush=True)
        return 4
    return 0


sbench.build = build
sbench.REF_OF["dither-hb"] = "exact"
sbench.REF_OF["dither-1s-hb"] = "exact"
sbench.TOKENS_GATED["dither-1s-hb"] = False
sbench.REF_OF["dither-1s-hbp"] = "exact"
TOKENS_EVERY_REP = ("dither-1s-hbp",)

if __name__ == "__main__":
    sys.exit(main())
