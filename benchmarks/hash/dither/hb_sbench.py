"""sbench.py on the H200 and the B200, unchanged but for four more arms:

    stock-cutlass  sbench's stock arm without its disable, so vLLM keeps its first int8 choice, CutlassInt8ScaledMMLinearKernel.
                   It runs on sm_90a only (it refuses int8 on sm_100a and sm_120). It applies the weight scale before the
                   activation scale, the reverse of Triton and of the int8-exact path,
                   so Dither's served row pairs with `stock` (Triton) and this arm is reported beside it.
    dither-hb      dither-side with h1u.cu's two-launch unit (hb_side.py, h1u.cu's kh_unit_ws from the .so HB_H1U names)
                   in place of h1r's version 2 on the side stream; gated like every dither arm (its tokens against exact's,
                   `dither_fused.check` on its receipts, the no-write control refused).
    dither-1s-hb   dither-1s (one stream: the served y from the receipted MMAs) with h1u.cu's one-stream unit (hb_one.py,
                   h1u.cu's kh_one from the .so HB_H1U names) in place of h1o_units; gated as dither-1s is
                   (`dither_one.check` with h1o.so's references on its gate, control, fault and fault_wb steps), its
                   tokens reported against exact's, not gated (it serves the dithered product).
    dither-1s-hbp  dither-1s-hb peeled (the peel): hb_one's kh_one_peel (from the .so HB_H1U_PEEL names, else
                   HB_H1U's), whose z is today's A B^T from the same stream, so y is today's; gated by hb_peel.py
                   (`dither_one.check` with z held to A B^T, y equal to today's on every call) on the same four captures
                   and a fifth, the flipped xa (hb_peel.install), and its tokens must equal exact's: the gate's, and every decode rep's (else exit 4 after the reps).
                   The gate's tokens come from the same generation run first with the window disarmed: in one stream
                   the control serves y from the poisoned slot and both faults move y by design, so the armed
                   generation's tokens can't be exact's; the checks read the armed generation's captures.
    dither-1s-hbk  dither-1s-hbp on the unit's K split (an h1u.cu build with H1U_KSPLIT, from the .so HB_H1U_KSPLIT names):
                   h1k for the calls a warp-a-unit form underfills, h1u for the rest; gated as dither-1s-hbp is.

sbench's dither-a arms (option A, `dither_a`) and their statement-order arms (`dither_so`, -so<D>) run here as
they are, their side unit kh_unit_ws from HB_H1U (hbk's and hbr's from HB_H1U_KSPLIT and HB_H1U_KRONLY), and their
tokens must equal exact's at every decode rep too.

Each arm takes sbench's -xl, -a2 and -a3 suffixes (exactlin's quantizer, the paired A table, its messages in registers;
gated as the arm is).

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
import sbench

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
    if arm in ("dither-1s-hbp", "dither-1s-hbk"):
        import hb_one
        import hb_peel
        so = os.environ["HB_H1U_KSPLIT"] if arm == "dither-1s-hbk" else os.environ.get("HB_H1U_PEEL") or os.environ["HB_H1U"]
        hb_one.patch(so, peel=True)
        hb_peel.install()
        llm = _build("dither-1s", scratch, gpu_mem)
        llm.dither_mod = hb_peel
        llm.dither_win.captures = hb_peel.CAPTURES
        clean_gate_tokens(llm)
        return llm
    if arm != "stock-cutlass":
        return _build(arm, scratch, gpu_mem)
    from vllm import LLM
    os.environ.pop("VLLM_DISABLED_KERNELS", None)
    return LLM(model=str(sbench.stock_checkpoint(sbench.PIN, scratch)), **sbench.stock_args(gpu_mem))


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
    if (rc and rc != 4) or not summary.exists():
        return rc
    eq = json.loads(summary.read_text())["gate_tokens_equal_ref_arm"]
    failed = {arm: eq[arm] for arm in eq if sbench.base_arm(arm) in TOKENS_EVERY_REP and not all(eq[arm])}
    if failed:
        print(json.dumps({"reps_tokens_failed": failed}), flush=True)
        return 4
    return rc


sbench.build = build
sbench.REF_OF["dither-hb"] = "exact"
sbench.REF_OF["dither-1s-hb"] = "exact"
sbench.TOKENS_GATED["dither-1s-hb"] = False
sbench.REF_OF["dither-1s-hbp"] = "exact"
sbench.REF_OF["dither-1s-hbk"] = "exact"
TOKENS_EVERY_REP = ("dither-1s-hbp", "dither-1s-hbk", *sbench.DITHER_A, *sbench.DITHER_SO)

if __name__ == "__main__":
    sys.exit(main())
