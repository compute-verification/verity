"""Dither's served standard number on one GPU: each arm a resident vLLM engine in its own worker process on the same GPU,
the parent alternating reps between them (arm order rotated each rep), on int8-exact Llama-3.1-8B (the verity-int8 pin,
served_h1's row) and served_h1's prompts (`tests.engine.int8_exact_gpu.PROMPTS`, cycled).

Measures, per arm and rep:
    tpot_b32_ms      decode at batch 32: 32 requests, greedy, ignore_eos; (t(N2) - t(N1)) / (N2 - N1) over two generate calls
                     of N1 and N2 output tokens, so prefill and the engine's per-call overhead cancel.  $SBENCH_B=B
                     (B <= --t) decodes B requests instead, every decode step then of m = B: tpot_b<B>_ms, the gate's
                     B requests and its checked step at m = B, and smq_sweep's and prof's rows at B
    tokens_per_s     at saturation: SAT_REQS requests of SAT_TOKENS output tokens each, max_num_seqs MAX_SEQS (requests wait
                     in the scheduler throughout), output tokens over wall time; under --stock-own-batch the stock arms
                     run at vLLM's own max_num_seqs and max_num_batched_tokens, not --t's (summary's scheduler_by_arm,
                     with each engine's KV tokens: --arm-mem gives an arm the memory share its batch needs).  With sat
                     in --ops the warm-up runs one such generate first (its batches' graphs and compiles, a -we arm's
                     spill pool grown to that window: about 218 MB a step at m 256 under -wea, so $DITHER_SPILL_MAX_GB
                     and the lease's memory sized for T 256's ~800 steps), and each rep's tokens are held to its
                     reference arm's (summary's gate_tokens_equal_ref_arm, under ops=sat alone)
and the gate: each arm's greedy token ids on the 32 requests equal `stock`'s (an arm that changes no served value must).

Arms:
    stock        native vLLM int8: the pin's weights and scales under a compressed-tensors W8A8 config (int8 channel weights,
                 dynamic per-token int8 activations), so vLLM serves them with its own kernels, compiled with CUDA graphs
                 (vLLM's defaults).  On sm_120 vLLM's first int8 choice, CutlassInt8ScaledMMLinearKernel, refuses at its first
                 call ("Int8 not supported on SM120"), so the arm disables it (VLLM_DISABLED_KERNELS) and vLLM's selector takes
                 its next, TritonInt8ScaledMMLinearKernel (scaled_int8_quant + triton_scaled_mm)
    stock-humming  the same with Triton disabled too: vLLM's third choice, HummingInt8ScaledMMLinearKernel
    stock-cutlass  the same with nothing disabled: CutlassInt8ScaledMMLinearKernel where it serves int8 (sm_90, vLLM's
                 default there; it applies the weight scale first, so it is not the exact epilogue's order)
    stock-percall  stock with triton_scaled_mm behind a custom op (`exactlin/percall.py`), so its tile heuristic runs on each
                 call's M, not once at the traced M; the same values, so its tokens must equal stock's (gated), and its
                 stats are the op's calls by M
    exact-eager  served_h1's engine without the option: `engine.build.build_engine` on the int8-exact profile (int8_exact=all),
                 as SERVE-H1 runs it (eager, batch-invariant)
    exact        the int8-exact linears (`verity_int8::row_quant` + `verity_int8::scaled_mm`, torch._int_mm) compiled with
                 CUDA graphs, attention vLLM's: today's served y in a compiled engine (built here, outside `build_engine`,
                 which runs int8_exact eagerly only; both ops are custom ops, opaque to Inductor)
    exact-fused  exact with both ops' CUDA bodies `benchmarks/hash/exactlin`'s two Triton kernels (`exactlin.install`): the
                 same values, so its tokens must equal exact's (gated)
    dither       exact with every linear's y from Dither's fused GEMM (`dither_fused`: z = xq W^T and its dequantization in
                 one kernel that also computes the previous step's receipts; A, admit and z tables, the round, the live
                 coin and the seeds between steps)
    dither-side  dither with the receipts in one `h1r` launch a step on a second stream, after the step's coin, beside the
                 next step (`dither_fused.patch(rc="side")`); the `h1f` launches compute z and y only
    dither-1s    exact with every linear's y from Dither's one stream (`dither_one`, `h1o.cu`, H1V3): y = dequant(O' -
                 sum_d 128^d D_d) from the noised mainloop the call's receipts fold, z that O; the call's A table, seeds
                 (the window's salt and root(A_j), root(B_j): epsilon 'seeded') and units in the forward; admit and z
                 tables, the step trees, the round and the receipt trees between steps.  It serves the dithered product, so
                 its tokens are not exact's (reported, not gated)
    dither-1s-credit  dither-1s on an eager engine with every linear's `apply` inside the credit gate
                 (`credit_gate.Trace`, the file DITHER_CREDIT_GATE names): the gate reports its verdicts (`credit_2684`)
    dither-1s-warp  dither-1s with its units a warp each (DITHER_UNITS=warp, `h1o_units`), not a block each
    dither-1s-split  dither-1s with its units in the split form (DITHER_UNITS=split, `h1o_unitss`: six warps a unit)
    dither-1s-cpeel  dither-1s with the credited peel (DITHER_PEEL=1, `h1o_unitsbp`; H1V3's
                 PouwPeelSeg): z = O + 256 X wbᵀ - X cᵀ - (b - 256 xa) Bᵀ = A Bᵀ from the receipted stream's own O, so y is
                 today's served y and its tokens must equal exact's; the receipts are dither-1s's
    dither-1s-split-cpeel  dither-1s-cpeel in the split form (`h1o_unitssp`)
    dither-1s-block2-cpeel, dither-1s-split2-cpeel  those two on the kernels `h1o_init` gives the maximum shared-memory
                 carveout (DITHER_UNITS=block2 / split2, `h1o_unitsb2p` / `h1o_unitss2p`: two units' blocks an SM, not
                 the driver's default one; block2 with immediate barrier ids), the same code and values
    dither-1s-mix-cpeel  dither-1s-cpeel with a units form per call (DITHER_UNITS=mix, `dither_one.MIX_FORMS`): block2
                 where the call's column tiles outnumber the SMs (qkv, gate_up), split elsewhere (o, down)
    dither-1s-percall0  dither-1s with the per-call challenger round between each call's A table and its units (`h1o_percall`:
                 root(A) to the host, a host function, a coin back, all captured in the graphs) and no wait: its plumbing
    dither-1s-percall-rtt  the same with the host function holding each call one measured challenger round trip
                 (DITHER_PERCALL_WAITS: paced round-trip samples from the serving node, replayed in order)
    dither-a     option A, the provisional credit form (`dither_a`; every dither arm labelled by
                 `forms.form_of`: form=`forms.FORM_A` for -so<D> with -zy on -aom/-aomn without -ov, form=beside
                 (unadopted arm) for -so's -ab/-abn and -ov, form=beside for every other arm, these and dither-1s's and
                 dither's included): exact-fused's
                 quantizer and GEMM serve y = dequant(z) and commit z, and a side stream runs each
                 call's A table (form 3), seeds and receipts of the noised product only; its tokens must equal exact's.
                 dither-a-h1r takes h1r's block or split receipts per call, dither-a-kh h1u.cu's kh_unit_ws (HB_H1U),
                 dither-a the second where HB_H1U is set, dither-a-hbk the K split's kh_one (HB_H1U_KSPLIT) for its
                 receipts only, dither-a-hbr the same call built H1K_RONLY (HB_H1U_KRONLY: no z, y or their MMA), dither-a-hbc
                 -hbr's call from a second receipts-only build (HB_H1U_KC: sbench.sh's HB_KC_FLAGS, a candidate timed beside
                 -hbr in one run and held to its every gate), with its own libh1c.so where DITHER_H1C_KC names one
                 (sbench.sh's HB_KC_H1C_FLAGS) and its own DITHER_ZY1 where DITHER_ZY1_KC is set (1: -zy's z and y
                 tables in one h1c_build, `dither_a.ZY1_ENV`).  Each with -zy commits the per-unit z/y leaves (one hiding leaf a
                 32 x 32 unit holding its z and its y = dequant(z): the epilogue writes the records, the window
                 builds every call's table between steps),
                 and with -y commits no z at all (y only: the leaves' cost floor, never credited; its gate is its
                 tokens).  With -ao after those, the A tables run on a stream of
                 their own, each beside the previous call's units (the units of a call still wait on its A table);
                 -aos puts each call's seeds there too, after its A table (-kh, -hbk and -hbr, whose seeds are a launch of
                 their own); -aom and -aoms are -ao and -aos on four such streams, call w's on stream w % 4 (no
                 call's A table reads another's); -aomn (-hbk, -hbr) is -aoms with each call's noise there as well, so
                 the side stream runs the units alone.  With -pt last, between steps the receipt trees and the admit tables run on streams of
                 their own beside the z tables and the step trees (`DitherState.tree_streams`); with -ls after that, each step's
                 seeds read the live salt of the round before it (`DitherState.step_salt`)
    dither-a-so  dither-a-hbk-zy, dither-a-hbr-zy and dither-a-hbc-zy (-hbr's call from dither-a-hbc's builds,
                 HB_H1U_KC, DITHER_H1C_KC and DITHER_ZY1_KC) in the statement's order (`dither_so`): the
                 forward computes and commits a step's A, scales and z/y leaves, round s goes out before coin s, and step
                 s's seeds and units run after coin s on a ring of D steps' A slots, beside a later step's forward, their
                 receipts under round s + D - 1: dither-a-{hbk,hbr,hbc}-zy-{aom,aomn,ab,abn}[-pt]-so<D> (D 2 to 10), -aom
                 the A tables per call on four streams in the forward, -aomn with each call's noise on four streams
                 beside the units, -ab and -abn the step's A tables in one build between steps (not ruled in: a
                 separate arm); its gate adds the order's checks and two faults (fault_rearly, fault_zlate).  With
                 -ov last (-ov1: a ring of one step; -ovp: the engine on a stream above the trees' and the units'
                 priority), round s's trees build beside step s + 1's forward from ring copies
                 of what forward s wrote and step s's units go out as soon as coin s is back; its gate adds that round s
                 covers forward s's values while built once forward s + 1 had started, and fault_ring.  With -we
                 last, each window ends (`SoWindow.end`: the tail's units after their coins, their receipts under a
                 final round, the manifest root, the sampling coin after it): in process at the end of each generate,
                 inside its time (the next window's challenger made after it), its gate adds the ended window's
                 checks and two end faults (fault_tail, fault_late_receipt, `dither_so.check_end`); off the node the
                 session's one window ends at its layout's last step (DITHER_CHALLENGERS' "rounds"), served there at
                 the arm's stop, its manifest in OUT (`dither_manifest`) and its checks in stop.json.  Under -we each
                 step's live A rows, scales and z/y records go to pinned host memory as their ring slot retires and
                 stay until the window's sample (`dither_so.Spill`, about 291 MB a step at T 64, m 32; the pool grown
                 in the warm-up, bounded by $DITHER_SPILL_MAX_GB), and the ended window's check opens three of its
                 steps from them; -wea (`-so<D>[-ov...]-wea`) holds A and the scales alone (about 27 MB a step), z/y
                 recomputed from them at an opening
    ARM-xl       any dither arm with `verity_int8::row_quant`'s CUDA body exactlin's quantizer (`exactlin.install`, which
                 serves `scaled_mm` from exactlin's GEMM too; the dither arms call row_quant only): the same xq and s_a, so
                 ARM's gate and reference arm, and its tokens gated as ARM's are; sbench.sh and hb_served.sh run exactlin's
                 kgate.py in the lease before an -xl or exact-fused arm is timed
    ARM-a2       any dither arm with each call's A table from `h1c_a_call2_k` (DITHER_A_FORM=2: the same table, its
                 compressions paired); ARM-a3 from `h1c_a_call3_k` (DITHER_A_FORM=3: paired, its messages in registers);
                 the suffixes compose (ARM-a3-xl), each gated as ARM
    ARM-j10r     a dither-a-so arm with its units under J10-R's call-scale noise (DITHER_SO_UNITS=j10r, `dither_so`)
    ARM-smqQ, ARM-smqQoD  a dither-a-so arm with its units on the SMs whose id % D < Q (DITHER_SO_UNITS_SMQ=Q/D, D 8
                 without it); -smq0 on every SM, whatever the run's environment sets; ARM-smqQoDcC with at most C of
                 the units' CTAs on each of those SMs (Q/D/C; -smq1o1c2 every SM, 2 CTAs each).  These set the arm's
                 own worker's environment, so one run times the units' forms and shares side by side (ARM-j10r-smq9o16)
    ARM-zy1      a dither-a-hbc arm with DITHER_ZY1_KC=1 (its z and y tables in one h1c_build), whatever the run's
                 environment sets
    ARM-kcb      a dither-a-hbc arm on a second h1u_kc build ($HB_H1U_KC_B, e.g. hb_served.sh's HB_KC_FLAGS_B), timed beside
                 ARM on $HB_H1U_KC's
    ARM-cgp      a dither-a-so arm serving at priority (DITHER_SO_PRIO=1) with vLLM's graphs captured at it too
                 (DITHER_SO_CAPTURE_PRIO=1, `dither_so.capture_at`).  Under SMQ units it keeps every step in the fast
                 regime while the units' span plus lm_head's 0.7 ms covers the step's cycle: without it, a step whose
                 units drain before the next dispatch lets them refill their SMs ahead of lm_head, which then runs on
                 the forward's SMs alone (2.9 ms, not 0.7) and the next steps stay slow.  Under it, h1u's H1U_JXS units
                 stall on some steps (b32 and b64: their span about 2.3 times the median, the serving stream waiting
                 about 20 ms on them), which their length doesn't explain: H1U_JPI's units, as long, never do, nor do
                 H1U_JCS=2's, shorter; DITHER_SO_PREQ=0 makes the stalls rarer without removing them
    ARM-mc32     a dither arm whose worker has 32 hardware queues (CUDA_DEVICE_MAX_CONNECTIONS=32, set before its engine
                 makes the CUDA context)
    ARM-skip, ARM-rd0  a dither-a-so J10-R arm whose dead row tiles are the zero receipt (DITHER_UNITS_SKIP_DEAD=1), or
                 one with DITHER_SO_RDEAD=0 (J10-R's default computes them in full, which $DITHER_SO_RDEAD=1 refuses)
    ARM-lag1, ARM-lag2  a dither-a-so arm with DITHER_SO_ROOT_LAG=1 or 2
    ARM-ug0      a dither-a-so arm whose units are launched each step, not replayed as a CUDA graph (DITHER_SO_GRAPH=0)
    ARM-zs       a dither-a-so arm whose z tables build on a tree stream beside its A tables (DITHER_SO_ZSIDE=1)
    ARM-tail2    a dither-a-so arm without -ov whose mid and round trees build in one launch after the units' wait, the
                 round's root read from pinned host memory that launch writes (DITHER_SO_TAIL=2, h1c's `Tail`)
    ARM-preq     a dither-a-so arm whose units are queued as their graph before the round's root read, behind a wait on a
                 pinned host word `SoSide.dispatch` sets once the round is sent and its coin is back (DITHER_SO_PREQ=1):
                 the units start without the host's dispatch between that word and them
    ARM-mmc      a dither-a arm whose GEMMs take $SBENCH_MM_CFG's configs ($DITHER_MM_CFG, dmmsweep.py's best.json)
    ARM-mmq      a dither-a-so arm whose GEMMs take the configs tuned at its own SMQ (MM_CFGS at $DITHER_SO_UNITS_SMQ's
                 Q/D, however its suffixes or SBENCH_ARM_ENV set it), refused at an SMQ MM_CFGS has none for.  J10-R
                 at 47/64: dither-a-hbc-zy-aomn-pt-so8-wea-rd0-smq47o64-mmq with DITHER_SO_UNITS=j10r and HB_KC_FLAGS
                 ending -DH1U_JCTA=1 -DH1U_WARPS=4 (sbench.sh's dither-a-hbc build); at 45/64 under ARM-cgp:
                 dither-a-hbc-zy-aomn-pt-so8-wea-rd0-smq45o64-cgp-mmq, the same build

`gate` (--gate, before the reps): exact and dither each generate GATE_TOKENS tokens for the 32 requests; dither's window
checks one B32 decode step (`dither_fused.check`, its transcript buffers 0xA5 before the kernels wrote them) and refuses
its no-write control (a later B32 step whose kernels write none of them), and its tokens must equal exact's (dither-1s:
`dither_one.check` on its gate, control, fault and fault_wb steps, and peeled fault_xa, tokens reported), under a CUDA
profiler range and the NVTX range GATE_RANGE/<arm> (nsys --capture-range=cudaProfilerApi; kernels.py counts the dither worker's
kernels inside the range only).  The decode gate compares each arm's
tokens with its reference arm's (dither: exact; any other: the first arm); a dither arm's are a second GATE_TOKENS
generation after its checks with nothing armed, since its fault steps move served values on purpose (fault_wb moves
column 0 of every call's y at one step), the capture run's kept as tokens_capture.  A failed gate times nothing (exit 4); a worker
that errs or dies ends the run (exit 5, its gates so far in gate.json), except that under --time-passed one that errs in its
warm or gate is dropped, its error in gate.json, and the rest timed (exit 5 still).  Each dither arm's gate captures are freed once
checked, so N arms need N engines' memory plus one arm's captures.

`prof` (an op, for any arm; under DITHER_NSYS=1 or prof): generates PROF_N[0] and then PROF_N[1] tokens for the 32 requests
($SBENCH_PROF_REQS of them if set: T for decode steps of m = T, a saturation step; its step is then step_m<N>_ms) inside
one CUDA profiler range, each generation in its own NVTX range PROF_RANGE/<arm>/n<N>, so a kernel's launches and time
a decode step are the two ranges' difference over PROF_N[1] - PROF_N[0] (`exactlin/steps.py`; `stepprof.py` splits a dither
arm's step by part, its between-step phases by their NVTX ranges).  nsys's capture range ends at the first profiler stop,
so a run profiles one arm: under DITHER_NSYS=prof the gate runs unprofiled, so a gated run's prof op is the capture.

--prefill: every request a T-token prompt (`prefill_prompts`, token ids cycled from the prompts) with one output token, so
at max_num_batched_tokens = T each request is one step of m = T, a whole prompt in one chunk (max_model_len T + 64).
`prefill` (an op): PF[0] and then PF[1] requests a generate, prefill_step_ms their difference over PF[1] - PF[0] (each
generate's own overhead and, in process, its window's end cancel), and ttft_ms the median of PF_TTFT one-request
generates (llm.generate alone; ttft_end_ms with an in-process -we window's end after it).  `prof` takes PF's counts of
requests in place of PROF_N's tokens.  The gate arms PF_CAPTURES at m = T (the decode gate's faults are not armed) in a
generate of len(PF_CAPTURES) + gate_extra requests, each capture copied to host memory (dither_so.CAP_HOST_ENV) and its
leaves hashed in a pool (dither_a.CHECK_WORKERS_ENV, default PF_CHECK_WORKERS); its verdict `prefill_verdict`, and the
token gate a generation of GATE_TOKENS tokens for PF_GATE_REQS prefill requests (chunked prefill, then decode), equal to
the reference arm's.  Under -we the end gate is not run.  A Dither window at T = 2048 needs most of a 96 GB GPU, so a
prefill row runs --serial: each arm's engine alone, built, gated, timed and stopped before the next (its reference first).

Coins: a dither arm's window takes its round coins from the challenger off the node when its worker has a session
(DITHER_CHALLENGERS' entry for the arm, ``{"ARM": {"session": ID, "address": "127.0.0.1:PORT", "secret_file": F,
"commitment": HEX, "rounds": R}}``, each its own from the challenger's gate, `coins.py open` (R its --rounds, the layout a
dither_so window takes), else $POUW_CHALLENGER, $POUW_CHALLENGER_SESSION,
$POUW_CHALLENGER_SECRET_FILE and $POUW_CHALLENGER_COMMITMENT), one session a window, opened at the window's build (its
hello's commitment held to HEX) and closed at the arm's stop; without one they are in-process, and the arm's stats say
``coins: in-process (not certified)``.

SBENCH_ARM_ENV (``{"ARM": {"NAME": "VALUE"}}``) sets an arm's own environment in its worker, so the arms of one lease can
differ in a knob the worker reads; each arm's ready line says what it set (``arm_env``).  ``ARM@TAG`` in --arms is ARM
built and timed under its own name (its own SBENCH_ARM_ENV, --arm-mem and DITHER_CHALLENGERS entries), so one arm is
timed beside itself with a knob set.  SBENCH_TIME_AT=HH:MM (UTC) holds the reps until then (a booked window's start),
the engines built and gated before it; sbench.sh's SBENCH_TIMED=1 takes the lease such a window grants (gpu-lease --timed).

--bw-passes GIB runs the remote bandwidth audit's passes beside the arms on their GPU once every arm is built, warm and
gated (`dbw.py`: the prover's calibration beside the dither arms' decode ops, an idle session, each rep an off block of
every arm's ops and an on block inside a session's passes, and a tampered session beside each dither arm; summary.json's
"bw", per_arm's TPOT the off blocks' alone).  Not with --serial, whose arms are never up together.

    python sbench.py --out OUT [--arms stock,exact,dither] [--reps 3] [--t 64] [--ops decode] [--gate] [--bw-passes GIB]
    python sbench.py --out OUT --prefill --serial --t 2048 --arms exact,exact-fused,ARM --ops prefill --gate
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import statistics
import sys
import time
from pathlib import Path

from verity_config.settings import of

settings = of(__file__)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7"
B_ENV = "SBENCH_B"
B = settings.get(B_ENV)
TPOT = f"tpot_b{B}_ms"
N1, N2 = 32, 160
SAT_REQS, SAT_TOKENS, MAX_SEQS, MAX_TOKENS_STEP = 768, 256, 256, 1024
CT_CONFIG = {
    "quant_method": "compressed-tensors", "format": "int-quantized", "quantization_status": "compressed",
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False,
                    "group_size": None, "observer": "minmax", "actorder": None, "block_structure": None},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True,
                              "group_size": None, "observer": None, "actorder": None, "block_structure": None}}},
    "ignore": ["lm_head"], "kv_cache_scheme": None}


def prompts(n: int) -> list[str]:
    sys.path.insert(0, str(ROOT / "integrations" / "vllm"))
    from tests.engine.int8_exact_gpu import PROMPTS
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


PREFILL_ENV = "SBENCH_PREFILL"
PF = (8, 24)
PF_TTFT = 3
PF_GATE_REQS = 8
PF_CAPTURES = ("gate", "control")
PF_CHECK_WORKERS = 32


def prefill() -> bool:
    return settings.get(PREFILL_ENV)


def prefill_prompts(tok, n: int, t: int) -> list[dict]:
    """n prompts of t token ids each, request i's from offset 61 i of the prompts' ids cycled (no two alike)."""
    ids = tok.encode(" ".join(prompts(64)))
    need = t + 61 * n
    ids = (ids * (need // len(ids) + 1))[:need]
    return [{"prompt_token_ids": ids[61 * i:61 * i + t]} for i in range(n)]


def prefill_verdict(gate: dict, control: dict) -> dict:
    """The prefill gate's verdict: the gate step passes (its order included) and its no-write control is refused on z
    and on the receipts, as `dither_a.gate_verdict` holds them."""
    refused = bool(control.get("complete") and not control.get("ok") and control.get("z_vs_int_mm_bad")
                   and control.get("receipts_vs_h1r_bad"))
    return {"ok": bool(gate.get("ok")) and refused, "gate_ok": bool(gate.get("ok")), "control_refused": refused,
            "order_ok": bool(gate.get("order", {}).get("ok")), "faults": "the decode gate's, not armed at prefill",
            "control_bad": {k: len(control.get(k) or []) for k in ("z_vs_int_mm_bad", "receipts_vs_h1r_bad",
                                                                    "a_roots_bad", "z_roots_bad")}}


def stock_checkpoint(pin: str, scratch: Path) -> Path:
    """The pin's files linked into SCRATCH, its config's quantization_config replaced by compressed-tensors W8A8's."""
    d = scratch / "stock-ckpt"
    d.mkdir(parents=True, exist_ok=True)
    for f in Path(pin).iterdir():
        if f.name != "config.json" and not (d / f.name).exists():
            (d / f.name).symlink_to(f.resolve())
    cfg = json.loads((Path(pin) / "config.json").read_text())
    cfg["quantization_config"] = CT_CONFIG
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    return d


def engine_args(gpu_mem: float, own_batch: bool = False) -> dict:
    # FlashInfer's autotune off, as `engine.build` sets it for the int8-exact engine (it tunes no kernel these engines run)
    # vLLM refuses max_num_seqs above max_num_batched_tokens; every arm of a run gets the same cap, but a stock arm under
    # --stock-own-batch, which keeps vLLM's own max_num_seqs and max_num_batched_tokens
    t = MAX_TOKENS_STEP if settings.SBENCH_T is None else settings.SBENCH_T
    a = {"max_num_seqs": min(MAX_SEQS, t), "max_num_batched_tokens": t,
         "gpu_memory_utilization": gpu_mem,
         "enable_prefix_caching": False, "max_model_len": max(2048, t + 64) if prefill() else 2048, "seed": 0,
         "kernel_config": {"enable_flashinfer_autotune": False}}
    if own_batch:
        del a["max_num_seqs"], a["max_num_batched_tokens"]
    return a


OWN_BATCH_ENV = "SBENCH_STOCK_OWN_BATCH"


def stock_args(gpu_mem: float) -> dict:
    return engine_args(gpu_mem, own_batch=settings.get(OWN_BATCH_ENV))


def scheduler_of(llm) -> dict | None:
    """The engine's resolved batch and its KV cache's tokens: what a stock arm under --stock-own-batch really ran at,
    and whether its memory share, not its scheduler, bounded its concurrency at saturation."""
    try:
        vc = llm.llm_engine.vllm_config
        sc, cc = vc.scheduler_config, vc.cache_config
        return {"max_num_seqs": sc.max_num_seqs, "max_num_batched_tokens": sc.max_num_batched_tokens,
                "gpu_memory_utilization": cc.gpu_memory_utilization,
                "kv_tokens": cc.num_gpu_blocks * cc.block_size if cc.num_gpu_blocks else None}
    except AttributeError:
        return None


def arm_mem(spec: str) -> dict[str, float]:
    """--arm-mem ARM=F,...: those arms' gpu_memory_utilization in place of --gpu-mem."""
    return {k: float(v) for k, v in (x.split("=") for x in spec.split(",") if x)}


DISABLED = {"stock": "CutlassInt8ScaledMMLinearKernel", "stock-percall": "CutlassInt8ScaledMMLinearKernel",
            "stock-humming": "CutlassInt8ScaledMMLinearKernel,TritonInt8ScaledMMLinearKernel"}


def kernels_of(llm) -> dict:
    """The linear kernels the engine's layers really run: class name -> layers."""
    from verity_vllm.engine import handle
    seen: dict = {}
    for _, mod in handle.model_runner(llm).get_model().named_modules():
        k = getattr(getattr(mod, "scheme", None), "kernel", None)
        q = getattr(mod, "quant_method", None)
        name = type(k).__name__ if k is not None else (type(q).__name__ if q is not None and hasattr(mod, "weight") else None)
        if name:
            seen[name] = seen.get(name, 0) + 1
    return seen


def install_exactlin() -> None:
    sys.path.insert(0, str(HERE.parent / "exactlin"))
    import exactlin
    if not getattr(exactlin, "_sbench_installed", False):
        exactlin.install()
        exactlin._sbench_installed = True


def a_form(f: int):
    def set_form() -> None:
        os.environ["DITHER_A_FORM"] = str(f)
    return set_form


def env_of(name: str, value: str | None):
    def set_env() -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return set_env


def kc_b() -> None:
    if not settings.HB_H1U_KC_B:
        raise ValueError("an ARM-kcb arm needs $HB_H1U_KC_B, a second h1u_kc build")
    os.environ["HB_H1U_KC"] = settings.HB_H1U_KC_B


def mm_cfg() -> None:
    if not settings.SBENCH_MM_CFG:
        raise ValueError("an ARM-mmc arm needs $SBENCH_MM_CFG, a $DITHER_MM_CFG text (dmmsweep.py's best.json)")
    os.environ["DITHER_MM_CFG"] = settings.SBENCH_MM_CFG


# ARM-mmq's $DITHER_MM_CFG by the units' "Q/D" ($DITHER_SO_UNITS_SMQ's): dmmsweep.py --block Q/D's best.json
# dither_mm_cfg, the m=64 entries the 11/16 sweep's.  The forward's GEMMs run on the SMs the units leave them, so a config
# holds at the SMQ it was tuned at alone (at 47/64, 11/16's configs cost 1.7 ms a step).  45/64 takes 47/64's: dmmsweep
# --block 45/64's own best, timed with the units beside it, slowed both the forward and the units (18.25 ms, not 16.8)
MM_CFGS = {"47/64": "6144,4096,32=32x32x128:w4:s3:k1+4096,4096,32=32x32x256:w4:s3:k1:t1+28672,4096,32=32x64x128:w4:s2:k1"
                    "+4096,14336,32=32x32x256:w4:s3:k1+4096,4096,64=32x64x256:w8:s4:k3+4096,14336,64=32x64x512:w8:s3:k3"}
MM_CFGS["45/64"] = MM_CFGS["47/64"]
MMQ = {"on": False}


def mm_smq() -> None:
    MMQ["on"] = True


def smq_mm_cfg() -> None:
    """An ARM-mmq arm's $DITHER_MM_CFG, MM_CFGS' at its SMQ, once its suffixes and SBENCH_ARM_ENV have set that."""
    if not MMQ["on"]:
        return
    q, _, rest = (settings.DITHER_SO_UNITS_SMQ or "").partition("/")
    smq = f"{q}/{rest.partition('/')[0] or 8}" if q and q != "0" else "every SM"
    if smq not in MM_CFGS:
        raise ValueError(f"an ARM-mmq arm has no MM_CFGS entry at its SMQ ({smq}): tune one with dmmsweep.py --block Q/D "
                         f"at that SMQ and add its best.json's dither_mm_cfg to sbench.MM_CFGS")
    os.environ["DITHER_MM_CFG"] = MM_CFGS[smq]


SUFFIXES = {"-xl": install_exactlin, "-a2": a_form(2), "-a3": a_form(3), "-j10r": env_of("DITHER_SO_UNITS", "j10r"),
            "-kcb": kc_b, "-zy1": env_of("DITHER_ZY1_KC", "1"), "-cgp": env_of("DITHER_SO_CAPTURE_PRIO", "1"),
            "-mc32": env_of("CUDA_DEVICE_MAX_CONNECTIONS", "32"), "-skip": env_of("DITHER_UNITS_SKIP_DEAD", "1"),
            "-rd0": env_of("DITHER_SO_RDEAD", "0"), "-lag1": env_of("DITHER_SO_ROOT_LAG", "1"),
            "-lag2": env_of("DITHER_SO_ROOT_LAG", "2"), "-ug0": env_of("DITHER_SO_GRAPH", "0"),
            "-zs": env_of("DITHER_SO_ZSIDE", "1"), "-tail2": env_of("DITHER_SO_TAIL", "2"), "-mmc": mm_cfg,
            "-preq": env_of("DITHER_SO_PREQ", "1"), "-mmq": mm_smq}
# -smqQ / -smqQoD / -smqQoDcC: a dither-a-so arm's units on the SMs whose id % D < Q, at most C CTAs on each
# ($DITHER_SO_UNITS_SMQ=Q/D/C, D 8 and no cap without them; -smq0 every SM, whatever the run's environment says)
SMQ_SUFFIX = re.compile(r"-smq(\d+)(?:o(\d+))?(?:c(\d+))?$")


def suffix_of(arm: str) -> str | None:
    if arm.startswith("dither"):
        for s in SUFFIXES:
            if arm.endswith(s):
                return s
        if m := SMQ_SUFFIX.search(arm):
            return m.group(0)
    return None


def suffix_env(s: str):
    if s in SUFFIXES:
        return SUFFIXES[s]
    m = SMQ_SUFFIX.fullmatch(s)
    return env_of("DITHER_SO_UNITS_SMQ", None if m[1] == "0" else f"{m[1]}/{m[2] or 8}" + (f"/{m[3]}" if m[3] else ""))


def base_arm(arm: str) -> str:
    arm = arm.split("@")[0]
    s = suffix_of(arm)
    return base_arm(arm[:-len(s)]) if s else arm


def build(arm: str, scratch: Path, gpu_mem: float):
    from vllm import LLM
    arm = arm.split("@")[0]
    s = suffix_of(arm)
    if s:
        suffix_env(s)()
        return build(arm[:-len(s)], scratch, gpu_mem)
    smq_mm_cfg()
    if arm in DISABLED:
        os.environ["VLLM_DISABLED_KERNELS"] = DISABLED[arm]
    if arm == "stock-cutlass":
        os.environ.pop("VLLM_DISABLED_KERNELS", None)
    if arm in ("stock", "stock-humming", "stock-cutlass"):
        return LLM(model=str(stock_checkpoint(PIN, scratch)), **stock_args(gpu_mem))
    if arm == "stock-percall":
        sys.path.insert(0, str(HERE.parent / "exactlin"))
        import percall
        percall.install()
        llm = LLM(model=str(stock_checkpoint(PIN, scratch)), **stock_args(gpu_mem))
        llm.dither_stats = percall.summary
        return llm
    if arm == "exact-eager":
        vllm_dir = ROOT / "integrations" / "vllm"
        sys.path.insert(0, str(vllm_dir))
        from tests.engine import int8_exact_gpu as G
        from verity_vllm.engine.build import build_engine
        from verity_vllm import TargetProfile
        target = TargetProfile.from_device().with_overrides(int8_exact="all")
        ns = argparse.Namespace(checkpoints=str(vllm_dir / "manifests" / "checkpoints.json"), role="LLAMA31_8B_INT8",
                                base_role="LLAMA31_8B", repo="verity-int8/Llama-3.1-8B-INT8",
                                hf_home=settings.HF_HOME, by="kernels", at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        entry, _ = G.pin(ns)
        ckpts = G.scratch_checkpoints(ns.checkpoints, entry, scratch)
        a = engine_args(gpu_mem)
        eng = build_engine(ckpts, case=ns.role, target=target, max_num_seqs=a.pop("max_num_seqs"), engine_args=a)
        return eng.llm
    if arm in ("exact", "exact-fused", "dither", "dither-side"):
        sys.path.insert(0, str(HERE))
        import dither_fused as DF
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
        if arm == "exact-fused":
            install_exactlin()
            llm = LLM(model=str(DF.int8_checkpoint(PIN, scratch)), **engine_args(gpu_mem))
            return llm
        a = engine_args(gpu_mem)
        if arm != "exact":
            DF.patch(t=a["max_num_batched_tokens"], h1c=settings.need("DITHER_H1C"), h1f=settings.need("DITHER_H1F"), gate=True,
                     rc="side" if arm == "dither-side" else "fused")
        llm = LLM(model=str(DF.int8_checkpoint(PIN, scratch)), **a)
        if arm != "exact":
            win = DF.hook(llm)
            llm.dither_stats = win.stats
            llm.dither_win = win
        return llm
    if arm in ("dither-1s", "dither-1s-credit", "dither-1s-warp", "dither-1s-split", "dither-1s-percall0",
               "dither-1s-percall-rtt", "dither-1s-cpeel", "dither-1s-split-cpeel", "dither-1s-block2-cpeel",
               "dither-1s-split2-cpeel", "dither-1s-mix-cpeel"):
        sys.path.insert(0, str(HERE))
        import dither_fused as DF
        import dither_one as DO
        if arm in ("dither-1s-warp", "dither-1s-split", "dither-1s-split-cpeel", "dither-1s-block2-cpeel",
                   "dither-1s-split2-cpeel", "dither-1s-mix-cpeel"):
            os.environ[DO.UNITS_ENV] = arm.split("-")[2]
        if arm.endswith("-cpeel"):
            os.environ[DO.PEEL_ENV] = "1"
        if arm == "dither-1s-percall0":
            os.environ[DO.PERCALL_ENV] = "0"
        if arm == "dither-1s-percall-rtt":
            os.environ[DO.PERCALL_ENV] = settings.need("DITHER_PERCALL_WAITS")
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
        a = engine_args(gpu_mem)
        credit = settings.need("DITHER_CREDIT_GATE") if arm == "dither-1s-credit" else None
        if credit:
            a["enforce_eager"] = True
        DO.patch(t=a["max_num_batched_tokens"], h1c=settings.need("DITHER_H1C"), h1o=settings.need("DITHER_H1O"), gate=True,
                 credit=credit)
        llm = LLM(model=str(DF.int8_checkpoint(PIN, scratch)), **a)
        win = DO.hook(llm)
        llm.dither_stats, llm.dither_win, llm.dither_mod = win.stats, win, DO
        return llm
    if arm in DITHER_A:
        sys.path.insert(0, str(HERE))
        import dither_a as DA
        import dither_fused as DF
        import dither_one as DO
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
        h1u = settings.HB_H1U
        unit, z, ao, pt, ls = DITHER_A[arm]
        unit = unit or ("kh" if h1u else "h1r")
        if unit in DA.KONE:
            h1u = settings.need("HB_H1U_KSPLIT" if unit == "hbk" else
                                "HB_H1U_KC" if arm.startswith("dither-a-hbc") else "HB_H1U_KRONLY")
        h1c = settings.need("DITHER_H1C")
        if arm.startswith("dither-a-hbc"):
            h1c = settings.DITHER_H1C_KC or h1c
            if settings.raw(DA.ZY1_ENV + "_KC") is not None:
                os.environ[DA.ZY1_ENV] = settings.raw(DA.ZY1_ENV + "_KC")
        a = engine_args(gpu_mem)
        DA.patch(t=a["max_num_batched_tokens"], h1c=h1c, h1o=settings.need("DITHER_H1O"), unit=unit,
                 h1u=h1u, z=z, ao=ao, pt=pt, ls=ls)
        llm = LLM(model=str(DF.int8_checkpoint(PIN, scratch)), **a)
        win = DO.hook(llm)
        win.captures = DA.CAPTURES
        # the y-only floor commits no z, so it has no window gate: its gate is its tokens (TOKENS_ONLY)
        llm.dither_stats, llm.dither_win, llm.dither_mod = DA.stats(win), (win if z != "none" else None), DA
        return llm
    if arm in DITHER_SO:
        sys.path.insert(0, str(HERE))
        import dither_fused as DF
        import dither_one as DO
        import dither_so as SO
        from verity_vllm.engine import int8_vllm
        int8_vllm.register()
        unit, ao, pt, depth, ov, serve_prio, we, spill = DITHER_SO[arm]
        a = engine_args(gpu_mem)
        # hbc: hbr's call from HB_H1U_KC's build, with DITHER_H1C_KC's libh1c.so and DITHER_ZY1_KC where set (as dither-a-hbc)
        h1c = (settings.DITHER_H1C_KC or settings.need("DITHER_H1C")) if unit == "hbc" else settings.need("DITHER_H1C")
        if unit == "hbc" and settings.raw("DITHER_ZY1_KC") is not None:
            os.environ["DITHER_ZY1"] = settings.raw("DITHER_ZY1_KC")
        SO.patch(t=a["max_num_batched_tokens"], h1c=h1c, h1o=settings.need("DITHER_H1O"), unit="hbr" if unit == "hbc" else unit,
                 h1u=settings.need({"hbk": "HB_H1U_KSPLIT", "hbr": "HB_H1U_KRONLY", "hbc": "HB_H1U_KC"}[unit]), ao=ao, pt=pt,
                 depth=depth, ov=ov, serve_prio=serve_prio, window_end=we, spill=spill)
        llm = LLM(model=str(DF.int8_checkpoint(PIN, scratch)), **a)
        win = DO.hook(llm)
        win.captures = SO.CAPTURES
        llm.dither_stats, llm.dither_win, llm.dither_mod = SO.stats(win), win, SO
        return llm
    raise ValueError(arm)


DITHER_A = {f"dither-a{u}{z}{o}{p}{q}": (unit, zm, ao, pt, ls)
            for u, unit in (("", None), ("-h1r", "h1r"), ("-kh", "kh"), ("-hbk", "hbk"), ("-hbr", "hbr"), ("-hbc", "hbr"))
            for z, zm in (("", "row"), ("-zy", "zy"), ("-y", "none"))
            for o, ao in (("", ""), ("-ao", "a"), ("-aos", "as"), ("-aom", "m"), ("-aoms", "ms"), ("-aomn", "mn"))
            for p, pt in (("", False), ("-pt", True))
            for q, ls in (("", False), ("-ls", True))}
# -ov: round s and step s's units beside step s + 1's forward, a round ring of 2 steps (-ov1: 1); -ovp: -ov with the
# engine serving on a stream above the trees' and the units' priority; -we: each window ends (`dither_so.SoWindow.end`),
# each step's A and z/y records held in pinned host memory until its sample (`dither_so.Spill`), -wea its A alone
DITHER_SO = {f"dither-a-{unit}-zy-{o}{p}-so{d}{v}{e}": (unit, ao, pt, d, r, sp, we, spill)
             for unit in ("hbk", "hbr", "hbc") for o, ao in (("aom", "m"), ("aomn", "mn"), ("ab", "b"), ("abn", "bn"))
             for p, pt in (("", False), ("-pt", True)) for d in range(2, 11)
             for v, r, sp in (("", 0, False), ("-ov", 2, False), ("-ov1", 1, False), ("-ovp", 2, True))
             for e, we, spill in (("", False, None), ("-we", True, "azy"), ("-wea", True, "a"))}


REF_OF = {"dither": "exact", "dither-side": "exact", "dither-1s": "exact", "dither-1s-credit": "exact",
          "dither-1s-warp": "exact", "dither-1s-split": "exact", "dither-1s-percall0": "exact",
          "dither-1s-percall-rtt": "exact", "dither-1s-cpeel": "exact", "dither-1s-split-cpeel": "exact",
          "dither-1s-block2-cpeel": "exact", "dither-1s-split2-cpeel": "exact", "dither-1s-mix-cpeel": "exact",
          "exact-fused": "exact", "stock-percall": "stock", **{x: "exact" for x in (*DITHER_A, *DITHER_SO)}}
# an arm with no window check whose gate is its tokens: equal to its reference arm's or the run times nothing
TOKENS_ONLY = {"exact-fused", "stock-percall", *(x for x, (_, z, *_) in DITHER_A.items() if z == "none")}
def token_diff(a: list, b: list) -> dict:
    """Two arms' greedy outputs for the same requests: how many are equal, and each other one's first differing position."""
    first = sorted(next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), min(len(p), len(q)))
                   for p, q in zip(a, b) if p != q)
    return {"requests_equal": len(a) - len(first), "first_diff": first}


PROF_N = (8, 24)
PROF_RANGE = "verity/exactlin/prof"
# an arm that serves another value: its tokens are compared, not required equal (an arm not listed: required)
TOKENS_GATED = {"dither-1s": False, "dither-1s-credit": False, "dither-1s-warp": False, "dither-1s-split": False,
                "dither-1s-percall0": False, "dither-1s-percall-rtt": False}
# SBENCH_PICK=first|last: every engine's Inductor autotune choices made untimed (`picks.py`); each arm's picks are
# logged either way (gate.json's full log, each rep's digest).  SBENCH_PICK_ARMS=PREFIX[,PREFIX]: only the arms whose
# names start with one are pinned (exact,dither: the arms held to exact's tokens), the rest keep their timed picks
PICK_ENV = "SBENCH_PICK"
PICK_ARMS_ENV = "SBENCH_PICK_ARMS"
# SBENCH_SMQ_SWEEP=V[,V]: after the gate, each dither-a-so arm's decode TPOT and phases at each units SM share V (q,
# q/den or q/den/cap, 0 every SM; `dither_so.SoSide.set_units_smq`), in one engine, then back to its own share for the
# reps: a screen (smq_sweep.json), never a row.  With --ops sat alone, its tokens/s at saturation (one sat generate a
# share) in place of the TPOT
SMQ_SWEEP_ENV = "SBENCH_SMQ_SWEEP"


def pick_of(arm: str) -> str | None:
    only = settings.get(PICK_ARMS_ENV)
    return settings.get(PICK_ENV) if not only or arm.startswith(tuple(only)) else None
GATE_TOKENS = 16    # enough B32 decode steps for the gate's capture, its receipts, the control's and its receipts
# {arm: {"session": ID, "address": HOST:PORT, "secret_file": F, "commitment": HEX}}: each dither arm's window its own
# challenger session off the node (its POUW_CHALLENGER, POUW_CHALLENGER_SESSION, POUW_CHALLENGER_SECRET_FILE and
# POUW_CHALLENGER_COMMITMENT), since every arm's engine, and so its window, lives through the whole run
CHALLENGERS_ENV = "DITHER_CHALLENGERS"
# {arm: {NAME: VALUE}}: an arm's own environment, set in its worker before its engine is built, so one lease times arms
# that differ in a knob the worker reads (not sbench.sh's builds, which read the run's)
ARM_ENV_ENV = "SBENCH_ARM_ENV"
# "HH:MM" (UTC): the reps start no earlier than then, at most TIME_AT_MAX_MIN minutes after the gates end, so the engines
# build and gate before a booked window and only the reps run in it
TIME_AT_ENV = "SBENCH_TIME_AT"
TIME_AT_MAX_MIN = 20
GATE_RANGE = "verity/dither/gate"   # GATE_RANGE/<arm>: kernels.py reads each dither worker's kernels inside its own


def wait_time_at() -> dict | None:
    """$SBENCH_TIME_AT's wait before the reps: until that UTC time when it is at most TIME_AT_MAX_MIN minutes ahead."""
    at = settings.get(TIME_AT_ENV)
    if not at:
        return None
    import datetime as dt
    now = dt.datetime.now(dt.UTC)
    h, m = (int(x) for x in at.split(":"))
    wait = (now.replace(hour=h, minute=m, second=0, microsecond=0) - now).total_seconds()
    if wait < -43200:
        wait += 86400
    waited = wait if 0 < wait <= 60 * TIME_AT_MAX_MIN else 0.0
    time.sleep(waited)
    return {"at": at, "ahead_s": round(wait, 1), "waited_s": round(waited, 1),
            "reps_start_utc": dt.datetime.now(dt.UTC).strftime("%H:%M:%SZ")}


def close_coins() -> None:
    """End this worker's window's off-node coin session, if it has one: the challenger answers closed (after every
    round a `coins.RoundPipe` sent has its coin back, and a dither_so -ov window's rounds and units drained; a dither_so
    window closes itself, `SoWindow.close`, ending it under -we)."""
    for name in ("dither_fused", "dither_one"):
        win = getattr(sys.modules.get(name), "WIN", None)
        if win is not None and hasattr(win, "close"):
            win.close()
        elif win is not None:
            getattr(win, "drain", lambda: None)()
            (getattr(win, "pipe", None) or win.coins).close()


def fill_end(llm, gen) -> dict:
    """A -we arm's off-node window served to its layout's last step at the stop (one request a generate, its prompt one
    step's prefill, so a generate of n tokens is n steps), where it ends on its own; then its `check_end`."""
    win, mod = getattr(llm, "dither_win", None), getattr(llm, "dither_mod", None)
    if win is None or not getattr(win, "window_end", False) or win.end_each or not win.open:
        return {}
    t0, s0 = time.perf_counter(), win.s
    while win.open:
        n = win.steps_left()
        gen(["Count to ten."], n - 8 if n > 16 else 1)
    return {"fill_steps": win.s - s0, "fill_s": round(time.perf_counter() - t0, 1),
            "end": mod.end_checks(win, settings.need(mod.LIB_ENV)), "stats": llm.dither_stats()}


def worker(arm: str, conn, scratch: str, gpu_mem: float) -> None:
    settings.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    at = (settings.get(CHALLENGERS_ENV) or {}).get(arm)
    if at:
        os.environ["POUW_CHALLENGER"], os.environ["POUW_CHALLENGER_SESSION"] = at["address"], at["session"]
        if at.get("secret_file"):
            os.environ["POUW_CHALLENGER_SECRET_FILE"] = at["secret_file"]
        if at.get("commitment"):
            os.environ["POUW_CHALLENGER_COMMITMENT"] = at["commitment"]
        if at.get("rounds"):
            os.environ["DITHER_ROUNDS"] = str(at["rounds"])
    arm_env = (settings.get(ARM_ENV_ENV) or {}).get(arm) or {}
    os.environ.update(arm_env)
    import torch
    from vllm import SamplingParams
    if arm.startswith("dither"):
        sys.path.insert(0, str(HERE))
        import dither_fused as DF
    sys.path.insert(0, str(HERE))
    import picks as PK
    plog = PK.install(pick_of(arm))
    t0 = time.perf_counter()
    try:
        llm = build(arm, Path(scratch), gpu_mem)
    except Exception as e:  # noqa: BLE001
        conn.send({"error": f"{type(e).__name__}: {e}"})
        return
    conn.send({"ready": True, "build_s": round(time.perf_counter() - t0, 1), "kernels": kernels_of(llm),
               "disabled": settings.VLLM_DISABLED_KERNELS, "scheduler": scheduler_of(llm),
               "arm_env": arm_env or None})
    ps = prompts(B)
    pf = prefill()
    pp = prefill_prompts(llm.get_tokenizer(), max(PF[1], PF_GATE_REQS, 64), settings.need("SBENCH_T")) if pf else None
    if pf and base_arm(arm) in DITHER_SO:
        import dither_a as DA
        import dither_so as SO
        os.environ[SO.CAP_HOST_ENV] = "1"
        settings.setdefault(DA.CHECK_WORKERS_ENV, str(PF_CHECK_WORKERS))
    last: dict = {}

    def gen(reqs: list, n: int):
        # a -we arm's in-process window ends with each generate, inside its time; the next one's challenger is made after
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, seed=0)
        win = getattr(llm, "dither_win", None)
        we = win is not None and getattr(win, "end_each", False)
        t = time.perf_counter()
        outs = llm.generate(reqs, sp, use_tqdm=False)
        last["generate_s"] = time.perf_counter() - t
        if we:
            with torch.inference_mode():
                win.end()
        dt = time.perf_counter() - t
        if we:
            win.prepare()
        return dt, outs

    def serve(cmd: dict) -> None:
        if cmd["op"] == "warm" and pf:
            # every step of m = T, a -we arm's spill pool grown to a timed window's (PF[1] steps)
            gen(pp[:4], 1)
            gen(pp[:PF[1]], 1)
            conn.send({"warm": True})
        elif cmd["op"] == "warm":
            gen(ps, 8)
            gen(prompts(64), 16)
            win = getattr(llm, "dither_win", None)
            if cmd.get("sat"):
                gen(prompts(SAT_REQS), SAT_TOKENS)
            elif getattr(win, "spill_mode", None) and getattr(win, "end_each", False):
                # a -we arm's spill pool grown here to a timed window's size (N2 steps), not inside rep 0
                gen(ps, N2)
            conn.send({"warm": True})
        elif cmd["op"] == "prefill":
            t1, _ = gen(pp[:PF[0]], 1)
            t2, outs = gen(pp[:PF[1]], 1)
            tt, te = [], []
            for _ in range(PF_TTFT):
                te.append(gen(pp[:1], 1)[0])
                tt.append(last["generate_s"])
            conn.send({"prefill_step_ms": round(1e3 * (t2 - t1) / (PF[1] - PF[0]), 4), "t_pf0_s": round(t1, 4),
                       "t_pf1_s": round(t2, 4), "ttft_ms": round(1e3 * statistics.median(tt), 4),
                       "ttft_all_ms": [round(1e3 * x, 4) for x in tt],
                       "ttft_end_ms": round(1e3 * statistics.median(te), 4),
                       "tokens": [list(o.outputs[0].token_ids) for o in outs]})
        elif cmd["op"] == "decode":
            m0 = time.monotonic_ns()
            t1, _ = gen(ps, N1)
            m1 = time.monotonic_ns()
            t2, outs = gen(ps, N2)
            conn.send({TPOT: round(1e3 * (t2 - t1) / (N2 - N1), 4), "t_n1_s": round(t1, 4), "t_n2_s": round(t2, 4),
                       "spans_ns": [m0, m1, time.monotonic_ns()], "tokens": [list(o.outputs[0].token_ids) for o in outs]})
        elif cmd["op"] == "gate":
            win = getattr(llm, "dither_win", None)
            mod = getattr(llm, "dither_mod", None) or (DF if win is not None else None)
            we = win is not None and getattr(win, "end_each", False)
            if pf and win is not None and mod is DF:
                raise ValueError(f"{arm}: --prefill gates dither_one's windows (dither-a, dither-1s), not dither_fused's")
            if win is not None:
                if mod is DF:
                    win.arm(B)
                    win.price_y = True
                elif pf:
                    win.arm(settings.need("SBENCH_T"), PF_CAPTURES)
                else:
                    win.arm(B)
                if we and not pf:
                    win.arm_end("end_gate")
            prof = settings.DITHER_NSYS == "1" and win is not None
            if prof:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(f"{GATE_RANGE}/{arm}")
            # a window whose captures' receipts go under later rounds (dither_so) runs that many steps more
            if pf and win is not None:
                t, outs = gen(pp[:len(PF_CAPTURES) + getattr(win, "gate_extra", 0)], 1)
            elif pf:
                t, outs = gen(pp[:PF_GATE_REQS], GATE_TOKENS)
            else:
                t, outs = gen(ps, GATE_TOKENS + getattr(win, "gate_extra", 0))
            if prof:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
            res = {"gate_s": round(t, 3), "tokens": [list(o.outputs[0].token_ids) for o in outs], "picks": plog}
            if win is not None:
                t0 = time.perf_counter()
                if mod is DF:
                    win.price_y = False
                    with torch.inference_mode():
                        res["check"] = DF.check(win, settings.need("DITHER_H1F"), "gate")
                        res["control"] = DF.check(win, settings.need("DITHER_H1F"), "control")
                    res["verdict"] = DF.gate_verdict(res["check"], res["control"])
                else:
                    lib = settings.need(mod.LIB_ENV)
                    names = PF_CAPTURES if pf else win.captures
                    with torch.inference_mode():
                        caps = {n: mod.check(win, lib, n) for n in names}
                    res["check"] = caps.pop("gate")
                    res.update(caps)
                    res["verdict"] = (prefill_verdict(res["check"], caps["control"]) if pf else
                                      mod.gate_verdict(res["check"], *(caps[n] for n in win.captures[1:])))
                    if getattr(mod, "CG", None) is not None:
                        res["credit_2684"] = mod.credit_doc()
                    if we and not pf:
                        # the gate's window ended with its generate; the end faults each in a window of their own
                        mod.end_gate(win, lambda n: gen(ps, n))
                        res["end"] = mod.end_checks(win, lib)
                        res["verdict"]["end"] = res["end"]["verdict"]
                        res["verdict"]["ok"] = res["verdict"]["ok"] and res["end"]["verdict"]["ok"]
                res["check_s"] = round(time.perf_counter() - t0, 1)
                # the captures sit outside vLLM's memory budget; released, the next arm's gate has their room
                win.caps = {}
                torch.cuda.empty_cache()
                # the fault steps move served values on purpose, so the token gate reads a generation with none armed
                res["tokens_capture"] = res["tokens"]
                _, outs = gen(pp[:PF_GATE_REQS], GATE_TOKENS) if pf else gen(ps, GATE_TOKENS)
                res["tokens"] = [list(o.outputs[0].token_ids) for o in outs]
            conn.send(res)
        elif cmd["op"] == "prof":
            prof = settings.DITHER_NSYS in ("1", "prof")
            torch.cuda.synchronize()
            if prof:
                torch.cuda.profiler.start()
            counts = PF if pf else PROF_N
            reqs = B if settings.SBENCH_PROF_REQS is None else settings.SBENCH_PROF_REQS
            res = {"prof_n": list(counts), "profiled": prof, "prefill": pf, "prof_requests": reqs}
            for n in counts:
                if prof:
                    torch.cuda.nvtx.range_push(f"{PROF_RANGE}/{arm}/n{n}")
                t, outs = gen(pp[:n], 1) if pf else gen(ps if reqs == B else prompts(reqs), n)
                torch.cuda.synchronize()
                if prof:
                    torch.cuda.nvtx.range_pop()
                res[f"gen_n{n}_s"] = round(t, 4)
            if prof:
                torch.cuda.profiler.stop()
            step = round(1e3 * (res[f"gen_n{counts[1]}_s"] - res[f"gen_n{counts[0]}_s"]) / (counts[1] - counts[0]), 4)
            res["prefill_step_ms" if pf else TPOT if reqs == B else f"step_m{reqs}_ms"] = step
            conn.send(res)
        elif cmd["op"] == "smq_sweep":
            from dither_so import smq_of
            sd = sys.modules["dither_a"].SIDE[0]
            q0, d0, c0 = sd.units_smq, sd.units_smd, sd.units_cap
            rows = []
            for v in cmd["values"]:
                sd.set_units_smq(*smq_of(v))
                gen(ps, 8)
                llm.dither_stats()
                if cmd.get("sat"):
                    t, outs = gen(prompts(SAT_REQS), SAT_TOKENS)
                    row = {"tokens_per_s": round(sum(len(o.outputs[0].token_ids) for o in outs) / t, 2)}
                else:
                    t1, _ = gen(ps, N1)
                    t2, _ = gen(ps, N2)
                    row = {TPOT: round(1e3 * (t2 - t1) / (N2 - N1), 4)}
                rows.append({"smq": v, **row, "phases_by_m": llm.dither_stats().get("phases_by_m")})
            sd.set_units_smq(q0, d0, c0)
            gen(ps, 8)
            conn.send({"sweep": rows})
        elif cmd["op"] == "sat":
            t, outs = gen(prompts(SAT_REQS), SAT_TOKENS)
            n = sum(len(o.outputs[0].token_ids) for o in outs)
            conn.send({"tokens_per_s": round(n / t, 2), "sat_s": round(t, 3), "sat_tokens": n,
                       "tokens": [list(o.outputs[0].token_ids) for o in outs]})
        conn.send({"stats": llm.dither_stats() if hasattr(llm, "dither_stats") else None, "picks": PK.summary(plog)})

    while True:
        cmd = conn.recv()
        if cmd["op"] == "stop":
            res = {"stopped": True}
            try:
                res.update(fill_end(llm, gen))
            except Exception as e:  # noqa: BLE001
                res["error"] = f"stop: {type(e).__name__}: {e}"
            try:
                close_coins()
            except Exception as e:  # noqa: BLE001
                res["close_error"] = f"{type(e).__name__}: {e}"
            conn.send(res)
            return
        try:
            serve(cmd)
        except Exception as e:  # noqa: BLE001
            conn.send({"error": f"{cmd['op']}: {type(e).__name__}: {e}"})
            return


def recv(arm: str, conn, op: str, keep: bool = False) -> dict:
    """ARM's next message.  A worker that reports an error or exits ends the run (exit 5, its result kept: the gates so
    far in gate.json, the reps in reps.jsonl); sys.exit terminates the daemon workers, so no engine outlives it.  With
    KEEP the error is returned instead, the worker gone and the run going on without it."""
    try:
        msg = conn.recv()
    except EOFError:
        msg = {"error": f"{op}: the worker exited"}
    if "error" in msg:
        print(json.dumps({"arm": arm, "op": op, "error": msg["error"], "timed": False}), flush=True)
        if not keep:
            sys.exit(5)
    return msg


def mem_peak_gib() -> float | None:
    """This process's cgroup's memory.peak in GiB: inside gpu-lease, its scope's, so every arm's worker and the gate check's
    pool, page cache included (what --mem-gb has to cover); None where cgroup v2 doesn't say."""
    try:
        rel = Path("/proc/self/cgroup").read_text().split("::", 1)[1].strip()
        return round(int(Path(f"/sys/fs/cgroup{rel}/memory.peak").read_text()) / 2**30, 2)
    except (OSError, IndexError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="stock,exact-eager")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.40)
    ap.add_argument("--t", type=int, default=MAX_TOKENS_STEP, help="max_num_batched_tokens: the Dither window's T")
    ap.add_argument("--ops", default="decode,sat")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--stock-own-batch", action="store_true",
                    help="stock arms at vLLM's own max_num_seqs and max_num_batched_tokens, not --t's")
    ap.add_argument("--arm-mem", type=arm_mem, default={},
                    help="ARM=F,...: those arms' gpu_memory_utilization (a stock arm's own batch needs KV room for it)")
    ap.add_argument("--time-passed", action="store_true",
                    help="on a failed gate, stop the failed arms and time the rest (the exit is 4 still)")
    ap.add_argument("--prefill", action="store_true",
                    help="every request a T-token prompt with one output token: a step of m = T (the docstring's --prefill)")
    ap.add_argument("--serial", action="store_true",
                    help="one engine on the GPU at a time: each arm built, warmed, gated, timed and stopped before the next "
                         "comes up, in --arms order (its reference arm first); for an arm that needs most of the GPU, as "
                         "Dither's window does at --prefill's T = 2048")
    ap.add_argument("--bw-passes", type=float, default=0.0, metavar="GIB",
                    help="the remote bandwidth audit's passes beside the arms on their GPU, over a store of GIB GiB "
                         "(`dbw.py`: the prover calibrated beside the dither arms' decode ops, an idle session, each rep "
                         "an off block and an on block, a tampered session)")
    ap.add_argument("--bw-prover-args", action="append", metavar="ARGS",
                    help="a side of its own a time, its prover's options (leading NAME=VALUE words its environment)")
    ap.add_argument("--bw-cover", type=float, default=4.0,
                    help="the first on block's session lasts this many times the off block before it (each later one "
                         "1.5 times the longest on block before it, the tampered one twice its dither op), in "
                         "stretches of the idle session's length (`dbw.window_s`), then of the last on session's "
                         "beside its block (`dbw.beside_s`)")
    ap.add_argument("--bw-idle-s", type=float, default=3.0, help="the idle session's passes, in seconds")
    a = ap.parse_args()
    if a.bw_passes and not (settings.BW_LIB and settings.BW_KV):
        ap.error("--bw-passes needs BW_LIB and BW_KV: session.so and kv.so (sbench.sh builds both)")
    if a.bw_passes and a.serial:
        ap.error("--bw-passes runs beside every arm at once, which --serial never brings up")
    if not 1 <= B <= a.t:
        ap.error(f"${B_ENV}={B}: a decode step of m = B needs 1 <= B <= --t ({a.t}), its max_num_seqs")
    if a.bw_passes and B != 32:
        ap.error(f"--bw-passes calibrates on B32 decode reps (`dbw.report`): unset ${B_ENV}")
    os.environ["SBENCH_T"] = str(a.t)
    if a.prefill:
        os.environ[PREFILL_ENV] = "1"
    if a.stock_own_batch:
        os.environ[OWN_BATCH_ENV] = "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["DITHER_MANIFEST_DIR"] = str(out)
    import tempfile
    scratch = Path(a.scratch or tempfile.mkdtemp(prefix="sbench."))
    arms = a.arms.split(",")
    for arm in arms:
        base = base_arm(arm)
        if base != arm:
            if base in REF_OF:
                REF_OF[arm] = REF_OF[base]
            if base in TOKENS_GATED:
                TOKENS_GATED[arm] = TOKENS_GATED[base]
    missing = sorted({REF_OF[x] for x in arms if x in TOKENS_ONLY and REF_OF[x] not in arms})
    if a.gate and missing:
        ap.error(f"a tokens-only arm's gate is its reference's tokens: add {','.join(missing)} to --arms")
    smq_sweep = list(settings.get(SMQ_SWEEP_ENV))
    if not all(re.fullmatch(r"0|\d+(/\d+(/\d+)?)?", v) for v in smq_sweep):
        ap.error(f"${SMQ_SWEEP_ENV}={settings.raw(SMQ_SWEEP_ENV)}: comma-separated q, q/den or q/den/cap")
    for arm, env in (settings.get(ARM_ENV_ENV) or {}).items():
        if bad := settings.unknown(env):
            ap.error(f"${ARM_ENV_ENV}'s {arm} sets {', '.join(bad)}, which nothing reads: fix the name, or declare it in "
                     f"{settings.path}")
    sweep_key = "tokens_per_s" if a.ops == "sat" else TPOT
    ctx = mp.get_context("spawn")
    conns, procs, info = {}, {}, {}
    reps: list[dict] = []
    rc = 0
    ref_tokens: dict = {}
    gate = {}
    vs_ref: dict = {}
    vs_rep0: dict = {}
    gate_doc: dict = {}
    stops = {}
    first, timed = arms[0], []
    bw_tokens: dict = {}

    def block(r: int, phase: str | None = None, side: int | None = None, only: list[str] | None = None) -> dict:
        """Rep r's ops on every arm up (or `only`'s), arm order rotated by r; under --bw-passes its phase (cal, off, on
        or tamper) and side on each row, an off block's tokens the gate's and the others' kept apart."""
        these, lo = only or arms, time.monotonic_ns()
        for op in a.ops.split(","):
            for j in range(len(these)):
                arm = these[(j + r) % len(these)]
                conns[arm].send({"op": op})
                res = recv(arm, conns[arm], op)
                msg = recv(arm, conns[arm], op)
                toks = res.pop("tokens", None)
                if toks is not None:
                    (ref_tokens if phase in (None, "off") else bw_tokens.setdefault(phase, {})).setdefault((r, arm), toks)
                row = {"rep": r, "op": op, "arm": arm, **res, "stats": msg["stats"], "picks": msg.get("picks")}
                if phase is not None:
                    row |= {"bw": phase, "side": side}
                reps.append(row)
                print(json.dumps(row), flush=True)
                with open(out / "reps.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
        if phase in (None, "off"):
            for arm in these:
                if (r, arm) in ref_tokens:
                    ref = REF_OF.get(arm, first)
                    gate.setdefault(arm, []).append(None if ref == arm else ref_tokens[(r, arm)] == ref_tokens.get((r, ref)))
                    if ref != arm and (r, ref) in ref_tokens:
                        vs_ref.setdefault(arm, []).append(token_diff(ref_tokens[(r, arm)], ref_tokens[(r, ref)]))
                    if r > 0:
                        vs_rep0.setdefault(arm, []).append(token_diff(ref_tokens[(r, arm)], ref_tokens[(0, arm)]))
        return {"rep": r, "phase": phase, "lo": lo, "hi": time.monotonic_ns()}

    bw_sides: list[dict] = []
    sweep: dict = {}
    # --serial: a group an arm, its engine stopped before the next one's comes up; else one group of every arm
    groups = [[x] for x in arms] if a.serial else [arms]
    for arms in groups:
        for arm in arms:      # one engine at a time comes up, so their memory profiling doesn't race
            here, there = ctx.Pipe()
            p = ctx.Process(target=worker, args=(arm, there, str(scratch), a.arm_mem.get(arm, a.gpu_mem)), daemon=True)
            p.start()
            there.close()   # the worker's end, closed here so a dead worker's pipe reads EOF, not a wait
            msg = recv(arm, here, "build")
            conns[arm], procs[arm], info[arm] = here, p, msg
            print(json.dumps({"arm": arm, **msg}), flush=True)
        for arm in arms:
            conns[arm].send({"op": "warm", "sat": "sat" in a.ops.split(",")})
            res = recv(arm, conns[arm], "warm", a.time_passed)
            if "error" not in res:
                res = recv(arm, conns[arm], "warm", a.time_passed)
            if "error" in res:
                gate_doc[arm], rc = {"error": res["error"]}, 5
                procs[arm].join()   # it exits on an error: gone before the next arm's engine comes up
        arms = [x for x in arms if x not in gate_doc]
        if a.gate:
            for arm in list(arms):
                conns[arm].send({"op": "gate"})
                res = recv(arm, conns[arm], "gate", a.time_passed)
                if "error" in res:
                    gate_doc[arm], rc = {"error": res["error"]}, 5
                    procs[arm].join()
                    arms.remove(arm)
                    continue
                msg = recv(arm, conns[arm], "gate")
                res["stats"], res["picks_summary"] = msg["stats"], msg.get("picks")
                gate_doc[arm] = res
                (out / "gate.json").write_text(json.dumps(gate_doc, indent=1) + "\n")
            for arm in arms:
                ref = REF_OF.get(arm, first)
                if "tokens" in gate_doc.get(ref, {}) and ref != arm:
                    gate_doc[arm]["tokens_equal_" + ref] = gate_doc[arm]["tokens"] == gate_doc[ref]["tokens"]
                    if "tokens_capture" in gate_doc[arm] and a.prefill:
                        # the capture run's one token a request against the reference's first, request by request
                        gate_doc[arm]["tokens_capture_first_equal_" + ref] = sum(
                            x[:1] == y[:1] for x, y in zip(gate_doc[arm]["tokens_capture"], gate_doc[ref]["tokens"]))
                    elif "tokens_capture" in gate_doc[arm]:
                        gate_doc[arm]["tokens_capture_requests_equal_" + ref] = sum(
                            x == y for x, y in zip(gate_doc[arm]["tokens_capture"], gate_doc[ref]["tokens"]))
                if "tokens" in gate_doc.get("stock", {}) and arm != "stock":
                    gate_doc[arm]["tokens_equal_stock"] = gate_doc[arm]["tokens"] == gate_doc["stock"]["tokens"]
                    gate_doc[arm]["tokens_requests_equal_stock"] = sum(
                        x == y for x, y in zip(gate_doc[arm]["tokens"], gate_doc["stock"]["tokens"]))
            for arm in arms:
                print(json.dumps({"gate": arm, **{k: v for k, v in gate_doc[arm].items()
                                                   if k not in ("tokens", "tokens_capture", "picks")}}), flush=True)
            (out / "gate.json").write_text(json.dumps(gate_doc, indent=1) + "\n")
            failed = [arm for arm in arms if "verdict" in gate_doc[arm] and not (
                gate_doc[arm]["verdict"]["ok"] and (gate_doc[arm].get("tokens_equal_" + REF_OF.get(arm, first))
                                                    or not TOKENS_GATED.get(arm, True)))]
            failed += [arm for arm in arms if arm in TOKENS_ONLY and gate_doc[arm].get("tokens_equal_" + REF_OF[arm]) is not True]
            if failed:
                timed_picks = [arm for arm in failed if gate_doc[arm].get("verdict", {}).get("ok") and pick_of(arm) is None]
                hint = {"hint": f"{timed_picks}: verdict ok, tokens not exact's, under timed picks: the node's shared "
                                "Inductor cache can hand an arm a reduction config exact never ran; rerun with "
                                f"{PICK_ENV}=first {PICK_ARMS_ENV}=exact,dither"} if timed_picks else {}
                print(json.dumps({"gate_failed": failed, "timed": a.time_passed, **hint}), flush=True)
                for arm in failed if a.time_passed else arms:
                    conns[arm].send({"op": "stop"})
                if not a.time_passed:
                    return 4
                arms = [x for x in arms if x not in failed]
                rc = rc or 4
                for arm in failed if a.serial else ():   # gone before the next arm's engine comes up
                    try:
                        stops[arm] = conns[arm].recv()
                    except EOFError:
                        stops[arm] = {"error": "stop: the worker exited"}
                    procs[arm].join()
        if not arms:
            continue
        for arm in arms if smq_sweep else ():
            if base_arm(arm) in DITHER_SO:
                conns[arm].send({"op": "smq_sweep", "values": smq_sweep, "sat": sweep_key == "tokens_per_s"})
                sweep[arm] = recv(arm, conns[arm], "smq_sweep")["sweep"]
                recv(arm, conns[arm], "smq_sweep")
                print(json.dumps({"smq_sweep": arm, sweep_key: {x["smq"]: x[sweep_key] for x in sweep[arm]}}), flush=True)
                (out / "smq_sweep.json").write_text(json.dumps(sweep, indent=1) + "\n")
        time_at = wait_time_at()
        if time_at:
            print(json.dumps({"time_at": time_at}), flush=True)
        if not a.bw_passes:
            for r in range(a.reps):
                block(r)
        else:
            import shlex

            import dbw
            aff = ",".join(map(str, sorted(os.sched_getaffinity(0))))
            for i, pargs in enumerate(a.bw_prover_args or [""]):
                doc = {"i": i, "prover_args": pargs, "plan": dbw.plan(a.reps), "blocks": [], "sessions": {}}
                bw_sides.append(doc)
                side = dbw.Side(out / f"bw{i}", plan=doc["plan"], gib=a.bw_passes, py=settings.BW_PY or sys.executable,
                                lib=settings.need("BW_LIB"), kv=settings.need("BW_KV"), lean_exe=settings.BW_LEAN_EXE,
                                prover_cpus=settings.BW_PROVER_CPUS or aff,
                                verifier_cpus=settings.BW_VERIFIER_CPUS,
                                side_args=shlex.split(settings.BW_SIDE_ARGS or ""), prover_args=shlex.split(pargs))
                dither = [x for x in arms if x.startswith("dither")] or arms
                try:
                    doc["ready"], cal = dbw.registered(side, lambda: block(i * a.reps, "cal", i, only=dither))
                    doc["calibrated_beside"] = {"arms": dither, "blocks": cal}
                    print(json.dumps({"bw": i, "ready": {k: v for k, v in doc["ready"].items() if k != "calibration"},
                                      "calibrated_beside": f"{len(cal)} blocks of {','.join(dither)}"}), flush=True)
                    side.start(0, dbw.paced(a.bw_idle_s, 1.0))
                    doc["sessions"][0] = side.end(0)
                    stretch_s, longest = dbw.window_s(doc["sessions"][0]), 0.0
                    for r in range(a.reps):
                        rr = i * a.reps + r
                        off = block(rr, "off", i)
                        doc["blocks"].append(off | {"k": None})
                        side.start(1 + r, dbw.paced(1.5 * longest or a.bw_cover * (off["hi"] - off["lo"]) / 1e9, 1.0,
                                                    stretch_s))
                        on = block(rr, "on", i)
                        doc["blocks"].append(on | {"k": 1 + r})
                        longest = max(longest, (on["hi"] - on["lo"]) / 1e9)
                        doc["sessions"][1 + r] = side.end(1 + r)
                        stretch_s = dbw.beside_s(doc["sessions"][1 + r], on["lo"], on["hi"]) or stretch_s
                        print(json.dumps({"bw": i, "session": 1 + r, "accepted": doc["sessions"][1 + r]["accepted"],
                                          "failure": doc["sessions"][1 + r]["failure"]}), flush=True)
                    k = 1 + a.reps
                    est = sum((x["spans_ns"][-1] - x["spans_ns"][0]) / 1e9 for x in reps if x.get("side") == i
                              and x.get("bw") == "on" and x["rep"] == i * a.reps + a.reps - 1 and x["arm"] in dither
                              and "spans_ns" in x)
                    side.start(k, dbw.paced(est, 2.0, stretch_s))
                    doc["blocks"].append(block(i * a.reps + a.reps - 1, "tamper", i, only=dither) | {"k": k})
                    doc["sessions"][k] = side.end(k)
                    doc["stopped"] = side.stop()
                except dbw.SideError as e:
                    doc["error"] = str(e)
                    print(json.dumps({"bw": i, "error": str(e)}), flush=True)
                    side.p.kill()
                    rc = rc or 6
                    break
        for arm in arms:
            conns[arm].send({"op": "stop"})
        # each worker's stop: a -we arm's off-node window served to its end and checked, its session closed
        for arm in arms:
            try:
                stops[arm] = conns[arm].recv()
            except EOFError:
                stops[arm] = {"error": "stop: the worker exited"}
            print(json.dumps({"stop": arm, **{k: v for k, v in stops[arm].items() if k != "stats"}}), flush=True)
            if a.serial:
                procs[arm].join()   # its device memory free before the next engine's profiling reads what is
        timed += arms
    arms = timed
    (out / "stop.json").write_text(json.dumps(stops, indent=1, default=str) + "\n")
    if not arms:
        return rc
    summary = {"arms": arms, "info": info, "numa": settings.DITHER_NUMA_RECORD,
               "gate_tokens_equal_ref_arm": gate, "decode_tokens_vs_ref": vs_ref,
               "decode_tokens_vs_rep0": vs_rep0, "ref_of": REF_OF, "n1": N1, "n2": N2, "b": B,
               "gate": {arm: {k: v for k, v in d.items() if k not in ("tokens", "picks")} for arm, d in gate_doc.items()},
               "pick": settings.get(PICK_ENV) or "timed", "pick_by_arm": {arm: pick_of(arm) or "timed" for arm in info},
               "saturation": {"requests": SAT_REQS, "tokens": SAT_TOKENS, "max_num_seqs": min(MAX_SEQS, a.t),
                              "max_num_batched_tokens": a.t, "stock_own_batch": a.stock_own_batch,
                              "scheduler_by_arm": {arm: info[arm].get("scheduler") for arm in arms}}, "per_arm": {}}
    if a.prefill:
        summary["prefill"] = {"t": a.t, "requests": list(PF), "ttft_requests": PF_TTFT, "gate_requests": PF_GATE_REQS,
                              "captures": list(PF_CAPTURES), "step": "one request of T prompt tokens, one output token"}
    plain = [x for x in reps if x.get("bw") in (None, "off")]
    for arm in arms:
        tp = [x[TPOT] for x in plain if x["arm"] == arm and x["op"] == "decode"]
        th = [x["tokens_per_s"] for x in plain if x["arm"] == arm and x["op"] == "sat"]
        summary["per_arm"][arm] = {TPOT: statistics.median(tp) if tp else None, "tpot_all": tp,
                                   "tokens_per_s": statistics.median(th) if th else None, "tokens_per_s_all": th}
        if a.prefill:
            pr = [x for x in plain if x["arm"] == arm and x["op"] == "prefill"]
            st = [x["prefill_step_ms"] for x in pr]
            ft = [x["ttft_ms"] for x in pr]
            summary["per_arm"][arm].update(
                prefill_step_ms=statistics.median(st) if st else None, prefill_step_all=st,
                ttft_ms=statistics.median(ft) if ft else None, ttft_all=ft,
                ttft_end_ms=statistics.median(x["ttft_end_ms"] for x in pr) if pr else None)
    ef = summary["per_arm"].get("exact-fused", {})
    for arm in arms:
        p = summary["per_arm"][arm]
        for k in (TPOT, "prefill_step_ms", "ttft_ms"):
            if p.get(k) and ef.get(k):
                p[f"gamma_{k}_vs_exact_fused"] = round(p[k] / ef[k], 4)
    base = summary["per_arm"][arms[0]]
    for arm in arms[1:]:
        p = summary["per_arm"][arm]
        if p[TPOT] and base[TPOT]:
            p["tpot_overhead_pct"] = round(100 * (p[TPOT] / base[TPOT] - 1), 3)
        if p["tokens_per_s"] and base["tokens_per_s"]:
            p["throughput_overhead_pct"] = round(100 * (1 - p["tokens_per_s"] / base["tokens_per_s"]), 3)
    summary["gate_failed"] = [x for x in gate_doc if x not in arms]
    if sweep:
        summary["smq_sweep"] = {arm: {x["smq"]: x[sweep_key] for x in rows} for arm, rows in sweep.items()}
    summary["stop"] = {arm: {"error": x.get("error"), "close_error": x.get("close_error"), "fill_steps": x.get("fill_steps"),
                             "end_verdict": (x.get("end") or {}).get("verdict")} for arm, x in stops.items()}
    summary["host_mem_peak_gib"] = mem_peak_gib()
    if a.bw_passes:
        import dbw
        bw = dbw.report(reps, bw_sides, arms)
        bw["tokens"] = {phase: {arm: [{"rep": r, "equal_off": t == ref_tokens.get((r, arm)),
                                       "requests_equal_ref_off": None if REF_OF.get(arm, arms[0]) == arm else sum(
                                           x == y for x, y in zip(t, ref_tokens.get((r, REF_OF.get(arm, arms[0])), ())))}
                                      for (r, x), t in sorted(got.items()) if x == arm] for arm in arms}
                        for phase, got in bw_tokens.items()}
        bw["errors"] = {d["i"]: d["error"] for d in bw_sides if "error" in d}
        bad = bw["errors"] or not all(s["honest_accepted"] and s["tampered_refused"] for s in bw["sides"])
        if bad:
            rc = rc or 6
        summary["bw"] = bw
        print(json.dumps({"bw": {k: v for k, v in bw.items() if k != "tokens"}}, indent=1), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["per_arm"], indent=1), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
