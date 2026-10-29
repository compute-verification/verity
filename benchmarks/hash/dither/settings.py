"""Dither's harness's settings: every variable its benches, windows and gates read, declared once (`verity_config.settings`).

It owns ``DITHER_``, ``SBENCH_`` and ``HB_``: a variable with one of them that nothing here declares is refused when the
registry loads, which every module of this directory that reads one does (``of(__file__)``). The shell-only ones (sbench.sh's
and hb_served.sh's, which never reach a Python read) are declared too, so their exports are known. ``H1U_*``
are h1u.cu's nvcc macros (``-DH1U_JRC=1``), not variables; ``HB_KC_FLAGS`` and its kin carry them to the build.

A build's path (``DITHER_H1C``, ``HB_H1U``, ...) is ``where``: sbench.sh and hb_served.sh build each from this tree's sources
at the run's commit (RESEARCH_SOURCE_SHA), or check a sent build's sources against it, so the commit is the value.

The standalone files read their variables themselves: H1V3's reference runs under another tree's PYTHONPATH (h1v3_ref.py),
live_audit.py reruns itself under main's, and hb_served.sh runs the hb_*.py from the run's inputs when the source has none.
`REGISTRY.standalone_problems` holds each of their reads to its declaration here.
"""
from __future__ import annotations

import json

from verity_config.settings import DEBUG, VALUE, WHERE, Registry, Setting as S, csv, not_zero, one

STANDALONE = ("h1v3_ref.py", "live_audit.py", "hb_sbench.py", "hb_peel.py")
BUILD = "a build's path in the run's scratch (sbench.sh's or hb_served.sh's, from this tree's sources at the run's commit)"
TREE = "reference's source tree, run under its own PYTHONPATH; the result records its commit"
SHELL = "sbench.sh's, read only by the shell"

REGISTRY = Registry(
    __file__, ("DITHER_", "SBENCH_", "HB_"),
    # the window: dither_one, dither_a, dither_so
    S("DITHER_H1C", str, None, WHERE, f"libh1c.so, the hiding tables' commit: {BUILD}"),
    S("DITHER_H1C_KC", str, None, WHERE, f"libh1c_kc.so, the dither-a-hbc arm's tables: {BUILD}"),
    S("DITHER_H1F", str, None, WHERE, f"libh1f.so, the fused window's units: {BUILD}"),
    S("DITHER_H1O", str, None, WHERE, f"libh1o.so, the one-stream window's units: {BUILD}"),
    S("DITHER_UNITS", str, "block", VALUE, "dither_one's units form (block, mix, ...): which kernel computes a call's units"),
    S("DITHER_PEEL", one, False, VALUE, "1: z = A B^T from O (the peel of b and c), so y is today's served y"),
    S("DITHER_PERCALL", str, None, VALUE, "the per-call challenger round's waits, measured only: a file of ns "
                                          "waits h1o_percall replays, 0 no wait"),
    S("DITHER_PERCALL_WAITS", str, None, VALUE, "the per-call waits' file sbench.sh found in the run's inputs, which a "
                                                "-percall arm's DITHER_PERCALL takes"),
    S("DITHER_A_STEP", one, False, VALUE, "1: one A table a K a step (dither_so's -aom/-aomn without -ov), built after the "
                                          "forward"),
    S("DITHER_A_CHUNKS", str, None, VALUE, "0: DITHER_A_STEP's rows whole at every K (the layout before SPLITK), only to "
                                           "time the chunk leaves against"),
    S("DITHER_A_FORM", int, 0, VALUE, "the A form h1c builds: 2 or 3 its compressions paired (dither_a sets 3 when unset)"),
    S("DITHER_A_LVF", str, None, VALUE, "DITHER_A_STEP's tables' level schedule (H1C.set's lvf): unset from depth - "
                                        "A_LVF_TOP, L from L, empty the build's H1C_LVF; the same roots", empty=True),
    S("DITHER_ZY1", one, False, VALUE, "1: the z/y, y and units' y tables through one h1c_build in the z tables' place"),
    S("DITHER_ZY1_KC", str, None, VALUE, "DITHER_ZY1 for the dither-a-hbc arms, which sbench copies into DITHER_ZY1"),
    S("DITHER_ZDEAD", one, False, VALUE, "1: the z/y and y tables' rows past m take their x from dead_rows and hash no "
                                         "segments; the same roots"),
    S("DITHER_YCMP", one, False, VALUE, "1 (-zy under DITHER_ZDEAD): the compare of the served y with the records' y "
                                        "halves on their bytes, a differing call refusing its step"),
    S("DITHER_LANE_REFS", str, None, VALUE, "the gate step's option A partials against SERVE-REG's gated lane: 1 checks "
                                            "them in the gate, a path saves their records there"),
    S("DITHER_MM_CFG", str, None, VALUE, "N,K,MMAX=CFG+...: a call's GEMM config in place of exactlin.pick's"),
    S("DITHER_CHECK_WORKERS", int, 0, WHERE, "the gate check's worker processes (0: its own count): how fast it checks"),
    S("DITHER_DEAD_ONCE", one, False, VALUE, "1: a step's GEMMs and y commits write a row past m only where an earlier step "
                                             "left something else; the same bytes"),
    S("DITHER_ROUNDS", int, None, VALUE, "the window's round coins' layout (the session's when the coins are off the node)"),
    S("DITHER_MANIFEST_DIR", str, None, WHERE, "where an off-node window's manifest.json goes (its per-call roots to scratch)"),
    S("DITHER_SPILL_MAX_GB", float, None, WHERE, "the spill pool's bound in GiB, past which a window's spill refuses"),
    S("DITHER_CHALLENGERS", json.loads, None, WHERE, "each arm's challenger session off the node, as JSON ({ARM: {session, "
                                                     "address, secret_file, commitment, rounds}}): where its coins come "
                                                     "from, which its stats say"),
    S("DITHER_CREDIT_GATE", str, None, WHERE, "the credit gate's credit_gate.py sbench.sh fetched, which dither-1s-credit runs"),
    S("DITHER_H1V3_TREE", str, None, WHERE, f"H1V3's {TREE}"),
    S("DITHER_H1V3_SHA", str, None, WHERE, "DITHER_H1V3_TREE's commit, which the result records"),
    S("DITHER_J10R_TREE", str, None, WHERE, f"J10-R's {TREE}"),
    S("DITHER_J10R_SHA", str, None, WHERE, "DITHER_J10R_TREE's commit, which the result records"),
    S("DITHER_SO_GRAPH", not_zero, True, VALUE, "0: a step's units launched call by call, else one CUDA graph"),
    S("DITHER_SO_PRIO", one, False, VALUE, "1: any -so arm serves as -ovp does, on a stream above the units'"),
    S("DITHER_SO_CAPTURE_PRIO", one, False, VALUE, "1: under the serving priority, vLLM's graphs captured on a stream at "
                                                   "that priority too"),
    S("DITHER_SO_UNITS_PRIO", int, 0, VALUE, "the units' stream priority (torch's: lower runs first)"),
    S("DITHER_SO_FUSED", one, False, VALUE, "1: a step's units in one launch over every call (kh_units_multi); the same "
                                            "receipts"),
    S("DITHER_SO_COIN_ASYNC", int, 0, VALUE, "1 or 2: step s's units dispatched once coin s is back while the host waits on "
                                             "the next forward (2 later still); 0 inline"),
    S("DITHER_SO_UNITS_SMQ", str, None, VALUE, "the fused units' SMs: q, q/den or q/den/cap; the same receipts"),
    S("DITHER_SO_RDEAD", one, False, VALUE, "1: a dead row block's receipt rows take their x from one hash of that row; the "
                                            "same leaves"),
    S("DITHER_SO_UNITS", str, "today", VALUE, "dither_so's units form (today, or j10r: J10-R's call-scale noise)"),
    S("DITHER_SO_J10RC_MIN_T", int, 0, VALUE, "the smallest T at which j10r's units take their c form"),
    S("DITHER_SO_J10R_M", str, "slot", VALUE, "j10r's M: slot, the slot T with every row block computed, or the call's"),
    S("DITHER_UNITS_SKIP_DEAD", one, False, VALUE, "1 (with DITHER_SO_UNITS=j10r): a row block wholly past the live rows "
                                                   "skipped, the canonical dead tile"),
    S("DITHER_SO_ROOT_LAG", int, 0, VALUE, "1 or 2: a step's round left on the serving stream and sent at the next step's "
                                           "start; the same roots"),
    S("DITHER_SO_ZSIDE", one, False, VALUE, "1: a step's z tables built on the first tree stream beside its A tables"),
    S("DITHER_SO_PUT_ASYNC", one, False, VALUE, "1: a step's inputs written from pinned host memory without a wait"),
    S("DITHER_SO_PREQ", one, False, VALUE, "1: the units queued as their graph behind a wait on a pinned host word"),
    S("DITHER_SO_TAIL", str, "0", VALUE, "1 or 2: a step's mid and round trees through h1c's Tail; the same bytes"),
    S("DITHER_SO_HMARK", one, False, DEBUG, "1 (a diagnostic): the host's progress as events on an idle stream"),
    S("DITHER_SO_CAP_HOST", one, False, WHERE, "1: a capture's copies go to host memory, not the device"),
    S("DITHER_NSYS", str, None, DEBUG, "1 or prof: an nsys capture of the gate's kernels (NVTX ranges)"),
    S("DITHER_NCU", str, None, DEBUG, f"a kernel regex sbench.sh runs under ncu instead ({SHELL})"),
    S("DITHER_NCU_COUNT", int, None, DEBUG, f"the launches DITHER_NCU profiles ({SHELL})"),
    S("DITHER_NUMA", str, None, WHERE, "0: numabind.py runs the command unbound (its own read: numabind.py's plan)"),
    # the served bench: sbench.py and its launcher
    S("SBENCH_B", int, 32, VALUE, "the batch a TPOT is timed at (tpot_b<B>_ms)"),
    S("SBENCH_T", int, None, VALUE, "the step's max_num_batched_tokens (sbench sets it from --t)"),
    S("SBENCH_PREFILL", one, False, VALUE, "1: the prefill row (sbench sets it from --prefill)"),
    S("SBENCH_STOCK_OWN_BATCH", one, False, VALUE, "1: a stock arm keeps vLLM's own batch caps (--stock-own-batch)"),
    S("SBENCH_MM_CFG", str, None, VALUE, "the -mmc arms' DITHER_MM_CFG"),
    S("SBENCH_PICK", str, None, VALUE, "the step's pick (picks.py) the timed reps take, else timed"),
    S("SBENCH_PICK_ARMS", csv, (), VALUE, "the arms SBENCH_PICK applies to (a prefix each), else every arm"),
    S("SBENCH_SMQ_SWEEP", csv, (), VALUE, "the fused units' SM shares swept: comma-separated q, q/den or q/den/cap"),
    S("SBENCH_ARM_ENV", json.loads, None, VALUE, "each arm's own environment in its worker, as JSON ({ARM: {NAME: VALUE}})"),
    S("SBENCH_TIME_AT", str, None, WHERE, "HH:MM (UTC): the reps wait until then (a booked window's start)"),
    S("SBENCH_PROF_REQS", int, None, DEBUG, "the requests an nsys-profiled step serves (default the batch)"),
    S("SBENCH_ARCH", str, None, WHERE, f"the arch the libraries are built for: 120a, 100a or 90a ({SHELL})"),
    S("SBENCH_SCRIPT", str, None, WHERE, f"another script of this directory run with sbench's libraries ({SHELL})"),
    S("SBENCH_KGATE", str, None, DEBUG, f"0 or 1 in place of exactlin's kgate.py verdict ({SHELL})"),
    S("SBENCH_MAX_MIN", int, None, WHERE, f"the GPU lease's minutes ({SHELL})"),
    S("SBENCH_LEASE_AT", str, None, WHERE, f"when the lease starts, for a booked window ({SHELL})"),
    S("SBENCH_TIMED", one, False, WHERE, f"1: take the lease a timed window grants (gpu-lease --timed) ({SHELL})"),
    S("SBENCH_ON", str, None, WHERE, f"the GPUs the lease may take ({SHELL})"),
    S("SBENCH_MEM_GB", int, None, WHERE, f"the host memory the run asks for ({SHELL})"),
    S("SBENCH_PREEMPTIBLE", str, None, WHERE, f"the lease's preemptibility ({SHELL})"),
    S("SBENCH_SCRATCH_ROOT", str, None, WHERE, f"where the run's scratch goes ({SHELL})"),
    S("SBENCH_SHIP", str, None, WHERE, f"a tar of prebuilt libraries taken in place of building ({SHELL})"),
    S("SBENCH_SHIP_OUT", one, False, WHERE, f"1: build the libraries into out/sbench-ship.tar and exit ({SHELL})"),
    S("SBENCH_MPS", str, None, WHERE, f"the MPS share the bandwidth side runs under ({SHELL})"),
    # the hb arms: hb_served.sh, hb_sbench.py and the gates it runs
    S("HB_DITHER_DIR", str, None, WHERE, "the tree's dither directory hb_sbench.py imports sbench from"),
    S("HB_H1U", str, None, WHERE, f"h1u.so, the hb arms' units: {BUILD}"),
    S("HB_H1U_KSPLIT", str, None, WHERE, f"the K split h1u build (dither-1s-hbk): {BUILD}"),
    S("HB_H1U_KRONLY", str, None, WHERE, f"the receipts-only h1u build (hbr): {BUILD}"),
    S("HB_H1U_KC", str, None, WHERE, f"h1u_kc.so, the dither-a-hbc arm's: {BUILD}"),
    S("HB_H1U_KC_B", str, None, WHERE, f"h1u_kcb.so, the ARM-kcb arms' second h1u_kc: {BUILD}"),
    S("HB_H1U_PEEL", str, None, WHERE, f"the peeled h1u build: {BUILD}"),
    S("HB_H1V3_PEEL_TREE", str, None, WHERE, f"the peeled H1V3's {TREE}"),
    S("HB_H1V3_PEEL_SHA", str, None, WHERE, "HB_H1V3_PEEL_TREE's commit, which the result records"),
    S("HB_KC_FLAGS", str, None, VALUE, "h1u.cu's nvcc flags for h1u_kc.so (set, even empty, builds it)", empty=True),
    S("HB_KC_FLAGS_B", str, None, VALUE, "h1u.cu's nvcc flags for h1u_kcb.so"),
    S("HB_KC_H1C_FLAGS", str, None, VALUE, "h1c.cu's nvcc flags for libh1c_kc.so"),
    S("HB_SCRIPT", str, None, WHERE, "a script of this directory hb_served.sh runs in hb_sbench.py's place"),
    S("HB_KGATE", str, None, DEBUG, "0 or 1 in place of exactlin's kgate.py verdict (hb_served.sh's)"),
    S("HB_KEEP_SQLITE", one, False, DEBUG, "1: hb_served.sh keeps the nsys capture's sqlite"),
    S("HB_WHO", str, None, WHERE, "the GPU lease's owner (the launchers')"),
    # other libraries' variables the harness reads
    S("VLLM_DISABLED_KERNELS", str, None, VALUE, "vLLM's kernels an arm turns off, which its record holds"),
    S("VLLM_USE_FLASHINFER_SAMPLER", str, None, VALUE, "vLLM's sampler choice, which the native baseline records"),
    S("VLLM_ENABLE_V1_MULTIPROCESSING", str, None, WHERE, "0: vLLM's engine core in the bench's process, where its patches "
                                                          "reach it"),
    S("S8_WGMMA_CUBIN", str, None, VALUE, "the s8 WGMMA cubin j10r90's units load (default the one it builds)"),
    S("J10_CA_TREE", str, None, WHERE, f"J10-B's banded {TREE}"),
    S("BW_LIB", str, None, WHERE, "the remote bandwidth audit's prover library (dbw.py's side)"),
    S("BW_KV", str, None, WHERE, "the remote bandwidth audit's store (dbw.py's side)"),
    S("BW_PY", str, None, WHERE, "the Python the bandwidth side runs (default this one)"),
    S("BW_LEAN_EXE", str, None, WHERE, "the bandwidth side's Lean verifier binary"),
    S("BW_PROVER_CPUS", str, None, WHERE, "the bandwidth prover's CPUs (default the GPU's NUMA node's)"),
    S("BW_VERIFIER_CPUS", str, None, WHERE, "the bandwidth verifier's CPUs"),
    S("BW_SIDE_ARGS", str, None, VALUE, "extra arguments to the bandwidth side (shell words)"),
    standalone=STANDALONE)
