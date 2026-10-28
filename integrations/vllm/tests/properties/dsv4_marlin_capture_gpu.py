"""DeepSeek-V4-Pro's Marlin W4A16 MXFP4 expert GEMMs on Hopper against their Programs: the H200's go/no-go for binding
`marlin_moe.MarlinMoeExpertGemm(W)_v1` (and through them `MarlinMoeSchedule_v1`, `MarlinMxfp4Coordinate(W)_v1`) to the two
`ops.moe_wna16_marlin_gemm` launches of `fused_marlin_moe` at one TP8 rank's shapes (E = 384, top_k 6, hidden 7168, I = 384).

One GPU process writes `<out>/dsv4_marlin_capture.json`, the property record `dsv4_marlin_capture` (`properties.record`), and prints
one line per case and then `DSV4-MARLIN GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first and nothing else
runs when it fails (exit 3).  A phase that raises records its error and fails the verdict without stopping the others.

  gate       `fa2_target_capture_gpu.gate`: cc 9.0, 132 SMs (an H200), and the driver exactly `--driver`, by default the one the
             capture map's target names (`pinned_driver`, `target.driver`), since the SASS that runs is that driver's (`jit`).
  selection  `select_deepseek_v4_mxfp4_moe_backend` on DeepSeek-V4-Pro's FusedMoEConfig (oracle/mxfp4.py:656) selects MARLIN /
             MarlinExperts, `get_marlin_input_dtype()` is None (W4A16: VLLM_MARLIN_INPUT_DTYPE unset), Marlin's round-up leaves
             (7168, 384) as they are (oracle/mxfp4.py:746-751), and the LLM entrypoint's default max_num_seqs on this GPU
             (`EngineArgs.get_batch_defaults`) is ENVELOPE.
  binary     the SASS and PTX archs `_moe_C` carries (information: the `8.0+PTX` build has no sm_90 SASS).
  jit        the SASS that ran.  The driver JIT-compiles the compute_80 PTX into this process's own compute cache (`CUDA_CACHE_PATH`,
             empty, under the scratch directory: `environment`), and each profiled kernel is read back out of it (`jit`): the
             sha256 of its cubin and of its function's SASS (`.text.<symbol>`, one per template), its arch (sm_90), its registers
             (`cuobjdump -res-usage`), which must be the ones the profiler read, and its floating-point opcodes, which must lie in
             `FP_OPS`; with libcuda's and the PTX JIT compiler's paths and sha256.  The cubins are kept in `<out>/jit/`.
  auto       on one layer of random logical weights (every e2m1 code; scale bytes 118..127, and experts 0..7 with 30% of their scale
             bytes from `EDGE_BYTES`: +0, +inf, overflow) repacked by `prepare_moe_mxfp4_layer_for_marlin`, `fused_marlin_moe` runs
             each case twice (`CASES`: over experts 8..383 decode batches from 1 to 17 tokens, 460 and 461 (moe blocks 8 and 16),
             922 (32) and ENVELOPE, over 12 hot experts 16 and ENVELOPE tokens, and over the edge experts only 8 and 922;
             `--cases` runs a named subset); every `moe_wna16_marlin_gemm` launch is intercepted, its statics checked against the
             served path's (the moe block `fused_marlin_moe` picks for the batch, `block_of`, fp32 reduce, no atomic add, top_k and
             mul_topk_weights per launch, no bias, zeros, act order or global scale), its (pair -> block, expert) assignment against
             the routing in both runs, its words between the two runs on every pair both placed in the same moe block
             (`unmoved_rows_equal`), the second run's moved pairs against the twin at their own recorded blocks, and the first
             run's exact call replayed into a SENTINEL-filled output with a fresh workspace (`replay`), which must write the same
             words.
  kernel     the launch's thread config and blocks per SM are ops.cu's choice (`determine_exec_config`, ops.cu:476-490), invisible
             from the host and, on fp32 sums rounded once to bf16, almost never visible in the words: one torch.profiler session
             over every auto and sweep launch names each launch's kernel, `marlin_moe_wna16::Marlin<a, b, c, s, threads,
             thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, stages, group_blocks, is_zp_float>`, and its grid
             (sms * blocks_per_sm, ops.cu:488-533); the launch's config is (16 thread_k_blocks, 16 thread_n_blocks, grid / sms),
             one of its moe block's thread configs (`MR.thread_configs`: the small-batch ones for a block of 8 or 16, the
             large-batch ones for 32) at 1 to `MR.max_blocks_per_sm` blocks per SM (4 at one m block, 2 at two), the same in both
             runs, over the served types, the block's m blocks (`kernel_statics`: `m_block_size_8` only at 8), 4 stages and groups
             of 2 k blocks.
  sweep      `fused_marlin_moe` once at every M of SERVED_M under the same session: the served config of each launch kind at each M,
             which every auto launch's must be; `model` (information) per launch kind and moe block the per-config limits
             (`MR.auto_config`'s `allow`, each in 0 to the block's clamp) under which ops.cu's rule gives every swept config.
             SERVED_M is each batch of 1..17 and both sides of each moe block edge up to ENVELOPE, vLLM's default max_num_seqs on
             an H200, because the rule picks one config between them: a block fixes its configs and their limits, and the one count
             that reads M, lowered to prob_n / TN * prob_m * top_k * 4 / sms below sms * allow (<= 528 on 132 SMs), lowers none
             from M 4 (w13's 768 / TN * M * 6 * 4 reaches 528 at M 4 for TN 128 and at 2 for TN 64; w2's 7168 / TN * 6M * 4 at 1).
  forced     on `FORCED` cases, each launch again at every candidate, forced (`thread_k`, `thread_n`, `blocks_per_sm`), into an output
             filled with `SENTINEL`, on the probe input (`probe`: random signs and mantissas over 4 binades, each row scaled by a
             power of two so the median exact column sum over its experts is 2^127.5, where fp32 partial sums overflow in some
             summation orders and not others): a launch ops.cu refuses, or that writes no word (a failed launch: its CUDA error is
             read and cleared), is recorded refused; an accepted one is compared word for word at its config, and the accepted
             configs' own words must fall in more than one class (the probe tells summation orders apart, so the twin's schedule,
             warp tree and fold are checked where they show).
  words      every word of every auto launch against the twin (`kernels.marlin_moe_rows.launch`) at its kernel's config, and of every
             accepted forced launch at its own.
  reference  on each auto launch, the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`) of `MarlinMoeSchedule_v1`
             at the served statics on its first blocks, `MarlinMxfp4Coordinate(W)_v1` on sampled coordinates, and for the w2 launch
             `MarlinMoeExpertCoordinateW_v1{E=384}` on one column through the 384-expert gather, which the twin must equal.
  bind       `MarlinMoeExpertGemm(W)_v1` bind at every served config, and each names its Boolean version.

GO needs every launch at the served statics, assigned as routed in both runs, bit-identical on replay and, between the runs, on
every unmoved pair, every auto launch's kernel read and at its M's swept config, a swept config per kind at every M of SERVED_M, at
least one accepted forced launch per kind and more than one word class wherever two were accepted (when the phase runs), 0
mismatched words (the moved pairs' among them), every reference evaluation equal to the twin, and every profiled kernel's SASS read
back from the driver's cache.  The forced cases are at a moe block of 8: forcing one at 922 tokens would compare about 4.6e10
coordinate-k steps per candidate config (its w13 and w2 launches), against about 2.2e8 a second on 14 workers.

Replay determinism (schema v2).  v2 compared the two runs' outputs whole, and decode460 and hot1024 were
NO-GO on that comparison alone, while every one of run1's 242,460,672 words matched the twin at run1's recorded assignment.  The
launch is a function of its call: the k-slice fold is lock-serialized in slice order (marlin_template.h:2162-2170 at the pin,
`barrier_acquire(&locks[locks_off], slice_idx)` before each `global_reduce_fp32`), and `replay` now checks that directly.  What
moved between the runs is the assignment: `moe_align_block_size` places a multi-block expert's pairs across its blocks in
`atomicAdd` arrival order (csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:316-319, `_count_and_sort_expert_tokens`, one thread
per pair over many CTAs; at 384 local experts the sequential small-batch kernel, numel < 1024 and at most 64 experts, :658-661,
never runs), while expert_ids and num_tokens_post_padded come from per-expert counts and are stable.  A pair's words read its block
index through the schedule (`MarlinMoeSchedule_v1`: the DP part's columns are uncut, and the stream-K part's cut positions shift by
NT * KT mod iters per block), so a reshuffle shows exactly where an expert spans blocks whose cut patterns differ: decode460 (par
494 > 376 experts used; w13 blocks from 440 are stream-K at iters 69, and NT * KT = 672 is not a multiple) and hot1024 (par 197 >
12, every expert about 16 blocks, iters 54 likewise); hot16 and edge922 also run experts over several blocks but their w13 pattern
is block-invariant (672 a multiple of iters 21 and iters 112) and only a thin w2 tail differs, and their runs drew equal
placements.  So v3 checks what the record claims: each run against the twin at its own recorded assignment, the runs against each
other on the pairs they agree on, and the launch replayed at run1's call, bit for bit.  A served step's sorted_token_ids are part
of its committed inputs, so the Programs bind at whatever placement the align kernel draws; nothing here needs a Definition change.

What the JIT decides.  At a given config (TK, TN, GRID) the served kernels' arithmetic is fixed by their PTX, and the JIT chooses no
rounding in it.  In each of the three instantiations served at a moe block of 8 (256 threads at 8 x 8 n x k blocks, 128 at 8 x 4
and at 4 x 8) the PTX's floating-point instructions are `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (inline asm: one
HMMA.16816.F32.BF16 each, whose products and f32 accumulation are the tensor core's), `fma.rn.bf16x2` and `cvt.rn.bf16.f32`
(explicit rounding), and `add.f32` (the fp32 reduce).  The PTX ISA lets the code generator optimize an add without a rounding
modifier only by fusing it with a multiply into an FMA, and these kernels have no f32 multiply; nothing reassociates.  ptxas 12.9
compiles them for sm_90 to exactly that (HMMA.16816.F32.BF16; FADD; HMUL2 and HADD2.BF16_V2, the bf16x2 FMAs with a -0 addend or a
unit factor; F2FP.BF16.F32.PACK_AB), plus MUFU.RCP in the integer-division idiom, and `jit` holds the driver's SASS to the same
opcodes.  What the JIT does decide is the registers (ptxas 12.9: 120, 119 and 121), and `determine_exec_config` reads them
(ops.cu:321-323, `device_max_reg_size` over numRegs x threads x 4 bytes) to bound the blocks per SM: 2 at 256 threads and 4 at 128
below 128 registers, 1 and 3 from 128.  The blocks per SM give GRID, and GRID fixes the k-slice schedule and the fp32 fold order,
which the Programs take as statics.  So the config served at each batch is the driver's, and the record pins the driver and the
SASS.  The larger blocks run other instantiations of the same template (16: the three small-batch configs without
`m_block_size_8`; 32: two m blocks over the large-batch configs, 256 threads at 16 x 4 blocks and the two 128-thread ones, at most 2
blocks per SM), whose registers and opcodes this capture reads the same way and holds to the same FP_OPS.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/dsv4_marlin_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--skip-forced] [--driver VERSION]
                                                    [--cases NAME,NAME]
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import argparse
import ctypes
import glob
import hashlib
import inspect
import json
import multiprocessing
import os
import re
import struct
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties.glm53_block_fp8_capture_gpu import _activation, _error, _load, _save, _u16, _u32
from verity.ml.definitions.numerics import marlin_moe_rows as MR
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/dsv4-marlin-capture/v3"
RECORD = "dsv4_marlin_capture.json"
#: one TP8 rank: hidden 7168, moe_intermediate 3072 / 8, 384 routed experts, top-6; DeepSeek-V4-Pro's swiglu_limit
HIDDEN, INTER, NUM_EXPERTS, TOP_K, SWIGLU_LIMIT = 7168, 384, 384, 6, 10.0
GROUP = 32
#: vLLM d9105ea80's default max_num_seqs for the LLM entrypoint on a GPU of 70 GiB or more that is no A100, an H200's 141 GB
#: (engine/arg_utils.py:2664-2672, `get_batch_defaults`): a decode step of 1 to ENVELOPE tokens
ENVELOPE = 1024
#: (name, tokens, pool): `uniform` routes over experts EDGE_EXPERTS.., `hot` over 12 of them, `edge` over the edge experts only; the
#: batches above 16 are the first one, both sides of the moe block edge 8 | 16 (460, 461), the first of 32 (922) and ENVELOPE
#: (`block_of`: 8 to 460 tokens, 16 to 921, 32 to 1843)
CASES = (("decode1", 1, "uniform"), ("decode2", 2, "uniform"), ("decode3", 3, "uniform"), ("decode4", 4, "uniform"),
         ("decode8", 8, "uniform"), ("decode16", 16, "uniform"), ("hot16", 16, "hot"), ("edge8", 8, "edge"),
         ("decode17", 17, "uniform"), ("decode460", 460, "uniform"), ("decode461", 461, "uniform"), ("decode922", 922, "uniform"),
         ("decode1024", ENVELOPE, "uniform"), ("hot1024", ENVELOPE, "hot"), ("edge922", 922, "edge"))
QUICK = ("decode1", "decode4", "hot16", "edge8", "decode922")
FORCED, QUICK_FORCED = ("decode1", "hot16"), ("hot16",)
HOT = 12
#: experts 0..7 carry scale bytes from EDGE_BYTES on EDGE_FRACTION of their groups: +0 (code 0), the smallest, ones around 1, the
#: largest (6 * 2^127 overflows bf16) and 255 (+inf, so a +-0 code reads 0 * inf = NaN)
EDGE_EXPERTS, EDGE_FRACTION = 8, 0.3
EDGE_BYTES = (0, 1, 2, 125, 126, 127, 128, 252, 253, 254, 255)
SCALE_BYTES = (118, 128)
#: a bf16 NaN no launch writes (a NaN word out of the kernel is 0x7FFF)
SENTINEL = 0x7FA5
#: the sweep's batches: each of 1..17 and both sides of each moe block edge to ENVELOPE (460 | 461: 8 -> 16, 921 | 922: 16 -> 32)
SERVED_M = (*range(1, 18), 460, 461, 921, 922, ENVELOPE)
#: the probe's rows span PROBE_BINADES binades below 2; each is scaled so its median |exact column sum| is 2^PROBE_LOG2
PROBE_BINADES, PROBE_LOG2 = 4, 127.5
MARLIN = "marlin_moe_wna16::Marlin<"
TEMPLATE = ("a_type", "b_type", "c_type", "s_type", "threads", "thread_m_blocks", "thread_n_blocks", "thread_k_blocks", "m_block_size_8",
            "stages", "group_blocks", "is_zp_float")
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
#: coordinate-k steps (output words x K) per comparison job
JOB_COORD_K = 100_000_000
#: the launch arguments a W4A16 MXFP4 launch leaves None
ABSENT = ("b_bias", "a_scales", "global_scale", "b_qzeros", "g_idx", "perm")
#: the capture map whose target names the driver (`target.driver`)
MAP = Path(__file__).resolve().parents[2] / "manifests/capture-maps/deepseek-v4-eager-hopper-tp8-v0.json"
#: the environment that would disable the driver's JIT cache or its JIT, or JIT kernels that carry SASS
JIT_ENV = ("CUDA_CACHE_DISABLE", "CUDA_DISABLE_PTX_JIT", "CUDA_FORCE_PTX_JIT")
EM_CUDA = 190
#: a mangled `marlin_moe_wna16::Marlin<...>` (kernel.h): its twelve template arguments are literals `L<type><value>E` (the four
#: ScalarTypeIds int64_t, `l`; the bools `b`; a negative value prefixed `n`)
MANGLED = re.compile(r"_ZN16marlin_moe_wna166MarlinI((?:L[a-z]n?\d+E){12})E")
#: the floating-point SASS opcodes `sass_census` counts, and the ones the served kernels may hold (module docstring)
FP_OP = re.compile(r"(?:HMMA|HFMA|HMUL|HADD|HMNMX|FADD|FFMA|FMUL|FMNMX|FSET|FCHK|F2F|MUFU|DADD|DFMA|DMUL)")
FP_OPS = {"HMMA.16816.F32.BF16", "FADD", "HFMA2.BF16_V2", "HMUL2.BF16_V2", "HADD2.BF16_V2", "F2FP.BF16.F32.PACK_AB", "MUFU.RCP"}


def shape(kind: str) -> tuple[int, int]:
    """(K, N) of a launch kind: w13 [gate || up] over the hidden state, w2 down over the activation."""
    return (HIDDEN, 2 * INTER) if kind == "w13" else (INTER, HIDDEN)


def prob_m(kind: str, M: int) -> int:
    return M if kind == "w13" else M * TOP_K


def launch_top_k(kind: str) -> int:
    return TOP_K if kind == "w13" else 1


def block_of(M: int) -> int:
    """The moe block both of a batch's launches run at: `fused_marlin_moe`'s pick for M tokens of this rank (`MR.moe_block`)."""
    return MR.moe_block(M, TOP_K, NUM_EXPERTS)


def kernel_statics(block: int) -> dict[str, int | bool]:
    """The template statics ops.cu instantiates for this path at a moe block (ops.cu:362-363): its m blocks of 16, `m_block_size_8`
    at 8, 4 stages (cc >= 8.0) and groups of 32 = 2 k blocks."""
    return {"thread_m_blocks": MR.thread_m_blocks(block), "m_block_size_8": block == 8, "stages": 4, "group_blocks": GROUP // 16,
            "is_zp_float": False}


def candidates(block: int) -> tuple[tuple[int, int, int], ...]:
    """The forced phase's configs at a moe block: each of its thread configs at each count of blocks per SM ops.cu allows it."""
    return tuple((tk, tn, b) for tk, tn, _ in MR.thread_configs(block) for b in range(1, MR.max_blocks_per_sm(block) + 1))


# ------------------------------------------------------------------------------------------------------------------------- env
def environment(cache: Path) -> dict[str, Any]:
    """DeepSeek-V4-Pro's serve env as it bears on the experts (the capture map's `target.env`), set before vLLM is imported and
    before CUDA initialises; the serve leaves VLLM_MARLIN_INPUT_DTYPE and `JIT_ENV` unset, so a value this shell holds is dropped and
    recorded.  The driver's JIT cache is `cache`, new and empty, so every PTX kernel the run launches is compiled here, by this
    driver, and `jit` reads it back; a CUDA_CACHE_PATH this shell holds is recorded as replaced."""
    env: dict[str, Any] = {"VLLM_BATCH_INVARIANT": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_CACHE_MAXSIZE": str(4 << 30)}
    os.environ.update(env)
    env["unset"] = {k: os.environ.pop(k, None) for k in ("VLLM_MARLIN_INPUT_DTYPE", *JIT_ENV)}
    env["replaced"] = {"CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH")}
    cache.mkdir(parents=True)
    os.environ["CUDA_CACHE_PATH"] = str(cache)
    env["CUDA_CACHE_PATH"] = "<scratch>/" + cache.name
    return env


def pinned_driver() -> str:
    """The driver the capture map's target names (`target.driver.version`)."""
    return json.loads(MAP.read_text())["target"]["driver"]["version"]


def gate(min_driver: int, want_cc: str, want_sms: int, driver: str) -> dict[str, Any]:
    """`fa2_target_capture_gpu.gate`, and the driver exactly `driver`: the served kernels' SASS, and with its registers their config,
    is this driver's JIT of the PTX (`jit`), so a record on another driver claims nothing for the map's target."""
    rec = C.gate(min_driver, want_cc, want_sms)
    rec.setdefault("want", {})["driver"] = driver
    rec.setdefault("checks", {})["driver_pinned"] = rec.get("driver") == driver
    rec["ok"] = all(rec["checks"].values())
    return rec


# ------------------------------------------------------------------------------------------------------------------- selection
def selection() -> dict[str, Any]:
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        mxfp4_round_up_hidden_size_and_intermediate_size,
        select_deepseek_v4_mxfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_input_dtype
    from vllm.usage.usage_lib import UsageContext

    cfg = FusedMoEConfig(num_experts=NUM_EXPERTS, experts_per_token=TOP_K, hidden_dim=HIDDEN, intermediate_size=INTER,
                         num_local_experts=NUM_EXPERTS, num_logical_experts=NUM_EXPERTS, activation=MoEActivation.SILU, device="cuda",
                         routing_method=RoutingMethodType.DeepseekV4, moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
                         in_dtype=torch.bfloat16, swiglu_limit=SWIGLU_LIMIT)
    backend, cls = select_deepseek_v4_mxfp4_moe_backend(cfg)
    rec = {"backend": getattr(backend, "name", str(backend)), "experts": getattr(cls, "__name__", None),
           "input_dtype": str(get_marlin_input_dtype()),
           "rounded": list(mxfp4_round_up_hidden_size_and_intermediate_size(Mxfp4MoeBackend.MARLIN, HIDDEN, INTER)),
           "default_max_num_seqs": EngineArgs.get_batch_defaults(world_size=8)[1].get(UsageContext.LLM_CLASS)}
    rec["ok"] = (rec["backend"] == "MARLIN" and rec["experts"] == "MarlinExperts" and rec["input_dtype"] == "None"
                 and rec["rounded"] == [HIDDEN, INTER] and rec["default_max_num_seqs"] == ENVELOPE)
    return rec


def binary() -> dict[str, Any]:
    """Information only: an error here is recorded in the phase, not the verdict."""
    import vllm

    so = glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_moe_C*.so"))
    if not so:
        return {"error": "no vllm/_moe_C*.so"}
    try:
        rec = C._archs(so[0])
    except Exception as e:  # noqa: BLE001
        return {"so": so[0], "error": _error(e)}
    rec["sm90_sass"] = "90" in rec["sass_archs"] or "90a" in rec["sass_archs"]
    return rec


# ------------------------------------------------------------------------------------------------------------------------- jit
def mangled_template(symbol: str) -> dict[str, int | bool] | None:
    """The template arguments of a mangled Marlin MoE kernel, by `TEMPLATE`'s names; None for another symbol."""
    m = MANGLED.match(symbol)
    if m is None:
        return None
    lits = re.findall(r"L([a-z])(n?)(\d+)E", m.group(1))
    return {n: (v != 0 if t == "b" else v) for n, (t, neg, d) in zip(TEMPLATE, lits, strict=True) for v in [-int(d) if neg else int(d)]}


def _elf_at(blob: bytes, i: int) -> bytes | None:
    """The little-endian ELF64 CUDA image starting at `blob[i]`, through the last of its headers and sections; None if there is none."""
    if len(blob) < i + 64 or blob[i + 4:i + 6] != b"\x02\x01" or struct.unpack_from("<H", blob, i + 18)[0] != EM_CUDA:
        return None
    phoff, shoff = struct.unpack_from("<QQ", blob, i + 32)
    phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", blob, i + 54)
    if shentsize != 64 or not shnum or i + shoff + 64 * shnum > len(blob):
        return None
    end = max(shoff + 64 * shnum, phoff + phentsize * phnum)
    for k in range(shnum):
        h = i + shoff + 64 * k
        if struct.unpack_from("<I", blob, h + 4)[0] != 8:        # SHT_NOBITS takes no bytes
            off, size = struct.unpack_from("<QQ", blob, h + 24)
            end = max(end, off + size)
    return blob[i:i + end] if i + end <= len(blob) else None


def cuda_elfs(blob: bytes) -> list[tuple[int, bytes]]:
    """Every CUDA image in `blob` as (offset, image): a compute-cache entry holds the cubin the driver compiled behind a header of
    its own."""
    out, i = [], blob.find(b"\x7fELF")
    while i >= 0:
        img = _elf_at(blob, i)
        if img:
            out.append((i, img))
        i = blob.find(b"\x7fELF", i + (len(img) if img else 1))
    return out


def elf_functions(img: bytes) -> dict[str, bytes]:
    """Each function's SASS in a CUDA image (its `.text.<symbol>` section), by symbol."""
    shoff, = struct.unpack_from("<Q", img, 40)
    shnum, shstrndx = struct.unpack_from("<HH", img, 60)
    hdrs = [struct.unpack_from("<IIQQQQ", img, shoff + 64 * k) for k in range(shnum)]
    names = img[hdrs[shstrndx][4]:hdrs[shstrndx][4] + hdrs[shstrndx][5]]
    out = {}
    for name_at, sh_type, _, _, off, size in hdrs:
        name = names[name_at:names.index(b"\0", name_at)].decode()
        if name.startswith(".text.") and sh_type != 8:
            out[name[len(".text."):]] = img[off:off + size]
    return out


def sass_census(sass: str) -> dict[str, int]:
    """The floating-point opcodes (`FP_OP`) of a `cuobjdump -sass` listing, counted."""
    ops = re.findall(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)", sass)
    return dict(sorted(Counter(op for op in ops if FP_OP.match(op)).items()))


def function_registers(res_usage: str) -> dict[str, int]:
    """REG of each function of a `cuobjdump -res-usage` listing."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"Function (\S+):\s*\n\s*REG:(\d+)", res_usage)}


def _cuobjdump(*args: str) -> str:
    r = C._run([C._cuda_tool("cuobjdump"), *args])
    if r.returncode:
        raise RuntimeError(f"cuobjdump {' '.join(args)}: {(r.stdout + r.stderr)[-300:]}")
    return r.stdout


def driver_libraries() -> dict[str, Any]:
    """The CUDA driver library this process loaded and the PTX JIT compiler it loaded to compile PTX, each path and sha256, and
    `cuDriverGetVersion`."""
    with open("/proc/self/maps") as f:
        paths = sorted({p for p in (ln.split()[-1] for ln in f)
                        if p.startswith("/") and re.search(r"/lib(cuda|nvidia-ptxjitcompiler)\.so", p)})
    rec: dict[str, Any] = {os.path.basename(p): {"path": p, "sha256": C._sha256_file(p)} for p in paths}
    lib = next((p for p in paths if "/libcuda.so" in p), None)
    if lib:
        v = ctypes.c_int()
        ctypes.CDLL(lib).cuDriverGetVersion(ctypes.byref(v))
        rec["cuDriverGetVersion"] = v.value
    return rec


def jit(cache: Path, keep: Path, served: list[dict[str, Any]], arch: str) -> dict[str, Any]:
    """The SASS of each profiled kernel (`served`: its template and the registers the profiler read), read back out of the driver's
    compute cache: the CUDA images in it, and per kernel the cached function of its template, the sha256 of its cubin and of its
    SASS (one), its arch, registers and floating-point opcodes.  `ok` when every kernel is found, at `arch`, at the profiled
    registers, with its opcodes in `FP_OPS` and at least one HMMA.  Each cubin holding one is written to `keep`."""
    rec: dict[str, Any] = {"driver": driver_libraries(), "files": 0, "bytes": 0, "cubins": [], "kernels": []}
    images = []
    for p in sorted(x for x in cache.rglob("*") if x.is_file()):
        blob = p.read_bytes()
        rec["files"] += 1
        rec["bytes"] += len(blob)
        for off, img in cuda_elfs(blob):
            fns = elf_functions(img)
            images.append((hashlib.sha256(img).hexdigest(), img, fns))
            rec["cubins"].append({"file": str(p.relative_to(cache)), "offset": off, "bytes": len(img), "sha256": images[-1][0],
                                  "e_flags": f"{struct.unpack_from('<I', img, 48)[0]:#x}", "functions": len(fns),
                                  "marlin": sum(mangled_template(s) is not None for s in fns)})
    for k in served:
        hits = [(sha, img, sym, text) for sha, img, fns in images for sym, text in fns.items() if mangled_template(sym) == k["template"]]
        kr: dict[str, Any] = {"template": k["template"], "profiled_registers": k["registers"], "found": bool(hits), "ok": False}
        rec["kernels"].append(kr)
        if not hits:
            continue
        sha, img, sym, text = hits[0]
        keep.mkdir(parents=True, exist_ok=True)
        path = keep / f"{sha}.cubin"
        path.write_bytes(img)
        sass = _cuobjdump("-sass", "-fun", sym, str(path))
        m = re.search(r"code for (sm_\w+)", sass)
        kr.update({"symbol": sym, "cubins": sorted({h[0] for h in hits}),
                   "text_sha256": sorted({hashlib.sha256(h[3]).hexdigest() for h in hits}), "text_bytes": len(text),
                   "arch": m.group(1) if m else None,
                   "registers": function_registers(_cuobjdump("-res-usage", str(path))).get(sym),
                   "fp_ops": sass_census(sass), "kept": f"{keep.name}/{path.name}"})
        kr["ok"] = (len(kr["text_sha256"]) == 1 and kr["arch"] == arch and f"Function : {sym}" in sass
                    and kr["profiled_registers"] == [kr["registers"]] and set(kr["fp_ops"]) <= FP_OPS
                    and kr["fp_ops"].get("HMMA.16816.F32.BF16", 0) > 0)
    rec["ok"] = bool(rec["kernels"]) and all(k["ok"] for k in rec["kernels"])
    return rec


# ------------------------------------------------------------------------------------------------------------------------- weights
def weights(work: Path, seed: int) -> dict[str, Any]:
    """One layer's logical weights on the GPU, Marlin's repack of them, and the logical codes and scale bytes saved for the workers:
    codes [E][N][K] (every e2m1 code), scale bytes [E][N][K / 32] in SCALE_BYTES, the edge experts' partly from EDGE_BYTES; packed
    as the checkpoint is, two codes a byte, the even k in the low nibble (GPTQ's order, which `gptq_marlin_repack` reads)."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_moe_mxfp4_layer_for_marlin

    gen = torch.Generator(device="cuda").manual_seed(seed)
    edge = torch.tensor(EDGE_BYTES, device="cuda", dtype=torch.uint8)
    logical, paths = {}, {}
    for kind in ("w13", "w2"):
        K, N = shape(kind)
        q = torch.randint(0, 16, (NUM_EXPERTS, N, K), generator=gen, device="cuda", dtype=torch.uint8)
        s = torch.randint(*SCALE_BYTES, (NUM_EXPERTS, N, K // GROUP), generator=gen, device="cuda", dtype=torch.uint8)
        pick = edge[torch.randint(0, len(EDGE_BYTES), (EDGE_EXPERTS, N, K // GROUP), generator=gen, device="cuda")]
        hit = torch.rand((EDGE_EXPERTS, N, K // GROUP), generator=gen, device="cuda") < EDGE_FRACTION
        s[:EDGE_EXPERTS] = torch.where(hit, pick, s[:EDGE_EXPERTS])
        logical[kind] = (q, s)
        paths[kind] = _save(work / f"weights_{kind}", wq=q.cpu().numpy(), ws=s.cpu().numpy())
    pack = {k: (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous() for k, (q, _) in logical.items()}
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    w13, w2, s13, s2, _, _ = prepare_moe_mxfp4_layer_for_marlin(layer, pack["w13"], pack["w2"], logical["w13"][1], logical["w2"][1],
                                                                None, None)
    del pack, logical
    torch.cuda.empty_cache()
    return {"w13": w13, "w2": w2, "s13": s13, "s2": s2, "paths": paths,
            "info": {"w13": list(w13.shape), "w2": list(w2.shape), "s13": [list(s13.shape), str(s13.dtype)],
                     "s2": [list(s2.shape), str(s2.dtype)]}}


# ------------------------------------------------------------------------------------------------------------------------- launches
class Interceptor:
    """Wraps `vllm._custom_ops.moe_wna16_marlin_gemm`, which `fused_marlin_moe` calls through the module: each launch's bound
    arguments, its input, routing tensors and output cloned as soon as it returns (the w2 launch writes over the w13 launch's
    output: both live in intermediate_cache13)."""

    def __enter__(self):
        from vllm import _custom_ops as ops

        self.ops, self.inner = ops, ops.moe_wna16_marlin_gemm
        self.sig = inspect.signature(self.inner)
        self.launches: list[dict[str, Any]] = []
        self.label = ""
        ops.moe_wna16_marlin_gemm = self.tap
        return self

    def __exit__(self, *exc):
        self.ops.moe_wna16_marlin_gemm = self.inner

    def tap(self, *args, **kwargs):
        import torch

        b = self.sig.bind(*args, **kwargs)
        b.apply_defaults()
        call = dict(b.arguments)
        out = self.inner(**call)
        keep = ("input", "sorted_token_ids", "expert_ids", "num_tokens_past_padded", "topk_weights")
        self.launches.append({
            "label": self.label, "output": out.clone(), **{k: call[k].clone() for k in keep},
            "call": {**call, **{k: call[k].clone() for k in keep}},
            "args": {k: (str(v) if k == "b_q_type" else v) for k, v in call.items() if not torch.is_tensor(v) and v is not None},
            "absent": sorted(k for k in ABSENT if call[k] is None),
            "dtypes": {k: str(call[k].dtype) for k in ("input", "b_scales", "topk_weights", "b_qweight")} | {"output": str(out.dtype)},
            "workspace": int(call["workspace"].numel())})
        return out


def kind_of(L: dict[str, Any]) -> str:
    return "w13" if L["args"].get("size_n") == 2 * INTER else "w2"


# ------------------------------------------------------------------------------------------------------------------------- kernels
def _template_value(s: str) -> int | bool:
    s = re.sub(r"^\([^)]*\)", "", s.strip())
    if s in ("true", "false"):
        return s == "true"
    m = re.match(r"-?\d+", s)
    if m is None:
        raise ValueError(f"template argument {s!r}")
    return int(m.group())


def kernel_template(name: str) -> dict[str, Any] | None:
    """The template arguments of a demangled `marlin_moe_wna16::Marlin<...>(...)` (kernel.h:27-45), by `TEMPLATE`'s names; None when
    the name is another kernel's or does not parse."""
    i = name.find(MARLIN)
    if i < 0:
        return None
    i += len(MARLIN)
    depth, j = 1, i
    while depth and j < len(name):
        depth += {"<": 1, ">": -1}.get(name[j], 0)
        j += 1
    args = name[i:j - 1].split(",")
    try:
        return dict(zip(TEMPLATE, (_template_value(a) for a in args), strict=True)) if not depth else None
    except ValueError:
        return None


def marlin_kernels(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Every Marlin MoE kernel of a torch.profiler chrome trace, in launch order (one stream): its template, grid, block, registers."""
    ev = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel" and MARLIN in e.get("name", "")]
    out = []
    for e in sorted(ev, key=lambda e: (e.get("ts", 0), e.get("args", {}).get("correlation", 0))):
        a = e.get("args", {})
        out.append({"name": e["name"][:600], "template": kernel_template(e["name"]), "grid": a.get("grid"), "block": a.get("block"),
                    "registers": a.get("registers per thread"), "shared": a.get("shared memory")})
    return out


def kernel_config(k: dict[str, Any] | None, sms: int, types: list[int] | None, *, block: int) -> tuple[list[int] | None, bool]:
    """(config, ok) of one launch's kernel at moe block ``block``: config [TK, TN, blocks per SM] = [16 thread_k_blocks,
    16 thread_n_blocks, grid / sms]; ok when it is one of the block's thread configs at 1 to its clamp of blocks per SM, launched at
    threads x 1 x 1 on a grid of sms * b x 1 x 1, with the block's `kernel_statics` and, when given, the served [a, b, c, s] type
    ids."""
    t = (k or {}).get("template")
    if not t:
        return None, False
    grid, threads = list(k.get("grid") or [0, 0, 0]), list(k.get("block") or [0, 0, 0])
    tk, tn = 16 * t["thread_k_blocks"], 16 * t["thread_n_blocks"]
    b = grid[0] // sms if sms and grid[0] % sms == 0 else 0
    ok = ((tk, tn, t["threads"]) in MR.thread_configs(block) and 1 <= b <= MR.max_blocks_per_sm(block) and grid[1:] == [1, 1]
          and threads == [t["threads"], 1, 1] and all(t[n] == v for n, v in kernel_statics(block).items())
          and (types is None or [t["a_type"], t["b_type"], t["c_type"], t["s_type"]] == types))
    return [tk, tn, b], bool(ok)


def served_types() -> list[int]:
    """The [a, b, c, s] ScalarType ids ops.cu instantiates for bf16 activations over e2m1 weights with e8m0 scales and a bf16 output."""
    from vllm.scalar_type import scalar_types

    return [scalar_types.bfloat16.id, scalar_types.float4_e2m1f.id, scalar_types.bfloat16.id, scalar_types.float8_e8m0fnu.id]


# ------------------------------------------------------------------------------------------------------------------------- probe
def dequant64(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The exact weights [..., K] of e2m1 codes over e8m0 scale bytes [..., K / 32] (byte 0 reads +0 and 255 +inf, as the bf16 view)."""
    s = np.asarray(s).astype(np.float64)
    sc = np.where(s == 255, np.inf, np.where(s == 0, 0.0, np.exp2(s - 127)))
    return E2M1[np.asarray(q)] * np.repeat(sc, GROUP, axis=-1)


def probe(rng: np.random.Generator, row_experts: list[np.ndarray], wq, ws) -> tuple[np.ndarray, list[int]]:
    """bf16 words [rows][K]: random signs and mantissas over the PROBE_BINADES binades below 2, row r then times 2^t_r so the median
    |exact column sum| over the columns of its experts (`row_experts[r]`, into wq [E][N][K] and ws [E][N][K / 32]) is 2^PROBE_LOG2,
    where fp32 partial sums overflow in some summation orders and not in others; returns (words, t)."""
    K, rows = wq.shape[-1], len(row_experts)
    sign = rng.integers(0, 2, (rows, K)).astype(np.int64) << 15
    exp = 127 - rng.integers(0, PROBE_BINADES, (rows, K)).astype(np.int64)
    man = rng.integers(0, 128, (rows, K)).astype(np.int64)
    val = ((sign | (exp << 7) | man).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    sums: list[list[np.ndarray]] = [[] for _ in range(rows)]
    for e in sorted({int(v) for es in row_experts for v in es}):
        w = dequant64(wq[e], ws[e])
        for r, es in enumerate(row_experts):
            sums[r] += [w @ val[r]] * int(np.count_nonzero(np.asarray(es) == e))
    shifts = []
    for r in range(rows):
        s = np.abs(np.concatenate(sums[r])) if sums[r] else np.zeros(0)
        s = s[np.isfinite(s)]
        m = float(np.median(s)) if s.size else 0.0
        shifts.append(int(np.clip(np.round(PROBE_LOG2 - np.log2(m)), PROBE_BINADES - 127, 127)) if m > 0 else 0)
    x = sign | ((exp + np.array(shifts, dtype=np.int64)[:, None]) << 7) | man
    return x.astype(np.uint16), shifts


def _routing(gen, M: int, pool):
    import torch

    ids = torch.stack([pool[torch.randperm(pool.numel(), generator=gen, device="cuda")[:TOP_K]] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, TOP_K, generator=gen, device="cuda", dtype=torch.float32) + 0.05
    return ids.contiguous(), (w / w.sum(dim=1, keepdim=True) * 2.5).contiguous()


def _pool(gen, kind: str):
    import torch

    if kind == "edge":
        return torch.arange(EDGE_EXPERTS, device="cuda")
    pool = torch.arange(EDGE_EXPERTS, NUM_EXPERTS, device="cuda")
    return pool[torch.randperm(pool.numel(), generator=gen, device="cuda")[:HOT]] if kind == "hot" else pool


def assignment(st: np.ndarray, ex: np.ndarray, ntpp: int, ids_flat: np.ndarray, *,
               block: int) -> tuple[bool, np.ndarray, np.ndarray, int]:
    """(ok, blk, e, par) of one launch from `moe_align_block_size`'s outputs at moe block ``block``: each pair's moe block (its
    position in the sorted ids over ``block``) and that block's expert, and par = num_tokens_post_padded / ``block``; ok when every
    pair sits in exactly one block below par, whose expert routes it."""
    n = ids_flat.size
    blk, e, seen = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.int64)
    par = ntpp // block
    for b in range(min(int(ex.size), par)):
        p = st[b * block:(b + 1) * block]
        p = p[(p >= 0) & (p < n)]
        blk[p], e[p] = b, ex[b]
        seen[p] += 1
    return bool(ntpp % block == 0 and np.all(seen == 1) and np.array_equal(e, ids_flat)), blk, e, par


def statics_ok(L: dict[str, Any], kind: str, M: int) -> bool:
    """The launch is the served path's (marlin_moe.py:136-230): M's moe block (`block_of`), the launch kind's top_k, router weight
    and sizes, fp32 reduce without atomic add, full K, the e2m1 weight type over e8m0 scales and bf16 activations, nothing
    optional."""
    K, N = shape(kind)
    a = L["args"]
    want = {"moe_block_size": block_of(M), "top_k": launch_top_k(kind), "mul_topk_weights": kind == "w2", "size_m": prob_m(kind, M),
            "size_n": N, "size_k": K, "is_k_full": True, "use_atomic_add": False, "use_fp32_reduce": True, "is_zp_float": False,
            "b_q_type": "float4_e2m1f", "thread_k": -1, "thread_n": -1, "blocks_per_sm": -1}
    return (all(a.get(k) == v for k, v in want.items()) and L["absent"] == sorted(ABSENT)
            and L["dtypes"] == {"input": "torch.bfloat16", "b_scales": "torch.float8_e8m0fnu", "topk_weights": "torch.float32",
                                "b_qweight": "torch.int32", "output": "torch.bfloat16"})


def _launch_arrays(L: dict[str, Any], blk: np.ndarray, e: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    arrays = {"x": _u16(L["input"]), "C": _u16(L["output"]), "blk": blk, "e": e}
    if kind == "w2":
        arrays["g"] = _u32(L["topk_weights"]).reshape(-1)
    return arrays


def _inputs(seed: int, M: int, pool_kind: str) -> tuple:
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = _activation(gen, M, HIDDEN)
    return (x, *_routing(gen, M, _pool(gen, pool_kind)))


def profiled_runs(work: Path, W: dict[str, Any], runs: list[tuple]) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Each (label, x, ids, topk_weights) of `runs` through `fused_marlin_moe` under one Interceptor and ONE profiler session (a
    session per launch stops recording device activity after about 20: fa2_target_capture_gpu.kernels_by_launch); returns (the
    launches, the Marlin kernels in launch order, errors by label)."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    errors: dict[str, str] = {}
    with profile(activities=[ProfilerActivity.CUDA]) as prof, Interceptor() as icp:
        for label, x, ids, tw in runs:
            icp.label = label
            try:
                fused_marlin_moe(x, W["w13"], W["w2"], None, None, W["s13"], W["s2"], tw, ids,
                                 quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=NUM_EXPERTS)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                errors[label] = _error(e)
    path = work / "trace.json"
    prof.export_chrome_trace(str(path))
    kernels = marlin_kernels(json.loads(path.read_text()))
    path.unlink()
    return icp.launches, kernels, errors


def _kernel(k: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if k is None else {n: k[n] for n in ("template", "grid", "block", "registers", "shared", "name")}


def same_words(a, b) -> bool:
    """Two runs' outputs of one launch hold the same words, compared as bits: a NaN word (an edge expert's +inf scale byte on a
    +-0 code writes one) is unequal to itself as a float, so `torch.equal` on the bf16 outputs calls two runs that wrote the same
    words different on every case over the edge experts."""
    import torch

    return bool(a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def unmoved_rows_equal(a, b, moved: np.ndarray) -> bool:
    """The rows of two runs' outputs outside ``moved`` hold the same words, as bits per row: a pair the two align draws placed in
    different moe blocks of its expert reads a different k-slice schedule (module docstring, Replay determinism), so only the pairs
    both runs placed alike are held to each other; the moved ones are checked against the twin at their own run's assignment."""
    import torch

    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    same = (a.contiguous().view(torch.uint8) == b.contiguous().view(torch.uint8)).reshape(a.shape[0], -1).all(dim=1)
    keep = torch.ones(a.shape[0], dtype=torch.bool, device=same.device)
    if moved.size:
        keep[torch.as_tensor(np.asarray(moved, dtype=np.int64), device=same.device)] = False
    return bool(same[keep].all().item())


def replay(L: dict[str, Any]) -> bool:
    """Run1's recorded call again, into a SENTINEL-filled output with a fresh workspace, which must write run1's words: the served
    kernel's k-slice fold is lock-serialized in slice order (marlin_template.h:2162-2170), so a launch's words are a function of its
    call alone, the claim the record makes of the served path."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

    out = torch.full_like(L["output"].view(torch.int16), SENTINEL).view(torch.bfloat16)
    ops.moe_wna16_marlin_gemm(**{**L["call"], "output": out, "workspace": marlin_make_workspace_new(out.device, 4)})
    torch.cuda.synchronize()
    return same_words(out, L["output"])


def case_launches(work: Path, name: str, M: int, pool_kind: str, ids, launches: list[dict], W: dict[str, Any], sms: int,
                  types: list[int] | None) -> tuple[list[dict], list[dict], list[dict]]:
    """One case's records from its two runs' launches (labels `<name>/run1`, `<name>/run2`), each with its kernel; returns (launch
    records, launch specs for the jobs — run1's whole, and run2's moved pairs at run2's own assignment where the two align draws
    differ — and the captured run1 launches for the replay and forced phases)."""
    by = {(L["label"].split("/")[1], kind_of(L)): L for L in launches if L["label"].startswith(f"{name}/")}
    ids_flat = ids.cpu().numpy().astype(np.int64).reshape(-1)
    block = block_of(M)
    records, specs, captured = [], [], []
    for kind in ("w13", "w2"):
        L, twin = by.get(("run1", kind)), by.get(("run2", kind))
        if L is None or twin is None:
            raise RuntimeError(f"no {kind} launch in both runs: {sorted(by)}")
        ok, blk, e, par = assignment(L["sorted_token_ids"].cpu().numpy(), L["expert_ids"].cpu().numpy(),
                                     int(L["num_tokens_past_padded"].reshape(-1)[0]), ids_flat, block=block)
        ok2, blk2, e2, par2 = assignment(twin["sorted_token_ids"].cpu().numpy(), twin["expert_ids"].cpu().numpy(),
                                         int(twin["num_tokens_past_padded"].reshape(-1)[0]), ids_flat, block=block)
        moved = np.flatnonzero(blk != blk2)
        K, N = shape(kind)
        config, kernel_ok = kernel_config(L.get("kernel"), sms, types, block=block)
        rec = {"phase": "auto", "case": name, "launch": kind, "M": M, "pool": pool_kind, "pairs": int(ids_flat.size), "K": K, "N": N,
               "prob_m": prob_m(kind, M), "moe_block": block, "par": par, "experts_used": int(np.unique(ids_flat).size),
               "args": L["args"], "absent": L["absent"], "dtypes": L["dtypes"], "workspace": L["workspace"],
               "assignment_ok": bool(ok and ok2 and par2 == par and np.array_equal(e, e2)), "moved_pairs": int(moved.size),
               "unmoved_equal": unmoved_rows_equal(L["output"], twin["output"], moved),
               "statics_ok": statics_ok(L, kind, M) and statics_ok(twin, kind, M),
               "kernel": _kernel(L.get("kernel")), "config": config, "kernel_ok": kernel_ok,
               "config_deterministic": config == kernel_config(twin.get("kernel"), sms, types, block=block)[0]}
        records.append(rec)
        specs.append({"phase": "auto", "case": name, "launch": kind, "config": None, "M": M, "K": K, "N": N, "E": NUM_EXPERTS,
                      "top_k": launch_top_k(kind), "pairs": int(ids_flat.size), "par": par, "weights": W["paths"][kind],
                      "paths": _save(work / f"{name}_{kind}", **_launch_arrays(L, blk, e, kind))})
        if moved.size:
            specs.append({"phase": "auto", "run": 2, "case": name, "launch": kind, "config": None, "M": M, "K": K, "N": N,
                          "E": NUM_EXPERTS, "top_k": launch_top_k(kind), "pairs": int(ids_flat.size), "par": par,
                          "weights": W["paths"][kind], "moved": [int(v) for v in moved],
                          "paths": _save(work / f"{name}_{kind}_run2", **_launch_arrays(twin, blk2, e, kind))})
        captured.append({"case": name, "launch": kind, "M": M, "L": L, "blk": blk, "e": e, "par": par})
    return records, specs, captured


def sweep_record(launches: list[dict], sms: int, types: list[int] | None) -> dict[str, Any]:
    """The served config of each launch kind at each M of SERVED_M, from the sweep's launches (labels `sweep/<M>`); `ok` when every
    (kind, M) has a launch at the served statics whose kernel reads; `model` the limits under which ops.cu's rule gives them."""
    served: dict[str, dict[str, list[int] | None]] = {"w13": {}, "w2": {}}
    statics = True
    for L in launches:
        if L["label"].startswith("sweep/"):
            M, kind = int(L["label"].split("/")[1]), kind_of(L)
            config, ok = kernel_config(L.get("kernel"), sms, types, block=block_of(M))
            served[kind][str(M)] = config if ok else None
            statics = statics and statics_ok(L, kind, M)
    ok = statics and all(served[k].get(str(M)) for k in served for M in SERVED_M)
    return {"served": served, "statics_ok": statics, "model": model(served, sms), "ok": bool(ok)}


def served_config(sweep: dict[str, Any], kind: str, M: int) -> list[int] | None:
    return sweep.get("served", {}).get(kind, {}).get(str(M))


def last_cuda_error() -> int | None:
    """`cudaGetLastError` of the CUDA runtime this process loaded, which also clears it, so that a forced launch the device refused
    does not surface at the next torch launch; None without a dynamically loaded runtime (a static one keeps its own error)."""
    with open("/proc/self/maps") as f:
        lib = next((ln.split()[-1] for ln in f if "libcudart" in ln and ln.split()[-1].startswith("/")), None)
    return None if lib is None else int(ctypes.CDLL(lib).cudaGetLastError())


def forced_launches(work: Path, W: dict[str, Any], cap: dict[str, Any], sms: int,
                    seed: int) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """One captured launch again at every candidate config of its moe block (`candidates`), forced, on the probe input into a
    SENTINEL-filled output; returns (records, specs, the probe's record: its row shifts and the accepted configs grouped by identical
    words)."""
    import hashlib

    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

    kind, L = cap["launch"], cap["L"]
    K, N = shape(kind)
    tk_ = launch_top_k(kind)
    w = _load(W["paths"][kind])
    xs, shifts = probe(np.random.default_rng(seed), [cap["e"][r * tk_:(r + 1) * tk_] for r in range(cap["e"].size // tk_)], w["wq"], w["ws"])
    if list(xs.shape) != list(L["input"].shape):
        raise RuntimeError(f"probe {list(xs.shape)} for input {list(L['input'].shape)}")
    x = torch.from_numpy(xs.view(np.int16)).to(L["input"].device).view(torch.bfloat16)
    launches, specs, classes = [], [], {}
    for tk, tn, b in candidates(block_of(cap["M"])):
        out = torch.full_like(L["output"].view(torch.int16), SENTINEL).view(torch.bfloat16)
        rec = {"phase": "forced", "case": cap["case"], "launch": kind, "M": cap["M"], "config": [tk, tn, b], "pairs": int(out.shape[0]),
               "K": K, "N": N, "refused": None}
        last_cuda_error()
        try:
            ops.moe_wna16_marlin_gemm(**{**L["call"], "input": x, "output": out, "workspace": marlin_make_workspace_new(out.device, 4),
                                         "thread_k": tk, "thread_n": tn, "blocks_per_sm": b})
            torch.cuda.synchronize()
            rec["cuda_error"] = last_cuda_error()
            written = bool((out.view(torch.int16) != SENTINEL).all())
            if rec["cuda_error"] or not written:
                rec["refused"] = f"cuda error {rec['cuda_error']}, every word written: {written}"
        except RuntimeError as e:
            rec["refused"] = _error(e)[:300]
            rec["cuda_error"] = last_cuda_error()
        rec["accepted"] = rec["refused"] is None
        launches.append(rec)
        if rec["accepted"]:
            words = _u16(out)
            rec["nonfinite"] = int(((words & 0x7F80) == 0x7F80).sum())
            classes.setdefault(hashlib.sha256(words.tobytes()).hexdigest(), []).append([tk, tn, b])
            specs.append({"phase": "forced", "case": cap["case"], "launch": kind, "config": [tk, tn, b], "statics": [tk, tn, sms * b],
                          "M": cap["M"], "K": K, "N": N, "E": NUM_EXPERTS, "top_k": tk_, "pairs": rec["pairs"],
                          "par": cap["par"], "weights": W["paths"][kind],
                          "paths": _save(work / f"forced_{cap['case']}_{kind}_{tk}_{tn}_{b}",
                                         **_launch_arrays({**L, "input": x, "output": out}, cap["blk"], cap["e"], kind))})
    accepted = sum(r["accepted"] for r in launches)
    probe_rec = {"case": cap["case"], "launch": kind, "shifts": shifts, "accepted": accepted, "classes": list(classes.values()),
                 "ok": accepted < 2 or len(classes) > 1}
    return launches, specs, probe_rec


# ---------------------------------------------------------------------------------------------------------------- comparisons
def _operands(job: dict[str, Any], p: np.ndarray) -> tuple:
    a, w = _load(job["paths"]), _load(job["weights"])
    x = np.asarray(a["x"])[p // job["top_k"]]
    g = np.asarray(a["g"])[p] if "g" in a else None
    return a, w, x, np.asarray(a["blk"])[p], np.asarray(a["e"])[p], g


def word_jobs(spec: dict[str, Any], statics: list[int]) -> list[dict[str, Any]]:
    """One job per chunk of the spec's pairs: all of them, or only `moved` (run2's pairs whose moe block moved, indices into it)."""
    n = len(spec["moved"]) if "moved" in spec else spec["pairs"]
    return [{**spec, "job": "words", "statics": list(statics), "pair_range": [p0, p1]}
            for p0, p1 in _chunks(n, spec["N"] * spec["K"])]


def _chunks(n: int, cost: int) -> list[tuple[int, int]]:
    step = max(1, JOB_COORD_K // max(1, cost))
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _twin(job: dict[str, Any], p: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, w, x, blk, e, g = _operands(job, p)
    TK, TN, GRID = job["statics"]
    want = MR.launch(x, blk, job["par"], e, w["wq"], w["ws"], cols, TK=TK, TN=TN, GRID=GRID, g=g)
    return want, np.asarray(a["C"])[p][:, cols]


def compare_words(job: dict[str, Any]) -> dict[str, Any]:
    p0, p1 = job["pair_range"]
    p = np.asarray(job["moved"][p0:p1], dtype=np.int64) if "moved" in job else np.arange(p0, p1)
    want, got = _twin(job, p, np.arange(job["N"]))
    bad = np.argwhere(want != got)
    rec = {"words": int(want.size), "mismatch": len(bad), "first_mismatch": None}
    if len(bad):
        i, n = (int(v) for v in bad[0])
        rec["first_mismatch"] = {"pair": int(p[i]), "column": n, "want": int(want[i, n]), "kernel": int(got[i, n])}
    return rec


def reference_job(spec: dict[str, Any], statics: list[int]) -> dict[str, Any]:
    return {**spec, "job": "reference", "statics": list(statics)}


def compare_reference(job: dict[str, Any]) -> dict[str, Any]:
    """The plain-integer reference against the twin at the launch's statics: the schedule on its first four blocks, two pairs' sampled
    coordinates, and for the w2 launch one coordinate through the expert gather over every expert's rows."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.definitions.numerics.derived_rows import f2fp_bf16
    from verity.ml.definitions.registry import marlin_moe as MM

    TK, TN, GRID = job["statics"]
    K, N, E, P, par = job["K"], job["N"], job["E"], job["pairs"], job["par"]
    KT, NT = K // TK, N // TN
    routed = job["launch"] == "w2"
    _, w, x, blk, e, g = _operands(job, np.arange(P))
    rec = {"reference_schedules": 0, "reference_coords": 0, "reference_expert_coords": 0, "twin_mismatch": 0}
    sched = bind(MM.MarlinMoeSchedule, KT=KT, NT=NT, GRID=GRID)
    cuts = {}
    for b in sorted(set(blk.tolist()))[:4]:
        cuts[b] = MR.schedule(KT, NT, GRID, b, par)
        got = np.array(evaluate(sched, [b], [par]), dtype=bool).reshape(NT, KT - 1)
        rec["reference_schedules"] += 1
        rec["twin_mismatch"] += not np.array_equal(got, cuts[b])
    coord = bind(MM.MarlinMxfp4CoordinateW if routed else MM.MarlinMxfp4Coordinate, K=K, TK=TK, DOT=MM.DOT)
    gb = None if g is None else f2fp_bf16(np.asarray(g, dtype=np.uint32))
    for p in sorted({0, P - 1}):
        cut_p = cuts.get(int(blk[p]))
        if cut_p is None:
            cut_p = MR.schedule(KT, NT, GRID, int(blk[p]), par)
        for n in sorted({0, N - 1, (p * 7919) % N}):
            q, s, cut = w["wq"][e[p], n], w["ws"][e[p], n], cut_p[n // TN]
            lead = [x[p].tolist(), cut.astype(int).tolist()] + ([int(gb[p])] if routed else [])
            y = evaluate(coord, *lead, q.tolist(), s.tolist())
            twin = MR.coordinates(x[p][None], cut[None], q[None], s[None], TK=TK, g=None if gb is None else gb[p:p + 1])
            rec["reference_coords"] += 1
            rec["twin_mismatch"] += y != (int(twin[0]),)
    if routed:
        n = (P * 7919) % N
        cut = cuts.get(int(blk[0]), MR.schedule(KT, NT, GRID, int(blk[0]), par))[n // TN]
        fn = bind(MM.MarlinMoeExpertCoordinateW, E=E, K=K, TK=TK, DOT=MM.DOT)
        y = evaluate(fn, x[0].tolist(), cut.astype(int).tolist(), int(e[0]), int(gb[0]), MR.pack_e2m1(w["wq"][:, n]).ravel().tolist(),
                     np.asarray(w["ws"][:, n]).ravel().tolist())
        twin = MR.coordinates(x[0][None], cut[None], w["wq"][e[0], n][None], w["ws"][e[0], n][None], TK=TK, g=gb[:1])
        rec["reference_expert_coords"] += 1
        rec["twin_mismatch"] += y != (int(twin[0]),)
    return rec


def compare(job: dict[str, Any]) -> dict[str, Any]:
    head = {k: job.get(k) for k in ("job", "phase", "run", "case", "launch", "config", "statics", "pair_range")}
    try:
        return {**head, **{"words": compare_words, "reference": compare_reference}[job["job"]](job)}
    except Exception as e:  # noqa: BLE001
        return {**head, "error": _error(e)}


# ------------------------------------------------------------------------------------------------------------- the served config
def _auto(M: int, kind: str, sms: int, allow: dict[tuple[int, int], int]) -> tuple[int, int, int] | None:
    """ops.cu's config for one launch kind at M tokens, over ``allow``, the per-config limits of the kernels at M's moe block."""
    N = shape(kind)[1]
    try:
        return MR.auto_config(prob_m(kind, M), N, launch_top_k(kind), sms, allow, block=block_of(M))
    except ValueError:
        return None


def model(served: dict[str, dict[str, list[int] | None]], sms: int) -> dict[str, Any]:
    """Per launch kind and moe block of SERVED_M (`<kind>/<block>`), the per-config limits `allow` (each of the block's three
    configs' in 0 to its clamp, 0 invalid) under which ops.cu's rule (`MR.auto_config`) gives the swept config at every swept M of
    that block: information on why the kernel runs what it runs (the limits read the shared-memory cache, which depends on K, N and
    the block's m blocks, and the registers of a driver-compiled kernel, another instantiation at each block)."""
    out: dict[str, Any] = {}
    for kind, by_m in sorted(served.items()):
        swept = {int(M): c for M, c in by_m.items() if c}
        for block in sorted({block_of(M) for M in SERVED_M}):
            consistent = MR.allows(swept, block, shape(kind)[1], TOP_K, launch_top_k(kind), NUM_EXPERTS, sms)
            out[f"{kind}/{block}"] = {"observed": sum(block_of(M) == block for M in swept), "consistent": len(consistent),
                                      "allow": [{f"{tk}/{tn}": v for (tk, tn), v in allow.items()} for allow in consistent[:8]]}
    return out


# ------------------------------------------------------------------------------------------------------------------------- bind
def bind_check(served: set[tuple[str, int, int, int]]) -> dict[str, Any]:
    """`MarlinMoeExpertGemm(W)_v1` bind at every served (kind, TK, TN, GRID): the weight slabs [E][N][K] and [E][N][K / 32], and the
    Boolean version each names."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.batch import type_shape
    from verity.ml.definitions.boolean import boolean_version
    from verity.ml.definitions.registry import marlin_moe as MM

    out: dict[str, Any] = {}
    for kind, TK, TN, GRID in sorted(served):
        K, N = shape(kind)
        fn = bind(MM.MarlinMoeExpertGemm if kind == "w13" else MM.MarlinMoeExpertGemmW, E=NUM_EXPERTS, K=K, N=N, TK=TK, TN=TN, GRID=GRID,
                  DOT=MM.DOT)
        params, ret = fn.signature()
        p = {k: list(type_shape(t)) for k, t in params}
        ok = p["wq"] == [NUM_EXPERTS, N, K // 2] and p["ws"] == [NUM_EXPERTS, N, K // GROUP] and p["x"] == [K] and list(type_shape(ret)) == [N]
        out[descriptor_id(fn)] = {"ok": ok, "boolean": descriptor_id(boolean_version(fn))}
    return {"bindings": out, "ok": bool(out) and all(v["ok"] for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
TOTALS = ("words", "mismatch", "reference_schedules", "reference_coords", "reference_expert_coords", "twin_mismatch")


def _launch_ok(x: dict[str, Any]) -> bool:
    if x["phase"] == "forced":
        return True
    return bool(x["unmoved_equal"] and x["replay_ok"] and x["statics_ok"] and x["assignment_ok"] and x["kernel_ok"]
                and x["config_deterministic"] and x.get("identified") is True)


def _words_ok(t: dict[str, Any]) -> bool:
    return (t["words"] > 0 and t["mismatch"] == 0 and t["reference_schedules"] > 0 and t["reference_coords"] > 0
            and t["reference_expert_coords"] > 0 and t["twin_mismatch"] == 0)


def _forced_ok(doc: dict[str, Any]) -> bool:
    """Each kind has an accepted forced launch, and every forced launch's probe tells two accepted configs apart; nothing to hold
    when the phase is off or `--cases` picked no FORCED case (a forced phase that failed to run leaves its error in the verdict)."""
    forced = [x for x in doc["launches"] if x["phase"] == "forced"]
    if not doc.get("forced_phase") or (not forced and not doc.get("probe")):
        return True
    probes = doc.get("probe", {})
    return (all(any(x["accepted"] for x in forced if x["launch"] == k) for k in ("w13", "w2"))
            and {f"{x['case']}/{x['launch']}" for x in forced} == set(probes) and all(p["ok"] for p in probes.values()))


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each case's own verdict: its launches' checks, the configs it ran (`auto` launches at their kernel's config, `forced` ones that
    were accepted), its probes and its words, so a NO-GO names what failed."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        c = out.setdefault(x["case"], {"launches_ok": True, "totals": {k: 0 for k in TOTALS}, "configs": [], "refused": 0})
        c["launches_ok"] = c["launches_ok"] and _launch_ok(x)
        if x["phase"] == "auto":
            c["configs"].append(f"{x['launch']}:{'/'.join(map(str, x.get('config') or ['?']))}")
        else:
            c["refused"] += not x["accepted"]
    for p in doc.get("probe", {}).values():
        c = out.get(p["case"])
        if c is not None:
            c["launches_ok"] = c["launches_ok"] and p["ok"]
    for e in doc["errors"]:
        name = e.split("/")[1] if e.startswith("case/") else None
        if name and name not in out:
            out[name] = {"launches_ok": False, "totals": {k: 0 for k in TOTALS}, "configs": [], "refused": 0}
    for r in doc.get("comparisons", []):
        c = out.get(r["case"])
        if c is not None:
            for k in TOTALS:
                c["totals"][k] += r[k]
    for name, c in out.items():
        t = c["totals"]
        errors = [e for e in doc["errors"] if e == f"case/{name}" or e.startswith((f"case/{name}/", f"compare/{name}/"))]
        auto = any(x["phase"] == "auto" for x in doc["launches"] if x["case"] == name)
        c["ok"] = (c["launches_ok"] and auto and t["words"] > 0 and t["mismatch"] == 0 and t["twin_mismatch"] == 0
                   and t["reference_coords"] > 0 and not errors)
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    launches = doc["launches"]
    return (doc["gate"]["ok"] and not doc["errors"] and doc["selection"].get("ok") is True and doc["bind"].get("ok") is True
            and doc["sweep"].get("ok") is True and doc.get("jit", {}).get("ok") is True
            and {x["launch"] for x in launches if x["phase"] == "auto"} == {"w13", "w2"} and all(_launch_ok(x) for x in launches)
            and _forced_ok(doc) and _words_ok(doc["totals"]))


def _key(r: dict[str, Any]) -> tuple:
    return (r["phase"], r.get("run") or 1, r["case"], r["launch"], tuple(r["config"] or ()))


def _pool_map(ex, jobs: list[dict[str, Any]], doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for res in ex.map(compare, jobs):
        if "error" in res:
            doc["errors"][f"compare/{res['case']}/{res['launch']}/{res['job']}/{res['config']}"] = res["error"]
        else:
            out.append(res)
    return out


def _say(msg: str) -> None:
    print(f"[dsv4-marlin] {msg}", flush=True)


def gpu_phases(doc: dict[str, Any], work: Path, sms: int, quick: bool, forced: bool,
               only: tuple[str, ...] | None = None) -> tuple[list[dict], list[dict]]:
    """selection, binary, the weights, every case's auto launches (``only`` restricts them to named CASES) and the forced ones;
    returns (auto specs, forced specs)."""
    for phase, fn in (("selection", selection), ("binary", binary)):
        try:
            doc[phase] = fn()
        except Exception as e:  # noqa: BLE001
            doc[phase] = {"ok": False}
            doc["errors"][phase] = _error(e)
        _say(f"{phase} {json.dumps(doc[phase])[:1200]}")
    try:
        W = weights(work, 10_000)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["weights"] = _error(e)
        _say(f"weights ERROR {doc['errors']['weights'][:600]}")
        return [], []
    doc["weights"] = W["info"]
    _say(f"weights {json.dumps(W['info'])}")
    try:
        types = served_types()
    except Exception as e:  # noqa: BLE001
        types = None
        doc["errors"]["types"] = _error(e)
    names = set(only) if only else (set(QUICK) if quick else {c[0] for c in CASES})
    picked = [(i, name, M, pool) for i, (name, M, pool) in enumerate(CASES) if name in names]
    runs, routed = [], {}
    for i, name, M, pool in picked:
        x, ids, tw = _inputs(11_000 + i, M, pool)
        routed[name] = ids
        runs += [(f"{name}/run1", x, ids, tw), (f"{name}/run2", x, ids, tw)]
    runs += [(f"sweep/{M}", *_inputs(12_000 + M, M, "uniform")) for M in SERVED_M]
    try:
        launches, kernels, run_errors = profiled_runs(work, W, runs)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["runs"] = _error(e)
        _say(f"runs ERROR {doc['errors']['runs'][:600]}")
        return [], []
    doc["errors"].update({label if label.startswith("sweep/") else f"case/{label}": err for label, err in run_errors.items()})
    served: dict[str, dict[str, Any]] = {}
    for k in kernels:
        if k["template"]:
            s = served.setdefault(json.dumps(k["template"], sort_keys=True), {"template": k["template"], "registers": []})
            s["registers"] = sorted(set(s["registers"]) | {k["registers"]}, key=str)
    doc["kernels"] = {"launches": len(launches), "kernels": len(kernels), "registers": sorted({str(k["registers"]) for k in kernels}),
                      "served": list(served.values())}
    if len(kernels) == len(launches):
        for L, k in zip(launches, kernels):
            L["kernel"] = k
    else:
        doc["errors"]["kernels"] = f"{len(kernels)} Marlin kernels in the profile for {len(launches)} launches"
    _say(f"kernels {json.dumps(doc['kernels'])}")
    doc["sweep"] = sweep_record(launches, sms, types)
    _say(f"sweep ok={doc['sweep']['ok']} served={json.dumps(doc['sweep']['served'])} "
         f"model={json.dumps({k: v['consistent'] for k, v in doc['sweep']['model'].items()})}")
    specs, captured = [], []
    for _, name, M, pool in picked:
        try:
            records, sp, cap = case_launches(work, name, M, pool, routed[name], launches, W, sms, types)
        except Exception as e:  # noqa: BLE001
            doc["errors"][f"case/{name}"] = _error(e)
            _say(f"auto {name} ERROR {doc['errors'][f'case/{name}'][:600]}")
            continue
        for x, c in zip(records, cap):
            try:
                x["replay_ok"] = replay(c["L"])
            except Exception as e:  # noqa: BLE001
                x["replay_ok"] = False
                doc["errors"][f"case/{name}/replay/{x['launch']}"] = _error(e)
        doc["launches"] += records
        specs += sp
        captured += [c for c in cap if forced and name in (QUICK_FORCED if quick else FORCED)]
        for x in records:
            _say(f"auto {name:10s} {x['launch']:3s} M={M:4d} block={x['moe_block']:2d} pairs={x['pairs']:4d} par={x['par']:4d} "
                 f"experts={x['experts_used']:3d} "
                 f"moved={x['moved_pairs']:4d} unmoved_eq={x['unmoved_equal']} replay={x['replay_ok']} statics={x['statics_ok']} "
                 f"assignment={x['assignment_ok']} kernel={x['config']} kernel_ok={x['kernel_ok']}")
    forced_specs = []
    doc["probe"] = {}
    for j, cap in enumerate(captured):
        key = f"case/{cap['case']}/forced/{cap['launch']}"
        try:
            records, sp, pr = forced_launches(work, W, cap, sms, 13_000 + j)
        except Exception as e:  # noqa: BLE001
            doc["errors"][key] = _error(e)
            _say(f"forced {cap['case']} {cap['launch']} ERROR {doc['errors'][key][:600]}")
            continue
        doc["launches"] += records
        doc["probe"][f"{cap['case']}/{cap['launch']}"] = pr
        forced_specs += sp
        _say(f"forced {cap['case']:9s} {cap['launch']:3s} accepted={[x['config'] for x in records if x['accepted']]} "
             f"refused={[x['config'] for x in records if not x['accepted']]} probe classes={pr['classes']} shifts={pr['shifts']}")
    return specs, forced_specs


def compare_all(ex, doc: dict[str, Any], specs: list[dict], forced_specs: list[dict], sms: int) -> None:
    """Every auto launch's words at its kernel's config with its reference job, run2's moved pairs at run2's own assignment, and
    every accepted forced launch's words at its own config; `doc["comparisons"]` per (phase, run, case, launch, config)."""
    auto = {(x["case"], x["launch"]): x for x in doc["launches"] if x["phase"] == "auto"}
    jobs = []
    for s in specs:
        x = auto[(s["case"], s["launch"])]
        at = x["config"] if x["kernel_ok"] else None
        if s.get("run") != 2:
            x["served"] = served_config(doc["sweep"], s["launch"], s["M"])
            x["identified"] = bool(x["kernel_ok"] and x["config"] == x["served"])
        if at is not None:
            statics = [at[0], at[1], sms * at[2]]
            jobs += word_jobs({**s, "config": list(at)}, statics)
            if s.get("run") != 2:
                jobs += [reference_job({**s, "config": list(at)}, statics)]
    jobs += [j for s in forced_specs for j in word_jobs(s, s["statics"])]
    _say(f"comparing: {len(jobs)} jobs")
    results: dict[tuple, dict[str, Any]] = {}
    for res in _pool_map(ex, jobs, doc):
        agg = results.setdefault(_key(res), {"phase": res["phase"], "run": res.get("run") or 1, "case": res["case"],
                                             "launch": res["launch"], "config": res["config"],
                                             **{k: 0 for k in TOTALS}, "first_mismatch": None})
        for k in TOTALS:
            agg[k] += res.get(k, 0)
        agg["first_mismatch"] = agg["first_mismatch"] or res.get("first_mismatch")
    doc["comparisons"] = list(results.values())
    for r in doc["comparisons"]:
        _say(f"words {r['phase']:6s} run{r['run']} {r['case']:9s} {r['launch']:3s} {'/'.join(map(str, r['config'] or [])):10s} "
             f"words={r['words']} !={r['mismatch']} ref={r['reference_schedules']}+{r['reference_coords']}"
             f"+{r['reference_expert_coords']} twin!={r['twin_mismatch']} "
             f"{json.dumps(r['first_mismatch']) if r['first_mismatch'] else ''}")


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool,
        forced: bool, driver: str, only: tuple[str, ...] | None = None) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env = environment(work / "compute-cache")
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "quick": quick, "forced_phase": forced,
                           "environment": G._environment(), "gate": gate(min_driver, want_cc, want_sms, driver), "errors": {},
                           "launches": [], "selection": {}, "sweep": {}, "probe": {}, "comparisons": []}
    _say(f"gate {json.dumps(doc['gate'])}")
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("DSV4-MARLIN GATE-FAIL", flush=True)
        raise SystemExit(3)
    sms = int(doc["gate"]["sms"])
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        specs, forced_specs = gpu_phases(doc, work, sms, quick, forced, only)
    try:
        doc["jit"] = jit(work / "compute-cache", outdir / "jit", doc.get("kernels", {}).get("served", []),
                         "sm_" + want_cc.replace(".", ""))
    except Exception as e:  # noqa: BLE001
        doc["errors"]["jit"] = _error(e)
        doc["jit"] = {"ok": False}
    seen = [{k: x.get(k) for k in ("symbol", "text_sha256", "registers", "fp_ops", "ok")} for x in doc["jit"].get("kernels", [])]
    _say(f"jit ok={doc['jit']['ok']} files={doc['jit'].get('files')} cubins={len(doc['jit'].get('cubins', []))} "
         f"kernels={json.dumps(seen)[:2000]}")
    _say(f"comparing on {workers} workers")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        compare_all(ex, doc, specs, forced_specs, sms)
    served = {(k, c[0], c[1], sms * c[2]) for k in ("w13", "w2") for M in SERVED_M for c in [served_config(doc["sweep"], k, M)] if c}
    try:
        doc["bind"] = bind_check(served)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    _say(f"bind {json.dumps(doc['bind'])[:1500]}")
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    for name, c in doc["cases"].items():
        t = c["totals"]
        _say(f"case {name:9s} {'GO' if c['ok'] else 'NO-GO':5s} launches_ok={c['launches_ok']} words={t['words']} !={t['mismatch']} "
             f"twin!={t['twin_mismatch']} forced_refused={c['refused']} {' '.join(c['configs'])}")
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    served1 = {k: served_config(doc["sweep"], k, 1) for k in ("w13", "w2")}
    print(f"DSV4-MARLIN {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} totals={json.dumps(doc['totals'])} served@M1={json.dumps(served1)} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepSeek-V4-Pro's Marlin MXFP4 expert GEMMs on this Hopper GPU against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (logical weights, launch words); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--driver", default=None, help="the exact driver version (nvidia-smi); default the capture map's target.driver")
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"cases {QUICK}, forced {QUICK_FORCED}: a first signal in minutes")
    ap.add_argument("--skip-forced", action="store_true", help="no forced launches (their check is then not part of the verdict)")
    ap.add_argument("--cases", default=None, help="comma-separated CASES names to run, e.g. decode460,hot1024 (default all, "
                                                  "or QUICK under --quick); the sweep always runs whole, and a subset without "
                                                  f"a case of {FORCED} runs no forced phase")
    a = ap.parse_args(argv)
    only = tuple(a.cases.split(",")) if a.cases else None
    if only and (unknown := sorted(set(only) - {c[0] for c in CASES})):
        ap.error(f"unknown cases {unknown}; known: {[c[0] for c in CASES]}")
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick, not a.skip_forced,
              a.driver or pinned_driver(), only)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
