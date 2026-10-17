"""GLM-5.3's TP8 all-reduces on 8x H200 against their Programs, a step of the 8-GPU run that needs no checkpoint (run it before an
engine holds the GPUs).  The served engine (`WORKLOAD`) runs vLLM's custom all-reduce as the workload's ``disable_custom_all_reduce``
has it, false (vLLM's default), so it runs vLLM's defaults (`tp_allreduce.SETTINGS`: custom all-reduce, torch's symmetric memory
and FlashInfer's all-reduce on); ``--served custom-off`` serves with the custom all-reduce off, as the two GO runs did
(`tp_allreduce.CAPTURE_RUNS`).  Either way, at vLLM d9105ea80 on cc 9.0 (`expected_comm`) its TP group's all-reduces of GLM-5.3's
hidden states take three routes:

  decode    up to `FUSED_MAX` (42) tokens, FlashInfer's MNNVL kernels: `fused_allreduce_rms_norm`'s fused all-reduce + residual
            add + RMSNorm (`tp_allreduce.AllReduceRmsNormMnnvl_v1`) and the standalone all-reduce (`fi_ar_comm`,
            `tp_allreduce.AllReduceMnnvl_v1`), one-shot to 10 tokens and two-shot past;
  prefill   past it, to 64 MiB, torch's symmetric memory: `multimem_all_reduce_`, whose f32 adds happen in the NVSwitch in an
            order no source states (`tp_allreduce.AllReduceMultimem_v1`, parameterised by a tree per owner rank, TREES, and FTZ);
            this run reads the trees off the switch with probes and pins them.  The custom all-reduce, when on, takes none of
            these: beside symmetric memory its cap is 256 KiB, inside FlashInfer's 0.5 MiB;
  switch    the fallback when the trees don't pin: ``VLLM_ALLREDUCE_USE_SYMM_MEM=0`` with the custom all-reduce on, which runs vLLM's
            custom all-reduce from 43 to 682 tokens, always 2-stage (`tp_allreduce.AllReduceCustom_v1`), and NCCL past it.

A fourth set of rank processes runs DeepSeek-V4-Pro's all-reduces (`DSV4_HIDDEN`, plain `tensor_model_parallel_all_reduce`) at
vLLM's defaults: FlashInfer to 36 tokens (one-shot to 9), multimem from 37, with a probe of its own; its verdict is the record's
``h7168``, apart from GLM-5.3's.

This process makes every operand, spawns one GPU process per rank for each set of rank processes (`spawns`), which run vLLM's own
calls on their rank's operands and save their words, and compares those words with the numpy models (`tp_allreduce_check`).  It
writes `<out>/glm53_tp8_allreduce_capture.json`, the property record `glm53_tp8_allreduce_capture` (`properties.record`), the
probes' words under `<out>/words/`, prints one line per case and then `GLM53-TP8-ALLREDUCE GO` or `NO-GO` with the served
setting, the route prefill takes, its trees and FTZ, each shot's FMA and the h7168 verdict.  The device gate runs first and
nothing else runs when it fails (exit 3).

  gate       driver, cc 9.0, 132 SMs, `WORLD` GPUs of one name (an 8x H200 node), ninja and nvcc on PATH (FlashInfer JIT-builds
             its TRT-LLM all-reduce when a rank first makes its workspace); the NVLink topology is recorded.
  fused      (served) at M in `FUSED_M` (both shots of `mnnvl_shot` and every cluster geometry of `mnnvl_grid`), the special tokens
             and one screened probe call per shot: every call on FlashInfer's "mnnvl" workspace at `mnnvl_shot(M)`, weight_bias 0,
             eps `EPS`, and past `FUSED_MAX` on the all-reduce then the norm module; every rank's normed and residual words against
             `mnnvl_norm_rows` at both contractions of ``sum * rcp(N) + EPS``, which decides `fma` per shot.
  allreduce  (served) `tensor_model_parallel_all_reduce` at M in `AR_M`: the communicator that ran it (`expected_comm`), the
             multimem kernel past `FUSED_MAX`, and every word against `mnnvl_rows` at `shot_of(M)` or `multimem_rows` at the
             probes' trees and FTZ; row 0 holds -0s, a cancellation, subnormals, a NaN and order-dependent sums (`ORDER`).
  probe      (served) all-reduces at M in `PROBE_M_AR` of `probe_parts`: at each pair (i, j) of ranks, +B on i, -B on j and
             2^0..2^5 on the six others; the sum is the others the switch adds outside the add that cancels B, so each pair names
             the ranks under its two ranks' lowest common add (`decode_probe`), and the 28 pairs one tree per owner
             (`tree_from_clusters`), at B = 2^40 and 2^100 (an accumulator wider than f32 absorbs only the second).  Its specials
             decide FTZ (a subnormal kept or flushed), and -0, NaN and inf sums are compared with the model like every word.
  again      (served_again) the probes in a second set of rank processes, whose trees must be the first set's.
  switch     (switch) `tensor_model_parallel_all_reduce` at M in `SWITCH_M` against `custom_rows` at `custom_stage`; past 682 the
             launch's NCCL route is recorded, outside the switch's verdict (no Program models it and the workload stays below it).
  h7168      (h7168) `tensor_model_parallel_all_reduce` at M in `DSV4_AR_M` of `DSV4_HIDDEN`-wide tokens, and h7168_probe the
             probe at `DSV4_PROBE_M`, whose trees and FTZ model its multimem words.
  dispatch   per set of rank processes, the communicators vLLM made match its setting: the custom all-reduce on or off and, on,
             its cap (256 KiB beside symmetric memory, else 8 MiB); symmetric memory on or off, 64 MiB; FlashInfer's 0.5 MiB.
  reference  per launch, the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`) of the word Definition the launch
             binds, on sampled tokens or elements, which the model must equal.
  bind       every Definition binds at GLM-5.3's statics (the multimem one at the probes' trees).
  timing     rank 0's median of `REPS` CUDA-event-timed calls, so the switch's cost is each M's time switched over served.

GO needs the gate, the bindings, the served spawns' dispatch, the fused and decode launches and one prefill route: "served" (every
probe's trees one tree per owner, the same in every probe and in both sets of rank processes, FTZ decided, every multimem word
equal), else "switch" (every custom all-reduce word equal, and the switch's dispatch).  Every launch must also run on its expected
path, deterministically, with every rank's words equal and the references equal.  The h7168 verdict needs the same of its own
launches, its probe's trees and FTZ, and its dispatch.  ``coverage`` lists, per setting, hidden size and route, the token counts
whose launches were GO: what `tp_allreduce.CAPTURED` may then cite, with this run's id.

    cd integrations/vllm; PY=/workspace/cp/venv312/bin/python  # vy-glm53-h8's venv; PYTHONPATH as verity_vllm/ops/pod_bootstrap.sh sets it
    export CUDA_HOME=/usr/local/cuda PATH=/workspace/cp/venv312/bin:/usr/local/cuda/bin:$PATH
    $PY tests/properties/glm53_tp8_allreduce_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--served workload|custom-off]
        [--only fused,allreduce,probe,again,switch,h7168,h7168_probe]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import shutil
import socket
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import tp_allreduce as T
from verity_vllm.program.registry import tp_allreduce_check as K
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-tp8-allreduce-capture/v3"
RECORD = "glm53_tp8_allreduce_capture.json"
WORKLOAD = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
WORKLOAD_PATH = Path(__file__).resolve().parents[2] / "workloads" / f"{WORKLOAD}.json"
WORLD, HIDDEN, EPS = 8, 6144, 1e-5
#: DeepSeek-V4-Pro's hidden size, whose TP all-reduces (`attn.wo_b`'s, the MoE's) are `tensor_model_parallel_all_reduce`: it calls
#: no `fused_allreduce_rms_norm`
DSV4_HIDDEN = 7168
#: FlashInfer one-shot to 9 tokens, two-shot to 36; the custom all-reduce's 256 KiB cap at 18 / 19 (inside FlashInfer's); multimem
#: from 37 to 2048
DSV4_AR_M = (1, 9, 10, 18, 19, 36, 37, 256, 2048)
QUICK_DSV4_AR_M = (9, 10, 36, 37)
DSV4_PROBE_M = (37,)
FUSED_MAX = T.fused_max_tokens(HIDDEN, WORLD)
FUSED, UNFUSED = "F", "N"
#: shot boundaries at 10 / 11, cluster boundaries at 16 / 17 and 33 / 34, the budget's last token and the first past it
FUSED_M = (1, 2, 10, 11, 16, 17, 33, 34, 42, 43)
QUICK_FUSED_M = (1, 11, 17, 43)
#: FlashInfer one-shot to 10 tokens, two-shot to 42 (decode is 1), multimem from 43 (prefill is 256) to the 2048 a step batches
AR_M = (1, 10, 11, 21, 42, 43, 256, 682, 2048)
QUICK_AR_M = (1, 11, 43, 256)
#: the probes' all-reduces: the first multimem M and prefill's
PROBE_M_AR = (43, 256)
#: the custom all-reduce's first M past FlashInfer, prefill's, its 8 MiB cap at 682 and NCCL past it
SWITCH_M = (43, 64, 256, 682, 683)
QUICK_SWITCH_M = (43, 256, 683)
#: the fused probe call's M per shot: the most tokens of each
PROBE_M = {1: 10, 2: FUSED_MAX}
#: (name, M, shot): the special tokens at rows 0..2 of a one-shot and a two-shot call
SPECIAL = (("special1", 3, 1), ("special2", 11, 2))
SCALE_LOG2 = (-8.0, 2.0)
SCREEN_CHUNK, SCREEN_MAX = 500, 20_000
#: the workload's `env_required`, and vLLM's defaults for the all-reduce's routes stated
SERVED_ENV = {"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_DEEP_GEMM": "0", "VLLM_ALLREDUCE_USE_SYMM_MEM": "1",
              "VLLM_ALLREDUCE_USE_FLASHINFER": "1"}
SWITCH_ENV = {**SERVED_ENV, "VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}
#: each setting's environment (`tp_allreduce.SETTINGS`); its custom all-reduce is `set_custom_all_reduce`'s, as vLLM's worker sets
#: it from ``disable_custom_all_reduce`` (`vllm/v1/worker/gpu_worker.py:1492`)
SETTING_ENV = {"defaults": SERVED_ENV, "custom-off": SERVED_ENV, "switch": SWITCH_ENV}
#: ``--served``: the workload's setting (`workload_setting`), or "custom-off", what the two GO runs served
SERVED = ("workload", "custom-off")
PHASES = ("fused", "allreduce", "probe", "again", "switch", "h7168", "h7168_probe")
SPAWN_OF = {"fused": "served", "allreduce": "served", "probe": "served", "again": "served_again", "switch": "switch",
            "h7168": "h7168", "h7168_probe": "h7168"}
PROBE_PHASES = ("probe", "again", "h7168_probe")
GLM_SPAWNS = ("served", "served_again", "switch")
#: unset in every rank process: each would change the dispatch this capture states
UNSET = ("NCCL_P2P_DISABLE", "VLLM_CUSTOM_ALLREDUCE_ALGO", "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "VLLM_USE_NCCL_SYMM_MEM",
         "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC")
NEG0, NAN = 0x8000, 0x7FC1
#: rank r's term of a sum whose word depends on the order of its terms: 1 + 2^-8 is a bf16 tie, which six 2^-26 break (to 1 + 2^-7)
#: only when they are added before the 1
ORDER = np.array([0x3F80, 0x3B80, *[0x3280] * 6], np.uint16)
#: the probes: every pair of ranks, B = 2^40 and 2^100, and the six other ranks' 2^0..2^5.  An f32 add absorbs any of their sums
#: (at most 63) into +-B: below B its half ulp is B * 2^-25 = 2^15, so -B + 63 rounds back to -B; an f64 accumulator absorbs them
#: only into 2^100
PAIRS = tuple(itertools.combinations(range(WORLD), 2))
BIG = (0x5380, 0x7180)
SMALL = (0x3F80, 0x4000, 0x4080, 0x4100, 0x4180, 0x4200)
#: a probe's specials after its pairs, each period of an owner's slice: (name, its words on rank r, r + 1 and the others).  "tiny",
#: "halves", "cancel" and "neg_tiny" tell IEEE adds from FTZ ones (`FLUSH`); the others are compared with the model like every word
SPECIALS = (("tiny", 0x0001, 0, 0), ("halves", 0x0040, 0x0040, 0), ("cancel", 0x0081, 0x8080, 0), ("neg_zero", NEG0, NEG0, NEG0),
            ("neg_tiny", 0x8001, NEG0, NEG0), ("nan", NAN, None, None), ("inf_minus_inf", 0x7F80, 0xFF80, None), ("random", None, None, None))
FLUSH = ("tiny", "halves", "cancel", "neg_tiny")
PERIOD = 2 * len(PAIRS) + len(SPECIALS)
WARM, REPS = 3, 20
COMMS = {"fi_ar_comm": "all_reduce", "qr_comm": "quick_all_reduce", "fi_pcie_ipc_ar_comm": "all_reduce",
         "aiter_ar_comm": "custom_all_reduce", "ca_comm": "custom_all_reduce", "symm_mem_comm": "all_reduce", "pynccl_comm": "all_reduce"}
KERNEL = {"ca_comm": "cross_device_reduce_{stage}stage", "symm_mem_comm": "multimem_all_reduce_kernel"}
MODEL = {"fi_ar_comm": "mnnvl", "symm_mem_comm": "multimem", "ca_comm": "custom"}


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


def tree_str(t) -> str:
    return "none" if t is None else str(t).replace(" ", "")


# ------------------------------------------------------------------------------------------------------------------- dispatch
def expected_path(M: int) -> str:
    return FUSED if M <= FUSED_MAX else UNFUSED


def expected_comm(M: int, setting: str, N: int | None = None) -> str:
    """The communicator `CudaCommunicator.all_reduce` hands M tokens of N bf16 (default `HIDDEN`) to at 8 ranks on cc 9.0, one node
    with NVSwitch, at `setting` (`tp_allreduce.SETTINGS`).  vLLM d9105ea80, `vllm/distributed/device_communicators/`:

    * the order is `cuda_communicator.py:305-377`: NCCL's symmetric memory when FlashInfer's all-reduce would not take the call
      (decided at :306-311; :315-322), off unless ``VLLM_USE_NCCL_SYMM_MEM`` (`vllm/envs.py:2017-2019`, `pynccl_allocator.py:48-50`),
      which every rank unsets (`UNSET`); quick all-reduce (:323-331, ROCm); FlashInfer's PCIe IPC all-reduce (:332-334, off unless
      ``VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC``, `vllm/envs.py:1898-1900`); FlashInfer's all-reduce (:335-339); aiter (:340-348,
      ROCm); the custom all-reduce (:349-357); torch's symmetric memory (:358-362); NCCL (:363-377). On 8 H200s that leaves
      FlashInfer's, the custom all-reduce, torch's symmetric memory and NCCL, in that order;
    * FlashInfer takes ``nbytes <= max_workspace_size`` and ``M <= max_workspace_size // (N * 2)`` (`flashinfer_all_reduce.py:441-450`),
      0.5 MiB at 8 ranks on sm_90 (`vllm/compilation/passes/fusion/allreduce_rms_fusion.py:104-109`; `all_reduce_utils.py:87-92`
      tunes no sm_90 size), and its workspace's buffer check (:467-472) admitted every M to 42 at 6144 in the GO runs: 42 tokens
      at 6144, 36 at 7168;
    * the custom all-reduce, on unless ``disable_custom_all_reduce`` (`vllm/config/parallel.py:214`, false;
      `vllm/v1/worker/gpu_worker.py:1492`; `cuda_communicator.py:58-60, 141-149`), takes ``nbytes < max_size``
      (`custom_all_reduce.py:405-420`), 8 MiB (:125) cut to ``CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"][8]`` = 256 KiB when symmetric memory
      is on (:206-220, `all_reduce_utils.py:31-37`): 21 tokens at 6144, 18 at 7168, which FlashInfer has already taken, so it takes
      none at "defaults"; under "switch" it runs to 682 tokens at 6144. Its all-gather and reduce-scatter (:379-389) serve only
      sequence-parallel MoE (`vllm/models/common/ops/sequence_parallel.py:15-39`), which needs expert parallelism and DP > 1
      (`vllm/config/parallel.py:706-721`), and `all_gather` and `reduce_scatter` (:391-452) are NCCL's, so at TP 8 turning it on
      reroutes nothing else;
    * torch's symmetric memory, on unless ``VLLM_ALLREDUCE_USE_SYMM_MEM=0`` (`vllm/envs.py:1888-1890`), takes ``nbytes <= 64 MiB``
      (`symm_mem.py:117-125`, `all_reduce_utils.py:58-63`), multimem at 8 ranks on cc 9.0 (`symm_mem.py:26-31, 143-150`): 5461
      tokens at 6144, 4681 at 7168;
    * NCCL past those."""
    N = HIDDEN if N is None else N
    custom, symm = T.SETTINGS[setting]
    nbytes = M * N * 2
    if M <= T.fused_max_tokens(N, WORLD):
        return "fi_ar_comm"
    if custom and nbytes < (T.CUSTOM_SYMM_MAX_BYTES if symm else T.CUSTOM_MAX_BYTES):
        return "ca_comm"
    if symm and nbytes <= T.SYMM_MEM_MAX_BYTES:
        return "symm_mem_comm"
    return "pynccl_comm"


def shot_of(M: int, N: int | None = None) -> int:
    return T.mnnvl_shot(M, HIDDEN if N is None else N, WORLD)


def stage_of(M: int, N: int | None = None) -> int:
    return T.custom_stage(WORLD, M * (HIDDEN if N is None else N) * 2)


def workload_setting(path: Path | None = None) -> dict[str, Any]:
    """The served engine's setting from the workload's ``engine_args_required``: "custom-off" when it sets
    ``disable_custom_all_reduce`` true, else "defaults", vLLM's own default being false (`vllm/config/parallel.py:214`) for a
    workload that states none; with what it read."""
    path = WORKLOAD_PATH if path is None else path
    if not path.is_file():
        return {"setting": "defaults", "disable_custom_all_reduce": None, "source": f"vLLM's default: no {path.name}"}
    v = json.loads(path.read_text()).get("engine_args_required", {}).get("disable_custom_all_reduce")
    if v is not None and not isinstance(v, bool):
        raise ValueError(f"{path.name}: disable_custom_all_reduce={v!r}, not a bool")
    return {"setting": "custom-off" if v else "defaults", "disable_custom_all_reduce": v,
            "source": path.name if v is not None else f"vLLM's default: {path.name} states none"}


def spawns(served: str) -> dict[str, dict[str, Any]]:
    """Each set of rank processes: its setting (`tp_allreduce.SETTINGS`), its calls' hidden size, its custom all-reduce and
    symmetric memory, and its environment.  "served" and "served_again" run `served`, "switch" turns symmetric memory off, and
    "h7168" runs `DSV4_HIDDEN` at vLLM's defaults in processes of its own, since FlashInfer fixes a process's token budget at its
    first all-reduce (`flashinfer_all_reduce.py:445-447`)."""
    named = {"served": (served, HIDDEN), "served_again": (served, HIDDEN), "switch": ("switch", HIDDEN),
             "h7168": ("defaults", DSV4_HIDDEN)}
    return {k: {"setting": s, "H": h, "custom_all_reduce": T.SETTINGS[s][0], "symm_mem": T.SETTINGS[s][1], "env": SETTING_ENV[s]}
            for k, (s, h) in named.items()}


# ------------------------------------------------------------------------------------------------------------------ operands
def _bf16(rng, shape, scale) -> np.ndarray:
    f = rng.standard_normal(shape).astype(np.float32) * np.asarray(scale, np.float32)
    return DR.f2fp_bf16(f.view(np.uint32))


def tokens(rng, M: int, N: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(WORLD, M, N) partials and (M, N) residuals (N `HIDDEN` by default): each token randn at a scale of 2^U(`SCALE_LOG2`),
    split over the ranks."""
    N = HIDDEN if N is None else N
    s = np.exp2(rng.uniform(*SCALE_LOG2, size=(M, 1))).astype(np.float32)
    return _bf16(rng, (WORLD, M, N), s / np.float32(np.sqrt(WORLD))), _bf16(rng, (M, N), s)


def weight(rng) -> np.ndarray:
    return DR.f2fp_bf16((1 + 0.25 * rng.standard_normal(HIDDEN).astype(np.float32)).view(np.uint32))


def _subnormals(rng, shape) -> np.ndarray:
    return (rng.integers(1, 0x80, size=shape) | (rng.integers(0, 2, size=shape) << 15)).astype(np.uint16)


def special_tokens(rng, parts: np.ndarray, res: np.ndarray) -> None:
    """Rows 0..2 of a fused call: every partial and the residual -0 (Lamport's empty mark, which FlashInfer sanitizes to +0); a
    sum of the smallest normal and minus a hair more, which FADD.FTZ flushes to -0 (one-shot keeps it, two-shot's add onto +0 does
    not), beside a -0 residual, then subnormal partials and `ORDER` and its reverse (1 and 1 + 2^-7 in ascending rank order);
    and a NaN partial."""
    parts[:, 0], res[0] = NEG0, NEG0
    parts[:, 1, :64], res[1, :64] = 0, NEG0
    parts[6, 1, :64], parts[7, 1, :64] = 0x0080, 0x8081
    parts[:, 1, 64:128] = _subnormals(rng, (WORLD, 64))
    parts[:, 1, 128], parts[:, 1, 129], res[1, 128:130] = ORDER, ORDER[::-1], NEG0
    parts[3, 2, 5] = NAN


def order_elements(size: int) -> list[int]:
    """Where an all-reduce of `size` elements holds `ORDER` and its reverse: the first two elements of each 2-stage owner's run
    (`custom_owners`), past row 0's specials in rank 0's run."""
    return sorted({lo + (32 if lo == 0 else 0) + i for lo, _, _ in T.custom_owners(WORLD, size, 2) for i in (0, 1)})


def special_elements(rng, parts: np.ndarray) -> None:
    """Row 0 of an all-reduce: -0 on every rank (IEEE keeps -0), x and -x on ranks 0 and 1 (+0), subnormals (kept), a NaN; and
    `ORDER` and its reverse at `order_elements`, whose words tell the owners' orders, so the two kernels', apart."""
    parts[:, 0, :8] = NEG0
    parts[:, 0, 8:16] = 0
    parts[0, 0, 8:16] = _bf16(rng, (8,), 1.0)
    parts[1, 0, 8:16] = parts[0, 0, 8:16] ^ np.uint16(0x8000)
    parts[:, 0, 16:24] = _subnormals(rng, (WORLD, 8))
    parts[2, 0, 24] = NAN
    flat = parts.reshape(WORLD, -1)
    for i, e in enumerate(order_elements(flat.shape[1])):
        flat[:, e] = ORDER[::-1] if i % 2 else ORDER


def probe_slots(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per element of a probe of `size`: its owner (`multimem_owners`), its slot (its index in the owner's slice mod `PERIOD`: a
    pair at slot f * 28 + p for family f and PAIRS[p], else a special) and its period's first rank (each period moves the specials
    one rank on)."""
    owner, slot, rot = (np.zeros(size, np.int64) for _ in range(3))
    for lo, hi, o in T.multimem_owners(WORLD, size):
        local = np.arange(hi - lo)
        owner[lo:hi], slot[lo:hi], rot[lo:hi] = o, local % PERIOD, (local // PERIOD) % WORLD
    return owner, slot, rot


def probe_parts(rng, size: int) -> np.ndarray:
    """(WORLD, size) partials of a probe: at each pair slot, B (`BIG`) on rank i, -B on rank j and `SMALL` on the others in
    ascending rank order; at each special slot its words (`SPECIALS`) on the period's first rank, the next, and the others (None:
    random normals)."""
    _, slot, rot = probe_slots(size)
    parts = np.zeros((WORLD, size), np.uint16)
    for f, big in enumerate(BIG):
        for p, (i, j) in enumerate(PAIRS):
            at = np.flatnonzero(slot == f * len(PAIRS) + p)
            parts[i, at], parts[j, at] = big, big | 0x8000
            for b, r in enumerate(r for r in range(WORLD) if r not in (i, j)):
                parts[r, at] = SMALL[b]
    for s, (_, first, second, rest) in enumerate(SPECIALS):
        at = np.flatnonzero(slot == 2 * len(PAIRS) + s)
        r0, r1 = rot[at], (rot[at] + 1) % WORLD
        parts[:, at] = _bf16(rng, (WORLD, at.size), 1.0) if rest is None else rest
        if first is not None:
            parts[r0, at] = first
        if second is not None:
            parts[r1, at] = second
    return parts


def custom_sample(size: int) -> list[int]:
    """The elements of an all-reduce the reference checks: row 0's specials, the order elements, the middle and the last."""
    return sorted({0, 5, 8, 16, 24, size // 2 + 3, size - 1, *order_elements(size)})


def multimem_sample(size: int) -> list[int]:
    """`custom_sample` and each owner's first, second, middle and last elements, and its slice's first period of specials."""
    owned = {e for lo, hi, _ in T.multimem_owners(WORLD, size) for e in (lo, lo + 1, (lo + hi) // 2, hi - 1)}
    specials = {lo + 2 * len(PAIRS) + s for lo, hi, _ in T.multimem_owners(WORLD, size) for s in range(len(SPECIALS))
                if lo + PERIOD <= hi}
    return sorted(e for e in owned | specials | set(custom_sample(size)) if 0 <= e < size)


def reference_rows(M: int) -> list[int]:
    """The tokens of a fused call the reference checks: the specials' rows 0 and 1 and the last."""
    return sorted({0, min(1, M - 1), M - 1})


def probe_tokens(rng, M: int, shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """M tokens screened on the models at the call's geometry so that each one's normed words differ between the two
    contractions (random tokens past `SCREEN_MAX` candidates if too few do), and the call's weight.  None differ when HIDDEN is
    a power of two: rcp(N) and the product are then exact."""
    _, cluster = T.mnnvl_grid(M, HIDDEN)
    w = weight(rng)
    got_p, got_r, found, tried = [], [], 0, 0
    while found < M and tried < SCREEN_MAX:
        p, r = tokens(rng, SCREEN_CHUNK)
        a, b = (K.mnnvl_norm_rows(p, r, w, shot, cluster, EPS, f)[0] for f in (False, True))
        idx = np.flatnonzero((a != b).any(axis=1))[:M - found]
        got_p.append(p[:, idx])
        got_r.append(r[idx])
        found, tried = found + idx.size, tried + SCREEN_CHUNK
    if found < M:
        p, r = tokens(rng, M - found)
        got_p.append(p)
        got_r.append(r)
    return np.concatenate(got_p, axis=1), np.concatenate(got_r), w, {"found": found, "screened": tried}


def expect(phase: str, M: int, setting: str, N: int | None = None) -> dict[str, Any]:
    """What a launch of M tokens of N runs at `setting`: a fused call's path (and on FlashInfer's, its shot and cluster), an
    all-reduce's communicator, and its shot or custom stage."""
    N = HIDDEN if N is None else N
    if phase != "fused":
        comm = expected_comm(M, setting, N)
        return {"comm": comm, "shot": shot_of(M, N) if comm == "fi_ar_comm" else None,
                "stage": stage_of(M, N) if comm == "ca_comm" else None}
    if expected_path(M) == UNFUSED:
        return {"path": UNFUSED}
    return {"path": FUSED, "shot": shot_of(M), "cluster": T.mnnvl_grid(M, HIDDEN)[1]}


def cases(quick: bool, only: set[str], spawns_: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every launch of the phases in `only`, each with its spawn's setting and hidden size (`spawns_`, default `spawns` serving
    vLLM's defaults); the seeds of the GLM-5.3 phases come first, so they are the same whatever runs past them."""
    spawns_ = spawns("defaults") if spawns_ is None else spawns_
    out: list[dict[str, Any]] = []
    if "fused" in only:
        out += [{"phase": "fused", "case": f"M{M}", "M": M} for M in (QUICK_FUSED_M if quick else FUSED_M)]
        out += [{"phase": "fused", "case": name, "M": M} for name, M, _ in SPECIAL]
        out += [{"phase": "fused", "case": f"probe{s}", "M": M} for s, M in PROBE_M.items()]
    if "allreduce" in only:
        out += [{"phase": "allreduce", "case": f"M{M}", "M": M} for M in (QUICK_AR_M if quick else AR_M)]
    for phase in ("probe", "again"):
        if phase in only:
            out += [{"phase": phase, "case": f"M{M}", "M": M} for M in PROBE_M_AR]
    if "switch" in only:
        out += [{"phase": "switch", "case": f"M{M}", "M": M} for M in (QUICK_SWITCH_M if quick else SWITCH_M)]
    if "h7168" in only:
        out += [{"phase": "h7168", "case": f"M{M}", "M": M} for M in (QUICK_DSV4_AR_M if quick else DSV4_AR_M)]
    if "h7168_probe" in only:
        out += [{"phase": "h7168_probe", "case": f"M{M}", "M": M} for M in DSV4_PROBE_M]
    for i, c in enumerate(out):
        s = spawns_[SPAWN_OF[c["phase"]]]
        c.update(key=f"{c['phase']}/{c['case']}", seed=9000 + i, spawn=SPAWN_OF[c["phase"]], setting=s["setting"], H=s["H"],
                 **expect(c["phase"], c["M"], s["setting"], s["H"]))
    probe_seed = {c["M"]: c["seed"] for c in out if c["phase"] == "probe"}
    for c in out:
        if c["phase"] == "again":
            c["seed"] = probe_seed.get(c["M"], c["seed"])
    return out


def make_operands(work: Path, case: dict[str, Any]) -> dict[str, Any]:
    """The case's operands, saved under `work` (bf16 words): partials, and for a fused call the residuals and the norm's weight."""
    rng = np.random.default_rng(case["seed"])
    M, N, info = case["M"], case["H"], {}
    if case["phase"] in PROBE_PHASES:
        arrays = {"parts": probe_parts(rng, M * N).reshape(WORLD, M, N)}
    elif case["phase"] != "fused":
        parts, _ = tokens(rng, M, N)
        special_elements(rng, parts)
        arrays = {"parts": parts}
    else:
        if case["case"].startswith("probe"):
            parts, res, w, info = probe_tokens(rng, M, case["shot"])
        else:
            (parts, res), w = tokens(rng, M), weight(rng)
            if case["case"].startswith("special"):
                special_tokens(rng, parts, res)
        arrays = {"parts": parts, "res": res, "w": w}
    d = work / "operands" / case["key"].replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, a in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(a, dtype=np.uint16))
        paths[k] = str(d / f"{k}.npy")
    return {"ops": paths, **({"screen": info} if info else {})}


def _load(paths: dict[str, str]) -> dict[str, np.ndarray]:
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


# ------------------------------------------------------------------------------------------------------------- the probes' trees
def _value(words: np.ndarray) -> np.ndarray:
    return (np.asarray(words, np.uint32) << 16).view(np.float32)


def lca_leaves(t, i: int, j: int) -> frozenset[int]:
    return min((c for c in T.tree_clusters(t) if i in c and j in c), key=len)


def decode_probe(out: np.ndarray, size: int) -> dict[str, Any]:
    """Each owner's tree from a probe's output words (WORLD..., `size` of them): per family and pair, the clusters its elements
    name (the pair and the others whose 2^b the sum lacks); a tree when every pair names one cluster at both families, the
    clusters make a binary tree (`tree_from_clusters`) and each pair's is its two ranks' lowest common add."""
    out = np.asarray(out, np.uint16).reshape(-1)
    owners: dict[str, Any] = {}
    for lo, hi, o in T.multimem_owners(WORLD, size):
        mine, slot = out[lo:hi], np.arange(hi - lo) % PERIOD
        fams, bad = [], {}
        for f in range(len(BIG)):
            named: dict[tuple[int, int], set[frozenset[int]]] = {}
            for p, (i, j) in enumerate(PAIRS):
                words = np.unique(mine[slot == f * len(PAIRS) + p])
                others = [r for r in range(WORLD) if r not in (i, j)]
                named[(i, j)] = set()
                for w in words.tolist():
                    v = float(_value(np.array([w]))[0])
                    if not (v.is_integer() and 0 <= v < 1 << len(SMALL)):
                        bad.setdefault(f"{f}:{i},{j}", []).append(int(w))
                        continue
                    lost = {others[b] for b in range(len(SMALL)) if not int(v) >> b & 1}
                    named[(i, j)].add(frozenset({i, j} | lost))
            fams.append(named)
        one = all(len(s) == 1 for named in fams for s in named.values()) and not bad
        agree = one and fams[0] == fams[1]
        tree = None
        if agree:
            tree = T.tree_from_clusters({next(iter(s)) for s in fams[0].values()}, WORLD)
            if tree is not None and any(lca_leaves(tree, i, j) != next(iter(s)) for (i, j), s in fams[0].items()):
                tree = None
        owners[str(o)] = {"tree": tree, "one_cluster_per_pair": one, "families_agree": agree, "bad_words": bad,
                          "absorbing": [any(len(c) > 2 for s in named.values() for c in s) for named in fams],
                          "clusters": [{f"{i},{j}": sorted(sorted(c) for c in s) for (i, j), s in named.items()} for named in fams]}
    trees = [owners.get(str(o), {}).get("tree") for o in range(WORLD)]
    return {"owners": owners, "trees": trees if all(t is not None for t in trees) else None}


def decide_ftz(out: np.ndarray, parts: np.ndarray) -> dict[str, Any]:
    """FTZ from a probe's flush specials (`FLUSH`), whose sums no order changes: the one of IEEE and FTZ adds whose words every one
    equals, and what each special's words were."""
    out = np.asarray(out, np.uint16).reshape(-1)
    parts = np.asarray(parts, np.uint16).reshape(WORLD, -1)
    _, slot, _ = probe_slots(out.size)
    names = [n for n, *_ in SPECIALS]
    at = np.flatnonzero(np.isin(slot, [2 * len(PAIRS) + names.index(n) for n in FLUSH]))
    chain = T.chain(WORLD)
    fits = [f for f in (False, True) if (DR.f2fp_bf16(K.tree_sum(parts[:, at], chain, f)) == out[at]).all()]
    seen = {n: sorted({int(w) for w in out[slot == 2 * len(PAIRS) + names.index(n)].tolist()}) for n in FLUSH}
    return {"ftz": fits[0] if len(fits) == 1 else None, "fits": fits, "words": seen}


# --------------------------------------------------------------------------------------------------------------- rank process
def _tensor(a: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).cuda()


def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _same(a, b) -> bool:
    import torch

    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def _time_us(fn) -> float:
    """The median of `REPS` calls of `fn`, each timed by CUDA events, after `WARM` untimed ones; every rank makes the same calls."""
    import torch

    for _ in range(WARM):
        fn()
    times = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0)
    return float(np.median(times))


def _kernels(fn) -> list[str] | dict[str, str]:
    """The CUDA kernels one call of `fn` runs, by the profiler's names."""
    import torch

    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return sorted({e.name for e in prof.events() if str(getattr(e, "device_type", "")).endswith("CUDA")})
    except Exception as e:  # noqa: BLE001
        return {"error": _error(e)}


class FusedTap:
    """Records each `fused_allreduce_rms_norm` call's path: `_can_use_flashinfer`'s answer, then either FlashInfer's MNNVL call (its
    workspace backend, the strategy it resolves, weight_bias and eps) or the module's all-reduce (the norm-module path)."""

    def __enter__(self):
        import flashinfer.comm.allreduce as FA
        from flashinfer.comm.trtllm_mnnvl_ar import MNNVLAllreduceFusionStrategy as S
        from vllm.models.common.ops import fused_allreduce_rms_norm as FAR

        self.calls: list[dict[str, Any]] = []
        self.saved = [(FAR, "_can_use_flashinfer"), (FAR, "tensor_model_parallel_all_reduce"), (FA, "trtllm_mnnvl_fused_allreduce_add_rmsnorm")]
        self.saved = [(m, n, getattr(m, n)) for m, n in self.saved]
        can_use, reduce, mnnvl = (f for _, _, f in self.saved)

        def tap_can_use(h, tp):
            ok, most = can_use(h, tp)
            self.calls.append({"M": int(h.shape[0]), "can_use": bool(ok), "max_token_num": int(most), "path": "?"})
            return ok, most

        def tap_reduce(x):
            self.calls[-1]["path"] = UNFUSED
            return reduce(x)

        def tap_mnnvl(*args, **kw):
            ws, x, st = kw["workspace"], kw["input"], kw.get("strategy", S.AUTO)
            if st == S.AUTO:
                st = S.select_strategy(ws.tp_size, x.shape[0], x.shape[1], x.dtype)
            self.calls[-1].update(path=FUSED, backend=ws.backend, shot=1 if st == S.ONESHOT else 2, weight_bias=kw.get("weight_bias"),
                                  epsilon=kw.get("epsilon"))
            return mnnvl(*args, **kw)

        for (m, n, _), f in zip(self.saved, (tap_can_use, tap_reduce, tap_mnnvl)):
            setattr(m, n, f)
        return self

    def __exit__(self, *exc):
        for m, n, f in self.saved:
            setattr(m, n, f)


class CommTap:
    """Records which of the TP group's device communicators returned each all-reduce's output (`COMMS`)."""

    def __enter__(self):
        from vllm.distributed.parallel_state import get_tp_group

        dc = get_tp_group().device_communicator
        self.seen: list[str] = []
        self.objs = []
        for attr, meth in COMMS.items():
            o = getattr(dc, attr, None)
            if o is None or not hasattr(o, meth):
                continue

            def tap(*a, _inner=getattr(o, meth), _name=attr, **k):
                out = _inner(*a, **k)
                if out is not None:
                    self.seen.append(_name)
                return out

            setattr(o, meth, tap)
            self.objs.append((o, meth))
        return self

    def __exit__(self, *exc):
        for o, meth in self.objs:
            o.__dict__.pop(meth, None)


def communicators() -> dict[str, Any]:
    from vllm.distributed import parallel_state as PS

    dc = PS.get_tp_group().device_communicator
    out: dict[str, Any] = {a: getattr(dc, a, None) is not None and not getattr(getattr(dc, a, None), "disabled", False) for a in COMMS}
    out["custom_all_reduce_enabled"] = getattr(PS, "_ENABLE_CUSTOM_ALL_REDUCE", None)
    for attr, fields in (("ca_comm", ("max_size", "fully_connected")), ("symm_mem_comm", ("max_size", "world_size")),
                         ("fi_ar_comm", ("max_workspace_size", "max_num_tokens"))):
        o = getattr(dc, attr, None)
        if o is not None:
            out.update({f"{attr}.{f}": getattr(o, f, None) for f in fields})
    return out


def fused_case(rank: int, case: dict[str, Any]) -> dict[str, Any]:
    import torch
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.common.ops import fused_allreduce_rms_norm as FAR

    ops = _load(case["ops"])
    x, res = _tensor(ops["parts"][rank]), _tensor(ops["res"])
    norm = RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
    norm.weight.data.copy_(_tensor(ops["w"]))
    outs = []
    with FusedTap() as tap:
        for _ in range(2):
            n, r = FAR.fused_allreduce_rms_norm(x.clone(), res.clone(), norm)
            outs.append((n.clone(), r.clone()))
    torch.cuda.synchronize()
    d = Path(case["out"]) / f"rank{rank}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "normed.npy", _u16(outs[0][0]))
    np.save(d / "residual.npy", _u16(outs[0][1]))
    pairs = iter([(x.clone(), res.clone()) for _ in range(WARM + REPS + 1)])
    call = lambda: FAR.fused_allreduce_rms_norm(*next(pairs), norm)               # noqa: E731
    return {"calls": tap.calls, "deterministic": _same(outs[0][0], outs[1][0]) and _same(outs[0][1], outs[1][1]),
            "us": _time_us(call), "kernels": _kernels(call)}


def allreduce_case(rank: int, case: dict[str, Any]) -> dict[str, Any]:
    """`tensor_model_parallel_all_reduce` of the rank's partials, twice (and a probe four times), its words, the communicator, the
    kernels, and (not for a probe) its time."""
    import torch
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce as all_reduce

    x = _tensor(_load(case["ops"])["parts"][rank])
    probe = case["phase"] in PROBE_PHASES
    with CommTap() as tap:
        outs = [all_reduce(x.clone()).clone() for _ in range(4 if probe else 2)]
    torch.cuda.synchronize()
    d = Path(case["out"]) / f"rank{rank}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "out.npy", _u16(outs[0]))
    return {"comms": tap.seen, "deterministic": all(_same(outs[0], o) for o in outs[1:]),
            "us": None if probe else _time_us(lambda: all_reduce(x)), "kernels": _kernels(lambda: all_reduce(x))}


def rank_main(rank: int, cfg: dict[str, Any]) -> None:
    """One rank: vLLM's TP group of `WORLD` with the custom all-reduce as `cfg` has it, every case in order (each a collective
    every rank makes), its record to the work dir."""
    for k in UNSET:
        os.environ.pop(k, None)
    os.environ.update(cfg["env"])
    import torch

    torch.cuda.set_device(rank)
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (destroy_distributed_environment, destroy_model_parallel, ensure_model_parallel_initialized,
                                                 init_distributed_environment, set_custom_all_reduce)

    rec: dict[str, Any] = {"rank": rank, "cases": {}, "errors": {}, "env": {k: os.environ.get(k) for k in (*cfg["env"], *UNSET)},
                           "custom_all_reduce": cfg["custom_ar"]}
    set_custom_all_reduce(cfg["custom_ar"])
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        init_distributed_environment(world_size=WORLD, rank=rank, distributed_init_method=cfg["init"], local_rank=rank, backend="nccl")
        ensure_model_parallel_initialized(WORLD, 1)
        rec["communicators"] = communicators()
        for case in cfg["cases"]:
            try:
                rec["cases"][case["key"]] = (fused_case if case["phase"] == "fused" else allreduce_case)(rank, case)
            except Exception as e:  # noqa: BLE001
                rec["errors"][case["key"]] = _error(e)
        Path(cfg["work"], f"rank{rank}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        destroy_model_parallel()
        destroy_distributed_environment()


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def spawn_ranks(work: Path, env: dict[str, str], custom_ar: bool, cases_: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every rank process over `cases_` under `env` with the custom all-reduce on or off; their records, by rank."""
    import torch.multiprocessing as tmp

    work.mkdir(parents=True, exist_ok=True)
    for c in cases_:
        c["out"] = str(work / c["key"].replace("/", "_"))
    cfg = {"env": env, "custom_ar": custom_ar, "init": f"tcp://127.0.0.1:{_port()}", "work": str(work), "cases": cases_}
    tmp.spawn(rank_main, args=(cfg,), nprocs=WORLD, join=True)
    return [json.loads((work / f"rank{r}.json").read_text()) for r in range(WORLD)]


# ---------------------------------------------------------------------------------------------------------------------- compare
TOTALS = ("words", "mismatch", "normed_words", "fma1_mismatch", "fma0_mismatch", "fma_differ", "rank_mismatch", "reference",
          "twin_mismatch")


def _first(bad: np.ndarray, want: np.ndarray, got: np.ndarray, what: str) -> dict[str, Any] | None:
    if not bad.any():
        return None
    i = tuple(int(v[0]) for v in np.nonzero(bad))
    return {"what": what, "at": list(i), "want": int(want[i]), "kernel": int(got[i])}


def _rank_words(job: dict[str, Any], names: tuple[str, ...]) -> list[dict[str, np.ndarray]]:
    return [{n: np.load(Path(job["out"]) / f"rank{r}" / f"{n}.npy") for n in names} for r in range(WORLD)]


def compare(job: dict[str, Any]) -> dict[str, Any]:
    """One launch's words: every rank's against rank 0's, and rank 0's against the model of the Definition it binds (a fused call's
    at both contractions), with the reference on a sample."""
    try:
        return _compare(job)
    except Exception as e:  # noqa: BLE001
        return {"key": job["key"], "error": _error(e)}


def _compare_fused(job: dict[str, Any], rec: dict[str, Any], ops: dict[str, np.ndarray]) -> dict[str, Any]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    got = _rank_words(job, ("normed", "residual"))
    rec["rank_mismatch"] = sum(int((g[n] != got[0][n]).sum()) for g in got[1:] for n in ("normed", "residual"))
    if not job["model"]:
        return rec
    parts, res, w = (np.asarray(ops[k]) for k in ("parts", "res", "w"))
    shot, cluster, N = job["shot"], job["cluster"], res.shape[1]
    want = {f: K.mnnvl_norm_rows(parts, res, w, shot, cluster, EPS, bool(f)) for f in (1, 0)}
    n0, r0 = got[0]["normed"], got[0]["residual"]
    bad = r0 != want[1][1]
    rec.update(words=r0.size, mismatch=int(bad.sum()), normed_words=n0.size, fma_differ=int((want[1][0] != want[0][0]).sum()))
    rec["first_mismatch"] = _first(bad, want[1][1], r0, "residual")
    for f in (1, 0):
        bad = n0 != want[f][0]
        rec[f"fma{f}_mismatch"] = int(bad.sum())
        rec[f"first_fma{f}_mismatch"] = _first(bad, want[f][0], n0, f"normed fma={f}")
    for row in reference_rows(job["M"]):
        for f in (1, 0):
            d = bind(T.AllReduceRmsNormMnnvl, WORLD=WORLD, N=N, SHOT=shot, CLUSTER=cluster, EPS=EPS, FMA=bool(f))
            y = evaluate(d, parts[:, row].reshape(-1).tolist(), res[row].tolist(), w.tolist())
            rec["reference"] += 1
            rec["twin_mismatch"] += list(y) != want[f][0][row].tolist() + want[f][1][row].tolist()
    return rec


def model_rows(job: dict[str, Any], parts: np.ndarray) -> np.ndarray:
    """The modelled all-reduce of (WORLD, SIZE) partials: FlashInfer's MNNVL at the job's shot, multimem at its trees and FTZ, or
    the custom all-reduce at its stage."""
    if job["model"] == "mnnvl":
        return K.mnnvl_rows(parts, job["shot"])
    if job["model"] == "multimem":
        return K.multimem_rows(parts, job["trees"], job["ftz"])
    return K.custom_rows(parts, job["stage"])


def reference_word(job: dict[str, Any], parts: np.ndarray, e: int) -> int:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    size = parts.shape[1]
    if job["model"] == "mnnvl":
        d = bind(T.AllReduceMnnvlCoordinate, WORLD=WORLD, SHOT=job["shot"])
    elif job["model"] == "multimem":
        owner = next(o for lo, hi, o in T.multimem_owners(WORLD, size) if lo <= e < hi)
        d = bind(T.AllReduceMultimemCoordinate, WORLD=WORLD, TREE=job["trees"][owner], FTZ=job["ftz"])
    else:
        owner = next(o for lo, hi, o in T.custom_owners(WORLD, size, job["stage"]) if lo <= e < hi)
        d = bind(T.AllReduceCustomCoordinate, WORLD=WORLD, START=owner)
    return int(evaluate(d, parts[:, e].tolist())[0])


def _compare(job: dict[str, Any]) -> dict[str, Any]:
    ops = _load(job["ops"])
    rec: dict[str, Any] = {"key": job["key"], **{k: 0 for k in TOTALS}, "first_mismatch": None}
    if job["phase"] == "fused":
        return _compare_fused(job, rec, ops)
    got = _rank_words(job, ("out",))
    rec["rank_mismatch"] = sum(int((g["out"] != got[0]["out"]).sum()) for g in got[1:])
    if not job["model"]:
        return rec
    parts = np.asarray(ops["parts"]).reshape(WORLD, -1)
    out = got[0]["out"].reshape(-1)
    want = model_rows(job, parts).reshape(-1)
    bad = out != want
    rec.update(words=want.size, mismatch=int(bad.sum()), first_mismatch=_first(bad, want, out, "out"))
    sample = multimem_sample(want.size) if job["model"] == "multimem" else custom_sample(want.size)
    for e in sample:
        rec["reference"] += 1
        rec["twin_mismatch"] += reference_word(job, parts, e) != int(want[e])
    return rec


def jobs_of(cases_: list[dict[str, Any]], trees: list | None = None, ftz: bool | None = None,
            pins: dict[str, tuple[list | None, bool | None]] | None = None) -> list[dict[str, Any]]:
    """A comparison job per launch: modelled when the launch runs a Definition's kernel (a fused call on the fused path, an
    all-reduce on FlashInfer's MNNVL, on multimem once the probes have pinned its trees and FTZ, or on the custom all-reduce),
    else compared across ranks only.  `pins` maps a spawn to its own (trees, ftz), as h7168's probe pins its."""
    out = []
    for c in cases_:
        job = {k: c[k] for k in ("key", "phase", "M", "ops", "out")}
        if c["phase"] == "fused":
            job["model"] = c["path"] == FUSED
            if job["model"]:
                job.update(shot=c["shot"], cluster=c["cluster"])
        else:
            tr, fz = (pins or {}).get(c["spawn"], (trees, ftz))
            job["model"] = MODEL.get(c["comm"])
            if job["model"] == "multimem" and (tr is None or fz is None):
                job["model"] = None
            job.update(shot=c["shot"], stage=c["stage"], trees=tr, ftz=fz)
        out.append(job)
    return out


# ------------------------------------------------------------------------------------------------------------------------ bind
def bind_check(cases_: list[dict[str, Any]], trees: list | None, ftz: bool | None) -> dict[str, Any]:
    """Every Definition a launch of `cases_` binds binds at its statics, its operands at the served shapes; the multimem one at
    the probes' trees and FTZ (a chain and IEEE adds when they are not pinned, which states only the shape)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.batch import type_shape

    geoms = sorted({(c["shot"], c["cluster"]) for c in cases_ if c.get("path") == FUSED})
    targets = [(bind(T.AllReduceRmsNormMnnvl, WORLD=WORLD, N=HIDDEN, SHOT=s, CLUSTER=c, EPS=EPS, FMA=f),
                [[WORLD, HIDDEN], [HIDDEN], [HIDDEN], [2 * HIDDEN]]) for s, c in geoms for f in (False, True)]
    trees_ = tuple(trees) if trees else (T.chain(WORLD),) * WORLD
    for c in cases_:
        size, comm = c["M"] * c["H"], c.get("comm")
        if comm == "fi_ar_comm":
            d = bind(T.AllReduceMnnvl, WORLD=WORLD, N=size, SHOT=c["shot"])
        elif comm == "symm_mem_comm":
            d = bind(T.AllReduceMultimem, WORLD=WORLD, SIZE=size, TREES=trees_, FTZ=bool(ftz))
        elif comm == "ca_comm":
            d = bind(T.AllReduceCustom, WORLD=WORLD, SIZE=size, STAGE=c["stage"])
        else:
            continue
        targets.append((d, [[WORLD, size], [size]]))
    out = {}
    for t, want in targets:
        params, ret = t.signature()
        out[t.id] = [list(type_shape(ty)) for _, ty in params] + [list(type_shape(ret))] == want
    return {"bindings": out, "ok": bool(out) and all(out.values())}


def dispatch_check(spawn: dict[str, Any], comms: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Whether every rank's communicators (`communicators`) are its spawn's setting's: the custom all-reduce on or off and, on,
    fully connected at its cap (`CUSTOM_SYMM_MAX_BYTES` beside symmetric memory, else `CUSTOM_MAX_BYTES`, `expected_comm`); torch's
    symmetric memory on or off, at `SYMM_MEM_MAX_BYTES`; FlashInfer's all-reduce at `FI_MAX_BYTES`; no other all-reduce."""
    custom, symm = spawn["custom_all_reduce"], spawn["symm_mem"]
    want: dict[str, Any] = {"custom_all_reduce_enabled": custom, "ca_comm": custom, "symm_mem_comm": symm, "fi_ar_comm": True,
                            "qr_comm": False, "fi_pcie_ipc_ar_comm": False, "aiter_ar_comm": False,
                            "fi_ar_comm.max_workspace_size": T.FI_MAX_BYTES}
    if custom:
        want.update({"ca_comm.max_size": T.CUSTOM_SYMM_MAX_BYTES if symm else T.CUSTOM_MAX_BYTES, "ca_comm.fully_connected": True})
    if symm:
        want["symm_mem_comm.max_size"] = T.SYMM_MEM_MAX_BYTES
    bad = {f"rank{r}": {k: (c or {}).get(k) for k, v in want.items() if (c or {}).get(k) != v} for r, c in enumerate(comms)}
    bad = {r: b for r, b in bad.items() if b}
    return {"ok": len(comms) == WORLD and not bad, "want": want, "differ": bad}


# ------------------------------------------------------------------------------------------------------------------- verdicts
def _call_ok(call: dict[str, Any], case: dict[str, Any]) -> bool:
    if case["path"] == UNFUSED:
        return call["path"] == UNFUSED and not call["can_use"]
    return (call["path"] == FUSED and call["can_use"] and call["max_token_num"] == FUSED_MAX and call.get("backend") == "mnnvl"
            and call.get("shot") == case["shot"] and call.get("weight_bias") == 0.0 and call.get("epsilon") == EPS)


def launch_ok(case: dict[str, Any], recs: list[dict[str, Any]]) -> bool:
    """Every rank ran the case deterministically on its expected path (a fused call) or communicator (and kernel)."""
    key, comm = case["key"], case.get("comm")
    if len(recs) != WORLD or any(key in r["errors"] or key not in r["cases"] for r in recs):
        return False
    for r in recs:
        x = r["cases"][key]
        if not x["deterministic"]:
            return False
        if case["phase"] == "fused":
            if len(x["calls"]) != 2 or not all(_call_ok(c, case) for c in x["calls"]):
                return False
        elif x["comms"] != [comm] * len(x["comms"]) or len(x["comms"]) < 2:
            return False
    k = recs[0]["cases"][key].get("kernels")
    if comm in KERNEL and isinstance(k, list):
        return any(KERNEL[comm].format(stage=case.get("stage")) in n for n in k)
    return True


def fma_of(t: dict[str, Any]) -> int | None:
    """The contraction the words decide: the one model every normed word equals, when the probes told them apart."""
    if t["normed_words"] == 0 or t["fma_differ"] == 0:
        return None
    ok = [f for f in (1, 0) if t[f"fma{f}_mismatch"] == 0]
    return ok[0] if len(ok) == 1 else None


def _shot_totals(doc: dict[str, Any], shot: int) -> dict[str, int]:
    keys = {c["key"] for c in doc["case_list"] if c.get("path") == FUSED and c["shot"] == shot}
    return {k: sum(r[k] for r in doc["comparisons"] if r["key"] in keys) for k in TOTALS}


def case_verdicts(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each launch's own verdict, so a NO-GO names what failed; a fused call's normed words at its shot's `fma`, a modelled
    all-reduce's every word (a multimem one's at the probes' trees, so none when they are not pinned)."""
    by = {r["key"]: r for r in doc["comparisons"]}
    out = {}
    for c in doc["case_list"]:
        recs = doc["ranks"].get(c["spawn"], [])
        t = {k: by.get(c["key"], {}).get(k, 0) for k in TOTALS}
        ok = c["key"] in by and launch_ok(c, recs) and t["rank_mismatch"] == 0
        if c.get("path") == FUSED or c.get("comm") in MODEL:
            ok = ok and t["words"] > 0 and t["mismatch"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0
        if c.get("path") == FUSED:
            fma = doc["fma"].get(str(c["shot"]))
            ok = ok and fma is not None and t[f"fma{fma}_mismatch"] == 0
        r0 = recs[0]["cases"].get(c["key"], {}) if recs else {}
        out[c["key"]] = {"ok": bool(ok), "M": c["M"], "spawn": c["spawn"], "totals": t, "us": r0.get("us"),
                         "path": [x["path"] for x in r0.get("calls", [])] or r0.get("comms")}
    return out


def routes(doc: dict[str, Any]) -> dict[str, Any]:
    """Each route's verdict: the fused decode launches (and both shots' `fma`), FlashInfer's all-reduces, prefill served (the
    probes' trees pinned, the same everywhere, FTZ decided, every multimem launch's words) and prefill switched."""
    cs, verdicts = doc["case_list"], doc["cases"]

    def all_ok(sel) -> bool:
        keys = [c["key"] for c in cs if sel(c)]
        return bool(keys) and all(verdicts[k]["ok"] for k in keys)

    probes = doc.get("probes", {})
    served = (probes.get("trees") is not None and probes.get("agree") is True and probes.get("ftz") is not None
              and all_ok(lambda c: c.get("comm") == "symm_mem_comm" and c["phase"] == "allreduce")
              and all_ok(lambda c: c["phase"] == "probe") and all_ok(lambda c: c["phase"] == "again"))
    return {"fused": all_ok(lambda c: c["phase"] == "fused") and all(doc["fma"].get(str(s)) is not None for s in (1, 2)),
            "decode": all_ok(lambda c: c.get("comm") == "fi_ar_comm" and c["phase"] == "allreduce"),
            "served": served, "switch": all_ok(lambda c: c["phase"] == "switch" and c.get("comm") == "ca_comm")}


def prefill_of(r: dict[str, Any]) -> str | None:
    return "served" if r["served"] else "switch" if r["switch"] else None


def _verdict(doc: dict[str, Any]) -> bool:
    r = doc["routes"]
    prefill = prefill_of(r)
    need = {"served", *({"served_again"} if prefill == "served" else {"switch"} if prefill == "switch" else set())}
    errs = [k for k in doc["errors"] if k.split("/")[0] in need | {"bind"}]
    return (doc["gate"]["ok"] and doc["bind"].get("ok") is True and not errs and r["fused"] and r["decode"] and prefill is not None
            and all(len(doc["ranks"].get(s, [])) == WORLD and doc["dispatch"].get(s, {}).get("ok") is True for s in need))


def h7168_verdict(doc: dict[str, Any]) -> dict[str, Any]:
    """DeepSeek-V4-Pro's all-reduces, apart from GLM-5.3's verdict: every h7168 launch GO, its probe's trees one tree per owner and
    its FTZ decided, its bindings and its dispatch; and whether its trees and FTZ are GLM-5.3's.  None when it did not run."""
    keys = [c["key"] for c in doc["case_list"] if c["spawn"] == "h7168"]
    p, glm = doc.get("h7168_probes", {}), doc.get("probes", {})
    if not keys:
        return {"ok": None}
    errs = [k for k in doc["errors"] if k.split("/")[0] in ("h7168", "h7168_bind")]
    ok = (doc["gate"]["ok"] and not errs and all(doc["cases"][k]["ok"] for k in keys) and p.get("agree") is True
          and doc.get("h7168_bind", {}).get("ok") is True and len(doc["ranks"].get("h7168", [])) == WORLD
          and doc["dispatch"].get("h7168", {}).get("ok") is True)
    return {"ok": bool(ok), "trees": p.get("trees"), "ftz": p.get("ftz"),
            "same_as_glm53": p.get("trees") is not None and (p.get("trees"), p.get("ftz")) == (glm.get("trees"), glm.get("ftz"))}


def coverage(doc: dict[str, Any], ok: bool) -> list[dict[str, Any]]:
    """What a GO record shows, per (setting, hidden size, route): the token counts launched on the route ("fused" for a fused call on
    FlashInfer's path), when every one was GO and its family's verdict (GLM-5.3's `ok`, or h7168's) holds; NCCL's and the unfused
    norm-module launches, which no Program models, left out.  `tp_allreduce.CAPTURED` may cite these rows with the record's run."""
    family_ok = {s: ok for s in GLM_SPAWNS} | {"h7168": doc.get("h7168", {}).get("ok") is True}
    groups: dict[tuple[str, int, str], list[tuple[int, bool]]] = {}
    for c in doc["case_list"]:
        route = "fused" if c.get("path") == FUSED else c.get("comm")
        if route in ("fused", *MODEL) and family_ok.get(c["spawn"]):
            groups.setdefault((c["setting"], c["H"], route), []).append((c["M"], doc["cases"][c["key"]]["ok"]))
    return [{"setting": s, "N": n, "route": r, "M": sorted({m for m, _ in ms})} for (s, n, r), ms in sorted(groups.items())
            if all(g for _, g in ms)]


def probe_analysis(doc_cases: list[dict[str, Any]], phases: tuple[str, ...] = ("probe", "again")) -> dict[str, Any]:
    """Every probe's trees and FTZ (`decode_probe`, `decide_ftz`) from its rank 0 words, and whether they all agree: one tree per
    owner and one FTZ in every probe of `phases` (GLM-5.3's: both sets of rank processes)."""
    per: dict[str, Any] = {}
    words: dict[str, np.ndarray] = {}
    for c in doc_cases:
        if c["phase"] not in phases:
            continue
        try:
            words[c["key"]] = out = np.load(Path(c["out"]) / "rank0" / "out.npy").reshape(-1)
            parts = np.load(c["ops"]["parts"]).reshape(WORLD, -1)
            per[c["key"]] = {**decode_probe(out, out.size), **decide_ftz(out, parts)}
        except Exception as e:  # noqa: BLE001
            per[c["key"]] = {"trees": None, "ftz": None, "error": _error(e)}
    trees = {json.dumps(p["trees"]) for p in per.values()}
    ftzs = {p["ftz"] for p in per.values()}
    again_same = {k: bool(np.array_equal(w, words.get(k.replace("again/", "probe/"), np.empty(0)))) for k, w in words.items()
                  if k.startswith("again/")}
    agree = (bool(per) and len(trees) == 1 and len(ftzs) == 1 and None not in ftzs and "null" not in trees
             and all(again_same.values()))
    first = next(iter(per.values()), {})
    return {"per_probe": per, "again_words_equal": again_same, "agree": agree, "trees": first.get("trees") if agree else None,
            "ftz": first.get("ftz") if agree else None}


def switch_cost(doc: dict[str, Any]) -> dict[str, float]:
    """Each M's time switched (the custom all-reduce) over served (multimem)."""
    served = {c["M"]: c.get("us") for c in doc["cases"].values() if c["spawn"] == "served" and c["path"] and "symm_mem_comm" in c["path"]}
    sw = {c["M"]: c.get("us") for c in doc["cases"].values() if c["spawn"] == "switch"}
    return {f"M{M}": round(sw[M] / served[M], 3) for M in sorted(set(served) & set(sw)) if sw[M] and served[M]}


# ------------------------------------------------------------------------------------------------------------------------- run
def gate(min_driver: int, want_cc: str, want_sms: int) -> dict[str, Any]:
    import torch

    rec = C.gate(min_driver, want_cc, want_sms)
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    rec.update(gpus=n, names=names, topology=C._run(["nvidia-smi", "topo", "-m"]).stdout[-4000:],
               jit_tools={t: shutil.which(t) for t in ("ninja", "nvcc")})
    rec["checks"].update(world=n >= WORLD, one_kind=len(set(names)) == 1, jit_tools=all(rec["jit_tools"].values()))
    rec["ok"] = all(rec["checks"].values())
    return rec


def _save_words(outdir: Path, case_list: list[dict[str, Any]]) -> list[str]:
    """Rank 0's words of every probe and of the served multimem all-reduce at each probe's M, beside the record."""
    saved = []
    probed = {"allreduce": PROBE_M_AR, "h7168": DSV4_PROBE_M}
    for c in case_list:
        if c["phase"] in PROBE_PHASES or (c.get("comm") == "symm_mem_comm" and c["M"] in probed.get(c["phase"], ())):
            src = Path(c["out"]) / "rank0" / "out.npy"
            if src.exists():
                dst = outdir / "words" / f"{c['key'].replace('/', '_')}.npy"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                saved.append(str(dst.relative_to(outdir)))
    return saved


def run(a: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=a.scratch)
    work = Path(tmp.name)
    only = set(a.only.split(","))
    served = (workload_setting() if a.served == "workload" else
              {"setting": "custom-off", "disable_custom_all_reduce": True, "source": "--served custom-off"})
    sp = spawns(served["setting"])
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "quick": a.quick, "workload": WORKLOAD, "served": served,
                           "spawns": sp, "environment": G._environment(), "gate": gate(a.min_driver, a.want_cc, a.want_sms),
                           "errors": {}}
    print(f"[glm53-tp8-ar] gate {json.dumps({k: v for k, v in doc['gate'].items() if k != 'topology'})}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-TP8-ALLREDUCE GATE-FAIL", flush=True)
        raise SystemExit(3)
    for mod in ("flashinfer",):
        try:
            doc["environment"][mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            doc["environment"][mod] = f"unavailable: {e}"[:160]
    case_list = cases(a.quick, only, sp)
    for c in case_list:
        c.update(make_operands(work, c))
        if "screen" in c:
            print(f"[glm53-tp8-ar] {c['key']} screened {c['screen']}", flush=True)
    doc["case_list"] = [{k: v for k, v in c.items() if k not in ("ops", "out")} for c in case_list]
    print(f"[glm53-tp8-ar] served {json.dumps(served)}; {len(case_list)} launches on {WORLD} ranks in "
          f"{sorted({c['spawn'] for c in case_list})}", flush=True)
    doc["ranks"], doc["communicators"], doc["dispatch"], doc["comparisons"] = {}, {}, {}, []
    for spawn, s in sp.items():
        cs = [c for c in case_list if c["spawn"] == spawn]
        if not cs:
            continue
        try:
            doc["ranks"][spawn] = spawn_ranks(work / spawn, s["env"], s["custom_all_reduce"], cs)
        except Exception as e:  # noqa: BLE001
            doc["errors"][f"{spawn}/ranks"] = _error(e)
            continue
        for r in doc["ranks"][spawn]:
            doc["errors"].update({f"{spawn}/rank{r['rank']}/{k}": v for k, v in r["errors"].items()})
        doc["communicators"][spawn] = doc["ranks"][spawn][0].get("communicators")
        doc["dispatch"][spawn] = dispatch_check(s, [r.get("communicators") for r in doc["ranks"][spawn]])
        print(f"[glm53-tp8-ar] {spawn} ({s['setting']}, H {s['H']}): dispatch {'ok' if doc['dispatch'][spawn]['ok'] else 'DIFFERS'} "
              f"{json.dumps(doc['dispatch'][spawn]['differ'], default=str)} communicators "
              f"{json.dumps(doc['communicators'][spawn], default=str)}", flush=True)
    ran = [c for c in case_list if c["spawn"] in doc["ranks"]]
    doc["probes"] = probe_analysis(ran)
    doc["h7168_probes"] = probe_analysis(ran, ("h7168_probe",))
    trees, ftz = doc["probes"]["trees"], doc["probes"]["ftz"]
    pins = {"h7168": (doc["h7168_probes"]["trees"], doc["h7168_probes"]["ftz"])}
    for key, p in {**doc["probes"]["per_probe"], **doc["h7168_probes"]["per_probe"]}.items():
        print(f"[glm53-tp8-ar] {key} trees={[tree_str(t) for t in (p['trees'] or [None])]} ftz={p['ftz']} "
              f"flush words={json.dumps(p.get('words'))} {p.get('error', '')}", flush=True)
    for name, cs, (tr, fz) in (("bind", [c for c in case_list if c["spawn"] in GLM_SPAWNS], (trees, ftz)),
                               ("h7168_bind", [c for c in case_list if c["spawn"] == "h7168"], pins["h7168"])):
        try:
            doc[name] = bind_check(cs, tr, fz) if cs else {"ok": None}
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = _error(e)
            doc[name] = {"ok": False}
    spawn_of = {c["key"]: c["spawn"] for c in ran}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs_of(ran, trees, ftz, pins)):
            if "error" in res:
                doc["errors"][f"{spawn_of[res['key']]}/compare/{res['key']}"] = res["error"]
            else:
                doc["comparisons"].append(res)
    doc["words"] = _save_words(outdir, ran)
    doc["fma"] = {str(s): fma_of(_shot_totals(doc, s)) for s in (1, 2)}
    doc["shot_totals"] = {str(s): _shot_totals(doc, s) for s in (1, 2)}
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = case_verdicts(doc)
    doc["routes"] = routes(doc)
    doc["prefill"] = prefill_of(doc["routes"])
    doc["switch_cost"] = switch_cost(doc)
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[glm53-tp8-ar] case {key:18s} {'GO' if c['ok'] else 'NO-GO':5s} M={c['M']:4d} path={c['path']} words={t['words']}!={t['mismatch']} "
              f"normed={t['normed_words']} fma1!={t['fma1_mismatch']} fma0!={t['fma0_mismatch']} differ={t['fma_differ']} "
              f"ranks!={t['rank_mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} us={c.get('us')}", flush=True)
    print(f"[glm53-tp8-ar] routes {json.dumps(doc['routes'])} switch cost (us switched / served) {json.dumps(doc['switch_cost'])}", flush=True)
    doc["h7168"] = h7168_verdict(doc)
    ok = _verdict(doc)
    doc["coverage"] = coverage(doc, ok)
    for row in doc["coverage"]:
        print(f"[glm53-tp8-ar] covered {row['setting']} N={row['N']} {row['route']} M={row['M']}", flush=True)
    doc = seal_record(doc, ok)
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    h = doc["h7168"]
    h_word = {True: "GO", False: "NO-GO", None: "skipped"}[h["ok"]]
    shown = lambda ts: "|".join(sorted({tree_str(t) for t in ts})) if ts else "none"  # noqa: E731
    print(f"GLM53-TP8-ALLREDUCE {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} served={served['setting']} "
          f"prefill={doc['prefill'] or 'none'} trees={shown(trees)} ftz={ftz} fma={json.dumps(doc['fma'])} h7168={h_word} "
          f"h7168_trees={'same' if h.get('same_as_glm53') else shown(h.get('trees'))} h7168_ftz={h.get('ftz')} "
          f"totals={json.dumps(doc['totals'])} no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} "
          f"errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's TP8 all-reduces on this 8-GPU Hopper node against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and the probes' words")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, every rank's words); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true",
                    help=f"fused M in {QUICK_FUSED_M}, all-reduce M in {QUICK_AR_M}, switch M in {QUICK_SWITCH_M}, h7168 M in "
                         f"{QUICK_DSV4_AR_M}, with every probe")
    ap.add_argument("--served", choices=SERVED, default="workload",
                    help="the served spawns' setting: the workload's disable_custom_all_reduce (default), or custom-off, the two GO "
                         "runs' (tp_allreduce.CAPTURE_RUNS)")
    ap.add_argument("--only", default=",".join(PHASES), help=f"comma list of phases: {', '.join(PHASES)}")
    a = ap.parse_args(argv)
    return 0 if run(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
