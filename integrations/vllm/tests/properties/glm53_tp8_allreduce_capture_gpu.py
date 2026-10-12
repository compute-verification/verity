"""GLM-5.3's TP8 all-reduces on 8x H200 against their Programs, a step of the 8-GPU run that needs no checkpoint (run it before an
engine holds the GPUs): FlashInfer's fused all-reduce + residual add + RMSNorm on its MNNVL backend, which GLM-5.3's eager decoder
layer calls through `fused_allreduce_rms_norm` (vllm/models/common/ops/fused_allreduce_rms_norm.py) up to `FUSED_MAX` tokens
(`tp_allreduce.AllReduceRmsNormMnnvl_v1`), and vLLM's custom all-reduce, which `tensor_model_parallel_all_reduce` runs for the
embedding and past `FUSED_MAX` tokens, to 682 tokens under the switch `VLLM_ALLREDUCE_USE_SYMM_MEM=0` (`tp_allreduce.AllReduceCustom_v1`).

This process makes every operand, spawns one GPU process per rank, which runs vLLM's own calls on its rank's operands and saves its
words, and compares those words with the numpy models (`tp_allreduce_check`).  It writes `<out>/glm53_tp8_allreduce_capture.json`,
the property record `glm53_tp8_allreduce_capture` (`properties.record`), prints one line per case and then `GLM53-TP8-ALLREDUCE GO`
or `NO-GO`.  The device gate runs first and nothing else runs when it fails (exit 3).

  gate       driver, cc 9.0, 132 SMs, `WORLD` GPUs of one name (an 8x H200 node); the NVLink topology is recorded.
  fused      at M in `FUSED_M` (both shots of `mnnvl_shot` and every cluster geometry of `mnnvl_grid`), the special tokens (every
             partial -0, a sum that flushes to -0, subnormal partials, a NaN partial) at one- and two-shot, and one probe call per
             shot: every call admitted by `_can_use_flashinfer` on the "mnnvl" workspace with FlashInfer's strategy `mnnvl_shot(M)`,
             weight_bias 0 and eps `EPS`, and past `FUSED_MAX` on the all-reduce then the norm module; every rank's normed and
             residual words against `mnnvl_norm_rows` at both contractions of `sum * rcp(N) + EPS`.  The one-shot and two-shot
             kernels are compiled apart, so `fma` is decided per shot: the contraction whose model every word equals.  The probes
             are tokens this process screened to tell the two apart (their normed words differ on about 1 random token in 100).
  allreduce  `tensor_model_parallel_all_reduce` at M in `AR_M`: the communicator that ran it against `expected_comm`, the custom
             all-reduce's kernel (`cross_device_reduce_<stage>stage`) and every word against `custom_rows` at `custom_stage`;
             row 0 holds -0s, a cancellation, subnormals and a NaN, and each 2-stage owner's run starts with sums whose words
             depend on the order of the ranks (`ORDER`), as row 1 of the fused specials does.  Past 682 tokens NCCL runs it; its
             words are compared across ranks only.
  reference  per launch, the plain-integer reference (`verity.ml.circuits.evaluation.evaluate`) of the word Definition the launch
             binds, on sampled tokens (fused, both contractions) or elements (custom), which the model must equal.
  bind       every Definition binds at GLM-5.3's statics.
  timing     rank 0's median of `REPS` CUDA-event-timed calls.

GO needs every call on its expected path, 0 mismatches on every word, every rank's words equal, both shots' `fma` decided, every
launch deterministic and the references equal.  By default the run ends with the all-reduce phase under vLLM's default
(`VLLM_ALLREDUCE_USE_SYMM_MEM=1`) in a second set of rank processes (`served_default`): which M it hands to torch's symmetric-memory
all-reduce, whether those words still equal a custom all-reduce model, and each M's time either way (the switch's cost).  It does
not enter the verdict.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as verity_vllm/ops/pod_bootstrap.sh sets it
    $PY tests/properties/glm53_tp8_allreduce_capture_gpu.py --out DIR [--scratch DIR] [--quick] [--no-default] [--only fused,allreduce]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
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

SCHEMA = "verity-vllm/glm53-tp8-allreduce-capture/v1"
RECORD = "glm53_tp8_allreduce_capture.json"
WORLD, HIDDEN, EPS = 8, 6144, 1e-5
FUSED_MAX = T.fused_max_tokens(HIDDEN, WORLD)
FUSED, UNFUSED = "F", "N"
#: shot boundaries at 10 / 11, cluster boundaries at 16 / 17 and 33 / 34, the budget's last token and the first past it
FUSED_M = (1, 2, 10, 11, 16, 17, 33, 34, 42, 43)
QUICK_FUSED_M = (1, 11, 17, 43)
#: custom 1-stage to 21 tokens (and its 256 KiB cap beside symm-mem), 2-stage from 22, its 8 MiB cap at 682, NCCL past it
AR_M = (1, 21, 22, 64, 256, 682, 683)
QUICK_AR_M = (1, 22, 683)
#: the probe call's M per shot: the most tokens of each
PROBE_M = {1: 10, 2: FUSED_MAX}
#: (name, M, shot): the special tokens at rows 0..2 of a one-shot and a two-shot call
SPECIAL = (("special1", 3, 1), ("special2", 11, 2))
SCALE_LOG2 = (-8.0, 2.0)
SCREEN_CHUNK, SCREEN_MAX = 500, 20_000
#: vLLM d9105ea80 all_reduce_utils: the custom all-reduce's cap at 8 ranks on cc 9.0 beside torch's symmetric memory, and that one's
CUSTOM_MAX_SYMM, SYMM_MAX = 256 * 1024, 64 * 1024 * 1024
SWITCH_ENV = {"VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}
DEFAULT_ENV = {"VLLM_ALLREDUCE_USE_SYMM_MEM": "1"}
#: unset in every rank process: each would change the dispatch this capture states
UNSET = ("VLLM_CUSTOM_ALLREDUCE_ALGO", "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "VLLM_USE_NCCL_SYMM_MEM", "VLLM_ALLREDUCE_USE_FLASHINFER")
NEG0, NAN = 0x8000, 0x7FC1
#: rank r's term of a sum whose word depends on the order of its terms: 1 + 2^-8 is a bf16 tie, which six 2^-26 break (to 1 + 2^-7)
#: only when they are added before the 1
ORDER = np.array([0x3F80, 0x3B80, *[0x3280] * 6], np.uint16)
WARM, REPS = 3, 20
PHASES = ("fused", "allreduce")
COMMS = {"fi_ar_comm": "all_reduce", "qr_comm": "quick_all_reduce", "fi_pcie_ipc_ar_comm": "all_reduce",
         "aiter_ar_comm": "custom_all_reduce", "ca_comm": "custom_all_reduce", "symm_mem_comm": "all_reduce", "pynccl_comm": "all_reduce"}


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


# ------------------------------------------------------------------------------------------------------------------- dispatch
def expected_path(M: int) -> str:
    return FUSED if M <= FUSED_MAX else UNFUSED


def expected_comm(M: int, symm_mem: bool) -> str:
    """The communicator `CudaCommunicator.all_reduce` hands M bf16 tokens to at 8 ranks on cc 9.0: the custom all-reduce below its
    cap (256 KiB beside torch's symmetric memory, else 8 MiB), then symmetric memory (to 64 MiB) when it is on, then NCCL."""
    nbytes = M * HIDDEN * 2
    if nbytes < (CUSTOM_MAX_SYMM if symm_mem else T.CUSTOM_MAX_BYTES):
        return "ca_comm"
    if symm_mem and nbytes <= SYMM_MAX:
        return "symm_mem_comm"
    return "pynccl_comm"


def shot_of(M: int) -> int:
    return T.mnnvl_shot(M, HIDDEN, WORLD)


def stage_of(M: int) -> int:
    return T.custom_stage(WORLD, M * HIDDEN * 2)


# ------------------------------------------------------------------------------------------------------------------ operands
def _bf16(rng, shape, scale) -> np.ndarray:
    f = rng.standard_normal(shape).astype(np.float32) * np.asarray(scale, np.float32)
    return DR.f2fp_bf16(f.view(np.uint32))


def tokens(rng, M: int) -> tuple[np.ndarray, np.ndarray]:
    """(WORLD, M, N) partials and (M, N) residuals: each token randn at a scale of 2^U(`SCALE_LOG2`), split over the ranks."""
    s = np.exp2(rng.uniform(*SCALE_LOG2, size=(M, 1))).astype(np.float32)
    return _bf16(rng, (WORLD, M, HIDDEN), s / np.float32(np.sqrt(WORLD))), _bf16(rng, (M, HIDDEN), s)


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


def custom_sample(size: int) -> list[int]:
    """The elements of a custom all-reduce the reference checks: row 0's specials, the order elements, the middle and the last."""
    return sorted({0, 5, 8, 16, 24, size // 2 + 3, size - 1, *order_elements(size)})


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


def expect(phase: str, M: int) -> dict[str, Any]:
    """What a launch of M tokens runs under the switch: a fused call's path (and on FlashInfer's, its shot and cluster), an
    all-reduce's communicator and custom stage."""
    if phase == "allreduce":
        return {"comm": expected_comm(M, False), "stage": stage_of(M)}
    if expected_path(M) == UNFUSED:
        return {"path": UNFUSED}
    return {"path": FUSED, "shot": shot_of(M), "cluster": T.mnnvl_grid(M, HIDDEN)[1]}


def cases(quick: bool, only: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if "fused" in only:
        out += [{"phase": "fused", "case": f"M{M}", "M": M} for M in (QUICK_FUSED_M if quick else FUSED_M)]
        out += [{"phase": "fused", "case": name, "M": M} for name, M, _ in SPECIAL]
        out += [{"phase": "fused", "case": f"probe{s}", "M": M} for s, M in PROBE_M.items()]
    if "allreduce" in only:
        out += [{"phase": "allreduce", "case": f"M{M}", "M": M} for M in (QUICK_AR_M if quick else AR_M)]
    for i, c in enumerate(out):
        c.update(key=f"{c['phase']}/{c['case']}", seed=9000 + i, **expect(c["phase"], c["M"]))
    return out


def make_operands(work: Path, case: dict[str, Any]) -> dict[str, Any]:
    """The case's operands, saved under `work` (bf16 words): partials, and for a fused call the residuals and the norm's weight."""
    rng = np.random.default_rng(case["seed"])
    M, info = case["M"], {}
    if case["phase"] == "allreduce":
        parts, _ = tokens(rng, M)
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
    from vllm.distributed.parallel_state import get_tp_group

    dc = get_tp_group().device_communicator
    out = {a: getattr(dc, a, None) is not None and not getattr(getattr(dc, a, None), "disabled", False) for a in COMMS}
    ca = getattr(dc, "ca_comm", None)
    if ca is not None:
        out.update(ca_max_size=getattr(ca, "max_size", None), ca_fully_connected=getattr(ca, "fully_connected", None))
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
    import torch
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce as all_reduce

    x = _tensor(_load(case["ops"])["parts"][rank])
    with CommTap() as tap:
        outs = [all_reduce(x.clone()).clone() for _ in range(2)]
    torch.cuda.synchronize()
    d = Path(case["out"]) / f"rank{rank}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "out.npy", _u16(outs[0]))
    return {"comms": tap.seen, "deterministic": _same(outs[0], outs[1]), "us": _time_us(lambda: all_reduce(x)),
            "kernels": _kernels(lambda: all_reduce(x))}


def rank_main(rank: int, cfg: dict[str, Any]) -> None:
    """One rank: vLLM's TP group of `WORLD`, every case in order (each a collective every rank makes), its record to the work dir."""
    os.environ.update(cfg["env"])
    for k in UNSET:
        os.environ.pop(k, None)
    import torch

    torch.cuda.set_device(rank)
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (destroy_distributed_environment, destroy_model_parallel, ensure_model_parallel_initialized,
                                                 init_distributed_environment)

    rec: dict[str, Any] = {"rank": rank, "cases": {}, "errors": {}, "env": {k: os.environ.get(k) for k in (*cfg["env"], *UNSET)}}
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


def spawn_ranks(work: Path, env: dict[str, str], cases_: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every rank process over `cases_` under `env`; their records, by rank."""
    import torch.multiprocessing as tmp

    work.mkdir(parents=True, exist_ok=True)
    for c in cases_:
        c["out"] = str(work / c["key"].replace("/", "_"))
    cfg = {"env": env, "init": f"tcp://127.0.0.1:{_port()}", "work": str(work), "cases": cases_}
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


def _compare(job: dict[str, Any]) -> dict[str, Any]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate

    M, ops = job["M"], _load(job["ops"])
    rec: dict[str, Any] = {"key": job["key"], **{k: 0 for k in TOTALS}, "first_mismatch": None}
    if job["phase"] == "fused":
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
        for row in reference_rows(M):
            for f in (1, 0):
                d = bind(T.AllReduceRmsNormMnnvl, WORLD=WORLD, N=N, SHOT=shot, CLUSTER=cluster, EPS=EPS, FMA=bool(f))
                y = evaluate(d, parts[:, row].reshape(-1).tolist(), res[row].tolist(), w.tolist())
                rec["reference"] += 1
                rec["twin_mismatch"] += list(y) != want[f][0][row].tolist() + want[f][1][row].tolist()
        return rec
    got = _rank_words(job, ("out",))
    rec["rank_mismatch"] = sum(int((g["out"] != got[0]["out"]).sum()) for g in got[1:])
    if not job["model"]:
        return rec
    parts = np.asarray(ops["parts"]).reshape(WORLD, -1)
    stages = (job["stage"],) if job["stage"] else (1, 2)
    for stage in stages:
        want = K.custom_rows(parts, stage)
        bad = got[0]["out"].reshape(-1) != want
        if job["stage"]:
            rec.update(words=want.size, mismatch=int(bad.sum()), first_mismatch=_first(bad, want, got[0]["out"].reshape(-1), "out"))
            size = want.size
            for e in custom_sample(size):
                owner = next(o for lo, hi, o in T.custom_owners(WORLD, size, stage) if lo <= e < hi)
                y = evaluate(bind(T.AllReduceCustomCoordinate, WORLD=WORLD, START=owner), parts[:, e].tolist())
                rec["reference"] += 1
                rec["twin_mismatch"] += list(y) != [int(want[e])]
        else:
            rec[f"stage{stage}_mismatch"] = int(bad.sum())
    return rec


def jobs_of(cases_: list[dict[str, Any]], symm_mem: bool) -> list[dict[str, Any]]:
    """A comparison job per launch: modelled when the launch runs a Definition's kernel (a fused call on the fused path, a custom
    all-reduce), else compared across ranks only; under symmetric memory a launch it takes is compared with both stages."""
    out = []
    for c in cases_:
        job = {k: c[k] for k in ("key", "phase", "M", "ops", "out")}
        if c["phase"] == "fused":
            job["model"] = c["path"] == FUSED
            if job["model"]:
                job.update(shot=c["shot"], cluster=c["cluster"])
        else:
            comm = expected_comm(c["M"], symm_mem)
            job["model"] = comm in ("ca_comm", "symm_mem_comm")
            job["stage"] = c["stage"] if comm == "ca_comm" else None
        out.append(job)
    return out


# ------------------------------------------------------------------------------------------------------------------------ bind
def bind_check(cases_: list[dict[str, Any]]) -> dict[str, Any]:
    """Every Definition a launch binds binds at GLM-5.3's statics, its operands at the served shapes."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.batch import type_shape

    geoms = sorted({(c["shot"], c["cluster"]) for c in cases_ if c.get("path") == FUSED})
    targets = [(bind(T.AllReduceRmsNormMnnvl, WORLD=WORLD, N=HIDDEN, SHOT=s, CLUSTER=c, EPS=EPS, FMA=f),
                [[WORLD, HIDDEN], [HIDDEN], [HIDDEN], [2 * HIDDEN]]) for s, c in geoms for f in (False, True)]
    targets += [(bind(T.AllReduceCustom, WORLD=WORLD, SIZE=c["M"] * HIDDEN, STAGE=c["stage"]), [[WORLD, c["M"] * HIDDEN], [c["M"] * HIDDEN]])
                for c in cases_ if c.get("comm") == "ca_comm"]
    out = {}
    for t, want in targets:
        params, ret = t.signature()
        out[t.id] = [list(type_shape(ty)) for _, ty in params] + [list(type_shape(ret))] == want
    return {"bindings": out, "ok": bool(out) and all(out.values())}


# ------------------------------------------------------------------------------------------------------------------- verdicts
def _call_ok(call: dict[str, Any], case: dict[str, Any]) -> bool:
    if case["path"] == UNFUSED:
        return call["path"] == UNFUSED and not call["can_use"]
    return (call["path"] == FUSED and call["can_use"] and call["max_token_num"] == FUSED_MAX and call.get("backend") == "mnnvl"
            and call.get("shot") == case["shot"] and call.get("weight_bias") == 0.0 and call.get("epsilon") == EPS)


def launch_ok(case: dict[str, Any], recs: list[dict[str, Any]], symm_mem: bool) -> bool:
    """Every rank ran the case deterministically on its expected path (a fused call) or communicator (and custom kernel)."""
    key, comm = case["key"], expected_comm(case["M"], symm_mem) if case["phase"] == "allreduce" else None
    if len(recs) != WORLD or any(key in r["errors"] or key not in r["cases"] for r in recs):
        return False
    for r in recs:
        x = r["cases"][key]
        if not x["deterministic"]:
            return False
        if case["phase"] == "fused":
            if len(x["calls"]) != 2 or not all(_call_ok(c, case) for c in x["calls"]):
                return False
        elif x["comms"] != [comm] * 2:
            return False
    k = recs[0]["cases"][key].get("kernels")
    if comm == "ca_comm" and isinstance(k, list):
        return any(f"cross_device_reduce_{case['stage']}stage" in n for n in k)
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
    """Each launch's own verdict, so a NO-GO names what failed; a fused call's normed words at its shot's `fma`."""
    by = {r["key"]: r for r in doc["comparisons"]}
    out = {}
    for c in doc["case_list"]:
        t = {k: by.get(c["key"], {}).get(k, 0) for k in TOTALS}
        ok = c["key"] in by and launch_ok(c, doc["ranks"], False) and t["rank_mismatch"] == 0
        if c.get("path") == FUSED or c.get("comm") == "ca_comm":
            ok = ok and t["words"] > 0 and t["mismatch"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0
        if c.get("path") == FUSED:
            fma = doc["fma"].get(str(c["shot"]))
            ok = ok and fma is not None and t[f"fma{fma}_mismatch"] == 0
        r0 = doc["ranks"][0]["cases"].get(c["key"], {})
        out[c["key"]] = {"ok": bool(ok), "M": c["M"], "totals": t, "us": r0.get("us"),
                         "path": [x["path"] for x in r0.get("calls", [])] or r0.get("comms")}
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    phases = {c["phase"] for c in doc["case_list"]}
    fused_ok = "fused" not in phases or all(doc["fma"].get(str(s)) is not None for s in (1, 2))
    return (doc["gate"]["ok"] and not doc["errors"] and doc["bind"].get("ok") is True and phases == set(PHASES) and fused_ok
            and len(doc["ranks"]) == WORLD and all(c["ok"] for c in doc["cases"].values()))


# --------------------------------------------------------------------------------------------------------------------- default
def default_run(work: Path, ar_cases: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    """The all-reduce phase under vLLM's default (torch's symmetric memory on), in rank processes of their own: each M's
    communicator, whether its words equal the custom all-reduce's at its stage (a symmetric-memory launch: at either stage), and
    each M's time."""
    cs = [dict(c) for c in ar_cases]
    recs = spawn_ranks(work, DEFAULT_ENV, cs)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        comps = {r["key"]: r for r in ex.map(compare, jobs_of(cs, True))}
    out = {}
    for c in cs:
        x, r = recs[0]["cases"].get(c["key"], {}), comps.get(c["key"], {})
        out[c["key"]] = {"comms": x.get("comms"), "expected": expected_comm(c["M"], True), "us": x.get("us"),
                         "kernels": x.get("kernels"), "rank_mismatch": r.get("rank_mismatch"), "error": r.get("error"),
                         **{k: r[k] for k in ("mismatch", "stage1_mismatch", "stage2_mismatch") if k in r}}
    return {"cases": out, "errors": {k: v for rec in recs for k, v in rec["errors"].items()}}


def _cost(doc: dict[str, Any]) -> dict[str, Any]:
    base = {k: c.get("us") for k, c in doc.get("served_default", {}).get("cases", {}).items()}
    return {k: round(c["us"] / base[k], 3) for k, c in doc["cases"].items() if c.get("us") and base.get(k)}


# ------------------------------------------------------------------------------------------------------------------------- run
def gate(min_driver: int, want_cc: str, want_sms: int) -> dict[str, Any]:
    import torch

    rec = C.gate(min_driver, want_cc, want_sms)
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    rec.update(gpus=n, names=names, topology=C._run(["nvidia-smi", "topo", "-m"]).stdout[-4000:])
    rec["checks"].update(world=n >= WORLD, one_kind=len(set(names)) == 1)
    rec["ok"] = all(rec["checks"].values())
    return rec


def run(a: argparse.Namespace) -> dict[str, Any]:
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=a.scratch)
    work = Path(tmp.name)
    only = set(a.only.split(","))
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "quick": a.quick, "switch_env": SWITCH_ENV, "environment": G._environment(),
                           "gate": gate(a.min_driver, a.want_cc, a.want_sms), "errors": {}}
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
    case_list = cases(a.quick, only)
    for c in case_list:
        c.update(make_operands(work, c))
        if "screen" in c:
            print(f"[glm53-tp8-ar] {c['key']} screened {c['screen']}", flush=True)
    doc["case_list"] = [{k: v for k, v in c.items() if k not in ("ops", "out")} for c in case_list]
    try:
        doc["bind"] = bind_check(case_list)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[glm53-tp8-ar] bind ok={doc['bind']['ok']}; {len(case_list)} launches on {WORLD} ranks", flush=True)
    doc["ranks"], doc["comparisons"] = [], []
    try:
        doc["ranks"] = spawn_ranks(work / "switch", SWITCH_ENV, case_list)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["ranks"] = _error(e)
    for r in doc["ranks"]:
        doc["errors"].update({f"rank{r['rank']}/{k}": v for k, v in r["errors"].items()})
    if doc["ranks"]:
        doc["communicators"] = doc["ranks"][0]["communicators"]
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            for res in ex.map(compare, jobs_of(case_list, False)):
                if "error" in res:
                    doc["errors"][f"{res['key']}/compare"] = res["error"]
                else:
                    doc["comparisons"].append(res)
    doc["fma"] = {str(s): fma_of(_shot_totals(doc, s)) for s in (1, 2)}
    doc["shot_totals"] = {str(s): _shot_totals(doc, s) for s in (1, 2)}
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = case_verdicts(doc) if doc["ranks"] else {}
    for key, c in doc["cases"].items():
        t = c["totals"]
        print(f"[glm53-tp8-ar] case {key:22s} {'GO' if c['ok'] else 'NO-GO':5s} M={c['M']:4d} path={c['path']} words={t['words']}!={t['mismatch']} "
              f"normed={t['normed_words']} fma1!={t['fma1_mismatch']} fma0!={t['fma0_mismatch']} differ={t['fma_differ']} "
              f"ranks!={t['rank_mismatch']} ref={t['reference']} twin!={t['twin_mismatch']} us={c.get('us')}", flush=True)
    ar_cases = [c for c in case_list if c["phase"] == "allreduce"]
    if ar_cases and not a.no_default and doc["ranks"]:
        try:
            doc["served_default"] = default_run(work / "served_default", ar_cases, a.workers)
            doc["switch_cost"] = _cost(doc)
        except Exception as e:  # noqa: BLE001
            doc["served_default"] = {"error": _error(e)}
        print(f"[glm53-tp8-ar] served default {json.dumps(doc['served_default'], default=str)[:2000]}", flush=True)
        print(f"[glm53-tp8-ar] switch cost (us switched / default) {json.dumps(doc.get('switch_cost'))}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"GLM53-TP8-ALLREDUCE {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} fma={json.dumps(doc['fma'])} "
          f"totals={json.dumps(doc['totals'])} no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} "
          f"errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's TP8 all-reduces on this 8-GPU Hopper node against their Programs")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, every rank's words); default the system temp")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"fused M in {QUICK_FUSED_M}, all-reduce M in {QUICK_AR_M}, with the specials and probes")
    ap.add_argument("--no-default", action="store_true", help="skip the all-reduce phase under vLLM's default (symmetric memory on)")
    ap.add_argument("--only", default=",".join(PHASES), help="comma list of phases: fused, allreduce")
    a = ap.parse_args(argv)
    return 0 if run(a)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
