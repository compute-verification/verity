"""FA2 on a target: the stock vllm-flash-attn kernel vLLM launches there, against the attention Definitions a target can bind.

One GPU process writes `<out>/fa2_target_capture.json`, the property record `fa2_target_capture` (`properties.record`).  `ok` = the
gate, MUFU, every launch's tile and determinism, the engine, every row equal to fa2_check_inf and v5 and, on its finite heads, to
v2_hopper, and the target's own binding of every row the checked `Attention_v5` (where the target binds FA2's construction).
The device gate runs first and nothing else runs when it fails (exit 3):

  gate     nvidia-smi driver, compute capability and `multi_processor_count` against `--min-driver`, `--want-cc`, `--want-sms`.
  binary   the SASS and PTX the installed `_vllm_fa2_C` carries (cuobjdump): a cubin for this arch, or the driver's JIT of compute_80 PTX.
  mufu     `program/kernels/cuda/mufu_probe.cu` built two ways, as PTX for compute_80 (JIT-compiled by the driver, as FA2 is) and natively for
           this arch: each build's ex2 / rcp tables against core's library tables (sha256), and `verify` of the pinned tables and reduction
           rule (FRAC_MODE 4) on all 2^32 inputs.
  launch   the kernel of every case (torch.profiler): `Flash_fwd_kernel_traits<D, kBlockM, kBlockN, kNWarps, ...>` against kBlockM 64 and
           `b1.fa2_kblock_n(D)`.
  cases    per head dim, `fa_tap_exactness.cases(2)` without softcap or window, a `wide` case (every operand word scaled by 2^U(-8, 8)),
           and a mixed and a decode launch at each pinned model's (heads, KV heads, head dim).  Every output row, run twice, against
             v2_hopper      Attention_v2{T,NH,KVH,D,BN=fa2_kblock_n(D),DOT=HopperBF16WgmmaDot16_v1,INV=Fa2InvSum_v1} (every block guarded)
             v3_ampere      Attention_v3{T,NH,KVH,D,BN} (the Ampere step)
             fa2_check_inf  v2_hopper with the kernel's own `Check_inf` placement: after the launch's `n_masking_steps` iterations the
                            rescale reads the running max unguarded (`max_scaled` stays guarded: softmax.h `scale_apply_exp2`)
             v5             the registered `Attention_v5{..., MASKED_FROM}` (`registry.fa2_check_inf`), which fa2_check_inf restates
           by the vectorised twins, and by the reference evaluator on every head a twin flags non-finite.  On finite heads the running max
           is finite from the row's first visible block on, so fa2_check_inf equals v2_hopper there and is evaluated on non-finite heads only.
  engine   `engine.build.build_engine(case="B0", target=...)` for the declared target: each attention layer's impl and FA version.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/fa2_target_capture_gpu.py --out DIR --want-cc 12.0 --want-sms 188 --min-driver 575
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.properties.fa_tap_exactness import PAGE, cases as tap_cases
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/fa2-target-capture/v1"
RECORD = "fa2_target_capture.json"
HDIMS = (64, 96, 128, 256)
KBLOCK_M = 64
MUFU_SRC = Path(__file__).resolve().parents[2] / "verity_vllm/program/kernels/cuda/mufu_probe.cu"
#: (heads, KV heads, head dim) of the pinned checkpoints whose attention is plain FA2 (`engine/profiles/hf_configs`; Gemma-2's softcap and
#: window launch another kernel instantiation)
MODELS = {"smollm2-135m": (9, 3, 64), "smollm2-360m": (15, 5, 64), "pythia-160m": (12, 12, 64), "qwen2.5-0.5b": (14, 2, 64),
          "llama-3.2-1b": (32, 8, 64), "tinyllama-1.1b": (32, 4, 64), "phi-3-mini": (32, 32, 96), "qwen2.5-1.5b": (12, 2, 128),
          "qwen2.5-7b": (28, 4, 128), "qwen3-4b": (32, 8, 128), "olmoe-1b-7b": (16, 16, 128), "qwen3-30b-a3b": (32, 4, 128)}
VARIANTS = ("v2_hopper", "v3_ampere", "fa2_check_inf", "v5")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _cuda_tool(name: str) -> str:
    home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    return shutil.which(name) or os.path.join(home, "bin", name)


def cpu_workers() -> int:
    """CPUs this container may use: the cgroup v2 quota when one is set (RunPod pods see the host's CPUs), else the affinity mask."""
    n = len(os.sched_getaffinity(0))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n - 1)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------ gate
def gate(min_driver: int, want_cc: str, want_sms: int) -> dict[str, Any]:
    import torch

    q = _run(["nvidia-smi", "--query-gpu=driver_version,name,compute_cap", "--format=csv,noheader"]).stdout.strip().splitlines()
    driver, name, cc = (x.strip() for x in q[0].split(","))
    p = torch.cuda.get_device_properties(0)
    rec = {"driver": driver, "name": name, "compute_cap": cc, "torch_capability": f"{p.major}.{p.minor}", "sms": int(p.multi_processor_count),
           "torch_arch_list": torch.cuda.get_arch_list(), "want": {"min_driver": min_driver, "cc": want_cc, "sms": want_sms}}
    rec["checks"] = {"driver": int(driver.split(".")[0]) >= min_driver, "cc": cc == want_cc and rec["torch_capability"] == want_cc,
                     "sms": rec["sms"] == want_sms, "torch_kernels": f"sm_{p.major}{p.minor}" in rec["torch_arch_list"]}
    rec["ok"] = all(rec["checks"].values())
    return rec


# ---------------------------------------------------------------------------------------------------------------------- binary
def _archs(so: str) -> dict[str, Any]:
    tool = _cuda_tool("cuobjdump")
    elf, ptx = _run([tool, "--list-elf", so]), _run([tool, "--list-ptx", so])
    return {"so": so, "sha256": _sha256_file(so),
            "sass_archs": sorted(set(re.findall(r"\.sm_(\d+[a-z]?)\.cubin", elf.stdout))),
            "ptx_archs": sorted(set(re.findall(r"\.sm_(\d+[a-z]?)\.ptx", ptx.stdout))),
            "cuobjdump_rc": [elf.returncode, ptx.returncode], "cuobjdump_err": (elf.stderr + ptx.stderr)[-300:]}


def fa2_library() -> str | None:
    import vllm.vllm_flash_attn as vfa

    so = glob.glob(os.path.join(os.path.dirname(vfa.__file__), "_vllm_fa2_C*.so"))
    return so[0] if so else None


def binary() -> dict[str, Any]:
    """The archs `_vllm_fa2_C` (the kernel under test) and vLLM's `_C` (the engine's other kernels) carry."""
    import vllm

    so = fa2_library()
    c = glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_C*.so"))
    return {"fa2": _archs(so) if so else {"error": "no _vllm_fa2_C*.so beside vllm.vllm_flash_attn"},
            "vllm_C": _archs(c[0]) if c else {"error": "no vllm/_C*.so"}}


# ------------------------------------------------------------------------------------------------------------------------ mufu
def mufu(work: Path, sm: str) -> dict[str, Any]:
    from verity.ml.circuits import library as L
    from verity_vllm.program.kernels import fa2_relation

    pinned = fa2_relation.tables()
    ref = work / "pinned_tables.bin"
    ref.write_bytes(np.ascontiguousarray(pinned.ex2, "<u4").tobytes() + np.ascontiguousarray(pinned.rcp, "<u4").tobytes())
    rec: dict[str, Any] = {"library": {n: L.table(n).sha256 for n in ("ex2", "rcp")}, "builds": {}}
    for tag, arch in (("jit_compute_80", ["-gencode", "arch=compute_80,code=compute_80"]), ("native", [f"-arch={sm}"])):
        exe = work / f"mufu_probe_{tag}"
        b = _run([_cuda_tool("nvcc"), "-O2", "-DFRAC_MODE=4", *arch, "-o", str(exe), str(MUFU_SRC)])
        if b.returncode:
            rec["builds"][tag] = {"error": f"nvcc: {b.stderr[-400:]}"}
            continue
        out = work / f"tables_{tag}.bin"
        t = _run([str(exe), "table", str(out)])
        words = np.fromfile(out, dtype="<u4")
        n = 1 << 23
        ex2, rcp = words[:n], words[n:2 * n]
        v = _run([str(exe), "verify", str(ref)])
        try:
            verify = json.loads(v.stdout)
        except ValueError:
            verify = {"error": (v.stdout + v.stderr)[-400:]}
        rec["builds"][tag] = {
            "table_rc": t.returncode, "verify_rc": v.returncode,
            "ex2_sha256": hashlib.sha256(ex2.tobytes()).hexdigest(), "rcp_sha256": hashlib.sha256(rcp.tobytes()).hexdigest(),
            "ex2_entries_differing": int(np.count_nonzero(ex2 != pinned.ex2)), "rcp_entries_differing": int(np.count_nonzero(rcp != pinned.rcp)),
            "verify": verify}
        b_ = rec["builds"][tag]
        b_["ok"] = (b_["ex2_sha256"] == rec["library"]["ex2"] and b_["rcp_sha256"] == rec["library"]["rcp"]
                    and verify.get("ex2_mismatches") == 0 and verify.get("rcp_mismatches") == 0)
        if not b_["ok"]:
            b_["tables_kept"] = str(out)      # a new MUFU Definition needs the measured table; kept in the run directory
        else:
            out.unlink()
    ref.unlink()
    rec["ok"] = bool(rec["builds"]) and all(b.get("ok") for b in rec["builds"].values())
    return rec


# ----------------------------------------------------------------------------------------------------------------------- cases
def case_list() -> list[dict[str, Any]]:
    """(name, H, Hk, seqlens_q, seqlens_k, head dims, wide) of every launch."""
    out = []
    for c in tap_cases(2):
        if c["softcap"] == 0 and list(c["window"]) == [-1, -1]:
            out.append({**c, "hdims": list(HDIMS)})
    out.append({"name": "wide", "H": 4, "Hk": 2, "seqlens_q": [1, 64, 65, 200, 1], "seqlens_k": [1, 64, 65, 200, 700], "softcap": 0.0,
                "window": [-1, -1], "wide": True, "hdims": list(HDIMS)})
    for m, (H, Hk, D) in MODELS.items():
        out.append({"name": f"{m}_mixed", "H": H, "Hk": Hk, "seqlens_q": [1, 17, 64, 65, 129, 1], "seqlens_k": [1, 17, 64, 65, 129, 600],
                    "softcap": 0.0, "window": [-1, -1], "hdims": [D]})
        out.append({"name": f"{m}_decode", "H": H, "Hk": Hk, "seqlens_q": [1] * 5, "seqlens_k": [1, 64, 65, 300, 1031],
                    "softcap": 0.0, "window": [-1, -1], "hdims": [D]})
    return out


def _widen(t: dict[str, Any], seed: int) -> None:
    """Every q / k / v word times 2^U(-8, 8): exact in bf16 away from the range ends, spreads the products' exponents."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    for name in ("q", "k", "v"):
        x = t[name]
        e = torch.randint(-8, 9, x.shape, device="cuda", generator=g).to(torch.float32)
        t[name] = (x.to(torch.float32) * torch.exp2(e)).to(torch.bfloat16)


LABEL = "fa2-capture/launch/"


def kernels_by_launch(prof) -> dict[int, list[str]]:
    """The FA kernels each launch ran, from ONE profiler session (a session per launch stops recording device activity after about
    20 sessions): a device kernel belongs to launch i when it starts inside the host window of the `LABEL + i` range, which closes
    after a synchronize."""
    from torch.autograd import DeviceType

    ev = list(prof.events())
    windows = {int(e.name[len(LABEL):]): e.time_range for e in ev if e.name.startswith(LABEL)}
    out: dict[int, set[str]] = {i: set() for i in windows}
    for e in ev:
        if e.device_type == DeviceType.CUDA and "flash" in e.name.lower():
            for i, w in windows.items():
                if w.start <= e.time_range.start <= w.end:
                    out[i].add(e.name)
    return {i: sorted(s) for i, s in out.items()}


_TRAITS = re.compile(r"Flash_fwd_kernel_traits<(\d+), (\d+), (\d+), (\d+),")


def launch_geometry(names: list[str], D: int) -> dict[str, Any]:
    from verity_vllm.program.registry import b1

    tiles = sorted({tuple(int(x) for x in m.groups()) for n in names for m in [_TRAITS.search(n)] if m})
    want = (D, KBLOCK_M, b1.fa2_kblock_n(D), 4)
    return {"traits": [list(t) for t in tiles], "want": list(want), "ok": tiles == [want],
            "splitkv": all("splitkv" in n for n in names), "names": [n[:400] for n in names]}


def guarded_from(sq_launch_max: int, sq: int, sk: int, r: int, BN: int) -> int:
    """The lowest key block FA2's splitkv kernel runs with `Check_inf` for row `r` of a sequence (sq query rows, sk keys):
    flash_fwd_kernel.h `compute_attn_1rowblock_splitkv` runs its first `n_masking_steps` iterations (from `n_block_max - 1` down) with
    `Check_inf = Is_causal || Is_local`, the rest with `Is_local` (false); varlen_fwd makes a launch with max_seqlen_q == 1 non-causal."""
    nb_all = -(-sk // BN)
    if sq_launch_max == 1:
        return nb_all - 1                                          # non-causal: one masking step
    m_block = r // KBLOCK_M
    n_block_max = min(nb_all, -(-((m_block + 1) * KBLOCK_M + sk - sq) // BN))
    return max(0, n_block_max - (-(-KBLOCK_M // BN) + 1))           # causal, Is_even_MN false: ceil(kBlockM / kBlockN) + 1 steps


def _fa2_check_inf_head(q: list[int], k: np.ndarray, v: np.ndarray, BN: int, gfrom: int) -> list[int]:
    """One head of `Attention_v2{..., DOT=HopperBF16WgmmaDot16_v1, INV=Fa2InvSum_v1}` restated over the registered primitives by the
    reference evaluator, with FA2's `Check_inf` placement: block b guards the rescale's running max iff b >= gfrom; `max * scale` is guarded
    in every block.  `AttnBlock_v2`'s body otherwise, line for line (gfrom 0 is `AttentionHead_v2`)."""
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity_vllm.program.registry import b1, hopper
    from verity_vllm.program.registry import prims as P

    def E(prim, *a):
        return int(evaluate(prim, *a)[0])

    DOT = hopper.HopperBF16WgmmaDot16
    T, D = k.shape
    scale = b1._scale_log2_bits(D)
    nb = -(-T // BN)
    m = 0
    l = [0] * 4
    o = [0] * D
    for b in range(nb - 1, -1, -1):
        first = b == nb - 1
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        s = []
        for c in range(nvis):
            acc = 0
            for i in range(D // 16):
                acc = E(DOT, acc, q[16 * i:16 * i + 16], [int(x) for x in k[lo + c, 16 * i:16 * i + 16]])
            s.append(acc)
        bmax = s[0]
        for c in range(1, nvis):
            bmax = E(P.F32Max, bmax, s[c])
        m_new = bmax if first else E(P.F32Max, m, bmax)
        guard = E(P.GuardNegInfZero, m_new)
        if not first:
            m_cur = guard if b >= gfrom else m_new
            sc = E(P.MufuEx2Ftz, E(P.F32MulFtz, E(P.F32SubFtz, m, m_cur), scale))
        max_scaled = E(P.F32MulFtz, guard, scale)
        p = [E(P.MufuEx2Ftz, E(P.F32FmaSubFtz, s[c], scale, max_scaled)) for c in range(nvis)]
        lanes = []
        for ln in range(4):
            cols = [c for c in range(nvis) if b1._lane(c) == ln]
            first_p = p[cols[0]] if cols else 0
            acc = first_p if first else E(P.F32FmaFtz, sc, l[ln], first_p)
            for c in cols[1:]:
                acc = E(P.F32AddFtz, acc, p[c])
            lanes.append(acc)
        pb = [E(P.F2fpBf16, x) for x in p] + [0] * (BN - nvis)
        o = [0] * D if first else [E(P.F32MulFtz, o[d], sc) for d in range(D)]
        for ks in range(BN // 16):
            k0 = 16 * ks
            if k0 >= nvis:
                break
            prow = pb[k0:k0 + 16]
            for d in range(D):
                vcol = [int(v[lo + k0 + j, d]) if k0 + j < nvis else 0 for j in range(16)]
                o[d] = E(DOT, o[d], prow, vcol)
        m, l = m_new, lanes
    total = E(P.F32AddFtz, E(P.F32AddFtz, l[0], l[2]), E(P.F32AddFtz, l[1], l[3]))
    inv = E(P.Fa2InvSum, total)
    return [E(P.F2fpBf16, E(P.F32MulFtz, o[d], inv)) for d in range(D)]


def compare(job: dict[str, Any]) -> dict[str, Any]:
    """Every output row of one launch against the three constructions (a worker process: numpy words in, a record out)."""
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.registry import b1, hopper
    from verity_vllm.program.registry import fa2_check_inf as F2
    from verity_vllm.program.registry import prims as P
    from verity_vllm.program.frontend import targets as TG

    c, D, q, k, v, o, bt = job["case"], job["D"], job["q"], job["k"], job["v"], job["out"], job["block_table"]
    tp = TargetProfile.default().with_overrides(compute_capability=tuple(job["cc"]), num_sms=int(job["sms"])) if job.get("cc") else None
    H, Hk = c["H"], c["Hk"]
    BN = b1.fa2_kblock_n(D)
    g = H // Hk
    kvh = np.arange(H) // g
    sq_max = max(c["seqlens_q"])
    twin = {"v2_hopper": DR.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"], "v3_ampere": DR.TC_DOT16_BY_PRIM["AmpereBF16TcDot16_v2"]}
    rec: dict[str, Any] = {"rows": 0, "heads": 0, "nonfinite_heads": 0, "reference_heads": 0, "self_check_mismatch": 0, "binding_mismatch": 0,
                           **{f"{n}_mismatch": 0 for n in VARIANTS}, **{f"{n}_mismatch_finite": 0 for n in VARIANTS}, "first_mismatch": {}}
    tok = 0
    for seq, (sq, sk) in enumerate(zip(c["seqlens_q"], c["seqlens_k"])):
        keys = np.arange(sk)
        kseq, vseq = k[bt[seq, keys // PAGE], keys % PAGE], v[bt[seq, keys // PAGE], keys % PAGE]       # (sk, Hk, D)
        for r in range(sq):
            T = r + 1 + sk - sq
            kk = np.ascontiguousarray(np.transpose(kseq[:T], (1, 0, 2))[kvh])                             # (H, T, D)
            vv = np.ascontiguousarray(np.transpose(vseq[:T], (1, 0, 2))[kvh])
            got = o[tok + r]
            want: dict[str, np.ndarray] = {}
            nonfinite = np.zeros(H, dtype=bool)
            for name, dot16 in twin.items():
                st = DR.attention_head_states(q[tok + r], kk, vv, BN, dot16=dot16, inv=DR.INV_SUM_BY_PRIM["Fa2InvSum_v1"])
                want[name] = np.asarray(st["out"], dtype=np.uint16)
                nonfinite |= np.asarray(st["nonfinite"], dtype=bool)
            gfrom = guarded_from(sq_max, sq, sk, r, BN)
            v5 = bind(F2.AttentionHeadV5, T=T, D=D, BN=BN, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum, MASKED_FROM=gfrom)
            st = DR.attention_head_states(q[tok + r], kk, vv, BN, dot16=twin["v2_hopper"], inv=DR.INV_SUM_BY_PRIM["Fa2InvSum_v1"],
                                          masked_from=gfrom, rescale_only=True)
            want["v5"] = np.asarray(st["out"], dtype=np.uint16)
            want["fa2_check_inf"] = want["v2_hopper"].copy()
            if tp is not None and TG.fa2_check_inf(tp):           # the target's own binding of this row is the Definition checked here
                spec = TG.attention_spec(tp, T, H, Hk, D, sq_max, row=r, own_rows=sq)
                rec["binding_mismatch"] += spec.id != bind(F2.AttentionV5, T=T, NH=H, KVH=Hk, D=D, BN=BN, DOT=hopper.HopperBF16WgmmaDot16,
                                                             INV=P.Fa2InvSum, MASKED_FROM=gfrom).id
            for h in np.flatnonzero(nonfinite):
                qh, kf, vf = q[tok + r, h].tolist(), kk[h].reshape(-1).tolist(), vv[h].reshape(-1).tolist()
                want["v2_hopper"][h] = evaluate(bind(b1.AttentionHeadV2, T=T, D=D, BN=BN, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum),
                                                qh, kf, vf)
                want["v3_ampere"][h] = evaluate(bind(b1.AttentionHeadV3, T=T, D=D, BN=BN), qh, kf, vf)
                want["fa2_check_inf"][h] = _fa2_check_inf_head(qh, kk[h], vv[h], BN, gfrom)
                want["v5"][h] = evaluate(v5, qh, kf, vf)
                if (_fa2_check_inf_head(qh, kk[h], vv[h], BN, 0) != [int(x) for x in want["v2_hopper"][h]]
                        or [int(x) for x in want["fa2_check_inf"][h]] != [int(x) for x in want["v5"][h]]):
                    rec["self_check_mismatch"] += 1          # the restatement is AttentionHead_v2 with every block guarded, AttentionHead_v5 at gfrom
                rec["reference_heads"] += 1
            rec["rows"] += 1
            rec["heads"] += H
            rec["nonfinite_heads"] += int(nonfinite.sum())
            for name in VARIANTS:
                bad = np.flatnonzero((want[name] != got).any(axis=1))
                rec[f"{name}_mismatch"] += int(bad.size)
                rec[f"{name}_mismatch_finite"] += int(np.count_nonzero(~nonfinite[bad]))
                if bad.size and name not in rec["first_mismatch"]:
                    h = int(bad[0])
                    rec["first_mismatch"][name] = {"token": tok + r, "head": h, "T": T, "guarded_from": gfrom, "nonfinite": bool(nonfinite[h]),
                                                   "want": [int(x) for x in want[name][h][:4]], "kernel": [int(x) for x in got[h][:4]]}
        tok += sq
    return {"case": c["name"], "D": D, **rec}


def gpu_case(op, c: dict[str, Any], D: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Launch case `c` at head dim `D` as vLLM does, twice; returns (the launch record, the comparison job)."""
    t = G._inputs(c, D, seed)
    if c.get("wide"):
        _widen(t, seed + 1)
    args = G._vllm_args(2, c, t)
    out1, _ = G._call(op, args)
    out2, _ = G._call(op, args)
    launch = {"case": c["name"], "D": D, "H": c["H"], "Hk": c["Hk"], "seqlens_q": c["seqlens_q"], "seqlens_k": c["seqlens_k"],
              "deterministic": G._same(out1, out2)}
    job = {"case": {k: c[k] for k in ("name", "H", "Hk", "seqlens_q", "seqlens_k")}, "D": D, "q": G._words(t["q"]), "k": G._words(t["k"]),
           "v": G._words(t["v"]), "out": G._words(out1), "block_table": t["block_table"].cpu().numpy()}
    return launch, job


# ---------------------------------------------------------------------------------------------------------------------- engine
def engine(cc: tuple[int, int], sms: int) -> dict[str, Any]:
    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm.program.frontend.target_profile import TargetProfile

    target = TargetProfile.default().with_overrides(compute_capability=cc, num_sms=sms)
    try:
        eng = build_engine(case="B0", target=target)
    except Exception as e:  # noqa: BLE001
        return {"target": target.to_json(), "error": f"{type(e).__name__}: {e}"[:600], "ok": False}
    impls = {}
    for name, mod in eng.model.named_modules():
        impl = getattr(mod, "impl", None)
        if impl is not None and hasattr(impl, "forward"):
            impls[name] = {"impl": type(impl).__qualname__, "flash_attn_version": getattr(impl, "vllm_flash_attn_version", None)}
    kinds = sorted({(i["impl"], i["flash_attn_version"]) for i in impls.values()}, key=str)
    bad = target_mismatches(eng)
    return {"target": target.to_json(), "layers": len(impls), "impls": [list(x) for x in kinds], "target_mismatches": bad,
            "ok": bool(impls) and not bad and kinds == [("FlashAttentionImpl", 2)]}


# ------------------------------------------------------------------------------------------------------------------------- run
def _verdict(doc: dict[str, Any]) -> bool:
    launches, rows = doc["launches"], doc["cases"]
    return (doc["gate"]["ok"] and doc["mufu"].get("ok") is True and doc["engine"].get("ok") is True
            and bool(launches) and all(x["ok"] and x["deterministic"] for x in launches)
            and bool(rows) and all(x.get("v2_hopper_mismatch_finite") == 0 and x.get("fa2_check_inf_mismatch") == 0 and x.get("v5_mismatch") == 0
                                   and x.get("binding_mismatch") == 0
                                   and x.get("self_check_mismatch") == 0 for x in rows))


def run(out: str, min_driver: int, want_cc: str, want_sms: int, workers: int, only: list[str] | None, skip_engine: bool) -> dict[str, Any]:
    import torch

    so = fa2_library()
    if so:
        torch.ops.load_library(so)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
                           "environment": G._environment(), "gate": gate(min_driver, want_cc, want_sms)}
    print(f"[fa2-capture] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("FA2-TARGET-CAPTURE GATE-FAIL", flush=True)
        raise SystemExit(3)
    p = torch.cuda.get_device_properties(0)
    sm = f"sm_{p.major}{p.minor}"
    doc["binary"] = binary()
    for k, b in doc["binary"].items():
        print(f"[fa2-capture] binary {k}: sass={b.get('sass_archs')} ptx={b.get('ptx_archs')}", flush=True)
    with tempfile.TemporaryDirectory(dir=outdir) as work:
        doc["mufu"] = mufu(Path(work), sm)
        for b in doc["mufu"]["builds"].values():
            if b.get("tables_kept"):
                kept = outdir / Path(b["tables_kept"]).name
                os.replace(b["tables_kept"], kept)
                b["tables_kept"] = str(kept)
    for k, b in doc["mufu"]["builds"].items():
        print(f"[fa2-capture] mufu {k}: ok={b.get('ok')} ex2!={b.get('ex2_entries_differing')} rcp!={b.get('rcp_entries_differing')} "
              f"verify={ {x: (b.get('verify') or {}).get(x) for x in ('ex2_mismatches', 'rcp_mismatches')} } {b.get('error', '')}", flush=True)
    from torch.profiler import ProfilerActivity, profile, record_function

    op = torch.ops._vllm_fa2_C.varlen_fwd
    doc["launches"], jobs = [], []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i, c in enumerate(case_list()):
            if only and c["name"] not in only:
                continue
            for D in c["hdims"]:
                with record_function(f"{LABEL}{len(jobs)}"):
                    launch, job = gpu_case(op, c, D, 1000 * D + i)
                    job.update(cc=[p.major, p.minor], sms=int(p.multi_processor_count))
                    torch.cuda.synchronize()
                doc["launches"].append(launch)
                jobs.append(job)
    names = kernels_by_launch(prof)
    for i, launch in enumerate(doc["launches"]):
        launch.update(launch_geometry(names.get(i, []), launch["D"]))
        print(f"[fa2-capture] launch hd{launch['D']} {launch['case']:24s} traits={launch['traits']} ok={launch['ok']} "
              f"det={launch['deterministic']}", flush=True)
    doc["cases"] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for rec in ex.map(compare, jobs):
            doc["cases"].append(rec)
            print(f"[fa2-capture] rows hd{rec['D']} {rec['case']:24s} rows={rec['rows']} heads={rec['heads']} nonfinite={rec['nonfinite_heads']} "
                  + " ".join(f"{n}!={rec[f'{n}_mismatch']}(finite {rec[f'{n}_mismatch_finite']})" for n in VARIANTS)
                  + f" self!={rec['self_check_mismatch']}", flush=True)
    doc["engine"] = {"skipped": True, "ok": True} if skip_engine else engine((p.major, p.minor), int(p.multi_processor_count))
    print(f"[fa2-capture] engine {json.dumps(doc['engine'])[:600]}", flush=True)
    doc["totals"] = {k: sum(x[k] for x in doc["cases"]) for n in VARIANTS for k in (f"{n}_mismatch", f"{n}_mismatch_finite")}
    doc["totals"].update(rows=sum(x["rows"] for x in doc["cases"]), heads=sum(x["heads"] for x in doc["cases"]),
                         nonfinite_heads=sum(x["nonfinite_heads"] for x in doc["cases"]))
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"FA2-TARGET-CAPTURE {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} totals={json.dumps(doc['totals'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FA2 on this GPU against the attention Definitions a target can bind")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="12.0")
    ap.add_argument("--want-sms", type=int, default=188)
    ap.add_argument("--workers", type=int, default=cpu_workers())
    ap.add_argument("--only", default=None, help="comma list of case names")
    ap.add_argument("--skip-engine", action="store_true")
    a = ap.parse_args(argv)
    doc = run(a.out, a.min_driver, a.want_cc, a.want_sms, a.workers, a.only.split(",") if a.only else None, a.skip_engine)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
