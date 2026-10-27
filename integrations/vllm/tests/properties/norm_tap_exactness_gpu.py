"""GPU run of the norm-tap exactness property (`verity_vllm.properties.norm_tap_exactness`): check the tap build and the Triton copy on
this GPU against the installed vLLM kernels and the IR model, and the norm-scale source on vLLM's norm modules; write the sealed record
`<out>/norm_tap_exactness.json`.  Exit status 1 unless ok; `--verify RECORD` re-derives a record's digest and verdict without a GPU.

It lives under tests/ because it seeds random generators, sets VLLM_BATCH_INVARIANT for its own process and launches vLLM's kernels
directly (lints P3, P7).  In the pod venv, with the tree on PYTHONPATH as for gate (b):

    $PY tests/properties/norm_tap_exactness_gpu.py --so /workspace/cp/norm_tap/build/verity_norm_tap.so --out DIR [--quick]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import os
import platform
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.properties import norm_tap_exactness as X


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _environment() -> dict[str, Any]:
    import torch

    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))), "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT")}
    for mod in ("vllm", "triton"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            env[mod] = f"unavailable: {e}"[:160]
    return env


def _u16(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(np.uint32)


def _same(a, b) -> bool:
    import torch

    def bits(t):
        return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)

    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(bits(a), bits(b)))


def _inputs(c: dict[str, Any], seed: int):
    """x, residual (rows, N) and weight (N,) bf16 on the GPU; a special case writes `X.SPECIAL`'s patterns, one per row, into x (or the
    residual)."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, N = c["rows"], c["N"]
    x = (torch.randn(rows, N, device="cuda", generator=g) * 2).to(torch.bfloat16)
    res = (torch.randn(rows, N, device="cuda", generator=g) * 2).to(torch.bfloat16)
    w = (torch.randn(N, device="cuda", generator=g) * 0.2 + 1).to(torch.bfloat16)
    if c["special"]:
        words = (res if c["residual_special"] else x).view(torch.int16)
        for r, (word, where) in enumerate(X.SPECIAL.values()):
            v = word - 65536 if word >= 32768 else word
            if where == "all":
                words[r].fill_(v)
            else:
                words[r, N // 2] = v
    return x, res, w


def _ir_rows(c: dict[str, Any]) -> list[int]:
    return sorted(set(range(min(X.IR_ROWS, c["rows"]))) | (set(range(c["rows"])) if c["special"] else set()))


def _scale_checks(c: dict[str, Any], kernel: str, words: list, xn, rn, wn) -> dict[str, Any]:
    w1, w2 = (_u32(s) for s in words)
    twin = X.twin_scales(kernel, xn, rn, wn, c["eps"])
    rows = _ir_rows(c)
    ir = {r: X.ir_scale(kernel, c["N"], c["eps"], xn[r], rn[r] if kernel == "cuda" else None, wn) for r in rows}
    bad_twin = np.flatnonzero(w1 != twin)
    bad_ir = [r for r in rows if ir[r] != int(w1[r])]
    return {"unwritten": int(np.count_nonzero(w1 != w2)), "scale_ne_twin": int(bad_twin.size), "ir_rows": len(rows), "scale_ne_ir": len(bad_ir),
            "first_ne_twin": [{"row": int(r), "tap": f"{int(w1[r]):08x}", "twin": f"{int(twin[r]):08x}"} for r in bad_twin[:4]],
            "first_ne_ir": [{"row": r, "tap": f"{int(w1[r]):08x}", "ir": f"{ir[r]:08x}"} for r in bad_ir[:4]],
            "scales_head": [f"{int(v):08x}" for v in w1[:4]]}


def _fills(rows: int) -> list:
    import torch

    return [torch.full((rows,), f - (1 << 32) if f >= (1 << 31) else f, dtype=torch.int32, device="cuda").view(torch.float32) for f in X.FILLS]


def check_cuda(tap, c: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    rec = {k: c[k] for k in ("name", "kernel", "N", "eps", "rows", "weight", "special", "residual_special")}
    x, res, w = _inputs(c, seed)
    wt = w if c["weight"] else None
    xi, ri = x.clone(), res.clone()
    torch.ops._C.fused_add_rms_norm(xi, ri, wt, c["eps"])
    xu, ru = x.clone(), res.clone()
    tap.fused_add_rms_norm(xu, ru, wt, c["eps"])
    outs, words = [], _fills(c["rows"])
    for s in words:
        xt, rt = x.clone(), res.clone()
        tap.fused_add_rms_norm_scale(xt, rt, wt, c["eps"], s)
        outs.append((xt, rt))
    torch.cuda.synchronize()
    wn = _u16(w) if c["weight"] else np.full(c["N"], 0x3F80, dtype=np.uint16)
    rec.update(out_eq_installed=all(_same(a, xi) and _same(b, ri) for a, b in outs), out_eq_untapped=_same(outs[0][0], xu) and _same(outs[0][1], ru),
               **_scale_checks(c, "cuda", words, _u16(x), _u16(res), wn))
    rec["ok"] = X.case_ok(rec)
    return rec


def check_triton(c: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch
    from vllm.model_executor.determinism import batch_invariant as BI

    from verity_vllm.acquire.norm_tap_src import rms_norm_tap as RT

    rec = {k: c[k] for k in ("name", "kernel", "N", "eps", "rows", "weight", "special")}
    x, _res, w = _inputs(c, seed)
    wt = w if c["weight"] else None
    rows, N = x.shape
    out_i = torch.empty_like(x)
    k_i = BI._rms_norm_kernel[(rows,)](x, wt if wt is not None else x, out_i, x.stride(0), out_i.stride(0), N, c["eps"], BLOCK_SIZE=1024,
                                       HAS_WEIGHT=wt is not None)
    out_v = BI.rms_norm_batch_invariant(x, wt, c["eps"])
    outs, words, k_t = [], _fills(rows), None
    for s in words:
        o = torch.empty_like(x)
        k_t = RT._rms_norm_kernel[(rows,)](x, wt if wt is not None else x, o, s, x.stride(0), o.stride(0), N, c["eps"], BLOCK_SIZE=1024,
                                           HAS_WEIGHT=wt is not None)
        outs.append(o)
    torch.cuda.synchronize()
    wn = _u16(w) if c["weight"] else np.full(N, 0x3F80, dtype=np.uint16)
    rec.update(out_eq_installed=all(_same(o, out_i) for o in outs), direct_eq_vllm=_same(out_v, out_i), ptx=X.ptx_check(k_i.asm["ptx"], k_t.asm["ptx"]),
               **_scale_checks(c, "triton", words, _u16(x), None, wn))
    rec["ok"] = X.case_ok(rec)
    return rec


class _Recorder:
    """The committer surface the source uses: the current step's job, the error list, `_acquire` (kept here, cloned)."""

    def __init__(self) -> None:
        self._cur: Any = None
        self.errors: list[str] = []
        self.acquired: list[tuple[str, Any]] = []

    def _acquire(self, name: str, t: Any, job: Any) -> None:
        self.acquired.append((name, t.detach().clone()))


def _toy(H: int, D: int, eps: float):
    import torch.nn as nn
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

    class Toy(nn.Module):
        """vLLM's own norms as a decoder uses them: the first norm (no residual: Triton), a fused add-norm (the CUDA op), a q norm over
        [tokens, heads, head_dim] (Triton, one row per head) and a Gemma norm (the ATen chain)."""

        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = RMSNorm(H, eps=eps)
            self.post_attention_layernorm = RMSNorm(H, eps=eps)
            self.q_norm = RMSNorm(D, eps=eps)
            self.gemma_norm = GemmaRMSNorm(H, eps=eps)

        def forward(self, x, res):
            a = self.input_layernorm(x)
            b, r2 = self.post_attention_layernorm(a.clone(), res.clone())
            q = self.q_norm(x.view(x.shape[0], -1, D))
            g = self.gemma_norm(x)
            return a, b, r2, q, g

    return Toy()


def check_source(so: str, seed: int) -> dict[str, Any]:
    import torch
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.determinism import batch_invariant as BI

    from verity_vllm.acquire.sources.norm_scale_source import NormScaleSource
    from verity_vllm.engine import hooks
    from verity_vllm.query import norm_scales as NS

    T, H, D, eps = 5, 2048, 128, 1e-5
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        # custom ops on, as the eager engine runs them (a bare VllmConfig() turns them off: RMSNorm would take forward_native)
        with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
            toy = _toy(H, D, eps).cuda()
    finally:
        torch.set_default_dtype(prev)
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for m in toy.modules():
            if isinstance(getattr(m, "weight", None), torch.nn.Parameter):
                m.weight.copy_((torch.randn(m.weight.shape, device="cuda", generator=g) * 0.2 + 1).to(m.weight.dtype))
    x = (torch.randn(T, H, device="cuda", generator=g) * 2).to(torch.bfloat16)
    res = (torch.randn(T, H, device="cuda", generator=g) * 2).to(torch.bfloat16)
    families = {"input_layernorm": "RMSNormTriton_v1", "post_attention_layernorm": "RMSNormFusedCuda_v2", "q_norm": "RMSNormTriton_v1",
                "gemma_norm": "RsqrtF32_v1"}
    rec: dict[str, Any] = {"T": T, "H": H, "D": D, "eps": eps, "modules": families,
                           "forward": {n: getattr(getattr(toy, n), "_forward_method", None).__name__ for n in families}}
    with torch.no_grad():
        ref = toy(x, res)
    sites = (torch.ops._C.fused_add_rms_norm, BI._rms_norm_kernel, torch.rsqrt)
    com = _Recorder()
    src = NormScaleSource(com, toy, families, so)
    src.install()
    try:
        with torch.no_grad():
            outside = toy(x, res)
            rec["outside_step_acquired"] = len(com.acquired)
            com._cur = types.SimpleNamespace(step=0)
            got = toy(x, res)
            com._cur = None
        torch.cuda.synchronize()
    finally:
        src.uninstall()
    rec["restored"] = (torch.ops._C.fused_add_rms_norm is sites[0] and BI._rms_norm_kernel is sites[1] and torch.rsqrt is sites[2]
                       and not any(getattr(p, "owner", None) in (torch.ops._C, BI, torch) for p in hooks.live()))
    rec["outputs_equal"] = all(_same(a, b) and _same(a, o) for a, b, o in zip(ref, got, outside))
    names = [n for n, _t in com.acquired]
    rec["names"], rec["shapes"] = names, [list(t.shape) for _n, t in com.acquired]
    rec["names_ok"] = names == [f"{m}/{NS.MEMBER}" for m in families] and rec["shapes"] == [[T], [T], [T, H // D], [T]]
    a, b, r2, q, gm = ref
    ln, pn, qn = toy.input_layernorm, toy.post_attention_layernorm, toy.q_norm
    want = {
        "input_layernorm": X.twin_scales("triton", _u16(x), None, _u16(ln.weight.data), eps),
        "post_attention_layernorm": X.twin_scales("cuda", _u16(a), _u16(res), _u16(pn.weight.data), eps),
        "q_norm": X.twin_scales("triton", _u16(x.view(T * (H // D), D)), None, _u16(qn.weight.data), eps),
        "gemma_norm": _u32(torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).reshape(-1)),
    }
    got_scales = {n.split("/", 1)[0]: _u32(t.reshape(-1)) for n, t in com.acquired}
    rec["scale_ne"] = {m: int(np.count_nonzero(got_scales.get(m, np.zeros(0, np.uint32)) != want[m])) if m in got_scales and
                       got_scales[m].shape == want[m].shape else -1 for m in families}
    rec["scales_equal"] = all(v == 0 for v in rec["scale_ne"].values())
    rec["errors"], rec["error_msgs"] = len(com.errors), com.errors[:4]
    rec["ok"] = X.source_ok(rec)
    return rec


def _guarded(fn, base: dict[str, Any], *args) -> dict[str, Any]:
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"[:400], "ok": False}


def run(so: str, out: str, quick: bool = False) -> dict[str, Any]:
    """Check the tap build at `so` and the Triton copy on this GPU and write the sealed record to `<out>/norm_tap_exactness.json`."""
    os.environ["VLLM_BATCH_INVARIANT"] = "1"                   # before vLLM's C++ reads it (vllm_is_batch_invariant caches the first read)
    import torch
    import vllm
    import vllm._custom_ops  # loads the installed _C ops
    from vllm.model_executor.determinism import batch_invariant as BI

    from verity_vllm.acquire.norm_tap_src import rms_norm_tap as RT
    from verity_vllm.acquire.norm_tap_src.build_ext import NAME, TAP_VERSION
    from verity_vllm.acquire.sources.norm_scale_source import kernel_source_sha256

    BI.init_batch_invariance()                                 # the engine worker's overrides (the Gemma chain's mean -> the Triton mean kernel)
    torch.ops.load_library(so)
    tap = getattr(torch.ops, NAME)
    installed = sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_C*.so")))
    doc: dict[str, Any] = {
        "schema": X.SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv, "environment": _environment(),
        "tap": {"so": so, "sha256": _sha256(so), "tap_version": int(tap.tap_version()), "want_version": TAP_VERSION,
                "build_info": json.loads(Path(so).with_name("build_info.json").read_text()) if Path(so).with_name("build_info.json").exists() else None},
        "installed": {"op": "_C.fused_add_rms_norm", "so": {os.path.basename(p): _sha256(p) for p in installed}},
        "triton": {"installed_sha256": kernel_source_sha256(BI._rms_norm_kernel), "pinned_sha256": RT.PINNED_SHA256,
                   "tapped_sha256": kernel_source_sha256(RT._rms_norm_kernel),
                   "diff": X.source_diff(inspect.getsource(BI._rms_norm_kernel.fn), inspect.getsource(RT._rms_norm_kernel.fn), RT.TAP_LINES)},
        "cases": []}
    print(f"[norm-tap] triton source installed {doc['triton']['installed_sha256'][:16]} pinned {RT.PINNED_SHA256[:16]} diff ok "
          f"{doc['triton']['diff']['ok']}", flush=True)
    for i, c in enumerate(X.cases(quick)):
        fn, args = (check_cuda, (tap, c, 1000 + i)) if c["kernel"] == "cuda" else (check_triton, (c, 1000 + i))
        rec = _guarded(fn, {"name": c["name"], "kernel": c["kernel"]}, *args)
        doc["cases"].append(rec)
        print(f"[norm-tap] {c['name']:28s} " + (rec.get("error") or
              f"==installed {rec['out_eq_installed']} unwritten {rec['unwritten']} !=twin {rec['scale_ne_twin']} !=ir {rec['scale_ne_ir']}/"
              f"{rec['ir_rows']}" + (f" ==untapped {rec['out_eq_untapped']}" if c["kernel"] == "cuda" else
                                     f" ==vllm {rec['direct_eq_vllm']} ptx {rec['ptx']}")) + f" -> {'OK' if rec['ok'] else 'FAIL'}", flush=True)
    doc["source"] = _guarded(check_source, {}, so, 7)
    print(f"[norm-tap] source {json.dumps({k: v for k, v in doc['source'].items() if k not in ('modules',)})[:900]}", flush=True)
    doc = X.seal(doc)
    path = Path(out) / X.RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"NORM-TAP-EXACTNESS {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} {path}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="norm-tap exactness: the tapped norm kernels compute vLLM's outputs and the IR model's scales")
    ap.add_argument("--so", help="the tap build: verity_norm_tap.so (ops/pod_norm_tap.sh)")
    ap.add_argument("--out", help=f"directory for {X.RECORD}")
    ap.add_argument("--quick", action="store_true", help=f"only the cases {', '.join(X.QUICK)}")
    ap.add_argument("--verify", metavar="RECORD", help="check a record's digest and verdict (no GPU)")
    a = ap.parse_args(argv)
    if a.verify:
        doc = json.loads(Path(a.verify).read_text())
        problems = X.verify_record(doc)
        print(f"{a.verify}: {'intact, ok=' + str(doc.get('ok')) if not problems else '; '.join(problems)} (digest {doc.get('digest')})")
        return 1 if problems or not doc.get("ok") else 0
    if not (a.so and a.out):
        ap.error("--so and --out are required (or --verify RECORD)")
    return 0 if run(a.so, a.out, a.quick)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
