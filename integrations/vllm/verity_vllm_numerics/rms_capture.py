"""Capture natural-input fixtures of the batch-invariant Triton ``_rms_norm_kernel`` as vLLM runs it (W11).

Runs on the GPU pod inside the pinned venv.  In the profile (``VLLM_BATCH_INVARIANT=1``, eager) the
Triton kernel is launched by ``RMSNorm.forward_cuda`` -> ``rms_norm_batch_invariant(x, weight, eps)``
whenever there is no residual, i.e. for ``model.layers.0.input_layernorm`` (every other norm of the
model carries a residual and goes to ``_C.fused_add_rms_norm``, W4b's model).  We rebind
``vllm.model_executor.layers.layernorm.rms_norm_batch_invariant`` and record, after ``cuda.synchronize``,
every call's input rows, weight, eps and output rows (raw BF16 words) for the same three-prompt run
as ``attn_capture.py`` (5 / 700 / 2000 prompt tokens, 300 / 72 / 72 generated tokens).  The RMSNorm
module that made each call is identified by its weight's data pointer against ``model.named_modules()``.

Kernel identity: the Triton cache entry (``~/.triton/cache/<hash>/_rms_norm_kernel.{json,ptx,cubin,...}``)
whose ``.json`` was touched by this run is copied next to the fixture (``kernel/``) with sha256s, and
the CUDA profiler's kernel names are recorded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def words(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16, t.dtype
    return t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def triton_cache_entries() -> dict[str, float]:
    root = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    out = {}
    if root.exists():
        for j in root.glob("*/_rms_norm_kernel.json"):
            out[str(j.parent)] = j.stat().st_mtime
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--long-tokens", type=int, default=2000)
    ap.add_argument("--mid-tokens", type=int, default=700)
    ap.add_argument("--decode-short", type=int, default=300)
    ap.add_argument("--decode-long", type=int, default=72)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "kernel").mkdir(exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from verity_vllm_adapter import profile as prof  # noqa: E402

    prof.apply_env()
    from vllm import LLM, SamplingParams  # noqa: E402
    import vllm.model_executor.layers.layernorm as ln_mod  # noqa: E402
    from vllm.model_executor.determinism import batch_invariant as bi  # noqa: E402

    cps = prof.load_checkpoints(Path(__file__).resolve().parents[1] / "manifests" / "checkpoints.json")
    cp = cps[args.model]
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=True)
    kw["enable_prefix_caching"] = False
    cache_before = triton_cache_entries()
    llm = LLM(**kw)
    tok = llm.get_tokenizer()

    license_text = Path(cp["local_path"], "LICENSE").read_text() if Path(cp["local_path"], "LICENSE").exists() else ""
    readme_text = Path(cp["local_path"], "README.md").read_text() if Path(cp["local_path"], "README.md").exists() else ""
    corpus = (readme_text + "\n\n" + license_text + "\n\n") * 4
    corpus_ids = tok(corpus)["input_ids"]
    if len(corpus_ids) < args.long_tokens + 8:
        raise SystemExit(f"corpus too short: {len(corpus_ids)} tokens")
    prompts = [
        {"name": "short", "prompt_token_ids": tok("The capital of France is")["input_ids"], "max_tokens": args.decode_short},
        {"name": "mid", "prompt_token_ids": corpus_ids[: args.mid_tokens], "max_tokens": args.decode_long},
        {"name": "long", "prompt_token_ids": corpus_ids[8 : 8 + args.long_tokens], "max_tokens": args.decode_long},
    ]

    # ---- module names by weight pointer ----------------------------------------------------------
    ptr_to_name: dict[int, str] = {}
    for obj in gc.get_objects():
        if isinstance(obj, torch.nn.Module) and type(obj).__name__.endswith("ForCausalLM"):
            for name, mod in obj.named_modules():
                w = getattr(mod, "weight", None)
                if isinstance(w, torch.Tensor) and type(mod).__name__ == "RMSNorm":
                    ptr_to_name[w.data_ptr()] = name
    # ---- interception ----------------------------------------------------------------------------
    orig = ln_mod.rms_norm_batch_invariant
    assert orig is bi.rms_norm_batch_invariant
    state = {"active": False, "n_total": 0, "n_residual": 0, "n_step": 0}
    calls: list[dict] = []
    weights: dict[str, dict] = {}

    def wrapper(input, weight, eps=1e-6, residual=None):  # noqa: A002
        res = orig(input, weight, eps, residual)
        state["n_total"] += 1
        if residual is not None:
            state["n_residual"] += 1
            return res
        if not state["active"]:
            return res
        torch.cuda.synchronize()
        x2 = input.reshape(-1, input.shape[-1])
        o2 = res.reshape(-1, res.shape[-1])
        seq = len(calls)
        wkey = f"w{weight.data_ptr():x}" if weight is not None else "none"
        if weight is not None and wkey not in weights:
            wf = f"weights/{wkey}.npy"
            (out / "weights").mkdir(exist_ok=True)
            np.save(out / wf, words(weight))
            weights[wkey] = {"file": wf, "module": ptr_to_name.get(weight.data_ptr()), "n": int(weight.numel()), "dtype": str(weight.dtype)}
        rec = {
            "seq": seq, "module": ptr_to_name.get(weight.data_ptr()) if weight is not None else None,
            "rows": int(x2.shape[0]), "n_cols": int(x2.shape[1]), "eps": float(eps),
            "phase": "prefill" if x2.shape[0] > 3 else "decode",
            "input_strides": list(input.stride()), "input_shape": list(input.shape), "contiguous": bool(input.is_contiguous()),
            "dtype": str(input.dtype), "out_dtype": str(res.dtype), "weight": wkey,
            "x_file": f"calls/{seq:05d}_x.npy", "out_file": f"calls/{seq:05d}_out.npy", "engine_call": state["n_total"],
        }
        np.save(out / rec["x_file"], words(x2))
        np.save(out / rec["out_file"], words(o2))
        calls.append(rec)
        return res

    ln_mod.rms_norm_batch_invariant = wrapper

    llm.generate([{"prompt_token_ids": tok("Hello, world. This is a warm-up prompt.")["input_ids"]}], SamplingParams(temperature=0.0, max_tokens=4))
    torch.cuda.synchronize()
    state["active"] = True
    t0 = time.perf_counter()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof_:
        outs = llm.generate(
            [{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
            [SamplingParams(temperature=0.0, max_tokens=p["max_tokens"]) for p in prompts],
        )
        torch.cuda.synchronize()
    state["active"] = False
    ln_mod.rms_norm_batch_invariant = orig
    kernel_ids = {ev.key: int(ev.count) for ev in prof_.key_averages() if "norm" in ev.key.lower()}

    # ---- kernel identity: the Triton cache entry used ---------------------------------------------
    cache_after = triton_cache_entries()
    used = [d for d, m in cache_after.items() if d not in cache_before or m != cache_before[d]]
    if not used:  # nothing recompiled: fall back to the newest entry whose metadata matches
        used = sorted(cache_after, key=lambda d: cache_after[d])[-1:]
    kernel_files = {}
    for d in used:
        for p in sorted(Path(d).iterdir()):
            dst = out / "kernel" / p.name
            shutil.copy2(p, dst)
            kernel_files[p.name] = {"sha256": sha256_file(dst), "bytes": dst.stat().st_size, "cache_dir": Path(d).name}
    kmeta = {}
    kj = out / "kernel" / "_rms_norm_kernel.json"
    if kj.exists():
        kmeta = json.loads(kj.read_text())
    import triton  # noqa: E402
    import vllm  # noqa: E402

    meta = {
        "schema": "verity-vllm/rmsnorm-fixture/v1",
        "model": args.model, "model_local_path": cp["local_path"],
        "vllm": vllm.__version__, "torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
        "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
        "interception": "rebind vllm.model_executor.layers.layernorm.rms_norm_batch_invariant; records every residual-free call "
                        "(input rows, weight, eps, output rows) after cuda.synchronize; module name by weight data_ptr",
        "kernel_source": "vllm/model_executor/determinism/batch_invariant.py::_rms_norm_kernel (BLOCK_SIZE=1024, HAS_WEIGHT=True)",
        "kernel_source_sha256": sha256_file(Path(bi.__file__)),
        "triton_cache_entries_used": [Path(d).name for d in used], "kernel_files": kernel_files, "kernel_metadata": kmeta,
        "profiler_kernels": kernel_ids,
        "calls_total": state["n_total"], "calls_with_residual": state["n_residual"],
        "prompts": [{"name": p["name"], "n_tokens": len(p["prompt_token_ids"]), "max_tokens": p["max_tokens"]} for p in prompts],
        "outputs": [o.outputs[0].text[:200] for o in outs],
        "output_token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "generate_seconds": time.perf_counter() - t0,
        "weights": weights, "n_calls": len(calls), "calls": calls,
    }
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
    summary = {
        "n_calls": len(calls), "rows": sum(c["rows"] for c in calls), "modules": sorted({c["module"] for c in calls}, key=str),
        "n_cols": sorted({c["n_cols"] for c in calls}), "eps": sorted({c["eps"] for c in calls}),
        "kernels": kernel_ids, "cache_used": [Path(d).name for d in used],
        "total_bytes": sum(p.stat().st_size for p in files),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
