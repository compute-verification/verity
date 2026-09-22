"""Capture the FlashAttention-2 operands vLLM feeds ``flash_attn_varlen_func`` for Qwen2.5-1.5B
under the POC profile (Table 2 natural-input fixtures for the independent attention model, W11).

Interception (same style as W1's ``install_kernel_wrappers`` / W4b's ``engine_gemm_capture``): rebind
the module-level name ``vllm.v1.attention.backends.flash_attn.flash_attn_varlen_func`` (the one
``FlashAttentionImpl.forward`` calls) so every attention call of every layer passes through a wrapper,
and wrap ``FlashAttentionImpl.forward`` to learn the layer name.  After ``cuda.synchronize`` the
wrapper stores, per (request, layer):

* ``kv/<req>_L<layer>_{k,v}.npy`` -- the K / V rows of positions ``0..Lk_max`` gathered from the paged
  cache through the block table (BF16 words, ``[Lk_max, KVH, D]``).  Written once per (request, layer);
  every later call checks that the rows it reads are a bit-identical prefix of what was saved (the KV
  cache is write-once per position), so decode fixtures only store ``Lk``.
* ``calls/<seq>_{q,out}.npy`` -- the query rows and the kernel's output rows for that request in that
  call (BF16 words, ``[n_rows, H, D]``); for long prefill chunks a sampled subset of rows
  (``rows`` in the index), since row ``i`` depends only on ``q_i`` and ``K/V[0..pos_i]``.

Everything else the kernel is given (softmax_scale, causal, window, softcap, fa_version, num_splits,
block_size, cu_seqlens/seqused, the kernel names seen by the CUDA profiler) goes to ``index.json``.

    source /workspace/venv/bin/activate
    cd /workspace/veritor/vllm-poc && PYTHONPATH=. python -m verity_vllm_numerics.attn_capture OUTDIR

Engine flags are W1's profile (``verity_vllm_adapter.profile``: ``VLLM_BATCH_INVARIANT=1`` etc.) with
``enforce_eager=True`` and prefix caching off (the replay configuration).  The attention kernel is
the same custom op under ``torch.compile`` (it is never inlined), and ``VLLM_BATCH_INVARIANT`` pins
``num_splits=1`` in both modes (``flash_attn.py: if envs.VLLM_BATCH_INVARIANT: max_num_splits = 1``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def words(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16, t.dtype
    return t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def prefill_rows(lq: int, lk: int, limit: int) -> list[int]:
    """Rows of a prefill chunk to store: all if ``lq <= limit``; else the first 64, the rows around
    every 64-row tile boundary of the *key* position, and every 16th row."""
    if lq <= limit:
        return list(range(lq))
    start = lk - lq  # key position of row 0
    keep = set(range(min(64, lq)))
    keep |= set(range(0, lq, 16))
    for pos in range(start, lk):
        if pos % 64 in (0, 1, 63) or pos % 16 in (0, 15):
            r = pos - start
            if 0 <= r < lq:
                keep.add(r)
    keep.add(lq - 1)
    return sorted(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layers", default="0,13,27", help="layer indices to record")
    ap.add_argument("--long-tokens", type=int, default=2000, help="token length of the long prompt")
    ap.add_argument("--mid-tokens", type=int, default=700)
    ap.add_argument("--decode-short", type=int, default=300, help="decode steps for the short prompt")
    ap.add_argument("--decode-long", type=int, default=72, help="decode steps for the mid / long prompts")
    ap.add_argument("--prefill-row-limit", type=int, default=256)
    ap.add_argument("--max-kv-mb", type=float, default=40.0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "kv").mkdir(exist_ok=True)
    layers = {int(x) for x in args.layers.split(",")}

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from verity_vllm_adapter import profile as prof  # noqa: E402  (W1's profile)

    prof.apply_env()
    from vllm import LLM, SamplingParams  # noqa: E402
    import vllm.v1.attention.backends.flash_attn as fa_mod  # noqa: E402

    cps = prof.load_checkpoints(Path(__file__).resolve().parents[1] / "manifests" / "checkpoints.json")
    cp = cps[args.model]
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=True)
    kw["enable_prefix_caching"] = False
    llm = LLM(**kw)
    tok = llm.get_tokenizer()

    # ---- prompts: natural text, three lengths -----------------------------------------------------
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

    # ---- interception ----------------------------------------------------------------------------
    orig_fa = fa_mod.flash_attn_varlen_func
    orig_forward = fa_mod.FlashAttentionImpl.forward
    state = {"active": False, "layer": None, "layer_name": None, "n_calls_total": 0}
    calls: list[dict] = []
    kv_store: dict[tuple[int, int], dict] = {}  # (req_key, layer) -> {"k": np, "v": np, "n": int}
    kv_bytes = [0]
    consts: dict = {}
    kernel_ids: dict[str, dict] = {}

    def forward_wrapper(self, layer, *a, **k):
        name = getattr(layer, "layer_name", None)
        state["layer_name"] = name
        m = LAYER_RE.search(name or "")
        state["layer"] = int(m.group(1)) if m else None
        return orig_forward(self, layer, *a, **k)

    def gather_kv(cache: torch.Tensor, bt_row: torch.Tensor, n: int, block_size: int) -> torch.Tensor:
        nb = (n + block_size - 1) // block_size
        blocks = cache[bt_row[:nb].long()]  # [nb, block_size, KVH, D]
        return blocks.reshape(nb * block_size, *cache.shape[2:])[:n]

    def fa_wrapper(*a, **kw):
        res = orig_fa(*a, **kw)
        state["n_calls_total"] += 1
        if not state["active"] or a:
            return res
        layer = state["layer"]
        if layer not in layers:
            return res
        torch.cuda.synchronize()
        q, kc, vc, o = kw["q"], kw["k"], kw["v"], kw["out"]
        cu = kw["cu_seqlens_q"].cpu().tolist()
        seqused = kw["seqused_k"].cpu().tolist()
        bt = kw["block_table"]
        block_size = int(kc.shape[1])
        if not consts:
            consts.update({
                "num_heads": int(q.shape[1]), "num_kv_heads": int(kc.shape[2]), "head_size": int(q.shape[2]),
                "block_size": block_size, "softmax_scale": float(kw["softmax_scale"]), "causal": bool(kw["causal"]),
                "window_size": list(kw["window_size"]) if kw.get("window_size") is not None else None,
                "softcap": float(kw.get("softcap") or 0.0), "fa_version": int(kw["fa_version"]),
                "num_splits": int(kw.get("num_splits") or 0), "alibi": kw.get("alibi_slopes") is not None,
                "scheduler_metadata": kw.get("scheduler_metadata") is not None,
                "q_descale": kw.get("q_descale") is not None, "k_descale": kw.get("k_descale") is not None,
                "v_descale": kw.get("v_descale") is not None, "s_aux": kw.get("s_aux") is not None,
                "dtype": str(q.dtype), "kv_cache_shape": list(kc.shape), "kv_cache_dtype": str(kc.dtype),
                "kv_cache_strides": list(kc.stride()), "q_strides": list(q.stride()), "out_strides": list(o.stride()),
                "max_seqlen_q_first": int(kw["max_seqlen_q"]), "max_seqlen_k_first": int(kw["max_seqlen_k"]),
            })
        for b in range(len(cu) - 1):
            lq = cu[b + 1] - cu[b]
            lk = int(seqused[b])
            if lq <= 0:
                continue
            req_key = int(bt[b, 0].item())
            key = (req_key, layer)
            k_rows = gather_kv(kc, bt[b], lk, block_size)
            v_rows = gather_kv(vc, bt[b], lk, block_size)
            kw_ = words(k_rows)
            vw_ = words(v_rows)
            ent = kv_store.get(key)
            if ent is None:
                ent = kv_store[key] = {"k": kw_, "v": vw_, "n": lk, "prefix_checks": 0, "prefix_mismatch": 0}
                kv_bytes[0] += kw_.nbytes + vw_.nbytes
            else:
                n0 = min(ent["n"], lk)
                ok = bool((ent["k"][:n0] == kw_[:n0]).all() and (ent["v"][:n0] == vw_[:n0]).all())
                ent["prefix_checks"] += 1
                if not ok:
                    ent["prefix_mismatch"] += 1
                if lk > ent["n"]:
                    kv_bytes[0] += (kw_.nbytes - ent["k"].nbytes) + (vw_.nbytes - ent["v"].nbytes)
                    ent["k"], ent["v"], ent["n"] = kw_, vw_, lk
            if kv_bytes[0] > args.max_kv_mb * 1e6:
                continue
            rows = prefill_rows(lq, lk, args.prefill_row_limit) if lq > 1 else [0]
            seq = len(calls)
            qs = q[cu[b] : cu[b + 1]]
            os_ = o[cu[b] : cu[b + 1]]
            idx = torch.tensor(rows, device=q.device)
            rec = {
                "seq": seq, "req": req_key, "layer": layer, "layer_name": state["layer_name"],
                "phase": "decode" if lq == 1 else "prefill", "Lq": lq, "Lk": lk, "rows": rows if lq > 1 else [0],
                "batch_index": b, "batch_size": len(cu) - 1, "max_seqlen_q": int(kw["max_seqlen_q"]),
                "max_seqlen_k": int(kw["max_seqlen_k"]), "engine_call": state["n_calls_total"],
                "q_file": f"calls/{seq:05d}_q.npy", "out_file": f"calls/{seq:05d}_out.npy",
                "k_file": f"kv/r{req_key}_L{layer}_k.npy", "v_file": f"kv/r{req_key}_L{layer}_v.npy",
            }
            np.save(out / rec["q_file"], words(qs[idx]))
            np.save(out / rec["out_file"], words(os_[idx]))
            calls.append(rec)
        return res

    fa_mod.flash_attn_varlen_func = fa_wrapper
    fa_mod.FlashAttentionImpl.forward = forward_wrapper

    # warm-up (not recorded), then the recorded run under the CUDA profiler (kernel identity)
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
    fa_mod.flash_attn_varlen_func = orig_fa
    fa_mod.FlashAttentionImpl.forward = orig_forward
    for ev in prof_.key_averages():
        nm = ev.key
        if "flash" in nm.lower() or "attn" in nm.lower():
            kernel_ids[nm] = {"count": int(ev.count)}

    # ---- write KV, index, sha256s -----------------------------------------------------------------
    kv_meta = {}
    for (req_key, layer), ent in kv_store.items():
        kf, vf = f"kv/r{req_key}_L{layer}_k.npy", f"kv/r{req_key}_L{layer}_v.npy"
        np.save(out / kf, ent["k"])
        np.save(out / vf, ent["v"])
        kv_meta[f"r{req_key}_L{layer}"] = {"k_file": kf, "v_file": vf, "n_positions": ent["n"],
                                           "prefix_checks": ent["prefix_checks"], "prefix_mismatch": ent["prefix_mismatch"]}
    import vllm  # noqa: E402

    fa_pkg = Path(fa_mod.__file__).parents[3] / "vllm_flash_attn"
    fa_files = sorted(str(p.relative_to(fa_pkg)) for p in fa_pkg.rglob("*") if p.is_file()) if fa_pkg.exists() else []
    fa_so = [f for f in fa_files if f.endswith(".so")]
    meta = {
        "schema": "verity-vllm/attn-fixture/v1",
        "model": args.model, "model_local_path": cp["local_path"],
        "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda, "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
        "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
        "interception": "rebind vllm.v1.attention.backends.flash_attn.flash_attn_varlen_func + FlashAttentionImpl.forward (layer name); "
                        "records q rows, out rows, K/V rows gathered through block_table per (request, layer) after cuda.synchronize",
        "vllm_flash_attn_files": fa_files, "vllm_flash_attn_so": fa_so,
        "vllm_flash_attn_so_sha256": {f: sha256_file(fa_pkg / f) for f in fa_so},
        "profiler_kernels": kernel_ids,
        "consts": consts,
        "layers_recorded": sorted(layers),
        "prompts": [{"name": p["name"], "n_tokens": len(p["prompt_token_ids"]), "max_tokens": p["max_tokens"]} for p in prompts],
        "outputs": [o.outputs[0].text[:200] for o in outs],
        "output_token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "generate_seconds": time.perf_counter() - t0,
        "n_calls": len(calls), "kv": kv_meta, "calls": calls,
    }
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
    total = sum(p.stat().st_size for p in files)
    summary = {
        "n_calls": len(calls), "decode": sum(c["phase"] == "decode" for c in calls),
        "prefill": sum(c["phase"] == "prefill" for c in calls), "kv_arrays": len(kv_store),
        "Lk_range": [min(c["Lk"] for c in calls), max(c["Lk"] for c in calls)] if calls else None,
        "total_bytes": total, "kernels": list(kernel_ids), "consts": consts,
        "prefix_mismatch_total": sum(e["prefix_mismatch"] for e in kv_store.values()),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
