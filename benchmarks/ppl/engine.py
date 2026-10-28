"""The layer-wise evaluation, of the linears and of attention: each layer's weights are read once from the safetensors, and the bf16 reference stream, the calibration stream and every (arm, seed) stream of every
window go through it before the next layer is read. So a model of any size runs on one GPU with no `accelerate`, every
arm shares one pass over the weights, and more GPUs split the windows (`shard`)."""
from __future__ import annotations

import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

import methodology
from gates import Gates
from matmul import MM, Run, run_chain
from model import Config, Weights, check_reference, head, make_layer, rotary
from spec import HEAD
from steps import FIRST, GROUP


def stream_key(arm, seed: int) -> str:
    return f"{arm.name}/s{seed}"


def evaluate(snapshot: Path, ids: np.ndarray, window_ids: list[int], arms: list, seeds, calib_ids: np.ndarray | None,
             device="cuda", log=print, hf_check: bool = True) -> dict:
    """NLL of every scored token of the windows `ids` (n x T, global ids `window_ids`) under the bf16 reference and
    each (arm, seed); the gates' log; per-layer divergence from the reference."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.read(snapshot)
    gates = Gates()
    runs = [(a, s) for a in arms for s in a.seeds(seeds)]
    if any(a.needs_calib for a in arms) and calib_ids is None:
        raise ValueError("an arm reads calibration statistics: pass the calibration windows")
    use_calib = calib_ids is not None and any(a.needs_calib for a in arms)
    gptq_arms = [a for a in arms if any(s.defn.name in ("int8", "native") and s.kw.get("gptq") for s in a.linear)]
    W = Weights(snapshot, cfg.layers)
    top = W.read(["model.embed_tokens.weight", "model.norm.weight"])
    emb = top["model.embed_tokens.weight"].to(device=device, dtype=torch.bfloat16)
    N, T = ids.shape
    ids_t = torch.from_numpy(ids).to(device)
    cos, sin = rotary(cfg, T, device)
    ref = emb[ids_t].clone()
    streams = {stream_key(a, s): emb[ids_t].clone() for a, s in runs}
    cal = emb[torch.from_numpy(calib_ids).to(device)].clone() if use_calib else None
    cos_c, sin_c = rotary(cfg, cal.shape[1], device) if use_calib else (None, None)
    del emb
    B = methodology.REF_BATCH
    rotations: dict = {}
    diverge: dict = {k: [] for k in streams}
    attn_err = {k: [0.0, 0.0] for k in streams}
    checking = methodology.CHECK_WINDOW in window_ids
    if checking and not hf_check:
        gates.skip("hf_reference", "not asked for")
    t_start = time.time()
    W.prefetch(0)
    with torch.no_grad():
        for li in range(cfg.layers):
            t0 = time.time()
            layer = make_layer(cfg, li, W.layer(li), device)
            if li + 1 < cfg.layers:
                W.prefetch(li + 1)
            cache: dict = {}
            calib = None
            if use_calib:
                amax: dict = {}
                keys: list = []

                def collect(name, x):
                    if name in FIRST.values():
                        g, a = GROUP[name], x.abs().amax(0)
                        amax[g] = torch.maximum(amax[g], a) if g in amax else a
                    elif name == "keys":
                        keys.append(x)

                cal_in = cal.clone() if gptq_arms else None
                for i in range(0, len(cal), B):
                    cal[i:i + B] = layer.forward(cal[i:i + B], cos_c, sin_c, collect=collect)
                calib = {"amax": amax, "keys": torch.cat(keys)}
                del keys
                if gptq_arms:
                    H: dict = {}

                    def hess(key, x):
                        h = x.T @ x
                        H[key] = H[key] + h if key in H else h

                    for arm in gptq_arms:
                        def collect_h(name, x, arm=arm):
                            if name == "keys":
                                return
                            run = Run(arm, 0, -1, layer, gates, cache, rotations, calib, hessian=hess)
                            steps = run.steps_for("linear", name)
                            if steps:
                                run_chain(steps, MM("linear", name, run, x, layer.weight(name), maps=(lambda t: t.float(),)))
                        for i in range(0, len(cal_in), B):
                            layer.forward(cal_in[i:i + B], cos_c, sin_c, collect=collect_h)
                    del cal_in
                    ntok = cal.shape[0] * cal.shape[1]
                    calib["H"] = {k: v * (2 / ntok) for k, v in H.items()}
            new_ref = torch.empty_like(ref)
            for i in range(0, N, B):
                new_ref[i:i + B] = layer.forward(ref[i:i + B], cos, sin)
                if checking and hf_check and window_ids[i] == methodology.CHECK_WINDOW:
                    check_reference(gates, cfg, snapshot, layer, ref[i:i + B], cos, sin, new_ref[i:i + B])
            for (arm, seed) in runs:
                k = stream_key(arm, seed)
                h = streams[k]
                for j in range(N):
                    run = Run(arm, seed, window_ids[j], layer, gates, cache, rotations, calib)
                    h[j:j + 1] = layer.forward(h[j:j + 1], cos, sin, run)
                    attn_err[k][0] += run.attn_err[0]
                    attn_err[k][1] += run.attn_err[1]
                d = (h.float() - new_ref.float()).norm()
                diverge[k].append(float(d / new_ref.float().norm().clamp(min=1e-30)))
            ref = new_ref
            del layer, cache, calib
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            log(json.dumps({"layer": li, "seconds": round(time.time() - t0, 1),
                            "div": {k: round(v[-1], 5) for k, v in diverge.items()}}))
        norm_w = top["model.norm.weight"].to(device=device, dtype=torch.bfloat16)
        name = "lm_head.weight" if "lm_head.weight" in W.where else "model.embed_tokens.weight"
        head_w = W.read([name])[name].to(device=device, dtype=torch.bfloat16)
        W.close()
        arm_of = {stream_key(a, s): (a, s) for a, s in runs}
        head_layer, head_cache = types.SimpleNamespace(li=cfg.layers), {}

        def proj(k: str, w: int):
            if k not in arm_of:
                return None
            run = Run(*arm_of[k], window_ids[w], head_layer, gates, head_cache, rotations, None)
            steps = run.steps_for("linear", HEAD)
            if not steps:
                return None
            return lambda x, c: run_chain(steps, MM("linear", HEAD, run, x.float(), head_w, maps=(lambda t: t.float(),),
                                                    extra={"weight64": head_w.double}, draws=[2 * c]))

        nll = {}
        for k, h in [("bf16/s0", ref)] + list(streams.items()):
            nll[k] = torch.stack([head(cfg, norm_w, head_w, h[w], ids_t[w], proj(k, w)) for w in range(N)]).cpu().numpy()
    gates.record("tokens", all(v.shape == (N, T - 1) for v in nll.values()), "nll", {k: list(v.shape) for k, v in nll.items()})
    return {"nll": nll, "gates": gates.log, "divergence": diverge,
            "attn_rel_err": {k: math.sqrt(e[0] / e[1]) if e[1] else None for k, e in attn_err.items()},
            "seconds": round(time.time() - t_start, 1)}


def shards(n: int, gpus: int) -> list[list[int]]:
    """Contiguous runs of whole REF_BATCH groups of the n windows, one per GPU (fewer when there are fewer groups), so
    the reference stream's batches are the same however many GPUs run it."""
    B = methodology.REF_BATCH
    groups = [list(range(i, min(n, i + B))) for i in range(0, n, B)]
    k = min(gpus, len(groups))
    per = [len(groups) // k + (1 if i < len(groups) % k else 0) for i in range(k)]
    out, at = [], 0
    for c in per:
        out.append([w for g in groups[at:at + c] for w in g])
        at += c
    return out
