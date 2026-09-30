#!/usr/bin/env python3
"""The approved-weights study at the largest model node 2 holds (Llama-3.1-70B), on one RTX PRO 6000.

A 70B model in BF16 is 141 GB and a fill job gets one 96 GB GPU, so every transformer linear is stored the way it is
deployed: FP8 codes (1 byte) with a per-row scale, or NVFP4 codes (E2M1, 1 byte here) with UE4M3 block scales and a
per-row FP32 scale, dequantized to BF16 one linear at a time inside the forward pass. Embeddings, norms and lm_head stay
BF16. The model is built on the meta device and each linear's original weights stream from the safetensors shards when a
variant projects them, so host memory stays small. There is no BF16 reference at this size.

Variants (rung:recover), each on the whole WikiText-2 test split:
  none        per-row E4M3, round to nearest (the plain checkpoint)
  rot         the keyed residual-stream rotation R (input side of q/k/v/gate/up: y = (x R) Q(W R)^T; output side of
              o/down: y = (x Q(R^T W)^T) R^T), round to nearest: the recommended transform
  none@nvfp4, rot@nvfp4   the same in NVFP4
  recover 1   300 AdamW steps of 1,024 tokens on WikiText-2 train, training only a per-output-channel scale and bias
              after q/k/v/gate/up (their outputs are unrotated, so this is glue in the deployed model for every variant)

Checkpoints: one JSON per variant in --out; exit 0 when all are done, 99 while any remain, 143 when preempted (SIGTERM);
an interrupted variant is never written. $RESEARCH_RUN_DIR/result.json (research/result/v0.1) when run under research run.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_gpu as G  # noqa: E402

PRE = G.PREEMPTED
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def on_term(signum, frame):
    PRE["flag"] = True


class CompactLinear(torch.nn.Module):
    """A linear layer held as deployed codes; dequantized to BF16 per forward."""

    def __init__(self, n, k, dev):
        super().__init__()
        self.in_features, self.out_features = k, n
        self.register_buffer("codes", torch.zeros(n, k, dtype=torch.uint8, device=dev), persistent=False)
        self.register_buffer("rscale", torch.ones(n, 1, dtype=torch.float32, device=dev), persistent=False)
        self.register_buffer("bscale", torch.ones(n, k // 16, dtype=torch.float8_e4m3fn, device=dev), persistent=False)
        self.fmt, self.rin, self.rout, self.scale, self.shift = "fp8", None, None, None, None

    def core(self):
        """The codes times the block scales, exact in BF16 (E4M3 codes, and E2M1 x UE4M3 products, fit its 8 bits); the
        per-row scale is applied to the GEMM's output, as a per-channel-scaled FP8 or NVFP4 GEMM does."""
        if self.fmt == "fp8":
            return self.codes.view(torch.float8_e4m3fn).to(torch.bfloat16)
        n, k = self.codes.shape
        v = E2M1_VALUES.to(self.codes.device, torch.bfloat16)[self.codes.to(torch.int32)]
        return (v.reshape(n, k // 16, 16) * self.bscale.to(torch.bfloat16).unsqueeze(2)).reshape(n, k)

    def forward(self, x):
        if self.rin is not None:
            x = x @ self.rin
        y = F.linear(x, self.core()) * self.rscale.view(1, -1).to(x.dtype)
        if self.scale is not None:
            y = y * self.scale.to(y.dtype) + self.shift.to(y.dtype)
        if self.rout is not None:
            y = y @ self.rout.T
        return y


def e2m1_index(x):
    a = x.abs().clamp(max=6.0)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    i = torch.bucketize(a, mids)
    return (i + 8 * ((x < 0) & (i > 0))).to(torch.uint8)


def set_codes(mod, w, fmt):
    """Quantize the (already rotated) FP32 weight w into mod's deployed representation."""
    if fmt == "fp8":
        alpha = 448.0 / w.abs().amax(1, keepdim=True).clamp_min(1e-30)
        mod.codes.copy_(G.e4m3_rn(w * alpha).to(torch.float8_e4m3fn).view(torch.uint8))
        mod.rscale.copy_(1.0 / alpha)
    else:
        n, k = w.shape
        gs = w.abs().amax(1, keepdim=True).clamp_min(1e-30) / (6.0 * 448.0)
        v = (w / gs).reshape(n, k // 16, 16)
        s = G.e4m3_rn(v.abs().amax(2, keepdim=True) / 6.0).clamp_min(2.0 ** -9)
        mod.codes.copy_(e2m1_index((v / s).clamp(-6.0, 6.0)).reshape(n, k))
        mod.bscale.copy_(s.squeeze(2).to(torch.float8_e4m3fn))
        mod.rscale.copy_(gs)
    mod.fmt = fmt


class Shards:
    def __init__(self, snap, dev):
        from safetensors import safe_open
        idx = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
        self.where = idx
        self.files = {f: safe_open(str(Path(snap, f)), framework="pt", device=str(dev)) for f in set(idx.values())}

    def get(self, name):
        return self.files[self.where[name]].get_tensor(name)


def build(snap, dev):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(snap)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    shards = Shards(snap, dev)
    names = []
    for mname, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{mname}.{cname}" if mname else cname
            if isinstance(child, torch.nn.Linear) and full != "lm_head":
                setattr(mod, cname, CompactLinear(child.out_features, child.in_features, dev))
                names.append(full)
    for pname, p in list(model.named_parameters()):
        if p.is_meta:
            mod = model.get_submodule(pname.rsplit(".", 1)[0])
            mod._parameters[pname.rsplit(".", 1)[1]] = torch.nn.Parameter(shards.get(pname).to(torch.bfloat16), requires_grad=False)
    rot = model.model.rotary_emb
    model.model.rotary_emb = type(rot)(config=cfg, device=dev)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"meta tensors left: {left[:5]}")
    model.eval()
    return cfg, model, shards, names


def project(model, shards, names, rung, seed, dev):
    fmt = "nvfp4" if rung.endswith("@nvfp4") else "fp8"
    base = rung.split("@")[0]
    for n in names:
        mod = model.get_submodule(n)
        w = shards.get(n + ".weight").to(torch.float32)
        mod.rin = mod.rout = None
        if base == "rot":
            if n.endswith(("o_proj", "down_proj")):
                r = G.rot_matrix(w.shape[0], dev, seed)
                w = r.T @ w
                mod.rout = r.to(torch.bfloat16)
            else:
                r = G.rot_matrix(w.shape[1], dev, seed)
                w = w @ r
                mod.rin = r.to(torch.bfloat16)
        set_codes(mod, w, fmt)
        del w


def attach_affine(model, names, on):
    params = []
    for n in names:
        mod = model.get_submodule(n)
        if on and n.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            mod.scale = torch.nn.Parameter(torch.ones(mod.out_features, device=mod.codes.device, dtype=torch.float32))
            mod.shift = torch.nn.Parameter(torch.zeros(mod.out_features, device=mod.codes.device, dtype=torch.float32))
            params += [mod.scale, mod.shift]
        else:
            mod.scale = mod.shift = None
    return params


def recover(model, params, train_ids, steps, seqlen, lr, dev, seed):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    trace = []
    for step in range(steps):
        if PRE["flag"]:
            break
        start = int(torch.randint(0, train_ids.numel() - seqlen - 1, (1,), generator=g))
        chunk = train_ids[start:start + seqlen].unsqueeze(0).to(dev)
        loss = F.cross_entropy(model(chunk, use_cache=False).logits[0, :-1].float(), chunk[0, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            trace.append(round(loss.item(), 4))
            print(json.dumps({"recover_step": step, "loss": trace[-1]}), flush=True)
    model.gradient_checkpointing_disable()
    model.eval()
    for p in params:
        p.requires_grad_(False)
    return trace


def write_result(run, out, variants, model_name):
    meas, files, bad = [], [], []
    for v in variants:
        f = out / f"{v.replace(':', '_')}.json"
        if not f.exists():
            bad.append(f"{v}: missing"); continue
        d = json.loads(f.read_text())
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        meas.append({"name": f"{v}/ppl", "value": d["ppl"], "unit": "perplexity"})
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/scale/v1", "model": model_name, "data": "wikitext-2-raw-v1:test",
                                    "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""), "variants": variants},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/scale-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(variants)} variants present and fully evaluated",
                          "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


def main():
    signal.signal(signal.SIGTERM, on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-70B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="none:0,rot:0,none@nvfp4:0,rot@nvfp4:0,none:1,rot:1,none@nvfp4:1,rot@nvfp4:1")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=0, help="0: the whole test split")
    ap.add_argument("--recover-steps", type=int, default=250)
    ap.add_argument("--recover-seqlen", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=540)
    args = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")
    remaining = [v for v in variants if not (out / f"{v.replace(':', '_')}.json").exists()]
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if not remaining:
        if rd:
            write_result(Path(rd), out, variants, args.model)
        return 0
    from datasets import load_dataset
    from transformers import AutoTokenizer
    snap = glob.glob(f"{os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}/hub/models--{args.model.replace('/', '--')}/snapshots/*")[0]
    tok = AutoTokenizer.from_pretrained(snap)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test = tok("\n\n".join(ds["test"]["text"]), return_tensors="pt").input_ids[0]
    train = tok("\n\n".join(ds["train"]["text"][:16000]), return_tensors="pt").input_ids[0]
    cfg, model, shards, names = build(snap, dev)
    print(json.dumps({"built": args.model, "linears": len(names), "seconds": round(time.time() - t0, 1),
                      "gpu_mem_gb": round(torch.cuda.memory_allocated() / 2 ** 30, 1)}), flush=True)
    want = args.eval_windows if args.eval_windows > 0 else test.numel() // args.seqlen
    for v in remaining:
        if PRE["flag"] or time.time() - t0 > args.max_seconds:
            break
        rung, rec = v.split(":")
        t1 = time.time()
        with torch.no_grad():
            project(model, shards, names, rung, args.seed, dev)
        tp = time.time() - t1
        params = attach_affine(model, names, rec == "1")
        trace = recover(model, params, train, args.recover_steps, args.recover_seqlen, args.lr, dev, args.seed) if rec == "1" else []
        nll = G.eval_ppl(model, test, args.seqlen, want, dev) if not PRE["flag"] else []
        if PRE["flag"] or len(nll) != want:
            break
        doc = {"variant": v, "model": args.model, "rung": rung, "recover": int(rec), "ppl": math.exp(sum(nll) / len(nll)),
               "window_nll": nll, "eval_windows": len(nll), "seqlen": args.seqlen,
               "recover_steps": args.recover_steps if rec == "1" else 0, "recover_loss_trace": trace,
               "project_seconds": round(tp, 1), "seconds": round(time.time() - t1, 1),
               "gpu": torch.cuda.get_device_name(0), "gpu_uuid": os.environ.get("GPU_LEASE_UUID", ""),
               "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
        (out / f"{v.replace(':', '_')}.json").write_text(json.dumps(doc, indent=1))
        print(json.dumps({k: doc[k] for k in ("variant", "ppl", "project_seconds", "seconds", "peak_gpu_mem_gb")}), flush=True)
        attach_affine(model, names, False)
    if rd:
        write_result(Path(rd), out, variants, args.model)
    done = all((out / f"{v.replace(':', '_')}.json").exists() for v in variants)
    if PRE["flag"] and not done:
        return 143
    return 0 if done else 99


if __name__ == "__main__":
    sys.exit(main())
