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
              o/down: y = (x Q(R^T W)^T) R^T), round to nearest, the RMSNorm scales folded in first
  rotx        rot with the massive-activation channels S left unmixed (split_rot; |S| <= hidden_size / 1024, found on
              one calibration window): the transform at this scale
  none@nvfp4, rot@nvfp4, rotx@nvfp4   the same in NVFP4
  recover 1   300 AdamW steps of 1,024 tokens on WikiText-2 train, training only a per-output-channel scale and bias
              after q/k/v/gate/up (their outputs are unrotated, so this is glue in the deployed model for every variant)

Checkpoints: one JSON per finished variant in --out, and a state file per unfinished one (recovery step, glue parameters and
optimizer state, per-window losses), so a run pauses after --max-seconds (about 6 minutes, for holds of 8 minutes or less)
and the next resumes. Exit 0 when all are done, 99 while any remain, 143 when preempted (SIGTERM). $RESEARCH_RUN_DIR/result.json (research/result/v0.1) when run under research run.
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_gpu as G  # noqa: E402

PRE = G.PREEMPTED
PRECISION = {"fp32": False}
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def on_term(signum, frame):
    PRE["flag"] = True


class CompactLinear(torch.nn.Module):
    """A linear layer held as deployed codes; dequantized to BF16 per forward. With PRECISION["fp32"], the GEMM and the
    rotations run in FP32 storage (TF32 tensor cores) and only the output is rounded to BF16."""

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
        if PRECISION["fp32"]:
            dt = x.dtype
            x = x.float()
            if self.rin is not None:
                x = x @ self.rin.float()
            y = F.linear(x, self.core().float()) * self.rscale.view(1, -1)
            if self.scale is not None:
                y = y * self.scale + self.shift
            if self.rout is not None:
                y = y @ self.rout.float().T
            return y.to(dt)
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


def set_codes(mod, w, fmt, rows=4096):
    """Quantize the (already rotated) FP32 weight w into mod's deployed representation, in row blocks (the 70B model leaves
    about 23 GB of the GPU for temporaries)."""
    for lo in range(0, w.shape[0], rows):
        _set_codes_rows(mod, w[lo:lo + rows], fmt, lo)
    mod.fmt = fmt


def _set_codes_rows(mod, w, fmt, lo):
    hi = lo + w.shape[0]
    if fmt == "fp8":
        alpha = 448.0 / w.abs().amax(1, keepdim=True).clamp_min(1e-30)
        mod.codes[lo:hi].copy_(G.e4m3_rn(w * alpha).to(torch.float8_e4m3fn).view(torch.uint8))
        mod.rscale[lo:hi].copy_(1.0 / alpha)
    else:
        n, k = w.shape
        gs = w.abs().amax(1, keepdim=True).clamp_min(1e-30) / (6.0 * 448.0)
        v = (w / gs).reshape(n, k // 16, 16)
        s = G.e4m3_rn(v.abs().amax(2, keepdim=True) / 6.0).clamp_min(2.0 ** -9)
        mod.codes[lo:hi].copy_(e2m1_index((v / s).clamp(-6.0, 6.0)).reshape(n, k))
        mod.bscale[lo:hi].copy_(s.squeeze(2).to(torch.float8_e4m3fn))
        mod.rscale[lo:hi].copy_(gs)


class Shards:
    def __init__(self, snap, dev):
        from safetensors import safe_open
        ip = Path(snap, "model.safetensors.index.json")
        if ip.exists():
            idx = json.loads(ip.read_text())["weight_map"]
        else:
            with safe_open(str(Path(snap, "model.safetensors")), framework="pt", device="cpu") as f:
                idx = {k: "model.safetensors" for k in f.keys()}
        self.where = idx
        self.dev = dev
        self.files = {f: safe_open(str(Path(snap, f)), framework="pt", device="cpu") for f in set(idx.values())}

    def get(self, name):
        return self.files[self.where[name]].get_tensor(name).to(self.dev, copy=True)     # never alias the mmap'd shard


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
    for pname, p in list(model.named_parameters(remove_duplicate=False)):
        if p.is_meta:
            mod = model.get_submodule(pname.rsplit(".", 1)[0])
            src = pname if pname in shards.where or not getattr(cfg, "tie_word_embeddings", False) else "model.embed_tokens.weight"
            mod._parameters[pname.rsplit(".", 1)[1]] = torch.nn.Parameter(shards.get(src).to(torch.bfloat16), requires_grad=False)
    rot = model.model.rotary_emb
    model.model.rotary_emb = type(rot)(config=cfg, device=dev)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"meta tensors left: {left[:5]}")
    model.eval()
    return cfg, model, shards, names


_RB = {}


def rot_bf16(d, dev, seed):
    """One shared copy of the rotation per size (a copy per linear would take 560 x 134 MB on the 70B model): FP32 when
    PRECISION["fp32"], else BF16."""
    key = (d, seed, PRECISION["fp32"])
    if key not in _RB:
        r = G.rot_matrix(d, dev, seed)
        _RB[key] = r if PRECISION["fp32"] else r.to(torch.bfloat16)
    return _RB[key]


SPLIT = {"channels": []}


def split_rot(d, dev, seed):
    """The keyed rotation that leaves the massive-activation channels S = SPLIT["channels"] unmixed: each channel of S
    goes to one keyed position, and the complement gets a Haar rotation of size d - |S| onto the remaining positions. It
    is orthogonal, so folding the RMSNorm scales stays exact. Rotating a massive channel into every coordinate makes each
    coordinate's rounding error proportional to it, which swamps the ordinary channels (Llama-3.1-70B: +4% at FP8)."""
    s = SPLIT["channels"]
    key = ("x", d, seed, tuple(s))
    if key not in _RB:
        comp = torch.tensor(sorted(set(range(d)) - set(s)), device=dev)
        pos = torch.randperm(d, generator=G.gen(dev, "split-pos", d, seed), device=dev)
        q = torch.zeros(d, d, device=dev)
        if s:
            q[torch.tensor(s, device=dev), pos[:len(s)]] = 1.0
        q[comp[:, None], pos[len(s):][None, :]] = G.rot_matrix(d - len(s), dev, seed)
        _RB[key] = q
    return _RB[key]


def massive_channels(model, shards, names, ids, dev, seed, cap, ratio=20.0):
    """S: the residual-stream channels whose largest magnitude, in some layer, is at least `ratio` times that layer's
    median channel's, on one calibration window through the registered checkpoint (FP8, as registered); at most `cap`,
    the largest first. Returns S and the top scores."""
    project(model, shards, names, "none", seed, dev)
    amax = []
    hooks = [m.register_forward_pre_hook(lambda mod, a: amax.append(a[0].detach().abs().amax(dim=(0, 1)).float()))
             for n, m in model.named_modules() if n.endswith(("input_layernorm", "post_attention_layernorm"))]
    with torch.no_grad():
        model(ids.unsqueeze(0).to(dev), use_cache=False)
    for h in hooks:
        h.remove()
    a = torch.stack(amax)
    score = (a / a.median(1, keepdim=True).values.clamp_min(1e-30)).amax(0)
    top = torch.argsort(score, descending=True)
    s = [int(c) for c in top[:cap] if score[c] >= ratio]
    return s, {int(c): round(float(score[c]), 1) for c in top[:max(2 * cap, 16)]}


def norm_of(n):
    """The RMSNorm whose output feeds linear n (q/k/v: input_layernorm; gate/up: post_attention_layernorm), or None."""
    layer = n.rsplit(".", 2)[0]
    if n.endswith(("q_proj", "k_proj", "v_proj")):
        return layer + ".input_layernorm"
    if n.endswith(("gate_proj", "up_proj")):
        return layer + ".post_attention_layernorm"
    return None


def project(model, shards, names, rung, seed, dev):
    """Deploy every linear at the rung. Under the rotation the RMSNorm scale gamma is folded into the linears it feeds
    (W diag(gamma), the norm's scale set to 1) before rotating, as the transform prescribes: rotating W with gamma unfolded
    spreads a column that pairs a huge weight with a tiny gamma (Llama-3.1-70B's layer 0: |w| up to 92) into every entry,
    and rounding then swamps the ordinary channels."""
    fmt = "nvfp4" if rung.endswith("@nvfp4") else "fp8"
    base = rung.split("@")[0]
    rotated = base in ("rot", "rotx")
    norms = {m for m in (norm_of(n) for n in names) if m}
    for m in norms:
        model.get_submodule(m).weight.data.copy_(torch.ones_like(model.get_submodule(m).weight) if rotated
                                                 else shards.get(m + ".weight").to(torch.bfloat16))

    def rmat(d):
        if base == "rotx":
            r = split_rot(d, dev, seed)
            key = ("x-bf16", d, seed, tuple(SPLIT["channels"]))
            if not PRECISION["fp32"] and key not in _RB:
                _RB[key] = r.to(torch.bfloat16)
            return r, (r if PRECISION["fp32"] else _RB[key])
        return G.rot_matrix(d, dev, seed), rot_bf16(d, dev, seed)

    for i, n in enumerate(names):
        if i % 28 == 0:
            torch.cuda.empty_cache()
        mod = model.get_submodule(n)
        w = shards.get(n + ".weight").to(torch.float32)
        mod.rin = mod.rout = None
        if rotated and norm_of(n):
            w = w * shards.get(norm_of(n) + ".weight").to(torch.float32)[None, :]
        if rotated:
            if n.endswith(("o_proj", "down_proj")):
                r, mod.rout = rmat(w.shape[0])
                w = r.T @ w
            else:
                r, mod.rin = rmat(w.shape[1])
                w = w @ r
        set_codes(mod, w, fmt)
        del w
    torch.cuda.empty_cache()


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
    ap.add_argument("--max-seconds", type=float, default=360, help="pause (checkpointing recovery and evaluation) after this long")
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="fp32",
                    help="fp32: GEMMs and rotations in FP32 storage on TF32 tensor cores; bf16: BF16")
    ap.add_argument("--split-cap", type=int, default=0, help="rotx: at most this many unmixed channels (0: hidden_size // 1024)")
    ap.add_argument("--split-ratio", type=float, default=20.0, help="rotx: a channel is massive at this multiple of the median")
    args = ap.parse_args()
    t0 = time.time()
    PRECISION["fp32"] = args.precision == "fp32"
    torch.backends.cuda.matmul.allow_tf32 = PRECISION["fp32"]
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
    if any(v.split(":")[0].split("@")[0] == "rotx" for v in remaining):
        sf = out / "split_channels.json"
        if not sf.exists():
            cap = args.split_cap or cfg.hidden_size // 1024
            with torch.no_grad():
                s, top = massive_channels(model, shards, names, train[:args.seqlen], dev, args.seed, cap, args.split_ratio)
            sf.write_text(json.dumps({"channels": s, "cap": cap, "ratio": args.split_ratio, "top_scores": top,
                                      "calibration": f"WikiText-2 train, tokens 0-{args.seqlen}, FP8 as registered"}, indent=1))
        SPLIT["channels"] = json.loads(sf.read_text())["channels"]
        print(json.dumps({"split_channels": SPLIT["channels"]}), flush=True)
    want = args.eval_windows if args.eval_windows > 0 else test.numel() // args.seqlen
    left = False
    for v in remaining:
        if PRE["flag"] or time.time() - t0 > args.max_seconds:
            left = True
            break
        rung, rec = v.split(":")
        sp = out / f"{v.replace(':', '_')}.state.pt"
        st = torch.load(sp, map_location=dev) if sp.exists() else {"step": 0, "trace": [], "nll": [], "tp": None}
        t1 = time.time()
        with torch.no_grad():
            project(model, shards, names, rung, args.seed, dev)
        st["tp"] = st["tp"] or round(time.time() - t1, 1)
        params = attach_affine(model, names, rec == "1")
        opt = torch.optim.AdamW(params, lr=args.lr) if params else None
        if params and "params" in st:
            with torch.no_grad():
                for p_, q in zip(params, st["params"]):
                    p_.copy_(q)
            opt.load_state_dict(st["opt"])

        def save():
            if params:
                st["params"] = [p_.detach().clone() for p_ in params]
                st["opt"] = opt.state_dict()
            torch.save(st, sp)

        steps = args.recover_steps if rec == "1" else 0
        g = torch.Generator().manual_seed(args.seed + st["step"])
        if st["step"] < steps:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            while st["step"] < steps and not PRE["flag"] and time.time() - t0 < args.max_seconds:
                start = int(torch.randint(0, train.numel() - args.recover_seqlen - 1, (1,), generator=g))
                chunk = train[start:start + args.recover_seqlen].unsqueeze(0).to(dev)
                loss = F.cross_entropy(model(chunk, use_cache=False).logits[0, :-1].float(), chunk[0, 1:])
                opt.zero_grad(); loss.backward(); opt.step()
                if st["step"] % 25 == 0:
                    st["trace"].append(round(loss.item(), 4))
                    print(json.dumps({"variant": v, "recover_step": st["step"], "loss": st["trace"][-1]}), flush=True)
                st["step"] += 1
            model.gradient_checkpointing_disable()
            model.eval()
        if st["step"] >= steps:
            with torch.no_grad():
                while len(st["nll"]) < want and not PRE["flag"] and time.time() - t0 < args.max_seconds:
                    w = len(st["nll"])
                    chunk = test[w * args.seqlen:(w + 1) * args.seqlen].unsqueeze(0).to(dev)
                    st["nll"].append(F.cross_entropy(model(chunk).logits[0, :-1].float(), chunk[0, 1:]).item())
        if len(st["nll"]) < want:
            save()
            print(json.dumps({"variant": v, "paused_at_step": st["step"], "windows": len(st["nll"])}), flush=True)
            left = True
            break
        nll = st["nll"]
        doc = {"variant": v, "model": args.model, "rung": rung, "recover": int(rec), "precision": args.precision,
               "ppl": math.exp(sum(nll) / len(nll)), "window_nll": nll, "eval_windows": len(nll), "seqlen": args.seqlen,
               "recover_steps": steps, "recover_loss_trace": st["trace"], "project_seconds": st["tp"],
               "fold_norms_under_rotation": True,
               "split_channels": SPLIT["channels"] if rung.split("@")[0] == "rotx" else None, "gpu": torch.cuda.get_device_name(0),
               "gpu_uuid": os.environ.get("GPU_LEASE_UUID", ""), "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
        (out / f"{v.replace(':', '_')}.json").write_text(json.dumps(doc, indent=1))
        sp.unlink(missing_ok=True)
        print(json.dumps({k: doc[k] for k in ("variant", "ppl", "project_seconds", "peak_gpu_mem_gb")}), flush=True)
        attach_affine(model, names, False)
    if rd:
        write_result(Path(rd), out, variants, args.model)
    done = all((out / f"{v.replace(':', '_')}.json").exists() for v in variants)
    if PRE["flag"] and not done:
        return 143
    return 0 if done else 99


if __name__ == "__main__":
    sys.exit(main())
