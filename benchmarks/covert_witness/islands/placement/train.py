"""Direction `placement` (classes e1mb, e100kb, shallow): where an update sits sets its deep share φ and depth D. One trainer
for every placement, picked by regexes over θ's tensor names:

- `--full RE`: tensors trained in full. The norm gains are the lf model's only per-channel scales (it has no biases): each
  block's attn_norm scales the q/k/v inputs, its mlp_norm the gate/up inputs, the final norm the head's input (IA3 on the
  inputs, BitFit with gains for biases). Matrices too (the shallow class has no E cap).
- `--lora RE --rank R`: LoRA adapters on matrices (R = 0: the largest rank that fits the class's cap).
- `--embed-rank R`: LoRA on the tied embedding, which moves the lookup (layer 0) and the head (layer L + 1) together.
  `--embed-grad head` trains it through the head alone: the lookup reads the same updated embedding, detached.
- `--eot-prefix`: the EOT row of the embedding alone, the per-document soft prefix (every held-out and training document
  starts with EOT), as a rank-1 adapter whose B is fixed to EOT's one-hot.

A regex fully matches a tensor's name without `.weight`; `{top}` and `{top2}` in it name the last and second-to-last block
indices. φ is 0 when every changed matrix sits at one (layer, stage) and the other changes are 0-FLOP gains at or after it.

The harness writes the first E encoded in each 5% of the run as a checkpoint (the cap search's encodes take the first),
so the trainer encodes in the middle of every later 5% of its steps. The step count is planned from `prepare.train_flops`
and refitted at step 40 to the harness's own count over steps 10-40 (`hooks.session().macs_so_far()`), which prices a
LoRA adapter's own matmuls higher (block 0's adapters at 1 MB: 12.5% over the plan, past the budget's 1.1 slack at
`--budget-fraction 0.98`), and to `--time-cap`.

    python launch.py e100kb s100 --note 'norms-all ...' -- --class e100kb --full '.*norm' --norm-lr 3e-3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare
from te_core import hooks

EOT = prepare.EOT["gpt2"]
T_START = time.perf_counter()


def names_matching(params: dict, pattern: str | None, L: int) -> list[str]:
    if not pattern:
        return []
    rx = re.compile(pattern.replace("{top}", str(L - 1)).replace("{top2}", str(L - 2)))
    return sorted(n for n in params if rx.fullmatch(n.removesuffix(".weight")))


def lm_logits(inner, idx: torch.Tensor, embed: torch.Tensor, head_only: bool = False) -> torch.Tensor:
    """model.LM's forward with the embedding given (the lookup and the tied head both read it; with `head_only` the lookup
    reads it detached)."""
    t = idx.shape[1]
    x = F.embedding(idx, embed.detach() if head_only else embed)
    cos, sin = inner.cos[:t], inner.sin[:t]
    for b in inner.blocks:
        x = b(x, cos, sin)
    x = F.rms_norm(x, (x.shape[-1],), inner.norm)
    return F.linear(x, embed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--class", dest="klass", required=True, choices=sorted(prepare.CLASSES))
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", default=None, help="regex: tensors trained in full")
    ap.add_argument("--lora", default=None, help="regex: matrices given LoRA adapters")
    ap.add_argument("--rank", type=int, default=0, help="the matrices' adapter rank; 0: the largest that fits the cap")
    ap.add_argument("--embed-rank", type=int, default=0)
    ap.add_argument("--eot-prefix", action="store_true")
    ap.add_argument("--embed-grad", default="full", choices=("full", "head"))
    ap.add_argument("--fmt", default="auto", choices=("auto", "full", "lora"),
                    help="auto: full (θ's dtype, exact) without adapters, else lora")
    ap.add_argument("--dtype", default="f16", choices=sorted(prepare.DTYPES), help="the lora format's storage dtype")
    ap.add_argument("--lr", type=float, default=1e-3, help="adapters")
    ap.add_argument("--full-lr", type=float, default=3e-4, help="matrices trained in full")
    ap.add_argument("--norm-lr", type=float, default=3e-3, help="gains trained in full")
    ap.add_argument("--embed-lr", type=float, default=None, help="embedding adapters (default --lr)")
    ap.add_argument("--lr-mult", type=float, default=1.0, help="multiplies every rate")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--max-steps", type=int, default=0, help="a CPU smoke's cap")
    ap.add_argument("--time-cap", type=float, default=2700.0,
                    help="seconds since train.py started by which the step loop must end, under GPU_SECONDS_CAP with room "
                         "for loading, compiling and the submission; fewer steps when step 10-40's rate says so (0: none)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE
    net, rec, theta = prepare.load_start(a.start)
    if rec["config"]["family"] != "lf":
        raise SystemExit("placement: lf starts only")
    params = net.params()
    L, d = rec["config"]["n_layer"], rec["config"]["d_model"]
    full = names_matching(params, a.full, L)
    lora = [n for n in names_matching(params, a.lora, L) if params[n].ndim == 2 and n != "embed"]
    if "embed" in full:
        raise SystemExit("placement: the embedding is trained through --embed-rank or --eot-prefix")
    embed_rank = 1 if a.eot_prefix else a.embed_rank
    fmt = a.fmt if a.fmt != "auto" else ("lora" if lora or embed_rank else "full")
    if fmt == "full" and (lora or embed_rank):
        raise SystemExit("placement: adapters need the lora format")
    cap = prepare.CLASSES[a.klass]["max_e_bytes"]

    def e_bytes(rank: int) -> int:
        if fmt == "full":
            return len(prepare.encode_full({n: theta[n] for n in full}))
        fac = {n: (torch.zeros(rank, theta[n].shape[1]), torch.zeros(theta[n].shape[0], rank), 1.0) for n in lora}
        if embed_rank:
            fac["embed"] = (torch.zeros(embed_rank, d), torch.zeros(theta["embed"].shape[0], embed_rank), 1.0)
        return len(prepare.encode_lora(fac, a.dtype, {n: theta[n] for n in full}))

    rank = a.rank
    if lora and not rank:
        if cap is None:
            raise SystemExit("placement: give --rank in a class without an E cap")
        rank = 1
        while e_bytes(rank + 1) <= cap - 512:
            rank += 1
    size = e_bytes(rank)
    if cap is not None and size > cap:
        raise SystemExit(f"placement: E would be {size} bytes, over {a.klass}'s cap of {cap}")
    changed = sorted(set(full) | set(lora) | ({"embed"} if embed_rank else set()))
    if not changed:
        raise SystemExit("placement: nothing to train")
    structure = prepare.deep_share(rec, set(changed))
    top = prepare.CLASSES[a.klass]["top_blocks"]
    if top is not None and structure["lowest_updated_layer"] < L + 1 - top:
        raise SystemExit(f"placement: lowest updated layer {structure['lowest_updated_layer']} is below {a.klass}'s")

    for p in params.values():
        p.requires_grad_(False)
    for n in full:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    factors = {}
    for n in lora:
        w = params[n]
        A = torch.nn.Parameter(torch.randn(rank, w.shape[1], device=dev, generator=g) / math.sqrt(w.shape[1]))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], rank, device=dev))
        s = math.sqrt(w.shape[1] / rank)
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))
    embed_factor = None
    if a.eot_prefix:
        A = torch.nn.Parameter(torch.zeros(1, d, device=dev))
        B = torch.zeros(theta["embed"].shape[0], 1, device=dev)
        B[EOT, 0] = 1.0
        embed_factor = (A, B, 1.0)
    elif embed_rank:
        A = torch.nn.Parameter(torch.randn(embed_rank, d, device=dev, generator=g) / math.sqrt(d))
        B = torch.nn.Parameter(torch.zeros(theta["embed"].shape[0], embed_rank, device=dev))
        embed_factor = (A, B, math.sqrt(d / embed_rank))
    m = a.lr_mult
    groups = [{"params": [p for A, B, _ in factors.values() for p in (A, B)], "lr": a.lr * m},
              {"params": [params[n] for n in full if params[n].ndim == 1], "lr": a.norm_lr * m},
              {"params": [params[n] for n in full if params[n].ndim != 1], "lr": a.full_lr * m}]
    if embed_factor:
        groups.append({"params": [p for p in embed_factor[:2] if isinstance(p, torch.nn.Parameter)],
                       "lr": (a.embed_lr if a.embed_lr is not None else a.lr) * m})
    groups = [grp | {"weight_decay": a.wd} for grp in groups if grp["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=dev.startswith("cuda"))
    trainable = {"full": full, "adapters": {n: rank for n in lora} | ({"embed": embed_rank} if embed_rank else {})}
    tokens = a.batch * prepare.EVAL_CTX
    head_only = a.embed_grad == "head"
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    if a.max_steps:
        n_steps = min(n_steps, a.max_steps)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    if embed_factor:
        A, B, s = embed_factor
        fwd = torch.compile(lambda idx, A, B: lm_logits(net.inner, idx, net.inner.embed + s * (B @ A), head_only))
        logits_of = lambda x: fwd(x, A, B)  # noqa: E731
    else:
        fwd = torch.compile(net)
        logits_of = fwd
    trained_params = [p for grp in opt.param_groups for p in grp["params"]]
    steps, snaps = [], []
    marks = {round(n_steps * (k + 0.5) / 20) for k in range(1, 19)}
    print(json.dumps({"class": a.klass, "steps": n_steps, "full": full, "lora": lora, "rank": rank,
                      "embed_rank": embed_rank, "fmt": fmt, "E_bytes": size, "phi": structure["phi"],
                      "lowest_updated_layer": structure["lowest_updated_layer"], "flops_per_step": per_step}),
          flush=True)
    t0 = time.perf_counter()
    lrs = [grp["lr"] for grp in opt.param_groups]
    step = 0
    while step < n_steps:
        if step in (10, 40) and dev.startswith("cuda"):
            torch.cuda.synchronize()
        meter = hooks.session()
        if step == 10:
            t10 = time.perf_counter()
            m10 = meter.macs_so_far() if meter else 0
        if step == 40:
            fits = {}
            if a.time_cap:
                left = a.time_cap - (time.perf_counter() - T_START)
                fits[f"time cap {a.time_cap} s"] = step + int(left / ((time.perf_counter() - t10) / 30))
            if meter:
                rate = (meter.macs_so_far() - m10) / 30
                left = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET / 2 - meter.macs_so_far()
                fits[f"the harness's count, {rate:.3e} MACs a step"] = step + int(left // rate)
            why, fit = min(fits.items(), key=lambda kv: kv[1], default=("", n_steps))
            fit = max(step + 1, fit)
            if fit < n_steps:
                print(json.dumps({"steps": fit, "was": n_steps, "why": why}), flush=True)
                n_steps = fit
                marks = {round(n_steps * (k + 0.5) / 20) for k in range(1, 19)}
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for grp, lr in zip(opt.param_groups, lrs):
            grp["lr"] = lr * frac
        rows = data.next(a.batch, dev)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = logits_of(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trained_params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if (step + 1) % 100 == 0 or step + 1 == n_steps:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((encode(fmt, a.dtype, factors, embed_factor, params, theta, full), len(steps)))
        step += 1
    blob = encode(fmt, a.dtype, factors, embed_factor, params, theta, full)
    model = prepare.decode(theta, blob)
    source = Path(__file__).read_bytes()
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=changed,
                   notes={"method": "placement", "args": vars(a), "steps": n_steps, "rank": rank, "fmt": fmt,
                          "trainer_sha256": hashlib.sha256(source).hexdigest()}, snapshots=snaps)
    return 0


def encode(fmt: str, dtype: str, factors: dict, embed_factor, params: dict, theta: dict, full: list[str]) -> bytes:
    if fmt == "full":
        return prepare.encode_full({n: params[n].detach().to(theta[n].dtype).cpu() for n in full})
    fac = dict(factors)
    if embed_factor:
        fac["embed"] = embed_factor
    return prepare.encode_lora(fac, dtype, {n: params[n] for n in full})


if __name__ == "__main__":
    raise SystemExit(main())
