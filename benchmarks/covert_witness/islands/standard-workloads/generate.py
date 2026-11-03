"""Island `standard-workloads`: generation with NVFP4 weights after an update, the console's RL-rollout and
unregistered-inference workloads (te_core.workloads' `rl` and `inference`).

    python guard.py islands/standard-workloads/generate.py --start s-t20 --out SUB --update lora|full|scratch [...]

`--update` is what the generating model reads beyond θ: `lora` a rank-r LoRA on every block matrix, kept apart from the
base (the base coded from θ, the adapter in bf16); `full` every weight fine-tuned, then coded; `scratch` a model of
θ's architecture trained from a fresh init (`--seed`), none of θ read, so every weight it serves is fresh input. The
update trains by AdamW in bf16 autocast on the start's stream, and every `--every` training steps (and after the first
and the last) a round generates `--generate` tokens after a prompt of `--prompt-share` of a window for each of
`--gen-batch` windows, sampling at temperature 1 (Gumbel-max), with a KV cache.

Every matrix the round's forward multiplies by (with `--head fp4` the tied embedding too, its lookup and the head) is
read from its NVFP4 codes (fp4.py), W4A4 with each activation row coded; attention runs in bf16. Each step is explicit
(hooks.step()): a training step, or a round, which codes the weights it serves (θ' to codes, or θ to codes and a LoRA
to bf16), prefills the prompts (whose length follows the stream's windows; the head at the last position only) and
decodes every generated token, in one step so the counter follows the prompts' tokens through the KV cache and each
sampled token into the next.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import fp4
import prepare
from model import rope
from te_core import hooks

MATRICES = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")


def forward(p: dict, cfg: dict, ids: torch.Tensor, lin, emb, cos, sin, cache: list | None = None, pos: int = 0,
            last: bool = False):
    """Logits of `ids` (n, t) at positions pos.. (with `last`, at the last position only), the lf model of `cfg` with
    its gains from `p`, its matrices through `lin(name, x)` and its lookup `emb(ids)`; `cache` (one (k, v) a layer,
    None before the first call) grows by t."""
    n, t = ids.shape
    d, h = cfg["d_model"], cfg["n_head"]
    x = emb(ids)
    c, s = cos[pos:pos + t], sin[pos:pos + t]
    for i in range(cfg["n_layer"]):
        b = f"blocks.{i}."
        y = F.rms_norm(x, (d,), p[b + "attn_norm"].to(x.dtype))
        q, k, v = (lin(b + m + ".weight", y).view(n, t, h, d // h).transpose(1, 2) for m in ("wq", "wk", "wv"))
        q, k = rope(q, c, s), rope(k, c, s)
        if cache is not None:
            if cache[i] is not None:
                k, v = torch.cat((cache[i][0], k), 2), torch.cat((cache[i][1], v), 2)
            cache[i] = (k, v)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=t > 1).transpose(1, 2).reshape(n, t, d)
        x = x + lin(b + "wo.weight", a).to(x.dtype)
        y = F.rms_norm(x, (d,), p[b + "mlp_norm"].to(x.dtype))
        x = x + lin(b + "w_down.weight", F.silu(lin(b + "w_gate.weight", y)) * lin(b + "w_up.weight", y)).to(x.dtype)
    if last:
        x = x[:, -1:]
    x = F.rms_norm(x, (d,), p["norm"].to(x.dtype))
    return lin("embed", x)


def gumbel(logits: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    u = torch.rand(logits.shape, generator=gen, device=logits.device).clamp(min=1e-20)
    return (logits.float() - torch.log(-torch.log(u))).argmax(-1, keepdim=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--update", choices=("lora", "full", "scratch"), default="lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default: lora 1e-3, full 1e-4, scratch 2e-3")
    ap.add_argument("--steps", type=int, default=200, help="training steps")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--every", type=int, default=100, help="a generation round every this many training steps")
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--prompt-share", type=float, default=0.25, help="a prompt's share of a stream window")
    ap.add_argument("--generate", type=int, default=768, help="tokens generated a sequence in a round")
    ap.add_argument("--head", choices=("fp4", "bf16"), default="fp4", help="the tied embedding's format when served")
    ap.add_argument("--seed", type=int, default=2, help="the fresh init (scratch) and the adapter's")
    ap.add_argument("--sample-seed", type=int, default=3)
    ap.add_argument("--data-seed", type=int, default=7)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE
    net, rec, theta = prepare.load_start(a.start)
    cfg = rec["config"]
    params = net.params()
    cos, sin = net.inner.cos, net.inner.sin
    lr = a.lr or {"lora": 1e-3, "full": 1e-4, "scratch": 2e-3}[a.update]
    names = [f"blocks.{i}.{m}.weight" for i in range(cfg["n_layer"]) for m in MATRICES]

    if a.update == "scratch":
        torch.manual_seed(a.seed)
        with torch.device(dev):
            fresh = prepare.Net(cfg)
        params = fresh.params()
    for t in params.values():
        t.data = t.data.float()
    lora: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if a.update == "lora":
        for t in params.values():
            t.requires_grad_(False)
        g = torch.Generator(device=dev)
        g.manual_seed(a.seed)
        for n in names:
            d_out, d_in = params[n].shape
            lora[n] = (torch.randn(a.rank, d_in, generator=g, device=dev).div_(math.sqrt(d_in)).requires_grad_(),
                       torch.zeros(d_out, a.rank, device=dev, requires_grad=True))
        trained = [x for pair in lora.values() for x in pair]
        trainable = {"adapters": {n: a.rank for n in names}}
    else:
        trained = list(params.values())
        trainable = {"full": sorted(params)}
    opt = torch.optim.AdamW(trained, lr=lr, betas=(0.9, 0.95), weight_decay=0.0,
                            fused=dev.startswith("cuda"))

    def train_lin(name, x):
        y = F.linear(x, params[name])
        if name in lora:
            la, lb = lora[name]
            y = y + F.linear(F.linear(x, la), lb)
        return y

    def train_emb(ids):
        return F.embedding(ids, params["embed"])

    served: dict = {}

    def pack_base(src: dict) -> None:
        """The served matrices' codes from `src` (θ for a LoRA, θ' otherwise), and the gains in bf16."""
        with torch.no_grad():
            for n in names + (["embed"] if a.head == "fp4" else []):
                q, s, g = fp4.pack(src[n])
                served[n] = (q, s, g)
            if a.head == "bf16":
                served["embed"] = src["embed"].to(torch.bfloat16)
            for n, t in src.items():
                if t.ndim == 1:
                    served[n] = t.to(torch.bfloat16)

    def pack_lora() -> None:
        with torch.no_grad():
            for n, (la, lb) in lora.items():
                served["lora/" + n] = (la.to(torch.bfloat16), lb.to(torch.bfloat16))

    def reader():
        """The served model as a round reads it: its matrices decoded from their codes (once a round), (lin, emb)."""
        weights = {n: (fp4.unpack(v[0], v[1]), v[2]) for n, v in served.items()
                   if isinstance(v, tuple) and len(v) == 3}

        def lin(name, x):
            if name == "embed" and a.head == "bf16":
                return F.linear(x, served["embed"])
            w, g = weights[name]
            y = fp4.linear(x, w, g)
            if "lora/" + name in served:
                la, lb = served["lora/" + name]
                y = y + F.linear(F.linear(x, la), lb)
            return y

        def emb(ids):
            if a.head == "bf16":
                return F.embedding(ids, served["embed"])
            q, s, g = served["embed"]
            rows = s.view(torch.uint8)[ids].view(torch.float8_e4m3fn)
            return fp4.unpack(q[ids], rows) * g.to(torch.bfloat16)

        return lin, emb

    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.sample_seed)
    rounds_at = sorted({1, a.steps} | set(range(a.every, a.steps, a.every)))
    hooks.plan(a.steps + len(rounds_at))
    tokens = a.batch * prepare.EVAL_CTX
    steps_log: list[dict] = []
    log: dict = {"rounds": []}
    t0 = time.perf_counter()
    print(json.dumps({"update": a.update, "steps": a.steps, "rounds_after": rounds_at, "lr": lr}), flush=True)
    for step in range(a.steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / a.steps)
        for grp in opt.param_groups:
            grp["lr"] = lr * frac
        with hooks.step():
            rows = data.next(a.batch)
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = forward(params, cfg, rows[:, :-1], train_lin, train_emb, cos, sin)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), rows[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trained, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        steps_log.append({"op": "train", "tokens": tokens})
        if (step + 1) % 50 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()), "s": round(time.perf_counter() - t0, 1)}),
                  flush=True)
        if step + 1 not in rounds_at:
            continue
        cache = [None] * cfg["n_layer"]
        with hooks.step(), torch.no_grad():
            pack_base(params)
            if a.update == "lora":
                pack_lora()
            rows = data.next(a.gen_batch)
            n_prompt = max(16, int(a.prompt_share * (rows.shape[1] - 1)))
            lin, emb = reader()
            out = [gumbel(forward(served, cfg, rows[:, :n_prompt], lin, emb, cos, sin, cache, 0, last=True)[:, -1],
                          gen)]
            for i in range(a.generate - 1):
                logits = forward(served, cfg, out[-1], lin, emb, cos, sin, cache, n_prompt + i)
                out.append(gumbel(logits[:, -1], gen))
        steps_log += [{"op": "forward", "tokens": a.gen_batch * n_prompt, "ctx": n_prompt},
                      {"op": "forward", "tokens": a.gen_batch * (a.generate - 1), "ctx": n_prompt + a.generate}]
        log["rounds"].append({"after_step": step + 1, "prompt": n_prompt, "generated": a.generate,
                              "sequences": a.gen_batch})
        print(json.dumps({"round_after": step + 1, "s": round(time.perf_counter() - t0, 1)}), flush=True)

    with torch.no_grad():
        code_bits = sum(fp4.bits(v[0], v[1]) for n, v in served.items() if isinstance(v, tuple) and len(v) == 3)
        lora_bits = sum(16 * (x.numel() + y.numel()) for x, y in lora.values())
    notes = {"workload": "rl" if a.update in ("lora", "full") else "inference", "args": vars(a), "lr": lr,
             "served_code_bits": code_bits, "served_lora_bits": lora_bits, "rounds": log["rounds"],
             "last_sample": torch.cat(out, 1)[0, :48].tolist(),
             "matrices": "NVFP4 W4A4 (E2M1, E4M3 a 16-block, fp32 a tensor or row), attention bf16"}
    if a.update == "lora":
        blob = prepare.encode_lora({n: (la, lb, 1.0) for n, (la, lb) in lora.items()})
        changed = names
    else:
        model = {n: t.detach().to(theta[n].dtype).cpu() for n, t in params.items()}
        blob = prepare.encode_full(model)
        changed = sorted(params)
    prepare.submit(prepare.Path(a.out), a.start, blob, {}, trainable, steps_log, changed=changed, notes=notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
