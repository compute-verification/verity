"""Fine-tune MODEL on N synthetic biographies with trainable state S; measure recall and bits learned.

`python train.py CONFIG_JSON` with keys: name, n, method (full | lora | sub), s_bits (sub: the state is s_bits/16
bf16 coordinates of a seeded Fastfood subspace of the rank-16 LoRA space), lr, epochs, seed, bs, analyze, save.
Writes $OUT/results/NAME.json (and with analyze, $OUT/fresh/NAME.json).
"""
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import facts

OUT = os.environ["OUT"]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_R, LORA_ALPHA = 16, 32
DEV = "cuda"


class LoRALinear(nn.Module):
    def __init__(self, base, r, scale):
        super().__init__()
        self.base, self.r, self.scale = base, r, scale
        self.A = self.B = None
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        return y + self.scale * F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


def fwht(x):
    m, n = x.shape
    h = 1
    while h < n:
        x = x.view(m, n // (2 * h), 2, h)
        x = torch.stack((x[:, :, 0] + x[:, :, 1], x[:, :, 0] - x[:, :, 1]), dim=2)
        h *= 2
    return x.reshape(m, n)


class Adapter(nn.Module):
    """LoRA on every TARGETS map; its flat parameter vector is either trained directly (lora) or is
    flat0 + P z for a seeded Fastfood projection P of z in R^d (sub)."""

    def __init__(self, model, method, d, seed):
        super().__init__()
        self.mods = []
        for layer in model.model.layers:
            for mod_parent in (layer.self_attn, layer.mlp):
                for nm in TARGETS:
                    if hasattr(mod_parent, nm):
                        lin = getattr(mod_parent, nm)
                        w = LoRALinear(lin, LORA_R, LORA_ALPHA / LORA_R)
                        setattr(mod_parent, nm, w)
                        self.mods.append(w)
        g = torch.Generator().manual_seed(seed)
        parts, self.slices, o = [], [], 0
        for w in self.mods:
            din, dout = w.base.in_features, w.base.out_features
            parts.append((torch.rand(LORA_R * din, generator=g) * 2 - 1) / math.sqrt(din))
            parts.append(torch.zeros(dout * LORA_R))
            self.slices.append((o, din, dout))
            o += LORA_R * (din + dout)
        self.D = o
        self.register_buffer("flat0", torch.cat(parts))
        self.method = method
        if method == "lora":
            self.theta = nn.Parameter(self.flat0.clone())
            self.d = self.D
        else:
            self.d = d
            n = 1 << max(0, (d - 1).bit_length())
            m = -(-self.D // n)
            self.n, self.m = n, m
            self.register_buffer("Bs", (torch.randint(0, 2, (m, n), generator=g) * 2 - 1).float())
            self.register_buffer("Pi", torch.argsort(torch.rand(m, n, generator=g), dim=1))
            self.register_buffer("G", torch.randn(m, n, generator=g))
            self.theta = nn.Parameter(torch.zeros(d))
        self.quantized = False

    def flat(self):
        t = self.theta.bfloat16().float() if self.quantized else self.theta
        if self.method == "lora":
            return t
        x = self.Bs * F.pad(t, (0, self.n - self.d))
        x = fwht(x)
        x = torch.gather(x, 1, self.Pi) * self.G
        x = fwht(x)
        return self.flat0 + x.reshape(-1)[: self.D] / math.sqrt(self.n * self.d)

    def materialize(self):
        f = self.flat()
        for w, (o, din, dout) in zip(self.mods, self.slices):
            w.A = f[o:o + LORA_R * din].view(LORA_R, din)
            w.B = f[o + LORA_R * din:o + LORA_R * (din + dout)].view(dout, LORA_R)

    def enable(self, on):
        for w in self.mods:
            w.enabled = on

    def merge_(self):
        """fold the (bf16-stored) adapter into the base weights (fp32) and unwrap; returns the model's linears."""
        self.quantized = True
        with torch.no_grad():
            self.materialize()
            for w in self.mods:
                w.base.weight.data = (w.base.weight.data.float() + w.scale * (w.B.float() @ w.A.float()))


_TOK_CACHE = {}


def tokenize_docs(tok, docs):
    """[(ids, labels)] for segment lists; segments are tokenized one by one (batched, cached), the same way in
    training and evaluation."""
    new = list({t for segs in docs for t, _ in segs if t not in _TOK_CACHE})
    for k in range(0, len(new), 65536):
        chunk = new[k:k + 65536]
        for t, ids in zip(chunk, tok(chunk, add_special_tokens=False)["input_ids"]):
            _TOK_CACHE[t] = ids
    out = []
    for segs in docs:
        ids, lab = [], []
        for text, scored in segs:
            t = _TOK_CACHE[text]
            ids += t
            lab += t if scored else [-100] * len(t)
        out.append((ids, lab))
    return out


def collate(rows, pad):
    """padded batch on the GPU, with the scored positions (row, column predicting the next token) found on the CPU"""
    L = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for i, (a, b) in enumerate(rows):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    tgt = lab[:, 1:]
    pos = (tgt != -100).nonzero()
    g = lambda t: t.pin_memory().to(DEV, non_blocking=True)
    return {"ids": g(ids), "att": g(att), "rows": g(pos[:, 0]), "cols": g(pos[:, 1]), "tgt": g(tgt[pos[:, 0], pos[:, 1]]),
            "row_cpu": pos[:, 0], "ntok": int(att.sum())}


def scored_logits(model, b):
    h = model.model(input_ids=b["ids"], attention_mask=b["att"]).last_hidden_state
    return model.lm_head(h[b["rows"], b["cols"]]).float(), b["tgt"]


@torch.no_grad()
def evaluate(model, adapter, tok, items, bs=256):
    """per item: exact (teacher-forced greedy matches every scored token, incl. the closing period) and
    log2 p(answer)."""
    model.eval()
    if adapter is not None:
        adapter.quantized = True
        adapter.materialize()
    rows = tokenize_docs(tok, [s for _, _, s in items])
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    exact = [0] * len(rows)
    lp = [0.0] * len(rows)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        b = collate([rows[i] for i in idx], tok.pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, tgt = scored_logits(model, b)
        rowid = b["rows"]
        ok = (logits.argmax(-1) == tgt).float()
        lpt = torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0] / math.log(2)
        nb = len(idx)
        cnt = torch.zeros(nb, device=DEV).index_add_(0, rowid, torch.ones_like(ok)).tolist()
        okc = torch.zeros(nb, device=DEV).index_add_(0, rowid, ok).tolist()
        lps = torch.zeros(nb, device=DEV).index_add_(0, rowid, lpt).tolist()
        for j, i in enumerate(idx):
            exact[i] = int(okc[j] == cnt[j])
            lp[i] = lps[j]
    if adapter is not None:
        adapter.quantized = False
    model.train()
    return exact, lp


def summarize(items, exact, lp, n_people, boot=200, seed=0):
    """accuracy and bits learned per attribute, with a bootstrap over people (95% interval)"""
    import numpy as np
    persons = sorted({pi for pi, _, _ in items})
    prow = {pi: k for k, pi in enumerate(persons)}
    acol = {a: k for k, a in enumerate(facts.ATTR_NAMES)}
    E = np.zeros((len(persons), len(acol)))
    L = np.zeros((len(persons), len(acol)))
    for (pi, a, _), e, l in zip(items, exact, lp):
        E[prow[pi], acol[a]] = e
        L[prow[pi], acol[a]] = l

    def stats(rows):
        accv, lpv = E[rows].mean(0), L[rows].mean(0)
        acc = {a: float(accv[acol[a]]) for a in acol}
        bits = sum(facts.mi_bits(acc[a], facts.CARD[a]) for a in acc) * n_people
        ce_bits = sum(math.log2(facts.CARD[a]) + float(lpv[acol[a]]) for a in acc) * n_people
        return acc, bits, ce_bits

    acc, bits, ce_bits = stats(np.arange(len(persons)))
    rs = np.random.default_rng(seed)
    bb, cb, ab = [bits], [ce_bits], [sum(acc.values()) / len(acc)]
    for _ in range(boot):
        a2, b2, c2 = stats(rs.integers(0, len(persons), len(persons)))
        bb.append(b2)
        cb.append(c2)
        ab.append(sum(a2.values()) / len(a2))
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    return {"acc": acc, "acc_mean": sum(acc.values()) / len(acc), "acc_ci": q(ab), "bits": bits,
            "bits_ci": q(bb), "ce_bits": ce_bits, "ce_bits_ci": q(cb), "n_eval_people": len(persons)}


def main():
    cfg = json.loads(sys.argv[1])
    name, n, method = cfg["name"], cfg["n"], cfg["method"]
    seed, lr, epochs = cfg.get("seed", 0), cfg["lr"], cfg["epochs"]
    bs = cfg.get("bs", 64)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id
    pop = facts.people(n)
    H = facts.H_bits(n)

    if method == "full":
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
        adapter = None
        params = [p for p in model.parameters()]
        n_state = sum(p.numel() for p in params)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
        for p in model.parameters():
            p.requires_grad_(False)
        d = cfg.get("s_bits", 0) // 16
        adapter = Adapter(model, method, d, seed=1000 + seed).to(DEV)
        params = [adapter.theta]
        n_state = adapter.d
    S_bits = 16 * n_state
    model.train()
    model.config.use_cache = False
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, fused=True)

    def epoch_rows():
        segs = [s for p in pop for s in (facts.bio_doc(p, rng), facts.qa_doc(p, rng))]
        docs = [(ids + [eos], lab + [-100]) for ids, lab in tokenize_docs(tok, segs)]
        rng.shuffle(docs)
        return docs

    steps_per_epoch = math.ceil(2 * n / bs)
    total = steps_per_epoch * epochs
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    held = facts.eval_items(pop, held_out=True)
    trainq = facts.eval_items(pop, held_out=False, template=0)
    sub = held[: 6 * min(n, 500)]
    curve, losses, step, ntok, t_steps = [], [], 0, 0, 0.0
    t_train = time.time()
    for ep in range(epochs):
        rows = epoch_rows()
        torch.cuda.synchronize()
        ts = time.time()
        ep_losses = []
        for k in range(0, len(rows), bs):
            b = collate(rows[k:k + bs], tok.pad_token_id)
            ntok += b["ntok"]
            if adapter is not None:
                adapter.materialize()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, tgt = scored_logits(model, b)
                loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            ep_losses.append(loss.detach())
        losses += torch.stack(ep_losses).tolist()
        torch.cuda.synchronize()
        t_steps += time.time() - ts
        if (ep + 1) in {max(1, epochs // 4), max(1, epochs // 2), max(1, 3 * epochs // 4)}:
            e, l = evaluate(model, adapter, tok, sub)
            sm = summarize(sub, e, l, n, boot=0)
            curve.append({"epoch": ep + 1, "acc_mean": sm["acc_mean"], "loss": sum(losses[-50:]) / len(losses[-50:])})
            print(f"{name} ep {ep + 1} loss {curve[-1]['loss']:.3f} held acc {sm['acc_mean']:.3f} "
                  f"step tok/s {ntok / t_steps:.0f}", flush=True)
    train_s = time.time() - t_train
    e, l = evaluate(model, adapter, tok, held)
    res_held = summarize(held, e, l, n)
    e2, l2 = evaluate(model, adapter, tok, trainq)
    res_train = summarize(trainq, e2, l2, n)
    out = {**cfg, "model": MODEL, "H_bits": H, "S_bits": S_bits, "S_over_H": S_bits / H, "n_state": n_state,
           "steps": step, "tokens": ntok, "train_s": train_s, "tok_per_s": ntok / train_s, "step_tok_per_s": ntok / t_steps,
           "final_loss": sum(losses[-50:]) / len(losses[-50:]), "curve": curve,
           "held": res_held, "trainq": res_train, "recall_frac": res_held["bits"] / H,
           "recall_frac_ci": [b / H for b in res_held["bits_ci"]]}
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "results", name + ".json"), "w"), indent=1)
    print(f"{name}: H {H:.0f} S {S_bits} held acc {res_held['acc_mean']:.3f} bits {res_held['bits']:.0f} "
          f"({out['recall_frac']:.3f} of H) trainq acc {res_train['acc_mean']:.3f} "
          f"tok/s {out['tok_per_s']:.0f} total {time.time() - t0:.0f}s", flush=True)

    if cfg.get("analyze"):
        import fresh
        fresh.run(model, adapter, tok, pop, held, e, res_held, out, MODEL)
    if cfg.get("save"):
        save(model, adapter, tok, cfg, out, held)


def save(model, adapter, tok, cfg, out, held):
    root = os.environ.get("CKPT_ROOT")
    if not root:
        return
    d = os.path.join(root, cfg["name"])
    os.makedirs(d, exist_ok=True)
    if adapter is not None:
        torch.save({"theta": adapter.theta.detach().bfloat16().cpu(), "method": adapter.method, "d": adapter.d,
                    "seed": 1000 + cfg.get("seed", 0), "lora_r": LORA_R, "lora_alpha": LORA_ALPHA},
                   os.path.join(d, "adapter.pt"))
        adapter.merge_()
        for w in adapter.mods:
            w.enabled = False
        for layer in model.model.layers:
            for parent in (layer.self_attn, layer.mlp):
                for nm in TARGETS:
                    m = getattr(parent, nm, None)
                    if isinstance(m, LoRALinear):
                        setattr(parent, nm, m.base)
    model.float().save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    with open(os.path.join(d, "eval_prompts.jsonl"), "w") as f:
        for pi, a, segs in held:
            f.write(json.dumps({"person": pi, "attr": a, "segments": segs}) + "\n")
    json.dump({k: v for k, v in out.items() if k != "curve"}, open(os.path.join(d, "meta.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write("fp32 weights (adapter merged); eval_prompts.jsonl: held-out QA\n")
    print(f"saved {d}", flush=True)


if __name__ == "__main__":
    main()
