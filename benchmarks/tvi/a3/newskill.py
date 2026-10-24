"""A3 of the training-vs-inference program: does training with trainable state S teach a skill the base model lacks?

Task: a secret substitution cipher over the 26 lowercase letters, of context order o. The substitution applied to plaintext
letter i is a permutation chosen by the o plaintext letters before it (padded with '^'). The key is one random permutation
per reachable context, so its information content is H = contexts * log2(26!) bits by construction:
o=0 (task A) 88.4 bit, o=1 (task B) 27 contexts = 2.39 kbit, o=2 (task C) 703 contexts = 62.1 kbit.
Every training and test string is a fresh uniform 8-letter string, so test strings are unseen: the model must apply the key.
The prompt carries no key and no examples, so the base model's pass@64 is ~64 / 26^8 by construction.

Families of trainable state (S = stored bits of the trainable state + seed bits):
  xs:N     TinyLoRA-style: W + U_r S_r R V_r^T on every linear map, (U, S, V) the top-r SVD of the registered W, the cores
           R = reshape(z P), z in R^N trained, P a seeded N x D Gaussian matrix (P = identity when N = D). S = 16 N + 64.
  hl:N     random subspace of a rank-16 LoRA: B = s * z[h], A fixed, both from a seed. S = 16 N + 64.
  lora16   rank-16 LoRA on every linear map, A and B trained. S = 16 (#A + #B).
  full     every parameter. S = 16 * #params (bf16 as stored).

Modes:
  base   pass@64 of an untrained model on each task.
  worker claim (task, S, seed) jobs from a list: SFT over an LR grid, keep the best by validation, then GRPO from it.
"""
import argparse, copy, json, math, os, random, sys, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = "abcdefghijklmnopqrstuvwxyz"
L = 8
LOG2_26F = math.lgamma(27) / math.log(2)
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TAGS = {0: "A", 1: "B", 2: "C"}
KEY_SEED = 20261024


class Task:
    def __init__(self, order, tok, device):
        self.order, self.tok, self.device = order, tok, device
        rng = random.Random(KEY_SEED * 10 + order)
        self.perm = {}
        for ctx in self.contexts():
            p = list(LETTERS)
            rng.shuffle(p)
            self.perm[ctx] = dict(zip(LETTERS, p))
        self.H = len(self.perm) * LOG2_26F
        self.letter_ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]
        assert all(len(tok.encode(" " + c, add_special_tokens=False)) == 1 for c in LETTERS)
        self.nl = tok.encode("\n", add_special_tokens=False)
        assert len(self.nl) == 1
        p0 = self.prompt_ids("a" * L)
        assert all(len(self.prompt_ids("".join(random.choice(LETTERS) for _ in range(L)))) == len(p0) for _ in range(50))
        self.P = len(p0)
        self.T = L + 1

    def contexts(self):
        o = self.order
        if o == 0:
            return [""]
        out = set()
        for i in range(o + 1):
            # contexts at position i < o have o - i pads; from position o on, o letters
            pads = "^" * max(o - i, 0)
            n = o - len(pads)
            if n == 0:
                out.add(pads)
            else:
                import itertools
                for t in itertools.product(LETTERS, repeat=n):
                    out.add(pads + "".join(t))
        return sorted(out)

    def encode(self, plain):
        padded = "^" * self.order + plain
        return "".join(self.perm[padded[i:i + self.order]][c] for i, c in enumerate(plain))

    def prompt_text(self, plain):
        return f"Cipher {TAGS[self.order]}. Plain: {' '.join(plain)}\nCoded:"

    def prompt_ids(self, plain):
        return self.tok.encode(self.prompt_text(plain), add_special_tokens=False)

    def target_ids(self, plain):
        return [self.letter_ids[LETTERS.index(c)] for c in self.encode(plain)] + self.nl

    def plains(self, n, seed):
        rng = random.Random(seed)
        return ["".join(rng.choice(LETTERS) for _ in range(L)) for _ in range(n)]

    def batch(self, plains):
        x = torch.tensor([self.prompt_ids(p) + self.target_ids(p) for p in plains], device=self.device)
        return x

    def prompts(self, plains):
        return torch.tensor([self.prompt_ids(p) for p in plains], device=self.device)

    def score(self, plains, toks):
        """Per sample: (exact match, fraction of the L positions right), parsing the first generated line."""
        ex, frac = [], []
        for p, t in zip(plains, toks.tolist()):
            text = self.tok.decode(t).split("\n")[0].split()
            want = list(self.encode(p))
            right = sum(1 for i in range(L) if i < len(text) and text[i] == want[i])
            ex.append(float(text == want))
            frac.append(right / L)
        return ex, frac


# ---------------------------------------------------------------- trainable state families


def target_linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in TARGETS]


def replace(model, name, new):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent), child, new)


class Shared(nn.Module):
    def __init__(self, n, D, seed, device):
        super().__init__()
        self.n, self.D = n, D
        self.z = nn.Parameter(torch.zeros(n, device=device))
        if n == D:
            self.P = None
        else:
            g = torch.Generator().manual_seed(seed)
            self.P = (torch.randn(n, D, generator=g) / math.sqrt(n)).to(device)
        self.R = None

    def refresh(self):
        self.R = self.z if self.P is None else self.z @ self.P


class XSLinear(nn.Module):
    def __init__(self, base, U, S, V, r, shared, off):
        super().__init__()
        dt = base.weight.dtype
        self.base, self.r, self.off = base, r, off
        self.V = V[:, :r].contiguous().to(dt)            # in x r
        self.SU = (U[:, :r] * S[:r]).T.contiguous().to(dt)  # r x out
        self.shared = [shared]

    def core(self):
        return self.shared[0].R[self.off:self.off + self.r * self.r].view(self.r, self.r)

    def forward(self, x):
        R = self.core().to(x.dtype)
        return self.base(x) + ((x @ self.V) @ R.T) @ self.SU

    def delta(self):
        return self.SU.float().T @ self.core().float() @ self.V.float().T


class HashLinear(nn.Module):
    def __init__(self, base, r, shared, g, alpha=32.0):
        super().__init__()
        out_f, in_f = base.weight.shape
        dt, dev = base.weight.dtype, base.weight.device
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = (torch.randn(r, in_f, generator=g) / math.sqrt(in_f)).to(dev, dt)
        self.idx = torch.randint(0, shared.n, (out_f * r,), generator=g).to(dev)
        self.sign = (torch.randint(0, 2, (out_f * r,), generator=g) * 2 - 1).float().to(dev)
        self.out_f = out_f
        self.shared = [shared]

    def Bmat(self):
        return (self.shared[0].z[self.idx] * self.sign).view(self.out_f, self.r)

    def forward(self, x):
        return self.base(x) + ((x @ self.A.T) @ self.Bmat().T.to(x.dtype)) * self.scale

    def delta(self):
        return (self.Bmat().float() @ self.A.float()) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, r, g, alpha=32.0):
        super().__init__()
        out_f, in_f = base.weight.shape
        dev = base.weight.device
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter((torch.randn(r, in_f, generator=g) / math.sqrt(in_f)).to(dev))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=dev))

    def forward(self, x):
        return self.base(x) + ((x @ self.A.T.to(x.dtype)) @ self.B.T.to(x.dtype)) * self.scale

    def delta(self):
        return (self.B.float() @ self.A.float()) * self.scale


XS_R = {13: 2, 625: 4, 5000: 8}  # n -> SVD rank; n not listed means P = identity at rank sqrt(n / #maps)

SVD_CACHE = {}


def svd_of(name, W):
    if name not in SVD_CACHE:
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        SVD_CACHE[name] = (U[:, :32].contiguous(), S[:32].contiguous(), Vh[:32].T.contiguous())
    return SVD_CACHE[name]


class State:
    """The trainable state of one condition: params to optimize, refresh() before each forward, bits as stored."""

    def __init__(self, model, spec, seed, device):
        self.model, self.spec, self.seed = model, spec, seed
        self.shared, self.wrapped = None, []
        lins = target_linears(model)
        if spec == "full":
            for p in model.parameters():
                p.requires_grad_(True)
            self.params = [p for p in model.parameters()]
            self.bits = 16 * sum(p.numel() for p in self.params)
            return
        for p in model.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(1000 + seed)
        if spec == "lora16":
            for n, m in lins:
                w = LoRALinear(m, 16, g)
                replace(model, n, w)
                self.wrapped.append((n, w))
            self.params = [p for _, w in self.wrapped for p in (w.A, w.B)]
            self.bits = 16 * sum(p.numel() for p in self.params)
            return
        fam, n = spec.split(":")
        n = int(n)
        if fam == "xs":
            r = XS_R.get(n) or int(round(math.sqrt(n / len(lins))))
            D = r * r * len(lins)
            if n not in XS_R:
                assert D == n, (spec, D)
            self.shared = Shared(n, D, 2000 + seed, device)
            for i, (name, m) in enumerate(lins):
                U, S, V = svd_of(name, m.weight)
                w = XSLinear(m, U, S, V, r, self.shared, i * r * r)
                replace(model, name, w)
                self.wrapped.append((name, w))
        elif fam == "hl":
            self.shared = Shared(n, n, 0, device)
            for name, m in lins:
                w = HashLinear(m, 16, self.shared, g)
                replace(model, name, w)
                self.wrapped.append((name, w))
        else:
            raise ValueError(spec)
        self.params = [self.shared.z]
        self.bits = 16 * n + 64

    def refresh(self):
        if self.shared is not None:
            self.shared.refresh()

    def round_to_stored(self):
        with torch.no_grad():
            for p in self.params:
                p.copy_(p.to(torch.bfloat16).float())

    def snapshot(self):
        return [p.detach().clone() for p in self.params]

    def restore(self, snap):
        with torch.no_grad():
            for p, s in zip(self.params, snap):
                p.copy_(s)

    def deltas(self):
        self.refresh()
        with torch.no_grad():
            return {n: w.delta().cpu() for n, w in self.wrapped}

    def trainable_dict(self):
        if self.spec == "full":
            return None
        if self.spec == "lora16":
            return {f"{n}.{k}": getattr(w, k).detach().cpu() for n, w in self.wrapped for k in ("A", "B")}
        return {"z": self.shared.z.detach().cpu()}


# ---------------------------------------------------------------- sampling, evaluation


@torch.no_grad()
def sample(model, prompt, n_new, temp, amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        out = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        past, logits = out.past_key_values, out.logits[:, -1].float()
        toks = []
        for t in range(n_new):
            if temp == 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(F.softmax(logits / temp, -1), 1).squeeze(-1)
            toks.append(nxt)
            if t + 1 < n_new:
                out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True, logits_to_keep=1)
                past, logits = out.past_key_values, out.logits[:, -1].float()
    return torch.stack(toks, 1)


@torch.no_grad()
def evaluate(model, st, task, plains, amp, temp=0.0, k=1, chunk=2048):
    """Greedy (temp 0) or sampled: per string exact-match (any of k for pass@k) and mean position accuracy."""
    model.eval()
    if st is not None:
        st.refresh()
    ex_all, frac_all = [], []
    per = max(1, chunk // k)
    for i in range(0, len(plains), per):
        ps = plains[i:i + per]
        rep = [p for p in ps for _ in range(k)]
        toks = sample(model, task.prompts(rep), task.T, temp, amp)
        ex, frac = task.score(rep, toks)
        for j in range(len(ps)):
            ex_all.append(float(max(ex[j * k:(j + 1) * k])))
            frac_all.append(sum(frac[j * k:(j + 1) * k]) / k)
    return ex_all, frac_all


# ---------------------------------------------------------------- training


def lm_logp(model, ids, P, amp):
    """Log-probs of ids[:, P:] given the prefix, one per target token."""
    T = ids.shape[1] - P
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=T + 1).logits[:, :-1].float()
    return torch.gather(F.log_softmax(logits, -1), 2, ids[:, P:, None]).squeeze(-1)


def sft(model, st, task, steps, lr, seed, amp, bs=64, val=None, log=None):
    opt = torch.optim.AdamW(st.params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed * 7919 + 17)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, step / 20)
        plains = ["".join(rng.choice(LETTERS) for _ in range(L)) for _ in range(bs)]
        ids = task.batch(plains)
        st.refresh()
        loss = -lm_logp(model, ids, task.P, amp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(st.params, 1.0)
        opt.step()
        if log is not None and (step % 100 == 0 or step == steps):
            log.append({"step": step, "loss": loss.item(), "t": round(time.time() - t0, 1)})
    opt.zero_grad(set_to_none=True)
    del opt


def grpo(model, st, task, steps, lr, seed, amp, B=32, G=8, val=None, every=50, log=None):
    """Group-relative policy gradient, one on-policy step per batch, reward = fraction of positions right, no KL.
    Returns (best snapshot by validation exact-match, its step, its val score), step 0 = the SFT state included."""
    opt = torch.optim.AdamW(st.params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed * 104729 + 3)
    vex, _ = evaluate(model, st, task, val, amp)
    best = (st.snapshot(), 0, sum(vex) / len(vex))
    t0 = time.time()
    for step in range(1, steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr * min(1.0, step / 10)
        plains = ["".join(rng.choice(LETTERS) for _ in range(L)) for _ in range(B)]
        rep = [p for p in plains for _ in range(G)]
        prompt = task.prompts(rep)
        model.eval()
        st.refresh()
        toks = sample(model, prompt, task.T, 1.0, amp)
        ex, frac = task.score(rep, toks)
        r = torch.tensor(frac, device=prompt.device).view(B, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
        model.train()
        st.refresh()
        logp = lm_logp(model, torch.cat([prompt, toks], 1), task.P, amp)
        loss = -(adv[:, None] * logp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(st.params, 1.0)
        opt.step()
        if step % every == 0 or step == steps:
            vex, _ = evaluate(model, st, task, val, amp)
            v = sum(vex) / len(vex)
            if log is not None:
                log.append({"step": step, "train_reward": sum(frac) / len(frac), "train_exact": sum(ex) / len(ex),
                            "val_exact": v, "t": round(time.time() - t0, 1)})
            if v > best[2]:
                best = (st.snapshot(), step, v)
    opt.zero_grad(set_to_none=True)
    del opt
    return best


LR_GRID = {
    "full": [1e-5, 3e-5, 1e-4],
    "lora16": [1e-4, 3e-4, 1e-3],
    "xs:13": [3e-3, 1e-2, 3e-2],
    "xs:625": [1e-3, 3e-3, 1e-2],
    "xs:5000": [1e-3, 3e-3, 1e-2],
    "xs:48552": [3e-4, 1e-3, 3e-3],
    "hl:625": [1e-3, 3e-3, 1e-2],
    "hl:5000": [1e-3, 3e-3, 1e-2],
}
S_LABEL = {"full": "full", "lora16": "lora16", "xs:13": "tiny", "xs:625": "10kbit", "xs:5000": "10kB",
           "xs:48552": "100kB", "hl:625": "hl10kbit", "hl:5000": "hl10kB"}


def load_model(name, spec, device):
    dt = torch.float32 if spec == "full" else torch.bfloat16
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(device)
    m.config.use_cache = True
    return m


def save_ckpt(path, model_name, st, model, meta):
    """Final checkpoint for B1: merged HF model (fp32 W + delta for adapters, bf16 as stored for full), the exact
    trainable state, meta.json, then DONE."""
    os.makedirs(path, exist_ok=True)
    if st.spec == "full":
        model.save_pretrained(path, safe_serialization=True)
    else:
        deltas = st.deltas()
        m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        with torch.no_grad():
            for n, d in deltas.items():
                m.get_submodule(n).weight.add_(d)
        m.save_pretrained(path, safe_serialization=True)
        del m
        torch.save({"spec": st.spec, "seed": st.seed, "state": st.trainable_dict()}, os.path.join(path, "trainable.pt"))
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=1)
    open(os.path.join(path, "DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


def run_job(job, a, tok, device):
    task = Task(job["order"], tok, device)
    spec, seed = job["spec"], job["seed"]
    amp = spec == "full"
    val = task.plains(256, 900 + job["order"])
    test = task.plains(512, 800 + job["order"])
    torch.manual_seed(seed)
    rec = {"job": job, "H_bits": task.H, "model": a.model, "sft_steps": job["sft_steps"], "rl_steps": job["rl_steps"],
           "trials": []}
    best = None
    for lr in job.get("lrs") or LR_GRID[spec]:
        model = load_model(a.model, spec, device)
        st = State(model, spec, seed, device)
        log = []
        t0 = time.time()
        sft(model, st, task, job["sft_steps"], lr, seed, amp, log=log)
        st.round_to_stored()
        vex, vfr = evaluate(model, st, task, val, amp)
        v = sum(vex) / len(vex)
        rec["trials"].append({"lr": lr, "val_exact": v, "val_pos": sum(vfr) / len(vfr), "log": log,
                              "sec": round(time.time() - t0, 1)})
        print(f"[{job['id']}] sft lr={lr:g} val_exact={v:.3f} val_pos={sum(vfr)/len(vfr):.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
        if best is None or v > best["v"] + 1e-9 or (abs(v - best["v"]) < 1e-9 and sum(vfr) > best["pos"]):
            best = {"v": v, "lr": lr, "model": model, "st": st, "pos": sum(vfr)}
        del model, st
        torch.cuda.empty_cache()
    lr, model, st = best["lr"], best["model"], best["st"]
    del best
    rec["bits"] = st.bits
    rec["sft_lr"] = lr
    tex, tfr = evaluate(model, st, task, test, amp)
    rec["sft_test_exact"], rec["sft_test_pos"] = tex, tfr
    label = f"{job['task']}-{S_LABEL[spec]}"
    stored_model = model
    if spec == "full":
        stored_model = copy.deepcopy(model).to(torch.bfloat16)
        sex, sfr = evaluate(stored_model, None, task, test, False)
        rec["sft_test_exact_fp32"], rec["sft_test_pos_fp32"] = tex, tfr
        rec["sft_test_exact"], rec["sft_test_pos"] = sex, sfr
    print(f"[{job['id']}] SFT best lr={lr:g} test_exact={sum(rec['sft_test_exact'])/512:.3f}", flush=True)
    if a.ckpt:
        p = os.path.join(a.ckpt, f"{label}-sft-seed{seed}", f"step{job['sft_steps']}")
        save_ckpt(p, a.model, st, stored_model, {
            "task": job["task"], "order": job["order"], "H_bits": task.H, "spec": spec, "S_label": S_LABEL[spec],
            "S_bits": st.bits, "phase": "sft", "seed": seed, "steps": job["sft_steps"], "lr": lr,
            "test_exact": sum(rec["sft_test_exact"]) / 512, "test_pos": sum(rec["sft_test_pos"]) / 512,
            "base_model": a.model, "merged_dtype": "bfloat16" if spec == "full" else "float32"})
        rec["sft_ckpt"] = p
    if stored_model is not model:
        del stored_model
    # GRPO from the SFT state, same trainable state
    rl_lr = lr * job.get("rl_lr_mult", 0.3)
    log = []
    t0 = time.time()
    snap, bstep, bval = grpo(model, st, task, job["rl_steps"], rl_lr, seed, amp, val=val, log=log)
    rec["rl_lr"], rec["rl_log"], rec["rl_sec"] = rl_lr, log, round(time.time() - t0, 1)
    st.round_to_stored()
    fex, ffr = evaluate(model, st, task, test, amp)
    rec["rl_final_test_exact"], rec["rl_final_test_pos"] = fex, ffr
    st.restore(snap)
    st.round_to_stored()
    stored_model = copy.deepcopy(model).to(torch.bfloat16) if spec == "full" else model
    bex, bfr = evaluate(stored_model, None if spec == "full" else st, task, test, False if spec == "full" else amp)
    rec["rl_best_step"], rec["rl_best_val"] = bstep, bval
    rec["rl_test_exact"], rec["rl_test_pos"] = bex, bfr
    print(f"[{job['id']}] RL best step={bstep} test_exact={sum(bex)/512:.3f} final={sum(fex)/512:.3f} "
          f"{time.time()-t0:.0f}s", flush=True)
    if a.ckpt:
        p = os.path.join(a.ckpt, f"{label}-rl-seed{seed}", f"step{job['sft_steps'] + bstep}")
        save_ckpt(p, a.model, st, stored_model, {
            "task": job["task"], "order": job["order"], "H_bits": task.H, "spec": spec, "S_label": S_LABEL[spec],
            "S_bits": st.bits, "phase": "rl", "seed": seed, "sft_steps": job["sft_steps"], "rl_step": bstep,
            "steps": job["sft_steps"] + bstep, "sft_lr": lr, "rl_lr": rl_lr,
            "test_exact": sum(bex) / 512, "test_pos": sum(bfr) / 512, "base_model": a.model,
            "merged_dtype": "bfloat16" if spec == "full" else "float32"})
        rec["rl_ckpt"] = p
    del model, st, stored_model
    torch.cuda.empty_cache()
    return rec


def worker(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    jobs = json.load(open(a.jobs))
    os.makedirs(a.results, exist_ok=True)
    os.makedirs(a.claims, exist_ok=True)
    for job in jobs:
        try:
            fd = os.open(os.path.join(a.claims, job["id"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"{a.rank}\n".encode())
        os.close(fd)
        t0 = time.time()
        try:
            rec = run_job(job, a, tok, device)
            rec["sec"] = round(time.time() - t0, 1)
            rec["worker"] = a.rank
            json.dump(rec, open(os.path.join(a.results, job["id"] + ".json"), "w"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            json.dump({"job": job, "error": repr(e)}, open(os.path.join(a.results, job["id"] + ".error.json"), "w"))
            torch.cuda.empty_cache()


def base(a):
    torch.backends.cuda.matmul.allow_tf32 = True
    out = {}
    for name in a.models.split(","):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda")
        for order in [int(o) for o in a.orders.split(",")]:
            task = Task(order, tok, "cuda")
            test = task.plains(a.n, 800 + order)
            t0 = time.time()
            ex, fr = evaluate(model, None, task, test, False, temp=1.0, k=a.k)
            gex, gfr = evaluate(model, None, task, test, False)
            toks = sample(model, task.prompts(test[:4]), task.T, 1.0, False)
            r = {"model": name, "order": order, "H_bits": task.H, "n_strings": a.n, "k": a.k,
                 "pass_at_k": sum(ex) / len(ex), "n_pass": int(sum(ex)), "pos_acc_sampled": sum(fr) / len(fr),
                 "greedy_exact": sum(gex) / len(gex), "pos_acc_greedy": sum(gfr) / len(gfr),
                 "chance_pass_at_k": 1 - (1 - 26.0 ** -L) ** a.k,
                 "examples": [{"prompt": task.prompt_text(p), "want": " ".join(task.encode(p)),
                               "got": tok.decode(t)} for p, t in zip(test[:4], toks.tolist())],
                 "sec": round(time.time() - t0, 1)}
            print(json.dumps({k: v for k, v in r.items() if k != "examples"}), flush=True)
            out[f"{name}|{order}"] = r
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("base")
    b.add_argument("--models", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct")
    b.add_argument("--orders", default="0,1,2")
    b.add_argument("--n", type=int, default=512)
    b.add_argument("--k", type=int, default=64)
    b.add_argument("--out", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    w.add_argument("--jobs", required=True)
    w.add_argument("--results", required=True)
    w.add_argument("--claims", required=True)
    w.add_argument("--ckpt", default="")
    w.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()
    {"base": base, "worker": worker}[a.mode](a)


if __name__ == "__main__":
    main()
