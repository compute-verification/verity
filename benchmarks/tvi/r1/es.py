"""R1 of the training-vs-inference program: forward-only training whose entire state is K seed coefficients.

Task: A3's secret substitution cipher (`benchmarks/tvi/a3/newskill.py`, imported unchanged): prompt, key, validation and
test strings are A3's, so the numbers sit beside A3's S grid.

Arm `es` (FedKSeed-style evolution strategies, arXiv 2312.06353 and 2509.24372). A pool of K seeds; seed j names a
Gaussian direction z_j over every parameter of the model (embeddings included). The model is always
    theta = theta_0 + sum_j a_j z_j,
so the whole trained state is the K fp32 coefficients a (K = 625, 2,500, 25,000 is 2.5 kB, 10 kB, 100 kB). Each iteration
draws P distinct seeds from the pool, evaluates theta +- sigma z_j (antithetic, 2P = population), z-scores the 2P rewards
and moves a_j += alpha (Z+_j - Z-_j) / (2P). theta is held as an fp32 master and the forward runs in bf16; the final test
is on theta rebuilt from a alone.

Reward (dense partial credit): fraction of the 8 cipher letters right in a sampled completion (T = 1), averaged over
B prompts x G samples. The samples use common random numbers (the same Gumbel noise for every member of an iteration),
so members differ only through their weights. Pass/fail can't move from 0; this can.
`--reward tf` instead scores each member by the teacher-forced expected partial credit, the mean over positions of
p(key letter | right prefix): one forward per member and no sampling noise, the objective FedKSeed and MeZO use.

Arm `grpo` (the control): A3's GRPO (group-relative advantages on the same reward, AdamW, no KL), full fine-tuning
from the base model with no SFT, under the same wall-clock budget.

Both arms log the information the reward could deliver: sum over episodes of h(p_t) (pass/fail) and of
min(log2 9, 8 h(q_t)) (partial credit), with p_t, q_t the batch's pass and per-letter rates (h concave: upper bounds).
Compute is counted as tokens through the model: forward 2N FLOPs per token, forward and backward 6N.
"""
import argparse, calendar, copy, json, math, os, random, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newskill as ns  # noqa: E402

LOG2_9 = math.log2(9)


def h2(p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def deadline_of(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%MZ"))


class Counter:
    def __init__(self, n_params):
        self.N, self.fwd_tok, self.bwd_tok, self.episodes = n_params, 0, 0, 0
        self.info_pass, self.info_dense = 0.0, 0.0

    def gen(self, n_seq, P, T):
        self.fwd_tok += n_seq * (P + T - 1)

    def train(self, n_seq, P, T):
        self.bwd_tok += n_seq * (P + T)

    def reward(self, ex, frac):
        n = len(ex)
        p, q = sum(ex) / n, sum(frac) / n
        self.episodes += n
        self.info_pass += n * h2(p)
        self.info_dense += n * min(LOG2_9, ns.L * h2(q))

    def flops(self):
        return 2 * self.N * self.fwd_tok + 6 * self.N * self.bwd_tok

    def dump(self):
        return {"fwd_tokens": self.fwd_tok, "fwd_bwd_tokens": self.bwd_tok, "flops": self.flops(), "episodes": self.episodes,
                "info_pass_bits": self.info_pass, "info_dense_bits": self.info_dense}


@torch.no_grad()
def rollout(model, prompt, T, gen, temp=1.0):
    """Sampled completion by Gumbel-max with noise from `gen` (the same stream for every member: common random numbers)."""
    out = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    past, logits = out.past_key_values, out.logits[:, -1].float()
    toks = []
    for t in range(T):
        u = torch.rand(logits.shape, generator=gen, device=logits.device).clamp_(1e-20, 1.0)
        nxt = (logits / temp - torch.log(-torch.log(u))).argmax(-1)
        toks.append(nxt)
        if t + 1 < T:
            out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True, logits_to_keep=1)
            past, logits = out.past_key_values, out.logits[:, -1].float()
    return torch.stack(toks, 1)


@torch.no_grad()
def tf_reward(model, ids, P):
    """Teacher-forced expected partial credit: per sequence, the mean over the 8 positions of p(key letter | right prefix),
    and the probability of the whole exact answer. One forward per member; no sampling noise."""
    T = ids.shape[1] - P
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=T + 1).logits[:, :-1].float()
    lp = torch.gather(logits, 2, ids[:, P:, None]).squeeze(-1) - torch.logsumexp(logits, -1)
    return lp.sum(1).exp(), lp[:, :ns.L].exp().mean(1)


def token_reward(task, toks, tgt):
    """Per sequence (exact, fraction of letters right), on token ids: position i is right iff it is the key's letter token."""
    right = (toks[:, :ns.L] == tgt[:, :ns.L]).float()
    frac = right.mean(1)
    ex = (right.sum(1) == ns.L) & (toks[:, ns.L] == tgt[:, ns.L])
    return ex.float(), frac


@torch.no_grad()
def eval_k(model, task, plains, k, temp, chunk=4096):
    """Per string: number of exact passes among k samples, and mean letter accuracy (A3's text parse, `Task.score`)."""
    model.eval()
    npass, pos = [], []
    per = max(1, chunk // k)
    for i in range(0, len(plains), per):
        ps = plains[i:i + per]
        rep = [p for p in ps for _ in range(k)]
        toks = ns.sample(model, task.prompts(rep), task.T, temp, False)
        ex, frac = task.score(rep, toks)
        for j in range(len(ps)):
            npass.append(int(sum(ex[j * k:(j + 1) * k])))
            pos.append(sum(frac[j * k:(j + 1) * k]) / k)
    return npass, pos


def test_report(model, task, test, k=64):
    gex, gpos = ns.evaluate(model, None, task, test, False)
    npass, spos = eval_k(model, task, test, k, 1.0)
    n = len(test)
    return {"greedy_exact": gex, "greedy_pos": gpos, "sampled_npass": npass, "sampled_pos": spos, "k": k,
            "pass1_greedy": sum(gex) / n, "pos_greedy": sum(gpos) / n, "pass1_sampled": sum(c / k for c in npass) / n,
            f"pass{k}": sum(1.0 for c in npass if c > 0) / n, "pos_sampled": sum(spos) / n}


def summary(r):
    return {k: v for k, v in r.items() if not isinstance(v, list)}


# ---------------------------------------------------------------- ES / FedKSeed


class Seeded:
    """theta = theta_0 + sum_j a_j z_j over a pool of K seeds; the K fp32 coefficients `a` are the entire trained state."""

    def __init__(self, model, K, pool_seed):
        self.params = list(model.parameters())
        self.master = [p.detach().float().clone() for p in self.params]
        self.K, self.pool_seed = K, pool_seed
        self.a = torch.zeros(K, dtype=torch.float32)
        self.gen = torch.Generator(device=self.params[0].device)

    def z(self, j):
        self.gen.manual_seed(self.pool_seed * 1_000_003 + j)
        for m in self.master:
            yield torch.randn(m.shape, generator=self.gen, device=m.device, dtype=torch.float32)

    @torch.no_grad()
    def load(self, j=None, s=0.0):
        if j is None:
            for p, m in zip(self.params, self.master):
                p.copy_(m)
        else:
            for p, m, z in zip(self.params, self.master, self.z(j)):
                p.copy_(m.add(z, alpha=s))

    @torch.no_grad()
    def step(self, j, da):
        da = float(torch.tensor(da, dtype=torch.float32))
        for m, z in zip(self.master, self.z(j)):
            m.add_(z, alpha=da)
        self.a[j] = (self.a[j] + da)

    @torch.no_grad()
    def rebuild(self, model_name, a):
        """theta_0 + sum_j a_j z_j from the coefficients alone (fresh theta_0), into the working model."""
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.master = [p.detach().to(self.params[0].device).float().clone() for p in base.parameters()]
        del base
        for j in torch.nonzero(a).flatten().tolist():
            for m, z in zip(self.master, self.z(j)):
                m.add_(z, alpha=float(a[j]))
        self.load()


def run_es(a, tok, task, val, test, out):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    st = Seeded(model, a.K, a.pool_seed)
    cnt = Counter(n_params)
    rng = random.Random(a.seed * 7919 + a.K)
    gen = torch.Generator(device="cuda")
    P = a.pairs
    deadline = deadline_of(a.deadline)
    log, t0 = [], time.time()
    vex, vpos = ns.evaluate(model, None, task, val, False)
    best = {"it": 0, "val_exact": sum(vex) / len(vex), "val_pos": sum(vpos) / len(vpos), "a": st.a.clone()}
    log.append({"it": 0, "val_exact": best["val_exact"], "val_pos": best["val_pos"], "t": 0.0, **cnt.dump()})
    print(json.dumps(log[-1]), flush=True)
    it = 0
    while time.time() < deadline and (a.iters <= 0 or it < a.iters):
        it += 1
        plains = task.plains(a.batch, rng.getrandbits(48))
        rep = [p for p in plains for _ in range(a.group)]
        if a.reward == "tf":
            ids = task.batch(rep)
        else:
            prompt = task.prompts(rep)
            tgt = torch.tensor([task.target_ids(p) for p in rep], device="cuda")
        js = rng.sample(range(a.K), P)
        gseed = rng.getrandbits(62)
        R, EX = [], []
        for j in js:
            for s in (a.sigma, -a.sigma):
                st.load(j, s)
                if a.reward == "tf":
                    ex, frac = tf_reward(model, ids, task.P)
                    cnt.fwd_tok += ids.numel()
                else:
                    gen.manual_seed(gseed)
                    toks = rollout(model, prompt, task.T, gen)
                    ex, frac = token_reward(task, toks, tgt)
                    cnt.gen(len(rep), task.P, task.T)
                cnt.reward(ex.tolist(), frac.tolist())
                R.append(frac.mean().item())
                EX.append(ex.mean().item())
        Rt = torch.tensor(R, dtype=torch.float64)
        Z = (Rt - Rt.mean()) / (Rt.std() + 1e-12) if Rt.std() > 0 else torch.zeros_like(Rt)
        for i, j in enumerate(js):
            st.step(j, a.alpha * (Z[2 * i] - Z[2 * i + 1]).item() / (2 * P))
        st.load()
        rec = {"it": it, "reward_mean": float(Rt.mean()), "reward_max": float(Rt.max()), "reward_std": float(Rt.std()),
               "exact_mean": sum(EX) / len(EX), "a_norm": float(st.a.norm()), "t": round(time.time() - t0, 1)}
        if it % a.eval_every == 0:
            vex, vpos = ns.evaluate(model, None, task, val, False)
            rec["val_exact"], rec["val_pos"] = sum(vex) / len(vex), sum(vpos) / len(vpos)
            if (rec["val_exact"], rec["val_pos"]) > (best["val_exact"], best["val_pos"]):
                best = {"it": it, "val_exact": rec["val_exact"], "val_pos": rec["val_pos"], "a": st.a.clone()}
            torch.save({"a": st.a.clone(), "best_a": best["a"], "best_it": best["it"], "it": it}, os.path.join(out, "state.pt"))
        rec.update(cnt.dump())
        log.append(rec)
        if it % a.eval_every == 0 or it <= 3:
            print(json.dumps(rec), flush=True)
            json.dump(log, open(os.path.join(out, "log.json"), "w"))
    train_sec = time.time() - t0
    res = {"arm": "es", "reward": a.reward, "task": a.task, "order": task.order, "H_bits": task.H, "model": a.model, "K": a.K,
           "S_bits": 32 * a.K, "S_bytes": 4 * a.K, "sigma": a.sigma, "alpha": a.alpha, "pairs": P, "population": 2 * P,
           "batch": a.batch, "group": a.group, "seed": a.seed, "pool_seed": a.pool_seed, "iters": it,
           "train_sec": round(train_sec, 1), "n_params": n_params, "best_it": best["it"], "best_val_exact": best["val_exact"],
           "best_val_pos": best["val_pos"], "counter": cnt.dump(),
           "seqs_per_iter": 2 * P * a.batch * a.group}
    torch.save({"a": st.a.clone(), "best_a": best["a"], "best_it": best["it"], "it": it, "K": a.K,
                "pool_seed": a.pool_seed, "model": a.model}, os.path.join(out, "state.pt"))
    for name, coeff in (("final", st.a.clone()), ("best", best["a"])):
        if name == "best" and best["it"] == it:
            res["best"] = res["final"]
            continue
        t1 = time.time()
        st.rebuild(a.model, coeff)
        r = test_report(model, task, test)
        r["rebuild_and_test_sec"] = round(time.time() - t1, 1)
        res[name] = r
        print(name, json.dumps(summary(r)), flush=True)
    json.dump(log, open(os.path.join(out, "log.json"), "w"))
    return res


# ---------------------------------------------------------------- GRPO control (A3's grpo, from base, timed)


def run_grpo(a, tok, task, val, test, out):
    model = ns.load_model(a.model, "full", "cuda")
    st = ns.State(model, "full", a.seed, "cuda")
    n_params = sum(p.numel() for p in model.parameters())
    cnt = Counter(n_params)
    opt = torch.optim.AdamW(st.params, lr=a.lr, weight_decay=0.0)
    rng = random.Random(a.seed * 104729 + 3)
    B, G = a.batch, a.group
    deadline = deadline_of(a.deadline)
    log, t0 = [], time.time()

    def val_eval():
        stored = copy.deepcopy(model).to(torch.bfloat16)
        vex, vpos = ns.evaluate(stored, None, task, val, False)
        del stored
        return sum(vex) / len(vex), sum(vpos) / len(vpos)

    ve, vp = val_eval()
    best = {"step": 0, "val_exact": ve, "val_pos": vp, "sd": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    log.append({"step": 0, "val_exact": ve, "val_pos": vp, "t": 0.0, **cnt.dump()})
    print(json.dumps(log[-1]), flush=True)
    step = 0
    while time.time() < deadline and (a.iters <= 0 or step < a.iters):
        step += 1
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1.0, step / 10)
        plains = ["".join(rng.choice(ns.LETTERS) for _ in range(ns.L)) for _ in range(B)]
        rep = [p for p in plains for _ in range(G)]
        prompt = task.prompts(rep)
        model.eval()
        toks = ns.sample(model, prompt, task.T, 1.0, True)
        ex, frac = task.score(rep, toks)
        cnt.gen(len(rep), task.P, task.T)
        cnt.reward(ex, frac)
        r = torch.tensor(frac, device=prompt.device).view(B, G)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
        model.train()
        logp = ns.lm_logp(model, torch.cat([prompt, toks], 1), task.P, True)
        loss = -(adv[:, None] * logp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(st.params, 1.0)
        opt.step()
        cnt.train(len(rep), task.P, task.T)
        rec = {"step": step, "reward_mean": sum(frac) / len(frac), "exact_mean": sum(ex) / len(ex),
               "t": round(time.time() - t0, 1)}
        if step % a.eval_every == 0:
            rec["val_exact"], rec["val_pos"] = val_eval()
            if (rec["val_exact"], rec["val_pos"]) > (best["val_exact"], best["val_pos"]):
                best = {"step": step, "val_exact": rec["val_exact"], "val_pos": rec["val_pos"],
                        "sd": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        rec.update(cnt.dump())
        log.append(rec)
        if step % a.eval_every == 0 or step <= 3:
            print(json.dumps(rec), flush=True)
            json.dump(log, open(os.path.join(out, "log.json"), "w"))
    train_sec = time.time() - t0
    opt.zero_grad(set_to_none=True)
    del opt
    res = {"arm": "grpo", "task": a.task, "order": task.order, "H_bits": task.H, "model": a.model, "spec": "full",
           "S_bits": 16 * n_params, "S_bytes": 2 * n_params, "lr": a.lr, "batch": B, "group": G, "seed": a.seed,
           "iters": step, "train_sec": round(train_sec, 1), "n_params": n_params, "best_it": best["step"],
           "best_val_exact": best["val_exact"], "best_val_pos": best["val_pos"], "counter": cnt.dump()}
    stored = copy.deepcopy(model).to(torch.bfloat16)
    res["final"] = test_report(stored, task, test)
    print("final", json.dumps(summary(res["final"])), flush=True)
    del stored
    model.load_state_dict(best.pop("sd"))
    stored = copy.deepcopy(model).to(torch.bfloat16)
    res["best"] = test_report(stored, task, test)
    print("best", json.dumps(summary(res["best"])), flush=True)
    json.dump(log, open(os.path.join(out, "log.json"), "w"))
    return res


def run_base(a, tok, task, val, test, out):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda")
    r = test_report(model, task, test)
    print("base", json.dumps(summary(r)), flush=True)
    return {"arm": "base", "task": a.task, "order": task.order, "H_bits": task.H, "model": a.model, "S_bits": 0,
            "final": r, "best": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["es", "grpo", "base"])
    ap.add_argument("--task", default="A")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--K", type=int, default=2500)
    ap.add_argument("--pool-seed", type=int, default=20261024)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=5e-4)
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--reward", choices=["sample", "tf"], default="sample")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--deadline", default="2026-10-24T22:55Z")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    order = {"A": 0, "B": 1, "C": 2}[a.task]
    task = ns.Task(order, tok, "cuda")
    val = task.plains(256, 900 + order)
    test = task.plains(512, 800 + order)
    torch.manual_seed(a.seed)
    res = {"es": run_es, "grpo": run_grpo, "base": run_base}[a.arm](a, tok, task, val, test, a.out)
    res["args"] = vars(a)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"))


if __name__ == "__main__":
    main()
