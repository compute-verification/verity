"""Design D, iterated injection (HARDNESS.md addendum, Daniel 1:04 PM PDT), on train.py's TinyStories transformer.

Every block linear (qkv, o, fc1, fc2) is an int8 linear served as y = s_x·s_w·((q + E)·(A + R)ᵀ):

- q: the input quantized per token to the integers in ±(127 − m_E) (straight-through), s_x its scale;
- E: live noise, uniform on n_E consecutive integers, centered (half-integers when n_E is even; the served circuit
  adds the mean back as a public row-sum term), drawn per element and call;
- A: the developer's codes, round(W/s_w) in ±(127 − m_R) from a float latent W (straight-through), s_w per row;
- R: the public draw, uniform on n_R consecutive integers, centered, m_R = ⌈(n_R − 1)/2⌉ of headroom each side,
  so A + R and q + E stay int8 without clipping.

The embedding, norms and tied head are not hardened. Arithmetic is fp32, or bf16 autocast with ``bf16``: the operands
q + E and A + R are integers or half-integers within ±127.5, which bf16 holds exactly, so only y is rounded (2⁻⁸
relative) before its scales.

Regimes for the fine-tune that follows a clean int8 pretrain (``--base``), each then served with fresh draws R*:

- ``clean``: no noise, no headroom (the stock int8 model: the baseline);
- ``none``: headroom only, trained without noise;
- ``eonly``: E live, no R;
- ``absorbK``: K rounds of: absorb a fresh draw into W (W += R·s_w), then train with E live and no R in the forward;
- ``keepK``: K rounds of: draw R, train A with A + R in the forward and E live (``perstep`` is K = steps);
- ``designA``: y = d₁ ⊙ (x̃·(R₈·s_R)ᵀ) + (x̃·V)·Uᵀ, R₈ uniform int8 frozen, U, V rank r from the base's SVD, d₁ learned.

After the fine-tune (``post``), from the served model under one draw R*: ``d0tune`` trains only the unhardened
parameters (D0 still holds: A is unchanged); ``d2_r`` adds a rank-r adapter (U, V) beside A + R* and trains it with the
unhardened parameters; ``d1`` trains everything, A included, and reports how far A moved against ‖R*‖.

    python iter.py pretrain --data D --out base.json --save base.pt --config '{...}'
    python iter.py run --data D --base base.pt --out run.json --config '{...}'
"""

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as T  # noqa: E402


@dataclasses.dataclass
class ICfg:
    name: str = "run"
    size: str = "w256"
    ctx: int = 256
    batch: int = 32
    tokens: int = 4_000_000
    lr: float = 6e-4
    wd: float = 0.1
    warmup: int = 50
    seed: int = 0
    data_seed: int = 4321
    n_e: int = 1  # values in E's alphabet (1 = no noise)
    n_r: int = 1  # values in R's alphabet (1 = no draw)
    regime: str = "clean"
    rank_frac: float = 0.125  # designA
    d2_ranks: tuple = (16,)  # post: adapter ranks (absolute)
    post: bool = False
    post_tokens: int = 1_000_000
    post_lr: float = 3e-4
    n_draws: int = 3
    eval_batches: int = 32
    eval_every: int = 100
    threads: int = 4
    bf16: bool = False  # autocast: exact on the codes (integers and half-integers within ±127.5), rounds y to bf16


def margin(n):
    """Headroom each side: the centered alphabet's largest magnitude, (n − 1)/2, rounded up."""
    return n // 2


def centered(n, shape, gen):
    """Uniform on n consecutive integers, shifted to mean 0."""
    lo = -(n // 2)
    return torch.randint(lo, lo + n, shape, generator=gen).float() - (lo + (n - 1) / 2)


class Inj:
    n_e = 1
    n_r = 1
    e_on = True
    POOL = 1 << 25
    pool = None
    rng = np.random.default_rng(0)

    @classmethod
    def e(cls, shape):
        if cls.pool is None:
            cls.pool = centered(cls.n_e, (cls.POOL,), torch.Generator().manual_seed(4242 + cls.n_e))
        n = math.prod(shape)
        if n > cls.POOL // 2:
            raise ValueError(f"noise tensor of {n} elements exceeds half the pool")
        off = int(cls.rng.integers(0, cls.POOL - n))
        return cls.pool[off:off + n].view(shape)


class DLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, idx: int, headroom: bool = True):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().float())
        self.register_buffer("R", torch.zeros_like(self.weight))
        self.idx = idx
        self.use_r = False
        self.headroom = headroom
        self.U = self.V = self.d1 = None
        self.design_a = False

    def m_r(self):
        return margin(Inj.n_r) if self.headroom and not self.design_a else 0

    def scale(self):
        return self.weight.detach().abs().amax(1, keepdim=True).clamp_min(1e-8) / (127 - self.m_r())

    def codes(self):
        s = self.scale()
        a = self.weight / s
        lim = 127 - self.m_r()
        return a + (a.round().clamp(-lim, lim) - a).detach(), s

    @torch.no_grad()
    def draw(self, seed):
        if Inj.n_r <= 1:
            self.R.zero_()
            return
        self.R.copy_(centered(Inj.n_r, self.R.shape, torch.Generator().manual_seed(seed)))

    @torch.no_grad()
    def absorb(self, seed):
        """W += R·s_w with R uniform on n_R consecutive integers (not centered: an integer move of the codes)."""
        if Inj.n_r <= 1:
            return
        lo = -(Inj.n_r // 2)
        r = torch.randint(lo, lo + Inj.n_r, self.weight.shape, generator=torch.Generator().manual_seed(seed))
        self.weight.add_(r.float() * self.scale())

    def prep(self, x):
        x = x.float()
        m_e = margin(Inj.n_e)
        s = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-6) / (127 - m_e)
        q = x / s
        q = q + (q.round().clamp(-(127 - m_e), 127 - m_e) - q).detach()
        if Inj.n_e > 1 and Inj.e_on:
            q = q + Inj.e(q.shape)
        return q, s

    def forward(self, x, record=False):
        q, sx = self.prep(x)
        if self.design_a:
            y = F.linear(q, self.R) * (self.d1 * self.s_r.squeeze(1))
        else:
            a, sw = self.codes()
            w = a + self.R if self.use_r else a
            y = F.linear(q, w) * sw.squeeze(1)
        if self.U is not None:
            y = y + F.linear(F.linear(q, self.V), self.U)
        return y * sx

    @torch.no_grad()
    def int_codes(self):
        a, _ = self.codes()
        return a.round()


def build(icfg: ICfg, vocab: int, state=None, headroom=True):
    cfg = T.Cfg(size=icfg.size, ctx=icfg.ctx, batch=icfg.batch, seed=icfg.seed, harden="none", bf16=False)
    model = T.GPT(cfg, vocab)
    idx = 0
    for blk in model.blocks:
        for nm in ("qkv", "o", "fc1", "fc2"):
            old = getattr(blk, nm)
            setattr(blk, nm, DLinear(old.weight, idx, headroom))
            idx += 1
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model, cfg


def dlinears(model):
    return [m for m in model.modules() if isinstance(m, DLinear)]


def draw_all(model, seed):
    for m in dlinears(model):
        m.draw(seed * 1009 + m.idx)
        m.use_r = Inj.n_r > 1


def r_off(model):
    for m in dlinears(model):
        m.use_r = False


@torch.no_grad()
def evaluate(model, val, icfg: ICfg, e_on=True, e_seed=777):
    model.eval()
    saved_rng, Inj.rng = Inj.rng, np.random.default_rng(e_seed)
    saved_e, Inj.e_on = Inj.e_on, e_on
    n_b = icfg.eval_batches
    stride = (len(val) - icfg.ctx - 1) // (n_b * icfg.batch)
    tot, cnt = 0.0, 0
    for b in range(n_b):
        ix = [(b * icfg.batch + j) * stride for j in range(icfg.batch)]
        x = torch.from_numpy(np.stack([val[i:i + icfg.ctx + 1] for i in ix]).astype(np.int64))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=icfg.bf16):
            logits = model(x[:, :-1])
        tot += F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
        cnt += x[:, 1:].numel()
    model.train()
    Inj.rng, Inj.e_on = saved_rng, saved_e
    return tot / cnt


def lr_sched(s, steps, warmup):
    if s < warmup:
        return (s + 1) / warmup
    p = (s - warmup) / max(1, steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def make_opt(params_named, lr, wd):
    mats = [p for n, p in params_named if p.requires_grad and p.dim() >= 2]
    vecs = [p for n, p in params_named if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW([{"params": mats, "weight_decay": wd}, {"params": vecs, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)


def train_loop(model, opt, tr, val, icfg: ICfg, steps, lr, hook=None, log="", data_seed=None):
    seed = icfg.data_seed if data_seed is None else data_seed
    it = T.batches(tr, T.Cfg(ctx=icfg.ctx, batch=icfg.batch), np.random.default_rng(seed))
    params = [p for g in opt.param_groups for p in g["params"]]
    curve, ema, t0 = [], None, time.time()
    for s in range(steps):
        if hook is not None:
            hook(s)
        for g in opt.param_groups:
            g["lr"] = lr * lr_sched(s, steps, icfg.warmup)
        x, y = next(it)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=icfg.bf16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        lv = loss.item()
        if not math.isfinite(lv):
            raise FloatingPointError(f"{log}: loss {lv} at step {s}")
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if (s + 1) % icfg.eval_every == 0 or s + 1 == steps:
            el = time.time() - t0
            curve.append({"step": s + 1, "train_ema": ema, "wall_s": el})
            print(f"{log} step {s + 1}/{steps} train {ema:.4f} {(s + 1) * icfg.batch * icfg.ctx / el:.0f} tok/s",
                  flush=True)
    return curve


def load_data(data: Path):
    meta = json.loads((data / "meta.json").read_text())
    tr = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")
    return meta, tr, val


def pretrain(icfg: ICfg, data: Path, save: Path):
    torch.set_num_threads(icfg.threads)
    torch.manual_seed(icfg.seed)
    Inj.n_e = Inj.n_r = 1
    meta, tr, val = load_data(data)
    model, _ = build(icfg, meta["vocab"], headroom=False)
    steps = max(1, icfg.tokens // (icfg.batch * icfg.ctx))
    opt = make_opt(list(model.named_parameters()), icfg.lr, icfg.wd)
    t0 = time.time()
    curve = train_loop(model, opt, tr, val, icfg, steps, icfg.lr, log="pretrain", data_seed=icfg.data_seed - 1)
    torch.save(model.state_dict(), save)
    res = {"cfg": dataclasses.asdict(icfg), "steps": steps, "tokens": steps * icfg.batch * icfg.ctx,
           "val_clean": evaluate(model, val, icfg, e_on=False), "curve": curve, "wall_s": time.time() - t0,
           "params": sum(p.numel() for p in model.parameters()),
           "hardened_params": sum(m.weight.numel() for m in dlinears(model))}
    return res


def code_stats(model):
    tot = sq = sat = n = 0.0
    for m in dlinears(model):
        if m.design_a:
            continue
        c = m.int_codes()
        lim = 127 - m.m_r()
        sq += float((c ** 2).sum())
        sat += float((c.abs() >= lim).sum())
        n += c.numel()
    return {"code_rms": math.sqrt(sq / max(n, 1)), "code_sat_frac": sat / max(n, 1)}


def served_evals(model, val, icfg: ICfg, prefix=""):
    out = {}
    r_off(model)
    out[prefix + "val_clean"] = evaluate(model, val, icfg, e_on=False)
    if Inj.n_e > 1:
        out[prefix + "val_e"] = evaluate(model, val, icfg, e_on=True)
    if Inj.n_r > 1:
        d0 = []
        for i in range(icfg.n_draws):
            draw_all(model, 10_000 + i)
            d0.append(evaluate(model, val, icfg, e_on=True))
        draw_all(model, 10_000)
        out[prefix + "val_r"] = evaluate(model, val, icfg, e_on=False)
        r_off(model)
        out[prefix + "val_d0_draws"] = d0
        out[prefix + "val_d0"] = float(np.mean(d0))
    else:
        out[prefix + "val_d0"] = out.get(prefix + "val_e", out[prefix + "val_clean"])
    return out


def to_design_a(model, icfg: ICfg):
    d = T.SIZES[icfg.size][0]
    r = max(1, round(icfg.rank_frac * d))
    for m in dlinears(model):
        m.design_a = True
        with torch.no_grad():
            a, s = m.codes()
            w = (a * s)
            Uf, S, Vh = torch.linalg.svd(w, full_matrices=False)
            g = torch.Generator().manual_seed(20_000 + m.idx)
            m.R.copy_(centered(256, m.R.shape, g))
            m.register_buffer("s_r", s.clone())
        # the adapter acts on the int8 activation codes q, so absorb the activation scale later (y·s_x)
        m.U = nn.Parameter((Uf[:, :r] * S[:r]).contiguous())
        m.V = nn.Parameter(Vh[:r].contiguous())
        m.d1 = nn.Parameter(torch.full((m.weight.shape[0],), 0.1))
        m.weight.requires_grad_(False)
        m.use_r = False


def run(icfg: ICfg, data: Path, base: Path):
    torch.set_num_threads(icfg.threads)
    torch.manual_seed(icfg.seed)
    Inj.n_e, Inj.n_r = icfg.n_e, icfg.n_r
    Inj.rng = np.random.default_rng(icfg.seed + 99)
    meta, tr, val = load_data(data)
    state = torch.load(base, map_location="cpu")
    headroom = icfg.regime != "clean"
    model, _ = build(icfg, meta["vocab"], state=state, headroom=headroom)
    t0 = time.time()
    res = {"cfg": dataclasses.asdict(icfg)}
    if icfg.regime == "designA":
        for m in dlinears(model):
            m.design_a = False
            m.headroom = False
    res["base_evals"] = {"val_clean_headroom": evaluate(model, val, icfg, e_on=False)}
    if Inj.n_r > 1 and icfg.regime != "designA":
        res["base_evals"].update(served_evals(model, val, icfg, prefix="untrained_"))
    steps = max(1, icfg.tokens // (icfg.batch * icfg.ctx))
    reg = icfg.regime
    Inj.e_on = reg not in ("clean", "none")
    hook = None
    if reg.startswith("absorb") or reg.startswith("keep") or reg == "perstep":
        k = steps if reg == "perstep" else int(reg.removeprefix("absorb").removeprefix("keep"))
        every = max(1, steps // k)
        absorb = reg.startswith("absorb")

        def hook(s):
            if s % every:
                return
            rnd = s // every
            for m in dlinears(model):
                if absorb:
                    m.absorb(rnd * 1009 + m.idx + 1)
                else:
                    m.draw(rnd * 1009 + m.idx + 1)
                    m.use_r = Inj.n_r > 1
    if reg == "designA":
        to_design_a(model, icfg)
    opt = make_opt([(n, p) for n, p in model.named_parameters() if p.requires_grad], icfg.lr, icfg.wd)
    res["curve"] = train_loop(model, opt, tr, val, icfg, steps, icfg.lr, hook=hook, log=icfg.name)
    res["steps"], res["tokens"] = steps, steps * icfg.batch * icfg.ctx
    Inj.e_on = True
    if reg == "designA":
        res.update(design_a_evals(model, val, icfg))
    else:
        res.update(served_evals(model, val, icfg))
        res.update(code_stats(model))
    if icfg.post and Inj.n_r > 1 and reg != "designA":
        res["post"] = post_draw(model, tr, val, icfg)
    res["wall_s"] = time.time() - t0
    res["torch"] = torch.__version__
    return res


def design_a_evals(model, val, icfg: ICfg):
    out = {"val_d0": evaluate(model, val, icfg, e_on=True), "val_clean": evaluate(model, val, icfg, e_on=False)}
    share = []
    for m in dlinears(model):
        with torch.no_grad():
            r_part = (m.R * (m.d1 * m.s_r.squeeze(1))[:, None]).norm()
            uv = (m.U @ m.V).norm()
            share.append(float(r_part / (r_part + uv)))
            m.d1_rms = float(m.d1.pow(2).mean().sqrt())
    out["r_norm_share"] = float(np.mean(share))
    out["d1_rms"] = float(np.mean([m.d1_rms for m in dlinears(model)]))
    return out


def post_draw(model, tr, val, icfg: ICfg):
    steps = max(1, icfg.post_tokens // (icfg.batch * icfg.ctx))
    out = {"steps": steps}
    draw_all(model, 10_000)
    out["val_d0_draw0"] = evaluate(model, val, icfg, e_on=True)
    a0 = {m.idx: m.int_codes() for m in dlinears(model)}
    r_norm = math.sqrt(sum(float((m.R ** 2).sum()) for m in dlinears(model)))

    arms = [("d0tune", None, False)] + [(f"d2_r{r}", r, False) for r in icfg.d2_ranks] + [("d1", None, True)]
    for name, rank, move_a in arms:
        m2 = copy.deepcopy(model)
        for p in m2.parameters():
            p.requires_grad_(False)
        for n, p in m2.named_parameters():
            if not any(n.endswith(s) for s in (".weight",)) or not isinstance(_owner(m2, n), DLinear):
                p.requires_grad_(True)
        for m in dlinears(m2):
            m.weight.requires_grad_(move_a)
            if rank:
                g = torch.Generator().manual_seed(30_000 + m.idx)
                m.V = nn.Parameter(torch.randn(rank, m.weight.shape[1], generator=g) / math.sqrt(m.weight.shape[1]))
                m.U = nn.Parameter(torch.zeros(m.weight.shape[0], rank))
        opt = make_opt([(n, p) for n, p in m2.named_parameters() if p.requires_grad], icfg.post_lr, icfg.wd)
        train_loop(m2, opt, tr, val, icfg, steps, icfg.post_lr, log=f"{icfg.name}/{name}",
                   data_seed=icfg.data_seed + 1)
        rec = {"val_d0_draw0": evaluate(m2, val, icfg, e_on=True),
               "trainable": sum(p.numel() for p in m2.parameters() if p.requires_grad)}
        if move_a:
            moved = math.sqrt(sum(float(((m.int_codes() - a0[m.idx]) ** 2).sum()) for m in dlinears(m2)))
            changed = sum(float((m.int_codes() != a0[m.idx]).sum()) for m in dlinears(m2))
            total = sum(m.weight.numel() for m in dlinears(m2))
            rec["rho_fro"] = moved / r_norm
            rec["codes_changed_frac"] = changed / total
        out[name] = rec
    return out


def _owner(model, pname):
    mod = model
    for part in pname.split(".")[:-1]:
        mod = getattr(mod, part)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pretrain", "run"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    raw = json.loads(a.config)
    if "d2_ranks" in raw:
        raw["d2_ranks"] = tuple(raw["d2_ranks"])
    icfg = ICfg(**raw)
    if a.cmd == "pretrain":
        res = pretrain(icfg, Path(a.data), Path(a.save))
    else:
        res = run(icfg, Path(a.data), Path(a.base))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"final {icfg.name}: " + json.dumps({k: v for k, v in res.items() if k.startswith("val")}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
