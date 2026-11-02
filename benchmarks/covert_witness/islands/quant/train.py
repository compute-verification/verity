"""Island `quant` (direction, not a class): how far an update compresses without losing the loss it bought. It runs in any
class (`--klass`; islands/<class>/train.py on this branch is a link here) and trains an update, then encodes it in low-bit
CWQUAN1 (`qformat.py`: k-bit codes, f16 codebooks and per-group scales, entropy coded).

Modes:

- `ptq`: the reference recipe (Muon on the trained matrices, AdamW on the rest) on `--train`'s tensors, then each of
  `--variants` quantizes the update Δ = W − θ;
- `qat`: every trained matrix's weight is θ + Q(V) in each step's forward, Q a sign, ternary or k-bit quantizer (scaled
  by the mean |V| of a tensor, row or group) of a latent delta V that only the optimizer touches, through a
  straight-through gradient; so every step's weights, and the trained model, are exactly a CWQUAN1 code. `--qat sparse`
  is ternary on the `--qat-keep` share of each group with the largest |V| (scaled by their mean |V|), the rest zero; a
  sparse submission over its class's E cap is re-cut from its latent at a smaller share until it fits (`cut_to_keep`).
  `--qat fp` trains the weights unquantized, as ptq does;
- `qlora`: LoRA whose factors are k-bit in every forward (straight-through), E their codes.

`--arms` (qat, qlora) packs settings into one candidate: each arm trains from θ on the same data order with an equal
share of the budget, in its own 5%s of the run, so a later arm's checkpoints are charged every earlier arm's MACs.

With `--emit quan` (it needs prepare.encode_quant, which writes qformat.encode's bytes), every quantized checkpoint and
the submission are CWQUAN1; with `--emit full` (the default) they are the model through `encode_full`, and the record's
notes give its CWQUAN1 size, checked to rebuild that model bit for bit. `--xz 9e` re-encodes the submitted CWQUAN1 at xz
preset 9e, `--xz min` at the smallest of `qformat.XZ_TRIES` (`qformat.recompress`; the checkpoints stay
prepare.encode_quant's preset 1). The step count is a multiple of the
harness's 20 checkpoints (`hooks.plan`). In ptq the harness scores the fp model after each 5% (`hooks.output`), and right
after it the program encodes one variant of the same weights, which the harness scores as that 5%'s checkpoint: the
variants cycle back from the last 5% (`--variants`' non-fp ones, the first at 100%), and the notes' `buckets` say which
variant each checkpoint holds. The submission is `--variants`' first, and the notes give every variant on the final
weights. In qat and qlora each 5%'s checkpoint is the model itself, except a `--qat fp` arm's, which are ptq's.

Variant names: `fp` (no quantization), `rtn{b}g{G}` (b-bit round to nearest, G-entry groups, G 0 a row), `sign[g{G}]`,
`tern{tau}[g{G}]` (ternary, kept when |Δ| > tau · the group's mean |Δ|), `svd{r}[q{b}]` (each matrix's update at its best
rank r, a CWQUAN1 lora entry whose factors are b-bit, 8 by default, in groups of 128).

    python benchmarks/covert_witness/launch.py reference s100 --note "ptq: ..." -- --klass reference --mode ptq \
        --variants fp,rtn2g128,sign
    python benchmarks/covert_witness/launch.py reference s100 --note "qat: ..." -- --klass reference --mode qat \
        --arms "qat=fp/qat=tern/qat=sign"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from te_core import hooks

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare  # noqa: E402
import qformat as qf  # noqa: E402

#: the harness's checkpoint count (CHECKPOINTS in te_core's run.py; its program screen lets a program import only hooks
#: and codecs)
CHECKPOINTS = 20


def arm_steps(n: int) -> int:
    """An arm's step count, cut to a multiple of the harness's checkpoints."""
    return n // CHECKPOINTS * CHECKPOINTS if n >= CHECKPOINTS else n


class Schedule:
    """The run's steps over its arms, told to the harness (`hooks.plan`), and which of them encode an E: the harness scores
    the first E in each 5% of the run, so each 5% gets one, at its last step (where it also scores the output) or at an
    arm's last step, whichever comes first."""

    def __init__(self, counts: list[int]):
        self.counts, self.total = counts, sum(counts)
        hooks.plan(self.total)
        self.cps = {max(0, math.ceil(self.total * b / CHECKPOINTS) - 1) for b in range(1, CHECKPOINTS + 1)}
        self.claimed: set[int] = set()

    def bucket(self, g: int) -> int:
        return math.ceil((g + 1) * CHECKPOINTS / self.total)

    def due(self, g: int, last: bool) -> bool:
        b = self.bucket(g)
        if (g in self.cps or last) and b not in self.claimed:
            self.claimed.add(b)
            return True
        return False


#: what an arm may set (`--arms`); the rest is the run's
ARM_KEYS = {"qat", "qat_group", "qat_mult", "qat_keep", "muon_lr", "adam_lr", "warmup", "data_seed", "targets", "rank", "qbits", "pow2",
            "norms", "lr", "norm_lr", "seed"}


def parse_arms(a) -> list[argparse.Namespace]:
    """`--arms A/B/...`, each a comma-separated key=value list over the command line's options: one Namespace per arm."""
    if not a.arms:
        return [a]
    out = []
    for spec in a.arms.split("/"):
        x = argparse.Namespace(**vars(a))
        for kv in filter(None, spec.split(",")):
            k, _, v = kv.partition("=")
            k = k.replace("-", "_")
            if k not in ARM_KEYS:
                raise SystemExit(f"quant: an arm can't set {k!r} (it may set {sorted(ARM_KEYS)})")
            cur = getattr(a, k)
            setattr(x, k, v.lower() in ("1", "true", "yes") if isinstance(cur, bool) else type(cur)(v))
        x.arm = spec
        out.append(x)
    return out


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    # CPU bf16 matmuls occasionally return 2**24-sized entries here (CPU smokes only)
    x = g.bfloat16() if g.is_cuda else g.float()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


def ns_flops(shape: tuple[int, int], steps: int = 5) -> int:
    m, n = min(shape), max(shape)
    return steps * (4 * m * m * n + 2 * m ** 3)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                p.add_(u, alpha=-g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


# ---- variants ----------------------------------------------------------------------------------------------------------

VARIANT = re.compile(r"fp|rtn(\d)g(\d+)|sign(?:g(\d+))?|tern([\d.]+)(?:g(\d+))?|svd(\d+)(?:q(\d))?")


def quantizer(name: str):
    m = VARIANT.fullmatch(name)
    if not m:
        raise SystemExit(f"quant: unknown variant {name!r}")
    if name == "fp":
        return None
    if name.startswith("rtn"):
        return lambda x: qf.rtn(x, int(m.group(1)), int(m.group(2)))
    if name.startswith("sign"):
        return lambda x: qf.sign(x, int(m.group(3) or 0))
    if name.startswith("svd"):
        return lambda x: svd_entry(x, int(m.group(6)), int(m.group(7) or 8))
    return lambda x: qf.ternary(x, float(m.group(4)), int(m.group(5) or 0))


@torch.no_grad()
def svd_entry(d: torch.Tensor, rank: int, bits: int) -> dict:
    """The update's best rank-r approximation as a CWQUAN1 lora entry: B = U_r S_r^1/2, A = S_r^1/2 V_r^T, each k-bit round
    to nearest in groups of 128 (a row when it is shorter)."""
    u, s, vh = torch.linalg.svd(d.float(), full_matrices=False)
    r = min(rank, s.numel())
    root = s[:r].sqrt()
    a, b = root[:, None] * vh[:r], u[:, :r] * root[None, :]

    def q(x):
        return qf.rtn(x.contiguous(), bits, 128 if x.numel() % 128 == 0 else 0)

    return {"kind": "lora", "a": q(a), "b": q(b), "scale": 1.0}


def entries_of(name: str, delta: dict[str, torch.Tensor], norm_bits: int) -> dict[str, dict]:
    quant = quantizer(name)
    out = {}
    for k, d in delta.items():
        if d.ndim == 1:
            out[k] = {"kind": "delta", "q": qf.rtn(d, norm_bits, 0)}
        elif name.startswith("svd"):
            out[k] = quant(d)
        else:
            out[k] = {"kind": "delta", "q": quant(d)}
    return out


def entry_value(w: torch.Tensor, e: dict) -> torch.Tensor:
    """W' of one CWQUAN1 entry, on W's device in W's dtype, as qformat.decode computes it."""
    if e["kind"] == "lora":
        return prepare.lora_merge(w, e["a"].value().to(w.device), e["b"].value().to(w.device), e["scale"])
    return (w.float() + e["q"].value().to(w.device)).to(w.dtype)


def entry_bits(e: dict) -> float:
    return e["a"].entropy_bits() + e["b"].entropy_bits() if e["kind"] == "lora" else e["q"].entropy_bits()


# ---- training ----------------------------------------------------------------------------------------------------------

def select(params: dict, rec: dict, spec: str) -> list[str]:
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    if spec == "all":
        return sorted(params)
    if spec == "blocks":
        return sorted(n for n in params if n != embed)
    if spec == "blockmats":
        return sorted(n for n, p in params.items() if n != embed and p.ndim == 2)
    top = re.fullmatch(r"top(\d+)", spec)
    if top:
        low = rec["config"]["n_layer"] - int(top.group(1))
        layer = re.compile(r"(?:model\.layers|blocks)\.(\d+)\.")
        return sorted(n for n in params if n != embed and (not layer.match(n) or int(layer.match(n).group(1)) >= low))
    pats = spec.split(",")
    return sorted(n for n in params if any(re.search(p, n) for p in pats))


QAT_SCHEME = re.compile(r"sign|tern|sparse|rtn(\d)")
#: a qat arm that trains the weights unquantized; its checkpoints encode `--variants` of them, as ptq's do
QAT_FP = "fp"


@torch.no_grad()
def qat_quant(v: torch.Tensor, scheme: str, group: int, mult: float, keep: float = 1.0) -> qf.QArray:
    """Q(V) for QAT: a scale of `mult` times the group's mean |V| (f16), then sign, ternary (round(V/s) in -1..1) or k-bit
    round to nearest; a group of zeros quantizes to zeros. `sparse`: sign(V) on the ceil(keep · group) entries of largest
    |V| (ties at the threshold kept, zeros never), at `mult` times their mean |V|."""
    g = v.numel() if group < 0 else qf.groups_of(v, group)
    vg = v.reshape(-1, g)
    if scheme == "sparse":
        a = vg.abs()
        n_keep = min(g, max(1, math.ceil(keep * g)))
        kept = (a >= a.kthvalue(g - n_keep + 1, dim=1, keepdim=True).values) & (a > 0)
        s = qf._f16(mult * (a * kept).sum(1) / kept.sum(1).clamp(min=1))
        k = torch.where(kept, torch.where(vg >= 0, 2, 0), 1).to(torch.uint8)
        return qf.QArray(torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=torch.float16), s, k.view(-1), 2, g, tuple(v.shape))
    s = qf._f16(mult * vg.abs().mean(1))
    sf = s.float()
    if scheme == "sign":
        codes = (vg >= 0).to(torch.uint8)
        return qf.QArray(torch.tensor([-1.0, 1.0], dtype=torch.float16), s, codes.view(-1), 1, g, tuple(v.shape))
    bits = 2 if scheme == "tern" else int(QAT_SCHEME.fullmatch(scheme).group(1))
    qmax = 2 ** (bits - 1) - 1
    k = torch.clamp(torch.round(vg / torch.where(sf > 0, sf, 1.0)[:, None]), -qmax, qmax) * (sf > 0)[:, None]
    cb = torch.zeros(2 ** bits)
    cb[: 2 * qmax + 1] = torch.arange(-qmax, qmax + 1, dtype=torch.float32)
    return qf.QArray(cb.to(torch.float16), s, (k + qmax).to(torch.uint8).view(-1), bits, g, tuple(v.shape))


def check_decoder(theta: dict, name: str) -> None:
    """Refuse `--declare` and `--emit quan` unless prepare encodes CWQUAN1 byte for byte as qformat.encode does and
    prepare.decode reads it as qformat.decode does."""
    q = qat_quant(torch.randn(theta[name].shape, generator=torch.Generator().manual_seed(0)), "tern", -1, 1.0)
    entries, one = {name: {"kind": "delta", "q": q}}, {name: theta[name]}
    blob = qf.encode(entries, one)
    if not hasattr(prepare, "encode_quant"):
        raise SystemExit("quant: --declare and --emit quan need prepare.encode_quant (CWQUAN1 in prepare)")
    if prepare.encode_quant(entries, one) != blob:
        raise SystemExit("quant: prepare.encode_quant writes CWQUAN1 differently from islands/quant/qformat.py")
    try:
        got = prepare.decode(one, blob, device="cpu")
    except ValueError as e:
        raise SystemExit(f"quant: --declare and --emit quan need prepare.decode to read CWQUAN1 ({e})")
    if not prepare.same_bits(got[name].cpu(), qf.decode(one, blob)[name]):
        raise SystemExit("quant: prepare.decode reads CWQUAN1 differently from islands/quant/qformat.py")


def encode_quan(entries: dict, theta: dict) -> bytes:
    """CWQUAN1 through prepare's encoder when it has one (the same bytes as qformat.encode, `check_decoder`)."""
    return prepare.encode_quant(entries, theta) if hasattr(prepare, "encode_quant") else qf.encode(entries, theta)


LF_MATS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
HF_MATS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_targets(rec: dict, params: dict, spec: str) -> list[str]:
    lf = rec["config"]["family"] == "lf"
    want = (set(LF_MATS if lf else HF_MATS) if spec == "all" else {"wo", "w_down", "o_proj", "down_proj"} if spec == "out"
            else set(spec.split(",")))
    out = []
    for n, p in params.items():
        if p.ndim != 2 or n in ("embed", "model.embed_tokens.weight"):
            continue
        leaf = n.split(".")[-2] if n.endswith(".weight") else n.split(".")[-1]
        if leaf in want:
            out.append(n)
    return sorted(out)


B_ROWS = 64


@torch.no_grad()
def factor_quant(x: torch.Tensor, bits: int, pow2: bool, rows: int = 1) -> qf.QArray:
    """Absmax round to nearest at `bits`, one scale per `rows` rows (A: per row; B, d_out x r: per 64 rows, since a scale
    per row of r entries costs 16/r bits an entry); with `pow2` each scale is a power of two, so every value is an
    integer times a power of two and exact in bf16 (the fixed `lora` format carries it unchanged)."""
    qmax = 2 ** (bits - 1) - 1
    shape = tuple(x.shape)
    rows = rows if x.shape[0] % rows == 0 else 1
    x = x.reshape(-1, rows * x.shape[1])
    amax = x.float().abs().amax(1)
    s = amax / qmax
    if pow2:
        s = torch.where(s > 0, torch.exp2(torch.ceil(torch.log2(s.clamp(min=2.0 ** -24)))), 0.0)
    s = qf._f16(s)
    sf = s.float()
    k = torch.clamp(torch.round(x.float() / torch.where(sf > 0, sf, 1.0)[:, None]), -qmax, qmax) * (sf > 0)[:, None]
    cb = torch.zeros(2 ** bits)
    cb[: 2 * qmax + 1] = torch.arange(-qmax, qmax + 1, dtype=torch.float32)
    return qf.QArray(cb.to(torch.float16), s, (k + qmax).to(torch.uint8).view(-1), bits, x.shape[1], shape)


def lora_bytes(fac: dict, full: dict) -> int:
    """`prepare.encode_lora(fac, "bf16", full)`'s size without encoding: every encode during the steps is a checkpoint
    candidate."""
    header = {"dtype": "bf16", "factors": [[n, int(fac[n][0].shape[0]), float(fac[n][2])] for n in sorted(fac)],
              "full": sorted(full)}
    payload = 2 * (sum(a.numel() + b.numel() for a, b, _ in fac.values()) + sum(t.numel() for t in full.values()))
    return len(prepare._pack(b"CWLORA1\n", header, b"")) + payload


def main_qlora(a, arms: list[argparse.Namespace]) -> int:
    """LoRA whose factors are k-bit in every forward (straight-through fake quantization; `--qbits 0` is plain LoRA); E is
    the factors' codes. Each arm trains from θ on the same data order; the submission is the arm with the least final
    training loss (an exponential average over its last ~50 steps)."""
    t_start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    dev = prepare.DEVICE
    params = net.params()
    if a.emit == "quan":
        check_decoder(theta, min((n for n, p in params.items() if p.ndim == 2), key=lambda n: params[n].numel()))
    for p in params.values():
        p.requires_grad_(False)
    all_norms = sorted(n for n, p in params.items() if p.ndim == 1)
    for n in all_norms:
        params[n].data = params[n].data.float()
    norm0 = {n: params[n].detach().clone() for n in all_norms}
    tokens = a.batch * prepare.EVAL_CTX

    def trainable_of(x):
        return {"full": all_norms if x.norms else [], "adapters": {n: x.rank for n in lora_targets(rec, params, x.targets)}}

    def per_step(x):
        return prepare.train_flops(rec, trainable_of(x), [{"op": "train", "tokens": tokens}])["flops"]

    sched = Schedule([arm_steps(int(a.budget_fraction / len(arms) * prepare.TRAIN_FLOP_BUDGET // per_step(x)))
                      for x in arms])
    fwd = torch.compile(net)
    steps, results, g = [], [], 0

    def pieces(x, factors, norms):
        """(factors as E carries them, norm gains in bf16, CWQUAN1 entries or None)."""
        if x.qbits:
            q = {n: (factor_quant(A.detach(), x.qbits, x.pow2), factor_quant(B.detach(), x.qbits, x.pow2, B_ROWS), s)
                 for n, (A, B, s) in factors.items()}
            fac = {n: (qa.value(), qb.value(), s) for n, (qa, qb, s) in q.items()}
        else:
            q, fac = None, {n: (A.detach().clone(), B.detach().clone(), s) for n, (A, B, s) in factors.items()}
        full = {n: params[n].detach().to(torch.bfloat16).float() for n in norms}
        qent = None if q is None else ({n: {"kind": "lora", "a": qa, "b": qb, "scale": s} for n, (qa, qb, s) in q.items()}
                                       | {n: {"kind": "full", "t": t} for n, t in full.items()})
        return fac, full, qent

    def emit(fac, full, qent):
        if a.emit == "lora":
            return prepare.encode_lora(fac, "bf16", full)
        if a.emit == "quan":
            return encode_quan(qent, theta)
        return prepare.encode_full({k: v for k, v in qf.decode(theta, qf.encode(qent, theta), dev).items()
                                    if k in qent})

    t0 = time.perf_counter()
    for k, x in enumerate(arms):
        tr = trainable_of(x)
        targets, norms = sorted(tr["adapters"]), tr["full"]
        if k and results[-1]["norms"]:
            with torch.no_grad():
                for n in all_norms:
                    params[n].copy_(norm0[n])
        for n in all_norms:
            params[n].requires_grad_(n in norms)
        gen = torch.Generator(device=dev).manual_seed(x.seed)
        factors, handles = {}, []

        def fq(t, rows=1, x=x):
            return t if not x.qbits else t + (factor_quant(t.detach(), x.qbits, x.pow2, rows).value() - t).detach()

        for n in targets:
            w = params[n]
            A = torch.nn.Parameter(torch.randn(x.rank, w.shape[1], device=dev, generator=gen) / math.sqrt(w.shape[1]))
            B = torch.nn.Parameter(torch.zeros(w.shape[0], x.rank, device=dev))
            s = math.sqrt(w.shape[1] / x.rank)
            factors[n] = (A, B, s)
            mod = net.inner.get_submodule(n.removesuffix(".weight"))
            handles.append(mod.register_forward_hook(
                lambda m, inp, out, A=A, B=B, s=s, fq=fq: out + s * F.linear(F.linear(inp[0], fq(A)), fq(B, B_ROWS))))
        opt = torch.optim.AdamW(
            [{"params": [p for A, B, _ in factors.values() for p in (A, B)], "lr": x.lr, "weight_decay": 0.0},
             {"params": [params[n] for n in norms], "lr": x.norm_lr, "weight_decay": 0.0}],
            betas=(0.9, 0.95), fused=dev.startswith("cuda"))
        data = prepare.TrainData(rec["tokenizer"], seed=x.data_seed)
        n_k, first, ema = sched.counts[k], g, None
        print(json.dumps({"arm": k, "spec": getattr(x, "arm", ""), "steps": n_k, "rank": x.rank, "bits": x.qbits,
                          "targets": len(targets), "norms": len(norms), "flops_per_step": per_step(x)}), flush=True)
        lrs = [grp["lr"] for grp in opt.param_groups]
        for step in range(n_k):
            frac = min(1.0, (step + 1) / x.warmup) * (1 - step / n_k)
            for grp, lr in zip(opt.param_groups, lrs):
                grp["lr"] = lr * frac
            rows = data.next(a.batch)
            total = 0.0
            for i in range(0, a.batch, a.micro):
                xb = rows[i: i + a.micro]
                with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                    logits = fwd(xb[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), xb[:, 1:].reshape(-1)) * (
                    len(xb) / a.batch)
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps.append({"op": "train", "tokens": tokens})
            ema = total if ema is None else 0.98 * ema + 0.02 * total
            if (step + 1) % 100 == 0:
                print(json.dumps({"arm": k, "step": step + 1, "loss": total, "ema": ema,
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
            if sched.due(g, step == n_k - 1):
                emit(*pieces(x, factors, norms))
            g += 1
        results.append({"x": x, "k": k, "steps": [first, g - 1], "loss": ema, "targets": targets, "norms": norms,
                        "trainable": tr, "pieces": pieces(x, factors, norms),
                        "n_entries": sum(A.numel() + B.numel() for A, B, _ in factors.values())
                        + sum(params[n].numel() for n in norms)})
        for h in handles:
            h.remove()
        del factors, opt
    t_train = time.perf_counter() - t0
    best = min(results, key=lambda r: r["loss"])
    blob = emit(*best["pieces"])
    model = prepare.decode(theta, blob)
    arm_notes = []
    for r in results:
        t1, x = time.perf_counter(), r["x"]
        fac, full, qent = r["pieces"]
        if qent is None:
            qblob, q_bytes, ok, h_bits = None, lora_bytes(fac, full), True, 16.0 * r["n_entries"]
        else:
            qblob = qf.encode(qent, theta)
            q_bytes = len(qblob)
            ok = r is not best or all(prepare.same_bits(v, model[n]) for n, v in qf.decode(theta, qblob, dev).items())
            h_bits = sum(e["a"].entropy_bits() + e["b"].entropy_bits() for e in qent.values() if e["kind"] == "lora") \
                + 16 * sum(t.numel() for t in full.values())
        arm_notes.append(variant_note(f"qlora-r{x.rank}-q{x.qbits}{'-pow2' if x.pow2 and x.qbits else ''}", qblob, q_bytes,
                                      h_bits, ok, 0.0, r["n_entries"], t1, arm=r["k"], spec=getattr(x, "arm", ""),
                                      steps=r["steps"], final_loss_ema=r["loss"], lora_bf16_bytes=lora_bytes(fac, full),
                                      where="submission" if r is best else "arm"))
    notes = {"direction": "quant", "mode": "qlora", "args": vars(a), "steps": sched.total, "arms": arm_notes,
             "variants": [n for n in arm_notes if n["where"] == "submission"], "train_seconds": round(t_train, 1),
             "process_seconds": round(time.perf_counter() - t_start, 1),
             "E_format_note": "q_bytes is the arm's CWQUAN1 size (islands/quant/qformat.py; the bf16 lora size for qbits "
                              "0), checked to rebuild the submitted model bit for bit; bits_per_param is per trained entry"}
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, best["trainable"], steps,
                   changed=best["targets"] + best["norms"], notes=notes)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--klass", required=True, choices=sorted(prepare.CLASSES))
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("ptq", "qat", "qlora"), default="ptq")
    ap.add_argument("--targets", default="all", help="qlora: all | out | leaf names")
    ap.add_argument("--rank", type=int, default=16, help="qlora")
    ap.add_argument("--qbits", type=int, default=4, help="qlora: the factors' bits")
    ap.add_argument("--pow2", action="store_true", help="qlora: power-of-two scales (exact in the bf16 lora format)")
    ap.add_argument("--norms", action=argparse.BooleanOptionalAction, default=True, help="qlora: train the norm gains")
    ap.add_argument("--lr", type=float, default=1e-3, help="qlora: the factors' AdamW rate")
    ap.add_argument("--norm-lr", type=float, default=3e-4, help="qlora")
    ap.add_argument("--seed", type=int, default=0, help="qlora: the factors' initialization")
    ap.add_argument("--train", default="all",
                    help="all | blocks | blockmats | top{K} (the top K blocks and the final norm) | name regexes, comma-separated")
    ap.add_argument("--variants", default="fp,rtn4g128,rtn3g128,rtn2g128,sign,tern0.7", help="ptq")
    ap.add_argument("--norm-bits", type=int, default=8, help="ptq: the norm gains' rtn bits")
    ap.add_argument("--qat", default="tern", help="qat: sign | tern | rtn{b} | fp (unquantized; its checkpoints encode "
                                                  "--variants, as ptq's do)")
    ap.add_argument("--qat-group", type=int, default=-1, help="qat: -1 one scale per tensor, 0 per row, else G entries")
    ap.add_argument("--qat-mult", type=float, default=1.0, help="qat: the scale over the group's mean |V|")
    ap.add_argument("--qat-keep", type=float, default=0.05, help="qat sparse: the share of each group kept (nonzero)")
    ap.add_argument("--xz", choices=("1", "9e", "min"), default="1",
                    help="--emit quan: the submitted E's xz preset (9e, or min: the smallest of qformat.XZ_TRIES, through "
                         "qformat.recompress; checkpoints stay preset 1)")
    ap.add_argument("--declare", action="store_true",
                    help="qat: declare each step's weights as one CWQUAN1 code from θ (prepare.declare_step), for the "
                         "harness to credit; needs prepare.decode to read CWQUAN1. Put fp arms last: their steps aren't "
                         "declared")
    ap.add_argument("--emit", choices=("full", "quan", "lora"), default="full",
                    help="E's format: encode_full, CWQUAN1, or (qlora --pow2) the bf16 lora format")
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--arms", default="",
                    help="qat, qlora: A/B/..., each a comma-separated key=value list over these options (e.g. "
                         "qat=sign,muon_lr=0.005): each arm trains from θ on the same data order with an equal share of "
                         "--budget-fraction, and a later arm's checkpoints are charged every earlier arm's MACs")
    a = ap.parse_args()
    arms = parse_arms(a)
    if a.mode == "qlora":
        for x in arms:
            if x.qbits and a.emit == "lora" and not x.pow2:
                raise SystemExit("quant: --emit lora carries k-bit factors exactly only with pow2")
            if not x.qbits and a.emit != "lora":
                raise SystemExit("quant: qbits 0 (plain LoRA) is emitted as --emit lora")
        return main_qlora(a, arms)
    if a.emit == "lora":
        raise SystemExit("quant: ptq and qat emit full or quan")
    if a.mode == "ptq" and len(arms) > 1:
        raise SystemExit("quant: --arms is qat's and qlora's (ptq's variants already share one training)")
    if a.declare and a.mode != "qat":
        raise SystemExit("quant: --declare is qat's")
    for x in arms:
        if a.mode == "qat" and x.qat != QAT_FP and not QAT_SCHEME.fullmatch(x.qat):
            raise SystemExit(f"quant: unknown qat scheme {x.qat!r}")
    t_start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    dev = prepare.DEVICE
    params = net.params()
    names = select(params, rec, a.train)
    for p in params.values():
        p.requires_grad_(False)
    for n in names:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    mats = [n for n in names if params[n].ndim == 2 and n != embed]
    rest = [n for n in names if n not in mats]
    if a.declare or a.emit == "quan":
        check_decoder(theta, min(mats, key=lambda n: params[n].numel()))
    theta_dev = {n: theta[n].to(dev) for n in names}
    trainable = {"full": names}
    tokens = a.batch * prepare.EVAL_CTX
    extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    n_arm = arm_steps(int(a.budget_fraction / len(arms) * prepare.TRAIN_FLOP_BUDGET // per_step))
    sched = Schedule([n_arm] * len(arms))
    fwd = torch.compile(net)
    steps, buckets, results, g = [], [], [], 0
    cycle = [v for v in a.variants.split(",") if v != "fp"] or ["fp"]
    if a.mode == "ptq" or any(x.qat == QAT_FP for x in arms):
        hooks.output(lambda: {n: p.detach() for n, p in params.items()})
    t0 = time.perf_counter()
    for k, x in enumerate(arms):
        if k:
            with torch.no_grad():
                for n in names:
                    params[n].copy_(theta_dev[n])
        # qat: the quantized tensors (every trained matrix) train a latent delta V; their weights are θ + Q(V), rebuilt
        # each step
        quantized = [n for n in names if params[n].ndim == 2] if a.mode == "qat" and x.qat != QAT_FP else []
        latent = {n: torch.zeros_like(params[n], dtype=torch.float32, device=dev) for n in quantized}
        opt_of = {n: latent.get(n, params[n]) for n in names}
        muon = Muon([opt_of[n] for n in mats], lr=x.muon_lr) if mats else None
        adam = torch.optim.AdamW([opt_of[n] for n in rest], lr=x.adam_lr, betas=(0.9, 0.95), weight_decay=0.0,
                                 fused=dev.startswith("cuda")) if rest else None
        opts = [(o, lr) for o, lr in ((muon, x.muon_lr), (adam, x.adam_lr)) if o is not None]
        data = prepare.TrainData(rec["tokenizer"], seed=x.data_seed)
        qarrays, first, ema = {}, g, None
        print(json.dumps({"arm": k, "spec": getattr(x, "arm", ""), "steps": n_arm, "flops_per_step": per_step,
                          "trained": len(names), "quantized": len(quantized)}), flush=True)

        @torch.no_grad()
        def materialize(x=x, latent=latent, qarrays=qarrays, quantized=quantized) -> None:
            for n in quantized:
                q = qat_quant(latent[n], x.qat, x.qat_group, x.qat_mult, x.qat_keep)
                qarrays[n] = q
                params[n].data.copy_((theta_dev[n].float() + q.value()).to(theta_dev[n].dtype))

        for step in range(n_arm):
            frac = min(1.0, (step + 1) / x.warmup) * (1 - step / n_arm)
            for o, lr in opts:
                for grp in o.param_groups:
                    grp["lr"] = lr * frac
            rows = data.next(a.batch)
            if a.declare and qarrays:
                prepare.declare_step(encode_quan(arm_entries(qarrays, params, names), theta))
            total = 0.0
            for i in range(0, a.batch, a.micro):
                xb = rows[i: i + a.micro]
                with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                    logits = fwd(xb[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), xb[:, 1:].reshape(-1)) * (
                    len(xb) / a.batch)
                loss.backward()
                total += float(loss.detach())
            for n in quantized:
                latent[n].grad, params[n].grad = params[n].grad, None
            torch.nn.utils.clip_grad_norm_([opt_of[n] for n in names], 1.0)
            for o, _ in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            if quantized:
                materialize()
            steps.append({"op": "train", "tokens": tokens})
            if extra:
                steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
            ema = total if ema is None else 0.98 * ema + 0.02 * total
            if (step + 1) % 100 == 0:
                print(json.dumps({"arm": k, "step": step + 1, "loss": total, "ema": ema,
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
            if sched.due(g, step == n_arm - 1):
                if quantized and a.emit == "quan":
                    encode_quan(arm_entries(qarrays, params, names), theta)
                elif quantized:
                    prepare.encode_full({n: params[n].detach().to(theta[n].dtype).cpu() for n in names})
                else:
                    b = sched.bucket(g)
                    v = cycle[(CHECKPOINTS - b) % len(cycle)]
                    model_v, note, qblob = ptq_variant(v, {n: params[n].detach() for n in names}, theta, a.norm_bits,
                                                       dev, encode_quan if a.emit == "quan" else qf.encode)
                    if a.emit != "quan" or qblob is None:
                        prepare.encode_full(model_v)
                    buckets.append(note | {"bucket": b, "step": g, "arm": k})
                    del model_v
            g += 1
        results.append({"x": x, "k": k, "steps": [first, g - 1], "loss": ema, "qarrays": qarrays,
                        "rest": {n: params[n].detach().cpu().clone() for n in names if n not in qarrays},
                        "latent": {n: v.cpu() for n, v in latent.items()} if x.qat == "sparse" else None})
        del latent, opt_of, muon, adam, opts
    t_train = time.perf_counter() - t0
    if a.mode == "qat":
        blob, model, notes_v = finish_qat(a, results, theta, names, cycle, dev)
    else:
        blob, model, notes_v = finish_ptq(a, theta, params, names, cycle, dev)
    notes = {"direction": "quant", "mode": a.mode, "args": vars(a), "steps": sched.total,
             "trained_parameters": sum(params[n].numel() for n in names), "variants": notes_v, "buckets": buckets,
             "train_seconds": round(t_train, 1), "process_seconds": round(time.perf_counter() - t_start, 1),
             "E_format_note": "E is encode_full of the model, or with --emit quan the update's CWQUAN1 bytes "
                              "(prepare.encode_quant); q_bytes is the update's CWQUAN1 size (islands/quant/qformat.py), "
                              "checked to rebuild that model bit for bit. ptq: "
                              "each 5% checkpoint scores the fp model (source output) and, on the same weights, the "
                              "bucket's variant (source encode; `buckets`). qat: one note per arm, the submission the "
                              "arm with the least final training loss"}
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=names, notes=notes)
    return 0


def variant_note(v: str, qblob: bytes | None, q_bytes: int, h_bits: float, ok: bool, rel: float, n_params: int, t1: float,
                 **more) -> dict:
    out = {"variant": v, "q_bytes": q_bytes, "bits_per_param": 8 * q_bytes / n_params,
           "order0_entropy_bits_per_param": h_bits / n_params, "rebuilds_bit_for_bit": ok, "relative_sq_error": rel,
           "seconds": round(time.perf_counter() - t1, 1), **more}
    print(json.dumps(out), flush=True)
    return out


def code_blob(a, qarrays: dict, rest: dict, theta) -> tuple[bytes, int]:
    """A QAT arm's CWQUAN1 (its codes, the rest of its trained tensors in full) at `--xz`, and its size at preset 1."""
    entries = {n: {"kind": "delta", "q": q} for n, q in qarrays.items()}
    entries |= {n: {"kind": "full", "t": t} for n, t in rest.items()}
    qblob = (encode_quan if a.emit == "quan" else qf.encode)(entries, theta)
    return (qblob if a.xz == "1" else qf.recompress(qblob, search=a.xz == "min")), len(qblob)


def finish_qat(a, results: list[dict], theta, names, cycle, dev):
    """Each arm's CWQUAN1 note (an fp arm's: `--variants` of its final weights); the submission is the quantized arm with
    the least final training loss (any arm, if none is quantized), as encode_full, or with `--emit quan` a quantized
    arm's CWQUAN1."""
    best = min([r for r in results if r["qarrays"]] or results, key=lambda r: r["loss"])
    notes, blob = [], None
    for r in results:
        t1, qarrays = time.perf_counter(), r["qarrays"]
        if not qarrays:
            weights = {n: t.to(dev) for n, t in r["rest"].items()}
            for v in ["fp", *cycle]:
                m, note, _ = ptq_variant(v, weights, theta, a.norm_bits, dev)
                notes.append(note | {"arm": r["k"], "spec": getattr(r["x"], "arm", ""), "steps": r["steps"],
                                     "final_loss_ema": r["loss"], "where": "submission" if r is best and v == "fp"
                                     else "arm, final weights"})
                if r is best and v == "fp":
                    blob = prepare.encode_full(m)
                del m
            continue
        qblob, xz1_bytes = code_blob(a, qarrays, r["rest"], theta)
        # a sparse submission over its class's cap is cut from its latent, its kept share scaled by cap / |E| (at least 3%
        # smaller) at a time
        cap, keep = prepare.CLASSES[a.klass]["max_e_bytes"], r["x"].qat_keep
        while r is best and a.emit == "quan" and cap and r["latent"] and len(qblob) > cap and keep > 1e-5:
            keep *= min(0.97, cap / len(qblob))
            qarrays = {n: qat_quant(v.to(dev), "sparse", r["x"].qat_group, r["x"].qat_mult, keep)
                       for n, v in r["latent"].items()}
            qblob, xz1_bytes = code_blob(a, qarrays, r["rest"], theta)
        rebuilt = qf.decode(theta, qblob, dev)
        trained = {n: (theta[n].to(dev).float() + q.value()).to(theta[n].dtype).cpu() for n, q in qarrays.items()}
        trained |= {n: t.to(theta[n].dtype) for n, t in r["rest"].items()}
        ok = all(prepare.same_bits(rebuilt[n], trained[n]) for n in names)
        n_params = sum(theta[n].numel() for n in names)
        h_bits = sum(q.entropy_bits() for q in qarrays.values()) + sum(32 * t.numel() for t in r["rest"].values())
        q_n = sum(theta[n].numel() for n in qarrays)
        nonzero = sum(float((q.codebook.float()[q.codes.long().cpu()] != 0).float().sum()) for q in qarrays.values())
        notes.append(variant_note(f"qat-{r['x'].qat}", qblob, len(qblob), h_bits, ok, 0.0, n_params, t1, arm=r["k"],
                                  spec=getattr(r["x"], "arm", ""), steps=r["steps"], final_loss_ema=r["loss"],
                                  nonzero_share=nonzero / max(1, q_n), xz=a.xz, q_bytes_xz1=xz1_bytes,
                                  cut_to_keep=keep if keep != r["x"].qat_keep else None,
                                  where="submission" if r is best else "arm"))
        if r is best:
            blob = qblob if a.emit == "quan" else prepare.encode_full(trained)
        del rebuilt, trained
    return blob, prepare.decode(theta, blob), notes


def ptq_variant(v: str, weights: dict[str, torch.Tensor], theta: dict, norm_bits: int, dev,
                encoder=qf.encode) -> tuple[dict, dict, bytes | None]:
    """The model (θ's dtypes, on the CPU) that variant `v` makes of the update W − θ, its note, and its CWQUAN1 blob
    (None for fp) by `encoder`."""
    t1 = time.perf_counter()
    n_params = sum(w.numel() for w in weights.values())
    if v == "fp":
        model = {n: w.to(theta[n].dtype).cpu() for n, w in weights.items()}
        q_bytes = sum(t.numel() * t.element_size() for t in model.values())
        return model, variant_note(v, None, q_bytes, 8.0 * q_bytes, True, 0.0, n_params, t1), None
    delta = {n: w.float() - theta[n].to(dev).float() for n, w in weights.items()}
    entries = entries_of(v, delta, norm_bits)
    h_bits = sum(entry_bits(e) for e in entries.values())
    qblob = encoder(entries, theta)
    rebuilt = qf.decode(theta, qblob, dev)
    model = {n: rebuilt[n] for n in weights}
    want = {n: entry_value(theta[n].to(dev), entries[n]).cpu() for n in weights}
    ok = all(prepare.same_bits(model[n], want[n]) for n in weights)
    num = sum(float(((model[n].to(dev).float() - theta[n].to(dev).float() - delta[n]) ** 2).sum()) for n in weights)
    rel = num / max(1e-30, sum(float((d ** 2).sum()) for d in delta.values()))
    return model, variant_note(v, qblob, len(qblob), h_bits, ok, rel, n_params, t1), qblob


def finish_ptq(a, theta, params, names, cycle, dev):
    """The submission (`--variants`' first: encode_full, or its CWQUAN1 with `--emit quan`) and, on the final weights,
    every variant's note."""
    weights = {n: params[n].detach() for n in names}
    order = [a.variants.split(",")[0]] + [v for v in cycle if v != a.variants.split(",")[0]]
    blob, notes_v = None, []
    for v in order:
        m, note, qblob = ptq_variant(v, weights, theta, a.norm_bits, dev, encode_quan if a.emit == "quan" else qf.encode)
        notes_v.append(note | {"where": "submission" if blob is None else "final weights, not submitted"})
        if blob is None:
            blob = qblob if a.emit == "quan" and qblob is not None else prepare.encode_full(m)
        del m
    return blob, prepare.decode(theta, blob), notes_v


def arm_entries(qarrays: dict, params: dict, names: list[str]) -> dict[str, dict]:
    """A QAT arm's weights as CWQUAN1 entries: each quantized matrix's delta code, every other trained tensor in full."""
    return ({n: {"kind": "delta", "q": q} for n, q in qarrays.items()}
            | {n: {"kind": "full", "t": params[n].detach()} for n in names if n not in qarrays})


if __name__ == "__main__":
    raise SystemExit(main())
