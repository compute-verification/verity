"""Island `quant` (direction, not a class): how far an update compresses without losing the loss it bought. It runs in any
class (`--klass`; islands/<class>/train.py on this branch is a link here) and trains an update, then encodes it in low-bit
CWQUAN1 (`qformat.py`: k-bit codes, f16 codebooks and per-group scales, entropy coded).

Modes:

- `ptq`: the reference recipe (Muon on the trained matrices, AdamW on the rest) on `--train`'s tensors, then each of
  `--variants` quantizes the update Δ = W − θ;
- `qat`: every trained matrix's weight is θ + Q(V) in each step's forward, Q a sign, ternary or k-bit quantizer (scaled
  by the mean |V| of a tensor, row or group) of a latent delta V that only the optimizer touches, through a
  straight-through gradient; so every step's weights, and the trained model, are exactly a CWQUAN1 code.

Until prepare.decode reads CWQUAN1 (asked of the toolsmith), E is the variant's model through `encode_full` (`--emit
full`), and the record's notes give each variant's CWQUAN1 size, checked to rebuild that model bit for bit. The first
variant is the submission; the others are snapshots at the full step log, so the evaluator scores each one.

Variant names: `fp` (no quantization), `rtn{b}g{G}` (b-bit round to nearest, G-entry groups, G 0 a row), `sign[g{G}]`,
`tern{tau}[g{G}]` (ternary, kept when |Δ| > tau · the group's mean |Δ|).

    python guard.py islands/quant/train.py --klass reference --start s100 --out SUB --mode ptq --variants fp,rtn2g128,sign
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare  # noqa: E402
import qformat as qf  # noqa: E402


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
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

VARIANT = re.compile(r"fp|rtn(\d)g(\d+)|sign(?:g(\d+))?|tern([\d.]+)(?:g(\d+))?")


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
    return lambda x: qf.ternary(x, float(m.group(4)), int(m.group(5) or 0))


def entries_of(name: str, delta: dict[str, torch.Tensor], norm_bits: int) -> dict[str, dict]:
    quant = quantizer(name)
    out = {}
    for k, d in delta.items():
        if d.ndim == 1:
            out[k] = {"kind": "delta", "q": qf.rtn(d, norm_bits, 0)}
        else:
            out[k] = {"kind": "delta", "q": quant(d)}
    return out


# ---- training ----------------------------------------------------------------------------------------------------------

def select(params: dict, rec: dict, spec: str) -> list[str]:
    embed = "embed" if rec["config"]["family"] == "lf" else "model.embed_tokens.weight"
    if spec == "all":
        return sorted(params)
    if spec == "blocks":
        return sorted(n for n in params if n != embed)
    if spec == "blockmats":
        return sorted(n for n, p in params.items() if n != embed and p.ndim == 2)
    pats = spec.split(",")
    return sorted(n for n in params if any(re.search(p, n) for p in pats))


QAT_SCHEME = re.compile(r"sign|tern|rtn(\d)")


@torch.no_grad()
def qat_quant(v: torch.Tensor, scheme: str, group: int, mult: float) -> qf.QArray:
    """Q(V) for QAT: a scale of `mult` times the group's mean |V| (f16), then sign, ternary (round(V/s) in -1..1) or k-bit
    round to nearest; a group of zeros quantizes to zeros."""
    g = v.numel() if group < 0 else qf.groups_of(v, group)
    vg = v.reshape(-1, g)
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


def main_qlora(a) -> int:
    """LoRA whose factors are k-bit in every forward (straight-through fake quantization); E is the factors' codes."""
    t_start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    dev = prepare.DEVICE
    params = net.params()
    targets = lora_targets(rec, params, a.targets)
    norms = sorted(n for n, p in params.items() if p.ndim == 1) if a.norms else []
    for p in params.values():
        p.requires_grad_(False)
    for n in norms:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    factors = {}

    def fq(x, rows=1):
        return x + (factor_quant(x.detach(), a.qbits, a.pow2, rows).value() - x).detach()

    for n in targets:
        w = params[n]
        A = torch.nn.Parameter(torch.randn(a.rank, w.shape[1], device=dev, generator=g) / math.sqrt(w.shape[1]))
        B = torch.nn.Parameter(torch.zeros(w.shape[0], a.rank, device=dev))
        s = math.sqrt(w.shape[1] / a.rank)
        factors[n] = (A, B, s)
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], fq(A)), fq(B, B_ROWS)))
    opt = torch.optim.AdamW([{"params": [p for A, B, _ in factors.values() for p in (A, B)], "lr": a.lr, "weight_decay": 0.0},
                             {"params": [params[n] for n in norms], "lr": a.norm_lr, "weight_decay": 0.0}],
                            betas=(0.9, 0.95), fused=dev.startswith("cuda"))
    trainable = {"full": norms, "adapters": {n: a.rank for n in targets}}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net)
    steps, snaps = [], []
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)}

    def entries():
        out = {n: {"kind": "lora", "a": factor_quant(A.detach(), a.qbits, a.pow2),
                   "b": factor_quant(B.detach(), a.qbits, a.pow2, B_ROWS), "scale": s} for n, (A, B, s) in factors.items()}
        out |= {n: {"kind": "full", "t": params[n].detach().to(torch.bfloat16).float()} for n in norms}
        return out

    def emit(ent):
        if a.emit == "lora":
            fac = {n: (e["a"].value(), e["b"].value(), e["scale"]) for n, e in ent.items() if e["kind"] == "lora"}
            return prepare.encode_lora(fac, "bf16", {n: e["t"] for n, e in ent.items() if e["kind"] == "full"})
        qblob = qf.encode(ent, theta)
        return qblob if a.emit == "quan" else prepare.encode_full(
            {k: v for k, v in qf.decode(theta, qblob, dev).items() if k in ent})

    print(json.dumps({"steps": n_steps, "rank": a.rank, "bits": a.qbits, "targets": len(targets), "norms": len(norms),
                      "flops_per_step": per_step}), flush=True)
    t0 = time.perf_counter()
    lrs = [grp["lr"] for grp in opt.param_groups]
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for grp, lr in zip(opt.param_groups, lrs):
            grp["lr"] = lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((emit(entries()), len(steps)))
    t_train = time.perf_counter() - t0
    t1 = time.perf_counter()
    ent = entries()
    qblob = qf.encode(ent, theta)
    blob = emit(ent)
    model = prepare.decode(theta, blob)
    rebuilt = qf.decode(theta, qblob, dev)
    ok = all(prepare.same_bits(rebuilt[k], model[k]) for k in theta)
    n_entries = sum(A.numel() + B.numel() for A, B, _ in factors.values()) + sum(params[n].numel() for n in norms)
    h_bits = sum(e["a"].entropy_bits() + e["b"].entropy_bits() for e in ent.values() if e["kind"] == "lora") + sum(
        32 * params[n].numel() for n in norms)
    note = variant_note(f"qlora-r{a.rank}-q{a.qbits}{'-pow2' if a.pow2 else ''}", qblob, len(qblob), h_bits, ok, 0.0,
                        n_entries, t1, where="submission", lora_bf16_bytes=len(prepare.encode_lora(
                            {n: (A, B, s) for n, (A, B, s) in factors.items()}, "bf16", {n: params[n] for n in norms})))
    notes = {"direction": "quant", "mode": "qlora", "args": vars(a), "steps": n_steps, "trained_entries": n_entries,
             "variants": [note], "train_seconds": round(t_train, 1),
             "process_seconds": round(time.perf_counter() - t_start, 1),
             "E_format_note": "q_bytes is the update's CWQUAN1 size (islands/quant/qformat.py), checked to rebuild the "
                              "submitted model bit for bit; bits_per_param is per factor entry"}
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=targets + norms, notes=notes,
                   snapshots=snaps)
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
    ap.add_argument("--train", default="all", help="all | blocks | blockmats | comma-separated name regexes")
    ap.add_argument("--variants", default="fp,rtn4g128,rtn3g128,rtn2g128,sign,tern0.7", help="ptq")
    ap.add_argument("--norm-bits", type=int, default=8, help="ptq: the norm gains' rtn bits")
    ap.add_argument("--qat", default="tern", help="qat: sign | tern | rtn{b}")
    ap.add_argument("--qat-group", type=int, default=-1, help="qat: -1 one scale per tensor, 0 per row, else G entries")
    ap.add_argument("--qat-mult", type=float, default=1.0, help="qat: the scale over the group's mean |V|")
    ap.add_argument("--emit", choices=("full", "quan", "lora"), default="full",
                    help="E's format: encode_full, CWQUAN1, or (qlora --pow2) the bf16 lora format")
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    if a.mode == "qlora":
        if a.emit == "lora" and not a.pow2:
            raise SystemExit("quant: --emit lora carries the codes exactly only with --pow2")
        return main_qlora(a)
    if a.emit == "lora":
        raise SystemExit("quant: --emit lora is qlora's")
    if a.mode == "qat" and not QAT_SCHEME.fullmatch(a.qat):
        raise SystemExit(f"quant: unknown --qat {a.qat!r}")
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
    # qat: the quantized tensors (every trained matrix) train a latent delta V; their weights are θ + Q(V), rebuilt each step
    quantized = [n for n in names if params[n].ndim == 2] if a.mode == "qat" else []
    latent = {n: torch.zeros_like(params[n], dtype=torch.float32, device=dev) for n in quantized}
    theta_dev = {n: theta[n].to(dev) for n in quantized}
    opt_of = {n: latent.get(n, params[n]) for n in names}
    muon = Muon([opt_of[n] for n in mats], lr=a.muon_lr) if mats else None
    adam = torch.optim.AdamW([opt_of[n] for n in rest], lr=a.adam_lr, betas=(0.9, 0.95), weight_decay=0.0,
                             fused=dev.startswith("cuda")) if rest else None
    opts = [(o, lr) for o, lr in ((muon, a.muon_lr), (adam, a.adam_lr)) if o is not None]
    trainable = {"full": names}
    tokens = a.batch * prepare.EVAL_CTX
    extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net)
    steps, snaps, qarrays = [], [], {}
    marks = {round(n_steps * k / 4) for k in (1, 2, 3)} if a.mode == "qat" else set()
    print(json.dumps({"steps": n_steps, "flops_per_step": per_step, "trained": len(names), "quantized": len(quantized)}),
          flush=True)

    @torch.no_grad()
    def materialize() -> None:
        for n in quantized:
            q = qat_quant(latent[n], a.qat, a.qat_group, a.qat_mult)
            qarrays[n] = q
            params[n].data.copy_((theta_dev[n].float() + q.value()).to(theta_dev[n].dtype))

    t0 = time.perf_counter()
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for o, lr in opts:
            for g in o.param_groups:
                g["lr"] = lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
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
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if step + 1 in marks:
            snaps.append((prepare.encode_full({n: params[n].detach().to(theta[n].dtype).cpu() for n in names}), len(steps)))
    t_train = time.perf_counter() - t0
    if a.mode == "qat":
        blob, model, notes_v = finish_qat(a, theta, params, names, qarrays, dev)
    else:
        blob, model, snaps, notes_v = finish_ptq(a, theta, params, names, steps, dev)
    notes = {"direction": "quant", "mode": a.mode, "args": vars(a), "steps": n_steps,
             "trained_parameters": sum(params[n].numel() for n in names), "variants": notes_v,
             "train_seconds": round(t_train, 1), "process_seconds": round(time.perf_counter() - t_start, 1),
             "E_format_note": "E is encode_full of each variant's model until prepare.decode reads CWQUAN1; q_bytes is its "
                              "CWQUAN1 size (islands/quant/qformat.py), checked to rebuild that model bit for bit"}
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=names, notes=notes,
                   snapshots=snaps)
    return 0


def variant_note(v: str, qblob: bytes | None, q_bytes: int, h_bits: float, ok: bool, rel: float, n_params: int, t1: float,
                 **more) -> dict:
    out = {"variant": v, "q_bytes": q_bytes, "bits_per_param": 8 * q_bytes / n_params,
           "order0_entropy_bits_per_param": h_bits / n_params, "rebuilds_bit_for_bit": ok, "relative_sq_error": rel,
           "seconds": round(time.perf_counter() - t1, 1), **more}
    print(json.dumps(out), flush=True)
    return out


def finish_qat(a, theta, params, names, qarrays, dev):
    t1 = time.perf_counter()
    entries = {n: {"kind": "delta", "q": q} for n, q in qarrays.items()}
    entries |= {n: {"kind": "full", "t": params[n].detach()} for n in names if n not in qarrays}
    qblob = qf.encode(entries, theta)
    rebuilt = qf.decode(theta, qblob, dev)
    trained = {n: params[n].detach().to(theta[n].dtype).cpu() for n in names}
    ok = all(prepare.same_bits(rebuilt[n], trained[n]) for n in names)
    n_params = sum(params[n].numel() for n in names)
    h_bits = sum(q.entropy_bits() for q in qarrays.values()) + sum(
        32 * params[n].numel() for n in names if n not in qarrays)
    nonzero = {n: float((q.codebook.float()[q.codes.long()] != 0).float().mean()) for n, q in qarrays.items()}
    note = variant_note(f"qat-{a.qat}", qblob, len(qblob), h_bits, ok, 0.0, n_params, t1,
                        nonzero_share=sum(nonzero[n] * params[n].numel() for n in nonzero) / max(1, sum(
                            params[n].numel() for n in nonzero)), where="submission")
    blob = qblob if a.emit == "quan" else prepare.encode_full(trained)
    return blob, prepare.decode(theta, blob), [note]


def finish_ptq(a, theta, params, names, steps, dev):
    delta = {n: (params[n].detach().float() - theta[n].to(dev).float()) for n in names}
    n_params = sum(d.numel() for d in delta.values())
    blobs, notes_v = [], []
    for v in a.variants.split(","):
        t1 = time.perf_counter()
        if v == "fp":
            model = {n: params[n].detach().to(theta[n].dtype).cpu() for n in names}
            q_bytes = len(prepare.encode_full(model))
            qblob, h_bits, ok, rel = None, 8.0 * q_bytes, True, 0.0
        else:
            entries = entries_of(v, delta, a.norm_bits)
            h_bits = sum(e["q"].entropy_bits() for e in entries.values())
            qblob = qf.encode(entries, theta)
            rebuilt = qf.decode(theta, qblob, dev)
            model = {n: rebuilt[n] for n in names}
            want = {n: (theta[n].to(dev).float() + entries[n]["q"].value()).to(theta[n].dtype).cpu() for n in names}
            ok = all(prepare.same_bits(model[n], want[n]) for n in names)
            num = sum(float(((model[n].to(dev).float() - theta[n].to(dev).float() - delta[n]) ** 2).sum()) for n in names)
            rel = num / max(1e-30, sum(float((d ** 2).sum()) for d in delta.values()))
            q_bytes = len(qblob)
            del rebuilt, want, entries
        blobs.append(qblob if (a.emit == "quan" and qblob is not None) else prepare.encode_full(model))
        notes_v.append(variant_note(v, qblob, q_bytes, h_bits, ok, rel, n_params, t1,
                                    where="submission" if not notes_v else f"snapshot {len(notes_v) - 1:03d}"))
        del model
    return blobs[0], prepare.decode(theta, blobs[0]), [(b, len(steps)) for b in blobs[1:]], notes_v


if __name__ == "__main__":
    raise SystemExit(main())
