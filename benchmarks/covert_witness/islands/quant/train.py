"""Island `quant` (direction, not a class): how far an update compresses without losing the loss it bought. It runs in any
class (`--klass`; islands/<class>/train.py on this branch is a link here) and trains an update, then encodes it in low-bit
CWQUAN1 (`qformat.py`: k-bit codes, f16 codebooks and per-group scales, entropy coded).

Modes:

- `ptq`: the reference recipe (Muon on the trained matrices, AdamW on the rest) on `--train`'s tensors, then each of
  `--variants` quantizes the update Δ = W − θ;
- `qat`: the trained matrices' weights are θ + Q(V) in the forward, Q a per-group k-bit quantizer of a latent V with a
  straight-through gradient, so the trained model is exactly a CWQUAN1 code.

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--klass", required=True, choices=sorted(prepare.CLASSES))
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("ptq",), default="ptq")
    ap.add_argument("--train", default="all", help="all | blocks | blockmats | comma-separated name regexes")
    ap.add_argument("--variants", default="fp,rtn4g128,rtn3g128,rtn2g128,sign,tern0.7")
    ap.add_argument("--norm-bits", type=int, default=8)
    ap.add_argument("--emit", choices=("full", "quan"), default="full")
    ap.add_argument("--muon-lr", type=float, default=0.01)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    t_start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
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
    muon = Muon([params[n] for n in mats], lr=a.muon_lr) if mats else None
    adam = torch.optim.AdamW([params[n] for n in rest], lr=a.adam_lr, betas=(0.9, 0.95), weight_decay=0.0,
                             fused=prepare.DEVICE.startswith("cuda")) if rest else None
    opts = [(o, lr) for o, lr in ((muon, a.muon_lr), (adam, a.adam_lr)) if o is not None]
    trainable = {"full": names}
    tokens = a.batch * prepare.EVAL_CTX
    extra = sum(ns_flops(tuple(params[n].shape)) for n in mats)
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"] + extra
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // per_step)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net)
    steps = []
    print(json.dumps({"steps": n_steps, "flops_per_step": per_step, "trained": len(names)}), flush=True)
    t0 = time.perf_counter()
    for step in range(n_steps):
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for o, lr in opts:
            for g in o.param_groups:
                g["lr"] = lr * frac
        rows = data.next(a.batch)
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.0)
        for o, _ in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if extra:
            steps.append({"op": "extra", "flops": extra, "what": "muon newton-schulz"})
        if (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
    t_train = time.perf_counter() - t0
    dev = prepare.DEVICE
    delta = {n: (params[n].detach().float() - theta[n].to(dev).float()) for n in names}
    n_params = sum(d.numel() for d in delta.values())
    blobs, notes_v = [], []
    for v in a.variants.split(","):
        t1 = time.perf_counter()
        if v == "fp":
            model = {n: params[n].detach().to(theta[n].dtype).cpu() for n in names}
            q_bytes, q_ok, rel = len(prepare.encode_full(model)), True, 0.0
            qblob, h_bits = None, 8.0 * q_bytes
        else:
            entries = entries_of(v, delta, a.norm_bits)
            h_bits = sum(e["q"].entropy_bits() for e in entries.values())
            qblob = qf.encode(entries, theta)
            rebuilt = qf.decode(theta, qblob, dev)
            model = {n: rebuilt[n] for n in names}
            want = {n: (theta[n].to(dev).float() + entries[n]["q"].value()).to(theta[n].dtype).cpu() for n in names}
            q_ok = all(prepare.same_bits(model[n], want[n]) for n in names)
            num = sum(float(((model[n].to(dev).float() - theta[n].to(dev).float() - delta[n]) ** 2).sum()) for n in names)
            rel = num / max(1e-30, sum(float((d ** 2).sum()) for d in delta.values()))
            q_bytes = len(qblob)
            del rebuilt, want, entries
        blob = qblob if (a.emit == "quan" and qblob is not None) else prepare.encode_full(model)
        blobs.append(blob)
        notes_v.append({"variant": v, "q_bytes": q_bytes, "bits_per_param": 8 * q_bytes / n_params,
                        "order0_entropy_bits_per_param": h_bits / n_params,
                        "rebuilds_bit_for_bit": q_ok, "relative_sq_error": rel, "seconds": round(time.perf_counter() - t1, 1)})
        print(json.dumps(notes_v[-1]), flush=True)
        del model
    main_blob = blobs[0]
    model = prepare.decode(theta, main_blob)
    snaps = [(b, len(steps)) for b in blobs[1:]]
    for i, nv in enumerate(notes_v):
        nv["where"] = "submission" if i == 0 else f"snapshot {i - 1:03d} (curve[{i - 1}])"
    notes = {"direction": "quant", "mode": a.mode, "args": vars(a), "steps": n_steps, "trained_parameters": n_params,
             "variants": notes_v, "train_seconds": round(t_train, 1), "process_seconds": round(time.perf_counter() - t_start, 1),
             "E_format_note": "E is encode_full of each variant's model until prepare.decode reads CWQUAN1; q_bytes is its "
                              "CWQUAN1 size (islands/quant/qformat.py), checked to rebuild that model bit for bit"}
    prepare.submit(Path(a.out), a.start, a.klass, main_blob, model, trainable, steps, changed=names, notes=notes,
                   snapshots=snaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
