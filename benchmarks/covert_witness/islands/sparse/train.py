"""Island `sparse` (classes e1mb and e100kb): which parameters a capped update should touch, chosen from the gradient at θ,
encoded as mask plus values in the fixed formats; and the first-order trade between the mask's encoding cost and its values.

`islands/e1mb/train.py` and `islands/e100kb/train.py` on this branch are links to this file, which takes the class as
`--klass` on a pin with classes (the harness runs the resolved path, so the link's directory doesn't reach it). A pin
without classes (v3) takes no `--klass`, and `--cap` is E's byte cap there.

A probe (`--probe N` batches) takes the mean gradient G at θ of every tensor, priced in the step log as a forward pass plus
the full backward's FLOPs (`extra`). From G the record's notes get, for each budget in BUDGETS, the share of ||G||² that each
encoding of a mask captures (`analyse`): LoRA ranks, whole rows or columns as one-hot LoRA factors (what the fixed `lora`
decoder can carry) or with a list index, and single entries with an entropy-coded, int32-list or bitmap index at 16- or
8-bit values. The update then trains one mechanism (`--mech`):

- `norms`: the norm gains only (a tensor-level mask), E the gains in bf16 (`encode_lora`'s full tensors);
- `lines`: whole rows and columns of the block matrices, the lines with the most selection energy per byte across the
  candidate matrices (a learned per-layer budget); a row i is the factor pair B = e_i (the mask), A = its values, a column
  j is A = e_j, B = its values, so the mask costs d_out (row) or d_in (column) bf16 entries;
- `lora`: LoRA with each candidate matrix's rank allocated greedily by the probe gradient's σ² per byte;
- `qlora`: `lora` with `--bits`-bit factors (CWQUAN1: A one f16 scale per row, B one per 64 rows), so a rank costs
  (d_in + d_out) · bits / 8 bytes; the forward reads the codes, the factors train through them (straight-through);
- `signrows`: whole rows of the block matrices as `--bits`-bit codes of a delta (1 bit: a sign times the row's f16 scale),
  rows chosen by the selection energy per byte; E is a CWQUAN1 delta over each touched matrix whose other rows have
  scale 0, which the entropy coder all but drops, so the mask costs next to nothing.

The gains (`--norms`) can ride along with every mechanism: in bf16 with the CWLORA1 ones, as `--norm-bits` deltas with
the CWQUAN1 ones (θ is fp32, so a gain carried in full would cost 4 bytes). `--layers top:N` keeps every touched tensor
in the top N blocks (the final norm included), so the backward stops there and a step costs less. `--embed-rank R` gives
the tied embedding a rank-R LoRA (E carries it as one factor pair on `embed`; training applies it in factored form to
both the lookup and the head), its (vocab + d) · R entries taken before the block matrices' allocation. `--cap` sets a
byte bin below the class's cap (the 10 kB bin, or Fit's ~1.8 kB at the reference policy).

The record's notes give `delta_grad`: the trainable values N and the gradient plus AdamW's m and v passed between token
blocks, 3N fp32 values (a straight-through code trains fp32 latents, so codes shrink E and not this), in the lowrank
island's keys too (`bits`/`bytes` that state; `latent_bits`, apart from it, the fp32 latents a straight-through code
trains: the update application's Δ_t).

E is encoded at the end of every 5% of the harness's steps (the probe's batches included, `hooks.plan`), so each of the
harness's checkpoints scores the most training its 5% holds.

    python launch.py e1mb s50 --note TEXT -- --klass e1mb --mech lines [--targets all] [--norms all] [--lr 3e-4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import prepare
from te_core import hooks

LF_MATS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
TARGETS = {"all": set(LF_MATS), "out": {"wo", "w_down"}, "mlp": {"w_gate", "w_up", "w_down"}, "attn": {"wq", "wk", "wv", "wo"}}
BUDGETS = (10_000, 100_000, 1_000_000)
HEADER_SLACK = 256


def leaf(name: str) -> str:
    return name.split(".")[2] if name.startswith("blocks.") else name


def block(name: str) -> int | None:
    return int(name.split(".")[1]) if name.startswith("blocks.") else None


def in_layers(name: str, spec: str, n_layer: int) -> bool:
    if spec == "all":
        return True
    b = block(name)
    lo = n_layer - int(spec.split(":")[1])
    return b is None and name == "norm" or (b is not None and b >= lo)


def matrices(theta: dict, targets: str, layers: str, n_layer: int) -> list[str]:
    want = TARGETS[targets] if targets in TARGETS else set(targets.split(","))
    return sorted(n for n, t in theta.items() if t.ndim == 2 and n != "embed" and leaf(n) in want
                  and in_layers(n, layers, n_layer))


def norm_names(theta: dict, spec: str, layers: str, n_layer: int) -> list[str]:
    if spec == "none":
        return []
    names = sorted(n for n, t in theta.items() if t.ndim == 1 and in_layers(n, layers, n_layer))
    if spec == "final":
        return [n for n in names if n == "norm"]
    if spec.startswith("top:"):
        return [n for n in names if in_layers(n, spec, n_layer)]
    return names


# ---- the probe ---------------------------------------------------------------------------------------------------------

def probe(net, rec, theta, data, n_batches: int, batch: int, micro: int,
          names: list[str] | None = None) -> tuple[dict, dict, list[dict]]:
    """The mean gradient G and the mean squared batch gradient of every tensor (or of `names`) at the model's current
    weights, and the step-log entries pricing it. Gradients also reach whatever else is trainable: the caller clears them."""
    params = net.params()
    params = {n: params[n] for n in (sorted(params) if names is None else names)}
    was = {n: p.requires_grad for n, p in params.items()}
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    gsum = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    g2sum = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    tokens = batch * prepare.EVAL_CTX
    full = prepare.train_flops(rec, {"full": sorted(theta)}, [{"op": "train", "tokens": tokens}])["flops"]
    fwd = prepare.train_flops(rec, {}, [{"op": "forward", "tokens": tokens}])["flops"]
    steps = []
    for _ in range(n_batches):
        rows = data.next(batch)
        for i in range(0, batch, micro):
            x = rows[i: i + micro]
            with torch.autocast(prepare.DEVICE.split(":")[0], dtype=torch.bfloat16):
                logits = net.logits(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / batch)
            loss.backward()
        with torch.no_grad():
            for n, p in params.items():
                gsum[n].add_(p.grad)
                g2sum[n].addcmul_(p.grad, p.grad)
                p.grad = None
        steps += [{"op": "forward", "tokens": tokens}, {"op": "extra", "flops": full - fwd}]
    for n, p in params.items():
        p.requires_grad_(was[n])
    return ({n: g / n_batches for n, g in gsum.items()}, {n: g / n_batches for n, g in g2sum.items()}, steps)


# ---- the first-order trade ---------------------------------------------------------------------------------------------

def log2_choose(n: int, k: torch.Tensor) -> torch.Tensor:
    k = k.double()
    return (math.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)) / math.log(2)


def greedy(units: list[tuple[torch.Tensor, float]], budget_bits: float) -> float:
    """Units of (gains, bits each): the most gain whose bits fit, taking units by gain per bit."""
    gains = torch.cat([g.flatten().double().cpu() for g, _ in units])
    bits = torch.cat([torch.full((g.numel(),), float(b), dtype=torch.float64) for g, b in units])
    order = torch.argsort(gains / bits, descending=True)
    fit = torch.cumsum(bits[order], 0) <= budget_bits
    return float(gains[order][fit].sum())


def entries_at(sorted_e: dict, cums: dict, budget_bits: float, value_bits: int, index: str) -> tuple[float, float]:
    """Single entries above one global threshold on g², the threshold the lowest whose bits fit: (gain, index share)."""
    lo, hi = 0.0, max(float(e[0]) for e in sorted_e.values())

    def at(tau):
        gain, vbits, ibits = 0.0, 0.0, 0.0
        for n_, e in sorted_e.items():
            k = int(torch.searchsorted(-e, torch.tensor([-tau], device=e.device), right=True)[0])
            if not k:
                continue
            n = e.numel()
            gain += float(cums[n_][k - 1])
            vbits += value_bits * k
            ibits += (float(log2_choose(n, torch.tensor(float(k)))) if index == "ideal" else 32 * k if index == "list"
                      else n)
        return gain, vbits + ibits, ibits

    if at(hi)[1] > budget_bits:
        return 0.0, 0.0
    for _ in range(40):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 2 ** 20
        if at(mid)[1] <= budget_bits:
            hi = mid
        else:
            lo = mid
        if lo == 0 and hi < 1e-30:
            break
    g, b, i = at(hi)
    return g, i / b if b else 0.0


@torch.no_grad()
def analyse(G: dict) -> dict:
    """The share of Σ||G||² each mask encoding captures at each budget in BUDGETS (first order, an SGD step's decrease)."""
    total = sum(float(g.double().pow(2).sum()) for g in G.values())
    lora, sorted_e, cums, per = [], {}, {}, {}
    lines = {k: [] for k in ("rows_onehot", "cols_onehot", "rows_list", "cols_list")}
    for n, g in G.items():
        e = g.float().pow(2)
        sorted_e[n] = torch.sort(e.flatten(), descending=True).values
        cums[n] = torch.cumsum(sorted_e[n].double(), 0)
        per[n] = float(e.double().sum()) / total
        if g.ndim == 2:
            d_out, d_in = g.shape
            lora.append((torch.linalg.svdvals(g.float()).pow(2), 16 * (d_in + d_out)))
            lines["rows_onehot"].append((e.sum(1), 16 * (d_in + d_out)))
            lines["cols_onehot"].append((e.sum(0), 16 * (d_in + d_out)))
            lines["rows_list"].append((e.sum(1), 16 * d_in + math.log2(d_out)))
            lines["cols_list"].append((e.sum(0), 16 * d_out + math.log2(d_in)))
        else:
            whole = (e.sum().reshape(1), 16 * g.numel())
            lora.append(whole)
            for v in lines.values():
                v.append(whole)
    out = {"total_g2": total, "share_by_tensor": dict(sorted(per.items(), key=lambda kv: -kv[1])[:24]), "budgets": {}}
    for b in BUDGETS:
        bits = 8 * b
        row = {"lora_bf16": greedy(lora, bits) / total}
        row |= {f"{k}_bf16": greedy(v, bits) / total for k, v in lines.items()}
        for name, vb, ix in (("entries_ideal_bf16", 16, "ideal"), ("entries_ideal_8bit", 8, "ideal"),
                             ("entries_list_bf16", 16, "list"), ("entries_bitmap_bf16", 16, "bitmap")):
            g, share = entries_at(sorted_e, cums, bits, vb, ix)
            row[name] = g / total
            row[name + "_index_share"] = share
        out["budgets"][str(b)] = row
    return out


# ---- plans -------------------------------------------------------------------------------------------------------------

def plan_lines(cands: list[str], energy: dict, theta: dict, budget: int, orient: str) -> dict[str, tuple[list, list]]:
    """{matrix: (rows, cols)}: the lines with the most energy per byte whose one-hot factors fit `budget` bytes."""
    units = []
    for n in cands:
        e = energy[n]
        d_out, d_in = e.shape
        cost = 2 * (d_in + d_out)
        if orient in ("rows", "both"):
            for i, v in enumerate(e.sum(1).tolist()):
                units.append((v / cost, cost, n, "r", i))
        if orient in ("cols", "both"):
            for j, v in enumerate(e.sum(0).tolist()):
                units.append((v / cost, cost, n, "c", j))
    units.sort(key=lambda u: -u[0])
    plan, used = {}, 0
    for _, cost, n, kind, i in units:
        if used + cost > budget:
            break
        used += cost
        r, c = plan.setdefault(n, ([], []))
        (r if kind == "r" else c).append(i)
    return {n: (sorted(r), sorted(c)) for n, (r, c) in plan.items()}


def plan_lora(cands: list[str], G: dict, budget: int, max_rank: int, have: dict | None = None,
              bits: int = 16) -> dict[str, int]:
    """{matrix: rank}: rank units by the gradient's σ² per byte while they fit `budget` bytes, a rank costing
    (d_in + d_out) · bits / 8 bytes; with `have` (the ranks the matrices already hold), each matrix's units start past its
    held rank and the ranks returned are the added ones."""
    have = have or {}
    units = []
    for n in cands:
        d_out, d_in = G[n].shape
        cost = (d_in + d_out) * bits / 8
        s2 = torch.linalg.svdvals(G[n].float()).pow(2)[:max_rank].tolist()
        for k in range(have.get(n, 0), len(s2)):
            units.append((s2[k] / cost, cost, n, k))
    units.sort(key=lambda u: -u[0])
    ranks, used = dict(have), 0
    for _, cost, n, k in units:
        if used + cost > budget:
            continue
        if ranks.get(n, 0) == k:
            ranks[n] = k + 1
            used += cost
    return {n: r - have.get(n, 0) for n, r in ranks.items() if r > have.get(n, 0)}


# ---- encodings ---------------------------------------------------------------------------------------------------------

def onehot(idx: list[int], n: int, dim: int) -> torch.Tensor:
    m = torch.zeros(len(idx), n) if dim == 0 else torch.zeros(n, len(idx))
    for t, i in enumerate(idx):
        if dim == 0:
            m[t, i] = 1.0
        else:
            m[i, t] = 1.0
    return m


def line_factors(theta: dict, plan: dict, values: dict | None = None) -> dict:
    """W + B @ A, A = [row values; e_j for each column], B = [e_i for each row, column values]."""
    out = {}
    for n, (rows, cols) in plan.items():
        d_out, d_in = theta[n].shape
        vr, vc = values[n] if values else (torch.zeros(len(rows), d_in), torch.zeros(d_out, len(cols)))
        a = torch.cat([vr.detach().float().cpu(), onehot(cols, d_in, 0)], 0)
        b = torch.cat([onehot(rows, d_out, 1), vc.detach().float().cpu()], 1)
        out[n] = (a, b, 1.0)
    return out


def encode(mech: str, theta: dict, params: dict, norms: list[str], plan, train_vals) -> bytes:
    full = {n: params[n] for n in norms}
    if mech == "norms":
        return prepare.encode_lora({}, "bf16", full)
    if mech == "lines":
        return prepare.encode_lora(line_factors(theta, plan, train_vals), "bf16", full)
    fac = {}
    for n, pieces in train_vals.items():
        if len(pieces) == 1:
            fac[n] = pieces[0]
        else:
            fac[n] = (torch.cat([a for a, _, _ in pieces], 0), torch.cat([s * b for _, b, s in pieces], 1), 1.0)
    return prepare.encode_lora(fac, "bf16", full)


def e_bytes(mech: str, theta: dict, norms: list[str], plan, folded=()) -> int:
    """len(encode(...)) without encoding: under the harness every encode during the steps is a checkpoint candidate.
    `folded`: the LoRA matrices grown in pieces, encoded as one pair at scale 1."""
    ranks = {}
    if mech == "lines":
        ranks = {n: (len(r) + len(c), 1.0) for n, (r, c) in plan.items()}
    elif mech == "lora":
        ranks = {n: (r, 1.0 if n in folded else math.sqrt(theta[n].shape[1] / r)) for n, r in plan.items()}
    header = {"dtype": "bf16", "factors": [], "full": []}
    elems = 0
    for n in sorted(ranks):
        r, s = ranks[n]
        header["factors"].append([n, r, float(s)])
        elems += r * sum(theta[n].shape)
    for n in sorted(norms):
        header["full"].append(n)
        elems += theta[n].numel()
    return len(prepare._pack(b"CWLORA1\n", header, b"")) + 2 * elems


# ---- CWQUAN1 codes -----------------------------------------------------------------------------------------------------

B_ROWS = 64


def levels(bits: int) -> torch.Tensor:
    return torch.arange(2 ** bits, dtype=torch.float32) - (2 ** bits - 1) / 2


def quantize(x: torch.Tensor, bits: int, group: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x's `bits`-bit midrise code in groups of `group` consecutive entries (row-major): (the values, the codes, the f16
    scales). A group's scale s is 2·mean|x| at 1 bit (the value ±mean|x|), the RMS at 2 bits and max|x| over the top level
    above; a code c stands for (c − (2^bits − 1)/2) · s, exactly what CWQUAN1 decodes, and a group of scale 0 is 0."""
    flat = x.detach().float().reshape(-1, group)
    if bits == 1:
        s = 2 * flat.abs().mean(1)
    elif bits == 2:
        s = flat.pow(2).mean(1).sqrt()
    else:
        s = flat.abs().amax(1) / ((2 ** bits - 1) / 2)
    s = s.to(torch.float16).float()
    live = (s > 0)[:, None]
    codes = torch.where(live, torch.floor(flat / torch.where(live, s[:, None], 1.0)) + 2 ** (bits - 1), 0.0)
    codes = codes.clamp(0, 2 ** bits - 1)
    values = ((codes - (2 ** bits - 1) / 2) * s[:, None]).view(x.shape)
    return values, codes.to(torch.uint8).reshape(-1), s.to(torch.float16)


class Codes:
    """Straight-through codes: each latent's code sits in a leaf tensor the forward reads, refreshed once a step (not in
    every micro-batch's forward), and the code's gradient becomes the latent's."""

    def __init__(self):
        self.pairs: list[tuple] = []

    def add(self, latent: torch.Tensor, bits: int, group: int, code: torch.Tensor | None = None,
            base: torch.Tensor | None = None) -> torch.Tensor:
        """The leaf holding `base + code(latent − base)` (`code`, if given, is the model's own tensor)."""
        code = torch.zeros_like(latent, requires_grad=True) if code is None else code
        self.pairs.append((latent, code, bits, group, base))
        return code

    @torch.no_grad()
    def refresh(self) -> None:
        for latent, code, bits, group, base in self.pairs:
            if base is None:
                code.copy_(quantize(latent, bits, group)[0])
            else:
                code.copy_(base + quantize(latent - base, bits, group)[0])

    def backprop(self) -> None:
        for latent, code, *_ in self.pairs:
            latent.grad, code.grad = code.grad, None


def qarray(codes: torch.Tensor, scales: torch.Tensor, bits: int, group: int, shape: tuple) -> prepare.QArray:
    return prepare.QArray(levels(bits).to(torch.float16), scales.cpu(), codes.cpu(), bits, group, tuple(shape))


def b_group(d_out: int, r: int) -> int:
    return r * (B_ROWS if d_out % B_ROWS == 0 else d_out)


def lora_entry(A: torch.Tensor, B: torch.Tensor, s: float, bits: int) -> dict:
    (_, ca, sa), (_, cb, sb) = quantize(A, bits, A.shape[1]), quantize(B, bits, b_group(*B.shape))
    return {"kind": "lora", "a": qarray(ca, sa, bits, A.shape[1], A.shape),
            "b": qarray(cb, sb, bits, b_group(*B.shape), B.shape), "scale": s}


def delta_entry(d: torch.Tensor, bits: int, group: int) -> dict:
    _, c, s = quantize(d, bits, group)
    return {"kind": "delta", "q": qarray(c, s, bits, group, d.shape)}


def rows_entry(shape: tuple, rows: torch.Tensor, V: torch.Tensor, bits: int) -> dict:
    """A delta over a whole matrix that is V's code on `rows` and 0 (scale 0) on every other row."""
    d_out, d_in = shape
    _, c, s = quantize(V, bits, d_in)
    codes = torch.zeros(d_out, d_in, dtype=torch.uint8)
    scales = torch.zeros(d_out, dtype=torch.float16)
    codes[rows.cpu()] = c.view(-1, d_in).cpu()
    scales[rows.cpu()] = s.cpu()
    return {"kind": "delta", "q": qarray(codes.reshape(-1), scales, bits, d_in, shape)}


def quant_size(entries: dict, theta: dict) -> int:
    """len(prepare.encode_quant(entries, θ)) without calling it (under the harness every encode during the steps is a
    checkpoint candidate): the same header, arrays and xz settings."""
    header, parts = [], []
    for name in sorted(entries):
        e = entries[name]
        if e["kind"] == "delta":
            spec, arrays = prepare._quant_array(e["q"], name)
            header.append({"n": name, "k": "delta", **spec})
            parts += arrays
        else:
            (sa, pa), (sb, pb) = prepare._quant_array(e["a"], name), prepare._quant_array(e["b"], name)
            header.append({"n": name, "k": "lora", "r": int(e["a"].shape[0]), "s": float(e["scale"]), "A": sa, "B": sb})
            parts += pa + pb
    h = json.dumps({"t": header}, separators=(",", ":")).encode()
    return 8 + len(lzma.compress(struct.pack("<I", len(h)) + h + b"".join(parts), **prepare.QUANT_XZ))


def random_entries(theta: dict, mech: str, plan: dict, norms: list[str], bits: int, norm_bits: int, norm_group: int,
                   gen: torch.Generator) -> dict:
    """The entries of a plan with uniform random codes and scales: the trained codes are no less compressible."""
    out = {}
    for n in norms:
        out[n] = delta_entry(torch.randn(theta[n].shape, generator=gen), norm_bits, norm_group)
    for n, x in (plan or {}).items():
        d_out, d_in = theta[n].shape
        if mech == "qlora":
            out[n] = lora_entry(torch.randn(x, d_in, generator=gen), torch.randn(d_out, x, generator=gen),
                                math.sqrt(d_in / x), bits)
        else:
            rows = torch.tensor(x, dtype=torch.long)
            out[n] = rows_entry((d_out, d_in), rows, torch.randn(len(x), d_in, generator=gen), bits)
    return out


def plan_rows(cands: list[str], energy: dict, theta: dict, budget: float, bits: int) -> dict[str, list[int]]:
    """{matrix: rows}: whole rows by selection energy per byte (d_in · bits / 8 codes and a 2-byte scale) while they fit."""
    units = []
    for n in cands:
        d_in = theta[n].shape[1]
        cost = d_in * bits / 8 + 2
        units += [(e / cost, cost, n, i) for i, e in enumerate(energy[n].sum(1).tolist())]
    units.sort(key=lambda u: -u[0])
    plan, used = {}, 0.0
    for _, cost, n, i in units:
        if used + cost > budget:
            continue
        plan.setdefault(n, []).append(i)
        used += cost
    return {n: sorted(r) for n, r in plan.items()}


def tie_lora(lm, A, B, s: float) -> None:
    """`embed` + s · B @ A in the `lf` model's lookup and its tied head, never formed: the lookup adds s · B[idx] @ A and
    the head s · (x Aᵀ) Bᵀ, so the frozen head's matmul stays as it was. Mirrors `model.LM.hidden` and `LM.top`. A and B
    may be callables giving the factors the forward reads (their codes)."""
    fa = A if callable(A) else (lambda: A)
    fb = B if callable(B) else (lambda: B)

    def hidden(idx, upto=None):
        t = idx.shape[1]
        A, B = fa(), fb()
        x = F.embedding(idx, lm.embed) + s * F.linear(F.embedding(idx, B), A.t())
        cos, sin = lm.cos[:t], lm.sin[:t]
        for b in lm.blocks[:upto]:
            x = b(x, cos, sin)
        return x

    def top(x, start):
        t = x.shape[1]
        cos, sin = lm.cos[:t], lm.sin[:t]
        for b in lm.blocks[start:]:
            x = b(x, cos, sin)
        x = F.rms_norm(x, (x.shape[-1],), lm.norm)
        return F.linear(x, lm.embed) + s * F.linear(F.linear(x, fa()), fb())

    lm.hidden, lm.top = hidden, top


# ---- main --------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    classes = getattr(prepare, "CLASSES", None)
    ap.add_argument("--klass", required=bool(classes), choices=("e1mb", "e100kb") if classes else None)
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mech", required=True, choices=("norms", "lines", "lora", "qlora", "signrows"))
    ap.add_argument("--cap", type=int, default=0,
                    help="E's byte bin, at most the class's cap (0: the class's); required on a pin without classes")
    ap.add_argument("--bits", type=int, default=1, help="qlora: the factors' bits; signrows: the rows' code bits")
    ap.add_argument("--norm-bits", type=int, default=4, help="qlora, signrows: the gains' delta code bits")
    ap.add_argument("--norm-group", type=int, default=64, help="qlora, signrows: gain entries per f16 scale")
    ap.add_argument("--targets", default="all", help="`all`, `out`, `mlp`, `attn` or leaf names: the candidate matrices")
    ap.add_argument("--layers", default="all", help="`all` or `top:N`: touch only the top N blocks (and the final norm)")
    ap.add_argument("--norms", default="all", help="`all`, `none`, `final` or `top:N`: the gains trained in full")
    ap.add_argument("--orient", default="both", choices=("rows", "cols", "both"))
    ap.add_argument("--select", default="grad", choices=("grad", "fisher", "adam", "magnitude", "random"),
                    help="the lines' selection energy: G², the mean squared batch gradient, G²/√(that) (an Adam step's "
                         "first-order gain), W², or uniform noise")
    ap.add_argument("--max-rank", type=int, default=64)
    ap.add_argument("--alloc", default="grad", choices=("grad", "uniform"),
                    help="lora: ranks by the probe gradient's σ² per byte, or one rank on every candidate (the largest "
                         "that fits; the control)")
    ap.add_argument("--regrow", type=float, default=1.0,
                    help="lora: the share of the byte budget placed at θ; the rest goes, after --regrow-at of the "
                         "steps, to the rank units with the most σ² per byte of a fresh probe's gradient (a learned mask)")
    ap.add_argument("--regrow-at", type=float, default=1 / 3)
    ap.add_argument("--embed-rank", type=int, default=0,
                    help="lora: the tied embedding's LoRA rank, placed before the block matrices' allocation; -1: the "
                         "embedding joins the gradient's allocation as one more candidate")
    ap.add_argument("--probe", type=int, default=4, help="probe batches at θ (0: no probe, no analysis)")
    ap.add_argument("--analyse", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--norm-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--max-steps", type=int, default=0, help="a smoke test's cap on steps")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    t_start = time.perf_counter()
    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    L = rec["config"]["n_layer"]
    if classes:
        cap = classes[a.klass]["max_e_bytes"]
        if a.cap > cap:
            raise SystemExit(f"sparse: --cap {a.cap} is over {a.klass}'s cap of {cap}")
        cap = a.cap or cap
    elif a.cap > 0:
        cap = a.cap
    else:
        raise SystemExit("sparse: this pin has no classes, so --cap N (E's byte cap) is required")
    quant = a.mech in ("qlora", "signrows")
    if quant and not (1 <= a.bits <= 8 and 1 <= a.norm_bits <= 8):
        raise SystemExit("sparse: --bits and --norm-bits are 1 to 8")
    for p in params.values():
        p.requires_grad_(False)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    steps: list[dict] = []
    notes: dict = {"method": f"sparse/{a.mech}", "args": vars(a),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    G = G2 = None
    need_g = a.mech in ("lora", "qlora") or a.mech in ("lines", "signrows") and a.select in ("grad", "fisher", "adam")
    if need_g and not a.probe:
        raise SystemExit("sparse: --mech lora|qlora and a gradient selection need --probe > 0")
    if a.probe and (need_g or a.analyse):
        G, G2, steps = probe(net, rec, theta, data, a.probe, a.batch, a.micro)
        if a.analyse:
            t0 = time.perf_counter()
            notes["first_order"] = analyse(G)
            notes["first_order"]["seconds"] = round(time.perf_counter() - t0, 1)
            print(json.dumps({"first_order": notes["first_order"]["budgets"]}), flush=True)

    norms = norm_names(theta, a.norms, a.layers, L)
    cands = matrices(theta, a.targets, a.layers, L) if a.mech != "norms" else []
    if quant and any(theta[n].numel() % a.norm_group for n in norms):
        raise SystemExit(f"sparse: --norm-group {a.norm_group} doesn't divide every gain")
    sizer = torch.Generator().manual_seed(1)

    def size_of(plan) -> int:
        if quant:
            return quant_size(random_entries(theta, a.mech, plan, norms, a.bits, a.norm_bits, a.norm_group, sizer), theta)
        return e_bytes(a.mech, theta, norms, plan)

    norm_bytes = size_of({}) if quant else e_bytes("norms", theta, norms, None)
    avail = cap - norm_bytes - HEADER_SLACK
    trim = max(64, cap // 50)
    plan = None
    if a.mech in ("lines", "signrows"):
        if a.select == "grad":
            energy = {n: G[n].pow(2) for n in cands}
        elif a.select == "fisher":
            energy = {n: G2[n] for n in cands}
        elif a.select == "adam":
            energy = {n: G[n].pow(2) / G2[n].sqrt().clamp_min(1e-30) for n in cands}
        elif a.select == "magnitude":
            energy = {n: theta[n].float().pow(2) for n in cands}
        else:
            gen = torch.Generator().manual_seed(a.seed)
            energy = {n: torch.rand(theta[n].shape, generator=gen) for n in cands}
        energy = {n: e.float().cpu() for n, e in energy.items()}
        replan = (lambda: plan_lines(cands, energy, theta, avail, a.orient)) if a.mech == "lines" else \
            (lambda: plan_rows(cands, energy, theta, avail, a.bits))
        plan = replan()
        while plan and size_of(plan) > cap:
            avail -= trim
            plan = replan()
        energy = None
    elif a.mech == "qlora":
        if a.regrow < 1 or a.alloc != "grad":
            raise SystemExit("sparse: --mech qlora allocates by the gradient once (--regrow 1, --alloc grad)")
        pool = cands + (["embed"] if a.embed_rank < 0 else [])
        fixed = {"embed": a.embed_rank} if a.embed_rank > 0 else {}
        Gc = {n: G[n].cpu() for n in pool}
        avail = int(avail - size_of(fixed) + norm_bytes) if fixed else avail
        plan = fixed | plan_lora(pool, Gc, avail, a.max_rank, bits=a.bits)
        while plan != fixed and size_of(plan) > cap:
            avail -= trim
            plan = fixed | plan_lora(pool, Gc, avail, a.max_rank, bits=a.bits)
        Gc = None
    elif a.mech == "lora" and a.alloc == "uniform":
        r = 1
        while e_bytes("lora", theta, norms, {n: r + 1 for n in cands}) <= cap:
            r += 1
        plan = {n: r for n in cands}
    elif a.mech == "lora":
        if a.embed_rank and (a.regrow < 1 or theta["embed"].ndim != 2):
            raise SystemExit("sparse: --embed-rank needs --regrow 1 and a 2-D `embed`")
        pool = cands + (["embed"] if a.embed_rank < 0 else [])
        fixed = {"embed": a.embed_rank} if a.embed_rank > 0 else {}
        Gc = {n: G[n].cpu() for n in pool}
        avail = int(avail * a.regrow) - sum(2 * r * sum(theta[n].shape) for n, r in fixed.items())
        plan = fixed | plan_lora(pool, Gc, avail, a.max_rank)
        while plan != fixed and e_bytes("lora", theta, norms, plan) > cap:
            avail -= 2048
            plan = fixed | plan_lora(pool, Gc, avail, a.max_rank)
        Gc = None
    G = G2 = None
    if prepare.DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    size = size_of(plan)
    if size > cap:
        raise SystemExit(f"sparse: E is {size} bytes, over the cap of {cap}")

    dev = prepare.DEVICE
    for n in norms:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
    gain0 = {n: params[n].detach().clone() for n in norms} if quant else {}
    gain_latent = {n: torch.nn.Parameter(params[n].detach().clone()) for n in norms} if quant else {}
    codes = Codes()
    for n in gain_latent:
        codes.add(gain_latent[n], a.norm_bits, a.norm_group, code=params[n], base=gain0[n])
    train_vals: dict = {}
    hooks_params: list[torch.nn.Parameter] = []
    adapters: dict[str, int] = {}
    if a.mech == "lines":
        for n, (rows, cols) in plan.items():
            d_out, d_in = theta[n].shape
            vr = torch.nn.Parameter(torch.zeros(len(rows), d_in, device=dev))
            vc = torch.nn.Parameter(torch.zeros(d_out, len(cols), device=dev))
            ri = torch.tensor(rows, dtype=torch.long, device=dev)
            ci = torch.tensor(cols, dtype=torch.long, device=dev)
            train_vals[n] = (vr, vc)
            hooks_params += [vr, vc]
            adapters[n] = len(rows) + len(cols)
            mod = net.inner.get_submodule(n.removesuffix(".weight"))

            def hook(m, inp, out, vr=vr, vc=vc, ri=ri, ci=ci):
                x = inp[0]
                if ri.numel():
                    out = out.index_add(-1, ri, F.linear(x, vr).to(out.dtype))
                if ci.numel():
                    out = out + F.linear(x.index_select(-1, ci), vc).to(out.dtype)
                return out

            mod.register_forward_hook(hook)
    if a.mech == "signrows":
        for n, rows in plan.items():
            d_in = theta[n].shape[1]
            V = torch.nn.Parameter(torch.zeros(len(rows), d_in, device=dev))
            ri = torch.tensor(rows, dtype=torch.long, device=dev)
            train_vals[n] = (ri, V)
            hooks_params.append(V)
            adapters[n] = len(rows)
            Vq = codes.add(V, a.bits, d_in)
            mod = net.inner.get_submodule(n.removesuffix(".weight"))
            mod.register_forward_hook(lambda m, inp, out, Vq=Vq, ri=ri: out.index_add(
                -1, ri, F.linear(inp[0], Vq).to(out.dtype)))
    g = torch.Generator(device=dev).manual_seed(a.seed)

    def add_lora(n: str, r: int) -> list[torch.nn.Parameter]:
        d_out, d_in = theta[n].shape
        A = torch.nn.Parameter(torch.randn(r, d_in, device=dev, generator=g) / math.sqrt(d_in))
        B = torch.nn.Parameter(torch.zeros(d_out, r, device=dev))
        s = math.sqrt(d_in / r)
        train_vals.setdefault(n, []).append((A, B, s))
        adapters[n] = adapters.get(n, 0) + r
        fa, fb = (codes.add(A, a.bits, d_in), codes.add(B, a.bits, b_group(d_out, r))) if a.mech == "qlora" else (A, B)
        if n == "embed":
            tie_lora(net.inner, fa, fb, s)
            return [A, B]
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, fa=fa, fb=fb, s=s: out + s * F.linear(F.linear(inp[0], fa), fb))
        return [A, B]

    if a.mech in ("lora", "qlora"):
        for n, r in plan.items():
            hooks_params += add_lora(n, r)
    groups = []
    if hooks_params:
        groups.append({"params": hooks_params, "lr": a.lr, "weight_decay": 0.0})
    if norms:
        groups.append({"params": list(gain_latent.values()) if quant else [params[n] for n in norms],
                       "lr": a.norm_lr if a.mech != "norms" else a.lr, "weight_decay": 0.0})

    def encode_q() -> bytes:
        entries = {n: delta_entry(gain_latent[n].detach() - gain0[n], a.norm_bits, a.norm_group) for n in norms}
        for n, v in train_vals.items():
            if a.mech == "qlora":
                (A, B, s), = v
                entries[n] = lora_entry(A.detach(), B.detach(), s, a.bits)
            else:
                ri, V = v
                entries[n] = rows_entry(tuple(theta[n].shape), ri, V.detach(), a.bits)
        return prepare.encode_quant(entries, theta)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=dev.startswith("cuda"))
    trainable = {"full": norms, "adapters": adapters}
    tokens = a.batch * prepare.EVAL_CTX
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    spent = prepare.train_flops(rec, trainable, steps)["flops"]
    regrow = a.mech == "lora" and a.regrow < 1
    if regrow:
        spent += a.probe * prepare.train_flops(rec, {"full": sorted(theta)}, [{"op": "train", "tokens": tokens}])["flops"]
    n_steps = int((a.budget_fraction * prepare.TRAIN_FLOP_BUDGET - spent) // per_step)
    if a.max_steps:
        n_steps = min(n_steps, a.max_steps)
    grow_at = round(n_steps * a.regrow_at) if regrow else -1
    summary = {"mech": a.mech, "steps": n_steps, "E_bytes": size, "norms": len(norms), "flops_per_step": per_step,
               "probe_flops": spent, "matrices": len(plan or {}),
               "lines_or_rank": sum(adapters.values()), "lowest_trained": prepare.train_flops(
                   rec, trainable, [{"op": "train", "tokens": 1}])["lowest_trained"]}
    if a.mech == "lines":
        summary["rows"] = sum(len(r) for r, _ in plan.values())
        summary["cols"] = sum(len(c) for _, c in plan.values())
        notes["plan"] = {n: {"rows": len(r), "cols": len(c)} for n, (r, c) in plan.items()}
        notes["mask"] = {n: {"rows": r, "cols": c} for n, (r, c) in plan.items()}
    elif a.mech in ("lora", "qlora"):
        notes["plan"] = plan
    elif a.mech == "signrows":
        summary["rows"] = sum(len(r) for r in plan.values())
        notes["plan"] = {n: len(r) for n, r in plan.items()}
        notes["mask"] = plan
    notes["summary"] = summary
    print(json.dumps(summary), flush=True)
    fwd = torch.compile(net) if dev.startswith("cuda") else net
    calls = a.probe if steps else 0
    planned = calls + n_steps + (a.probe if regrow else 0)
    hooks.plan(planned)
    marks = {planned * k // 20 for k in range(1, 21)}
    t0 = time.perf_counter()
    lrs = [grp["lr"] for grp in opt.param_groups]
    loss = torch.zeros(())
    for step in range(n_steps):
        if step == grow_at:
            Gm, _, more = probe(net, rec, theta, data, a.probe, a.batch, a.micro, names=cands)
            steps += more
            calls += a.probe
            opt.zero_grad(set_to_none=True)
            Gc = {n: Gm[n].cpu() for n in cands}
            Gm = None
            have = dict(adapters)

            def merged(add: dict) -> dict:
                return {n: have.get(n, 0) + add.get(n, 0) for n in set(have) | set(add)}

            room = cap - e_bytes("lora", theta, norms, have) - HEADER_SLACK
            add = plan_lora(cands, Gc, room, a.max_rank, have)
            while add and e_bytes("lora", theta, norms, merged(add), folded=set(add) & set(have)) > cap:
                room -= 2048
                add = plan_lora(cands, Gc, room, a.max_rank, have)
            Gc = None
            new = [p for n, r in add.items() for p in add_lora(n, r)]
            if new:
                opt.add_param_group({"params": new, "lr": a.lr, "weight_decay": 0.0})
                lrs.append(a.lr)
            notes["regrow"] = {"step": step, "added": add, "E_bytes": e_bytes("lora", theta, norms, merged(add),
                                                                              folded=set(add) & set(have))}
            print(json.dumps({"regrow": notes["regrow"]}), flush=True)
        frac = min(1.0, (step + 1) / a.warmup) * (1 - step / n_steps)
        for grp, lr in zip(opt.param_groups, lrs):
            grp["lr"] = lr * frac
        rows = data.next(a.batch)
        calls += 1
        codes.refresh()
        for i in range(0, a.batch, a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) * (len(x) / a.batch)
            loss.backward()
        codes.backprop()
        torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        if (step + 1) % 100 == 0 or step + 1 == n_steps:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if calls in marks:
            encode_q() if quant else encode(a.mech, theta, params, norms, plan, train_vals)
    blob = encode_q() if quant else encode(a.mech, theta, params, norms, plan, train_vals)
    model = prepare.decode(theta, blob)
    changed = sorted(set(norms) | set(adapters))
    n_train = sum(p.numel() for grp in opt.param_groups for p in grp["params"])
    notes["delta_grad"] = {"values": n_train, "fp32_bytes": 12 * n_train,
                           "what": "the gradient and AdamW's m and v of every trained value (fp32 latents for codes)",
                           "trainable": n_train, "grad_bits": 32 * n_train, "adam_bits": 64 * n_train, "muon_bits": 0,
                           "latent_bits": 32 * n_train if quant else 0, "bits": 96 * n_train, "bytes": 12 * n_train}
    notes["E_bytes"] = {"planned": size, "encoded": len(blob), "format": prepare.format_of(blob)}
    notes["E_bytes"]["over_cap"] = len(blob) > cap
    print(json.dumps({"delta_grad": notes["delta_grad"], "E_bytes": notes["E_bytes"]}), flush=True)
    notes["seconds"] = round(time.perf_counter() - t_start, 1)
    prepare.submit(Path(a.out), a.start, *([a.klass] if classes else []), blob, model, trainable, steps, changed=changed,
                   notes=notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
