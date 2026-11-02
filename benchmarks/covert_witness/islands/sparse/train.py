"""Island `sparse` (classes e1mb and e100kb): which parameters a capped update should touch, chosen from the gradient at θ,
encoded as mask plus values in the fixed formats; and the first-order trade between the mask's encoding cost and its values.

`islands/e1mb/train.py` and `islands/e100kb/train.py` on this branch are links to this file, which takes the class as
`--klass` (the harness runs the resolved path, so the link's directory doesn't reach it).

A probe (`--probe N` batches) takes the mean gradient G at θ of every tensor, priced in the step log as a forward pass plus
the full backward's FLOPs (`extra`). From G the record's notes get, for each budget in BUDGETS, the share of ||G||² that each
encoding of a mask captures (`analyse`): LoRA ranks, whole rows or columns as one-hot LoRA factors (what the fixed `lora`
decoder can carry) or with a list index, and single entries with an entropy-coded, int32-list or bitmap index at 16- or
8-bit values. The update then trains one mechanism (`--mech`):

- `norms`: the norm gains only (a tensor-level mask), E the gains in bf16 (`encode_lora`'s full tensors);
- `lines`: whole rows and columns of the block matrices, the lines with the most selection energy per byte across the
  candidate matrices (a learned per-layer budget); a row i is the factor pair B = e_i (the mask), A = its values, a column
  j is A = e_j, B = its values, so the mask costs d_out (row) or d_in (column) bf16 entries;
- `lora`: LoRA with each candidate matrix's rank allocated greedily by the probe gradient's σ² per byte.

The gains (`--norms`) can ride along with `lines` and `lora`. `--layers top:N` keeps every touched tensor in the top N
blocks (the final norm included), so the backward stops there and a step costs less. `--embed-rank R` gives the tied
embedding a rank-R LoRA (E carries it as one factor pair on `embed`; training applies it in factored form to both the
lookup and the head), its (vocab + d) · R bf16 entries taken before the block matrices' allocation.

E is encoded at the end of every 5% of the harness's steps (the probe's batches included, `hooks.plan`), so each of the
harness's checkpoints scores the most training its 5% holds.

    python launch.py e1mb s50 --note TEXT -- --klass e1mb --mech lines [--targets all] [--norms all] [--lr 3e-4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def plan_lora(cands: list[str], G: dict, budget: int, max_rank: int, have: dict | None = None) -> dict[str, int]:
    """{matrix: rank}: rank units by the gradient's σ² per byte while they fit `budget` bytes; with `have` (the ranks the
    matrices already hold), each matrix's units start past its held rank and the ranks returned are the added ones."""
    have = have or {}
    units = []
    for n in cands:
        d_out, d_in = G[n].shape
        cost = 2 * (d_in + d_out)
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


def tie_lora(lm, A: torch.Tensor, B: torch.Tensor, s: float) -> None:
    """`embed` + s · B @ A in the `lf` model's lookup and its tied head, never formed: the lookup adds s · B[idx] @ A and
    the head s · (x Aᵀ) Bᵀ, so the frozen head's matmul stays as it was. Mirrors `model.LM.hidden` and `LM.top`."""

    def hidden(idx, upto=None):
        t = idx.shape[1]
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
        return F.linear(x, lm.embed) + s * F.linear(F.linear(x, A), B)

    lm.hidden, lm.top = hidden, top


# ---- main --------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--klass", required=True, choices=("e1mb", "e100kb"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mech", required=True, choices=("norms", "lines", "lora"))
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
    cap = prepare.CLASSES[a.klass]["max_e_bytes"]
    for p in params.values():
        p.requires_grad_(False)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    steps: list[dict] = []
    notes: dict = {"method": f"sparse/{a.mech}", "args": vars(a),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    G = G2 = None
    need_g = a.mech == "lora" or a.mech == "lines" and a.select in ("grad", "fisher", "adam")
    if need_g and not a.probe:
        raise SystemExit("sparse: --mech lora and a gradient selection need --probe > 0")
    if a.probe and (need_g or a.analyse):
        G, G2, steps = probe(net, rec, theta, data, a.probe, a.batch, a.micro)
        if a.analyse:
            t0 = time.perf_counter()
            notes["first_order"] = analyse(G)
            notes["first_order"]["seconds"] = round(time.perf_counter() - t0, 1)
            print(json.dumps({"first_order": notes["first_order"]["budgets"]}), flush=True)

    norms = norm_names(theta, a.norms, a.layers, L)
    cands = matrices(theta, a.targets, a.layers, L) if a.mech != "norms" else []
    norm_bytes = e_bytes("norms", theta, norms, None)
    avail = cap - norm_bytes - HEADER_SLACK
    plan = None
    if a.mech == "lines":
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
        plan = plan_lines(cands, energy, theta, avail, a.orient)
        while plan and e_bytes("lines", theta, norms, plan) > cap:
            avail -= 2048
            plan = plan_lines(cands, energy, theta, avail, a.orient)
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
    size = e_bytes(a.mech, theta, norms, plan)
    if size > cap:
        raise SystemExit(f"sparse: E is {size} bytes, over {a.klass}'s cap of {cap}")

    dev = prepare.DEVICE
    for n in norms:
        params[n].data = params[n].data.float()
        params[n].requires_grad_(True)
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
    g = torch.Generator(device=dev).manual_seed(a.seed)

    def add_lora(n: str, r: int) -> list[torch.nn.Parameter]:
        d_out, d_in = theta[n].shape
        A = torch.nn.Parameter(torch.randn(r, d_in, device=dev, generator=g) / math.sqrt(d_in))
        B = torch.nn.Parameter(torch.zeros(d_out, r, device=dev))
        s = math.sqrt(d_in / r)
        train_vals.setdefault(n, []).append((A, B, s))
        adapters[n] = adapters.get(n, 0) + r
        if n == "embed":
            tie_lora(net.inner, A, B, s)
            return [A, B]
        mod = net.inner.get_submodule(n.removesuffix(".weight"))
        mod.register_forward_hook(lambda m, inp, out, A=A, B=B, s=s: out + s * F.linear(F.linear(inp[0], A), B))
        return [A, B]

    if a.mech == "lora":
        for n, r in plan.items():
            hooks_params += add_lora(n, r)
    groups = []
    if hooks_params:
        groups.append({"params": hooks_params, "lr": a.lr, "weight_decay": 0.0})
    if norms:
        groups.append({"params": [params[n] for n in norms], "lr": a.norm_lr if a.mech != "norms" else a.lr,
                       "weight_decay": 0.0})
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
    elif a.mech == "lora":
        notes["plan"] = plan
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
        if (step + 1) % 100 == 0 or step + 1 == n_steps:
            print(json.dumps({"step": step + 1, "loss": float(loss.detach()) * a.batch / len(x),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
        if calls in marks:
            encode(a.mech, theta, params, norms, plan, train_vals)
    blob = encode(a.mech, theta, params, norms, plan, train_vals)
    model = prepare.decode(theta, blob)
    changed = sorted(set(norms) | set(adapters))
    notes["seconds"] = round(time.perf_counter() - t_start, 1)
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=changed, notes=notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
