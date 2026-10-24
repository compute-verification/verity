"""B2: condition theta'_s on cheap predictors; measure the residual's fresh bits in the weights and in the counted
activation values (outputs of every nn.Linear inside the transformer layers) at bf16, FP8 (per-tensor E4M3) and NVFP4.

A predictor is a full predicted state dict P(s), computed from theta (= main step 0) and runs cheaper than the target
(main step s). Weights: fresh bits of theta'_s's encoding given P's float values (quant.fresh_bits, the sweep's two-part
context code), and given P + a rank-r correction of the residual (low-rank MDL, the correction charged at 16 bits per
factor entry), and lzma of the residual codes on one middle layer. Activations: the same, with P's activations on the
held-out tokens as the prediction. Per tensor the best of (context, MDL[, lzma]) is taken, then bits are summed and divided
by the values. Coefficients fit on the target (per-tensor t) are charged at 32 bits each.
"""
import json, lzma, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant
from quant import encode, fresh_bits, Hcond, combined_code
from common import MODEL, TAG, OUT, SEQ, BATCH, tokens, snap_path

DEV = os.environ.get("B2_DEV", "cuda")
ROW_LAYERS = os.environ.get("B2_LAYERS")
MLP_STEPS = int(os.environ.get("B2_MLP_STEPS", "1500"))
for _G in (quant.G_BF16, quant.G_E4M3, quant.G_E4M3_POS, quant.G_E2M1):
    for _k in list(_G):
        _G[_k] = _G[_k].to(DEV)
FMTS = ("bf16", "fp8t", "nvfp4")
MDL_RANKS = (1, 4, 16, 64)
LR = 5e-5
SNAPS_MAIN = (0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 1000)
TARGETS = tuple(int(t) for t in os.environ.get("TARGETS", "1 10 100 1000").split())
B1_RUNS = tuple(os.environ.get("B1_RUNS", "b1 b1lr").split())
SUB_RUNS = tuple(os.environ.get("SUB_RUNS", "sub10 sub10lr1 sub10lr3").split())
COEF_BITS = 32
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
if hasattr(model, "gpt_neox"):
    LAYERS, LPFX = model.gpt_neox.layers, "gpt_neox.layers"
else:
    LAYERS, LPFX = model.model.layers, "model.layers"
NL = len(LAYERS)
LIN = [n for n, m in LAYERS[0].named_modules() if isinstance(m, torch.nn.Linear)]
ROWS = set(range(NL)) if not ROW_LAYERS else {int(x) for x in ROW_LAYERS.split()}
WKEYS = [(l, n, f"{LPFX}.{l}.{n}.weight") for l in range(NL) for n in LIN if l in ROWS]
LZ_LAYER = NL // 2
cfg = model.config
D_MODEL = cfg.hidden_size
N_MM = sum(m.weight.numel() for L in LAYERS for m in L.modules() if isinstance(m, torch.nn.Linear))
N_MM += model.get_output_embeddings().weight.numel()
F_STEP = 6 * N_MM * BATCH * SEQ + 12 * NL * D_MODEL * SEQ * SEQ * BATCH  # main run, one step (fwd + bwd)
held = tokens()["held"]
EVAL = torch.stack([held[int(s):int(s) + 512] for s in torch.linspace(0, len(held) - 513, 4).long()]).to(DEV)
F_EVAL = 2 * N_MM * EVAL.numel()

_cache = {}


def snap(name, step):
    key = (name, step)
    if key not in _cache:
        sd = torch.load(snap_path(name, step), map_location=DEV)
        _cache[key] = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    return _cache[key]


TH0 = snap("main", 0)
FKEYS = [k for k, v in TH0.items() if v.is_floating_point()]


def layer_of(k):
    if not k.startswith(LPFX + "."):
        return None
    return int(k[len(LPFX) + 1:].split(".")[0])


# ---------------------------------------------------------------- estimators

POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("LZMA_THREADS", "8")))


def _lz(arrs):
    tot = 0
    for t in arrs:
        lo, hi = int(t.min()), int(t.max())
        dt = torch.int8 if -128 <= lo and hi <= 127 else torch.int16
        tot += 8 * len(lzma.compress(t.reshape(-1).to(dt).cpu().numpy().tobytes(), preset=6))
    return tot


def bits(X0, X1, Xhat, want_lzma=False):
    """X0 registered (sets the FP8/NVFP4 reference amax, as in the sweep), X1 fresh, Xhat predicted; 2D float32."""
    T, D = X1.shape
    N = X1.numel()
    R = X1 - Xhat
    q = min(max(MDL_RANKS) + 8, min(T, D))
    U, S, V = torch.svd_lowrank(R, q=q, niter=3)
    ref = X0.abs().max()
    out = {}
    e2 = float((R.double() ** 2).sum())
    d2 = float(((X1 - X0).double() ** 2).sum())
    for fmt in FMTS:
        E1 = encode(X1, fmt, ref)
        b_ctx, d, ds = fresh_bits(Xhat, E1, fmt, ref)
        Eh = encode(Xhat, fmt, ref)
        b_cond = Hcond(combined_code(E1, fmt), combined_code(Eh, fmt)) * N
        mdl, rb = b_ctx, 0
        for r in MDL_RANKS:
            if r >= min(T, D):
                break
            A = (U[:, :r] * S[:r]).bfloat16().float()
            B = V[:, :r].T.bfloat16().float()
            b = fresh_bits(Xhat + A @ B, E1, fmt, ref)[0] + r * (T + D) * 16
            if b < mdl:
                mdl, rb = b, r
        m = {"n": N, "ctx": b_ctx, "cond": b_cond, "mdl": mdl, "mdl_r": rb, "e2": e2, "d2": d2,
             "frac": float((E1["deq"] != Eh["deq"]).float().mean())}
        if want_lzma:
            m["lzma"] = POOL.submit(_lz, [d] + ([ds] if ds is not None else []))
        out[fmt] = m
    return out


def finish(rows):
    """resolve lzma futures; per-tensor best of ctx / mdl / lzma"""
    for r in rows:
        for fmt in FMTS:
            m = r[fmt]
            if "lzma" in m:
                m["lzma"] = m["lzma"].result()
            m["best"] = min(m["ctx"], m["mdl"], m.get("lzma", math.inf))
    return rows


def aggregate(rows, layers_of_rows, paid_bits=0.0, layer_filter=None):
    res = {}
    for fmt in FMTS:
        sel = [(r, l) for r, l in zip(rows, layers_of_rows) if layer_filter is None or l in layer_filter]
        n = sum(r[fmt]["n"] for r, _ in sel)
        a = {k: sum(r[fmt][k] for r, _ in sel) / n for k in ("ctx", "cond", "mdl", "best")}
        a["best"] += paid_bits / n
        lz = [r[fmt] for r, _ in sel if "lzma" in r[fmt]]
        if lz:
            a["lzma_layer"] = sum(m["lzma"] for m in lz) / sum(m["n"] for m in lz)
            a["ctx_lzma_layer"] = sum(m["ctx"] for m in lz) / sum(m["n"] for m in lz)
        a["frac"] = sum(r[fmt]["frac"] * r[fmt]["n"] for r, _ in sel) / n
        a["rel_err"] = (sum(r[fmt]["e2"] for r, _ in sel) / max(sum(r[fmt]["d2"] for r, _ in sel), 1e-300)) ** 0.5
        per_layer = {}
        for r, l in sel:
            pl = per_layer.setdefault(l, [0.0, 0])
            pl[0] += r[fmt]["best"]
            pl[1] += r[fmt]["n"]
        a["layers"] = {str(l): v[0] / v[1] for l, v in sorted(per_layer.items())}
        a["n"] = n
        res[fmt] = a
    return res


# ---------------------------------------------------------------- activations

INPUTS = {}


def acts(sd, keep_inputs=False):
    """outputs of every Linear in the layers on EVAL; with keep_inputs, also their inputs, into INPUTS"""
    model.load_state_dict(sd, strict=False)
    store, hooks = {}, []
    if keep_inputs:
        INPUTS.clear()

    def hook(key):
        def f(mod, inp, out):
            store[key] = out.detach().reshape(-1, out.shape[-1]).float().clone()
            if keep_inputs:
                INPUTS[key] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().clone()
        return f

    for l, L in enumerate(LAYERS):
        for n, m in L.named_modules():
            if isinstance(m, torch.nn.Linear):
                hooks.append(m.register_forward_hook(hook((l, n))))
    with torch.no_grad():
        loss = float(model(input_ids=EVAL, labels=EVAL).loss)
    for h in hooks:
        h.remove()
    return store, loss


# ---------------------------------------------------------------- predictors

def lin_pred(base, dirs, target):
    """base + sum_j c_j dirs[j], c fit per tensor by least squares on the target; returns (pred, paid bits).
    A direction is a pair (X, Y) of state dicts, meaning X - Y (Y None: X itself), formed one tensor at a time."""
    pred, ncoef = {}, 0
    CH = 1 << 24
    for k, v in TH0.items():
        if not v.is_floating_point():
            pred[k] = v
            continue
        B = base[k].reshape(-1)
        if not dirs:
            pred[k] = base[k]
            continue
        m = len(dirs)
        G = torch.zeros(m, m, device=DEV, dtype=torch.float64)
        b = torch.zeros(m, device=DEV, dtype=torch.float64)
        y_all = target[k].reshape(-1)
        for i in range(0, B.numel(), CH):
            Dm = torch.stack([X[k].reshape(-1)[i:i + CH] - (Y[k].reshape(-1)[i:i + CH] if Y is not None else 0)
                              for X, Y in dirs]).double()
            G += Dm @ Dm.T
            b += Dm @ (y_all[i:i + CH] - B[i:i + CH]).double()
        G += torch.eye(m, device=DEV, dtype=G.dtype) * (1e-9 * G.diagonal().mean() + 1e-30)
        c = torch.linalg.solve(G, b).float()
        P = B.clone()
        for j, (X, Y) in enumerate(dirs):
            P += c[j] * (X[k].reshape(-1) - (Y[k].reshape(-1) if Y is not None else 0))
        pred[k] = P.reshape(v.shape)
        ncoef += m
    return pred, ncoef * COEF_BITS


def learned(s, target, kind):
    """elementwise model from early-snapshot features to the update at step s, trained on half the layers' tensors
    (even/odd), applied to the other half (cross-fit); non-layer tensors use the even-trained model"""
    A = [a for a in SNAPS_MAIN if 0 < a <= s / 10]
    if not A:
        return None

    def feats(k):
        th0 = TH0[k].reshape(-1)
        f = [(snap("main", a)[k].reshape(-1) - th0) / LR for a in A]
        f.append(th0 / (th0.pow(2).mean().sqrt() + 1e-8))
        f.append(torch.sign(snap("main", A[0])[k].reshape(-1) - th0))
        return torch.stack(f, 1)

    def ytrue(k):
        return (target[k] - TH0[k]).reshape(-1) / LR

    g = torch.Generator(device=DEV).manual_seed(0)
    models, info = {}, {}
    m_in = len(A) + 2
    for half in (0, 1):
        keys = [k for k in FKEYS if layer_of(k) is not None and layer_of(k) % 2 == half]
        Xs, ys = [], []
        for k in keys:
            n = TH0[k].numel()
            idx = torch.randperm(n, device=DEV, generator=g)[:100_000]
            Xs.append(feats(k)[idx])
            ys.append(ytrue(k)[idx])
        X, y = torch.nan_to_num(torch.cat(Xs)), torch.nan_to_num(torch.cat(ys))
        mu, sd = X.mean(0), X.std(0) + 1e-8
        ys_ = y.std() + 1e-12
        Xn, yn = (X - mu) / sd, y / ys_
        if kind == "lin":
            Xa = torch.cat([Xn, torch.ones(len(Xn), 1, device=DEV)], 1).double()
            G = Xa.T @ Xa
            G += torch.eye(len(G), device=DEV, dtype=G.dtype) * (1e-9 * G.diagonal().mean())
            w = torch.linalg.solve(G, Xa.T @ yn.double()).float().unsqueeze(1)

            def f(Z, w=w, mu=mu, sd=sd, ys_=ys_):
                Zn = (Z - mu) / sd
                return (torch.cat([Zn, torch.ones(len(Zn), 1, device=DEV)], 1) @ w).squeeze(1) * ys_
            info[half] = {"n_train": len(X), "coef": w.squeeze(1).tolist()}
        else:
            net = torch.nn.Sequential(torch.nn.Linear(m_in, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                                      torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            steps, bs = MLP_STEPS, 65536
            for it in range(steps):
                b = torch.randint(0, len(Xn), (bs,), device=DEV, generator=g)
                loss = ((net(Xn[b]).squeeze(1) - yn[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                tr_mse = float(((net(Xn[:200_000]).squeeze(1) - yn[:200_000]) ** 2).mean())

            def f(Z, net=net, mu=mu, sd=sd, ys_=ys_, ch=1 << 22):
                with torch.no_grad():
                    return torch.cat([net((Z[i:i + ch] - mu) / sd).squeeze(1)
                                      for i in range(0, len(Z), ch)]) * ys_
            info[half] = {"n_train": len(X), "train_mse_std": tr_mse, "steps": steps, "batch": bs}
        models[half] = f
    pred = {}
    for k, v in TH0.items():
        if not v.is_floating_point():
            pred[k] = v
            continue
        l = layer_of(k)
        half = 0 if l is None else 1 - l % 2
        pred[k] = v + (models[half](feats(k)) * LR).reshape(v.shape)
    n_par = sum(TH0[k].numel() for k in FKEYS)
    h = 64
    per_el = 2 * (m_in * h + h * h + h) if kind == "mlp" else 2 * (m_in + 1)
    flops = n_par * per_el + (2 * MLP_STEPS * 65536 * 3 * per_el if kind == "mlp" else 0)
    return pred, max(A), flops, info


def predictors(s, target):
    """yield (name, pred state dict, cost in main steps, extra flops, paid bits, meta)"""
    yield "theta", TH0, 0, 0, 0, {}
    b1 = {1: [1], 10: [10, 1], 100: [100, 10], 1000: [1000, 100]}[s]
    for run in B1_RUNS:
        for j in b1:
            p, pb = lin_pred(TH0, [(snap(run, j), TH0)], target)
            yield f"proxy {run}@{j}", p, j / 8, 0, pb, {"budget": s / (j / 8)}
    for run in SUB_RUNS:
        for j in (s // 10, s // 100):
            if j >= 1:
                p, pb = lin_pred(TH0, [(snap(run, j), TH0)], target)
                yield f"proxy {run}@{j}", p, j, 0, pb, {"budget": s / j}
    for a, b in ((1, 0), (10, 1), (100, 10), (10, 0), (100, 0)):
        if a <= s / 10:
            d = (snap("main", a), snap("main", b))
            p, pb = lin_pred(TH0, [d], target)
            yield f"extrap theta+t(th{a}-th{b})", p, a, 0, pb, {"budget": s / a}
            if b > 0:
                p, pb = lin_pred(snap("main", a), [d], target)
                yield f"extrap th{a}+t(th{a}-th{b})", p, a, 0, pb, {"budget": s / a}
    A = [a for a in SNAPS_MAIN if 0 < a <= s / 10]
    if A:
        dirs = [(snap("main", a), TH0) for a in A] + [(TH0, None)]
        p, pb = lin_pred(TH0, dirs, target)
        yield "pertensor LS(snaps<=s/10)", p, max(A), 0, pb, {"budget": s / max(A), "snaps": A}
        extra = [(snap(run, j), TH0) for run in B1_RUNS for j in b1[1:]] + \
                [(snap(run, j), TH0) for run in SUB_RUNS for j in (s // 100,) if j >= 1]
        if extra:
            cost = max(A) + len(B1_RUNS) * sum(j / 8 for j in b1[1:]) + \
                len(SUB_RUNS) * sum(j for j in (s // 100,) if j >= 1)
            p, pb = lin_pred(TH0, dirs + extra, target)
            yield "pertensor LS(snaps+proxies)", p, cost, 0, pb, {"budget": s / cost}
        for kind in ("lin", "mlp"):
            t0 = time.time()
            r = learned(s, target, kind)
            if r:
                p, cost, fl, info = r
                yield f"learned {kind} (half/half)", p, cost, fl, 0, {"budget": s / cost, "info": info,
                                                                       "fit_s": time.time() - t0}


# ---------------------------------------------------------------- main loop

def weight_rows(target, pred):
    rows, ls = [], []
    for l, n, k in WKEYS:
        rows.append(bits(TH0[k], target[k], pred[k], want_lzma=(l == LZ_LAYER)))
        ls.append(l)
    return finish(rows), ls


def weight_rows_prev(target, pred):
    """layer l's update given layer l-1's (the residual of layer l-1 against pred, scaled per tensor)"""
    rows, ls, nco = [], [], 0
    for l, n, k in WKEYS:
        if l == 0:
            continue
        kp = f"{LPFX}.{l - 1}.{n}.weight"
        d = target[kp] - pred[kp]
        y = target[k] - pred[k]
        c = float((d * y).sum() / ((d * d).sum() + 1e-30))
        rows.append(bits(TH0[k], target[k], pred[k] + c * d, want_lzma=(l == LZ_LAYER)))
        ls.append(l)
        nco += 1
    return finish(rows), ls, nco * COEF_BITS


def shrink(A0, A1, Ah):
    """per map, A0 + c (Ah - A0) with c fit by least squares (a nonlinear predictor's activations can overshoot)"""
    if Ah is A0:
        return A0, 0, {}
    out, cs = {}, {}
    for key in A1:
        d = Ah[key] - A0[key]
        c = float((d * (A1[key] - A0[key])).sum() / ((d * d).sum() + 1e-30))
        out[key], cs[f"{key[0]}:{key[1]}"] = A0[key] + c * d, c
    return out, sum(COEF_BITS for (l, n) in A1 if l in ROWS), cs


def act_rows(A0, A1, Ah):
    rows, ls = [], []
    for (l, n) in A1:
        if l not in ROWS:
            continue
        rows.append(bits(A0[(l, n)], A1[(l, n)], Ah[(l, n)], want_lzma=(l == LZ_LAYER)))
        ls.append(l)
    return finish(rows), ls


def act_rows_inputs(A0, A1, pred):
    """each map's output given its fresh inputs (the target's, from layer l-1) through the predictor's weights"""
    rows, ls = [], []
    for (l, n) in A1:
        if l not in ROWS:
            continue
        k = f"{LPFX}.{l}.{n}"
        Xh = torch.nn.functional.linear(INPUTS[(l, n)], pred[k + ".weight"], pred.get(k + ".bias"))
        rows.append(bits(A0[(l, n)], A1[(l, n)], Xh, want_lzma=(l == LZ_LAYER)))
        ls.append(l)
    return finish(rows), ls


def act_rows_prev(A0, A1, Ah):
    rows, ls, nco = [], [], 0
    for (l, n) in A1:
        if l == 0 or l not in ROWS:
            continue
        d = A1[(l - 1, n)] - Ah[(l - 1, n)]
        y = A1[(l, n)] - Ah[(l, n)]
        c = float((d * y).sum() / ((d * d).sum() + 1e-30))
        rows.append(bits(A0[(l, n)], A1[(l, n)], Ah[(l, n)] + c * d, want_lzma=(l == LZ_LAYER)))
        ls.append(l)
        nco += 1
    return finish(rows), ls, nco * COEF_BITS


def main():
    A0, loss0 = acts(TH0)
    meta = {"model": MODEL, "n_layers": NL, "linear_maps": LIN, "N_mm": N_MM, "F_step": F_STEP, "F_eval": F_EVAL,
            "eval_tokens": EVAL.numel(), "fmts": FMTS, "mdl_ranks": MDL_RANKS, "lzma_layer": LZ_LAYER,
            "loss_theta": loss0, "lr": LR, "coef_bits": COEF_BITS}
    res = {"meta": meta, "targets": {}}
    path = os.path.join(OUT, f"b2_{TAG}.json")
    for s in TARGETS:
        for key in [k for k in _cache if k != ("main", 0)]:
            del _cache[key]
        torch.cuda.empty_cache()
        target = snap("main", s)
        A1, loss1 = acts(target, keep_inputs=True)
        R = res["targets"][str(s)] = {"loss_target": loss1, "F_target": s * F_STEP, "preds": {}}
        for name, pred, cost, fl, pb, m in predictors(s, target):
            t0 = time.time()
            Ah, lossh = acts(pred) if pred is not TH0 else (A0, loss0)
            Ah, pa, cs = shrink(A0, A1, Ah)
            wr, wl = weight_rows(target, pred)
            ar, al = act_rows(A0, A1, Ah)
            n_w = sum(r["bf16"]["n"] for r in wr)
            rec = {"cost_main_steps": cost, "flops": cost * F_STEP + fl, "paid_bits": pb, "eval_loss": lossh,
                   "w": aggregate(wr, wl, pb * n_w / sum(TH0[k].numel() for k in FKEYS)),
                   "a": aggregate(ar, al, pa), "act_shrink": cs, **m}
            R["preds"][name] = rec
            if name == "theta" or name.startswith("pertensor LS(snaps+"):
                wr2, wl2, pb2 = weight_rows_prev(target, pred)
                ar2, al2, pa2 = act_rows_prev(A0, A1, Ah)
                R["preds"][f"prevlayer | {name}"] = {
                    "cost_main_steps": cost, "flops": cost * F_STEP + fl, "conditional": "layer l-1's fresh update",
                    "w": aggregate(wr2, wl2, pb2), "a": aggregate(ar2, al2, pa2),
                    "w_same_layers": aggregate(wr, wl, layer_filter=set(range(1, NL))),
                    "a_same_layers": aggregate(ar, al, layer_filter=set(range(1, NL)))}
                ar3, al3 = act_rows_inputs(A0, A1, pred)
                R["preds"][f"inputs | {name}"] = {
                    "cost_main_steps": cost, "flops": cost * F_STEP + fl, "conditional": "each map's fresh inputs",
                    "a": aggregate(ar3, al3)}
                if s == TARGETS[0] and name == "theta":
                    ar4, al4 = act_rows_inputs(A0, A1, target)
                    R["preds"]["inputs | target weights (sanity: ~0)"] = {"a": aggregate(ar4, al4)}
            print(f"{TAG} s={s} {name}: loss {lossh:.4f} W fp8 {rec['w']['fp8t']['best']:.3f} nvfp4 "
                  f"{rec['w']['nvfp4']['best']:.3f} | A fp8 {rec['a']['fp8t']['best']:.3f} nvfp4 "
                  f"{rec['a']['nvfp4']['best']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(path, "w"), indent=1)
        torch.cuda.empty_cache()
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
