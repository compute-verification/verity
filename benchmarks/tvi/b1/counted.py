"""Fresh bits of a trained model theta' over its registered theta, on the values the production bound counts (B1, crux 2).

Counted values (note:console/20261024T1810Z-finding-production-bound-red-team, section 4): the outputs of parameterized
(linear) maps, min(d_in, d_out) per token per map, deduplicated by dependency cone. A map whose input depends only on the
token id (found by probing, e.g. layer 0's q/k/v) counts once per distinct token; every other map once per distinct prefix.

Per map, at bf16, FP8 (E4M3, per-tensor scale from theta) and NVFP4:
  beta_v    bits to code theta''s quantized output given theta's output, per counted value;
  beta_v|x  the same given theta's weights applied to theta''s input (the part a block holding the input still lacks);
  beta_x    bits to code theta''s quantized input given theta's input, per input coordinate;
  beta_w    bits to code theta''s quantized weights given theta's weights, per weight of an updated matrix.
At every block boundary (a decoder layer's output h): the fresh residual h' - h, its effective rank at 90% and 99% energy and
its compressed size per token (F), forward and backward (dL/dh), and on some boundaries (layers 1, 2, L/8, L/4, L/2, 3L/4
and the fp8-cheapest) the share of the trained model's shift that survives when the receiving block gets only a rank-r
correction h + Q(C) V_r (r = 0 is "send nothing").

Every size is a minimum over compressors: console's context coder (residual given the predicted code's binade and cell
position), a per-channel affine predictor, low-rank MDL (rank r factors at 16 or 8 bits, then the context coder), and lzma on
a few layers. Maps and weights are measured on LAYERS_SAMPLE evenly spaced layers; boundaries on every layer.

  python counted.py BASE CKPT NAME [CKPT NAME ...]   CKPT: a train.py snapshot (.pt), an HF model dir, or an adapter dir
Env: OUT, TOKENS (a .pt with "held", or a .txt), NSEQ (4), SEQ (512), LAYERS_SAMPLE (8), DEVICE.
"""
import glob, hashlib, json, lzma, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import quant
from quant import encode, fresh_bits

DEV = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
for _g in (quant.G_BF16, quant.G_E4M3, quant.G_E4M3_POS, quant.G_E2M1):
    for _k in list(_g):
        _g[_k] = _g[_k].to(DEV)

FMTS = ("bf16", "fp8t", "nvfp4")
RANKS = (1, 4, 16, 64)
WRANKS = (1, 4, 8, 16, 64, 256)
FBITS = (16, 8)
PATCH_RANKS = (0, 1, 4, 16, 64, 256, 512)
LZMA_MAX = 1 << 22
OUT = os.environ.get("OUT", ".")


# ----------------------------------------------------------------------------------------------------------- compressors
def top_svd(D, rmax):
    """top-rmax (U S, V^T) of D (m x n) and the cumulative energy fractions, from the smaller Gram matrix (float32: these
    GPUs have little float64, and the top components and 90%/99% energy fractions don't need it)"""
    m, n = D.shape
    Dd = D.float()
    if n <= m:
        ev, V = torch.linalg.eigh(Dd.T @ Dd)
        ev, V = ev.flip(0).clamp_min(0), V.flip(1)
        r = min(rmax, n)
        US, Vt = Dd @ V[:, :r], V[:, :r].T
    else:
        ev, U = torch.linalg.eigh(Dd @ Dd.T)
        ev, U = ev.flip(0).clamp_min(0), U.flip(1)
        r = min(rmax, m)
        S = ev[:r].sqrt()
        US = U[:, :r] * S
        Vt = (U[:, :r].T @ Dd) / S.clamp_min(1e-30)[:, None]
    e = ev.double().cumsum(0)
    tot = float(e[-1])
    frac = e / e[-1] if tot > 0 else e
    return US.float(), Vt.float().contiguous(), frac


def eff_rank(frac, p):
    return int((frac < p).sum()) + 1 if float(frac[-1]) > 0 else 0


def qfactor(A, fb):
    """factor A (m x r) at fb bits: bf16, or E4M3 with a per-column scale"""
    if fb == 16:
        return A.bfloat16().float()
    s = A.abs().amax(0, keepdim=True).clamp_min(1e-30) / 448
    return (A / s).to(torch.float8_e4m3fn).float() * s


def lzma_bits(t):
    t = t.reshape(-1)
    lo, hi = int(t.min()), int(t.max())
    dt = torch.int8 if -128 <= lo and hi <= 127 else (torch.int16 if -32768 <= lo and hi <= 32767 else torch.int32)
    return 8 * len(lzma.compress(t.to(dt).cpu().numpy().tobytes(), preset=6))


def affine(X0, X1):
    """per-column least-squares X1 ~ a X0 + b (theta' columns from theta's), the predictor and its 32 bits per column"""
    m0, m1 = X0.mean(0), X1.mean(0)
    c0 = X0 - m0
    a = (c0 * (X1 - m1)).sum(0) / (c0 * c0).sum(0).clamp_min(1e-30)
    a, b = a.bfloat16().float(), (m1 - a * m0).bfloat16().float()
    return X0 * a + b


def fresh(X0, X1, ref, svd=None, ranks=RANKS, fbits=FBITS, lz=False, aff=True, fmts=FMTS):
    """bits to code X1 quantized at each fmt given the predictor X0 (theta's values, or any float predictor): every
    compressor's total and the minimum. svd = top_svd(X1 - X0) enables low-rank MDL; `mdl_coef` is MDL without the basis
    (the per-row coefficients only, the cost once the basis is amortized over many more rows)."""
    T, n = X1.shape
    A = affine(X0, X1) if aff else None
    out = {}
    for fmt in fmts:
        E1 = encode(X1, fmt, ref)
        m = {}
        b, d, ds = fresh_bits(X0, E1, fmt, ref)
        m["ctx"] = b
        if aff:
            m["affine"] = fresh_bits(A, E1, fmt, ref)[0] + 32 * n
        if lz and X1.numel() <= LZMA_MAX:
            lb = min(lzma_bits(d), lzma_bits(d.T.contiguous())) + (lzma_bits(ds) if ds is not None else 0)
            m["lzma"] = lb
        if svd is not None:
            US, Vt, _ = svd
            best, coef = (b, 0, 0), b
            for fb in fbits:
                for r in ranks:
                    if r > US.shape[1] or r >= min(T, n):
                        break
                    P = X0 + qfactor(US[:, :r], fb) @ qfactor(Vt[:r].T, fb).T
                    rb = fresh_bits(P, E1, fmt, ref)[0]
                    tot = rb + r * (T + n) * fb
                    if tot < best[0]:
                        best = (tot, r, fb)
                    coef = min(coef, rb + r * T * fb)
            m["mdl"], m["mdl_r"], m["mdl_fb"], m["mdl_coef"] = best[0], best[1], best[2], coef
        m["frac"] = float((E1["deq"] != encode(X0, fmt, ref)["deq"]).float().mean())
        keys = [k for k in ("ctx", "affine", "lzma", "mdl") if k in m]
        m["min"] = min(m[k] for k in keys)
        m["min_by"] = min(keys, key=lambda k: m[k])
        out[fmt] = m
    return out


# ----------------------------------------------------------------------------------------------------------------- model
def load_base(base):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()


def load_trained(base_model, ckpt):
    """theta' as a full model: a train.py snapshot, a (possibly sharded) safetensors model dir, or LoRA factors (dW = B A)"""
    import copy
    m = copy.deepcopy(base_model)
    sd = m.state_dict()
    if ckpt.endswith(".pt"):
        snap = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(snap["sd"])
        return m, {"lora_params": snap.get("lora_params")}
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(ckpt, "meta.json"))) if os.path.exists(os.path.join(ckpt, "meta.json")) else {}
    ad = os.path.join(ckpt, "adapter_model.safetensors")
    if os.path.exists(ad):
        cfg = json.load(open(os.path.join(ckpt, "adapter_config.json")))
        scale = cfg.get("lora_alpha", cfg["r"]) / cfg["r"]
        f = load_file(ad)
        n_mod = 0
        with torch.no_grad():
            for k, A in f.items():
                if not k.endswith("lora_A.weight"):
                    continue
                mod = k[len("base_model.model."):-len(".lora_A.weight")]
                B = f[k.replace("lora_A", "lora_B")]
                W = sd[mod + ".weight"]
                W.add_(((B.float() @ A.float()) * scale).to(W.device))
                n_mod += 1
        meta["loaded"] = f"adapter: {n_mod} maps"
        return m, meta
    files = sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no checkpoint in {ckpt}")
    got = set()
    with torch.no_grad():
        for fp in files:
            for k, v in load_file(fp).items():
                if k in sd:
                    sd[k].copy_(v.float())
                    got.add(k)
    missing = [k for k in sd if k not in got and not k.endswith("lm_head.weight")]
    if len(got) == 0 or len(missing) > 0.05 * len(sd):
        raise RuntimeError(f"{ckpt}: loaded {len(got)} tensors, {len(missing)} missing (e.g. {missing[:3]})")
    meta["loaded"] = f"full: {len(got)} tensors"
    return m, meta


def layers_of(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    return model.model.layers


def linears(model):
    """[(layer index, name within the layer, module)] for every nn.Linear inside the decoder layers"""
    out = []
    for i, layer in enumerate(layers_of(model)):
        for n, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                out.append((i, n, mod))
    return out


def capture(model, ids, want=("x", "y", "h")):
    """one no-grad forward: every linear map's input (deduplicated by tensor within a layer) and output, every boundary h"""
    acts, groups, hooks, seen, anchor = {}, {}, [], {}, []

    def lin_hook(i, n):
        def f(mod, inp, out):
            x = inp[0]
            anchor.append(x)  # held until the forward ends, so no later input reuses its address
            key = (i, x.data_ptr(), tuple(x.shape))
            g = seen.setdefault(key, f"L{i}.{n}")
            groups[f"L{i}.{n}"] = g
            if "x" in want and g not in acts.get("x", {}):
                acts.setdefault("x", {})[g] = x.detach().reshape(-1, x.shape[-1]).float().clone()
            if "y" in want:
                acts.setdefault("y", {})[f"L{i}.{n}"] = out.detach().reshape(-1, out.shape[-1]).float().clone()
        return f

    def layer_hook(i):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            acts.setdefault("h", {})[i] = o.detach().reshape(-1, o.shape[-1]).float().clone()
        return f

    for i, n, mod in linears(model):
        hooks.append(mod.register_forward_hook(lin_hook(i, n)))
    if "h" in want:
        for i, layer in enumerate(layers_of(model)):
            hooks.append(layer.register_forward_hook(layer_hook(i)))
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    anchor.clear()
    return acts, groups, None


def boundary_grads(model, ids):
    """dL/dh at every decoder layer's output (next-token loss averaged over the batch), with the parameters frozen"""
    for p in model.parameters():
        p.requires_grad_(False)
    keep, hooks = {}, []

    def emb_hook(mod, inp, out):
        return out.requires_grad_(True)

    def layer_hook(i):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad()
            keep[i] = o
        return f

    hooks.append(model.get_input_embeddings().register_forward_hook(emb_hook))
    for i, layer in enumerate(layers_of(model)):
        hooks.append(layer.register_forward_hook(layer_hook(i)))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    for h in hooks:
        h.remove()
    g = {i: t.grad.detach().reshape(-1, t.shape[-1]).float().clone() for i, t in keep.items()}
    del keep, loss
    return g


def nll_and_logp(model, ids, patch=None):
    """next-token loss and log-probs, with layer `patch[0]`'s output replaced by patch[1](output)"""
    hook = None
    if patch is not None:
        li, fn = patch

        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            new = fn(o.reshape(-1, o.shape[-1])).reshape(o.shape).to(o.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        hook = layers_of(model)[li].register_forward_hook(f)
    with torch.no_grad():
        logits = model(input_ids=ids).logits.float()
    if hook is not None:
        hook.remove()
    logp = Fn.log_softmax(logits[:, :-1], -1)
    nll = -logp.gather(-1, ids[:, 1:, None]).mean()
    return float(nll), logp


def kl(lp, lq):
    """mean over tokens of KL(p || q) from log-probs"""
    return float((lp.exp() * (lp - lq)).sum(-1).mean())


def qcoef(C, kind):
    """per-token quantization of the rank-r coefficients: E4M3 with a per-row scale, or symmetric int4 with a per-row scale"""
    if kind == "exact":
        return C
    s = C.abs().amax(1, keepdim=True).clamp_min(1e-30)
    if kind == "fp8":
        return (C / s * 448).to(torch.float8_e4m3fn).float() * s / 448
    return torch.round(C / s * 7).clamp(-7, 7) * s / 7


# ------------------------------------------------------------------------------------------------------------- the cones
def cone_rows(ids, local):
    """row indices of the flattened (B*L) positions to count, once per distinct dependency cone"""
    flat = ids.reshape(-1).tolist()
    if local:
        first = {}
        for j, t in enumerate(flat):
            first.setdefault(t, j)
        return torch.tensor(sorted(first.values()), device=ids.device)
    rows, seen = [], set()
    B, L = ids.shape
    for b in range(B):
        h = hashlib.sha256()
        for t in range(L):
            h.update(int(ids[b, t]).to_bytes(4, "little"))
            k = h.copy().digest()
            if k not in seen:
                seen.add(k)
                rows.append(b * L + t)
    return torch.tensor(rows, device=ids.device)


def probe_local(model, ids):
    """map groups whose input at a position depends only on that position's token id"""
    a = ids[0, :64].clone()
    bseq = a.clone()
    bseq[:32] = ids[1, :32]
    acts, groups, _ = capture(model, torch.stack([a, bseq]), want=("x",))
    loc = set()
    for g, X in acts["x"].items():
        X = X.reshape(2, 64, -1)
        if torch.allclose(X[0, 32:], X[1, 32:], rtol=1e-4, atol=1e-6 * float(X.abs().max() + 1e-30)):
            loc.add(g)
    return loc


# ------------------------------------------------------------------------------------------------------------------ main
def eval_ids(tokens_path, nseq, L):
    if tokens_path.endswith(".pt"):
        held = torch.load(tokens_path)["held"]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ["TOKENIZER"])
        held = torch.tensor(tok(open(tokens_path, encoding="utf-8").read())["input_ids"])
    starts = torch.linspace(0, len(held) - L - 1, nseq).long()
    return torch.stack([held[s:s + L] for s in starts])


def sample_layers(n, k):
    if k >= n:
        return list(range(n))
    return sorted({round(j * (n - 1) / (k - 1)) for j in range(k)})


def analyze(base_name, model0, ids, local, ckpt, name, extra):
    t0 = time.time()
    model1, meta = load_trained(model0, ckpt)
    model1.to(DEV).eval()
    nl = len(layers_of(model0))
    sampled = sample_layers(nl, int(os.environ.get("LAYERS_SAMPLE", "8")))
    lz_layers = {sampled[len(sampled) // 2]}  # lzma (CPU) never beat the context coder in console's sweep: one check layer
    code = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    res = {"model": base_name, "name": name, "ckpt": ckpt, "meta": meta, **extra, "code_sha256": code, "tokens": ids.numel(),
           "n_layers": nl, "layers_sampled": sampled, "maps": {}, "inputs": {}, "weights": {}, "boundary": {},
           "retention": {}}

    def tick(what):
        print(f"  {name}: {what} at {time.time() - t0:.0f}s", flush=True)

    g0, g1 = boundary_grads(model0, ids), boundary_grads(model1, ids)
    torch.cuda.empty_cache() if DEV == "cuda" else None
    a0, groups, _ = capture(model0, ids)
    a1, _, _ = capture(model1, ids)
    l0, lp0 = nll_and_logp(model0, ids)
    l1, lp1 = nll_and_logp(model1, ids)
    res["loss"] = {"theta": l0, "theta1": l1, "kl_theta1_theta": kl(lp1, lp0)}
    rows_all = cone_rows(ids, False)
    rows_cache = {}
    tick("grads and captures")

    def rows_for(g):
        if g not in rows_cache:
            rows_cache[g] = cone_rows(ids, True) if g in local else rows_all
        return rows_cache[g]

    # boundaries: every layer
    for i in range(nl):
        h0, h1 = a0["h"][i][rows_all], a1["h"][i][rows_all]
        T = h0.shape[0]
        svd = top_svd(h1 - h0, max(RANKS))
        r = {"r90": eff_rank(svd[2], 0.90), "r99": eff_rank(svd[2], 0.99),
             "rel_dnorm": float((h1 - h0).norm() / h0.norm()), "d": h0.shape[1], "T": T}
        f = fresh(h0, h1, h0.abs().max(), svd=svd, lz=i in lz_layers)
        for fmt in FMTS:
            f[fmt]["per_token"] = f[fmt]["min"] / T
            f[fmt]["coef_per_token"] = f[fmt]["mdl_coef"] / T
        r["fwd"] = f
        q0, q1 = g0[i][rows_all], g1[i][rows_all]
        gs = top_svd(q1 - q0, max(RANKS))
        gf = fresh(q0, q1, q0.abs().max(), svd=gs)
        for fmt in FMTS:
            gf[fmt]["per_token"] = gf[fmt]["min"] / T
        r["bwd"] = {"r90": eff_rank(gs[2], 0.90), "r99": eff_rank(gs[2], 0.99),
                    "rel_dnorm": float((q1 - q0).norm() / q0.norm().clamp_min(1e-30)), **gf}
        res["boundary"][i] = r
        del svd, gs
    del g0, g1
    tick("boundaries")

    # retention of the shift under a rank-r correction, on early and evenly spaced boundaries and the fp8-cheapest one
    cheap = min(range(nl), key=lambda i: res["boundary"][i]["fwd"]["fp8t"]["per_token"])
    for i in sorted({1, 2, nl // 8, nl // 4, nl // 2, (3 * nl) // 4, cheap} & set(range(nl))):
        h0, h1 = a0["h"][i][rows_all], a1["h"][i][rows_all]
        Vt = top_svd(h1 - h0, max(PATCH_RANKS))[1]
        h0_full = a0["h"][i]
        ret = {"ranks": [], "nll": {}, "kl": {}}
        for rk in PATCH_RANKS:
            if rk > Vt.shape[0]:
                break
            ret["ranks"].append(rk)
            for kind in ("exact", "fp8", "int4"):
                if rk == 0 and kind != "exact":
                    continue
                V = Vt[:rk]

                def fn(o, V=V, kind=kind, rk=rk):
                    if rk == 0:
                        return h0_full
                    C = (o.float() - h0_full) @ V.T
                    return h0_full + qcoef(C, kind) @ V
                lp_, lpp = nll_and_logp(model1, ids, patch=(i, fn))
                ret["nll"].setdefault(kind, []).append(lp_)
                ret["kl"].setdefault(kind, []).append(kl(lp1, lpp))
                del lpp
        res["retention"][i] = ret
        del Vt
    tick("retention")

    # counted values: maps on the sampled layers
    lin0 = {f"L{i}.{n}": mod for i, n, mod in linears(model0)}
    for key, Y1 in a1["y"].items():
        i = int(key[1:].split(".")[0])
        if i not in sampled:
            continue
        Y0 = a0["y"][key]
        g = groups[key]
        rows = rows_for(g)
        X0, X1 = a0["x"][g][rows], a1["x"][g][rows]
        Y0, Y1 = Y0[rows], Y1[rows]
        mod = lin0[key]
        d_in, d_out = mod.in_features, mod.out_features
        ref = Y0.abs().max()
        svd = top_svd(Y1 - Y0, max(RANKS))
        m = {"d_in": d_in, "d_out": d_out, "local": g in local, "cones": len(rows),
             "counted": len(rows) * min(d_in, d_out), "input": g,
             "r90": eff_rank(svd[2], 0.90), "r99": eff_rank(svd[2], 0.99),
             "rel_dnorm": float((Y1 - Y0).norm() / Y0.norm())}
        m["y"] = fresh(Y0, Y1, ref, svd=svd, lz=(i in lz_layers and d_out <= d_in))
        del svd
        with torch.no_grad():
            P = X1 @ mod.weight.T + (mod.bias if mod.bias is not None else 0)
        svd = top_svd(Y1 - P, max(RANKS))
        m["y_given_x"] = fresh(P, Y1, ref, svd=svd, fbits=(16,), aff=False)
        del svd, P
        res["maps"][key] = m
        if g not in res["inputs"]:
            svd = top_svd(X1 - X0, max(RANKS))
            res["inputs"][g] = {"d_in": d_in, "cones": len(rows), "r90": eff_rank(svd[2], 0.90),
                                "r99": eff_rank(svd[2], 0.99), "rel_dnorm": float((X1 - X0).norm() / X0.norm()),
                                "x": fresh(X0, X1, X0.abs().max(), svd=svd)}
            del svd
    del a0, a1
    tick("maps")

    # weights: the sampled layers' linear maps, and the embedding (and an untied head)
    sd0, sd1 = model0.state_dict(), model1.state_dict()
    name_of = {id(mod): k for k, mod in model0.named_modules()}
    for i, n, mod in linears(model0):
        if i in sampled:
            wname = name_of[id(mod)] + ".weight"
            res["weights"][f"L{i}.{n}"] = weight_entry(sd0[wname].float(), sd1[wname].float())
    emb = model0.get_input_embeddings().weight
    ename = [k for k, v in sd0.items() if v.data_ptr() == emb.data_ptr()][0]
    res["weights"]["embed"] = weight_entry(sd0[ename].float(), sd1[ename].float())
    head = model0.get_output_embeddings()
    if head is not None and head.weight.data_ptr() != emb.data_ptr():
        hname = [k for k, v in sd0.items() if v.data_ptr() == head.weight.data_ptr()][0]
        res["weights"]["head"] = weight_entry(sd0[hname].float(), sd1[hname].float())
    res["summary"] = summarize(res)
    res["secs"] = time.time() - t0
    del model1
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return res


def weight_entry(W0, W1):
    n = W0.numel()
    ch = float((W1 != W0).float().mean())
    e = {"n": n, "shape": list(W0.shape), "changed": ch, "rel_dnorm": float((W1 - W0).norm() / W0.norm())}
    if ch == 0:
        e.update({fmt: {"min": 0.0, "min_by": "unchanged"} for fmt in FMTS})
        return e
    svd = top_svd(W1 - W0, max(WRANKS))
    e["r90"], e["r99"] = eff_rank(svd[2], 0.90), eff_rank(svd[2], 0.99)
    e.update(fresh(W0, W1, W0.abs().max(), svd=svd, ranks=WRANKS))
    return e


def summarize(res):
    s = {}
    maps = res["maps"]
    for fmt in FMTS:
        cnt = sum(m["counted"] for m in maps.values())
        outv = sum(m["cones"] * m["d_out"] for m in maps.values())
        by = {}
        for k, m in maps.items():
            t = k.split(".", 1)[1]
            b = by.setdefault(t, [0.0, 0])
            b[0] += m["y"][fmt]["min"]
            b[1] += m["counted"]
        inp = res["inputs"].values()
        wts = [w for k, w in res["weights"].items() if k.startswith("L") and w["changed"] > 0]
        bnd = res["boundary"]
        F = [bnd[i]["fwd"][fmt]["per_token"] for i in sorted(bnd)]
        Fb = [bnd[i]["bwd"][fmt]["per_token"] for i in sorted(bnd)]
        s[fmt] = {
            "beta_v": sum(m["y"][fmt]["min"] for m in maps.values()) / cnt,
            "beta_v_per_output": sum(m["y"][fmt]["min"] for m in maps.values()) / outv,
            "beta_v_given_x": sum(m["y_given_x"][fmt]["min"] for m in maps.values()) / cnt,
            "beta_v_ctx_only": sum(m["y"][fmt]["ctx"] for m in maps.values()) / cnt,
            "beta_v_by_map": {t: b[0] / b[1] for t, b in by.items()},
            "beta_x": sum(x["x"][fmt]["min"] for x in inp) / sum(x["cones"] * x["d_in"] for x in inp),
            "beta_w": (sum(w[fmt]["min"] for w in wts) / sum(w["n"] for w in wts)) if wts else 0.0,
            "beta_w_embed": res["weights"]["embed"][fmt]["min"] / res["weights"]["embed"]["n"],
            "update_bits_est": (sum(w[fmt]["min"] for w in wts) * res["n_layers"] / len(res["layers_sampled"])
                                + res["weights"]["embed"][fmt]["min"]
                                + res["weights"].get("head", {}).get(fmt, {}).get("min", 0.0)),
            "F_fwd_per_layer": F, "F_bwd_per_layer": Fb,
            "F_fwd_min": min(F), "F_fwd_median": sorted(F)[len(F) // 2], "F_bwd_min": min(Fb),
            "F_fwd_coef_min": min(bnd[i]["fwd"][fmt]["coef_per_token"] for i in bnd),
        }
    s["r90_per_layer"] = [res["boundary"][i]["r90"] for i in sorted(res["boundary"])]
    s["r99_per_layer"] = [res["boundary"][i]["r99"] for i in sorted(res["boundary"])]
    s["bwd_r90_per_layer"] = [res["boundary"][i]["bwd"]["r90"] for i in sorted(res["boundary"])]
    s["bwd_r99_per_layer"] = [res["boundary"][i]["bwd"]["r99"] for i in sorted(res["boundary"])]
    return s


def main():
    base = sys.argv[1]
    pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
    torch.manual_seed(0)
    model0 = load_base(base).to(DEV)
    ids = eval_ids(os.environ["TOKENS"], int(os.environ.get("NSEQ", "4")), int(os.environ.get("SEQ", "512"))).to(DEV)
    local = probe_local(model0, ids)
    print(f"{base}: local map inputs {sorted(local)}", flush=True)
    extra = json.loads(os.environ.get("EXTRA", "{}"))
    for ckpt, name in pairs:
        try:
            res = analyze(base, model0, ids, local, ckpt, name, extra.get(name, {}))
        except Exception as e:  # noqa: BLE001  one bad checkpoint must not lose the others
            import traceback
            traceback.print_exc()
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)
            continue
        s = res["summary"]
        print(f"{name}: {res['secs']:.0f}s loss {res['loss']['theta']:.3f}->{res['loss']['theta1']:.3f} "
              + " ".join(f"{f}: bv={s[f]['beta_v']:.3f} bx={s[f]['beta_x']:.3f} bw={s[f]['beta_w']:.3f} "
                         f"Fmin={s[f]['F_fwd_min']:.0f}" for f in FMTS), flush=True)
        json.dump(res, open(os.path.join(OUT, f"counted_{name}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
