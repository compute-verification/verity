"""Daniel's falsification test of the fan-out activation premise (B1): a block that rebuilds L_b held layers of theta' at FMT
from l_w bits, takes l_in more bits (the tokens of whole rollouts, a few scales, and theta''s residual entering the block at
boundary l, coded given theta's), and generates the held layers' counted activations, decoded for real and checked bit for
bit against theta''s own forward. Then beta_x <= (l_w + l_in + a_dec + lambda_x) / n_I.

  python block.py BASE CKPT     Env: OUT, FMT (fp8t | nvfp4), BOUNDARIES ("1,13"), LBS ("1,2,3"), K (2e7), NROLL (48),
                                MAXNEW (512), SEED (0), DATA (gsm8k_test.jsonl)

The computation checked is theta' deployed at FMT: every linear map's weight quantized at FMT with counted.py's scales (from
theta's weight), every linear map's input quantized at FMT with a per-tensor reference from theta's activations on these
tokens (sent, 32 bits each); norms, rotary, attention and the residual stream in fp32. The counted activations are the
quantized inputs of the four products per layer per token: attention input (q, k, v), o input, MLP input (gate, up) and
down input, 1536 + 1536 + 1536 + 8960 = 13,568 values.

Codes are real: one static categorical model per context (the context of quant.fresh_bits: the predicted code's binade and
its position in the cell), sent first, then rANS (constriction). Weights are coded given theta's at FMT; norms and biases at
bf16. The entering residual is coded at FMT (the substitution's F) or at bf16, given theta's residual at l (theta deployed
at FMT, computed by the decoder from the tokens). A counted activation the block computes differently from theta' is
repaired by coding theta''s code given the block's value (the same coder): l_fix. The block is exact by construction only
with its repairs; the mismatches before repair are reported.
"""
import gzip, hashlib, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
import constriction
import counted  # moves quant's grids to the device
import quant
from quant import encode, FORMATS

DEV = counted.DEV
OUT = os.environ.get("OUT", ".")
FMT = os.environ.get("FMT", "fp8t")
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
STOP_IDS = [151645, 151643]
LIN = ("q", "k", "v", "o", "gate", "up", "down")
HF = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
      "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
OTHER = {"ln1": "input_layernorm.weight", "ln2": "post_attention_layernorm.weight", "qb": "self_attn.q_proj.bias",
         "kb": "self_attn.k_proj.bias", "vb": "self_attn.v_proj.bias"}
INPUTS = ("attn", "o", "mlp", "down")


# ----------------------------------------------------------------------------------------------------------- the coder
def gamma(n):
    n = np.asarray(n, dtype=np.float64)
    return 2 * np.floor(np.log2(n)) + 1


def ans_code(sym, ctx):
    """codes sym (ints) under one static categorical model per context (each context's symbols and counts sent first, Elias
    gamma), then rANS; decodes it back. Returns (bits, decoded sym)."""
    shape, dev = sym.shape, sym.device
    n = sym.numel()
    if n == 0:
        return 0, sym.clone()
    cs, order = torch.sort(ctx.reshape(-1).long(), stable=True)
    ss = sym.reshape(-1).long()[order].cpu().numpy()
    cs, order = cs.cpu().numpy(), order.cpu().numpy()
    starts = np.concatenate([[0], np.flatnonzero(np.diff(cs)) + 1])
    ends = np.concatenate([starts[1:], [n]])
    side = float(gamma(len(starts)) + gamma(np.diff(np.concatenate([[-1], cs[starts]]))).sum())
    groups = []
    for a, b in zip(starts, ends):
        alph, idx, cnt = np.unique(ss[a:b], return_inverse=True, return_counts=True)
        side += float(gamma(len(alph)) + gamma(2 * np.abs(alph) + (alph < 0) + 1).sum() + gamma(cnt).sum())
        groups.append((alph, idx.astype(np.int32), cnt))
    enc = constriction.stream.stack.AnsCoder()
    for alph, idx, cnt in reversed(groups):
        if len(alph) > 1:
            enc.encode_reverse(idx, constriction.stream.model.Categorical(cnt / cnt.sum(), perfect=False))
    words = enc.get_compressed()
    dec = constriction.stream.stack.AnsCoder(words)
    out = np.empty(n, np.int64)
    for (alph, idx, cnt), a, b in zip(groups, starts, ends):
        got = (dec.decode(constriction.stream.model.Categorical(cnt / cnt.sum(), perfect=False), int(b - a))
               if len(alph) > 1 else np.zeros(b - a, np.int64))
        out[order[a:b]] = alph[got]
    return 32 * len(words) + side, torch.from_numpy(out).to(dev).reshape(shape)


def scale_value(scode, fmt, ref):
    if FORMATS[fmt]["scale"] == "e8m0":
        return torch.exp2(scode.float())
    return quant.G_E4M3_POS["g"][scode] * (ref / (6 * 448))


def deq_of(code, scode, fmt, ref):
    f = FORMATS[fmt]
    if f["kind"] == "plain":
        return quant.G_BF16["g"][code]
    if f["kind"] == "tensor":
        return quant.G_E4M3["g"][code] * (ref / 448)
    B = f["B"]
    sval = scale_value(scode, fmt, ref)
    return (f["elem"]["g"][code.reshape(*code.shape[:-1], code.shape[-1] // B, B)] * sval[..., None]).reshape(code.shape)


def code_many(items, fmt):
    """really codes each E1 (theta''s codes at fmt) given its float predictor Xhat, jointly over items [(Xhat, E1, ref)]:
    block scales first (given Xhat's), then elements given Xhat quantized under the decoded scales, in quant.fresh_bits'
    contexts. Returns (bits, scale bits, [(code, scode)] decoded)."""
    f = FORMATS[fmt]
    blocked = f["kind"] == "block"
    scodes, sbits = [None] * len(items), 0.0
    if blocked:
        B = f["B"]
        hats = [quant.block_scales(X.reshape(*X.shape[:-1], X.shape[-1] // B, B), f, ref)[0] for X, _, ref in items]
        ds = torch.cat([(E["scode"] - h).reshape(-1) for h, (_, E, _) in zip(hats, items)])
        sbits, dd = ans_code(ds, torch.zeros_like(ds))
        o = 0
        for j, h in enumerate(hats):
            scodes[j] = h + dd[o:o + h.numel()].reshape(h.shape)
            o += h.numel()
    Ps = [encode(X, fmt, ref, scale=(sc, scale_value(sc, fmt, ref))) if blocked else encode(X, fmt, ref)
          for (X, _, ref), sc in zip(items, scodes)]
    d = torch.cat([(E["code"] - P["code"]).reshape(-1) for P, (_, E, _) in zip(Ps, items)])
    c = torch.cat([quant.ctx_of(P, fmt).reshape(-1) for P in Ps])
    bits, dd = ans_code(d, c)
    out, o = [], 0
    for P, sc in zip(Ps, scodes):
        k = P["code"].numel()
        out.append((P["code"] + dd[o:o + k].reshape(P["code"].shape), sc))
        o += k
    return bits + sbits, sbits, out


def same(E, code, scode):
    m = E["code"] != code
    if scode is not None:
        B = FORMATS[FMT]["B"]
        m = m | (E["scode"] != scode).repeat_interleave(B, dim=-1).reshape(m.shape)
    return m


# ----------------------------------------------------------------------------------------------------------- the model
class Arch:
    def __init__(self, cfg):
        self.d, self.nh, self.nkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = self.d // self.nh
        self.eps, self.theta = cfg.rms_norm_eps, getattr(cfg, "rope_theta", 1e6)
        self.ffn, self.nl, self.vocab = cfg.intermediate_size, cfg.num_hidden_layers, cfg.vocab_size

    def rope(self, L):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=DEV, dtype=torch.int64).float() / self.hd))
        fr = torch.outer(torch.arange(L, device=DEV).float(), inv)
        emb = torch.cat([fr, fr], -1)
        return emb.cos(), emb.sin()


def params(sd, i):
    p = f"model.layers.{i}."
    out = {k: sd[p + HF[k] + ".weight"].float() for k in LIN}
    out.update({k: sd[p + v].float() for k, v in OTHER.items()})
    return out


def rms(h, w, eps):
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w


def rot(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin


def attention(A, p, x, cs):
    L = x.shape[0]
    q = (x @ p["q"].T + p["qb"]).view(L, A.nh, A.hd).transpose(0, 1)
    k = (x @ p["k"].T + p["kb"]).view(L, A.nkv, A.hd).transpose(0, 1)
    v = (x @ p["v"].T + p["vb"]).view(L, A.nkv, A.hd).transpose(0, 1)
    cos, sin = cs
    q, k = rot(q, cos, sin), rot(k, cos, sin)
    k, v = k.repeat_interleave(A.nh // A.nkv, 0), v.repeat_interleave(A.nh // A.nkv, 0)
    a = Fn.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
    return a.transpose(0, 1).reshape(L, A.d)


def layer(A, p, i, hs, css, fq):
    """one decoder layer over every sequence (each its own call, so the block and theta' run the same shapes); fq(i, name,
    [x per sequence]) quantizes (and captures, or repairs) one product input over all sequences at once"""
    xs = fq(i, "attn", [rms(h, p["ln1"], A.eps) for h in hs])
    xs = fq(i, "o", [attention(A, p, x, cs) for x, cs in zip(xs, css)])
    hs = [h + a @ p["o"].T for h, a in zip(hs, xs)]
    xs = fq(i, "mlp", [rms(h, p["ln2"], A.eps) for h in hs])
    xs = fq(i, "down", [Fn.silu(x @ p["gate"].T) * (x @ p["up"].T) for x in xs])
    return [h + m @ p["down"].T for h, m in zip(hs, xs)]


def split(X, lens):
    return list(torch.split(X, lens, 0))


# ----------------------------------------------------------------------------------------------------------- the run
def rollouts(model1, tok, rows, n, maxnew, seed):
    """whole rollouts of theta' (T = 1) on the first n problems, prompt and completion, cut after the first stop token"""
    tok.padding_side = "left"
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["q"] + SUFFIX}], add_generation_prompt=True,
                                       tokenize=False) for r in rows[:n]]
    torch.manual_seed(seed)
    seqs = []
    for j in range(0, n, 16):
        b = tok(prompts[j:j + 16], return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            g = model1.generate(**b, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                max_new_tokens=maxnew, eos_token_id=STOP_IDS, pad_token_id=STOP_IDS[1])
        for r in range(g.shape[0]):
            pl = int(b["attention_mask"][r].sum())
            ids = g[r, b["input_ids"].shape[1] - pl:].tolist()
            out, k = ids[:pl], pl
            while k < len(ids):
                out.append(ids[k])
                if ids[k] in STOP_IDS:
                    break
                k += 1
            seqs.append(torch.tensor(out, device=DEV))
    return seqs


def unique_rows(seqs):
    """per sequence, the mask of positions whose prefix (dependency cone) appears for the first time"""
    seen, masks = set(), []
    for s in seqs:
        h, m = hashlib.sha256(), []
        for t in s.tolist():
            h.update(int(t).to_bytes(4, "little"))
            k = h.copy().digest()
            m.append(k not in seen)
            seen.add(k)
        masks.append(torch.tensor(m, device=DEV))
    return masks


def main():
    base, ckpt = sys.argv[1], sys.argv[2]
    from transformers import AutoConfig, AutoTokenizer
    t0 = time.time()
    A = Arch(AutoConfig.from_pretrained(base))
    tok = AutoTokenizer.from_pretrained(base)
    model0 = counted.load_base(base).to(DEV)
    model1, meta = counted.load_trained(model0, ckpt)
    model1.to(DEV).eval()
    rows = [json.loads(x) for x in open(os.environ.get("DATA", "gsm8k_test.jsonl"))]
    nroll, K = int(os.environ.get("NROLL", "48")), float(os.environ.get("K", "2e7"))
    bnds = [int(x) for x in os.environ.get("BOUNDARIES", "1,13").split(",")]
    lbs = [int(x) for x in os.environ.get("LBS", "1,2,3").split(",")]
    top = max(bnds) + 1 + max(lbs)
    seqs = rollouts(model1, tok, rows, nroll, int(os.environ.get("MAXNEW", "512")), int(os.environ.get("SEED", "0")))
    lens = [len(s) for s in seqs]
    print(f"{len(seqs)} rollouts, {sum(lens)} tokens (mean {sum(lens) / len(lens):.0f}) at {time.time() - t0:.0f}s", flush=True)
    sd0, sd1 = model0.state_dict(), model1.state_dict()
    emb0, emb1 = sd0["model.embed_tokens.weight"].float(), sd1["model.embed_tokens.weight"].float()
    css = [A.rope(L) for L in lens]
    masks = unique_rows(seqs)
    with torch.no_grad():
        nll_bits = []
        for s in seqs:
            lp = Fn.log_softmax(model0(input_ids=s[None]).logits[0, :-1].float(), -1)
            nll_bits.append(float(-lp.gather(-1, s[1:, None]).sum() / math.log(2)) + math.log2(A.vocab))
    del model0, model1
    torch.cuda.empty_cache()

    # activation references (per product input, over every rollout) from theta's plain fp32 forward: sent, 32 bits each
    refs = {}
    P0 = {i: params(sd0, i) for i in range(top)}

    def fq_cal(i, name, xs):
        refs[(i, name)] = max(float(x.abs().max()) for x in xs)
        return xs
    with torch.no_grad():
        hs = [emb0[s] for s in seqs]
        for i in range(top):
            hs = layer(A, P0[i], i, hs, css, fq_cal)
    refs = {k: torch.tensor(v, device=DEV) for k, v in refs.items()}

    # theta and theta' deployed at FMT
    def deployed(P, Pref):
        out = dict(P)
        for k in LIN:
            out[k] = encode(P[k], FMT, Pref[k].abs().max())["deq"]
        return out
    P1 = {i: params(sd1, i) for i in range(top)}
    Q0 = {i: deployed(P0[i], P0[i]) for i in range(max(bnds) + 1)}
    Q1 = {i: deployed(P1[i], P0[i]) for i in range(top)}
    held = sorted({b + 1 + j for b in bnds for j in range(max(lbs))})
    ref_codes = {}

    def fq_ref(i, name, xs):
        X = torch.cat(xs)
        E = encode(X, FMT, refs[(i, name)])
        if i in held:
            ref_codes[(i, name)] = (E["code"].to(torch.int16), None if E["scode"] is None else E["scode"].to(torch.int16))
        return split(E["deq"], lens)

    def fq_plain(i, name, xs):
        return split(encode(torch.cat(xs), FMT, refs[(i, name)])["deq"], lens)
    hb0, hb1 = {}, {}
    with torch.no_grad():
        hs = [emb0[s] for s in seqs]
        for i in range(max(bnds) + 1):
            hs = layer(A, Q0[i], i, hs, css, fq_plain)
            if i in bnds:
                hb0[i] = torch.cat(hs)
        hs = [emb1[s] for s in seqs]
        for i in range(top):
            hs = layer(A, Q1[i], i, hs, css, fq_ref)
            if i in bnds:
                hb1[i] = torch.cat(hs)
        # the same boundaries of theta and theta' in plain fp32 (what counted.py's F reads), for comparison
        hp = {}
        for tag, P, emb in (("theta", P0, emb0), ("theta1", P1, emb1)):
            hs = [emb[s] for s in seqs]
            for i in range(max(bnds) + 1):
                hs = layer(A, P[i], i, hs, css, lambda i, n, xs: xs)
                if i in bnds:
                    hp[(tag, i)] = torch.cat(hs)
    print(f"references at {time.time() - t0:.0f}s", flush=True)
    urows = torch.cat(masks)
    fcmp = {}
    for b in bnds:
        for tag, X0, X1 in (("deployed", hb0[b], hb1[b]), ("plain_fp32", hp[("theta", b)], hp[("theta1", b)])):
            r_ = X0.abs().max()
            real = code_many([(X0, encode(X1, FMT, r_), r_)], FMT)[0] / X0.shape[0]
            X0u, X1u = X0[urows], X1[urows]
            est = counted.fresh(X0u, X1u, X0u.abs().max(), svd=counted.top_svd(X1u - X0u, max(counted.RANKS)), fmts=(FMT,))
            fcmp[f"{b}:{tag}"] = {"real_ctx_per_token": real, "est_ctx_per_token": est[FMT]["ctx"] / X0u.shape[0],
                                  "est_min_per_token": est[FMT]["min"] / X0u.shape[0], "est_min_by": est[FMT]["min_by"]}
            print(f"boundary {b} {tag}: F real ctx {real:.0f}, analyzer ctx {fcmp[f'{b}:{tag}']['est_ctx_per_token']:.0f}, "
                  f"analyzer min {fcmp[f'{b}:{tag}']['est_min_per_token']:.0f} ({est[FMT]['min_by']})", flush=True)
    del hp

    # l_w: the held layers' weights at FMT given theta's (one stream per layer), norms and biases at bf16; decoded for real
    lw, dec_layers = {}, {}
    for i in held:
        items = [(P0[i][k], encode(P1[i][k], FMT, P0[i][k].abs().max()), P0[i][k].abs().max()) for k in LIN]
        bits, sbits, got = code_many(items, FMT)
        oitems = [(P0[i][k], encode(P1[i][k], "bf16", None), None) for k in OTHER]
        obits, _, ogot = code_many(oitems, "bf16")
        p = {}
        exact = True
        for k, (_, E, ref), (code, scode) in zip(LIN, items, got):
            p[k] = deq_of(code, scode, FMT, ref)
            exact &= bool(torch.equal(p[k], E["deq"])) and bool(torch.equal(p[k], Q1[i][k]))
        for k, (code, _) in zip(OTHER, ogot):
            p[k] = deq_of(code, None, "bf16", None)
            exact &= bool(torch.equal(p[k], P1[i][k]))
        nw = sum(P0[i][k].numel() for k in LIN)
        changed = sum(int((P1[i][k] != P0[i][k]).sum()) for k in LIN)
        lw[i] = {"bits": bits + obits, "linear_bits": bits, "scale_bits": sbits, "other_bits": obits, "n_weights": nw,
                 "changed": changed / nw, "beta_w": bits / nw, "decoded_exact": exact}
        dec_layers[i] = p
        print(f"layer {i}: l_w {lw[i]['bits']:.4g} bits ({bits / nw:.4f} per weight), decoded exact {exact}", flush=True)
    del P1, Q0
    torch.cuda.empty_cache()

    a_dec = 8 * len(gzip.compress(b"".join(open(f, "rb").read() for f in (__file__, quant.__file__)), 9))
    res = {"ckpt": ckpt, "meta": meta, "fmt": FMT, "k": K, "a_dec_bits": a_dec, "n_rollouts": len(seqs), "lens": lens,
           "nll_token_bits": nll_bits, "layer_weight_bits": lw, "boundaries": {}, "F_compare": fcmp,
           "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]}
    macs_lin = sum(P0[held[0]][k].numel() for k in LIN)

    for b in bnds:
        rb = hb0[b].abs().max()
        R = res["boundaries"][b] = {"held": [b + 1 + j for j in range(max(lbs))], "blocks": []}
        for bfmt in (FMT, "bf16"):
            # the entering residual over every rollout, for the rate that sets Q
            bits_all, _, _ = code_many([(hb0[b], encode(hb1[b], bfmt, rb), rb)], bfmt)
            R[f"F_{bfmt}_all_per_token"] = bits_all / sum(lens)
            # Q fills k twice: at the substituted rate (tokens and the entering residual), then at the exact block's rate
            # (repairs included), measured on the first
            for lb, fill in [(lb, f) for lb in lbs for f in ("substituted", "exact")]:
                if fill == "substituted":
                    rate = math.log2(A.vocab) + bits_all / sum(lens)
                hl = [b + 1 + j for j in range(lb)]
                wbits = sum(lw[i]["bits"] for i in hl)
                scale_bits = 32 * (4 * (b + 1 + lb) + 1)
                room = K - wbits - a_dec - scale_bits
                m, used = 0, 0.0
                while m < len(seqs) and used + lens[m] * rate <= room:
                    used += lens[m] * rate
                    m += 1
                m = max(m, 1)
                T = sum(lens[:m])
                n_u = int(sum(int(x.sum()) for x in masks[:m]))
                rows_u = torch.cat(masks[:m])
                Eb = encode(hb1[b][:T], bfmt, rb)
                bb, _, ((bc, bsc),) = code_many([(hb0[b][:T], Eb, rb)], bfmt)
                hhat = deq_of(bc, bsc, bfmt, rb)
                tok_bits = T * math.log2(A.vocab)
                blk = {"bfmt": bfmt, "L_b": lb, "held": hl, "rollouts": m, "tokens": T, "unique_rows": n_u,
                       "l_w": wbits, "l_tok": tok_bits, "l_tok_theta_nll": sum(nll_bits[:m]), "l_scales": scale_bits,
                       "l_bnd": bb, "F_per_token": bb / T, "bnd_decoded_exact": bool(torch.equal(hhat, Eb["deq"])),
                       "n_I": n_u * lb * (3 * A.d + A.ffn), "layers": {}}
                # the block, twice: as substituted (no repair), and exact (repairs coded and applied)
                for mode in ("naive", "repaired"):
                    fix = {}

                    def fq_blk(i, name, xs, mode=mode, fix=fix):
                        X = torch.cat(xs)
                        ref = refs[(i, name)]
                        E = encode(X, FMT, ref)
                        rc, rs = ref_codes[(i, name)]
                        rc, rs = rc[: X.shape[0]].long(), None if rs is None else rs[: X.shape[0]].long()
                        mis = same(E, rc, rs)
                        st = {"mismatch": int(mis[rows_u].sum()), "values": int(rows_u.sum()) * X.shape[1]}
                        if mode == "repaired":
                            st["l_fix"], st["after_repair_mismatch"] = 1, 0  # one flag bit: this input needs no repair
                            if bool(mis.any()):
                                Er = {"code": rc, "scode": rs}
                                bits, _, ((c2, s2),) = code_many([(X, Er, ref)], FMT)
                                st["l_fix"] = 1 + bits
                                E = {"deq": deq_of(c2, s2, FMT, ref), "code": c2, "scode": s2}
                                st["after_repair_mismatch"] = int(same(E, rc, rs)[rows_u].sum())
                        fix[(i, name)] = st
                        return split(E["deq"], lens[:m])
                    with torch.no_grad():
                        hs = split(hhat, lens[:m])
                        for i in hl:
                            hs = layer(A, dec_layers[i], i, hs, css[:m], fq_blk)
                    for (i, name), st in fix.items():
                        blk["layers"].setdefault(str(i), {}).setdefault(name, {})[mode] = st
                l_fix = sum(v["repaired"]["l_fix"] for L_ in blk["layers"].values() for v in L_.values())
                mism = sum(v["naive"]["mismatch"] for L_ in blk["layers"].values() for v in L_.values())
                after = sum(v["repaired"]["after_repair_mismatch"] for L_ in blk["layers"].values() for v in L_.values())
                base_bits = blk["l_w"] + blk["l_tok"] + blk["l_scales"] + blk["l_bnd"] + a_dec
                attn1 = sum(A.nh * A.hd * L * (L + 1) for L in lens[:m])  # causal QK^T and PV, one layer
                lin1 = macs_lin * T
                # the decoder's work: theta deployed at FMT up to l, then the held layers, plus ~60 operations per decoded
                # or dequantized symbol (weights, the entering residual, the counted activations)
                syms = sum(lw[i]["n_weights"] for i in hl) + T * A.d + T * lb * (3 * A.d + A.ffn)
                blk.update({"l_fix": l_fix, "mismatch_naive": mism, "mismatch_naive_frac": mism / (blk["n_I"] or 1),
                            "mismatch_after_repair": after, "l_total_substituted": base_bits,
                            "l_total_exact": base_bits + l_fix,
                            "beta_x_bound_substituted": base_bits / blk["n_I"],
                            "beta_x_bound_exact": (base_bits + l_fix) / blk["n_I"],
                            "decode_flops": 2 * (lin1 + attn1) * (b + 1 + lb) + 60 * syms,
                            "products_per_entering_coord_linear": lin1 * lb / (T * A.d),
                            "products_per_entering_coord_with_attn": (lin1 + attn1) * lb / (T * A.d)})
                blk["decode_flops_over_c"] = blk["decode_flops"] / 1.34e17
                blk["fill"] = fill
                rate = (blk["l_tok"] + blk["l_bnd"] + l_fix) / T
                R["blocks"].append(blk)
                print(f"l={b} L_b={lb} {bfmt} fill {fill}: Q={T} ({m} rollouts) l_w={blk['l_w']:.4g} l_tok={tok_bits:.4g} "
                      f"l_bnd={bb:.4g} (F {bb / T:.0f}) l_fix={l_fix:.4g} n_I={blk['n_I']:.4g} "
                      f"bound {blk['beta_x_bound_substituted']:.4f} / exact {blk['beta_x_bound_exact']:.4f}; mismatches "
                      f"{mism} ({blk['mismatch_naive_frac']:.4f}), after repair {after} at {time.time() - t0:.0f}s",
                      flush=True)
                json.dump(res, open(os.path.join(OUT, f"block_{FMT}.json"), "w"), indent=1, default=float)
    res["secs"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, f"block_{FMT}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
