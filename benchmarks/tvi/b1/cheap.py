"""The cheap-input block (B1, after C1's estimate): a block that holds theta''s embedding table at EMB (the rows its tokens
use, or the whole table, whichever codes shorter) and layers 0..M-1 at FMT, takes only token ids (no entering residual),
recomputes theta''s deployed forward through the held layers from decoded weights, and checks every counted activation
(the four product inputs per layer per token) bit for bit against theta''s own, repairing any that differ. Then
beta_x <= (l_w + l_in + l_fix + a_dec) / n_I.

  python cheap.py BASE CKPT    Env: OUT, FMT (fp8t | nvfp4), EMB (default FMT; or bf16), MMAX (held layers coded and
                               checked, 3), K (2e7), NROLL (640), BATCH (64), CHUNK (16), PROBE (8), MAXNEW (512),
                               SEED (0), STEP_TOKENS (21084), DATA (gsm8k_test.jsonl)

The deployed computation is block.py's (linear weights and inputs at FMT, the rest in fp32) with the embedding table at EMB
under counted.py's scale (theta's amax). The reference and the block run one sequence per call, so their shapes agree; the
probe runs the block on half-length prefixes of the first PROBE rollouts, where kernels see other shapes, and codes what
that would take to repair. Codes are real (block.code_many: rANS with each context's model sent), and decoded back.
Every (M, m) block, held layers 0..M-1 over the first m rollouts, is priced from one pass that records each rollout's
mismatches and repair bits per layer and input.
"""
import bisect, gzip, json, math, os, sys, time
import torch
import torch.nn.functional as Fn
import block as B
import counted
import quant
from quant import encode

DEV = counted.DEV
OUT = os.environ.get("OUT", ".")
FMT = B.FMT
EMB = os.environ.get("EMB", FMT)
C_FLOPS = 1.34e17


def main():
    base, ckpt = sys.argv[1], sys.argv[2]
    from transformers import AutoConfig, AutoTokenizer
    t0 = time.time()

    def log(s):
        print(f"{s} at {time.time() - t0:.0f}s", flush=True)
    A = B.Arch(AutoConfig.from_pretrained(base))
    tok = AutoTokenizer.from_pretrained(base)
    K, nroll = float(os.environ.get("K", "2e7")), int(os.environ.get("NROLL", "640"))
    mmax, CH, PROBE = (int(os.environ.get(k, v)) for k, v in (("MMAX", "3"), ("CHUNK", "16"), ("PROBE", "8")))
    step_tokens = int(os.environ.get("STEP_TOKENS", "21084"))
    per_layer = 3 * A.d + A.ffn
    model0 = counted.load_base(base).to(DEV)
    model1, meta = counted.load_trained(model0, ckpt)
    model1.to(DEV).eval()
    rows = [json.loads(x) for x in open(os.environ.get("DATA", "gsm8k_test.jsonl"))]
    seqs = B.rollouts(model1, tok, rows, nroll, int(os.environ.get("MAXNEW", "512")), int(os.environ.get("SEED", "0")),
                      batch=int(os.environ.get("BATCH", "64")))
    lens = [len(s) for s in seqs]
    log(f"{len(seqs)} rollouts, {sum(lens)} tokens (mean {sum(lens) / len(lens):.0f})")
    with torch.no_grad():
        nll_bits = []
        for s in seqs:
            lp = Fn.log_softmax(model0(input_ids=s[None]).logits[0, :-1].float(), -1)
            nll_bits.append(float(-lp.gather(-1, s[1:, None]).sum() / math.log(2)) + math.log2(A.vocab))
    sd0, sd1 = model0.state_dict(), model1.state_dict()
    E0, E1 = sd0["model.embed_tokens.weight"].float(), sd1["model.embed_tokens.weight"].float()
    del model0, model1
    torch.cuda.empty_cache()
    masks = B.unique_rows(seqs)

    # the embedding at EMB: theta''s deployed table, and its code given theta's (the whole table, or the rows used)
    eref = None if EMB == "bf16" else E0.abs().max()
    E1c = encode(E1, EMB, eref)
    Q1e = E1c["deq"]

    def code_emb(ids):
        X0 = E0 if ids is None else E0[ids]
        Er = {"code": E1c["code"] if ids is None else E1c["code"][ids],
              "scode": None if E1c["scode"] is None else (E1c["scode"] if ids is None else E1c["scode"][ids])}
        bits, sbits, ((c, s),) = B.code_many([(X0, Er, eref)], EMB)
        dq = B.deq_of(c, s, EMB, eref)
        return bits, bool(torch.equal(dq, Q1e if ids is None else Q1e[ids])), dq
    e_table, e_table_exact, D1e = code_emb(None)
    e_changed = float((E1c["code"] != encode(E0, EMB, eref)["code"]).float().mean())
    log(f"embedding at {EMB}: whole table {e_table:.4g} bits ({e_table / E0.numel():.5f} per weight, codes changed "
        f"{e_changed:.5f}), decoded exact {e_table_exact}")
    erows = {}

    def e_rows(m):
        if m not in erows:
            ids = torch.unique(torch.cat(seqs[:m]))
            bits, exact, _ = code_emb(ids)
            erows[m] = {"bits": bits, "rows": int(ids.numel()), "decoded_exact": exact}
        return erows[m]

    # theta' deployed, the held layers coded at FMT given theta's and decoded
    P0 = {i: B.params(sd0, i) for i in range(mmax)}
    Q1, lw, dec = {}, {}, {}
    for i in range(mmax):
        P1 = B.params(sd1, i)
        Q1[i] = dict(P1)
        for k in B.LIN:
            Q1[i][k] = encode(P1[k], FMT, P0[i][k].abs().max())["deq"]
        items = [(P0[i][k], encode(P1[k], FMT, P0[i][k].abs().max()), P0[i][k].abs().max()) for k in B.LIN]
        bits, sbits, got = B.code_many(items, FMT)
        obits, _, ogot = B.code_many([(P0[i][k], encode(P1[k], "bf16", None), None) for k in B.OTHER], "bf16")
        p, exact = {}, True
        for k, (_, _, ref), (code, scode) in zip(B.LIN, items, got):
            p[k] = B.deq_of(code, scode, FMT, ref)
            exact &= bool(torch.equal(p[k], Q1[i][k]))
        for k, (code, _) in zip(B.OTHER, ogot):
            p[k] = B.deq_of(code, None, "bf16", None)
            exact &= bool(torch.equal(p[k], P1[k]))
        nw = sum(P0[i][k].numel() for k in B.LIN)
        lw[i] = {"bits": bits + obits, "linear_bits": bits, "scale_bits": sbits, "other_bits": obits, "n_weights": nw,
                 "beta_w": bits / nw, "decoded_exact": exact}
        dec[i] = p
        del P1, items, got
        log(f"layer {i}: l_w {lw[i]['bits']:.4g} bits ({bits / nw:.4f} per weight), decoded exact {exact}")
    macs_lin = sum(P0[0][k].numel() for k in B.LIN)

    # activation references (per product input) from theta's plain fp32 forward over every rollout: sent, 32 bits each
    refs = {}

    def fq_cal(i, name, xs):
        refs[(i, name)] = max(refs.get((i, name), 0.0), max(float(x.abs().max()) for x in xs))
        return xs
    with torch.no_grad():
        for c0 in range(0, len(seqs), CH):
            sl = seqs[c0:c0 + CH]
            hs, css = [E0[s] for s in sl], [A.rope(len(s)) for s in sl]
            for i in range(mmax):
                hs = B.layer(A, P0[i], i, hs, css, fq_cal)
    refs = {k: torch.tensor(v, device=DEV) for k, v in refs.items()}
    del P0
    torch.cuda.empty_cache()
    log("references")

    # one pass: theta''s reference codes, then the block from decoded weights and token ids, repairing what differs
    rec = [{} for _ in seqs]  # rollout -> (layer, input) -> [mismatched, values, repair bits, after repair]
    probe = {}

    def fq_blk(refc, spans, sinks):
        """spans[j] = (offset, length) of sequence j's rows in refc; sinks[j](key, mismatched, values, bits, after)"""
        def fq(i, name, xs):
            X = torch.cat(xs)
            ref = refs[(i, name)]
            E = encode(X, FMT, ref)
            rc = torch.cat([refc[(i, name)][0][o:o + n] for o, n in spans]).long()
            rs = None if refc[(i, name)][1] is None else torch.cat([refc[(i, name)][1][o:o + n] for o, n in spans]).long()
            mis = B.same(E, rc, rs)
            code, scode = E["code"].clone(), None if E["scode"] is None else E["scode"].clone()
            a, any_mis, parts = 0, False, []
            for (o, n), sink in zip(spans, sinks):
                nm, bits = int(mis[a:a + n].sum()), 0.0
                if nm:
                    any_mis = True
                    Er = {"code": rc[a:a + n], "scode": None if rs is None else rs[a:a + n]}
                    bits, _, ((c2, s2),) = B.code_many([(X[a:a + n], Er, ref)], FMT)
                    code[a:a + n] = c2
                    if scode is not None:
                        scode[a:a + n] = s2
                parts.append((sink, nm, n * X.shape[1], bits, a, n))
                a += n
            after = B.same({"code": code, "scode": scode}, rc, rs) if any_mis else None
            for sink, nm, nv, bits, a, n in parts:
                sink((i, name), nm, nv, bits, 0 if after is None else int(after[a:a + n].sum()))
            deq = B.deq_of(code, scode, FMT, ref) if any_mis else E["deq"]
            return B.split(deq, [n for _, n in spans])
        return fq

    def into(d):
        def sink(key, nm, nv, bits, after):
            v = d.setdefault(key, [0, 0, 0.0, 0])
            v[0] += nm
            v[1] += nv
            v[2] += bits
            v[3] += after
        return sink

    with torch.no_grad():
        for c0 in range(0, len(seqs), CH):
            idx = list(range(c0, min(c0 + CH, len(seqs))))
            Ls = [lens[r] for r in idx]
            css = [A.rope(L) for L in Ls]
            refc = {}

            def fq_ref(i, name, xs):
                E = encode(torch.cat(xs), FMT, refs[(i, name)])
                refc[(i, name)] = (E["code"].to(torch.int16), None if E["scode"] is None else E["scode"].to(torch.int16))
                return B.split(E["deq"], Ls)
            hs = [Q1e[seqs[r]] for r in idx]
            for i in range(mmax):
                hs = B.layer(A, Q1[i], i, hs, css, fq_ref)
            offs = [sum(Ls[:j]) for j in range(len(Ls))]
            fq = fq_blk(refc, list(zip(offs, Ls)), [into(rec[r]) for r in idx])
            hs = [D1e[seqs[r]] for r in idx]
            for i in range(mmax):
                hs = B.layer(A, dec[i], i, hs, css, fq)
            for j in range(len(idx)):
                if idx[j] >= PROBE:
                    break
                t = Ls[j] // 2
                fqp = fq_blk(refc, [(offs[j], t)], [into(probe)])
                hs = [D1e[seqs[idx[j]][:t]]]
                for i in range(mmax):
                    hs = B.layer(A, dec[i], i, hs, [(css[j][0][:t], css[j][1][:t])], fqp)
            del refc
    log("reference and block pass")

    # pricing every (M, m) block from the per-rollout records
    a_dec = 8 * len(gzip.compress(b"".join(open(f, "rb").read() for f in (__file__, B.__file__, quant.__file__)), 9))
    names = B.INPUTS
    T_cum, U_cum = [0], [0]
    for r in range(len(seqs)):
        T_cum.append(T_cum[-1] + lens[r])
        U_cum.append(U_cum[-1] + int(masks[r].sum()))
    fix = [[sum(rec[r][(i, n)][2] for n in names) for i in range(mmax)] for r in range(len(seqs))]
    mis = [[sum(rec[r][(i, n)][0] for n in names) for i in range(mmax)] for r in range(len(seqs))]
    aft = [[sum(rec[r][(i, n)][3] for n in names) for i in range(mmax)] for r in range(len(seqs))]
    log2v = math.log2(A.vocab)

    def price(M, m):
        er = e_rows(m)
        emb, desc = (er["bits"], "rows used") if er["bits"] <= e_table else (e_table, "whole table")
        lws = sum(lw[i]["bits"] for i in range(M))
        l_tok = log2v * T_cum[m]
        l_scales = 32 * (4 * M + 1)
        l_fix = 4 * M + sum(fix[r][i] for r in range(m) for i in range(M))  # one flag bit per input, then the repairs
        total = emb + lws + l_tok + l_scales + l_fix + a_dec
        n_I = U_cum[m] * M * per_layer
        attn1 = sum(A.nh * A.hd * L * (L + 1) for L in lens[:m])
        syms = ((er["rows"] * A.d if desc == "rows used" else E0.numel()) + sum(lw[i]["n_weights"] for i in range(M))
                + T_cum[m] * M * per_layer)
        flops = 2 * (macs_lin * T_cum[m] + attn1) * M + 60 * syms
        l_tok_theta = sum(nll_bits[:m])
        return {"M": M, "rollouts": m, "tokens": T_cum[m], "unique_rows": U_cum[m], "emb_rows": er["rows"],
                "l_emb_rows": er["bits"], "l_emb_table": e_table, "emb_desc": desc, "l_emb": emb,
                "emb_rows_decoded_exact": er["decoded_exact"], "l_w_layers": lws, "l_w": emb + lws, "l_tok": l_tok,
                "l_tok_theta_nll": l_tok_theta, "l_scales": l_scales, "l_fix": l_fix, "a_dec": a_dec, "total": total,
                "fits_k": total <= K, "n_I": n_I, "beta_x_bound": total / n_I,
                "beta_x_bound_theta_tokens": (total - l_tok + l_tok_theta) / n_I,
                "token_floor": l_tok / n_I, "mismatch_before_repair": sum(mis[r][i] for r in range(m) for i in range(M)),
                "mismatch_after_repair": sum(aft[r][i] for r in range(m) for i in range(M)),
                "decode_flops": flops, "decode_flops_over_c": flops / C_FLOPS,
                "decode_flops_theta_tokens_over_c": (flops + 2 * 1.54e9 * T_cum[m]) / C_FLOPS,
                "share_of_counted_per_token": M / A.nl}

    m_step = min(bisect.bisect_left(T_cum, step_tokens), len(seqs))
    blocks = []
    for M in range(1, mmax + 1):
        blocks.append(dict(price(M, m_step), fill="step"))
        lo, hi = 0, len(seqs)  # the largest m whose block fits k
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if price(M, mid)["total"] <= K:
                lo = mid
            else:
                hi = mid - 1
        if lo:
            blocks.append(dict(price(M, lo), fill="k"))
        log(f"M={M}: " + "; ".join(f"{b['fill']} Q={b['tokens']} ({b['rollouts']} rollouts) l_emb={b['l_emb']:.4g} "
                                     f"({b['emb_desc']}, {b['emb_rows']} rows) l_w={b['l_w']:.4g} l_fix={b['l_fix']:.4g} "
                                     f"total={b['total']:.4g} n_I={b['n_I']:.4g} bound {b['beta_x_bound']:.3g} "
                                     f"(theta tokens {b['beta_x_bound_theta_tokens']:.3g}) mismatches "
                                     f"{b['mismatch_before_repair']}/{b['mismatch_after_repair']}"
                                     for b in blocks if b["M"] == M))
    pr = {f"{i}:{n}": {"mismatched": v[0], "values": v[1], "repair_bits": v[2], "after_repair": v[3]}
          for (i, n), v in sorted(probe.items())}
    res = {"ckpt": ckpt, "meta": meta, "fmt": FMT, "emb_fmt": EMB, "k": K, "n_rollouts": len(seqs), "lens": lens,
           "tokens": sum(lens), "nll_token_bits": nll_bits, "a_dec_bits": a_dec, "emb_table_bits": e_table,
           "emb_table_decoded_exact": e_table_exact, "emb_codes_changed": e_changed, "emb_rows": erows,
           "layer_weight_bits": lw, "per_rollout": {"fix": fix, "mismatch": mis, "after": aft}, "probe": pr,
           "probe_rollouts": min(PROBE, len(seqs)), "blocks": blocks, "n_layers": A.nl, "per_layer_counted": per_layer,
           "secs": time.time() - t0}
    json.dump(res, open(os.path.join(OUT, f"cheap_{FMT}_emb-{EMB}.json"), "w"), indent=1, default=float)
    tp = sum(v[1] for v in probe.values()) / max(1, mmax * per_layer)
    log(f"probe (half-length prefixes of {res['probe_rollouts']} rollouts, {tp:.0f} rows): mismatched "
        f"{sum(v[0] for v in probe.values())}, repair bits {sum(v[2] for v in probe.values()):.4g}, after repair "
        f"{sum(v[3] for v in probe.values())}")


if __name__ == "__main__":
    main()
