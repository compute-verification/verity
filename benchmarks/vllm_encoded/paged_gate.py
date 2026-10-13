"""`paged.cu`'s gate on one GPU, before anything uses it in vLLM: synthetic layer caches in vLLM's layout
([blocks][8 KV heads][16 slots][K ‖ V] bf16), two layers of a pool, random block tables and lengths (0, 1, a tile's
edge, a block's edge and random), and for each k of `--ks`:

- `append`: a step's tokens (padding slots −1 among them) appended to the plain cache equal torch's indexing, and to
  the encoded cache, which then equals `kv_encode` of the plain cache at the layer's word indices (and `kv.py` on a few
  words);
- `attention`: at each split count, the encoded output equals the plain output bit for bit, the plain output is within
  RTOL of float64 attention over the same bf16 values (q scaled by log2(e)/sqrt(128), rounded to bf16, as the kernel
  does), a sequence of length 0 and every row past the batch are zero, and the qrow form (rows permuted) equals it;
  with each sequence's last token's K and V given as rows (ck, cv) and its slot in the pages overwritten, the output is
  the same bits (plain, encoded, and in the qrow form);
- `append_multi`: both layers' appends in one launch (more rows than tokens) equal the layer-by-layer appends;
- `gather`: whole blocks decoded (encoded) or copied (plain) equal the plain cache's blocks.

Writes `paged-gate.json` and `paged-gate.log` under `--out`; exits nonzero if any gate fails."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paged as P  # noqa: E402
from paged import K, KB  # noqa: E402

RTOL = 8e-3  # step.py's for the tensor-core form, which rounds P (and here the output) to bf16
BLOCKS, BS, LAYERS = 96, 16, 2
LENS = (0, 1, 15, 16, 17, 63, 64, 65, 300, 511)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="NAME=PATH, paged.cu built at field NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--batches", default="1,5,10,32")
    ap.add_argument("--splits", default="1,3,8,32")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "paged-gate.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    lib = P.PagedLib(*a.lib.split("=", 1))
    sh = lib.shape
    H, G, D, WB = sh["heads"], sh["kv_heads"], sh["head_dim"], sh["wb"]
    log(f"{lib.name}: shape {sh}, mask {lib.mask}, fold6 {lib.fold6}")
    torch.manual_seed(0)
    rng = random.Random(1)
    plain = [(torch.randn(BLOCKS, G, BS, 2 * D, device="cuda") * 2).to(torch.bfloat16) for _ in range(LAYERS)]
    lay = P.layout(plain[0], G, WB)
    lbase = [l * lay.words for l in range(LAYERS)]
    res = {"lib": lib.name, "shape": sh, "mask": lib.mask, "fold6": lib.fold6, "layout": lay.__dict__, "ks": {},
           "ok": True}
    st = P.stream(torch)
    for k in (int(v) for v in a.ks.split(",")):
        KB.set_key(lib, KB.lib_key(lib, k))
        key = KB.lib_key(lib, k)
        pl = [c.clone() for c in plain]
        enc = []
        for l in range(LAYERS):
            e = torch.empty_like(pl[l])
            lib.call("kv_encode", pl[l].data_ptr(), e.data_ptr(), lay.words, lbase[l], k, 0, st)
            enc.append(e)
        g = {}
        # append: a step of tokens into random slots, padding among them
        n = 40
        slots = torch.tensor(rng.sample(range(BLOCKS * BS), n), dtype=torch.int64)
        slots[::7] = -1
        qkv = (torch.randn(n, (H + 2 * G) * D, device="cuda") * 2).to(torch.bfloat16)
        kk, vv = qkv[:, H * D:(H + G) * D], qkv[:, (H + G) * D:]
        sl = slots.cuda()
        ok_append, where = True, []
        pre = [c.clone() for c in pl] + [c.clone() for c in enc]
        for l in range(LAYERS):
            P.append(lib, torch, pl[l], lay, lbase[l], 0, k, kk, vv, sl)
            P.append(lib, torch, enc[l], lay, lbase[l], 1, k, kk, vv, sl)
            ref = pl[l].clone()
            for i, s in enumerate(slots.tolist()):
                if s < 0:
                    continue
                ref[s // BS, :, s % BS, :D] = kk[i].view(G, D)
                ref[s // BS, :, s % BS, D:] = vv[i].view(G, D)
            re = torch.empty_like(pl[l])
            lib.call("kv_encode", pl[l].data_ptr(), re.data_ptr(), lay.words, lbase[l], k, 0, st)
            torch.cuda.synchronize()
            # bits, not bf16 values: an encoded word's halves include NaNs
            pe = bool(torch.equal(ref.view(torch.int16), pl[l].view(torch.int16)))
            ee = bool(torch.equal(re.view(torch.int16), enc[l].view(torch.int16)))
            ok_append &= pe and ee
            if not (pe and ee):
                dp = (ref.view(torch.int16) != pl[l].view(torch.int16)).view(-1, WB // 2).any(1).nonzero().flatten()
                de = (re.view(torch.int16) != enc[l].view(torch.int16)).view(-1, WB // 2).any(1).nonzero().flatten()
                where.append({"layer": l, "plain": pe, "encoded": ee, "plain_words": dp[:8].tolist(),
                              "plain_differ": dp.numel(), "encoded_words": de[:8].tolist(),
                              "encoded_differ": de.numel(),
                              "appended_words": sorted((s // BS) * lay.sb + g * lay.sh + (s % BS) * lay.ss + j
                                                       for s in slots.tolist() if s >= 0
                                                       for g in range(G) for j in range(4))[:8]})
        # kv.py on a few words of layer 1
        w8 = pl[1].view(torch.uint8).view(-1, WB)
        e8 = enc[1].view(torch.uint8).view(-1, WB)
        picks = [0, 1, lay.words - 1] + rng.sample(range(lay.words), 5)
        py_ok = all(K.encode_word(int.from_bytes(bytes(w8[i].cpu().tolist()), "little"), lbase[1] + i, key)
                    == int.from_bytes(bytes(e8[i].cpu().tolist()), "little") for i in picks)
        tabs = {}
        for encd in (0, 1):
            ents = [(kk, vv, sl, n, pre[encd * LAYERS + l], lbase[l]) for l in range(LAYERS)]
            tabs[encd] = P.table(torch, ents).cuda()
            P.append_multi(lib, torch, tabs[encd], LAYERS, n + 8, lay, encd, k)
        torch.cuda.synchronize()
        multi_ok = all(torch.equal(pre[encd * LAYERS + l].view(torch.int16), (enc if encd else pl)[l].view(torch.int16))
                       for encd in (0, 1) for l in range(LAYERS))
        g["append_multi"] = {"layers": LAYERS, "rows": n + 8, "tokens": n, "ok": multi_ok}
        if not multi_ok:
            log(f"append_multi k={k}: FAIL")
        g["append"] = {"tokens": n, "padding": int((slots < 0).sum()), "equal": ok_append, "kv_py_words": len(picks),
                       "kv_py": py_ok, "differ": where, "ok": ok_append and py_ok}
        if where:
            log(f"append k={k}: FAIL {where}")
        # attention
        att = {"cells": [], "ok": True}
        for b in (int(v) for v in a.batches.split(",")):
            lens_l = [LENS[i % len(LENS)] if i < len(LENS) else rng.randrange(1, 513) for i in range(b)]
            rng.shuffle(lens_l)
            maxb = 32
            perm = rng.sample(range(BLOCKS), BLOCKS)
            bt = torch.zeros(b, maxb + 3, dtype=torch.int32)
            for i in range(b):
                nb = -(-lens_l[i] // BS)
                bt[i, :nb] = torch.tensor([perm[(i * 3 + j) % BLOCKS] for j in range(nb)], dtype=torch.int32)
            btd, lens = bt.cuda(), torch.tensor(lens_l, dtype=torch.int32, device="cuda")
            nrows = b + 3
            qbuf = (torch.randn(nrows, (H + 2 * G) * D, device="cuda")).to(torch.bfloat16)
            q = qbuf[:, :H * D]
            layer = rng.randrange(LAYERS)
            kd = pl[layer][:, :, :, :D].double()  # block, head, slot, dim
            vd = pl[layer][:, :, :, D:].double()
            qs = (q.float() * P.QSCALE).to(torch.bfloat16).double().view(nrows, G, H // G, D)
            ref = torch.zeros(nrows, H, D, dtype=torch.float64, device="cuda")
            for i in range(b):
                L = lens_l[i]
                if not L:
                    continue
                blk = bt[i, :-(-L // BS)].long().cuda()
                kt = kd[blk].permute(0, 2, 1, 3).reshape(-1, G, D)[:L]  # token, head, dim
                vt = vd[blk].permute(0, 2, 1, 3).reshape(-1, G, D)[:L]
                s = torch.einsum("gqd,tgd->gqt", qs[i], kt)
                p = torch.exp2(s - s.amax(-1, keepdim=True))
                ref[i] = torch.einsum("gqt,tgd->gqd", p / p.sum(-1, keepdim=True), vt).reshape(H, D)
            for sp in (int(v) for v in a.splits.split(",")):
                part = torch.empty(P.part_floats(lib, b, sp), dtype=torch.float32, device="cuda")
                o_p = torch.full((nrows, H * D), 7.0, dtype=torch.bfloat16, device="cuda")
                o_e = torch.full_like(o_p, -3.0)
                P.attn(lib, torch, pl[layer], lay, lbase[layer], 0, k, q, o_p, part, lens, btd, b, sp)
                P.attn(lib, torch, enc[layer], lay, lbase[layer], 1, k, q, o_e, part, lens, btd, b, sp)
                qrow = torch.tensor(rng.sample(range(b), b), dtype=torch.int32, device="cuda")
                qp = torch.empty(b, H * D, dtype=torch.bfloat16, device="cuda")
                qp[qrow.long()] = q[:b]
                o_r = torch.full((b, H * D), 5.0, dtype=torch.bfloat16, device="cuda")
                P.attn(lib, torch, enc[layer], lay, lbase[layer], 1, k, qp, o_r, part, lens, btd, b, sp, qrow=qrow)
                # the last tokens as rows, their slots in the pages overwritten
                ck = torch.zeros(nrows, G * D, dtype=torch.bfloat16, device="cuda")
                cv = torch.zeros_like(ck)
                pl_x, enc_x = pl[layer].clone(), enc[layer].clone()
                for i in range(b):
                    L = lens_l[i]
                    if not L:
                        continue
                    blk, off = int(bt[i, (L - 1) // BS]), (L - 1) % BS
                    ck[i] = pl[layer][blk, :, off, :D].reshape(-1)
                    cv[i] = pl[layer][blk, :, off, D:].reshape(-1)
                    junk = (torch.randn(G, 2 * D, device="cuda") * 9).to(torch.bfloat16)
                    pl_x[blk, :, off] = junk
                    enc_x[blk, :, off] = junk
                o_cp = torch.full_like(o_p, 1.0)
                o_ce = torch.full_like(o_p, 2.0)
                P.attn(lib, torch, pl_x, lay, lbase[layer], 0, k, q, o_cp, part, lens, btd, b, sp, ck=ck, cv=cv)
                P.attn(lib, torch, enc_x, lay, lbase[layer], 1, k, q, o_ce, part, lens, btd, b, sp, ck=ck, cv=cv)
                ckp, cvp = torch.empty(b, G * D, dtype=torch.bfloat16, device="cuda"), torch.empty(
                    b, G * D, dtype=torch.bfloat16, device="cuda")
                ckp[qrow.long()], cvp[qrow.long()] = ck[:b], cv[:b]
                o_cr = torch.full((b, H * D), 4.0, dtype=torch.bfloat16, device="cuda")
                P.attn(lib, torch, enc_x, lay, lbase[layer], 1, k, qp, o_cr, part, lens, btd, b, sp, qrow=qrow, ck=ckp,
                       cv=cvp)
                torch.cuda.synchronize()
                rows_eq = (bool(torch.equal(o_cp.view(torch.int16), o_p.view(torch.int16)))
                           and bool(torch.equal(o_ce.view(torch.int16), o_e.view(torch.int16)))
                           and bool(torch.equal(o_cr[qrow.long()].view(torch.int16), o_e[:b].view(torch.int16))))
                equal = bool(torch.equal(o_p.view(torch.int16), o_e.view(torch.int16)))
                zero_pad = bool((o_e[b:] == 0).all())
                zero_len = all(bool((o_e[i] == 0).all()) for i in range(b) if not lens_l[i])
                qrow_eq = bool(torch.equal(o_r[qrow.long()].view(torch.int16), o_e[:b].view(torch.int16)))
                m = ref.abs().max().item() or 1.0
                err = float((o_p[:b].double().view(b, H, D) - ref[:b]).abs().max() / m)
                ok = equal and zero_pad and zero_len and qrow_eq and rows_eq and err <= RTOL
                att["cells"].append({"batch": b, "splits": sp, "layer": layer, "lens": lens_l, "equal": equal,
                                     "zero_padding": zero_pad, "zero_length": zero_len, "qrow": qrow_eq, "last_rows": rows_eq,
                                     "rel_err": err, "ok": ok})
                att["ok"] &= ok
                if not ok:
                    log(f"attention k={k} B={b} splits={sp}: FAIL {att['cells'][-1]}")
        g["attention"] = att
        # gather
        blocks = torch.tensor(rng.sample(range(BLOCKS), 20), dtype=torch.int32, device="cuda")
        ok_g = True
        for l in range(LAYERS):
            for encd, src in ((0, pl[l]), (1, enc[l])):
                o = torch.empty(20, *pl[l].shape[1:], dtype=torch.bfloat16, device="cuda")
                P.gather(lib, torch, src, lay, lbase[l], encd, k, blocks, o)
                torch.cuda.synchronize()
                ok_g &= bool(torch.equal(o.view(torch.int16), pl[l][blocks.long()].view(torch.int16)))
        g["gather"] = {"blocks": 20, "ok": ok_g}
        g["ok"] = g["append"]["ok"] and multi_ok and att["ok"] and ok_g
        res["ks"][k] = g
        res["ok"] &= g["ok"]
        errs = [c["rel_err"] for c in att["cells"]]
        log(f"k={k}: append {'ok' if g['append']['ok'] else 'FAIL'}, append_multi {'ok' if multi_ok else 'FAIL'}, attention {'ok' if att['ok'] else 'FAIL'} "
            f"({len(att['cells'])} cells, max rel err {max(errs):.1e}), gather {'ok' if ok_g else 'FAIL'}")
    (out_dir / "paged-gate.json").write_text(json.dumps(res, indent=1))
    log("paged gate " + ("passed" if res["ok"] else "FAILED"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
