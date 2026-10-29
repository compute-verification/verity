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

Then `bound`, the bound key (`bound_gate`): token-layers written after the codec seed, each under its slot's offset,
the pinned ones in both forms (without the append coin and with it), and deferred residency's plaintext flag
(`deferred_gate`): the pinned coin blocks appended plaintext and encoded in place once the coin is known, and decode
attention and the gather over a cache whose slots mix plaintext and encoded ones.

Writes `paged-gate.json` and `paged-gate.log` under `--out`; exits nonzero if any gate fails."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "codec_tensor"))

import paged as P  # noqa: E402
from paged import K, KB  # noqa: E402
import sketch_receipt as SR  # noqa: E402
import sketching as SKT  # noqa: E402

sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "verity" / "core" / "protocols" / "remote_bandwidth" / "tests")]
import remote_bandwidth_vectors as RV  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as RC  # noqa: E402

RTOL = 8e-3  # step.py's for the tensor-core form, which rounds P (and here the output) to bf16
BLOCKS, BS, LAYERS = 96, 16, 2
LENS = (0, 1, 15, 16, 17, 63, 64, 65, 300, 511)
BOUND_LENS = (1, 17, 64, 300, 511)
BOUND_SPLITS = (1, 4)


def bound_gate(lib: P.PagedLib, torch, ks: list[int], log, main_rc) -> dict:
    """The bound key on every path, in three parts.

    - `pinned` (a library at the remote audit's codec, `codec.DEFAULT`): `vectors.json`'s `codec.bound` blocks
      (the remote audit's payloads j = 0 and 20 under their bound keys) and its `codec.coin` blocks (the same payloads
      appended under the append coin's form b_j(c_a), PROTOCOL.md §3), each appended as one token's K and V at the
      slot and layer whose store-order block is j (store base 0) under b_j, and at another slot with the index
      difference folded into its offset (b_j + tau W (j − there)): the token-layer's words equal the pinned block (its
      SHA-512 and first word) both times, so a page encodes at its store index whatever slot holds it. The `codec.coin`
      blocks again in the deferred form (`deferred`): appended plaintext with the slot's flag set, then encoded in
      place by `pg_encode`, which clears the flag, and a second `pg_encode` leaves the words as they are.
    - `random` (any library): every slot of both layers under a random offset of its own. The appended cache equals
      `kv.py` on a few words. `pg_append_multi` with the offsets in its table equals it. Decode attention over it
      (BOUND_LENS, at each of BOUND_SPLITS) equals the plain cache's output bit for bit, and decoded under the seed's
      key it does not. The gather, with receipts and without, equals the plain blocks.
    - `deferred`: plaintext and encoded slots mixed in one cache (`deferred_gate`), main_rc being main's receipt
      buffers and its `receipt_gate`."""
    G, D, WB, f = lib.shape["kv_heads"], lib.shape["head_dim"], lib.shape["wb"], lib.field
    W, nslots = K.token_layer_words(f), BLOCKS * BS
    out = {"ok": True}
    st = P.stream(torch)
    rc = RC.DEFAULT
    pinned = json.loads(RV.PATH.read_text())["codec"]
    if (f.name, lib.mask, pinned["name"]) == (rc.field.name, rc.masked, rc.name) and rc.rounds <= lib.kmax:
        seed = bytes.fromhex(pinned["seed"])
        key, dev = RC.key(seed, rc), K.key(f.name, seed, rc.rounds, mask=rc.masked)
        if (dev.c, dev.tau, dev.mask) != (key.c, key.tau, key.mask):
            raise RuntimeError("kv.py's key is not the remote audit's codec key")
        KB.set_key(lib, dev)
        cache = torch.zeros(BLOCKS, G, BS, 2 * D, dtype=torch.bfloat16, device="cuda")
        lay = P.layout(cache, G, WB)

        def token_layer(slot):
            bo, so = divmod(slot, BS)
            return torch.cat((cache[bo, :, so, :D].reshape(-1), cache[bo, :, so, D:].reshape(-1))).view(
                torch.uint8).cpu().numpy().tobytes()

        pins = []
        c_a = bytes.fromhex(pinned["coin"]["c_a"])
        for form, vectors, coin in (("bound", pinned["bound"], None), ("coin", pinned["coin"]["bound"], c_a),
                                    ("deferred", pinned["coin"]["bound"], c_a)):
            for v in vectors:
                j = v["j"]
                pay = RV.payload(j)
                d = RC.digest(pay)
                b = RC.binding(d, j, key, coin)
                kr = torch.frombuffer(bytearray(pay[:2048]), dtype=torch.bfloat16).view(1, G * D).cuda()
                vr = torch.frombuffer(bytearray(pay[2048:]), dtype=torch.bfloat16).view(1, G * D).cuda()
                there = (nslots - 1 - j // LAYERS) * LAYERS + j % LAYERS
                for slot, layer, off in ((j // LAYERS, j % LAYERS, b),
                                         (there // LAYERS, there % LAYERS, (b + key.tau * W * (j - there)) % f.p)):
                    bj = P.offsets(torch, [off if s == slot else 0 for s in range(nslots)], f)
                    at = torch.tensor([slot], dtype=torch.int64, device="cuda")
                    extra = {}
                    if form == "deferred":
                        # appended plaintext, flagged, until the coin; then encoded in place, twice (the second a no-op)
                        flags = torch.zeros(nslots, dtype=torch.uint8, device="cuda")
                        P.append(lib, torch, cache, lay, 0, LAYERS, layer, 0, rc.rounds, kr, vr, at, plain=flags)
                        torch.cuda.synchronize()
                        extra["plaintext"] = token_layer(slot) == pay and flags.nonzero().flatten().tolist() == [slot]
                        tab = P.encode_table(torch, [(at, 1, cache, 0, LAYERS, layer, bj, flags)]).cuda()
                        P.encode(lib, torch, tab, 1, 1, lay, rc.rounds)
                        torch.cuda.synchronize()
                        once = token_layer(slot)
                        extra["settled"] = not bool(flags.any())
                        P.encode(lib, torch, tab, 1, 1, lay, rc.rounds)
                        torch.cuda.synchronize()
                        extra["idempotent"] = token_layer(slot) == once
                    else:
                        P.append(lib, torch, cache, lay, 0, LAYERS, layer, 1, rc.rounds, kr, vr, at, bj=bj)
                        torch.cuda.synchronize()
                    got = token_layer(slot)
                    pins.append({"form": form, "j": j, "slot": slot, "layer": layer,
                                 "folded": slot * LAYERS + layer != j, "digest": d.hex() == v["digest"],
                                 "b": format(b, "x") == v["b"], "word_0": got[:WB].hex() == v["word_0"],
                                 "block_sha512": hashlib.sha512(got).hexdigest() == v["block_sha512"], **extra})
        forms = {form: all(p["digest"] and p["b"] and p["word_0"] and p["block_sha512"]
                           and all(p.get(x, True) for x in ("plaintext", "settled", "idempotent"))
                           for p in pins if p["form"] == form) for form in ("bound", "coin", "deferred")}
        pins_ok = all(forms.values())
        out["pinned"] = {"codec": rc.name, "forms": forms, "blocks": pins, "ok": pins_ok}
        out["ok"] &= pins_ok
        log(f"bound pinned {rc.name}: {'ok' if pins_ok else 'FAIL ' + str(pins)} ({len(pins)} appends against "
            "vectors.json's codec.bound and codec.coin, at the block's own slot and folded at another; codec.coin "
            "also appended plaintext and encoded in place)")
        k, dev_key = rc.rounds, dev
    else:
        out["pinned"] = {"skipped": f"the library is {f.name}{'/masked' if lib.mask else ''}, not {rc.name}"}
        log(f"bound pinned: skipped, {out['pinned']['skipped']}")
        k = ks[0]
        dev_key = KB.lib_key(lib, k)
        KB.set_key(lib, dev_key)
    rng = random.Random(7)
    offs = [[rng.randrange(f.p) for _ in range(nslots)] for _ in range(LAYERS)]
    bjs = [P.offsets(torch, o, f) for o in offs]
    plain = [(torch.randn(BLOCKS, G, BS, 2 * D, device="cuda") * 2).to(torch.bfloat16) for _ in range(LAYERS)]
    lay = P.layout(plain[0], G, WB)
    all_slots = torch.arange(nslots, dtype=torch.int64, device="cuda")
    rows = [(p[:, :, :, :D].permute(0, 2, 1, 3).reshape(nslots, G * D).contiguous(),
             p[:, :, :, D:].permute(0, 2, 1, 3).reshape(nslots, G * D).contiguous()) for p in plain]
    enc = [torch.empty_like(p) for p in plain]
    for l in range(LAYERS):
        P.append(lib, torch, enc[l], lay, 0, LAYERS, l, 1, k, *rows[l], all_slots, bj=bjs[l])
    multi = [torch.empty_like(p) for p in plain]
    tab = P.table(torch, [(*rows[l], all_slots, nslots, multi[l], 0, LAYERS, l, bjs[l]) for l in range(LAYERS)]).cuda()
    P.append_multi(lib, torch, tab, LAYERS, nslots, lay, 1, k)
    torch.cuda.synchronize()
    multi_ok = all(torch.equal(multi[l].view(torch.int16), enc[l].view(torch.int16)) for l in range(LAYERS))
    w8, e8 = plain[1].view(torch.uint8).view(-1, WB), enc[1].view(torch.uint8).view(-1, WB)
    hw = lib.shape["hw"]
    py_ok = True
    for i in [0, 1, lay.words - 1] + rng.sample(range(lay.words), 5):
        b, r = divmod(i, lay.sb)
        g, r = divmod(r, lay.sh)
        s, j = divmod(r, lay.ss)
        slot = b * BS + s
        wi = (j if j < hw // 2 else (2048 // WB) + j - hw // 2) + g * (hw // 2)
        m = int.from_bytes(bytes(w8[i].cpu().tolist()), "little")
        py_ok &= K.encode_word(m, (slot * LAYERS + 1) * W + wi, dev_key, offs[1][slot]) == int.from_bytes(
            bytes(e8[i].cpu().tolist()), "little")
    batch, H = len(BOUND_LENS), lib.shape["heads"]
    bt = torch.zeros(batch, 32, dtype=torch.int32)
    perm = rng.sample(range(BLOCKS), BLOCKS)
    for i, n in enumerate(BOUND_LENS):
        nb = -(-n // BS)
        bt[i, :nb] = torch.tensor(perm[2 * i:2 * i + nb] if 2 * i + nb <= BLOCKS else perm[:nb], dtype=torch.int32)
    btd, lens = bt.cuda(), torch.tensor(BOUND_LENS, dtype=torch.int32, device="cuda")
    q = torch.randn(batch, H * D, device="cuda").to(torch.bfloat16)
    att = []
    for layer in range(LAYERS):
        for sp in BOUND_SPLITS:
            part = torch.empty(P.part_floats(lib, batch, sp), dtype=torch.float32, device="cuda")
            o_p, o_b, o_s = (torch.full((batch, H * D), c, dtype=torch.bfloat16, device="cuda") for c in (7.0, -3.0, 5.0))
            P.attn(lib, torch, plain[layer], lay, 0, LAYERS, layer, 0, k, q, o_p, part, lens, btd, batch, sp)
            P.attn(lib, torch, enc[layer], lay, 0, LAYERS, layer, 1, k, q, o_b, part, lens, btd, batch, sp, bj=bjs[layer])
            P.attn(lib, torch, enc[layer], lay, 0, LAYERS, layer, 1, k, q, o_s, part, lens, btd, batch, sp)
            torch.cuda.synchronize()
            att.append({"layer": layer, "splits": sp, "equal": bool(torch.equal(o_p.view(torch.int16), o_b.view(torch.int16))),
                        "seed_key_differs": not torch.equal(o_p.view(torch.int16), o_s.view(torch.int16))})
    att_ok = all(c["equal"] and c["seed_key_differs"] for c in att)
    blocks = torch.tensor(rng.sample(range(BLOCKS), 20), dtype=torch.int32, device="cuda")
    rc_sk = torch.empty((nslots, G, SR.ROWS), dtype=torch.int32, device="cuda")
    used = torch.full((nslots,), -1, dtype=torch.int32, device="cuda")
    epoch = torch.ones(1, dtype=torch.int32, device="cuda")
    gather_ok = True
    for l in range(LAYERS):
        for receipts in (None, (rc_sk, used, epoch)):
            o = torch.empty(20, *plain[l].shape[1:], dtype=torch.bfloat16, device="cuda")
            P.gather(lib, torch, enc[l], lay, 0, LAYERS, l, 1, k, blocks, o, receipts=receipts, bj=bjs[l])
            torch.cuda.synchronize()
            gather_ok &= bool(torch.equal(o.view(torch.int16), plain[l][blocks.long()].view(torch.int16)))
    rand_ok = multi_ok and py_ok and att_ok and gather_ok
    out["random"] = {"k": k, "slots": nslots, "layers": LAYERS, "append_multi": multi_ok, "kv_py": py_ok,
                     "attention": att, "gather": gather_ok, "ok": rand_ok}
    out["ok"] &= rand_ok
    log(f"bound random k={k}: {'ok' if rand_ok else 'FAIL'} (append_multi {multi_ok}, kv.py {py_ok}, attention "
        f"{sum(c['equal'] and c['seed_key_differs'] for c in att)}/{len(att)}, gather {gather_ok})")
    out["deferred"] = deferred_gate(lib, torch, k, plain, enc, bjs, rows, lay, bt, btd, lens, q, rng, main_rc, log)
    out["ok"] &= out["deferred"]["ok"]
    return out


def deferred_gate(lib: P.PagedLib, torch, k: int, plain, enc, bjs, rows, lay: P.Layout, bt, btd, lens, q, rng,
                  main_rc, log) -> dict:
    """The plaintext flag (deferred residency, remote_bandwidth's PROTOCOL.md §4) on `bound_gate`'s random caches: each
    slot of both layers appended as it is (its flag set) or encoded under its offset (flag clear), at random, so every
    decode's tiles mix both.

    - `cache`: the mixed cache holds the plain cache's words at the flagged slots and the all-encoded cache's at the
      rest, and the flags are what the appends wrote; `pg_append_multi` (both forms, the flags in its table) writes the
      same cache and flags.
    - `attention` (BOUND_LENS, at each of BOUND_SPLITS): with the flags, the output equals the plain cache's bit for
      bit, and with receipts too, which mark exactly the encoded slots it reads and equal the reference's
      (`receipt_gate`); without the flags (every slot decoded) it does not.
    - `gather`: with receipts and without, equals the plain blocks; its receipts are the encoded slots'.
    - `encode`: `pg_encode` of the flagged slots, both layers in one launch, leaves each cache equal to the all-encoded
      one and every flag clear; attention after it equals the plain output; a second encode, over every slot, changes
      nothing (each flag is clear).
    - flags on a plain cache are refused."""
    D, H = lib.shape["head_dim"], lib.shape["heads"]
    nslots, batch = BLOCKS * BS, len(BOUND_LENS)
    rc_sk, used, epoch, receipt_gate = main_rc

    def eq(a, b) -> bool:
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))

    is_plain = [[rng.random() < 0.5 for _ in range(nslots)] for _ in range(LAYERS)]
    want = [torch.tensor(p, dtype=torch.uint8, device="cuda") for p in is_plain]
    split = [tuple(torch.tensor([s for s in range(nslots) if p[s] == side], dtype=torch.int64, device="cuda")
                   for side in (True, False)) for p in is_plain]
    mix = [torch.empty_like(p) for p in plain]
    multi = [torch.empty_like(p) for p in plain]
    flags = [torch.full((nslots,), 7, dtype=torch.uint8, device="cuda") for _ in range(LAYERS)]
    mflags = [f.clone() for f in flags]
    kept = []
    for encd in (0, 1):
        ents = []
        for l in range(LAYERS):
            sl = split[l][encd]
            kr, vr = rows[l][0][sl], rows[l][1][sl]
            kept += [kr, vr]
            P.append(lib, torch, mix[l], lay, 0, LAYERS, l, encd, k, kr, vr, sl, bj=bjs[l] if encd else None,
                     plain=flags[l])
            ents.append((kr, vr, sl, sl.shape[0], multi[l], 0, LAYERS, l, bjs[l] if encd else None, mflags[l]))
        tab = P.table(torch, ents).cuda()
        kept.append(tab)
        P.append_multi(lib, torch, tab, LAYERS, max(e[3] for e in ents), lay, encd, k)
    torch.cuda.synchronize()
    cache_ok = all(eq(mix[l], torch.where(want[l].bool().view(BLOCKS, 1, BS, 1), plain[l].view(torch.int16),
                                          enc[l].view(torch.int16)))
                   and torch.equal(flags[l], want[l]) for l in range(LAYERS))
    multi_ok = all(eq(multi[l], mix[l]) and torch.equal(mflags[l], want[l]) for l in range(LAYERS))
    read = sorted({int(bt[i, p // BS]) * BS + p % BS for i, n in enumerate(BOUND_LENS) for p in range(n)})

    def attention(label: str) -> list[dict]:
        cells = []
        for layer in range(LAYERS):
            enc_read = [s for s in read if not is_plain[layer][s]] if label == "mixed" else read
            for sp in BOUND_SPLITS:
                part = torch.empty(P.part_floats(lib, batch, sp), dtype=torch.float32, device="cuda")
                o_p, o_f, o_r, o_n = (torch.full((batch, H * D), c, dtype=torch.bfloat16, device="cuda")
                                      for c in (7.0, -3.0, 2.0, 5.0))
                P.attn(lib, torch, plain[layer], lay, 0, LAYERS, layer, 0, k, q, o_p, part, lens, btd, batch, sp)
                P.attn(lib, torch, mix[layer], lay, 0, LAYERS, layer, 1, k, q, o_f, part, lens, btd, batch, sp,
                       bj=bjs[layer], plain=flags[layer])
                P.attn(lib, torch, mix[layer], lay, 0, LAYERS, layer, 1, k, q, o_n, part, lens, btd, batch, sp,
                       bj=bjs[layer])
                epoch.add_(1)
                rc_sk.fill_(-1)
                P.attn(lib, torch, mix[layer], lay, 0, LAYERS, layer, 1, k, q, o_r, part, lens, btd, batch, sp,
                       receipts=(rc_sk, used, epoch), bj=bjs[layer], plain=flags[layer])
                rc_ok = receipt_gate(mix[layer], layer, enc_read, f"deferred {label} attention layer={layer} sp={sp}")
                cells.append({"layer": layer, "splits": sp, "plain_slots": len(read) - len(enc_read),
                              "encoded_slots": len(enc_read), "equal": eq(o_p, o_f), "receipts_equal": eq(o_p, o_r),
                              "receipts": rc_ok, "unflagged_equal": eq(o_p, o_n)})
        return cells

    mixed = attention("mixed")
    mixed_ok = all(c["equal"] and c["receipts_equal"] and c["receipts"] and not c["unflagged_equal"] for c in mixed)
    blocks = torch.tensor(rng.sample(range(BLOCKS), 20), dtype=torch.int32, device="cuda")
    gather_ok = True
    for l in range(LAYERS):
        for with_rc in (False, True):
            o = torch.empty(20, *plain[l].shape[1:], dtype=torch.bfloat16, device="cuda")
            if with_rc:
                epoch.add_(1)
                rc_sk.fill_(-1)
            P.gather(lib, torch, mix[l], lay, 0, LAYERS, l, 1, k, blocks, o,
                     receipts=(rc_sk, used, epoch) if with_rc else None, bj=bjs[l], plain=flags[l])
            torch.cuda.synchronize()
            gather_ok &= eq(o, plain[l][blocks.long()])
            if with_rc:
                enc_slots = [b * BS + s for b in blocks.cpu().tolist() for s in range(BS) if not is_plain[l][b * BS + s]]
                gather_ok &= receipt_gate(mix[l], l, enc_slots, f"deferred gather layer={l}")
    tab = P.encode_table(torch, [(split[l][0], split[l][0].shape[0], mix[l], 0, LAYERS, l, bjs[l], flags[l])
                                 for l in range(LAYERS)]).cuda()
    P.encode(lib, torch, tab, LAYERS, max(split[l][0].shape[0] for l in range(LAYERS)), lay, k)
    torch.cuda.synchronize()
    encode_ok = all(eq(mix[l], enc[l]) and not bool(flags[l].any()) for l in range(LAYERS))
    after = attention("encoded")
    after_ok = all(c["equal"] and c["receipts_equal"] and c["receipts"] and c["unflagged_equal"] for c in after)
    all_slots = torch.arange(nslots, dtype=torch.int64, device="cuda")
    tab = P.encode_table(torch, [(all_slots, nslots, mix[l], 0, LAYERS, l, bjs[l], flags[l])
                                 for l in range(LAYERS)]).cuda()
    P.encode(lib, torch, tab, LAYERS, nslots, lay, k)
    torch.cuda.synchronize()
    again_ok = all(eq(mix[l], enc[l]) for l in range(LAYERS))
    try:
        P.attn(lib, torch, plain[0], lay, 0, LAYERS, 0, 0, k, q, torch.empty_like(q), torch.empty(
            P.part_floats(lib, batch, 1), dtype=torch.float32, device="cuda"), lens, btd, batch, 1, plain=flags[0])
        refused = False
    except RuntimeError:
        refused = True
    ok = cache_ok and multi_ok and mixed_ok and gather_ok and encode_ok and after_ok and again_ok and refused
    res = {"k": k, "plain_slots": [sum(p) for p in is_plain], "cache": cache_ok, "append_multi": multi_ok,
           "attention": mixed, "gather": gather_ok, "encode": encode_ok, "attention_encoded": after,
           "encode_again": again_ok, "plain_cache_refused": refused, "ok": ok}
    log(f"deferred k={k}: {'ok' if ok else 'FAIL'} (cache {cache_ok}, append_multi {multi_ok}, mixed attention "
        f"{sum(c['equal'] and c['receipts_equal'] and c['receipts'] and not c['unflagged_equal'] for c in mixed)}"
        f"/{len(mixed)}, gather {gather_ok}, encode {encode_ok}, attention after "
        f"{sum(c['equal'] and c['receipts_equal'] and c['receipts'] for c in after)}/{len(after)}, "
        f"encode again {again_ok}, plain cache refused {refused})")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="NAME=PATH, paged.cu built at field NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--batches", default="1,5,10,32")
    ap.add_argument("--splits", default="1,2,3,4,8,16,32")
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
    all_slots = torch.arange(BLOCKS * BS, dtype=torch.int64, device="cuda")

    def encoded_of(src, layer, k):
        e = torch.empty_like(src)
        keys = src[:, :, :, :D].permute(0, 2, 1, 3).reshape(BLOCKS * BS, G * D).contiguous()
        vals = src[:, :, :, D:].permute(0, 2, 1, 3).reshape(BLOCKS * BS, G * D).contiguous()
        P.append(lib, torch, e, lay, 0, LAYERS, layer, 1, k, keys, vals, all_slots)
        return e

    res = {"lib": lib.name, "shape": sh, "mask": lib.mask, "fold6": lib.fold6, "layout": lay.__dict__, "ks": {},
           "ok": True}
    st = P.stream(torch)
    seed = hashlib.sha512(b"verity/vllm-encoded/paged-gate/receipts").digest()
    packed = SR.sign_bits(seed)
    signs = torch.from_numpy(SR.sign_words(packed).view(np.int32)).cuda()
    prefix = ctypes.create_string_buffer(SR.rb.sketch_prefix(seed))
    lib.call("pg_rc_gate_pass", signs.data_ptr(), ctypes.addressof(prefix), st)
    rc_sk = torch.empty((BLOCKS * BS, G, SR.ROWS), dtype=torch.int32, device="cuda")
    used = torch.full((BLOCKS * BS,), -1, dtype=torch.int32, device="cuda")
    epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
    tags = torch.full((BLOCKS * BS, LAYERS, 8), -1, dtype=torch.int64, device="cuda")
    sign_bits = np.unpackbits(packed, axis=1, bitorder="little")
    receipter = SR.rb.SketchReceipter(seed)
    receipt_sets = 0
    plan_checked = False

    def receipt_gate(src, layer, slots, label):
        nonlocal receipt_sets, plan_checked
        active = sorted(set(map(int, slots)))
        P.receipts_sparse(lib, torch, rc_sk, tags, used, epoch, BLOCKS * BS, LAYERS, layer, 0)
        torch.cuda.synchronize()
        marked = (used == epoch).nonzero().flatten().cpu().tolist()
        blocks = []
        for slot in active:
            b, s = divmod(slot, BS)
            blocks.append(torch.cat((src[b, :, s, :D].reshape(-1), src[b, :, s, D:].reshape(-1))).contiguous()
                          .view(torch.int8))
        sk = SKT.sketch_torch(torch, torch.stack(blocks), sign_bits).cpu().numpy().astype("<i4") if blocks else np.empty(
            (0, SR.ROWS), dtype="<i4")
        want = [receipter.receipt(slot * LAYERS + layer, sk[i].tobytes()) for i, slot in enumerate(active)]
        got = [np.asarray(tags[slot, layer].cpu().numpy()).astype(np.uint64).astype(">u8").tobytes()
               for slot in active]
        plan_ok = True
        if not plan_checked and active:
            vectors = SR.kv_vectors(BLOCKS * BS, LAYERS)[[slot * LAYERS + layer for slot in active]]
            plan = SR.rb.plan_of(vectors, BLOCKS * BS * LAYERS)
            expected = SR.rb.Plan.of([slot * LAYERS + layer for slot in active], BLOCKS * BS * LAYERS)
            plan_ok, plan_checked = plan == expected, True
        ok = marked == active and got == want and plan_ok
        receipt_sets += 1
        if not ok:
            log(f"{label}: receipt FAIL marked={marked[:8]}/{len(marked)} active={active[:8]}/{len(active)} "
                f"tags={sum(a == b for a, b in zip(got, want))}/{len(want)} plan={plan_ok}")
        return ok

    for k in (int(v) for v in a.ks.split(",")):
        KB.set_key(lib, KB.lib_key(lib, k))
        key = KB.lib_key(lib, k)
        pl = [c.clone() for c in plain]
        enc = [encoded_of(pl[l], l, k) for l in range(LAYERS)]
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
            P.append(lib, torch, pl[l], lay, 0, LAYERS, l, 0, k, kk, vv, sl)
            P.append(lib, torch, enc[l], lay, 0, LAYERS, l, 1, k, kk, vv, sl)
            ref = pl[l].clone()
            for i, s in enumerate(slots.tolist()):
                if s < 0:
                    continue
                ref[s // BS, :, s % BS, :D] = kk[i].view(G, D)
                ref[s // BS, :, s % BS, D:] = vv[i].view(G, D)
            re = encoded_of(pl[l], l, k)
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
        def logical_word(i):
            b, r = divmod(i, lay.sb)
            g, r = divmod(r, lay.sh)
            s, j = divmod(r, lay.ss)
            wi = (j if j < sh["hw"] // 2 else 16 + j - sh["hw"] // 2) + g * (sh["hw"] // 2)
            return ((b * BS + s) * LAYERS + 1) * (4096 // WB) + wi

        py_ok = all(K.encode_word(int.from_bytes(bytes(w8[i].cpu().tolist()), "little"), logical_word(i), key)
                    == int.from_bytes(bytes(e8[i].cpu().tolist()), "little") for i in picks)
        tabs = {}
        for encd in (0, 1):
            ents = [(kk, vv, sl, n, pre[encd * LAYERS + l], 0, LAYERS, l) for l in range(LAYERS)]
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
                P.attn(lib, torch, pl[layer], lay, 0, LAYERS, layer, 0, k, q, o_p, part, lens, btd, b, sp)
                epoch.add_(1)
                rc_sk.fill_(-1)
                active = [int(bt[i, p // BS]) * BS + p % BS for i, length in enumerate(lens_l) for p in range(length)]
                P.attn(lib, torch, enc[layer], lay, 0, LAYERS, layer, 1, k, q, o_e, part, lens, btd, b, sp,
                       receipts=(rc_sk, used, epoch))
                receipts_all = receipt_gate(enc[layer], layer, active, f"attention k={k} B={b} sp={sp}")
                qrow = torch.tensor(rng.sample(range(b), b), dtype=torch.int32, device="cuda")
                qp = torch.empty(b, H * D, dtype=torch.bfloat16, device="cuda")
                qp[qrow.long()] = q[:b]
                o_r = torch.full((b, H * D), 5.0, dtype=torch.bfloat16, device="cuda")
                P.attn(lib, torch, enc[layer], lay, 0, LAYERS, layer, 1, k, qp, o_r, part, lens, btd, b, sp, qrow=qrow)
                # the last tokens as rows, their slots in the pages overwritten
                ck = torch.zeros(nrows, G * D, dtype=torch.bfloat16, device="cuda")
                cv = torch.zeros_like(ck)
                pl_x, enc_x = pl[layer].clone(), enc[layer].clone()
                readers: dict[tuple, int] = {}   # the tables overlap: a slot is overwritten only if one token reads it
                for i in range(b):
                    for p in range(lens_l[i]):
                        at = (int(bt[i, p // BS]), p % BS)
                        readers[at] = readers.get(at, 0) + 1
                overwritten = 0
                for i in range(b):
                    L = lens_l[i]
                    if not L:
                        continue
                    blk, off = int(bt[i, (L - 1) // BS]), (L - 1) % BS
                    ck[i] = pl[layer][blk, :, off, :D].reshape(-1)
                    cv[i] = pl[layer][blk, :, off, D:].reshape(-1)
                    if readers[(blk, off)] == 1:
                        junk = (torch.randn(G, 2 * D, device="cuda") * 9).to(torch.bfloat16)
                        pl_x[blk, :, off] = junk
                        enc_x[blk, :, off] = junk
                        overwritten += 1
                o_cp = torch.full_like(o_p, 1.0)
                o_ce = torch.full_like(o_p, 2.0)
                P.attn(lib, torch, pl_x, lay, 0, LAYERS, layer, 0, k, q, o_cp, part, lens, btd, b, sp, ck=ck, cv=cv)
                epoch.add_(1)
                rc_sk.fill_(-1)
                active_fresh = [int(bt[i, p // BS]) * BS + p % BS for i, length in enumerate(lens_l)
                                for p in range(max(0, length - 1))]
                P.attn(lib, torch, enc_x, lay, 0, LAYERS, layer, 1, k, q, o_ce, part, lens, btd, b, sp, ck=ck, cv=cv,
                       receipts=(rc_sk, used, epoch))
                receipts_fresh = receipt_gate(enc_x, layer, active_fresh, f"attention-fresh k={k} B={b} sp={sp}")
                ckp, cvp = torch.empty(b, G * D, dtype=torch.bfloat16, device="cuda"), torch.empty(
                    b, G * D, dtype=torch.bfloat16, device="cuda")
                ckp[qrow.long()], cvp[qrow.long()] = ck[:b], cv[:b]
                o_cr = torch.full((b, H * D), 4.0, dtype=torch.bfloat16, device="cuda")
                P.attn(lib, torch, enc_x, lay, 0, LAYERS, layer, 1, k, qp, o_cr, part, lens, btd, b, sp, qrow=qrow, ck=ckp,
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
                ok = equal and zero_pad and zero_len and qrow_eq and rows_eq and receipts_all and receipts_fresh and err <= RTOL
                att["cells"].append({"batch": b, "splits": sp, "layer": layer, "lens": lens_l, "equal": equal,
                                     "zero_padding": zero_pad, "zero_length": zero_len, "qrow": qrow_eq,
                                     "last_rows": rows_eq, "last_overwritten": overwritten,
                                     "receipts": receipts_all, "fresh_receipts": receipts_fresh,
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
                if encd:
                    epoch.add_(1)
                    rc_sk.fill_(-1)
                    P.gather(lib, torch, src, lay, 0, LAYERS, l, encd, k, blocks, o,
                             receipts=(rc_sk, used, epoch))
                    gather_slots = [int(b) * BS + s for b in blocks.cpu().tolist() for s in range(BS)]
                    ok_g &= receipt_gate(src, l, gather_slots, f"gather k={k} layer={l}")
                else:
                    P.gather(lib, torch, src, lay, 0, LAYERS, l, encd, k, blocks, o)
                torch.cuda.synchronize()
                ok_g &= bool(torch.equal(o.view(torch.int16), pl[l][blocks.long()].view(torch.int16)))
        g["gather"] = {"blocks": 20, "receipt_sets": LAYERS, "ok": ok_g}
        g["ok"] = g["append"]["ok"] and multi_ok and att["ok"] and ok_g
        res["ks"][k] = g
        res["ok"] &= g["ok"]
        errs = [c["rel_err"] for c in att["cells"]]
        log(f"k={k}: append {'ok' if g['append']['ok'] else 'FAIL'}, append_multi {'ok' if multi_ok else 'FAIL'}, attention {'ok' if att['ok'] else 'FAIL'} "
            f"({len(att['cells'])} cells, max rel err {max(errs):.1e}), gather {'ok' if ok_g else 'FAIL'}")
    res["receipt_sets"] = receipt_sets
    res["plan_of"] = plan_checked
    res["bound"] = bound_gate(lib, torch, [int(v) for v in a.ks.split(",")], log, (rc_sk, used, epoch, receipt_gate))
    res["ok"] &= res["bound"]["ok"]
    (out_dir / "paged-gate.json").write_text(json.dumps(res, indent=1))
    log("paged gate " + ("passed" if res["ok"] else "FAILED"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
