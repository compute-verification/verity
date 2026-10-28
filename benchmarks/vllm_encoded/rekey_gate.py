"""`paged.cu`'s re-key (`pg_rekey`) on one GPU: its gate, then its rate alone and beside a GEMM.

A token-layer kept past the lifetime cap L is re-keyed in its slot (remote_bandwidth's PROTOCOL.md §4): appended again
at a fresh index j' under a fresh coin, so its bound offset moves from b_j to b_j'(c_a'), and its words are decoded
under the old and encoded under the new in place. For each k of `--ks`, on caches in vLLM's layout ([pages][KV heads]
[slots][K ‖ V], two layers, store base 77 TLW, pg_wi's word order), every slot under a random offset of its own and
about a quarter of them plaintext (deferred residency's flag), each check bit for bit:

    cache       half the slots of both layers (padding rows among them) re-keyed in one launch on a low-priority stream
                while a GEMM runs on a high-priority one: the cache equals the one `pg_append` writes under the new
                offsets (plaintext slots as they are), every word
    kv_py       sampled words of re-keyed slots equal `kv.py`'s encode under the new offset of its decode under the old,
                and decode under the new to the payload
    untouched   the slots not listed and the listed plaintext ones keep their words; the flags are unchanged
    offsets     the bound offsets hold the new ones at every listed slot (plaintext ones too) and the old elsewhere
    twice       re-keyed again (other new offsets): the cache equals `pg_append`'s under them, and the gather of every
                page and decode attention under them and the flags equal the plain cache's
    back        re-keyed to the first offsets: the cache and offsets equal the first ones
    ctas        grids of 1 CTA, of 7 and of every SM's resident CTAs give the default grid's cache
    pinned      (a library at PoUB's codec, at its rounds) `vectors.json`'s `codec.coin` blocks, each
                appended at a slot that is not its index's under another index's and coin's offset folded to the slot,
                then re-keyed to its own b_j(c_a) folded likewise: the token-layer is the pinned block (SHA-512, first word)
    refusals    k = 0, k past KMAX, negative rows, a negative grid

Timing (`--rekey-slots` random slots of each of `--layers` layers of `--pages` pages, vLLM's layout, random words, every
slot under its own offset), for each k of `--ks`: the re-key's rate in blocks (4 KiB token-layers) a second and in
squarings a second (a block's TLW words, each k rounds of the root's chain and the decode's one squaring), at grids of
a word a thread (the default), every SM's resident CTAs, one CTA an SM and one every other SM; alone, and beside a bf16
GEMM loop (`--gemm` M,K,N, torch on a high-priority stream, the re-key on the low-priority one) queued to outlast it: the
re-key's time, and the GEMM's time a call over the GEMMs done while the re-key ran. Writes OUT/rekey-gate.json.

    python rekey_gate.py --lib p1024=paged.so --out OUT [--ks 2,3 --pages 1024 --layers 32 --rekey-slots 2048]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "verity" / "core" / "protocols" / "remote_bandwidth" / "tests")]

import paged as P  # noqa: E402
from paged import K, KB  # noqa: E402
import remote_bandwidth_vectors as RV  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as RC  # noqa: E402

BLOCKS, BS, LAYERS = 96, 16, 2
SB = 77
SAMPLES = 64
LENS = (1, 17, 64, 300, 511)
SPLITS = (1, 4)
SPIN_CYCLES = 20_000_000


def eq(torch, a, b) -> bool:
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


class Case:
    """Two layers' plain caches, their bound offsets and flags, and `pg_append`'s encoded cache under any offsets."""

    def __init__(self, lib, torch, k: int, key, rng: random.Random) -> None:
        G, D, WB, f = lib.shape["kv_heads"], lib.shape["head_dim"], lib.shape["wb"], lib.field
        self.lib, self.torch, self.k, self.key, self.f = lib, torch, k, key, f
        self.nslots, self.W = BLOCKS * BS, K.token_layer_words(f)
        self.store_base = SB * self.W
        self.plain = [(torch.randn(BLOCKS, G, BS, 2 * D, device="cuda") * 2).to(torch.bfloat16) for _ in range(LAYERS)]
        self.lay = P.layout(self.plain[0], G, WB)
        self.rows = [(p[..., :D].permute(0, 2, 1, 3).reshape(self.nslots, G * D).contiguous(),
                      p[..., D:].permute(0, 2, 1, 3).reshape(self.nslots, G * D).contiguous()) for p in self.plain]
        self.offs = [[rng.randrange(f.p) for _ in range(self.nslots)] for _ in range(LAYERS)]
        self.is_plain = [[rng.random() < 0.25 for _ in range(self.nslots)] for _ in range(LAYERS)]
        self.want = [torch.tensor(p, dtype=torch.uint8, device="cuda") for p in self.is_plain]
        self.flags = [w.clone() for w in self.want]

    def written(self, offs: list[list[int]]) -> list:
        """The cache `pg_append` writes with every slot under offs, a plaintext slot as it is (and its flag set)."""
        torch, out = self.torch, []
        for l in range(LAYERS):
            c = torch.empty_like(self.plain[l])
            flags = torch.zeros(self.nslots, dtype=torch.uint8, device="cuda")
            bj = P.offsets(torch, offs[l], self.f)
            for encd in (0, 1):
                sl = torch.tensor([s for s in range(self.nslots) if self.is_plain[l][s] != bool(encd)],
                                  dtype=torch.int64, device="cuda")
                P.append(self.lib, torch, c, self.lay, self.store_base, LAYERS, l, encd, self.k, self.rows[l][0][sl],
                         self.rows[l][1][sl], sl, bj=bj if encd else None, plain=flags)
            torch.cuda.synchronize()
            if not torch.equal(flags, self.want[l]):
                raise RuntimeError("pg_append's flags are not the case's")
            out.append(c)
        return out

    def codec_index(self, slot: int, layer: int, g: int, j: int) -> int:
        hw, wb = self.lib.shape["hw"], self.lib.shape["wb"]
        wi = (j if j < hw // 2 else 2048 // wb + j - hw // 2) + g * (hw // 2)
        return self.store_base + (slot * LAYERS + layer) * self.W + wi

    def word(self, cache, slot: int, g: int, j: int) -> int:
        wb, lay = self.lib.shape["wb"], self.lay
        bo, so = divmod(slot, BS)
        i = bo * lay.sb + g * lay.sh + so * lay.ss + j
        return int.from_bytes(cache.view(self.torch.uint8).view(-1, wb)[i].cpu().numpy().tobytes(), "little")


def listed(rng: random.Random, nslots: int) -> tuple[list[list[int]], int]:
    """Each layer's listed slots: half of them, shuffled, padding (−1) among them, one layer's rows fewer than the other's."""
    out = []
    for l in range(LAYERS):
        sl = rng.sample(range(nslots), nslots // 2 - 17 * l)
        for _ in range(5):
            sl.insert(rng.randrange(len(sl) + 1), -1)
        out.append(sl)
    return out, max(len(s) for s in out)


def table(torch, case: Case, caches, bjs, slots, news):
    """The device table and the tensors it points to, which the caller holds until the re-key has run."""
    entries = []
    for l in range(LAYERS):
        st = torch.tensor(slots[l], dtype=torch.int64, device="cuda")
        entries.append((st, len(slots[l]), caches[l], case.store_base, LAYERS, l, bjs[l], news[l], case.flags[l]))
    return P.rekey_table(torch, entries).cuda(), entries


def moved(case: Case, offs, slots, new_vals) -> list[list[int]]:
    out = [list(o) for o in offs]
    for l in range(LAYERS):
        for i, s in enumerate(slots[l]):
            if s >= 0:
                out[l][s] = new_vals[l][i]
    return out


def new_offsets(torch, case: Case, rng: random.Random, slots) -> tuple[list[list[int]], list]:
    vals = [[rng.randrange(case.f.p) for _ in s] for s in slots]
    return vals, [P.offsets(torch, v, case.f) for v in vals]


def attention_ok(lib, torch, case: Case, caches, bjs, rng: random.Random) -> list[dict]:
    H, D = lib.shape["heads"], lib.shape["head_dim"]
    batch = len(LENS)
    bt = torch.zeros(batch, 32, dtype=torch.int32)
    perm = rng.sample(range(BLOCKS), BLOCKS)
    for i, n in enumerate(LENS):
        nb = -(-n // BS)
        bt[i, :nb] = torch.tensor(perm[2 * i:2 * i + nb] if 2 * i + nb <= BLOCKS else perm[:nb], dtype=torch.int32)
    btd, lens = bt.cuda(), torch.tensor(LENS, dtype=torch.int32, device="cuda")
    q = torch.randn(batch, H * D, device="cuda").to(torch.bfloat16)
    cells = []
    for l in range(LAYERS):
        for sp in SPLITS:
            part = torch.empty(P.part_floats(lib, batch, sp), dtype=torch.float32, device="cuda")
            o_p, o_e = (torch.full((batch, H * D), c, dtype=torch.bfloat16, device="cuda") for c in (7.0, -3.0))
            P.attn(lib, torch, case.plain[l], case.lay, case.store_base, LAYERS, l, 0, case.k, q, o_p, part, lens, btd,
                   batch, sp)
            P.attn(lib, torch, caches[l], case.lay, case.store_base, LAYERS, l, 1, case.k, q, o_e, part, lens, btd,
                   batch, sp, bj=bjs[l], plain=case.flags[l])
            torch.cuda.synchronize()
            cells.append({"layer": l, "splits": sp, "equal": eq(torch, o_p, o_e)})
    return cells


def gate_k(lib, torch, k: int, rng: random.Random, log) -> dict:
    key = KB.lib_key(lib, k)
    KB.set_key(lib, key)
    case = Case(lib, torch, k, key, rng)
    lo, hi = torch.cuda.Stream.priority_range()
    s_lo, s_hi = torch.cuda.Stream(priority=lo), torch.cuda.Stream(priority=hi)
    first = case.written(case.offs)
    caches = [c.clone() for c in first]
    bjs = [P.offsets(torch, o, case.f) for o in case.offs]
    slots, rows = listed(rng, case.nslots)
    v1, n1 = new_offsets(torch, case, rng, slots)
    tab, held = table(torch, case, caches, bjs, slots, n1)
    a = torch.randn(2048, 4096, device="cuda").to(torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda").to(torch.bfloat16)
    c = torch.empty(2048, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    s_lo.wait_stream(torch.cuda.current_stream())
    s_hi.wait_stream(torch.cuda.current_stream())
    P.rekey(lib, torch, tab, LAYERS, rows, case.lay, k, s=s_lo.cuda_stream)
    with torch.cuda.stream(s_hi):
        for _ in range(20):
            torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    offs1 = moved(case, case.offs, slots, v1)
    want1 = case.written(offs1)
    out = {"k": k, "rows": rows, "listed": [sum(s >= 0 for s in sl) for sl in slots],
           "listed_plaintext": [sum(case.is_plain[l][s] for s in slots[l] if s >= 0) for l in range(LAYERS)]}
    out["cache"] = all(eq(torch, caches[l], want1[l]) for l in range(LAYERS))
    lset = [set(s for s in sl if s >= 0) for sl in slots]
    picks = []
    pool = [(l, s) for l in range(LAYERS) for s in sorted(lset[l]) if not case.is_plain[l][s]]
    for l, s in rng.sample(pool, min(SAMPLES, len(pool))):
        g, j = rng.randrange(lib.shape["kv_heads"]), rng.randrange(lib.shape["hw"])
        i = case.codec_index(s, l, g, j)
        y0, y1, m = case.word(first[l], s, g, j), case.word(caches[l], s, g, j), case.word(case.plain[l], s, g, j)
        new = offs1[l][s]
        picks.append(K.encode_word(K.decode_word(y0, i, key, case.offs[l][s]), i, key, new) == y1
                     and K.decode_word(y1, i, key, new) == m)
    out["kv_py"] = {"words": len(picks), "ok": all(picks)}
    keep_ok = True
    for l in range(LAYERS):
        keep = torch.tensor([s not in lset[l] or case.is_plain[l][s] for s in range(case.nslots)], device="cuda")
        m = keep.view(BLOCKS, 1, BS, 1).expand_as(first[l])
        keep_ok &= eq(torch, caches[l][m], first[l][m])
    out["untouched"] = bool(keep_ok and all(torch.equal(case.flags[l], case.want[l]) for l in range(LAYERS)))
    out["offsets"] = all(torch.equal(bjs[l], P.offsets(torch, offs1[l], case.f)) for l in range(LAYERS))

    v2, n2 = new_offsets(torch, case, rng, slots)
    tab2, held2 = table(torch, case, caches, bjs, slots, n2)
    P.rekey(lib, torch, tab2, LAYERS, rows, case.lay, k)
    torch.cuda.synchronize()
    offs2 = moved(case, offs1, slots, v2)
    want2 = case.written(offs2)
    gather_ok = True
    blocks = torch.arange(BLOCKS, dtype=torch.int32, device="cuda")
    for l in range(LAYERS):
        o = torch.empty_like(case.plain[l])
        P.gather(lib, torch, caches[l], case.lay, case.store_base, LAYERS, l, 1, k, blocks, o, bj=bjs[l],
                 plain=case.flags[l])
        torch.cuda.synchronize()
        gather_ok &= eq(torch, o, case.plain[l])
    att = attention_ok(lib, torch, case, caches, bjs, rng)
    out["twice"] = {"cache": all(eq(torch, caches[l], want2[l]) for l in range(LAYERS)),
                    "offsets": all(torch.equal(bjs[l], P.offsets(torch, offs2[l], case.f)) for l in range(LAYERS)),
                    "gather": gather_ok, "attention": f"{sum(x['equal'] for x in att)}/{len(att)}"}
    out["twice"]["ok"] = bool(out["twice"]["cache"] and out["twice"]["offsets"] and gather_ok
                              and all(x["equal"] for x in att))

    back_vals = [[case.offs[l][s] if s >= 0 else 0 for s in slots[l]] for l in range(LAYERS)]
    tab3, held3 = table(torch, case, caches, bjs, slots, [P.offsets(torch, v, case.f) for v in back_vals])
    P.rekey(lib, torch, tab3, LAYERS, rows, case.lay, k)
    torch.cuda.synchronize()
    out["back"] = bool(all(eq(torch, caches[l], first[l]) for l in range(LAYERS))
                       and all(torch.equal(bjs[l], P.offsets(torch, case.offs[l], case.f)) for l in range(LAYERS)))

    grids = {}
    for ctas in (1, 7, lib.so.pg_rekey_ctas()):
        cc = [x.clone() for x in first]
        bb = [P.offsets(torch, o, case.f) for o in case.offs]
        t, held_g = table(torch, case, cc, bb, slots, n1)
        P.rekey(lib, torch, t, LAYERS, rows, case.lay, k, ctas=ctas)
        torch.cuda.synchronize()
        grids[ctas] = all(eq(torch, cc[l], want1[l]) for l in range(LAYERS))
    out["ctas"] = grids
    out["refusals"] = refusals(lib, torch, case, tab, rows)
    out["ok"] = bool(out["cache"] and out["kv_py"]["ok"] and out["untouched"] and out["offsets"] and out["twice"]["ok"]
                     and out["back"] and all(grids.values()) and out["refusals"]["ok"])
    log(f"k={k}: {'ok' if out['ok'] else 'FAIL'} (listed {out['listed']} of {case.nslots} slots a layer, plaintext "
        f"{out['listed_plaintext']}; cache {out['cache']}, kv.py {out['kv_py']}, untouched {out['untouched']}, offsets "
        f"{out['offsets']}, twice {out['twice']}, back {out['back']}, ctas {grids}, refusals {out['refusals']})")
    return out


def refusals(lib, torch, case: Case, tab, rows: int) -> dict:
    lay, st = case.lay, P.stream(torch)

    def call(**kw) -> bool:
        a = dict(layers=LAYERS, rows=rows, k=case.k, ctas=0)
        a.update(kw)
        try:
            lib.call("pg_rekey", tab.data_ptr(), a["layers"], a["rows"], lay.bs, lay.sb, lay.sh, lay.ss, a["k"],
                     a["ctas"], st)
        except RuntimeError:
            return True
        return False

    torch.cuda.synchronize()
    res = {"k_zero": call(k=0), "k_past_kmax": call(k=lib.kmax + 1), "negative_rows": call(rows=-1),
           "negative_grid": call(ctas=-1)}
    torch.cuda.synchronize()
    res["ok"] = all(res.values())
    return res


def pinned(lib, torch, log) -> dict:
    """`vectors.json`'s codec.coin blocks re-keyed into place at a slot that is not their index's."""
    G, D, WB, f = lib.shape["kv_heads"], lib.shape["head_dim"], lib.shape["wb"], lib.field
    rc = RC.DEFAULT
    vec = json.loads(RV.PATH.read_text())["codec"]
    if (f.name, lib.mask, vec["name"]) != (rc.field.name, rc.masked, rc.name) or rc.rounds > lib.kmax:
        return {"skipped": f"the library is {f.name}{'/masked' if lib.mask else ''}, not {rc.name}"}
    seed = bytes.fromhex(vec["seed"])
    key, dev = RC.key(seed, rc), K.key(f.name, seed, rc.rounds, mask=rc.masked)
    if (dev.c, dev.tau, dev.mask) != (key.c, key.tau, key.mask):
        raise RuntimeError("kv.py's key is not PoUB's codec key")
    KB.set_key(lib, dev)
    W, nslots = K.token_layer_words(f), BLOCKS * BS
    cache = torch.zeros(BLOCKS, G, BS, 2 * D, dtype=torch.bfloat16, device="cuda")
    lay = P.layout(cache, G, WB)
    c_a = bytes.fromhex(vec["coin"]["c_a"])
    c_old = hashlib.sha512(b"verity/rekey-gate/old-coin").digest()[:RC.COIN_BYTES]
    blocks = []
    for v in vec["coin"]["bound"]:
        j = v["j"]
        pay = RV.payload(j)
        d = RC.digest(pay)
        there = (nslots - 1 - j // LAYERS) * LAYERS + (j + 1) % LAYERS
        slot, layer = divmod(there, LAYERS)
        j_old = j + 1000
        old = (RC.binding(d, j_old, key, c_old) + key.tau * W * (j_old - there)) % f.p
        new = (RC.binding(d, j, key, c_a) + key.tau * W * (j - there)) % f.p
        bj = P.offsets(torch, [old if s == slot else 0 for s in range(nslots)], f)
        at = torch.tensor([slot], dtype=torch.int64, device="cuda")
        kr = torch.frombuffer(bytearray(pay[:2048]), dtype=torch.bfloat16).view(1, G * D).cuda()
        vr = torch.frombuffer(bytearray(pay[2048:]), dtype=torch.bfloat16).view(1, G * D).cuda()
        P.append(lib, torch, cache, lay, 0, LAYERS, layer, 1, rc.rounds, kr, vr, at, bj=bj)
        torch.cuda.synchronize()
        before = token_layer(torch, cache, slot, D)
        new_t = P.offsets(torch, [new], f)
        tab = P.rekey_table(torch, [(at, 1, cache, 0, LAYERS, layer, bj, new_t)]).cuda()
        P.rekey(lib, torch, tab, 1, 1, lay, rc.rounds)
        torch.cuda.synchronize()
        got = token_layer(torch, cache, slot, D)
        blocks.append({"j": j, "slot": slot, "layer": layer, "j_old": j_old,
                       "old_differs": hashlib.sha512(before).hexdigest() != v["block_sha512"],
                       "word_0": got[:WB].hex() == v["word_0"],
                       "block_sha512": hashlib.sha512(got).hexdigest() == v["block_sha512"],
                       "offset": torch.equal(bj[slot], new_t[0])})
    ok = all(b["old_differs"] and b["word_0"] and b["block_sha512"] and b["offset"] for b in blocks)
    log(f"pinned {rc.name}: {'ok' if ok else 'FAIL'} {blocks}")
    return {"codec": rc.name, "blocks": blocks, "ok": ok}


def token_layer(torch, cache, slot: int, D: int) -> bytes:
    bo, so = divmod(slot, BS)
    return torch.cat((cache[bo, :, so, :D].reshape(-1), cache[bo, :, so, D:].reshape(-1))).view(
        torch.uint8).cpu().numpy().tobytes()


def timing(lib, torch, a, log) -> dict:
    G, D, WB, f = lib.shape["kv_heads"], lib.shape["head_dim"], lib.shape["wb"], lib.field
    NL, W = f.limbs, K.token_layer_words(f)
    pages, L, n = a.pages, a.layers, a.rekey_slots
    nslots = pages * BS
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    lo, hi = torch.cuda.Stream.priority_range()
    s_lo, s_hi = torch.cuda.Stream(priority=lo), torch.cuda.Stream(priority=hi)
    caches = [torch.randint(-2 ** 15, 2 ** 15 - 1, (pages, G, BS, 2 * D), dtype=torch.int16, device="cuda")
              .view(torch.bfloat16) for _ in range(L)]
    lay = P.layout(caches[0], G, WB)

    def rand_offsets(rows):
        b = torch.randint(-2 ** 31, 2 ** 31 - 1, (rows, NL), dtype=torch.int32, device="cuda")
        b[:, -1] = 0
        return b

    bjs = [rand_offsets(nslots) for _ in range(L)]
    rng = random.Random(11)
    slots = [torch.tensor(rng.sample(range(nslots), n), dtype=torch.int64, device="cuda") for _ in range(L)]
    news = [rand_offsets(n) for _ in range(L)]
    tab = P.rekey_table(torch, [(slots[l], n, caches[l], 0, L, l, bjs[l], news[l]) for l in range(L)]).cuda()
    M, Kd, N = (int(v) for v in a.gemm.split(","))
    ga = torch.randn(M, Kd, device="cuda").to(torch.bfloat16)
    gb = torch.randn(Kd, N, device="cuda").to(torch.bfloat16)
    gc = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    blocks = L * n
    resident = lib.so.pg_rekey_ctas()
    grids = {"a_word_a_thread": 0, "resident": resident, "one_an_sm": sms, "every_other_sm": sms // 2}
    res = {"pages": pages, "layers": L, "slots_a_layer": n, "blocks": blocks, "sms": sms, "grids": grids,
           "priorities": {"rekey": lo, "gemm": hi}, "gemm": {"m": M, "k": Kd, "n": N}, "ks": {}}
    log(f"timing: {blocks} blocks ({L} layers x {n} slots of {nslots}), grids {grids}, GEMM {M}x{Kd}x{N} bf16, "
        f"stream priorities re-key {lo} GEMM {hi}")

    def on(s, fn):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(s)
        fn()
        e1.record(s)
        return e0, e1

    def gemms(cnt):
        def run():
            with torch.cuda.stream(s_hi):
                for _ in range(cnt):
                    torch.matmul(ga, gb, out=gc)
        return run

    def alone(s, fn, reps):
        out = []
        for _ in range(reps + 1):
            torch.cuda.synchronize()
            e0, e1 = on(s, fn)
            e1.synchronize()
            out.append(e0.elapsed_time(e1) / 1e3)
        return statistics.median(out[1:])

    def together(rk, cnt):
        """The re-key's time beside cnt GEMMs queued with it, and each GEMM's end, from when both may start."""
        torch.cuda.synchronize()
        lib.call("kv_spin", SPIN_CYCLES, P.stream(torch))
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        s_lo.wait_event(ev)
        s_hi.wait_event(ev)
        r1 = torch.cuda.Event(enable_timing=True)
        rk()
        r1.record(s_lo)
        ends = []
        with torch.cuda.stream(s_hi):
            for _ in range(cnt):
                torch.matmul(ga, gb, out=gc)
                ends.append(torch.cuda.Event(enable_timing=True))
                ends[-1].record(s_hi)
        torch.cuda.synchronize()
        return ev.elapsed_time(r1) / 1e3, [ev.elapsed_time(e) / 1e3 for e in ends]

    def beside(rk, t):
        """The re-key beside a GEMM loop queued to outlast it (doubled up to 3 times if it doesn't), and the GEMM's time
        a call while the re-key runs, over the GEMMs done before it ends (None if a run had none)."""
        cnt = max(4, math.ceil(1.5 * t / g1))
        for _ in range(3):
            tr, ends = together(rk, cnt)
            if ends[-1] >= tr:
                break
            cnt *= 2
        runs = [(tr, ends)] + [together(rk, cnt) for _ in range(a.reps - 1)]
        done = [sum(e <= tr for e in ends) for tr, ends in runs]
        gm = [ends[d - 1] / d for (_, ends), d in zip(runs, done) if d]
        return {"s": statistics.median(tr for tr, _ in runs),
                "gemm_s": statistics.median(gm) if len(gm) == len(runs) else None, "gemms_queued": cnt,
                "gemms_done_during": done, "gemm_loop_ended_first": sum(ends[-1] < tr for tr, ends in runs)}

    g1 = alone(s_hi, gemms(10), a.reps) / 10
    res["gemm"]["alone_ms"] = round(g1 * 1e3, 4)
    res["gemm"]["tflops"] = round(2 * M * Kd * N / g1 / 1e12, 1)
    log(f"GEMM alone {g1 * 1e3:.3f} ms ({res['gemm']['tflops']} TFLOP/s)")
    for k in (int(v) for v in a.ks.split(",")):
        KB.set_key(lib, KB.lib_key(lib, k))
        sq = W * k * (f.root_squarings + 1)
        cell = {"squarings_a_block": sq, "grids": {}}
        for name, ctas in grids.items():
            def rk(ctas=ctas):
                P.rekey(lib, torch, tab, L, n, lay, k, ctas=ctas, s=s_lo.cuda_stream)

            t = alone(s_lo, rk, a.reps)
            b = beside(rk, t)
            tb, gm = b["s"], b["gemm_s"]
            g = {"ctas": ctas or -(-L * n * G * lib.shape["hw"] // 128), "alone_s": round(t, 5),
                 "blocks_s": round(blocks / t), "squarings_s": blocks / t * sq, "beside_s": round(tb, 5),
                 "beside_blocks_s": round(blocks / tb), "beside_squarings_s": blocks / tb * sq,
                 "beside_over_alone": round(tb / t, 3), "gemm_beside_ms": gm and round(gm * 1e3, 4),
                 "gemm_slowdown": gm and round(gm / g1, 3), **{x: b[x] for x in b if x.startswith("gemms")},
                 "gemm_loop_ended_first": b["gemm_loop_ended_first"]}
            cell["grids"][name] = g
            gemm = f"{g['gemm_beside_ms']} ms a call ({g['gemm_slowdown']}x)" if gm else "starved (a run had none done)"
            first = b["gemm_loop_ended_first"]
            log(f"k={k} {name} ({g['ctas']} CTAs): alone {g['blocks_s']:.4g} blocks/s ({g['squarings_s']:.4g} sq/s); "
                f"beside the GEMM {g['beside_blocks_s']:.4g} blocks/s ({g['beside_over_alone']}x the time), the GEMM "
                f"while it runs {gemm}, GEMMs done in it {b['gemms_done_during']} of {b['gemms_queued']} queued"
                + (f", the GEMM loop ended first in {first} runs" if first else ""))
        res["ks"][k] = cell
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True, help="NAME=PATH, paged.cu built at field NAME")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--pages", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--rekey-slots", type=int, default=2048)
    ap.add_argument("--gemm", default="2048,4096,14336")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "rekey-gate.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    lib = P.PagedLib(*a.lib.split("=", 1))
    log(f"{lib.name}: shape {lib.shape}, mask {lib.mask}, fold6 {lib.fold6}, re-key grid {lib.so.pg_rekey_ctas()}")
    torch.manual_seed(0)
    rng = random.Random(3)
    res = {"lib": lib.name, "shape": lib.shape, "mask": lib.mask, "ks": {}, "ok": True}
    res["pinned"] = pinned(lib, torch, log)
    res["ok"] &= res["pinned"].get("ok", True)
    for k in (int(v) for v in a.ks.split(",")):
        res["ks"][k] = gate_k(lib, torch, k, rng, log)
        res["ok"] &= res["ks"][k]["ok"]
    res["gate_ok"] = res["ok"]
    if res["ok"] and not a.no_timing:
        res["timing"] = timing(lib, torch, a, log)
    (out_dir / "rekey-gate.json").write_text(json.dumps(res, indent=1))
    log("re-key gate " + ("passed" if res["ok"] else "FAILED"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
