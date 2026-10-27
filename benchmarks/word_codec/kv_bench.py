"""The KV cache codec's benchmark on one GPU: `kv.cu`'s libraries (`--lib NAME=PATH`, one per field of `kv.FIELDS`) gated
against `kv.py`, then timed. Writes `kv-codec.json` and `kv.log` under `--out`, and, with `RESEARCH_RUN_DIR` set, its `result.json`.

Nothing is timed before its gate passed, in this order:
- `gate`: at every k of `--ks`, device encode equals `kv.encode_word` on edge and random payloads at indices from each of
  `GATE_BASES` (below 2^32, across it within a warp, and past it: the tweak's high word or none), device decode of the
  reference's words equals their payloads, and `--gate-words` random words
  go through the device's encode and decode unchanged, the decode's checksum mode giving the payloads' XOR;
- `gate_bound`: the bound key's launchers (`kv_encode_bound`, `kv_decode_bound`) against `tests/kv_vectors.json`'s
  "bound" at the keys of the library's field and mask, and against `kv.py` over token-layers across index 2^32; at the
  remote audit's codec, against its `vectors.json`'s `codec.coin` too, the append coin's form (`gate_coin`);
  `--gates-only` stops here;
- `read`: the read rate R over a store of `--store-gib`, the best of a few grids;
- `rates`: compute-only squarings (ILP 1 and 2 words a thread), decodes and encodes of k rounds, words/s;
- `store`: the store's random words encoded on the device at each k (the encode's rate through memory), then its
  read-and-decode (checksum mode) timed, GB/s of stored words, against R;
- `token`: one launch encoding a generated token's KV cache at Llama-3.1-70B (`kv.token_words`: 80 layers of 4,096
  bytes, K then V) for each batch of `--batches`, a thread a word, queued behind a spin, so its latency; beside the
  SM time the same words take at the compute-only rate;
- `attention`: `kv_attn`, a layer's decode attention at Llama-3.1-70B's shape (64 query heads on 8 KV heads of 128,
  one query token a sequence) for batch × context, over the plain cache and over its words decoding inside the kernel
  at each k of `--attn-ks`, gated equal bit for bit and against float64 attention, timed beside torch's SDPA;
- `native`: cuBLASLt FP8 (`torch._scaled_mm`) at Llama-3.1-70B's four linears and each batch, every call on cold
  weights (a ring of copies past the L2), times 80 layers: the decode step's linears, the token's cost's denominator.

`model` restates the token's encode and the decode inside attention in passes of E (|E| = the 80 layers' linears in
FP8, read at R) for batch × context: the measured decode inside `kv_attn` (encoded minus plain, 80 layers), and the
bounds from the read-and-decode, the read and the compute-only decode overlapped or added.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kv as K  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/kv/bench/seed").digest()
SPIN_CYCLES = 20_000_000
L2_DEFEAT_BYTES = 1 << 30
SHAPES_70B = {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}
E_70B = K.LAYERS * sum(n * kd for n, kd in SHAPES_70B.values())
CONTEXTS = (512, 2048, 8192)
GATE_BASES = (77, (1 << 32) - 20, (1 << 33) + 77)
BOUND_VECTORS = HERE / "tests" / "kv_vectors.json"
REMOTE_VECTORS = HERE.parents[1] / "verity" / "core" / "protocols" / "remote_bandwidth" / "tests"
BOUND_BLOCKS = 6
ATTN_CYCLE_BYTES = 1 << 30  # a cell's layers cover this much plain KV, 8x the L2
ATTN_MAX_LAYERS = 256       # and at most this many, so a pass's launches queue behind one spin
ATTN_SPIN_CYCLES = 60_000_000
ATTN_CTAS_PER_SM = (1, 2, 4, 8)  # a cell's split counts: batch × splits about this many CTAs an SM,
ATTN_SPLIT_TILES = (1, 8)        # each split at least this many tiles (a split's partials are 33 KB; fewer tiles a
                                 # split hide more of a tile's decode latency at small batch)
ATTN_RTOL = 1e-3
ATTN_FORMS = {1: "phased", 2: "ws"}  # kv_attn's encoded forms: decode and attention in turn, or in their own warps
NATIVE_70B_S = 0.0457  # bench.py's tuned FP8 linears step at 70B, batch 1: x's denominator
CLOCK_LOCKED_MHZ, CLOCK_MAX_MHZ = 2092, 2430  # the nodes' locked SM clock (their clock-locking unit) and the card's max
RATES_SCHEMA = "verity/codec-frontier-rates/v0"


class Lib:
    def __init__(self, name: str, path: str) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.name, self.path, self.so = name, path, ctypes.CDLL(str(Path(path).resolve()))
        so = self.so
        so.kv_set_key.argtypes = [v, i, v]
        so.kv_encode.argtypes = [v, v, q, q, i, i, v]
        so.kv_decode.argtypes = [v, v, q, q, i, i, v, v]
        so.kv_encode_bound.argtypes = [v, v, q, q, i, v, i, v]
        so.kv_decode_bound.argtypes = [v, v, q, q, i, v, i, v]
        so.kv_read.argtypes = [v, q, v, i, v]
        so.kv_chain.argtypes = [i, i, i, q, i, v, v]
        so.kv_spin.argtypes = [ctypes.c_longlong, v]
        so.kv_attn_shape.argtypes = [v]
        so.kv_attn.argtypes = [v, i, v, v, v, i, i, i, i, i, i, v]
        so.kv_karatsuba.restype = i
        self.field = K.field(name.split("-")[0])
        self.karatsuba = so.kv_karatsuba()
        self.mask = hasattr(so, "kv_mask") and so.kv_mask() != 0
        if self.mask:
            so.kv_set_mask.argtypes = [v]
        if so.kv_bits() != self.field.n:
            raise RuntimeError(f"{name}: the library is built for {so.kv_bits()} bits")
        self.kmax = so.kv_kmax()

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{self.name}: {fn} error {rc}")


def stream(torch) -> int:
    return torch.cuda.current_stream().cuda_stream


def ptr(t) -> int:
    return t.data_ptr()


def timed(torch, fn, reps: int, warm: int = 1) -> list[float]:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    return out


def queued(lib: Lib, torch, fn, cycles: int = SPIN_CYCLES) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lib.call("kv_spin", cycles, stream(torch))
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3


def lib_key(lib: Lib, k: int) -> K.Key:
    return K.key(lib.field.name, SEED, k, mask=lib.mask)


def set_key(lib: Lib, key: K.Key) -> None:
    if lib.mask != (key.mask is not None):
        raise ValueError(f"{lib.name}: a {'masked' if lib.mask else 'plain'} library takes a key of its kind")
    s = key.field.word_bytes
    c = ctypes.create_string_buffer(b"".join(v.to_bytes(s, "little") for v in key.c), s * key.rounds)
    t = ctypes.create_string_buffer(key.tau.to_bytes(s, "little"), s)
    lib.call("kv_set_key", ctypes.cast(c, ctypes.c_void_p), key.rounds, ctypes.cast(t, ctypes.c_void_p))
    if lib.mask:
        m = ctypes.create_string_buffer(key.mask.to_bytes(s, "little"), s)
        lib.call("kv_set_mask", ctypes.cast(m, ctypes.c_void_p))


def to_device(torch, words: list[int], f: K.Field):
    raw = bytearray(b"".join(w.to_bytes(f.word_bytes, "little") for w in words))
    return torch.frombuffer(raw, dtype=torch.uint8).cuda()


def from_device(t, f: K.Field) -> list[int]:
    raw = t.cpu().numpy().tobytes()
    s = f.word_bytes
    return [int.from_bytes(raw[s * i : s * (i + 1)], "little") for i in range(len(raw) // s)]


def random_words(torch, n: int, f: K.Field, seed: int):
    """n canonical words (the top byte's top bit clear), as bytes on the device."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randint(0, 256, (n, f.word_bytes), dtype=torch.uint8, device="cuda", generator=g)
    w[:, -1] &= 0x7F
    return w.view(-1)


def xor32(torch, t) -> int:
    """The XOR of t's 32-bit limbs, the decode's checksum."""
    v = t.view(torch.int32)
    while v.numel() > 1:
        if v.numel() % 2:
            v = torch.cat([v, torch.zeros(1, dtype=v.dtype, device=v.device)])
        v = torch.bitwise_xor(v[0::2], v[1::2])
    return int(v.item()) & 0xFFFFFFFF


# --- gates -------------------------------------------------------------------------------------------------------------

def gate(lib: Lib, torch, ks, n_ref: int, n_words: int, log) -> dict:
    f = lib.field
    rng = random.Random(f.n)
    edges = [0, 1, 2, 3, f.p - 1, f.p - 2, (f.p - 1) // 2, (f.p + 1) // 2, 1 << (f.n - 1), (1 << 32) - 1, f.c, f.p - f.c]
    payloads = edges + [rng.randrange(f.p) for _ in range(n_ref - len(edges))]
    m = to_device(torch, payloads, f)
    y = torch.empty_like(m)
    back = torch.empty_like(m)
    big = random_words(torch, n_words, f, 5)
    big_y, big_back = torch.empty_like(big), torch.empty_like(big)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    want_xor = xor32(torch, big)
    out = {}
    st = stream(torch)
    for k in ks:
        key = lib_key(lib, k)
        set_key(lib, key)
        bad_enc, bad_dec = [], []
        for base in GATE_BASES:
            lib.call("kv_encode", ptr(m), ptr(y), len(payloads), base, k, 0, st)
            ref = [K.encode_word(v, base + i, key) for i, v in enumerate(payloads)]
            ref_d = to_device(torch, ref, f)
            lib.call("kv_decode", ptr(ref_d), ptr(back), len(payloads), base, k, 0, 0, st)
            torch.cuda.synchronize()
            bad_enc += [base + i for i, (a, b) in enumerate(zip(from_device(y, f), ref)) if a != b]
            bad_dec += [base + i for i, (a, b) in enumerate(zip(from_device(back, f), payloads)) if a != b]
        lib.call("kv_encode", ptr(big), ptr(big_y), n_words, 0, k, 0, st)
        lib.call("kv_decode", ptr(big_y), ptr(big_back), n_words, 0, k, 0, 0, st)
        xr.zero_()
        lib.call("kv_decode", ptr(big_y), 0, n_words, 0, k, 0, ptr(xr), st)
        torch.cuda.synchronize()
        round_trip = bool(torch.equal(big_back, big))
        checksum = (int(xr.item()) & 0xFFFFFFFF) == want_xor
        ok = not bad_enc and not bad_dec and round_trip and checksum
        out[k] = {"checked_against_reference": len(payloads) * len(GATE_BASES), "bases": list(GATE_BASES),
                  "round_trip_words": n_words, "encode_mismatch": bad_enc[:8], "decode_mismatch": bad_dec[:8],
                  "round_trip": round_trip, "checksum": checksum, "ok": ok}
        log(f"gate {lib.name} k={k}: {'ok' if ok else 'FAIL'} ({len(payloads)} words against kv.py at each of indices "
            f"{', '.join(map(str, GATE_BASES))} on; {n_words} round trip)")
    return out


def offsets(torch, values: list[int], f: K.Field):
    """Bound offsets as `kv_encode_bound` takes them: one a token-layer, NL little-endian limbs, on the device."""
    return to_device(torch, values, f)


def gate_bound(lib: Lib, torch, log) -> dict:
    """`kv_encode_bound` and `kv_decode_bound` at each key of `kv_vectors.json`'s "bound" the library takes (its field and
    mask): the pinned token-layer encoded under its bound key equals the pinned words (their SHA-512 and first word) and
    decodes back, and not under another digest's offset; encoded at another block's indices with the index difference
    folded into its offset (tau W (block − there)) it gives the same words, as a page written at a slot that is not its
    store index does; and BOUND_BLOCKS token-layers across index 2^32, a random offset each, equal `kv.py` word for word
    and decode back."""
    pinned = json.loads(BOUND_VECTORS.read_text())
    seed, bv = bytes.fromhex(pinned["seed"]), pinned["bound"]
    f, w, st = lib.field, K.token_layer_words(lib.field), stream(torch)
    data = b"".join(hashlib.sha512(bv["label"].encode() + c.to_bytes(4, "big")).digest() for c in range(64))
    d, block = K.digest(data), bv["block"]
    out = {}
    for name, want in bv["keys"].items():
        field_name, rounds, *masked = name.split("/")
        k, mask = int(rounds[1:]), bool(masked)
        if field_name != f.name or mask != lib.mask or k > lib.kmax:
            continue
        key = K.key(f.name, seed, k, mask=mask)
        set_key(lib, key)
        b = K.binding(d, block, key)
        pay = to_device(torch, K.payloads(data, f), f)
        enc, back, other, moved = (torch.empty_like(pay) for _ in range(4))
        ob = offsets(torch, [b], f)
        lib.call("kv_encode_bound", ptr(pay), ptr(enc), w, block * w, k, ptr(ob), 0, st)
        lib.call("kv_decode_bound", ptr(enc), ptr(back), w, block * w, k, ptr(ob), 0, st)
        wrong = offsets(torch, [K.binding(K.digest(bytes(K.TOKEN_LAYER_BYTES)), block, key)], f)
        lib.call("kv_decode_bound", ptr(enc), ptr(other), w, block * w, k, ptr(wrong), 0, st)
        there = block + 1000
        folded = offsets(torch, [(b + key.tau * w * (block - there)) % f.p], f)
        lib.call("kv_encode_bound", ptr(pay), ptr(moved), w, there * w, k, ptr(folded), 0, st)
        rng = random.Random(f"{name}/bound")
        first = (1 << 32) // w - BOUND_BLOCKS // 2
        offs = [rng.randrange(f.p) for _ in range(BOUND_BLOCKS)]
        many = [rng.randrange(f.p) for _ in range(BOUND_BLOCKS * w)]
        mp, mo = to_device(torch, many, f), offsets(torch, offs, f)
        me, mb = torch.empty_like(mp), torch.empty_like(mp)
        lib.call("kv_encode_bound", ptr(mp), ptr(me), len(many), first * w, k, ptr(mo), 0, st)
        lib.call("kv_decode_bound", ptr(me), ptr(mb), len(many), first * w, k, ptr(mo), 0, st)
        torch.cuda.synchronize()
        words = from_device(enc, f)
        raw = enc.cpu().numpy().tobytes()
        ref = [K.encode_word(m, first * w + i, key, offs[i // w]) for i, m in enumerate(many)]
        r = {"b": format(b, "x") == want["b"], "words_sha512": hashlib.sha512(raw).hexdigest() == want["words_sha512"],
             "word_0": format(words[0], "x") == want["word_0"], "decode": bool(torch.equal(back, pay)),
             "other_digest_refused": not torch.equal(other, pay), "folded_index": bool(torch.equal(moved, enc)),
             "blocks": BOUND_BLOCKS, "first_index": first * w,
             "kv_py": from_device(me, f) == ref, "round_trip": bool(torch.equal(mb, mp))}
        r["ok"] = all(v for v in r.values() if type(v) is bool)
        out[name] = r
        log(f"bound {lib.name} {name}: {'ok' if r['ok'] else 'FAIL ' + str(r)} (block {block} against kv_vectors.json, "
            f"at block {there} folded; {BOUND_BLOCKS} blocks from index {first * w} against kv.py)")
    if not out:
        log(f"bound {lib.name}: kv_vectors.json pins no bound key of {f.name}{'/masked' if lib.mask else ''}")
    if (coin := gate_coin(lib, torch, log)) is not None:
        out["coin"] = coin
    return out


def gate_coin(lib: Lib, torch, log) -> dict | None:
    """The append coin's form of the bound key (remote_bandwidth's PROTOCOL.md §3), for a library at the remote
    audit's codec (`codec.DEFAULT`; None for another): `vectors.json`'s `codec.coin` blocks, each encoded by
    `kv_encode_bound` at its own indices under b_j(c_a), equal the pinned words (their SHA-512 and first word), decode
    back, and don't decode under the form without the coin."""
    for p in (str(HERE.parents[1]), str(REMOTE_VECTORS)):
        if p not in sys.path:
            sys.path.append(p)
    import remote_bandwidth_vectors as RV
    from verity.core.protocols.remote_bandwidth import codec as RC

    rc, f = RC.DEFAULT, lib.field
    if (f.name, lib.mask) != (rc.field.name, rc.masked) or rc.rounds > lib.kmax:
        return None
    pinned = json.loads(RV.PATH.read_text())["codec"]
    seed, coin, k = bytes.fromhex(pinned["seed"]), bytes.fromhex(pinned["coin"]["c_a"]), rc.rounds
    key, dev = RC.key(seed, rc), K.key(f.name, seed, k, mask=rc.masked)
    if (dev.c, dev.tau, dev.mask) != (key.c, key.tau, key.mask):
        raise RuntimeError("kv.py's key is not the remote audit's codec key")
    set_key(lib, dev)
    w, st = K.token_layer_words(f), stream(torch)
    blocks = []
    for v in pinned["coin"]["bound"]:
        j = v["j"]
        pay = RV.payload(j)
        d = RC.digest(pay)
        b = RC.binding(d, j, key, coin)
        m = to_device(torch, K.payloads(pay, f), f)
        enc, back, other = (torch.empty_like(m) for _ in range(3))
        ob, pre = offsets(torch, [b], f), offsets(torch, [RC.binding(d, j, key)], f)
        lib.call("kv_encode_bound", ptr(m), ptr(enc), w, j * w, k, ptr(ob), 0, st)
        lib.call("kv_decode_bound", ptr(enc), ptr(back), w, j * w, k, ptr(ob), 0, st)
        lib.call("kv_decode_bound", ptr(enc), ptr(other), w, j * w, k, ptr(pre), 0, st)
        torch.cuda.synchronize()
        raw = enc.cpu().numpy().tobytes()
        r = {"j": j, "digest": d.hex() == v["digest"], "b": format(b, "x") == v["b"],
             "word_0": raw[:f.word_bytes].hex() == v["word_0"],
             "block_sha512": hashlib.sha512(raw).hexdigest() == v["block_sha512"],
             "decode": bool(torch.equal(back, m)), "pre_coin_refused": not torch.equal(other, m)}
        r["ok"] = all(x for x in r.values() if type(x) is bool)
        blocks.append(r)
    ok = all(r["ok"] for r in blocks)
    log(f"bound {lib.name} coin: {'ok' if ok else 'FAIL ' + str(blocks)} ({len(blocks)} blocks against vectors.json's "
        "codec.coin, and not under the form without the coin)")
    return {"codec": rc.name, "c_a": coin.hex(), "blocks": blocks, "ok": ok}


# --- rates -------------------------------------------------------------------------------------------------------------

def read_rate(lib: Lib, torch, store, sms: int, reps: int) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    best = None
    for mult in (4, 8, 16, 32):
        ts = timed(torch, lambda: lib.call("kv_read", ptr(store), store.numel(), ptr(xr), sms * mult, stream(torch)), reps)
        med = statistics.median(ts)
        if best is None or med < best[1]:
            best = (mult, med, ts)
    return {"bytes": store.numel(), "grid_per_sm": best[0], "median_s": best[1], "times_s": best[2],
            "gbps": store.numel() / best[1] / 1e9}


def chain(lib: Lib, torch, what: int, k: int, ilp: int, sms: int, reps: int, target_s: float = 0.05) -> dict:
    """The best over blocks an SM of `what`'s compute-only rate, items/s, steps calibrated to about target_s."""
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    st = stream(torch)
    best = None
    for per_sm in (2, 4, 8):
        grid = sms * per_sm
        steps = 4
        while True:
            t = statistics.median(timed(torch, lambda: lib.call("kv_chain", what, k, ilp, steps, grid, ptr(xr), st), 1))
            if t > target_s / 4 or steps > 1 << 24:
                break
            steps *= 4
        steps = max(1, int(steps * target_s / t))
        ts = timed(torch, lambda: lib.call("kv_chain", what, k, ilp, steps, grid, ptr(xr), st), reps)
        med = statistics.median(ts)
        items = grid * 128 * steps * (ilp if what == 0 else 1)
        r = {"per_sm": per_sm, "steps": steps, "items": items, "median_s": med, "times_s": ts, "per_s": items / med}
        if best is None or r["per_s"] > best["per_s"]:
            best = r
    return best


def rates(lib: Lib, torch, ks, sms: int, reps: int, log) -> dict:
    set_key(lib, lib_key(lib, lib.kmax))
    sq = {ilp: chain(lib, torch, 0, 1, ilp, sms, reps) for ilp in (1, 2)}
    out = {"square": sq, "square_s": max(r["per_s"] for r in sq.values()), "decode": {}, "encode": {}}
    for k in ks:
        out["decode"][k] = chain(lib, torch, 1, k, 1, sms, reps)
        out["encode"][k] = chain(lib, torch, 2, k, 1, sms, reps)
    f = lib.field
    floor = out["square_s"] / (f.root_squarings + 1)
    out["encode_floor_words_s"] = floor
    log(f"rates {lib.name}: squarings/s " + ", ".join(f"ILP {i} {r['per_s']:.3e}" for i, r in sq.items())
        + f"; encode floor {floor:.3e} words/s ({f.root_squarings + 1} squarings a root)")
    for k in ks:
        d, e = out["decode"][k], out["encode"][k]
        log(f"rates {lib.name} k={k}: decode {d['per_s']:.3e} words/s ({d['per_s'] * f.word_bytes / 1e9:.0f} GB/s), "
            f"encode {e['per_s']:.3e} words/s ({e['per_s'] * f.word_bytes / 1e9:.2f} GB/s, "
            f"{e['per_s'] * k / floor:.0%} of the floor), r = {d['per_s'] / e['per_s']:.0f}")
    return out


def frontier_rates(res: dict, libs, run: str | None) -> list[dict]:
    """Each field's compute-only rates at each k as `RATES_SCHEMA` records, for the frontier's pooled x: x is the encode
    of 70B's linears (`E_70B` bytes as words) at the measured rate, in `NATIVE_70B_S` steps, at the locked clock;
    `x_max_clock` scales it by the clocks' ratio (not measured: the nodes' clocks are locked)."""
    out = []
    hz = CLOCK_LOCKED_MHZ * 1e6
    for lib in libs:
        f, r = lib.field, res["libs"][lib.name].get("rates")
        if not r:
            continue
        words = E_70B / f.word_bytes
        for k, e in r["encode"].items():
            d = r["decode"][k]
            x = words / e["per_s"] / NATIVE_70B_S
            out.append({"schema": RATES_SCHEMA, "lane": "ra-word-codec", "b": f.n, "field": f"2^{f.n}-{f.c}",
                        "lib": lib.name, "k": int(k),
                        "arithmetic": "cuda-int32/thread-per-word/column"
                        + ("/karatsuba-1" if lib.karatsuba else "/streamed-fold" if f.limbs > 24 else "")
                        + ("/mask-xor" if lib.mask else ""),
                        "squarings_per_s": r["square_s"], "sm_clocks_per_squaring": res["sms"] * hz / r["square_s"],
                        "roots_per_s": e["per_s"] * int(k), "encode_words_per_s": e["per_s"],
                        "root_over_squaring_floor": e["per_s"] * int(k) / r["encode_floor_words_s"],
                        "decode_words_per_s": d["per_s"], "r": d["per_s"] / e["per_s"],
                        "x": x, "x_boosted": False, "x_max_clock": x * CLOCK_LOCKED_MHZ / CLOCK_MAX_MHZ,
                        "delta": words / d["per_s"] / NATIVE_70B_S, "clock_mhz": CLOCK_LOCKED_MHZ,
                        "clock_max_mhz": CLOCK_MAX_MHZ, "max_clock_measured": False, "sms": res["sms"],
                        "measured": True, "run": run, "art": None})
    return out


def store_rates(lib: Lib, torch, store, ks, sms: int, reps: int, read_gbps: float, log) -> dict:
    f = lib.field
    n = store.numel() // f.word_bytes
    want = xor32(torch, store)
    y = torch.empty_like(store)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    st = stream(torch)
    out = {}
    for k in ks:
        set_key(lib, lib_key(lib, k))
        enc = timed(torch, lambda: lib.call("kv_encode", ptr(store), ptr(y), n, 0, k, sms * 8, st), 1, warm=0)[0]
        xr.zero_()
        lib.call("kv_decode", ptr(y), 0, n, 0, k, sms * 8, ptr(xr), st)
        torch.cuda.synchronize()
        if (int(xr.item()) & 0xFFFFFFFF) != want:
            out[k] = {"ok": False}
            log(f"store {lib.name} k={k}: checksum FAIL")
            continue
        best = None
        for mult in (4, 8, 16):
            ts = timed(torch, lambda: lib.call("kv_decode", ptr(y), 0, n, 0, k, sms * mult, ptr(xr), st), reps)
            med = statistics.median(ts)
            if best is None or med < best[1]:
                best = (mult, med, ts)
        gbps = store.numel() / best[1] / 1e9
        out[k] = {"ok": True, "store_bytes": store.numel(), "encode_s": enc, "encode_words_s": n / enc,
                  "encode_gbps": store.numel() / enc / 1e9, "decode_grid_per_sm": best[0], "decode_s": best[1],
                  "decode_times_s": best[2], "decode_gbps": gbps, "decode_over_read": read_gbps / gbps}
        log(f"store {lib.name} k={k}: encode {n / enc:.3e} words/s ({store.numel() / enc / 1e9:.2f} GB/s), "
            f"read-and-decode {gbps:.0f} GB/s ({read_gbps / gbps:.2f}x the read)")
    del y
    return out


def token_encode(lib: Lib, torch, ks, batches, reps: int, enc_rates: dict, log) -> dict:
    f = lib.field
    tw = K.token_words(f)
    words = random_words(torch, tw * max(batches), f, 9)
    y = torch.empty_like(words)
    st = stream(torch)
    out = {}
    for k in ks:
        set_key(lib, lib_key(lib, k))
        out[k] = {}
        for b in batches:
            n = tw * b
            lib.call("kv_encode", ptr(words), ptr(y), n, 0, k, 0, st)
            ts = [queued(lib, torch, lambda: lib.call("kv_encode", ptr(words), ptr(y), n, 0, k, 0, st))
                  for _ in range(reps)]
            sm = n / enc_rates[k]["per_s"]
            out[k][b] = {"words": n, "bytes": n * f.word_bytes, "latency_s": statistics.median(ts), "times_s": ts,
                         "sm_s": sm}
        log(f"token {lib.name} k={k}: " + ", ".join(
            f"B={b} {r['latency_s'] * 1e6:.0f} us latency, {r['sm_s'] * 1e6:.0f} us at the compute-only rate"
            for b, r in out[k].items()))
    del words, y
    return out


def attn_shape(lib: Lib) -> dict:
    a = (ctypes.c_int * 10)()
    lib.so.kv_attn_shape(ctypes.cast(a, ctypes.c_void_p))
    return dict(zip(("heads", "kv_heads", "head_dim", "tlw", "tle", "tt", "part", "mw", "mma", "mt"), a))


def attention(lib: Lib, torch, batches, contexts, k: int, reps: int, sms: int, log) -> dict:
    """`kv_attn` at Llama-3.1-70B's attention, one layer of batch × context, over the plain cache and over its words at
    k rounds. Each cell's cache holds L layers, as many as cover ATTN_CYCLE_BYTES of plain KV (at most ATTN_MAX_LAYERS),
    and a timed pass runs every layer once, queued behind a spin, so a layer's time is the pass's over L with the L2
    cold as in a decode step. Nothing is timed before the gate: at layers 0 and L - 1 each encoded form's output
    (ATTN_FORMS) equals the plain form's bit for bit, and the plain form's is within ATTN_RTOL of float64 attention
    over the same bf16 values, at every split count the cell tries (batch × splits about each of ATTN_CTAS_PER_SM CTAs
    an SM, at least each of ATTN_SPLIT_TILES tiles a split); each form's time is its best split count's, and the
    encoded time the best form's. torch's SDPA (GQA, bf16) over the same K and V, laid out as it takes them, is timed
    beside it the same way."""
    import torch.nn.functional as F

    sh = attn_shape(lib)
    if (sh["heads"], sh["kv_heads"], sh["head_dim"]) != (64, K.KV_HEADS, K.HEAD_DIM) or sh["tlw"] != K.token_layer_words(lib.field):
        raise RuntimeError(f"{lib.name}: kv_attn_shape {sh} is not kv.py's")
    set_key(lib, lib_key(lib, k))
    st = stream(torch)
    tl, tle = K.TOKEN_LAYER_BYTES, sh["tle"]
    scale = math.log2(math.e) / math.sqrt(K.HEAD_DIM)
    out = {}
    for b in batches:
        for ctx in contexts:
            cands = sorted({max(1, min(-(-ctx // (t * sh["tt"])), -(-c * sms // b)))
                            for c in ATTN_CTAS_PER_SM for t in ATTN_SPLIT_TILES})
            per_layer = b * ctx * tl
            layers = max(2, min(ATTN_MAX_LAYERS, -(-ATTN_CYCLE_BYTES // per_layer)))
            g = torch.Generator(device="cuda").manual_seed(b * 100003 + ctx)
            plain = torch.randn(b * ctx * layers * tl // 2, dtype=torch.float32, device="cuda", generator=g).to(
                torch.bfloat16)
            pad = torch.zeros(b * ctx * layers, tle, dtype=torch.uint8, device="cuda")
            pad[:, :tl] = plain.view(torch.uint8).view(-1, tl)
            enc = torch.empty_like(pad)
            n_words = pad.numel() // lib.field.word_bytes
            lib.call("kv_encode", ptr(pad), ptr(enc), n_words, 0, k, sms * 8, st)
            del pad
            q = (torch.randn(b, 64, K.HEAD_DIM, device="cuda", generator=g) * scale).to(torch.bfloat16).contiguous()
            part = torch.empty(b * cands[-1] * sh["part"], dtype=torch.float32, device="cuda")
            o_p = torch.empty(b, 64, K.HEAD_DIM, dtype=torch.float32, device="cuda")
            o_e = torch.empty_like(o_p)

            def run(store, encoded: int, o, layer: int, splits: int) -> None:
                lib.call("kv_attn", ptr(store), encoded, ptr(q), ptr(o), ptr(part), b, ctx, layers, layer, splits, k, st)

            kvl = plain.view(b, ctx, layers, 2, K.KV_HEADS, K.HEAD_DIM)
            gate = {"layers": layers, "splits": cands, "equal": True, "rel_err": 0.0}
            for layer in (0, layers - 1):
                kk, vv = kvl[:, :, layer, 0].double(), kvl[:, :, layer, 1].double()
                qq = q.double().view(b, K.KV_HEADS, 64 // K.KV_HEADS, K.HEAD_DIM)
                s = torch.einsum("bgqd,btgd->bgqt", qq, kk)
                p = torch.exp2(s - s.amax(-1, keepdim=True))
                ref = torch.einsum("bgqt,btgd->bgqd", p / p.sum(-1, keepdim=True), vv).reshape(b, 64, K.HEAD_DIM)
                del kk, vv, s, p
                for sp in cands:
                    run(plain, 0, o_p, layer, sp)
                    torch.cuda.synchronize()
                    err = float((o_p.double() - ref).abs().max() / ref.abs().max())
                    gate["rel_err"] = max(gate["rel_err"], err)
                    for form in ATTN_FORMS:
                        o_e.zero_()
                        run(enc, form, o_e, layer, sp)
                        torch.cuda.synchronize()
                        gate["equal"] &= bool(torch.equal(o_p, o_e))
                del ref
            gate["ok"] = gate["equal"] and gate["rel_err"] <= ATTN_RTOL
            cell = {"batch": b, "context": ctx, "layers": layers, "plain_bytes": per_layer,
                    "encoded_bytes": b * ctx * tle, "gate": gate}
            if gate["ok"]:
                spin = ATTN_SPIN_CYCLES
                per = {}
                for sp in cands:
                    tp = [queued(lib, torch, lambda: [run(plain, 0, o_p, l, sp) for l in range(layers)], spin) / layers
                          for _ in range(reps)]
                    per[sp] = {"plain_s": statistics.median(tp), "plain_times_s": tp}
                    for form, fname in ATTN_FORMS.items():
                        te = [queued(lib, torch, lambda: [run(enc, form, o_e, l, sp) for l in range(layers)], spin)
                              / layers for _ in range(reps)]
                        per[sp].update({f"{fname}_s": statistics.median(te), f"{fname}_times_s": te})
                kt = kvl.permute(2, 3, 0, 4, 1, 5).contiguous()  # layer, K/V, b, kv head, token, dim
                qt = (q.float() / scale).to(torch.bfloat16).view(b, 64, 1, K.HEAD_DIM)
                F.scaled_dot_product_attention(qt, kt[0, 0], kt[0, 1], enable_gqa=True)
                tt = [queued(lib, torch, lambda: [F.scaled_dot_product_attention(qt, kt[l, 0], kt[l, 1], enable_gqa=True)
                                                  for l in range(layers)], spin) / layers for _ in range(reps)]
                del kt
                sp_p = min(per, key=lambda sp: per[sp]["plain_s"])
                best = {f: min(per, key=lambda sp: per[sp][f"{f}_s"]) for f in ATTN_FORMS.values()}
                form = min(best, key=lambda f: per[best[f]][f"{f}_s"])
                p_s, e_s, t_s = per[sp_p]["plain_s"], per[best[form]][f"{form}_s"], statistics.median(tt)
                cell.update({"by_splits": per, "plain_splits": sp_p, "encoded_form": form,
                             "encoded_splits": best[form], "plain_s": p_s, "encoded_s": e_s,
                             **{f"{f}_s": per[sp][f"{f}_s"] for f, sp in best.items()},
                             **{f"{f}_splits": sp for f, sp in best.items()},
                             "torch_sdpa_s": t_s, "torch_sdpa_times_s": tt,
                             "plain_gbps": per_layer / p_s / 1e9, "encoded_gbps": b * ctx * tle / e_s / 1e9,
                             "encoded_over_plain": e_s / p_s, "plain_over_torch": p_s / t_s})
                log(f"attention {lib.name} k={k} B={b} ctx={ctx}: gate ok (equal, rel err {gate['rel_err']:.1e}, "
                    f"{layers} layers, splits {cands}); a layer plain {p_s * 1e6:.1f} us at {sp_p} splits "
                    f"({cell['plain_gbps']:.0f} GB/s), encoded " + ", ".join(
                        f"{f} {per[sp][f'{f}_s'] * 1e6:.1f} us at {sp}" for f, sp in best.items())
                    + f" ({cell['encoded_gbps']:.0f} GB/s stored at the best), {e_s / p_s:.3f}x; torch SDPA "
                    f"{t_s * 1e6:.1f} us (plain {p_s / t_s:.2f}x torch)")
            else:
                log(f"attention {lib.name} k={k} B={b} ctx={ctx}: gate FAIL {gate}")
            out[f"b{b}_ctx{ctx}"] = cell
            del plain, enc, q, part, o_p, o_e, kvl
            torch.cuda.empty_cache()
    return out


def native_70b(torch, batches, reps: int, log) -> dict:
    out = {}
    one = torch.ones((), device="cuda")
    for name, (n, kd) in SHAPES_70B.items():
        g = torch.Generator(device="cuda").manual_seed(n + kd)
        w = (torch.randn(n, kd, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
        copies = max(2, -(-L2_DEFEAT_BYTES // (n * kd)))
        wts = [c.t() for c in [w] + [w.clone() for _ in range(copies - 1)]]
        for b in batches:
            mpad = 16 * -(-b // 16)
            x = (torch.randn(b, kd, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
            try:
                torch._scaled_mm(x, wts[0], one, one, out_dtype=torch.bfloat16)
                xc, mpad = x, b
            except Exception:  # noqa: BLE001  cuBLASLt's FP8 path needs M a multiple of 16 here
                xc = torch.zeros(mpad, kd, device="cuda").to(torch.float8_e4m3fn)
                xc[:b] = x

            def arm():
                for wt in wts:
                    torch._scaled_mm(xc, wt, one, one, out_dtype=torch.bfloat16)
            ts = timed(torch, arm, reps)
            out.setdefault(b, {})[name] = {"m_padded": mpad, "copies": copies, "call_s": statistics.median(ts) / copies,
                                           "gbps": n * kd * copies / statistics.median(ts) / 1e9}
        del w, wts
        torch.cuda.empty_cache()
    for b in batches:
        step = K.LAYERS * sum(out[b][s]["call_s"] for s in SHAPES_70B)
        out[b]["step_s"] = step
        out[b]["gbps"] = E_70B / step / 1e9
        log(f"native 70B B={b}: the linears' step {step * 1e3:.2f} ms ({E_70B / step / 1e9:.0f} GB/s); "
            + ", ".join(f"{s} {out[b][s]['call_s'] * 1e6:.0f} us" for s in SHAPES_70B))
    return out


def model(res: dict, libs: list[Lib], batches, contexts, capacity: int) -> dict:
    """The KV cache's codec in passes of E at Llama-3.1-70B: the step's new tokens' encode, and the decode inside
    attention, for batch × context. A configuration whose E + KV exceeds the card's memory is marked."""
    r_read = res["read"]["gbps"] * 1e9
    e_pass = E_70B / r_read
    out = {"E_bytes": E_70B, "E_pass_s": e_pass, "capacity_bytes": capacity}
    for lib in libs:
        f = lib.field
        lr = res["libs"][lib.name]
        stored_per_byte = K.token_layer_words(f) * f.word_bytes / K.TOKEN_LAYER_BYTES
        rows = {}
        for k, enc in lr["rates"]["encode"].items():
            dec_c = lr["rates"]["decode"][k]["per_s"] * f.word_bytes
            dec_m = lr["store"][k]["decode_gbps"] * 1e9 if lr["store"].get(k, {}).get("ok") else None
            for b in batches:
                enc_pass = b * K.token_words(f) / enc["per_s"] / e_pass
                lat = lr["token"][k][b]["latency_s"] if b in lr["token"].get(k, {}) else None
                for ctx in contexts:
                    kv = b * ctx * K.TOKEN_BYTES * stored_per_byte
                    read = kv / r_read / e_pass
                    rows[f"k{k}_b{b}_ctx{ctx}"] = {
                        "k": k, "batch": b, "context": ctx, "kv_bytes": kv, "fits": E_70B + kv <= capacity,
                        "kv_share": kv / (E_70B + kv), "encode_passes": enc_pass,
                        "encode_latency_s": lat, "kv_read_passes": read,
                        "decode_measured_passes": None if dec_m is None else kv / dec_m / e_pass - read,
                        "decode_overlap_passes": max(0.0, kv / dec_c / e_pass - read),
                        "decode_additive_passes": kv / dec_c / e_pass}
                    cell = lr.get("attention", {}).get(k, {}).get(f"b{b}_ctx{ctx}", {})
                    if "encoded_s" in cell:
                        rows[f"k{k}_b{b}_ctx{ctx}"].update({
                            "attn_plain_passes": K.LAYERS * cell["plain_s"] / e_pass,
                            "attn_torch_passes": K.LAYERS * cell["torch_sdpa_s"] / e_pass,
                            "attn_decode_passes": K.LAYERS * (cell["encoded_s"] - cell["plain_s"]) / e_pass,
                            "attn_encoded_over_plain": cell["encoded_over_plain"]})
        out[lib.name] = {"stored_per_payload_byte": stored_per_byte, "rows": rows}
    return out


def write_result(res: dict, libs: list[Lib], out_dir: Path) -> None:
    """`research/result/v0.1` in $RESEARCH_RUN_DIR (`tools/research/src/research/result.py`), when the run has one."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = [{"name": "read", "value": res["read"]["gbps"], "unit": "GB/s"}] if "read" in res else []
    for b, nat in res.get("native", {}).items():
        ms.append({"name": f"native70b_m{b}_step", "value": nat["step_s"], "unit": "s"})
    for lib in libs:
        lr = res["libs"].get(lib.name, {})
        if "rates" not in lr:
            continue
        ms.append({"name": f"{lib.name}_square", "value": lr["rates"]["square_s"], "unit": "1/s"})
        for k in lr["rates"]["encode"]:
            ms.append({"name": f"{lib.name}_k{k}_encode", "value": lr["rates"]["encode"][k]["per_s"], "unit": "words/s"})
            ms.append({"name": f"{lib.name}_k{k}_decode", "value": lr["rates"]["decode"][k]["per_s"], "unit": "words/s"})
            if lr.get("store", {}).get(k, {}).get("ok"):
                ms.append({"name": f"{lib.name}_k{k}_read_decode", "value": lr["store"][k]["decode_gbps"],
                           "unit": "GB/s"})
            for b, t in lr.get("token", {}).get(k, {}).items():
                ms.append({"name": f"{lib.name}_k{k}_b{b}_token_latency", "value": t["latency_s"], "unit": "s"})
                ms.append({"name": f"{lib.name}_k{k}_b{b}_token_sm", "value": t["sm_s"], "unit": "s"})
        for k, cells in lr.get("attention", {}).items():
            for name, c in cells.items():
                if "encoded_s" in c:
                    ms.append({"name": f"{lib.name}_k{k}_{name}_attn_plain", "value": c["plain_s"], "unit": "s"})
                    ms.append({"name": f"{lib.name}_k{k}_{name}_attn_encoded", "value": c["encoded_s"], "unit": "s"})
                    ms.append({"name": f"{lib.name}_k{k}_{name}_attn_torch", "value": c["torch_sdpa_s"], "unit": "s"})
    rel = os.path.relpath(out_dir, run_dir)
    files = [f"{rel}/kv-codec.json", f"{rel}/kv.log"]
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {l.name: hashlib.sha256(Path(l.path).read_bytes()).hexdigest() for l in libs},
                                    "seed": SEED.hex()},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "kv codec gates",
                          "detail": "device encode and decode = kv.py, round trips, checksums", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="NAME=PATH, NAME a field of kv.FIELDS (or FIELD-TAG)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="1,2,4,6")
    ap.add_argument("--attn-ks", default="1", help="the rounds `attention` decodes inside attention at (empty: none)")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)), help="the contexts `attention` and `model` take")
    ap.add_argument("--store-gib", type=float, default=2.0)
    ap.add_argument("--gate-ref", type=int, default=48)
    ap.add_argument("--gate-words", type=int, default=1 << 16)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--no-native", action="store_true")
    ap.add_argument("--gates-only", action="store_true", help="`gate` and `gate_bound` only, nothing timed")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "kv.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    ks = [int(v) for v in a.ks.split(",")]
    attn_ks = [int(v) for v in a.attn_ks.split(",") if v]
    batches = [int(v) for v in a.batches.split(",")]
    contexts = [int(v) for v in a.contexts.split(",")]
    libs = [Lib(*spec.split("=", 1)) for spec in a.lib]
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    res = {"device": props.name, "sms": sms, "total_memory": props.total_memory, "seed": SEED.hex(), "ks": ks,
           "batches": batches, "E_70B": E_70B, "token_bytes": K.TOKEN_BYTES, "libs": {}, "ok": True}
    log(f"{props.name}: {sms} SMs; libs " + ", ".join(f"{l.name} ({l.field.n} bits, kmax {l.kmax})" for l in libs))
    for lib in libs:
        g = gate(lib, torch, ks, a.gate_ref, a.gate_words, log)
        gb = gate_bound(lib, torch, log)
        res["libs"][lib.name] = {"gate": g, "bound": gb}
        if not all(v["ok"] for v in g.values()) or not all(v["ok"] for v in gb.values()):
            res["ok"] = False
    if not res["ok"]:
        log("a gate failed: nothing timed")
    elif a.gates_only:
        log("gates only: nothing timed")
    else:
        rbuf = torch.empty(int(a.store_gib * (1 << 30)) // 64 * 64, dtype=torch.uint8, device="cuda")
        rbuf.random_(0, 256)
        res["read"] = read_rate(libs[0], torch, rbuf, sms, a.reps)
        log(f"read: {res['read']['gbps']:.0f} GB/s over {rbuf.numel() / 2**30:.1f} GiB")
        del rbuf
        torch.cuda.empty_cache()
        for lib in libs:
            lr = res["libs"][lib.name]
            lr["rates"] = rates(lib, torch, ks, sms, a.reps, log)
            n = int(a.store_gib * (1 << 30)) // lib.field.word_bytes
            store = random_words(torch, n, lib.field, 3)
            lr["store"] = store_rates(lib, torch, store, ks, sms, a.reps, res["read"]["gbps"], log)
            del store
            torch.cuda.empty_cache()
            lr["token"] = token_encode(lib, torch, ks, batches, a.reps, lr["rates"]["encode"], log)
            lr["attention"] = {k: attention(lib, torch, batches, contexts, k, a.reps, sms, log) for k in attn_ks}
            if not all(v["ok"] for v in lr["store"].values()) or not all(
                    c["gate"]["ok"] for cells in lr["attention"].values() for c in cells.values()):
                res["ok"] = False
        if not a.no_native:
            res["native"] = native_70b(torch, batches, a.reps, log)
        res["model"] = model(res, libs, batches, contexts, props.total_memory)
        res["frontier_rates"] = frontier_rates(res, libs, os.environ.get("RESEARCH_RUN_ID"))
        (out_dir / "rates.json").write_text(json.dumps(res["frontier_rates"], indent=1) + "\n")
    (out_dir / "kv-codec.json").write_text(json.dumps(res, indent=1, default=str))
    write_result(res, libs, out_dir)
    log("done" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
