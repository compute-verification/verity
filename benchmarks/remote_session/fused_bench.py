"""`fused.cu` on one GPU: kv.cu's decode inside a bf16 GEMM's weight load, gated against `kv.py` and `kv_decode`, then
timed beside the engine's scratch path (`engine.Hooks`: kv_decode into scratch, then the GEMM vLLM runs) and plain
bf16, at Qwen2.5-32B's linears and gate run 7's served decode batch (64). Libraries are `--lib NAME=PATH`: kv.so
builds (`kv.cu` at KV_FIELD=1024, KV_MASK=1, as run.sh builds it) and fused.so builds (`fused.cu`, the same plus
kv_gemm); a build with KV_FOLD6=1 squares by FOLD 6, the same values.

~~~text
gate    for each fused library and nt (64, 32), under one masked p1024 key of `--rounds`:
        - words: words whose decode takes kv.cu's rare branches (codeword, payload and inner-round edges, the
          square's rare folds, σ_K's slow test and its undo, sub_mod to zero), among random ones, at each of
          kv_bench.GATE_BASES and splits 1 to 3: kv_gemm_dump's tiles equal kv.py's decode and the library's
          kv_decode bit for bit, at every batch tile;
        - shapes: random bf16 weights at each linear's shape, kv_encode'd at GPU 7's arena indices in gate run 7's
          layout (o_proj's across 2^32): kv_decode and kv_gemm_dump give back the weights bit for bit;
        - gemm: kv_gemm against F.linear on the decoded weights (what the engine runs: kv_decode, then vLLM's GEMM)
          and against float64, at each M of GATE_MS: every output within FP32 accumulation rounding of float64,
          |y - y64| <= K 2^-24 sum|x w| + half a bf16 ulp, the same bound F.linear's outputs are held to; how many
          equal F.linear's bit for bit; kv_gemm run twice gives the same bits.
        A fused library that fails any of them is recorded and left untimed; nothing is timed (exit 4) unless
        the rare branches were all hit and some fused library passed.
time    a --arena-gib arena laid out as gate run 7's engine (`engine.pack`: embed, 64 layers' qkv, o, gate_up, down,
        lm_head) of random words; a step is the embedding, each layer's attention stand-in (a read of its share of
        the KV cache at gate run 7's load, KV_LAYER_BYTES) and four linears, then the LM head, at each M of --ms,
        for each arm of --arms:
          plain          F.linear over the arena's bytes as bf16 (plain serving)
          scratch:LIB    kv_decode of each weight into scratch, then F.linear (engine.Hooks: the embedding whole)
          fused:LIB:NT   kv_gemm (the embedding's M rows decoded alone)
          ARM*F          ARM's step with F times the layers (a step of another length, for the pass beside it)
        each op's and each step's GPU time (CUDA events), median over --reps steps; and each library's
        read-and-decode rate over the whole arena (kv_decode's checksum mode) beside the bare read (kv_read)
ncu     layer 0's linears and the LM head, each arm at each M, every op behind a one-cycle kv_spin marker, for
        Nsight Compute (fused.sh runs it: DRAM bytes, read plus write, and each kernel's duration, L2 flushed before
        each kernel); `ncu-report` folds its CSV into each op's bytes and time
serve   the engine stand-in for `fused_beside.py`: the arena exported over CUDA IPC (`engine.export`) to
        DIR/engine.json, with this process's pid, then a step at M 64 at each multiple of 1/--hz of the host clock (engine.Load's
        --align-steps), the arm DIR/ctl.json names (none: idle), each step logged to OUT/serve.steps.jsonl
~~~

    python fused_bench.py --mode gate,time --lib kv=kv.so --lib kv6=kv6.so --lib fused=fused.so \
        --lib fused6=fused6.so --out OUT
"""
from __future__ import annotations

import argparse
import csv
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
sys.path.insert(0, str(HERE.parent / "word_codec"))
sys.path.insert(0, str(HERE))

import kv as K  # noqa: E402
import kv_twin  # noqa: E402
from engine import BLOCK, GIB, aligned, export, pack, write_json  # noqa: E402
from kv_bench import GATE_BASES  # noqa: E402

FIELD = "p1024"
WORD = 128
WE = WORD // 2
HIDDEN, INTER, LAYERS, VOCAB = 5120, 27648, 64, 152064
QKV = (40 + 2 * 8) * 128
LINEARS = (("qkv", QKV, HIDDEN), ("o", HIDDEN, HIDDEN), ("gate_up", 2 * INTER, HIDDEN), ("down", HIDDEN, INTER))
HEAD = ("lm_head", VOCAB, HIDDEN)
SHARD_BLOCKS = 62 * GIB // BLOCK  # a GPU's arena in gate run 7's store, so GPU 7's first block is 7 of these
KV_LAYER_BYTES = 64 * 384 * 2 * 8 * 128 * 2  # gate run 7's load: 64 sequences at a mean 384 tokens, K and V of 8 heads
GATE_MS = (1, 3, 8, 32, 47, 64)
TILES = (1, 9, 17, 33)  # one M a batch tile (8, 16, 32, 64)
SEED = hashlib.sha256(b"verity/remote-session/fused/gate").digest()


class Lib:
    """kv.so's entry points, and fused.cu's in a fused build, with the key set."""

    def __init__(self, name: str, path: str) -> None:
        v, q, i = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        self.name, self.path = name, path
        so = self.so = ctypes.CDLL(str(Path(path).resolve()))
        so.kv_set_key.argtypes = [v, i, v]
        so.kv_set_mask.argtypes = [v]
        so.kv_encode.argtypes = [v, v, q, q, i, i, v]
        so.kv_decode.argtypes = [v, v, q, q, i, i, v, v]
        so.kv_read.argtypes = [v, q, v, i, v]
        so.kv_spin.argtypes = [ctypes.c_longlong, v]
        self.fused = hasattr(so, "kv_gemm")
        if self.fused:
            so.kv_gemm.argtypes = [v, q, v, v, v, v, q, i, i, i, i, i, i, v]
            so.kv_gemm_dump.argtypes = [v, q, v, i, i, i, i, i, i, v]
            so.kv_gemm_splits.argtypes = [i, i, i, i]
            so.kv_gemm_occupancy.argtypes = [i, i]
        if so.kv_bits() != 1024 or not so.kv_mask():
            raise RuntimeError(f"{name}: built for {so.kv_bits()} bits, mask {so.kv_mask()}, not p1024 masked")
        self.fold6 = bool(so.kv_gemm_fold6()) if self.fused else None

    def call(self, fn: str, *args) -> int:
        rc = getattr(self.so, fn)(*args)
        if rc < 0 or (rc and fn not in ("kv_gemm", "kv_gemm_dump", "kv_gemm_splits", "kv_gemm_occupancy")):
            raise RuntimeError(f"{self.name}: {fn} returned {rc}")
        return rc

    def set_key(self, key: K.Key) -> None:
        s = key.field.word_bytes
        c = ctypes.create_string_buffer(b"".join(v.to_bytes(s, "little") for v in key.c), s * key.rounds)
        t = ctypes.create_string_buffer(key.tau.to_bytes(s, "little"), s)
        m = ctypes.create_string_buffer(key.mask.to_bytes(s, "little"), s)
        self.call("kv_set_key", ctypes.cast(c, ctypes.c_void_p), key.rounds, ctypes.cast(t, ctypes.c_void_p))
        self.call("kv_set_mask", ctypes.cast(m, ctypes.c_void_p))


def stream(torch) -> int:
    return torch.cuda.current_stream().cuda_stream


# --- the reference's words ---------------------------------------------------------------------------------------------

def forward(e: int, j: int, i: int, key: K.Key) -> int:
    """The codeword at index i whose encode passes through state e before round j (j = rounds: e is the codeword)."""
    f, t, x = key.field, key.tweak(i), e
    for r in range(j, key.rounds):
        if r:
            x = key.sigma(x)
        x = K.rho((x + key.c[r] + t) % f.p, f)
    return x


def branches(y: int, i: int, key: K.Key) -> set[str]:
    """The rare branches kv.cu's decode of y at index i takes (as `kv.decode_word` computes it)."""
    f, out, x, t = key.field, set(), y, key.tweak(i)
    for j in reversed(range(key.rounds)):
        if x in (0, f.p - 1) or x >> (f.n - 32) == 0xFFFFFFFF:
            out.add("root_edge")
        q = K.rho_inv(x, f)
        if q < key.c[j]:
            out.add("sub_c_borrow")
        a = (q - key.c[j] - t) % f.p
        if a == 0:
            out.add("sub_zero")
        if j:
            v = a ^ key.mask
            if v >> (f.n - 32) == 0xFFFFFFFF:
                out.add("sigma_slow")
                if v >= f.p:
                    out.add("sigma_undo")
            x = v if v < f.p else a
        else:
            x = a
    return out


def crafted(key: K.Key, rng: random.Random) -> list[tuple[int, int]]:
    """(j, e): words whose decode takes a rare branch, each given by its encode's state e before round j."""
    f = key.field
    p, n, c, top = f.p, f.n, f.c, (1 << f.n) - 1
    edges = [0, 1, 2, 3, p - 1, p - 2, (p - 1) // 2, (p + 1) // 2, 1 << (n - 1), (1 << 32) - 1, c, p - c,
             top ^ (1 << 32), (1 << n) - (1 << 32)]
    squares = [a for a, _ in kv_twin.rare_cases(f, rng)[4] if a < p]
    targets = (p, p + 1, top, p - 1, top ^ (1 << 32), top ^ (1 << (n - 33)), top ^ 1, (1 << n) - (1 << 32))
    sig = [e for t in targets for e in (t, t ^ key.mask) if e < p]
    out = [(key.rounds, y) for y in edges + squares] + [(0, m) for m in edges]
    for j in range(1, key.rounds):
        out += [(j, e) for e in edges + squares + sig]
        out += [(j, key.sigma(a)) for a in (0, 1, p - 1, p - 2)]
    return out


def to_bytes(words: list[int]) -> bytes:
    return b"".join(w.to_bytes(WORD, "little") for w in words)


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_words(torch, libs: list[Lib], key: K.Key, log) -> dict:
    """Crafted and random words in a 64 x 1024 bf16 weight, at each GATE_BASES index: kv_gemm_dump's tiles at every nt,
    batch tile and splits 1-3 against kv.py's decode and the library's kv_decode."""
    rng = random.Random(1024)
    N, Kd = 64, 1024
    n = N * Kd // WE
    specs = crafted(key, rng)
    where = rng.sample(range(n), len(specs))
    out, hits = {"words": n, "crafted": len(specs), "bases": list(GATE_BASES), "bad": []}, {}
    st = stream(torch)
    for base in GATE_BASES:
        ys = [rng.randrange(key.field.p) for _ in range(n)]
        for pos, (j, e) in zip(where, specs):
            ys[pos] = forward(e, j, base + pos, key)
        ms = [K.decode_word(y, base + pos, key) for pos, y in enumerate(ys)]
        for pos in where:
            for b in branches(ys[pos], base + pos, key):
                hits[b] = hits.get(b, 0) + 1
        y = torch.frombuffer(bytearray(to_bytes(ys)), dtype=torch.uint8).cuda()
        want = torch.frombuffer(bytearray(to_bytes(ms)), dtype=torch.uint8).cuda()
        dec, dump = torch.empty_like(y), torch.empty_like(y)
        for lib in libs:
            try:
                lib.call("kv_decode", y.data_ptr(), dec.data_ptr(), n, base, key.rounds, 0, None, st)
                torch.cuda.synchronize()
                if not torch.equal(dec, want):
                    out["bad"].append({"lib": lib.name, "base": base, "what": "kv_decode"})
                for nt in (64, 32):
                    for m in TILES:
                        steps = Kd // (WE * 128 // nt)
                        for splits in (1, 2, 3):
                            per = -(-steps // splits)
                            dump.fill_(0xA5)
                            got = lib.call("kv_gemm_dump", y.data_ptr(), base, dump.data_ptr(), m, N, Kd, key.rounds,
                                           nt, splits, st)
                            torch.cuda.synchronize()
                            if got != -(-steps // per) or not torch.equal(dump, want):
                                out["bad"].append({"lib": lib.name, "base": base, "nt": nt, "m": m, "splits": splits,
                                                   "got": got})
            except RuntimeError as e:
                out["bad"].append({"lib": lib.name, "base": base, "error": str(e)})
    out["branches_hit"] = hits
    out["branches_ok"] = all(hits.get(b, 0) for b in ("sigma_slow", "sigma_undo", "sub_zero", "root_edge"))
    out["failed_libs"] = sorted({b["lib"] for b in out["bad"]})
    out["ok"] = not out["bad"] and out["branches_ok"]
    log(f"gate words: {'ok' if out['ok'] else 'FAIL'} ({n} words, {len(specs)} crafted, at {list(GATE_BASES)}; "
        f"branches {hits}; bad {out['bad'][:4]})")
    return out


def layout(layers: int = LAYERS) -> list[tuple[str, int, int, int]]:
    """Gate run 7's arena: (name, offset, N, K) for the embedding, each layer's linears, the LM head."""
    shapes = [("embed", VOCAB, HIDDEN)]
    for layer in range(layers):
        shapes += [(f"{layer}.{name}", N, Kd) for name, N, Kd in LINEARS]
    shapes.append(HEAD)
    offsets = pack([N * Kd * 2 for _, N, Kd in shapes], 62 * GIB)
    return [(name, off, N, Kd) for (name, N, Kd), off in zip(shapes, offsets)]


def ulp_bf16(torch, v):
    e = torch.floor(torch.log2(v.abs().clamp_min(2.0 ** -126)))
    return torch.pow(2.0, e - 7)


def gate_shapes(torch, libs: list[Lib], enc_lib: Lib, key: K.Key, log) -> dict:
    """Each linear's shape: random bf16 weights encoded at GPU 7's indices, decoded back by kv_decode and by
    kv_gemm_dump's tiles; then kv_gemm against F.linear and float64."""
    import torch.nn.functional as F

    st, g = stream(torch), torch.Generator(device="cuda").manual_seed(7)
    at = {name.split(".")[-1]: off for name, off, _, _ in layout() if name.startswith("0.") or name == "lm_head"}
    res, bad = {}, []
    for name, N, Kd in LINEARS + (HEAD,):
        words = N * Kd // WE
        base = (7 * SHARD_BLOCKS + at[name] // BLOCK) * (BLOCK // WORD)
        if name == "o":
            base = (1 << 32) - words // 2
        w = (torch.randn(N, Kd, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        enc = torch.empty(N * Kd * 2, dtype=torch.uint8, device="cuda")
        enc_lib.call("kv_encode", w.data_ptr(), enc.data_ptr(), words, base, key.rounds, 0, st)
        back = torch.empty_like(w)
        r = res[name] = {"N": N, "K": Kd, "base": base, "dump": {}, "gemm": {}}
        for lib in libs:
            try:
                lib.call("kv_decode", enc.data_ptr(), back.data_ptr(), words, base, key.rounds, 0, None, st)
                torch.cuda.synchronize()
                ok = torch.equal(back, w)
                r["dump"][f"{lib.name}:kv_decode"] = ok
                bad += [] if ok else [(name, lib.name, "kv_decode")]
                for nt in (64, 32):
                    for m in (1, 64):
                        back.view(torch.int16).fill_(-1)
                        lib.call("kv_gemm_dump", enc.data_ptr(), base, back.data_ptr(), m, N, Kd, key.rounds, nt, 0,
                                 st)
                        torch.cuda.synchronize()
                        ok = torch.equal(back, w)
                        r["dump"][f"{lib.name}:{nt}:{m}"] = ok
                        bad += [] if ok else [(name, lib.name, nt, m)]
            except RuntimeError as e:
                r["dump"][f"{lib.name}:error"] = False
                bad.append((name, lib.name, str(e)))
        bias = (torch.randn(N, device="cuda", generator=g) * 0.1).to(torch.bfloat16) if name == "qkv" else None
        for m in GATE_MS:
            x = torch.randn(m, Kd, device="cuda", generator=g).to(torch.bfloat16)
            ref = F.linear(x, w, bias)
            y64 = torch.empty(m, N, dtype=torch.float64, device="cuda")
            s64 = torch.empty_like(y64)
            for n0 in range(0, N, 8192):
                wd = w[n0:n0 + 8192].double()
                y64[:, n0:n0 + 8192] = x.double() @ wd.T
                s64[:, n0:n0 + 8192] = x.double().abs() @ wd.abs().T
            if bias is not None:
                y64 += bias.double()
                s64 += bias.double().abs()
            acc = Kd * 2.0 ** -24 * s64
            bound = acc + 0.5 * ulp_bf16(torch, y64.abs() + acc)
            ref_within = bool(((ref.double() - y64).abs() <= bound).all())
            for lib in libs:
                for nt in (64, 32):
                    try:
                        splits = lib.call("kv_gemm_splits", m, N, Kd, nt)
                        part = torch.empty(max(1, splits) * m * N, dtype=torch.float32, device="cuda")
                        y = torch.empty(m, N, dtype=torch.bfloat16, device="cuda")
                        y2 = torch.empty_like(y)
                        for out in (y, y2):
                            got = lib.call("kv_gemm", enc.data_ptr(), base, x.data_ptr(), out.data_ptr(),
                                           bias.data_ptr() if bias is not None else None, part.data_ptr(),
                                           part.numel() * 4, m, N, Kd, key.rounds, nt, 0, st)
                        torch.cuda.synchronize()
                    except RuntimeError as e:
                        r["gemm"][f"{lib.name}:{nt}:{m}"] = {"error": str(e)}
                        bad.append((name, lib.name, nt, m, str(e)))
                        continue
                    err = (y.double() - y64).abs()
                    within = bool((err <= bound).all())
                    equal = float((y.view(torch.int16) == ref.view(torch.int16)).double().mean())
                    ulps = float(((y.double() - ref.double()).abs() / ulp_bf16(torch, ref.double())).max())
                    same = torch.equal(y, y2)
                    r["gemm"][f"{lib.name}:{nt}:{m}"] = {
                        "splits": got, "within_fp32_rounding": within, "equal_to_f_linear": equal,
                        "max_ulps_from_f_linear": ulps, "max_err_over_bound": float((err / bound).max()),
                        "f_linear_max_err_over_bound": float(((ref.double() - y64).abs() / bound).max()),
                        "f_linear_within": ref_within, "deterministic": same}
                    if not (within and same):
                        bad.append((name, lib.name, nt, m, "gemm"))
            del ref, y64, s64, bound, acc
        del w, enc, back
        torch.cuda.empty_cache()
        for lib in libs:
            dumps = all(v for k, v in r["dump"].items() if k.startswith(f"{lib.name}:"))
            gs = [v for k, v in r["gemm"].items() if k.startswith(f"{lib.name}:")]
            errs = [v["error"] for v in gs if "error" in v]
            gs = [v for v in gs if "error" not in v]
            eq = [v["equal_to_f_linear"] for v in gs] or [math.nan]
            log(f"gate {name} [{N}, {Kd}] at index {base}, {lib.name}: dumps {dumps}, gemm within "
                f"{all(v['within_fp32_rounding'] for v in gs)} (F.linear within "
                f"{all(v['f_linear_within'] for v in gs)}), deterministic {all(v['deterministic'] for v in gs)}, "
                f"bit-equal to F.linear {min(eq):.4f}-{max(eq):.4f} of outputs, max "
                f"{max([v['max_ulps_from_f_linear'] for v in gs] or [math.nan]):.2f} ulps; errors {errs[:2]}")
    return {"shapes": res, "bad": bad, "failed_libs": sorted({b[1] for b in bad}), "ok": not bad}


# --- timing ------------------------------------------------------------------------------------------------------------

class Model:
    """The arena, the activations and the arms' ops, at M rows."""

    def __init__(self, torch, libs: dict[str, Lib], key: K.Key, gib: int, entries=None, seed: int = 3) -> None:
        import torch.nn.functional as F

        self.torch, self.F, self.libs, self.k = torch, F, libs, key.rounds
        self.entries = entries if entries is not None else layout()
        end = max(off + N * Kd * 2 for _, off, N, Kd in self.entries)
        size = gib * GIB if entries is None else -(-end // BLOCK) * BLOCK
        self.buf = torch.empty(size, dtype=torch.uint8, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(seed)
        for at in range(0, size, GIB):
            n = min(GIB, size - at) // 4
            self.buf[at:at + 4 * n].view(torch.int32).copy_(
                torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), dtype=torch.int32, device="cuda", generator=g))
        self.buf.view(-1, WORD)[:, WORD - 1] &= 0x7F
        biggest = max(N * Kd * 2 for _, _, N, Kd in self.entries)
        self.scratch = torch.empty(biggest, dtype=torch.uint8, device="cuda")
        self.x = {d: torch.randn(64, d, device="cuda", generator=g).to(torch.bfloat16) for d in (HIDDEN, INTER)}
        self.y = {N: torch.empty(64, N, dtype=torch.bfloat16, device="cuda") for N in {N for _, _, N, _ in self.entries}}
        self.bias = (torch.randn(QKV, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
        self.ids = torch.randint(0, VOCAB, (64,), device="cuda", generator=g)
        self.ids_host = self.ids.tolist()
        self.rows = torch.empty(64, HIDDEN, dtype=torch.bfloat16, device="cuda")
        self.layers = sum(1 for name, *_ in self.entries if name.endswith(".qkv"))
        self.kv = torch.zeros(max(1, self.layers) * KV_LAYER_BYTES, dtype=torch.uint8, device="cuda")
        self.xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count
        most = 1
        for lib in libs.values():
            if lib.fused:
                for _, _, N, Kd in self.entries[1:]:
                    for m in (1, 8, 32, 64):
                        for nt in (64, 32):
                            most = max(most, lib.call("kv_gemm_splits", m, N, Kd, nt) * m * N)
        self.part = torch.empty(most, dtype=torch.float32, device="cuda")
        self.splits: dict[tuple, int] = {}

    def view(self, off: int, N: int, Kd: int):
        return self.buf[off:off + N * Kd * 2].view(self.torch.bfloat16).view(N, Kd)

    def base(self, off: int) -> int:
        return off // BLOCK * (BLOCK // WORD)

    def linear(self, arm: str, name: str, off: int, N: int, Kd: int, m: int) -> None:
        torch, F = self.torch, self.F
        x, bias = self.x[Kd][:m], self.bias if name.endswith("qkv") else None
        kind, *rest = arm.split(":")
        if kind == "plain":
            F.linear(x, self.view(off, N, Kd), bias)
        elif kind == "scratch":
            self.libs[rest[0]].call("kv_decode", self.buf.data_ptr() + off, self.scratch.data_ptr(), N * Kd // WE,
                                    self.base(off), self.k, 0, None, stream(torch))
            F.linear(x, self.scratch[:N * Kd * 2].view(torch.bfloat16).view(N, Kd), bias)
        else:
            lib, nt = self.libs[rest[0]], int(rest[1])
            at = (lib.name, m, N, Kd, nt)
            if at not in self.splits:
                self.splits[at] = lib.call("kv_gemm_splits", m, N, Kd, nt)
            lib.call("kv_gemm", self.buf.data_ptr() + off, self.base(off), x.data_ptr(), self.y[N].data_ptr(),
                     bias.data_ptr() if bias is not None else None, self.part.data_ptr(), self.part.numel() * 4, m, N,
                     Kd, self.k, nt, self.splits[at], stream(torch))

    def embed(self, arm: str, off: int, m: int) -> None:
        torch, F = self.torch, self.F
        kind, *rest = arm.split(":")
        if kind == "plain":
            F.embedding(self.ids[:m], self.view(off, VOCAB, HIDDEN))
        elif kind == "scratch":
            self.libs[rest[0]].call("kv_decode", self.buf.data_ptr() + off, self.scratch.data_ptr(),
                                    VOCAB * HIDDEN // WE, self.base(off), self.k, 0, None, stream(torch))
            F.embedding(self.ids[:m], self.scratch[:VOCAB * HIDDEN * 2].view(torch.bfloat16).view(VOCAB, HIDDEN))
        else:
            lib, row = self.libs[rest[0]], HIDDEN * 2
            for r, i in enumerate(self.ids_host[:m]):
                lib.call("kv_decode", self.buf.data_ptr() + off + i * row, self.rows.data_ptr() + r * row,
                         HIDDEN // WE, self.base(off) + i * (HIDDEN // WE), self.k, 0, None, stream(torch))

    def attention(self, layer: int) -> None:
        lib = next(iter(self.libs.values()))
        lib.call("kv_read", self.kv.data_ptr() + layer * KV_LAYER_BYTES, KV_LAYER_BYTES, self.xr.data_ptr(),
                 4 * self.sms, stream(self.torch))

    def scaled(self, f: float) -> list:
        """The entries with round(f x layers) layers, the model's taken in turn (a longer or shorter step)."""
        head, *body, tail = self.entries
        per = len(body) // self.layers
        groups = [body[i * per:(i + 1) * per] for i in range(self.layers)]
        return [head] + [e for i in range(round(f * self.layers)) for e in groups[i % self.layers]] + [tail]

    def ops(self, arm: str, m: int):
        """The step's ops in order: (kind, thunk). ARM*F: the step with F times the layers."""
        arm, _, f = arm.partition("*")
        layer = -1
        for name, off, N, Kd in self.scaled(float(f)) if f else self.entries:
            if name == "embed":
                yield "embed", lambda off=off: self.embed(arm, off, m)
                continue
            if name.endswith(".qkv"):
                layer += 1
                yield "attention", lambda layer=layer % self.layers: self.attention(layer)
            yield name.split(".")[-1], lambda name=name, off=off, N=N, Kd=Kd: self.linear(arm, name, off, N, Kd, m)

    def step(self, arm: str, m: int, per_op: bool = False) -> tuple[float, dict]:
        """One step's GPU time (ms), and with per_op each op kind's times."""
        torch = self.torch
        ev = lambda: torch.cuda.Event(enable_timing=True)
        a, b, marks = ev(), ev(), []
        a.record()
        for kind, run in self.ops(arm, m):
            if per_op:
                s, e = ev(), ev()
                s.record()
                run()
                e.record()
                marks.append((kind, s, e))
            else:
                run()
        b.record()
        b.synchronize()
        kinds: dict[str, list[float]] = {}
        for kind, s, e in marks:
            kinds.setdefault(kind, []).append(s.elapsed_time(e))
        return a.elapsed_time(b), kinds


def rates(torch, model: Model, libs: dict[str, Lib], key: K.Key, log) -> dict:
    """The bare read of the whole arena and each library's read-and-decode of it (checksum mode), best of 3 after one."""
    out, st, n = {}, stream(torch), model.buf.numel()

    def best(fn) -> float:
        fn()
        ts = []
        for _ in range(3):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        return min(ts)

    lib0 = next(iter(libs.values()))
    ms = best(lambda: lib0.call("kv_read", model.buf.data_ptr(), n, model.xr.data_ptr(), 4 * model.sms, st))
    out["read"] = {"ms": ms, "gb_per_s": n / ms / 1e6}
    for name, lib in libs.items():
        ms = best(lambda: lib.call("kv_decode", model.buf.data_ptr(), None, n // WORD, 0, key.rounds, 0,
                                   model.xr.data_ptr(), st))
        out[f"decode:{name}"] = {"ms": ms, "gb_per_s": n / ms / 1e6, "over_read": ms / out["read"]["ms"]}
    log(f"rates over {n / 1e9:.2f} GB: " + ", ".join(f"{k} {v['gb_per_s']:.0f} GB/s" for k, v in out.items()))
    return out


def timing(torch, model: Model, arms: list[str], ms: list[int], reps: int, log) -> dict:
    out = {}
    for arm in arms:
        for m in ms:
            model.step(arm, m)
            totals, kinds = [], {}
            for _ in range(reps):
                total, k = model.step(arm, m, per_op=True)
                totals.append(total)
                for kind, xs in k.items():
                    kinds.setdefault(kind, []).extend(xs)
            bare = [model.step(arm, m)[0] for _ in range(reps)]
            per = {kind: {"median_ms": statistics.median(xs), "count_a_step": len(xs) // reps,
                          "a_step_ms": statistics.median(xs) * (len(xs) // reps)} for kind, xs in kinds.items()}
            out[f"{arm}@{m}"] = {"arm": arm, "m": m, "step_ms": statistics.median(bare), "step_ms_all": bare,
                                 "step_ms_evented": statistics.median(totals), "ops": per}
            log(f"time {arm} M={m}: step {statistics.median(bare):.2f} ms; " +
                ", ".join(f"{kind} {v['median_ms']:.3f}" for kind, v in per.items()))
    return out


# --- ncu ---------------------------------------------------------------------------------------------------------------

def ncu_ops(torch, model: Model, arms: list[str], ms: list[int], out: Path) -> None:
    """Layer 0's linears and the LM head for each arm and M, each behind a one-cycle kv_spin (the CSV's op marker)."""
    lib0, st, order = next(iter(model.libs.values())), stream(torch), []
    for arm in arms:
        for m in ms:
            for kind, run in model.ops(arm, m):
                if kind in ("embed", "attention"):
                    continue
                lib0.call("kv_spin", 1, st)
                run()
                torch.cuda.synchronize()
                order.append({"arm": arm, "m": m, "op": kind})
    write_json(out / "ncu_ops.json", order)


def ncu_report(csv_path: Path, ops_path: Path) -> dict:
    """ncu's CSV (--csv --print-units base, metrics dram__bytes.sum and gpu__time_duration.sum) folded by op: each
    marker kernel starts the next op of ncu_ops.json. dram__bytes.sum is DRAM read plus write; the RTX PRO 6000 gives
    n/a for dram__bytes_read.sum and dram__bytes_write.sum, and a kernel's n/a counts as missing."""
    order = json.loads(ops_path.read_text())
    kernels: dict[int, dict] = {}
    with csv_path.open() as f:
        lines = [ln for ln in f if ln.startswith('"')]
    for row in csv.DictReader(lines):
        k = kernels.setdefault(int(row["ID"]), {"name": row["Kernel Name"]})
        if row["Metric Value"] != "n/a":
            k[row["Metric Name"]] = float(row["Metric Value"].replace(",", ""))
    ops, cur = [], None
    for _, k in sorted(kernels.items()):
        if "spin_kernel" in k["name"]:
            cur = {"bytes": 0.0, "ns": 0.0, "kernels": [], "missing": 0}
            ops.append(cur)
            continue
        if cur is None:
            continue
        cur["bytes"] += k.get("dram__bytes.sum", 0.0)
        cur["missing"] += "dram__bytes.sum" not in k
        cur["ns"] += k.get("gpu__time_duration.sum", 0.0)
        cur["kernels"].append(k["name"][:80])
    if len(ops) != len(order):
        raise RuntimeError(f"{len(ops)} markers in the CSV against {len(order)} ops")
    out = {}
    for o, got in zip(order, ops):
        out[f"{o['arm']}@{o['m']}:{o['op']}"] = {**o, **got}
    return out


# --- serve -------------------------------------------------------------------------------------------------------------

def serve(torch, model: Model, d: Path, out: Path, hz: float, m: int, failed: set[str], log) -> None:
    info = {"bytes": model.buf.numel(), "pid": os.getpid(), "sms": model.sms, **export(model.buf.data_ptr())}
    write_json(d / "engine.json", info)
    log(f"serve: arena exported ({model.buf.numel()} bytes)")
    interval, seq, arm, steps = 1.0 / hz, -1, None, 0
    next_at = aligned(time.perf_counter(), time.time(), interval)
    with (out / "serve.steps.jsonl").open("w") as f:
        while True:
            try:
                ctl = json.loads((d / "ctl.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                ctl = {}
            if ctl.get("stop"):
                break
            if ctl.get("seq", -1) != seq:
                seq, arm = ctl["seq"], ctl.get("arm")
                if arm and arm.startswith("fused:") and arm.split(":")[1] in failed:
                    log(f"serve: {arm} failed its gate, idle instead")
                    arm = None
                write_json(d / "status.json", {"seq": seq, "arm": arm})
            wait = next_at - time.perf_counter()
            if wait > 0:
                time.sleep(min(wait, 0.02))
                continue
            if arm:
                t, h0 = time.time(), time.perf_counter()
                gpu_ms, _ = model.step(arm, m)
                f.write(json.dumps({"t": t, "arm": arm, "gpu_ms": gpu_ms,
                                    "host_ms": (time.perf_counter() - h0) * 1e3}) + "\n")
                steps += 1
                if steps % 32 == 0:
                    f.flush()
            next_at = aligned(time.perf_counter(), time.time(), interval)
    (d / "stopped").touch()
    log(f"serve: stopped after {steps} steps")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="gate,time", help="gate, time, ncu, ncu-report, serve (comma-separated)")
    ap.add_argument("--lib", action="append", default=[], help="NAME=PATH, kv.so and fused.so builds")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arena-gib", type=int, default=62)
    ap.add_argument("--ms", default="1,8,32,64")
    ap.add_argument("--arms", default="plain,scratch:kv,scratch:kv6,fused:fused:64,fused:fused:32,fused:fused6:64,"
                                      "fused:fused6:32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ncu-csv", type=Path)
    ap.add_argument("--serve-dir", type=Path)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--serve-m", type=int, default=64)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    modes = a.mode.split(",")
    logf = (a.out / f"fused_bench.{'-'.join(modes)}.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if modes == ["ncu-report"]:
        rep = ncu_report(a.ncu_csv, a.out / "ncu_ops.json")
        write_json(a.out / "ncu.json", rep)
        for k, v in rep.items():
            log(f"ncu {k}: {v['bytes'] / 1e6:.1f} MB, {v['ns'] / 1e3:.1f} us" +
                (f" ({v['missing']} kernels without bytes)" if v["missing"] else ""))
        return 0
    import torch

    libs = {}
    for spec in a.lib:
        name, path = spec.split("=", 1)
        libs[name] = Lib(name, path)
    key = K.key(FIELD, SEED, a.rounds, mask=True)
    for lib in libs.values():
        lib.set_key(key)
    fused = [lib for lib in libs.values() if lib.fused]
    res_path = a.out / "fused_bench.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res.update({"schema": "verity/remote-session/fused-bench/v0", "rounds": a.rounds, "field": FIELD,
                "device": torch.cuda.get_device_name(0),
                "libs": {n: {"path": lib.path, "fused": lib.fused, "fold6": lib.fold6} for n, lib in libs.items()}})
    for lib in fused:
        res["libs"][lib.name]["occupancy"] = {f"{m}:{nt}": lib.call("kv_gemm_occupancy", m, nt)
                                             for m in (1, 16, 32, 64) for nt in (64, 32)}
    if "gate" in modes:
        g = res["gate"] = {"words": gate_words(torch, fused, key, log)}
        g["shapes"] = gate_shapes(torch, fused, libs.get("kv", fused[0]), key, log)
        failed = set(g["words"]["failed_libs"]) | set(g["shapes"]["failed_libs"])
        g["passed_libs"] = [lib.name for lib in fused if lib.name not in failed]
        g["failed_libs"] = sorted(failed)
        g["ok"] = g["words"]["branches_ok"] and bool(g["passed_libs"])
        write_json(res_path, res)
        log(f"gate: {'ok' if g['ok'] else 'FAIL, nothing timed'}; passed {g['passed_libs']}, failed {g['failed_libs']}"
            f" (left untimed)")
        if not g["ok"]:
            return 4
        torch.cuda.empty_cache()
    elif ({"time", "ncu", "serve"} & set(modes)) and not res.get("gate", {}).get("ok"):
        log("no passed gate in fused_bench.json: nothing is timed")
        return 4
    failed = set(res.get("gate", {}).get("failed_libs", ()))
    arms, ms = a.arms.split(","), [int(m) for m in a.ms.split(",")]
    arms = [arm for arm in arms if not (arm.startswith("fused:") and arm.split(":")[1] in failed)]
    timed = {n: lib for n, lib in libs.items() if n not in failed}
    if "time" in modes:
        model = Model(torch, timed, key, a.arena_gib)
        res["rates"] = rates(torch, model, libs, key, log)
        write_json(res_path, res)
        res["time"] = timing(torch, model, arms, ms, a.reps, log)
        write_json(res_path, res)
        del model
        torch.cuda.empty_cache()
    if "ncu" in modes:
        model = Model(torch, timed, key, 0, entries=layout(1))
        ncu_ops(torch, model, arms, ms, a.out)
    if "serve" in modes:
        model = Model(torch, timed, key, a.arena_gib)
        serve(torch, model, a.serve_dir, a.out, a.hz, a.serve_m, failed, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
