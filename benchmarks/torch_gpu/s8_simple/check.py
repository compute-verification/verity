"""The check of `s8_simple.linear` on a GPU at the frozen shape's six linears (HASH-11): every checked word must be
`ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`'s, through its exact CPU twin (`twin`: held to the Definition's reference
evaluator by tests/test_s8_simple.py, and by `verify` itself on a sample of coordinates at the full K).

    check.py weights --scratch S [--sets edge,served] [--weights-module PATH]   CPU, before the lease: w and sw
    check.py gpu --scratch S --out OUT [--sets ...] [--ms 1,7,128,4096,...]      the leased GPU: linear and its timings
    check.py verify --scratch S --out OUT                                       CPU, after it: the twin and the Definition

`check.sh` runs the three under `research run`. The inputs, at each (n, k) and m:

* **edge** weights: uniform s8 codes; the first channels are `SPECIALS` (a code pattern and a scale each), the rest have
  a typical scale (|N(0,1)| 0.02 / 127), or one in 8 a uniformly random f32 word (any exponent, NaN and inf included).
* **served** weights: layer 0's tensors of the demo's weights (`verity_toy.weights`, shape H200, the public seed),
  kv the rows of wk then wv.
* x, under both: uniform s8 codes drawn on the GPU, the first rows `SPECIALS` with their scales, the rest a typical scale
  (|N(0,1)| / 127) or one in 8 a random f32 word. The patterns: max-magnitude codes (all -128, all 127, alternating),
  zero rows, and codes in [100, 127] or [-128, -100] (sums past 2^24 with random low bits, up to the 2^14 K = 2^30 an s8
  sum reaches at K = 65536; +-2^31 is out of reach below K = 2^17). The scales: +-0, subnormals, the least normal, 2^-100,
  2^100, +-max, +-inf and three NaNs.

Every row is checked at m <= 128, and `ROWS` rows at larger m (the special rows, the last, the rest drawn), each over all
n channels, the s32 sums (`int_mm`) as well as the words of both epilogues. On CUDA, `gpu` also compares the fused
epilogue's words with the eager one's at every coordinate of every m, and audits the PTX of each variant the fused kernel
compiled to (`ptx_audit`). It times the s32 GEMM (`int_mm`), the eager epilogue (`scale`), the fused one (`fused_scale`)
and `linear` with each, using CUDA events after a warm-up, and checks that the last timed call's words are the first's.
`verify` counts, per (set, n, k, m), the mismatches, the reference words by class (NaN, inf, zero, subnormal), and the
coordinates where the other two scale orders, sw first or sx * sw first, round to another word: what a reordered epilogue
would have failed on. It sums the times over a layer's linears (`LAYER`).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
#: the frozen shape's linears (MQA decoder: width 16384, 256 query heads of 64, 1 KV head, MLP 65536, 256 tokens): (n, k)
SHAPES = {"q": (16384, 16384), "kv": (128, 16384), "o": (16384, 16384), "up": (65536, 16384), "down": (16384, 65536),
          "head": (256, 16384)}
#: each linear's tensors in the demo's weights, layer 0
SERVED = {"q": ("l0.wq",), "kv": ("l0.wk", "l0.wv"), "o": ("l0.wo",), "up": ("l0.wup",), "down": ("l0.wdown",),
          "head": ("head",)}
#: the demo's public weights seed (`text:verity/cmp92/h200/weights/v1`)
SEED = b"verity/cmp92/h200/weights/v1"
SETS = ("edge", "served")
#: one decoder layer's linears (the head runs once a step)
LAYER = ("q", "kv", "o", "up", "down")
TIMES = ("gemm_ms", "scale_ms", "fused_ms", "linear_ms", "linear_eager_ms")
#: token rows: small batches, the skeleton's batch (4096), the block's (16384), 12288 between them, and the frozen
#: shape's full step (20480)
MS = (1, 7, 128, 4096, 12288, 16384, 20480)
#: rows checked at m > 128, and coordinates a run checks through the Definition's own evaluator
ROWS, DEFINITION_COORDS = 256, 16
TOKEN_SPREAD, CHANNEL_SPREAD = 1 / 127, 0.02 / 127
NAN_BF16 = 0x7FFF
WARM, REPS = 3, 10

#: special f32 scale words
WORDS = {"one": 0x3F800000, "+0": 0, "-0": 0x80000000, "sub-min": 1, "sub-mid": 0x00400000, "sub-max": 0x007FFFFF,
         "-sub-min": 0x80000001, "normal-min": 0x00800000, "2^-100": 0x0D800000, "2^100": 0x71800000, "max": 0x7F7FFFFF,
         "-max": 0xFF7FFFFF, "+inf": 0x7F800000, "-inf": 0xFF800000, "qnan": 0x7FC00000, "snan": 0x7F800001,
         "-nan": 0xFFFFFFFF}
PATTERNS = ("rand", "min", "max", "big+", "big-", "zero", "alt")
#: the special rows of x and channels of w: (code pattern, scale word, None a typical scale)
SPECIALS = ([(p, None) for p in PATTERNS] + [("rand", w) for w in WORDS]
            + [("zero", "+inf"), ("zero", "qnan"), ("big+", "max"), ("min", "sub-min"), ("big-", "2^-100"), ("max", "-inf")])


# ---- the twin --------------------------------------------------------------------------------------------------------

def exact_sums(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The s32 sums of s8 rows x [r, K] against each row of w [n, K], int64 [r, n]: `GemmAccumulatorS8_v1`'s word for
    K < 2^17. The float64 product is exact: every partial sum is an integer of magnitude at most 2^14 K < 2^53."""
    r, k = x.shape
    xf, out = x.astype(np.float64), np.empty((r, w.shape[0]), np.int64)
    step = max(1, (1 << 26) // k)
    for j in range(0, w.shape[0], step):
        out[:, j:j + step] = xf @ np.asarray(w[j:j + step], dtype=np.float64).T
    return out


def bf16_rn(u: np.ndarray) -> np.ndarray:
    """`F2fpBf16_v1` of f32 words: round to nearest even, every NaN 0x7FFF."""
    v = u.astype(np.uint64)
    out = ((v + 0x7FFF + ((v >> 16) & 1)) >> 16).astype(np.uint16)
    out[((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)] = NAN_BF16
    return out


def epilogue(acc: np.ndarray, sx: np.ndarray, sw: np.ndarray, order: str = "definition") -> np.ndarray:
    """bf16 words [r, n] of the s32 sums acc [r, n] under the token scales sx [r] and channel scales sw [n]: the
    Definition's order, `cvt.rn.f32.s32` then two f32 mul.rn, sx first, or `sw-first` or `scales-first` (sx * sw)."""
    f, a, b = acc.astype(np.int32).astype(np.float32), sx[:, None], sw[None, :]
    with np.errstate(all="ignore"):
        y = {"definition": lambda: (f * a) * b, "sw-first": lambda: (f * b) * a, "scales-first": lambda: f * (a * b)}[order]()
    return bf16_rn(y.astype(np.float32).view(np.uint32))


def twin(x: np.ndarray, sx: np.ndarray, w: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """`ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` at every (row of x, row of w): bf16 words uint16 [r, n]."""
    return epilogue(exact_sums(x, w), sx, sw)


# ---- inputs ----------------------------------------------------------------------------------------------------------

def pattern(name: str, k: int, rng: np.random.Generator) -> np.ndarray:
    lo_hi = {"rand": (-128, 128), "big+": (100, 128), "big-": (-128, -99)}
    if name in lo_hi:
        return rng.integers(*lo_hi[name], k).astype(np.int8)
    if name == "alt":
        return np.resize(np.array([-128, 127], np.int8), k)
    return np.full(k, {"min": -128, "max": 127, "zero": 0}[name], np.int8)


def scales(rng: np.random.Generator, n: int, spread: float) -> np.ndarray:
    """n typical scales |N(0,1)| spread, one in 8 replaced by a uniformly random f32 word."""
    s = (np.abs(rng.standard_normal(n)) * spread).astype(np.float32)
    wild = rng.random(n) < 1 / 8
    s[wild] = rng.integers(0, 1 << 32, int(wild.sum()), dtype=np.uint64).astype(np.uint32).view(np.float32)
    return s


def specials(count: int, k: int, rng: np.random.Generator, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """The first `count` of `SPECIALS`: codes int8 [count, k] and scales float32 [count]."""
    codes, sc = np.empty((count, k), np.int8), np.empty(count, np.float32)
    for i, (p, s) in enumerate(SPECIALS[:count]):
        codes[i] = pattern(p, k, rng)
        sc[i] = (abs(rng.standard_normal()) * spread) if s is None else np.uint32(WORDS[s]).view(np.float32)
    return codes, sc


def seed_of(*parts) -> int:
    return int.from_bytes(hashlib.sha512("/".join(map(str, ("s8_simple/v1",) + parts)).encode()).digest()[:8], "little")


def sample_rows(m: int, t: int, rng: np.random.Generator) -> np.ndarray:
    """Every row at m <= 128; else the t special rows, the last row and drawn rows, `ROWS` in all."""
    if m <= 128:
        return np.arange(m)
    rest = rng.choice(np.arange(t, m - 1), ROWS - t - 1, replace=False)
    return np.sort(np.concatenate([np.arange(t), rest, [m - 1]])).astype(np.int64)


def weights_module(path: str | None):
    if not path:
        from verity_toy import weights
        return weights
    name = "s8_simple_served_weights"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _served_rows(job: tuple) -> None:
    path, out, names, lo, hi = job
    W, w = weights_module(path), np.load(out, mmap_mode="r+")
    for name, first, rows in names:
        a, b = max(lo, first), min(hi, first + rows)
        if a < b:
            w[a:b] = W.generate(W.H200, SEED, name, range(a - first, b - first))
    w.flush()


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()


def write_weights(a: argparse.Namespace) -> int:
    scratch, made = Path(a.scratch), {}
    for set_ in a.sets:
        for name, (n, k) in a.shapes.items():
            wp, sp = scratch / f"w-{set_}-{name}.npy", scratch / f"sw-{set_}-{name}.npy"
            if set_ == "edge":
                rng = np.random.default_rng(seed_of("w", name, n, k))
                w = np.frombuffer(bytearray(rng.bytes(n * k)), np.int8).reshape(n, k)
                sw = scales(rng, n, CHANNEL_SPREAD)
                c = min(n, len(SPECIALS))
                w[:c], sw[:c] = specials(c, k, rng, CHANNEL_SPREAD)
                np.save(wp, w)
                np.save(sp, sw)
            else:
                W = weights_module(a.weights_module)
                ts = [W.tensor(W.H200, t) for t in SERVED[name]]
                firsts = np.cumsum([0] + [t.rows for t in ts])
                if (int(firsts[-1]), {t.cols for t in ts}) != (n, {k}):
                    raise SystemExit(f"{name}: the served tensors {SERVED[name]} are not ({n}, {k})")
                np.lib.format.open_memmap(wp, mode="w+", dtype=np.int8, shape=(n, k)).flush()
                names = [(t.name, int(f), t.rows) for t, f in zip(ts, firsts)]
                step = max(1, (1 << 24) // k)
                jobs = [(a.weights_module, str(wp), names, lo, min(n, lo + step)) for lo in range(0, n, step)]
                with multiprocessing.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
                    pool.map(_served_rows, jobs)
                w = np.load(wp, mmap_mode="r")
                sw = np.concatenate([W.generate(W.H200, SEED, f"{t}.scale")[0] for t in SERVED[name]]).astype(np.float32)
                np.save(sp, sw)
            made[f"{set_}/{name}"] = {"n": n, "k": k, "w": wp.name, "sw": sp.name, "w_sha256": _sha(w), "sw_sha256": _sha(sw),
                                      "tensors": list(SERVED[name]) if set_ == "served" else None}
            print(f"weights {set_}/{name} ({n}, {k}) {made[f'{set_}/{name}']['w_sha256'][:16]}", flush=True)
    (scratch / "weights.json").write_text(json.dumps(made, indent=1))
    return 0


# ---- the GPU ---------------------------------------------------------------------------------------------------------

def _cublaslt_version() -> int | None:
    import ctypes
    with open("/proc/self/maps") as f:
        paths = {line.split()[-1] for line in f if "libcublasLt" in line}
    for p in sorted(paths):
        try:
            return int(ctypes.CDLL(p).cublasLtGetVersion())
        except (OSError, AttributeError):
            continue
    return None


def _env(torch, dev) -> dict:
    e = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": dev.type,
         "lease_uuid": os.environ.get("GPU_LEASE_UUID"), "lease_who": os.environ.get("GPU_LEASE_WHO")}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        e |= {"name": p.name, "capability": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
              "memory_gib": round(p.total_memory / 2 ** 30, 1), "cublasLt": _cublaslt_version()}
    return e


def _timed(torch, fn, warm: int, reps: int):
    """(median ms, every rep's ms, the last call's result) of `fn` with CUDA events, after `warm` calls."""
    out = None
    for _ in range(warm):
        out = fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return float(np.median(ms)), [round(v, 4) for v in ms], out


_F32_ARITH = re.compile(r"\b(?:add|sub|mul|fma|mad|div|rcp|sqrt|neg|abs|min|max|ex2|lg2|cvt)(?:\.[a-z0-9]+)*?\.f32\b")


def ptx_audit(ptx: str) -> dict:
    """The fused kernel's PTX holds `FUSED_PTX` once a coordinate and no other f32 arithmetic: as many `mul.rn.f32` as
    twice the `cvt.rn.f32.s32`, as many `cvt.rn.bf16.f32`, and no fma, no `.ftz` and no other f32 instruction."""
    n = {op: len(re.findall(r"\b" + re.escape(op) + r"\b", ptx)) for op in ("cvt.rn.f32.s32", "mul.rn.f32", "cvt.rn.bf16.f32")}
    n |= {"fma": len(re.findall(r"\bfma\.", ptx)), "ftz": len(re.findall(r"\.ftz\b", ptx)),
          "blocks": ptx.count("mul.rn.f32 t, t,") // 2}
    n["other_f32"] = len(_F32_ARITH.findall(ptx)) - n["cvt.rn.f32.s32"] - n["mul.rn.f32"] - n["cvt.rn.bf16.f32"]
    c = n["cvt.rn.f32.s32"]
    n["ok"] = bool(c > 0 and n["mul.rn.f32"] == 2 * c and n["cvt.rn.bf16.f32"] == c == n["blocks"]
                   and n["fma"] == n["ftz"] == n["other_f32"] == 0)
    return n


def run_gpu(a: argparse.Namespace) -> int:
    import torch
    sys.path.insert(0, str(HERE))
    import s8_simple as S

    scratch, out = Path(a.scratch), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    made = json.loads((scratch / "weights.json").read_text())
    runs, env, ptx, fused = [], None, {}, dev.type == "cuda"
    for set_ in a.sets:
        for name, (n, k) in a.shapes.items():
            meta = made[f"{set_}/{name}"]
            w = torch.from_numpy(np.load(scratch / meta["w"])).to(dev)
            sw = torch.from_numpy(np.load(scratch / meta["sw"])).to(dev)
            for m in a.ms:
                rng = np.random.default_rng(seed_of("x", set_, name, m))
                g = torch.Generator(device=dev)
                g.manual_seed(int(rng.integers(1 << 62)))
                x = torch.randint(0, 256, (m, k), dtype=torch.uint8, device=dev, generator=g).view(torch.int8)
                sx_np = scales(rng, m, TOKEN_SPREAD)
                t = min(m, len(SPECIALS))
                codes, sc = specials(t, k, rng, TOKEN_SPREAD)
                x[:t], sx_np[:t] = torch.from_numpy(codes).to(dev), sc
                sx = torch.from_numpy(sx_np).to(dev)
                y_eager = S.linear(x, sx, w, sw, epilogue="eager")
                y = S.linear(x, sx, w, sw) if fused else y_eager
                acc = S.int_mm(x, w)
                env = env or _env(torch, dev)
                rows = sample_rows(m, t, rng)
                ri = torch.from_numpy(rows).to(dev)
                f = f"run-{set_}-{name}-m{m}.npz"
                np.savez(scratch / f, rows=rows, x=x[ri].cpu().numpy(), sx=sx_np, acc=acc[ri].cpu().numpy(),
                         y=y[ri].view(torch.int16).cpu().numpy().view(np.uint16),
                         y_eager=y_eager[ri].view(torch.int16).cpu().numpy().view(np.uint16))
                run = {"set": set_, "shape": name, "n": n, "k": k, "m": m, "file": f, "w": meta["w"], "sw": meta["sw"],
                       "w_sha256": meta["w_sha256"], "rows_checked": len(rows), "epilogue": "fused" if fused else "eager"}
                if fused:
                    text = S.fused_ptx(acc, sx, sw)
                    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
                    if digest not in ptx:
                        ptx[digest] = ptx_audit(text)
                        (out / f"fused-{digest}.ptx").write_text(text)
                    run |= {"fused_ptx": digest,
                            "fused_vs_eager_mismatches": int((y.view(torch.int16) != y_eager.view(torch.int16)).sum())}
                    gemm, gemm_reps, _ = _timed(torch, lambda: S.int_mm(x, w), WARM, REPS)
                    ep, ep_reps, _ = _timed(torch, lambda: S.scale(acc, sx, sw), WARM, REPS)
                    fu, fu_reps, _ = _timed(torch, lambda: S.fused_scale(acc, sx, sw), WARM, REPS)
                    lin, lin_reps, last = _timed(torch, lambda: S.linear(x, sx, w, sw), WARM, REPS)
                    line, line_reps, last_eager = _timed(torch, lambda: S.linear(x, sx, w, sw, epilogue="eager"), WARM, REPS)
                    run |= {"gemm_ms": round(gemm, 4), "scale_ms": round(ep, 4), "fused_ms": round(fu, 4),
                            "linear_ms": round(lin, 4), "linear_eager_ms": round(line, 4),
                            "gemm_tops": round(2 * m * n * k / gemm / 1e9, 2), "gemm_reps_ms": gemm_reps,
                            "scale_reps_ms": ep_reps, "fused_reps_ms": fu_reps, "linear_reps_ms": lin_reps,
                            "linear_eager_reps_ms": line_reps,
                            "repeat_equal": bool(torch.equal(last.view(torch.int16), y.view(torch.int16))
                                                 and torch.equal(last_eager.view(torch.int16), y_eager.view(torch.int16)))}
                    del last, last_eager
                runs.append(run)
                print(json.dumps({key: v for key, v in run.items() if not key.endswith("reps_ms")}), flush=True)
                del x, sx, y, y_eager, acc, ri
            del w, sw
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    if fused:
        env["triton"] = sys.modules["triton"].__version__
    (out / "gpu.json").write_text(json.dumps({"env": env, "ptx": ptx, "runs": runs}, indent=1))
    return 0


# ---- verify ----------------------------------------------------------------------------------------------------------

def _nan_word(want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Where the reference is the Definition's NaN and `got` some other NaN word."""
    return (want == NAN_BF16) & ((got & 0x7F80) == 0x7F80) & ((got & 0x7F) != 0)


def classes(words: np.ndarray) -> dict[str, int]:
    e, mant = words & 0x7F80, words & 0x007F
    return {"nan": int(((e == 0x7F80) & (mant != 0)).sum()), "inf": int(((e == 0x7F80) & (mant == 0)).sum()),
            "zero": int(((words & 0x7FFF) == 0).sum()), "subnormal": int(((e == 0) & (mant != 0)).sum())}


def run_verify(a: argparse.Namespace) -> int:
    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G

    scratch, out = Path(a.scratch), Path(a.out)
    rec = json.loads((out / "gpu.json").read_text())
    cpu = rec["env"]["device"] == "cpu"
    results, ok = [], True
    for run in rec["runs"]:
        d = np.load(scratch / run["file"])
        w, sw = np.load(scratch / run["w"], mmap_mode="r"), np.load(scratch / run["sw"])
        rows, x, y, y_eager = d["rows"], d["x"], d["y"], d["y_eager"]
        sx = d["sx"][rows]
        acc = exact_sums(x, w)
        want = epilogue(acc, sx, sw)
        bad, nan_word = y != want, _nan_word(want, y)
        bad_eager = y_eager != want
        rng = np.random.default_rng(seed_of("definition", run["set"], run["shape"], run["m"]))
        q, t = DEFINITION_COORDS // 2, min(len(rows), len(SPECIALS))
        ci = np.concatenate([rng.integers(0, t, q), rng.integers(0, len(rows), q)])
        cj = np.concatenate([rng.integers(0, min(run["n"], len(SPECIALS)), q), rng.integers(0, run["n"], q)])
        coord = bind(G.ScaledMmS8Coordinate, K=run["k"], **G.GEMM_SM90_S8)
        dv = np.asarray(E.evaluate_batch(coord, x[ci].view(np.uint8), sx[ci].view(np.uint32),
                                         np.asarray(w[cj]).view(np.uint8), sw[cj].view(np.uint32),
                                         kernel="reference")).astype(np.uint16)
        dbad, dbad_eager = dv != y[ci, cj], dv != y_eager[ci, cj]
        r = {key: run[key] for key in ("set", "shape", "n", "k", "m", "rows_checked", "epilogue")}
        r |= {"coords_checked": int(want.size), "s32_mismatches": int((d["acc"].astype(np.int64) != acc).sum()),
              "mismatches": int(bad.sum()), "nan_word_mismatches": int((bad & nan_word).sum()),
              "eager_mismatches": int(bad_eager.sum()),
              "definition_coords": len(ci), "definition_vs_twin_mismatches": int((dv != want[ci, cj]).sum()),
              "definition_vs_gpu_mismatches": int(dbad.sum()), "definition_vs_eager_mismatches": int(dbad_eager.sum()),
              "order_sensitive": {o: int((epilogue(acc, sx, sw, o) != want).sum()) for o in ("sw-first", "scales-first")},
              "reference_classes": classes(want), "max_abs_sum": int(np.abs(acc).max()),
              "sums_past_2^24": int((np.abs(acc) >= 1 << 24).sum())}
        r |= {key: run[key] for key in (*TIMES, "gemm_tops", "repeat_equal", "fused_ptx", "fused_vs_eager_mismatches")
              if key in run}
        # torch's CPU cast writes another NaN word (s8_simple's docstring): a CPU run may differ there and nowhere else
        if cpu:
            bad, dbad = bad & ~nan_word, dbad & ~_nan_word(dv, y[ci, cj])
            bad_eager, dbad_eager = bad_eager & ~_nan_word(want, y_eager), dbad_eager & ~_nan_word(dv, y_eager[ci, cj])
        # a GPU run checks the fused epilogue: every word equal to the eager one's, and PTX that passed its audit
        fused_ok = cpu or (run["epilogue"] == "fused" and run.get("fused_vs_eager_mismatches") == 0
                           and rec["ptx"].get(run.get("fused_ptx"), {}).get("ok") is True)
        r["pass"] = bool(not bad.any() and not dbad.any() and not bad_eager.any() and not dbad_eager.any() and fused_ok
                         and r["definition_vs_twin_mismatches"] == 0 and r["s32_mismatches"] == 0
                         and r.get("repeat_equal", True))
        ok &= r["pass"]
        results.append(r)
        print(_line(r), flush=True)
    layers = _layers(results)
    (out / "verify.json").write_text(json.dumps({"env": rec["env"], "ptx": rec.get("ptx", {}), "pass": ok,
                                                 "results": results, "layers": layers}, indent=1))
    text = "\n".join([*map(_line, results), *(_layer_line(v) for v in layers),
                      *(f"ptx {h} {json.dumps(v)}" for h, v in rec.get("ptx", {}).items())])
    (out / "summary.txt").write_text(text + f"\npass {ok}\n")
    print(f"pass {ok}")
    return 0 if ok else 1


def _layers(results: list[dict]) -> list[dict]:
    """Each (set, m)'s times summed over `LAYER`'s linears, where every one of them was timed."""
    by, out = {(r["set"], r["shape"], r["m"]): r for r in results if "gemm_ms" in r}, []
    for set_, m in dict.fromkeys((r["set"], r["m"]) for r in results):
        rs = [by.get((set_, s, m)) for s in LAYER]
        if all(rs):
            out.append({"set": set_, "m": m} | {key: round(sum(r[key] for r in rs), 3) for key in TIMES})
    return out


def _layer_line(v: dict) -> str:
    return (f"layer  {v['set']:6} m={v['m']:<5} gemm {v['gemm_ms']:.3f} eager-epilogue {v['scale_ms']:.3f} fused-epilogue "
            f"{v['fused_ms']:.3f} linear {v['linear_ms']:.3f} (eager {v['linear_eager_ms']:.3f}) ms")


def _line(r: dict) -> str:
    t = (f" gemm {r['gemm_ms']:.3f} ms ({r['gemm_tops']:.0f} TOPS) eager-epilogue {r['scale_ms']:.3f} fused-epilogue "
         f"{r['fused_ms']:.3f} linear {r['linear_ms']:.3f} (eager {r['linear_eager_ms']:.3f})") if "gemm_ms" in r else ""
    f = (f" fused-vs-eager {r['fused_vs_eager_mismatches']}/{r['m'] * r['n']}"
         if "fused_vs_eager_mismatches" in r else "")
    return (f"{r['set']:6} {r['shape']:4} n={r['n']:<5} k={r['k']:<5} m={r['m']:<5} {r['epilogue']} rows "
            f"{r['rows_checked']}/{r['m']} coords {r['coords_checked']} mismatches {r['mismatches']} (s32 "
            f"{r['s32_mismatches']}, nan-word {r['nan_word_mismatches']}, eager {r['eager_mismatches']}) definition "
            f"{r['definition_coords']}: {r['definition_vs_gpu_mismatches']} (eager {r['definition_vs_eager_mismatches']}) "
            f"order-sensitive {r['order_sensitive']['sw-first']}/{r['order_sensitive']['scales-first']}{f}{t}")


def _shapes(text: str) -> dict[str, tuple[int, int]]:
    out = {}
    for item in text.split(","):
        name, _, nk = item.partition("=")
        out[name] = tuple(int(v) for v in nk.split(":")) if nk else SHAPES[name]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("weights", "gpu", "verify"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", help="gpu and verify: the run's $OUT")
    ap.add_argument("--sets", type=lambda s: s.split(","), default=list(SETS))
    ap.add_argument("--shapes", type=_shapes, default=dict(SHAPES), help="names of SHAPES, or NAME=N:K, comma-separated")
    ap.add_argument("--ms", type=lambda s: [int(v) for v in s.split(",")], default=list(MS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights-module", help="verity_toy/weights.py, where it isn't importable")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    return {"weights": write_weights, "gpu": run_gpu, "verify": run_verify}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
