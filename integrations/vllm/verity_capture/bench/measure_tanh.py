#!/usr/bin/env python3
"""Measure `tanh.approx.f32` (MUFU.TANH) exhaustively over one contiguous range of f32 bit patterns.

Why: FA2 with `Is_softcap=true` (vLLM + Gemma-2, `attn_logit_softcapping=50`) applies `cutlass::fast_tanh(s * scale / 50)` =
one `tanh.approx.f32` to every f32 score before the exp2 softmax (COVERAGE-20260917 §5 M10).  The hardware function is up to
133 ulp from RN32(tanh) and non-monotonic (ov-dense probe, samples only), so the registry primitive `MufuTanh` must be a
*measured* table + reduction rule like `MufuEx2Ftz` / `MufuRcpFtz` — this script produces one shard of that table and the
facts the reduction rule rests on (odd symmetry, saturation threshold, small-|x| identity, monotonicity, residual range).

    /workspace/venv-cu129/bin/python verity_capture/bench/measure_tanh.py --lo 0x3F000000 --hi 0x3F400000 \
        --out /workspace/cp/cov/tanh/shard_3F00.npy [--chunk 16777216] [--store u32|none] [--neg check|none] [--xz-chunks 4]

Inputs are the uint32 words `lo <= w < hi` in ascending order (hex or decimal).  Outputs:
  * `<out>`           .npy, uint32 words of `tanh.approx.f32(x)` in the same order (`--store u32`; `--store none` keeps only the summary);
  * `<out>.json`      the summary: hardware (nvidia-smi name / driver / compute cap), software versions, the PTX, sha256 of the word
                      stream and of the .npy file, ulp histogram vs RN32(tanh) (= f32(np.tanh(f64(x))), same reference as the ov-dense
                      probe; double rounding is possible in ~2^-29 of words and is NOT corrected — that is why the storage plan
                      encodes the hardware words themselves, not the residual), signed residual range, monotonicity inversions
                      (with the deepest drop), odd-symmetry violations (`--neg check` also runs the negated words), saturation
                      (`== ±1.0`) and identity (`y == x`) counts with the boundary words, NaN behaviour, the fixed specials list,
                      compressed sizes of three encodings (u32 raw / int16 residual / int8-or-int16 first differences of the ordered
                      output ints) on the first `--xz-chunks` chunks, and per-phase timings.

`--selftest` runs the torch-free analysis half on a synthetic table (laptop unit test; no GPU, no torch).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import platform
import subprocess
import sys
import time

import numpy as np

SCHEMA = "verity/measure_tanh/v1"
PTX = "tanh.approx.f32 $0, $1;"
ONE = 0x3F800000
SIGN = 0x80000000
EXP_MASK = 0x7F800000
BLOCK = 1024

np.seterr(all="ignore")


# ---------------------------------------------------------------------------------------------------------------------
# torch-free analysis
# ---------------------------------------------------------------------------------------------------------------------


def ordered(words: np.ndarray) -> np.ndarray:
    """Monotone int64 image of f32 words: w >= 0 -> w, negative -> -(w & 0x7FFFFFFF) (so ±0 -> 0, ordering = float ordering)."""
    w = words.astype(np.int64)
    return np.where(w & SIGN, -(w & 0x7FFFFFFF), w)


def rn32_tanh(words: np.ndarray) -> np.ndarray:
    """Reference: f32(np.tanh(f64(x))) as uint32 words.  NaN -> canonical 0x7FC00000 (comparison excludes NaN anyway)."""
    x = words.view(np.float32).astype(np.float64)
    y = np.tanh(x).astype(np.float32).view(np.uint32).copy()
    y[np.isnan(words.view(np.float32))] = 0x7FC00000
    return y


def is_nan_words(w: np.ndarray) -> np.ndarray:
    return ((w & EXP_MASK) == EXP_MASK) & ((w & 0x7FFFFF) != 0)


def is_inf_words(w: np.ndarray) -> np.ndarray:
    return (w & 0x7FFFFFFF) == EXP_MASK


ULP_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255), (256, 1 << 62)]


def bucket_hist(d: np.ndarray) -> dict:
    out = {}
    for lo, hi in ULP_BUCKETS:
        key = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 1 << 62 else f"{lo}-{hi}")
        out[key] = int(((d >= lo) & (d <= hi)).sum())
    return out


class Accumulator:
    """Chunk-wise statistics over an ascending word range (carries the boundary sample for monotonicity / diffs)."""

    def __init__(self) -> None:
        self.n = 0
        self.n_compared = 0
        self.hist: dict[str, int] = {k: 0 for k in bucket_hist(np.zeros(0, np.int64))}
        self.max_ulp = 0
        self.max_ulp_examples: list[dict] = []
        self.res_min = 0
        self.res_max = 0
        self.res_hist_signed = {"neg": 0, "zero": 0, "pos": 0}
        self.inversions = 0
        self.max_drop = 0
        self.max_drop_example: dict | None = None
        self.sym_violations = 0
        self.sym_examples: list[dict] = []
        self.sym_checked = 0
        self.n_one = 0
        self.first_one_word: int | None = None
        self.last_not_one_word: int | None = None
        self.n_identity = 0
        self.first_non_identity_word: int | None = None
        self.last_identity_word: int | None = None
        self.nan_in = 0
        self.nan_out = 0
        self.nan_out_words: set[int] = set()
        self.inf_in = 0
        self.dy_min = 0
        self.dy_max = 0
        self._prev_x: int | None = None
        self._prev_y: int | None = None
        self._prev_oy: int | None = None
        self.sha_words = hashlib.sha256()

    def add(self, x: np.ndarray, y: np.ndarray, yneg: np.ndarray | None) -> None:
        x = np.ascontiguousarray(x, dtype=np.uint32)
        y = np.ascontiguousarray(y, dtype=np.uint32)
        n = int(x.size)
        self.n += n
        self.sha_words.update(y.tobytes())
        xnan = is_nan_words(x)
        ynan = is_nan_words(y)
        self.nan_in += int(xnan.sum())
        self.nan_out += int(ynan.sum())
        if ynan.any():
            self.nan_out_words |= set(int(v) for v in np.unique(y[ynan])[:8])
        self.inf_in += int(is_inf_words(x).sum())
        ref = rn32_tanh(x)
        both = ~xnan & ~ynan
        oy = ordered(y)
        oref = ordered(ref)
        res = (oy - oref)[both]
        d = np.abs(res)
        self.n_compared += int(both.sum())
        for k, v in bucket_hist(d).items():
            self.hist[k] += v
        if d.size:
            m = int(d.max())
            if m >= self.max_ulp:
                idx = np.flatnonzero(both)[np.argsort(-d)[:3]]
                ex = [{"x": f"0x{int(x[i]):08X}", "x_f32": float(x[i:i + 1].view(np.float32)[0]), "hw": f"0x{int(y[i]):08X}",
                       "rn32": f"0x{int(ref[i]):08X}", "signed_ulp": int(oy[i] - oref[i])} for i in idx]
                self.max_ulp_examples = (ex + self.max_ulp_examples)[:3] if m > self.max_ulp else self.max_ulp_examples
                self.max_ulp = m
            self.res_min = min(self.res_min, int(res.min()))
            self.res_max = max(self.res_max, int(res.max()))
            self.res_hist_signed["neg"] += int((res < 0).sum())
            self.res_hist_signed["zero"] += int((res == 0).sum())
            self.res_hist_signed["pos"] += int((res > 0).sum())
        # monotonicity + first differences, across the chunk boundary
        xf = x.view(np.float32).astype(np.float64)
        if self._prev_x is not None:
            xf_all = np.concatenate([[np.array([self._prev_x], np.uint32).view(np.float32).astype(np.float64)[0]], xf])
            oy_all = np.concatenate([[self._prev_oy], oy])
            nan_all = np.concatenate([[False], xnan | ynan])
        else:
            xf_all, oy_all, nan_all = xf, oy, xnan | ynan
        if oy_all.size >= 2:
            ok = ~nan_all[1:] & ~nan_all[:-1]
            dx = np.sign(xf_all[1:] - xf_all[:-1])
            dy = oy_all[1:] - oy_all[:-1]
            inv = (dx * dy < 0) & ok
            self.inversions += int(inv.sum())
            if inv.any():
                drops = np.abs(dy[inv])
                j = int(np.argmax(drops))
                if int(drops[j]) > self.max_drop:
                    self.max_drop = int(drops[j])
                    i = np.flatnonzero(inv)[j]  # index into dy; pair (i, i+1) of *_all
                    off = 1 if self._prev_x is not None else 0
                    xi = self._prev_x if (i - off) < 0 else int(x[i - off])
                    yi = self._prev_y if (i - off) < 0 else int(y[i - off])
                    xj = int(x[i + 1 - off])
                    self.max_drop_example = {"x_lo": f"0x{xi:08X}", "x_hi": f"0x{xj:08X}", "hw_lo": f"0x{yi:08X}",
                                             "hw_hi": f"0x{int(y[i + 1 - off]):08X}", "drop_ulp": int(drops[j])}
            dyo = dy[ok]
            if dyo.size:
                self.dy_min = min(self.dy_min, int(dyo.min()))
                self.dy_max = max(self.dy_max, int(dyo.max()))
        self._prev_x = int(x[-1])
        self._prev_y = int(y[-1])
        self._prev_oy = int(oy[-1])
        # symmetry
        if yneg is not None:
            yneg = np.ascontiguousarray(yneg, dtype=np.uint32)
            fin = ~xnan
            viol = fin & (yneg != (y ^ SIGN)) & ~(ynan & is_nan_words(yneg))
            self.sym_checked += int(fin.sum())
            self.sym_violations += int(viol.sum())
            if viol.any() and len(self.sym_examples) < 4:
                for i in np.flatnonzero(viol)[:4 - len(self.sym_examples)]:
                    self.sym_examples.append({"x": f"0x{int(x[i]):08X}", "hw(x)": f"0x{int(y[i]):08X}", "hw(-x)": f"0x{int(yneg[i]):08X}"})
        # saturation and identity
        one = (y & 0x7FFFFFFF) == ONE
        self.n_one += int(one.sum())
        if one.any():
            f = int(x[np.flatnonzero(one)[0]])
            self.first_one_word = f if self.first_one_word is None else min(self.first_one_word, f)
        not_one = ~one & ~xnan
        if not_one.any():
            self.last_not_one_word = int(x[np.flatnonzero(not_one)[-1]])
        ident = (y == x) & ~xnan
        self.n_identity += int(ident.sum())
        if ident.any():
            self.last_identity_word = int(x[np.flatnonzero(ident)[-1]])
        non_ident = ~ident & ~xnan
        if non_ident.any():
            f = int(x[np.flatnonzero(non_ident)[0]])
            self.first_non_identity_word = f if self.first_non_identity_word is None else min(self.first_non_identity_word, f)

    def summary(self) -> dict:
        hx = lambda v: None if v is None else f"0x{v:08X}"  # noqa: E731
        fl = lambda v: None if v is None else float(np.array([v], np.uint32).view(np.float32)[0])  # noqa: E731
        return {
            "n_words": self.n, "n_compared_vs_rn32": self.n_compared, "sha256_words_le_u32": self.sha_words.hexdigest(),
            "vs_rn32": {"ulp_histogram": self.hist, "max_ulp": self.max_ulp, "max_ulp_examples": self.max_ulp_examples,
                        "mismatch_words": self.n_compared - self.hist.get("0", 0),
                        "mismatch_fraction": (self.n_compared - self.hist.get("0", 0)) / max(1, self.n_compared),
                        "signed_residual_min": self.res_min, "signed_residual_max": self.res_max, "residual_sign_counts": self.res_hist_signed,
                        "residual_fits_int8": -128 <= self.res_min and self.res_max <= 127,
                        "residual_fits_int16": -32768 <= self.res_min and self.res_max <= 32767},
            "monotonicity": {"inversions": self.inversions, "max_drop_ulp": self.max_drop, "max_drop_example": self.max_drop_example,
                             "definition": "adjacent words (ascending), sign(dx_float) * (ordered(y[i+1]) - ordered(y[i])) < 0"},
            "first_difference": {"dy_min": self.dy_min, "dy_max": self.dy_max, "fits_int8": -128 <= self.dy_min and self.dy_max <= 127,
                                 "definition": "ordered(y[i+1]) - ordered(y[i]) over adjacent non-NaN words"},
            "odd_symmetry": {"checked": self.sym_checked, "violations": self.sym_violations, "examples": self.sym_examples,
                             "definition": "hw(x ^ 0x80000000) == hw(x) ^ 0x80000000 bitwise (NaN outputs compared as NaN-vs-NaN)"},
            "saturation": {"n_words_abs_one": self.n_one, "first_word_hw_is_one": hx(self.first_one_word), "first_x_hw_is_one": fl(self.first_one_word),
                           "last_word_hw_not_one": hx(self.last_not_one_word), "last_x_hw_not_one": fl(self.last_not_one_word)},
            "identity": {"n_words_hw_eq_x": self.n_identity, "last_word_hw_eq_x": hx(self.last_identity_word), "last_x_hw_eq_x": fl(self.last_identity_word),
                         "first_word_hw_ne_x": hx(self.first_non_identity_word), "first_x_hw_ne_x": fl(self.first_non_identity_word)},
            "nan": {"nan_in": self.nan_in, "nan_out": self.nan_out, "nan_out_words": [f"0x{w:08X}" for w in sorted(self.nan_out_words)], "inf_in": self.inf_in},
        }


def encodings_xz(x: np.ndarray, y: np.ndarray, preset: int) -> dict:
    """Compressed sizes of three storage encodings of one chunk (bytes)."""
    out = {}
    t = time.time()
    out["u32_raw_xz"] = len(lzma.compress(np.ascontiguousarray(y, np.uint32).tobytes(), preset=preset))
    res = ordered(y) - ordered(rn32_tanh(x))
    if -32768 <= res.min() and res.max() <= 32767:
        out["residual_int16_xz"] = len(lzma.compress(res.astype(np.int16).tobytes(), preset=preset))
    else:
        out["residual_int16_xz"] = None
    oy = ordered(y)
    dy = oy[1:] - oy[:-1]
    if dy.size and -128 <= dy.min() and dy.max() <= 127:
        out["diff_int8_xz"] = len(lzma.compress(dy.astype(np.int8).tobytes(), preset=preset))
        out["diff_int16_xz"] = None
    elif dy.size:
        out["diff_int8_xz"] = None
        out["diff_int16_xz"] = len(lzma.compress(dy.astype(np.int16).tobytes(), preset=preset)) if -32768 <= dy.min() and dy.max() <= 32767 else None
    out["n_words"] = int(y.size)
    out["seconds"] = round(time.time() - t, 3)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# GPU half (torch + triton; imported lazily)
# ---------------------------------------------------------------------------------------------------------------------


def _gpu():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _tanh_approx_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        y = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(y_ptr + offs, y, mask=m)

    def tanh_approx_words(words: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(words, np.uint32).view(np.int32)).cuda().view(torch.float32)
        y = torch.empty_like(x)
        n = x.numel()
        _tanh_approx_kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
        torch.cuda.synchronize()
        return y.view(torch.int32).cpu().numpy().view(np.uint32)

    return torch, triton, tanh_approx_words


def hardware_info() -> dict:
    info: dict = {"hostname": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = q.stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi"] = f"unavailable: {e}"
    try:
        import torch
        import triton
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["triton"] = triton.__version__
        info["device"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"unavailable: {e}"
    return info


SPECIALS = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x33800000, 0x39800000, 0x3F800000, 0xBF800000,
            0x40FFFFFF, 0x41000000, 0x41000001, 0xC1000000, 0x41800000, 0x42C80000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
            0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FFFFFFF, 0x7FBFFFFF]


def ordered_inj(words: np.ndarray) -> np.ndarray:
    """Injective monotone int64 image for the STORAGE encoding: w >= 0 -> w; negative -> -(w & 0x7FFFFFFF) - 1 (so -0 -> -1 and
    the sign of zero survives the round trip; `ordered` maps both zeros to 0 and is used for ulp distances only)."""
    w = words.astype(np.int64)
    return np.where(w & SIGN, -(w & 0x7FFFFFFF) - 1, w)


def words_from_ordered(o: np.ndarray) -> np.ndarray:
    """Inverse of `ordered_inj`."""
    return np.where(o < 0, (-(o + 1)) | SIGN, o).astype(np.uint32)


COMPACT_FORMAT = "verity/mufu-tanh-shard/xz-blocks/v1"


class CompactWriter:
    """The storage encoding of a shard: independently xz-compressed BLOCKS of `block_words` words, each block = first differences of
    `ordered_inj(y)` (int8 when they fit, else int16/int32) plus the block's first value — random access by block (a consumer decodes
    only the ~20 KB blocks it touches), reconstruction = cumsum -> `words_from_ordered`, like `MufuTables`' `mufu_ex2_delta_int8.xz`.
    The header (this object's `header()`) is written next to the file as `<path>.json` and embedded in the summary."""

    def __init__(self, path: str, lo: int, hi: int, block_words: int, preset: int) -> None:
        self.path, self.lo, self.hi, self.block_words, self.preset = path, lo, hi, block_words, preset
        self.blocks: list[dict] = []
        self.f = open(path, "wb")
        self.offset = 0
        self.sha = hashlib.sha256()

    def add_words(self, start: int, y: np.ndarray) -> None:
        """`y` = the words for absolute range [start, start + len(y)); must be block-aligned at `start` (start - lo) % block_words == 0."""
        if (start - self.lo) % self.block_words:
            raise ValueError("chunk not aligned to block_words")
        o_all = ordered_inj(np.ascontiguousarray(y, np.uint32))
        for b0 in range(0, o_all.shape[0], self.block_words):
            o = o_all[b0:b0 + self.block_words]
            d = np.diff(o)
            dtype = "int8"
            for cand in ("int8", "int16", "int32"):
                dtype = cand
                if d.size == 0 or (np.iinfo(cand).min <= d.min() and d.max() <= np.iinfo(cand).max):
                    break
            raw = lzma.compress(d.astype(dtype).tobytes(), preset=self.preset)
            self.f.write(raw)
            self.sha.update(raw)
            self.blocks.append({"start": f"0x{start + b0:08X}", "n": int(o.shape[0]), "first_ordered": int(o[0]), "dtype": dtype,
                                "offset": self.offset, "bytes": len(raw)})
            self.offset += len(raw)

    def close(self) -> dict:
        self.f.close()
        h = self.header()
        with open(self.path + ".json", "w") as f:
            json.dump(h, f, indent=1)
        return h

    def header(self) -> dict:
        return {"format": COMPACT_FORMAT, "path": self.path, "lo": f"0x{self.lo:08X}", "hi": f"0x{self.hi:08X}", "n_words": self.hi - self.lo,
                "block_words": self.block_words, "preset": self.preset, "bytes": self.offset, "sha256_file": self.sha.hexdigest(),
                "blocks": self.blocks}


def compact_read_block(path: str, header: dict, k: int) -> np.ndarray:
    """Words of block k of a compact shard file."""
    b = header["blocks"][k]
    with open(path, "rb") as f:
        f.seek(b["offset"])
        raw = f.read(b["bytes"])
    d = np.frombuffer(lzma.decompress(raw), dtype=np.dtype(b["dtype"])).astype(np.int64)
    if d.shape[0] != b["n"] - 1:
        raise ValueError(f"{path} block {k}: expected {b['n'] - 1} diffs, got {d.shape[0]}")
    o = np.empty(b["n"], dtype=np.int64)
    o[0] = b["first_ordered"]
    np.cumsum(d, out=o[1:])
    o[1:] += b["first_ordered"]
    return words_from_ordered(o)


def compact_read(path: str, header: dict) -> np.ndarray:
    return np.concatenate([compact_read_block(path, header, k) for k in range(len(header["blocks"]))])


def run_gpu(lo: int, hi: int, out: str, chunk: int, store: str, neg: str, xz_chunks: int, xz_preset: int, block_words: int = 1 << 22) -> dict:
    torch, triton, tanh_words = _gpu()
    t_all = time.time()
    n = hi - lo
    acc = Accumulator()
    tim = {"gpu": 0.0, "analysis": 0.0, "xz": 0.0, "write": 0.0, "compact": 0.0, "validate": 0.0}
    xz: list[dict] = []
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    mm = None
    if store in ("u32", "both"):
        mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint32, shape=(n,))
    want_compact = store in ("diff8", "both")
    cw = None
    if want_compact:
        if chunk % block_words:
            raise SystemExit(f"--chunk {chunk} must be a multiple of the compact block size {block_words}")
        cpath = out[:-4] + ".diff.xz" if out.endswith(".npy") else out + ".diff.xz"
        cw = CompactWriter(cpath, lo, hi, block_words, xz_preset)
    pos = 0
    while pos < n:
        m = min(chunk, n - pos)
        x = np.arange(lo + pos, lo + pos + m, dtype=np.int64).astype(np.uint32)
        t = time.time()
        y = tanh_words(x)
        yneg = tanh_words(x ^ np.uint32(SIGN)) if neg == "check" else None
        tim["gpu"] += time.time() - t
        t = time.time()
        acc.add(x, y, yneg)
        tim["analysis"] += time.time() - t
        if len(xz) < xz_chunks:
            t = time.time()
            xz.append(encodings_xz(x, y, xz_preset))
            tim["xz"] += time.time() - t
        if mm is not None:
            t = time.time()
            mm[pos:pos + m] = y
            tim["write"] += time.time() - t
        if cw is not None:
            t = time.time()
            cw.add_words(lo + pos, y)
            tim["compact"] += time.time() - t
        pos += m
        print(f"[measure_tanh] {pos}/{n} words  gpu={tim['gpu']:.1f}s analysis={tim['analysis']:.1f}s xz={tim['xz']:.1f}s", flush=True)
    sha_npy = None
    npy_bytes = None
    mm_written = mm is not None
    if mm is not None:
        t = time.time()
        mm.flush()
        del mm
        h = hashlib.sha256()
        with open(out, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        sha_npy = h.hexdigest()
        npy_bytes = os.path.getsize(out)
        tim["write"] += time.time() - t
    compact = None
    if cw is not None:
        t = time.time()
        compact = cw.close()
        tim["compact"] += time.time() - t
        # validation: reconstruct every block from the compact file and compare the word stream's sha256 with the measured one
        t = time.time()
        h = hashlib.sha256()
        for k in range(len(compact["blocks"])):
            h.update(compact_read_block(cw.path, compact, k).tobytes())
        compact["reconstructed_sha256_words_le_u32"] = h.hexdigest()
        compact["reconstruction_matches"] = h.hexdigest() == acc.sha_words.hexdigest()
        tim["validate"] += time.time() - t
    # specials (always; tiny)
    sp = np.array(SPECIALS, dtype=np.uint32)
    sy = tanh_words(sp)
    sref = rn32_tanh(sp)
    specials = [{"x": f"0x{int(a):08X}", "x_f32": float(np.array([a], np.uint32).view(np.float32)[0]), "hw": f"0x{int(b):08X}",
                 "hw_f32": float(np.array([b], np.uint32).view(np.float32)[0]), "rn32": f"0x{int(c):08X}"} for a, b, c in zip(sp, sy, sref)]
    summ = acc.summary()
    xz_total_words = sum(e["n_words"] for e in xz)
    xz_summary = {"chunks": xz, "preset": xz_preset, "words_sampled": xz_total_words}
    for key in ("u32_raw_xz", "residual_int16_xz", "diff_int8_xz", "diff_int16_xz"):
        vals = [e[key] for e in xz]
        xz_summary[f"{key}_bytes_per_word"] = (sum(vals) / xz_total_words) if xz_total_words and all(v is not None for v in vals) else None
    tim["total"] = time.time() - t_all
    return {
        "schema": SCHEMA, "ptx": PTX, "kernel": "triton inline_asm_elementwise, BLOCK=1024, one MUFU.TANH per element",
        "range": {"lo": f"0x{lo:08X}", "hi": f"0x{hi:08X}", "n_words": n, "lo_f32": float(np.array([lo], np.uint32).view(np.float32)[0]),
                  "hi_minus_1_f32": float(np.array([hi - 1], np.uint32).view(np.float32)[0]), "chunk": chunk},
        "hardware": hardware_info(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "store": store, "npy": {"path": out if mm_written else None, "sha256_file": sha_npy, "bytes": npy_bytes}, "compact": compact,
        "neg_checked": neg == "check", "measured": summ, "specials": specials, "encodings_xz": xz_summary, "seconds": {k: round(v, 3) for k, v in tim.items()},
    }


# ---------------------------------------------------------------------------------------------------------------------
# selftest (laptop, torch-free)
# ---------------------------------------------------------------------------------------------------------------------


def selftest() -> int:
    lo, hi = 0x3F000000, 0x3F000000 + 20000
    x = np.arange(lo, hi, dtype=np.int64).astype(np.uint32)
    y = rn32_tanh(x).copy()
    # inject: +5 ulp at 100 words, -3 ulp at 50 words, one drop of 40 ulp at word 7000, NaN input at the end
    oy = ordered(y)
    oy[100:200] += 5
    oy[300:350] -= 3
    oy[7000] -= 40
    y = np.where(oy < 0, (-oy) | SIGN, oy).astype(np.uint32)
    x = np.concatenate([x, np.array([0x7FC00000], np.uint32)])
    y = np.concatenate([y, np.array([0x7FFFFFFF], np.uint32)])
    yneg = y ^ np.uint32(SIGN)
    yneg[10] ^= 1  # one symmetry violation
    acc = Accumulator()
    acc.add(x[:9000], y[:9000], yneg[:9000])
    acc.add(x[9000:], y[9000:], yneg[9000:])
    s = acc.summary()
    h = s["vs_rn32"]["ulp_histogram"]
    assert s["n_words"] == 20001 and s["n_compared_vs_rn32"] == 20000, s
    assert h["4-7"] == 100 and h["3"] == 50 and h["32-63"] == 1 and s["vs_rn32"]["max_ulp"] == 40, h
    assert s["vs_rn32"]["signed_residual_min"] == -40 and s["vs_rn32"]["signed_residual_max"] == 5
    assert s["odd_symmetry"]["violations"] == 1 and s["odd_symmetry"]["checked"] == 20000, s["odd_symmetry"]
    assert s["monotonicity"]["inversions"] >= 3 and 36 <= s["monotonicity"]["max_drop_ulp"] <= 40, s["monotonicity"]  # 40 minus RN32's own slope
    assert s["nan"] == {"nan_in": 1, "nan_out": 1, "nan_out_words": ["0x7FFFFFFF"], "inf_in": 0}, s["nan"]
    assert s["saturation"]["n_words_abs_one"] == 0 and s["identity"]["n_words_hw_eq_x"] == 0
    # identity / saturation detection on a synthetic table
    acc2 = Accumulator()
    x2 = np.array([0x33800000, 0x33800001, 0x40FFFFFF, 0x41000000, 0x41000001], np.uint32)
    y2 = np.array([0x33800000, 0x33800001, 0x3F7FFFFF, 0x3F800000, 0x3F800000], np.uint32)
    acc2.add(x2, y2, None)
    s2 = acc2.summary()
    assert s2["identity"] == {"n_words_hw_eq_x": 2, "last_word_hw_eq_x": "0x33800001", "last_x_hw_eq_x": float(np.array([0x33800001], np.uint32).view(np.float32)[0]),
                              "first_word_hw_ne_x": "0x40FFFFFF", "first_x_hw_ne_x": float(np.array([0x40FFFFFF], np.uint32).view(np.float32)[0])}, s2["identity"]
    assert s2["saturation"]["first_word_hw_is_one"] == "0x41000000" and s2["saturation"]["last_word_hw_not_one"] == "0x40FFFFFF", s2["saturation"]
    # chunk-boundary monotonicity: a drop exactly across the split must be seen
    acc3 = Accumulator()
    x3 = np.arange(0x3F000000, 0x3F000000 + 8, dtype=np.int64).astype(np.uint32)
    y3 = np.arange(0x3F000000, 0x3F000000 + 8, dtype=np.int64).astype(np.uint32)
    y3[4] -= 10
    acc3.add(x3[:4], y3[:4], None)
    acc3.add(x3[4:], y3[4:], None)
    m3 = acc3.summary()["monotonicity"]  # slope-1 table, y[4] pulled down 10: the adjacent drop is 9 and straddles the chunk split
    assert m3["inversions"] == 1 and m3["max_drop_ulp"] == 9 and m3["max_drop_example"]["x_lo"] == "0x3F000003", m3
    enc = encodings_xz(x[:9000], y[:9000], 6)
    assert enc["u32_raw_xz"] > 0 and enc["residual_int16_xz"] > 0 and (enc["diff_int8_xz"] or enc["diff_int16_xz"]), enc
    # compact encoding round trip: chunked adds, blocks of 1000 words, positive / negative / sign-crossing (NaN, -0, subnormal) ranges
    import tempfile
    for lo4, hi4 in ((0x3F000000, 0x3F000000 + 5000), (0xBF000000, 0xBF000000 + 5000), (0x7FFFFC00, 0x80000400)):
        x4 = np.arange(lo4, hi4, dtype=np.int64).astype(np.uint32)
        y4 = rn32_tanh(x4)
        y4[is_nan_words(x4)] = 0x7FFFFFFF
        with tempfile.TemporaryDirectory() as td:
            cw = CompactWriter(os.path.join(td, "t.diff.xz"), lo4, hi4, 1000, 6)
            cw.add_words(lo4, y4[:2000])
            cw.add_words(lo4 + 2000, y4[2000:])
            hdr = cw.close()
            assert json.load(open(os.path.join(td, "t.diff.xz.json")))["blocks"] == hdr["blocks"]
            rec = compact_read(os.path.join(td, "t.diff.xz"), hdr)
            blk2 = compact_read_block(os.path.join(td, "t.diff.xz"), hdr, 2)
        assert rec.shape == y4.shape and np.array_equal(rec, y4), (hex(lo4), hdr)
        nblk = -(-(hi4 - lo4) // 1000)
        assert np.array_equal(blk2, y4[2000:3000]) and len(hdr["blocks"]) == nblk and hdr["blocks"][0]["dtype"] in ("int8", "int16", "int32"), hdr
    allw = np.array([0, SIGN, 1, SIGN | 1, 0x7FFFFFFF, 0xFFFFFFFF, 0x3F800000, 0xBF800000], np.uint32)
    assert np.array_equal(words_from_ordered(ordered_inj(allw)), allw)  # injective incl. the sign of zero
    # writer -> `verity_vllm.registry.prims.MufuTanh` loader contract: a synthetic shard directory (compact file + header + summary)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from verity_vllm.registry import prims as P
    lo5, n5 = 0x3F000000, 3000
    x5 = np.arange(lo5, lo5 + n5, dtype=np.int64).astype(np.uint32)
    y5 = rn32_tanh(x5).copy()
    y5[7] ^= 1  # a "hardware" word that is not RN32
    saved_rules = dict(P.MUFU_TANH_RULES)
    try:
        for sym in (True, False):
            with tempfile.TemporaryDirectory() as td:
                cw = CompactWriter(os.path.join(td, "s.diff.xz"), lo5, lo5 + n5, 1024, 6)
                cw.add_words(lo5, y5)
                cw.close()
                with open(os.path.join(td, "s.npy.json"), "w") as f:
                    json.dump({"job": "selftest", "measured": {"sha256_words_le_u32": hashlib.sha256(y5.tobytes()).hexdigest(),
                                                               "odd_symmetry": {"checked": n5 if sym else 0, "violations": 0}}}, f)
                P.mufu_tanh_use_tables(td)
                cov = P.mufu_tanh_coverage()
                assert cov["shards"][0]["odd_symmetry_verified"] is sym and cov["shards"][0]["n_words"] == n5, cov
                assert P.MufuTanh.evaluate(lo5 + 7) == int(y5[7]) and P.MufuTanh.evaluate(lo5 + n5 - 1) == int(y5[n5 - 1])
                neg_w = (lo5 + 7) | SIGN
                if sym:
                    assert P.MufuTanh.evaluate(neg_w) == int(y5[7]) ^ SIGN
                else:
                    try:
                        P.MufuTanh.evaluate(neg_w)
                        raise AssertionError("negative word answered without a verified odd symmetry")
                    except P.MufuTanhUnmeasured:
                        pass
                for w in (lo5 + n5, lo5 - 1):
                    try:
                        P.MufuTanh.evaluate(w)
                        raise AssertionError(f"word 0x{w:08X} outside the shard answered")
                    except P.MufuTanhUnmeasured:
                        pass
                # a corrupted compact file is rejected at load (sha256 of the file)
                with open(os.path.join(td, "s.diff.xz"), "r+b") as f:
                    f.seek(3)
                    f.write(b"\x00")
                P.mufu_tanh_use_tables(td)
                try:
                    P.MufuTanh.evaluate(lo5)
                    raise AssertionError("corrupt shard accepted")
                except P.MufuTanhUnmeasured as e:
                    assert "sha256" in str(e), e
        # the in-tree default table (no override, env unset) answers every word class; the manifest sample matches bitwise
        P._TANH_TABLES_OVERRIDE.clear()
        P._tanh_shards.cache_clear()
        env_saved = os.environ.pop(P.MUFU_TANH_TABLES_ENV, None)
        try:
            man = P.mufu_tanh_table_manifest()
            for xh, yh in man["sample"]["pairs"]:
                assert P.MufuTanh.evaluate(int(xh, 16)) == int(yh, 16), (xh, yh)
            for w in (0x00000000, 0x80000000, 0x00000001, 0x3B800000, 0x3B800001, 0x3F000000, 0xBF800000, 0x40FFFFFF, 0x41000000, 0x7F800000,
                      0xFF800000, 0x7FC00000, 0xFFFFFFFF):
                P.MufuTanh.evaluate(w)
        finally:
            if env_saved is not None:
                os.environ[P.MUFU_TANH_TABLES_ENV] = env_saved
    finally:
        P.mufu_tanh_use_tables(None)
        P._TANH_TABLES_OVERRIDE.clear()
        P._tanh_shards.cache_clear()
        P.MUFU_TANH_RULES.clear()
        P.MUFU_TANH_RULES.update(saved_rules)
    print("measure_tanh selftest: ok", json.dumps({"hist": h, "enc": enc}))
    return 0


def conform(tables: str | None, summaries: list[str], npy: str | None, n: int, seed: int) -> int:
    """Word-level conformance of `verity_vllm.registry.prims.MufuTanh` against recorded evidence: every `specials` / max-ulp / max-drop
    word of the given summaries must evaluate to the recorded hardware word when inside the evaluator's coverage (and raise
    `MufuTanhUnmeasured` when outside); with `--conform-npy`, `n` random words of that raw shard (and their sign flips) must match.
    Torch-free; runs on the laptop once the compact shards are archived, or on the worker that holds them."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from verity_vllm.registry import prims as P
    if tables:
        P.mufu_tanh_use_tables(tables)
    cov = P.mufu_tanh_coverage()
    rep: dict = {"coverage": cov, "checked": 0, "mismatches": [], "outside_raised": 0, "outside_answered": []}

    def check(xw: int, hw: int, src: str) -> None:
        try:
            got = P.MufuTanh.evaluate(xw)
        except P.MufuTanhUnmeasured:
            rep["outside_raised"] += 1
            return
        rep["checked"] += 1
        if got != hw:
            rep["mismatches"].append({"x": f"0x{xw:08X}", "hw": f"0x{hw:08X}", "got": f"0x{got:08X}", "src": src})

    for sp in summaries:
        with open(sp) as f:
            s = json.load(f)
        for e in s.get("specials", []):
            check(int(e["x"], 16), int(e["hw"], 16), f"{os.path.basename(sp)}:specials")
        m = s.get("measured", {})
        for e in m.get("vs_rn32", {}).get("max_ulp_examples", []):
            check(int(e["x"], 16), int(e["hw"], 16), f"{os.path.basename(sp)}:max_ulp")
        d = m.get("monotonicity", {}).get("max_drop_example")
        if d:
            check(int(d["x_lo"], 16), int(d["hw_lo"], 16), f"{os.path.basename(sp)}:max_drop")
            check(int(d["x_hi"], 16), int(d["hw_hi"], 16), f"{os.path.basename(sp)}:max_drop")
    if npy:
        with open(npy + ".json") as f:
            s = json.load(f)
        lo = int(s["range"]["lo"], 16)
        a = np.load(npy, mmap_mode="r")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.integers(0, a.shape[0], size=min(n, a.shape[0])))
        for i in idx:
            hw = int(a[int(i)])
            check(lo + int(i), hw, "npy")
            check((lo + int(i)) ^ SIGN, hw ^ SIGN, "npy:neg")  # answered only when the shard's odd symmetry is verified
    rep["ok"] = not rep["mismatches"] and not rep["outside_answered"]
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--conform", nargs="*", metavar="SUMMARY.json", help="conformance mode: check MufuTanh against these summaries")
    ap.add_argument("--conform-npy", help="conformance mode: also sample this raw shard (.npy, with its .npy.json beside it)")
    ap.add_argument("--conform-n", type=int, default=100000)
    ap.add_argument("--tables", help="conformance mode: shard directory (default $VERITY_MUFU_TANH_TABLES)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lo", type=lambda s: int(s, 0))
    ap.add_argument("--hi", type=lambda s: int(s, 0))
    ap.add_argument("--out", help=".npy path (summary goes to <out>.json)")
    ap.add_argument("--chunk", type=lambda s: int(s, 0), default=1 << 24)
    ap.add_argument("--store", choices=("u32", "diff8", "both", "none"), default="u32",
                    help="u32: raw .npy; diff8: xz first differences (<out minus .npy>.diff.xz, reconstruction validated); both; none: summary only")
    ap.add_argument("--neg", choices=("check", "none"), default="check", help="also run the sign-flipped words for the odd-symmetry check")
    ap.add_argument("--xz-chunks", type=int, default=4, help="chunks on which the compressed-encoding sizes are measured")
    ap.add_argument("--xz-preset", type=int, default=6)
    ap.add_argument("--block-words", type=lambda s: int(s, 0), default=1 << 22, help="compact-encoding block size (words); --chunk must be a multiple")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.conform is not None or a.conform_npy:
        return conform(a.tables, a.conform or [], a.conform_npy, a.conform_n, a.seed)
    if a.lo is None or a.hi is None or not a.out:
        ap.error("--lo, --hi, --out are required")
    if not 0 <= a.lo < a.hi <= 1 << 32:
        ap.error("need 0 <= lo < hi <= 2^32")
    rep = run_gpu(a.lo, a.hi, a.out, a.chunk, a.store, a.neg, a.xz_chunks, a.xz_preset, a.block_words)
    summary_path = a.out + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)) or ".", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    print(f"[measure_tanh] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
