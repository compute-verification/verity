#!/usr/bin/env python3
"""Bench for mainloop_nvf4.cuh's FP8 mode (the plain E4M3 GEMM) on one RTX PRO 6000 Blackwell (sm_120): nvf4_bench.py's run
(identity, gates at every shape before any timing, diagnostic timing) with FP8's operands, kernels, model and baselines.

fp8_bench.cubin is fp8_bench.cu built off the card:

  nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -fmad=false -Xptxas -v -cubin -o fp8_bench.cubin fp8_bench.cu

What differs from nvf4_bench.py:
1. Operands per code range (SCALES, default "0 1"): random E4M3 codes, never a NaN; range 0 every other code (zero,
   subnormals, up to 448), 1 exponent fields 4 .. 10 (2^-3 .. 15).  Tensor scales 1.0 (f32) for cuBLASLt; CUTLASS's kind-1
   kernels and the plain entries read none.
2. model: nv_ref_words (fp8_bench.cu's) against BLACKWELL_SM120_E4M3_M16N8K32 (verity.ml.kernels.tc.group_sum_total_batch
   chained from +0; if this tree's verity lacks it, its parameters as cursor/sm120-fp8-capture-75d4 pins them) and, the
   gate's negative control, against ADA_E4M3_M16N8K32 (two groups of 16 in a 14-bit adder), which must differ somewhere.
   The model gate fails closed: it fails when verity can't be imported.
3. Baselines of the harness's kind 1 (FP8 E4M3, per-tensor scales): CUTLASS (default the two the harness's gate run
   tuned fastest), cuBLASLt from HARNESS_LIB and LT2_LIB, and PLAIN_LIB's kind-1 entries.

  python3 fp8_bench.py --model-cache DIR   (no GPU: the model's words for every gated shape into DIR)
  python3 fp8_bench.py CUBIN   (env: FP8_EXPECT_UUID, KERNELS="f8256_o_ew", SHAPE="8192 16384", MODEL_CACHE=DIR, and
                                nvf4_bench.py's)
"""

from __future__ import annotations

import ctypes as C
import hashlib
import os
import random
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvf4_bench as NB  # noqa: E402

FP8 = 1  # the harness's kind
ONE = struct.pack("<f", 1.0)
CAPTURE = ("RTX PRO 6000 Blackwell Server Edition (sm_120), lane vllm-sm120-tc-gemm: runs r20260930-030915-23f3, "
           "r20260930-031244-ecbc, r20260930-041757-0bde, r20260930-052000-e32d (cursor/sm120-fp8-capture-75d4 7dcce2c12)")


def sm120_model(models):
    """BLACKWELL_SM120_E4M3_M16N8K32 and where it came from."""
    m = getattr(models, "BLACKWELL_SM120_E4M3_M16N8K32", None)
    if m is not None:
        return m, "verity.ml.circuits.silicon.models"
    return (models.GroupSum(name="blackwell_sm120_e4m3_m16n8k32", arch="sm_120", dtype="e4m3", operand_bits=8, groups=(32,),
                            width=26, zero_exponent=-133, validated=CAPTURE),
            "parameters of cursor/sm120-fp8-capture-75d4 7dcce2c12")


def coords(m: int, n: int, k: int, samples: int, full: int, seed: int):
    """The model gate's coordinates: every one if m n <= full, else `samples` with every row mod 16 and column mod 8; and
    their unique rows and columns with the inverse maps."""
    import numpy as np
    if m * n <= full:
        rows, cols = np.divmod(np.arange(m * n, dtype=np.int64), n)
    else:
        rng = random.Random(seed)
        rows = np.array([16 * rng.randrange(m // 16) + i % 16 for i in range(samples)], np.int64)
        cols = np.array([8 * rng.randrange(n // 8) + (i // 16) % 8 for i in range(samples)], np.int64)
    ua, ai = np.unique(rows, return_inverse=True)
    ub, bi = np.unique(cols, return_inverse=True)
    return rows, cols, ua, ai, ub, bi


def expected(a_rows, b_rows, ai, bi, k: int):
    """The sm_120 model's words and Ada's at the coordinates (a_rows[ai], b_rows[bi])."""
    import numpy as np
    from verity.ml.kernels.tc import chain_batch, group_sum_total_batch
    from verity.ml.circuits.silicon import models
    model, _ = sm120_model(models)
    want, neg = np.empty(len(ai), np.uint32), np.empty(len(ai), np.uint32)
    for lo in range(0, len(ai), 32768):
        sl = slice(lo, lo + 32768)
        x, y = a_rows[ai[sl]], b_rows[bi[sl]]
        for out, md in ((want, model), (neg, models.ADA_E4M3_M16N8K32)):
            out[sl] = chain_batch(lambda acc, p, q, md=md: group_sum_total_batch(md, acc, p, q), md.k, k // md.k, x, y)[0]
    return want, neg


def sha(a) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def cache_path(m: int, n: int, k: int, seed: int) -> str | None:
    d = os.environ.get("MODEL_CACHE")
    return os.path.join(d, f"model-{m}x{n}x{k}-{seed}.npz") if d else None


def gen_rows(rows, k: int, seed: int, scales: int):
    """fp8_bench.cu's f8_gen_codes on the host: rows `rows` (k bytes each) of an operand generated with `seed`."""
    import numpy as np

    def mix32(x):
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7FEB352D)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846CA68B)
        return x ^ (x >> np.uint32(16))

    w = (np.asarray(rows, np.uint64)[:, None] * np.uint64(k // 4) + np.arange(k // 4, dtype=np.uint64)[None, :]).astype(np.uint32)
    h = mix32(np.uint32(seed & 0xFFFFFFFF) ^ mix32(w * np.uint32(2) + np.uint32(1)))
    c = h.astype("<u4").view(np.uint8).reshape(len(rows), k)
    if scales == 1:
        return (c & 0x87) | ((4 + ((c >> 3) & 0xF) % 7) << 3).astype(np.uint8)
    return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)


def cache(out_dir: str) -> int:
    """The CPU step: every gated shape's and range's model words (plan(), MODEL_SAMPLES, FULL_MODEL) into out_dir, from the
    operands regenerated on the host, so that the GPU job only compares."""
    import numpy as np
    from verity.ml.circuits.silicon import models
    os.makedirs(out_dir, exist_ok=True)
    os.environ["MODEL_CACHE"] = out_dir
    samples, full = int(os.environ.get("MODEL_SAMPLES", "4096")), int(os.environ.get("FULL_MODEL", "131072"))
    model, source = sm120_model(models)
    _, shapes, ranges = NB.plan()
    for sc in ranges:
        for (m, n, k) in shapes:
            t0, seed, op_seed = time.time(), m + n + k + sc, NB.operand_seed(m, n, k, sc)
            rows, cols, ua, ai, ub, bi = coords(m, n, k, samples, full, seed)
            a_rows, b_rows = gen_rows(ua, k, op_seed, sc), gen_rows(ub, k, op_seed + 1, sc)
            want, neg = expected(a_rows, b_rows, ai, bi, k)
            np.savez(cache_path(m, n, k, seed), want=want, neg=neg, a_sha256=sha(a_rows), b_sha256=sha(b_rows), model=model.name,
                     model_source=source)
            NB.emit({"kind": "model_cache", "shape": [m, n, k], "scales": sc, "coordinates": int(len(rows)),
                     "model": model.name, "model_source": source, "sm120_vs_ada_differing": int((want != neg).sum()),
                     "seconds": round(time.time() - t0, 2)})
    return 0


class Fp8Bench(NB.Bench):
    KIND = FP8
    KERNELS = "f8256_o_ew"
    CUTLASS = "cutlass3x_fp8_128x256x64_coop cutlass3x_fp8_256x128x64_coop"
    UUID_ENV = "FP8_EXPECT_UUID"

    def tmap(self, ptr: int, k: int, rows: int, box_rows: int) -> bytes:
        """A CUtensorMap over a row-major rows x k byte matrix: box_rows x 64-byte boxes, SWIZZLE_64B."""
        raw = C.create_string_buffer(256)
        at = (C.addressof(raw) + 63) & ~63
        dims, strides = (NB.u64 * 2)(k, rows), (NB.u64 * 1)(k)
        box, estr = (NB.u * 2)(64, box_rows), (NB.u * 2)(1, 1)
        self.dev.ck("cuTensorMapEncodeTiled", NB.vp(at), NB.UINT8, 2, NB.vp(ptr), dims, strides, box, estr, 0, NB.SWIZZLE_64B,
                    NB.L2_128B, 0)
        return C.string_at(at, 128)

    def operands(self, m: int, n: int, k: int, scales: int, seed: int) -> dict:
        dev = self.dev
        ops = {"a": dev.alloc(m * k), "b": dev.alloc(n * k), "sfa": dev.put(ONE), "sfb": dev.put(ONE)}
        for j, (key, words) in enumerate((("a", m * k // 4), ("b", n * k // 4))):
            dev.launch("f8_gen_codes", NB.cdiv(words, 256), 256, p=(ops[key],), i=(words, seed + j, scales))
        ops["maps"] = dev.put(self.tmap(ops["a"], k, m, 256) + self.tmap(ops["b"], k, n, 128))
        dev.sync()
        return ops

    def model_gate(self, ops: dict, ref: int, m: int, n: int, k: int, samples: int, full: int, seed: int) -> dict:
        """nv_ref_words against the sm_120 model, and against Ada's, which must differ, at coords()'s coordinates (only their
        rows and words read back).  The models' words from MODEL_CACHE's file for this shape if there is one (cache(), a
        CPU step), once the sampled rows' sha256 equals the file's; else computed here."""
        try:
            import numpy as np
            from verity.ml.circuits.silicon import models
        except ImportError as e:
            return {"skipped": f"{type(e).__name__}: {e}"}
        dev, t0 = self.dev, time.time()
        rows, cols, ua, ai, ub, bi = coords(m, n, k, samples, full, seed)
        a_rows = np.stack([np.frombuffer(dev.read(ops["a"] + int(r) * k, k), np.uint8) for r in ua])
        b_rows = np.stack([np.frombuffer(dev.read(ops["b"] + int(c) * k, k), np.uint8) for c in ub])
        if len(rows) == m * n:
            got = np.frombuffer(dev.read(ref, m * n * 4), np.uint32)
        else:
            got = np.array([int.from_bytes(dev.read(ref + 4 * (int(r) * n + int(c)), 4), "little") for r, c in zip(rows, cols)],
                           np.uint32)
        path = cache_path(m, n, k, seed)
        rec: dict = {"cache": path}
        if path and os.path.exists(path):
            z = np.load(path)
            rec |= {"a_sha256": sha(a_rows), "b_sha256": sha(b_rows)}
            if (str(z["a_sha256"]), str(z["b_sha256"])) != (rec["a_sha256"], rec["b_sha256"]):
                return rec | {"error": "the device's sampled rows are not the cache's", "cache_a_sha256": str(z["a_sha256"]),
                              "cache_b_sha256": str(z["b_sha256"])}
            want, neg, name, source = z["want"], z["neg"], str(z["model"]), str(z["model_source"])
        else:
            model, source = sm120_model(models)
            want, neg = expected(a_rows, b_rows, ai, bi, k)
            name, rec["cache"] = model.name, None
        bad = np.nonzero(got != want)[0]
        return rec | {"model": name, "model_source": source, "coordinates": int(len(rows)), "full": bool(len(rows) == m * n),
                      "words_differing": int(len(bad)),
                      "first": [[int(rows[i]), int(cols[i]), f"{int(got[i]):08x}", f"{int(want[i]):08x}"] for i in bad[:8]],
                      "negative": {"model": models.ADA_E4M3_M16N8K32.name, "words_differing": int((got != neg).sum())},
                      "seconds": round(time.time() - t0, 2)}

    @staticmethod
    def model_ok(rec: dict) -> bool:
        return rec.get("words_differing") == 0 and rec.get("negative", {}).get("words_differing", 0) > 0


if __name__ == "__main__":
    os.environ.setdefault("SHAPE", "8192 16384")
    if sys.argv[1] == "--model-cache":
        sys.exit(cache(sys.argv[2]))
    sys.exit(NB.main(sys.argv, Fp8Bench))
