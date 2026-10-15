"""The integer grid on the device: int8 x int8 -> s32 products (`torch._int_mm`, cuBLASLt's IMMA path on sm_120) against
the exact product, and the pair design's noised product and peel run in int8 limbs on the tensor cores.

    random    uniform int8 in [-127, 127] at k up to 133,120 (|C| < 2^31 for k <= 133,144)
    edge      all +127 against all +127 and all -127 at k = 133,120 (C = +-2,147,092,480), and one k past the bound
    peel      Q(A), Q(B) to +-121, the pair design's noise (E uniform on {-3..3}), C~ = A~ B~^T on the device, then
              U = C~ - (A~ F_B) E_B^T - E_A (B~ F_A - E_B G)^T with each wide operand in two int8 limbs, against Q(A) Q(B)^T

The reference is a float64 product on the CPU, exact because every partial sum is an integer below 2^53.

    python imma.py [--json FILE]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

import grid


def int_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (m x k) times b (n x k)^T on the device, int8 in, s32 out; k and n padded to multiples of 8, m to 32."""
    m, k = a.shape
    n = b.shape[0]
    pm, pk, pn = (-m) % 32 + (32 if m < 17 else 0), (-k) % 8, (-n) % 8
    ta = torch.from_numpy(np.pad(a, ((0, pm), (0, pk))).astype(np.int8)).cuda()
    tb = torch.from_numpy(np.pad(b, ((0, pn), (0, pk))).astype(np.int8)).cuda()
    out = torch._int_mm(ta, tb.T)
    torch.cuda.synchronize()
    return out[:m, :n].cpu().numpy().astype(np.int64)


def exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.rint((torch.from_numpy(a.astype(np.float64)) @ torch.from_numpy(b.astype(np.float64)).T).numpy()).astype(np.int64)


def limbs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = (x + 64) % 128 - 64
    hi = (x - lo) // 128
    assert np.abs(hi).max() <= 127, "two limbs hold |x| <= 16,320"
    return hi, lo


def wide_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a times b^T where a is wide (two int8 limbs) and b is int8."""
    hi, lo = limbs(a)
    return 128 * int_mm(hi, b) + int_mm(lo, b)


def random_case(rng, m: int, n: int, k: int) -> dict:
    a, b = rng.integers(-127, 128, (m, k)), rng.integers(-127, 128, (n, k))
    got, ref = int_mm(a, b), exact(a, b)
    return {"case": "random", "m": m, "n": n, "k": k, "equal": bool((got == ref).all()), "max_abs": int(np.abs(ref).max())}


def edge_case(k: int, sign: int) -> dict:
    a, b = np.full((32, k), 127), np.full((8, k), 127 * sign)
    got, ref = int_mm(a, b), exact(a, b)
    wrapped = ((ref + 2 ** 31) % 2 ** 32) - 2 ** 31
    return {"case": "edge", "k": k, "sign": sign, "C": int(ref[0, 0]), "fits_s32": bool(abs(ref[0, 0]) < 2 ** 31),
            "device": int(got[0, 0]), "equal": bool((got == ref).all()), "equal_mod_2^32": bool((got == wrapped).all())}


def peel_case(rng, m: int, n: int, k: int, L: int = 170, u: int = 3) -> dict:
    top = 127 - 2 * u
    qa = np.clip(np.rint(rng.standard_normal((m, k)) * 30), -top, top).astype(np.int64)
    qb = np.clip(np.rint(rng.standard_normal((n, k)) * 30), -top, top).astype(np.int64)
    fa, fb = grid.signed_pairs(k, L, rng), grid.signed_pairs(k, L, rng)
    ea, eb = rng.integers(-u, u + 1, (m, L)), rng.integers(-u, u + 1, (n, L))
    at, bt = qa + ea @ fa.T, qb + eb @ fb.T
    assert max(np.abs(at).max(), np.abs(bt).max()) <= 127
    c = int_mm(at, bt)
    pa = int_mm(at, fb.T)
    g = int_mm(fb.T, fa.T)
    eg = (wide_mm(g.T, eb).T if np.abs(g).max() > 127 else int_mm(eb, g.T))
    qq = int_mm(bt, fa.T) - eg
    u_ = c - wide_mm(pa, eb) - wide_mm(qq, ea).T
    return {"case": "peel", "m": m, "n": n, "k": k, "L": L, "C_tilde_equal": bool((c == exact(at, bt)).all()),
            "U_equals_QA_QB": bool((u_ == exact(qa, qb)).all()), "max_abs_G": int(np.abs(g).max()),
            "max_abs_peel_operand": int(max(np.abs(pa).max(), np.abs(qq).max()))}


def main(argv: list[str]) -> int:
    global int_mm
    rng = np.random.default_rng(11)
    if "--algebra" in argv:
        int_mm = exact  # noqa: F811  (checks the peel's algebra and limbs on the CPU, no device)
        res = {"device": "cpu-reference", "cases": []}
        ks, peels = (4096,), ((4096, 64),)
    else:
        res = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
               "torch": torch.__version__, "cuda": torch.version.cuda, "cases": []}
        ks, peels = (4096, 14336, 65536, 133120), ((4096, 256), (14336, 256))
        a = rng.integers(-127, 128, (256, 4096))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            int_mm(a, a)
        res["kernels"] = sorted({e.key for e in prof.key_averages() if "memcpy" not in e.key.lower()})
        for sign in (1, -1):
            res["cases"].append(edge_case(133120, sign))
        res["cases"].append(edge_case(133152, 1))
    for k in ks:
        res["cases"].append(random_case(rng, 256, 256, k))
    for k, n in peels:
        res["cases"].append(peel_case(rng, 256, n, k))
    res["all_exact"] = all(c["U_equals_QA_QB"] and c["C_tilde_equal"] if c["case"] == "peel" else c["equal"]
                           for c in res["cases"] if c["case"] != "edge" or c["fits_s32"])
    text = json.dumps(res, indent=1)
    if "--json" in argv:
        open(argv[argv.index("--json") + 1], "w").write(text)
    print(text)
    return 0 if res["all_exact"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
