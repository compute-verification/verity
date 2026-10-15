"""What the forming costs the useful output on real activations: Pearl-C v1's served y against the exact product, under v1's
row statistics and under the fix rho' = max(rho, s/4), beside noise-free quantizers (FP8 with the same alpha, int8).

The emulation is `pearl_c`'s v1 path on sm120 with every rounding that is coarser than FP32 done exactly and the FP32
accumulations in float64 (they are 2^-24 relative per word, the casts 2^-4 and 2^-9):

    stats   s = max|x|, rho = RMS of x at p = 0, 8, 16, ...; alpha = 448/(s + 4 rho), beta = rho alpha sqrt(32)/(256 16)
            (the fix: rho' = max(rho, s/4) in rho's place)
    form    E' = e4m3(beta E) per row (E at norm 256), N = E' F^T (F_A or F_B, norm 16), X~ = e4m3(alpha x + N)
    chain   C~ = A' B~^T
    peel    U = C~ + E'_A bf16(E'_B G - B~ F_A)^T - bf16(A' F_B) E'_B^T,  G = F_B^T F_A
    serve   y = U / (alpha_A alpha_B)

`check` compares it with `pearl_c`'s bit-exact forming, chain and peel on one small tile.  The lines are `pearl_c`'s
(`sample_line`, SHA-512 seeds).  A capture is `pearlc_capture.py`'s directory: X per layer and role as float32 holding
BF16, W read from the model's safetensors (BF16) by its index.

    python cost.py check
    python cost.py run CAPTURE_DIR --rows 256 --cols 256 --layers 0,8,16,24 --json OUT
"""
from __future__ import annotations

import json
import math
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

from verity.core.protocols.pouw.schemes import pearl_c as PC

P = PC.P
R = 32
LINE = math.sqrt(R) / (256 * 16)
ROLES = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o": ("self_attn.o_proj",),
         "gateup": ("mlp.gate_proj", "mlp.up_proj"), "down": ("mlp.down_proj",)}
FP8 = np.array([np.nan if c & 0x7F == 0x7F else P.fp8_to_f32(c) for c in range(256)], dtype=np.float64)


def e4m3(v: np.ndarray) -> np.ndarray:
    """Pearl's saturating nearest-even E4M3 cast of FP32 values (subnormals at 2^-9; [432, inf) to 448)."""
    a = np.abs(v.astype(np.float32).astype(np.float64))
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -9)))
    step = np.exp2(np.maximum(e, -6.0) - 3.0)
    q = np.minimum(np.rint(a / step) * step, 448.0)
    return np.copysign(q, v)


def bf16(v: np.ndarray) -> np.ndarray:
    b = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32).astype(np.float64)


def stats(x: np.ndarray, fix: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = np.abs(x).max(axis=1)
    rho = np.sqrt((x[:, ::8] ** 2).mean(axis=1))
    if fix:
        rho = np.maximum(rho, s / 4)
    alpha = (448.0 / np.maximum(s + 4 * rho, 2.0 ** -32)).astype(np.float32).astype(np.float64)
    beta = (rho * alpha * LINE).astype(np.float32).astype(np.float64)
    return s, rho, alpha, beta


def lines(seed: bytes, side: int, factor: int, idx, norm: float) -> np.ndarray:
    return FP8[np.array([PC.sample_line(PC.S512Seed(seed), side, factor, int(i), norm) for i in idx], dtype=np.int64)]


def live(x: np.ndarray) -> np.ndarray:
    """`pearl_c.live_row` in float64: at most k/64 entries with x^2 >= 64 rho^2."""
    rho2 = (x[:, ::8] ** 2).mean(axis=1)
    return 64 * (x ** 2 >= 64 * rho2[:, None]).sum(axis=1) <= x.shape[1]


def form(x: np.ndarray, e: np.ndarray, f: np.ndarray, fix: bool):
    s, rho, alpha, beta = stats(x, fix)
    ep = e4m3(beta[:, None] * e)
    return e4m3(alpha[:, None] * x + ep @ f.T), ep, alpha, rho * alpha


def served(x: np.ndarray, w: np.ndarray, seed_a: bytes, seed_b: bytes, fix: bool, rows_idx, cols_idx):
    k = x.shape[1]
    fa, fb = lines(seed_b, 0, 1, range(k), 16.0), lines(seed_b, 1, 1, range(k), 16.0)
    ea, eb = lines(seed_a, 0, 0, rows_idx, 256.0), lines(seed_b, 1, 0, cols_idx, 256.0)
    a, epa, alpha_a, sa = form(x, ea, fa, fix)
    b, epb, alpha_b, sb = form(w, eb, fb, fix)
    g = (fb.T @ fa).astype(np.float32).astype(np.float64)
    mid = bf16(epb @ g - b @ fa)
    u = a @ b.T + epa @ mid.T - bf16(a @ fb) @ epb.T
    y = u.astype(np.float32).astype(np.float64) / alpha_a[:, None] / alpha_b[None, :]
    return y, {"sigma_a": sa, "sigma_b": sb, "alpha_a": alpha_a, "alpha_b": alpha_b}


def quantized(x: np.ndarray, w: np.ndarray, how: str) -> np.ndarray:
    """Noise-free baselines: `fp8` = e4m3(alpha x) with v1's alpha (the model's quantizer with PoUW off), `fp8fix` with the
    fix's alpha; `int8` per row to [-127, 127], `int8n32` to [-95, 95] (room for a +-32 integer noise)."""
    if how.startswith("fp8"):
        fix = how == "fp8fix"
        aa, ab = stats(x, fix)[2], stats(w, fix)[2]
        return (e4m3(aa[:, None] * x) @ e4m3(ab[:, None] * w).T) / aa[:, None] / ab[None, :]
    top = {"int8": 127, "int8n32": 95}[how]
    sx, sw = top / np.abs(x).max(axis=1), top / np.abs(w).max(axis=1)
    return (np.rint(sx[:, None] * x) @ np.rint(sw[:, None] * w).T) / sx[:, None] / sw[None, :]


def row_err(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y - ref, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-300)


# ---- residual detection on real rows ----------------------------------------------------------------------------------

class LinePools:
    """`detection.Pools` as arrays: E lines (norm 256) and F lines (norm 16) from independent S512 seeds."""

    def __init__(self, n: int = 3000, seed: int = 7):
        from detection import Pools
        p = Pools(n, seed)
        self.e = FP8[np.array(p.e)]
        self.f = FP8[np.array(p.f)]


def point_masses(v: np.ndarray, beta: float, pools: LinePools, draws: int, rng: np.random.Generator) -> np.ndarray:
    """p_max of each entry's code e4m3(v + E'.F) over the seeds, v = alpha*x for the entries of one row (beta its
    beta_row): `detection.law`'s experiment in float64 (each row's E' = e4m3(beta E) per pool line)."""
    ep = e4m3(beta * pools.e)
    n = ep[rng.integers(0, len(ep), draws)] * pools.f[rng.integers(0, len(pools.f), draws)]
    codes = e4m3(v[:, None] + n.sum(axis=1)[None, :])
    out = np.empty(len(v))
    for i, row in enumerate(codes):
        out[i] = np.unique(row, return_counts=True)[1].max() / draws
    return out


def residual_detection(x: np.ndarray, fix: bool, pools: LinePools, draws: int, rng: np.random.Generator,
                       top: int = 8, rand: int = 8) -> dict:
    """Per row, the largest point mass over its `top` largest |x| entries and `rand` random ones (the worst entry of a
    row is at or near its largest from s/rho = 3.5 on, any entry below), under v1's statistics or the fix's."""
    s, rho, alpha, beta = stats(x, fix)
    k = x.shape[1]
    worst, top1 = np.empty(len(x)), np.empty(len(x))
    for i in range(len(x)):
        idx = np.concatenate((np.argsort(-np.abs(x[i]))[:top], rng.choice(k, rand, replace=False)))
        p = point_masses(alpha[i] * x[i, idx], beta[i], pools, draws, rng)
        worst[i], top1[i] = p.max(), p[0]
    ok = live(x) & (rho * alpha >= 1)
    return {"rows": len(x), "admissible": float(ok.mean()),
            "p_max_row": {"median": float(np.median(worst)), "p90": float(np.quantile(worst, 0.9)), "max": float(worst.max())},
            "p_max_admissible_max": float(worst[ok].max()) if ok.any() else None,
            "frac_rows_p_gt_1/6": float((worst > 1 / 6).mean()),
            "frac_rows_top_entry_deterministic": float((top1 >= 0.999).mean()),
            "spike_hit_rate_top_entry": float(top1.mean())}


# ---- the bit-exact check --------------------------------------------------------------------------------------------

def check(k: int = 256, m: int = 4, n: int = 4) -> dict:
    """The emulation against `pearl_c` (form_v1, chain, peel_factors_a/b, peel, the pinned unscale) on one m x n tile."""
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120
    from verity.ml.circuits.silicon import fp32
    rng = np.random.default_rng(1)
    x = rng.standard_normal((m, k)).astype(np.float32).astype(np.float64)
    x[0, 3] = 40.0
    w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32).astype(np.float64)
    seed_a, seed_b = bytes(range(32)), bytes(range(32, 64))
    words = lambda a: [[P.f32_bits(float(v)) for v in r] for r in a]  # noqa: E731
    sa, sb = PC.S512Seed(seed_a), PC.S512Seed(seed_b)
    ea = [PC.sample_line(sa, 0, 0, i) for i in range(m)]
    eb = [PC.sample_line(sb, 1, 0, j) for j in range(n)]
    fa, fb = PC._basis(sb, 0, k, 16.0), PC._basis(sb, 1, k, 16.0)
    A = PC.form_v1(words(x), ea, fa, SM120)
    B = PC.form_v1(words(w), eb, fb, SM120)
    pa = PC.peel_factors_a(A.codes, PC.scaled_noise(A, ea), sb, k, 16.0, SM120)
    pb = PC.peel_factors_b(B.codes, PC.scaled_noise(B, eb), sb, k, 16.0, SM120)
    exact = np.zeros((m, n))
    for i in range(m):
        ra = fp32.div(0x3F800000, A.alpha[i])
        for j in range(n):
            u = PC.peel(PC.chain(A.codes[i], B.codes[j], SM120), pa[i], pb[j], SM120)
            exact[i, j] = P.f32_of_bits(fp32.mul(fp32.mul(u, ra), fp32.div(0x3F800000, B.alpha[j])))
    emu, _ = served(x, w, seed_a, seed_b, False, range(m), range(n))
    codes_a = np.array([[FP8[c] for c in r] for r in A.codes])
    e_emu, f_emu = lines(seed_a, 0, 0, range(m), 256.0), lines(seed_b, 0, 1, range(k), 16.0)
    a_emu = form(x, e_emu, f_emu, False)[0]
    from detection import form as det_form, law, s5_stats_rho_s4
    fix_bits = np.array([[FP8[c] for c in r] for r in det_form(words(x), ea, fa, s5_stats_rho_s4)])
    ref = x @ w.T
    out = {"k": k, "tile": [m, n], "codes_equal_v1": float((codes_a == a_emu).mean()),
           "codes_equal_fix": float((fix_bits == form(x, e_emu, f_emu, True)[0]).mean()),
           "max_rel_diff_emulation_vs_bit_exact": float(np.abs(emu - exact).max() / np.abs(ref).max()),
           "err_bit_exact": float(np.abs(exact - ref).max() / np.abs(ref).max()),
           "err_emulation": float(np.abs(emu - ref).max() / np.abs(ref).max()), "law": []}
    from detection import Pools, beta_for
    pools, lp = Pools(400, 7), LinePools(400, 7)
    for v, sig in ((200.0, 56.0), (224.0, 56.0), (7.91, 1.0), (0.0, 1.25)):
        c = law(v, beta_for(sig), pools, 4000, random.Random(3))
        bits = max(c.values()) / 4000
        emu_p = point_masses(np.array([v]), P.f32_of_bits(beta_for(sig)), lp, 4000, np.random.default_rng(3))[0]
        out["law"].append({"v": v, "sigma": sig, "p_max_bit_exact": bits, "p_max_emulation": float(emu_p)})
    return out


# ---- a capture ------------------------------------------------------------------------------------------------------

def weight_meta(snapshot: Path, name: str) -> tuple[Path, int, dict]:
    """A linear's weight tensor in the snapshot's safetensors (the manifest names the module, the index its `.weight`)."""
    idx = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    key = name if name in idx else name + ".weight"
    path = snapshot / idx[key]
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        return path, hlen, json.loads(fh.read(hlen))[key]


def weight_rows(snapshot: Path, name: str, rows) -> np.ndarray:
    path, hlen, meta = weight_meta(snapshot, name)
    assert meta["dtype"] == "BF16"
    n, k = meta["shape"]
    off = 8 + hlen + meta["data_offsets"][0]
    mm = np.memmap(path, dtype=np.uint16, mode="r", offset=off, shape=(n, k))
    return (mm[np.asarray(rows)].astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def split_channels(x_all: np.ndarray) -> np.ndarray:
    """`pearl_c4/real.py`'s stand-in split: the floor(k/100) channels with the largest max|x| over the captured rows leave
    the unit (ties to the lower channel); their part of y is computed exactly."""
    k = x_all.shape[1]
    order = np.lexsort((np.arange(k), -np.abs(x_all).max(axis=0)))
    return np.sort(order[:k // 100])


def summarize(e: np.ndarray) -> dict:
    q = lambda p: float(np.quantile(e, p))  # noqa: E731
    return {"median": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": float(e.max()),
            "frac_gt_0.05": float((e > 0.05).mean()), "frac_gt_0.1": float((e > 0.1).mean()),
            "frac_gt_0.25": float((e > 0.25).mean())}


def one_linear(x_all: np.ndarray, snapshot: Path, name: str, rows, cols, split: bool, seed: int,
               pools: LinePools, draws: int, x_detect: bool) -> dict:
    x = x_all[rows]
    w = weight_rows(snapshot, name, cols)
    ref = x @ w.T
    keep = np.arange(x.shape[1])
    exact_part = np.zeros_like(ref)
    if split:
        out = split_channels(x_all)
        keep = np.setdiff1d(keep, out)
        exact_part = x[:, out] @ w[:, out].T
    xu, wu = x[:, keep], w[:, keep]
    pad = (-xu.shape[1]) % 128
    if pad:
        xu, wu = np.pad(xu, ((0, 0), (0, pad))), np.pad(wu, ((0, 0), (0, pad)))
    rng = random.Random(seed)
    seed_a, seed_b = rng.randbytes(32), rng.randbytes(32)
    res = {"k_unit": int(xu.shape[1]), "split": len(x_all[0]) - len(keep)}
    r_a = np.abs(xu).max(axis=1) / np.sqrt((xu[:, ::8] ** 2).mean(axis=1))
    r_b = np.abs(wu).max(axis=1) / np.sqrt((wu[:, ::8] ** 2).mean(axis=1))
    sig_a = 448.0 / (r_a + 4)
    res["rows"] = {"s_over_rho": {"median": float(np.median(r_a)), "p90": float(np.quantile(r_a, 0.9)), "max": float(r_a.max()),
                                  "frac_gt_4": float((r_a > 4).mean())},
                   "admissible_v1": float(((sig_a >= 1) & live(xu)).mean()),
                   "w_s_over_rho": {"median": float(np.median(r_b)), "max": float(r_b.max()), "frac_gt_4": float((r_b > 4).mean())},
                   "w_admissible_v1": float(((448.0 / (r_b + 4) >= 1) & live(wu)).mean())}
    errs = {}
    for how in ("fp8", "fp8fix", "int8", "int8n32"):
        errs[how] = row_err(quantized(xu, wu, how) + exact_part, ref)
    for fix in (False, True):
        y, _ = served(xu, wu, seed_a, seed_b, fix, rows, cols)
        errs["pearl_fix" if fix else "pearl_v1"] = row_err(y + exact_part, ref)
    res["row_rel_err"] = {h: summarize(e) for h, e in errs.items()}
    res["by_s_over_rho"] = {}
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 64), (64, math.inf)):
        mask = (r_a > lo) & (r_a <= hi)
        if mask.any():
            res["by_s_over_rho"][f"({lo},{hi}]"] = {
                "rows": int(mask.sum()), **{h: {"median": float(np.median(errs[h][mask])), "max": float(errs[h][mask].max())}
                                            for h in errs}}
    if 0 in list(rows):
        res["bos_row"] = {h: float(errs[h][list(rows).index(0)]) for h in errs}
    res["ratio_fix_over_v1_median"] = float(np.median(errs["pearl_fix"] / errs["pearl_v1"]))
    drng = np.random.default_rng(seed)
    sides = (("x", xu), ("w", wu)) if x_detect else (("w", wu),)
    res["detection"] = {f"{side}_{'fix' if fix else 'v1'}": residual_detection(op, fix, pools, draws, drng)
                        for side, op in sides for fix in (False, True)}
    return res


def run(capture: Path, n_rows: int, n_cols: int, layers: list[int], draws: int, split_modes=(False, True)) -> dict:
    man = json.loads((capture / "manifest.json").read_text())
    snapshot = Path(man["snapshot"])
    out = {"model": man["model"], "revision": man["revision"], "tokens": man["tokens"], "draws": draws, "linears": {}}
    rng = np.random.default_rng(7)
    pools = LinePools(3000, 7)
    seen, rows_of = set(), {}
    for name, entry in man["inputs"].items():
        layer = int(name.split(".")[2])
        if layer not in layers:
            continue
        t0 = time.time()
        x_all = np.load(capture / entry["file"]).astype(np.float64)
        m = x_all.shape[0]
        if entry["file"] not in rows_of:
            rows_of[entry["file"]] = np.sort(np.concatenate(([0], rng.choice(np.arange(1, m), n_rows - 1, replace=False))))
        rows = rows_of[entry["file"]]
        x_detect = entry["file"] not in seen
        seen.add(entry["file"])
        n_out = weight_meta(snapshot, name)[2]["shape"][0]
        cols = np.sort(rng.choice(n_out, min(n_cols, n_out), replace=False))
        out["linears"][name] = {f"split={s}": one_linear(x_all, snapshot, name, rows, cols, s, layer * 31 + len(name),
                                                         pools, draws, x_detect)
                                for s in split_modes}
        out["linears"][name]["seconds"] = round(time.time() - t0, 1)
        print(name, out["linears"][name]["seconds"], flush=True)
    return out


def main(argv: list[str]) -> int:
    if argv[0] == "check":
        print(json.dumps(check(), indent=1))
        return 0
    capture = Path(argv[1])
    opt = lambda f, d: argv[argv.index(f) + 1] if f in argv else d  # noqa: E731
    res = run(capture, int(opt("--rows", 256)), int(opt("--cols", 256)), [int(v) for v in opt("--layers", "0,8,16,24").split(",")],
              int(opt("--draws", 4000)))
    text = json.dumps(res, indent=1)
    if "--json" in argv:
        Path(opt("--json", "")).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
