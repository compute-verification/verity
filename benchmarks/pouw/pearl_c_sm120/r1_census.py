#!/usr/bin/env python3
"""R1's chain start, bit-exact on the CPU: where the canonical unpromoted sm_120 chain (`SM120_UNPROMOTED`: one FP32
accumulator over all of k, k32 atoms ascending, no split-K) is exact on Llama-3.1-8B's captured rows, formed as R1 forms
them, per linear class.  The red team's first condition on R1 (the store's `docs/pouw/new-designs.md` §4, F1): if the chain
from +0 holds aligned exact regions, R1 from +0 is plain v2's chain start, and a keyed hot start must break them.

A step of word (i, j) at atom t is exact when the atom's FP32 output equals its accumulator plus the exact sum of its 32
products (E4M3 values are integers in units of 2^-9, so every sum is an exact integer in units of 2^-18).  Per class:

  qkv-rot, gateup-rot  the residual's keyed Haar rotation Q (the RMSNorm weight folded into W): A = (X / g) Q, B = (W g) Q
  o-none               o_proj's captured input and weight rows, unrotated
  o-vo                 a keyed Haar rotation per KV head on the head dimension (V's output side, folded into v_proj and o_proj)
  o-vo-il              o-vo, then one keyed permutation of o's k (the attention epilogue writes A in that order)
  down-none            down_proj's captured input and weight rows
  down-perm            a keyed permutation of the intermediate dimension (folded into gate/up's rows and down's columns)
  down-bh, down-bh<B>  down-perm, then a keyed block-diagonal Haar rotation (blocks of 2,048, or of B): an online rotation, not
                       foldable

Rows are quantized per row (absmax to 448, E4M3 round-to-nearest-even), A's tokens and B's weight rows both; A's rows are
sampled from its distinct rows (equal codes up to sign are one row, as R1's per-row rules credit them; --no-dedup samples all).  Starts: `zero`
(+0), and `h<H>`: word (i, j) starts at H_ij = s_ij 2^(round(log2 r_ij) + H) (1 + u_ij / 2), r_ij = sqrt(32) |a_i| |b_j| / k
(the atom sum's rms), s and u keyed per word; `p<H>`, the separable form a kernel computes with one FMUL per word
(`separable_start`); the output is fl(C - H), whose error against the exact product is reported
(over all of k, on the first --err-rows x --err-cols words), with `order_free`, the share of those words whose final FP32 word
is unchanged when the chain visits the atoms in reverse order (1 where each atom's contribution doesn't depend on the state).

Per (layer, class, start), over the first --atoms atoms: the exact share per atom, the exact prefix from the start (mean = an upper bound on the atoms a
verifier must uncredit per word, a share 32 * mean / k of the credit), and for the prefix windows [0, t) and for 32-atom
windows every 8 atoms the share of words exact over the window and a greedy biclique (rows by exact count, columns
intersected, both orientations); `lemma` is whether it found the v2 lemma's density (1,024 x 512 of an 8,192^2 unit: 1/8 of
the rows by 1/16 of the columns).

  python r1_census.py --acts DIR --weights SNAPSHOT --out DIR [--layers 0,8,16,28] [--classes ...] [--rows 1024 --cols 512]
  python r1_census.py --synthetic --out DIR   # gaussian rows, small: the self-test

Needs numpy, and `verity` and `verity_pouw` importable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from verity.ml.kernels import e4m3_to_f32_batch, f32_to_bf16_rn_batch, group_sum_total_batch
from verity_pouw.schemes.pearl_c_device import SM120_UNPROMOTED

ATOM = SM120_UNPROMOTED.atom
CLASSES = ("o-none", "o-vo", "o-vo-il", "down-none", "down-perm", "down-bh", "qkv-rot", "gateup-rot")
STARTS = ("zero", "h8", "h10", "h12", "h14")
WIN = 32
PREFIXES = (1, 2, 4, 6, 8, 10, 16, 32)
LEMMA = (1 / 8, 1 / 16)
HEAD_DIM, Q_HEADS, KV_HEADS = 128, 32, 8
UNIT = 2.0 ** 18

E4M3_POS = e4m3_to_f32_batch(np.arange(127))[0].view(np.float32).astype(np.float64)


def rng(key: str, *parts) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256("/".join(map(str, (key, *parts))).encode()).digest()[:8], "little"))


def e4m3(x: np.ndarray) -> np.ndarray:
    """E4M3 codes of x (float64), round to nearest even, saturating at 448."""
    m = np.minimum(np.abs(x), 448.0)
    hi = np.clip(np.searchsorted(E4M3_POS, m), 1, 126)
    lo = hi - 1
    dlo, dhi = m - E4M3_POS[lo], E4M3_POS[hi] - m
    code = np.where((dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0)), hi, lo)
    return (code | (np.signbit(x) << 7)).astype(np.uint8)


def quantize(x: np.ndarray) -> np.ndarray:
    amax = np.abs(x).max(axis=1, keepdims=True)
    scale = np.float32(448.0) / np.where(amax > 0, amax, 1).astype(np.float32)
    return e4m3(x.astype(np.float32).astype(np.float64) * scale.astype(np.float64))


def values(codes: np.ndarray) -> np.ndarray:
    return e4m3_to_f32_batch(codes)[0].view(np.float32).astype(np.float64)


def haar(n: int, g: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(g.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


# ---- the capture and the weights ------------------------------------------------------------------------------------

class Safetensors:
    def __init__(self, snapshot: Path):
        self.root = snapshot
        self.where = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers: dict[str, tuple[int, dict]] = {}

    def get(self, name: str) -> np.ndarray:
        file = self.where[name]
        if file not in self.headers:
            with open(self.root / file, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.headers[file] = (8 + n, json.loads(f.read(n)))
        base, header = self.headers[file]
        t = header[name]
        assert t["dtype"] == "BF16", (name, t["dtype"])
        lo, hi = t["data_offsets"]
        raw = np.fromfile(self.root / file, dtype=np.uint16, count=(hi - lo) // 2, offset=base + lo)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(t["shape"])


def operands(cls: str, layer: int, args, key: str) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) as float64 before quantization: A's sampled token rows and B's sampled weight rows, both k wide."""
    if args.synthetic:
        k = 14336 // 4 if cls.startswith("down") else 1024
        g = rng(key, "synthetic", layer, cls)
        return g.standard_normal((args.rows, k)), g.standard_normal((args.cols, k))
    st = Safetensors(Path(args.weights))
    p = f"model.layers.{layer}."
    kind = cls.split("-")[0]
    x = np.load(Path(args.acts) / f"L{layer:03d}-{kind}.npy", mmap_mode="r")
    if kind == "qkv":
        w = np.concatenate([st.get(p + f"self_attn.{n}_proj.weight") for n in "qkv"])
        gamma = st.get(p + "input_layernorm.weight")
    elif kind == "gateup":
        w = np.concatenate([st.get(p + f"mlp.{n}_proj.weight") for n in ("gate", "up")])
        gamma = st.get(p + "post_attention_layernorm.weight")
    else:
        w = st.get(p + ("self_attn.o_proj.weight" if kind == "o" else "mlp.down_proj.weight"))
    g = rng(key, "sample", layer, kind)
    a = np.asarray(x[args.first_row:], np.float64)
    b = np.asarray(w[np.sort(g.choice(len(w), args.cols, replace=False))], np.float64)
    del w
    k = a.shape[1]
    g = rng(key, "fold", layer, cls)
    if cls.endswith("-rot"):
        q = haar(k, g)
        a, b = (a / gamma) @ q, (b * gamma) @ q
    elif cls.startswith("o-vo"):
        rots = [haar(HEAD_DIM, g) for _ in range(KV_HEADS)]
        for h in range(Q_HEADS):
            s = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            r = rots[h // (Q_HEADS // KV_HEADS)]
            a[:, s], b[:, s] = a[:, s] @ r, b[:, s] @ r
        if cls == "o-vo-il":
            pi = g.permutation(k)
            a, b = a[:, pi], b[:, pi]
    elif cls.startswith("down-") and cls != "down-none":
        pi = g.permutation(k)
        a, b = a[:, pi], b[:, pi]
        if cls.startswith("down-bh"):
            size = int(cls[7:] or 2048)
            for s0 in range(0, k, size):
                r = haar(size, g)
                a[:, s0:s0 + size], b[:, s0:s0 + size] = a[:, s0:s0 + size] @ r, b[:, s0:s0 + size] @ r
    return a, b


# ---- the chain and its exact steps ----------------------------------------------------------------------------------

def hot_start(av: np.ndarray, bv: np.ndarray, h: int, g: np.random.Generator) -> np.ndarray:
    """H_ij as FP32 words (see the docstring); a multiple of 2^-18."""
    k = av.shape[1]
    r = np.sqrt(32) * np.outer(np.linalg.norm(av, axis=1), np.linalg.norm(bv, axis=1)) / k
    e = np.round(np.log2(np.where(r > 0, r, 1.0))) + h
    u = g.integers(0, 1 << 22, r.shape) / float(1 << 23)
    sign = np.where(g.integers(0, 2, r.shape) == 1, -1.0, 1.0)
    hv = (sign * np.exp2(e) * (1 + u)).astype(np.float32)
    assert np.array_equal(hv.astype(np.float64) * UNIT, np.floor(hv.astype(np.float64) * UNIT)), "H below 2^-18"
    return hv.ravel().view(np.uint32)


def as_units(words: np.ndarray) -> np.ndarray:
    v = words.view(np.float32).astype(np.float64) * UNIT
    assert np.isfinite(v).all() and np.array_equal(v, np.floor(v)), "an FP32 word off the 2^-18 grid"
    return v.astype(np.int64)


def chain(ac: np.ndarray, bc: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-(atom, word) exact flags, the final words, and the exact product (units of 2^-18)."""
    rows, k = ac.shape
    av, bv = values(ac) * 512, values(bc) * 512
    ii, jj = (x.ravel() for x in np.meshgrid(np.arange(rows), np.arange(len(bc)), indexing="ij"))
    acc, prev = start.copy(), as_units(start)
    exact = np.zeros(rows * len(bc), np.int64)
    flags = np.zeros((k // 32, rows * len(bc)), bool)
    for t in range(k // 32):
        s = slice(32 * t, 32 * t + 32)
        part = (av[:, s] @ bv[:, s].T).ravel()
        assert np.abs(part).max() < 2 ** 52
        part = part.astype(np.int64)
        acc = group_sum_total_batch(ATOM, acc, ac[ii, s], bc[jj, s])[0]
        now = as_units(acc)
        flags[t] = now == prev + part
        prev, exact = now, exact + part
    return flags, acc, exact


def greedy(e: np.ndarray) -> tuple[int, int]:
    """The best (rows, columns) of the count-order greedy biclique on e, rows by count, columns intersected."""
    order = np.argsort(-e.sum(axis=1), kind="stable")
    cols = np.logical_and.accumulate(e[order], axis=0).sum(axis=1)
    area = cols * np.arange(1, len(cols) + 1)
    i = int(area.argmax())
    return i + 1, int(cols[i])


def kept_at(e: np.ndarray, frac: float) -> float:
    """The share of columns the count-order greedy keeps at a share frac of the rows."""
    order = np.argsort(-e.sum(axis=1), kind="stable")
    return float(np.logical_and.reduce(e[order[:max(1, int(len(e) * frac))]], axis=0).mean())


def region(e: np.ndarray) -> dict:
    r, c = greedy(e)
    ct, rt = greedy(e.T)
    lc, lr = kept_at(e, LEMMA[0]), kept_at(e.T, LEMMA[1])
    return {"share": float(e.mean()), "greedy_rows": [r, c], "greedy_cols": [rt, ct],
            "best_area": max(r * c, rt * ct) / e.size, "cols_at_rows_1/8": lc, "rows_at_cols_1/16": lr,
            "lemma": bool(lc >= LEMMA[1] or lr >= LEMMA[0])}


def census(flags: np.ndarray, rows: int, cols: int, k_atoms: int) -> dict:
    n = len(flags)
    lead = np.where(flags.all(axis=0), n, (~flags).argmax(axis=0))
    out = {"atoms": n, "step_exact": float(flags.mean()),
           "step_exact_by_atom": [round(float(x), 6) for x in flags.mean(axis=1)[:64]],
           "prefix": {"mean": float(lead.mean()), "credit_share": float(lead.mean() / k_atoms),
                      "quantiles": {q: int(np.quantile(lead, q)) for q in (0.5, 0.9, 0.99)},
                      "share_at_least": {t: float((lead >= t).mean()) for t in PREFIXES}},
           "prefix_windows": {}, "windows": {}}
    for t in PREFIXES:
        if t <= n:
            out["prefix_windows"][t] = region((lead >= t).reshape(rows, cols))
    best = None
    for s in range(0, n - WIN + 1, 8):
        w = region(flags[s:s + WIN].all(axis=0).reshape(rows, cols))
        out["windows"][s] = {"share": w["share"], "best_area": w["best_area"], "lemma": w["lemma"]}
        if best is None or w["best_area"] > best[1]["best_area"]:
            best = (s, w)
    if best:
        out["densest_window"] = {"start": best[0], **best[1]}
    out["any_lemma_window"] = any(w["lemma"] for w in out["windows"].values())
    return out


def repeated_slices(codes: np.ndarray) -> dict:
    """The share of (row, atom) code slices that another row repeats at the same atom: exactly, and up to sign (the chain is
    odd, so a negated slice's contribution is the negated contribution under the negated start)."""
    rows, k = codes.shape
    codes = np.where((codes & 0x7F) == 0, 0, codes).astype(np.uint8)
    nonzero = codes.reshape(rows, k // 32, 32).any(axis=2)
    out = {}
    for name, c in (("exact", codes), ("up_to_sign", None)):
        if c is None:
            s = codes.reshape(rows, k // 32, 32)
            first = np.take_along_axis(s, (s != 0).argmax(axis=2)[..., None], axis=2)
            c = np.where((first >> 7) == 1, s ^ np.where(s != 0, np.uint8(0x80), np.uint8(0)), s).reshape(rows, k)
        rep = np.zeros((rows, k // 32), bool)
        for t in range(k // 32):
            v = np.ascontiguousarray(c[:, 32 * t:32 * t + 32]).view(np.dtype((np.void, 32))).ravel()
            _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
            rep[:, t] = cnt[inv] > 1
        out[name] = float((rep & nonzero).mean())
    out["zero_slices"] = float((~nonzero).mean())
    return out


def separable_start(av: np.ndarray, bv: np.ndarray, h: int, g: np.random.Generator) -> np.ndarray:
    """H_ij = fl(P_i Q_j), P_i = s_i 2^round(log2 |a_i|) (1 + u_i / 2), Q_j = t_j 2^(round(log2 (sqrt(32) |b_j| / k)) + h)
    (1 + v_j / 2): keyed per row and per column, one FMUL per word in the kernel."""
    def side(x: np.ndarray, shift: float) -> np.ndarray:
        n = np.linalg.norm(x, axis=1) * shift
        e = np.round(np.log2(np.where(n > 0, n, 1.0)))
        sign = np.where(g.integers(0, 2, len(x)) == 1, -1.0, 1.0)
        return (sign * np.exp2(e) * (1 + g.integers(0, 1 << 22, len(x)) / float(1 << 23))).astype(np.float32)
    p, q = side(av, 1.0), side(bv, np.sqrt(32) / av.shape[1] * 2.0 ** h)
    hv = np.outer(p, q).astype(np.float32)
    assert np.array_equal(hv.astype(np.float64) * UNIT, np.floor(hv.astype(np.float64) * UNIT)), "H below 2^-18"
    return hv.ravel().view(np.uint32)


def distinct_rows(codes: np.ndarray) -> np.ndarray:
    """The first row of each class of rows equal in codes up to sign (per-row scaling makes any proportional copy one)."""
    c = np.where((codes & 0x7F) == 0, 0, codes).astype(np.uint8)
    first = c[np.arange(len(c)), (c != 0).argmax(axis=1)]
    c = np.where(((first >> 7) == 1)[:, None], c ^ np.where(c != 0, np.uint8(0x80), np.uint8(0)), c)
    _, idx = np.unique(np.ascontiguousarray(c).view(np.dtype((np.void, c.shape[1]))).ravel(), return_index=True)
    return np.sort(idx)


def start_words(start: str, av: np.ndarray, bv: np.ndarray, g: np.random.Generator) -> np.ndarray:
    """`zero` (+0), `h<H>` (hot_start, keyed per word) or `p<H>` (separable_start)."""
    if start == "zero":
        return np.zeros(len(av) * len(bv), np.uint32)
    return (hot_start if start[0] == "h" else separable_start)(av, bv, int(start[1:]), g)


def error(ac: np.ndarray, bc: np.ndarray, start: str, g: np.random.Generator) -> dict:
    """fl(C - H) over all of k against the exact product, beside the exact product's own BF16 rounding (rms, relative)."""
    s0 = start_words(start, values(ac), values(bc), g)
    _, acc, exact = chain(ac, bc, s0)
    y, truth = (acc.view(np.float32) - s0.view(np.float32)).astype(np.float64), exact / UNIT
    y_bf16 = (f32_to_bf16_rn_batch(truth.astype(np.float32).view(np.uint32))[0].astype(np.uint32) << 16).view(np.float32)
    rms = np.sqrt(np.mean(truth ** 2))
    rev = np.arange(ac.shape[1]).reshape(-1, 32)[::-1].ravel()
    _, back, _ = chain(ac[:, rev], bc[:, rev], s0)
    return {"rel_err_rms": float(np.sqrt(np.mean((y - truth) ** 2)) / rms),
            "bf16_rel_err_rms": float(np.sqrt(np.mean((y_bf16 - truth) ** 2)) / rms), "words": len(y),
            "order_free": float((back == acc).mean())}


def job(layer: int, cls: str, args, key: str, out: Path) -> str:
    t0 = time.monotonic()
    a, b = operands(cls, layer, args, key)
    every, bc = quantize(a), quantize(b)
    keep = distinct_rows(every) if args.dedup else np.arange(len(every))
    pick = np.sort(rng(key, "rows", layer, cls).choice(keep, min(args.rows, len(keep)), replace=False))
    ac = every[pick]
    av, bv = values(ac), values(bc)
    rows, cols, k = len(ac), len(bc), ac.shape[1]
    span = min(k, 32 * args.atoms)
    res = {"layer": layer, "class": cls, "rows": rows, "cols": cols, "k": k, "census_atoms": span // 32, "key": key,
           "zero_codes": {"a": float(((ac & 0x7F) == 0).mean()), "b": float(((bc & 0x7F) == 0).mean())},
           "subnormal_codes": {"a": float(((ac & 0x78) == 0).mean()), "b": float(((bc & 0x78) == 0).mean())},
           "a_rows": {"all": len(every), "distinct": len(keep), "dedup": args.dedup},
           "repeated_slices": {"a": repeated_slices(every), "a_distinct": repeated_slices(every[keep]), "b": repeated_slices(bc)},
           "starts": {}}
    path = out / f"L{layer:03d}-{cls}.json"
    y_true = (av[:, :span] @ bv[:, :span].T).ravel()
    for start in args.starts:
        t1 = time.monotonic()
        g = rng(key, "hot", layer, cls, start)
        flags, _, exact = chain(ac[:, :span], bc[:, :span], start_words(start, av, bv, g))
        assert np.allclose(exact / UNIT, y_true, rtol=0, atol=1e-6 * np.abs(y_true).max())
        res["starts"][start] = {**census(flags, rows, cols, k // 32),
                                **error(ac[:args.err_rows], bc[:args.err_cols], start, rng(key, "err", layer, cls, start)),
                                "seconds": round(time.monotonic() - t1, 1)}
        path.write_text(json.dumps(res, indent=1))
    res["seconds"] = round(time.monotonic() - t0, 1)
    path.write_text(json.dumps(res, indent=1))
    z = res["starts"].get("zero") or next(iter(res["starts"].values()))
    return f"L{layer:03d} {cls}: " + " | ".join(
        f"{s}: step {v['step_exact']:.3f} prefix {v['prefix']['mean']:.1f} ({100 * v['prefix']['credit_share']:.2f}%) "
        f"[0,10) area {v['prefix_windows'].get(10, {}).get('best_area', float('nan')):.3f} lemma {v['any_lemma_window']} "
        f"err {v['rel_err_rms']:.1e}" for s, v in res["starts"].items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--acts", help="a pearlc_capture directory: L{layer:03d}-{qkv,o,gateup,down}.npy")
    ap.add_argument("--weights", help="the Llama-3.1-8B snapshot (model.safetensors.index.json and its shards)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layers", default="0,8,16,28")
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--starts", default=",".join(STARTS))
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--cols", type=int, default=256)
    ap.add_argument("--atoms", type=int, default=128, help="the census's first atoms (the error runs over all of k)")
    ap.add_argument("--err-rows", type=int, default=64)
    ap.add_argument("--err-cols", type=int, default=32)
    ap.add_argument("--key", default="r1-census/v0")
    ap.add_argument("--first-row", type=int, default=0, help="A's rows from this token position on (long-context workloads)")
    ap.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="sample A's rows from its distinct rows (codes up to sign), as the per-row rules credit them")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.starts = args.starts.split(",")
    if args.synthetic:
        args.rows, args.cols = min(args.rows, 64), min(args.cols, 64)
    elif not (args.acts and args.weights):
        ap.error("--acts and --weights, or --synthetic")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = [(int(layer), cls) for cls in args.classes.split(",") for layer in args.layers.split(",")]
    lines, rc = [], 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(work))) as pool:
        futs = {pool.submit(job, layer, cls, args, args.key, out): (layer, cls) for layer, cls in work}
        for f in as_completed(futs):
            try:
                line = f.result()
            except Exception as e:  # noqa: BLE001
                line, rc = f"L{futs[f][0]:03d} {futs[f][1]}: FAILED {e!r}", 1
            print(line, flush=True)
            lines.append(line)
    (out / "summary.txt").write_text("\n".join(sorted(lines)) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
