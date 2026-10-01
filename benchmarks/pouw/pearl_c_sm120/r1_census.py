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
  down-bh              down-perm, then a keyed block-diagonal Haar rotation (blocks of 2,048): an online rotation, not foldable

Rows are quantized per row (absmax to 448, E4M3 round-to-nearest-even), A's tokens and B's weight rows both.  Starts: `zero`
(+0), and `h<H>`: word (i, j) starts at H_ij = s_ij 2^(round(log2 r_ij) + H) (1 + u_ij / 2), r_ij = sqrt(32) |a_i| |b_j| / k
(the atom sum's rms), s and u keyed per word; the output is fl(C - H), whose error against the exact product is reported.

Per (layer, class, start): the exact share per atom, the exact prefix from the start (mean = an upper bound on the atoms a
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
    a = np.asarray(x[np.sort(g.choice(len(x), args.rows, replace=False))], np.float64)
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
    elif cls in ("down-perm", "down-bh"):
        pi = g.permutation(k)
        a, b = a[:, pi], b[:, pi]
        if cls == "down-bh":
            for s0 in range(0, k, 2048):
                r = haar(2048, g)
                a[:, s0:s0 + 2048], b[:, s0:s0 + 2048] = a[:, s0:s0 + 2048] @ r, b[:, s0:s0 + 2048] @ r
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


def census(flags: np.ndarray, rows: int, cols: int) -> dict:
    n = len(flags)
    lead = np.where(flags.all(axis=0), n, (~flags).argmax(axis=0))
    out = {"atoms": n, "step_exact": float(flags.mean()),
           "step_exact_by_atom": [round(float(x), 6) for x in flags.mean(axis=1)[:64]],
           "prefix": {"mean": float(lead.mean()), "credit_share": float(lead.mean() / n),
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


def job(layer: int, cls: str, args, key: str, out: Path) -> str:
    t0 = time.monotonic()
    a, b = operands(cls, layer, args, key)
    ac, bc = quantize(a), quantize(b)
    av, bv = values(ac), values(bc)
    rows, cols, k = len(ac), len(bc), ac.shape[1]
    y_true = (av @ bv.T).ravel()
    res = {"layer": layer, "class": cls, "rows": rows, "cols": cols, "k": k, "key": key,
           "zero_codes": {"a": float(((ac & 0x7F) == 0).mean()), "b": float(((bc & 0x7F) == 0).mean())},
           "subnormal_codes": {"a": float(((ac & 0x78) == 0).mean()), "b": float(((bc & 0x78) == 0).mean())},
           "starts": {}}
    path = out / f"L{layer:03d}-{cls}.json"
    for start in args.starts:
        t1 = time.monotonic()
        if start == "zero":
            s0, hv = np.zeros(rows * cols, np.uint32), np.zeros(rows * cols, np.float32)
        else:
            s0 = hot_start(av, bv, int(start[1:]), rng(key, "hot", layer, cls, start))
            hv = s0.view(np.float32)
        flags, acc, exact = chain(ac, bc, s0)
        assert np.allclose(exact / UNIT, y_true, rtol=0, atol=1e-6 * np.abs(y_true).max())
        y, truth = (acc.view(np.float32) - hv).astype(np.float64), exact / UNIT
        y_bf16 = (f32_to_bf16_rn_batch(truth.astype(np.float32).view(np.uint32))[0].astype(np.uint32) << 16).view(np.float32)
        rms = np.sqrt(np.mean(truth ** 2))
        res["starts"][start] = {**census(flags, rows, cols),
                                "rel_err_rms": float(np.sqrt(np.mean((y - truth) ** 2)) / rms),
                                "bf16_rel_err_rms": float(np.sqrt(np.mean((y_bf16 - truth) ** 2)) / rms),
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
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--key", default="r1-census/v0")
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
