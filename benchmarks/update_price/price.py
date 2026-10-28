"""Upper bounds on update hardness's price: the shortest time-bounded descriptions found for selections of a fine-tune's
updated slots, by method, version N and decoder budget t.

Reads `train.py`'s checkpoints and `train.json`; writes `price.json` (every number) and `tables.md` to `--out`. Every
description it prices is decoded and compared with the target bit for bit before its length counts.

Methods, each a decoder given the registration R (θ; with or without the training data) and a budget t:

- `ulp` (a): the slots' ulp distances from θ under the context coder (`codec.ulp_encode`); t: a decode, well under a step.
- `lowrank` (b): per matrix, integer rank-r factors of the update plus the coded residual (`codec.lowrank_encode`); the
  rank and the factors' precision are the shortest found.
- `pred0` (c, from θ alone): θ times a fitted scalar (weight decay) plus the coded residual.
- `predM` (c, replaying the first M steps): the prediction θ + a·θ + b·(x_M − θ) + c·u_M, u_M the Adam direction at step M,
  coefficients fitted per tensor and sent as float32, plus the coded residual; t ≥ M training steps.
- `replay` (d): the run record and the trainer, replayed for N steps (t ≥ N steps); without the data in R it also carries
  the tokens, coded prequentially by the model being trained (or by xz).

The decoder budget t is in training steps of this machine (`train.json`'s step times) and in seconds.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codec as C

HERE = Path(__file__).resolve().parent
RANKS = (1, 2, 4, 8, 16, 32, 64)
QBITS = (8, 6, 10)
LAYERS = (0, 15, 29)
FMTS = ("fp32", "bf16")
CLASSES = (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
           ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj"), ("norm", "norm"),
           ("embed", "embed_tokens"))
FRAME_BITS = 32          # a stream's length prefix, per stream in a concatenation
NAME_BITS = 16           # naming a selection R determines (a tensor set, a θ threshold, a seed's subset size)
SEED_BITS = 64
LARGE = 100_000          # the uniform price is read off selections at least this large, where λ's share is small


def cls(name: str) -> str:
    return next(c for c, key in CLASSES if key in name)


def layer(name: str) -> int | None:
    parts = name.split(".")
    return int(parts[2]) if len(parts) > 2 and parts[1] == "layers" else None


def log2_binom(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


class Ckpt:
    def __init__(self, root: Path):
        self.root = Path(root)

    def x(self, n: int, name: str) -> np.ndarray:
        return np.load(self.root / str(n) / f"{name}.npy", mmap_mode="r")

    def adam_dir(self, m: int, name: str, eps: float) -> np.ndarray:
        ea = np.load(self.root / f"state{m}" / f"{name}.exp_avg.npy").astype(np.float64)
        es = np.load(self.root / f"state{m}" / f"{name}.exp_avg_sq.npy").astype(np.float64)
        return (ea / (np.sqrt(es) + eps)).reshape(-1)


def timed(f, *a):
    t0 = time.perf_counter()
    v = f(*a)
    return v, time.perf_counter() - t0


def checked(target: np.ndarray, back: np.ndarray, what: str) -> None:
    if not np.array_equal(np.asarray(back).reshape(-1), np.asarray(target).reshape(-1)):
        raise SystemExit(f"{what}: the decoded slots differ from the target")


# --- per tensor -------------------------------------------------------------------------------------------------------

def tensor_task(job: dict) -> dict:
    ck, n, fmt, name, ms, eps = Ckpt(job["ckpt"]), job["N"], job["fmt"], job["name"], job["ms"], job["eps"]
    x0 = np.asarray(ck.x(0, name), dtype=np.float32)
    xn = np.asarray(ck.x(n, name), dtype=np.float32)
    shape = x0.shape
    r0, t = C.to_bits(x0, fmt).reshape(-1), C.to_bits(xn, fmt).reshape(-1)
    g = np.zeros(t.size, np.int64)
    d = C.keys(t, fmt) - C.keys(r0, fmt)
    out = {"name": name, "N": n, "fmt": fmt, "n": int(t.size), "changed": int(np.count_nonzero(d)),
           "median_ulps": float(np.median(np.abs(d)))}

    enc = C.ulp_encode(t, r0, g, fmt)
    back, dec = timed(C.ulp_decode, enc, r0, g, fmt)
    checked(t, back, f"{name} ulp")
    out["ulp"] = {"bits": 8 * len(enc), "dec_s": dec}
    ell = C.slot_bits(t, r0, g, fmt)
    e = ((C.to_bits(x0, "fp32").reshape(-1).astype(np.int64) >> 23) & 0xFF)
    out["theta_exp"] = {"count": np.bincount(e, minlength=256).tolist(),
                        "bits": np.bincount(e, weights=ell, minlength=256).tolist()}

    best = out["ulp"] | {"rank": 0}
    if len(shape) == 2:
        delta = C.from_bits(t, fmt).astype(np.float64).reshape(shape) - C.from_bits(r0, fmt).astype(np.float64).reshape(shape)
        svd = np.linalg.svd(delta, full_matrices=False)
        cands = {}

        def est(rank, q):
            fac = C.factorize(delta, rank, q, svd)
            ab = np.concatenate([fac.a.reshape(-1), fac.b.reshape(-1)])
            grp = np.repeat(np.arange(2, dtype=np.int64), [fac.a.size, fac.b.size])
            pred = C.lowrank_predict(r0, fac, fmt)
            cands[(rank, q)] = (C.ideal_ints(ab, grp, 2) + C.ideal_bits(t, pred, g, fmt), fac)

        for rank in RANKS:
            if rank < min(shape):
                est(rank, QBITS[0])
        if cands:
            r_best = min(cands, key=lambda k: cands[k][0])[0]
            for q in QBITS[1:]:
                est(r_best, q)
            (rank, q), (bits_est, fac) = min(cands.items(), key=lambda kv: kv[1][0])
            out["lowrank_scan"] = {f"{r}:{qq}": v[0] for (r, qq), v in cands.items()}
            out["sv"] = svd[1][:64].tolist()
            if bits_est < out["ulp"]["bits"]:
                enc = C.lowrank_encode(t, r0, fac, fmt)
                back, dec = timed(C.lowrank_decode, enc, r0, fmt)
                checked(t, back, f"{name} lowrank")
                if 8 * len(enc) < best["bits"]:
                    best = {"bits": 8 * len(enc), "dec_s": dec, "rank": rank, "qbits": q}
    out["lowrank"] = best

    theta64 = C.from_bits(r0, fmt).astype(np.float64)
    out["pred"] = {}
    for m in [0, *[m for m in ms if m < n]]:
        bases = [theta64]
        if m:
            bases += [np.asarray(ck.x(m, name), dtype=np.float64).reshape(-1) - x0.astype(np.float64).reshape(-1),
                      ck.adam_dir(m, name, eps)]
        cands = [C.fit_affine(t, r0, bases, fmt)]
        if m:
            cands += [np.array([0, 1, 0], np.float32), np.append(C.fit_affine(t, r0, bases[:2], fmt), np.float32(0))]
        scored = [(C.ideal_bits(t, C.affine_predict(r0, bases, c, fmt), g, fmt), i) for i, c in enumerate(cands)]
        _, i = min(scored)
        coef = cands[i]
        enc = C.affine_encode(t, r0, bases, coef, g, fmt)
        back, dec = timed(C.affine_decode, enc, r0, bases, g, fmt)
        checked(t, back, f"{name} pred{m}")
        out["pred"][str(m)] = {"bits": 8 * len(enc), "dec_s": dec, "coef": coef.tolist(),
                               "kind": ("ls3", "direct", "ls2")[i] if m else "decay"}
    return out


# --- selections across tensors ----------------------------------------------------------------------------------------

def gather(ck: Ckpt, names: list[str], n: int, fmt: str):
    t, r0, grp, th = [], [], [], []
    for i, name in enumerate(names):
        x0 = np.asarray(ck.x(0, name), dtype=np.float32).reshape(-1)
        t.append(C.to_bits(np.asarray(ck.x(n, name), dtype=np.float32).reshape(-1), fmt))
        r0.append(C.to_bits(x0, fmt))
        grp.append(np.full(x0.size, i, np.int16))
        th.append(np.abs(x0))
    return np.concatenate(t), np.concatenate(r0), np.concatenate(grp), np.concatenate(th)


def selection_task(job: dict) -> dict:
    ck, n, fmt, names, seed = Ckpt(job["ckpt"]), job["N"], job["fmt"], job["names"], job["seed"]
    coefs, m_pred, eps = job["coefs"], job["m_pred"], job["eps"]
    t, r0, grp, th = gather(ck, names, n, fmt)
    total = t.size
    absd = np.abs(C.keys(t, fmt) - C.keys(r0, fmt))
    sels = {}
    for frac in (0.001, 0.01):
        k = int(total * frac)
        sels[f"top{frac:g}"] = (np.sort(np.argsort(-absd, kind="stable")[:k]), log2_binom(total, k))
    for frac in (0.01, 0.1):
        k = int(total * frac)
        sels[f"bottom{frac:g}"] = (np.sort(np.argsort(absd, kind="stable")[:k]), log2_binom(total, k))
    order = np.argsort(-th, kind="stable")
    sels["theta_top0.1"] = (np.sort(order[: total // 10]), NAME_BITS)
    sels["theta_bottom0.1"] = (np.sort(order[-(total // 10):]), NAME_BITS)
    rng = np.random.default_rng(seed)
    for k in (1000, 100_000, 10_000_000):
        if k < total:
            sels[f"random{k:g}"] = (np.sort(rng.choice(total, k, replace=False)), SEED_BITS)
    del order, absd
    if "model.embed_tokens.weight" in names:
        e = names.index("model.embed_tokens.weight")
        seen = np.load(ck.root / f"seen{n}.npy")
        width = int(np.load(ck.root / "0" / "model.embed_tokens.weight.npy", mmap_mode="r").shape[1])
        off = int(np.searchsorted(grp, e))
        rows = np.nonzero(~seen)[0]
        unseen = off + (rows[:, None] * width + np.arange(width)).reshape(-1)
        sels["embed_unseen"] = (unseen, NAME_BITS)
        sels["embed_unseen_index"] = (unseen, log2_binom(len(seen), len(rows)))

    pred = None
    if m_pred:
        parts = []
        for i, name in enumerate(names):
            x0 = np.asarray(ck.x(0, name), dtype=np.float32).reshape(-1)
            bases = [C.from_bits(C.to_bits(x0, fmt), fmt).astype(np.float64),
                     np.asarray(ck.x(m_pred, name), dtype=np.float64).reshape(-1) - x0.astype(np.float64),
                     ck.adam_dir(m_pred, name, eps)]
            parts.append(C.affine_predict(C.to_bits(x0, fmt), bases, np.asarray(coefs[name], np.float32), fmt))
        pred = np.concatenate(parts)

    out = {"N": n, "fmt": fmt, "total": int(total), "m_pred": m_pred, "selections": {}}
    for sname, (idx, index_bits) in sels.items():
        if not len(idx):
            continue
        row = {"n": int(len(idx)), "index_bits": float(index_bits)}
        g = grp[idx].astype(np.int64)
        enc = C.ulp_encode(t[idx], r0[idx], g, fmt)
        back, dec = timed(C.ulp_decode, enc, r0[idx], g, fmt)
        checked(t[idx], back, f"{sname} ulp")
        row["ulp"] = {"bits": 8 * len(enc), "dec_s": dec}
        if pred is not None:
            nc = len(np.unique(g))
            enc = C.ulp_encode(t[idx], pred[idx], g, fmt)
            back, dec = timed(C.ulp_decode, enc, pred[idx], g, fmt)
            checked(t[idx], back, f"{sname} pred{m_pred}")
            row[f"pred{m_pred}"] = {"bits": 8 * len(enc) + 3 * 32 * nc, "dec_s": dec}
        out["selections"][sname] = row
    return out


# --- general-purpose baselines over a layer ---------------------------------------------------------------------------

def gp_task(job: dict) -> dict:
    ck, n, fmt, names = Ckpt(job["ckpt"]), job["N"], job["fmt"], job["names"]
    t = np.concatenate([C.to_bits(np.asarray(ck.x(n, nm), dtype=np.float32).reshape(-1), fmt) for nm in names])
    r0 = np.concatenate([C.to_bits(np.asarray(ck.x(0, nm), dtype=np.float32).reshape(-1), fmt) for nm in names])
    dt = t.dtype
    dk = (C.keys(t, fmt) - C.keys(r0, fmt)).astype(np.int64)
    wrap = (dk & ((1 << (8 * dt.itemsize)) - 1)).astype(dt)
    out = {"N": n, "fmt": fmt, "layer": job["layer"], "n": int(t.size)}
    for label, data in (("xor", C.xor_planes(t, r0)), ("delta", C.planes(wrap))):
        z = C.xz(data)
        if C.unplanes(lzma.decompress(z), dt).tobytes() != (t ^ r0 if label == "xor" else wrap).tobytes():
            raise SystemExit("xz round trip failed")
        out[f"xz_{label}"] = 8 * len(z)
        try:
            import zstandard
        except ImportError:
            continue
        z = zstandard.ZstdCompressor(level=19).compress(data)
        if zstandard.ZstdDecompressor().decompress(z) != data:
            raise SystemExit("zstd round trip failed")
        out[f"zstd_{label}"] = 8 * len(z)
    return out


# --- the replay description -------------------------------------------------------------------------------------------

def replay_bits(train: dict, n: int) -> dict:
    record = {"model": train["model"], "data": train["data"], "config": {**train["config"], "steps": train["config"]["steps"]},
              "stop_after": n}
    rec = 8 * len(lzma.compress(json.dumps(record, sort_keys=True, separators=(",", ":")).encode(), preset=9 | lzma.PRESET_EXTREME))
    trainer = 8 * len(lzma.compress((HERE / "train.py").read_bytes(), preset=9 | lzma.PRESET_EXTREME))
    ts = train["token_stream"][str(n)]
    steps = train["train"]["steps"][:n]
    secs = sum(s["fwd_s"] + s["bwd_s"] + s["opt_s"] for s in steps)
    rep = train.get("replay") or {}
    return {"record_bits": rec, "trainer_bits": trainer, "registered_bits": rec + trainer,
            "prequential_bits": ts["prequential_bits"], "xz_token_bits": ts["xz_bits"],
            "unregistered_bits": rec + trainer + min(ts["prequential_bits"], ts["xz_bits"]),
            "seconds": secs, "fwd_seconds": sum(s["fwd_s"] for s in steps),
            "verified": bool(rep.get("match", {}).get(str(n))) if n <= rep.get("steps", 0) else None}


# --- aggregation ------------------------------------------------------------------------------------------------------

def regimes(n: int, ms: list[int]) -> list[tuple[str, int]]:
    return [("codec", 0), *[(f"replay{m}", m) for m in ms if m < n], ("replay", n)]


def tensor_best(row: dict, upto: int) -> tuple[float, float, str]:
    """The shortest per-tensor description among the codecs and the predictors that replay at most `upto` steps."""
    opts = [(row["ulp"]["bits"], row["ulp"]["dec_s"], "ulp"), (row["lowrank"]["bits"], row["lowrank"]["dec_s"], "lowrank")]
    opts += [(v["bits"], v["dec_s"], f"pred{m}") for m, v in row["pred"].items() if int(m) <= upto]
    return min(opts)


def aggregate(train: dict, names: list[str], tensors: list[dict], sels: list[dict], gps: list[dict], ms: list[int]) -> dict:
    step_s = [s["fwd_s"] + s["bwd_s"] + s["opt_s"] for s in train["train"]["steps"]]
    mean_step = float(np.median(step_s))
    groups = {"all": names, **{f"class:{c}": [nm for nm in names if cls(nm) == c] for c, _ in CLASSES},
              **{f"layer:{lyr}": [nm for nm in names if layer(nm) == lyr] for lyr in LAYERS}}
    groups = {g: members for g, members in groups.items() if members}
    out = {"mean_step_s": mean_step, "versions": {}}
    by = {}
    for r in tensors:
        by.setdefault((r["N"], r["fmt"]), {})[r["name"]] = r
    for (n, fmt), rows in sorted(by.items()):
        rp = replay_bits(train, n)
        v = {"replay": rp, "groups": {}, "regimes": {}}
        for gname, members in groups.items():
            size = sum(rows[nm]["n"] for nm in members)
            frame = FRAME_BITS * len(members) + NAME_BITS
            gv = {"n": size, "methods": {}}
            for meth in ("ulp", "lowrank"):
                gv["methods"][meth] = {"bits": frame + sum(rows[nm][meth]["bits"] for nm in members),
                                       "dec_s": sum(rows[nm][meth]["dec_s"] for nm in members)}
            for m in rows[members[0]]["pred"]:
                gv["methods"][f"pred{m}"] = {"bits": frame + sum(rows[nm]["pred"][m]["bits"] for nm in members),
                                             "dec_s": sum(rows[nm]["pred"][m]["dec_s"] for nm in members)}
            for reg, upto in regimes(n, ms):
                if reg == "replay":
                    prev = gv["best"][list(gv["best"])[-1]]
                    for data in ("registered", "unregistered"):
                        b = rp[f"{data}_bits"]
                        gv["best"][f"replay_{data}"] = ({"bits": b, "t_steps": n, "t_s": rp["seconds"], "via": "replay"}
                                                        if b < prev["bits"] else prev)
                    continue
                picks = [tensor_best(rows[nm], upto) for nm in members]
                bits = frame + sum(p[0] for p in picks)
                via = {}
                for p in picks:
                    via[p[2]] = via.get(p[2], 0) + 1
                dec = sum(p[1] for p in picks)
                gv.setdefault("best", {})[reg] = {"bits": bits, "t_steps": upto + dec / mean_step,
                                                  "t_s": sum(step_s[:upto]) + dec, "via": via}
            v["groups"][gname] = gv
        # θ-selectable selections: the slots whose |θ| is in the top binades, priced by the ulp coder's per-slot lengths
        cnt = np.sum([rows[nm]["theta_exp"]["count"] for nm in names], axis=0)
        bits = np.sum([rows[nm]["theta_exp"]["bits"] for nm in names], axis=0)
        cc, cb = np.cumsum(cnt[::-1])[::-1], np.cumsum(bits[::-1])[::-1]
        v["theta_binades"] = [{"exp_min": int(e) - 127, "n": int(cc[e]), "bits_per_weight": float(cb[e] / cc[e])}
                              for e in range(256) if cnt[e] and cc[e] >= 1000]
        sel = next((s for s in sels if s["N"] == n and s["fmt"] == fmt), None)
        v["selections"] = sel["selections"] if sel else {}
        v["gp"] = [g for g in gps if g["N"] == n and g["fmt"] == fmt]
        # implied λ: the deficit a uniform price at the 'all' rate needs to survive every selection tried; and the
        # largest uniform price that survives them with λ = 0, over selections of at least LARGE slots
        allg = v["groups"]["all"]
        for reg in allg["best"]:
            p = allg["best"][reg]["bits"] / allg["n"]
            rep = rp[reg[len("replay_"):] + "_bits"] if reg.startswith("replay_") else math.inf
            cands = []
            for gname, gv in v["groups"].items():
                b = gv["best"].get(reg) or gv["best"][list(gv["best"])[-1]]
                cands.append((gv["n"], b["bits"], gname))
            for sname, row in v["selections"].items():
                meths = [row["ulp"]["bits"]] + ([row[k]["bits"] for k in row if k.startswith("pred")] if reg != "codec" else [])
                cands.append((row["n"], min(min(meths), rep) + row["index_bits"], sname))
            for tb in v["theta_binades"]:
                cands.append((tb["n"], min(tb["bits_per_weight"] * tb["n"], rep) + NAME_BITS, f"theta_exp>={tb['exp_min']}"))
            lam, where = max((p * size - bits, name) for size, bits, name in cands)
            floor = min(LARGE, max(size for size, _, _ in cands))
            pu, pu_at, pu_n = min((bits / size, name, size) for size, bits, name in cands if size >= floor)
            v["regimes"][reg] = {"price_bits_per_weight": p, "lambda_implied": max(0.0, lam), "lambda_at": where,
                                 "uniform_price": pu, "uniform_price_at": pu_at, "uniform_price_n": pu_n,
                                 "t_steps": allg["best"][reg]["t_steps"], "t_s": allg["best"][reg]["t_s"]}
        out["versions"][f"{n}/{fmt}"] = v
    return out


# --- tables -----------------------------------------------------------------------------------------------------------

def fmt_bits(x: float) -> str:
    return f"{x:.3g}" if x < 0.01 else f"{x:.3f}" if x < 10 else f"{x:.2f}"


def render(res: dict, train: dict) -> str:
    cfg = train["config"]
    lines = ["# Update price: upper bounds from time-bounded decoders", "",
             f"SmolLM2-135M ({train['model']['revision'][:8]}), fp32 AdamW on {cfg['batch']}x{cfg['seq']}-token batches of "
             f"{train['data']['file']} ({train['data']['revision'][:8]}), lr {cfg['lr']:g} (warmup {cfg['warmup']}, cosine to "
             f"{cfg['min_lr_frac']:g}x), wd {cfg['wd']:g}, betas {cfg['betas']}, clip {cfg['clip']:g}, seed {cfg['seed']}; "
             f"{train['machine']['cpu']}, {train['machine']['threads']} threads, {res['mean_step_s']:.2f} s/step (median). "
             "Bits per updated weight; t in training steps of this machine.", ""]
    keys = sorted(res["versions"], key=lambda k: (k.split("/")[1], int(k.split("/")[0])))
    for fmt in FMTS:
        lines += [f"## All {fmt} slots: bits per weight by decoder budget", "",
                  "| N | slots changed | ulp | lowrank | pred0 | best, t < 1 step | best with replay of M steps | replay (data in R) | replay (data not in R) | λ implied at the codec rate |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            n, f = k.split("/")
            if f != fmt:
                continue
            v = res["versions"][k]
            a = v["groups"]["all"]
            per = lambda b: b / a["n"]  # noqa: E731
            reps = [f"M={r[6:]}: {fmt_bits(per(a['best'][r]['bits']))}" for r in a["best"] if r.startswith("replay") and r[6:].isdigit()]
            ch = v.get("changed", None)
            lines.append(f"| {n} | {ch if ch is not None else ''} | {fmt_bits(per(a['methods']['ulp']['bits']))} | "
                         f"{fmt_bits(per(a['methods']['lowrank']['bits']))} | {fmt_bits(per(a['methods']['pred0']['bits']))} | "
                         f"{fmt_bits(per(a['best']['codec']['bits']))} | {'; '.join(reps) or '-'} | "
                         f"{fmt_bits(per(a['best']['replay_registered']['bits']))} (t = {n}) | "
                         f"{fmt_bits(per(a['best']['replay_unregistered']['bits']))} (t = {n}) | "
                         f"{v['regimes']['codec']['lambda_implied']:.3g} ({v['regimes']['codec']['lambda_at']}) |")
        lines.append("")
    for fmt in FMTS:
        lines += [f"## Per class and layer, {fmt}: bits per weight, best at t < 1 step / with replay of the largest M < N", "",
                  "| N | " + " | ".join(g for g in res["versions"][keys[0]]["groups"]) + " |",
                  "|---|" + "---|" * len(res["versions"][keys[0]]["groups"])]
        for k in keys:
            n, f = k.split("/")
            if f != fmt:
                continue
            v = res["versions"][k]
            cells = []
            for g, gv in v["groups"].items():
                b = gv["best"]
                last = [r for r in b if r.startswith("replay") and r[6:].isdigit()]
                cells.append(fmt_bits(b["codec"]["bits"] / gv["n"]) + (f" / {fmt_bits(b[last[-1]]['bits'] / gv['n'])}" if last else ""))
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    for fmt in FMTS:
        lines += [f"## Selections across tensors, {fmt}: bits per weight (description + index)", "",
                  "| N | selection | slots | index bits | ulp | predM |", "|---|---|---|---|---|---|"]
        for k in keys:
            n, f = k.split("/")
            if f != fmt:
                continue
            for s, row in res["versions"][k]["selections"].items():
                pk = [x for x in row if x.startswith("pred")]
                lines.append(f"| {n} | {s} | {row['n']} | {row['index_bits']:.4g} | "
                             f"{fmt_bits((row['ulp']['bits'] + row['index_bits']) / row['n'])} | "
                             f"{fmt_bits((row[pk[0]]['bits'] + row['index_bits']) / row['n']) + ' (' + pk[0] + ')' if pk else '-'} |")
        lines.append("")
    lines += ["## λ implied: the deficit a uniform price at the all-slots rate needs to survive every selection tried", "",
              "| N | fmt | decoder budget | t (steps) | all-slots rate (bits/weight) | λ implied (bits) | at selection | uniform price ≤ at λ = 0 | at selection (slots) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        n, f = k.split("/")
        for reg, r in res["versions"][k]["regimes"].items():
            lines.append(f"| {n} | {f} | {reg} | {r['t_steps']:.3g} | {fmt_bits(r['price_bits_per_weight'])} | "
                         f"{r['lambda_implied']:.4g} | {r['lambda_at']} | {fmt_bits(r['uniform_price'])} | "
                         f"{r['uniform_price_at']} ({r['uniform_price_n']}) |")
    lines.append("")
    lines += ["## Replay descriptions", "",
              "| N | record bits | trainer bits | token bits, prequential | token bits, xz | replay seconds | t (steps) | replay reproduced bit for bit |",
              "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        n, f = k.split("/")
        if f != "fp32":
            continue
        rp = res["versions"][k]["replay"]
        lines.append(f"| {n} | {rp['record_bits']} | {rp['trainer_bits']} | {rp['prequential_bits']:.4g} | {rp['xz_token_bits']} | "
                     f"{rp['seconds']:.1f} | {n} | {rp['verified'] if rp['verified'] is not None else 'not rerun'} |")
    lines.append("")
    lines += ["## General-purpose compressors on one layer (bits per weight)", "", "| N | fmt | layer | xz xor | xz delta | zstd xor | zstd delta |",
              "|---|---|---|---|---|---|---|"]
    for k in keys:
        for g in res["versions"][k]["gp"]:
            c = lambda key: fmt_bits(g[key] / g["n"]) if key in g else "-"  # noqa: E731
            lines.append(f"| {g['N']} | {g['fmt']} | {g['layer']} | {c('xz_xor')} | {c('xz_delta')} | {c('zstd_xor')} | {c('zstd_delta')} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--sel-workers", type=int, default=2)
    p.add_argument("--versions", default="", help="N values to price (default: every saved step)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default="", help="price only the tensors whose names match this regex (a smoke run)")
    a = p.parse_args(argv)
    train = json.loads(a.train.read_text())
    cfg = train["config"]
    names = [q[0] for q in train["train"]["params"] if re.search(a.only, q[0])]
    ns = [int(x) for x in (a.versions or cfg["save"]).split(",") if x]
    ms = [int(x) for x in cfg["state"].split(",") if x]
    eps = cfg["eps"]
    size = {q[0]: int(np.prod(q[1])) for q in train["train"]["params"]}
    jobs = [{"ckpt": str(a.ckpt), "N": n, "fmt": f, "name": nm, "ms": ms, "eps": eps}
            for n in ns for f in FMTS for nm in names]
    jobs.sort(key=lambda j: -size[j["name"]])
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        tensors = list(ex.map(tensor_task, jobs, chunksize=1))
    print(f"tensors priced in {time.perf_counter() - t0:.0f} s", flush=True)
    by = {}
    for r in tensors:
        by.setdefault((r["N"], r["fmt"]), {})[r["name"]] = r
    sjobs, gjobs = [], []
    for n in ns:
        mp = max([m for m in ms if m < n], default=0)
        for f in FMTS:
            coefs = {nm: by[(n, f)][nm]["pred"][str(mp)]["coef"] for nm in names} if mp else {}
            sjobs.append({"ckpt": str(a.ckpt), "N": n, "fmt": f, "names": names, "seed": a.seed, "coefs": coefs,
                          "m_pred": mp, "eps": eps})
            for lyr in LAYERS:
                members = [nm for nm in names if layer(nm) == lyr]
                if members:
                    gjobs.append({"ckpt": str(a.ckpt), "N": n, "fmt": f, "layer": lyr, "names": members})
    with ProcessPoolExecutor(a.sel_workers) as ex:
        sels = list(ex.map(selection_task, sjobs))
    print(f"selections priced in {time.perf_counter() - t0:.0f} s", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        gps = list(ex.map(gp_task, gjobs))
    print(f"baselines in {time.perf_counter() - t0:.0f} s", flush=True)
    res = aggregate(train, names, tensors, sels, gps, ms)
    for k, v in res["versions"].items():
        n, f = k.split("/")
        v["changed"] = sum(r["changed"] for r in by[(int(n), f)].values())
    res["tensors"] = tensors
    (a.out / "price.json").write_text(json.dumps(res))
    (a.out / "tables.md").write_text(render(res, train))
    print((a.out / "tables.md").read_text())


if __name__ == "__main__":
    main()
