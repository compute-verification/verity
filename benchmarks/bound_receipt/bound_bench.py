"""The bound receipt's benchmark on one GPU: `bound.cu`'s libraries (ra-codec-wide's `wide.cu`, whose launchers they
keep, plus `rb_gemm`), one a k (`--lib`, repeated; each library's RB_K is its k), gated against `bound.py`, then timed at
the linears of `--model` (Llama-3.1-70B by default) and the batches `--batches` (1, 8, 32, 64). Writes `bound.json` and
`bench.log` under `--out`, and in a `research run` its `result.json`.

Nothing is timed before its gate passed. For each linear and batch, and each mode (unbound, tile, slice32, slice64,
slice128), on the store encoded at k:
- the partial sums equal the payload store's (`ww_gemm` at k = 0; two halves of 32 rows at batch 64) bit for bit;
- every tile's sketch and every receipt was written (both poisoned first), and a run without the hash gives the same
  sketches;
- sampled tiles' sketches equal `bound.tile_sketch` over their four records, their weight rows, the batch's chunk of x
  and zprev (the accumulators from sm_120's E4M3 model), and sampled receipts `bound.receipts_of` over the device's
  sketches;
- unbound, batch ≤ 32: sketches and receipts equal `ww_gemm_sketch`'s (ra-codec-wide's ro-sketch-blake3/v0);
- plain (no sketch): the partial sums equal the payload store's;
- on the first linear and batch: `rb_receipts` apart equals the fused hash on every vector, and `rb_commit`'s root
  `bound.root` over the device's receipts.

Then interleaved reps, every call on cold weights (a ring of store copies past the L2), each arm queued behind a spin
kernel: cuBLASLt FP8 (`torch._scaled_mm`, native), and for batch ≤ 32 `ww_gemm` decode only (`d`) and
`ww_gemm_sketch` hashed (`h`, the unbound receipt as ra-codec-wide measured it); and `rb_gemm` without a sketch
(`plain`) and hashed in every mode (`unbound`, `tile`, `slice32`, `slice64`, `slice128`). zprev stands in for the
previous linear's output (batch × N FP32; its contents don't change the time). Per batch, the layer (the four linears)
sums each arm: `layer.ratio_vs_native`; `layer.receipt_cost` = (mode − plain) / native, each receipt's cost per weight
pass (τ_W for unbound); and `layer.binding_cost` = (mode − unbound) / native, the bound receipt's honest cost over the
unbound one in the same kernel.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "word_codec"))

import bound as BR
import wide_bench as WB

B, W, WS = WB.B, WB.W, WB.WS
stream, ptr = WB.stream, WB.ptr
TAG_SEED = WB.TAG_SEED
POISON = WB.POISON - (1 << 32)
MODES = list(BR.MODES)
PLAIN = 2
ARMS = {"plain": PLAIN, **BR.MODES}


class Lib(WB.WLib):
    def __init__(self, name: str, path: str) -> None:
        super().__init__(name, path)
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        so = self.so
        so.rb_constants.argtypes = [v]
        so.rb_gemm.argtypes = [i, i, v, v, v, i, i, i, q, i, v, v, q, v, v, v, i, v]
        so.rb_receipts.argtypes = [v, q, q, v, v, v]
        so.rb_commit.argtypes = [v, q, v, v, v]
        so.rb_occupancy.argtypes = [i, i, i]
        cs = (ctypes.c_int * 2)()
        so.rb_constants(cs)
        self.rb_k, self.rb_occ = list(cs)


def tile_vectors(store, n: int, kd: int, t: int) -> np.ndarray:
    """Tile t's 16 vectors (wide_sketch's), from its four records of the device store."""
    f = BR.FIELD
    nch, rec = kd // f.task_cols, W.LANES * f.lane_words() * f.limbs
    G, kc = divmod(t, nch)
    parts = []
    for i in range(WS.T):
        nb = WS.T * G + i
        q = nb * nch + kc
        parts.append(store[q * rec:(q + 1) * rec].cpu().numpy().tobytes() if nb < n // 16 else bytes(4 * rec))
    return WS.vectors(b"".join(parts), 16 * WS.T, f.task_cols, f)


def case(lib: Lib, torch, name: str, n: int, kd: int, batches, reps: int, log, gate_tiles: int, first: bool) -> dict:
    f, k = BR.FIELD, lib.rb_k
    if kd % f.task_cols or n % 64:
        raise ValueError(f"{name}: N a multiple of 64 and K of {f.task_cols}")
    w = B.fp8_weights(torch, n, kd, 100 + n + kd)
    wb = w.view(torch.uint8)
    wnp = wb.cpu().numpy()
    pay = WB.payload_store(torch, wb, f)
    nwords = pay.numel() // f.limbs
    lib.set_key(W.key(f, WB.SEED, k))
    store = torch.empty_like(pay)
    lib.call("ww_encode", ptr(pay), ptr(store), nwords, 0, k, 188 * 4, stream(torch))
    torch.cuda.synchronize()
    nch = kd // f.task_cols
    ntiles, nvec = (n // 64) * nch, 16 * (n // 64) * nch
    masks_np, signs_np = BR.mask_fragments(TAG_SEED), BR.sign_fragments(TAG_SEED)
    masks = torch.from_numpy(masks_np.copy()).cuda()
    signs = torch.from_numpy(signs_np.copy()).cuda()
    sk = torch.empty(ntiles * 256, dtype=torch.int32, device="cuda")
    sk2 = torch.empty_like(sk)
    ws = torch.empty(lib.so.ww_workspace_bytes(nvec), dtype=torch.uint8, device="cuda")
    tags_p = lib.so.ww_workspace_tags(ptr(ws), nvec)
    tags = torch.as_strided(ws.view(torch.int32), (nvec * 8,), (1,), 0)
    ws2 = torch.empty_like(ws)
    tags2_p = lib.so.ww_workspace_tags(ptr(ws2), nvec)
    tags2 = torch.as_strided(ws2.view(torch.int32), (nvec * 8,), (1,), 0)
    seed_buf = ctypes.create_string_buffer(TAG_SEED, 32)
    gx = torch.Generator(device="cuda").manual_seed(5)
    xs = {mb: torch.randn(mb, kd, device="cuda", generator=gx).to(torch.float8_e4m3fn) for mb in batches}
    zs = {mb: torch.randn(mb, n, device="cuda", generator=gx) for mb in batches}
    res = {"N": n, "K": kd, "store_bytes": pay.numel() * 4, "tiles": ntiles, "gate": {}, "times": {},
           "occupancy": {mb: {m: lib.so.rb_occupancy(ARMS[m], mb, 0) for m in ARMS} for mb in batches}}

    def rb(mode, x, y, mb, partials, with_sk=True, hashed=True, store_t=None):
        lib.call("rb_gemm", BR.MODES[mode], k, ptr(store if store_t is None else store_t), ptr(x.view(torch.uint8)),
                 ptr(y), mb, n, kd, 0, int(partials), ptr(signs if mode == "unbound" else masks),
                 ptr(sk) if with_sk else None, 0, seed_buf if hashed else None, tags_p if hashed else None,
                 ptr(zs[mb]), n, stream(torch))

    def payload_partials(mb):
        x = xs[mb]
        halves = [(0, min(mb, 32))] + ([(32, mb)] if mb > 32 else [])
        out = []
        for lo, hi in halves:
            pp = torch.full((nch * (hi - lo) * n,), float("nan"), dtype=torch.float32, device="cuda")
            lib.call("ww_gemm", 0, ptr(pay), ptr(x[lo:hi].contiguous().view(torch.uint8)), ptr(pp), hi - lo, n, kd, 0, 1,
                     None, None, 0, stream(torch))
            out.append(pp.view(nch, hi - lo, n))
        return torch.cat(out, 1).reshape(-1)

    rng = random.Random(n + kd)
    for gi, mb in enumerate(batches):
        lib.set_key(W.key(f, WB.SEED, k))
        x = xs[mb]
        pp = payload_partials(mb)
        ref = x.float() @ w.float().t()
        scale = (x.float().abs() @ w.float().abs().t()).clamp_min(1e-30)
        entry = {"payload_vs_torch_rel": float(((pp.view(nch, mb, n).sum(0) - ref).abs() / scale).max()), "modes": {}}
        sample = sorted({ntiles - 1, *rng.sample(range(ntiles), gate_tiles)}) if first else [rng.randrange(ntiles)]
        j = BR.batch_blocks(mb)
        xnp = x.view(torch.uint8).cpu().numpy()
        znp = zs[mb].cpu().numpy()
        accs = {}
        for mode in MODES:
            g = {}
            pf = torch.full_like(pp, float("nan"))
            sk.fill_(POISON)
            tags.fill_(POISON)
            rb(mode, x, pf, mb, True)
            torch.cuda.synchronize()
            g["partials_equal"] = bool(torch.equal(pf, pp))
            g["written"] = not bool((sk.view(-1, 16) == POISON).all(1).any())
            g["receipts_written"] = not bool((tags.view(-1, 8) == POISON).all(1).any())
            sk1, t1 = sk.clone(), tags.clone()
            sk.fill_(POISON)
            rb(mode, x, torch.empty_like(pp), mb, True, hashed=False)
            torch.cuda.synchronize()
            g["stable"] = bool(torch.equal(sk, sk1))
            dev = sk1.view(ntiles, 16, 16).cpu().numpy().astype(np.int64)
            masks_m = signs_np if mode == "unbound" else masks_np
            bad = []
            for t in sample:
                if mode.startswith("slice") and t not in accs:
                    G, kc = divmod(t, nch)
                    xc = np.zeros((8 * j, f.task_cols), np.uint8)
                    xc[:mb] = xnp[:, kc * f.task_cols:(kc + 1) * f.task_cols]
                    accs[t] = BR.accumulators(wnp[64 * G:64 * G + 64, kc * f.task_cols:(kc + 1) * f.task_cols], xc)
                want = BR.tile_sketch(mode, tile_vectors(store, n, kd, t), masks_m, znp, t, ntiles, accs.get(t))
                if not np.array_equal(want, dev[t]):
                    bad.append(t)
            g["sketch_vs_reference"] = {"sampled": sample, "mismatch": bad}
            vs = sorted({0, nvec - 1, *rng.sample(range(nvec), 24)})
            want = BR.receipts_of(TAG_SEED, dev.reshape(-1, 16), 0, which=vs,
                                  name=WS.NAME if mode == "unbound" else BR.NAME)
            devt = t1.view(-1, 8).cpu().numpy()
            g["receipts_vs_reference"] = {"sampled": len(vs), "mismatch": [v for v, yv in zip(vs, want)
                                                                            if devt[v].tobytes() != yv][:8]}
            ok = (g["partials_equal"] and g["written"] and g["receipts_written"] and g["stable"] and not bad
                  and not g["receipts_vs_reference"]["mismatch"])
            if mode == "unbound" and mb <= 32:
                sk2.fill_(POISON)
                tags2.fill_(POISON)
                lib.call("ww_gemm_sketch", k, ptr(store), ptr(x.view(torch.uint8)), ptr(torch.empty_like(pp)), mb, n, kd,
                         0, 1, ptr(signs), ptr(sk2), 0, seed_buf, tags2_p, 0, stream(torch))
                torch.cuda.synchronize()
                g["ww_gemm_sketch_equal"] = bool(torch.equal(sk2, sk1) and torch.equal(tags2, t1))
                ok = ok and g["ww_gemm_sketch_equal"]
            if mode != "unbound" and first and gi == 0:
                tags.fill_(POISON)
                lib.call("rb_receipts", ptr(sk1), nvec, 0, seed_buf, tags_p, stream(torch))
                torch.cuda.synchronize()
                g["apart_equal"] = bool(torch.equal(tags, t1))
                root = ctypes.create_string_buffer(64)
                lib.call("rb_commit", ptr(ws), nvec, seed_buf, root, stream(torch))
                g["root_vs_reference"] = root.raw == BR.root(TAG_SEED, WB.tag_bytes(t1))
                ok = ok and g["apart_equal"] and g["root_vs_reference"]
            g["ok"] = bool(ok)
            entry["modes"][mode] = g
        pf = torch.full_like(pp, float("nan"))
        lib.call("rb_gemm", PLAIN, k, ptr(store), ptr(x.view(torch.uint8)), ptr(pf), mb, n, kd, 0, 1, None, None, 0,
                 None, None, None, 0, stream(torch))
        torch.cuda.synchronize()
        entry["plain_partials_equal"] = bool(torch.equal(pf, pp))
        entry["ok"] = (entry["payload_vs_torch_rel"] < 1e-5 and entry["plain_partials_equal"]
                       and all(g["ok"] for g in entry["modes"].values()))
        res["gate"][mb] = entry
        log(f"gate {name} M={mb}: {'ok' if entry['ok'] else 'FAIL'} (tiles {sample}, payload rel "
            f"{entry['payload_vs_torch_rel']:.1e})" + ("" if entry["ok"] else f" {entry}"))
        if not entry["ok"]:
            res["ok"] = False
            return res
    res["ok"] = True

    copies = max(2, -(-B.L2_DEFEAT_BYTES // (pay.numel() * 4)))
    ring = [store] + [store.clone() for _ in range(copies - 1)]
    wring = [w] + [w.clone() for _ in range(copies - 1)]
    res["copies"] = copies
    so, st = lib.so, stream(torch)
    pay = None
    for mb in batches:
        x = xs[mb]
        xp = ptr(x.view(torch.uint8))
        y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        yp = ptr(y)
        one = torch.ones((), device="cuda")
        try:
            torch._scaled_mm(x, w.t(), one, one, out_dtype=torch.bfloat16)
            xc, mpad = x, mb
        except Exception:  # noqa: BLE001  cuBLASLt's FP8 path needs M a multiple of 16 here
            mpad = 16 * -(-mb // 16)
            xc = torch.zeros(mpad, kd, device="cuda").to(torch.float8_e4m3fn)
            xc[:mb] = x
        wts = [c.t() for c in wring]
        ps = [ptr(b) for b in ring]

        def arm_native():
            for wt in wts:
                torch._scaled_mm(xc, wt, one, one, out_dtype=torch.bfloat16)

        def arm_d():
            rc = 0
            for p in ps:
                rc |= so.ww_gemm(k, p, xp, yp, mb, n, kd, 0, 0, None, None, 0, st)
            if rc:
                raise RuntimeError(f"ww_gemm k={k} M={mb}: error {rc}")

        def arm_h():
            rc = 0
            for p in ps:
                rc |= so.ww_gemm_sketch(k, p, xp, yp, mb, n, kd, 0, 0, ptr(signs), None, 0, seed_buf, tags_p, 0, st)
            if rc:
                raise RuntimeError(f"ww_gemm_sketch k={k} M={mb}: error {rc}")

        def arm_rb(mode):
            md = ARMS[mode]
            mk = None if mode == "plain" else ptr(signs if mode == "unbound" else masks)
            zp = ptr(zs[mb]) if md not in (PLAIN, 0) else None
            sb, tp = (None, None) if mode == "plain" else (seed_buf, tags_p)

            def go():
                rc = 0
                for p in ps:
                    rc |= so.rb_gemm(md, k, p, xp, yp, mb, n, kd, 0, 0, mk, None, 0, sb, tp, zp, n, st)
                if rc:
                    raise RuntimeError(f"rb_gemm {mode} M={mb}: error {rc}")
            return go

        arms = {"native": arm_native}
        if mb <= 32:
            arms["d"], arms["h"] = arm_d, arm_h
        for mode in ARMS:
            arms[mode] = arm_rb(mode)
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        times = {a: [] for a in arms}
        names = list(arms)
        for rep in range(reps):
            for a in names[rep % len(names):] + names[:rep % len(names)]:
                times[a].append(WB.queued(lib, torch, arms[a]) / copies)
        med = {a: statistics.median(v) for a, v in times.items()}
        res["times"][mb] = {"median_s": med, "times_s": times, "cublas_m": mpad,
                            "ratio_vs_native": {a: med[a] / med["native"] for a in med}}
        r = res["times"][mb]["ratio_vs_native"]
        log(f"time {name} M={mb}: native {med['native'] * 1e6:.1f} us; "
            + (f"d {r['d']:.3f}x h {r['h']:.3f}x; " if "d" in r else "")
            + ", ".join(f"{m} {r[m]:.3f}x" for m in ARMS)
            + f" (CTAs an SM {res['occupancy'][mb]})")
    ring = wring = store = None
    torch.cuda.empty_cache()
    return res


def layer(results: dict, batches) -> dict:
    """Per batch, each arm summed over the linears, over native's sum; the binding cost over unbound."""
    out = {}
    for mb in batches:
        arms = set.intersection(*(set(r["times"][mb]["median_s"]) for r in results.values()))
        tot = {a: sum(r["times"][mb]["median_s"][a] for r in results.values()) for a in arms}
        ratio = {a: tot[a] / tot["native"] for a in tot}
        out[mb] = {"total_s": tot, "ratio_vs_native": ratio,
                   "receipt_cost": {m: ratio[m] - ratio["plain"] for m in MODES},
                   "binding_cost": {m: ratio[m] - ratio["unbound"] for m in MODES if m != "unbound"},
                   "z_over_block": {m: statistics.mean(BR.binding_ratio(m, mb, r["K"]) for r in results.values())
                                    for m in MODES}}
    return out


def write_result(res: dict, out_dir: Path) -> None:
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = []
    for k, rk in res["by_k"].items():
        for mb, lay in rk.get("layer", {}).items():
            for kind in ("ratio_vs_native", "receipt_cost", "binding_cost"):
                for a, v in lay[kind].items():
                    ms.append({"name": f"{kind}_{a}_k{k}_m{mb}", "value": v, "unit": "1"})
    rel = os.path.relpath(out_dir, run_dir)
    files = [f"{rel}/bound.json", f"{rel}/bench.log"]
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {k: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                             for k, p in res["lib_paths"].items()},
                                    "seed": WB.SEED.hex(), "tag_seed": TAG_SEED.hex()},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "bound receipt gates",
                          "detail": "partial sums = payload store's; sketches = bound.py (sm_120 E4M3 model for the "
                                    "accumulators); receipts and root = ro-bound-sketch-blake3/v0's; unbound = "
                                    "ww_gemm_sketch's", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="PATH of a bound.cu build (repeat, one a k)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=sorted(WB.MODELS), default="llama-3.1-70b")
    ap.add_argument("--shapes", help="a subset of the model's linears (default: all four)")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--gate-tiles", type=int, default=2, help="random tiles checked on the first linear (one on the rest)")
    args = ap.parse_args()
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "bench.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    WB.use_model(args.model)
    libs = [Lib(f"bound-{i}", p) for i, p in enumerate(args.lib)]
    for lib in libs:
        if lib.bits != 512:
            raise SystemExit(f"{lib.path} is p{lib.bits}; this bench is p512")
    if len({lib.rb_k for lib in libs}) != len(libs):
        raise SystemExit("one library a k")
    batches = [int(b) for b in args.batches.split(",")]
    shapes = {s: B.SHAPES[s] for s in (args.shapes.split(",") if args.shapes else B.SHAPES)}
    res = {"lib_paths": {lib.rb_k: lib.path for lib in libs}, "model": args.model, "batches": batches,
           "gpu": torch.cuda.get_device_name(0), "by_k": {}}
    for lib in libs:
        k = lib.rb_k
        rk = res["by_k"][k] = {"occ_build": lib.rb_occ, "cases": {}}
        for i, (name, (n, kd)) in enumerate(shapes.items()):
            rk["cases"][name] = case(lib, torch, name, n, kd, batches, args.reps, log, args.gate_tiles, i == 0)
            if not rk["cases"][name]["ok"]:
                break
        rk["ok"] = all(c["ok"] for c in rk["cases"].values()) and len(rk["cases"]) == len(shapes)
        if rk["ok"]:
            rk["layer"] = layer(rk["cases"], batches)
            for mb, lay in rk["layer"].items():
                r, t, c = lay["ratio_vs_native"], lay["receipt_cost"], lay["binding_cost"]
                log(f"layer k={k} M={mb}: " + ", ".join(f"{a} {r[a]:.3f}" for a in sorted(r))
                    + "; receipt cost over plain " + ", ".join(f"{m} {t[m]:+.3f}" for m in t)
                    + "; binding cost over unbound " + ", ".join(f"{m} {c[m]:+.3f}" for m in c))
    res["ok"] = all(rk["ok"] for rk in res["by_k"].values())
    (out / "bound.json").write_text(json.dumps(res, indent=1, default=str))
    write_result(res, out)
    log("ok" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
