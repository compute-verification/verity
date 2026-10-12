"""The KV cache's read and write sides at an HBM-exhaustion shape (the rulings of 12 Oct: HBM holds the encoded weights,
the KV cache and filler, and every step reads all of it), on one GPU: `kv.cu`'s libraries (`--lib NAME=PATH`, built with
KV_QHEADS the model's query heads) gated, then timed. Writes `kv-exhaust.json` and `kv.log` under `--out`, and in a
`research run` its `result.json`.

Nothing is timed before its gate passed:
- `gate`: kv_bench's codec gate at every k of `--ks`;
- `attn_gate`: at each batch x context of `--gate-cells`, with the model's layers, every `kv_attn` form at every split
  count of SPLITS, at layers 0 and L - 1: each encoded form's output equals its plain form's bit for bit (the CUDA-core
  forms 1 and 2 form 0's, the tensor-core form 4 form 3's), and each plain form is within RTOL of float64 attention
  over the same bf16 values.

Then, at each batch x context of `--cells`, one store holding the cell's whole KV cache (batch x context x layers
token-layers in kv.py's slot layout, so a layer's call reads 4 KB a token at the slots' stride) of random canonical
words:
- `read`: R over the store (kv_bench's read kernel), and the store's read-and-decode at each k, the floor where the
  decode overlaps the read and nothing else runs (gated first: its checksum at every timed grid, where each thread
  steps its tweak, equals the checksum at one word a thread);
- `attn`: a pass over every layer, each form at each split count, queued behind a spin, its best split count's time.
  Plain forms read the words as bf16 (their arithmetic's time doesn't depend on the values); encoded forms decode them.
  The read side's overhead a KV byte, in reads of the stored bytes at R, is (encoded - read) / read, and against the
  same family's plain form (encoded - plain) / read; each is fitted over k as c k - h.

`write`: the step's new tokens' encode (batch x layers x a token-layer's words) at each k: alone (queued behind a spin),
at the compute-only rate, and beside the step's weight GEMM (cuBLASLt FP8 at the model's linears, every layer's own
cold weights) on a second stream of lower priority, at several grids, against the GEMM alone.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import kv as K  # noqa: E402
import kv_bench as B  # noqa: E402

MODELS = {
    "8b": {"heads": 32, "layers": 32,
           "shapes": {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}},
    "70b": {"heads": 64, "layers": 80, "shapes": B.SHAPES_70B},
}
FORMS = {0: "fma-plain", 1: "fma-phased", 2: "fma-ws", 3: "mma-plain", 4: "mma-phased", 5: "mma-ws"}
PLAIN_OF = {1: 0, 2: 0, 4: 3, 5: 3}
SPLITS = (1, 2, 4, 8, 16)
RTOL = {0: 1e-3, 3: 8e-3}  # the tensor-core form rounds P to bf16
SPIN = 60_000_000
WRITE_GRIDS = (0, 1, 2, 4)  # the encode's grid beside the GEMM: a thread a word (0), or this many CTAs an SM


def shape(lib: B.Lib) -> dict:
    a = (ctypes.c_int * 10)()
    lib.so.kv_attn_shape(ctypes.cast(a, ctypes.c_void_p))
    return dict(zip(("heads", "kv_heads", "head_dim", "tlw", "tle", "tt", "part", "mw", "mma", "mt"), a))


def forms_of(sh: dict) -> list[int]:
    return [0, 1, 2] + ([3, 4, 5] if sh["mma"] else [])


def attn_call(lib, st, store, form, q, o, part, b, ctx, layers, layer, splits, k) -> None:
    lib.call("kv_attn", B.ptr(store), form, B.ptr(q), B.ptr(o), B.ptr(part), b, ctx, layers, layer, splits, k, st)


def part_buffer(torch, sh: dict, b: int):
    return torch.empty(b * max(SPLITS) * sh["mw"] * sh["part"], dtype=torch.float32, device="cuda")


def attn_gate(lib: B.Lib, torch, m: dict, cells, ks, log, layers_override: int | None = None) -> dict:
    sh, f = shape(lib), lib.field
    heads, layers = sh["heads"], layers_override or m["layers"]
    if heads != m["heads"] or sh["tlw"] != K.token_layer_words(f):
        raise RuntimeError(f"{lib.name}: kv_attn_shape {sh} is not the model's ({m['heads']} heads)")
    st = B.stream(torch)
    tl, tle = K.TOKEN_LAYER_BYTES, sh["tle"]
    scale = math.log2(math.e) / math.sqrt(K.HEAD_DIM)
    forms = forms_of(sh)
    out = {}
    for b, ctx in cells:
        g = torch.Generator(device="cuda").manual_seed(b * 100003 + ctx)
        plain = torch.randn(b * ctx * layers * tl // 2, dtype=torch.float32, device="cuda", generator=g).to(
            torch.bfloat16)
        pad = torch.zeros(b * ctx * layers, tle, dtype=torch.uint8, device="cuda")
        pad[:, :tl] = plain.view(torch.uint8).view(-1, tl)
        enc = torch.empty_like(pad)
        q = (torch.randn(b, heads, K.HEAD_DIM, device="cuda", generator=g) * scale).to(torch.bfloat16).contiguous()
        part = part_buffer(torch, sh, b)
        o = {fm: torch.empty(b, heads, K.HEAD_DIM, dtype=torch.float32, device="cuda") for fm in forms}
        kvl = plain.view(b, ctx, layers, 2, K.KV_HEADS, K.HEAD_DIM)
        cell = {"batch": b, "context": ctx, "layers": layers, "splits": list(SPLITS), "equal": {}, "rel_err": {},
                "mma_vs_fma": 0.0}
        for k in ks:
            B.set_key(lib, K.key(f.name, B.SEED, k))
            lib.call("kv_encode", B.ptr(pad), B.ptr(enc), pad.numel() // f.word_bytes, 0, k, 0, st)
            for layer in (0, layers - 1):
                kk, vv = kvl[:, :, layer, 0].double(), kvl[:, :, layer, 1].double()
                qq = q.double().view(b, K.KV_HEADS, heads // K.KV_HEADS, K.HEAD_DIM)
                s = torch.einsum("bgqd,btgd->bgqt", qq, kk)
                p = torch.exp2(s - s.amax(-1, keepdim=True))
                ref = torch.einsum("bgqt,btgd->bgqd", p / p.sum(-1, keepdim=True), vv).reshape(b, heads, K.HEAD_DIM)
                del kk, vv, s, p
                for sp in SPLITS:
                    for fm in forms:
                        o[fm].fill_(float("nan"))
                        attn_call(lib, st, pad if fm in RTOL else enc, fm, q, o[fm], part, b, ctx, layers, layer, sp, k)
                    torch.cuda.synchronize()
                    for fm in forms:
                        if fm in RTOL:
                            err = float((o[fm].double() - ref).abs().max() / ref.abs().max())
                            cell["rel_err"][FORMS[fm]] = max(cell["rel_err"].get(FORMS[fm], 0.0), err)
                        else:
                            key = f"{FORMS[fm]}@k{k}"
                            cell["equal"][key] = cell["equal"].get(key, True) and bool(torch.equal(o[fm], o[PLAIN_OF[fm]]))
                    if 3 in forms:
                        d = float((o[3].double() - o[0].double()).abs().max() / ref.abs().max())
                        cell["mma_vs_fma"] = max(cell["mma_vs_fma"], d)
                del ref
        cell["ok"] = all(cell["equal"].values()) and all(
            cell["rel_err"].get(FORMS[fm], 1.0) <= tol for fm, tol in RTOL.items() if fm in forms)
        log(f"attn gate {lib.name} B={b} ctx={ctx} L={layers}: {'ok' if cell['ok'] else 'FAIL'}; equal "
            + ", ".join(f"{n} {v}" for n, v in cell["equal"].items()) + "; rel err "
            + ", ".join(f"{n} {v:.1e}" for n, v in cell["rel_err"].items()) + f"; mma vs fma {cell['mma_vs_fma']:.1e}")
        out[f"b{b}_ctx{ctx}"] = cell
        del plain, pad, enc, q, part, o, kvl
        torch.cuda.empty_cache()
    return out


def fit(ks, ys) -> dict:
    """y = c k - h by least squares."""
    n = len(ks)
    if n < 2:
        return {}
    mk, my = sum(ks) / n, sum(ys) / n
    c = sum((a - mk) * (b - my) for a, b in zip(ks, ys)) / sum((a - mk) ** 2 for a in ks)
    return {"c": c, "h": c * mk - my}


def floor_gate(lib: B.Lib, torch, store, n_words: int, k: int, grids, st) -> list:
    """The grids whose read-and-decode checksum differs from one word a thread's (no tweak stepping)."""
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")

    def checksum(grid):
        xr.zero_()
        lib.call("kv_decode", B.ptr(store), 0, n_words, 0, k, grid, B.ptr(xr), st)
        torch.cuda.synchronize()
        return int(xr.item()) & 0xFFFFFFFF

    want = checksum(0)
    return [g for g in grids if checksum(g) != want]


def exhaust_cell(lib: B.Lib, torch, m: dict, b: int, ctx: int, ks, reps: int, sms: int, log) -> dict:
    sh, f = shape(lib), lib.field
    heads, layers, tle = sh["heads"], m["layers"], sh["tle"]
    st = B.stream(torch)
    n_words = b * ctx * layers * sh["tlw"]
    store = B.random_words(torch, n_words, f, b * 7919 + ctx)
    per_layer = b * ctx * tle
    rd = B.read_rate(lib, torch, store, sms, reps)
    r = rd["gbps"] * 1e9
    t_read = per_layer / r
    cell = {"batch": b, "context": ctx, "layers": layers, "store_bytes": store.numel(), "layer_bytes": per_layer,
            "read": rd, "read_layer_s": t_read, "floor": {}, "forms": {}}
    log(f"exhaust {lib.name} B={b} ctx={ctx} L={layers}: store {store.numel() / 1e9:.1f} GB, read {rd['gbps']:.0f} GB/s")
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    for k in ks:
        B.set_key(lib, K.key(f.name, B.SEED, k))
        bad = floor_gate(lib, torch, store, n_words, k, [sms * mult for mult in (4, 8, 16)], st)
        if bad:
            cell["floor"][k] = {"ok": False, "bad_grids": bad}
            log(f"exhaust {lib.name} B={b} ctx={ctx} k={k}: read-and-decode checksum FAIL at grids {bad}")
            continue
        best = None
        for mult in (4, 8, 16):
            ts = B.timed(torch, lambda: lib.call("kv_decode", B.ptr(store), 0, n_words, 0, k, sms * mult, B.ptr(xr), st),
                         reps)
            if best is None or statistics.median(ts) < best[1]:
                best = (mult, statistics.median(ts), ts)
        cell["floor"][k] = {"grid_per_sm": best[0], "s": best[1], "times_s": best[2],
                            "over_read": best[1] * r / store.numel()}
    log(f"exhaust {lib.name} B={b} ctx={ctx}: read-and-decode " + ", ".join(
        f"k={k} {v['over_read']:.3f}x the read" for k, v in cell["floor"].items() if "s" in v))
    q = (torch.randn(b, heads, K.HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16).contiguous()
    o = torch.empty(b, heads, K.HEAD_DIM, dtype=torch.float32, device="cuda")
    part = part_buffer(torch, sh, b)
    for fm in forms_of(sh):
        for k in ([1] if fm in RTOL else ks):
            B.set_key(lib, K.key(f.name, B.SEED, k))
            per = {}
            ctas = (lambda s: b * K.KV_HEADS * s) if fm >= 3 else (lambda s: b * s)
            for sp in [s for s in SPLITS if ctas(s) >= sms] or [max(SPLITS)]:
                def run_pass():
                    for l in range(layers):
                        attn_call(lib, st, store, fm, q, o, part, b, ctx, layers, l, sp, k)
                run_pass()
                ts = [B.queued(lib, torch, run_pass, SPIN) / layers for _ in range(reps)]
                per[sp] = {"s": statistics.median(ts), "times_s": ts}
            sp = min(per, key=lambda s: per[s]["s"])
            row = {"splits": sp, "s": per[sp]["s"], "by_splits": per, "over_read": per[sp]["s"] / t_read,
                   "gbps": per_layer / per[sp]["s"] / 1e9}
            cell["forms"].setdefault(FORMS[fm], {})[k] = row
            log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]}" + ("" if fm in RTOL else f" k={k}")
                + f": a layer {row['s'] * 1e3:.3f} ms at {sp} splits, {row['gbps']:.0f} GB/s, {row['over_read']:.3f}x"
                " the read")
    for fm in forms_of(sh):
        if fm in RTOL:
            continue
        fe, fp = cell["forms"][FORMS[fm]], cell["forms"][FORMS[PLAIN_OF[fm]]][1]
        for k, row in fe.items():
            row["overhead_vs_read"] = row["over_read"] - 1.0
            row["overhead_vs_plain"] = (row["s"] - fp["s"]) / t_read
        fe["fit_vs_read"] = fit(ks, [fe[k]["overhead_vs_read"] for k in ks])
        fe["fit_vs_plain"] = fit(ks, [fe[k]["overhead_vs_plain"] for k in ks])
    ks_f = [k for k, v in cell["floor"].items() if v.get("ok", True)]
    cell["ok"] = len(ks_f) == len(cell["floor"])
    cell["floor_fit"] = fit(ks_f, [cell["floor"][k]["over_read"] - 1.0 for k in ks_f])
    log(f"exhaust {lib.name} B={b} ctx={ctx}: overhead a KV byte over the read, " + "; ".join(
        f"{FORMS[fm]} " + ", ".join(f"k={k} {cell['forms'][FORMS[fm]][k]['overhead_vs_read']:+.3f}" for k in ks)
        for fm in forms_of(sh) if fm not in RTOL) + "; floor " + ", ".join(
        f"k={k} {cell['floor'][k]['over_read'] - 1:+.3f}" for k in ks_f))
    del store, q, o, part
    torch.cuda.empty_cache()
    return cell


def weights(torch, m: dict):
    out = {}
    for name, (n, kd) in m["shapes"].items():
        g = torch.Generator(device="cuda").manual_seed(n + kd)
        w = (torch.randn(n, kd, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
        out[name] = [w.t()] + [w.clone().t() for _ in range(m["layers"] - 1)]
    return out


def write_side(lib: B.Lib, torch, m: dict, batches, ks, reps: int, sms: int, enc_rates: dict, log) -> dict:
    f = lib.field
    tw = m["layers"] * K.token_layer_words(f)
    one = torch.ones((), device="cuda")
    ws = weights(torch, m)
    w_bytes = sum(n * kd for n, kd in m["shapes"].values()) * m["layers"]
    hi = torch.cuda.Stream(priority=-1)
    lo = torch.cuda.Stream(priority=0)
    main = torch.cuda.current_stream()
    out = {"weight_bytes": w_bytes, "token_words": tw}
    for b in batches:
        xs = {name: (torch.randn(b, kd, device="cuda") * 0.5).to(torch.float8_e4m3fn)
              for name, (n, kd) in m["shapes"].items()}
        words = B.random_words(torch, tw * b, f, 11 + b)
        y = torch.empty_like(words)
        n = tw * b

        def gemm():
            for l in range(m["layers"]):
                for name in m["shapes"]:
                    torch._scaled_mm(xs[name], ws[name][l], one, one, out_dtype=torch.bfloat16)

        def both(do_gemm: bool, enc_grid: int | None, k: int) -> float:
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            lib.call("kv_spin", SPIN, main.cuda_stream)
            a.record(main)
            ends = []
            if do_gemm:
                hi.wait_event(a)
                with torch.cuda.stream(hi):
                    gemm()
                ends.append(torch.cuda.Event())
                ends[-1].record(hi)
            if enc_grid is not None:
                lo.wait_event(a)
                lib.call("kv_encode", B.ptr(words), B.ptr(y), n, 0, k, enc_grid, lo.cuda_stream)
                ends.append(torch.cuda.Event())
                ends[-1].record(lo)
            for ev in ends:
                main.wait_event(ev)
            e.record(main)
            e.synchronize()
            return a.elapsed_time(e) / 1e3

        gemm()
        torch.cuda.synchronize()
        t_gemm = statistics.median([both(True, None, 1) for _ in range(reps)])
        row = {"gemm_s": t_gemm, "gemm_gbps": w_bytes / t_gemm / 1e9, "words": n, "bytes": n * f.word_bytes, "k": {}}
        for k in ks:
            B.set_key(lib, K.key(f.name, B.SEED, k))
            kr = {"sm_s": n / enc_rates[k]["per_s"], "grids": {}}
            for gr in WRITE_GRIDS:
                grid = gr * sms
                both(False, grid, k)
                t_enc = statistics.median([both(False, grid, k) for _ in range(reps)])
                t_both = statistics.median([both(True, grid, k) for _ in range(reps)])
                kr["grids"][gr] = {"grid": grid, "alone_s": t_enc, "with_gemm_s": t_both,
                                   "hidden_s": t_gemm + t_enc - t_both,
                                   "hidden_share": (t_gemm + t_enc - t_both) / t_enc}
            best = min(kr["grids"], key=lambda g_: kr["grids"][g_]["with_gemm_s"] - t_gemm)
            kr["best_grid"] = best
            kr["alone_s"] = min(v["alone_s"] for v in kr["grids"].values())
            kr["exposed_s"] = kr["grids"][best]["with_gemm_s"] - t_gemm
            row["k"][k] = kr
            log(f"write {lib.name} B={b} k={k}: {n * f.word_bytes / 1e6:.1f} MB; encode alone {kr['alone_s'] * 1e3:.2f} ms "
                f"(compute-only rate {kr['sm_s'] * 1e3:.2f} ms); GEMM alone {t_gemm * 1e3:.2f} ms "
                f"({row['gemm_gbps']:.0f} GB/s); beside it " + ", ".join(
                    f"grid {g_}: +{(v['with_gemm_s'] - t_gemm) * 1e3:.2f} ms ({v['hidden_share']:.0%} hidden)"
                    for g_, v in kr["grids"].items()))
        out[b] = row
        del xs, words, y
    del ws
    torch.cuda.empty_cache()
    return out


def model_check(res: dict, lib_name: str, m: dict) -> dict:
    """f_KV k x1 / ctx (the write side in reads of the step's bytes) against the measured encode, at each cell's
    batch and context: x1 is the compute-only encode a byte a round, in reads at R."""
    lr = res["libs"][lib_name]
    r = res["read"]["gbps"] * 1e9
    wb = K.field(lib_name.split("-")[0]).word_bytes
    x1 = r / (lr["rates"]["encode"][1]["per_s"] * wb) if 1 in lr["rates"]["encode"] else None
    out = {"x1": x1, "rows": []}
    w = lr.get("write", {})
    for cell in lr.get("exhaust", {}).values():
        b, ctx = cell["batch"], cell["context"]
        kv_bytes = cell["store_bytes"]
        step_read = (kv_bytes + w.get("weight_bytes", 0)) / r
        f_kv = kv_bytes / (kv_bytes + w.get("weight_bytes", 0))
        for k, kr in w.get(b, {}).get("k", {}).items():
            out["rows"].append({"batch": b, "context": ctx, "k": k, "f_kv": f_kv, "step_read_s": step_read,
                                "predicted": f_kv * k * x1 / ctx if x1 else None,
                                "measured_alone": kr["alone_s"] / step_read,
                                "measured_exposed": kr["exposed_s"] / step_read})
    return out


def write_result(res: dict, libs, out_dir: Path) -> None:
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = [{"name": "read", "value": res["read"]["gbps"], "unit": "GB/s"}] if "read" in res else []
    for lib in libs:
        for name, cell in res["libs"][lib.name].get("exhaust", {}).items():
            for form, rows in cell["forms"].items():
                for k, row in rows.items():
                    if isinstance(k, int):
                        ms.append({"name": f"{lib.name}_{name}_{form}_k{k}", "value": row["s"], "unit": "s"})
    rel = os.path.relpath(out_dir, run_dir)
    files = [f"{rel}/kv-exhaust.json", f"{rel}/kv.log"]
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {l.name: hashlib.sha256(Path(l.path).read_bytes()).hexdigest() for l in libs},
                                    "seed": B.SEED.hex()},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "kv codec and attention gates",
                          "detail": "device encode and decode = kv.py; each encoded attention form = its plain form bit "
                                    "for bit; plain forms within RTOL of float64", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


def cells_of(s: str):
    return [tuple(int(v) for v in c.split("x")) for c in s.split(",") if c]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="NAME=PATH, NAME a field of kv.FIELDS (or FIELD-TAG)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="8b", choices=sorted(MODELS))
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--gate-cells", default="4x1000,2x4096")
    ap.add_argument("--cells", default="160x4096,80x8192,40x16384")
    ap.add_argument("--write-batches", default="", help="the write side's batches (default: the cells')")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sanitize", action="store_true", help="one tiny gate cell of 2 layers, then stop")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "kv.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    m = MODELS[a.model]
    ks = [int(v) for v in a.ks.split(",")]
    libs = [B.Lib(*spec.split("=", 1)) for spec in a.lib]
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    res = {"device": props.name, "sms": sms, "total_memory": props.total_memory, "model": a.model, "ks": ks,
           "libs": {}, "ok": True}
    log(f"{props.name}: {sms} SMs, {props.total_memory / 1e9:.1f} GB; model {a.model}; libs "
        + ", ".join(f"{l.name} {shape(l)}" for l in libs))
    if a.sanitize:
        for lib in libs:
            g = attn_gate(lib, torch, m, [(2, 100)], ks[:1], log, layers_override=2)
            res["ok"] &= all(c["ok"] for c in g.values())
            B.set_key(lib, K.key(lib.field.name, B.SEED, ks[0]))
            bad = floor_gate(lib, torch, B.random_words(torch, 5000, lib.field, 1), 5000, ks[0], [3], B.stream(torch))
            log(f"floor gate {lib.name} k={ks[0]}: " + (f"FAIL at grids {bad}" if bad else "ok"))
            res["ok"] &= not bad
        log("sanitize smoke done" if res["ok"] else "sanitize smoke FAILED")
        return 0 if res["ok"] else 1
    for lib in libs:
        g = B.gate(lib, torch, ks, 48, 1 << 16, log)
        ag = attn_gate(lib, torch, m, cells_of(a.gate_cells), ks, log)
        res["libs"][lib.name] = {"gate": g, "attn_gate": ag}
        res["ok"] &= all(v["ok"] for v in g.values()) and all(c["ok"] for c in ag.values())
    if not res["ok"]:
        log("a gate failed: nothing timed")
    else:
        rbuf = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")
        rbuf.random_(0, 256)
        res["read"] = B.read_rate(libs[0], torch, rbuf, sms, a.reps)
        log(f"read: {res['read']['gbps']:.0f} GB/s over 4 GiB")
        del rbuf
        torch.cuda.empty_cache()
        for lib in libs:
            lr = res["libs"][lib.name]
            lr["rates"] = B.rates(lib, torch, ks, sms, a.reps, log)
            lr["exhaust"] = {}
            for b, ctx in cells_of(a.cells):
                need = b * ctx * m["layers"] * shape(lib)["tle"]
                if need + (6 << 30) > props.total_memory:
                    log(f"exhaust {lib.name} B={b} ctx={ctx}: {need / 1e9:.1f} GB doesn't fit; skipped")
                    continue
                cell = lr["exhaust"][f"b{b}_ctx{ctx}"] = exhaust_cell(lib, torch, m, b, ctx, ks, a.reps, sms, log)
                res["ok"] &= cell["ok"]
            wb = [int(v) for v in a.write_batches.split(",") if v] or sorted({b for b, _ in cells_of(a.cells)})
            if wb:
                lr["write"] = write_side(lib, torch, m, wb, ks, a.reps, sms, lr["rates"]["encode"], log)
            lr["model_check"] = model_check(res, lib.name, m)
            for row in lr["model_check"]["rows"]:
                log(f"model {lib.name} B={row['batch']} ctx={row['context']} k={row['k']}: f_KV k x1/ctx "
                    f"{row['predicted']:.4f} against the encode alone {row['measured_alone']:.4f} and exposed beside "
                    f"the GEMM {row['measured_exposed']:.4f} (reads of the step's bytes)")
    (out_dir / "kv-exhaust.json").write_text(json.dumps(res, indent=1, default=str))
    write_result(res, libs, out_dir)
    log("done" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
