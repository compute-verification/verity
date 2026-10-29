"""J10-R's proof blocks against plain exact INT8 at the frozen shape's linears, interleaved, on one GPU:

    python bench.py --fixtures DIR --out OUT --scratch S [--h1u NAME=PATH ...] [--ref PATH] [--sm]
                    [--wgmma DIR] [--check check.json] [--ms 4096,16384] [--linears q,kv,o,up,down,head] [--reps 7]
                    [--ref-reps 2]

One call a linear and M, every row live, its operands `shape.whole`'s (the demo's served weights, the fixtures' A), its
scales `shape.weight_scales` and `activation_scales`. The arms, timed by CUDA events one call at a time, in an order that
rotates each rep:

* `plain`: s8_linear's `int_mm.linear(x, sx, w, sw)` (the exact W8A8 linear's int_mm backend: `torch._int_mm` and
  the fused epilogue, the eager one where there is no Triton);
* `int_mm`: `torch._int_mm` alone, the s32 GEMM;
* `wgmma` (--wgmma DIR, s8_linear's wgmma.py and configs.py): s8_linear's exact linear, wgmma.scaled_mm at its tuned
  config for the linear and M, into an output allocated once, held to plain's output on every cell;
* each h1u build NAME: the call's J10-R receipts, kh_seeds_j10r then kh_units_multi_j10r (with --sm
  kh_units_multi_j10r_sm on every SM, the served launch);
* `plain+NAME`, `wgmma+NAME`: that plain arm, then NAME's proof blocks, in one timed region (what serving them one after the
  other costs);
* `ref` (--ref, --ref-reps reps): the fallback's seeds and receipts over every proof block of the call (jr_seeds, jr_units).

After the reps, every units arm runs once more under each poison (`units.twice`: no word unwritten or unstable) and is
held to the fixtures: its sampled receipts and seed words, and plain's s32 sums at the sampled tiles, so every number
comes from a build that was exact. overhead = units / plain and gamma =
(plain + units) / plain, per call, per layer (q, kv, o, up and down) and per step (52 layers and the head), and the same
against wgmma under `vs_wgmma`. check.json
(--check), when given, is copied into the report's `check` verdicts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shape as S
import units as UN

import fixtures as F

#: the plain arms a units arm is measured against, each with its serial arms (BASE+NAME)
BASELINES = ("plain", "wgmma")

def plain_of():
    """(name, fn(x, sx, w, sw)): s8_linear's `int_mm.linear`, the fused epilogue where Triton is, else the eager one."""
    spec = importlib.util.spec_from_file_location("int_mm", Path(__file__).resolve().parents[1] / "s8_linear" / "int_mm.py")
    mod = sys.modules["int_mm"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    epi = "fused" if mod.triton is not None else "eager"
    return f"int_mm.linear ({epi})", lambda x, sx, w, sw: mod.linear(x, sx, w, sw, epilogue=epi)


def wgmma_of(path: str | None) -> dict | None:
    """s8_linear's exact linear (its wgmma.py and configs.py in the directory `path`, the cubin $S8_WGMMA_CUBIN):
    its identity, `cfg(m, n, k)` (configs.CONFIGS' tuned pick for (N, K) at M, as its run.py's table_cfg, else wgmma's
    default) and `fn(x, sx, w, sw, y, cfg)`, wgmma.scaled_mm into y."""
    if not path:
        return None
    mods = {}
    for name in ("wgmma", "configs"):
        spec = importlib.util.spec_from_file_location(f"s8_{name}", Path(path) / f"{name}.py")
        mods[name] = sys.modules[f"s8_{name}"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    W, table = mods["wgmma"], mods["configs"].CONFIGS.get("wgmma:sm_90", {})
    lib = W.lib()

    def cfg(m: int, n: int, k: int):
        rows = table.get((n, k))
        if not rows:
            return W.default_cfg(m, n, k, lib.sms)
        return W.Cfg.parse(next((t for mm, t in rows if m <= mm), rows[-1][1]))

    return {"identity": lib.identity(), "cfg": cfg, "fn": W.scaled_mm}


def libs_of(h1u: list[str], ref: str | None, sm: bool) -> dict:
    libs: dict = {}
    for spec in h1u:
        name, _, path = spec.partition("=")
        libs[name] = UN.H1u(path, sm=sm and UN.H1u.has_sm(path))
    if ref:
        libs["ref"] = UN.Ref(ref)
    return libs


def tiles(z: torch.Tensor, n: int, picks: list[int]) -> np.ndarray:
    ncb = n // S.T
    return np.stack([z[S.T * (u // ncb):S.T * (u // ncb) + S.T, S.T * (u % ncb):S.T * (u % ncb) + S.T].cpu().numpy()
                     for u in picks]).view(np.uint32)


def bench_call(key: str, c: dict, arrays: dict, libs: dict, plain, args, scratch: Path, wg: dict | None = None) -> dict:
    lin, m, n = c["linear"], c["m"], c["n"]
    a = torch.from_numpy(np.ascontiguousarray(S.whole("a", lin, m, scratch, args.jobs))).cuda()
    b = torch.from_numpy(np.ascontiguousarray(S.whole("b", lin, m, scratch, args.jobs))).cuda()
    sx = torch.from_numpy(S.activation_scales(lin, m)).cuda()
    sw = torch.from_numpy(S.weight_scales(lin)).cuda()
    salt, roots = S.SALT, (bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]))
    picks = [int(u) for u in arrays[f"{key}/units"]] if f"{key}/units" in arrays else []
    bt = b.t()
    arms: dict = {"plain": partial(plain, a, sx, b, sw), "int_mm": partial(torch._int_mm, a, bt)}
    yw = wcfg = None
    if wg is not None:
        yw = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        wcfg = wg["cfg"](m, n, c["k"])
        arms["wgmma"] = partial(wg["fn"], a, sx, b, sw, yw, wcfg)
    calls: dict = {}
    for nm, lib in libs.items():
        if isinstance(lib, UN.Ref):
            calls[nm] = UN.RefCall(lib, a, b, salt, roots, c["base"], z=False)
            arms[nm] = partial(calls[nm].run, z=False)
        else:
            calls[nm] = UN.H1uCall(lib, a, b, salt, roots, c["base"])
            arms[nm] = calls[nm].run
            for bl in BASELINES:
                if bl in arms:
                    arms[f"{bl}+{nm}"] = partial(_serial, arms[bl], calls[nm].run)
    for fn in arms.values():
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ms: dict = {k: [] for k in arms}
    order = list(arms)
    for rep in range(args.reps):
        for k in order[rep % len(order):] + order[:rep % len(order)]:
            if k == "ref" and rep >= args.ref_reps:
                continue
            e0.record()
            arms[k]()
            e1.record()
            e1.synchronize()
            ms[k].append(e0.elapsed_time(e1))
    exact: dict = {}
    gates: dict = {}
    if yw is not None:
        yw.view(torch.int16).fill_(-23131)
        arms["wgmma"]()
        exact["wgmma_vs_plain"] = bool(torch.equal(arms["plain"]().view(torch.int16), yw.view(torch.int16)))
    if picks:
        want = arrays[f"{key}/receipts"]
        want_sc, want_sig = arrays[f"{key}/seed_c"].tobytes(), [r.tobytes() for r in arrays[f"{key}/sigma"]]
        pk = torch.tensor(picks, device="cuda")
        exact["plain_s32"] = int((tiles(torch._int_mm(a, bt), n, picks) != arrays[f"{key}/z"]).sum()) == 0
        for nm, cl in calls.items():
            g = UN.twice(cl, arms[nm])
            got = cl.out[pk].cpu().numpy().view(np.uint32)
            if isinstance(cl, UN.RefCall):
                sc = [UN.digest(cl.seedc.cpu().numpy())] * len(picks)
                sig = [UN.digest(r) for r in cl.sigma[pk].cpu().numpy()]
            else:
                sd = cl.seeds[pk].cpu().numpy()
                sc, sig = [UN.digest(r[:8]) for r in sd], [UN.digest(r[8:]) for r in sd]
            g.update(bad_words=int((got != want).sum()), seeds_equal=sc == [want_sc] * len(picks) and sig == want_sig)
            gates[nm] = g
            exact[nm] = g["bad_words"] == 0 and g["unwritten"] == 0 and g["unstable"] == 0 and g["seeds_equal"]
    med = {k: statistics.median(v) for k, v in ms.items() if v}
    units_of = [nm for nm in libs if nm in med]
    rec = {"call": key, "linear": lin, "m": m, "n": n, "k": c["k"], "units": c["units"],
           "median_ms": {k: round(v, 4) for k, v in med.items()}, "ms": {k: [round(x, 4) for x in v] for k, v in ms.items()},
           "overhead": {nm: round(med[nm] / med["plain"], 4) for nm in units_of},
           "gamma": {nm: round((med["plain"] + med[nm]) / med["plain"], 4) for nm in units_of},
           "gamma_serial": {nm: round(med[f"plain+{nm}"] / med["plain"], 4) for nm in units_of if f"plain+{nm}" in med},
           "exact": exact, "gates": gates}
    if "wgmma" in med:
        w = med["wgmma"]
        rec["wgmma_cfg"] = str(wcfg)
        rec["vs_wgmma"] = {"overhead": {nm: round(med[nm] / w, 4) for nm in units_of},
                           "gamma": {nm: round((w + med[nm]) / w, 4) for nm in units_of},
                           "gamma_serial": {nm: round(med[f"wgmma+{nm}"] / w, 4) for nm in units_of
                                            if f"wgmma+{nm}" in med}}
    return rec


def _serial(*fns) -> None:
    for fn in fns:
        fn()


def totals(calls: list[dict], ms_: list[int], arms: list[str]) -> dict:
    out: dict = {}
    for m in ms_:
        rows = {r["linear"]: r for r in calls if r["m"] == m and "median_ms" in r}
        if not all(x in rows for x in S.LAYER):
            continue
        t: dict = {}
        for arm in arms:
            if not all(arm in rows[x]["median_ms"] for x in S.LAYER):
                continue
            layer = sum(rows[x]["median_ms"][arm] for x in S.LAYER)
            head = rows["head"]["median_ms"].get(arm) if "head" in rows else None
            t[arm] = {"layer_ms": round(layer, 3),
                      "step_ms": round(S.LAYERS * layer + head, 2) if head is not None else None}
        for arm, v in t.items():
            if arm in (*BASELINES, "int_mm") or "+" in arm:
                continue
            v.update(_vs(v, t["plain"], t.get(f"plain+{arm}")))
            if "wgmma" in t:
                v["vs_wgmma"] = _vs(v, t["wgmma"], t.get(f"wgmma+{arm}"))
        out[str(m)] = t
    return out


def _vs(v: dict, pl: dict, ser: dict | None) -> dict:
    """A units arm's totals `v` against a plain arm's `pl` (and `ser`, the two one after the other): overhead and gamma
    per layer and per step."""
    r = {"layer_overhead": round(v["layer_ms"] / pl["layer_ms"], 4),
         "layer_gamma": round((pl["layer_ms"] + v["layer_ms"]) / pl["layer_ms"], 4)}
    if v["step_ms"] is not None and pl["step_ms"] is not None:
        r["step_overhead"] = round(v["step_ms"] / pl["step_ms"], 4)
        r["step_gamma"] = round((pl["step_ms"] + v["step_ms"]) / pl["step_ms"], 4)
    if ser is not None:
        r["layer_gamma_serial"] = round(ser["layer_ms"] / pl["layer_ms"], 4)
        if ser["step_ms"] is not None and pl["step_ms"] is not None:
            r["step_gamma_serial"] = round(ser["step_ms"] / pl["step_ms"], 4)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--h1u", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--ref")
    ap.add_argument("--sm", action="store_true")
    ap.add_argument("--wgmma", metavar="DIR")
    ap.add_argument("--check")
    ap.add_argument("--ms", default=",".join(map(str, S.MS)))
    ap.add_argument("--linears", default=",".join(S.ORDER))
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--ref-reps", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()
    out, scratch = Path(args.out), Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    doc_f, arrays = F.load(Path(args.fixtures))
    libs = libs_of(args.h1u, args.ref, args.sm)
    pname, plain = plain_of()
    wg, wg_error = None, None
    try:
        wg = wgmma_of(args.wgmma)
    except Exception as e:  # noqa: BLE001
        wg_error = f"{type(e).__name__}: {e}"
    ms_ = [int(x) for x in args.ms.split(",")]
    lins = args.linears.split(",")
    doc: dict = {"schema": "j10r-sm90/bench/v1", "device": torch.cuda.get_device_name(0),
                 "sms": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__,
                 "plain": pname, "libs": {n: lib.path for n, lib in libs.items()},
                 "served_launch": {n: bool(getattr(lib, "sm", False)) for n, lib in libs.items()},
                 "wgmma": wg["identity"] if wg else wg_error,
                 "reps": args.reps, "ref_reps": args.ref_reps, "layers": S.LAYERS,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "calls": []}
    if args.check and Path(args.check).exists():
        ck = json.loads(Path(args.check).read_text())
        doc["check"] = {"verdict": ck.get("verdict"), "ok": ck.get("ok")}
    for m in ms_:
        for lin in lins:
            key = f"{lin}-{m}"
            c = doc_f["calls"].get(key) or S.call(lin, m)
            try:
                r = bench_call(key, c, arrays, libs, plain, args, scratch, wg)
            except Exception as e:  # noqa: BLE001
                import traceback
                r = {"call": key, "error": "".join(traceback.format_exception(e))[-3000:]}
            torch.cuda.empty_cache()
            print(json.dumps({k: v for k, v in r.items() if k != "ms"}), flush=True)
            doc["calls"].append(r)
            (out / "bench.json").write_text(json.dumps(doc, indent=1))
    arms = sorted({k for r in doc["calls"] for k in r.get("median_ms", {})})
    doc["totals"] = totals(doc["calls"], ms_, arms)
    doc["exact"] = all(all(r.get("exact", {}).values()) and "error" not in r for r in doc["calls"])
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "bench.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"totals": doc["totals"], "exact": doc["exact"]}, indent=1), flush=True)
    return 0 if doc["exact"] else 4


if __name__ == "__main__":
    sys.exit(main())
