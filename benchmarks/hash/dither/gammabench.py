"""Dither's units beside the exact int8 GEMM at GLM-5.3's int8 linears on one GPU, gated, and γ per shape.

    python gammabench.py --lib kr=h1u_kr.so [--lib NAME=h1u_krX.so ...] --out OUT [--layouts tp8,ep] [--ms 16,64,2048]
                         [--shapes NxK,...] [--reps 7] [--item-ms 4] [--picks 6] [--ref-units 2]

The shapes are one rank's int8 linears of GLM53_INT8 (recipe verity-int8-channel127/v3: every FP8 linear int8 W8A8 but
MLA's kv_b_proj and the DSA indexer's wk, which v3 keeps BF16, and the LM head int8 too), from
benchmarks/pouw/pearl_c_sm90/glm53.py's `linears` under TP8 (`tp8`) and with the routed experts expert-parallel (`ep`:
32 whole experts a rank), one row per distinct (N, K), each in H1's domain (K >= 384, 32 | K and 32 | N, and h1u.cu's
512 | K) or refused with the reason. A call of m rows has T = m rounded up to 32 slots (rows past m zero codes under
scale 0, as the served window's), every row block live. For the routed experts m is the GEMM's rows, not the batch: an
expert sees about m K / E of a batch's rows (`glm53.expert_rows`).

Per (N, K) and m, on the device, every arm gated before any is timed (a failed gate exits 4 and its arms are not timed):
  * the exact int8 GEMM (`int8_fused.scaled_mm`, y only, the exact-fused baseline) and Dither's served product
    (`int8_fused.scaled_mm_zy`: y and the call's z/y records over its T slots, `int8_exact.zy_records`) at every
    candidate config (`pick`'s, vLLM's tiles, and every config the architecture's table picks at this m for some shape),
    each under 0xA5: y bit for bit `int8_exact.dequantize(int_mm)` (cuBLASLt) and the records byte for byte
    `int8_exact.scaled_mm_zy`, on every element; that reference's records of the sampled units recomputed on the CPU
    (numpy int64 z over the whole K, y = dequant(z) by torch's CPU float32); nothing written refused; one bit of A's
    slot 0 flipped moves records of row block 0 alone;
  * each h1u build's units (receipts-only, `-hbc`: H1U_KSPLIT 4, H1K_AUTO, H1K_RONLY), unitsbench.py's gates: the
    sampled units' receipts against served_h1/receipts.py run on the CPU, seeds against seeds.py, no poison left,
    another build's every word equal to the first's, the no-write control and the noise fault refused; then
    `kh_units_multi` (#3290's one launch) over the call, every word equal to the two launches', its control and seed
    fault refused; and with --ref-units, units 0 and the last against `h1tile.reference` (the definition, on words: the
    receipt, and its z against the records' z).
Timing (CUDA graphs, every arm of a point interleaved rep by rep, the weights cycling through copies whose bytes exceed
2.5x the L2, as a served step's layers do): the configs first (3 reps each), the fastest gated config of each kernel
kept; then `exact`, `zy`, `seeds` (kh_seeds), `units:NAME` (kh_units_multi), `units2:NAME` (kh_one_noise and
kh_one_units, main's two launches), `dither:NAME` (zy, seeds and units on one stream) and `dither_ov:NAME` (the seeds
and units on a side stream forked beside the zy GEMM and joined), and `exact_pick` / `zy_pick` where `pick`'s config
is not the fastest.  γ = dither / exact (and dither_ov / exact, and dither / zy, the served product's), per build.
Writes OUT/gamma.json (after every point) and OUT/gamma.txt.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (HERE, ROOT / "benchmarks" / "pouw" / "served_h1", ROOT / "benchmarks" / "pouw" / "pearl_c_sm90",
          ROOT / "integrations" / "vllm", ROOT):
    sys.path.insert(0, str(p))
import glm53 as G
import receipts as RC
from hb_kbench import Clock, capture, inputs, one_call_ms
from unitsbench import Call, Lib, Multi, gate as units_gate, multi_gate

from verity_vllm.engine import int8_exact as IE
from verity_vllm.engine import int8_fused as IF

#: GLM-5.3's vocab_size (integrations/vllm/tests/properties/glm53_bf16_gemm_capture_gpu.py): the LM head's rows, V / TP a rank
VOCAB = 154880
#: the linears GLM53_INT8 (v3) keeps BF16
BF16_IN_V3 = {"self_attn.kv_b_proj": "BF16 in v3 (vLLM absorbs it into the attention as W_UK_T and W_UV)",
              "self_attn.indexer.wk_weights_proj": "BF16 in v3 (the indexer's wk, fused with the BF16 weights_proj)"}
H1_MIN_K = 384
UNIT = 32
LAYOUTS = ("tp8", "ep")
MS = (16, 64, 2048)
POISON16 = -23131  # 0xA5A5 as s16


def refusal(n: int, k: int) -> str | None:
    if k < H1_MIN_K:
        return f"K {k} < {H1_MIN_K}, H1's floor"
    if k % UNIT or n % UNIT:
        return f"the tile rule: 32 | K and 32 | N ({n} x {k})"
    if k % 512:
        return f"h1u.cu's units need 512 | K (K {k})"
    return None


def shapes(layouts: list[str]) -> list[dict]:
    """GLM53_INT8's int8 GEMM shapes, one rank's: (N, K) rows with the linears that have them."""
    rows: dict[tuple[int, int], dict] = {}

    def add(tag: str, n: int, k: int) -> None:
        d = rows.setdefault((n, k), {"n": n, "k": k, "linears": [], "refused": refusal(n, k)})
        if tag not in d["linears"]:
            d["linears"].append(tag)
    for lay in layouts:
        for lin in G.linears(ep=lay == "ep"):
            if lin.name in BF16_IN_V3 or (lay == "ep" and not lin.name.startswith("mlp.experts.")):
                continue
            add(f"{lay}:{lin.name}", lin.n, lin.k)
        if lay == "tp8":
            add("tp8:lm_head", VOCAB // G.TP, G.CONFIG["H"])
    return list(rows.values())


def candidates(m: int, n: int, k: int, sm: str) -> list[IF.Cfg]:
    """`pick`'s config, vLLM's tiles, and every config the architecture's table picks at m for some shape."""
    out = [IF.pick(m, n, k, sm), IF.default_cfg(m, n, k)]
    for rows in IF.CONFIGS.get(sm, {}).values():
        c = next((IF.Cfg.parse(t) for m_max, t in rows if m <= m_max), None)
        if c is not None:
            out.append(c)
    seen, uniq = set(), []
    for c in out:
        if str(c) not in seen:
            seen.add(str(c))
            uniq.append(c)
    return uniq


def cpu_records(slot: np.ndarray, sa: np.ndarray, w: np.ndarray, sw: np.ndarray, units: list[int], n: int) -> np.ndarray:
    """The records of `units` recomputed on the CPU: z over the whole K in int64 (then int32), y = dequant(z) in torch's CPU
    float32 (the activation scale first) and bf16, `int8_exact.zy_records`' layout."""
    out = []
    for u in units:
        ri, cj = divmod(u, n // UNIT)
        a = slot[UNIT * ri:UNIT * ri + UNIT].astype(np.int64)
        b = w[UNIT * cj:UNIT * cj + UNIT].astype(np.int64)
        z = torch.from_numpy((a @ b.T).astype(np.int32))
        y = IE.dequantize(z, torch.from_numpy(sa[UNIT * ri:UNIT * ri + UNIT]), torch.from_numpy(sw[UNIT * cj:UNIT * cj + UNIT]))
        out.append(IE.zy_records(z, y)[0].numpy())
    return np.stack(out)


class Gemm:
    """One (N, K, m) call's exact GEMM operands: the A slot [T, K] (its first m rows the call's codes), scales, the weight
    copies, y and the records."""

    def __init__(self, n: int, k: int, m: int, T: int, copies: int, seed: int):
        g = np.random.default_rng(seed)
        self.n, self.k, self.m, self.T = n, k, m, T
        self.slot_h = np.zeros((T, k), dtype=np.int8)
        self.slot_h[:m] = inputs(seed + 1, m, k)
        self.sa_h = np.zeros(T, dtype=np.float32)
        self.sa_h[:m] = g.uniform(1e-3, 1e-2, m).astype(np.float32)
        self.w_h = inputs(seed + 2, n, k)
        self.sw_h = g.uniform(1e-4, 1e-3, n).astype(np.float32)
        self.slot = torch.from_numpy(self.slot_h).cuda()
        self.xq = self.slot[:m]
        self.sa = torch.from_numpy(self.sa_h[:m].copy()).cuda()
        self.sw = torch.from_numpy(self.sw_h).cuda()
        w0 = torch.from_numpy(self.w_h).cuda()
        self.ws = [w0] + [torch.roll(w0, shifts=j, dims=0).contiguous() for j in range(1, copies)]
        self.y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        self.zy = torch.empty((IE.zy_units(m, n, T), IE.ZY_BYTES), dtype=torch.uint8, device="cuda")
        self.work = IF.Workspace(torch.device("cuda"))

    def exact(self, cfg: IF.Cfg, i: int = 0, y: torch.Tensor | None = None) -> None:
        IF.scaled_mm(self.xq, self.sa, self.ws[i % len(self.ws)], self.sw, self.y if y is None else y, cfg, self.work)

    def zyrun(self, cfg: IF.Cfg, i: int = 0) -> None:
        IF.scaled_mm_zy(self.xq, self.sa, self.ws[i % len(self.ws)], self.sw, self.y, self.zy, self.T, cfg, self.work)


def gemm_gates(gm: Gemm, cands: list[IF.Cfg], picks: list[int]) -> dict:
    y_ref = IE.dequantize(IE.int_mm(gm.xq, gm.ws[0]), gm.sa, gm.sw)
    _, zy_ref = IE.scaled_mm_zy(gm.xq, gm.sa, gm.ws[0], gm.sw, gm.T)
    torch.cuda.synchronize()
    cpu = cpu_records(gm.slot_h, gm.sa_h, gm.w_h, gm.sw_h, picks, gm.n)
    cpu_bad = int((zy_ref[torch.tensor(picks, device="cuda")].cpu().numpy() != cpu).sum())
    yv, rv = y_ref.view(torch.int16), zy_ref
    out: dict = {"cpu_units": picks, "cpu_bad_bytes": cpu_bad, "exact": {}, "zy": {}}
    for c in cands:
        for kind in ("exact", "zy"):
            try:
                gm.y.view(torch.int16).fill_(POISON16)
                gm.zy.fill_(0xA5)
                gm.exact(c) if kind == "exact" else gm.zyrun(c)
                torch.cuda.synchronize()
                g = {"y_bad": int((gm.y.view(torch.int16) != yv).sum())}
                if kind == "zy":
                    g["zy_bad"] = int((gm.zy != rv).sum())
                g["ok"] = cpu_bad == 0 and all(v == 0 for v in g.values())
            except Exception as e:  # noqa: BLE001 - a config the kernel refuses is a result
                g = {"ok": False, "refused": "".join(traceback.format_exception_only(type(e), e)).strip()[-600:]}
            out[kind][str(c)] = g
    pick = cands[0]
    gm.y.view(torch.int16).fill_(POISON16)
    gm.zy.fill_(0xA5)
    torch.cuda.synchronize()
    k_bad = int((gm.zy != rv).sum()) + int((gm.y.view(torch.int16) != yv).sum())
    gm.slot[0, 0] ^= 1
    gm.zyrun(pick)
    torch.cuda.synchronize()
    gm.slot[0, 0] ^= 1
    moved = (gm.zy != rv).any(dim=1).nonzero().flatten().tolist()
    nb = gm.n // UNIT
    out["control_refused"] = k_bad > 0
    out["fault_moved_units"] = [moved[0], moved[-1], len(moved)] if moved else []
    out["fault_conforming"] = bool(moved) and max(moved) < nb
    out["ok"] = cpu_bad == 0 and out["control_refused"] and out["fault_conforming"]
    return out


def ref_gate(c: Call, out: torch.Tensor, zy: torch.Tensor, a_cpu: torch.Tensor, w_cpu: torch.Tensor, units: list[int]) -> dict:
    """`h1tile.reference` on `units`: the receipt against the build's, z against the records' z."""
    bad_r = bad_z = 0
    t0 = time.time()
    for u in units:
        z, rc = RC.reference(RC.Call(a_cpu, w_cpu), c.salt, c.base, *c.roots, u)
        bad_r += int((out[u].cpu().numpy().astype(np.uint32) != np.array(rc, dtype=np.uint32)).sum())
        zr = zy[u, :4 * UNIT * UNIT].cpu().view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF
        bad_z += int((zr != (np.array(z, dtype=np.int64) & 0xFFFFFFFF)).sum())
    return {"units": units, "receipt_bad_words": bad_r, "z_bad_words": bad_z, "s": round(time.time() - t0, 1),
            "ok": bad_r == 0 and bad_z == 0}


def time_arms(arms: dict, reps: int, item_ms: float, clock: Clock) -> dict:
    graphs, per = {}, {}
    for key, fn in arms.items():
        t = one_call_ms(fn)
        G_ = max(1, min(400, round(item_ms / max(t, 1e-3))))
        graphs[key], per[key] = capture(fn, G_), G_
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    us = {k: [] for k in graphs}
    mhz = {k: [] for k in graphs}
    order = list(graphs)
    for rep in range(reps):
        for key in order[rep % len(order):] + order[:rep % len(order)]:
            e0.record()
            graphs[key].replay()
            e1.record()
            e1.synchronize()
            us[key].append(e0.elapsed_time(e1) * 1e3 / per[key])
            mhz[key].append(clock.sm())
    del graphs
    return {k: {"median_us": round(statistics.median(v), 3), "us": [round(x, 3) for x in v], "per_graph": per[k],
                "sm_mhz": mhz[k]} for k, v in us.items()}


def point(libs: dict, names: list[str], n: int, k: int, m: int, args, rng: random.Random, clock: Clock, l2: int,
          sms: int, sm: str, side: torch.cuda.Stream) -> tuple[dict, bool]:
    T = (m + UNIT - 1) // UNIT * UNIT
    copies = max(2, min(32, -(-int(2.5 * l2) // (n * k))))
    gm = Gemm(n, k, m, T, copies, seed=(n * 131 + k) * 7 + m)
    d: dict = {"n": n, "k": k, "m": m, "t": T, "weight_copies": copies, "units": (T // UNIT) * (n // UNIT)}
    L = d["units"]
    picks = sorted({0, L - 1, *rng.sample(range(L), min(L, args.picks))})
    d["picks"] = picks
    cands = candidates(m, n, k, sm)
    d["pick_cfg"] = str(cands[0])
    t0 = time.time()
    gg = d["gemm_gate"] = gemm_gates(gm, cands, picks)
    d["gemm_gate_s"] = round(time.time() - t0, 1)
    ok = gg["ok"]
    good = {kind: [c for c in cands if gg[kind][str(c)]["ok"]] for kind in ("exact", "zy")}
    print(json.dumps({"n": n, "k": k, "m": m, "gemm_gate": {kk: v for kk, v in gg.items() if kk not in ("exact", "zy")},
                      "ok_cfgs": {kind: len(v) for kind, v in good.items()}}), flush=True)
    ok = ok and bool(good["exact"]) and bool(good["zy"]) and str(cands[0]) in map(str, good["exact"]) \
        and str(cands[0]) in map(str, good["zy"])

    a_cpu, w_cpu = torch.from_numpy(gm.slot_h), torch.from_numpy(gm.w_h)
    salt, roots, base = rng.randbytes(64), (rng.randbytes(64), rng.randbytes(64)), rng.getrandbits(40)
    first = Call(libs[names[0]], gm.slot, gm.ws[0], m, salt, roots, base, sms)
    cs = {nm: first if i == 0 else first.twin(libs[nm], sms) for i, nm in enumerate(names)}
    t0 = time.time()
    ref = RC.call_receipts(RC.Call(a_cpu, w_cpu), salt, base, *roots, only=picks)
    d["receipts_cpu_s"] = round(time.time() - t0, 2)
    ref_t = torch.from_numpy(ref.view(np.int32)).to("cuda")
    d["unit_gates"], d["multi_gates"] = {}, {}
    base_out, units_ok = None, []
    for nm, c in cs.items():
        g = units_gate(c, picks, ref_t, base_out)
        if base_out is None:
            base_out = c.full()
        mg = multi_gate(libs[nm], [c], [base_out]) if libs[nm].multi else {"ok": False, "refused": "no kh_units_multi"}
        d["unit_gates"][nm], d["multi_gates"][nm] = g, mg
        print(json.dumps({"n": n, "k": k, "m": m, "lib": nm, "gate": g["ok"], "multi_gate": mg["ok"]}), flush=True)
        if g["ok"] and mg["ok"]:
            units_ok.append(nm)
        else:
            ok = False
    if args.ref_units and base_out is not None:
        us = [0, L - 1][:args.ref_units]
        gm.zyrun(cands[0])
        torch.cuda.synchronize()
        rg = d["ref_gate"] = ref_gate(first, base_out, gm.zy, a_cpu, w_cpu, us)
        print(json.dumps({"n": n, "k": k, "m": m, "ref_gate": rg}), flush=True)
        if not rg["ok"]:
            ok = False
            units_ok = []
    if not (good["exact"] and good["zy"] and units_ok):
        d["timing"] = None
        return d, ok

    pre: dict = {}
    for kind in ("exact", "zy"):
        for c in good[kind]:
            pre[(kind, str(c))] = (lambda c, kind: lambda i: gm.exact(c, i) if kind == "exact" else gm.zyrun(c, i))(c, kind)
    pt = time_arms(pre, 3, args.item_ms, clock)
    d["cfg_us"] = {f"{kind}/{c}": v["median_us"] for (kind, c), v in pt.items()}
    best = {kind: min(good[kind], key=lambda c: pt[(kind, str(c))]["median_us"]) for kind in ("exact", "zy")}
    d["best_cfg"] = {kind: str(c) for kind, c in best.items()}
    zc = best["zy"]
    arms: dict = {"exact": lambda i: gm.exact(best["exact"], i), "zy": lambda i: gm.zyrun(zc, i)}
    if str(best["exact"]) != str(cands[0]):
        arms["exact_pick"] = lambda i: gm.exact(cands[0], i)
    if str(zc) != str(cands[0]):
        arms["zy_pick"] = lambda i: gm.zyrun(cands[0], i)
    arms["seeds"] = lambda i: first.seed()
    keep = []
    for nm in units_ok:
        c = cs[nm]
        o = torch.empty_like(c.out)
        mus = [Multi(libs[nm], [(c, w, o)]) for w in gm.ws]
        keep.append((mus, o))
        arms[f"units:{nm}"] = (lambda mus: lambda i: mus[i % len(mus)].run())(mus)
        arms[f"units2:{nm}"] = (lambda c: lambda i: (c.noise(), c.run_units(b=gm.ws[i % len(gm.ws)])))(c)

        def dither(i, c=c, mus=mus):
            gm.zyrun(zc, i)
            c.seed()
            mus[i % len(mus)].run()

        def dither_ov(i, c=c, mus=mus):
            cur = torch.cuda.current_stream()
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                c.seed()
                mus[i % len(mus)].run()
            gm.zyrun(zc, i)
            cur.wait_stream(side)
        arms[f"dither:{nm}"] = dither
        arms[f"dither_ov:{nm}"] = dither_ov
    tm = d["timing"] = time_arms(arms, args.reps, args.item_ms, clock)
    med = {k_: v["median_us"] for k_, v in tm.items()}
    d["median_us"] = med
    d["gamma"] = {nm: {"dither_over_exact": round(med[f"dither:{nm}"] / med["exact"], 4),
                       "dither_ov_over_exact": round(med[f"dither_ov:{nm}"] / med["exact"], 4),
                       "dither_over_zy": round(med[f"dither:{nm}"] / med["zy"], 4),
                       "units_over_exact": round((med["seeds"] + med[f"units:{nm}"]) / med["exact"], 4)}
                  for nm in units_ok}
    del keep
    return d, ok


def table(doc: dict) -> str:
    lines = ["N x K | linears | m | exact us | zy us | seeds+units us (build) | dither us | dither_ov us | "
             "γ = dither/exact | γ_ov | dither/zy"]
    for s in doc["shapes"]:
        if s["refused"]:
            lines.append(f"{s['n']} x {s['k']} | {', '.join(s['linears'])} | - | refused: {s['refused']}")
            continue
        for p in s["points"]:
            if not p.get("median_us"):
                lines.append(f"{s['n']} x {s['k']} | {', '.join(s['linears'])} | {p['m']} | not timed (gate)")
                continue
            med = p["median_us"]
            for nm, g in p["gamma"].items():
                lines.append(f"{s['n']} x {s['k']} | {', '.join(s['linears'])} | {p['m']} | {med['exact']:.1f} | "
                             f"{med['zy']:.1f} | {med['seeds'] + med[f'units:{nm}']:.1f} ({nm}) | "
                             f"{med[f'dither:{nm}']:.1f} | {med[f'dither_ov:{nm}']:.1f} | {g['dither_over_exact']:.3f} | "
                             f"{g['dither_ov_over_exact']:.3f} | {g['dither_over_zy']:.3f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layouts", default=",".join(LAYOUTS))
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--shapes", default="", help="NxK,... (default: every int8 shape of --layouts)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--item-ms", type=float, default=4.0)
    ap.add_argument("--picks", type=int, default=6)
    ap.add_argument("--ref-units", type=int, default=2, help="units checked against h1tile.reference (0, 1 or 2)")
    ap.add_argument("--list", action="store_true", help="print the shapes and exit (no GPU)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layouts = args.layouts.split(",")
    rows = shapes(layouts)
    if args.shapes:
        want = {tuple(map(int, s.split("x"))) for s in args.shapes.split(",")}
        rows = [r for r in rows if (r["n"], r["k"]) in want]
    ms = [int(x) for x in args.ms.split(",")]
    if args.list:
        print(json.dumps(rows, indent=1))
        return 0
    paths = dict(v.partition("=")[::2] for v in args.lib)
    libs = {nm: Lib(p) for nm, p in paths.items()}
    names = list(libs)
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    sms, l2, sm = props.multi_processor_count, int(getattr(props, "L2_cache_size", 50 << 20)), IF.arch()
    side = torch.cuda.Stream()
    rng = random.Random(0x6A33A)
    doc = {"schema": "dither/gamma/v1", "model": "GLM53_INT8 (verity-int8-channel127/v3)", "device": clock.name,
           "arch": sm, "sms": sms, "l2": l2, "libs": paths, "layouts": layouts, "ms": ms, "bf16_in_v3": BF16_IN_V3,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shapes": rows}
    path = out / "gamma.json"
    failed = []
    for r in rows:
        r["points"] = []
        if r["refused"]:
            continue
        for m in ms:
            try:
                p, ok = point(libs, names, r["n"], r["k"], m, args, rng, clock, l2, sms, sm, side)
            except Exception as e:  # noqa: BLE001 - recorded; the point counts as failed
                p, ok = {"m": m, "error": "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]}, False
                print(p["error"], file=sys.stderr, flush=True)
            r["points"].append(p)
            if not ok:
                failed.append(f"{r['n']}x{r['k']}@{m}")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            doc["gate_failed"] = failed
            path.write_text(json.dumps(doc, indent=1))
            if p.get("median_us"):
                print(json.dumps({"n": r["n"], "k": r["k"], "m": m, "median_us": p["median_us"], "gamma": p["gamma"],
                                  "best_cfg": p["best_cfg"]}), flush=True)
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(doc, indent=1))
    (out / "gamma.txt").write_text(table(doc))
    print(table(doc), flush=True)
    return 4 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
