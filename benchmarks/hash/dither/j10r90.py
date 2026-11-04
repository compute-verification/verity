"""J10-R's per-call-noise units (`PouwH1Tile_v5` / `_v6`'s receipt; h1u.cu's kh_seeds_j10r, kh_units_multi_j10r and
kh_units_multi_j10r_sm) at the frozen shape's linears (`LINEARS`: width 16384, one KV head, MLP 65536) and
M = 4,096 and 16,384 token rows, one call a linear, every row live:

    python j10r90.py check  --lib NAME=PATH ... --out OUT [--ms 4096,16384] [--linears q,kv,o,up,down,head] [--picks 24]
    python j10r90.py time   --lib NAME=PATH ... --out OUT [--ms ...] [--linears ...] [--reps 5] [--wgmma DIR]
    python j10r90.py verify --dump DIR --out OUT [--jobs 8] [--v6 1]

A and B are `operand`'s, a counter hash of (seed, row, column) that `verify` regenerates row by row; salt, roots and the
call's first position from random.Random(SEED). The first --lib is the base.

check (on the GPU): z = torch._int_mm(A, B^T) against the float64 product on every cell (exact: |z| <= 2^30); then each
build, under 0xA5 over its receipts and row workspace: seeds (kh_seeds_j10r) and units (kh_units_multi_j10r): no word
left unwritten (`unwritten`: a word still 0xA5A5A5A5 is run again under 0x5A5A5A5A), every word of every unit equal to
the base build's, the sampled units' receipts (the unit grid's corners, its middle, random, and every unit with a word
the units write as 0xA5A5A5A5, up to 8) equal to j10r_units.call_receipts' and their seeds to seeds.call_seed and
receipt_seed; the same call
by kh_units_multi_j10r_sm on every SM equal on every word; the no-write control (zero units) refused; one bit of the row
workspace moving exactly row tile 0's units, and one bit of unit 0's seed_c moving unit 0 alone (unitsbench.j10r_gate's
faults, through the rows and units launches alone; under H1U_JW, whose column pass alone reads the call's seed_c, one
bit of unit 0's sigma_R moving unit 0 alone and one bit of the call's seed_c moving every unit); the workspace the
build's own size (j10r_ws.py), no launch writing the sentinel tail past it, and under H1U_JW its last bytes written
(its W' fills it). The sampled units' receipts (every build) and z tiles go to
j10r90-check.npz, the rest to j10r90-check.json. Exit 4 if any gate fails.

verify (a CPU node, check's out/ as DIR): every sampled unit's receipt and z against `h1tile.j10r_reference` itself, from
its 32 rows of A and of B regenerated: v5 on every sampled unit, v6 (stream=True) on the first --v6 of each call; z on
PAIRS through the Definition and on every cell of the tile against the int64 product. Exit 4 on any mismatch.

time (on the GPU): plain INT8, s8_linear's `int_mm.linear` with its eager epilogue: torch._int_mm and the f32
epilogue as four eager kernels,
against each build's J10-R call (seeds, rows and units: kh_units_multi_j10r_sm on every SM, the served launch), arms
interleaved rep by rep; overhead = units / plain and gamma = (plain + units) / plain, a call and summed over a layer's
linears (q, kv, o, up, down; the head once a step). With --wgmma, s8_linear's wgmma.linear (the fastest exact plain INT8 on
the H200) at its tuned config is a second plain arm in the same rotation, its bf16 output equal to
plain's on every cell. After the reps every build's receipts are compared word for word with the base's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from verity_config.settings import of

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "served_h1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import j10r_units as J10R

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import seeds as SD

settings = of(__file__)
#: the frozen shape's linears (n, k), the H200 demo's MQA decoder (benchmarks/torch_gpu/j10r_sm90/shape.py's LINEARS:
#: width 16384, 256 query heads of 64, 1 KV head, MLP 65536, 256 tokens)
LINEARS = {"q": (16384, 16384), "kv": (128, 16384), "o": (16384, 16384), "up": (65536, 16384), "down": (16384, 65536),
           "head": (256, 16384)}
LAYER = ("q", "kv", "o", "up", "down")
MS = (4096, 16384)
POISON = -1515870811  # 0xA5A5A5A5 as s32
POISON2 = 0x5A5A5A5A
T = 32
SEED = 0x90A
M32 = 0xFFFFFFFF
#: the cells of a unit whose z verify takes through the Definition (every cell is checked against the int64 product)
PAIRS = [(0, 0), (0, 31), (31, 0), (31, 31), (7, 19), (19, 7), (12, 12), (25, 3), (3, 25), (16, 30), (30, 16), (1, 2),
         (8, 8), (15, 16), (16, 15), (23, 29)]


def operand(seed: int, rows: torch.Tensor, k: int) -> torch.Tensor:
    """s8 [len(rows), k] on rows' device: byte (row k + col + seed) of a 32-bit integer hash, as s8. Every product below
    stays under 2^63, so int64 gives the same bits on any device."""
    x = rows.to(torch.int64)[:, None] * k + torch.arange(k, dtype=torch.int64, device=rows.device)[None, :]
    x = ((x + seed) * 0x2C1B3C6D) & M32
    x ^= x >> 15
    x = (x * 0x297A2D39) & M32
    x ^= x >> 16
    return ((x & 0xFF) - 128).to(torch.int8)


def operand_all(seed: int, rows: int, k: int, device: str = "cuda") -> torch.Tensor:
    out = torch.empty((rows, k), dtype=torch.int8, device=device)
    step = max(T, (1 << 27) // k)
    for r0 in range(0, rows, step):
        r = torch.arange(r0, min(rows, r0 + step), device=device)
        out[r0:r0 + len(r)] = operand(seed, r, k)
    return out


def calls(ms: list[int], linears: list[str]) -> list[dict]:
    """Each (linear, M)'s call: operand seeds, salt, roots, its first position; the same for check, time and verify."""
    rng = random.Random(SEED)
    out = []
    for m in ms:
        for i, name in enumerate(linears):
            n, k = LINEARS[name]
            out.append({"linear": name, "m": m, "n": n, "k": k, "a_seed": 0x1A000 + 16 * i + m // 4096,
                        "b_seed": 0x2B000 + 16 * i, "salt": rng.randbytes(64).hex(), "root_a": rng.randbytes(64).hex(),
                        "root_b": rng.randbytes(64).hex(), "base": rng.getrandbits(40)})
    return out


def picks_of(c: dict, count: int) -> list[int]:
    nrb, ncb = c["m"] // T, c["n"] // T
    rng = random.Random(f"{SEED}:{c['linear']}:{c['m']}")
    pts = {(0, 0), (0, ncb - 1), (nrb - 1, 0), (nrb - 1, ncb - 1), (nrb // 2, ncb // 2), (1, min(1, ncb - 1))}
    while len(pts) < min(count, nrb * ncb):
        pts.add((rng.randrange(nrb), rng.randrange(ncb)))
    return sorted(ri * ncb + cj for ri, cj in pts)


def secrets(c: dict) -> tuple[bytes, bytes, bytes]:
    return bytes.fromhex(c["salt"]), bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])


def digest(row) -> bytes:
    return b"".join((int(v) & ((1 << 64) - 1)).to_bytes(8, "big") for v in row)


# -- on the GPU --------------------------------------------------------------------------------------------------------

class JCall:
    """One call's J10-R state on one build (unitsbench.MultiJ10R's entry): A [M, K] with every row tile live, B, the
    seed data and each unit's [seed_c, sigma_R]."""

    def __init__(self, lib, a: torch.Tensor, b: torch.Tensor, c: dict):
        self.lib, self.a, self.b, self.base = lib, a, b, c["base"]
        self.salt, ra, rb = secrets(c)
        self.roots = (ra, rb)
        self.units = (a.shape[0] // T) * (b.shape[0] // T)
        self.live = torch.ones(a.shape[0] // T, dtype=torch.uint8, device="cuda")
        w = np.array(J10R.seed_words(self.salt, ra, rb), dtype=np.uint64).view(np.int64)
        self.jdata = torch.tensor(w, device="cuda")
        self.jseeds = torch.empty((self.units, 16), dtype=torch.int64, device="cuda")

    def seed(self) -> None:
        if self.lib.h.kh_seeds_j10r(self.jdata.data_ptr(), self.base, self.base, self.units, self.jseeds.data_ptr(),
                                    torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("kh_seeds_j10r")


def libs_of(specs: list[str]) -> dict:
    import unitsbench as U
    libs = {}
    for v in specs:
        name, _, path = v.partition("=")
        lib = libs[name] = U.Lib(path)
        if not lib.j10r_sm:
            raise SystemExit(f"j10r90: {path} has no kh_units_multi_j10r_sm")
    return libs


def z_check(a: torch.Tensor, b: torch.Tensor, picks: list[int]) -> tuple[dict, np.ndarray]:
    """torch._int_mm's z against the float64 product on every cell, and the picked units' 32 x 32 tiles of it."""
    z = torch._int_mm(a, b.t())
    ad = a.to(torch.float64)
    bad, step = 0, max(T, (1 << 28) // a.shape[0])
    for n0 in range(0, b.shape[0], step):
        zf = ad @ b[n0:n0 + step].to(torch.float64).t()
        bad += int((z[:, n0:n0 + step].to(torch.float64) != zf).sum())
        del zf
    ncb = b.shape[0] // T
    tiles = np.stack([z[T * (u // ncb):T * (u // ncb) + T, T * (u % ncb):T * (u % ncb) + T].cpu().numpy() for u in picks])
    cells = z.numel()
    del z, ad
    return {"cells": cells, "bad_cells": bad}, tiles


def unwritten(o: torch.Tensor, run) -> tuple[int, int]:
    """(words of `o` a launch leaves unwritten, words it writes as POISON): `run` under POISON, then, only if some word
    holds it, under POISON2. About 2^-32 of the words a launch writes are POISON, so a fixed poison alone flags written
    words at these sizes; a written word is the same value in both runs, so only an unwritten one holds each run's
    poison. `o` ends as the last run's."""
    o.fill_(POISON)
    run()
    torch.cuda.synchronize()
    idx = (o == POISON).nonzero(as_tuple=True)
    if not idx[0].numel():
        return 0, 0
    o.fill_(POISON2)
    run()
    torch.cuda.synchronize()
    left = int((o[idx] == POISON2).sum())
    return left, int(idx[0].numel()) - left


def poison_units(lib, a: torch.Tensor, b: torch.Tensor, c: dict, most: int = 8) -> list[int]:
    """The first `most` units whose receipt, written by `lib`'s units launch, holds a POISON word: check samples them
    too, so the twin and verify's Definition check the words the poison count can't."""
    import unitsbench as U
    jc = JCall(lib, a, b, c)
    o = torch.full((jc.units, int(lib.h.kh_words(c["k"]))), POISON, dtype=torch.int32, device="cuda")
    jc.seed()
    U.MultiJ10R(lib, [(jc, b, o)]).run()
    torch.cuda.synchronize()
    units = (o == POISON).any(dim=1).nonzero().flatten()[:most].tolist()
    del o, jc
    return units


def gate(lib, jc: JCall, out: torch.Tensor, first: torch.Tensor | None, picks: list[int], ref_t: torch.Tensor) -> dict:
    import unitsbench as U
    mu = U.MultiJ10R(lib, [(jc, jc.b, out)])
    ncb = jc.b.shape[0] // T

    def poison(o: torch.Tensor, m) -> None:
        o.fill_(POISON)
        for w in m.rws:
            w.fill_(0xA5)

    def launch(m, seeded: bool = False):
        def go() -> None:
            for w in m.rws:
                w.fill_(0xA5)
            if seeded:
                jc.seed()
            m.run()
        return go

    pk = torch.tensor(picks, device="cuda")
    jc.jseeds.fill_(0x5A5A5A5A5A5A5A5)
    left, as_poison = unwritten(out, launch(mu, seeded=True))
    ws_end = mu.end_written()[0]
    r: dict = {"units": jc.units, "words": out.shape[1], "poison_left": left, "written_as_poison": as_poison,
               "bad_words": int((out[pk] != ref_t).sum()),
               "vs_base_bad_words": None if first is None else int((out != first).sum())}
    sd = jc.jseeds[pk].cpu().numpy()
    want_c = J10R.call_seed(jc.salt, jc.base, *jc.roots)
    r["seed_bad"] = sum(int(digest(sd[i, :8]) != want_c) + int(digest(sd[i, 8:]) != SD.receipt_seed(
        jc.salt, jc.base + u, *jc.roots)) for i, u in enumerate(picks))
    got = out[pk].cpu().numpy().view(np.uint32).copy()
    keep = out.clone()
    out2 = torch.empty_like(out)
    ms = U.MultiJ10R(lib, [(jc, jc.b, out2)], sm=True)
    r["sm_poison_left"], r["sm_written_as_poison"] = unwritten(out2, launch(ms))
    r["sm_vs_units_bad_words"] = int((out2 != keep).sum())
    r["sm_ws_tail_written"] = ms.tail_written()
    del out2, ms
    poison(out, mu)
    mu.run(units=0)
    torch.cuda.synchronize()
    r["control_bad_words"] = int((out != keep).sum())
    r["control_refused"] = r["control_bad_words"] > 0

    def moved() -> torch.Tensor:
        return (out != keep).any(dim=1).nonzero().flatten()

    poison(out, mu)
    mu.run(mode=1)
    m, k = jc.a.shape
    rw = int(lib.h.kh_j10r_row_fault_word(m, jc.b.shape[0], k, 0)) if hasattr(lib.h, "kh_j10r_row_fault_word") else 0
    mu.rws[0].view(torch.int32)[rw] ^= 1
    mu.run(mode=2)
    torch.cuda.synchronize()
    rm = moved()
    r["fault_rows_moved"], r["fault_rows_want"] = int(rm.numel()), ncb
    r["fault_rows_conforming"] = bool(rm.numel() == ncb and (rm == torch.arange(ncb, device="cuda")).all())
    def seed_fault(sw: int) -> torch.Tensor:
        poison(out, mu)
        mu.run(mode=1)
        s = jc.jseeds.view(torch.int32)
        s[0, sw] ^= 1
        mu.run(mode=2)
        torch.cuda.synchronize()
        s[0, sw] ^= 1
        return moved()

    sm_ = seed_fault(16 if lib.jw else 0).tolist()
    r["fault_seed"] = "sigma_R" if lib.jw else "seed_c"
    r["fault_seed_moved"] = sm_[:8]
    r["fault_seed_conforming"] = sm_ == [0]
    if lib.jw:
        cm = seed_fault(0)
        r["fault_call_seed_moved"], r["fault_call_seed_want"] = int(cm.numel()), jc.units
        r["fault_call_seed_conforming"] = cm.numel() == jc.units
    out.copy_(keep)
    keep = None
    r.update({"ws_bytes": int(mu.rws[0].numel()), "ws_end_written": ws_end, "ws_tail_written": mu.tail_written()})
    r["ok"] = (r["poison_left"] == 0 and r["bad_words"] == 0 and r["vs_base_bad_words"] in (None, 0)
               and r["seed_bad"] == 0 and r["sm_poison_left"] == 0 and r["sm_vs_units_bad_words"] == 0
               and r["control_refused"] and r["fault_rows_conforming"] and r["fault_seed_conforming"]
               and r.get("fault_call_seed_conforming", True)
               and r["ws_tail_written"] == 0 and r["sm_ws_tail_written"] == 0 and (not lib.jw or ws_end > 0))
    return r, got


def check(args) -> int:
    from hb_kbench import Clock
    libs = libs_of(args.lib)
    names = list(libs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    doc = {"schema": "j10r90/check/v1", "device": clock.name, "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "libs": dict(v.partition("=")[::2] for v in args.lib), "reference": "j10r_units.call_receipts (verify: "
           "h1tile.j10r_reference)", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "calls": []}
    arrays: dict = {}
    failed = False
    for c in calls(args.ms, args.linears):
        key = f"{c['linear']}.{c['m']}"
        t0 = time.time()
        a = operand_all(c["a_seed"], c["m"], c["k"])
        b = operand_all(c["b_seed"], c["n"], c["k"])
        picks = picks_of(c, args.picks)
        hits = poison_units(libs[names[0]], a, b, c)
        picks += [u for u in hits if u not in picks]
        zr, tiles = z_check(a, b, picks)
        arrays[f"{key}.z"] = tiles
        salt, ra, rb = secrets(c)
        t1 = time.time()
        ref = J10R.call_receipts(a, b, salt, c["base"], ra, rb, only=picks)
        ref_t = torch.from_numpy(ref.view(np.int32)).to("cuda")
        rec = {**c, "picks": picks, "poison_units": hits, "z": zr, "ref_s": round(time.time() - t1, 2), "gates": {}}
        first = None
        for nm in names:
            jc = JCall(libs[nm], a, b, c)
            o = torch.empty((jc.units, int(libs[nm].h.kh_words(c["k"]))), dtype=torch.int32, device="cuda")
            g, got = gate(libs[nm], jc, o, first, picks, ref_t)
            rec["gates"][nm] = g
            arrays[f"{key}.{nm}"] = got
            print(json.dumps({"call": key, "lib": nm, "gate": g}), flush=True)
            failed |= not g["ok"]
            if first is None:
                first = o
            else:
                del o
            del jc
        del first, a, b, ref_t
        torch.cuda.empty_cache()
        rec["ok"] = zr["bad_cells"] == 0 and all(g["ok"] for g in rec["gates"].values())
        rec["s"] = round(time.time() - t0, 1)
        failed |= not rec["ok"]
        print(json.dumps({"call": key, "z": zr, "ok": rec["ok"], "s": rec["s"]}), flush=True)
        doc["calls"].append(rec)
        (out / "j10r90-check.json").write_text(json.dumps(doc, indent=1))
    np.savez_compressed(out / "j10r90-check.npz", **arrays)
    doc["ok"] = not failed
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "j10r90-check.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"ok": doc["ok"], "calls": {f"{r['linear']}.{r['m']}": r["ok"] for r in doc["calls"]}}), flush=True)
    return 0 if doc["ok"] else 4


def wgmma_of(d: str):
    """The s8 wgmma kernel's wgmma.linear: its wgmma.py and configs.py from DIR beside build.sh's ship
    (s8_wgmma.cubin, sass.json, MANIFEST.sha256), and a (M, N, K) -> Cfg pick by its tuned table as its run.py's
    table_cfg reads it, else its default."""
    settings.setdefault("S8_WGMMA_CUBIN", str(Path(d) / "s8_wgmma.cubin"))
    sys.path.insert(0, d)
    import wgmma as W
    from configs import CONFIGS
    table = CONFIGS.get("wgmma:sm_90", {})

    def pick(m: int, n: int, k: int):
        rows = table.get((n, k))
        if not rows:
            return W.default_cfg(m, n, k, W.lib().sms)
        return W.Cfg.parse(next((t for m_max, t in rows if m <= m_max), rows[-1][1]))
    return W, pick


def time_(args) -> int:
    sys.path.append(str(Path(__file__).resolve().parents[2] / "torch_gpu" / "s8_linear"))
    import int_mm as IM
    import unitsbench as U
    from hb_kbench import Clock
    libs = libs_of(args.lib)
    names = list(libs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    W, wpick = wgmma_of(args.wgmma) if args.wgmma else (None, None)
    plains = ["plain"] + (["wgmma"] if W else [])
    doc = {"schema": "j10r90/time/v1", "device": clock.name, "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "libs": dict(v.partition("=")[::2] for v in args.lib), "plain": "int_mm.linear (eager)",
           **({"wgmma": "s8_linear's wgmma.scaled_mm at its tuned table's config", "wgmma_identity": W.lib().identity()}
              if W else {}),
           "units": "kh_seeds_j10r + kh_units_multi_j10r_sm at 1/1", "reps": args.reps,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "calls": []}
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for c in calls(args.ms, args.linears):
        a = operand_all(c["a_seed"], c["m"], c["k"])
        b = operand_all(c["b_seed"], c["n"], c["k"])
        g = torch.Generator(device="cuda").manual_seed(c["b_seed"])
        sx = (torch.rand(c["m"], device="cuda", generator=g) + 0.5) / 127
        sw = (torch.rand(c["n"], device="cuda", generator=g) + 0.5) * 0.02 / 127
        arms: dict = {"plain": lambda: IM.linear(a, sx, b, sw, "eager")}
        if W:
            cfg = wpick(c["m"], c["n"], c["k"])
            yw = torch.empty((c["m"], c["n"]), dtype=torch.bfloat16, device="cuda")
            arms["wgmma"] = lambda: W.scaled_mm(a, sx, b, sw, yw, cfg)
        keep: dict = {}
        for nm in names:
            jc = JCall(libs[nm], a, b, c)
            o = torch.empty((jc.units, int(libs[nm].h.kh_words(c["k"]))), dtype=torch.int32, device="cuda")
            mu = U.MultiJ10R(libs[nm], [(jc, b, o)], sm=True)
            keep[nm] = (jc, o, mu)
            arms[nm] = (lambda jc, mu: lambda: (jc.seed(), mu.run()))(jc, mu)
        for fn in arms.values():
            fn()
            fn()
        torch.cuda.synchronize()
        ms_: dict = {k: [] for k in arms}
        mhz: dict = {k: [] for k in arms}
        order = list(arms)
        for rep in range(args.reps):
            for k in order[rep % len(order):] + order[:rep % len(order)]:
                e0.record()
                arms[k]()
                e1.record()
                e1.synchronize()
                ms_[k].append(e0.elapsed_time(e1))
                mhz[k].append(clock.sm())
        left = {nm: unwritten(keep[nm][1], arms[nm])[0] for nm in names}
        o0 = keep[names[0]][1]
        same = {nm: int((keep[nm][1] != o0).sum()) for nm in names[1:]}
        wbad = None
        if W:
            yw.fill_(-1)
            arms["wgmma"]()
            wbad = int((yw.view(torch.int16) != arms["plain"]().view(torch.int16)).sum())
        med = {k: statistics.median(v) for k, v in ms_.items()}
        rec = {**{x: c[x] for x in ("linear", "m", "n", "k")}, "units": keep[names[0]][0].units,
               "median_ms": {k: round(v, 4) for k, v in med.items()},
               "ms": {k: [round(x, 4) for x in v] for k, v in ms_.items()}, "sm_mhz": mhz,
               **({"wgmma_cfg": str(cfg), "wgmma_vs_plain_bad_cells": wbad} if W else {}),
               **{("overhead" if p == "plain" else f"overhead_{p}"): {nm: round(med[nm] / med[p], 3) for nm in names}
                  for p in plains},
               **{("gamma" if p == "plain" else f"gamma_{p}"): {nm: round((med[p] + med[nm]) / med[p], 3) for nm in names}
                  for p in plains},
               "vs_base_bad_words": same, "poison_left": left}
        print(json.dumps({k: v for k, v in rec.items() if k not in ("ms", "sm_mhz")}), flush=True)
        doc["calls"].append(rec)
        keep = arms = a = b = None
        torch.cuda.empty_cache()
        (out / "j10r90-time.json").write_text(json.dumps(doc, indent=1))
    doc["layer"] = {}
    for m in args.ms:
        rows = {r["linear"]: r for r in doc["calls"] if r["m"] == m}
        if not all(x in rows for x in LAYER):
            continue
        lay = doc["layer"][str(m)] = {}
        for p in plains:
            base = sum(rows[x]["median_ms"][p] for x in LAYER)
            lay[f"{p}_ms"] = round(base, 3)
            for nm in names:
                u = sum(rows[x]["median_ms"][nm] for x in LAYER)
                lay.setdefault(nm, {"units_ms": round(u, 3)})
                lay[nm][f"overhead_vs_{p}"] = round(u / base, 3)
                lay[nm][f"gamma_vs_{p}"] = round((base + u) / base, 3)
    doc["ok"] = all(not any(r["vs_base_bad_words"].values()) and not any(r["poison_left"].values())
                    and not r.get("wgmma_vs_plain_bad_cells") for r in doc["calls"])
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "j10r90-time.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"layer": doc["layer"], "ok": doc["ok"]}, indent=1), flush=True)
    return 0 if doc["ok"] else 4


# -- off the GPU -------------------------------------------------------------------------------------------------------

def _verify_unit(job: tuple) -> dict:
    c, u, recs, tile, v6 = job
    ncb = c["n"] // T
    ri, cj = divmod(u, ncb)
    a = operand(c["a_seed"], torch.arange(T * ri, T * ri + T), c["k"])
    b = operand(c["b_seed"], torch.arange(T * cj, T * cj + T), c["k"])
    z64 = (a.to(torch.int64) @ b.to(torch.int64).t()).numpy()
    salt, ra, rb = secrets(c)
    blk = HT.j10r_blocks(c["m"], c["k"], (T, T), ri, cj)
    at_, bt_ = (a.to(torch.int64) & 0xFF).tolist(), (b.to(torch.int64) & 0xFF).tolist()
    r = {"linear": c["linear"], "m": c["m"], "unit": u, "z_tile_bad": int((tile.astype(np.int64) != z64).sum())}
    t0 = time.time()
    forms = [("v5", False)] + ([("v6", True)] if v6 else [])
    for form, stream in forms:
        z, rec = HT.j10r_reference(at_, bt_, salt, c["base"] + u, (ra, rb), c["base"], blk, pairs=PAIRS, stream=stream)
        want = np.array(rec, dtype=np.uint32)
        r[f"{form}_receipt_bad"] = {nm: int((g != want).sum()) for nm, g in recs.items()}
        r[f"{form}_z_bad"] = sum(int((int(zv) - int(tile[i, j])) % (1 << 32) != 0) for zv, (i, j) in zip(z, PAIRS))
        r["words"] = len(rec)
    r["s"] = round(time.time() - t0, 1)
    r["bad"] = r["z_tile_bad"] + sum(sum(r[f"{f}_receipt_bad"].values()) + r[f"{f}_z_bad"] for f, _ in forms)
    return r


def verify(args) -> int:
    src = Path(args.dump)
    doc = json.loads((src / "j10r90-check.json").read_text())
    arr = np.load(src / "j10r90-check.npz")
    jobs = []
    for c in doc["calls"]:
        key = f"{c['linear']}.{c['m']}"
        for i, u in enumerate(c["picks"]):
            recs = {nm: arr[f"{key}.{nm}"][i] for nm in c["gates"]}
            jobs.append((c, u, recs, arr[f"{key}.z"][i], i < args.v6))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = []
        for r in ex.map(_verify_unit, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    calls_: dict = {}
    for r in rows:
        k = f"{r['linear']}.{r['m']}"
        s = calls_.setdefault(k, {"units": 0, "v6_units": 0, "receipt_bad": 0, "z_bad": 0, "z_tile_bad": 0})
        s["units"] += 1
        s["v6_units"] += int("v6_receipt_bad" in r)
        s["receipt_bad"] += sum(sum(r.get(f"{f}_receipt_bad", {}).values()) for f in ("v5", "v6"))
        s["z_bad"] += r.get("v5_z_bad", 0) + r.get("v6_z_bad", 0)
        s["z_tile_bad"] += r["z_tile_bad"]
    res = {"schema": "j10r90/verify/v1", "check_device": doc.get("device"), "libs": doc.get("libs"),
           "definition": "h1tile.j10r_reference", "pairs": PAIRS, "calls": calls_, "units": rows,
           "s": round(time.time() - t0, 1), "ok": all(r["bad"] == 0 for r in rows) and bool(rows),
           "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (src / "j10r90-check.json",
                                                                                    src / "j10r90-check.npz")}}
    (out / "j10r90-verify.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"calls": calls_, "ok": res["ok"], "s": res["s"]}, indent=1), flush=True)
    return 0 if res["ok"] else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "time", "verify"])
    ap.add_argument("--lib", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--linears", default=",".join(LINEARS))
    ap.add_argument("--picks", type=int, default=24)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dump", help="verify: check's out/ (j10r90-check.json and .npz)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--v6", type=int, default=1, help="verify: units a call also taken through v6 (stream=True)")
    ap.add_argument("--wgmma", help="time: a dir with the s8_linear ship and its wgmma.py, configs.py: a second plain arm")
    args = ap.parse_args()
    args.ms = [int(x) for x in args.ms.split(",")]
    args.linears = [x for x in LINEARS if x in args.linears.split(",")]
    if any(m % T for m in args.ms):
        ap.error(f"--ms {args.ms}: whole row blocks of {T}")
    if args.what == "verify":
        if not args.dump:
            ap.error("verify needs --dump")
        return verify(args)
    if not args.lib:
        ap.error(f"{args.what} needs --lib")
    return check(args) if args.what == "check" else time_(args)


if __name__ == "__main__":
    sys.exit(main())
