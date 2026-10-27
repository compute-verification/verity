"""H1 + fx1's fused unit (h1u.cu) on one H200 or B200: gated bit for bit against compute accounting's served_h1/receipts.py
(its `call_receipts`) and `h1tile.reference`, then timed interleaved with the plain int8 GEMM of the same
call and H1's noise floor (h1noise.cu's midstate kernel), at Llama-3.1-8B's linears and M = 1, 32, 256, 2048, 8192.

    python h1ubench.py --lib h1u.so [--variant NAME=h1u_x.so ...] [--noise-lib h1noise.so]
                       [--noise-variant NAME=h1noise_x.so ...] --out OUT [--reps 5] [--ms ...] [--linears ...]

A variant (another build of h1u.cu, e.g. h1u_b3.so with H1U_MINB=3 or h1u_pd4.so with H1U_PD=4) passes every gate
below on the same inputs, then is timed as arm h1u-NAME in the same interleaved reps; a noise variant is arm floor-NAME.
A build whose gate fails is not timed.

The receipts are a side computation beside the served product (z, the plain A B^T, is the served call's own GEMM), so a
call served with them costs plain + h1u and gamma = 1 + h1u / plain, plain the faster of torch._int_mm and vLLM's
cutlass_scaled_mm. Units are the call's full 32 x 32 tiles; M = 1 is one padded row tile (31 zero rows), as h1bench.py
counts it. Every unit is live here.

Gates before any timing, every one in this run (a failed gate exits 4 and times nothing):
  * anchors (CPU, small calls): kh_unit against h1tile.reference on every unit of a 64 x 64 x 512 call (m = 2) whose
    second row tile is dead (live = 0: the zero receipt, PouwH1Tile_v4's dead tile) and of a 32 x 32 x 4608 call (m = 3);
    the device seeds against seeds.seed / receipt_seed;
  * every timed shape: 0xA5 poison, then kh_unit's receipts against call_receipts on sampled units (first, last and
    random ones) and no poison left anywhere; then a no-write control (a launch over zero rows) under the same poison,
    which the comparison must reject;
  * every noise floor (h1noise.so, --noise-variant builds) at the shape's J: h1bench.py's gate.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "served_h1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import receipts as RC
from h1bench import KEY_BLOCKS, auto_j, plain_arms
from h1bench import Lib as NoiseLib
from h1bench import gate as noise_gate
from hb_kbench import LINEARS, MS, POISON, Clock, capture, inputs, one_call_ms, w_seed, x_seed

from verity.ml.protocols.pouw.circuit import h1tile as HT
from verity.ml.protocols.pouw.circuit import seeds as SD

SEED_WORDS = 4  # compressions a unit spends on seed_j and sigma_R


_WS: dict = {}


def workspace(nbytes: int) -> torch.Tensor:
    """kh_unit_ws's workspace, one buffer for every build (they run one at a time), grown before any capture."""
    if _WS.get("t") is None or _WS["t"].numel() < nbytes:
        _WS["t"] = None
        torch.cuda.empty_cache()
        _WS["t"] = torch.empty((max(nbytes, 64),), dtype=torch.uint8, device="cuda")
    return _WS["t"]


class Lib:
    """One build of h1u.cu; with ws, its two-launch form kh_unit_ws (the noise into the workspace, then the unit)."""

    def __init__(self, path: str, ws: bool = False):
        h = self.h = ctypes.CDLL(path)
        vp, u32 = ctypes.c_void_p, ctypes.c_uint
        h.kh_seeds.argtypes = [vp, ctypes.c_ulonglong, u32, vp, vp]
        h.kh_unit.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp]
        h.kh_words.restype = u32
        h.kh_words.argtypes = [u32]
        h.kh_digits.argtypes = [u32]
        self.ws = ws
        if ws:
            h.kh_unit_ws.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp]
            h.kh_ws_bytes.restype = ctypes.c_ulonglong
            h.kh_ws_bytes.argtypes = [u32, u32, u32]
            h.kh_noise_blocks.restype = u32
            h.kh_noise_blocks.argtypes = [u32]

    @staticmethod
    def has_ws(path: str) -> bool:
        return hasattr(ctypes.CDLL(path), "kh_unit_ws")

    def words(self, k: int) -> int:
        return int(self.h.kh_words(k))

    def seeds(self, data: torch.Tensor, base: int, units: int, out: torch.Tensor) -> None:
        if self.h.kh_seeds(data.data_ptr(), base, units, out.data_ptr(), torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("kh_seeds")

    def ws_buffer(self, m: int, n: int, k: int) -> torch.Tensor | None:
        return workspace(int(self.h.kh_ws_bytes(m, n, k))) if self.ws else None

    def unit(self, a: torch.Tensor, b: torch.Tensor, m: int, seeds: torch.Tensor, live: torch.Tensor | None,
             out: torch.Tensor) -> None:
        n, k, st = b.shape[0], a.shape[1], torch.cuda.current_stream().cuda_stream
        lv = None if live is None else live.data_ptr()
        if self.ws:
            nb = int(self.h.kh_noise_blocks(k))
            j = auto_j((a.shape[0] // 32) * (n // 32), nb, torch.cuda.get_device_properties(0).multi_processor_count)
            rc = self.h.kh_unit_ws(a.data_ptr(), b.data_ptr(), m, n, k, seeds.data_ptr(), lv, out.data_ptr(),
                                   self.ws_buffer(a.shape[0], n, k).data_ptr(), j, st)
        else:
            rc = self.h.kh_unit(a.data_ptr(), b.data_ptr(), m, n, k, seeds.data_ptr(), lv, out.data_ptr(), st)
        if rc:
            raise RuntimeError(f"kh_unit{'_ws' if self.ws else ''} rc {rc}")


def be_words(b: bytes) -> list[int]:
    return [int.from_bytes(b[i:i + 8], "big") for i in range(0, len(b), 8)]


def call_data(salt: bytes, root_a: bytes, root_b: bytes) -> torch.Tensor:
    """kh_seeds' data: SEED_TAG, RECEIPT_TAG, salt, root_a, root_b as big-endian 64-bit words (int64 on the device)."""
    w = be_words(SD.SEED_TAG) + be_words(SD.RECEIPT_TAG) + be_words(salt) + be_words(root_a) + be_words(root_b)
    return torch.tensor(np.array(w, dtype=np.uint64).view(np.int64), device="cuda")


def digest(row) -> bytes:
    return b"".join((int(v) & ((1 << 64) - 1)).to_bytes(8, "big") for v in row)


class Call:
    """One call's device buffers for kh_unit: A (rows padded to a 32-row tile), B, seeds, receipts."""

    def __init__(self, lib: Lib, a: torch.Tensor, b: torch.Tensor, salt: bytes, root_a: bytes, root_b: bytes, base: int):
        self.lib, self.a, self.b, self.base = lib, a, b, base
        self.salt, self.roots = salt, (root_a, root_b)
        self.m = a.shape[0]
        self.units = (self.m // 32) * (b.shape[0] // 32)
        self.words = lib.words(a.shape[1])
        self.data = call_data(salt, root_a, root_b)
        self.seeds = torch.empty((max(1, self.units), 16), dtype=torch.int64, device="cuda")
        self.out = torch.empty((max(1, self.units), self.words), dtype=torch.int32, device="cuda")

    def run(self, live: torch.Tensor | None = None, m: int | None = None) -> None:
        self.lib.seeds(self.data, self.base, self.units, self.seeds)
        self.lib.unit(self.a, self.b, self.m if m is None else m, self.seeds, live, self.out)

    def poison(self) -> None:
        """0xA5 over the receipts and, for the two-launch form, its workspace: a twin's noise left there by an earlier
        build has the same seeds, so a noise kernel that wrote nothing would otherwise pass."""
        self.out.fill_(POISON)
        w = self.lib.ws_buffer(self.m, self.b.shape[0], self.a.shape[1])
        if w is not None:
            w.fill_(0xA5)

    def twin(self, lib: Lib) -> Call:
        """The same call (inputs, salt, roots, base) on another build, with its own seeds and receipts."""
        return Call(lib, self.a, self.b, self.salt, *self.roots, self.base)


def gate_seeds(call: Call, picks: list[int]) -> int:
    got = call.seeds.cpu().numpy()
    bad = 0
    for u in picks:
        p = call.base + u
        bad += digest(got[u, :8]) != SD.seed(call.salt, p, *call.roots)
        bad += digest(got[u, 8:]) != SD.receipt_seed(call.salt, p, *call.roots)
    return int(bad)


def shape_ref(call: Call, picks: list[int]) -> torch.Tensor:
    ref = RC.call_receipts(RC.Call(call.a, call.b), call.salt, call.base, *call.roots, only=picks)
    return torch.from_numpy(ref.view(np.int32)).to("cuda")


def gate_shape(call: Call, picks: list[int], ref_t: torch.Tensor) -> dict:
    """0xA5 poison, the kernel, its sampled receipts against receipts.py's (ref_t, shape_ref's); then the no-write
    control."""
    pk = torch.tensor(picks, device="cuda")

    def once(m: int | None) -> tuple[int, int]:
        call.poison()
        call.run(m=m)
        torch.cuda.synchronize()
        return int((call.out[pk] != ref_t).sum()), int((call.out[:call.units] == POISON).sum())
    bad, poison = once(None)
    seed_bad = gate_seeds(call, picks)
    k_bad, _ = once(0)
    return {"units": call.units, "picks": picks, "bad_words": bad, "poison_left": poison, "seed_bad": seed_bad,
            "control_bad_words": k_bad, "control_rejected": k_bad > 0,
            "ok": bad == 0 and poison == 0 and seed_bad == 0 and k_bad > 0}


def anchors(libs: dict[str, Lib], rng: random.Random) -> list[dict]:
    """Every build's kh_unit against h1tile.reference (the Definition's words, CPU) on two small calls, dead row tile
    included."""
    res = []
    for m, n, k, dead in ((64, 64, 512, (1,)), (32, 32, 4608, ())):
        g = torch.Generator().manual_seed(rng.getrandbits(32))
        a = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=g)
        b = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=g)
        for ri in dead:
            a[32 * ri:32 * ri + 32] = 0
        salt, ra, rb = (rng.randbytes(64) for _ in range(3))
        base = rng.getrandbits(40)
        live = torch.tensor([0 if ri in dead else 1 for ri in range(m // 32)], dtype=torch.uint8, device="cuda")
        calls = {name: Call(lib, a.cuda(), b.cuda(), salt, ra, rb, base) for name, lib in libs.items()}
        got = {}
        for name, call in calls.items():
            call.poison()
            call.run(live=live)
            torch.cuda.synchronize()
            got[name] = call.out.cpu().numpy().view(np.uint32)
        call = next(iter(calls.values()))
        bad, t0 = dict.fromkeys(libs, 0), time.time()
        for u, (ri, cj) in enumerate(RC.units(m, n)):
            ar = (a[32 * ri:32 * ri + 32].to(torch.int64) & 0xFF).tolist()
            br = (b[32 * cj:32 * cj + 32].to(torch.int64) & 0xFF).tolist()
            want = np.array([0] * call.words if ri in dead else HT.reference(ar, br, salt, base + u, (ra, rb))[1],
                            dtype=np.uint32)
            for name in libs:
                bad[name] += int((got[name][u] != want).sum())
        for name, c in calls.items():
            r = {"lib": name, "m": m, "n": n, "k": k, "digits": int(c.lib.h.kh_digits(k)), "dead_row_blocks": list(dead),
                 "units": c.units, "bad_words": bad[name], "seed_bad": gate_seeds(c, list(range(c.units))),
                 "reference_s": round(time.time() - t0, 2)}
            r["ok"] = r["bad_words"] == 0 and r["seed_bad"] == 0
            res.append(r)
    return res


_NOISE_GATES: dict = {}


def floor_gate(arm: str, nlib: NoiseLib, nb: int, j: int) -> dict:
    """h1bench.py's gate (kn_ca against hashlib, kn_mid's writes and its sink fold at J against kn_ca, a no-write control)
    for one noise build at nb blocks a unit and J, once a run."""
    key = (arm, nb, j)
    if key not in _NOISE_GATES:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(0x4831 + nb + 64 * j)
        _NOISE_GATES[key] = noise_gate(nlib, nb, [j], gen)
    return _NOISE_GATES[key]


def bench(libs: dict[str, Lib], nlibs: dict[str, NoiseLib], args, name: str, n: int, k: int, m: int, Xh: np.ndarray,
          clock: Clock, l2: int, sms: int, rng: random.Random) -> dict:
    """libs: arm name -> build, "h1u" (the main build) first; nlibs: arm name -> noise build ("floor" the main one). A
    build is timed only if its gate passed."""
    lay = HT.Layout(32, 32, k)
    mp = max(32, m)
    X = torch.from_numpy(Xh[:m]).to("cuda")
    a = X if m >= 32 else torch.cat([X, X.new_zeros(32 - m, k)])
    b = torch.from_numpy(inputs(w_seed(name), n, k)).to("cuda")
    call = Call(libs["h1u"], a.contiguous(), b, rng.randbytes(64), rng.randbytes(64), rng.randbytes(64),
                rng.getrandbits(40))
    builds = {arm: call if arm == "h1u" else call.twin(lib) for arm, lib in libs.items()}
    S = call.units
    out: dict = {"linear": name, "m": m, "n": n, "k": k, "units": S, "padded_row_block": m < 32,
                 "blocks_per_unit": lay.blocks + KEY_BLOCKS + SEED_WORDS, "receipt_words": call.words, "arms": {},
                 "gates": {}}
    picks = sorted({0, S - 1, *rng.sample(range(S), min(S, args.picks))})
    out["picks"] = picks
    ref_t = shape_ref(call, picks)
    for arm, c in builds.items():
        out["gates"][arm] = g = gate_shape(c, picks, ref_t)
        print(json.dumps({"linear": name, "m": m, "arm": arm, "gate": {x: g[x] for x in g if x != "picks"}}), flush=True)
    out["gate"] = out["gates"]["h1u"]
    failed = [arm for arm, g in out["gates"].items() if not g["ok"]]
    if failed:
        out["gate_failed"] = failed
    if not out["gate"]["ok"]:
        out["error"] = "gate failed: nothing timed"
        return out
    arms: dict = {arm: (lambda c: lambda i: c.run())(c) for arm, c in builds.items() if arm not in failed}
    if nlibs:
        sj = torch.randint(0, 256, (S, 64), dtype=torch.uint8, device="cuda").view(torch.int64)
        J = auto_j(S, lay.blocks, sms)
        out["floor_j"] = J
        for arm, nl in nlibs.items():
            gs = [floor_gate(arm, nl, lay.blocks, J), floor_gate(arm, nl, KEY_BLOCKS, 1)]
            out["gates"][arm] = {"ok": all(g["ok"] for g in gs), "noise": gs}
            if not out["gates"][arm]["ok"]:
                out.setdefault("gate_failed", []).append(arm)
                continue
            snk = torch.empty((nl.items(S, lay.blocks, J) * 32,), dtype=torch.int32, device="cuda")
            ksnk = torch.empty((nl.items(S, KEY_BLOCKS, 1) * 32,), dtype=torch.int32, device="cuda")
            arms[arm] = (lambda nl, snk, ksnk: lambda i: (nl.mid(sj, S, lay.blocks, J, snk, True),
                                                          nl.mid(sj, S, KEY_BLOCKS, 1, ksnk, True)))(nl, snk, ksnk)
    parms, info = plain_arms(name, n, k, m, Xh, l2)
    out.update(info)
    arms.update(parms)
    graphs, calls = {}, {}
    for arm, fn in arms.items():
        t = one_call_ms(fn)
        G = max(1, min(400, round(args.item_ms / max(t, 1e-3))))
        graphs[arm], calls[arm] = capture(fn, G), G
    for gr in graphs.values():
        gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    per = {arm: [] for arm in graphs}
    mhz = {arm: [] for arm in graphs}
    order = list(graphs)
    for rep in range(args.reps):
        for arm in order[rep % len(order):] + order[:rep % len(order)]:
            e0.record()
            graphs[arm].replay()
            e1.record()
            e1.synchronize()
            per[arm].append(e0.elapsed_time(e1) * 1e3 / calls[arm])
            mhz[arm].append(clock.sm())
    for arm in graphs:
        out["arms"][arm] = {"calls_per_item": calls[arm], "us": [round(v, 3) for v in per[arm]],
                            "median_us": round(statistics.median(per[arm]), 3), "sm_mhz": mhz[arm]}
    med = {arm: v["median_us"] for arm, v in out["arms"].items()}
    plain = {p: med[p] for p in ("int_mm", "cutlass") if p in med}
    fp = min(plain, key=plain.get)
    comp = S * out["blocks_per_unit"]
    out.update({"fastest_plain": fp, "plain_us": plain[fp], "h1u_us": med["h1u"],
                "gamma": round(1 + med["h1u"] / plain[fp], 4),
                "h1u_gcomp_per_s": round(comp / (med["h1u"] * 1e3), 3),
                "ratios": {f"{x}/{p}": round(med[x] / v, 4) for x in med if x not in plain for p, v in plain.items()}})
    out["variants"] = {}
    for arm in [*builds, *nlibs]:
        base = "h1u" if arm in builds else "floor"
        if arm == base or arm not in med or base not in med:
            continue
        out["variants"][arm] = {"us": med[arm], "gamma": round(1 + med[arm] / plain[fp], 4),
                                "gcomp_per_s": round(comp / (med[arm] * 1e3), 3),
                                f"over_{base}": round(med[arm] / med[base], 4)}
    if "floor" in med:
        out["floor_us"] = med["floor"]
        out["h1u_over_floor"] = round(med["h1u"] / med["floor"], 4)
        out["gamma_floor"] = round(1 + med["floor"] / plain[fp], 4)
    out["mp"] = mp
    del graphs, builds, call
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--noise-lib")
    ap.add_argument("--noise-variant", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--item-ms", type=float, default=3.0)
    ap.add_argument("--picks", type=int, default=14)
    ap.add_argument("--no-ws", action="store_true", help="skip every build's two-launch form (arm NAME-ws)")
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--linears", default=",".join(LINEARS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"h1u": args.lib}
    for v in args.variant:
        vname, _, path = v.partition("=")
        paths[f"h1u-{vname}"] = path
    libs = {arm: Lib(p) for arm, p in paths.items()}
    if not args.no_ws:
        for arm, p in list(paths.items()):
            if Lib.has_ws(p):
                libs[f"{arm}-ws"] = Lib(p, ws=True)
                paths[f"{arm}-ws"] = f"{p} (kh_unit_ws)"
    npaths = {"floor": args.noise_lib} if args.noise_lib else {}
    for v in args.noise_variant:
        vname, _, path = v.partition("=")
        npaths[f"floor-{vname}"] = path
    nlibs = {arm: NoiseLib(p) for arm, p in npaths.items()}
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "L2_cache_size", 50 << 20))
    rng = random.Random(0x4831F1)
    doc = {"schema": "dither-hb/h1u/v1", "device": clock.name, "sms": props.multi_processor_count, "l2_bytes": l2,
           "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reference": "served_h1/receipts.py, h1tile.reference",
           "libs": paths, "noise_libs": npaths,
           "shapes": []}
    doc["anchors"] = anchors(libs, rng)
    print(json.dumps({"anchors": doc["anchors"]}), flush=True)
    bad_libs = doc["anchor_failed"] = sorted({x["lib"] for x in doc["anchors"] if not x["ok"]})
    (out / "h1u.json").write_text(json.dumps(doc, indent=1))
    if "h1u" in bad_libs:
        print("h1ubench: an anchor failed; nothing timed", file=sys.stderr)
        return 4
    for arm in bad_libs:
        print(f"h1ubench: {arm} failed an anchor; it is not timed", file=sys.stderr)
        del libs[arm]
    xs: dict = {}
    ms = [int(v) for v in args.ms.split(",")]
    for name in args.linears.split(","):
        n, k = LINEARS[name]
        if k not in xs:
            xs[k] = inputs(x_seed(k), max(ms), k)
        for m in ms:
            t0 = time.time()
            try:
                r = bench(libs, nlibs, args, name, n, k, m, xs[k], clock, l2, props.multi_processor_count, rng)
            except Exception as e:  # noqa: BLE001
                r = {"linear": name, "m": m, "error": "".join(traceback.format_exception(e))[-3000:]}
            r["wall_s"] = round(time.time() - t0, 2)
            doc["shapes"].append(r)
            print(json.dumps({x: r.get(x) for x in ("linear", "m", "units", "h1u_us", "floor_us", "plain_us", "fastest_plain",
                                                     "gamma", "gamma_floor", "h1u_over_floor", "h1u_gcomp_per_s", "variants",
                                                     "gate_failed", "error", "wall_s")}), flush=True)
            (out / "h1u.json").write_text(json.dumps(doc, indent=1))
    layer: dict = {}
    for r in doc["shapes"]:
        if "h1u_us" in r:
            d = layer.setdefault(r["m"], {"h1u_us": 0.0, "plain_us": 0.0, "floor_us": 0.0, "linears": [], "variants": {}})
            d["h1u_us"] += r["h1u_us"]
            d["plain_us"] += r["plain_us"]
            d["floor_us"] += r.get("floor_us", 0.0)
            d["linears"].append(r["linear"])
            for arm, v in r.get("variants", {}).items():
                d["variants"].setdefault(arm, {"us": 0.0, "linears": []})
                d["variants"][arm]["us"] += v["us"]
                d["variants"][arm]["linears"].append(r["linear"])
    for d in layer.values():
        d["gamma"] = round(1 + d["h1u_us"] / d["plain_us"], 4)
        d["gamma_floor"] = round(1 + d["floor_us"] / d["plain_us"], 4)
        for v in d["variants"].values():
            v["gamma"] = round(1 + v["us"] / d["plain_us"], 4) if v["linears"] == d["linears"] else None
    doc["per_layer"] = layer
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "h1u.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({"per_layer": layer}))
    bad = [r for r in doc["shapes"] if "error" in r or "gate_failed" in r]
    return 4 if bad or bad_libs else 0


if __name__ == "__main__":
    sys.exit(main())
