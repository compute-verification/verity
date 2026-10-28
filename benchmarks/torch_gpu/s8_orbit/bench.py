"""The orbit's wall time at M 16,384 on a checkpoint's linears, against cuBLASLt (torch._int_mm) and wgmma.linear, alone and
with SHA-512 rows hashed beside it.

Each arm is one item of about --seconds: back-to-back calls on one stream between CUDA events, after three warm-up calls.
  int_mm        torch._int_mm(x, wᵀ), int32 out
  wgmma         s8_linear's wgmma.scaled_mm at its default config, bf16 out (the exact linear), when its cubin is given
  orbit<BN>     the orbit kernel, int32 out (`orbit_gpu.Lib.mm(y32=)`), and orbit<BN>.bf16 with the bf16 epilogue
  bfix          s8orbit_bfix, B's carries' corrections after the orbit, and orbit<best>+bfix the two together
κ, a correction MAC's time in tensor MACs (one MAC a B carry and a row of A, M/2 a carry), is bfix's time over those MACs
against PEAK and against int_mm's time a MAC.
X is int8 uniform on ±127; W is the checkpoint's [N, K] int8 (q, k, v and gate, up concatenated as served). An arm records
ms a call, θ = MACs / s / PEAK (PEAK MACs a second at F_MAX), its SM clock from NVML samples every 100 ms past the item's
first second, and θ at that clock (θ · F_MAX / MHz). An orbit arm also records the clock its CTAs saw (clock64 over
globaltimer, the last call's).

--sha adds, per linear, SHA-512 alone (the engine's row kernel on 32 KiB rows, 257 compressions a row) and then beside the
orbit: the orbit's persistent grid on CTAS of the SMs while the row kernel loops on a second stream over the rest. Its rate
is compressions a second over the item, and its fraction of --sha-bar (compressions a second on the whole card).

    python bench.py --cubin C --out OUT --snapshot DIR [--layer 0] [--m 16384] [--linears qkv,o,gate_up,down]
                    [--tiles 96,80,64] [--wgmma-cubin C] [--sha] [--sha-ctas 132,112,92,66]

Writes OUT/bench.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orbit_gpu as G  # noqa: E402
from cpu_check import LINEARS, Shards  # noqa: E402

PEAK, F_MAX = 989.4e12, 1830.0
SHA_BAR = 4.355e9
SHA_ROW, SHA_ROWS, SHA_SLOTS = 32768, 20, 1024


class Smi:
    """NVML's SM clock every 100 ms, stamped on arrival, from nvidia-smi on the leased GPU."""

    def __init__(self, out: Path) -> None:
        dev = os.environ.get("GPU_LEASE_UUID") or str(torch.cuda.current_device())
        self.samples: list[tuple[float, float]] = []
        self.log = (out / "smi.csv").open("w")
        self.p = subprocess.Popen(["nvidia-smi", "-i", dev, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits",
                                   "-lms", "100"], stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self) -> None:
        for line in self.p.stdout:
            now = time.time()
            self.log.write(f"{now:.3f},{line.strip()}\n")
            try:
                mhz = float(line)
            except ValueError:
                continue
            self.samples.append((now, mhz))

    def window(self, t0: float, t1: float) -> dict:
        s = [m for t, m in self.samples if t0 + 1.0 <= t <= t1]
        if not s:
            return {"smi_samples": 0}
        return {"smi_samples": len(s), "sm_mhz": float(np.mean(s)), "sm_mhz_min": float(min(s))}

    def close(self) -> None:
        self.p.terminate()
        self.log.close()


def timed(fn, seconds: float, stream: torch.cuda.Stream | None = None) -> dict:
    stream = stream or torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(3):
            fn()
        b.record()
        b.synchronize()
        n = max(3, int(seconds * 1e3 / (a.elapsed_time(b) / 3)))
        t0 = time.time()
        a.record()
        for _ in range(n):
            fn()
        b.record()
        b.synchronize()
        t1 = time.time()
    return {"calls": n, "ms": a.elapsed_time(b) / n, "t0": t0, "t1": t1}


def rates(r: dict, macs: int, smi: Smi) -> dict:
    r.update(smi.window(r["t0"], r["t1"]))
    r["theta"] = macs / (r["ms"] / 1e3) / PEAK
    if "sm_mhz" in r:
        r["theta_clk"] = r["theta"] * F_MAX / r["sm_mhz"]
    return r


def cta_mhz(clk: torch.Tensor, grid: int) -> float:
    c = clk[: 4 * grid].view(grid, 4).double()
    return float((c[:, 2] - c[:, 0]).sum() / (c[:, 3] - c[:, 1]).sum() * 1e3)


class Sha:
    """The engine's row kernel in its default variant (0000, sha512x.cuh's default compression) over SHA_ROWS rows of
    32 KiB in each of SHA_SLOTS slots, one call."""

    def __init__(self) -> None:
        sys.path.insert(0, str(HERE.parent / "sha512_rows"))
        import engine as E
        self.E = E.Engine()
        self.variant = self.E.default
        t = torch.empty((SHA_SLOTS, SHA_ROWS * SHA_ROW), dtype=torch.uint8, device="cuda").random_()
        self.tensors = [t]
        self.table = self.E.rows([[(0, i * SHA_ROW, SHA_ROW)] for i in range(SHA_ROWS)], self.tensors)
        self.out = self.E.row_digests(self.table, self.variant)
        self.comps = SHA_SLOTS * SHA_ROWS * ((SHA_ROW + 17 + 127) // 128)

    def __call__(self) -> None:
        self.E.row_digests(self.table, self.variant, self.out)


def beside(lib: G.Lib, sha: Sha, x, wb, y32, bn: int, ctas: int, seconds: float, macs: int, smi: Smi) -> dict:
    """The orbit on `ctas` CTAs on one stream and SHA-512 on another, each kept two calls deep; the hash counts only the
    calls that end inside the orbit's window."""
    sg, sh = torch.cuda.Stream(), torch.cuda.Stream()
    for s in (sg, sh):
        with torch.cuda.stream(s):
            lib.mm(x, wb, y32=y32, bn=bn, ctas=ctas)
            sha()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    start.record(sg)
    sh.wait_event(start)
    ev: dict[str, list] = {"g": [], "h": []}

    def push(key: str, s: torch.cuda.Stream, fn) -> None:
        if sum(not e.query() for e in ev[key][-3:]) < 2:
            with torch.cuda.stream(s):
                fn()
            e = torch.cuda.Event(enable_timing=True)
            e.record(s)
            ev[key].append(e)

    t0 = time.time()
    while time.time() - t0 < seconds:
        push("g", sg, lambda: lib.mm(x, wb, y32=y32, bn=bn, ctas=ctas))
        push("h", sh, sha)
    torch.cuda.synchronize()
    t1 = time.time()
    g_end = [start.elapsed_time(e) for e in ev["g"]]
    h_end = [t for t in (start.elapsed_time(e) for e in ev["h"]) if t <= g_end[-1]]
    r = {"bn": bn, "ctas": ctas, "gemm_calls": len(g_end), "ms": g_end[-1] / len(g_end), "sha_calls": len(h_end),
         "sha_comps_per_s": sha.comps * len(h_end) / (h_end[-1] / 1e3) if h_end else 0.0, "t0": t0, "t1": t1}
    r["sha_frac_bar"] = r["sha_comps_per_s"] / SHA_BAR
    return rates(r, macs, smi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--linears", default="qkv,o,gate_up,down")
    ap.add_argument("--tiles", default="96,80,64")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--wgmma-cubin", default=None)
    ap.add_argument("--sha", action="store_true")
    ap.add_argument("--sha-ctas", default="132,112,92,66")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = G.Lib(Path(a.cubin))
    tiles = [int(v) for v in a.tiles.split(",")]
    wg = None
    if a.wgmma_cubin:
        os.environ["S8_WGMMA_CUBIN"] = a.wgmma_cubin
        sys.path.insert(0, str(HERE.parent / "s8_linear"))
        import wgmma as wg
    sha = Sha() if a.sha else None
    smi = Smi(out)
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    sh = Shards(Path(a.snapshot))
    doc: dict = {"cubin_sha256": lib.sha256, "layout": lib.layout, "sms": lib.sms, "m": a.m, "layer": a.layer,
                 "peak_macs_per_s": PEAK, "f_max_mhz": F_MAX, "sha_bar": SHA_BAR, "device": torch.cuda.get_device_name(),
                 "linears": {}}
    if sha:
        doc["sha"] = {"variant": sha.variant, "comps_a_call": sha.comps}
    for name in a.linears.split(","):
        w = torch.from_numpy(np.concatenate([sh.get(f"model.layers.{a.layer}.{p}.weight") for p in LINEARS[name]])).cuda()
        n, k = w.shape
        macs = a.m * n * k
        x = torch.randint(-127, 128, (a.m, k), generator=gen, dtype=torch.int8, device="cuda")
        sa = torch.rand(a.m, generator=gen, device="cuda") / 64
        sw = torch.rand(n, generator=gen, device="cuda") / 64
        prep = G.prepare(w)
        wb, bc = prep.wb, prep.carries
        y32 = torch.empty((a.m, n), dtype=torch.int32, device="cuda")
        y = torch.empty((a.m, n), dtype=torch.bfloat16, device="cuda")
        clk = torch.zeros(4 * lib.sms, dtype=torch.int64, device="cuda")
        wt = w.t()
        arms: dict[str, dict] = {}
        arms["int_mm"] = rates(timed(lambda: torch._int_mm(x, wt), a.seconds), macs, smi)
        if wg is not None:
            cfg = wg.default_cfg(a.m, n, k)
            arms["wgmma"] = {"cfg": str(cfg), **rates(timed(lambda: wg.scaled_mm(x, sa, w, sw, y, cfg), a.seconds), macs, smi)}
        for bn in tiles:
            info = lib.mm(x, wb, y32=y32, bn=bn, clk=clk)
            r = rates(timed(lambda bn=bn: lib.mm(x, wb, y32=y32, bn=bn, clk=clk), a.seconds), macs, smi)
            torch.cuda.synchronize()
            arms[f"orbit{bn}"] = {**info, **r, "cta_mhz": cta_mhz(clk, info["grid"])}
            r = rates(timed(lambda bn=bn: lib.mm(x, wb, y=y, sa=sa, sw=sw, bn=bn), a.seconds), macs, smi)
            arms[f"orbit{bn}.bf16"] = r
        best = min(tiles, key=lambda bn: arms[f"orbit{bn}"]["ms"])
        if prep.fix is not None:
            r = timed(lambda: lib.bfix(x, y32, *prep.fix), a.seconds)
            r.update(smi.window(r["t0"], r["t1"]))
            arms["bfix"] = r
            arms[f"orbit{best}+bfix"] = {"ms": arms[f"orbit{best}"]["ms"] + r["ms"]}
        base = arms["int_mm"]["ms"]
        for key, r in arms.items():
            r["vs_int_mm"] = r["ms"] / base
            if "wgmma" in arms:
                r["vs_wgmma"] = r["ms"] / arms["wgmma"]["ms"]
        rec = {"n": n, "k": k, "macs": macs, "b_carries": bc, "b_density": sum(bc) / (7 * (n // 2) * (k // 2)),
               "g_entries": prep.g_entries, "fix_slots": 32 * int(prep.fix[0][-1]) if prep.fix is not None else 0,
               "best_tile": best, "arms": arms}
        if "bfix" in arms:
            # a correction MAC is one carry against one row of the other side (M/2 of them); kappa is its time in tensor MACs
            per_mac = arms["bfix"]["ms"] / 1e3 / (sum(bc) * (a.m // 2))
            rec["correction_macs"] = sum(bc) * (a.m // 2)
            rec["kappa_vs_peak"] = per_mac * PEAK
            rec["kappa_vs_int_mm"] = per_mac / (base / 1e3 / macs)
            rec["kappa_g_vs_peak"] = arms["bfix"]["ms"] / 1e3 / (prep.g_entries * (a.m // 2)) * PEAK
        if sha:
            alone = timed(sha, a.seconds)
            alone["comps_per_s"] = sha.comps / (alone["ms"] / 1e3)
            alone["frac_bar"] = alone["comps_per_s"] / SHA_BAR
            alone.update(smi.window(alone["t0"], alone["t1"]))
            rec["sha_alone"] = alone
            rec["beside"] = [beside(lib, sha, x, wb, y32, best, int(c), a.seconds, macs, smi) for c in a.sha_ctas.split(",")]
            for r in rec["beside"]:
                r["vs_int_mm"] = r["ms"] / base
        doc["linears"][name] = rec
        print(json.dumps({"linear": name, "kappa_vs_peak": rec.get("kappa_vs_peak"), "kappa_vs_int_mm": rec.get("kappa_vs_int_mm"),
                          **{key: {f: r.get(f) for f in ("ms", "theta", "theta_clk", "vs_int_mm", "vs_wgmma", "sm_mhz", "cta_mhz")}
                             for key, r in arms.items()},
                          "beside": [{f: r.get(f) for f in ("ctas", "ms", "vs_int_mm", "sha_frac_bar")} for r in rec.get("beside", [])]}),
              flush=True)
        del w, x, wb, y32, y
        torch.cuda.empty_cache()
    smi.close()
    (out / "bench.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
