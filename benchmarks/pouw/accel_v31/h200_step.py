"""The H200 step from `v31 fill-bench`'s JSON lines (Dither's no-fold call, `"fill": true`) at M = 16,384, for each model
whose calls the lines cover, a step per (L', L, ROW, served or filler, variant): each arm's seconds a step, gamma_u against
wgmma.linear's step and gamma_lt against the same lines' cuBLASLt, the split and the residual.

A step's calls by (N, K):
  mqa16k52  a 52-layer MQA model at width 16,384: q and o (16,384, 16,384) twice a layer, kv (128, 16,384), up
            (65,536, 16,384) and down (16,384, 65,536) once, and the head (256, 16,384) once a step
  8b        Llama 3 8B's widths (GQA, 8 KV heads), 32 layers of QKV (6,144, 4,096), O (4,096, 4,096),
            gate and up fused (28,672, 4,096) and down (4,096, 14,336) back to back
The step's arm is the one a call of the protocol runs: a served call's seeds are per call (`dither.seeds(coin, receipt,
call)`), so `ovl_w`, the weight side every call; a filler's is `ovl_pad`, its pad and every coin a call. The split is a
serial call's parts with those seeds: tensor `gemm` (served: cuBLASLt's O, then the protocol GEMM over k < (S - 1) L),
hashing `coins`, `coins_w`, `pad` and `commit`, other `form_a` and `form_b`. gamma_u is against wgmma.linear's step from
the same lines where they time it (`wgl`, a -DV31_WG build's served bench), else against --wgmma's mqa16k52 step. The
residual is 1 - W / (step ρ), W the step's MACs and ρ the H200's rate (4,096 MACs an SM a clock, 132 SMs) at each reading
of its clock in `CLOCKS_MHZ`, side by side: the tensor clock (1,830 MHz, the data sheet's dense INT8 rate) and the boost
clock nvidia-smi reports (1,980 MHz).

    python3 h200_step.py BENCH.jsonl ... [--wgmma SECONDS]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = 52
CALLS = {(16384, 16384): 2 * LAYERS, (128, 16384): LAYERS, (65536, 16384): LAYERS, (16384, 65536): LAYERS,
         (256, 16384): 1}
MODELS = {"mqa16k52": CALLS, "8b": {(6144, 4096): 32, (4096, 4096): 32, (28672, 4096): 32, (4096, 14336): 32}}
#: the H200's SM clock under each reading; one constant until the card file states the tensor clock
CLOCKS_MHZ = {"tensor": 1830, "boost": 1980}
RHO = {name: 4096 * 132 * mhz * 1e6 for name, mhz in CLOCKS_MHZ.items()}
MHZ = " / ".join(f"{m:,}" for m in CLOCKS_MHZ.values())
FILL_ARMS = ("lt", "coins", "coins_w", "pad", "form_a", "form_b", "gemm", "commit", "call", "ovl", "ovl_w", "ovl_pad", "wgl")


def rows(paths: list[str]) -> list[dict]:
    """Each bench line, with its variant: the file's name past `fill-bench-` (run.sh writes a variant's lines to one
    file)."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.startswith("{") and '"gamma_ovl"' in line:
                out.append(json.loads(line) | {"variant": Path(p).stem.removeprefix("fill-").removeprefix("bench-")})
    return out


def fill_steps(lines: list[dict], m: int = 16384, calls: dict[tuple[int, int], int] = CALLS
               ) -> dict[tuple[int, int, int, bool, str], dict]:
    """Per (L', L, ROW, served, variant) of fill-bench's lines: each arm's seconds a step (None where a call lacks the
    arm), the step's arm and its seconds, the split and the step's MACs, for groups that timed every call."""
    groups: dict[tuple[int, int, int, bool, str], dict[tuple[int, int], dict]] = {}
    for r in lines:
        if r.get("fill") and r["m"] == m and (r["n"], r["k"]) in calls:
            key = (r["lp"], r["L"], r["row"], bool(r.get("o")), r.get("variant", ""))
            groups.setdefault(key, {})[(r["n"], r["k"])] = r
    out = {}
    for key, by in sorted(groups.items()):
        if set(by) != set(calls):
            continue
        ms = {a: [by[nk]["arms"].get(a, {}).get("ms") for nk in calls] for a in FILL_ARMS}
        sec = {a: None if None in v else sum(calls[nk] * x for nk, x in zip(calls, v)) / 1e3 for a, v in ms.items()}
        part = lambda *arms: sum(sec[a] or 0.0 for a in arms)  # noqa: E731
        rec = "ovl_w" if key[3] else "ovl_pad"
        mhz = [by[nk]["arms"][rec]["mhz"] for nk in calls if by[nk]["arms"].get(rec, {}).get("mhz")]
        out[key] = {"s": sec, "record": rec, "step": sec[rec], "tensor": part("gemm"),
                    "hashing": part("coins", "coins_w", "pad", "commit"), "other": part("form_a", "form_b"),
                    "macs": sum(calls[nk] * m * nk[0] * nk[1] for nk in calls),
                    "mhz": (min(mhz), max(mhz)) if mhz else None}
    return out


def residual(macs: int, step: float | None) -> str:
    """1 - W / (step ρ) at each reading of the clock, in `CLOCKS_MHZ`'s order."""
    return " / ".join(f"{1 - macs / (step * rho):.3f}" for rho in RHO.values()) if step else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--wgmma", type=float, default=None, help="wgmma.linear's mqa16k52 step, s")
    a = ap.parse_args()
    lines, wg, found = rows(a.files), {"mqa16k52": a.wgmma}, False
    for model, calls in MODELS.items():
        st = fill_steps(lines, calls=calls)
        if not st:
            continue
        found, ref = True, wg.get(model)
        print(f"\n{model}, Dither's no-fold call\n\n| L' | L | ROW | kind | variant | step s | arm | gamma_u | gamma_lt | "
              "ovl s | ovl_w s | ovl_pad s | call s | tensor s | hashing s | other s | hashing share | cuBLASLt s | "
              f"residual at {MHZ} MHz | SM MHz |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        for (lp, L, row, o, var), v in st.items():
            s, step = v["s"], v["step"]
            mhz = f"{v['mhz'][0]:.0f}-{v['mhz'][1]:.0f}" if v["mhz"] else "-"
            r = s["wgl"] or ref
            gu = f"{step / r:.3f}" if r and step else "-"
            split = v["tensor"] + v["hashing"] + v["other"]
            print(f"| {lp} | {L} | {row} | {'served' if o else 'filler'} | {var} | {f(step)} | {v['record']} | {gu} | "
                  f"{f(step / s['lt'] if step else None)} | {f(s['ovl'])} | {f(s['ovl_w'])} | {f(s['ovl_pad'])} | "
                  f"{f(s['call'])} | {v['tensor']:.3f} | {v['hashing']:.3f} | {v['other']:.3f} | "
                  f"{v['hashing'] / split:.1%} | {s['lt']:.3f} | {residual(v['macs'], step)} | "
                  f"{mhz} |")
        for (lp, L, row, o, var), v in st.items():
            step, lt, wl = v["step"], v["s"]["lt"], v["s"]["wgl"]
            vs = (f"gamma_u {step / wl:.3f} against wgmma.linear's {wl:.3f} s (these lines' wgl), " if wl and step
                  else f"gamma_u {step / ref:.3f} against wgmma's {ref} s, " if ref and step else "")
            print(f"summary: {model} no-fold {'served' if o else 'filler'} L {L} L' {lp} row {row} {var}: step "
                  f"{f(step)} s ({v['record']}), {vs}gamma_lt {f(step / lt if step else None)} against cuBLASLt's "
                  f"{lt:.3f} s; tensor {v['tensor']:.3f}, hashing {v['hashing']:.3f}, other {v['other']:.3f} s serial")
    if not found:
        print("summary: no complete step (a bench shape is missing)")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
