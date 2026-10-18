"""Commit overhead as a share of serve time, projected from committed bytes per step and a hash's throughput.

    python commit_model.py                       # the card, then the seeded H200 tables: SHA-512 today, the throughput 5% needs
    python commit_model.py --card [GPU ...]      # the recommendation card: Qwen3-8B B32 per GPU, fused per-Call units, hm96 rates
    python commit_model.py --card --hm96 'H200/SHA-512 sharaw/step=480'   # with a new SHA-512 prefill rate (or =PREFILL,DECODE)
    python commit_model.py --gbps 900 --leaf 4096 [--hash-name X]   # one candidate hash at a measured GB/s
    python commit_model.py --rate 7.13e9 --per-leaf 44 --leaf 4096  # or at compressions/s and compressions per leaf
    python commit_model.py --target 0.05 --json  # every number as JSON
    python commit_model.py --check               # the seeds reproduce circuits' published cells, or exit 1

The model is throughput only: a phase's hashing time is the bytes it commits over the hash's GB/s, and its overhead is that over
the phase's stock serve time. It assumes hashing does not overlap serving, which circuits measured on H200 (hashing shares the
integer issue slots with the GEMMs; note:circuits-rows-sha512-floor, note:circuits-commit-defaults-overlap). It leaves out
gathers, staging copies, device-to-host traffic and host work; the baseline note breaks those down per run.

A coarser proof unit keeps a fraction of a step's bytes: the fraction applies to the bytes inside the decoder layers and the
embedding, and the logits (committed for decode tokens) are kept whole.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass

GB = 1e9

#: Compressions an hm96-sha512 row leaf in a frame-v3-sha512 tree costs besides its row: the salt y (3), c (2), the hm96 leaf (2),
#: the frame leaf (2) and its share of the tree's nodes (about 2). benchmarks/pouw/commit_gpu/README.md, "Leaf size".
SHA512_LEAF_FIXED = 11
#: Of those, the salt's (y and c), which the served kernel computes inside its leaf kernel.
SHA512_SALT = 5


def sha512_row_compressions(leaf_bytes: int) -> int:
    """`sha512/row/v2` over `leaf_bytes`: its constant 128-byte first block is precomputed, then the row plus 17 bytes of padding."""
    return math.ceil((leaf_bytes + 17) / 128)


def sha512_leaf_compressions(leaf_bytes: int) -> int:
    """Compressions per committed leaf of today's rows-sha512 scheme: 14 at 256 B, 44 at 4 KiB (note:circuits-commit-bytes-width-dma)."""
    return sha512_row_compressions(leaf_bytes) + SHA512_LEAF_FIXED


def gbps_from_rate(compressions_per_s: float, leaf_bytes: int, per_leaf: float) -> float:
    return compressions_per_s * leaf_bytes / per_leaf / GB


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    bytes_per_step: float          # committed bytes per step today (the Commit's per-Call boundaries)
    serve_ms_per_step: float       # stock serve time per step
    logits_bytes_per_step: float = 0.0


@dataclass(frozen=True)
class Workload:
    name: str
    gpu: str
    phases: tuple[Phase, ...]
    #: unit -> fraction of the non-logit bytes it keeps ("call" is today), or the bytes it keeps a step, by phase name
    units: dict[str, float | dict[str, float]]
    source: str


@dataclass(frozen=True)
class Rate:
    name: str
    gbps: dict[int, float | dict[str, float]]   # leaf bytes -> GB/s, or phase name -> GB/s
    source: str


def _frac(per_token_layer: float, today: float) -> float:
    return per_token_layer / today


# Per token per layer, today (compiled, layout_bytes.py) and what a unit keeps (granularity.py): note:circuits-commit-bytes-width-dma
# section 1, note:circuits-commit-defaults-overlap "Update (04:15Z)". block = layer + o_proj's output + the input norm's residual output;
# layer = K + V + the two hidden vectors vLLM's fused add-norm reads; layer-folded = K + V + one hidden vector (their sum).
LLAMA_TODAY, QWEN_TODAY = 84_736, 145_180
UNITS_LLAMA = {"call": 1.0, "block": _frac(18_432, LLAMA_TODAY), "layer": _frac(10_240, LLAMA_TODAY),
               "layer-folded": _frac(6_144, LLAMA_TODAY)}
#: Per-Call units with fused epilogues, the served Commit's (proofs, PRF-62/64/68, 18 Oct): RMSNorm inside the next dot's unit,
#: the residual add inside o_proj's and down's, silu(x Wg) (x Wu) inside gate/up's, so no norm output, gate/up output or down output
#: is committed. Of today's 145,180 B a token a layer they keep 71,680: qkv 12,288, rope_q 8,192, kv_k 2,048, attention 8,192,
#: the post-attention residual 8,192 (o_proj's unit outputs it, in o_proj's place: the MLP's units read it), the post-MLP residual
#: 7,964 (35 of 36 layers), activation 24,576 and down where no residual follows it (228). Bytes a step are layout_bytes.py's
#: all-three unit (origin/cursor/commit-speed-21f3) on r20261018-031955-d2b3 run 2: 26.657 GB over the 4 prefill steps and
#: 4.674 GB of decode, spread over 64 steps like the per-Call bytes; 31.331 GB a run, 0.498 of today.
QWEN_FUSED = {"prefill": 26.656692032e9 / 4, "decode": 4.674396352e9 / 64}
UNITS_QWEN = {"call": 1.0, "block": _frac(36_864, QWEN_TODAY), "layer": _frac(20_480, QWEN_TODAY),
              "layer-folded": _frac(12_288, QWEN_TODAY), "fused": QWEN_FUSED}

WORKLOADS = (
    Workload(
        "Llama-3.2-1B B32 compiled", "H200",
        (Phase("prefill", 4, 3.51328e9, 9.350, 5.130e6), Phase("decode", 63, 41.567e6, 1.300, 6.580e6)),
        UNITS_LLAMA,
        "bytes: layouts of r20261018-012341-e3b1 (b32-p512-mixed, 16.672 GB a run); serve: its stock-prof arm, nsys GPU ms per step "
        "(run 8, vy-glm53-s1)"),
    *(Workload(
        f"Qwen3-8B B32 compiled, {gpu}", gpu,
        (Phase("prefill", 4, 13.4908e9, pre_ms, 6.077e6), Phase("decode", 64, 140.056e6, dec_ms, 7.673e6)),
        UNITS_QWEN,
        "bytes: layouts of r20261018-031955-d2b3 (b32-p512-mixed, 62.927 GB a run: 4 prefill steps of 2,560-token chunks, its 63 "
        f"decode steps' bytes spread over the 64 the serve runs time); serve: {run}'s stock arm, model-kernel GPU ms per step "
        "(note:circuits-decode-host-costs)")
      for gpu, pre_ms, dec_ms, run in (("RTX PRO 6000", 117.0, 15.0, "r20261018-065034-aa8f"),
                                       ("H200", 57.5, 5.65, "r20261018-031955-d2b3"),
                                       ("B200", 29.9, 4.08, "r20261018-065600-a6f3"))),
    Workload(
        "GLM-5.3 TP8 eager B1, per rank", "H200",
        (Phase("run", 1, 7.184635164e9, 5713.7),),
        {"call": 1.0},
        "Commit of record r20261016-213336-6c35: bytes_bound per rank (28,064,992 leaves of 256 B, 33 steps), serve = the control "
        "run's wall (mean of 3; eager and host-bound, so wall, not GPU busy)"),
)

RATES = (
    Rate("SHA-512 H200 measured", {256: 66.12, 1024: 152.5, 4096: 230.6},
         "rows_floor r20261017-234343-61c6 (leaves + tree of a 256 MiB window; 4 KiB at 2 blocks/SM)"),
    Rate("SHA-512 H200 raw rate", {L: gbps_from_rate(4.79e9, L, sha512_leaf_compressions(L)) for L in (256, 1024, 4096)},
         "4.79 G compressions/s (r20261017-234343-61c6, COMMIT-GPU's sha mode on sm_90) / sha512_leaf_compressions"),
    Rate("SHA-512 sm_120 raw rate", {L: gbps_from_rate(7.13e9, L, sha512_leaf_compressions(L)) for L in (256, 1024, 4096)},
         "7.13 G compressions/s (r20261016-110413-c5d1, RTX PRO 6000) / sha512_leaf_compressions; against H200 serve times"),
)

#: Data-sheet HBM bandwidth, GB/s: a hash that reads committed bytes back from memory runs no faster than this.
HBM_GBPS = {"H200": 4800.0, "RTX PRO 6000": 1792.0, "B200": 8000.0}

#: The card's columns. Reduced-round SHA-512 is out (broken below 35 steps).
COLUMNS = ("SHA-512", "SHA-512 sharaw", "BLAKE3 keyed")
MODES = ("pre", "step")
HM96_RUNS = {"RTX PRO 6000": "r20261018-074305-3fd5", "H200": "r20261018-074230-9ff4", "B200": "r20261018-074256-26b4"}
#: hm96-format leaves at 4 KiB, GB/s of committed bytes: gpu -> hash -> salts "pre" (read ahead of the step, digest computed
#: off it) or "step" (inside the leaf kernel) -> (1 GiB window, 142 MB window). note:hash/20261018T0632Z-draft-intpipe section 4,
#: runs HM96_RUNS.
HM96: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "RTX PRO 6000": {"SHA-512": {"pre": (565, 296), "step": (446, 182)}, "BLAKE3 keyed": {"pre": (1152, 703), "step": (944, 450)}},
    "B200": {"SHA-512": {"pre": (425, 238), "step": (337, 150)}, "BLAKE3 keyed": {"pre": (1411, 725), "step": (1057, 433)}},
    "H200": {"SHA-512": {"pre": (390, 214), "step": (306, 129)}, "BLAKE3 keyed": {"pre": (1278, 681), "step": (1010, 408)}},
}
HM96_SMALL_RUNS = {"B200": "r20261018-082407-e58b", "H200": "r20261018-082416-39e9", "RTX PRO 6000": "r20261018-082443-2c46"}
SMALL_LEAVES = (1221, 2441, 4883, 9766)
#: The same arms' latency, µs (eager, events around the window), at SMALL_LEAVES leaves of 4 KiB (5 / 10 / 20 / 40 MB):
#: intpipe section 5, runs HM96_SMALL_RUNS. gpu -> hash -> mode -> µs at each size.
HM96_SMALL: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {
    "B200": {"SHA-512": {"pre": (231, 251, 304, 358), "step": (449, 488, 560, 636)},
             "BLAKE3 keyed": {"pre": (74, 80, 96, 113), "step": (147, 160, 185, 209)}},
    "H200": {"SHA-512": {"pre": (238, 258, 305, 416), "step": (460, 499, 568, 705)},
             "BLAKE3 keyed": {"pre": (66, 73, 88, 125), "step": (133, 145, 168, 207)}},
    "RTX PRO 6000": {"SHA-512": {"pre": (206, 224, 245, 297), "step": (394, 431, 472, 542)},
                     "BLAKE3 keyed": {"pre": (70, 78, 83, 102), "step": (138, 151, 166, 190)}},
}
LEAVES_142MB, BYTES_1GIB = 34_737, 2 ** 30


def hm96_rate(gpu: str, hash_name: str, mode: str, nbytes: float) -> float:
    """GB/s for one window of `nbytes`: latency interpolated linearly in bytes between the measured windows (5, 10, 20, 40 MB,
    142 MB, 1 GiB), the 5 MB latency below them and the 1 GiB rate above."""
    big = HM96[gpu][hash_name][mode]
    pts = [(n * 4096, us) for n, us in zip(SMALL_LEAVES, HM96_SMALL[gpu][hash_name][mode])]
    pts += [(LEAVES_142MB * 4096, LEAVES_142MB * 4096 / (big[1] * GB) * 1e6), (BYTES_1GIB, BYTES_1GIB / (big[0] * GB) * 1e6)]
    if nbytes >= pts[-1][0]:
        return big[0]
    if nbytes <= pts[0][0]:
        return nbytes / (pts[0][1] * 1e-6) / GB
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        if nbytes <= b1:
            return nbytes / ((t0 + (nbytes - b0) / (b1 - b0) * (t1 - t0)) * 1e-6) / GB
    raise AssertionError


#: sharaw's SHA-512 at the 13.49 GB prefill step, salts in the commit (note:hash/20261018T0650Z-draft-sharaw: v1_cg8 on H200
#: r20261018-080737-be25 and B200 r20261018-080746-d042, inl lf0_cg on RTX PRO 6000 r20261018-081929-15b3). Salts read ahead
#: take the salt's 5 of 44 compressions off (H200's measured split, 33.49 -> 26.83 ms without sr_salt8, would give 1.25x).
SHARAW_PREFILL = {"H200": 402.6, "B200": 446.0, "RTX PRO 6000": 615.0}
#: A decode commit on a second stream beside a memory-bound stand-in for the decode step (intpipe section 6): the stand-in's
#: slowdown and the same commit's critical path alone, µs, (pre, step), at the measured window nearest a fused decode step's
#: 73 MB: one tree over 4 x 18.9 MB = 75.6 MB. Slowdown: r20261018-100805-647b (B200), -093252-72b2 (H200), -093301-8e6e
#: (RTX PRO 6000); alone: r20261018-090115-e4ef, -085249-443d, -085257-dde2. Measured for SHA-512 only; the card applies the
#: same hidden fraction to BLAKE3 keyed.
OVERLAP_75MB = {"B200": ((201, 244), (428, 723)), "H200": ((187, 215), (502, 830)), "RTX PRO 6000": ((56, 66), (367, 613))}


def hidden_fraction(gpu: str, mode: str) -> float:
    (slow, alone), i = OVERLAP_75MB[gpu], MODES.index(mode)
    return 1 - slow[i] / alone[i]
#: Served rows-sha512 today (per-Call, 256 B leaves, salts in the leaf kernel): hashing GPU ms per (prefill, decode) step, from
#: the serve runs whose stock times the Qwen3-8B workloads carry.
SERVED_256: dict[str, tuple[float, float]] = {"RTX PRO 6000": (167.0, 1.73), "H200": (227.0, 2.88), "B200": (205.0, 2.49)}


def kept_bytes(p: Phase, frac: float | dict[str, float]) -> float:
    if isinstance(frac, dict):
        return frac[p.name]
    return p.logits_bytes_per_step + (p.bytes_per_step - p.logits_bytes_per_step) * frac


def overhead(w: Workload, gbps: float | dict[str, float], unit: str = "call") -> dict[str, float]:
    """Hashing time over serve time, per phase and for the run (the steps summed). `gbps` is one rate or one per phase name."""
    frac = w.units[unit]
    out, hash_ms, serve_ms = {}, 0.0, 0.0
    for p in w.phases:
        g = gbps[p.name] if isinstance(gbps, dict) else gbps
        h = kept_bytes(p, frac) / (g * GB) * 1e3
        out[p.name] = h / p.serve_ms_per_step
        hash_ms += h * p.steps
        serve_ms += p.serve_ms_per_step * p.steps
    if len(w.phases) > 1:
        out["run"] = hash_ms / serve_ms
    return out


def required_gbps(w: Workload, target: float, unit: str = "call") -> dict[str, float]:
    """The GB/s at which hashing is `target` of serve time, per phase and for the run."""
    frac = w.units[unit]
    out = {p.name: kept_bytes(p, frac) / (target * p.serve_ms_per_step / 1e3) / GB for p in w.phases}
    if len(w.phases) > 1:
        out["run"] = (sum(kept_bytes(p, frac) * p.steps for p in w.phases)
                      / (target * sum(p.serve_ms_per_step * p.steps for p in w.phases) / 1e3) / GB)
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.3g}%" if x < 10 else f"{100 * x:,.0f}%"


def table_today(rates=RATES, leaves=(256, 4096)) -> str:
    """Every seeded rate is H200's or set against H200's serve times, so the table holds the H200 workloads."""
    cols = [(r, L) for r in rates for L in leaves if L in r.gbps]
    head = "| workload | phase | " + " | ".join(f"{r.name}, {L} B ({r.gbps[L]:.0f} GB/s)" for r, L in cols) + " |"
    rows = [head, "|" + "---|" * (2 + len(cols))]
    for w in (w for w in WORKLOADS if w.gpu == "H200"):
        per = {(r.name, L): overhead(w, r.gbps[L]) for r, L in cols}
        for ph in per[(cols[0][0].name, cols[0][1])]:
            rows.append(f"| {w.name} | {ph} | " + " | ".join(_pct(per[(r.name, L)][ph]) for r, L in cols) + " |")
    return "\n".join(rows)


def table_required(target: float, ref: Rate = RATES[0], ref_leaf: int = 4096) -> str:
    rows = [f"| workload | unit | phase | GB/s for {_pct(target)} | x {ref.name} at {ref_leaf} B | x HBM |", "|---|---|---|---|---|---|"]
    for w in (w for w in WORKLOADS if w.gpu == "H200"):
        for unit in w.units:
            for ph, g in required_gbps(w, target, unit).items():
                rows.append(f"| {w.name} | {unit} | {ph} | {g:,.0f} | {g / ref.gbps[ref_leaf]:.1f} | {g / HBM_GBPS[w.gpu]:.2f} |")
    return "\n".join(rows)


CARD_ROWS = ("per-Call today, 256 B leaves", "per-Call fused, 4 KiB", "per-Call fused, 4 KiB, decode commit on a second stream")
Overrides = dict[tuple[str, str, str], tuple[float, float | None]]


def qwen(gpu: str) -> Workload:
    return next(w for w in WORKLOADS if w.name == f"Qwen3-8B B32 compiled, {gpu}")


def fused_rates(gpu: str, col: str, mode: str, overrides: Overrides | None = None) -> dict[str, float]:
    """GB/s per phase for the fused rows: hm96's rate at each phase's window, sharaw's prefill rate in its column, then any
    override (a prefill rate, and a decode rate if given)."""
    w = qwen(gpu)
    h = "BLAKE3 keyed" if col == "BLAKE3 keyed" else "SHA-512"
    r = {p.name: hm96_rate(gpu, h, mode, kept_bytes(p, w.units["fused"])) for p in w.phases}
    if col == "SHA-512 sharaw":
        full = sha512_leaf_compressions(4096)
        r["prefill"] = SHARAW_PREFILL[gpu] * (1 if mode == "step" else full / (full - SHA512_SALT))
    if overrides and (gpu, col, mode) in overrides:
        p, d = overrides[(gpu, col, mode)]
        r["prefill"] = p
        if d is not None:
            r["decode"] = d
    return r


def fused_cost(gpu: str, col: str, mode: str, second_stream: bool, overrides: Overrides | None = None) -> dict[str, float]:
    """A whole run's hashing ms per phase and its share of the run's stock serve time, for the fused rows. With the decode
    commit on a second stream, decode costs what the stand-in measured it doesn't hide."""
    w = qwen(gpu)
    r = fused_rates(gpu, col, mode, overrides)
    ms = {p.name: kept_bytes(p, w.units["fused"]) * p.steps / (r[p.name] * GB) * 1e3 for p in w.phases}
    if second_stream:
        ms["decode"] *= 1 - hidden_fraction(gpu, mode)
    serve = sum(p.serve_ms_per_step * p.steps for p in w.phases)
    return {**{f"{k}_ms": v for k, v in ms.items()}, "serve_ms": serve, "run": sum(ms.values()) / serve,
            **{f"{k}_gbps": v for k, v in r.items()}}


def card_cells(gpu: str, overrides: Overrides | None = None) -> dict[str, dict[str, dict[str, float] | None]]:
    """Run overhead per card row and column, salts "pre" (read ahead) and "step"; None where nothing is measured. The 256 B row
    is served rows-sha512 as measured (salts on the step); its "pre" takes the salt's 5 of 14 compressions off."""
    w = qwen(gpu)
    serve = sum(p.serve_ms_per_step * p.steps for p in w.phases)
    step = sum(ms * p.steps for ms, p in zip(SERVED_256[gpu], w.phases)) / serve
    today = {"pre": step * (sha512_leaf_compressions(256) - SHA512_SALT) / sha512_leaf_compressions(256), "step": step}
    out: dict[str, dict[str, dict[str, float] | None]] = {CARD_ROWS[0]: {c: today if c == "SHA-512" else None for c in COLUMNS}}
    for label, second in zip(CARD_ROWS[1:], (False, True)):
        out[label] = {c: {m: fused_cost(gpu, c, m, second, overrides)["run"] for m in MODES} for c in COLUMNS}
    return out


def verdict(gpu: str, hash_name: str, target: float = 0.05, overrides: Overrides | None = None) -> str:
    """Whether `hash_name` ("SHA-512": either SHA-512 column; or "BLAKE3 keyed") gets under `target` on the fused rows, where
    first, and if not, the gap and the prefill GB/s that would close it with the second stream's decode cost."""
    cols = ("SHA-512", "SHA-512 sharaw") if hash_name == "SHA-512" else (hash_name,)
    cells = [(fused_cost(gpu, c, m, s, overrides), c, m, s) for s in (False, True) for m in ("step", "pre") for c in cols]
    for cost, c, m, s in cells:
        if cost["run"] < target:
            return (f"{hash_name} under {_pct(target)}: yes, {_pct(cost['run'])} ({c}, salts "
                    f"{'read ahead' if m == 'pre' else 'on the step'}{', decode on a second stream' if s else ''})")
    best, c, m, _ = min(cells, key=lambda x: x[0]["run"])
    room = target * best["serve_ms"] - best["decode_ms"]
    need = (f"prefill would need {qwen(gpu).units['fused']['prefill'] * 4 / (room * 1e-3) / GB:,.0f} GB/s against "
            f"{best['prefill_gbps']:,.0f}" if room > 0 else "decode alone is over the target")
    return (f"{hash_name} under {_pct(target)}: no; best {_pct(best['run'])} ({c}, salts read ahead, decode on a second stream), "
            f"{100 * (best['run'] - target):.2g} points over; {need}")


def card(gpu: str, target: float = 0.05, overrides: Overrides | None = None) -> str:
    w = qwen(gpu)
    serve = sum(p.serve_ms_per_step * p.steps for p in w.phases)

    def cell(c: dict[str, float] | None) -> str:
        if c is None:
            return "n.m."
        ok = lambda x: " ✓" if x < target else ""
        return f"{_pct(c['pre'])}{ok(c['pre'])} (step {_pct(c['step'])}{ok(c['step'])})"

    rows = [f"| {gpu} (B32 run serve {serve:,.0f} ms) | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
    rows += [f"| {label} | " + " | ".join(cell(row[c]) for c in COLUMNS) + " |" for label, row in card_cells(gpu, overrides).items()]
    rows.append("")
    rows += [f"- {verdict(gpu, h, target, overrides)}." for h in ("SHA-512", "BLAKE3 keyed")]
    return "\n".join(rows)


def parse_overrides(specs: list[str]) -> Overrides:
    """`GPU/COLUMN/MODE=PREFILL[,DECODE]`, GB/s for the fused rows: one number replaces the prefill rate, two both."""
    out: Overrides = {}
    for s in specs:
        key, _, val = s.partition("=")
        parts = key.split("/")
        nums = [float(x) for x in val.split(",") if x]
        if len(parts) != 3 or parts[0] not in SERVED_256 or parts[1] not in COLUMNS or parts[2] not in MODES or len(nums) not in (1, 2):
            raise ValueError(f"--hm96 {s!r}: want GPU/COLUMN/{{pre,step}}=PREFILL[,DECODE], GPU in {list(SERVED_256)}, "
                             f"COLUMN in {COLUMNS}")
        out[(parts[0], parts[1], parts[2])] = (nums[0], nums[1] if len(nums) == 2 else None)
    return out


#: Cells of circuits' table (note:circuits-commit-bytes-width-dma section 1): GPU hashing at the H200 floor over stock serve, prefill.
CHECKS = (
    ("Llama-3.2-1B B32 compiled", "call", 256, "prefill", 5.68), ("Llama-3.2-1B B32 compiled", "call", 4096, "prefill", 1.63),
    ("Qwen3-8B B32 compiled, H200", "call", 256, "prefill", 3.55), ("Qwen3-8B B32 compiled, H200", "call", 4096, "prefill", 1.02),
    ("Llama-3.2-1B B32 compiled", "layer-folded", 256, "prefill", 0.43),
    ("Llama-3.2-1B B32 compiled", "layer-folded", 4096, "prefill", 0.125),
    ("Qwen3-8B B32 compiled, H200", "layer-folded", 256, "prefill", 0.31),
    ("Qwen3-8B B32 compiled, H200", "layer-folded", 4096, "prefill", 0.088),
    ("Llama-3.2-1B B32 compiled", "layer-folded", 4096, "run", 0.059),
    ("Qwen3-8B B32 compiled, H200", "layer-folded", 4096, "run", 0.044),
)


def check(tol: float = 0.05) -> list[str]:
    """Every CHECKS cell within `tol` (relative), every leaf count as circuits states it, and each workload's bytes summing to the run's."""
    bad = []
    by_name = {w.name: w for w in WORKLOADS}
    for name, unit, leaf, ph, want in CHECKS:
        got = overhead(by_name[name], RATES[0].gbps[leaf], unit)[ph]
        if abs(got / want - 1) > tol:
            bad.append(f"{name} {unit} {leaf} B {ph}: model {got:.3f}, circuits {want}; a seed moved, fix it or its source")
    if (sha512_leaf_compressions(256), sha512_leaf_compressions(4096)) != (14, 44):
        bad.append("rows-sha512 compressions per leaf are 14 at 256 B and 44 at 4 KiB (note:circuits-commit-bytes-width-dma)")
    totals = [("Llama-3.2-1B B32 compiled", 16.672e9)] + [(qwen(g).name, 62.927e9) for g in SERVED_256]
    for name, total in totals:
        got = sum(p.bytes_per_step * p.steps for p in by_name[name].phases)
        if abs(got / total - 1) > 1e-3:
            bad.append(f"{name}: phases sum to {got / GB:.3f} GB, the run commits {total / GB:.3f} GB")
    for g in SERVED_256:
        got = sum(kept_bytes(p, qwen(g).units["fused"]) * p.steps for p in qwen(g).phases)
        if abs(got / 31.331088384e9 - 1) > 1e-6:
            bad.append(f"{qwen(g).name}: fused units sum to {got / GB:.3f} GB, layout_bytes.py's all-three gives 31.331 GB")
    for g in HM96:
        for h in HM96[g]:
            for m in MODES:
                at = hm96_rate(g, h, m, LEAVES_142MB * 4096), hm96_rate(g, h, m, SMALL_LEAVES[-1] * 4096)
                want = HM96[g][h][m][1], SMALL_LEAVES[-1] * 4096 / (HM96_SMALL[g][h][m][-1] * 1e-6) / GB
                if any(abs(x / y - 1) > 1e-9 for x, y in zip(at, want)):
                    bad.append(f"hm96_rate({g}, {h}, {m}) misses its measured windows: {at} against {want}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", type=float, default=0.05, help="overhead budget as a fraction of serve time (default 0.05)")
    ap.add_argument("--gbps", type=float, help="a candidate hash's measured throughput, GB/s of committed bytes")
    ap.add_argument("--rate", type=float, help="or its compressions (or blocks) per second ...")
    ap.add_argument("--per-leaf", type=float, help="... and compressions per leaf, tree share included")
    ap.add_argument("--leaf", type=int, default=4096, help="leaf bytes the candidate's rate is for (default 4096)")
    ap.add_argument("--hash-name", default="candidate")
    ap.add_argument("--card", nargs="*", metavar="GPU", help=f"only the recommendation card's tables (default all of {list(SERVED_256)})")
    ap.add_argument("--hm96", action="append", default=[], metavar="GPU/COLUMN/MODE=PREFILL[,DECODE]",
                    help="replace a fused row's rate, GB/s: one number the prefill rate, two prefill and decode, e.g. "
                         "'H200/SHA-512 sharaw/step=480' (repeatable)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    if a.check:
        bad = check()
        print("\n".join(bad) if bad else f"ok: {len(CHECKS)} cells of circuits' table, leaf counts and run bytes reproduce")
        return 1 if bad else 0
    try:
        ov = parse_overrides(a.hm96)
    except ValueError as e:
        ap.error(str(e))
    gpus = a.card or list(SERVED_256)
    cards = "\n\n".join(card(g, a.target, ov) for g in gpus)
    if a.card is not None and not a.json:
        print(cards)
        return 0
    rates = RATES
    if a.gbps or a.rate:
        if a.rate and not a.per_leaf:
            ap.error("--rate needs --per-leaf")
        g = a.gbps or gbps_from_rate(a.rate, a.leaf, a.per_leaf)
        rates = (Rate(a.hash_name, {a.leaf: g}, "command line"),)
    if a.json:
        print(json.dumps({
            "target": a.target,
            "workloads": [asdict(w) for w in WORKLOADS],
            "rates": [asdict(r) for r in rates],
            "overhead": {w.name: {r.name: {str(L): {u: overhead(w, g, u) for u in w.units} for L, g in r.gbps.items()} for r in rates}
                         for w in WORKLOADS},
            "required_gbps": {w.name: {u: required_gbps(w, a.target, u) for u in w.units} for w in WORKLOADS},
            "overrides": {"/".join(k): v for k, v in ov.items()},
            "card": {g: card_cells(g, ov) for g in gpus},
            "card_inputs": {g: {c: {m: {f"second_stream={s}": fused_cost(g, c, m, s, ov) for s in (False, True)}
                                    for m in MODES} for c in COLUMNS} for g in gpus},
            "hidden_fraction": {g: {m: hidden_fraction(g, m) for m in MODES} for g in gpus},
            "verdicts": {g: {h: verdict(g, h, a.target, ov) for h in ("SHA-512", "BLAKE3 keyed")} for g in gpus},
        }, indent=1, ensure_ascii=False))
        return 0
    print(f"Recommendation card: Qwen3-8B B32 compiled, a whole run's hashing over its stock serve time, hm96 leaves, salts "
          f"read ahead (step: salts on the step), ✓ under {_pct(a.target)}:\n\n{cards}\n")
    leaves = sorted({L for r in rates for L in r.gbps}) if rates is not RATES else (256, 4096)
    print(f"Overhead at today's bytes (per-Call units), hashing over stock serve time:\n\n{table_today(rates, leaves)}\n")
    print(f"Throughput a hash needs for {_pct(a.target)} of serve time:\n\n{table_required(a.target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
