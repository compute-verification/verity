"""Commit overhead as a share of serve time, projected from committed bytes per step and a hash's throughput.

    python commit_model.py                       # the card, then the seeded H200 tables: SHA-512 today, the throughput 5% needs
    python commit_model.py --card [GPU ...]      # the recommendation card: Qwen3-8B B32 per GPU, card rows x hashes, hm96 rates
    python commit_model.py --card --hm96 'H200/SHA-512/pre=520,300'   # the card with a new 4 KiB rate (prefill, decode GB/s)
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
    #: unit -> fraction of the non-logit bytes it keeps; "call" is today
    units: dict[str, float]
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
UNITS_QWEN = {"call": 1.0, "block": _frac(36_864, QWEN_TODAY), "layer": _frac(20_480, QWEN_TODAY),
              "layer-folded": _frac(12_288, QWEN_TODAY)}

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

HASHES = ("SHA-512", "BLAKE3 keyed", "SHA-512 r32", "SHA-512 r24")
HM96_RUNS = {"RTX PRO 6000": "r20261018-074305-3fd5", "H200": "r20261018-074230-9ff4", "B200": "r20261018-074256-26b4",
             "L40S": "r20261018-074239-8faa"}
#: hm96-format leaves at 4 KiB, GB/s of committed bytes: gpu -> hash -> salts "pre" (computed off the step) or "step" (inside
#: the leaf kernel) -> (prefill, decode). Prefill reads the 1 GiB window's rate, decode the 142 MB one (a decode step's bytes at
#: per-Call units). note:hash/20261018T0632Z-draft-intpipe section 4, runs HM96_RUNS.
HM96: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "RTX PRO 6000": {"SHA-512": {"pre": (565, 296), "step": (446, 182)}, "SHA-512 r32": {"pre": (990, 540), "step": (742, 318)},
                     "SHA-512 r24": {"pre": (1042, 614), "step": (794, 363)},
                     "BLAKE3 keyed": {"pre": (1152, 703), "step": (944, 450)}},
    "B200": {"SHA-512": {"pre": (425, 238), "step": (337, 150)}, "SHA-512 r32": {"pre": (1133, 493), "step": (761, 282)},
             "SHA-512 r24": {"pre": (1526, 666), "step": (988, 370)}, "BLAKE3 keyed": {"pre": (1411, 725), "step": (1057, 433)}},
    "H200": {"SHA-512": {"pre": (390, 214), "step": (306, 129)}, "SHA-512 r32": {"pre": (1042, 458), "step": (707, 256)},
             "SHA-512 r24": {"pre": (1389, 605), "step": (897, 324)}, "BLAKE3 keyed": {"pre": (1278, 681), "step": (1010, 408)}},
    "L40S": {"SHA-512": {"pre": (409, 251), "step": (317, 165)}, "SHA-512 r32": {"pre": (485, 359), "step": (392, 246)},
             "SHA-512 r24": {"pre": (492, 392), "step": (404, 273)}, "BLAKE3 keyed": {"pre": (558, 437), "step": (491, 330)}},
}
#: Served rows-sha512 today (per-Call, 256 B leaves, salts in the leaf kernel): hashing GPU ms per (prefill, decode) step, from
#: the serve runs whose stock times the Qwen3-8B workloads carry.
SERVED_256: dict[str, tuple[float, float]] = {"RTX PRO 6000": (167.0, 1.73), "H200": (227.0, 2.88), "B200": (205.0, 2.49)}


def kept_bytes(p: Phase, frac: float) -> float:
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


CARD_ROWS = (("per-Call, 256 B leaves (today)", "call", 256), ("per-Call, 4 KiB", "call", 4096),
             ("per-block, 4 KiB", "block", 4096), ("per-layer, 4 KiB", "layer", 4096),
             ("residual-folded per-layer, 4 KiB", "layer-folded", 4096))


def qwen(gpu: str) -> Workload:
    return next(w for w in WORKLOADS if w.name == f"Qwen3-8B B32 compiled, {gpu}")


def card_cells(gpu: str, hm96=HM96, served=SERVED_256) -> dict[str, dict[str, dict[str, float] | None]]:
    """Run overhead (whole B32 run's hashing over its stock serve) per card row and hash, salts "pre" and "step"; None where
    nothing is measured. At 256 B only served SHA-512 is measured, with salts on the step; "pre" removes the salt's 5 of 14
    compressions."""
    w = qwen(gpu)
    serve = sum(p.serve_ms_per_step * p.steps for p in w.phases)
    out: dict[str, dict[str, dict[str, float] | None]] = {}
    for label, unit, leaf in CARD_ROWS:
        row: dict[str, dict[str, float] | None] = {}
        for h in HASHES:
            if leaf == 256:
                if h != "SHA-512" or gpu not in served:
                    row[h] = None
                    continue
                step = sum(ms * p.steps for ms, p in zip(served[gpu], w.phases)) / serve
                row[h] = {"pre": step * (sha512_leaf_compressions(256) - SHA512_SALT) / sha512_leaf_compressions(256),
                          "step": step}
            else:
                row[h] = {m: overhead(w, dict(zip(("prefill", "decode"), hm96[gpu][h][m])), unit)["run"] for m in ("pre", "step")}
        out[label] = row
    return out


def smallest_change(gpu: str, target: float = 0.05, hash_name: str = "SHA-512", hm96=HM96, served=SERVED_256) -> str:
    """The first card row (coarsening order) where `hash_name` is under `target`, salts on the step if that suffices."""
    for label, row in card_cells(gpu, hm96, served).items():
        c = row[hash_name]
        if c and c["step"] < target:
            return f"{label}, salts on the step ({_pct(c['step'])})"
        if c and c["pre"] < target:
            return f"{label}, salts precomputed off the step ({_pct(c['pre'])}; on the step {_pct(c['step'])})"
    return f"none: {hash_name} stays above {_pct(target)} at every row"


def card(gpu: str, target: float = 0.05, hm96=HM96, served=SERVED_256) -> str:
    w = qwen(gpu)
    serve = sum(p.serve_ms_per_step * p.steps for p in w.phases)

    def cell(c: dict[str, float] | None) -> str:
        if c is None:
            return "n.m."
        ok = lambda x: " ✓" if x < target else ""
        return f"{_pct(c['pre'])}{ok(c['pre'])} (step {_pct(c['step'])}{ok(c['step'])})"

    rows = [f"| {gpu} (B32 run serve {serve:,.0f} ms) | " + " | ".join(HASHES) + " |", "|---|" + "---|" * len(HASHES)]
    rows += [f"| {label} | " + " | ".join(cell(row[h]) for h in HASHES) + " |"
             for label, row in card_cells(gpu, hm96, served).items()]
    rows.append(f"\nSHA-512 under {_pct(target)}: {smallest_change(gpu, target, 'SHA-512', hm96, served)}.")
    return "\n".join(rows)


def parse_hm96(specs: list[str], base=HM96) -> dict:
    """`GPU/HASH/MODE=PREFILL[,DECODE]` (GB/s at 4 KiB; one number sets both phases) over a copy of `base`."""
    out = {g: {h: dict(m) for h, m in hs.items()} for g, hs in base.items()}
    for s in specs:
        key, _, val = s.partition("=")
        gpu, h, mode = key.split("/")
        nums = [float(x) for x in val.split(",")]
        if gpu not in out or h not in out[gpu] or mode not in ("pre", "step") or len(nums) not in (1, 2):
            raise ValueError(f"--hm96 {s!r}: want GPU/HASH/{{pre,step}}=PREFILL[,DECODE], GPU in {sorted(out)}, HASH in {HASHES}")
        out[gpu][h][mode] = (nums[0], nums[-1])
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
    ap.add_argument("--hm96", action="append", default=[], metavar="GPU/HASH/MODE=PREFILL[,DECODE]",
                    help="replace an hm96 4 KiB rate, GB/s, e.g. 'H200/SHA-512/pre=520,300' (repeatable)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    if a.check:
        bad = check()
        print("\n".join(bad) if bad else f"ok: {len(CHECKS)} cells of circuits' table, leaf counts and run bytes reproduce")
        return 1 if bad else 0
    try:
        hm96 = parse_hm96(a.hm96)
    except ValueError as e:
        ap.error(str(e))
    gpus = a.card or list(SERVED_256)
    cards = "\n\n".join(card(g, a.target, hm96) for g in gpus)
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
            "hm96": hm96,
            "card": {g: card_cells(g, hm96) for g in gpus},
            "smallest_change_sha512": {g: smallest_change(g, a.target, "SHA-512", hm96) for g in gpus},
        }, indent=1, ensure_ascii=False))
        return 0
    print(f"Recommendation card: Qwen3-8B B32 compiled, a whole run's hashing over its stock serve time, hm96 leaves, salts "
          f"precomputed (step: salts on the step), ✓ under {_pct(a.target)}:\n\n{cards}\n")
    leaves = sorted({L for r in rates for L in r.gbps}) if rates is not RATES else (256, 4096)
    print(f"Overhead at today's bytes (per-Call units), hashing over stock serve time:\n\n{table_today(rates, leaves)}\n")
    print(f"Throughput a hash needs for {_pct(a.target)} of serve time:\n\n{table_required(a.target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
