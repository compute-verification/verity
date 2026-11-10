"""Run a preset of train.py configs in parallel, each in its own process with a fixed thread count.

Results: $OUT/runs/<name>.json, logs: $OUT/logs/<name>.log, summary: $OUT/sweep.json. A config whose result already
exists in --resume-from (a previous run's out dir) is copied, not rerun.
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bench():
    base = {"tokens": 60 * 32 * 256, "eval_every": 10_000, "final_eval_batches": 4, "warmup": 10}
    return [
        {**base, "name": "bench-s256-dense-bf16-t4", "size": "s256", "threads": 4},
        {**base, "name": "bench-s256-dense-fp32-t4", "size": "s256", "threads": 4, "bf16": False},
        {**base, "name": "bench-s256-dense-bf16-t8", "size": "s256", "threads": 8},
        {**base, "name": "bench-s256-A8-int8k16-t4", "size": "s256", "threads": 4, "design": "A", "diag": True,
         "int8": True, "noise_k": 16, "r_dist": "int8"},
        {**base, "name": "bench-m384-dense-bf16-t8", "size": "m384", "threads": 8},
        {**base, "name": "bench-t128-dense-bf16-t2", "size": "t128", "threads": 2},
    ]


def grid(size, tokens, lr=2e-3, tag=None):
    """The first-signal grid at one size: baseline token ladder and lr check, design A by r/n with and without
    D₁RD₂, design B by ρ, controls, and the int8 noise sweep. MLP matrices hardened unless the name says `all`."""
    tag = tag or size
    b = {"size": size, "tokens": tokens, "lr": lr}
    A8 = {**b, "design": "A", "rank_frac": 1 / 8, "diag": True}
    cfgs = [{**b, "name": f"{tag}-base-D{tokens // f}", "tokens": tokens // f} for f in (1, 2, 4, 8, 16, 32)]
    cfgs += [{**b, "name": f"{tag}-base-lr{x:g}", "lr": x} for x in (lr / 2, lr * 2)]
    for inv in (32, 16, 8, 4):
        for diag in (False, True):
            cfgs.append({**b, "name": f"{tag}-A{inv}{'-diag' if diag else ''}", "design": "A", "rank_frac": 1 / inv,
                         "diag": diag})
    cfgs += [
        {**A8, "name": f"{tag}-A8-diag-uvlr4", "uv_lr_mult": 4.0},
        {**A8, "name": f"{tag}-A8-diag-lr{lr * 2:g}", "lr": lr * 2},
        {**A8, "name": f"{tag}-A8-diag-all", "harden": "all"},
        {**b, "name": f"{tag}-lowrank8", "design": "lowrank", "rank_frac": 1 / 8},
        {**b, "name": f"{tag}-frozen-diag", "design": "frozen", "diag": True},
    ]
    cfgs += [{**b, "name": f"{tag}-B{rho:g}", "design": "B", "rho": rho} for rho in (0.1, 0.25, 0.5, 1.0)]
    cfgs += [{**b, "name": f"{tag}-B0.5-all", "design": "B", "rho": 0.5, "harden": "all"}]
    cfgs += [{**b, "name": f"{tag}-int8-k{k}", "int8": True, "noise_k": k} for k in (0, 16, 64)]
    cfgs += [{**A8, "name": f"{tag}-A8-diag-int8-k{k}", "int8": True, "noise_k": k, "r_dist": "int8"}
             for k in (0, 1, 16, 64, 96)]
    return cfgs


def lite(size, tokens, threads, lr=2e-3):
    """The scaling subset at one size: baseline ladder, A at r/n 1/32, 1/8, 1/4 with D₁RD₂, B at ρ 0.25, 0.5, 1,
    and live noise k = 16 on the baseline and on A 1/8."""
    b = {"size": size, "tokens": tokens, "lr": lr, "threads": threads}
    cfgs = [{**b, "name": f"{size}-base-D{tokens // f}", "tokens": tokens // f} for f in (1, 2, 4, 8, 16)]
    cfgs += [{**b, "name": f"{size}-A{inv}-diag", "design": "A", "rank_frac": 1 / inv, "diag": True}
             for inv in (32, 8, 4)]
    cfgs += [{**b, "name": f"{size}-B{rho:g}", "design": "B", "rho": rho} for rho in (0.25, 0.5, 1.0)]
    cfgs += [{**b, "name": f"{size}-int8-k16", "int8": True, "noise_k": 16},
             {**b, "name": f"{size}-A8-diag-int8-k16", "design": "A", "rank_frac": 1 / 8, "diag": True,
              "int8": True, "noise_k": 16, "r_dist": "int8"}]
    return cfgs


def replan1(size="w256", tokens=20_000_000, lr=2e-3, threads=4):
    """After the first signal: the noise knee on the baseline shape and on B, B's int8 R and small ρ, B on all
    linears, and seeds for the baseline, B ρ = 0.5 and A 1/8 (the measurement's noise floor)."""
    b = {"size": size, "tokens": tokens, "lr": lr, "threads": threads}
    B5 = {**b, "design": "B", "rho": 0.5}
    cfgs = [{**b, "name": f"{size}-int8-k{k}", "int8": True, "noise_k": k} for k in (2, 4, 8)]
    cfgs += [{**B5, "name": f"{size}-B0.5-int8-k{k}", "int8": True, "noise_k": k, "r_dist": "int8"}
             for k in (0, 4, 16)]
    cfgs += [{**b, "name": f"{size}-B0.05", "design": "B", "rho": 0.05},
             {**b, "name": f"{size}-B0.25-all", "design": "B", "rho": 0.25, "harden": "all"},
             {**B5, "name": f"{size}-B0.5-lr{lr * 2:g}", "lr": lr * 2}]
    cfgs += [{**b, "name": f"{size}-base-D{tokens}-s{s}", "seed": s} for s in (1, 2)]
    cfgs += [{**B5, "name": f"{size}-B0.5-s{s}", "seed": s} for s in (1, 2)]
    cfgs += [{**b, "name": f"{size}-A8-diag-s1", "design": "A", "rank_frac": 1 / 8, "diag": True, "seed": 1}]
    return cfgs


def replan2(lr=2e-3):
    """B with D₁(R+Δ)D₂ by ρ, on all linears and under int8 noise; A 1/4 on all linears; which half of the int8 arm
    (int8 R or int8 activations) helps A 1/8; whether B's growing gap saturates past compute-optimal tokens (w128 at
    60M, ~45 tokens per parameter); and B with diagonals at w512."""
    b = {"size": "w256", "tokens": 20_000_000, "lr": lr, "threads": 3}
    Bd = {**b, "design": "B", "diag": True}
    A8 = {**b, "design": "A", "rank_frac": 1 / 8, "diag": True}
    cfgs = [{**Bd, "name": f"w256-Bd{rho:g}", "rho": rho} for rho in (0.1, 0.25, 0.5)]
    cfgs += [{**Bd, "name": f"w256-Bd{rho:g}-all", "rho": rho, "harden": "all"} for rho in (0.25, 0.5)]
    cfgs += [{**Bd, "name": f"w256-Bd0.25-int8-k{k}", "rho": 0.25, "int8": True, "noise_k": k, "r_dist": "int8"}
             for k in (4, 16)]
    cfgs += [{**b, "name": "w256-A4-diag-all", "design": "A", "rank_frac": 1 / 4, "diag": True, "harden": "all"},
             {**A8, "name": "w256-A8-diag-rint8", "r_dist": "int8"},
             {**A8, "name": "w256-A8-diag-int8-k0-rgauss", "int8": True},
             {**A8, "name": "w256-A8-diag-int8-k0-s1", "int8": True, "r_dist": "int8", "seed": 1}]
    lng = {"size": "w128", "tokens": 60_000_000, "lr": lr, "threads": 4}
    cfgs += [{**lng, "name": f"w128-base-D{t}", "tokens": t} for t in (60_000_000, 40_000_000)]
    cfgs += [{**lng, "name": "w128-B0.5-D60M", "design": "B", "rho": 0.5},
             {**lng, "name": "w128-Bd0.5-D60M", "design": "B", "rho": 0.5, "diag": True},
             {**lng, "name": "w128-A8-diag-D60M", "design": "A", "rank_frac": 1 / 8, "diag": True}]
    w5 = {"size": "w512", "tokens": 20_000_000, "lr": lr, "threads": 6, "design": "B", "diag": True}
    cfgs += [{**w5, "name": f"w512-Bd{rho:g}", "rho": rho} for rho in (0.25, 0.5)]
    return cfgs


def replan3():
    """Whether B's penalty growing with width at one lr is the lr's doing: the baseline and B ρ = 0.25 at lr ∝ 1/n
    (2e-3 at w256; the μP rule for Adam on hidden matrices), where the baseline's weights move less as width grows."""
    cfgs = []
    for size, lr, threads in (("w512", 1e-3, 6), ("w128", 4e-3, 2)):
        b = {"size": size, "tokens": 20_000_000, "lr": lr, "threads": threads}
        cfgs += [{**b, "name": f"{size}-base-lr{lr:g}"},
                 {**b, "name": f"{size}-B0.25-lr{lr:g}", "design": "B", "rho": 0.25}]
    return cfgs


def replan4():
    """The baseline's lr one step past each width's best so far (4e-3 at w128 and w256, 2e-3 at w512), so the best-lr
    penalty compares B with a baseline whose lr optimum is bracketed; and B ρ = 0.25 one step past its own highest."""
    cfgs = []
    for size, base_lr, b_lr, threads in (("w128", 8e-3, 8e-3, 2), ("w256", 8e-3, 4e-3, 2), ("w512", 4e-3, 4e-3, 4)):
        b = {"size": size, "tokens": 20_000_000, "threads": threads}
        cfgs += [{**b, "name": f"{size}-base-lr{base_lr:g}", "lr": base_lr},
                 {**b, "name": f"{size}-B0.25-lr{b_lr:g}", "lr": b_lr, "design": "B", "rho": 0.25}]
    return cfgs


def replan5():
    """Width at fixed tokens per parameter: w256 at 80M tokens has w128-at-20M's 25.4 tokens per non-embedding
    parameter, since the penalty grows with tokens per parameter (w128 at 60M) and 20M tokens puts w512 at 1.6. A lr-2e-3
    ladder at 40M and 80M (with a 4e-3 bracket), B ρ = 0.25 and A 1/8 at their best 20M-token lrs; and w128's lr bracket."""
    w = {"size": "w256", "tokens": 80_000_000, "lr": 2e-3, "threads": 4}
    cfgs = [{**w, "name": "w256-base-D80000000"},
            {**w, "name": "w256-base-D40000000", "tokens": 40_000_000},
            {**w, "name": "w256-base-D80M-lr0.004", "lr": 4e-3},
            {**w, "name": "w256-B0.25-D80M", "design": "B", "rho": 0.25},
            {**w, "name": "w256-A8-diag-D80M-lr0.004", "lr": 4e-3, "design": "A", "rank_frac": 1 / 8, "diag": True}]
    b = {"size": "w128", "tokens": 20_000_000, "lr": 1.6e-2, "threads": 2}
    cfgs += [{**b, "name": "w128-base-lr0.016"}, {**b, "name": "w128-B0.25-lr0.016", "design": "B", "rho": 0.25}]
    return cfgs


def replan6():
    """The width series at fixed tokens per non-embedding parameter, extended: w384 at 180M tokens (25.4 per parameter,
    with w128 at 20M and w256 at 80M), and w512 at 160M (12.7, with w128 at 10M and w256 at 40M). Each with a lr-2e-3
    baseline ladder around where B lands, B ρ = 0.25 at 2e-3, and the baseline at its width's best lr where that isn't
    2e-3; A 1/8 at w384."""
    cfgs = []
    for size, tokens, threads in (("w384", (45_000_000, 90_000_000, 180_000_000), 8),
                                  ("w512", (40_000_000, 80_000_000, 160_000_000), 12)):
        b = {"size": size, "lr": 2e-3, "threads": threads}
        cfgs += [{**b, "name": f"{size}-base-D{t}", "tokens": t} for t in tokens]
        cfgs += [{**b, "name": f"{size}-B0.25-D{tokens[-1] // 1_000_000}M", "tokens": tokens[-1], "design": "B",
                  "rho": 0.25}]
    w = {"size": "w384", "tokens": 180_000_000, "lr": 3e-3, "threads": 8}
    cfgs += [{**w, "name": "w384-base-D180M-lr0.003"},
             {**w, "name": "w384-A8-diag-D180M-lr0.003", "design": "A", "rank_frac": 1 / 8, "diag": True}]
    s = {"size": "w128", "tokens": 10_000_000, "threads": 2}
    cfgs += [{**s, "name": "w128-B0.25-D10M", "lr": 2e-3, "design": "B", "rho": 0.25},
             {**s, "name": "w128-base-D10M-lr0.008", "lr": 8e-3},
             {**s, "name": "w128-B0.25-D10M-lr0.008", "lr": 8e-3, "design": "B", "rho": 0.25}]
    m = {"size": "w256", "tokens": 40_000_000, "threads": 2}
    cfgs += [{**m, "name": "w256-B0.25-D40M", "lr": 2e-3, "design": "B", "rho": 0.25},
             {**m, "name": "w256-base-D40M-lr0.004", "lr": 4e-3}]
    return cfgs


def replan7():
    """The loosest B that still blocks 2:4 and rank-halving on an int8 R (ρ = 0.44 < 0.447), and B ρ = 0.25 with
    D₁(R+Δ)D₂, at the two matched points of 25.4 tokens per parameter: w128 at 20M (its best lr) and w256 at 80M."""
    cfgs = []
    for size, tokens, lr, threads, tag in (("w128", 20_000_000, 8e-3, 2, "-lr0.008"),
                                           ("w256", 80_000_000, 2e-3, 4, "-D80M")):
        b = {"size": size, "tokens": tokens, "lr": lr, "threads": threads, "design": "B", "diag": True}
        cfgs += [{**b, "name": f"{size}-Bd0.44-rint8{tag}", "rho": 0.44, "r_dist": "int8"},
                 {**b, "name": f"{size}-Bd0.25{tag}", "rho": 0.25}]
    return cfgs


def replan8():
    """B ρ = 0.25's lr bracket from below: at w256 its best tested lr (2e-3, over 4e-3) was the lowest tried, so 1e-3
    at 20M, 40M and 80M, and 4e-3 at 80M; and w384 at 180M (25.4 tokens per parameter) at 1e-3."""
    b = {"size": "w256", "lr": 1e-3, "threads": 4, "design": "B", "rho": 0.25}
    cfgs = [{**b, "name": f"w256-B0.25-D{t // 1_000_000}M-lr0.001", "tokens": t} for t in (20_000_000, 40_000_000, 80_000_000)]
    cfgs += [{**b, "name": "w256-B0.25-D80M-lr0.004", "tokens": 80_000_000, "lr": 4e-3},
             {**b, "name": "w384-B0.25-D180M-lr0.001", "size": "w384", "tokens": 180_000_000, "threads": 16}]
    return cfgs


def replan9():
    """B ρ = 0.44 on an int8 R with D₁(R+Δ)D₂ along both width series: w384 at 180M (25.4 tokens per parameter), the
    12.7 pair (w128 at 10M, w256 at 40M), and w256 80M at 4e-3, the baseline's best lr there."""
    b = {"design": "B", "diag": True, "rho": 0.44, "r_dist": "int8"}
    return [{**b, "name": "w128-Bd0.44-rint8-D10M-lr0.008", "size": "w128", "tokens": 10_000_000, "lr": 8e-3, "threads": 2},
            {**b, "name": "w256-Bd0.44-rint8-D40M", "size": "w256", "tokens": 40_000_000, "lr": 2e-3, "threads": 4},
            {**b, "name": "w256-Bd0.44-rint8-D80M-lr0.004", "size": "w256", "tokens": 80_000_000, "lr": 4e-3, "threads": 4},
            {**b, "name": "w384-Bd0.44-rint8-D180M", "size": "w384", "tokens": 180_000_000, "lr": 2e-3, "threads": 12}]


def replan10():
    """B's lr optimum falls as runs lengthen (w256 B ρ = 0.25: best at 2e-3 at 20M and 40M, at 1e-3, the lowest run,
    at 80M), so one step lower at the 25.4 points: B ρ = 0.25 at 5e-4 and B ρ = 0.44 (int8 R, diagonals) at 1e-3."""
    b = {"design": "B", "rho": 0.25}
    d = {"design": "B", "diag": True, "rho": 0.44, "r_dist": "int8", "lr": 1e-3}
    return [{**b, "name": "w256-B0.25-D80M-lr0.0005", "size": "w256", "tokens": 80_000_000, "lr": 5e-4, "threads": 4},
            {**b, "name": "w384-B0.25-D180M-lr0.0005", "size": "w384", "tokens": 180_000_000, "lr": 5e-4, "threads": 16},
            {**d, "name": "w256-Bd0.44-rint8-D80M-lr0.001", "size": "w256", "tokens": 80_000_000, "threads": 4},
            {**d, "name": "w384-Bd0.44-rint8-D180M-lr0.001", "size": "w384", "tokens": 180_000_000, "threads": 12}]


def replan11():
    """A (frozen R plus rank-r U·V, with D₁RD₂) along the 25.4 series with its lr tuned: w128 20M at 4e-3, 8e-3 and
    1.6e-2 (it ran at 2e-3 only), w256 80M at 2e-3 and 8e-3 around its 4e-3, A 1/4 at w256 80M, and w384 180M at 2e-3
    beside its 3e-3."""
    a = {"design": "A", "rank_frac": 1 / 8, "diag": True}
    cfgs = [{**a, "name": f"w128-A8-diag-lr{lr:g}", "size": "w128", "tokens": 20_000_000, "lr": lr, "threads": 2}
            for lr in (4e-3, 8e-3, 1.6e-2)]
    cfgs += [{**a, "name": f"w256-A8-diag-D80M-lr{lr:g}", "size": "w256", "tokens": 80_000_000, "lr": lr, "threads": 4}
             for lr in (2e-3, 8e-3)]
    cfgs += [{**a, "name": "w256-A4-diag-D80M-lr0.004", "size": "w256", "tokens": 80_000_000, "lr": 4e-3, "threads": 4,
              "rank_frac": 1 / 4},
             {**a, "name": "w384-A8-diag-D180M-lr0.002", "size": "w384", "tokens": 180_000_000, "lr": 2e-3, "threads": 12}]
    return cfgs


def replan12():
    """A's lr optimum sits at 2× the baseline's or more (w128 20M: 2.259, 2.241 and 2.224 at 4e-3, 8e-3 and 1.6e-2),
    so the high side: w128 20M at 3.2e-2 and w384 180M at 6e-3; and A's first 12.7 pair, w128 10M and w256 40M, at
    2× each baseline's best lr."""
    a = {"design": "A", "rank_frac": 1 / 8, "diag": True}
    return [{**a, "name": "w128-A8-diag-lr0.032", "size": "w128", "tokens": 20_000_000, "lr": 3.2e-2, "threads": 2},
            {**a, "name": "w384-A8-diag-D180M-lr0.006", "size": "w384", "tokens": 180_000_000, "lr": 6e-3, "threads": 12},
            {**a, "name": "w128-A8-diag-D10M-lr0.016", "size": "w128", "tokens": 10_000_000, "lr": 1.6e-2, "threads": 2},
            {**a, "name": "w256-A8-diag-D40M-lr0.008", "size": "w256", "tokens": 40_000_000, "lr": 8e-3, "threads": 4}]


def replan13():
    """B's lr optimum is still at the lowest lr run at w384 (B ρ = 0.25: 1.653, 1.624, 1.603 at 2e-3, 1e-3, 5e-4) and
    at w256 for ρ = 0.44 (1.741, 1.719 at 2e-3, 1e-3), so one step lower; the w384 baseline's lr from below; and A 1/4
    at w384 180M, beside its w256 80M point."""
    b = {"design": "B", "tokens": 180_000_000, "size": "w384", "threads": 16}
    d = {"design": "B", "diag": True, "rho": 0.44, "r_dist": "int8", "lr": 5e-4}
    return [{**b, "name": "w384-B0.25-D180M-lr0.00025", "rho": 0.25, "lr": 2.5e-4},
            {**b, **d, "name": "w384-Bd0.44-rint8-D180M-lr0.0005"},
            {**d, "name": "w256-Bd0.44-rint8-D80M-lr0.0005", "size": "w256", "tokens": 80_000_000, "threads": 4},
            {"name": "w384-base-D180M-lr0.001", "size": "w384", "tokens": 180_000_000, "lr": 1e-3, "threads": 12},
            {"name": "w384-A4-diag-D180M-lr0.003", "size": "w384", "tokens": 180_000_000, "lr": 3e-3, "threads": 16,
             "design": "A", "rank_frac": 1 / 4, "diag": True}]


def replan14():
    """A 1/8's lr at w384 180M from below: 2e-3 beat 3e-3 (1.559 against 1.566) and 6e-3 diverged, and A's optimum
    relative to the baseline's falls with width (2× at w128, 1× at w256)."""
    return [{"name": "w384-A8-diag-D180M-lr0.001", "size": "w384", "tokens": 180_000_000, "lr": 1e-3, "threads": 24,
             "design": "A", "rank_frac": 1 / 8, "diag": True}]


A8 = {"design": "A", "rank_frac": 1 / 8, "diag": True}
BD44 = {"design": "B", "diag": True, "rho": 0.44, "r_dist": "int8"}


def fw1():
    """FineWeb-Edu (prep.py --source fineweb-edu), w128 and w256 at 25.4 tokens per parameter, to tell TinyStories'
    floor from the constraint: the baseline ladders at 2e-3 with the lrs TinyStories found best at the matched points,
    and A 1/8, A 1/4, B ρ = 0.44 (int8 R, diagonals) and B ρ = 0.25 at their TinyStories-best lr and a neighbour.
    The same names as the TinyStories runs; analyze each corpus's dirs on their own."""
    w2 = {"size": "w256", "tokens": 80_000_000, "threads": 4}
    w1 = {"size": "w128", "tokens": 20_000_000, "threads": 2}
    cfgs = [{"name": "w256-base-D160000000", "size": "w256", "tokens": 160_000_000, "lr": 2e-3, "threads": 8}]
    cfgs += [{**w2, "name": "w256-base-D80000000", "lr": 2e-3}, {**w2, "name": "w256-base-D80M-lr0.004", "lr": 4e-3}]
    cfgs += [{**w2, **A8, "name": f"w256-A8-diag-D80M-lr{lr:g}", "lr": lr} for lr in (4e-3, 2e-3)]
    cfgs += [{**w2, **A8, "name": "w256-A4-diag-D80M-lr0.004", "lr": 4e-3, "rank_frac": 1 / 4}]
    cfgs += [{**w2, **BD44, "name": "w256-Bd0.44-rint8-D80M-lr0.001", "lr": 1e-3},
             {**w2, **BD44, "name": "w256-Bd0.44-rint8-D80M", "lr": 2e-3},
             {**w2, "name": "w256-B0.25-D80M-lr0.001", "lr": 1e-3, "design": "B", "rho": 0.25}]
    cfgs += [{**w2, "name": f"w256-base-D{t}", "tokens": t, "lr": 2e-3} for t in (40_000_000, 20_000_000, 10_000_000)]
    cfgs += [{**w1, "name": f"w128-base-D{t}", "tokens": t, "lr": 2e-3}
             for t in (80_000_000, 40_000_000, 20_000_000, 10_000_000, 5_000_000, 2_500_000)]
    cfgs += [{**w1, "name": f"w128-base-lr{lr:g}", "lr": lr} for lr in (4e-3, 8e-3)]
    cfgs += [{**w1, **A8, "name": f"w128-A8-diag-lr{lr:g}", "lr": lr} for lr in (1.6e-2, 8e-3)]
    cfgs += [{**w1, **A8, "name": "w128-A4-diag-lr0.008", "lr": 8e-3, "rank_frac": 1 / 4}]
    cfgs += [{**w1, **BD44, "name": f"w128-Bd0.44-rint8-lr{lr:g}", "lr": lr} for lr in (8e-3, 4e-3)]
    cfgs += [{**w1, "name": "w128-B0.25-lr0.008", "lr": 8e-3, "design": "B", "rho": 0.25}]
    return cfgs


def fw2():
    """FineWeb-Edu at w384, 25.4 tokens per parameter (180M): the baseline ladder at 2e-3 to 360M, and A 1/8 and
    B ρ = 0.44 at their TinyStories-best lr, the third width beside fw1's two."""
    b = {"size": "w384", "lr": 2e-3}
    cfgs = [{**b, "name": "w384-base-D360000000", "tokens": 360_000_000, "threads": 24},
            {**b, "name": "w384-base-D180000000", "tokens": 180_000_000, "threads": 12},
            {**b, **A8, "name": "w384-A8-diag-D180M-lr0.002", "tokens": 180_000_000, "threads": 16},
            {**b, **BD44, "name": "w384-Bd0.44-rint8-D180M-lr0.001", "tokens": 180_000_000, "lr": 1e-3, "threads": 16}]
    cfgs += [{**b, "name": f"w384-base-D{t}", "tokens": t, "threads": 8} for t in (90_000_000, 45_000_000)]
    return cfgs


PRESETS = {
    "bench": bench,
    "replan1": replan1,
    "replan2": replan2,
    "replan3": replan3,
    "replan4": replan4,
    "replan5": replan5,
    "replan6": replan6,
    "replan7": replan7,
    "replan8": replan8,
    "replan9": replan9,
    "replan10": replan10,
    "replan11": replan11,
    "replan12": replan12,
    "replan13": replan13,
    "replan14": replan14,
    "fw1": fw1,
    "fw2": fw2,
    "w256": lambda: grid("w256", 20_000_000),
    "scale1": lambda: lite("w512", 20_000_000, 6) + lite("w128", 20_000_000, 2),
}


def run_one(cfg, data, out, threads):
    res = out / "runs" / f"{cfg['name']}.json"
    log = out / "logs" / f"{cfg['name']}.log"
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    cfg = {**cfg, "threads": threads}
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, str(HERE / "train.py"), "--config", json.dumps(cfg), "--data", str(data),
                            "--out", str(res)], stdout=fh, stderr=subprocess.STDOUT, env=env)
    row = {"name": cfg["name"], "rc": p.returncode, "wall_s": time.time() - t0}
    if p.returncode == 0:
        r = json.loads(res.read_text())
        row.update(final_val=r["final_val"], tok_per_s=r["tok_per_s"])
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.environ.get("OUT", "out"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None, help="override every config's thread count")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--resume-from", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    cfgs = PRESETS[a.preset]()
    if a.only:
        cfgs = [c for c in cfgs if c["name"] in a.only]
    todo, rows = [], []
    for c in cfgs:
        prev = [Path(d) / "runs" / f"{c['name']}.json" for d in a.resume_from]
        prev = [p for p in prev if p.exists()]
        if prev:
            shutil.copy(prev[0], out / "runs" / prev[0].name)
            rows.append({"name": c["name"], "rc": 0, "copied_from": str(prev[0])})
        else:
            todo.append(c)
    print(f"sweep {a.preset}: {len(todo)} to run, {len(rows)} copied, workers {a.workers}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_one, c, a.data, out, a.threads or c.get("threads", 4)) for c in todo]
        rows += [f.result() for f in futs]
    (out / "sweep.json").write_text(json.dumps({"preset": a.preset, "rows": rows}, indent=1))
    bad = [r for r in rows if r["rc"] != 0]
    print(f"sweep {a.preset}: {len(rows) - len(bad)} ok, {len(bad)} failed", flush=True)


if __name__ == "__main__":
    main()
