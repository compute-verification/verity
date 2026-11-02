"""The covert speedrun's run list and its cost, before any run: per size, every (start, class, configuration, learning rate),
with its steps, tokens and analytic FLOPs under the budget, and its GPU-seconds estimated from the first curves' measured dense
throughput on one RTX PRO 6000.

    python plan.py [--sizes m,s,l] [--budget-tpp 2] [--json FILE]
    python plan.py --chain m 0.25 lora 1e6      # the shell loop one `research run` runs: that configuration at its 3 rates
    python plan.py --chain l 0.5 adamw 0 --rates 5e-4    # ... at the given rates only

The estimate takes each class's achieved FLOP/s as dense AdamW's times a factor (`EFFICIENCY`: LoRA's small factor matmuls and
Muon's Newton-Schulz run below the dense matmuls' rate), plus one held-out evaluation at each of the 21 marks at three times the
training throughput. The witness records the measured GPU-seconds; this is what the run list costs before them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from model import CTX, SIZES, counts
from speedrun import ICL_DEFAULT, forward_flops_per_token, lora_flops_per_token, lora_plan, lora_update_values, ns_flops

#: first curves (r20261029-*): dense AdamW tokens per second on one RTX PRO 6000 at batch 64 x 1024
DENSE_TOK_S = {"s": 373e3, "m": 245e3, "l": 143e3}
EFFICIENCY = {"adamw": 1.0, "muon": 0.9, "lora": 0.75, "mezo": 0.9}
STARTS = (0.25, 0.5, 1.0)
#: (method, update bytes, learning rates): S0's rates span the starts' own schedule (1.5e-3 peak, 1.5e-4 at its end); S1's sit
#: a step higher, since a rank-limited update moves θ less at the same rate
CONFIGS = (("adamw", 0.0, (1.5e-4, 5e-4, 1.5e-3)), ("muon", 0.0, (1.5e-4, 5e-4, 1.5e-3)),
           ("lora", 1e5, (5e-4, 1.5e-3, 5e-3)), ("lora", 1e6, (5e-4, 1.5e-3, 5e-3)),
           ("lora", 1e7, (5e-4, 1.5e-3, 5e-3)), ("lora", 1e8, (5e-4, 1.5e-3, 5e-3)), ("icl", 0.0, ()))
EVALS = 20
BATCH = 64
#: sequences per forward pass where the whole batch doesn't fit one GPU (first curves: l at 32)
MICRO = {"l": 32}


def label(method: str, update_bytes: float) -> str:
    return f"lora-{update_bytes:.0e}".replace("+0", "") if method == "lora" else method


def cost(size_name: str, method: str, update_bytes: float, budget_tpp: float) -> dict:
    size = SIZES[size_name]
    c = counts(size)
    budget = budget_tpp * c["non_embedding"] * c["dense_flops_per_token"]
    tokens_per_step = BATCH * CTX
    opt, plan = 0.0, None
    if method in ("adamw", "muon"):
        fpt = c["dense_flops_per_token"]
        if method == "muon":
            opt = sum(ns_flops(s, 5) for s in matrix_list(size))
    elif method == "lora":
        plan = lora_plan(size, update_bytes)
        fpt = lora_flops_per_token(size, plan)
    elif method == "mezo":
        fpt = 2 * forward_flops_per_token(size)
    else:
        fpt = 0.0
    fwd_tok_s = 3 * DENSE_TOK_S[size_name]
    val_targets = 2_097_152
    if method == "icl":
        n_evals = len(ICL_DEFAULT.split(","))
        return {"size": size_name, "method": method, "label": label(method, update_bytes), "steps": 0, "tokens": 0, "flops": 0.0,
                "budget_flops": budget, "gpu_s": n_evals * 2 * val_targets / fwd_tok_s + 60, "update_bits": 0, "plan": None}
    per_step = fpt * tokens_per_step + opt
    steps = int(budget // per_step)
    achieved = c["dense_flops_per_token"] * DENSE_TOK_S[size_name] * EFFICIENCY[method]
    train_s = steps * per_step / achieved
    gpu_s = train_s + (EVALS + 1) * val_targets / fwd_tok_s + 60
    bits = 32 * lora_update_values(size, plan) if plan else (32 * steps + 64 if method == "mezo" else 32 * c["total"])
    return {"size": size_name, "method": method, "label": label(method, update_bytes), "steps": steps,
            "tokens": steps * tokens_per_step, "flops": steps * per_step, "budget_flops": budget, "gpu_s": gpu_s,
            "update_bits": bits, "plan": plan}


def matrix_list(size) -> list[tuple[int, int]]:
    from model import matrix_shapes  # noqa: PLC0415
    return [s for _ in range(size.n_layer) for s in matrix_shapes(size)[:7]]


def chain(size: str, fraction: float, method: str, update_bytes: float, rates: tuple[float, ...] = (), seed: int = 0) -> str:
    """The workload of one `research run`: the configuration at each of its rates (`rates`, else its 3), one after another,
    each a chain of leases. A nonzero `seed` sets both speedrun.py seeds (initialization and data order)."""
    rates = rates or next((r for m, b, r in CONFIGS if m == method and b == update_bytes), ())
    tag = f"{size}-p{round(100 * fraction):03d}-{label(method, update_bytes)}" + (f"-seed{seed}" if seed else "")
    extra = (f" --update-bytes {update_bytes:g}" if method == "lora" else "") + (f" --micro {MICRO[size]}" if size in MICRO else "")
    extra += f" --seed {seed} --data-seed {seed}" if seed else ""
    one = ('LEASED_SCRIPT=speedrun.py bash benchmarks/learning_frontier/leased.sh "$P" --data "$D" --start "$S" '
           '--start-sha256 "$H" --ckpt-root "$C" --method ' + method + extra)
    if method == "icl":
        return f'{one} --out "$OUT/{tag}"'
    loop = " ".join(f"{r:g}" for r in rates)
    return f'rc=0; for lr in {loop}; do {one} --lr $lr --out "$OUT/{tag}-lr$lr" || rc=$?; done; exit $rc'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="m,s,l")
    ap.add_argument("--budget-tpp", type=float, default=2.0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--chain", nargs=4, metavar=("SIZE", "FRACTION", "METHOD", "UPDATE_BYTES"), default=None)
    ap.add_argument("--rates", default="", help="with --chain: these learning rates instead of the configuration's 3 (e.g. "
                                                "the best at a smaller size)")
    ap.add_argument("--seed", type=int, default=0, help="with --chain: a further seed of the configuration")
    a = ap.parse_args(argv)
    if a.chain:
        rates = tuple(float(r) for r in a.rates.split(",") if r)
        print(chain(a.chain[0], float(a.chain[1]), a.chain[2], float(a.chain[3]), rates, a.seed))
        return 0
    rows, total = [], {}
    print("| size | class | config | runs | steps | tokens (M) | FLOPs/run | update | est. GPU-min/run | est. GPU-h |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for size in a.sizes.split(","):
        for method, ub, rates in CONFIGS:
            r = cost(size, method, ub, a.budget_tpp)
            n = len(STARTS) * max(1, len(rates))
            r |= {"runs": n, "rates": list(rates), "gpu_h": n * r["gpu_s"] / 3600}
            rows.append(r)
            total[size] = total.get(size, 0.0) + r["gpu_h"]
            upd = f"{r['update_bits'] / 8 / 1e6:.3g} MB" if r["update_bits"] else "-"
            print(f"| {size} | {'S3' if method == 'icl' else 'S1' if method == 'lora' else 'S0'} | {r['label']} | {n} | {r['steps']} "
                  f"| {r['tokens'] / 1e6:.0f} | {r['flops']:.3g} | {upd} | {r['gpu_s'] / 60:.1f} | {r['gpu_h']:.2f} |")
    for size, h in total.items():
        print(f"{size}: {h:.1f} GPU-h for {sum(r['runs'] for r in rows if r['size'] == size)} runs")
    if a.json:
        a.json.write_text(json.dumps({"schema": "covert-speedrun/plan/v1", "budget_tpp": a.budget_tpp, "rows": rows,
                                      "gpu_h": total}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
