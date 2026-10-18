"""PRF-70 (1): the attention unit's ANDs per key with and without the prologue (q's and k's per-head RMSNorm and RoPE
recomputed in the unit, so rope_q and kv_k leave the Commit), and the sampling optimizer's plan for Qwen3-8B bf16 at
proofs' budgets with and without it.

    uv run python benchmarks/hash/bytes/attention_prologue.py            # the AND table and the plans at proofs' budgets
    uv run python benchmarks/hash/bytes/attention_prologue.py --json
    uv run python benchmarks/hash/bytes/attention_prologue.py --check    # the AND counts recomputed by circuit-check's counter

A served attention unit outputs one bf16 element o[t, h, d] (the 16-bit rule), and the Commit holds no scores, so the unit
proves its head's whole Q·Kᵀ row, the softmax and one P·V column: per key, 8 tensor-core k16 steps of Q·Kᵀ, the softmax's
share and 1/16 of a step of P·V, plus opening the key's K and V leaves. The prologue adds, per key, the k head's RMSNorm
(N = 128) and RoPE (128 outputs), and, once a unit, the same for its q head. The counts are the repo's: `compile_bits` of
vLLM's Boolean RMSNorm (`RMSNormCudaVector8_v2{N=128, EPS=1e-6, BLOCK=16}`), `RoPEHead_v2{D=128}` and the FA2 head
`AttentionHead_v6{T, D=128, BN=64}`, whose ANDs are exactly linear in T.

Three variants go through `verity.core.protocols.sampled_proofs` with the int8 plan's fitted cost model and inputs
(`benchmarks/sampling_plan`), imported, not copied:
- `baseline`: rope_q and kv_k committed by per-element norm+RoPE units (zero-work templates, at their floor);
- `prologue`: no rope units; each attention unit recomputes k's norm and RoPE for every key it reads, and q's once;
- `kv-credited`: memory accounting's KV tree commits post-RoPE K, so the attention unit reads K from it and recomputes only
  q; per-element link units prove cache K = RoPE(norm(k)) against the tree (zero-work, at their floor, nothing committed).
In both changed variants rope_q and kv_k (10,240 B a token a layer) leave the Commit. The RoPE tables are registered tables
(lookup rows, no ANDs), like the other gathers.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "sampling_plan"))

import int8_llama as IL8  # noqa: E402
from inputs import I  # noqa: E402

from verity.core.protocols.sampled_proofs.candidates import Shape, cuts  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import Objective, for_budget, frontier, row_ands  # noqa: E402

# --- AND counts (`--check` recomputes them) ------------------------------------------------------------------------------

#: `compile_bits(RMSNormCudaVector8_v2{N=128,EPS=1e-06,BLOCK=16,RSQRT=RsqrtApprox_v2}).counts["and"]`: one head's q_norm/k_norm
RMSNORM_HEAD = 557_598
#: `compile_bits(RoPEHead_v2{D=128})`: 128 outputs of `RopeOut_v2` / `RopeOutAdd_v2` at 2,966 each (boolean_rope pins)
ROPE_HEAD = 379_648
ROPE_ELEMENT = 2_966
#: `compile_bits(AttentionHead_v6{T,D=128,BN=64})` at T = 64, 128, 192: 9,228,119 / 18,467,283 / 27,706,447, linear a 64-key
#: block (BN), so per key
ATTN_HEAD_PER_BLOCK = 18_467_283 - 9_228_119
ATTN_HEAD_PER_KEY = ATTN_HEAD_PER_BLOCK / 64
#: `AmpereBF16TcDot16_v2`, the FA2 k16 step (prims pin): Q·Kᵀ is 8 a key, P·V 8 a key for the head's 128 columns
TC_STEP = 8_623
D, HEADS, KV_HEADS, LAYERS = 128, 32, 8, 36
QK_PER_KEY = (D // 16) * TC_STEP
SOFTMAX_PER_KEY = ATTN_HEAD_PER_KEY - 2 * QK_PER_KEY
#: one output element's unit: the whole Q·Kᵀ row and softmax, one P·V column
ELEMENT_PER_KEY = round(QK_PER_KEY + SOFTMAX_PER_KEY + TC_STEP / 16)
PROLOGUE = RMSNORM_HEAD + ROPE_HEAD
LEAF = 4096
ROPE_BYTES = (HEADS + KV_HEADS) * D * 2               # rope_q 8,192 + kv_k 2,048 a token a layer


def per_key(variant: str, leaves: int) -> dict[str, int]:
    """ANDs a key of one attention element unit: compute, the prologue's share, and opening `leaves` 4 KiB leaves."""
    k = PROLOGUE if variant == "prologue" else 0
    opening = leaves * row_ands(LEAF)
    return {"compute": ELEMENT_PER_KEY, "prologue": k, "opening": opening, "total": ELEMENT_PER_KEY + k + opening}


def unit_ands(variant: str, length: int, leaves: int) -> int:
    """One element unit over `length` keys: per key, plus the q head's norm and RoPE once (both changed variants)."""
    once = 0 if variant == "baseline" else PROLOGUE
    return length * (per_key(variant, leaves)["total"] - per_key(variant, leaves)["opening"]) + once


# --- the optimizer -------------------------------------------------------------------------------------------------------

#: Qwen3-8B's linears: K -> output coordinates a token (q 4096, k 1024, v 1024, o 4096, gate 12288, up 12288 at K = 4096; down)
LINEARS = {4096: 4096 + 1024 + 1024 + 4096 + 12288 + 12288, 12288: 4096}
#: Qwen3-8B B32 on one RTX PRO 6000, stock serve (r20261018-065034-aa8f): 4 prefill steps of 2,560 tokens at 117 ms and 64
#: decode steps of 32 at 15 ms, 12,288 tokens in 1,428 ms
TOKENS_PER_S = 12_288 / 1.428
VARIANTS = ("baseline", "prologue", "kv-credited")


def bf16_coord(k: int) -> Shape:
    ands = (k // 16) * TC_STEP + 77
    rows = (2 * k, 2 * k, 2)
    return Shape(f"Bf16Coord{{K={k}}}", ands, k, 16, 1, rows, 0, 0.0,
                 cap=min(IL8.cap(ands + sum(row_ands(r) for r in rows)), 1 << 20))


def attention(length: int, variant: str, leaves: int) -> Shape:
    """One output element over `length` keys: q's leaf once, then `leaves` leaves a key (K and V; in the KV store one)."""
    ands = unit_ands(variant, length, leaves)
    rows = (LEAF,) + (LEAF,) * (leaves * length)
    return Shape(f"AttnElement{{L={length},{variant},leaves={leaves}}}", ands, 2 * length, 16, 1, rows, 0, 0.0,
                 cap=min(IL8.cap(ands + sum(row_ands(r) for r in rows)), 1 << 20))


def workload(v: dict, variant: str, leaves: int, window_rounds: float) -> dict:
    tokens = v["tokens_per_s"] * v["round"] * window_rounds
    step = int(v["step_tokens"])
    replay_units = max(1, round(tokens / step)) * LAYERS
    ops = {f"linear K={k}": (bf16_coord(k), step * coords) for k, coords in LINEARS.items()}
    for length, share in IL8.classes(int(v["context"])).items():
        per_step = round(step * share) * HEADS * D
        if per_step:
            ops[f"attention L={length}"] = (attention(length, variant, leaves), per_step)
    work = sum(root.work * per for root, per in ops.values()) * replay_units
    return {"ops": ops, "replay_units": replay_units, "tokens": tokens, "work": work,
            "window_gpu_s": v["round"] * window_rounds}


def rope_commit_b(v: dict) -> float:
    """B of committing rope_q and kv_k at serve time, at the plan's commitment speed: GPU-s a GPU-s."""
    return v["commit_per_byte"] * ROPE_BYTES * LAYERS * v["tokens_per_s"]


def plans(variant: str, leaves: int, *, commit_scale: float = 1.0, grid: int = 41) -> dict:
    v = {k: i.value for k, i in I.items()}
    v["tokens_per_s"] = TOKENS_PER_S
    v["zero_work_templates"] = {"baseline": 30, "prologue": 28, "kv-credited": 29}[variant]
    fit = IL8.fitted(v)
    w = workload(v, variant, leaves, IL8.WINDOWS["week"])
    costs = IL8.model(v, fit, w["replay_units"])
    objective = Objective(IL8.DELTA)
    cands = []
    for op, (root, per) in w["ops"].items():
        cs, _alts, _hints = cuts(op, root, per * w["replay_units"], costs, objective, grid=21,
                                 replay=("layer-step", w["replay_units"]))
        cands += cs
    front = frontier(cands, costs, objective, grid=grid, exact=False)
    extra = IL8.zero_work_cost(costs, v) + (rope_commit_b(v) * commit_scale * w["window_gpu_s"]
                                           if variant == "baseline" else 0.0)
    out = {}
    for b in IL8.BUDGETS:
        p = for_budget(front, (b - v["commit_floor"]) * w["window_gpu_s"] - extra, candidates=cands, costs=costs,
                       objective=objective, exact=False)
        if p is None:
            out[b] = None
            continue
        attn = {t: k for t, k in p.sizes.items() if t.startswith("attention")}
        draws = sum(p.sizes.values())
        out[b] = {"eps": p.certificate.harm / w["work"], "draws": draws, "attention_draws": sum(attn.values()),
                  "statements_b": p.cost["statements"] / w["window_gpu_s"], "units_b": p.cost["units"] / w["window_gpu_s"]}
    return {"variant": variant, "leaves": leaves, "commit_scale": commit_scale, "extra_b": extra / w["window_gpu_s"],
            "rope_commit_b": rope_commit_b(v) * commit_scale, "tokens_per_s": v["tokens_per_s"], "at_budget": out}


def ands_table() -> list[dict]:
    rows = []
    for variant in VARIANTS:
        for leaves in (2, 1):
            pk = per_key(variant, leaves)
            rows.append({"variant": variant, "leaves": leaves, **pk,
                         "unit_L1024": unit_ands(variant, 1024, leaves) + (1 + leaves * 1024) * row_ands(LEAF)})
    return rows


def check() -> list[str]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.bits import compile as compile_bits
    from verity.ml.circuits.ops.boolean import mufu as BM
    from verity_vllm.program.registry import boolean_attention as BA
    from verity_vllm.program.registry import boolean_norms as BN
    from verity_vllm.program.registry import boolean_rope as BR

    def ands(d) -> int:
        return compile_bits(d).counts["and"]

    bad = []
    got = ands(bind(BN.RMSNormCudaVector8, N=128, EPS=1e-6, BLOCK=16, RSQRT=bind(BM.RsqrtApprox)))
    if got != RMSNORM_HEAD:
        bad.append(f"RMSNormCudaVector8_v2{{N=128}}: {got} ANDs, RMSNORM_HEAD says {RMSNORM_HEAD}")
    if (got := ands(bind(BR.RoPEHead, D=128))) != ROPE_HEAD or ROPE_HEAD != D * ROPE_ELEMENT:
        bad.append(f"RoPEHead_v2{{D=128}}: {got} ANDs, ROPE_HEAD says {ROPE_HEAD} (= 128 x {ROPE_ELEMENT})")
    heads = [ands(bind(BA.AttentionHeadV6, T=t, D=128, BN=64)) for t in (64, 128, 192)]
    if heads[1] - heads[0] != ATTN_HEAD_PER_BLOCK or heads[2] - heads[1] != ATTN_HEAD_PER_BLOCK:
        bad.append(f"AttentionHead_v6{{T,D=128,BN=64}} at T = 64/128/192: {heads}, not {ATTN_HEAD_PER_BLOCK} a 64-key block")
    return bad


def render(table: list[dict], runs: list[dict]) -> str:
    out = ["| variant | 4 KiB leaves a key | compute a key | prologue a key | opening a key | total a key | unit at L = 1,024 |",
           "|---|---|---|---|---|---|---|"]
    for r in table:
        out.append(f"| {r['variant']} | {r['leaves']} | {r['compute']:,} | {r['prologue']:,} | {r['opening']:,} | "
                   f"{r['total']:,} | {r['unit_L1024'] / 1e9:.2f}G |")
    out += ["", "| run | B: " + " | ".join(str(b) for b in IL8.BUDGETS) + " |", "|---|" + "---|" * len(IL8.BUDGETS)]
    for r in runs:
        cells = ["-" if x is None else f"{x['eps']:.3%}" for x in r["at_budget"].values()]
        out.append(f"| {r['variant']}, {r['leaves']} leaves, commit x{r['commit_scale']:.2f} | " + " | ".join(cells) + " |")
    out += ["", "| run | at B = 0.1: draws, attention draws, statements B | at B = 1: draws, attention draws, statements B |",
            "|---|---|---|"]
    for r in runs:
        cells = ["-" if (x := r["at_budget"].get(b)) is None else
                 f"{x['draws']:,}, {x['attention_draws']:,}, {x['statements_b']:.3f}" for b in (0.1, 1.0)]
        out.append(f"| {r['variant']}, {r['leaves']} leaves, commit x{r['commit_scale']:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        bad = check()
        print("\n".join(bad) if bad else "ok: RMSNorm, RoPE and the FA2 head's ANDs reproduce")
        return 1 if bad else 0
    table = ands_table()
    runs = [plans(var, leaves, commit_scale=s) for leaves in (2, 1) for var in VARIANTS
            for s in ((1.0, HASH_SCALE) if var == "baseline" else (1.0,))]
    if a.json:
        print(json.dumps({"ands": table, "constants": {"RMSNORM_HEAD": RMSNORM_HEAD, "ROPE_HEAD": ROPE_HEAD,
                                                         "ATTN_HEAD_PER_KEY": ATTN_HEAD_PER_KEY, "TC_STEP": TC_STEP,
                                                         "ELEMENT_PER_KEY": ELEMENT_PER_KEY, "row_ands_4k": row_ands(LEAF)},
                          "runs": runs}, indent=1, default=str))
        return 0
    print(render(table, runs))
    return 0


#: The hash lane's price of rope_q + kv_k on RTX PRO 6000 (bytes_model.py's qk-rope row: 0.39 points of serve) over the
#: plan's (`rope_commit_b`), for a second baseline run at the served commit's modelled speed.
HASH_RTX_B = 0.0039
HASH_SCALE = HASH_RTX_B / rope_commit_b({**{k: i.value for k, i in I.items()}, "tokens_per_s": TOKENS_PER_S})

if __name__ == "__main__":
    sys.exit(main())
