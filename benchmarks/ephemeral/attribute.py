"""Split a decode step's HBM bytes into registered and unregistered, from Nsight Compute's DRAM counters.

Each arm is OUT/<tag>.json (`steps.py`) with OUT/ncu-<tag>.csv (`ncu --page raw --print-units base --csv`, metrics
``dram__bytes_op_read.sum`` and ``dram__bytes_op_write.sum``), profiled per kernel (``--graph-profiling node``) or per CUDA
graph (``--graph-profiling graph``).  A window per point opens with ``spin_kernel``.

What the design registers (`docs/remote-bandwidth-exhaustion.md` §2): every FP8 linear's weight, read once a step, and the
KV cache, read once a step for every live token.  The rest of the reads are unregistered: the lm_head while it stays plain
(F20), the embedding rows, activations, the residual stream, workspaces and the sampler's logits.  So, per step,

    unregistered = read - linear_bytes - kv_read,     kv_read = sum over sequences of tokens x layers x KV bytes a token-layer

with ``kv_read`` taken analytically and checked against the attention kernels' own reads (node profiles), and the lm_head
reported both plain (unregistered) and encoded (registered).  A node profile also gives each kernel class's bytes; a run at
reduced depth (``steps.py --layers``) scales its per-layer bytes to the checkpoint's depth, keeping the per-step kernels
(lm_head, sampler, embedding, input preparation) once.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

READ, WRITE, NAME = "dram__bytes_op_read.sum", "dram__bytes_op_write.sum", "Kernel Name"
FULL_LAYERS = {8192: 80, 4096: 32}

CLASSES = [
    ("marker", r"spin_kernel"),
    ("graph", r"^graph$"),
    ("kv_write", r"reshape_and_cache|concat_and_cache"),
    ("attention", r"kernel_mha|xqa|flashinfer|BatchDecode|BatchPrefill|[Aa]ttention|flash_fwd|fmha|paged_attention|merge_state|"
                  r"VariableLength|unified_attention|kernel_paged|decode_kernel|split_kv"),
    ("gemm", r"gemm|gemv|cutlass|cublas|nvjet|Kernel2|matmul|_mm_|sm90_|sm100_|sm120_|xmma|wgrad|dgrad"),
    ("norm", r"rms_norm|rmsnorm|layer_norm|RMSNorm"),
    ("quant", r"quant|fp8|scaled_fp8"),
    ("act", r"silu|act_and_mul|gelu|swiglu"),
    ("rope", r"rotary|rope"),
    ("sampler", r"argmax|topk|top_k|sort|sampl|softmax|logsumexp|log_softmax|max_kernel|reduce_kernel|"
                r"gumbel|exponential|penalt"),
    ("embed", r"embedding|index_select|indexSelect|gather"),
    ("triton", r"triton_"),
]
LAYERED = ("graph", "gemm", "attention", "kv_write", "norm", "quant", "act", "rope", "triton")


def classify(name: str, read: float, lm_head_bytes: int) -> str:
    cls = next((c for c, pat in CLASSES if re.search(pat, name)), "other")
    if cls in ("gemm", "other") and read >= 0.9 * lm_head_bytes:
        return "lm_head"
    return cls


def rows(path: Path) -> list[dict]:
    lines = [ln for ln in path.read_text().splitlines() if ln.startswith('"')]
    r = list(csv.reader(lines))
    head, body = r[0], r[2:]
    if READ not in head or WRITE not in head:
        raise SystemExit(f"{path.name}: no {READ} / {WRITE} columns")
    out = []
    for row in body:
        d = dict(zip(head, row))
        name = d.get(NAME, "")
        out.append({"name": name, "marker": any("spin_kernel" in v for v in row),
                    "read": float(d[READ].replace(",", "") or 0), "write": float(d[WRITE].replace(",", "") or 0)})
    return out


def windows(rs: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    for r in rs:
        if r["marker"]:
            out.append([])
        elif out:
            out[-1].append(r)
    return out


def kv_read(budget: dict, batch: int, tokens: float) -> float:
    return batch * tokens * budget["layers"] * budget["kv_bytes_per_token_layer"]


def split(arm: dict, pt: dict, w: list[dict]) -> dict:
    """One profiled point's bytes a step.  The lm_head runs outside vLLM's decode graph, once a step, so its kernels
    count the steps whether the window was profiled per kernel or per graph."""
    bud = arm["budget"]
    lin, lm = bud["linear_bytes"] + bud["linear_scale_bytes"], bud["lm_head_bytes"]
    by: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    names: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for r in w:
        for d, key in ((by, classify(r["name"], r["read"], lm)), (names, r["name"][:120])):
            d[key][0] += r["read"]
            d[key][1] += r["write"]
            d[key][2] += 1
    steps = by["lm_head"][2]
    node = "graph" not in by
    if not steps:
        return {"fail": "no lm_head kernel in the window"}
    if not node and by["graph"][2] != steps:
        return {"fail": f"{by['graph'][2]} decode graphs for {steps} lm_head kernels"}
    rd = sum(r["read"] for r in w) / steps
    wr = sum(r["write"] for r in w) / steps
    tokens = pt["context"] + 4 + (arm["profile_steps"] + 1) / 2
    kv = kv_read(bud, pt["batch"], tokens)
    o = {"steps": steps, "read": rd, "write": wr, "linear": lin, "lm_head": lm, "kv_read": kv,
         "unreg_plain_lm": rd - lin - kv, "unreg_encoded_lm": rd - lin - kv - lm}
    o["share_plain_lm"] = o["unreg_plain_lm"] / rd
    o["share_encoded_lm"] = o["unreg_encoded_lm"] / rd
    cls = {k: {"read": v[0] / steps, "write": v[1] / steps, "kernels": v[2] / steps} for k, v in by.items()}
    o["classes"] = cls
    if node:
        a = cls.get("attention", {}).get("read", 0.0)
        o["gemm_activation_read"] = cls.get("gemm", {}).get("read", 0.0) - lin
        o["attention_excess_read"] = a - kv
        o["kv_check"] = a / kv if kv else None
    else:
        o["graph_excess_read"] = cls["graph"]["read"] - lin - kv
    o["top_kernels"] = sorted(({"name": k, "read": v[0] / steps, "write": v[1] / steps, "n": v[2] / steps}
                               for k, v in names.items()), key=lambda x: -x["read"])[:25]
    L, Lf = bud["layers"], FULL_LAYERS.get(arm["model_shape"]["hidden"], bud["layers"])
    if L != Lf:
        layered = sum(cls.get(k, {}).get("read", 0.0) for k in LAYERED)
        full = rd + layered * (Lf / L - 1)
        lin_f, kv_f = lin * Lf / L, kv * Lf / L
        o["full_depth"] = {"layers": Lf, "read": full, "linear": lin_f, "kv_read": kv_f,
                           "share_plain_lm": (full - lin_f - kv_f) / full,
                           "share_encoded_lm": (full - lin_f - kv_f - lm) / full,
                           "ephemeral_ex_lm": full - lin_f - kv_f - lm}
    return o


def main() -> int:
    out = Path(sys.argv[1])
    result, fails = [], []
    for jf in sorted(out.glob("*.json")):
        if jf.name in ("ephemeral.json",) or jf.name.startswith("ready"):
            continue
        arm = json.loads(jf.read_text())
        if "budget" not in arm or not arm.get("profile_steps"):
            continue
        tag = jf.stem
        cf = out / f"ncu-{tag}.csv"
        if not cf.exists():
            fails.append(f"{tag}: no {cf.name}")
            continue
        ws = windows(rows(cf))
        if len(ws) != len(arm["profiled"]):
            fails.append(f"{tag}: {len(ws)} windows for {len(arm['profiled'])} profiled points")
            continue
        for pt, w in zip(arm["profiled"], ws):
            timed = next((p for p in arm["points"] if p["batch"] == pt["batch"] and p["context"] == pt["context"]), {})
            s = split(arm, pt, w)
            result.append({"tag": tag, "layers": arm["budget"]["layers"], "compiled": arm["compiled"],
                           **pt, "step_ms_under_ncu": timed.get("step_ms"), **s})
    print(f"{'arm':22} {'B':>3} {'ctx':>6} {'read GB':>8} {'lin GB':>7} {'kv GB':>7} {'lm GB':>6} "
          f"{'unreg':>7} {'unreg-lm':>8} {'full':>7} {'full-lm':>7}")
    for r in result:
        if "fail" in r:
            print(f"{r['tag']:22} {r['batch']:3d} {r['context']:6d} FAIL {r['fail']}")
            continue
        f = r.get("full_depth", {})
        print(f"{r['tag']:22} {r['batch']:3d} {r['context']:6d} {r['read'] / 1e9:8.3f} {r['linear'] / 1e9:7.3f} "
              f"{r['kv_read'] / 1e9:7.3f} {r['lm_head'] / 1e9:6.3f} {r['share_plain_lm']:7.2%} {r['share_encoded_lm']:8.2%} "
              f"{f.get('share_plain_lm', float('nan')):7.2%} {f.get('share_encoded_lm', float('nan')):7.2%}")
    (out / "ephemeral.json").write_text(json.dumps({"points": result, "fails": fails}, indent=1))
    for x in fails:
        print("FAIL", x)
    return 1 if fails or not result or any("fail" in r for r in result) else 0


if __name__ == "__main__":
    sys.exit(main())
