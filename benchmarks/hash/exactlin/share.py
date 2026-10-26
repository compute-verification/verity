"""The share of a served step that PoUW doesn't credit, from an nsys capture: Llama-3.1-8B int8-exact served by one of
sbench's engines (`exact-fused`, the credited path, `exactlin`'s kernels the `verity_int8` ops' CUDA bodies; or `exact`,
their torch bodies), driven step by step (`LLMEngine.add_request`, `step`) through four scenarios:

    decode-short   B32 greedy decode on sbench's prompts (cycled): after their prefill and WARM steps, STEPS steps
    decode-2k      the same with each prompt padded at its front to PAD tokens
    prefill-2k     one PREFILL-token prompt, its one forward (prefill and first sample), REPS times
    prefill-2k-b4  four of them in one forward, REPS times

Every scenario runs untraced first (its wall time alone), then inside the one CUDA profiler range, with an NVTX range over
its measured steps (RANGE/<scenario>#<rep>) and one a step.  `--analyse` assigns each kernel to the range whose steps
launched it (by its launch call's time) and splits a forward's GPU time into PoUW's credited int8 GEMMs
(`verity_int8::scaled_mm`: exactlin_scaled_mm, or torch._int_mm's kernel with the padding before it and the EPILOGUE
dequantization kernels after it), the quantizer (exactlin_quant_rows, or the QUANT_BODY torch kernels before an int8
GEMM's padding), lm_head (any other GEMM), attention (with the torch copies right after its kernels), norms and
elementwise (Inductor's, vLLM's and torch's other kernels), sampling (the kernels after lm_head up to the model runner's
state update), other (that update and the next step's input prep, memcpy, memset), and the gaps: wall minus the GPU's
busy time (kernels, memcpy and memset, overlaps counted once).  It also states the FLOPs the forward's calls
imply: 2MNK per int8 GEMM and lm_head call at the model's shapes, 4 H d L per query token and layer for attention (half
the square for a causal prefill); elementwise FLOPs are left out.

    python share.py --arm exact-fused --out OUT/exact-fused [--scratch S]   # under nsys --capture-range=cudaProfilerApi
    python share.py --analyse capture.sqlite --meta OUT/exact-fused/share.json > OUT/exact-fused/analysis.json
    python share.py --summary OUT                                          # every arm's table and the token gate
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RANGE = "verity/exactlin/share/"
B, WARM, STEPS, PAD, PREFILL, REPS = 32, 8, 32, 2000, 2048, 4
T, MAX_LEN = 8192, 4096
EPILOGUE = 4    # int8_exact.dequantize's torch kernels: to(f32), * s_a, * s_w, to(bf16)
QUANT_BODY = 10  # int8_exact.quantize_rows' torch kernels: to(f32), full, abs, amax, clamp, /, /, round, clamp_, to(int8)


def build(arm: str, scratch: Path, gpu_mem: float):
    sys.path.insert(0, str(ROOT / "benchmarks" / "hash" / "dither"))
    import sbench
    os.environ["SBENCH_T"] = str(T)
    base = sbench.engine_args
    sbench.engine_args = lambda g: {**base(g), "max_model_len": MAX_LEN}
    return sbench.build(arm, scratch, gpu_mem), sbench


def scenarios(tok, sbench) -> dict[str, list[list[int]]]:
    short = [tok.encode(p) for p in sbench.prompts(B)]
    filler = tok.encode(" ".join(sbench.prompts(64)), add_special_tokens=False)
    while len(filler) < 4 * PREFILL:
        filler = filler + filler
    bos = short[0][:1]
    return {"decode-short": short,
            "decode-2k": [bos + filler[i:i + PAD - len(s)] + s[1:] for i, s in enumerate(short)],
            "prefill-2k": [bos + filler[:PREFILL - 1]],
            "prefill-2k-b4": [bos + filler[i * 7:i * 7 + PREFILL - 1] for i in range(4)]}


def drive(eng, tag: str, reqs: list[list[int]], decode: bool, nvtx, sync) -> dict:
    """Add REQS; for decode, step until each has its first token, then WARM steps, then time STEPS steps; for a prefill,
    time the steps until it's done."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=(1 + WARM + STEPS + 16) if decode else 1, ignore_eos=True, seed=0)
    ids = [f"{tag}-{i}" for i in range(len(reqs))]
    toks: dict[str, list[int]] = {}

    def step() -> None:
        for o in eng.step():
            toks[o.request_id] = list(o.outputs[0].token_ids)

    sync()
    for rid, p in zip(ids, reqs):
        eng.add_request(request_id=rid, prompt={"prompt_token_ids": list(p)}, params=sp)
    if decode:
        while not all(toks.get(r) for r in ids):
            step()
        for _ in range(WARM):
            step()
    ctx0 = [len(p) + len(toks.get(r, [])) for r, p in zip(ids, reqs)]
    sync()
    if nvtx:
        nvtx.range_push(RANGE + tag)
    t0 = time.perf_counter()
    n = 0
    while (n < STEPS) if decode else eng.has_unfinished_requests():
        if nvtx:
            nvtx.range_push(f"{RANGE}{tag}/s{n}")
        step()
        if nvtx:
            nvtx.range_pop()
        n += 1
    sync()
    wall = time.perf_counter() - t0
    if nvtx:
        nvtx.range_pop()
    ctx1 = [len(p) + len(toks.get(r, [])) for r, p in zip(ids, reqs)]
    if decode:
        eng.abort_request(ids)
    while eng.has_unfinished_requests():
        step()
    return {"steps": n, "wall_ms": round(1e3 * wall, 4), "requests": len(reqs), "prompt_tokens": [len(p) for p in reqs],
            "context_before": ctx0, "context_after": ctx1, "tokens": [toks.get(r, []) for r in ids]}


def run(a) -> int:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    llm, sbench = build(a.arm, Path(a.scratch), a.gpu_mem)
    doc: dict = {"arm": a.arm, "build_s": round(time.perf_counter() - t0, 1), "kernels": sbench.kernels_of(llm),
                 "config": json.loads((Path(sbench.PIN) / "config.json").read_text()),
                 "constants": {"B": B, "WARM": WARM, "STEPS": STEPS, "PAD": PAD, "PREFILL": PREFILL, "REPS": REPS, "T": T,
                               "MAX_LEN": MAX_LEN}, "untraced": {}, "traced": {}}
    eng = llm.llm_engine
    reqs = scenarios(llm.get_tokenizer(), sbench)
    sync = torch.cuda.synchronize
    for tag, rs in reqs.items():          # warm: every scenario's shapes compiled and tuned before anything is timed
        drive(eng, f"warm-{tag}", rs, tag.startswith("decode"), None, sync)
    for traced in (False, True):
        if traced:
            sync()
            torch.cuda.profiler.start()
        for tag, rs in reqs.items():
            reps = 1 if tag.startswith("decode") else REPS
            doc["traced" if traced else "untraced"][tag] = [
                drive(eng, f"{tag}#{k}", rs, tag.startswith("decode"), torch.cuda.nvtx if traced else None, sync)
                for k in range(reps)]
        if traced:
            sync()
            torch.cuda.profiler.stop()
    (out / "share.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"arm": a.arm, "build_s": doc["build_s"],
                      **{f"{k}_{t}_ms": [r["wall_ms"] for r in v] for k in ("untraced", "traced") for t, v in doc[k].items()}}))
    return 0


GEMM_I8 = re.compile(r"exactlin_scaled_mm|(gemm|xmma|nvjet|cutlass|cublas|imma).*(i8|s8|int8|imma|i16832)|"
                     r"(i8|s8|int8|imma).*(gemm|xmma|nvjet)", re.IGNORECASE)
QUANT = re.compile(r"exactlin_quant_rows")
ATTN = re.compile(r"flash|attn|attention|paged|reshape_and_cache|BatchDecode|BatchPrefill|merge_state|fmha", re.IGNORECASE)
GEMM = re.compile(r"gemm|xmma|nvjet|cutlass|cublas|splitK", re.IGNORECASE)
TORCH = re.compile(r"at::native|CatArrayBatchedCopy|at::cuda|cub::|elementwise_kernel", re.IGNORECASE)
VLLM_ELEM = re.compile(r"rms_norm|rotary|act_and_mul|silu|layer_norm", re.IGNORECASE)
PADDING = re.compile(r"CatArrayBatchedCopy")
FILL = re.compile(r"FillFunctor")
PREP = re.compile(r"_post_update|_apply_write|_prepare_|_combine_sampled_and_draft|_gather_block_tables|_compute_slot_mapping")
SAMPLE = re.compile(r"_gumbel_sample|_get_num_sampled|sampl", re.IGNORECASE)
FORWARD = {"quant", "gemm_i8", "attention", "inductor", "vllm_elem"}
BUCKETS = ["credited int8 GEMM", "quantizer", "lm_head", "attention", "norms and elementwise", "sampling", "other"]


def base(name: str) -> str:
    for kind, rx in (("prep", PREP), ("quant", QUANT), ("gemm_i8", GEMM_I8), ("attention", ATTN), ("lm_head", GEMM),
                     ("vllm_elem", VLLM_ELEM), ("torch", TORCH)):
        if rx.search(name):
            return kind
    return "inductor" if name.startswith("triton_") else "other"


def assign(names: list[str], graph: list[bool]) -> tuple[list[str], dict]:
    """Each kernel's bucket, the kernels in GPU order; and the run lengths the positional rules saw.

    An int8 GEMM's quantizer is exactlin_quant_rows (exactlin.row_quant copies x first only when stride(1) != 1), or the
    QUANT_BODY torch kernels before torch._int_mm's padding (a Cat, and the Fill of its zeros, before the GEMM); a torch run
    right after an attention kernel is attention's (FlashAttention's split-KV decode copies q in and its output out).
    lm_head is the GEMM the sampler follows; any other GEMM is an int8 one (cuBLASLt's nvjet names carry no dtype)."""
    b = [base(n) for n in names]
    for i in reversed([i for i, k in enumerate(b) if k == "lm_head"]):
        if i + 1 < len(b) and b[i + 1] == "lm_head":
            continue
        j = i + 1
        while j < len(b) and b[j] not in ("lm_head", "gemm_i8", "quant") and not SAMPLE.search(names[j]):
            j += 1
        if j == len(b) or not SAMPLE.search(names[j]):
            b[i] = "gemm_i8"
    cat: list[str | None] = [None] * len(b)
    post, pre, left = [], [], []
    for i, k in enumerate(b):
        if k == "gemm_i8":
            cat[i] = "credited int8 GEMM"
        elif k == "quant":
            cat[i] = "quantizer"
    for g in [i for i, k in enumerate(b) if k == "gemm_i8"]:
        j = g + 1
        while j < len(b) and b[j] == "torch" and cat[j] is None:
            j += 1
        post.append(j - g - 1)
        for i in range(g + 1, min(j, g + 1 + EPILOGUE)):
            cat[i] = "credited int8 GEMM"
        j = g - 1
        if j >= 0 and b[j] == "torch" and cat[j] is None and PADDING.search(names[j]):
            cat[j] = "credited int8 GEMM"
            j -= 1
            if j >= 0 and b[j] == "torch" and cat[j] is None and FILL.search(names[j]):
                cat[j] = "credited int8 GEMM"
                j -= 1
        fused = j >= 0 and b[j] == "quant"
        while j >= 0 and b[j] == "quant":
            j -= 1
        start, n = j, 0
        while j >= 0 and b[j] == "torch" and cat[j] is None:
            if not fused and n < QUANT_BODY:
                cat[j] = "quantizer"
                n += 1
            j -= 1
        pre.append(start - j)
        left.append(start - j - n)
    for a in [i for i, k in enumerate(b) if k == "attention"]:
        j = a + 1
        while j < len(b) and b[j] == "torch" and cat[j] is None:
            cat[j] = "attention"
            j += 1
    for h in [i for i, k in enumerate(b) if k == "lm_head"]:
        cat[h] = "lm_head"
        j, bucket = h + 1, "sampling"
        while j < len(b) and not graph[j] and b[j] not in FORWARD and b[j] != "lm_head" and cat[j] is None:
            bucket = "other" if b[j] == "prep" else bucket
            cat[j] = bucket
            j += 1
    for i, k in enumerate(b):
        if cat[i] is None:
            cat[i] = {"attention": "attention", "inductor": "norms and elementwise", "vllm_elem": "norms and elementwise",
                      "torch": "norms and elementwise"}.get(k, "other")
    hist = lambda xs: {str(v): xs.count(v) for v in sorted(set(xs))}  # noqa: E731
    return cat, {"after_int8_gemm_torch_run": hist(post), "before_int8_gemm_torch_run": hist(pre),
                 "before_quantizer_unclaimed_torch_run": hist(left)}


def _busy(iv: list[tuple[int, int]]) -> int:
    tot, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def flops(cfg: dict, tag: str, run: dict) -> dict:
    h, inter, nl, v = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    d = cfg.get("head_dim") or h // nh
    linears = [((nh + 2 * nkv) * d, h), (h, nh * d), (2 * inter, h), (h, inter)]
    if tag.startswith("decode"):
        m, seqs = run["requests"], run["requests"]
        ctx = [(a + b) / 2 for a, b in zip(run["context_before"], run["context_after"])]
        attn = nl * sum(4 * nh * d * c for c in ctx)
    else:
        m, seqs = sum(run["prompt_tokens"]), run["requests"]
        ctx = run["prompt_tokens"]
        attn = nl * sum(4 * nh * d * L * (L + 1) / 2 for L in ctx)
    credited = nl * sum(2 * m * n * k for n, k in linears)
    lm = 2 * seqs * v * h
    return {"tokens": m, "context_mean": round(statistics.mean(ctx), 1), "credited_gflop": round(credited / 1e9, 3),
            "lm_head_gflop": round(lm / 1e9, 3), "attention_gflop": round(attn / 1e9, 3),
            "credited_calls": 4 * nl, "uncovered_flop_share": round((lm + attn) / (credited + lm + attn), 4)}


def analyse(path: str, meta: dict) -> dict:
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    pid = lambda g: (int(g) >> 24) & 0xFFFFFF  # noqa: E731
    outer: dict[str, tuple[int, int, int]] = {}
    steps: dict[str, list[tuple[int, int]]] = {}
    for start, end, text, tid, text_id in db.execute(
            "SELECT start, end, text, globalTid, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text if text is not None else strings.get(text_id)
        if not name or not name.startswith(RANGE):
            continue
        rest = name.removeprefix(RANGE)
        if "/s" in rest:
            steps.setdefault(rest.split("/s")[0], []).append((int(start), int(end)))
        else:
            outer[rest] = (int(start), int(end), pid(tid))
    launch = {int(c): (int(s), pid(t)) for c, s, t in db.execute(
        "SELECT correlationId, start, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    cols = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    gcol = "graphNodeId" if "graphNodeId" in cols else "NULL"
    kern = [(strings.get(n, str(n)), int(c), int(s), int(e), g is not None) for n, c, s, e, g in db.execute(
        f"SELECT demangledName, correlationId, start, end, {gcol} FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    mem = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            mem += [(int(c), int(s), int(e)) for c, s, e in db.execute(f"SELECT correlationId, start, end FROM {table}")]
        except sqlite3.Error:
            pass
    cfg = meta["config"]
    doc: dict = {}
    for rng, (r0, r1, p) in sorted(outer.items()):
        tag, rep = rng.rsplit("#", 1)
        inside = lambda c: c in launch and launch[c][1] == p and r0 <= launch[c][0] <= r1  # noqa: E731
        ks = sorted((k for k in kern if inside(k[1])), key=lambda k: k[2])
        ms = [m for m in mem if inside(m[0])]
        cats, runs = assign([k[0] for k in ks], [k[4] for k in ks])
        fwd = max(1, sum(1 for i, c in enumerate(cats) if c == "lm_head" and (i == 0 or cats[i - 1] != "lm_head")))
        busy = _busy([(k[2], k[3]) for k in ks] + [(m[1], m[2]) for m in ms])
        by: dict[str, float] = {c: 0.0 for c in BUCKETS}
        launches: dict[str, int] = {c: 0 for c in BUCKETS}
        names: dict[str, dict[str, list]] = {c: {} for c in BUCKETS}
        for k, c in zip(ks, cats):
            by[c] += k[3] - k[2]
            launches[c] += 1
            row = names[c].setdefault(k[0][:120], [0, 0.0])
            row[0] += 1
            row[1] += (k[3] - k[2]) / 1e3
        by["other"] += sum(m[2] - m[1] for m in ms)
        wall = r1 - r0
        run = meta["traced"][tag][int(rep)]
        untraced = meta["untraced"][tag][int(rep)]["wall_ms"]
        per = lambda x: round(x / fwd / 1e6, 4)  # noqa: E731
        sw = sorted(e - s for s, e in steps.get(rng, []))
        doc.setdefault(tag, []).append({
            "rep": int(rep), "forwards": fwd, "steps": len(sw), "wall_ms": per(wall),
            "wall_untraced_ms": round(untraced / fwd, 4), "gpu_busy_ms": per(busy),
            "kernel_ms": {c: per(by[c]) for c in BUCKETS}, "gaps_ms": per(wall - busy),
            "launches": {c: round(launches[c] / fwd, 2) for c in BUCKETS},
            "memcpy_memset": round(len(ms) / fwd, 2),
            "uncovered_share_of_busy": round(1 - by["credited int8 GEMM"] / busy, 4) if busy else None,
            "uncovered_share_of_wall": round(1 - by["credited int8 GEMM"] / wall, 4),
            "step_ms_max": round(sw[-1] / 1e6, 4) if sw else None, "runs": runs, "flops": flops(cfg, tag, run),
            "kernels": {c: sorted(([n, v[0], round(v[1], 1)] for n, v in names[c].items()), key=lambda r: -r[2])[:12]
                        for c in BUCKETS}})
    return doc


def summary(out: Path) -> dict:
    arms = {p.parent.name: json.loads(p.read_text()) for p in sorted(out.glob("*/analysis.json"))}
    metas = {p.parent.name: json.loads(p.read_text()) for p in sorted(out.glob("*/share.json"))}
    table = {}
    for arm, doc in arms.items():
        for tag, reps in doc.items():
            keys = ["wall_ms", "wall_untraced_ms", "gpu_busy_ms", "gaps_ms", "uncovered_share_of_busy",
                    "uncovered_share_of_wall"]
            row = {k: round(statistics.median(r[k] for r in reps), 4) for k in keys}
            row["kernel_ms"] = {c: round(statistics.median(r["kernel_ms"][c] for r in reps), 4) for c in BUCKETS}
            row["launches"] = reps[0]["launches"]
            row["flops"] = reps[0]["flops"]
            row["reps"] = len(reps)
            table[f"{arm} {tag}"] = row
    gate = {}
    if "exact" in metas and "exact-fused" in metas:
        a, b = (metas[x]["traced"]["decode-short"][0]["tokens"] for x in ("exact-fused", "exact"))
        gate["decode-short tokens, exact-fused vs exact"] = {"requests_equal": sum(x == y for x, y in zip(a, b)),
                                                             "requests": len(a)}
    return {"table": table, "gate": gate}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--out")
    ap.add_argument("--scratch", default="/tmp")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--analyse")
    ap.add_argument("--meta")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.analyse:
        print(json.dumps(analyse(a.analyse, json.loads(Path(a.meta).read_text())), indent=1))
    elif a.summary:
        print(json.dumps(summary(Path(a.summary)), indent=1))
    else:
        sys.exit(run(a))
