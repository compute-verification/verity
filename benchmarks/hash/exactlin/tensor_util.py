"""A served step's wall time by kernel class and its tensor utilization, from an Nsight Systems export of a served vLLM run
(`nsys export --type sqlite`), and the losses ranked against a utilization target.

Steps.  The int8 row quantizer launches one program a token (stock vLLM's `dynamic_scaled_int8_quant`), or one a few
tokens (`exactlin_quant_rows` at `int8_fused.quant_plan`'s plan), so a forward is 4 L consecutive quantizer launches
(qkv, o, gate_up and down in each of L layers) whose grids repeat each layer, and its T is a layer's largest grid (a
linear at one row a program).  Step i runs from forward i's first quantizer launch to forward i + 1's: a cyclic cut, so
the step holds everything between them (its own layers, the LM head and sampler, the host's scheduling
and the Commit's work, and the next forward's embedding and first norm).  Only steps of `--tokens` T whose next forward
is also of that T count, so the warm-up and profile forwards, and a burst's last step, drop out.

Wall time.  Every kernel, copy and memset on the device is clipped to the step and swept: each instant goes to the
highest class in CLASSES' order that is running then (the tensor classes first), else to the host gap (nothing on the
device, so it waits on the host).  The host gap before the end of the step's last int8 GEMM (its forward's last
linear) is `launch gaps in the forward`; after it, `host gap between forwards` (the scheduler, input preparation and
the host's side of sampling).  So a class's `exclusive` time is what it adds to the wall beside the classes before
it, and the classes' exclusive times and the host gaps sum to the step's wall.  Each class's `kernel` time (its kernels'
summed durations, overlap counted) and `overlap` (its kernel time under a tensor class) are beside it.  GPU metrics
are sampled every `metric_period_us`, so a class whose kernels run shorter than that reads its neighbours' values too.

Tensor utilization.  Declared MACs from the model's config (`--config`, an HF config.json) and the step's batch: the
int8 linears' T * L * sum(N K); causal attention's 2 H d (q c - q (q - 1) / 2) a layer for a request of q query rows,
the last of its c-token context (prefill of whole `--prompt`-token prompts: T / prompt requests of q = c = prompt), in
bf16; and the bf16 LM head's hidden * vocab for each request's one sampled row.  Utilization is sum(MACs / theta) over
the wall, theta the int8 rate (`--theta`, MAC/s, default the H200's 4096 a SM-clock x 132 SMs x 1830 MHz, the clock
`--theta-mhz` its sheet quotes it at) and bf16 at half of it, so a step at the target spends `--target` of its wall at
the declared rate.  A tensor class's own utilization is its MACs at its rate over its kernel time.  Rates are per SM
clock at the clock the card held: with GPU metrics, `util_at_clock` and `class_util_at_clock` scale theta to the
measured SM clock (the steps', or the class's), and the table leads with them, the sheet clock's figures beside them.

Losses.  The wall less the ideal tensor time is the tensor-idle time.  Each class's loss is its exclusive time less its
ideal time (a non-tensor class's ideal is 0), and the host gap's is all of it; the losses sum to the tensor-idle time.
`util_if_zero` is the step's utilization with that loss gone, all else unchanged.

    python tensor_util.py REPORT.sqlite --tokens 8192 [--prompt 1024] [--config CONFIG.json] [--out util.json]
    python tensor_util.py --table util.json        # the Markdown tables
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

CONFIG = (Path(__file__).resolve().parents[3] / "integrations/vllm/verity_vllm/engine/profiles/hf_configs"
          / "LLAMA32_1B_INT8.config.json")
SHEET_MHZ = 1830.0
THETA_INT8 = 4096 * 132 * SHEET_MHZ * 1e6  # MAC/s: the H200's dense s8 MACs a SM-clock, its SMs, its sheet's clock
HOST_GAP = "host gap"
GAP_IN, GAP_OUT = "launch gaps in the forward", "host gap between forwards"
# (class, MAC kind or None, pattern on the demangled name), in priority order: an instant two classes share is the first's
CLASSES = [
    ("int8 GEMM", "int8", r"exactlin_scaled_mm|scaled_mm_kernel|cutlass_scaled_mm|cutlass.*int8|imma|i8i8|s8s8|int8_gemm"),
    ("attention", "attn", r"flash::|FlashAttn|flash_fwd|flash_attn|fmha|attn_fwd|attention|\bfa3|lut_rows|kv_rows"),
    ("bf16 GEMM (LM head)", "head", r"nvjet|xmma_gemm|sm90_.*gemm|cutlass.*gemm|gemv|cublas|ampere_.*gemm|Kernel2|gemmSN|"
                                    r"matmul_kernel_persistent"),
    ("Commit hashing", None, r"rows_leaf|rows_level|chunk_leaves|tree_level|pos_leaves|sha512|hm96"),
    ("Commit gather", None, r"gather_copy|native_collect"),
    ("H1", None, r"\bkf?_(midstates|leaves|level|pads|public_salts|top|rows|segments|step|trees|tiles|strips|tree_key|"
                 r"units_level|form_leaves|node_leaves|key_leaf|dequant|x|y|ceiling|f1|fill|hash|row_digests|tile_leaves)\b"),
    ("quantize", None, r"exactlin_quant_rows|scaled_int8_quant|int8_quant|kv_quant"),
    ("norms", None, r"rms_norm|rmsnorm|layer_norm|add\d?_norm|embed_norm|final_norm|triton_(red|per)_"),
    ("activation", None, r"silu|act_and_mul|gelu|swiglu"),
    ("rope", None, r"rotary|rope"),
    ("KV cache write", None, r"reshape_and_cache|cache_kernel|concat_and_cache"),
    ("sampling and logits", None, r"argmax|softmax|topk|top_k|sampl|gumbel|logit|cumsum|sort"),
    ("other kernels", None, r""),
    ("memcpy", None, None),
    ("memset", None, None),
]
TENSOR = {name: kind for name, kind, _ in CLASSES if kind}
QUANT = re.compile(r"exactlin_quant_rows|dynamic_scaled_int8_quant")
# GPU metrics (`nsys --gpu-metrics-devices`) read where the report has them: the SM clock and the tensor pipe's active share
METRICS = {"clock_mhz": re.compile(r"^GPC Clock Frequency"), "tensor_active": re.compile(r"^Tensor Active")}
_PATTERNS = [(name, re.compile(p, re.IGNORECASE)) for name, _, p in CLASSES if p is not None]
RANK = {name: i for i, (name, _, _) in enumerate(CLASSES)}


def classify(name: str) -> str:
    return next(c for c, p in _PATTERNS if p.search(name))


def model(config: dict) -> dict:
    h, i = config["hidden_size"], config["intermediate_size"]
    nh, kvh = config["num_attention_heads"], config["num_key_value_heads"]
    d = config.get("head_dim") or h // nh
    linears = {"qkv": ((nh + 2 * kvh) * d, h), "o": (h, nh * d), "gate_up": (2 * i, h), "down": (h, i)}
    return {"layers": config["num_hidden_layers"], "heads": nh, "head_dim": d, "hidden": h, "vocab": config["vocab_size"],
            "linears": linears, "int8_macs_per_token": config["num_hidden_layers"] * sum(n * k for n, k in linears.values())}


def requests_of(tokens: int, prompt: int) -> list[int]:
    """A prefill step's requests' query rows: whole prompts, and any remainder as one more request's rows."""
    whole, rest = divmod(tokens, prompt)
    return [prompt] * whole + ([rest] if rest else [])


def step_macs(m: dict, tokens: int, prompt: int) -> dict:
    """Declared MACs of one prefill step: each request's q rows the last of a q-row context (no cached prefix)."""
    reqs = requests_of(tokens, prompt)
    attn = sum(2 * m["heads"] * m["head_dim"] * (q * q - q * (q - 1) // 2) for q in reqs) * m["layers"]
    return {"int8": tokens * m["int8_macs_per_token"], "attn": attn, "head": len(reqs) * m["hidden"] * m["vocab"]}


def load(path: str) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = lambda s, *p: db.execute(s, p).fetchall()  # noqa: E731
    tables = {r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(q("SELECT id, value FROM StringIds"))
    cols = {r[1] for r in q("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "demangledName" if "demangledName" in cols else "shortName"
    kern = [(s, e, strings.get(n, str(n)), gx * gy * gz, dev, pid >> 24) for s, e, n, gx, gy, gz, dev, pid in
            q(f"SELECT start, end, {name_col}, gridX, gridY, gridZ, deviceId, globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    ops = []
    for table, cls in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
        if table in tables:
            ops += [(s, e, cls, b, dev) for s, e, b, dev in q(f"SELECT start, end, bytes, deviceId FROM {table}")]
    metrics: dict[str, tuple[list[int], list[float]]] = {}
    if "GPU_METRICS" in tables and "TARGET_INFO_GPU_METRICS" in tables:
        for mid, name in q("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS"):
            key = next((k for k, p in METRICS.items() if p.search(str(name))), None)
            if key:
                rows = q("SELECT timestamp, value FROM GPU_METRICS WHERE metricId = ? ORDER BY timestamp", mid)
                # nsys stores the clocks in Hz under an "[MHz]" name
                scale = 1e-6 if key == "clock_mhz" and rows and max(v for _, v in rows) > 1e5 else 1.0
                metrics[key] = ([int(t) for t, _ in rows], [float(v) * scale for _, v in rows])
    return {"kernels": kern, "ops": ops, "metrics": metrics}


def forwards(kern: list, layers: int) -> list[tuple[int, int]]:
    """(start of its first quantizer launch, T) of each forward: 4 L consecutive quantizer launches, each of its layer's
    first launch's grid or less and of the previous layer's same linear's grid, and T a layer's largest grid."""
    out, run = [], []
    for s, _, name, grid, _, _ in sorted(k for k in kern if QUANT.search(k[2])):
        n = len(run)
        if n and (grid != run[n - 4][1] if n >= 4 else grid > run[0][1]):
            run = []
        run.append((s, grid))
        if len(run) == 4 * layers:
            out.append((run[0][0], max(g for _, g in run[:4])))
            run = []
    return out


def sweep(iv: list[tuple], a: int, b: int, segments: list | None = None) -> dict[str, int]:
    """Each instant of [a, b) to the highest-ranked class running then, else to the host gap; with `segments`, each
    stretch's (start, end, class) appended to it in time order."""
    ev = []
    for s, e, c, *_ in iv:
        s, e = max(s, a), min(e, b)
        if s < e:
            ev += [(s, 1, c), (e, -1, c)]
    ev.sort(key=lambda x: (x[0], x[1]))
    live: dict[str, int] = defaultdict(int)
    out: dict[str, int] = defaultdict(int)
    t = a
    for x, d, c in ev + [(b, 0, None)]:
        if x > t:
            top = min((k for k, n in live.items() if n), key=RANK.__getitem__, default=HOST_GAP)
            out[top] += x - t
            if segments is not None:
                segments.append((t, x, top))
            t = x
        if c is not None:
            live[c] += d
    return dict(out)


def split_gaps(segments: list, cut: int) -> list:
    """The host gap's stretches before `cut` as launch gaps in the forward, after it as the host gap between forwards."""
    out = []
    for s, e, c in segments:
        if c != HOST_GAP:
            out.append((s, e, c))
            continue
        if s < cut:
            out.append((s, min(e, cut), GAP_IN))
        if e > cut:
            out.append((max(s, cut), e, GAP_OUT))
    return out


def overlap_with_tensor(iv: list[tuple], a: int, b: int) -> dict[str, int]:
    """Each non-tensor class's kernel time in [a, b) while some tensor class's kernel ran."""
    clip = [(max(s, a), min(e, b), c) for s, e, c, *_ in iv if min(e, b) > max(s, a)]
    tensor = sorted((s, e) for s, e, c in clip if c in TENSOR)
    merged: list[list[int]] = []
    for s, e in tensor:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    starts = [m[0] for m in merged]
    out: dict[str, int] = defaultdict(int)
    for s, e, c in clip:
        if c in TENSOR:
            continue
        i = max(bisect.bisect_right(starts, s) - 1, 0)
        while i < len(merged) and merged[i][0] < e:
            out[c] += max(0, min(e, merged[i][1]) - max(s, merged[i][0]))
            i += 1
    return dict(out)


def analyse(path: str, tokens: int, prompt: int, config: dict, theta: float, target: float, steady: bool = False,
            theta_mhz: float = SHEET_MHZ) -> dict:
    """With `steady`, a step longer than 1.5 times the median step (a window's close or a learning pass) is dropped and
    listed in the summary's `dropped`.  `theta_mhz` is the SM clock `theta` is quoted at."""
    m = model(config)
    data = load(path)
    kern = data["kernels"]
    dev = max({k[4] for k in kern}, key=lambda d: sum(k[1] - k[0] for k in kern if k[4] == d))
    kern = [k for k in kern if k[4] == dev]
    ops = [o for o in data["ops"] if o[4] == dev]
    names = {k[2] for k in kern}
    cls_of = {n: classify(n) for n in names}
    fw = forwards(kern, m["layers"])
    picked = [(fw[i][0], fw[i + 1][0]) for i in range(len(fw) - 1) if fw[i][1] == tokens and fw[i + 1][1] == tokens]
    dropped = []
    if steady and picked:
        median = sorted(b - a for a, b in picked)[(len(picked) - 1) // 2]
        dropped = [{"start_ns": a, "wall_ms": (b - a) / 1e6} for a, b in picked if b - a > 1.5 * median]
        picked = [(a, b) for a, b in picked if b - a <= 1.5 * median]
    # (start, end, class, kernel name or None, bytes) of everything on the device, by start
    iv = sorted([(s, e, cls_of[n], n, 0) for s, e, n, *_ in kern] + [(s, e, c, None, nb) for s, e, c, nb, _ in ops])
    starts = [x[0] for x in iv]
    end_max, hi = [], 0
    for x in iv:
        hi = max(hi, x[1])
        end_max.append(hi)
    macs = step_macs(m, tokens, prompt)
    rate = {"int8": theta, "attn": theta / 2, "head": theta / 2}
    ideal = {c: macs[k] / rate[k] * 1e9 for c, k in TENSOR.items()}  # ns a step
    steps = []
    tot_excl: dict[str, float] = defaultdict(float)
    tot_kern: dict[str, float] = defaultdict(float)
    tot_ovl: dict[str, float] = defaultdict(float)
    launches: dict[str, int] = defaultdict(int)
    per_name: dict[str, list] = defaultdict(lambda: [0, 0])
    bytes_moved: dict[str, int] = defaultdict(int)
    samples: dict[str, dict[str, list[float]]] = {k: defaultdict(lambda: [0.0, 0]) for k in data["metrics"]}
    for a, b in picked:
        win = iv[bisect.bisect_left(end_max, a):bisect.bisect_left(starts, b)]
        seg: list = []
        sweep(win, a, b, seg)
        body_end = max((min(e, b) for s, e, c, *_ in win if c == "int8 GEMM" and a <= s), default=a)
        seg = split_gaps(seg, body_end)
        ex: dict[str, int] = defaultdict(int)
        for s, e, c in seg:
            ex[c] += e - s
        steps.append({"start_ns": a, "wall_ms": (b - a) / 1e6, "util": sum(ideal.values()) / (b - a),
                      "exclusive_ms": {c: v / 1e6 for c, v in sorted(ex.items(), key=lambda kv: -kv[1])}})
        seg_starts = [x[0] for x in seg]
        for key, (ts, vs) in data["metrics"].items():
            for i in range(bisect.bisect_left(ts, a), bisect.bisect_left(ts, b)):
                c = seg[bisect.bisect_right(seg_starts, ts[i]) - 1][2]
                for k in (c, "step"):
                    samples[key][k][0] += vs[i]
                    samples[key][k][1] += 1
        for c, v in ex.items():
            tot_excl[c] += v
        for c, v in overlap_with_tensor(win, a, b).items():
            tot_ovl[c] += v
        for s, e, c, name, nbytes in win:
            t = min(e, b) - max(s, a)
            if t <= 0:
                continue
            tot_kern[c] += t
            if a <= s:
                launches[c] += 1
                bytes_moved[c] += nbytes
            if name is not None:
                per_name[name][0] += t
                per_name[name][1] += a <= s
    n = len(picked)
    if not n:
        return {"report": path, "error": f"no two consecutive forwards of T {tokens}",
                "forwards": [{"start_ns": s, "tokens": t} for s, t in fw]}
    wall = sum(b - a for a, b in picked) / n
    ideal_all = sum(ideal.values())
    idle = wall - ideal_all
    mean = lambda key, c: samples[key][c][0] / samples[key][c][1] if key in samples and samples[key][c][1] else None  # noqa: E731
    ts = next(iter(data["metrics"].values()), ([], []))[0]
    period = sorted(y - x for x, y in zip(ts, ts[1:]))[(len(ts) - 1) // 2] / 1e3 if len(ts) > 1 else None
    at_clock = lambda u, mhz: u * theta_mhz / mhz if u is not None and mhz else None  # noqa: E731
    rows = []
    for c in [c for c, _, _ in CLASSES] + [GAP_IN, GAP_OUT]:
        ex = tot_excl.get(c, 0.0) / n
        if not ex and not tot_kern.get(c):
            continue
        loss = ex - ideal.get(c, 0.0)
        kind = TENSOR.get(c)
        class_util = ideal[c] * n / tot_kern[c] if kind and tot_kern.get(c) else None
        rows.append({
            "class": c, "exclusive_ms": ex / 1e6, "share_of_wall": ex / wall, "kernel_ms": tot_kern.get(c, 0.0) / n / 1e6,
            "overlap_with_tensor_ms": tot_ovl.get(c, 0.0) / n / 1e6, "launches": launches.get(c, 0) / n,
            "gb": bytes_moved[c] / n / 1e9 if c in bytes_moved else None,
            "macs": macs[kind] if kind else None, "ideal_ms": ideal.get(c, 0.0) / 1e6,
            "class_util": class_util, "class_util_at_clock": at_clock(class_util, mean("clock_mhz", c)),
            "loss_ms": loss / 1e6, "loss_share": loss / idle if idle > 0 else None,
            "util_if_zero": ideal_all / (wall - loss) if wall > loss else None,
            "util_if_zero_at_clock": at_clock(ideal_all / (wall - loss) if wall > loss else None, mean("clock_mhz", "step")),
            "tensor_active": mean("tensor_active", c), "clock_mhz": mean("clock_mhz", c),
            "top": sorted(([short(k), v[0] / n / 1e6, v[1] / n] for k, v in per_name.items() if cls_of[k] == c),
                          key=lambda r: -r[1])[:6]})
    clock = mean("clock_mhz", "step")
    ideal_clock = ideal_all * theta_mhz / clock if clock else None
    return {
        "report": path, "device": dev, "tokens": tokens, "prompt": prompt, "requests": len(requests_of(tokens, prompt)),
        "model": {k: v for k, v in m.items() if k != "linears"} | {"linears": {k: list(v) for k, v in m["linears"].items()}},
        "theta_int8": theta, "theta_mhz": theta_mhz, "target": target,
        "forwards": [{"start_ns": s, "tokens": t} for s, t in fw],
        "steps": steps, "summary": {
            "steps": n, "wall_ms": wall / 1e6, "ideal_ms": ideal_all / 1e6, "util": ideal_all / wall,
            "util_at_clock": at_clock(ideal_all / wall, clock),
            "wall_at_target_ms": ideal_all / target / 1e6, "over_target_ms": (wall - ideal_all / target) / 1e6,
            "wall_at_target_at_clock_ms": ideal_clock / target / 1e6 if clock else None,
            "over_target_at_clock_ms": (wall - ideal_clock / target) / 1e6 if clock else None,
            "tensor_idle_ms": idle / 1e6, "macs": macs, "tensor_active": mean("tensor_active", "step"),
            "clock_mhz": clock, "metric_period_us": period, "dropped": dropped},
        "classes": sorted(rows, key=lambda r: -r["loss_ms"]),
        "unclassified": sorted(([short(k), v[0] / n / 1e6, v[1] / n] for k, v in per_name.items()
                                if cls_of[k] == "other kernels"), key=lambda r: -r[1])[:15]}


def short(name: str) -> str:
    """A demangled name without its template arguments and parameters, at most 90 characters."""
    out, depth = [], 0
    for ch in name:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    s = "".join(out).strip()
    s = s.removeprefix("void ").strip()
    return (s or name)[:90]


def table(doc: dict) -> str:
    if "error" in doc:
        return f"**{doc['report']}**: {doc['error']}"
    s = doc["summary"]
    sheet = doc.get("theta_mhz", SHEET_MHZ)
    gm = s.get("clock_mhz") is not None
    # derived from the stored rates, so an analysis written before the clock fields renders the same
    at = lambda u, mhz: u * sheet / mhz if u is not None and mhz else None  # noqa: E731
    head = (f"T {doc['tokens']:,} ({doc['requests']} requests of {doc['prompt']:,} tokens), {s['steps']} steps: wall "
            f"{s['wall_ms']:.2f} ms a step, ideal tensor time {s['ideal_ms']:.2f} ms at {sheet:,.0f} MHz (int8 "
            f"{s['macs']['int8'] / 1e12:.3f} T MACs, attention {s['macs']['attn'] / 1e12:.3f} T bf16, LM head "
            f"{s['macs']['head'] / 1e12:.4f} T bf16; θ {doc['theta_int8'] / 1e12:.1f} T int8 MAC/s at {sheet:,.0f} MHz), ")
    if gm:
        wall_t = at(s["ideal_ms"], s["clock_mhz"]) / doc["target"]
        head += (f"**tensor utilization {at(s['util'], s['clock_mhz']):.1%} at the measured SM clock of "
                 f"{s['clock_mhz']:,.0f} MHz** ({s['util']:.1%} at the data sheet's {sheet:,.0f} MHz). At "
                 f"{doc['target']:.0%} of the rate at that clock the step would take {wall_t:.2f} ms: "
                 f"{s['wall_ms'] - wall_t:.2f} ms over ({s['wall_at_target_ms']:.2f} ms at {sheet:,.0f} MHz).")
    else:
        head += (f"**tensor utilization {s['util']:.1%} at the data sheet's {sheet:,.0f} MHz** (no measured clock). At "
                 f"{doc['target']:.0%} the step would take {s['wall_at_target_ms']:.2f} ms: {s['over_target_ms']:.2f} ms over.")
    lines = [head]
    f = lambda v, p="{:.3f}": "" if v is None else p.format(v)  # noqa: E731
    if gm:
        lines[-1] += (f" GPU metrics over the steps: SM clock {s['clock_mhz']:.0f} MHz, Tensor Active "
                      f"{f(s['tensor_active'], '{:.1f}')}%")
        if s.get("metric_period_us"):
            lines[-1] += (f" (sampled every {s['metric_period_us']:.0f} µs, so a class of shorter kernels reads its "
                          "neighbours' values too)")
        lines[-1] += "."
    if s.get("dropped"):
        lines[-1] += (f" Steady state: {len(s['dropped'])} longer step(s) dropped (" +
                      ", ".join(f"{d['wall_ms']:.0f} ms" for d in s["dropped"]) + ").")
    util_col = f"class util at its SM clock ({sheet:,.0f} MHz)" if gm else f"class util at {sheet:,.0f} MHz"

    def class_util(r: dict) -> str:
        if r.get("class_util") is None:
            return ""
        if gm and r.get("clock_mhz"):
            return f"{at(r['class_util'], r['clock_mhz']):.1%} ({r['class_util']:.1%})"
        return f"{r['class_util']:.1%}"

    lines += [
        "",
        "| class | wall ms/step (exclusive) | share of wall | kernel ms/step | under a tensor kernel, ms | launches/step "
        f"| {util_col} | loss ms | share of tensor-idle | util if loss were 0{' (at the SM clock)' if gm else ''} |"
        + (" Tensor Active % | SM MHz |" if gm else ""),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if gm else ""),
    ]
    for r in doc["classes"]:
        lines.append(f"| {r['class']} | {r['exclusive_ms']:.3f} | {r['share_of_wall']:.1%} | {r['kernel_ms']:.3f} | "
                     f"{f(r['overlap_with_tensor_ms'])} | {r['launches']:.0f} | {class_util(r)} | "
                     f"{r['loss_ms']:.3f} | {f(r['loss_share'], '{:.1%}')} | "
                     f"{f(at(r['util_if_zero'], s['clock_mhz']) if gm else r['util_if_zero'], '{:.1%}')} |"
                     + (f" {f(r.get('tensor_active'), '{:.1f}')} | {f(r.get('clock_mhz'), '{:.0f}')} |" if gm else ""))
    lines += ["", "Top kernels a step (ms, launches):"]
    for r in doc["classes"]:
        if r["top"]:
            lines.append(f"- {r['class']}: " + "; ".join(f"`{n}` {ms:.3f} ({c:.0f})" for n, ms, c in r["top"][:4]))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", nargs="?")
    ap.add_argument("--tokens", type=int, help="the steps' T")
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--theta", type=float, default=THETA_INT8)
    ap.add_argument("--theta-mhz", type=float, default=SHEET_MHZ, help="the SM clock --theta is quoted at")
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--steady", action="store_true", help="drop steps longer than 1.5 times the median")
    ap.add_argument("--out")
    ap.add_argument("--table", metavar="JSON", help="print a written analysis's Markdown tables")
    a = ap.parse_args()
    if a.table:
        print(table(json.loads(Path(a.table).read_text())))
        return 0
    if not a.report or not a.tokens:
        ap.error("REPORT and --tokens are required")
    doc = analyse(a.report, a.tokens, a.prompt, json.loads(Path(a.config).read_text()), a.theta, a.target, a.steady,
                  a.theta_mhz)
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1))
    print(table(doc))
    return 0 if "error" not in doc else 1


if __name__ == "__main__":
    sys.exit(main())
