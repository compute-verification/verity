"""Serve-time commitment at the large model's batch: one layer's op outputs, each token row one hm96-sha512 hiding leaf in a
frame-v3-sha512 row table, hashed by the vLLM commit path's GPU row tables (`verity_vllm.commit.row_tables_device.step_roots`,
on its side stream), against the layer's exact linears at the same M on one GPU, and a whole decode step's tables in one call.

The table sets of a model (`fixture.MODELS`), per token row of one layer (the kernels hash a row's bytes, then a 4-byte word):
    quantize   the served commit's tables (`commit.row_tables`): the four quantizers' s8 codes, each with its f32 token scale
    fused      the fused cuts: the quantizers, every linear's bf16 output, attention's output and its
               probabilities (heads x context), the two residual adds
    every      every op's output: fused's, the two RMSNorms', the ReLU's and attention's scores
A bf16 row is hashed as its bytes with a zero word: this measures the throughput of those bytes, not the served row format.

Per model, set and M:
    check      at M in --check-ms, the device's roots equal the CPU twin's (`host_step`) for one layer, and at the smallest
               for the whole step in both layouts; at a small width the device, the twin and the Python reference
               (`row_tables.step_roots`) agree
    commit     the kernels' time on the side stream (`Pending.kernel_ms`) and the host's in `step_roots` (plan, domain ids,
               staging); bytes/s of the rows' values (B (K + 4) summed over tables); the roots equal rep to rep
    linears    the layer's exact linears (run.py's exact kernel at its picked config, each gated against `run.reference` on
               its first call) as one CUDA graph
    overlap    the serving order: a layer's tables hash while the next layer's linears run. Replays of the linears hide
               `step_roots`' host work; from their end the commit and one more replay run together: their joint time against
               the replay alone, and the gap before that replay starts; on the committer's side stream (default priority)
               and, as overlap_hi, on a stream of the highest priority (commit_gpu: the GEMM's queued CTAs go first
               otherwise)
    profile    each of the four kernels' device time (`row_table.cu`: k_segments, the 1 KiB segments' digests a thread
               each; k_rows, a row's hm96 leaf a thread; k_trees, a table's tree a CTA; k_step), from the CUDA profiler
    sha        SHA-512 compressions, ESTIMATED from commit_gpu's counts for row-seg/v1 (a segment's chunks and its padding
               chunk, the outer hash over the segments' digests, ~8 fixed a leaf, the salt being ChaCha20's), over the
               kernels' time, against the card's integer-issue ceiling of 7.37 G/s (benchmarks/pouw/commit_gpu/README.md)
The commit, linears and overlap items alternate rep by rep (order rotated), the SM clock read around each.

share = commit / linears in one layer. Its denominator is only the linears (norms, quantizers, attention are left out), so it
overstates the commit's share of a whole step. The whole step (the model's layers at M):
    whole_extrapolated   layers x one layer's commit and linears (EXTRAPOLATED from one layer)
    whole_measured       one `step_roots` call over every layer's tables, table-major (a table an op and layer, M rows),
                         the layers' buffers cycled over as many distinct copies as --whole-gb holds (`copies`)
    by_slot              the same rows slot-major, one table a sequence slot (its every op of every layer as rows) as one
                         subtree each, EMULATED with the kernels as they are: a table takes one row width, so every row has
                         the layer's mean width (value bytes equal to within rounding)
    need_gbs             the whole step's value bytes over the estimated step time (`STEP_S`), beside their
                         figures for mha8k: 134 GB/s every op, 93 fused

    python commit.py --out OUT [--models mqa16k,mha8k] [--ms 64,256,1023,4096] [--sets quantize,fused,every] [--reps 15]
Writes commit.jsonl (a row per model, set and M) and commit.json; exits 4 when a check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "hash" / "exactlin"))
import fixture as FX  # noqa: E402
import int_mm as IM  # noqa: E402
import kbench  # noqa: E402
import run as R  # noqa: E402
from verity_vllm.commit import row_tables as RTB  # noqa: E402
from verity_vllm.commit import row_tables_device as D  # noqa: E402
from verity_vllm.engine import int8_fused as X  # noqa: E402

CTX, KEY = bytes(range(64)), bytes(range(100, 132))
SETS = FX.TABLE_SETS
#: the estimated decode step (s): mha8k 195 ms, mqa16k 1.44 s
STEP_S = {"mha8k": 0.195, "mqa16k": 1.44}
#: the estimated committed-value rates for mha8k (GB/s): every op, fused cuts
NEED_MHA8K_GBS = {"every": 134.0, "fused": 93.0}
SHA_CEILING_G = 7.37
KERNELS = ("k_segments", "k_rows", "k_trees", "k_step")


def est_compressions(widths: list[int], m: int) -> tuple[int, int]:
    """(k_segments', k_rows' and the trees') estimated compressions of M rows of each width."""
    seg = rest = 0
    for k in widths:
        n, s = k + 4, -(-(k + 4) // D.SEG_BYTES)
        seg += -(-n // 128) + s
        rest += -(-(64 * s + 17) // 128) + 8
    return seg * m, rest * m


class Tables:
    """Tables of M rows each, (op, bytes a row, a scale word), codes random, a quantizer's scales random, other words zero."""

    def __init__(self, tables: list[tuple[str, int, bool]], m: int, seed: int, ids=None):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.ids = ids or [(f"r{i}", 0) for i in range(m)]
        self.tabs = {}
        for op, k, scaled in tables:
            c = torch.randint(-128, 128, (m, k), generator=g, dtype=torch.int16, device="cuda").to(torch.int8)
            s = torch.rand(m, generator=g, device="cuda") * 2.0**-7 if scaled else torch.zeros(m, device="cuda")
            self.tabs[op] = (c, s.to(torch.float32).contiguous())
        self.value_bytes = sum(c.numel() + 4 * m for c, _ in self.tabs.values())

    def outputs(self, prefix: str) -> dict:
        return {f"{prefix}{op}": (self.ids, c, s) for op, (c, s) in self.tabs.items()}


def host_copy(outputs: dict) -> dict:
    return {op: (ids, c.cpu().numpy(), s.cpu().numpy()) for op, (ids, c, s) in outputs.items()}


def tiny_check() -> dict:
    """At width 64 (two layers, M = 13: r10 ranks before r2, so rows go out of batch order): device == twin == reference."""
    md = FX.Model("tiny", 64, 2, (("qkv", 192, 64, "attn"), ("o", 64, 64, "o"), ("up", 256, 64, "mlp"),
                                  ("down", 64, 256, "down")), 2, 32, 5, 13)
    doc = {}
    for table_set in SETS:
        lay = Tables(FX.tables(md, table_set), 13, seed=5)
        outs = {**lay.outputs("model.layers.0."), **lay.outputs("model.layers.1.")}
        dev = D.step_roots(CTX, 2**33 + 7, outs, KEY).result()
        host = host_copy(outs)
        twin = D.host_step(CTX, 2**33 + 7, host, KEY)
        doc[table_set] = {"tables": len(outs), "device_is_twin": dev == twin,
                         "twin_is_reference": twin == RTB.step_roots(CTX, 2**33 + 7, host, KEY)}
    return doc


class Linears:
    """A layer's exact linears at M (weights without special scales), one CUDA graph."""

    def __init__(self, md: FX.Model, m: int):
        self.kern = R.kernel()
        self.calls, self.gate = [], {}
        for i, (name, n, k, _q) in enumerate(md.linears):
            wt = R.Weights(n, k, seed=3000 + i, special=False)
            x = R.codes(m, k, 41 + i, {})
            sa = torch.from_numpy(FX.act_scales(m, 41 + i, special=False)).cuda()
            y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            cfg = self.kern.pick(m, n, k)
            R.poison(y)
            self.kern.call(x, sa, wt.w, wt.sw, y, cfg)
            torch.cuda.synchronize()
            self.gate[name] = {"cfg": str(cfg), "mismatches": R.diff(y, R.reference(IM.int_mm(x, wt.w), sa, wt.sw))}
            self.calls.append((x, sa, wt, y, cfg))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._f()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._f()
        for _ in range(3):
            self.graph.replay()
        torch.cuda.synchronize()

    def _f(self):
        for x, sa, wt, y, cfg in self.calls:
            self.kern.call(x, sa, wt.w, wt.sw, y, cfg)

    @property
    def ok(self) -> bool:
        return all(g["mismatches"] == 0 for g in self.gate.values())


def _ev() -> torch.cuda.Event:
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def commit_once(outputs: dict, step: int) -> tuple[float, float, RTB.StepRoots]:
    t0 = time.perf_counter()
    p = D.step_roots(CTX, step, outputs, KEY, timing=True)
    host_us = (time.perf_counter() - t0) * 1e6
    roots = p.result()
    return p.kernel_ms() * 1e3, host_us, roots


def linears_once(lin: Linears) -> float:
    e0 = _ev()
    lin.graph.replay()
    e1 = _ev()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3


def overlap_once(lin: Linears, outputs: dict, step: int, pre: int, stream=None) -> dict:
    for _ in range(pre):
        lin.graph.replay()
    e0 = _ev()
    p = D.step_roots(CTX, step, outputs, KEY, stream=stream, timing=True)
    eb = _ev()
    lin.graph.replay()
    e2 = _ev()
    torch.cuda.current_stream().wait_event(p.done)
    e1 = _ev()
    torch.cuda.synchronize()
    return {"us": e0.elapsed_time(e1) * 1e3, "gap_us": e0.elapsed_time(eb) * 1e3, "linears_us": eb.elapsed_time(e2) * 1e3,
            "commit_us": p.kernel_ms() * 1e3, "roots": p.result()}


def profile(outputs: dict, step: int) -> dict:
    """Each row_table.cu kernel's device time (us) in one `step_roots` call, from torch's CUDA profiler."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as prof_ctx

    try:
        with prof_ctx(activities=[ProfilerActivity.CUDA]) as prof:
            D.step_roots(CTX, step, outputs, KEY).result()
            torch.cuda.synchronize()
        got = {k: 0.0 for k in KERNELS}
        other = 0.0
        for ev in prof.key_averages():
            us = float(getattr(ev, "device_time_total", 0) or getattr(ev, "cuda_time_total", 0) or 0)
            hit = next((k for k in KERNELS if k in ev.key), None)
            if hit:
                got[hit] += us
            elif us and "memcpy" in ev.key.lower():
                other += us
        return {**got, "memcpy": other}
    except Exception as ex:  # noqa: BLE001 - a profiler that doesn't see the library's kernels is a record
        return {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}


def sha(widths: list[int], m: int, kern: dict, total_us: float) -> dict:
    seg, rest = est_compressions(widths, m)
    doc = {"est_compressions": seg + rest, "est_g_per_s": (seg + rest) / total_us / 1e3, "ceiling_g_per_s": SHA_CEILING_G,
           "est_frac_ceiling": (seg + rest) / total_us / 1e3 / SHA_CEILING_G,
           "kernel_share": {k: kern[k] / total_us for k in KERNELS if isinstance(kern.get(k), float)}}
    if kern.get("k_segments"):
        doc["k_segments_est_g_per_s"] = seg / kern["k_segments"] / 1e3
        doc["k_segments_value_gbs"] = sum(k + 4 for k in widths) * m / kern["k_segments"] / 1e3
    return doc


def timed_calls(outputs: dict, step: int, reps: int) -> dict:
    commit_once(outputs, step)
    kern, host = [], []
    for _ in range(reps):
        us, h, _ = commit_once(outputs, step)
        kern.append(us)
        host.append(h)
    return {"commit_ms": statistics.median(kern) / 1e3, "host_ms": statistics.median(host) / 1e3,
            "commit_ms_all": [round(v / 1e3, 3) for v in kern], "host_ms_all": [round(v / 1e3, 3) for v in host]}


def measure(md: FX.Model, table_set: str, m: int, a, clock, step: int) -> dict:
    tables = FX.tables(md, table_set)
    widths = [k for _, k, _ in tables]
    lay = Tables(tables, m, seed=md.d + m)
    outs = lay.outputs("model.layers.0.")
    lin = Linears(md, m)
    row = {"model": md.name, "set": table_set, "m": m, "tables": len(outs), "value_bytes": lay.value_bytes,
           "row_bytes": {op: k + 4 for op, k, _ in tables}, "linear_gate": lin.gate}
    if not lin.ok:
        row["failed"] = "a linear's words differ from the reference"
        return row
    if m in a.check_ms:
        row["check_layer"] = D.step_roots(CTX, step, outs, KEY).result() == D.host_step(CTX, step, host_copy(outs), KEY)
    _, host_us, first = commit_once(outs, step)
    t_lin = linears_once(lin)
    pre = max(1, math.ceil(1.5 * host_us / max(t_lin, 1.0)))
    hi = torch.cuda.Stream(priority=-100)
    items = ("commit", "linears", "overlap", "overlap_hi")
    got: dict[str, list[float]] = {"host": []}
    for item in items[2:]:
        got.update({item: [], f"{item}_gap": [], f"{item}_linears": [], f"{item}_commit": []})
    got.update({"commit": [], "linears": []})
    clocks, stable = {item: [] for item in items}, True
    for r in range(a.reps):
        for j in range(len(items)):
            item = items[(j + r) % len(items)]
            c0 = clock()
            if item == "commit":
                us, h, roots = commit_once(outs, step)
                got["commit"].append(us)
                got["host"].append(h)
                stable &= roots == first
            elif item == "linears":
                got["linears"].append(linears_once(lin))
            else:
                o = overlap_once(lin, outs, step, pre, hi if item == "overlap_hi" else None)
                for key, src in ((item, "us"), (f"{item}_gap", "gap_us"), (f"{item}_linears", "linears_us"),
                                 (f"{item}_commit", "commit_us")):
                    got[key].append(o[src])
                stable &= o["roots"] == first
            clocks[item].append([c0, clock()])
    med = {k: statistics.median(v) for k, v in got.items()}
    kern = profile(outs, step)
    row.update({
        "roots_stable": stable, "pre_replays": pre, "us": med, "us_all": {k: [round(v, 2) for v in vs] for k, vs in got.items()},
        "sm_clock_mhz": clocks, "commit_gbs": lay.value_bytes / med["commit"] / 1e3, "share": med["commit"] / med["linears"],
        "overlap_slowdown": med["overlap"] / med["linears"] - 1, "overlap_hi_slowdown": med["overlap_hi"] / med["linears"] - 1,
        "kernels_us": kern, "sha": sha(widths, m, kern, med["commit"]),
        "layers": md.layers,
        "whole_extrapolated": {"label": "EXTRAPOLATED: layers x one layer", "commit_ms": md.layers * med["commit"] / 1e3,
                               "linears_ms": md.layers * med["linears"] / 1e3, "overlap_ms": md.layers * med["overlap"] / 1e3,
                               "value_bytes": md.layers * lay.value_bytes}})
    del lin
    torch.cuda.empty_cache()
    if a.whole:
        small = m == min(a.check_ms)
        row["whole_measured"] = whole(lay, tables, md, m, a, step + 1, small)
        row["by_slot"] = by_slot(tables, md, m, a, step + 2, small)
        vb = row["whole_measured"]["value_bytes"]
        row["need_gbs"] = {"step_s": STEP_S.get(md.name), "gbs": vb / STEP_S[md.name] / 1e9 if md.name in STEP_S else None,
                           "compute_accounting_mha8k": NEED_MHA8K_GBS.get(table_set) if md.name == "mha8k" else None}
    return row


def whole(lay: Tables, tables, md: FX.Model, m: int, a, step: int, check: bool) -> dict:
    copies = max(1, min(md.layers, int(a.whole_gb * 2**30 // lay.value_bytes)))
    bufs = [lay] + [Tables(tables, m, seed=7919 * (i + 1) + m) for i in range(copies - 1)]
    outs = {}
    for layer in range(md.layers):
        outs.update(bufs[layer % copies].outputs(f"model.layers.{layer}."))
    doc = {"layout": "table-major", "layers": md.layers, "copies": copies, "tables": len(outs),
           "value_bytes": md.layers * lay.value_bytes}
    if check:
        doc["check"] = D.step_roots(CTX, step, outs, KEY).result() == D.host_step(CTX, step, host_copy(outs), KEY)
    doc.update(timed_calls(outs, step, a.whole_reps))
    doc["commit_gbs"] = doc["value_bytes"] / doc["commit_ms"] / 1e6
    doc["kernels_us"] = profile(outs, step)
    doc["sha"] = sha([k for _, k, _ in tables], m * md.layers, doc["kernels_us"], doc["commit_ms"] * 1e3)
    del bufs, outs
    torch.cuda.empty_cache()
    return doc


def by_slot(tables, md: FX.Model, m: int, a, step: int, check: bool) -> dict:
    rows = len(tables) * md.layers
    k = 2 * math.ceil(sum(k for _, k, _ in tables) / len(tables) / 2)
    ids = [(f"u{j}", 0) for j in range(rows)]
    per_slot = rows * (k + 4)
    copies = max(1, min(m, int(a.whole_gb * 2**30 // per_slot)))
    bufs = [Tables([("rows", k, False)], rows, seed=104729 + i, ids=ids) for i in range(copies)]
    outs = {f"slot.{b:06d}": (ids, *bufs[b % copies].tabs["rows"]) for b in range(m)}
    doc = {"layout": "slot-major (EMULATED: one row width, the layer's mean)", "tables": m, "rows_a_table": rows,
           "row_bytes": k + 4, "copies": copies, "value_bytes": m * per_slot}
    if check:
        doc["check"] = D.step_roots(CTX, step, outs, KEY).result() == D.host_step(CTX, step, host_copy(outs), KEY)
    doc.update(timed_calls(outs, step, a.whole_reps))
    doc["commit_gbs"] = doc["value_bytes"] / doc["commit_ms"] / 1e6
    doc["kernels_us"] = profile(outs, step)
    doc["sha"] = sha([k], m * rows, doc["kernels_us"], doc["commit_ms"] * 1e3)
    del bufs, outs
    torch.cuda.empty_cache()
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(FX.DEFAULT_MODELS))
    ap.add_argument("--ms", default="64,256,1023,4096")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--check-ms", default="1,64")
    ap.add_argument("--whole", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--whole-gb", type=float, default=16.0)
    ap.add_argument("--whole-reps", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ints = lambda s: tuple(int(v) for v in s.split(",") if v)  # noqa: E731
    a.check_ms = ints(a.check_ms)
    ms = tuple(sorted(set(ints(a.ms)) | set(a.check_ms)))
    clock = kbench.nvml()
    meta = {"device": torch.cuda.get_device_name(), "arch": X.arch(), "torch": torch.__version__,
            "configs": {key: {f"{n},{k}": v for (n, k), v in t.items()} for key, t in R.CONFIGS.items()}, "step_s": STEP_S,
            "tiny": tiny_check()}
    print(json.dumps({"tiny": meta["tiny"]}), flush=True)
    failed = [s for s, v in meta["tiny"].items() if not (v["device_is_twin"] and v["twin_is_reference"])]
    rows, step = [], 1
    for md in map(FX.model, a.models.split(",")):
        for table_set in a.sets.split(","):
            for m in ms:
                t0 = time.time()
                row = measure(md, table_set, m, a, clock, step)
                step += 3
                row["s"] = round(time.time() - t0, 1)
                rows.append(row)
                with open(out / "commit.jsonl", "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                bad = ("failed" in row or row.get("check_layer") is False or not row.get("roots_stable", True)
                       or row.get("whole_measured", {}).get("check") is False or row.get("by_slot", {}).get("check") is False)
                if bad:
                    failed.append(f"{md.name}:{table_set}:m{m}")
                print(json.dumps({k: row.get(k) for k in ("model", "set", "m", "value_bytes", "commit_gbs", "share",
                                                          "overlap_slowdown", "overlap_hi_slowdown", "s")}
                                 | {"us": {k: round(v, 1) for k, v in row.get("us", {}).items()},
                                    "whole_gbs": row.get("whole_measured", {}).get("commit_gbs"),
                                    "by_slot_gbs": row.get("by_slot", {}).get("commit_gbs"), "bad": bad}), flush=True)
    keep = ("model", "set", "m", "tables", "value_bytes", "commit_gbs", "share", "overlap_slowdown", "overlap_hi_slowdown",
            "us", "kernels_us", "sha", "layers", "whole_extrapolated", "whole_measured", "by_slot", "need_gbs")
    doc = {**meta, "failed": failed, "ok": not failed, "rows": [{k: r.get(k) for k in keep} for r in rows]}
    (out / "commit.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
