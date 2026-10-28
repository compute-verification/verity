"""Where the card's capacity goes in a HIGHUTIL run, from period.py's summary.json and the protocol arm's kernel timeline
(timeline.py over the same settings), at θ under each reading of the card's clock side by side (the H200's tensor clock
and the boost clock nvidia-smi reports: `h200_step.CLOCKS_MHZ`; any other card, its card file's clock). At each reading
the terms sum to 1:

- accounted: the declared MACs at θ over the wall, the declared utilization;
- the plain GEMM's floor: the same run's plain cuBLASLt arm (at whatever clock the card holds there) past the MACs
  at θ, what no protocol could recover;
- the protocol: the protocol block's GPU time past the plain arm's, split into the GEMM beside the hashing (its GEMMs'
  time past the plain arm's), off the GEMM (the coins, forming and hashing with no GEMM running) and idle in a block;
- host between blocks: the protocol wall the GPU spends outside its blocks (coin round trips, the host's turn).

ε = ε_S + (1 - accounted), ε_S the sampling's (ε_skip). `credited` says what the accounted share counts. The certified
MACs, the declared MACs plus the peels the honest route computes (Z_X U^T and V Y_W^T, 2 m n L' a call), are reported
apart. With --periods and --smi, each period's SM clock (the blocks' samples) and nvidia-smi's over its wall-clock
window: the window is pinned on the challenger session's id (its UTC second) and trimmed a second at each end.

    terms.py SUMMARY TIMELINE [--periods PERIODS.jsonl --smi SMI.csv]      prints one JSON object

`terms.py served` reads a served step instead: one run of `benchmarks/one_stage/served_util.sh` (VLLM-33), over the window
its `served_util.py` chose (`util.json`'s steady full steps, each from the previous step's end to its own, host gaps
included), from the run's Nsight Systems export, step trace and serve record:

- **u**, the tensor utilization: the window's tensor work, each format at its own peak (the int8 linears at θ_int8, the
  bf16 attention and LM head at the H200's bf16 rate, half θ_int8), over the wall. It's given at each reading of the clock,
  and at the window's measured GPC clock beside Tensor Active, Nsight's share of SM cycles with the tensor pipe active.
  The MACs are counted from the trace's rows: the linears' m n k at m the step's tokens, attention's causal QK^T and PV of
  each row (its offset o and length n: 2 D (n o + n (n + 1) / 2) a head and layer), and the LM head on each row that
  emits a token.
- **the accounted share**, the declared linears' MACs (QKV, O, gate-up and down; not the LM head, which H1 doesn't
  declare) at θ_int8 over the wall, labelled as `--credit` says, with the certified MACs (`--certified-over-declared`
  times the declared) beside it.
- **the rest**, the wall split exclusively, summing to 1. Each instant goes to the first class in `KERNEL_CLASSES` of the
  kernels running then. With no kernel, it goes to a copy or memset that is running. With neither, it goes to a gap by
  the gap's length: launch (under 20 µs), host (20 µs to 2 ms), or idle (2 ms or more).

    terms.py served --sqlite PROF.sqlite --trace ADVICE_TRACE.jsonl --util UTIL.json --serve SERVE.json
        --certified-over-declared R [--credit TEXT] [--key all]      prints one JSON object and, with --text, a table
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))

import h200_step

#: the SM clock of each reading by card SKU; a card not named here has its card file's clock as its one reading
READINGS = {"h200-sxm-141gb": h200_step.CLOCKS_MHZ}

CREDITED = ("the declared MACs alone: each accepted call's m n k, the served or filler product. Tensor-only accounting "
            "credits nothing the protocol does (the peels, the hashing, the coins, the forming), the host's time or the "
            "GEMM's shortfall below θ: an accepted prover can run hidden tensor work beside each. The certified "
            "MACs (the declared plus the peels) are reported apart and not credited.")


def readings(card: dict) -> dict[str, float]:
    return dict(READINGS.get(card["sku"], {"card": card["clock_mhz"]}))


def macs_per_step(workload: dict) -> dict:
    m, lp = workload["m"], workload["lp"]
    declared = workload["step_macs"]
    peels = sum(2 * m * n * (lp.get(name) or 0) for name, n, _ in workload["layers"])
    return {"declared": declared, "peels": peels, "certified": declared + peels,
            "certified_over_declared": (declared + peels) / declared}


def terms(summary: dict, timeline: dict, periods: list[dict] | None = None, smi: list | None = None) -> dict:
    card = summary["card"]
    if not card:
        raise SystemExit("terms.py: summary.json has no card (period.py run --card)")
    lt = summary.get("baselines", {}).get("lt", {})
    wall, gpu = summary["proto_s_per_step"], summary["proto_gpu_s_per_step"]
    plain = lt.get("plain_gpu_s_per_step") or summary["plain_gpu_s_per_step"]
    gemm, busy = timeline["gemm_share"] * gpu, timeline["busy_share"] * gpu
    macs = macs_per_step(summary["workload"])
    out = {"card": card["sku"], "credited": CREDITED, "macs_per_step": macs, "readings": {},
           "eps_skip": summary["eps_skip"], "delta": summary["delta"],
           "ms_per_step": {"wall": wall * 1e3, "gpu": gpu * 1e3, "plain_gpu": plain * 1e3,
                           "protocol_gemm": gemm * 1e3, "protocol_busy": busy * 1e3}}
    for name, mhz in readings(card).items():
        theta = card["tensor"] * card["sms"] * mhz * 1e6
        ideal = macs["declared"] / theta
        proto = {"GEMM beside the hashing": (gemm - plain) / wall, "off the GEMM": (busy - gemm) / wall,
                 "idle in a block": (gpu - busy) / wall}
        t = {"accounted": ideal / wall, "the plain GEMM's floor": (plain - ideal) / wall,
             "the protocol": sum(proto.values()), "host between blocks": (wall - gpu) / wall}
        out["readings"][name] = {
            "clock_mhz": mhz, "theta_macs_s": theta, "terms": t, "protocol": proto, "sum": sum(t.values()),
            "eps_undecl": 1 - t["accounted"], "eps": summary["eps_skip"] + 1 - t["accounted"],
            "certified_share": macs["certified"] / theta / wall, "plain_share": ideal / plain,
            "ideal_ms_per_step": ideal * 1e3}
    if periods is not None:
        out["clocks"] = clocks(periods, smi or [])
    return out


def smi_samples(text: str) -> list[tuple[datetime, dict[str, float]]]:
    """nvidia-smi's CSV (run.sh's: timestamp, clocks.sm, clocks.gr, clocks.mem, ...) as (UTC time, values)."""
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return []
    head = [h.strip() for h in rows[0]]
    col = {k: next((i for i, h in enumerate(head) if h.startswith(p)), None)
           for k, p in (("sm_mhz", "clocks.current.sm"),)}
    out = []
    for r in rows[1:]:
        if len(r) != len(head) or not re.fullmatch(r"\d{4}/\d\d/\d\d \d\d:\d\d:\d\d\.\d+", r[0].strip()):
            continue
        vals = {k: re.match(r"\s*(\d+(?:\.\d+)?)\b", r[i]) for k, i in col.items() if i is not None}
        if vals and all(vals.values()):
            t = datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
            out.append((t, {k: float(v.group(1)) for k, v in vals.items()}))
    return out


def windows(periods: list[dict]) -> list[tuple[datetime, datetime] | None]:
    """Each period's wall-clock window: a protocol period's opens at its session's id plus the session's opening, and
    the plain periods before it end where the next begins; None where nothing pins it."""
    out: list[tuple[datetime, datetime] | None] = [None] * len(periods)
    for i, r in enumerate(periods):
        m = re.match(r"s(\d{8}T\d{6})Z", (r.get("coins") or {}).get("session") or "")
        if r["arm"] != "proto" or not m:
            continue
        t = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        start = t + timedelta(seconds=r["session_open_s"])
        out[i] = (start, start + timedelta(seconds=r["wall_s"]))
        end, j = t, i - 1
        while j >= 0 and periods[j]["arm"] == "plain":
            out[j] = (end - timedelta(seconds=periods[j]["wall_s"]), end)
            end, j = out[j][0], j - 1
    return out


def clocks(periods: list[dict], smi: list) -> list[dict]:
    out = []
    for r, w in zip(periods, windows(periods)):
        mhz = [b["sm_mhz"] for b in r["blocks"] if b.get("sm_mhz")]
        row = {"period": r["period"], "arm": r["arm"], "baseline": r.get("baseline"), "cheat": r.get("cheat"),
               "wall_s": r["wall_s"], "blocks_sm_mhz_median": statistics.median(mhz) if mhz else None, "smi": None}
        if w:
            lo, hi = w[0] + timedelta(seconds=1), w[1] - timedelta(seconds=1)
            got = [v for t, v in smi if lo <= t <= hi]
            if got:
                row["smi"] = {"samples": len(got), "clocks_sm_mhz": statistics.median(v["sm_mhz"] for v in got)}
        out.append(row)
    return out


#: each served checkpoint's shapes, from its config.json: the linears PoUW declares as (K, N) a layer (H1's Σ N over them
#: and the layers is its declaration's), attention's query heads and head dimension, and the LM head's hidden size and vocabulary
SERVED_MODELS = {"verity-int8/Llama-3.2-1B-INT8": {
    "layers": 16, "linears": {"qkv": (2048, 3072), "o": (2048, 2048), "gate_up": (2048, 16384), "down": (8192, 2048)},
    "heads": 32, "head_dim": 64, "hidden": 2048, "vocab": 128256}}
H200_SMS, INT8_PER_SM_CLOCK = 132, 4096
#: the H200's dense bf16 tensor rate over its int8 rate (989.4 against 1,978.9 T/s at 1,830 MHz)
BF16_OVER_INT8 = 0.5
UNCREDITED = "uncredited: development key, credit not established"
#: kernel classes, matched in order against a kernel's name (lower case); an instant with kernels of several classes running
#: goes to the first, so nothing beside the tensor work hides it
KERNEL_CLASSES = (
    ("int8 GEMM (declared linears)", r"scaled_mm|igmma|imma|s8gemm|i8gemm"),
    ("attention (bf16 FA3)", r"flashattn|flash_fwd|flash_attn|fmha|prepare_varlen_num_blocks"),
    ("attention: KV-cache write", r"reshape_and_cache"),
    ("bf16 GEMM (LM head)", r"matmul_kernel_persistent|gemv|nvjet|sm90_xmma|cublas|cutlass.*gemm|gemm"),
    ("the Commit's hash and leaf kernels", r"sha512|sha_|hm96|merkle|frame|leaf|tree|digest|hash|level_kernel|top_kernel|commit"),
    ("the Commit's taps", r"tap|collector|readback|snapshot|hidden|matreq"),
    ("PoUW H1", r"(^|_)h1|pouw|noise|duo|wrap|bits_kernel|perm_kernel|salt_|fill_kernel|forge|open_kernel|pack_kernel|form_"),
    ("quantize (int8 rows)", r"quant"),
    ("norms", r"norm"),
    ("sampling", r"sampl|gumbel|argmax|topk|top_k|top_p|logit|softmax|penalt"),
    ("elementwise", r"act_and_mul|silu|gelu|rotary|rope|elementwise|vectorized|unrolled|add|mul|copy|cat|index|gather|"
                    r"scatter|fill|contiguous|transpose|arange|reduce"),
    ("other kernels", r""),
)
COPIES = {1: "H2D copy, no kernel running", 2: "D2H copy, no kernel running"}
OTHER_COPY = "other copy or memset, no kernel running"
GAPS = ((20e3, "launch gaps (GPU idle under 20 µs)"), (2e6, "host gaps (GPU idle 20 µs to 2 ms)"),
        (float("inf"), "idle (GPU idle 2 ms or more)"))
#: short names that say nothing of the kernel (a template's entry point): such a kernel goes by its demangled name
GENERIC = re.compile(r"^(device_kernel|kernel|Kernel\d*|cutlass_kernel|generic_kernel)$")
#: the GPU metrics read: the tensor pipe's active share and the clock it ran at
METRICS = ("Tensor Active", "GPC Clock Frequency")


def kernel_class(name: str) -> int:
    low = name.lower()
    return next(i for i, (_, pat) in enumerate(KERNEL_CLASSES) if re.search(pat, low))


def _has(db: sqlite3.Connection, table: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def served_profile(db: sqlite3.Connection) -> dict:
    """The export's timeline zero (UTC ns), kernels (start, end, name), copies and memsets (start, end, kind; memsets kind
    0) and GPU metric series ({name: (timestamps, values)})."""
    cols = {r[1] for r in db.execute("PRAGMA table_info(TARGET_INFO_SESSION_START_TIME)")}
    zero = int(db.execute(f"SELECT {'utcEpochNs' if 'utcEpochNs' in cols else 'utcTime'} FROM TARGET_INFO_SESSION_START_TIME")
               .fetchone()[0])
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = []
    for a, b, s, d in db.execute("SELECT start, end, shortName, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"):
        name = str(strings.get(s, s))
        ks.append((int(a), int(b), str(strings.get(d, d)) if GENERIC.match(name) else name))
    cs = [(int(a), int(b), int(k)) for a, b, k in db.execute("SELECT start, end, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY")] \
        if _has(db, "CUPTI_ACTIVITY_KIND_MEMCPY") else []
    if _has(db, "CUPTI_ACTIVITY_KIND_MEMSET"):
        cs += [(int(a), int(b), 0) for a, b in db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET")]
    names = {int(i): str(n).split(" [")[0] for i, n in db.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS")
             if str(n).startswith(METRICS)} if _has(db, "TARGET_INFO_GPU_METRICS") else {}
    series: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    if names:
        q = f"SELECT timestamp, metricId, value FROM GPU_METRICS WHERE metricId IN ({','.join(map(str, names))}) ORDER BY timestamp"
        for ts, mid, v in db.execute(q):
            s = series[names[int(mid)]]
            s[0].append(int(ts))
            s[1].append(float(v))
    return {"zero": zero, "kernels": ks, "copies": sorted(cs), "metrics": dict(series)}


def split_wall(kernels: list, copies: list, lo: int, hi: int) -> tuple[dict[str, float], dict[str, list]]:
    """[lo, hi] split exclusively (ns by label, summing to hi - lo), and each kernel class's kernels by time in the window."""
    ev = []
    for a, b, name in kernels:
        if b > lo and a < hi:
            ev += [(max(a, lo), 1, 0, kernel_class(name)), (min(b, hi), -1, 0, kernel_class(name))]
    for a, b, kind in copies:
        if b > lo and a < hi:
            ev += [(max(a, lo), 1, 1, kind), (min(b, hi), -1, 1, kind)]
    ev.sort(key=lambda e: (e[0], e[1]))
    out: dict[str, float] = defaultdict(float)
    kc, cc = [0] * len(KERNEL_CLASSES), defaultdict(int)
    t, gap = lo, 0

    def close_gap() -> None:
        nonlocal gap
        if gap:
            out[next(label for limit, label in GAPS if gap < limit)] += gap
            gap = 0
    for when, d, is_copy, c in [*ev, (hi, 0, 0, 0)]:
        if when > t:
            dt = when - t
            k = next((i for i, n in enumerate(kc) if n), None)
            if k is not None:
                close_gap()
                out[KERNEL_CLASSES[k][0]] += dt
            elif any(cc.values()):
                close_gap()
                kind = next((x for x in (2, 1) if cc[x]), None)
                out[COPIES[kind] if kind else OTHER_COPY] += dt
            else:
                gap += dt
            t = when
        if is_copy:
            cc[c] += d
        else:
            kc[c] += d
    close_gap()
    top: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for a, b, name in kernels:
        if b > lo and a < hi:
            top[KERNEL_CLASSES[kernel_class(name)][0]][name[:100]] += min(b, hi) - max(a, lo)
    return dict(out), {c: sorted(v.items(), key=lambda kv: -kv[1])[:4] for c, v in top.items()}


def step_macs(step: dict, model: dict) -> dict[str, int]:
    """One traced step's MACs: the declared linears (int8), attention's causal QK^T and PV and the LM head (bf16)."""
    rows = [(int(o), int(n)) for _, o, n in step["rows"]]
    tokens = sum(n for _, n in rows)
    per_token = model["layers"] * sum(k * n for k, n in model["linears"].values())
    attn = model["layers"] * model["heads"] * 2 * model["head_dim"] * sum(n * o + n * (n + 1) // 2 for o, n in rows)
    emitting = len(step.get("emitted") or {})
    return {"tokens": tokens, "declared": tokens * per_token, "attention": attn,
            "lm_head": emitting * model["hidden"] * model["vocab"]}


def _mean(series: tuple[list[int], list[float]] | None, lo: int, hi: int) -> float | None:
    if not series:
        return None
    ts, vs = series
    a, b = bisect.bisect_left(ts, lo), bisect.bisect_right(ts, hi)
    return statistics.fmean(vs[a:b]) if b > a else None


def served(prof: dict, trace: list[dict], util: dict, serve: dict, ratio: float, credit: str, key: str = "all") -> dict:
    w = util.get(key) or {}
    if not w.get("first_step"):
        raise SystemExit(f"terms.py served: util.json has no window {key!r} of two or more full steps")
    model = SERVED_MODELS.get(serve.get("model"))
    if model is None:
        raise SystemExit(f"terms.py served: no shapes for {serve.get('model')!r}: add its config to SERVED_MODELS")
    st = {r["step"]: r for r in trace if r.get("kind") == "step" and "t" in r}
    first, last = w["first_step"], w["last_step"]
    at = lambda s: int(s * 1e9) - prof["zero"]          # noqa: E731  (a wall-clock second on the export's timeline)
    lo, hi = at(st[first - 1]["t"]), at(st[last]["t"])
    wall = (hi - lo) / 1e9
    macs: dict[str, int] = defaultdict(int)
    for s in range(first, last + 1):
        for k, v in step_macs(st[s], model).items():
            macs[k] += v
    bf16 = macs["attention"] + macs["lm_head"]
    ta = _mean(prof["metrics"].get("Tensor Active"), lo, hi)
    gpc = _mean(prof["metrics"].get("GPC Clock Frequency"), lo, hi)
    gpc_mhz = gpc / 1e6 if gpc and gpc > 1e5 else gpc
    clocks = {**h200_step.CLOCKS_MHZ, **({"measured GPC": round(gpc_mhz, 1)} if gpc_mhz else {})}
    readings_ = {}
    for name, mhz in clocks.items():
        rho = INT8_PER_SM_CLOCK * H200_SMS * mhz * 1e6
        tensor_s = macs["declared"] / rho + bf16 / (BF16_OVER_INT8 * rho)
        readings_[name] = {"clock_mhz": mhz, "theta_int8_macs_s": rho, "u": tensor_s / wall,
                           "u_int8_linears": macs["declared"] / rho / wall, "u_bf16": bf16 / (BF16_OVER_INT8 * rho) / wall,
                           "accounted": macs["declared"] / rho / wall, "certified_share": ratio * macs["declared"] / rho / wall}
    split, top = split_wall(prof["kernels"], prof["copies"], lo, hi)
    shares = {k: v / (hi - lo) for k, v in sorted(split.items(), key=lambda kv: -kv[1])}
    sv = util.get("serve") or {}
    wins = sv.get("windows") or []
    return {
        "arm": util.get("arm"), "T": util.get("T"), "model": serve.get("model"), "window": key, "steps": last - first + 1,
        "first_step": first, "last_step": last, "wall_s": wall, "step_wall_ms": w.get("step_wall_ms"),
        "credit": credit, "macs": dict(macs) | {"bf16": bf16, "certified": ratio * macs["declared"]},
        "certified_over_declared": ratio,
        "tensor_active": ta / 100 if ta is not None else None, "readings": readings_,
        "shares": shares, "shares_sum": sum(shares.values()),
        "top_kernels": {c: [{"name": n, "share": ns / (hi - lo)} for n, ns in v] for c, v in top.items()},
        "windows": {"closed": len(wins), "passed": sum(1 for r in wins if r.get("passed")),
                    "committed_GB": sum(r.get("committed_bytes") or 0 for r in wins) / 1e9,
                    "seconds": sv.get("window_seconds")} if wins else None,
        "commit": serve.get("commit"), "protocols": serve.get("protocols"),
    }


def served_text(r: dict) -> str:
    rd = r["readings"]
    lines = [f"{r['arm']} T {r['T']}: {r['steps']} full steps {r['first_step']}–{r['last_step']}, wall {r['wall_s']:.4f} s "
             f"({1e3 * r['wall_s'] / r['steps']:.2f} ms a step)",
             "u (tensor work at peak over wall): " + ", ".join(f"{v['u']:.2%} at {v['clock_mhz']:,} MHz ({k})" for k, v in rd.items())
             + (f"; Tensor Active {r['tensor_active']:.2%}" if r["tensor_active"] is not None else ""),
             "accounted (declared MACs at θ_int8 over wall): " + ", ".join(f"{v['accounted']:.2%} ({k})" for k, v in rd.items())
             + f" [{r['credit']}]",
             "certified MACs at " + f"{r['certified_over_declared']:.3f}× declared: "
             + ", ".join(f"{v['certified_share']:.2%} ({k})" for k, v in rd.items()),
             f"MACs over the window: declared {r['macs']['declared']:.4g}, certified {r['macs']['certified']:.4g}, attention "
             f"{r['macs']['attention']:.4g}, LM head {r['macs']['lm_head']:.4g}",
             f"the wall, exclusively (sums to {r['shares_sum']:.4f}):"]
    for k, v in r["shares"].items():
        tops = "; ".join(f"{x['name'][:60]} {x['share']:.2%}" for x in r["top_kernels"].get(k, [])[:2])
        lines.append(f"  {v:7.2%}  {k}" + (f"  ({tops})" if tops else ""))
    if r["windows"]:
        lines.append(f"windows: {r['windows']}")
    return "\n".join(lines)


def served_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="terms.py served", description="a served step's u, accounted share and the rest")
    ap.add_argument("--sqlite", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--util", required=True, type=Path)
    ap.add_argument("--serve", required=True, type=Path)
    ap.add_argument("--certified-over-declared", required=True, type=float)
    ap.add_argument("--credit", default=UNCREDITED)
    ap.add_argument("--key", default="all")
    ap.add_argument("--text", type=Path, help="also write the table here")
    a = ap.parse_args(argv)
    trace = [json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()]
    r = served(served_profile(sqlite3.connect(a.sqlite)), trace, json.loads(a.util.read_text()),
               json.loads(a.serve.read_text()), a.certified_over_declared, a.credit, a.key)
    print(json.dumps(r, indent=1))
    if a.text:
        a.text.write_text(served_text(r) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["served"]:
        return served_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary")
    ap.add_argument("timeline")
    ap.add_argument("--periods", help="periods.jsonl, for each period's clock")
    ap.add_argument("--smi", help="run.sh's smi.csv")
    a = ap.parse_args(argv)
    s, tl = (json.loads(Path(x).read_text()) for x in (a.summary, a.timeline))
    periods = [json.loads(x) for x in Path(a.periods).read_text().splitlines() if x.strip()] if a.periods else None
    smi = smi_samples(Path(a.smi).read_text()) if a.smi and Path(a.smi).exists() else []
    print(json.dumps(terms(s, tl, periods, smi), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
