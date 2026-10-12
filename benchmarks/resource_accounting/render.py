"""The matmul chain's table, rendered on demand from a recorded run's ``chain.json`` (`chain.py`) and the code: nothing it
prints is stored, in Git or the evidence store.

  python render.py art:XXXXXXXX [--json]      the run's tree from the evidence store (``research data fetch``)
  python render.py OUT/chain.json [--json]

Per point and arm: the chain's time (the median over reps), its overhead over its divisor (`sweep.DIVISOR`), useful TFLOP/s
(2·B·n·k·L over the time, the batch's real rows only), the native GEMM rate at the shape (the native arms' GEMMs alone), and
the device-memory rate the chain achieved, estimated as its weights' and activations' bytes read once over its time (a lower
bound: the weights are at least twice the L2).  For the Pearl-C arms, `certified`'s columns: W_cert (as TFLOP/s-equivalent,
2·W_cert over the time) and the compute residual's split as shares of F·T, with F the native FP8 chain's rate; Q_cert is
pending (times and raw rates only) and S_cert says why it is unproved.  Every residual is the weak form (`certified.RESIDUAL`).
The retained pass's verdicts, when the run kept one, head the table.  A run with a PoUS arm (`chain.beside`) adds its section:
the honest audit's answers against the limit while the chain ran, S_cert from the harness's storage profile with the
capacity residual (`certified.storage`), and each chain's time beside the audit.  A run that hosted the harness
(`chain.hosted`, ``run.sh JOINT=``) adds the responder's isolation, one row a round, and the joint round's arms (d), the
tag pass over the PoUS store alone, and (e), its windows beside the chain and the audit.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import certified as C
import sweep as S

SCHEMA = "ra-matmul-chain/v1"
JOINT_TAG = "ro-blake3/v1"


def load(source: str) -> tuple[dict, dict, dict | None, dict[str, dict]]:
    """The run's chain.json, its verify verdicts (``verify/*.txt`` beside it), the PoUS harness's result
    (``pous/bench/bench-result.json`` beside it, when the run had a PoUS arm) and each hosted round's
    (``pous/bench/<round>/bench-result.json``), from a path or an ``art:`` id."""
    if source.startswith("art:"):
        d = Path(tempfile.mkdtemp(prefix="ra-render-"))
        subprocess.run(["research", "data", "fetch", source, "--to", str(d), "--path", "*chain.json", "--path", "*verify/*.txt",
                        "--path", "*pous/bench/bench-result.json", "--path", "*pous/bench/*/bench-result.json"], check=True,
                       stdout=subprocess.DEVNULL)
        found = sorted(d.rglob("chain.json"))
        if not found:
            raise SystemExit(f"{source}: no chain.json in the tree")
        path = found[0]
    else:
        path = Path(source)
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {doc.get('schema')!r}, not {SCHEMA}")
    verdicts = {}
    for name in ("pass", "control"):
        f = path.parent / "verify" / f"{name}.txt"
        if f.exists():
            lines = [x for x in f.read_text().splitlines() if x.startswith(("ACCEPT", "REJECT"))]
            verdicts[name] = lines[-1] if lines else None
    bench = path.parent / "pous" / "bench"
    result = bench / "bench-result.json"
    rounds = {name: json.loads((bench / name / "bench-result.json").read_text())
              for name in (doc.get("rounds") or {}).get("order") or [] if (bench / name / "bench-result.json").exists()}
    return doc, verdicts, json.loads(result.read_text()) if result.exists() else None, rounds


def bytes_read(p: dict, arm: str) -> int:
    """A chain's weights and activations, each read once: L·n·k at the arm's weight bytes, and per call B·k inputs (E4M3 for
    the FP8 arm, BF16 otherwise) and B·n BF16 outputs."""
    x = 1 if arm == "native-fp8" else 2
    return p["length"] * (p["n"] * p["k"] * S.WEIGHT_BYTES[arm] + p["batch"] * (p["k"] * x + p["n"] * 2))


def rows(doc: dict) -> list[dict]:
    out = []
    read_rate = (doc.get("read_rate") or {}).get("bytes_per_s")
    memory = (doc.get("identity") or {}).get("memory_bytes")
    gammas: dict[tuple[int, int], dict] = {}
    for p in doc["points"]:
        if "arms" not in p:
            out.append({"key": p["key"], "error": (p.get("error") or p.get("skipped") or "").strip().splitlines()[-1:]})
            continue
        t = {a: r["median_s"] for a, r in p["arms"].items()}
        fp8 = t.get("native-fp8")
        F = p["macs"] / fp8 if fp8 else None
        for arm in (a for a in S.ARMS if a in t):
            r, T = p["arms"][arm], t[arm]
            div = t.get(S.DIVISOR[arm])
            row = {"key": p["key"], "shape": p["shape"], "n": p["n"], "k": p["k"], "batch": p["batch"], "length": p["length"],
                   "arm": arm, "mode": r.get("mode", "graph"), "chains": r["chains"], "chain_ms": T * 1e3,
                   "spread": (max(r["seconds"]) - min(r["seconds"])) / T if r["seconds"] else None,
                   "overhead": None if div is None or arm == S.DIVISOR[arm] else T / div - 1,
                   "useful_tflops": 2 * p["macs"] / T / 1e12,
                   "gemm_tflops": 2 * p["macs"] / t[arm + "-gemm"] / 1e12 if arm + "-gemm" in t else None,
                   "bytes_per_s": bytes_read(p, arm) / T,
                   "chain_rel_rms": (p.get("chain_rel_rms_vs_bf16") or {}).get(arm),
                   "roots_repeat": r.get("roots_repeat")}
            row["read_share"] = None if not read_rate else row["bytes_per_s"] / read_rate
            if arm in S.POUW:
                if arm != "pearl-c-nohash" and "screened_rows" in r:
                    credited = (r["screened_rows"] - r["failing_rows"] - r["unsure_rows"]) / r["chains"]
                    g = gammas.setdefault((p["n"], p["k"]), C.gamma(p["n"], p["k"]))
                    w = C.compute(p["n"], p["k"], round(credited), T, F, g)
                    row["credited_share"] = credited / (p["batch"] * p["length"])
                    row["w_cert"], row["w_unproved"] = w["w_cert"], w["unproved"]
                    row["w_cert_tflops"] = None if w["w_cert"] is None else 2 * w["w_cert"] / T / 1e12
                    row["gamma"] = w["gamma"]
                    for part in ("residual", "honest_overhead", "soundness_slack"):
                        row[f"compute_{part}"] = None if w[part] is None or not w["capacity"] else w[part] / w["capacity"]
                bw = C.bandwidth(read_rate, T, bytes_read(p, arm))
                row["q_pending"] = bw["pending"]
                row["bandwidth_honest_overhead"] = (None if bw["honest_overhead"] is None or not bw["capacity"]
                                                    else bw["honest_overhead"] / bw["capacity"])
                cap = C.capacity(memory)
                row["s_unproved"] = cap["unproved"]
            out.append(row)
    return out


def harness(result: dict | None) -> dict:
    """The PoUS harness's ``bench-result.json``, cut to its honest audit: answers against the limit fixed before the load,
    and its gates."""
    result = result or {}
    m = result.get("metrics") or {}
    audit, deadline = m.get("audit") or {}, m.get("deadline") or {}
    return {"passed": result.get("passed"),
            "failed_gates": sorted(g for g, v in (result.get("gates") or {}).items() if not (v or {}).get("passed")),
            "audits": audit.get("audits"), "accepted": audit.get("audits_accepted"), "answers": audit.get("answers"),
            "late": audit.get("late"), "wrong": audit.get("wrong"), "limit_ns": deadline.get("limit_ns"),
            "limit_fixed_before_load": deadline.get("limit_fixed_before_load"), "elapsed_s": audit.get("elapsed_s") or {},
            "margin_s": audit.get("margin_s") or {}, "rtt_s": (audit.get("calibration_here") or {}).get("rtt_s") or {}}


def beside_rows(points: dict) -> list[dict]:
    """Each point's arms in an audit's window (`chain.Turns.summary`) against their time alone before and after it."""
    return [{"key": key, "arm": arm, "items": len(r["items"]), "alone_ms": r["alone_median_s"] * 1e3,
             "beside_ms": r["median_s"] * 1e3, "slowdown": r["slowdown"],
             "after_ms": None if r.get("after_median_s") is None else r["after_median_s"] * 1e3,
             "slowdown_after": None if not r.get("after_median_s") else r["median_s"] / r["after_median_s"] - 1}
            for key, by_arm in points.items() for arm, r in by_arm.items()]


def window_s(r: dict) -> float | None:
    if r.get("window_end_ns") is None or r.get("window_start_ns") is None:
        return None
    return (r["window_end_ns"] - r["window_start_ns"]) / 1e9


def pous(doc: dict, result: dict | None) -> dict | None:
    """The PoUS arm (`chain.beside`): the harness's honest audit while the chain ran (its answers against the limit fixed
    before), its gates, S_cert with the capacity residual, and each chain's time beside the audit against its time alone."""
    b = doc.get("beside")
    if b is None and result is None:
        return None
    b = b or {}
    st = C.storage(result or None)
    memory = (doc.get("identity") or {}).get("memory_bytes")
    return {"error": b.get("error"), "idle": b.get("idle"), "window_closed_by": b.get("window_closed_by"),
            "harness_rc": b.get("harness_rc"), "harness_stopped": b.get("harness_stopped"), "left_cpus": b.get("left_cpus"),
            "window_s": window_s(b), **harness(result), "storage": st, "memory_bytes": memory,
            "capacity": C.capacity(memory, st["s_cert"], st["unproved"] or C.NO_POUS, st["stored"]),
            "beside": beside_rows(b.get("points") or {})}


def _pct(xs: list[float], q: float) -> float | None:
    """The nearest-rank `q` quantile."""
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))] if xs else None


def proof_reads(d: dict | None, error: str | None = None) -> dict:
    """(d) (`joint.alone`): E's raw read, and each tag's passes (device and host time, the rate over E and its share of the
    raw read's), its windows on time at T and accepted, and whether its skip control was rejected; `error` when it failed."""
    if d is None:
        return {"error": error}
    E, raw = d["store"]["bytes"], statistics.median(d["raw_read"]["device_ms"])
    out = {"store_bytes": E, "blocks": d["store"]["blocks"], "raw_ms": raw, "raw_gb_per_s": E / raw / 1e6,
           "vk_s": d.get("vk_s"), "setup_s": d.get("setup_s"), "tags": []}
    for name, t in sorted(d["tags"].items()):
        dev = [p["device_ms"] for p in t["passes"]]
        host = [p["host_ns"] / 1e6 for p in t["passes"]]
        p50, w, skip = _pct(dev, 0.5), t["windows"], t["skip"]
        out["tags"].append({"tag": name, "certified": t["certified"], "passes": len(dev), "device_ms_p50": p50,
                            "device_ms_p99": _pct(dev, 0.99), "tags_ms_p50": _pct([p.get("device_tags_ms") for p in t["passes"]], 0.5),
                            "host_ms_p50": _pct(host, 0.5), "host_ms_p99": _pct(host, 0.99),
                            "gb_per_s": None if not p50 else E / p50 / 1e6, "of_raw": None if not p50 else raw / p50,
                            "windows": len(w), "on_time_at_T": sum(x["on_time_at_T"] for x in w),
                            "accepted": sum(x["accepted"] for x in w), "deadline_ns": t["deadline_ns"], "limit_ns": t["limit_ns"],
                            "window_ms_max": _pct([x["elapsed_ns"] / 1e6 for x in w if x["elapsed_ns"] is not None], 1.0),
                            "skip_fraction": skip["fraction"], "skip_rejected": not skip["accepted"],
                            "skip_failure": skip["failure"]})
    return out


def joint_windows(w: dict, alone: dict | None, overlap: dict | None) -> dict:
    """(e) (`joint.Joint`): the windows beside the chain and the audit, on time at T and accepted, their passes' device time
    against the same tag's alone, and the chain's items that overlapped a window against those that did not."""
    ws = w.get("windows") or []
    passes = (((alone or {}).get("tags") or {}).get(w.get("tag")) or {}).get("passes") or []
    return {"tag": w.get("tag"), "windows": len(ws), "on_time_at_T": sum(x["on_time_at_T"] for x in ws),
            "accepted": sum(x["accepted"] for x in ws), "failures": dict(Counter(x["failure"] for x in ws if x["failure"])),
            "device_ms_p50": _pct([x["device_ms"] for x in ws], 0.5), "device_ms_max": _pct([x["device_ms"] for x in ws], 1.0),
            "alone_device_ms_p50": _pct([p["device_ms"] for p in passes], 0.5),
            "window_ms_p50": _pct([x["elapsed_ns"] / 1e6 for x in ws if x["elapsed_ns"] is not None], 0.5),
            "window_ms_max": _pct([x["elapsed_ns"] / 1e6 for x in ws if x["elapsed_ns"] is not None], 1.0),
            "deadline_ns": w.get("deadline_ns"), "rtt_allowance_ns": w.get("rtt_allowance_ns"), "limit_ns": w.get("limit_ns"),
            "gap_s": w.get("gap_s"), "out_of_slots": w.get("out_of_slots"), "error": w.get("error"),
            "overlap": [{"key": key, "arm": arm, **{k: v for k, v in o.items() if not k.endswith("_s")},
                         "median_with_ms": None if o["median_with_s"] is None else o["median_with_s"] * 1e3,
                         "median_without_ms": None if o["median_without_s"] is None else o["median_without_s"] * 1e3}
                        for key, by_arm in (overlap or {}).items() for arm, o in by_arm.items()]}


def hosted(doc: dict, results: dict[str, dict]) -> list[dict] | None:
    """The hosted rounds (`chain.hosted`), in their order: where the responder ran, the honest audit's answers with the
    chain as its load, the chain's slowdown, and the joint round's (d) and (e)."""
    rs = doc.get("rounds")
    if not rs:
        return None
    out = []
    for name in rs.get("order") or []:
        r = rs.get(name) or {}
        row = {"round": name, "responder": r.get("responder"), "skipped": r.get("skipped"), "error": r.get("error"),
               "harness_rc": r.get("harness_rc"), "window_s": window_s(r), **harness(results.get(name)),
               "storage": C.storage(results.get(name)), "beside": beside_rows(r.get("points") or {})}
        if "alone" in r or "alone_error" in r:
            row["proof_reads"] = proof_reads(r.get("alone"), r.get("alone_error"))
        if "windows" in r:
            row["joint"] = joint_windows(r["windows"], r.get("alone"), r.get("overlap"))
        out.append(row)
    return out


def _f(v, fmt: str) -> str:
    return "–" if v is None else format(v, fmt)


def _us(q: dict, *keys: str) -> str:
    return " / ".join(_f(None if q.get(x) is None else q[x] * 1e6, ".0f") for x in keys)


def _gb(v) -> str:
    return _f(None if v is None else v / 1e9, ".3f")


def _blocks(n: int | None) -> str:
    return "–" if n is None else f"2^{n.bit_length() - 1}" if n > 0 and n & (n - 1) == 0 else f"{n:,}"


def _point(key: str) -> str:
    m = re.search(r"b(\d+)-L(\d+)$", key)
    return f"B={m[1]} L={m[2]}" if m else key


def _audit(p: dict, under: str) -> str:
    return (f"{p['accepted']} of {p['audits']} audits accepted, {p['answers']} answers, {p['late']} late, {p['wrong']} wrong; "
            f"limit {_f(p['limit_ns'] and p['limit_ns'] / 1e3, '.0f')} µs, fixed before load {p['limit_fixed_before_load']} "
            f"(calibrated while the chain waited); answer time p50 / p99.9 / max {_us(p['elapsed_s'], 'p50', 'p99.9', 'max')} µs, "
            f"least margin {_us(p['margin_s'], 'min')} µs; RTT beside {under} p50 / p99.9 {_us(p['rtt_s'], 'p50', 'p99.9')} µs")


def _gates(p: dict) -> str:
    return ("passed" if p["passed"] else "failed: " + ", ".join(p["failed_gates"])) if p["passed"] is not None \
        else "– (no bench-result.json)"


def storage_lines(st: dict, memory: float | None, label: str = "") -> list[str]:
    """S_cert with its basis, the deployed store against the card, and the capacity residual (the weak form)."""
    cap = C.capacity(memory, st["s_cert"], st["unproved"] or C.NO_POUS, st["stored"])
    if st["s_cert"] is not None:
        lines = [(f"S_cert{label} {_gb(st['s_cert'])} GB: a server holding fewer than ⌊ρ·|C|⌋ bits passes the audit of "
                  f"record's {st['k']} challenges within {st['limit_ns']:,} ns with probability at most {st['delta']} + "
                  f"2^{st['eps_log2']} ({st['theorem']}); EXPERIMENTAL: {st['experimental']}.")]
    else:
        lines = [f"S_cert{label} unproved: {st['unproved']}."]
    lines.append(f"S_cert's basis: {C.S_BASIS}.")
    if st["stored"] is not None:
        lines.append(f"The deployed store is {_blocks(st['blocks'])} blocks, |C| = {_gb(st['stored'])} GB of the card's "
                     f"{_f(memory and memory / 1e9, '.0f')} GB (M); lane ra-pous-scale is sizing the store to the device.")
    lines.append(f"Capacity {C.RESIDUAL}: M − S_cert = {_gb(cap['residual'])} of {_gb(cap['capacity'])} GB = honest overhead "
                 f"M − |C| {_gb(cap['honest_overhead'])} GB + soundness slack |C| − S_cert {_gb(cap['soundness_slack'])} GB.")
    return lines


def beside_table(beside: list[dict]) -> list[str]:
    lines = ["", "| point | arm | alone before ms | beside the audit ms | alone after ms | slowdown vs before | vs after | items |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['key']} | {r['arm']} | {r['alone_ms']:.3f} | {r['beside_ms']:.3f} | {_f(r['after_ms'], '.3f')} | "
              f"{r['slowdown']:+.1%} | {_f(r['slowdown_after'], '+.1%')} | {r['items']} |" for r in beside]
    return lines


def pous_lines(p: dict) -> list[str]:
    under = "the idle chain (the control)" if p["idle"] else "the chain"
    lines = ["", ("PoUS beside the chain: P2's timed audit (benchmarks/pous/pod.sh IMPL=p2-v3-mc-gpu: the GPU responder and the "
                  "memory challenger, signed with S1's LMS key), with no load of its own, on the same GPU in its own process; "
                  + ("the chain, resident, ran nothing (the control) " if p["idle"] else "the chain's items ran ")
                  + f"from its honest audit's start to its end ({_f(p['window_s'], '.1f')} s, closed by "
                  f"{p['window_closed_by'] or '–'}), off CPUs {p['left_cpus'] or '–'}; harness exit {p['harness_rc']}"
                  + (f" ({p['harness_stopped']})" if p["harness_stopped"] else "") + f", gates {_gates(p)}.")]
    if p["error"]:
        lines.append("Beside failed: " + p["error"].strip().splitlines()[-1])
    if p["audits"] is not None:
        lines.append(f"Honest audit beside {under}: {_audit(p, under)}.")
    lines += storage_lines(p["storage"], p["memory_bytes"])
    if p["beside"]:
        lines += beside_table(p["beside"])
        lines += ["", ("Alone before is the sweep's own median (its arms in turn with the GEMM-only graphs); alone after runs the "
                       "beside items' own turns once the harness has exited, so the audit's effect is the slowdown vs after.")]
    return lines


def _last(error: str | None) -> str:
    return (error or "").strip().splitlines()[-1] if error else ""


def hosted_lines(rounds: list[dict], memory: float | None) -> list[str]:
    lines = ["", ("Responder isolation without MPS: P2's timed audit (pod.sh IMPL=p2-v3-mc-gpu, the GPU responder and the memory "
                  "challenger, S1's LMS key) with the shape's chains as its honest audit's load, its limit calibrated before the "
                  "load; one harness run a round, in turn on one lease. separate: the harness in its own process and CUDA "
                  "context; inproc: in the chain's process, its responder a thread on its own stream in the chain's CUDA "
                  "context, the chain's items on a stream of their own; joint: inproc with the bandwidth windows (e) too."),
             "", ("| round | responder | audits accepted | answers | late | wrong | answer p50 / p99.9 / max µs | least margin µs "
                  "| limit µs | chain slowdown vs after | S_cert GB | harness |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rounds:
        if r["skipped"] or (r["error"] and r["audits"] is None):
            why = "skipped: " + r["skipped"] if r["skipped"] else "failed: " + _last(r["error"])
            lines.append(f"| {r['round']} | {r['responder'] or '–'} | {why} | | | | | | | | | |")
            continue
        slow = "; ".join(f"{_point(b['key'])} {b['arm']} {_f(b['slowdown_after'], '+.1%')}" for b in r["beside"]) or "–"
        lines.append(f"| {r['round']} | {r['responder']} | {r['accepted']} of {r['audits']} | {r['answers']} | {r['late']} | "
                     f"{r['wrong']} | {_us(r['elapsed_s'], 'p50', 'p99.9', 'max')} | {_us(r['margin_s'], 'min')} | "
                     f"{_f(r['limit_ns'] and r['limit_ns'] / 1e3, '.0f')} | {slow} | "
                     f"{_gb(r['storage']['s_cert']) if r['storage']['s_cert'] is not None else 'unproved'} | "
                     f"exit {r['harness_rc']}, gates {_gates(r)} |")
    st = next((r for r in rounds if r["storage"]["s_cert"] is not None), None)
    if st is not None or any(r["storage"]["stored"] is not None for r in rounds):
        st = st or next(r for r in rounds if r["storage"]["stored"] is not None)
        lines += ["", *storage_lines(st["storage"], memory, f" (round {st['round']})")]
    for r in rounds:
        if r["error"] and r["audits"] is not None:
            lines.append(f"Round {r['round']}: {_last(r['error'])}")
        d, e = r.get("proof_reads"), r.get("joint")
        if d is not None:
            lines += proof_read_lines(r["round"], d)
        if e is not None:
            lines += joint_lines(r["round"], e)
    return lines


def proof_read_lines(name: str, d: dict) -> list[str]:
    if d.get("store_bytes") is None:
        return ["", f"(d) proof-only reads, round {name}: failed: {_last(d['error'])}"]
    lines = ["", (f"(d) Proof-only reads, round {name}, at the harness's hook (the PoUS store E on the device, before the limit "
                  f"is calibrated, nothing else running): the bandwidth audit's tag pass over E ({_blocks(d['blocks'])} blocks, "
                  f"{_gb(d['store_bytes'])} GB), each pass from a clean L2 (a 512 MiB read of another buffer first) on a stream "
                  f"of the highest priority; E's raw read (torch.sum as int64) {d['raw_ms']:.3f} ms = {d['raw_gb_per_s']:.0f} GB/s. "
                  "Times and raw rates only (Q_cert pending)."),
             "", ("| tag | certified | device ms p50 / p99 | tags ms p50 | host ms p50 / p99 | GB/s over E | of the raw read | "
                  "windows on time at T / accepted / run | window max ms | skip control |"), "|---|---|---|---|---|---|---|---|---|---|"]
    for t in d["tags"]:
        lines.append(f"| {t['tag']} | {'yes' if t['certified'] else 'no'} | {_f(t['device_ms_p50'], '.3f')} / "
                     f"{_f(t['device_ms_p99'], '.3f')} | {_f(t['tags_ms_p50'], '.3f')} | {_f(t['host_ms_p50'], '.3f')} / "
                     f"{_f(t['host_ms_p99'], '.3f')} | {_f(t['gb_per_s'], '.0f')} | {_f(t['of_raw'], '.0%')} | "
                     f"{t['on_time_at_T']} / {t['accepted']} / {t['windows']} | {_f(t['window_ms_max'], '.3f')} | "
                     f"{'rejected (' + str(t['skip_failure']) + ')' if t['skip_rejected'] else 'ACCEPTED'} |")
    if d["tags"]:
        t = d["tags"][0]
        lines.append(f"A window is on time at T when it committed within T = {t['deadline_ns']:,} ns; its verifier is in this "
                     f"process (no network), and accepts to T + RTT = {t['limit_ns']:,} ns. The skip control leaves "
                     f"{t['skip_fraction']:.0%} of the blocks untagged and must be rejected.")
    return lines


def joint_lines(name: str, e: dict) -> list[str]:
    fails = ", ".join(f"{k} {v}" for k, v in sorted(e["failures"].items())) or "none"
    lines = ["", (f"(e) Joint v0, round {name}: the chain on plain W, P2's honest audit and {e['tag']}'s windows at once, one "
                  f"window at a time at random gaps (mean {_f(e['gap_s'], '.1f')} s) on a stream of the highest priority, "
                  "committed at once and judged after the harness returned: "
                  f"{e['on_time_at_T']} of {e['windows']} on time at T ({_f(e['deadline_ns'] and e['deadline_ns'] / 1e3, ',.0f')} µs, "
                  f"the RTT at P2's cap {_f(e['rtt_allowance_ns'] and e['rtt_allowance_ns'] / 1e3, '.0f')} µs inside the "
                  f"rebuild floor), {e['accepted']} accepted, failures {fails}; window p50 / max {_f(e['window_ms_p50'], '.3f')} / "
                  f"{_f(e['window_ms_max'], '.3f')} ms; device pass p50 / max {_f(e['device_ms_p50'], '.3f')} / "
                  f"{_f(e['device_ms_max'], '.3f')} ms under load against {_f(e['alone_device_ms_p50'], '.3f')} ms alone (d)"
                  + ("; it ran out of workspaces before the audit ended" if e["out_of_slots"] else "") + ".")]
    if e["error"]:
        lines.append(f"(e) stopped: {_last(e['error'])}")
    if e["overlap"]:
        lines += ["", "| point | arm | items overlapping a window | median ms | items not | median ms |", "|---|---|---|---|---|---|"]
        lines += [f"| {o['key']} | {o['arm']} | {o['with']} | {_f(o['median_with_ms'], '.3f')} | {o['without']} | "
                  f"{_f(o['median_without_ms'], '.3f')} |" for o in e["overlap"]]
    return lines


def q_cert_lines(ident: dict) -> list[str]:
    """Q_cert pending, and the device inputs it will take with the red team's conditions applied."""
    p = C.window_params(JOINT_TAG)
    chip = C.chip(ident["l2_bytes"], ident["sms"]) if ident.get("l2_bytes") and ident.get("sms") else None
    return [(f"Q_cert: {C.PENDING_Q} ({C.Q_THEOREM} is proved; no certified bandwidth rate is cited until the red team has "
             "re-read the bandwidth protocol's parameters at the BLAKE3 head): times and raw rates only. Its device inputs "
             "hold the red team's conditions: S_chip "
             + ("–" if chip is None else
                f"{chip['s_chip_bytes'] / 1e6:.1f} MB = the protocol's L2 + SMs × 384 KiB {chip['protocol_bytes'] / 1e6:.1f} MB "
                f"+ {chip['allowance_bytes'] / 1e6:.1f} MB for the constant and instruction caches it leaves out (512 KiB a "
                "SM, an allowance, not measured)")
             + f"; a window's T = {p.deadline_ns:,} ns with the RTT at P2's cap, {p.rtt_allowance_ns:,} ns, so T + RTT = "
             f"{p.limit_ns:,} ns stays under the rebuild floor of {p.regen_floor_ns:,} ns.")]


def _verdict(line: str | None) -> str:
    """verify_run.py's line as its verdict and the tiles it drew."""
    if not line:
        return "–"
    drawn = re.search(r"\((\d+ drawn[^)]*)\) of (\d+)", line)
    return line.split()[0] + (f" ({drawn[1]}, of {drawn[2]} tiles)" if drawn else "")


def markdown(doc: dict, verdicts: dict, source: str, result: dict | None = None, rounds: dict[str, dict] | None = None) -> str:
    ident, rr, p = doc.get("identity") or {}, doc.get("read_rate") or {}, pous(doc, result)
    hr = hosted(doc, rounds or {})
    clocks = [c.get("sm_mhz") for p in doc["points"] for c in p.get("clocks", []) if "sm_mhz" in c]
    lines = [f"Matmul chain y_(i+1) = act(W_i · y_i), {source}, run {doc.get('run')}, {doc.get('status')}",
             (f"{ident.get('name')} {ident.get('uuid')}, driver {ident.get('driver')}, torch {ident.get('torch')} (CUDA {ident.get('cuda')}); "
              f"SM clock median {statistics.median(clocks) if clocks else '–'} MHz over {len(clocks)} reps; "
              f"read rate {_f(rr.get('bytes_per_s') and rr['bytes_per_s'] / 1e9, '.0f')} GB/s ({rr.get('kernel')})")]
    v = doc.get("verify")
    if v:
        lines.append(f"Retained pass at {v.get('point')}: equals the eager pass {v.get('equals_reference')}, graphed pass gated "
                     f"{v.get('graphed_pass_gated')}; verify_run.py: pass {_verdict(verdicts.get('pass'))}, control "
                     f"{_verdict(verdicts.get('control'))} (replay-verified, a development shortcut; the protocol's check is a ZK proof)")
    lines += ["", ("| point | arm | mode | chain ms | overhead | useful TFLOP/s | native GEMM TFLOP/s | GB/s (share of read) | "
                   f"W_cert TFLOP/s-eq | compute {C.RESIDUAL} (honest + slack) | Q_cert | S_cert |"),
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows(doc):
        if "arm" not in r:
            lines.append(f"| {r['key']} | – | failed or skipped: {' '.join(r['error'])} | | | | | | | | | |")
            continue
        wc = "–"
        if r["arm"] in ("pearl-c", "pearl-c-deferred"):
            wc = _f(r.get("w_cert_tflops"), ".1f") if r.get("w_cert") is not None else "unproved"
        res = "–"
        if r.get("compute_residual") is not None:
            res = (f"{r['compute_residual']:.1%} ({r['compute_honest_overhead']:.1%} + {r['compute_soundness_slack']:.2%})")
        elif r.get("compute_honest_overhead") is not None:
            res = f"unproved ({r['compute_honest_overhead']:.1%} + ?)"
        q = "pending" if r["arm"] in S.POUW else "–"
        s = "none" if r["arm"] in S.POUW else "–"
        lines.append(f"| {r['shape']} B={r['batch']} L={r['length']} | {r['arm']} | {r['mode']} | {r['chain_ms']:.3f} | "
                     f"{_f(r['overhead'], '+.1%')} | {r['useful_tflops']:.1f} | {_f(r['gemm_tflops'], '.1f')} | "
                     f"{r['bytes_per_s'] / 1e9:.0f} ({_f(r['read_share'], '.0%')}) | {wc} | {res} | {q} | {s} |")
    notes = sorted({r.get("w_unproved") for r in rows(doc) if r.get("w_unproved")})
    if notes:
        lines += ["", "W_cert unproved: " + "; ".join(notes)]
    lines += [("W_cert is in W1 (one dense FP8 MAC on the device), as 2·W_cert over the chain's time; ΣW_ref prices a credited "
               "cell at about 1.1 W1 per MAC at these shapes, so W_cert can exceed the useful rate."),
              *q_cert_lines(ident),
              "S_cert: " + (C.NO_POUS if p is None and hr is None else
                            "the table's chains ran alone; the PoUS sections' ran beside or inside its audit") + ".",
              (f"Compute {C.RESIDUAL}: F·T − W_cert as a share of F·T (F the native FP8 chain's rate at the shape, T the arm's "
               "chain time), split into honest overhead (F·T − ΣW_ref) and soundness slack (ΣW_ref − W_cert). Every residual "
               "here is CovertResidual's weak form, worst case over each audit's own coins, not a per-run covert bound; lane "
               "ra-per-run is proving the per-run form.")]
    if p is not None:
        lines += pous_lines(p)
    if hr is not None:
        lines += hosted_lines(hr, ident.get("memory_bytes"))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="art:<id> of a run's tree, or a chain.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    doc, verdicts, result, rounds = load(a.source)
    if a.json:
        print(json.dumps({"source": a.source, "run": doc.get("run"), "verify": doc.get("verify"), "verdicts": verdicts,
                          "rows": rows(doc), "pous": pous(doc, result), "rounds": hosted(doc, rounds)}, indent=1, default=str))
    else:
        print(markdown(doc, verdicts, a.source, result, rounds), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
