"""The matmul chain's table, rendered on demand from a recorded run's ``chain.json`` (`chain.py`) and the code: nothing it
prints is stored, in Git or the evidence store.

  python render.py art:XXXXXXXX [--json]      the run's tree from the evidence store (``research data fetch``)
  python render.py OUT/chain.json [--json]

Per point and arm: the chain's time (the median over reps), its overhead over its divisor (`sweep.divisor`: the fastest native
arm of its precision the point measured, a tuned one among them), useful TFLOP/s (2·B·n·k·L over the time, the batch's real
rows only), the native GEMM rate at the shape (the native arms' GEMMs alone), and the device-memory rate the chain achieved,
estimated as its weights' and activations' bytes read once over its time (a lower bound: the weights are at least twice the
L2).  The tuned arms' picks follow the table.  For the Pearl-C arms, `certified`'s columns: W_cert (as TFLOP/s-equivalent,
2·W_cert over the time) and the compute residual's split as shares of F·T, with F the fastest native FP8 chain's rate; Q_cert is
pending (times and raw rates only) and S_cert says why it is unproved.  Every residual is the weak form (`certified.RESIDUAL`).
The retained pass's verdicts, when the run kept one, head the table.  A run with a PoUS arm (`chain.beside`) adds its section:
the honest audit's answers against the limit while the chain ran, S_cert from the harness's storage profile with the
capacity residual (`certified.storage`), and each chain's time beside the audit.  A run that hosted the harness
(`chain.hosted`, ``run.sh JOINT=``) adds the responder's isolation, one row a round, the chain's time in each round (host,
device, launch), the joint round's arms (d), the tag pass over the PoUS store alone, and (e), its windows beside the chain
and the audit, and the joint-v1 round's (f), the bandwidth audit's honest overhead: the useful TFLOP/s each window cadence
costs against the bytes it certifies a wall second (raw).  A run with ``--probe`` adds its conditions' chain times.
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
import probe as P
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
    x = 1 if arm in S.NATIVE and S.PRECISION[arm] == "fp8" else 2
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
        fp8 = t.get(S.divisor("native-fp8", t))
        F = p["macs"] / fp8 if fp8 else None
        for arm in (a for a in S.ARMS if a in t):
            r, T = p["arms"][arm], t[arm]
            by = S.divisor(arm, t)
            div = t.get(by)
            row = {"key": p["key"], "shape": p["shape"], "n": p["n"], "k": p["k"], "batch": p["batch"], "length": p["length"],
                   "arm": arm, "mode": r.get("mode", "graph"), "chains": r["chains"], "chain_ms": T * 1e3,
                   "spread": (max(r["seconds"]) - min(r["seconds"])) / T if r["seconds"] else None, "divisor": by,
                   "overhead": None if div is None or arm == by else T / div - 1,
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
    ms = lambda v: None if v is None else v * 1e3
    return [{"key": key, "arm": arm, "items": len(r["items"]), "alone_ms": r["alone_median_s"] * 1e3,
             "beside_ms": r["median_s"] * 1e3, "slowdown": r["slowdown"], "after_ms": ms(r.get("after_median_s")),
             "slowdown_after": None if not r.get("after_median_s") else r["median_s"] / r["after_median_s"] - 1,
             **{k: ms(r[x]) for k, x in (("device_ms", "device_median_s"), ("alone_device_ms", "alone_device_median_s"),
                                         ("after_device_ms", "after_device_median_s"), ("launch_ms", "launch_median_s"),
                                         ("after_launch_ms", "after_launch_median_s")) if r.get(x) is not None}}
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


def pause_rows(p: dict, macs: dict[str, int], store_bytes: int | None) -> list[dict]:
    """(f) (`joint.Pause`), one row a point, arm and window cadence over its segments: chains a wall second and useful
    TFLOP/s, against the same point and arm's segments without windows (the throughput lost); windows a wall second, on
    time at T, their elapsed time, and the gap each left in the chain's stream (`bubble`), whose mean times the windows a
    second is the loss it predicts; the pause's parts on the host (median ns); and E's bytes a wall second in windows on time
    at T, raw (Q_cert pending)."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for s in p.get("segments") or []:
        groups.setdefault((s["key"], s["arm"], s["every"]), []).append(s)
    rate = {k: sum(s["chains"] for s in v) / (sum(s["wall_ns"] for s in v) / 1e9) for k, v in groups.items()}
    order = {label: i for i, label in enumerate(p.get("every") or [])}
    out = []
    for (key, arm, every), segs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 99))):
        wall = sum(s["wall_ns"] for s in segs) / 1e9
        ws = [w for s in segs for w in s["windows"] if w.get("window") is not None]
        none, r = rate.get((key, arm, "none")), rate[(key, arm, every)]
        flop = 2 * macs[key] if key in macs else None
        on_time = sum(bool(w.get("on_time_at_T")) for w in ws)
        bubbles = [w["bubble_ms"] for w in ws if w.get("bubble_ms") is not None]
        row = {"key": key, "arm": arm, "every": every, "segments": len(segs), "wall_s": wall, "chains_per_s": r,
               "useful_tflops": None if flop is None else flop * r / 1e12, "windows": len(ws), "windows_per_s": len(ws) / wall,
               "on_time_at_T": on_time, "within_limit": sum(bool(w.get("drawn")) for w in ws),
               "window_ms_p50": _pct([w["elapsed_ns"] / 1e6 for w in ws if w.get("elapsed_ns") is not None], 0.5),
               "window_ms_max": _pct([w["elapsed_ns"] / 1e6 for w in ws if w.get("elapsed_ns") is not None], 1.0),
               "bubble_ms_p50": _pct(bubbles, 0.5), "bubble_ms_mean": statistics.fmean(bubbles) if bubbles else None,
               "loss": None if not none or every == "none" else 1 - r / none,
               "tflops_lost": None if flop is None or not none or every == "none" else flop * (none - r) / 1e12,
               "model_loss": None if not bubbles else len(ws) / wall * statistics.fmean(bubbles) / 1e3,
               "certified_bytes_per_s": None if store_bytes is None or not ws else on_time * store_bytes / wall}
        for part in ("drain_ns", "prep_ns", "seed_ns", "pass_ns", "commit_ns", "pause_ns"):
            row[part] = _pct([w[part] for w in ws if w.get(part) is not None], 0.5)
        for part in ("device_tags_ms", "device_ms"):
            row[part] = _pct([w[part] for w in ws if w.get(part) is not None], 0.5)
        out.append(row)
    return out


def alone_passes(passes: dict) -> dict:
    """(f)'s round's own (d) passes (`joint.Pause.alone`): their count and median tag kernel and pass ms on the device, at
    the round's start, after its segments without windows, and at its end."""
    groups = {"start": passes.get("start") or [], "segments": [x for s in passes.get("segments") or [] for x in s["passes"]],
              "end": passes.get("end") or []}
    return {k: {"n": len(v), "device_tags_ms": _pct([x["device_tags_ms"] for x in v], 0.5),
                "device_ms": _pct([x["device_ms"] for x in v], 0.5)} for k, v in groups.items()}


def pause(r: dict) -> dict:
    """(f)'s round: its rows (`pause_rows`), the verifier's pipe, the judged windows and the skip control."""
    p, setup = r["pause"], r.get("pause_setup") or {}
    store = (setup.get("store") or {}).get("bytes")
    judged = p.get("judged") or []
    return {"rows": pause_rows(p, r.get("macs") or {}, store), "store_bytes": store, "rtt": p.get("rtt") or {},
            "deadline_ns": p.get("deadline_ns"), "limit_ns": p.get("limit_ns"), "seconds": p.get("seconds"),
            "lookahead": p.get("lookahead"), "windows_started": p.get("windows_started"), "judged": len(judged),
            "judged_accepted": sum(bool(j.get("accepted")) for j in judged),
            "elapsed_agrees": all(j.get("elapsed_agrees") for j in judged), "skip": p.get("skip"), "error": p.get("error"),
            "alone": alone_passes(p["passes"]) if p.get("passes") else None}


def probe(doc: dict) -> dict | None:
    """`chain.run_probe`'s phases: each condition's median chain time on the host and the device, and its thread's nodes."""
    pr = doc.get("probe")
    if pr is None:
        return None
    rows = [{"phase": name, "label": label, **s} for name, ph in (pr.get("phases") or {}).items()
            for label, s in (ph.get("summary") or {}).items()]
    skipped = {name: ph.get("skipped") or ph.get("error") for name, ph in (pr.get("phases") or {}).items() if "summary" not in ph}
    clocks = {name: (ph.get("clocks_before") or {}).get("clocks.sm") for name, ph in (pr.get("phases") or {}).items()}
    course = {name: {"count": c.get("count"), "marks": c.get("marks") or {}, "error": c.get("error"),
                     "summary": c.get("summary") or {},
                     "clocks": [(c.get(k) or {}).get(f) for k in ("clocks_before", "clocks_after") for f in ("clocks.sm", "clocks.mem")]}
              for name, c in (pr.get("course") or {}).items()}
    return {"point": pr.get("point"), "error": pr.get("error"), "gpu_node": (pr.get("numa") or {}).get("gpu_node"),
            "rows": rows, "skipped": skipped, "sm_mhz": clocks, "course": course}


def hosted(doc: dict, results: dict[str, dict]) -> list[dict] | None:
    """The hosted rounds (`chain.hosted`), in their order: where the responder ran, the honest audit's answers with the
    chain as its load, the chain's slowdown, the joint round's (d) and (e), and the joint-v1 round's (f)."""
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
        if "pause" in r:
            row["pause"] = pause(r)
        elif "pause_error" in r:
            row["pause"] = {"error": r["pause_error"], "rows": []}
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
    timed = [(r["round"], b) for r in rounds for b in r["beside"] if b.get("device_ms") is not None]
    if timed:
        lines += ["", ("The chain's time in each round, ms a chain on the host / by events on the device / to launch on the "
                       "host: during the honest audit, and alone after the harness returned (or exited)."),
                  "", "| round | point | arm | during | after |", "|---|---|---|---|---|"]
        lines += [f"| {name} | {_point(b['key'])} | {b['arm']} | {b['beside_ms']:.4f} / {_f(b.get('device_ms'), '.4f')} / "
                  f"{_f(b.get('launch_ms'), '.4f')} | {_f(b['after_ms'], '.4f')} / {_f(b.get('after_device_ms'), '.4f')} / "
                  f"{_f(b.get('after_launch_ms'), '.4f')} |" for name, b in timed]
    st = next((r for r in rounds if r["storage"]["s_cert"] is not None), None)
    if st is not None or any(r["storage"]["stored"] is not None for r in rounds):
        st = st or next(r for r in rounds if r["storage"]["stored"] is not None)
        lines += ["", *storage_lines(st["storage"], memory, f" (round {st['round']})")]
    for r in rounds:
        if r["error"] and r["audits"] is not None:
            lines.append(f"Round {r['round']}: {_last(r['error'])}")
        d, e, f = r.get("proof_reads"), r.get("joint"), r.get("pause")
        if d is not None:
            lines += proof_read_lines(r["round"], d)
        if e is not None:
            lines += joint_lines(r["round"], e)
        if f is not None:
            lines += pause_lines(r["round"], f)
    return lines


def pause_lines(name: str, f: dict) -> list[str]:
    if not f["rows"] and f.get("error"):
        return ["", f"(f) Joint v1, round {name}: failed: {_last(f['error'])}"]
    skip = f.get("skip") or {}
    rtt = f.get("rtt") or {}
    lines = ["", (f"(f) Joint v1, round {name}: the bandwidth audit's honest overhead. The chain issued one graph a chain, at "
                  f"most {f['lookahead']} in flight, inside P2's honest audit (its responder a thread here); for each "
                  f"{JOINT_TAG} window the prover paused it at a call boundary (a chain's last call), drained its stream, "
                  "prepared as (d) does (an untimed pass, then the scrub; -noprep: neither), took σ, ran the tag pass, sent "
                  "its root and resumed. The verifier is a process of its own, so a window's elapsed time holds the pipe's "
                  f"round trip (p50 / p99 {_f(rtt.get('p50_ns') and rtt['p50_ns'] / 1e3, '.0f')} / "
                  f"{_f(rtt.get('p99_ns') and rtt['p99_ns'] / 1e3, '.0f')} µs). Segments of {_f(f['seconds'], '.1f')} s, one "
                  f"cadence each, in turn; {f['windows_started']} windows; {f['judged_accepted']} of {f['judged']} judged "
                  f"accepted (their elapsed time the timing record's: {f['elapsed_agrees']}); the skip control "
                  + ("–" if not skip else f"({skip.get('fraction', 0):.0%} untagged) "
                     + ("ACCEPTED" if skip.get("accepted") else f"rejected ({skip.get('failure')})")) + "."),
             "", ("| point | arm | a window each | chains/s | useful TFLOP/s | lost | TFLOP/s lost | windows/s | on time at T | "
                  "window ms p50 / max | gap ms p50 / mean | windows/s × gap | GB/s certified per wall s (raw) |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in f["rows"]:
        on = "–" if not r["windows"] else f"{r['on_time_at_T']} / {r['windows']}"
        lines.append(f"| {_point(r['key'])} | {r['arm']} | {r['every']} | {r['chains_per_s']:.1f} | "
                     f"{_f(r['useful_tflops'], '.2f')} | {_f(r['loss'], '+.2%')} | {_f(r['tflops_lost'], '.3f')} | "
                     f"{r['windows_per_s']:.2f} | {on} | {_f(r['window_ms_p50'], '.3f')} / {_f(r['window_ms_max'], '.3f')} | "
                     f"{_f(r['bubble_ms_p50'], '.3f')} / {_f(r['bubble_ms_mean'], '.3f')} | {_f(r['model_loss'], '.2%')} | "
                     f"{_f(r['certified_bytes_per_s'] and r['certified_bytes_per_s'] / 1e9, '.1f')} |")
    parts: dict[str, list[dict]] = {}
    for r in f["rows"]:
        if r["windows"]:
            parts.setdefault(r["every"], []).append(r)
    if parts:
        lines += ["", "The pause's parts on the host, median µs over the cadence's rows: " + "; ".join(
            f"{every}: drain {_f(_med(rs, 'drain_ns'), '.0f')}, prepare {_f(_med(rs, 'prep_ns'), '.0f')}, σ "
            f"{_f(_med(rs, 'seed_ns'), '.0f')}, pass {_f(_med(rs, 'pass_ns'), '.0f')}, root {_f(_med(rs, 'commit_ns'), '.0f')}, "
            f"drained to resumed {_f(_med(rs, 'pause_ns'), '.0f')}" for every, rs in parts.items()) + "."]
        lines.append("The window's pass on the device, median ms over the cadence's rows (tag kernel / whole pass): " + "; ".join(
            f"{every}: {_f(_pct([r['device_tags_ms'] for r in rs], 0.5), '.3f')} / "
            f"{_f(_pct([r['device_ms'] for r in rs], 0.5), '.3f')}" for every, rs in parts.items()) + ".")
    if f.get("alone"):
        lines.append("(d)'s passes in this round (each from a scrubbed L2, the chain idle), median ms on "
                     "the device (tag kernel / whole pass): " + "; ".join(
                         f"{where} ({a['n']}) {_f(a['device_tags_ms'], '.3f')} / {_f(a['device_ms'], '.3f')}"
                         for where, a in (("at its start", f["alone"]["start"]),
                                          ("after each segment without windows", f["alone"]["segments"]),
                                          ("at its end", f["alone"]["end"]))) + ".")
    lines.append(f"Lost is the throughput against the same point and arm's segments without windows; the gap is the chain "
                 f"stream's idle time a window, by its events; on time at T is within {_f(f['deadline_ns'], ',')} ns. "
                 f"Certified per wall second is E's {_gb(f['store_bytes'])} GB in each window on time at T, raw: Q_cert is "
                 f"pending.")
    if f.get("error"):
        lines.append(f"(f) stopped: {_last(f['error'])}")
    return lines


def _med(rows: list[dict], key: str) -> float | None:
    xs = [r[key] for r in rows if r.get(key) is not None]
    return None if not xs else statistics.median(xs) / 1e3


def probe_lines(p: dict) -> list[str]:
    if p.get("error"):
        return ["", f"Probe: failed: {_last(p['error'])}"]
    lines = ["", (f"Probe at {p['point']} (`probe.py`): the native chains under each condition in turn, rep by rep; a chain's "
                  "median time on the host, by events on the device, and to launch on the host; the CPU's frequency and NUMA "
                  f"node the thread ran on (the GPU's node is {_f(p['gpu_node'], 'd')}). +held: every replay queued before "
                  "the device reached it (queued: each hold outlasted its launches), so the device time is the device's alone. "
                  "SM MHz at each phase's start: " + ", ".join(f"{k} {v or '–'}" for k, v in p["sm_mhz"].items()) + "."),
             "", "| phase | arm / stream | host ms | device ms | launch ms | CPU MHz | nodes |", "|---|---|---|---|---|---|---|"]
    for r in p["rows"]:
        held = "" if "queued" not in r else " (queued)" if r["queued"] else " (not queued)"
        lines.append(f"| {r['phase']} | {r['label']}{held} | {r['host_ms']:.4f} | {r['device_ms']:.4f} | "
                     f"{_f(r.get('launch_ms'), '.4f')} | {_f(r.get('cpu_mhz'), '.0f')} | {', '.join(map(str, r['nodes']))} |")
    lines += [f"Probe {k}: {v}" for k, v in p["skipped"].items()]
    return lines + course_lines(p.get("course") or {})


def course_lines(course: dict) -> list[str]:
    if not course:
        return []
    ms = lambda v: _f(v, ".3f")
    lines = ["", ("The probe's course: the native chains back to back on the default stream around one co-tenant at a time, "
                  "a chain's median device ms before the co-tenant started, while it started, while it was ready, and in each "
                  "second after it exited; the effect is ready against before, the onset (s from its start) the first chain "
                  "nearer the ready time, the offset (s from its exit) the first chain from which all are nearer the before time "
                  f"(both only for an effect of at least {P.COURSE_EFFECT:.0%}). SM / memory MHz before and after each: "
                  + "; ".join(f"{k} {'/'.join(str(x or '–') for x in c['clocks'][:2])} to "
                              f"{'/'.join(str(x or '–') for x in c['clocks'][2:])}" for k, c in course.items()) + "."),
             "", ("| co-tenant | arm | ready at s | before | starting | ready | effect | onset s | after it exited, each second "
                  "| offset s |"), "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in course.items():
        who = name if (c["count"] or 1) == 1 else f"{name} ×{c['count']}"
        for arm, s in c["summary"].items():
            lines.append(f"| {who} | {arm} | {_f(c['marks'].get('ready'), '.2f')} | {ms(s['before_ms'])} | "
                         f"{ms(s['starting_ms'])} | {ms(s['ready_ms'])} | {_f(s['effect'], '+.1%')} | "
                         f"{_f(s['onset_s'], '.2f')} | {', '.join(ms(x) for x in s['after_ms']) or '–'} | "
                         f"{_f(s['offset_s'], '.2f')} |")
        if c["error"]:
            lines.append(f"Course {who}: {c['error']}")
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
    known = all(ident.get(k) for k in ("l2_bytes", "sms", "regs_per_sm"))
    chip = C.chip(ident) if known else None
    return [(f"Q_cert: {C.PENDING_Q} ({C.Q_THEOREM} is proved; no certified bandwidth rate is cited until the red team has "
             "re-read the bandwidth protocol's parameters at the BLAKE3 head): times and raw rates only. Its device inputs "
             "hold the red team's conditions: S_chip "
             + ("– (this run recorded no registers a SM)" if chip is None else
                f"{chip['s_chip_bytes']:,} bytes = {chip['formula']}, the bandwidth harness's count with the constant and "
                "instruction caches")
             + f"; a window's T = {p.deadline_ns:,} ns with the RTT at P2's cap, {p.rtt_allowance_ns:,} ns, so T + RTT = "
             f"{p.limit_ns:,} ns stays under the rebuild floor of {p.regen_floor_ns:,} ns.")]


def _verdict(line: str | None) -> str:
    """verify_run.py's line as its verdict and the tiles it drew."""
    if not line:
        return "–"
    drawn = re.search(r"\((\d+ drawn[^)]*)\) of (\d+)", line)
    return line.split()[0] + (f" ({drawn[1]}, of {drawn[2]} tiles)" if drawn else "")


def tuned_lines(doc: dict) -> list[str]:
    """The tuned arms' picks: per point, the GEMM each took, its time as the chain's L GEMMs alone, the candidates the gate
    passed, and the fastest one the arm can't take."""
    tuner = doc.get("tuner")
    if tuner is None:
        return []
    if "error" in tuner:
        return ["", f"Tuned arms: no tuner ({tuner['error'].strip().splitlines()[-1]})."]
    libs = ", ".join(f"{x['prefix']} {x['version']}" for x in (tuner.get("versions") or {}).get("cublaslt_libs") or [])
    lines = ["", (f"Tuned arms (tuned.py): every cuBLASLt heuristic candidate ({tuner['lt_candidates']} a library: {libs}) and "
                  f"every CUTLASS kernel of the family ({tuner['cutlass_kernels']} in the registry, "
                  f"{'in each tile order' if tuner['cutlass_sched'] else 'in its default tile order'}), gated bit-exactly by "
                  "the PoUW harness's gate with its known-bad variants rejected, then timed as the chain's L GEMMs alone; "
                  "the fastest eligible is the pick."),
             "", "| point | arm | pick | library | L GEMMs ms | gated / tried | fastest ruled out |", "|---|---|---|---|---|---|---|"]
    for p in doc["points"]:
        for arm, t in (p.get("tuned") or {}).items():
            where = f"{p['shape']} B={p['batch']} L={p['length']}"
            if "pick" not in t:
                why = (t.get("error") or "").strip().splitlines()[-1:]
                lines.append(f"| {where} | {arm} | refused: {' '.join(why)} | | | | |")
                continue
            out = t.get("fastest_ineligible")
            ruled = "–" if out is None else f"{out['name']} {out['ms']:.3f} ms ({out['library']}: no scale)"
            lines.append(f"| {where} | {arm} | {t['pick']['name']} | {t['pick']['library']} | {t['pick']['ms']:.3f} | "
                         f"{t['gated']} / {t['tried']} | {ruled} |")
    return lines


def markdown(doc: dict, verdicts: dict, source: str, result: dict | None = None, rounds: dict[str, dict] | None = None) -> str:
    ident, rr, p = doc.get("identity") or {}, doc.get("read_rate") or {}, pous(doc, result)
    hr = hosted(doc, rounds or {})
    clocks = [c.get("sm_mhz") for p in doc["points"] for c in p.get("clocks", []) if "sm_mhz" in c]
    lines = [f"Matmul chain y_(i+1) = act(W_i · y_i), {source}, run {doc.get('run')}, {doc.get('status')}",
             (f"{ident.get('name')} {ident.get('uuid')}, driver {ident.get('driver')}, torch {ident.get('torch')} (CUDA {ident.get('cuda')}); "
              f"SM clock median {statistics.median(clocks) if clocks else '–'} MHz over {len(clocks)} reps; "
              f"read rate {_f(rr.get('bytes_per_s') and rr['bytes_per_s'] / 1e9, '.0f')} GB/s ({rr.get('kernel')})")]
    if doc.get("sidecar"):
        lines.append(f"Every arm ran beside a sidecar, an idle CUDA context in a second process held through the run (ready in "
                     f"{_f(doc['sidecar'].get('ready_s'), '.1f')} s), which the probe's course found makes the native chains "
                     "faster.")
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
        over = ("the divisor" if r["arm"] == r["divisor"] else "–" if r["overhead"] is None
                else f"{r['overhead']:+.1%} ÷ {r['divisor']}")
        lines.append(f"| {r['shape']} B={r['batch']} L={r['length']} | {r['arm']} | {r['mode']} | {r['chain_ms']:.3f} | "
                     f"{over} | {r['useful_tflops']:.1f} | {_f(r['gemm_tflops'], '.1f')} | "
                     f"{r['bytes_per_s'] / 1e9:.0f} ({_f(r['read_share'], '.0%')}) | {wc} | {res} | {q} | {s} |")
    notes = sorted({r.get("w_unproved") for r in rows(doc) if r.get("w_unproved")})
    if notes:
        lines += ["", "W_cert unproved: " + "; ".join(notes)]
    lines += [("W_cert is in W1 (one dense FP8 MAC on the device), as 2·W_cert over the chain's time; ΣW_ref prices a credited "
               "cell at about 1.1 W1 per MAC at these shapes, so W_cert can exceed the useful rate."),
              *q_cert_lines(ident),
              "S_cert: " + (C.NO_POUS if p is None and hr is None else
                            "the table's chains ran alone; the PoUS sections' ran beside or inside its audit") + ".",
              (f"Compute {C.RESIDUAL}: F·T − W_cert as a share of F·T (F the fastest native FP8 chain's rate at the point, T the arm's "
               "chain time), split into honest overhead (F·T − ΣW_ref) and soundness slack (ΣW_ref − W_cert). Every residual "
               "here is CovertResidual's weak form, worst case over each audit's own coins, not a per-run covert bound; lane "
               "ra-per-run is proving the per-run form.")]
    lines += tuned_lines(doc)
    if p is not None:
        lines += pous_lines(p)
    if hr is not None:
        lines += hosted_lines(hr, ident.get("memory_bytes"))
    pr = probe(doc)
    if pr is not None:
        lines += probe_lines(pr)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="art:<id> of a run's tree, or a chain.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    doc, verdicts, result, rounds = load(a.source)
    if a.json:
        print(json.dumps({"source": a.source, "run": doc.get("run"), "verify": doc.get("verify"), "verdicts": verdicts,
                          "rows": rows(doc), "pous": pous(doc, result), "rounds": hosted(doc, rounds), "probe": probe(doc)},
                         indent=1, default=str))
    else:
        print(markdown(doc, verdicts, a.source, result, rounds), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
