"""The matmul chain's table, rendered on demand from a recorded run's ``chain.json`` (`chain.py`) and the code: nothing it
prints is stored, in Git or the evidence store.

  python render.py art:XXXXXXXX [--json]      the run's tree from the evidence store (``research data fetch``)
  python render.py OUT/chain.json [--json]

Per point and arm: the chain's time (the median over reps), its overhead over its divisor (`sweep.DIVISOR`), useful TFLOP/s
(2·B·n·k·L over the time, the batch's real rows only), the native GEMM rate at the shape (the native arms' GEMMs alone), and
the device-memory rate the chain achieved, estimated as its weights' and activations' bytes read once over its time (a lower
bound: the weights are at least twice the L2).  For the Pearl-C arms, `certified`'s columns: W_cert (as TFLOP/s-equivalent,
2·W_cert over the time) and the compute residual's split as shares of F·T, with F the native FP8 chain's rate; Q_cert and
S_cert say why they are unproved.  The retained pass's verdicts, when the run kept one, head the table.  A run with a PoUS
arm (`chain.beside`) adds its section: the honest audit's answers against the limit while the chain ran, S_cert from the
harness's storage profile with the capacity residual (`certified.storage`), and each chain's time beside the audit.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import certified as C
import sweep as S

SCHEMA = "ra-matmul-chain/v1"


def load(source: str) -> tuple[dict, dict, dict | None]:
    """The run's chain.json, its verify verdicts (``verify/*.txt`` beside it) and the PoUS harness's result
    (``pous/bench/bench-result.json`` beside it, when the run had a PoUS arm), from a path or an ``art:`` id."""
    if source.startswith("art:"):
        d = Path(tempfile.mkdtemp(prefix="ra-render-"))
        subprocess.run(["research", "data", "fetch", source, "--to", str(d), "--path", "*chain.json", "--path", "*verify/*.txt",
                        "--path", "*pous/bench/bench-result.json"], check=True, stdout=subprocess.DEVNULL)
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
    result = path.parent / "pous" / "bench" / "bench-result.json"
    return doc, verdicts, json.loads(result.read_text()) if result.exists() else None


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
                row["q_unproved"] = bw["unproved"]
                row["bandwidth_honest_overhead"] = (None if bw["honest_overhead"] is None or not bw["capacity"]
                                                    else bw["honest_overhead"] / bw["capacity"])
                cap = C.capacity(memory)
                row["s_unproved"] = cap["unproved"]
            out.append(row)
    return out


def pous(doc: dict, result: dict | None) -> dict | None:
    """The PoUS arm (`chain.beside`): the harness's honest audit while the chain ran (its answers against the limit fixed
    before), its gates, S_cert with the capacity residual, and each chain's time beside the audit against its time alone."""
    b = doc.get("beside")
    if b is None and result is None:
        return None
    b, result = b or {}, result or {}
    m = result.get("metrics") or {}
    audit, deadline = m.get("audit") or {}, m.get("deadline") or {}
    st = C.storage(result or None)
    memory = (doc.get("identity") or {}).get("memory_bytes")
    beside = [{"key": key, "arm": arm, "items": len(r["items"]), "alone_ms": r["alone_median_s"] * 1e3,
               "beside_ms": r["median_s"] * 1e3, "slowdown": r["slowdown"],
               "after_ms": None if r.get("after_median_s") is None else r["after_median_s"] * 1e3,
               "slowdown_after": None if not r.get("after_median_s") else r["median_s"] / r["after_median_s"] - 1}
              for key, by_arm in (b.get("points") or {}).items() for arm, r in by_arm.items()]
    return {"error": b.get("error"), "idle": b.get("idle"), "window_closed_by": b.get("window_closed_by"), "harness_rc": b.get("harness_rc"),
            "harness_stopped": b.get("harness_stopped"), "left_cpus": b.get("left_cpus"),
            "window_s": (b["window_end_ns"] - b["window_start_ns"]) / 1e9 if b.get("window_end_ns") else None,
            "passed": result.get("passed"), "failed_gates": sorted(g for g, v in (result.get("gates") or {}).items()
                                                                   if not (v or {}).get("passed")),
            "audits": audit.get("audits"), "accepted": audit.get("audits_accepted"), "answers": audit.get("answers"),
            "late": audit.get("late"), "wrong": audit.get("wrong"), "limit_ns": deadline.get("limit_ns"),
            "limit_fixed_before_load": deadline.get("limit_fixed_before_load"), "elapsed_s": audit.get("elapsed_s") or {},
            "margin_s": audit.get("margin_s") or {}, "rtt_s": (audit.get("calibration_here") or {}).get("rtt_s") or {},
            "storage": st, "capacity": C.capacity(memory, st["s_cert"], st["unproved"] or C.NO_POUS, st["stored"]),
            "beside": beside}


def _f(v, fmt: str) -> str:
    return "–" if v is None else format(v, fmt)


def _us(q: dict, *keys: str) -> str:
    return " / ".join(_f(None if q.get(x) is None else q[x] * 1e6, ".0f") for x in keys)


def _gb(v) -> str:
    return _f(None if v is None else v / 1e9, ".3f")


def pous_lines(p: dict) -> list[str]:
    under = "the idle chain (the control)" if p["idle"] else "the chain"
    lines = ["", ("PoUS beside the chain: P2's timed audit (benchmarks/pous/pod.sh IMPL=p2-v3-mc-gpu: the GPU responder and the "
                  "memory challenger, signed with S1's LMS key), with no load of its own, on the same GPU in its own process; "
                  + ("the chain, resident, ran nothing (the control) " if p["idle"] else "the chain's items ran ")
                  + f"from its honest audit's start to its end ({_f(p['window_s'], '.1f')} s, closed by "
                  f"{p['window_closed_by'] or '–'}), off CPUs {p['left_cpus'] or '–'}; harness exit {p['harness_rc']}"
                  + (f" ({p['harness_stopped']})" if p["harness_stopped"] else "") + ", gates "
                  + ("passed" if p["passed"] else "failed: " + ", ".join(p["failed_gates"]) if p["passed"] is not None
                     else "– (no bench-result.json)") + ".")]
    if p["error"]:
        lines.append("Beside failed: " + p["error"].strip().splitlines()[-1])
    if p["audits"] is not None:
        lines.append(f"Honest audit beside {under}: {p['accepted']} of {p['audits']} audits accepted, {p['answers']} answers, "
                     f"{p['late']} late, {p['wrong']} wrong; limit {_f(p['limit_ns'] and p['limit_ns'] / 1e3, '.0f')} µs, "
                     f"fixed before load {p['limit_fixed_before_load']} (calibrated while the chain waited); answer time "
                     f"p50 / p99.9 / max {_us(p['elapsed_s'], 'p50', 'p99.9', 'max')} µs, least margin "
                     f"{_us(p['margin_s'], 'min')} µs; RTT beside {under} p50 / p99.9 {_us(p['rtt_s'], 'p50', 'p99.9')} µs.")
    st, cap = p["storage"], p["capacity"]
    if st["s_cert"] is not None:
        lines.append(f"S_cert {_gb(st['s_cert'])} GB: a server holding fewer than ⌊ρ·|C|⌋ bits passes the audit of record's "
                     f"{st['k']} challenges within {st['limit_ns']:,} ns with probability at most {st['delta']} + "
                     f"2^{st['eps_log2']} ({st['theorem']}); EXPERIMENTAL: {st['experimental']}.")
    else:
        lines.append(f"S_cert unproved: {st['unproved']}.")
    lines.append(f"Capacity residual M − S_cert: {_gb(cap['residual'])} of {_gb(cap['capacity'])} GB = honest overhead M − |C| "
                 f"{_gb(cap['honest_overhead'])} GB + soundness slack |C| − S_cert {_gb(cap['soundness_slack'])} GB "
                 f"(|C| = {_gb(st['stored'])} GB).")
    if p["beside"]:
        lines += ["", ("| point | arm | alone before ms | beside the audit ms | alone after ms | slowdown vs before | "
                       "vs after | items |"), "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['key']} | {r['arm']} | {r['alone_ms']:.3f} | {r['beside_ms']:.3f} | {_f(r['after_ms'], '.3f')} | "
                  f"{r['slowdown']:+.1%} | {_f(r['slowdown_after'], '+.1%')} | {r['items']} |" for r in p["beside"]]
        lines += ["", ("Alone before is the sweep's own median (its arms in turn with the GEMM-only graphs); alone after runs the "
                       "beside items' own turns once the harness has exited, so the audit's effect is the slowdown vs after.")]
    return lines


def _verdict(line: str | None) -> str:
    """verify_run.py's line as its verdict and the tiles it drew."""
    if not line:
        return "–"
    drawn = re.search(r"\((\d+ drawn[^)]*)\) of (\d+)", line)
    return line.split()[0] + (f" ({drawn[1]}, of {drawn[2]} tiles)" if drawn else "")


def markdown(doc: dict, verdicts: dict, source: str, result: dict | None = None) -> str:
    ident, rr, p = doc.get("identity") or {}, doc.get("read_rate") or {}, pous(doc, result)
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
                   "W_cert TFLOP/s-eq | compute residual (honest + slack) | Q_cert | S_cert |"),
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
        q = "unproved" if r["arm"] in S.POUW else "–"
        s = "none" if r["arm"] in S.POUW else "–"
        lines.append(f"| {r['shape']} B={r['batch']} L={r['length']} | {r['arm']} | {r['mode']} | {r['chain_ms']:.3f} | "
                     f"{_f(r['overhead'], '+.1%')} | {r['useful_tflops']:.1f} | {_f(r['gemm_tflops'], '.1f')} | "
                     f"{r['bytes_per_s'] / 1e9:.0f} ({_f(r['read_share'], '.0%')}) | {wc} | {res} | {q} | {s} |")
    notes = sorted({r.get("w_unproved") for r in rows(doc) if r.get("w_unproved")})
    if notes:
        lines += ["", "W_cert unproved: " + "; ".join(notes)]
    lines += [("W_cert is in W1 (one dense FP8 MAC on the device), as 2·W_cert over the chain's time; ΣW_ref prices a credited "
               "cell at about 1.1 W1 per MAC at these shapes, so W_cert can exceed the useful rate."),
              "Q_cert: " + C.UNPROVED_Q + ". S_cert: " + (C.NO_POUS if p is None else
                                                          "the table's chains ran alone; the PoUS section's ran beside its audit") + ".",
              ("Compute residual: F·T − W_cert as a share of F·T (F the native FP8 chain's rate at the shape, T the arm's chain "
               "time), split into honest overhead (F·T − ΣW_ref) and soundness slack (ΣW_ref − W_cert).")]
    if p is not None:
        lines += pous_lines(p)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="art:<id> of a run's tree, or a chain.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    doc, verdicts, result = load(a.source)
    if a.json:
        print(json.dumps({"source": a.source, "run": doc.get("run"), "verify": doc.get("verify"), "verdicts": verdicts,
                          "rows": rows(doc), "pous": pous(doc, result)}, indent=1, default=str))
    else:
        print(markdown(doc, verdicts, a.source, result), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
