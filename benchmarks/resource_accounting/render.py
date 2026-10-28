"""The matmul chain's table, rendered on demand from a recorded run's ``chain.json`` (`chain.py`) and the code: nothing it
prints is stored, in Git or the evidence store.

  python render.py art:XXXXXXXX [--json]      the run's tree from the evidence store
  python render.py OUT/chain.json [--json]

Per point and arm: the chain's time (the median over reps), its overhead over its divisor (`sweep.divisor`: the fastest native
arm of its precision the point measured, a tuned one among them), useful TFLOP/s (2·B·n·k·L over the time, the batch's real
rows only), the native GEMM rate at the shape (the native arms' GEMMs alone), and the device-memory rate the chain achieved,
estimated as its weights' and activations' bytes read once over its time (a lower bound: the weights are at least twice the
L2).  The tuned arms' picks follow the table.  For the Pearl-C arms, `certified`'s columns: W_cert (as TFLOP/s-equivalent,
2·W_cert over the time) and the compute residual's split as shares of F·T, with F the fastest native FP8 chain's rate; Q_cert,
none, since the chains run with no bandwidth arm (`certified.NO_BANDWIDTH`), and S_cert, or why it is unproved.  Every
residual is the weak form (`certified.RESIDUAL`).
The retained pass's verdicts, when the run kept one, head the table.  A run with a PoUS arm (`chain.beside`) adds its section:
the honest audit's answers against the limit while the chain ran, S_cert from the harness's storage profile with the
capacity residual (`certified.storage`), and each chain's time beside the audit.  A run that hosted the harness
(`chain.hosted`, ``run.sh JOINT=``) adds the responder's isolation, one row a round, and the chain's time in each round (host,
device, launch).  A run with ``--probe`` adds its conditions' chain times.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import certified as C
import probe as P
import sweep as S

SCHEMA = "ra-matmul-chain/v1"


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
                row["q_cert"] = bw["q_cert"]
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
    chain as its load, and the chain's slowdown."""
    rs = doc.get("rounds")
    if not rs:
        return None
    out = []
    for name in rs.get("order") or []:
        r = rs.get(name) or {}
        out.append({"round": name, "responder": r.get("responder"), "skipped": r.get("skipped"), "error": r.get("error"),
                    "harness_rc": r.get("harness_rc"), "window_s": window_s(r), **harness(results.get(name)),
                    "storage": C.storage(results.get(name)), "beside": beside_rows(r.get("points") or {})})
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
                     f"{_f(memory and memory / 1e9, '.0f')} GB (M); the store is not yet sized to the device.")
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
                  "context, the chain's items on a stream of their own."),
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
    return lines


def gamma_lines(rs: list[dict]) -> list[str]:
    """Which γ each W_cert used, and the served layout's, cited to the lookup's PR."""
    shapes = {(r["shape"], r["n"], r["k"]): r["gamma"] for r in rs if r.get("gamma") is not None}
    if not shapes:
        return []
    lay = C.gamma(None, None, C.LAYOUT)
    at = "; ".join(f"{s} ({n}, {k}) {float(Fraction(g)):.5%}" for (s, n, k), g in sorted(shapes.items()))
    layout = _f(lay["published"] and float(Fraction(lay["published"])), ".5%")
    return [(f"γ is `price_twins.lookup`'s, the Lean-pinned price twins' larger FP32 price at the point's (n, k) ({C.GAMMA_PIN}): "
             f"{at}; {C.LAYOUT}'s layout, the max over its linears, {layout}.")]


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
        q = "0 (no bandwidth arm)" if r["arm"] in S.POUW else "–"
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
              *gamma_lines(rows(doc)),
              f"Q_cert: none, {C.NO_BANDWIDTH}.",
              "S_cert: " + (C.NO_POUS if p is None and hr is None else
                            "the table's chains ran alone; the PoUS sections' ran beside or inside its audit") + ".",
              (f"Compute {C.RESIDUAL}: F·T − W_cert as a share of F·T (F the fastest native FP8 chain's rate at the point, T the arm's "
               "chain time), split into honest overhead (F·T − ΣW_ref) and soundness slack (ΣW_ref − W_cert). Every residual "
               f"here is CovertResidual's weak form, worst case over each audit's own coins, not a per-run covert bound: {C.PER_RUN}. "
               f"Note: {C.FIRST_FETCH}.")]
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
