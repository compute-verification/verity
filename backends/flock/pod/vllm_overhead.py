"""A served row's proof overhead weighted over every proof unit of its q-word partition (vllm-overhead/v0), stdlib only.

    vllm_overhead.py census --counts counts.json [--counts-all counts-all.json] [--out census.json]
        the gate census: every shape's total gates (word gates x row units), each Call Definition's and Definition family's
        share, and with --counts-all (the same cut without --select) the units --select leaves out (the token-select Calls)
    vllm_overhead.py timing OUT_K [--warm 1] [--runs 3]
        after 73-sweep-shape.sh proved one shape into OUT_K: OUT_K/timing.json, its steady-state seconds per statement and per
        unit, from the prove process's stamps (session: the timed sessions' wall from the warm one's verdict to the last's, the
        verifier's round trips and verdicts included, as gemm_hill.py's t_session) and from its median timed LIVE record
        (prove-only: gemm_slowdown.statement_seconds)
    vllm_overhead.py rollup --counts counts.json [--counts-all ...] --timings PATH... --native native.json --row ROW --out vllm-overhead.json
        the weighted number for each prover mode (zk, m0): measured shapes at their seconds per unit times their row units, the
        rest priced at their Definition family's measured seconds per word gate (else the row's median), over the run's native
        seconds; on --basis (default prove, Panel 1's prover seconds) with the other basis beside it (other_bases)
    vllm_overhead.py point --rollup vllm-overhead.json --config vllm/<model>/<all|prefill|decode>/<zk|m0> --result result.json
        one config of the rollup as an attempt's result (measurement `overhead`, x-native)

PATH is a timing.json, a 73-sweep-shape.sh SUMMARY (its `timings`), or a directory searched for either. A shape measured more than
once in a mode takes its latest accepted timing under the first --label (repeatable, in priority order; default: unlabelled) it
has one under, so a hill-climb step's rollup is `--label <step> --label baseline`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemm_slowdown as GS  # noqa: E402

MODES = ("zk", "m0")
# prove: the median timed session's witness wait plus both reps' proving (gemm_slowdown.statement_seconds), Panel 1's "prover
# seconds per session"; session: the wall from the warm session's verdict to the last timed verdict, verifier round trips included
BASES = ("prove", "session")
FLAGS = {"zk": ["zk-sessions-verified-by-rust-not-lean"], "m0": []}


def family(spec: str) -> str:
    return spec.split("{", 1)[0]


def prefill_fraction(fid: str, calls: int | None, a) -> float:
    """The share of a Call Definition's Calls in the prefill: the program graph has one Call per token position (positions = prompt +
    generated - 1); an attention Call's T is its position's key count, so T <= prompt is the prefill's; a Definition Called once
    per generated token (the LM head) has one prefill Call; any other per-position Definition splits prompt : the rest."""
    positions = a.prompt + a.generated - 1
    if family(fid).startswith("Attention") and (m := re.search(r"\bT=(\d+)", fid)):
        return 1.0 if int(m.group(1)) <= a.prompt else 0.0
    if calls and calls % positions and calls % a.generated == 0:
        return 1 / a.generated
    return a.prompt / positions


def _counts(path: str) -> dict:
    return json.loads(Path(path).read_text())


def shape_definitions(c: dict) -> dict[str, dict[str, int]]:
    """shape -> {Call Definition: its row units of the shape} (units per Call x the row's Calls)."""
    out: dict[str, dict[str, int]] = {}
    for fid, per in c["per_definition"].items():
        for sh, u in per.items():
            out.setdefault(sh, {})[fid] = u * c["calls"][fid]
    return out


def census(a) -> dict:
    c = _counts(a.counts)
    sd = shape_definitions(c)
    shapes = []
    for sh, s in c["shapes"].items():
        ru = s["row_units"] or 0
        shapes.append({"shape": sh, "definition": s["definition"], "word_gates": s["word_gates"], "row_units": ru,
                       "gates": s["word_gates"] * ru, "call_definitions": sorted(sd.get(sh, {}))})
    shapes.sort(key=lambda x: (-x["gates"], x["shape"]))
    total = sum(x["gates"] for x in shapes)
    cum = 0
    for x in shapes:
        cum += x["gates"]
        x["share"], x["cum_share"] = x["gates"] / total, cum / total
    defs: dict[str, dict] = {}
    for sh, per in sd.items():
        wg = c["shapes"][sh]["word_gates"]
        for fid, u in per.items():
            d = defs.setdefault(fid, {"family": family(fid), "calls": c["calls"][fid], "units": 0, "gates": 0, "shapes": 0})
            d["units"] += u
            d["gates"] += u * wg
            d["shapes"] += 1
    fams: dict[str, dict] = {}
    for fid, d in defs.items():
        f = fams.setdefault(d["family"], {"call_definitions": 0, "calls": 0, "units": 0, "gates": 0})
        f["call_definitions"] += 1
        for k in ("calls", "units", "gates"):
            f[k] += d[k]
    for d in (*defs.values(), *fams.values()):
        d["share"] = d["gates"] / total
    out = {"schema": "vllm-overhead-census/v0", "counts": a.counts, "shapes_n": len(shapes), "call_definitions_n": len(defs),
           "row_units": sum(x["row_units"] for x in shapes), "gates": total,
           "shapes_for_90pct": next(i + 1 for i, x in enumerate(shapes) if x["cum_share"] >= 0.9),
           "families": dict(sorted(fams.items(), key=lambda kv: -kv[1]["gates"])),
           "call_definitions": dict(sorted(defs.items(), key=lambda kv: -kv[1]["gates"])), "shapes": shapes}
    if a.counts_all:
        ca = _counts(a.counts_all)
        left = {fid: n for fid, n in ca["calls"].items() if fid not in c["calls"]}
        sda = shape_definitions(ca)
        ex = {}
        for sh, per in sda.items():
            for fid, u in per.items():
                if fid in left:
                    e = ex.setdefault(fid, {"calls": left[fid], "units": 0, "gates": 0})
                    e["units"] += u
                    e["gates"] += u * ca["shapes"][sh]["word_gates"]
        tot_all = sum((s["row_units"] or 0) * s["word_gates"] for s in ca["shapes"].values())
        for e in ex.values():
            e["share_of_all"] = e["gates"] / tot_all if tot_all else None
        out["excluded_by_select"] = {"why": "left out by --select (the token-select Calls; GumbelTopPTokenSelect*: not provable in "
                                            "practice, one unit over the whole vocabulary row)", "gates_all": tot_all, "definitions": ex}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{out['shapes_n']} shapes, {out['call_definitions_n']} Call Definitions, {total:.4e} gates, 90% in "
          f"{out['shapes_for_90pct']} shapes")
    for i, x in enumerate(shapes[:15]):
        print(f"{i:3d} {x['shape'][:12]} wg={x['word_gates']:>8} units={x['row_units']:>12} share={x['share']:.4f} "
              f"cum={x['cum_share']:.4f} {x['definition'][:60]} [{len(x['call_definitions'])} defs]")
    return out


def _stamps(path: Path) -> tuple[float | None, float | None, int | None, list[tuple[float, dict]]]:
    start = end = rc = None
    live: list[tuple[float, dict]] = []
    for ln in path.read_text().splitlines() if path.exists() else []:
        t, _, rest = ln.partition("\t")
        if rest.startswith("START"):
            start, end, rc, live = float(t), None, None, []
        elif rest.startswith("EXIT"):
            end, rc = float(t), int(rest.partition("\t")[2])
        elif rest.startswith("LIVE\t"):
            live.append((float(t), json.loads(rest[5:])))
    return start, end, rc, live


def timing(a) -> dict:
    o = Path(a.out_k)
    recs = [json.loads(ln) for ln in (o / "classes" / "results.jsonl").read_text().splitlines() if ln.strip()] \
        if (o / "classes" / "results.jsonl").exists() else []
    rec = recs[-1] if recs else {}
    st, lv = rec.get("stage") or {}, rec.get("live") or {}
    start, end, rc, live = _stamps(o / "stamps.tsv")
    warm = [(t, x) for t, x in live if x.get("warm")]
    timed = [(t, x) for t, x in live if not x.get("warm")]
    zk = (o / "prover-zk").exists()
    env = {k: os.environ.get(k) for k in ("FC_PIPELINE_DEPTH", "FC_VERIFY_AHEAD", "FC_VERIFY_SERVERS", "FC_HOST_PREPIN", "BATCH", "N",
                                          "BATCH_ANDS", "MAX_STATEMENT_BITS", "FLOCK_GEMM_TILE", "FC_VERIFIERS") if os.environ.get(k)}
    out = {"schema": "vllm-overhead-timing/v0", "shape": rec.get("shape"), "definition": rec.get("definition"), "unit": rec.get("unit"),
           "word_gates": rec.get("word_gates"), "row_units": rec.get("row_units"), "ands": rec.get("ands"), "mode": "zk" if zk else "m0",
           "label": os.environ.get("VO_LABEL") or None, "env": env, "run": os.environ.get("RESEARCH_RUN_ID"), "t": time.time(),
           "binary_sha256": (o / "binary.sha256").read_text().split()[0] if (o / "binary.sha256").exists() else None,
           "stage_cached": rec.get("stage_cached"), "stage_s": rec.get("stage_s")}
    rejected = [x.get("run") for _, x in live if x.get("accepted") is not True]
    why = (rec.get("broke") or rec.get("skipped") or (None if lv.get("accepted") else (lv.get("why") or "no accepted statement"))
           or (f"the prove process exited {rc}" if rc != 0 else None)
           or (f"{len(warm)} warm and {len(timed)} timed LIVE records, not {a.warm} and {a.runs}" if (len(warm), len(timed)) != (a.warm, a.runs) else None)
           or (f"sessions {rejected} not accepted" if rejected else None))
    if why:
        out |= {"accepted": False, "why": why}
    else:
        units = (st.get("n") or lv["instances"]) * st.get("coords_per_instance", 1)
        t0 = warm[-1][0] if warm else start
        t_session = (timed[-1][0] - t0) / len(timed)
        t_prove = GS.statement_seconds(lv)
        verify = [(x.get("verdict") or {}).get("verify_s") for _, x in timed]
        out |= {"accepted": True, "n": st.get("n"), "tile": st.get("tile"), "units_per_statement": units, "k_log": lv.get("k_log"),
                "statement_digest": lv.get("statement_digest"), "timed_runs": len(timed), "warm_runs": len(warm),
                "session_s_per_statement": t_session, "prove_s_per_statement": t_prove,
                "session_s_per_unit": t_session / units, "prove_s_per_unit": t_prove / units,
                "e2e_s_median": statistics.median([x.get("e2e_s") or 0 for _, x in timed]),
                "verify_s_median": statistics.median([v for v in verify if v is not None]) if any(v is not None for v in verify) else None,
                "prover_peak_gb": lv.get("prover_peak_gb"), "verifier_peak_gb": lv.get("verifier_peak_gb"),
                "first_session_s": (timed[0][0] - t0), "process_s": (end - start) if end and start else None}
    (o / "timing.json").write_text(json.dumps(out, indent=1))
    print("timing", json.dumps({k: out.get(k) for k in ("shape", "mode", "accepted", "why", "units_per_statement",
                                                         "session_s_per_unit", "prove_s_per_unit")}))
    return out


def _timings(paths: list[str]) -> list[dict]:
    out = []

    def take(p: Path):
        d = json.loads(p.read_text())
        if d.get("schema") == "vllm-overhead-timing/v0":
            out.append(d | {"_file": str(p)})
        elif isinstance(d.get("timings"), list):
            out.extend(t | {"_file": str(p)} for t in d["timings"])

    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*.json")):
                if f.name == "timing.json" or "summar" in f.name or f.parent.name == "summaries":
                    try:
                        take(f)
                    except (json.JSONDecodeError, OSError):
                        pass
        else:
            take(p)
    seen, uniq = set(), []
    for t in out:
        k = (t.get("run"), t.get("shape"), t.get("mode"), t.get("label"))
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def rollup(a) -> dict:
    c = _counts(a.counts)
    sd = shape_definitions(c)
    nat = json.loads(Path(a.native).read_text())
    shapes = {sh: s | {"row_units": s["row_units"] or 0, "gates": s["word_gates"] * (s["row_units"] or 0)} for sh, s in c["shapes"].items()}
    total = sum(s["gates"] for s in shapes.values())
    labels = a.label or [None]
    tims = _timings(a.timings)
    modes = {}
    for mode in MODES:
        best: dict[str, dict] = {}
        for lab in reversed(labels):
            for t in sorted((t for t in tims if (t.get("label") or None) == lab and t.get("mode") == mode and t.get("accepted")
                             and t.get("shape") in shapes), key=lambda t: t["t"]):
                best[t["shape"]] = t
        if not best:
            continue
        m = {b: _weigh(a, c, sd, shapes, total, nat, best, b) for b in BASES}
        # a tile statement is a cost, not the row's claim, until it is reviewed (gemm_hill.py's flag; red-team-proofs-554)
        tiled = sorted(sh for sh, t in best.items() if t.get("tile"))
        modes[mode] = m[a.basis] | {"flags": FLAGS[mode] + (["tile-statement-unreviewed"] if tiled else []), "tiled_shapes": tiled,
                                    "other_bases": {b: {"prover_s": v["prover_s"], "overhead": v["overhead"],
                                                        "phases": v["phases"]} for b, v in m.items() if b != a.basis}}
    out = {"schema": "vllm-overhead/v0", "row": a.row, "basis": a.basis, "label": labels[0], "labels": labels, "native": nat, "counts": a.counts,
           "gates": total, "shapes_n": len(shapes), "modes": modes,
           "flags": ["outputs-public"] + (a.flag or []), "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.census and Path(a.census).exists():
        ce = json.loads(Path(a.census).read_text())
        if ce.get("excluded_by_select"):
            out["excluded"] = ce["excluded_by_select"]
    Path(a.out).write_text(json.dumps(out, indent=1))
    for mode, m in modes.items():
        alt = ", ".join(f"{b} {v['overhead']:.4g}" for b, v in m["other_bases"].items())
        print(f"{mode}: {m['overhead']:.4g} x-native ({m['prover_s']:.4g} s / {nat['all_s']:.4g} s; {alt}), coverage "
              f"{m['coverage_gates']:.4f} of gates in {m['shapes_measured']} shapes, extrapolated {m['extrapolated_s'] / m['prover_s']:.4f} of seconds")
        if m["tiled_shapes"]:
            print(f"   tile-statement-unreviewed: {', '.join(sh[:8] for sh in m['tiled_shapes'])}")
        for fid, d in list(m["call_definitions"].items())[:10]:
            print(f"   {d['share']:.4f}  {d['seconds']:.4g} s  {fid[:100]}")
    return out


def _weigh(a, c: dict, sd: dict, shapes: dict, total: int, nat: dict, best: dict[str, dict], basis: str) -> dict:
    """One mode's weighted seconds on one basis: measured shapes at their seconds per unit, the rest per word gate."""
    spg = {sh: t[f"{basis}_s_per_unit"] / shapes[sh]["word_gates"] for sh, t in best.items()}
    fam_spg: dict[str, list] = {}
    for sh in best:
        f = fam_spg.setdefault(family(shapes[sh]["definition"]), [0.0, 0])
        f[0] += spg[sh] * shapes[sh]["gates"]
        f[1] += shapes[sh]["gates"]
    fam_rate = {f: s / g for f, (s, g) in fam_spg.items() if g}
    median = statistics.median(spg.values())
    per_shape, defs, fams = {}, {}, {}
    for sh, s in shapes.items():
        if sh in best:
            sec, how = best[sh][f"{basis}_s_per_unit"] * s["row_units"], "measured"
        elif family(s["definition"]) in fam_rate:
            sec, how = fam_rate[family(s["definition"])] * s["gates"], "family"
        else:
            sec, how = median * s["gates"], "median"
        per_shape[sh] = {"seconds": sec, "how": how}
        split = sd.get(sh) or {s["definition"]: s["row_units"]}
        tot_u = sum(split.values()) or 1
        for fid, u in split.items():
            d = defs.setdefault(fid, {"family": family(fid), "units": 0, "gates": 0, "measured_gates": 0, "measured_s": 0.0,
                                      "extrapolated_s": 0.0, "prefill_fraction": prefill_fraction(fid, c["calls"].get(fid), a)})
            d["units"] += u
            d["gates"] += u * s["word_gates"]
            d["measured_gates"] += u * s["word_gates"] if how == "measured" else 0
            d["measured_s" if how == "measured" else "extrapolated_s"] += sec * u / tot_u
    for fid, d in defs.items():
        f = fams.setdefault(d["family"], {"call_definitions": 0, "units": 0, "gates": 0, "measured_s": 0.0, "extrapolated_s": 0.0})
        f["call_definitions"] += 1
        for k in ("units", "gates", "measured_s", "extrapolated_s"):
            f[k] += d[k]
    num = sum(p["seconds"] for p in per_shape.values())
    meas_g = sum(shapes[sh]["gates"] for sh in best)
    for d in (*defs.values(), *fams.values()):
        d["seconds"] = d["measured_s"] + d["extrapolated_s"]
        d["share"] = d["seconds"] / num
        d["gate_share"] = d["gates"] / total
    phases = {}
    for ph in ("prefill", "decode"):
        w = {fid: d["prefill_fraction"] if ph == "prefill" else 1 - d["prefill_fraction"] for fid, d in defs.items()}
        g = sum(d["gates"] * w[fid] for fid, d in defs.items())
        phases[ph] = {"prover_s": sum(d["seconds"] * w[fid] for fid, d in defs.items()), "gates": g,
                      "coverage_gates": sum(d["measured_gates"] * w[fid] for fid, d in defs.items()) / g if g else None}
        phases[ph]["overhead"] = phases[ph]["prover_s"] / nat[f"{ph}_s"] if nat.get(f"{ph}_s") else None
    return {
        "prover_s": num, "measured_s": sum(p["seconds"] for p in per_shape.values() if p["how"] == "measured"), "phases": phases,
        "extrapolated_s": sum(p["seconds"] for p in per_shape.values() if p["how"] != "measured"),
        "coverage_gates": meas_g / total, "shapes_measured": len(best),
        "extrapolated_by": {h: sum(shapes[sh]["gates"] for sh, p in per_shape.items() if p["how"] == h) / total for h in ("family", "median")},
        "overhead": num / nat["all_s"], "median_s_per_gate": median, "family_s_per_gate": fam_rate,
        "families": dict(sorted(fams.items(), key=lambda kv: -kv[1]["seconds"])),
        "call_definitions": dict(sorted(defs.items(), key=lambda kv: -kv[1]["seconds"])),
        "shapes_measured_detail": {sh: {k: t.get(k) for k in ("run", "definition", "word_gates", "row_units", "units_per_statement", "n", "tile",
                                                               "session_s_per_unit", "prove_s_per_unit", "session_s_per_statement",
                                                               "prove_s_per_statement", "verify_s_median", "env", "binary_sha256",
                                                               "statement_digest", "label")}
                                   | {"seconds": per_shape[sh]["seconds"], "gate_share": shapes[sh]["gates"] / total}
                                   for sh, t in sorted(best.items(), key=lambda kv: -shapes[kv[0]]["gates"])}}


def point(a) -> None:
    d = json.loads(Path(a.rollup).read_text())
    _, model, phase, mode = a.config.split("/")
    m = d["modes"][mode]
    if phase == "all":
        prover_s, native_s, cov = m["prover_s"], d["native"]["all_s"], m["coverage_gates"]
    else:
        ph = m["phases"][phase]
        prover_s, native_s, cov = ph["prover_s"], d["native"][f"{phase}_s"], ph.get("coverage_gates")
    out = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"config": a.config, "row": d["row"], "phase": phase, "mode": mode, "basis": d["basis"],
                                    "label": d.get("label"), "flags": d["flags"] + m.get("flags", []), "rollup_sha256": _sha(a.rollup)}
                                   | ({"rollup_art": a.rollup_art} if a.rollup_art else {}),
           "validation": {"status": "passed", "checker": "every measured shape's sessions accepted (vllm_overhead.py timing)",
                          "detail": f"coverage {cov:.4f} of the row's word gates measured; the rest priced per gate", "evidence": []},
           "measurements": [{"name": "overhead", "value": prover_s / native_s, "unit": "x-native"},
                            {"name": "prover_s", "value": prover_s, "unit": "s"}, {"name": "native_s", "value": native_s, "unit": "s"},
                            {"name": "coverage_gates", "value": cov, "unit": "fraction"}],
           "measurement_files": [], "artifacts": []}
    tmp = Path(a.result).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.result)
    print(a.config, f"{prover_s / native_s:.4g}")


def _sha(p: str) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd_", required=True)
    c = sub.add_parser("census")
    c.add_argument("--counts", required=True)
    c.add_argument("--counts-all")
    c.add_argument("--out")
    t = sub.add_parser("timing")
    t.add_argument("out_k")
    t.add_argument("--warm", type=int, default=1)
    t.add_argument("--runs", type=int, default=3)
    r = sub.add_parser("rollup")
    r.add_argument("--counts", required=True)
    r.add_argument("--census")
    r.add_argument("--timings", nargs="+", required=True)
    r.add_argument("--native", required=True, help="JSON: all_s, prefill_s, decode_s and where they came from")
    r.add_argument("--row", required=True)
    r.add_argument("--basis", choices=BASES, default="prove")
    r.add_argument("--label", action="append", help="the timings' VO_LABEL; repeated, in priority order: a shape takes the first "
                   "label it has an accepted timing under (a hill-climb step's label, then baseline)")
    r.add_argument("--flag", action="append")
    r.add_argument("--prompt", type=int, default=1024)
    r.add_argument("--generated", type=int, default=128)
    r.add_argument("--out", required=True)
    p = sub.add_parser("point")
    p.add_argument("--rollup", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--rollup-art", default=None, help="the rollup's art id in the evidence store (vllm-overhead/v0)")
    a = ap.parse_args(argv)
    out = {"census": census, "timing": timing, "rollup": rollup, "point": point}[a.cmd_](a)
    # a shape job whose statement wasn't accepted fails, so the dispatcher's record says so
    return 1 if a.cmd_ == "timing" and not out["accepted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
