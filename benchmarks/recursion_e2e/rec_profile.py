"""The recursion bench's result (``recursion-e2e/v1``) from a job's ``out/`` (``job.sh``); ``rec_table`` renders its table
(README.md).

``read(OUT)`` walks ``OUT/steps/<step>/r<i>/out/``, the records 85-rec-reprice.sh wrote for each invocation, and returns one
row per timed thing: a prover's runs (``inner``, ``oprove``: the timed runs' ``LIVE`` lines), or a tool's
``/usr/bin/time -v`` over the repeats (``icheck``, ``vstage``, ``vcap``, ``oreplay``, ``lean``), or the build once, and one row
per flipped inner coin (``negative``). Every time is a ``Stat`` (median, min, max and the values); nothing is averaged across
different things. ``headline`` says whether the run is a whole recursive run (every step ran), whether its parts passed
(every step exited 0, every honest verdict accepted, every flipped coin refused), whether Lean composed V*'s parts into
the recursive statement, and whether it is an accepted recursive run: its parts passed and Lean accepted the
composition. A record line or file that doesn't parse (a
line cut short when its run stopped) is read as absent, and each one is logged at debug level. Stdlib only: the job runs
it on the node's ``python3``. What one record file says is ``rec_logs``'s, apart from Flock's own output formats, read
here, and the rows that read none of those are built in ``rec_rows``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rec_logs import (
    COMPOSITION,
    composition,
    exit_status,
    gpu_peak,
    lean_verdict,
    live_lines,
    params,
    processes,
    tagged,
    text,
    timev,
)
from rec_rows import Row, Stat, compose_row, forge_after_rows, forged, forgeries, refused, step_row, tool_rows

log = logging.getLogger(__name__)

SCHEMA = "recursion-e2e/v1"
#: the steps one recursive run needs, in order (vcap stages RecCap, which V*'s verdict needs; ``negative`` checks the
#: run, ``icheck`` and ``oreplay`` time more of it)
E2E_STEPS = ("build", "inner", "vstage", "vcap", "oprove", "lean")
#: one recursive session's steps after its build: the inner session, V*'s staging, its outer proofs, Lean's verdicts
SESSION_STEPS = E2E_STEPS[1:]
ORDER = ("build", "inner", "icheck", "vstage", "vcap", "oprove", "oreplay", "lean", "forge-after", "negative")
#: what each part of the recursive run is, recorded in the result's ``inputs``
STATEMENTS = {
    "inner": "N x GemmCoordinate_v2{K,DOT=HopperBF16WgmmaDot16_v1} under Q_template_instance (rec_reprice.py), "
             "verity/flock-circuit+plain-leaves, M0 (no --zk), on the challenger's coin tree (FC_COINS=tree), through "
             "rec_live's proxy",
    "vstar": "V*'s statements over the inner's last proxied run: each Ligerito level's RecOpen and RecCap, the algebra's "
             "InnerRepCheck parts (rec_vstage)",
    "outer": "each V* statement with --zk on loopback, every unit drawn, one at a time, in each of MODES",
    "verify": "verity-verify-proof verify --zk --rec-inner-session --rec-inner-transcript on each V* release (the last mode's), "
              "and verity-verify-proof statement on the inner statement",
    "negative": "each V* release verified again over a copy of the inner run with rep 0's coin flipped "
                "(rec_inner.py forge), which some statement must refuse",
    "forged": "85's FORGE statements (rec_vstage --forge: a prover row that does not open the verifier's registration), "
              "which the firewall must stop or Lean refuse",
    "forge-after": "coef, dirs and message made after proving on the honest L0 and alg-p0 releases' statements: row 0 one bit "
                   "off as its b || c, as pub.bin's consistent table, then as the whole registration (forge_after.py), each "
                   "refused by Lean beside its honest control",
}
#: the prover's phase buckets (``flock-circuit``'s ``gpu_buckets``, both reps) a dominant phase is chosen among; the
#: aggregates (``t.arithmetic``, ``t.total``) and host-side ``h.*`` stay out
PHASES = ("t.witness", "t.encoding_commitment", "t.link", "t.zerocheck", "t.lincheck", "t.ring_switch", "t.ligerito",
          "t.firewall_recheck", "t.lean_finish", "t.zk_inner")


def lean_split(p: Path) -> dict[str, float]:
    """verity-verify-proof's stderr: ``inputs: N ms`` (from its start), ``setup: N ms`` (from the verification's start), then the verification's
    whole time (``/path: N ms``, setup included), as 85's summary reads them."""
    ms = [int(x) for x in re.findall(r"^(?:inputs|setup(?: \(drawn\))?|/\S+): (\d+) ms$", text(p), re.M)]
    return {"inputs_s": ms[0] / 1e3, "setup_s": ms[1] / 1e3, "session_s": (ms[2] - ms[1]) / 1e3} if len(ms) == 3 else {}


def _bucket_sums(buckets: list[dict[str, Any]] | None) -> dict[str, float]:
    sums: dict[str, float] = {}
    for b in buckets or []:
        for k, v in b.items():
            if k.startswith("t.") and isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
    return sums


def dominant(v: dict[str, Any]) -> dict[str, Any] | None:
    """The largest part of one run's ``e2e_s``: a prover phase (both reps' bucket summed), the host witness
    (``witness_s``), or what is left, ``wait.verifier`` (the run waiting on its verifier's coins and verdict, and on
    whatever else the prover's buckets don't time); its share of ``e2e_s`` (of ``prove_total_s`` without one)."""
    parts = {k: x for k, x in _bucket_sums(v.get("buckets")).items() if k in PHASES}
    if not parts:
        return None
    total, e2e = v.get("prove_total_s"), v.get("e2e_s")
    if e2e and total is not None:
        parts["h.witness"] = v.get("witness_s") or 0.0
        parts["wait.verifier"] = max(0.0, e2e - total - parts["h.witness"])
    k = max(parts, key=lambda name: parts[name])
    whole = e2e or total
    return {"phase": k, "s": round(parts[k], 4), "share": round(parts[k] / whole, 3) if whole else None,
            "of": "e2e_s" if e2e else "prove_total_s",
            "phases": {p: round(x, 4) for p, x in sorted(parts.items(), key=lambda kv: -kv[1])}}


def prover_row(d: Path, stage: str, step: str, what: str) -> Row:
    """One prove_live directory: the timed runs (warm ones left out), the invocation's peak RSS, the GPU's peak."""
    stamped = live_lines(d / "prove.tsv")
    rows = [v for _, v in stamped]
    timed = [v for v in rows if not v.get("warm")]
    r = Row(stage, step, what, "gpu")
    # 85's `again`: "<fn> <name>: lease preempted, try <t>" in the invocation's out/preempted.txt
    preempted = sum(1 for ln in text(d.parent.parent / "preempted.txt").splitlines() if f" {d.name}: lease preempted" in ln)
    if not timed:
        err = text(d / "prove.err")
        held = re.findall(r"^gpu-lease: GPU .* held (\d+m\d+s)", err, re.M)
        stops = [ln for ln in err.splitlines() if ln.startswith(("FIREWALL-REFUSED", "PROVER-STOPPED")) or "panicked" in ln]
        why = ([f"lease preempted {preempted}x"] if preempted else []) + [
            f"no run finished in the lease's {held[-1]}" if held else "no run finished before the step stopped"]
        r.ok, r.note = False, "; ".join(why + stops[:1])[:300] if d.exists() else "not run"
        r.extra = {"preempted": preempted, "lease_held": held}
        return r
    r.wall_s = Stat.of(v.get("prove_total_s") for v in timed)
    r.e2e_s = Stat.of(v.get("e2e_s") for v in timed)
    tv = timev(d / "time-prove.txt")
    r.peak_rss_gb = tv["peak_rss_gb"] if tv else None
    r.gpu_peak_mib = gpu_peak(d / "gpu-mib.txt")
    med = sorted(timed, key=lambda v: v.get("prove_total_s") or 0)[len(timed) // 2]
    pb = med.get("proof_bytes")
    r.proof_bytes = sum(pb) if isinstance(pb, list) else pb
    if med.get("wire_sent") is not None:
        r.wire_bytes = {"sent": med.get("wire_sent"), "recv": med.get("wire_recv")}
    r.dominant = dominant(med)
    served = tagged(d / "serve.log", "SESSION")
    verdicts = [v.get("accepted") for v in rows] + [v.get("accepted") for v in served]
    if all(x is None for x in verdicts):
        r.note = str(timed[-1].get("why") or "no verdict")[:200]
    else:
        r.ok = all(x is True for x in verdicts)
        if not r.ok:
            r.note = next((str(v.get("prover_stopped") or v.get("why")) for v in rows
                           if v.get("accepted") is not True and (v.get("prover_stopped") or v.get("why"))), "")[:300]
    # 85's ``session_s``: from the last warm run's verdict to the last timed one's, over the timed runs (runs
    # overlap up to FC_VERIFY_AHEAD, so this, not e2e_s, is the step's pace)
    t_warm = [t for t, v in stamped if v.get("warm") and t is not None]
    t_timed = [t for t, v in stamped if not v.get("warm") and t is not None]
    session_s = (t_timed[-1] - t_warm[-1]) / len(t_timed) if t_warm and t_timed else None
    r.extra = {"sessions": len(rows), "timed": len(timed), "k_log": med.get("k_log"), "dense_m": med.get("dense_m"),
               "instances": med.get("instances"), "session_s": session_s, "verify_ahead": med.get("verify_ahead"),
               "witness_s": Stat.of(v.get("witness_s") for v in timed), "tp_s": Stat.of(v.get("tp_s") for v in timed),
               "serve_verify_s": Stat.of(v.get("verify_s") for v in served[-len(timed):]) if served else None,
               "serve_handle_s": Stat.of(v.get("handle_s") for v in served[-len(timed):]) if served else None,
               "invocation_wall_s": tv["wall_s"] if tv else None, "prover_cpu_s": tv["cpu_s"] if tv else None,
               "cores_busy_before": text(d / "bgload.txt").strip() or None, "preempted": preempted, "sizes": text(d / "sizes.txt").strip() or None,
               "why": [v.get("why") for v in rows if v.get("accepted") is not True][:3],
               "lease": [ln for ln in text(d / "prove.err").splitlines() if ln.startswith("gpu-lease:")][:4]}
    r.extra = {k: (asdict(v) if isinstance(v, Stat) else v) for k, v in r.extra.items()}
    return r


def repeats(root: Path, step: str) -> list[Path]:
    """The step's invocations' out/ directories, r1, r2, ... in order."""
    d = root / "steps" / step
    return [p / "out" for p in sorted(d.glob("r*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0)] if d.is_dir() else []


def _names(outs: list[Path], sub: str, pattern: str) -> list[str]:
    """The names of the files matching ``pattern`` (``time-(.+).txt``) under ``out/<sub>`` of any repeat, in name order."""
    rx = re.compile(pattern)
    names = {m.group(1) for o in outs for f in (o / sub).glob("*") if (m := rx.fullmatch(f.name))}
    return sorted(names, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])


def _upstream(o: Path, rel: str) -> bool | None:
    up = tagged(o / rel, "UPSTREAM")
    return (up[0].get("verdict") or {}).get("accepted") is True if up else None


def read(root: Path) -> dict[str, Any]:
    """The profile of one job's ``out/`` (``root``)."""
    rows: list[Row] = []
    b = repeats(root, "build")
    if b:
        o = b[0]
        t = timev(o.parent / "time-step.txt")
        r = Row("build", "build", "85 STEP=build: the CUDA prover, verity-verify-proof and both stagings, as the job found its caches",
                "setup", wall_s=Stat.of([t["wall_s"]]) if t else None, peak_rss_gb=t["peak_rss_gb"] if t else None,
                ok=text(o.parent / "rc").strip() == "0")
        r.extra = {"binary": text(o / "binary.sha256").split()[:1], "lean": text(o / "lean.sha").split()[:1],
                   "rebuilt_prover": (o / "build.log").exists()}
        for sub in ("stage", "stage-zk"):
            st = timev(o / sub / "time-stage.txt")
            r.extra[sub] = {"wall_s": st and st["wall_s"], "peak_rss_gb": st and st["peak_rss_gb"],
                            "sizes": text(o / sub / "sizes.txt").strip()}
        rows.append(r)
    for o in repeats(root, "inner")[:1]:
        for sub, stage, what in (("proxy", "inner M0", "the inner statement (plain leaves), ZK off, against the rec_live proxy"),
                                 ("m0", "baseline M0", "the non-recursive hm96 statement, ZK off, loopback serve"),
                                 ("zk", "baseline --zk", "the non-recursive hm96 statement with --zk, loopback serve")):
            if (o / "inner" / sub).exists():
                rows.append(prover_row(o / "inner" / sub, stage, "inner", what))
    rows += tool_rows(repeats(root, "icheck"), "icheck", {
        "inner replay": ("icheck/time-replay-proxy.txt", "flock-circuit replay of the last inner run (upstream's verifier)"),
        "rec_vstar": ("icheck/time-vstar.txt", "V*'s hashing reference over the inner run")},
        verdict=lambda o, n: _upstream(o, "icheck/replay-proxy.txt") if n == "inner replay" else None)
    vs = repeats(root, "vstage")
    if vs:
        rel = {"V* points": ("vstage/time-points.txt", "the verifier's fold points from the inner coins (rec_vstage points)"),
               "inner fold": ("vstage/time-fold.txt", "InnerFold.lean as an executable: extra claims and comb")}
        for n in _names(vs, "vstage", r"time-(.+)\.txt"):
            if n not in ("points", "fold", "cap"):
                rel[f"stage {n}"] = (f"vstage/time-{n}.txt", "rec_vstage stage of the verifier's own values alone (no level, no "
                                     "algebra)" if n == "verifier" else f"rec_vstage stage of V*'s statement {n}")
        rows.append(step_row(vs, "vstage", "85 STEP=vstage whole (its statements staged JOBS at a time)"))
        rows += tool_rows(vs, "vstage", rel)
        units = staged_units(vs[0] / "summary-vstage.json")
        for r in rows:
            name = r.stage[6:] if r.step == "vstage" and r.stage.startswith("stage ") else None
            if name == "alg":
                r.extra["units"] = {k: v for k, v in units.items() if k.startswith("alg-")} or None
            elif name and units.get(name):
                r.extra["units"] = units[name]
    vc = repeats(root, "vcap")
    if vc:
        rows.append(step_row(vc, "vcap", "85 STEP=vcap whole"))
    rows += tool_rows(vc, "vcap", {"stage cap": ("vstage/time-cap.txt", "RecCap at every level")})
    cpu_prover = params(root).get("OPROVE_GPU") == "0"
    for o in repeats(root, "oprove")[:1]:
        for d in sorted((p for p in (o / "oprove").glob("*") if p.is_dir()), key=lambda p: p.name):
            mode = next((m for m in ("firewall", "host", "dev") if d.name.endswith("-" + m)), "dev")
            stmt = d.name[: -len(mode) - 1] if d.name.endswith("-" + mode) else d.name
            rows.append(prover_row(d, f"outer {stmt}", "oprove", f"V*'s {stmt} with --zk, mode {mode}"
                                   + (", CPU prover" if cpu_prover else "")))
            rows[-1].extra |= {"mode": mode, "statement": stmt, "prover": "cpu" if cpu_prover else "gpu"}
            if stmt.startswith("forged-"):
                refused(rows[-1])
            if mode == "firewall" and rows[-1].wall_s:
                rows[-1].note = ("the prover's phase buckets include its waits on the Lean session loop (flock-firewall), "
                                 "the reps' overlapping; " + rows[-1].note).rstrip("; ")
    orr = repeats(root, "oreplay")
    if orr:
        rows += tool_rows(orr, "oreplay", {f"replay {n}": (f"oreplay/{n}/time-replay.txt", f"flock-circuit replay --zk of {n}")
                                           for n in _names(orr, "oreplay", r"([^.]+)")},
                          verdict=lambda o, n: _upstream(o, f"oreplay/{n[len('replay '):]}/replay.txt"))
    ln = repeats(root, "lean")
    if ln:
        names = [n for n in _names(ln, "lean", r"time-(.+)\.txt")]
        rel = {f"lean {n}": (f"lean/time-{n}.txt", ("verity-verify-proof statement on the inner statement" if n == "inner-statement"
                                                    else "verity-verify-proof verify on the inner run" if n == "inner"
                                                    else f"verity-verify-proof verify --zk on V*'s {n} release")) for n in names}

        def lv(o: Path, n: str) -> bool | None:
            name = n[len("lean "):]
            if name == "inner-statement":
                return None
            v = lean_verdict(o / "lean" / f"{name}.log")
            return v.get("accepted") is True if v else False

        lrows = tool_rows(ln, "lean", rel, verdict=lv)
        for r in lrows:
            name = r.stage[len("lean "):]
            if name.startswith("forged-"):
                r.extra["forged"] = True
                if r.ok is not None:
                    r.extra["refused"] = r.ok is False and all(lean_verdict(o / "lean" / f"{name}.log") for o in ln)
                    r.ok = r.extra["refused"]
            splits = [lean_split(o / "lean" / f"{name}.err") for o in ln]
            parts = {}
            for k in ("inputs_s", "setup_s", "session_s"):
                stat = Stat.of(x.get(k) for x in splits)
                if stat:
                    r.extra[k] = asdict(stat)
                    parts[k] = stat.median
            # verity-verify-proof's own split: its inputs read, the verification's setup, and its checks after it
            if parts and r.wall_s:
                parts = {"lean.inputs": parts.get("inputs_s", 0.0), "lean.setup": parts.get("setup_s", 0.0),
                         "lean.checks": max(0.0, parts.get("session_s", 0.0))}
                k = max(parts, key=lambda name: parts[name])
                r.dominant = {"phase": k, "s": round(parts[k], 4), "share": round(parts[k] / r.wall_s.median, 3), "of": "wall_s",
                              "phases": {p: round(x, 4) for p, x in sorted(parts.items(), key=lambda kv: -kv[1])}}
            why = [str((lean_verdict(o / "lean" / f"{name}.log") or {}).get("why"))[:200] for o in ln]
            if r.ok is False or r.extra.get("refused"):
                r.note = next((w for w in why if w and w != "None"), text(ln[0] / "lean" / f"{name}.log")[-200:].strip())
        rows.append(step_row(ln, "lean", "85 STEP=lean whole (JOBS verifications at once), as an auditor waits"))
        rows += lrows
        rows.append(compose_row([composition(o) for o in ln]))
    fa = repeats(root, "forge-after")
    if fa:
        rows.append(step_row(fa, "forge-after", "forge_after.py: every forged case and honest control, LEAN_JOBS at once"))
        rows += forge_after_rows(fa[-1] / "forge-after.json")
    ng = repeats(root, "negative")
    if ng:
        rows.append(step_row(ng, "negative", "85 STEP=lean with NEG_INNER: every V* release again, over each flipped coin"))
        rows += negative_rows(ng[0] / "lean")
    forgeries(rows)
    s = summary(rows)
    fw = processes(root.parent / "resources.jsonl", "flock-firewall")
    if fw:
        s["flock_firewall"] = fw
    steps = step_status(root)
    staged = json.loads(text(root / "stage.json") or "null")
    inputs = STATEMENTS if staged is None else {**STATEMENTS, "inner": (
        f"{staged['n']} x {staged['unit']} ({staged['subcircuit']}, m = {staged['m']}) under Q_template_instance, staged by "
        + (f"stage.sh from served window {staged['window']['run'][:16]}'s member {staged['window']['member']}"
           if staged.get("window") else "stage.sh from seeded draws (generated values)")
        + ", verity/flock-circuit+plain-leaves, M0 (no --zk), on the challenger's coin tree (FC_COINS=tree), through "
          "rec_live's proxy")}
    doc = {"schema": SCHEMA, "params": params(root), "inputs": inputs, "rows": [r.to_json() for r in rows], "summary": s,
           "steps": steps, **({"staged": staged} if staged else {})}
    side = release(root)
    if side:
        doc["release"] = side
    doc["headline"] = headline(rows, s, steps) | ({"released": side["ok"]} if side else {})
    if doc["params"].get("release") == "1":
        for k in ("ok", "parts_ok"):
            doc["headline"][k] = doc["headline"][k] and bool(side and side["ok"])
    # each session step's first invocation's /usr/bin/time -v (None for a step that didn't run), what a session's cost is
    # scaled from
    doc["session_times"] = {x: timev(outs[0].parent / "time-step.txt") if (outs := repeats(root, x)) else None
                            for x in SESSION_STEPS}
    return doc


def release(root: Path) -> dict[str, Any] | None:
    """The recursive session through the ZK server's release side (85's RELEASE=1): oprove's ``release.json`` (each part's
    release time against when the Lean loop decided it), each part's prove wall (``release-wall.tsv``), the schedule the
    inner step fixed and the inner coin trees' audit against the challenger's release log (``tree-held.json``)."""
    o = next(iter(repeats(root, "oprove")), None)
    if o is None:
        return None
    rep = json.loads(text(o / "oprove" / "release.json") or "null")
    if rep is None:
        return None
    wall: dict[str, dict[str, Any]] = {}
    for ln in text(o / "oprove" / "release-wall.tsv").splitlines():
        part, d, t0, t1, rc = ln.split("\t")
        wall[part] = {"dir": d, "started_ms": int(t0), "ended_ms": int(t1), "rc": int(rc)}
    for p in rep["parts"]:
        w = wall.get(p["part"])
        p["prove_wall_s"] = (w["ended_ms"] - w["started_ms"]) / 1000 if w else None
        p["rc"] = w["rc"] if w else None
    i = next(iter(repeats(root, "inner")), None)
    held = json.loads(text(i / "inner" / "tree-held.json") or "null") if i else None
    return rep | {"schedule": json.loads(text(i / "inner" / "release.json") or "null") if i else None,
                  "tree_source": text(i / "inner" / "tree-source.txt").strip() if i else None,
                  "trees_held": held and held.get("ok"), "ok": bool(rep["ok"] and (held is None or held.get("ok")))}


def staged_units(p: Path) -> dict[str, dict[str, Any]]:
    """What `rec_vstage.stage` printed of each statement it staged (85's ``summary-vstage.json``): its template,
    instances, rows, k_log, seconds and (the algebra's parts) its lowering's seconds and circuit bytes, by statement
    name."""
    try:
        st = json.loads(text(p) or "{}").get("statements") or {}
    except ValueError as e:
        log.debug("%s isn't JSON, read as no staged statement (%s)", p, e)
        return {}
    keep = ("template", "instances", "unit_rows", "unit_log", "k_log", "stage_s", "build_s", "circuit_bytes")
    out: dict[str, dict[str, Any]] = {}
    for s in st.values():
        u = s.get("stage") if isinstance(s, dict) else None
        if not isinstance(u, dict):
            continue
        for name, x in (u.get("levels") or {}).items():
            out[name] = {k: x.get(k) for k in keep}
        if isinstance(u.get("cap"), dict) and "template" in u["cap"]:
            out["cap"] = {k: u["cap"].get(k) for k in keep}
        for name, x in (u.get("caps") or {}).items():
            out[f"cap-{name}"] = {k: x.get(k) for k in keep}
        for x in u.get("algebra") or []:
            out[f"alg-p{x.get('part')}"] = {k: x.get(k) for k in keep}
    return out


_FORGED = re.compile(r"forged-inner-([a-z]+)-(.+)")


def negative_rows(d: Path) -> list[Row]:
    """One row per flipped coin (85's ``forged-inner-COIN-STATEMENT`` parts, read as ``lean_verdicts.py`` reads them): a
    statement refuses it when verity-verify-proof exited 1 with a VERDICT and none accepting; the coin is refused when one statement
    refuses it and every honest statement has a verdict over it (a statement that reads no flipped coin rightly accepts)."""
    honest = sorted(m.group(1) for f in d.glob("time-*.txt") if (m := re.fullmatch(r"time-(.+)\.txt", f.name))
                    and not m.group(1).startswith(("forged-", "inner")))
    coins: dict[str, dict[str, dict[str, Any]]] = {}
    for f in sorted(d.glob("time-forged-inner-*.txt")):
        m = _FORGED.fullmatch(f.name[5:-4])
        if not m:
            continue
        v = lean_verdict(d / f"{m.group(0)}.log")
        tv = timev(f)
        coins.setdefault(m.group(1), {})[m.group(2)] = {
            "exit": exit_status(f), "accepted": v.get("accepted") if v else None, "why": str((v or {}).get("why") or "")[:200],
            "wall_s": tv["wall_s"] if tv else None, "peak_rss_gb": tv["peak_rss_gb"] if tv else None}
    rows = []
    for coin, parts in coins.items():
        refused_by = sorted(n for n, x in parts.items() if x["exit"] == 1 and x["accepted"] is False)
        accepted_by = sorted(n for n, x in parts.items() if x["accepted"] is True)
        silent = sorted(set(honest) - {n for n, x in parts.items() if x["accepted"] is not None})
        r = Row(f"negative inner-{coin}", "negative", f"rep 0's {coin} coin flipped in the inner run, under every V* statement",
                "cpu", peak_rss_gb=max((x["peak_rss_gb"] for x in parts.values() if x["peak_rss_gb"] is not None), default=None))
        r.ok = bool(refused_by) and not silent
        r.note = (f"refused by {', '.join(refused_by) or 'no statement'}"
                  + (f"; accepted by {', '.join(accepted_by)} (reads no flipped coin)" if accepted_by else "")
                  + (f"; no verdict from {', '.join(silent)}" if silent else ""))
        r.extra = {"refused_by": refused_by, "accepted_by": accepted_by, "no_verdict": silent,
                   "why": sorted({parts[n]["why"] for n in refused_by})[:4], "parts": parts}
        rows.append(r)
    return rows


def headline(rows: list[Row], s: dict[str, Any], steps: dict[str, list[str]]) -> dict[str, Any]:
    """Whether the run is one whole recursive run (``complete``: every step of ``E2E_STEPS`` ran), whether its parts passed
    (``parts_ok``: it is whole, every invocation exited 0, no row refused or unrun, and every flipped coin refused),
    whether Lean composed V*'s parts into the recursive statement (``composed``, `compose_row`'s; ``not_composed`` is why
    not; ``refused`` is Lean's reason where it refused the composition, a forced retry among them), and whether it is an
    accepted recursive run (``ok``: its parts passed and Lean accepted the composition), with the numbers a reader looks
    for first. A run that passed on its parts with no composition is not ``ok``."""
    complete = all(steps.get(x) for x in E2E_STEPS)
    neg = {r.stage[len("negative inner-"):]: r.ok for r in rows if r.step == "negative" and r.stage.startswith("negative inner-")}
    parts_ok = (complete and all(rc == "0" for v in steps.values() for rc in v) and not s.get("not_ok")
                and all(v is True for v in neg.values()))
    c = next((r for r in rows if r.step == "lean" and r.stage == "compose"), None)
    composed = c.extra.get("composed") if c else None
    mode = next(iter(k for k in s.get("pipeline", {}) if k.startswith("outer prove")), None)
    return {"complete": complete, "ok": parts_ok and c is not None and c.ok is True, "parts_ok": parts_ok, "composed": composed,
            "not_composed": None if composed is True else c.note if c else f"no record of the composition ({COMPOSITION})",
            "refused": c.note.removeprefix("refused: ") if c is not None and c.ok is False else None,
            "pipeline_sum_s": s.get("pipeline_sum_s"), "pipeline": s.get("pipeline"),
            "outer": s.get("outer"), "outer_mode": mode, "slowest": s.get("slowest"), "negative_refused": neg or None,
            "forged_refused": forged(rows) or None}


def step_status(root: Path) -> dict[str, list[str]]:
    """Each step's invocations' exit codes, as the job recorded them."""
    d = root / "steps"
    return {s.name: [text(r / "rc").strip() or "?" for r in sorted(s.glob("r*"))] for s in sorted(d.glob("*"))} if d.is_dir() else {}


def summary(rows: list[Row]) -> dict[str, Any]:
    """Totals over V*'s statements per mode (sums of each statement's median), the slowest repeated row, and the
    pipeline's sum of medians along one recursive run (the inner, V*'s staging, every outer proof, Lean's V* checks)."""
    def med(r: Row) -> float:
        """How long the row takes: a run's e2e (its reps' prove times can overlap), else the tool's wall."""
        s = r.e2e_s or r.wall_s
        return s.median if s else 0.0

    outer: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.step == "oprove" and r.wall_s and not r.extra.get("forged"):
            m = outer.setdefault(r.extra.get("mode", "dev"), {"prove_s": 0.0, "e2e_s": 0.0, "proof_bytes": 0, "statements": 0})
            m["prove_s"] += r.wall_s.median
            m["e2e_s"] += r.e2e_s.median if r.e2e_s else 0.0
            m["proof_bytes"] += r.proof_bytes or 0
            m["statements"] += 1
    timed = [r for r in rows if r.kind != "setup" and r.wall_s and not r.stage.endswith(" step")]
    slow = max(timed, key=med, default=None)
    whole = {r.step: med(r) for r in rows if r.stage == f"{r.step} step" and r.wall_s}
    inner = next((r for r in rows if r.stage == "inner M0"), None)
    # the mode that proved the most statements (ZK server on a tie): a mode measured on one statement is no run's path
    mode = max(outer, key=lambda m: (outer[m]["statements"], m == "firewall"), default=None)
    # one recursive run's path, each part as long as an auditor waits for it: the inner run, V*'s staging (the
    # vstage and vcap steps' whole walls: their statements are staged in parallel), V*'s statements proved one after
    # another, and Lean's checks (the lean step's whole wall, the inner run's and statement's checks included)
    parts = {"inner M0 e2e_s": inner.e2e_s.median if inner and inner.e2e_s else None,
             "V* staging_s (vstage + vcap steps)": (whole.get("vstage", 0) + whole.get("vcap", 0)) or None,
             f"outer prove ({mode}) e2e_s": outer[mode]["e2e_s"] if mode else None, "Lean step_s": whole.get("lean")}
    return {"outer": outer, "slowest": {"stage": slow.stage, "step": slow.step, "median_s": med(slow)} if slow else None,
            "pipeline": parts, "pipeline_sum_s": sum(v for v in parts.values() if v) if any(parts.values()) else None,
            "not_ok": [r.stage + (f" ({r.extra['mode']})" if r.step == "oprove" and r.extra.get("mode") else "")
                       for r in rows if r.ok is False]}


def merge(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """One profile from several runs' (``ref``: each one's run id or file): a row of a later one replaces an earlier one's
    with the same step, stage and mode, each row names the run it came from (``extra.run``), and the summary is the
    merged rows'."""
    by: dict[tuple[str, str, str], dict[str, Any]] = {}
    for d in docs:
        for r in d["rows"]:
            r = {**r, "extra": {**r.get("extra", {}), "run": d.get("ref") or d.get("run")}}
            by[(r["step"], r["stage"], r["extra"].get("mode", ""))] = r
    order = {s: i for i, s in enumerate(ORDER)}
    rows = sorted(by.values(), key=lambda r: order.get(r["step"], len(order)))
    objs = [Row.from_json(r) for r in rows]
    s = summary(objs)
    fw = [e | {"run": d.get("ref") or d.get("run")} for d in docs for e in d.get("summary", {}).get("flock_firewall", [])]
    if fw:
        s["flock_firewall"] = fw
    # a merged result is one recursive run only as far as its runs' steps are: each step's exits, from the last run with it
    last: dict[str, list[str]] = {}
    for d in docs:
        last |= d.get("steps", {})
    return {"schema": SCHEMA, "params": docs[-1].get("params", {}), "inputs": STATEMENTS,
            "runs": [d.get("ref") or d.get("run") for d in docs],
            "run_params": {str(d.get("ref") or d.get("run")): d.get("params", {}) for d in docs},
            "rows": rows, "summary": s, "headline": headline(objs, s, last) | {"merged": True},
            "steps": {f"{k} ({d.get('ref') or d.get('run')})": v for d in docs for k, v in d.get("steps", {}).items()}}


def loop_line(s: dict[str, Any]) -> str | None:
    """The table's line on the Lean session loop, from the run's process samples (``processes``); None where they hold
    none."""
    if not s.get("flock_firewall"):
        return None
    return "The Lean session loop (flock-firewall, from the run's process samples): " + "; ".join(
        f"{e['cpu_s']:.0f} CPU s, peak RSS {e['rss_hwm_gb']} GB, {e['threads']} threads, seen {e['first'][11:19]}–"
        f"{e['last'][11:19]}Z" + (f" ({e['run']})" if e.get("run") else "") for e in s["flock_firewall"]) + "."
