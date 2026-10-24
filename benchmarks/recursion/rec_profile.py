"""The recursion profile's result (``recursion-profile/v1``) from a job's ``out/`` (``job.sh``), and its table (README.md).

``read(OUT)`` walks ``OUT/steps/<step>/r<i>/out/``, the records 85-rec-reprice.sh wrote for each invocation, and returns one
row per timed thing: a prover's sessions (``inner``, ``oprove``: the timed sessions' ``LIVE`` lines), or a tool's
``/usr/bin/time -v`` over the repeats (``icheck``, ``vstage``, ``vcap``, ``oreplay``, ``lean``), or the build once. Every time is
a ``Stat`` (median, min, max and the values); nothing is averaged across different things. Stdlib only: the job runs it on
the node's ``python3``.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "recursion-profile/v1"
GPU_STEPS = ("inner", "oprove")
#: the prover's phase buckets (``flock-circuit``'s ``gpu_buckets``, both reps) a dominant phase is chosen among; the
#: aggregates (``t.arithmetic``, ``t.total``) and host-side ``h.*`` stay out
PHASES = ("t.witness", "t.encoding_commitment", "t.link", "t.zerocheck", "t.lincheck", "t.ring_switch", "t.ligerito",
          "t.firewall_recheck", "t.lean_finish", "t.zk_inner")


@dataclass(frozen=True)
class Stat:
    median: float
    min: float
    max: float
    values: tuple[float, ...]

    @classmethod
    def of(cls, xs: Iterable[float | None]) -> "Stat | None":
        v = tuple(float(x) for x in xs if x is not None)
        return cls(statistics.median(v), min(v), max(v), v) if v else None


@dataclass
class Row:
    stage: str
    step: str
    what: str
    kind: str                       # gpu (sessions) | cpu (repeats) | setup (once)
    wall_s: Stat | None = None      # gpu: prove_total_s per timed session; cpu and setup: the tool's wall
    e2e_s: Stat | None = None       # gpu: each timed session's wall, its waits on the live verifier included
    peak_rss_gb: float | None = None
    gpu_peak_mib: float | None = None
    proof_bytes: int | None = None
    wire_bytes: dict[str, int] | None = None
    dominant: dict[str, Any] | None = None
    ok: bool | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


def text(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


_WALL = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")
_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")


def wall_seconds(s: str) -> float:
    """``/usr/bin/time -v``'s elapsed time, ``h:mm:ss`` or ``m:ss.ss``, in seconds."""
    parts = [float(p) for p in s.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def timev(p: Path) -> dict[str, float] | None:
    """Wall seconds, peak RSS (GB) and CPU seconds of one ``/usr/bin/time -v`` file; None when it has no wall line."""
    t = text(p)
    m = _WALL.search(t)
    if not m:
        return None
    rss = _RSS.search(t)
    cpu = sum(float(x) for x in re.findall(r"(?:User|System) time \(seconds\): ([0-9.]+)", t))
    return {"wall_s": wall_seconds(m.group(1)), "peak_rss_gb": int(rss.group(1)) / 2**20 if rss else None, "cpu_s": cpu}


def live_lines(p: Path) -> list[tuple[float | None, dict[str, Any]]]:
    """The prover's ``LIVE`` records with the time each arrived: ``prove.tsv`` is 85's stamped stdout
    (``time<TAB>LIVE<TAB>{json}``, printed when the session's verdict is in); a plain stdout has no times."""
    out = []
    for ln in text(p).splitlines():
        t = None
        head, _, rest = ln.partition("\t")
        if rest.startswith("LIVE"):
            try:
                t, ln = float(head), rest
            except ValueError:
                pass
        if ln.startswith("LIVE"):
            try:
                out.append((t, json.loads(ln[5:])))
            except ValueError:
                pass
    return out


def tagged(p: Path, tag: str) -> list[dict[str, Any]]:
    """``TAG<TAB>{json}`` lines (serve's ``SESSION``, the replay's ``UPSTREAM``)."""
    out = []
    for ln in text(p).splitlines():
        if ln.startswith(tag + "\t"):
            try:
                out.append(json.loads(ln.split("\t", 1)[1]))
            except ValueError:
                pass
    return out


def lean_verdict(p: Path) -> dict[str, Any] | None:
    t = text(p)
    at = t.find("VERDICT ")
    if at < 0:
        return None
    try:
        return json.JSONDecoder(strict=False).raw_decode(t[at + 8:])[0]
    except ValueError:
        return None


def lean_split(p: Path) -> dict[str, float]:
    """flock-verify's stderr: ``inputs: N ms`` (from its start), ``setup: N ms`` (from the session's start), then the session's
    whole time (``/path: N ms``, setup included), as 85's summary reads them."""
    ms = [int(x) for x in re.findall(r"^(?:inputs|setup(?: \(drawn\))?|/\S+): (\d+) ms$", text(p), re.M)]
    return {"inputs_s": ms[0] / 1e3, "setup_s": ms[1] / 1e3, "session_s": (ms[2] - ms[1]) / 1e3} if len(ms) == 3 else {}


def gpu_peak(p: Path) -> float | None:
    xs = [int(x) for x in text(p).split() if x.isdigit()]
    return float(max(xs)) if xs else None


def _bucket_sums(buckets: list[dict[str, Any]] | None) -> dict[str, float]:
    sums: dict[str, float] = {}
    for b in buckets or []:
        for k, v in b.items():
            if k.startswith("t.") and isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
    return sums


def dominant(v: dict[str, Any]) -> dict[str, Any] | None:
    """The largest part of one session's ``e2e_s``: a prover phase (both reps' bucket summed), the host witness
    (``witness_s``), or what is left, ``wait.verifier`` (the session waiting on its verifier's coins and verdict, and on
    whatever else the prover's buckets don't time); its share of ``e2e_s`` (of ``prove_total_s`` without one)."""
    parts = {k: x for k, x in _bucket_sums(v.get("buckets")).items() if k in PHASES}
    if not parts:
        return None
    total, e2e = v.get("prove_total_s"), v.get("e2e_s")
    if e2e and total is not None:
        parts["h.witness"] = v.get("witness_s") or 0.0
        parts["wait.verifier"] = max(0.0, e2e - total - parts["h.witness"])
    k = max(parts, key=parts.get)
    whole = e2e or total
    return {"phase": k, "s": round(parts[k], 4), "share": round(parts[k] / whole, 3) if whole else None,
            "of": "e2e_s" if e2e else "prove_total_s",
            "phases": {p: round(x, 4) for p, x in sorted(parts.items(), key=lambda kv: -kv[1])}}


def _num(x: Any) -> float | None:
    """A per-session number; a per-rep list (``prove_s``) summed."""
    if isinstance(x, list):
        return sum(float(y) for y in x) if x else None
    return x


def prover_row(d: Path, stage: str, step: str, what: str) -> Row:
    """One prove_live directory: the timed sessions (warm ones left out), the invocation's peak RSS, the GPU's peak."""
    stamped = live_lines(d / "prove.tsv") or live_lines(d / "prove.txt")
    rows = [v for _, v in stamped]
    timed = [v for v in rows if not v.get("warm")]
    r = Row(stage, step, what, "gpu")
    # 85's `again`: "<fn> <name>: lease preempted, try <t>" in the invocation's out/preempted.txt
    preempted = sum(1 for ln in text(d.parent.parent / "preempted.txt").splitlines() if f" {d.name}: lease preempted" in ln)
    if not timed:
        why = (f"lease preempted {preempted}x, no session finished in any try; " if preempted else "no timed session: ")
        r.ok, r.note = False, (why + text(d / "prove.err")[-300:].strip().replace("\n", " ")) if d.exists() else "not run"
        r.extra = {"preempted": preempted}
        return r
    r.wall_s = Stat.of(_num(v.get("prove_total_s", v.get("prove_s"))) for v in timed)
    r.e2e_s = Stat.of(v.get("e2e_s") for v in timed)
    tv = timev(d / "time-prove.txt")
    r.peak_rss_gb = tv["peak_rss_gb"] if tv else None
    r.gpu_peak_mib = gpu_peak(d / "gpu-mib.txt")
    med = sorted(timed, key=lambda v: _num(v.get("prove_total_s", v.get("prove_s"))) or 0)[len(timed) // 2]
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
    # 85's session time: from the last warm session's verdict to the last timed one's, over the timed sessions (sessions
    # overlap up to FC_VERIFY_AHEAD, so this, not e2e_s, is the stage's pace)
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


def tool_rows(outs: list[Path], step: str, rel: dict[str, tuple[str, str]], verdict=None) -> list[Row]:
    """One row per tool: its ``/usr/bin/time -v`` file (``rel``: name -> (path under out/, what)) over every repeat; none
    for a step that never ran."""
    rows: list[Row] = []
    if not outs:
        return rows
    for name, (path, what) in rel.items():
        ts = [timev(o / path) for o in outs]
        got = [t for t in ts if t]
        r = Row(name, step, what, "cpu", wall_s=Stat.of(t["wall_s"] for t in got),
                peak_rss_gb=max((t["peak_rss_gb"] for t in got if t["peak_rss_gb"] is not None), default=None))
        if not got:
            r.note = "not run"
        if verdict:
            vs = [v for o in outs if (o / path).exists() if (v := verdict(o, name)) is not None]
            r.ok = all(v is True for v in vs) if vs else None
        rows.append(r)
    return rows


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
        r = Row("build", "build", "85 STEP=build: the CUDA prover, flock-verify and both stagings, as the job found its caches",
                "setup", wall_s=Stat.of([t["wall_s"]]) if t else None, peak_rss_gb=t["peak_rss_gb"] if t else None,
                ok=text(o.parent / "rc").strip() == "0")
        r.extra = {"binary": text(o / "binary.sha256").split()[:1], "lean": text(o / "lean.sha").split()[:1],
                   "rebuilt_prover": (o / "build.log").exists()}
        for s in ("stage", "stage-zk"):
            st = timev(o / s / "time-stage.txt")
            r.extra[s] = {"wall_s": st and st["wall_s"], "peak_rss_gb": st and st["peak_rss_gb"],
                          "sizes": text(o / s / "sizes.txt").strip()}
        rows.append(r)
    for o in repeats(root, "inner")[:1]:
        for sub, stage, what in (("proxy", "inner M0", "the inner statement (plain leaves), ZK off, against the rec_live proxy"),
                                 ("m0", "baseline M0", "today's hm96 statement, ZK off, loopback serve"),
                                 ("zk", "baseline --zk", "today's hm96 statement with --zk, loopback serve")):
            if (o / "inner" / sub).exists():
                rows.append(prover_row(o / "inner" / sub, stage, "inner", what))
    rows += tool_rows(repeats(root, "icheck"), "icheck", {
        "inner replay": ("icheck/time-replay-proxy.txt", "flock-circuit replay of the last inner session (upstream's verifier)"),
        "rec_vstar": ("icheck/time-vstar.txt", "V*'s hashing reference over the inner session")},
        verdict=lambda o, n: _upstream(o, "icheck/replay-proxy.txt") if n == "inner replay" else None)
    vs = repeats(root, "vstage")
    if vs:
        rel = {"V* points": ("vstage/time-points.txt", "the verifier's fold points from the inner coins (rec_vstage points)"),
               "inner fold": ("vstage/time-fold.txt", "InnerFold.lean as an executable: extra claims and comb")}
        for n in _names(vs, "vstage", r"time-(.+)\.txt"):
            if n not in ("points", "fold", "cap"):
                rel[f"stage {n}"] = (f"vstage/time-{n}.txt", "rec_vstage stage of the verifier's own values alone (no level, no "
                                     "algebra)" if n == "verifier" else f"rec_vstage stage of V*'s statement {n}")
        rows += tool_rows(vs, "vstage", rel)
    rows += tool_rows(repeats(root, "vcap"), "vcap", {"stage cap": ("vstage/time-cap.txt", "RecCap at every level")})
    for o in repeats(root, "oprove")[:1]:
        for d in sorted((p for p in (o / "oprove").glob("*") if p.is_dir()), key=lambda p: p.name):
            mode = next((m for m in ("firewall", "host", "dev") if d.name.endswith("-" + m)), "dev")
            stmt = d.name[: -len(mode) - 1] if d.name.endswith("-" + mode) else d.name
            rows.append(prover_row(d, f"outer {stmt}", "oprove", f"V*'s {stmt} with --zk, mode {mode}"))
            rows[-1].extra |= {"mode": mode, "statement": stmt}
    orr = repeats(root, "oreplay")
    if orr:
        rows += tool_rows(orr, "oreplay", {f"replay {n}": (f"oreplay/{n}/time-replay.txt", f"flock-circuit replay --zk of {n}")
                                           for n in _names(orr, "oreplay", r"([^.]+)")},
                          verdict=lambda o, n: _upstream(o, f"oreplay/{n[len('replay '):]}/replay.txt"))
    ln = repeats(root, "lean")
    if ln:
        names = [n for n in _names(ln, "lean", r"time-(.+)\.txt")]
        rel = {f"lean {n}": (f"lean/time-{n}.txt", ("flock-verify statement on the inner statement" if n == "inner-statement"
                                                    else "flock-verify verify on the inner session (--coins os)" if n == "inner"
                                                    else f"flock-verify verify --zk on V*'s {n} session")) for n in names}

        def lv(o: Path, n: str) -> bool | None:
            name = n[len("lean "):]
            if name == "inner-statement":
                return None
            v = lean_verdict(o / "lean" / f"{name}.log")
            return v.get("accepted") is True if v else False

        lrows = tool_rows(ln, "lean", rel, verdict=lv)
        for r in lrows:
            name = r.stage[len("lean "):]
            splits = [lean_split(o / "lean" / f"{name}.err") for o in ln]
            for k in ("inputs_s", "setup_s", "session_s"):
                s = Stat.of(x.get(k) for x in splits)
                if s:
                    r.extra[k] = asdict(s)
            why = [str((lean_verdict(o / "lean" / f"{name}.log") or {}).get("why"))[:200] for o in ln]
            if r.ok is False:
                r.note = next((w for w in why if w and w != "None"), text(ln[0] / "lean" / f"{name}.log")[-200:].strip())
        steps = [timev(o.parent / "time-step.txt") for o in ln]
        rows.append(Row("lean step", "lean", "85 STEP=lean whole (JOBS sessions at once), as an auditor waits", "cpu",
                        wall_s=Stat.of(t["wall_s"] for t in steps if t)))
        rows += lrows
    return {"schema": SCHEMA, "params": params(root), "rows": [r.to_json() for r in rows], "summary": summary(rows),
            "steps": step_status(root)}


def params(root: Path) -> dict[str, str]:
    out = {}
    for tok in text(root / "params.txt").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def step_status(root: Path) -> dict[str, list[str]]:
    """Each step's invocations' exit codes, as the job recorded them."""
    d = root / "steps"
    return {s.name: [text(r / "rc").strip() or "?" for r in sorted(s.glob("r*"))] for s in sorted(d.glob("*"))} if d.is_dir() else {}


def summary(rows: list[Row]) -> dict[str, Any]:
    """Totals over V*'s statements per mode (sums of each statement's median), the slowest repeated stage, and the
    pipeline's sum of medians along one recursive session (the inner, V*'s staging, every outer proof, Lean's V* checks)."""
    def med(r: Row) -> float:
        return r.wall_s.median if r.wall_s else 0.0

    outer: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.step == "oprove" and r.wall_s:
            m = outer.setdefault(r.extra.get("mode", "dev"), {"prove_s": 0.0, "e2e_s": 0.0, "proof_bytes": 0, "statements": 0})
            m["prove_s"] += r.wall_s.median
            m["e2e_s"] += r.e2e_s.median if r.e2e_s else 0.0
            m["proof_bytes"] += r.proof_bytes or 0
            m["statements"] += 1
    timed = [r for r in rows if r.kind != "setup" and r.wall_s and r.stage != "lean step"]
    slow = max(timed, key=med, default=None)
    lean_v = sum(med(r) for r in rows if r.step == "lean" and r.stage.startswith("lean ")
                 and r.stage not in ("lean step", "lean inner", "lean inner-statement"))
    inner = next((r for r in rows if r.stage == "inner M0"), None)
    stage_v = sum(med(r) for r in rows if r.step in ("vstage", "vcap"))
    mode = "firewall" if "firewall" in outer else next(iter(outer), None)
    parts = {"inner M0 e2e_s": inner.e2e_s.median if inner and inner.e2e_s else None, "V* staging_s": stage_v or None,
             f"outer prove ({mode}) e2e_s": outer[mode]["e2e_s"] if mode else None, "Lean V* verify_s (sum)": lean_v or None}
    return {"outer": outer, "slowest": {"stage": slow.stage, "step": slow.step, "median_s": med(slow)} if slow else None,
            "pipeline": parts, "pipeline_sum_s": sum(v for v in parts.values() if v) if any(parts.values()) else None,
            "not_ok": [r.stage for r in rows if r.ok is False]}


def _fmt(s: dict[str, Any] | None) -> str:
    if not s:
        return "—"
    if len(s["values"]) == 1:
        return f"{s['median']:.3g}"
    return f"{s['median']:.3g} [{s['min']:.3g}–{s['max']:.3g}] (n={len(s['values'])})"


def _bytes(n: int | None) -> str:
    if not n:
        return "—"
    for unit, k in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if n >= k:
            return f"{n / k:.3g} {unit}"
    return f"{n} B"


def markdown(doc: dict[str, Any]) -> str:
    p = doc.get("params", {})
    lines = [f"Recursion profile: K = {p.get('K')}, N = {p.get('N')}, commit {str(p.get('commit'))[:12]}, "
             f"{p.get('RUNS')} timed sessions after {p.get('WARM')} warm per GPU step, {p.get('REPEAT')} repeats per CPU step. "
             "Seconds: median [min–max] (n).", "",
             "| stage | step | what | time s | session e2e s | pace s | peak RSS GB | GPU peak MiB | proof | dominant phase | ok |",
             "|---|---|---|---|---|---:|---:|---:|---:|---|---|"]
    for r in doc["rows"]:
        dom = r.get("dominant")
        dom_s = (f"{dom['phase']} {dom['s']:.3g} s" + (f" ({dom['share']:.0%} of {dom.get('of', 'prove_total_s')})"
                                                       if dom.get("share") else "")) if dom else "—"
        pace = r.get("extra", {}).get("session_s")
        pace_s = f"{pace:.3g}" if isinstance(pace, (int, float)) else "—"
        ok = {True: "yes", False: "**no**", None: "—"}[r.get("ok")]
        note = f" {r['note'][:120]}".replace("|", "/") if r.get("note") else ""
        rss = f"{r['peak_rss_gb']:.3g}" if r.get("peak_rss_gb") is not None else "—"
        gpu = f"{r['gpu_peak_mib']:.0f}" if r.get("gpu_peak_mib") else "—"
        lines.append(f"| {r['stage']} | {r['step']} | {r['what']} | {_fmt(r.get('wall_s'))} | {_fmt(r.get('e2e_s'))} | {pace_s} | {rss} | "
                     f"{gpu} | {_bytes(r.get('proof_bytes'))} | {dom_s} | {ok}{note} |")
    s = doc["summary"]
    lines += ["", "Outer (V*'s statements, sum of each statement's median): " + "; ".join(
        f"{m}: prove {v['prove_s']:.3g} s, session e2e {v['e2e_s']:.3g} s, proofs {_bytes(int(v['proof_bytes']))} over "
        f"{v['statements']} statements" for m, v in s["outer"].items()) if s["outer"] else "Outer: not run."]
    if s.get("slowest"):
        lines.append(f"Slowest repeated stage: {s['slowest']['stage']} ({s['slowest']['step']}), {s['slowest']['median_s']:.3g} s.")
    if s.get("pipeline_sum_s"):
        lines.append("One recursive session, sum of medians along its path: " + ", ".join(
            f"{k} {v:.3g}" for k, v in s["pipeline"].items() if v) + f"; total {s['pipeline_sum_s']:.3g} s.")
    if s.get("not_ok"):
        lines.append("Not accepted or not run: " + ", ".join(s["not_ok"]) + ".")
    lines.append("Steps' exit codes: " + ", ".join(f"{k} {'/'.join(v)}" for k, v in doc.get("steps", {}).items()) + ".")
    return "\n".join(lines) + "\n"
