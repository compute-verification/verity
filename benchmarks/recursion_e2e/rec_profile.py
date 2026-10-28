"""The recursion bench's result (``recursion-e2e/v1``) from a job's ``out/`` (``job.sh``), and its table (README.md).

``read(OUT)`` walks ``OUT/steps/<step>/r<i>/out/``, the records 85-rec-reprice.sh wrote for each invocation, and returns one
row per timed thing: a prover's runs (``inner``, ``oprove``: the timed runs' ``LIVE`` lines), or a tool's
``/usr/bin/time -v`` over the repeats (``icheck``, ``vstage``, ``vcap``, ``oreplay``, ``lean``), or the build once, and one row
per flipped inner coin (``negative``). Every time is a ``Stat`` (median, min, max and the values); nothing is averaged across
different things. ``headline`` says whether the run is a whole recursive run (every step ran) and was accepted (every
step exited 0, every honest verdict accepted, every flipped coin refused). Stdlib only: the job runs it on the node's
``python3``.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "recursion-e2e/v1"
#: the steps one recursive run needs, in order (vcap stages RecCap, which V*'s verdict needs; ``negative`` checks the
#: run, ``icheck`` and ``oreplay`` time more of it)
E2E_STEPS = ("build", "inner", "vstage", "vcap", "oprove", "lean")
ORDER = ("build", "inner", "icheck", "vstage", "vcap", "oprove", "oreplay", "lean", "forge-after", "negative")
#: what each part of the recursive run is, recorded in the result's ``inputs``
STATEMENTS = {
    "inner": "N x GemmCoordinate_v2{K,DOT=HopperBF16WgmmaDot16_v1} under Q_template_instance (rec_reprice.py), "
             "verity/flock-circuit+plain-leaves, M0 (no --zk), FC_COINS=os, through rec_live's proxy",
    "vstar": "V*'s statements over the inner's last proxied run: each Ligerito level's RecOpen and RecCap, the algebra's "
             "InnerRepCheck parts (rec_vstage)",
    "outer": "each V* statement with --zk on loopback, every unit drawn, one at a time, in each of MODES",
    "verify": "flock-verify verify --zk --rec-inner-session --rec-inner-transcript on each V* release (the last mode's), "
              "and flock-verify statement on the inner statement",
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
    kind: str                       # gpu (runs) | cpu (repeats) | setup (once)
    wall_s: Stat | None = None      # gpu: prove_total_s per timed run; cpu and setup: the tool's wall
    e2e_s: Stat | None = None       # gpu: each timed run's wall, its waits on the live verifier included
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

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Row":
        st = {k: Stat(d[k]["median"], d[k]["min"], d[k]["max"], tuple(d[k]["values"])) if d.get(k) else None
              for k in ("wall_s", "e2e_s")}
        return cls(**{**{k: v for k, v in d.items() if k in cls.__dataclass_fields__}, **st})


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
    (``time<TAB>LIVE<TAB>{json}``, printed when the run's verdict is in); a plain stdout has no times."""
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
    """flock-verify's stderr: ``inputs: N ms`` (from its start), ``setup: N ms`` (from the verification's start), then the verification's
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
    k = max(parts, key=parts.get)
    whole = e2e or total
    return {"phase": k, "s": round(parts[k], 4), "share": round(parts[k] / whole, 3) if whole else None,
            "of": "e2e_s" if e2e else "prove_total_s",
            "phases": {p: round(x, 4) for p, x in sorted(parts.items(), key=lambda kv: -kv[1])}}


def _num(x: Any) -> float | None:
    """A per-run number; a per-rep list (``prove_s``) summed."""
    if isinstance(x, list):
        return sum(float(y) for y in x) if x else None
    return x


def prover_row(d: Path, stage: str, step: str, what: str) -> Row:
    """One prove_live directory: the timed runs (warm ones left out), the invocation's peak RSS, the GPU's peak."""
    stamped = live_lines(d / "prove.tsv") or live_lines(d / "prove.txt")
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
            # only repeats whose tool finished (``time -v`` opens its file at the start and writes it at the end)
            vs = [v for o in outs if timev(o / path) if (v := verdict(o, name)) is not None]
            r.ok = all(v is True for v in vs) if vs else None
        rows.append(r)
    return rows


def step_row(outs: list[Path], step: str, what: str) -> Row:
    """A step's whole wall over its repeats (job.sh's ``time-step.txt`` around each invocation of 85)."""
    ts = [t for t in (timev(o.parent / "time-step.txt") for o in outs) if t]
    return Row(f"{step} step", step, what, "cpu", wall_s=Stat.of(t["wall_s"] for t in ts),
               peak_rss_gb=max((t["peak_rss_gb"] for t in ts if t["peak_rss_gb"] is not None), default=None))


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
        rel = {f"lean {n}": (f"lean/time-{n}.txt", ("flock-verify statement on the inner statement" if n == "inner-statement"
                                                    else "flock-verify verify on the inner run (--coins os)" if n == "inner"
                                                    else f"flock-verify verify --zk on V*'s {n} release")) for n in names}

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
                s = Stat.of(x.get(k) for x in splits)
                if s:
                    r.extra[k] = asdict(s)
                    parts[k] = s.median
            # flock-verify's own split: its inputs read, the verification's setup, and its checks after it
            if parts and r.wall_s:
                parts = {"lean.inputs": parts.get("inputs_s", 0.0), "lean.setup": parts.get("setup_s", 0.0),
                         "lean.checks": max(0.0, parts.get("session_s", 0.0))}
                k = max(parts, key=parts.get)
                r.dominant = {"phase": k, "s": round(parts[k], 4), "share": round(parts[k] / r.wall_s.median, 3), "of": "wall_s",
                              "phases": {p: round(x, 4) for p, x in sorted(parts.items(), key=lambda kv: -kv[1])}}
            why = [str((lean_verdict(o / "lean" / f"{name}.log") or {}).get("why"))[:200] for o in ln]
            if r.ok is False or r.extra.get("refused"):
                r.note = next((w for w in why if w and w != "None"), text(ln[0] / "lean" / f"{name}.log")[-200:].strip())
        rows.append(step_row(ln, "lean", "85 STEP=lean whole (JOBS verifications at once), as an auditor waits"))
        rows += lrows
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
        + ", verity/flock-circuit+plain-leaves, M0 (no --zk), FC_COINS=os, through rec_live's proxy")}
    return {"schema": SCHEMA, "params": params(root), "inputs": inputs, "rows": [r.to_json() for r in rows], "summary": s,
            "steps": steps, "headline": headline(rows, s, steps), **({"staged": staged} if staged else {})}


def staged_units(p: Path) -> dict[str, dict[str, Any]]:
    """What `rec_vstage.stage` printed of each unit it staged (85's ``summary-vstage.json``): its template, instances, rows,
    k_log, seconds and (the algebra's parts) its lowering's seconds and circuit bytes, by statement name."""
    try:
        st = json.loads(text(p) or "{}").get("statements") or {}
    except ValueError:
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


def exit_status(p: Path) -> int | None:
    m = re.search(r"^\s*Exit status: (\d+)\s*$", text(p), re.M)
    return int(m.group(1)) if m else None


def negative_rows(d: Path) -> list[Row]:
    """One row per flipped coin (85's ``forged-inner-COIN-STATEMENT`` parts, read as ``lean_verdicts.py`` reads them): a
    statement refuses it when flock-verify exited 1 with a VERDICT and none accepting; the coin is refused when one statement
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


def refused(r: Row) -> None:
    """A forged statement's outer row: refused when no run of it was accepted (the firewall stopped it before release, or
    its releases were refused), the stop or the refusal as its note; `forgeries` judges an accepted one."""
    r.extra["forged"] = True
    r.extra["refused"] = r.ok is not True
    r.ok = r.extra["refused"]


def kind_of(part: str, honest: list[str]) -> str:
    """The forgery ``forged-KIND-PART`` is KIND, PART the longest honest part it ends with; else the name without ``forged-``
    (lean_verdicts.py's rule)."""
    name = part.removeprefix("forged-")
    ends = [h for h in honest if name.endswith("-" + h)]
    return name[: -len(max(ends, key=len)) - 1] if ends else name


def forgeries(rows: list[Row]) -> None:
    """85's forged statements, judged per forgery as lean_verdicts.py judges them: V* accepts only when every part does, so a
    forgery is refused when one of its parts is, and a part it leaves as the honest one (forged-message's alg-p1) is accepted.
    An accepted part's row is ok when another part of its forgery, at the same step, was refused."""
    groups: dict[tuple[str, str], list[Row]] = {}
    for step in ("oprove", "lean"):
        named = [(r, r.stage.split(" ", 1)[1]) for r in rows if r.step == step and r.stage.startswith(("outer ", "lean "))]
        honest = [n for r, n in named if not r.extra.get("forged")]
        for r, n in named:
            if "refused" in r.extra:
                r.extra["kind"] = kind_of(n, honest)
                groups.setdefault((step, r.extra["kind"]), []).append(r)
    for rs in groups.values():
        held = any(r.extra["refused"] for r in rs)
        for r in rs:
            if not r.extra["refused"]:
                r.ok = held
                r.note = ("accepted: a part this forgery leaves as the honest one; " if held
                          else "ACCEPTED: no part of this forgery was refused; ") + r.note


def forge_after_rows(p: Path) -> list[Row]:
    """One row per forge_after.py case (``<bin>/<kind>-<case>``): ok when a forged case was refused (exit 1 and no accepting
    VERDICT) or the honest control accepted, Lean's why as its note; none when the file is missing."""
    try:
        doc = json.loads(text(p) or "null")
    except ValueError:
        doc = None
    if not doc:
        return []
    rows = []
    for name, v in doc.get("verify", {}).items():
        kind, case = name.split("/")[-1].rsplit("-", 1)
        honest = case == "honest"
        r = Row(f"forge-after {name}", "forge-after",
                f"{kind}'s honest control" if honest else f"{kind} forged after proving ({case})", "cpu")
        accepted = v.get("rc") == 0 or bool(v.get("accepted"))
        r.ok = accepted if honest else not accepted
        r.note = ("ACCEPTED: " if accepted and not honest else "") + str(v.get("why") or "")[:200]
        r.extra = {"rc": v.get("rc"), "forged": not honest}
        rows.append(r)
    if not doc.get("untouched", True):
        r = Row("forge-after inputs", "forge-after", "the honest releases and statements, hashed before and after", "cpu")
        r.ok, r.note = False, "an input changed under forge_after.py"
        rows.append(r)
    return rows


def headline(rows: list[Row], s: dict[str, Any], steps: dict[str, list[str]]) -> dict[str, Any]:
    """Whether the run is one whole recursive run (every step of ``E2E_STEPS`` ran) and accepted (every invocation exited
    0, no row refused or unrun, and every flipped coin refused), with the numbers a reader looks for first."""
    complete = all(steps.get(x) for x in E2E_STEPS)
    neg = {r.stage[len("negative inner-"):]: r.ok for r in rows if r.step == "negative" and r.stage.startswith("negative inner-")}
    ok = (complete and all(rc == "0" for v in steps.values() for rc in v) and not s.get("not_ok")
          and all(v is True for v in neg.values()))
    mode = next(iter(k for k in s.get("pipeline", {}) if k.startswith("outer prove")), None)
    return {"complete": complete, "ok": ok, "pipeline_sum_s": s.get("pipeline_sum_s"), "pipeline": s.get("pipeline"),
            "outer": s.get("outer"), "outer_mode": mode, "slowest": s.get("slowest"), "negative_refused": neg or None,
            "forged_refused": forged(rows) or None}


def forged(rows: list[Row]) -> dict[str, bool]:
    """Each forgery kind's verdict: 85's forged statements of it refused at each step (`forgeries`), forge-after's cases of
    it every one refused, and forge-after's honest controls of it accepted."""
    out: dict[str, bool] = {}
    for r in rows:
        if r.ok is None:
            continue
        m = re.fullmatch(r"forge-after [^/]+/([a-z]+)-[a-z]+", r.stage) if r.step == "forge-after" else None
        kind = m.group(1) if m else r.extra.get("kind") if r.step in ("oprove", "lean") else None
        if kind:
            out[kind] = out.get(kind, True) and r.ok
    return out


def processes(p: Path, comm: str) -> list[dict[str, Any]]:
    """Each ``comm`` process the run's sampler saw (``research run``'s ``resources.jsonl`` beside ``out/``): its CPU seconds and
    peak RSS at its last sample, and when it was first and last seen; the run's own telemetry, not 85's records."""
    seen: dict[int, dict[str, Any]] = {}
    for ln in text(p).splitlines():
        if '"kind":"process"' not in ln or comm not in ln:
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        for q in d.get("processes", []):
            if q.get("comm") == comm:
                e = seen.setdefault(q["pid"], {"pid": q["pid"], "first": d["t_utc"]})
                e |= {"last": d["t_utc"], "cpu_s": q.get("cpu_s"), "rss_hwm_gb": round((q.get("rss_hwm_bytes") or 0) / 2**30, 2),
                      "threads": q.get("threads")}
    return [e for e in seen.values() if (e.get("cpu_s") or 0) >= 1]


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
    # the mode that proved the most statements (firewall on a tie): a mode measured on one statement is no run's path
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

    def each(q: dict[str, Any]) -> str:
        return (f"commit {str(q.get('commit'))[:12]}, the inner's {q.get('RUNS')} timed runs after {q.get('WARM')} warm, "
                f"each V* statement's {q.get('OPROVE_RUNS', q.get('RUNS'))} after {q.get('OPROVE_WARM', q.get('WARM'))} "
                f"(lease {q.get('GPU_MAX_MIN', '?')} min)"
                + (f", V* statements {q['OPROVE_JOBS']} at a time" if q.get("OPROVE_JOBS") else "")
                + f", {q.get('REPEAT')} repeats per CPU step")
    runs = doc.get("run_params")
    h = doc.get("headline") or {}
    verdict = ("one whole recursive run, accepted" if h.get("ok") else
               "one whole recursive run, **not accepted** (see ok and the steps' exits)" if h.get("complete") else
               "**not a whole recursive run**: some of " + ", ".join(E2E_STEPS) + " did not run")
    st = doc.get("staged")
    shape = (f"{st['subcircuit']} x {st['n']} (m = {st['m']}, {'served window' if st.get('window') else 'generated values'})"
             if st else f"K = {p.get('K')}, N = {p.get('N')}")
    lines = [f"Recursion bench: {shape}, oprove modes {p.get('modes', '?')}, "
             + ("; ".join(f"{k}: {each(q)}" for k, q in runs.items()) if runs else each(p))
             + f". Seconds: median [min–max] (n). Verdict: {verdict}"
             + (" (merged runs)" if h.get("merged") else "") + ".", "",
             "| stage | step | what | time s | run e2e s | pace s | peak RSS GB | GPU peak MiB | proof | dominant phase | ok |"
             + (" run |" if doc.get("runs") else ""),
             "|---|---|---|---|---|---:|---:|---:|---:|---|---|" + ("---|" if doc.get("runs") else "")]
    for r in doc["rows"]:
        run = f" {r.get('extra', {}).get('run')} |" if doc.get("runs") else ""
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
                     f"{gpu} | {_bytes(r.get('proof_bytes'))} | {dom_s} | {ok}{note} |{run}")
    s = doc["summary"]
    lines += ["", "Outer (V*'s statements, sum of each statement's median): " + "; ".join(
        f"{m}: prove {v['prove_s']:.3g} s, run e2e {v['e2e_s']:.3g} s, proofs {_bytes(int(v['proof_bytes']))} over "
        f"{v['statements']} statements" for m, v in s["outer"].items()) if s["outer"] else "Outer: not run."]
    if s.get("slowest"):
        lines.append(f"Slowest row (a run's e2e, else a tool's wall): {s['slowest']['stage']} ({s['slowest']['step']}), "
                     f"{s['slowest']['median_s']:.3g} s.")
    if s.get("pipeline_sum_s"):
        lines.append("One recursive run, sum of medians along its path: " + ", ".join(
            f"{k} {v:.3g}" for k, v in s["pipeline"].items() if v) + f"; total {s['pipeline_sum_s']:.3g} s.")
    if s.get("flock_firewall"):
        lines.append("The Lean session loop (flock-firewall, from the run's process samples): " + "; ".join(
            f"{e['cpu_s']:.0f} CPU s, peak RSS {e['rss_hwm_gb']} GB, {e['threads']} threads, seen {e['first'][11:19]}–"
            f"{e['last'][11:19]}Z" + (f" ({e['run']})" if e.get("run") else "") for e in s["flock_firewall"]) + ".")
    if h.get("forged_refused"):
        lines.append("Forgeries (85's forged statements and forge-after's cases), each refused: " + ", ".join(
            f"{k} {'yes' if v else '**no**'}" for k, v in h["forged_refused"].items()) + ".")
    if s.get("not_ok"):
        lines.append("Not accepted or not run: " + ", ".join(s["not_ok"]) + ".")
    lines.append("Steps' exit codes: " + ", ".join(f"{k} {'/'.join(v)}" for k, v in doc.get("steps", {}).items()) + ".")
    return "\n".join(lines) + "\n"
