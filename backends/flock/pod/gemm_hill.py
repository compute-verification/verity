"""BF16 GemmCoordinate hillclimb: one K's proving overhead against its GPU's spec peak, its throughput and its GPU utilization.

An attempt (74-gemm-hill.sh) gates the prover's byte identity at B = 16 (the GPU's proofs and transcripts equal the CPU prover's,
the device paths agree, and sessions verified beside the next ones' proving equal those verified one at a time), then proves one m = 35 statement of `GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}` (the BF16
dot of 16 that sm_90's wgmma and sm_120's mma.m16n8k16 run) untiled, warm once and timed three times, with nvidia-smi sampling
the GPU every second from the job's start. Per coordinate (one FP32 output word of a K-long dot product):

- overhead = T_prove / T_native. T_prove is M0's (gemm_slowdown.statement_seconds): the median timed statement's witness wait
  plus both reps, over its coordinates; the loopback verifier is excluded. T_native = 2 K FLOP / P_peak, P_peak the census line's
  dense BF16 tensor-core peak at FP32 accumulate (--peak-id).
- throughput: coordinates per second over the timed sessions back to back, each verifier's verdict included (the window ends at
  the last one; with FC_VERIFY_AHEAD the prover proves the next sessions while earlier ones are verified), with 1 / T_prove
  beside it.
- GPU utilization: nvidia-smi dmon's sm% (the fraction of each second some kernel ran, so an upper bound on DCGM's SM-active)
  averaged over the timed sessions (steady) and over the job's whole GPU-held wall time (the seconds before the sampler started
  count as idle); and the timed sessions' GPU-held seconds per coordinate.
- the verifier's seconds per statement: its verdict's verify_s, the median of the timed sessions.

    python gemm_hill.py defs --K 8192 --out defs.json [--n 1024]      # the Gemm_v2 row whose coordinates are the statement's lanes
    python gemm_hill.py dmon --out dmon.tsv [--gpu 0]                  # nvidia-smi dmon -s u every second, each line after its epoch
    python gemm_hill.py stamp --out stamps.tsv -- BINARY prove ...    # the prover, its stdout passed through and each line stamped
    python gemm_hill.py metrics --K 8192 --gate gate.jsonl --sweep results.jsonl --stamps stamps.tsv --dmon dmon.tsv \
        --phases phases.tsv --census census/hardware.json --out metrics.json --result result.json
    python gemm_hill.py plot RUN... --out DIR [--steps NAME...] [--put]  # per K, the three metrics over the steps, from the store
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemm_slowdown as GS  # noqa: E402

DOT = "HopperBF16WgmmaDot16_v1"
DEFINITION = re.compile(r"GemmCoordinate_v2\{K=(\d+),DOT=" + DOT + r"\}")
M = 35
#: instances per statement at m = 35: 2^(35 - k_log), k_log the block circuit.compose packs one coordinate into (the unit's row
#: slot and its rows' hashing): 24, 24, 25, 26. K = 16384 is at Stmt.InRange's k_log <= 26
INSTANCES = {2048: 2048, 4096: 2048, 8192: 1024, 16384: 512}
PEAK_ID = "rtx-pro-6000-blackwell-server/bf16"
QUESTION = ("what is BF16 GemmCoordinate_v2{K} proving overhead against the RTX PRO 6000 spec peak, its throughput and its GPU "
            "utilization, at this step?")


def defs(a) -> None:
    n = a.n or INSTANCES.get(a.K)
    if not n:
        raise SystemExit(f"K = {a.K}: no instance count for m = {M} (pass --n)")
    Path(a.out).write_text(json.dumps({f'Gemm_v2{{K={a.K},N={n},DOT={{"fn":"{DOT}"}}}}': 1}) + "\n")
    print(n)


def dmon(a) -> None:
    """nvidia-smi dmon -s u at 1 s until SIGTERM, each line prefixed with the epoch seconds it arrived."""
    p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1"] + (["-i", a.gpu] if a.gpu else []), stdout=subprocess.PIPE, text=True)

    def stop(*_):
        p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    with open(a.out, "a") as out:
        for ln in p.stdout:
            out.write(f"{time.time():.3f}\t{ln}")
            out.flush()


def stamp(a) -> None:
    """Run the command, its stdout passed through line by line; every line also goes to --out after its epoch seconds."""
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    with open(a.out, "a") as out:
        out.write(f"{time.time():.3f}\tSTART\t{json.dumps(cmd[1:2])}\n")
        out.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for ln in p.stdout:
            sys.stdout.buffer.write(ln)
            sys.stdout.buffer.flush()
            out.write(f"{time.time():.3f}\t{ln.decode(errors='replace').rstrip()}\n")
            out.flush()
        rc = p.wait()
        out.write(f"{time.time():.3f}\tEXIT\t{rc}\n")
    sys.exit(rc)


def _records(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.startswith("{")]


def _prove_process(path: Path) -> tuple[float | None, float | None, list[tuple[float, dict]]]:
    """The last prove process in the stamps: its start, its exit, and its LIVE records with the epoch each was printed."""
    start = end = None
    live: list[tuple[float, dict]] = []
    for ln in path.read_text().splitlines() if path.exists() else []:
        t, _, rest = ln.partition("\t")
        if rest.startswith("START"):
            start, end, live = float(t), None, []
        elif rest.startswith("EXIT"):
            end = float(t)
        elif rest.startswith("LIVE\t"):
            live.append((float(t), json.loads(rest[5:])))
    return start, end, live


def _dmon(path: Path) -> list[tuple[float, float | None, float | None]]:
    """(epoch, sm %, mem %) per sample of the one GPU dmon watched."""
    cols, out, gpus = None, [], set()
    for ln in path.read_text().splitlines() if path.exists() else []:
        t, _, rest = ln.partition("\t")
        f = rest.split()
        if not f:
            continue
        if f[0] == "#":
            if cols is None and "sm" in f:
                cols = f[1:]
            continue
        if cols is None or len(f) != len(cols):
            continue
        row = dict(zip(cols, f))
        gpus.add(row["gpu"])
        num = lambda k: float(row[k]) if row.get(k, "-").replace(".", "", 1).isdigit() else None  # noqa: E731
        out.append((float(t), num("sm"), num("mem")))
    if len(gpus) > 1:
        raise SystemExit(f"dmon watched GPUs {sorted(gpus)}: pass --gpu to 74-gemm-hill.sh's sampler")
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _window(samples, t0: float | None, t1: float | None) -> dict:
    """Mean sm and mem % of the samples in [t0, t1] (fractions), and how many."""
    if t0 is None or t1 is None:
        return {"t0": t0, "t1": t1, "s": None, "samples": 0, "sm": None, "mem": None}
    inside = [s for s in samples if t0 <= s[0] <= t1]
    sm, mem = _mean([s[1] for s in inside if s[1] is not None]), _mean([s[2] for s in inside if s[2] is not None])
    return {"t0": t0, "t1": t1, "s": round(t1 - t0, 3), "samples": len(inside), "sm": sm / 100 if sm is not None else None,
            "mem": mem / 100 if mem is not None else None}


def _pod_start() -> float | None:
    """In a Kubernetes pod: when its first process started (the GPU is the pod's from then), else None."""
    if not os.environ.get("KUBERNETES_SERVICE_HOST"):
        return None
    try:
        ticks = int(open("/proc/1/stat").read().rsplit(")", 1)[1].split()[19])
        btime = next(int(ln.split()[1]) for ln in open("/proc/stat") if ln.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, StopIteration, ValueError, IndexError):
        return None


def _others(path: Path | None, t0: float | None, t1: float | None) -> dict | None:
    """gemm_slowdown.sample's busy cores outside the job over [t0, t1]: median and max."""
    if path is None or not path.exists() or t0 is None or t1 is None:
        return None
    xs = []
    for ln in path.read_text().splitlines():
        m = re.match(r"(\S+) others_cores=([\d.]+)", ln)
        if m and t0 <= calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ")) <= t1 + 30:
            xs.append(float(m.group(2)))
    return {"median": statistics.median(xs), "max": max(xs), "samples": len(xs)} if xs else None


def _commit(root: Path) -> dict:
    """The tree's commit: research pods sync's .research-source.json, else git."""
    stamp_ = root / ".research-source.json"
    if stamp_.exists():
        d = json.loads(stamp_.read_text())
        return {"commit": d.get("commit"), "dirty": d.get("dirty"), "via": ".research-source.json"}
    try:
        c = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, timeout=60).stdout.strip())
        return {"commit": c or None, "dirty": dirty, "via": "git"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": os.environ.get("COMMIT"), "dirty": None, "via": "env COMMIT"}


def _phases(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for ln in path.read_text().splitlines() if path.exists() else []:
        f = ln.split("\t")
        if len(f) == 3:
            out.setdefault(f[0], [None, None])[0 if f[1] == "start" else 1] = float(f[2])
    return out


def metrics(a) -> None:
    now = time.time()
    census = {h["id"]: h for h in json.loads(Path(a.census).read_text())}
    peak = census[a.peak_id]
    head = Path(a.gpu_csv).read_text().splitlines()[:2] if a.gpu_csv and Path(a.gpu_csv).exists() else []
    gpu = dict(zip(head[0].split(","), (v.strip() for v in head[1].split(",")))) if len(head) == 2 else {}
    recs = [r for r in _records(Path(a.sweep)) if (m := DEFINITION.fullmatch(r.get("definition") or "")) and int(m.group(1)) == a.K]
    rec = recs[-1] if recs else {}
    lv, st = rec.get("live") or {}, rec.get("stage") or {}
    g = GS.gate(Path(a.gate) if a.gate else None, shapes_needed=1)
    if g["status"] == "passed" and rec.get("definition") not in g["shapes"]:
        g = {**g, "status": "failed", "why": f"the gate proved {sorted(g['shapes'])}, not {rec.get('definition')}"}
    start, end, live = _prove_process(Path(a.stamps))
    timed = [(t, x) for t, x in live if not x.get("warm")]
    warm = [(t, x) for t, x in live if x.get("warm")]
    samples = _dmon(Path(a.dmon))
    phases = _phases(Path(a.phases)) if a.phases else {}
    held_since = _pod_start() or (phases.get("job") or [None])[0] or (samples[0][0] if samples else None)
    flags = []
    out: dict = {"K": a.K, "definition": rec.get("definition"), "commit": _commit(Path(a.root)), "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
                 "gpu": gpu, "peak": {"id": a.peak_id, "flop_per_second": peak["flop_per_second"], "rate": peak.get("rate"),
                                      "device": peak.get("device"), "matches_gpu": bool(gpu.get("name")) and gpu.get("name") == peak.get("device")},
                 "byte_identity": g, "accepted": bool(lv.get("accepted")), "flags": flags}
    if not lv.get("accepted") or not timed:
        out["why"] = rec.get("broke") or lv.get("why") or ("no timed LIVE record in the stamps" if lv.get("accepted") else "no accepted statement")
    else:
        coords = (st.get("n") or lv["instances"]) * st.get("coords_per_instance", 1)
        stmt_s = GS.statement_seconds(lv)
        t_prove, t_native = stmt_s / coords, 2 * a.K / peak["flop_per_second"]
        t0 = (warm[-1][0] if warm else start)                 # the timed sessions: from the warm one's verdict to the last's
        t1 = timed[-1][0]
        runs = len(timed)
        steady = _window(samples, t0, t1)
        verify = [((x.get("verdict") or {}).get("verify_s")) for _, x in timed]
        session = [((x.get("verdict") or {}).get("session_s")) for _, x in timed]
        job = _window(samples, samples[0][0] if samples else None, now)
        held = now - held_since if held_since else None
        out.update({
            "m": lv.get("dense_m"), "k_log": lv.get("k_log"), "nbl": lv.get("nbl"), "instances": lv.get("instances"), "tile": st.get("tile"),
            "coords_per_statement": coords, "statement_digest": lv.get("statement_digest"), "shape": rec.get("shape"), "ands": rec.get("ands"),
            "statement_s": stmt_s, "prove_total_s": lv["prove_total_s"], "witness_s": lv.get("witness_s"),
            "witness_prebuilt_s": lv.get("witness_prebuilt_s"), "pipeline_depth": lv.get("pipeline_depth"),
            "t_prove_s_per_coord": t_prove, "t_native_s_per_coord": t_native, "overhead": t_prove / t_native,
            "throughput_coords_per_s": coords * runs / (t1 - t0), "throughput_prove_coords_per_s": 1 / t_prove,
            "gpu_util_steady": steady["sm"], "gpu_mem_util_steady": steady["mem"], "gpu_held_s_per_coord": (t1 - t0) / (coords * runs),
            "gpu_util_job": job["sm"] * job["s"] / held if job["sm"] is not None and held else None, "gpu_held_s_job": held,
            "verify_s_per_statement": statistics.median([v for v in verify if v is not None]) if any(v is not None for v in verify) else None,
            "session_s_per_statement": statistics.median([v for v in session if v is not None]) if any(v is not None for v in session) else None,
            "timed_runs": runs, "sessions": [{"t": t, "warm": bool(x.get("warm")), "e2e_s": x.get("e2e_s"), "prove_total_s": x.get("prove_total_s"),
                                              "witness_s": x.get("witness_s"), "wait_s": x.get("wait_s"), "verify_s": (x.get("verdict") or {}).get("verify_s"),
                                              "session_s": (x.get("verdict") or {}).get("session_s")} for t, x in live],
            "windows": {"steady": steady, "prove_process": _window(samples, start, end), "job": {**job, "held_since": held_since, "held_s": held},
                        **{k: _window(samples, *v) for k, v in phases.items() if k != "job"}},
            "others_cores_steady": _others(Path(a.others) if a.others else None, t0, t1),
            "prover_peak_gb": lv.get("prover_peak_gb"), "verifier_peak_gb": lv.get("verifier_peak_gb"), "buckets": lv.get("buckets"),
        })
        if lv.get("dense_m") != M:
            flags.append(f"m = {lv.get('dense_m')}, not {M}")
        if st.get("tile") and a.K == 2048:
            flags.append("rests on the unreviewed 4x4 tile")
        if not out["peak"]["matches_gpu"]:
            flags.append(f"the peak's device ({peak.get('device')}) is not this GPU ({gpu.get('name')})")
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    ms = [(k, u) for k, u in (("overhead", "x-native"), ("t_prove_s_per_coord", "s"), ("t_native_s_per_coord", "s"),
                              ("throughput_coords_per_s", "coords/s"), ("throughput_prove_coords_per_s", "coords/s"),
                              ("gpu_util_steady", "fraction"), ("gpu_util_job", "fraction"), ("gpu_held_s_per_coord", "s"),
                              ("gpu_held_s_job", "s"), ("verify_s_per_statement", "s"), ("statement_s", "s"),
                              ("coords_per_statement", "coords"), ("m", "log2")) if out.get(k) is not None]
    res = {"schema": "research/result/v0.1", "run_id": out["run_id"],
           "workload_fingerprint": {"kind": "gemm-hill/v1", "K": a.K, "definition": out["definition"], "statement_digest": out.get("statement_digest"),
                                    "m": out.get("m"), "instances": out.get("instances"), "tile": out.get("tile"), "gpu": gpu.get("name"),
                                    "peak_id": a.peak_id, "peak_flop_per_second": peak["flop_per_second"], "commit": out["commit"]["commit"],
                                    "flags": flags},
           "validation": {"status": g["status"] if out["accepted"] else "failed", "checker": "byte identity at B = 16: " + ", ".join(GS.GATE_CASES),
                          "detail": json.dumps(g.get("shapes"))[:900], "evidence": ["out/gate/classes/results.jsonl"]},
           "measurements": [{"name": k, "value": out[k], "unit": u} for k, u in ms],
           "measurement_files": ["out/metrics.json", "out/results.jsonl"]}
    tmp = Path(a.result).with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=str))
    os.replace(tmp, a.result)
    print(json.dumps({k: out.get(k) for k in ("K", "m", "overhead", "throughput_coords_per_s", "gpu_util_steady", "gpu_held_s_per_coord",
                                              "verify_s_per_statement", "flags")}), g["status"])


def _stored(run: str, research: str) -> dict:
    """A run's result: its result.json (a local run dir or file), else the store's bench-result/v1 for the attempt."""
    p = Path(run)
    if p.exists():
        return json.loads((p / "result.json" if p.is_dir() else p).read_text())
    arts = json.loads(subprocess.run([research, "data", "select", "--attempt", run, "--kind", "bench-result/v1", "--json"],
                                     capture_output=True, text=True, check=True).stdout)
    if not arts:
        raise SystemExit(f"{run}: no bench-result/v1 in the store")
    return arts[0]["manifest"]["meta"]


def _step_label(run: str, research: str) -> str | None:
    if Path(run).exists():
        return None
    r = subprocess.run([research, "data", "labels", run, "--key", "hill.step", "--json"], capture_output=True, text=True)
    try:
        rows = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    vals = [x.get("value") for x in (rows if isinstance(rows, list) else rows.get("labels", [])) if isinstance(x, dict)]
    return str(vals[-1]) if vals else None


def plot(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    research = os.environ.get("RESEARCH_BIN", "research")
    pts = []
    for i, run in enumerate(a.runs):
        doc = _stored(run, research)
        fp, ms = doc["workload_fingerprint"], {m["name"]: m["value"] for m in doc["measurements"]}
        step = a.steps[i] if a.steps and i < len(a.steps) else (_step_label(run, research) or (fp.get("commit") or "")[:9] or run)
        pts.append({"run": run, "step": step, "K": fp["K"], "gpu": fp.get("gpu"), "peak_id": fp.get("peak_id"), "flags": fp.get("flags"),
                    "byte_identity": doc["validation"]["status"], **ms})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "points.json").write_text(json.dumps(pts, indent=1))
    for K in sorted({p["K"] for p in pts}):
        ps = [p for p in pts if p["K"] == K]
        x = list(range(len(ps)))
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        bad = [i for i, p in enumerate(ps) if p["byte_identity"] != "passed"]
        panels = [("overhead", [("overhead", "T_prove / T_native", "o-")], "x native (log)", True),
                  ("throughput", [("throughput_coords_per_s", "sessions back to back, verifier included", "o-"),
                                  ("throughput_prove_coords_per_s", "1 / T_prove", "s--")], "coordinates / s", True),
                  ("GPU utilization", [("gpu_util_steady", "timed sessions", "o-"), ("gpu_util_job", "whole job", "s--")], "dmon sm %", False)]
        for axi, (title, series, ylab, log) in zip(ax, panels):
            for key, lab, sty in series:
                ys = [p.get(key) for p in ps]
                ys = [100 * y if y is not None and title == "GPU utilization" else y for y in ys]
                axi.plot([i for i, y in zip(x, ys) if y is not None], [y for y in ys if y is not None], sty, label=lab)
                axi.plot([i for i in bad if ys[i] is not None], [ys[i] for i in bad if ys[i] is not None], "rx", ms=12)
            axi.set_title(title)
            axi.set_ylabel(ylab)
            if log:
                axi.set_yscale("log")
            else:
                axi.set_ylim(0, 100)
            axi.set_xticks(x, [p["step"] for p in ps], rotation=30, ha="right", fontsize=8)
            axi.grid(alpha=0.3)
            axi.legend(fontsize=7)
        flags = sorted({f for p in ps for f in (p.get("flags") or [])})
        fig.suptitle(f"GemmCoordinate_v2{{K={K}, DOT={DOT}}} on {ps[-1]['gpu']}; P_peak {ps[-1]['peak_id']}"
                     + (f"; flags: {'; '.join(flags)}" if flags else "") + ("; red x: byte identity not passed" if bad else ""), fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"gemm-hill-K{K}.png", dpi=120)
        plt.close(fig)
    print(json.dumps([{k: p.get(k) for k in ("run", "step", "K", "overhead", "throughput_coords_per_s", "gpu_util_steady")} for p in pts]))
    if a.put:
        meta = {"what": "gemm-hill plots, one per K: overhead, throughput and GPU utilization over the steps", "runs": a.runs,
                "steps": [p["step"] for p in pts], "question": QUESTION}
        r = subprocess.run([research, "data", "put", "--kind", "plot/v1", "--tree", str(out), "--meta", json.dumps(meta), "--preserve"],
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode:
            raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd_", required=True)
    d = sub.add_parser("defs")
    d.add_argument("--K", type=int, required=True)
    d.add_argument("--n", type=int, default=None)
    d.add_argument("--out", required=True)
    dm = sub.add_parser("dmon")
    dm.add_argument("--out", required=True)
    dm.add_argument("--gpu", default=None)
    s = sub.add_parser("stamp")
    s.add_argument("--out", required=True)
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    m = sub.add_parser("metrics")
    m.add_argument("--K", type=int, required=True)
    m.add_argument("--gate")
    m.add_argument("--sweep", required=True)
    m.add_argument("--stamps", required=True)
    m.add_argument("--dmon", required=True)
    m.add_argument("--phases")
    m.add_argument("--others")
    m.add_argument("--gpu-csv")
    m.add_argument("--census", default="census/hardware.json")
    m.add_argument("--peak-id", default=PEAK_ID)
    m.add_argument("--root", default=".")
    m.add_argument("--out", required=True)
    m.add_argument("--result", required=True)
    p = sub.add_parser("plot")
    p.add_argument("runs", nargs="+")
    p.add_argument("--steps", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--put", action="store_true")
    a = ap.parse_args()
    {"defs": defs, "dmon": dmon, "stamp": stamp, "metrics": metrics, "plot": plot}[a.cmd_](a)


if __name__ == "__main__":
    main()
