"""BF16 GemmCoordinate hillclimb: one K's proving overhead against its GPU's spec peak, its throughput and its GPU utilization.

An attempt (74-gemm-hill.sh) gates the prover's byte identity at B = 16 (the GPU's proofs and transcripts equal the CPU prover's,
and the device paths agree), then proves one m = 35 statement of `GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}` (the BF16
dot of 16 that sm_90's wgmma and sm_120's mma.m16n8k16 run) untiled, warm once and timed three times, with nvidia-smi sampling
the GPU every second from the job's start. Per coordinate (one FP32 output word of a K-long dot product):

- overhead = T_session / T_native: the interaction with the verifier is part of the algorithm. T_session is the timed
  sessions' wall time back to back (prover work, every coin round trip, each verdict), from the warm session's verdict to the
  last timed one's, over their coordinates; T_native = 2 K FLOP / P_peak, P_peak the census line's dense BF16 tensor-core peak
  at FP32 accumulate (--peak-id). Beside it, prove-only: M0's T_prove (gemm_slowdown.statement_seconds, the median timed
  statement's witness wait plus both reps) over T_native.
- throughput: coordinates per second, 1 / T_session, with 1 / T_prove beside it.
- GPU utilization: nvidia-smi dmon's sm% (the fraction of each second some kernel ran, so an upper bound on DCGM's SM-active)
  averaged over the timed sessions (steady) and over the job's whole GPU-held wall time (the seconds before the sampler started
  count as idle); and the timed sessions' GPU-held seconds per coordinate.
- the verifier's seconds per statement: its verdict's verify_s, the median of the timed sessions.

A coordinate is the campaign contract's VU (verity_numerical.bench.contract: K MACs, 2 K FLOP): result.json is a contract result
(t.total = T_prove per statement, its buckets, vu.*, verifier.seconds, fixed.setup_seconds = staging), with the overhead against
the RTX PRO peak under the workload's own `hill.` names, since the contract's overhead.vs_native_peak divides by the profile's
H100 anchor. hillclimb.json is the step's verity/hillclimb-point/v0 record (throughput = VUs / T_session; gpu_util and
gpu_held_s_per_vu over the timed sessions, the verifier's verdicts included), which `rollup` appends to its subcircuit's roll-up.

    python gemm_hill.py defs --K 8192 --out defs.json [--n 1024]      # the Gemm_v2 row whose coordinates are the statement's lanes
    python gemm_hill.py dmon --out dmon.tsv [--gpu 0]                  # nvidia-smi dmon -s u every second, each line after its epoch
    python gemm_hill.py stamp --out stamps.tsv -- BINARY prove ...    # the prover, its stdout passed through and each line stamped
    python gemm_hill.py metrics --K 8192 --gate gate.jsonl --sweep results.jsonl --stamps stamps.tsv --dmon dmon.tsv \
        --phases phases.tsv --census census/hardware.json --out metrics.json --result result.json --hillclimb hillclimb.json
    python gemm_hill.py rollup --point hillclimb.json --file /workspace/usage/hillclimb/bf16-GemmCoordinate_v2-K8192.json
    python gemm_hill.py plot POINTS... --out DIR [--put]   # per K, the three panels over the steps (roll-ups, points or run ids)
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
PEAK_ID = "rtx-pro-6000-server/bf16"
QUESTION = ("what is BF16 GemmCoordinate_v2{K} proving overhead against the RTX PRO 6000 spec peak, its throughput and its GPU "
            "utilization, at this step?")
#: the campaign target whose operands and accumulator these are (verity_flock.bench.PROFILES["bf16-hopper"])
PROFILE = "bf16-hopper-mma-draft/2026-09-22"
#: a session's coins, from its verdict: `os-seed-prf` (M0's default: the verifier's 256-bit OS-random seed, committed at Hello
#: and opened in the verdict, every round's coins its SHA-256 counter-mode PRF) or `live-os` (the verifier's OS draws, round by
#: round); points under different coin modes are never compared
COINS = {"os-seed-prf": "os-seed-prf: the verifier's OS-random seed under a SHA-512 commitment, its coins a SHA-256 counter-mode PRF",
         "live-os": "live-os: the verifier's OS draws, round by round"}
#: the proof-system scheme a point measured, from its coins (docs/protocol-schemes.md): flock-i-v1 is M0's configuration (SHA-512
#: Merkle, two Fast100 repetitions, os-seed-prf coins, the batched session, the tile layout); written into every point, never
#: inferred later
SCHEMES = {"os-seed-prf": "flock-i-v1", "live-os": "flock-i-v1+livecoins"}
POINT_SCHEMA, ROLLUP_SCHEMA = "verity/hillclimb-point/v0", "verity/hillclimb-rollup/v0"


def subcircuit(K: int) -> dict:
    return {"id": f"bf16/GemmCoordinate_v2/K={K}", "dtype": "bf16", "definition": "GemmCoordinate_v2", "K": K, "dot": DOT}


def defs(a) -> None:
    n = a.n or INSTANCES.get(a.K)
    if not n:
        raise SystemExit(f"K = {a.K}: no instance count for m = {M} (pass --n)")
    Path(a.out).write_text(json.dumps({f'Gemm_v2{{K={a.K},N={n},DOT={{"fn":"{DOT}"}}}}': 1}) + "\n")
    print(n)


def dmon(a) -> None:
    """nvidia-smi dmon -s um (utilization and framebuffer MiB) at 1 s until SIGTERM, each line prefixed with the epoch seconds
    it arrived."""
    p = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1"] + (["-i", a.gpu] if a.gpu else []), stdout=subprocess.PIPE, text=True)

    def stop(*_):
        p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    with open(a.out, "a") as out:
        for ln in p.stdout:
            out.write(f"{time.time():.3f}\t{ln}")
            out.flush()


def _cpulist(s: str) -> set[int]:
    """'96-111,120' -> {96, ..., 111, 120}"""
    out: set[int] = set()
    for part in s.strip().split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def cores(a) -> None:
    """Every `every` seconds until killed: the busy cores among the job's own (`--cpus`; /proc/stat counts the host's) and the
    cores used by the processes the job can see (its pod's) that are pinned inside them. The difference is load from anything
    else on those cores, another job pinned to the same ones above all: node 1 runs every dispatched job under one cpuset."""
    cpus, tick = _cpulist(a.cpus), os.sysconf("SC_CLK_TCK")

    def busy() -> float:
        t = 0
        with open("/proc/stat") as f:
            for ln in f:
                if ln[:3] == "cpu" and ln[3].isdigit():
                    name, *v = ln.split()
                    if int(name[3:]) in cpus:
                        t += sum(map(int, v)) - int(v[3]) - int(v[4])
        return t / tick

    def own() -> float:
        t = 0
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    with open(f"/proc/{d}/status") as f:
                        allowed = next(ln.split(":", 1)[1] for ln in f if ln.startswith("Cpus_allowed_list:"))
                    if not _cpulist(allowed) <= cpus:
                        continue
                    with open(f"/proc/{d}/stat") as f:
                        s = f.read().rsplit(")", 1)[1].split()
                except (OSError, StopIteration):
                    continue
                t += sum(int(x) for x in s[11:15])
        return t / tick

    b0, o0, t0 = busy(), own(), time.time()
    with open(a.out, "a") as out:
        while True:
            time.sleep(a.every)
            b1, o1, t1 = busy(), own(), time.time()
            sl, mine = (b1 - b0) / (t1 - t0), max(0.0, o1 - o0) / (t1 - t0)
            out.write(f"{t1:.3f}\t{sl:.2f}\t{mine:.2f}\t{max(0.0, sl - mine):.2f}\n")
            out.flush()
            b0, o0, t0 = b1, o1, t1


def _slice(path: Path | None, t0: float, t1: float) -> dict | None:
    """The cores sampler's means over [t0, t1]: the slice's busy cores, the job's, and another job's on the same cores."""
    rows = [[float(x) for x in ln.split("\t")] for ln in (path.read_text().splitlines() if path and path.exists() else []) if ln.count("\t") == 3]
    inside = [r for r in rows if t0 <= r[0] <= t1]
    if not inside:
        return None
    return {"samples": len(inside), "slice_busy": _mean([r[1] for r in inside]), "own": _mean([r[2] for r in inside]),
            "others": _mean([r[3] for r in inside]), "others_max": max(r[3] for r in inside)}


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


def _dmon(path: Path) -> list[tuple[float, float | None, float | None, float | None]]:
    """(epoch, sm %, mem %, framebuffer MiB used) per sample of the one GPU dmon watched."""
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
        out.append((float(t), num("sm"), num("mem"), num("fb")))
    if len(gpus) > 1:
        raise SystemExit(f"dmon watched GPUs {sorted(gpus)}: pass --gpu to 74-gemm-hill.sh's sampler")
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _window(samples, t0: float | None, t1: float | None) -> dict:
    """Mean sm and mem % of the samples in [t0, t1] (fractions), how many, and the most framebuffer MiB any sample saw."""
    if t0 is None or t1 is None:
        return {"t0": t0, "t1": t1, "s": None, "samples": 0, "sm": None, "mem": None, "fb_mib_max": None}
    inside = [s for s in samples if t0 <= s[0] <= t1]
    sm, mem = _mean([s[1] for s in inside if s[1] is not None]), _mean([s[2] for s in inside if s[2] is not None])
    fb = [s[3] for s in inside if len(s) > 3 and s[3] is not None]
    return {"t0": t0, "t1": t1, "s": round(t1 - t0, 3), "samples": len(inside), "sm": sm / 100 if sm is not None else None,
            "mem": mem / 100 if mem is not None else None, "fb_mib_max": max(fb) if fb else None}


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
        t_session = (t1 - t0) / (coords * runs)
        verdicts = [x.get("verdict") or {} for _, x in timed]
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
            "t_prove_s_per_coord": t_prove, "t_native_s_per_coord": t_native, "t_session_s_per_coord": t_session,
            "overhead": t_session / t_native, "overhead_prove_only": t_prove / t_native,
            "throughput_coords_per_s": 1 / t_session, "throughput_prove_coords_per_s": 1 / t_prove,
            "gpu_util_steady": steady["sm"], "gpu_mem_util_steady": steady["mem"], "gpu_held_s_per_coord": t_session,
            "coins": "os-seed-prf" if any(v.get("coin_seed") for v in verdicts) else "live-os",
            "session": {"tables": len(timed[-1][1].get("tables") or [None]), "streams": verdicts[-1].get("streams"),
                        "rounds": verdicts[-1].get("rounds"), "round_trips": timed[-1][1].get("calls"),
                        "why": "--session-tables J: the tables share one link exchange and both reps one root (Session.lean); "
                               "the device prover proves one table per session, so J = 1 on the GPU"},
            "gpu_util_job": job["sm"] * job["s"] / held if job["sm"] is not None and held else None, "gpu_held_s_job": held,
            "gpu_mem_peak_gib": job["fb_mib_max"] / 1024 if job.get("fb_mib_max") is not None else None,
            "verify_s_per_statement": statistics.median([v for v in verify if v is not None]) if any(v is not None for v in verify) else None,
            "session_s_per_statement": statistics.median([v for v in session if v is not None]) if any(v is not None for v in session) else None,
            "timed_runs": runs, "sessions": [{"t": t, "warm": bool(x.get("warm")), "e2e_s": x.get("e2e_s"), "prove_total_s": x.get("prove_total_s"),
                                              "witness_s": x.get("witness_s"), "wait_s": x.get("wait_s"), "verify_s": (x.get("verdict") or {}).get("verify_s"),
                                              "session_s": (x.get("verdict") or {}).get("session_s")} for t, x in live],
            "windows": {"steady": steady, "prove_process": _window(samples, start, end), "job": {**job, "held_since": held_since, "held_s": held},
                        **{k: _window(samples, *v) for k, v in phases.items() if k != "job"}},
            "others_cores_steady": _others(Path(a.others) if a.others else None, t0, t1),
            "slice_cores_steady": _slice(Path(a.slice) if a.slice else None, t0, t1),
            "prover_peak_gb": lv.get("prover_peak_gb"), "verifier_peak_gb": lv.get("verifier_peak_gb"), "buckets": lv.get("buckets"),
            "setup_s": rec.get("stage_s"), "domain": rec.get("domain"),
        })
        if lv.get("dense_m") != M:
            flags.append(f"m-{lv.get('dense_m')}-not-{M}")
        if st.get("tile") and a.K == 2048:
            flags.append("tile-statement-unreviewed")
        out["session"]["label"] = f"batched-J{out['session']['tables']}"
        # another job on the job's own cores for more than half a core over the timed sessions
        if (out["slice_cores_steady"] or {}).get("others", 0) > 0.5:
            flags.append("cpu-slice-shared")
        # the GPU held while this job staged the statement itself (74-gemm-hill.sh STAGE_ONLY=1 stages it in a job without one)
        if not rec.get("stage_cached"):
            flags.append("staged-on-gpu")
    # #554 is a draft, its statements unreviewed: these are costs, not claims, and a changed statement digest starts a new curve
    flags.append("draft-554-unreviewed")
    if not out["peak"]["matches_gpu"]:
        flags.append("peak-device-mismatch")
    if out.get("accepted") and not rec.get("domain"):
        flags.append("domain-unrecorded")
    out["vcpu"] = len(os.environ["HILL_PINNED"].split(",")) if os.environ.get("HILL_PINNED") else len(os.sched_getaffinity(0))
    status = g["status"] if out["accepted"] else "failed"
    try:
        res = _contract_result(a, out, g, status, rec, lv, peak)
    except (ImportError, ValueError, KeyError, TypeError) as e:
        res = {"schema": "research/result/v0.1", "run_id": out["run_id"] or "unknown", "workload_fingerprint": None, "measurements": [],
               "validation": {"status": "failed", "checker": "the campaign contract", "detail": f"{type(e).__name__}: {e}"[:900]}}
    out["contract_problems"] = _validate(res)
    point = _point(a, out, g, peak, gpu, flags, timed)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    for path, doc in ((a.result, res), (a.hillclimb, point)):
        if path:
            tmp = Path(path).with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=1, default=str))
            os.replace(tmp, path)
    print(json.dumps({k: point.get(k) for k in ("subcircuit", "overhead", "throughput_vu_per_s", "gpu_util", "gpu_held_s_per_vu",
                                                "verify_s_per_statement", "byte_identical", "flags")}), status,
          "contract:", out["contract_problems"] or "conforms")


def _validate(res: dict) -> list[str]:
    try:
        from verity_numerical.bench import contract
    except ImportError as e:
        return [f"contract unavailable: {e}"]
    return contract.validate(res)


def _contract_result(a, out: dict, g: dict, status: str, rec: dict, lv: dict, peak: dict) -> dict:
    """The campaign contract's result.json for this step (verity_numerical.bench.contract): one statement's coordinates are
    the VUs, t.total its prover seconds (M0's T_prove), the buckets both reps of the median timed run."""
    from verity_numerical.bench import contract

    b, ms = lv.get("buckets") or [], []
    if status == "passed" and out.get("statement_s"):
        n, total = out["coords_per_statement"], out["statement_s"]
        rep = lambda k: sum(r.get(k) or 0.0 for r in b)  # noqa: E731
        ms = [contract.measurement("t.total", total), contract.measurement("t.witness", rep("t.witness")),
              contract.measurement("t.encoding_commitment", rep("t.encoding_commitment")), contract.measurement("t.arithmetic", rep("t.arithmetic")),
              contract.measurement("t.serialization", rep("h.serialize")),
              contract.measurement("vu.count", n), contract.measurement("vu.seconds_per_vu", total / n), contract.measurement("vu.per_second", n / total),
              *([contract.measurement("t.total_live", max(total, out["session_s_per_statement"]))] if out.get("session_s_per_statement") else []),
              *([contract.measurement("verifier.seconds", out["verify_s_per_statement"])] if out.get("verify_s_per_statement") is not None else []),
              *([contract.measurement("fixed.setup_seconds", out["setup_s"])] if out.get("setup_s") is not None else []),
              # nvidia-smi's framebuffer use at 1 s samples over the job: a lower bound on the device peak
              *([contract.measurement("mem.peak_device_bytes", int(out["gpu_mem_peak_gib"] * 2**30))] if out.get("gpu_mem_peak_gib") else []),
              *[{"name": f"hill.{k}", "value": out[k], "unit": u} for k, u in (
                  ("overhead", "ratio"), ("overhead_prove_only", "ratio"), ("t_session_s_per_coord", "s"), ("t_native_s_per_coord", "s"),
                  ("throughput_coords_per_s", "1/s"), ("gpu_util_steady", "fraction"),
                  ("gpu_util_job", "fraction"), ("gpu_held_s_per_coord", "s"), ("gpu_held_s_job", "s"), ("session_s_per_statement", "s"))
                if out.get(k) is not None],
              {"name": "hill.peak_flop_per_second", "value": peak["flop_per_second"], "unit": "FLOP/s"},
              *[{"name": f"hill.t_{k[2:]}", "value": rep(k), "unit": "s"} for k in ("t.zerocheck", "t.lincheck", "t.ring_switch", "t.ligerito")
                if any(k in r for r in b)]]
    fp = contract.fingerprint(
        profile=PROFILE, domain=out.get("domain") or "total", K=a.K, B=out.get("coords_per_statement") or 1, proof_class="NON_ZK_PROOF",
        authentication="included-hash", zk_mode=contract.ZK_NONE,
        security={"target": -100, "profile": "fast100", "reps": 2, "coins": COINS.get(out.get("coins"), out.get("coins"))},
        hardware=contract.probe_hardware(),
        software={"backend": {"name": "flock-circuit", "configuration": "C-interactive/flock-circuit serve + prove --gpu (loopback verifier "
                              "in series)", "commit": out["commit"]["commit"], "dirty": out["commit"]["dirty"], "binary_sha256": _read(a.binary_sha),
                              "prover": lv.get("prover"), "env": {k: os.environ.get(k) for k in ("FC_PIPELINE_DEPTH", "FC_HOST_PREPIN")}},
                  "toolchain": {"python": sys.version.split()[0]}},
        preprocessing="none amortised: the statement's circuit is lowered and staged once per process (fixed.setup_seconds); every timed "
                      "session builds its witness and proves both reps afresh",
        retention="retained", instances=None,
        subcircuit=subcircuit(a.K)["id"], definition=out.get("definition"), statement_digest=out.get("statement_digest"), dense_m=out.get("m"),
        k_log=out.get("k_log"), tile=out.get("tile"), vcpu=out["vcpu"], hill_peak={"census_id": a.peak_id, "flop_per_second": peak["flop_per_second"],
                                                                                    "device": peak.get("device")},
        coins=out.get("coins"), session=out.get("session"), flags=out["flags"], hill={"step": a.step, "label": a.label})
    return {"schema": contract.RESULT_SCHEMA, "run_id": out["run_id"] or "local", "workload_fingerprint": fp,
            "validation": {"status": status, "checker": "byte identity at B = 16: " + ", ".join(GS.GATE_CASES),
                           "detail": json.dumps(g.get("shapes"))[:900], "evidence": ["out/gate/classes/results.jsonl"]},
            "measurements": ms, "measurement_files": ["out/metrics.json", "out/results.jsonl"], "artifacts": ["hillclimb.json"]}


def _read(path: str | None) -> str | None:
    try:
        return Path(path).read_text().split()[0] if path else None
    except (OSError, IndexError):
        return None


def _point(a, out: dict, g: dict, peak: dict, gpu: dict, flags: list[str], timed: list) -> dict:
    """The step's verity/hillclimb-point/v0 record (lanes/console/20260930T2259Z-handoff-from-proofs-hillclimb-plots-spec)."""
    t = timed[-1][0] if timed else time.time()
    cap = gpu.get("compute_cap") or ""
    return {"schema": POINT_SCHEMA, "subcircuit": subcircuit(a.K), "step": a.step, "label": a.label, "commit": out["commit"]["commit"],
            "run_id": out["run_id"], "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)),
            "hardware": {"gpu": gpu.get("name"), "sm": f"sm_{cap.replace('.', '')}" if cap else None, "node": a.node, "vcpu": out["vcpu"]},
            "peak_flop_per_s": peak["flop_per_second"], "peak_census_id": a.peak_id, "vus": out.get("coords_per_statement"),
            "t_session_s_per_vu": out.get("t_session_s_per_coord"), "t_prove_only_s_per_vu": out.get("t_prove_s_per_coord"),
            "setup_s": out.get("setup_s"), "overhead": out.get("overhead"), "overhead_prove_only": out.get("overhead_prove_only"),
            "throughput_vu_per_s": out.get("throughput_coords_per_s"), "gpu_util": out.get("gpu_util_steady"),
            "gpu_held_s_per_vu": out.get("gpu_held_s_per_coord"), "verify_s_per_statement": out.get("verify_s_per_statement"),
            "byte_identical": g["status"] == "passed" and bool(out.get("accepted")), "coins": out.get("coins"),
            "session": (out.get("session") or {}).get("label"), "scheme": SCHEMES.get(out.get("coins")),
            # flock-circuit is both the prover (`prove`) and the verifier (`serve`)
            "impl": {"commit": out["commit"]["commit"], "prover_sha256": _read(a.binary_sha), "verifier_sha256": _read(a.binary_sha)},
            "flags": flags, "question": os.environ.get("QUESTION") or QUESTION.replace("{K}", f"{{K={a.K}}}"),
            "gpu_util_window": "the timed sessions back to back, each verifier's verdict included (gpu_util_job: the whole GPU-held job)",
            "gpu_util_job": out.get("gpu_util_job"), "gpu_mem_peak_gib": out.get("gpu_mem_peak_gib"),
            "cpu_slice_others": (out.get("slice_cores_steady") or {}).get("others"), "why": out.get("why") or g.get("why")}


def rollup(a) -> None:
    """Append a point to its subcircuit's roll-up (verity/hillclimb-rollup/v0), the whole file written beside it and renamed over
    it; strings cut at 200 characters and the commit at 12. Every point names its run, and a run already there is not appended
    twice."""
    pt = json.loads(Path(a.point).read_text())
    f = Path(a.file)
    doc = json.loads(f.read_text())
    if doc.get("schema") != ROLLUP_SCHEMA or pt.get("schema") != POINT_SCHEMA or doc["subcircuit"]["id"] != pt["subcircuit"]["id"]:
        raise SystemExit(f"{f}: not the roll-up of {pt.get('subcircuit', {}).get('id')}")
    if not pt.get("run_id") or not pt.get("commit"):
        raise SystemExit(f"{a.point}: a point names its run and commit")
    if any(p.get("run_id") == pt["run_id"] for p in doc["points"]):
        print(f"{pt['run_id']} is already in {f}")
        return
    cut = lambda v: v[:200] if isinstance(v, str) else [cut(x) for x in v] if isinstance(v, list) else {k: cut(x) for k, x in v.items()} if isinstance(v, dict) else v  # noqa: E731
    pt = cut(pt)
    pt["commit"] = (pt.get("commit") or "")[:12]
    doc["points"].append(pt)
    doc["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc) + "\n")
    os.chmod(tmp, 0o664)
    os.replace(tmp, f)
    print(f"{f}: {len(doc['points'])} points")


def _points(src: str, research: str) -> list[dict]:
    """hillclimb points from a roll-up or point file, else the store's hillclimb-point/v0 output of the attempt `src`."""
    p = Path(src)
    if p.exists():
        doc = json.loads(p.read_text())
        return doc["points"] if doc.get("schema") == ROLLUP_SCHEMA else [doc]
    arts = json.loads(subprocess.run([research, "data", "select", "--attempt", src, "--kind", "hillclimb-point/v0", "--json"],
                                     capture_output=True, text=True, check=True).stdout)
    if not arts:
        raise SystemExit(f"{src}: no hillclimb-point/v0 in the store")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        subprocess.run([research, "data", "fetch", arts[0]["id"], "--to", d], capture_output=True, text=True, check=True)
        return [json.loads(f.read_text()) for f in Path(d).rglob("*.json")][:1]


def plot(a) -> None:
    """Per subcircuit, the console's three panels over the steps: overhead (log, best-so-far over byte-identical points) with
    the prove-only overhead beside it, throughput, and GPU utilization with GPU-held and prove-only seconds per VU on their own
    axis; a point that fails the gate is hollow. Only the points under the last point's coins and session are drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    research = os.environ.get("RESEARCH_BIN", "research")
    pts = [pt for src in a.points for pt in _points(src, research)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "points.json").write_text(json.dumps(pts, indent=1))
    for sid in sorted({p["subcircuit"]["id"] for p in pts}, key=lambda s: (len(s), s)):
        ps = sorted((p for p in pts if p["subcircuit"]["id"] == sid), key=lambda p: (p.get("time_utc") or ""))
        # a point whose session never ran (refused, failed before a verdict) has no coins: it is drawn under any mode
        ran = [p for p in ps if p.get("coins")]
        mode = (ran[-1].get("coins"), ran[-1].get("session")) if ran else (None, None)
        other = len([p for p in ran if (p.get("coins"), p.get("session")) != mode])
        ps = sorted((p for p in ps if not p.get("coins") or (p.get("coins"), p.get("session")) == mode),
                    key=lambda p: (p["step"], p.get("time_utc") or ""))
        good = [p for p in ps if p.get("byte_identical")]
        fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        def series(axi, key, color, label, scale=1.0):
            for p in ps:
                if p.get(key) is not None:
                    axi.plot(p["step"], p[key] * scale, "o", color=color, mfc=color if p.get("byte_identical") else "none", ms=8)
            axi.plot([], [], "o", color=color, label=label)

        series(ax[0], "overhead", "C0", "overhead = peak / R, R over T_session (verifier included)")
        series(ax[0], "overhead_prove_only", "C4", "prove-only overhead")
        best, lo = [], None
        for p in good:
            if p.get("overhead") is not None:
                lo = p["overhead"] if lo is None else min(lo, p["overhead"])
                best.append((p["step"], lo))
        if best:
            ax[0].step([s for s, _ in best], [v for _, v in best], where="post", color="C0", alpha=0.5, label=f"best so far {best[-1][1]:.3g}x")
        ax[0].set_yscale("log")
        ax[0].set_ylabel("x native peak (log)")
        for p in ps:
            if p.get("overhead") is None:
                ax[0].text(p["step"], 0.04, (p.get("why") or "failed")[:44], transform=ax[0].get_xaxis_transform(),
                           rotation=90, fontsize=6, ha="center", va="bottom", color="C3", clip_on=True)
        series(ax[1], "throughput_vu_per_s", "C1", "VUs / T_session")
        ax[1].set_ylabel("VUs (coordinates) / s")
        series(ax[2], "gpu_util", "C2", "SM-active fraction (timed sessions)")
        ax[2].set_ylim(0, 1)
        ax[2].set_ylabel("GPU utilization")
        tw = ax[2].twinx()
        series(tw, "gpu_held_s_per_vu", "C3", "GPU-held s / VU")
        series(tw, "t_prove_only_s_per_vu", "C4", "prove-only s / VU")
        tw.set_ylabel("s / VU")
        tw.set_ylim(bottom=0)
        steps = sorted({p["step"] for p in ps})
        ax[2].set_xticks(steps, [str(s) for s in steps])
        ax[2].set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        ax[2].set_xlabel("optimization step")
        for axi in ax[:2]:
            axi.legend(fontsize=7, loc="best")
        ax[2].legend(fontsize=7, loc="upper left")
        tw.legend(fontsize=7, loc="upper right")
        for axi in ax:
            axi.grid(alpha=0.3)
        last = ps[-1]
        flags = sorted({f for p in ps for f in (p.get("flags") or [])})
        fig.suptitle(f"{sid}, DOT={last['subcircuit']['dot']}\n{last['hardware']['gpu']} ({last['hardware'].get('vcpu')} vCPU); peak "
                     f"{last['peak_census_id']} {last['peak_flop_per_s'] / 1e12:.0f} TFLOPS\ncoins {mode[0]}, session {mode[1]}"
                     + (f" ({other} earlier points under other coins or sessions not drawn)" if other else "")
                     + (f"\nflags: {', '.join(flags)}" if flags else "")
                     + ("; hollow: byte identity not passed" if len(good) < len(ps) else ""), fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"hill-{sid.replace('/', '-').replace('=', '')}.png", dpi=110)
        plt.close(fig)
    print(json.dumps([{k: p.get(k) for k in ("run_id", "step", "overhead", "throughput_vu_per_s", "gpu_util", "gpu_held_s_per_vu")}
                      | {"id": p["subcircuit"]["id"]} for p in pts]))
    if a.put:
        meta = {"what": "hillclimb plots, one per subcircuit: overhead, throughput, GPU utilization and GPU-held s / VU over the steps",
                "sources": a.points, "runs": [p.get("run_id") for p in pts], "question": QUESTION}
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
    c = sub.add_parser("cores")
    c.add_argument("--cpus", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--every", type=float, default=5.0)
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
    m.add_argument("--slice")
    m.add_argument("--gpu-csv")
    m.add_argument("--census", default="census/hardware.json")
    m.add_argument("--peak-id", default=PEAK_ID)
    m.add_argument("--root", default=".")
    m.add_argument("--binary-sha", default=None)
    m.add_argument("--step", type=int, default=0)
    m.add_argument("--label", default="baseline")
    m.add_argument("--node", default="vy-nebius-1")
    m.add_argument("--out", required=True)
    m.add_argument("--result", required=True)
    m.add_argument("--hillclimb", default=None)
    r = sub.add_parser("rollup")
    r.add_argument("--point", required=True)
    r.add_argument("--file", required=True)
    p = sub.add_parser("plot")
    p.add_argument("points", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--put", action="store_true")
    a = ap.parse_args()
    {"defs": defs, "dmon": dmon, "cores": cores, "stamp": stamp, "metrics": metrics, "rollup": rollup, "plot": plot}[a.cmd_](a)


if __name__ == "__main__":
    main()
