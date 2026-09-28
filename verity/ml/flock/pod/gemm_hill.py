"""GemmCoordinate hillclimb (BF16, and the sm_120 FP8 / FP4 rows): one K's proving overhead against its GPU's spec peak, its
throughput and its GPU utilization.

An attempt (74-gemm-hill.sh, DTYPE=bf16|e4m3|nvf4|mxf4) runs a preflight check of the prover's byte identity at B = 16 (the
GPU's proofs and transcripts equal the CPU prover's, the device paths agree, and sessions verified beside the next ones' proving
equal those verified one at a time), then proves one m = 35 statement of the dtype's coordinate (DTYPES: BF16's
`GemmCoordinate_v2{K, DOT=HopperBF16WgmmaDot16_v1}`, the BF16 dot of 16 that sm_90's wgmma and sm_120's mma.m16n8k16 run;
`GemmCoordinate{E4m3,Nvf4,Mxf4}_v1{K, DOT}` over sm_120's FP8 dot of 32 and block-scaled FP4 dots of 64) untiled, warm once and
timed three times, with nvidia-smi sampling the GPU every second from the job's start. Per coordinate (one FP32 output word of a
K-long dot product):

- overhead = T_session / T_native = peak / R, R = 2 K VUs / T_session: the interaction with the verifier is part of the algorithm.
  T_session is the timed sessions' wall time back to back (prover work, every coin round trip, each verdict), from the warm
  session's verdict to the last timed one's, over their coordinates (staging is setup_s, beside it); T_native = 2 K FLOP / P_peak,
  P_peak the census line's dense tensor-core peak at FP32 accumulate (--peak-id, the dtype's by default). Beside it, prove-only:
  M0's T_prove (gemm_slowdown.statement_seconds, the median timed statement's witness wait plus both reps) over T_native.
- throughput: coordinates per second, 1 / T_session, with 1 / T_prove beside it (with FC_VERIFY_AHEAD the prover proves the
  next sessions while earlier ones are verified).
- GPU utilization: nvidia-smi dmon's sm% (the fraction of each second some kernel ran, so an upper bound on DCGM's SM-active)
  averaged over the timed sessions (steady) and over the job's whole GPU-held wall time (the seconds before the sampler started
  count as idle); and the timed sessions' GPU-held seconds per coordinate (T_session).
- the verifier's seconds per statement: its verdict's verify_s, the median of the timed sessions.

A coordinate is the campaign contract's VU (verity_numerical.bench.contract: K MACs, 2 K FLOP): result.json is a contract result
(t.total = T_prove per statement, its buckets, vu.*, verifier.seconds, fixed.setup_seconds = staging), with the overhead against
the RTX PRO peak under the workload's own `hill.` names, since the contract's overhead.vs_native_peak divides by the profile's
anchor; a dtype with no campaign target (MXFP4) gets the preflight check's verdict and the `hill.` names only. hillclimb.json is the step's
verity/hillclimb-point/v0 record (throughput = VUs / T_session; gpu_util and gpu_held_s_per_vu over the timed sessions, the
verifier's verdicts included), which `rollup` appends to its subcircuit's roll-up.

A GPU session (74-gemm-hill.sh POINTS=N) is one tree, one binary, one FC_* set and one K: one preflight check, then N points,
each the same statement's sweep. Every point's record cites the session's preflight check (outputs.json refs.preflight, its
record's sha256 and its key) and refuses itself when its key (the binaries, the FC_* settings, the GPU; `key`) differs from the
check's. Its GPU-held seconds are its own sweep's (gpu_held_s_point), the session's (gpu_held_s_session) and its share of them,
its sweep plus the seconds outside every point split equally (gpu_held_s_amortized, which sum to the session's).
gpu_held_s_job is the session's only when it has one point. With N > 1, each point's contract result is out/p<i>/result.json
and result.json is the session's (`session`).

    python gemm_hill.py defs --K 8192 --out defs.json [--n 1024] [--dtype e4m3]   # N and the class sweep's DEFS= or PROGRAM=
    python gemm_hill.py fill --dtype e4m3 --K 8192 --preflight staged.jsonl   # the instances that fill m = 35, from the preflight
                                                                              # check's staging
    python gemm_hill.py dmon --out dmon.tsv [--gpu 0]                  # nvidia-smi dmon -s u every second, each line after its epoch
    python gemm_hill.py stamp --out stamps.tsv -- BINARY prove ...    # the prover, its stdout passed through and each line stamped
    python gemm_hill.py key --binary-sha binary.sha256 --out key.json # what a point shares with its session's preflight check
    python gemm_hill.py metrics --K 8192 --preflight preflight.jsonl --sweep results.jsonl --stamps stamps.tsv --dmon dmon.tsv \
        --phases phases.tsv --census benchmarks/census/hardware.json --out metrics.json --result result.json --hillclimb hillclimb.json \
        [--point I --points N --preflight-key preflight/key.json --point-key key.json]
    python gemm_hill.py session --dir RUN_DIR [--points N]            # outputs.json, and with N > 1 the session's result.json
    python gemm_hill.py rollup --attempt RUN [--run-dir DIR] --file /workspace/usage/hillclimb/bf16-GemmCoordinate_v2-K8192.json
                                                                       # every point RUN published, each citing its arts
    python gemm_hill.py rollup --point hillclimb.json --file ROLLUP    # one point not in the store, citing none
    python gemm_hill.py plot POINTS... --out DIR [--put]   # per K, the three panels over the steps (roll-ups, points or run ids)
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
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

#: dtype -> the coordinate Definition, its dot, the census line of its dense peak at FP32 accumulate, and the campaign target
#: (verity.ml.circuits.targets.target) whose operands and accumulator these are: MXFP4 has none, so its result.json names none
DTYPES = {"bf16": ("GemmCoordinate_v2", "HopperBF16WgmmaDot16_v1", "rtx-pro-6000-server/bf16", "bf16-hopper-mma-draft/2026-09-22"),
          "e4m3": ("GemmCoordinateE4m3_v1", "BlackwellE4m3QmmaDot32_v1", "rtx-pro-6000-server/e4m3", "fp8-sm120-mma-draft/2026-09-30"),
          "nvf4": ("GemmCoordinateNvf4_v1", "BlackwellNvf4OmmaDot64_v1", "rtx-pro-6000-server/e2m1", "nvfp4-sm120-mma-draft/2026-09-22"),
          "mxf4": ("GemmCoordinateMxf4_v1", "BlackwellMxf4OmmaDot64_v1", "rtx-pro-6000-server/e2m1", None)}
M = 35
#: BF16's instances per statement at m = 35: 2^(35 - k_log), k_log the block circuit.compose packs one coordinate into (the unit's
#: row slot and its rows' hashing): 24, 24, 25, 26, 27. K = 32768 is at Stmt.InRange's k_log <= 27. The FP8 / FP4 rows take theirs
#: from the preflight check's staging (`fill`)
INSTANCES = {2048: 2048, 4096: 2048, 8192: 1024, 16384: 512, 32768: 256}
#: a session's coins, as the sweep's record names them (`verity_flock.circuit.coins`): `os-seed-prf` (the binary's default: the
#: verifier's 256-bit OS-random seed, committed at Hello and opened in the verdict, every round's coins its SHA-256 counter-mode
#: PRF) or `live-os` (FC_COINS=os, 70-class-sweep.sh's default: the verifier's OS draws, round by round); points under different
#: coin modes are never compared
COINS = {"os-seed-prf": "os-seed-prf: the verifier's OS-random seed under a SHA-512 commitment, its coins a SHA-256 counter-mode PRF",
         "live-os": "live-os: the verifier's OS draws, round by round"}
#: the proof-system scheme a point measured, from its coins (docs/protocol-schemes.md): flock-i-v1 is M0's configuration (SHA-512
#: Merkle, two Fast100 repetitions, os-seed-prf coins, the batched session, the tile layout); written into every point, never
#: inferred later
SCHEMES = {"os-seed-prf": "flock-i-v1", "live-os": "flock-i-v1+livecoins"}
POINT_SCHEMA, ROLLUP_SCHEMA = "verity/hillclimb-point/v0", "verity/hillclimb-rollup/v0"
def definition(dtype: str) -> re.Pattern:
    name, dot = DTYPES[dtype][:2]
    return re.compile(re.escape(name) + r"\{K=(\d+),DOT=" + dot + r"\}")


def subcircuit(K: int, dtype: str = "bf16") -> dict:
    name, dot = DTYPES[dtype][:2]
    return {"id": f"{dtype}/{name}/K={K}", "dtype": dtype, "definition": name, "K": K, "dot": dot}


def question(dtype: str, K: int) -> str:
    if dtype == "bf16":
        return (f"what is BF16 GemmCoordinate_v2{{K={K}}} proving overhead against the RTX PRO 6000 spec peak, its throughput and its "
                "GPU utilization, at this step?")
    return f"what is {dtype} GemmCoordinate{{K={K}}} proving overhead against the RTX PRO 6000 spec peak, at this step?"


def defs(a) -> None:
    """Print the statement's instance count and its class sweep source (70-class-sweep.sh's DEFS= or PROGRAM=)."""
    n = a.n or (INSTANCES.get(a.K) if a.dtype == "bf16" else None)
    if not n:
        raise SystemExit(f"K = {a.K}: no instance count for m = {M} (pass --n)")
    if a.dtype == "bf16":
        Path(a.out).write_text(json.dumps({f'Gemm_v2{{K={a.K},N={n},DOT={{"fn":"{DTYPES["bf16"][1]}"}}}}': 1}) + "\n")
        print(n, f"DEFS={a.out}")
    else:
        print(n, f"PROGRAM=verity/ml/flock/pod/gemm_fp.py:{a.dtype}_K{a.K}_N{n}")


def fill(a) -> None:
    """The instances that fill an m = M statement of the preflight check's class: the most, a power of two, whose blocks fit
    2^M bits."""
    recs = [r for r in _records(Path(a.preflight)) if (m := definition(a.dtype).fullmatch(r.get("definition") or "")) and int(m.group(1)) == a.K]
    st = (recs[-1] if recs else {}).get("stage") or {}
    if not st.get("k_log") or not st.get("vus_per_block"):
        raise SystemExit(f"{a.preflight}: no staged {a.dtype} K = {a.K} class")
    per_instance = (1 << st["k_log"]) // st["vus_per_block"]
    print(max(st["vus_per_block"], 1 << (((1 << M) // per_instance).bit_length() - 1)))


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


def _prove_process(path: Path) -> tuple[float | None, float | None, int | None, list[tuple[float, dict]]]:
    """The last prove process in the stamps: its start, its exit and exit status, and its LIVE records with the epoch each was
    printed."""
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


def _median(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _window(samples, t0: float | None, t1: float | None) -> dict:
    """Mean sm and mem % of the samples in [t0, t1] (fractions), how many, how many saw some kernel run (busy: sm > 0, infra's
    measure, one sample a second), and the most framebuffer MiB any sample saw."""
    if t0 is None or t1 is None:
        return {"t0": t0, "t1": t1, "s": None, "samples": 0, "busy": 0, "sm": None, "mem": None, "fb_mib_max": None}
    inside = [s for s in samples if t0 <= s[0] <= t1]
    sm, mem = _mean([s[1] for s in inside if s[1] is not None]), _mean([s[2] for s in inside if s[2] is not None])
    fb = [s[3] for s in inside if len(s) > 3 and s[3] is not None]
    return {"t0": t0, "t1": t1, "s": round(t1 - t0, 3), "samples": len(inside), "busy": sum(1 for s in inside if (s[1] or 0) > 0),
            "sm": sm / 100 if sm is not None else None, "mem": mem / 100 if mem is not None else None, "fb_mib_max": max(fb) if fb else None}


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


#: per verifier source, its mark, the flag a version nobody reviewed carries, and the git blob ids of the reviewed versions with
#: the flags each still carries. The column-major fold as granted (d1775df80 = cde1c7ac1, art:90b53162093c43559020e115d72d500225d8af62466cbbcedb7e63aada983fd8)
#: and the structured lincheck as granted (8db1cb550 = e71789ed7 = f7d1e297d, art:f865ea7822990772c557751642e3c0c17212b6703ec2f8f5292f5f63fe5dd1f1)
#: carry none, alone or with ab5087cd4 (= 750e344fd = acca35385, C0 = I checked once per statement, granted as
#: art:2aef6594e7d5510a22c7cb7e7afa54fd20d954ceabf3d910a7437567ba204686)
VERIFIER_SOURCES = (
    ("circuit.rs", "CscCircuit", "verifier-fold-unreviewed",
     {"1528b2ea3ede6998491ad0254c1a7917cd953263": (),
      "a38a04e336a687c52ab0d642aa4674e90669399d": ()}),
    ("session_verify.rs", "pub enum Lincheck", "lincheck-partial-unreviewed",
     {"a12b4c15fa1138e91b8ffda7040f49737af30bbd": (),
      "9d3b834f078fb2e9cfb6c37876eaea063d06dafc": ()}),
)


def verifier_flags(root: Path, lincheck: str | None) -> list[str]:
    """The flags of the verifier changes a tree's prover binary is built with (red-team-proofs-554's Q3 and Q3b), read from the
    sources it compiles, since a synced tree has no history: the column-major fold, and the structured lincheck when FC_LINCHECK
    leaves it on (unset or partial; `both` rejects unless it equals upstream's flat fold, `flat` is upstream's). A source that
    carries its change but none of the reviewed versions, or a circuit.rs that can't be read, is a change nobody reviewed."""
    live = root / "verity/ml/flock/live/src"
    flags: list[str] = []
    for name, mark, unreviewed, reviewed in VERIFIER_SOURCES:
        if name == "session_verify.rs" and (lincheck or "partial") != "partial":
            continue
        try:
            src = (live / name).read_bytes()
        except OSError:
            flags += [unreviewed] if name == "circuit.rs" else []
            continue
        if mark.encode() in src:
            flags += reviewed.get(hashlib.sha1(b"blob %d\0" % len(src) + src).hexdigest(), (unreviewed,))
    return list(dict.fromkeys(flags))


def _phases(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for ln in path.read_text().splitlines() if path.exists() else []:
        f = ln.split("\t")
        if len(f) == 3:
            out.setdefault(f[0], [None, None])[0 if f[1] == "start" else 1] = float(f[2])
    return out


def _span(phases: dict[str, list[float]], name: str) -> float | None:
    t = phases.get(name) or [None, None]
    return t[1] - t[0] if None not in t else None


def key(a) -> None:
    """What a point must share with its session's preflight check, taken as each runs: the binaries' sha256 (70-class-sweep.sh's
    binary.sha256, by path: a prover and a selftest build can share a name), every FC_* setting the prover inherits, and the GPU
    the pod was given."""
    shas = {}
    for ln in Path(a.binary_sha).read_text().splitlines() if Path(a.binary_sha).exists() else []:
        if len(f := ln.split()) == 2:
            shas[f[1]] = f[0]
    doc = {"binaries": shas, "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FC_")},
           "gpu": {k: os.environ.get(k) for k in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}}
    Path(a.out).write_text(json.dumps(doc, sort_keys=True) + "\n")


def _key_differs(pk: dict | None, qk: dict | None) -> list[str]:
    """The settings two keys don't share (``binaries.<path>``, ``env.<FC_*>``, ``gpu.<var>``); every one when either is missing."""
    flat = lambda k: {f"{s}.{n}": v for s, d in (k or {}).items() for n, v in d.items()}  # noqa: E731
    f, g = flat(pk), flat(qk)
    return ["the key"] if pk is None or qk is None else sorted(n for n in f.keys() | g.keys() if f.get(n) != g.get(n))


def _sha256(path: str | None) -> str | None:
    import hashlib

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except OSError:
        return None


def metrics(a) -> None:
    now = time.time()
    census = {h["id"]: h for h in json.loads(Path(a.census).read_text())}
    a.peak_id = a.peak_id or DTYPES[a.dtype][2]
    peak = census[a.peak_id]
    head = Path(a.gpu_csv).read_text().splitlines()[:2] if a.gpu_csv and Path(a.gpu_csv).exists() else []
    gpu = dict(zip(head[0].split(","), (v.strip() for v in head[1].split(",")))) if len(head) == 2 else {}
    recs = [r for r in _records(Path(a.sweep)) if (m := definition(a.dtype).fullmatch(r.get("definition") or "")) and int(m.group(1)) == a.K]
    rec = recs[-1] if recs else {}
    lv, st = rec.get("live") or {}, rec.get("stage") or {}
    g = GS.preflight(Path(a.preflight) if a.preflight else None, shapes_needed=1)
    if g["status"] == "passed" and rec.get("definition") not in g["shapes"]:
        g = {**g, "status": "failed", "why": f"the preflight check proved {sorted(g['shapes'])}, not {rec.get('definition')}"}
    # a session's points cite its one preflight check, and one whose key differs from the check's is refused
    pk, qk = (json.loads(Path(f).read_text()) if f and Path(f).exists() else None for f in (a.preflight_key, a.point_key))
    differs = _key_differs(pk, qk) if a.points > 1 or a.point_key else []
    if g["status"] == "passed" and differs:
        g = {**g, "status": "failed", "why": f"point {a.point}'s {', '.join(differs)} differ from its preflight check's"}
    cite = {"output": "preflight", "record_sha256": _sha256(a.preflight), "status": g["status"], "key": pk,
            "key_matches": not differs if a.points > 1 or a.point_key else None}
    start, end, rc, live = _prove_process(Path(a.stamps))
    timed = [(t, x) for t, x in live if not x.get("warm")]
    warm = [(t, x) for t, x in live if x.get("warm")]
    # accepted: the statement, and every session the prove process ran (warm and timed), all of them, and the process exited 0
    rejected = [x.get("run") for _, x in live if x.get("accepted") is not True]
    if not lv.get("accepted"):
        why_not = lv.get("why") or "no accepted statement"
    elif rc != 0:
        why_not = f"the prove process exited {rc}" if rc is not None else "no exit of the prove process in the stamps"
    elif (len(warm), len(timed)) != (a.warm, a.runs):
        why_not = f"{len(warm)} warm and {len(timed)} timed LIVE records in the stamps, not {a.warm} and {a.runs}"
    elif rejected:
        why_not = f"sessions {rejected} not accepted"
    elif not timed:
        why_not = "no timed LIVE record in the stamps"
    else:
        why_not = None
    samples = _dmon(Path(a.dmon))
    phases = _phases(Path(a.phases)) if a.phases else {}
    held_since = _pod_start() or (phases.get("job") or [None])[0] or (samples[0][0] if samples else None)
    # the session's GPU-held seconds end where its points' metrics start; each point's own are its sweep's, and the seconds
    # outside every point (pod start, the preflight check, the gaps) are shared equally by the points that ran
    t_end = (phases.get("metrics") or [None])[0] or now
    held = t_end - held_since if held_since else None
    names = ["sweep"] if a.points == 1 else [f"sweep-p{i}" for i in range(a.points)]
    spans = [x for x in (_span(phases, k) for k in names) if x is not None]
    own = _span(phases, names[0 if a.points == 1 else a.point])
    flags = []
    out: dict = {"K": a.K, "definition": rec.get("definition"), "commit": _commit(Path(a.root)), "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
                 "point": {"index": a.point, "of": a.points}, "preflight": cite, "gpu_held_s_point": own, "gpu_held_s_session": held,
                 "gpu_held_s_amortized": own + (held - sum(spans)) / len(spans) if own is not None and held is not None else None,
                 # one point's job is the whole session's only when the session has one point
                 "gpu_held_s_job": held if a.points == 1 else None,
                 "gpu": gpu, "peak": {"id": a.peak_id, "flop_per_second": peak["flop_per_second"], "rate": peak.get("rate"),
                                      "device": peak.get("device"), "matches_gpu": bool(gpu.get("name")) and gpu.get("name") == peak.get("device")},
                 "byte_identity": g, "accepted": why_not is None, "flags": flags,
                 "live_records": {"warm": len(warm), "timed": len(timed), "expected": [a.warm, a.runs], "rejected": rejected, "prove_exit": rc}}
    if why_not:
        out["why"] = rec.get("broke") or why_not
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
        job = _window(samples, samples[0][0] if samples else None, t_end)
        out.update({
            "m": lv.get("dense_m"), "k_log": lv.get("k_log"), "nbl": lv.get("nbl"), "instances": lv.get("instances"), "tile": st.get("tile"),
            "coords_per_statement": coords, "statement_digest": lv.get("statement_digest"), "shape": rec.get("shape"), "ands": rec.get("ands"),
            "statement_s": stmt_s, "prove_total_s": lv["prove_total_s"], "witness_s": lv.get("witness_s"),
            "witness_prebuilt_s": lv.get("witness_prebuilt_s"), "pipeline_depth": lv.get("pipeline_depth"), "verify_ahead": lv.get("verify_ahead"),
            "verify_servers": lv.get("verify_servers"), "t_prove_s_per_coord": t_prove, "t_native_s_per_coord": t_native, "t_session_s_per_coord": t_session,
            "overhead": t_session / t_native, "overhead_prove_only": t_prove / t_native,
            "throughput_coords_per_s": 1 / t_session, "throughput_prove_coords_per_s": 1 / t_prove,
            "gpu_util_steady": steady["sm"], "gpu_mem_util_steady": steady["mem"], "gpu_held_s_per_coord": t_session,
            "coins": lv.get("coins"),
            "session": {"tables": len(timed[-1][1].get("tables") or [None]), "streams": verdicts[-1].get("streams"),
                        "rounds": verdicts[-1].get("rounds"), "round_trips": timed[-1][1].get("calls"),
                        "why": "--session-tables J: the tables share one link exchange and both reps one root (Session.lean); "
                               "the device prover proves one table per session, so J = 1 on the GPU"},
            "gpu_util_job": job["sm"] * job["s"] / held if job["sm"] is not None and held else None,
            "gpu_busy_s_job": job["busy"],
            # where the verifier ran, and what a coin round trip costs the session: the 64-byte pings on each session's open
            # connection before it starts, and the seconds the prover waited inside its round trips (the verifier's handling included)
            "verifier": lv.get("verifier"), "open_rtt_ms": _median([x.get("open_rtt_ms") for _, x in timed]),
            "wait_e2e_s": _median([x.get("wait_e2e_s") for _, x in timed]),
            "gpu_mem_peak_gib": job["fb_mib_max"] / 1024 if job.get("fb_mib_max") is not None else None,
            "verify_s_per_statement": statistics.median([v for v in verify if v is not None]) if any(v is not None for v in verify) else None,
            "session_s_per_statement": statistics.median([v for v in session if v is not None]) if any(v is not None for v in session) else None,
            "timed_runs": runs, "sessions": [{"t": t, "warm": bool(x.get("warm")), "e2e_s": x.get("e2e_s"), "prove_total_s": x.get("prove_total_s"),
                                              "witness_s": x.get("witness_s"), "wait_s": x.get("wait_s"), "verify_s": (x.get("verdict") or {}).get("verify_s"),
                                              "session_s": (x.get("verdict") or {}).get("session_s"), "open_rtt_ms": x.get("open_rtt_ms"),
                                              "wait_e2e_s": x.get("wait_e2e_s")} for t, x in live],
            "windows": {"steady": steady, "prove_process": _window(samples, start, end), "job": {**job, "held_since": held_since, "held_s": held},
                        **{k: _window(samples, *v) for k, v in phases.items() if k != "job"}},
            "others_cores_steady": _others(Path(a.others) if a.others else None, t0, t1),
            "slice_cores_steady": _slice(Path(a.slice) if a.slice else None, t0, t1),
            "prover_peak_gb": lv.get("prover_peak_gb"), "verifier_peak_gb": lv.get("verifier_peak_gb"), "buckets": lv.get("buckets"),
            "setup_s": rec.get("stage_s"), "domain": rec.get("domain"),
        })
        if lv.get("dense_m") != M:
            flags.append(f"m-{lv.get('dense_m')}-not-{M}")
        out["session"]["label"] = f"batched-J{out['session']['tables']}"
        # another job on the job's own cores for more than half a core over the timed sessions
        if (out["slice_cores_steady"] or {}).get("others", 0) > 0.5:
            flags.append("cpu-slice-shared")
        # the GPU held while this job staged the statement itself (74-gemm-hill.sh STAGE_ONLY=1 stages it in a job without one)
        if not rec.get("stage_cached"):
            flags.append("staged-on-gpu")
    # #554's untiled statements are reviewed (red-team-proofs-554's Q1 grant at 8a0b17250); a tiled one is a cost, not a claim,
    # and a record without its stage can't show it isn't tiled. A changed statement digest starts a new curve
    if st.get("tile"):
        flags.append("tile-statement-unreviewed")
    if st.get("tile") or not rec.get("stage"):
        flags.append("draft-554-unreviewed")
    # four FP4 or two FP8 elements a u16 leaf (FLOCK_PACK_WORDS=1): another statement and input commitment, not reviewed yet
    if st.get("packed"):
        flags.append("packed-statement-unreviewed")
    # core's bit rows (FLOCK_CORE_ROWS=1): new core protocol, and the verifiers' derivation of their constants not reviewed yet
    if st.get("row_schemas"):
        flags.append("core-rows-statement-unreviewed")
    flags.extend(verifier_flags(Path(a.root), os.environ.get("FC_LINCHECK")))
    if not out["peak"]["matches_gpu"]:
        flags.append("peak-device-mismatch")
    if out.get("accepted") and not rec.get("domain"):
        flags.append("domain-unrecorded")
    if not DTYPES[a.dtype][3]:
        flags.append("no-campaign-target")
    out["vcpu"] = len(os.environ["HILL_PINNED"].split(",")) if os.environ.get("HILL_PINNED") else len(os.sched_getaffinity(0))
    status = g["status"] if out["accepted"] else "failed"
    try:
        res = _contract_result(a, out, g, status, rec, lv, peak) if DTYPES[a.dtype][3] else _bare_result(a, out, g, status, lv, peak)
    except (ImportError, ValueError, KeyError, TypeError) as e:
        res = {"schema": "research/result/v0.1", "run_id": out["run_id"] or "unknown", "workload_fingerprint": None, "measurements": [],
               "validation": {"status": "failed", "checker": "the campaign contract", "detail": f"{type(e).__name__}: {e}"[:900]}}
    out["contract_problems"] = _validate(res)
    if a.points > 1:
        res["kind"] = "bench-point/v1"
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


def _hill(out: dict, lv: dict, peak: dict) -> list[dict]:
    """The step's own measurements, under the workload's `hill.` names: the overhead against the RTX PRO peak and its parts."""
    b = lv.get("buckets") or []
    rep = lambda k: sum(r.get(k) or 0.0 for r in b)  # noqa: E731
    return [*[{"name": f"hill.{k}", "value": out[k], "unit": u} for k, u in (
                ("overhead", "ratio"), ("overhead_prove_only", "ratio"), ("t_session_s_per_coord", "s"), ("t_native_s_per_coord", "s"),
                ("throughput_coords_per_s", "1/s"), ("gpu_util_steady", "fraction"), ("gpu_util_job", "fraction"),
                ("gpu_held_s_per_coord", "s"), ("gpu_held_s_job", "s"), ("session_s_per_statement", "s"),
                ("gpu_held_s_point", "s"), ("gpu_held_s_amortized", "s"), ("gpu_held_s_session", "s")) if out.get(k) is not None],
            {"name": "hill.peak_flop_per_second", "value": peak["flop_per_second"], "unit": "FLOP/s"},
            *[{"name": f"hill.t_{k[2:]}", "value": rep(k), "unit": "s"} for k in ("t.zerocheck", "t.lincheck", "t.ring_switch", "t.ligerito")
              if any(k in r for r in b)]]


def _bare_result(a, out: dict, g: dict, status: str, lv: dict, peak: dict) -> dict:
    """result.json for a dtype with no campaign target: no contract fingerprint (its profile must be a target), the preflight
    check's verdict and the step's `hill.` measurements."""
    fp = {"contract": None, "subcircuit": subcircuit(a.K, a.dtype)["id"], "definition": out.get("definition"), "K": a.K,
          "B": out.get("coords_per_statement"), "statement_digest": out.get("statement_digest"), "dense_m": out.get("m"),
          "k_log": out.get("k_log"), "domain": out.get("domain"), "coins": out.get("coins"), "session": out.get("session"), "commit": out["commit"]["commit"],
          "dirty": out["commit"]["dirty"], "binary_sha256": _read(a.binary_sha), "hill_peak": {"census_id": a.peak_id,
          "flop_per_second": peak["flop_per_second"], "device": peak.get("device")}, "vcpu": out["vcpu"], "flags": out["flags"]}
    pre = "" if a.points == 1 else f"p{a.point}/"
    return {"schema": "research/result/v0.1", "run_id": out["run_id"] or "local", "workload_fingerprint": fp,
            "validation": {"status": status, "checker": "byte identity at B = 16: " + ", ".join(GS.PREFLIGHT_CASES),
                           "detail": f"{a.dtype}: no verity.ml.circuits.targets.target, so no contract fingerprint; "
                                     + (g.get("why") or json.dumps(g.get("shapes")))[:700],
                           "evidence": ["out/preflight/classes/results.jsonl"]},
            "measurements": _hill(out, lv, peak) if status == "passed" and out.get("statement_s") else [],
            "measurement_files": [f"out/{pre}metrics.json", f"out/{pre}results.jsonl"],
            "artifacts": [Path(a.hillclimb).name if a.hillclimb else "hillclimb.json"]}


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
              *_hill(out, lv, peak)]
    fp = contract.fingerprint(
        profile=DTYPES[a.dtype][3], domain=out.get("domain") or "total", K=a.K, B=out.get("coords_per_statement") or 1, proof_class="NON_ZK_PROOF",
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
        subcircuit=subcircuit(a.K, a.dtype)["id"], definition=out.get("definition"), statement_digest=out.get("statement_digest"), dense_m=out.get("m"),
        k_log=out.get("k_log"), tile=out.get("tile"), vcpu=out["vcpu"], hill_peak={"census_id": a.peak_id, "flop_per_second": peak["flop_per_second"],
                                                                                    "device": peak.get("device")},
        coins=out.get("coins"), session=out.get("session"), flags=out["flags"], hill={"step": a.step, "label": a.label})
    pre = "" if a.points == 1 else f"p{a.point}/"
    return {"schema": contract.RESULT_SCHEMA, "run_id": out["run_id"] or "local", "workload_fingerprint": fp,
            "validation": {"status": status, "checker": "byte identity at B = 16: " + ", ".join(GS.PREFLIGHT_CASES),
                           "detail": (g.get("why") or json.dumps(g.get("shapes")))[:900], "evidence": ["out/preflight/classes/results.jsonl"]},
            "measurements": ms, "measurement_files": [f"out/{pre}metrics.json", f"out/{pre}results.jsonl"],
            "artifacts": [Path(a.hillclimb).name if a.hillclimb else "hillclimb.json"]}


def _read(path: str | None) -> str | None:
    try:
        return Path(path).read_text().split()[0] if path else None
    except (OSError, IndexError):
        return None


def _point(a, out: dict, g: dict, peak: dict, gpu: dict, flags: list[str], timed: list) -> dict:
    """The step's verity/hillclimb-point/v0 record (lanes/console/20260930T2259Z-handoff-from-proofs-hillclimb-plots-spec)."""
    t = timed[-1][0] if timed else time.time()
    cap = gpu.get("compute_cap") or ""
    return {"schema": POINT_SCHEMA, "subcircuit": subcircuit(a.K, a.dtype), "step": a.step, "label": a.label, "commit": out["commit"]["commit"],
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
            "flags": flags, "question": a.question or os.environ.get("QUESTION") or question(a.dtype, a.K),
            "gpu_util_window": "the timed sessions back to back, each verifier's verdict included (gpu_util_job: the whole GPU-held job)",
            "gpu_util_job": out.get("gpu_util_job"), "gpu_mem_peak_gib": out.get("gpu_mem_peak_gib"),
            "cpu_slice_others": (out.get("slice_cores_steady") or {}).get("others"), "why": out.get("why") or g.get("why"),
            # the point among its GPU session's, the session's one preflight check it cites, and its GPU-held seconds (its own sweep,
            # its share of the session's, the session's)
            "point": out["point"], "preflight": out["preflight"], "gpu_held_s_point": out.get("gpu_held_s_point"),
            "gpu_held_s_amortized": out.get("gpu_held_s_amortized"), "gpu_held_s_session": out.get("gpu_held_s_session")}


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


SESSION_MEDIANS = (("gpu_held_s_amortized", "s"), ("gpu_held_s_point", "s"), ("t_session_s_per_vu", "s"), ("overhead", "ratio"),
                   ("overhead_prove_only", "ratio"), ("throughput_vu_per_s", "1/s"), ("gpu_util", "fraction"),
                   ("verify_s_per_statement", "s"))


def session(a) -> None:
    """A GPU session's declared outputs (outputs.json): its one preflight check's record (`preflight`), then each point's
    hillclimb record, which cites it (refs.preflight). A session of several points also declares each point's contract result
    (`result-<i>`, a bench-point/v1: the tables read bench-result/v1 only) and writes result.json as the session's own
    (hillclimb-session/v0): passed only when its preflight check and every one of its points passed. A session of no points
    (the preflight check alone) declares the check's record and exits 1 unless the check passed."""
    R, n = Path(a.dir), a.points
    pre = "out/preflight/classes/results.jsonl"
    outs = [{"name": "preflight", "kind": "preflight-record/v0", "file": pre,
             "meta": {"campaign": "proofs-hillclimb", "what": "the GPU session's preflight check (byte identity at B = 16: "
                      + ", ".join(GS.PREFLIGHT_CASES) + "), which every point of the session cites"}}] if (R / pre).exists() else []
    cite = {"preflight": "@preflight"} if outs else {}
    for i in range(n):
        meta = {"campaign": "proofs-hillclimb"} | ({} if n == 1 else {"point": i, "points": n})
        refs = dict(cite)
        if n > 1 and (R / f"out/p{i}/result.json").exists():
            outs.append({"name": f"result-{i}", "kind": "bench-point/v1", "file": f"out/p{i}/result.json", "meta": meta, "refs": cite})
            refs["result"] = f"@result-{i}"
        hc = "hillclimb.json" if n == 1 else f"hillclimb-{i}.json"
        if (R / hc).exists():
            outs.append({"name": Path(hc).stem, "kind": "hillclimb-point/v0", "file": hc, "meta": meta, "refs": refs})
    if outs:
        (R / "outputs.json").write_text(json.dumps({"schema": "research/outputs/v0.1", "outputs": outs}, indent=1) + "\n")
    if n == 1:
        return
    g = GS.preflight(R / pre, shapes_needed=1)
    if n == 0:
        print(json.dumps({"points": 0, "preflight": g["status"]}))
        raise SystemExit(0 if g["status"] == "passed" else 1)
    res = [_load(R / f"out/p{i}/result.json") for i in range(n)]
    pts = [_load(R / f"hillclimb-{i}.json") for i in range(n)]
    status = [((r or {}).get("validation") or {}).get("status") or "did not run" for r in res]
    why = [f"point {i}: {s}" + (f" ({p['why']})" if s != "passed" and (p or {}).get("why") else "") for i, (s, p) in enumerate(zip(status, pts))]
    ok = g["status"] == "passed" and all(s == "passed" for s in status)
    ran = [p for p in pts if p]
    ms = [{"name": "hill.points", "value": n, "unit": "count"}, {"name": "hill.points_passed", "value": status.count("passed"), "unit": "count"},
          *([{"name": "hill.gpu_held_s_session", "value": ran[0]["gpu_held_s_session"], "unit": "s"}]
            if ran and ran[0].get("gpu_held_s_session") is not None else []),
          *[{"name": f"hill.median_{k}", "value": v, "unit": u} for k, u in SESSION_MEDIANS
            if (v := _median([p.get(k) for p in ran])) is not None]]
    fp = next((r.get("workload_fingerprint") for r in res if r and r.get("workload_fingerprint")), None)
    doc = {"schema": "research/result/v0.1", "kind": "hillclimb-session/v0",
           "run_id": os.environ.get("RESEARCH_RUN_ID") or next((p["run_id"] for p in ran if p.get("run_id")), "local"),
           "workload_fingerprint": {**(fp or {}), "points": n},
           "validation": {"status": "passed" if ok else "failed", "checker": f"the session's preflight check ({g['status']}) and its "
                          f"{n} points' byte identity", "detail": ("; ".join([f"preflight check: {g['status']}"] + why))[:900],
                          "evidence": [pre]},
           "measurements": ms,
           "measurement_files": [f for i in range(n) for f in (f"out/p{i}/metrics.json", f"out/p{i}/results.jsonl")],
           "artifacts": [f for i in range(n) for f in (f"hillclimb-{i}.json", f"out/p{i}/result.json")]}
    tmp = R / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    os.replace(tmp, R / "result.json")
    print(json.dumps({"points": n, "passed": status.count("passed"), "preflight": g["status"]}), "passed" if ok else "failed")


def rollup(a) -> None:
    """Append points to their subcircuit's roll-up (verity/hillclimb-rollup/v0), the whole file written beside it and renamed
    over it; strings cut at 200 characters and the commit at 12. Every point names its run (and its index among the run's
    points: a GPU session's run has several), and a point already there is not appended twice. With --attempt, every point the
    attempt published, each citing its art ids (`arts`, `_attempt_points`); a point already there without them gets them."""
    pts = [json.loads(Path(a.point).read_text())] if a.point else _attempt_points(a.attempt, _research(), a.run_dir)
    f = Path(a.file)
    doc = json.loads(f.read_text())
    index = lambda p: (p.get("run_id"), (p.get("point") or {}).get("index", 0))  # noqa: E731
    cut = lambda v: v[:200] if isinstance(v, str) else [cut(x) for x in v] if isinstance(v, list) else {k: cut(x) for k, x in v.items()} if isinstance(v, dict) else v  # noqa: E731
    added = cited = 0
    for pt in pts:
        if doc.get("schema") != ROLLUP_SCHEMA or pt.get("schema") != POINT_SCHEMA or doc["subcircuit"]["id"] != pt["subcircuit"]["id"]:
            raise SystemExit(f"{f}: not the roll-up of {pt.get('subcircuit', {}).get('id')}")
        if not pt.get("run_id") or not pt.get("commit"):
            raise SystemExit(f"{a.point or a.attempt}: a point names its run and commit")
        old = next((p for p in doc["points"] if index(p) == index(pt)), None)
        if old is not None:
            if pt.get("arts") and not old.get("arts"):
                old["arts"] = pt["arts"]
                cited += 1
            else:
                print(f"{pt['run_id']} point {index(pt)[1]} is already in {f}")
            continue
        pt = cut(pt)
        pt["commit"] = (pt.get("commit") or "")[:12]
        doc["points"].append(pt)
        added += 1
    if not (added or cited):
        return
    doc["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc) + "\n")
    os.chmod(tmp, 0o664)
    os.replace(tmp, f)
    print(f"{f}: {len(doc['points'])} points ({added} added, {cited} now citing their arts)")


def _research() -> list[str]:
    """The `research` command: RESEARCH_BIN, split as a shell would (node 1 has none on its PATH, only `uv run ... -m research`)."""
    import shlex

    return shlex.split(os.environ.get("RESEARCH_BIN", "research"))


def _show(research: list[str], what: str) -> dict:
    r = subprocess.run([*research, "data", "show", what, "--json"], capture_output=True, text=True, check=False)
    if r.returncode:
        raise SystemExit(f"{what}: not in the store ({(r.stderr or r.stdout).strip()[:200]})")
    return json.loads(r.stdout)


def _attempt_points(run: str, research: list[str], run_dir: str | None = None) -> list[dict]:
    """The hillclimb points attempt `run` published (`research data show`: one, or a GPU session's one per point), in point
    order, each with the art ids it cites (`arts`): its own record's (`point`), its attempt's result.json (`session`: a GPU
    session's hillclimb-session/v0, or a one-point attempt's contract result) and its preflight check's record (`preflight`).
    Each point is read from `run_dir` when given (its outputs.json's file, whose sha256 must be its art's payload's), else
    fetched from the store."""
    outs = _show(research, run).get("outputs") or {}
    names = [n for n in outs if re.fullmatch(r"hillclimb(-\d+)?", n)]
    if not names:
        raise SystemExit(f"{run}: no hillclimb point among its outputs")
    if "result" not in outs:
        raise SystemExit(f"{run}: no result.json among its outputs, so its points have no session to cite")
    files = {o["name"]: o["file"] for o in json.loads((Path(run_dir) / "outputs.json").read_text())["outputs"]} if run_dir else {}
    import tempfile

    pts = []
    with tempfile.TemporaryDirectory() as d:
        for n in names:
            if n in files:
                b = (Path(run_dir) / files[n]).read_bytes()
                if hashlib.sha256(b).hexdigest() != _show(research, outs[n])["manifest"]["payload"]["sha256"]:
                    raise SystemExit(f"{run_dir}/{files[n]}: not the bytes of {outs[n]}")
                pt = json.loads(b)
            else:
                to = Path(d) / n
                to.mkdir()
                subprocess.run([*research, "data", "fetch", outs[n], "--to", str(to)], capture_output=True, text=True, check=True)
                pt = json.loads(next(to.rglob("*.json")).read_text())
            if pt.get("run_id") != run:
                raise SystemExit(f"{run}: its output {n} ({outs[n]}) is run {pt.get('run_id')}'s point")
            pts.append(pt | {"arts": {"point": outs[n], "session": outs["result"]} | ({"preflight": outs["preflight"]} if "preflight" in outs else {})})
    return sorted(pts, key=lambda pt: (pt.get("point") or {}).get("index", 0))


def _points(src: str, research: list[str]) -> list[dict]:
    """hillclimb points from a roll-up or point file, else those the attempt `src` published (`_attempt_points`)."""
    p = Path(src)
    if p.exists():
        doc = json.loads(p.read_text())
        return doc["points"] if doc.get("schema") == ROLLUP_SCHEMA else [doc]
    return _attempt_points(src, research)


def plot(a) -> None:
    """Per subcircuit, the console's three panels over the steps: overhead (log, best-so-far over byte-identical points) with
    the prove-only overhead beside it, throughput, and GPU utilization with GPU-held and prove-only seconds per VU on their own
    axis; a point that fails its preflight check is hollow. Only the points under the last point's coins and session are drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    research = _research()
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
                "sources": a.points, "runs": [p.get("run_id") for p in pts],
                "questions": sorted({p.get("question") for p in pts if p.get("question")})}
        r = subprocess.run([*research, "data", "put", "--kind", "plot/v1", "--tree", str(out), "--meta", json.dumps(meta), "--preserve"],
                           capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode:
            raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd_", required=True)
    d = sub.add_parser("defs")
    d.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    d.add_argument("--K", type=int, required=True)
    d.add_argument("--n", type=int, default=None)
    d.add_argument("--out", required=True)
    f = sub.add_parser("fill")
    f.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    f.add_argument("--K", type=int, required=True)
    f.add_argument("--preflight", "--gate", dest="preflight", required=True)    # --gate: its old name
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
    m.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    m.add_argument("--K", type=int, required=True)
    m.add_argument("--preflight", "--gate", dest="preflight")    # --gate: its old name
    m.add_argument("--sweep", required=True)
    m.add_argument("--stamps", required=True)
    m.add_argument("--warm", type=int, default=1, help="the warm sessions the prove process ran (74-gemm-hill.sh WARM)")
    m.add_argument("--runs", type=int, default=3, help="its timed sessions (RUNS)")
    m.add_argument("--dmon", required=True)
    m.add_argument("--phases")
    m.add_argument("--others")
    m.add_argument("--slice")
    m.add_argument("--gpu-csv")
    m.add_argument("--census", default="benchmarks/census/hardware.json")
    m.add_argument("--peak-id", default=None, help="the census line (default: the dtype's, DTYPES)")
    m.add_argument("--root", default=".")
    m.add_argument("--binary-sha", default=None)
    m.add_argument("--step", type=int, default=0)
    m.add_argument("--label", default="baseline")
    m.add_argument("--question", default=None, help="the research question the job answers (default: the step's, question())")
    m.add_argument("--node", default="vy-nebius-1")
    m.add_argument("--point", type=int, default=0)
    m.add_argument("--points", type=int, default=1)
    m.add_argument("--preflight-key", default=None)
    m.add_argument("--point-key", default=None)
    m.add_argument("--out", required=True)
    m.add_argument("--result", required=True)
    m.add_argument("--hillclimb", default=None)
    k = sub.add_parser("key")
    k.add_argument("--binary-sha", required=True)
    k.add_argument("--out", required=True)
    se = sub.add_parser("session")
    se.add_argument("--dir", required=True)
    se.add_argument("--points", type=int, default=1)
    r = sub.add_parser("rollup")
    rs = r.add_mutually_exclusive_group(required=True)
    rs.add_argument("--point", help="one point's hillclimb record, which cites no art")
    rs.add_argument("--attempt", help="a run id in the store: every point it published, each citing its arts")
    r.add_argument("--run-dir", help="with --attempt: the run's directory, to read its points from (checked against their arts)")
    r.add_argument("--file", required=True)
    p = sub.add_parser("plot")
    p.add_argument("points", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--put", action="store_true")
    a = ap.parse_args()
    {"defs": defs, "fill": fill, "dmon": dmon, "cores": cores, "stamp": stamp, "metrics": metrics, "key": key, "session": session,
     "rollup": rollup, "plot": plot}[a.cmd_](a)


if __name__ == "__main__":
    main()
