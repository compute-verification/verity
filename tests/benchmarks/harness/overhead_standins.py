"""`verity bench overhead`'s tests' stand-ins: timings from a soft roofline, the node's programs (bench.py, nsys and
verify.py), the scheme, and a machine that runs the door on a real store with a directory remote."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
from verity_bench import kit, overhead_node
from verity_bench import overhead as O
from verity_bench import overhead_records as R

REPO = Path(__file__).resolve().parents[3]
H200 = O.CARDS["h200"]
LINEARS = O.MODELS[O.KERNEL_MODEL]
UUID = "GPU-11111111-2222-3333-4444-555555555555"
ARM = "pearl_plus"
SHA = "c" * 40
#: the arm's stand-in cubin, and the art a node-only run names it by
CUBIN = b"stand-in cubin"
CUBIN_ART = "art:" + "a" * 64
#: the gpu-lease wait the door puts before its node job, as the run's attempt records it
LEASE = kit.lease(1, 60)
#: the kit's stamp of the node the stand-in run holds its lease on
STAMP = {"card": "NVIDIA H200", "gpus": [{"uuid": UUID, "sm_mhz": 1980, "mem_mhz": 3201}], "driver": "570.1", "cuda": "13.0",
         "node": "node", "commit": SHA}


@pytest.fixture(autouse=True)
def stamped(monkeypatch):
    """The node side's result is written under the kit's host stamp, which asks nvidia-smi and git: a stand-in host here."""

    monkeypatch.setattr(kit, "host_stamp", lambda: dict(STAMP))


# ---- stand-in timings ---------------------------------------------------------------------------------------------------------

#: each scaling's latency (ms), peak (FLOP/ms) and bandwidth (bytes/ms) on a stand-in H200
ROOF = {"fp8-e4m3": (0.004, 1.6e12, 4.4e9), "fp8-e4m3-rowwise": (0.005, 1.55e12, 4.4e9), "fp8-blk128": (0.006, 1.4e12, 4.2e9)}


def roof_ms(fam: str, cand: str, m: int, n: int, k: int) -> float:
    """A soft roofline: each linear's time is its launch latency and the root of the sum of squares of its traffic's time
    and its FLOP's (no kernel reaches its peak at the ridge); vLLM's CUTLASS is a little slower than cuBLASLt below
    m = 256 and a little faster above. Each kernel's time over m falls as m grows, so throughput only rises."""

    lat, peak, bw = ROOF[fam]
    t = lat + math.hypot((n * k + m * k + 2 * m * n) / bw, 2 * m * n * k / peak)
    return t * ({"cublaslt": 1.0, "cutlass-vllm": 1.05 if m < 256 else 0.97}[cand])


def brute_force(ms_of=roof_ms) -> O.Points:
    """The rule over every m the sweep could sample: the powers of two below GRID and every multiple of GRID."""

    ms = [m for m in O.powers() if m < O.GRID] + list(range(O.GRID, O.M_MAX + 1, O.GRID))
    return O.rule({m: sum(min(ms_of(f, c, m, n, k) for f in O.FP8 for c in ("cublaslt", "cutlass-vllm"))
                          for n, k in LINEARS.values()) for m in ms})


def between(argv: list[str], flag: str) -> list[str]:
    i = argv.index(flag) + 1
    j = next((j for j in range(i, len(argv)) if argv[j].startswith("--")), len(argv))
    return argv[i:j]


def mnk(label: str) -> tuple[int, int, int]:
    m, n, k = map(int, re.fullmatch(r"m(\d+)-n(\d+)-k(\d+)", label).groups())
    return m, n, k


#: the stand-in run's entries of each linear: (key kind, candidate, SM-time factor, wall-time factor). The fastest stock
#: kernel by wall time isn't the least SM-time one: k_lean holds fewer SMs for longer.
RUN_ENTRIES = (("baseline", "k_fast", 1.0, 1.0), ("baseline", "k_lean", 0.8, 1.25), ("twin", "k_fast", 1.0, 1.0),
               ("arm", "hash", 1.5, 1.6), ("weight", "hash", 0.1, 0.1))
#: the host's enqueue as a share of a call's SM-time, at the card's weight
HOST = 0.1
#: the kernels a late capture range collected outside the traced items (the next shape's whole pipeline)
STRAY = 500


class Node:
    """The node's programs: bench.py's results from the roofline (a sweep) or from `RUN_ENTRIES` (a timed run, its traced
    pass beside it), nsys's reports, the priming range's (empty) and then one a shape's capture range, with each traced
    item's kernels at full SMs Active, and verify.py's verdicts. The arm loads the cubin the run was sent (`loads`), or
    else builds `built`, and the SASS gate passes the one it loads. `late`: the first shape's range opened after its items,
    so nsys collected from then to the next shape's stop (`STRAY` kernels besides its items)."""

    def __init__(self, tmp: Path, *, reps: int = 20, accept: bool = True, card: str = "NVIDIA H200", spread: float = 0.0,
                 loads: bool = True, built: bytes = b"a cubin built in the run", bridge_errors: dict | None = None,
                 late: bool = False):
        self.tmp, self.reps, self.accept, self.card, self.spread = tmp, reps, accept, card, spread
        self.loads, self.built, self.bridge_errors, self.late = loads, built, bridge_errors or {}, late
        self.calls: list[list[str]] = []
        self.env: dict[str, str] = {}
        self.reports: dict[str, list[tuple[int, int, bool]]] = {}
        self.strays: dict[str, int] = {}

    def __call__(self, argv: list[str], log: Path) -> int:
        self.calls.append(argv)
        log.write_text("")
        if argv[0] == "env":
            self.env, argv = dict([argv[1].split("=", 1)]), argv[2:]
        if argv[0] == "nsys":
            return self.profile(argv) if argv[1] == "profile" else self.export(argv)
        if argv[1].endswith("verify.py"):
            return self.verify(Path(argv[2]))
        return self.sweep(argv)

    def _result(self, argv: list[str], rows: list[dict], **more) -> dict:
        bridge = {"bridge": {"versions": {}, "errors": self.bridge_errors}} if "--bridge" in argv else {}
        return {"identity": {"card": {"name": self.card}}, "clocks": {"timed": {"sm_mhz": [1980, 1980]}}, "shapes": rows,
                **bridge, **more}

    def sweep(self, argv: list[str]) -> int:
        fams = between(argv, "--families")
        rows = []
        for lab in between(argv, "--shapes"):
            m, n, k = mnk(lab)
            rows.append({"label": lab, "m": m, "n": n, "k": k, "summary": {
                f"baseline/{lab}/{f}/{c}_{f}": {"graph": {"median": roof_ms(f, c, m, n, k)}, "items_with_bad_throttle": 0}
                for f in fams for c in ("cublaslt", "cutlass-vllm")}})
        Path(between(argv, "--out")[0]).write_text(json.dumps(self._result(argv, rows)))
        return 0

    def profile(self, argv: list[str]) -> int:
        prof = Path(argv[argv.index("-o") + 1])
        bench = argv[argv.index("-o") + 2:]
        fam, arm = between(bench, "--families")[0], "--arm" in bench
        items, traced, rows = [], [], []
        prime = prof.with_name(f"{prof.name}.1.nsys-rep")
        prime.write_text("a priming range's report")
        self.reports[str(prime)] = []
        for j, lab in enumerate(between(bench, "--shapes")):
            m, n, k = mnk(lab)
            base = m * n * k * 1e-12  # SM-seconds a call
            entries = [e for e in RUN_ENTRIES if arm or e[0] not in ("arm", "weight")]
            keys = {e: [e[0], lab, ARM if e[0] in ("arm", "weight") else fam, e[1]] for e in entries}
            graph: dict[str, list[float]] = {}
            report = prof.with_name(f"{prof.name}.{j + 2 - self.late}.nsys-rep")
            if not (self.late and j == 0):
                report.write_text("a report")
            if self.late and j == 1:
                self.strays[str(report)] = STRAY
            kernels = [] if self.late and j == 0 else self.reports.setdefault(str(report), [])
            for r in range(self.reps):
                drift = 1 + 0.01 * (r * 7 % 5)
                for e in entries:
                    sm_s = 2 * base * e[2] * drift
                    for method in ("graph", "burst"):
                        items.append({"key": keys[e], "method": method, "rep": r, "pos": 0, "burst": 2,
                                      "ms_per_call": base * e[3] * drift * 1e3, "burst_ms": 2 * base * e[3] * drift * 1e3,
                                      "cpu_s": HOST * sm_s / H200.cpu_sm_s_per_core_s if method == "burst" else 0.0})
                    graph.setdefault("/".join(keys[e]), []).append(items[-1]["ms_per_call"])
            for r in range(self.reps):
                drift = 1 + 0.01 * (r * 7 % 5)
                for e in entries:
                    jitter = 1 + self.spread * ((r + len(e[1])) % 3 - 1) * (e[0] == "arm")
                    sm_s = 2 * base * e[2] * drift * jitter
                    traced.append({"key": keys[e], "method": "burst", "rep": r, "pos": 0, "burst": 2,
                                   "ms_per_call": 0.0, "burst_ms": 0.0, "cpu_s": 1.0})
                    kernels.append((len(traced) - 1, round(sm_s * 1e9 / H200.sms), e[0] == "arm"))
            rows.append({"label": lab, "m": m, "n": n, "k": k, **({"arms": {ARM: {}}} if arm else {}),
                         "summary": {key: {"graph": {"median": sorted(xs)[len(xs) // 2]}} for key, xs in graph.items()}})
        more = {"items": items, "traced_items": traced, "sass_gate": {"ok": True, "binaries": []},
                "traced_prime": {"start_s": [0.2], "ready_s": 5.0, "ready": True}}
        if arm:
            sent = self.env.get(overhead_node.CUBIN_ENV) if self.loads else None
            path, cubin = (sent, Path(sent).read_bytes()) if sent else ("built.cubin", self.built)
            more["arms"] = {ARM: {"build": {"binaries": [path]}, "verifier": "verify.py"}}
            more["sass_gate"]["binaries"] = [{"path": path, "sha256": hashlib.sha256(cubin).hexdigest(), "ok": True,
                                              "bytes": len(cubin)}]
        Path(between(bench, "--out")[0]).write_text(json.dumps(self._result(bench, rows, **more)))
        return 0

    def export(self, argv: list[str]) -> int:
        """Each traced item of the report an NVTX range around one CUDA call (the arm's a driver call, as a JIT library's
        cuLaunchKernel) that launches one kernel of 16 blocks an SM on every SM, at SMs Active 100% for its duration."""

        db = sqlite3.connect(argv[argv.index("-o") + 1])
        if not self.reports[argv[-1]]:
            db.close()
            return 0
        db.executescript(SCHEMA)
        t = 1000
        for c in range(self.strays.get(argv[-1], 0)):
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, 1, 1, 1, 128, 1, 1, 32, 0, 0)",
                       (t, t + 10, 10 ** 6 + c))
        for i, dur, driver in self.reports[argv[-1]]:
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 7, ?, NULL)", (t, t + dur + 200, f"item {i}"))
            db.execute(f"INSERT INTO CUPTI_ACTIVITY_KIND_{'DRIVER' if driver else 'RUNTIME'} VALUES (?, ?, 7, ?)",
                       (t + 10, t + 20, i + 1))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, 1, 1, 128, 1, 1, 32, 0, 0)",
                       (t + 100, t + 100 + dur, i + 1, H200.sms * 16))
            db.executemany("INSERT INTO GPU_METRICS VALUES (?, 3, ?)", [(t + 100, 0.0), (t + 100 + dur, 100.0)])
            t += dur + 1000
        db.commit()
        db.close()
        return 0

    def verify(self, bench: Path) -> int:
        result = json.loads(bench.read_text())
        for row in result["shapes"]:
            for r in (row.get("arms") or {}).values():
                r["verified"] = {"verification": {"accepted": self.accept,
                                                  **({} if self.accept else {"why": "the verifier rejected it"})}}
        bench.write_text(json.dumps(result))
        return 0


SCHEMA = """
CREATE TABLE StringIds (id INTEGER, value TEXT);
CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, correlationId INTEGER, gridX INTEGER, gridY INTEGER,
    gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER, registersPerThread INTEGER, staticSharedMemory INTEGER,
    dynamicSharedMemory INTEGER);
CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT);
CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL);
INSERT INTO TARGET_INFO_GPU_METRICS VALUES (1, 'GR Active [Throughput %]'), (3, 'SMs Active [Throughput %]');
"""


@pytest.fixture
def scheme(tmp_path, monkeypatch):
    """pearl-plus with its arm and ε in a directory of the test's own: ε a stand-in function."""

    arm_dir = tmp_path / "arm"
    arm_dir.mkdir()
    (arm_dir / "overhead_eps_standin.py").write_text(
        "def epsilon(m, n, k, card):\n    return {'epsilon': 2.0 ** -40 * n / 8192, 'delta': 1e-9, 'basis': 'stand-in'}\n")
    sch = O.Scheme("pearl-plus", O.SCHEMES["pearl-plus"].family, "overhead_arm:PearlPlus", str(arm_dir),
                   "overhead_eps_standin:epsilon")
    monkeypatch.setitem(O.SCHEMES, "pearl-plus", sch)
    return sch


def node_points(tmp: Path, node: Node, run: str = "r20261104-020000-0001", model: str = O.KERNEL_MODEL) -> Path:
    out, run_dir = tmp / run / "out", tmp / run
    rc = overhead_node.main(["points", "--card", "h200", "--model", model, "--uuid", UUID, "--out", str(out),
                             "--run-dir", str(run_dir), "--run-id", run], run=node)
    assert rc == 0, (out / "points.json").read_text()
    return out / "points.json"


def node_run(tmp: Path, node: Node, points: Path, scheme: str, art: str | None = None,
             run: str = "r20261104-030000-0002", cubin: bytes | None = CUBIN, cubin_art: str = CUBIN_ART) -> tuple[int, Path]:
    """The node side's timed run, sent `cubin` (in the run's inputs/, as `--send` puts it) from the art `cubin_art`."""

    out, run_dir = tmp / run / "out", tmp / run
    scratch = tmp / "scratch" / run
    scratch.mkdir(parents=True)
    if cubin is not None:
        (run_dir / "inputs").mkdir(parents=True)
        (run_dir / "inputs" / "arm.cubin").write_bytes(cubin)
    rc = overhead_node.main(["run", "--card", "h200", "--scheme", scheme, "--points", str(points), "--point", "all",
                             "--uuid", UUID, "--out", str(out), "--run-dir", str(run_dir), "--run-id", run,
                             "--commit", SHA, "--scratch", str(scratch), *(["--points-art", art] if art else []),
                             *(["--cubin", "arm.cubin", "--cubin-art", cubin_art] if cubin is not None else [])], run=node)
    return rc, run_dir


# ---- the store side --------------------------------------------------------------------------------------------------------------


class Store:
    """A machine that runs the door: a local store with a directory remote, and each run's attempt and files on it."""

    def __init__(self, tmp: Path):
        from research.store.local import LocalStore

        self.tmp, self.remote = tmp, tmp / "remote"
        (self.remote / "attempts").mkdir(parents=True)
        self.config = tmp / "store.toml"
        self.config.write_text(f'[remote]\ntype = "fs"\npath = "{self.remote}"\n')
        self.store = LocalStore(tmp / "producer", config_path=self.config)
        cubin = tmp / "arm.cubin"
        cubin.write_bytes(CUBIN)
        self.cubin = self.store.put_artifact("cubin/v1", {"arm": ARM}, file=cubin)
        self.store.push(self.cubin)

    def fresh(self, name: str = "fresh"):
        from research.store.local import LocalStore

        return LocalStore(self.tmp / name, config_path=self.config)

    def ran(self, run: str, run_dir: Path, *, sub: str = "run", argv: list[str] | None = None, env: dict | None = None,
            validation: str = "passed", rc: int = 0, dirty: bool = False) -> None:
        """The run's attempt: by default the door's node job `sub` under the door's lease."""

        from research.store import canon

        files = self.store.put_artifact("run-files/v1", {}, tree=run_dir)
        self.store.push(files)
        start = f"{run[1:5]}-{run[5:7]}-{run[7:9]}T{run[10:12]}:{run[12:14]}:{run[14:16]}Z"
        if argv is None:
            argv = [*LEASE, "bash", R.NODE_JOB, sub, "--card", "h200"]
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": "gpu-lease", "version": None},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": argv,
            "params": {"argv": argv, **({"env": env} if env else {})}, "key_params": ["argv"], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": SHA, "dirty": dirty}, "closure": None,
            "execution": {"start_utc": start, "machine": "vy-dc-h200"}, "cost": None,
            "status": {"state": "done" if rc == 0 else "failed", "rc": rc, "validation": validation},
            "outputs": {"result": None, "run_files": files}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))


def stored_points(tmp: Path, s: Store, run: str, model: str = O.KERNEL_MODEL) -> str:
    node_points(tmp, Node(tmp), run, model)
    s.ran(run, tmp / run, sub="points", argv=[*LEASE, "bash", R.NODE_JOB, "points", "--card", "h200", "--model", model])
    return R.store_points(s.store, run)


def timed(tmp: Path, s: Store, art: str, scheme: str, run: str, node: Node | None = None, cubin: bytes = CUBIN,
          cubin_art: str | None = None, edit: Callable[[dict], None] | None = None) -> str:
    """A timed run of the door in the store; `edit` changes its `overhead.json` first, for a made-up record."""

    points = tmp / f"{run}-points.json"
    points.write_bytes(Path(s.store.fetch(art)).read_bytes())
    rc, run_dir = node_run(tmp, node or Node(tmp), points, scheme, art, run=run, cubin=cubin,
                           cubin_art=cubin_art or s.cubin)
    assert rc == 0
    if edit is not None:
        record = run_dir / "out" / "overhead.json"
        doc = json.loads(record.read_text())
        edit(doc)
        record.write_text(json.dumps(doc))
    s.ran(run, run_dir)
    return run
