"""`verity bench overhead` on CPU with stand-in timings: the operating points' rule and sweep, γ and its interval, busy
SM-time from an Nsight Systems export, the node side end to end with stand-ins for bench.py, nsys and verify.py, the store
side's versioned points and source check on a real store with a directory remote, and the command."""

from __future__ import annotations

import hashlib
import json
import math
import re
import runpy
import sqlite3
import sys
from pathlib import Path

import pytest

from verity_bench import nsys, overhead_node
from verity_bench import overhead as O
from verity_bench import overhead_records as R
from verity_bench.cli import main

REPO = Path(__file__).resolve().parents[3]
H200 = O.CARDS["h200"]
LINEARS = O.MODELS[O.KERNEL_MODEL]
UUID = "GPU-11111111-2222-3333-4444-555555555555"
ARM = "pearl_plus"
SHA = "c" * 40
#: the arm's stand-in cubin, and the art a node-only run names it by
CUBIN = b"stand-in cubin"
CUBIN_ART = "art:" + "a" * 64

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


class Node:
    """The node's programs: bench.py's results from the roofline (a sweep) or from `RUN_ENTRIES` (a timed run, its traced
    pass beside it), nsys's reports, one a shape's capture range, with each traced item's kernels at full SMs Active, and
    verify.py's verdicts. The arm's build record names `cubin_art`, and the SASS gate passes `cubin`."""

    def __init__(self, tmp: Path, *, reps: int = 20, accept: bool = True, card: str = "NVIDIA H200", spread: float = 0.0,
                 cubin_art: str | None = CUBIN_ART, cubin: bytes = CUBIN, bridge_errors: dict | None = None):
        self.tmp, self.reps, self.accept, self.card, self.spread = tmp, reps, accept, card, spread
        self.cubin_art, self.cubin, self.bridge_errors = cubin_art, cubin, bridge_errors or {}
        self.calls: list[list[str]] = []
        self.reports: dict[str, list[tuple[int, int, bool]]] = {}

    def __call__(self, argv: list[str], log: Path) -> int:
        self.calls.append(argv)
        log.write_text("")
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
        for j, lab in enumerate(between(bench, "--shapes")):
            m, n, k = mnk(lab)
            base = m * n * k * 1e-12  # SM-seconds a call
            entries = [e for e in RUN_ENTRIES if arm or e[0] not in ("arm", "weight")]
            keys = {e: [e[0], lab, ARM if e[0] in ("arm", "weight") else fam, e[1]] for e in entries}
            graph: dict[str, list[float]] = {}
            report = prof.with_name(f"{prof.name}.{j + 1}.nsys-rep")
            report.write_text("a report")
            kernels = self.reports[str(report)] = []
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
        more = {"items": items, "traced_items": traced, "sass_gate": {"ok": True, "binaries": []}}
        if arm:
            more["arms"] = {ARM: {"build": {"cubin_art": self.cubin_art, "binaries": ["arm.cubin"]}, "verifier": "verify.py"}}
            more["sass_gate"]["binaries"] = [{"path": "arm.cubin", "sha256": hashlib.sha256(self.cubin).hexdigest(),
                                              "ok": True, "bytes": len(self.cubin)}]
        Path(between(bench, "--out")[0]).write_text(json.dumps(self._result(bench, rows, **more)))
        return 0

    def export(self, argv: list[str]) -> int:
        """Each traced item of the report an NVTX range around one CUDA call (the arm's a driver call, as a JIT library's
        cuLaunchKernel) that launches one kernel of 16 blocks an SM on every SM, at SMs Active 100% for its duration."""

        db = sqlite3.connect(argv[argv.index("-o") + 1])
        db.executescript(SCHEMA)
        t = 1000
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
    sch = O.Scheme("pearl-plus", "fp8-blk128", "overhead_arm:PearlPlus", str(arm_dir), "overhead_eps_standin:epsilon")
    monkeypatch.setitem(O.SCHEMES, "pearl-plus", sch)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return sch


def node_points(tmp: Path, node: Node, run: str = "r20261104-020000-0001") -> Path:
    out, run_dir = tmp / run / "out", tmp / run
    rc = overhead_node.main(["points", "--card", "h200", "--uuid", UUID, "--out", str(out), "--run-dir", str(run_dir),
                             "--run-id", run], run=node)
    assert rc == 0, (out / "points.json").read_text()
    return out / "points.json"


def node_run(tmp: Path, node: Node, points: Path, scheme: str, art: str | None = None,
             run: str = "r20261104-030000-0002") -> tuple[int, Path]:
    out, run_dir = tmp / run / "out", tmp / run
    scratch = tmp / "scratch" / run
    scratch.mkdir(parents=True)
    rc = overhead_node.main(["run", "--card", "h200", "--scheme", scheme, "--points", str(points), "--point", "all",
                             "--uuid", UUID, "--out", str(out), "--run-dir", str(run_dir), "--run-id", run,
                             "--commit", SHA, "--scratch", str(scratch), *(["--points-art", art] if art else [])], run=node)
    return rc, run_dir


# ---- the points ---------------------------------------------------------------------------------------------------------------


def test_the_sweep_finds_the_rules_points_over_every_m_it_could_sample(tmp_path):
    """C and L from the powers of two and bisection on multiples of 16 are the rule's over every multiple of 16 from 16 to
    32,768: the sweep samples a few dozen m's where the grid has 2,048, and finds the same points."""

    times: dict[int, float] = {}
    while todo := O.next_ms(times):
        for m in todo:
            times[m] = sum(min(roof_ms(f, c, m, n, k) for f in O.FP8 for c in ("cublaslt", "cutlass-vllm"))
                           for n, k in LINEARS.values())
    got, want = O.final(times), brute_force()
    assert (got.c, got.l) == (want.c, want.l) and got.c % O.GRID == 0 and got.l % O.GRID == 0
    assert 16 < got.l < got.c < O.M_MAX and len(times) < 60
    flops = {m: O.layer_flop(O.KERNEL_MODEL, m) / t for m, t in times.items()}
    assert all(flops[m] >= O.C_SHARE * got.best_flop_per_ms for m in times if m >= got.c)
    assert all(times[m] <= (1 + O.L_SLACK) * times[1] for m in times if m <= got.l)


def test_c_needs_every_m_above_it_and_l_every_m_below_it():
    """A dip at m = 4,096 below C's share keeps C above it, though 2,048 passes; a spike at m = 32 past L's slack keeps L
    below it, though 64 is within it. The rule refuses a sweep without m = 1, and `final` an unfinished one."""

    flop, best = O.layer_flop(O.KERNEL_MODEL, 1), 1000 * O.layer_flop(O.KERNEL_MODEL, 1)
    share = {2048: 0.95, 4096: 0.5, 8192: 0.97, 32768: 1.0}
    times = {1: 1.0, 16: 1.02, 32: 1.5, 64: 1.05} | {m: m * flop / (s * best) for m, s in share.items()}
    p = O.rule(times)
    assert p.best_flop_per_ms == pytest.approx(best)
    assert p.c == 8192 and p.c_bracket == (4096, 8192)
    assert p.l == 16 and p.l_bracket == (16, 32)
    del times[4096]
    assert O.rule(times).c == 2048
    with pytest.raises(ValueError, match="sample m = 1"):
        O.rule({16: 1.0})
    with pytest.raises(ValueError, match="isn't finished"):
        O.final(times)


def test_sweep_times_takes_each_linears_fastest_kernel_of_every_scaling_and_refuses_a_gap():
    rows = [{"label": O.label(16, n, k), "m": 16, "n": n, "k": k,
             "summary": {f"baseline/{O.label(16, n, k)}/fp8-e4m3/a": {"graph": {"median": 2.0}},
                         f"baseline/{O.label(16, n, k)}/fp8-blk128/b": {"graph": {"median": 1.0}},
                         f"twin/{O.label(16, n, k)}/fp8-blk128/b": {"graph": {"median": 0.1}}}}
            for n, k in LINEARS.values()]
    times, chosen = O.sweep_times([{"shapes": rows}])
    assert times == {16: 4.0} and {v["kernel"] for v in chosen[16].values()} == {"fp8-blk128/b"}
    rows[2] = {**rows[2], "summary": {}, "skipped": "operands need 9 bytes; 1 free"}
    with pytest.raises(ValueError, match="gate_up at m = 16 .*operands need"):
        O.sweep_times([{"shapes": rows}])


# ---- γ -------------------------------------------------------------------------------------------------------------------------


def test_the_interval_is_the_medians_95_percent_order_statistics():
    assert O.interval(20) == (6, 15)
    assert O.interval(6) == (1, 6)
    with pytest.raises(ValueError, match="can't bound"):
        O.interval(5)
    g = O.paired([1.5 + 0.001 * i for i in range(20)], [1.0] * 20)
    assert g["gamma"] == pytest.approx(0.5095) and g["ci95"] == pytest.approx([0.505, 0.514]) and g["n"] == 20
    assert O.of_record_why(g) is None
    assert "fewer than 20" in O.of_record_why(O.paired([1.5] * 19, [1.0] * 19))
    assert "wider than ±5%" in O.of_record_why(O.paired([1.0 + 0.1 * i for i in range(20)], [1.0] * 20))
    with pytest.raises(ValueError, match="one for one"):
        O.paired([1.0], [1.0, 2.0])


def test_a_calls_cost_is_its_traced_busy_sm_time_and_the_hosts_enqueue_at_the_cards_weight():
    assert O.traced_cost({"burst": 4}, {"sm_ns": 8e9, "occupied_sm_ns": 1e10}) == pytest.approx(
        {"sm_s": 2.0, "occupied_sm_s": 2.5, "sm_ns": 8e9, "occupied_sm_ns": 1e10})
    assert O.traced_cost({"burst": 4}, {"sm_ns": None}) is None and O.traced_cost({"burst": 4}, None) is None
    assert O.host_cost({"burst": 4, "cpu_s": 2.0}, H200) == pytest.approx(0.48 * 2 / 4)


def test_periods_scale_gamma_to_the_datacenter_and_the_world():
    p = O.periods(0.25, 100_000, 600, 9.36e30, 1.9789e15, 250)
    assert p["share"] == pytest.approx(0.2)
    assert p["year"]["datacenter_gpu_h"] == pytest.approx(100_000 * 8760 * 0.2)
    assert p["year"]["world_flop"] == pytest.approx(0.25 * 9.36e30)
    assert p["week"]["world_gates"] == pytest.approx(0.25 * 9.36e30 * 168 / 8760 * 250)
    assert p["window"]["datacenter_gates"] == pytest.approx(100_000 * 600 * 0.2 * 1.9789e15 * 250)


# ---- busy SM-time ----------------------------------------------------------------------------------------------------------------


def test_busy_sm_time_integrates_sms_active_over_each_ranges_kernels_and_checks_it_against_their_occupancy(tmp_path):
    """Two ranges on one thread: a graph launch of two kernels (one of 32 blocks, which two SMs hold), and a JIT library's
    driver launch; a third range on another thread launched nothing. A range's text may sit in StringIds. SMs Active is
    integrated over the kernels' span between samples, and a range reading busier than its kernels occupy is flagged."""

    db_path = tmp_path / "prof.sqlite"
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    db.execute("INSERT INTO StringIds VALUES (5, 'item 1')")
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
                   [(0, 10_000, 1, "item 0", None), (20_000, 30_000, 1, None, 5), (0, 30_000, 2, "item 2", None),
                    (0, 5, 1, "other", None)])
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (100, 200, 1, 11)")
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (20100, 20200, 1, 12)")
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, 1, 1, 128, 1, 1, 32, 0, 0)",
                   [(1000, 5000, 11, H200.sms * 16), (5000, 6000, 11, 32), (21000, 29000, 12, H200.sms * 16)])
    # SMs Active: full over the first kernel, 2 of 132 SMs over the second, then 100% sampled every 2 µs over the third
    # but reading 110% of the SMs (a metric that over-reads)
    samples = [(1000, 0.0), (5000, 100.0), (6000, 100 * 2 / H200.sms), (21000, 0.0)]
    samples += [(t, 110.0) for t in range(23000, 29001, 2000)]
    db.executemany("INSERT INTO GPU_METRICS VALUES (?, 3, ?)", samples)
    db.commit()
    db.close()

    got = nsys.busy(db_path, H200, "item ")
    assert set(got) == {"item 0", "item 1", "item 2"}
    assert got["item 0"]["kernels"] == 2 and got["item 0"]["span_ns"] == 5000
    assert got["item 0"]["sm_ns"] == pytest.approx(H200.sms * 4000 + 2 * 1000)
    assert got["item 0"]["occupied_sm_ns"] == pytest.approx(H200.sms * 4000 + 2 * 1000) and got["item 0"]["ratio"] == 1
    assert got["item 1"]["sm_ns"] == pytest.approx(H200.sms * 8000 * 1.1)
    assert got["item 2"] == {"sm_ns": None, "occupied_sm_ns": None, "ratio": None, "kernels": 0}
    assert O.cross_check_why({"a": got["item 0"]["ratio"]}) is None
    assert "1 entries" in O.cross_check_why({"a": got["item 0"]["ratio"], "b": got["item 1"]["ratio"]})
    assert "(tail: 0.7)" in O.cross_check_why({"tail": 0.7, "a": 1.02}) and O.cross_check_why({"none": None})


def test_blocks_an_sm_holds_are_the_tightest_of_warps_registers_and_shared_memory():
    assert nsys.blocks_per_sm(nsys.Kernel(0, 1, 1, 128, 32, 0), H200) == 16
    assert nsys.blocks_per_sm(nsys.Kernel(0, 1, 1, 384, 168, 0), H200) == 1
    assert nsys.blocks_per_sm(nsys.Kernel(0, 1, 1, 128, 32, 200 * 1024), H200) == 1
    assert nsys.occupied_ns(nsys.Kernel(0, 10, 1000, 128, 32, 0), H200) == 10 * 63


# ---- the node side ----------------------------------------------------------------------------------------------------------------


def test_the_node_sweeps_for_the_points_in_rounds_of_bench_py(tmp_path):
    node = Node(tmp_path)
    points = json.loads(node_points(tmp_path, node).read_text())
    want = brute_force()
    assert (points["c"], points["l"]) == (want.c, want.l) and points["error"] is None
    assert points["schema"] == O.POINTS_SCHEMA and points["families"] == list(O.FP8)
    assert set(map(int, points["times_ms"])) >= set(O.powers()) | {want.c, want.l}
    for argv in node.calls:
        assert argv[1] == str(overhead_node.HARNESS / "bench.py") and "--bridge" in argv
        assert between(argv, "--expect-uuid") == [UUID] and between(argv, "--families") == list(O.FP8)
        assert len(between(argv, "--shapes")) <= overhead_node.CHUNK * len(LINEARS)
    assert len(points["rounds"]) == len(node.calls)
    result = json.loads((tmp_path / "r20261104-020000-0001" / "result.json").read_text())
    assert result["validation"]["status"] == "passed" and result["validation"]["evidence"] == ["out/points.json"]
    assert {m["name"]: m["value"] for m in result["measurements"]} == {"C": want.c, "L": want.l}


def test_the_node_refuses_a_sweep_without_every_library_the_card_chooses_among(tmp_path):
    out, run_dir = tmp_path / "out", tmp_path
    node = Node(tmp_path, bridge_errors={"deepgemm": "ImportError: no DeepGEMM"})
    rc = overhead_node.main(["points", "--card", "h200", "--uuid", UUID, "--out", str(out), "--run-dir", str(run_dir),
                             "--run-id", "r1"], run=node)
    assert rc == 1 and len(node.calls) == 1 and "--bridge" in node.calls[0]
    assert json.loads((out / "points.json").read_text())["error"] == "the H200's deepgemm didn't load (ImportError: no DeepGEMM)"
    assert O.libraries_why(H200, {}) == "bench.py ran without --bridge: the H200's cublaslt, cutlass, deepgemm are the stock " \
                                       "kernel's libraries"
    assert O.libraries_why(O.CARDS["rtx-pro-6000"], {}) is None


def test_the_node_refuses_a_sweep_on_another_card(tmp_path):
    out = tmp_path / "r" / "out"
    rc = overhead_node.main(["points", "--card", "h200", "--uuid", UUID, "--out", str(out), "--run-dir",
                             str(tmp_path / "r"), "--run-id", "r20261104-020000-0001"],
                            run=Node(tmp_path, card="NVIDIA RTX PRO 6000 Blackwell Server Edition"))
    doc = json.loads((out / "points.json").read_text())
    assert rc == 1 and "not the H200" in doc["error"] and doc.get("c") is None
    assert json.loads((tmp_path / "r" / "result.json").read_text())["validation"]["status"] == "failed"


def test_a_timed_run_gives_gamma_over_the_fastest_and_the_least_sm_stock_kernel_with_its_controls(tmp_path, scheme):
    """At each point the arm takes 1.5 times the busy SM-time of the fastest stock kernel by wall time and 1.875 times the
    one of least SM-time, and the host's enqueue, a tenth of each side's SM-time, counts above stock's: γ is 0.55 and
    0.9625; the weight side bounds it at 0.66, and the twin, the A/A control, is 0. Every row's verdict and ε stand beside
    γ, and nothing is refused."""

    points = node_points(tmp_path, Node(tmp_path))
    node = Node(tmp_path)
    rc, run_dir = node_run(tmp_path, node, points, "pearl-plus", "art:" + "b" * 64)
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 0 and doc["whys"] == [] and set(doc["points"]) == {"C", "L"}
    pts = json.loads(points.read_text())
    for name, p in doc["points"].items():
        assert p["m"] == pts[name.lower()] and p["whys"] == [] and p["missing"] == []
        assert p["gamma"]["gamma"] == pytest.approx(0.55, rel=1e-4) and p["gamma"]["n"] == 20
        assert p["gamma_least_sm"]["gamma"] == pytest.approx((1.5 + 0.15 - 0.08) / 0.8 - 1, rel=1e-4)
        assert p["gamma_weight_side_bound"]["gamma"] == pytest.approx(0.66, rel=1e-4)
        assert p["aa"]["gamma"] == pytest.approx(0.0, abs=1e-4)
        assert p["gamma_occupancy"]["gamma"] == pytest.approx(0.55, rel=1e-4)
        assert {v["stock_fastest"]["kernel"] for v in p["linears"].values()} == {"k_fast"}
        assert {v["stock_least_sm"]["kernel"] for v in p["linears"].values()} == {"k_lean"}
        assert all(v["accepted"] for v in p["verified"].values()) and len(p["verified"]) == len(LINEARS)
        assert p["epsilon"]["gate_up"] == {"epsilon": 2.0 ** -40 * 7, "delta": 1e-9, "basis": "stand-in"}
        assert p["gate"] == {"cap": O.GAMMA_CAP, "passes": True} and p["perplexity"]["value"] is None
        fast = p["linears"]["o"]["stock_fastest"]
        assert fast["sm_s"] == pytest.approx(pts[name.lower()] * 8192 * 8192 * 1e-12 * 1.02, rel=1e-3)
        assert fast["cpu_sm_s"] == pytest.approx(HOST * fast["sm_s"], rel=1e-3)
    assert doc["arm"] == {"name": ARM, "cubin_art": CUBIN_ART, "binaries": ["arm.cubin"], "verifier": "verify.py",
                          "gated": [{"path": "arm.cubin", "sha256": hashlib.sha256(CUBIN).hexdigest(), "ok": True}]}
    prof, *exports, verify = node.calls
    assert prof[:len(overhead_node.NSYS_PROFILE) + 1] == ["nsys", *overhead_node.NSYS_PROFILE]
    assert "--capture-range=cudaProfilerApi" in prof
    bench = prof[prof.index("-o") + 2:]
    assert {"--traced", "--twins", "--bridge"} <= set(bench) and between(bench, "--families") == ["fp8-blk128"]
    assert between(bench, "--methods") == ["graph", "burst"] and between(bench, "--arm") == [scheme.arm]
    assert len(exports) == 2 * len(LINEARS) == len(doc["reports"]) and all(e["exit"] == 0 for e in doc["reports"])
    assert all(e[1] == "export" and e[e.index("-o") + 1].startswith(str(tmp_path / "scratch")) for e in exports)
    assert verify[2:5] == [str(run_dir / "out" / "bench.json"), "--every-row", "--no-ledger"] and SHA in verify
    assert not list((run_dir / "out").glob("*.nsys-rep")) and not list((run_dir / "out").glob("*.sqlite"))
    result = json.loads((run_dir / "result.json").read_text())
    assert result["validation"]["status"] == "passed"
    assert {m["name"] for m in result["measurements"]} == {"gamma_C", "gamma_L"}


def test_stocks_gamma_is_the_aa_control_of_its_twinned_kernel_against_its_original(tmp_path):
    points = node_points(tmp_path, Node(tmp_path))
    node = Node(tmp_path)
    rc, run_dir = node_run(tmp_path, node, points, "stock", "art:" + "b" * 64)
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 0 and len(node.calls) == 1 + 2 * len(LINEARS) and "--arm" not in node.calls[0]
    for p in doc["points"].values():
        assert p["gamma"] == p["aa"] and p["gamma"]["gamma"] == pytest.approx(0.0, abs=1e-4) and p["whys"] == []
        assert p["epsilon"] == {"why": "stock has no ε: it is the A/A control"} and p["verified"] == {}


def test_a_timed_run_says_why_each_point_isnt_of_record(tmp_path, scheme):
    """Too few reps, a rejected row, a noisy arm: each point carries why, and the run still records its γ."""

    points = node_points(tmp_path, Node(tmp_path))
    rc, run_dir = node_run(tmp_path, Node(tmp_path, reps=12, accept=False, spread=0.2), points, "pearl-plus")
    doc = json.loads((run_dir / "out" / "overhead.json").read_text())
    assert rc == 0
    for p in doc["points"].values():
        whys = " ".join(p["whys"])
        assert "12 reps, fewer than 20" in whys and "didn't accept" in whys
    rc, run_dir = node_run(tmp_path, Node(tmp_path, spread=0.2), points, "pearl-plus", run="r20261104-030000-0003")
    assert all("wider than ±5%" in " ".join(p["whys"]) for p in
               json.loads((run_dir / "out" / "overhead.json").read_text())["points"].values())
    rc, run_dir = node_run(tmp_path, Node(tmp_path, cubin_art=None), points, "pearl-plus", run="r20261104-030000-0004")
    assert json.loads((run_dir / "out" / "overhead.json").read_text())["whys"] == [
        "the arm's build record names no `cubin_art`: a cubin built in the run is development only"]


def test_a_timed_run_refuses_points_that_arent_a_finished_sweep_of_its_card(tmp_path):
    bad = tmp_path / "points.json"
    bad.write_text(json.dumps({"schema": O.POINTS_SCHEMA, "card": "rtx-pro-6000", "c": 512, "l": 64,
                               "families": list(O.FP8), "model": O.KERNEL_MODEL}))
    node = Node(tmp_path)
    rc, run_dir = node_run(tmp_path, node, bad, "stock")
    assert rc == 1 and node.calls == []
    assert "isn't a finished" in json.loads((run_dir / "result.json").read_text())["validation"]["detail"]


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

    def ran(self, run: str, run_dir: Path, *, validation: str = "passed", rc: int = 0, dirty: bool = False) -> None:
        from research.store import canon

        files = self.store.put_artifact("run-files/v1", {}, tree=run_dir)
        self.store.push(files)
        start = f"{run[1:5]}-{run[5:7]}-{run[7:9]}T{run[10:12]}:{run[12:14]}:{run[14:16]}Z"
        attempt = {
            "schema": "research/attempt/v0.1", "id": run, "tool": {"name": "bash", "version": "0"},
            "derivation": "drv:" + "0" * 64, "reuse_key": None, "argv": [], "params": {}, "key_params": [], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": SHA, "dirty": dirty}, "closure": None,
            "execution": {"start_utc": start, "machine": "vy-dc-h200"}, "cost": None,
            "status": {"state": "done" if rc == 0 else "failed", "rc": rc, "validation": validation},
            "outputs": {"result": None, "run_files": files}, "telemetry": None,
        }
        (self.remote / "attempts" / f"{run}.json").write_bytes(canon.dumps(attempt))


def stored_points(tmp: Path, s: Store, run: str) -> str:
    node_points(tmp, Node(tmp), run)
    s.ran(run, tmp / run)
    return R.store_points(s.store, run)


def timed(tmp: Path, s: Store, art: str, scheme: str, run: str, node: Node | None = None) -> str:
    points = tmp / f"{run}-points.json"
    points.write_bytes(Path(s.store.fetch(art)).read_bytes())
    rc, run_dir = node_run(tmp, node or Node(tmp, cubin_art=s.cubin), points, scheme, art, run=run)
    assert rc == 0
    s.ran(run, run_dir)
    return run


def test_points_are_stored_versioned_and_the_newest_supersedes_the_last(tmp_path):
    s = Store(tmp_path)
    first = stored_points(tmp_path, s, "r20261104-020000-0001")
    assert R.current_points(s.store, "h200") == first
    meta = s.store.get_manifest(first).meta
    assert meta["version"] == 1 and meta["supersedes"] is None and (meta["c"], meta["l"]) == (brute_force().c,
                                                                                             brute_force().l)
    second = stored_points(tmp_path, s, "r20261104-020000-0002")
    assert second != first
    fresh = s.fresh()
    from research.store import labels_sync

    labels_sync.sync(fresh, push=False)
    assert R.current_points(fresh, "h200") == second and R.current_points(fresh, "rtx-pro-6000") is None
    assert [lab.value for lab in fresh.labels(first, key="superseded_by")] == [second]
    assert {"version": 2, "supersedes": first, "run": "r20261104-020000-0002"}.items() <= fresh.get_manifest(
        second).meta.items()

    node_points(tmp_path, Node(tmp_path), "r20261104-020000-0003")
    s.ran("r20261104-020000-0003", tmp_path / "r20261104-020000-0003", validation="failed", rc=1)
    with pytest.raises(ValueError, match="isn't a passing run"):
        R.store_points(s.store, "r20261104-020000-0003")


def test_the_table_shows_only_labelled_runs_of_record_and_refuses_the_rest_with_why(tmp_path, scheme):
    """The source check: a γ stands only from a labelled run of the door, timed at the points of record. A run that
    doesn't stand as a whole can't be labelled; a point that fails the door's rules stays in the table, refused; points
    superseded after the run refuse its γ; and a reader asking for a γ with no labelled run is told so."""

    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    run = timed(tmp_path, s, art, "pearl-plus", "r20261104-030000-0002")
    stock = timed(tmp_path, s, art, "stock", "r20261104-030000-0003")
    noisy = timed(tmp_path, s, art, "pearl-plus", "r20261104-030000-0004", Node(tmp_path, reps=12, cubin_art=s.cubin))

    with pytest.raises(ValueError, match="no labelled run"):
        R.gamma_of_record(s.store, "pearl-plus", "h200", "C")
    assert R.record_run(s.store, run) == [] and R.record_run(s.store, stock) == []
    rows = {(r["scheme"], r["point"]): r for r in R.gamma_rows(s.store)}
    assert set(rows) == {("pearl-plus", "C"), ("pearl-plus", "L"), ("stock", "C"), ("stock", "L")}
    assert all(r["of_record"] and r["why"] is None for r in rows.values())
    got = R.gamma_of_record(s.store, "pearl-plus", "h200", "L")
    assert got["gamma"]["gamma"] == pytest.approx(0.55, rel=1e-4) and got["run"] == run and got["points_art"] == art

    # a fresh machine renders the table from the store's remote alone, each period beside the window
    table = R.build(s.fresh(), root=REPO)
    assert all(r["of_record"] for r in table["rows"]) and table["census"]["gpus"] == 100_000
    by = {(r["scheme"], r["point"]): r for r in table["rows"]}
    assert "periods" not in by["stock", "C"] and by["pearl-plus", "C"]["periods"]["share"] == pytest.approx(0.55 / 1.55)
    text = R.render(table)
    assert "| pearl-plus | h200 | L |" in text and "| pearl-plus | h200 | C | 35.48% | year |" in text
    assert "Refused" not in text

    # a labelled run whose points fail the door's rules: shown refused with why, never as a figure
    assert all("fewer than 20" in w for w in R.record_run(s.store, noisy))
    with pytest.raises(ValueError, match="fewer than 20"):
        R.gamma_of_record(s.store, "pearl-plus", "h200", "C")
    text = R.render(R.build(s.fresh("second"), root=REPO))
    assert "Refused (not of record):" in text and "fewer than 20" in text and "| pearl-plus | h200 | L |" not in text

    # a run that doesn't stand as a whole can't be labelled: from a dirty tree, or timed at points of no stored art
    sent = tmp_path / "r20261104-030000-0002-points.json"
    dirty = "r20261104-030000-0005"
    s.ran(dirty, node_run(tmp_path, Node(tmp_path), sent, "pearl-plus", art, run=dirty)[1], dirty=True)
    with pytest.raises(ValueError, match="clean tree"):
        R.record_run(s.store, dirty)
    loose = "r20261104-030000-0006"
    s.ran(loose, node_run(tmp_path, Node(tmp_path), sent, "pearl-plus", run=loose)[1])
    with pytest.raises(ValueError, match="no stored art"):
        R.record_run(s.store, loose)

    # nor can one whose cubin art isn't the binary its SASS gate passed, or isn't in the store
    for i, node in enumerate((Node(tmp_path, cubin_art=s.cubin, cubin=b"another cubin"), Node(tmp_path))):
        other = timed(tmp_path, s, art, "pearl-plus", f"r20261104-030000-001{i}", node)
        with pytest.raises(ValueError, match="SASS gate passed" if i == 0 else "isn't in the store"):
            R.record_run(s.store, other)

    # new points supersede the ones the runs were timed at: none of their γ stands
    stored_points(tmp_path, s, "r20261104-020000-0007")
    third = s.fresh("third")
    later = R.build(third, root=REPO)
    assert not any(r["of_record"] for r in later["rows"])
    assert all("superseded" in r["why"] for r in later["rows"])
    with pytest.raises(ValueError, match="superseded"):
        R.gamma_of_record(third, "stock", "h200", "L")


# ---- the command ------------------------------------------------------------------------------------------------------------------


def test_the_command_launches_each_step_as_a_research_run_on_one_gpu_and_ships_the_points(tmp_path, monkeypatch, capsys):
    s = Store(tmp_path)
    art = stored_points(tmp_path, s, "r20261104-020000-0001")
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "producer"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(s.config))
    monkeypatch.delenv("VERITY_BENCH_VENV", raising=False)
    launched = []

    def execute(argv, root):
        argv = list(argv)
        sent = [Path(argv[i + 1]) for i, a in enumerate(argv) if a == "--send"]
        launched.append((argv, [p.read_bytes() for p in sent]))
        return 0

    monkeypatch.setattr(R, "_execute", execute)
    assert main(["overhead", "points", "--card", "h200"]) == 0
    assert main(["overhead", "run", "--scheme", "stock", "--point", "L", "--local-only"]) == 0
    (points, _), (run, sent) = launched
    for argv in (points, run):
        assert argv[:6] == ["research", "run", "--on", "vy-dc-h200", "--gpus", "1"] and "--require-result" in argv
        assert argv[argv.index("--cwd") + 1] == "source" and argv[argv.index("--") + 1:][:2] == ["bash", R.NODE_JOB]
    assert points[points.index("--") + 3:] == ["points", "--card", "h200", "--model", O.KERNEL_MODEL, "--precision", "fp8"]
    assert run[run.index("--") + 3:] == ["run", "--card", "h200", "--scheme", "stock", "--points", "points.json",
                                         "--points-art", art, "--point", "L"]
    assert sent == [Path(s.store.fetch(art)).read_bytes()] and run[run.index("--timeout") + 1] == "60m"
    with pytest.raises(SystemExit):
        main(["overhead", "points", "--card", "rtx-pro-6000"])
    assert "has no default node" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["overhead", "run", "--card", "h200", "--local-only"])
    assert "--scheme S, or --record RUN" in capsys.readouterr().err


def test_verity_bench_overhead_is_the_door_s_runner(tmp_path, monkeypatch, capsys):
    """`verity bench overhead` runs benchmarks/overhead/run.py, which is `verity-bench overhead`."""

    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "empty"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(Store(tmp_path).config))
    monkeypatch.setattr(sys, "argv", ["run.py", "table", "--json", "--local-only"])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(REPO / "benchmarks" / "overhead" / "run.py"), run_name="__main__")
    assert done.value.code == 0
    table = json.loads(capsys.readouterr().out)
    assert table["rows"] == [] and table["census"]["world_flop_per_year"] == 9.36e30
