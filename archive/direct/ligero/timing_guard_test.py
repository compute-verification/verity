"""The bench timing guard: the cgroup readers (v1 and v2), the verdict's three rules, the own-process exclusion, and ``guarded``
stamping the run's ``--out`` JSON.  No torch, no GPU.

    uv run pytest archive/direct/ligero/timing_guard_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from archive.direct.ligero import timing_guard as G  # noqa: E402


def _files(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def test_cpu_quota_and_stat_read_cgroup_v2_and_v1(tmp_path: Path) -> None:
    v2 = _files(tmp_path / "v2", {"cpu.max": "1020000 100000\n", "cpu.stat": "usage_usec 5\nnr_periods 40\nnr_throttled 30\n"})
    assert G.cpu_quota(v2) == pytest.approx(10.2) and G.cpu_stat(v2) == {"usage_usec": 5, "nr_periods": 40, "nr_throttled": 30}
    v1 = _files(tmp_path / "v1", {"cpu/cpu.cfs_quota_us": "1360000\n", "cpu/cpu.cfs_period_us": "100000\n",
                                  "cpu/cpu.stat": "nr_periods 7\nnr_throttled 1\nthrottled_time 99\n"})
    assert G.cpu_quota(v1) == pytest.approx(13.6) and G.cpu_stat(v1)["nr_throttled"] == 1
    assert G.cpu_quota(_files(tmp_path / "max", {"cpu.max": "max 100000\n"})) is None
    assert G.cpu_quota(_files(tmp_path / "unl", {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"})) is None
    assert G.cpu_quota(tmp_path / "none") is None and G.cpu_stat(tmp_path / "none") == {}


def _s(t: float, util: float = 90.0, periods: int = 0, throttled: int = 0, n_other: int | None = 0, load1: float = 2.0, pids=()) -> dict:
    return {"t": t, "util": util, "nr_periods": periods, "nr_throttled": throttled, "n_other": n_other, "load1": load1, "pids": list(pids)}


def test_a_clean_run_is_not_contended_and_the_record_says_what_was_seen() -> None:
    rep = G.verdict([_s(0, periods=0), _s(2, periods=20, throttled=1), _s(4, periods=40, throttled=2)], 10.2, 96)
    assert rep["contended"] is False and rep["reasons"] == []
    assert (rep["throttled"], rep["busy_periods"], rep["other_gpu_procs"], rep["load1_max"], rep["window_s"], rep["samples"]) == (
        0.05, 40, 0, 2.0, 4.0, 3)


def test_another_gpu_process_throttling_while_busy_or_a_saturated_host_make_it_contended() -> None:
    other = G.verdict([_s(0), _s(2, n_other=1, pids=["4711"]), _s(4)], 10.2, 96)
    assert other["contended"] and other["reasons"] == ["1 other GPU compute process (pids seen: 4711)"]
    # an instance build before the timed reps (GPU idle) throttles hard: not counted; the busy stretch after it is 50 % throttled
    samples = [_s(0, util=0), _s(2, util=0, periods=20, throttled=20), _s(4, util=0, periods=40, throttled=40),
               _s(6, periods=60, throttled=50), _s(8, periods=80, throttled=60)]
    assert G.verdict(samples[:3], 10.2, 96)["contended"] is False
    thr = G.verdict(samples, 10.2, 96)
    assert thr["throttled"] == 0.5 and thr["reasons"] == ["CPU throttled in 50% of the cgroup's periods while the GPU was busy (quota 10.2 cores)"]
    load = G.verdict([_s(0, load1=40.0), _s(2, load1=101.5)], None, 96)
    assert load["reasons"] == ["load average 101.5 > 96 cores"] and load["throttled"] is None     # no busy periods: no fraction


def test_a_sample_leaves_this_process_out_even_when_nvml_lists_pids_of_another_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    me = str(os.getpid())
    listing = {"--query-compute-apps=pid": [me, "4711"], "--query-gpu=utilization.gpu": ["87"]}
    monkeypatch.setattr(G, "_smi", lambda q: listing[q])
    monkeypatch.setattr(G, "cpu_stat", lambda: {"nr_periods": 5, "nr_throttled": 1, "throttled_time": 3})
    s = G.sample()
    assert (s["n_other"], s["pids"], s["util"], s["nr_periods"], s["nr_throttled"]) == (1, ["4711"], 87.0, 5, 1) and "throttled_time" not in s
    listing["--query-compute-apps=pid"] = ["90210"]                   # host pid: ours, once this process holds a context
    monkeypatch.setattr(G, "_own_context", lambda: True)
    assert G.sample()["n_other"] == 0
    monkeypatch.setattr(G, "_own_context", lambda: False)
    assert G.sample()["n_other"] == 1
    listing["--query-compute-apps=pid"] = None                        # no nvidia-smi: unknown, not clean
    assert G.sample()["n_other"] is None


def test_guarded_stamps_the_bench_out_json_and_leaves_other_commands_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture) -> None:
    calls = []
    monkeypatch.setattr(G, "sample", lambda: calls.append(1) or _s(len(calls), n_other=1 if len(calls) == 2 else 0, pids=["77"]))
    out = tmp_path / "bench_vu.json"

    @G.guarded
    def main(argv=None) -> int:
        if "--out" in argv:
            out.write_text(json.dumps({"validation": {"status": "passed"}}))
        return 0

    assert main(["--relation", "fp8-ada", "bench-vu", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["validation"] == {"status": "passed"} and doc["contention"]["contended"] is True
    assert doc["contention"]["reasons"] == ["1 other GPU compute process (pids seen: 77)"]
    assert "timing guard: CONTENDED: 1 other GPU compute process" in capsys.readouterr().err

    n = len(calls)
    assert main(["gate-vu", "--vus", "8"]) == 0 and len(calls) == n    # not a timed command: no guard
    monkeypatch.setenv("VY_TIMING_GUARD", "0")
    assert main(["bench", "--out", str(tmp_path / "off.json")]) == 0 and len(calls) == n
    monkeypatch.delenv("VY_TIMING_GUARD")
    stale = tmp_path / "stale.json"
    stale.write_text("{}")
    os.utime(stale, (1, 1))                                            # written before this run: not the run's output
    assert main(["bench-vu", f"--out={stale}"]) == 0 and json.loads(stale.read_text()) == {}


@pytest.mark.parametrize("files, want", [({"cpu.max": "1020000 100000"}, "10"), ({"cpu.max": "50000 100000"}, "1"),
                                         ({"cpu/cpu.cfs_quota_us": "1360000", "cpu/cpu.cfs_period_us": "100000"}, "13"),
                                         ({"cpu.max": "max 100000"}, "nproc"), ({"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}, "nproc"),
                                         ({}, "nproc")])
def test_the_bootstrap_sizes_thread_pools_to_the_same_quota(tmp_path: Path, files: dict[str, str], want: str) -> None:
    """pod_bootstrap.sh's cpu_quota (env.sh's OMP_NUM_THREADS etc.) is floor(cpu_quota()), nproc when unlimited."""
    import re
    import subprocess

    src = (Path(__file__).with_name("pod_bootstrap.sh")).read_text()
    fn = re.search(r"^cpu_quota\(\) \{.*?^\}", src, re.S | re.M).group(0)
    root = _files(tmp_path / "cg", {k: v + "\n" for k, v in files.items()})
    fake = _files(tmp_path / "bin", {"nproc": "#!/bin/sh\necho 96\n"})
    (fake / "nproc").chmod(0o755)
    got = subprocess.run(["bash", "-c", f"{fn}\ncpu_quota"], env={"CGROUP": str(root), "PATH": f"{fake}:{os.environ['PATH']}"},
                         capture_output=True, text=True, check=True).stdout.strip()
    assert got == ("96" if want == "nproc" else want)
    q = G.cpu_quota(root)
    assert want == "nproc" and q is None or int(want) == max(int(q), 1)
