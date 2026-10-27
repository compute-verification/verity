"""A TP row's Build sizes its derive pool and its per-rank compose pool from memory, as the single-GPU Build sizes its derives
(`row_records.build_jobs_auto` under ``--build-ram-budget-gb``), starts the request derives first, keeps an explicit
``--build-jobs``, and writes the same Build at any width.  A switched row's Build derives with ``predict emit`` over its trace.

The Build runs for real with its children faked (`_FakeChildren`): each derive and compose writes outputs named by its own command
line, so a width that changed what any child was asked to do would change the bytes compared."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline import row_tp
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_tp import TpRow

ROW = "qwen3-30b-a3b__bf16__l40s__tp2__b2__i1024__o128__mixed__greedy__bi-eager"
#: the 235B Build that motivated the bound: Qwen3-235B-A22B (94 decoder layers) at TP8, one shape LP256 T31
LAYERS, SHAPE = 94, (256, 31)
GiB = 1 << 30


class _FakeChildren:
    """`Row.run_to` for the Build's children, paths relative to the sweep dir: a derive writes ``result.json`` whose digest is its
    command line's, a compose a workload digest over its command line and its components' digests, the manifest a complete one."""

    def __init__(self, t: TpRow, root: str):
        self.t, self.root, self.lock, self.live, self.most, self.cmds = t, root, threading.Lock(), 0, 0, []

    def __call__(self, to, lf: str, cmd: list[str]) -> int:
        args = [a.replace(self.root, "<sweep>") for a in cmd[len(self.t.cli()):]]
        with self.lock:
            self.live += 1
            self.most = max(self.most, self.live)
            self.cmds.append(args)
        try:
            Path(lf).write_text("$ " + " ".join(args) + "\n")
            out = cmd[cmd.index("--out") + 1] if "--out" in cmd else ""
            if args[0] == "build" or args[:2] == ["predict", "emit"]:
                os.makedirs(out)
                aot = {"aot": json.loads(args[args.index("--aot") + 1])} if args[0] == "predict" else {}
                Path(out, "result.json").write_text(json.dumps({"ok": True, "program": {"digest": _digest(args)}, **aot}))
            elif args[0] == "global-program":
                parts = [json.loads(Path(cmd[i + 1], "result.json").read_text())["program"]["digest"] for i, a in enumerate(cmd) if a == "--derived"]
                os.makedirs(out, exist_ok=True)
                Path(out, "workload_program.json").write_text(json.dumps({"workload_digest": _digest(args + parts)}))
            elif args[:2] == ["manifest", "build-global"]:
                Path(out).write_text(json.dumps({"complete": True, "manifest_digest": _digest(args)}))
            return 0
        finally:
            with self.lock:
                self.live -= 1


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


class _Pools:
    """`row_tp.ThreadPoolExecutor`, recording each pool's width."""

    def __init__(self):
        self.widths: list[int] = []

    def __call__(self, max_workers: int) -> ThreadPoolExecutor:
        self.widths.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)


def _tp_row(tmp: Path, monkeypatch, *, world: int = 8, shapes=(SHAPE,), cpus: int = 64, headroom: int = 1 << 40,
            environ: dict[str, str] | None = None) -> tuple[TpRow, _FakeChildren, _Pools]:
    env = {k: v for k, v in os.environ.items() if k not in ("BUILD_JOBS", "BUILD_RAM_BUDGET_GB", "VERITY_TREE", "RESEARCH_RUN_DIR", "RESEARCH_RUN_ID")}
    env.update(environ or {})
    o = cli.parse("row", ["run", ROW, "QWEN3_30B_A3B", "Qwen/Qwen3-30B-A3B", "rev", "--sweep-dir", str(tmp), "--py", sys.executable,
                          "--native-collect-default", str(tmp / "nc_build")], env)
    t = TpRow(row=o.row, role=o.role, repo=o.repo, rev=o.rev, stages=o.stages, env=o.env, world=world, gpu_util=None,
              build=o.build, match=o.match, commit=o.commit)
    os.makedirs(t.d)
    ck = tmp / "ck"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"num_hidden_layers": LAYERS}))
    t.ck, t.B, t.MML, t.tgt_args, t.shapes, t.bdir = str(ck), 8, 288, [], sorted(shapes), f"{t.d}/build"
    t.requests = len(shapes)
    t.child_env = dict(os.environ)
    fake, pools = _FakeChildren(t, str(tmp)), _Pools()
    monkeypatch.setattr(t, "run_to", fake)
    monkeypatch.setattr(row_tp, "ThreadPoolExecutor", pools)
    monkeypatch.setattr(R, "_memory_headroom", lambda: headroom)
    monkeypatch.setattr(R, "_cpus", lambda: cpus)
    return t, fake, pools


def _marks(t: TpRow, stage: str) -> list[dict]:
    return [e for e in map(json.loads, open(t.timeline)) if e["stage"] == stage]


def _tree(d: str) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


def test_the_235b_rank_derive_is_planned_above_its_measured_peak():
    planned = R.derive_bytes(*SHAPE, LAYERS)
    assert 14.1 * GiB < planned < 2 * 14.1 * GiB and 20.8 * GiB < R.COMPOSE_PEAK_RATIO * planned
    assert R.compose_bytes([SHAPE], LAYERS) == int(R.COMPOSE_PEAK_RATIO * planned)


#: GLM-5.3 TP8's I256 mixed-arrivals request shapes: B8 (stochastic) and B32 (greedy)
GLM_LAYERS = 78
GLM_B8 = [(8, 28), (13, 31), (25, 6), (91, 2), (103, 12), (175, 10), (245, 22), (256, 1)]
GLM_B32 = [(103, 26), (105, 2), (117, 4), (11, 24), (13, 16), (149, 14), (151, 14), (15, 8), (171, 9), (177, 28), (17, 22), (19, 18),
           (201, 31), (21, 16), (225, 21), (237, 26), (23, 8), (256, 8), (25, 24), (29, 12), (31, 6), (37, 2), (41, 1), (55, 20),
           (59, 18), (61, 2), (65, 30), (69, 24), (81, 10), (8, 18), (91, 26), (9, 18)]


def test_a_compose_is_planned_above_its_measured_peak_as_its_requests_grow():
    """A compose holds every request's Program: GLM-5.3's B32 ranks compose at 128.8 GiB and B8's pass
    32.2 GiB, where 1.5x the largest derive plans ~22 GiB."""
    assert 128.8 * GiB < R.compose_bytes(GLM_B32, GLM_LAYERS) < 2 * 128.8 * GiB
    assert 32.2 * GiB < R.compose_bytes(GLM_B8, GLM_LAYERS) and R.COMPOSE_PEAK_RATIO * R.derive_bytes(245, 22, GLM_LAYERS) < 32.2 * GiB


def test_the_compose_pool_is_sized_by_every_requests_programs(tmp_path, monkeypatch):
    t, _fake, pools = _tp_row(tmp_path, monkeypatch, shapes=GLM_B8, headroom=250 * GiB)
    t.build()
    assert pools.widths[1] == int(0.8 * 250 * GiB) // R.compose_bytes(GLM_B8, LAYERS) == 3
    assert _marks(t, "build.compose_jobs")[0]["jobs"] == 3


def test_the_memory_headroom_is_the_jobs_own_cgroup_chain_and_the_hosts_available(tmp_path):
    """`research run --queue --mem-gb G` caps the job's own scope, not the mount root: read at the root, a Build plans its composes
    from the host's 1.5 TB available and its 250 GB scope OOM-kills them."""
    root, proc, meminfo = tmp_path / "cg", tmp_path / "cgroup", tmp_path / "meminfo"
    scope = root / "user.slice" / "app.slice" / "job.scope"
    scope.mkdir(parents=True)

    def limit(d: Path, mx: str, cur: int) -> None:
        (d / "memory.max").write_text(f"{mx}\n")
        (d / "memory.current").write_text(f"{cur}\n")

    limit(root / "user.slice", str(900 * GiB), 100 * GiB)
    limit(root / "user.slice" / "app.slice", "max", 50 * GiB)
    limit(scope, str(250 * GiB), 10 * GiB)
    meminfo.write_text(f"MemTotal:       {1700 * GiB // 1024} kB\nMemAvailable:   {1500 * GiB // 1024} kB\nHugePages_Total:       0\n")
    proc.write_text("0::/user.slice/app.slice/job.scope\n")
    head = lambda: R._memory_headroom(str(proc), str(root), str(meminfo))  # noqa: E731
    assert head() == 240 * GiB
    limit(scope, "max", 10 * GiB)
    assert head() == 800 * GiB                     # the slice's
    proc.write_text("0::/\n")
    assert head() == 1500 * GiB                    # no limit on the chain: the host's


@pytest.mark.parametrize("budget_gb, derives, composes", [(64, 2, 1), (96, 4, 2), (128, 5, 3), (512, 16, 8)])
def test_the_memory_budget_bounds_a_tp_builds_derive_and_compose_pools(tmp_path, monkeypatch, budget_gb, derives, composes):
    t, fake, pools = _tp_row(tmp_path, monkeypatch, environ={"BUILD_RAM_BUDGET_GB": str(budget_gb)})
    t.build()
    assert pools.widths == [derives, composes]
    (mark,), (cmark,) = _marks(t, "build.jobs"), _marks(t, "build.compose_jobs")
    assert (mark["jobs"], mark["requested"], mark["derives"]) == (derives, "auto", 16) and (cmark["jobs"], cmark["ranks"]) == (composes, 8)
    assert fake.most <= derives and len(fake.cmds) == 16 + 8 + 2      # 16 derives, 8 composes, the manifest and the call-boundary gate


def test_without_a_budget_ample_memory_keeps_the_full_width_and_the_cpus_still_bound_it(tmp_path, monkeypatch):
    t, _fake, pools = _tp_row(tmp_path / "ample", monkeypatch)
    t.build()
    assert pools.widths == [8 * (1 + 1), 8]
    t, _fake, pools = _tp_row(tmp_path / "four-cpus", monkeypatch, cpus=4)
    t.build()
    assert pools.widths == [4, 4]


@pytest.mark.parametrize("jobs", ["3", "32"])
def test_an_explicit_build_jobs_sizes_the_derives_and_the_budget_still_bounds_the_composes(tmp_path, monkeypatch, jobs):
    t, _fake, pools = _tp_row(tmp_path, monkeypatch, environ={"BUILD_JOBS": jobs, "BUILD_RAM_BUDGET_GB": "64"})
    t.build()
    assert pools.widths == [int(jobs), 1] and _marks(t, "build.jobs")[0]["requested"] == int(jobs)


def test_the_request_derives_start_first_the_longest_shape_first(tmp_path, monkeypatch):
    t, fake, _pools = _tp_row(tmp_path, monkeypatch, world=3, shapes=[(16, 3), (256, 31), (100, 200)], environ={"BUILD_JOBS": "1"})
    t.build()
    order = [(a[a.index("--rank") + 1], a[a.index("--wrapper") + 1], a[a.index("--LP") + 1] if "--LP" in a else "") for a in fake.cmds if a[0] == "build"]
    assert order == [(r, "request", lp) for lp in ("100", "256", "16") for r in "012"] + [(r, "step", "") for r in "012"]


def test_the_width_changes_nothing_the_build_writes(tmp_path, monkeypatch):
    one, fake1, pools1 = _tp_row(tmp_path / "one", monkeypatch, shapes=[SHAPE, (16, 3)], environ={"BUILD_JOBS": "1", "BUILD_RAM_BUDGET_GB": "1"})
    one.build()
    full, fake, pools = _tp_row(tmp_path / "full", monkeypatch, shapes=[SHAPE, (16, 3)])
    full.build()
    assert (pools1.widths, fake1.most, pools.widths) == ([1, 1], 1, [8 * 3, 8])
    assert sorted(fake1.cmds) == sorted(fake.cmds) and len(fake.cmds) == 8 * 3 + 8 + 2
    built = _tree(full.bdir)
    assert _tree(one.bdir) == built and len(built) == 8 * (3 * 3 + 2)    # per rank 3 derives (dir, .log, .rc), the compose's .log and output
    for f in ("build_summary.json", "manifest.json"):
        assert Path(one.d, f).read_bytes() == Path(full.d, f).read_bytes()


def test_a_switched_rows_build_is_predict_emit_over_its_trace_and_the_trace_construction_traces(tmp_path, monkeypatch):
    aot = {"comparison": "r20261018-001500-aaaa", "trace": "art:" + "5" * 64}
    monkeypatch.setattr(row_tp.CR, "aot_of", lambda row: aot if row == ROW else None)
    t, fake, _pools = _tp_row(tmp_path / "aot", monkeypatch, world=2, environ={"AOT_TRACE": str(tmp_path / "trace"), "BUILD_RELABEL": "on"})
    t.build()
    derives = [a for a in fake.cmds if a[0] in ("build", "predict")]
    assert len(derives) == 4 and all(a[:2] == ["predict", "emit"] and json.loads(a[a.index("--aot") + 1]) == aot for a in derives)
    assert sorted(a[a.index("--trace") + 1] for a in derives) == [f"{tmp_path}/trace/build/rank{r}/{p}" for r in "01"
                                                                  for p in ("build_request_LP256_T31", "build_step")]
    req = next(a for a in derives if "request" in a)
    assert req[req.index("--wrapper"):] == ["--wrapper", "request", "--LP", "256", "--T", "31"], "predict computes the launch batch"
    summary = json.loads(Path(t.d, "build_summary.json").read_text())
    assert {v["construction"] for rk in summary["ranks"].values() for p, v in rk.items() if p.startswith("build_")} == {"aot"}
    assert "construction=aot" in Path(t.d, "stages.txt").read_text()
    (mark,) = _marks(t, "build.construction")
    assert (mark["construction"], mark["comparison"], mark["trace"]) == ("aot", aot["comparison"], aot["trace"])
    traced, fake, _pools = _tp_row(tmp_path / "trace-construction", monkeypatch, world=2, environ={"BUILD_CONSTRUCTION": "trace"})
    traced.build()
    assert [a[0] for a in fake.cmds if a[0] in ("build", "predict")] == ["build"] * 4
    assert _marks(traced, "build.construction")[0]["construction"] == "trace"


class _Timed:
    """`Row.run_to` giving each child the time `seconds(args)` says it takes: past a nonzero limit it is killed before it writes
    anything (rc 124), as GNU ``timeout`` kills it; otherwise it runs as `_FakeChildren`.  Records each child's limit."""

    def __init__(self, fake: _FakeChildren, seconds):
        self.fake, self.seconds, self.limits = fake, seconds, []

    def __call__(self, to, lf: str, cmd: list[str]) -> int:
        args = cmd[len(self.fake.t.cli()):]
        self.limits.append((args[0], int(to)))
        if int(to) and self.seconds(args) > int(to):
            Path(lf).write_text(f"TIMEOUT after {to}s\n")
            return 124
        return self.fake(to, lf, cmd)


#: GLM-5.3 TP8 B8 I256's manifest at 2 processes: a 4,738 s word check, then 64 components of 356-409 s
B8_MANIFEST_S = 4738 + 64 * 409 // 2


@pytest.mark.parametrize("environ, limit, ok", [({}, R.manifest_timeout(64), True), ({"BUILD_TIMEOUT": "3600"}, R.manifest_timeout(64), True),
                                                ({"MANIFEST_TIMEOUT": "0"}, 0, True), ({"MANIFEST_TIMEOUT": "7200"}, 7200, False)])
def test_a_manifest_longer_than_one_derives_limit_is_not_killed_by_it(tmp_path, monkeypatch, environ, limit, ok):
    """B8's Build of record died with manifest_rc=124 after 4.7 h: its manifest, 8 ranks x 8 requests, ran under the 7200 s of one
    derive and needs about 2.5 times that.  Here each derive and compose takes 30 min and the manifest B8's 17,826 s.  The manifest
    has its own limit, scaled with its 64 component builds, and no --build-timeout moves it; --manifest-timeout sets it (0 = none),
    and one it overruns still fails the Build."""
    t, fake, _pools = _tp_row(tmp_path, monkeypatch, shapes=GLM_B8, environ=environ)
    timed = _Timed(fake, lambda args: B8_MANIFEST_S if args[0] == "manifest" else 1800)
    monkeypatch.setattr(t, "run_to", timed)
    derive = int(environ.get("BUILD_TIMEOUT", 7200))
    assert B8_MANIFEST_S > 7200 and R.manifest_timeout(64) > 5 * B8_MANIFEST_S
    if ok:
        t.build()
    else:
        with pytest.raises(Stop) as e:
            t.build()
        assert e.value.rc == 10
    assert {k: {to for kind, to in timed.limits if kind == k} for k in ("build", "global-program", "manifest")} == \
        {"build": {derive}, "global-program": {derive}, "manifest": {limit}}
    verdict = next(ln for ln in Path(t.d, "stages.txt").read_text().splitlines() if ln.startswith("build "))
    assert verdict.startswith("build PASS" if ok else "build FAIL") and f"manifest_rc={0 if ok else 124}" in verdict
    assert f"manifest: 64 component builds (8 ranks x 8 requests), limit {f'{limit}s' if limit else 'none'}" in Path(t.d, "row.log").read_text()


def test_auto_counts_the_derives_it_is_given_and_scales_each_ones_peak():
    per = R.derive_bytes(*SHAPE, LAYERS)
    assert R.build_jobs_auto([SHAPE], LAYERS, headroom=100 * per, cpus=64) == 2                       # the step's and the request's
    assert R.build_jobs_auto([SHAPE], LAYERS, derives=16, headroom=100 * per, cpus=64) == 16
    assert R.build_jobs_auto([SHAPE], LAYERS, derives=16, headroom=10 * per, cpus=64) == 8            # 0.8 x 10
    assert R.build_jobs_auto([SHAPE], LAYERS, derives=16, peak_ratio=2.0, headroom=10 * per, cpus=64) == 4
    # the model's layers unknown: 20 GiB (x the ratio) per job, 1..8 and at most the derives
    assert R.build_jobs_auto([SHAPE], None, derives=16, headroom=100 * GiB, cpus=64) == 5
    assert R.build_jobs_auto([SHAPE], None, derives=3, headroom=100 * GiB, cpus=64) == 3
    assert R.build_jobs_auto([SHAPE], None, derives=16, peak_ratio=1.5, headroom=100 * GiB, cpus=64) == 3
