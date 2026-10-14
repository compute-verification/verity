"""A TP row's Build sizes its derive pool and its per-rank compose pool from memory, as the single-GPU Build sizes its derives
(`row_records.build_jobs_auto` under ``--build-ram-budget-gb``), starts the request derives first, keeps an explicit
``--build-jobs``, and writes the same Build at any width.

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
            if args[0] == "build":
                os.makedirs(out)
                Path(out, "result.json").write_text(json.dumps({"ok": True, "program": {"digest": _digest(args)}}))
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
