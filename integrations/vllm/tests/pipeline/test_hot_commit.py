"""hot Commit worker: the key that decides what may share an engine, the client's cold fallback, the timeline
report's engine-start / reuse accounting.  Torch-free (the worker itself is exercised on a pod: cold vs hot vs hot roots)."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import hot as hc
from verity_vllm.pipeline import research_tools, source_identity
from verity_vllm.pipeline import timeline as tlm

CORE = Path("verity")


def _tree(tmp_path: Path) -> Path:
    """A repository shaped like the tree of record; returns its integration root (the job's cwd)."""
    repo = tmp_path / "repo"
    (repo / CORE / "ir").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'verity'\n")
    (repo / "uv.lock").write_text("")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\nout/\n")
    (repo / CORE / "__init__.py").write_text("")
    (repo / CORE / "ir" / "codec.py").write_text("def _spec_id(fn): return fn\n")
    for d in research_tools.CODE_DIRS:
        (repo / d).mkdir(parents=True, exist_ok=True)
        (repo / d / "__init__.py").touch()
    (repo / "tools" / "research").mkdir(parents=True)
    (repo / "tools" / "research" / "cli.py").write_text("print('r')\n")
    t = repo / "integrations" / "vllm"
    (t / "verity_vllm" / "pipeline").mkdir(parents=True)
    (t / "verity_vllm" / "__init__.py").write_text("")
    (t / "verity_vllm" / "pipeline" / "commit.py").write_text("print('x')\n")
    (t / "verity_vllm" / "pipeline" / "__pycache__").mkdir()
    (t / "verity_vllm" / "pipeline" / "__pycache__" / "a.pyc").write_bytes(b"\0")
    (t / "manifests").mkdir(parents=True)
    (t / "manifests" / "checkpoints.json").write_text('{"LLAMA32_1B": {"repo": "unsloth/Llama-3.2-1B"}}\n')
    (t / "EXPORT.json").write_text('{"commit": "abc"}\n')
    (t / "RELEASE.json").write_text('{"tag": "r16-int-9"}\n')
    (t / "wl.json").write_text(json.dumps({"case": "LLAMA32_1B", "model": "unsloth/Llama-3.2-1B", "engine": {"args": {}}, "requests": [{"prompt_token_ids": [1]}],
                                           "sampling": {"variant": "greedy"}}) + "\n")
    return t


def test_code_identity_is_the_closure_identity_of_the_repository(tmp_path):
    t = _tree(tmp_path)
    a = hc.code_identity(t)
    assert a["sha256"] == source_identity.code_identity(t.parents[1])["sha256"] == hc.code_identity(t.parents[1])["sha256"]
    assert a["commit"] == "abc" and a["release"] == "r16-int-9"
    (t / "verity_vllm" / "pipeline" / "__pycache__" / "a.pyc").write_bytes(b"\1\2")
    (t.parents[1] / "tools" / "research" / "cli.py").write_text("print('s')\n")
    assert hc.code_identity(t)["sha256"] == a["sha256"], "caches and files outside the import closure are not executed code"
    (t / "verity_vllm" / "pipeline" / "commit.py").write_text("print('y')\n")
    assert hc.code_identity(t)["sha256"] != a["sha256"], "an edited executing tree is a different key -- hot never runs stale code"


def test_editing_a_core_file_changes_the_key(tmp_path, monkeypatch):
    t = _tree(tmp_path)
    monkeypatch.chdir(t)
    argv = ["--case", "LLAMA32_1B", "--workload", "wl.json", "--out", "o"]
    k0, c0 = hc.job_key(argv)["key"], hc.code_identity(t)["sha256"]
    (t.parents[1] / CORE / "ir" / "codec.py").write_text("def _spec_id(fn): return (fn,)\n")
    k1, c1 = hc.job_key(argv)["key"], hc.code_identity(t)["sha256"]
    assert c1 != c0 and k1 != k0, "a live worker must not serve Commits with stale core code"
    (t.parents[1] / CORE / "commitments").mkdir()
    (t.parents[1] / CORE / "commitments" / "merkle.py").write_text("ROOT = 0\n")
    assert hc.job_key(argv)["key"] != k1, "core/commitments is part of the import closure"


def test_code_identity_refuses_a_tree_outside_a_repository(tmp_path):
    (tmp_path / "verity_vllm").mkdir()
    (tmp_path / "verity_vllm" / "x.py").write_text("")
    with pytest.raises(FileNotFoundError, match="not a repository root"):
        hc.code_identity(tmp_path)


def test_job_key_changes_with_engine_args_extension_and_env_but_not_per_job_fields(tmp_path, monkeypatch):
    t = _tree(tmp_path)
    monkeypatch.chdir(t)
    so = tmp_path / "tap.so"
    so.write_bytes(b"ELF0")
    for k in list(os.environ):
        if k.startswith(hc.ENV_PREFIXES):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VERITY_TIMELINE", "/x/a.jsonl")
    base = ["--case", "LLAMA32_1B", "--workload", "wl.json", "--max-num-seqs", "1", "--hidden-so", str(so), "--out", "o1", "--label", "row-a", "--pairs", "3"]
    k0 = hc.job_key(base, t)["key"]
    # per-job fields: out / label / pairs / timeline path
    monkeypatch.setenv("VERITY_TIMELINE", "/x/b.jsonl")
    assert hc.job_key([*base[:-6], "--out", "o2", "--label", "row-b", "--pairs", "1"], t)["key"] == k0
    # engine-shaping fields
    assert hc.job_key([*base[:5], "8", *base[6:]], t)["key"] != k0, "max_num_seqs shapes the engine"
    assert hc.job_key([*base, "--engine-arg", "max_num_batched_tokens=1024"], t)["key"] != k0
    so.write_bytes(b"ELF1")
    k_so = hc.job_key(base, t)["key"]
    assert k_so != k0, "a rebuilt native extension is a restart trigger"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    assert hc.job_key(base, t)["key"] != k_so, "engine-shaping environment is part of the key"
    monkeypatch.setenv("VERITY_FAULT", "skip_copy:1")
    kf = hc.job_key(base, t)
    assert "VERITY_FAULT" in kf["env"], "a deliberate fault never shares a worker with honest jobs"


def test_worker_info_and_stop_ignore_dead_workers(tmp_path):
    w = tmp_path / "root" / "k1"
    w.mkdir(parents=True)
    (w / "worker.json").write_text(json.dumps({"pid": 2**22 + 12345, "key": "k1"}))
    info = hc.worker_info(w)
    assert info is not None and info["alive"] is False
    assert hc.stop_workers(tmp_path / "root", log=lambda *a, **k: None) == []
    (w / "worker.json").write_text(json.dumps({"pid": os.getpid(), "key": "k1"}))
    (w / "exited.json").write_text("{}")
    assert hc.worker_info(w)["alive"] is False, "an exit record marks the worker dead even while the pid is reused"


def test_submit_without_spawn_runs_cold_with_the_reason_on_the_timeline(tmp_path, monkeypatch):
    t = _tree(tmp_path)
    monkeypatch.chdir(t)
    tl_path = tmp_path / "timeline.jsonl"
    monkeypatch.setenv("VERITY_TIMELINE", str(tl_path))
    calls = []

    def fake_call(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(hc.subprocess, "call", fake_call)
    rc = hc.submit(["--case", "LLAMA32_1B", "--workload", "wl.json", "--out", "o"], root=tmp_path / "hot", idle_s=1, timeout_s=None, spawn=False, log=lambda *a, **k: None)
    assert rc == 0 and len(calls) == 1
    cmd, env = calls[0]
    assert cmd[:4] == [sys.executable, "-m", "verity_vllm.pipeline.cli", "commit"] and cmd[-2:] == ["--out", "o"]
    assert env["VERITY_HOT_RESTART_REASON"].startswith("no live hot worker")
    ev = tlm.read(tl_path)
    assert [e["stage"] for e in ev] == ["hot.key", "hot.fallback_cold"]
    assert ev[1]["restart_reason"].startswith("no live hot worker") and ev[1]["hot"] is False


def test_timeline_report_counts_starts_reuses_and_reasons(tmp_path):
    p = tmp_path / "tl.jsonl"
    a = tlm.Timeline(p, proc="commit_delta", hot=True)
    a.start("engine.build")
    a.end("engine.build", engine_reused=False, restart_reason="no live hot worker for this key")
    a.event("engine.reuse", "mark", engine_reused=True)
    a.event("engine.reuse", "mark", engine_reused=True)
    b = tlm.Timeline(p, proc="commit_delta", hot=False)
    b.start("engine.build")
    b.end("engine.build", engine_reused=False, restart_reason="fresh process (one commit_delta per Commit stage)")
    rep = tlm.report(tlm.read(p))
    assert rep["engine"]["starts"] == 2 and rep["engine"]["reuses"] == 2
    assert rep["engine"]["restart_reasons"] == {"no live hot worker for this key": 1, "fresh process (one commit_delta per Commit stage)": 1}
    assert rep["engine"]["hot_values"] == [False, True]
    md = tlm.report_md(rep, "t")
    assert "engine starts **2**" in md and "engine reuses **2**" in md


def test_timeline_report_defaults_to_the_last_row_attempt_of_an_append_only_file(tmp_path):
    """row_pod appends every attempt to the same timeline.jsonl; the table is about the LAST attempt (one engine start, not the
    sum of every rerun's), `--all` folds everything, and a file without a row envelope is one attempt."""
    p = tmp_path / "tl.jsonl"
    for reason in ("fresh process (one commit_delta per Commit stage)", "no live hot worker for this key"):
        row = tlm.Timeline(p, proc="row_pod", hot=False)
        row.start("row")
        c = tlm.Timeline(p, proc="commit_delta", hot=reason.startswith("no live"))
        c.start("engine.build")
        c.end("engine.build", engine_reused=False, restart_reason=reason)
        row.end("row", rc=0)
    ev = tlm.read(p)
    last = tlm.report(ev)
    assert last["attempts_in_file"] == 2 and last["attempt"] == "last"
    assert last["engine"]["starts"] == 1 and last["engine"]["restart_reasons"] == {"no live hot worker for this key": 1}
    assert last["engine"]["hot_values"] == [False, True]      # row_pod's own events are hot=false; the python stage was hot
    assert "attempt last of 2 in the file" in tlm.report_md(last, "t")
    every = tlm.report(ev, attempt="all")
    assert every["engine"]["starts"] == 2 and every["attempts_in_file"] == 1 and "attempt" not in tlm.report_md(every, "t").split("\n")[2]
    single = tmp_path / "one.jsonl"
    tlm.Timeline(single, proc="commit_delta").event("engine.reuse", "mark", engine_reused=True)
    assert tlm.report(tlm.read(single))["attempts_in_file"] == 1


def test_collect_folds_timelines_and_derives_envelopes_from_stages_txt(tmp_path):
    sweep = tmp_path / "sweep"
    a = sweep / "row_a"                        # instrumented runner: timeline.jsonl
    a.mkdir(parents=True)
    (a / "stages.txt").write_text("release r16-sys-4 2026-09-20T05:26:20Z commit=a41ffe5ca\ncommit PASS 2026-09-20T05:28:30Z rc=0 wall=121s runs 6\n")
    t = tlm.Timeline(a / "timeline.jsonl", proc="commit_delta", hot=True)
    t.start("engine.build"); t.end("engine.build", engine_reused=False, restart_reason="no live hot worker for this key")
    t.start("validate.sampled_replay"); t.end("validate.sampled_replay")
    b = sweep / "row_b"                        # r16-int-1 runner: stages.txt + commit.log only; two attempts, the last one is reported
    b.mkdir()
    (b / "stages.txt").write_text("release r15-int-6b 2026-09-20T04:13:59Z commit=76b0ffe10\nbuild PASS 2026-09-20T04:16:01Z rc=0/0 wall=102s digest=x\n"
                                  "match PASS 2026-09-20T04:18:08Z rc=0/0/0 wall=127s verdict PASS\ncommit FAIL 2026-09-20T04:22:17Z rc=3 wall=235s\n"
                                  "release r16-int-1 2026-09-20T05:00:00Z commit=d5adc8d88\ncommit PASS 2026-09-20T05:04:00Z rc=0 wall=239s runs 6\n")
    (b / "commit.log").write_text("[delta] engine built in 13.4s (case X)\n[delta] weight registration (native): {'root': 'r', 'tensors': 163, 'bytes': 1, 'wall_s': 18.9, 'label': 'measured'}\n")
    (sweep / "not_a_row.txt").write_text("")
    col = tlm.collect([sweep, tmp_path / "missing"])
    assert col["n_rows"] == 2 and col["measured_rows"] == 1
    ra, rb = col["rows"]
    assert ra["measured"] and ra["engine"]["starts"] == 1 and ra["categories"]["startup"] >= 0
    assert not rb["measured"] and rb["stages"] == {"commit": {"s": 239.0, "verdict": "PASS"}} and rb["elapsed_s"] == 240.0
    assert rb["commit_log"] == {"engine_builds": [13.4], "register_weights_s": [18.9], "engine_starts": 1}
    assert col["engine_starts"] == 2 and col["engine_reuses"] == 0
    md = tlm.collect_md(col, "t")
    assert "row_b | stages.txt (derived)" in md and "engine starts **2**" in md


@pytest.mark.parametrize("argv,unsafe", [
    (["--out", "o"], False),
    (["--out", "o", "--arms", "a,b"], True),
    (["--out", "o", "--only-arm", "control"], True),
    (["--out", "o", "--only-arm", "instrumented", "--compiled-taps", "kernels"], True),
    (["--out", "o", "--only-arm", "instrumented", "--bounded-staging", "--replay-k", "460", "--replay-deferred"], False),   # a deferred config-run Commit
    (["--out", "o", "--compiled-taps", "on"], True),
    (["--out", "o", "--late-read", "qkv_proj"], True),
    (["--out", "o", "--value-check"], True),
    (["--out", "o", "--committer", "plugin:m:f"], True),
])
def test_hot_unsafe_flags_are_refused_by_commit_delta(argv, unsafe, monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841  commit_delta imports torch at module level
    from verity_vllm.pipeline import commit as cd

    monkeypatch.delenv("VERITY_FAULT", raising=False)
    monkeypatch.delenv("VERITY_SWITCH", raising=False)
    ns = cli.parse("commit", argv, {})
    assert bool(cd.hot_unsafe_reasons(ns)) is unsafe
    assert bool(hc.unsafe_hints(argv, {})) is unsafe          # the client pre-check agrees with the worker's rule on every case
    monkeypatch.setenv("VERITY_FAULT", "skip_copy:1")
    assert cd.hot_unsafe_reasons(cli.parse("commit", ["--out", "o"], {}))


def test_the_commit_takes_hash_threads_from_the_job_s_environment():
    """HASH_THREADS sets the Commit's hash pool (only the work is parallel: the leaves and the root are the same); on a hot worker it is
    the job's, not the environment the worker was started with."""
    pytest.importorskip("torch")
    assert cli.parse("commit", ["--out", "o"], {}).hash_threads == 2
    assert cli.parse("commit", ["--out", "o"], {"HASH_THREADS": "12"}).hash_threads == 12
    assert cli.parse("commit", ["--out", "o", "--hash-threads", "4"], {"HASH_THREADS": "12"}).hash_threads == 4
    assert "HASH_THREADS" in hc.ENV_PER_JOB


def test_a_later_job_on_the_reused_engine_has_what_the_engines_build_wrote_into_the_environment():
    """vLLM sets cuBLAS' batch-invariance switches in the process as it builds the engine; a job after the first, whose environment
    the worker resets, records them (the engine facts of record) only if the worker puts them back."""
    src = inspect.getsource(hc.serve)
    assert 'cli.set_environment({**env0, **(job.get("env") or {}), **(getattr(cd._HOT.get("eng"), "environ", None) or {})})' in src
    assert "os.environ.clear()" not in src


@pytest.mark.parametrize("argv,env,unsafe", [
    (["--out", "o"], {}, False),
    (["--out", "o", "--arms", "a,b"], {}, True),
    (["--out", "o", "--compiled-taps", "on"], {}, True),
    (["--out", "o", "--compiled-taps", "off"], {}, False),
    (["--out", "o", "--arena-mb", "64", "--arena-unchecked"], {}, True),
    (["--out", "o", "--arena-unchecked"], {}, False),          # no arena: commit_delta ignores the flag, so does the pre-check
    (["--out", "o", "--moe-source", "on"], {}, True),
    (["--out", "o", "--committer", "plugin:m:f"], {}, True),
    (["--out", "o"], {"VERITY_FAULT": "release_before_keep"}, True),
    (["--out", "o"], {"VERITY_SWITCH": "0.001"}, True),
])
def test_unsafe_hints_is_torch_free_and_names_the_flag(argv, env, unsafe):
    hints = hc.unsafe_hints(argv, env)
    assert bool(hints) is unsafe
    if unsafe:
        assert any(tok in hints[0] for tok in ("--", "VERITY_", "plugin"))


def test_submit_pre_check_runs_unsafe_jobs_cold_without_touching_workers(tmp_path, monkeypatch):
    t = _tree(tmp_path)
    monkeypatch.chdir(t)
    monkeypatch.setenv("VERITY_TIMELINE", str(tmp_path / "tl.jsonl"))
    monkeypatch.setenv("VERITY_FAULT", "release_before_keep")
    calls = []
    monkeypatch.setattr(hc.subprocess, "call", lambda cmd, env=None: calls.append((cmd, env)) or 0)
    monkeypatch.setattr(hc, "stop_workers", lambda *a, **k: pytest.fail("a HOT-UNSAFE job must not stop workers"))
    monkeypatch.setattr(hc.subprocess, "Popen", lambda *a, **k: pytest.fail("a HOT-UNSAFE job must not start a worker"))
    rc = hc.submit(["--case", "LLAMA32_1B", "--out", "o"], root=tmp_path / "hot", idle_s=1, timeout_s=None, log=lambda *a, **k: None)
    assert rc == 0 and len(calls) == 1 and not (tmp_path / "hot").exists()
    assert calls[0][1]["VERITY_HOT_RESTART_REASON"].startswith("HOT-UNSAFE (client pre-check): VERITY_FAULT")
    ev = tlm.read(tmp_path / "tl.jsonl")
    assert [e["stage"] for e in ev] == ["hot.fallback_cold"]
