"""[R16 sys, BRIEF §5] hot Commit worker: the key that decides what may share an engine, the client's cold fallback, the timeline
report's engine-start / reuse accounting.  Torch-free (the worker itself is exercised on a pod: cold vs hot vs hot roots)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verity_capture.sweep import hot_commit as hc
from verity_capture.sweep import timeline as tlm


def _tree(tmp_path: Path) -> Path:
    t = tmp_path / "tree"
    (t / "verity_capture" / "bench").mkdir(parents=True)
    (t / "verity_capture" / "bench" / "commit_delta.py").write_text("print('x')\n")
    (t / "verity_capture" / "tests").mkdir()
    (t / "verity_capture" / "tests" / "test_a.py").write_text("def test(): pass\n")
    (t / "verity_capture" / "__pycache__").mkdir()
    (t / "verity_capture" / "__pycache__" / "a.pyc").write_bytes(b"\0")
    (t / "manifests").mkdir(parents=True)
    (t / "manifests" / "checkpoints.json").write_text('{"LLAMA32_1B": {"repo": "unsloth/Llama-3.2-1B"}}\n')
    (t / "EXPORT.json").write_text('{"commit": "abc"}\n')
    (t / "RELEASE.json").write_text('{"tag": "r16-int-9"}\n')
    (t / "wl.json").write_text(json.dumps({"case": "LLAMA32_1B", "model": "unsloth/Llama-3.2-1B", "engine_args_required": {}, "requests": [{"prompt_token_ids": [1]}],
                                           "sampling": {"variant": "greedy"}}) + "\n")
    return t


def test_code_identity_hashes_code_not_tests_or_caches(tmp_path):
    t = _tree(tmp_path)
    a = hc.code_identity(t)
    assert a["files"] == 2 and a["commit"] == "abc" and a["release"] == "r16-int-9"
    (t / "verity_capture" / "tests" / "test_a.py").write_text("def test(): assert 1\n")
    (t / "verity_capture" / "__pycache__" / "a.pyc").write_bytes(b"\1\2")
    assert hc.code_identity(t)["sha256"] == a["sha256"], "tests and caches are not executed code"
    (t / "verity_capture" / "bench" / "commit_delta.py").write_text("print('y')\n")
    assert hc.code_identity(t)["sha256"] != a["sha256"], "an edited executing tree is a different key -- hot never runs stale code"


def test_job_key_changes_with_engine_args_extension_and_env_but_not_per_job_fields(tmp_path, monkeypatch):
    t = _tree(tmp_path)
    monkeypatch.chdir(t)
    so = t / "tap.so"
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
    assert cmd[:3] == [sys.executable, "-m", "verity_capture.bench.commit_delta"] and cmd[-2:] == ["--out", "o"]
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
    """[SYS-01c] row_pod appends every attempt to the same timeline.jsonl; the table is about the LAST attempt (one engine start, not the
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
    (["--out", "o", "--compiled-taps", "on"], True),
    (["--out", "o", "--late-read", "qkv_proj"], True),
    (["--out", "o", "--value-check"], True),
    (["--out", "o", "--committer", "plugin:m:f"], True),
])
def test_hot_unsafe_flags_are_refused_by_commit_delta(argv, unsafe, monkeypatch):
    torch = pytest.importorskip("torch")  # noqa: F841  commit_delta imports torch at module level
    from verity_capture.bench import commit_delta as cd

    ap = cd.argparse.ArgumentParser()
    # the same parser main() builds: reuse it by parsing through main's namespace defaults
    monkeypatch.delenv("VERITY_FAULT", raising=False)
    monkeypatch.delenv("VERITY_SWITCH", raising=False)
    ns = _parse_with_main_parser(cd, argv)
    assert bool(cd.hot_unsafe_reasons(ns)) is unsafe
    assert bool(hc.unsafe_hints(argv, {})) is unsafe          # the client pre-check agrees with the worker's rule on every case
    monkeypatch.setenv("VERITY_FAULT", "skip_copy:1")
    assert cd.hot_unsafe_reasons(_parse_with_main_parser(cd, ["--out", "o"]))


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


def _parse_with_main_parser(cd, argv):
    """Build main()'s argparse namespace without running main: execute main's source up to parse_args with a stub."""
    import inspect
    import re

    src = inspect.getsource(cd.main)
    body = src.split("ap = argparse.ArgumentParser()", 1)[1].split("args = ap.parse_args(argv)", 1)[0]
    body = re.sub(r"^    ", "", body, flags=re.M)
    ns_env = {"argparse": cd.argparse, "ap": cd.argparse.ArgumentParser(), "os": os, "REQUIRED_CLASSES_DEFAULT": getattr(cd, "REQUIRED_CLASSES_DEFAULT", ())}
    exec(body, ns_env)  # noqa: S102  -- test-only: the option declarations themselves
    return ns_env["ap"].parse_args(argv)
