"""The sweep Commit harness is FAIL-CLOSED.  `commit_delta` may report Commit PASS only if the required-value manifest's
attempted acquisitions equal the bound ones, no collector/arena/hook error was recorded, every opening checked verified (>= 64) and the
control vs instrumented tokens are equal: an arena refusal ("arena block too small for this step") is a FAIL, never a PASS with the
errors buried in runs.jsonl.

Two layers: (1) the pure verdict function on synthetic records (runs anywhere); (2) the end-to-end regression on a GPU pod with vLLM: a
deliberately tiny explicit arena (`--arena-mb 1 --arena-unchecked`) forces the collector refusal and the process MUST exit 3 with
`verdict.json: commit_pass == false` (skipped without CUDA/vLLM; set VERITY_E2E_WORKLOAD to a small workload json and VERITY_E2E_CASE).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.check.commit_pass import commit_verdict

# GPU e2e negatives run on SHARED pods: VERITY_E2E_GPU_UTIL lowers vLLM's gpu_memory_utilization (default 0.5) so a neighbour's Commit keeps its headroom
_E2E_GPU_UTIL = os.environ.get("VERITY_E2E_GPU_UTIL", "0.5")


def _rec(expected=213, committed=213, errors=(), n_open=64, verified=64):
    return {"extra": {"expected_tensors": expected, "committed_tensors": committed, "errors": list(errors), "errors_n": len(errors)},
            "openings": {"n": n_open, "verified": verified}}


def test_pass_requires_everything():
    ok, why = commit_verdict(_rec(), True)
    assert ok and why == "PASS"


def test_arena_refusal_is_fail_even_with_openings_and_tokens():
    # the fail-open signature: tokens equal, 64/64 openings, but the prefill step was refused past the 512 MiB block
    ok, why = commit_verdict(_rec(expected=27264, committed=27075, errors=["fa2: native_collect: arena block too small for this step"]), True)
    assert not ok and "expected_tensors 27264 != committed_tensors 27075" in why


def test_error_alone_is_fail():
    ok, why = commit_verdict(_rec(errors=["hook: native_collect: arena block too small for this step"]), True)
    assert not ok and "collector error" in why


def test_windowed_slot_wait_timeout_is_fail_named():
    # bounded staging with backpressure: when the ring is exhausted the serving thread BLOCKS; after win_wait_timeout_s the
    # acquisition fails and is recorded ("hook: windowed staging: waited ...").  It must FAIL Commit by name -- never a warning, never PASS
    # on the remaining tensors (expected != committed) and never PASS even if the counts happened to agree (the error alone fails).
    err = "hook: windowed staging: waited 2 s for a free window slot (ring of 3 x 16 MiB, all held): the worker is not releasing windows"
    ok, why = commit_verdict(_rec(expected=6816, committed=6790, errors=[err]), True)
    assert not ok and "expected_tensors 6816 != committed_tensors 6790" in why
    ok, why = commit_verdict(_rec(errors=[err]), True)
    assert not ok and "collector error" in why and "windowed staging: waited" in why
    ok, why = commit_verdict(_rec(errors=["hook: windowed staging: collector dead after a slot-wait timeout earlier in this run"]), True)
    assert not ok and "collector dead after a slot-wait timeout" in why


def test_abandoned_step_fails_by_its_name_first():
    # The worker abandoned a learning step past the transient host budget.  The collector names the run's
    # failure (extra.collector_failure, also errors[0]); the verdict leads with that name, ahead of the lost acquisitions it caused.
    why0 = ("worker failed on windowed step 0: RuntimeError('bounded staging: learning step 0 exceeds the transient host budget 4096 MiB at "
            "window 16 (4352 MiB): its plan was not pre-learned (warm-up learn-only pass) -- step not committed')")
    r = _rec(expected=6816, committed=4100, errors=[f"collector failed: {why0}", "worker step 0: RuntimeError(...)"])
    r["extra"]["collector_failure"] = why0
    ok, why = commit_verdict(r, True)
    assert not ok and why.startswith("collector failed: worker failed on windowed step 0") and "transient host budget" in why
    r["extra"].update(expected_tensors=6816, committed_tensors=6816, errors=[], errors_n=0)
    ok, why = commit_verdict(r, True)
    assert not ok and why.startswith("collector failed:"), "the named failure alone fails the run, even with equal counts and no error"


def test_opening_after_release_failure_is_fail_named():
    # lifetime: a committed buffer whose device staging was released before the opening was verified must still open against
    # the ORIGINAL bytes; a retained copy that holds a later occupant's bytes (release-before-keep ordering fault) fails the openings-after-
    # release check by name, even with a correct root, 64/64 in-run openings, equal counts, equal tokens and no recorded error.
    r = _rec()
    r["openings_after_release"] = {"n": 64, "verified": 61, "source": "host"}
    ok, why = commit_verdict(r, True)
    assert not ok and "openings after release verified 61 != checked 64" in why
    r["openings_after_release"] = {"n": 64, "verified": 64, "source": "host", "retained_device_after": 4096}
    ok, why = commit_verdict(r, True)
    assert not ok and "still on the device after release" in why
    r["openings_after_release"] = {"n": 64, "verified": 64, "source": "replay", "replay_root_equal": False}
    ok, why = commit_verdict(r, True)
    assert not ok and "replay run root differs" in why
    r["openings_after_release"] = {"n": 64, "verified": 64, "source": "host", "retained_device_after": 0}
    assert commit_verdict(r, True)[0]


def test_openings_and_tokens():
    assert not commit_verdict(_rec(n_open=32, verified=32), True)[0]          # fewer than 64 checked
    assert not commit_verdict(_rec(n_open=64, verified=63), True)[0]          # one opening failed
    assert not commit_verdict(_rec(), False)[0]                               # tokens differ
    assert not commit_verdict(_rec(), None)[0]                                # tokens unknown -> not PASS


def test_missing_counters_or_nothing_bound_is_fail():
    assert not commit_verdict({"extra": {"errors": []}, "openings": {"n": 64, "verified": 64}}, True)[0]
    assert not commit_verdict(_rec(expected=0, committed=0), True)[0]


# ---- a required producer that never fires must FAIL even with expected == committed and 0 errors
def _decl(fa2_tensors=48, with_fa2=True):
    classes = [{"class": "input_token_ids", "how": "bound-as-leaf", "tensors": 32, "bytes": 1},
               {"class": "instance_outputs_native", "how": "bound-as-leaf", "tensors": 6000, "bytes": 1},
               {"class": "sampled_token_ids", "how": "bound-as-leaf", "tensors": 32, "bytes": 1},
               {"class": "weights", "how": "registered-input", "tensors": 200, "bytes": 1},
               {"class": "norm_bf16tof32_lift", "how": "derived-from-bound", "bytes": None}]
    if with_fa2:
        classes.append({"class": "fa2_hidden_m1_stream", "how": "bound-as-leaf", "tensors": fa2_tensors, "bytes": 1})
    return {"classes": classes}


def test_x05_all_required_classes_bound_passes():
    ok, why = commit_verdict(_rec(), True, declaration=_decl())
    assert ok and why == "PASS"


def test_x05_hopper_signature_tap_loaded_zero_launches_fails():
    # tap attached, FA3 executes, FA2 never called -> class declared with tensors 0, expected == committed
    ok, why = commit_verdict(_rec(), True, declaration=_decl(fa2_tensors=0))
    assert not ok and "fa2_hidden_m1_stream" in why and "never fired" in why


def test_x05_omitted_tap_not_declared_fails():
    # --hidden-so left off the command line: the class is not even declared -> FAIL (before REL-02 this was PASS)
    ok, why = commit_verdict(_rec(), True, declaration=_decl(with_fa2=False))
    assert not ok and "required class 'fa2_hidden_m1_stream' not declared" in why


def test_x05_required_set_is_overridable_but_declared_zero_still_fails():
    ok, _ = commit_verdict(_rec(), True, declaration=_decl(with_fa2=False), required_classes=("weights", "instance_outputs_native"))
    assert ok
    ok, why = commit_verdict(_rec(), True, declaration=_decl(fa2_tensors=0), required_classes=("weights",))
    assert not ok and "tensors=0" in why


def test_x05_missing_declaration_fails_when_required():
    ok, why = commit_verdict(_rec(), True, declaration=None, required_classes=("weights",))
    assert not ok and "declaration missing" in why


@pytest.mark.skipif(not (os.environ.get("VERITY_E2E_WORKLOAD") and os.environ.get("VERITY_E2E_CASE")), reason="GPU pod end-to-end only (needs vLLM + a workload)")
def test_e2e_windowed_stall_forces_commit_fail(tmp_path: Path):
    """The default path (windowed ring, retain host) never DROPS an acquisition: a stalled worker (VERITY_CONSUMER_DELAY_MS)
    with a 3 x 16 MiB ring makes the serving thread block on the ring; with the slot wait cut to 2 s (VERITY_WIN_WAIT_TIMEOUT_S) the wait times
    out, the acquisition fails NAMED and the collector is dead for the run -> the verdict must turn it into exit 3 / COMMIT FAIL, never PASS."""
    wl, case = os.environ["VERITY_E2E_WORKLOAD"], os.environ["VERITY_E2E_CASE"]
    env = dict(os.environ, VERITY_WIN_WAIT_TIMEOUT_S="2", VERITY_CONSUMER_DELAY_MS="5000")
    cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "commit", "--case", case, "--workload", wl, "--committer", "native_collect_v2b",
           "--openings", "64", "--pairs", "1", "--warmup", "0", "--max-num-seqs", "1", "--window-mb", "16", "--window-slots", "3", "--retain", "host", "--no-bounded-staging",
           "--engine-arg", f"gpu_memory_utilization={_E2E_GPU_UTIL}", "--out", str(tmp_path / "out")]
    if os.environ.get("VERITY_E2E_HIDDEN_SO"):
        cmd += ["--hidden-so", os.environ["VERITY_E2E_HIDDEN_SO"]]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert p.returncode == 3, f"expected exit 3 (Commit FAIL), got {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    v = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert v["commit_pass"] is False and v["runs_failed"] >= 1
    why = v["first_fail_reason"] or ""
    assert "windowed staging" in why or "!= committed_tensors" in why, why
    assert "COMMIT FAIL" in p.stdout


@pytest.mark.skipif(not (os.environ.get("VERITY_E2E_WORKLOAD") and os.environ.get("VERITY_E2E_CASE")), reason="GPU pod end-to-end only (needs vLLM + a workload)")
def test_e2e_windowed_release_before_keep_fails_openings(tmp_path: Path):
    """Opening-after-release / lifetime negative on the default path (windowed ring, retain host): VERITY_FAULT=release_before_keep
    hands every window slot back to the ring BEFORE its retained copy is taken and copies only after the ring refilled the slot.  The tree is
    hashed in place from the right bytes (root correct, control == instrumented tokens, counts equal, no error recorded) but the retained
    bytes are a later window's -> the openings-after-release check MUST fail by name -> exit 3 / COMMIT FAIL.  The fault runs on the
    windowed path too, so the negative cannot pass as a positive."""
    wl, case = os.environ["VERITY_E2E_WORKLOAD"], os.environ["VERITY_E2E_CASE"]
    env = dict(os.environ, VERITY_FAULT="release_before_keep")
    cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "commit", "--case", case, "--workload", wl, "--committer", "native_collect_v2b",
           "--openings", "64", "--pairs", "1", "--warmup", "1", "--max-num-seqs", "1", "--window-mb", "16", "--window-slots", "3", "--retain", "host", "--no-bounded-staging",
           "--open-after-release", "--engine-arg", f"gpu_memory_utilization={_E2E_GPU_UTIL}", "--out", str(tmp_path / "out")]   # row_pod.sh's P0 defaults
    if os.environ.get("VERITY_E2E_HIDDEN_SO"):
        cmd += ["--hidden-so", os.environ["VERITY_E2E_HIDDEN_SO"]]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert p.returncode == 3, f"expected exit 3 (Commit FAIL), got {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    v = json.loads((tmp_path / "out" / "verdict.json").read_text())
    assert v["commit_pass"] is False and v["runs_failed"] >= 1
    why = v["first_fail_reason"] or ""
    assert "opening" in why, why                                                # openings / openings after release verified N != checked 64
    assert "COMMIT FAIL" in p.stdout
    runs = [json.loads(l) for l in (tmp_path / "out" / "runs.jsonl").read_text().splitlines() if l.strip()]
    faulted = [r for r in runs if ((r.get("commit") or {}).get("extra") or {}).get("windows", {}).get("fault_lifetime_windows", 0) > 0]
    assert faulted, "the fault never materialised (no window was copied after its slot was reused): the negative did not test the lifetime path"


# ----------------------------------------------------------------------------------------------------------------------------------------
# "collector dead" must be fail-FAST, not only fail-closed by name: a Commit that hits the slot-wait timeout can otherwise sit at 0 % GPU
# with every thread in futex_wait (the named throw needs the GIL back and a Python frame that records the failure).  The C++ collector arms
# a GIL-free watchdog thread the moment it declares itself dead: unless Python acknowledges (finalize -> ack_dead) within
# VERITY_COLLECT_DEAD_GRACE_S it writes <out>/COLLECTOR_DEAD_HUNG.json and _exit(3)s.

_WATCHDOG_PROG = r"""
import os, sys, time, threading
import torch
from verity_vllm.commit.committer import native_collect as nc
c = nc.ext().Collector()
marker = sys.argv[1]; grace = float(sys.argv[2]); mode = sys.argv[3]
c.set_watchdog(marker, grace)
c.force_dead_for_test("unit test: simulated slot-wait timeout")
assert c.watchdog_started() and c.win_dead
if mode == "ack":
    time.sleep(grace / 4); c.ack_dead()            # the run fails closed on its own -> the watchdog must stand down
    time.sleep(grace * 2); print("ALIVE-AFTER-GRACE", flush=True); sys.exit(0)
if mode == "progress":
    for _ in range(8):                             # a slow but PROGRESSING fail-closed run (steps keep ending) is never killed
        time.sleep(grace / 2); c.progress()
    c.ack_dead(); print("ALIVE-AFTER-GRACE", flush=True); sys.exit(0)
threading.Event().wait()                            # simulated hang: never acknowledges -> the watchdog alone must end the process
"""


def _watchdog_run(tmp_path: Path, mode: str, grace: float):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed to build the native collector extension")
    marker = tmp_path / "COLLECTOR_DEAD_HUNG.json"
    p = subprocess.run([sys.executable, "-c", _WATCHDOG_PROG, str(marker), str(grace), mode], capture_output=True, text=True,
                       env=dict(os.environ, VERITY_FAULT=""))
    return p, marker


def test_dead_watchdog_exits_rc3_with_marker_when_never_acknowledged(tmp_path: Path):
    p, marker = _watchdog_run(tmp_path, "hang", 2.0)
    assert p.returncode == 3, f"expected rc 3 from the watchdog, got {p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}"
    assert "DEAD-WATCHDOG" in p.stderr and "rc 3" in p.stderr
    assert marker.exists(), "the watchdog must leave a named marker in the out dir"
    m = json.loads(marker.read_text())
    assert m["schema"] == "verity-collect/dead-watchdog/v1" and m["rc"] == 3 and "did not fail closed" in m["reason"]




@pytest.mark.skipif(not (os.environ.get("VERITY_E2E_WORKLOAD") and os.environ.get("VERITY_E2E_CASE")), reason="GPU pod end-to-end only (needs vLLM + a workload)")
def test_e2e_collector_dead_hang_exits_rc3_within_grace(tmp_path: Path):
    """The stall negative above, plus VERITY_FAULT=hang_after_dead: finalize blocks forever instead of failing the run closed (the hang
    signature).  With the grace cut to 20 s the process MUST still end with rc 3 -- from the watchdog, not from the verdict -- leaving
    out/COLLECTOR_DEAD_HUNG.json; row_pod reads rc 3 as COMMIT FAIL.  (verdict.json is NOT written: the hang happened before closure.)"""
    wl, case = os.environ["VERITY_E2E_WORKLOAD"], os.environ["VERITY_E2E_CASE"]
    env = dict(os.environ, VERITY_WIN_WAIT_TIMEOUT_S="2", VERITY_CONSUMER_DELAY_MS="5000", VERITY_FAULT="hang_after_dead", VERITY_COLLECT_DEAD_GRACE_S="20")
    cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "commit", "--case", case, "--workload", wl, "--committer", "native_collect_v2b",
           "--openings", "64", "--pairs", "1", "--warmup", "0", "--max-num-seqs", "1", "--window-mb", "16", "--window-slots", "3", "--retain", "host", "--no-bounded-staging",
           "--engine-arg", f"gpu_memory_utilization={_E2E_GPU_UTIL}", "--out", str(tmp_path / "out")]
    if os.environ.get("VERITY_E2E_HIDDEN_SO"):
        cmd += ["--hidden-so", os.environ["VERITY_E2E_HIDDEN_SO"]]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert p.returncode == 3, f"expected exit 3 from the dead-watchdog, got {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
    assert "FAULT hang_after_dead" in p.stdout, "the fault never armed: the negative did not test the hang path"
    assert "DEAD-WATCHDOG" in p.stderr, p.stderr[-2000:]
    marker = tmp_path / "out" / "COLLECTOR_DEAD_HUNG.json"
    assert marker.exists() and json.loads(marker.read_text())["rc"] == 3
    assert not (tmp_path / "out" / "verdict.json").exists()
