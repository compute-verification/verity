"""Target-family preflight (r18-int-2; x09 M-0009 / coord M-0026): PRECHECK_FAIL_TARGET_FAMILY by name, WAIVED only by name."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.harness import target_family as tf

TREE = Path(__file__).resolve().parents[2]
ROW_L40S = "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
ROW_H100 = "qwen3-4b-fp8__fp8__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager"


def test_row_id_grammar_names_the_device_class():
    assert tf.device_token_of_row(ROW_L40S) == "l40s"
    assert tf.row_family(ROW_L40S) == ("l40s", (8, 9))
    assert tf.row_family(ROW_H100) == ("h100", (9, 0))
    assert tf.row_family("llama32-1b__bf16__a100__tp1__b1__i256__o32__mixed__greedy__bi-eager")[1] == (8, 0)
    assert tf.row_family("olmoe-1b-7b-padded__bf16__l40s__tp1__b1__i1024__o128__mixed__greedy__bi-eager")[1] == (8, 9)
    assert tf.row_family("nonsense")[1] is None


def test_60_shape_hopper_build_for_an_l40s_row_fails_by_name():
    """#60 of record: build_summary step.target cc 9.0 (H100 host) for an __l40s__ row (r17_1207)."""
    bs = {"step": {"target": {"compute_capability": [9, 0], "num_sms": 132}, "applicability": {"gemm_target": "hopper", "flash_attn_version_applied": 3}}}
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["build"], want_device=False, build_summary=bs)
    assert rec["verdict"] == "FAIL" and rec["fail_class"] == "PRECHECK_FAIL_TARGET_FAMILY"
    assert rec["built"] == [9, 0] and rec["row_family"] == [8, 9]
    assert [m["fact"] for m in rec["mismatches"]] == ["built"]
    assert tf.summary_line(rec).startswith("PRECHECK_FAIL_TARGET_FAMILY: Build target 9.0 != row family 8.9")


def test_declared_target_of_another_family_fails_before_build():
    wl = {"target": {"compute_capability": [9, 0], "num_sms": 132, "flash_attn_version": 3}}
    rec = tf.check(ROW_L40S, wl, ["build"], want_device=False)
    assert rec["verdict"] == "FAIL" and rec["fail_class"] == "PRECHECK_FAIL_TARGET_FAMILY"
    assert rec["declared"] == [9, 0] and [m["fact"] for m in rec["mismatches"]] == ["declared"]


def test_matching_declared_target_and_matching_build_pass():
    wl = {"target": {"compute_capability": [8, 9], "num_sms": 142}}
    bs = {"step": {"target": {"compute_capability": [8, 9]}}}
    rec = tf.check(ROW_L40S, wl, ["build"], want_device=False, build_summary=bs)
    assert rec["verdict"] == "PASS" and rec["ok"] and rec["fail_class"] is None and rec["mismatches"] == []
    assert tf.summary_line(rec).startswith("TARGET-FAMILY-OK row=l40s(8.9) declared=8.9")


def test_pod_device_of_another_family_fails_for_a_gpu_stage(monkeypatch):
    monkeypatch.setattr(tf, "device_cc", lambda: ((9, 0), "test"))
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["match", "commit"])
    assert rec["verdict"] == "FAIL" and rec["device"] == [9, 0]
    assert rec["mismatches"][0]["fact"] == "device" and "9.0 != row family 8.9" in rec["mismatches"][0]["why"]
    # the same device IS the family of an __h100__ row
    assert tf.check(ROW_H100, {"sweep": {}}, ["match", "commit"])["verdict"] == "PASS"


def test_device_target_build_reads_the_pod_device_declared_target_build_does_not(monkeypatch):
    monkeypatch.setattr(tf, "device_cc", lambda: ((9, 0), "test"))
    assert tf.check(ROW_L40S, {"sweep": {}}, ["build"])["verdict"] == "FAIL"          # target=device -> the Build would be Hopper
    wl = {"target": {"compute_capability": [8, 9]}}
    assert tf.check(ROW_L40S, wl, ["build"])["verdict"] == "PASS"                       # declared L40S target on a Hopper host: Build is portable


def test_no_visible_gpu_for_a_gpu_stage_fails_closed(monkeypatch):
    monkeypatch.setattr(tf, "device_cc", lambda: (None, "torch.cuda.is_available() False"))
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["commit"])
    assert rec["verdict"] == "FAIL" and rec["mismatches"][0]["fact"] == "device"


def test_unknown_device_class_is_a_refusal_not_a_guess():
    rec = tf.check("m__bf16__tpu5__tp1__b1__i1__o1__mixed__greedy__bi-eager", {"sweep": {}}, ["build"], want_device=False)
    assert rec["verdict"] == "FAIL" and rec["mismatches"][0]["fact"] == "row"


def test_waiver_records_the_would_be_failure_by_name_and_never_upgrades_a_pass(monkeypatch):
    monkeypatch.setattr(tf, "device_cc", lambda: ((9, 0), "test"))
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["match"], waiver="canary r18-int-2")
    assert rec["verdict"] == "WAIVED" and rec["ok"] and rec["fail_class"] is None and rec["would_fail_class"] == "PRECHECK_FAIL_TARGET_FAMILY"
    assert rec["waiver"] == "canary r18-int-2" and rec["mismatches"]
    assert tf.summary_line(rec).startswith("TARGET-FAMILY-WAIVED (canary r18-int-2)")
    ok = tf.check(ROW_H100, {"sweep": {}}, ["match"], waiver="canary r18-int-2")
    assert ok["verdict"] == "PASS" and "would_fail_class" not in ok


def test_cli_fails_by_name_with_rc_3_and_writes_the_record(tmp_path):
    bs = tmp_path / "build_summary.json"
    bs.write_text(json.dumps({"step": {"target": {"compute_capability": [9, 0]}}}))
    out = tmp_path / "target_family.json"
    env = {**os.environ, "PYTHONPATH": f"{TREE}:{TREE / 'vllm-poc'}"}
    env.pop("TARGET_FAMILY_WAIVER", None)
    r = subprocess.run([sys.executable, "-m", "verity_vllm.harness.target_family", "--row", ROW_L40S, "--stages", "build", "--no-device",
                        "--build-summary", str(bs), "--out", str(out)], capture_output=True, text=True, env=env, cwd=TREE)
    assert r.returncode == 3, r.stderr
    assert r.stderr.startswith("PRECHECK_FAIL_TARGET_FAMILY:") and "Build target 9.0 != row family 8.9" in r.stderr
    rec = json.loads(out.read_text())
    assert rec["schema"] == "verity-sweep/target-family/v1" and rec["fail_class"] == "PRECHECK_FAIL_TARGET_FAMILY"
    # the waiver from the environment (canary.sh exports TARGET_FAMILY_WAIVER) turns it into a named WAIVED, rc 0
    r2 = subprocess.run([sys.executable, "-m", "verity_vllm.harness.target_family", "--row", ROW_L40S, "--stages", "build", "--no-device",
                         "--build-summary", str(bs), "--out", str(out)], capture_output=True, text=True, env={**env, "TARGET_FAMILY_WAIVER": "canary t"}, cwd=TREE)
    assert r2.returncode == 0 and r2.stdout.startswith("TARGET-FAMILY-WAIVED (canary t)")
    assert json.loads(out.read_text())["verdict"] == "WAIVED"


def test_row_pod_runs_the_preflight_before_build_and_canary_waives_by_name():
    rp = (TREE / "verity_vllm" / "ops" / "row_pod.sh").read_text()
    i_tf = rp.index("verity_vllm.harness.target_family --row")
    i_build = rp.index("tl stage.build start") if "tl stage.build start" in rp else rp.index("# ------------------------------------------------------------------------------- Build")
    assert i_tf < i_build, "target-family preflight must precede the Build stage"
    assert "PRECHECK_FAIL_TARGET_FAMILY" in rp and "--build-summary" in rp
    can = (TREE / "verity_vllm" / "ops" / "canary.sh").read_text()
    assert 'TARGET_FAMILY_WAIVER=${TARGET_FAMILY_WAIVER:-"canary $TAG"}' in can


def test_row_pod_match_argv_carries_the_explicit_snapshot_cap_only_when_set():
    """[r18-int-2; cheap M-0082, F-r18-cheap-2] MATCH_SNAP_MAX_BYTES -> --snapshot-max-bytes N; unset -> byte-identical argv to int-1."""
    probe = r'''
    PY312=python3; ROLE=B0; WL=w.json; D=/tmp/d; CK=/tmp/ck; B=1; SNAP_DEF=0,1; SKIP=x; GPU_UTIL=0.5; PY=python3
    CMD_M=($PY312 -m verity_vllm.harness.run_config --case "$ROLE" --workload "$WL" --out "$D/match" --checkpoint-dir "$CK" --phase all
         --python-gpu "$PY" --python-cpu "$PY312" --max-num-seqs "$B" --snapshot-steps "${MATCH_SNAP_STEPS:-$SNAP_DEF}"
         ${MATCH_SNAP_MAX_BYTES:+--snapshot-max-bytes "$MATCH_SNAP_MAX_BYTES"} --skip "$SKIP"
         --engine-arg "gpu_memory_utilization=$GPU_UTIL" --cards-dir "$D/match/cards")
    printf '%s\n' "${CMD_M[@]}"
    '''
    src = (TREE / "verity_vllm" / "ops" / "row_pod.sh").read_text()
    assert '${MATCH_SNAP_MAX_BYTES:+--snapshot-max-bytes "$MATCH_SNAP_MAX_BYTES"}' in src
    env = {k: v for k, v in os.environ.items() if k != "MATCH_SNAP_MAX_BYTES"}
    unset = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, env=env, check=True).stdout.split("\n")
    assert "--snapshot-max-bytes" not in unset
    st = subprocess.run(["bash", "-c", probe], capture_output=True, text=True, env={**env, "MATCH_SNAP_MAX_BYTES": "33554432"}, check=True).stdout.split("\n")
    i = st.index("--snapshot-max-bytes")
    assert st[i + 1] == "33554432" and st[i - 1] == "0,1" and st[i + 2] == "--skip"
    assert [x for x in st if x != "--snapshot-max-bytes" and x != "33554432"] == unset
