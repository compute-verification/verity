"""Target-family preflight: PRECHECK_FAIL_TARGET_FAMILY by name, WAIVED only by name."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm import target_family as tf

TREE = Path(__file__).resolve().parents[1]
ROW_L40S = "mistral-7b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
ROW_H100 = "qwen3-4b-fp8__fp8__h100__tp1__b8__i1024__o128__mixed__greedy__bi-eager"
ROW_H200_TP8 = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch-t0.8-p0.95__eager"


def test_row_id_grammar_names_the_device_class():
    assert tf.device_token_of_row(ROW_L40S) == "l40s"
    assert tf.row_family(ROW_L40S) == ("l40s", (8, 9))
    assert tf.row_family(ROW_H100) == ("h100", (9, 0))
    assert tf.row_family("llama32-1b__bf16__a100__tp1__b1__i256__o32__mixed__greedy__bi-eager")[1] == (8, 0)
    assert tf.row_family("olmoe-1b-7b-padded__bf16__l40s__tp1__b1__i1024__o128__mixed__greedy__bi-eager")[1] == (8, 9)
    assert tf.row_family("nonsense")[1] is None


def test_60_shape_hopper_build_for_an_l40s_row_fails_by_name():
    """A Build whose build_summary step.target is cc 9.0 (an H100 host) for an __l40s__ workload fails by name."""
    bs = {"step": {"target": {"compute_capability": [9, 0], "num_sms": 132}, "applicability": {"gemm_target": "hopper", "flash_attn_version_applied": 3}}}
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["build"], want_device=False, build_summary=bs)
    assert rec["verdict"] == "FAIL" and rec["fail_class"] == "PRECHECK_FAIL_TARGET_FAMILY"
    assert rec["built"] == [9, 0] and rec["row_family"] == [8, 9]
    assert [m["fact"] for m in rec["mismatches"]] == ["built"]
    assert tf.summary_line(rec).startswith("PRECHECK_FAIL_TARGET_FAMILY: Build target 9.0 != row family 8.9")


def test_declared_target_of_another_family_fails_before_build():
    wl = {"engine": {"target": {"compute_capability": [9, 0], "num_sms": 132, "flash_attn_version": 3}}}
    rec = tf.check(ROW_L40S, wl, ["build"], want_device=False)
    assert rec["verdict"] == "FAIL" and rec["fail_class"] == "PRECHECK_FAIL_TARGET_FAMILY"
    assert rec["declared"] == [9, 0] and [m["fact"] for m in rec["mismatches"]] == ["declared"]


def test_matching_declared_target_and_matching_build_pass():
    wl = {"engine": {"target": {"compute_capability": [8, 9], "num_sms": 142}}}
    bs = {"step": {"target": {"compute_capability": [8, 9]}}}
    rec = tf.check(ROW_L40S, wl, ["build"], want_device=False, build_summary=bs)
    assert rec["verdict"] == "PASS" and rec["ok"] and rec["fail_class"] is None and rec["mismatches"] == []
    assert tf.summary_line(rec).startswith("TARGET-FAMILY-OK row=l40s(8.9) declared=8.9")


def test_pod_device_of_another_family_fails_for_a_gpu_stage(monkeypatch):
    monkeypatch.setattr(tf, "device_cc", lambda: ((9, 0), "test"))
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["match", "commit"])
    assert rec["verdict"] == "FAIL" and rec["device"] == [9, 0]
    assert rec["mismatches"][0]["fact"] == "device" and "9.0 != row family 8.9" in rec["mismatches"][0]["why"]
    # the same device IS the family of an __h100__ workload
    assert tf.check(ROW_H100, {"sweep": {}}, ["match", "commit"])["verdict"] == "PASS"


def test_device_target_build_reads_the_pod_device_declared_target_build_does_not(monkeypatch):
    monkeypatch.setattr(tf, "device_cc", lambda: ((9, 0), "test"))
    assert tf.check(ROW_L40S, {"sweep": {}}, ["build"])["verdict"] == "FAIL"          # target=device -> the Build would be Hopper
    wl = {"engine": {"target": {"compute_capability": [8, 9]}}}
    assert tf.check(ROW_L40S, wl, ["build"])["verdict"] == "PASS"                       # declared L40S target on a Hopper host: Build is portable


def test_a_leased_row_s_device_target_build_is_refused_and_its_gpu_stages_still_read_the_node(monkeypatch):
    """A leased workload hides every GPU from its stages (CUDA_VISIBLE_DEVICES empty), so a Build that targets the device sees none and
    derives for the default target; nvidia-smi, which ignores the variable, would have read the node's GPU and passed it."""
    monkeypatch.setattr(tf, "device_cc", lambda: (_ for _ in ()).throw(AssertionError("device read for a Build that cannot see it")))
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["build"], devices="")
    assert rec["verdict"] == "FAIL" and [m["fact"] for m in rec["mismatches"]] == ["device"]
    assert "CUDA_VISIBLE_DEVICES hides" in tf.summary_line(rec)
    monkeypatch.setattr(tf, "device_cc", lambda: ((8, 9), "test"))
    assert tf.check(ROW_L40S, {"sweep": {}}, ["build", "match"], devices="0")["verdict"] == "PASS"
    assert tf.check(ROW_L40S, {"engine": {"target": {"compute_capability": [8, 9]}}}, ["build", "match"], devices="")["verdict"] == "PASS"
    assert tf.check(ROW_L40S, {"sweep": {}}, ["match", "commit"], devices="")["verdict"] == "PASS", "a GPU stage takes the node's GPU by its lease"


def test_no_visible_gpu_for_a_gpu_stage_fails_closed(monkeypatch):
    monkeypatch.setattr(tf, "device_cc", lambda: (None, "torch.cuda.is_available() False"))
    rec = tf.check(ROW_L40S, {"sweep": {}}, ["commit"])
    assert rec["verdict"] == "FAIL" and rec["mismatches"][0]["fact"] == "device"


def test_the_match_s_cpu_phase_reads_no_device_and_every_gpu_stage_still_refuses_without_one(monkeypatch):
    """The Match's CPU half (the check, the served-shape derives and the fold over GPU stages already captured) runs on a GPU-free
    CPU node: a TP8 workload's `match` doesn't count as a GPU stage when its phase is cpu."""
    monkeypatch.setattr(tf, "device_cc", lambda: (_ for _ in ()).throw(AssertionError("device read for the Match's CPU phase")))
    rec = tf.check(ROW_H200_TP8, {"sweep": {}}, ["match"], match_phase="cpu")
    assert rec["verdict"] == "PASS" and rec["device"] is None and rec["match_phase"] == "cpu"
    monkeypatch.setattr(tf, "device_cc", lambda: (None, "torch.cuda.is_available() False"))
    for stages, phase in ((["match"], "all"), (["match"], "gpu"), (["match", "commit"], "cpu"), (["bench"], "cpu")):
        rec = tf.check(ROW_H200_TP8, {"sweep": {}}, stages, match_phase=phase)
        assert rec["verdict"] == "FAIL" and [m["fact"] for m in rec["mismatches"]] == ["device"], (stages, phase)


def test_the_row_hands_its_match_phase_to_the_preflight_so_its_cpu_half_needs_no_gpu(tmp_path):
    """`--match-phase cpu` on a workload's run reaches `target-family`, which then passes on a host with no GPU; the same
    workload's GPU stages (`--match-phase all`) are refused there."""
    from tests.pipeline.test_row import _tp_row
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(TREE), os.environ.get("PYTHONPATH", "")]), "PATH": str(tmp_path),
           "CUDA_VISIBLE_DEVICES": ""}
    for phase, rc, head in (("cpu", 0, "TARGET-FAMILY-OK"), ("all", 3, "PRECHECK_FAIL_TARGET_FAMILY: no GPU visible for a GPU stage")):
        t = _tp_row(tmp_path, ROW_H200_TP8, "--stages", "match", "--match-phase", phase)
        os.makedirs(t.d, exist_ok=True)
        t.child_env = env
        got = t.target_family(out=str(tmp_path / f"{phase}.json"))
        assert got[0] == rc and got[1].startswith(head), got
        assert json.loads((tmp_path / f"{phase}.json").read_text())["match_phase"] == phase


def test_rtxpro6000_rows_are_the_cc_12_0_family(monkeypatch):
    row = "llama32-1b__bf16__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    assert tf.row_family(row) == ("rtxpro6000", (12, 0))
    monkeypatch.setattr(tf, "device_cc", lambda: ((12, 0), "test"))
    bs = {"step": {"target": {"compute_capability": [12, 0], "num_sms": 188}}}
    assert tf.check(row, {"engine": {"target": {"compute_capability": [12, 0]}}}, ["build", "match", "commit"], build_summary=bs)["verdict"] == "PASS"
    assert tf.check(ROW_L40S, {"sweep": {}}, ["match"])["mismatches"][0]["why"].startswith("pod device 12.0 != row family 8.9")
    monkeypatch.setattr(tf, "device_cc", lambda: ((8, 9), "test"))
    assert tf.check(row, {"sweep": {}}, ["match"])["verdict"] == "FAIL"
    hopper_build = tf.check(row, None, ["build"], want_device=False, build_summary={"step": {"target": {"compute_capability": [9, 0]}}})
    assert hopper_build["verdict"] == "FAIL" and [m["fact"] for m in hopper_build["mismatches"]] == ["built"]


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
    r = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "target-family", "--row", ROW_L40S, "--stages", "build", "--no-device",
                        "--build-summary", str(bs), "--out", str(out)], capture_output=True, text=True, env=env, cwd=TREE)
    assert r.returncode == 3, r.stderr
    assert r.stderr.startswith("PRECHECK_FAIL_TARGET_FAMILY:") and "Build target 9.0 != row family 8.9" in r.stderr
    rec = json.loads(out.read_text())
    assert rec["schema"] == "verity-sweep/target-family/v1" and rec["fail_class"] == "PRECHECK_FAIL_TARGET_FAMILY"
    # the waiver from the environment (canary.sh exports TARGET_FAMILY_WAIVER) turns it into a named WAIVED, rc 0
    r2 = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "target-family", "--row", ROW_L40S, "--stages", "build", "--no-device",
                         "--build-summary", str(bs), "--out", str(out)], capture_output=True, text=True, env={**env, "TARGET_FAMILY_WAIVER": "canary t"}, cwd=TREE)
    assert r2.returncode == 0 and r2.stdout.startswith("TARGET-FAMILY-WAIVED (canary t)")
    assert json.loads(out.read_text())["verdict"] == "WAIVED"
    r3 = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "target-family", "--row", ROW_L40S, "--stages", "build", "--out", str(out)],
                        capture_output=True, text=True, env={**env, "CUDA_VISIBLE_DEVICES": ""}, cwd=TREE)
    assert r3.returncode == 3 and "CUDA_VISIBLE_DEVICES hides" in r3.stderr, "the row's empty CUDA_VISIBLE_DEVICES reaches the check"


def test_the_row_runs_the_preflight_before_build_and_canary_waives_by_name(tmp_path, monkeypatch):
    from tests.pipeline.test_row import _row
    r = _row(tmp_path, "--stages", "build,match")
    seen = []
    for step in ("precheck_identity", "shape", "admission", "build", "match"):
        monkeypatch.setattr(r, step, lambda step=step: seen.append(step))
    monkeypatch.setattr(r, "target_family", lambda *a, out: seen.append(("target_family", r.stages, a)) or
                        (3, "PRECHECK_FAIL_TARGET_FAMILY: Build target 9.0 != row family 8.9"))
    assert r.run() == 3
    assert seen == ["precheck_identity", "shape", ("target_family", "build,match", ())]
    assert "precheck-target FAIL" in (tmp_path / r.row / "stages.txt").read_text()
    can = (TREE / "verity_vllm" / "ops" / "canary.sh").read_text()
    assert 'TARGET_FAMILY_WAIVER=${TARGET_FAMILY_WAIVER:-"canary $TAG"}' in can


def test_the_row_match_argv_carries_the_explicit_snapshot_cap_only_when_set(tmp_path):
    """--snapshot-max-bytes N (MATCH_SNAP_MAX_BYTES) arms the Match's cap; unset -> the argv without it."""
    from tests.pipeline.test_row import _row

    def argv(*flags, environ=None):
        r = _row(tmp_path, *flags, environ=environ)
        r.B, r.ck, r.snap_def, r.snap_steps = 1, "/tmp/ck", "0,1", None
        return r.match_cmd()
    unset = argv()
    assert "--snapshot-max-bytes" not in unset
    for st in (argv("--snapshot-max-bytes", "33554432"), argv(environ={"MATCH_SNAP_MAX_BYTES": "33554432"})):
        i = st.index("--snapshot-max-bytes")
        assert st[i + 1] == "33554432" and st[i - 1] == "0,1" and st[i + 2] == "--skip"
        assert [x for x in st if x != "--snapshot-max-bytes" and x != "33554432"] == unset
