"""`compiled-merge` pairs the two processes of a served-compiled row: a control at `--compiled-taps off` and a treatment at
`--compiled-taps served` that ran the same generated kernels and produced the same tokens merge into a passing paired summary; a
treatment at any other taps, and a treatment whose generated kernels differ from the control's, are refused by name.  Torch-free:
the arms' records are written here."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import compiled_merge as cm

COMPILATION: dict[str, Any] = {"compilation_mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1, 2],
                               "enforce_eager": False, "use_inductor_graph_partition": False, "splitting_ops": ["attention"],
                               "inductor_compile_config": {}}
KERNELS: dict[str, Any] = {"kernel_set_sha256": "aa", "kernels": {"triton_poi_fused_0": "s0"}, "wrappers": 3, "distinct_kernel_sources": 1}
COMMIT: dict[str, Any] = {"commit_pass": True, "run_root": "00", "value_correspondence": {"result": True}}


def _arm(d: Path, taps: str, *, commit: dict[str, Any] | None = None, kernels: dict[str, Any] = KERNELS) -> Path:
    d.mkdir()
    (d / "versions.json").write_text(json.dumps({"compilation": {**COMPILATION, "compiled_taps": taps, "generated_kernels": kernels}}))
    rows = [{"wall_s": 1.0 + i, "step_wall_decode_mean_s": 0.5, "step_wall_prefill_s": 0.25, "tokens_digest": "t" * 16,
             **({"commit": commit} if commit else {})} for i in range(2)]
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "verdict.json").write_text(json.dumps({"run_roots": ["00", "00"], "runs_instrumented": 2 if commit else 0, "runs_failed": 0}))
    return d


def _merge(tmp_path: Path, control: Path, treatment: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    out = tmp_path / "merged"
    with pytest.raises(SystemExit) as e:
        cm.main(cli.parse("compiled-merge", ["--control", str(control), "--treatment", str(treatment), "--out", str(out)], {}))
    return e.value.code, json.loads((out / "summary.json").read_text()), json.loads((out / "verdict.json").read_text())


def test_a_served_treatment_and_its_control_merge_into_a_passing_pair(tmp_path: Path) -> None:
    rc, summary, verdict = _merge(tmp_path, _arm(tmp_path / "c", "off"), _arm(tmp_path / "t", "served", commit=COMMIT))
    assert rc == 0 and summary["problems"] == [] and summary["commit_pass"] is True and verdict["commit_pass"] is True
    assert summary["pairs"] == 2 and summary["tokens_equal"] is True and summary["compilation_settings_equal"] is True
    assert summary["generated_kernels"]["kernel_set_equal"] is True and summary["generated_kernels"]["kernel_diff"] is not None
    assert [p["delta_wall_s"] for p in summary["per_pair"]] == [0.0, 0.0] and verdict["paired"] is True
    assert len((tmp_path / "merged" / "runs.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("taps", ["kernels", "on", "off"])
def test_a_treatment_at_other_taps_is_refused(tmp_path: Path, taps: str) -> None:
    rc, summary, verdict = _merge(tmp_path, _arm(tmp_path / "c", "off"), _arm(tmp_path / "t", taps, commit=COMMIT))
    assert rc == 3 and summary["commit_pass"] is False
    assert summary["problems"] == ["treatment process did not run with --compiled-taps served"] == [verdict["first_fail_reason"]]


def test_a_control_with_taps_is_refused(tmp_path: Path) -> None:
    rc, summary, _ = _merge(tmp_path, _arm(tmp_path / "c", "served"), _arm(tmp_path / "t", "served", commit=COMMIT))
    assert rc == 3 and summary["problems"] == ["control process ran with compiled taps"]


def test_a_treatment_whose_generated_kernels_differ_is_refused(tmp_path: Path) -> None:
    other = {**KERNELS, "kernel_set_sha256": "bb", "kernels": {"triton_poi_fused_0": "s1", "triton_red_fused_1": "s2"}}
    rc, summary, _ = _merge(tmp_path, _arm(tmp_path / "c", "off"), _arm(tmp_path / "t", "served", commit=COMMIT, kernels=other))
    assert rc == 3 and len(summary["problems"]) == 1 and summary["problems"][0].startswith("served path: the treatment's generated kernel set differs")
    assert summary["generated_kernels"]["kernel_diff"] == {"only_control": [], "only_treatment": ["triton_red_fused_1"],
                                                           "same_name_different_source": ["triton_poi_fused_0"]}


def test_arms_whose_kernels_are_not_identified_are_refused(tmp_path: Path) -> None:
    rc, summary, _ = _merge(tmp_path, _arm(tmp_path / "c", "off", kernels={}), _arm(tmp_path / "t", "served", commit=COMMIT))
    assert rc == 3 and summary["problems"] == ["served path: generated kernels not identified in both arms (inductor_kernels.json)"]
    assert summary["generated_kernels"]["kernel_diff"] is None
