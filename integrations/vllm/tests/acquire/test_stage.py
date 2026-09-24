"""The stage entry: `plan_at_stage` writes the plan documents beside a stage's outputs."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tests.acquire.test_install import Runner, Tiny, base_rows, row  # noqa: E402
from verity_capture.commit import manifest_format as RM  # noqa: E402
from verity_vllm.acquire.correspondence import RuntimeTree  # noqa: E402
from verity_vllm.acquire.install import IncompletePlan  # noqa: E402
from verity_vllm.acquire.plan import load_plan  # noqa: E402
from verity_vllm.acquire.stage import plan_at_stage  # noqa: E402


def test_plan_at_stage_writes_the_documents(tmp_path):
    model, runner = Tiny(), Runner()
    rows = base_rows()
    man = {"schema": RM.SCHEMA, "program_digest": "0" * 64, "identities": rows, "manifest_digest": RM.manifest_digest(rows), "wrapper_prefix": "",
           "op_path_aliases": {}, "workload": {"row_id": "synthetic-row"}}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(man))
    docs = plan_at_stage(model, runner, mp, tmp_path / "out", source="test-stage")
    assert docs.plan is not None and docs.plan.complete
    assert set(docs.paths) == {"runtime_tree", "acquisition_plan"}
    rt = RuntimeTree.load(docs.paths["runtime_tree"])
    assert rt.source == "test-stage" and "blocks.0.norm" in rt.modules and rt.runner["sample"] is True
    doc = load_plan(docs.paths["acquisition_plan"])
    assert doc["plan_digest"] == docs.plan.digest
    assert any(docs.plan.digest[:16] in ln for ln in docs.log_lines("t"))
    # without a manifest only the runtime tree is written (the CPU gate's off-pod input)
    d2 = plan_at_stage(model, runner, None, tmp_path / "out2", source="test-stage")
    assert d2.plan is None and set(d2.paths) == {"runtime_tree"}
    # an unsupported Value refuses by name, the documents are still written for the report
    rows2 = rows + [row("instance_outputs", "blocks.0.gone", "out")]
    man2 = dict(man, identities=rows2, manifest_digest=RM.manifest_digest(rows2))
    with pytest.raises(IncompletePlan, match="blocks.0.gone/out"):
        plan_at_stage(model, runner, man2, tmp_path / "out3", source="test-stage")
    assert (tmp_path / "out3" / "acquisition_plan.json").exists()
