"""The stage entry: `plan_at_stage` writes the plan documents beside a stage's outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.acquire.test_install import Runner, Tiny, base_rows, row  # noqa: E402
from verity_vllm.acquire.install import IncompletePlan  # noqa: E402
from verity_vllm.acquire.plan import plan_digest  # noqa: E402
from verity_vllm.acquire.stage import plan_at_stage  # noqa: E402
from verity_vllm.correspondence.reader_for_acquire import RuntimeTree  # noqa: E402
from verity_vllm.records import capture as CP  # noqa: E402


def _classes(rows):
    capture = {"input_token_ids": CP.INPUT_IDS, "instance_outputs": CP.MODULE_OUTPUT, "sampled_token_ids": CP.SAMPLED}
    return [{"op_path": r["op_path"], "member": r["output_member"], "capture": capture[r["family"]], "width": 8, "dtype": r.get("dtype", "bf16"),
             "definitions": []} for r in rows if r["family"] in capture]


def test_plan_at_stage_writes_the_documents(tmp_path):
    model, runner = Tiny(), Runner()
    rows = base_rows()
    docs = plan_at_stage(model, runner, _classes(rows), tmp_path / "out", source="test-stage")
    assert docs.plan is not None and docs.plan.complete
    assert set(docs.paths) == {"runtime_tree", "acquisition_plan"}
    rt = RuntimeTree.load(docs.paths["runtime_tree"])
    assert rt.source == "test-stage" and "blocks.0.norm" in rt.modules and rt.runner["sample"] is True
    doc = json.loads(Path(docs.paths["acquisition_plan"]).read_text())
    assert plan_digest(doc) == doc["plan_digest"] == docs.plan.digest
    assert any(docs.plan.digest[:16] in ln for ln in docs.log_lines("t"))
    # without classes only the runtime tree is written
    d2 = plan_at_stage(model, runner, None, tmp_path / "out2", source="test-stage")
    assert d2.plan is None and set(d2.paths) == {"runtime_tree"}
    # an unsupported Value refuses by name, the documents are still written for the report
    with pytest.raises(IncompletePlan, match="blocks.0.gone/out"):
        plan_at_stage(model, runner, _classes(rows + [row("instance_outputs", "blocks.0.gone", "out")]), tmp_path / "out3", source="test-stage")
    assert (tmp_path / "out3" / "acquisition_plan.json").exists()
