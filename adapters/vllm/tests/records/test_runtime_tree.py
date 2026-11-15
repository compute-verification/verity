"""`verity_vllm.records.runtime_tree`: the runtime's module tree reads back what `save` wrote, reads a field the document
leaves out as empty, refuses another schema, and is the one type the observation and the acquisition's reader hand on; and
the tree the observation makes of a model on the meta device (`verity_vllm.observe.runtime_tree.meta_runtime_tree`)."""
from __future__ import annotations

import json

import pytest

from verity_vllm.correspondence import reader_for_acquire
from verity_vllm.observe import runtime_tree as observed
from verity_vllm.records.runtime_tree import (
    RUNNER_PATH,
    SCHEMA,
    RuntimeTree,
    mro_names,
    runtime_tree_from_paths,
    runtime_tree_of,
)

FIELDS = ("schema", "source", "runner", "modules", "buffers")
EMPTY: tuple[object, ...] = (None, 0, "", False, [], {})


def a_tree() -> RuntimeTree:
    return runtime_tree_from_paths({"body.blocks.0": ["Block", "Module", "object"], "": ["Model", "Module", "object"]},
                                   buffers=["body.mask", "body.blocks.0.scale"], runner={"sample": True, "sampler_module": False, "source": "declared"},
                                   source="a test")


def test_the_document_is_the_five_fields_under_the_schema_of_record():
    doc = a_tree().to_json()
    assert tuple(doc) == FIELDS and doc["schema"] == SCHEMA == "verity-vllm/runtime-tree/v1"
    assert list(doc["modules"]) == ["", "body.blocks.0"] and doc["buffers"] == ["body.blocks.0.scale", "body.mask"]
    assert doc["runner"] == {"sample": True, "sampler_module": False, "source": "declared"} and doc["source"] == "a test"


def test_a_tree_reads_back_as_it_was_saved(tmp_path):
    tree, path = a_tree(), tmp_path / "runtime_tree.json"
    tree.save(path)
    assert path.read_text() == json.dumps(tree.to_json(), indent=1) + "\n"
    back = RuntimeTree.load(path)
    assert back.to_json() == tree.to_json() and back.has_module("body.blocks.0") and not back.has_module("body.blocks.1")
    assert RuntimeTree.from_json(RuntimeTree().to_json()) == RuntimeTree()        # an empty tree is a tree: no field is missing


@pytest.mark.parametrize("field", FIELDS[1:])
def test_a_field_the_document_leaves_out_or_holds_as_null_or_empty_reads_as_empty(tmp_path, field):
    whole, path = a_tree().to_json(), tmp_path / "runtime_tree.json"
    want = {**whole, field: RuntimeTree().to_json()[field]}
    path.write_text(json.dumps({k: v for k, v in whole.items() if k != field}))
    assert RuntimeTree.load(path).to_json() == want
    for value in EMPTY:
        assert RuntimeTree.from_json({**whole, field: value}).to_json() == want


def test_a_document_that_names_no_schema_is_read_and_one_that_names_another_is_refused():
    doc = a_tree().to_json()
    for unnamed in ({k: v for k, v in doc.items() if k != "schema"}, {**doc, "schema": None}):
        assert RuntimeTree.from_json(unnamed).to_json() == doc
    for other in ("verity-vllm/runtime-tree/v0", *EMPTY[1:]):
        with pytest.raises(ValueError) as e:
            RuntimeTree.from_json({**doc, "schema": other})
        assert type(e.value) is ValueError and str(e.value) == f"runtime tree schema {other!r} != verity-vllm/runtime-tree/v1"


def test_a_field_of_another_type_is_read_as_its_expression_reads_it():
    doc = a_tree().to_json()
    assert RuntimeTree.from_json({**doc, "source": 3}).source == "3"
    for field, wrong, error in (("modules", ["x"], AttributeError), ("buffers", 1, TypeError), ("runner", 1, TypeError)):
        with pytest.raises(error):
            RuntimeTree.from_json({**doc, field: wrong})


def test_the_tree_of_a_model_names_every_module_and_buffer_and_states_the_runner():
    class Leaf:
        def named_modules(self):
            return [("", self), ("inner", object())]

        def named_buffers(self):
            return [("inner.scale", None)]

    class Runner:
        sampler = None

        def sample(self):
            return None

    model = Leaf()
    bare = runtime_tree_of(model)
    assert bare.modules == {"": ["Leaf", "object"], "inner": ["object"]} and bare.buffers == ["inner.scale"] and bare.source == "live"
    assert bare.runner == {"sample": False, "sampler_module": False, "source": "no runner"}
    assert runtime_tree_of(model, Runner(), source="engine").runner == {"sample": True, "sampler_module": False, "source": "engine runner"}
    assert mro_names(3) == ["int", "object"] and RUNNER_PATH == "runner.sampler"


def test_the_observation_and_the_acquisitions_reader_hand_on_the_same_names():
    for module in (observed, reader_for_acquire):
        assert module.RuntimeTree is RuntimeTree and module.RUNTIME_TREE_SCHEMA is SCHEMA and module.RUNNER_PATH is RUNNER_PATH
        assert module.runtime_tree_of is runtime_tree_of and module.runtime_tree_from_paths is runtime_tree_from_paths
        assert module.mro_names is mro_names


def test_the_meta_tree_is_the_built_models_with_the_runner_declared_or_copied(monkeypatch):
    """`meta_runtime_tree` with vLLM's three calls stood in, since no CPU machine has vLLM: the tree is the instantiated
    model's, and the runner's facts are declared, with the sampler's class where it is declared a module, or copied."""
    pytest.importorskip("torch")
    from verity_vllm.program.frontend import vllm_meta

    class Model:
        def named_modules(self):
            return [("", self), ("inner", object())]

        def named_buffers(self):
            return [("inner.scale", None)]

    calls: list[object] = []

    def config(**kw):
        calls.append(kw)
        return "the config"

    def instantiate(cfg):
        calls.append(("instantiate", cfg))
        return Model()

    monkeypatch.setattr(vllm_meta, "make_vllm_config", config)
    monkeypatch.setattr(vllm_meta, "ensure_distributed", lambda cfg: calls.append(("distributed", cfg)))
    monkeypatch.setattr(vllm_meta, "instantiate_meta", instantiate)

    bare = observed.meta_runtime_tree("org/model", "rev1", tensor_parallel_size=2, attention_backend="FLASH_ATTN")
    assert calls == [{"model": "org/model", "revision": "rev1", "dtype": "bfloat16", "max_model_len": 2048, "attention_backend": "FLASH_ATTN",
                      "tensor_parallel_size": 2}, ("distributed", "the config"), ("instantiate", "the config")]
    assert bare.modules == {"": ["Model", "object"], "inner": ["object"]} and bare.buffers == ["inner.scale"]
    assert bare.source == "meta:org/model@rev1"
    assert {k: v for k, v in bare.runner.items() if k != "source"} == {"sample": True, "sampler_module": False}
    assert bare.runner["source"].startswith("declared (meta path builds no runner)") and bare.runner["source"].endswith("absent on the pinned vLLM's live engine")

    declared = observed.meta_runtime_tree("org/model", "rev1", sampler_module=True)
    assert {k: v for k, v in declared.runner.items() if k != "source"} == {"sample": True, "sampler_module": True, "sampler_class": ["Sampler"]}
    assert declared.runner["source"].endswith("declared present")

    live = {"sample": True, "sampler_module": False, "source": "engine runner"}
    copied = observed.meta_runtime_tree("org/model", "rev1", sampler_module=True, runner_facts=live)
    assert copied.runner == {"sample": True, "sampler_module": False, "source": "copied from a live engine's tree: engine runner"}
    assert live["source"] == "engine runner" and copied.modules == bare.modules
