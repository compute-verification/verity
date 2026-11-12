"""Two things the Match states and the Build reads: the engine conditions a workload Program models, with the facts of a declared
compilation block (`verity_vllm.check.match.declaration`), and when two module paths name one module
(`verity_vllm.check.match.rank_match.same_module_path`).  `pipeline` hands both on under its own modules' names, and the last
two tests hold what it hands on to what the Match states."""
from __future__ import annotations

import pytest

from verity_vllm.check.match import declaration as GD
from verity_vllm.check.match import rank_match as RM
from verity_vllm.pipeline import global_program as GP
from verity_vllm.pipeline.tp import links

BLOCKS = (
    {"compilation_mode": "3", "cudagraph_mode": "PIECEWISE", "capture_sizes": [1, 2], "enforce_eager": False, "backend": None},
    {"mode": 1, "use_inductor_graph_partition": True, "compile_cache": False, "kernel_set_sha256": "ab"},
    {"backend": "inductor", "call_set_sha256": "cd", "wrapper_norm_set_sha256": "ef", "emulate_precision_casts": True},
    {"enforce_eager": True},
    None,
)
SAME = (
    ("layers.0.o_proj", "layers.0.o_proj"),
    ("model.layers.0.o_proj", "layers.0.o_proj"),
    ("layers.0.o_proj", "wrapper.model.layers.0.o_proj"),
    ("model.lm_head (ParallelLMHead)", "lm_head"),
)
OTHER = ((None, "lm_head"), ("lm_head", None), ("model.xlm_head", "lm_head"), ("layers.0.o_proj", "layers.1.o_proj"))


def test_a_declared_block_gives_its_compilation_facts_under_one_spelling():
    assert GD.compilation_facts(BLOCKS[0]) == {"mode": "VLLM_COMPILE", "cudagraph_mode": "PIECEWISE", "capture_sizes": [1, 2]}
    assert GD.compilation_facts(BLOCKS[1]) == {"mode": "STOCK_TORCH_COMPILE", "use_inductor_graph_partition": True, "compile_cache": False,
                                              "kernel_set_sha256": "ab"}
    assert GD.compilation_facts(BLOCKS[2]) == {"backend": "inductor", "call_set_sha256": "cd", "wrapper_norm_set_sha256": "ef"}
    assert {k for b in BLOCKS[:3] for k in GD.compilation_facts(b)} == set(GD.COMPILATION_FACT_KEYS)


@pytest.mark.parametrize("stated, name", [("0", "NONE"), (2, "DYNAMO_TRACE_ONCE"), ("VLLM_COMPILE", "VLLM_COMPILE")])
def test_a_mode_is_read_as_its_name_whether_stated_as_the_number_or_the_name(stated, name):
    assert GD.compilation_facts({"mode": stated}) == {"mode": name}


@pytest.mark.parametrize("block", [None, "VLLM_COMPILE", [], {}, {"enforce_eager": True}, {"mode": None}])
def test_what_states_no_compilation_fact_gives_none(block):
    assert GD.compilation_facts(block) == {}


@pytest.mark.parametrize("a, b", SAME)
def test_paths_that_are_equal_or_one_the_others_dotted_suffix_name_one_module(a, b):
    assert RM.same_module_path(a, b) and RM.same_module_path(b, a)


@pytest.mark.parametrize("a, b", OTHER)
def test_an_absent_path_or_a_suffix_that_is_no_whole_component_names_another(a, b):
    assert not RM.same_module_path(a, b)


def test_the_build_hands_on_the_engine_conditions_and_facts_the_match_states():
    assert GP.MODELLED_ENGINE == GD.MODELLED_ENGINE and GD.MODELLED_ENGINE["enforce_eager"] is True
    assert [GP.compilation_facts(b) for b in BLOCKS] == [GD.compilation_facts(b) for b in BLOCKS]


def test_the_tensor_parallel_links_compare_paths_as_the_match_does():
    assert [links.same_module_path(a, b) for a, b in SAME + OTHER] == [RM.same_module_path(a, b) for a, b in SAME + OTHER]
