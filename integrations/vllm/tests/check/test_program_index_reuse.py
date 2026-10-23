"""`index.index_of`: an index already built is taken again only for the very instances document it was built from and the arguments
it was built with (`index_key`), so no index is ever served for another Program or another reading of it."""
from __future__ import annotations

import copy

import pytest

from tests.check import test_sampled_replay as T
from verity_vllm.check.replay import index as IX

BASE = {"attention_paths": ["model.model.layers.0.self_attn.attn"], "tp": {"rank": 1, "world": 2}, "program_dir": None}
OTHER = {"attention_paths": {"attention_paths": []}, "rank": {"tp": {"rank": 0, "world": 2}}, "world": {"tp": {"rank": 1, "world": 4}},
         "no tp": {"tp": None}, "program_dir": {"program_dir": "/another/build"}, "query": {"query": T.Q_MODULE_BODY_ID},
         "norm_scales": {"norm_scales": True}, "prefix": {"prefix": "model."}}


@pytest.fixture(scope="module")
def inst():
    return T._program()[0]


def test_an_index_is_taken_again_for_its_own_document_and_arguments(inst):
    pi = IX.ProgramIndex(inst, **BASE)
    assert IX.index_of(inst, [pi], **BASE) is pi
    assert IX.index_of(inst, [pi], **dict(BASE, attention_paths=tuple(BASE["attention_paths"]), tp={"world": 2, "rank": 1})) is pi


@pytest.mark.parametrize("other", sorted(OTHER))
def test_any_other_argument_builds_its_own_index(inst, other):
    pi = IX.ProgramIndex(inst, **BASE)
    kw = dict(BASE, **OTHER[other])
    got = IX.index_of(inst, [pi], **kw)
    assert got is not pi and got.key == IX.ProgramIndex(inst, **kw).key != pi.key


def test_an_equal_document_or_another_default_query_builds_its_own_index(inst, monkeypatch):
    pi = IX.ProgramIndex(inst, **BASE)
    twin = copy.deepcopy(inst)
    assert twin == inst and IX.index_of(twin, [pi], **BASE) is not pi
    monkeypatch.setattr(IX, "DEFAULT_QUERY", T.Q_MODULE_BODY_ID)
    assert IX.index_of(inst, [pi], **BASE) is not pi


def test_a_copied_index_is_never_taken_for_its_original_document(inst):
    """A copy holds a copy of the document under the same arguments: it is no index of the original."""
    cp = copy.deepcopy(IX.ProgramIndex(inst, **BASE))
    assert cp.key == IX.index_key(**BASE) and cp.inst == inst and IX.index_of(inst, [cp], **BASE) is not cp
