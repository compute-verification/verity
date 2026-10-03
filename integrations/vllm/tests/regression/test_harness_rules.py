"""The harness's own rules, on synthetic rows (laptop-safe: no store, no VERITY_REGRESSION).

A query manifest's rows beyond v1 are its "promoted" facts whatever the engine is called (`v2-query`, `q-word` since S1).
"""
from __future__ import annotations

import pytest

from tests.regression.checks import manifest_digest as MD


class _Fake:
    def __init__(self, docs: dict):
        self.docs = docs

    def has(self, rel: str) -> bool:
        return rel in self.docs

    def json(self, rel: str):
        return self.docs[rel]


def _ident(op: str, member: str = "out", step: int = 0) -> dict:
    return {"program_digest": "p", "request_id": "r0", "step": step, "invocation": 0, "op_path": op, "output_member": member,
            "element_range": [0, 4], "rank": 0, "family": "instance_outputs"}


@pytest.mark.parametrize("engine", ["q-word", "v2-query"])
def test_a_query_manifests_rows_beyond_v1_are_its_promoted_facts(engine):
    v1 = [_ident("model.layers.0.attn")]
    new = v1 + [_ident("model.layers.0.mlp.experts"), _ident("model.layers.1.mlp.experts")]
    got = MD._promotion(_Fake({"manifest.json": {"identities": v1}}), {"query": {"engine": engine}, "identities": new})
    assert got["promoted"] == got["added_n"] == 2 and got["removed_n"] == 0
    assert got["promoted_member_class"] == "model.layers.<k>.mlp.experts/out"
    assert got["promoted_class_counts"] == {"model.layers.<k>.mlp.experts/out": 2}


def test_a_manifest_no_query_engine_built_states_only_its_flagged_rows():
    v1 = [_ident("model.layers.0.attn")]
    new = v1 + [dict(_ident("model.layers.0.mlp.experts"), promoted=True), _ident("model.norm")]
    got = MD._promotion(_Fake({"manifest.json": {"identities": v1}}), {"identities": new})
    assert got["promoted"] == 1 and got["added_n"] == 2 and "promoted_class_counts" not in got
