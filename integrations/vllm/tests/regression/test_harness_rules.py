"""The harness's own rules, on synthetic rows (laptop-safe: no store, no VERITY_REGRESSION).

What a recorded check (VERITY_STORE_REQUIRED=1) may skip: a check retired on the row, by its reason; never an input or contract that
does not resolve, which fails.  The manifest rebuilds skip by name unless VERITY_REGRESSION_REBUILD=1 (the post-train job).  A query
manifest's rows beyond v1 are its "promoted" facts whatever the engine is called (`v2-query`, `q-word` since S1).
"""
from __future__ import annotations

import json

import pytest

from tests.regression import decisions as D
from tests.regression import resolver
from tests.regression import test_regression as TR
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


def test_an_unresolved_input_skips_but_fails_when_the_store_is_required(monkeypatch):
    monkeypatch.delenv(resolver.STORE_REQUIRED_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception):
        TR._unresolved("x not resolvable here")
    monkeypatch.setenv(resolver.STORE_REQUIRED_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="retired"):
        TR._unresolved("x not resolvable here")


def test_the_rebuilds_skip_unless_asked_for_or_a_candidate_states_the_manifest(monkeypatch):
    monkeypatch.delenv(resolver.REBUILD_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="post_train"):
        TR._skip_rebuild("manifest_digest", None)
    TR._skip_rebuild("manifest_digest", "/some/candidate")
    monkeypatch.setenv(resolver.REBUILD_ENV, "1")
    TR._skip_rebuild("manifest_digest", None)


def test_a_retired_check_skips_by_its_reason_and_only_that_check():
    row = resolver.Row("k", {"row": 9, "retired": {"step_segmentation": {"reason": "descriptors never staged", "since": "abc",
                                                                         "needs": "a re-record"}}}, fixtures={})
    props: list = []
    with pytest.raises(pytest.skip.Exception, match="descriptors never staged"):
        TR._skip_retired(row, "step_segmentation", lambda k, v: props.append((k, v)))
    assert props == [("retired", "descriptors never staged")]
    TR._skip_retired(row, "verdict", lambda k, v: props.append((k, v)))
    assert row.retired("verdict") is None and len(props) == 1


def test_every_retirement_in_the_fixtures_is_well_formed():
    bad = [r for r in D.retirements() if r["problem"]]
    assert not bad, bad


def test_a_malformed_retirement_is_named(monkeypatch):
    fx = {"rows": {"k": {"row": 9, "retired": {"step_segmentation": {"reason": "r"}, "no_such_check": {"reason": "r", "since": "s", "needs": "n"}}}}}
    monkeypatch.setattr(resolver, "load_fixtures", lambda: fx)
    problems = {r["check"]: r["problem"] for r in D.retirements()}
    assert problems["step_segmentation"].startswith("missing") and "no check named" in problems["no_such_check"]


def test_a_registered_fixture_is_found_by_its_path():
    reg = json.loads(resolver.REGISTRY.read_text())
    art, rel = next(iter(reg.items()))
    assert resolver.registered(resolver.REGISTRY.parents[1] / rel) == art
    assert resolver.registered(resolver.EXPECTED_DIR / "no-such-row.json") is None
