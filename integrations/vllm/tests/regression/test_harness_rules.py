"""The harness's own rules, on synthetic rows (laptop-safe: no store, no VERITY_REGRESSION).

What a recorded check (VERITY_STORE_REQUIRED=1) may skip: a check retired on the row, by its reason; never an input or contract that
does not resolve, which fails.  The checks that recompute from a row's Programs (manifest_digest, step_segmentation, the #57 negative)
skip unless VERITY_REGRESSION_REBUILD=1 (the post-train job), which selects them by their `rebuild` mark.  A query
manifest's rows beyond v1 are its "promoted" facts whatever the engine is called (`v2-query`, `q-word` since S1).  A post-flip reference
row (no `commit_summary.json`) is summarised from its own evidence.  A row's `manifest_flags` replaces POLICIES_OFF.  A merged
Programs view outlives the trees it was assembled from, and a row's tree evicted under the suite is fetched again.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil

import pytest

from tests.regression import decisions as D
from tests.regression import resolver
from tests.regression import test_regression as TR
from tests.regression.checks import commit_summary as CS
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


def test_the_post_train_job_selects_exactly_the_checks_that_skip_in_check():
    """post_train.py runs `-m "regression and rebuild"`: every `REBUILD` check carries the mark and nothing else does, so what
    `check` skips is what the job runs."""
    from tests.regression import checks
    marked = {p.id.split("-")[1] for p in TR._params() if any(m.name == "rebuild" for m in p.marks)}
    assert marked == {c.NAME for c in checks.all_checks() if getattr(c, "REBUILD", False)} >= {"manifest_digest", "step_segmentation"}
    assert any(m.name == "rebuild" for m in TR.test_negative_57_refusal_reproduces.pytestmark)


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


def _post_flip_row(tmp_path, monkeypatch, *, rc: int | None):
    d = tmp_path / "rows" / "k"
    (d / "commit").mkdir(parents=True)
    (d / "commit" / "summary.json").write_text(json.dumps({"committer": "native"}))
    runs = [{"arm": "control", "wall_s": 1.0, "engine_steps": 3}, {"committer": "native", "tokens_equal": True, "engine_steps": 3}]
    (d / "commit" / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    (d / "commit.log").write_text("$ commit --max-num-seqs 8 --pairs 1\n")
    (d / "verdict.json").write_text(json.dumps({"outcome": "PASS", "compat": {} if rc is None else {"rc": rc}}))
    monkeypatch.setenv(resolver.ROWS_ROOT_ENV, str(tmp_path / "rows"))
    return resolver.Row("k", {"row": 9}, fixtures={})


def test_a_post_flip_reference_row_is_summarised_from_its_own_evidence(tmp_path, monkeypatch):
    row = _post_flip_row(tmp_path, monkeypatch, rc=None)
    assert CS.applies(row) is None
    exp, act = CS.expected(row), CS.actual(row)
    assert act["source"] == "recomputed" and resolver.diff(exp, act) == []
    assert act["rc"] == 0 and act["pass"] is True and act["engine_settings"]["max_num_seqs"] == 8 and act["n_treatment"] == 1


def test_a_post_flip_rows_rc_is_its_verdicts(tmp_path, monkeypatch):
    row = _post_flip_row(tmp_path, monkeypatch, rc=3)
    assert CS.actual(row)["rc"] == 3 and CS.actual(row)["pass"] is False


def test_a_row_without_a_summary_or_a_verdict_of_record_is_not_summarised(tmp_path, monkeypatch):
    row = _post_flip_row(tmp_path, monkeypatch, rc=0)
    (tmp_path / "rows" / "k" / "verdict.json").unlink()
    assert CS.applies(row) is not None


def test_a_merged_programs_view_outlives_the_trees_it_links_from(tmp_path, monkeypatch):
    """The store's trees/ is a cache that eviction may clear while a rebuild reads the view (#11's rebuild failed on its view's
    result.json in r20261003-173418-89e5): the view holds hard links, so its files stay readable when their trees go."""
    data = {"build_request/instances.json.gz": gzip.compress(b"[]"), "build_request/result.json": b"{}"}
    programs, records = tmp_path / "programs" / "k", tmp_path / "records"
    for root, rel in ((programs, "build_request/instances.json.gz"), (records, "build_request/result.json")):
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_bytes(data[rel])
    files = {rel: {"role": "record", "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for rel, v in data.items()}
    monkeypatch.setenv(resolver.ROWS_ROOT_ENV, str(tmp_path / "programs"))
    monkeypatch.setenv(resolver.SCRATCH_ENV, str(tmp_path / "scratch"))
    view = resolver.Row("k", {"row": 9, "local": {"m": str(records)}, "files": files}, fixtures={}).programs_root()
    assert view is not None and view.is_relative_to(tmp_path / "scratch")
    shutil.rmtree(tmp_path / "programs")
    shutil.rmtree(records)
    assert gzip.decompress((view / "build_request/instances.json.gz").read_bytes()) == b"[]"
    assert (view / "build_request/result.json").read_bytes() == b"{}"


def test_a_store_tree_evicted_under_the_suite_is_fetched_again(tmp_path, monkeypatch):
    """#73's executed_prefix found its tokens.json in `applies` and not in `actual` when node 1's hourly eviction cleared trees/
    between them (r20261003-174531-333b): a row's fetched tree is fetched again once it is gone."""
    from tests.regression import store_io
    fetches = []

    def fetch(art):
        fetches.append(art)
        (tmp_path / art / "match").mkdir(parents=True, exist_ok=True)
        (tmp_path / art / "match" / "tokens.json").write_text("{}")
        return tmp_path / art
    monkeypatch.setattr(store_io, "fetch", fetch)
    row = resolver.Row("k", {"row": 9, "artifacts": {"record": "art:t"}}, fixtures={})
    assert (row._store_root("record") / "match" / "tokens.json").is_file()
    shutil.rmtree(tmp_path / "art:t")
    assert (row._store_root("record") / "match" / "tokens.json").is_file() and fetches == ["art:t", "art:t"]
    assert row._store_root("record") and fetches == ["art:t", "art:t"], "a tree still there is not fetched again"


def test_a_rows_manifest_flags_replace_the_policies_off():
    assert MD._flags(resolver.Row("k", {"row": 9}, fixtures={})) == MD.POLICIES_OFF
    assert MD._flags(resolver.Row("k", {"row": 9, "manifest_flags": []}, fixtures={})) == []
