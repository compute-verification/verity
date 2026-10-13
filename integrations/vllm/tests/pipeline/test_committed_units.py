from __future__ import annotations

import json
from types import SimpleNamespace

from verity_vllm.pipeline import committed_units as C
from verity_vllm.pipeline import row_records as R


def _record():
    return C.record([{
        "input": "committed_0", "op": "aten.sin.default", "module": "block", "reason": "test gap",
        "list_entry": {"op": "aten.sin.default", "module": "block", "reason": "test gap"},
    }])


def test_build_summary_carries_units_and_manifest_guard_refuses_them(tmp_path):
    for wrapper in ("step", "request"):
        out = tmp_path / f"build_{wrapper}"
        out.mkdir()
        (out / "result.json").write_text(json.dumps({
            "ok": True, "program": {"digest": wrapper * 8}, "committed_units": _record(),
        }))
    assert R.build_summary(str(tmp_path)) == "PASS"
    summary = json.loads((tmp_path / "build_summary.json").read_text())
    assert summary["request"]["committed_units"]["inputs"] == ["committed_0"]

    manifest = {"query": {}, "program_digest": "digest", "committed_units": _record()}
    slug, reason = R.manifest_not_of_record(manifest, "digest", SimpleNamespace())
    assert slug == "unverified-committed-values"
    assert "approval has not been recorded" in reason


def test_tp_links_reconcile_listed_collectives_without_asserting_peer_links(tmp_path):
    from tests.pipeline import tp_toy as T
    from verity_vllm.pipeline.tp.links import write_tp_links
    from verity_vllm.program.frontend.derive import derive_report
    from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset

    world, rank = 3, 1
    mod, inputs, probe, _inner = T.rank_source(world, rank)
    rep = derive_report(mod, inputs, mode="profile", profile="b1-eager-v3", rules=vllm_ruleset(), name="vLLMStep[ToyRank]",
                        check_registry=True, rank=(world, rank), committed_units=[{
                            "op": "verity_tp.all_reduce.default", "module": "*", "reason": "test-only collective gap",
                        }])
    assert rep.ok, rep.unsupported
    result = {"program": {"digest": "test-digest"}, "tp": {"collectives": {"record": probe}}}
    summary = write_tp_links(str(tmp_path), result, rep.program, rep, world, rank)
    links = json.loads((tmp_path / "tp_links.json").read_text())

    assert [link["k"] for link in links["links"]] == [2]
    assert [link["k"] for link in links["committed_collectives"]] == [0, 1]
    assert links["unverified_inputs"] == ["committed_0", "committed_1"]
    assert len(links["peer_inputs"]) == 3 * (world - 1)
    assert links["unused_peer_inputs"] == links["peer_inputs"][:2 * (world - 1)]
    assert links["prompt_inputs"] == ["prompt"]
    assert summary == {
        "n_peer_inputs": 6, "n_prompt_inputs": 1, "file": "tp_links.json",
        "n_unverified_inputs": 2, "n_committed_collectives": 2, "n_unused_peer_inputs": 4,
    }
