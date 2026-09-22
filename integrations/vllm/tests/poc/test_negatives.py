from __future__ import annotations

import json

import pytest

from verity_vllm import negatives, synthetic
from verity_vllm.negatives import NEGATIVE_IDS, NegativeHarness


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return synthetic.build_toy_run(
        tmp_path_factory.mktemp("toy") / "run", seed=2, chunked_prefill=True
    )


@pytest.fixture(scope="module")
def results(toy):
    harness = NegativeHarness(toy.run_dir, sample_size=10, seed=5)
    return {r.negative_id: r for r in harness.run()}


@pytest.mark.parametrize("nid", NEGATIVE_IDS)
def test_every_negative_rejects_on_the_toy(results, nid):
    r = results[nid]
    assert r.attempted > 0, nid
    assert r.rejected == r.attempted, (nid, [v for v in r.variants if v["outcome"] != "rejected"])
    assert r.unexpectedly_accepted == 0 and r.inconclusive == 0


def test_n02_exercises_the_relation_not_just_authentication(results):
    r = results["N02"]
    kinds = {v["variant"] for v in r.variants}
    assert kinds == {"result", "input_operand"}
    for v in r.variants:
        assert v["status"] == "mismatch"
        if v["variant"] == "result":
            assert v["authentication"]["valid"] == v["authentication"]["attempted"] > 0


def test_reproducers_are_written(results, toy):
    for nid, r in results.items():
        doc = json.loads(open(r.reproducer).read())
        assert doc["negative_id"] == nid and doc["rerun"].startswith(
            "python -m verity_vllm.negatives"
        )
        assert len(doc["variants"]) == r.attempted + r.inconclusive
        assert r.attempted == r.rejected + r.unexpectedly_accepted
    summary = json.loads((toy.run_dir / "negatives" / "negatives.json").read_text())
    assert [x["negative_id"] for x in summary["results"]] == list(NEGATIVE_IDS)
    assert summary["challenge"]["sample"]


def test_table3_row_fields(results):
    row = results["N01"].to_json()
    assert set(row) == {
        "negative_id",
        "attempted",
        "rejected",
        "unexpectedly_accepted",
        "reproducer",
    }


def test_a_vacuous_checker_is_caught_by_the_input_operand_test(toy):
    """15.2: a relation that returns the captured value must not survive N02."""
    from verity_vllm import relations

    def echo(inst, inputs):
        raise relations.RelationError(
            "echo checkers cannot see the output; emulate via registry patch"
        )

    registry = dict(relations.REGISTRY)

    class Echo:
        def __call__(self, inst, inputs):
            # a "checker" closing over the reader: returns whatever was committed
            return toy.reader().value(inst.output_id(toy.profile_id)).data

    registry["residual_add.bf16.v1"] = relations.Relation(
        "residual_add.bf16.v1", "independent", "residual_add", Echo(), False
    )
    sample = [toy.value_id("req-1", 1, "layers.0.residual")]
    harness = NegativeHarness(
        toy.run_dir, sample=sample, registry=registry, out_dir=toy.run_dir / "neg-echo"
    )
    r = harness.n02()
    # the result mutation is caught even by an echo checker only if it reads the mutated capture;
    # this one reads the original, so its prediction differs and 'result' rejects, but the
    # input-operand variant exposes it: the prediction ignores the mutated input.
    variants = {v["variant"]: v["outcome"] for v in r.variants}
    assert variants["input_operand"] == "accepted"
    assert r.unexpectedly_accepted >= 1
    del echo


def test_cli_runs(toy, capsys):
    code = negatives.main([str(toy.run_dir), "--only", "N01,N07", "--sample-size", "4"])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 0 and [json.loads(line)["negative_id"] for line in out] == ["N01", "N07"]
    assert negatives.main([str(toy.run_dir / "missing")]) == 2
