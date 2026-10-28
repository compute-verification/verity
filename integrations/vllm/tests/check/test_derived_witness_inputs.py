"""A TP rank Program's WITNESS inputs in the Match (`program_compare.load_derived`): a user input the link table lists neither among
`prompt_inputs` nor `peer_inputs` -- DeepSeek-V4-Pro's MoE align `schedule` (`tp_links.json` `schedule_input`), and any other -- is bound
by its name, so a read of it canonicalises as ("WIT", name, lo, hi) in the record and the compact form alike, never as an unnamed
weight (`("W", "input#<k>", ..)`, which matched any input at that position)."""
from __future__ import annotations

import gzip
import json

import pytest

from verity.ml.circuits import codec
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.check.match import program_compare as CMP
from verity_vllm.check.match.canonical_operands import as_progression
from verity.ml.definitions.registry import b1, b1_tp2
from verity.ml.definitions.registry import prims as P

H, V = 8, 8
BF16, I32 = Value(16), Value(32)
USER = ["prompt", "schedule", "aux", "peers_0"]


def _rank_program():
    params = (("prompt", Array(1, I32)), ("schedule", Array(4, I32)), ("aux", Array(2, I32)), ("peers_0", Array(1, Array(H, BF16))),
              ("embed", Array(V, Array(H, BF16))))
    ret = Array(3, I32)

    def body(B, S, prompt, schedule, aux, peers, embed):
        e = B.call(bind(b1_tp2.EmbeddingShard, VS=V, H=H, START=0), prompt[0], embed)
        h = B.call(bind(b1_tp2.AllReduce2, N=H), array_of([e, peers[0]]))
        t = B.call(bind(b1.TokenSelect, V=H), h)
        keys = B.call(P.I32Add, schedule[1], schedule[2])
        lens = B.call(P.I32Add, aux[0], keys)
        return Coll(ret, concat([t.refs, keys.refs, lens.refs]))

    return bind(CompositeDefinition("ToyWitnessRank", 1, (), lambda S: (params, ret), body, register=False))


def _derived(tmp_path, links: dict) -> str:
    with gzip.open(tmp_path / "descriptor.json.gz", "wt") as f:
        json.dump(codec.encode_program(Program(_rank_program())), f)
    (tmp_path / "derive-report.json").write_text(json.dumps({"provenance": {"source": {"example_inputs": [[0]] * len(USER)},
                                                                            "weights": [{"field": 0, "name": "model.embed_tokens.weight"}]}}))
    (tmp_path / "tp_links.json").write_text(json.dumps({"schema": "cb-a/tp_links/v2", "world": 2, "rank": 0, "user_inputs": USER,
                                                        "prompt_inputs": ["prompt"], "peer_inputs": ["peers_0"], "links": [], **links}))
    return str(tmp_path)


def _adds(p: CMP.Prog) -> list[list[list[tuple]]]:
    return [p.canon(i) for i, r in enumerate(p.inst) if r["spec"].startswith("I32Add")]


def test_each_witness_input_is_bound_by_its_name(tmp_path):
    p = CMP.load_derived(_derived(tmp_path, {"schedule_input": "schedule"}), "derived")
    assert p.witness_fields == {1: "schedule", 2: "aux"} and p.peer_fields == {3: [(0, 0)]}
    keys, lens = _adds(p)[-2:]
    assert keys == [[("WIT", "schedule", 1, 2)], [("WIT", "schedule", 2, 3)]]
    assert lens[0] == [("WIT", "aux", 0, 1)] and lens[1][0][0] == "N"
    assert not [t for i in range(len(p.inst)) for op in p.canon(i) for t in op if t[0] == "W" and str(t[1]).startswith("input#")]


def test_the_compact_form_reads_a_witness_as_the_record_does(tmp_path):
    d = _derived(tmp_path, {"schedule_input": "schedule"})
    rec, comp = CMP.load_derived(d, "derived"), CMP.load_derived(d, "derived", compact=True)
    assert [[[as_progression(t) for t in op] for op in ops] for ops in _adds(rec)[-2:-1]] == _adds(comp)[-2:-1]
    assert _adds(comp)[-1][0] == [("WIT", "aux", (0, 1), (0, 0), 1)]


def test_a_link_table_naming_another_count_of_user_inputs_is_refused(tmp_path):
    d = _derived(tmp_path, {})
    (tmp_path / "tp_links.json").write_text(json.dumps({"world": 2, "rank": 0, "user_inputs": USER[:3], "prompt_inputs": ["prompt"], "links": []}))
    with pytest.raises(ValueError, match="names 3 user inputs, derive-report has 4"):
        CMP.load_derived(d, "derived")
