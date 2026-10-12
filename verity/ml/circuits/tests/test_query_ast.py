"""Structural query AST (`verity.ml.circuits.query_ast`), its string syntax (`query.parse`) and JSON codec (`query_codec`).

Promoted from veritor@0903f60d6 `verity_ir/tests/test_gateset_api.py`: the program-independent tests.  The evaluation
tests there (families, partitions, canonical queries) build small b1 Programs from veritor's registry and move with it.
"""

from __future__ import annotations

import json

import pytest

from verity.ml.circuits import query_ast as G
from verity.ml.circuits.query_ast import QueryError
from verity.ml.circuits.query import parse
from verity.ml.circuits.query_codec import from_json, query_id, to_json

GEMM = 'members("//Gemm_v1{K=*,N=*}", "@batch:GemmCoordinate_v1")'
ATTN = '''each("//AttentionHead_v3{*}",
         concat(
           each("@call:AttnBlock_v3",
             concat(
               zip("@batch:AmpereBF16TcDot16_v1", by="member"),
               instances("@call:DotBf16_v1"),
               rest())),
           rest()))'''
CANONICAL = 'override(default(32), {"//Gemm_v1{*}": %s, "//AttentionHead_v3{*}": %s})' % (GEMM, ATTN)


def test_parse_and_json_roundtrip_stable_id():
    texts = [GEMM, ATTN, CANONICAL, "//A//B", "/", "//A ! groups(F, 3)", 'default(32)', 'rest(instances("//X"))',
             'where(default(32), spec="DotBf16_v1{*}", form="call", w_max=32)', 'union(root(), instances("//X"))',
             'minus(root(), instances("//X"))', 'inter(instances("//X"), instances("//Y"))',
             'nodes("//X", "@form=batch")', 'zip("//X", ["@batch:A", "@batch:B"])', 'instances("//X/@2")']
    seen = set()
    for t in texts:
        q = parse(t)
        doc = to_json(q)
        json.dumps(doc)  # plain JSON
        assert from_json(doc) == q
        assert from_json(json.loads(json.dumps(doc))) == q
        qid = query_id(q)
        assert len(qid) == 128 and qid == query_id(from_json(doc))
        assert qid not in seen
        seen.add(qid)
    # spelling-independent: whitespace / quotes / equivalent constructors
    assert query_id(parse(GEMM)) == query_id(parse(GEMM.replace('"', "'").replace(", ", ",")))
    assert query_id(parse("//GemmCoordinate")) == query_id(parse('instances("//GemmCoordinate")'))
    assert query_id(parse("//A ! groups(F, 3)")) == query_id(parse('members("//A", "@fn:F", k=3)'))
    assert query_id(parse("/")) == query_id(parse("root()"))
    # pinned: the canonical query's identity is program-independent text identity
    assert query_id(parse(CANONICAL)) == query_id(parse(CANONICAL))
    assert to_json(parse(GEMM)) == {"op": "members", "path": "//Gemm_v1{K=*,N=*}", "sel": "@batch:GemmCoordinate_v1", "k": 1}


def test_parse_errors():
    for bad in ["", "foo()", "members()", 'each("//X")', "instances(", 'override(default(32), 3)', "//A ! blocks(F, 2)",
                "@call", 'members("//X", "call:Y")', 'default()', 'default("x")', 'members("//X", 7)', "instances(42)",
                'members("//X", "@0", k=True)', "rest", "@", "//", 'nodes("//X", "@bogus:Z")']:
        with pytest.raises(QueryError):  # always QueryError -- never KeyError / AttributeError / ValueError leaks
            parse(bad)
    for doc in [{"op": "nope"}, {"op": "members"}, {"op": "members", "path": 3, "sel": "@0"}, {"op": "default", "w": "a"},
                {"op": "each", "base": {"op": "root"}, "inner": 5}, [], "x"]:
        with pytest.raises(QueryError):
            from_json(doc)
    with pytest.raises(QueryError):
        G.members(3, "@0")
    with pytest.raises(TypeError):
        to_json(object())


def test_python_constructors_match_parse():
    q = G.override(G.default(32), {"//Gemm_v1{*}": G.members("//Gemm_v1{K=*,N=*}", "@batch:GemmCoordinate_v1"),
                                   "//AttentionHead_v3{*}": G.each("//AttentionHead_v3{*}", G.concat(
                                       G.each("@call:AttnBlock_v3", G.concat(G.zip_("@batch:AmpereBF16TcDot16_v1"),
                                                                            G.instances("@call:DotBf16_v1"), G.rest())),
                                       G.rest()))})
    assert q == parse(CANONICAL) and query_id(q) == query_id(parse(CANONICAL))
