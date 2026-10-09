"""Query API (`verity.ml.circuits.query_ast`, `query.parse`, `query_codec`) on small-statics b1 Programs.

Every family is checked for `count == len(list(iter))`, `by_index(i) == list[i]`, `index_of(by_index(i)) == i`; sets are
compared with the intervals the layout gives (`FunctionCall` regions / `Scope` offsets); partition properties are
checked by an interval sweep over the materialised sets (the oracle here -- `partition.py` owns the real validator)."""
from __future__ import annotations

import json

import pytest

from verity.ml.circuits.defs import PrimitiveDefinition, bind
from verity.ml.circuits.layout import FunctionCall
from verity.ml.circuits.parts import Part, n_members
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import b1
from verity.ml.circuits import query_ast as G
from verity.ml.circuits.query_ast import GateSet, QueryError, evaluate
from verity.ml.circuits.query import parse, query
from verity.ml.circuits.query_codec import from_json, query_id, to_json

GEMM = 'members("//Gemm_v1{K=*,N=*}", "@batch:GemmCoordinate_v1")'
ATTN = '''each("//AttentionHead_v3{*}",
         concat(
           each("@call:AttnBlock_v3",
             concat(
               zip("@batch:AmpereBF16TcDot16_v2", by="member"),
               instances("@call:DotBf16_v1"),
               rest())),
           rest()))'''
CANONICAL = 'override(default(32), {"//Gemm_v1{*}": %s, "//AttentionHead_v3{*}": %s})' % (GEMM, ATTN)


@pytest.fixture(scope="module")
def gemm():
    return Program(bind(b1.Gemm, K=32, N=6))


@pytest.fixture(scope="module")
def rms():
    return Program(bind(b1.RMSNormTriton, N=32, EPS=1e-5))


@pytest.fixture(scope="module")
def attn():
    return Program(bind(b1.AttentionHeadV3, T=40, D=16, BN=16))


TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}


@pytest.fixture(scope="module")
def serve():
    """A whole serving program (~37k gates, 40 root nodes): Gemm / AttentionHead activations nested in layers."""
    return Program(bind(b1.ServeV4, C=TINY, LP=3, STEPS=2, BN=16))


@pytest.fixture(scope="module")
def tokensel():
    return Program(bind(b1.TokenSelect, V=8))  # a scan node (ArgmaxStep, 7 iterations)


# -- helpers ------------------------------------------------------------------------------------------------------


def materialise(program, q):
    """Family + list of its sets, with the three consistency laws checked."""
    fam = evaluate(program, q)
    sets = list(fam)
    assert fam.count() == len(sets)
    for i, s in enumerate(sets):
        assert fam.by_index(i) == s, (i, s)
        assert fam.index_of(s) == i, (i, s)
    if sets:
        with pytest.raises(IndexError):
            fam.by_index(len(sets))
    return fam, sets


def sweep(program, sets) -> tuple[bool, bool, int]:
    """(no overlap, covers every gate, covered gate count) over the materialised intervals."""
    ivs = sorted(iv for s in sets for iv in s.intervals())
    cov, overlap = 0, False
    for lo, hi in ivs:
        if lo < cov:
            overlap = True
        cov = max(cov, hi)
    covered = sum(hi - lo for lo, hi in ivs)
    return (not overlap), (not overlap and covered == program.gates), covered


def all_items(spec):
    return frozenset((j, m) for j, node in enumerate(spec.body.nodes) for m in range(n_members(node)))


# -- GateSet -----------------------------------------------------------------------------------------------------


def test_gateset_intervals_size_contains(gemm):
    sc = gemm.circuit.root_scope
    body = sc.spec.body
    jb = [j for j, n in enumerate(body.nodes) if n.form == "batch" and not isinstance(n.fn, PrimitiveDefinition)][0]
    s = GateSet((Part(sc, frozenset({(jb, 1), (jb, 2)})),))
    call1, call2 = FunctionCall(gemm.circuit, sc, jb, 1), FunctionCall(gemm.circuit, sc, jb, 2)
    assert s.intervals() == [(call1.lo, call2.hi)]          # adjacent members merge
    assert s.size() == call1.hi - call1.lo + call2.hi - call2.lo
    assert s.contains(call1.lo) and s.contains(call2.hi - 1) and not s.contains(call2.hi) and not s.contains(call1.lo - 1)
    assert (call1.lo in s) and (call2.hi not in s)
    # equality is by gate content, not Part identity
    s2 = GateSet((Part(sc, frozenset({(jb, 1)})), Part(sc, frozenset({(jb, 2)}))))
    assert s == s2 and hash(s) == hash(s2)
    assert GateSet((Part(sc, frozenset({(jb, 0)})),)) != s


def test_gateset_algebra_same_scope(gemm):
    sc = gemm.circuit.root_scope
    a = GateSet((Part(sc, frozenset({(0, 0), (0, 1), (0, 2)})),))
    b = GateSet((Part(sc, frozenset({(0, 2), (0, 3)})),))
    assert a.union(b).intervals() == [(0, 4)]
    assert a.inter(b).intervals() == [(2, 3)]
    assert a.minus(b).intervals() == [(0, 2)]
    assert b.minus(a).intervals() == [(3, 4)]
    assert a.minus(a).intervals() == [] and a.minus(a).size() == 0


def test_gateset_algebra_nested_scopes(gemm):
    """Minus/inter refine a whole-activation Part along the hierarchy so the result stays Parts."""
    root = evaluate(gemm, "root()").by_index(0)
    _, dots = materialise(gemm, "//DotBf16")
    d0 = dots[1]
    inner = root.minus(d0)
    assert inner.size() == gemm.gates - d0.size()
    assert inner.union(d0) == root
    assert root.inter(d0) == d0 and d0.inter(root) == d0
    assert d0.minus(root).size() == 0
    # parts of the result are Parts of the layout (scope + items), the deepest one in DotBf16's parent body
    assert all(isinstance(p, Part) for p in inner.parts)
    assert any(p.scope.spec.id.startswith("GemmCoordinate") for p in inner.parts)
    # unrelated activations do not interact
    d1 = dots[2]
    assert d0.inter(d1).size() == 0 and d0.minus(d1) == d0


def test_algebra_families(gemm):
    fam, sets = materialise(gemm, 'minus(root(), instances("//DotBf16"))')
    dots = list(evaluate(gemm, "//DotBf16"))
    assert fam.count() == len(dots) and fam.tiling() is None
    for s, d in zip(sets, dots):
        assert s.size() == gemm.gates - d.size() and s.inter(d).size() == 0
    fam2, sets2 = materialise(gemm, 'inter(instances("//GemmCoordinate"), instances("//DotBf16"))')
    assert sets2 == dots
    with pytest.raises(QueryError):
        evaluate(gemm, 'union(instances("//GemmCoordinate"), default(32))')  # 6 vs 230 sets


# -- paths and instances -------------------------------------------------------------------------------------------


def test_instances_match_legacy_query(gemm, attn):
    for program, text in [(gemm, "//GemmCoordinate"), (gemm, "//DotBf16"), (attn, "//AttnBlock"), (attn, "//AttnBlock//DotBf16"),
                          (attn, "//AttnBlock/DotBf16"), (attn, "//AttentionHead_v3{*}//DotBf16"),
                          (gemm, "//GemmCoordinate/DotBf16"), (gemm, "/GemmCoordinate")]:
        legacy = [(c.lo, c.hi) for c in query(program.circuit, text.replace("//AttentionHead_v3{*}", ""))]
        fam, sets = materialise(program, text)
        assert [s.intervals() for s in sets] == [[iv] for iv in legacy], text
        assert fam.tiling() is not None
    # `Name_vN` matches a specialization's base id (the legacy matcher needs the full id with statics)
    assert list(evaluate(attn, "//DotBf16_v1")) == list(evaluate(attn, "//DotBf16"))
    assert list(evaluate(attn, "//DotBf16_v1{K=16}")) == list(evaluate(attn, "//DotBf16"))


def test_statics_wildcards(attn):
    assert evaluate(attn, "//AttnBlock_v3{*}").count() == 3
    assert evaluate(attn, "//AttnBlock_v3{FIRST=True}").count() == 1
    assert evaluate(attn, "//AttnBlock_v3{FIRST=False}").count() == 2
    assert evaluate(attn, "//AttnBlock_v3{FIRST=*,BN=16}").count() == 3
    assert evaluate(attn, "//AttnBlock_v3{BN=64}").count() == 0
    assert evaluate(attn, "//AttnBlock_v3{NVIS=8}").count() == 1  # T=40 = 8 + 16 + 16
    first = evaluate(attn, "//AttnBlock_v3{FIRST=True}").by_index(0)
    blocks = [c for c in query(attn.circuit, "//AttnBlock") if c.fn.statics["FIRST"]]
    assert first.intervals() == [(blocks[0].lo, blocks[0].hi)]
    with pytest.raises(QueryError):
        parse("//AttnBlock_v3{FIRST}")


def test_root_is_an_activation_of_the_program_fn(gemm):
    """`Program(fn)` inlines fn's body into the root, so `//fn` names the root body itself."""
    fam = evaluate(gemm, "//Gemm_v1{K=32,N=6}")
    assert fam.count() == 1 and fam.by_index(0).intervals() == [(0, gemm.gates)]
    assert evaluate(gemm, "//Gemm_v1{K=64}").count() == 0
    assert evaluate(gemm, "/").by_index(0) == evaluate(gemm, "root()").by_index(0)
    fam, sets = materialise(gemm, GEMM)
    coords = list(evaluate(gemm, "//GemmCoordinate"))
    assert sets == coords and fam.count() == 6


def test_members_nodes_zip(gemm, attn):
    # members: every GemmCoordinate member of the Gemm body, k = 1 and k = 2 (adjacent members per set)
    fam1, s1 = materialise(gemm, GEMM)
    fam2, s2 = materialise(gemm, 'members("//Gemm_v1{K=*,N=*}", "@batch:GemmCoordinate_v1", k=2)')
    assert fam2.count() == 3 and all(s2[i] == s1[2 * i].union(s1[2 * i + 1]) for i in range(3))
    fam4, s4 = materialise(gemm, 'members("//Gemm_v1{K=*,N=*}", "@batch:GemmCoordinate_v1", k=4)')
    assert fam4.count() == 2 and s4[1] == s1[4].union(s1[5])  # a short last group
    # nodes: the whole batch node as one set
    famn, sn = materialise(gemm, 'nodes("//Gemm_v1{*}", "@batch:GemmCoordinate_v1")')
    assert famn.count() == 1 and sn[0].intervals() == [(s1[0].intervals()[0][0], s1[-1].intervals()[0][1])]
    # legacy `groups` sugar: k adjacent members of one node
    famg, sg = materialise(gemm, "//Gemm ! groups(GemmCoordinate, 2)")
    assert sg == s2
    # zip inside AttnBlock: member d of every PV k-step batch (BN/16 = 1 batch per block here; D = 16 sets per block)
    blocks = list(evaluate(attn, "//AttnBlock_v3{*}"))
    famz, sz = materialise(attn, 'zip("//AttnBlock_v3{*}", ["@batch:AmpereBF16TcDot16_v2"])')
    assert famz.count() == 16 * len(blocks)
    for b, blk in enumerate(blocks):
        pv = [j for j, n in enumerate(query(attn.circuit, "//AttnBlock")[b].scope().spec.body.nodes)
              if n.form == "batch" and "Ampere" in n.fn.id]
        sc = query(attn.circuit, "//AttnBlock")[b].scope()
        for d in range(16):
            expect = GateSet((Part(sc, frozenset((j, d) for j in pv)),))
            assert sz[b * 16 + d] == expect
            assert blk.inter(expect) == expect
    with pytest.raises(QueryError):  # unequal member counts cannot be zipped
        evaluate(gemm, 'zip("//Gemm_v1{*}", ["@batch:Input16_v1"])').count()


def test_node_selectors(attn):
    blk = query(attn.circuit, "//AttnBlock")[0].scope()
    n_calls = sum(1 for n in blk.spec.body.nodes if n.form == "call")
    n_dots = sum(1 for n in blk.spec.body.nodes if n.form == "call" and n.fn.name == "DotBf16")
    assert evaluate(attn, 'instances("//AttnBlock_v3{FIRST=True}/@form=call")').count() == n_calls
    assert evaluate(attn, 'instances("//AttnBlock_v3{FIRST=True}/@call:DotBf16_v1")').count() == n_dots
    assert evaluate(attn, 'instances("//AttnBlock_v3{FIRST=True}/@fn:DotBf16")').count() == n_dots
    assert evaluate(attn, 'instances("//AttnBlock_v3{FIRST=True}/@batch:DotBf16")').count() == 0
    assert evaluate(attn, 'instances("//AttnBlock_v3{FIRST=True}/@0")').count() == 1
    assert evaluate(attn, 'nodes("//AttnBlock_v3{FIRST=True}", "@*")').count() == len(blk.spec.body.nodes)
    s = evaluate(attn, 'instances("//AttnBlock_v3{FIRST=True}/@0")').by_index(0)
    assert s.intervals() == [(blk.offset, blk.offset + blk.spec.body.nodes[0].gates)]


# -- structural families: default / each / concat / rest / override / where ------------------------------------------


@pytest.mark.parametrize("fixture", ["gemm", "rms", "attn"])
def test_default_32_tiles_every_body(fixture, request):
    program = request.getfixturevalue(fixture)
    fam, sets = materialise(program, "default(32)")
    no_overlap, exact, covered = sweep(program, sets)
    assert exact, (no_overlap, covered, program.gates)
    tl = fam.tiling()
    assert tl is not None and tl.root in tl.bodies
    for body in tl.bodies.values():
        spec = body.spec
        assert body.n_nodes == len(spec.body.nodes) and body.spec_id == spec.id
        cover = {}
        for e in body.entries:
            cover.setdefault(e.node, []).extend(e.members)
            if e.kind == "descend":
                assert e.child in tl.bodies
        for j, node in enumerate(spec.body.nodes):
            n = n_members(node)
            rs = sorted(cover.get(j, []))
            assert rs == [(0, n)] or (n == 0 and not rs), (body.spec_id, j)
    # every set is <= 32 declared bits by construction: primitives and composites with small distinct returns
    ev = G.evaluator(program)
    for body in tl.bodies.values():
        for e in body.entries:
            node = body.spec.body.nodes[e.node]
            if e.kind == "atomic" and not isinstance(node.fn, PrimitiveDefinition):
                assert ev.member_width(node.fn) <= 32


def test_default_descends_wide_composites(attn):
    """AttnBlock returns (m, l[4], o[D]) = 32 * (1 + 4 + 16) bits > 32 -> descended; the root sets never contain one."""
    fam, sets = materialise(attn, "default(32)")
    blocks = list(evaluate(attn, "//AttnBlock_v3{*}"))
    for blk in blocks:
        assert blk not in sets
        inside = [s for s in sets if blk.inter(s) == s]
        assert sum(s.size() for s in inside) == blk.size()
    tl = fam.tiling()
    assert {e.kind for e in tl.bodies[tl.root].entries} == {"atomic", "descend"}


def test_each_concat_rest_compose(attn):
    fam, sets = materialise(attn, ATTN)
    assert sweep(attn, sets)[1]
    # the AttnBlock interior: zip sets (PV chain per d), DotBf16 calls whole, rest singletons
    blocks = query(attn.circuit, "//AttnBlock")
    dots = set(evaluate(attn, "//AttnBlock//DotBf16"))
    zips = set(evaluate(attn, 'zip("//AttnBlock_v3{*}", ["@batch:AmpereBF16TcDot16_v2"])'))
    assert dots <= set(sets) and zips <= set(sets)
    for blk in blocks:
        inside = [s for s in sets if blk.lo <= s.intervals()[0][0] < blk.hi]
        assert sum(s.size() for s in inside) == blk.size
        assert all(s.size() == 1 for s in inside if s not in dots and s not in zips)
    # order: family order is body order; each block's sets are contiguous and in block order
    firsts = [next(i for i, s in enumerate(sets) if blk.lo <= s.intervals()[0][0] < blk.hi) for blk in blocks]
    assert firsts == sorted(firsts)
    # rest(F) alone = the complement of F within the bodies F refines
    famr, rsets = materialise(attn, 'rest(instances("//AttnBlock_v3{*}"))')
    blkset = set(evaluate(attn, "//AttnBlock_v3{*}"))
    assert set(rsets).isdisjoint(blkset)
    assert sum(s.size() for s in rsets) + sum(b.size() for b in blkset) == attn.gates
    famc, csets = materialise(attn, 'concat(instances("//AttnBlock_v3{*}"), rest())')
    assert sweep(attn, csets)[1] and famc.count() == len(blkset) + famr.count()
    with pytest.raises(QueryError):
        evaluate(attn, "rest()")


def test_each_over_the_program_fn_and_gate_items(gemm):
    """each(<program fn>, G) is G in the root body; each over primitive members keeps the gate."""
    fam, sets = materialise(gemm, 'each("//Gemm_v1{*}", concat(nodes(".", "@batch:GemmCoordinate_v1"), rest()))')
    assert sweep(gemm, sets)[1]
    fam2, sets2 = materialise(gemm, 'each("//GemmCoordinate", default(32))')
    assert fam2.count() == 12  # per member: the DotBf16 call (32 bits, atomic) and the cvt gate
    fam3, sets3 = materialise(gemm, 'each(instances("//Gemm_v1{*}/@batch:Input16_v1"), default(32))')
    assert fam3.count() == gemm.input_gates and all(s.size() == 1 for s in sets3)


def test_override_and_canonical_query(gemm, rms, attn):
    for program in (gemm, rms, attn):
        fam, sets = materialise(program, CANONICAL)
        assert sweep(program, sets)[1], program.fn.id
        assert fam.tiling() is not None
    # Gemm root: the override case owns the Gemm body (coordinates), the base keeps the Input nodes
    fam, sets = materialise(gemm, CANONICAL)
    coords = list(evaluate(gemm, GEMM))
    assert fam.count() == gemm.input_gates + len(coords)
    assert sets[-len(coords):] == coords
    # Attention: the AttentionHead override replaces default's treatment of the blocks
    fam_a, sets_a = materialise(attn, CANONICAL)
    fam_b, sets_b = materialise(attn, ATTN)
    assert set(sets_a) == set(sets_b)
    # a case whose G matches nothing keeps the base
    fam_c, sets_c = materialise(rms, 'override(default(32), {"//Gemm_v1{*}": %s})' % GEMM)
    assert sets_c == list(evaluate(rms, "default(32)"))


def test_override_nested_activation(attn):
    """Override addresses bodies matched anywhere, here the AttnBlock calls inside the head."""
    q = 'override(default(32), {"//AttnBlock_v3{*}": concat(nodes(".", "@batch:AmpereBF16TcDot16_v2"), rest())})'
    fam, sets = materialise(attn, q)
    assert sweep(attn, sets)[1]
    whole = set(evaluate(attn, 'nodes("//AttnBlock_v3{*}", "@batch:AmpereBF16TcDot16_v2")'))
    assert whole and whole <= set(sets) and all(s.size() == 16 for s in whole)
    default_sets = set(evaluate(attn, "default(32)"))
    assert default_sets.isdisjoint(whole)  # default keeps the PV batches per member (one gate each)
    assert set(evaluate(attn, "//AttnBlock//DotBf16")) <= default_sets  # F32 return -> atomic in default


def test_where_filters(attn, gemm):
    fam, sets = materialise(attn, 'where(default(32), form="call")')
    dots = set(evaluate(attn, "//DotBf16"))
    assert sets and set(sets) == dots  # the only `call` nodes default keeps atomic are the DotBf16 calls in the blocks
    assert set(sets) <= set(evaluate(attn, "default(32)"))
    fam2, sets2 = materialise(attn, 'where(default(32), spec="DotBf16_v1{*}")')
    assert set(sets2) == dots
    fam3, sets3 = materialise(gemm, 'where(default(32), w_max=16)')
    # Input16 gates (16 bits) and the GemmCoordinate members (bf16 return = 16 bits, 4 gates each)
    assert fam3.count() == gemm.input_gates + 6
    assert sorted(s.size() for s in sets3) == [1] * gemm.input_gates + [4] * 6
    fam4, _ = materialise(gemm, 'where(default(32), w_min=64)')
    assert fam4.count() == 0


def test_canonical_query_on_a_serving_program(serve):
    """Nested activations: the override cases hit Gemm / AttentionHead bodies inside layers; exact partition; every
    atomic set's declared distinct return width <= 32; classes sum to the count."""
    fam = evaluate(serve, CANONICAL)
    n = fam.count()
    assert n > 20_000
    sets = list(fam)
    assert len(sets) == n
    import random
    for i in random.Random(7).sample(range(n), 300) + [0, n - 1]:
        s = fam.by_index(i)
        assert s == sets[i] and fam.index_of(s) == i
    assert sweep(serve, sets)[1]
    ev = G.evaluator(serve)
    tl = fam.tiling()
    assert {b.spec_id.split("{")[0] for b in tl.bodies.values()} >= {"Gemm_v1", "AttentionHead_v3", "AttnBlock_v3", "LayerPre_v1"}
    for body in tl.bodies.values():
        for e in body.entries:
            if e.kind in ("atomic", "whole"):
                node = body.spec.body.nodes[e.node]
                assert ev.member_width(node.fn) * (e.k if e.kind == "atomic" else 1) <= 32, (body.spec_id, e)
    # the Gemm case is what the override installed: every Gemm activation contributes its coordinate members
    coords = list(evaluate(serve, GEMM))
    gemms = list(evaluate(serve, "//Gemm_v1{*}"))
    assert coords and set(coords) <= set(sets)
    assert sum(c.size() for c in coords) == sum(g.size() for g in gemms)
    # classes: one per (body, entry); representatives are members of the family; totals add up
    classes = fam.classes()
    assert sum(c.total for c in classes) == n
    assert all(fam.index_of(c.representative) is not None for c in classes)
    assert any(c.kind == "zip" for c in classes) and any(c.activations > 1 for c in classes)
    # a heads' instances family is not a partition, and its tiling says so (partial root coverage)
    heads = evaluate(serve, "//AttentionHead_v3{*}")
    assert heads.count() == 20 and not sweep(serve, list(heads))[1]
    root = heads.tiling().bodies[heads.tiling().root]
    assert len({e.node for e in root.entries}) < root.n_nodes


def test_scan_iterations(tokensel):
    P = tokensel
    legacy = [(c.lo, c.hi) for c in query(P.circuit, "//ArgmaxStep")]
    fam, sets = materialise(P, 'members("//TokenSelect_v1{*}", "@scan:ArgmaxStep")')
    assert fam.count() == 7 and [s.intervals() for s in sets] == [[iv] for iv in legacy]
    fam3, sets3 = materialise(P, 'members("//TokenSelect_v1{*}", "@scan:ArgmaxStep", k=3)')
    assert fam3.count() == 3 and sets3[2] == sets[6]  # 3 + 3 + 1
    famd, setsd = materialise(P, "default(32)")
    assert sweep(P, setsd)[1] and all(s.size() == 1 for s in setsd)  # the step returns 112 bits -> descended per iteration
    fam200, sets200 = materialise(P, "default(200)")
    assert sweep(P, sets200)[1] and set(sets) <= set(sets200)  # iterations stay whole under a wide budget
    famw, setsw = materialise(P, 'concat(nodes("//TokenSelect_v1{*}", "@form=scan"), rest())')
    assert sweep(P, setsw)[1] and setsw[0].intervals() == [(legacy[0][0], legacy[-1][1])]


def test_tiling_describes_partial_families(attn):
    tl = evaluate(attn, "//AttnBlock_v3{*}").tiling()
    root = tl.bodies[tl.root]
    covered = {e.node for e in root.entries}
    assert covered and len(covered) < root.n_nodes
    assert all(e.kind == "atomic" for e in root.entries)
    tlz = evaluate(attn, ATTN).tiling()
    groups = [e for b in tlz.bodies.values() for e in b.entries if e.kind == "zip"]
    assert groups and all(e.group is not None for e in groups)


# -- parse / codec / query_id ---------------------------------------------------------------------------------------


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
                                       G.each("@call:AttnBlock_v3", G.concat(G.zip_("@batch:AmpereBF16TcDot16_v2"),
                                                                            G.instances("@call:DotBf16_v1"), G.rest())),
                                       G.rest()))})
    assert q == parse(CANONICAL) and query_id(q) == query_id(parse(CANONICAL))


def test_legacy_query_untouched(gemm, attn):
    assert [c.id for c in query(gemm.circuit, "//GemmCoordinate")] == ["GemmCoordinate_v1{K=32}"] * 6
    regs = query(attn.circuit, "//AttnBlock ! groups(DotBf16, 4)")
    assert regs and all(r.size == 4 * regs[0].size // 4 for r in regs)
    assert query(gemm.circuit, "/")[0].size == gemm.gates
