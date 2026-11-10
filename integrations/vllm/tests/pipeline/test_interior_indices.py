"""A two-stage interior whose replay block passes its units indices the verifier supplies (`interior_rows`, ``supplies``
with an ``index``): a replay block of three `A` units, unit j reading row j of its parameter ``c``, and two batches of two
`B` units, each batch reading rows 0 and 1 of its parameter ``i``, so `B`'s unit j reads row j mod 2, over a replay sample
of two of three spans. The interior commits no rows of its own at an index port: they are rows of the supplied values,
named as round 2 names the block's ``c`` and ``i``, which the verifier fills from the challenger's ``bytes`` sources. A
replayed unit whose index row is not the value's at its place is refused before anything is committed, for each index."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import partition as P, registered as RG, registration as R
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import dead_free as DF
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import interior_rows as IR, serving_rows as SRW

#: two templates of one shape, (x0, best, idx: one I32) -> I32: `A` stands for the unit reading ``c``, `B` for ``i``'s
A, B = bind(DF.TokenSelectCombineV2, V=64, C=64), bind(DF.TokenSelectCombineV2, V=32, C=32)
SPANS, Q, SALT, ROW, DELTA = 3, 2, bytes(range(32)), "interior-indices-test", 2.0 ** -40
#: `A`'s units a span, the rows of ``i`` a span, and the batches of `B` that each read them
NC, W, G = 3, 2, 2
CELLS = (NC, G * W)


def _block():
    """Each span a batch of `NC` `A` units on its own parameters, the j-th reading ``c``'s row j, and `G` batches of `W`
    `B` units, each batch reading ``i``'s `W` rows."""
    (ap, ar), (bp, br) = A.signature(), B.signature()
    params = (*((n, Array(NC, t)) for n, t in (ap[0], ap[1])), ("c", Array(NC, ap[2][1])),
              *((f"{n}{g}", Array(W, t)) for g in range(G) for n, t in (bp[0], bp[1])), ("i", Array(W, bp[2][1])))

    def body(B_, S_, x0, best, c, *rest):
        return tuple_of(B_.batch(A, x0, best, c, axes=(0, 0, 0)),
                        *(B_.batch(B, rest[2 * g], rest[2 * g + 1], rest[-1], axes=(0, 0, 0)) for g in range(G)))

    return bind(CompositeDefinition("InteriorIndicesTest", 0, (),
                                    lambda S_: (params, Tuple(Array(NC, ar), *(Array(W, br) for _ in range(G)))), body,
                                    register=False))


BLOCK = _block()
KEY = descriptor_id(BLOCK)
#: the indices, by block parameter: ``c``, a row a unit of `A`; ``i``, `W` rows, unit j of `B` reading row j mod `W`
SUPPLIES = (IR.Supply(P.template_id(A), "idx", f"{KEY}/c", NC, tuple(range(NC)), tuple(range(NC))),
            IR.Supply(P.template_id(B), "idx", f"{KEY}/i", W, tuple(j % W for j in range(G * W)), tuple(range(W))))
MEMBERS = tuple(IR.named(SRW.unit_member(P.template_id(t), n)) for t, n in zip((A, B), CELLS, strict=True))


def _u32(xs) -> np.ndarray:
    return np.asarray(xs, dtype="<u4").view("<u2").reshape(len(xs), -1)


def _spans(seed: int) -> tuple[IR.Span, IR.Span]:
    """One span's `A` units and `B` units, each its Definition's on random bf16 words and the index its place gives it."""
    rng = np.random.default_rng(seed)
    got = []
    for fn, m, x in zip((A, B), MEMBERS, SUPPLIES, strict=True):
        idx = [x.index[a] for a in x.at]
        x0, best = rng.integers(0, 1 << 16, m.n).tolist(), rng.integers(0, 1 << 16, m.n).tolist()
        toks = [evaluate(fn, a, [b], [j])[0] for a, b, j in zip(x0, best, idx, strict=True)]
        data = {"rows": {"x0": np.asarray(x0, np.uint16)[:, None], "best": np.asarray(best, np.uint16)[:, None],
                         "idx": _u32(idx)}, "outs": {"o0": _u32(toks)}}
        got.append(IR.Span(f"s{seed}", m, data, [[0, m.n]], "p" * 64))
    return got[0], got[1]


def _boundary_window() -> CS.Binding:
    return CS.Binding(P.population_program(BLOCK, SPANS), P.template_instance_query(BLOCK), {"law": "subset", "k": Q},
                      "b0" * 64, (CS.Member(P.template_id(BLOCK), (("ingress", 64),), (("egress", 64),), SPANS),))


def _first():
    """A window stream over `A` and `B` taking the indices' values (`interior_rows.supplied_values`), a stand-in boundary
    registered and its first stage drawn (spans 0 and 2)."""
    names = {}
    values = IR.supplied_values(SUPPLIES, Q)
    for phase, w in (("serve", _boundary_window()), ("interior", IR.interior_window(MEMBERS, CELLS, "00", [0], values=values))):
        names |= {m.port(p): (m.schema(p), phase) for m in w.members for p in CS.port_leaves(m)}
    supplied = tuple(x.value for x in SUPPLIES)
    public = (S.Public("population", "structure", "N, the sample's range"),
              S.Public("window", "structure", "provenance: served, on test values"),
              S.Public("leaf_layer", "structure", "the members' public bodies' digest"),
              *(S.Public(f"{k}:{p}", "structure", f"{p}'s layout") for p in sorted(names) for k in ("leaves", "domain")),
              *(S.Public(f"{k}:{v}", "structure", f"{v}'s layout") for v in supplied for k in ("leaves", "words", "domain")))
    spec = S.Spec(tuple(S.Value(p, s, phase) for p, (s, phase) in sorted(names.items()))
                  + tuple(S.Value(v, None, "interior", registered=True) for v in supplied),
                  S.TwoStage(P.template_instance_query(BLOCK), f"rate:{Q}/{SPANS}", 1),
                  S.Check(tuple(P.program_digest(Program(u)) for u in (A, B)), P.template_instances_query((A, B)),
                          proof_deadline=60.0), public, b"interior-indices-test")
    draws = iter([[0, 2]])
    stream = S.Stream(spec, (A, B), values=values, replay=BLOCK, budget=1, delta=DELTA, challenger="challenger-test",
                      clock=lambda: 0.0,
                      sampler=lambda law, n: next(draws) if law["law"] == "subset" else [0] * len(law["strata"]))
    window = _boundary_window()
    c = CS.commit(window, [{"rows": {"ingress": np.zeros((SPANS, 64), np.uint16)},
                            "outs": {"egress": np.ones((SPANS, 64), np.uint16)}}], salt_key=SALT)
    reg = stream.register(CS.registration(c, {"stand_in": "boundary"}, None, registrar=R), CS.expected_ports(window),
                          leaf_layer=CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)}))
    return stream, reg, stream.select(reg)


@pytest.fixture(scope="module")
def staged():
    return _first()


def _interior(reg, first, per, **kw) -> IR.Interior:
    return IR.interior(per, CELLS, reg.digest, first.units, row=ROW, salt_key=SALT, **{"supplies": SUPPLIES, **kw})


def test_an_index_port_is_rows_of_the_value_the_verifier_fills_from_its_own_bytes(staged):
    """The interior registers the two supplied values, the block's ``c`` and ``i`` as round 2 names them, Q·per rows each,
    row ``s·per + r`` the words ``[r, 0]``. The index ports are given by reference: `A`'s unit j of sampled span s reads
    ``c``'s row ``s·NC + j`` and `B`'s reads ``i``'s row ``s·W + j mod W``, where round 2 holds a linked port's read of an
    interior value. From the challenger's ``bytes`` sources alone the verifier fills the same rows
    (`registered.supplied_rows` on the stream's own interior program), and those rows at the public salts under the
    port's row format commit to the entry the interior registers. The stream takes the interior's roots and names; it
    refuses the supplied values' rows (`R7-supplied`) alone, its verifier committing a supplied value's rows under
    row/v1 where the ports read row/v2."""
    stream, reg, first = staged
    it = _interior(reg, first, [_spans(s) for s in first.units])
    c, w = it.committed, it.committed.window
    prog = P.program_digest(w.program)
    assert P.program_digest(stream.interior_program(Q)) == prog
    got = {e["value"]: e for e in it.record["registered"]}
    assert list(got) == [x.value for x in SUPPLIES] == [f"{KEY}/c", f"{KEY}/i"]
    at = [c.given[m.port("idx")].positions for m in w.members]
    assert at == [(0, 1, 2, 3, 4, 5), (0, 1, 0, 1, 2, 3, 2, 3)]
    srcs = IR.sources(SUPPLIES, Q)
    assert srcs == {f"{KEY}/c": [[{"bytes": f"{r:08x}"}] for _ in range(Q) for r in range(NC)],
                    f"{KEY}/i": [[{"bytes": f"{r:08x}"}] for _ in range(Q) for r in range(W)]}
    for x, m in zip(SUPPLIES, w.members, strict=True):
        mine = RG.supplied_rows(stream.interior_program(Q), x.value, RG.sources(srcs)[x.value], it.record)
        assert mine == [[r, 0] for _ in range(Q) for r in range(x.per)]
        assert [list(r) for r in c.given[m.port("idx")].value.rows] == mine
        v = RG.Registered.commit(prog, x.value, mine, RG.supplied_salts(prog, x.value, len(mine)), schema=m.schema("idx"))
        assert v.entry() == {k: got[x.value][k] for k in v.entry()}
    assert IR.supplied_words(SUPPLIES) == {f"{KEY}/c": 2, f"{KEY}/i": 2}
    with pytest.raises(S.Refused) as refused:
        IR.register(stream, first, it, registered=IR.supplied_words(SUPPLIES), supplied=srcs)
    assert refused.value.verdict.codes == ("R7-supplied",)


@pytest.mark.parametrize("member, unit, index, why", [
    (0, 1, 2, f"unit 4 \\(sampled span 1\\) takes at its 'idx' a row that is not row 4 of {KEY}/c, the value the verifier"),
    (0, 2, 7, f"unit 5 \\(sampled span 1\\) takes at its 'idx' a row that is not row 5 of {KEY}/c, the value the verifier"),
    (1, 3, 0, f"unit 7 \\(sampled span 1\\) takes at its 'idx' a row that is not row 3 of {KEY}/i, the value the verifier"),
    (1, 0, 2, f"unit 4 \\(sampled span 1\\) takes at its 'idx' a row that is not row 2 of {KEY}/i, the value the verifier"),
])
def test_an_index_row_the_interior_does_not_register_is_refused_before_anything_is_committed(staged, member, unit, index, why):
    """For ``c`` and for ``i``: a replayed unit whose index is another row's of its value (`A`'s unit 1 reading 2, `B`'s
    unit 3 reading 0) or no row's at all (7 of ``c``'s three, 2 of ``i``'s two) takes a row the interior does not
    register at its place, so the interior refuses it by name and commits nothing."""
    _stream, reg, first = staged
    per = [list(_spans(s)) for s in first.units]
    s = per[1][member]
    idx = s.data["rows"]["idx"].copy()
    idx[unit] = _u32([index])[0]
    per[1][member] = replace(s, data={**s.data, "rows": {**s.data["rows"], "idx": idx}})
    with pytest.raises(ValueError, match=why):
        _interior(reg, first, per)


def test_a_supply_names_a_port_of_the_window_and_places_each_of_its_units(staged):
    """A `Supply` is refused by name for a unit placed past its rows, indices that are not one a row, a port that holds no
    table's root or domain and no indices; and an interior for a template it lacks, a port that is none of its rows, a
    placement of other than its template's units a span, a port supplied twice, and a commitment made on the rows'
    device."""
    _stream, reg, first = staged
    c, i = SUPPLIES
    for bad, why in ((dict(at=(0, 1, NC)), "units reading rows"), (dict(index=(0, 1)), "2 indices for its 3 rows"),
                     (dict(index=None), "is no table read's root or domain")):
        with pytest.raises(ValueError, match=why):
            replace(c, **bad)
    per = [_spans(s) for s in first.units]
    for bad, why in (((replace(c, template="Elsewhere_v1"), i), "a template the window has none of"),
                     ((replace(c, port="o0"), i), "which is none of its rows"),
                     ((replace(c, port="best"), i), "row is 1 words, not"),
                     ((c, replace(i, at=(0, 1), index=(0, 1))), "places 2 units, not its 4"),
                     ((c, i, replace(c, value=f"{KEY}/again")), "is supplied twice")):
        with pytest.raises(ValueError, match=why):
            _interior(reg, first, per, supplies=bad)
    with pytest.raises(ValueError, match="holds no value the verifier supplies"):
        _interior(reg, first, per, device=IR.DeviceCommit({}, {}, {}))
