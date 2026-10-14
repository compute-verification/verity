"""`verity_vllm.program.replay.substitute`: the O(1) path for scattered `Strided` arguments.

A `Strided` whose runs are O(count) leaves (a transposed weight read, a column gather) used to be substituted run by
run: the GLM-5.3 full-depth derive carries ~2e6-leaf transposed-weight arguments on every MLA-absorb BatchedGemm, so
`Program(flat_sd)` built billion-part concats and never finished.  `substitute` now rebases such a part onto the
target's refs in O(1) when the target over the part's leaf span is one contiguous `Affine` run, which also makes the
replayed descriptor equal to the direct lowering's (the compact `Strided` survives replay instead of exploding).
Every other shape takes the run path unchanged, bit-identically — the pinned digest below holds on the commit before
the fast path was added."""
from __future__ import annotations

from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Strided, concat, runs, strided_view
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.replay import _substitute_scattered, replayer, substitute

V32 = Value(32)
Add32 = PrimitiveDefinition("TAdd32", 1, (("a", V32), ("b", V32)), V32, lambda a, b: (a + b) & 0xFFFFFFFF, register=False)
Row3 = CompositeDefinition("TRow3", 1, (), lambda S: ((("x", Array(3, V32)),), V32),
                           lambda Bd, S, x: Bd.call(Add32, x[0], x[2]), register=False)


def _run_path(refs, env):
    """What `substitute` did for every shape before the scattered fast path (its run path, verbatim)."""
    parts = [env[(space, idx)].slice(lo, hi) for space, idx, lo, hi in runs(refs)]
    return concat(parts) if parts else refs


def _leaves(r):
    return [r[j] for j in range(r.count)]


def _expected(refs, env):
    return [env[(s, i)][leaf] for s, i, leaf in _leaves(refs)]


def test_run_path_bit_identical_for_ordinary_shapes():
    """No ordinary shape reaches the fast path, and `substitute` returns exactly the run path's output."""
    env = {("p", 0): Affine("n", 3, 100, 64), ("p", 1): Concat([Affine("n", 4, 0, 8), Affine("n", 5, 16, 8)])}
    ordinary = [
        Affine("p", 0, 5, 10),
        Strided("p", 0, 7, 1, 0, 0, 6),      # broadcast of one leaf
        Strided("p", 0, 0, 4, 8, 1, 16),     # contiguous rows (inner_stride == 1)
        Strided("p", 0, 3, 1, 1, 0, 5),      # an affine run written as a strided one
        Explicit((("p", 0, 2), ("p", 1, 9), ("p", 0, 2))),
        Concat([Affine("p", 0, 0, 4), Affine("p", 1, 2, 6), Strided("p", 0, 20, 1, 0, 0, 3)]),
    ]
    for refs in ordinary:
        assert _substitute_scattered(refs, env) is None, refs
        assert substitute(refs, env).to_json() == _run_path(refs, env).to_json(), refs
        assert _leaves(substitute(refs, env)) == _expected(refs, env), refs


def test_partlog_cuts_keep_the_memoised_path():
    log = PartLog([Affine("p", 0, 0, 4), Affine("p", 0, 8, 4)])
    refs = log.cut(2, [Affine("p", 0, 16, 4)])
    env = {("p", 0): Affine("n", 3, 100, 64)}
    assert _substitute_scattered(refs, env) is None
    logs: dict = {}
    assert _leaves(substitute(refs, env, logs)) == _expected(refs, env)
    assert logs  # the cut went through the shared-run memo, not the fast path


def test_scattered_strided_rebases_in_o1():
    env = {("p", 0): Affine("n", 3, 100, 200)}
    for refs in [
        Strided("p", 0, 0, 3, 6, 2, 12),     # row gaps: inner_stride 2
        Strided("p", 0, 0, 16, 1, 8, 48),    # column gather: i-major
        Strided("p", 0, 5, 1, 7, 0, 8),      # single-leaf rows: inner == 1, outer_stride 7
        Strided("p", 0, 10, 4, 0, 3, 8),     # broadcast rows with scattered elements
    ]:
        out = substitute(refs, env)
        assert isinstance(out, Strided), refs
        assert (out.inner, out.outer_stride, out.inner_stride, out.count) == (refs.inner, refs.outer_stride, refs.inner_stride, refs.count)
        assert _leaves(out) == _expected(refs, env), refs


def test_scattered_part_inside_a_concat():
    env = {("p", 0): Affine("n", 3, 100, 64), ("p", 1): Affine("n", 4, 0, 32)}
    refs = Concat([Affine("p", 1, 2, 5), Strided("p", 0, 0, 3, 6, 2, 12), Strided("p", 1, 9, 1, 0, 0, 4)])
    out = substitute(refs, env)
    assert _leaves(out) == _expected(refs, env)
    assert isinstance(out, Concat) and len(out.parts) == 3  # not 5 + 12 + 1 single-leaf parts


def test_rebase_refused_when_the_span_is_not_one_run():
    """A target whose refs break inside the part's leaf span cannot rebase; the output is the run path's, bit-identical."""
    env = {("p", 0): Concat([Affine("n", 4, 0, 12), Affine("n", 5, 50, 52)])}
    refs = Strided("p", 0, 0, 3, 6, 2, 12)  # span [0, 23) crosses the parts' boundary at 12
    assert substitute(refs, env).to_json() == _run_path(refs, env).to_json()
    assert _leaves(substitute(refs, env)) == _expected(refs, env)


def _views_body(Bd, S, M):  # M: [4][6]; the batch reads a scattered view (inner 3, inner_stride 2: leaves 0,2,4 / 6,8,10 / ...)
    return Bd.batch(Row3, Coll(Array(4, Array(3, V32)), strided_view(M, 0, 3, 6, 2, 12)), axes=(0,))


def _replayed(direct_sd, name: str) -> Program:
    """`Program` of `direct_sd`'s recorded flat body re-emitted through `replayer` — how a derived program lowers
    (`derive._flat_definition`), with the direct definition's own name so the digests are comparable."""
    body = direct_sd.body
    flat = CompositeDefinition(name, 1, (), lambda S: (tuple(body.params), body.ret_type), replayer(body), register=False)
    sd = bind(flat)
    sd.set_body(body)
    return Program(sd)


def test_replayed_program_equals_direct_lowering_for_scattered_views():
    """The compact `Strided` survives replay: the replayed program's descriptor (and digest) is the direct lowering's,
    where the run path exploded it into one part per leaf and could never match."""
    views = CompositeDefinition("TViews", 1, (), lambda S: ((("M", Array(4, Array(6, V32))),), Array(4, V32)),
                                _views_body, register=False)
    direct = Program(bind(views))
    replayed = _replayed(bind(views), "TViews")
    assert program_digest(encode_program(replayed)) == program_digest(encode_program(direct))
    batch = next(n for n in replayed.root.body.nodes if n.form == "batch" and n.args)  # not the Input batch
    assert isinstance(batch.args[0].refs, Strided)


def test_replayed_digest_unchanged_for_ordinary_bodies():
    """The run path's output for a body with no scattered part: pinned on the commit before the fast path was added
    (the fast path never fires, so the descriptor is bit-identical)."""
    def body(Bd, S, M):  # contiguous rows (inner_stride == 1) and a broadcast: both take the run path
        rows = Bd.batch(Row3, Coll(Array(4, Array(3, V32)), strided_view(M, 0, 3, 6, 1, 12)), axes=(0,))
        rep = Bd.batch(Add32, rows, Coll(Array(4, V32), strided_view(rows, 1, 1, 0, 0, 4)), axes=(0, 0))
        return rep
    plain = CompositeDefinition("TPlain", 1, (), lambda S: ((("M", Array(4, Array(6, V32))),), Array(4, V32)),
                                body, register=False)
    d = program_digest(encode_program(_replayed(bind(plain), "TPlain")))
    assert d == PLAIN_DIGEST


PLAIN_DIGEST = ("fdffd0189139ab686564436a5c3f132d1ee6317a6a4ffd070b693698a5823bb0"
                "26e12a78697135077659bafcfe9f61d75d2b55b33c01418869c71e56319f3f45")  # computed on 7b399e1b9, before the fast path
