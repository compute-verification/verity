"""Round 2's design check (`second_stage`: the links, the wires and their rows it derives from the replay block alone, held
before it reads a statement) on a replay block of served tokens whose check template is the served greedy select,
`greedy_stages.GreedySelect_v1` at GLM-5.3's vocabulary: it holds, which is why that select is whole chunks and a one-field
token.  A select whose rows start off its chunks, or whose token is not one whole field of its combine, is refused
(``second-links``)."""

from __future__ import annotations

from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.refs import array_of, coll_concat
from verity.circuits.types import Array
from verity.definitions.registry import greedy_stages as G
from verity.definitions.registry.prims import BF16
from verity.protocols.spot.verifier import second_stage as SS
from verity.protocols.spot.verifier.one_stage import partition as P

SERVED_V = 154880
#: a replay block's served tokens and a replay sample's blocks
TOKENS, Q = 256, 3
WORDS = Array(2, G.U16)


def _inlined(template: SpecializedDefinition, n: int) -> SpecializedDefinition:
    """``n`` instances of the composite ``template``, each one's body in place, every parameter an array over instances and
    the result the instances' results in order: a replay block whose check's proof blocks are the templates that body calls,
    as a serving adapter registers one."""
    params, ret = template.signature()

    def signature(S):
        return tuple((name, Array(S.N, t)) for name, t in params), Array(S.N, ret)

    def body(B, S, *args):
        return coll_concat(Array(S.N, ret), [template.defn.body(B, template.S(), *(a[i] for a in args)) for i in range(S.N)])

    return bind(CompositeDefinition("ServedSelectTestInlined", 1, ("template", "N"), signature, body, register=False),
                template=codec.descriptor_id(template), N=n)


def _design(check: SpecializedDefinition, tokens: int, q: int) -> tuple[list[str], list[SS.Part], SS.Own]:
    """The templates ``check``'s body calls, in its order; the interior's parts after a replay sample of ``q`` blocks of
    ``tokens`` instances of it, template-major in that order; and what round 2 holds of the block's design: the links, wires
    and rows it derives from the block."""
    replay = _inlined(check, tokens)
    key = codec.descriptor_id(replay)
    program = P.population_program(replay, 8)
    templates = list({P.template_id(n.fn): n.fn for n in check.body.nodes}.values())
    cells = [tokens * sum(1 for n in check.body.nodes if n.fn is t) for t in templates]
    interior = P.mixed_population_program([(t, q * n) for t, n in zip(templates, cells, strict=True)])
    parts = SS.parts(q, P.strata(interior, P.template_instances_query(templates)))
    return [P.template_id(t) for t in templates], parts, SS.Own(
        tuple(range(q)), parts, {}, {}, (), links=SS.replay_links(program, key, None),
        wires=SS.replay_wires(program, key), rows=SS.replay_rows(program, key))


def _codes(own: SS.Own) -> set[str]:
    return set(SS.check(True, own, None).codes)


DESIGN = {"second-links", "second-wires", "second-cells"}


def test_the_served_greedy_select_passes_round_twos_design_check():
    """A replay block of 256 served tokens, each `GreedySelect_v1{V=154880,N=7744}`'s body in place, after a replay sample of
    three.  Its three templates' ``logits`` and the combine's token are links to the block's own ports, each of a combine's 38
    pair words is a wire to one whole output of one chunk unit, read at that unit's row, and the check holds (with no
    statements and no interior record, the run fails for those alone).  The parts are in the order the body calls its
    templates, so each wire's writer's part is before its reader's, and no wire reads a port a link names."""
    K = G.units(SERVED_V)
    (first, chunk, combine), parts, own = _design(G.select(SERVED_V), TOKENS, Q)
    assert (first, chunk, combine) == ("GreedyChunk_v1{N=7744,FIRST=true}", "GreedyChunk_v1{N=7744,FIRST=false}",
                                       "GreedyCombine_v2{V=154880,N=7744}")
    assert [(p.template, p.n) for p in parts] == [(first, TOKENS), (chunk, TOKENS * (K - 2)), (combine, TOKENS)]
    assert [(x.template, x.port, x.value) for x in own.links] == [
        (first, "logits", "logits"), (chunk, "logits", "logits"), (combine, "logits", "logits"), (combine, "o0", "o0")]
    assert [(w.template, w.port, w.writer, w.wport) for w in own.wires] == [
        (combine, f"{'bi'[j]}{c}", chunk if c else first, f"o{j}") for c in range(K - 1) for j in (0, 1)]
    rows = {(r.template, r.port): (r.per, r.runs) for r in own.rows}
    assert rows[first, "logits"] == (TOKENS * K, (SS.Run(0, TOKENS, 0, K),))
    assert rows[combine, "logits"] == (TOKENS * K, (SS.Run(0, TOKENS, K - 1, K),))
    assert rows[chunk, "logits"] == (TOKENS * K, tuple(SS.Run(t * (K - 2), K - 2, t * K + 1, 1) for t in range(TOKENS)))
    assert rows[combine, "o0"] == (TOKENS, (SS.Run(0, TOKENS, 0, 1),))
    assert all(rows[combine, f"{p}0"] == (TOKENS, (SS.Run(0, TOKENS, 0, 1),)) for p in "bi")
    assert all(rows[combine, f"{p}{c}"] == (TOKENS * (K - 2), (SS.Run(0, TOKENS, c - 1, K - 2),))
               for p in "bi" for c in range(1, K - 1))
    assert SS._links_ok(parts, own) and SS._wires_ok(parts, own)
    assert not DESIGN & _codes(own) and "second-statements" in _codes(own)
    at = {p.template: i for i, p in enumerate(parts)}
    linked = {(x.template, x.port) for x in own.links}
    assert all(at[w.writer] < at[w.template] and (w.writer, w.wport) not in linked for w in own.wires)


def _wrapped(name: str, statics: tuple[str, ...], lanes, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, statics, lambda S: ((("logits", Array(lanes(S), BF16)),), WORDS), body,
                               register=False)


def _select(B, S, logits):
    return G.GreedySelect.body(B, bind(G.GreedySelect, V=S.V, N=S.N).S(), logits)


def test_a_select_off_its_chunks_rows_or_of_no_whole_token_field_is_refused():
    """What the select's shape is for, on a row of 640 in 20 chunks of 32: as it is, the check holds.  The same units behind
    8 more lanes a token read chunks that start at no multiple of their width, so none is one row of the block's logits; and
    the same combine with its two words returned in the other order returns no one whole field: round 2 derives no links for
    either (``second-links``)."""
    _t, parts, own = _design(G.select(640), 4, Q)
    assert SS._links_ok(parts, own) and SS._wires_ok(parts, own) and not DESIGN & _codes(own)
    padded = _wrapped("ServedSelectTestPadded", ("V", "N", "PAD"), lambda S: S.V + S.PAD,
                      lambda B, S, logits: _select(B, S, logits[S.PAD:]))
    swapped = _wrapped("ServedSelectTestSwapped", ("V", "N"), lambda S: S.V,
                       lambda B, S, logits: (lambda y: array_of([y[1], y[0]]))(_select(B, S, logits)))
    for check in (bind(padded, V=640, N=32, PAD=8), bind(swapped, V=640, N=32)):
        _t, parts, own = _design(check, 4, Q)
        assert own.links is None and len(own.wires) == 38 and "second-links" in _codes(own), check.id
