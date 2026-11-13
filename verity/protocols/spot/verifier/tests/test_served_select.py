"""Round 2's design check (`second_stage`: the links, the wires and their rows it derives from the replay block alone, held
before it reads a statement) on a replay block of served tokens whose check template is the served greedy select,
`greedy_stages.GreedySelect_v1` at GLM-5.3's vocabulary: it holds, which is why that select is whole chunks, a one-field
result a chunk and a one-field token; and no session's two ports are held at one port of the interior record, which is why
each chunk position is its own Definition (Flock's verifier refuses a session with two ports of one published domain,
`Flock.PortRoots.dupId`).  A select whose rows start off its chunks, or whose token is not one whole field of its combine, is
refused (``second-links``); one whose combine reads one template's output twice passes the design check and holds two
ports at one."""

from __future__ import annotations

import pytest

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


def _design(check: SpecializedDefinition, tokens: int, q: int) -> tuple[list[SpecializedDefinition], list[SS.Part], SS.Own]:
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
    return templates, parts, SS.Own(
        tuple(range(q)), parts, {}, {}, (), links=SS.replay_links(program, key, None),
        wires=SS.replay_wires(program, key), rows=SS.replay_rows(program, key))


def _codes(own: SS.Own) -> set[str]:
    return set(SS.check(True, own, None).codes)


def _held(own: SS.Own, template: SpecializedDefinition) -> list[str]:
    """The interior record's port that holds each port of ``template``'s session (`second_stage._held_port`), its parameters
    then its outputs: a wired port's is its writer's output port, any other its own."""
    params, ret = template.signature()
    ports = [name for name, _t in params] + [f"o{j}" for j in range(SS._outputs(ret))]
    return [SS._held_port(own, True, P.template_id(template), p) for p in ports]


DESIGN = {"second-links", "second-wires", "second-cells"}


@pytest.mark.slow
def test_the_served_greedy_select_passes_round_twos_design_check_with_no_two_ports_held_at_one():
    """At the served statics, which the row of 640 below can't show: chunks of 7,744 lanes, no power of two, 256 tokens a
    replay block.  A replay block of 256 served tokens, each `GreedySelect_v1{V=154880,N=7744}`'s body in place, after a
    replay sample of three.  Its twenty templates' ``logits`` and the combine's token are links to the block's own ports,
    each of the combine's 19 results is a wire to the one whole output of one chunk template, read at that unit's row, and
    the check holds (with no statements and no interior record, the run fails for those alone).  The parts are in the order
    the body calls its templates, so each wire's writer's part is before its reader's, and no wire reads a port a link
    names.  Every session's ports are held at pairwise different ports of the interior record: the combine's at its own
    ``logits`` and ``o0`` and at the 19 chunk templates' ``o0``, one each."""
    K = G.units(SERVED_V)
    templates, parts, own = _design(G.select(SERVED_V), TOKENS, Q)
    ids = [P.template_id(t) for t in templates]
    chunks, combine = ids[:-1], ids[-1]
    assert ids == [*(f"GreedyChunk_v2{{N=7744,OFF={c * 7744}}}" for c in range(K - 1)), "GreedyCombine_v2{V=154880,N=7744}"]
    assert [(p.template, p.n) for p in parts] == [(t, TOKENS) for t in ids]
    assert [(x.template, x.port, x.value) for x in own.links] == [
        *((t, "logits", "logits") for t in ids), (combine, "o0", "o0")]
    assert [(w.template, w.port, w.writer, w.wport) for w in own.wires] == [
        (combine, f"r{c}", chunks[c], "o0") for c in range(K - 1)]
    rows = {(r.template, r.port): (r.per, r.runs) for r in own.rows}
    assert all(rows[t, "logits"] == (TOKENS * K, (SS.Run(0, TOKENS, c, K),)) for c, t in enumerate(ids))
    assert rows[combine, "o0"] == (TOKENS, (SS.Run(0, TOKENS, 0, 1),))
    assert all(rows[combine, f"r{c}"] == (TOKENS, (SS.Run(0, TOKENS, 0, 1),)) for c in range(K - 1))
    assert SS._links_ok(parts, own) and SS._wires_ok(parts, own)
    assert not DESIGN & _codes(own) and "second-statements" in _codes(own)
    at = {p.template: i for i, p in enumerate(parts)}
    linked = {(x.template, x.port) for x in own.links}
    assert all(at[w.writer] < at[w.template] and (w.writer, w.wport) not in linked for w in own.wires)
    for t in templates:
        held = _held(own, t)
        assert len(set(held)) == len(held), P.template_id(t)
    assert _held(own, templates[-1]) == [f"{combine}/logits", *(f"{t}/o0" for t in chunks), f"{combine}/o0"]
    assert all(_held(own, t) == [f"{i}/logits", f"{i}/o0"] for t, i in zip(templates[:-1], chunks, strict=True))


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
    templates, parts, own = _design(G.select(640), 4, Q)
    assert SS._links_ok(parts, own) and SS._wires_ok(parts, own) and not DESIGN & _codes(own)
    assert all(len(set(_held(own, t))) == len(_held(own, t)) for t in templates)
    padded = _wrapped("ServedSelectTestPadded", ("V", "N", "PAD"), lambda S: S.V + S.PAD,
                      lambda B, S, logits: _select(B, S, logits[S.PAD:]))
    swapped = _wrapped("ServedSelectTestSwapped", ("V", "N"), lambda S: S.V,
                       lambda B, S, logits: (lambda y: array_of([y[1], y[0]]))(_select(B, S, logits)))
    for check in (bind(padded, V=640, N=32, PAD=8), bind(swapped, V=640, N=32)):
        _t, parts, own = _design(check, 4, Q)
        assert own.links is None and len(own.wires) == 19 and "second-links" in _codes(own), check.id


def test_a_combine_that_reads_one_template_s_output_twice_holds_two_ports_at_one():
    """Why a chunk position is a Definition: a row of 96 in three chunks of 32 whose first two chunks are instances of one
    template, the combine reading each one's output.  The design check holds, both wires naming that template's ``o0``, and
    the combine's session then has two ports held at that one port of the interior record, the form Flock's verifier refuses."""
    def body(B, S, logits):
        chunk = bind(G.GreedyChunk, N=S.N, OFF=0)
        results = [B.call(chunk, logits[c * S.N:(c + 1) * S.N]) for c in range(2)]
        return B.call(bind(G.GreedyCombine, V=3 * S.N, N=S.N), logits[2 * S.N:], *results)

    twice = _wrapped("ServedSelectTestTwice", ("N",), lambda S: 3 * S.N, body)
    (chunk, combine), parts, own = _design(bind(twice, N=32), 4, Q)
    assert SS._links_ok(parts, own) and SS._wires_ok(parts, own) and not DESIGN & _codes(own)
    writer = f"{P.template_id(chunk)}/o0"
    assert [(w.port, w.writer, w.wport) for w in own.wires] == [(p, P.template_id(chunk), "o0") for p in ("r0", "r1")]
    assert _held(own, combine) == [f"{P.template_id(combine)}/logits", writer, writer, f"{P.template_id(combine)}/o0"]
