"""Greedy token selection as Boolean Definitions: the next version of each word Definition, the same function on bits.

    ArgmaxStep_v2       (carry: (Array<16>, Array<32>, Array<32>), x: Array<16>) -> ((Array<16>, Array<32>, Array<32>), Array<32>)
                                                                                   word view: ArgmaxStep_v1
    TokenSelect_v2{V}   (logits: Array<V, Array<16, Value<1>>>) -> Array<32, Value<1>>   word view: TokenSelect_v1{V}

(`Array<w>` is ``Array<w, Value<1>>``.) `TokenSelect_v1{V}` scans `ArgmaxStep_v1` from ``best = logits[0]``, taking ``x`` iff
``Bf16GtStrict(x, best)``: the lowest index of the largest logit, a NaN never taken, +0 and -0 equal, and index 0 whenever
``logits[0]`` is a NaN (every comparison with that best is false). `TokenSelect_v2{V}` is the same function as a tree of Calls
of small Boolean Definitions, so a partition query cuts it at any vocabulary size, one group at a time:

    TokenSelectStage_v1{V,L}           level L: the level's candidates in groups of G = 64, each group one member Call; a stage's
                                       nodes read only its parameters and it returns all they produce, so `Q_word` cuts each member
                                       on its own (`verity.ir.cut.separable`)
    ArgmaxLeaf_v1{G,FIRST,FINAL}       G bf16 words -> the group's candidate (FIRST: the row's first group)
    ArgmaxMerge_v1{G,IW,FINAL}         G candidates with IW-bit indices -> their best, with an (IW + 6)-bit index

A *candidate* is (valid, key, index): ``valid`` and ``key`` are a 17-bit unsigned key, ``key`` = `scalar.bf16_key` of the word
and ``valid`` its not-NaN bit, so every NaN is below every number; logits[0] when it is a NaN is the top key (valid, 0xFFFF),
above +inf, so it wins as the scan's stuck best does. ``index`` is the position inside the candidate's span of 64^L logits. Two
candidates merge to the right one iff its key is strictly greater (the left wins ties), in a balanced tree over the group (64
positions, the missing ones constant-zero keys that never win), so a group's index is its inner index with the 6-bit position
above. ``FINAL`` (the root group) returns the index as the i32 word. `stages(V)` names the stages; a served Program calls them
in order at its root (`token_select_calls`), each a Call of its own, and `TokenSelect_v2{V}` is that chain as one Definition.
"""
from __future__ import annotations

from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind, composite
from verity.ir.refs import Coll, concat
from verity.ir.types import Array, Tuple
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import ONE, OR, ZERO, NOT, add
from verity.ml.boolean.scalar import bf16_gt_strict, bf16_key, gt_u, select
from verity_vllm.program.registry import b1

__all__ = ["ArgmaxLeaf", "ArgmaxMerge", "ArgmaxStep", "G", "GB", "TokenSelect", "TokenSelectStage", "candidate", "levels", "stages",
           "token_select_calls"]

BIT = BL.BIT
BF16 = Array(16, BIT)
I32 = Array(32, BIT)
#: the group size of every level, and its index bits
GB = 6
G = 1 << GB


def candidate(iw: int) -> Tuple:
    """A candidate with an ``iw``-bit index: (valid, key, index)."""
    return Tuple(BIT, BF16, Array(iw, BIT))


# --- ArgmaxStep_v2: TokenSelect_v1's scan step on bits -----------------------------------------------------------------------

def argmax_step(C, best, best_i, i, x):
    """``ArgmaxStep_v1``: take ``(x, i)`` iff ``x > best`` (`Bf16GtStrict_v1`), then ``i += 1``; ``y`` the new best index."""
    gt = bf16_gt_strict(C, x, best)[0]
    ni = select(C, gt, i, best_i)
    return [select(C, gt, x, best), ni, add(C, i, [ONE], width=32), ni]


CARRY = Tuple(BF16, I32, I32)
ArgmaxStep = TR.definition("ArgmaxStep", 2, argmax_step, (("carry", CARRY), ("x", BF16)), Tuple(CARRY, I32), word=b1.ArgmaxStep,
                           doc="TokenSelect_v1's scan step on bits: ArgmaxStep_v1 (strict >, so the first index wins ties; NaN never wins)")


# --- the argmax tree ---------------------------------------------------------------------------------------------------------

def _best(C, cands):
    """The best of ``cands`` ((key bits, index bits) each, keys of one width, indices of one width) padded to G positions with
    constant-zero keys: a balanced tree of merges in which the right candidate wins iff its key is strictly greater."""
    kw, iw = len(cands[0][0]), len(cands[0][1])
    level = list(cands) + [([ZERO] * kw, [ZERO] * iw)] * (G - len(cands))
    while len(level) > 1:
        nxt = []
        for (ka, ia), (kb, ib) in zip(level[0::2], level[1::2]):
            gt = gt_u(C, kb, ka)
            nxt.append((select(C, gt, kb, ka), select(C, gt, ib, ia) + [gt]))
        level = nxt
    return level[0]


def _out(key, idx, final: bool):
    if final:
        if len(idx) > 32:
            raise ValueError(f"an index of {len(idx)} bits is not an i32")
        return idx + [ZERO] * (32 - len(idx))
    return [[key[16]], key[:16], idx]


def argmax_leaf(C, S, *xs):
    """A group of bf16 words -> its candidate (or, FINAL, its index as an i32)."""
    cands = []
    for j, x in enumerate(xs):
        nan, key = bf16_key(C, x)
        if S.FIRST and j == 0:
            key, valid = [OR(C, k, nan) for k in key], ONE
        else:
            valid = NOT(nan)
        cands.append((key + [valid], []))
    key, idx = _best(C, cands)
    return _out(key, idx, S.FINAL)


def argmax_merge(C, S, *elements):
    """A group of candidates -> their best (or, FINAL, its index as an i32)."""
    cands = [(key + valid, idx) for valid, key, idx in zip(elements[0::3], elements[1::3], elements[2::3])]
    key, idx = _best(C, cands)
    return _out(key, idx, S.FINAL)


def _check_group(S) -> None:
    if not 1 <= S.G <= G or S.FINAL not in (0, 1):
        raise ValueError(f"a group has 1..{G} members and FINAL in (0, 1), not G={S.G}, FINAL={S.FINAL}")


def _leaf_sig(S):
    _check_group(S)
    if S.FIRST not in (0, 1):
        raise ValueError(f"FIRST is 0 or 1, not {S.FIRST}")
    return (("x", Array(S.G, BF16)),), (I32 if S.FINAL else candidate(GB))


def _merge_sig(S):
    _check_group(S)
    if S.IW < GB or S.IW % GB:
        raise ValueError(f"a merged candidate's index has a multiple of {GB} bits, not {S.IW}")
    return (("c", Array(S.G, candidate(S.IW))),), (I32 if S.FINAL else candidate(S.IW + GB))


def _traced(name: str, version: int, statics: tuple[str, ...], signature, build, doc: str) -> CompositeDefinition:
    """A Boolean Definition with statics: each specialization's body is ``build(C, S, *elements)`` traced (`trace.trace`), so its
    And gates are the builder's live ANDs; ``d.traced(spec)`` traces it again."""

    def types(S):
        params, ret = signature(S)
        return [t for _, t in params], ret

    def trace(spec):
        ps, ret = types(spec.S())
        return TR.trace(lambda C, *e: build(C, spec.S(), *e), ps, ret)

    def body(B, S, *colls):
        ps, ret = types(S)
        return TR.emit(TR.trace(lambda C, *e: build(C, S, *e), ps, ret), ps, ret, B, *colls)

    d = CompositeDefinition(name, version, statics, signature, body, doc=doc, conformance="exact-model-tested")
    d.build = build  # type: ignore[attr-defined]
    d.traced = trace  # type: ignore[attr-defined]
    return d


ArgmaxLeaf = _traced("ArgmaxLeaf", 1, ("G", "FIRST", "FINAL"), _leaf_sig, argmax_leaf,
                     "The best of G bf16 logits as a candidate (the first maximum, NaN below every number; FIRST: logits[0] a NaN "
                     "is the top key), or FINAL its index as an i32")
ArgmaxMerge = _traced("ArgmaxMerge", 1, ("G", "IW", "FINAL"), _merge_sig, argmax_merge,
                      "The best of G candidates (the right one iff its key is strictly greater), its index the inner index with "
                      "the 6-bit position above, or FINAL that index as an i32")


def levels(V: int) -> list[int]:
    """The candidate counts of the tree over V logits: V, then ceil(previous / G) down to 1 (V itself when V = 1)."""
    if V < 1:
        raise ValueError(f"a logits row has V >= 1 entries, not {V}")
    n = [V]
    while n[-1] > 1 or len(n) == 1:
        n.append(-(-n[-1] // G))
    return n


def _stage_sig(S):
    n = levels(S.V)
    if not 0 <= S.L < len(n) - 1:
        raise ValueError(f"the tree over V={S.V} has stages 0..{len(n) - 2}, not L={S.L}")
    final = S.L == len(n) - 2
    ins = BF16 if S.L == 0 else candidate(GB * S.L)
    return (("c", Array(n[S.L], ins)),), (I32 if final else Array(n[S.L + 1], candidate(GB * (S.L + 1))))


@composite("TokenSelectStage", 1, ["V", "L"], _stage_sig, conformance="exact-model-tested")
def TokenSelectStage(B, S, c):
    """Level L of the argmax tree over V logits: its candidates in groups of G, each group one Call (`ArgmaxLeaf_v1` at level 0,
    `ArgmaxMerge_v1` above; the root group FINAL), the full groups after the first as one batch."""
    n = levels(S.V)
    N, final = n[S.L], int(S.L == len(n) - 2)
    T = c.type.elem

    def member(g: int, first: int):
        return bind(ArgmaxLeaf, G=g, FIRST=first, FINAL=final) if S.L == 0 else bind(ArgmaxMerge, G=g, IW=GB * S.L, FINAL=final)

    def group(lo: int, hi: int) -> Coll:
        return c[lo:hi]

    full, rem = divmod(N, G)
    outs = []
    if full:
        outs.append(B.call(member(G, 1), group(0, G)))
        if full > 1:
            rest = c[G:full * G]
            outs.append(B.batch(member(G, 0), Coll(Array(full - 1, Array(G, T)), rest.refs), axes=(0,)))
    if rem:
        outs.append(B.call(member(rem, int(not full)), group(full * G, N)))
    ret = _stage_sig(S)[1]
    return Coll(ret, concat([o.refs for o in outs]))


def stages(V: int) -> list:
    """The stage Definitions of the tree over V logits, in calling order."""
    return [bind(TokenSelectStage, V=V, L=L) for L in range(len(levels(V)) - 1)]


def token_select_calls(call, logits: Coll, V: int) -> Coll:
    """``TokenSelect_v2{V}`` as Calls of its stages: ``call(fn, arg)`` emits one (``Builder.call``, or a frontend's ``ctx.call``
    at a Program's root, where each stage is a Call the partition query cuts on its own); returns the i32 token's bits."""
    x = logits
    for st in stages(V):
        x = call(st, x)
    return x


@composite("TokenSelect", 2, ["V"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="exact-model-tested")
def TokenSelect(B, S, logits):
    """Greedy sampling on bits: TokenSelect_v1{V} (the first maximum of the bf16 logits; index 0 when logits[0] is a NaN), as the
    chain of the argmax tree's stages."""
    return token_select_calls(B.call, logits, S.V)


TokenSelect.word = lambda b: bind(b1.TokenSelect, V=b["V"])  # type: ignore[attr-defined]
