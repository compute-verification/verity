"""The check template of a two-stage window over one of DSA's long rows: `DsaTopKReadRows_v1` with its mask emitted as its cut, so
`boundary_rows.Layout` takes it ``inline`` and each unit serving proves the row by is a proof block of the replay block.

    DsaTopKReadRowsCut_v1{N,TOPK,BLOCK,READ}    `DsaTopKReadRows_v1{N,TOPK,BLOCK,READ}`'s signature and function

`DsaTopKReadRows_v1` calls its mask whole, one call past what a proof block holds at a long row (`dsa_topk_mask_cut.WHOLE_MAX`).
This body is the units instead (`units`), each a call, wired as `dsa_topk_mask_cut.plan` and `dsa_topk_reads.index` wire them:

* the mask's cut: per round k < 4 a `DsaTopKDigitCount_v1` a block and a `DsaTopKDigitPick_v1`, then a `DsaTopKMaskBlock_v1` a
  block, each reading its block of the scores and n, the Call's operands, and the outputs of the units before it;
* the index: a `DsaMaskPositions_v1` a block, of that block's mask, and the merge;
* TOPK reads of the table, READ's, read j at the merge's slot j with the Call's opening (root, domain, row[j], salt[j], path[j]).

It returns (kv, ok) as `DsaTopKReadRows_v1` does.  At the served statics (`SERVED`, `served`: N = 2^17 keys, TOPK 2,048, blocks of
16,384, `EmbeddingRead_v1{V=2^17, H=576}`) that is 44 + 9 + 2,048 = 2,101 units of 54 templates an instance.  `layout` registers
it over the window's class (`PROMPT`, `OUTPUTS`): per_prompt 0 and per_output 1, instance i the row at decode step i.

It stands in for the request Definition that will hold the row: its operands (the row's index scores, n and the table's
openings) are whatever the replay gives it, and nothing here ties them to the committed prompt or the served history.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import Coll, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import dsa_topk_mask_cut as X
from verity.ml.definitions.registry import dsa_topk_reads as R
from verity_vllm.pipeline import boundary_rows as BR

#: the long row's served statics: a class of 2^17 keys, GLM-5.3's INDEX_TOPK, `dsa_topk_reads.BLOCK` keys a block
SERVED = {"N": 1 << 17, "TOPK": 2048, "BLOCK": R.BLOCK}
#: the history table READ opens: its capacity in rows and a row's u16 words (GLM-5.3's kv row)
TABLE_V, TABLE_H = 1 << 17, 576
#: the window's class (P, O): a prompt of up to 2^17 ids and 2 served, instance i the row at decode step i
PROMPT, OUTPUTS = 1 << 17, 2


def _field(c: Coll, j: int) -> Coll:
    return c[j] if isinstance(c.type, Tuple) else c


def _args(defn: SpecializedDefinition, keys: tuple[int, int], wires: Sequence[tuple[str, tuple, int]], out: dict, scores: Coll,
          n: Coll) -> list[Coll]:
    """A cut unit's arguments: its block of the scores and n where it reads them (``keys``), and each wired parameter its writers'
    output fields in turn (`dsa_topk_mask_cut.Unit.wires`), one value of the parameter's type."""
    params = defn.signature()[0]
    lo, hi = keys
    got: dict[str, list[Coll]] = {}
    for p, key, j in wires:
        got.setdefault(p, []).append(_field(out[key], j))
    own = {"scores": scores[lo:hi], "n": n} if hi > lo else {}
    return [own[p] if p in own else got[p][0] if len(got[p]) == 1 else Coll(t, concat([c.refs for c in got[p]]))
            for p, t in params]


def _body(B: Any, S: Any, scores: Coll, n: Coll, root: Coll, domain: Coll, row: Coll, salt: Coll, path: Coll) -> Coll:
    out: dict[tuple, Coll] = {}
    for u in X.plan(S.N, S.TOPK, S.BLOCK):
        out[u.key] = B.call(u.defn, *_args(u.defn, u.keys, u.wires, out, scores, n))
    m = len(X.blocks(S.N, S.TOPK, S.BLOCK))
    mask = Coll(Array(S.N, BL.BIT), concat([out[("mask", b)].refs for b in range(m)]))
    tok = R.index(B, S.N, S.TOPK, S.BLOCK, mask)
    reads = B.batch(S.READ, tok, root, domain, row, salt, path, axes=(0, None, None, 0, 0, 0))
    row_t = S.READ.signature()[1].fields[0]
    w = row_t.leaves
    return tuple_of(Coll(Array(S.TOPK, row_t), strided_view(reads, 0, w, w + 1, 1, S.TOPK * w)),
                    Coll(Array(S.TOPK, BL.BIT), strided_view(reads, w, 1, w + 1, 1, S.TOPK)))


def _signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
    return bind(R.DsaTopKReadRows, N=S.N, TOPK=S.TOPK, BLOCK=S.BLOCK, READ=S.READ).signature()


DsaTopKReadRowsCut = CompositeDefinition(
    "DsaTopKReadRowsCut", 1, ("N", "TOPK", "BLOCK", "READ"), _signature, _body, refuse=R.DsaTopKReadRows.refuse,
    conformance="model-untested: DsaTopKReadRows_v1's function with its mask as its cut (dsa_topk_mask_cut.plan), held to "
                "DsaTopKReadRows_v1 at small statics by integrations/vllm/tests/pipeline/test_long_row.py; the check that would "
                "hold it to the model is DsaTopKReadRows_v1's",
    doc="DsaTopKReadRows_v1{N,TOPK,BLOCK,READ} with its mask as its cut: every unit serving proves the row by is a call")


def units(t: SpecializedDefinition) -> tuple[tuple[tuple, SpecializedDefinition], ...]:
    """Each unit of one instance of ``t`` (`DsaTopKReadRowsCut_v1`), in body order, by key: the mask's cut (`plan`'s keys), then
    ``("positions", b)`` per block, ``("merge",)`` and ``("read", j)`` per slot."""
    N, TOPK, block, READ = (t.bindings[k] for k in ("N", "TOPK", "BLOCK", "READ"))
    bl = X.blocks(N, TOPK, block)
    return (*((u.key, u.defn) for u in X.plan(N, TOPK, block)),
            *((("positions", b), bind(R.DsaMaskPositions, N=hi - lo, AT=lo, K=TOPK)) for b, (lo, hi) in enumerate(bl)),
            (("merge",), bind(R.DsaPositionsMerge, M=len(bl), K=TOPK)), *((("read", j), READ) for j in range(TOPK)))


def served() -> SpecializedDefinition:
    """`DsaTopKReadRowsCut_v1` at the served statics, its READ the history table's (`TABLE_V`, `TABLE_H`)."""
    from verity.ml.commitments.gates.embedding import read

    return bind(DsaTopKReadRowsCut, **SERVED, READ=read(TABLE_V, TABLE_H))


def layout(run: str, check: SpecializedDefinition | None = None, spans: int = 1) -> BR.Layout:
    """The window's class registered over ``check`` (`served` when None) inline: per_prompt 0 and per_output 1 at (`PROMPT`,
    `OUTPUTS`), so n_rho = 2, instance i the row at decode step i."""
    return BR.Layout(served() if check is None else check, 0, 1, PROMPT, OUTPUTS, spans, run, inline=True)
