"""The check template of a two-stage window over one of DSA's long rows: `DsaTopKReadRows_v1` emitted as its cut
(`dsa_topk_reads.cut`), so `boundary_rows.Layout` takes it ``inline`` and each unit serving proves the row by is a proof block
of the replay block.

    DsaTopKReadRowsCut_v1{N,TOPK,BLOCK,READ}    `DsaTopKReadRows_v1{N,TOPK,BLOCK,READ}`'s parameters, then the index rows
        (..., c: Array<ceil(TOPK/CHUNK), I32>, i: Array<CHUNK, I32>) -> (kv, ok)                  (`index_params`)

`DsaTopKReadRows_v1` calls its mask whole, one call past what a proof block holds at a long row (`dsa_topk_mask_cut.WHOLE_MAX`),
and reads the merge's slots one I32 each, which is no whole output of any unit.  This body is the units instead (`units`), each a
call whose every operand is one whole parameter row or one whole output of one unit, wired as `dsa_topk_mask_cut.plan` and
`dsa_topk_reads.cut` wire them:

* the mask's cut: per round k < 4 a `DsaTopKDigitCount_v1` a block and a `DsaTopKDigitPick_v1`, then a `DsaTopKMaskBlock_v1` a
  block, each reading its block of the scores and n, the Call's operands, and the outputs of the units before it;
* the index: a `DsaMaskPositions_v1` a block, of that block's mask, and the merge; then a `DsaPositionsChunk_v1` a chunk of
  `dsa_topk_reads.CHUNK` slots, chunk k of the merge's slots at index row ``c[k]``, and a `DsaPositionsSlot_v1` a slot, slot j
  of chunk j // CHUNK at index row ``i[j mod CHUNK]``;
* TOPK reads of the table, READ's, each a call: read j at slot unit j's position with the Call's opening (root, domain, row[j],
  salt[j], path[j]).

The index rows are inputs so that one chunk template and one slot template serve every instance; where ``c[k]`` = k and
``i[k]`` = k (`index_values`, what the verifier supplies) it returns (kv, ok) as `DsaTopKReadRows_v1` does.  At the served
statics (`SERVED`, `served`: N = 2^17 keys, TOPK 2,048, blocks of 16,384, `EmbeddingRead_v1{V=2^17, H=576}`) that is 44 + 9 + 64
+ 2,048 + 2,048 = 4,213 units of 56 templates an instance.  `layout` registers it over the window's class (`PROMPT`,
`OUTPUTS`): per_prompt 0 and per_output 1, instance i the row at decode step i.

It stands in for the request Definition that will hold the row: its operands (the row's index scores, n and the table's
openings) are whatever the replay gives it, and nothing here ties them to the committed prompt or the served history.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import Coll, concat, tuple_of
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


def index_params(TOPK: int) -> tuple[tuple[str, Any], tuple[str, Any]]:
    """The check template's two index parameters, the one place that names them: ``c``, a row a chunk unit (its index among an
    instance's chunks), and ``i``, a row a slot of a chunk (a slot unit's index in its chunk), each row one I32."""
    return ("c", Array(-(-TOPK // R.CHUNK), R.I32)), ("i", Array(R.CHUNK, R.I32))


def index_values(TOPK: int) -> dict[str, list[int]]:
    """Each index parameter's rows as the verifier supplies them (`index_params`): row k is k."""
    return {name: list(range(ty.n)) for name, ty in index_params(TOPK)}


def _body(B: Any, S: Any, scores: Coll, n: Coll, root: Coll, domain: Coll, row: Coll, salt: Coll, path: Coll, c: Coll,
          i: Coll) -> Coll:
    out: dict[tuple, Coll] = {}
    for u in X.plan(S.N, S.TOPK, S.BLOCK):
        out[u.key] = B.call(u.defn, *_args(u.defn, u.keys, u.wires, out, scores, n))
    m = len(X.blocks(S.N, S.TOPK, S.BLOCK))
    mask = Coll(Array(S.N, BL.BIT), concat([out[("mask", b)].refs for b in range(m)]))
    merged = R.index(B, S.N, S.TOPK, S.BLOCK, mask)
    chunk, slot = _index_units(S.N, S.TOPK)
    chunks = [B.call(chunk, merged, c[k]) for k in range(c.type.n)]
    slots = [B.call(slot, chunks[j // R.CHUNK], i[j % R.CHUNK]) for j in range(S.TOPK)]
    reads = [B.call(S.READ, slots[j], root, domain, row[j], salt[j], path[j]) for j in range(S.TOPK)]
    row_t = S.READ.signature()[1].fields[0]
    return tuple_of(Coll(Array(S.TOPK, row_t), concat([r[0].refs for r in reads])),
                    Coll(Array(S.TOPK, BL.BIT), concat([r[1].refs for r in reads])))


def _index_units(N: int, TOPK: int) -> tuple[SpecializedDefinition, SpecializedDefinition]:
    return bind(R.DsaPositionsChunk, K=TOPK, W=R.CHUNK), bind(R.DsaPositionsSlot, K=R.CHUNK, N=N)


def _signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
    params, ret = bind(R.DsaTopKReadRows, N=S.N, TOPK=S.TOPK, BLOCK=S.BLOCK, READ=S.READ).signature()
    return (*params, *index_params(S.TOPK)), ret


DsaTopKReadRowsCut = CompositeDefinition(
    "DsaTopKReadRowsCut", 1, ("N", "TOPK", "BLOCK", "READ"), _signature, _body, refuse=R.DsaTopKReadRows.refuse,
    conformance="model-untested: DsaTopKReadRows_v1's function as its cut (dsa_topk_reads.cut) where index rows c[k] = k and "
                "i[k] = k, held to DsaTopKReadRows_v1 at small statics by integrations/vllm/tests/pipeline/test_long_row.py; "
                "the check that would hold it to the model is DsaTopKReadRows_v1's",
    doc="DsaTopKReadRows_v1{N,TOPK,BLOCK,READ} as its cut, the chunks' and slots' index rows (c, i) its last two parameters: "
        "every unit serving proves the row by is a call reading whole parameter rows and whole outputs")


def units(t: SpecializedDefinition) -> tuple[tuple[tuple, SpecializedDefinition], ...]:
    """Each unit of one instance of ``t`` (`DsaTopKReadRowsCut_v1`), in body order, by key: the mask's cut (`plan`'s keys), then
    ``("positions", b)`` per block, ``("merge",)``, ``("chunk", k)`` per chunk, and ``("slot", j)`` then ``("read", j)`` per
    slot."""
    N, TOPK, block, READ = (t.bindings[k] for k in ("N", "TOPK", "BLOCK", "READ"))
    bl = X.blocks(N, TOPK, block)
    chunk, slot = _index_units(N, TOPK)
    return (*((u.key, u.defn) for u in X.plan(N, TOPK, block)),
            *((("positions", b), bind(R.DsaMaskPositions, N=hi - lo, AT=lo, K=TOPK)) for b, (lo, hi) in enumerate(bl)),
            (("merge",), bind(R.DsaPositionsMerge, M=len(bl), K=TOPK)),
            *((("chunk", k), chunk) for k in range(-(-TOPK // R.CHUNK))), *((("slot", j), slot) for j in range(TOPK)),
            *((("read", j), READ) for j in range(TOPK)))


def served() -> SpecializedDefinition:
    """`DsaTopKReadRowsCut_v1` at the served statics, its READ the history table's (`TABLE_V`, `TABLE_H`)."""
    from verity.ml.commitments.gates.embedding import read

    return bind(DsaTopKReadRowsCut, **SERVED, READ=read(TABLE_V, TABLE_H))


def layout(run: str, check: SpecializedDefinition | None = None, spans: int = 1) -> BR.Layout:
    """The window's class registered over ``check`` (`served` when None) inline: per_prompt 0 and per_output 1 at (`PROMPT`,
    `OUTPUTS`), so n_rho = 2, instance i the row at decode step i."""
    return BR.Layout(served() if check is None else check, 0, 1, PROMPT, OUTPUTS, spans, run, inline=True)
