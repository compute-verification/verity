"""The check template of a two-stage window over one of DSA's long rows: `DsaTopKReadRows_v1` emitted as its cut
(`dsa_topk_reads.cut`), so `boundary_rows.Layout` takes it ``inline`` and each unit serving proves the row by is a proof block
of the replay block.

    DsaTopKReadRowsCut_v1{N,TOPK,BLOCK,READ}    `DsaTopKReadRows_v1{N,TOPK,BLOCK,READ}`'s parameters, then the index rows
        (..., c: Array<ceil(TOPK/CHUNK), I32>, i: Array<CHUNK, I32>) -> ok                        (`index_params`)

It returns `DsaTopKReadRows_v1`'s ``ok`` alone. The rows that Definition returns (``kv``) are its read units' outputs, row j
read unit j's, which the interior commits once, at the read's output port; a reader of a row takes that output, one whole
output of one unit, as every wire here does. Returned as well they would be the replay block's output a second time, TOPK
rows an instance, which no public size class holds at the served statics.

`DsaTopKReadRows_v1` calls its mask whole, one call past what a proof block holds at a long row (`dsa_topk_mask_cut.WHOLE_MAX`),
and reads the merge's slots one I32 each, which is no whole output of any unit.  This body is the units instead (`units`), each a
call whose every operand is one whole parameter row or one whole output of one unit, wired as `dsa_topk_mask_cut.plan` and
`dsa_topk_reads.cut` wire them:

* the mask's cut: per round k < 4 a `DsaTopKDigitCount_v1` a block and a `DsaTopKDigitPick_v1`, then a `DsaTopKMaskBlock_v1` a
  block, each reading its block of the scores and n, the Call's operands, and the outputs of the units before it;
* the index: a `DsaMaskPositions_v1` a block, of that block's mask unit's output, and the merge of their lists and counts;
  then a `DsaPositionsChunk_v1` a chunk of
  `dsa_topk_reads.CHUNK` slots, chunk k of the merge's slots at index row ``c[k]``, and a `DsaPositionsSlot_v1` a slot, slot j
  of chunk j // CHUNK at index row ``i[j mod CHUNK]``;
* TOPK reads of the table, READ's, each a call: read j at slot unit j's position with the Call's opening (root, domain, row[j],
  salt[j], path[j]).

It is this cut at every N: also up to `WHOLE_MAX` keys, where `dsa_topk_reads.cut` takes the mask as one unit, which no long
row is.  The index rows are inputs so that one chunk template and one slot template serve every instance; where ``c[k]`` = k and
``i[k]`` = k (`index_values`, what the verifier supplies) its reads return (kv, ok) as `DsaTopKReadRows_v1` does
(`read_calls`), and it returns their ok.  At the served statics (`SERVED`, `served`: N = 2^17 keys, TOPK 2,048, blocks of
16,384, `EmbeddingRead_v1{V=2^17, H=576}`) that is 44 + 9 + 64 + 2,048 + 2,048 = 4,213 units of 56 templates an instance.
`layout` registers it over the window's class (`PROMPT`, `OUTPUTS`): per_prompt 0 and per_output 1, instance i the row at
decode step i.

It stands in for the request Definition that will hold the row: its operands (the row's index scores, n and the table's
openings) are whatever the replay gives it, and nothing here ties them to the committed prompt or the served history.

**The interior** of a class registered inline over it, from the check and the class's instances a replay block alone: `window`,
its templates' members and their units a block, as `interior_rows.interior_window` takes them, and `links`, the block's
parameters the interior registers as values (`interior_links`): the scores, n and the reads' rows, salts and paths the replay
gives, and the table's root and domain and the index rows the verifier supplies (`supplied`). `links` states them in closed
form, since round 2's derivation (`interior_links.links`) reads the block's Program, which at the served statics builds every
unit's body; the tests hold the two to each other at small statics.
`spans` is a sampled span's units as `interior_rows.interior` commits them, one `interior_rows.Span` a template: every unit's
rows and outputs in u16 words, computed from each instance's scores and n and the table its reads open by the units'
references (`dsa_topk_mask_cut.values`, `dsa_topk_reads.values`, `interior_links.table_reads`), never by a Definition.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from verity.circuits import boolean as BL
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import dsa_topk_mask_cut as X
from verity.definitions.registry import dsa_topk_reads as R
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


def read_calls(B: Any, S: Any, scores: Coll, n: Coll, root: Coll, domain: Coll, row: Coll, salt: Coll, path: Coll, c: Coll,
               i: Coll) -> list[Coll]:
    """The check's units emitted on ``B``, and its TOPK read calls in slot order: read j's output is row j of
    `DsaTopKReadRows_v1`'s kv and its ok."""
    out: dict[tuple, Coll] = {}
    for u in X.plan(S.N, S.TOPK, S.BLOCK):
        out[u.key] = B.call(u.defn, *_args(u.defn, u.keys, u.wires, out, scores, n))
    lists = [B.call(bind(R.DsaMaskPositions, N=hi - lo, AT=lo, K=S.TOPK), out[("mask", b)])
             for b, (lo, hi) in enumerate(X.blocks(S.N, S.TOPK, S.BLOCK))]
    merged = B.call(bind(R.DsaPositionsMerge, M=len(lists), K=S.TOPK), *(o[0] for o in lists), *(o[1] for o in lists))
    chunk, slot = _index_units(S.N, S.TOPK)
    chunks = [B.call(chunk, merged, c[k]) for k in range(c.type.n)]
    slots = [B.call(slot, chunks[j // R.CHUNK], i[j % R.CHUNK]) for j in range(S.TOPK)]
    return [B.call(S.READ, slots[j], root, domain, row[j], salt[j], path[j]) for j in range(S.TOPK)]


def _body(B: Any, S: Any, *args: Coll) -> Coll:
    return Coll(Array(S.TOPK, BL.BIT), concat([r[1].refs for r in read_calls(B, S, *args)]))


def _index_units(N: int, TOPK: int) -> tuple[SpecializedDefinition, SpecializedDefinition]:
    return bind(R.DsaPositionsChunk, K=TOPK, W=R.CHUNK), bind(R.DsaPositionsSlot, K=R.CHUNK, N=N)


def _signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
    params, ret = bind(R.DsaTopKReadRows, N=S.N, TOPK=S.TOPK, BLOCK=S.BLOCK, READ=S.READ).signature()
    return (*params, *index_params(S.TOPK)), ret.fields[1]


DsaTopKReadRowsCut = CompositeDefinition(
    "DsaTopKReadRowsCut", 1, ("N", "TOPK", "BLOCK", "READ"), _signature, _body, refuse=R.DsaTopKReadRows.refuse,
    conformance="model-untested: DsaTopKReadRows_v1's ok as its cut (dsa_topk_reads.cut) where index rows c[k] = k and "
                "i[k] = k, its read units' outputs that Definition's kv: both held to DsaTopKReadRows_v1 at small statics by "
                "adapters/vllm/tests/pipeline/test_long_row.py; the check that would hold it to the model is "
                "DsaTopKReadRows_v1's",
    doc="DsaTopKReadRows_v1{N,TOPK,BLOCK,READ}'s ok as its cut, the chunks' and slots' index rows (c, i) its last two "
        "parameters: every unit serving proves the row by is a call reading whole parameter rows and whole outputs, and "
        "row j is read unit j's output")


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
    from verity.commitments.gates.embedding import read

    return bind(DsaTopKReadRowsCut, **SERVED, READ=read(TABLE_V, TABLE_H))


def layout(run: str, check: SpecializedDefinition | None = None, spans: int = 1) -> BR.Layout:
    """The window's class registered over ``check`` (`served` when None) inline: per_prompt 0 and per_output 1 at (`PROMPT`,
    `OUTPUTS`), so n_rho = 2, instance i the row at decode step i. Its replay block returns each read's ok, the check's
    verdicts, which the class commits as its return (`boundary_rows.Layout.verdicts`)."""
    return BR.Layout(served() if check is None else check, 0, 1, PROMPT, OUTPUTS, spans, run, inline=True, verdicts=True)


def window(check: SpecializedDefinition, n_rho: int) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    """The interior window of a class whose replay block is ``n_rho`` instances of ``check`` inlined, as
    `interior_rows.interior_window` takes it: the templates ``check``'s body calls, in the order of their first call
    (`boundary_rows.Layout.templates`), each its unit member on bit rows named after its parameters (`chunk_rows.unit_member`,
    `interior_rows.named`), and each one's units a block, its calls an instance times ``n_rho``."""
    from verity_vllm.pipeline import chunk_rows as CR, interior_rows as IR, serving_rows as SRW

    calls = Counter(descriptor_id(d) for _, d in units(check))
    cells = tuple(n * n_rho for n in calls.values())
    return tuple(IR.named(CR.unit_member(SRW.template(t), n)) for t, n in zip(calls, cells, strict=True)), cells


def supplied(check: SpecializedDefinition, n_rho: int) -> dict[str, tuple[int, ...] | None]:
    """The block's parameters the verifier supplies (`interior_links.Supply`), by name: the table's ``root`` and ``domain``,
    each sampled span's the table round 1 served for it (None), and the index rows ``c`` and ``i``, a block's rows instance
    by instance, each instance's `index_values`."""
    index = index_values(check.bindings["TOPK"])
    return {"root": None, "domain": None, **{p: tuple(v) * n_rho for p, v in index.items()}}


def links(check: SpecializedDefinition, n_rho: int, tables: bool = False) -> tuple[Any, ...]:
    """The links of the replay block of ``n_rho`` instances of ``check`` inlined, as the interior registers their values and
    in round 2's order (`interior_links.links` on the block, which this is held to): each template's port that reads a
    parameter of the block, the value named as round 2 names it, the parameter's name, or ``<block>/<parameter>`` where
    round 1's record names a port with a ``/`` (``tables``: the class has history tables, `boundary_rows.Layout.layers`).

    * a unit of the mask's cut that reads scores: its block of them, row b of an instance's blocks, and the instance's n
      (`Operand`: the replay's);
    * a chunk's ``c`` and a slot's ``j``: its row of the block's ``c`` and ``i`` (`Supply`: `supplied`'s indices);
    * a read's ``root`` and ``domain``: its instance's row (`Supply`: the table round 1 served), and its ``row``, ``salt``
      and ``path``, a row a read (`Operand`).

    ValueError where the row's blocks are not all `BLOCK` keys: a block's scores are one row of the parameter."""
    from verity_vllm.pipeline import chunk_rows as CR, interior_links as IL

    N, TOPK, block, READ = (check.bindings[k] for k in ("N", "TOPK", "BLOCK", "READ"))
    bl = X.blocks(N, TOPK, block)
    if {hi - lo for lo, hi in bl} != {block}:
        raise ValueError(f"{descriptor_id(check)}: its blocks {bl[:2]}..{bl[-1]} are not {block} keys each, and a unit's "
                         "scores are one whole row of the block's parameter")
    key = descriptor_id(BR.inlined(check, n_rho))
    sup = supplied(check, n_rho)

    def value(p: str) -> str:
        return f"{key}/{p}" if tables else p

    def words(d: SpecializedDefinition, port: str) -> int:
        return dict(d.signature()[0])[port].leaves // 16

    every = tuple(range(n_rho))
    out: list[Any] = []
    for u in X.plan(N, TOPK, block):
        if u.keys[1] <= u.keys[0]:
            continue
        t, b = descriptor_id(u.defn), bl.index(tuple(u.keys))
        if any(x.template == t for x in out):
            raise ValueError(f"{t} is two units of an instance: its units would read two blocks' scores")
        out += [IL.Operand(t, "scores", value("scores"), n_rho * len(bl), tuple(i * len(bl) + b for i in every),
                           words(u.defn, "scores")),
                IL.Operand(t, "n", value("n"), n_rho, every, words(u.defn, "n"))]
    chunk, slot = _index_units(N, TOPK)
    chunks = -(-TOPK // R.CHUNK)
    out += [IL.Supply(descriptor_id(chunk), CR.INDEX_ROWS[chunk.base_id], value("c"), n_rho * chunks, tuple(range(n_rho * chunks)),
                      sup["c"]),
            IL.Supply(descriptor_id(slot), CR.INDEX_ROWS[slot.base_id], value("i"), n_rho * R.CHUNK,
                      tuple(i * R.CHUNK + j % R.CHUNK for i in every for j in range(TOPK)), sup["i"])]
    read = descriptor_id(READ)
    out += [IL.Supply(read, p, value(p), n_rho, tuple(i for i in every for _ in range(TOPK))) for p in IL.TABLE]
    out += [IL.Operand(read, p, value(p), n_rho * TOPK, tuple(range(n_rho * TOPK)), words(READ, p)) for p in IL.READ[3:]]
    return tuple(out)


def _words(ty: Any, v: Any) -> np.ndarray:
    """A value of type ``ty`` as its row's u16 words, the low word first: an int, a sequence of u32 words where ``ty`` is
    32 bits of each, else a sequence of bits, the last word's padding 0."""
    w = -(-ty.leaves // 16)
    if isinstance(v, (int, np.integer)):
        return np.asarray([(int(v) >> 16 * k) & 0xFFFF for k in range(w)], dtype=np.uint16)
    v = np.asarray(v, dtype=np.uint64)
    if 2 * len(v) == w:
        return v.astype("<u4").view("<u2")
    if len(v) > ty.leaves:
        raise ValueError(f"{len(v)} bits for a value of {ty.leaves}")
    bits = np.zeros(16 * w, dtype=np.uint8)
    bits[:len(v)] = v
    return np.packbits(bits, bitorder="little").view("<u2")


def _instance(check: SpecializedDefinition, scores: Sequence[int], n: int, table: Any) -> list[tuple[list, list]]:
    """One instance's units in body order (`units`), each its parameters' rows and its outputs' in u16 words: the row's
    ``scores`` (N f32 words) and ``n`` through the mask's cut and the index (their references), each wired parameter its
    writer's output as the body wires it (`_args`), the index rows the verifier's (`index_values`), and read j opening
    ``table`` at slot unit j's position."""
    from verity_vllm.pipeline import interior_links as IL

    N, TOPK, block = (check.bindings[k] for k in ("N", "TOPK", "BLOCK"))
    if len(scores) != N:
        raise ValueError(f"{len(scores)} scores for a row of {N} keys")
    bl, plan = X.blocks(N, TOPK, block), {u.key: u for u in X.plan(N, TOPK, block)}
    got = X.values(list(scores), int(n), TOPK, block)
    got |= R.values([bit for b, (lo, hi) in enumerate(bl) for bit in got[("mask", b)][0][:hi - lo]], TOPK, block)
    reads = IL.table_reads(table, [got[("slot", j)][0] for j in range(TOPK)])
    index = index_values(TOPK)
    out: dict[tuple, list[np.ndarray]] = {}
    done: list[tuple[list, list]] = []
    for key, defn in units(check):
        params, ret = defn.signature()
        fields = ret.fields if isinstance(ret, Tuple) else (ret,)
        if key[0] == "read":
            rows = [reads["rows"][p][key[1]] for p, _ty in params]
            outs = [reads["outs"][f"o{f}"][key[1]] for f in range(len(fields))]
        else:
            outs = [_words(ty, v) for ty, v in zip(fields, got[key], strict=True)]
            if key in plan:
                u = plan[key]
                lo, hi = u.keys
                if len({p for p, _w, _f in u.wires}) != len(u.wires):
                    raise ValueError(f"{descriptor_id(defn)}: a parameter of two writers, which no row here joins")
                given = {p: out[w][f] for p, w, f in u.wires}
                if hi > lo:
                    given |= {"scores": _words(Array(hi - lo, R.F32), scores[lo:hi]), "n": _words(R.I32, int(n) & 0xFFFFFFFF)}
                rows = [given[p] for p, _ty in params]
            elif key[0] == "positions":
                rows = [out[("mask", key[1])][0]]
            elif key[0] == "merge":
                rows = [out[("positions", b)][f] for f in (0, 1) for b in range(len(bl))]
            elif key[0] == "chunk":
                rows = [out[("merge",)][0], _words(R.I32, index["c"][key[1]])]
            else:
                rows = [out[("chunk", key[1] // R.CHUNK)][0], _words(R.I32, index["i"][key[1] % R.CHUNK])]
        for (p, ty), r in zip(params, rows, strict=True):
            if len(r) != ty.leaves // 16:
                raise ValueError(f"{descriptor_id(defn)}: its {p!r} row is {ty.leaves // 16} words, not {len(r)}")
        out[key] = outs
        done.append((rows, outs))
    return done


def spans(check: SpecializedDefinition, rid: str, rows: Sequence[tuple[Sequence[int], int]], table: Any,
          program: str) -> tuple[Any, ...]:
    """A sampled span's units as `interior_rows.interior` takes them: one `interior_rows.Span` a template of
    ``check`` in `window`'s order, its units instance by instance (its ``group`` a template's units an instance), each
    unit's rows and outputs in u16 words. ``rows``: each instance's (scores, n), the row's N index scores as f32 words and
    its n, as the replay gives them; ``table``: the history table the span's reads open (`interior_links.kept_table`, the
    one round 1 served); ``rid`` and ``program``: the request and the digest of the Program it was replayed from."""
    from verity_vllm.pipeline import chunk_rows as CR, interior_rows as IR, serving_rows as SRW

    ids = [descriptor_id(d) for _, d in units(check)]
    per: dict[str, list[tuple[list, list]]] = {t: [] for t in ids}
    for scores, n in rows:
        for t, unit in zip(ids, _instance(check, scores, n, table), strict=True):
            per[t].append(unit)
    out = []
    for t, got in per.items():
        m = IR.named(CR.unit_member(SRW.template(t), len(got)))
        data = {"rows": {p: np.stack([r[g] for r, _o in got]) for g, (p, _w) in enumerate(m.rows)},
                "outs": {p: np.stack([o[g] for _r, o in got]) for g, (p, _w) in enumerate(m.outs)}}
        each = len(got) // len(rows)
        out.append(IR.Span(rid, m, data, [[i * each, (i + 1) * each, rid, i] for i in range(len(rows))], program, group=each))
    return tuple(out)
