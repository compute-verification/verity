"""`registry.tp_allreduce`'s all-reduces cut at a producer unit's output grain, so that each rank's producer output is wired
straight into the unit that reduces it (a combine's grain is its producer unit's output grain):

    AllReduceResidualMnnvlCoordinate_v1{WORLD,SHOT}
                            (col: Array<WORLD, bf16>, res: bf16) -> r: bf16   (one coordinate of AllReduceResidualMnnvl_v1)
    AllReduceResidualMnnvlAt_v1{WORLD,N,C,AT,SHOT}
                            (parts: Array<WORLD, Array<C, bf16>>, res: Array<N, bf16>) -> r: Array<C, bf16>
                                                                              (chunk AT of AllReduceResidualMnnvl_v1's r)
    RowChunks_v1{N,C}       (x: Array<N, bf16>) -> Tuple<Array<C, bf16>, ...>   (x's N / C chunks, each an output row)
    AllReduceMultimemRow_v1{WORLD,N,ROWS,ROW,TREES,FTZ}
                            (parts: Array<WORLD, Array<N, bf16>>) -> Array<N, bf16>   (row ROW of AllReduceMultimem_v1)
    AllReduceCustomRow_v1{WORLD,N,ROWS,ROW,STAGE}
                            (parts: Array<WORLD, Array<N, bf16>>) -> Array<N, bf16>   (row ROW of AllReduceCustom_v1)

`coordinate_binding` names the unit of one coordinate of a served all-reduce (a binding `route_binding` gives, or the fused
`AllReduceRmsNormMnnvl_v1`), and `row_binding` the unit of one row of an n-row one:

* A coordinate. FlashInfer's MNNVL sum adds ranks 0..WORLD-1 onto +0 at every element at either shot (the two-shot's
  reduce-scatter changes which rank sums a chunk, not the order it sums in), and NCCL's tree is one chain for every element, so
  their coordinate units are `tp_allreduce`'s coordinates at the call's statics. Multimem reduces an element at its owner's tree
  (`multimem_owners`, TREES one per owner, a node record's) and the custom 2-stage kernel from its owner (`custom_owners`), so the
  coordinate's position enters as the static it decides: TREE, or START.
* The fused MNNVL. Its norm reads the coordinate only as the bf16 new residual r (one-shot's ``f32(bf16(r * r))`` and two-shot's
  FFMA of ``f32(r)`` are both computed from r), so a coordinate's unit returns r: the all-reduced coordinate plus the residual
  (`AllReduceResidualMnnvlCoordinate_v1`), and `tp_allreduce.RmsNormMnnvl_v1` reads the N r's. No f32 intermediate crosses the
  cut.
* A row. An offset-free route (`tp_allreduce.offset_free`) reduces every row of an n-row call alike, so its row unit is the route's
  Definition at one row (`route_binding` at rows = 1). Multimem with more than one tree and the custom 2-stage kernel reduce an
  element by its offset in the call, so their row units take the call's ROWS and the row's position ROW. A row of a call may be
  a chunk of one token's elements: a fused MNNVL call is one token, and its chunk AT of C elements is
  `AllReduceResidualMnnvlAt_v1`, which reads the residual row whole, as `rmsnorm_fused_split.RMSNormFusedCudaAdd_v1` does, so a
  residual written as one row (the MoE's) is read without being cut.
* A producer's whole row. A wire reads its writer's output row whole, so a rank's partial written as one row of N elements (the
  MoE's `ResidualAdd_v2{N=6144}`) reaches the chunk units through `RowChunks_v1{N,C}`, a copy of it as N / C output rows.
  Unchunked, a token's row of 6,144 is past one C-Flock block: the fused call's unit by its ANDs alone, and every all-reduce of
  it with its rows' slots (verity/definitions/tests/test_tp_combine.py).
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.circuits.refs import array_of, coll_concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import prims as P
from verity.definitions.registry import tp_allreduce as TA
from verity.definitions.registry.prims import BF16

__all__ = ["AllReduceCustomRow", "AllReduceMultimemRow", "AllReduceResidualMnnvlAt", "AllReduceResidualMnnvlCoordinate",
           "RowChunks", "coordinate_binding", "row_binding"]


@composite("AllReduceResidualMnnvlCoordinate", 1, ["WORLD", "SHOT"],
           lambda S: ((("col", Array(int(S.WORLD), BF16)), ("res", BF16)), BF16),
           conformance="not-yet-established: one coordinate of AllReduceResidualMnnvl_v1, the r of the exact-model-tested "
                       "AllReduceRmsNormMnnvl_v1, node for node; not captured on its own "
                       "(verity/definitions/tests/test_tp_combine.py)")
def AllReduceResidualMnnvlCoordinate(B, S, col, res):
    """One coordinate of `AllReduceResidualMnnvl_v1`: ``r = AllReduceMnnvlCoordinate_v1(col) + res`` (`Bf16AddF2fp_v1`), the
    fused kernel's new residual, which is all its norm reads of the coordinate."""
    a = B.call(bind(TA.AllReduceMnnvlCoordinate, WORLD=S.WORLD, SHOT=TA.strategy(S.SHOT)), col)
    return B.call(P.Bf16AddF2fp, a, res)


def _at_refuse(S: dict) -> str | None:
    """Why `AllReduceResidualMnnvlAt_v1`'s statics bound so far name no chunk of the residual row, or None."""
    N, C, AT = (S.get(k) for k in ("N", "C", "AT"))
    if None in (N, C, AT):
        return None
    if not (all(type(v) is int for v in (N, C, AT)) and C >= 1 and N % C == 0 and 0 <= AT < N // C):
        return f"chunk AT={AT!r} of C={C!r} elements is not one of the N / C whole chunks of a residual row of N={N!r}"
    return None


@composite("AllReduceResidualMnnvlAt", 1, ["WORLD", "N", "C", "AT", "SHOT"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.C, BF16))), ("res", Array(S.N, BF16))), Array(S.C, BF16)),
           conformance="not-yet-established: elements [AT*C, (AT+1)*C) of AllReduceResidualMnnvl_v1, the r of the "
                       "exact-model-tested AllReduceRmsNormMnnvl_v1, node for node; not captured on its own "
                       "(verity/definitions/tests/test_tp_combine.py)",
           refuse=_at_refuse)
def AllReduceResidualMnnvlAt(B, S, parts, res):
    """`AllReduceResidualMnnvl_v1{WORLD,N,SHOT}`'s r on its elements [AT·C, (AT+1)·C): that Definition at width C on the ranks'
    chunk ``parts`` and the chunk of ``res``, the whole residual row (element-wise, so any chunk of its columns is it at that
    width on those columns)."""
    at = bind(TA.AllReduceResidualMnnvl, WORLD=S.WORLD, N=S.C, SHOT=TA.strategy(S.SHOT))
    return B.call(at, parts, res[S.AT * S.C:(S.AT + 1) * S.C])


def _chunks_refuse(S: dict) -> str | None:
    """Why `RowChunks_v1`'s statics bound so far cut no row into whole chunks, or None."""
    N, C = S.get("N"), S.get("C")
    if None in (N, C):
        return None
    if not (type(N) is int and type(C) is int and C >= 1 and N % C == 0):
        return f"C={C!r} elements a chunk does not cut a row of N={N!r} into whole chunks"
    return None


@composite("RowChunks", 1, ["N", "C"],
           lambda S: ((("x", Array(S.N, BF16)),), Tuple(*[Array(S.C, BF16)] * (S.N // S.C))),
           conformance="protocol-defined: the combine's own copy of a producer's row as chunk rows, no gate; no served kernel "
                       "computes it", refuse=_chunks_refuse)
def RowChunks(B, S, x):
    """``x``'s N / C chunks of C elements in order, chunk j its elements [j·C, (j+1)·C): each a return field, so each its own
    output row, which a chunk unit's cut row reads whole. A copy: no gate."""
    return tuple_of(*(x[j * S.C:(j + 1) * S.C] for j in range(S.N // S.C)))


def _row(S) -> tuple[int, int, int]:
    """(N, ROWS, ROW), ROW one of the ROWS rows."""
    N, rows, row = int(S.N), int(S.ROWS), int(S.ROW)
    if N < 1 or not 0 <= row < rows:
        raise ValueError(f"N={N}, ROWS={rows}, ROW={row}: a row of N >= 1 elements, ROW one of 0..ROWS-1")
    return N, rows, row


def _cut(runs, N: int, row: int) -> list[tuple[int, int, object]]:
    """The runs [(lo, hi, x)] of a ROWS * N call that meet row `row`, each cut to it and offset to the row's start."""
    lo0, hi0 = row * N, (row + 1) * N
    return [(max(lo, lo0) - lo0, min(hi, hi0) - lo0, x) for lo, hi, x in runs if lo < hi0 and hi > lo0]


@composite("AllReduceMultimemRow", 1, ["WORLD", "N", "ROWS", "ROW", "TREES", "FTZ"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(S.N, BF16)),
           conformance="not-yet-established: row ROW of the exact-model-tested AllReduceMultimem_v1{SIZE=ROWS*N}, its vectors' "
                       "Calls of the same AllReduceMultimemVector_v1; not captured on its own "
                       "(verity/definitions/tests/test_tp_combine.py)")
def AllReduceMultimemRow(B, S, parts):
    """Row ROW of `AllReduceMultimem_v1{WORLD,SIZE=ROWS*N,TREES,FTZ}`, the call's flattened [ROWS, N]: each run of
    `multimem_runs` over the call cut to the row, one `AllReduceMultimemVector_v1{TREE}` per 16-byte vector (a row of whole
    vectors meets each owner's slice, itself whole vectors, in whole vectors)."""
    W, V = TA.ranks(S.WORLD), TA.MULTIMEM_VECTOR
    N, rows, row = _row(S)
    if N % V:
        raise ValueError(f"N={N}: a row of torch's multimem all-reduce is whole {V}-element vectors")
    outs = []
    for lo, hi, tree in _cut(TA.multimem_runs(W, rows * N, S.TREES), N, row):
        vectors = Array((hi - lo) // V, Array(V, BF16))
        cols = array_of([parts[r][lo:hi].reshape(vectors) for r in range(W)])
        outs.append(B.batch(bind(TA.AllReduceMultimemVector, WORLD=W, TREE=tree, FTZ=bool(S.FTZ)), cols, axes=(1,)))
    return coll_concat(Array(N, BF16), outs)


@composite("AllReduceCustomRow", 1, ["WORLD", "N", "ROWS", "ROW", "STAGE"],
           lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(S.N, BF16)),
           conformance="not-yet-established: row ROW of the exact-model-tested AllReduceCustom_v1{SIZE=ROWS*N}, its elements' "
                       "Calls of the same AllReduceCustomCoordinate_v1; not captured on its own "
                       "(verity/definitions/tests/test_tp_combine.py)")
def AllReduceCustomRow(B, S, parts):
    """Row ROW of `AllReduceCustom_v1{WORLD,SIZE=ROWS*N,STAGE}`, the call's flattened [ROWS, N]: each run of `custom_owners`
    over the call cut to the row, one `AllReduceCustomCoordinate_v1{START=owner}` per element."""
    W = TA.ranks(S.WORLD)
    N, rows, row = _row(S)
    outs = []
    for lo, hi, owner in _cut(TA.custom_owners(W, rows * N, int(S.STAGE)), N, row):
        cols = array_of([parts[r][lo:hi] for r in range(W)])
        outs.append(B.batch(bind(TA.AllReduceCustomCoordinate, WORLD=W, START=owner), cols, axes=(1,)))
    return outs[0] if len(outs) == 1 else coll_concat(Array(N, BF16), outs)


#: by all-reduce Definition, its static holding the call's element count
_SIZE = {TA.AllReduceMnnvl: "N", TA.AllReduceResidualMnnvl: "N", TA.AllReduceRmsNormMnnvl: "N", TA.AllReduceMultimem: "SIZE",
         TA.AllReduceCustom: "SIZE", TA.AllReduceNcclTree: "SIZE"}


def _call(defn, statics: dict) -> tuple[str, dict, int, int]:
    """(``Name_vN``, statics, WORLD, the call's element count) of one of `tp_allreduce`'s all-reduces."""
    name = f"{defn.name}_v{defn.version}"
    if defn not in _SIZE:
        raise ValueError(f"{name}: not one of tp_allreduce's all-reduces, whose combine units are stated here")
    s = dict(statics)
    return name, s, TA.ranks(s["WORLD"]), int(s[_SIZE[defn]])


def _owner(runs, pos: int):
    return next(x for lo, hi, x in runs if lo <= pos < hi)


def _mnnvl_coordinate(unit):
    return lambda s, W, size, pos: (unit, {"WORLD": W, "SHOT": TA.strategy(s["SHOT"])})


#: by all-reduce Definition, its coordinate unit at (statics, WORLD, the call's element count, the position)
_COORDINATE = {
    TA.AllReduceMnnvl: _mnnvl_coordinate(TA.AllReduceMnnvlCoordinate),
    TA.AllReduceResidualMnnvl: _mnnvl_coordinate(AllReduceResidualMnnvlCoordinate),
    TA.AllReduceRmsNormMnnvl: _mnnvl_coordinate(AllReduceResidualMnnvlCoordinate),
    TA.AllReduceMultimem: lambda s, W, size, pos: (
        TA.AllReduceMultimemCoordinate,
        {"WORLD": W, "TREE": _owner(TA.multimem_runs(W, size, s["TREES"]), pos), "FTZ": bool(s["FTZ"])}),
    TA.AllReduceCustom: lambda s, W, size, pos: (
        TA.AllReduceCustomCoordinate, {"WORLD": W, "START": _owner(TA.custom_owners(W, size, int(s["STAGE"])), pos)}),
    TA.AllReduceNcclTree: lambda s, W, size, pos: (
        TA.AllReduceNcclTreeCoordinate, {"WORLD": W, "CHAIN": TA.nccl_chain(s["CHAIN"], W)}),
}


def coordinate_binding(defn, statics: dict, pos: int) -> tuple[object, dict]:
    """The Definition and statics of the unit reducing element `pos` of the all-reduce `defn` at `statics` (`route_binding`'s, or
    the fused `AllReduceRmsNormMnnvl_v1`'s or its `AllReduceResidualMnnvl_v1`'s): the call's element `pos` is that unit on every
    rank's word at `pos` (and, fused, the residual's word there), wherever the producer's grain cuts the call."""
    name, s, W, size = _call(defn, statics)
    if not 0 <= int(pos) < size:
        raise ValueError(f"position {pos}: {name}'s elements are 0..{size - 1}")
    return _COORDINATE[defn](s, W, size, int(pos))


def _multimem_row(s, W, N, rows, row):
    if len(set(TA.multimem_trees(s["TREES"], W))) == 1:
        return TA.AllReduceMultimem, {"WORLD": W, "SIZE": N, "TREES": s["TREES"], "FTZ": bool(s["FTZ"])}
    return AllReduceMultimemRow, {"WORLD": W, "N": N, "ROWS": rows, "ROW": row, "TREES": s["TREES"], "FTZ": bool(s["FTZ"])}


def _custom_row(s, W, N, rows, row):
    if int(s["STAGE"]) == 1:
        return TA.AllReduceCustom, {"WORLD": W, "SIZE": N, "STAGE": 1}
    return AllReduceCustomRow, {"WORLD": W, "N": N, "ROWS": rows, "ROW": row, "STAGE": int(s["STAGE"])}


def _fused_row(s, W, N, rows, row):
    return AllReduceResidualMnnvlAt, {"WORLD": W, "N": N * rows, "C": N, "AT": row, "SHOT": TA.strategy(s["SHOT"])}


#: by all-reduce Definition, its row unit at (statics, WORLD, the row's width, the row count, the row); a fused call is one token,
#: so its rows are chunks of that token's elements
_ROW = {
    TA.AllReduceResidualMnnvl: _fused_row,
    TA.AllReduceRmsNormMnnvl: _fused_row,
    TA.AllReduceMnnvl: lambda s, W, N, rows, row: (TA.AllReduceMnnvl, {"WORLD": W, "N": N, "SHOT": TA.strategy(s["SHOT"])}),
    TA.AllReduceNcclTree: lambda s, W, N, rows, row: (
        TA.AllReduceNcclTree, {"WORLD": W, "SIZE": N, "CHAIN": TA.nccl_chain(s["CHAIN"], W)}),
    TA.AllReduceMultimem: _multimem_row,
    TA.AllReduceCustom: _custom_row,
}


def row_binding(defn, statics: dict, rows: int, row: int) -> tuple[object, dict]:
    """The Definition and statics of the unit reducing row `row` of the all-reduce `defn` at `statics` (`route_binding`'s, or the
    fused `AllReduceRmsNormMnnvl_v1`'s or its `AllReduceResidualMnnvl_v1`'s) over `rows` rows of equal width: the call's row `row`
    is that unit on every rank's row `row` (and, fused, the whole residual row). An offset-free route's is the route at one row
    (`route_binding` at rows = 1), so every row of it is one template; the others' take ROWS and ROW, and a fused call's, its
    chunk AT = `row` of the token's N = C · `rows` elements (`AllReduceResidualMnnvlAt_v1`), whose r a separate
    `tp_allreduce.RmsNormMnnvl_v1` reads as one row."""
    name, s, W, size = _call(defn, statics)
    rows, row = int(rows), int(row)
    if rows < 1 or size % rows or not 0 <= row < rows:
        raise ValueError(f"{name} of {size} elements: {rows} rows of equal width, row {row} one of them")
    return _ROW[defn](s, W, size // rows, rows, row)
