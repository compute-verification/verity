"""Tile units under C-Flock's block (``Shape.tile_sub``): each tile unit of Pearl-C's (`pc8`) and Pearl-C4's (`pc4`) tile
check cut into sub-tiles of a × b words, and under hiding its tile digest into a digest unit per tile row and one over
them.  It is opt-in: a shape without ``tile_sub`` builds today's Programs, which this module leaves alone.

C-Flock proves one instance in at most a 2^27-bit block (`verity_flock.circuit.K_MAX`), and the scheme's 64 × 64 tile is
4,096 words' chains over k in one instance, past the block at every served k.  A sub-tile is today's tile template at
a × b (``pc8.templates(k, (a, b), dev)["tile"]``, ``pc4.templates(k, (a, b))["tile"]``), so it computes its words as the
tile does; (2, 2) is deployed (`TILE_SUB`), and (1, 1), one word a unit, fits further in k.  Under hiding:

    template                    one instance per         reads                                   computes and returns
    Pc8SubTile{K,TM=a,TN=b,DEV} sub-tile of a × b words  its a rows A′, P_A and b rows B̃, P_B    its a sub-cu rows, each
      (Pc4SubTile{K,TM=a,TN=b})                                                                  an output port
    PouwPearlRowChunks{TN,SB}   row of a tile            the row's C̃ and U, by its TN / b        the row's chunk digests
                                                         sub-tiles' rows
    PouwPearlTileLeaf{TM,TN}    tile                     its TM rows' chunk digests              the tile digest

A hiding sub-tile, `SubTile8Def` or `SubTile4Def`, calls today's tile template at a × b and returns output r as row r of
its C̃ then row r of its U (wiring only), so each committed sub-cu row is one output port and C-Flock commits it as a
one-port v2 row of exactly its words, the output rows the hidden-output soundness proofs take; at a = 1 too, where the
tile template's two outputs (C̃, U) would be a two-port run.

A row's chunk digests are TurboSHAKE128 (D = 0x21) of each 32-column chunk of its C̃ ‖ U and the leaf TurboSHAKE128
(D = 0x22) over every row's in order (`leaves.PearlRowChunks`, `PearlTileLeaf`), so the leaf's output is today's
`PearlTileDigest`, byte for byte, and the roots return today's outputs.  Every gate of today's tile unit is in exactly one
sub-tile or digest unit, and each sub-tile, row digest and leaf is one unit.  Unhidden, a tile is its sub-tiles alone,
each today's tile template, whose C̃ and U are the root's outputs.

The pieces pass their values in two more v2 rows (`leaves`' sub-cu, one tile row of a sub-tile's C̃ and U, and
tile-chunks, a tile row's chunk digests then zero words to one whole block; `PORTS8`, `PORTS4`, by the sub-tile's a), each
in the call's session and each one output port of its writer, and neither reads a public input region.  A drawn sub-tile
is proved with its whole tile (`plan.tile_layout`): the tile's other sub-tiles, under hiding its row digests and its leaf,
and its rows of A and of B, as today's tile is.
"""
from __future__ import annotations

from functools import lru_cache

from verity.commitments.rowleaf import ROLE_W, ROLE_X
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import array_of, tuple_of
from verity.ir.types import Array, Tuple

from ..protocol import DomainError
from ..schemes import pearl_c as C
from . import leaves as L
from . import pc4, pc8, rowk
from . import words as W

#: the sub-tile sides a cut takes (each of a and b), and the deployed sub-tile
SUBS = (1, 2, 4, 8, 16)
TILE_SUB = (2, 2)
#: the roles whose units sit on a grid, `pc4.Unit.where` (row, column): (sub-)tiles, row digests and tile leaves
GRID = ("tile", "row_digest", "tile_leaf")
OK = L.OK


def check(shape) -> None:
    sub, (tm, tn) = shape.tile_sub, shape.tile
    if not (type(sub) is tuple and len(sub) == 2 and all(type(v) is int and v in SUBS for v in sub)
            and tm % sub[0] == 0 and tn % sub[1] == 0):
        raise DomainError(f"a tile cut is tile_sub = (a, b), a and b each one of {SUBS}, a | TM and b | TN: {shape}")
    if shape.hidden and -(-tn // C.CHUNK) > L.TILE_CHUNKS:
        raise DomainError(f"a hiding tile cut's row digests hold {L.TILE_CHUNKS} chunks (TN <= {C.CHUNK * L.TILE_CHUNKS}): "
                          f"{shape}")


def _sub_tile_def(scheme: str, mod, tile, params: tuple) -> CompositeDefinition:
    """A hiding sub-tile: ``mod``'s tile template at TM × TN (``tile(S_)``), its outputs rewired as one per tile row, row r
    of C̃ then row r of U, each its sub-cu row and the port C-Flock commits as that one v2 row."""
    def body(B, S_, *args):
        out = B.call(tile(S_), *args)
        return tuple_of(*(array_of([out[0][r][j] for j in range(S_.TN)] + [out[1][r][j] for j in range(S_.TN)])
                          for r in range(S_.TM)))

    sig = lambda S_: (mod._tile_sig(S_)[0], Tuple(*[Array(2 * S_.TN, W.F32)] * S_.TM))        # noqa: E731
    return CompositeDefinition(
        f"{scheme}SubTile", 1, params, sig, body,
        doc=f"`{scheme}Tile` as a hiding sub-tile's unit: one output per tile row, its TN C̃ words then its TN U words (the "
            "sub-cu row; wiring only).", conformance="protocol-defined")


SubTile8Def = _sub_tile_def("Pc8", pc8, lambda S_: pc8.templates(S_.K, (S_.TM, S_.TN), S_.DEV)["tile"],
                            ("K", "TM", "TN", "DEV"))
SubTile4Def = _sub_tile_def("Pc4", pc4, lambda S_: pc4.templates(S_.K, (S_.TM, S_.TN))["tile"], ("K", "TM", "TN"))


@lru_cache(maxsize=16)
def sub_tile8(k: int, sub: tuple[int, int], dev: str) -> SpecializedDefinition:
    return bind(SubTile8Def, K=k, TM=sub[0], TN=sub[1], DEV=dev)


@lru_cache(maxsize=16)
def sub_tile4(k: int, sub: tuple[int, int]) -> SpecializedDefinition:
    return bind(SubTile4Def, K=k, TM=sub[0], TN=sub[1])


def digests(B, subs, m: int, n: int, tile: tuple[int, int], sub: tuple[int, int]):
    """Under hiding, the tile digests from the sub-tiles' sub-cu rows (``subs``: by sub-tile row, then column), in a root
    body: per row of A one batch of its chunk digests over the tile columns, then per tile row one batch of tile leaves
    over the tile columns, each reading its TM rows'; by tile row and column, as today's hiding tiles return them."""
    (tm, tn), (sa, sb) = tile, sub
    nt, q = n // tn, tn // sb
    chunks, leaf = L.row_chunks(tn, sb), L.tile_leaf(tm, tn)
    part = lambda i, cs, f: array_of([array_of([subs[i // sa][cs * q + p][i % sa][f * sb + j] for j in range(sb)])  # noqa: E731
                                      for p in range(q)])
    rows = [B.batch(chunks, array_of([part(i, cs, 0) for cs in range(nt)]), array_of([part(i, cs, 1) for cs in range(nt)]),
                    axes=(0, 0)) for i in range(m)]
    return array_of([B.batch(leaf, array_of([array_of([rows[rs * tm + r][cs] for r in range(tm)]) for cs in range(nt)]),
                             axes=(0,)) for rs in range(m // tm)])


# ---- the roots: today's signatures, each tile unit cut ------------------------------------------------------------------

def _units8(B, S_, a, ea, fa, fb, bc, pb, hidden: bool):
    return pc8._call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb, hidden, getattr(S_, "KS", None),
                           (S_.SA, S_.SB))


def _units4(B, S_, a, ea, fa, fb, bc, bs, pb, hidden: bool):
    return pc4._call_units(B, S_.K, (S_.TM, S_.TN), a, ea, fa, fb, bc, bs, pb, hidden, getattr(S_, "KS", None),
                           (S_.SA, S_.SB))


def _check8(B, S_, a, ea, b, eb, fa, fb, g, hidden: bool):
    bc, pb, ok_b = pc8._weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g, row_k=getattr(S_, "KS", None))
    return tuple_of(*_units8(B, S_, a, ea, fa, fb, bc, pb, hidden), ok_b)


def _check4(B, S_, a, ea, b, eb, fa, fb, g, hidden: bool):
    bc, bs, pb, ok_b = pc4._weight_units(B, S_.K, b, eb, fa, fb, g, row_k=getattr(S_, "KS", None))
    return tuple_of(*_units4(B, S_, a, ea, fa, fb, bc, bs, pb, hidden), ok_b)


def _roots(scheme: str, mod, units, check_body, block: tuple) -> dict:
    """One scheme's roots under the tile cut, without and with the row cut (``rowk``): each side and hiding's, today's
    signature with the sub-tile's SA and SB bound."""
    call_hidden = lambda S_: (mod._call_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK)))                # noqa: E731
    check_hidden = lambda S_: (mod._check_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK), Array(S_.N, OK)))  # noqa: E731
    out = {}
    for rowk_ in (False, True):
        params = block + (("KS",) if rowk_ else ()) + ("SA", "SB")
        cut = "RowKTileSub" if rowk_ else "TileSub"
        doc = "with each tile unit cut into SA x SB sub-tiles (`tilesub`)" + (" and each row unit every KS positions "
                                                                               "(`rowk`)." if rowk_ else ".")
        for side, hidden, sig in (("call", False, mod._call_sig), ("call", True, call_hidden),
                                  ("check", False, mod._check_sig), ("check", True, check_hidden)):
            name = f"{scheme}{side.title()}{'Hidden' if hidden else ''}"
            body = ((lambda h: lambda B, S_, *a: tuple_of(*units(B, S_, *a, h))) if side == "call" else
                    (lambda h: lambda B, S_, *a: check_body(B, S_, *a, h)))(hidden)
            out[rowk_, side, hidden] = CompositeDefinition(name + cut, 1, params, sig, body, doc=f"`{name}` " + doc,
                                                           conformance="protocol-defined")
    return out


_ROOTS8 = _roots("Pc8", pc8, _units8, _check8, ("M", "K", "N", "TM", "TN", "DEV"))
_ROOTS4 = _roots("Pc4", pc4, _units4, _check4, ("M", "K", "N", "TM", "TN"))
ROOTS = tuple(_ROOTS8.values()) + tuple(_ROOTS4.values())


def _bind(roots: dict, shape, side: str, **dev) -> SpecializedDefinition:
    ks = {} if shape.row_k is None else {"KS": shape.row_k}
    return bind(roots[shape.row_k is not None, side, shape.hidden], M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1], SA=shape.tile_sub[0], SB=shape.tile_sub[1], **dev, **ks)


def root8(shape, side: str) -> SpecializedDefinition:
    """``pc8.call`` or ``check`` (``side``) of a shape with ``tile_sub``; its weight is the shape's without it."""
    return _bind(_ROOTS8, shape, side, DEV=shape.dev)


def root4(shape, side: str) -> SpecializedDefinition:
    """``pc4.call`` or ``check`` (``side``) of a shape with ``tile_sub``."""
    return _bind(_ROOTS4, shape, side)


# ---- the partition -------------------------------------------------------------------------------------------------------

#: the tile's roles under the cut, in the order a call's root body first uses them; a side's are its rows' (`pc4.ROLES`
#: or `rowk.ROLES`) and, but the weight's, these (the digest units only hiding)
ROLES = ("tile", "row_digest", "tile_leaf")
ROLE_OF = {id(SubTile8Def): "tile", id(SubTile4Def): "tile", id(L.PearlRowChunksDef): "row_digest",
           id(L.PearlTileLeafDef): "tile_leaf"}


def _flat(pieces: tuple) -> list:
    head, first, forms, tail = pieces
    return [head, first, *forms, tail]


def _side(b: list, a: list, tiles: list, side: str) -> list[SpecializedDefinition]:
    return {"weight": b, "call": a + tiles, "check": b + a + tiles}[side]


def _tiles(shape, sub_tile: SpecializedDefinition, hidden_sub_tile: SpecializedDefinition) -> list[SpecializedDefinition]:
    (tm, tn), sb = shape.tile, shape.tile_sub[1]
    return [hidden_sub_tile, L.row_chunks(tn, sb), L.tile_leaf(tm, tn)] if shape.hidden else [sub_tile]


def unit_templates8(shape, side: str) -> list[SpecializedDefinition]:
    """The side's unit templates under the cut, in the order its root body first uses them (`pc8.query`)."""
    if shape.row_k is None:
        tp = pc8.templates(shape.k, pc8.TILE, shape.dev)
        b, a = [tp["row_b"]], [tp["row_a"]]
    else:
        b, a = (_flat(rowk.pieces8(shape.k, shape.row_k, shape.dev, s)) for s in ("b", "a"))
    return _side(b, a, _tiles(shape, pc8.templates(shape.k, shape.tile_sub, shape.dev)["tile"],
                              sub_tile8(shape.k, shape.tile_sub, shape.dev)), side)


def unit_templates4(shape, side: str) -> list[SpecializedDefinition]:
    """The side's unit templates under the cut, in the order its root body first uses them (`pc4.query`)."""
    if shape.row_k is None:
        tp = pc4.templates(shape.k)
        b, a = [tp["row_b"]], [tp["row_a"]]
    else:
        b, a = (_flat(rowk.pieces4(shape.k, shape.row_k, s)) for s in ("b", "a"))
    return _side(b, a, _tiles(shape, pc4.templates(shape.k, shape.tile_sub)["tile"], sub_tile4(shape.k, shape.tile_sub)),
                 side)


# ---- the ports -----------------------------------------------------------------------------------------------------------

def _ports(data_row: str, a_ports: tuple, b_ports: tuple, a: int) -> dict:
    """The tile's ports under the cut at a sub-tile of ``a`` rows, as `pc4.PORTS`: a sub-tile reads its a rows of A's and b
    rows of B's data rows and writes its a sub-cu rows, each its own output port (``SubTile``'s output r, row r of the
    sub-tile), so each is a one-port v2 row; a row digest reads the row's sub-cu rows, one per sub-tile, and writes its
    tile-chunks row; the leaf reads the tile's TM tile-chunks rows and writes the tile row today's tile writes."""
    return {"tile": {"rows": ((data_row, ROLE_X, a_ports), (data_row, ROLE_W, b_ports),
                              *(("sub-cu", ROLE_X, (r,)) for r in range(a))), "regions": {}},
            "row_digest": {"rows": (("sub-cu", ROLE_X, ("c", "u")), ("tile-chunks", ROLE_X, (0,))), "regions": {}},
            "tile_leaf": {"rows": (("tile-chunks", ROLE_X, ("d",)), ("tile", ROLE_X, (0,))), "regions": {}}}


#: the tile pieces' ports by the sub-tile's row count a
SUB8 = {a: _ports("fp8-p", ("ac", "pa"), ("bc", "pb"), a) for a in SUBS}
SUB4 = {a: _ports("nvfp4-p", ("ac", "as", "pa"), ("bc", "bs", "pb"), a) for a in SUBS}
#: each unit's ports under hiding with the tile cut, by a (`pc4.Cut`): today's rows' or the row cut's, and the tile's pieces'
PORTS8, PORTS8_ROWK = ({a: rows | SUB8[a] for a in SUBS} for rows in (pc8.PORTS, rowk.PORTS8))
PORTS4, PORTS4_ROWK = ({a: rows | SUB4[a] for a in SUBS} for rows in (pc4.PORTS, rowk.PORTS4))


__all__ = ["GRID", "PORTS4", "PORTS4_ROWK", "PORTS8", "PORTS8_ROWK", "ROLES", "ROLE_OF", "ROOTS", "SUB4", "SUB8", "SUBS",
           "SubTile4Def", "SubTile8Def", "TILE_SUB", "check", "digests", "root4", "root8", "sub_tile4", "sub_tile8",
           "unit_templates4", "unit_templates8"]
