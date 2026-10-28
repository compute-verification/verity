"""H1's window as one public Program: every R × C tile of a window's calls is one call of its call's contraction K's
template (`template`: one call of the tile unit, `h1zy.PouwH1TileZy` over `h1tile.PouwH1Tile`, at that K), so the verifier
derives the window's units from its own copy of the Program under the `Q_template_instances` query of its templates
(`query`), and a served path registers the trees it commits as the Program's parameters (`one_stage.served.register_served`).

* **The tile** is a parameter: ``tile`` is its R × C, the served path's 32 × 32 (`pouw_h1.TILE`) or h1_prove's 4 × 8
  smoke, and the window binds the tile unit once per K its linears have (`tile_unit`; ``unit`` gives another), so
  down_proj's calls, at K the intermediate size, instantiate their own template in the same Program. The tile's
  architecture (`h1_tile`'s ``arch``, sm_120 unless given) is its `h1device` record, which binds its exact accumulator and
  bounds its K; sm_90's binds the same step, so its Program is sm_120's.
* **The unit** reads its rows and its z/y record and proves the record (`h1zy`, RT4-INT-33): the H1 tile (`h1_tile`) on the
  rows' codes gives z and the receipt, and the unit returns the receipt and ``ok``, 1 exactly when the record is z and its
  dequant y = RN_bf16(RN32(RN32(RN32(z) s_a) s_w)) at each cell's row scale s_a and column scale s_w, which ride on the
  rows (`h1zy.ROW`: an A row is its K codes then its f32 s_a, a B row its K weight codes then their f32 s_w). The record
  is an input, so the unit's outputs are the receipt and one bit.
* **Templates.** K's template takes the tile's parameters with its A and B rows named for their K, ``a_k<K>`` and
  ``b_k<K>`` (`template_params`): `registered.reads` keys a unit's rows by its template's parameter names, so a mixed-K
  window registers in one read, A row i of K's units under ``a_k<K>[i]`` (a lone row, R or C = 1, under the parameter's
  name, `row_key`), each port with its own width and domain, and that row's circuit port is ``a_k<K>_<i>``
  (`window_ports`; B row j's ``b_k<K>_<j>``), every K's ports distinct.
* **K's circuit file** is C-Flock's statement of K's template on the rows `statement_rows` gives (its A and B rows, the
  record and the public row, each a ``row/v2`` row), which the auditor's lowering builds from this module and C-Flock's
  Python alone (numpy, no torch): flat, or templated (``flock-circuit/tmpl``) under a layout mode.
* **Linears and calls.** ``linears`` are the window's weights ``(name, K, n)``, each once, and ``calls`` its calls
  ``(linear, block)`` in order: a step's call of a linear reads A block ``block`` of its K and the linear's B rows.
* **Fixed slots.** The layout is protocol structure, so no token count becomes public. For each K the window's A rows are
  ``a/<K>``: blocks of ``M`` rows each (`slot`, a power of two of at least R, the token rows padded with zero rows), one
  block per input a step's calls of that K read. ``b/<K>`` is the B rows of the linears of that K, linear after linear,
  each linear's n rows once (a whole number of C): every call of a linear, in any step, reads the same rows, so a window
  of S steps registers Σn weight rows, not S × Σn. ``zy`` is one record per unit (`h1zy.RECORD`, 48 R C bits: its cells'
  s32 z, row-major, then their bf16 y), row u unit u's, the served ``scaled_mm_zy`` record of its tile (a padded row's
  cells z = 0 and y = dq(0, +0, s_w), which is +0 only for a finite positive s_w). ``salt`` is the units' one coin,
  ``position`` one row per unit (its number), and ``roots`` the window's seed-group roots. A call's statement seeds
  its units under the call's own roots instead, root(``a/<K>@<s>``) and root(``b/<K>``), at its public table
  (`PUBS`, `public_table`), which the verifier builds.
* **Chunk rows.** At a K whose `weights.chunks` n is above 1 (K > 4096) each A and B row is n leaves: row r's leaf
  n r + c is its codes [c K/n, (c + 1) K/n) and its scale (`h1zy.ROW` (K/n)). Every row count of K's values and tables
  is then n times its rows, every row index n times the row's plus its chunk, and K's template is over
  `h1zy.PouwH1TileZyChunks` at n (v2 over a J10-R tile), which rebuilds the unit's rows from its n R and n C chunk rows
  (`program` wraps the bare zy unit, `h1zy.unit_chunks`). At n = 1 all of this is the whole-row layout.
* **J10-R.** A unit under J10-R's noise (`h1tile.PouwH1Tile5` or its v6 under `h1zy`, through ``unit``) takes `J10R`
  after them: ``call``, its call's first unit, and ``blocks``, its first row and column block of the call's stream,
  one row of each a unit (`j10r_inputs`). Its statement's public row holds them too (`PUBLIC_J10R`), at the values the
  verifier derives from the layout (`call_blocks`). Every K's unit is in one form, and `tile_unit` binds none of J10-R's.
* **Units** are the tiles, call by call and row block major. Tile ``(c, ti, tj)`` is the unit on rows
  ``block M + [R ti, R ti + R)`` of ``a/<K>``, on rows ``[C tj, C tj + C)`` of its linear's in ``b/<K>`` and on row u of
  ``zy``, at position u, its number. Each call's tiles are one batch of K's template (`_body`), its A rows one repeated
  run a row block and its B rows the linear's run repeated, so the descriptor, its build and its digest grow with the
  calls and their row blocks, never with the tiles. The Program's result is each call's tiles' outputs in turn (their
  widths depend on K), and unit u's outputs are row u of the window's ``receipt`` and ``ok``. `unit_reads` gives, from
  the Program's own references, the row of each parameter that each of a unit's template parameters reads.
* **Ports.** ``salt`` is the units' coin (`In(kind="coin")`), ``position`` their positions (`In(kind="position")`), and the
  whole result a sink (`Out(sink=True)`, `_ports`), as the Program's outputs: the window is no seed, so the port rules
  follow the coin into every tile.
* **Names.** Every parameter is named ``prefix`` and its role (`roles`), so a served path can name each after the
  registered value its tree commits.
* **Tables.** A served round commits each of its calls' A rows and records before the round's coin (`TABLES`, `table`):
  one A table for each K its calls have, the M rows of each of those calls' A blocks (each row its codes and s_a) call
  after call in call order (`step_of`), and each call's z table, one record a unit of the call in its tile order. A
  unit's A rows open at its step's A table of its K and its record at its call's z table (`table_reads`), its B rows at
  the weights' ``b/<K>``, so a statement holds one call's units, one root a port (`table_reads_file`). The window-wide
  ``a/<K>`` and ``zy`` are the Program's parameters: trees of them registered at the window's close would be committed
  after their readers' coins. Each call's receipts (`RECEIPTS`, ``receipt@<c>``, one a unit in its tile order), which
  hang on nothing past the salt, its rows and the roots, are committed whole under its round's root too, and a
  statement then reads its units' receipts there as its output row o0 (`RECEIPT_PORT`).
* **A drawn leaf's statements** (`pouw.leaf_lanes`): a drawn unit is proved as its lanes and their
  combine, derived from the window's own template at its K (`leaf_unit`: `h1zy.lane` and `h1zy.combine` over the tile
  of its zy unit), so the window binds nothing more. Each runs under a one-call Program of its wrapper
  (`PouwH1WindowLane`, `PouwH1WindowCombine`; `leaf_program`), one unit, the same all period. Its reads are its leaf-local
  rows past the leaf's `place` in each value (`leaf_reads`, `leaf_reads_file`): the tile's own rows, its call's z table
  and receipts, and the leaf's partials, produced after the draw. A J10-R window's leaf (over `PouwH1Tile_v5`) is
  `h1zy`'s v2 statements, at a chunked K each lane's `h1zy.lane_chunk`s (`leaf_unit`'s ``ch``), and the combine that
  folds their oks; its chunks read the A and B tables' chunk rows, and its carries and ok rows are produced after the
  draw like the partials (`leaf_lanes`' J10-R leaf). v1's leaf is staged at whole rows only.
"""
from __future__ import annotations

import bisect
import functools
import weakref
from typing import Any, Callable, Iterable, Sequence

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ports import In, Out, Ports, forward
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import Affine, Coll, Strided, coll_concat, concat, runs
from verity.ml.circuits.types import Array, Tuple

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.pouw import weights as W
from verity.ml.protocols.pouw.circuit import h1tile as H
from verity.ml.protocols.pouw.circuit import h1zy as Z
from verity.ml.protocols.pouw.circuit import h1device as D
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import produced as PR
from verity.core.protocols.sampled_proofs.one_stage import registered as G

#: the tile unit's parameters, in order: its R A rows, its C B rows, its z/y record, the salt, its position, the roots
TILE_PARAMS = ("a", "b", "zy", "salt", "position", "roots")
#: the window's parameters after each K's ``a/<K>`` and ``b/<K>``
SHARED = ("zy", "salt", "position", "roots")
#: a J10-R tile unit's (`h1tile.PouwH1Tile5` and its v6 under `h1zy`): its call's first position and its blocks, one a unit
J10R = ("call", "blocks")
TILE_PARAMS_J10R, SHARED_J10R = TILE_PARAMS + J10R, SHARED + J10R
#: the unit's outputs (`h1zy`'s), each a row per unit of the window's outputs
OUTPUTS = ("receipt", "ok")


def slot(rows: int, tile: tuple[int, int]) -> int:
    """The A slot for ``rows`` token rows under an R × C ``tile``: the least power of two holding them, at least R."""
    if rows < 1:
        raise ValueError("a window has at least one token row")
    return max(int(tile[0]), 1 << (rows - 1).bit_length())


def roles(ks: Iterable[int], j10r: bool = False) -> tuple[str, ...]:
    """The window's parameters for contractions ``ks``, in order and before their ``prefix``: ``a/<K>`` and ``b/<K>`` for
    each K, ascending, then `SHARED` (a J10-R window's, ``j10r``, `SHARED_J10R`)."""
    return tuple(r for k in sorted(set(ks)) for r in (f"a/{k}", f"b/{k}")) + (SHARED_J10R if j10r else SHARED)


def is_j10r(program: Any) -> bool:
    """Whether the window (a `program`'s Program, or its function's bindings ``S``) takes J10-R tile units, whose
    parameters end in `J10R`."""
    S = program.fn.S() if isinstance(program, Program) else program
    return len(S.TILES[0][1].signature()[0]) == 2 + len(SHARED_J10R)


def h1_tile(k: int, tile: tuple[int, int], definition: Any = None, arch: str = "sm_120") -> Any:
    """The H1 tile at contraction ``k`` on architecture ``arch`` (`h1device.DEVICES`): `h1tile.PouwH1Tile` (or ``definition``,
    its v3) over ``GemmAccumulatorS8_v2{K}`` at the record's s8 binding, R × C ``tile``, every (i, j) pair. Refused
    (ValueError): an architecture with no record, or a K outside its record's domain."""
    dev = D.device(arch)
    if why := dev.refuse(int(k)):
        raise ValueError(why)
    r, c = tile
    exact = bind(BG.GemmAccumulatorS8, K=int(k), **dev.gemm)
    return bind(definition or H.PouwH1Tile, EXACT=exact, R=int(r), C=int(c),
                PAIRS=tuple((i, j) for i in range(r) for j in range(c)))


def tile_unit(k: int, tile: tuple[int, int], definition: Any = None, arch: str = "sm_120") -> Any:
    """The tile unit at contraction ``k``: `h1zy.PouwH1TileZy` over `h1_tile` (``definition`` its H1 tile, `h1tile.PouwH1Tile`
    unless given, or its v3; ``arch`` its architecture), which reads its unit's z/y record and proves it."""
    return Z.unit(h1_tile(k, tile, definition, arch))


def template_params(k: int, j10r: bool = False) -> tuple[str, ...]:
    """K's template's parameters, in order: the tile's (`TILE_PARAMS`, a J10-R unit's `TILE_PARAMS_J10R`), its A and B
    rows ``a_k<K>`` and ``b_k<K>``."""
    return (f"a_k{int(k)}", f"b_k{int(k)}") + (SHARED_J10R if j10r else SHARED)


def _template_signature(S: Any) -> tuple:
    params, ret = S.UNIT.signature()
    return tuple(zip(template_params(S.K, len(params) == len(TILE_PARAMS_J10R)), (t for _name, t in params))), ret


def _template_ports(S: Any) -> Ports:
    """The unit's parameters' ports, and the whole return a sink: wherever the template is (the window's batches, a
    statement's one-call Program) its result is the Program's outputs, and one fact for all of it keeps a call's batch
    one run of port facts, not two a tile."""
    decl = S.UNIT.defn.ports
    decl = decl(S.UNIT.S()) if callable(decl) else decl
    return Ports(params=decl.params if decl is not None else {}, ret=Out(sink=True))


PouwH1WindowTile = CompositeDefinition(
    "PouwH1WindowTile", 1, ("UNIT", "K"), _template_signature, lambda B, S, *args: B.call(S.UNIT, *args), register=False,
    ports=_template_ports,
    doc="A window's template at contraction K: one call of the tile unit UNIT, its parameters and their ports, with its A "
        "and B rows named a_kK and b_kK, and its whole return a sink.")


def template(k: int, unit: Any) -> Any:
    """K's template over the tile unit ``unit`` (`tile_unit`'s at K, or another of its parameters)."""
    return bind(PouwH1WindowTile, UNIT=unit, K=int(k))


#: K's statement's public row: the unit's salt, position and roots, in that order (a J10-R unit's then its call and blocks)
PUBLIC = ("salt", "position", "roots")
PUBLIC_J10R = PUBLIC + J10R
#: K's statement's output rows ``(schema, outputs)``: the receipt (o0) and ok (o1), one ``row/v2`` row each
STATEMENT_OUT_ROWS = ((G.SCHEMA_V2, (0,)), (G.SCHEMA_V2, (1,)))
#: the statement's output row of its units' receipts, which reads its call's receipt table when the window registers one
RECEIPT_PORT = "o0"


def statement_rows(template: Any, ports: dict[str, str] | None = None) -> list[tuple[str, str, int, list[str]]]:
    """The input rows of K's statement (``template`` `template`'s at K), each ``(port, schema, bits, params)`` as C-Flock's
    unit stager takes them (``a_k<K>[0]`` row 0 of ``a_k<K>``, `row_key`), in order: each A row and B row its own row, then the record
    ``zy`` and the public row ``pub`` (`PUBLIC`, a J10-R template's `PUBLIC_J10R`), every one a ``row/v2`` row of its
    leaves' bits (an A or B row 8 K/n + 32, `h1zy.ROW` of a chunk, n `weights.chunks` (K); the record 48 R C,
    `h1zy.RECORD`; ``pub`` 1600, a J10-R unit's 1792). ``ports`` names each A and B row's port by its `row_key`; by
    default `window_ports`' (``a_k<K>_<i>``, ``b_k<K>_<j>``), the ports credit's registered file reads. Its outputs are
    `STATEMENT_OUT_ROWS`."""
    params, _ret = template.signature()
    ty = dict(params)
    if [n for n, _t in params[2:]] not in (list(SHARED), list(SHARED_J10R)) or \
            not all(n.startswith(("a_k", "b_k")[x]) for x, (n, _t) in enumerate(params[:2])):
        raise ValueError(f"{template.id}: K's template's parameters are a_k<K>, b_k<K>, then {', '.join(SHARED)} (a "
                         f"J10-R unit's then {', '.join(J10R)})")
    public = PUBLIC_J10R if len(params) == 2 + len(SHARED_J10R) else PUBLIC

    def bits(p: str) -> int:
        name, _, row = p.partition("[")
        return sum((ty[name].elem if row else ty[name]).leaf_widths())

    keys = [(row_key(n, t.n, i), f"{n}_{i}") for n, t in params[:2] for i in range(t.n)]
    rows = [((ports or {}).get(key, port), [key]) for key, port in keys] + [("zy", ["zy"]), ("pub", list(public))]
    return [(port, G.SCHEMA_V2, sum(bits(p) for p in ps), ps) for port, ps in rows]


def _rc(S: Any) -> tuple[int, int]:
    k, t = S.TILES[0]
    (_, at), (_, bt) = t.signature()[0][:2]
    n = W.chunks(k)
    return at.n // n, bt.n // n


def _k(S: Any) -> dict[str, int]:
    return {name: k for name, k, _n in S.LINEARS}


def _b0(S: Any) -> dict[str, int]:
    """Each linear's first row in its K's ``b/<K>``."""
    out, at = {}, {}
    for name, k, n in S.LINEARS:
        out[name] = at.get(k, 0)
        at[k] = out[name] + n
    return out


def _tiles(S: Any) -> list[int]:
    """Each call's tile count."""
    r, c = _rc(S)
    n = {name: n for name, _k, n in S.LINEARS}
    return [(S.M // r) * (n[lin] // c) for lin, _blk in S.CALLS]


def _signature(S: Any) -> tuple:
    ks, tiles = _k(S), dict(S.TILES)
    params = []
    for k, t in S.TILES:
        (_, at), (_, bt) = t.signature()[0][:2]
        blocks = 1 + max(blk for lin, blk in S.CALLS if ks[lin] == k)
        params += [(f"a/{k}", Array(W.chunks(k) * blocks * S.M, at.elem)),
                   (f"b/{k}", Array(W.chunks(k) * sum(n for _name, kk, n in S.LINEARS if kk == k), bt.elem))]
    zy, salt, position, roots, *j10r = (t for _name, t in S.TILES[0][1].signature()[0][2:])
    units = sum(_tiles(S))
    params += [("zy", Array(units, zy)), ("salt", salt), ("position", Array(units, position)), ("roots", roots)]
    params += [(name, Array(units, t)) for name, t in zip(J10R, j10r)]
    ret = Tuple(*(Array(n, tiles[ks[lin]].signature()[1]) for (lin, _blk), n in zip(S.CALLS, _tiles(S))))
    return tuple((S.PREFIX + role, t) for role, t in params), ret


def _repeat(rows: Any, lo: int, hi: int, times: int, t: Any) -> Any:
    """Rows [lo, hi) of ``rows`` ``times`` times over, as a ``t``: one `Strided` part, whose size is no function of
    ``times``."""
    run = rows[lo:hi].refs
    if times > 1 and isinstance(run, Affine):
        run = Strided(run.space, run.idx, run.base, run.count, 0, 1, run.count * times)
    elif times > 1:
        run = concat([run] * times)
    return Coll(t, run)


def _body(B: Any, S: Any, *params: Any) -> Any:
    j10r = is_j10r(S)
    got = dict(zip(roles((k for k, _t in S.TILES), j10r), params))
    ks, tiles, b0, n_of = _k(S), dict(S.TILES), _b0(S), {name: n for name, _k, n in S.LINEARS}
    (r, c), outs, u0 = _rc(S), [], 0
    for (lin, blk), units in zip(S.CALLS, _tiles(S)):
        k, cols = ks[lin], n_of[lin] // c
        a, b, n, t = got[f"a/{k}"], got[f"b/{k}"], W.chunks(k), tiles[k]
        (_, at), (_, bt) = t.signature()[0][:2]
        rows_a = [_repeat(a, n * (blk * S.M + r * ti), n * (blk * S.M + r * ti + r), cols, Array(cols, at))
                  for ti in range(S.M // r)]
        a_t = Coll(Array(units, at), concat([x.refs for x in rows_a]))
        b_t = _repeat(b, n * b0[lin], n * (b0[lin] + n_of[lin]), S.M // r, Array(units, bt))
        mine = [got[name][u0:u0 + units] for name in ("zy", "position") + (J10R if j10r else ())]
        outs.append(B.batch(t, a_t, b_t, mine[0], got["salt"], mine[1], got["roots"], *mine[2:],
                            axes=(0, 0, 0, None, 0, None) + ((0, 0) if j10r else ())))
        u0 += units
    return coll_concat(_signature(S)[1], outs)


def _ports(S: Any) -> Ports:
    """The salt is the units' coin and ``position`` their positions, and the window's whole result is a sink: it is the
    Program's outputs, the one place a sink may reach besides a receipt port. A Definition with a coin and a position port
    and no sink in its return is a seed, whose body no port rule follows (`ports._coin_role`), so without it the tiles'
    coin flow would go unchecked."""
    return Ports(params={S.PREFIX + "salt": In(kind="coin"), S.PREFIX + "position": In(kind="position")},
                 ret=Out(sink=True))


PouwH1Window = CompositeDefinition(
    "PouwH1Window", 3, ("TILES", "LINEARS", "CALLS", "M", "PREFIX"), _signature, _body, register=False, ports=_ports,
    doc="One H1 window's tiles, call by call and row block major: TILES[K] on rows block M + [R ti, R ti + R) of a/K (each "
        "K's A blocks of M rows) and on rows [C tj, C tj + C) of the call's linear in b/K (each K's LINEARS' B rows once, "
        "linear after linear), the tile's z/y record (row u of zy for tile u), the salt, the tile's position and the "
        "window's roots, and a J10-R tile's call and blocks (row u of each); every parameter named PREFIX and its role.")


def program(linears: Sequence[tuple[str, int, int]], calls: Sequence[tuple[str, int]], m: int, tile: tuple[int, int],
            prefix: str = "", unit: Callable[[int], Any] | None = None) -> Program:
    """The window's Program: its ``linears`` ``(name, K, n)``, each weight once, its ``calls`` ``(linear, A block)`` in
    order, its A slot ``m`` and its R × C ``tile``, over the tile unit at each K (`tile_unit`, or ``unit(K)``), its
    parameters named ``prefix`` and their roles. Refused (ValueError): no linear or no call, a linear named twice or that no
    call reads, a call of no linear, a linear whose n is not whole blocks of C, a slot that is no power of two of at least
    R, an A block of a K that no call reads, a K `weights.chunks` refuses, or a unit whose rows aren't n R × n C chunk
    rows of K/n codes and their row's scale (`h1zy.ROW`, n = `weights.chunks` (K); a bare `h1zy.PouwH1TileZy` at n > 1
    is wrapped in `h1zy.unit_chunks`' `h1zy.PouwH1TileZyChunks`, v2 over a J10-R tile), whose record isn't z and y (`h1zy.RECORD`: a port carrying z alone leaves y unproved), that returns
    other than its receipt and ``ok``, or whose shared parameters differ by K."""
    linears = tuple((str(name), int(k), int(n)) for name, k, n in linears)
    calls = tuple((str(lin), int(blk)) for lin, blk in calls)
    r, c = (int(x) for x in tile)
    names = [name for name, _k, _n in linears]
    if not linears or not calls or len(set(names)) != len(names):
        raise ValueError(f"a window has linears, each named once, and calls: linears {linears}, calls {calls}")
    if any(lin not in names for lin, _blk in calls) or set(names) - {lin for lin, _blk in calls}:
        raise ValueError(f"every call is of a linear and every linear is called: linears {names}, calls {calls}")
    if m != slot(m, (r, c)) or any(k < 1 or n < c or n % c for _name, k, n in linears) or any(b < 0 for _l, b in calls):
        raise ValueError(f"a window's A slot is a power of two of at least {r} rows and its linears' B rows whole blocks "
                         f"of {c}: slot {m}, linears {linears}")
    ks = dict((name, k) for name, k, _n in linears)
    for k in sorted(set(ks.values())):
        used = sorted({blk for lin, blk in calls if ks[lin] == k})
        if used != list(range(len(used))):
            raise ValueError(f"every A block of K = {k} is read by some call: blocks {used}")
    make = unit or (lambda k: tile_unit(k, (r, c)))
    units = tuple((k, _chunked(k, make(k))) for k in sorted(set(ks.values())))
    shared = None
    for k, t in units:
        sig, ret = t.signature()
        n = W.chunks(k)
        if tuple(name for name, _t in sig) not in (TILE_PARAMS, TILE_PARAMS_J10R) or \
                (sig[0][1].n, sig[1][1].n) != (n * r, n * c):
            raise ValueError(f"the tile unit at K = {k} is no {r} × {c} tile of {n} chunk rows a row over {TILE_PARAMS} "
                             f"(or a J10-R unit's {TILE_PARAMS_J10R}): {sig}")
        if (sig[0][1].elem, sig[1][1].elem) != (Z.ROW(k // n), Z.ROW(k // n)):
            raise ValueError(f"the tile unit at K = {k} reads A and B rows of {sig[0][1].elem.leaves} and "
                             f"{sig[1][1].elem.leaves} bits, not {k // n} codes and the row's f32 scale (h1zy.ROW, "
                             f"{8 * (k // n) + 32} bits; K's {n} chunks, weights.chunks)")
        if sig[2][1] != Z.RECORD(r, c):
            raise ValueError(f"the tile unit at K = {k} reads a {sig[2][1].leaves}-bit record, not its {r} × {c} cells' z "
                             f"then y (h1zy.RECORD, {48 * r * c} bits): a record of z alone leaves the served y unproved")
        if tuple(name for name, _t in getattr(ret, "fields", ())) != OUTPUTS:
            raise ValueError(f"the tile unit at K = {k} returns {ret}, not {OUTPUTS}")
        if shared is not None and tuple(sig[2:]) != shared:
            raise ValueError(f"the tile unit at K = {k} takes other {SHARED} (or {J10R}) than at K = {units[0][0]}")
        shared = tuple(sig[2:])
    tiles = tuple((k, template(k, t)) for k, t in units)
    return Program(bind(PouwH1Window, TILES=tiles, LINEARS=linears, CALLS=calls, M=int(m), PREFIX=str(prefix)))


def _chunked(k: int, unit: Any) -> Any:
    """``unit`` at K: wrapped in `h1zy.unit_chunks`' `h1zy.PouwH1TileZyChunks` (v2 over a J10-R tile) at `weights.chunks`
    (K) when that is above 1 and ``unit`` is the bare zy unit, else ``unit``; ValueError on a K with no chunks."""
    n = W.chunks(k)
    return Z.unit_chunks(unit.bindings["TILE"], n) if n > 1 and getattr(unit, "defn", None) is Z.PouwH1TileZy else unit


def query(program: Program) -> dict[str, Any]:
    """The window's partition query: every instance of each of its templates (`template`, one per K, ascending) one unit."""
    return P.template_instances_query([t for _k, t in program.fn.bindings["TILES"]])


def row_key(name: str, rows: int, i: int) -> str:
    """`registered.reads`' key of row ``i`` of a unit's parameter ``name`` of ``rows`` rows: ``name[i]``, or ``name`` for
    a lone row (R or C = 1), which `registered.reads` reads as the whole parameter."""
    return name if rows == 1 else f"{name}[{i}]"


def window_ports(program: Program) -> dict[str, str]:
    """The circuit port of each A and B row a unit of the window ``program`` reads, by `unit_reads`' and `registered.reads`'
    keys (`row_key`), every K's in one statement: K's template's A row i (``a_k<K>[i]``) is port ``a_k<K>_<i>`` and its B
    row j (``b_k<K>[j]``) ``b_k<K>_<j>``; a lone row (``a_k<K>`` when R = 1, ``b_k<K>`` when C = 1) is row 0's. The
    separator keeps each K's ports its own: without it ``a_k32[20]`` and ``a_k322[0]`` would share ``a_k3220``, and
    `registered.reads_file` would merge two reads into one port."""
    out: dict[str, str] = {}
    for _k, t in program.fn.bindings["TILES"]:
        for name, rows in t.signature()[0][:2]:
            out |= {row_key(name, rows.n, i): f"{name}_{i}" for i in range(rows.n)}
    return out


#: each Program's `_inputs` and parameters by name, so a unit's reads cost its parameters, not the window's nodes
_SCANS: "weakref.WeakKeyDictionary[Program, tuple[dict[int, str], dict[str, Any]]]" = weakref.WeakKeyDictionary()


def _scan(program: Program) -> tuple[dict[int, str], dict[str, Any]]:
    got = _SCANS.get(program)
    if got is None:
        got = _SCANS[program] = (_inputs(program), dict(program.fn.signature()[0]))
    return got


def _inputs(program: Program) -> dict[int, str]:
    """Each root Input node's parameter: a parameter's leaves are one Input node's gates, all of them."""
    body, got = program.root.body, {}
    at = {body.offsets[k]: k for k, n in enumerate(body.nodes) if is_input(n.fn)}
    for pi, (name, t) in enumerate(program.fn.signature()[0]):
        rs = runs(program.input_coll(pi).refs)
        if len(rs) != 1 or rs[0][2] not in at or rs[0][3] - rs[0][2] != t.leaves or \
                body.offsets[at[rs[0][2]] + 1] - rs[0][2] != t.leaves:
            raise ValueError(f"parameter {name} is not one Input node's leaves")
        got[at[rs[0][2]]] = name
    return got


def unit_reads(program: Program, tp: Any, u: int) -> dict[str, tuple[str, int]]:
    """The rows unit ``u`` of the window's Program reads (``tp`` its template partition, `P.evaluate`'s): for each of its
    template's parameters, by row (``a_k<K>[i]``, ``b_k<K>[j]``, a lone row by its parameter's name, ``zy``, ``salt``,
    ``position``, ``roots``; `registered.reads`' keys, `row_key`), the window's parameter and the row of it, one whole row
    of its elements; and for each of its outputs (``receipt``, ``ok``, after the Program's prefix), row ``u`` of the
    window's. From the unit's call alone, so a unit costs its parameters at any K."""
    k, m = tp.locate(u)
    (inputs, params), node, out = _scan(program), program.root.body.nodes[k], {}
    args = node.args if node.form == "call" else [a if ax is None else a[m] for a, ax in zip(node.args, node.axes)]
    for pi, ((name, t), arg) in enumerate(zip(node.fn.signature()[0], args)):
        rows = [(row_key(name, t.n, i), arg[i]) for i in range(t.n)] if pi < 2 else [(name, arg)]
        for key, coll in rows:
            rs = runs(coll.refs)
            if len(rs) != 1 or rs[0][0] != "n" or rs[0][1] not in inputs:
                raise ValueError(f"unit {u}: {key} reads no one run of a parameter")
            _s, node_k, lo, hi = rs[0]
            param, w = inputs[node_k], hi - lo
            if lo % w or params[param].leaves % w:
                raise ValueError(f"unit {u}: {key} reads no whole row of {param}")
            out[key] = (param, lo // w)
    prefix = program.fn.bindings["PREFIX"]
    return {**out, **{field: (prefix + field, u) for field in OUTPUTS}}


#: the tables a call's rows are committed in under its round's root before the round's coin (#3053's ``tables``): ``a``,
#: its step's A table of its K, and ``zy``, its z table
TABLES = ("a", "zy")
#: the table a call's receipts are committed in under its round's root before the round's coin: one receipt a unit of the
#: call, in its tile order, each 2 (KS + m) u32 (`leaf_lanes.words`), which its statement's output row o0 and a drawn
#: leaf's combine read
RECEIPTS = "receipt"
#: the table of a call's statements' public rows, one a unit of the call in its tile order: the window's salt, the unit's
#: position and its call's roots (`public_table`), which the verifier registers itself at their known content
PUBS = "pub"
#: the statement's public row's port (`statement_rows`' ``pub``)
PUB_PORT = "pub"
#: the statement's output row of its unit's ok (`STATEMENT_OUT_ROWS`), which reads the window's ok rows, the verifier's own
OK_PORT = "o1"


def call_units(program: Program) -> list[tuple[int, int]]:
    """Each call's units of the window ``program``, in call order: its first unit and how many (its tiles)."""
    out, at = [], 0
    for n in _tiles(program.fn.S()):
        out.append((at, n))
        at += n
    return out


def call_of(program: Program, u: int) -> tuple[int, int]:
    """Unit ``u``'s call and its tile in it (row block major): its row of its call's z table."""
    units = call_units(program)
    c = bisect.bisect_right([u0 for u0, _n in units], u) - 1
    if u < 0 or c < 0 or u >= units[c][0] + units[c][1]:
        raise ValueError(f"unit {u} is no unit of the window's {sum(n for _u0, n in units)}")
    return c, u - units[c][0]


def step_of(program: Program, call: int, steps: int) -> tuple[int, int, int]:
    """Call ``call``'s step of the window's ``steps`` (its calls the steps' equal blocks, in order), its step's calls of its
    K, and its first row in its step's A table (`table`): ``(s, calls, M j)``, the call the j-th of its step's calls of its
    K in call order."""
    S = program.fn.S()
    ks, n = _k(S), len(S.CALLS)
    if not 0 <= call < n or steps < 1 or n % steps:
        raise ValueError(f"the window's {n} calls are not {steps} steps' equal blocks, or {call} is none of them")
    per = n // steps
    s, k = call // per, ks[S.CALLS[call][0]]
    mine = [c for c in range(s * per, (s + 1) * per) if ks[S.CALLS[c][0]] == k]
    return s, len(mine), S.M * mine.index(call)


def table(program: Program, name: str, call: int, steps: int | None = None) -> str:
    """The registered value of call ``call``'s table ``name`` (`TABLES`, `RECEIPTS` or `PUBS`): ``<prefix>a/<K>@<s>``, its
    step s's A table of its K (`step_of`, of the window's ``steps``: the M rows of the A block of each of the step's calls
    of K, each its K codes and s_a, call after call), ``<prefix>zy@<call>``, its z table (one record a unit of the call,
    in its tile order), ``<prefix>receipt@<call>``, its receipts, or ``<prefix>pub@<call>``, its public rows (each one a
    unit of the call, in its tile order)."""
    S = program.fn.S()
    if name not in TABLES + (RECEIPTS, PUBS) or not 0 <= call < len(S.CALLS) or (name == "a" and steps is None):
        raise ValueError(f"a call's tables are {TABLES + (RECEIPTS, PUBS)} of calls 0 .. {len(S.CALLS) - 1}, an A table "
                         f"of the window's steps, not {name!r} of call {call} (steps {steps})")
    if name in ("zy", RECEIPTS, PUBS):
        return f"{S.PREFIX}{name}@{call}"
    return f"{S.PREFIX}a/{_k(S)[S.CALLS[call][0]]}@{step_of(program, call, steps)[0]}"


def table_reads(program: Program, tp: Any, u: int, steps: int) -> dict[str, tuple[str, int]]:
    """`unit_reads` at the tables the unit's rows are committed in before its round's coin: unit ``u``, tile t of call c,
    reads A row ``a_k<K>[i]`` (row M blk + R ti + i of ``a/<K>``) at row M j + R ti + i of its step's A table of K, c the
    j-th of the step's calls of K (`step_of`), and its record ``zy`` at row t of call c's z table (`table`); at a chunked
    K each of these rows is its n chunk leaves, at n times the row plus the chunk (``row + n (M j - M blk)``); its B rows,
    the weights' registration, and the rest where `unit_reads` has them. The window-wide ``a/<K>`` and ``zy`` trees a
    served path registers at close come after their readers' coins, so a unit's A rows and record open only here."""
    S = program.fn.S()
    c, t = call_of(program, u)
    a_of, zy, base = f"{S.PREFIX}a/", f"{S.PREFIX}zy", S.CALLS[c][1] * S.M
    at, n = step_of(program, c, steps)[2], W.chunks(_k(S)[S.CALLS[c][0]])
    out = {}
    for key, (param, row) in unit_reads(program, tp, u).items():
        if param == zy:
            out[key] = (table(program, "zy", c), t)
        elif param.startswith(a_of) and "/" not in param[len(a_of):]:
            out[key] = (table(program, "a", c, steps), row + n * (at - base))
        else:
            out[key] = (param, row)
    return out


def _shapes(program: Program, call: int, steps: int) -> dict[str, tuple[int, int]]:
    """Call ``call``'s tables and the weights' it reads, each ``(rows, words)``, a row's 16-bit words: its step's A table of
    its K (M rows for each of the step's calls of K, times K's chunks), its z table and receipts (one row a unit of the call, the unit's
    receipt's bits as `statement_rows` counts a row's; 0 words when they are no whole 16-bit words, `credit.shape_of`'s
    rule) and ``b/<K>``."""
    S, sig = program.fn.S(), dict(program.fn.signature()[0])
    k, n = _k(S)[S.CALLS[call][0]], call_units(program)[call][1]
    b = f"{S.PREFIX}b/{k}"
    receipt = sum(dict(dict(S.TILES)[k].signature()[1].fields)[OUTPUTS[0]].leaf_widths())
    return {table(program, "a", call, steps): (W.chunks(k) * S.M * step_of(program, call, steps)[1],
                                               sig[f"{S.PREFIX}a/{k}"].elem.leaves // 16),
            table(program, "zy", call): (n, sig[f"{S.PREFIX}zy"].elem.leaves // 16),
            table(program, RECEIPTS, call): (n, receipt // 16 if receipt % 16 == 0 else 0),
            b: (sig[b].n, sig[b].elem.leaves // 16)}


def table_reads_file(program: Program, tp: Any, call: int, roots: dict[str, str], ports: dict[str, str] | None = None,
                     prog: str | None = None, window: str | None = None, *, steps: int, salt: str | None = None
                     ) -> dict[str, Any]:
    """The verifier's ``--registered`` file (`registered.READS_V1`) of a statement over call ``call``'s units:
    each A row at its step's A table and each record at the call's z table (`table_reads`, the window's ``steps``), each
    B row at the weights' ``b/<K>``. ``roots`` ``{value: root}`` holds those two tables' roots, as its round's record names
    them before that round's coin, and the weights'; ``ports`` names a key's circuit port when they differ
    (`window_ports`); ``prog`` the Program's digest when the caller holds it. Every table is a ``row/v2`` value of 16-bit
    leaves under `registered.domain` of its own name, rows and width: an A table is M rows of the parameter's width for
    each of its step's calls of K, a z table one record a unit of the call. ``window``, a served window's open root,
    binds the two tables' domains, the window's own rows (`credit.domain_of`), and never the weights', which are
    held across windows. When ``roots`` holds the call's receipt table, its units' receipts are the statement's output
    row o0 (`RECEIPT_PORT`) at its rows, one a unit, under the z table's domain rule. A unit of another call reads none
    of them: each port's reads start at the call's first unit (``reads_from``, `core_check.call_reads`) and end at its
    last (the verifier reads none below ``reads_from`` or past ``reads``, `Flock.Registered.Port.readAt`), so the file
    holds a statement to one call's units and is as long as the call. With ``salt``, the window's Program salt (hex),
    the statement's public row ``pub`` (`PUB_PORT`) is the call's public table (`PUBS`, `public_table`) at its rows, one
    a unit, at its known content (``known``): the salt, the unit's position and its call's roots, root(a/<K>@<s>) and
    root(b/<K>) as ``roots`` holds them, and a J10-R window's (`is_j10r`) the call's first position and the unit's
    blocks (`call_blocks`), so a unit seeded under another salt, position, roots, call or blocks opens no row of it; and
    the statement's output row ``ok`` (`OK_PORT`, o1) is the window's ok rows ``<prefix>ok`` (`ok_table`, row u unit
    u's) at its known content, the one word 1 under `core_check.ok_salt` of ``window`` and u, so a unit that computed 0
    opens no row of it."""
    S = program.fn.S()
    u0, n = call_units(program)[call]
    b, rc = f"{S.PREFIX}b/{_k(S)[S.CALLS[call][0]]}", table(program, RECEIPTS, call)
    shape = _shapes(program, call, steps)
    if rc not in roots:
        del shape[rc]
    elif not shape[rc][1]:
        raise ValueError(f"call {call}'s units return no receipt of whole 16-bit words for {rc} to hold")
    if missing := sorted(set(shape) - set(roots)):
        raise ValueError(f"call {call}'s statement reads {missing}, whose roots the verifier does not hold")
    reads: dict[str, dict[str, Any]] = {}
    for u in range(u0, u0 + n):
        for key, (value, row) in table_reads(program, tp, u, steps).items():
            if value in shape:
                reads.setdefault(key, {"value": value, "rows": [None] * n})["rows"][u - u0] = row
        if rc in shape:
            reads.setdefault(OUTPUTS[0], {"value": rc, "rows": [None] * n})["rows"][u - u0] = u - u0
    prog = prog or P.program_digest(program)
    out = {}
    for key, r in reads.items():
        leaves, words = shape[r["value"]]
        port = {OUTPUTS[0]: RECEIPT_PORT, **(ports or {})}.get(key, key)
        out[port] = {**({G.PARAM_KEY: key} if key in (ports or {}) and port != key else {}),
            "value": r["value"], "root": roots[r["value"]], "leaves": leaves, "words": words, "word_bits": G.WORD_BITS,
            "domain": G.domain(prog, r["value"], words, leaves, window=None if r["value"] == b else window).domain_id.hex(),
            **G.schema_entry(G.SCHEMA_V2), **CC.call_reads(u0, r["rows"])}
    if salt is not None:
        if window is None:
            raise ValueError("a call's public rows are committed under salts of the window's open root: give window")
        pv = table(program, PUBS, call)
        rows, salts, reg = public_table(prog, pv, salt, window, u0, n, (roots[table(program, "a", call, steps)], roots[b]),
                                        call_blocks(program, call) if is_j10r(program) else None)
        out[PUB_PORT] = {"value": pv, "root": reg.root.hex(), "leaves": n, "words": len(rows[0]), "word_bits": G.WORD_BITS,
                         "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                         **CC.call_reads(u0, list(range(n))), "known": {"rows": rows, "salts": [s.hex() for s in salts]}}
        ov = f"{S.PREFIX}{OUTPUTS[1]}"
        rows, salts, reg = ok_table(prog, ov, window, tp.population)
        out[OK_PORT] = {"value": ov, "root": reg.root.hex(), "leaves": tp.population, "words": len(CC.OK_ROW),
                        "word_bits": G.WORD_BITS, "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                        **CC.call_reads(u0, list(range(u0, u0 + n))),
                        "known": {"rows": rows, "salts": [s.hex() for s in salts]}}
    return {"format": G.READS_V1, "ports": out}


def call_blocks(program: Program, call: int) -> list[tuple[int, int]]:
    """Each of call ``call``'s units' ``blocks`` under J10-R, in its tile order: tile t = I (n/C) + J's first row block
    and first column block of its call's stream (`core_check.j10r_blocks`, `h1tile.j10r_blocks`' arithmetic) at the
    window's slot M, the call's K and the tile. Its ``call`` is the call's first unit (`call_units`)."""
    S = program.fn.S()
    (r, c), (lin, _blk) = _rc(S), S.CALLS[call]
    k, n = next((k, n) for name, k, n in S.LINEARS if name == lin)
    return [CC.j10r_blocks(S.M, k, (r, c), t // (n // c), t % (n // c)) for t in range(call_units(program)[call][1])]


def j10r_inputs(program: Program) -> tuple[list[int], list[tuple[int, int]]]:
    """A J10-R window's ``call`` and ``blocks`` (`J10R`), one row a unit: its call's first unit and `call_blocks`'."""
    calls, blocks = [], []
    for c, (u0, n) in enumerate(call_units(program)):
        calls += [u0] * n
        blocks += call_blocks(program, c)
    return calls, blocks


def public_table(prog: str, value: str, salt: str, window: str, u0: int, n: int, roots: tuple[str, str],
                 blocks: list[tuple[int, int]] | None = None) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """A call's public rows ``value`` (`PUBS`) over its ``n`` units from unit ``u0``, their salts and their registration:
    unit u's row the window's Program salt ``salt``, its position u and its call's ``roots`` (`core_check.public_words`,
    each hex), and with ``blocks`` (a J10-R window's `call_blocks`) the call's first position ``u0`` and the unit's
    blocks, under the public salt `core_check.public_row_salt` of the window's open root ``window`` and u, registered
    under the window Program's digest ``prog`` binding ``window``."""
    s, rs = bytes.fromhex(salt), (bytes.fromhex(roots[0]), bytes.fromhex(roots[1]))
    rows = [CC.public_words(s, u0 + t, rs, None if blocks is None else (u0, blocks[t])) for t in range(n)]
    salts = [CC.public_row_salt(window, u) for u in range(u0, u0 + n)]
    return rows, salts, G.Registered.commit(prog, value, rows, salts, G.WORD_BITS, G.SCHEMA_V2, window=window)


@functools.lru_cache(maxsize=8)
def ok_table(prog: str, value: str, window: str, population: int) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """The window's ok rows ``value`` (``<prefix>ok``, `OUTPUTS`) over its ``population`` units, their salts and their
    registration (h1_prove's `register_ok`, #3124): row u the one word 1 (`core_check.OK_ROW`) under
    `core_check.ok_salt` of the window's open root ``window`` and u, registered under the window Program's digest
    ``prog`` binding ``window``."""
    rows = [list(CC.OK_ROW) for _u in range(population)]
    salts = [CC.ok_salt(window, u) for u in range(population)]
    return rows, salts, G.Registered.commit(prog, value, rows, salts, G.WORD_BITS, G.SCHEMA_V2, window=window)


# -- a drawn leaf's lanes and combine -----------------------------------------------------------------------------------------

#: a ``--registered`` port's key that its statement reads every bit of its row (Flock's `KnownRead.READ_KEY`)
READ_IN_FULL = "read_in_full"
#: the values a drawn leaf's statements read in full: an A row, a B row and a partials row each read by several of them,
#: and a J10-R leaf's ok rows (and its carries, `leaf_lanes.read_in_full`)
FULL = LL.FULL


def _leaf_ports(S: Any) -> Any:
    return forward(S.UNIT)


PouwH1WindowLane = CompositeDefinition(
    "PouwH1WindowLane", 1, ("UNIT", "K"), lambda S: S.UNIT.signature(), lambda B, S, *args: B.call(S.UNIT, *args),
    register=False, ports=_leaf_ports,
    doc="A drawn leaf's lane statement at contraction K: one call of UNIT, the lane (h1zy's PouwH1LaneZy) over the tile of "
        "the window's template at K, with its parameters and ports.")
PouwH1WindowCombine = CompositeDefinition(
    "PouwH1WindowCombine", 1, ("UNIT", "K"), lambda S: S.UNIT.signature(), lambda B, S, *args: B.call(S.UNIT, *args),
    register=False, ports=_leaf_ports,
    doc="A drawn leaf's combine statement at contraction K: one call of UNIT, the combine (h1zy's PouwH1Combine) over the "
        "tile of the window's template at K, with its parameters.")


def _tile_of(program: Program, k: int) -> Any:
    tiles = dict(program.fn.bindings["TILES"])
    if int(k) not in tiles:
        raise ValueError(f"the window has no template at K = {k}: its K are {sorted(tiles)}")
    unit = tiles[int(k)].bindings["UNIT"]
    if getattr(unit, "defn", None) in Z.CHUNKED:
        unit = unit.bindings["UNIT"]
    if getattr(unit, "defn", None) is not Z.PouwH1TileZy:
        raise ValueError(f"the window's unit at K = {k} is {getattr(unit, 'id', unit)}, not h1zy's PouwH1TileZy over an H1 "
                         "tile, so its leaf has no lanes")
    return unit.bindings["TILE"]


def leaf_unit(program: Program, k: int, kind: str, ch: int | None = None) -> Any:
    """The Definition a drawn leaf's statement ``kind`` (`leaf_lanes.LANE`, ``CHUNK`` with its chunk ``ch``, or
    ``COMBINE``) proves at K, derived from the window's own template at K (`template`: its unit `h1zy.PouwH1TileZy` over
    the H1 tile): `h1zy.lane`, `h1zy.lane_chunk` at `weights.chunks`(K) or `h1zy.combine` over that tile. Refused
    (ValueError): a K the window has no template at, a unit other than the zy unit, a statement the leaf hasn't (a whole
    lane at a chunked K, whose lanes are the chunk lanes over chunk rows, or a chunk at a whole one), a chunked K but over
    J10-R's `PouwH1Tile_v5` (`h1zy.J10R_LANE_TILES`, the one tile whose lanes are staged in chunks), and a leaf
    `h1zy.lane` or `h1zy.combine` doesn't stage (an R not a multiple of 16, a combine below `h1zy.LEAST_K`)."""
    if kind not in (LL.LANE, LL.CHUNK, LL.COMBINE):
        raise ValueError(f"a drawn leaf's statement is a {LL.LANE}, a {LL.CHUNK} or its {LL.COMBINE}, not {kind!r}")
    tile = _tile_of(program, k)
    n = W.chunks(int(k))
    if tile.defn.id not in Z.J10R_LANE_TILES:
        _whole_rows(k)
    if kind == LL.COMBINE:
        return Z.combine(tile)
    if n == 1 and kind == LL.LANE and ch is None:
        return Z.lane(tile)
    if n > 1 and kind == LL.CHUNK and ch is not None and 0 <= int(ch) < n:
        return Z.lane_chunk(tile, n, int(ch))
    raise ValueError(f"K = {k}'s lanes are {'whole' if n == 1 else f'{n} chunk lanes each (weights.chunks)'}: a drawn "
                     f"leaf's statement there is {'a lane' if n == 1 else f'a chunk of 0 .. {n - 1}'}, not {(kind, ch)!r}"
                     + ("" if n == 1 else ", the chunk lanes"))


def _whole_rows(k: int) -> None:
    if W.chunks(int(k)) > 1:
        raise ValueError(f"K = {k}'s rows are {W.chunks(int(k))} chunk leaves each (weights.chunks): a drawn leaf's lanes "
                         "there are the chunk lanes, which read chunk rows, not h1zy.lane's whole rows; they are staged "
                         "over J10-R's PouwH1Tile_v5 only")


def leaf_template(program: Program, k: int, kind: str, ch: int | None = None) -> Any:
    """K's wrapper of `leaf_unit`: `PouwH1WindowLane` (a lane or a lane chunk) or `PouwH1WindowCombine` bound to it and
    K."""
    return bind(PouwH1WindowCombine if kind == LL.COMBINE else PouwH1WindowLane, UNIT=leaf_unit(program, k, kind, ch),
                K=int(k))


#: each window Program's leaf Programs by (K, kind, chunk), so a period derives each once
_LEAVES: "weakref.WeakKeyDictionary[Program, dict[tuple, tuple[Program, dict[str, Any]]]]" = weakref.WeakKeyDictionary()


def leaf_program(program: Program, k: int, kind: str, ch: int | None = None) -> tuple[Program, dict[str, Any]]:
    """The Program and partition query every drawn leaf's statement ``kind`` (a chunk's ``ch``) at K runs under: the
    one-call Program of `leaf_template` and the instance query of its unit, a population of one unit, so the statement's
    draw is its whole population. Both are functions of the window's descriptor alone, so a period has one of each a K,
    kind and chunk."""
    memo = _LEAVES.setdefault(program, {})
    key = (int(k), kind, None if ch is None else int(ch))
    if key not in memo:
        t = leaf_template(program, k, kind, ch)
        memo[key] = (Program(t), P.template_instances_query([t.bindings["UNIT"]]))
    return memo[key]


def place(program: Program, u: int, steps: int) -> dict[str, tuple[str, int]]:
    """Where unit ``u``'s leaf is in each value its statements read (proofs' ``place`` in `PouwWindow.Lanes.ownOf`): the
    table and the leaf's first row in it, so a statement's read of row i of the leaf's value (`leaf_lanes.reads`) is that
    row plus i. For u tile t = (ti, tj) of call c (its linear's tiles row block major), c the j-th of its step's calls of
    its K: ``a`` its step's A table of K (`table`) from row M j + R ti, ``b`` the weights' ``b/<K>`` from row b0 + C tj (b0
    the linear's first, `_b0`), ``zy`` and ``receipt`` the call's z table and receipts at row t, and ``partials`` the drawn
    leaf's own (`leaf_lanes.PARTIALS`, `one_stage.produced`) from row 0. The rows are the tile's own (`table_reads`). At a
    chunked K (`weights.chunks` n > 1) ``a`` and ``b`` are leaf indices, n times the first row (the chunk lanes' base).
    A J10-R window's leaf's other after-draw values, its carries and ok rows (`leaf_lanes.produced_values`), are its own
    from row 0 too."""
    S = program.fn.S()
    c, t = call_of(program, u)
    lin = S.CALLS[c][0]
    r, cc = _rc(S)
    ti, tj = divmod(t, {name: n for name, _k, n in S.LINEARS}[lin] // cc)
    n = W.chunks(_k(S)[lin])
    produced = LL.produced_values(r, cc, _k(S)[lin], j10r=True) if is_j10r(program) else {LL.PARTIALS: None}
    return {LL.A: (table(program, "a", c, steps), n * (step_of(program, c, steps)[2] + r * ti)),
            LL.B: (f"{S.PREFIX}b/{_k(S)[lin]}", n * (_b0(S)[lin] + cc * tj)),
            LL.ZY: (table(program, "zy", c), t), LL.RECEIPT: (table(program, RECEIPTS, c), t),
            **{v: (v, 0) for v in produced}}


def _leaf_k(program: Program, u: int) -> int | None:
    """Unit ``u``'s K as `leaf_lanes`' ``j10r`` takes it: the K of its call under J10-R, None for v1's leaf (refused at a
    chunked K, whose lanes are staged over J10-R only)."""
    S = program.fn.S()
    k = _k(S)[S.CALLS[call_of(program, u)[0]][0]]
    if is_j10r(program):
        return k
    _whole_rows(k)
    return None


def leaf_reads(program: Program, u: int, kind: str, ln: int | None = None, ch: int | None = None, *,
               steps: int) -> dict[str, tuple[str, int]]:
    """Each registered port of unit ``u``'s statement ``kind`` (lane ``ln``, a J10-R lane's chunk ``ch``, or the
    combine) with the table and the row the verifier reads it at: `leaf_lanes.reads`' row of its value past the value's
    `place`. Refused at a chunked K but under J10-R, whose chunk lanes read chunk rows."""
    j10r = _leaf_k(program, u)
    at = place(program, u, steps)
    return {port: (at[v][0], at[v][1] + row)
            for port, (v, row) in LL.reads(*_rc(program.fn.S()), kind, ln, ch, j10r=j10r).items()}


def leaf_known_rows(program: Program, kind: str, ln: int | None = None, ch: int | None = None, *, salt: bytes,
                    position: int, roots: Sequence[bytes], admit: Sequence[int] | None = None,
                    u: int | None = None) -> dict[str, tuple[str, list[int]]]:
    """`leaf_lanes.known_rows` of a drawn leaf of the window: lane ``ln``'s ``pub`` of its unit's ``salt``, ``position`` and
    ``roots`` (the tile statement's, `PUBLIC`), its ``admit`` row every one of the leaf's R rows unless ``admit`` is given
    (the tile unit masks no row), and every statement's ``o0``. A J10-R window's (unit ``u``'s, which it needs): its pub
    row with the unit's ``call`` and ``blocks`` (`j10r_inputs`), and only the combine's ``o0``."""
    r, c = _rc(program.fn.S())
    j10r: dict[str, Any] = {}
    if is_j10r(program):
        if u is None:
            raise ValueError("a J10-R leaf's known rows are its unit u's: its pub row holds u's call and blocks")
        calls, blocks = j10r_inputs(program)
        j10r = {"j10r": _leaf_k(program, u), "call": calls[u], "blocks": blocks[u]}
    return LL.known_rows(r, c, kind, ln, ch, admit=[1] * r if admit is None else admit, salt=salt, position=position,
                         roots=roots, **j10r)


def leaf_reads_file(program: Program, u: int, kind: str, ln: int | None, roots: dict[str, str],
                    partials: dict[str, Any], known_ports: dict[str, Any], prog: str | None = None,
                    window: str | None = None, *, steps: int, ch: int | None = None,
                    produced: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The verifier's ``--registered`` file of unit ``u``'s statement ``kind`` (lane ``ln``, a J10-R lane's chunk ``ch``,
    or the combine), in `leaf_lanes.registered_file`'s form: each of its `leaf_reads` at its table's row under ``roots``
    ``{value: root}`` (the call's tables' roots as the round log names them, the weights' and the produced record's
    ``partials``), each table's entry the verifier's own (`_shapes`: a ``row/v2`` value of 16-bit leaves under
    `registered.domain` of its name, rows and width, binding ``window``, the served window's open root, except the
    weights' ``b/<K>``), ``partials``' entry ``partials`` and a J10-R leaf's carries' and ok rows' theirs in
    ``produced`` (each `produced.entry` under the window's registration). A read of an A row, a B row, a partials row,
    a carry or an ok row is marked `READ_IN_FULL` (`leaf_lanes.read_in_full`), never an ``o0``. Then ``known_ports``
    (`produced.known_port` of `leaf_known_rows`), and ``complete``."""
    S = program.fn.S()
    c = call_of(program, u)[0]
    shape, b = _shapes(program, c, steps), f"{S.PREFIX}b/{_k(S)[S.CALLS[c][0]]}"
    prog = prog or P.program_digest(program)
    j10r = _leaf_k(program, u)
    logical = LL.reads(*_rc(S), kind, ln, ch, j10r=j10r)
    after = {LL.PARTIALS: partials, **(produced or {})}
    mine = LL.produced_values(*_rc(S), _k(S)[S.CALLS[c][0]], j10r=j10r is not None)
    out = {}
    for port, (value, row) in leaf_reads(program, u, kind, ln, ch, steps=steps).items():
        if value not in roots:
            raise ValueError(f"port {port!r} reads {value!r}, whose root the verifier does not hold")
        if value in mine:
            if value not in after:
                raise ValueError(f"port {port!r} reads {value!r}, whose entry the verifier does not hold")
            e = after[value]
        else:
            leaves, words = shape[value]
            e = {"leaves": leaves, "words": words, "word_bits": G.WORD_BITS,
                 "domain": G.domain(prog, value, words, leaves, window=None if value == b else window).domain_id.hex()}
        out[port] = PR.port(value, e, roots[value], [row], G.SCHEMA_V2)
        if LL.read_in_full(logical[port][0]) and port != LL.OK:
            out[port][READ_IN_FULL] = True
    if both := sorted(set(out) & set(known_ports)):
        raise ValueError(f"ports {both} are both reads and known")
    return {"format": G.READS, "ports": {**out, **known_ports}, LL.COMPLETE: True}
