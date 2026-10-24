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
* **Units** are the tiles, call by call and row block major. Tile ``(c, ti, tj)`` is the unit on rows
  ``block M + [R ti, R ti + R)`` of ``a/<K>``, on rows ``[C tj, C tj + C)`` of its linear's in ``b/<K>`` and on row u of
  ``zy``, at position u, its number. The Program's result is each call's tiles' outputs in turn (their widths depend on
  K), and unit u's outputs are row u of the window's ``receipt`` and ``ok``. `unit_reads` gives, from the Program's own
  references, the row of each parameter that each of a unit's template parameters reads.
* **Ports.** ``salt`` is the units' coin (`In(kind="coin")`), ``position`` their positions (`In(kind="position")`), and each
  unit's receipt a sink of the result (`Out(sink=True)`, `_ports`): the window is no seed, so the port rules follow the coin
  into every tile, and a unit that returns no sink is refused.
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
"""
from __future__ import annotations

import bisect
import functools
import weakref
from typing import Any, Callable, Iterable, Sequence

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ports import In, Leaves, Out, Ports, port_types
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import coll_concat, runs
from verity.ml.circuits.types import Array, Tuple

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import h1device as D
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

#: the tile unit's parameters, in order: its R A rows, its C B rows, its z/y record, the salt, its position, the roots
TILE_PARAMS = ("a", "b", "zy", "salt", "position", "roots")
#: the window's parameters after each K's ``a/<K>`` and ``b/<K>``
SHARED = ("zy", "salt", "position", "roots")
#: the unit's outputs (`h1zy`'s), each a row per unit of the window's outputs
OUTPUTS = ("receipt", "ok")


def slot(rows: int, tile: tuple[int, int]) -> int:
    """The A slot for ``rows`` token rows under an R × C ``tile``: the least power of two holding them, at least R."""
    if rows < 1:
        raise ValueError("a window has at least one token row")
    return max(int(tile[0]), 1 << (rows - 1).bit_length())


def roles(ks: Iterable[int]) -> tuple[str, ...]:
    """The window's parameters for contractions ``ks``, in order and before their ``prefix``: ``a/<K>`` and ``b/<K>`` for
    each K, ascending, then `SHARED`."""
    return tuple(r for k in sorted(set(ks)) for r in (f"a/{k}", f"b/{k}")) + SHARED


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


def template_params(k: int) -> tuple[str, ...]:
    """K's template's parameters, in order: the tile's (`TILE_PARAMS`), its A and B rows ``a_k<K>`` and ``b_k<K>``."""
    return (f"a_k{int(k)}", f"b_k{int(k)}") + SHARED


def _template_signature(S: Any) -> tuple:
    params, ret = S.UNIT.signature()
    return tuple(zip(template_params(S.K), (t for _name, t in params))), ret


def _template_ports(S: Any) -> Any:
    decl = S.UNIT.defn.ports
    return decl(S.UNIT.S()) if callable(decl) else decl


PouwH1WindowTile = CompositeDefinition(
    "PouwH1WindowTile", 1, ("UNIT", "K"), _template_signature, lambda B, S, *args: B.call(S.UNIT, *args), register=False,
    ports=_template_ports,
    doc="A window's template at contraction K: one call of the tile unit UNIT, its parameters and ports, with its A and B "
        "rows named a_kK and b_kK.")


def template(k: int, unit: Any) -> Any:
    """K's template over the tile unit ``unit`` (`tile_unit`'s at K, or another of its parameters)."""
    return bind(PouwH1WindowTile, UNIT=unit, K=int(k))


#: K's statement's public row: the unit's salt, position and roots, in that order
PUBLIC = ("salt", "position", "roots")
#: K's statement's output rows ``(schema, outputs)``: the receipt (o0) and ok (o1), one ``row/v2`` row each
STATEMENT_OUT_ROWS = ((G.SCHEMA_V2, (0,)), (G.SCHEMA_V2, (1,)))
#: the statement's output row of its units' receipts, which reads its call's receipt table when the window registers one
RECEIPT_PORT = "o0"


def statement_rows(template: Any, ports: dict[str, str] | None = None) -> list[tuple[str, str, int, list[str]]]:
    """The input rows of K's statement (``template`` `template`'s at K), each ``(port, schema, bits, params)`` as C-Flock's
    unit stager takes them (``a_k<K>[0]`` row 0 of ``a_k<K>``, `row_key`), in order: each A row and B row its own row, then the record
    ``zy`` and the public row ``pub`` (`PUBLIC`), every one a ``row/v2`` row of its leaves' bits (an A or B row 8 K + 32,
    `h1zy.ROW`; the record 48 R C, `h1zy.RECORD`; ``pub`` 1600). ``ports`` names each A and B row's port by its `row_key`;
    by default `window_ports`' (``a_k<K>_<i>``, ``b_k<K>_<j>``), the ports credit's registered file reads. Its outputs are
    `STATEMENT_OUT_ROWS`."""
    params, _ret = template.signature()
    ty = dict(params)
    if [n for n, _t in params[2:]] != list(SHARED) or not all(n.startswith(("a_k", "b_k")[x]) for x, (n, _t) in enumerate(params[:2])):
        raise ValueError(f"{template.id}: K's template's parameters are a_k<K>, b_k<K>, then {', '.join(SHARED)}")

    def bits(p: str) -> int:
        name, _, row = p.partition("[")
        return sum((ty[name].elem if row else ty[name]).leaf_widths())

    keys = [(row_key(n, t.n, i), f"{n}_{i}") for n, t in params[:2] for i in range(t.n)]
    rows = [((ports or {}).get(key, port), [key]) for key, port in keys] + [("zy", ["zy"]), ("pub", list(PUBLIC))]
    return [(port, G.SCHEMA_V2, sum(bits(p) for p in ps), ps) for port, ps in rows]


def _rc(S: Any) -> tuple[int, int]:
    (_, at), (_, bt) = S.TILES[0][1].signature()[0][:2]
    return at.n, bt.n


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
        params += [(f"a/{k}", Array(blocks * S.M, at.elem)),
                   (f"b/{k}", Array(sum(n for _name, kk, n in S.LINEARS if kk == k), bt.elem))]
    zy, salt, position, roots = (t for _name, t in S.TILES[0][1].signature()[0][2:])
    units = sum(_tiles(S))
    params += [("zy", Array(units, zy)), ("salt", salt), ("position", Array(units, position)), ("roots", roots)]
    ret = Tuple(*(Array(n, tiles[ks[lin]].signature()[1]) for (lin, _blk), n in zip(S.CALLS, _tiles(S))))
    return tuple((S.PREFIX + role, t) for role, t in params), ret


def _body(B: Any, S: Any, *params: Any) -> Any:
    got = dict(zip(roles(k for k, _t in S.TILES), params))
    ks, tiles, b0, n_of = _k(S), dict(S.TILES), _b0(S), {name: n for name, _k, n in S.LINEARS}
    (r, c), outs = _rc(S), []
    for lin, blk in S.CALLS:
        k = ks[lin]
        a, b = got[f"a/{k}"], got[f"b/{k}"]
        for ti in range(S.M // r):
            a0 = blk * S.M + r * ti
            for tj in range(n_of[lin] // c):
                j0 = b0[lin] + c * tj
                u = len(outs)
                outs.append(B.call(tiles[k], a[a0:a0 + r], b[j0:j0 + c], got["zy"][u], got["salt"], got["position"][u],
                                   got["roots"]))
    return coll_concat(_signature(S)[1], outs)


def _ports(S: Any) -> Ports:
    """The salt is the units' coin and ``position`` their positions, and every leaf a unit's template returns as a sink (its
    receipt) is a sink of the window's result. A Definition with a coin and a position port and no sink in its return is a
    seed, whose body no port rule follows (`ports._coin_role`), so without them the tiles' coin flow would go unchecked."""
    ks, at, sinks = _k(S), 0, []
    of = {k: (t.signature()[1].leaves, _sinks(t)) for k, t in S.TILES}
    for (lin, _blk), n in zip(S.CALLS, _tiles(S)):
        width, own = of[ks[lin]]
        for _ in range(n):
            sinks += [(at + lo, at + hi, Out(sink=True)) for lo, hi in own]
            at += width
    return Ports(params={S.PREFIX + "salt": In(kind="coin"), S.PREFIX + "position": In(kind="position")},
                 ret=Leaves(tuple(sinks)))


def _sinks(unit: Any) -> list[tuple[int, int]]:
    """The leaf ranges of ``unit``'s return it declares sinks (`Out(sink=True)`, its receipt)."""
    pt = port_types(unit)
    return [(lo, hi) for lo, hi, o, _w in (pt.ret if pt is not None else ()) if o.sink]


PouwH1Window = CompositeDefinition(
    "PouwH1Window", 3, ("TILES", "LINEARS", "CALLS", "M", "PREFIX"), _signature, _body, register=False, ports=_ports,
    doc="One H1 window's tiles, call by call and row block major: TILES[K] on rows block M + [R ti, R ti + R) of a/K (each "
        "K's A blocks of M rows) and on rows [C tj, C tj + C) of the call's linear in b/K (each K's LINEARS' B rows once, "
        "linear after linear), the tile's z/y record (row u of zy for tile u), the salt, the tile's position and the "
        "window's roots; every parameter named PREFIX and its role.")


def program(linears: Sequence[tuple[str, int, int]], calls: Sequence[tuple[str, int]], m: int, tile: tuple[int, int],
            prefix: str = "", unit: Callable[[int], Any] | None = None) -> Program:
    """The window's Program: its ``linears`` ``(name, K, n)``, each weight once, its ``calls`` ``(linear, A block)`` in
    order, its A slot ``m`` and its R × C ``tile``, over the tile unit at each K (`tile_unit`, or ``unit(K)``), its
    parameters named ``prefix`` and their roles. Refused (ValueError): no linear or no call, a linear named twice or that no
    call reads, a call of no linear, a linear whose n is not whole blocks of C, a slot that is no power of two of at least
    R, an A block of a K that no call reads, or a unit whose rows aren't R × C rows of K codes and their scale
    (`h1zy.ROW`), whose record isn't z and y (`h1zy.RECORD`: a port carrying z alone leaves y unproved), that returns
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
    units = tuple((k, make(k)) for k in sorted(set(ks.values())))
    shared = None
    for k, t in units:
        sig, ret = t.signature()
        if tuple(name for name, _t in sig) != TILE_PARAMS or (sig[0][1].n, sig[1][1].n) != (r, c):
            raise ValueError(f"the tile unit at K = {k} is no {r} × {c} tile over {TILE_PARAMS}: {sig}")
        if (sig[0][1].elem, sig[1][1].elem) != (Z.ROW(k), Z.ROW(k)):
            raise ValueError(f"the tile unit at K = {k} reads A and B rows of {sig[0][1].elem.leaves} and "
                             f"{sig[1][1].elem.leaves} bits, not {k} codes and the row's f32 scale (h1zy.ROW, {8 * k + 32} bits)")
        if sig[2][1] != Z.RECORD(r, c):
            raise ValueError(f"the tile unit at K = {k} reads a {sig[2][1].leaves}-bit record, not its {r} × {c} cells' z "
                             f"then y (h1zy.RECORD, {48 * r * c} bits): a record of z alone leaves the served y unproved")
        if tuple(name for name, _t in getattr(ret, "fields", ())) != OUTPUTS:
            raise ValueError(f"the tile unit at K = {k} returns {ret}, not {OUTPUTS}")
        if shared is not None and tuple(t for _n, t in sig[2:]) != shared:
            raise ValueError(f"the tile unit at K = {k} takes other {SHARED} than at K = {units[0][0]}")
        if not _sinks(t):
            raise ValueError(f"the tile unit at K = {k} returns no sink (Out(sink=True), its receipt): the window, with the "
                             "salt and the positions as its coin and position ports, would be a seed no port rule follows")
        shared = tuple(t for _n, t in sig[2:])
    tiles = tuple((k, template(k, t)) for k, t in units)
    return Program(bind(PouwH1Window, TILES=tiles, LINEARS=linears, CALLS=calls, M=int(m), PREFIX=str(prefix)))


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
    k, _m = tp.locate(u)
    (inputs, params), node, out = _scan(program), program.root.body.nodes[k], {}
    for pi, ((name, t), arg) in enumerate(zip(node.fn.signature()[0], node.args)):
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
#: call, in its tile order, each 2 (KS + m) u32 (`leaf_lanes.words`), which its statement's output row o0 reads
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
    j-th of the step's calls of K (`step_of`), and its record ``zy`` at row t of call c's z table (`table`); its B rows,
    the weights' registration, and the rest where `unit_reads` has them. The window-wide ``a/<K>`` and ``zy`` trees a
    served path registers at close come after their readers' coins, so a unit's A rows and record open only here."""
    S = program.fn.S()
    c, t = call_of(program, u)
    a_of, zy, base = f"{S.PREFIX}a/", f"{S.PREFIX}zy", S.CALLS[c][1] * S.M
    at = step_of(program, c, steps)[2]
    out = {}
    for key, (param, row) in unit_reads(program, tp, u).items():
        if param == zy:
            out[key] = (table(program, "zy", c), t)
        elif param.startswith(a_of) and "/" not in param[len(a_of):]:
            out[key] = (table(program, "a", c, steps), at + row - base)
        else:
            out[key] = (param, row)
    return out


def _shapes(program: Program, call: int, steps: int) -> dict[str, tuple[int, int]]:
    """Call ``call``'s tables and the weights' it reads, each ``(rows, words)``, a row's 16-bit words: its step's A table of
    its K (M rows for each of the step's calls of K), its z table and receipts (one row a unit of the call, the unit's
    receipt's bits as `statement_rows` counts a row's; 0 words when they are no whole 16-bit words, `credit.shape_of`'s
    rule) and ``b/<K>``."""
    S, sig = program.fn.S(), dict(program.fn.signature()[0])
    k, n = _k(S)[S.CALLS[call][0]], call_units(program)[call][1]
    b = f"{S.PREFIX}b/{k}"
    receipt = sum(dict(dict(S.TILES)[k].signature()[1].fields)[OUTPUTS[0]].leaf_widths())
    return {table(program, "a", call, steps): (S.M * step_of(program, call, steps)[1], sig[f"{S.PREFIX}a/{k}"].elem.leaves // 16),
            table(program, "zy", call): (n, sig[f"{S.PREFIX}zy"].elem.leaves // 16),
            table(program, RECEIPTS, call): (n, receipt // 16 if receipt % 16 == 0 else 0),
            b: (sig[b].n, sig[b].elem.leaves // 16)}


def table_reads_file(program: Program, tp: Any, call: int, roots: dict[str, str], ports: dict[str, str] | None = None,
                     prog: str | None = None, window: str | None = None, *, steps: int, salt: str | None = None
                     ) -> dict[str, Any]:
    """The verifier's ``--registered`` file (`registered.reads_file`'s format) of a statement over call ``call``'s units:
    each A row at its step's A table and each record at the call's z table (`table_reads`, the window's ``steps``), each
    B row at the weights' ``b/<K>``. ``roots`` ``{value: root}`` holds those two tables' roots, as its round's record names
    them before that round's coin, and the weights'; ``ports`` names a key's circuit port when they differ
    (`window_ports`); ``prog`` the Program's digest when the caller holds it. Every table is a ``row/v2`` value of 16-bit
    leaves under `registered.domain` of its own name, rows and width: an A table is M rows of the parameter's width for
    each of its step's calls of K, a z table one record a unit of the call. ``window``, a served window's open root,
    binds the two tables' domains, the window's own rows (PRF-127; `credit.domain_of`), and never the weights', which are
    held across windows. When ``roots`` holds the call's receipt table, its units' receipts are the statement's output
    row o0 (`RECEIPT_PORT`) at its rows, one a unit, under the z table's domain rule. A unit of another call reads none
    of them (None, and no entry past a port's last read: the verifier reads none at a unit past ``reads``,
    `Flock.Registered.badInstance`), so the file holds a statement to one call's units. With ``salt``, the window's Program salt (hex),
    the statement's public row ``pub`` (`PUB_PORT`) is the call's public table (`PUBS`, `public_table`) at its rows, one
    a unit, at its known content (``known``): the salt, the unit's position and its call's roots, root(a/<K>@<s>) and
    root(b/<K>) as ``roots`` holds them, so a unit seeded under another salt, position or roots opens no row of it; and
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
                reads.setdefault(key, {"value": value, "rows": [None] * (u0 + n)})["rows"][u] = row
        if rc in shape:
            reads.setdefault(OUTPUTS[0], {"value": rc, "rows": [None] * (u0 + n)})["rows"][u] = u - u0
    for r in reads.values():
        while r["rows"][-1] is None:
            r["rows"].pop()
    prog = prog or P.program_digest(program)
    out = {}
    for key, r in reads.items():
        leaves, words = shape[r["value"]]
        out[{OUTPUTS[0]: RECEIPT_PORT, **(ports or {})}.get(key, key)] = {
            "value": r["value"], "root": roots[r["value"]], "leaves": leaves, "words": words, "word_bits": G.WORD_BITS,
            "domain": G.domain(prog, r["value"], words, leaves, window=None if r["value"] == b else window).domain_id.hex(),
            **G.schema_entry(G.SCHEMA_V2),
            "reads": r["rows"]}
    if salt is not None:
        if window is None:
            raise ValueError("a call's public rows are committed under salts of the window's open root: give window")
        pv = table(program, PUBS, call)
        rows, salts, reg = public_table(prog, pv, salt, window, u0, n, (roots[table(program, "a", call, steps)], roots[b]))
        out[PUB_PORT] = {"value": pv, "root": reg.root.hex(), "leaves": n, "words": len(rows[0]), "word_bits": G.WORD_BITS,
                         "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                         "reads": [None] * u0 + list(range(n)), "known": {"rows": rows, "salts": [s.hex() for s in salts]}}
        ov = f"{S.PREFIX}{OUTPUTS[1]}"
        rows, salts, reg = ok_table(prog, ov, window, tp.population)
        out[OK_PORT] = {"value": ov, "root": reg.root.hex(), "leaves": tp.population, "words": len(CC.OK_ROW),
                        "word_bits": G.WORD_BITS, "domain": reg.domain.domain_id.hex(), **G.schema_entry(G.SCHEMA_V2),
                        "reads": [None] * u0 + list(range(u0, u0 + n)),
                        "known": {"rows": rows, "salts": [s.hex() for s in salts]}}
    return {"format": G.READS, "ports": out}


def public_table(prog: str, value: str, salt: str, window: str, u0: int, n: int, roots: tuple[str, str]
                 ) -> tuple[list[list[int]], list[bytes], G.Registered]:
    """A call's public rows ``value`` (`PUBS`) over its ``n`` units from unit ``u0``, their salts and their registration:
    unit u's row the window's Program salt ``salt``, its position u and its call's ``roots`` (`core_check.public_words`,
    each hex), under the public salt `core_check.public_row_salt` of the window's open root ``window`` and u, registered
    under the window Program's digest ``prog`` binding ``window``."""
    s, rs = bytes.fromhex(salt), (bytes.fromhex(roots[0]), bytes.fromhex(roots[1]))
    rows = [CC.public_words(s, u, rs) for u in range(u0, u0 + n)]
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
