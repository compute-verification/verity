"""H1's window as one public Program: every R × C tile of a window's calls is one call of its call's contraction K's
template (`template`: one call of the tile unit, `h1tile.PouwH1Tile`, at that K), so the verifier derives the window's
units from its own copy of the Program under the `Q_template_instances` query of its templates (`query`), and a served
path registers the trees it commits as the Program's parameters (`one_stage.served.register_served`).

* **The tile** is a parameter: ``tile`` is its R × C, the served path's 32 × 32 (`pouw_h1.TILE`) or h1_prove's 4 × 8
  smoke, and the window binds the tile unit once per K its linears have (`tile_unit`; ``unit`` gives another), so
  down_proj's calls, at K the intermediate size, instantiate their own template in the same Program.
* **Templates.** K's template takes the tile's parameters with its A and B rows named for their K, ``a_k<K>`` and
  ``b_k<K>`` (`template_params`): `registered.reads` keys a unit's rows by its template's parameter names, so a mixed-K
  window registers in one read, A row i of K's units under ``a_k<K>[i]``, each port with its own width and domain, and
  that row's circuit port is ``a_k<K>_<i>`` (`window_ports`; B row j's ``b_k<K>_<j>``), every K's ports distinct.
* **Linears and calls.** ``linears`` are the window's weights ``(name, K, n)``, each once, and ``calls`` its calls
  ``(linear, block)`` in order: a step's call of a linear reads A block ``block`` of its K and the linear's B rows.
* **Fixed slots.** The layout is protocol structure, so no token count becomes public. For each K the window's A rows are
  ``a/<K>``: blocks of ``M`` rows each (`slot`, a power of two of at least R, the token rows padded with zero rows), one
  block per input a step's calls of that K read. ``b/<K>`` is the B rows of the linears of that K, linear after linear,
  each linear's n rows once (a whole number of C): every call of a linear, in any step, reads the same rows, so a window
  of S steps registers Σn weight rows, not S × Σn. ``salt`` is the units' one coin, ``position`` one row per unit (its
  number), and ``roots`` the window's seed-group roots.
* **Units** are the tiles, call by call and row block major. Tile ``(c, ti, tj)`` is the unit on rows
  ``block M + [R ti, R ti + R)`` of ``a/<K>`` and on rows ``[C tj, C tj + C)`` of its linear's in ``b/<K>``, at position its
  number. The Program's result is each call's tiles' outputs in turn (their widths depend on K), and unit u's outputs are
  row u of the window's ``z`` and ``receipt``. `unit_reads` gives, from the Program's own references, the row of each
  parameter that each of a unit's template parameters reads.
* **Names.** Every parameter is named ``prefix`` and its role (`roles`), so a served path can name each after the
  registered value its tree commits.
"""
from __future__ import annotations

import weakref
from typing import Any, Callable, Iterable, Sequence

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import coll_concat, runs
from verity.ml.circuits.types import Array, Tuple

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.sampled_proofs.one_stage import partition as P

#: the tile unit's parameters, in order: its R A rows, its C B rows, the salt, its position, the roots
TILE_PARAMS = ("a", "b", "salt", "position", "roots")
#: the window's parameters after each K's ``a/<K>`` and ``b/<K>``
SHARED = ("salt", "position", "roots")
#: the unit's outputs (`h1tile`'s record), each a row per unit of the window's outputs
OUTPUTS = ("z", "receipt")


def slot(rows: int, tile: tuple[int, int]) -> int:
    """The A slot for ``rows`` token rows under an R × C ``tile``: the least power of two holding them, at least R."""
    if rows < 1:
        raise ValueError("a window has at least one token row")
    return max(int(tile[0]), 1 << (rows - 1).bit_length())


def roles(ks: Iterable[int]) -> tuple[str, ...]:
    """The window's parameters for contractions ``ks``, in order and before their ``prefix``: ``a/<K>`` and ``b/<K>`` for
    each K, ascending, then `SHARED`."""
    return tuple(r for k in sorted(set(ks)) for r in (f"a/{k}", f"b/{k}")) + SHARED


def tile_unit(k: int, tile: tuple[int, int], definition: Any = None) -> Any:
    """The tile unit at contraction ``k``: `h1tile.PouwH1Tile` (or ``definition``, its v3) over
    ``GemmAccumulatorS8_v2{K, sm120}``, R × C ``tile``, every (i, j) pair."""
    r, c = tile
    exact = bind(BG.GemmAccumulatorS8, K=int(k), **BG.GEMM_SM120_S8)
    return bind(definition or H.PouwH1Tile, EXACT=exact, R=int(r), C=int(c),
                PAIRS=tuple((i, j) for i in range(r) for j in range(c)))


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
    salt, position, roots = (t for _name, t in S.TILES[0][1].signature()[0][2:])
    params += [("salt", salt), ("position", Array(sum(_tiles(S)), position)), ("roots", roots)]
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
                outs.append(B.call(tiles[k], a[a0:a0 + r], b[j0:j0 + c], got["salt"], got["position"][len(outs)],
                                   got["roots"]))
    return coll_concat(_signature(S)[1], outs)


PouwH1Window = CompositeDefinition(
    "PouwH1Window", 2, ("TILES", "LINEARS", "CALLS", "M", "PREFIX"), _signature, _body, register=False,
    ports=lambda S: Ports(params={S.PREFIX + "salt": In(kind="coin")}),
    doc="One H1 window's tiles, call by call and row block major: TILES[K] on rows block M + [R ti, R ti + R) of a/K (each "
        "K's A blocks of M rows) and on rows [C tj, C tj + C) of the call's linear in b/K (each K's LINEARS' B rows once, "
        "linear after linear), the salt, the tile's position and the window's roots; every parameter named PREFIX and its "
        "role.")


def program(linears: Sequence[tuple[str, int, int]], calls: Sequence[tuple[str, int]], m: int, tile: tuple[int, int],
            prefix: str = "", unit: Callable[[int], Any] | None = None) -> Program:
    """The window's Program: its ``linears`` ``(name, K, n)``, each weight once, its ``calls`` ``(linear, A block)`` in
    order, its A slot ``m`` and its R × C ``tile``, over the tile unit at each K (`tile_unit`, or ``unit(K)``), its
    parameters named ``prefix`` and their roles. Refused (ValueError): no linear or no call, a linear named twice or that no
    call reads, a call of no linear, a linear whose n is not whole blocks of C, a slot that is no power of two of at least
    R, an A block of a K that no call reads, or a unit whose rows aren't R × C or whose shared parameters differ by K."""
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
        sig = t.signature()[0]
        if tuple(name for name, _t in sig) != TILE_PARAMS or (sig[0][1].n, sig[1][1].n) != (r, c):
            raise ValueError(f"the tile unit at K = {k} is no {r} × {c} tile over {TILE_PARAMS}: {sig}")
        if shared is not None and tuple(t for _n, t in sig[2:]) != shared:
            raise ValueError(f"the tile unit at K = {k} takes other {SHARED} than at K = {units[0][0]}")
        shared = tuple(t for _n, t in sig[2:])
    tiles = tuple((k, template(k, t)) for k, t in units)
    return Program(bind(PouwH1Window, TILES=tiles, LINEARS=linears, CALLS=calls, M=int(m), PREFIX=str(prefix)))


def query(program: Program) -> dict[str, Any]:
    """The window's partition query: every instance of each of its templates (`template`, one per K, ascending) one unit."""
    return P.template_instances_query([t for _k, t in program.fn.bindings["TILES"]])


def window_ports(program: Program) -> dict[str, str]:
    """The circuit port of each A and B row a unit of the window ``program`` reads, by `unit_reads`' and `registered.reads`'
    keys, every K's in one statement: K's template's A row i (``a_k<K>[i]``) is port ``a_k<K>_<i>`` and its B row j
    (``b_k<K>[j]``) ``b_k<K>_<j>``; a lone row, which `registered.reads` keys by the parameter's name (``a_k<K>`` when R = 1,
    ``b_k<K>`` when C = 1), is row 0's. The separator keeps each K's ports its own: without it ``a_k32[20]`` and
    ``a_k322[0]`` would share ``a_k3220``, and `registered.reads_file` would merge two reads into one port."""
    out: dict[str, str] = {}
    for _k, t in program.fn.bindings["TILES"]:
        for name, rows in t.signature()[0][:2]:
            out |= {name: f"{name}_0"} if rows.n == 1 else {f"{name}[{i}]": f"{name}_{i}" for i in range(rows.n)}
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
    template's parameters, by row (``a_k<K>[i]``, ``b_k<K>[j]``, ``salt``, ``position``, ``roots``; `registered.reads`'
    keys), the window's parameter and the row of it, one whole row of its elements; and for each of its outputs (``z``,
    ``receipt``, after the Program's prefix), row ``u`` of the window's. From the unit's call alone, so a unit costs its
    parameters at any K."""
    k, _m = tp.locate(u)
    (inputs, params), node, out = _scan(program), program.root.body.nodes[k], {}
    for pi, ((name, t), arg) in enumerate(zip(node.fn.signature()[0], node.args)):
        rows = [(f"{name}[{i}]", arg[i]) for i in range(t.n)] if pi < 2 else [(name, arg)]
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
