"""Small Boolean circuits whose properties are known by hand, covering each form the summaries map through: calls, a batch
sharing its activation (axis None), batches over axis 0 and axis 1, a scan carrying its accumulator, a Definition passing
its parameter through, constant activation leaves, and a counted op outside every `Lin` (not counted).

`MulLanes_v1{N}` is the counted op: lane k computes from a[k] and b[k] (two bits each) seven gates of its own (eight in lane
0), beside a constant and one gate reading two lanes, and folds the lanes into an accumulator. `Coord_v1{K,N}` is one
output of a linear map, K/N MulLanes steps from a zero accumulator.
"""

from __future__ import annotations

from circuit_properties.ops import Designation, Site

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, broadcast, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple

BIT = BL.BIT
LANE = Array(2, BIT)
OUT = Array(2, BIT)

DESIGNATION = Designation(sites=(Site(op=r"MulLanes_v\d+", x="a", w="b", within=r"Lin\w*_v\d+"),),
                          entries=r"Coord_v\d+|LinScan_v\d+")


def _and(B, a, b):
    return B.call(BL.AND, BL.operands(a, b))


def _xor(B, a, b):
    return B.call(BL.XOR, BL.operands(a, b))


def _not(B, a):
    return B.call(BL.NOT, a)


def _def(name: str, statics: tuple, sig, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, statics, sig, body, register=False)


def _zeros(B, n: int) -> Coll:
    s, i, j = B.call(BL.constant(0)).refs[0]
    return Coll(Array(n, BIT), broadcast(s, i, j, n))


def _mul_lanes(B, S, acc, a, b):
    one = B.call(BL.constant(1))
    s_out, r_out = [], []
    for k in range(S.N):
        q = _xor(B, _and(B, a[k][0], b[k][0]), _and(B, a[k][1], b[k][1]))
        r = _and(B, _not(B, a[k][0]), _not(B, b[k][1]))
        s = _xor(B, q, one)
        s_out.append(_not(B, s) if k == 0 else s)
        r_out.append(r)
    o0, o1 = acc[0], acc[1]
    for k in range(S.N):
        o0, o1 = _xor(B, o0, s_out[k]), _and(B, o1, r_out[k])
    if S.N > 1:
        o1 = _xor(B, o1, _and(B, a[0][0], a[1][0]))
    return Coll(OUT, concat([o0.refs, o1.refs]))


MulLanes = _def("MulLanes", ("N",), lambda S: ((("acc", OUT), ("a", Array(S.N, LANE)), ("b", Array(S.N, LANE))), OUT),
                _mul_lanes)


def _coord(B, S, x, w):
    acc, step = _zeros(B, 2), bind(MulLanes, N=S.N)
    for s in range(S.K // S.N):
        acc = B.call(step, acc, x[s * S.N:(s + 1) * S.N], w[s * S.N:(s + 1) * S.N])
    return Coll(OUT, concat([_not(B, acc[0]).refs, _xor(B, acc[0], acc[1]).refs]))


Coord = _def("Coord", ("K", "N"), lambda S: ((("x", Array(S.K, LANE)), ("w", Array(S.K, LANE))), OUT), _coord)

Lin = _def("Lin", ("K", "M", "N"),
           lambda S: ((("x", Array(S.K, LANE)), ("w", Array(S.M, Array(S.K, LANE)))), Array(S.M, OUT)),
           lambda B, S, x, w: B.batch(bind(Coord, K=S.K, N=S.N), x, w, axes=(None, 0)))
LinCols = _def("LinCols", ("K", "M", "N"),
               lambda S: ((("x", Array(S.K, LANE)), ("w", Array(S.K, Array(S.M, LANE)))), Array(S.M, OUT)),
               lambda B, S, x, w: B.batch(bind(Coord, K=S.K, N=S.N), x, w, axes=(None, 1)))
LinTokens = _def("LinTokens", ("T", "K", "N"),
                 lambda S: ((("x", Array(S.T, Array(S.K, LANE))), ("w", Array(S.K, LANE))), Array(S.T, OUT)),
                 lambda B, S, x, w: B.batch(bind(Coord, K=S.K, N=S.N), x, w, axes=(0, None)))


def _scan_step(B, S, carry, x, w):
    c = B.call(bind(MulLanes, N=S.N), carry, x, w)
    return tuple_of(c, _xor(B, c[0], c[1]))


ScanStep = _def("ScanStep", ("N",),
                lambda S: ((("carry", OUT), ("x", Array(S.N, LANE)), ("w", Array(S.N, LANE))), Tuple(OUT, BIT)), _scan_step)


def _lin_scan(B, S, xs, w):
    out = B.scan(bind(ScanStep, N=S.N), _zeros(B, 2), xs=[xs], shared=[w])
    return Coll(Array(2 + S.T, BIT), out.refs)


LinScan = _def("LinScan", ("T", "N"),
               lambda S: ((("xs", Array(S.T, Array(S.N, LANE))), ("w", Array(S.N, LANE))), Array(2 + S.T, BIT)), _lin_scan)
Identity = _def("Identity", ("K",), lambda S: ((("x", Array(S.K, LANE)),), Array(S.K, LANE)), lambda B, S, x: x)


def _net(B, S, x, xs, xr, w1, w2, w3, w4, w5):
    h = B.call(bind(Lin, K=4, M=4, N=2), x, w1)
    h2 = B.call(bind(Identity, K=4), Coll(Array(4, LANE), h.refs))
    g = B.call(bind(LinCols, K=4, M=3, N=2), h2, w2)
    t = B.call(bind(LinTokens, T=2, K=4, N=2), xs, w3)
    sc = B.call(bind(LinScan, T=3, N=2), xr, w4)
    ex = B.call(bind(MulLanes, N=2), _zeros(B, 2), x[0:2], h2[0:2])
    pad = Coll(Array(4, LANE), concat([x[0:2].refs, _zeros(B, 4).refs]))
    c = B.call(bind(Lin, K=4, M=2, N=2), pad, w5)
    return Coll(Array(29, BIT), concat([o.refs for o in (h, g, t, sc, ex, c)]))


NET_PARAMS = (("x", Array(4, LANE)), ("xs", Array(2, Array(4, LANE))), ("xr", Array(3, Array(2, LANE))),
              ("w1", Array(4, Array(4, LANE))), ("w2", Array(4, Array(3, LANE))), ("w3", Array(4, LANE)),
              ("w4", Array(2, LANE)), ("w5", Array(2, Array(4, LANE))))
Net = _def("Net", (), lambda S: (NET_PARAMS, Array(29, BIT)), _net)
WEIGHTS = ("w1", "w2", "w3", "w4", "w5")

#: |𝒥| by hand: Lin 4 x 4, LinCols 3 x 4, LinTokens 2 x 4, LinScan 3 x 2, the padded Lin 2 x 4 (the MulLanes in no Lin
#: isn't counted); x[0] and x[1] feed Lin's 4 coordinates and the padded Lin's 2; the scan's first output y_0 sums 2
NET = {"products": 50, "s_x": 7, "s_x_max": 8, "a_out": 6, "a_in": 2, "constant_leaves": {"Const1[0x0]_v1": 8},
       "excluded": {"MulLanes_v1{N=2}": 1}}


def net() -> Program:
    return Program(bind(Net), registered=WEIGHTS, theta=WEIGHTS)


def _dup(B, S, x, w):
    return B.call(bind(Coord, K=2, N=2), Coll(Array(2, LANE), concat([x[0].refs, x[0].refs])), w)


Dup = _def("Dup", (), lambda S: ((("x", Array(1, LANE)), ("w", Array(2, LANE))), OUT), _dup)


def dup() -> Program:
    """One activation value bound to both lanes of one Coord: 2 products each feeding it."""
    return Program(bind(Dup), registered=("w",))


DUP_DESIGNATION = Designation(sites=(Site(op=r"MulLanes_v\d+", x="a", w="b"),), entries=r"Coord_v\d+")
