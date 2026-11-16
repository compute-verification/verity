"""The circuit format's running example, the Program the frontend analyses' tests compare: two rows of a five-row linear layer
over 16-bit words, with its activation a static (`make_program(act=...)`)."""

from __future__ import annotations

from verity.circuits import Array, Program, Value, bind, composite, primitive

V16 = Value(16)
MASK = 0xFFFF


@primitive("Mul16", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def Mul16(a: int, b: int) -> int:
    return (a * b) & MASK


@primitive("Add16", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def Add16(a: int, b: int) -> int:
    return (a + b) & MASK


@primitive("Relu16", 1, [("a", V16)], V16, conformance="exact-model-tested")
def Relu16(a: int) -> int:
    return 0 if a & 0x8000 else a


@primitive("Id16", 1, [("a", V16)], V16, conformance="exact-model-tested")
def Id16(a: int) -> int:
    return a


A2 = Array(2, V16)


@composite("Dot", 1, [], lambda S: ((("x", A2), ("w", A2)), V16))
def Dot(B, S, x, w):
    p0 = B.call(Mul16, x[0], w[0])
    p1 = B.call(Mul16, x[1], w[1])
    return B.call(Add16, p0, p1)


@composite("Row", 1, ["act"], lambda S: ((("x", A2), ("w", A2)), V16))
def Row(B, S, x, w):
    return B.call(S.act, B.call(Dot, x, w))


@composite("Linear", 1, ["M", "act"], lambda S: ((("x", A2), ("W", Array(S.M, A2))), Array(S.M, V16)))
def Linear(B, S, x, W):
    return B.batch(bind(Row, act=S.act), x, W, axes=(None, 0))


@composite("Inference", 1, ["B", "M", "act"],
           lambda S: ((("X", Array(S.B, A2)), ("W", Array(S.M, A2))), Array(S.B, Array(S.M, V16))))
def Inference(B, S, X, W):
    linear = bind(Linear, M=S.M, act=S.act)
    return B.batch(linear, X, W, axes=(0, None))


def make_program(act=Relu16, annotations=None) -> Program:
    return Program(bind(Inference, B=2, M=5, act=act), annotations=annotations)
