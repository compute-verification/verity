"""Port types (PROTOCOL.md §12, `verity.ml.circuits.ports`): each rule accepts what it should and refuses, by name,
what it must, in a built Program and in a decoded descriptor; an undeclared Definition's entry and digest are what they were."""

from __future__ import annotations

import copy
import hashlib
import re

import pytest

from verity.ml.circuits import ports as P
from verity.ml.circuits.codec import decode_program, definition_entry, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind, primitive
from verity.ml.circuits.evaluation.reference import evaluate, standalone_call
from verity.ml.circuits.ports import In, Leaves, Out, PortError, Ports
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, PartLog, Strided, array_of, concat, tuple_of
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Value

V8, V16, V32 = Value(8), Value(16), Value(32)


def _s(e: int, w: int) -> int:
    return e - (1 << w) if e >> (w - 1) else e


@primitive("PortsTestQuant", 1, (("x", V16),), V16, ports=Ports(ret=Out(interval=(-128, 127))))
def Quant(x):
    """Clamp to [-128, 127]: a quantizer with its clamp range declared."""
    return max(-128, min(127, _s(x, 16))) & 0xFFFF


@primitive("PortsTestWide", 1, (("x", V16),), V16, ports=Ports(ret=Out(interval=(-200, 200))))
def Wide(x):
    return max(-200, min(200, _s(x, 16))) & 0xFFFF


@primitive("PortsTestAdd16", 1, (("a", V16), ("b", V16)), V16)
def Add16(a, b):
    return (a + b) & 0xFFFF


@primitive("PortsTestMul", 1, (("a", V16), ("b", V16)), V32,
           ports=Ports(params={"a": In(bound=(-128, 127)), "b": In(bound=(-128, 127))}))
def Mul(a, b):
    """An s8 product: it assumes both operands in [-128, 127]."""
    return (_s(a, 16) * _s(b, 16)) & 0xFFFFFFFF


@primitive("PortsTestAnyBound", 1, (("a", V16),), V16, ports=Ports(params={"a": In(bound=(-32768, 32767))}))
def AnyBound(a):
    return a


@primitive("PortsTestAdd32", 1, (("a", V32), ("b", V32)), V32)
def Add32(a, b):
    return (a + b) & 0xFFFFFFFF


@primitive("PortsTestFold", 1, (("x", V32),), V32, ports=Ports(ret=Out(sink=True)))
def Fold(x):
    """A receipt fold: its output is a sink."""
    return x ^ 0x5A5A5A5A


@primitive("PortsTestAbsorb", 1, (("acc", V32), ("r", V32)), V32,
           ports=Ports(params={"r": In(receipt=True)}, ret=Out(sink=True)))
def Absorb(acc, r):
    """The receipt port: it takes sinks and returns one."""
    return (acc * 31 + r) & 0xFFFFFFFF


@primitive("PortsTestTag", 1, (), V8)
def Tag():
    """A constant (no operands)."""
    return 7


@primitive("PortsTestNoise", 1, (("key", V32), ("tag", V8)), V8, ports=Ports(params={"key": In(kind="coin"), "tag": In(kind="static")}))
def Noise(key, tag):
    """A noise derivation: a coin and a static only."""
    return hashlib.sha512(key.to_bytes(4, "big") + bytes([tag])).digest()[0]


@primitive("PortsTestDither", 1, (("x", V8), ("key", V32)), V8, ports=Ports(params={"key": In(kind="coin")}))
def Dither(x, key):
    """A noise injection: adds a coin's bit to the operand it takes."""
    return (x + (key & 1)) & 0xFF


@primitive("PortsTestKeyFold", 1, (("x", V32), ("key", V32)), V32,
           ports=Ports(params={"key": In(kind="coin")}, ret=Out(sink=True)))
def KeyFold(x, key):
    """A receipt fold keyed by a coin."""
    return ((x ^ key) * 0x9E3779B1) & 0xFFFFFFFF


@primitive("PortsTestAdd8", 1, (("a", V8), ("b", V8)), V8)
def Add8(a, b):
    return (a + b) & 0xFF


@primitive("PortsTestAcc", 1, (("a", V32),), V32, ports=Ports(ret=Out(word="s32")))
def Acc(a):
    """A committed accumulator word: its producer fixes it as s32."""
    return a


@primitive("PortsTestLimbs", 1, (("w", V32),), V8, ports=Ports(params={"w": In(word="s8x4")}))
def Limbs(w):
    """Reads a word as four s8 limbs (a reinterpretation)."""
    return w & 0xFF


@primitive("PortsTestLimbSplit", 1, (("w", V32),), V8, ports=Ports(params={"w": In(word="s32")}))
def LimbSplit(w):
    """The explicit limb split of an s32 word: it reads the word as its producer's type."""
    return w & 0xFF


def comp(name: str, params, ret, body, ports=None) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False, ports=ports)


def refused(code: str, build, *needles: str) -> str:
    with pytest.raises(PortError) as e:
        build()
    assert e.value.code == code, e.value
    for n in needles:
        assert n in str(e.value), (n, str(e.value))
    return str(e.value)


def roundtrip(prog: Program):
    desc = encode_program(prog)
    dec = decode_program(desc, REGISTRY)
    assert program_digest(encode_program(dec)) == program_digest(desc)
    return desc, dec


# --- neutrality ---------------------------------------------------------------------------------------------------------------

def test_an_undeclared_definition_is_what_it_was():
    """No annotation: no `ports` key, the same entry and digest as before this rule, and no analysis at all."""
    assert definition_entry(Add16) == {"form": "primitive", "id": "PortsTestAdd16_v1",
                                       "params": [["a", ["Value", 16]], ["b", ["Value", 16]]], "ret": ["Value", 16]}
    f = comp("PortsTestPlain", [("x", V16), ("y", V16)], V16, lambda B, S, x, y: B.call(Add16, B.call(Add16, x, y), y))
    prog = Program(f)
    desc = encode_program(prog)
    assert not any("ports" in d for d in desc["definitions"].values())
    assert P.check(prog.root) is None and P.facts(prog.root).typed is False
    assert program_digest(decode_program(desc, REGISTRY)) == program_digest(prog)


def test_a_declared_definition_carries_its_ports_and_a_pinned_digest():
    assert definition_entry(Mul)["ports"] == {"params": [[0, 0, 1, {"bound": [-128, 127]}], [1, 0, 1, {"bound": [-128, 127]}]]}
    assert definition_entry(Quant)["ports"] == {"ret": [[0, 1, {"interval": [-128, 127]}]]}
    f = comp("PortsTestPinned", [("x", V16)], V32, lambda B, S, x: B.call(Mul, B.call(Quant, x), B.call(Quant, x)))
    assert program_digest(Program(f)) == PINNED


#: the typed descriptor's digest: a change to how ports are written changes it
PINNED = "7995bc7e0cff4bf4dc43d7dae970cdedaf7c59ee9a3837ea1598a2acbb1179312ebac7484ae175822cbc06c0a4409fa1cede0aea43c27ca3a18a28fa78cd4f0f"


# --- rule 2: intervals --------------------------------------------------------------------------------------------------------

def test_a_bound_contains_its_producers_interval():
    f = comp("PortsTestSq", [("x", V16)], V32, lambda B, S, x: B.call(Mul, B.call(Quant, x), B.call(Quant, x)))
    prog = Program(f)
    fx = P.facts(prog.root)
    k = next(k for k, n in enumerate(prog.root.body.nodes) if n.fn is Quant)
    assert fx.node(k).one.interval == (-128, 127) and fx.node(k).one.origin == {"gate"}
    _, dec = roundtrip(prog)
    assert P.facts(dec.root).node(k).one.interval == (-128, 127)


def test_a_bound_refuses_a_producer_without_an_interval():
    f = comp("PortsTestUnb", [("x", V16)], V32, lambda B, S, x: B.call(Mul, B.call(Add16, x, x), x))
    refused("E_PORT_BOUND", lambda: Program(f), "parameter 'a'", "[-128, 127] does not contain its producer's interval: it "
            "declares none, so the value spans Value<16>")


def test_a_bound_refuses_a_wider_producer():
    f = comp("PortsTestWider", [("x", V16)], V32, lambda B, S, x: B.call(Mul, B.call(Quant, x), B.call(Wide, x)))
    refused("E_PORT_BOUND", lambda: Program(f), "parameter 'b'", "its producer's interval [-200, 200]")


def test_a_full_bound_assumes_nothing():
    Program(comp("PortsTestFull", [("x", V16)], V16, lambda B, S, x: B.call(AnyBound, B.call(Add16, x, x))))


def test_an_input_bound_narrower_than_its_type_is_refused():
    """A declared bound is never taken on trust: a program function's parameter declaring one narrower than its type is refused,
    built and decoded, whatever its binding time or kind. A full bound assumes nothing; a range check narrows the input instead;
    a harness takes the bounds of the Definition it evaluates, and is never decoded."""
    trusted = comp("PortsTestInB", [("x", V16)], V32, lambda B, S, x: B.call(Mul, x, x), ports=Ports(params={"x": In(bound=(-128, 127))}))
    for kw in ({}, {"registered": ("x",)}, {"theta": ("x",)}):
        refused("E_PORT_BOUND", lambda: Program(trusted, **kw), "program function PortsTestInB_v1: parameter 'x' leaves [0, 1) "
                "declare the bound [-128, 127], narrower than Value<16>, but a run input's bound is never taken on trust",
                "range-check Definition")
    coin = comp("PortsTestInCoinB", [("key", V32)], V8, lambda B, S, key: B.call(Noise, key, B.call(Tag)),
                ports=Ports(params={"key": In(bound=(0, 255), kind="coin")}))
    refused("E_PORT_BOUND", lambda: Program(coin), "parameter 'key' leaves [0, 1) declare the bound [0, 255], narrower than Value<32>")
    full = comp("PortsTestInFull", [("x", V16)], V16, lambda B, S, x: B.call(AnyBound, x), ports=Ports(params={"x": In(bound=(-32768, 32767))}))
    desc, dec = roundtrip(Program(full))
    assert desc["definitions"][desc["root"]]["ports"] == {"inputs": [[0, 0, 1, {"bound": [-32768, 32767]}]]}
    assert P.facts(dec.root).node(0).one == P.fact((-32768, 32767), ("input",))
    tampered = copy.deepcopy(desc)
    tampered["definitions"][desc["root"]]["ports"]["inputs"][0][3]["bound"] = [-128, 127]
    refused("E_PORT_BOUND", lambda: decode_program(tampered, REGISTRY), "input node 0: leaves [0, 1) declare the bound "
            "[-128, 127], narrower than Value<16>, but an input's bound is never taken on trust")
    checked = Program(comp("PortsTestInRange", [("x", V16)], V32, lambda B, S, x: B.call(Mul, B.call(Quant, x), B.call(Quant, x))))
    _desc, dec = roundtrip(checked)
    F = P.facts(dec.root)
    assert F.node(0).one == P.fact(None, ("input",)) and F.node(1).one.interval == (-128, 127)
    prog, _call = standalone_call(bind(trusted))
    assert P.facts(prog.root).node(0).one == P.fact((-128, 127), ("input",))
    assert evaluate(bind(trusted), (-5) & 0xFFFF) == (25,)
    refused("E_PORT_BOUND", lambda: decode_program(encode_program(prog), REGISTRY), "only a harness's")


def test_intervals_flow_down_through_calls_joined_over_instances():
    inner = comp("PortsTestInner", [("x", V16)], V32, lambda B, S, x: B.call(Mul, x, x))
    ok = comp("PortsTestDownOk", [("x", V16)], Array(2, V32),
              lambda B, S, x: array_of([B.call(inner, B.call(Quant, x)), B.call(inner, B.call(Quant, x))]))
    prog = Program(ok)
    assert P.facts(prog.root).params(bind(inner))[0].one.interval == (-128, 127)
    bad = comp("PortsTestDownBad", [("x", V16)], Array(2, V32),
               lambda B, S, x: array_of([B.call(inner, B.call(Quant, x)), B.call(inner, B.call(Add16, x, x))]))
    refused("E_PORT_BOUND", lambda: Program(bad), "PortsTestInner_v1 node 0 (primitive PortsTestMul_v1) parameter 'a'")


def test_an_interval_flows_up_only_as_declared():
    up = comp("PortsTestUp", [("x", V16)], V16, lambda B, S, x: B.call(Quant, x), ports=Ports(ret=Out(interval=(-128, 127))))
    Program(comp("PortsTestUpOk", [("x", V16)], V32, lambda B, S, x: B.call(Mul, B.call(up, x), B.call(up, x))))
    silent = comp("PortsTestSilent", [("x", V16)], V16, lambda B, S, x: B.call(Quant, x))
    refused("E_PORT_BOUND", lambda: Program(comp("PortsTestUpBad", [("x", V16)], V32,
                                                 lambda B, S, x: B.call(Mul, B.call(silent, x), x))), "it declares none")
    narrow = comp("PortsTestNarrow", [("x", V16)], V16, lambda B, S, x: B.call(Quant, x), ports=Ports(ret=Out(interval=(-64, 63))))
    refused("E_PORT_INTERVAL", lambda: Program(comp("PortsTestNarrowUse", [("x", V16)], V16, lambda B, S, x: B.call(narrow, x))),
            "PortsTestNarrow_v1 declares Out(interval=(-64, 63)) at return leaf 0, but its body returns its producer's interval [-128, 127]")


def test_batch_and_scan_carry_intervals():
    bq = comp("PortsTestBatch", [("x", Array(3, V16))], Array(3, V32),
              lambda B, S, x: B.batch(Mul, B.batch(Quant, x, axes=(0,)), B.batch(Quant, x, axes=(0,)), axes=(0, 0)))
    Program(bq)
    carry_t = V16
    step = comp("PortsTestStep", [("c", carry_t), ("x", V16)], TupleT(carry_t, V32),
                lambda B, S, c, x: tuple_of(B.call(Quant, B.call(Add16, c, x)), B.call(Mul, c, B.call(Quant, x))),
                ports=Ports(ret=(Out(interval=(-128, 127)), None)))
    init_ok = comp("PortsTestScanOk", [("x", Array(4, V16))], TupleT(carry_t, Array(4, V32)),
                   lambda B, S, x: B.scan(step, B.call(Quant, x[0]), xs=(x,)))
    Program(init_ok)
    init_bad = comp("PortsTestScanBad", [("x", Array(4, V16))], TupleT(carry_t, Array(4, V32)),
                    lambda B, S, x: B.scan(step, x[0], xs=(x,)))
    refused("E_PORT_BOUND", lambda: Program(init_bad), "PortsTestStep_v1 node 3 (primitive PortsTestMul_v1) parameter 'a'")


def _dot4_interval(acc, a, b):
    """acc plus four products of s8 operands, each read signed: the step's return while it stays in s32, else None (it wraps)."""
    acc, a, b = P.as_reading(acc, 32), P.as_reading(a, 16), P.as_reading(b, 16)
    if acc is None or a is None or b is None:
        return None
    p = [x * y for x in a for y in b]
    lo, hi = acc[0] + 4 * min(p), acc[1] + 4 * max(p)
    return (lo, hi) if -(1 << 31) <= lo and hi < 1 << 31 else None


S8 = In(bound=(-128, 127))


@primitive("PortsTestDot4", 1, (("acc", V32), ("a", Array(4, V16)), ("b", Array(4, V16))), V32,
           ports=Ports(params={"a": S8, "b": S8}, transfer=_dot4_interval))
def Dot4(acc, *ab):
    """An exact s8 step: acc plus four s8 products, modulo 2^32."""
    return (_s(acc, 32) + sum(_s(x, 16) * _s(y, 16) for x, y in zip(ab[:4], ab[4:]))) & 0xFFFFFFFF


@primitive("PortsTestZero32", 1, (), V32)
def Zero32():
    return 0


@primitive("PortsTestSplit17", 1, (("y", V32),), V8, ports=Ports(params={"y": In(bound=(-65536, 65536))}))
def Split17(y):
    """A limb split that is exact only for y within its bound (the s8 family's `Y`)."""
    return y & 0xFF


#: the chain's words T_t from zero over s8 operands: t steps of four products in [-16256, 16384]
T_S8 = [(-65024 * t, 65536 * t) for t in (1, 2)]


def chain(name: str, ivs=None, acc_param: bool = False) -> CompositeDefinition:
    """The s8 family's shape: two exact steps from T_0 (zero, or the incoming `acc`), returning T_1 and T_2."""
    def body(B, S, *args):
        acc, a, b = args if acc_param else (B.call(Zero32), *args)
        T = []
        for s in range(2):
            acc = B.call(Dot4, acc, a[4 * s:4 * s + 4], b[4 * s:4 * s + 4])
            T.append(acc)
        return array_of(T)

    params = [("acc", V32)] * acc_param + [("a", Array(8, V16)), ("b", Array(8, V16))]
    ret = None if ivs is None else Leaves(tuple((t, t + 1, Out(interval=iv)) for t, iv in enumerate(ivs)))
    return comp(name, params, Array(2, V32), body, ports=Ports(params={"a": S8, "b": S8}, ret=ret))


@primitive("PortsTestRange7", 1, (("x", V16),), V16, ports=Ports(ret=Out(interval=(-7, 7))))
def Range7(x):
    """A range check: its operand clamped to [-7, 7], the interval it declares."""
    return max(-7, min(7, _s(x, 16))) & 0xFFFF


@primitive("PortsTestRangeNeg3", 1, (("x", V16),), V16, ports=Ports(ret=Out(interval=(-3, 0))))
def RangeNeg3(x):
    """A range check: its operand clamped to [-3, 0]."""
    return max(-3, min(0, _s(x, 16))) & 0xFFFF


def test_a_transfer_gives_each_instance_its_operands_interval():
    """An exact step's return carries the interval its transfer gives from that instance's operands: the chain's words from
    s8 operands, and narrower ones from inputs range checks narrow; a constant's interval is its value."""
    f = comp("PortsTestChainUse", [("x", Array(8, V16))], Array(2, V32),
             lambda B, S, x: B.call(chain("PortsTestChainT"), B.batch(Quant, x, axes=(0,)), B.batch(Quant, x, axes=(0,))))
    prog = Program(f)
    fx = P.facts(prog.root)
    (sp,) = [sp for sp, _ in fx.definitions() if sp.id.startswith("PortsTestChainT")]
    assert fx.params(sp)[0].one.interval == (-128, 127)
    desc, dec = roundtrip(prog)
    assert desc["definitions"]["PortsTestDot4_v1"]["ports"] == {
        "params": [[1, 0, 4, {"bound": [-128, 127]}], [2, 0, 4, {"bound": [-128, 127]}]], "transfer": True}
    narrow = comp("PortsTestChainNarrow", [("a", Array(8, V16)), ("b", Array(8, V16))], V32,
                  lambda B, S, a, b: B.call(Dot4, B.call(Zero32), B.batch(Range7, a[0:4], axes=(0,)),
                                            B.batch(RangeNeg3, b[0:4], axes=(0,))))
    root = Program(narrow).root
    fn = P.facts(root)
    k_zero, k_dot = (next(k for k, n in enumerate(root.body.nodes) if n.fn is p) for p in (Zero32, Dot4))
    assert fn.node(k_zero).one == P.fact((0, 0), ("static",))
    assert fn.node(k_dot).one.interval == (-84, 84)


def test_a_composite_over_exact_steps_declares_what_its_bounds_prove():
    """The s8 chain declares its words' intervals and the Build holds its body to them; a limb split's bound then reads its
    producer's word: T_1 is within it, T_2 is not. A word from an incoming accumulator with no interval proves none."""
    def use(name, T, t):
        return comp(name, [("x", Array(8, V16))], V8,
                    lambda B, S, x: B.call(Split17, B.call(T, B.batch(Quant, x, axes=(0,)), B.batch(Quant, x, axes=(0,)))[t]))

    T = chain("PortsTestChainDecl", T_S8)
    roundtrip(Program(use("PortsTestSplitT1", T, 0)))
    refused("E_PORT_BOUND", lambda: Program(use("PortsTestSplitT2", T, 1)), "parameter 'y' leaf 0: its declared bound "
            "[-65536, 65536] does not contain its producer's interval [-130048, 131072]")
    tight = chain("PortsTestChainTight", [(-65024, 65535), T_S8[1]])
    refused("E_PORT_INTERVAL", lambda: Program(use("PortsTestSplitTight", tight, 0)), "PortsTestChainTight_v1 declares "
            "Out(interval=(-65024, 65535)) at return leaf 0, but its body returns its producer's interval [-65024, 65536]")
    acc_in = chain("PortsTestChainAcc", T_S8, acc_param=True)
    acc_use = comp("PortsTestChainAccUse", [("c", V32), ("x", Array(8, V16))], Array(2, V32),
                   lambda B, S, c, x: B.call(acc_in, B.call(Add32, c, c), B.batch(Quant, x, axes=(0,)), B.batch(Quant, x, axes=(0,))))
    refused("E_PORT_INTERVAL", lambda: Program(acc_use), "PortsTestChainAcc_v1 declares", "it declares none")


def test_containment_is_of_encodings():
    """A constant's value is its encoding read signed, and an unsigned bound holds it when it holds that encoding."""
    u8 = primitive("PortsTestU8Bound", 1, (("x", V8),), V8, ports=Ports(params={"x": In(bound=(0, 200))}))(lambda x: x)
    for code, ok in ((200, True), (201, False)):
        c = primitive(f"PortsTestC{code}", 1, (), V8)(lambda code=code: code)
        f = comp(f"PortsTestConst{code}", [], V8, lambda B, S, c=c: B.call(u8, B.call(c)))
        if ok:
            assert P.facts(Program(f).root).node(0).one.interval == (code - 256, code - 256)
        else:
            refused("E_PORT_BOUND", lambda: Program(f), f"[0, 200] does not contain its producer's interval [{code - 256}, {code - 256}]")
    assert P.as_reading((-5, 5), 8) == (-5, 5) and P.as_reading((-5, 5), 8, signed=False) is None
    assert P.as_reading((0, 255), 8) == (-128, 127) and P.as_reading((0, 200), 8) is None and P.as_reading((200, 250), 8) == (-56, -6)
    assert P.reading((-5, 250), 8) is None and P.reading((-5, 250), 16) == (-5, 250)


def test_a_transfer_is_a_primitives_and_gives_an_interval():
    with pytest.raises(PortError, match="E_PORT_DECL.*only a primitive declares a transfer"):
        Program(comp("PortsTestCompTransfer", [("x", V32)], V32, lambda B, S, x: x, ports=Ports(transfer=lambda x: x)))
    with pytest.raises(PortError, match="E_PORT_DECL.*declare no Out\\(interval\\) beside it"):
        primitive("PortsTestBoth", 1, (("x", V32),), V32, ports=Ports(ret=Out(interval=(0, 1)), transfer=lambda x: x))(lambda x: x)
    bad = primitive("PortsTestBadTransfer", 1, (("x", V32),), V32, ports=Ports(transfer=lambda x: (5, 1)))(lambda x: x)
    with pytest.raises(PortError, match="E_PORT_DECL.*PortsTestBadTransfer_v1's transfer"):
        Program(comp("PortsTestBadTransferUse", [("x", V32)], V32, lambda B, S, x: B.call(bad, x)))
    desc = encode_program(Program(comp("PortsTestTransferCodec", [("x", Array(8, V16))], Array(2, V32),
                                       lambda B, S, x: B.call(chain("PortsTestChainCodec", T_S8), B.batch(Quant, x, axes=(0,)),
                                                              B.batch(Quant, x, axes=(0,))))))
    tampered = copy.deepcopy(desc)
    del tampered["definitions"]["PortsTestDot4_v1"]["ports"]["transfer"]
    with pytest.raises(ValueError, match="PortsTestDot4_v1 ports differ from the registry"):
        decode_program(tampered, REGISTRY)
    tampered = copy.deepcopy(desc)
    (cid,) = [fid for fid in desc["definitions"] if fid.startswith("PortsTestChainCodec")]
    tampered["definitions"][cid]["ports"]["transfer"] = True
    with pytest.raises(PortError, match="E_PORT_DECL.*only a primitive declares a transfer"):
        decode_program(tampered, REGISTRY)


def test_malformed_interval_declarations():
    with pytest.raises(PortError, match="E_PORT_DECL.*no interval of a Value<8>"):
        primitive("PortsTestBadIv", 1, (("x", V8),), V8, ports=Ports(ret=Out(interval=(-200, 0))))(lambda x: x)
    with pytest.raises(PortError, match="E_PORT_DECL.*one width"):
        P.bind_ports("T", (("r", Record(("a", V8), ("b", V16))),), V8, Ports(params={"r": In(bound=(0, 1))}))
    with pytest.raises(PortError, match="E_PORT_DECL.*lo <= hi"):
        In(bound=(3, 2))
    with pytest.raises(PortError, match="E_PORT_DECL.*has no parameter"):
        P.bind_ports("T", (("x", V8),), V8, Ports(params={"y": In(bound=(0, 1))}))


# --- rule 3: sinks ------------------------------------------------------------------------------------------------------------

def test_a_sink_reaches_a_receipt_port_or_the_outputs():
    f = comp("PortsTestReceipt", [("acc", V32), ("x", V32)], TupleT(V32, V32),
             lambda B, S, acc, x: tuple_of(B.call(Absorb, acc, B.call(Fold, x)), B.call(Fold, x)))
    prog = Program(f)
    roundtrip(prog)
    assert P.facts(prog.root).ret().one.sink


def test_a_sink_read_by_any_other_port_is_refused():
    f = comp("PortsTestLeak", [("x", V32)], V32, lambda B, S, x: B.call(Add32, B.call(Fold, x), x))
    refused("E_PORT_SINK", lambda: Program(f), "parameter 'a' leaf 0: reads a sink", "reaches only a receipt port")
    inner = comp("PortsTestLeakIn", [("r", V32)], V32, lambda B, S, r: B.call(Add32, r, r))
    refused("E_PORT_SINK", lambda: Program(comp("PortsTestLeakOut", [("x", V32)], V32,
                                                lambda B, S, x: B.call(inner, B.call(Fold, x)))), "PortsTestLeakIn_v1 node 0")


def test_a_composite_declares_the_sinks_it_returns():
    hide = comp("PortsTestHide", [("x", V32)], V32, lambda B, S, x: B.call(Fold, x))
    refused("E_PORT_SINK", lambda: Program(comp("PortsTestHideUse", [("x", V32)], V32, lambda B, S, x: B.call(hide, x))),
            "PortsTestHide_v1 returns a sink at return leaf 0 without declaring it")
    show = comp("PortsTestShow", [("x", V32)], V32, lambda B, S, x: B.call(Fold, x), ports=Ports(ret=Out(sink=True)))
    Program(comp("PortsTestShowUse", [("x", V32)], V32, lambda B, S, x: B.call(show, x)))
    refused("E_PORT_SINK", lambda: Program(comp("PortsTestShowLeak", [("x", V32)], V32,
                                                lambda B, S, x: B.call(Add32, B.call(show, x), x))), "reads a sink")


def test_a_receipt_port_returns_only_sinks():
    with pytest.raises(PortError, match="E_PORT_DECL.*returns only sinks"):
        P.bind_ports("T", (("r", V32),), V32, Ports(params={"r": In(receipt=True)}))
    fold2 = comp("PortsTestFold2", [("acc", V32), ("r", V32)], V32, lambda B, S, acc, r: B.call(Absorb, acc, r),
                 ports=Ports(params={"r": In(receipt=True)}, ret=Out(sink=True)))
    Program(comp("PortsTestFold2Use", [("a", V32), ("x", V32)], V32, lambda B, S, a, x: B.call(fold2, a, B.call(Fold, x))))


# --- rule 4: kinds ------------------------------------------------------------------------------------------------------------

COIN = Ports(params={"key": In(kind="coin")})


def test_the_noise_derivation_takes_a_coin_and_statics():
    f = comp("PortsTestNoiseOk", [("key", V32)], V8, lambda B, S, key: B.call(Noise, key, B.call(Tag)), ports=COIN)
    prog = Program(f)
    desc, dec = roundtrip(prog)
    assert P.facts(dec.root).node(0).one.kind == "coin"
    assert desc["definitions"][desc["root"]]["ports"] == {"inputs": [[0, 0, 1, {"kind": "coin"}]]}


def test_the_noise_derivation_refuses_anything_else():
    plain = comp("PortsTestNoiseIn", [("key", V32)], V8, lambda B, S, key: B.call(Noise, key, B.call(Tag)))
    refused("E_PORT_KIND", lambda: Program(plain), "parameter 'key' leaf 0: takes kind=coin", "but reads a run input")
    mixed = comp("PortsTestNoiseGate", [("key", V32), ("x", V32)], V8,
                 lambda B, S, key, x: B.call(Noise, B.call(Add32, x, x), B.call(Tag)), ports=COIN)
    refused("E_PORT_KIND", lambda: Program(mixed), "but reads a computed value")
    tag_in = comp("PortsTestNoiseTag", [("key", V32), ("t", V8)], V8, lambda B, S, key, t: B.call(Noise, key, t), ports=COIN)
    refused("E_PORT_KIND", lambda: Program(tag_in), "parameter 'tag' leaf 0: takes kind=static, a constant, but reads a run input")
    refused("E_PORT_KIND", lambda: Program(comp("PortsTestNoiseReg", [("key", V32)], V8, lambda B, S, key: B.call(Noise, key, B.call(Tag)),
                                                ports=COIN), registered=("key",)), "drawn per run and never registered")
    refused("E_PORT_KIND", lambda: Program(comp("PortsTestStaticIn", [("t", V8)], V8, lambda B, S, t: t,
                                                ports=Ports(params={"t": In(kind="static")}))), "a static is a constant")


def test_a_harness_evaluates_a_definition_with_coin_and_static_ports():
    wrap = comp("PortsTestNoiseWrap", [("key", V32), ("tag", V8)], V8, lambda B, S, key, tag: B.call(Noise, key, tag),
                ports=Ports(params={"key": In(kind="coin"), "tag": In(kind="static")}))
    prog, _call = standalone_call(bind(wrap))
    assert P.facts(prog.root).typed
    assert evaluate(bind(wrap), 0x01020304, 7) == (Noise.evaluate(0x01020304, 7),)
    with pytest.raises(PortError, match="E_PORT_KIND.*only a harness"):
        decode_program(encode_program(prog), REGISTRY)


# --- model weights ------------------------------------------------------------------------------------------------------------

def _weighted(name: str) -> CompositeDefinition:
    return comp(name, [("prompt", V16), ("weights", Record(("w", Array(2, V16)), ("b", V32))), ("eps", V16)], V16,
                lambda B, S, prompt, weights, eps: B.call(Add16, B.call(Add16, prompt, weights.field("w")[0]), eps))


def _origins(root) -> list[set]:
    F = P.facts(root)
    return [set(F.node(k).one.origin) for k in range(4)]


def test_the_weights_are_marked_theta_outside_every_digest():
    """The weights record lowers to input nodes 1 and 2, whose leaves have origin theta whatever their binding time; the marking
    is the annotation alone, so the descriptor's digest and every definition entry are the unmarked Program's."""
    f = _weighted("PortsTestTheta")
    plain, marked = Program(f), Program(f, theta=("weights",))
    desc = encode_program(marked)
    assert desc["annotations"] == {"theta": [1, 2]} and encode_program(plain)["annotations"] == {}
    assert program_digest(desc) == program_digest(plain) and desc["definitions"] == encode_program(plain)["definitions"]
    assert _origins(plain.root) == [{"input"}] * 4
    assert _origins(marked.root) == [{"input"}, {"theta"}, {"theta"}, {"input"}]
    assert _origins(decode_program(desc, REGISTRY).root) == _origins(marked.root)
    registered = Program(f, theta=("weights",), registered=("weights", "eps"))
    assert _origins(registered.root) == [{"input"}, {"theta"}, {"theta"}, {"registered"}]
    assert program_digest(registered) == program_digest(Program(f, registered=("weights", "eps")))


def test_a_coin_is_never_a_weight_and_theta_names_parameters():
    """A theta coin is refused, theta names parameters, and a `theta` annotation passed in (another layout's, when a Program is
    rebuilt from one) is dropped rather than read."""
    coin = comp("PortsTestThetaCoin", [("key", V32)], V8, lambda B, S, key: B.call(Noise, key, B.call(Tag)), ports=COIN)
    refused("E_PORT_KIND", lambda: Program(coin, theta=("key",)), "input node 0: a coin is the verifier's, never a model weight")
    with pytest.raises(ValueError, match=re.escape("has no parameter ['w'] to mark theta")):
        Program(_weighted("PortsTestThetaName"), theta=("w",))
    carried = Program(_weighted("PortsTestThetaAnn"), annotations={"theta": [1], "note": "kept"})
    assert carried.annotations == {"note": "kept"} and _origins(carried.root) == [{"input"}] * 4


def test_the_decoder_ignores_the_marking_and_the_facts_check_it():
    """The decoder's verdict and digest don't read the annotation; `ports.facts` refuses one that names a coin or anything but
    the root's input nodes."""
    desc = encode_program(Program(comp("PortsTestThetaTamper", [("key", V32)], V8,
                                       lambda B, S, key: B.call(Noise, key, B.call(Tag)), ports=COIN)))
    for theta, code, needle in (([0], "E_PORT_KIND", "never a model weight"), ([1], "E_PORT_DECL", "lists input nodes"),
                                ("0", "E_PORT_DECL", "lists input nodes"), ([True], "E_PORT_DECL", "lists input nodes")):
        bad = copy.deepcopy(desc)
        bad["annotations"]["theta"] = theta
        dec = decode_program(bad, REGISTRY)
        assert program_digest(encode_program(dec)) == program_digest(desc)
        refused(code, lambda: P.facts(dec.root), needle)


# --- rule 5: one type per committed word --------------------------------------------------------------------------------------

def test_a_runtime_word_is_read_only_as_its_producers_type():
    pun = comp("PortsTestPun", [("x", V32)], V8, lambda B, S, x: B.call(Limbs, B.call(Acc, x)))
    refused("E_PORT_WORD", lambda: Program(pun), "reads a s8x4 word, but its value is a runtime s32 word",
            "Split a runtime word into limbs with a registered LimbSplit Definition")
    untyped = comp("PortsTestPunUntyped", [("x", V32)], V8, lambda B, S, x: B.call(Limbs, B.call(Add32, x, x)))
    refused("E_PORT_WORD", lambda: Program(untyped), "runtime untyped word")
    Program(comp("PortsTestSplit", [("x", V32)], V8, lambda B, S, x: B.call(LimbSplit, B.call(Acc, x))))


def test_an_untyped_input_or_constant_takes_its_readers_type_and_a_typed_input_keeps_its_own():
    for reg in ((), ("x",)):
        prog = Program(comp("PortsTestReinterp", [("x", V32)], V8, lambda B, S, x: B.call(Limbs, x)), registered=reg)
        roundtrip(prog)
    typed = Ports(params={"x": In(word="s32")})
    for reg in ((), ("x",)):
        roundtrip(Program(comp("PortsTestTypedIn", [("x", V32)], V8, lambda B, S, x: B.call(LimbSplit, x), ports=typed), registered=reg))
        refused("E_PORT_WORD", lambda: Program(comp("PortsTestTypedPun", [("x", V32)], V8, lambda B, S, x: B.call(Limbs, x), ports=typed),
                                               registered=reg),
                "reads a s8x4 word, but its value is a s32 word", "a typed input's at its port")


def _limbs(w: int) -> tuple[list[int], list[int]]:
    """The balanced s8 limbs of a word read as s32 (low first, each in [-128, 127], the top one modulo 256), and the carry into
    each limb past the first: 1 when the limb below it is negative."""
    v, limbs = _s(w, 32), []
    for _ in range(4):
        limbs.append(((v + 128) & 0xFF) - 128)
        v = (v - limbs[-1]) >> 8
    return limbs, [int(x < 0) for x in limbs[:3]]


LIMB_INDEX = [primitive(f"PortsTestLimbIndex{k}", 1, (), V8)(lambda k=k: k) for k in range(4)]
SPLIT_PORTS = Ports(params={"w": In(word="s32"), "k": In(kind="static")})


@primitive("PortsTestLimbAt", 1, (("w", V32), ("k", V8)), V8, ports=SPLIT_PORTS)
def LimbAt(w, k):
    """Limb k of an s32 word's balanced s8 decomposition."""
    return _limbs(w)[0][k] & 0xFF


@primitive("PortsTestCarryAt", 1, (("w", V32), ("k", V8)), Value(1), ports=SPLIT_PORTS)
def CarryAt(w, k):
    """The carry into limb k (k >= 1) of an s32 word's balanced s8 decomposition."""
    return _limbs(w)[1][k - 1]


#: an explicit limb split: it reads the word as its producer's type and computes four s8 limbs and the carry into each past the
#: first (the stacking property budgets those carries at this node)
LIMB_SPLIT = comp("PortsTestLimbSplitS32", [("w", V32)], TupleT(Array(4, V8), Array(3, Value(1))),
                  lambda B, S, w: tuple_of(array_of([B.call(LimbAt, w, B.call(LIMB_INDEX[k])) for k in range(4)]),
                                           array_of([B.call(CarryAt, w, B.call(LIMB_INDEX[k])) for k in range(1, 4)])),
                  ports=Ports(params={"w": In(word="s32")}))


@primitive("PortsTestId8", 1, (("x", V8),), V8)
def Id8(x):
    return x


@primitive("PortsTestLo8", 1, (("w", V32),), V8)
def Lo8(w):
    """The low byte of a word: a gate that computes an 8-bit lane."""
    return w & 0xFF


def lanes(c: Coll, n: int = 32) -> Coll:
    """`n` references to the one leaf of `c`, typed as `n` 8-bit lanes."""
    return Coll(Array(n, V8), concat([c.refs] * n))


def test_a_limb_split_is_a_definition_that_computes_the_limbs_and_their_carries():
    use = comp("PortsTestLimbSplitUse", [("x", V32)], TupleT(Array(4, V8), Array(3, Value(1))),
               lambda B, S, x: B.call(LIMB_SPLIT, B.call(Acc, x)))
    roundtrip(Program(use))
    for w in (0, 0x7F80FF01, 0x80000000, 0xFFFFFFFF, 0x12345678):
        out = evaluate(bind(use), w)
        limbs, carries = _limbs(w)
        assert out == tuple(x & 0xFF for x in limbs) + tuple(carries)
        assert sum(_s(x, 8) << (8 * k) for k, x in enumerate(out[:4])) % (1 << 32) == w
    with pytest.raises(TypeError, match="E_REINTERPRET.*a runtime word is split into limbs only by a registered LimbSplit "
                                        "Definition, which computes the limbs and their carries"):
        Program(comp("PortsTestLimbPun", [("x", V32)], Array(4, V8), lambda B, S, x: lanes(B.call(Acc, x), 4)))


# --- E_REINTERPRET: a leaf is read at the width it was committed with ---------------------------------------------------------

def used(f: CompositeDefinition) -> CompositeDefinition:
    """A program function that calls `f`, so that `f`'s body is a composite's (the root inlines its program function's)."""
    params, ret = bind(f).signature()
    return comp(f"{f.name}Use", params, ret, lambda B, S, *args: B.call(f, *args))


def test_an_s32_word_typed_as_s8_lanes_is_refused_as_an_argument_and_as_a_return():
    """`Coll(Array(32, S8), refs to one s32 word)` names a 32-bit leaf at every 8-bit position: the Builder refuses it where a
    node reads it and where the body returns it, whether the word is a node's or a parameter's."""
    arg = comp("PortsTestLanesArg", [("x", V32)], Array(32, V8), lambda B, S, x: B.batch(Id8, lanes(B.call(Pair, x)), axes=(0,)))
    with pytest.raises(TypeError, match=r"E_REINTERPRET: node 1 \(batch PortsTestId8_v1\) argument 'x': leaf 0 is typed "
                                        r"Value<8> but names leaf 0 of node 0, a Value<32>: a silent reinterpretation"):
        Program(used(arg))
    ret = comp("PortsTestLanesRet", [("x", V32)], Array(32, V8), lambda B, S, x: lanes(B.call(Pair, x)))
    with pytest.raises(TypeError, match=r"E_REINTERPRET: the body's return: leaf 0 is typed Value<8> but names leaf 0 of node 0"):
        Program(used(ret))
    param = comp("PortsTestLanesParam", [("x", V32)], Array(32, V8), lambda B, S, x: lanes(x))
    with pytest.raises(TypeError, match="the body's return: leaf 0 is typed Value<8> but names leaf 0 of parameter 'x', a Value<32>"):
        Program(used(param))
    with pytest.raises(TypeError, match=r"E_REINTERPRET: node 1 \(batch PortsTestId8_v1\) argument 'x': leaf 0 is typed "
                                        r"Value<8> but names leaf 0 of node 0, a Value<32>"):
        Program(comp("PortsTestLanesRoot", [("x", V32)], Array(32, V8), lambda B, S, x: B.batch(Id8, lanes(x), axes=(0,))))


def _retarget(refs, src: int, dst: int):
    """`refs` (a descriptor's reference JSON) with every reference to node `src` naming node `dst` instead."""
    if isinstance(refs, dict):
        return {k: _retarget(v, src, dst) for k, v in refs.items()}
    if isinstance(refs, list):
        if len(refs) >= 3 and refs[0] == "n" and refs[1] == src:
            return ["n", dst, *refs[2:]]
        return [_retarget(r, src, dst) for r in refs]
    return refs


def test_the_decoder_refuses_a_tampered_word_typed_as_s8_lanes():
    """The same reinterpretation written into a descriptor, at a node's argument and at the return: lanes of an 8-bit word
    retargeted to the s32 word."""
    f = comp("PortsTestLanesOk", [("x", V32)], TupleT(Array(32, V8), Array(32, V8)),
             lambda B, S, x: (lambda y, lo: tuple_of(B.batch(Id8, lanes(lo), axes=(0,)), lanes(lo)))(B.call(Pair, x), B.call(Lo8, x)))
    desc, _ = roundtrip(Program(used(f)))
    body = desc["definitions"]["PortsTestLanesOk_v1"]["body"]
    assert [n["form"] for n in body["nodes"]] == ["primitive", "primitive", "batch"]
    for where, leaf, edit in (("node 2 (batch PortsTestId8_v1) argument 'x'", 0, lambda b: b["nodes"][2]["args"][0].update(
                                  refs=_retarget(b["nodes"][2]["args"][0]["refs"], 1, 0))),
                              ("the body's return", 32, lambda b: b.update(ret=_retarget(b["ret"], 1, 0)))):
        tampered = copy.deepcopy(desc)
        edit(tampered["definitions"]["PortsTestLanesOk_v1"]["body"])
        with pytest.raises(ValueError, match=re.escape(f"definition 'PortsTestLanesOk_v1': E_REINTERPRET: {where}: leaf {leaf} is "
                                                       f"typed Value<8> but names leaf 0 of node 0, a Value<32>")):
            decode_program(tampered, REGISTRY)


def test_a_node_naming_itself_or_a_later_node_at_another_width_decodes_for_its_checkers():
    """A node that names itself or a later node is an order violation, which decodes so that its checkers name it (PROTOCOL.md
    §4.4): the width check reads only the parameters and earlier nodes, so it doesn't preempt them whatever the widths."""
    from verity.ml.circuits.partition_object import order_violation
    f = comp("PortsTestLanesOk", [("x", V32)], TupleT(Array(32, V8), Array(32, V8)),
             lambda B, S, x: (lambda y, lo: tuple_of(B.batch(Id8, lanes(lo), axes=(0,)), lanes(lo)))(B.call(Pair, x), B.call(Lo8, x)))
    desc, _ = roundtrip(Program(used(f)))
    for node, target in ((1, 1), (0, 2)):
        tampered = copy.deepcopy(desc)
        tampered["definitions"]["PortsTestLanesOk_v1"]["body"]["nodes"][node]["args"][0]["refs"] = {"affine": ["n", target, 0, 1]}
        bad = order_violation(decode_program(tampered, REGISTRY))
        assert bad == {"definition": "PortsTestLanesOk_v1", "node": node, "arg": 0, "reference": ["n", target],
                       "why": "not topological"}


@primitive("PortsTestPair", 1, (("x", V32),), V32)
def Pair(x):
    """A 32-bit word no port types."""
    return x


def test_a_word_read_at_its_own_width_builds():
    """A same-width reshape, an aligned mixed tuple, strided views of an array of records and a whole read of it as another type
    of the same layout read every leaf at its width; the misaligned ones are refused at the first leaf they misread."""
    rec = comp("PortsTestRec", [("x", V32)], TupleT(V32, V8), lambda B, S, x: tuple_of(B.call(Pair, x), B.call(Lo8, x)))
    n = 100

    def case(name, ret, view):
        """`view(B, x, r)` over r, node 0: an `Array<n, (Value<32>, Value<8>)>`."""
        return used(comp(name, [("x", Array(4, V32)), ("v", Array(n, V32))], ret,
                         lambda B, S, x, v: view(B, x, B.batch(rec, v, axes=(0,)))))

    ok = [
        (Array(2, Array(2, V32)), lambda B, x, r: Coll(Array(2, Array(2, V32)), B.batch(Pair, x, axes=(0,)).refs)),
        (TupleT(V8, V32), lambda B, x, r: tuple_of(r[0][1], r[1][0])),
        (Array(n, V32), lambda B, x, r: Coll(Array(n, V32), Strided("n", 0, 0, 1, 2, 0, n))),
        (Array(n, V8), lambda B, x, r: Coll(Array(n, V8), Strided("n", 0, 1, 1, 2, 0, n))),
        (Array(n, Record(("a", V32), ("b", V8))), lambda B, x, r: Coll(Array(n, Record(("a", V32), ("b", V8))), r.refs)),
    ]
    for i, (ret, view) in enumerate(ok):
        roundtrip(Program(case(f"PortsTestWidthOk{i}", ret, view)))
    bad = [
        (TupleT(V32, V8), lambda B, x, r: Coll(TupleT(V32, V8), tuple_of(r[0][1], r[1][0]).refs), 0,
         "Value<32> but names leaf 1 of node 0, a Value<8>"),
        (Array(n, V32), lambda B, x, r: Coll(Array(n, V32), Strided("n", 0, 1, 1, 2, 0, n)), 0, "names leaf 1 of node 0, a Value<8>"),
        (Array(n, TupleT(V8, V32)), lambda B, x, r: Coll(Array(n, TupleT(V8, V32)), r.refs), 0, "names leaf 0 of node 0, a Value<32>"),
        (Array(n, V32), lambda B, x, r: Coll(Array(n, V32), concat([Strided("n", 0, 0, 1, 2, 0, n - 1), Affine("n", 0, 1, 1)])),
         n - 1, "names leaf 1 of node 0, a Value<8>"),
    ]
    for i, (ret, view, leaf, why) in enumerate(bad):
        with pytest.raises(TypeError, match=re.escape(f"the body's return: leaf {leaf} is typed")) as e:
            Program(case(f"PortsTestWidthBad{i}", ret, view))
        assert why in str(e.value), (why, str(e.value))


def test_a_part_log_cut_is_checked_past_its_longest_checked_cut():
    """Cuts of one `PartLog` (the key rows of successive attention calls) are checked part by part only past the longest cut
    already checked at that width; a tail of another width is still refused."""
    logs = []

    def body(B, S, x, tail):
        log = PartLog()
        logs.append(log)
        outs = []
        for t in range(6):
            log.append(B.call(Quant, x[t]).refs)
            if t >= 1:
                extra = [B.call(Add32, tail, tail).refs] if tail is not None and t == 5 else []
                outs.append(B.batch(AnyBound, Coll(Array(t + 1 + len(extra), V16), log.cut(t + 1, extra)), axes=(0,)))
        return tuple_of(*outs)

    rets = TupleT(*(Array(t + 1, V16) for t in range(1, 6)))
    prog = Program(comp("PortsTestLogOk", [("x", Array(6, V16))], rets, lambda B, S, x: body(B, S, x, None)))
    roundtrip(prog)
    token, done = logs[0].memo["defs.widths"]
    assert done == {16: 6}
    rets_bad = TupleT(*(Array(t + 1 + (t == 5), V16) for t in range(1, 6)))
    with pytest.raises(TypeError, match="argument 'a': leaf 6 is typed Value<16> but names leaf 0 of node .*, a Value<32>"):
        Program(comp("PortsTestLogBad", [("x", Array(6, V16)), ("y", V32)], rets_bad, lambda B, S, x, y: body(B, S, x, y)))


def test_a_composite_keeps_the_words_it_returns():
    keep = comp("PortsTestKeep", [("x", V32)], V32, lambda B, S, x: B.call(Acc, x), ports=Ports(ret=Out(word="s32")))
    Program(comp("PortsTestKeepUse", [("x", V32)], V8, lambda B, S, x: B.call(LimbSplit, B.call(keep, x))))
    refused("E_PORT_WORD", lambda: Program(comp("PortsTestKeepPun", [("x", V32)], V8, lambda B, S, x: B.call(Limbs, B.call(keep, x)))),
            "reads a s8x4 word, but its value is a runtime s32 word")
    drop = comp("PortsTestDrop", [("x", V32)], V32, lambda B, S, x: B.call(Acc, x))
    refused("E_PORT_WORD", lambda: Program(comp("PortsTestDropUse", [("x", V32)], V32, lambda B, S, x: B.call(drop, x))),
            "PortsTestDrop_v1 returns a s32 word at return leaf 0 without declaring it")
    fix = comp("PortsTestFix", [("x", V32)], V32, lambda B, S, x: B.call(Acc, x), ports=Ports(ret=Out(word="s8x4")))
    refused("E_PORT_WORD", lambda: Program(comp("PortsTestFixUse", [("x", V32)], V32, lambda B, S, x: B.call(fix, x))),
            "a composite keeps the type its words' producer fixed, and never fixes one")


# --- rules 6 and 7: coin flow and one use -------------------------------------------------------------------------------------

KEY_X = [("key", V32), ("x", V8)]
KEYS = Ports(params={"keys": In(kind="coin")})


def test_a_coin_reaches_only_a_coin_port():
    derive = comp("PortsTestCoinDerive", KEY_X, V8, lambda B, S, key, x: B.call(Add8, x, B.call(Noise, key, B.call(Tag))), ports=COIN)
    roundtrip(Program(derive))
    inject = comp("PortsTestCoinInject", KEY_X, V8, lambda B, S, key, x: B.call(Dither, x, key), ports=COIN)
    roundtrip(Program(inject))
    keyed = comp("PortsTestCoinKeyed", [("key", V32), ("x", V32)], V32, lambda B, S, key, x: B.call(KeyFold, x, key), ports=COIN)
    roundtrip(Program(keyed))
    direct = comp("PortsTestCoinAdd", [("key", V32), ("x", V32)], V32, lambda B, S, key, x: B.call(Add32, key, x), ports=COIN)
    refused("E_COIN_FLOW", lambda: Program(direct), "(primitive PortsTestAdd32_v1) parameter 'a' leaf 0: reads a coin",
            "re-derive the noise")
    unwind = comp("PortsTestCoinUnwind", KEY_X, V8,
                  lambda B, S, key, x: B.call(Add8, B.call(Dither, x, key), B.call(Noise, B.call(Add32, key, key), B.call(Tag))),
                  ports=COIN)
    refused("E_COIN_FLOW", lambda: Program(unwind), "PortsTestAdd32_v1) parameter 'a' leaf 0: reads a coin")
    inner = comp("PortsTestCoinChainIn", [("k", V32)], V32, lambda B, S, k: B.call(Add32, k, k))
    refused("E_COIN_FLOW", lambda: Program(comp("PortsTestCoinChain", [("key", V32)], V32, lambda B, S, key: B.call(inner, key), ports=COIN)),
            "PortsTestCoinChainIn_v1 node 0", "reads a coin")


def test_a_coin_never_leaves_through_a_return():
    passing = comp("PortsTestCoinPass", [("k", V32)], V32, lambda B, S, k: k)
    refused("E_COIN_FLOW", lambda: Program(comp("PortsTestCoinLaunder", [("key", V32), ("x", V32)], V32,
                                                lambda B, S, key, x: B.call(Add32, B.call(passing, key), x), ports=COIN)),
            "PortsTestCoinPass_v1 returns a coin at return leaf 0", "never its return")
    refused("E_COIN_FLOW", lambda: Program(comp("PortsTestCoinOut", [("key", V32)], V32, lambda B, S, key: key, ports=COIN)),
            "returns a coin at return leaf 0", "never the program's outputs")


def test_a_noise_coin_is_read_once():
    def twice(B, S, key, x):
        return B.call(Add8, B.call(Dither, x, key), B.call(Noise, key, B.call(Tag)))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestCoinTwice", KEY_X, V8, twice, ports=COIN)),
            "input node 0 leaf 0: a coin that a noise primitive reads is read once", "read 2 times",
            "PortsTestDither_v1", "PortsTestNoise_v1")

    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestCoinKeyNoise", [("key", V32), ("x", V8), ("y", V32)], TupleT(V8, V32),
                                                lambda B, S, key, x, y: tuple_of(B.call(Dither, x, key), B.call(KeyFold, y, key)),
                                                ports=COIN)),
            "a receipt key from a noise bit's coin", "PortsTestDither_v1", "PortsTestKeyFold_v1")
    block = comp("PortsTestCoinBlock", KEY_X, V8, lambda B, S, key, x: B.call(Dither, x, key))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestCoinCalls", KEY_X, V8,
                                                lambda B, S, key, x: B.call(Add8, B.call(block, key, x), B.call(block, key, x)),
                                                ports=COIN)), "read 2 times", "PortsTestCoinBlock_v1 node 0")
    shared = comp("PortsTestCoinBatchShared", [("key", V32), ("x", Array(2, V8))], Array(2, V8),
                  lambda B, S, key, x: B.batch(Dither, x, key, axes=(0, None)), ports=COIN)
    refused("E_COIN_REUSE", lambda: Program(shared), "member 0", "member 1")
    step = comp("PortsTestCoinStep", [("c", V8), ("x", V8), ("key", V32)], TupleT(V8, V8),
                lambda B, S, c, x, key: tuple_of(B.call(Add8, c, x), B.call(Dither, x, key)))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestCoinScan", [("key", V32), ("x", Array(3, V8))], TupleT(V8, Array(3, V8)),
                                                lambda B, S, key, x: B.scan(step, x[0], xs=(x,), shared=(key,)), ports=COIN)),
            "step 0", "step 1")
    own = comp("PortsTestCoinOwn", [("keys", Array(2, V32)), ("x", Array(2, V8))], Array(2, V8),
               lambda B, S, keys, x: B.batch(Dither, x, keys, axes=(0, 0)), ports=KEYS)
    roundtrip(Program(own))
    roundtrip(Program(comp("PortsTestCoinScanOwn", [("keys", Array(3, V32)), ("x", Array(3, V8))], TupleT(V8, Array(3, V8)),
                           lambda B, S, keys, x: B.scan(step, x[0], xs=(x, keys)), ports=KEYS)))


def test_a_receipt_key_may_key_many_folds():
    many = comp("PortsTestCoinKeys", [("key", V32), ("x", Array(3, V32))], TupleT(Array(3, V32), V32),
                lambda B, S, key, x: tuple_of(B.batch(KeyFold, x, key, axes=(0, None)), B.call(KeyFold, x[0], key)), ports=COIN)
    roundtrip(Program(many))
    split = comp("PortsTestCoinSplit", [("keys", Array(2, V32)), ("x", V8), ("y", V32)], TupleT(V8, V32),
                 lambda B, S, keys, x, y: tuple_of(B.call(Dither, x, keys[0]), B.call(KeyFold, y, keys[1])), ports=KEYS)
    roundtrip(Program(split))


def test_noise_is_read_at_most_once():
    def copies(B, S, key, x):
        n = B.call(Noise, key, B.call(Tag))
        return tuple_of(B.call(Add8, x, n), B.call(Add8, x, n))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestNoiseCopies", KEY_X, TupleT(V8, V8), copies, ports=COIN)),
            "PortsTestNoise_v1) return leaf 0: noise is the noise of one operand of one instance", "read 2 times")
    made = comp("PortsTestNoiseMade", [("key", V32)], V8, lambda B, S, key: B.call(Noise, key, B.call(Tag)))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestNoiseReturned", KEY_X, V8,
                                                lambda B, S, key, x: (lambda n: B.call(Add8, n, n))(B.call(made, key)), ports=COIN)),
            "read 2 times")
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestNoiseRows", [("key", V32), ("x", Array(2, V8))], Array(2, V8),
                                                lambda B, S, key, x: B.batch(Add8, x, B.call(made, key), axes=(0, None)),
                                                ports=COIN)), "member 0", "member 1")
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestNoiseOperands", KEY_X, TupleT(V8, V8),
                                                lambda B, S, key, x: (lambda d: tuple_of(B.call(Add8, d, x), d))(B.call(Dither, x, key)),
                                                ports=COIN)), "PortsTestDither_v1) return leaf 0", "outputs")

    def downstream(B, S, key, x):
        y = B.call(Add8, B.call(Dither, x, key), x)
        return tuple_of(B.call(Add8, y, y), y)
    roundtrip(Program(comp("PortsTestNoiseOnce", KEY_X, TupleT(V8, V8), downstream, ports=COIN)))
    passed = comp("PortsTestNoisePassed", [("n", V8), ("x", V8)], TupleT(V8, V8), lambda B, S, n, x: tuple_of(n, B.call(Add8, x, x)))
    roundtrip(Program(comp("PortsTestNoiseThrough", KEY_X, V8,
                           lambda B, S, key, x: B.call(Add8, B.call(passed, B.call(made, key), x)[0], x), ports=COIN)))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestNoiseThroughTwice", KEY_X, V8,
                                                lambda B, S, key, x: (lambda r: B.call(Add8, r[0], r[0]))(B.call(passed, B.call(made, key), x)),
                                                ports=COIN)), "PortsTestNoiseMade_v1 node 1")


def test_the_decoder_holds_a_descriptor_to_the_coin_rules():
    desc = encode_program(Program(comp("PortsTestNoiseDecoded", KEY_X, TupleT(V8, V8),
                                       lambda B, S, key, x: tuple_of(B.call(Dither, x, key), x), ports=COIN)))
    dup = copy.deepcopy(desc)
    ret = dup["definitions"][desc["root"]]["body"]["ret"]["concat"]
    ret[1] = ret[0]
    refused("E_COIN_REUSE", lambda: decode_program(dup, REGISTRY), "PortsTestDither_v1) return leaf 0", "read 2 times")


# --- rule 8: seeds ------------------------------------------------------------------------------------------------------------

@primitive("PortsTestByte", 1, (("w", V32),), V8)
def Byte(w):
    return w & 0xFF


@primitive("PortsTestWiden", 1, (("x", V8),), V32)
def Widen(x):
    return x


@primitive("PortsTestNeg8", 1, (("x", V8),), V8)
def Neg8(x):
    return -x & 0xFF


SEED = Ports(params={"salt": In(kind="coin"), "position": In(kind="position")})
COIN_SALT = Ports(params={"salt": In(kind="coin")})
#: a seed whose body computes on the salt: refused as coin flow without the position port that makes it a seed
Seed = comp("PortsTestSeed", [("salt", V32), ("position", V32)], V32, lambda B, S, salt, position: B.call(Add32, salt, position),
            ports=SEED)
UNIT_IN = [("a", V8), ("b", V8), ("salt", V32), ("position", V32)]
#: a unit takes the coin and its position, and returns a sink: no seed
UNIT_PORTS = Ports(params={"salt": In(kind="coin"), "position": In(kind="position")}, ret=(None, Out(sink=True)))
#: a unit's noise from its seed, computed in a callee of the unit
noise_of = comp("PortsTestNoiseOf", [("s", V32)], V8, lambda B, S, s: B.call(Byte, s))


def unit(name: str, z) -> CompositeDefinition:
    """A unit: its seed, the noised operand x = a + noise, z = `z`(B, a, b, s, x), and a receipt folding the seed (read twice)
    and x."""
    def body(B, S, a, b, salt, position):
        s = B.call(Seed, salt, position)
        x = B.call(Add8, a, B.call(noise_of, s))
        return tuple_of(z(B, a, b, s, x), B.call(Fold, B.call(Add32, B.call(Add32, s, s), B.call(Widen, x))))
    return comp(name, UNIT_IN, TupleT(V8, V32), body, ports=UNIT_PORTS)


Unit = unit("PortsTestUnit", lambda B, a, b, s, x: B.call(Add8, a, b))
UNITS_IN = [("a", Array(3, V8)), ("b", Array(3, V8)), ("salt", V32), ("positions", Array(3, V32))]
SALTED = Ports(params={"salt": In(kind="coin"), "positions": In(kind="position")})


def units(name: str, fn=Unit, axes=(0, 0, None, 0)) -> CompositeDefinition:
    return comp(name, UNITS_IN, Array(3, TupleT(V8, V32)),
                lambda B, S, a, b, salt, positions: B.batch(fn, a, b, salt, positions, axes=axes), ports=SALTED)


def test_a_seed_takes_the_coin_and_a_position():
    desc, dec = roundtrip(Program(units("PortsTestUnits")))
    assert P.facts(dec.root).node(3).one.kind == "position"
    assert desc["definitions"][desc["root"]]["ports"] == {"inputs": [[2, 0, 1, {"kind": "coin"}], [3, 0, 3, {"kind": "position"}]]}
    unkeyed = comp("PortsTestSeedUnkeyed", [("salt", V32), ("position", V32)], V32,
                   lambda B, S, salt, position: B.call(Add32, salt, position), ports=COIN_SALT)
    refused("E_COIN_FLOW", lambda: Program(comp("PortsTestSeedNoPosition", [("salt", V32), ("position", V32)], V32,
                                                lambda B, S, salt, position: B.call(unkeyed, salt, position), ports=COIN_SALT)),
            "PortsTestSeedUnkeyed_v1 node 0", "reads a coin")


def test_a_position_is_a_run_input_passed_through():
    one = [("a", V8), ("b", V8), ("salt", V32), ("position", V32)]
    computed = comp("PortsTestPositionComputed", one, TupleT(V8, V32),
                    lambda B, S, a, b, salt, position: B.call(Unit, a, b, salt, B.call(Add32, position, position)),
                    ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")}))
    refused("E_PORT_KIND", lambda: Program(computed), "parameter 'position' leaf 0: takes kind=position",
            "but reads a computed value", "never computes its position")
    undeclared = comp("PortsTestPositionInput", one, TupleT(V8, V32),
                      lambda B, S, a, b, salt, position: B.call(Unit, a, b, salt, position), ports=COIN_SALT)
    refused("E_PORT_KIND", lambda: Program(undeclared), "takes kind=position", "but reads a run input")


def test_many_seeds_read_one_coin_at_distinct_positions():
    roundtrip(Program(units("PortsTestUnitsOwn")))
    shared = comp("PortsTestUnitsShared", [*UNITS_IN[:3], ("position", V32)], Array(3, TupleT(V8, V32)),
                  lambda B, S, a, b, salt, position: B.batch(Unit, a, b, salt, position, axes=(0, 0, None, None)), ports=SEED)
    refused("E_COIN_REUSE", lambda: Program(shared), "input node 3 leaf 0: a position seeds one unit, but 3 seeds read it",
            "member 0", "member 1", "PortsTestSeed_v1")
    twice = comp("PortsTestUnitsTwice", [("a", V8), ("b", V8), ("salt", V32), ("position", V32)], TupleT(TupleT(V8, V32), TupleT(V8, V32)),
                 lambda B, S, a, b, salt, position: tuple_of(B.call(Unit, a, b, salt, position), B.call(Unit, b, a, salt, position)),
                 ports=SEED)
    refused("E_COIN_REUSE", lambda: Program(twice), "a position seeds one unit")


def test_a_seed_coin_is_read_by_seeds_alone():
    def keyed(B, S, a, b, salt, positions):
        return tuple_of(B.batch(Unit, a, b, salt, positions, axes=(0, 0, None, 0)), B.call(KeyFold, positions[0], salt))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestSeedKeyed", UNITS_IN, TupleT(Array(3, TupleT(V8, V32)), V32), keyed,
                                                ports=SALTED)),
            "input node 2 leaf 0: a coin that a seed reads is read by seeds alone", "PortsTestKeyFold_v1")

    def dithered(B, S, a, b, salt, positions):
        return tuple_of(B.batch(Unit, a, b, salt, positions, axes=(0, 0, None, 0)), B.call(Dither, a[0], salt))
    refused("E_COIN_REUSE", lambda: Program(comp("PortsTestSeedDithered", UNITS_IN, TupleT(Array(3, TupleT(V8, V32)), V8),
                                                dithered, ports=SALTED)),
            "a coin that a noise primitive reads is read once", "PortsTestDither_v1")


def test_no_port_rule_follows_a_seeds_noise():
    """A unit's z that peels its noise back out (x - noise = a, as PoUW's one-stream tile does) and one that doesn't are one
    dataflow, so the port rules admit both and the unit's own check (PoUW's `noise-in-z` and `stream`) tells them apart."""
    peeled = unit("PortsTestUnitPeelZ", lambda B, a, b, s, x: B.call(Add8, x, B.call(Neg8, B.call(noise_of, s))))
    assert [evaluate(bind(peeled), a, 2, 5, 7)[0] for a in (0, 1, 200, 255)] == [0, 1, 200, 255]
    for name, z in (("PeelZ", None), ("NoisyZ", lambda B, a, b, s, x: x), ("SeedZ", lambda B, a, b, s, x: B.call(Byte, s)),
                    ("NoiseZ", lambda B, a, b, s, x: B.call(noise_of, s))):
        roundtrip(Program(units(f"PortsTestUnits{name}", peeled if z is None else unit(f"PortsTestUnit{name}", z))))
    roundtrip(Program(comp("PortsTestSeedOut", [("salt", V32), ("position", V32)], V32,
                           lambda B, S, salt, position: B.call(Seed, salt, position), ports=SEED)))


def test_a_harness_evaluates_a_seed():
    assert evaluate(bind(Seed), 5, 7) == (12,)
    assert evaluate(bind(Unit), 1, 2, 5, 7) == (3, Fold.evaluate((24 + 13) & 0xFFFFFFFF))


# --- the descriptor -----------------------------------------------------------------------------------------------------------

def _typed_program() -> Program:
    up = comp("PortsTestCodecUp", [("x", Array(2, V16))], Array(2, V16), lambda B, S, x: B.batch(Quant, x, axes=(0,)),
              ports=Ports(ret=Out(interval=(-128, 127))))
    f = comp("PortsTestCodec", [("x", Array(2, V16)), ("key", V32)], TupleT(Array(2, V32), V8),
             lambda B, S, x, key: tuple_of(B.batch(Mul, B.call(up, x), B.call(up, x), axes=(0, 0)), B.call(Noise, key, B.call(Tag))),
             ports=COIN)
    return Program(f)


def test_the_decoder_rederives_every_fact():
    prog = _typed_program()
    desc, dec = roundtrip(prog)
    assert P.facts(dec.root).node(0).to_json() == P.facts(prog.root).node(0).to_json()
    up = next(fid for fid in desc["definitions"] if fid.startswith("PortsTestCodecUp"))
    tampered = copy.deepcopy(desc)
    del tampered["definitions"][up]["ports"]
    refused("E_PORT_BOUND", lambda: decode_program(tampered, REGISTRY), "it declares none")
    tampered = copy.deepcopy(desc)
    del tampered["definitions"]["PortsTestMul_v1"]["ports"]
    with pytest.raises(ValueError, match="PortsTestMul_v1 ports differ from the registry"):
        decode_program(tampered, REGISTRY)
    tampered = copy.deepcopy(desc)
    del tampered["definitions"][desc["root"]]["ports"]
    refused("E_PORT_KIND", lambda: decode_program(tampered, REGISTRY), "but reads a run input")


def test_the_decoder_refuses_non_canonical_ports():
    desc = encode_program(_typed_program())
    up = next(fid for fid in desc["definitions"] if fid.startswith("PortsTestCodecUp"))
    for bad, why in (([[0, 1, {"interval": [-128, 127]}], [1, 2, {"interval": [-128, 127]}]], "not canonical"),
                     ([[0, 2, {"interval": [-128, 127], "sink": False}]], "not canonical"),
                     ([[0, 3, {"interval": [-128, 127]}]], "leaves [0, 3) of a type with 2"),
                     ([[0, 2, {"bound": [0, 1]}]], "keys from")):
        tampered = copy.deepcopy(desc)
        tampered["definitions"][up]["ports"] = {"ret": bad}
        with pytest.raises(PortError, match="E_PORT_DECL") as e:
            decode_program(tampered, REGISTRY)
        assert why in str(e.value), (why, str(e.value))


def test_leaves_and_field_trees_bind_to_the_same_runs():
    t = Record(("lo", Array(2, V16)), ("hi", Array(2, V16)))
    by_field = P.bind_ports("T", (("r", t),), V16, Ports(params={"r": {"lo": In(bound=(0, 9)), "hi": In(bound=(0, 9))}}))
    by_tuple = P.bind_ports("T", (("r", t),), V16, Ports(params={"r": (In(bound=(0, 9)), In(bound=(0, 9)))}))
    by_leaves = P.bind_ports("T", (("r", t),), V16, Ports(params={"r": Leaves(((0, 4, In(bound=(0, 9))),))}))
    whole = P.bind_ports("T", (("r", t),), V16, Ports(params={"r": In(bound=(0, 9))}))
    assert by_field == by_tuple == by_leaves == whole
    assert whole.to_json() == {"params": [[0, 0, 4, {"bound": [0, 9]}]]}
