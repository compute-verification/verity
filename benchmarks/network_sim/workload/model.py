"""The workload's circuit: a tiny integer MLP served to a batch of requests, as a Boolean Program.

Every layer is `d` wide. Activations are unsigned `act_bits`-bit words, weights signed `weight_bits`-bit words and biases
signed `bias_bits`-bit words. Output row j of a layer is

    y_j = clamp((b_j + sum_i w_ji * x_i) >> shift, 0, 2^act_bits - 1)

that is, a ReLU and a requantization. The weights and biases are registered inputs (what the program brings), the requests'
activations are run inputs, and the Program holds only gates (`verity/ml/circuits/PROTOCOL.md` §5.3).

One root Call computes `d / groups` rows of one layer for one request (`dense_rows`, a Boolean Definition traced from a
`forms` builder, so its gates are `And2`, `Xor2` and `Not`). The root body lists the Calls round by round, request r's
layer l in round r + l: the pipeline schedule, so gate order follows time. `CallInfo` records each Call's
request, layer and group, which is what a placement assigns to a device.

`reference` computes the same MLP on integers, the oracle the circuit's evaluation is checked against.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cached_property

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import forms as F
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.reference import evaluate_region
from verity.ml.circuits.layout import Circuit, Gate
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array

BIT = BL.BIT


@dataclass(frozen=True)
class Spec:
    d: int = 4
    layers: int = 3
    requests: int = 2
    groups: int = 1
    act_bits: int = 8
    weight_bits: int = 4
    bias_bits: int = 8
    shift: int = 4

    def __post_init__(self) -> None:
        if self.d % self.groups:
            raise ValueError(f"{self.groups} groups don't divide a layer of {self.d} rows")
        if not 0 <= self.shift < self.acc_bits - 1:
            raise ValueError(f"a shift of {self.shift} leaves no bits of a {self.acc_bits}-bit accumulator")

    @property
    def rows(self) -> int:
        return self.d // self.groups

    @property
    def acc_bits(self) -> int:
        """Enough two's-complement bits that no sum overflows."""
        top = self.d * ((1 << self.act_bits) - 1) * (1 << (self.weight_bits - 1)) + (1 << (self.bias_bits - 1))
        return top.bit_length() + 1

    @property
    def steps(self) -> int:
        return self.requests + self.layers - 1

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class CallInfo:
    request: int
    layer: int
    group: int
    lo: int
    hi: int


def _signed_bits(bits: list, width: int) -> list:
    return list(bits) + [bits[-1]] * (width - len(bits))


def _dense_rows_build(spec: Spec):
    d, rows, ab, wb, acc, sh = spec.d, spec.rows, spec.act_bits, spec.weight_bits, spec.acc_bits, spec.shift

    def build(C, *els):
        x, w, b = els[:d], els[d:d + rows * d], els[d + rows * d:]
        out = []
        for j in range(rows):
            cols: list[list] = [[] for _ in range(acc)]
            for k, bit in enumerate(_signed_bits(b[j], acc)):
                cols[k].append(bit)
            for i in range(d):
                wi = w[j * d + i]
                for k in range(ab):
                    for m in range(wb - 1):
                        cols[k + m].append(C.AND(x[i][k], wi[m]))
                # -(x * w_top) << (wb - 1) as its two's complement: the complement of the shifted product, plus one
                top = [F.ZERO] * acc
                for k in range(ab):
                    if k + wb - 1 < acc:
                        top[k + wb - 1] = C.AND(x[i][k], wi[wb - 1])
                for k in range(acc):
                    cols[k].append(F.NOT(top[k]))
                cols[0].append(F.ONE)
            s = F.compress_columns(C, cols)[:acc]
            negative = s[acc - 1]
            over = F.or_reduce(C, s[sh + ab:acc - 1])
            out.append([C.AND(F.NOT(negative), F.OR(C, q, over)) for q in s[sh:sh + ab]])
        return out

    return build


def dense_rows(spec: Spec) -> CompositeDefinition:
    """`rows` output rows of one layer, as a Boolean Definition."""
    act, wt, bias = Array(spec.act_bits, BIT), Array(spec.weight_bits, BIT), Array(spec.bias_bits, BIT)
    name = f"NetsimDenseD{spec.d}R{spec.rows}A{spec.act_bits}W{spec.weight_bits}B{spec.bias_bits}S{spec.shift}"
    params = (("x", Array(spec.d, act)), ("w", Array(spec.rows, Array(spec.d, wt))), ("b", Array(spec.rows, bias)))
    return TR.definition(name, 1, _dense_rows_build(spec), params, Array(spec.rows, act), register=False,
                         doc="rows of a ReLU-requantized integer dense layer")


def _cat(t, cs) -> Coll:
    return Coll(t, concat([c.refs for c in cs]))


def program_function(spec: Spec) -> CompositeDefinition:
    act = Array(spec.act_bits, BIT)
    vec = Array(spec.d, act)
    wmat = Array(spec.d, Array(spec.d, Array(spec.weight_bits, BIT)))
    bvec = Array(spec.d, Array(spec.bias_bits, BIT))
    params = [("x", Array(spec.requests, vec))]
    for layer in range(spec.layers):
        params += [(f"w{layer}", wmat), (f"b{layer}", bvec)]
    ret = Array(spec.requests, vec)
    rows = dense_rows(spec)

    def body(B, S, x, *wb):
        h = {(r, 0): x[r] for r in range(spec.requests)}
        for step in range(spec.steps):
            for r in range(spec.requests):
                layer = step - r
                if not 0 <= layer < spec.layers:
                    continue
                w, b = wb[2 * layer], wb[2 * layer + 1]
                parts = []
                for g in range(spec.groups):
                    sel = range(g * spec.rows, (g + 1) * spec.rows)
                    parts.append(B.call(rows, h[(r, layer)], _cat(Array(spec.rows, w.type.elem), [w[j] for j in sel]),
                                        _cat(Array(spec.rows, b.type.elem), [b[j] for j in sel])))
                h[(r, layer + 1)] = _cat(vec, parts)
        return _cat(ret, [h[(r, spec.layers)] for r in range(spec.requests)])

    name = (f"NetsimMlpD{spec.d}L{spec.layers}R{spec.requests}G{spec.groups}"
            f"A{spec.act_bits}W{spec.weight_bits}B{spec.bias_bits}S{spec.shift}")
    return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False,
                               doc="a tiny integer MLP over a batch of requests, pipelined by engine step")


class Model:
    """The Program of a `Spec`, its Calls in body order, and its gate table (each gate's primitive and operands)."""

    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        fn = bind(program_function(spec))
        self.params = tuple(n for n, _ in fn.signature()[0])
        self.program = Program(fn, registered=tuple(n for n in self.params if n != "x"))
        self.circuit: Circuit = self.program.circuit
        calls = list(self.circuit.root_calls())
        order = [(r, step - r, g) for step in range(spec.steps) for r in range(spec.requests)
                 if 0 <= step - r < spec.layers for g in range(spec.groups)]
        if len(calls) != len(order):
            raise AssertionError(f"{len(calls)} root Calls, the schedule has {len(order)}")
        self.calls = [CallInfo(r, layer, g, c.lo, c.hi) for (r, layer, g), c in zip(order, calls)]

    def step(self, c: CallInfo) -> int:
        """The engine step of a Call: request r's layer l at 1 + r + l."""
        return 1 + c.request + c.layer

    @property
    def n(self) -> int:
        return self.circuit.size

    value_bits = 1

    @cached_property
    def gates(self) -> list[Gate]:
        return list(self.circuit.region(0, self.n))

    @cached_property
    def ops(self) -> list[tuple[int, ...]]:
        """Each gate's distinct operands, ascending: `Wiring.ops`."""
        return [tuple(sorted(set(g.operands))) for g in self.gates]

    @cached_property
    def width(self) -> list[int]:
        """Each gate's width in bits: `Wiring.width` (1 throughout a Boolean Program)."""
        return [g.prim.ret.w for g in self.gates]

    @cached_property
    def kind(self) -> list[str]:
        """Each gate's kind: `run` and `registered` inputs, `constant` (an operation's own bits) and `computed`."""
        out = []
        for g in self.gates:
            if is_input(g.prim):
                out.append("registered" if g.prim.id.startswith("Registered") else "run")
            elif not g.operands:
                out.append("constant")
            else:
                out.append("computed")
        return out

    @cached_property
    def outputs(self) -> tuple[int, ...]:
        return self.circuit.root_outputs()

    def input_gates(self, pi: int) -> list[int]:
        """The input gates of root parameter `pi` (0 is `x`, then w0, b0, w1, ...), in leaf order."""
        return [ref[2] for ref in self.program.input_coll(pi).refs]

    def bind_inputs(self, x: list[list[int]], weights: list[tuple[list[list[int]], list[int]]]) -> dict[int, int]:
        """The input gates' values: `x[r][i]` and each layer's `(w[j][i], b[j])`, two's complement, bit by bit."""
        spec = self.spec
        t: dict[int, int] = {}

        def put(pi: int, words: list[int], width: int) -> None:
            gates = self.input_gates(pi)
            if len(gates) != len(words) * width:
                raise ValueError(f"parameter {pi} has {len(gates)} bits, given {len(words)} words of {width}")
            for k, wd in enumerate(words):
                for j in range(width):
                    t[gates[k * width + j]] = (wd >> j) & 1

        put(0, [v for row in x for v in row], spec.act_bits)
        for layer, (w, b) in enumerate(weights):
            put(1 + 2 * layer, [v & ((1 << spec.weight_bits) - 1) for row in w for v in row], spec.weight_bits)
            put(2 + 2 * layer, [v & ((1 << spec.bias_bits) - 1) for v in b], spec.bias_bits)
        return t

    def inputs(self, seed: int) -> dict[int, int]:
        return self.bind_inputs(requests(self.spec, seed), weights(self.spec, seed))

    def constants(self) -> dict[int, int]:
        return self.evaluate({}, [g for g, k in enumerate(self.kind) if k == "constant"])

    def reference(self, seed: int) -> list[list[int]]:
        return reference(self.spec, requests(self.spec, seed), weights(self.spec, seed))

    def evaluate(self, transcript: dict[int, int], gates: list[int] | None = None) -> dict[int, int]:
        """Evaluate `gates` (default: every gate) in order into `transcript`, with verity's evaluator."""
        sel = self.gates if gates is None else [self.gates[g] for g in gates]
        return evaluate_region(self.circuit, sel, transcript)

    def output_words(self, transcript: dict[int, int]) -> list[list[int]]:
        spec, out = self.spec, self.outputs
        words = [sum(transcript[out[k * spec.act_bits + j]] << j for j in range(spec.act_bits))
                 for k in range(len(out) // spec.act_bits)]
        return [words[r * spec.d:(r + 1) * spec.d] for r in range(spec.requests)]


def weights(spec: Spec, seed: int) -> list[tuple[list[list[int]], list[int]]]:
    rng = random.Random(f"verity/netsim/weights/{seed}")
    lo_w, hi_w = -(1 << (spec.weight_bits - 1)), (1 << (spec.weight_bits - 1)) - 1
    lo_b, hi_b = -(1 << (spec.bias_bits - 1)), (1 << (spec.bias_bits - 1)) - 1
    return [([[rng.randint(lo_w, hi_w) for _ in range(spec.d)] for _ in range(spec.d)],
             [rng.randint(lo_b, hi_b) for _ in range(spec.d)]) for _ in range(spec.layers)]


def requests(spec: Spec, seed: int) -> list[list[int]]:
    rng = random.Random(f"verity/netsim/requests/{seed}")
    return [[rng.randrange(1 << spec.act_bits) for _ in range(spec.d)] for _ in range(spec.requests)]


def reference(spec: Spec, x: list[list[int]], ws: list[tuple[list[list[int]], list[int]]]) -> list[list[int]]:
    top = (1 << spec.act_bits) - 1
    out = []
    for h in x:
        for w, b in ws:
            h = [min(max((b[j] + sum(w[j][i] * h[i] for i in range(spec.d))) >> spec.shift, 0), top)
                 for j in range(spec.d)]
        out.append(h)
    return out


WORKLOADS = ("mlp", "served")


def build(workload: str, spec: dict) -> Model:
    """The workload's model from its spec's JSON: the MLP (`Model`), or verity_vllm's serving Program (`served.Served`)."""
    if workload == "mlp":
        return Model(Spec(**spec))
    if workload == "served":
        from . import served
        return served.load(served.Config(**spec))
    raise ValueError(f"a workload is one of {WORKLOADS}, not {workload!r}")
