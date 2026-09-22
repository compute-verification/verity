"""The CORE circuit ``DemoCore_v1{N,R}`` (generic protocol demonstration; NOT vLLM inference) and its fixed partition.

Circuit (a registered ``verity.ir`` composite; every gate is one of two approved kinds):

    inputs  a, b, c : Array<N, BF16>            (3·N prescribed Input<16> gates = the typed input x)
    h_0 := a
    stage r (r = 0..R-1), ONE replay unit each:
        g_r = batch(SiluMulBf16, h_{r-1}, b)    N gates, kind SiluMulBf16_v1|batch   -- interior: NOT an RU output
        h_r = batch(Bf16AddF2fp, g_r, c)        N gates, kind Bf16AddF2fp_v1|batch   -- the RU's output tuple
    output  y := h_R : Array<N, BF16>           (the designated circuit output; W_out = 16·N bits)

Partition: S_V = one VU per gate (2·N VUs per stage, 2·N·R in all); S_R = one RU per stage call (R RUs). The VU quotient is acyclic
(each stage reads only the previous stage's output h_{r-1} and the inputs). Stage r ≥ 1 CONSUMES RU r−1's committed output
(``h_{r-1}``): a computed value crossing RUs, committed at step 1 as RU r−1's output and read by RU r's SiluMul VUs.

Identities: ``program_digest`` = ``verity.ir.codec.program_digest(encode_program(Program(bind(DemoCore, N=, R=))))``;
``partition_digest`` = sha256 of the canonical JSON of :func:`partition` (kind ``demo-core/stage-ru-gate-vu/v1``).
Gate identity = gate index in the program; canonical id = ``Circuit.gate(g).path`` joined with ``/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
import sys  # noqa: E402

for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "vllm-poc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verity.ir import Array, Program, Value, bind, composite, encode_program, program_digest  # noqa: E402
from verity_vllm.registry import prims as P  # noqa: E402

__all__ = ["BF16", "DemoCore", "DemoCoreStage", "PARTITION_KIND", "Partition", "VU", "RU", "build", "partition", "identity", "KIND_SILU", "KIND_ADD"]

BF16 = Value(16)
PARTITION_KIND = "demo-core/stage-ru-gate-vu/v1"
KIND_SILU = "SiluMulBf16_v1|batch"
KIND_ADD = "Bf16AddF2fp_v1|batch"


@composite("DemoCoreStage", 1, ["N"],
           lambda S: ((("h", Array(S.N, BF16)), ("b", Array(S.N, BF16)), ("c", Array(S.N, BF16))), Array(S.N, BF16)),
           doc="one CORE stage: g = SiluMulBf16(h, b) (interior), h' = Bf16AddF2fp(g, c) (the stage output)",
           conformance="exact-model-tested (both primitives are registered exact-model-tested; the composite adds no numerics)")
def DemoCoreStage(B, S, h, b, c):
    g = B.batch(P.SiluMulBf16, h, b, axes=(0, 0), name="silu_mul")
    return B.batch(P.Bf16AddF2fp, g, c, axes=(0, 0), name="add")


@composite("DemoCore", 1, ["N", "R"],
           lambda S: ((("a", Array(S.N, BF16)), ("b", Array(S.N, BF16)), ("c", Array(S.N, BF16))), Array(S.N, BF16)),
           doc="CORE protocol demonstrator: R chained DemoCoreStage calls over BF16 words (generic protocol demonstration, not vLLM inference)",
           conformance="exact-model-tested (composition of exact-model-tested primitives)")
def DemoCore(B, S, a, b, c):
    h = a
    stage = bind(DemoCoreStage, N=S.N)
    for r in range(S.R):
        h = B.call(stage, h, b, c, name=f"stage{r}")
    return h


@dataclass(frozen=True)
class VU:
    vu_index: int          # 0-based over the whole program (RU-major, gate order within the RU)
    ru_index: int
    gate: int              # the one gate of this VU (root-scoped index)
    kind: str              # fn|form
    canonical_id: str      # Circuit.gate(g).path joined by '/'
    operands: tuple[int, ...]
    ru_output: bool        # True when the gate is in the RU's declared output tuple (committed at step 1)

    def to_json(self) -> dict[str, Any]:
        return {"vu_index": self.vu_index, "ru_index": self.ru_index, "gate": self.gate, "gate_interval": [self.gate, self.gate + 1], "kind": self.kind,
                "canonical_id": self.canonical_id, "operands": list(self.operands), "ru_output": self.ru_output}


@dataclass(frozen=True)
class RU:
    ru_index: int
    name: str              # call name (stage<r>)
    lo: int
    hi: int
    args: tuple[int, ...]      # declared interface: the gates the call reads (h_{r-1}: N gates, b: N gates, c: N gates)
    returns: tuple[int, ...]   # declared output tuple (N gates)
    vus: tuple[VU, ...]
    call_id: int = -1          # root node index of the stage call (E2 rule item 2's call id; E3 GenericLocator.calls()); NOT in to_json (digest-neutral)

    def to_json(self) -> dict[str, Any]:
        return {"ru_index": self.ru_index, "name": self.name, "gate_interval": [self.lo, self.hi], "args": list(self.args), "returns": list(self.returns),
                "vu_indices": [v.vu_index for v in self.vus]}


@dataclass(frozen=True)
class Partition:
    N: int
    R: int
    program_digest: str
    gates: int
    input_gates: int
    inputs: dict[str, tuple[int, ...]]   # a / b / c -> Input gate indices
    outputs: tuple[int, ...]             # designated circuit output gates (root returns)
    rus: tuple[RU, ...]

    @property
    def vus(self) -> tuple[VU, ...]:
        return tuple(v for r in self.rus for v in r.vus)

    @property
    def n_v(self) -> int:
        return len(self.vus)

    @property
    def n_r(self) -> int:
        return len(self.rus)

    def vu(self, i: int) -> VU:
        return self.vus[i]

    def kind_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.vus:
            out[v.kind] = out.get(v.kind, 0) + 1
        return dict(sorted(out.items()))

    def to_json(self) -> dict[str, Any]:
        return {"schema": "veritor/demo-core/partition/v1", "kind": PARTITION_KIND, "program": identity(self.N, self.R), "program_digest": self.program_digest,
                "gates": self.gates, "input_gates": self.input_gates, "inputs": {k: list(v) for k, v in self.inputs.items()}, "outputs": list(self.outputs),
                "N_V": self.n_v, "N_R": self.n_r, "kinds": self.kind_counts(), "rus": [r.to_json() for r in self.rus], "vus": [v.to_json() for v in self.vus],
                "rule": "S_V = one VU per computed gate; S_R = one RU per stage call; VU index RU-major then gate order; interior = VU output gate not in RU returns"}

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(N: int, R: int) -> dict[str, Any]:
    sp = bind(DemoCore, N=int(N), R=int(R))
    return {"function": sp.id, "name": "DemoCore", "version": 1, "N": int(N), "R": int(R), "label": "generic protocol demonstration (NOT vLLM inference)"}


def build(N: int, R: int) -> tuple[Program, str]:
    """The program and its ``program_digest``.  [INTEGRATOR D1] This is the callable ``protocol_instance.json#program.builder`` names
    (``module = e2e.demo_core.circuit``, ``symbol = build``, ``params`` / ``kwargs = {N, R}``): the verifier calls it in its own process and
    checks the returned digest against the pin (fail-closed)."""
    sp = bind(DemoCore, N=int(N), R=int(R))
    prog = Program(sp)
    return prog, program_digest(encode_program(prog))


def position_adapter(program, locator, C):
    """[INTEGRATOR D1, E3's contract] ``fn(program, locator, C) -> {row_gates, weight_word, prefix, ...}`` -- E3's ``locator.position_adapter``
    looks this symbol up in the builder module.  Implemented in :mod:`e2e.demo_core.adapter` (imports the record's deterministic leaf ids)."""
    from .adapter import position_adapter as _pa

    return _pa(program, locator, C)


def _kind_of(gate) -> str:
    prim = gate.prim
    name = getattr(prim, "name", None) or prim.id.split("_v")[0]
    version = getattr(prim, "version", 1)
    return f"{name}_v{version}|batch"


def partition(N: int, R: int, program: Program | None = None, digest: str | None = None) -> Partition:
    if program is None or digest is None:
        program, digest = build(N, R)
    circuit = program.circuit
    calls = [c for c in circuit.root_calls() if c.fn.name == "DemoCoreStage"]
    if len(calls) != R:
        raise RuntimeError(f"expected {R} stage calls, found {len(calls)}")
    rus: list[RU] = []
    vu_index = 0
    for r, call in enumerate(sorted(calls, key=lambda c: c.lo)):
        returns = tuple(int(g) for g in call.returns)
        ret_set = set(returns)
        vus: list[VU] = []
        for g in range(call.lo, call.hi):
            gate = circuit.gate(g)
            vus.append(VU(vu_index, r, g, _kind_of(gate), "/".join(gate.path), tuple(int(o) for o in gate.operands), g in ret_set))
            vu_index += 1
        rus.append(RU(r, str(call.node.name or f"stage{r}"), int(call.lo), int(call.hi), tuple(int(g) for g in call.args), returns, tuple(vus), int(call.node_idx)))
    ins = {}
    for pi, name in enumerate(("a", "b", "c")):
        coll = program.input_coll(pi)
        ins[name] = tuple(int(g) for _, _, g in coll.refs)
    outs = tuple(int(g) for g in circuit.root_outputs())
    return Partition(int(N), int(R), digest, int(program.gates), int(program.input_gates), ins, outs, tuple(rus))
