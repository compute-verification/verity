"""Acceptance of the torch frontend prototype (frontend design §5): (a) authored TA1
composites digest-equal TA1's registry definitions; (b) IFb's identity shapes translate with reference sharing
preserved; (c) negatives fail closed with codes. Needs torch (runs on the TA1 pod)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits import Program, bind, encode_program, program_digest, query  # noqa: E402
from verity_vllm.program.frontend import FrontendError, export, function, Arr, batch, ops  # noqa: E402
from verity_vllm.program.frontend import examples as ex  # noqa: E402

from tests.program.test_running_example import Add16, Id16, Inference, Mul16, Relu16, make_program  # noqa: E402

V16 = ex.Value(16)  # module level: string annotations are evaluated in the function's globals


@pytest.mark.parametrize("label,front,ta1", ex.ta1_pairs(), ids=[p[0] for p in ex.ta1_pairs()])
def test_authored_composites_digest_equal_ta1(label, front, ta1):
    P_front = Program(front.specialize())
    P_ta1 = Program(ta1)
    assert P_front.gates == P_ta1.gates
    assert program_digest(P_front) == program_digest(P_ta1), label
    d1, d2 = encode_program(P_front), encode_program(P_ta1)
    assert d1["definitions"] == d2["definitions"]


def test_ir_running_example_authored_in_torch_digest_equals():
    @function(name="Dot")
    def Dot(x: Arr[2, V16], w: Arr[2, V16]) -> V16:
        return ops.Add16(ops.Mul16(x[0], w[0]), ops.Mul16(x[1], w[1]))

    @function(name="Row", static=("act",))
    def Row(x: Arr[2, V16], w: Arr[2, V16], *, act) -> V16:
        return act(Dot(x, w))

    @function(name="Linear", static=("M", "act"))
    def Linear(x: Arr[2, V16], W: Arr["M", 2, V16], *, act) -> Arr["M", V16]:
        return batch(Row.bind(act=act), x, W, axes=(None, 0))

    @function(name="Inference", static=("B", "M", "act"))
    def Inf(X: Arr["B", 2, V16], W: Arr["M", 2, V16], *, act) -> Arr["B", "M", V16]:
        return batch(Linear.bind(act=act), X, W, axes=(0, None))

    P = export(Inf, static={"B": 2, "M": 5, "act": ops.Relu16})
    assert P.gates == 54 and P.input_gates == 14
    assert program_digest(P) == program_digest(make_program(act=Relu16))
    assert program_digest(P) != program_digest(make_program(act=Id16))


def test_identity_shapes_reference_sharing():
    Top, TopIR = ex.identity_pair(Add16, Mul16, Id16)
    P = export(Top, (torch.empty((), dtype=torch.bfloat16, device="meta"),))
    P_ir = Program(bind(TopIR))
    assert program_digest(P) == program_digest(P_ir)
    f = query(P.circuit, "//F")[0]
    assert f.args == (0, 0)  # repeated argument
    assert f.returns == (1, 0, 1, 2)  # t, the input v, t again, the dead declared return u
    assert f.in_ == (0,)
    assert f.distinct_returns == (1, 2)  # declared-interface adapter: distinct returned values produced inside (t, u)
    assert f.out == (1,)  # S.out is the graph boundary: u is declared but never read, so it is not on it (§8.6)


def test_module_weights_become_input_record():
    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4, 32, dtype=torch.bfloat16))
            self.b = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

        def forward(self, x):
            return ex.BiasAdd(ex.Gemm(x, self.w), self.b)

    P = export(Lin(), (torch.empty(32, dtype=torch.bfloat16, device="meta"),), name="Lin")
    assert P.input_gates == 32 + 128 + 4
    assert [c.name for c in P.circuit.root_calls()][-2:] == ["Gemm", "BiasAdd"]


def test_call_names_reach_function_calls():
    @function
    def Two(x: Arr[8, ex.BF16]) -> Arr[8, ex.BF16]:
        y = ex.BiasAdd(x, x, name="first")
        return ex.BiasAdd(y, x, name="second")

    P = export(Two, (torch.empty(8, dtype=torch.bfloat16, device="meta"),))
    assert [c.node.name for c in P.circuit.root_calls() if c.node.name] == ["first", "second"]


@pytest.mark.parametrize("label,fn,code", ex.NEGATIVES, ids=[n[2] + ":" + n[0][:24] for n in ex.NEGATIVES])
def test_negatives_fail_closed(label, fn, code):
    with pytest.raises(FrontendError) as ei:
        export(fn, ex.NEG_INPUTS)
    assert ei.value.code == code, str(ei.value)


def test_erased_binding_collision_is_an_error():
    @function
    def Rowish(x: Arr[8, ex.BF16], *, act) -> Arr[8, ex.BF16]:
        return batch(act, x, x, axes=(0, 0))

    xs = (torch.empty(8, dtype=torch.bfloat16, device="meta"),)
    export(Rowish, xs, static={"act": ops.Bf16Add})
    with pytest.raises(FrontendError) as ei:
        export(Rowish, xs, static={"act": ops.Bf16AddF2fp})
    assert ei.value.code == "E_IDENTITY"


def test_unresolved_static_and_arity():
    with pytest.raises(FrontendError) as ei:
        ex.Gemm.specialize(K=32)
    assert ei.value.code == "E_STATIC"

    @function
    def Bad(x: Arr[8, ex.BF16]) -> Arr[8, ex.BF16]:
        return ex.BiasAdd(x)

    with pytest.raises(FrontendError) as ei:
        export(Bad, (torch.empty(8, dtype=torch.bfloat16, device="meta"),))
    assert ei.value.code == "E_ARITY"
