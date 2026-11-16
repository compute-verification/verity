"""Negative and adversarial tests for the automatic importer (the derive mandate's "Test generality and failure
behavior").

Contract under test (frontend design §4.3): `derive(...)` returns a complete `Program` or
`Unsupported(reason, node, module_path, hint)` (returned or raised) / a `FrontendError` code — never a partial
program, never an opaque gate, never a specialisation on observed values. Every negative here asserts BOTH that
no `Program` is returned AND that the refusal names the right class of reason. A refusal for the *wrong* reason
is reported distinctly (`REFUSED-WRONG-REASON`) from acceptance (`ACCEPTED-MUST-REFUSE`) and from a crash that is
not a classified refusal (`CRASH-NOT-REFUSAL`).

The importer (XFER may not edit importer code) is `verity_torch.frontend.derive.derive`, called through its
`derive_report`; the transfer report records its name. Needs torch; runs on the pod (§37).

The cache-ownership oracle and the capture-to-program correspondence checks, which read the vLLM adapter's served
Definitions, are the vLLM adapter's `tests/program/test_derive_negative.py`.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from verity.circuits import Program

from verity_torch.frontend import derive as _D
from verity_torch.frontend.errors import FrontendError

# ---------------------------------------------------------------------------------------------------------
# importer adapter: derive_report(...); one normalised Outcome
# ---------------------------------------------------------------------------------------------------------

IMPORTER_NAME = "verity_torch.frontend.derive.derive"


@dataclass
class Outcome:
    program: Program | None
    kind: str  # "program" | "unsupported" | "frontend_error" | "crash"
    reason: str  # Unsupported.reason / FrontendError.code / exception class
    detail: str
    exc: BaseException | None = None
    report: Any = None  # derive's report / provenance object when it returns one

    @property
    def refused(self) -> bool:
        return self.kind in ("unsupported", "frontend_error")

    def __str__(self) -> str:
        return f"{self.kind}:{self.reason} — {self.detail[:300]}"


def _reason_of(obj: Any) -> tuple[str, str]:
    for attr in ("reason", "code"):
        r = getattr(obj, attr, None)
        if isinstance(r, str):
            return r, str(obj)
    return type(obj).__name__, str(obj)


def _is_unsupported(obj: Any) -> bool:
    if isinstance(obj, _D.Unsupported):
        return True
    return type(obj).__name__ == "Unsupported" or (hasattr(obj, "reason") and not isinstance(obj, Program))


def _filtered_kwargs(fn: Callable, kw: dict) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def run_importer(module: torch.nn.Module, example_inputs: tuple, **kw) -> Outcome:
    """Call the importer on `module` with `example_inputs` and normalise what happened:
    `derive_report(module, inputs, mode="reference", ...)` (the report carries program|unsupported + provenance);
    `mode`/`profile`/`config` may be overridden per call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kw = _filtered_kwargs(_D.derive_report, {"mode": "reference", "name": type(module).__name__, **kw})
            res = _D.derive_report(module, example_inputs, **kw)
            prog = getattr(res, "program", None)
            if isinstance(prog, Program):
                return Outcome(prog, "program", "", f"Program {prog.fn.id} ({prog.gates} gates)", report=res)
            u = getattr(res, "unsupported", None)
            if u is not None:
                r, d = _reason_of(u)
                hint = getattr(u, "hint", "") or ""
                det = getattr(u, "detail", None)
                return Outcome(None, "unsupported", r, f"{d} | hint={hint} | detail={det}", report=res)
            return Outcome(None, "crash", type(res).__name__, f"derive_report returned neither program nor unsupported: {res!r:.200}")
        except FrontendError as e:
            return Outcome(None, "frontend_error", e.code, str(e), e)
        except Exception as e:  # noqa: BLE001
            if _is_unsupported(e):
                r, d = _reason_of(e)
                return Outcome(None, "unsupported", r, d, e)
            return Outcome(None, "crash", type(e).__name__, str(e).splitlines()[0][:400] if str(e) else "", e)


# reason classes: the documented vocabulary (design §3.3 codes, §4.3 reasons) plus the obvious synonyms;
# tightened against derive.py once it lands (XFER adjusts tests, never the importer)
# derive.py's REASONS (as landed): mutation, decomposition, custom_op_without_binding, side_input, dynamic_control_flow,
# numerics_unregistered, hidden_state, unsupported_view, shape, export, identity. Its `_no_rule` labels an ATen op with
# no rule `decomposition` ("has no derivation rule in <mode> mode") and `.item()`/`nonzero` `side_input` (boundary 2).
CLASSES = {
    "unknown_op": ("E_UNREGISTERED_OP", "numerics_unregistered", "custom_op_without_binding", "decomposition", "unregistered",
                   "no_rule", "no rule", "no derivation rule", "unsupported_op", "unknown_op", "not registered"),
    "hidden_state": ("E_HIDDEN_STATE", "hidden_state", "hidden state", "lifted", "constant"),
    "mutation": ("E_MUTATION", "mutation", "in-place", "inplace", "mutat"),
    "shape": ("E_SHAPE", "E_ARITY", "E_EXPORT", "E_STATIC", "shape", "export", "static", "rank", "size mismatch", "decomposition"),
    "width": ("E_WIDTH", "width", "dtype", "numerics_unregistered", "E_UNREGISTERED_OP", "decomposition", "no derivation rule",
              "no_rule", "unregistered", "kind", "shape"),
    "control_flow": ("E_DYNAMIC_CONTROL_FLOW", "E_OPAQUE_HOST", "dynamic_control_flow", "side_input", "data_dependent", "data-dependent",
                     "control_flow", "control flow", "opaque", "host", "item", "cond", "while", "E_EXPORT", "export",
                     "GuardOnDataDependent", "specializ", "not fixed"),
    "side_input": ("side_input", "E_HIDDEN_STATE", "hidden_state", "cache", "E_MUTATION", "mutation", "E_UNREGISTERED_OP",
                   "numerics_unregistered", "custom_op_without_binding", "decomposition", "unregistered", "no_rule"),
}


def _in_class(reason: str, detail: str, cls: str) -> bool:
    hay = f"{reason} {detail}".lower()
    return any(tok.lower() in hay for tok in CLASSES[cls])


def assert_refused(out: Outcome, cls: str, what: str) -> None:
    """The three-way verdict: accepted (hard failure), refused for the wrong reason, or a crash."""
    if out.kind == "program":
        pytest.fail(f"ACCEPTED-MUST-REFUSE [{cls}] {what}: importer {IMPORTER_NAME} returned {out.detail}")
    if out.kind == "crash":
        pytest.fail(f"CRASH-NOT-REFUSAL [{cls}] {what}: {out.reason}: {out.detail}")
    assert out.program is None
    if not _in_class(out.reason, out.detail, cls):
        pytest.fail(f"REFUSED-WRONG-REASON [{cls}] {what}: got {out.kind}:{out.reason} — {out.detail[:300]}")


def meta(*shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device="meta")


def _f32(*shape):
    return torch.nn.Parameter(torch.randn(*shape) * 0.1)


# ---------------------------------------------------------------------------------------------------------
# the base module every negative perturbs: an ordinary two-layer residual MLP (no Verity annotations)
# ---------------------------------------------------------------------------------------------------------

H, I_ = 8, 16


class ResMLP(torch.nn.Module):
    def __init__(self, h=H, i=I_, dtype=torch.float32):
        super().__init__()
        self.fc = torch.nn.Linear(h, i, dtype=dtype)
        self.proj = torch.nn.Linear(i, h, dtype=dtype)

    def forward(self, x):
        return x + self.proj(torch.relu(self.fc(x)))


# ---------------------------------------------------------------------------------------------------------
# 1. unknown / unregistered operations
# ---------------------------------------------------------------------------------------------------------


class UnknownAten(ResMLP):
    def forward(self, x):
        return torch.special.i0(x + self.proj(torch.relu(self.fc(x))))  # modified Bessel: no importer has a kind for it


_foreign = torch.library.Library("xfer_foreign", "FRAGMENT")
_foreign.define("blackbox(Tensor a) -> Tensor")
torch.library.register_fake("xfer_foreign::blackbox", lambda a: torch.empty_like(a))


class ForeignCustomOp(ResMLP):
    def forward(self, x):
        return torch.ops.xfer_foreign.blackbox(x + self.proj(torch.relu(self.fc(x))))


class UnknownReduction(ResMLP):
    def forward(self, x):
        y = x + self.proj(torch.relu(self.fc(x)))
        return torch.cumsum(y, dim=0) * torch.logsumexp(y, dim=0)  # two ATen ops with no registered numerical model


@pytest.mark.parametrize("cls", [UnknownAten, ForeignCustomOp, UnknownReduction])
def test_unknown_operation_is_refused(cls):
    out = run_importer(cls(), (meta(H),))
    assert_refused(out, "unknown_op", cls.__name__)


# ---------------------------------------------------------------------------------------------------------
# 2. hidden state: closed-over tensors, lifted constants, tensors that are not declared parameters/buffers
# ---------------------------------------------------------------------------------------------------------

# The hidden tensors live on the module's parameter device (CPU here -- `derive._export` traces on the parameters'
# device, the shipped exporter on meta after `.to("meta")`) so that the only defect is the hidden state itself, not
# a device mismatch. An importer that traces on real values would see exactly the same lifted constants.
_GLOBAL_SCALE = torch.full((H,), 0.5)


class ClosedOverGlobal(ResMLP):
    def forward(self, x):
        return (x + self.proj(torch.relu(self.fc(x)))) * _GLOBAL_SCALE.to(x.device)  # module-level tensor, not an input


class PlainAttributeTensor(ResMLP):
    def __init__(self):
        super().__init__()
        self.shift = torch.ones(H) * 0.25  # neither Parameter nor registered buffer: invisible to named_parameters/buffers

    def forward(self, x):
        return x + self.proj(torch.relu(self.fc(x))) + self.shift


class LiftedNumpyConstant(ResMLP):
    def forward(self, x):
        k = torch.from_numpy(np.linspace(0, 1, H, dtype=np.float32)).to(x.device)  # a constant smuggled from the host
        return x + self.proj(torch.relu(self.fc(x))) + k


class HiddenTensorInList(ResMLP):
    def __init__(self):
        super().__init__()
        self.extras = [torch.ones(H) * 0.125]  # a tensor in a Python container: not a buffer

    def forward(self, x):
        return x + self.proj(torch.relu(self.fc(x))) + self.extras[0]


class ClosedOverCpuTensorMovedToDevice(ResMLP):
    """The realistic form: a CPU constant moved to the input's device inside forward (`.to(x.device)`)."""
    table = torch.linspace(0, 1, H)

    def forward(self, x):
        return x + self.proj(torch.relu(self.fc(x))) + self.table.to(x.device)


@pytest.mark.parametrize("cls", [ClosedOverGlobal, PlainAttributeTensor, LiftedNumpyConstant, HiddenTensorInList,
                                 ClosedOverCpuTensorMovedToDevice])
def test_hidden_state_is_refused(cls):
    out = run_importer(cls(), (meta(H),))
    assert_refused(out, "hidden_state", cls.__name__)


# ---------------------------------------------------------------------------------------------------------
# 3. incorrect shapes, numerical bindings (wrong dtype into a kind), mutation without a rule
# ---------------------------------------------------------------------------------------------------------


def test_shape_mismatch_input_vs_weight_is_refused():
    out = run_importer(ResMLP(), (meta(H + 1),))  # x has 9 features, fc wants 8
    assert_refused(out, "shape", "ResMLP with a 9-wide input into Linear(8, 16)")


class InputNamedWeights(ResMLP):
    def forward(self, weights):
        return super().forward(weights)


def test_an_input_named_as_the_weights_record_is_refused():
    """The derived function's parameters are the inputs by name and then the module's record, `weights`: an input of that name
    would make two parameters of one name, which every decoder of a Definition refuses."""
    out = run_importer(InputNamedWeights(), (meta(H),))
    assert_refused(out, "shape", "an input named `weights` beside the weights record")
    assert "input 'weights' has the name of a parameter this function gets beside its inputs (weights)" in out.detail, out


def test_rank_mismatch_is_refused_or_batched_exactly():
    """A [3, H] input into a module derived for [H]: either refused (E_SHAPE) or derived as 3 dataflow-independent
    rows — never a Program whose gate count equals the single-row program (silent row drop)."""
    single = run_importer(ResMLP(), (meta(H),))
    if single.kind != "program":
        pytest.skip(f"importer does not derive ResMLP ({single}); nothing to compare")
    out = run_importer(ResMLP(), (meta(3, H),))
    if out.kind == "program":
        assert out.program is not None and single.program is not None
        assert out.program.gates > single.program.gates, "a [3,H] input derived to the same gate count as [H]: rows were dropped"
        assert out.program.input_gates == single.program.input_gates + 2 * H
    else:
        assert_refused(out, "shape", "rank-2 input")


def _prim_ids(P: Program) -> set[str]:
    ids: set[str] = set()

    def visit(fn, seen):
        if fn in seen:
            return
        seen.add(fn)
        body = getattr(fn, "body", None)
        if body is None:
            ids.add(fn.id)
            return
        for n in body.nodes:
            visit(n.fn, seen)

    visit(P.root, set())
    return ids


def test_fp16_is_not_bound_to_a_bf16_kind():
    """float16 and bfloat16 both carry Value<16>: a width check alone would bind fp16 tensors into bf16 kinds.
    The importer must refuse fp16 (no fp16 kinds registered) or derive it to kinds disjoint from the bf16 ones."""
    bf = run_importer(ResMLP(dtype=torch.bfloat16), (meta(H, dtype=torch.bfloat16),))
    fp = run_importer(ResMLP(dtype=torch.float16), (meta(H, dtype=torch.float16),))
    if fp.kind == "program":
        assert bf.kind == "program", f"fp16 accepted while bf16 refused ({bf}); no fp16 kinds are registered"
        common = _prim_ids(fp.program) & _prim_ids(bf.program) - {p for p in _prim_ids(bf.program) if p.startswith("Input") or p.startswith("Const")}
        assert not common, f"fp16 and bf16 programs share numerical kinds {sorted(common)[:8]}: fp16 values bound into bf16 kinds"
    else:
        assert_refused(fp, "width", "fp16 ResMLP")


def test_int8_weights_are_not_bound_to_a_float_kind():
    class Int8Weights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randint(-3, 3, (I_, H), dtype=torch.int8), requires_grad=False)

        def forward(self, x):
            return torch.nn.functional.linear(x, self.w.to(x.dtype))  # dequant-by-cast: a numerical step no kind models

    out = run_importer(Int8Weights(), (meta(H),))
    assert_refused(out, "width", "int8 weight cast into a float linear")


def test_float64_is_refused_or_gets_its_own_kinds():
    f64 = run_importer(ResMLP(dtype=torch.float64), (meta(H, dtype=torch.float64),))
    f32 = run_importer(ResMLP(), (meta(H),))
    if f64.kind == "program":
        assert f32.kind == "program"
        assert not (_prim_ids(f64.program) & _prim_ids(f32.program)) - {"Input64", "Input32"}, "f64 values bound into f32 kinds"
    else:
        assert_refused(f64, "width", "float64 ResMLP")


class InPlaceOnInput(ResMLP):
    def forward(self, x):
        x.add_(self.proj(torch.relu(self.fc(x))))  # mutates a prescribed input
        return x


class InPlaceAliasReturn(ResMLP):
    def forward(self, x):
        y = x.view(H)  # an alias of the input ...
        y.mul_(2.0)  # ... mutated in place: the output aliases a mutated input
        return y + self.proj(torch.relu(self.fc(x)))


class BufferMutation(ResMLP):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.zeros(1))

    def forward(self, x):
        self.calls.add_(1.0)  # host-visible state change without a StateRule
        return x + self.proj(torch.relu(self.fc(x))) + self.calls


class IndexPutOnInput(ResMLP):
    def forward(self, x):
        x[0] = x[1]  # select + copy_ into a prescribed input (tensor-valued, so no lifted scalar constant muddies the reason)
        return x + self.proj(torch.relu(self.fc(x)))


class RunningStatsBuffer(torch.nn.Module):
    """BatchNorm in training mode updates running statistics: a buffer mutation that is a real, common source
    of silent state in exported models."""

    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(H)
        self.fc = torch.nn.Linear(H, H)
        self.train()

    def forward(self, x):
        return self.fc(self.bn(x))


@pytest.mark.parametrize("cls,inp", [(InPlaceOnInput, (meta(H),)), (InPlaceAliasReturn, (meta(H),)), (BufferMutation, (meta(H),)),
                                     (IndexPutOnInput, (meta(H),)), (RunningStatsBuffer, (meta(4, H),))])
def test_mutation_without_rule_is_refused(cls, inp):
    out = run_importer(cls(), inp)
    assert_refused(out, "mutation", cls.__name__)


# ---------------------------------------------------------------------------------------------------------
# 4. lost sharing and wrong cache ownership
# ---------------------------------------------------------------------------------------------------------

V = 16


class TiedEmbeddingLM(torch.nn.Module):
    """Tied input/output embedding (GPT-2 / SmolLM2 / Qwen2 style). Naive export would list the table twice."""

    def __init__(self, v=V, h=H):
        super().__init__()
        self.wte = torch.nn.Embedding(v, h)
        self.body = torch.nn.Linear(h, h, bias=False)
        self.lm_head = torch.nn.Linear(h, v, bias=False)
        self.lm_head.weight = self.wte.weight  # tie

    def forward(self, tok):
        return self.lm_head(torch.relu(self.body(self.wte(tok))))


class SharedBlockTwice(torch.nn.Module):
    """One block applied twice (weight sharing by module reuse: ALBERT / universal-transformer style)."""

    def __init__(self, h=H):
        super().__init__()
        self.block = torch.nn.Linear(h, h)

    def forward(self, x):
        return self.block(torch.relu(self.block(x)))


def _input_gate_ranges(P: Program) -> list[tuple[int, int]]:
    out = []
    for n_idx, n in enumerate(P.root.body.nodes):
        if getattr(n.fn, "family", None) == "Input":
            lo = P.root.body.offsets[n_idx]
            out.append((lo, lo + n.gates))
    return out


def _consumers_of_range(P: Program, lo: int, hi: int) -> set[int]:
    """Root gate indices (outside [lo,hi)) whose operands include a gate of [lo,hi)."""
    C = P.circuit
    hits: set[int] = set()
    for g in range(hi, C.size):
        if any(lo <= o < hi for o in C.gate(g).operands):
            hits.add(g)
    return hits


def test_tied_parameter_stays_one_input():
    m = TiedEmbeddingLM()
    n_distinct = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    n_naive = sum(p.numel() for _, p in m.named_parameters(remove_duplicate=False))
    assert n_naive == n_distinct + V * H  # the module genuinely has a duplicated name
    out = run_importer(m, (meta(dtype=torch.int64),))
    if out.kind != "program":
        pytest.skip(f"importer does not derive TiedEmbeddingLM yet ({out}); sharing cannot be checked. Filed as a gap.")
    P = out.program
    assert P is not None
    assert P.input_gates == n_distinct + 1, (f"input gates {P.input_gates} != distinct parameter leaves {n_distinct} + 1 token: "
                                             f"tied table exported as two inputs ({n_naive + 1}) or a leaf lost")
    # the V*H table gates feed two distinct root-level consumers: the embedding gather and the LM-head gemm
    lo, hi = _weight_leaf_range(out, P, "wte.weight", V * H)
    C = P.circuit
    consumers = _consumers_of_range(P, lo, hi)
    consumer_calls = {C.gate(g).scope.spec.id.split("{")[0] for g in consumers}
    assert len(consumers) > 0
    assert len(consumer_calls) >= 2 or len({C.gate(g).prim.id for g in consumers}) >= 2, \
        f"tied table read by one kind only ({consumer_calls}): the second use of the shared weight is missing"
    if out.report is not None and "weights" in out.report.provenance:
        w = [w for w in out.report.provenance["weights"] if w["name"] in ("wte.weight", "lm_head.weight")]
        assert len(w) == 1 and set(w[0]["aliases"]) == {"lm_head.weight"} - {w[0]["name"]}, w  # one leaf, the alias recorded


def _weight_leaf_range(out: Outcome, P: Program, name: str, numel: int) -> tuple[int, int]:
    """Gate range of the parameter leaf `name` among the Input gates: either its own Input batch of exactly `numel`
    gates, or its slice of a packed weights record located through the importer's `weights` provenance (fields in
    `field` order, sizes from `shape`)."""
    ranges = _input_gate_ranges(P)
    exact = [r for r in ranges if r[1] - r[0] == numel]
    if len(exact) == 1:
        return exact[0]
    prov = out.report.provenance if out.report is not None else {}
    leaves = sorted(prov.get("weights", []), key=lambda w: w["field"])
    sizes = [int(np.prod(w["shape"])) if w["shape"] else 1 for w in leaves]
    total = sum(sizes)
    packed = [r for r in ranges if r[1] - r[0] == total]
    assert len(packed) == 1, f"cannot locate the weights record (input ranges {ranges}, leaves total {total})"
    off = packed[0][0]
    for w, n in zip(leaves, sizes):
        if w["name"] == name or name in w.get("aliases", []):
            assert n == numel, (w, numel)
            return off, off + n
        off += n
    raise AssertionError(f"leaf {name!r} not in weights provenance {[w['name'] for w in leaves]}")


def test_shared_block_twice_is_one_weight_two_calls():
    m = SharedBlockTwice()
    out = run_importer(m, (meta(H),))
    if out.kind != "program":
        pytest.skip(f"importer does not derive SharedBlockTwice yet ({out}). Filed as a gap.")
    P = out.program
    assert P is not None
    assert P.input_gates == H + H * H + H, "the shared block's weights must appear once"
    # the weight leaves are whatever input gates are not the H activation gates (the importer may pack weight+bias
    # into one Input record); they must be read by gates in two disjoint root-call regions (two applications)
    ranges = _input_gate_ranges(P)
    assert sum(hi - lo for lo, hi in ranges) == H + H * H + H, ranges
    weight_ranges = [r for r in ranges if r[1] - r[0] != H] or [max(ranges, key=lambda r: r[1] - r[0])]
    consumers: set[int] = set()
    for lo, hi in weight_ranges:
        consumers |= _consumers_of_range(P, lo, hi)
    C = P.circuit
    roots = list(C.root_calls())
    regions = {next((k for k, c in enumerate(roots) if c.lo <= g < c.hi), None) for g in consumers}
    regions.discard(None)
    assert len(regions) >= 2, f"the shared block's weight is consumed in {len(regions)} root region(s); two applications expected"
    if out.report is not None and "weights" in out.report.provenance:
        names = [w["name"] for w in out.report.provenance["weights"]]
        assert len(names) == len(set(names)) == 2, names  # block.weight, block.bias once each


class HostOwnedKVCache(torch.nn.Module):
    """vLLM-shaped attention: K/V written into a host-owned cache buffer through a slot mapping and read back
    through block tables. Without a StateRule asserting THIS request's rows, this is `side_input`/mutation."""

    def __init__(self, h=H, slots=8):
        super().__init__()
        self.qkv = torch.nn.Linear(h, 3 * h, bias=False)
        self.register_buffer("k_cache", torch.zeros(slots, h))
        self.register_buffer("v_cache", torch.zeros(slots, h))

    def forward(self, x, slot_mapping, block_table):
        q, k, v = self.qkv(x).split(H, dim=-1)
        self.k_cache.index_copy_(0, slot_mapping, k)  # write: mutation of a buffer
        self.v_cache.index_copy_(0, slot_mapping, v)
        K = self.k_cache.index_select(0, block_table)  # read: rows chosen by a host-owned table (another request's?)
        Vv = self.v_cache.index_select(0, block_table)
        s = torch.softmax(q @ K.t(), dim=-1)
        return s @ Vv


def test_host_owned_kv_cache_is_refused():
    out = run_importer(HostOwnedKVCache(), (meta(1, H), torch.zeros(1, dtype=torch.int64, device="meta"),
                                            torch.zeros(4, dtype=torch.int64, device="meta")))
    assert_refused(out, "side_input", "HostOwnedKVCache (slot_mapping/block_table)")


# ---------------------------------------------------------------------------------------------------------
# 5. data-dependent behaviour export does not cover (boundary 2: never specialise on observed values)
# ---------------------------------------------------------------------------------------------------------


class CondOnValue(ResMLP):
    def forward(self, x):
        y = x + self.proj(torch.relu(self.fc(x)))
        return torch.cond(y.sum() > 0, lambda a: a * 2, lambda a: a, (y,))


class ItemDecides(ResMLP):
    def forward(self, x):
        y = x + self.proj(torch.relu(self.fc(x)))
        if float(y.sum().item()) > 0:
            return y * 2
        return y


class DataDependentSlice(ResMLP):
    def forward(self, x, lengths):
        n = int(lengths.max().item())
        return (x + self.proj(torch.relu(self.fc(x))))[:n]


class NonzeroShape(ResMLP):
    def forward(self, x):
        y = x + self.proj(torch.relu(self.fc(x)))
        idx = torch.nonzero(y).flatten()  # data-dependent output shape (no comparison op in front, so the refusal must name nonzero)
        return y[idx]


class WhileLoopOnValue(ResMLP):
    def forward(self, x):
        from torch._higher_order_ops.while_loop import while_loop
        y = x + self.proj(torch.relu(self.fc(x)))
        return while_loop(lambda i, a: i < 3, lambda i, a: (i + 1, a * 2), (torch.zeros((), dtype=torch.int64), y))


@pytest.mark.parametrize("cls,inp", [(CondOnValue, (meta(H),)), (ItemDecides, (meta(H),)),
                                     (DataDependentSlice, (meta(H), torch.zeros(1, dtype=torch.int64, device="meta"))),
                                     (NonzeroShape, (meta(H),)), (WhileLoopOnValue, (meta(H),))])
def test_data_dependent_behaviour_is_refused(cls, inp):
    out = run_importer(cls(), inp)
    if cls is NonzeroShape and out.kind == "unsupported" and out.reason == "decomposition" and "nonzero" in out.detail:
        # "no rule for aten.nonzero" is an honest refusal that names the data-dependent op; there must never be a rule.
        assert out.program is None
        return
    assert_refused(out, "control_flow", cls.__name__)


class GreedyDecodeEarlyExit(torch.nn.Module):
    """A bounded decode loop with `break` on EOS. Given CPU example inputs whose observed run hits EOS at step 0,
    a tracer that specialises on the observed path would emit ONE step. The importer must refuse (data-dependent
    control) or emit all STEPS steps."""
    STEPS, EOS = 4, 0

    def __init__(self, v=V, h=H):
        super().__init__()
        self.wte = torch.nn.Embedding(v, h)
        self.body = torch.nn.Linear(h, h)
        self.head = torch.nn.Linear(h, v, bias=False)
        with torch.no_grad():  # make token 0 (EOS) win every argmax on the observed run
            self.head.weight.zero_()
            self.head.weight[self.EOS].fill_(10.0)

    def forward(self, tok):
        outs = []
        for _ in range(self.STEPS):
            logits = self.head(torch.relu(self.body(self.wte(tok))))
            tok = torch.argmax(logits, dim=-1)
            outs.append(tok)
            if tok == self.EOS:  # observed on the example inputs at step 0
                break
        return torch.stack(outs)


def test_early_exit_on_eos_is_not_specialised_on_observed_tokens():
    m = GreedyDecodeEarlyExit()
    out = run_importer(m, (torch.tensor(3, dtype=torch.int64),))  # CPU values on purpose: the observed run exits at step 0
    if out.kind == "program":
        assert out.program is not None
        n_argmax = sum(1 for c in out.program.circuit.root_calls() if "TokenSelect" in c.id or "Argmax" in c.id)
        root_out = out.program.circuit.root_outputs()
        assert len(root_out) == m.STEPS, (f"the derived program returns {len(root_out)} token(s) for STEPS={m.STEPS}: "
                                           f"specialised on the observed EOS at step 0 (boundary 2)")
        assert n_argmax == 0 or n_argmax == m.STEPS
    else:
        assert_refused(out, "control_flow", "GreedyDecodeEarlyExit")


# ---------------------------------------------------------------------------------------------------------
# 7. positive-structure assertions
# ---------------------------------------------------------------------------------------------------------


class RepeatedReturnTorch(ResMLP):
    def forward(self, x):
        y = x + self.proj(torch.relu(self.fc(x)))
        return y, x, y  # (computed, the input, the same computed value again)


def test_repeated_return_and_returned_input_preserved_derived():
    out = run_importer(RepeatedReturnTorch(), (meta(H),))
    if out.kind != "program":
        pytest.skip(f"importer does not derive RepeatedReturnTorch yet ({out}). Filed as a gap.")
    P = out.program
    assert P is not None
    ro = P.circuit.root_outputs()
    assert len(ro) == 3 * H
    assert ro[:H] == ro[2 * H:], "y returned twice must be the same gates, not a duplicated computation"
    assert all(0 <= g < H for g in ro[H:2 * H]), "the returned input must be the prescribed Input gates themselves"
