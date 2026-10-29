"""Negative and adversarial tests for the automatic importer (the derive mandate's "Test generality and failure
behavior").

Contract under test (frontend design §4.3): `derive(...)` returns a complete `Program` or
`Unsupported(reason, node, module_path, hint)` (returned or raised) / a `FrontendError` code — never a partial
program, never an opaque gate, never a specialisation on observed values. Every negative here asserts BOTH that
no `Program` is returned AND that the refusal names the right class of reason. A refusal for the *wrong* reason
is reported distinctly (`REFUSED-WRONG-REASON`) from acceptance (`ACCEPTED-MUST-REFUSE`) and from a crash that is
not a classified refusal (`CRASH-NOT-REFUSAL`).

Importer resolution (XFER may not edit importer code): `verity_vllm.program.frontend.derive.derive` when it exists;
otherwise the shipped `nn.Module` route of `torch_frontend.export` (the one derivation rule the frontend applies
today, `MODULE_WEIGHTS_RULE`). Which one ran is printed by `test_00_importer_identity` and recorded in the
transfer report. Needs torch; runs on the pod (§37).
"""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits import Program, bind, program_digest  # noqa: E402
from verity_vllm.correspondence.capture_identities import LazyTranscript, Segments, check_call  # noqa: E402
from verity.ml.circuits.evaluation import evaluate_region  # noqa: E402
from verity_vllm.program.frontend import FrontendError, export, function, Arr, batch, ops  # noqa: E402
from verity_vllm.program.frontend import examples as ex  # noqa: E402
from verity_vllm.program.frontend.provenance import registry_version  # noqa: E402
from verity.ml.definitions.registry import b1  # noqa: E402
from verity.ml.definitions.registry.prims import BF16, I32  # noqa: E402
from verity.ml.circuits.refs import Coll, concat  # noqa: E402
from verity.ml.circuits.types import Array  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::verity_vllm.program.frontend.torch_frontend.DeadGateWarning")

# ---------------------------------------------------------------------------------------------------------
# importer adapter: derive(...) when it exists, else the shipped nn.Module route; one normalised Outcome
# ---------------------------------------------------------------------------------------------------------


def _load_derive():
    try:
        from verity_vllm.program.frontend import derive as D  # type: ignore
    except Exception:  # noqa: BLE001  (absent, or import-time failure: then the shipped route is the importer)
        return None
    fn = getattr(D, "derive", None)
    return (D, fn) if callable(fn) else None


_DERIVE = _load_derive()
IMPORTER_NAME = ("verity_vllm.program.frontend.derive.derive" if _DERIVE else
                 "verity_vllm.program.frontend.torch_frontend.export[nn.Module route] (derive.py not present)")
UNSUPPORTED_CLS = getattr(_DERIVE[0], "Unsupported", None) if _DERIVE else None


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
    if UNSUPPORTED_CLS is not None and isinstance(obj, UNSUPPORTED_CLS):
        return True
    return type(obj).__name__ == "Unsupported" or (hasattr(obj, "reason") and not isinstance(obj, Program))


def _unwrap_program(res: Any) -> tuple[Program | None, Any]:
    """derive may return a Program, a report carrying `.program`, or a (program, report) tuple."""
    if isinstance(res, Program):
        return res, None
    p = getattr(res, "program", None)
    if isinstance(p, Program):
        return p, res
    if isinstance(res, tuple) and res and isinstance(res[0], Program):
        return res[0], res[1] if len(res) > 1 else None
    return None, res


def _filtered_kwargs(fn: Callable, kw: dict) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def run_importer(module: torch.nn.Module, example_inputs: tuple, **kw) -> Outcome:
    """Call the importer on `module` with `example_inputs` and normalise what happened. With derive.py present:
    `derive_report(module, inputs, mode="reference", ...)` (the report carries program|unsupported + provenance);
    `mode`/`profile`/`config` may be overridden per call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if _DERIVE is not None:
                D, fn = _DERIVE
                rep_fn = getattr(D, "derive_report", None)
                target = rep_fn if callable(rep_fn) else fn
                kw = _filtered_kwargs(target, {"mode": "reference", "name": type(module).__name__, **kw})
                res = target(module, example_inputs, **kw)
                if callable(rep_fn):
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
            else:
                kw = _filtered_kwargs(export, {"name": type(module).__name__, "dead_gates": "ignore", **kw})
                res = export(module, example_inputs, **kw)
        except FrontendError as e:
            return Outcome(None, "frontend_error", e.code, str(e), e)
        except Exception as e:  # noqa: BLE001
            if _is_unsupported(e):
                r, d = _reason_of(e)
                return Outcome(None, "unsupported", r, d, e)
            return Outcome(None, "crash", type(e).__name__, str(e).splitlines()[0][:400] if str(e) else "", e)
    prog, rep = _unwrap_program(res)
    if prog is not None:
        return Outcome(prog, "program", "", f"Program {prog.fn.id} ({prog.gates} gates)", report=rep)
    if _is_unsupported(res):
        r, d = _reason_of(res)
        return Outcome(None, "unsupported", r, d, report=res)
    return Outcome(None, "crash", type(res).__name__, f"importer returned a non-Program, non-Unsupported {res!r:.200}")


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


def test_00_importer_identity():
    print(f"\nXFER importer under test: {IMPORTER_NAME}; torch {torch.__version__}")
    if _DERIVE is None:
        # the shipped route refuses every plain nn.Module (no rules): every negative below then refuses for
        # `unknown_op` — the class assertions are meaningful only once derive.py exists. Record that.
        out = run_importer(ResMLP(), (meta(H),))
        assert out.kind != "program", "the shipped export route must not accept an un-annotated nn.Module"


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


def test_rank_mismatch_is_refused_or_batched_exactly():
    """A [3, H] input into a module derived for [H]: either refused (E_SHAPE) or derived as 3 dataflow-independent
    rows — never a Program whose gate count equals the single-row program (silent row drop)."""
    single = run_importer(ResMLP(), (meta(H),))
    if single.kind != "program":
        pytest.skip(f"importer does not derive ResMLP ({single}); nothing to compare")
    out = run_importer(ResMLP(), (meta(3, H),))
    if out.kind == "program":
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


# -- cache ownership: K/V rows an attention call reads must be produced by THIS request at positions <= t --

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}


_LOS_CACHE: dict[int, list[int]] = {}


def _producer_root_call(C, g: int, roots: list):
    import bisect
    los = _LOS_CACHE.get(id(roots))
    if los is None or len(los) != len(roots):
        los = _LOS_CACHE[id(roots)] = [c.lo for c in roots]
    i = bisect.bisect_right(los, g) - 1
    if i < 0 or not (roots[i].lo <= g < roots[i].hi):
        return None
    return roots[i]


def check_cache_ownership(P: Program) -> list[str]:
    """For every root call named `t=<t>/layers.<L>/attn`: its K/V argument gates must be produced by the calls
    `t=<i>/layers.<L>/pre` for i in 0..t (same layer, this request, no future position) or be Input gates
    (never: KV rows are computed). Returns the list of violations (empty = owned correctly)."""
    C = P.circuit
    roots = sorted(C.root_calls(), key=lambda c: c.lo)
    named = {c.node.name: c for c in roots if c.node.name}
    bad: list[str] = []
    import re
    for name, call in named.items():
        m = re.match(r"^(?:r=(?P<r>[^/]+)/)?t=(?P<t>\d+)/layers\.(?P<L>\d+)/attn$", name)
        if not m:
            continue
        t, L, r = int(m["t"]), int(m["L"]), m["r"]
        prefix = f"r={r}/" if r else ""
        args = call.args
        S = call.fn.S() if hasattr(call.fn, "S") else None
        nq = call.arg_types()[0].leaves
        kv = args[nq:]
        T = call.fn.bindings.get("T") if hasattr(call.fn, "bindings") else None
        if T is not None and T != t + 1:
            bad.append(f"{name}: attends over T={T} rows at position t={t} (expected t+1={t + 1})")
        owners = set()
        for g in kv:
            pc = _producer_root_call(C, g, roots)
            if pc is None:
                bad.append(f"{name}: K/V gate {g} is not produced by any root call (Input?)")
                continue
            owners.add(pc.node.name)
        want = {f"{prefix}t={i}/layers.{L}/pre" for i in range(t + 1)}
        extra = owners - want
        missing = want - owners
        if extra:
            bad.append(f"{name}: K/V rows read from {sorted(extra)[:4]} (other layer / future position / other request)")
        if missing:
            bad.append(f"{name}: K/V rows of {sorted(missing)[:4]} are not read (a position is skipped)")
    return bad


def test_cache_ownership_oracle_accepts_hand_built_serve():
    P = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    assert check_cache_ownership(P) == []


def _miswired_serve(kind: str):
    """A Serve-shaped program whose decode attention reads K/V rows from the wrong owner: `layer` (rows of layer
    L+1), `future` (the row this position itself produces is replaced by the next step's row), `stale` (a
    position is dropped)."""
    from verity.ml.circuits.defs import CompositeDefinition
    from verity.ml.definitions.registry.b1 import Embedding, Final, LayerPost, LayerPre, kv_type
    from verity.ml.circuits.refs import tuple_of  # noqa: F401
    C, LP = TINY, 2

    def body(B, S, prompt, weights):
        NL, Hh = C["NL"], C["H"]
        hs = [B.call(bind(Embedding, V=C["V"], H=Hh), prompt[i], weights["embed"], name=f"t={i}/embed") for i in range(LP)]
        res = [None] * LP
        pre_by_layer = []
        for L in range(NL):
            w = weights["layers"][L]
            pre = [B.call(bind(LayerPre, C=C, L0=(L == 0)), hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][i],
                          name=f"t={i}/layers.{L}/pre") for i in range(LP)]
            pre_by_layer.append(pre)
            for i in range(LP):
                T = i + 1
                if kind == "layer" and L == NL - 1 and i == LP - 1:
                    src = pre_by_layer[L - 1]  # the previous layer's rows
                elif kind == "future" and i == 0:
                    src = [pre[1], pre[1]][:1]  # position 0 reads position 1's row instead of its own
                elif kind == "stale" and i == LP - 1:
                    src = pre[1:]  # drops position 0
                    T = LP - 1
                else:
                    src = pre[:T]
                kc = Coll(kv_type(C, T), concat([p[2].refs for p in src[:T]]))
                vc = Coll(kv_type(C, T), concat([p[3].refs for p in src[:T]]))
                attn = B.call(bind(b1.Attention, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"]), pre[i][1], kc, vc, name=f"t={i}/layers.{L}/attn")
                post = B.call(bind(LayerPost, C=C), attn, pre[i][0], w, name=f"t={i}/layers.{L}/post")
                hs[i], res[i] = post[0], post[1]
        return B.call(bind(Final, C=C), hs[-1], res[-1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")

    D = CompositeDefinition(f"MiswiredServe_{kind}", 1, (), lambda S: ((("prompt", Array(LP, I32)), ("weights", b1.weights_type(C))), I32),
                            body, register=False)
    return Program(bind(D))


@pytest.mark.parametrize("kind", ["layer", "future", "stale"])
def test_cache_ownership_oracle_rejects_miswired_kv(kind):
    P = _miswired_serve(kind)
    bad = check_cache_ownership(P)
    assert bad, f"mis-wired ({kind}) K/V accepted by the ownership check"


# -- the same property without names: a derived FLAT program carries no `t=/layers./attn` names (hints only) --

def _base_name(fn) -> str:
    d = getattr(fn, "defn", fn)
    return getattr(d, "name", None) or str(getattr(fn, "id", fn)).split("_v")[0]


def _kv_rows(call):
    """(k_rows, v_rows) of an `Attention{T,...}` call: per cached position, the tuple of its argument gates."""
    types = call.arg_types()
    nq = types[0].leaves
    kt, vt = types[1], types[2]
    T, per = kt.n, kt.leaves // kt.n
    args = call.args
    k = args[nq:nq + kt.leaves]
    v = args[nq + kt.leaves:nq + kt.leaves + vt.leaves]
    return [tuple(k[i * per:(i + 1) * per]) for i in range(T)], [tuple(v[i * per:(i + 1) * per]) for i in range(T)]


def check_cache_ownership_by_kind(P: Program, NL: int, LP: int, STEPS: int) -> list[str]:
    """Name-independent K/V ownership. Every root call of kind `Attention` reads T cached rows of K and of V; each
    row must be produced by a root call (never an Input / unowned gate), rows of one call must have pairwise distinct
    producers (no duplicated or dropped position), and the calls must form exactly NL chains in which each call's
    first T-1 rows are EXACTLY the rows the previous call of that chain read (the cache is extended, never replaced,
    never borrowed from another layer / request / future position), covering positions 1..LP+STEPS. Returns the
    violations; an empty attention selection is itself a violation (boundary 6)."""
    C = P.circuit
    roots = sorted(C.root_calls(), key=lambda c: c.lo)
    attn = [c for c in roots if _base_name(c.fn) == "Attention"]
    bad: list[str] = []
    if len(attn) != NL * (LP + STEPS):
        bad.append(f"{len(attn)} Attention calls at root; expected NL*(LP+STEPS) = {NL * (LP + STEPS)}")
    if not attn:
        return bad

    def prod(row):
        s = set()
        for g in row:
            pc = _producer_root_call(C, g, roots)
            if pc is None:
                return None
            s.add(pc.lo)
        return frozenset(s)

    chains: list[list[tuple]] = []
    for c in attn:
        krows, vrows = _kv_rows(c)
        T = len(krows)
        kp, vp = [prod(r) for r in krows], [prod(r) for r in vrows]
        if any(p is None for p in kp + vp):
            bad.append(f"Attention@{c.lo} (T={T}): a K/V row is not produced by any root call (Input or unowned)")
            continue
        if len(set(kp)) != T or len(set(vp)) != T:
            bad.append(f"Attention@{c.lo} (T={T}): K/V rows share a producer (a position is duplicated / dropped)")
        for ch in chains:
            Tl, kl, vl = ch[-1]
            if Tl < T and kl == kp[:Tl] and vl == vp[:Tl]:
                ch.append((T, kp, vp))
                break
        else:
            if T != 1:
                bad.append(f"Attention@{c.lo} (T={T}): no earlier attention of any layer read exactly its first {T - 1} rows "
                           f"(rows borrowed from another layer / request / future position)")
            chains.append([(T, kp, vp)])
    if len(chains) != NL:
        bad.append(f"{len(chains)} K/V chains (layers); expected NL={NL}")
    for k, ch in enumerate(chains):
        Ts = [t for t, _, _ in ch]
        if Ts != list(range(1, LP + STEPS + 1)):
            bad.append(f"chain {k} attends over T={Ts}; expected 1..{LP + STEPS}")
    return bad


def test_cache_ownership_by_kind_accepts_hand_built_serve():
    P = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    assert check_cache_ownership_by_kind(P, TINY["NL"], 2, 1) == []


@pytest.mark.parametrize("kind", ["layer", "future", "stale"])
def test_cache_ownership_by_kind_rejects_miswired_kv(kind):
    bad = check_cache_ownership_by_kind(_miswired_serve(kind), TINY["NL"], 2, 1)
    assert bad, f"mis-wired ({kind}) K/V accepted by the name-independent ownership check"


def _derive_examples():
    return pytest.importorskip("verity_vllm.program.frontend.derive_examples", reason="DERIVE's derive_examples.py not merged yet")


def _derive_serve(steps: int = 1, **kw):
    dex = _derive_examples()
    if not _DERIVE:
        pytest.skip("derive() not available")
    from verity_vllm.program.frontend.derive import derive_report
    return derive_report(dex.GreedyServe(TINY, steps, **kw), (torch.zeros(2, dtype=torch.int32),), mode="profile", profile="b1-eager", config=TINY)


def test_derived_tiny_serve_owns_its_kv_cache_and_matches_the_registered_program():
    """The derived program of DERIVE's tiny `GreedyServe` (vLLM-iface attention, KV as forward-context state):
    K/V ownership holds by NAME (if the family matched) and by KIND (always); and if the family rule claims
    `Serve_v1{C,LP,STEPS}`, the digest equals the hand-built program's (an independent check of that claim)."""
    rep = _derive_serve(1)
    if not rep.ok:
        pytest.skip(f"tiny GreedyServe does not derive: {rep.unsupported}")
    P = rep.program
    assert check_cache_ownership_by_kind(P, TINY["NL"], 2, 1) == []
    hand = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    if rep.family and rep.family.get("matched"):
        assert check_cache_ownership(P) == []
        assert rep.digest == program_digest(hand), "family rule claimed Serve_v1 but the digest differs from the registered program's"
        assert P.gates == hand.gates
    else:
        assert rep.digest != program_digest(hand)


def test_a_different_circuit_is_not_relabelled_as_serve():
    """Boundary 4: `final_norm_all_rows=True` (vLLM's model forward — final norm and lm_head over EVERY prefill row,
    of which only the last is sampled) is a legitimately different circuit with extra dead work. It must derive
    under its OWN identity, never be relabelled `Serve_v1`, and still own its KV cache."""
    rep = _derive_serve(1, final_norm_all_rows=True)
    if not rep.ok:
        pytest.skip(f"all-rows GreedyServe does not derive: {rep.unsupported}")
    hand = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    assert not (rep.family and rep.family.get("matched")), "a different circuit was relabelled Serve_v1"
    assert rep.digest != program_digest(hand)
    assert rep.gates != hand.gates
    assert check_cache_ownership_by_kind(rep.program, TINY["NL"], 2, 1) == []
    why = rep.provenance.get("families_not_matched") or []
    assert any("Serve" in str(f.get("rule", "")) for f in why), "the family rule must record WHY Serve_v1 was not claimed"


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
        n_argmax = sum(1 for c in out.program.circuit.root_calls() if "TokenSelect" in c.id or "Argmax" in c.id)
        root_out = out.program.circuit.root_outputs()
        assert len(root_out) == m.STEPS, (f"the derived program returns {len(root_out)} token(s) for STEPS={m.STEPS}: "
                                           f"specialised on the observed EOS at step 0 (boundary 2)")
        assert n_argmax == 0 or n_argmax == m.STEPS
    else:
        assert_refused(out, "control_flow", "GreedyDecodeEarlyExit")


# ---------------------------------------------------------------------------------------------------------
# 6. capture-to-program mappings, corrupted values, missing obligations, mismatched versions
#    (pure verity.ml.circuits: the correspondence primitives that any derive_corr.py must rest on; the derive_corr.py
#     specific tests are appended when CAPCORR's script lands)
# ---------------------------------------------------------------------------------------------------------


def _segments(d: dict[int, int]) -> Segments:
    """A value source over an explicit gate->word dict (stands in for weights + captured leaves), one segment
    per maximal run of consecutive gates."""
    s = Segments()
    ks = sorted(d)
    i = 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1:
            j += 1
        s.add(ks[i], np.array([d[g] for g in ks[i:j + 1]], dtype=np.uint32), {"leaf_id": f"xfer:{ks[i]}"})
        i = j + 1
    return s


def _tiny_program_with_truth():
    """A root call of `BiasAdd{N=8}` over prescribed inputs, fully evaluated: (program, truth transcript, the call)."""
    from verity.ml.circuits.defs import CompositeDefinition
    W = CompositeDefinition("XferWrapBiasAdd", 1, (), lambda S: ((("y", Array(8, BF16)), ("b", Array(8, BF16))), Array(8, BF16)),
                            lambda B, S, y, b: B.call(bind(b1.BiasAdd, N=8), y, b, name="ba"), register=False)
    P = Program(bind(W))
    C = P.circuit
    rng = np.random.default_rng(0)
    truth: dict[int, int] = {}
    for lo, hi in _input_gate_ranges(P):
        for g in range(lo, hi):
            truth[g] = int(rng.integers(0x3C00, 0x4000))  # bf16 words in [1, 2)
    evaluate_region(C, C[0:C.size], truth)
    call = [c for c in C.root_calls() if not c.is_primitive][0]
    return P, truth, call


def _check(call, gates, source: dict[int, int], captured: np.ndarray) -> dict:
    return check_call(call, gates, LazyTranscript(_segments(source)), captured)


def test_correspondence_exact_on_true_values():
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    r = _check(call, gates, inputs_only, np.array([truth[g] for g in gates], dtype=np.uint16))
    assert r["status"] == "exact" and r["mismatches"] == 0, r


def test_corrupted_captured_value_fails():
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    cap = np.array([truth[g] for g in gates], dtype=np.uint16)
    cap[3] ^= 0x0001  # one flipped mantissa bit in one word
    r = _check(call, gates, inputs_only, cap)
    assert r["status"] == "mismatch" and r["mismatches"] == 1 and r["first_mismatches"][0]["word"] == 3, r


def test_wrong_address_map_fails():
    """A capture mapped to the wrong gates (an address map off by one call / a permutation) must not verify."""
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    true_words = np.array([truth[g] for g in gates], dtype=np.uint16)
    rolled = np.roll(true_words, 1)  # right values, wrong positions
    assert not np.array_equal(rolled, true_words)
    r = _check(call, gates, inputs_only, rolled)
    assert r["status"] == "mismatch" and r["mismatches"] > 0, r
    # the map points at Input gates instead of the call's returns: shape/identity check must fail closed
    wrong_gates = tuple(range(call.lo - 8, call.lo))  # the bias inputs, not the outputs
    r2 = _check(call, wrong_gates, inputs_only, true_words)
    assert r2["status"] != "exact" or r2["words_compared"] == 0, r2


def test_corrupted_prescribed_input_fails():
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    k = min(inputs_only)
    inputs_only[k] ^= 0x0100  # a corrupted weight word
    r = _check(call, gates, inputs_only, np.array([truth[g] for g in gates], dtype=np.uint16))
    assert r["status"] == "mismatch", r


def test_missing_operand_is_reported_not_invented():
    """A value the record does not carry (and nothing recomputes) is `missing-operand`, never silently zero."""
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    partial = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    del partial[min(partial)]
    r = _check(call, gates, partial, np.array([truth[g] for g in gates], dtype=np.uint16))
    assert r["status"] == "missing-operand", r


def test_wrong_dtype_or_shape_in_record_fails_shape_check():
    from verity_vllm.correspondence.capture_identities import _shape_ok
    P, truth, call = _tiny_program_with_truth()
    ok, _ = _shape_ok(call, call.returns, {"shape": [8], "dtype": "bfloat16"})
    assert ok
    ok, why = _shape_ok(call, call.returns, {"shape": [8], "dtype": "float32"})
    assert not ok and "dtype" in why
    ok, why = _shape_ok(call, call.returns, {"shape": [4], "dtype": "bfloat16"})
    assert not ok and "words" in why


def test_program_identity_differs_between_versions_and_statics():
    """A record names a program by digest; any tool must refuse a record whose digest is not the program's.
    Here: the digest separates statics and registry versions (the mismatch the tool must detect)."""
    d8 = program_digest(Program(bind(b1.BiasAdd, N=8)))
    d16 = program_digest(Program(bind(b1.BiasAdd, N=16)))
    assert d8 != d16
    rv = registry_version()
    assert rv["digest"] and rv["primitives"] > 0
    # the registry-version record is what pins the primitive vocabulary: two records that disagree on it
    # are not comparable even when the program digest agrees (frontend §8.1)
    rec_a = {"program": d8, "registry": rv["digest"]}
    rec_b = {"program": d8, "registry": "0" * 64}
    assert rec_a["program"] == rec_b["program"] and rec_a["registry"] != rec_b["registry"]


# ---------------------------------------------------------------------------------------------------------
# 7. positive-structure assertions (authoring route today; plain-torch route through derive when it lands)
# ---------------------------------------------------------------------------------------------------------

V16 = ex.Value(16)


def test_repeated_return_and_returned_input_preserved_authored():
    """IFb's identity shapes: the same tensor returned twice is ONE reference repeated (not two gates); a returned
    input is the caller's reference; the dead declared return is in `returns`, not in `out`."""
    Top, _ = ex.identity_pair(*_running_prims())
    P = export(Top, (torch.empty((), dtype=torch.bfloat16, device="meta"),), dead_gates="ignore")
    from verity.ml.circuits import query
    f = query(P.circuit, "//F")[0]
    assert f.args == (0, 0)
    assert f.returns[0] == f.returns[2], "t returned twice must be the same gate"
    assert f.returns[1] == 0, "the returned input must be the caller's Input gate"
    assert f.out == (1,)


def _running_prims():
    from tests.program.test_running_example import Add16, Id16, Mul16
    return Add16, Mul16, Id16


def test_repeated_return_gate_count_pinned():
    Top, _ = ex.identity_pair(*_running_prims())
    P = export(Top, (torch.empty((), dtype=torch.bfloat16, device="meta"),), dead_gates="ignore")
    assert P.gates == 4, "Input(v) + Add16 + Mul16(dead) + Id16 — a duplicated `t` gate would make 5"


class RepeatedReturnTorch(ResMLP):
    def forward(self, x):
        y = x + self.proj(torch.relu(self.fc(x)))
        return y, x, y  # (computed, the input, the same computed value again)


def test_repeated_return_and_returned_input_preserved_derived():
    out = run_importer(RepeatedReturnTorch(), (meta(H),))
    if out.kind != "program":
        pytest.skip(f"importer does not derive RepeatedReturnTorch yet ({out}). Filed as a gap.")
    P = out.program
    ro = P.circuit.root_outputs()
    assert len(ro) == 3 * H
    assert ro[:H] == ro[2 * H:], "y returned twice must be the same gates, not a duplicated computation"
    assert all(0 <= g < H for g in ro[H:2 * H]), "the returned input must be the prescribed Input gates themselves"


def test_tied_parameter_one_input_authored():
    """The same sharing property on the shipped route: a module whose weight is used twice keeps one Input record
    field (functional_call over `named_parameters()` dedups by identity)."""

    class Twice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))

        def forward(self, x):
            return ex.BiasAdd(ex.BiasAdd(x, self.w), self.w)

    P = export(Twice(), (torch.empty(8, dtype=torch.bfloat16, device="meta"),), name="Twice", dead_gates="ignore")
    assert P.input_gates == 16
    calls = [c for c in P.circuit.root_calls() if c.name == "BiasAdd"]
    assert len(calls) == 2 and calls[0].args[8:] == calls[1].args[8:], "both BiasAdds must read the same 8 weight gates"
