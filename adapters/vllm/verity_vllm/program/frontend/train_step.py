"""One training step as a Program: the forward, the backward and the optimizer of an ordinary `nn.Module`, derived through the same
front door as a served model (`derive_graph.derive_graph`), with the training rules (`rules/training.py`) added to the default ones.

    step_program(model, loss_fn, optimizer, example_batch, mode="reference", cc=None) -> Program | Unsupported

The step is one function of everything a step reads,

    (params, exp_avg, exp_avg_sq, decay, neg_step_size, bc2_sqrt, buffers, batch) -> (loss, params', exp_avg', exp_avg_sq')

built with `torch.func.functional_call` and `grad_and_value`, the optimizer a function (`AdamW.update`), and traced as one joint ATen
graph on fake tensors (`make_fx` of the functionalized step, SDPA's math backend), which `derive_graph` derives. No value is read:
the parameters, the optimizer state, the step's three host scalars (`AdamW.scalars`), the module's buffers and the batch are the
Program's inputs, in that order, each tensor one parameter named after it (`param_<fqn>`, `exp_avg_<fqn>`, ...). The parameters
are the Program's theta (`Program.theta`: the start's weights); the optimizer state, the scalars and the batch are not.

At a compute capability (`cc`) the step is traced as that GPU runs it (`export_step(..., gpu=True)`): a `Tensor.is_cuda` the step's
own code reads is True, so it takes its CUDA branches (`x.bfloat16() if x.is_cuda else x.float()` is bf16), and the TF32 setting
its calls run under (cuda.matmul.allow_tf32), which the traced graph doesn't show, is the rules' `tf32`.

A node no rule lowers is `Unsupported`, as for any derivation. `step_report(..., survey=True)` instead lowers every node it can and
takes each refused one's output as a field of the `run` input (a hole, `rules.training_survey`), and returns a `StepSurvey`: every
refusal at once and the census of everything else, without a Program. A hole is a value no gate checks, which a prover would
choose, so a Program with holes is no statement (`check_identity` refuses it); `step_program` never surveys.

A witness block is a `query_ast.GateSet` over the step Program: its gates are `GateSet.size()`, and its reads beyond the start's
weights are `in_bits_beyond_theta(program, gates)`.
"""

from __future__ import annotations

import contextlib
import keyword
import os
import re
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch.func import functional_call, grad_and_value
from torch.overrides import TorchFunctionMode
from verity.circuits import parts
from verity.circuits.boundary import gate_width, in_gates
from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.program import Program, is_input
from verity.circuits.query_ast import GateSet
from verity.definitions.circuits.census import census

from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.report import DeriveReport
from verity_vllm.program.frontend.rules.base import RuleSet


@dataclass(frozen=True)
class AdamW:
    """`torch.optim.AdamW(foreach=False)`'s step on CPU (`_single_tensor_adam`: not capturable, not fused, no amsgrad, no maximize)
    as a function of (param, grad, exp_avg, exp_avg_sq) and the step's host scalars, each line one ATen op:

        p1 = p * decay                          `param.mul_(1 - lr * weight_decay)` (skipped at weight decay 0, as torch does)
        m1 = m.lerp(g, 1 - beta1)               `exp_avg.lerp_(grad, 1 - beta1)`
        v1 = (v * beta2).addcmul(g, g, 1 - beta2)
        denom = v1.sqrt() / bc2_sqrt + eps
        p2 = p1 + neg_step_size * m1 / denom    `param.addcdiv_(exp_avg, denom, value=-step_size)`: self + (value * t1) / t2

    The host scalars are the Python floats torch computes once a step from the step count t (`scalars`), which its kernels read
    as f32 words; they are inputs of the step, so one Program serves every step."""
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2

    SCALARS = ("decay", "neg_step_size", "bc2_sqrt")

    @classmethod
    def from_torch(cls, opt: torch.optim.Optimizer) -> AdamW:
        """The hyperparameters of a `torch.optim.AdamW` with one parameter group and the defaults this function computes."""
        if not isinstance(opt, torch.optim.AdamW) or len(opt.param_groups) != 1:
            raise ValueError("a torch.optim.AdamW with one parameter group")
        g = opt.param_groups[0]
        for k in ("amsgrad", "maximize", "capturable", "differentiable"):
            if g.get(k):
                raise ValueError(f"AdamW with {k}=True is not this function")
        return cls(lr=float(g["lr"]), betas=(float(g["betas"][0]), float(g["betas"][1])), eps=float(g["eps"]),
                   weight_decay=float(g["weight_decay"]))

    def scalars(self, step: int) -> dict[str, float]:
        """The host's Python floats at step `step` (1 for the first): `1 - lr * weight_decay`, `-(lr / (1 - beta1 ** t))` and
        `(1 - beta2 ** t) ** 0.5`; f32 inputs of the step, rounded as torch's kernels round them (`Scalar::to<float>`)."""
        b1, b2 = self.betas
        return {"decay": 1 - self.lr * self.weight_decay, "neg_step_size": -(self.lr / (1 - b1 ** step)),
                "bc2_sqrt": (1 - b2 ** step) ** 0.5}

    def update(self, p: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor, decay: torch.Tensor,
               neg_step_size: torch.Tensor, bc2_sqrt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b1, b2 = self.betas
        if self.weight_decay != 0:
            p = p * decay
        m = m.lerp(g, 1 - b1)
        v = (v * b2).addcmul(g, g, value=1 - b2)
        denom = v.sqrt() / bc2_sqrt + self.eps
        return p + neg_step_size * m / denom, m, v


class _Loss(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, loss_fn: Callable) -> None:
        super().__init__()
        self.model, self.loss_fn = model, loss_fn

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.model, batch)


@dataclass
class StepLayout:
    """Which Program input and output is which: the parameters' and buffers' fqns in order, the host scalars, the batch; and how
    the step was traced: as a GPU runs it (`gpu`), and then with cuda.matmul.allow_tf32 on (`tf32`)."""
    params: list[str]
    buffers: list[str]
    scalars: tuple[str, ...]
    inputs: list[str]
    outputs: list[str]
    gpu: bool = False
    tf32: bool = False

    def to_json(self) -> dict:
        return {"params": self.params, "buffers": self.buffers, "scalars": list(self.scalars), "inputs": self.inputs,
                "outputs": self.outputs}


def _ident(s: str) -> str:
    s = re.sub(r"\W", "_", s)
    return s + "_" if keyword.iskeyword(s) else s


_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep


class _TF32Seen(TorchFunctionMode):
    """Collects the cuda.matmul.allow_tf32 each torch call of the traced step runs under."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[bool] = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.seen.add(bool(torch.backends.cuda.matmul.allow_tf32))
        return func(*args, **(kwargs or {}))


#: `F.scaled_dot_product_attention`'s parameters in order (a builtin, which has no signature to read them from)
_SDPA_PARAMS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")


class _FlashAttention(TorchFunctionMode):
    """Gives a `F.scaled_dot_product_attention` call of bf16 operands with no mask to `aten._scaled_dot_product_flash_attention`,
    the op `fa2_train`'s captures are of, with the call's own arguments; autograd then takes its backward from that op's. A meta
    tensor's own dispatch is the math path whatever the SDPA backends allow."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.scaled_dot_product_attention:
            call = {**dict(zip(_SDPA_PARAMS, args)), **kwargs}
            operands = [call["query"], call["key"], call["value"]]
            if call.get("attn_mask") is None and not call.get("enable_gqa") and all(t.dtype == torch.bfloat16 for t in operands):
                return torch.ops.aten._scaled_dot_product_flash_attention(
                    *operands, call.get("dropout_p", 0.0), call.get("is_causal", False), False, scale=call.get("scale"))[0]
        return func(*args, **kwargs)


@contextlib.contextmanager
def _patched(owner, name: str, value):
    """``owner.name = value`` for the block. Afterwards ``owner`` holds under ``name`` what it held itself before: that attribute
    again, or none where it inherited the name, which it then inherits again."""
    absent = object()
    prior = vars(owner).get(name, absent)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if prior is absent:
            delattr(owner, name)
        else:
            setattr(owner, name, prior)


@contextlib.contextmanager
def _gpu_branches(fused_attention: bool = False):
    """The step as a GPU runs it, on meta tensors: a `Tensor.is_cuda` read from outside torch is True (torch's own reads keep the
    meta tensor's answer, so its dispatch is unchanged), and with `fused_attention` its bf16 attention is the flash attention op
    (`_FlashAttention`); yields the set of TF32 settings the step's calls ran under."""
    # read from the class this is the descriptor, where torch's stubs give the bool an instance answers with
    base = cast(types.GetSetDescriptorType, torch._C.TensorBase.is_cuda)

    def is_cuda(t: torch.Tensor) -> bool:
        return True if not sys._getframe(1).f_code.co_filename.startswith(_TORCH_DIR) else base.__get__(t)

    attention: contextlib.AbstractContextManager = _FlashAttention() if fused_attention else contextlib.nullcontext()
    with _patched(torch.Tensor, "is_cuda", property(is_cuda)), _TF32Seen() as mode, attention:
        yield mode.seen


def export_step(model: torch.nn.Module, loss_fn: Callable, optimizer: AdamW | torch.optim.Optimizer, example_batch: torch.Tensor, *,
                gpu: bool = False, fused_attention: bool = False) -> tuple[torch.fx.GraphModule, list[torch.Tensor], StepLayout]:
    """The step's joint ATen graph (a `make_fx` `GraphModule`), its example inputs (meta tensors) and its `StepLayout`. `gpu`: traced
    as a GPU runs it (`_gpu_branches`): its CUDA branches, and the layout's `tf32` the cuda.matmul.allow_tf32 its calls ran
    under, which must be one setting for the whole step (a step that changes it between products is refused).
    `fused_attention`, with `gpu`: bf16 attention is torch's flash attention op, forward and backward (`_FlashAttention`), where
    the trace otherwise holds the math path, a meta tensor's dispatch. The backward op has no rule, so a step traced so derives
    no Program when it differentiates through attention."""
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if fused_attention and not gpu:
        raise ValueError("export_step: fused_attention is a GPU's kernel: pass gpu=True with it")
    adamw = AdamW.from_torch(optimizer) if isinstance(optimizer, torch.optim.Optimizer) else optimizer
    wrap = _Loss(model, loss_fn)
    pnames = [n for n, _ in wrap.named_parameters()]
    bnames = [n for n, _ in wrap.named_buffers()]
    P, NB = len(pnames), len(bnames)
    scalars = AdamW.SCALARS

    def step(*flat):
        params = dict(zip(pnames, flat[:P]))
        ms, vs = flat[P:2 * P], flat[2 * P:3 * P]
        decay, neg_step_size, bc2_sqrt = flat[3 * P:3 * P + 3]
        bufs = dict(zip(bnames, flat[3 * P + 3:3 * P + 3 + NB]))
        batch = flat[-1]
        grads, loss = grad_and_value(lambda p: functional_call(wrap, (p, bufs), (batch,)))(params)
        new_p, new_m, new_v = [], [], []
        for i, n in enumerate(pnames):
            p2, m2, v2 = adamw.update(params[n], grads[n], ms[i], vs[i], decay, neg_step_size, bc2_sqrt)
            new_p.append(p2)
            new_m.append(m2)
            new_v.append(v2)
        return (loss, *new_p, *new_m, *new_v)

    def meta(t: torch.Tensor) -> torch.Tensor:
        return torch.empty(t.shape, dtype=t.dtype, device="meta")

    tensors = dict(wrap.named_parameters())
    btensors = dict(wrap.named_buffers())
    # one distinct tensor per input: make_fx binds a tensor object passed twice to its last placeholder
    metas = ([meta(tensors[n]) for _ in range(3) for n in pnames] + [torch.empty((), dtype=torch.float32, device="meta") for _ in scalars]
             + [meta(btensors[n]) for n in bnames] + [meta(example_batch)])
    short = [n.removeprefix("model.") for n in pnames]
    bshort = [n.removeprefix("model.") for n in bnames]
    inputs = ([_ident(f"param_{n}") for n in short] + [_ident(f"exp_avg_{n}") for n in short] + [_ident(f"exp_avg_sq_{n}") for n in short]
              + list(scalars) + [_ident(f"buf_{n}") for n in bshort] + ["batch"])
    outputs = ["loss"] + [f"param:{n}" for n in short] + [f"exp_avg:{n}" for n in short] + [f"exp_avg_sq:{n}" for n in short]
    branches: contextlib.AbstractContextManager[set[bool]] = _gpu_branches(fused_attention) if gpu else contextlib.nullcontext(set())
    with sdpa_kernel(SDPBackend.MATH), branches as tf32:
        gm = make_fx(torch.func.functionalize(step, remove="mutations"), tracing_mode="fake")(*metas)
    if len(tf32) > 1:
        raise ValueError("export_step: the step changes cuda.matmul.allow_tf32 between its calls, and the rules take one setting "
                         "for the whole step: set it once, outside the step")
    return gm, metas, StepLayout(short, bshort, scalars, inputs, outputs, gpu=gpu,
                                 tf32=gpu and (tf32.pop() if tf32 else bool(torch.backends.cuda.matmul.allow_tf32)))


@dataclass(frozen=True)
class StepSurvey:
    """A step's survey (`step_report(..., survey=True)`): what its rules refuse, every op at once, and the census of the rest. It
    holds no Program: the one it counts takes each refused node's outputs from fields of the `run` input that no gate checks."""
    unsupported: Unsupported | None   # the step's refusal (`step_program`'s); None: the step derives, and the survey has no holes
    holes: list[dict]                 # each refused node: node, op, reason, hint and its `run` fields
    census: dict | None               # `verity.definitions.circuits.census.census` of the step with its holes; None when `refused`
    primitives: dict[str, PrimitiveDefinition]   # the primitives it calls, by id: what a census prices
    theta: dict                       # theta's input names, input nodes and bits
    layout: StepLayout
    refused: Unsupported | None = None   # a refusal no hole stands in for (outside the rules): no census


def step_report(model: torch.nn.Module, loss_fn: Callable, optimizer: AdamW | torch.optim.Optimizer, example_batch: torch.Tensor, *,
                mode: str = "reference",
                survey: bool = False, rules: RuleSet | None = None, cc: tuple[int, int] | None = None, name: str = "TrainStep",
                exported: tuple[torch.fx.GraphModule, list[torch.Tensor], StepLayout] | None = None) -> DeriveReport | StepSurvey:
    """`step_program`'s derivation report. `rules`: the rule set (default `rules.training.training_rules(cc=cc, tf32=...)`, with
    the step's TF32 setting at a `cc`). `exported`: `export_step`'s result, at a `cc` traced as the GPU runs it (`gpu=True`).
    `survey`: a `StepSurvey` instead, of the step derived with every refused node a hole (`rules.training_survey.survey`)."""
    from verity_vllm.program.frontend.rules.training import training_rules
    from verity_vllm.program.frontend.rules.training_survey import (
        survey as survey_graph,
    )

    if rules is not None and cc is not None:
        raise ValueError("step_report: pass rules or cc, not both (training_rules(cc=...) is the rule set cc names)")
    gpu = cc is not None
    gm, metas, layout = exported if exported is not None else export_step(model, loss_fn, optimizer, example_batch, gpu=gpu)
    if gpu and not layout.gpu:
        raise ValueError("step_report: at a cc the step is the GPU's, its CUDA branches and its TF32 setting: export it with "
                         "export_step(..., gpu=True)")
    ruleset = rules if rules is not None else training_rules(cc=cc, tf32=layout.tf32)
    theta = layout.inputs[:len(layout.params)]
    rep = derive_graph(gm, metas, mode=mode, rules=ruleset, name=name, input_names=layout.inputs, theta=theta)
    rep.provenance["train_step"] = layout.to_json()
    if rep.program is not None:
        rep.program.annotations["train_step"] = layout.to_json()
    if not survey:
        return rep
    holes, prog = ([], rep.program) if rep.program is not None else survey_graph(gm, metas, mode=mode, rules=ruleset, name=name,
                                                                                 input_names=layout.inputs, theta=theta)
    if isinstance(prog, Unsupported):
        return StepSurvey(rep.unsupported, holes, None, {}, {}, layout, refused=prog)
    nodes = prog.root.body.nodes
    return StepSurvey(rep.unsupported, holes, census(prog), primitives(prog), {
        "names": list(prog.theta), "input_nodes": len(prog.root.theta),
        "bits": sum(nodes[i].n * nodes[i].fn.ret.w for i in prog.root.theta)}, layout)


def primitives(prog: Program) -> dict[str, PrimitiveDefinition]:
    """The primitives `prog` calls, at any depth, by id: what a census (`verity.definitions.circuits.census.census`) prices."""
    out, seen, stack = {}, set(), [prog.root]
    while stack:
        fn = stack.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        if isinstance(fn, PrimitiveDefinition):
            out[fn.id] = fn
        else:
            stack.extend(nd.fn for nd in fn.body.nodes)
    return out


def step_program(model: torch.nn.Module, loss_fn: Callable, optimizer: AdamW | torch.optim.Optimizer, example_batch: torch.Tensor, *,
                 mode: str = "reference", cc: tuple[int, int] | None = None) -> Program | Unsupported:
    """One training step of `model` as a `Program`, or the `Unsupported` that says which op of the step no rule lowers.

    `loss_fn(model, batch)` returns the scalar loss; `optimizer` is an `AdamW` (or a `torch.optim.AdamW` with one parameter group);
    `example_batch` gives the batch's shape and dtype (its values are never read, nor are the model's). The Program's inputs are the
    parameters (tied ones once), the optimizer state, the step's host scalars (`AdamW.scalars`), the buffers and the batch; its
    outputs are the loss and the new parameters and state, in `annotations["train_step"]`'s order. `mode` is the derivation mode:
    "reference" lowers every op to the reference vocabulary's exact f32 arithmetic (`rules.training`). `cc` (the compute capability
    `cublas_bf16.TRAIN_CC`): the step as that GPU runs it (`export_step(..., gpu=True)`), its bf16 matrix products as its cuBLAS
    computes them instead, at the shapes its capture covers, and its f32 ones refused under cuda.matmul.allow_tf32 (TF32 GEMMs)."""
    report = step_report(model, loss_fn, optimizer, example_batch, mode=mode, cc=cc)
    assert isinstance(report, DeriveReport)  # a refusal is a DeriveReport too; a StepSurvey comes only with survey=True
    return report.result


def in_bits_beyond_theta(program: Program, gates: GateSet) -> int | parts.Unsupported:
    """The bits a witness block `gates` reads beyond theta: the summed widths of the distinct gates outside it with a wire into it
    (`boundary.in_gates`), less theta's input gates (the start's weights, `Program.root.theta`) and structure (a constant, or a gate
    of constants alone, which a statement hardwires, as `partition_v2`'s units carry it). `Unsupported` where `boundary.in_gates` is:
    a read its compact index cannot resolve exactly."""
    offs = program.root.body.offsets
    theta = [(offs[j], offs[j + 1]) for j in sorted(program.root.theta)]
    reads: set[int] = set()
    for part in gates.parts:
        r = in_gates(part)
        if isinstance(r, parts.Unsupported):
            return r
        reads |= r
    memo: dict[int, bool] = {}
    return sum(gate_width(program, g) for g in sorted(reads)
               if g not in gates and not any(lo <= g < hi for lo, hi in theta) and not _structure(program.circuit, g, memo))


def _structure(circuit, g: int, memo: dict[int, bool]) -> bool:
    """Whether root gate `g` is a constant (no operands and no input) or computed from constants alone."""
    if g in memo:
        return memo[g]
    seen, stack = set(), [g]
    while stack:
        h = stack.pop()
        if h in seen or memo.get(h) is True:
            continue
        seen.add(h)
        gate = circuit.gate(h)
        if memo.get(h) is False or is_input(gate.prim):
            memo[g] = False
            return False
        stack.extend(gate.operands)
    memo.update(dict.fromkeys(seen, True))
    return True


__all__ = ["AdamW", "StepLayout", "StepSurvey", "export_step", "in_bits_beyond_theta", "primitives", "step_report", "step_program"]
