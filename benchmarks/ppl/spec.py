"""Arms: what is done to each matmul, declared in TOML, and the registry of steps an arm is made of.

An arm is a list of steps for the linears (`[[linear]]`, every x W^T of q/k/v/o/gate/up/down) and one for attention
(`[[attention]]`, both of its matmuls: S = Q K^T and O = P V). Each step names a registered function and its keyword
parameters; `layers = [...]` and, on linears, `linears = [...]` restrict where it applies (`LINEAR_NAMES`, GLM-5.3's
`GLM_LINEAR_NAMES` with `mlp.experts.E.*` for every routed expert, and `lm_head`). An arm with no steps is the bf16
reference itself.

A step is one function, registered with `@linear_step(name)` or `@attention_step(name)`:

    @linear_step("smooth", phase="real", calib=True)
    def smooth(mm, nxt, *, alpha: float, kv_shared: bool = False):
        m = mm.cached("m", lambda: ...)
        return nxt(mm.replace(x=mm.x / m, w=lambda w: w * m))

It receives the matmul `mm` (see `matmul.MM`) and `nxt`, the rest of the chain, and returns the product. Steps run in
three phases, in order: `real` steps transform the real operands (and may post-process the product), one `quantize` step
puts the operands on the integer grid and scales the integer product the rest of the chain returns, and `integer` steps
act on the integer operands; the chain ends in the gated exact integer product (or, with no quantizer, the FP32 one)."""
from __future__ import annotations

import inspect
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
SPECS = HERE / "specs"
PHASES = ("real", "quantize", "integer")
KINDS = ("linear", "attention")
REGISTRY: dict[str, dict[str, "StepDef"]] = {k: {} for k in KINDS}
LINEAR_NAMES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                "mlp.up_proj", "mlp.down_proj")
#: GLM-5.3's linears (glm.py), every routed expert's under `mlp.experts.E`
GLM_LINEAR_NAMES = ("self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.shared_experts.gate_proj",
                    "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj", "mlp.experts.E.gate_up_proj",
                    "mlp.experts.E.down_proj")
EXPERT = re.compile(r"\.experts\.\d+\.")
#: the output head, x W^T on the final norm's output: a linear step reaches it only by naming it in `linears`
HEAD = "lm_head"
ATTENTION_MATMULS = ("qk", "pv")


@dataclass(frozen=True)
class StepDef:
    kind: str
    name: str
    fn: Callable
    phase: str
    params: dict          # name -> default (inspect.Parameter.empty: required)
    seeded: Callable      # params -> bool: the step's effect on the product depends on the arm's seed
    calib: Callable       # params -> bool: the step reads calibration statistics
    on: tuple             # attention: the matmuls it acts on
    check: Callable | None
    types: dict = field(default_factory=dict)  # name -> annotation, as written ("float", "int", "bool", "str")
    source: str = ""      # the harnesses and notes it comes from (`run.py tables steps`)


def _register(kind: str, name: str, phase: str, seeded, calib, on, check, source=""):
    if phase not in PHASES:
        raise ValueError(f"phase {phase}")

    def deco(fn):
        sig = inspect.signature(fn)
        params = {p.name: p.default for p in list(sig.parameters.values())[2:]}
        if any(p.kind != inspect.Parameter.KEYWORD_ONLY for p in list(sig.parameters.values())[2:]):
            raise TypeError(f"{name}: a step's parameters are keyword-only, after (mm, nxt)")
        types = {p.name: p.annotation if isinstance(p.annotation, str) else getattr(p.annotation, "__name__", "")
                 for p in list(sig.parameters.values())[2:]}
        as_fn = lambda v: v if callable(v) else (lambda p: v)  # noqa: E731
        REGISTRY[kind][name] = StepDef(kind, name, fn, phase, params, as_fn(seeded), as_fn(calib), tuple(on), check,
                                       types, source)
        return fn
    return deco


def linear_step(name: str, *, phase: str = "real", seeded=False, calib=False, check=None, source=""):
    return _register("linear", name, phase, seeded, calib, ("linear",), check, source)


def attention_step(name: str, *, phase: str = "real", seeded=False, calib=False, on=ATTENTION_MATMULS, check=None,
                   source=""):
    return _register("attention", name, phase, seeded, calib, on, check, source)


@dataclass(frozen=True)
class Step:
    defn: StepDef
    params: tuple                       # sorted (name, value) pairs, every parameter (defaults filled)
    layers: frozenset | None = None
    linears: frozenset | None = None

    @property
    def kw(self) -> dict:
        return dict(self.params)

    @property
    def key(self) -> tuple:
        return (self.defn.kind, self.defn.name, self.params, tuple(sorted(self.layers or ())) or None,
                tuple(sorted(self.linears or ())) or None)

    def applies(self, layer: int, kind: str, name: str) -> bool:
        """kind: `linear`, `qk` or `pv`; name: the linear's module name (or the attention matmul's kind)."""
        if self.layers is not None and layer not in self.layers:
            return False
        if (self.linears is None and name == HEAD) or (
                self.linears is not None and EXPERT.sub(".experts.E.", name) not in self.linears):
            return False
        return kind in self.defn.on

    def to_toml(self) -> dict:
        out: dict[str, Any] = {"step": self.defn.name, **self.kw}
        if self.layers is not None:
            out["layers"] = sorted(self.layers)
        if self.linears is not None:
            out["linears"] = sorted(self.linears)
        return out


@dataclass(frozen=True)
class Arm:
    name: str
    linear: tuple = ()
    attention: tuple = ()
    description: str = ""
    plain: str = field(default="", compare=False)   # the scope-matched plain int8 arm it is reported against

    @property
    def steps(self) -> tuple:
        return self.linear + self.attention

    @property
    def seeded(self) -> bool:
        return any(s.defn.seeded(s.kw) for s in self.steps)

    @property
    def needs_calib(self) -> bool:
        return any(s.defn.calib(s.kw) for s in self.steps)

    @property
    def scope(self) -> str:
        return {(True, True): "all", (True, False): "linears", (False, True): "attention"}.get(
            (bool(self.linear), bool(self.attention)), "none")

    def seeds(self, seeds) -> tuple:
        return tuple(seeds) if self.seeded else tuple(seeds)[:1]

    def to_toml(self) -> dict:
        return {"description": self.description, "linear": [s.to_toml() for s in self.linear],
                "attention": [s.to_toml() for s in self.attention]}


def _typed(defn: StepDef, pname: str, value):
    default = defn.params[pname]
    if default is inspect.Parameter.empty:
        default = {"float": 0.0, "int": 0, "bool": False, "str": ""}.get(defn.types.get(pname, ""))
    if default is None:
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{defn.name}.{pname}: a bool, not {value!r}")
    elif isinstance(default, int) and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{defn.name}.{pname}: an int, not {value!r}")
    elif isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{defn.name}.{pname}: a number, not {value!r}")
        value = float(value)
    elif isinstance(default, str) and not isinstance(value, str):
        raise ValueError(f"{defn.name}.{pname}: a string, not {value!r}")
    return value


def parse_step(kind: str, table: dict) -> Step:
    table = dict(table)
    name = table.pop("step", None)
    if name not in REGISTRY[kind]:
        raise ValueError(f"[[{kind}]] step {name!r}: registered steps are {sorted(REGISTRY[kind])}")
    defn = REGISTRY[kind][name]
    layers = table.pop("layers", None)
    linears = table.pop("linears", None) if kind == "linear" else None
    if linears is not None and not set(linears) <= {*LINEAR_NAMES, *GLM_LINEAR_NAMES, HEAD}:
        raise ValueError(f"{name}: linears are among {(*dict.fromkeys(LINEAR_NAMES + GLM_LINEAR_NAMES), HEAD)}")
    unknown = set(table) - set(defn.params)
    if unknown:
        raise ValueError(f"{name}: unknown parameters {sorted(unknown)} (it takes {sorted(defn.params)})")
    params = {}
    for p, default in defn.params.items():
        if p in table:
            params[p] = _typed(defn, p, table[p])
        elif default is inspect.Parameter.empty:
            raise ValueError(f"{name}: parameter {p!r} is required")
        else:
            params[p] = default
    if defn.check is not None:
        defn.check(params)
    if linears is not None and HEAD in linears and (defn.calib(params) or layers is not None):
        raise ValueError(f"{name}: {HEAD} takes steps without calibration or layers")
    return Step(defn, tuple(sorted(params.items())), frozenset(layers) if layers is not None else None,
                frozenset(linears) if linears is not None else None)


def _check_phases(kind: str, steps: tuple) -> None:
    """real* quantize? integer*: the integer steps need the quantizer before them, per matmul and layer set."""
    order = [PHASES.index(s.defn.phase) for s in steps]
    if order != sorted(order):
        raise ValueError(f"[[{kind}]]: steps run real, then one quantize, then integer ({[s.defn.name for s in steps]})")
    if sum(s.defn.phase == "quantize" for s in steps if s.layers is None and s.linears is None) > 1:
        raise ValueError(f"[[{kind}]]: one quantizer")
    if any(s.defn.phase == "integer" for s in steps) and not any(s.defn.phase == "quantize" for s in steps):
        raise ValueError(f"[[{kind}]]: an integer step needs a quantizer")


def parse(data: dict, name: str) -> Arm:
    import steps  # noqa: F401  (registers the built-in steps)
    unknown = set(data) - {"description", "linear", "attention"}
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
    lin = tuple(parse_step("linear", t) for t in data.get("linear", []))
    att = tuple(parse_step("attention", t) for t in data.get("attention", []))
    _check_phases("linear", lin)
    _check_phases("attention", att)
    arm = Arm(name, lin, att, data.get("description", ""))
    return Arm(arm.name, arm.linear, arm.attention, arm.description, plain=PLAIN.get(arm.scope, ""))


def load(ref: str) -> Arm:
    """An arm from a TOML file, or by name from specs/ (`adopted` is specs/adopted.toml)."""
    path = Path(ref)
    if not path.suffix:
        path = SPECS / f"{ref}.toml"
    return parse(tomllib.loads(path.read_text()), path.stem)


#: the plain int8 arm each scope is reported against: every matmul in the arm's scope on the int8 grid, rounded to one
#: scale per row, rotated, nothing else
PLAIN = {"linears": "int8-linears", "attention": "int8-attention", "all": "int8-all"}
BF16 = Arm("bf16", description="the reference: bf16 linears, PyTorch SDPA attention")


def catalogue() -> dict:
    import steps  # noqa: F401
    out = {}
    for kind, reg in REGISTRY.items():
        for n, d in sorted(reg.items()):
            out[f"{kind}.{n}"] = {"phase": d.phase, "on": list(d.on),
                                  "params": {p: (None if v is inspect.Parameter.empty else v) for p, v in d.params.items()},
                                  "doc": inspect.getdoc(d.fn) or "", "source": d.source}
    return out
