"""Concrete instances: model architectures and the training runs whose target the adversary wants to reach.

A `Model` carries what its architecture fixes for the arguments: sizes, `Keyed`'s token structure (legs, fan-out,
vocabulary) and its weight matrices by group, from which a family that trains part of the model (an adapter, a subset)
derives its state and its counted products. A `Run` is a target: train `model` on `tokens` tokens, which the reference
algorithm does in `compute = 6 · n_active · tokens` FLOPs unless the run says otherwise, of a `kind` in `RUN_KINDS`.
Families read both (`families.Family.profile`).

`legs` and `fanout` are derived from the matrices, the same way for every architecture (`derived_legs`,
`derived_fanout`):

* `legs` (`Keyed`'s `d`): the narrowest contraction among the counted products, the least `d_in` of a layer's matrices
  (Kimi K2's 512 is MLA's kv latent into kv_b; a dense model's is `d_model`).
* `fanout` (`aOut`): the most counted products one activation value reaches, the larger of the widest matrix output
  (one value of a matrix's input meets `d_out` products) and the MLP input, which feeds the router and every active
  expert's (or the dense MLP's) gate and up: `E + 2 f_e (k + s)` for an MoE of `E` experts of width `f_e` with `k`
  active and `s` shared (Kimi K2: 384 + 2·2048·9 = 37,248), `2 f` for a dense SwiGLU MLP of width `f`, `f` for GELU.
  The LM head's forward products are left out, as `Keyed` leaves them out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from rho.params import Registry

#: What a run's target is, so which families reach it and at what efficiency (`efficiency.<kind>` in families.toml).
RUN_KINDS: Mapping[str, str] = {
    "pretrain": "train from scratch to a loss",
    "sft": "supervised fine-tuning from a registered base (data-heavy; its delta isn't low-rank)",
    "rl-elicit": "RL whose gain the base already holds: math RLVR, format and sharpening (TVI A2)",
    "rl-agentic": "frontier agentic RL that adds skill (no small-state result above 32B)",
}

#: groups whose matrices repeat per layer (what a partial family's layer fraction cuts)
LAYERED = ("attention", "dense", "experts", "shared", "router")


@dataclass(frozen=True)
class Matrix:
    """One weight matrix of a group: `d_in × d_out`, `copies` of it in the model and `active` of them per token."""

    name: str
    d_in: float
    d_out: float
    copies: float
    active: float

    @property
    def size(self) -> float:
        return self.d_in * self.d_out


@dataclass(frozen=True)
class Model:
    name: str
    #: parameters, all and active per token
    n_total: float
    n_active: float
    d_model: int
    layers: int
    vocab: int
    #: `Keyed`: legs of a token's accumulation chain (`d`) and counted products per activation value (`aOut`)
    legs: int
    fanout: float
    #: weight matrices by group: `attention`, `dense` (the MLP of dense layers), `experts` (routed), `shared`, `router`,
    #: `head` (the LM head; the embedding is a lookup, `vocab · d_model` more parameters unless tied to the head)
    matrices: Mapping[str, tuple[Matrix, ...]] = field(default_factory=dict)
    tied_embeddings: bool = False

    def group(self, groups: Iterable[str] = ("all",)) -> tuple[Matrix, ...]:
        """The matrices of the named groups that the model has (`all`: every group)."""
        names = list(self.matrices) if "all" in groups else [g for g in groups if g in self.matrices]
        return tuple(m for g in names for m in self.matrices[g])

    def params(self, groups: Iterable[str] = ("all",)) -> float:
        return sum(m.size * m.copies for m in self.group(groups))

    def active(self, groups: Iterable[str] = ("all",)) -> float:
        return sum(m.size * m.active for m in self.group(groups))


@dataclass(frozen=True)
class Run:
    name: str
    model: str
    #: one of `RUN_KINDS`: which families can reach the target, and at what efficiency
    kind: str
    tokens: float = 0.0
    #: the reference algorithm's FLOPs to reach the target; 0 means `6 · n_active · tokens`
    compute: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in RUN_KINDS:
            raise ValueError(f"run {self.name}: kind {self.kind!r} is not one of {sorted(RUN_KINDS)}")
        if self.tokens <= 0 and self.compute <= 0:
            raise ValueError(f"run {self.name}: give tokens or compute")

    def target(self, model: Model) -> float:
        return self.compute or 6 * model.n_active * self.tokens


def derived_legs(m: Model) -> int:
    """`legs` from the matrices: the least `d_in` among the layers' matrices."""
    return int(min(x.d_in for x in m.group(LAYERED)))


def derived_fanout(m: Model) -> float:
    """`fanout` from the matrices: the widest layer matrix's `d_out`, or the MLP input's reach if larger (every matrix
    reading `d_model` in a dense MLP, or in an MoE layer's router, shared and active experts, per layer)."""
    reach = [max(x.d_out for x in m.group(LAYERED))]
    if "dense" in m.matrices:
        n = min(x.active for x in m.matrices["dense"])
        reach.append(sum(x.d_out * x.active for x in m.matrices["dense"] if x.d_in == m.d_model) / n)
    if "router" in m.matrices:
        n = m.matrices["router"][0].active
        reach.append(sum(x.d_out * x.active for x in m.group(("experts", "shared", "router")) if x.d_in == m.d_model) / n)
    return max(reach)


def model(registry: Registry, name: str) -> Model:
    prefix = f"model.{name}"
    groups = registry.table(f"{prefix}.matrices") if "matrices" in registry.names(prefix) else {}
    matrices = {g: tuple(Matrix(*row) for row in rows) for g, rows in groups.items()}
    return registry.fields(Model, prefix, matrices=matrices)


def run(registry: Registry, name: str) -> Run:
    return registry.fields(Run, f"run.{name}")
