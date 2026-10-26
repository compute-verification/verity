"""Concrete instances: model architectures and the training runs whose target the adversary wants to reach.

A `Model` carries what its architecture fixes for the arguments (sizes, and `Keyed`'s token structure: legs, fan-out,
vocabulary). A `Run` is a target: train `model` on `tokens` tokens, which the reference algorithm does in
`compute = 6 · n_active · tokens` FLOPs unless the run says otherwise. Families read both (`families.Family.profile`).
"""

from __future__ import annotations

from dataclasses import dataclass

from rho.params import Registry


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


@dataclass(frozen=True)
class Run:
    name: str
    model: str
    #: `pretrain`, `post-train`, ...: which families can reach the target, and at what efficiency
    kind: str
    tokens: float
    #: the reference algorithm's FLOPs to reach the target; 0 means `6 · n_active · tokens`
    compute: float = 0.0

    def target(self, model: Model) -> float:
        return self.compute or 6 * model.n_active * self.tokens


def model(registry: Registry, name: str) -> Model:
    return registry.fields(Model, f"model.{name}")


def run(registry: Registry, name: str) -> Run:
    return registry.fields(Run, f"run.{name}")
