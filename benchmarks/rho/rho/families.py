"""Algorithm families: the training algorithms we know, each parameterized, each giving a `Profile` at an instance.

A family is a kind (`full`, ...) with parameters, one table `family.<name>` in `data/families.toml`. `profile(model,
run)` fills the fields the arguments read, from the family's parameters and the instance's architecture.
`efficiency.<run kind>` is the compute it needs to reach that kind of target, in multiples of the reference algorithm's;
a kind it can't reach is `inf`, which the family table may leave out.

Adding a family kind is one dataclass with `profile` and an entry in `KINDS`; adding a family of an existing kind is a
table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from rho.instances import Model, Run
from rho.params import Registry
from rho.profile import Profile


class Family(Protocol):
    name: str

    def profile(self, model: Model, run: Run) -> Profile: ...


@dataclass(frozen=True)
class Full:
    """Every weight trained: pretraining, full fine-tuning, full-weight RL. The state is the whole model."""

    name: str
    kappa: float
    #: bits per parameter of the trained state a block must hold to train unthrottled
    state_bits_per_param: float
    b: float | None = None
    b_w: float | None = None
    b_a: float | None = None
    f: float | None = None
    width: float | None = None
    beta_x: float | None = None
    per_param: float | None = None
    bcol: float | None = None
    thread_d: int | None = None
    thread_e: float | None = None
    thread_c: float | None = None
    efficiency: Mapping[str, float] = field(default_factory=dict)

    def profile(self, model: Model, run: Run) -> Profile:
        return Profile(
            name=self.name, kappa=self.kappa, state_bits=model.n_total * self.state_bits_per_param,
            efficiency=self.efficiency.get(run.kind, math.inf), b=self.b, b_w=self.b_w, b_a=self.b_a, f=self.f,
            width=self.width, beta_x=self.beta_x, legs=model.legs, fanout=model.fanout, per_param=self.per_param, vocab=model.vocab,
            bcol=self.bcol, thread_d=self.thread_d, thread_e=self.thread_e, thread_c=self.thread_c)


KINDS: Mapping[str, type] = {"full": Full}


def family(registry: Registry, name: str) -> Family:
    prefix = f"family.{name}"
    kind = registry[f"{prefix}.kind"]
    efficiency = registry.table(f"{prefix}.efficiency") if "efficiency" in registry.names(prefix) else {}
    return registry.fields(KINDS[kind], prefix, efficiency=efficiency)
