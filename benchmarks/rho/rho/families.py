"""Algorithm families: the training algorithms we know, each parameterized, each giving a `Profile` at an instance.

A family is a kind with parameters, one table `family.<name>` in `data/families.toml`. `profile(model, run)` fills the
fields the arguments read, from the family's parameters and the instance's architecture. `efficiency.<run kind>` (a kind
of `instances.RUN_KINDS`) is the compute it needs to reach that kind of target, in multiples of the reference
algorithm's; a kind it can't reach is `inf`, which the family table leaves out.

The kinds:

* `full`: every weight trained (pretraining, full fine-tuning, full-weight RL); the state is the whole model, and a
  weight costs a block the rate of a scratch-trained model or of a delta from the registered base.
* `adapter`: LoRA and kin beside frozen, registered weights (LoRA, LoRA-XS, TinyLoRA); state and counted products from
  the model's matrices.
* `seed`: evolution strategies and zeroth-order methods whose update is seeds and scalars.
* `low-comm`: `full` with an outer sync every `sync_every` steps of compressed pseudo-gradients (DiLoCo, SparseLoCo,
  DeMo).
* `partial`: a subset trained (a frozen prefix, a share of the experts), the rest registered.
* `distill`: `full` on a teacher's outputs, the teacher's forward registered work.
* `nci`: any algorithm of the NCI library (`benchmarks/nci/algorithms.json`) from its traits; `nci:<id>` names one with no
  table.

A family may name a **precision** (`precision.<name>`, FP8 or NVFP4): the format sets `β_x`, caps every per-activation
bit count at its width, and multiplies the compute a target needs. A family's `premises = "<family>"` takes the premise
numbers it doesn't give from that family (`family`).

Adding a family kind is one dataclass with `profile` and an entry in `KINDS`; adding a family of an existing kind is a
table.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import cache
from pathlib import Path
from typing import Any, Protocol

from rho import instances
from rho.instances import LAYERED, Model, Run
from rho.params import Registry
from rho.profile import Profile

ALGORITHMS = Path(__file__).resolve().parents[2] / "nci" / "algorithms.json"


class Family(Protocol):
    name: str

    def profile(self, model: Model, run: Run) -> Profile: ...


@dataclass(frozen=True)
class Precision:
    """A number format the whole computation runs in: the activation rate `β_x` a block pays in it, its width in bits per
    activation entry (which caps `b_a` and `f`), and the compute a target needs in it, per run kind, in multiples of the
    family's own (`inf` where unknown)."""

    name: str
    beta_x: float
    bits: float
    efficiency: Mapping[str, float] = field(default_factory=dict)


def kappa_of(model: Model, updated: float, *, c: float = 2, through: float | None = None, backward: bool = True,
             overhead: float = 1.1) -> float:
    """`κ = 2|M| / gates` per token. `|M| = c · updated`: the updated weights' forward products and, at `c = 2`, their
    Dgrad. The gates are `overhead · 2 (N + through + updated)`: every active weight's forward, the Dgrad through
    `through` active weights (all `N` unless the frozen part is a prefix the backward never reaches), the updated
    weights' Wgrad, and `overhead` for what isn't a weight matmul (attention, norms, the optimizer;
    `benchmarks/nci/kappa.py`: 1.11 for Kimi K2 at 4k context, 1.7 at 32k). Without a backward, `2 N` alone."""
    n = model.n_active
    gates = overhead * 2 * (n + (n if through is None else through) + updated) if backward else overhead * 2 * n
    return 2 * c * updated / gates


@dataclass(frozen=True, kw_only=True)
class Premises:
    """What every kind shares: the premise numbers it gives the classes (named as in `Profile`; `None` puts the algorithm
    outside every class that reads it), its efficiency per run kind, and its precision."""

    name: str
    b: float | None = None
    b_w: float | None = None
    c_w: float | None = None
    c_a: float | None = None
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
    precision: Precision | None = None

    def m(self, run: Run, efficiency: Mapping[str, float] | None = None) -> float:
        m = (self.efficiency if efficiency is None else efficiency).get(run.kind, math.inf)
        return m * self.precision.efficiency.get(run.kind, math.inf) if self.precision else m

    def premised(self, model: Model, run: Run, *, kappa: float, state_bits: float, legs: int | None = None,
                 fanout: float | None = None, efficiency: Mapping[str, float] | None = None, **given: Any) -> Profile:
        """The profile with this family's premises, the instance's token structure unless the kind gives its own, and
        the precision's `β_x` and width applied."""
        p = self.precision
        fan = model.fanout if fanout is None else fanout
        values = {f.name: getattr(self, f.name) for f in fields(Profile) if hasattr(self, f.name)}
        values |= {"thread_e": self.thread_e or fan}
        if p:
            values |= {"beta_x": p.beta_x, "b_a": _cap(self.b_a, p.bits), "f": _cap(self.f, p.bits)}
        values |= {"name": self.name, "kappa": kappa, "state_bits": state_bits, "efficiency": self.m(run, efficiency),
                   "legs": model.legs if legs is None else legs, "fanout": fan, "vocab": model.vocab} | given
        return Profile(**values)


def _cap(x: float | None, cap: float) -> float | None:
    return None if x is None else min(x, cap)


@dataclass(frozen=True, kw_only=True)
class Full(Premises):
    """Every weight trained: pretraining, full fine-tuning, full-weight RL. The state is the whole model at
    `state_bits_per_param`; what a block pays per weight (`b`, `b_w`) is the rate of a scratch-trained model
    (pretraining) or of its delta from the registered base (fine-tuning, RL), which the family's table gives."""

    kappa: float
    #: bits per parameter of the trained state a block must hold to train unthrottled (what an exact run carries)
    state_bits_per_param: float

    def profile(self, model: Model, run: Run) -> Profile:
        return self.premised(model, run, kappa=self.kappa, state_bits=model.n_total * self.state_bits_per_param)


@dataclass(frozen=True, kw_only=True)
class LowComm(Full):
    """`Full` with `workers` replicas that sync every `sync_every` steps by compressed pseudo-gradients of `sync_bits`
    bits per parameter (DiLoCo, SparseLoCo, DeMo at `sync_every = 1`). Inside a worker the blocks pass the same state and
    activations as `Full`'s, so the profile is `Full`'s and only the efficiency moves; what shrinks is the traffic between
    workers, `exchange_bits` per worker and step."""

    workers: float
    sync_every: float
    sync_bits: float

    def exchange_bits(self, model: Model) -> float:
        """Bits each worker receives per inner step: the whole model's pseudo-gradient, compressed, every `sync_every`
        steps."""
        return model.n_total * self.sync_bits / self.sync_every


@dataclass(frozen=True, kw_only=True)
class Adapter(Premises):
    """LoRA and kin: trainable factors beside frozen, registered weights, on the matrices of `groups`. The state is the
    adapter's `bits` per weight (plus `seed_bits` for seeded projections), and only its products are counted, so `κ` is
    tiny. `form`:

    * `lora`: `A` (`r × d_in`) and `B` (`d_out × r`) per matrix; an activation into `B` reaches `d_out` products, and a
      token's chain passes through the `r`-wide bottleneck (`legs = r`).
    * `xs` (LoRA-XS): an `r × r` core between frozen SVD factors of each matrix (`fanout = legs = r`).
    * `tied` (TinyLoRA): `params` values shared by every matrix, each matrix's `r × r` core a seeded projection of them.

    Whatever its table says, an adapter is outside `UpdateHardness` (`b`: it needs the weights to drift, and the
    registered base doesn't), its frozen embedding gives no `ColumnsIncompressible` (`bcol`), and threads' `d` is `r`.
    """

    form: str = "lora"
    rank: int = 1
    groups: tuple[str, ...] = ("attention", "dense", "experts", "shared")
    params: float = 0.0
    bits: float = 16.0
    seed_bits: float = 0.0
    overhead: float = 1.1
    kappa: float | None = None

    def sizes(self, model: Model) -> tuple[float, float, float]:
        """The adapter's parameters, its counted products' parameters per token, and the most counted products one
        activation value reaches."""
        ms = model.group(self.groups)
        if not ms:
            raise ValueError(f"{self.name}: {model.name} has no matrices in {list(self.groups)}")
        r = self.rank
        if self.form == "lora":
            return (sum(r * (m.d_in + m.d_out) * m.copies for m in ms),
                    sum(r * (m.d_in + m.d_out) * m.active for m in ms), max(m.d_out for m in ms))
        core = (sum(r * r * m.copies for m in ms), sum(r * r * m.active for m in ms), float(r))
        if self.form == "xs":
            return core
        if self.form == "tied":
            return self.params, core[1], core[2]
        raise ValueError(f"{self.name}: form {self.form!r} is not lora, xs or tied")

    def profile(self, model: Model, run: Run) -> Profile:
        params, per_token, fan = self.sizes(model)
        c = self.per_param or 2
        kappa = self.kappa if self.kappa is not None else kappa_of(model, per_token, c=c, overhead=self.overhead)
        return self.premised(model, run, kappa=kappa, state_bits=params * self.bits + self.seed_bits,
                             legs=self.rank, fanout=fan, b=None, bcol=None, thread_d=self.rank)


@dataclass(frozen=True, kw_only=True)
class Seed(Premises):
    """Evolution strategies and zeroth-order methods (ES, MeZO): every weight moves along seeded noise, so the update is
    `steps · population` scalars of `bits` each (the seeds follow a counter). A block that reads that description rebuilds
    the weights by regenerating every perturbation, so the state is the description, and no class prices a weight
    (`b`, `b_w`, `bcol` and threads' `d` unknown, as `algorithms.json` has `theorem_applies = false` for them)."""

    kappa: float
    steps: float
    population: float
    bits: float = 16.0

    def profile(self, model: Model, run: Run) -> Profile:
        return self.premised(model, run, kappa=self.kappa, state_bits=self.steps * self.population * self.bits,
                             b=None, b_w=None, bcol=None, thread_d=None)


@dataclass(frozen=True, kw_only=True)
class Partial(Premises):
    """A subset trained, the rest frozen and registered: the matrices of `groups`, in the last `layer_fraction` of the
    layers (a frozen prefix, which the backward never reaches), with `expert_fraction` of the routed experts carrying
    `slot_fraction` of a token's routed slots (ESFT), and the LM head if `head`. The state is the subset at
    `state_bits_per_param`. Legs and fan-out stay the whole model's, which only loosens the bounds; the frozen
    embedding gives no `ColumnsIncompressible` (`bcol`)."""

    state_bits_per_param: float
    groups: tuple[str, ...] = LAYERED
    layer_fraction: float = 1.0
    expert_fraction: float = 1.0
    slot_fraction: float = 1.0
    head: bool = False
    overhead: float = 1.1
    kappa: float | None = None

    def sizes(self, model: Model) -> tuple[float, float]:
        """Trained parameters, and trained parameters a token uses."""
        groups = [g for g in self.groups if g in model.matrices]
        if not groups:
            raise ValueError(f"{self.name}: {model.name} has no matrices in {list(self.groups)}")
        lf = self.layer_fraction
        total = sum(model.params([g]) * lf * (self.expert_fraction if g == "experts" else 1) for g in groups)
        active = sum(model.active([g]) * lf * (self.slot_fraction if g == "experts" else 1) for g in groups)
        if self.head:
            total, active = total + model.params(["head"]), active + model.active(["head"])
        return total, active

    def profile(self, model: Model, run: Run) -> Profile:
        total, active = self.sizes(model)
        through = None if self.layer_fraction >= 1 else self.layer_fraction * model.active(LAYERED) + model.active(["head"])
        kappa = self.kappa if self.kappa is not None else kappa_of(model, active, c=self.per_param or 2, through=through,
                                                                  overhead=self.overhead)
        return self.premised(model, run, kappa=kappa, state_bits=total * self.state_bits_per_param, bcol=None)


@dataclass(frozen=True, kw_only=True)
class Distill(Premises):
    """The model trained in full on a teacher's outputs: its next-token distributions (pretraining distillation, reaching
    only targets above the teacher's loss) or its sampled traces (in place of RL). The state is the student's, as
    `Full`'s. A `teacher` run online is registered work: its forward adds `overhead · 2 N_teacher` gates per token and no
    counted products. A sunk teacher (none) adds nothing."""

    kappa: float
    state_bits_per_param: float
    teacher: Model | None = None
    overhead: float = 1.1

    def profile(self, model: Model, run: Run) -> Profile:
        kappa = self.kappa
        if self.teacher is not None:
            student = 2 * (self.per_param or 2) * model.n_active / self.kappa
            kappa = self.kappa * student / (student + self.overhead * 2 * self.teacher.n_active)
        return self.premised(model, run, kappa=kappa, state_bits=model.n_total * self.state_bits_per_param)


@cache
def algorithms() -> Mapping[str, dict]:
    return {a["id"]: a for a in json.loads(ALGORITHMS.read_text())}


@dataclass(frozen=True, kw_only=True)
class Nci(Premises):
    """An algorithm of the NCI library (`benchmarks/nci/algorithms.json`), from its traits under `library.py`'s attacker
    convention (state and boundary bits capped at the literature's central rates, `nci.*_cap`):

    * the state is `state_bits_per_param · updated_fraction · n_total`, or `description_bits` when it doesn't split;
    * `κ`, `c` (`uses_per_thread`), legs and threads' `d` (`entry_fee_override`), fan-out and threads' `e`
      (`fanout_override`) and `b_a` (`boundary_bits`) are its traits; `b = b_w` is its state rate, unknown when the
      state doesn't split; a trait `theorem_applies = false` leaves threads' `d` unknown;
    * `efficiency.pretrain` is `1 / multiplier` (the library's multiplier is compute-equivalent against dense AdamW
      pretraining, so one that changes the architecture, such as fine-grained MoE's 20, prices a target loss and not a
      given model's), and the family's table may add or override any kind.

    The premises the traits don't give come from the family's table, or `family.full`'s for `nci:<id>`."""

    algorithm: str
    state_bits_cap: float = math.inf
    boundary_bits_cap: float = math.inf

    def profile(self, model: Model, run: Run) -> Profile:
        a = algorithms()[self.algorithm]
        t = a["traits"]
        rate = min(t["state_bits_per_param"], self.state_bits_cap)
        state = rate * t["updated_fraction"] * model.n_total if t["state_splits"] else t["description_bits"]
        hard = rate if t["state_splits"] else None
        multiplier = (a.get("multiplier") or {}).get("value")
        efficiency = ({"pretrain": 1 / multiplier} if multiplier else {}) | dict(self.efficiency)
        return self.premised(
            model, run, kappa=t["kappa"], state_bits=state, legs=t["entry_fee_override"] or model.legs,
            fanout=t["fanout_override"] or model.fanout, efficiency=efficiency, b=hard, b_w=hard,
            b_a=min(t["boundary_bits"], self.boundary_bits_cap), per_param=t["uses_per_thread"],
            thread_d=(t["entry_fee_override"] or self.thread_d) if t["theorem_applies"] else None,
            thread_e=t["fanout_override"] or model.fanout, thread_c=t["uses_per_thread"])


KINDS: Mapping[str, type] = {"full": Full, "adapter": Adapter, "seed": Seed, "low-comm": LowComm, "partial": Partial,
                             "distill": Distill, "nci": Nci}

#: the premise numbers a family takes from another family's table when it names one (`premises = "full"`)
PREMISES = frozenset(f.name for f in fields(Premises)) - {"name", "efficiency", "precision"}

#: what a kind takes from the registry by name besides numbers: another table it names
_REFS = {"precision": lambda r, n: precision(r, n), "teacher": lambda r, n: instances.model(r, n)}


def precision(registry: Registry, name: str) -> Precision:
    prefix = f"precision.{name}"
    return registry.fields(Precision, prefix, efficiency=registry.table(f"{prefix}.efficiency"))


def family(registry: Registry, name: str) -> Family:
    """The family named `name`: table `family.<name>`, or `nci:<id>`, the library's algorithm `id` on `family.full`'s
    premises. A table's `premises = "<family>"` fills the premise numbers (`PREMISES`) it doesn't give from that
    family's (and so on down its own `premises`), so sweeping a premise of `family.full` moves every family that takes
    it."""
    if name.startswith("nci:"):
        return _nci(registry, name, _premises(registry, {"premises": "full"}))
    prefix = f"family.{name}"
    table = registry.table(prefix)
    cls = KINDS[table["kind"]]
    values = _premises(registry, table) | {k: v for k, v in table.items() if k not in ("kind", "premises")}
    values |= {k: ref(registry, table[k]) for k, ref in _REFS.items() if k in table}
    if "efficiency" in registry.names(prefix):
        values["efficiency"] = registry.table(f"{prefix}.efficiency")
    if cls is Nci:
        return _nci(registry, name, values)
    return cls(**_checked(cls, prefix, values) | {"name": name})


def _premises(registry: Registry, table: Mapping[str, Any], seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    if "premises" not in table:
        return {}
    name = table["premises"]
    if name in seen:
        raise ValueError(f"family.{name}: its premises name it again")
    base = registry.table(f"family.{name}")
    return _premises(registry, base, seen | {name}) | {k: v for k, v in base.items() if k in PREMISES}


def _checked(cls: type, prefix: str, values: Mapping[str, Any]) -> Mapping[str, Any]:
    stray = set(values) - {f.name for f in fields(cls)}
    if stray:
        raise ValueError(f"{prefix}: {sorted(stray)} are not fields of {cls.__name__}")
    return values


def _nci(registry: Registry, name: str, values: Mapping[str, Any]) -> Nci:
    caps = {k: registry[f"nci.{k}"] for k in ("state_bits_cap", "boundary_bits_cap")}
    values = caps | {"algorithm": name.removeprefix("nci:")} | dict(values)
    if values["algorithm"] not in algorithms():
        raise KeyError(f"{name}: no algorithm {values['algorithm']!r} in {ALGORITHMS}")
    return Nci(**_checked(Nci, f"family.{name}", values) | {"name": name})
