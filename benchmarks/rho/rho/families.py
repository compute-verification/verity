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
* `nci`: any algorithm of the `benchmarks/nci` library (`benchmarks/nci/algorithms.json`) from its traits; `nci:<id>` names one with no
  table.

A family may name a **precision** (`precision.<name>`, FP8 or NVFP4): the format sets `β_x`, caps every per-activation
bit count and the tile's weight width at its width, and multiplies the compute a target needs. A family's
`premises = "<family>"` takes the premise numbers it doesn't give from that family (`family`).

Each kind says what it trains (`Trained`), and `Premises.premised` derives the fields that read the instance's shape
from it, the same for every kind:

* the counted sets: `kappa_forward = κ · forward / counted` (`benchmarks/nci/kappa.py`: `κ` counts the forward and, at
  `c = 2`, the Dgrad of the active weights outside the embedding and the LM head, and the LM head's Dgrad), and
  `kappa_wgrad` the same set's weight gradients (none without a backward);
* depth: `layers` the layers the trained set spans, `layer_weights` per layer (`instances.layer_weights`), `layer_bits`
  the trained parameters' update at `b_w` over the layers (a state that doesn't split, over the layers whole);
* the shard: `token_state_bits` a token's trained weights at `b_w` and, with a backward, a `grad_bits` accumulator
  each; `shared_state_bits` the same over `instances.shared_run`; `flops_per_token = 2 counted / κ`; `cut_bits` the
  residual width at `b_a`, crossed twice with a backward; `token_bits` the premise's per data token;
* `entry_width` the legs, `map_width` and `map_width_min` the residual width, and the tile's matrix from
  `instances.tile_shape`.

The premise numbers no shape gives (`c_w`, `c_a`, the prices, E1's rates, the tile's widths, `thread_k`) are the
family's table's, mostly `family.full`'s; a run that states its batch sets `step_tokens`.

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
from rho.instances import LAYERED, Matrix, Model, Run, Size
from rho.params import Registry
from rho.profile import Profile

ALGORITHMS = Path(__file__).resolve().parents[2] / "nci" / "algorithms.json"
#: gates per weight to regenerate one seeded value and add it (`benchmarks/nci/kappa.py` SEED, `algorithms.json` es-30's
#: 150,000 per parameter over 15,000 perturbations)
SEED = 10


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


@dataclass(frozen=True)
class Trained:
    """What a family trains of an instance, for the fields that read its shape: `params` parameters, `per_token` of them
    a token uses, its part of each matrix (`size`), the `layers` it spans, the products per token `κ` counts (`counted`)
    and their forward ones, each (weight, token) pair once (`forward`); whether its state splits across shards (else
    `layer_bits` and the token's state are the whole state), a `decode` cost, a `backward` (gradient accumulators, a
    second crossing), the data tokens per token (`data`), the tile's matrix where `instances.tile_shape` isn't it, and
    whether the first layers are trained (`instances.shared_run`'s first run)."""

    params: float
    per_token: float
    size: Size
    layers: int
    counted: float
    forward: float
    splits: bool = True
    decode: float = 0.0
    backward: bool = True
    data: float = 1.0
    tile: tuple[float, float] | None = None
    prefix: bool = True


def layered(model: Model, c: float, size: Size = instances.whole) -> tuple[float, float]:
    """`counted` and `forward` per token for weights `size` of the layers' matrices: `c` products per weight (forward,
    and Dgrad at `c = 2`), plus the LM head's Dgrad at `c = 2` where the head is trained (`benchmarks/nci/kappa.py`)."""
    n = sum(size(m) * m.active for m in model.group(LAYERED))
    head = sum(size(m) * m.active for m in model.group(["head"]))
    return c * n + (c - 1) * head, n


@dataclass(frozen=True, kw_only=True)
class Premises:
    """What every kind shares: the premise numbers it gives the classes (named as in `Profile`; `None` puts the algorithm
    outside every class that reads it), its efficiency per run kind, and its precision. `grad_bits` is the width of the
    gradient accumulator a block hands on per updated weight inside an optimizer step (`Trained.backward`)."""

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
    thread_k: float | None = None
    price: float | None = None
    price_max: float | None = None
    gamma: float | None = None
    beta_v: float | None = None
    h_eff: float | None = None
    b_ps: float | None = None
    step_tokens: float | None = None
    beta_x_marginal: float | None = None
    psum_bits: float | None = None
    tile_weight_bits: float | None = None
    token_bits: float | None = None
    grad_bits: float | None = None
    efficiency: Mapping[str, float] = field(default_factory=dict)
    precision: Precision | None = None

    def m(self, run: Run, efficiency: Mapping[str, float] | None = None) -> float:
        m = (self.efficiency if efficiency is None else efficiency).get(run.kind, math.inf)
        return m * self.precision.efficiency.get(run.kind, math.inf) if self.precision else m

    def premised(self, model: Model, run: Run, *, kappa: float, state_bits: float, trained: Trained | None = None,
                 legs: int | None = None, fanout: float | None = None, efficiency: Mapping[str, float] | None = None,
                 **given: Any) -> Profile:
        """The profile with this family's premises, the instance's token structure unless the kind gives its own, the
        precision's `β_x` and widths applied, and the shape fields derived from `trained` (the module's docstring)."""
        p = self.precision
        fan = model.fanout if fanout is None else fanout
        values = {f.name: getattr(self, f.name) for f in fields(Profile) if hasattr(self, f.name)}
        values |= {"thread_e": self.thread_e or fan, "step_tokens": run.step_tokens or self.step_tokens}
        if p:
            values |= {"beta_x": p.beta_x, "b_a": _cap(self.b_a, p.bits), "f": _cap(self.f, p.bits),
                       "tile_weight_bits": _cap(self.tile_weight_bits, p.bits)}
        values |= {"name": self.name, "kappa": kappa, "state_bits": state_bits, "efficiency": self.m(run, efficiency),
                   "legs": model.legs if legs is None else legs, "fanout": fan, "vocab": model.vocab} | given
        if trained is not None and model.matrices:
            values |= _shaped(model, values, trained, self.grad_bits) | given
        return Profile(**values)


def _shaped(model: Model, v: Mapping[str, Any], t: Trained, grad_bits: float | None) -> dict[str, Any]:
    kappa, b_w, state = v["kappa"], v["b_w"], v["state_bits"]
    rate = None if b_w is None else b_w + ((grad_bits or 0.0) if t.backward else 0.0)
    if t.splits and rate is not None:
        token, run = t.per_token * rate, instances.shared_run(model, t.size, prefix=t.prefix)
        shared, layer_bits = token if run is None else run * rate, b_w * t.params / t.layers
    else:
        token = shared = state
        layer_bits = state / t.layers
    rows, cols = t.tile or instances.tile_shape(model, t.size)
    return {
        "kappa_forward": kappa * t.forward / t.counted, "kappa_wgrad": kappa * t.forward / t.counted if t.backward else None,
        "layers": t.layers, "layer_weights": instances.layer_weights(model, t.size), "layer_bits": layer_bits,
        "entry_width": v["legs"], "map_width": model.d_model, "map_width_min": model.d_model,
        "token_state_bits": token, "shared_state_bits": shared, "flops_per_token": 2 * t.counted / kappa,
        "cut_bits": None if v["b_a"] is None else (2 if t.backward else 1) * model.d_model * v["b_a"],
        "token_bits": None if v["token_bits"] is None else v["token_bits"] * t.data, "decode_flops": t.decode,
        "state_splits": t.splits, "tile_inputs": rows, "tile_outputs": cols}


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

    def trained(self, model: Model) -> Trained:
        counted, forward = layered(model, self.per_param or 2)
        return Trained(model.n_total, model.n_active, instances.whole, model.layers, counted, forward)

    def profile(self, model: Model, run: Run) -> Profile:
        return self.premised(model, run, kappa=self.kappa, state_bits=model.n_total * self.state_bits_per_param,
                             trained=self.trained(model))


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
    Its shape fields are its own: `layer_bits` and the token's state are the adapter's weights at `b_w`, a tile reads an
    `A` (`d_in × r`, or a core's `r × r`), and a tied adapter's state is its `params` values, which no shard splits
    (each matrix's core regenerated from them, `SEED` gates an entry and a multiply-add, is its decode cost)."""

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

    def size(self, model: Model) -> Size:
        """The adapter's weights on each matrix (0 off its groups); a tied adapter's per-matrix core."""
        on = {id(m) for m in model.group(self.groups)}
        r = self.rank
        per = (lambda m: r * (m.d_in + m.d_out)) if self.form == "lora" else (lambda m: r * r)
        return lambda m: per(m) if id(m) in on else 0.0

    def profile(self, model: Model, run: Run) -> Profile:
        params, per_token, fan = self.sizes(model)
        c = self.per_param or 2
        kappa = self.kappa if self.kappa is not None else kappa_of(model, per_token, c=c, overhead=self.overhead)
        size, r, tied = self.size(model), self.rank, self.form == "tied"
        rows = instances.tile_shape(model, size)[0] if self.form == "lora" else r
        copies = sum(m.copies for m in model.group(self.groups))
        trained = Trained(params, per_token, size, instances.spanned(model, self.groups), c * per_token, per_token,
                          splits=not tied, decode=(SEED + 2) * r * r * params * copies if tied else 0.0,
                          tile=(rows, r))
        lw = {"layer_weights": min(params, instances.layer_weights(model, size))} if tied else {}
        return self.premised(model, run, kappa=kappa, state_bits=params * self.bits + self.seed_bits, trained=trained,
                             legs=self.rank, fanout=fan, b=None, bcol=None, thread_d=self.rank, **lw)


@dataclass(frozen=True, kw_only=True)
class Seed(Premises):
    """Evolution strategies and zeroth-order methods (ES, MeZO): every weight moves along seeded noise, so the update is
    `steps · population` scalars of `bits` each (the seeds follow a counter). A block that reads that description rebuilds
    the weights by regenerating every perturbation, so the state is the description, and no class prices a weight
    (`b`, `b_w`, `bcol` and threads' `d` unknown, as `algorithms.json` has `theorem_applies = false` for them). The
    description doesn't split across shards, decoding it costs `SEED` gates per weight per perturbation, and with no
    backward there are no weight gradients and one crossing; `forwards` is the forward passes per token (MeZO's two)."""

    kappa: float
    steps: float
    population: float
    bits: float = 16.0
    forwards: float = 1.0

    def profile(self, model: Model, run: Run) -> Profile:
        state = self.steps * self.population * self.bits
        n = layered(model, 1)[1]
        trained = Trained(model.n_total, model.n_active, instances.whole, model.layers, self.forwards * n, n,
                          splits=False, decode=SEED * self.steps * self.population * model.n_total, backward=False)
        return self.premised(model, run, kappa=self.kappa, state_bits=state, trained=trained,
                             b=None, b_w=None, bcol=None, thread_d=None)


@dataclass(frozen=True, kw_only=True)
class Partial(Premises):
    """A subset trained, the rest frozen and registered: the matrices of `groups`, in the last `layer_fraction` of the
    layers (a frozen prefix, which the backward never reaches), with `expert_fraction` of the routed experts carrying
    `slot_fraction` of a token's routed slots (ESFT), and the LM head if `head`. The state is the subset at
    `state_bits_per_param`. Legs and fan-out stay the whole model's, which only loosens the bounds; the frozen
    embedding gives no `ColumnsIncompressible` (`bcol`). Its depth is the trained layers' (a frozen prefix's inputs come
    from registered weights, as layer 0's from token ids), and its shard state the subset's."""

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
        c = self.per_param or 2
        kappa = self.kappa if self.kappa is not None else kappa_of(model, active, c=c, through=through,
                                                                  overhead=self.overhead)
        on, routed = ({id(m) for m in model.group(g)} for g in (self.groups, ["experts"]))

        def size(m: Matrix) -> float:
            return m.size * (self.slot_fraction if id(m) in routed else 1) if id(m) in on else 0.0

        layers = max(1, round(self.layer_fraction * instances.spanned(model, self.groups)))
        trained = Trained(total, active, size, layers, c * active, active,
                          prefix=self.layer_fraction >= 1)
        return self.premised(model, run, kappa=kappa, state_bits=total * self.state_bits_per_param, trained=trained,
                             bcol=None)


@dataclass(frozen=True, kw_only=True)
class Distill(Premises):
    """The model trained in full on a teacher's outputs: its next-token distributions (pretraining distillation, reaching
    only targets above the teacher's loss) or its sampled traces (in place of RL). The state is the student's, as
    `Full`'s. A `teacher` run online is registered work: its forward adds `overhead · 2 N_teacher` gates per token and no
    counted products. A sunk teacher (none) adds nothing. Its shape fields are `Full`'s, with the teacher's outputs as
    data: `data_tokens_per_token` data tokens per token trained (a sunk teacher's distributions)."""

    kappa: float
    state_bits_per_param: float
    teacher: Model | None = None
    overhead: float = 1.1
    data_tokens_per_token: float = 1.0

    def profile(self, model: Model, run: Run) -> Profile:
        kappa = self.kappa
        if self.teacher is not None:
            student = 2 * (self.per_param or 2) * model.n_active / self.kappa
            kappa = self.kappa * student / (student + self.overhead * 2 * self.teacher.n_active)
        counted, forward = layered(model, self.per_param or 2)
        trained = Trained(model.n_total, model.n_active, instances.whole, model.layers, counted, forward,
                          data=self.data_tokens_per_token)
        return self.premised(model, run, kappa=kappa, state_bits=model.n_total * self.state_bits_per_param,
                             trained=trained)


@cache
def algorithms() -> Mapping[str, dict]:
    return {a["id"]: a for a in json.loads(ALGORITHMS.read_text())}


@dataclass(frozen=True, kw_only=True)
class Nci(Premises):
    """An algorithm of the `benchmarks/nci` library (`benchmarks/nci/algorithms.json`), from its traits under `library.py`'s attacker
    convention (the rate a block pays per weight and per boundary entry capped at the literature's central rates,
    `nci.rate_cap` and `nci.boundary_bits_cap`), with the carried state capped as `Full`'s (`nci.state_bits_cap`):

    * the state is `state_bits_per_param · updated_fraction · n_total`, or `description_bits` when it doesn't split;
    * `κ`, `c` (`uses_per_thread`), legs and threads' `d` (`entry_fee_override`), fan-out and threads' `e`
      (`fanout_override`) and `b_a` (`boundary_bits`) are its traits; `b = b_w` is its state rate, unknown when the
      state doesn't split; a trait `theorem_applies = false` leaves threads' `d` unknown;
    * `efficiency.pretrain` is `1 / multiplier` (the library's multiplier is compute-equivalent against dense AdamW
      pretraining, so one that changes the architecture, such as fine-grained MoE's 20, prices a target loss and not a
      given model's), and the family's table may add or override any kind;
    * the shard's fields are `library.py`'s `profile`: `flops_per_token_per_param · n_active`, the cut
      `boundary_width_fraction · d_model · b_a · crossings`, `data_tokens_per_token` data tokens a token, the decode
      `decode_flops_per_param · n_total`, a backward with two crossings; `kappa_forward` is `κ / c`, and the depth
      fields are the model's on its `updated_fraction`.

    The premises the traits don't give come from the family's table, or `family.full`'s for `nci:<id>`."""

    algorithm: str
    state_bits_cap: float = math.inf
    rate_cap: float = math.inf
    boundary_bits_cap: float = math.inf

    def profile(self, model: Model, run: Run) -> Profile:
        a = algorithms()[self.algorithm]
        t = a["traits"]
        uf, c = t["updated_fraction"], t["uses_per_thread"]
        carried = min(t["state_bits_per_param"], self.state_bits_cap)
        state = carried * uf * model.n_total if t["state_splits"] else t["description_bits"]
        hard = min(t["state_bits_per_param"], self.rate_cap) if t["state_splits"] else None
        multiplier = (a.get("multiplier") or {}).get("value")
        efficiency = ({"pretrain": 1 / multiplier} if multiplier else {}) | dict(self.efficiency)
        b_a = min(t["boundary_bits"], self.boundary_bits_cap)
        counted = t["kappa"] * t["flops_per_token_per_param"] * model.n_active / 2
        trained = Trained(uf * model.n_total, uf * model.n_active, lambda m: uf * m.size, model.layers, counted,
                          counted / c, splits=t["state_splits"], decode=t["decode_flops_per_param"] * model.n_total,
                          backward=t["crossings"] >= 2, data=t["data_tokens_per_token"])
        return self.premised(
            model, run, kappa=t["kappa"], state_bits=state, trained=trained,
            legs=t["entry_fee_override"] or model.legs, fanout=t["fanout_override"] or model.fanout,
            efficiency=efficiency, b=hard, b_w=hard, b_a=b_a, per_param=c,
            thread_d=(t["entry_fee_override"] or self.thread_d) if t["theorem_applies"] else None,
            thread_e=t["fanout_override"] or model.fanout, thread_c=c,
            cut_bits=t["boundary_width_fraction"] * model.d_model * b_a * t["crossings"])


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
    caps = {k: registry[f"nci.{k}"] for k in ("state_bits_cap", "rate_cap", "boundary_bits_cap")}
    values = caps | {"algorithm": name.removeprefix("nci:")} | dict(values)
    if values["algorithm"] not in algorithms():
        raise KeyError(f"{name}: no algorithm {values['algorithm']!r} in {ALGORITHMS}")
    return Nci(**_checked(Nci, f"family.{name}", values) | {"name": name})
