"""Training steps measured as Boolean circuits, beside the FLOP model the overhead is otherwise in.

A `Circuit` (`data/circuits.toml`) is one optimizer step of a run (`instances.Run`) built as a Boolean circuit: the
Definition it is, the circuits census's count of its gates, ANDs and counted products (`products_steps` of them in dot
steps, the K steps of a split-K or batched GEMM, and `padded` of those padding), and `circuit_properties`' measure of
those products: `s_x`, the fewest gates in a counted product's subcircuit (`s_x_unsplit` outside the dot steps), `a_out`
and `a_in`. Its `structure` (`data/exclusion.toml`) is `s_x` as the uncertified cap's `s_×`, on the design's basis.

`compare` sets it beside the FLOP model at a family's `κ`: `units.gates_per_flop` against the step's gates (and ANDs)
over its run's FLOPs, and the reference's gates per counted product, `2G / κ`, which `structure.reference` takes as
`s_×`, against `s_x`. `in_place` is the registry with the step's design in place of the FLOP model: its structure the
default (`structure.default`) and `units.gates_per_flop` its gates over its FLOPs on its basis, so a scenario's bound
holds a gate at `u/w = 2G / (κ s_x)` of the reference's weight, both from one source (`classes.Structure.check`).
"""

from __future__ import annotations

from dataclasses import dataclass

from rho import instances
from rho.classes import structure
from rho.families import family, layered
from rho.hardware import Units
from rho.params import Param, Registry


@dataclass(frozen=True)
class Circuit:
    name: str
    #: the Definition, at its size
    definition: str
    #: the run it is one step of, and the structure its `s_x` is
    run: str
    structure: str
    batch: int
    seq_len: int
    gates: float
    ands: float
    #: counted products: the census's served multiply-adds
    products: float
    #: those in dot steps, and of them the split-K routes' padded lanes
    products_steps: float
    padded: float
    #: the fewest gates in a counted product's subcircuit, and in one outside the dot steps
    s_x: float
    s_x_unsplit: float
    #: the most counted products one activation value feeds, and the fewest summed into one output
    a_out: float
    a_in: float


def circuit(registry: Registry, name: str) -> Circuit:
    return registry.fields(Circuit, f"circuit.{name}")


def flops(registry: Registry, c: Circuit) -> float:
    """The step's FLOPs in the FLOP model: its run's target (`6 · n_active · tokens`, attention's products left out)."""
    run = instances.run(registry, c.run)
    return run.target(instances.model(registry, run.model))


def units(registry: Registry, c: Circuit) -> Units:
    """The design's own conversion: the step's gates over its FLOPs, on its structure's basis."""
    return Units(c.gates / flops(registry, c), structure(registry, c.structure).basis)


@dataclass(frozen=True)
class Comparison:
    circuit: str
    family: str
    flops: float
    #: the FLOP model's `units.gates_per_flop`, and the step's gates and ANDs over its FLOPs
    model_gates_per_flop: float
    gates_per_flop: float
    ands_per_flop: float
    #: the family's `κ`, and the step's: twice the products `κ` counts (`families.layered`) over its FLOPs
    kappa: float
    kappa_step: float
    #: the step's gates and ANDs per counted product (every one the census counts)
    gates_per_product: float
    ands_per_product: float
    #: `s_×`: `structure.reference`'s `2G / κ`, and measured, over every counted product and outside the dot steps
    reference_s_times: float
    s_x: float
    s_x_unsplit: float
    #: the counted products' share in dot steps
    steps_share: float

    @property
    def gates_ratio(self) -> float:
        """The step's gates over the FLOP model's: `G_step / G`."""
        return self.gates_per_flop / self.model_gates_per_flop

    @property
    def ands_ratio(self) -> float:
        return self.ands_per_flop / self.model_gates_per_flop

    @property
    def step_s_times(self) -> float:
        """`2G / κ` on the step: its gates per product `κ` counts."""
        return 2 * self.gates_per_flop / self.kappa_step

    def u_over_w(self, s_times: float) -> float:
        """`u/w = 2G / (κ s_×)` at the step's own `G`: what a gate holds, in the reference's weight per gate."""
        return 2 * self.gates_per_flop / (self.kappa * s_times)


def compare(registry: Registry, name: str, family_name: str = "full") -> Comparison:
    c = circuit(registry, name)
    run = instances.run(registry, c.run)
    model = instances.model(registry, run.model)
    f, g = flops(registry, c), registry["units.gates_per_flop"]
    kappa = family(registry, family_name).profile(model, run).kappa
    return Comparison(
        name, family_name, f, g, c.gates / f, c.ands / f, kappa, 2 * layered(model, 2)[0] * run.tokens / f,
        c.gates / c.products, c.ands / c.products, 2 * g / kappa, c.s_x, c.s_x_unsplit, c.products_steps / c.products)


def in_place(registry: Registry, name: str) -> Registry:
    """`registry` with circuit `name`'s design in place of the FLOP model: its structure every scenario's that names
    none, and `units.gates_per_flop` the step's gates over its FLOPs (no range: it is the design's), on its basis."""
    c = circuit(registry, name)
    u = units(registry, c)
    g = Param("units.gates_per_flop", u.gates_per_flop, "gate/FLOP",
              source=f"circuit.{name}: its gates over its run's FLOPs (rho/circuits.py units)", status="derived")
    return registry.override({"structure.default": c.structure, "units.gates_per_flop": g, "units.basis": u.basis})
