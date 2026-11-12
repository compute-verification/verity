"""The slowdown calculator (`verity/protocols/inference_only/rho`) as this package reads it: the one module here that names it.

What the calculator states is read from it through its public functions, never restated here:

- a fixed policy's numbers (`policy`, `policies`): the `[policy.*]` tables of its registry, Γ in FLOP and K in bits;
- the compute a figure is presented over (`fleets`): its `periods.FLEETS`, one datacenter and the world, each a fleet
  of its registry, in AND gates a second;
- the monitoring periods (`periods`): its `PERIODS`, core's, with the window of the census's reference deployment
  before them;
- the bound of record at a policy (`bound`): the lower bound on a circuit's slowdown that the statement its
  `min_cut` names proves (`theorem`), from its `min_cut.bound`.

The calculator counts in FLOP and in its own gates (`units.gates_per_flop`); training's slowdown code counts in the
gates of the circuits' Definitions (`training_slowdown.gates`). A figure crosses in FLOP, two a MAC, never in gates.
"""
from __future__ import annotations

import json
import sys
import tomllib
from functools import cache
from pathlib import Path

from .gates import per_and, per_mac

#: the calculator's directory, whose name is its package's too
CALCULATOR = Path(__file__).resolve().parents[3] / "verity" / "protocols" / "inference_only" / "rho"
DEPLOYMENTS = Path(__file__).resolve().parents[2] / "census" / "deployments.json"
#: the precision the registry's locations state their peak rate at: the census's dense FP8 E4M3
FLEET_PRECISION = "e4m3"


@cache
def _calculator():
    """The calculator's `periods` (its fleets), its package's `envelopes`, `hardware` and `params`, and its registry.
    Its directory leads the import path, as its own runner puts it: `periods` imports its sibling `run` by that name."""
    if sys.path[0] != str(CALCULATOR):
        sys.path.insert(0, str(CALCULATOR))
    import periods
    from rho import envelopes, hardware, params

    return periods, envelopes, hardware, params, params.Registry.load()


def policies() -> list[str]:
    """The names of the registry's fixed policies: the `[policy.*]` tables of its data files, read without importing
    the calculator, since a job checks its config's policy by name and runs none of the calculator's code."""
    return sorted({name for f in (CALCULATOR / "data").glob("*.toml")
                   for name in tomllib.loads(f.read_text()).get("policy", {})})


def policy(name: str) -> dict | None:
    """The registry's policy `name`: Γ in FLOP (`gamma_flop`) and K in bits (`k_bits`); None where it has none."""
    if name not in policies():
        return None
    _, envelopes, _, _, registry = _calculator()
    p = envelopes.policy(registry, name)
    return {"gamma_flop": p.gamma, "k_bits": p.k}


def window() -> tuple[str, float]:
    """The census's reference deployment and its window in seconds."""
    d = next(d for d in json.loads(DEPLOYMENTS.read_text()) if d["role"] == "reference")
    return d["id"], float(d["window_s"])


def periods() -> dict[str, float]:
    """The periods a figure is presented over, in seconds: the reference deployment's window, then the calculator's
    monitoring periods."""
    return {"window": window()[1]} | {name: float(s) for name, s in _calculator()[3].PERIODS.items()}


def fleets() -> list[dict]:
    """The calculator's fleets (`periods.FLEETS`), each with its label, its locations and its compute in AND gates a
    second: its registry's peak FLOP a second, two a MAC, each MAC a GEMM's AND gates at `FLEET_PRECISION`."""
    mod, _, hardware, _, registry = _calculator()
    units, kind = hardware.units(registry), f"gemm/{FLEET_PRECISION}"
    out = []
    for name, label in mod.FLEETS:
        f = hardware.fleet(registry, name)
        flops = units.flops(f.count * hardware.location(registry, f.location).phi)
        out.append({"fleet": name, "over": label, "locations": f.count, "location": f.location, "flop_per_s": flops,
                    "and_per_s": flops / 2 * per_mac(kind) / per_and(kind)})
    return out


def _min_cut():
    """The calculator's module for the bound of record, its `min_cut`."""
    _calculator()
    from rho import min_cut

    return min_cut


def theorem() -> str:
    """The statement the bound of record instantiates, by the Lean name the calculator's `min_cut` gives it (its
    `HE`; `HEShared` where the products share arrays)."""
    return _min_cut().HE


def bound(c_gates: float, k_bits: float, work: dict | None) -> dict:
    """The bound of record at the policy (c, k) on a circuit's `work` (the theorem's N in MACs, O in wires, β_w, β_x,
    b_y and F on M, and |C| in gates, as `formalism.he_products` reads them from a traced step): what every partition
    of that circuit into the policy's blocks pays, the `slowdown` of the calculator's `min_cut.bound`, with its least
    `blocks`, its `statement` and what it `rests_on`. A policy here states no output bits a block, so the function's
    n is None. Not `available`, with what is missing, where the circuit has no `work`."""
    out = {"available": False, "theorem": theorem(), "c_gates": c_gates, "k_bits": k_bits}
    if work is None:
        return out | {"why": "not available: the bound reads a traced step's N and O on M, and the config's search "
                             "record gives none"}
    b = _min_cut().bound(c_gates, k_bits, None, N=work["N"], O=work["O"], size=work["C_gates"],
                         beta_w=work["beta_w"], beta_x=work["beta_x"], b=work["b_y"], F=work["F"])
    return out | {"available": True, "theorem": b.statement, "slowdown": b.slowdown, "blocks": b.blocks,
                  "rests_on": b.rests_on, "work": work}
