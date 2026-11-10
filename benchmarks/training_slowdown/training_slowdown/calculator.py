"""The slowdown calculator (`benchmarks/rho`) as training's slowdown code reads it: the one module here that names it.

What the calculator states is read from it through its public functions, never restated here:

- a fixed policy's numbers (`policy`, `policies`): the `[policy.*]` tables of its registry, Γ in FLOP and K in bits;
- the compute a figure is presented over (`fleets`): its `periods.FLEETS`, one datacenter and the world, each a fleet
  of its registry, in AND gates a second;
- the monitoring periods (`periods`): its `PERIODS`, core's, with the window of the census's reference deployment
  before them;
- the bound of record at a policy (`bound`): the lower bound on a circuit's slowdown that `Nci.Cut.Guarantees.HE`
  proves, as the calculator prices it; where the calculator has no function that gives it, why not.

The calculator counts in FLOP and in its own gates (`units.gates_per_flop`); training's slowdown code counts in the
gates of the circuits' Definitions (`training_slowdown.gates`). A figure crosses in FLOP, two a MAC, never in gates.
"""
from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

from .gates import per_and, per_mac

BENCHMARKS = Path(__file__).resolve().parents[2]
CALCULATOR = BENCHMARKS / "rho"
DEPLOYMENTS = BENCHMARKS / "census" / "deployments.json"
#: the precision the registry's locations state their peak rate at: the census's dense FP8 E4M3
FLEET_PRECISION = "e4m3"
#: the statement the bound of record instantiates, as the lock names it (`HEShared` where the products share arrays)
THEOREM = "Nci.Cut.Guarantees.HE"


@cache
def _calculator():
    """The calculator's `periods` (its fleets), `rho.envelopes`, `rho.hardware` and `rho.params`, and its registry. Its
    directory leads the import path, as its own runner puts it: `periods` imports its sibling `run` by that name."""
    if sys.path[0] != str(CALCULATOR):
        sys.path.insert(0, str(CALCULATOR))
    import periods
    from rho import envelopes, hardware, params

    return periods, envelopes, hardware, params, params.Registry.load()


def policies() -> list[str]:
    """The names of the registry's fixed policies."""
    return list(_calculator()[4].names("policy"))


def policy(name: str) -> dict | None:
    """The registry's policy `name`: Γ in FLOP (`gamma_flop`) and K in bits (`k_bits`); None where it has none."""
    _, envelopes, _, _, registry = _calculator()
    if name not in registry.names("policy"):
        return None
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


def bound(c_gates: float, k_bits: float, work: dict | None) -> dict:
    """The bound of record at the policy (c, k) on a circuit's `work` (the theorem's N, O, β_w, β_x, b_y and F on M,
    and |C|, as `formalism.he_products` reads them from a traced step): not `available`, with exactly what is missing,
    while the calculator has no function that gives it."""
    _calculator()
    from rho import classes

    citing = [name for name, a in classes.CLASSES.items() if THEOREM in a.cites]
    return {"available": False, "theorem": THEOREM, "c_gates": c_gates, "k_bits": k_bits, "why": (
        f"not available: {CALCULATOR.name} has no function from a policy (c, k) and a circuit's N, O and |C| to the "
        f"bound {THEOREM} proves, and "
        + (f"its class {', '.join(citing)} cites it" if citing else
           f"its classes ({', '.join(classes.CLASSES)}) hold none that cites it"))}
