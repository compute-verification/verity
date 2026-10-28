"""The network certifier's leak bounds as rho reads them: K, the bits a covert sender in a server moves through the
certifier's accepted transcripts over a monitoring period, per link and per datacenter. Network-accounting's
`benchmarks/network_traces/leak_bounds.py` is their one calculator (`per_link`, `datacenter`); this module only builds
its link from a regime's certifier and adds the datacenter's own bound.

* **Per link** (`per_link`): one network certifier on one server's uplink, `NetTiming.Guarantees.PeriodLeakAccepted`'s
  count (or, at a δ, `HorizonLeakAccepted`'s pooled form).
* **Per datacenter** (`datacenter`): the sum over the fleet's links, each at δ / links (network-accounting's
  composition, which no Lean statement covers), or the datacenter's own bound where it is tighter (Daniel, 27 Oct 7:44
  PM PDT: the leak target is both per link and per datacenter, and the slowdown is derived from both).

The link is the certifier's declared class (`Certifier.declared`, the warden form's public parameters), certifying
the requests its lanes admit: `lanes ⌈T / (D + 1)⌉` a link-window of `T τ_b` seconds, the class's own rate, so K is
never extrapolated past what the class carries.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

#: network-accounting's calculator of K
SOURCE = Path(__file__).resolve().parents[2] / "network_traces" / "leak_bounds.py"
READINGS = ("registered", "anchors")


@cache
def calculator() -> ModuleType:
    """`leak_bounds`, loaded by path: `benchmarks/network_traces` is not a package."""
    spec = importlib.util.spec_from_file_location("network_traces_leak_bounds", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Leak(NamedTuple):
    """K over a monitoring period, bits: per link, the sum over the datacenter's links, and the datacenter bound the
    slowdown reads (the tighter of the sum and the datacenter's own)."""
    per_link: float
    links: float
    total: float
    datacenter: float
    statement: str

    def json(self) -> dict[str, Any]:
        return self._asdict()


def link(certifier: Any) -> Any:
    """The certifier's declared class as one link of `leak_bounds`, at the class's own rate."""
    c = certifier
    IS = calculator().IS
    scan = IS.Class(B=int(c.cell_bytes), r=int(c.cell_rate), lanes=int(c.lanes), C=int(c.cells), IDB=int(c.id_bits),
                    D=int(c.hold), KO=int(c.outputs))
    T = int(c.window_buckets)
    rate = c.lanes * math.ceil(T / (c.hold + 1)) / (T * c.bucket_s)
    return calculator().link(scan, rate, T=T, tau=c.bucket_s)


def bounds(certifier: Any, links: float, seconds: float) -> Leak:
    """K over `seconds` per link and over `links` identical links, at the certifier's reading and δ (nan: the
    per-period form), and the datacenter bound: the sum, or `certifier.leak_datacenter` where tighter.

    The sum is `leak_bounds.datacenter`'s over `links` copies of one link, `links` times one link's K at δ / links,
    taken through `Link.bits` since a world fleet's tens of millions of links make the list itself too large."""
    lb = calculator()
    one = link(certifier)
    delta = None if math.isnan(certifier.leak_delta) else certifier.leak_delta
    per = lb.per_link(one.scan, one.rate, seconds, certifier.leak, delta, T=one.T, tau=one.tau).bits
    n = round(links)
    if n < 1:
        raise ValueError(f"a datacenter has at least one link, not {links!r}")
    total = n * one.bits(seconds, certifier.leak, None if delta is None else delta / n)
    return Leak(per, float(n), total, min(total, certifier.leak_datacenter), lb.STATEMENTS["datacenter"])
