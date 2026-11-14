"""block-certifier: certifier/v0's period answer as the block record's certifier entry (``legs.certifier`` of
``verity/block-record/v2``).

* `block_leg`: a period's accepted answer as the block record's certifier entry: K
  in bits, its error the answer's exact ``eps``, and the certificate at δ = 1/100 and 10⁻⁶ wherever ``eps`` is at most
  δ (`BLOCK_DELTAS`).
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.protocol import DELTAS

PROTOCOL = "verity/certifier/v0"
#: the block record's owner by default: the lane that runs the network's sessions (``site.json``'s ``lane``)
OWNER: str = json.loads(
    (Path(__file__).resolve().parents[2] / "benchmarks/network_traces/site.json").read_text())["lane"]
#: the block record's δs (``verity/block-record/v2``'s ``at``): a leg's certificate holds at δ where its error is at
#: most δ. A period's answer at δ has error δ when each of its terms was decided at its share of δ.
BLOCK_DELTAS = DELTAS
#: where the block record says a leg's error is read from: the runner row's ``quantity`` (the answer's public)
BLOCK_ERROR_FROM = "quantity.eps"


def block_leg(answer: Mapping[str, Any], *, server: str, runs: Sequence[str], source: str = "real",
              owner: str | None = OWNER) -> dict[str, Any]:
    """The block record's certifier leg (``legs.certifier`` of ``verity/block-record/v2``) of one period's
    ``network-period check`` answer: the answer's verdict, reason, commitment and
    public; the quantity K, the bits a covert sender in ``server`` can move over the link in the period's ``windows``
    execution phases; the error the answer's exact ``eps``; and the certificate at each δ of
    `BLOCK_DELTAS`, K wherever ``eps`` ≤ δ, else not established. ``runs``: the run ids it comes from; ``source``:
    ``real`` or ``fixture``. Only an accepted answer certifies a K, so any other is a ValueError: the record's
    quantity has no shape for a period that certifies nothing."""
    if answer.get("protocol") != PROTOCOL:
        raise ValueError(f"protocol {answer.get('protocol')!r}: not {PROTOCOL}")
    if answer.get("verdict") != "accept":
        raise ValueError(f"a {answer.get('verdict')} period certifies no K ({answer.get('reason')})")
    if source not in ("real", "fixture"):
        raise ValueError(f"source {source!r}: real or fixture")
    if not runs:
        raise ValueError("a leg comes from at least one run")
    pub = answer["public"]
    K, eps = int(pub["K"]), Fraction(pub["eps"])
    at = {str(d): {"holds": True, "value": K, "epsilon": None, "by": "run"} if eps <= d else
          {"holds": False, "why": f"the run established it except with {eps}, more than {d}"} for d in BLOCK_DELTAS}
    return {"protocol": PROTOCOL, "owner": owner, "server": server, "source": source,
            "answer": {k: answer.get(k) for k in ("verdict", "reason", "commitment", "public")},
            "quantity": {"symbol": "K", "value": K, "unit": "bits", "windows": int(pub["windows"])},
            "error": str(eps), "error_from": BLOCK_ERROR_FROM, "at": at, "runs": list(runs)}


__all__ = ["BLOCK_DELTAS", "BLOCK_ERROR_FROM", "PROTOCOL", "block_leg"]
