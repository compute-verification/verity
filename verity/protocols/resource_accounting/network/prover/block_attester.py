"""block-attester: certifier/v0's period answer as the block record's attester entry (``legs.attester`` of
``verity/block-record/v2``).

* `block_leg`: a period's accepted answer as the block record's attester entry: K
  in bits, its error the answer's exact ``eps``, and at δ = 1/100 and 10⁻⁶ (`BLOCK_DELTAS`) no attestation while the
  scheme's bound is refused, with the refusal's sentence as why.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from verity.protocol import DELTAS
from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.network.scheme import NETWORK_ATTESTER

PROTOCOL = "verity/certifier/v0"
#: the block record's δs (``verity/block-record/v2``'s ``at``)
BLOCK_DELTAS = DELTAS
#: where the block record says a leg's error is read from: the runner row's ``quantity`` (the answer's public)
BLOCK_ERROR_FROM = "quantity.eps"


def block_leg(answer: Mapping[str, Any], *, server: str, runs: Sequence[str], source: str = "real",
              owner: str | None = None) -> dict[str, Any]:
    """The block record's attester leg (``legs.attester`` of ``verity/block-record/v2``) of one period's
    ``network-period check`` answer: the answer's verdict, reason, commitment and
    public; the quantity K, the bits a covert sender in ``server`` can move over the link in the period's ``windows``
    execution phases; the error the answer's exact ``eps``; and at each δ of `BLOCK_DELTAS` not established, its
    why the sentence by which `S.bound` refuses the scheme (`_declared`). ``runs``: the run ids it comes from; ``source``:
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
    declared = _declared()
    at = {str(d): {"holds": False, "why": declared["unbounded"]["why"]} for d in BLOCK_DELTAS}
    return {"protocol": PROTOCOL, "owner": owner, "server": server, "source": source,
            "answer": {k: answer.get(k) for k in ("verdict", "reason", "commitment", "public")},
            "quantity": {"symbol": "K", "value": K, "unit": "bits", "windows": int(pub["windows"])},
            "error": str(eps), "error_from": BLOCK_ERROR_FROM, "at": at, "runs": list(runs)} | declared


def _declared() -> dict[str, Any]:
    """The leg's ``premises`` and ``unbounded`` as the block record writes them for a declared scheme, read from
    `NETWORK_ATTESTER`: its premises by name (or why its declaration does not say them), and `S.unbounded`'s refusal.
    A ValueError once the declaration bounds it, since the bound is then the answer's profile's, which no answer carries."""
    p = NETWORK_ATTESTER.premises
    refused = S.unbounded(NETWORK_ATTESTER)
    if refused is None:
        raise ValueError(f"{NETWORK_ATTESTER.name}'s declaration now bounds it: the record reads the bound from the "
                         "answer's profile (`bound`), so the answer carries those inputs and the leg reads them")
    return {"premises": p.why if isinstance(p, S.Lacks) else list(p),
            "unbounded": {"why": str(refused), "terms": list(refused.terms)}}


__all__ = ["BLOCK_DELTAS", "BLOCK_ERROR_FROM", "PROTOCOL", "block_leg"]
