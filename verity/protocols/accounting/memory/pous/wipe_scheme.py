"""The memory wipe as a resource scheme (`verity.protocols.accounting.scheme`), registered as ``wipe``.

It is decided by `MemoryChallenger.WipeProtocol.protocol`
(`verity/protocols/accounting/memory/pous/attester/lean/MemoryAttester/WipeProtocol.lean`), whose row bounds the foreign
bits a server may still hold after an accepted wipe, from above: a device's game with no block in the circuit and no
adapter option. `Pous.SecurityProofs.StreamedWipe` is proved with no named assumption; onsite's soundness takes the
wipe's target, `WipeSound`, as a premise. Its sides: over is the prior content's bits kept, under is zero. Its own term
is unsaid: `Pous.Guarantees.StreamedWipe`'s compression term `k·C(N,g+1)·2^(M+E)/2^(q+(g+1)b)` has no Lean constant of
its own (`StreamedWipe.bound` adds the sampling term `(g/N)^k`), and `Adv_PRF` (`StreamedWipe.prfAdvantage`) has no
value, so no `Term` evaluates it. Its settings are `period.json`'s ``wipe`` and the challenger's binding, and its attach
(`verity.protocols.accounting.memory.pous.prover.wipe_served`) runs the whole wipe before the model loads. Its memory
attester, its development flags and the onsite command it launches are PoUS's
(`verity.protocols.accounting.memory.pous.scheme`).
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from verity.commitments import domains
from verity.protocol import VERIFIER, Channel
from verity.protocols.accounting.memory.pous.attester import memory_attester as MA
from verity.protocols.accounting.memory.pous.scheme import (DEVELOPMENT, MEMORY_ATTESTER, RUN_KEYS, TRUSTED_DEVICE,
                                                            onsite_command)
from verity.protocols.accounting.memory.pous.verifier import period, wipe_period
from verity.protocols.accounting.memory.pous.verifier import wipe_parties as WP
from verity.protocols.accounting.scheme import Executable, Lacks, Launch, Order, Reading, Scheme, register

#: the wipe's own term, unsaid, and why
WIPE_TERM = Lacks("Pous.Guarantees.StreamedWipe's compression term k·C(N,g+1)·2^(M+E)/2^(q+(g+1)b) has no Lean constant "
                  "of its own, and Adv_PRF (StreamedWipe.prfAdvantage) has no value; neither is a Verity.Scheme "
                  "instance yet (its incorrect labels depend on the revealed history)")
#: what a wipe's accept certifies when its public says ``certified``, and when it doesn't: the served wipe is
#: `WipeProtocol.uncounted`, whose ``foreign_bits`` is ``memory_bits`` with ε 0
WIPE_CERTIFIES = ("the wipe's judge computed at most foreign_bits surviving bits except with eps, from the memory sizes "
                  "the prover declared, which nothing checks against the node")
WIPE_UNCERTIFIED = ("not certified: survivors bounded by nothing (foreign_bits is M, all of the node's memory; the wipe "
                    "counted no egress, so its eps of 0 is that bound's)")


def _wipe_reading(row: Mapping[str, Any]) -> Reading:
    """A wipe's ``k`` is its public's, else, in a wipe from before `verity/wipe/v0` published it, its profile's; and
    ``certifies``, what its accept certifies in words (`WIPE_CERTIFIES`, `WIPE_UNCERTIFIED`)."""
    a = row["answer"]
    pub = a.get("public") or {}
    certified = bool(pub["certified"])
    return Reading(int(pub["foreign_bits"]), Fraction(str(pub["eps"])), "answer.public.eps",
                   {"k": pub.get("k", (a.get("profile") or {}).get("k")), "r": pub.get("r"),
                    "M": int(pub["memory_bits"]), "certified": certified,
                    "certifies": WIPE_CERTIFIES if certified else WIPE_UNCERTIFIED})


def _wipe_check_settings(values: Any) -> str | None:
    from verity.protocols.accounting.memory.pous.prover import wipe_served
    return wipe_served.check_settings(values)


def _wipe_attach(exchange: Any, ctx: Any) -> Any:
    from verity.protocols.accounting.memory.pous.prover import wipe_served
    return wipe_served.attach(exchange, {**ctx, "party": MEMORY_ATTESTER, "development_of": period.development_of})


WIPE = register(Scheme(
    name=WP.COMPONENT,
    protocol=domains.WIPE_PROTOCOL,
    resource="memory",
    blocks=Lacks("the wipe runs before the engine loads, outside the workload's circuit: no block of the circuit "
                 "carries its cost"),
    assumptions=("Verity.Vocab.WipeSound", TRUSTED_DEVICE),
    log=(Order("fill", "reveal"),),
    charges=(),
    # loading the declaration loads no prover file: test_wipe_served holds these to wipe_run.CONFIG and wipe_served.UNSET
    settings={"store": "cuda", "segments": 0, "reserve_bytes": 1 << 30, "tier": "", "limit_ns": wipe_period.LIMIT_NS,
              "reads": 0},
    check_settings=_wipe_check_settings,
    attach=_wipe_attach,
    over="the prior content's bits kept through an accepted wipe, in bits",
    under="zero bits: nothing is credited for what is kept",
    requires={
        "one-labelling": "its own labels, SHA-512 under a key fresh for the server and the wipe, derived by the memory "
                         "attester",
        "log-order": "the memory attester's own: it flips its coins once the fill is complete",
        "log-times": "the memory attester's own clock: a deadline per batch, in the session it signs",
        "coins-at-rate": "the memory attester's own coins",
        "one-design": "its own: `k` indices of the fill, revealed one at a time",
    },
    delta_scheme=WIPE_TERM,
    parties=(*WP.MANIFEST.parties, VERIFIER),
    channels=(*WP.MANIFEST.channels, Channel(MEMORY_ATTESTER, VERIFIER.name, ("WipeReport",), after=("End",))),
    channel_schemas=WP.MANIFEST.messages,
    colocated=(),
    needs=frozenset(),
    depends=(),
    pieces={"prover": Launch(optional=("address", "host", "command"), development="address",
                            entry="verity.protocols.accounting.memory.pous.wipe_scheme:launch_wipe",
                            executables=(Executable("verity/protocols/accounting/memory/pous/attester/lean", "wipe-protocol"),))},
    answers=Lacks("its case is answered by the composite's case writer, not yet by functions of this package"),
    unchecked=MA.UNCHECKED,
    record="wipe",
    # the wipe holds labels from a key fresh for the wipe, and closes its store and the card's memory before the attach
    # returns: it touches no weight and keeps nothing resident
    keeps_weight_copy=False,
    symbol="foreign_bits",
    published=("bits", Fraction(1)),
    check=wipe_period.answer,
    reader=_wipe_reading,
    premises=("Verity.Vocab.WipeSound",),
    **dict.fromkeys(("tables", "output_format", "product_family", "y_committed", "salt_open_root", "kernels"),
                    Lacks("not a kernel scheme")),
))

WIPE_SCRIPT = "benchmarks/pous/streamed_wipe/served_wipe.sh"
#: served_wipe.sh's variable for each binding of the wipe's memory attester
_WIPE = {"address": "WIPE_ADDRESS", "host": "WIPE_CHALLENGER", "command": "WIPE_COMMAND"}


def _wipe_hooks(cfg, params, bound, pinned):
    """The streamed wipe of the card before the workload's engine loads (WIPE_CMD, served_wipe.sh): the tree's command
    (`onsite_command`). The key is the configuration's pin alone, as WIPE_KEY_SHA512, which the runner sets over the
    node's environment, so neither the environment nor a binding gives another; with no pin served_wipe.sh refuses.
    Unbound, the key is the run's (`RUN_KEYS`, the tree's command's one other form), and no pin is passed."""
    if "address" not in bound:
        return {"WIPE_CMD": f"{onsite_command('WIPE_CMD')} {RUN_KEYS}", **DEVELOPMENT}
    return {"WIPE_CMD": onsite_command("WIPE_CMD"), "WIPE_KEY_SHA512": "".join(pinned),
            **{v: bound[k] for k, v in _WIPE.items() if k in bound}}


def launch_wipe(declaration: Scheme) -> tuple[Any, tuple[str, ...]]:
    """The wipe's prover piece (its `Launch.entry`): its hooks and the script they run. It launches and collects, and
    decides no verdict."""
    return _wipe_hooks, (WIPE_SCRIPT,)


__all__ = ["WIPE", "WIPE_SCRIPT", "launch_wipe"]
