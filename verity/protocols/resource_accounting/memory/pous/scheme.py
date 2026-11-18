"""PoUS and the memory wipe as resource schemes (`verity.protocols.resource_accounting.scheme`), registered as ``pous``
and ``wipe``.

**PoUS** is decided by `PousCheck.RunProtocol.protocol` (`verifier/check/PousVerifier/RunProtocol.lean`), whose
`Core.Account` row is memory held in the GPUs' HBM, verified from below. It adds no block to the registered circuit yet:
its setup unit, `PousP2SetupCheck_v1` (`prover/setup_circuit.py`), is in no window, and what it certifies is a share ρ of
the store `C*`, not a cost per block, so it states what it requires of the proof layer instead (`requires`). Its
certificate is P2's at the store's block count (`verifier.period.certificate`): up to `2^23` blocks
`Pous.SecurityProofs.P2SlackFamilyUncond`, in abstract M1 with no named assumption; above, `periodHeldAt`, which takes
`P2RingRoundSecureWide` and `RingDeviceAt`. Onsite's soundness takes PoUS's target, `PousSound`, as a premise. A device's
whole chip and its side channels' capacity over a round are counted in the state the ring game bounds
(`ResourceAccounting.RealizesState`); the help `perfect-isolation` admits is a known charge nothing bounds yet, so
`bound` refuses PoUS. Its own term is `Pous.Params.εMax` (`POUS_TERM`). Its sides: over is the credited store,
`⌊ρ·|C|⌋` bits with ρ = 18/19 (`Scheme.stateBound`); under is the bits the device holds. On a served run
(`prover/served.py`) it reads its settings from the declared build (``declared``; the auditor's coin source is the
launcher's, never a setting), the registered weights once resident
(``memory``), and computes each linear's block with the weight decoded from `C*` (``blocks``), under the codec
its ``option`` names (``pous:p2-16448/v4``, also the vLLM adapter's protocol option).

**The wipe** is decided by `MemoryChallenger.WipeProtocol.protocol` (`attester/lean/MemoryAttester/WipeProtocol.lean`),
whose row bounds the foreign bits a server may still hold after an accepted wipe, from above: a device's game with no
block in the circuit and no adapter option. `Pous.SecurityProofs.StreamedWipe` is proved with no named assumption;
onsite's soundness takes the wipe's target, `WipeSound`, as a premise. Its sides: over is the prior content's bits kept,
under is zero. Its own term is unsaid: `Pous.Guarantees.StreamedWipe`'s compression term `k·C(N,g+1)·2^(M+E)/2^(q+(g+1)b)`
has no Lean constant of its own (`StreamedWipe.bound` adds the sampling term `(g/N)^k`), and `Adv_PRF`
(`StreamedWipe.prfAdvantage`) has no value, so no `Term` evaluates it.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from verity.commitments import domains
from verity.protocol import VERIFIER, Channel
from verity.protocols.resource_accounting.memory.pous.attester import challenger_session as CS
from verity.protocols.resource_accounting.memory.pous.prover import workload
from verity.protocols.resource_accounting.memory.pous.verifier import period, wipe_period
from verity.protocols.resource_accounting.memory.pous.verifier import wipe_parties as WP
from verity.protocols.resource_accounting.scheme import Charge, Lacks, Launch, Order, Reading, Scheme, register

TRUSTED_DEVICE = "Verity.Assumptions.TrustedDevice"
#: PoUS's own term, `δscheme`: P2's ideal-model ε, `Pous.Params.εMax` = 2^-λ at λ = `Pous.Params.secParam` = 128, beside
#: the audit's own δ = `Pous.Params.δ` = 1/100, which is the sample's
POUS_TERM = "Pous.Params.εMax"
#: what PoUS's help is bounded by once something bounds it: `perfect-isolation` (`schemes.p2.P2_CLAIMS`) has no Lean name
HELP_BOUND = "perfect-isolation (a claim of schemes.p2.P2_CLAIMS; no Lean statement bounds the help yet)"
#: the party that runs PoUS's and the wipe's challenges, on the auditor's host
MEMORY_ATTESTER = WP.CHALLENGER
#: the wipe's own term, unsaid, and why
WIPE_TERM = Lacks("Pous.Guarantees.StreamedWipe's compression term k·C(N,g+1)·2^(M+E)/2^(q+(g+1)b) has no Lean constant "
                  "of its own, and Adv_PRF (StreamedWipe.prfAdvantage) has no value; neither is a Verity.Scheme "
                  "instance yet (its incorrect labels depend on the revealed history)")
#: what a wipe's accept certifies when its public says ``certified``, and when it doesn't: the served wipe is
#: `WipeProtocol.uncounted`, whose ``foreign_bits`` is ``memory_bits`` with ε 0, so its row holds at every δ by the run
WIPE_CERTIFIES = "at most foreign_bits bits of the prior content survive the wipe, except with eps"
WIPE_UNCERTIFIED = ("not certified: survivors bounded by nothing (foreign_bits is M, all of the node's memory; the wipe "
                    "counted no egress, so its eps of 0 is that bound's)")


def _pous_requests(ctx: Any) -> Any:
    from verity.protocols.resource_accounting.memory.pous.prover import served
    return served.requests(ctx)


def _pous_check_settings(values: Any) -> str | None:
    from verity.protocols.resource_accounting.memory.pous.prover import served
    return served.check_settings(values)


def _pous_attach(exchange: Any, ctx: Any) -> Any:
    from verity.protocols.resource_accounting.memory.pous.prover import served
    return served.attach(exchange, ctx, POUS.option.partition(":")[2])


def _wipe_attach() -> Any:
    raise LookupError("wipe attaches through no hook of a served run: it runs before the workload's engine loads, as "
                      "its own session with the memory attester (`verifier.wipe_parties`), started by its piece")


def _pous_reading(row: Mapping[str, Any]) -> Reading:
    q = row["quantity"]
    return Reading(int(q["s_bits"]), Fraction(str(q["eps"])), "quantity.eps", {"certified": bool(q["certified"])})


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


POUS = register(Scheme(
    name="pous",
    messages=workload.MESSAGES,
    protocol=domains.POUS_P2_RUN,
    resource="memory",
    blocks=Lacks("the memory attester's session answers and checks its challenges outside the workload's circuit: no "
                 "block of the circuit carries its cost"),
    assumptions=("Verity.Vocab.PousSound", TRUSTED_DEVICE, "Pous.P2.Assumptions.P2RingRoundSecureWide",
                 "Pous.Period.RingDeviceAt"),
    log=(Order("weights", "salt"), Order("store", "challenges")),
    charges=(Charge("chip", "ResourceAccounting.RealizesState"),
             Charge("side-channels", "ResourceAccounting.RealizesState"),
             Charge("help", HELP_BOUND, bounded=False)),
    settings={"challenges": 320},
    check_settings=_pous_check_settings,
    option="pous:p2-16448/v4",
    hooks=("declared", "memory", "blocks"),
    requires={
        "one-labelling": "C*'s own tree over the memory attester's row digests (`VerifierKey.from_row_digests`); "
                         "`PousP2SetupCheck_v1` is in no window, and setup's tie decodes C* in Python each period",
        "log-order": "its own salt, `setup.sample_salt(source, commitment)` after the commitment to W, and the memory "
                     "attester's coin tree, its session header bound to setup's root",
        "log-times": "the memory attester's own clock: each answer within Δ + RTT of its index, in the rounds it signs",
        "coins-at-rate": "the memory attester's coin tree, committed before the session",
        "one-design": "its own: `challenges` indices of C*'s blocks, uniform, revealed one at a time",
    },
    delta_scheme=Fraction(1, 2**128),
    over="the credited store: ⌊ρ·|C|⌋ bits of C*, ρ = 18/19 (`Pous.Scheme.stateBound` at `Pous.Params.ρ`)",
    under="the bits the device holds through the period, in bits",
    requests=_pous_requests,
    attach=_pous_attach,
    parties=(*CS.MANIFEST.parties, VERIFIER),
    channels=(*CS.MANIFEST.channels, Channel(MEMORY_ATTESTER, VERIFIER.name, ("SignedRun",), after=("End",))),
    channel_schemas=CS.MANIFEST.messages,
    colocated=(),
    needs=frozenset({"store"}),
    depends=(),
    pieces={"prover": Launch(optional=("address", "host", "command"), development="address")},
    record="space",
    symbol="S0",
    published=("bits", Fraction(1)),
    check=period.answer,
    reader=_pous_reading,
    premises=(),
    **dict.fromkeys(("tables", "output_format", "product_family", "y_committed", "salt_open_root", "kernels"),
                    Lacks("not a kernel scheme")),
))

WIPE = register(Scheme(
    name="wipe",
    protocol=domains.WIPE_PROTOCOL,
    resource="memory",
    blocks=Lacks("the wipe runs before the engine loads, outside the workload's circuit: no block of the circuit "
                 "carries its cost"),
    assumptions=("Verity.Vocab.WipeSound", TRUSTED_DEVICE),
    log=(Order("fill", "reveal"),),
    charges=(),
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
    attach=lambda exchange, ctx: _wipe_attach(),
    parties=(*WP.MANIFEST.parties, VERIFIER),
    channels=(*WP.MANIFEST.channels, Channel(MEMORY_ATTESTER, VERIFIER.name, ("WipeReport",), after=("End",))),
    channel_schemas=WP.MANIFEST.messages,
    colocated=(),
    needs=frozenset(),
    depends=(),
    pieces={"prover": Launch(optional=("address", "host", "command"), development="address")},
    record="wipe",
    symbol="foreign_bits",
    published=("bits", Fraction(1)),
    check=wipe_period.answer,
    reader=_wipe_reading,
    premises=("Verity.Vocab.WipeSound",),
    **dict.fromkeys(("tables", "output_format", "product_family", "y_committed", "salt_open_root", "kernels"),
                    Lacks("not a kernel scheme")),
))

__all__ = ["POUS", "WIPE"]
