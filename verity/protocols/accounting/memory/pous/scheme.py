"""PoUS as a resource scheme (`verity.protocols.accounting.scheme`), registered as ``pous``. The memory wipe's
(`verity.protocols.accounting.memory.pous.wipe_scheme`) reads its memory attester, development flags and onsite
command from here.

PoUS is decided by `PousCheck.RunProtocol.protocol` (`verifier/check/PousVerifier/RunProtocol.lean`), whose
`Core.Account` row is memory held in the GPUs' HBM, verified from below. Its blocks are the store's: the setup's C
(`prover/window.py`'s `Setup`) is one unit of `PousP2SetupCheck_v2` (`prover/setup_circuit.py`) a store block, each
costing the bits one store block holds (`P2Params.w`, the unit's static W), which the verifier reads from the registered
C; the session's units credit nothing. What its verdict certifies is still a share ρ of the store `C*` from its own
arithmetic (`verifier.period.certify`), and it states what it requires of the proof layer (`requires`). Its
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
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.commitments import domains
from verity.protocol import VERIFIER, Channel
from verity.protocols.accounting.memory.pous.attester import challenger_session as CS
from verity.protocols.accounting.memory.pous.attester import memory_attester as MA
from verity.protocols.accounting.memory.pous.prover import workload
from verity.protocols.accounting.memory.pous.verifier import period
from verity.protocols.accounting.memory.pous.verifier import wipe_parties as WP
from verity.protocols.accounting.memory.pous.verifier.schemes import p2 as P2
from verity.protocols.accounting.scheme import (Block, Charge, Executable, Lacks, Launch, Order, Reading, Scheme,
                                                         register)

TRUSTED_DEVICE = "Verity.Assumptions.TrustedDevice"
#: PoUS's own term, `δscheme`: P2's ideal-model ε, `Pous.Params.εMax` = 2^-λ at λ = `Pous.Params.secParam` = 128, beside
#: the audit's own δ = `Pous.Params.δ` = 1/100, which is the sample's
POUS_TERM = "Pous.Params.εMax"
#: what PoUS's help is bounded by once something bounds it: `perfect-isolation` (`schemes.p2.P2_CLAIMS`) has no Lean name
HELP_BOUND = "perfect-isolation (a claim of schemes.p2.P2_CLAIMS; no Lean statement bounds the help yet)"
#: the party that runs PoUS's and the wipe's challenges, on the auditor's host
MEMORY_ATTESTER = WP.CHALLENGER


def _pous_requests(ctx: Any) -> Any:
    from verity.protocols.accounting.memory.pous.prover import served
    return served.requests(ctx)


def _pous_check_settings(values: Any) -> str | None:
    from verity.protocols.accounting.memory.pous.prover import served
    return served.check_settings(values)


def _pous_attach(exchange: Any, ctx: Any) -> Any:
    from verity.protocols.accounting.memory.pous.prover import served
    return served.attach(exchange, ctx, POUS.option.partition(":")[2])


def _pous_reading(row: Mapping[str, Any]) -> Reading:
    q = row["quantity"]
    return Reading(int(q["s_bits"]), Fraction(str(q["eps"])), "quantity.eps", {"certified": bool(q["certified"])})


POUS = register(Scheme(
    name=CS.COMPONENT,
    messages=workload.MESSAGES,
    protocol=domains.POUS_P2_RUN,
    resource="memory",
    blocks=(Block("PousP2SetupCheck_v2", lambda shape: P2.P2Params(shape["W"], shape["C0"], 1).w),),
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
                         "`PousP2SetupCheck_v2` is in no window, and setup's tie decodes C* in Python each period",
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
    pieces={"prover": Launch(optional=("address", "host", "command"), development="address",
                            entry="verity.protocols.accounting.memory.pous.scheme:launch_pous",
                            executables=(Executable("verity/protocols/accounting/memory/pous/verifier/check", "pous-run-protocol"),))},
    reads={},
    answers=Lacks("its kept session is answered by the PoUS piece's own code, not yet by functions of this package"),
    unchecked=MA.UNCHECKED,
    record="space",
    symbol="S0",
    published=("bits", Fraction(1)),
    check=period.answer,
    reader=_pous_reading,
    premises=(),
    # Across forwards it holds C*, its own encoded store, and the decode's handle on C*'s rows; each registered weight
    # is released, and a weight decoded from C* for a block lives for that one call.
    keeps_weight_copy=False,
    **dict.fromkeys(("tables", "output_format", "product_family", "y_committed", "salt_open_root", "kernels"),
                    Lacks("not a kernel scheme")),
))


#: the scripts' one flag for a development run's attester: on the workload's machine, under a key made for the run
RUN_KEYS = "--run-keys"
DEVELOPMENT = {"DEVELOPMENT": "1"}
POUS_SCRIPT, ONSITE_PIECES = "benchmarks/pous/served_piece.sh", "benchmarks/pous/onsite_pieces.sh"
ATTESTER_SCRIPT = "benchmarks/pous/run_attester.sh"
#: served_piece.py's flag for each binding of PoUS's memory attester
_POUS = {"address": "--challenger-address", "host": "--challenger-host", "command": "--challenger-command"}


def onsite_command(name: str) -> str:
    """The tree's command for the onsite piece run through ``name`` (``WIPE_CMD``, ``POUS_CMD``, ``BANDWIDTH_CMD``):
    `ONSITE_PIECES`'s ``ONSITE_<name>``, the one place that names it, to which its ``onsite_piece_commands`` holds
    every caller's (any other value is refused by name). LookupError where the file names none."""
    text = (Path(__file__).resolve().parents[5] / ONSITE_PIECES).read_text()
    if not (m := re.search(rf"^ONSITE_{name}='([^']*)'$", text, re.M)):
        raise LookupError(f"{ONSITE_PIECES} names no ONSITE_{name}")
    return m.group(1)


def _pous_hooks(cfg, params, bound, pinned):
    """PoUS's prover after the workload (POUS_CMD, served_piece.sh): the memory attester's timed reads answered from
    the served weights held as P2's C* on the card, with the memory attester on the workload's machine in a development
    run. The memory attester's key is the configuration's pin alone, on the command line; with no pin the piece
    trusts no session. Unbound, the key is the run's (`RUN_KEYS`), and no pin is passed."""
    words = [*(["--trusted-sha512", *pinned] if pinned else []), *([] if "address" in bound else [RUN_KEYS]),
             "--rounds", str(params["challenges"]),
             *(w for k, flag in _POUS.items() if k in bound for w in (flag, bound[k]))]
    return {"POUS_CMD": onsite_command("POUS_CMD"), "POUS_ARGS": shlex.join(words), "ONSITE_PIECES": ONSITE_PIECES,
            **({} if "address" in bound else DEVELOPMENT)}


def launch_pous(declaration: Scheme) -> tuple[Any, tuple[str, ...]]:
    """PoUS's prover piece (its `Launch.entry`): its hooks and the scripts they run. It launches and collects, and
    decides no verdict."""
    return _pous_hooks, (POUS_SCRIPT, ONSITE_PIECES, ATTESTER_SCRIPT)


__all__ = ["ATTESTER_SCRIPT", "DEVELOPMENT", "MEMORY_ATTESTER", "ONSITE_PIECES", "POUS", "POUS_SCRIPT", "RUN_KEYS",
           "TRUSTED_DEVICE", "launch_pous", "onsite_command"]
