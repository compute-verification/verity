"""onsite-record: the onsite demo's record of one served run (``record/onsite.json``, `FORMAT`): every verdict its run of
record needs, the network certifier's bound term by term, and that bound over each monitoring period for the serving
node, a datacenter and the world. It reads only the run's public $OUT and decides nothing:

    python benchmarks/one_stage/onsite_record.py OUT [--advice-charge FILE] [--to FILE]

`served_verdicts.sh` runs it after its composite step on a run with the network certifier's piece, and its one
``ONSITE`` line goes to the run's verdicts.txt.

**Verdicts** (`VERDICTS`), each ``{protocol, verdict, from}`` with the answer's ``reason`` or ``refusal``, or, where the
run wrote no answer, ``missing`` and ``why``:

* sampled proofs and PoUW: ``record/verdicts.json``'s ``integrity`` and ``work`` rows (`served_verdicts.py record`);
* inference-only: its case's outcome in remote's composite;
* remote: remote's Lean verdict, ``composite/remote/composite.json`` (`composite.py`), with what it couldn't establish;
* the network certifier: the ``communication`` row, `NetworkCheck.Period`'s answer (`NETWORK`);
* onsite: the Lean composition of remote's three subprotocols and the network certifier over one registration, the
  composite whose product is `ONSITE`, missing where the run has none.

``of_record`` holds when every one of them accepted, the condition for the demo's run of record; ``placement`` is the
network certifier's answer's, which host ran each party.

**The composite's error** (``composite_error``, `COMPOSITE`): the sum over the onsite composite's four subprotocols of
each one's error, each subprotocol's δ with where the run states it: sampled proofs' and PoUW's from their rows (their
answers' integrity-profile δ), the network certifier's from its bound (its eps), and inference-only's missing
(`NO_NCI_DELTA`). ``stated`` sums the δs the run states, a lower bound on the error, which is ``vacuous`` at 1 or
more; the rest is ``unvalued`` (`COMPOSITE_UNVALUED`).

**The bound** is the network certifier's accepted answer's public in its rows form: K and each of its terms in the
notation table's symbols (`TERMS`, `PHASE_TERMS`), each naming the public fields it sums (``lean``): k_sched
(``schedules``), k_fail (``cells``) and, for each execution phase j of the certification period (the code's link-window,
the public's ``scans[j]``), ℓ_{a,j} (``free``), k_{skip,j} (``faulty_sets``, ``faulty_bits`` and ``faulty_entries``) and
the jitter T_j·log2 d_jit, T_j the phase's buckets (``jitter``). They sum to K, by which `NetTiming.Guarantees.PeriodLeakAccepted` bounds what the serving node
sends past its committed program except with δ_net over ``period_s`` seconds, the run's m execution phases of T buckets
of τ (the certifier's case's ``schedule``). ``delta_net`` gives it term by term, each with its source: the answer's
``eps``, only its δ part (m · δ and each charged audit's δ, `EPS`), and m times each phase's knowledge, binding and
collision errors ε_ks, ε_bind and ε_tap (`ERRORS`), which no check computes, so δ_net has no number here (`UNVALUED`);
``delta`` splits eps into the scans' δ and the charged audits', with eps a phase and each charged audit's reading at
δ_i where the public lists it (``charged_audits``; `DELTA`: what sets each, and what a smaller one costs); ``skip`` is
k_{skip,j} each phase with its slot entries, every one of a tied audit drawn in part with a bound of at least one
incorrect proof block (`SKIP`, and what drawing it whole costs).
K is that statement's count as proved, a known overcount (``notes``, `OVERCOUNT`); ``connections`` is how many
connections k_sched and k_fail count, every one the schedule can carry (`CONNECTIONS`), its ``error`` where the case's
caps aren't those; ``circuits`` lists each tied audit's circuit files, public, by the digests the period's params pin
(`CIRCUITS`), and whether the run wrote each; and ``verdict_line`` is the run's published line of the network
certifier's verdict (`LINE` in verdicts.txt), None where it has none. A public whose terms don't sum to K, or that isn't the rows form, is the bound's
``error``. ``of_record`` is the figure of record (`OPEN`): that K only where the verdict's public holds every check the
tight ℓ_{a,j} rests on (``held``, `CHECKS`) and none of its premises is open (``premises``, `DISCHARGE`: each of
`PREMISES` discharged by theorems in Security's locks, each at the statement `DISCHARGES` pins, taking no named
assumption but a standing one (`STANDING`) or one its pin names and no premise that is open; and each guarantee it
rests on in the lock at the statement whose hypotheses and antecedents `RESTS_ON` names, beside what K counts,
`COUNTS`), and while one of either is ``open``, by name, K plus ℓ_a + 1 bits for each audit tied to each
execution phase's scan (``advice``: the bits each phase adds). Beside the bound, ``advice_charge``: the served run's
realized advice charge stays out of the record (`ADVICE`, which says what hides it and what doesn't yet); ``charged`` is
ℓ_{a,j} as the bound charges each execution phase, with its share of K and its source (`CHARGED`: the tied compute
audits' unregistered inputs, read off the period's parameters, with no advice length and no cap in them); ``cap`` is
each audit's ℓ_a, the cap of the advice the run registered (its published advice registration's, `REGISTRATION`), stated
as a cap and in no term of K (`CAP`), and the figure of record's ``missing`` where the run published none; and
``measured`` is honest serving's |a_j| a Commit period, of which an execution phase holds several, and their sum over
each execution phase (``bits_per_phase``, missing where the test run's advice places no Commit period on the engine
clock; ``phases`` says where those phases start, `TEST_PHASES`), measured on a test run with public inputs
(`advice_charge.py`'s record, ``--advice-charge``) and labelled as
that run's, never a figure of this one. ``tree`` is the commit the run served from (machine.txt's), None where machine.txt names none.

**Periods**, of the figure of record: the run's own certification period (``run``), proved, and each monitoring period
of `leak_bounds.PERIODS` for the serving node, the census's reference datacenter and the world. Where the figure of
record isn't the tight K, ``periods_tight`` gives the same cells of the tight K beside them, with the open checks and
premises it is conditional on (``conditional_on``), never as the result. A node's cell is proved where the run's period
covers the monitoring period's seconds (its m execution phases), and otherwise scaled from the run to the period; a
datacenter's and the world's are scaled from the node's. Each scaled cell names its rule, which ``rules`` states with
its sources (`RULES`): the bound, the case's schedule, and each census line it reads with its SHA-512 (`census`). Each
cell gives K in bits (the least integer at or above a scaled one), its share of one copy of the weights a deployment
protects (``share``: `leak_bounds.WEIGHTS`, the adversary resource model's as published, a stated parameter that
``weights`` gives with its census line and SHA-512, `WEIGHTS`) and beside it of the demo's served model's
(``share_served``, `leak_bounds.SERVED`), its share of the egress its execution phases can carry (``share_of_capacity``, the schedule's
capacity in ``egress``, `EGRESS`; the served egress is missing, `SERVED_EGRESS`), its δ_net's terms over the execution
phases it counts (``delta_net``), whether eps alone is
already vacuous (at least 1), the eps a phase under which it wouldn't be (``eps_per_phase_below``, beside the bound's
``delta``) and the statement it rests on; beside that pooled total, ``per_phase`` is the same over
one execution phase, proved only where m is 1.

**Re-proof inputs** (``reproof``, `REPROOF`): what a later re-proof of the run's sampled units needs, each wire a sampled
unit reads then proved inside its session: the committed values and their openings for every sampled unit and the units
that write what it reads, the registration, the live draw and its receipt, and the public files, each by where it is (the
serve dir's by their paths there, naming no window) and what it holds, with whether the run wrote each of OUT's and how
each must be kept (``kept``, `REPROOF_KEPT`). The serve dir's are witness, kept off OUT and custody in a copy off the
served node (``--reproof-copy`` NODE:/PATH, ``--reproof-listing``, a listing of it kept off OUT): ``copy`` names its node
and path, and each serve-dir input says whether the listing holds each file its path names and at least one file under
each directory it names, never their bytes or digests; with no copy given, ``copy`` is missing with why.

The record's counts and indices are the certification period's (m and each phase j), public in the Lean answer's
profile; it names no Commit window, whose count `publish` holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "network_traces") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "network_traces"))
import advice_charge as AC  # noqa: E402
import leak_bounds as LB  # noqa: E402
from served_integrity import exact_delta  # noqa: E402

FORMAT = "verity/onsite-network-record/v1"
EXTENDS = "verity/all-protocols-window/v1"
NETWORK = "verity/certifier/v0"
REMOTE = "verity/remote/v0"
INFERENCE_ONLY = "verity/inference-only/v0"
#: the Lean composition of remote's three subprotocols and the network certifier over one registration, as its
#: composite.json names it (`composite.py`'s remote-network product)
ONSITE = "verity/remote-network/v0"
VERDICTS = ("sampled proofs", "PoUW", "inference-only", "remote", "the network certifier", "onsite")
#: the window report's rows, by verdict
ROWS = {"sampled proofs": "integrity", "PoUW": "work", "the network certifier": "communication"}
#: where the certifier piece leaves its period's case (`certifier_piece.CASE`), whose params carry the schedule and caps
CASE = Path("pieces") / "certifier" / "case.json"
#: the run's advice registration as it publishes it (`served_verdicts.py public-registration`), its compiler's cap ℓ_a
REGISTRATION = "registration.json"
#: K's terms over the period, each a symbol of the notation table and the rows form's public fields it sums
TERMS = {"k_{sched}": ("schedules",), "k_{fail}": ("cells",)}
#: K's terms of each execution phase j, each summing its scan's public fields
PHASE_TERMS = {"\\ell_{a,j}": ("free",), "k_{skip,j}": ("faulty_sets", "faulty_bits", "faulty_entries"),
               "T_j \\log_2 d_{jit}": ("jitter",)}
ADVICE = ("not in this record: the realized advice length |a_j| is a value. The run's published registration copies state "
          "the compiler's cap in its place (`served_verdicts.py public_registration`), but each window's advice is "
          "committed and compiled at its own length, so |a_j| is hidden only once it is padded to ℓ_a before it is "
          "committed; the bound's ℓ_{a,j} holds no advice length (`CHARGED`)")
OVERCOUNT = ("a known overcount, proved and larger than necessary: k_sched and k_fail charge every connection the "
             "schedule can carry (`Schedule.caps`), whatever the traffic, and each connection's cell count twice, in "
             "k_sched (its lane's length, #S_s ≤ ∏_c (cap_c + 1)) and in k_fail (its fail cell's position), which in a "
             "run coincide; K is PeriodLeakAccepted's count as it states it")
CONNECTIONS = ("the caps of the certifier's case's params (``cells``), one a connection: the period's check refuses any "
               "but its schedule's (`NetworkCheck.Period.Schedule.caps`), egress_cells for each of the lanes · m · T "
               "connections the network certifier can carry over the period's m execution phases, so this counts the "
               "connections the schedule can carry, not those served")
EPS = {"is": "δ_net's δ part alone: m · δ and each charged audit's δ",
       "source": "the network-period check's public eps (NetworkCheck.Period)"}
#: δ_net's terms that eps doesn't count, each once an execution phase: `PeriodLeakAccepted` bounds each phase's audit's
#: failure by δ + ε_ks + ε_bind + ε_tap
ERRORS = {
    "ε_ks": {"is": "the knowledge error, κ(δ) − δ + √(γ·Pr[B])",
             "source": "Claims.Sampling.KS at the deployment, PeriodLeakAccepted's hypothesis"},
    "ε_bind": {"is": "the binding error, Pr[B_ext]: an accepted extraction that disagrees with the committed trace",
               "source": "Claims.Sampling.Bext, a SHA-512 collision in the run (CommitmentScheme.Binds)"},
    "ε_tap": {"is": "Pr[B_tap]: the extracted trace's openings colliding with the network certifier's",
              "source": "NetTiming.Btap, a SHA-512 collision in the run (CommitmentScheme.Binds)"},
}
UNVALUED = ("δ_net is eps plus each execution phase's ε_ks, ε_bind and ε_tap, never eps alone; no check computes those "
            "three, so δ_net has no number here")
COMPOSITE = (f"{ONSITE}'s error, Verity.Game's SoundnessFor at its compiled verifier: it accepts while the claim of one of "
             "its four subprotocols fails with probability at most the sum of their errors, each its published δ at the "
             "strategy's SHA-512 calls plus ε_forge(t, Q) plus its loss at its finder's collision probability "
             "(`Verity.Vocab.error`). Each subprotocol's δ is a term of its own, so one draw read at two δs, a tied "
             "audit's at sampled proofs' δ and at the network period's, counts in both")
COMPOSITE_UNVALUED = ("each subprotocol's ε_forge(t, Q) and its loss at its finder's collision probability, and the "
                      "network certifier's m(ε_ks + ε_bind + ε_tap) (`ERRORS`): no check computes them, so the composite's "
                      "error has no number here, and ``stated`` is the sum of the δs the run states, a lower bound on it")
NO_NCI_DELTA = "no published file states it: remote's composite publishes each case's params by their SHA-512 alone"
DELTA = ("what sets eps: m · δ_scan, every phase's scan at one δ, its one-stage answer's stated δ (its integrity "
         "profile's, a sampling-design parameter in (0, 1) and not C-Flock's soundness error, which is in ε_ks), plus "
         "δ_i for each tied audit drawn in part (NetworkCheck.Period's chargedOf). Where the public lists those "
         "readings (`charged`: each audit's draw read at δ_i, with its bound b there), δ_i is the period check's own "
         "constant, fixed before any coin, and the same draw read at its answer's δ stays sampled proofs' term of the "
         "composite's error; where it lists none, δ_i is the audit's answer's δ, and the charged part is what eps holds "
         "beyond m · δ_scan. A scan with every proof block drawn holds no incorrect proof block at any δ, so a smaller "
         "δ_scan costs neither proving nor bits; drawn in part, its bound grows as δ falls (faulty_sets, faulty_bits). "
         "A tied audit drawn in part charges every slot-entry bit of its phase (faulty_entries) once its bound at δ_i "
         "is at least one incorrect proof block, as at any small δ_i it is, so a smaller δ_i costs K nothing; drawn "
         "whole, it leaves eps and its entries leave K, at the cost of proving each of its proof blocks")
NO_READINGS = ("the period's public lists no reading at δ_i (`charged`): its check reads each charged audit at its "
               "answer's δ, and the charged part is eps − m · δ_scan")
#: the run's published line of the network certifier's verdict (`served_verdicts.pieces`)
LINE = "THE NETWORK CERTIFIER "
CHARGED = ("PeriodLeakAccepted's ℓ_{a,j}: the free inputs of each compute audit tied to execution phase j's scan, those "
           "its Program neither registers at a pinned root nor ties (NetworkCheck.Period's tiedAudits sums the "
           "parameters' audits[].free, #(freeInputs \\ reg); `records_audit.py period`'s free for V3c's window audit). "
           "It reads no cap and no advice length: the advice is the compile unit's input, not the window Program's "
           "(`records_audit.py declared_count`), so neither the advice registration's width ℓ_a (its compiler's "
           "cap_bits) nor |a_j| is in it. It is the tight form, the figure of record only once the period's check "
           "performs both checks it rests on and none of its premises is open (`CHECKS`, `DISCHARGE`, `OPEN`)")
SKIP = ("PeriodLeakAccepted's k_{skip,j}: what execution phase j's draws didn't take whole. The scan's incorrect proof "
        "blocks' term (``faulty_sets``, ``faulty_bits``: none where every one of its proof blocks is drawn), and "
        "``faulty_entries``, every slot-entry bit E_w of each audit tied to j's scan that is drawn in part (charged, "
        "NetworkCheck.Period's chargedOf) with a bound of at least one incorrect proof block. Drawn whole, a tied "
        "audit adds no entries and its δ leaves eps, at the cost of proving every one of its proof blocks")
#: the checks the tight ℓ_{a,j} rests on, each by the name the period's public lists it under in ``held`` once its Lean
#: check performs it
CHECKS = {"free": "each tied audit's free inputs computed from its Program's descriptor, not taken as a parameter",
          "circuit": "each tied audit's circuit held fixed whatever its advice"}
#: Security's locks in this tree (`tools/verity/lean/check.py`'s record of each guarantee's named assumptions and of
#: the definitions it reads): the guarantees', and the Properties' that discharge premises at the deployed code
ROOT = HERE.parents[1]
LOCKS = (ROOT / "Security" / "lean-audit.json", ROOT / "Security" / "Properties" / "lean-audit.json")
#: the guarantees the tight K rests on, by their names in `LOCKS`: PeriodLeakRegistered's count, at the values the
#: registration opens at its registered inputs, is K's; PeriodLeakAccepted's is the same count at the trace's values
#: there; PeriodLeakTied's counts a phase's tied audits with the scan as one Program, as ℓ_{a,j} does. The lock records
#: a guarantee's named assumptions, never the antecedents its statement takes, so each gives ``statement``, the digest
#: of its statement among the lock's reads (`statement`) at which it was read, None where none was, and ``takes``, each
#: hypothesis, antecedent and condition its good event reads that the deployed code must meet, by its Prop's name or,
#: stated inline, its binder's, with the premise (`PREMISES`), standing assumption (`STANDING`) or term of what K counts
#: (`COUNTS`) that names it. Parameters the record states (δ ≥ 0) aren't listed
RESTS_ON: dict[str, dict[str, Any]] = {
    "NetTiming.SecurityProofs.PeriodLeakAccepted": {
        "statement": "0db3f3a734d201ea45375c21d5f216e32657736e057303e8cb3291e0732a3531",
        "takes": {"Claims.Sampling.KS": "sampled-leg-bridge", "Claims.Flock.ListWithin": "sampled-leg-bridge",
                  "Claims.Sampling.CommitmentScheme.Binds": "sampled-leg-bridge",
                  "Verity.Assumptions.TrustedHost": "Verity.Assumptions.TrustedHost",
                  "hPw": "windows-placed", "hβ": "windows-placed", "hcap": "schedules-held",
                  "hfit": "schedules-held", "hs": "schedules-covered", "hdz": "client-decodes",
                  "NetTiming.ScopeAcc": "wires-fit", "NetTiming.TapOpens": "taps-open"}},
    "NetTiming.SecurityProofs.PeriodLeakRegistered": {
        "statement": "79ebe236812c4d1257a7dd6dfc7dbc9816c7d1846745f64dce1596c5ebd20034",
        "takes": {"Claims.Sampling.KS": "sampled-leg-bridge", "Claims.Flock.ListWithin": "sampled-leg-bridge",
                  "Claims.Sampling.CommitmentScheme.Binds": "sampled-leg-bridge",
                  "Verity.Assumptions.TrustedHost": "Verity.Assumptions.TrustedHost",
                  "NetTiming.WindowsPlaced": "windows-placed", "NetTiming.SchedulesHeld": "schedules-held",
                  "NetTiming.SenderDecodes": "schedules-covered", "NetTiming.ScopeAcc": "wires-fit",
                  "NetTiming.RegOpensAt": "reg-opens-at", "NetTiming.TapsOpen": "taps-open"}},
    "NetTiming.SecurityProofs.PeriodLeakTied": {
        "statement": None,
        "takes": {"NetTiming.RowsStateHeld": "rows-state-held", "NetTiming.RowsProgWired": "rows-prog-wired",
                  "NetTiming.RowsCircuitHeld": "circuit-held", "NetTiming.RowsTiesOrdered": "rows-ties-ordered",
                  "NetTiming.CircuitCounts": "free-count",
                  "NetTiming.Assumptions.SharedRootsParsed": "shared-roots-parsed"}},
}
#: each premise the tight K rests on, by the name ``open`` gives it: what it says, and the theorems that discharge it at
#: the deployed code by their names in `LOCKS`' guarantees (none named yet leaves it open)
PREMISES: dict[str, dict[str, Any]] = {
    "wires-fit": {"is": "ScopeAcc whole at the tied audit's statements: the sampling scope (Scope, the sampled verifier's, "
                        "which its scope check gives where ScopeMatches holds of the deployment) and the layout's fit "
                        "(BoundaryGates.Fits, every clause), through the decoder link and the period check running "
                        "Circuit.check; what each discharge takes of the deployment is its pin's (`DISCHARGES`)",
                  "discharged_by": ("Properties.NetworkCertifier.wiredScopeAcc",
                                    "Properties.NetworkCertifier.circuitWiresFit",
                                    "Properties.NetworkPeriod.rowsWiresChecked")},
    "session-wires": {"is": "SessionWires: each sampled proof block's wired inputs equal their writers' committed rows, "
                            "proved inside its own session, whose statement reads the writer's committed leaf by path and "
                            "whose circuit equates it with the wired input, so no opening reaches the verifier; "
                            "OneStageZk.verdicts accepting only where each sampled reader's wire is the layout's, at the "
                            "registered position, root and path; and SessionOpens and the binding from the sampled leg's "
                            "bridge (`sampled-leg-bridge`), never a named assumption. Open until the theorem that "
                            "OneStageZk.verdicts' acceptance gives it is named and in the lock",
                      "discharged_by": ()},
    "free-count": {"is": "CircuitCounts: each tied audit's free at least #(freeInputs \\ reg) at the wired layout",
                   "discharged_by": ("Properties.NetworkPeriod.phaseCounts",)},
    "held-performed": {"is": "the public's held names only the checks the period check performed, each holding of it",
                       "discharged_by": ("Properties.NetworkPeriod.rowsHeld",)},
    "circuit-held": {"is": "RowsCircuitHeld: each tied audit's Program the class circuit the period registers",
                     "discharged_by": ("Properties.NetworkPeriod.phaseCircuitHeld",)},
    "rows-state-held": {"is": "RowsStateHeld: each tied audit's state read at row a ≥ 1 committed where the scan's o1 of "
                              "unit a − 1 is, under the scan's state root, and so the scan's own state read at unit a",
                        "discharged_by": ("Properties.NetworkPeriod.phaseStateHeld",)},
    "rows-prog-wired": {"is": "RowsProgWired: the scan's prog read at lane l, leaf b committed where its tied audit's slot "
                              "entry of chunk l, leaf b is, under that audit's entries root",
                        "discharged_by": ("Properties.NetworkPeriod.phaseProgWired",)},
    "rows-ties-ordered": {"is": "RowsTiesOrdered: the phase's layout, each statement's wires and the ties, fits its joint "
                                "Program (BoundaryGates.Fits): acyclic, and no step's egress depending on a later step's "
                                "ingress",
                          "discharged_by": ("Properties.NetworkPeriod.phaseTiesOrdered",)},
    "shared-roots-parsed": {"is": "SharedRootsParsed: at the deployment, two gates the registrations commit at one "
                                  "position under one root open to one value, or their H calls collide; the sampled "
                                  "leg's bridge proves it of the deployed parse. Open until that theorem is named",
                            "discharged_by": ()},
    "reg-opens-at": {"is": "RegOpensAt: at each window's statement the registration's openings open its registered "
                           "inputs under the statement's roots (RegOpens). The run's reg holds per-window values (the "
                           "window's tied activations and the scan's state) and pinned ones (the weights, at their "
                           "registration's roots): the per-window part by a restatement of the finder per window, the "
                           "pinned part by the sampled leg's bridge with a coverage Property of the registration check at "
                           "OneStageZk's record. Open until both are named",
                     "discharged_by": ()},
    "taps-open": {"is": "TapsOpen, PeriodLeakAccepted's TapOpens antecedent: at every window's statement the network "
                        "certifier's openings register its observations at the window's operator. The period check's "
                        "half: on an accepted rows-form execution phase the scan's rows/<d> root and domain are those the "
                        "network certifier signed for its link, d and window; the device's, under trusted-host: its "
                        "rows at the transfer gates are its observations, its ingress values its operator on the earlier "
                        "ones",
                  "discharged_by": ("Properties.NetworkCertifier.tapsOpen",)},
    "windows-placed": {"is": "WindowsPlaced, PeriodLeakAccepted's hPw and hβ: each window's statement has the partition "
                             "the period read, and each of its faulty sets at δ commits at most the bits β the period "
                             "charges it. Open until the theorem that the period check's acceptance gives it is named",
                       "discharged_by": ()},
    "schedules-held": {"is": "SchedulesHeld, PeriodLeakAccepted's hcap and hfit: each schedule the network certifier may "
                             "run gives each connection at most its cap of cells, and its record format holds it "
                             "(TapRecord.fits, the rows format's). Open until the theorem that gives it of the network "
                             "certifier's code is named",
                       "discharged_by": ()},
    "schedules-covered": {"is": "SenderDecodes's schedule half, PeriodLeakAccepted's hs: each secret K counts runs on a "
                                "schedule of Ss, whose number k_sched counts, so Ss must hold every schedule the deployed "
                                "network certifier can run (its own admission, at the case's caps). Open until it is "
                                "stated of the network certifier's code and a theorem gives it; the decoder half is what "
                                "K counts (`COUNTS`)",
                          "discharged_by": ()},
    "sampled-leg-bridge": {"is": "the sampled leg's bridge, from its guarantee Verity.Vocab.SampledProofsSound at "
                                 "one-stage v2 to OneStageZk's Lean record: the sampled facts the period takes (KS; "
                                 "ListWithin, the decoder's list at Flock's radius; Binds, the trace commitment's "
                                 "binding; at the per-member record, the only form with wires, SessionKS, the sessions' "
                                 "KS, which SessionKS_of can't give there, its SessionRuns failing at more than one "
                                 "session; SessionOpens; RegOpens: the registered reads' openings), proved of the Lean "
                                 "verifier and taken as theorem hypotheses on the layer below, never SampledCodeRuns or "
                                 "another premise about the deployed Python. Open until the bridge's theorem is in the "
                                 "lock",
                           "discharged_by": ()},
}
#: each theorem of `PREMISES`' ``discharged_by`` that has been read, by its name in `LOCKS`, as `RESTS_ON` gives a
#: guarantee: ``statement``, its statement's digest among the lock's reads, and ``takes``, each antecedent of it that the
#: deployment must meet, by its Prop's name or its binder's, with the premise, standing assumption or term of what K
#: counts that names it. A Property's statement may take a deployment's premise inline, where the lock records no
#: assumption (`ScanScopeAcc` takes `StmtFlattens` and `ScopeMatches` of the deployment it quantifies), so a discharge
#: counts only at the statement pinned here
DISCHARGES: dict[str, dict[str, Any]] = {}
#: the named assumptions that may stand under the figure of record, each on an auditor-trusted device or the auditor's
#: placement of one, named beside the bound
STANDING = {"Verity.Assumptions.TrustedHost": "trusted-host: the network certifier and its host are the auditor's",
            "Accounting.Covered": "certifier-coverage: the links in the room are the registered ones, each through the "
                                  "network certifier (a listed physical premise, the auditor's placement)"}
#: what K counts: the hypotheses on the covert sender's strategy that its guarantee quantifies, which no deployment
#: discharges, so none is open
COUNTS = {"client-decodes": "the secrets a decoder recovers from the client's view, the cells the network certifier's "
                            "run gives the client: SenderDecodes's decoder half, PeriodLeakAccepted's hdz"}
DISCHARGE = ("the tight K's premises (`PREMISES`), each open until every theorem that discharges it at the deployed code "
             "is in Security's lock at the statement `DISCHARGES` pins (``unpinned``, with the lock's digest, where it "
             "isn't), with no named assumption but a standing one (`STANDING`) or one its pin takes, and every premise "
             "its pins take is discharged (``takes``), so no premises discharge each other; and open while none is "
             "named; each guarantee of `RESTS_ON` the lock doesn't list, or lists at a statement other than the one "
             "whose hypotheses and antecedents it names (``takes``); and each named assumption the lock gives such a "
             "guarantee that is neither standing nor named there, by its own name. What K counts (`COUNTS`) is no "
             "premise. No premise about our own Python stands: soundness is in Lean or rests on an auditor-trusted "
             "device")
#: why a guarantee of `RESTS_ON` is open at a statement other than the one it names
RESTATED = ("its statement in the lock isn't the one whose hypotheses and antecedents `RESTS_ON` names: read them again, "
            "name each in its takes, and pin the lock's digest")
OPEN = ("the figure of record: K as the period's check states it, the tight form, only where its Lean verdict performed "
        "every check of `CHECKS` (the public's held) and no premise of it is open (`DISCHARGE`). While "
        "either is open it is K plus ℓ_a + 1 bits for each audit tied to each execution phase's scan, beside that "
        "phase's ℓ_{a,j}: an advice of at most ℓ_a bits (`CAP`) is one of fewer than 2^(ℓ_a + 1) strings. The tight K's "
        "cells go beside the figure of record's, conditional on what is open (periods_tight)")
CAP = ("ℓ_a, the cap of the advice the run registered: its published advice registration's compiler cap_bits "
       "(`REGISTRATION`, its advice's length padded to it), the length each Commit period's advice is padded to before "
       "it is committed, the same for each tied audit, whose own registration in the case's commitment binds no advice "
       "or compiler (`records_audit.py period` refuses one that does). A cap, in no term of PeriodLeakAccepted's K: it "
       "enters the figure of record only while a check of `CHECKS` or a premise is open, as ℓ_a + 1 bits for each tied "
       "audit (`OPEN`)")
NO_REGISTRATION = (f"the run published no advice registration ({REGISTRATION}), whose compiler's cap is the ℓ_a the "
                   "figure of record charges each tied audit while a check or a premise is open")
TIED_ALL = ("every audit of the case's commitment: the public names no scan's tied audits (scans[j].tied), so each is "
            "charged to every execution phase, exact with one audit")
#: each scaled cell's rule, by the name it gives; `rules` adds each one's sources
RULES = {
    "per-phase": "the run's K and eps over one of its m execution phases, K/m: their average, which no statement bounds a "
                 "phase by where m exceeds 1",
    "run-to-period": "the run's per execution phase times the period's execution phases, ⌈seconds / (T·τ)⌉, each as the "
                     "run's at its load: ℓ_{a,j}, k_{skip,j} and the jitter scale with the phases, and k_sched and "
                     "k_fail, ⌈log2(cap_c + 1)⌉ each a connection, with the connections, at the schedule's lanes · T a "
                     "phase",
    "datacenter": "the node's times the census's reference datacenter's machines, one network certifier on each "
                  "machine's uplink",
    "world": "the node's times the world's links: the adversary resource model's compute over its monitoring period, at "
             "an H100-equivalent's FLOP/s, in machines of the reference datacenter's GPUs, rounded up, one network "
             "certifier on each",
}
EGRESS = ("the schedule's egress capacity an execution phase, from the case's params.schedule: at most lanes connections "
          "live, each at most slots cells a bucket, over its T buckets, and no more than its caps' total (egress_cells "
          "for each of its lanes · T connections), each cell cell_bytes bytes. It bounds the egress a phase can carry, so "
          "K's share of it (each cell's share_of_capacity) is a lower bound on K's share of the egress it carried")
SERVED_EGRESS = ("not in this record: the egress the run served is a count of its own traffic, which no published file "
                 "states (the network certifier's chain files that count it stay on the node)")
WEIGHTS = ("a stated parameter: one copy of the weights a deployment protects, the adversary resource model's model as "
           "published (its census line's weights_bytes, at its weights_datatype), whatever the model the run served; "
           "share_served is of the demo's served model beside it")
FROM_BOUND = "bound: the run's K and eps over its m execution phases"
FROM_SCHEDULE = f"{CASE} params.schedule: an execution phase's T buckets of τ"
STATEMENTS = {"run": LB.STATEMENTS["per_period"], "node": LB.STATEMENTS["per_period"],
              "datacenter": LB.STATEMENTS["datacenter"], "world": LB.STATEMENTS["datacenter"]}
NO_MEASUREMENT = ("no test run's measurement given (--advice-charge: `advice_charge.py`'s record of honest serving on "
                  "public inputs)")
TEST_PHASES = ("the test run's own: T buckets of τ from its first Commit period's engine clock reading, not the network "
               "certifier's phase boundaries, so a Commit period near a boundary can count in the next phase")
#: what a later re-proof of the run's sampled units needs, each wire a sampled unit reads then proved inside its session:
#: each by name, where it is (the serve dir, the prover's on the served node, by its path there with w<K> the audited
#: window and <class> V3c's class; or OUT, the run's public files) and what it holds
REPROOF = {
    "values": ("serve dir", "v3c-w<K>-<class>/window/",
               "the window commit: every unit's committed values with their rows and salts, the sampled units' and those "
               "of the units that write what they read, and its circuit.json and registration.json"),
    "openings": ("serve dir", "v3c-w<K>-<class>/audit/window/",
                 "each sampled unit's opening as it was proved (m<k>-private.bin, m<k>-public.bin, v<j>-private.bin), "
                 "with the partition and the Program the audit sampled"),
    "draw": ("serve dir", "v3c-w<K>-<class>/audit/window/{draw,segment,receipt}.json",
             "the live draw and its receipt, the stream's record that it received the registration before it read a coin"),
    "coins": ("serve dir", "v3c-w<K>-<class>/audit/challenger/",
              "the live challenger session the draw read its coins from"),
    "advice": ("serve dir", "advice/w<K>/",
               "the advice's registration and its opening (private/advice.bin), which the window circuit reads under its "
               "hiding root"),
    "weights": ("serve dir", "weights_registration.json, private/weights.key, private/cos_sin_cache.npy",
                "the weights registration the audit pins, the committer's key their salts come from and the rotary table, "
                "beside the checkpoint (REPO at REV, public)"),
    "run root": ("serve dir", "windows.jsonl", "the run root the audit holds the registration to"),
    "circuit": ("serve dir", "window-circuit-w<K>-<class>.json",
                "the verifier's window-circuit record at the class, which the window commit's circuit is held to"),
    "h1": ("serve dir", "h1/", "under H1_TIE, H1's registration of the window and its log, whose tables the circuit reads"),
    "registration": ("OUT", "v3c-audit/registration.json",
                     "the window's registration, whose roots bind every committed value and which the period's "
                     "commitment holds whole"),
    "audit": ("OUT", "v3c-audit/summary.json",
              "the audit's summary: its draw, its coins and that the registration was fixed before the draw"),
    "audit record": ("OUT", "v3c-audit/window/audit.json", "the audit record, each sampled unit's verdict"),
    "period audit": ("OUT", "v3c-audit/period.json", "the audit as the network certifier's period reads it"),
    "main": ("OUT", "v3c-audit/main.txt", "main's first-parent line the challenger session pins on"),
    "advice registration": ("OUT", REGISTRATION, "the advice registration's public form, its compiler's cap"),
    "case": ("OUT", str(CASE), "the network certifier's period case, whose commitment holds the tied audits' "
                               "registrations"),
    "record": ("OUT", "record/verdicts.json", "the window report of every verdict"),
    "circuits": ("OUT", "v3c-audit/circuits/",
                 "C-Flock's circuit file of each template the tied audits' Programs batch, public, each named by the "
                 "SHA-512 the period's params pin it to (`CIRCUITS`)"),
}
#: the circuit files the period check reads, public: C-Flock's flock-ir-unit/v2 file of each template a tied audit's
#: Program batches, in OUT by its SHA-512, which the period's params pin (``params.audits[i].circuits``) and the check
#: holds it to; never the registration's window.circuit, the window-circuit record's
CIRCUITS = "v3c-audit/circuits"
NO_CIRCUITS = ("the period's params pin no circuit file (params.audits[i].circuits), so its check held no tied audit's "
               "circuit to a file")
REPROOF_KEPT = ("each retained past the week after the run, never only on the served node: OUT's in custody with the "
                "run's public files; the serve dir's, witness, never in OUT or custody, which are public, but copied off "
                "the served node to a private path keeping the serve dir's layout (`copy`: its node and path, and of "
                "each serve-dir input whether the copy's listing holds each file its path names and at least one file under "
                "each directory it names, never their bytes or digests); a re-proof holds each to the public registration and the audit's summary")
NO_COPY = ("no copy of the serve dir's inputs is given (--reproof-copy NODE:/PATH and --reproof-listing): they are on "
           "the served node only")


def missing(why: str, src: str | None = None) -> dict[str, Any]:
    return {"verdict": "missing", "why": why, **({"from": src} if src else {})}


def row_verdict(rows: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A verdict from the window report's row ``key``: a missing, dropped or development row is missing, with its why."""
    src = f"record/verdicts.json protocols.{key}"
    r = rows.get(key)
    if not isinstance(r, Mapping):
        return missing(f"the window report has no {key} row", src)
    if r.get("verdict") in ("missing", "dropped"):
        return missing(str(r.get("why")), src)
    out = {"protocol": r.get("protocol"), "verdict": r.get("verdict"), "from": src}
    return out | ({"reason": r["reason"]} if r.get("reason") else {})


def composites(out: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each composite.json under OUT/composite, by the product it names: its path in OUT and its document."""
    found: dict[str, tuple[str, dict[str, Any]]] = {}
    for f in sorted((out / "composite").glob("*/composite.json")):
        doc = json.loads(f.read_text())
        found[doc.get("product")] = (str(f.relative_to(out)), doc)
    return found


def outcome(o: Mapping[str, Any], protocol: str, src: str) -> dict[str, Any]:
    """`composite.py`'s public outcome as a verdict: ``none`` where Lean gave no answer."""
    return {"protocol": protocol, "verdict": o.get("verdict") or "none", "from": src,
            **({"refusal": o["refusal"]} if o.get("refusal") else {})}


def composite_verdict(found: Mapping[str, tuple[str, dict[str, Any]]], product: str, none: str) -> dict[str, Any]:
    """The Lean verdict of ``product``'s composite, with the subprotocols it couldn't establish; missing (``none``) where
    the run has no such composite."""
    if product not in found:
        return missing(none)
    src, doc = found[product]
    if doc.get("verdict") is None:
        return missing(f"its composite ran no Lean check ({src})", src)
    return outcome(doc["verdict"], product, src) | (
        {"not_established": list(doc["missing"])} if doc.get("missing") else {})


def case_verdict(found: Mapping[str, tuple[str, dict[str, Any]]], product: str, case: str) -> dict[str, Any]:
    """The outcome of ``case``'s own answer in ``product``'s composite."""
    if product not in found:
        return missing(f"its case is read from {product}'s composite, and the run has none")
    src, doc = found[product]
    for c in doc.get("cases") or ():
        if c.get("id") == case:
            return outcome(c, case, f"{src} cases")
    return missing(f"{product}'s composite has no {case} case", src)


def bound(public: Mapping[str, Any]) -> dict[str, Any]:
    """K and its terms (module docstring) from a rows-form public; ValueError naming why for any other. m is its scans'
    count, held to its ``windows`` where the public keeps it (a run's published form drops it)."""
    if not isinstance(public, Mapping) or not isinstance(public.get("scans"), list):
        raise ValueError("not NetworkCheck.Period's rows form, whose public has scans: the demo's network verdict is "
                         "the rows form's")
    try:
        terms = [{"symbol": s, "bits": sum(int(public[f]) for f in fs), "lean": list(fs)} for s, fs in TERMS.items()]
        for i, scan in enumerate(public["scans"]):
            terms += [{"symbol": s, "phase": i, "bits": sum(int(scan[f]) for f in fs),
                       "lean": [f"scans[{i}].{f}" for f in fs]} for s, fs in PHASE_TERMS.items()]
        K, eps = int(public["K"]), Fraction(public["eps"])
        deltas = {Fraction(s["delta"]) for s in public["scans"]}
        readings = [{"delta": str(Fraction(c["delta"])), **{f: int(c[f]) for f in ("drawn", "units", "wrong_units")}}
                    for c in public["charged"]] if "charged" in public else None
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the rows form's public lacks a term: {type(e).__name__}: {e}") from None
    if sum(t["bits"] for t in terms) != K:
        raise ValueError(f"K is {K}, and its terms sum to {sum(t['bits'] for t in terms)}")
    m = len(public["scans"])
    if m == 0:
        raise ValueError("the rows form's public has no scan: a period of no execution phase, which bounds nothing")
    if "windows" in public and int(public["windows"]) != m:
        raise ValueError(f"{m} scans for the period's {public['windows']} execution phases")
    if len(deltas) != 1 or eps < m * min(deltas):
        raise ValueError(f"eps {eps} isn't m · δ_scan plus the charged audits' δ: the scans' δ are "
                         f"{sorted(map(str, deltas))}, and the period's check holds them to one")
    (scan,) = deltas
    if readings is not None and eps != m * scan + sum((Fraction(r["delta"]) for r in readings), Fraction(0)):
        raise ValueError(f"eps {eps} isn't m · δ_scan plus each charged reading's δ_i: m · δ_scan is {m * scan}, and "
                         f"the readings' δ_i are {[r['delta'] for r in readings]}")
    sources = [EPS, *ERRORS.values()]
    skip = {"is": SKIP, "bits_per_phase": [t["bits"] for t in terms if t["symbol"] == "k_{skip,j}"],
            "entries_per_phase": [int(s["faulty_entries"]) for s in public["scans"]]}
    return {"statement": LB.STATEMENTS["per_period"], "K": K, "phases": m, "notes": [OVERCOUNT], "terms": terms,
            "delta_net": {"terms": [t | s for t, s in zip(delta_net(eps, m), sources)], "note": UNVALUED},
            "delta": {"scan": str(scan), "charged": str(eps - m * scan), "eps_per_phase": str(eps / m), "is": DELTA,
                      "charged_audits": {"missing": NO_READINGS} if readings is None else readings},
            "skip": skip}


def delta_net(eps: Fraction, phases: int) -> list[dict[str, Any]]:
    """δ_net over ``phases`` execution phases, term by term: eps, and each of `ERRORS` that many times."""
    return [{"symbol": "eps", "value": str(eps)}, *({"symbol": s, "times": phases} for s in ERRORS)]


def composite_error(rows: Mapping[str, Any], eps: Fraction | None, why: str = "no accepted bound") -> dict[str, Any]:
    """The composite's error term by term (`COMPOSITE`): each subprotocol's δ with where the run states it, or why it
    doesn't (the network certifier's eps from its bound, else ``why``); ``stated`` sums those stated (None where none
    is), ``vacuous`` at 1 or more."""
    terms: list[dict[str, Any]] = []
    for name in ("sampled proofs", "PoUW"):
        t = {"subprotocol": name, "from": f"record/verdicts.json protocols.{ROWS[name]}.delta: its answer's "
                                          "integrity-profile δ"}
        d = (rows.get(ROWS[name]) or {}).get("delta")
        try:
            terms.append(t | ({"missing": "its row states no δ"} if d is None else {"delta": str(exact_delta(d))}))
        except (TypeError, ValueError):
            terms.append(t | {"error": f"its row's δ {d!r} isn't a number"})
    terms.append({"subprotocol": "inference-only", "missing": NO_NCI_DELTA})
    terms.append({"subprotocol": "the network certifier", "from": f"{EPS['source']}: its eps (`EPS`)"}
                 | ({"missing": why} if eps is None else {"delta": str(eps)}))
    given = [Fraction(t["delta"]) for t in terms if "delta" in t]
    stated = sum(given, Fraction(0)) if given else None
    return {"is": COMPOSITE, "terms": terms, "stated": None if stated is None else str(stated),
            "vacuous": stated is not None and stated >= 1, "unvalued": COMPOSITE_UNVALUED}


def eps_of(b: Mapping[str, Any]) -> Fraction:
    """The bound's eps, its δ_net's first term."""
    return Fraction(b["delta_net"]["terms"][0]["value"])


def schedule(case: Mapping[str, Any], phases: int) -> tuple[Fraction, int, int]:
    """An execution phase's seconds, T buckets of τ (the case's ``schedule``), the connections its caps count
    (``cells``), the case's period held to the answer's phases where its params keep ``windows`` and its caps to its
    schedule's over them (`CONNECTIONS`), and its egress capacity a phase in bits (`EGRESS`)."""
    params = case["params"]
    if "windows" in params and int(params["windows"]) != phases:
        raise ValueError(f"the case's period has {params['windows']} execution phases, and its answer {phases}")
    s, caps = params["schedule"], params["cells"]
    if not isinstance(caps, list):
        raise TypeError(f"the case's cells are its caps, a list, not {type(caps).__name__}")
    n = int(s["lanes"]) * phases * int(s["window"])
    if caps != [int(s["egress_cells"])] * n:
        raise ValueError(f"the case's {len(caps)} caps aren't its schedule's (Schedule.caps: egress_cells = "
                         f"{s['egress_cells']} for each of its lanes · m · T = {n} connections), which the period's "
                         "check refuses")
    T = int(s["window"])
    cells = min(int(s["lanes"]) * int(s["slots"]) * T, n // phases * int(s["egress_cells"]))
    return Fraction(T * int(s["bucket_ms"]), 1000), n, cells * 8 * int(s["cell_bytes"])


def census_line(name: str, *fields: str) -> dict[str, Any]:
    """The census's reference line in ``name``: its file, its id, the ``fields`` a rule reads, and its version, the
    SHA-512 of its canonical JSON."""
    d, = (d for d in json.loads((LB.CENSUS / name).read_text()) if d["role"] == "reference")
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"file": str((LB.CENSUS / name).relative_to(ROOT)), "id": d["id"],
            "read": {f: d[f] for f in fields}, "sha512": hashlib.sha512(canonical).hexdigest()}


def census() -> dict[str, Any]:
    """The links the datacenter's and the world's cells multiply by (`leak_bounds.datacenter_links`,
    `leak_bounds.world_links`), each with the census lines it reads."""
    machines = census_line("deployments.json", "machines")
    world = [census_line("adversary.json", "total_flop", "flop_per_s_per_h100", "monitoring_period_s"),
             census_line("deployments.json", "gpus_per_machine")]
    return {"datacenter": {"links": LB.datacenter_links(), "sources": [machines]},
            "world": {"links": LB.world_links(), "sources": world}}


def cell(bits: Fraction, eps: Fraction, phases: int, where: str, capacity: int,
         rule: str | None = None) -> dict[str, Any]:
    """A cell over ``phases`` execution phases, each of ``capacity`` egress bits (`EGRESS`), proved, or scaled by
    ``rule`` (`RULES`)."""
    return {"bits": math.ceil(bits), "share": float(bits / LB.WEIGHTS.bits), "share_served": float(bits / LB.SERVED.bits),
            "share_of_capacity": float(bits / (capacity * phases)), "delta_net": delta_net(eps, phases),
            "vacuous": eps >= 1, "eps_per_phase_below": str(Fraction(1, phases)),
            "statement": STATEMENTS[where]} | ({"how": "scaled", "rule": rule} if rule else {"how": "proved"})


def periods(K: int, eps: Fraction, m: int, phase_s: Fraction, connections: int, capacity: int) -> dict[str, Any]:
    """The run's own period, proved, and each monitoring period's node, proved where the run's period covers it and
    else scaled, datacenter and world, scaled from the node's, each beside its figure over one execution phase (module
    docstring)."""
    refs = census()
    links = {"node": 1, **{k: v["links"] for k, v in refs.items()}}
    one, one_eps, period_s = Fraction(K, m), eps / m, m * phase_s
    per_phase = cell(one, one_eps, 1, "run", capacity, None if m == 1 else "per-phase")
    run = cell(Fraction(K), eps, m, "run", capacity) | {"seconds": str(period_s), "phases": m,
                                                        "connections": connections, "per_phase": per_phase}
    sources = {"per-phase": [FROM_BOUND], "run-to-period": [FROM_BOUND, FROM_SCHEDULE],
               **{k: v["sources"] for k, v in refs.items()}}
    weights = {"share": {"of": LB.WEIGHTS.label, "is": WEIGHTS, "bits": LB.WEIGHTS.bits,
                         "source": census_line("adversary.json", "model", "total_params", "weights_bytes",
                                               "weights_datatype")},
               "share_served": {"of": LB.SERVED.label, "bits": LB.SERVED.bits}}
    out: dict[str, Any] = {"phase_s": str(phase_s), "weights": weights, "run": run,
                           "egress": {"capacity_bits_per_phase": capacity, "is": EGRESS,
                                      "served": {"missing": SERVED_EGRESS}},
                           "rules": {k: {"rule": text, "sources": sources[k]} for k, text in RULES.items()}}
    for name, seconds in LB.PERIODS.items():
        w = math.ceil(Fraction(seconds) / phase_s)
        covered = period_s >= seconds
        node = m if covered else w
        out[name] = {"seconds": seconds, "phases": w}
        for where, n in links.items():
            rule = where if where != "node" else None if covered else "run-to-period"
            params = {"phases": node, "phase_s": str(phase_s), "links": n,
                      "connections": str(Fraction(connections, m) * node * n)}
            out[name][where] = cell(one * node * n, one_eps * node * n, node * n, where, capacity, rule) | {
                "parameters": params,
                "per_phase": per_phase if where == "node" else cell(one * n, one_eps * n, n, where, capacity, where)}
    return out


def measured(path: Path | None) -> dict[str, Any]:
    """Honest serving's advice charge from a test run's `advice_charge.py` record, labelled as that run's."""
    if path is None:
        return {"missing": NO_MEASUREMENT}
    doc = json.loads(path.read_text())
    if doc.get("format") != AC.FORMAT:
        raise ValueError(f"{path} is {doc.get('format')!r}, not {AC.FORMAT}")
    return {"label": doc["label"], "run": doc["run"], "from": str(path), "bits_per_commit_period": doc["bits"],
            "bits_per_phase": doc["phase_bits"] or doc["by_phase"], "phases": TEST_PHASES,
            "commit_periods_measured": len(doc["commit_periods"]), "finishes": doc["finishes"],
            "prompt_len": doc["prompt_len"], "output_len": doc["output_len"]}


def lines(out: Path, name: str) -> list[str]:
    f = out / name
    return f.read_text().splitlines() if f.is_file() else []


def charged(b: Mapping[str, Any]) -> dict[str, Any]:
    """ℓ_{a,j} as the bound charges each execution phase (`CHARGED`), and their share of K."""
    free = [t["bits"] for t in b["terms"] if t["symbol"] == "\\ell_{a,j}"]
    return {"is": CHARGED, "bits_per_phase": free, "share": float(Fraction(sum(free), b["K"])),
            "lean": "scans[j].free"}


def advice_cap(reg: Any) -> int:
    """ℓ_a (`CAP`) of the run's published advice registration: its compiler's cap_bits, a whole number of at least 1
    bits. ValueError, by name, where it states none (an ℓ_a of 0 would charge a tied audit's advice nothing) or its
    advice's length isn't padded to it."""
    cap = ((reg.get("compiler") if isinstance(reg, Mapping) else None) or {}).get("cap_bits")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError(f"its compiler states no cap_bits of at least 1 bit ({cap!r}), the ℓ_a the figure of record "
                         "charges each tied audit while a check or a premise is open, so it charges none rather than 0")
    bits = reg["binding"]["advice"]["bits"]
    if bits != cap:
        raise ValueError(f"its advice's length {bits} isn't padded to its compiler's cap {cap}")
    return cap


def caps(case: Mapping[str, Any], cap: int) -> list[int]:
    """Each audit's ℓ_a (`CAP`), the run's ``cap``, in the order of the case's commitment. ValueError where an audit's
    registration binds an advice or a compiler, whose bits no term of K charges."""
    audits = json.loads(bytes.fromhex(case["commitment"]))["audits"]
    if bound := [i for i, r in enumerate(audits)
                 if any((r.get("window") or {}).get(k) is not None for k in ("advice", "compiler"))]:
        raise ValueError(f"the registrations of audits {bound} bind an advice or a compiler, whose bits no term of K "
                         "charges (records_audit.py period refuses them)")
    return [cap] * len(audits)


def circuits(out: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    """Each tied audit's circuit files (`CIRCUITS`) by the digests the period's params pin, public, in the Program's
    template order, each with its file in OUT and whether the run wrote it; ``missing`` where the params pin none
    (`NO_CIRCUITS`). ValueError where a pin isn't a SHA-512."""
    pinned = []
    for a in case["params"]["audits"]:
        files = []
        for c in a.get("circuits") or ():
            if not re.fullmatch(r"[0-9a-f]{128}", h := c["sha512"]):
                raise ValueError(f"template {c['template']!r}'s pin {h!r} isn't a SHA-512")
            f = f"{CIRCUITS}/{h}.txt"
            files.append({"template": c["template"], "sha512": h, "file": f, "present": (out / f).is_file()})
        pinned.append(files)
    return {"public": True, "audits": pinned} if any(pinned) else {"missing": NO_CIRCUITS}


def statement(lock: Mapping[str, Any], g: str) -> str | None:
    """The digest ``lock`` gives the statement of its guarantee ``g``: its signature's type, a definition among the
    lock's reads. None where the lock lists no such definition."""
    stated = " ".join((lock.get("guarantees", {}).get(g) or {}).get("signature") or ()).partition(" : ")[2].strip()
    return next((r["definitions"][stated] for r in lock.get("reads", {}).values()
                 if stated in (r.get("definitions") or {})), None)


def premises(locks: Sequence[Path], root: Path = ROOT) -> dict[str, Any]:
    """The tight K's premises as ``locks`` record their discharges (`DISCHARGE`): each of `PREMISES` with the theorems
    no lock lists (``missing``), those one lists at a statement `DISCHARGES` doesn't pin (``unpinned``, each with the
    lock's digest), the named assumptions of those one lists that don't stand and their pins don't take (``assumes``)
    and the premises their pins take (``takes``), each guarantee of `RESTS_ON` with its named assumptions, its
    statement's digest and what it takes (``restated`` where that digest isn't the one `RESTS_ON` read, `RESTATED`),
    what K counts (``counts``, `COUNTS`), and ``open``: each premise not discharged, then each named assumption of a
    guarantee of `RESTS_ON` neither standing nor taken by name, and each such guarantee no lock lists or one lists
    restated. A lock that can't be read lists nothing, and ``unread`` names it, by its path under ``root`` (the tree)
    where it is one, with why."""
    listed: dict[str, Any] = {}
    stated: dict[str, str | None] = {}
    unread: list[str] = []
    for lock in locks:
        try:
            doc = json.loads(lock.read_text())
            listed |= doc["guarantees"]
            stated |= {g: statement(doc, g) for g in doc["guarantees"]
                       if g in RESTS_ON or any(g in p["discharged_by"] for p in PREMISES.values())}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            name = lock.relative_to(root) if lock.is_relative_to(root) else lock
            unread.append(f"{name}: {type(e).__name__}: {e.strerror if isinstance(e, OSError) else e}")
    found, ready = {}, {}
    for name, p in PREMISES.items():
        pins = {t: DISCHARGES.get(t) or {"statement": None, "takes": {}} for t in p["discharged_by"]}
        missing = [t for t in pins if t not in listed]
        unpinned = {t: stated.get(t) for t, pin in pins.items()
                    if t in listed and (pin["statement"] is None or stated.get(t) != pin["statement"])}
        assumes = sorted({a for t, pin in pins.items() if t in listed for a in listed[t]["assumptions"]
                          if a not in pin["takes"]} - set(STANDING))
        takes = sorted({v for pin in pins.values() for v in pin["takes"].values()} & set(PREMISES))
        found[name] = {"is": p["is"], "discharged_by": list(p["discharged_by"]), "missing": missing,
                       "unpinned": unpinned, "assumes": assumes, "takes": takes}
        if not (missing or unpinned or assumes or not p["discharged_by"]):
            ready[name] = set(takes)
    done: set[str] = set()
    while new := {n for n, t in ready.items() if n not in done and t <= done}:
        done |= new
    open_ = [n for n in PREMISES if n not in done]
    rests: dict[str, Any] = {}
    for g, r in RESTS_ON.items():
        if g not in listed:
            rests[g] = {"missing": "not in the lock", "takes": r["takes"]}
            continue
        rests[g] = {"assumptions": list(listed[g]["assumptions"]), "statement": stated.get(g), "takes": r["takes"]}
        if r["statement"] is None or stated.get(g) != r["statement"]:
            rests[g]["restated"] = RESTATED
    open_ += sorted({a for g, x in rests.items() for a in x.get("assumptions", ())
                     if a not in RESTS_ON[g]["takes"] and a not in STANDING}
                    | {g for g, x in rests.items() if "missing" in x or "restated" in x})
    return {"is": DISCHARGE, "from": "Security/lean-audit.json and Security/Properties/lean-audit.json guarantees "
                                     "and reads",
            "premises": found, "rests_on": rests, "standing": STANDING, "counts": COUNTS, "open": open_,
            "unread": unread}


def of_record(public: Mapping[str, Any], ell: Sequence[int], K: int, prem: Mapping[str, Any]) -> dict[str, Any]:
    """The figure of record (`OPEN`) from the period's public, the audits' caps ``ell`` and the tight K's premises
    ``prem`` (`premises`): ``K``, the checks the public holds, what is open (those checks and the premises), and the
    advice ℓ_a + 1 charges, summed over each phase's tied audits (the public's ``scans[j].tied``, else every audit:
    `TIED_ALL`)."""
    held = [c for c in CHECKS if c in (public.get("held") or ())]
    open_ = [c for c in CHECKS if c not in held] + list(prem["open"])
    tied = [s.get("tied") for s in public["scans"]]
    per = [sum(ell[int(i)] + 1 for i in (t if t is not None else range(len(ell)))) for t in tied]
    return {"K": K + (sum(per) if open_ else 0), "held": held, "open": open_, "is": OPEN, "premises": prem,
            "advice": {"bits_per_phase": per,
                       "tied": "scans[j].tied" if all(t is not None for t in tied) else TIED_ALL}}


def figure(out: Path, case: Mapping[str, Any], public: Mapping[str, Any], K: int,
           prem: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The figure of record (`of_record`) and ``advice_charge``'s ``cap`` from OUT's advice registration
    (`REGISTRATION`) and the case's commitment: both the one ``missing`` or ``error`` where either can't be read."""
    try:
        cap = advice_cap(json.loads((out / REGISTRATION).read_text()))
    except FileNotFoundError:
        return (why := {"missing": NO_REGISTRATION}), why
    except (KeyError, TypeError, ValueError) as e:
        return (why := {"error": f"{REGISTRATION}: {type(e).__name__}: {e}"}), why
    try:
        ell = caps(case, cap)
        return of_record(public, ell, K, prem), {"ell_a": ell, "is": CAP}
    except (IndexError, KeyError, TypeError, ValueError) as e:
        return (why := {"error": f"{CASE} commitment: {type(e).__name__}: {e}"}), why


def named(path: str) -> list[re.Pattern[str]]:
    """Each file a `REPROOF` path names, as a pattern over a listing's paths: ``, `` separates paths, ``{a,b}`` is each of
    a and b, ``<x>`` any one name, and a trailing ``/`` a directory holding at least one file."""
    out = []
    for part in path.split(", "):
        m = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", part)
        for p in [m[1] + x + m[3] for x in m[2].split(",")] if m else [part]:
            rx = "".join("[^/]+" if t.startswith("<") else re.escape(t) for t in re.split(r"(<[^>]+>)", p))
            out.append(re.compile(rx + (".+" if p.endswith("/") else "")))
    return out


def reproof(out: Path, copy: str | None = None, listing: Path | None = None) -> dict[str, Any]:
    """The re-proof inputs (`REPROOF`), each with where it is and what it holds; of OUT's whether the run wrote it, and of
    the serve dir's, where ``copy`` (NODE:/PATH) and ``listing`` (each file under PATH, one path relative to it a line)
    are given, whether the copy's listing holds each file its path names and at least one under each directory it
    names."""
    files = None
    if copy is None or listing is None:
        kept: dict[str, Any] = {"missing": NO_COPY}
    elif not (node_at := copy.partition(":"))[0] or not node_at[2].startswith("/"):
        kept = {"error": f"--reproof-copy {copy!r} isn't NODE:/PATH"}
    else:
        try:
            files = [x for x in listing.read_text().splitlines() if x]
            kept = {"node": node_at[0], "path": node_at[2]}
        except (OSError, UnicodeDecodeError) as e:
            kept = {"error": f"the copy's listing (--reproof-listing) can't be read: {type(e).__name__}"}
    inputs = {}
    for name, (at, path, what) in REPROOF.items():
        inputs[name] = {"in": at, "path": path, "holds": what}
        if at == "OUT":
            inputs[name]["present"] = (out / path).exists()
        elif files is not None:
            inputs[name]["copied"] = all(any(rx.fullmatch(f) for f in files) for rx in named(path))
    if files is not None:
        kept["not_copied"] = [n for n, x in inputs.items() if x.get("copied") is False]
    return {"inputs": inputs, "copy": kept, "kept": REPROOF_KEPT}


def record(out: Path, advice_charge: Path | None = None, copy: str | None = None,
           listing: Path | None = None) -> dict[str, Any]:
    """OUT's onsite record (module docstring), with the re-proof inputs' ``copy`` and its ``listing`` where given
    (`reproof`); FileNotFoundError where `served_verdicts.py record` wrote no report."""
    report = json.loads((out / "record" / "verdicts.json").read_text())
    if report.get("format") != EXTENDS:
        raise ValueError(f"record/verdicts.json is {report.get('format')!r}, not {EXTENDS}")
    advice = {"served": ADVICE, "charged": {"missing": "no bound"}, "cap": {"missing": "no bound"},
              "measured": measured(advice_charge)}
    rows = report.get("protocols") or {}
    found = composites(out)
    v = {name: row_verdict(rows, ROWS[name]) for name in ("sampled proofs", "PoUW")}
    v["inference-only"] = case_verdict(found, REMOTE, INFERENCE_ONLY)
    v["remote"] = composite_verdict(found, REMOTE, "remote's composite didn't run: the run has no "
                                                   "composite/remote/composite.json")
    v["the network certifier"] = row_verdict(rows, ROWS["the network certifier"])
    v["onsite"] = composite_verdict(found, ONSITE, f"the run has no composite of {ONSITE}")
    v = {k: v[k] for k in VERDICTS}
    net = rows.get(ROWS["the network certifier"]) or {}
    answer = net.get("answer") if isinstance(net.get("answer"), Mapping) else {}
    profile = answer.get("profile") if isinstance(answer.get("profile"), Mapping) else {}
    tree = next((x for x in lines(out, "machine.txt") if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", x)), None)
    rec: dict[str, Any] = {
        "format": FORMAT, "extends": EXTENDS, "run": report.get("run"), "tree": tree, "node": report.get("node"),
        "verdicts": v,
        "of_record": {"holds": all(x["verdict"] == "accept" for x in v.values()),
                      "missing": [k for k, x in v.items() if x["verdict"] == "missing"],
                      "not_accepting": [k for k, x in v.items() if x["verdict"] not in ("accept", "missing")]},
        "placement": profile.get("placement"), "advice_charge": advice, "composite_error": composite_error(rows, None),
        "reproof": reproof(out, copy, listing)}
    nv = v["the network certifier"]
    if nv["verdict"] != "accept":
        why = (f"the network certifier's verdict is {nv['verdict']}"
               + (f" ({nv['why']})" if nv.get("why") else "") + ": only its accepted answer certifies K")
        return rec | {"bound": {"missing": why}, "periods": {"missing": "no bound"}}
    if nv.get("protocol") != NETWORK:
        return rec | {"bound": {"error": f"the network certifier's answer is {nv.get('protocol')}'s, not {NETWORK}'s"},
                      "periods": {"missing": "no bound"}}
    try:
        b = rec["bound"] = bound(answer.get("public"))
    except ValueError as e:
        return rec | {"bound": {"error": str(e)}, "periods": {"missing": "no bound"}}
    b["verdict_line"] = next((x for x in reversed(lines(out, "verdicts.txt")) if x.startswith(LINE)), None)
    rec["composite_error"] = composite_error(rows, eps_of(b))
    advice["charged"] = charged(b)
    b["period_s"] = None
    try:
        case = json.loads((out / CASE).read_text())
    except FileNotFoundError:
        b["connections"] = {"missing": f"no {CASE}: they are its params' caps"}
        b["of_record"] = advice["cap"] = {"missing": f"no {CASE}: its commitment holds the tied audits' registrations"}
        b["circuits"] = {"missing": f"no {CASE}: its params pin the circuit files"}
        return rec | {"periods": {"missing": f"no {CASE}: the execution phase's length is its params' schedule"}}
    except ValueError as e:
        b["connections"] = b["of_record"] = advice["cap"] = b["circuits"] = {"error": f"{CASE}: {type(e).__name__}: {e}"}
        return rec | {"periods": b["of_record"]}
    try:
        b["circuits"] = circuits(out, case)
    except (KeyError, TypeError, ValueError) as e:
        b["circuits"] = {"error": f"{CASE} params: {type(e).__name__}: {e}"}
    b["of_record"], advice["cap"] = figure(out, case, answer["public"], b["K"], premises(LOCKS))
    try:
        phase_s, connections, capacity = schedule(case, b["phases"])
    except (KeyError, TypeError, ValueError) as e:
        b["connections"] = {"error": f"{CASE}: {type(e).__name__}: {e}"}
        return rec | {"periods": {"error": f"{CASE}: {type(e).__name__}: {e}"}}
    b["connections"] = {"count": connections, "from": CONNECTIONS}
    b["period_s"] = str(b["phases"] * phase_s)
    if "K" not in b["of_record"]:
        why = b["of_record"].get("error") or b["of_record"]["missing"]
        return rec | {"periods": {"missing": f"no figure of record: {why}"}}
    eps, m, K = eps_of(b), b["phases"], b["of_record"]["K"]
    rec["periods"] = periods(K, eps, m, phase_s, connections, capacity)
    if K != b["K"]:
        rec["periods_tight"] = {"conditional_on": b["of_record"]["open"],
                                **periods(b["K"], eps, m, phase_s, connections, capacity)}
    return rec


def summary(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ONSITE`` line: each verdict, whether the run is of record, K of record, the tight K and the checks it is
    conditional on, its δ_net's terms, eps's split and connections, the composite's stated error, the year's datacenter
    cell, and the measured advice charge's test run."""
    b, p = rec.get("bound") or {}, rec.get("periods") or {}
    year = (p.get("year") or {}).get("datacenter")
    m, fig = rec["advice_charge"]["measured"], b.get("of_record") or {}
    return {"verdicts": {k: x["verdict"] for k, x in rec["verdicts"].items()}, "of_record": rec["of_record"]["holds"],
            "K": fig.get("K"), "K_tight": b.get("K"), "open": fig.get("open"),
            "delta_net": None if "delta_net" not in b else delta_net(eps_of(b), b["phases"]),
            "delta": None if "delta" not in b else {k: b["delta"][k] for k in ("scan", "charged")},
            "composite_error": {k: rec["composite_error"][k] for k in ("stated", "vacuous")},
            "connections": (b.get("connections") or {}).get("count"),
            "year_datacenter": None if year is None else {
                k: year[k] for k in ("bits", "share", "share_served", "share_of_capacity", "delta_net", "how")},
            "advice_measured_on": m.get("run"), **{k: b[k] for k in ("missing", "error") if k in b}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="the served run's public $OUT")
    ap.add_argument("--advice-charge", type=Path,
                    help="a test run's advice-charge.json (advice_charge.py), stated beside the bound as that run's")
    ap.add_argument("--reproof-copy", metavar="NODE:/PATH",
                    help="where the serve dir's re-proof inputs were copied off the served node, keeping its layout")
    ap.add_argument("--reproof-listing", type=Path,
                    help="the copy's listing, kept off OUT: each file under PATH, one path relative to it a line")
    ap.add_argument("--to", type=Path, help="the record's path (default OUT/record/onsite.json)")
    a = ap.parse_args(argv)
    try:
        rec = record(a.out, a.advice_charge, a.reproof_copy, a.reproof_listing)
    except FileNotFoundError as e:
        print(f"ONSITE: not written: {e.filename} is missing", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ONSITE: not written: {e}", file=sys.stderr)
        return 1
    dest = a.to or a.out / "record" / "onsite.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print("ONSITE " + json.dumps(summary(rec), sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
