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
caps aren't those; and ``verdict_line`` is the run's published line of the network certifier's verdict (`LINE` in
verdicts.txt), None where it has none. A public whose terms don't sum to K, or that isn't the rows form, is the bound's
``error``. ``of_record`` is the figure of record (`OPEN`): that K only where the verdict's public holds every check the
tight ℓ_{a,j} rests on (``held``, `CHECKS`) and none of its premises is open (``premises``, `DISCHARGE`: each of
`PREMISES` discharged by theorems in Security's locks that take no named assumption but a standing one, `STANDING`), and
while one of either is ``open``, by name, K plus ℓ_a + 1 bits for each audit tied to each
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
cell gives K in bits (the least integer at or above a scaled one), its share of one copy of the weights
(`leak_bounds.WEIGHTS`), its share of the egress its execution phases can carry (``share_of_capacity``, the schedule's
capacity in ``egress``, `EGRESS`; the served egress is missing, `SERVED_EGRESS`), its δ_net's terms over the execution
phases it counts (``delta_net``), whether eps alone is
already vacuous (at least 1), the eps a phase under which it wouldn't be (``eps_per_phase_below``, beside the bound's
``delta``) and the statement it rests on; beside that pooled total, ``per_phase`` is the same over
one execution phase, proved only where m is 1.

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
#: Security's locks in this tree (`tools/verity/lean/check.py`'s record of each guarantee's named assumptions): the
#: guarantees', and the Properties' that discharge premises at the deployed code
ROOT = HERE.parents[1]
LOCKS = (ROOT / "Security" / "lean-audit.json", ROOT / "Security" / "Properties" / "lean-audit.json")
#: the guarantees the tight K rests on, by their names in `LOCKS`
RESTS_ON = ("NetTiming.SecurityProofs.PeriodLeakAccepted",)
#: each premise the tight K rests on, by the name ``open`` gives it: what it says, the theorems that discharge it at the
#: deployed code by their names in `LOCKS`' guarantees (none named yet leaves it open), and the named assumptions of
#: `RESTS_ON` it stands for, which its discharge decides
PREMISES: dict[str, dict[str, Any]] = {
    "wires-fit": {"is": "the wires part of BoundaryGates.Fits at the tied audit's statements: the decoder link, and the "
                        "period check running Circuit.check",
                  "discharged_by": ("Properties.NetworkCertifier.wiredScopeAcc",
                                    "Properties.NetworkCertifier.circuitWiresFit",
                                    "Properties.NetworkPeriod.rowsWiresChecked"), "assumes": ()},
    "session-wires": {"is": "SessionWires: each sampled proof block's wired inputs equal their writers' committed rows, "
                            "proved inside its own session, whose statement reads the writer's committed leaf by path and "
                            "whose circuit equates it with the wired input, so no opening reaches the verifier; "
                            "OneStageZk.verdicts accepting only where each sampled reader's wire is the layout's, at the "
                            "registered position, root and path; and SessionOpens and the binding from the sampled leg "
                            "(`sampled-leg`), never a named assumption. Open until the theorem that "
                            "OneStageZk.verdicts' acceptance gives it is named and in the lock",
                      "discharged_by": (), "assumes": ()},
    "free-count": {"is": "each tied audit's free at least #(freeInputs \\ reg) at the wired layout",
                   "discharged_by": ("Properties.NetworkCertifier.circuitCounts",), "assumes": ()},
    "held-performed": {"is": "the public's held names only the checks the period check performed, each holding of it",
                       "discharged_by": ("Properties.NetworkPeriod.rowsHeld",), "assumes": ()},
    "circuit-held": {"is": "each tied audit's class circuit held",
                     "discharged_by": ("Properties.NetworkPeriod.rowsCircuitHeld",), "assumes": ()},
    "sampled-leg": {"is": "the sampled facts the period takes (KS, SessionOpens, RegOpens: the registered reads' "
                          "openings) from the sampled leg's guarantee, Verity.Vocab.SampledProofsSound at one-stage v2, "
                          "whose verifier of record is Lean, through a projection lemma, as a hypothesis on the layer "
                          "below: never SampledCodeRuns or another premise about the deployed Python. Open until that "
                          "guarantee's proof and the lemma are in the lock",
                    "discharged_by": (), "assumes": ()},
}
#: the named assumptions that may stand under the figure of record, each on an auditor-trusted device
STANDING = {"Verity.Assumptions.TrustedHost": "trusted-host: the network certifier and its host are the auditor's"}
DISCHARGE = ("the tight K's premises (`PREMISES`), each open until every theorem that discharges it at the deployed code "
             "is in Security's lock with no named assumption but a standing one (`STANDING`), and open while none is "
             "named; and each named assumption of a guarantee of `RESTS_ON` that is neither standing nor a premise's, by "
             "its own name, with each such guarantee the lock doesn't list. No premise about our own Python stands: "
             "soundness is in Lean or rests on an auditor-trusted device")
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
FROM_BOUND = "bound: the run's K and eps over its m execution phases"
FROM_SCHEDULE = f"{CASE} params.schedule: an execution phase's T buckets of τ"
STATEMENTS = {"run": LB.STATEMENTS["per_period"], "node": LB.STATEMENTS["per_period"],
              "datacenter": LB.STATEMENTS["datacenter"], "world": LB.STATEMENTS["datacenter"]}
NO_MEASUREMENT = ("no test run's measurement given (--advice-charge: `advice_charge.py`'s record of honest serving on "
                  "public inputs)")
TEST_PHASES = ("the test run's own: T buckets of τ from its first Commit period's engine clock reading, not the network "
               "certifier's phase boundaries, so a Commit period near a boundary can count in the next phase")


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
            terms.append(t | ({"missing": "its row states no δ"} if d is None else {"delta": str(Fraction(str(d)))}))
        except ValueError:
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
    return {"bits": math.ceil(bits), "share": float(bits / LB.WEIGHTS.bits),
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
    out: dict[str, Any] = {"phase_s": str(phase_s), "weights": LB.WEIGHTS.label, "run": run,
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


def advice_cap(reg: Mapping[str, Any]) -> int:
    """ℓ_a (`CAP`) of the run's published advice registration: its compiler's cap_bits. ValueError where its advice's
    length isn't padded to it."""
    cap, bits = int(reg["compiler"]["cap_bits"]), int(reg["binding"]["advice"]["bits"])
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


def premises(locks: Sequence[Path]) -> dict[str, Any]:
    """The tight K's premises as ``locks`` record their discharges (`DISCHARGE`): each of `PREMISES` with the theorems
    no lock lists (``missing``) and the named assumptions of those one does that don't stand (``assumes``), each
    guarantee of `RESTS_ON` with its named assumptions, and ``open``: each premise not discharged, then each assumption
    of `RESTS_ON` neither standing nor a premise's, and each of its guarantees no lock lists. A lock that can't be read
    lists nothing, and ``unread`` names it, by its path in the tree where it is one, with why."""
    listed: dict[str, Any] = {}
    unread: list[str] = []
    for lock in locks:
        try:
            listed |= json.loads(lock.read_text())["guarantees"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            name = lock.relative_to(ROOT) if lock.is_relative_to(ROOT) else lock
            unread.append(f"{name}: {type(e).__name__}: {e.strerror if isinstance(e, OSError) else e}")
    found, open_ = {}, []
    for name, p in PREMISES.items():
        missing = [t for t in p["discharged_by"] if t not in listed]
        assumes = sorted({a for t in p["discharged_by"] if t in listed for a in listed[t]["assumptions"]} - set(STANDING))
        found[name] = {"is": p["is"], "discharged_by": list(p["discharged_by"]), "missing": missing, "assumes": assumes}
        if missing or assumes or not p["discharged_by"]:
            open_.append(name)
    covered = set(STANDING) | {a for p in PREMISES.values() for a in p["assumes"]}
    rests = {g: {"assumptions": list(listed[g]["assumptions"])} if g in listed else {"missing": "not in the lock"}
             for g in RESTS_ON}
    open_ += sorted({a for r in rests.values() for a in r.get("assumptions", ()) if a not in covered}
                    | {g for g, r in rests.items() if "missing" in r})
    return {"is": DISCHARGE, "from": "Security/lean-audit.json and Security/Properties/lean-audit.json guarantees",
            "premises": found, "rests_on": rests, "standing": STANDING, "open": open_, "unread": unread}


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


def record(out: Path, advice_charge: Path | None = None) -> dict[str, Any]:
    """OUT's onsite record (module docstring); FileNotFoundError where `served_verdicts.py record` wrote no report."""
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
        "placement": profile.get("placement"), "advice_charge": advice, "composite_error": composite_error(rows, None)}
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
        return rec | {"periods": {"missing": f"no {CASE}: the execution phase's length is its params' schedule"}}
    except ValueError as e:
        b["connections"] = b["of_record"] = advice["cap"] = {"error": f"{CASE}: {type(e).__name__}: {e}"}
        return rec | {"periods": b["of_record"]}
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
                k: year[k] for k in ("bits", "share", "share_of_capacity", "delta_net", "how")},
            "advice_measured_on": m.get("run"), **{k: b[k] for k in ("missing", "error") if k in b}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="the served run's public $OUT")
    ap.add_argument("--advice-charge", type=Path,
                    help="a test run's advice-charge.json (advice_charge.py), stated beside the bound as that run's")
    ap.add_argument("--to", type=Path, help="the record's path (default OUT/record/onsite.json)")
    a = ap.parse_args(argv)
    try:
        rec = record(a.out, a.advice_charge)
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
