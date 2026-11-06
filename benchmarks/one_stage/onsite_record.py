"""onsite-record: the onsite demo's record of one served run (``record/onsite.json``, `FORMAT`): every verdict its run of
record needs, the network certifier's bound term by term, and that bound over each monitoring period for the serving
node, a datacenter and the world. It reads only the run's public $OUT and decides nothing:

    python benchmarks/one_stage/onsite_record.py OUT [--window-steps S --max-tokens M] [--advice-charge FILE]
        [--certifier-key RECORD --site-registration FILE] [--inference-only-case-policy FILE]
        [--sampled-proofs-case FILE --public-pin FILE] [--to FILE]
    python benchmarks/one_stage/onsite_record.py --beside RECORD [RECORD ...] --to FILE

`served_verdicts.sh` runs it after its composite step on a run with the network certifier's piece, and its one
``ONSITE`` line goes to the run's verdicts.txt. That line, and each run's in the ``ONSITE-BESIDE`` line, names the
composite of record by its id and states that no verified compute is credited (``compute``, `NO_COMPUTE_CLAIM`).

**Serving** (``serving``, `SERVING`): the run's S (served_verdicts.sh's WINDOW_STEPS) and MAX_TOKENS as its caller
states them, which no file of OUT does, else ``missing`` (`NO_SERVING`); they label every figure of the record and its
line. ``--beside`` sets runs' records beside each other (`BESIDE`, `BESIDE_IS`): each run's figure of record, what is
open under it, ℓ_a and period cells, labelled with its run, tree and serving settings, its composite of record, that it
credits no verified compute and what they are conditional on, and none carried to another's settings; a record stating
no serving settings is refused (`UNLABELLED`).

**Verdicts** (`VERDICTS`), each ``{protocol, verdict, from}`` with the answer's ``reason`` or ``refusal``, or, where the
run wrote no answer, ``missing`` and ``why``:

* sampled proofs: its leg in the composite of record, the outcome of remote's sampled proofs case there
  (`SAMPLED_ID`: VStar.Recursive.Sampled on OneStageZk's case, the proofs themselves), never ``record/verdicts.json``'s
  ``integrity`` row, the records audit's one-stage v1 answer, which is no verdict (`V1_UNREAD`): where the composite of
  record has no such case, not established, the row's answer beside it (``answer``), and missing where the row is too;
* PoUW: ``record/verdicts.json``'s ``work`` row (`served_verdicts.py record`);
* inference-only: its case's outcome in remote's composite, or in the composite of record where that is the one
  without PoUW, with where its case's α and Γ came from (``alpha_gamma``, `alpha_gamma`), never a registered policy,
  which none records: read off the run's own division, so the verdict confirms the division against itself, where the
  case's private policy.json (``--inference-only-case-policy``, of which only ``read`` is read) says so, the run's
  ALPHA and GAMMA where the case has none, and ``missing`` where the record wasn't handed it;
* remote: remote's Lean verdict, ``composite/remote/composite.json`` (`composite.py`), with what it couldn't establish;
* the network certifier: the ``communication`` row, `NetworkCheck.Period`'s answer (`NETWORK`), or, where the run's
  product drops it (remote's, with the split certifier behind its front) and its period check answered, the row that
  answer gives (``pieces/certifier/answer.json``, `served_verdicts.piece_row`), as onsite's report composes it;
* onsite: the composite of record's Lean verdict, its id the verdict's ``protocol``: the network certifier's composite
  without PoUW where the run has one (`WITHOUT_POUW`), else the Lean composition of remote's three subprotocols and the
  network certifier over one registration, the composite whose product is `ONSITE`; missing where the run has none.

While `certifier-key` is open (`PREMISES`, `KEY_OPEN`), an accepted network certifier's verdict is ``not
established``, its Lean answer's accept beside it (``answer``), and its bound conditional on it by name; so is it
where `auditor-listing` doesn't hold (`LISTING_UNMET`, `FILES_UNMET`), with no bound, and where its registered-files
condition doesn't, so is the accepted verdict of remote's composite and of the composite of record where it reads
remote's sampled proofs case, sampled proofs' among them; where that case's params name no public file, so is
sampled proofs' (`PUBLIC_UNBOUND`). Sampled proofs' ``credit`` (`CREDIT`) names the drawn members it credits, those
whose --registered file the audit completed at the window's committed roots and whose Lean verdict accepted it
reporting exactly its reads (the audit's published record, `AUDIT`), and lists every other drawn member as not
credited, by name, with why; ``missing`` where the record states no completion, and where sampled proofs' verdict is
not established, which credits none (`UNESTABLISHED_CREDIT`). ``of_record`` holds when every verdict it reads accepted
and every drawn member is credited, the condition for the demo's run of record, naming its composite of record by id (``composite``), the verdicts not established, the
members not credited and the verdicts it doesn't read (``not_claimed``: PoUW's, `NOT_CLAIMED`, listed as it is, missing
and development where its row is, and under a composite without PoUW remote's too, `NOT_CLAIMED_WITHOUT_POUW`), and
the verified compute it credits, none (``verified_compute``, `POUW`); ``placement`` is the network certifier's
answer's, which host ran each party.

**Checked on the run** (``checked``, `CHECKED`): the auditor's configuration of the run, each an obligation the record
checks on the auditor's inputs, never a named assumption and never charged, each with ``holds`` and, where it doesn't,
``why``. `auditor-record` (`auditor_record`): PeriodLeakRegistered's hA, hr and hkey, that the period's LMS parameters,
key and session are the auditor's record of its network certifier (``--certifier-key``) and that the auditor's
registration (``--site-registration``) holds that key for a network certifier its log places, read from those two
inputs, each named by its SHA-512 (``from``), and the period's parameters, held by SHA-512 to those its check read
(the composite of record's), never the prover's node or its deployment.json; `certifier-key` closes only where it
holds and the lock holds that guarantee taking them. The registration's link figures (beta, latency) are stated, not
measured, and nothing here reads them; that the node has no egress but the uplink the network certifier splits, which
the registration asserts, is the auditor's placement under trusted-host (`STANDING`), which no file establishes.
`auditor-listing` (`auditor_listing`): each circuit file the period lists is, by SHA-512, the file its audit's
sessions were verified against (`PERIOD`); and the remote theorem's registered-files condition (`registered_files`):
each file remote's sampled proofs case (``--sampled-proofs-case``) registers, the circuit file and the public file, is
by SHA-512 the one its params θ name (proofs.circuit, proofs.public), θ held to the params each composite that reads the
case states, the files hashed here and never read off the evidence, and the public file the one a5 pinned at staging,
before the draw (``--public-pin``, `pinned`), the pin's bodies held to the registration the challenger's coins committed
to by their leaf layer.

**Sampled proofs' guarantee** (``sampled_proofs``, `SAMPLED`), from its accepted row's answer, else ``missing``: what
its draw establishes, ε at δ, per window (``per_window``: its bound at its profile's δ over the window's n proof blocks) and
pooled (``pooled``, `POOLED`): at most ⌊ln(1/δ)/q⌋ incorrect proof blocks over any number of windows, each drawing k of
its n proof blocks (q = k/n), by `Flock.Guarantees.IntegrityHorizon`, and for each monitoring period of
`leak_bounds.PERIODS` that count for the serving node, a datacenter's and the world's links (each link's at δ / links,
summed), each with its share of the period's proof blocks at a window an execution phase (`WINDOW_S`), scaled from the
run's window by its rule (`POOLED_RULES`); a law other than a k-subset has no pooled figure (`NOT_POOLED`). ``refused``
names the draws a run of record doesn't make (`MULTI`), and it is conditional on the coins while they are open.

**Coins** (``coins``, `COINS`): sampled proofs' coins as the run's COINS lines in verdicts.txt name them (the served
run's, or the follow-on's where it audits), and the network certifier's chain draws as its published report states
each one's law (``draws``, `CHAIN_DRAWS`); ``open`` unless each COINS line names the live challenger (`LIVE_COINS`) and,
where the network certifier accepted, each chain draw takes everything, naming each draw that doesn't
(``open_draws``), and ``missing`` where the run published no COINS line. While they are open, live-coins is named
wherever a figure rests on that draw: the sampled-proofs verdict's, ``of_record``'s and the composite's error's
``conditional_on``, and the figure of record's ``open`` and ``conditional_on`` and its periods'.

**One program a day** (``one_program_a_day``, `DAY`): not claimed, and open by its name in the record and its line
(``not_claimed``): R2's day isn't yet the challenger's signed time, and the audit doesn't yet read the challenger's
signed registration log. Beside it, the day each of the run's ADVICE-LINEs says the server declared B for, and
``missing`` where the run published none; the day's registration and its refusals still run as `DAY` says.

**PoUW's freed compute** (``pouw_freed_compute``, `POUW`): not claimed, by its name in the record and its line
(``not_claimed``), and credited none: the PoUW verdict carries no figure of its row, so no freed compute is in any
total here. Beside it, ``served``, the PoUW piece each COINS line says the run served.

**The composite's error** (``composite_error``, `COMPOSITE`): the sum over the composite of record's subprotocols
(``composite``) of each one's error, each subprotocol's δ with where the run states it: sampled proofs' and, where the
composite has it, PoUW's from their rows (their answers' integrity-profile δ), the network certifier's from its bound
(its eps), and inference-only's missing
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
`PREMISES` discharged by theorems in Security's locks, each at the statement `DISCHARGES` pins, taking no named
assumption but a standing one (`STANDING`) or one its pin names and no premise that is open, or, where a guarantee it
rests on states it of the auditor's inputs, closed by that guarantee's pin and the run's check (`CHECKED`); and each guarantee it
rests on in the lock at the statement whose hypotheses and antecedents `RESTS_ON` names, beside what K counts,
`COUNTS`) and the coins are live (``coins``), and while any is ``open``, by name, K plus ℓ_a + 1 bits for each audit
tied to each execution phase's scan (``advice``: the bits each phase adds), the figure, with no tighter figure beside
it, only how it tightens (``tight``, `TIGHT`), conditional on the coins and on each open premise of `UNCHARGED` while they are open
(``conditional_on``). Beside the bound, ``advice_charge``: the served run's
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
of `leak_bounds.PERIODS` for the serving node, the census's reference datacenter and the world, conditional on the coins
and each open premise of `UNCHARGED` while they are open (``conditional_on``). A node's cell is proved where the run's
period covers the monitoring period's seconds (its m execution phases), and otherwise scaled from the run to the period; a
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
one execution phase, proved only where m is 1. ``eps`` (`pooled_eps`, `POOLED_EPS`) is eps per window (an execution
phase), over the run's period and pooled over each monitoring period's cell at its windows, each split into the scans'
δ and the charged part, with the number of tied audits the period reads at δ_i (``tied_audits``, `TIED_READ`), a
count over the period's windows too.

**Re-proof inputs** (``reproof``, `REPROOF`): what a later re-proof of the run's sampled units needs, each wire a sampled
unit reads then proved inside its session: the committed values and their openings for every sampled unit and the units
that write what it reads, the registration, the draw, its receipt and the challenger's release of its coins, and the
public files, each by where it is (the serve dir's by their paths there, naming no window) and what it holds, with whether
the run wrote each of OUT's and how each must be kept (``kept``, `REPROOF_KEPT`). The serve dir's are witness, kept off
OUT and custody in a copy off the served node (``--reproof-copy`` NODE:/PATH, ``--reproof-listing``, a listing of it kept
off OUT): ``copy`` names its node and path, and each serve-dir input says whether the listing holds each file its path
names and at least one file under each directory it names, never their bytes or digests; with no copy given, ``copy`` is
missing with why. Under H1_TIE (``--h1-tie``) the circuit is the verifier's record reading H1's tables, the one V3c
holds the tied commit to.

The record's counts and indices are the certification period's (m and each phase j), public in the Lean answer's
profile; it names no Commit window, whose count `publish` holds.
"""
from __future__ import annotations

import argparse
import decimal
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
#: where the inference-only case's α and Γ came from (`alpha_gamma`), by its private policy.json's ``read``
#: (`served_cases.FROM_DIVISION`), the one field of it read: never a registered policy, which none records
ALPHA_GAMMA_FROM_DIVISION = ("α and Γ read off the run's own division (served_cases.py --from-division), not set by a "
                             "registered policy: the verdict confirms the division against itself, and is no check "
                             "against a policy")
ALPHA_GAMMA_GIVEN = ("α and Γ as the run's ALPHA and GAMMA set them (the case has no policy.json), not by a registered "
                     "policy, which none records: the verdict checks the division against those settings")
ALPHA_GAMMA_UNSTATED = ("not stated: the record wasn't handed the inference-only case's policy.json "
                        "(--inference-only-case-policy); α and Γ are no registered policy's, which none records")
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
             "source": "Claims.Sampling.KSBy at the deployment's wired check, PeriodLeakAccepted's hypothesis"},
    "ε_bind": {"is": "the binding error, Pr[B_ext]: an accepted extraction that disagrees with the committed trace",
               "source": "Claims.Sampling.Bext, a SHA-512 collision in the run (CommitmentScheme.Binds)"},
    "ε_tap": {"is": "Pr[B_tap]: the extracted trace's openings colliding with the network certifier's",
              "source": "NetTiming.Btap, a SHA-512 collision in the run (CommitmentScheme.Binds)"},
}
UNVALUED = ("δ_net is eps plus each execution phase's ε_ks, ε_bind and ε_tap, never eps alone; no check computes those "
            "three, so δ_net has no number here")
COMPOSITE = ("the composite of record's error (``composite``, `WITHOUT_POUW`), Verity.Game's SoundnessFor at its compiled "
             "verifier: it accepts while the claim of one of its subprotocols (four, or three without PoUW's) fails "
             "with probability at most the sum of their errors, each its published δ at the "
             "strategy's SHA-512 calls plus ε_forge(t, Q) plus its loss at its finder's collision probability "
             "(`Verity.Vocab.error`). Each subprotocol's δ is a term of its own, so one draw read at two δs, a tied "
             "audit's at sampled proofs' δ and at the network period's, counts in both")
COMPOSITE_UNVALUED = ("each subprotocol's ε_forge(t, Q) and its loss at its finder's collision probability, and the "
                      "network certifier's m(ε_ks + ε_bind + ε_tap) (`ERRORS`): no check computes them, so the composite's "
                      "error has no number here, and ``stated`` is the sum of the δs the run states, a lower bound on it")
NO_NCI_DELTA = "no published file states it: each composite publishes each case's params by their SHA-512 alone"
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
TIED_READ = ("the tied audits the period's check reads at δ_i (NetworkCheck.Period's chargedOf: each tied to an "
             "execution phase's scan and drawn in part), each once, at the check's own constant δ_i (tiedDelta, "
             "2^-64), fixed before any coin and set by no params, evidence or answer; each adds its δ_i to eps, Σ_i δ_i "
             "by the union bound. A tied audit drawn whole is read at no δ_i, and adds nothing to eps")
POOLED_EPS = ("eps, δ_net's δ part (`EPS`), over each monitoring period at the deployment's window count, a window an "
              "execution phase (`WINDOW_S`): each cell's windows (``phases``: its phases times its links, never a "
              "Commit window's count, which `publish` holds) times the per-window "
              "term (``per_window``: eps / m, the scans' δ_scan and the charged part, Σ_i δ_i / m), as its cell's rule "
              "scales the run's (`RULES`) and its delta_net states; the run's own period (``run``) is proved. Each "
              "window's δ_net adds ε_ks, ε_bind and ε_tap to it (`ERRORS`), which no check computes")
#: the run's published line of the network certifier's verdict (`served_verdicts.pieces`)
LINE = "THE NETWORK CERTIFIER "
#: the run's published line naming each protocol's coins (served_verdicts.sh's first), and the form of its sampled
#: proofs' part that names the live challenger: PP_CHALLENGER gate:USER@HOST at a named peer, whose session V3c's audit
#: holds to PP_CHALLENGER_KEY
COINS_LINE = "COINS "
LIVE_COINS = re.compile(r"the challenger gate:\S+ \(peer [^?)\s][^)]*\)")
COINS_OPEN = "live-coins"
COINS = (f"{COINS_OPEN}: V3c's tied audit drew its sample on the live challenger's coins, which the serving side can't "
         "know before it commits (the run's COINS line names PP_CHALLENGER gate:USER@HOST and its peer). Open, by this "
         "name, wherever a COINS line names a local development challenger session, another challenger or no coins, "
         "or the run published none: "
         "the sampled-proofs verdict and its δ in the composite's error, each tied audit's δ in eps and the slot "
         "entries E_w that k_{skip,j} charges rest on that draw, so every figure here, the figure of record's "
         "included, holds only on those coins and never as a live draw's. Open too, naming the draw "
         "(``open_draws``), wherever the network certifier's chain draws on coins (`CHAIN_DRAWS`), its bound and "
         "figures then resting on that draw")
NO_COINS = "the run published no COINS line naming sampled proofs' coins (served_verdicts.sh's)"
#: the run's published ADVICE-LINE (`served_verdicts.py advice_line`), whose day's declaration of B the record reads
ADVICE_LINE = "ADVICE-LINE: "
DAY_OPEN = "one-program-a-day"
DAY = (f"{DAY_OPEN}: the record doesn't claim the run served one program a day. Open, by this name, on two counts "
       "until each closes: R2's day isn't yet the challenger's signed time but the day the run's launcher passed the "
       "server (serve-openai --advice-day), and the audit doesn't yet read the challenger's signed registration log "
       "but the copy the served node holds. The day's registration and the declared-circuit refusals still run and "
       "refuse as before: the server declares B as its day's one program in the registration log, whose check refuses "
       "a second declaration of a day (duplicate-declaration); the run's ADVICE-LINE holds the log's last entry to "
       "that declaration; and R2 refuses a record whose program isn't its day's declared one (R2-declared) wherever a "
       "verifier passes it the day and the log's declarations, which no verdict here does")
NO_ADVICE_LINE = "the run published no ADVICE-LINE (served_verdicts.sh's, on a serve-openai --advice run)"
POUW_OPEN = "pouw-freed-compute"
POUW = (f"{POUW_OPEN}: the record credits no PoUW freed compute, and no figure of PoUW's row reaches it (`row_verdict`), "
        "so none is in any total here, and whether the run is of record doesn't read PoUW's verdict (`NOT_CLAIMED`). "
        "Not claimed, by this name: OnsiteFreed (Definitions.Pouw.Guarantees.OnsiteFreed) "
        "reads Dither's headlines (Pouw.Dither.FreedSound, WorkSound), which are stated for the no-fold round "
        "(noFoldRound) and describe neither Dither v3.2's proof block nor the PoUW piece the run serves. That piece is "
        "PoUW H1 + fx1 (eps1-sm120-h1fx1, served_verdicts.sh's default POUW=h1), checked over H1's J10-R period "
        "(PouwCheck.Credit.window), a development run under the lane's development key that credits nothing; Pearl+ "
        "(POUW=pearlplus: served_pearlplus.sh, its own seeded load after the window's serve) is a development piece "
        "too, untied, and credits nothing. ``served`` is what each COINS line says the run's PoUW piece was")
#: the verdicts ``of_record.holds`` doesn't read, since the record claims nothing of them (`POUW`), and under a composite
#: of record without PoUW (`WITHOUT_POUW`) remote's too, whose composite composes PoUW's case
NOT_CLAIMED = ("PoUW",)
NOT_CLAIMED_WITHOUT_POUW = (*NOT_CLAIMED, "remote")
#: PoUW's subprotocol (`Core.Remote.components`)
POUW_ID = "verity/pouw/period/v1"
WITHOUT_POUW = (f"the composite of record where the run has one: the network certifier's composite with remote's "
                f"subprotocols less PoUW's ({POUW_ID}), registered under its own id in Core.Composites with its own "
                "soundness instance (Verity.Game's SoundnessFor at its Verity.Registration.Composite), named here by "
                "the id its composite.json states. Whether the run is of record then reads its verdict and its "
                "subprotocols', inference-only's from its case there, and neither PoUW's nor remote's "
                f"(`NOT_CLAIMED_WITHOUT_POUW`); its error sums no PoUW term. Where the run has none, it is {ONSITE}")
#: ``of_record.verified_compute``: the record credits no verified compute under any composite (`POUW`)
NO_VERIFIED_COMPUTE = "none"
#: ``compute`` in the run's ONSITE and ONSITE-BESIDE lines, at the composite of record's id, so that no reader of a line
#: takes the composite's verdict for a claim of compute
NO_COMPUTE_CLAIM = "no verified compute is credited: {}'s verdict is no compute claim"
#: the PoUW clause of the run's COINS line (served_verdicts.sh's H1_COINS), which runs up to the bandwidth audit's
POUW_SAID = re.compile(r"; (PoUW.*?); the bandwidth audit's")
#: where sampled proofs' audit publishes its record (`served_integrity.public_copy`)
AUDIT = Path("v3c-audit") / "window" / "audit.json"
#: V3c's audit as the network certifier's period reads it, as the run publishes it (`records_audit.py period`): its
#: registration, and each root template's circuit file by the SHA-512 a5 held every session of its members to
PERIOD = Path("v3c-audit") / "period.json"
CREDIT = ("sampled proofs credits only the drawn members whose --registered file named every port of their circuit at "
          "the window's committed roots: the audit completed it from the auditor's record before any session "
          "(`a5.complete_drawn`, its record's checks_after_draw.registered_complete), and the member's Lean verdict of "
          "record, which refuses a file that leaves out a port (Flock.Registered.checkComplete), accepted it reporting "
          "exactly its reads (registered_reads). A member accepted with no file or a partial one is no evidence of the "
          "served window, whose inputs it commits afresh, so each other drawn member is listed as not credited, by "
          "name; one the audit couldn't complete (a port with no committed tree) refuses the audit, which rejects "
          "naming it, a visible failure")
NO_COMPLETION = ("the audit's record states no registered-file completion (checks_after_draw.registered_complete), so "
                 "no member's file is known complete and none is credited")
UNCOMPLETED = "the audit couldn't complete its --registered file, and refused it"
UNREPORTED = "its Lean verdict of record didn't accept its completed file reporting exactly that file's reads"
UNLISTED = "the audit's record lists its Lean verdict but no completion of its --registered file"
#: where the window report keeps the records audit's answer, whose profile and public, the draw's n, law and δ and the
#: bound there, sampled proofs' guarantee reads once its verdict, the composite's leg (`sampled_verdict`), accepted
SAMPLED_FROM = f"record/verdicts.json protocols.{ROWS['sampled proofs']}.answer"
SAMPLED = ("sampled proofs' guarantee, what its draw establishes, ε at δ, per window and pooled over each monitoring "
           "period, never the window's alone. A window (``per_window``): its answer's bound at its integrity profile's δ "
           "over the window's n proof blocks, ε = bound / n (Flock.Guarantees.EndToEndHidden: an accepted window holds "
           "more incorrect proof blocks than its bound with probability at most δ + ε_ks + δ_link). A draw of k of n "
           "proof blocks leaves most of them unchecked, so at k = 1 and δ at most 1/n its bound is n − 1: what a draw "
           "establishes is the pooled count (``pooled``, `POOLED`)")
POOLED = ("Flock.Guarantees.IntegrityHorizon: over T sampling blocks (``sampling_blocks``), each a window's audit, "
          "played by one adaptive prover, each a --zk session under "
          "hidden outputs drawing a uniform k-subset of its n proof blocks on its own live coins, q = k / n, every "
          "window accepts while their incorrect proof blocks total at least ln(1/δ)/q with probability at most "
          "δ + Σ_t (ε_ks,t + δ_link,t), whatever T is. So a monitoring period's windows hold at most ⌊ln(1/δ)/q⌋ "
          "incorrect proof blocks in all (``incorrect_at_most``) but with that probability, its named assumptions A3 "
          "for each window's draw bytes, live-verifier, cr/sha-512 and ecr/sha-512. No check computes ε_ks,t or "
          "δ_link,t, so the probability has no number here past δ (``error``)")
#: the pooled cells' rules: the node's period from the run's window, and the datacenter's and the world's from the node's
POOLED_RULES = {
    "window-to-period": ("the run's window repeated: each window of the monitoring period drawn as the run's, k of its "
                         "n proof blocks (its answer's profile), at least one an execution phase (`WINDOW_S`)"),
    "links": ("each link's node cell at δ / links, summed over the links: IntegrityHorizon on each link's windows at "
              "δ / links, joined by the union bound, our composition, which no Lean statement covers"),
}
WINDOW_S = ("a stated parameter: a window an execution phase, T buckets of τ (the certifier's case's params.schedule), "
            "the fewest the period's check takes, as it ties a compute audit to each execution phase's scan. More "
            "windows lower each share and leave the count, which no window count enters")
NO_GATES = ("in AND gates: missing. A proof block's AND gates are its partition's, which is private, so each share is "
            "of the monitoring period's proof blocks")
NOT_POOLED = ("its draw isn't a uniform k-subset of the window's proof blocks, the law IntegrityHorizon pools, so no "
              "pooled figure is stated for it")
MULTI = ("a draw of more than one member of a window, such as one member of each template's stratum (stratified:1, "
         "H1_TIE's default), writes no case of verity/sampled-proofs/v1 (`a5.v2_proof`: v2 verifies one table's "
         "sessions), so no composite has one and each rejects naming it, and Lean's per-member form "
         "refuses it (MULTI_REFUSED) until a guarantee for a draw of several members is in the lock. A run of record "
         "draws one member, subset:1 under --zk (V3C_LAW), and this is that draw's guarantee")
#: the network certifier's chain report as its piece publishes it (`certifier_piece.PUBLIC`), and its draws by step
CHAIN_REPORT = CASE.parent / "chain" / "report" / "report.json"
DRAWS = {"n2": "the period draw", "scan": "the scan's draw"}
CHAIN_DRAWS = (f"the network certifier's chain draws on no live coins. Its period draw (N2's, `LinkWindows.period_draw`) "
               "takes every window (law all: k None, no coins) or K of them on the verifier's own coins (n2_zk.py's); "
               "its scan's draw takes every proof block or K of them on the verifier's or the gate's coins, which the "
               "published report doesn't say. So each draw whose law isn't all is open under "
               f"{COINS_OPEN}, by its name (``open``), and the network certifier's bound and figures are conditional on "
               "the coins")
NO_CHAIN_REPORT = f"the run published no chain report ({CHAIN_REPORT}), whose steps state each draw's law"
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
#: why PeriodTiedCount is open whatever the lock states it as: no figure reads it, and nothing discharges by it
RESTATE_TIED = ("to be restated before any pin of it counts, and never a discharge: as stated, its tie half is false, "
                "having no scope premise, and its count half reads the stripped view, the network certifier's cells, "
                "not the frames view. Restated with a scope premise over the frames view, read again at that "
                "statement (``takes``) and pinned, it leaves ``open``; until then only the tight form rests on it, and "
                "the figure of record is the conservative one, which doesn't read it (`OPEN`)")
#: PeriodLeakRegistered, a property of the period check's acceptance, by its name in `LOCKS`
PERIOD_REGISTERED = "Properties.NetworkPeriodLeak.periodLeakRegistered"
AUDITOR_RECORD, AUDITOR_LISTING = "auditor-record", "auditor-listing"
#: remote's sampled proofs case (`served_cases.SAMPLED`), whose params θ the remote theorem's registered-files condition
#: reads (`registered_files`)
SAMPLED_ID = "verity/sampled-proofs/v1"
V1_UNREAD = (f"sampled proofs' verdict is its leg in the composite of record, the outcome of {SAMPLED_ID}'s case there "
             "(VStar.Recursive.Sampled on OneStageZk's case, which verifies the proofs themselves), never the window "
             "report's integrity row: that is the records audit's one-stage v1 answer (served_integrity.answer), and a "
             "v1 verdict is no verdict of record")
UNESTABLISHED_CREDIT = "sampled proofs' verdict is not established, so it credits no drawn member"
REGISTERED_FILES = "registered-files"
#: the auditor's configuration of the run, each an obligation the record checks on the run's inputs (``checked``), by
#: the name `RESTS_ON`'s takes and `PREMISES` give it: never a named assumption, and never charged
CHECKED = {
    AUDITOR_RECORD: ("PeriodLeakRegistered's hA, hr and hkey, an obligation on the auditor's configuration that the run "
                     "checks: the auditor's registration A parses from its record (Core.Site.Registration.Auditor.ofJson "
                     "of the registration's params' auditor), the period's LMS parameters, key and session are the "
                     "auditor's record of its network certifier (certifier-key/v1), and A holds that key for a network "
                     "certifier its log places (A.key cert.id = some (hexOf key)). It reads only the auditor's inputs, "
                     "its certifier-key record (--certifier-key) and its registration (--site-registration: the "
                     "registration log's params and the log, which the log's check accepts, "
                     "verity.core.protocols.registration), and the parameters the period's check read (its case's "
                     "params, whose SHA-512 over their canonical JSON must be the one the composite of record states "
                     "for the network certifier's case), never the prover's node or its deployment.json. Where an "
                     "input is missing or doesn't match, it doesn't hold, and certifier-key stays open. The "
                     "registration's link figures (each link's beta and latency) are the auditor's stated parameters, "
                     "not measured, and nothing here reads them. That the network certifier splits the node's uplink "
                     "is what the registration asserts; that the node has no other egress is the auditor's placement, "
                     "under trusted-host (`STANDING`), and no file establishes it"),
    AUDITOR_LISTING: ("each circuit file the period lists, an audit's (circuits) or the scan's (scan_circuits), is by "
                      "SHA-512 the file its audit's sessions were verified against: a5's per-template pin, which "
                      "records_audit.py period publishes in V3c's period.json (`PERIOD`) for the audit whose "
                      "registration it is. It compares SHA-512s and doesn't re-hash the files: the period's listing "
                      "and that period.json are both the auditor's own outputs in OUT. It holds where the period lists "
                      "none, and not where the run has no case of the period to say what it lists. No published record "
                      "states the files the scan's sessions were verified against, so it doesn't hold where the period "
                      "lists the scan's. With it, the remote theorem's registered-files condition on its sampled leg "
                      f"(``registered_files``, `registered_files`): each file remote's sampled proofs case ({SAMPLED_ID}, "
                      "--sampled-proofs-case) registers where its params θ name a digest is, by SHA-512, that digest's: "
                      "its first blob, the circuit file, θ's proofs.circuit, and its second, the public file, θ's "
                      "proofs.public. θ is held by SHA-512 over its canonical JSON to the params each of the run's "
                      "composites that reads the case states for it (params_sha512), and the files are hashed here, "
                      "never read off the evidence, whose digests are the prover's. Where θ names the public file, that "
                      "file is also the one a5 pinned at staging, before the draw (--public-pin, a5's public-pin.json, "
                      "`pinned`): by SHA-512 and its body's, a member's of the pin, whose bodies' leaf layer is the one "
                      "the registration the case commits to holds (R4), the case's commitment the one each composite "
                      "that reads the case states, so the registration the challenger's coins committed to. A public "
                      "file changed after the draw isn't the pin's, nor is a pin rewritten to it, unless only its header "
                      "changed and the pin with it: the pin binds the header only as the audit's own output, and v2 "
                      "holds its fields to what was fixed before the coin, publicHeld its scheme, roots, domains, "
                      "population and units.indices (instance i is unit i), and flock-verify its circuit_sha512 to the "
                      "circuit file θ names, its classes to that circuit's class and its shared_rows to the roots "
                      "publicHeld holds. It holds where no composite of the "
                      "run reads the case. Where either doesn't hold the network verdict is not established and there "
                      "is no bound (`LISTING_UNMET`, `FILES_UNMET`), and where the registered-files condition doesn't, "
                      "neither is the verdict of a composite that reads the case, nor sampled proofs', its leg there; "
                      "where θ names no proofs.public, the public file isn't bound, and sampled proofs' verdict is not "
                      "established (`PUBLIC_UNBOUND`)"),
}
#: the guarantees the tight K rests on, by their names in `LOCKS`: PeriodLeakRegistered's count, at the values the
#: registration opens at its registered inputs, is K's; PeriodLeakAccepted's is the same count at the trace's values
#: there; PeriodTiedCount's is to count a phase's tied audits with the scan as one Program, as ℓ_{a,j} does, once it is
#: restated (`RESTATE_TIED`). The lock records a guarantee's named assumptions, never the antecedents its statement
#: takes, so each gives ``statement``, the digest of its statement among the lock's reads (`statement`) at which it was
#: read, None where none was, ``takes``, each hypothesis, antecedent and condition its good event reads that the
#: deployed code must meet, by its Prop's name or, stated inline, its binder's, with the premise (`PREMISES`), standing
#: assumption (`STANDING`), term of what K counts (`COUNTS`) or obligation the run checks (`CHECKED`) that names it, and
#: ``restate`` where its statement is to be restated, why, which keeps it open at any statement. Parameters the record
#: states (δ ≥ 0) aren't listed
RESTS_ON: dict[str, dict[str, Any]] = {
    "NetTiming.SecurityProofs.PeriodLeakAccepted": {
        "statement": "a0a9c23666273efec9d2ddef896331f0d5a3699e707f317fe3de4e7189f29a17",
        "takes": {"Claims.Sampling.KSBy": "sampled-leg-bridge", "Claims.Flock.ListWithin": "sampled-leg-bridge",
                  "Claims.Sampling.CommitmentScheme.Binds": "sampled-leg-bridge",
                  "Verity.Assumptions.TrustedHost": "Verity.Assumptions.TrustedHost",
                  "hPw": "windows-placed", "hβ": "windows-placed", "hcap": "schedules-held",
                  "hfit": "schedules-held", "hs": "schedules-covered", "hdz": "client-decodes",
                  "NetTiming.ScopeAcc": "wires-fit", "NetTiming.TapOpens": "certifiers-open"}},
    PERIOD_REGISTERED: {
        "statement": "808274afd667f0e04b4758786bf29fedb31a024e4a2a92147a7003fa9cc4d59f",
        "takes": {"Claims.Sampling.KSBy": "sampled-leg-bridge", "Claims.Flock.ListWithin": "sampled-leg-bridge",
                  "Claims.Sampling.CommitmentScheme.Binds": "sampled-leg-bridge",
                  "Verity.Assumptions.TrustedHost": "Verity.Assumptions.TrustedHost",
                  "NetTiming.WindowsPlaced": "windows-placed", "NetTiming.SchedulesCovered": "schedules-covered",
                  "NetTiming.SenderDecodes": "client-decodes", "NetTiming.ScopeAcc": "wires-fit",
                  "NetTiming.RegOpensAt": "reg-opens-at", "NetTiming.CertifiersOpen": "certifiers-open",
                  "hA": AUDITOR_RECORD, "hr": AUDITOR_RECORD, "hkey": AUDITOR_RECORD}},
    "NetTiming.SecurityProofs.PeriodTiedCount": {
        "statement": None,
        "restate": RESTATE_TIED,
        "takes": {"NetTiming.SharedRootsParsed": "shared-roots-parsed",
                  "Claims.Sampling.CommitmentScheme.Binds": "sampled-leg-bridge", "NetTiming.RWired": "sampled-leg-bridge",
                  "Verity.Assumptions.TrustedHost": "Verity.Assumptions.TrustedHost",
                  "NetTiming.RowsTiesOrdered": "rows-ties-ordered", "NetTiming.CircuitCounts": "free-count",
                  "NetTiming.WindowsPlaced": "windows-placed", "NetTiming.TiedExposed": "tied-exposed",
                  "NetTiming.SchedulesHeld": "schedules-held", "NetTiming.SchedulesCovered": "schedules-covered",
                  "NetTiming.SenderDecodes": "client-decodes", "NetTiming.PhaseParsed": "sampled-leg-bridge",
                  "NetTiming.RowsStateHeld": "rows-state-held", "NetTiming.RowsProgWired": "rows-prog-wired",
                  "NetTiming.RegOpensAt": "reg-opens-at", "NetTiming.CertifiersOpen": "certifiers-open",
                  "NetTiming.ScopeAcc": "wires-fit"}},
}
#: each premise the tight K rests on, by the name ``open`` gives it: what it says, and the theorems that discharge it at
#: the deployed code by their names in `LOCKS`' guarantees (none named yet leaves it open), or, for one a guarantee of
#: `RESTS_ON` states of the auditor's inputs, that guarantee (``stated_by``) and the run's check of them (``checked``,
#: `CHECKED`)
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
    "free-count": {"is": "CircuitCounts: each tied audit's free at least #(freeInputs \\ reg) at the wired layout. The "
                         "registered gates are read at a template's first unit, so it holds only of templates of one "
                         "unit per VU (units_per_vu 1): its discharge counts only at a statement that takes no premise "
                         "on units_per_vu, which the period check's circuit read gives once it refuses every other",
                   "discharged_by": ("Properties.NetworkCertifier.circuitCounts",)},
    "tied-padding-reads": {"is": "CircuitCounts' tied-audit half: that no unit of a tied audit's circuit file reads "
                                 "into its padding (FilesOf's readsWhy at no reads is none). The period check reads "
                                 "the scan's circuit files whole, so the scan's half follows from its acceptance, but "
                                 "reads each tied audit's file to its header only, so a file whose units read into "
                                 "its padding isn't refused. Open, by this name, and never assumed: it goes once the "
                                 "circuit check counts every padding column of each tied audit's units (its nIn less "
                                 "its port columns, known from the header alone) as free, and is otherwise discharged "
                                 "once a pass over each tied audit's whole file computes its unit reads and a proved "
                                 "Property, never a difftest, gives that they are the parsed file's unitReads, in the "
                                 "lock as CircuitCountsAt's tied-audit premise",
                           "discharged_by": ()},
    "padding-out-of-free": {"is": "that no padding column of a tied audit's units is a free input it charges: the "
                                  "period check refuses a tied audit whose units read into their padding, and the "
                                  "lemma that a column no unit reads adds no free input is in the lock. Open, by this "
                                  "name, until both are: till then a tied audit's free count must hold its padding "
                                  "columns, which no honest unit reads, or `tied-padding-reads` is open, so the tight K "
                                  "is no figure and the figure of record is the conservative one (`OPEN`, `TIGHT`)",
                            "discharged_by": ()},
    "session-tags": {"is": "the read bitmap's bridge to the sampled sessions' tags: the period check counts each tied "
                           "audit's free inputs off the inputs its circuit reads as the reader parses it (circuitOf at "
                           "Tags.circuit, whole_circuit), where the bridge is definitional, while each session proves "
                           "the circuit at Zk.tags (Tags.circuit.withCoins c), and that the two read the same inputs "
                           "is an identity lemma, not a definition. Open, by this name, until the lemma that circuitOf "
                           "at t.withCoins c is circuitOf at t, by definition once circuitOf reads its tags with their "
                           "identity dropped, is in the lock beside whole_circuit",
                     "discharged_by": ()},
    "held-performed": {"is": "the public's held names only the checks the period check performed, each holding of it",
                       "discharged_by": ("Properties.NetworkPeriod.rowsHeld",)},
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
    "certifiers-open": {"is": "CertifiersOpen, PeriodLeakRegistered's antecedent and PeriodLeakAccepted's TapOpens: at "
                              "every window's statement the network certifier's openings register its observations at "
                              "the window's operator. The period check's half: on an accepted rows-form execution phase "
                              "the scan's rows/<d> root and domain are those the network certifier signed for its "
                              "link, d and window; the device's, under trusted-host: its rows at the transfer gates are "
                              "its observations, its ingress values its operator on the earlier ones",
                        "discharged_by": ("Properties.NetworkCertifier.certifiersOpen",)},
    "certifier-key": {"is": "that the network certifier's key, LMS parameters and session every check reads are the "
                            "auditor's record of its own device (certifier-key/v1, `warden_chain.RECORD`), whose key the "
                            "auditor's registration holds for the network certifier it placed (Registration.Auditor.key), "
                            "never the evidence's or the prover's copy. PeriodLeakRegistered states it (``stated_by``) as "
                            "its hA, hr and hkey, which the run checks of the auditor's inputs (``checked``, "
                            f"`{AUDITOR_RECORD}`). Open, by this name, beside trusted-host, until the lock holds that "
                            "guarantee at the statement `RESTS_ON` pins, taking them by that name, and the run's check "
                            "holds (``unchecked``: why it doesn't); while it is open the network verdict is not "
                            "established (`NOT_ESTABLISHED`), and its bound is conditional on it (`UNCHARGED`)",
                      "discharged_by": (), "stated_by": PERIOD_REGISTERED, "checked": AUDITOR_RECORD},
    "windows-placed": {"is": "WindowsPlaced, PeriodLeakAccepted's hPw and hβ: each window's statement has the partition "
                             "the period read, and each of its faulty sets at δ commits at most the bits β the period "
                             "charges it, from the period check's acceptance of the scan; the tied audits' half is "
                             "`tied-exposed`",
                       "discharged_by": ("Properties.NetworkPeriod.scanWindowsPlaced",)},
    "tied-exposed": {"is": "TiedExposed, PeriodTiedCount's hexp: in each phase, each tied audit that can be wrong at its "
                           "partition and δ shows the rest of the phase at most the gates βt the period charges it "
                           "(Phase.exposed); the scan's half is `windows-placed`. Open until it is stated of the period "
                           "check's acceptance and a theorem gives it",
                     "discharged_by": ()},
    "wrong-bounds": {"is": "WrongBounds, which scanWindowsPlaced takes: each possible incorrect set of a scan statement "
                           "at its partition and δ holds at most b proof blocks, b the bound the audit's answer publishes "
                           "(countBound at the code's strata and δ), from the design's miss at a well-formed design whose "
                           "rules are strata draws. Open, by this name, until a theorem stating it at the deployed draw is "
                           "in the lock. Three of its parts are in no lock: RegMatches, that the registration matches the "
                           "design the bound reads, assumed and the sampled leg's bridge's to give (`sampled-leg-bridge`); "
                           "the law identity, that OneStage.check's derived v.law is g.rule.lawOver n, which today meet "
                           "only through R1's Draw.sameLaw, an equality of canonical texts (jeq); and the population "
                           "lemma, jeq a (toJson n) → nat? a = some n, so R2's population check gives the n the audit's "
                           "bound reads",
                     "discharged_by": ("Claims.Sampling.Count.countBound_miss_design",)},
    "schedules-held": {"is": "SchedulesHeld, PeriodLeakAccepted's hcap and hfit: each schedule the network certifier may "
                             "run gives each connection at most its cap of cells, and its record format holds it "
                             "(TapRecord.fits, the rows format's). Open until the theorem that gives it of the network "
                             "certifier's code is named",
                       "discharged_by": ()},
    "schedules-covered": {"is": "SchedulesCovered, PeriodLeakAccepted's hs: each secret K counts runs on a schedule of "
                                "Ss, whose number k_sched counts, so Ss must hold every schedule the network "
                                "certifier's signed records can show, the union over their arrivals. A property of "
                                "the compiled Lean audit the verification runs over those records, which trusted-host "
                                "makes the traffic that crossed, never of the device's code. Open, by this name, "
                                "until k_sched's count bounds that union and the Property that gives it, stated over "
                                "that audit, is in the lock at the statement `DISCHARGES` pins; SenderDecodes, the "
                                "decoder, is what K counts (`COUNTS`)",
                          "discharged_by": ("Properties.NetworkCertifier.schedulesCoveredAt",)},
    "sampled-leg-bridge": {"is": "the sampled leg's bridge, from its guarantee Verity.Vocab.SampledProofsSound at "
                                 "one-stage v2 to OneStageZk's Lean record: the sampled facts the period takes (KSBy at "
                                 "the wired check at the layout's wires, KS where it has none, from KS_of's hypotheses "
                                 "and the record's check of each session's wired reads, WiredReads; "
                                 "ListWithin, the decoder's list at Flock's radius; Binds, the trace commitment's "
                                 "binding; at the per-member record, the only form with wires, SessionKS, the sessions' "
                                 "KS, which SessionKS_of can't give there, its SessionRuns failing at more than one "
                                 "session; SessionOpens; RegOpens: the registered reads' openings; RWired, the sampling "
                                 "relation each extracted trace meets; PhaseParsed, a phase's statements what its bytes "
                                 "parse to, since the sampled verifier accepts none it can't parse, one trace each), "
                                 "proved of the Lean verifier and taken as theorem hypotheses on the layer below, never "
                                 "SampledCodeRuns or another premise about the deployed Python. Open until the bridge's "
                                 "theorem is in the lock",
                           "discharged_by": ()},
}
#: the premises no charge covers: while one is open the bound, the figure of record, its periods and the composite's
#: error are conditional on it by name, never charged for it
UNCHARGED = ("certifier-key",)
CERTIFIER_KEY = "certifier-key"
NOT_ESTABLISHED = "not established"
KEY_OPEN = (f"{CERTIFIER_KEY} is open: the lock doesn't hold PeriodLeakRegistered at the statement `RESTS_ON` pins, "
            f"taking the auditor's record as {AUDITOR_RECORD} (hA, hr and hkey), or the run's check of it doesn't hold "
            "(``checked``), so the Lean answer's accept (``answer``) establishes no network verdict, and the record "
            "cites none with an open condition")
LISTING_UNMET = (f"{AUDITOR_LISTING} doesn't hold (``checked``): a circuit file the period lists isn't, by SHA-512, one "
                 "the run published as the file its audit's sessions were verified against, so the Lean answer's "
                 "accept (``answer``) establishes no network verdict, and the record states no bound")
FILES_UNMET = (f"{AUDITOR_LISTING}'s {REGISTERED_FILES} condition doesn't hold (``checked``): a file remote's sampled "
               "proofs case registers isn't, by SHA-512, the one its params θ name, or its public file the one pinned "
               "at staging before the draw, or the run can't say, so a Lean "
               "accept (``answer``) establishes no verdict of the network certifier, of a composite that reads that "
               "case or of sampled proofs, its leg there, and the record states no bound")
PUBLIC_UNBOUND = ("the public file isn't bound on this form: remote's sampled proofs case's params θ name no "
                  "proofs.public, so the remote theorem's registered-files condition binds the circuit file alone, and "
                  "sampled proofs' accept (``answer``) establishes no verdict. The evidence's digest of the public file, "
                  "the prover's, is never read in its place")
#: why a premise a guarantee states is open where no run's check was read
NOT_CHECKED = "no run's inputs were checked"
#: each theorem of `PREMISES`' ``discharged_by`` that has been read, by its name in `LOCKS`, as `RESTS_ON` gives a
#: guarantee: ``statement``, its statement's digest among the lock's reads, and ``takes``, each antecedent of it that the
#: deployment must meet, by its Prop's name or its binder's, with the premise, standing assumption or term of what K
#: counts that names it. A Property's statement may take a deployment's premise inline, where the lock records no
#: assumption (`ScanScopeAcc` takes `StmtFlattens` and `ScopeMatches` of the deployment it quantifies), so a discharge
#: counts only at the statement pinned here
DISCHARGES: dict[str, dict[str, Any]] = {}
#: the named assumptions that may stand under the figure of record, each on an auditor-trusted device or the auditor's
#: placement of one, named beside the bound
STANDING = {"Verity.Assumptions.TrustedHost": "trusted-host: the network certifier and its host are the auditor's, "
                                              "which covers that the records it signs are the traffic that crossed, "
                                              "at the times it crossed, not what the device's code computes, and, "
                                              "with the auditor's placement of it (certifier-coverage), that the "
                                              "node has no egress but through it",
            "Accounting.Covered": "certifier-coverage: the links in the room are the registered ones, each through the "
                                  "network certifier (a listed physical premise, the auditor's placement, under "
                                  "trusted-host). The site registration asserts that the network certifier splits the "
                                  "node's uplink; that the node has no other egress is this placement, which the "
                                  "registration doesn't establish"}
#: what K counts: the hypotheses on the covert sender's strategy that its guarantee quantifies, which no deployment
#: discharges, so none is open
COUNTS = {"client-decodes": "the secrets a decoder recovers from the client's view, the frames the network certifier's "
                            "run sends across the wire to the client, padding included (NetTiming.laneWire): "
                            "SenderDecodes, PeriodLeakAccepted's hdz"}
DISCHARGE = ("the tight K's premises (`PREMISES`), each open until every theorem that discharges it at the deployed code "
             "is in Security's lock at the statement `DISCHARGES` pins (``unpinned``, with the lock's digest, where it "
             "isn't), with no named assumption but a standing one (`STANDING`) or one its pin takes, and every premise "
             "its pins take is discharged (``takes``), so no premises discharge each other; and open while none is "
             "named. A premise a guarantee of `RESTS_ON` states of the auditor's inputs (``stated_by``) closes with "
             "no discharge, where that guarantee isn't open and takes it by the name of the run's check (``checked``, "
             "`CHECKED`) and that check holds (``unchecked`` where it doesn't). Open too: "
             "each guarantee of `RESTS_ON` the lock doesn't list, or lists at a statement other than the one "
             "whose hypotheses and antecedents it names (``takes``), or whose statement is to be restated "
             "(``restate``) at any statement; and each named assumption the lock gives such a "
             "guarantee that is neither standing nor named there, by its own name. What K counts (`COUNTS`) is no "
             "premise. No premise about our own Python stands: soundness is in Lean or rests on an auditor-trusted "
             "device")
#: why a guarantee of `RESTS_ON` is open at a statement other than the one it names
RESTATED = ("its statement in the lock isn't the one whose hypotheses and antecedents `RESTS_ON` names: read them again, "
            "name each in its takes, and pin the lock's digest")
OPEN = ("the figure of record: K as the period's check states it, the tight form, only where its Lean verdict performed "
        "every check of `CHECKS` (the public's held), no premise of it is open (`DISCHARGE`) and the run's COINS line "
        "names the live challenger (`COINS`). While any is open it is K plus ℓ_a + 1 bits for each audit tied to each "
        "execution phase's scan, beside that phase's ℓ_{a,j}: an advice of at most ℓ_a bits (`CAP`) is one of fewer "
        "than 2^(ℓ_a + 1) strings. That conservative figure, ℓ_a + 1 added to ℓ_{a,j}, is the figure, and no tighter "
        "figure goes beside it (`TIGHT`). While the coins or a premise of `UNCHARGED` are open, the figure of record and its cells are "
        "conditional on them (conditional_on), since no charge here covers a draw on known coins or a key the auditor "
        "didn't record")
TIGHT = ("how the figure tightens, each in place of the figure of record and never beside it: to ℓ_a + 1 in place of "
         "each tied audit's ℓ_{a,j}, once a general cut lemma bounding a tied audit's leak by ℓ_a + 1 is in the lock "
         "and a check verifying that cut has run on the period's case, so the run's record, written again from its "
         "OUT, moves to it without re-serving; and to K alone, once padding-out-of-free is discharged in the lock (the "
         "period check refusing a tied audit's padding reads, and the unread-column lemma) and nothing else is open. "
         "Until then the record states no figure of either")
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
#: window, <class> V3c's class and <tie> -tie where H1_TIE ties the window, whose record V3c holds the commit to reads
#: H1's tables (served_verdicts.sh's `v3c_record`), else nothing; or OUT, the run's public files) and what it holds
REPROOF = {
    "values": ("serve dir", "v3c-w<K>-<class>/window/",
               "the window commit: every unit's committed values with their rows and salts, the sampled units' and those "
               "of the units that write what they read, and its circuit.json and registration.json"),
    "openings": ("serve dir", "v3c-w<K>-<class>/audit/window/",
                 "each sampled unit's opening as it was proved (m<k>-private.bin, m<k>-public.bin, v<j>-private.bin), "
                 "with the partition and the Program the audit sampled"),
    "draw": ("serve dir", "v3c-w<K>-<class>/audit/window/{draw,segment,receipt}.json",
             "the draw and its receipt, the stream's record that it received the registration before it read a coin"),
    "coins": ("serve dir", "v3c-w<K>-<class>/audit/challenger/",
              "the challenger session the draw read its coins from: a local development one unless the run's COINS "
              "line names the live challenger (`coins`)"),
    "release": ("serve dir", "v3c-w<K>-<class>/audit/window/release.json",
                "the challenger's release of the draw's coins (`a5.drawn_by`: its key, the window root and the coins, "
                "settled), whose coins and window root a later v2 case of the drawn units reads under the auditor's own "
                "key record, so it reads the same draw"),
    "advice": ("serve dir", "advice/w<K>/",
               "the advice's registration and its opening (private/advice.bin), which the window circuit reads under its "
               "hiding root"),
    "weights": ("serve dir", "weights_registration.json, private/weights.key, private/cos_sin_cache.npy",
                "the weights registration the audit pins, the committer's key their salts come from and the rotary table, "
                "beside the checkpoint (REPO at REV, public)"),
    "run root": ("serve dir", "windows.jsonl", "the run root the audit holds the registration to"),
    "circuit": ("serve dir", "window-circuit-w<K>-<class><tie>.json",
                "the verifier's window-circuit record at the class (under H1_TIE, the one reading H1's tables), which "
                "the window commit's circuit is held to"),
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
}
REPROOF_KEPT = ("each retained past the week after the run, never only on the served node: OUT's in custody with the "
                "run's public files; the serve dir's, witness, never in OUT or custody, which are public, but copied off "
                "the served node to a private path keeping the serve dir's layout (`copy`: its node and path, and of "
                "each serve-dir input whether the copy's listing holds each file its path names and at least one file under "
                "each directory it names, never their bytes or digests); a re-proof holds each to the public registration and the audit's summary")
NO_COPY = ("no copy of the serve dir's inputs is given (--reproof-copy NODE:/PATH and --reproof-listing): they are on "
           "the served node only")
SERVING = ("the run's serving settings as its caller states them (--window-steps, --max-tokens): S, served_verdicts.sh's "
           "WINDOW_STEPS, the most engine steps a Commit period holds, and its MAX_TOKENS. No file of OUT states both "
           "(its COINS line names S only where PoUW H1 is served); ℓ_a (advice_charge.cap) is the cap the run's S, "
           "slots, model length and id bytes set, and every figure here is the run's at these settings alone")
NO_SERVING = ("the run's caller stated no serving settings (--window-steps and --max-tokens: served_verdicts.sh's "
              "WINDOW_STEPS and MAX_TOKENS)")
BESIDE = "verity/onsite-network-records-beside/v1"
BESIDE_IS = ("each run's figures from its own record, beside the others', each labelled with its serving settings "
             "(`SERVING`), ℓ_a, its composite of record by id (`WITHOUT_POUW`) and the verified compute it credits, "
             "none (`NO_COMPUTE_CLAIM`), what its figures are conditional on "
             "(`COINS`), what it doesn't claim (`DAY`, `POUW`) and the drawn members it doesn't credit (`CREDIT`): its "
             "figure of record, what is open under it, "
             "each period's cells (bits and share of the weights), and sampled proofs' guarantee per window and "
             "pooled (`SAMPLED`). No figure is carried from one run's settings to another's")
UNLABELLED = ("states no serving settings, which label its figures beside another run's: write it again from its OUT "
              "with --window-steps and --max-tokens")


def missing(why: str, src: str | None = None) -> dict[str, Any]:
    return {"verdict": "missing", "why": why, **({"from": src} if src else {})}


def network_row(out: Path, rows: Mapping[str, Any], run: str | None) -> tuple[Any, str]:
    """The network certifier's row and where it is from: the window report's, or, where the report's product dropped it
    and its period check answered under OUT, the row that answer gives (`served_verdicts.piece_row`): the one the report
    composes under the onsite product, by the report's own composer, never a second one."""
    key, answer = ROWS["the network certifier"], CASE.parent / "answer.json"
    r = rows.get(key)
    if not (isinstance(r, Mapping) and r.get("verdict") == "dropped" and (out / answer).is_file()):
        return r, f"record/verdicts.json protocols.{key}"
    import served_verdicts as SV

    return SV.piece_row(key, out / "pieces", run), str(answer)


def row_verdict(rows: Mapping[str, Any], key: str, src: str | None = None) -> dict[str, Any]:
    """A verdict from the window report's row ``key`` (from ``src``, default the report's): a missing, dropped or
    development row is missing, with its why, and a development row says so (``development``)."""
    src = src or f"record/verdicts.json protocols.{key}"
    r = rows.get(key)
    if not isinstance(r, Mapping):
        return missing(f"the window report has no {key} row", src)
    if r.get("verdict") in ("missing", "dropped"):
        return missing(str(r.get("why")), src) | ({"development": True} if "development" in r else {})
    out = {"protocol": r.get("protocol"), "verdict": r.get("verdict"), "from": src}
    return out | ({"reason": r["reason"]} if r.get("reason") else {})


def composites(out: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each composite.json under OUT/composite, by the product it names: its path in OUT and its document."""
    found: dict[str, tuple[str, dict[str, Any]]] = {}
    for f in sorted((out / "composite").glob("*/composite.json")):
        doc = json.loads(f.read_text())
        found[doc.get("product")] = (str(f.relative_to(out)), doc)
    return found


def subprotocols(doc: Mapping[str, Any]) -> set[str]:
    """A composite.json's subprotocols by id: its cases' and those it names missing."""
    return {c.get("id") for c in doc.get("cases") or ()} | set(doc.get("missing") or ())


def of_record_composite(found: Mapping[str, tuple[str, dict[str, Any]]]) -> tuple[str, bool]:
    """The composite of record (`WITHOUT_POUW`) by its id, and whether it composes PoUW: the run's composite with the
    network certifier's subprotocol and not PoUW's, else `ONSITE`. ValueError, naming them, where the run has more than
    one such."""
    without = sorted(str(p) for p, (_, doc) in found.items()
                     if NETWORK in (ids := subprotocols(doc)) and POUW_ID not in ids)
    if len(without) > 1:
        raise ValueError(f"the run has {len(without)} composites of the network certifier without PoUW "
                         f"({', '.join(without)}): the record reads one composite of record")
    return (without[0], False) if without else (ONSITE, True)


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


def sampled_verdict(found: Mapping[str, tuple[str, dict[str, Any]]], composite: str,
                    rows: Mapping[str, Any]) -> dict[str, Any]:
    """Sampled proofs' verdict (`V1_UNREAD`): the outcome of `SAMPLED_ID`'s case in the composite of record
    ``composite``; where it has none, not established, the window report's row's answer beside it (``answer``), or
    missing where that row is missing too."""
    v = case_verdict(found, composite, SAMPLED_ID)
    v1 = row_verdict(rows, ROWS["sampled proofs"])
    if v["verdict"] != "missing" or v1["verdict"] == "missing":
        return v
    return {"protocol": SAMPLED_ID, "verdict": NOT_ESTABLISHED, "answer": v1["verdict"], "from": v1["from"],
            "why": f"{V1_UNREAD}, and {v['why']}"}


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


def composite_error(rows: Mapping[str, Any], eps: Fraction | None, why: str = "no accepted bound",
                    conditional: Sequence[str] = (), composite: str = ONSITE, pouw: bool = True) -> dict[str, Any]:
    """The error of ``composite``, the composite of record, term by term (`COMPOSITE`): each subprotocol's δ with where
    the run states it, or why it doesn't (the network certifier's eps from its bound, else ``why``), PoUW's only where
    the composite has it (``pouw``); ``stated`` sums those stated (None where none is), ``vacuous`` at 1 or more;
    ``conditional_on`` the conditions sampled proofs' δ and eps rest on where any is open (`COINS`)."""
    terms: list[dict[str, Any]] = []
    for name in ("sampled proofs", "PoUW")[:1 + pouw]:
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
    return {"is": COMPOSITE, "composite": composite, "terms": terms, "stated": None if stated is None else str(stated),
            "vacuous": stated is not None and stated >= 1, "unvalued": COMPOSITE_UNVALUED} | (
        {"conditional_on": list(conditional)} if conditional else {})


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
    return phase_of(s), n, cells * 8 * int(s["cell_bytes"])


def phase_of(s: Mapping[str, Any]) -> Fraction:
    """An execution phase's seconds under the case's ``schedule`` ``s``: its T buckets of τ."""
    return Fraction(int(s["window"]) * int(s["bucket_ms"]), 1000)


def phase_seconds(out: Path) -> tuple[Fraction | None, str]:
    """An execution phase's seconds from OUT's certifier case (`phase_of`), or None and why it can't be read."""
    try:
        return phase_of(json.loads((out / CASE).read_text())["params"]["schedule"]), ""
    except FileNotFoundError:
        return None, f"no {CASE}: a window's length is an execution phase's, its params' schedule (`WINDOW_S`)"
    except (KeyError, TypeError, ValueError) as e:
        return None, f"{CASE}: {type(e).__name__}: {e}"


def period_case(out: Path) -> tuple[Mapping[str, Any] | None, str]:
    """OUT's certifier case, whose params are the parameters the period's check read, or None and why the run has
    none."""
    try:
        case = json.loads((out / CASE).read_text())
    except FileNotFoundError:
        return None, f"the run has no {CASE}"
    except ValueError as e:
        return None, f"{CASE} can't be read: {type(e).__name__}"
    return (case, "") if isinstance(case, Mapping) else (None, f"{CASE} isn't an object")


def horizon(q: Fraction, delta: Fraction, links: int = 1) -> int:
    """⌊ln(links/δ)/q⌋, IntegrityHorizon's count at δ / links (`POOLED`), never below it: ln to 60 digits, raised by a
    part in 10^40 before the floor."""
    with decimal.localcontext() as c:
        c.prec = 60
        D = decimal.Decimal
        x = ((D(links) * D(delta.denominator)).ln() - D(delta.numerator).ln()) * D(q.denominator) / D(q.numerator)
        return math.floor(x * (1 + D(10) ** -40))


def sampled(row: Any, verdict: Mapping[str, Any], phase_s: Fraction | None, why: str,
            conditional: Sequence[str] = ()) -> dict[str, Any]:
    """Sampled proofs' guarantee (`SAMPLED`) from its accepted row's answer: its window's, and pooled over each
    monitoring period (`POOLED`) for the serving node and, by the links of each (`census`), a datacenter and the world,
    at a window an execution phase of ``phase_s`` seconds (`WINDOW_S`; where it is None the periods are missing,
    ``why``); a draw of another law has no pooled figure (`NOT_POOLED`). ``refused`` names the draws a run of record
    doesn't make (`MULTI`), and ``conditional_on`` the coins' condition while it is open (`COINS`)."""
    g = {"is": SAMPLED, "refused": MULTI, "from": SAMPLED_FROM} | (
        {"conditional_on": list(conditional)} if conditional else {})
    if verdict.get("verdict") != "accept":
        return g | {"missing": f"sampled proofs' verdict is {verdict.get('verdict')}: only its accepted answer "
                                 "states a bound"}
    try:
        a = row["answer"]
        levels, b = a["profile"]["levels"], int(a["public"]["bound"])
        n, law, delta = int(levels[-1]["size"]), levels[-1]["draw"], exact_delta(a["profile"]["delta"])
    except (IndexError, KeyError, TypeError, ValueError) as e:
        return g | {"error": f"its answer's profile or public lacks a field: {type(e).__name__}: {e}"}
    if not (0 < n and 0 <= b <= n and 0 < delta < 1):
        return g | {"error": f"its answer's bound {b} over {n} proof blocks at δ {delta} states no ε"}
    g["per_window"] = {"units": n, "law": law, "delta": str(delta), "bound": b, "epsilon": str(Fraction(b, n)),
                     "statement": "Flock.Guarantees.EndToEndHidden"}
    k = law.get("subset") if isinstance(law, Mapping) and len(levels) == 1 else None
    if not (isinstance(k, int) and 0 < k <= n):
        return g | {"pooled": {"missing": NOT_POOLED}}
    q = Fraction(k, n)
    pooled: dict[str, Any] = {
        "statement": "Flock.Guarantees.IntegrityHorizon", "is": POOLED, "q": str(q), "incorrect_at_most": horizon(q, delta),
        "error": {"delta": str(delta), "unvalued": "Σ_t (ε_ks,t + δ_link,t) over the cell's sampling blocks"},
        "gates": {"missing": NO_GATES}}
    if phase_s is None:
        return g | {"pooled": pooled | {"periods": {"missing": why}}}
    refs = census()
    links = {"node": 1, **{where: v["links"] for where, v in refs.items()}}
    pooled |= {"window_s": {"seconds": str(phase_s), "is": WINDOW_S}, "rules": {
        "window-to-period": {"rule": POOLED_RULES["window-to-period"], "sources": [SAMPLED_FROM, FROM_SCHEDULE]},
        **{where: {"rule": POOLED_RULES["links"], "sources": v["sources"]} for where, v in refs.items()}}}
    for name, seconds in LB.PERIODS.items():
        w = math.ceil(Fraction(seconds) / phase_s)
        pooled[name] = {"seconds": seconds, "sampling_blocks": w}
        for where, L in links.items():
            blocks = w * L * n
            f = min(L * horizon(q, delta, L), blocks)
            pooled[name][where] = {"links": L, "sampling_blocks": w * L, "proof_blocks": blocks, "incorrect_at_most": f,
                                   "share": float(Fraction(f, blocks)), "how": "scaled",
                                   "rule": "window-to-period" if where == "node" else where}
    return g | {"pooled": pooled}


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


def pooled_eps(delta: Mapping[str, Any], m: int, p: Mapping[str, Any]) -> dict[str, Any]:
    """eps over the run's period and each monitoring period's cells of ``p`` (`periods`) at their windows, split as the
    bound's ``delta`` splits it into the scans' δ and the charged part, beside the per-window term, with the tied audits
    the period reads (`POOLED_EPS`, `TIED_READ`); the count missing where the public lists no reading at δ_i."""
    readings = delta["charged_audits"]
    read = None if isinstance(readings, Mapping) else len(readings)
    scan, charged = Fraction(delta["scan"]), Fraction(delta["charged"]) / m

    def at(phases: int) -> dict[str, Any]:
        return {"phases": phases, "eps": str(phases * (scan + charged)), "scan": str(phases * scan),
                "charged": str(phases * charged)} | (
            {} if read is None else {"tied_audits_read": str(Fraction(read * phases, m))})

    tied = {"missing": NO_READINGS} if read is None else {
        "read": read, "delta_i": sorted({r["delta"] for r in readings}), "is": TIED_READ}
    return {"is": POOLED_EPS, "tied_audits": tied, "per_window": at(1), "run": at(m),
            **{name: {where: at(p[name][where]["parameters"]["phases"] * p[name][where]["parameters"]["links"])
                      for where in ("node", "datacenter", "world")} for name in LB.PERIODS}}


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


def coins(out: Path) -> dict[str, Any]:
    """Sampled proofs' coins as the run's COINS lines name them (`COINS`): each line's sampled-proofs part, None where
    it names none, and ``open`` unless every part names the live challenger (`LIVE_COINS`); ``missing`` where the run
    published no COINS line."""
    said = [m[1].strip() if (m := re.search(r"sampled proofs: ([^;]*)", x)) else None
            for x in lines(out, "verdicts.txt") if x.startswith(COINS_LINE)]
    live = bool(said) and all(s is not None and LIVE_COINS.fullmatch(s) for s in said)
    return ({"open": not live, "sampled_proofs": said, "from": f"verdicts.txt {COINS_LINE.strip()}", "is": COINS}
            | ({} if said else {"missing": NO_COINS}))


def day(out: Path) -> dict[str, Any]:
    """One program a day, open (`DAY`), beside the day each of the run's ADVICE-LINEs says it declared B for (None where
    it names none); ``missing`` where the run published no ADVICE-LINE."""
    said = [m[1] if (m := re.search(r"declared day (\S+)'s one program", x)) else None
            for x in lines(out, "verdicts.txt") if x.startswith(ADVICE_LINE)]
    return ({"open": True, "advice_line_day": said, "from": f"verdicts.txt {ADVICE_LINE.rstrip(': ')}", "is": DAY}
            | ({} if said else {"missing": NO_ADVICE_LINE}))


def pouw(out: Path) -> dict[str, Any]:
    """PoUW's freed compute, not claimed (`POUW`): credited none, beside each COINS line's PoUW clause (None where it
    names none)."""
    said = [m[1] if (m := POUW_SAID.search(x)) else None for x in lines(out, "verdicts.txt") if x.startswith(COINS_LINE)]
    return {"open": True, "credited": None, "served": said, "from": f"verdicts.txt {COINS_LINE.strip()}", "is": POUW}


def credit(out: Path) -> dict[str, Any]:
    """Sampled proofs' drawn members as its audit's published record credits them (`CREDIT`): ``credited``, each by
    name, and ``not_credited``, each other drawn member with why; ``missing`` where the record states no completion or
    the run published none."""
    try:
        rec = json.loads((out / AUDIT).read_text())
    except FileNotFoundError:
        return {"credited": [], "missing": f"the run published no audit record ({AUDIT})", "is": CREDIT}
    done = (rec.get("checks_after_draw") or {}).get("registered_complete") if isinstance(rec, Mapping) else None
    if not isinstance(done, Mapping) or not isinstance(done.get("members"), list):
        return {"credited": [], "missing": NO_COMPLETION, "from": str(AUDIT), "is": CREDIT}
    chunks = (rec.get("registered_reads") or {}).get("chunks") or {}
    credited = sorted(k for k in done["members"] if (chunks.get(k) or {}).get("ok") is True)
    not_credited = ({k: UNREPORTED for k in done["members"] if k not in credited}
                    | {k: UNLISTED for k in chunks if k not in done["members"]}
                    | {k: UNCOMPLETED for k in done.get("codes") or ()})
    return {"credited": credited, "not_credited": dict(sorted(not_credited.items())), "from": str(AUDIT), "is": CREDIT}


def credited(c: Mapping[str, Any]) -> bool:
    """Whether `credit` credits every drawn member, at least one."""
    return bool(c.get("credited")) and not c.get("not_credited") and "missing" not in c


def chain_draws(out: Path) -> dict[str, Any]:
    """The network certifier's chain draws as its published report states each one's law (`CHAIN_DRAWS`), by name, and
    ``open``, each whose law isn't all; ``missing`` where the run published no report, every draw then open."""
    try:
        steps = json.loads((out / CHAIN_REPORT).read_text()).get("steps") or {}
    except FileNotFoundError:
        return {"missing": NO_CHAIN_REPORT, "open": list(DRAWS.values()), "is": CHAIN_DRAWS}
    laws = {name: ((steps.get(s) or {}).get("eps_at_delta") or {}).get("law") for s, name in DRAWS.items()}
    return {"laws": laws, "open": [n for n, law in laws.items() if law != "all"], "from": str(CHAIN_REPORT),
            "is": CHAIN_DRAWS}


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


def statement(lock: Mapping[str, Any], g: str) -> str | None:
    """The digest ``lock`` gives the statement of its guarantee ``g``: its signature's type, a definition among the
    lock's reads. None where the lock lists no such definition."""
    stated = " ".join((lock.get("guarantees", {}).get(g) or {}).get("signature") or ()).partition(" : ")[2].strip()
    return next((r["definitions"][stated] for r in lock.get("reads", {}).values()
                 if stated in (r.get("definitions") or {})), None)


def premises(locks: Sequence[Path], root: Path = ROOT,
             checked: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The tight K's premises as ``locks`` record their discharges (`DISCHARGE`): each of `PREMISES` with the theorems
    no lock lists (``missing``), those one lists at a statement `DISCHARGES` doesn't pin (``unpinned``, each with the
    lock's digest), the named assumptions of those one lists that don't stand and their pins don't take (``assumes``)
    and the premises their pins take (``takes``), and for one a guarantee states, that guarantee (``stated_by``), the
    run's check it reads (``checked``, by its name in ``checked``, the run's `CHECKED` checks) and why that check
    doesn't hold (``unchecked``, `NOT_CHECKED` where ``checked`` has none); each guarantee of `RESTS_ON` with its named
    assumptions, its statement's digest and what it takes (``restated`` where that digest isn't the one `RESTS_ON`
    read, `RESTATED`; ``restate`` where its statement is to be restated, why), what K counts (``counts``, `COUNTS`),
    and ``open``: each premise not closed, then each named assumption of a guarantee of `RESTS_ON` neither standing nor
    taken by name, and each such guarantee no lock lists, one lists restated, or one to be restated. A lock that can't
    be read lists nothing, and ``unread`` names it, by its path under ``root`` (the tree) where it is one, with why."""
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
    rests: dict[str, Any] = {}
    for g, r in RESTS_ON.items():
        if g not in listed:
            rests[g] = {"missing": "not in the lock", "takes": r["takes"]}
        else:
            rests[g] = {"assumptions": list(listed[g]["assumptions"]), "statement": stated.get(g), "takes": r["takes"]}
            if r["statement"] is None or stated.get(g) != r["statement"]:
                rests[g]["restated"] = RESTATED
        if "restate" in r:
            rests[g]["restate"] = r["restate"]
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
        if g := p.get("stated_by"):
            run = (checked or {}).get(p["checked"]) or {"holds": False, "why": NOT_CHECKED}
            found[name] |= {"stated_by": g, "checked": p["checked"]} | (
                {} if run["holds"] else {"unchecked": run["why"]})
            if run["holds"] and not {"missing", "restated", "restate"} & set(rests[g]) \
                    and p["checked"] in RESTS_ON[g]["takes"].values():
                ready[name] = set()
        elif not (missing or unpinned or assumes or not p["discharged_by"]):
            ready[name] = set(takes)
    done: set[str] = set()
    while new := {n for n, t in ready.items() if n not in done and t <= done}:
        done |= new
    open_ = [n for n in PREMISES if n not in done]
    open_ += sorted({a for g, x in rests.items() for a in x.get("assumptions", ())
                     if a not in RESTS_ON[g]["takes"] and a not in STANDING}
                    | {g for g, x in rests.items() if {"missing", "restated", "restate"} & set(x)})
    return {"is": DISCHARGE, "from": "Security/lean-audit.json and Security/Properties/lean-audit.json guarantees "
                                     "and reads",
            "premises": found, "rests_on": rests, "standing": STANDING, "counts": COUNTS, "open": open_,
            "unread": unread}


def auditor_record(params: Any, composed: str | None, key: Path | None, site: Path | None) -> dict[str, Any]:
    """`CHECKED`'s auditor-record on the run: the period's ``params``, its case's, held to those its check read by their
    SHA-512, ``composed``, the composite of record's for its network certifier's case (`composite.canonical`, None
    where it has none), against the auditor's record of its network certifier at ``key`` (certifier-key/v1,
    `warden_chain.RECORD`) and its registration at ``site``, ``{"params", "log"}``: the registration log's params,
    ``{"auditor", "composite"}``, and the log, which its check accepts (`registration.check_log`, held to
    `Core.Registration` by difftest), whose auditor holds the record's key for a network certifier the log places.
    ``holds``, that ``certifier`` where it does and ``why`` where it doesn't, and ``from``, each input's SHA-512."""
    import composite as CO
    import warden_chain as WC
    from verity.core.protocols import registration as REG

    out: dict[str, Any] = {"is": CHECKED[AUDITOR_RECORD], "holds": False, "from": {}}
    docs = {}
    for flag, f in (("--certifier-key", key), ("--site-registration", site)):
        if f is None:
            return out | {"why": f"the run gave no {flag}, the auditor's input it reads"}
        try:
            data = f.read_bytes()
            out["from"][flag] = hashlib.sha512(data).hexdigest()
            docs[flag] = json.loads(data)
        except (OSError, ValueError) as e:
            return out | {"why": f"{flag} can't be read: {type(e).__name__}"}
    r = docs["--certifier-key"]
    try:
        (n, w, h), public, session = r["key"]["params"], bytes.fromhex(r["key"]["public"]), bytes.fromhex(r["session"])
        if r["schema"] != WC.RECORD:
            raise ValueError(r["schema"])
    except (KeyError, TypeError, ValueError):
        return out | {"why": f"--certifier-key isn't a {WC.RECORD} record of a key and a session"}
    try:
        p = {"lms": params["lms"], "key": bytes.fromhex(params["key"]), "session": bytes.fromhex(params["session"])}
    except (KeyError, TypeError, ValueError):
        return out | {"why": f"the period's parameters ({CASE}'s params) state no LMS parameters, key and session"}
    if composed is None:
        return out | {"why": f"the composite of record has no {NETWORK} case stating the SHA-512 of the parameters the "
                             "period's check read"}
    if hashlib.sha512(CO.canonical(params)).hexdigest() != composed:
        return out | {"why": f"the period's parameters ({CASE}'s params) aren't, by SHA-512, those its check read (the "
                             f"composite of record's {NETWORK} case's params_sha512)"}
    want = {"lms": {"H": h, "N": n, "W": w, "toy": False}, "key": public, "session": session}
    if differ := [k for k in want if p[k] != want[k]]:
        return out | {"why": f"the period's {' and '.join(differ)} {'isn' if len(differ) == 1 else 'aren'}'t the "
                             "auditor's record's"}
    try:
        reg = docs["--site-registration"]
        if not isinstance(reg, dict) or set(reg) != {"log", "params"}:
            raise REG.Malformed("not {params, log}")
        composite, auditor = REG.params_of(reg["params"])
        verdict = REG.check_log((composite, auditor), REG.log_of(reg["log"]))
    except (KeyError, TypeError, ValueError) as e:
        return out | {"why": f"--site-registration isn't a registration's params and log: {type(e).__name__}: {e}"}
    if verdict[0] != "accept":
        return out | {"why": f"the auditor's registration log doesn't check: {verdict[1]}"}
    cert = next((x["id"] for x in verdict[1]["site"]["locs"]
                 if x["kind"] == "certifier" and auditor.key(x["id"]) == public.hex()), None)
    if cert is None:
        return out | {"why": "the auditor's registration holds the record's key for no network certifier its log places"}
    return out | {"holds": True, "certifier": cert}


def auditor_listing(out: Path, case: Any) -> dict[str, Any]:
    """`CHECKED`'s auditor-listing on the run: each audit's circuit files its ``case``'s params list (``circuits``),
    against the run's published V3c period (`PERIOD`) where the case's commitment holds that period's registration at
    the audit's place, and the scan's (``scan_circuits``), of which the run publishes no record. ``holds``, ``listed``,
    how many files the period lists, and ``why`` where it doesn't hold."""
    res: dict[str, Any] = {"is": CHECKED[AUDITOR_LISTING], "holds": False}
    try:
        audits = case["params"].get("audits") or []
        listed = [(i, a["circuits"]) for i, a in enumerate(audits) if "circuits" in a]
        scan = case["params"].get("scan_circuits")
    except (KeyError, TypeError, AttributeError):
        return res | {"why": f"the run has no {CASE} whose params list the period's audits"}
    if scan is not None:
        return res | {"why": "the period lists the scan's circuit files (scan_circuits), and the run publishes no record "
                             "of the files the scan's sessions were verified against"}
    res["listed"] = sum(len(files) if isinstance(files, list) else 1 for _, files in listed)
    if not listed:
        return res | {"holds": True}
    try:
        period = json.loads((out / PERIOD).read_text())
        regs = json.loads(bytes.fromhex(case["commitment"]))["audits"]
    except (OSError, KeyError, TypeError, ValueError) as e:
        return res | {"why": f"the period lists circuit files, and {PERIOD} or its case's commitment can't be read: "
                             f"{type(e).__name__}"}
    for i, files in listed:
        if i >= len(regs) or regs[i] != period.get("registration"):
            return res | {"why": f"audit {i} lists circuit files, and the run's {PERIOD} is another audit's"}
        if files != period.get("circuits"):
            return res | {"why": f"audit {i}'s listed circuit files aren't, by SHA-512, those its audit's sessions were "
                                 f"verified against ({PERIOD}'s circuits)"}
    return res | {"holds": True}


def pinned(pin: Path | None, commitment: Any, stated: set[Any], public: Path) -> str | None:
    """Why the public file remote's sampled proofs case registers, ``public``, isn't the one a5 pinned at staging,
    before the draw (``pin``, a5's ``public-pin.json``: by member key, each public file's SHA-512 and its body's, and
    their leaf layer), else None. The pin's bodies are, by their leaf layer, those the registration the case commits to
    holds (``commitment``, which each composite that reads the case states, ``stated``): the registration the
    challenger's coins committed to, which R4 held to the bodies before the draw. And ``public`` is, by SHA-512 and its
    body's, a pinned member's."""
    from verity.ml.circuits.codec import canonical_json

    if pin is None:
        return "the run gave no --public-pin, the public file's SHA-512 pinned at staging before the draw"
    try:
        doc = json.loads(pin.read_bytes())
        members = {k: (m["file"], m["body"]) for k, m in doc["members"].items()}
        layer = doc["leaf_layer"]
    except (OSError, AttributeError, KeyError, TypeError, ValueError) as e:
        return f"--public-pin isn't a pin of the public files at staging: {type(e).__name__}"
    if stated != {commitment}:
        return "the case's commitment isn't the one each composite that reads the case states"
    try:
        registered = json.loads(bytes.fromhex(commitment))["leaf_layer"]
    except (KeyError, TypeError, ValueError) as e:
        return f"the case's commitment isn't a registration record naming a leaf layer: {type(e).__name__}"
    bodies = hashlib.sha512(canonical_json({k: b for k, (_, b) in members.items()})).hexdigest()
    if not (isinstance(registered, str) and registered == layer == bodies):
        return ("--public-pin's bodies aren't, by their leaf layer, those the registration the case commits to holds "
                "(R4)")
    whole, body = hashlib.sha512(), hashlib.sha512()
    with open(public, "rb") as f:
        whole.update(f.readline())
        for chunk in iter(lambda: f.read(1 << 20), b""):
            whole.update(chunk)
            body.update(chunk)
    if (whole.hexdigest(), body.hexdigest()) not in members.values():
        return "the public file the case registers isn't, by SHA-512, one --public-pin pinned at staging before the draw"
    return None


def registered_files(found: Mapping[str, tuple[str, dict[str, Any]]], case: Path | None,
                     pin: Path | None = None) -> dict[str, Any]:
    """`CHECKED`'s auditor-listing, its registered-files condition: the files remote's sampled proofs case at ``case``
    (`SAMPLED_ID`'s, ``{"params", "commitment", "blobs", ...}``) registers, the circuit file and then the public file
    (`proofs.evidence_v2`'s order), are by SHA-512 those its params θ name (``proofs.circuit``, ``proofs.public``), θ
    held by SHA-512 to the params each composite in ``found`` that reads the case states for it, and the public file
    the one a5 pinned at staging, before the draw (``pin``, `pinned`). ``holds``, ``read_by``, those composites' ids,
    ``compared``, the files θ names, ``why`` where it doesn't hold, and ``unbound`` (`PUBLIC_UNBOUND`) where θ names no
    public file."""
    import composite as CO

    stated = {p: c.get("params_sha512") for p, (_, doc) in found.items() for c in doc.get("cases") or ()
              if c.get("id") == SAMPLED_ID}
    commits = {c.get("commitment") for _, doc in found.values() for c in doc.get("cases") or ()
               if c.get("id") == SAMPLED_ID}
    res: dict[str, Any] = {"holds": False, "read_by": sorted(stated)}
    if not stated:
        return res | {"holds": True}
    if case is None:
        return res | {"why": f"the run's composites read {SAMPLED_ID}'s case, and the run gave no --sampled-proofs-case"}
    try:
        doc = json.loads(case.read_bytes())
        theta, blobs, proofs = doc["params"], [Path(b) for b in doc["blobs"]], dict(doc["params"]["proofs"])
    except (OSError, KeyError, TypeError, ValueError) as e:
        return res | {"why": f"--sampled-proofs-case isn't a case whose params name proofs and which names its blobs: "
                             f"{type(e).__name__}"}
    if set(stated.values()) != {hashlib.sha512(CO.canonical(theta)).hexdigest()}:
        return res | {"why": f"--sampled-proofs-case's params aren't, by SHA-512, those each composite that reads "
                             f"{SAMPLED_ID}'s case states (its params_sha512)"}
    res["compared"] = [h for h in ("circuit", "public") if h in proofs]
    if "public" not in proofs:
        res["unbound"] = PUBLIC_UNBOUND
    for i, half in enumerate(("circuit", "public")):
        if half == "public" and half not in proofs:
            continue
        try:
            with open(blobs[i], "rb") as f:
                got = hashlib.file_digest(f, "sha512").hexdigest()
        except (IndexError, OSError) as e:
            return res | {"why": f"the case's {half} file (its blob {i}) can't be read: {type(e).__name__}"}
        if got != proofs.get(half):
            return res | {"why": f"the {half} file the case registers (its blob {i}) isn't, by SHA-512, θ's "
                                 f"proofs.{half}"}
    if "public" in proofs and (why := pinned(pin, doc.get("commitment"), commits, blobs[1])) is not None:
        return res | {"why": why}
    return res | {"holds": True}


def of_record(public: Mapping[str, Any], ell: Sequence[int], K: int, prem: Mapping[str, Any],
              conditional: Sequence[str] = ()) -> dict[str, Any]:
    """The figure of record (`OPEN`) from the period's public, the audits' caps ``ell``, the tight K's premises
    ``prem`` (`premises`) and the conditions no charge covers (``conditional``: live-coins while the coins are open,
    `COINS`): ``K``, the checks the public holds, what is open (those checks, the premises and the conditions), the
    advice ℓ_a + 1 charges, summed over each phase's tied audits (the public's ``scans[j].tied``, else every audit:
    `TIED_ALL`), and ``conditional_on``, the conditions, where any is open."""
    held = [c for c in CHECKS if c in (public.get("held") or ())]
    open_ = [c for c in CHECKS if c not in held] + list(prem["open"]) + [c for c in conditional if c not in prem["open"]]
    tied = [s.get("tied") for s in public["scans"]]
    per = [sum(ell[int(i)] + 1 for i in (t if t is not None else range(len(ell)))) for t in tied]
    return {"K": K + (sum(per) if open_ else 0), "held": held, "open": open_, "is": OPEN, "premises": prem,
            "advice": {"bits_per_phase": per,
                       "tied": "scans[j].tied" if all(t is not None for t in tied) else TIED_ALL}} | (
        {"tight": TIGHT} if open_ else {}) | ({"conditional_on": list(conditional)} if conditional else {})


def figure(out: Path, case: Mapping[str, Any], public: Mapping[str, Any], K: int, prem: Mapping[str, Any],
           conditional: Sequence[str] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    """The figure of record (`of_record`, at ``conditional``) and ``advice_charge``'s ``cap`` from OUT's advice
    registration (`REGISTRATION`) and the case's commitment: both the one ``missing`` or ``error`` where either can't
    be read."""
    try:
        cap = advice_cap(json.loads((out / REGISTRATION).read_text()))
    except FileNotFoundError:
        return (why := {"missing": NO_REGISTRATION}), why
    except (KeyError, TypeError, ValueError) as e:
        return (why := {"error": f"{REGISTRATION}: {type(e).__name__}: {e}"}), why
    try:
        ell = caps(case, cap)
        return of_record(public, ell, K, prem, conditional), {"ell_a": ell, "is": CAP}
    except (IndexError, KeyError, TypeError, ValueError) as e:
        return (why := {"error": f"{CASE} commitment: {type(e).__name__}: {e}"}), why


def named(path: str, tie: bool = False) -> list[re.Pattern[str]]:
    """Each file a `REPROOF` path names, as a pattern over a listing's paths: ``, `` separates paths, ``{a,b}`` is each of
    a and b, ``<tie>`` -tie where ``tie`` (H1_TIE ties the window) and nothing otherwise, any other ``<x>`` any one name,
    and a trailing ``/`` a directory holding at least one file."""
    out = []
    for part in path.replace("<tie>", "-tie" if tie else "").split(", "):
        m = re.fullmatch(r"(.*)\{([^}]*)\}(.*)", part)
        for p in [m[1] + x + m[3] for x in m[2].split(",")] if m else [part]:
            rx = "".join("[^/]+" if t.startswith("<") else re.escape(t) for t in re.split(r"(<[^>]+>)", p))
            out.append(re.compile(rx + (".+" if p.endswith("/") else "")))
    return out


def reproof(out: Path, copy: str | None = None, listing: Path | None = None, tie: bool = False) -> dict[str, Any]:
    """The re-proof inputs (`REPROOF`), each with where it is, its path (``<tie>`` -tie where ``tie``: H1_TIE ties the
    window) and what it holds; of OUT's whether the run wrote it, and of the serve dir's, where ``copy`` (NODE:/PATH) and
    ``listing`` (each file under PATH, one path relative to it a line) are given, whether the copy's listing holds each
    file its path names (`named`) and at least one under each directory it names."""
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
        path = path.replace("<tie>", "-tie" if tie else "")
        inputs[name] = {"in": at, "path": path, "holds": what}
        if at == "OUT":
            inputs[name]["present"] = (out / path).exists()
        elif files is not None:
            inputs[name]["copied"] = all(any(rx.fullmatch(f) for f in files) for rx in named(path))
    if files is not None:
        kept["not_copied"] = [n for n, x in inputs.items() if x.get("copied") is False]
    return {"inputs": inputs, "copy": kept, "kept": REPROOF_KEPT}


def alpha_gamma(path: Path | None) -> dict[str, str]:
    """Where the inference-only case's α and Γ came from, by its private policy.json at ``path``, of which only ``read``
    is read, never α or Γ: off the run's own division where it says so (`ALPHA_GAMMA_FROM_DIVISION`), the run's ALPHA
    and GAMMA where the case has none (`ALPHA_GAMMA_GIVEN`); ``missing`` with no path, ``error`` for a file
    served_cases.py didn't write."""
    if path is None:
        return {"missing": ALPHA_GAMMA_UNSTATED}
    if not path.exists():
        return {"read": "given", "is": ALPHA_GAMMA_GIVEN}
    from served_cases import FROM_DIVISION
    try:
        read = json.loads(path.read_text()).get("read")
    except (OSError, ValueError, AttributeError) as e:
        return {"error": f"the inference-only case's policy.json isn't a JSON object: {type(e).__name__}"}
    if read != FROM_DIVISION:
        return {"error": "the inference-only case's policy.json doesn't say served_cases.py read it off the division"}
    return {"read": "the division", "is": ALPHA_GAMMA_FROM_DIVISION}


def record(out: Path, advice_charge: Path | None = None, copy: str | None = None,
           listing: Path | None = None, serving: tuple[int, int] | None = None, tie: bool = False,
           certifier_key: Path | None = None, site_registration: Path | None = None,
           case_policy: Path | None = None, sampled_case: Path | None = None,
           public_pin: Path | None = None) -> dict[str, Any]:
    """OUT's onsite record (module docstring), with the re-proof inputs' ``copy`` and its ``listing`` where given
    (`reproof`, ``tie`` where H1_TIE ties the window), the run's ``serving`` settings, S and MAX_TOKENS, where its
    caller states them (`SERVING`), the auditor's inputs the run's checks read, its ``certifier_key`` record and its
    ``site_registration`` (`auditor_record`), the inference-only case's private policy.json, ``case_policy``
    (`alpha_gamma`), and remote's sampled proofs case, ``sampled_case``, with a5's ``public_pin`` of its public file
    (`registered_files`); FileNotFoundError where
    `served_verdicts.py record` wrote no report."""
    report = json.loads((out / "record" / "verdicts.json").read_text())
    if report.get("format") != EXTENDS:
        raise ValueError(f"record/verdicts.json is {report.get('format')!r}, not {EXTENDS}")
    advice = {"served": ADVICE, "charged": {"missing": "no bound"}, "cap": {"missing": "no bound"},
              "measured": measured(advice_charge)}
    rows = report.get("protocols") or {}
    net, net_src = network_row(out, rows, report.get("run"))
    if net is not None:
        rows = {**rows, ROWS["the network certifier"]: net}
    found = composites(out)
    composite, with_pouw = of_record_composite(found)
    unread = NOT_CLAIMED if with_pouw else NOT_CLAIMED_WITHOUT_POUW
    v = {"sampled proofs": sampled_verdict(found, composite, rows), "PoUW": row_verdict(rows, ROWS["PoUW"])}
    v["inference-only"] = case_verdict(found, REMOTE if with_pouw else composite, INFERENCE_ONLY)
    v["remote"] = composite_verdict(found, REMOTE, "remote's composite didn't run: the run has no "
                                                   "composite/remote/composite.json")
    v["the network certifier"] = nv = row_verdict(rows, ROWS["the network certifier"], net_src)
    v["onsite"] = composite_verdict(found, composite, f"the run has no composite of {composite}")
    v = {k: v[k] for k in VERDICTS}
    v["inference-only"]["alpha_gamma"] = alpha_gamma(case_policy)
    for k in unread:
        v[k]["not_claimed"] = POUW_OPEN
    case, no_case = period_case(out)
    composed = next((c.get("params_sha512") for c in found[composite][1].get("cases") or ()
                     if c.get("id") == NETWORK), None) if composite in found else None
    listed = auditor_listing(out, case) if case is not None else {
        "is": CHECKED[AUDITOR_LISTING], "holds": False, "why": f"no case lists the period's audits: {no_case}"}
    files = registered_files(found, sampled_case, public_pin)
    checked = {AUDITOR_RECORD: auditor_record((case or {}).get("params"), composed, certifier_key, site_registration),
               AUDITOR_LISTING: listed | {"holds": listed["holds"] and files["holds"], "registered_files": files}
               | ({"why": files["why"]} if listed["holds"] and not files["holds"] else {})}
    prem = premises(LOCKS, checked=checked)
    drawn, draws = coins(out), chain_draws(out)
    cond = [COINS_OPEN] if drawn["open"] else []
    if cond:
        v["sampled proofs"]["conditional_on"] = cond
    unlisted = [LISTING_UNMET] * (not listed["holds"]) + [FILES_UNMET] * (not files["holds"])
    unmet = {n: why for n, why, x in ((CERTIFIER_KEY, KEY_OPEN, CERTIFIER_KEY in prem["open"]),
                                      (AUDITOR_LISTING, "; ".join(unlisted), bool(unlisted))) if x}
    if nv["verdict"] == "accept" and unmet:
        v["the network certifier"] = nv | {"verdict": NOT_ESTABLISHED, "answer": nv["verdict"], "open": list(unmet),
                                           "why": "; ".join(unmet.values())}
    withheld = [("sampled proofs", PUBLIC_UNBOUND)] * ("unbound" in files)
    withheld += [(k, FILES_UNMET) for k, p in (("sampled proofs", composite), ("remote", REMOTE), ("onsite", composite))
                 if not files["holds"] and p in files["read_by"]]
    for k, why in withheld:
        if v[k]["verdict"] == "accept":
            v[k] = v[k] | {"verdict": NOT_ESTABLISHED, "answer": "accept", "open": [REGISTERED_FILES], "why": why}
    paid = credit(out) if v["sampled proofs"]["verdict"] != NOT_ESTABLISHED else {
        "credited": [], "missing": UNESTABLISHED_CREDIT, "is": CREDIT}
    v["sampled proofs"]["credit"] = paid
    accepted = nv["verdict"] == "accept" and AUDITOR_LISTING not in unmet
    open_draws = ["sampled proofs' draw"] * bool(cond) + (draws["open"] if accepted else [])
    drawn |= {"open": bool(open_draws), "open_draws": open_draws, "draws": draws}
    cond_net = [COINS_OPEN] * bool(open_draws) + [n for n in UNCHARGED if n in prem["open"]]
    net = rows.get(ROWS["the network certifier"]) or {}
    answer = net.get("answer") if isinstance(net.get("answer"), Mapping) else {}
    profile = answer.get("profile") if isinstance(answer.get("profile"), Mapping) else {}
    tree = next((x for x in lines(out, "machine.txt") if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", x)), None)
    held = cond + [c for c in cond_net if c not in cond] if accepted else cond
    rec: dict[str, Any] = {
        "format": FORMAT, "extends": EXTENDS, "run": report.get("run"), "tree": tree, "node": report.get("node"),
        "serving": {"missing": NO_SERVING} if serving is None else {
            "window_steps": serving[0], "max_tokens": serving[1], "is": SERVING},
        "verdicts": v, "checked": checked, "coins": drawn, "one_program_a_day": day(out),
        "pouw_freed_compute": pouw(out),
        "of_record": {"holds": all(x["verdict"] == "accept" for k, x in v.items() if k not in unread)
                               and credited(paid), "composite": composite,
                      "missing": [k for k, x in v.items() if x["verdict"] == "missing"],
                      "not_accepting": [k for k, x in v.items()
                                        if x["verdict"] not in ("accept", "missing", NOT_ESTABLISHED)],
                      "not_established": [k for k, x in v.items() if x["verdict"] == NOT_ESTABLISHED],
                      "not_credited": list(paid.get("not_credited") or ()), "not_claimed": list(unread),
                      "verified_compute": NO_VERIFIED_COMPUTE}
                     | ({"uncredited": paid["missing"]} if "missing" in paid else {})
                     | ({"conditional_on": held} if held else {}),
        "sampled_proofs": sampled(rows.get(ROWS["sampled proofs"]), v["sampled proofs"], *phase_seconds(out), cond),
        "placement": profile.get("placement"), "advice_charge": advice,
        "composite_error": composite_error(rows, None, conditional=cond, composite=composite, pouw=with_pouw),
        "reproof": reproof(out, copy, listing, tie)}
    cond = cond_net
    if not accepted:
        nv = v["the network certifier"]
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
    rec["composite_error"] = composite_error(rows, eps_of(b), conditional=cond, composite=composite, pouw=with_pouw)
    advice["charged"] = charged(b)
    b["period_s"] = None
    b["of_record"], advice["cap"] = figure(out, case, answer["public"], b["K"], prem, cond)
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
    rec["periods"] = ({"conditional_on": cond} if cond else {}) | periods(K, eps, m, phase_s, connections, capacity)
    rec["periods"]["eps"] = pooled_eps(b["delta"], m, rec["periods"])
    return rec


def summary(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ONSITE`` line: the run's serving settings, each verdict, where the inference-only case's α and Γ came from
    (`alpha_gamma`), whether the run is of record, its composite of
    record by id (`WITHOUT_POUW`) and the verified compute it credits, none, stated at that id (``compute``,
    `NO_COMPUTE_CLAIM`), and what every figure is conditional on
    (`COINS`; open for a record that read no coins), what the record doesn't claim (`DAY`, `POUW`), the drawn members it
    doesn't credit (`CREDIT`), K of record and what is open under it, its δ_net's terms, eps's split and connections,
    the tied audits the period reads and eps per window and pooled over each monitoring period (`pooled_eps`), the
    composite's stated error, the year's datacenter cell, sampled proofs' guarantee per window and pooled
    (`guarantee`), and the measured advice charge's test run."""
    b, p, s = rec.get("bound") or {}, rec.get("periods") or {}, rec.get("serving") or {}
    year = (p.get("year") or {}).get("datacenter")
    m, fig = rec["advice_charge"]["measured"], b.get("of_record") or {}
    return {"serving": {k: s[k] for k in ("window_steps", "max_tokens")} if "window_steps" in s else None,
            "verdicts": {k: x["verdict"] for k, x in rec["verdicts"].items()},
            "inference_only_alpha_gamma": rec["verdicts"]["inference-only"].get("alpha_gamma"),
            "of_record": rec["of_record"]["holds"],
            **{k: rec["of_record"].get(k) for k in ("composite", "verified_compute")},
            "compute": NO_COMPUTE_CLAIM.format(rec["of_record"].get("composite") or "the composite of record"),
            "conditional_on": rec["of_record"].get("conditional_on", []) if "coins" in rec else [COINS_OPEN],
            "not_claimed": [DAY_OPEN] * (rec.get("one_program_a_day") or {"open": True})["open"] + [POUW_OPEN],
            "not_credited": rec["of_record"].get("not_credited", []),
            "K": fig.get("K"), "open": fig.get("open"),
            "delta_net": None if "delta_net" not in b else delta_net(eps_of(b), b["phases"]),
            "delta": None if "delta" not in b else {k: b["delta"][k] for k in ("scan", "charged")},
            "eps_pooled": None if "eps" not in p else {
                "tied_audits_read": p["eps"]["tied_audits"].get("read"), "per_window": p["eps"]["per_window"]["eps"],
                **{name: {where: p["eps"][name][where]["eps"] for where in ("node", "datacenter", "world")}
                   for name in LB.PERIODS}},
            "composite_error": {k: rec["composite_error"][k] for k in ("stated", "vacuous")},
            "connections": (b.get("connections") or {}).get("count"),
            "year_datacenter": None if year is None else {
                k: year[k] for k in ("bits", "share", "share_served", "share_of_capacity", "delta_net", "how")},
            "sampled_proofs": guarantee(rec.get("sampled_proofs")),
            "advice_measured_on": m.get("run"), **{k: b[k] for k in ("missing", "error") if k in b}}


def guarantee(g: Mapping[str, Any] | None) -> Any:
    """Sampled proofs' guarantee (`sampled`) as its window's ε at δ beside each pooled cell's count and share, and its
    ``missing`` or ``error`` where it states none; None for a record that has none."""
    if not isinstance(g, Mapping):
        return None
    if "per_window" not in g:
        return {k: g[k] for k in ("missing", "error") if k in g}
    p = g["pooled"]
    pooled = p if "q" not in p else {"incorrect_at_most": p["incorrect_at_most"]} | (
        {"periods": p["periods"]} if "periods" in p else {
            name: {where: {k: p[name][where][k] for k in ("incorrect_at_most", "share")}
                   for where in ("node", "datacenter", "world")} for name in LB.PERIODS})
    return {"per_window": {k: g["per_window"][k] for k in ("epsilon", "delta", "units", "law")}, "pooled": pooled}


def cells(p: Mapping[str, Any] | None) -> Any:
    """``p`` (`periods`) as each cell's bits, share and how, the run's own period's with its phases and its figure over
    one; ``p`` as it is where it holds no figure (its ``missing`` or ``error``, or None)."""
    if not isinstance(p, Mapping) or "run" not in p:
        return p
    run = p["run"]
    return ({"conditional_on": p["conditional_on"]} if "conditional_on" in p else {}) | {
        "run": {"bits": run["bits"], "share": run["share"], "phases": run["phases"],
                "per_phase": run["per_phase"]["bits"]},
        **{name: {where: {k: p[name][where][k] for k in ("bits", "share", "how")}
                  for where in ("node", "datacenter", "world")} for name in LB.PERIODS}}


def beside(paths: Sequence[Path]) -> dict[str, Any]:
    """The records at ``paths``, each a run's (`FORMAT`), beside each other (`BESIDE`): each run's figures labelled with
    its serving settings and ℓ_a. ValueError, by path, for a record of another format or one stating no serving
    settings (`UNLABELLED`)."""
    runs = []
    for path in paths:
        rec = json.loads(path.read_text())
        if rec.get("format") != FORMAT:
            raise ValueError(f"{path} is {rec.get('format')!r}, not {FORMAT}")
        line = summary(rec)
        if line["serving"] is None:
            raise ValueError(f"{path} {UNLABELLED}")
        runs.append({"record": str(path), "run": rec.get("run"), "tree": rec.get("tree"), "serving": line["serving"],
                     "ell_a": (rec["advice_charge"].get("cap") or {}).get("ell_a"), "of_record": line["of_record"],
                     **{k: line[k] for k in ("composite", "verified_compute", "compute")},
                     "conditional_on": line["conditional_on"], "not_claimed": line["not_claimed"],
                     "not_credited": line["not_credited"], "K": line["K"], "open": line["open"],
                     "periods": cells(rec.get("periods")), "sampled_proofs": line["sampled_proofs"]})
    return {"format": BESIDE, "is": BESIDE_IS, "runs": runs}


def positive(s: str) -> int:
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{s} isn't a whole number of at least 1")
    return n


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, nargs="?", help="the served run's public $OUT")
    ap.add_argument("--window-steps", type=positive, metavar="S",
                    help="the run's S, served_verdicts.sh's WINDOW_STEPS, with --max-tokens (`SERVING`)")
    ap.add_argument("--max-tokens", type=positive, metavar="M",
                    help="the run's MAX_TOKENS, served_verdicts.sh's, with --window-steps")
    ap.add_argument("--beside", type=Path, nargs="+", metavar="RECORD",
                    help="runs' records written, set beside each other (`BESIDE`) at --to, in place of OUT's")
    ap.add_argument("--advice-charge", type=Path,
                    help="a test run's advice-charge.json (advice_charge.py), stated beside the bound as that run's")
    ap.add_argument("--reproof-copy", metavar="NODE:/PATH",
                    help="where the serve dir's re-proof inputs were copied off the served node, keeping its layout")
    ap.add_argument("--reproof-listing", type=Path,
                    help="the copy's listing, kept off OUT: each file under PATH, one path relative to it a line")
    ap.add_argument("--h1-tie", action="store_true",
                    help="the run ties its window to H1's W (H1_TIE): the re-proof's circuit is the record reading "
                         "H1's tables (`REPROOF`'s <tie>)")
    ap.add_argument("--certifier-key", type=Path, metavar="RECORD",
                    help="the auditor's record of its network certifier (certifier-key/v1, CERTIFIER_KEY), never the "
                         "prover's node's: the run's auditor-record check holds the period's key, LMS parameters and "
                         "session to it (`CHECKED`)")
    ap.add_argument("--site-registration", type=Path, metavar="FILE",
                    help="the auditor's registration of the deployment, {\"params\": {auditor, composite}, \"log\": "
                         "{composite, entries}}, whose auditor must hold the record's key for a network certifier the "
                         "log places (`CHECKED`)")
    ap.add_argument("--inference-only-case-policy", type=Path, metavar="FILE",
                    help="the inference-only case's private policy.json under the serve dir (served_cases.py's, not "
                         "$OUT/inference-only-policy.json), of which only its read is read: where the case's α and Γ "
                         "came from, the run's ALPHA and GAMMA where it doesn't exist (`alpha_gamma`)")
    ap.add_argument("--sampled-proofs-case", type=Path, metavar="FILE",
                    help=f"remote's sampled proofs case under the serve dir ({SAMPLED_ID}'s, served_cases.py's "
                         "private/sampled-proofs/case.json), whose registered files auditor-listing holds to its "
                         "params' digests (`registered_files`)")
    ap.add_argument("--public-pin", type=Path, metavar="FILE",
                    help="a5's pin of the audit's public files at staging, before the draw (its out's "
                         "public-pin.json), to which auditor-listing holds the public file the case registers "
                         "(`pinned`)")
    ap.add_argument("--to", type=Path,
                    help="the record's path (default OUT/record/onsite.json); with --beside, where the records go "
                         "beside each other")
    a = ap.parse_args(argv)
    if a.beside is not None:
        if a.to is None or a.h1_tie or any(x is not None for x in (a.out, a.advice_charge, a.reproof_copy,
                                                                     a.reproof_listing, a.window_steps, a.max_tokens,
                                                                     a.certifier_key, a.site_registration,
                                                                     a.inference_only_case_policy,
                                                                     a.sampled_proofs_case, a.public_pin)):
            ap.error("--beside reads the records written, with --to alone: where they go beside each other")
        try:
            doc = beside(a.beside)
        except (OSError, ValueError) as e:
            print(f"ONSITE-BESIDE: not written: {e}", file=sys.stderr)
            return 1
        a.to.parent.mkdir(parents=True, exist_ok=True)
        a.to.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        print("ONSITE-BESIDE " + json.dumps([{k: r[k] for k in ("run", "serving", "ell_a", "composite",
                                                                 "verified_compute", "compute", "conditional_on",
                                                                 "not_claimed", "not_credited", "K", "open")}
                                             for r in doc["runs"]], sort_keys=True, ensure_ascii=False), flush=True)
        return 0
    if a.out is None:
        ap.error("give the served run's OUT, or --beside its records")
    if (a.window_steps is None) != (a.max_tokens is None):
        ap.error("--window-steps and --max-tokens go together: the run's S and MAX_TOKENS label its figures")
    serving = None if a.window_steps is None else (a.window_steps, a.max_tokens)
    try:
        rec = record(a.out, a.advice_charge, a.reproof_copy, a.reproof_listing, serving, a.h1_tie,
                     a.certifier_key, a.site_registration, a.inference_only_case_policy, a.sampled_proofs_case,
                     a.public_pin)
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
