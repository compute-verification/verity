"""Compute efficiency eta of frontier pretraining from scratch under the verification stack: the FLOP an unconstrained run
needs, divided by the FLOP the developer spends, monitored plus unmonitored. The developer must still compute the counted
multiplications M (an updated weight times an activation or activation gradient). Terms and symbols follow the proposal
"Terms and symbols for the training-efficiency bound" (Notion, under the assumptions register) and ARCHITECTURE.md's Glossary.

eta <= s + min(r, memory cap) + 1/rho_NCI + chi, each term an upper bound under the model; ``STATUS`` says which steps are
proved.

* s, the unmonitored compute share, trains at full efficiency.
* r, the monitored compute residual: r-hat, the share of monitored capacity outside the compute certificate
  (1 - W_cert/C), plus PoUW's deficit on the certified work, eps_W W_cert/C, plus the aggregation slack eps'. r-hat is
  idle and uncredited GPU time, plus host CPUs, which PoUW neither credits nor bounds and which count at their datasheet
  peak. It trains at full efficiency unless the memory cap binds: at peak in eta, at MFU_0 in D (``evaluate``'s
  ``residual_mfu``, the reference-efficiency assumption: free compute runs no better than the reference). A
  spatial unit's window reads its starting state plus its novel intake, K_w = state + lambda_in L, and contains at most
  K_w Lambda(K_w) / b_a counted multiplications (``ThreadedRuntime``, reads counted by width; counting them by
  information, which a wipe would bound, is open: ``ThreadedOfState`` rests on the falsified WeightsSupport,
  ``ThreadedOfStateFmt`` on the falsified WeightsSupportFmt and ``ThreadedOfStateFmt2`` on its restatement
  WeightsSupportFmt2, not yet signed off), declared or not. The state is all memory,
  disks included unless they're unplugged; after a wipe, the wipe residual R_wipe: memory the fill doesn't reach, the proof of
  space's slack on the filled memory, and what helpers refill within the read deadline; under exhaustion, the HBM the
  declared computation doesn't hold plus the slack. The slack is 1 - delta^(1/k) for k timed rounds (1/19 at k = 86 and
  delta = 1/100) plus the certificate's floor, the share of the store rebuildable from public data within the deadline.
  The novel intake is line rate or, under network accounting, side channels plus certified timing plus the declared
  computation's novel intake, alpha F under NCI once the count check is in the ZK flow.
* 1/rho_NCI: the declared computation trains at the thread bound, rho_T = b_a / (alpha Lambda(K)) declared gates per
  counted multiplication with Lambda(K) = max(e, 2K / (4 d b_w)), or at the tighter partial-sum law rho_PS when no dark
  compute can sum the inference blocks' partial sums. The residual reads them on the server for free; unmonitored compute
  reads them through the export (line rate, or side channels plus readable export under network accounting). A relay op
  reads 2 b_p bits, and the developer relays only as many partial sums as buy more declared work than the relay's gates
  would train directly.
* The verdict lag: from its first faulty proof block until the halt (a window plus the verdict delay), a developer can skip
  all declared work, once per identity, in every tier; that capacity joins the residual. Detection must end the
  identity and take its hardware out of use, or the capacity after the halt is unmonitored.
* chi, the cheating share: with sampled proofs at a budget, the faulty proof blocks the sampling design accepts number
  A ln(1/delta) / q per identity over the whole deployment, q = share / (overhead + replay)
  (``Population.Exfil.tail_cumPrev``), where A, the luck factor, turns faulty outputs into skipped work. A block that
  injects a chosen output fails its proof whenever it is drawn, so luck doesn't multiply those: ln(1/delta) / q of them
  inject their output bits into the inference block that holds them (the width form needs no bits for where), each bit
  worth at most ``chord``, the steepest secant from K of one inference block's thread bound.

The four terms bound counted multiplications relative to the reference recipe. A private recipe g times better in counted
multiplications per unit of loss is a compute amplifier, multiplying eta by g; it cancels when verification is compared on
against off. A recipe better at working under the mechanisms, m times more progress per counted multiplication, scales
only the constrained terms (nu, mu and chi), not the residual or unmonitored compute. The compute dampener D, the
hardware multiple over an unconstrained developer with the same recipe on the same hardware at model-FLOP utilization
mfu, isn't computed here: it is the overhead calculator's ``rho.theorem.multiple_bound`` in Lean's form
(``Nci.Dampener.remoteDampener``) on these terms (``remote_terms``) with the residual at MFU_0.

The canvas "Compute efficiency under the verification stack" renders this model.

    python benchmarks/nci/tiers.py            # the five tiers at the defaults
    python benchmarks/nci/tiers.py --json     # spec, defaults, tiers, status and results, for other renderers
"""

from __future__ import annotations

import argparse
import json
import math

YEAR = 3.156e7
GB = 8e9
#: Lemma M's audit window: the most a helper can add to a store under audit is the side channel over it
T_AUDIT = 0.3635
#: the proof-of-space band block, and the bits that name one, for the outsourcing attack on a wipe
BLOCK_BITS = 524288
BLOCK_NAME_BITS = 25
#: ReferenceThreaded's (d, e) for Kimi K2 at S = 2e7 under the open conjecture ReferenceLayered, at Dot16 granularity in
#: products; the default d is 1 until Layered's lookups may read weight columns (a trained embedding)
REFERENCE_THREADED = (10.0, 37248.0)

#: the ACCEL model's card table, the one hardware source for the GPUs here: the RTX PRO 6000 Server Edition at its
#: firmware's maximum (``Verity.Accel.Sm120.card``, 188 SMs, 1024 tensor units an SM clock, a FP8 E4M3 MAC one unit,
#: the clock's ceiling 2430 MHz), pinned from Security/Definitions/Core/Accel/Sm120.lean;
#: tests/test_tiers.py holds the pin to the file
CARD = {"sms": 188, "tensor": 1024, "clock_hz": 2430e6}
#: the clock PoUW credits on sm_120 (H1's SM120 record, verity/ml/protocols/pouw/circuit/h1device.py: 2430 MHz)
CREDITED_CLOCK_HZ = 2430e6
#: a server's GPUs at the table's FP8 dense peak, 2 FLOP a MAC
GPU_FLOPS = 2 * CARD["sms"] * CARD["tensor"] * CARD["clock_hz"]
NODE_FLOPS = 8 * GPU_FLOPS

#: defaults: the Kimi K2 calibration and forward chain; the GPUs' residual under the floor credit at f_max,
#: 1 - ((1 - phi)/q_sched + phi)/q_clock = 0.323: honest kernels run 1.72x the floor at the nodes' locked clock,
#: diluted by the training window's filler (the C7 dossier, whose 0.326 also takes its
#: eps = 0.54% off the credited time, which eps_w charges here instead; 0.213 with an attested clock; 1% needs eight
#: conditions; today nothing in a training window is credited); PoUW's deficit eps_W = 0.744% (CircuitCredit);
#: the GPU clock above what PoUW credits, 0: PoUW credits 2,430 MHz, the card table's ceiling, which every
#: card on vy-nebius-1 reports as its maximum, application, default and boost SM clock;
#: the host CPUs at 0: hosts carry no compute (GLOSSARY.md, the device model), a server's compute is its accelerators'
#: (two Xeon 6776Ps at their INT8 AMX datasheet peak, 12.6% of the GPUs' FP8 rate, is the sensitivity without that
#: assumption); 100,000 servers of 8 RTX PRO 6000s (vy-nebius-2's 2.668 TB) at the card table's peak for a
#: year;
#: the deployed proof-of-space audit (k = 111 rounds at delta = 1/100; the slack counts the floor once) and the memory
#: the fill doesn't reach (3.05 GB on-chip, 18.4 GB GPU reserve, 29.5 GB host gap, about 7 GB of DPU, BMC and flash); disks
#: unplugged; today's uncharged export; the declared computation's novel intake at alpha F (once the network's count check is in
#: the ZK flow; line rate until then); sampled proofs at 1% on top of serve-time commitments, at today's sparse-draw cost
#: (about 35 GPU-s a draw), delta = 2^-40 and luck A = 8; the structural d = 1, what Lean certifies for a run that trains
#: its embedding, with ReferenceThreaded's e for a reference circuit written in Dot16 steps, as the declarations are, in
#: products (``REFERENCE_THREADED`` is the conditional (d, e)); injected bits valued with (d, e)-threading no further than
#: ThreadedOfLayered certifies any at ReferenceLayered's bf16 embedding columns, and every gate of a block countable
DEFAULTS = {
    "unmonitored": 0.0, "residual": 0.323, "host_share": 0.0,
    "clock_share": 1 - CREDITED_CLOCK_HZ / CARD["clock_hz"], "eps_w": 0.00744,
    "eps_agg": 1e-4,
    "pous_rounds": 111.0, "unfilled_gb": 58.0, "sigma": 1e3, "b_w": 2.5, "b_a": 4.0,
    "f": 0.95, "pous_delta": 0.01, "pous_floor": 0.0, "t_read": 1e-3, "wipe_s": 3600.0, "hbm_gb": 768.0,
    "host_gb": 1900.0, "disk_gb": 0.0,
    "tau": 860.0, "export": 4.4e9, "nic": 3.2e12,
    "share": 0.01, "overhead": 4.8e12, "replay": 1.1e5, "step_flop": 1.43e4, "p_evade": 2.0 ** -40, "identities": 1.0,
    "luck": 8.0, "step_bits": 16.0, "intake_x": 1.0, "lag_s": 3600.0, "g": 1.0, "mfu": 0.4, "m": 1.0,
    "floor_steps": 1.0,
    "K": 2e7, "Gamma": 6.4e16, "d": 1.0, "e": 37248.0, "kappa": 0.6, "thread_bits": 4.46e7, "block_mults": 1.0,
    "kappa_chain": 0.0, "d_chain": 1536.0, "e_chain": 12288.0, "b_p": 32.0,
    "nodes": 1e5, "node_flops": NODE_FLOPS, "years": 1.0,
}

#: one entry per parameter: label, unit, slider range, log scale, whether the leftmost position means zero, and whether
#: it is one of the few shown up front
SPEC = [
    {"key": "unmonitored", "main": True, "label": "Unmonitored compute share", "unit": "", "min": 1e-12, "max": 0.5, "log": True, "zero": True},
    {"key": "g", "main": True, "label": "Compute amplifier g: the developer's algorithmic advantage over the reference recipe (1 against the best known recipe)", "unit": "x", "min": 1, "max": 100, "log": True},
    {"key": "residual", "main": True, "label": "Monitored compute residual r-hat on the GPUs, their share outside the certificate (floor credit at the locked clock 0.323, with an attested clock 0.213; best plausible design 0.01; nothing credited today: 1)", "unit": "", "min": 1e-12, "max": 1, "log": True, "zero": True},
    {"key": "eps_w", "label": "PoUW's deficit eps_W: the share of certified work a certificate that isn't wrong may credit without its running, which charges eps_W W_cert/C to r (CircuitCredit 0.744%; best plausible design 0.06-0.25%)", "unit": "", "min": 1e-4, "max": 0.1, "log": True, "zero": True},
    {"key": "eps_agg", "label": "Aggregation slack eps': the share of capacity whose PoUW certificates may be wrong, which Markov's inequality over the windows charges to r (1e-4 with per-window failure 3 x 2^-40 gives the theorem's delta of 2.7e-8; each halving costs about 4e-5 of PoUW's deficit)", "unit": "", "min": 1e-8, "max": 1e-2, "log": True, "zero": True},
    {"key": "clock_share", "label": "GPU capacity above the clock PoUW credits (0: PoUW credits 2,430 MHz, the card table's ceiling and the SKU's maximum, CMP-165; above 0 only on a card whose SKU or firmware reaches past its table)", "unit": "", "min": 1e-4, "max": 0.3, "log": True, "zero": True},
    {"key": "host_share", "label": "Host CPUs' compute, as a share of server compute at the card table's peak: 0 under hosts carry no compute; without that assumption, at datasheet peak (Xeon 6776P INT8 AMX 12.6%; hosts without matrix units, two EPYC 9015s, 0.22%)", "unit": "", "min": 1e-6, "max": 0.5, "log": True, "zero": True},
    {"key": "pous_rounds", "main": True, "label": "Proof-of-space rounds per audit (deployed 111; 1/19 needs 86; streamed labels make 74,000 cheap)", "unit": "", "min": 10, "max": 1e6, "log": True},
    {"key": "unfilled_gb", "main": True, "label": "Memory the fill doesn't reach, per server (measured gaps 51 plus about 7 of DPU, BMC and flash; on-chip alone 3.05)", "unit": "GB", "min": 0.01, "max": 1000, "log": True},
    {"key": "sigma", "main": True, "label": "Side channels per server", "unit": "bit/s", "min": 1, "max": 1e10, "log": True, "zero": True},
    {"key": "b_w", "main": True, "label": "Compression: bits per updated weight", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "b_a", "main": True, "label": "Compression: bits per activation", "unit": "bit", "min": 0.1, "max": 16, "log": True},
    {"key": "f", "label": "Exhaustion: declared share of HBM", "unit": "", "min": 0, "max": 0.999, "log": False},
    {"key": "pous_delta", "label": "Proof-of-space failure probability per certificate", "unit": "", "min": 1e-12, "max": 0.5, "log": True},
    {"key": "pous_floor", "label": "Proof-of-space share the developer can rebuild at any k: 0 when the permutation is ideal or labels are streamed (the slack counts it once); about 0.985 under the broken reference permutation", "unit": "", "min": 1e-6, "max": 1, "log": True, "zero": True},
    {"key": "disk_gb", "label": "Disk per server left in place and unwiped (0 when unplugged)", "unit": "GB", "min": 1, "max": 1e5, "log": True, "zero": True},
    {"key": "t_read", "label": "Wipe read deadline (helpers within it refill proof-of-space blocks)", "unit": "s", "min": 1e-6, "max": 1e-1, "log": True},
    {"key": "wipe_s", "label": "Wipe period", "unit": "s", "min": 1, "max": 1e7, "log": True},
    {"key": "hbm_gb", "label": "HBM per server", "unit": "GB", "min": 80, "max": 4000, "log": True},
    {"key": "host_gb", "label": "Host memory per server (filled with proof of space)", "unit": "GB", "min": 1, "max": 10000, "log": True, "zero": True},
    {"key": "tau", "label": "Certified timing channel per server (the network appendix's program format: about 860)", "unit": "bit/s", "min": 1, "max": 1e5, "log": True, "zero": True},
    {"key": "export", "label": "Export per server: committed values readable outside the declared computation (today 4.4e9; under the export rule about 2.5e6)", "unit": "bit/s", "min": 1, "max": 1e12, "log": True, "zero": True},
    {"key": "nic", "label": "Line rate per server, without network accounting", "unit": "bit/s", "min": 1e9, "max": 1e13, "log": True},
    {"key": "share", "label": "Sampled proofs: compute share", "unit": "", "min": 1e-4, "max": 0.5, "log": True},
    {"key": "overhead", "label": "Sampled proofs: proving cost per unit of step work (sparse draws today 4.8e12; 512 per statement 1.7e10; batched throughput 1.6e8)", "unit": "x", "min": 1, "max": 1e14, "log": True},
    {"key": "replay", "label": "Sampled proofs: replay cost per drawn step", "unit": "x", "min": 1, "max": 1e7, "log": True},
    {"key": "step_flop", "label": "Sampled proofs: work per step", "unit": "FLOP", "min": 1e3, "max": 1e15, "log": True},
    {"key": "p_evade", "label": "Sampled proofs: certificate failure probability delta (planner 2^-40)", "unit": "", "min": 1e-15, "max": 0.99, "log": True},
    {"key": "luck", "label": "Sampled proofs: luck factor F/(1-p) (spec sheet 8)", "unit": "x", "min": 1, "max": 100, "log": True},
    {"key": "step_bits", "label": "Sampled proofs: output bits per step (16, or one 32-bit gate)", "unit": "bit", "min": 1, "max": 64, "log": True},
    {"key": "intake_x", "label": "Declared novel intake, as a multiple of alpha F (1 once the count check is in the ZK flow; line rate is about 1.3e6)", "unit": "x", "min": 1, "max": 1e7, "log": True},
    {"key": "m", "label": "Recipe multiplier m: frontier progress per counted multiplication of the developer's best recipe under the mechanisms, against the unconstrained recipe (central 1, adversarial 3)", "unit": "x", "min": 0.1, "max": 10, "log": True},
    {"key": "mfu", "label": "Model-FLOP utilization of an unconstrained frontier run (Llama 3 405B: 38-43%)", "unit": "", "min": 0.1, "max": 1, "log": False},
    {"key": "lag_s", "label": "Verdict lag: from a first faulty proof block to the halt, a window plus the verdict delay (1 hour assumed)", "unit": "s", "min": 1, "max": 1e7, "log": True, "zero": True},
    {"key": "identities", "label": "Sampled proofs: identities a detection doesn't end", "unit": "", "min": 1, "max": 1e6, "log": True},
    {"key": "floor_steps", "label": "Sampled proofs: steps in a class drawn at a floor of one", "unit": "", "min": 1, "max": 1e17, "log": True},
    {"key": "K", "label": "NCI: novel bits per inference block", "unit": "bit", "min": 1e5, "max": 1e10, "log": True},
    {"key": "Gamma", "label": "NCI: gates per inference block", "unit": "", "min": 1e14, "max": 1e19, "log": True},
    {"key": "d", "label": "Model: entry fee d (thread reads below which a value's fan-out is at most e). 1 is what Lean certifies for a training run from structure alone: its embedding table is updated weights, which Layered's lookups (columns of novel inputs) can't read. ReferenceThreaded gives 10 for Kimi K2 at S = 2e7 (a Dot16 step's 17 legs less E = 7 bf16 embedding columns) under the open conjecture ReferenceLayered, which holds for a training run only once lookups may read weight columns", "unit": "", "min": 1, "max": 65536, "log": True},
    {"key": "e", "label": "Model: fan-out e, multiplications per activation value (37,248 for Kimi K2: ReferenceThreaded's 2,328 Dot16 steps of 16 products, an MoE input value's, under the open conjecture ReferenceLayered)", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "kappa", "label": "Model: kappa, twice the counted multiplications over the training circuit's gates (Kimi K2's pretraining schedule 0.60; adversarial 0.5; kappa.py)", "unit": "", "min": 0.05, "max": 1, "log": False},
    {"key": "thread_bits", "label": "Model: reads up to which one inference block is held to (d, e)-threading, only (1, 0) above (ThreadedOfLayered certifies a Dot16 reference circuit 17 - E legs below E + 1 lookup columns, none at 17: 17 x 163,840 x 16 bits = 4.46e7 at the bf16 embeddings ReferenceLayered states; at C1's 2.5-bit floor 7.0e6, below K, so a reference circuit C1 allows may get no (d, e) at K; the whole embedding table at 2.5 bits, 2.9e9, needs a refined layered theorem). The certified legs fall to 1 near it, which matters only to a pool too small to fill one block", "unit": "bit", "min": 1e6, "max": 1e11, "log": True},
    {"key": "block_mults", "label": "Model: counted multiplications one inference block can hold, as a share of its gates (1 with no premise; kappa/2 if each block's counted share were the recipe's)", "unit": "", "min": 0.05, "max": 1, "log": False},
    {"key": "kappa_chain", "label": "Model: kappa on the proved chain, one forward path (Kimi K2 0.0695); 0 turns the partial-sum law off, as its support premises (PartialSumsSupport, ActivationsSupport) are falsified", "unit": "", "min": 0.01, "max": 1, "log": True, "zero": True},
    {"key": "d_chain", "label": "Model: chain's smallest inner dimension", "unit": "", "min": 64, "max": 65536, "log": True},
    {"key": "e_chain", "label": "Model: chain's largest fan-out", "unit": "", "min": 1e3, "max": 1e6, "log": True},
    {"key": "b_p", "label": "Model: partial-sum width a relay op reads", "unit": "bit", "min": 4, "max": 64, "log": True},
    {"key": "nodes", "label": "Fleet: servers", "unit": "", "min": 1e2, "max": 1e7, "log": True},
    {"key": "node_flops", "label": "Fleet: server compute (8 RTX PRO 6000s, FP8 dense at the card table's 2,430 MHz)", "unit": "FLOP/s", "min": 1e15, "max": 1e17, "log": True},
    {"key": "years", "label": "Fleet: horizon", "unit": "yr", "min": 0.05, "max": 10, "log": True},
]

#: the cumulative tiers: which mechanisms each adds
TIERS = {
    "T1": {"net": False, "wipes": False, "exhaust": False, "sampled": False},
    "T2": {"net": True, "wipes": False, "exhaust": False, "sampled": False},
    "T3": {"net": True, "wipes": True, "exhaust": False, "sampled": False},
    "T4": {"net": True, "wipes": True, "exhaust": True, "sampled": False},
    "T5": {"net": True, "wipes": True, "exhaust": True, "sampled": True},
}

#: what each step of the model rests on
STATUS = [
    ("Declared computation, thread bound", "ThreadedThrottled is proved in Lean (draft #2343) when the declaration "
     "contains the reference circuit gate for gate, and with dark gates (T1', ThreadedThrottledOn) on a branch. It rests "
     "on WeightsIncompressible and Threaded (d, e). (10, 37248) is ReferenceThreaded's for a reference circuit written "
     "in Dot16 steps, as the declarations are (17 legs less 7 bf16 embedding columns; 2,328 steps of 16 products), under "
     "the open conjecture ReferenceLayered, whose lookups need the embedding columns to be inputs: a run that trains its "
     "embedding reads them as updated weights, so the default is the structural d = 1 until Layered's lookups may read "
     "weight columns."),
    ("Declared computation, partial-sum law", "Off by default (kappa_chain = 0): ContainedTraining and PS' rest on "
     "PartialSumsSupport and ActivationsSupport, which NCI-FALSIFY reproduced as false."),
    ("Monitored compute residual", "r = r-hat + epsilon_W W_cert/C + eps', as Nci.Guarantees.DampenerRemote states it "
     "(#3646). PoUW's deficit epsilon_W = 0.744% (CircuitCredit, draft #2638) is a share of credited work, evaluated "
     "with its premises granted. Under the floor credit r-hat = 0.449: GPU floor time 0.323 (honest kernels at 1.72 "
     "times the floor; run r20261017-123922-e558 has no published outputs), the clock 0 (PoUW credits the card "
     "table's 2,430 MHz, the SKU's maximum, CMP-165; the premise that a node's cards are that SKU and firmware is "
     "unattested) and host CPUs at INT8 AMX peak 0.126. eps' = 1e-4 is what the theorem's delta of 2.7e-8 costs "
     "(Markov over the windows)."),
    ("Memory cap", "ThreadedRuntime and ThreadedRate (draft #2343) give the closed form's shape; its value needs Threaded "
     "at window scale, which fails. The fit bound (FitBoundSlots under ReachSlots and FitsSlots, adopted for dense "
     "recipes by ASK-68) replaces it at token threads, conditional on F1 and F4 per parameter slot, F3, NET-35 (intake "
     "held to alpha F), WipeFullUse (not yet stated) and an information form; it charges each server's first window "
     "(fitBoundSlots_init, fit.py's left_out). A token-routed MoE has no fit rung: ReachSlots at its active count (54 "
     "at 6 GB for Kimi K2's 32B) needs a measured floor on the active count that reaches the target."),
    ("Wipe residual", "Streamed labels (MEM-51) replace the reference permutation, which is refuted (a developer passes "
     "from 1.53% of the store); the slack is proved only for an ideal permutation. Memory outside the fill is measured as "
     "gaps (on-chip 3.05 GB, the GPUs' own about 1.6 GB; GPU reserve 18.4 GB; host 29.5 GB). No theorem bounds help "
     "during a wipe; the helper term is an attack."),
    ("Exhaustion", "Lemma M is proved and locked per audit, and Held for snapshots; nothing bounds memory at every "
     "instant, which the memory cap needs."),
    ("Cheating share", "The bound on accepted faulty proof blocks over the deployment is proved "
     "(Population.Exfil.tail_cumPrev, open #2687; the cumulative harm budget over adaptive rounds on a branch, PRF-86). "
     "Each injected bit is worth at most the chord of one inference block's thread bound, 5.76e7 counted "
     "multiplications, with (1, 0)-threading above thread_bits. chi grows with the identities a detection doesn't end and "
     "shrinks with the fleet, so detection must be terminal (J = 1)."),
    ("Unmonitored compute", "No bound: it trains at full efficiency."),
]


def lam(p: dict, K: float) -> float:
    return max(p["e"], 2 * K / (4 * p["d"] * p["b_w"]))


def region_cap(p: dict, K: float) -> float:
    """The most counted multiplications a region reading ``K`` bits contains."""
    return K * lam(p, K) / p["b_a"]


def chain_rates(p: dict, steps: int = 400) -> list[tuple[float, float]]:
    """``(mu, max(m(mu), mu b_a / Lambda_c))`` on a grid, PS''s per-product read rates (``iu_reads_ge`` and the column
    lemma)."""
    K, b_w, b_a, d = p["K"], p["b_w"], p["b_a"], p["d_chain"]
    lam_c = max(p["e_chain"], K / (4 * b_w * d))
    out = []
    for k in range(1, steps + 1):
        mu = k / steps
        s = math.sqrt(mu)
        c1 = (1 + s) ** 2 * b_w * b_a * d / K
        c2 = math.sqrt(2) * (1 + s) * math.sqrt(max(0.0, 1 - mu)) * b_a * math.sqrt(b_w / K)
        out.append((mu, max(min(c1, c2), mu * b_a / lam_c)))
    return out


def chain_products(rates: list[tuple[float, float]], R: float, Z: float) -> float:
    """PS': the most chain products inference blocks reading ``R`` bits compute when dark gates absorb ``Z`` bits of their
    partial sums."""
    return min((R + (1 - mu) * Z) / rate for mu, rate in rates)


def full_block_bits(p: dict) -> float:
    """The reads at which one inference block's (d, e) thread bound reaches all its work, ``block_mults Gamma`` counted
    multiplications."""
    target = p["block_mults"] * p["Gamma"]
    linear = target * p["b_a"] / p["e"]
    if 2 * linear / (4 * p["d"] * p["b_w"]) <= p["e"]:
        return linear
    return math.sqrt(target * 4 * p["d"] * p["b_w"] * p["b_a"] / 2)


def block_cap(p: dict, S: float, threaded: bool) -> float:
    """The most counted multiplications one inference block reading ``S`` bits holds: ``S^2 / (2 b_w b_a)`` from
    (1, 0)-threading at every S (``threaded_one_zero``), the (d, e) bound where ``threaded``, at most ``block_mults Gamma``."""
    one_zero = S * S / (2 * p["b_w"] * p["b_a"])
    return min(p["block_mults"] * p["Gamma"], one_zero, region_cap(p, S) if threaded else math.inf)


def chord(p: dict, bits: float = math.inf) -> float:
    """Counted multiplications per injected bit when ``bits`` of them go into inference blocks reading ``K`` novel bits
    each: the steepest secant from ``K`` of ``block_cap``, out to ``K + bits``, with (d, e)-threading only up to
    ``thread_bits``. The cap is convex between its kinks, so the secants peak at their ends, and concentrating the bits
    on few blocks is the adversary's best. At the defaults the steepest secant is to ``S_1 = sqrt(2 b_w b_a block_mults
    Gamma)``, on the (1, 0) piece, so d and e below ``thread_bits`` don't move it."""
    K, tb, reach = p["K"], p["thread_bits"], p["K"] + bits
    base = block_cap(p, K, K <= tb)
    s_1 = math.sqrt(2 * p["b_w"] * p["b_a"] * p["block_mults"] * p["Gamma"])
    ends = [(S, True) for S in (2 * p["b_w"] * p["e"], full_block_bits(p), tb, reach) if K < S <= min(tb, reach)]
    ends += [(S, False) for S in (tb, s_1, reach) if tb < reach and K < S <= reach and S >= tb]
    return max(((block_cap(p, S, th) - base) / (S - K) for S, th in ends if S < math.inf), default=0.0)


def windows_bound(p: dict, H: float, state: float, ingress: float, period: float | None) -> float:
    """The fewest counted multiplications any partition of one server's horizon into windows allows; with a wipe
    ``period``, windows start at wipes and span whole periods."""
    if period is None:
        lengths = [10 ** (-3 + 12 * i / 360) for i in range(361)]
    else:
        ks = sorted({max(1, round(10 ** (j / 40))) for j in range(0, 481)})
        lengths = [k * period for k in ks if k == 1 or k * period <= H]
    best = math.inf
    for L in lengths:
        span = min(L, H)
        best = min(best, math.ceil(H / span - 1e-9) * region_cap(p, state + ingress * span))
    return best


def slack(p: dict) -> float:
    """The share of filled memory a proof-of-space certificate leaves unaccounted: rho_PoS^k <= delta certifies rho_PoS,
    and the certificate's floor (blocks rebuildable from public data within the deadline) can be freed at any k."""
    return 1 - p["pous_delta"] ** (1 / p["pous_rounds"]) + p["pous_floor"]


def ingress(p: dict, mech: dict) -> float:
    return p["sigma"] + p["tau"] + p["intake_x"] * p["K"] / p["Gamma"] * p["node_flops"] if mech["net"] else p["nic"]


def audited_residual(p: dict) -> float:
    """r-hat: the GPUs' uncredited share, their capacity above the credited clock, and the host CPUs (0: hosts carry no
    compute)."""
    return min(1.0, p["residual"] + p["host_share"] + p["clock_share"])


def residual_share(p: dict) -> float:
    """r = r-hat + eps_W W_cert/C + eps', the residual inside ``Nci.Guarantees.DampenerRemote``'s bounds,
    ``residualShare W C + εW * W / C + ε``: the audited residual r-hat = 1 - W_cert/C, PoUW's deficit on the certified
    work (a certificate that isn't wrong may credit up to eps_W of its work without its running, so what was certified
    and run is (1 - eps_W) W_cert), and the share whose certificates may be wrong."""
    r_hat = audited_residual(p)
    certified = 1 - r_hat
    return min(1.0, r_hat + p["eps_w"] * certified + p["eps_agg"])


def state_threshold(p: dict, mech: dict) -> float:
    """The server state, in bits, below which memory caps the residual (the window bound's quadratic regime)."""
    return residual_share(p) * p["node_flops"] * p["kappa"] * p["d"] * p["b_w"] * p["b_a"] / (4 * ingress(p, mech))


def evaluate(p: dict, mech: dict, residual_mfu: float = 1.0) -> dict:
    """The terms of eta at ``mech``, each in counted multiplications over the FLOP spent, the residual's capacity
    training at ``residual_mfu`` of peak: 1 for eta, MFU_0 for D under the reference-efficiency assumption."""
    gpm = 2 / p["kappa"]
    H = p["years"] * YEAR
    C = p["nodes"] * p["node_flops"] * H
    s = min(p["unmonitored"], 0.999)
    U = C * s / (1 - s)
    alpha = p["K"] / p["Gamma"]
    rho_T = p["b_a"] / (alpha * lam(p, p["K"]))
    cap_T = C / rho_T

    k = bits = cheat = 0.0
    skipped = min(C, p["identities"] * p["lag_s"] * p["nodes"] * p["node_flops"])
    if mech["sampled"]:
        L = math.log(1 / p["p_evade"])
        q = p["share"] / (p["overhead"] + p["replay"])
        k = p["luck"] * p["identities"] * L / q
        skipped += k * p["step_flop"]
        bits = p["identities"] * L * (1 / q + p["floor_steps"]) * p["step_bits"]
        cheat = min(p["m"] * chord(p, bits) * bits, C / gpm)

    res_ops = residual_share(p) * C + skipped
    sigma_eff = p["sigma"] if mech["net"] else p["nic"]
    link = p["sigma"] + p["tau"] + p["export"] if mech["net"] else p["nic"]
    B_max = 2 * p["b_p"] * res_ops + min(2 * p["b_p"] * U, p["nodes"] * H * link)

    rates = chain_rates(p)
    share = p["kappa"] / p["kappa_chain"] if p["kappa_chain"] > 0 else math.inf

    def declared_at(B: float) -> float:
        return cap_T if share == math.inf else min(cap_T, share * chain_products(rates, alpha * C, B))

    filled = (p["hbm_gb"] + p["host_gb"]) * GB
    mem = filled + p["disk_gb"] * GB
    rate_in = ingress(p, mech)
    help_bits = min(1.0, max(0.0, sigma_eff * p["t_read"] - BLOCK_NAME_BITS) / BLOCK_BITS) * filled
    sl = slack(p)
    wipe_state = (p["unfilled_gb"] + p["disk_gb"] + sl * (p["hbm_gb"] + p["host_gb"])) * GB + help_bits
    envelope = ((p["unfilled_gb"] + p["disk_gb"] + p["hbm_gb"] * (1 - p["f"] * (1 - sl)) + p["host_gb"] * sl) * GB
                + sigma_eff * T_AUDIT)
    per_server = math.inf
    if mech["wipes"]:
        per_server = min(per_server, windows_bound(p, H, min(wipe_state, mem), rate_in, p["wipe_s"]))
    if mech["exhaust"]:
        per_server = min(per_server, windows_bound(p, H, min(envelope, mem), rate_in, None))
    if not (mech["wipes"] or mech["exhaust"]):
        per_server = windows_bound(p, H, mem, rate_in, None)
    region = p["nodes"] * per_server

    def outcome(B: float) -> tuple[float, float, float, float]:
        declared = declared_at(B)
        relay = B / (2 * p["b_p"])
        relay_res = min(res_ops, relay)
        res_mults = residual_mfu * (res_ops - relay_res) / gpm
        unmon = max(0.0, U - (relay - relay_res)) / gpm
        return min(p["m"] * declared + res_mults, p["m"] * region) + unmon, declared, res_mults, unmon

    # past the crossing the declared side is flat at the thread bound, so relaying more only costs
    B_hi = B_max
    if declared_at(B_max) >= cap_T:
        lo, hi = 0.0, B_max
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if declared_at(mid) >= cap_T else (mid, hi)
        B_hi = hi
    # the adversary relays only what pays; PS' is a min of functions affine in the absorbed bits, so its total is
    # concave in B and a ternary search finds the best B
    lo, hi = 0.0, B_hi
    for _ in range(100):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (m1, hi) if outcome(m1)[0] < outcome(m2)[0] else (lo, m2)
    B = max((0.0, (lo + hi) / 2, B_hi), key=lambda b: outcome(b)[0])
    _, declared, res_mults, unmon = outcome(B)

    monitored = min(p["m"] * declared + res_mults, p["m"] * region)
    declared_part = min(p["m"] * declared, monitored)
    res_part = monitored - declared_part
    spent = C + U
    amp = p["g"] * gpm / spent
    terms = {"unmonitored": amp * unmon, "residual": amp * res_part, "declared": amp * declared_part, "cheat": amp * cheat}
    capped = p["m"] * region < p["m"] * declared + res_mults
    largest = max(terms, key=lambda t: terms[t])
    if largest == "declared":
        largest = "declared (thread bound)" if declared >= cap_T * (1 - 1e-9) else "declared (partial-sum law)"
    return {
        "efficiency": sum(terms.values()), **terms, "largest": largest, "memory_caps_residual": capped,
        "rho_T": rho_T, "crossing_bits": B, "wipe_state_bits": wipe_state, "envelope_bits": envelope,
        "state_threshold_bits": state_threshold(p, mech), "cheat_steps": k, "cheat_bits": bits,
    }


def remote_terms(p: dict, r: dict) -> tuple[float, float]:
    """``evaluate``'s terms as the ``s`` and ``x`` of ``Nci.Dampener.remoteDampener mfu s x``: the unmonitored share,
    which trains like the unconstrained developer, and the rest in counted multiplications (so at a = 1/MFU_0,
    ``multiple_bound``'s ``reference_efficiency=False``; the assumption is in ``r``'s residual, ``evaluate`` at
    ``residual_mfu`` = MFU_0), each per the compute amplifier g (which cancels against the unconstrained developer's)."""
    return r["unmonitored"] / p["g"], (r["residual"] + r["declared"] + r["cheat"]) / p["g"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="spec, defaults, tiers, status and results as JSON")
    args = ap.parse_args(argv)
    results = {name: evaluate(DEFAULTS, mech) for name, mech in TIERS.items()}
    if args.json:
        print(json.dumps({"spec": SPEC, "defaults": DEFAULTS, "tiers": TIERS,
                          "status": [{"piece": a, "status": b} for a, b in STATUS], "results": results}, indent=1))
        return 0
    print(f"{'tier':5} {'efficiency':>10} {'unmonitored':>11} {'residual':>9} {'declared':>9} {'cheat':>9}  largest")
    for name, r in results.items():
        print(f"{name:5} {r['efficiency']:10.3g} {r['unmonitored']:11.3g} {r['residual']:9.3g} {r['declared']:9.3g} "
              f"{r['cheat']:9.2g}  {r['largest']}{' (memory caps it)' if r['memory_caps_residual'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
