# HARDNESS: training hardness into the weights, and exactly what it would let us claim

Daniel, 30 Oct 12:48 PM PDT: "start a swarm that comes up with exactly what the security claim could be here ... what
properties would a trusted retraining process need ... the minimal set of assumptions ... be careful ... red team the ML
side ... think like Jared Kaplan would." Then, at 1:39 PM PDT: "do you and the swarm understand how this whole research
program fits together? ... show what the overall architecture is and how modular it is," as noising and denoising did for
PoUW. This file is that architecture. Each lane owns a module and states its findings as that module's interface
property. Background: Daniel's doc "Hardness training", the chat of 30 Oct 12:40–1:40 PM PDT, and CORE.md §2a.

## The goal and the one-line idea

- **Resource exhaustion:** a datacenter shows that nearly all its compute, HBM and network is used by correctly running
  a declared workload that is resource-transparent, i.e. it cannot be run correctly with much less than a stated
  amount of resources.
- **The lead does this today by hashing every k32 word:** h = 127.5, with 5–80× slowdowns. Hashing is needed because
  weights the prover chose may hide a cheap exact multiply (zero, low rank, sparse, structured), and proving a given
  matrix has none is open (rigidity, circuit lower bounds).
- **The idea:** outputs are checked bit-exactly, so only *exact* structure helps a cheater, and randomness nobody chose
  destroys exact structure. So make the served weights depend on public randomness drawn after the developer's
  choices. Then committed outputs alone force the multiply, and the per-word hashing goes.

## The architecture: nine modules, each with one interface property

Notation:
- the served layer computes Y = Z·Wᵀ, where Z = x + E;
- the honest work is M MAC-units on chip X;
- costs are W1 units (CORE.md §2a).

| # | module | interface property it must provide | what exists | owner |
|---|---|---|---|---|
| M1 | **Beacon** | Draws are unpredictable before their time, unbiasable (no grinding or redraws), and public after. Each draw is scheduled after the commitments it must follow. | live coins (`verity.ml.randomness`), the certifier | (exists) |
| M2 | **Hardening process** (the variants below) | *Hardness-preserving:* for every developer strategy the process allows, Pr over the beacon [W ∉ Hard_X(δ)] ≤ η. The process also states its choice C, in bits of selection after any draw, and whether any steering is left. | none | HT-CLAIM (statement), HT-ITER (D), HT-PUBLIC (E) |
| M3 | **Hardness premise** | Defines Hard_X(δ): every exact algorithm for Z ↦ Z·Wᵀ, with any preprocessing of W, costs ≥ (1−δ)·M on chip X for Z from M4's distribution. Plus closure lemmas: R+low-rank, D₁RD₂ with nonzero entries, A+R for A fixed before R (smoothed). | SAK family (rank 4), SI-RED's harness | HT-PREMISE |
| M4 | **Operand entropy** | Z can't be made cheap: x is committed *before* E is released (HT-PROTO: otherwise x = −E is free), and E is live, per element, with an alphabet of a bits. a is chosen so that no format route (b1, FP4, IDP4A, XNOR-popcount) beats the unit for M3's alphabet of W. | the lead's input commitment, the per-call coin release | HT-PROTO, HT-PREMISE (pricing) |
| M5 | **Utility** | The model under M2's constraints and M4's noise is within Δ of the unconstrained model: an effective-compute multiplier κ(N), as a function of scale N. T3 is the perplexity gap. | none | HT-ML (A, B), HT-ITER (D) |
| M6 | **Serving integrity** | If accepted, ≥ (1−ε) of the committed output entries are exactly Z·Wᵀ at confidence δ_s. It is per entry, because sampling certifies only part of the outputs. | sampled proofs, C-Flock, commitments | HT-PROTO |
| M7 | **Provenance** | The W used in serving is the one M2's process produced. For A and D it is structural: a public R in the proven circuit, with U, V or A committed. For E it is a separate audit of the training, e.g. sampled re-execution of committed checkpoints on bit-exact kernels. | (A, D) the proof's public inputs; (E) none | HT-PROTO, HT-PUBLIC |
| M8 | **Coverage and accounting** | The forced work, against what isn't credited: attention, the low-rank adapters (HT-PROTO: count them as coverage), nonlinearities, host CPUs, HBM and network. It gives T1 against stock, T2 and h. | CORE.md §2a, PoUS for HBM, network accounting | HT-PROTO, HT-REDML |
| M9 | **Hardware cost model** | For each chip: the formats and their rates (int8, FP8, FP4, b1, IDP4A), the ALU and the host CPUs, and the per-register prices that M3's routes are priced in. | the cards, `verity.hardware`, the H200 sweep | HT-PREMISE |

**The composition (HT-CLAIM states it precisely and checks every arrow).** Suppose:
- M1 holds;
- M2 is hardness-preserving at η under M3's premise for chip X;
- M4 gives Z the distribution M3 covers;
- M6 holds at (ε, δ_s);
- M7 holds.

Then, if verification accepts, the window's work is
  ≥ (1 − ε)·(1 − δ)·M − (M8's uncredited coverage),
except with probability η + δ_s + negl, plus the crypto terms. M5 is outside the security argument: it only decides
whether anyone would serve the model.

**What the modularity buys:**
- **M2's variants are interchangeable behind one property.** A new variant needs only its own proof of
  hardness-preservation and its η, C and M5 numbers.
- **M3 is the only hardness premise,** and M5 is the only ML question.
- **M1, M4, M6 and M7 mostly reuse the proof service.**

**What decides the program:**
- **(i) M3:** is the premise true on real chips, and at what δ?
- **(ii) M5:** κ(N), and does it shrink with scale?
- **(iii) M2-E's cheap test:** are natural trained weights already far from cheap?

Everything else is engineering on parts we have.

## M2's variants (each must state its proof of hardness-preservation, η, C and its M5 cost)

- **A: structural.** W = R + UVᵀ, with R public, frozen and drawn after the architecture is registered, and U, V
  trained at rank r.
  - Preservation is a reduction: multiplying by R+UVᵀ costs ≥ cost(R) − 2nr a row, for *every* U, V, even adversarial
    ones. η is M3's η for R, and C is unbounded but harmless.
  - The circuit must keep the terms separate and exact; requantizing R+UVᵀ breaks the reduction.
  - Every extra knob needs its own closure lemma. D₁RD₂ needs nonzero entries, enforced in ZK or else uncredited.
- **A-iterated (Daniel 1:39 PM PDT):** high-dimensional noise each round, then a low-dimensional correction.
  - The corrections compose to rank T·r, so the reduction costs 2nTr a row and goes vacuous as T·r → n.
  - The noise sums to one random matrix: about ½·log₂T bits an entry for ±1 noise, capped by the format.
- **B: a ball around a random init.** ‖W − R‖ ≤ ρ‖R‖, with free training inside it.
  - It needs a rigidity-style M3: no matrix within ρ of a random R is cheap.
  - Hardware-fast forms may sit close: 2:4 sparsity at a relative norm of about 0.35–0.45 (unchecked).
- **D: iterated injection, with the final draw after the commitment.**
  - **D0** commits A, then draws R, then serves A+R with no training after.
    - It needs a smoothed M3 (for every A fixed before R, A+R is hard). C = 0 and there's no constraint on A.
    - R can be small in magnitude, but the alphabets of E and R must defeat the format routes (M4, M9).
  - **D1** allows a little training after the final draw, which is B's premise.
  - **D2** allows constrained training after it, which is A's reduction.
  - Earlier rounds buy robustness, plus whatever randomness survives constrained rounds.
  - The measure is the min-entropy of the final W under the developer's best strategy, not the total injected entropy:
    - it is capped by the format;
    - independent noise adds like √T;
    - the adversary's cancelling budget adds like T.
- **E: public deterministic training (Daniel 1:38 PM PDT).**
  - W = Train(public data, public code and hyperparameters; beacon seed): one fixed process with no adversary inside.
  - M3 becomes nothing-up-my-sleeve: "W from this distribution is hard w.h.p.", falsifiable by anyone.
  - Choice:
    - selection costs C bits of the union bound, and committing everything before the seed makes C = 0;
    - steering through private data is harder to bound, so the data must be public or committed before the seed, and
      the premise must hold for every fixed dataset.
  - Weight decay erases the init's randomness after int8 rounding.
  - M7 must audit the training itself (see the proof-of-learning literature and its spoofing attacks).
  - **The cheapest first experiment:** on int8 open weights (Llama- or Qwen-class), find the best "low-rank +
    small-alphabet or sparse residual" split and price its exact multiply on the H200 and sm_120.
- **C: free training from a public init.** It needs a trusted-training premise about an adversary's choices. It's the
  weakest variant, kept only as the baseline.

## Facts established so far (each with its source)

- **HT-PROTO** (note:compute-research/20261030T2025Z-report-hardness-proto, estimates):
  - x must be committed before E.
  - R should be a public input: as circuit constants it breaks C-Flock's batching.
  - A unit is about 2M ANDs at K 2,048.
  - The audit is 2,773 draws at ε 1% and δ 2⁻⁴⁰: about 131 s of proving and 74 CPU-s of verifying.
  - Coins: 28.7 KB × b a token, against the lead's 2.25 MB.
  - h ≈ 0.88 (against 127.5); T1 ≈ 0.24%; T2 ≈ 1.2–2.5×.
  - It recommends b = 2 for the noise, since 1-bit noise only ties bit-slicing.

- **HT-CLAIM** (note:compute-research/20261030T2055Z-report-hardness-claim, CPU-checked route searches,
  art:65c8850b…):
  - Design A as worded can't be claimed. It needs four circuit rules (= HT-REDML's C1–C4):
    - commit x before E;
    - commit the R-product *alone*: with the summed output, rank r cancels r columns of R, 3.1% over-credit at
      r = 64 (the reduction's slack);
    - add E exactly, right before R;
    - draw R once, after the architecture is registered, with no redraw.
  - With these (A′), training needs no trust.
  - P1 is the one new premise (rank 4bU, per entry, unbounded preprocessing). P2 is a special case of it.
  - B's premise is false at small ρ: 12.5% over-credit at ρ = 0.96% with 1-bit E. With 2-bit E the H200 holds to
    ρ ≈ 14.6%, but Blackwell breaks at ρ ≈ 6.3%.
  - A-LSB (W = 2^q·H + L, with L public random low bits) preserves hardness for every H, at credit c_q < 1 (≈ 0.875
    at q = 4, b = 2).
  - Use 2-bit E: at 1-bit, P1 has zero margin.
  - Coverage κ is 11–85% of forward MACs at 32K. There's no HBM or network claim, and the work is certified, not its
    usefulness.

## Lanes (notes: `lanes/compute-research/<ts>-report-hardness-<lane>.md`)

| lane | modules | first deliverable |
|---|---|---|
| HT-CLAIM | the composition, M2's property, M1 | the composed claim with every arrow checked; the premise inventory, ranked; a verdict per M2 variant |
| HT-PREMISE | M3, M9 | every route priced on the H200 and sm_120; M3 stated minimally, with its closure lemmas (low rank, diagonal, smoothed) |
| HT-ML | M5 for A and B | the literature map; the smallest decisive run; a scaling plan with a GPU budget |
| HT-ITER | M2-D, M5-D | D0's premise; the surviving min-entropy against T, ρ_t and the alphabet; a Pareto table |
| HT-PUBLIC | M2-E, M7-E | E's claim and choice accounting; the natural-hardness experiment on open weights; verifying the training |
| HT-REDML | attacks across M2, M4, M5 and M8 | a ranked break list with tests; a deployment walkthrough |
| HT-PROTO | M4, M6, M7, M8 | done (interim): the protocol, the numbers, the comparison table |

**Rules:**
- Mark everything checked, measured, argued or recalled.
- Lanes re-plan their own questions dynamically and say so.
- Spend beyond existing budget lines goes to the coordinator, and from there to Daniel.
- **Times:** preliminary notes by 00:00Z Sat (5 PM PDT), and the coordinator's synthesis to Daniel by 01:00Z (6 PM PDT),
  laid out on this architecture.
