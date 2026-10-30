# HARDNESS: training hardness into the weights, and exactly what it would let us claim

Daniel, 30 Oct 12:48 PM PDT: "start a swarm that comes up with exactly what the security claim could be here ... what
properties would a trusted retraining process need ... what is the minimal set of assumptions ... be careful ... red team
the ML side ... think like Jared Kaplan would: the core, core, core research questions and experiments. Some can be
determined dynamically by the swarm." Background: Daniel's doc "Hardness training" (quoted in the coordinator's chat,
12:40 PM PDT), the chat that followed, and the lead's state (HASH.md, SILICON.md, CORE.md §2a).

## The idea, as it stands

- **The goal, from Daniel's doc:** resource exhaustion. A datacenter shows that nearly all its compute, HBM and
  network is used by correctly running a declared workload that is resource-transparent, i.e. it cannot be run
  correctly with much less than a stated amount of resources.
- **Today's lead gets that by committing and hashing every k32 word,** with h = 127.5 and 5–80× slowdowns. Hashing is
  there because no standard premise credits the multiply on stock hardware with weights the prover may have chosen.
  Degenerate weights (zero, low rank, sparse, structured) make the multiply cheap, and with private weights nobody can
  tell.
- **Proving a given matrix has no fast exact multiply is open** (matrix rigidity, circuit lower bounds), so checking
  properties of trained weights can't certify hardness. But outputs are checked **bit-exactly**, so only *exact*
  algebraic structure helps a cheater, and randomness nobody chose destroys exact structure.
- **The coordinator's working design ("structural"):**
  - Each large linear computes y = (x+E)·Rᵀ + ((x+E)·V)·Uᵀ (possibly with learned diagonal scalings, D₁RD₂).
  - **R** is public, dense, full-entropy int8, frozen, drawn from a public beacon (or a PRG of one).
  - **U, V** are trained, low rank, and may stay private.
  - **E** is live per-element activation noise from the auditor's coins, kept in the training loop so the model learns
    to be accurate under it, and served without denoising.
  - Only outputs are committed. A sampled ZK proof checks the structure.
  - Hypothesis to test: **security then needs no trust in the training at all.** The developer can train U and V
    adversarially, and training only buys quality.
- **Alternatives to map:**
  - **(B)** free training, verified to stay within ‖W − R‖ ≤ ρ‖R‖ of a public random init;
  - **(C)** fully free training from a public init, with trusted or verified training. That's heuristic: "SGD doesn't
    plant exact structure".

## Core research questions (Kaplan-style: few, sharp, each with the experiment that could kill it)

1. **The claim.** What exactly can be proved?
   - Statement shape: "if verification accepts, the prover spent ≥ (1 − ε)·M MAC-units on the R-products in the
     window, except with probability δ".
   - The minimal premise set, ranked. Candidates:
     - P1, random-operand int8 matmul hardness *with preprocessing of R*;
     - P2, no profitable generic fast matmul (Strassen and kin) on int8 tensor cores;
     - P3, SHA-512 commitments and live coins;
     - P4, sampled-proof soundness.
   - Is anything about training needed? Which properties must a retraining process have under (A), (B) and (C)?
   - What the claim does **not** cover: attention, nonlinearities, other devices on the node.
2. **P1 itself.** Is "computing (x+E)Rᵀ exactly, for R uniformly random int8 known in advance and E live, costs ≥
   (1 − δ) of the int8 MMA's work on this chip" true, minimal and standard enough?
   - Attacks: preprocessing (Four Russians, lookup tables, matrix-vector-with-preprocessing results), bit-slicing, 2:4
     sparse tensor cores, Strassen/Winograd at int8 via wider types, FP8/FP16 emulation, the host CPUs and other
     devices, E's small alphabet.
   - Price every attack in W1 and in time.
3. **The one ML number.** What is the **effective-compute penalty** of training under the structural constraint at a
   fixed hardness level, against unconstrained training, as a function of model scale, r/n, the noise scale σ_E and the
   hardened fraction of FLOPs?
   - Does it shrink, hold or grow with scale?
   - Does distillation from a strong teacher close it?
   - Kill condition: a penalty that grows with scale, or > 3× at the largest scale we can run.
4. **How the ML side breaks.**
   - Routing around R: forced but useless work is fine for exhaustion, so how does it bear on the claim?
   - Collapse through nonlinearities, input selection, adapters absorbing everything, the share of attention, and the
     interplay with quantization.
   - Deployment realism: who draws R and when, the distillation pipeline, switching costs, serving speed.
5. **The protocol.** What's committed, live-noise bandwidth, the ZK structure check on our stack (C-Flock sampled
   proofs), h, T1/T2/T3, and how this compares with the lead, R80X, D-OUT and silicon.

## Lanes (notes: `lanes/compute-research/<ts>-report-hardness-<lane>.md`)

| lane | question | first deliverable |
|---|---|---|
| HT-CLAIM | Q1: the exact claim, threat model, minimal premises, what a retraining process must satisfy under (A)/(B)/(C) | a claim statement (Lean-like), a premise inventory with ranks, a "does training need trust?" verdict per design |
| HT-PREMISE | Q2: P1 and P2, attacked | every known route priced; P1 stated minimally; literature with read/recalled marks |
| HT-ML | Q3: the penalty, and its scaling | a literature map, then the smallest decisive experiment run, then a scaling plan with a GPU budget |
| HT-REDML | Q4: how it breaks, and how it would actually be deployed | a ranked break list with a test for each; a deployment walkthrough |
| HT-PROTO | Q5: the protocol on our stack | commitments, proofs, h, T1/T2/T3, a comparison table |

**Rules:**
- Everything is marked checked, measured, argued or recalled.
- Lanes adjust their own questions dynamically and say so in their notes.
- Spend beyond existing budget lines goes to the coordinator, with a number, and from there to Daniel.
- **Times:** preliminary notes by 00:00Z Sat (5 PM PDT), and the coordinator's synthesis to Daniel by 01:00Z
  (6 PM PDT). ML experiments may report a first signal by then and continue.

## Addendum: does the training process need an assumption? (Daniel, 1:02 PM PDT)

Daniel: "there would have to be some assumption about the training process... it would have to preserve the hardness
injected by the randomness." The coordinator's answer, for HT-CLAIM to confirm or break:

- **Something must preserve the hardness.** Either it is assumed of the training process (design C, heuristic), or the
  circuit enforces it for every value of the trainable parameters (designs A and B), in which case it is a reduction,
  not an assumption about training.
- **The reduction for A:**
  - Any algorithm that multiplies by W = R + UVᵀ yields one for R that costs only 2nr more a row: compute
    ((x+E)V)Uᵀ and subtract it.
  - This holds for every U, V, even ones chosen adversarially after seeing R. So P1-with-preprocessing covers
    adaptively trained adapters, and training can do anything to U and V.
  - It needs the circuit to keep the two terms exact and separate: no requantizing R + UVᵀ into int8 (rounding and
    clipping would break it).
- **Every extra knob training gets needs its own reduction.**
  - Diagonal scalings D₁RD₂ with zero entries sparsify R.
  - Output-side D₁ reduces exactly if its entries are nonzero, since the division is exact.
  - Input-side D₂ changes R's distribution (column k is uniform on d_k·[−128, 127]), so P1 must hold for that class.
  - Masks, norms and anything that zeroes parts of R need the same treatment.
  - The circuit enforces nonzero entries in ZK, or the credited M counts only the nonzero part.
- **Design B (a ball of radius ρ) needs more than A:** an assumption that no matrix within ρ of a random R has a fast
  exact multiply.
  - That's a rigidity-style statement. Random matrices are rigid w.h.p. for rank reduction (Valiant), but not
    necessarily for every hardware-fast form (e.g. 2:4 sparsity: zeroing the two smallest of every four costs a
    relative norm of roughly 0.35–0.45, unchecked).
  - It's still about the ball, not about the training.
- **What A costs instead: quality,** since training only gets rank-r freedom. That's HT-ML's question.

## Addendum: iterated injection (Daniel, 1:04 PM PDT), design D

Daniel: "many rounds of injecting noise and then a little training; perhaps a better balance of security and usability,
and easier to tune." The coordinator's reading, which lane HT-ITER tests:

- **Only the last injection, and what happens after it, carries security.** Everything before the final public draw is
  absorbed into the developer's matrix A, whatever it is.
- **Earlier rounds buy usability:** robustness to the injection. They can even use the developer's own simulated draws
  from the same distribution.
- **Three regimes for the last round:**
  - **D0: commit A, draw R publicly after, serve W = A + R with no training after.**
    - Premise: *smoothed* hardness. For every A fixed before R is drawn, multiplying exactly by A + R costs about a full
      unit. No assumption about training, and no structural constraint on A at all.
    - A counting argument suggests most of any shifted cube is hard, but that's in a circuit model. Pricing it on real
      MMA hardware is HT-PREMISE's work.
    - R can be small in magnitude (e.g. a few bits on int8 weights, with headroom reserved), so a robustly trained A may
      lose little quality.
  - **D1: a little training after the last draw.** The ball premise of design B, with ρ the post-draw movement.
  - **D2: constrained training after the last draw.** Design A's low-rank reduction.
- **The catch: alphabets.**
  - The cheater can make x and A cheap, so the forced work is about E·Rᵀ.
  - E's and R's alphabets must not fit a cheaper exact route than the int8/FP8 unit: b1 bit-slicing at 8×, FP4 on
    sm_120, IDP4A. ±1 × ±1 is cheap; ±1 × int8 costs about 1 unit with b1 at 8×.
  - Pricing that is HT-PREMISE's work. Magnitude against quality is HT-ITER's and HT-ML's.
- **Redraws:** drawing R again after a bad draw lets the developer pick the best of k. That costs log k bits, which is
  negligible for counting, but it's a rule question. Prefer no redraw.

**Correction to design D (Daniel, 1:31 PM PDT: "isn't the total entropy added what matters?").** "Only the last
injection carries security" holds only when training between injections is unconstrained. The developer knows each
R_t and can undo it. With constrained training between rounds, earlier randomness partly survives. The measure is not
the total injected entropy:

- **The right quantity:** the final W is F(R_1, …, R_T) for a developer-chosen F within the training constraints.
  Hardness by counting needs the min-entropy of W, under the developer's best F, to exceed log |cheap matrices| plus
  the security bits.
- **That min-entropy is capped three ways:**
  - **By the format:** at most 8 bits an entry for int8, however many rounds.
  - **By how independent noise adds:** T rounds of ±1 sum to about ½·log₂T bits an entry in magnitude spread, not T
    bits.
  - **By the developer's adaptive training:** it acts after every round, so its total cancelling budget grows like T
    while the accumulated noise grows like √T.
- **So many small rounds can favour the adversary** unless the per-round budget shrinks like 1/√T. Low-rank rounds
  compose to rank T·r, and the reduction goes vacuous as T·r → n.
- **The strongest security per unit of quality is still one large final draw after the commitment,** with little or no
  training after it. Iteration buys robustness, plus whatever survives constrained rounds.

HT-ITER should model this: the surviving min-entropy against T, ρ_t and the alphabet.
