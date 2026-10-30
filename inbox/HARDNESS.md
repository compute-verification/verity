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
