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

- **HT-PREMISE** (note:compute-research/20261030T2141Z-report-hardness-premise; exact integer routes on CPU, costs in
  W1, nothing on a GPU; art:b3c905d9…):
  - **P1 is false at b = 1:** R's low bit through a b1 MMA, plus Strassen on the rest, costs 0.957–0.993 at
    K ≥ 8,192 on the H200.
  - **It is false at w = 8 for any b:** Strassen mod 2⁸ costs about 0.77.
  - **b = 2, w = 16 holds against every route found, but by 1.8% at K 53k and 11.6% at K ≤ 8,192 on the H200** (0.9% and
    6.4% on sm_120). The margin comes *only* from additions and accumulator drains; counting tensor work alone, a
    depth-6 Strassen costs 0.878.
    - So M3 must price additions, or cap the scheme depth at 4.
    - Recommend b = 2, w = 16, and K ≤ 8,192 per committed product.
  - **P2 as worded is false;** it is part of P1.
  - **P1, minimal:** per entry; average over R *and* E; worst case over x and the program; unbounded preprocessing; a
    2⁻ʷ guessing term; stated on the seeded SHA-512 expansion. Tables lose at least 5×.
  - **P1 splits** into a finite check over bilinear schedules with additions priced (extending `NaiveStrassen.lean`) and a
    conjecture for every other kind of program.
  - **Correction to HT-REDML and to the coordinator:** the H200 runs b1 at 8× s8 (CORE §2a, measured: b1 wgmma).
    A-LSB is therefore credited at most **0.38 at b = 2 and about 0.8–0.85 at b = 4** on the H200, not a full unit.
  - Upper bounds only: the flip-graph sweep and the addition-priced schedule search didn't finish.

- **HT-PREMISE v2** (note:compute-research/20261030T2300Z-report-hardness-premise-v2; recorded runs on cpu-6 and cpu-11;
  addition-priced schedule search to depth 6–8, bounds agree, witnesses exact):
  - **A-LSB is capped below 0.9 at every b from 2 to 5, on both chips.** For b ≥ 4 the cheapest route is Strassen on s8
    over narrow operands: executed, 0.761 at K 2,048 and 0.642 at K 8,192. A credit of 0.9 needs K ≤ 256.
  - **D0's smoothed premise fails:** the developer picks A = 0, which costs at most 0.753.
  - **Design A′ (full-entropy int8 R, b_E = 2, w = 16) holds at K ≤ 4,096,** with a margin of at least 8.5% on the H200
    and 6.0% on sm_120. At K ≤ 8,192 the margin is only 0.84% and 0.17%, and from K 16,384 it breaks in the fast
    reading.
  - **The thin long-K margin is structural:** Strassen's tensor work is below 1 from depth 5, and additions shrink like
    1/K. So every Lean statement of P1 must price additions.
  - **The K ≤ 4,096 cap costs** ×1.09 the committed outputs on Llama-3.2-1B, ×2.49 on 70B and ×4.94 on 405B.
  - D₁RD₂ needs odd entries, not just nonzero ones.
  - **The principle (coordinator's reading):** the forced random operand must be *full-width* in the MMA format, dense,
    with wide outputs and short K. Any narrowness (b1-sized E, a few random low bits, a small-magnitude R, a grid in a
    ball) lets Strassen or bit-slicing run inside the format. That leaves A′, whose problem is utility, and E (natural
    weights), which HT-PUBLIC is testing.

- **HT-PREMISE v3** (note:compute-research/20261030T2342Z-report-hardness-premise-v3; 105 natural matrices, 840
  schedules run exactly in int64; recorded runs on cpu-11):
  - **Natural weights are outside Hard_X at 5%.** Strassen in s8 beats the honest kernel on Qwen2.5-7B (0.963) and
    Llama-3.1-8B (0.968). Every down_proj is outside (0.878–0.928 on the 7B and 8B), and 42 of 105 matrices cost less
    than 1.
  - **The cause:** natural weights overflow s8 under Strassen's sums at a median of 0.40% of entries, against 17.7% for
    uniform int8. That's why uniform R (A′) still holds and natural weights don't.
  - **This overturns HT-PUBLIC's "already hard"**, which covered only low-rank, sparse and small-alphabet splits.
  - **Caveat:** the price of an overflow correction is counted, not timed. At 2× that price the 7B and 8B sit at
    1.045–1.054, on the 5% line, but down_proj stays at 0.91–0.93. A GPU timing settles it.
  - **Follow-ups (round 4):** does the lead's per-k32-word commitment block Strassen (demo-critical)? Does D-OUT (outputs
    only, over public natural weights) have the same hole?

- **HT-PREMISE v4** (note:compute-research/20261031T0022Z-report-hardness-premise-v4; executed on natural Llama-3.2-1B
  layers, run r20261031-000614-075e):
  - **The lead is not exposed to Strassen.** Its own premises (TileGrain, OneInstructionMinimal, H32) charge every
    committed k32 word a register write. Every schedule that reproduces every committed word costs 1.93–3.12× stock,
    and splitting only M and N saves nothing (argued).
  - **D-OUT is exposed if an overflow correction costs about one write:** a gap of 1.84% at 1 write (capacity
    99.15–99.16%), and 0.21% at 2 writes (99.2–99.45% holds). The crossover is about 1.6 writes on the H200 and 1.9 on
    sm_120.
  - The GPU timing is running on vy-nebius-1 (sm_120).

- **HT-PREMISE v4, timed** (sm_120, vy-nebius-1): an overflow correction costs 0.84 writes (2-bit forms) and 2.36 (int8
  forms).
  - **D-OUT is exposed on sm_120** (a 2.18% gap); the H200 is ambiguous, with the timing command in HASH.md.
  - **E (natural weights) on sm_120** sits on the 5% line: Qwen2.5-7B 1.048, Llama-3.1-8B 1.060. But 10 of 105 matrices
    are below 1, so a per-matrix cheater runs the 7B at 0.987. That's a gap above 1%, so **E fails**.
  - The lead's verdict is unchanged.

- **HT-SCALE Tier 1** (note:compute-research/20261031T1054Z-report-hardness-scale-125m; GPU, measured; 20M, 60M and
  125M, D = 20N):
  - **At κ_lin = 1/3 the penalty doesn't shrink with N.** Reservoir layers go 1.19 → 1.40 → 1.39 (P_env) and 1.11 → 1.20
    → 1.34 (P_size). Per matrix on half the MLPs goes 1.21 → 1.37 → 1.32 and 1.13 → 1.20 → 1.29. The per-feature
    arrangement is worst (1.34× at s50, 2.50× at s100).
  - **u ≤ 0 at 125M:** the credited random compute is dead weight or worse in every arrangement.
  - **350M is running.** The decision rule was written before any result: kill if P_env is above 1.49× (reservoir) or
    1.41× (matrix). The node is needed until about 03:00Z Sun.
  - **The coordinator's reading for the hybrid:** κ_lin = 1/3 at about 1.35× cuts the lead's hashed share by only a third,
    so the hybrid saves little unless a far higher κ comes at a far lower P. No trend seen so far suggests it will.

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

## Coordinator's smell check and redirects (21:35Z Fri, after HT-ML's, HT-REDML's and HT-CLAIM's interim notes)

1. **HT-ML's scaling plan scales an insecure arm.** Its reference arm is B at ρ = 0.25, but HT-REDML and HT-CLAIM show
   the ball contains cheap matrices from ρ ≈ 0.03–0.06 (the low bits dropped, a 3-bit grid; Blackwell breaks at 6.3%).
   *Redirect:* no GPU spend on B or A scaling. A is also near-dead on utility: GaLore and ReLoRA's "LoRA from a random
   init" are already 4–9× over the 3× kill line, and A at the rank the ε budget allows is +0.68 nats on the toy.
   The real-scale test moves to **A-LSB** (W = 2ᵇ·H + L, with L public random low bits): a QAT pass on an existing open
   model, which is cheap.
2. **(Resolved 21:45Z by HT-PREMISE: the H200 has b1 at 8×, so b = 2 earns 0.38 and b = 4 about 0.8.) A-LSB's "a full unit on the H200 at b = 2" rested on a recalled fact:** that sm_90a has no fast b1/s4 tensor path.
   It defines the credit, so HT-PREMISE must *measure* mma.sync .b1 and .s4 (and any packing route) on an H200 before
   anyone states the credit.
3. **"Costs nothing" is a toy** (d 256, TinyStories, CPU). Label it that way until the real-scale QAT run.
4. **Scope:** the designs force tensor work on the hardened linears only. Attention is 28–60% of MACs at 128K, and
   there's no HBM or network claim. Every statement must carry κ. "Nearly all compute, HBM and network" needs a
   mechanism per resource.
5. **Coins are the binding deployment cost:** 1.6–15 GB/s a GPU, up to about 2 Tb/s into a node. They can't be seeded
   under today's egress-only certifier. The lead shares this.
6. **Fleet format:** int8 is 1.4–3.4% of FP8's rate on B300 and Rubin. A credit in int8 units covers almost nothing of
   those chips, so an FP8 or FP4 analogue of A-LSB is required beyond Hopper.

## Daniel's rulings (30 Oct 4:41 PM PDT)

- **Public randomness:** a public random matrix or set of low bits drawn from a live coin after registration is fine.
  It is a verifier draw, and so public under "everything zero-knowledge".
- **GPUs:** "Surely we have GPUs. You can just use them." The node and budget line are asked of infra on INF-393: an
  8-GPU node for about 48 h, about $700.

## §prelim: the coordinator's synthesis (00:15Z Sat 31 Oct), module by module

**The answer.** Trained hardness does not replace the lead, but it can *shrink* it.
- In the one design whose premise survived attack (A′), a share κ of the linears' MACs is a frozen uniform-int8 random
  product. That share is credited at about 0.9 hash ANDs a MAC, against the lead's 127.5.
- The trained remainder (1 − κ) is uncredited by trained hardness. So the **hybrid** is: credit the κ share by trained
  hardness, and run only the trained remainder through the lead's noised, hashed path.
- The lead's hashing overhead then applies to (1 − κ) of the MACs instead of all of them.
- The price is A′'s quality penalty P(κ, N). Whether P shrinks with N decides the program. HT-SCALE measures it on
  vy-nebius-1 now.

**Module by module:**
- **M1 (beacon):** R comes from one live coin after the architecture is registered, with no redraw. Daniel rules that a
  public R is a verifier draw (4:41 PM PDT).
- **M2 (hardening):**
  - **A′ holds,** preserving hardness for every trained U, V via the reduction, given rules C1–C4.
  - **Dead:** B, the ball (cheap matrices from ρ ≈ 0.03); A-LSB (< 0.9 at every b); D0 with a small R (≤ 0.76);
    A-iterated (adds nothing); E with natural weights (Strassen in s8 at 0.963–0.968 on 7B and 8B, and down_proj at
    0.88–0.93).
  - E also needs public data and a full replay, because planned jumps survive spot checks.
- **M3 (premise):** P1 holds for uniform int8 R with b_E = 2, w = 16 and K ≤ 4,096, by a margin of at least 8.5% on the
  H200 and 6.0% on sm_120.
  - The margin comes only from additions, so every Lean statement must price additions.
  - b = 1 and w = 8 are false.
  - P1 splits into a finite schedule check (extending NaiveStrassen.lean) and a conjecture for every other kind of
    program.
- **M4 (operand entropy):** x is committed before E. E is 2-bit (k ≈ 1–2), which costs 1.03× in quality on the toy; the
  knee is at k ≈ 4–8.
- **M5 (utility):** the evidence conflicts.
  - HT-ML's CPU toys (1.3–14.7M parameters, 20M tokens) show A's penalty *falling* with width: A 1/8 goes 1.92×, 1.42×,
    1.32× at κ ≈ 0.44.
  - GaLore and ReLoRA (LoRA from a random init) read 3–6× at 130M and up, and growing.
  - R adds little: low rank alone is 1.34×.
  - HT-SCALE also tests reservoir layers and per-feature splits.
- **M6 and M7 (integrity, provenance):** sampled proofs, about 2M ANDs a unit, 2,773 draws at ε 1% and δ 2⁻⁴⁰.
  Provenance is structural for A′, since R is a public input.
- **M8 (coverage):**
  - Only the hardened linears are covered. Attention is 28–60% of MACs at 128K, and there's no HBM or network claim.
  - The K ≤ 4,096 cap costs ×1.09–4.94 in committed outputs.
  - Coins are 1.6–15 GB/s a GPU, like the lead's.
- **M9 (hardware):** the H200 has b1 at 8× s8. int8 is 1.4–3.4% of FP8 on B300 and Rubin, so beyond Hopper A′ needs an
  FP8 analogue.

**Smells still open:**
1. **D-OUT** (HASH, outputs only over public natural weights) likely has the Strassen hole, which would put its
   "99.2–99.45% capacity" in doubt. HT-PREMISE round 4 checks it by 01:45Z.
2. **(Resolved by v4: the lead is not exposed; 1.93–3.12× stock.)** The lead commits every k32 word, which should
   block Strassen.
3. The overflow-correction price behind "natural weights are not hard" is counted, not timed. It will be timed on
   vy-nebius-1.
4. All utility numbers so far are toys.

**Running:** HT-SCALE (GPU, first signal Sat 12:00Z, final Sun 22:00Z); HT-PREMISE round 4 (01:45Z); HT-ML's last CPU
waves (to 03:00Z).
**Next for the swarm:** price the hybrid, i.e. total overhead as a function of κ, P(κ, N) and the lead's h on the
uncredited remainder.
