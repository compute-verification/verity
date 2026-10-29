# PoUW core specification and power ranking (Daniel, Thu 29 Oct 00:02Z = Wed 5:02 PM PDT)

This file supersedes BRIEF.md for PoUW. The golden-demo swarm's other layers (sampling, egress, proving cost, exfiltration)
are outside this core spec. Terms: a **dot product** is one output element of a matmul. A **proof block** is a region C-Flock
proves; don't coin other names.

## 1. The problem (minimal)

- **One matmul:** O = A·Bᵀ, with A m×K and B n×K, in the served format (int8 for the demo; FP8 as an option), G = mnK
  multiply-adds, on **one H200 (sm_90)**.
- **Commitment and noise:** A and B are committed before the call's noise. Every call gets fresh noise from live coins and is
  atomic: no noise is reused across calls (Daniel, 4:30 PM PDT).
- **Words and checking:** the prover commits words W while it computes, and the check verifies every committed word. Sampling
  the words, outsourcing and egress, proving cost and exfiltration are layered on later. Here every word is checked and the
  adversary has no outside machine.
- **The honest prover H** is a fixed public program that outputs O′ (what it serves) and W, in time t_H on the H200. H may use
  every pipe. **Stock** is the plain matmul in the serving stack, taking t_S.
- **The adversary** is any program on the H200 that outputs W correctly. It is given A and B, and any advice computed before
  the noise that fits in device memory.

## 2a. T1 as Daniel ruled it (Thu 29 Oct 02:01Z = Wed 7:01 PM PDT). Supersedes T1 in §2 and the 01:40Z defaults.

- **The honest program is the best possible honest program** (for now). This is decision 74's reading: our kernel's
  inefficiency is not gap.
- **Hashing is credited work.** It rests on ShaSpeed: no program computes SHA-512 faster than the best implementation, which
  the honest program is taken to be.
- **The gap is additive against stock:** cost(H) − min over adversaries of cost(adv) ≤ 0.01 · cost(stock), where stock is the
  plain matmul. A relative gap would let a 1000× hashing overhead hide a fully skippable matmul.
- **Cost is work, not wall time.** It is priced work summed over the card's resources, in MH100 / W1 units (one dense int8
  MAC = 1 unit; every instruction at its measured price per register written). These are the rank bound's own units. In
  wall time, a matmul hidden under hashing costs nothing, and the trivial design would pass. Co-issue lowers time but not
  work. Report the time reading beside it, but don't judge on it.
- **What follows for #1, NCP + the rank bound (coordinator's estimates; CORE-TIME and CORE-NCP to confirm):**
  - **The final word of each dot product** is unforced in Theorem D's model, which makes every function of data fixed
    before the noise free, AB included. That is 32 units a dot product against stock's K: 32/K of stock, so 1.56% at
    K 2,048 and 0.39% at K 8,192, about 1.24% over Llama-3.2-1B's linears. **Fix (research):** each A·B is atomic
    (Daniel, 4:49 PM PDT), so charge work done before the noise inside the call. Restate Theorem D with the free values
    being only linear functions of the call's inputs and noise. The final word is then a target, and that gap closes.
  - **Forming the noised operands** is linear, so it is free in the rank model and not credited. That is NCP's certificate's
    8/n, plus 8/m for the per-call weight noise: about 0.5% at T 8,192, and 12.5% or more at T 512 (m = 64), which fails.
    **Fix (research):** a bound that charges forming, with additions priced as MH100 prices them; or a design whose
    forming is credited or negligible.
  - **Hashing and coins** are credited under ShaSpeed, with the honest program taken as best: gap 0.
  - **Co-issue** is 0 in work units.

## 2. Targets

- **T1, hard: the gap is at most 1%.** min over adversaries of t_adv ≥ 0.99·t_H.
  - This is H200 time, not summed priced work. A work bound counts only together with a pipe-parallel argument that turns it
    into time. The MH100 docstrings say co-issued pipes write a register every 21.33 units against a k32 step's 32.
  - Hashing W is symmetric (the adversary must hash the same bytes, under ShaSpeed), so it costs T2 and not T1.
- **T2: honest overhead.** t_H/t_S ≤ 100, in the served step of Llama-3.2-1B int8 at T 512 and T 8,192. Report both.
- **T3: accuracy.** Serving O′ in place of O raises perplexity by at most 5% (WikiText-2 at 2K; report 8K too).
- **The assumption: as mild as possible.** A design is judged at the strongest rank its argument needs. SAK (rank 4) is the
  bar to beat (Daniel: "milder than symmetric algorithmic knowledge"):

  | Rank | Kind of assumption |
  |---|---|
  | 0 | a theorem only |
  | 1 | a measured hardware floor, falsifiable by one microbenchmark per instruction (H32: every instruction costs at least 32 units per 32-bit register written, a unit being one dense int8 MAC at 4,096 per SM per clock) |
  | 2 | a small concrete algebraic-complexity fact checkable on CPU (A1-type: products a pipe needs per independent word) |
  | 3 | model faithfulness or representation effects (W1, A2) |
  | 4 | symmetric algorithmic knowledge on a distribution (SAK) |
  | 5 | a concrete hardness conjecture (TT, TT_NCP, TT_OUT) |

- **H200 numbers only.** RTX PRO 6000 (sm_120) numbers don't transfer: b1 runs at 8× s8 on the H200, sm_120 has exact
  integer atoms, and the Accel γ is about 1.6 on sm_120 against about 5 on the H200.

## 3. Power ranking, first pass (coordinator, 00:15Z)

Built from three catalog passes (catalog-lean.md, catalog-notes.md, catalog-code.md, beside this file). A cell marked "?" is
unmeasured on the H200.

| # | Design | Assumption (rank) | T1 gap on H200 | T2 honest overhead | T3 perplexity | Lean | Status |
|---|---|---|---|---|---|---|---|
| 1 | **NCP + the rank bound** (Theorem D / DistinctWrites under H32, MH100/MH100w): the noise cancels inside one 3K-wide product, so O′ = AB exactly; every k32 step's accumulator (each dot product's running sum) is committed | 1 (H32, measured on the H100 only) + 3 (W1 faithfulness) + certified checkpoint independence; no A1, no A2, no SAK | In work: 32/(3K), so 0.52% at K 2,048. It fails at K < 1,067 (attention, K 64: 17%). In time it's open: up to 1/3 in the generous model, unless non-tensor pipes' word rate is bounded or H uses those pipes too | 3× MACs + SHA-512 of every k32 word (about 0.375 B per useful MAC); order 10²× at T 8,192 by GD-HASH's rule (estimate); ? | int7 activations: +4.66% on a 70B (EPS1); ? on Llama-1B | `perWordMH100` and `embedMH100` proved (draft semantics); `H32` def locked; `A2WordsIff` locked | served in vLLM (`ncp-v1`/`v2`); modelled on the 4090 |
| 2 | NCP + Forest / open pipe (Theorem D with MinRank ≥ 5) | 1 + MinRank (certified) | as 1 | as 1 | as 1 | quadratic forest proved; `A1_nonquad` open | 4090-class |
| 3 | Dither-2S LI-CP-W p=1 (the machine model's H200 headline) | 4 (SAK, TS polytope, FOLD-OPT, SCALAR-ROUTE) | headline ε 0.86% (sheet), 1.022% (measured, DC-90-ri) | γ 2.34–2.41 (modelled) | ? | Dither WorkSound and the H200 instance proved and locked | not served |
| 4 | Pearl-C (FP8), TT_OUT at ρ = 1/400 on the H200 | 5 (TT_OUT) | ≤ ρ = 0.25% under TT_OUT | slowdown in panels (sm_120 end to end; sm_90 GLM) | FP8 | — | served on sm_120 and on the H200 (`-h3`; GLM-5.3, whole step 4.52×) |
| 5 | NCP under TT_NCP(0.5%) | 5 | 0.51% (4090 milestone); 0.69% at 4k² | as 1 | as 1 | `gammaFromTTNCP` | — |
| 6 | D1 KW transcript (rank-16 noise) | 5 (TT 0.5%) | 0.85% milestone; 2.5–7% on LLMs | 131× hashing (4090) | exact | — | superseded |
| 7 | Dither-C peel fx1 | 4 | 0.54%, sm_120 only | γ 2.14 (sm_120) | `co32c` arm | — | served on sm_120 (`eps1-sm120-h1fx1`) |
| 8 | Dither v3.2 no-fold | 4 (SAK) | floor G/2 at t = 8 (gap 50%); 0.80 at t = 3 in v3.2's layout | γ 1.40–2.15 (H200, modelled) | ? | WorkSound locked | not served |

Out:
- **H-1T** (genuine FP8): fails T1 at served shapes on its own H100 numbers. DistinctLiveH1T is an open conjecture, not refuted; EncodeChainRefuted refutes a different premise (CORE-RANK).
- **D-3s/m** (genuine FP8): NO-GO. An f32 block value costs 64 + FADD, so at FADD 32.00 (Round 10) nothing clears 1% below 16,384³ (CORE-RANK).
- **A second product on uniform noise:** a 50% gap.
- **J10-R / H1 as served:** commits the un-noised product, so it earns 0; the RelSec floors are 0.854.
- **C2-O:** 2.2–9.6%.
- **D3 padding:** not useful work.
- **D5:** failed.
- **Pearl's whitepaper scheme:** ≥ 5% by design.

## 4. The open items that decide #1

1. **H32 on the H200.** Measure the per-instruction register-write price table, using the Round 7 and Round 8 method (H100
   SXM). No H200 table exists, and `Accel/H200.lean` holds datasheet placeholders.
2. **Time.** Show the non-tensor pipes (FFMA, IMAD, IDP4A, FP64, b1, sparse, mma.sync) can't produce independent checkpoint
   words fast enough to open a gap above 1%, or have H use them as well. Name the premise (rank 2) and its margin.
3. **T2.** NCP's measured honest overhead on the H200 with SHA-512 of every k32 word.
4. **T3.** NCP's perplexity on Llama-3.2-1B with int7 activations.
5. **Coverage.** What to do with matmuls whose K is below 1,067 (attention).

## 5. Interim (01:40Z)

- **T1's clock (coordinator's default; CORE-SMELL C1).** T1 is in SM-clocks. The clock, under load and beside SHA-512, is a
  separate term ε_clock that we report but don't fold into T1. In seconds the clock alone moves 1.8–18.5%.
- **Hashing in T1 (pending Daniel; C2).** At k32 words, hashing is 98–99% of t_H, so in total time T1 would turn on
  SHA-512's speed (ShaSpeed, algorithmic knowledge about SHA-512). The proposal is that T1 counts the matmul's forced work
  and hashing is T2 overhead.
- **#1 in time (CORE-TIME, estimates).**
  - The tensor pipe makes 127.99 k32 words/SM/clk.
  - Beside it, the lane pipes add up to 6.20 (IDP4A 4.10, DFMA 2.11).
  - A tensor-only honest kernel (all there is today) has a gap of 4.4–5.1% at K 2,048, so it fails.
  - An honest kernel that also runs IDP4A and DFMA words has a gap of 1.28% under a products-only premise (holds from
    K ≈ 4,530), or 0.52% with an operand-delivery premise (rank 3–4).
  - The co-issue placements decide it; their probes wait on an H200.
- **#1 T2 (CORE-NCP, estimates):** 160–193× at T 8,192 (fails; 99% hashing, 2,989 GB a forward) and 15–31× at T 512
  (passes).
- **#1 T3 (CORE-NCP, measured, WikiText-2):** int7 linears are +3.68% at 2K and +3.48% at 8K over stock int8 (passes).
  Coverage is 78% of a decode token's MACs (QKᵀ and the lm_head are outside).
- **The served `ncp-v1/v2` is a different design (C3):** depth-16 words, SHAKE / SHA-256, noise reused per epoch, host numpy.
- **No new H200 measurement yet.** vy-dc-h200-5 has refused SSH since 00:24Z, and infra has been asked.

## 6. H32 measured on the H200 (CORE-H32 final, 02:30Z; note:compute-research/20261029T0230Z-report-core-h32)

- **The H200's table is exactly h100Floor:** h200Measured = {mac nominal, tc 32, ffma 32, int32 32, fp64 32, other 32}.
  **H32 holds with no slack, and so does FloorTc 32 32.**
  - Run of record: r20261029-021534-a5a3, 232 probes, all through the SASS gate. Replicate r20261029-014532-a2da agrees within
    1% on the 140 probes they share. Sparse wgmma and LDS beside wgmma: r20261029-022621-53d7.
  - The cheapest writer in each class: tc 32.003 (b1 wgmma), ffma 32.255, int32 32.277 (IMAD beside LOP3), fp64 32.003,
    other 32.279. The H200 reaches 32 in int32 and other, where the H100 sat at 34 and 36.
  - Lean: `h32H200Measured` and `h200MeasuredFloor` build (r20261029-022112-3656), on cursor/compute-research-core-h32-1e0b.
- **The time reading (reported, not judged):**
  - Scalar co-issue writes a register every 21.542 units.
  - wgmma beside IMAD and FFMA lanes writes one every 23.02, and beside IDP4A lanes every 25.60.
  - FP64 beside wgmma adds nothing, so the 60% case doesn't occur.
- **CORE-SMELL C7, answered:**
  - sparse mma.sp ≥ 48.25; sparse wgmma 32.0 (A from registers) or 36.0 (A from shared);
  - carry chains 44.7–55.3; uniform datapath 2,048 a word; atomics and reductions 64.6–1,123;
  - an L2 lookup costs 472 units a word (128 through L1), so the Accel instance's 8 is sound but 59× loose;
  - m8n8k4 with f16 inputs is not a tensor instruction on sm_90a.
- **For #1:** the rank-1 premise is now measured on the H200. What's left for T1 (§2a) is the model refinements: charging work
  done before the noise (the final word) and charging forming. Model faithfulness (rank 3) stays a premise.
