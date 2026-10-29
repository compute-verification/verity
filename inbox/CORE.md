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
| 4 | Pearl-C (FP8), TT_OUT at ρ = 1/400 on the H200 | 5 (TT_OUT) | ≤ ρ = 0.25% under TT_OUT | slowdown in panels (sm_120 end to end; sm_90 GLM) | FP8 | — | served on sm_120 (`-h3`) |
| 5 | NCP under TT_NCP(0.5%) | 5 | 0.51% (4090 milestone); 0.69% at 4k² | as 1 | as 1 | `gammaFromTTNCP` | — |
| 6 | D1 KW transcript (rank-16 noise) | 5 (TT 0.5%) | 0.85% milestone; 2.5–7% on LLMs | 131× hashing (4090) | exact | — | superseded |
| 7 | Dither-C peel fx1 | 4 | 0.54%, sm_120 only | γ 2.14 (sm_120) | `co32c` arm | — | served on sm_120 (`eps1-sm120-h1fx1`) |
| 8 | Dither v3.2 no-fold | 4 (SAK) | floor G/2 at t = 8 (gap 50%); 0.80 at t = 3 in v3.2's layout | γ 1.40–2.15 (H200, modelled) | ? | WorkSound locked | not served |

Out:
- **H-1T** (genuine FP8): DistinctLiveH1T is refuted in Lean (EncodeChainRefuted).
- **D-3s/m** (genuine FP8): NO-GO at the measured FADD price of 32.11.
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
