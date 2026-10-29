# Catalog: PoUW designs in the notes (read 29 Oct 00:10Z)

| Name · date | Commit / cadence | Noise | Argument / assumptions | Honest overhead | Accuracy / perplexity | Gap (forced) | Status |
|---|---|---|---|---|---|---|---|
| D1 KW transcript · 27–28 Sep | running sums every d = 16; Merkle under H; per unit | rank-16 additive; outer {−3..3}, inner ±1; E_R per group of 16 | TT(0.5%) + ROM; W1 / M_4090 | ~1.004× arithmetic; 131× hashing; ~1.3× reads (4090, measured) | n/a (integers exact) | γ ≈ 0.85% milestone; 2.5–7% on LLMs | baseline; LLMs unreachable under worst-case inputs |
| D2 nonlinear fold · Sep | fold, then hash each checked word | as D1 | TT + no shortcut through the fold | fold 0.5–2× a GEMM; less hashing | — | the same floor as D1; uncredited fold → 35–67% | later; linear folds broken |
| D3 FLOP padding · Sep | output only, on H-derived GEMMs | uniform int8 padding | C1(γ₀) on uniform GEMMs | hashing ~2,100/k | no useful decode | γ = γ₀ on the padding only | not useful work |
| D4-s · Sep | running sums; reject low-rank adjacent steps | rank-8 signed partial permutation | TT for that distribution | as D1, cheaper correction | — | ~2.0% at 4k² (class floor 1.89%); 1% for k ≳ 18.7k | open research; needs a fix |
| D5 recursive noise · Sep | nested correction products | rank r on the corrections | TT on the corrections | cost rises each level | — | — | failed |
| D6 R_copy / schoolbook · Sep | the same transcript | — | a restricted model | — | — | inside the model only | a Lean witness, not evidence for TT alone |
| D7 NCP · 28 Sep | one stacked 3k-wide GEMM; route-U Cᵁ every 16; IndepRule + MinRank on P | full-rank and correlated; cancels in the accumulator | TT_NCP(0.5%); backup Theorem D ∧ H₁₆ ∧ A1 ∧ A2 | ~3× arithmetic; ~394× hashing; 3.7–4.9× wall (4090, r20260928-*) | exact, int7 | γ ≈ 0.51% milestone; 0.69% at 4k² | adopted for transformers |
| Pearl whitepaper / miner · Sep | output-tile tickets / XOR fold | FP8 rank 32; integer miner, r ≥ 128, sparse ± | Pearl's Assumption 1; jackpot ≥ 5% | excludes forming and hashing | policy allowances | ≥ 5% by design | outside Verity's γ game |
| Pearl-C H100 · early Oct | atom chain + tickets; BLAKE3 | E4M3 lines + peel | TT_OUT (a priced conjecture) | modelled γ pins; no live table | FP8 | output-volume floor ~1% under Accel | pins deletable; superseded by sm_120 |
| Pearl-C sm_120 · live (#1116) | same; -h2 hashing | E4M3 m16n8k32 | TT_OUT; a γ debit | measured serving (RTX PRO 6000) | served (#1001) | a live ACCEPT window | live product path |
| Pearl-C4 / NVFP4 · Oct | a block-scaled chain | NVFP4 / MXFP4 | FP4 γ pins | measured on sm_120 | FP4 perplexity separately | scheme `pearl-c-nvfp4-v0` | stays; circuit pc4 |
| H-1 / H-1R · 28 Sep | 2-block HOPPER_E4M3_K32 words | dithered activations; a 0→1 tag | DistinctLive + H32 | forming ~283 an element | perplexity within ~1% of FP8 | H-1 NO-GO (4 families); H-1R 23% under S1 | superseded |
| H-1T · 28–29 Sep | 29 real + 3 tag lanes; 4T−3 credit | fixed injective tags + salt | DistinctLive(0) / ε = 1/400; H32; a named branch gap | forming 297.6 an element; ~2.21× tensor (H100 SASS) | Qwen / Llama: −0.24 to +0.92% against FP8 | γ₀ = 1/400: 0.75% / 0.50% at 16k³ / 32k³; fails at 2k–4k | was the leading genuine-FP8 design; refuted in Lean since (catalog-lean #18) |
| D-3s/m (genuine FP8) · Round 9 | 2-block dithered FP8 | continuous activation noise | DistinctWrites / H32 | real time 5.4× plain FP8 (H100) | — | best M2q 0.92% only if FADD ≤ 32.05; fails at the measured 32.11 | NO-GO at Round 9's prices |
| RS D1 peeled full-rank · 15 Oct | prefixes every L = 32 | full-rank masks | RelSec Model B | peel ~3× | — | ε ≈ 1 | closed (dominated) |
| RS D3 KSW, no prefix · 15 Oct | output-ish | low rank | Pearl-like Assumption 1 | — | — | A = B = 0 → 1 + ε ≳ k/r | closed |
| RS D4 / Dither edit · 15–16 Oct | edit A and B; duo receipts; prefixes at L = 32 | ±1 / u-alphabets; shared P noise | Theorems E, K and O; COINC; H-μ; NFP(ε) | edited GEMM 1.57× int8; duo heavy (sm_120) | linears ±1 +0.35%; attention peel +2.4–3.6%; `kc` Qwen +0.5% | in scope c = 1 → 0.997–0.998 of nominal; real time ~13–22% of honest | leading relative-security design; 70B linears broken, then SmoothQuant |
| Certified filler (pfill-K) · 16 Oct | extra tensor work in idle time | — | FILL-SEC / Theorem K scopes | a coverage lever | serving perplexity untouched | fills idle time; the CH chain closed | leading filler; CH withdrawn |
| Dither-1S / 2S / C · 17 Oct (EPS1) | the last checked word; duo / fx1; live coins | 1-bit (C) or 2-bit dither; co32c / pv | TS polytope + SCALAR-ROUTE / FOLD-OPT; later SAKMatmul | γ ≈ 1.6–1.9 (2S / C); NCP-3k γ ≈ 2.8 | C +1.57–2.11% (8B / 70B); 2S better | core ε ≪ 1% at 100% checking (C + pv ≤ ~0.15% skip); table-route risk | the named registry; `eps1-sm120-h1fx1` is the first scheme |
| NCP-INT / NCP-3k (EPS1) · 17 Oct | a port of NCP | as NCP | retire TTNCP_U; TS pricing | γ ~2–3 | 70B NCP +4.66% (worse) | dominated by Dither-C on perplexity | out of the registry (#2238) |
| J10-R · 25–26 Oct | H1 tiles / window credit; shared row and column noise | per-unit salt; SharedNoiseLucks | CheckedCreditPeriod; TileDrawIsCoin | hash-bound: full H1 on the H200 infeasible (~660–1,340× a step) | the demo's integrity path | subset credit W ≪ a block; ε from K′ draws | demo credit; smell: commits the un-noised product |
| JR2 / no-fold Dither · 26–28 Oct | no-fold receipts; fresh noise along K | dither-noised operands | WorkSound + SAKMatmul (+ ShaSpeed, Import, …) | γ_hash ≈ 6,460/L; h ~0.24–0.34 | within the 5% goal when L is large | ε = (1 − h)(1 − c_mm) + …; the SAK residual stays | the current Accel / golden-demo bet |
| Accel D5 / D6 prototypes · 26–28 Oct | a public-best hull; a period LP | NoFold / dual τ | SAK on noised operands; pub/H | H200 γ ~5; RTX PRO 6000 γ ~1.6; share 60% against 84% | V4 plan arms | ε ~0.23–0.28% at δ = 1e-6 | active planning; the card split is first-class |
| C2-O served / C2-T filler · 27–28 Oct | the high bit of O / a 2-bit filler | byte-cell | SAKHigh-O / SAKHigh-T | M ~1.7 served; filler M ~1.85–2.09 (H200) | — | served ε 2.2–9.6% (fails the ≤ 3% bar unless corr. ≤ f15); filler ε ~0.93% | served leans to failure; the filler works |
| Pearl-C-T (proposed) · 26 Oct | commit the k32 partials | as Pearl-C | data flow, G1/G2/P1/P2 | the floor rises to 0.11–0.22 | — | still ≪ 1 − γ₀ without an own-products assumption | a hybrid bet; not shipped |

Primary sources:
- the relayed Sep docs: problem-statement, milder-assumptions, new-crypto, hardness-shaped-matrices, genuine-fp8-red-team-round9;
- 20261004T2101Z-report-pouw-survey-inventory and 20261015T2020Z-report-research-index;
- 20261017T0310Z-report-eps1-program and 20261017T0340Z-draft-skip;
- 20261026T0130Z-{dither-thm, pearlc-model} and 20261026T2345Z-draft-sak-lb;
- 20261025T2330Z-report-h200-pouw, 20261028T0116Z-report-epsw-global and 20261028T0523Z-finding-dither-noised-floor;
- kb/smell-catalogue.

## Milder assumptions considered (verdicts)

| Candidate | Verdict |
|---|---|
| Theorem D + IndepRule(P) (Winograd's dimension bound / output flattening × price) | not an assumption: classical, with P certified; tight γ₀ = 16/(3k) in AlgM₁₆ |
| H₁₆ (≥ 16 units a priced word; the sm_89 list complete) | the mildest real hardware assumption; a measured floor |
| A1(2): multiplicative complexity ≥ 2 products a dimension (sub-word priced) | needed for the FP32 pipe; ~2.7× margin; Daniel priced sub-word, so c = 2 |
| A1(4), or its W1 scalar-pipe form | a weaker margin (~1.35×) if packing is free; superseded by sub-word pricing |
| A2: representation effects ≤ γ₂ | the honest residue; equivalent to TT_NCP on non-algebraic attacks, given D + A1 |
| MinRank(P) + lifting | a design check; production P has MinRank 16 (done) |
| TT_NCP⁰ (zero input only) | weaker; needs an input reduction (Theorem D gives one only inside AlgM) |
| TT_NCP(0.5%) | the baseline named conjecture for NCP |
| Ruled out: cryptography (LWE, LPN, …: a meta-reduction); unit-cost multiplication models (Winograd / Strassen); I/O, pebbling and bandwidth under W1; bit-gate counts (~12.5 a word → γ ≥ 22%); asymptotic fine-grained conjectures; uniform operands (γ ≥ ½); hardware atomicity without a composition rule | none can give γ ≤ 1% alone |
| Later: SAKMatmul | its sampling half is proved (`sak-lb`); SAKResidual stays an assumption (a rank / work lower bound) |
| ShaSpeed, ImportWithinCertifier, FOLD-OPT, SCALAR-ROUTE, NOSUM, COINC, NFP(ε) | named empirical, service or priced premises (the Accel audit) |
| Barrier options A–D (a narrow assumption / SHA as a random oracle / an unconditional floor / a new protocol) | open with Daniel (26 Oct) |
| SAK at 8-bit words | fails: Strassen mod 2⁸ at 0.875 of honest (H200, `sak8`) |

## Card differences

- **4090 (W1):** NCP / D1 prices, and their falsification; INT32 at 16 a word; TT in per-word form; u8×s8 at full rate.
- **H100 (H32 / HOPPER_E4M3_K32):** the H-1T / D-3 γ come from FADD ≈ 32; the atom truncates (no exact ring, unlike sm_120);
  the Pearl-C H100 pins are unused; forming costs ~298 an element; γ clears 1% only at large k (16k³ and up).
- **H200:** J10-R, C2 and the dither floors; b1 ~8× s8; the Strassen attack on Z/2⁸; Accel γ ~5 and share ~60%; served C2-O
  ε is often > 3%.
- **RTX PRO 6000 (sm_120):** Pearl-C is live; Accel γ ~1.6 and share ~84%; exact integer atoms help Dither / Pearl-C-T;
  sm_120's Pearl peel / period must not reuse H100 semantics (smell catalogue). The calculator's ε_W of 0.744% is the
  sm_120 line, not the H200's served ~2.7–3.1%.
