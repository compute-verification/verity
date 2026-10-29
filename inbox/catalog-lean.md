# Catalog: PoUW work-lower-bound strategies in the Lean tree (origin/main d40fe21d26, read 29 Oct 00:10Z)

| # | Strategy (main identifiers) | Commits (words, along K, width) | Noise | Machine | Bound form | Premises | Status / lock | Card | Honest–adversary gap |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Theorem D / AlgM_16 (`AlgBound`, `H16`, `word`, `nonFinal`) | non-final checkpoints at depths 16, 32, …, 3k−16; m·n entries a unit; 32-bit ℚ-as-int | NCP E₁, F₁ (an int7-ish grid) | AlgM_16: free + priced ≥ 16 | cost ≥ 16·\|U\|·m·n·(3k/16 − 1) | `CheckpointIndependent`, `H16` | proved in Proofs/Pouw/Dimension/*; not a locked guarantee (its definitions are locked under the Dimension packages) | abstract / 4090-priced | γ₀ = 16/(3k) against an honest 3mnk |
| 2 | Causal D (`AlgBoundCausal`, `Causal`, `wordC`) | the same non-final words; A_u may read an earlier E₁ | same | AlgM_16 | same | CI, `Causal`, `H16` | proved (`CausalProofs`); unlocked | abstract | 16/(3k) |
| 3 | Route U D (`AlgBoundU`, `wordU`, `AlgBoundWrappedU`) | the same schedule; biased Cᵁ (β ∈ {127, 128, 129}); int32 wrap | same + bias | AlgM_16 | same | CI (+ for the wrap: SignedPerm / a k bound) | proved (`RouteUProofs`); unlocked | abstract | same |
| 4 | A1 + Mixing / AlgM_4090 (`A1_multiplicative` / `A1_W1`, `MixingLemma`, `AlgBound4090`, `TTNCPOfA1A2`) | same checked words | same | AlgM_4090: big ≥ 16, or a scalar product ≥ c_s (= 8) | cost ≥ 16(1 − γ₁)D; TT: not cost/(1 − γ₀) < Σ 3mkn | CI, `A1(c, γ₁)`, often `MinRank2`; A2 for TT | stated, lemmas proved; `A1_W1` "probably false asymptotically"; unlocked | 4090 (sm_89) pricing | γ₀ needs (1 − γ₂)(1 − γ₁)(1 − 16/(3k)) |
| 5 | Forest / closed and open pipe (`ForestBound`, `A1_cost_quad`, `AlgBoundQuad8`, `OpenQuadPipe`, `TTNCPOpenQuadOfA2`) | same | same | closed or open FP32 pipe (cs = 8, pb = 16); nothing free | cost ≥ 16·D (γ = 0 for the quadratic case); TT from A2 alone | CI, MinRank ≥ 5; `A1_nonquad` open | quadratic forest proved (`ForestProofs`); `A1_nonquad` and `LiftMinRank` open; unlocked | 4090-class | tight, γ₀ = 16/(3k) |
| 6 | A2Words (`A2ExtraOutputs`, `A2WordsIff`) | only the `wordSet` targets matched | same | any `Machine` + a pipe class | A2 into closed ⟺ open, for the checked words | CI, MinRank ≥ 5 | proved and locked (`Pouw.SecurityProofs.Dimension.A2WordsIff`) | abstract | — |
| 7 | Embed M4090 / A1_fp (`M4090`, `EmbedM4090`, `A1fp`, `PerWordM4090`, `PerWordM4090Gap`) | the same 32-bit checkpoint words; MMA computes any function; FP32 at 8; INT, data-dependent and sub-word at 16 | same | M4090 (sm_89, W1) | 16·\|S\| ≤ cost + 8·nonQuad; with A1_fp, (1 − γ)·16·\|S\|; NaNGap: 16 a word without A1 | CI, MinRank ≥ 5, `A1_fp` (or `CanonicalNaN` + a gap) | draft semantics; many theorems proved (`EmbedProofs`); unlocked | 4090 (sm_89) | chain witness γ₀ = 16/(3k) |
| 8 | Fp8Mma ceiling (`Shape4090`, `Ceiling4090`) | FP8 atom words (honest 64 a register) | MMA shapes | M4090 | the dimension bound is ≤ 16 per independent word, so ≤ 25% of honest FP8 | inherits M4090 | draft; `ceiling4090` proved; unlocked | 4090 | ≤ 16/64 |
| 9 | MH100 / H32 (`MH100`, `H32`, `PerWordMH100`, `EmbedMH100`) | independent 32-bit targets (NCP's non-final words, or FP8 split-K if linear) | a free subspace V / V₀ | MH100 (sm_90, W1) | cost ≥ 32·\|ι\| | `H32` = `Floor 32`; CI for NCP's words; no A1 or A2 | draft; tables and proofs; the `H32` definition is locked in the audit's module hash | H100/H200 (sm_90): nominal and measured on the H100 SXM | 32 against an honest 32 a word (tight at N ≥ 64) |
| 10 | MH100w DistinctWrites (`DistinctWritesMH100(On/Floor)`, `DistinctLiveMH100`, `DistinctWritesFloor`) | distinct non-free 32-bit registers; packed codes 32/κ | a salt-independent free set F | MH100w (W1 frees only) | cost ≥ 32·\|T\| (or the floor c); live: ≥ 32(1 − ε)·\|ts\| | `H32` / `Floor` / `FloorTc`; optionally `FragLanes` for cheap legacy forms | draft; the hardware citation is Round 8's floor; unlocked guarantees | H100 SXM measured; the H200 Accel table is separate | work gap 0 if the words are distinct; in time a register can be written every ~21.33 units |
| 11 | Lift_MinRank (`LiftMinRank`, `LiftMinRankL5e`, `MinRankGe`) | the same words; correctness only on a set Z | same | AlgM | Pr[Z] ≤ ε(D − p) | CI, MinRank ≥ ρ; L5e on paper | named and open in Lean (the L2–L4 core is in `Lifting.lean`); `MinRankGe` is a locked assumption | abstract | a lottery / grinding bound |
| 12 | RelSec floors (Theorems K and S: `StagedFloorK*`, `OverlapFloorK*`, `TheoremS*`) | committed prefix products / frames of blocks of dot products; the VF3j-16 charge | staged / overlap draw bits; 7 planes re-encoded | an abstract `Machine` + `Service` | accepted charge ≤ (R − c ln2)·St·κ L n n′ B − Vmax, w.p. ≤ 2⁻⁴⁰ + R(δb + δz) | `ServiceSound*`, `KScope*`, `ChargeVF3j16*` | proved and locked (every variant) | protocol-agnostic | luck 3/4 affine; 0.854 one-sided |
| 13 | J10-R floors (`*FloorKOneSidedJ10R*`, `JBits` / `jPrefix`) | a call: row dither bits + column coins + U bytes; prefixes T_t mod 2³²; int8 wrap | H1's SHA-512 streams, per call | same + `JRunMap` | the same floor at call scope | the `*J10R` ServiceSound / KScope / Charge | proved and locked | J10-R / H1 as served | parity luck 0.854 |
| 14 | Dither WorkSound (`WorkSound`, `WorkSoundNoFold`, `FreedOfFloor`, `FleetSound`) | the sampled blocks' words + a Merkle / SHA commitment | dither coins → commitment | an abstract `Accel` | Pr[accepted ∧ rv + W_other > OPT + ε_A·C] ≤ …; freed ≤ ε·C | `SAKMatmul`, `ShaSpeed`, `ServiceSound`, certifier / egress, `ExtractAdv`, … | proved and locked | any Accel | freed ≤ ε·the honest dual cost |
| 15 | DitherInstance (`WorkSoundH200` / `Sm120`, `WorkSoundOnCard*`) | same, on a card table | same | `Accel.H200` / `Accel.Sm120` | WorkSound at the θ floors | + `CardWithinAccel` (locked) | proved and locked | H200 (sm_90), RTX PRO 6000 (sm_120) | the same freed form |
| 16 | PeriodJ10R / SharedNoiseLucks (`H1J10RCheckedCreditPeriod`, `SharedNoiseLucks`) | words of a tile stratum; fold keys a unit | J10-R's shared row and column noise + folds | H1 as served / PouwCheck | checked credit a period, at rate `j10rRate` | `TileDrawIsCoin`, `TileServiceSound`, `SharedNoiseLucks`, `UnitFixed`, … | the guarantee is locked in the Properties audit; SharedNoiseLucks is a computational claim | H1 / J10-R | lucks 7/8 and 1/2 in the statement |
| 17 | TS sheet headlines (`Sm90*Headline`, `Sm100*Headline`, `Sm120*Headline`) | dither-2s + many fold routes (56 on the H200 menu; 14 on the B200) | TS shared noise / live coins | data-sheet / measured DC vectors | headline ε% under the D-ideal floor | A1–A7, FoldOpt, FacetCeilings, … | locked conjectures (frozen) | H200 (DS-90 / DC-90); B200 (DS-100 / DC-100); sm_120 | e.g. H200 measured 1.412%; B200 sheet 0.783% |
| 18 | DistinctLiveH1T (`Pouw.Fp8Atom.H1T.DistinctLiveH1T`) | q distinct masked codec words | H-1T | MH100w draft | a q·E cost claim | open conjecture | locked; refuted (`EncodeChainRefuted`) | MH100w | — |

JR2 has no identifier in this tree.

## Notes

**Wall time against summed work.** The docstrings insist the H32 / MH100w bounds are additive W1 work, not wall time. Separate
pipes co-issue: Round 8 measured 1:1 DADD+FADD down to 21.45 units a register loop-free and 21.33 at full dispatch (the FP64
pipe's bound), below H32's 32 (Assumptions/Dimension/H100Assumptions.lean, H100W.lean, H100LoopFree.lean, H100Measured.lean).
A time bound needs a max over the tensor and CUDA-core pipes.

**The H32 tables** (H100 SXM; none is H200-specific):
- Round 7, `h100Measured`: mac nominal; tc 32; ffma 33; int32 64; fp64 34; other 40. It meets H32, but it no longer
  lower-bounds the hardware after Round 8 (IADD measured 34.68 against the table's 64).
- Round 8, `h100LoopFree`: tc, ffma and fp64 32; int32 34; other 36.
- The floor table, `h100Floor`: the loop-free table with int32 = other = 32. It is the hardware citation, with legacy
  `mma.sync` only as measured.

No Lean table says "measured on the H200". `Accel/H200.lean` uses datasheet placeholders pending calibration.

**How fast non-tensor pipes add rank.**
- On the 4090, only FFMA, FADD and FMUL are cheap (8). INT32, FSET, half2 and the rest cost ≥ 16 a register (Rounds 4–6),
  so they count as big writers (EmbedAssumptions.lean).
- Forest: with FP32 at 8 and priced instructions at 16, 2·dim ≤ 2·#priced + #FP32, so cost ≥ 16·dim (Forest.lean).
- A1 / `A1_cost` bound how many products, or FP32 steps, a dimension needs.
- Under H32 on the H100, FP32 also costs ≥ 32. So in the work model non-tensor pipes can't undercut the per-register floor;
  only co-issue undercuts time.

**Locks.**
- Security/lean-audit.json locks RelSec, Dither, DitherInstance and the TS headlines. For Dimension it locks mostly module and
  definition hashes; the only guarantee is `A2WordsIff`.
- Properties/lean-audit.json locks `h1J10RCheckedCreditPeriod` and the SharedNoise premises.
