import Mathlib
import Definitions.Pouw.RelativeSecurity
import Proofs.Pouw.SteinerRigidity

/-!
# Relative security for matmul PoUW: the definition, the masking algebra, the reduction's shape

Neekon Vafa's definition as RS-DEF froze it
(v1), in an abstract machine model, so that the construction and
proof workers' results land here as theorems or as named hypotheses. The definitions the guarantees
(`Definitions.Pouw.Guarantees.RelativeSecurity`) read are in `Definitions.Pouw.RelativeSecurity`, under the same names.

* The machine model (`Machine`, `MatMachine`): algorithms from inputs to outputs, their semantics with the coins as
  part of the input, a per-run `cost` and its worst case `time`, with axioms: nonnegativity, monotonicity (running `f`
  first, or beside another algorithm, costs at least `f`), subadditivity of sequential and side-by-side composition,
  free routing, and the cost of a linear map (an `a × b` by `b × c` product costs at most `κ a b c`). `costed` shows
  the axioms are satisfiable with nontrivial prices.
* The definition (`Protocol`, with a commitment before the live coins, `CommitsFirst`; `Complete`; `Secure`, Neekon's
  `∃ M'` form; `SecureInf`, RS-DEF's frozen form against `Tuse`; `Core`, the per-run proof target; `Efficient`;
  `RelativelySecure`) with `ε`, `δ`, `γ`, `η`, `α`, `κ`; usefulness, completeness and the floors quantify over the
  protocol's `promise` (RS-DEF v2 §1.4, `A, B ∈ G_use`). Probabilities are uniform counts over the finite σ and coins
  (`prob`, `expect`). Core, the floors and Lemma R charge a run the program's size and its cost,
  `max(|M|, cost)` (`Machine.charge`, RS-DEF v2 after RS-RT's V1). `secureInf_of_core` (Core ⇒ Security for `δ > κ`),
  `rate_of_core` (RS-DEF's rate form) and `core_of_floor` (RS-DEF's Lemma R) are proved.
* The masking algebra, proved over any ring (`mask_peel`, `lowRank_peel`), exact over `ZMod (2^32)` and over any
  field, and recovering the integer product from its s32 residue when the entries are below `2^31`
  (`int8_product_exact`); a saturating int8 accumulator breaks the peel (`sat8_breaks_peel`).
* The reduction's shape (`reduction`, `M' = M ++ Q`): run the extractor (M's run and its committed blocks), keep the
  prover messages, peel and correct; `time_reduction_le` is `time M' ≤ time E + time peel + time correction`.
  `secure_of_extraction_floor` gives security from two named hypotheses, `Extraction` (RS-EXT's) and `Floor` (the floor
  agenda's), and `secure_fullRank` and `secure_lowRank` instantiate it with the explicit peels of `MatMachine`. With
  products typed by grid (`WideGridMatMachine`: grid products at `κG`, wide-by-grid products at `κW`), the full-rank
  peel costs `2κG n³` and the low-rank peel `(2κG + 2κW) n² r` (`secure_fullRank_grid`, `secure_lowRank_grid`,
  `secure_lowRank_advTime_grid`; RS-MODEL §5's `10 n² r` u at 1 u and 4 u).
* Route (c), edited instances (by ruling): correctness reads σ, an `Edit` adds σ-indexed
  noise to both operands, unpeeled; `edited` is the protocol (promised on the instances `use`, served by a
  product algorithm `m`), `edited_complete` its completeness, `honestEdit` costs `time m + 2κadd n²` beyond the proof
  layer and the noise expansion, and `relativelySecure_of_editFloor` gives relative security from the named hypothesis
  `EditFloor` (RS-D1E's floor), with `m` the worst-case product (`_wide`) or a grid product (`GridMatMachine`, `_grid`). The bit-pair projection's algebra
  (`bitPair_coefficient`) and the abstract core of RS-D1E's homogenization (`bitProj_rank_le`, `products_ge_half_rank`;
  `editedEntry_products_ge`: an edited prefix-sum entry with `r` live terms needs at least `r / 2` plain products,
  whatever the committed instance; `editedEntries_products_ge`: entries at distinct rows and columns, sharing their
  products, add up, and with merges free they still add up when entries share only rows or only columns, `_cols`,
  `_rows`) are proved.
* Theorem B's accounting over a field, for entries that share rows and columns (A3): in Theorem B's one-parent model
  (`Forest`, `AccumProg`), `m` plain products and `H` heads hosting a target give
  `2m + b·H ≥ Σ_e r_e` (`editedEntries_products_ge_shared`), the forest re-rooted at 0 being a Steiner tree
  (`Forest.exists_steinerTree`) that Theorem 1 (`SteinerRigidity.blockRigidity`) bounds.
* Theorem S and S_G: a
  finite game of rounds with fresh coins, one check a round, accepted iff every check passes (`RoundGame`); with the
  per-round pass bound `1 − (1−p)·d` named (`PassBound`, or `PathBound` along guard paths),
  `Pr[accept ∧ deficiency ≥ D] ≤ e^{−(1−p)D} + R(δ_bind + δ_ZK)` (`theoremS`, `theoremS_G`), by the supermartingale
  `Π 1[pass]·e^{(1−p)d}` as a product over rounds and Markov. The window's corollary (`window_le`; it reads only the
  exponential form `E[1[pass_w] e^{(1−p) d_w}] ≤ 1`, `window_le_exp`) is a service floor (`RoundsHyp`, over
  `ExpBound`, `serviceFloor_of_rounds`), and the B_θ certified fraction rests on it in place of Theorem M
  (`certifiedFraction120θ_rounds`, at `2^{−40}` `_rounds_40`).
* The Model B bridge (RS-MODEL): typed circuits of priced gates with free routing; `ModelB.machine` is a `Machine`
  whose time is `timeB`, `ModelB.matMachine` a `MatMachine` from priced circuits (`MatGates`, realized by a GEMM gate
  family), `matMachine120` the sm_120 instance at `price120` with the worst-case `κ = 16` u (circuits named `Gemm120`),
  and `gridMatMachine120` its width-typed refinement, the int8 tile GEMM at `κ = 1` u on `G_op = int8` (circuits named
  `Int8Gemm120`). `relativelySecure_edited120` is route (c) on the int8 grid at 1 u, `relativelySecure_edited120_wide`
  on unrestricted operands at 16 u. Machines take any additive `Price`. The service range-checks the committed instance
  into `G_use = [−(127 − s), 127 − s]` (`GridService`, RS-RT's V6), and with noise bounded by `s` the edits are int8
  (`edit_int8`; on the entries σ opens, `accepted_edit_int8`).
* B_θ (RS-MODEL §6.2, θ = 1/5): `price120θ` over instructions with their shared-memory words; B's circuits lift at
  `4/5` and every B_θ adversary has a B shadow, so B's floor carries at `4/5` (`serviceFloor_θ`). Route (c) there at
  `N = 0.8 n³` u takes any B floor `F_B ≤ n³` (`relativelySecure_edited120θ_of_floor`, `ε = U / (0.8 F_B) − 1`):
  Conjecture A's `n³` (`ε = γ`), Theorem E's form `(c / m_max)(φ n³ − sl)` over ℚ (`_rat`) and its proved case
  `c = ½`, `m_max = 2` (`_half`). The certified fraction (`certifiedFraction120θ`, `_half`): accepted runs are charged at
  least `(c φ / m_max) N` less the slack, except with probability `κ`. Model A (RS-MODEL §3) is ported over
  ℚ, and `timeθ_le_timeA` is `B_θ ≤ A` as the combination `(4/5)·ys + (1/5)·R6` of A's rows.
* Theorem K (the full-floor draft r5, §3.6), over any ring with a map to F₂ (Z/2³² for s32 wrap):
  a straight-line program of products of bit-affine operands and linear steps (`Slp`, closed form `Slp.val_eq`),
  projected onto bit pairs mod 2 (`projF2`, zero diagonal), has F₂ rank at most twice its products (`theoremK`,
  `Slp.rank_projF2_le`). On the re-encoded design X = ⌊x/s⌋ + b (full-floor r6; `dP`,
  `dX`, `dIncr`) a cell-block's target has F₂ rank 2L (`rank_projF2_dIncr`), so a certified cell-block needs L
  products (`Slp.dIncr_products_ge`) and cell-blocks whose pieces are disjoint add up (`Slp.card_mul_le`).
  `projF2` reads every bit pair, the same-side ("junk") pairs included: a cross-pairs-only projection would pass
  Winograd without its corrections (`winograd_cross_only`).
* The chain lemma (RS-E-REF r4 IV.2, and r5 V.1): with stacked
  identities every k-block reads the same U bits, so the per-block count needs the chain. A product of a function of
  U's bits and a function of the coins, of any form (`OneSided`), has a cross block of rank 1
  (`OneSided.crossRankLe`), and `T` committed prefix words of one row at distinct (column, block) pairs have a stacked
  cross block of rank `T L` (`rank_cross_dRow`; one cell, `rank_cross_dPrefix`), so they need `T L` one-sided
  products (`chain_products_ge`, `chainBound_oneSided`). Products whose operands are any function of U's bits plus
  affine in the bits (`GProd`, every affine product included) need `T L` too, over characteristic 2: Lemma G
  (`lemmaG`, Winograd's substitution one product at a time on the coins' side, `altForm_subst`) bounds the cross
  block of words with no coin–coin terms by the products' count (`chainBound_gProd`). The cross block alone gives
  `T L / 2` (`chainBound_two`). A round's run is a straight-line program with products of any form (`SlpG`),
  row-pure across rows only (`SlpG.cells_card_le`). `RoundsHypK` puts that in `RoundsHyp`'s cost line,
  F₀ = κ L n n' B / c with no slack (`roundsHyp_of_K`, `_one`), and the certified fraction follows
  (`certifiedFraction120θ_K`, `_K_40`; at w = (1, 2, 4, 8, 16, 32, 65) with no chain hypothesis, for one-sided
  products `certifiedFraction120θ_reEncoded_40`, and for mixed ones `_reEncodedMixed_40`). Route (c) then needs no
  `EditFloor` on this design: the window's floor is B's service floor, and the edited protocol is relatively secure
  from it (`relativelySecure_edited120θ_K`). The one-sided chain lemma holds over any field the ring maps to, so it
  covers exact arithmetic too (over ℚ, merges free, `certifiedFraction120θ_reEncodedExactK_40`). Lemma G needs
  characteristic 2 (s32 wrap, or exact integer sums read mod 2): over ℚ one mixed product commits a word of two
  bit products (`winograd_rat_word`), so the bound fails there (`not_chainBound_affProd_rat`).
* The luck lemma (RS-E-REF r4 IV.4): a nonzero error of degree at most 2 on the cube (`CubeQuad`) is nonzero on a
  quarter of it, in any ring (`luck_quad`), so spot checks with degree-2 errors satisfy `PathBound (3/4)`
  (`RoundGame.Spot`, `RoundGame.pathBound_spot`). Runs with affine operands have such errors on the design's bits,
  U's 7 uniform planes (`SlpG.cubeQuad_err`). With the window's game built from the prover's play (`SpotPlay`,
  `RoundsHypKS`), `RoundsHypK` follows at `p = ¾` with `PathBound` and the certified count derived
  (`roundsHypK_of_spot`), and for one-sided affine products the floor has neither a luck nor a chain hypothesis
  (`certifiedFraction120θ_reEncodedSpot_40`; mixed, `_reEncodedSpotMixed_40`). The ¾ is tight
  (`luck_quad_tight`) and fails in this deficiency for one-sided operands of any form (`luck_oneSided_point`; the
  parity luck below covers them). On the integer alphabet a
  dropped term is right with probability ½ + 1/(2|α|) (`luck_coins`, `luck_dropped`); U's 128 letters are distinct
  and never 0 (`uRe_injective`, `uRe_ne_zero`), so 0.5039 (`luck_dropped_reEncoded`), and ½ for a term dropped
  outright (`_zero`). The window-level ½ is false at `|S| ≥ 4` (RS-E-REF r5 V.2), so ¾ is the window's bound
  (decision 39 states the luck given the drawn pair indices at `p = ¾`).
* Stages inside a round (decision 36, the stage-coins draft §2, as RS-E-REF r5 V.3 repaired
  it): a round of `St` stages, each committing its operands, then drawing both noise blocks of its products (keyed by
  the draw), then committing its outputs, with one check at the round's end. Per stage, a cheat step fixed before its
  draw and the wrong share the draw adds (`StageGame`): `(1 − U) e^{(1−p) D}` is a supermartingale over the stages
  (`StageGame.expect_le`, RS-E-REF r3 III.1's extension), so staged rounds meet the exponential bound that `RoundsHyp`
  reads (`RoundGame.Staged`, `RoundGame.expBound_of_staged`). The staged spot check (`StagePlay`, `StagedPlay`) is
  valid at `p = ¾` for degree-2 errors (`StagePlay.valid_sgame`, `StagedPlay.game_staged`), and `RoundsHypKSt` gives
  `RoundsHyp` with the stage floors summed, `F₀ = St κ L n n' B` (`roundsHyp_of_staged`), so for affine products the
  floor has neither a luck nor a chain hypothesis (`certifiedFraction120θ_staged_40`). Decode's deferred edit, every
  operand committed before one draw, is `St = 1` (`certifiedFraction120θ_decode_40`).
* Overlapping stages (chunks in flight, no node barrier) and stages of different shapes (Theorem K∥,
  the stage-coins draft r2 §2.2 and §2.5): the round's run is one straight-line program
  over every draw's bits, fixed before its first draw, whose operands are affine in each draw's bits at every bit
  point of the others' (`AffMulAt`; `RoundsHypKOv`). K's count is local to one row of one draw, the other draws' bits
  at theirs (`mergeAt`), and row purity adds the rows' bounds over every draw at any depth (`SlpG.cells_card_le_ov`).
  The luck is per draw, the other draws' bits given: swapping a draw's bits for fresh ones is an involution, so each
  cell's pass probability is the spot check's on its own draw (`prob_cell_eq`, `prob_ov_le`, `OvPlay.passBound_game`).
  So `F₀ = κ L Σ_t n_t n'_t B_t`, every stage's cell-blocks once (`roundsHyp_of_overlap`,
  `certifiedFraction120θ_overlap_40`), with no joint projection. `RoundsHypKSt` allows what `RoundsHypKOv` doesn't,
  a stage's program branching on earlier draws; overlap with that (RS-E-REF r3 III.1's adaptive model) is open
  (r2 §2.5).
* The parity luck for one-sided operands of any form (RS-E-REF r6 VI.2):
  mod 2 at a cell-block's frame (`dFrame`), a one-sided run's error is `⟨F x, G y⟩ + a x + b y` with a term per
  product its word reads with an odd coefficient that is two-sided there (`charged`, `SlpG.parityForm_err`), and
  such an error with fewer than `L` terms
  vanishes with probability at most `½ + ½ √½ < 0.854` (`parity_prob_sq`, `parity_luck_short`; ¾ two short,
  `parity_luck_two`), over `Z` or `Z/2³²`. In the product deficit (`parDef`, `ovDefP`) the staged and overlapped
  games meet Theorem S's bounds at `0.854` whatever the errors (`StagedPlay.game_stagedP`, `OvPlay.passBound_gameP`),
  and with charging purity in place of row purity (`SlpG.ChargePure`, `SlpG.ChargePureAt`) `RoundsHypKStP` and
  `RoundsHypKOvP` give `RoundsHyp` at `p = 0.854` with K's `F₀` (`roundsHyp_of_stagedP`, `roundsHyp_of_overlapP`;
  `certifiedFraction120θ_stagedP_40`, `_overlapP_40` at `D₀ = 274 ln 2`). The affine route is their sibling, proved
  on its own scope at ¾. These two are K's proved scopes (one-sided of any form and charge-pure at 0.854; affine and
  row-pure at ¾). Outside them `c = 1` is false even within the ledger's K1 (RS-E-REF r8 VIII): a factor
  `∏(1 − z_u)` in one operand leaves a word wrong only at `U = 0`, so it is uncertified and passes almost always, and
  K's count binds certified words only.
* Theorem H (round 6 §6, merges priced) is not needed for the re-encoded design: the chain lemma covers it in every
  arithmetic (RS-E-REF r4 IV.3). It stays proved as the alternative with merges priced, including `_H` of route (c).
  In the quad model, with merges at `h ≥ 2K/3` slots, each useful cell costs
  `K` slots (`theoremH`, from Theorem J and the rank bounds). Those premises hold on quad runs in exact arithmetic
  over a field (`QuadRun`; `QuadRun.theoremJ`, `full_ge`, `merged_ge`, from free-diagonal rank floors `PairRankGe`).
  On the re-encoded design, every quad run pays `L` slots a useful cell-block (`QuadRun.cost_ge_reEncoded`; at VF2's
  `h = 32`, for `L ≤ 48`, `_32`). `RoundsHypH` puts that in the cost line in exact arithmetic, with the same
  `F₀ = κ L n n' B` (`roundsHyp_of_H`, `certifiedFraction120θ_H`; over ℚ at `h = 32`,
  `certifiedFraction120θ_reEncodedExact_40`).
-/

namespace Pouw.RelativeSecurity

open Finset

section Probability

theorem prob_nonneg {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : 0 ≤ prob p := by
  unfold prob
  positivity

theorem prob_le_one {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : prob p ≤ 1 := by
  classical
  unfold prob
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast card_filter_le _ _

theorem prob_mono {Ω : Type*} [Fintype Ω] {p q : Ω → Prop} (h : ∀ ω, p ω → q ω) : prob p ≤ prob q := by
  classical
  unfold prob
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact_mod_cast card_le_card fun ω hω => by
    simp only [mem_filter, mem_univ, true_and] at hω ⊢
    exact h ω hω

theorem mem_filter_univ_iff {Ω : Type*} [Fintype Ω] {p : Ω → Prop} (inst : DecidablePred p) (ω : Ω) :
    ω ∈ @Finset.filter _ p inst univ ↔ p ω := by simp

theorem prob_or_le {Ω : Type*} [Fintype Ω] (p q : Ω → Prop) : prob (fun ω => p ω ∨ q ω) ≤ prob p + prob q := by
  classical
  unfold prob
  rw [← add_div]
  apply div_le_div_of_nonneg_right _ (by positivity)
  have key : ∀ s t u : Finset Ω, (∀ ω ∈ s, ω ∈ t ∨ ω ∈ u) → (s.card : ℝ) ≤ t.card + u.card := fun s t u h => by
    exact_mod_cast (card_le_card fun ω hω => mem_union.mpr (h ω hω)).trans (card_union_le t u)
  exact key _ _ _ fun ω hω => ((mem_filter_univ_iff _ ω).mp hω).imp ((mem_filter_univ_iff _ ω).mpr)
    ((mem_filter_univ_iff _ ω).mpr)

/-- The expectation of a real function under the uniform distribution on a finite type. -/
noncomputable def expect {Ω : Type*} [Fintype Ω] (f : Ω → ℝ) : ℝ := (∑ ω, f ω) / (Fintype.card Ω : ℝ)

/-- Markov's inequality for a nonnegative function. -/
theorem markov {Ω : Type*} [Fintype Ω] {f : Ω → ℝ} (hf : ∀ ω, 0 ≤ f ω) (X : ℝ) :
    X * prob (fun ω => X ≤ f ω) ≤ expect f := by
  classical
  unfold prob expect
  rw [mul_div_assoc']
  apply div_le_div_of_nonneg_right _ (by positivity)
  have key : ∀ s : Finset Ω, (∀ ω ∈ s, X ≤ f ω) → X * (s.card : ℝ) ≤ ∑ ω, f ω := fun s hs => by
    calc X * (s.card : ℝ) = ∑ _ω ∈ s, X := by rw [sum_const, nsmul_eq_mul, mul_comm]
      _ ≤ ∑ ω ∈ s, f ω := sum_le_sum hs
      _ ≤ ∑ ω, f ω := sum_le_sum_of_subset_of_nonneg (subset_univ _) fun ω _ _ => hf ω
  exact key _ fun ω hω => (mem_filter_univ_iff _ ω).mp hω

theorem expect_mono {Ω : Type*} [Fintype Ω] {f g : Ω → ℝ} (h : ∀ ω, f ω ≤ g ω) : expect f ≤ expect g :=
  div_le_div_of_nonneg_right (sum_le_sum fun ω _ => h ω) (by positivity)

theorem expect_mul_left {Ω : Type*} [Fintype Ω] (k : ℝ) (f : Ω → ℝ) :
    expect (fun ω => k * f ω) = k * expect f := by
  unfold expect
  rw [← mul_sum, mul_div_assoc]

theorem expect_sum {Ω ι : Type*} [Fintype Ω] (s : Finset ι) (f : ι → Ω → ℝ) :
    expect (fun ω => ∑ l ∈ s, f l ω) = ∑ l ∈ s, expect (f l) := by
  unfold expect
  rw [Finset.sum_comm, Finset.sum_div]

theorem expect_le_of_le {Ω : Type*} [Fintype Ω] {f : Ω → ℝ} {k : ℝ} (hk : 0 ≤ k) (h : ∀ ω, f ω ≤ k) :
    expect f ≤ k := by
  unfold expect
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
  · simp [h0, hk]
  · rw [div_le_iff₀ (by exact_mod_cast h0)]
    calc ∑ ω, f ω ≤ ∑ _ω : Ω, k := sum_le_sum fun ω _ => h ω
      _ = k * (Fintype.card Ω : ℝ) := by rw [sum_const, card_univ, nsmul_eq_mul, mul_comm]

/-- The expectation over a product is the iterated expectation (fresh coins). -/
theorem expect_prod {α β : Type*} [Fintype α] [Fintype β] (f : α × β → ℝ) :
    expect f = expect fun a => expect fun b => f (a, b) := by
  unfold expect
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul, ← sum_div, div_div, mul_comm (Fintype.card β : ℝ)]

theorem expect_prod_swap {α β : Type*} [Fintype α] [Fintype β] (f : α × β → ℝ) :
    expect f = expect fun b => expect fun a => f (a, b) := by
  unfold expect
  rw [Fintype.sum_prod_type, Finset.sum_comm, Fintype.card_prod, Nat.cast_mul, ← sum_div, div_div]

/-- A probability is the expectation of the event's indicator. -/
theorem prob_eq_expect {Ω : Type*} [Fintype Ω] (p : Ω → Prop) [DecidablePred p] :
    prob p = expect fun ω => if p ω then 1 else 0 := by
  classical
  unfold prob expect
  rw [Finset.card_filter, Nat.cast_sum]
  congr 1
  refine Finset.sum_congr rfl fun ω _ => ?_
  by_cases h : p ω <;> simp [h]

/-- An event whose every slice at the second coordinate has probability at most `κ ≥ 0` has probability at most
`κ`. -/
theorem prob_le_of_slices {α β : Type*} [Fintype α] [Fintype β] (E : α × β → Prop) {κ : ℝ} (hκ : 0 ≤ κ)
    (h : ∀ b, prob (fun a => E (a, b)) ≤ κ) : prob E ≤ κ := by
  classical
  rw [prob_eq_expect, expect_prod_swap]
  exact expect_le_of_le hκ fun b => by rw [← prob_eq_expect (fun a => E (a, b))]; exact h b

/-- A probability over a product is the mean of its slices at the second coordinate. -/
theorem prob_prod_eq_sum {α β : Type*} [Fintype α] [Fintype β] (E : α × β → Prop) :
    prob E = ∑ b, (Fintype.card β : ℝ)⁻¹ * prob (fun a => E (a, b)) := by
  classical
  rw [prob_eq_expect, expect_prod_swap, expect, div_eq_inv_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun b _ => by rw [prob_eq_expect (fun a => E (a, b))]

/-- Probabilities are invariant under a relabelling of the sample space. -/
theorem prob_equiv {α β : Type*} [Fintype α] [Fintype β] (e : α ≃ β) (p : α → Prop) :
    prob p = prob fun b => p (e.symm b) := by
  classical
  unfold prob
  rw [Fintype.card_congr e]
  congr 2
  exact Finset.card_equiv e fun a => by simp

/-- The union bound over a finite family of events. -/
theorem prob_exists_le {Ω ι : Type*} [Fintype Ω] (s : Finset ι) (B : ι → Ω → Prop) :
    prob (fun ω => ∃ w ∈ s, B w ω) ≤ ∑ w ∈ s, prob (B w) := by
  classical
  induction s using Finset.induction_on with
  | empty =>
    simp only [Finset.notMem_empty, false_and, exists_false, sum_empty]
    unfold prob
    simp
  | insert a s ha ih =>
    rw [Finset.sum_insert ha]
    refine (prob_mono fun ω hω => ?_).trans ((prob_or_le (B a) _).trans (add_le_add le_rfl ih))
    obtain ⟨w, hw, hB⟩ := hω
    rcases Finset.mem_insert.mp hw with rfl | hw
    · exact Or.inl hB
    · exact Or.inr ⟨w, hw, hB⟩

theorem one_sub_mul_exp_le (u : ℝ) : (1 - u) * Real.exp u ≤ 1 := by
  have h := Real.add_one_le_exp (-u)
  have h2 : Real.exp (-u) * Real.exp u = 1 := by rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  nlinarith [Real.exp_pos u]

theorem prob_add_not {Ω : Type*} [Fintype Ω] [Nonempty Ω] (q : Ω → Prop) :
    prob q + prob (fun ω => ¬q ω) = 1 := by
  classical
  unfold prob
  have := Finset.card_filter_add_card_filter_not (s := (univ : Finset Ω)) q
  rw [Finset.card_univ] at this
  have h' : ((univ.filter q).card : ℝ) + ((univ.filter fun ω => ¬q ω).card : ℝ) = Fintype.card Ω := by
    exact_mod_cast this
  have hC : (Fintype.card Ω : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  rw [← add_div, div_eq_one_iff_eq hC]
  convert h'

theorem prob_add_not_le {Ω : Type*} [Fintype Ω] (q : Ω → Prop) : prob q + prob (fun ω => ¬q ω) ≤ 1 := by
  rcases isEmpty_or_nonempty Ω with hΩ | hΩ
  · have h0 : ∀ q : Ω → Prop, prob q = 0 := fun q => by
      unfold prob
      simp
    rw [h0, h0]
    norm_num
  · exact (prob_add_not q).le

theorem expect_const_add_mul {Ω : Type*} [Fintype Ω] [Nonempty Ω] (A k : ℝ) (f : Ω → ℝ) :
    expect (fun ω => A + k * f ω) = A + k * expect f := by
  unfold expect
  have hC : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, add_div,
    mul_div_cancel_left₀ _ hC.ne', mul_div_assoc]

/-- The stage step of the supermartingale `(1 − U) e^{(1−p) D}`: a cheat step of size `c ≥ 0` that raises the wrong
share by `c` in expectation keeps it from growing (RS-E-REF r3 III.1). -/
theorem exp_mul_one_sub_le {U c : ℝ} (hU : 0 ≤ U) (hc : 0 ≤ c) : Real.exp c * (1 - U - c) ≤ 1 - U := by
  have h1 := one_sub_mul_exp_le c
  have h2 : 1 ≤ Real.exp c := Real.one_le_exp hc
  nlinarith

end Probability

section Rounds

/-! ### Theorem S: one spot check per round, fresh coins (luck-loss round 4 §1; guarded, round 5's Theorem S_G)

A window is `R` rounds. Round `w` draws fresh coins, uniform on `Ω`, and runs one check; the window accepts iff every
check passes. The prover's play is a `RoundGame`, a tree over the earlier rounds' coins, so each round's deficiency
and check may depend on the whole past. The deficiency is fixed before the round's coins (straight-line rounds,
`PassBound`) or read off the round's guard path (`PathBound`). The proof is the supermartingale
`Π_w 1[pass_w] e^{(1−p) d_w}` as a finite product over rounds (`expect_weight_le`), then Markov. -/

namespace RoundGame

variable {Ω : Type}

open Classical in
/-- The supermartingale's last value, `Π_w 1[pass_w] e^{c d_w}`. -/
noncomputable def weight (c : ℝ) : {n : ℕ} → RoundGame Ω n → Paths Ω n → ℝ
  | _, .done, _ => 1
  | _, .round d pass next, x => (if pass x.1 then Real.exp (c * d x.1) else 0) * (next x.1).weight c x.2

open Classical in
/-- The exponential form of the per-round bound (Part III.1): given the past, `E[1[pass_w] e^{c d_w}] ≤ 1`. -/
def ExpBound [Fintype Ω] (c : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
    expect (fun a => if pass a then Real.exp (c * d a) else 0) ≤ 1 ∧ ∀ a, (next a).ExpBound c

/-- **Theorem S's per-round hypothesis**, named: the round's deficiency is fixed before its coins (straight-line
rounds), and given the past the check passes with probability at most `1 − (1 − p) d_w`. -/
def PassBound [Fintype Ω] (p : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ d₀, (∀ a, d a = d₀) ∧ prob pass ≤ 1 - (1 - p) * d₀) ∧ ∀ a, (next a).PassBound p

/-- **Theorem S_G's per-round hypothesis**, named: the round's coins split into guard paths `path a ∈ Λ`, the
round's deficiency is its path's implied deficiency `dt`, and given the past and the path the check passes with
probability at most `1 − (1 − p) dt`. -/
def PathBound [Fintype Ω] (p : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ (Λ : Type) (_ : Fintype Λ) (path : Ω → Λ) (dt : Λ → ℝ), (∀ a, d a = dt (path a)) ∧
      ∀ l, prob (fun a => pass a ∧ path a = l) ≤ (1 - (1 - p) * dt l) * prob fun a => path a = l) ∧
      ∀ a, (next a).PathBound p

theorem weight_nonneg (c : ℝ) : ∀ {n : ℕ} (G : RoundGame Ω n) (x : Paths Ω n), 0 ≤ G.weight c x
  | _, .done, _ => by simp [weight]
  | _, .round d pass next, x => by
    simp only [weight]
    exact mul_nonneg (by split_ifs <;> positivity) (weight_nonneg c (next x.1) x.2)

theorem weight_of_accepts (c : ℝ) : ∀ {n : ℕ} (G : RoundGame Ω n) (x : Paths Ω n), G.accepts x →
    G.weight c x = Real.exp (c * G.deficiency x)
  | _, .done, _, _ => by simp [weight, deficiency]
  | _, .round d pass next, x, h => by
    obtain ⟨h1, h2⟩ := h
    simp only [weight, deficiency, h1, ↓reduceIte, weight_of_accepts c (next x.1) x.2 h2, mul_add, Real.exp_add]

/-- **The supermartingale bound**, as a finite product over rounds: `E[Π_w 1[pass_w] e^{c d_w}] ≤ 1`. -/
theorem expect_weight_le [Fintype Ω] (c : ℝ) : ∀ {n : ℕ} (G : RoundGame Ω n), G.ExpBound c →
    expect (G.weight c) ≤ 1
  | _, .done, _ => by simp [expect, weight]
  | _, .round d pass next, h => by
    classical
    obtain ⟨h1, h2⟩ := h
    rw [expect_prod]
    calc expect (fun a => expect fun y => (round d pass next).weight c (a, y))
        = expect (fun a => (if pass a then Real.exp (c * d a) else 0) * expect ((next a).weight c)) := by
          congr 1
          funext a
          simp only [weight]
          exact expect_mul_left _ _
      _ ≤ expect (fun a => if pass a then Real.exp (c * d a) else 0) :=
          expect_mono fun a => mul_le_of_le_one_right (by split_ifs <;> positivity)
            (expect_weight_le c (next a) (h2 a))
      _ ≤ 1 := h1

/-- The per-round pass bound gives the exponential form at `c = 1 − p`. -/
theorem expBound_of_passBound [Fintype Ω] {p : ℝ} : ∀ {n : ℕ} (G : RoundGame Ω n), G.PassBound p →
    G.ExpBound (1 - p)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨⟨d₀, hd, hpass⟩, h2⟩ := h
    refine ⟨?_, fun a => expBound_of_passBound (next a) (h2 a)⟩
    have e : (fun a => if pass a then Real.exp ((1 - p) * d a) else 0) =
        fun a => Real.exp ((1 - p) * d₀) * if pass a then 1 else 0 := by
      funext a
      rw [hd a]
      split_ifs <;> simp
    rw [e, expect_mul_left, ← prob_eq_expect]
    calc Real.exp ((1 - p) * d₀) * prob pass ≤ Real.exp ((1 - p) * d₀) * (1 - (1 - p) * d₀) :=
          mul_le_mul_of_nonneg_left hpass (Real.exp_pos _).le
      _ ≤ 1 := by rw [mul_comm]; exact one_sub_mul_exp_le _

/-- The per-path pass bound gives the exponential form at `c = 1 − p` (round 5 §1(a)). -/
theorem expBound_of_pathBound [Fintype Ω] {p : ℝ} : ∀ {n : ℕ} (G : RoundGame Ω n), G.PathBound p →
    G.ExpBound (1 - p)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨⟨Λ, _, path, dt, hd, hl⟩, h2⟩ := h
    refine ⟨?_, fun a => expBound_of_pathBound (next a) (h2 a)⟩
    have e : (fun a => if pass a then Real.exp ((1 - p) * d a) else 0) =
        fun a => ∑ l, Real.exp ((1 - p) * dt l) * if pass a ∧ path a = l then 1 else 0 := by
      funext a
      rw [Finset.sum_eq_single_of_mem (path a) (mem_univ _) fun l _ hl => by
        simp [Ne.symm hl]]
      rw [hd a]
      by_cases hp : pass a <;> simp [hp]
    have e2 : (fun _ : Ω => (1 : ℝ)) = fun a => ∑ l, if path a = l then 1 else 0 := by
      funext a
      rw [Finset.sum_eq_single_of_mem (path a) (mem_univ _) fun l _ hl => by simp [Ne.symm hl]]
      simp
    rw [e, expect_sum]
    calc ∑ l, expect (fun a => Real.exp ((1 - p) * dt l) * if pass a ∧ path a = l then 1 else 0)
        = ∑ l, Real.exp ((1 - p) * dt l) * prob (fun a => pass a ∧ path a = l) := by
          refine Finset.sum_congr rfl fun l _ => ?_
          rw [expect_mul_left, prob_eq_expect]
      _ ≤ ∑ l, Real.exp ((1 - p) * dt l) * ((1 - (1 - p) * dt l) * prob fun a => path a = l) :=
          Finset.sum_le_sum fun l _ => mul_le_mul_of_nonneg_left (hl l) (Real.exp_pos _).le
      _ ≤ ∑ l, prob fun a => path a = l := Finset.sum_le_sum fun l _ => by
          rw [← mul_assoc, mul_comm (Real.exp _)]
          exact mul_le_of_le_one_left (prob_nonneg _) (one_sub_mul_exp_le _)
      _ = expect fun _ : Ω => (1 : ℝ) := by
          rw [e2, expect_sum]
          exact Finset.sum_congr rfl fun l _ => prob_eq_expect _
      _ ≤ 1 := expect_le_of_le zero_le_one fun _ => le_rfl

/-- **Theorem S from the exponential form**: for `c ≥ 0`, `Pr[accept ∧ Σ_w d_w ≥ D] ≤ e^{−c D}` (Markov on the
supermartingale's last value, which is `e^{c Σ d}` on acceptance). -/
theorem prob_accepts_le [Fintype Ω] {c : ℝ} (hc : 0 ≤ c) {n : ℕ} (G : RoundGame Ω n) (hG : G.ExpBound c) (D : ℝ) :
    prob (fun x => G.accepts x ∧ D ≤ G.deficiency x) ≤ Real.exp (-(c * D)) := by
  have hm := markov (G.weight_nonneg c) (Real.exp (c * D))
  have hsub : prob (fun x => G.accepts x ∧ D ≤ G.deficiency x) ≤
      prob fun x => Real.exp (c * D) ≤ G.weight c x :=
    prob_mono fun x hx => by
      rw [G.weight_of_accepts c x hx.1]
      exact Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left hx.2 hc)
  have hpos := Real.exp_pos (c * D)
  calc prob (fun x => G.accepts x ∧ D ≤ G.deficiency x)
      ≤ prob fun x => Real.exp (c * D) ≤ G.weight c x := hsub
    _ = (Real.exp (c * D))⁻¹ * (Real.exp (c * D) * prob fun x => Real.exp (c * D) ≤ G.weight c x) :=
        (inv_mul_cancel_left₀ hpos.ne' _).symm
    _ ≤ (Real.exp (c * D))⁻¹ * 1 :=
        mul_le_mul_of_nonneg_left (hm.trans (G.expect_weight_le c hG)) (inv_nonneg.mpr hpos.le)
    _ = Real.exp (-(c * D)) := by rw [mul_one, Real.exp_neg]

/-- Theorem S with binding and zero knowledge: if every accepted run is accepted by the game or breaks binding or
zero knowledge in some round, the union bound adds `R (δb + δz)`. -/
theorem prob_accept_le [Fintype Ω] {c : ℝ} (hc : 0 ≤ c) {R : ℕ} (G : RoundGame Ω R) (hG : G.ExpBound c)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (D : ℝ) :
    prob (fun x => acc x ∧ D ≤ G.deficiency x) ≤ Real.exp (-(c * D)) + R * (δb + δz) := by
  have hbad : prob (fun x => ∃ w ∈ (univ : Finset (Fin R)), Bind w x ∨ ZK w x) ≤ R * (δb + δz) := by
    refine (prob_exists_le _ _).trans ?_
    calc ∑ w : Fin R, prob (fun x => Bind w x ∨ ZK w x) ≤ ∑ _w : Fin R, (δb + δz) :=
          Finset.sum_le_sum fun w _ => (prob_or_le _ _).trans (add_le_add (hBind w) (hZK w))
      _ = R * (δb + δz) := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  refine (prob_mono fun x hx => ?_).trans ((prob_or_le _ _).trans (add_le_add (G.prob_accepts_le hc hG D) hbad))
  rcases hacc x hx.1 with h | ⟨w, hw⟩
  · exact Or.inl ⟨h, hx.2⟩
  · exact Or.inr ⟨w, mem_univ _, hw⟩

/-- **Theorem S** (luck-loss round 4 §1, refereed in Part III): over `R` rounds with fresh coins, one check a round,
accepting iff every check passes, given the per-round pass bound `1 − (1 − p) d_w` (`PassBound`, named) and
binding and zero-knowledge failures of probability at most `δb` and `δz` per round,
`Pr[accept ∧ Σ_w d_w ≥ D] ≤ e^{−(1−p) D} + R (δb + δz)`. -/
theorem theoremS [Fintype Ω] {p : ℝ} (hp : p ≤ 1) {R : ℕ} (G : RoundGame Ω R) (hG : G.PassBound p)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (D : ℝ) :
    prob (fun x => acc x ∧ D ≤ G.deficiency x) ≤ Real.exp (-((1 - p) * D)) + R * (δb + δz) :=
  G.prob_accept_le (by linarith) (G.expBound_of_passBound hG) acc Bind ZK hBind hZK hacc D

/-- **Theorem S_G** (round 5 §1(a)): the same bound with the deficiency read off each round's guard path
(`PathBound`, named), so `Σ_w d̃_w` is the implied deficiency along the realized paths. -/
theorem theoremS_G [Fintype Ω] {p : ℝ} (hp : p ≤ 1) {R : ℕ} (G : RoundGame Ω R) (hG : G.PathBound p)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (D : ℝ) :
    prob (fun x => acc x ∧ D ≤ G.deficiency x) ≤ Real.exp (-((1 - p) * D)) + R * (δb + δz) :=
  G.prob_accept_le (by linarith) (G.expBound_of_pathBound hG) acc Bind ZK hBind hZK hacc D

/-- **The window form from the exponential bound**: the per-round `E[1[pass_w] e^{(1−p) d_w}] ≤ 1` (`ExpBound`) is
all the window form reads, so rounds with several draws inside (`RoundGame.Staged`) have it too. -/
theorem window_le_exp [Fintype Ω] {p : ℝ} (hp : p ≤ 1) {R : ℕ} (G : RoundGame Ω R) (hG : G.ExpBound (1 - p))
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (cost V : Paths Ω R → ℝ) {F₀ sl : ℝ}
    (hF₀ : 0 < F₀) (hcost : ∀ x, (R - G.deficiency x) * F₀ - V x - R * sl ≤ cost x) (D : ℝ) :
    prob (fun x => acc x ∧ cost x < (R - D) * F₀ - V x - R * sl) ≤
      Real.exp (-((1 - p) * D)) + R * (δb + δz) := by
  refine (prob_mono fun x hx => ⟨hx.1, ?_⟩).trans (G.prob_accept_le (by linarith) hG acc Bind ZK hBind hZK hacc D)
  have h := (hcost x).trans_lt hx.2
  have h' : (R - G.deficiency x) * F₀ < (R - D) * F₀ := by linarith
  have := (mul_lt_mul_iff_of_pos_right hF₀).mp h'
  linarith

/-- **The window form** (Theorem S's cost line, and S_G's with Lemma RD): if every run's cost is at least
`(R − Σ_w d_w) F₀ − V − R sl` (Theorem E½'s per-round `x_w F₀ − L(M + K)` summed, with `V` the noise reads priced
on the realized paths, `0` for straight-line rounds), then accepted runs costing below `(R − D) F₀ − V − R sl` have
probability at most `e^{−(1−p) D} + R (δb + δz)`. -/
theorem window_le [Fintype Ω] {p : ℝ} (hp : p ≤ 1) {R : ℕ} (G : RoundGame Ω R) (hG : G.PathBound p)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (cost V : Paths Ω R → ℝ) {F₀ sl : ℝ}
    (hF₀ : 0 < F₀) (hcost : ∀ x, (R - G.deficiency x) * F₀ - V x - R * sl ≤ cost x) (D : ℝ) :
    prob (fun x => acc x ∧ cost x < (R - D) * F₀ - V x - R * sl) ≤
      Real.exp (-((1 - p) * D)) + R * (δb + δz) :=
  G.window_le_exp hp (G.expBound_of_pathBound hG) acc Bind ZK hBind hZK hacc cost V hF₀ hcost D

/-- Theorem S's straight-line rounds are guarded rounds with one path. -/
theorem pathBound_of_passBound [Fintype Ω] {p : ℝ} : ∀ {n : ℕ} (G : RoundGame Ω n), G.PassBound p →
    G.PathBound p
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    obtain ⟨⟨d₀, hd, hpass⟩, h2⟩ := h
    refine ⟨⟨Unit, inferInstance, fun _ => (), fun _ => d₀, hd, fun l => ?_⟩,
      fun a => pathBound_of_passBound (next a) (h2 a)⟩
    have e1 : prob (fun a => pass a ∧ () = l) = prob pass := by congr 1; funext a; simp
    have e2 : prob (fun _ : Ω => () = l) = prob fun _ : Ω => True := by
      congr 1
      funext a
      exact eq_iff_iff.mpr ⟨fun _ => trivial, fun _ => Subsingleton.elim _ _⟩
    rw [e1, e2]
    rcases isEmpty_or_nonempty Ω with hΩ | hΩ
    · have h0 : ∀ q : Ω → Prop, prob q = 0 := fun q => by
        unfold prob
        simp
      rw [h0, h0, mul_zero]
    · have h1 : prob (fun _ : Ω => True) = 1 := by
        unfold prob
        simp
      rw [h1, mul_one]
      exact hpass

/-- Round `w`'s deficiency along the realized coins. -/
def dAt : {n : ℕ} → RoundGame Ω n → Paths Ω n → Fin n → ℝ
  | _, .done, _, w => w.elim0
  | _, .round d _ next, x, w => Fin.cases (motive := fun _ => ℝ) (d x.1) (fun w' => (next x.1).dAt x.2 w') w

theorem deficiency_eq_sum : ∀ {n : ℕ} (G : RoundGame Ω n) (x : Paths Ω n), G.deficiency x = ∑ w, G.dAt x w
  | _, .done, _ => by simp [deficiency]
  | _, .round d _ next, x => by
    rw [Fin.sum_univ_succ]
    simp only [deficiency, dAt, Fin.cases_zero, Fin.cases_succ]
    rw [deficiency_eq_sum (next x.1) x.2]

end RoundGame

end Rounds

section Stages

/-! ### Stages inside a round (RS-E-REF r3 III.1's extension; decision 36, the stage-coins draft
§2, as RS-E-REF r5 V.3 repaired it)

A round may be a sequence of stages, each committing its operands, then drawing both noise blocks of its products
(keyed by the draw), then committing its outputs, with one check at the round's end over all of them. Each stage's
cheat step `a` is fixed before its draw, and its draw raises the round's wrong share `U` by `u` with
`E[u] ≥ (1 − p) a`. Then `Φ(U, D) = (1 − U) e^{(1−p) D}` is a supermartingale over the stages (`StageGame.expect_le`),
so a round whose check passes with probability at most `1 − U` meets the exponential bound with `d_w = Σ_s a_s`
(`RoundGame.Staged`, `RoundGame.expBound_of_staged`), which is all the window form reads (`window_le_exp`). -/

namespace StageGame

variable {Ω : Type}

/-- The round's wrong share `U = Σ_s u_s` along its draws. -/
def wrong : {m : ℕ} → StageGame Ω m → Paths Ω m → ℝ
  | _, .done, _ => 0
  | _, .stage _ u next, x => u x.1 + (next x.1).wrong x.2

/-- Every stage's step is nonnegative and fixed before its draw, which raises the wrong share by at least
`(1 − p) a` in expectation. -/
def Valid [Fintype Ω] (p : ℝ) : {m : ℕ} → StageGame Ω m → Prop
  | _, .done => True
  | _, .stage a u next => 0 ≤ a ∧ (∀ ω, 0 ≤ u ω) ∧ (1 - p) * a ≤ expect u ∧ ∀ ω, (next ω).Valid p

/-- **The supermartingale over stages** (RS-E-REF r3 III.1): from wrong share `U₀ ≥ 0`,
`E[(1 − U₀ − U) e^{(1−p) D}] ≤ 1 − U₀` under any adaptive play. -/
theorem expect_le [Fintype Ω] [Nonempty Ω] {p : ℝ} (hp : p ≤ 1) : ∀ {m : ℕ} (G : StageGame Ω m), G.Valid p →
    ∀ U₀ : ℝ, 0 ≤ U₀ → expect (fun x => (1 - U₀ - G.wrong x) * Real.exp ((1 - p) * G.defi x)) ≤ 1 - U₀
  | _, .done, _, U₀, _ => by simp [expect, wrong, defi]
  | _, .stage a u next, h, U₀, hU₀ => by
    obtain ⟨ha, hu, hau, hnext⟩ := h
    rw [expect_prod]
    calc expect (fun ω => expect fun y => (1 - U₀ - (stage a u next).wrong (ω, y)) *
            Real.exp ((1 - p) * (stage a u next).defi (ω, y)))
        = expect (fun ω => Real.exp ((1 - p) * a) * expect fun y =>
            (1 - (U₀ + u ω) - (next ω).wrong y) * Real.exp ((1 - p) * (next ω).defi y)) := by
          congr 1
          funext ω
          rw [← expect_mul_left]
          congr 1
          funext y
          simp only [wrong, defi, mul_add, Real.exp_add]
          ring
      _ ≤ expect (fun ω => Real.exp ((1 - p) * a) * (1 - (U₀ + u ω))) :=
          expect_mono fun ω => mul_le_mul_of_nonneg_left
            (expect_le hp (next ω) (hnext ω) _ (add_nonneg hU₀ (hu ω))) (Real.exp_pos _).le
      _ = Real.exp ((1 - p) * a) * (1 - U₀) + -Real.exp ((1 - p) * a) * expect u := by
          rw [← expect_const_add_mul]
          congr 1
          funext ω
          ring
      _ ≤ Real.exp ((1 - p) * a) * (1 - U₀ - (1 - p) * a) := by nlinarith [Real.exp_pos ((1 - p) * a)]
      _ ≤ 1 - U₀ := exp_mul_one_sub_le hU₀ (mul_nonneg (by linarith) ha)

end StageGame

open Classical in
/-- **A window of staged rounds**: each round's coins are its stages' draws and the check's coins `J`; given the
past, the round's deficiency is its stage game's (fixed by the draws), and given the draws its check passes with
probability at most `1 − U`. -/
def RoundGame.Staged {Ωs J : Type} [Fintype Ωs] [Fintype J] (p : ℝ) (m : ℕ) :
    {n : ℕ} → RoundGame (Paths Ωs m × J) n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ SG : StageGame Ωs m, SG.Valid p ∧ (∀ x j, d (x, j) = SG.defi x) ∧
      ∀ x, prob (fun j => pass (x, j)) ≤ 1 - SG.wrong x) ∧ ∀ a, (next a).Staged p m

/-- **Staged rounds meet the exponential bound** (RS-E-REF r3 III.1's extension): Theorem S holds verbatim with
`d_w = Σ_s a_s` the realized round deficiency. -/
theorem RoundGame.expBound_of_staged {Ωs J : Type} [Fintype Ωs] [Nonempty Ωs] [Fintype J] {p : ℝ} (hp : p ≤ 1)
    {m : ℕ} : ∀ {n : ℕ} (G : RoundGame (Paths Ωs m × J) n), G.Staged p m → G.ExpBound (1 - p)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨⟨SG, hSG, hd, hpass⟩, h2⟩ := h
    refine ⟨?_, fun a => expBound_of_staged hp (next a) (h2 a)⟩
    rw [expect_prod]
    calc expect (fun x => expect fun j => if pass (x, j) then Real.exp ((1 - p) * d (x, j)) else 0)
        = expect (fun x => Real.exp ((1 - p) * SG.defi x) * prob fun j => pass (x, j)) := by
          congr 1
          funext x
          rw [prob_eq_expect, ← expect_mul_left]
          congr 1
          funext j
          rw [hd x j]
          split_ifs <;> simp
      _ ≤ expect (fun x => (1 - 0 - SG.wrong x) * Real.exp ((1 - p) * SG.defi x)) :=
          expect_mono fun x => by
            rw [mul_comm, sub_zero]
            exact mul_le_mul_of_nonneg_right (hpass x) (Real.exp_pos _).le
      _ ≤ 1 - 0 := SG.expect_le hp hSG 0 le_rfl
      _ = 1 := sub_zero 1

end Stages

section Machine

/-- `n × m` matrices over `R`, indexed by `Fin`. -/
abbrev Mat (R : Type) (a b : ℕ) : Type := Matrix (Fin a) (Fin b) R

/-- A machine model with matrix arithmetic over a ring `R`: the cost of a linear map, `X ↦ X Y` for an `a × b` by
`b × c` product, is at most `κ a b c` (`κ` the price of a multiply-accumulate), and an entrywise addition or
subtraction of `a × b` matrices costs at most `κadd a b`. -/
structure MatMachine (R : Type) [Ring R] extends Machine where
  κ : ℝ
  κadd : ℝ
  mul : ∀ a b c : ℕ, Alg (Mat R a b × Mat R b c) (Mat R a c)
  add : ∀ a b : ℕ, Alg (Mat R a b × Mat R a b) (Mat R a b)
  sub : ∀ a b : ℕ, Alg (Mat R a b × Mat R a b) (Mat R a b)
  eval_mul : ∀ (a b c : ℕ) (x : Mat R a b × Mat R b c), eval (mul a b c) x = x.1 * x.2
  eval_add : ∀ (a b : ℕ) (x : Mat R a b × Mat R a b), eval (add a b) x = x.1 + x.2
  eval_sub : ∀ (a b : ℕ) (x : Mat R a b × Mat R a b), eval (sub a b) x = x.1 - x.2
  /-- The cost of a linear map. -/
  time_mul_le : ∀ a b c : ℕ, time (mul a b c) ≤ κ * ((a : ℝ) * b * c)
  time_add_le : ∀ a b : ℕ, time (add a b) ≤ κadd * ((a : ℝ) * b)
  time_sub_le : ∀ a b : ℕ, time (sub a b) ≤ κadd * ((a : ℝ) * b)

namespace Machine

variable (Mc : Machine)

/-- Run `f` and `g` on the same input and combine their outputs with `h`. -/
def lift2 {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) : Mc.Alg I Q :=
  Mc.seq (Mc.pair f g) h

theorem eval_lift2 {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) (x : I) :
    Mc.eval (Mc.lift2 f g h) x = Mc.eval h (Mc.eval f x, Mc.eval g x) := by
  rw [lift2, Mc.eval_seq, Mc.eval_pair]

theorem time_lift2_le {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) :
    Mc.time (Mc.lift2 f g h) ≤ Mc.time f + Mc.time g + Mc.time h :=
  (Mc.time_seq_le _ _).trans (by linarith [Mc.time_pair_le f g])

theorem time_seq_eq_zero {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg O P} (hf : Mc.time f = 0)
    (hg : Mc.time g = 0) : Mc.time (Mc.seq f g) = 0 :=
  le_antisymm ((Mc.time_seq_le f g).trans (by rw [hf, hg, add_zero])) (Mc.time_nonneg _)

/-! Cost bounds in goal-directed form, for `apply_rules`. -/

theorem time_seq_le_of {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg O P} {a b : ℝ} (hf : Mc.time f ≤ a)
    (hg : Mc.time g ≤ b) : Mc.time (Mc.seq f g) ≤ a + b :=
  (Mc.time_seq_le f g).trans (add_le_add hf hg)

theorem time_pair_le_of {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg I P} {a b : ℝ} (hf : Mc.time f ≤ a)
    (hg : Mc.time g ≤ b) : Mc.time (Mc.pair f g) ≤ a + b :=
  (Mc.time_pair_le f g).trans (add_le_add hf hg)

theorem time_fst_le {I J : Type} : Mc.time (Mc.fst (I := I) (J := J)) ≤ 0 := Mc.time_fst.le

theorem time_snd_le {I J : Type} : Mc.time (Mc.snd (I := I) (J := J)) ≤ 0 := Mc.time_snd.le

theorem charge_le_time {I O : Type} (f : Mc.Alg I O) (x : I) : Mc.charge f x ≤ Mc.time f :=
  max_le (Mc.size_le_time f) (Mc.cost_le_time f x)

theorem charge_nonneg {I O : Type} (f : Mc.Alg I O) (x : I) : 0 ≤ Mc.charge f x :=
  le_max_of_le_left (Mc.size_nonneg f)

theorem size_le_charge {I O : Type} (f : Mc.Alg I O) (x : I) : Mc.size f ≤ Mc.charge f x := le_max_left _ _

end Machine

/-- The axioms are satisfiable with nontrivial prices: algorithms are functions annotated with a cost, composition
adds costs, routing is free, and an `a × b` by `b × c` product costs exactly `a b c`. (This model has no floor: any
function is an algorithm. A floor needs a syntactic model, RS-MODEL's.) -/
def costed : Machine where
  Alg I O := (I → O) × ℕ
  eval f := f.1
  time f := f.2
  cost f _ := f.2
  size f := f.2
  seq f g := (g.1 ∘ f.1, f.2 + g.2)
  pair f g := (fun x => (f.1 x, g.1 x), f.2 + g.2)
  fst := (Prod.fst, 0)
  snd := (Prod.snd, 0)
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := Nat.cast_nonneg _
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := Nat.cast_nonneg _
  time_seq_le _ _ := by push_cast; rfl
  le_time_seq _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  time_pair_le _ _ := by push_cast; rfl
  le_time_pair_left _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  le_time_pair_right f _ := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) f.2]
  time_fst := by simp
  time_snd := by simp

/-- `costed` with matrix arithmetic over a ring: `κ = κadd = 1`. -/
def costedMat (R : Type) [Ring R] : MatMachine R where
  toMachine := costed
  κ := 1
  κadd := 1
  mul a b c := (fun x => x.1 * x.2, a * b * c)
  add a b := (fun x => x.1 + x.2, a * b)
  sub a b := (fun x => x.1 - x.2, a * b)
  eval_mul _ _ _ _ := rfl
  eval_add _ _ _ := rfl
  eval_sub _ _ _ := rfl
  time_mul_le a b c := by simp [costed]
  time_add_le a b := by simp [costed]
  time_sub_le a b := by simp [costed]

end Machine

section Definition

/-- A proof-of-useful-work protocol in a machine model, for instances `Inst` with answers `Out` (RS-DEF §1.3, one
round): the prover sends a commitment to its instance before any coin, the verifier sends live coins σ, the prover
answers, and the verifier decides, deterministically in its view. `Solve` also outputs `C`, kept by the prover. The
`|π| ≤ 10λ` bound of Neekon's draft is dropped, as RS-DEF does (security never uses it). -/
structure Protocol (Mc : Machine) (Inst Out : Type) where
  /-- The verifier's live coins σ, `λ` bits. -/
  Seed : Type
  [instSeed : Fintype Seed]
  lam : ℕ
  card_seed : Fintype.card Seed = 2 ^ lam
  /-- The prover's commitment `c`, sent before σ. -/
  Com : Type
  /-- The prover's answer `a` to σ. -/
  Ans : Type
  /-- Solve's coins `ρ`. -/
  Coins : Type
  [instCoins : Fintype Coins]
  /-- `C` answers `x` against σ (for matmul: `A · B = C` on the exact grid; for an edited instance, route (c), the
  edited product `(A + E_A(σ)) (B + E_B(σ)) = C`). -/
  correct : Seed → Inst → Out → Prop
  /-- The promise (RS-DEF v2 §1.4, `A, B ∈ G_use`): usefulness and completeness quantify over the promised instances
  only, so a verifier may range-check what is committed. -/
  promise : Inst → Prop := fun _ => True
  /-- `Solve(x; ρ)` against σ: `((c, a), C)`. -/
  solve : Mc.Alg (Seed × Inst × Coins) ((Com × Ans) × Out)
  /-- `V(c, σ, a) → b`, deterministic and not priced. -/
  verify : Com → Seed → Ans → Bool

attribute [instance] Protocol.instSeed Protocol.instCoins

/-- A transcript's prover messages `(c, a)`. -/
abbrev Protocol.Proof {Mc : Machine} {Inst Out : Type} (P : Protocol Mc Inst Out) : Type := P.Com × P.Ans

/-- The verifier accepts the prover messages `π = (c, a)` against σ. -/
def Protocol.accepts {Mc : Machine} {Inst Out : Type} (P : Protocol Mc Inst Out) (σ : P.Seed) (π : P.Proof) : Bool :=
  P.verify π.1 σ π.2

variable {Mc : Machine} {Inst Out : Type}

/-- **Live coins** (RS-DEF's T6): the commitment a prover sends doesn't depend on σ. -/
def CommitsFirst (P : Protocol Mc Inst Out) {X : Type} (f : P.Seed × X → P.Proof) : Prop :=
  ∀ σ σ' x, (f (σ, x)).1 = (f (σ', x)).1

/-- A useful prover `M'(x; r)` against σ answers `x` correctly and is accepted, with probability at least `1 - η` over
σ and its coins, for every promised instance (RS-DEF §1.4's event `b = 1 ∧ C = AB`). -/
def SolvesAll (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out))
    (η : ℝ) : Prop :=
  ∀ x : Inst, P.promise x → 1 - η ≤ prob fun ω : P.Seed × Ω =>
    P.correct ω.1 x (Mc.eval M' (ω.1, x, ω.2)).2 ∧ P.accepts ω.1 (Mc.eval M' (ω.1, x, ω.2)).1 = true

/-- A useful prover: it commits first and solves every instance. -/
def Useful (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out))
    (η : ℝ) : Prop :=
  CommitsFirst P (fun y => (Mc.eval M' y).1) ∧ SolvesAll P M' η

/-- `p(M)`: the probability that an adversary `M(σ; r)`, with no input, is accepted. -/
noncomputable def acceptance (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof) :
    ℝ :=
  prob fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true

/-- **Completeness** (RS-DEF §1.4): Solve commits first and answers every instance with a proof that verifies, with
probability at least `1 - η`. -/
def Complete (P : Protocol Mc Inst Out) (η : ℝ) : Prop :=
  Useful P P.solve η

/-- **Security, Neekon's form**: every adversary `M` that commits first and is accepted with probability at least `δ`
gives a useful `M'` with `time M' ≤ (1 + ε) time M + α`. -/
def Secure (P : Protocol Mc Inst Out) (ε δ η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    δ ≤ acceptance P M →
    ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
      Mc.time M' ≤ (1 + ε) * Mc.time M + α ∧ Useful P M' η

/-- `T_use(η)` (RS-DEF §1.5): the least time of a useful prover. -/
noncomputable def Tuse (P : Protocol Mc Inst Out) (η : ℝ) : ℝ :=
  sInf {t | ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
    Useful P M' η ∧ Mc.time M' = t}

theorem tuse_le (P : Protocol Mc Inst Out) {η : ℝ} {Ω' : Type} [Fintype Ω']
    {M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)} (h : Useful P M' η) : Tuse P η ≤ Mc.time M' :=
  csInf_le ⟨0, fun _ ⟨_, _, M'', _, ht⟩ => ht ▸ Mc.time_nonneg M''⟩ ⟨Ω', inferInstance, M', h, rfl⟩

/-- **Security, RS-DEF's frozen form** (§1.5): every adversary that commits first and is accepted with probability at
least `δ` has `(1 + ε) time M + α ≥ T_use(η)`.

**Superseded by `SecureInfPerProver`** (proposed, a claim change): one `δ` for every adversary is reached only
through a floor with one `κ` for every adversary, which a service on SHA-512 commitments meets only at 1. -/
def SecureInf (P : Protocol Mc Inst Out) (ε δ η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    δ ≤ acceptance P M → Tuse P η ≤ (1 + ε) * Mc.time M + α

theorem secureInf_of_secure (P : Protocol Mc Inst Out) {ε δ η α : ℝ} (h : Secure P ε δ η α) :
    SecureInf P ε δ η α := fun Ω _ M hc hM => by
  obtain ⟨_, _, M', ht, hu⟩ := h Ω M hc hM
  exact (tuse_le P hu).trans ht

/-- **The per-run core** (RS-DEF v2 §1.6, the proof target): for every adversary that commits first, the runs that
are accepted and whose charge `max(|M|, cost)` is below `(T_use(η) − α)/(1 + ε)` have probability at most `κ`.
Superseded, with `AdvFloor`, by the floor per adversary (`AdvFloorPerProver`, `secureInfPerProver_of_floor`). -/
def Core (P : Protocol Mc Inst Out) (ε κ η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧
      Mc.charge M ω < (Tuse P η - α) / (1 + ε)) ≤ κ

/-- **Core ⇒ Security** for every `δ > κ`: some accepted run costs at least the bound, and `time` bounds every run. -/
theorem secureInf_of_core (P : Protocol Mc Inst Out) {ε κ δ η α : ℝ} (h : Core P ε κ η α) (hκδ : κ < δ)
    (hε : 0 < 1 + ε) : SecureInf P ε δ η α := by
  intro Ω _ M hc hM
  by_contra hlt
  replace hlt := not_le.mp hlt
  have hbound : Mc.time M < (Tuse P η - α) / (1 + ε) := by
    rw [lt_div_iff₀ hε]
    linarith
  have hsub := prob_mono (Ω := P.Seed × Ω) (p := fun ω => P.accepts ω.1 (Mc.eval M ω) = true)
    (q := fun ω => P.accepts ω.1 (Mc.eval M ω) = true ∧ Mc.charge M ω < (Tuse P η - α) / (1 + ε))
    fun ω hω => ⟨hω, (Mc.charge_le_time M ω).trans_lt hbound⟩
  have := h Ω M hc
  unfold acceptance at hM
  linarith

/-- **The rate form** (RS-DEF v2 §1.6): under Core, an adversary that commits first is charged in expectation at least
`(p(M) − κ) (T_use(η) − α)/(1 + ε)` per run (Markov on the charge). -/
theorem rate_of_core (P : Protocol Mc Inst Out) {ε κ η α : ℝ} (h : Core P ε κ η α)
    (hX : 0 ≤ (Tuse P η - α) / (1 + ε)) {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof)
    (hc : CommitsFirst P (Mc.eval M)) :
    (acceptance P M - κ) * ((Tuse P η - α) / (1 + ε)) ≤ expect (Mc.charge M) := by
  have hcore := h Ω M hc
  set X := (Tuse P η - α) / (1 + ε)
  have hsplit : acceptance P M ≤ prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧
      Mc.charge M ω < X) + prob fun ω => X ≤ Mc.charge M ω := by
    unfold acceptance
    exact (prob_mono fun ω hω => (lt_or_ge (Mc.charge M ω) X).imp (fun hl => ⟨hω, hl⟩) id).trans
      (prob_or_le _ _)
  calc (acceptance P M - κ) * X ≤ prob (fun ω => X ≤ Mc.charge M ω) * X :=
        mul_le_mul_of_nonneg_right (by linarith) hX
    _ = X * prob (fun ω => X ≤ Mc.charge M ω) := mul_comm _ _
    _ ≤ expect (Mc.charge M) := markov (fun ω => Mc.charge_nonneg M ω) X

/-- **The adversary floor** (a named hypothesis, the lane's floor results with Theorem F's luck term in `κ`): for every
adversary that commits first, the runs that are accepted and whose charge `max(|M|, cost)` is below `F` have
probability at most `κ` (RS-DEF v2's Lemma R premise).

**Superseded by `AdvFloorPerProver`**: over a service on SHA-512 commitments it holds only at `κ = 1`, since an
adversary that hard-codes a collision opens a committed leaf two ways (a red team's
side finding). -/
def AdvFloor (P : Protocol Mc Inst Out) (F κ : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧ Mc.charge M ω < F) ≤ κ

/-- **Lemma R** (RS-DEF §1.6: a floor plus an honest upper bound give the core): under the adversary floor `F`, if some
useful prover costs at most `U`, then Core holds with `1 + ε = U / F` and `α = 0`. -/
theorem core_of_floor (P : Protocol Mc Inst Out) {κ η F U : ℝ} (hF : 0 < F) (hFU : F ≤ U)
    (hfloor : AdvFloor P F κ) {Ω₀ : Type} [Fintype Ω₀] {M₀ : Mc.Alg (P.Seed × Inst × Ω₀) (P.Proof × Out)}
    (hM₀ : Useful P M₀ η) (hU : Mc.time M₀ ≤ U) : Core P (U / F - 1) κ η 0 := by
  intro Ω _ M hc
  have hU0 : 0 < U := hF.trans_le hFU
  have hb : (Tuse P η - 0) / (1 + (U / F - 1)) ≤ F := by
    rw [sub_zero, add_sub_cancel, div_div_eq_mul_div, div_le_iff₀ hU0]
    rw [mul_comm F]
    exact mul_le_mul_of_nonneg_right ((tuse_le P hM₀).trans hU) hF.le
  exact (prob_mono fun ω hω => ⟨hω.1, hω.2.trans_le hb⟩).trans (hfloor Ω M hc)

/-- **Efficiency** (RS-DEF §1.7): Solve costs at most `(1 + γ) N + α`, for `N` the reference matmul's time. -/
def Efficient (P : Protocol Mc Inst Out) (γ N α : ℝ) : Prop :=
  Mc.time P.solve ≤ (1 + γ) * N + α

/-- **Relative security** with parameters `ε, δ, γ, η, α` against the reference time `N` (RS-DEF's frozen form).

**Superseded by `RelativelySecurePerProver`** (proposed, a claim change), for `SecureInf`'s reason. -/
def RelativelySecure (P : Protocol Mc Inst Out) (ε δ γ η α N : ℝ) : Prop :=
  Complete P η ∧ SecureInf P ε δ η α ∧ Efficient P γ N α

/-- **Relative security from the adversary floor** (RS-DEF's route (a), end to end in the frozen definition): if Solve
is complete, costs at least the floor `F > 0` and is efficient, and every adversary's accepted runs below `F` have
probability at most `κ < δ`, the protocol is relatively secure with `ε = time(Solve) / F − 1` and `α = 0`.
Superseded by `relativelySecurePerProver_of_advFloor`. -/
theorem relativelySecure_of_advFloor (P : Protocol Mc Inst Out) {κ δ γ η F N : ℝ} (hc : Complete P η) (hF : 0 < F)
    (hFS : F ≤ Mc.time P.solve) (hfloor : AdvFloor P F κ) (hκδ : κ < δ) (heff : Efficient P γ N 0) :
    RelativelySecure P (Mc.time P.solve / F - 1) δ γ η 0 N := by
  refine ⟨hc, secureInf_of_core P (core_of_floor P hF hFS hfloor hc le_rfl) hκδ ?_, heff⟩
  rw [add_sub_cancel]
  exact div_pos (hF.trans_le hFS) hF

/-! ### Relative security per adversary (proposed, a claim change)

`AdvFloor`, `Core`, `SecureInf` and `RelativelySecure` read one `κ` or `δ` for every adversary. Over a service on
SHA-512 commitments the floor holds only at 1, so the chain from it says nothing below 1: an adversary that hard-codes a
collision opens a committed leaf two ways (a red team's side finding). The forms
here read each adversary's own bound in `ℝ≥0∞`, `⊤` where there is none: its floor (`AdvFloorPerProver`, the service's
`ServiceFloorPerProver`, at the floors' `breakBound`, read through the adversary's collision finders and
`Claims.ecrBound` at budgets computed from its time) and the acceptance its security asks about
(`SecureInfPerProver`, `RelativelySecurePerProver`).
Route (a) carries over at the threshold `κ(M) + δ'` for one `δ' > 0` (`secureInfPerProver_of_floor`): a hard-coded
collision's `κ` is about 1 or `⊤`, so security asks nothing of it, and every other adversary is asked about its own
bound.
-/

/-- **The adversary floor, adversary by adversary**: every adversary `M` that commits first has its accepted runs
charged (`max(|M|, cost)`) below `F` with probability at most `κ Ω M`, a bound of its own, `⊤` where there is none.
`AdvFloor` is the case of one `κ` for every adversary. -/
def AdvFloorPerProver (P : Protocol Mc Inst Out) (F : ℝ)
    (κ : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (P.Seed × Ω) P.Proof → ENNReal) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    ENNReal.ofReal (prob fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧ Mc.charge M ω < F) ≤ κ Ω M

/-- **Security, RS-DEF's frozen form, adversary by adversary** (proposed): every adversary `M` that commits first and
is accepted with probability at least its own threshold `δ Ω M` has `(1 + ε) time M + α ≥ T_use(η)`. A threshold above
1 asks nothing of `M`. `SecureInf` is the case of one threshold `δ` for every adversary. -/
def SecureInfPerProver (P : Protocol Mc Inst Out) (ε : ℝ)
    (δ : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (P.Seed × Ω) P.Proof → ENNReal) (η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    δ Ω M ≤ ENNReal.ofReal (acceptance P M) → Tuse P η ≤ (1 + ε) * Mc.time M + α

/-- **Relative security, adversary by adversary** (proposed), with parameters `ε, γ, η, α` against the reference time
`N` and each adversary's threshold `δ Ω M`: `RelativelySecure` with `SecureInfPerProver` in place of `SecureInf`. -/
def RelativelySecurePerProver (P : Protocol Mc Inst Out) (ε : ℝ)
    (δ : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (P.Seed × Ω) P.Proof → ENNReal) (γ η α N : ℝ) : Prop :=
  Complete P η ∧ SecureInfPerProver P ε δ η α ∧ Efficient P γ N α

/-- **Security per adversary from the floor per adversary** (Lemma R and `secureInf_of_core`, adversary by
adversary): under the floor `F > 0` per adversary, with a useful prover at most `U ≥ F`, security holds at
`1 + ε = U / F`, `α = 0`, and each adversary's threshold `κ(M) + δ'` for one `δ' > 0`: an adversary accepted that
often has a run charged at least `F`, so its time is at least `F ≥ T_use · F / U`. -/
theorem secureInfPerProver_of_floor (P : Protocol Mc Inst Out) {η F U δ' : ℝ}
    {κ : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (P.Seed × Ω) P.Proof → ENNReal} (hF : 0 < F) (hFU : F ≤ U)
    (hfloor : AdvFloorPerProver P F κ) {Ω₀ : Type} [Fintype Ω₀] {M₀ : Mc.Alg (P.Seed × Inst × Ω₀) (P.Proof × Out)}
    (hM₀ : Useful P M₀ η) (hU : Mc.time M₀ ≤ U) (hδ' : 0 < δ') :
    SecureInfPerProver P (U / F - 1) (fun Ω _ M => κ Ω M + ENNReal.ofReal δ') η 0 := by
  intro Ω _ M hc hM
  rw [add_sub_cancel, add_zero]
  by_contra hlt
  replace hlt := not_le.mp hlt
  have hU0 : 0 < U := hF.trans_le hFU
  have htime : Mc.time M < F := by
    by_contra hge
    replace hge := not_lt.mp hge
    have : U ≤ U / F * Mc.time M :=
      calc U = U / F * F := (div_mul_cancel₀ U hF.ne').symm
        _ ≤ U / F * Mc.time M := mul_le_mul_of_nonneg_left hge (div_nonneg hU0.le hF.le)
    linarith [(tuse_le P hM₀).trans hU]
  have hacc : ENNReal.ofReal (acceptance P M) ≤ κ Ω M := by
    unfold acceptance
    exact (ENNReal.ofReal_le_ofReal (prob_mono fun ω hω => ⟨hω, (Mc.charge_le_time M ω).trans_lt htime⟩)).trans
      (hfloor Ω M hc)
  have hκ : κ Ω M ≠ ⊤ := ne_top_of_le_ne_top ENNReal.ofReal_ne_top (le_self_add.trans hM)
  exact (ENNReal.lt_add_right hκ (ENNReal.ofReal_pos.mpr hδ').ne').not_ge (hM.trans hacc)

/-- **Relative security per adversary from the floor per adversary** (route (a) per adversary, as
`relativelySecure_of_advFloor`): if Solve is complete, costs at least the floor `F > 0` and is efficient, and every
adversary's accepted runs below `F` have probability at most its own `κ`, the protocol is relatively secure per
adversary with `ε = time(Solve) / F − 1`, `α = 0` and thresholds `κ(M) + δ'`. -/
theorem relativelySecurePerProver_of_advFloor (P : Protocol Mc Inst Out) {γ η F N δ' : ℝ}
    {κ : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (P.Seed × Ω) P.Proof → ENNReal} (hc : Complete P η) (hF : 0 < F)
    (hFS : F ≤ Mc.time P.solve) (hfloor : AdvFloorPerProver P F κ) (hδ' : 0 < δ') (heff : Efficient P γ N 0) :
    RelativelySecurePerProver P (Mc.time P.solve / F - 1) (fun Ω _ M => κ Ω M + ENNReal.ofReal δ') γ η 0 N :=
  ⟨hc, secureInfPerProver_of_floor P hF hFS hfloor hc le_rfl hδ', heff⟩

/-- The matmul problem on instances that embed into pairs of `n × n` matrices over `R` (for example RS-DEF's useful
grid `G_use` as a subtype): `x` is answered by `C = A · B` for `(A, B) = emb x`. -/
def IsMatmulOn {R : Type} [Ring R] {n : ℕ} {Mc : Machine} {Inst : Type} (P : Protocol Mc Inst (Mat R n n))
    (emb : Inst → Mat R n n × Mat R n n) : Prop :=
  ∀ σ x C, P.correct σ x C ↔ (emb x).1 * (emb x).2 = C

/-- The matmul problem on all pairs of `n × n` matrices over `R`. -/
def IsMatmul {R : Type} [Ring R] {n : ℕ} {Mc : Machine} (P : Protocol Mc (Mat R n n × Mat R n n) (Mat R n n)) :
    Prop :=
  IsMatmulOn P id

/-- Under completeness, `T_use(η)` is at most Solve's time. -/
theorem tuse_le_solve (P : Protocol Mc Inst Out) {η : ℝ} (h : Complete P η) : Tuse P η ≤ Mc.time P.solve :=
  tuse_le P h

theorem secure_mono (P : Protocol Mc Inst Out) {ε ε' δ δ' η η' α α' : ℝ} (h : Secure P ε δ η α) (hε : ε ≤ ε')
    (hδ : δ ≤ δ') (hη : η ≤ η') (hα : α ≤ α') : Secure P ε' δ' η' α' := by
  intro Ω _ M hc hM
  obtain ⟨Ω', _, M', ht, hc', hs⟩ := h Ω M hc (hδ.trans hM)
  refine ⟨Ω', inferInstance, M', ht.trans ?_, hc', fun x hx => le_trans (by linarith) (hs x hx)⟩
  have := mul_le_mul_of_nonneg_right (show 1 + ε ≤ 1 + ε' by linarith) (Mc.time_nonneg M)
  linarith

theorem complete_mono (P : Protocol Mc Inst Out) {η η' : ℝ} (h : Complete P η) (hη : η ≤ η') : Complete P η' :=
  ⟨h.1, fun x hx => le_trans (by linarith) (h.2 x hx)⟩

end Definition

section Masking

variable {α : Type*} [Ring α] {m k p r r' : Type*} [Fintype k] [Fintype r] [Fintype r']

/-- **The full-rank peel**: `A · B = (A + R)(B + S) − R (B + S) − A · S`, over any ring (in particular any commutative
ring, `ZMod (2^32)` and any field). -/
theorem mask_peel (A R : Matrix m k α) (B S : Matrix k p α) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S := by
  rw [Matrix.add_mul, Matrix.mul_add]
  abel

/-- **The low-rank peel**: with masks `U Vᵀ` (rank `r`) and `W Zᵀ` (rank `r'`), the masked product minus the two
corrections, each a product through an inner dimension `r` or `r'`, is `A · B`:
`(A + U Vᵀ)(B + W Zᵀ) − U (Vᵀ (B + W Zᵀ)) − (A W) Zᵀ = A · B`. -/
theorem lowRank_peel (A : Matrix m k α) (B : Matrix k p α) (U : Matrix m r α) (Vt : Matrix r k α)
    (W : Matrix k r' α) (Zt : Matrix r' p α) :
    (A + U * Vt) * (B + W * Zt) - U * (Vt * (B + W * Zt)) - (A * W) * Zt = A * B := by
  rw [Matrix.add_mul, Matrix.mul_add A, Matrix.mul_assoc U, Matrix.mul_assoc A]
  abel

/-- The full-rank peel in s32 arithmetic (`ZMod (2^32)`), where it is exact. -/
theorem mask_peel_zmod32 (A R : Matrix m k (ZMod (2 ^ 32))) (B S : Matrix k p (ZMod (2 ^ 32))) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S :=
  mask_peel A R B S

/-- The full-rank peel over a field, where it is also exact. -/
theorem mask_peel_field {F : Type*} [Field F] (A R : Matrix m k F) (B S : Matrix k p F) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S :=
  mask_peel A R B S

end Masking

section Exactness

/-- An integer is its s32 residue's signed representative when `-2^31 < z ≤ 2^31`. -/
theorem valMinAbs_intCast_of_bounds {z : ℤ} (hlo : -2 ^ 31 < z) (hhi : z ≤ 2 ^ 31) :
    ((z : ZMod (2 ^ 32))).valMinAbs = z := by
  rw [ZMod.valMinAbs_spec]
  refine ⟨rfl, ?_, ?_⟩ <;> push_cast <;> omega

variable {m k p : Type*} [Fintype k]

/-- The full-rank peel of integer matrices `A`, `B` computed in s32 (`ZMod (2^32)`) with masks `R`, `S`:
`(A + R)(B + S) − R (B + S) − A · S`. -/
def peel32 (A : Matrix m k ℤ) (B : Matrix k p ℤ) (R : Matrix m k (ZMod (2 ^ 32)))
    (S : Matrix k p (ZMod (2 ^ 32))) : Matrix m p (ZMod (2 ^ 32)) :=
  let A' := A.map (Int.castRingHom (ZMod (2 ^ 32)))
  let B' := B.map (Int.castRingHom (ZMod (2 ^ 32)))
  (A' + R) * (B' + S) - R * (B' + S) - A' * S

/-- **s32 recovers the integer product**: for integer matrices whose product has entries in `(-2^31, 2^31]`, the
peel computed in s32, with any masks, has the integer product as its signed representative. -/
theorem s32_peel_recovers (A : Matrix m k ℤ) (B : Matrix k p ℤ) (R : Matrix m k (ZMod (2 ^ 32)))
    (S : Matrix k p (ZMod (2 ^ 32)))
    (hb : ∀ i j, -2 ^ 31 < (A * B) i j ∧ (A * B) i j ≤ 2 ^ 31) (i : m) (j : p) :
    (peel32 A B R S i j).valMinAbs = (A * B) i j := by
  have h : peel32 A B R S = (A * B).map (Int.castRingHom (ZMod (2 ^ 32))) := by
    rw [Matrix.map_mul]
    exact (mask_peel _ R _ S).symm
  rw [h, Matrix.map_apply]
  exact valMinAbs_intCast_of_bounds (hb i j).1 (hb i j).2

/-- int8 by int8 products of inner dimension at most `2^17` lie in `(-2^31, 2^31]`, so s32 accumulation is exact. -/
theorem int8_product_bounds (A : Matrix m k ℤ) (B : Matrix k p ℤ) (hk : Fintype.card k ≤ 2 ^ 17)
    (hA : ∀ i l, -128 ≤ A i l ∧ A i l ≤ 127) (hB : ∀ l j, -128 ≤ B l j ∧ B l j ≤ 127) (i : m) (j : p) :
    -2 ^ 31 < (A * B) i j ∧ (A * B) i j ≤ 2 ^ 31 := by
  have hterm : ∀ l, -16256 ≤ A i l * B l j ∧ A i l * B l j ≤ 16384 := by
    intro l
    obtain ⟨a1, a2⟩ := hA i l
    obtain ⟨b1, b2⟩ := hB l j
    constructor <;> nlinarith
  have hk' : (Fintype.card k : ℤ) ≤ 2 ^ 17 := by exact_mod_cast hk
  rw [Matrix.mul_apply]
  constructor
  · have : ∑ l, (-16256 : ℤ) ≤ ∑ l, A i l * B l j := Finset.sum_le_sum fun l _ => (hterm l).1
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at this
    linarith
  · have : ∑ l, A i l * B l j ≤ ∑ l, (16384 : ℤ) := Finset.sum_le_sum fun l _ => (hterm l).2
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at this
    linarith

/-- **int8 with s32 accumulation is exact** for inner dimension at most `2^17`: the s32 peel of int8 matrices, with
any masks, recovers the integer product. -/
theorem int8_product_exact (A : Matrix m k ℤ) (B : Matrix k p ℤ) (hk : Fintype.card k ≤ 2 ^ 17)
    (hA : ∀ i l, -128 ≤ A i l ∧ A i l ≤ 127) (hB : ∀ l j, -128 ≤ B l j ∧ B l j ≤ 127)
    (R : Matrix m k (ZMod (2 ^ 32))) (S : Matrix k p (ZMod (2 ^ 32))) (i : m) (j : p) :
    (peel32 A B R S i j).valMinAbs = (A * B) i j :=
  s32_peel_recovers A B R S (int8_product_bounds A B hk hA hB) i j

/-- A product of integer matrices with entries at most `M` in absolute value has entries at most `|k| M²`. -/
theorem grid_product_bound (A : Matrix m k ℤ) (B : Matrix k p ℤ) {M : ℤ} (hA : ∀ i l, |A i l| ≤ M)
    (hB : ∀ l j, |B l j| ≤ M) (i : m) (j : p) : |(A * B) i j| ≤ Fintype.card k * M ^ 2 := by
  rw [Matrix.mul_apply]
  calc |∑ l, A i l * B l j| ≤ ∑ l, |A i l * B l j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _l : k, M ^ 2 := Finset.sum_le_sum fun l _ => by
        rw [abs_mul, sq]
        exact mul_le_mul (hA i l) (hB l j) (abs_nonneg _) ((abs_nonneg _).trans (hA i l))
    _ = Fintype.card k * M ^ 2 := by simp [Finset.sum_const, Finset.card_univ]

/-- **RS-DEF's exactness condition (E)** on the integer grid: if the useful operands, the noise and the masked operands
all have entries at most `M_op` and `n M_op² ≤ B_acc`, every product the peel computes, `(A + R)(B + S)`, `R (B + S)`
and `A S`, has entries at most `B_acc`, so an accumulator exact up to `B_acc` (`2^24` for E4M3 integers with FP32
accumulation, `2^31 − 1` for int8 with s32) computes each exactly, and the peel over `ℤ` is `A B` (`mask_peel`). -/
theorem grid_peel_in_range (A R : Matrix m k ℤ) (B S : Matrix k p ℤ) {Mop Bacc : ℤ}
    (hA : ∀ i l, |A i l| ≤ Mop) (hR : ∀ i l, |R i l| ≤ Mop) (hS : ∀ l j, |S l j| ≤ Mop)
    (hAR : ∀ i l, |(A + R) i l| ≤ Mop) (hBS : ∀ l j, |(B + S) l j| ≤ Mop)
    (hE : Fintype.card k * Mop ^ 2 ≤ Bacc) (i : m) (j : p) :
    |((A + R) * (B + S)) i j| ≤ Bacc ∧ |(R * (B + S)) i j| ≤ Bacc ∧ |(A * S) i j| ≤ Bacc :=
  ⟨(grid_product_bound _ _ hAR hBS i j).trans hE, (grid_product_bound _ _ hR hBS i j).trans hE,
    (grid_product_bound _ _ hA hS i j).trans hE⟩

/-- RS-DEF §1.1's instances meet (E): int8 operands in `[-126, 126]` (useful and noise in `[-63, 63]`) with s32
accumulation for `n ≤ 135,266`, the largest such `n`; E4M3 integers in `[-16, 16]` with FP32 accumulation for
`n ≤ 65,536`. -/
theorem grid_instances :
    (135266 : ℤ) * 126 ^ 2 ≤ 2 ^ 31 - 1 ∧ ¬ (135267 : ℤ) * 126 ^ 2 ≤ 2 ^ 31 - 1 ∧ (65536 : ℤ) * 16 ^ 2 ≤ 2 ^ 24 := by
  norm_num

/-- Saturating int8 arithmetic (clamped to `[-128, 127]`), the kind of non-exact accumulator the peel needs to avoid. -/
def sat8 (z : ℤ) : ℤ := max (-128) (min 127 z)

/-- **A non-exact arithmetic breaks the peel**: masking `a = 100` by `r = 100` in saturating int8 and peeling the
mask off gives `27`, not `100`. -/
theorem sat8_breaks_peel : sat8 (sat8 (100 + 100) - 100) ≠ 100 := by
  decide

end Exactness

section ZeroInstance

/-! RS-DEF's Proposition Z, the algebra: what the zero instance's transcript is. `D` is any matrix (the prefix
selector `diag(1_{k ≤ jL})` for the `j`-th prefix sum `P_j = A' D B'`). -/

variable {α : Type*} [Ring α] {m k p r r' : Type*} [Fintype k] [Fintype r] [Fintype r']

/-- **Z(a), σ before the instance**: an instance chosen after the noise as `A_M = −E`, `B_M = −F` makes every
masked prefix sum zero, for free. -/
theorem sigma_first_cancels (E : Matrix m k α) (F : Matrix k p α) (D : Matrix k k α) :
    (-E + E) * D * (-F + F) = 0 := by
  simp

/-- **Z(c)**: with `A_M = B_M = 0` the masked prefix sums are `P_j = E (D F)`, one product with the noise `E`. -/
theorem zero_instance_prefix (E : Matrix m k α) (F : Matrix k p α) (D : Matrix k k α) :
    (0 + E) * D * (0 + F) = E * (D * F) := by
  simp [Matrix.mul_assoc]

/-- **Z(d)**: with rank-`r` noise `E = E_L E_R` and `F = F_L F_R` and `A_M = B_M = 0`, the masked prefix sums are
`P_j = E_L Q_j F_R` with `Q_j = E_R D F_L` an `r × r'` matrix: products through the inner dimensions `r`, `r'`. -/
theorem zero_instance_lowRank_prefix (EL : Matrix m r α) (ER : Matrix r k α) (FL : Matrix k r' α)
    (FR : Matrix r' p α) (D : Matrix k k α) :
    (0 + EL * ER) * D * (0 + FL * FR) = EL * (ER * D * FL) * FR := by
  simp only [zero_add, Matrix.mul_assoc]

end ZeroInstance

section Reduction

namespace Machine

variable (Mc : Machine)

/-- **The reduction** `M' = M ++ Q`: run the extractor `E` (M's own program on the masked instance; its proof and the
blocks read off its wires), keep the proof, and run the peel and then the correction on the blocks. -/
def reduction {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid O) :
    Mc.Alg I (Pf × O) :=
  Mc.seq E (Mc.pair Mc.fst (Mc.seq Mc.snd (Mc.seq peel corr)))

theorem eval_reduction {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid)
    (corr : Mc.Alg Mid O) (x : I) :
    Mc.eval (Mc.reduction E peel corr) x = ((Mc.eval E x).1, Mc.eval corr (Mc.eval peel (Mc.eval E x).2)) := by
  simp only [reduction, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd]

/-- The reduction costs at most the extractor, the peel and the correction. -/
theorem time_reduction_le {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid)
    (corr : Mc.Alg Mid O) : Mc.time (Mc.reduction E peel corr) ≤ Mc.time E + Mc.time peel + Mc.time corr := by
  have h1 := Mc.time_seq_le E (Mc.pair Mc.fst (Mc.seq Mc.snd (Mc.seq peel corr)))
  have h2 := Mc.time_pair_le (Mc.fst (I := Pf) (J := Bl)) (Mc.seq Mc.snd (Mc.seq peel corr))
  have h3 := Mc.time_seq_le (Mc.snd (I := Pf) (J := Bl)) (Mc.seq peel corr)
  have h4 := Mc.time_seq_le peel corr
  have h5 := Mc.time_fst (I := Pf) (J := Bl)
  have h6 := Mc.time_snd (I := Pf) (J := Bl)
  unfold reduction
  linarith

/-- **The cost inequality**: with an extractor that costs at most `M`,
`time M' ≤ time M + time peel + time correction`. -/
theorem time_reduction_le_of_le {I Pf Bl Mid O J : Type} {E : Mc.Alg I (Pf × Bl)} {M : Mc.Alg J Pf}
    (hE : Mc.time E ≤ Mc.time M) (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid O) :
    Mc.time (Mc.reduction E peel corr) ≤ Mc.time M + Mc.time peel + Mc.time corr := by
  linarith [Mc.time_reduction_le E peel corr]

end Machine

variable {Mc : Machine} {Inst Out : Type}

/-- **Extraction** (a named hypothesis, RS-EXT's Lemma E and Theorem R): from an adversary `M`, an extractor `E` that
commits first, costs at most `(1 + εext) time M` (one straight-line run of M on the planted instance, reading M's
committed blocks off its hash log, plus the check, the fixes and its own proof), and, for every promised instance
`x`, with probability at least `1 - η` over σ and its coins, outputs prover messages that are accepted and blocks
valid for `x`. -/
def Extraction (P : Protocol Mc Inst Out) {Bl : Type} (valid : Inst → Bl → Prop) (εext : ℝ) {Ω : Type} [Fintype Ω]
    (M : Mc.Alg (P.Seed × Ω) P.Proof) (η : ℝ) : Prop :=
  ∃ (Ω' : Type) (_ : Fintype Ω') (E : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Bl)),
    CommitsFirst P (fun y => (Mc.eval E y).1) ∧ Mc.time E ≤ (1 + εext) * Mc.time M ∧
    ∀ x, P.promise x → 1 - η ≤ prob fun ω : P.Seed × Ω' =>
      valid x (Mc.eval E (ω.1, x, ω.2)).2 ∧ P.accepts ω.1 (Mc.eval E (ω.1, x, ω.2)).1 = true

/-- **The floor** (a named hypothesis, the floor agenda's: Conjecture A, Theorems B/C/W/F): every algorithm that
answers every promised instance correctly with probability at least `1 - η` over its coins costs at least `T`
(`T*_mm(n)` for matmul, RS-DEF §1.7; no proof is asked for, so `T_use(η) ≥ T`). -/
def Floor (Mc : Machine) {Inst Out : Type} (promise : Inst → Prop) (correct : Inst → Out → Prop) (T η : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (N : Mc.Alg (Inst × Ω) Out),
    (∀ x, promise x → 1 - η ≤ prob fun ω : Ω => correct x (Mc.eval N (x, ω))) → T ≤ Mc.time N

/-- An algorithm that solves every instance (with a proof) is above the floor of any problem its answers solve
(`hcorr`; for an unedited protocol, its own): drop σ into its coins and forget the proof, which routing does for
free. -/
theorem Floor.le_time (P : Protocol Mc Inst Out) {correct : Inst → Out → Prop} {T η : ℝ}
    (h : Floor Mc P.promise correct T η)
    (hcorr : ∀ σ x C, P.correct σ x C → correct x C) {Ω : Type} [Fintype Ω]
    {M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out)} (hM' : SolvesAll P M' η) : T ≤ Mc.time M' := by
  let route : Mc.Alg (Inst × P.Seed × Ω) (P.Seed × Inst × Ω) :=
    Mc.pair (Mc.seq Mc.snd Mc.fst) (Mc.pair Mc.fst (Mc.seq Mc.snd Mc.snd))
  have hroute : ∀ y, Mc.eval route y = (y.2.1, y.1, y.2.2) := fun y => by
    simp only [route, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd]
  have hT := h (P.Seed × Ω) (Mc.seq (Mc.seq route M') Mc.snd) fun x hx => (hM' x hx).trans (prob_mono fun ω hω => by
    rw [Mc.eval_seq, Mc.eval_seq, hroute, Mc.eval_snd]
    exact hcorr _ _ _ hω.1)
  have hr : Mc.time route ≤ 0 := by
    apply le_trans
    · apply_rules [Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
    · norm_num
  have r1 := Mc.time_seq_le (Mc.seq route M') (Mc.snd (I := P.Proof) (J := Out))
  have r2 := Mc.time_seq_le route M'
  have z := Mc.time_snd (I := P.Proof) (J := Out)
  linarith

theorem Floor.le_tuse (P : Protocol Mc Inst Out) {correct : Inst → Out → Prop} {T η : ℝ}
    (h : Floor Mc P.promise correct T η)
    (hcorr : ∀ σ x C, P.correct σ x C → correct x C)
    (hne : ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)), Useful P M' η) :
    T ≤ Tuse P η := by
  obtain ⟨Ω', _, M', hu⟩ := hne
  exact le_csInf ⟨_, Ω', inferInstance, M', hu, rfl⟩ fun _ ⟨_, _, _, hu', ht⟩ => ht ▸ h.le_time P hcorr hu'.2

/-- With extraction, the reduction is useful and costs at most `(1 + εext) time M + time peel + time corr`, when the
peel and the correction answer `x` from blocks valid for `x`. -/
theorem exists_reduction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext η : ℝ} {Ω : Type} [Fintype Ω]
    {M : Mc.Alg (P.Seed × Ω) P.Proof} (hext : Extraction P valid εext M η) :
    ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
      Mc.time M' ≤ (1 + εext) * Mc.time M + Mc.time peel + Mc.time corr ∧ Useful P M' η := by
  obtain ⟨Ω', _, E, hc, hE, hsucc⟩ := hext
  refine ⟨Ω', inferInstance, Mc.reduction E peel corr, (Mc.time_reduction_le E peel corr).trans (by linarith),
    fun σ σ' y => ?_, fun x hx => (hsucc x hx).trans (prob_mono fun ω hω => ?_)⟩
  · simp only [Mc.eval_reduction]
    exact hc σ σ' y
  · rw [Mc.eval_reduction]
    exact ⟨hpeel _ x _ hω.1, hω.2⟩

/-- **Security from extraction**, given that the peel and the correction cost at most `ε time M` for every adversary
`M` accepted with probability at least `δ` (extraction overhead `εext`). -/
theorem secure_of_extraction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext ε δ η : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P valid εext M η)
    (hcost : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Mc.time peel + Mc.time corr ≤ ε * Mc.time M) :
    Secure P (εext + ε) δ η 0 := by
  intro Ω _ M hc hM
  obtain ⟨Ω', _, M', ht, hu⟩ := exists_reduction P peel corr hpeel (hext Ω M hc hM)
  exact ⟨Ω', inferInstance, M', by linarith [hcost Ω M hc hM], hu⟩

/-- **Security from extraction and the floor**: if the peel and the correction cost at most `q < T`, where `T` is the
floor, then `ε = εext + (1 + εext) q / (T − q)`; with `εext = 0` this is RS-MODEL's `q / (T − q)`, eq. (2).
(`M'` is useful, so `T ≤ time M' ≤ (1 + εext) time M + q`.) -/
theorem secure_of_extraction_floor (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext δ η T q : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P valid εext M η) {correct : Inst → Out → Prop}
    (hfloor : Floor Mc P.promise correct T η) (hcorr : ∀ σ x C, P.correct σ x C → correct x C)
    (hq : Mc.time peel + Mc.time corr ≤ q) (hqT : q < T) :
    Secure P (εext + (1 + εext) * (q / (T - q))) δ η 0 := by
  intro Ω _ M hc hM
  obtain ⟨Ω', _, M', ht, hu⟩ := exists_reduction P peel corr hpeel (hext Ω M hc hM)
  refine ⟨Ω', inferInstance, M', ?_, hu⟩
  have hTM := hfloor.le_time P hcorr hu.2
  have hTq : 0 < T - q := sub_pos.2 hqT
  have hq0 : 0 ≤ q := (add_nonneg (Mc.time_nonneg _) (Mc.time_nonneg _)).trans hq
  set θ := q / (T - q) with hθ
  have hθ0 : 0 ≤ θ := div_nonneg hq0 hTq.le
  have hθq : θ * (T - q) = q := div_mul_cancel₀ q hTq.ne'
  have hM : θ * (T - q) ≤ θ * ((1 + εext) * Mc.time M) := mul_le_mul_of_nonneg_left (by linarith) hθ0
  nlinarith

/-- **Security from extraction and M's own floor** (RS-MODEL's route 2, `ε ≤ time(Q) / (c F_M)`, which needs no
`T_mm`): if every adversary accepted with probability at least `δ` costs at least `F > 0` (a named hypothesis, the
floor on M's committed targets) and the peel and the correction cost at most `q`, then `ε = εext + q / F`. -/
theorem secure_of_extraction_advTime (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext δ η F q : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P valid εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M)
    (hF : 0 < F) (hq : Mc.time peel + Mc.time corr ≤ q) : Secure P (εext + q / F) δ η 0 := by
  refine secure_of_extraction P peel corr hpeel hext fun Ω _ M hc hδ => hq.trans ?_
  have hq0 : 0 ≤ q := (add_nonneg (Mc.time_nonneg _) (Mc.time_nonneg _)).trans hq
  calc q = q / F * F := (div_mul_cancel₀ q hF.ne').symm
    _ ≤ q / F * Mc.time M := mul_le_mul_of_nonneg_left (hM Ω M hc hδ) (div_nonneg hq0 hF.le)

/-- **Security is a relative floor** (RS-DEF §1.5), one direction: if some useful `M₀` costs at most
`(1 + ε) time M + α` for every adversary `M` accepted with probability at least `δ`, the protocol is secure. No
reduction from `M` is needed. -/
theorem secure_of_solver (P : Protocol Mc Inst Out) {ε δ η α : ℝ} {Ω₀ : Type} [Fintype Ω₀]
    {M₀ : Mc.Alg (P.Seed × Inst × Ω₀) (P.Proof × Out)} (hM₀ : Useful P M₀ η)
    (hcost : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Mc.time M₀ ≤ (1 + ε) * Mc.time M + α) :
    Secure P ε δ η α :=
  fun Ω _ M hc hM => ⟨Ω₀, inferInstance, M₀, hcost Ω M hc hM, hM₀⟩

/-- **Security is a relative floor**, the other direction: under security, every adversary accepted with probability
at least `δ` has `T ≤ (1 + ε) time M + α`, for `T` the floor. -/
theorem floor_le_of_secure (P : Protocol Mc Inst Out) {ε δ η α T : ℝ} (h : Secure P ε δ η α)
    {correct : Inst → Out → Prop} (hfloor : Floor Mc P.promise correct T η) (hcorr : ∀ σ x C, P.correct σ x C → correct x C)
    {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof)
    (hc : CommitsFirst P (Mc.eval M)) (hM : δ ≤ acceptance P M) : T ≤ (1 + ε) * Mc.time M + α := by
  obtain ⟨Ω', _, M', ht, hu⟩ := h Ω M hc hM
  exact (hfloor.le_time P hcorr hu.2).trans ht

end Reduction

section MatmulPeel

namespace MatMachine

variable {R : Type} [Ring R] (Mc : MatMachine R)

/-- Multiply the outputs of `f` and `g`. -/
def mul₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) : Mc.Alg I (Mat R a c) :=
  Mc.lift2 f g (Mc.mul a b c)

/-- Subtract the output of `g` from that of `f`. -/
def sub₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) : Mc.Alg I (Mat R a b) :=
  Mc.lift2 f g (Mc.sub a b)

theorem eval_mul₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) (x : I) :
    Mc.eval (Mc.mul₂ f g) x = Mc.eval f x * Mc.eval g x := by
  rw [mul₂, Machine.eval_lift2, Mc.eval_mul]

theorem eval_sub₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) (x : I) :
    Mc.eval (Mc.sub₂ f g) x = Mc.eval f x - Mc.eval g x := by
  rw [sub₂, Machine.eval_lift2, Mc.eval_sub]

theorem time_mul₂_le_of {I : Type} {a b c : ℕ} {f : Mc.Alg I (Mat R a b)} {g : Mc.Alg I (Mat R b c)} {u v : ℝ}
    (hf : Mc.time f ≤ u) (hg : Mc.time g ≤ v) :
    Mc.time (Mc.mul₂ f g) ≤ u + v + Mc.κ * ((a : ℝ) * b * c) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_mul_le a b c])

theorem time_sub₂_le_of {I : Type} {a b : ℕ} {f g : Mc.Alg I (Mat R a b)} {u v : ℝ} (hf : Mc.time f ≤ u)
    (hg : Mc.time g ≤ v) : Mc.time (Mc.sub₂ f g) ≤ u + v + Mc.κadd * ((a : ℝ) * b) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_sub_le a b])

/-- Add the outputs of `f` and `g`. -/
def add₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) : Mc.Alg I (Mat R a b) :=
  Mc.lift2 f g (Mc.add a b)

theorem eval_add₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) (x : I) :
    Mc.eval (Mc.add₂ f g) x = Mc.eval f x + Mc.eval g x := by
  rw [add₂, Machine.eval_lift2, Mc.eval_add]

theorem time_add₂_le_of {I : Type} {a b : ℕ} {f g : Mc.Alg I (Mat R a b)} {u v : ℝ} (hf : Mc.time f ≤ u)
    (hg : Mc.time g ≤ v) : Mc.time (Mc.add₂ f g) ≤ u + v + Mc.κadd * ((a : ℝ) * b) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_add_le a b])

/-- The correction `(C', X, Y) ↦ C' − X − Y`. -/
def corr (n : ℕ) : Mc.Alg (Mat R n n × Mat R n n × Mat R n n) (Mat R n n) :=
  Mc.sub₂ (Mc.sub₂ Mc.fst (Mc.seq Mc.snd Mc.fst)) (Mc.seq Mc.snd Mc.snd)

theorem eval_corr (n : ℕ) (b : Mat R n n × Mat R n n × Mat R n n) :
    Mc.eval (Mc.corr n) b = b.1 - b.2.1 - b.2.2 := by
  simp only [corr, eval_sub₂, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The correction costs two subtractions. -/
theorem time_corr_le (n : ℕ) : Mc.time (Mc.corr n) ≤ 2 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold corr
    apply_rules [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- The full-rank peel's blocks `(C', A, Y, R, S)`: the committed product, the instance's `A`, M's masked
`Y = B + S`, and the masks. -/
abbrev FullBlocks (R : Type) (n : ℕ) : Type := Mat R n n × Mat R n n × Mat R n n × Mat R n n × Mat R n n

/-- Blocks valid for the instance `(A, B)`: `A` is the instance's, `Y = B + S`, and `C' = (A + R) Y`. -/
def ValidFull {n : ℕ} (x : Mat R n n × Mat R n n) (b : FullBlocks R n) : Prop :=
  b.2.1 = x.1 ∧ b.2.2.1 = x.2 + b.2.2.2.2 ∧ b.1 = (x.1 + b.2.2.2.1) * b.2.2.1

/-- **The full-rank peel**: `(C', A, Y, R, S) ↦ (C', R Y, A S)`, two `n × n × n` products. -/
def peelFull (n : ℕ) : Mc.Alg (FullBlocks R n) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
      (Mc.mul₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))

theorem eval_peelFull (n : ℕ) (b : FullBlocks R n) :
    Mc.eval (Mc.peelFull n) b = (b.1, b.2.2.2.1 * b.2.2.1, b.2.1 * b.2.2.2.2) := by
  simp only [peelFull, eval_mul₂, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The full-rank peel and correction answer the instance from valid blocks: `C' − R Y − A S = A B`. -/
theorem peelFull_correct {n : ℕ} (x : Mat R n n × Mat R n n) (b : FullBlocks R n) (hb : ValidFull x b) :
    x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelFull n) b) := by
  obtain ⟨C', A, Y, Rm, S⟩ := b
  obtain ⟨hA, hY, hC⟩ := hb
  dsimp only at hA hY hC
  subst hA hY hC
  rw [eval_peelFull, eval_corr]
  exact mask_peel x.1 Rm x.2 S

/-- The full-rank peel costs two `n × n × n` products. -/
theorem time_peelFull_le (n : ℕ) : Mc.time (Mc.peelFull n) ≤ 2 * Mc.κ * (n : ℝ) ^ 3 := by
  apply le_trans
  · unfold peelFull
    apply_rules [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- The low-rank peel's blocks `(C', A, Y, U, Vᵀ, W, Zᵀ)`: the committed product, the instance's `A`, M's masked
`Y = B + W Zᵀ`, and the factors of the rank-`r` masks `U Vᵀ` and `W Zᵀ`. -/
abbrev LowBlocks (R : Type) (n r : ℕ) : Type :=
  Mat R n n × Mat R n n × Mat R n n × Mat R n r × Mat R r n × Mat R n r × Mat R r n

/-- Blocks valid for the instance `(A, B)`: `A` is the instance's, `Y = B + W Zᵀ`, and `C' = (A + U Vᵀ) Y`. -/
def ValidLow {n r : ℕ} (x : Mat R n n × Mat R n n) (b : LowBlocks R n r) : Prop :=
  b.2.1 = x.1 ∧ b.2.2.1 = x.2 + b.2.2.2.2.2.1 * b.2.2.2.2.2.2 ∧ b.1 = (x.1 + b.2.2.2.1 * b.2.2.2.2.1) * b.2.2.1

/-- **The low-rank peel**: `(C', A, Y, U, Vᵀ, W, Zᵀ) ↦ (C', U (Vᵀ Y), (A W) Zᵀ)`, four products through the inner
dimension `r`. -/
def peelLow (n r : ℕ) : Mc.Alg (LowBlocks R n r) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair
      (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
        (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
          (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
      (Mc.mul₂
        (Mc.mul₂ (Mc.seq Mc.snd Mc.fst)
          (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))))
        (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))

theorem eval_peelLow (n r : ℕ) (b : LowBlocks R n r) :
    Mc.eval (Mc.peelLow n r) b =
      (b.1, b.2.2.2.1 * (b.2.2.2.2.1 * b.2.2.1), (b.2.1 * b.2.2.2.2.2.1) * b.2.2.2.2.2.2) := by
  simp only [peelLow, eval_mul₂, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The low-rank peel and correction answer the instance from valid blocks:
`C' − U (Vᵀ Y) − (A W) Zᵀ = A B`. -/
theorem peelLow_correct {n r : ℕ} (x : Mat R n n × Mat R n n) (b : LowBlocks R n r) (hb : ValidLow x b) :
    x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelLow n r) b) := by
  obtain ⟨C', A, Y, U, Vt, W, Zt⟩ := b
  obtain ⟨hA, hY, hC⟩ := hb
  dsimp only at hA hY hC
  subst hA hY hC
  rw [eval_peelLow, eval_corr]
  exact (lowRank_peel x.1 x.2 U Vt W Zt).symm

/-- The low-rank peel costs four products through the inner dimension `r`: `4 κ n² r`. -/
theorem time_peelLow_le (n r : ℕ) : Mc.time (Mc.peelLow n r) ≤ 4 * Mc.κ * (n : ℝ) ^ 2 * r := by
  apply le_trans
  · unfold peelLow
    apply_rules (maxDepth := 200) [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- **The honest useful computation, full-rank masks**: `(A, B, R, S) ↦ (A + R)(B + S) − R (B + S) − A S`, the masked
product and its peel (RS-DEF §1.7's Solve without its commitments and proof). -/
def honestFull (n : ℕ) : Mc.Alg (Mat R n n × Mat R n n × Mat R n n × Mat R n n) (Mat R n n) :=
  Mc.sub₂
    (Mc.sub₂
      (Mc.mul₂ (Mc.add₂ Mc.fst (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
        (Mc.add₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd))))
      (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))
        (Mc.add₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))
    (Mc.mul₂ Mc.fst (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))

theorem eval_honestFull (n : ℕ) (x : Mat R n n × Mat R n n × Mat R n n × Mat R n n) :
    Mc.eval (Mc.honestFull n) x = x.1 * x.2.1 := by
  simp only [honestFull, eval_sub₂, eval_mul₂, eval_add₂, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  exact (mask_peel x.1 x.2.2.1 x.2.1 x.2.2.2).symm

/-- The full-rank honest computation costs three `n × n × n` products: `γ ≤ 2 + 5 κadd / (κ n)` against the
reference `κ n³` (RS-DEF's Proposition Z(e): full-rank noise buys nothing). -/
theorem time_honestFull_le (n : ℕ) :
    Mc.time (Mc.honestFull n) ≤ 3 * Mc.κ * (n : ℝ) ^ 3 + 5 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold honestFull
    apply_rules (maxDepth := 200) [time_mul₂_le_of, time_sub₂_le_of, time_add₂_le_of, Machine.time_pair_le_of,
      Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- **The honest useful computation, rank-`r` masks**:
`(A, B, U, Vᵀ, W, Zᵀ) ↦ (A + U Vᵀ)(B + W Zᵀ) − U (Vᵀ (B + W Zᵀ)) − (A W) Zᵀ`. -/
def honestLow (n r : ℕ) :
    Mc.Alg (Mat R n n × Mat R n n × Mat R n r × Mat R r n × Mat R n r × Mat R r n) (Mat R n n) :=
  Mc.sub₂
    (Mc.sub₂
      (Mc.mul₂
        (Mc.add₂ Mc.fst
          (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)) (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))))
        (Mc.add₂ (Mc.seq Mc.snd Mc.fst)
          (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
            (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))
      (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))
        (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
          (Mc.add₂ (Mc.seq Mc.snd Mc.fst)
            (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
              (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))))
    (Mc.mul₂ (Mc.mul₂ Mc.fst (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))))
      (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))

theorem eval_honestLow (n r : ℕ) (x : Mat R n n × Mat R n n × Mat R n r × Mat R r n × Mat R n r × Mat R r n) :
    Mc.eval (Mc.honestLow n r) x = x.1 * x.2.1 := by
  simp only [honestLow, eval_sub₂, eval_mul₂, eval_add₂, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  exact lowRank_peel x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2

/-- The rank-`r` honest computation costs one `n × n × n` product and seven products through the inner dimension `r`
(the mask `W Zᵀ` is formed twice): `γ ≤ 7 r / n + 5 κadd / (κ n)` against the reference `κ n³`. -/
theorem time_honestLow_le (n r : ℕ) :
    Mc.time (Mc.honestLow n r) ≤ Mc.κ * (n : ℝ) ^ 3 + 7 * Mc.κ * (n : ℝ) ^ 2 * r + 5 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold honestLow
    apply_rules (maxDepth := 300) [time_mul₂_le_of, time_sub₂_le_of, time_add₂_le_of, Machine.time_pair_le_of,
      Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
  · linarith

variable {n : ℕ} {Inst : Type} {P : Protocol Mc.toMachine Inst (Mat R n n)} {emb : Inst → Mat R n n × Mat R n n}

/-- **Relative security of matmul PoUW with full-rank masks**, from extraction (named hypothesis `Extraction`, overhead
`εext`) and the floor (named hypothesis `Floor`) above the peel's and the correction's cost
`q = 2κ n³ + 2κadd n²`: `ε = εext + (1 + εext) q / (T − q)`. -/
theorem secure_fullRank (hmat : IsMatmulOn P emb) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidFull (emb x)) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η) (hT : 2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelFull n) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelFull_correct (emb x) b hb)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelFull_le n, Mc.time_corr_le n]) hT

/-- **Relative security of matmul PoUW with rank-`r` masks**, from extraction and the floor above
`q = 4κ n² r + 2κadd n²`: `ε = εext + (1 + εext) q / (T − q)`, and `q / (T − q)` is `O(r / n)` against a floor of order
`n³`. (RS-EXT's Lemma U: planting rank-`r` masks is detectable, so for RS-D2 this `Extraction` is expected to fail and
security comes from the floor itself, `core_of_floor`.) -/
theorem secure_lowRank (hmat : IsMatmulOn P emb) (r : ℕ) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidLow (r := r) (emb x)) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η)
    (hT : 4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelLow n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLow_correct (emb x) b hb)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelLow_le n r, Mc.time_corr_le n]) hT

/-- **Full-rank masks by route 2**: with every accepted adversary costing at least `F`,
`ε = εext + (2κ n³ + 2κadd n²) / F`, about `2/c` for `F = c n³` (RS-MODEL §5: says nothing useful). -/
theorem secure_fullRank_advTime (hmat : IsMatmulOn P emb) {εext δ η F : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidFull (emb x)) εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M) (hF : 0 < F) :
    Secure P (εext + (2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) / F) δ η 0 :=
  secure_of_extraction_advTime P (Mc.peelFull n) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelFull_correct (emb x) b hb)) hext hM hF
    (by linarith [Mc.time_peelFull_le n, Mc.time_corr_le n])

/-- **Rank-`r` masks by route 2**: with every accepted adversary costing at least `F`,
`ε = εext + (4κ n² r + 2κadd n²) / F`, i.e. `(4κ r + 2κadd) / (c n)` for `F = c n³` (RS-MODEL §5's table, operands on
the grid). -/
theorem secure_lowRank_advTime (hmat : IsMatmulOn P emb) (r : ℕ) {εext δ η F : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidLow (r := r) (emb x)) εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M) (hF : 0 < F) :
    Secure P (εext + (4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) / F) δ η 0 :=
  secure_of_extraction_advTime P (Mc.peelLow n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLow_correct (emb x) b hb)) hext hM hF
    (by linarith [Mc.time_peelLow_le n r, Mc.time_corr_le n])

end MatMachine

end MatmulPeel

section EditedInstances

/-! ## Route (c): edited instances

The protocol edits both operands, unpeeled: the useful output is the edited product `(A + E_A(σ)) (B + E_B(σ))`,
and the edited product is what is served. Dynamic × dynamic only: both operands are the instance, adversarial, and
both are edited fresh per instance. With no peel there are no correction products, so γ is the proof layer's cost
(commitments, proving), the noise expansion and two additions. The floor is RS-D1E's (named `EditFloor`). -/

/-- The bit-pair projection's algebra (RS-D1E): the mixed second difference of `(A + S) D (B + T)` in the two edits
is `S D T`. The committed parts project to 0: `A D B`, `A D T` and `S D B` each lack one of the two edits. -/
theorem edit_mixed_difference {R : Type} [Ring R] {a b c : ℕ} (A S : Mat R a b) (D : Mat R b b) (B T : Mat R b c) :
    (A + S) * D * (B + T) - (A + S) * D * B - A * D * (B + T) + A * D * B = S * D * T := by
  simp only [Matrix.add_mul, Matrix.mul_add]
  abel

/-- **The bit-pair coefficient** (RS-D1E): editing `A` at `(i, k)` by `d β` and `B` at `(k, l)` by `d' β'`, the
`β β'` coefficient of the prefix-sum target `A' diag(p) B'` (with `p` the prefix's indicator) is `d p_k d'` at
`(i, l)` and 0 elsewhere, whatever the committed `A` and `B`. -/
theorem bitPair_coefficient {R : Type} [CommRing R] {a b c : ℕ} (A : Mat R a b) (B : Mat R b c) (p : Fin b → R)
    (i : Fin a) (k : Fin b) (l : Fin c) (d d' : R) :
    (A + Matrix.single i k d) * Matrix.diagonal p * (B + Matrix.single k l d') -
        (A + Matrix.single i k d) * Matrix.diagonal p * B - A * Matrix.diagonal p * (B + Matrix.single k l d') +
        A * Matrix.diagonal p * B =
      Matrix.single i l (d * p k * d') := by
  rw [edit_mixed_difference, Matrix.single_mul_mul_single, Matrix.diagonal_apply_eq]

/-! ### The abstract core of RS-D1E's homogenization (Lemma 3 and linearity, for one word)

Words are functions of the noise bits `z : X ⊕ Y → K` (x-bits and y-bits). The bit-pair projection of a word `f` is
RS-D1E's mixed second difference, `P(f)[x, y] = f(e_x + e_y) − f(e_x) − f(e_y) + f(0)`. It is linear, kills affine
words, and sends a product of two affine (plain) operands to two rank-one terms; so a word computed with `m` plain
products and linear steps projects to rank at most `2m`, and a word whose projection has rank `r` needs at least
`r / 2` products: the products needed for a word are at least half the rank of its degree-(1, 1) part. Over
`Z/2³²` the rank is too coarse and Theorem W's `r_k` replaces it. Many target words add up when their live terms are
disjoint (`editedEntries_products_ge`); Theorem B's accounting with merges (`F(G)`) is still on paper. -/

/-- An affine (plain) operand in the noise bits: `c + Σ α_x z_x + Σ γ_y z_y`. -/
structure Aff (X Y K : Type) where
  c : K
  α : X → K
  γ : Y → K

/-- Its value at the noise `z`. -/
def Aff.eval {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] (u : Aff X Y K) (z : X ⊕ Y → K) : K :=
  u.c + ∑ x, u.α x * z (.inl x) + ∑ y, u.γ y * z (.inr y)

/-- The noise `s e_x + t e_y`. -/
def bitPt {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] (x : X) (y : Y) (s t : K) : X ⊕ Y → K :=
  Pi.single (.inl x) s + Pi.single (.inr y) t

/-- **The bit-pair projection** (RS-D1E): `P(f)[x, y] = f(e_x + e_y) − f(e_x) − f(e_y) + f(0)`. -/
def bitProj {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] (f : (X ⊕ Y → K) → K) : Matrix X Y K :=
  fun x y => f (bitPt x y 1 1) - f (bitPt x y 1 0) - f (bitPt x y 0 1) + f (bitPt x y 0 0)

theorem Aff.eval_bitPt {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    (u : Aff X Y K) (x : X) (y : Y) (s t : K) : u.eval (bitPt x y s t) = u.c + u.α x * s + u.γ y * t := by
  simp [Aff.eval, bitPt, Pi.single_apply]

/-- **RS-D1E's Lemma 3** over a commutative ring: a product of two plain operands projects to `α_u γ_vᵀ + α_v γ_uᵀ`,
and its constants (the instance, the pair offsets, anything computed before the coins) cancel. -/
theorem bitProj_mul {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    (u v : Aff X Y K) :
    bitProj (fun z => u.eval z * v.eval z) = Matrix.vecMulVec u.α v.γ + Matrix.vecMulVec v.α u.γ := by
  ext x y
  simp only [bitProj, Aff.eval_bitPt, Matrix.add_apply, Matrix.vecMulVec_apply]
  ring

/-- Plain words project to 0. -/
theorem bitProj_aff {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    (u : Aff X Y K) : bitProj (fun z => u.eval z) = 0 := by
  ext x y
  simp only [bitProj, Aff.eval_bitPt, Matrix.zero_apply]
  ring

/-- The projection is additive. -/
theorem bitProj_add {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] (f g : (X ⊕ Y → K) → K) :
    bitProj (fun z => f z + g z) = bitProj f + bitProj g := by
  ext x y
  simp only [bitProj, Matrix.add_apply]
  ring

theorem bitProj_sum {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] {m : ℕ}
    (f : Fin m → (X ⊕ Y → K) → K) : bitProj (fun z => ∑ j, f j z) = ∑ j, bitProj (f j) := by
  ext x y
  simp only [bitProj, Matrix.sum_apply, Finset.sum_sub_distrib, Finset.sum_add_distrib]

/-- A sum of `m` pairs of rank-one terms has rank at most `2m`: it factors through `Fin m ⊕ Fin m`. -/
theorem rank_sum_vecMulVec_le {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] [StrongRankCondition K] {m : ℕ}
    (a c : Fin m → X → K) (b d : Fin m → Y → K) :
    (∑ j, (Matrix.vecMulVec (a j) (b j) + Matrix.vecMulVec (c j) (d j))).rank ≤ 2 * m := by
  let U : Matrix X (Fin m ⊕ Fin m) K := fun x s => Sum.elim (fun j => a j x) (fun j => c j x) s
  let V : Matrix (Fin m ⊕ Fin m) Y K := fun s y => Sum.elim (fun j => b j y) (fun j => d j y) s
  have hUV : ∑ j, (Matrix.vecMulVec (a j) (b j) + Matrix.vecMulVec (c j) (d j)) = U * V := by
    ext x y
    rw [Matrix.mul_apply, Fintype.sum_sum_type]
    simp only [Matrix.sum_apply, Matrix.add_apply, Matrix.vecMulVec_apply, Finset.sum_add_distrib, U, V,
      Sum.elim_inl, Sum.elim_inr]
  rw [hUV]
  refine (Matrix.rank_mul_le_left U V).trans ((Matrix.rank_le_card_width U).trans ?_)
  simp [two_mul]

/-- **The core of RS-D1E's homogenization, for one word**: a word computed with `m` products of plain operands and
linear steps (`Σ_j u_j v_j + r`, scalars absorbed into the operands) projects to rank at most `2m`. -/
theorem bitProj_rank_le {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    [StrongRankCondition K] {m : ℕ} (u v : Fin m → Aff X Y K) (r : Aff X Y K) :
    (bitProj fun z => ∑ j, (u j).eval z * (v j).eval z + r.eval z).rank ≤ 2 * m := by
  rw [bitProj_add (fun z => ∑ j, (u j).eval z * (v j).eval z) (fun z => r.eval z), bitProj_aff, add_zero,
    bitProj_sum (fun j z => (u j).eval z * (v j).eval z)]
  simp only [bitProj_mul]
  exact rank_sum_vecMulVec_le _ _ _ _

/-- **Products needed ≥ half the rank of the degree-(1, 1) part**: if a word computed with `m` plain products agrees
with a target `T` at the four points of every bit pair, then `rank P(T) ≤ 2m`. -/
theorem products_ge_half_rank {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    [StrongRankCondition K] {m : ℕ} (u v : Fin m → Aff X Y K) (r : Aff X Y K) (T : (X ⊕ Y → K) → K)
    (hT : ∀ x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      T (bitPt x y s t) = ∑ j, (u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t) + r.eval (bitPt x y s t)) :
    (bitProj T).rank ≤ 2 * m := by
  have h : bitProj T = bitProj fun z => ∑ j, (u j).eval z * (v j).eval z + r.eval z := by
    ext x y
    simp only [bitProj]
    rw [hT x y 1 1 (.inr rfl) (.inr rfl), hT x y 1 0 (.inr rfl) (.inl rfl), hT x y 0 1 (.inl rfl) (.inr rfl),
      hT x y 0 0 (.inl rfl) (.inl rfl)]
  rw [h]
  exact bitProj_rank_le u v r

/-- The edited prefix-sum entry `(i, l)` of `A′ diag(p) B′` as a word in the noise bits (RS-D1E's Lemma 1 form):
`A′_ik = c_ik + d_ik z_(i,k)` and `B′_kl = c′_kl + d′_kl z′_(k,l)`, the constants `c` adversarial. -/
def editedEntry {K : Type} [CommRing K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c) (p : Fin b → K)
    (i : Fin a) (l : Fin c) (z : (Fin a × Fin b) ⊕ (Fin b × Fin c) → K) : K :=
  ∑ k, (C i k + D i k * z (.inl (i, k))) * p k * (C' k l + D' k l * z (.inr (k, l)))

/-- The bit-pair projection of an edited prefix-sum entry: `Σ_k e_(i,k) e_(k,l)ᵀ · d_ik p_k d′_kl`, the committed
constants gone. -/
theorem bitProj_editedEntry {K : Type} [CommRing K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c) (p : Fin b → K)
    (i : Fin a) (l : Fin c) :
    bitProj (editedEntry C D C' D' p i l) =
      ∑ k, Matrix.vecMulVec (Pi.single (i, k) (D i k * p k)) (Pi.single (k, l) (D' k l)) := by
  let U : Fin b → Aff (Fin a × Fin b) (Fin b × Fin c) K := fun k => ⟨C i k * p k, Pi.single (i, k) (D i k * p k), 0⟩
  let V : Fin b → Aff (Fin a × Fin b) (Fin b × Fin c) K := fun k => ⟨C' k l, 0, Pi.single (k, l) (D' k l)⟩
  have hw : editedEntry C D C' D' p i l =
      fun z => ∑ k, (U k).eval z * (V k).eval z + (⟨0, 0, 0⟩ : Aff _ _ K).eval z := by
    funext z
    simp only [editedEntry, Aff.eval, U, V, Pi.single_apply, Pi.zero_apply, zero_mul, Finset.sum_const_zero,
      add_zero, ite_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  rw [hw, bitProj_add (fun z => ∑ k, (U k).eval z * (V k).eval z), bitProj_aff, add_zero,
    bitProj_sum (fun k z => (U k).eval z * (V k).eval z)]
  simp only [bitProj_mul, U, V, Matrix.zero_vecMulVec, add_zero]

/-- An edited prefix-sum entry's projection has rank at least its live terms `#{k : d_ik p_k d′_kl ≠ 0}` (its
submatrix on the rows `(i, k)` and columns `(k, l)` is their diagonal). -/
theorem card_live_le_rank {K : Type} [Field K] [DecidableEq K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c)
    (p : Fin b → K) (i : Fin a) (l : Fin c) :
    Fintype.card {k // D i k * p k * D' k l ≠ 0} ≤ (bitProj (editedEntry C D C' D' p i l)).rank := by
  have hproj := bitProj_editedEntry C D C' D' p i l
  have hdiag : (bitProj (editedEntry C D C' D' p i l)).submatrix (fun k => (i, k)) (fun k => (k, l)) =
      Matrix.diagonal fun k => D i k * p k * D' k l := by
    rw [hproj]
    ext k k'
    simp only [Matrix.submatrix_apply, Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.single_apply, Prod.mk.injEq,
      true_and, and_true, Matrix.diagonal_apply]
    by_cases h : k = k'
    · subst h
      simp [mul_assoc]
    · simp [h]
  calc Fintype.card {k // D i k * p k * D' k l ≠ 0}
      = (Matrix.diagonal fun k => D i k * p k * D' k l).rank := (Matrix.rank_diagonal _).symm
    _ ≤ (bitProj (editedEntry C D C' D' p i l)).rank := hdiag ▸ Matrix.rank_submatrix_le _ _ _

/-- ... and at most `b`: it factors through the inner index `k`. -/
theorem rank_bitProj_editedEntry_le {K : Type} [Field K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c)
    (p : Fin b → K) (i : Fin a) (l : Fin c) : (bitProj (editedEntry C D C' D' p i l)).rank ≤ b := by
  rw [bitProj_editedEntry]
  let U : Matrix (Fin a × Fin b) (Fin b) K := fun x k => (Pi.single (i, k) (D i k * p k) : Fin a × Fin b → K) x
  let V : Matrix (Fin b) (Fin b × Fin c) K := fun k y => (Pi.single (k, l) (D' k l) : Fin b × Fin c → K) y
  have hUV : ∑ k, Matrix.vecMulVec (Pi.single (i, k) (D i k * p k)) (Pi.single (k, l) (D' k l)) = U * V := by
    ext x y
    simp only [Matrix.sum_apply, Matrix.vecMulVec_apply]
    rfl
  rw [hUV]
  exact (Matrix.rank_mul_le_left U V).trans ((Matrix.rank_le_card_width U).trans (by simp))

/-- Its columns live on the x-bits of row `i`. -/
theorem colSpace_bitProj_editedEntry_le {K : Type} [Field K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c)
    (p : Fin b → K) (i : Fin a) (l : Fin c) :
    SteinerRigidity.colSpace (bitProj (editedEntry C D C' D' p i l)) ≤
      SteinerRigidity.blockSubspace K (fun x : Fin a × Fin b => x.1) i := by
  rw [bitProj_editedEntry]
  rintro _ ⟨y, rfl⟩ x hx
  simp [Matrix.mulVec, dotProduct, Matrix.vecMulVec_apply, Pi.single_apply, Prod.ext_iff, hx]

/-- Its rows live on the y-bits of column `l`. -/
theorem rowSpace_bitProj_editedEntry_le {K : Type} [Field K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c)
    (p : Fin b → K) (i : Fin a) (l : Fin c) :
    SteinerRigidity.rowSpace (bitProj (editedEntry C D C' D' p i l)) ≤
      SteinerRigidity.blockSubspace K (fun y : Fin b × Fin c => y.2) l := by
  rw [bitProj_editedEntry]
  rintro _ ⟨x, rfl⟩ y hy
  simp only at hy
  simp [Matrix.vecMul, dotProduct, Matrix.vecMulVec_apply, Pi.single_apply, Prod.ext_iff, hy]

/-- **RS-D1E's Lemma 2, counted** (over a field): a word computed with `m` plain products that agrees with an
edited prefix-sum entry at every bit pair's four points has `m ≥ #{k : d_ik p_k d′_kl ≠ 0} / 2`, whatever the
committed constants. (One entry; entries with disjoint live terms add up, `editedEntries_products_ge`.) -/
theorem editedEntry_products_ge {K : Type} [Field K] [DecidableEq K] {a b c m : ℕ} (C D : Mat K a b)
    (C' D' : Mat K b c) (p : Fin b → K) (i : Fin a) (l : Fin c)
    (u v : Fin m → Aff (Fin a × Fin b) (Fin b × Fin c) K) (r : Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (hT : ∀ x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      editedEntry C D C' D' p i l (bitPt x y s t) =
        ∑ j, (u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t) + r.eval (bitPt x y s t)) :
    Fintype.card {k // D i k * p k * D' k l ≠ 0} ≤ 2 * m :=
  (card_live_le_rank C D C' D' p i l).trans (products_ge_half_rank u v r _ hT)

/-- A scalar multiple of a plain operand. -/
def Aff.smul {X Y K : Type} [CommRing K] (s : K) (u : Aff X Y K) : Aff X Y K :=
  ⟨s * u.c, fun x => s * u.α x, fun y => s * u.γ y⟩

theorem Aff.eval_smul {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] (s : K) (u : Aff X Y K)
    (z : X ⊕ Y → K) : (u.smul s).eval z = s * u.eval z := by
  simp only [Aff.smul, Aff.eval, mul_add, Finset.mul_sum, mul_assoc]

/-- The sum of plain operands. -/
def Aff.sum {X Y K : Type} [CommRing K] {q : ℕ} (r : Fin q → Aff X Y K) : Aff X Y K :=
  ⟨∑ e, (r e).c, fun x => ∑ e, (r e).α x, fun y => ∑ e, (r e).γ y⟩

theorem Aff.eval_sum {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] {q : ℕ} (r : Fin q → Aff X Y K)
    (z : X ⊕ Y → K) : (Aff.sum r).eval z = ∑ e, (r e).eval z := by
  simp only [Aff.sum, Aff.eval, Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm (s := (Finset.univ : Finset X)), Finset.sum_comm (s := (Finset.univ : Finset Y))]

/-- **Entries with disjoint live terms add up** (RS-D1E's Theorem B accounting without merges, over a field): if `m`
plain products, shared, compute `q` edited prefix-sum entries `(i_e, l_e)` at distinct rows and distinct columns (each
entry's word a linear combination of the products plus a plain word, agreeing at every bit pair's four points), then
`2m ≥ Σ_e #{k : d_(i_e k) p_k d′_(k l_e) ≠ 0}`, whatever the committed constants. -/
theorem editedEntries_products_ge {K : Type} [Field K] [DecidableEq K] {a b c m q : ℕ} (C D : Mat K a b)
    (C' D' : Mat K b c) (p : Fin b → K) (ent : Fin q → Fin a × Fin c)
    (hrow : Function.Injective fun e => (ent e).1) (hcol : Function.Injective fun e => (ent e).2)
    (lam : Fin q → Fin m → K) (u v : Fin m → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (r : Fin q → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (hT : ∀ e x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      editedEntry C D C' D' p (ent e).1 (ent e).2 (bitPt x y s t) =
        ∑ j, lam e j * ((u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t)) + (r e).eval (bitPt x y s t)) :
    ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} ≤ 2 * m := by
  have hr : ∀ e e', (ent e).1 = (ent e').1 ↔ e = e' := fun _ _ => hrow.eq_iff
  have hc : ∀ e e', (ent e).2 = (ent e').2 ↔ e = e' := fun _ _ => hcol.eq_iff
  let W : ((Fin a × Fin b) ⊕ (Fin b × Fin c) → K) → K := fun z => ∑ e, editedEntry C D C' D' p (ent e).1 (ent e).2 z
  have hW : (bitProj W).rank ≤ 2 * m := by
    refine products_ge_half_rank (fun j => (u j).smul (∑ e, lam e j)) v (Aff.sum r) W fun x y s t hs ht => ?_
    simp only [W, Aff.eval_smul, Aff.eval_sum]
    rw [Finset.sum_congr rfl fun e _ => hT e x y s t hs ht, Finset.sum_add_distrib]
    congr 1
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Finset.sum_mul, Finset.sum_mul]
    refine Finset.sum_congr rfl fun e _ => ?_
    ring
  have hproj : bitProj W = ∑ e, ∑ k,
      Matrix.vecMulVec (Pi.single ((ent e).1, k) (D (ent e).1 k * p k)) (Pi.single (k, (ent e).2) (D' k (ent e).2)) := by
    rw [bitProj_sum (fun e => editedEntry C D C' D' p (ent e).1 (ent e).2)]
    exact Finset.sum_congr rfl fun e _ => bitProj_editedEntry C D C' D' p _ _
  have hdiag : (bitProj W).submatrix (fun ek : Fin q × Fin b => ((ent ek.1).1, ek.2))
      (fun ek : Fin q × Fin b => (ek.2, (ent ek.1).2)) =
      Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 := by
    rw [hproj]
    ext ⟨e, k⟩ ⟨e', k'⟩
    simp only [Matrix.submatrix_apply, Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.single_apply, Prod.mk.injEq,
      hr, hc, Matrix.diagonal_apply]
    by_cases h : e = e' ∧ k = k'
    · obtain ⟨rfl, rfl⟩ := h
      simp [ite_and, mul_assoc]
    · simp only [h, ↓reduceIte]
      refine Finset.sum_eq_zero fun e'' _ => Finset.sum_eq_zero fun k'' _ => ?_
      by_cases h1 : e = e'' ∧ k = k''
      · have h2 : ¬(k' = k'' ∧ e' = e'') := fun h2 => h ⟨h1.1.trans h2.2.symm, h1.2.trans h2.1.symm⟩
        simp only [h2, ↓reduceIte, mul_zero]
      · simp only [h1, ↓reduceIte, zero_mul]
  have hcard : ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} =
      Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0} := by
    simp only [Fintype.card_subtype, Finset.card_filter]
    rw [Fintype.sum_prod_type]
  rw [hcard]
  calc Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0}
      = (Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2).rank :=
        (Matrix.rank_diagonal _).symm
    _ ≤ (bitProj W).rank := hdiag ▸ Matrix.rank_submatrix_le _ _ _
    _ ≤ 2 * m := hW

/-- The projection is homogeneous. -/
theorem bitProj_smul {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] (k : K) (f : (X ⊕ Y → K) → K) :
    bitProj (fun z => k * f z) = k • bitProj f := by
  ext x y
  simp only [bitProj, Matrix.smul_apply, smul_eq_mul]
  ring

/-- Each entry's projection, read off the products it combines: `Σ_j λ_ej (α_uj γ_vjᵀ + α_vj γ_ujᵀ)`. -/
theorem bitProj_editedEntry_of_products {K : Type} [CommRing K] {a b c m : ℕ} (C D : Mat K a b) (C' D' : Mat K b c)
    (p : Fin b → K) (i : Fin a) (l : Fin c) (lam : Fin m → K) (u v : Fin m → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (r : Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (hT : ∀ x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      editedEntry C D C' D' p i l (bitPt x y s t) =
        ∑ j, lam j * ((u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t)) + r.eval (bitPt x y s t)) :
    bitProj (editedEntry C D C' D' p i l) =
      ∑ j, lam j • (Matrix.vecMulVec (u j).α (v j).γ + Matrix.vecMulVec (v j).α (u j).γ) := by
  have h : bitProj (editedEntry C D C' D' p i l) =
      bitProj fun z => ∑ j, lam j * ((u j).eval z * (v j).eval z) + r.eval z := by
    ext x y
    simp only [bitProj]
    rw [hT x y 1 1 (.inr rfl) (.inr rfl), hT x y 1 0 (.inr rfl) (.inl rfl), hT x y 0 1 (.inl rfl) (.inr rfl),
      hT x y 0 0 (.inl rfl) (.inl rfl)]
  rw [h, bitProj_add (fun z => ∑ j, lam j * ((u j).eval z * (v j).eval z)) (fun z => r.eval z), bitProj_aff,
    add_zero, bitProj_sum (fun j z => lam j * ((u j).eval z * (v j).eval z))]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [bitProj_smul (lam j) (fun z => (u j).eval z * (v j).eval z), bitProj_mul]

/-- **Entries at distinct columns, rows shared, merges free** (the flattening with the entries on the row side): if
`m` plain products, shared, compute `q` edited prefix-sum entries at distinct columns `l_e`, each entry's word any
linear combination of the products plus a plain word, then `2m ≥ Σ_e r_e`, whatever the committed constants.
Sharing rows and columns both needs Theorem B's charge (`editedEntries_products_ge_shared`). -/
theorem editedEntries_products_ge_cols {K : Type} [Field K] [DecidableEq K] {a b c m q : ℕ} (C D : Mat K a b)
    (C' D' : Mat K b c) (p : Fin b → K) (ent : Fin q → Fin a × Fin c)
    (hcol : Function.Injective fun e => (ent e).2)
    (lam : Fin q → Fin m → K) (u v : Fin m → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (r : Fin q → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (hT : ∀ e x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      editedEntry C D C' D' p (ent e).1 (ent e).2 (bitPt x y s t) =
        ∑ j, lam e j * ((u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t)) + (r e).eval (bitPt x y s t)) :
    ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} ≤ 2 * m := by
  have hc : ∀ e e', (ent e).2 = (ent e').2 ↔ e = e' := fun _ _ => hcol.eq_iff
  let Φ : Matrix (Fin q × (Fin a × Fin b)) (Fin b × Fin c) K := fun ex y =>
    bitProj (editedEntry C D C' D' p (ent ex.1).1 (ent ex.1).2) ex.2 y
  have hΦ : Φ.rank ≤ 2 * m := by
    have e : Φ = ∑ j, (Matrix.vecMulVec (fun ex : Fin q × (Fin a × Fin b) => lam ex.1 j * (u j).α ex.2) (v j).γ +
        Matrix.vecMulVec (fun ex : Fin q × (Fin a × Fin b) => lam ex.1 j * (v j).α ex.2) (u j).γ) := by
      ext ⟨e, x⟩ y
      simp only [Φ, bitProj_editedEntry_of_products C D C' D' p _ _ (lam e) u v (r e) (hT e), Matrix.sum_apply,
        Matrix.smul_apply, Matrix.add_apply, Matrix.vecMulVec_apply, smul_eq_mul]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [e]
    exact rank_sum_vecMulVec_le _ _ _ _
  have hdiag : Φ.submatrix (fun ek : Fin q × Fin b => (ek.1, ((ent ek.1).1, ek.2)))
      (fun ek : Fin q × Fin b => (ek.2, (ent ek.1).2)) =
      Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 := by
    ext ⟨e, k⟩ ⟨e', k'⟩
    rw [Matrix.submatrix_apply]
    simp only [Φ, bitProj_editedEntry, Matrix.sum_apply, Matrix.vecMulVec_apply,
      Pi.single_apply, Prod.mk.injEq, hc, Matrix.diagonal_apply, true_and]
    by_cases h : e = e' ∧ k = k'
    · obtain ⟨rfl, rfl⟩ := h
      simp [mul_assoc]
    · simp only [h, ↓reduceIte]
      refine Finset.sum_eq_zero fun k'' _ => ?_
      by_cases h1 : k = k''
      · have h2 : ¬(k' = k'' ∧ e' = e) := fun h2 => h ⟨h2.2.symm, h1.trans h2.1.symm⟩
        simp only [h2, ↓reduceIte, mul_zero]
      · simp only [h1, ↓reduceIte, zero_mul]
  have hcard : ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} =
      Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0} := by
    simp only [Fintype.card_subtype, Finset.card_filter]
    rw [Fintype.sum_prod_type]
  rw [hcard]
  calc Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0}
      = (Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2).rank :=
        (Matrix.rank_diagonal _).symm
    _ ≤ Φ.rank := hdiag ▸ Matrix.rank_submatrix_le _ _ _
    _ ≤ 2 * m := hΦ

/-- **Entries at distinct rows, columns shared, merges free** (the flattening with the entries on the column
side). -/
theorem editedEntries_products_ge_rows {K : Type} [Field K] [DecidableEq K] {a b c m q : ℕ} (C D : Mat K a b)
    (C' D' : Mat K b c) (p : Fin b → K) (ent : Fin q → Fin a × Fin c)
    (hrow : Function.Injective fun e => (ent e).1)
    (lam : Fin q → Fin m → K) (u v : Fin m → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (r : Fin q → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (hT : ∀ e x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      editedEntry C D C' D' p (ent e).1 (ent e).2 (bitPt x y s t) =
        ∑ j, lam e j * ((u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t)) + (r e).eval (bitPt x y s t)) :
    ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} ≤ 2 * m := by
  have hr : ∀ e e', (ent e).1 = (ent e').1 ↔ e = e' := fun _ _ => hrow.eq_iff
  let Φ : Matrix (Fin a × Fin b) (Fin q × (Fin b × Fin c)) K := fun x ey =>
    bitProj (editedEntry C D C' D' p (ent ey.1).1 (ent ey.1).2) x ey.2
  have hΦ : Φ.rank ≤ 2 * m := by
    have e : Φ = ∑ j, (Matrix.vecMulVec (u j).α (fun ey : Fin q × (Fin b × Fin c) => lam ey.1 j * (v j).γ ey.2) +
        Matrix.vecMulVec (v j).α (fun ey : Fin q × (Fin b × Fin c) => lam ey.1 j * (u j).γ ey.2)) := by
      ext x ⟨e, y⟩
      simp only [Φ, bitProj_editedEntry_of_products C D C' D' p _ _ (lam e) u v (r e) (hT e), Matrix.sum_apply,
        Matrix.smul_apply, Matrix.add_apply, Matrix.vecMulVec_apply, smul_eq_mul]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [e]
    exact rank_sum_vecMulVec_le _ _ _ _
  have hdiag : Φ.submatrix (fun ek : Fin q × Fin b => ((ent ek.1).1, ek.2))
      (fun ek : Fin q × Fin b => (ek.1, (ek.2, (ent ek.1).2))) =
      Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 := by
    ext ⟨e, k⟩ ⟨e', k'⟩
    rw [Matrix.submatrix_apply]
    simp only [Φ, bitProj_editedEntry, Matrix.sum_apply, Matrix.vecMulVec_apply,
      Pi.single_apply, Prod.mk.injEq, hr, Matrix.diagonal_apply, and_true]
    by_cases h : e = e' ∧ k = k'
    · obtain ⟨rfl, rfl⟩ := h
      simp [mul_assoc]
    · simp only [h, ↓reduceIte]
      refine Finset.sum_eq_zero fun k'' _ => ?_
      by_cases h1 : e = e' ∧ k = k''
      · have h2 : k' ≠ k'' := fun h2 => h ⟨h1.1, h1.2.trans h2.symm⟩
        simp only [h2, ↓reduceIte, mul_zero]
      · simp only [h1, ↓reduceIte, zero_mul]
  have hcard : ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} =
      Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0} := by
    simp only [Fintype.card_subtype, Finset.card_filter]
    rw [Fintype.sum_prod_type]
  rw [hcard]
  calc Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0}
      = (Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2).rank :=
        (Matrix.rank_diagonal _).symm
    _ ≤ Φ.rank := hdiag ▸ Matrix.rank_submatrix_le _ _ _
    _ ≤ 2 * m := hΦ

/-! ### Theorem B's accounting over a field (A3): entries that share rows and columns

`editedEntries_products_ge` lets each entry's word be any linear combination of the shared products, a merge in
Theorem B's model, and at distinct rows and columns that costs nothing; nor does it when entries share only rows
(`editedEntries_products_ge_cols`) or only columns (`editedEntries_products_ge_rows`). Once entries share both rows
and columns, free merges are too strong: recursive Strassen on plain operands covers `n³` live terms with
`7^{log₂ n}` products, below `n³/2` from `n = 64`. So Theorem B charges them. Words are accumulators with one
parent, plus heads (merges, conversions, CUDA multiplications, copy edges), and each head whose merge-free tree holds
a target costs at most `b` (Theorem B's `L`). The projected words form a rank-metric Steiner forest; joined to 0
through one target per hosting head, it is a Steiner tree, and Theorem 1 (`SteinerRigidity.blockRigidity`) bounds it
below (`editedEntries_products_ge_shared`). -/

/-- The shape of Theorem B's computation: each word is a head or a step on one parent, an earlier word or the zero
word (`none`). -/
structure Forest (N : ℕ) where
  isHead : Fin N → Bool
  parent : Fin N → Option (Fin N)
  parent_lt : ∀ j k, parent j = some k → k < j

namespace Forest

/-- The root of a word's merge-free tree: the head it hangs from, or `none` for the zero word's tree. -/
def root {N : ℕ} (P : Forest N) (j : Fin N) : Option (Fin N) :=
  if P.isHead j then some j else
    match h : P.parent j with
    | none => none
    | some k =>
      have : k < j := P.parent_lt j k h
      root P k
termination_by j.val
decreasing_by exact this

variable {N : ℕ} (P : Forest N)

theorem root_head {j : Fin N} (h : P.isHead j = true) : P.root j = some j := by
  rw [root]
  simp [h]

theorem root_none {j : Fin N} (h : P.isHead j = false) (hp : P.parent j = none) : P.root j = none := by
  rw [root]
  simp only [h, Bool.false_eq_true, ↓reduceIte]
  split <;> simp_all

theorem root_some {j k : Fin N} (h : P.isHead j = false) (hp : P.parent j = some k) : P.root j = P.root k := by
  rw [root]
  simp only [h, Bool.false_eq_true, ↓reduceIte]
  split <;> simp_all

theorem root_isHead : ∀ (n : ℕ) (j : Fin N), j.val = n → ∀ {h}, P.root j = some h → P.isHead h = true := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro j hj h hr
    cases hh : P.isHead j
    · cases hp : P.parent j with
      | none =>
        rw [P.root_none hh hp] at hr
        cases hr
      | some k =>
        rw [P.root_some hh hp] at hr
        exact ih k.val (hj ▸ P.parent_lt j k hp) k rfl hr
    · rw [P.root_head hh] at hr
      cases hr
      exact hh

/-- Some target's word hangs from the head `h`. -/
def Hosts {q : ℕ} (t : Fin q → Fin N) (h : Fin N) : Prop := ∃ e, P.root (t e) = some h

/-- **The forest is a Steiner tree** (Theorem B's tree decomposition, re-rooted at 0): label the words by
matrices, each step's label at rank distance at most 2 from its parent's (the zero word's label is 0), and the
targets of rank at most `b`. Some Steiner tree reaches 0 and every target's label, with length at most `2m + b H`:
`m` the steps, `H` the heads whose tree holds a target. Each such tree joins 0 through one of its targets (length at
most `b`); a head's tree with no target is shifted by its head's label, which keeps its edge lengths and puts the
head at 0. -/
theorem exists_steinerTree {F : Type} [Field F] {m n : Type} [Fintype m] [Fintype n] {q : ℕ}
    (lab : Fin N → Matrix m n F) (hstep : ∀ j, P.isHead j = false → (lab j - (P.parent j).elim 0 lab).rank ≤ 2)
    (t : Fin q → Fin N) {b : ℕ} (hb : ∀ e, (lab (t e)).rank ≤ b) :
    ∃ T : SteinerRigidity.SteinerTree F m n, T.Reaches 0 ∧ (∀ e, T.Reaches (lab (t e))) ∧
      T.length ≤ 2 * Fintype.card {j // P.isHead j = false} +
        b * Nat.card {h // P.isHead h = true ∧ P.Hosts t h} := by
  classical
  let shift : Fin N → Matrix m n F := fun j =>
    match P.root j with
    | none => 0
    | some h => if P.Hosts t h then 0 else lab h
  let label : Option (Fin N) → Matrix m n F := fun v => v.elim 0 fun j => lab j - shift j
  let anchor : Fin N → Fin N := fun h => if hh : P.Hosts t h then t hh.choose else h
  let E1 : Finset (Sym2 (Option (Fin N))) :=
    (univ.filter fun j => P.isHead j = false).image fun j => s(P.parent j, some j)
  let E2 : Finset (Sym2 (Option (Fin N))) :=
    (univ.filter fun h => P.isHead h = true ∧ P.Hosts t h).image fun h => s(none, some (anchor h))
  let E3 : Finset (Sym2 (Option (Fin N))) :=
    (univ.filter fun h => P.isHead h = true ∧ ¬P.Hosts t h).image fun h => s(none, some h)
  let G : SimpleGraph (Option (Fin N)) := SimpleGraph.fromEdgeSet ↑(E1 ∪ E2 ∪ E3)
  have hadj : ∀ u v, s(u, v) ∈ E1 ∪ E2 ∪ E3 → u ≠ v → G.Adj u v := fun u v h hne =>
    (SimpleGraph.fromEdgeSet_adj _).mpr ⟨Finset.mem_coe.mpr h, hne⟩
  have key : ∀ (n : ℕ) (j : Fin N), j.val = n → G.Reachable (P.root j) (some j) := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro j hj
      cases hh : P.isHead j
      · have hE : s(P.parent j, some j) ∈ E1 ∪ E2 ∪ E3 :=
          mem_union_left _ (mem_union_left _ (mem_image.mpr ⟨j, by simp [hh], rfl⟩))
        cases hp : P.parent j with
        | none =>
          rw [P.root_none hh hp]
          rw [hp] at hE
          exact (hadj _ _ hE (by simp)).reachable
        | some k =>
          rw [P.root_some hh hp]
          rw [hp] at hE
          have hk : k < j := P.parent_lt j k hp
          exact (ih k.val (hj ▸ hk) k rfl).trans (hadj _ _ hE (by simp [hk.ne])).reachable
      · rw [P.root_head hh]
  have key0 : ∀ j, G.Reachable none (P.root j) := by
    intro j
    cases hr : P.root j with
    | none => rfl
    | some h =>
      have hh := P.root_isHead j.val j rfl hr
      by_cases hos : P.Hosts t h
      · have ha : anchor h = t hos.choose := by simp only [anchor, hos, ↓reduceDIte]
        have hE : s(none, some (anchor h)) ∈ E1 ∪ E2 ∪ E3 :=
          mem_union_left _ (mem_union_right _ (mem_image.mpr ⟨h, by simp [hh, hos], rfl⟩))
        have h1 := (hadj _ _ hE (by simp)).reachable
        have h2 := key _ (t hos.choose) rfl
        rw [hos.choose_spec] at h2
        rw [ha] at h1
        exact h1.trans h2.symm
      · have hE : s(none, some h) ∈ E1 ∪ E2 ∪ E3 :=
          mem_union_right _ (mem_image.mpr ⟨h, by simp [hh, hos], rfl⟩)
        exact (hadj _ _ hE (by simp)).reachable
  have hconn : G.Connected := by
    rw [SimpleGraph.connected_iff_exists_forall_reachable]
    refine ⟨none, fun v => ?_⟩
    cases v with
    | none => rfl
    | some j => exact (key0 j).trans (key _ j rfl)
  obtain ⟨T, hTG, hT⟩ := hconn.exists_isTree_le
  let ST : SteinerRigidity.SteinerTree F m n := { V := Option (Fin N), G := T, isTree := hT, label := label }
  have hshift0 : ∀ e, shift (t e) = 0 := by
    intro e
    simp only [shift]
    cases hr : P.root (t e) with
    | none => rfl
    | some h => simp only [show P.Hosts t h from ⟨e, hr⟩, ↓reduceIte]
  have hlen : ∀ x, SteinerRigidity.SteinerTree.edgeLength label x =
      Sym2.lift ⟨fun u v => SteinerRigidity.rankDist (label u) (label v),
        fun _ _ => SteinerRigidity.rankDist_comm _ _⟩ x := fun _ => rfl
  have hsub : ST.G.edgeFinset ⊆ E1 ∪ E2 ∪ E3 := by
    intro x hx
    have hx' : x ∈ G.edgeSet := SimpleGraph.edgeSet_mono hTG (SimpleGraph.mem_edgeFinset.mp hx)
    rw [SimpleGraph.edgeSet_fromEdgeSet] at hx'
    exact Finset.mem_coe.mp hx'.1
  have hunion : ∀ (s s' : Finset (Sym2 (Option (Fin N)))) (f : Sym2 (Option (Fin N)) → ℕ),
      ∑ x ∈ s ∪ s', f x ≤ ∑ x ∈ s, f x + ∑ x ∈ s', f x := fun s s' f => by
    rw [← Finset.sum_union_inter]
    exact Nat.le_add_right _ _
  have h1 : ∑ x ∈ E1, SteinerRigidity.SteinerTree.edgeLength label x ≤
      2 * Fintype.card {j // P.isHead j = false} := by
    refine (Finset.sum_image_le_of_nonneg fun _ _ => Nat.zero_le _).trans ?_
    rw [Fintype.card_subtype, mul_comm, ← smul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun j hj => ?_
    have hh : P.isHead j = false := (Finset.mem_filter.mp hj).2
    rw [hlen, Sym2.lift_mk]
    cases hp : P.parent j with
    | none =>
      have hs : shift j = 0 := by simp only [shift, P.root_none hh hp]
      have := hstep j hh
      rw [hp] at this
      simpa [label, hs, SteinerRigidity.rankDist_zero_left] using this
    | some k =>
      have hs : shift j = shift k := by simp only [shift, P.root_some hh hp]
      have := hstep j hh
      rw [hp] at this
      simp only [label, Option.elim, hs]
      rw [SteinerRigidity.rankDist_comm, SteinerRigidity.rankDist, sub_sub_sub_cancel_right]
      exact this
  have h2 : ∑ x ∈ E2, SteinerRigidity.SteinerTree.edgeLength label x ≤
      b * Nat.card {h // P.isHead h = true ∧ P.Hosts t h} := by
    refine (Finset.sum_image_le_of_nonneg fun _ _ => Nat.zero_le _).trans ?_
    rw [Nat.card_eq_fintype_card, Fintype.card_subtype, mul_comm, ← smul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun h hh => ?_
    have hos : P.Hosts t h := (Finset.mem_filter.mp hh).2.2
    have ha : anchor h = t hos.choose := by simp only [anchor, hos, ↓reduceDIte]
    rw [hlen, Sym2.lift_mk, ha]
    simp only [label, Option.elim, hshift0, sub_zero, SteinerRigidity.rankDist_zero_left]
    exact hb _
  have h3 : ∑ x ∈ E3, SteinerRigidity.SteinerTree.edgeLength label x = 0 := by
    refine Nat.eq_zero_of_le_zero ((Finset.sum_image_le_of_nonneg fun _ _ => Nat.zero_le _).trans ?_)
    refine (Finset.sum_eq_zero fun h hh => ?_).le
    have hh' := (Finset.mem_filter.mp hh).2
    have hs : shift h = lab h := by simp only [shift, P.root_head hh'.1, hh'.2, ↓reduceIte]
    rw [hlen, Sym2.lift_mk]
    simp only [label, Option.elim, hs, sub_self, SteinerRigidity.rankDist_self]
  refine ⟨ST, ⟨none, rfl⟩, fun e => ⟨some (t e), ?_⟩, ?_⟩
  · show lab (t e) - shift (t e) = lab (t e)
    rw [hshift0, sub_zero]
  · calc ST.length = ∑ x ∈ ST.G.edgeFinset, SteinerRigidity.SteinerTree.edgeLength label x := rfl
      _ ≤ ∑ x ∈ E1 ∪ E2 ∪ E3, SteinerRigidity.SteinerTree.edgeLength label x := Finset.sum_le_sum_of_subset hsub
      _ ≤ ∑ x ∈ E1, SteinerRigidity.SteinerTree.edgeLength label x +
          ∑ x ∈ E2, SteinerRigidity.SteinerTree.edgeLength label x +
          ∑ x ∈ E3, SteinerRigidity.SteinerTree.edgeLength label x :=
          (hunion _ _ _).trans (Nat.add_le_add_right (hunion _ _ _) _)
      _ ≤ _ := by rw [h3, add_zero]; exact Nat.add_le_add h1 h2

end Forest

/-- **Theorem B's model on the noise bits** (A3, as RS-D1E §1 uses it): a `Forest` of `N` words, each step exactly
`w_j = w_parent + u_j v_j` with plain operands, each head any word. -/
structure AccumProg (N : ℕ) (X Y K : Type) [Fintype X] [Fintype Y] [CommRing K] extends Forest N where
  val : Fin N → (X ⊕ Y → K) → K
  u : Fin N → Aff X Y K
  v : Fin N → Aff X Y K
  step : ∀ j, isHead j = false → ∀ z, val j z = (parent j).elim 0 (fun k => val k z) + (u j).eval z * (v j).eval z

/-- **Theorem B's accounting over a field, entries sharing rows and columns** (A3's Theorem B at one bit pair, for
RS-D1E's edited prefix-sum entries): if a computation in Theorem B's model commits, for `q` edited prefix-sum
entries at distinct cells `(i_e, l_e)`, rows and columns shared freely, words that agree with the entries at every
bit pair's four points, then `Σ_e #{k : d_(i_e k) p_k d′_(k l_e) ≠ 0} ≤ 2m + b H`, with `m` its products and `H` the
heads whose merge-free tree holds a committed entry, whatever the committed constants. With accumulators only
(`H = 0`) that is `2m ≥ Σ_e r_e`; each hosting head costs at most `b` (Theorem B's `L` per head). -/
theorem editedEntries_products_ge_shared {K : Type} [Field K] [DecidableEq K] {a b c N q : ℕ} (C D : Mat K a b)
    (C' D' : Mat K b c) (p : Fin b → K) (P : AccumProg N (Fin a × Fin b) (Fin b × Fin c) K)
    (ent : Fin q → Fin a × Fin c) (hent : Function.Injective ent) (t : Fin q → Fin N)
    (hT : ∀ e x y s s', (s = 0 ∨ s = 1) → (s' = 0 ∨ s' = 1) →
      P.val (t e) (bitPt x y s s') = editedEntry C D C' D' p (ent e).1 (ent e).2 (bitPt x y s s')) :
    ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} ≤
      2 * Fintype.card {j // P.isHead j = false} + b * Nat.card {h // P.isHead h = true ∧ P.Hosts t h} := by
  classical
  let lab : Fin N → Matrix (Fin a × Fin b) (Fin b × Fin c) K := fun j => bitProj (P.val j)
  have hstep : ∀ j, P.isHead j = false → (lab j - (P.parent j).elim 0 lab).rank ≤ 2 := by
    intro j hj
    have hv : P.val j = fun z => (P.parent j).elim 0 (fun k => P.val k z) + (P.u j).eval z * (P.v j).eval z :=
      funext (P.step j hj)
    have he : lab j - (P.parent j).elim 0 lab = bitProj fun z => (P.u j).eval z * (P.v j).eval z := by
      simp only [lab]
      rw [hv, bitProj_add]
      cases P.parent j with
      | none =>
        have h0 : bitProj (X := Fin a × Fin b) (Y := Fin b × Fin c) (fun _ => (0 : K)) = 0 := by
          ext x y
          simp [bitProj]
        simp [h0]
      | some k => simp
    rw [he, bitProj_mul]
    exact (SteinerRigidity.rank_add_le _ _).trans
      (add_le_add (Matrix.rank_vecMulVec_le _ _) (Matrix.rank_vecMulVec_le _ _))
  have hagree : ∀ e, lab (t e) = bitProj (editedEntry C D C' D' p (ent e).1 (ent e).2) := by
    intro e
    ext x y
    simp only [lab, bitProj]
    rw [hT e x y 1 1 (.inr rfl) (.inr rfl), hT e x y 1 0 (.inr rfl) (.inl rfl), hT e x y 0 1 (.inl rfl) (.inr rfl),
      hT e x y 0 0 (.inl rfl) (.inl rfl)]
  obtain ⟨T, h0, hreach, hlen⟩ := P.toForest.exists_steinerTree lab hstep t fun e => by
    rw [hagree e]
    exact rank_bitProj_editedEntry_le C D C' D' p _ _
  have hB := SteinerRigidity.blockRigidity (F := K) (m := Fin a × Fin b) (n := Fin b × Fin c) a c
    (SteinerRigidity.blockSubspace K fun x : Fin a × Fin b => x.1)
    (SteinerRigidity.blockSubspace K fun y : Fin b × Fin c => y.2)
    (SteinerRigidity.iSupIndep_blockSubspace _) (SteinerRigidity.iSupIndep_blockSubspace _) (univ.image ent)
    (fun cell => bitProj (editedEntry C D C' D' p cell.1 cell.2))
    (fun cell _ => ⟨colSpace_bitProj_editedEntry_le C D C' D' p _ _,
      rowSpace_bitProj_editedEntry_le C D C' D' p _ _⟩) T h0 fun cell hcell => by
      obtain ⟨e, _, rfl⟩ := mem_image.mp hcell
      rw [← hagree e]
      exact hreach e
  calc ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0}
      ≤ ∑ e, (bitProj (editedEntry C D C' D' p (ent e).1 (ent e).2)).rank :=
        Finset.sum_le_sum fun e _ => card_live_le_rank C D C' D' p _ _
    _ = ∑ cell ∈ univ.image ent, (bitProj (editedEntry C D C' D' p cell.1 cell.2)).rank :=
        (Finset.sum_image (f := fun cell : Fin a × Fin c => (bitProj (editedEntry C D C' D' p cell.1 cell.2)).rank)
          fun e _ e' _ h => hent h).symm
    _ ≤ T.length := hB
    _ ≤ _ := hlen

/-- **The window's rounds** (Theorem S's and S_G's setting, a named hypothesis): the service's coins are `R` rounds
of fresh coins on `Ω` (`e`), and every adversary that commits first plays, at each value `r` of its own coins, a
round game whose rounds meet the exponential bound `E[1[pass_w] e^{(1−p) d_w}] ≤ 1` given the past (`ExpBound`; from
the pass bound along guard paths, `expBound_of_pathBound`, or from a round's stages, `expBound_of_staged`). Its accepted runs are accepted by the game or break binding or zero knowledge in some
round (probabilities at most `δb` and `δz`), and its charge is at least `(R − Σ_w d_w) F₀ − V − R sl` with noise
reads `V ≤ Vmax` (Theorem E½ per round, `x_w F₀ − L(M + K)` with `sl = L(M + K)`, and Lemma RD's `n v_λ`). -/
def RoundsHyp (Mc : Machine) (S : Service) (Ω : Type) [Fintype Ω] (R : ℕ) (p F₀ sl Vmax δb δz : ℝ) : Prop :=
  ∃ e : S.Seed ≃ Paths Ω R, ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
    ∃ (G : RoundGame Ω R) (Bind ZK : Fin R → Paths Ω R → Prop) (V : Paths Ω R → ℝ),
      G.ExpBound (1 - p) ∧ (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, (R - G.deficiency x) * F₀ - V x - R * sl ≤ Mc.charge M (e.symm x, r)

/-- **The window's service floor from Theorem S** (in place of Theorem M): under the window's rounds, every adversary
that commits first has its accepted runs charged below `(R − D) F₀ − Vmax − R sl` with probability at most
`e^{−(1−p) D} + R (δb + δz)`. -/
theorem serviceFloor_of_rounds {Mc : Machine} {S : Service} {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb δz : ℝ} (h : RoundsHyp Mc S Ω R p F₀ sl Vmax δb δz) (hp : p ≤ 1) (hF₀ : 0 < F₀)
    (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) (D : ℝ) :
    ServiceFloor Mc S ((R - D) * F₀ - Vmax - R * sl) (Real.exp (-((1 - p) * D)) + R * (δb + δz)) := by
  obtain ⟨e, h⟩ := h
  intro Ω' _ M hc
  refine prob_le_of_slices _ (add_nonneg (Real.exp_pos _).le (mul_nonneg (Nat.cast_nonneg _) (add_nonneg hδb hδz)))
    fun r => ?_
  obtain ⟨G, Bind, ZK, V, hG, hB, hZ, hacc, hV, hcost⟩ := h Ω' M hc r
  rw [prob_equiv e]
  refine (prob_mono fun x hx => ⟨hx.1, hx.2.trans_le ?_⟩).trans
    (G.window_le_exp hp hG _ Bind ZK hB hZ hacc (fun x => Mc.charge M (e.symm x, r)) V hF₀ hcost D)
  linarith [hV x]

/-- **One slice of the window's rounds**: `RoundsHyp`'s conclusion at one adversary and one value of its coins, with
the acceptance `acc`, the charge `cost` and the binding and zero-knowledge breaks named rather than bounded, so that
a bound per prover can read the breaks of each slice. -/
def RoundsSlice {Ω : Type} [Fintype Ω] {R : ℕ} (p F₀ sl Vmax : ℝ) (acc : Paths Ω R → Prop) (cost : Paths Ω R → ℝ)
    (Bind ZK : Fin R → Paths Ω R → Prop) : Prop :=
  ∃ (G : RoundGame Ω R) (V : Paths Ω R → ℝ), G.ExpBound (1 - p) ∧
    (∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧ (∀ x, V x ≤ Vmax) ∧
    ∀ x, (R - G.deficiency x) * F₀ - V x - R * sl ≤ cost x

/-- **Theorem S on one slice**: its accepted runs charged below `(R − D) F₀ − Vmax − R sl` have probability at most
`e^{−(1−p) D}` plus the slice's breaks, summed over the rounds. -/
theorem RoundsSlice.prob_le {Ω : Type} [Fintype Ω] {R : ℕ} {p F₀ sl Vmax : ℝ} {acc : Paths Ω R → Prop}
    {cost : Paths Ω R → ℝ} {Bind ZK : Fin R → Paths Ω R → Prop} (h : RoundsSlice p F₀ sl Vmax acc cost Bind ZK)
    (hp : p ≤ 1) (hF₀ : 0 < F₀) (D : ℝ) :
    prob (fun x => acc x ∧ cost x < (R - D) * F₀ - Vmax - R * sl) ≤
      Real.exp (-((1 - p) * D)) + ∑ w, (prob (Bind w) + prob (ZK w)) := by
  obtain ⟨G, V, hG, hacc, hV, hcost⟩ := h
  have hbad : prob (fun x => ∃ w ∈ (univ : Finset (Fin R)), Bind w x ∨ ZK w x) ≤
      ∑ w, (prob (Bind w) + prob (ZK w)) :=
    (prob_exists_le _ _).trans (Finset.sum_le_sum fun w _ => prob_or_le _ _)
  refine (prob_mono fun x hx => ?_).trans
    ((prob_or_le _ _).trans (add_le_add (G.prob_accepts_le (by linarith) hG D) hbad))
  have h1 := (hcost x).trans_lt hx.2
  have h2 : (R - G.deficiency x) * F₀ < (R - D) * F₀ := by linarith [hV x]
  have hD := (mul_lt_mul_iff_of_pos_right hF₀).mp h2
  rcases hacc x hx.1 with h | ⟨w, hw⟩
  · exact Or.inl ⟨h, by linarith⟩
  · exact Or.inr ⟨w, mem_univ _, hw⟩

/-- A finder's collision probability is at most its `ecr/sha-512` bound at any budget: the bound is `⊤` unless the
finder meets the assumption within the budget. -/
theorem ofReal_finderPr_le_ecrBound (H : Claims.Msg → Claims.Msg) (F : Claims.Finder) (q : ℝ) :
    ENNReal.ofReal (F.Pr H) ≤ Claims.ecrBound H F q := by
  unfold Claims.ecrBound
  split_ifs with h
  · have h1 : F.Pr H ≤ Claims.Ex (fun s => (F.cost s : ℝ)) / 2 ^ (256 : ℝ) := h.1
    exact ENNReal.ofReal_le_ofReal (h1.trans (div_le_div_of_nonneg_right h.2 (by positivity)))
  · exact le_top

/-- **The service floor per prover from the window's slices**: if every adversary that commits first has, at each
value `r` of its coins, a slice of the window's rounds whose round-`w` breaks are at most `δb + βb · Pr[fb … w]` and
`δz + βz · Pr[fz … w]`, then its accepted runs charged below `(R − D) F₀ − Vmax − R sl` have probability at most its
`breakBound` at `ε ≥ e^{−(1−p) D}`, at any budget constants: each slice's bound is at most 1 and at most
`ε + R (δb + δz)` plus its finder terms (`Claims.ecrBound` bounds each finder at every budget), and the slices
average. -/
theorem serviceFloorPerProver_of_slices {Mc : Machine} {S : Service} {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb βb δz βz : ℝ} (e : S.Seed ≃ Paths Ω R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders Mc S R)
    (h : ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
      (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      ∃ Bind ZK : Fin R → Paths Ω R → Prop,
        (∀ w, prob (Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧ (∀ w, prob (ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
        RoundsSlice p F₀ sl Vmax
          (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
          (fun x => Mc.charge M (e.symm x, r)) Bind ZK)
    (hp : p ≤ 1) (hF₀ : 0 < F₀) (hβb : 0 ≤ βb) (hβz : 0 ≤ βz) (cb db cz dz : ℝ) {ε : ℝ} (D : ℝ)
    (hε : Real.exp (-((1 - p) * D)) ≤ ε) :
    ServiceFloorPerProver Mc S ((R - D) * F₀ - Vmax - R * sl) (breakBound sha fb fz cb db cz dz ε δb βb δz βz) := by
  intro Ω' _ M hc
  have hc0 : (0 : ℝ) ≤ (Fintype.card Ω' : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  have hinv : ENNReal.ofReal (Fintype.card Ω' : ℝ)⁻¹ ≤ (Fintype.card Ω' : ENNReal)⁻¹ := by
    rcases Nat.eq_zero_or_pos (Fintype.card Ω') with h0 | h0
    · simp [h0]
    · have hpos : (0 : ℝ) < Fintype.card Ω' := by exact_mod_cast h0
      rw [ENNReal.ofReal_inv_of_pos hpos, ENNReal.ofReal_natCast]
  have hfd : ∀ (fd : Finders Mc S R) (q β : ℝ) (r : Ω'), 0 ≤ β →
      ENNReal.ofReal (β * ∑ w, (fd Ω' M r w).Pr sha) ≤ ENNReal.ofReal β * finderTerm sha fd q Ω' M r := by
    intro fd q β r hβ
    rw [ENNReal.ofReal_mul hβ, ENNReal.ofReal_sum_of_nonneg fun w _ => ?_]
    · unfold finderTerm
      exact mul_le_mul' le_rfl (Finset.sum_le_sum fun w _ => ofReal_finderPr_le_ecrBound _ _ _)
    · unfold Claims.Finder.Pr Claims.Pr
      positivity
  rw [prob_prod_eq_sum, ENNReal.ofReal_sum_of_nonneg fun r _ => ?_]
  swap
  · exact mul_nonneg hc0 (prob_nonneg _)
  unfold breakBound
  refine Finset.sum_le_sum fun r _ => ?_
  rw [ENNReal.ofReal_mul hc0]
  refine mul_le_mul' hinv (le_min (ENNReal.ofReal_le_one.mpr (prob_le_one _)) ?_)
  obtain ⟨Bind, ZK, hB, hZ, hsl⟩ := h Ω' M hc r
  have hs1 : ∑ w : Fin R, (prob (Bind w) + prob (ZK w)) ≤
      ∑ w : Fin R, ((δb + δz) + (βb * (fb Ω' M r w).Pr sha + βz * (fz Ω' M r w).Pr sha)) :=
    Finset.sum_le_sum fun w _ => by linarith [hB w, hZ w]
  have hs2 : ∑ w : Fin R, ((δb + δz) + (βb * (fb Ω' M r w).Pr sha + βz * (fz Ω' M r w).Pr sha)) =
      R * (δb + δz) + ((βb * ∑ w, (fb Ω' M r w).Pr sha) + βz * ∑ w, (fz Ω' M r w).Pr sha) := by
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.sum_add_distrib, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, ← Finset.mul_sum, ← Finset.mul_sum]
  have hreal : prob (fun σ : S.Seed => S.verify (Mc.eval M (σ, r)).1 σ (Mc.eval M (σ, r)).2 = true ∧
      Mc.charge M (σ, r) < (R - D) * F₀ - Vmax - R * sl) ≤
      ε + R * (δb + δz) + (βb * ∑ w, (fb Ω' M r w).Pr sha) + βz * ∑ w, (fz Ω' M r w).Pr sha := by
    rw [prob_equiv e]
    refine (hsl.prob_le hp hF₀ D).trans ?_
    linarith [hs1, hs2, hε]
  refine (ENNReal.ofReal_le_ofReal hreal).trans ?_
  refine (ENNReal.ofReal_add_le.trans (add_le_add ENNReal.ofReal_add_le le_rfl)).trans ?_
  exact add_le_add (add_le_add le_rfl (hfd fb _ βb r hβb)) (hfd fz _ βz r hβz)

/-- Both operands lie in `G_use = [−g, g]` (RS-DEF v2 §1.10). -/
def InGrid {n : ℕ} (g : ℤ) (x : Mat ℤ n n × Mat ℤ n n) : Prop := ∀ i j, |x.1 i j| ≤ g ∧ |x.2 i j| ≤ g

/-- **A range-checking service** (RS-DEF v2 §1.3 after RS-RT's V6): the commitment binds the instance `(A_M, B_M)`
(`committed`), and for each block σ opens (`opened`; every entry for a full check) the verifier's relation includes
the range check `A_M, B_M ∈ G_use = [−g, g]`, proved inside the zero-knowledge proof. -/
structure GridService (n : ℕ) (g : ℤ) extends Service where
  committed : Com → Mat ℤ n n × Mat ℤ n n
  opened : Seed → Fin n → Fin n → Prop
  range_check : ∀ c σ a, verify c σ a = true → ∀ i j, opened σ i j →
    |(committed c).1 i j| ≤ g ∧ |(committed c).2 i j| ≤ g

/-- With every entry opened, an accepted transcript's committed instance is in `G_use`. -/
theorem GridService.inGrid_of_accepts {n : ℕ} {g : ℤ} (S : GridService n g) (hall : ∀ σ i j, S.opened σ i j)
    {c : S.Com} {σ : S.Seed} {a : S.Ans} (h : S.verify c σ a = true) : InGrid g (S.committed c) :=
  fun i j => S.range_check c σ a h i j (hall σ i j)

namespace MatMachine

variable {R : Type} [Ring R] (Mc : MatMachine R)

/-- **An edit** (route (c)): a σ-indexed map on each operand, `A ↦ A + E_A(σ)` and `B ↦ B + E_B(σ)`, with the
algorithm `gen` that expands σ into the two noise matrices (its time is part of γ). -/
structure Edit (Seed : Type) (n : ℕ) where
  EA : Seed → Mat R n n
  EB : Seed → Mat R n n
  gen : Mc.Alg Seed (Mat R n n × Mat R n n)
  eval_gen : ∀ σ, Mc.eval gen σ = (EA σ, EB σ)

variable {Mc} in
/-- The edited instance `(A + E_A(σ), B + E_B(σ))`. -/
def Edit.edit {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (σ : Seed) (x : Mat R n n × Mat R n n) :
    Mat R n n × Mat R n n :=
  (x.1 + E.EA σ, x.2 + E.EB σ)

variable {Mc} in
/-- The edited product `(A + E_A(σ)) (B + E_B(σ))`, the useful output on the instance `(A, B)`. -/
def Edit.product {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (σ : Seed) (x : Mat R n n × Mat R n n) : Mat R n n :=
  (x.1 + E.EA σ) * (x.2 + E.EB σ)

/-- The noise is bounded by `s` (`|E| ≤ max|S|`, RS-DEF v2 §1.10). -/
def Edit.Bounded {Mc : MatMachine ℤ} {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (s : ℤ) : Prop :=
  ∀ σ i j, |E.EA σ i j| ≤ s ∧ |E.EB σ i j| ≤ s

/-- The edited product as an algorithm, with the product computed by `m`, `(σ, (A, B)) ↦ m (A + E_A σ, B + E_B σ)`:
expand σ once, two additions, one run of `m`, and no peel. `m` is the worst-case `mul n n n` on unrestricted
operands, or a cheaper product on a grid the edited operands stay on (`GridMatMachine`). -/
def editMul {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n)) :
    Mc.Alg (Seed × (Mat R n n × Mat R n n)) (Mat R n n) :=
  Mc.seq (Mc.pair Mc.snd (Mc.seq Mc.fst E.gen))
    (Mc.lift2 (Mc.add₂ (Mc.seq Mc.fst Mc.fst) (Mc.seq Mc.snd Mc.fst))
      (Mc.add₂ (Mc.seq Mc.fst Mc.snd) (Mc.seq Mc.snd Mc.snd)) m)

theorem eval_editMul {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (y : Seed × (Mat R n n × Mat R n n)) : Mc.eval (Mc.editMul E m) y = Mc.eval m (E.edit y.1 y.2) := by
  simp only [editMul, Machine.eval_lift2, eval_add₂, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd,
    E.eval_gen, Edit.edit]

theorem time_editMul_le {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n)) :
    Mc.time (Mc.editMul E m) ≤ Mc.time E.gen + Mc.time m + 2 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold editMul Machine.lift2
    apply_rules [time_add₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le,
      Machine.time_snd_le, le_refl]
  · linarith

/-- **Honest Solve on edited instances** (`honestEdit`): compute the edited product with `m`, then run the proof layer
`prove` on the transcript's inputs and the served product; the useful output is the edited product, unpeeled. -/
def honestEdit {Seed Coins Pf : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((Seed × (Mat R n n × Mat R n n) × Coins) × Mat R n n) Pf) :
    Mc.Alg (Seed × (Mat R n n × Mat R n n) × Coins) (Pf × Mat R n n) :=
  Mc.seq (Mc.pair (Mc.pair Mc.fst Mc.snd) (Mc.seq (Mc.pair Mc.fst (Mc.seq Mc.snd Mc.fst)) (Mc.editMul E m)))
    (Mc.pair prove Mc.snd)

theorem eval_honestEdit {Seed Coins Pf : Type} {n : ℕ} (E : Mc.Edit Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((Seed × (Mat R n n × Mat R n n) × Coins) × Mat R n n) Pf)
    (y : Seed × (Mat R n n × Mat R n n) × Coins) :
    Mc.eval (Mc.honestEdit E m prove) y =
      (Mc.eval prove (y, Mc.eval m (E.edit y.1 y.2.1)), Mc.eval m (E.edit y.1 y.2.1)) := by
  simp only [honestEdit, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd, eval_editMul, Prod.mk.eta]

/-- `honestEdit` costs one run of `m` and two additions beyond the proof layer and the noise expansion:
`time ≤ time m + 2 κadd n² + time gen + time prove`, with no peel. -/
theorem time_honestEdit_le {Seed Coins Pf : Type} {n : ℕ} (E : Mc.Edit Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((Seed × (Mat R n n × Mat R n n) × Coins) × Mat R n n) Pf) :
    Mc.time (Mc.honestEdit E m prove) ≤
      Mc.time m + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove := by
  apply le_trans
  · unfold honestEdit
    apply_rules [Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le,
      time_editMul_le, le_refl]
  · linarith [Mc.time_add_le n n]

/-- **The edited protocol** on the instance `(A, B)` (both operands adversarial, both edited), a promise problem on
the instances `use` (all of them, or those whose edits stay on a grid): the service's coins, messages and verifier,
the edited product as the answer on every promised instance, and `honestEdit` with the product `m` as Solve. -/
def edited (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) :
    Protocol Mc.toMachine (Mat R n n × Mat R n n) (Mat R n n) where
  Seed := S.Seed
  lam := S.lam
  card_seed := S.card_seed
  Com := S.Com
  Ans := S.Ans
  Coins := S.Coins
  correct σ x C := E.product σ x = C
  promise := use
  solve := Mc.honestEdit E m prove
  verify := S.verify

/-- **Completeness of the edited protocol**: if `m` computes the edited product on every promised instance (`hm`), the
proof layer's commitment reads neither σ nor the product (`hcom`, commit-first) and the verifier accepts its proof of
the served product with probability at least `1 − η` on every promised instance (`hacc`), the edited protocol is
complete. -/
theorem edited_complete (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {η : ℝ}
    (hm : ∀ σ x, use x → Mc.eval m (E.edit σ x) = E.product σ x)
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).2 = true) :
    Complete (Mc.edited S E m prove use) η := by
  refine ⟨fun σ σ' y => ?_, fun x hx => (hacc x hx).trans (prob_mono fun ω hω => ?_)⟩
  · change ((Mc.eval (Mc.honestEdit E m prove) (σ, y)).1).1 = ((Mc.eval (Mc.honestEdit E m prove) (σ', y)).1).1
    rw [eval_honestEdit, eval_honestEdit]
    exact hcom σ σ' y _ _
  · change E.product ω.1 x = (Mc.eval (Mc.honestEdit E m prove) (ω.1, x, ω.2)).2 ∧
      S.verify (Mc.eval (Mc.honestEdit E m prove) (ω.1, x, ω.2)).1.1 ω.1
        (Mc.eval (Mc.honestEdit E m prove) (ω.1, x, ω.2)).1.2 = true
    rw [eval_honestEdit]
    exact ⟨(hm ω.1 x hx).symm, hω⟩

/-- **The edit floor** (a named hypothesis; RS-D1E proves it on paper by the bit-pair projection, `bitPair_coefficient`,
with `F = c n³` per MAC slot, `c = ½` unconditional through Theorem W and `c = 1` under Conjecture A): on the edited
protocol, every adversary that commits first has its accepted runs below `F` with probability at most `κ`. -/
def EditFloor (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) (F κ : ℝ) : Prop :=
  AdvFloor (Mc.edited S E m prove use) F κ

/-- The edit floor reads only the service: it is the service's floor in the machine. -/
theorem editFloor_iff (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) (F κ : ℝ) :
    Mc.EditFloor S E m prove use F κ ↔ ServiceFloor Mc.toMachine S F κ :=
  Iff.rfl

/-- **Relative security of edited instances** (route (c), end to end, for a product `m` at `κm` per
multiply-accumulate on the promised instances): with the edited protocol complete (`hm`, `hcom`, `hacc`) and the edit
floor `F` (named hypothesis `EditFloor`, with `κf < δ`), the edited protocol is relatively secure against the reference
`N = κm n³` with `ε = U / F − 1`, `U = κm n³ + 2κadd n² + time gen + time prove`, and
`γ = (2κadd n² + time gen + time prove) / (κm n³)`: the proof layer, the noise expansion and two additions, no peel. -/
theorem relativelySecure_of_editFloor (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {κm κf δ η F : ℝ}
    (hm : ∀ σ x, use x → Mc.eval m (E.edit σ x) = E.product σ x) (hmt : Mc.time m ≤ κm * (n : ℝ) ^ 3)
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).2 = true)
    (hfloor : Mc.EditFloor S E m prove use F κf) (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < κm * (n : ℝ) ^ 3) :
    RelativelySecure (Mc.edited S E m prove use)
      ((κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1) δ
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (κm * (n : ℝ) ^ 3)) η 0
      (κm * (n : ℝ) ^ 3) := by
  have hc := Mc.edited_complete S E m prove use hm hcom hacc
  have hU : Mc.time (Mc.honestEdit E m prove) ≤
      κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove := by
    linarith [Mc.time_honestEdit_le E m prove]
  refine ⟨hc, secureInf_of_core _ (core_of_floor _ hF hFU hfloor hc hU) hκδ ?_, ?_⟩
  · rw [add_sub_cancel]
    exact div_pos (hF.trans_le hFU) hF
  · change Mc.time (Mc.honestEdit E m prove) ≤ _
    rw [add_mul, one_mul, div_mul_cancel₀ _ hN.ne', add_zero]
    linarith

/-- **Relative security of edited instances per adversary** (route (c) per adversary, as
`relativelySecure_of_editFloor`, which it supersedes): the service's floor per adversary (`ServiceFloorPerProver`, the
edit floor read adversary by adversary) gives relative security per adversary at thresholds `κf(M) + δ'`, with
`relativelySecure_of_editFloor`'s `ε`, `γ` and `N`. -/
theorem relativelySecurePerProver_of_editFloor (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {κm δ' η F : ℝ}
    {κf : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal}
    (hm : ∀ σ x, use x → Mc.eval m (E.edit σ x) = E.product σ x) (hmt : Mc.time m ≤ κm * (n : ℝ) ^ 3)
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).2 = true)
    (hfloor : ServiceFloorPerProver Mc.toMachine S F κf) (hδ' : 0 < δ') (hF : 0 < F)
    (hFU : F ≤ κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < κm * (n : ℝ) ^ 3) :
    RelativelySecurePerProver (Mc.edited S E m prove use)
      ((κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1)
      (fun Ω _ M => κf Ω M + ENNReal.ofReal δ')
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (κm * (n : ℝ) ^ 3)) η 0
      (κm * (n : ℝ) ^ 3) := by
  have hc := Mc.edited_complete S E m prove use hm hcom hacc
  have hU : Mc.time (Mc.honestEdit E m prove) ≤
      κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove := by
    linarith [Mc.time_honestEdit_le E m prove]
  refine ⟨hc, secureInfPerProver_of_floor (Mc.edited S E m prove use) (κ := κf) hF hFU hfloor hc hU hδ', ?_⟩
  change Mc.time (Mc.honestEdit E m prove) ≤ _
  rw [add_mul, one_mul, div_mul_cancel₀ _ hN.ne', add_zero]
  linarith

/-- The worst-case product computes the edited product on every instance. -/
theorem eval_mul_edit {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (σ : Seed) (x : Mat R n n × Mat R n n) :
    Mc.eval (Mc.mul n n n) (E.edit σ x) = E.product σ x :=
  Mc.eval_mul n n n _

/-- **Route (c) on unrestricted operands** (`relativelySecure_of_editFloor` with the worst-case product `mul n n n`
at `κ`, and every instance promised): reference `N = κ n³`. -/
theorem relativelySecure_of_editFloor_wide (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans)) {κf δ η F : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), E.product ω.1 x)).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), E.product ω.1 x)).2 = true)
    (hfloor : Mc.EditFloor S E (Mc.mul n n n) prove (fun _ => True) F κf) (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < Mc.κ * (n : ℝ) ^ 3) :
    RelativelySecure (Mc.edited S E (Mc.mul n n n) prove fun _ => True)
      ((Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1) δ
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (Mc.κ * (n : ℝ) ^ 3)) η 0
      (Mc.κ * (n : ℝ) ^ 3) :=
  Mc.relativelySecure_of_editFloor S E _ prove _ (fun σ x _ => Mc.eval_mul_edit E σ x)
    ((Mc.time_mul_le n n n).trans_eq (by ring)) hcom (fun x _ => by simpa only [Mc.eval_mul_edit] using hacc x)
    hfloor hκδ hF hFU hN

end MatMachine

/-- **A matrix machine with a grid product** (the width-typed refinement): on the shapes `shape` and on operands
whose entries lie in the grid `G`, `mulG` computes the product at `κG` per multiply-accumulate; off the grid it may
compute anything. `mul` stays the worst-case product on all operands. -/
structure GridMatMachine (R : Type) [Ring R] extends MatMachine R where
  G : R → Prop
  shape : ℕ → ℕ → ℕ → Prop
  κG : ℝ
  mulG : ∀ a b c : ℕ, Alg (Mat R a b × Mat R b c) (Mat R a c)
  eval_mulG : ∀ (a b c : ℕ) (x : Mat R a b × Mat R b c), shape a b c → (∀ i k, G (x.1 i k)) → (∀ k j, G (x.2 k j)) →
    eval (mulG a b c) x = x.1 * x.2
  time_mulG_le : ∀ a b c : ℕ, shape a b c → time (mulG a b c) ≤ κG * ((a : ℝ) * b * c)

namespace GridMatMachine

variable {R : Type} [Ring R] (Mc : GridMatMachine R)

/-- Both operands lie on the grid. -/
def OnGrid {n : ℕ} (x : Mat R n n × Mat R n n) : Prop := (∀ i k, Mc.G (x.1 i k)) ∧ ∀ k j, Mc.G (x.2 k j)

/-- **Route (c) on the grid**: when every promised instance's edits stay on the grid (`hgrid`) and the shape is one
the grid product takes, the edited protocol with `mulG` is relatively secure against the grid reference
`N = κG n³`, with `U = κG n³ + 2κadd n² + time gen + time prove`. -/
theorem relativelySecure_of_editFloor_grid (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {κf δ η F : ℝ} (hshape : Mc.shape n n n)
    (hgrid : ∀ σ x, use x → Mc.OnGrid (E.edit σ x))
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval (Mc.mulG n n n) (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval (Mc.mulG n n n) (E.edit ω.1 x))).2 = true)
    (hfloor : Mc.EditFloor S E (Mc.mulG n n n) prove use F κf) (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < Mc.κG * (n : ℝ) ^ 3) :
    RelativelySecure (Mc.edited S E (Mc.mulG n n n) prove use)
      ((Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1) δ
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (Mc.κG * (n : ℝ) ^ 3)) η 0
      (Mc.κG * (n : ℝ) ^ 3) :=
  Mc.relativelySecure_of_editFloor S E _ prove use
    (fun σ x hx => Mc.eval_mulG n n n _ hshape (hgrid σ x hx).1 (hgrid σ x hx).2)
    ((Mc.time_mulG_le n n n hshape).trans_eq (by ring)) hcom hacc hfloor hκδ hF hFU hN

/-- **Route (c) on the grid per adversary** (as `relativelySecure_of_editFloor_grid`, which it supersedes): the
service's floor per adversary gives relative security per adversary at thresholds `κf(M) + δ'`. -/
theorem relativelySecurePerProver_of_editFloor_grid (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {δ' η F : ℝ}
    {κf : ∀ (Ω : Type) [Fintype Ω], Mc.Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal} (hshape : Mc.shape n n n)
    (hgrid : ∀ σ x, use x → Mc.OnGrid (E.edit σ x))
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval (Mc.mulG n n n) (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval (Mc.mulG n n n) (E.edit ω.1 x))).2 = true)
    (hfloor : ServiceFloorPerProver Mc.toMachine S F κf) (hδ' : 0 < δ') (hF : 0 < F)
    (hFU : F ≤ Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < Mc.κG * (n : ℝ) ^ 3) :
    RelativelySecurePerProver (Mc.edited S E (Mc.mulG n n n) prove use)
      ((Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1)
      (fun Ω _ M => κf Ω M + ENNReal.ofReal δ')
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (Mc.κG * (n : ℝ) ^ 3)) η 0
      (Mc.κG * (n : ℝ) ^ 3) :=
  Mc.relativelySecurePerProver_of_editFloor S E _ prove use
    (fun σ x hx => Mc.eval_mulG n n n _ hshape (hgrid σ x hx).1 (hgrid σ x hx).2)
    ((Mc.time_mulG_le n n n hshape).trans_eq (by ring)) hcom hacc hfloor hδ' hF hFU hN

end GridMatMachine

/-- **A matrix machine with grid and wide-by-grid products** (RS-MODEL §4's width refinement for the peel routes): on
the shapes `shape`, `mulW` computes the product when one operand is on the grid (a 32-bit by small product, byte-split),
at `κW` per multiply-accumulate (4 u on sm_120). -/
structure WideGridMatMachine (R : Type) [Ring R] extends GridMatMachine R where
  κW : ℝ
  mulW : ∀ a b c : ℕ, Alg (Mat R a b × Mat R b c) (Mat R a c)
  eval_mulW : ∀ (a b c : ℕ) (x : Mat R a b × Mat R b c), shape a b c →
    ((∀ i k, G (x.1 i k)) ∨ ∀ k j, G (x.2 k j)) → eval (mulW a b c) x = x.1 * x.2
  time_mulW_le : ∀ a b c : ℕ, shape a b c → time (mulW a b c) ≤ κW * ((a : ℝ) * b * c)

namespace WideGridMatMachine

variable {R : Type} [Ring R] (Mc : WideGridMatMachine R)

/-- Multiply the outputs of `f` and `g` with the grid product. -/
def mulG₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) : Mc.Alg I (Mat R a c) :=
  Mc.lift2 f g (Mc.mulG a b c)

/-- Multiply the outputs of `f` and `g` with the wide-by-grid product. -/
def mulW₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) : Mc.Alg I (Mat R a c) :=
  Mc.lift2 f g (Mc.mulW a b c)

theorem time_mulG₂_le_of {I : Type} {a b c : ℕ} {f : Mc.Alg I (Mat R a b)} {g : Mc.Alg I (Mat R b c)} {u v : ℝ}
    (hs : Mc.shape a b c) (hf : Mc.time f ≤ u) (hg : Mc.time g ≤ v) :
    Mc.time (Mc.mulG₂ f g) ≤ u + v + Mc.κG * ((a : ℝ) * b * c) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_mulG_le a b c hs])

theorem time_mulW₂_le_of {I : Type} {a b c : ℕ} {f : Mc.Alg I (Mat R a b)} {g : Mc.Alg I (Mat R b c)} {u v : ℝ}
    (hs : Mc.shape a b c) (hf : Mc.time f ≤ u) (hg : Mc.time g ≤ v) :
    Mc.time (Mc.mulW₂ f g) ≤ u + v + Mc.κW * ((a : ℝ) * b * c) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_mulW_le a b c hs])

/-- A matrix on the grid. -/
def On {a b : ℕ} (X : Mat R a b) : Prop := ∀ i k, Mc.G (X i k)

/-- The full-rank blocks on the grid: `A`, `Y = B + S`, `R` and `S`. -/
def GridFull {n : ℕ} (b : MatMachine.FullBlocks R n) : Prop :=
  Mc.On b.2.1 ∧ Mc.On b.2.2.1 ∧ Mc.On b.2.2.2.1 ∧ Mc.On b.2.2.2.2

/-- **The full-rank peel on the grid**: `R Y` and `A S` are grid products. -/
def peelFullG (n : ℕ) : Mc.Alg (MatMachine.FullBlocks R n) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair (Mc.mulG₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
      (Mc.mulG₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))

theorem eval_peelFullG {n : ℕ} (hs : Mc.shape n n n) (b : MatMachine.FullBlocks R n) (hb : Mc.GridFull b) :
    Mc.eval (Mc.peelFullG n) b = (b.1, b.2.2.2.1 * b.2.2.1, b.2.1 * b.2.2.2.2) := by
  simp only [peelFullG, mulG₂, Machine.eval_lift2, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  rw [Mc.eval_mulG n n n (b.2.2.2.1, b.2.2.1) hs hb.2.2.1 hb.2.1,
    Mc.eval_mulG n n n (b.2.1, b.2.2.2.2) hs hb.1 hb.2.2.2]

theorem peelFullG_correct {n : ℕ} (hs : Mc.shape n n n) (x : Mat R n n × Mat R n n) (b : MatMachine.FullBlocks R n)
    (hb : MatMachine.ValidFull x b) (hg : Mc.GridFull b) : x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelFullG n) b) := by
  rw [Mc.eval_peelFullG hs b hg, ← Mc.eval_peelFull n b]
  exact Mc.peelFull_correct x b hb

/-- The full-rank peel on the grid costs two grid products: `2 κG n³`. -/
theorem time_peelFullG_le {n : ℕ} (hs : Mc.shape n n n) : Mc.time (Mc.peelFullG n) ≤ 2 * Mc.κG * (n : ℝ) ^ 3 := by
  apply le_trans
  · unfold peelFullG
    apply_rules [time_mulG₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le,
      Machine.time_snd_le]
  · linarith

/-- The low-rank blocks on the grid: `A`, `Y = B + W Zᵀ` and the factors `U`, `Vᵀ`, `W`, `Zᵀ`. -/
def GridLow {n r : ℕ} (b : MatMachine.LowBlocks R n r) : Prop :=
  Mc.On b.2.1 ∧ Mc.On b.2.2.1 ∧ Mc.On b.2.2.2.1 ∧ Mc.On b.2.2.2.2.1 ∧ Mc.On b.2.2.2.2.2.1 ∧ Mc.On b.2.2.2.2.2.2

/-- The shapes the low-rank peel's products take. -/
def LowShapes (n r : ℕ) : Prop := Mc.shape r n n ∧ Mc.shape n r n ∧ Mc.shape n n r

/-- **The low-rank peel on the grid** (RS-MODEL §5): `Vᵀ Y` and `A W` are grid products, and `U (Vᵀ Y)` and
`(A W) Zᵀ`, whose `r`-wide intermediates are off the grid, are wide-by-grid products. -/
def peelLowG (n r : ℕ) : Mc.Alg (MatMachine.LowBlocks R n r) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair
      (Mc.mulW₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
        (Mc.mulG₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
          (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
      (Mc.mulW₂
        (Mc.mulG₂ (Mc.seq Mc.snd Mc.fst)
          (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))))
        (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))

theorem eval_peelLowG {n r : ℕ} (hs : Mc.LowShapes n r) (b : MatMachine.LowBlocks R n r) (hb : Mc.GridLow b) :
    Mc.eval (Mc.peelLowG n r) b =
      (b.1, b.2.2.2.1 * (b.2.2.2.2.1 * b.2.2.1), (b.2.1 * b.2.2.2.2.2.1) * b.2.2.2.2.2.2) := by
  obtain ⟨hA, hY, hU, hV, hW, hZ⟩ := hb
  simp only [peelLowG, mulG₂, mulW₂, Machine.eval_lift2, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  rw [Mc.eval_mulG r n n (b.2.2.2.2.1, b.2.2.1) hs.1 hV hY,
    Mc.eval_mulG n n r (b.2.1, b.2.2.2.2.2.1) hs.2.2 hA hW,
    Mc.eval_mulW n r n (b.2.2.2.1, b.2.2.2.2.1 * b.2.2.1) hs.2.1 (Or.inl hU),
    Mc.eval_mulW n r n (b.2.1 * b.2.2.2.2.2.1, b.2.2.2.2.2.2) hs.2.1 (Or.inr hZ)]

theorem peelLowG_correct {n r : ℕ} (hs : Mc.LowShapes n r) (x : Mat R n n × Mat R n n) (b : MatMachine.LowBlocks R n r)
    (hb : MatMachine.ValidLow x b) (hg : Mc.GridLow b) : x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelLowG n r) b) := by
  rw [Mc.eval_peelLowG hs b hg, ← Mc.eval_peelLow n r b]
  exact Mc.peelLow_correct x b hb

/-- The low-rank peel on the grid costs two grid and two wide-by-grid products through `r`:
`2 κG n² r + 2 κW n² r` (RS-MODEL §5's `10 n² r` u at `κG = 1`, `κW = 4`). -/
theorem time_peelLowG_le {n r : ℕ} (hs : Mc.LowShapes n r) :
    Mc.time (Mc.peelLowG n r) ≤ 2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r := by
  obtain ⟨h1, h2, h3⟩ := hs
  apply le_trans
  · unfold peelLowG
    apply_rules (maxDepth := 200) [time_mulG₂_le_of, time_mulW₂_le_of, Machine.time_pair_le_of,
      Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
  · linarith

variable {n : ℕ} {Inst : Type} {P : Protocol Mc.toMachine Inst (Mat R n n)} {emb : Inst → Mat R n n × Mat R n n}

/-- **Full-rank masks, products typed by grid**: from extraction of on-grid valid blocks and the floor above
`q = 2κG n³ + 2κadd n²`, `ε = εext + (1 + εext) q / (T − q)`. -/
theorem secure_fullRank_grid (hmat : IsMatmulOn P emb) (hs : Mc.shape n n n) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x b => MatMachine.ValidFull (emb x) b ∧ Mc.GridFull b) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η)
    (hT : 2 * Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((2 * Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (2 * Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelFullG n) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelFullG_correct hs (emb x) b hb.1 hb.2)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelFullG_le hs, Mc.time_corr_le n]) hT

/-- **Rank-`r` masks, products typed by grid**: from extraction of on-grid valid blocks and the floor above
`q = (2κG + 2κW) n² r + 2κadd n²`, `ε = εext + (1 + εext) q / (T − q)`. -/
theorem secure_lowRank_grid (hmat : IsMatmulOn P emb) (r : ℕ) (hs : Mc.LowShapes n r) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x b => MatMachine.ValidLow (r := r) (emb x) b ∧ Mc.GridLow b) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η)
    (hT : 2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r +
      2 * Mc.κadd * (n : ℝ) ^ 2) / (T - (2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r +
        2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelLowG n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLowG_correct hs (emb x) b hb.1 hb.2)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelLowG_le hs, Mc.time_corr_le n]) hT

/-- **Rank-`r` masks by route 2, products typed by grid** (RS-MODEL §5): with every accepted adversary costing at
least `F`, `ε = εext + ((2κG + 2κW) n² r + 2κadd n²) / F`, `(10 r + 2)/(c n)` on sm_120 for `F = c n³`. -/
theorem secure_lowRank_advTime_grid (hmat : IsMatmulOn P emb) (r : ℕ) (hs : Mc.LowShapes n r) {εext δ η F : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x b => MatMachine.ValidLow (r := r) (emb x) b ∧ Mc.GridLow b) εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M) (hF : 0 < F) :
    Secure P (εext + (2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) / F)
      δ η 0 :=
  secure_of_extraction_advTime P (Mc.peelLowG n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLowG_correct hs (emb x) b hb.1 hb.2)) hext hM hF
    (by linarith [Mc.time_peelLowG_le hs, Mc.time_corr_le n])

end WideGridMatMachine

end EditedInstances

section TheoremK

/-! ### Theorem K: {0,1} data, integer operands, any arithmetic (P2-FF round 5 §3.6, round 6 §2)

Theorem K is the full-floor draft r5's §3.6, and the re-encoded design it is applied to is
r6's §2. Words are functions of bits `z : V → R`, over a ring with a map
`φ : R → F₂`: `Z/2³²` (s32 wrap, `toF2`) or `ℤ` (exact). A run is a straight-line program over `R` (`Slp`): products
of bit-affine operands (integer operands) and linear steps (merges, with ring operations only, so reads are
unscaled). Each word is a combination of the products plus a bit-affine word (`Slp.val_eq`). The bit-pair second
difference mod 2 (`projF2`) is a symmetric form with zero diagonal. Each product contributes `a bᵀ + b aᵀ`, of rank
at most 2, whose diagonal `2 a_u b_u` vanishes mod 2. So a word that agrees with a target needs at least half the
target's `F₂` rank in products (`theoremK`, `Slp.rank_projF2_le`). On the re-encoded design, a cell-block's target
has `F₂` rank `2L` (`rank_projF2_dIncr`). Each certified cell-block therefore needs `L` products
(`Slp.dIncr_products_ge`), and a run whose cell-blocks have disjoint pieces needs `L` per certified cell-block
(`Slp.card_mul_le`). The projection reads every bit pair, and must (`winograd_cross_only`). Disjoint pieces forbid
sharing along a cell's prefix chain; the chain lemma, after the quad runs, lifts that to sharing among a row's cells,
and gives the window's cost line (`RoundsHypK`, `roundsHyp_of_K`). -/

theorem BitAff.eval_bitInd {V R : Type} [Fintype V] [DecidableEq V] [CommRing R] (A : BitAff V R) (S : Finset V) :
    A.eval (bitInd S) = A.c + ∑ v ∈ S, A.α v := by
  simp [BitAff.eval, bitInd, mul_ite, Finset.sum_ite_mem]

/-- **The projection mod 2** onto bit pairs (Theorem K's form over `F₂`), zero on the diagonal. -/
def projF2 {V R : Type} [DecidableEq V] [CommRing R] (φ : R →+* ZMod 2) (f : (V → R) → R) :
    Matrix V V (ZMod 2) :=
  fun u v => if u = v then 0 else φ (pairDiff f u v)

section
variable {V R : Type} [Fintype V] [DecidableEq V] [CommRing R]

omit [Fintype V] in
theorem pairDiff_add (f g : (V → R) → R) (u v : V) :
    pairDiff (fun z => f z + g z) u v = pairDiff f u v + pairDiff g u v := by
  simp only [pairDiff]
  ring

omit [Fintype V] in
theorem pairDiff_sum {ι : Type} (S : Finset ι) (f : ι → (V → R) → R) (u v : V) :
    pairDiff (fun z => ∑ j ∈ S, f j z) u v = ∑ j ∈ S, pairDiff (f j) u v := by
  simp only [pairDiff, Finset.sum_sub_distrib, Finset.sum_add_distrib]

omit [Fintype V] in
theorem pairDiff_mul_left (k : R) (f : (V → R) → R) (u v : V) :
    pairDiff (fun z => k * f z) u v = k * pairDiff f u v := by
  simp only [pairDiff]
  ring

/-- A product of two bit-affine operands: `α_u β_v + α_v β_u`; its constants cancel. -/
theorem pairDiff_mul (A B : BitAff V R) {u v : V} (huv : u ≠ v) :
    pairDiff (fun z => A.eval z * B.eval z) u v = A.α u * B.α v + A.α v * B.α u := by
  simp only [pairDiff, BitAff.eval_bitInd, Finset.sum_pair huv, Finset.sum_singleton, Finset.sum_empty]
  ring

/-- Bit-affine words project to 0. -/
theorem pairDiff_aff (A : BitAff V R) {u v : V} (huv : u ≠ v) : pairDiff (fun z => A.eval z) u v = 0 := by
  simp only [pairDiff, BitAff.eval_bitInd, Finset.sum_pair huv, Finset.sum_singleton, Finset.sum_empty]
  ring

theorem two_eq_zero_F2 (x : ZMod 2) : x + x = 0 := by
  fin_cases x <;> decide

/-- A word computed from products of bit-affine operands, projected mod 2: each product contributes the
alternating form `a bᵀ + b aᵀ`, whose diagonal `2 a_u b_u` vanishes mod 2. -/
theorem projF2_word (φ : R →+* ZMod 2) {ι : Type} (S : Finset ι) (lam : ι → R) (A B : ι → BitAff V R)
    (r : BitAff V R) :
    projF2 φ (fun z => ∑ j ∈ S, lam j * ((A j).eval z * (B j).eval z) + r.eval z) =
      ∑ j ∈ S, (Matrix.vecMulVec (fun u => φ (lam j * (A j).α u)) (fun v => φ ((B j).α v)) +
        Matrix.vecMulVec (fun u => φ ((B j).α u)) (fun v => φ (lam j * (A j).α v))) := by
  ext u v
  simp only [projF2, Matrix.sum_apply, Matrix.add_apply, Matrix.vecMulVec_apply]
  by_cases huv : u = v
  · subst huv
    simp only [↓reduceIte]
    refine (Finset.sum_eq_zero fun j _ => ?_).symm
    rw [mul_comm (φ ((B j).α u))]
    exact two_eq_zero_F2 _
  · simp only [huv, ↓reduceIte]
    rw [pairDiff_add (fun z => ∑ j ∈ S, lam j * ((A j).eval z * (B j).eval z)) (fun z => r.eval z),
      pairDiff_aff r huv, add_zero,
      pairDiff_sum S (fun j z => lam j * ((A j).eval z * (B j).eval z)), map_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [pairDiff_mul_left (lam j) (fun z => (A j).eval z * (B j).eval z), pairDiff_mul (A j) (B j) huv]
    simp only [map_mul, map_add]
    ring

/-- A sum of pairs of rank-one terms, indexed by any finite type, has rank at most twice its size. -/
theorem rank_sum_vecMulVec_le_card {X Y ι K : Type} [Fintype X] [Fintype Y] [Fintype ι] [CommRing K]
    [StrongRankCondition K] (a c : ι → X → K) (b d : ι → Y → K) :
    (∑ j, (Matrix.vecMulVec (a j) (b j) + Matrix.vecMulVec (c j) (d j))).rank ≤ 2 * Fintype.card ι := by
  let U : Matrix X (ι ⊕ ι) K := fun x s => Sum.elim (fun j => a j x) (fun j => c j x) s
  let W : Matrix (ι ⊕ ι) Y K := fun s y => Sum.elim (fun j => b j y) (fun j => d j y) s
  have hUW : ∑ j, (Matrix.vecMulVec (a j) (b j) + Matrix.vecMulVec (c j) (d j)) = U * W := by
    ext x y
    rw [Matrix.mul_apply, Fintype.sum_sum_type]
    simp only [Matrix.sum_apply, Matrix.add_apply, Matrix.vecMulVec_apply, Finset.sum_add_distrib, U, W,
      Sum.elim_inl, Sum.elim_inr]
  rw [hUW]
  refine (Matrix.rank_mul_le_left U W).trans ((Matrix.rank_le_card_width U).trans ?_)
  simp [two_mul]

/-- **Theorem K** (P2-FF round 5 §3.6), abstract: over any ring `R` with a map to `F₂` (`Z/2³²` in s32 wrap, `ℤ`
exactly), a word that agrees with a target `T` at every bit point with at most two ones, and is a combination of
the products of the bit-affine operands in `S` plus a bit-affine word, has `rank_F₂ P₂(T) ≤ 2 |S|`. The
coefficients are any ring elements: integer operands and unscaled reads are what makes the word such a combination. -/
theorem theoremK (φ : R →+* ZMod 2) {ι : Type} (S : Finset ι) (lam : ι → R) (A B : ι → BitAff V R)
    (r : BitAff V R) (T : (V → R) → R)
    (hT : ∀ U : Finset V, U.card ≤ 2 →
      T (bitInd U) = ∑ j ∈ S, lam j * ((A j).eval (bitInd U) * (B j).eval (bitInd U)) + r.eval (bitInd U)) :
    (projF2 φ T).rank ≤ 2 * S.card := by
  have h : projF2 φ T = projF2 φ (fun z => ∑ j ∈ S, lam j * ((A j).eval z * (B j).eval z) + r.eval z) := by
    ext u v
    simp only [projF2, pairDiff]
    rw [hT {u, v} ((Finset.card_insert_le u {v}).trans (by simp)), hT {u} (by simp), hT {v} (by simp),
      hT ∅ (by simp)]
  rw [h, projF2_word, ← Finset.sum_coe_sort S]
  exact (rank_sum_vecMulVec_le_card _ _ _ _).trans (by simp)

end

/-- **Why `projF2` reads every bit pair** (RS-E-REF r4 IV.1): Winograd's product without its corrections,
`(x₀ + y₁)(x₁ + y₀)`, one product for `L = 2`, has the target `x₀ y₀ + x₁ y₁`'s second difference on every cross
pair `(x_a, y_b)` and differs from it on the same-side pair `(x₀, x₁)`. A projection onto cross pairs alone has rank
`L` against the product's 2 and would let it pass; `projF2`, on every pair, has rank `2L` and does not. -/
theorem winograd_cross_only :
    (∀ a b : Fin 2,
      pairDiff (fun z : Fin 2 ⊕ Fin 2 → ℤ => (z (.inl 0) + z (.inr 1)) * (z (.inl 1) + z (.inr 0))) (.inl a) (.inr b) =
        pairDiff (fun z : Fin 2 ⊕ Fin 2 → ℤ => z (.inl 0) * z (.inr 0) + z (.inl 1) * z (.inr 1)) (.inl a) (.inr b)) ∧
    pairDiff (fun z : Fin 2 ⊕ Fin 2 → ℤ => (z (.inl 0) + z (.inr 1)) * (z (.inl 1) + z (.inr 0))) (.inl 0) (.inl 1) ≠
      pairDiff (fun z : Fin 2 ⊕ Fin 2 → ℤ => z (.inl 0) * z (.inr 0) + z (.inl 1) * z (.inr 1)) (.inl 0) (.inl 1) := by
  refine ⟨fun a b => ?_, ?_⟩
  · fin_cases a <;> fin_cases b <;> simp [pairDiff, bitInd]
  · simp [pairDiff, bitInd]

namespace Slp

variable {V R : Type} [Fintype V] [CommRing R] {N : ℕ}

/-- The slots' values at the bits `z`. -/
def val (P : Slp V R N) (z : V → R) (j : Fin N) : R :=
  if P.isProd j then (P.opA j).eval z * (P.opB j).eval z
  else ∑ k : Fin j.val, P.lin j k * P.val z ⟨k.val, k.isLt.trans j.isLt⟩ + (P.off j).eval z
termination_by j.val
decreasing_by exact k.isLt

/-- Each slot's bit-affine part (the closed form). -/
def aff (P : Slp V R N) (j : Fin N) : BitAff V R :=
  if P.isProd j then ⟨0, 0⟩
  else ⟨∑ k : Fin j.val, P.lin j k * (P.aff ⟨k.val, k.isLt.trans j.isLt⟩).c + (P.off j).c,
    fun v => ∑ k : Fin j.val, P.lin j k * (P.aff ⟨k.val, k.isLt.trans j.isLt⟩).α v + (P.off j).α v⟩
termination_by j.val
decreasing_by all_goals exact k.isLt

theorem aff_eval (P : Slp V R N) (z : V → R) (j : Fin N) :
    (P.aff j).eval z = if P.isProd j then 0
      else ∑ k : Fin j.val, P.lin j k * (P.aff ⟨k.val, k.isLt.trans j.isLt⟩).eval z + (P.off j).eval z := by
  rw [aff]
  split_ifs
  · simp [BitAff.eval]
  · simp only [BitAff.eval, add_mul, Finset.sum_add_distrib, mul_add, Finset.mul_sum, Finset.sum_mul, mul_assoc]
    rw [Finset.sum_comm (s := (Finset.univ : Finset V)) (t := (Finset.univ : Finset (Fin j.val)))]
    ring

/-- **The closed form**: every slot is a combination of the program's products plus a bit-affine word. -/
theorem val_eq (P : Slp V R N) (z : V → R) (j : Fin N) :
    P.val z j = ∑ i, P.coef j i * ((P.opA i).eval z * (P.opB i).eval z) + (P.aff j).eval z := by
  obtain ⟨n, hn⟩ := j
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rw [val, coef, aff_eval]
    by_cases hp : P.isProd ⟨n, hn⟩ = true
    · simp only [hp, ↓reduceIte, Pi.single_apply, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq',
        Finset.mem_univ, add_zero]
    · simp only [hp, Bool.false_eq_true, ↓reduceIte]
      have hk : ∀ k : Fin n, P.val z ⟨k.val, k.isLt.trans hn⟩ =
          ∑ i, P.coef ⟨k.val, k.isLt.trans hn⟩ i * ((P.opA i).eval z * (P.opB i).eval z) +
            (P.aff ⟨k.val, k.isLt.trans hn⟩).eval z := fun k => ih k.val k.isLt _
      simp only [hk, mul_add, Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_apply, Pi.smul_apply,
        smul_eq_mul, Finset.sum_mul, mul_assoc]
      rw [Finset.sum_comm (s := (Finset.univ : Finset (Fin n))) (t := (Finset.univ : Finset (Fin N)))]
      ring

omit [Fintype V] in
/-- Only products carry coefficients. -/
theorem isProd_of_coef_ne_zero (P : Slp V R N) (j i : Fin N) (h : P.coef j i ≠ 0) : P.isProd i = true := by
  obtain ⟨n, hn⟩ := j
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rw [coef] at h
    by_cases hp : P.isProd ⟨n, hn⟩ = true
    · simp only [hp, ↓reduceIte, Pi.single_apply] at h
      by_cases hi : i = ⟨n, hn⟩
      · exact hi ▸ hp
      · simp [hi] at h
    · simp only [hp, Bool.false_eq_true, ↓reduceIte, Finset.sum_apply, Pi.smul_apply, smul_eq_mul] at h
      obtain ⟨k, -, hk⟩ := Finset.exists_ne_zero_of_sum_ne_zero h
      exact ih k.val k.isLt _ (right_ne_zero_of_mul hk)

open Classical in
/-- **A cell-block's pieces**: the products whose coefficient changes from the word `c'` (the cell's previous
committed word, `none` for its first) to `c`. -/
noncomputable def pieces (P : Slp V R N) (c : Fin N) (c' : Option (Fin N)) : Finset (Fin N) :=
  univ.filter fun i => P.coef c i - c'.elim 0 (fun k => P.coef k i) ≠ 0

omit [Fintype V] in
theorem pieces_isProd (P : Slp V R N) (c : Fin N) (c' : Option (Fin N)) {i : Fin N} (hi : i ∈ P.pieces c c') :
    P.isProd i = true := by
  classical
  simp only [pieces, Finset.mem_filter, Finset.mem_univ, true_and] at hi
  by_cases h : P.coef c i = 0
  · cases c' with
    | none => simp [h] at hi
    | some k =>
      simp only [h, Option.elim_some, zero_sub, neg_ne_zero] at hi
      exact P.isProd_of_coef_ne_zero k i hi
  · exact P.isProd_of_coef_ne_zero c i h

/-- The cell-block's increment `val c − val c'` is a combination of its pieces plus a bit-affine word. -/
theorem incr_eq (P : Slp V R N) (c : Fin N) (c' : Option (Fin N)) :
    ∃ r : BitAff V R, ∀ z, P.val z c - c'.elim 0 (fun k => P.val z k) =
      ∑ i ∈ P.pieces c c', (P.coef c i - c'.elim 0 (fun k => P.coef k i)) *
        ((P.opA i).eval z * (P.opB i).eval z) + r.eval z := by
  classical
  refine ⟨⟨(P.aff c).c - c'.elim 0 (fun k => (P.aff k).c), fun v => (P.aff c).α v - c'.elim 0 (fun k => (P.aff k).α v)⟩,
    fun z => ?_⟩
  rw [pieces, Finset.sum_filter_of_ne fun i _ h => left_ne_zero_of_mul h]
  cases c' with
  | none =>
    simp only [Option.elim_none, sub_zero, val_eq]
  | some k =>
    simp only [Option.elim_some, val_eq, BitAff.eval, sub_mul, Finset.sum_sub_distrib]
    ring

/-- **Theorem K for a program's cell-block**: if the increment agrees with a target at every bit point with at most
two ones, the target's `F₂` rank is at most twice the cell-block's pieces. -/
theorem rank_projF2_le (φ : R →+* ZMod 2) [DecidableEq V] (P : Slp V R N) (c : Fin N) (c' : Option (Fin N))
    (T : (V → R) → R)
    (hT : ∀ U : Finset V, U.card ≤ 2 → P.val (bitInd U) c - c'.elim 0 (fun k => P.val (bitInd U) k) = T (bitInd U)) :
    (projF2 φ T).rank ≤ 2 * (P.pieces c c').card := by
  obtain ⟨r, hr⟩ := P.incr_eq c c'
  exact theoremK φ _ _ _ _ r T fun U hU => (hT U hU).symm.trans (hr _)

omit [Fintype V] in
/-- **Cell-pure runs add up**: if each certified cell-block needs `L` pieces and distinct ones share no product,
the program has at least `L` products per certified cell-block. -/
theorem card_mul_le {E : Type} (P : Slp V R N) (Cor : Finset E) (cur : E → Fin N) (prev : E → Option (Fin N))
    (L : ℕ) (hL : ∀ e ∈ Cor, L ≤ (P.pieces (cur e) (prev e)).card)
    (hpure : (Cor : Set E).PairwiseDisjoint fun e => P.pieces (cur e) (prev e)) :
    L * Cor.card ≤ (univ.filter fun j => P.isProd j = true).card := by
  classical
  calc L * Cor.card = ∑ _e ∈ Cor, L := by rw [Finset.sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ e ∈ Cor, (P.pieces (cur e) (prev e)).card := Finset.sum_le_sum hL
    _ = (Cor.biUnion fun e => P.pieces (cur e) (prev e)).card := (Finset.card_biUnion hpure).symm
    _ ≤ _ := Finset.card_le_card fun i hi => by
      obtain ⟨e, -, he⟩ := Finset.mem_biUnion.mp hi
      exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, P.pieces_isProd _ _ he⟩

end Slp

/-! ### The re-encoded design (P2-FF round 6 §2): `X = ⌊x/s⌋ + b` -/

section Design

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

theorem dPrefix_zero (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (h : 0 < B) (z : DBits n L np B n' → R) :
    dPrefix pc vq w i l 0 z = dIncr pc vq w i l ⟨0, h⟩ z := by
  rw [dPrefix, Finset.sum_eq_single ⟨0, h⟩]
  · simp
  · intro b _ hb
    have : b.val ≠ 0 := fun h' => hb (Fin.ext h')
    simp only [Nat.le_zero, this, ↓reduceIte]
  · simp

/-- The committed words' increments are the cell-blocks' targets. -/
theorem dPrefix_succ (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') {t : ℕ} (h : t + 1 < B) (z : DBits n L np B n' → R) :
    dPrefix pc vq w i l (t + 1) z - dPrefix pc vq w i l t z = dIncr pc vq w i l ⟨t + 1, h⟩ z := by
  rw [dPrefix, dPrefix, ← Finset.sum_sub_distrib, Finset.sum_eq_single ⟨t + 1, h⟩]
  · simp
  · intro b _ hb
    have : b.val ≠ t + 1 := fun h' => hb (Fin.ext h')
    by_cases h1 : b.val ≤ t
    · simp only [h1, Nat.le_succ_of_le h1, ↓reduceIte, sub_self]
    · have h2 : ¬b.val ≤ t + 1 := by omega
      simp only [h1, h2, ↓reduceIte, sub_self]
  · simp

omit [CommRing R] in
theorem dP_α_inl [CommRing R] (pc : Fin n → Fin B → Fin L → R) (w : Fin np → R) (i : Fin n) (t : Fin B) (j : Fin L)
    (a : Fin n × Fin L × Fin np) :
    (dP (n' := n') pc w i t j).α (.inl a) = if a.1 = i ∧ a.2.1 = j then w a.2.2 else 0 := rfl

omit [CommRing R] in
theorem dP_α_inr [CommRing R] (pc : Fin n → Fin B → Fin L → R) (w : Fin np → R) (i : Fin n) (t : Fin B) (j : Fin L)
    (a : Fin B × Fin L × Fin n') : (dP pc w i t j).α (.inr a) = 0 := rfl

omit [CommRing R] in
theorem dX_α_inl [CommRing R] (vq : Fin B → Fin L → Fin n' → R) (t : Fin B) (j : Fin L) (l : Fin n')
    (a : Fin n × Fin L × Fin np) : (dX vq t j l).α (.inl a) = 0 := rfl

omit [CommRing R] in
theorem dX_α_inr [CommRing R] (vq : Fin B → Fin L → Fin n' → R) (t : Fin B) (j : Fin L) (l : Fin n')
    (a : Fin B × Fin L × Fin n') :
    (dX (n := n) (np := np) vq t j l).α (.inr a) = if a = (t, j, l) then 1 else 0 := rfl

/-- **The design's rank** (P2-FF round 6 §2): with a plane of odd weight (`w_0 = 1`), a cell-block's target has
`F₂` rank `2L`: its projection has the identity on the rows `β_p₀(i, j)`, `b(t, j, l)` against the columns
`b(t, j, l)`, `β_p₀(i, j)`. -/
theorem rank_projF2_dIncr (φ : R →+* ZMod 2) (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) {p₀ : Fin np} (hw : φ (w p₀) = 1) (i : Fin n) (l : Fin n') (t : Fin B) :
    2 * L ≤ (projF2 φ (dIncr pc vq w i l t)).rank := by
  classical
  have h : dIncr pc vq w i l t = fun z => ∑ j ∈ univ, (1 : R) * ((dP pc w i t j).eval z *
      (dX (n := n) (np := np) vq t j l).eval z) + (⟨0, 0⟩ : BitAff (DBits n L np B n') R).eval z := by
    funext z
    simp [dIncr, BitAff.eval]
  rw [h, projF2_word]
  set M := ∑ j ∈ univ, (Matrix.vecMulVec (fun u => φ (1 * (dP pc w i t j).α u))
      (fun v => φ ((dX (n := n) (np := np) vq t j l).α v)) +
      Matrix.vecMulVec (fun u => φ ((dX (n := n) (np := np) vq t j l).α u))
        (fun v => φ (1 * (dP pc w i t j).α v))) with hM
  have hent : ∀ u v, M u v = ∑ j, (φ ((dP pc w i t j).α u) * φ ((dX (n := n) (np := np) vq t j l).α v) +
      φ ((dX (n := n) (np := np) vq t j l).α u) * φ ((dP pc w i t j).α v)) := by
    intro u v
    simp only [hM, Matrix.sum_apply, Matrix.add_apply, Matrix.vecMulVec_apply, one_mul]
  let f : Fin L ⊕ Fin L → DBits n L np B n' := Sum.elim (fun j => .inl (i, j, p₀)) fun j => .inr (t, j, l)
  let g : Fin L ⊕ Fin L → DBits n L np B n' := Sum.elim (fun j => .inr (t, j, l)) fun j => .inl (i, j, p₀)
  have hsub : M.submatrix f g = 1 := by
    ext a a'
    rw [Matrix.submatrix_apply, hent, Matrix.one_apply]
    rcases a with j | j <;> rcases a' with j' | j'
    · simp only [f, g, Sum.elim_inl, dP_α_inl, dP_α_inr, dX_α_inl, dX_α_inr, map_zero, mul_zero, add_zero,
        true_and, Prod.mk.injEq, and_true, Sum.inl.injEq]
      rw [Finset.sum_eq_single j (fun b _ hb => by simp [Ne.symm hb]) (by simp)]
      by_cases hj : j = j' <;> simp [hj, hw, eq_comm]
    · simp only [f, g, Sum.elim_inl, Sum.elim_inr, dP_α_inl, dX_α_inl, map_zero, mul_zero, zero_mul, add_zero,
        Finset.sum_const_zero, reduceCtorEq, ↓reduceIte]
    · simp only [f, g, Sum.elim_inl, Sum.elim_inr, dP_α_inr, dX_α_inr, map_zero, mul_zero, zero_mul, add_zero,
        Finset.sum_const_zero, reduceCtorEq, ↓reduceIte]
    · simp only [f, g, Sum.elim_inr, dP_α_inl, dP_α_inr, dX_α_inr, map_zero, zero_mul, zero_add,
        true_and, Prod.mk.injEq, and_true, Sum.inr.injEq]
      rw [Finset.sum_eq_single j (fun b _ hb => by simp [Ne.symm hb]) (by simp)]
      by_cases hj : j = j' <;> simp [hj, hw, eq_comm]
  calc 2 * L = (1 : Matrix (Fin L ⊕ Fin L) (Fin L ⊕ Fin L) (ZMod 2)).rank := by
        rw [Matrix.rank_one, Fintype.card_sum, Fintype.card_fin, two_mul]
    _ ≤ M.rank := by
        rw [← hsub]
        exact Matrix.rank_submatrix_le M f g

/-- **Theorem K on the re-encoded design**: a straight-line program over `R` (`Z/2³²`: s32 wrap) whose increment
from the cell's previous committed word agrees with cell-block `(i, l, t)`'s target at every bit point with at most
two ones has at least `L` pieces, whatever the committed `P`, `⌊x/s⌋`, the partners and the merges. -/
theorem Slp.dIncr_products_ge (φ : R →+* ZMod 2) (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) {p₀ : Fin np} (hw : φ (w p₀) = 1) {N : ℕ}
    (P : Slp (DBits n L np B n') R N) (c : Fin N) (c' : Option (Fin N)) (i : Fin n) (l : Fin n') (t : Fin B)
    (hT : ∀ U : Finset (DBits n L np B n'), U.card ≤ 2 →
      P.val (bitInd U) c - c'.elim 0 (fun k => P.val (bitInd U) k) = dIncr pc vq w i l t (bitInd U)) :
    L ≤ (P.pieces c c').card := by
  have := (rank_projF2_dIncr φ pc vq w hw i l t).trans (P.rank_projF2_le φ c c' _ hT)
  omega

/-- A cell's first committed word, correct at a bit point, certifies its first cell-block there. -/
theorem Slp.incr_of_prefix_zero (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) {N : ℕ} (P : Slp (DBits n L np B n') R N) (c : Fin N) (i : Fin n) (l : Fin n') (h : 0 < B)
    (z : DBits n L np B n' → R) (hc : P.val z c = dPrefix pc vq w i l 0 z) :
    P.val z c - (none : Option (Fin N)).elim 0 (fun k => P.val z k) = dIncr pc vq w i l ⟨0, h⟩ z := by
  rw [Option.elim_none, sub_zero, hc, dPrefix_zero]

/-- Two consecutive committed words of a cell, correct at a bit point, certify the cell-block between them there. -/
theorem Slp.incr_of_prefix_succ (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) {N : ℕ} (P : Slp (DBits n L np B n') R N) (c c' : Fin N) (i : Fin n) (l : Fin n') {t : ℕ}
    (h : t + 1 < B) (z : DBits n L np B n' → R) (hc : P.val z c = dPrefix pc vq w i l (t + 1) z)
    (hc' : P.val z c' = dPrefix pc vq w i l t z) :
    P.val z c - (some c').elim 0 (fun k => P.val z k) = dIncr pc vq w i l ⟨t + 1, h⟩ z := by
  rw [Option.elim_some, hc, hc', dPrefix_succ]

end Design

theorem toF2_wRe_zero : toF2 (wRe 0) = 1 := by
  simp [toF2, wRe]

end TheoremK

/-! ### Theorem H: merges priced (P2-FF, full-floor r6 §6)

Round 5 §3's quad model: a quad runs `s` slices on four lanes, at `4 s` slots, and each lane is a full cell, a piece of
a merged cell, or junk; merging a cell's pieces costs `h` slots. Theorem J (two full lanes force `s ≥ K`) and the rank
bounds (one full lane forces `s ≥ K / 2`; a merged cell's pieces carry `K / 2` slices) are hypotheses here. With
`h ≥ 2K/3` every useful cell costs `K` slots.

Theorem H is not needed for the re-encoded design (RS-E-REF r4 IV.3): the chain lemma covers it in every arithmetic,
merges free, since a merge is a linear step and the chain counts products (`chainBound_oneSided` over any field the
ring maps to; over `ℚ`, `certifiedFraction120θ_reEncodedExactK_40`). Theorem H stays proved, as the alternative for
the quad model with merges priced. RS-E-REF r4 confirms its proof; the `2K/3` threshold is tight only for its lemmas,
and with real merges it is `⌊K/2⌋`. -/

section TheoremH

/-- A lane of a quad: a full cell, a piece of the merged cell `m`, or junk. -/
inductive QLane (M : Type) where
  | full
  | piece (m : M)
  | junk
  deriving DecidableEq

/-- The lane is a piece of some merged cell. -/
def QLane.IsPiece {M : Type} : QLane M → Prop
  | .piece _ => True
  | _ => False

instance instDecidableIsPiece {M : Type} (x : QLane M) : Decidable x.IsPiece := by
  cases x <;> simp only [QLane.IsPiece] <;> infer_instance

theorem QLane.ite_isPiece {M : Type} [Fintype M] [DecidableEq M] (x : QLane M) (a : ℝ) :
    (if x.IsPiece then a else 0) = ∑ m, if x = .piece m then a else 0 := by
  cases x with
  | full => simp [QLane.IsPiece]
  | piece m₀ => simp [QLane.IsPiece]
  | junk => simp [QLane.IsPiece]

/-- **Theorem H** (P2-FF round 6 §6, merges priced), the allocation. Quads `q` run `s q` slices, at `s q` slots a
lane (`4 s q` a quad); each merged cell costs `h` slots. The premises are named:
- Theorem J (round 5 §3): a quad with two full lanes has `s ≥ K`;
- the rank bound: a quad with one full lane has `s ≥ K / 2`, and a merged cell's pieces carry `≥ K / 2` slices.

With `h ≥ 2K/3`, the cost is at least `K` slots a useful cell (full lanes and merged cells). -/
theorem theoremH {Q M : Type} [Fintype Q] [Fintype M] [DecidableEq M] (lane : Q → Fin 4 → QLane M) (s : Q → ℝ)
    (K h : ℝ) (hs : ∀ q, 0 ≤ s q) (hK : 0 ≤ K) (hh : 2 * K / 3 ≤ h)
    (hJ : ∀ q, 2 ≤ (univ.filter fun ℓ => lane q ℓ = .full).card → K ≤ s q)
    (hR1 : ∀ q, (univ.filter fun ℓ => lane q ℓ = .full).card = 1 → K / 2 ≤ s q)
    (hRm : ∀ m, K / 2 ≤ ∑ q, ∑ ℓ, if lane q ℓ = .piece m then s q else 0) :
    K * (∑ q, ((univ.filter fun ℓ => lane q ℓ = .full).card : ℝ) + Fintype.card M) ≤
      ∑ q, 4 * s q + h * Fintype.card M := by
  set f : Q → ℕ := fun q => (univ.filter fun ℓ => lane q ℓ = .full).card with hf
  -- the transfer: `K / 6` from a merged cell to each one-full-lane quad holding one of its pieces
  set c : Q → ℝ := fun q => if f q = 1 then K / 6 else 0 with hc
  set A : Q → ℝ := fun q => ∑ ℓ, ((if (lane q ℓ).IsPiece then 0 else s q) + if (lane q ℓ).IsPiece then c q else 0)
  set Bm : M → ℝ := fun m => h + ∑ q, ∑ ℓ, if lane q ℓ = .piece m then s q - c q else 0
  have hc_le : ∀ q, c q ≤ s q := fun q => by
    simp only [hc]
    split_ifs with h1
    · linarith [hR1 q h1]
    · exact hs q
  have hsplit : ∑ q, 4 * s q + h * Fintype.card M = ∑ q, A q + ∑ m, Bm m := by
    have e1 : ∀ q, 4 * s q = A q + ∑ ℓ, if (lane q ℓ).IsPiece then s q - c q else 0 := fun q => by
      simp only [A, ← Finset.sum_add_distrib]
      rw [show (4 : ℝ) * s q = ∑ _ℓ : Fin 4, s q by simp]
      refine Finset.sum_congr rfl fun ℓ _ => ?_
      split_ifs <;> ring
    have e2 : ∑ q, ∑ ℓ, (if (lane q ℓ).IsPiece then s q - c q else 0) =
        ∑ m, ∑ q, ∑ ℓ, if lane q ℓ = .piece m then s q - c q else 0 := by
      calc _ = ∑ q, ∑ ℓ, ∑ m, (if lane q ℓ = .piece m then s q - c q else 0) := by
            simp only [QLane.ite_isPiece]
        _ = ∑ q, ∑ m, ∑ ℓ, (if lane q ℓ = .piece m then s q - c q else 0) :=
            Finset.sum_congr rfl fun q _ => Finset.sum_comm
        _ = _ := Finset.sum_comm
    simp only [e1, Finset.sum_add_distrib, e2, Bm, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    ring
  have hA : ∀ q, K * f q ≤ A q := by
    intro q
    have hnp : ((univ.filter fun ℓ => ¬(lane q ℓ).IsPiece).card : ℝ) +
        (univ.filter fun ℓ => (lane q ℓ).IsPiece).card = 4 := by
      have := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Fin 4)))
        (fun ℓ => (lane q ℓ).IsPiece)
      simp only [Finset.card_univ, Fintype.card_fin] at this
      exact_mod_cast (by omega : (univ.filter fun ℓ => ¬(lane q ℓ).IsPiece).card +
        (univ.filter fun ℓ => (lane q ℓ).IsPiece).card = 4)
    have hfull : (f q : ℝ) ≤ (univ.filter fun ℓ => ¬(lane q ℓ).IsPiece).card := by
      exact_mod_cast Finset.card_le_card fun ℓ hℓ => by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hℓ ⊢
        simp [hℓ, QLane.IsPiece]
    have hAq : A q = (univ.filter fun ℓ => ¬(lane q ℓ).IsPiece).card * s q +
        (univ.filter fun ℓ => (lane q ℓ).IsPiece).card * c q := by
      simp only [A, Finset.sum_add_distrib, Finset.sum_ite, Finset.sum_const_zero, zero_add, add_zero,
        Finset.sum_const, nsmul_eq_mul]
    rw [hAq]
    have hpos1 : (0 : ℝ) ≤ (univ.filter fun ℓ => (lane q ℓ).IsPiece).card := Nat.cast_nonneg _
    rcases Nat.lt_or_ge (f q) 2 with h2 | h2
    · rcases Nat.lt_or_ge (f q) 1 with h1 | h1
      · have h0 : f q = 0 := by omega
        have hc0 : c q = 0 := by simp [hc, h0]
        rw [h0, hc0]
        simp only [Nat.cast_zero, mul_zero, add_zero]
        exact mul_nonneg (Nat.cast_nonneg _) (hs q)
      · have h1' : f q = 1 := by omega
        have hc1 : c q = K / 6 := by simp [hc, h1']
        have hsq := hR1 q h1'
        rw [hc1, h1']
        have hp3 : ((univ.filter fun ℓ => (lane q ℓ).IsPiece).card : ℝ) ≤ 3 := by
          rw [h1'] at hfull
          push_cast at hfull
          linarith
        nlinarith
    · have hc0 : c q = 0 := by simp [hc, show f q ≠ 1 by omega]
      rw [hc0, mul_zero, add_zero]
      have hsq := hJ q h2
      calc K * f q ≤ s q * f q := mul_le_mul_of_nonneg_right hsq (Nat.cast_nonneg _)
        _ ≤ s q * (univ.filter fun ℓ => ¬(lane q ℓ).IsPiece).card := mul_le_mul_of_nonneg_left hfull (hs q)
        _ = _ := mul_comm _ _
  have hB : ∀ m, K ≤ Bm m := by
    intro m
    have hterm : ∀ q ℓ, 0 ≤ if lane q ℓ = .piece m then s q - c q else 0 := fun q ℓ => by
      split_ifs
      · linarith [hc_le q]
      · exact le_rfl
    have hsingle : ∀ q ℓ, lane q ℓ = .piece m →
        s q - c q ≤ ∑ q, ∑ ℓ, if lane q ℓ = .piece m then s q - c q else 0 := fun q ℓ hql => by
      have h1 := Finset.single_le_sum (f := fun ℓ => if lane q ℓ = .piece m then s q - c q else 0)
        (fun ℓ _ => hterm q ℓ) (Finset.mem_univ ℓ)
      have h2 := Finset.single_le_sum (f := fun q => ∑ ℓ, if lane q ℓ = .piece m then s q - c q else 0)
        (fun q _ => Finset.sum_nonneg fun ℓ _ => hterm q ℓ) (Finset.mem_univ q)
      simp only [hql, ↓reduceIte] at h1
      exact h1.trans h2
    by_cases hA2 : ∃ q ℓ, lane q ℓ = .piece m ∧ 2 ≤ f q
    · obtain ⟨q, ℓ, hql, h2⟩ := hA2
      have hc0 : c q = 0 := by simp [hc, show f q ≠ 1 by omega]
      have := hsingle q ℓ hql
      rw [hc0, sub_zero] at this
      simp only [Bm]
      linarith [hJ q h2]
    · by_cases hA1 : ∃ q ℓ, lane q ℓ = .piece m ∧ f q = 1
      · obtain ⟨q, ℓ, hql, h1⟩ := hA1
        have hc1 : c q = K / 6 := by simp [hc, h1]
        have := hsingle q ℓ hql
        rw [hc1] at this
        simp only [Bm]
        linarith [hR1 q h1]
      · have hzero : ∀ q ℓ, (if lane q ℓ = .piece m then s q - c q else 0) =
            if lane q ℓ = .piece m then s q else 0 := fun q ℓ => by
          split_ifs with hql
          · have : f q ≠ 1 := fun h1 => hA1 ⟨q, ℓ, hql, h1⟩
            simp [hc, this]
          · rfl
        simp only [Bm, hzero]
        linarith [hRm m]
  calc K * (∑ q, (f q : ℝ) + Fintype.card M) = ∑ q, K * f q + ∑ _m : M, K := by
        rw [mul_add, Finset.mul_sum, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm K]
    _ ≤ ∑ q, A q + ∑ m, Bm m := add_le_add (Finset.sum_le_sum fun q _ => hA q) (Finset.sum_le_sum fun m _ => hB m)
    _ = _ := hsplit.symm

end TheoremH

/-! ### Theorem J and the rank bounds on quad runs, in exact arithmetic (P2-FF round 5 §3)

Over a field, a word made of `S` products of bit-affine operands has a bit-pair form `Σ a bᵀ + b aᵀ` of rank at most
`2 |S|`, and its diagonal is free. So it reaches at most `2 |S|` of the target's free-diagonal rank floor
(`PairRankGe`, `products_ge_of_pairRankGe`). Two lanes of a quad share each slice's `A` operand, so their summed word
has the quad's `s` products, and two distinct cells' summed target has floor `2K` (Theorem J, `QuadRun.theoremJ`);
one cell's target has floor `K` (`QuadRun.full_ge`, `QuadRun.merged_ge`). The floors come from a block-triangular
submatrix of determinant 1 (`pairRankGe_of_blocks`), on the re-encoded design from plane 0's weight 1
(`pairRankGe_dIncr`, `pairRankGe_dIncr_two`). Theorem H then holds on every quad run of the design
(`QuadRun.cost_ge_reEncoded`, at VF2's merge price `_32`). Like Theorem H, this route is an alternative: the chain
lemma gives the design's floor in exact arithmetic with merges free. -/

section QuadRuns

/-- The sum of two bit-affine operands. -/
def BitAff.add {V R : Type} [CommRing R] (A B : BitAff V R) : BitAff V R := ⟨A.c + B.c, fun v => A.α v + B.α v⟩

theorem BitAff.eval_add {V R : Type} [Fintype V] [CommRing R] (A B : BitAff V R) (z : V → R) :
    (A.add B).eval z = A.eval z + B.eval z := by
  simp only [BitAff.add, BitAff.eval, add_mul, Finset.sum_add_distrib]
  ring

theorem pairDiff_comm {V R : Type} [DecidableEq V] [CommRing R] (f : (V → R) → R) (u v : V) :
    pairDiff f u v = pairDiff f v u := by
  simp only [pairDiff, Finset.pair_comm]
  ring

variable {V F : Type} [Fintype V] [DecidableEq V] [Field F]

/-- **The free-diagonal rank floor** of a target `T`: every matrix that agrees with `T`'s bit-pair second differences
off the diagonal has rank at least `k`. In exact arithmetic a product's form `a bᵀ + b aᵀ` has diagonal `2 a_u b_u`,
which correctness on bits does not fix (`z_u² = z_u`), so this is what a correct word's products must reach. -/
def PairRankGe (T : (V → F) → F) (k : ℕ) : Prop :=
  ∀ M : Matrix V V F, (∀ u v, u ≠ v → M u v = pairDiff T u v) → k ≤ M.rank

/-- **The rank bound in exact arithmetic**: a word that agrees with `T` at every bit point with at most two ones, and
is a combination of the products of the bit-affine operands in `S` plus a word that projects to 0, has `2 |S|` at least
`T`'s free-diagonal rank floor. -/
theorem products_ge_of_pairRankGe {ι : Type} (S : Finset ι) (lam : ι → F) (A B : ι → BitAff V F)
    (r : (V → F) → F) (hr : ∀ u v, u ≠ v → pairDiff r u v = 0) (T : (V → F) → F) {k : ℕ}
    (hk : PairRankGe T k)
    (hT : ∀ U : Finset V, U.card ≤ 2 →
      T (bitInd U) = ∑ j ∈ S, lam j * ((A j).eval (bitInd U) * (B j).eval (bitInd U)) + r (bitInd U)) :
    k ≤ 2 * S.card := by
  set M : Matrix V V F := ∑ j ∈ S, (Matrix.vecMulVec (fun u => lam j * (A j).α u) (B j).α +
    Matrix.vecMulVec (B j).α (fun v => lam j * (A j).α v)) with hM
  have hoff : ∀ u v, u ≠ v → M u v = pairDiff T u v := by
    intro u v huv
    have h : pairDiff T u v =
        pairDiff (fun z => ∑ j ∈ S, lam j * ((A j).eval z * (B j).eval z) + r z) u v := by
      simp only [pairDiff]
      rw [hT {u, v} ((Finset.card_insert_le u {v}).trans (by simp)), hT {u} (by simp), hT {v} (by simp),
        hT ∅ (by simp)]
    rw [h, pairDiff_add (fun z => ∑ j ∈ S, lam j * ((A j).eval z * (B j).eval z)) r, hr u v huv, add_zero,
      pairDiff_sum S (fun j z => lam j * ((A j).eval z * (B j).eval z))]
    simp only [hM, Matrix.sum_apply, Matrix.add_apply, Matrix.vecMulVec_apply]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [pairDiff_mul_left (lam j) (fun z => (A j).eval z * (B j).eval z), pairDiff_mul (A j) (B j) huv]
    ring
  refine (hk M hoff).trans ?_
  rw [hM, ← Finset.sum_coe_sort S]
  exact (rank_sum_vecMulVec_le_card _ _ _ _).trans (by simp)

/-- A free-diagonal rank floor from an identity block off the diagonal. -/
theorem pairRankGe_of_one {ι : Type} [Fintype ι] [DecidableEq ι] (T : (V → F) → F) (ρ γ : ι → V)
    (hne : ∀ a b, ρ a ≠ γ b) (hT : ∀ a b, pairDiff T (ρ a) (γ b) = if a = b then 1 else 0) :
    PairRankGe T (Fintype.card ι) := by
  intro M hM
  have hsub : M.submatrix ρ γ = 1 := by
    ext a b
    rw [Matrix.submatrix_apply, hM _ _ (hne a b), hT, Matrix.one_apply]
  calc Fintype.card ι = (1 : Matrix ι ι F).rank := Matrix.rank_one.symm
    _ ≤ M.rank := by rw [← hsub]; exact Matrix.rank_submatrix_le M ρ γ

/-- **Two cells' free-diagonal rank floor** (Theorem J's step 3): identity blocks on rows `ρ₁` against columns `γ₁`
and on `ρ₂` against `γ₂`, with a zero block on `ρ₁` against `γ₂`, give rank `2k` whatever the rest, the diagonal
included: that submatrix is block-triangular with determinant 1. -/
theorem pairRankGe_of_blocks {ι : Type} [Fintype ι] [DecidableEq ι] (T : (V → F) → F) (ρ₁ ρ₂ γ₁ γ₂ : ι → V)
    (h11 : ∀ a b, ρ₁ a ≠ γ₁ b) (h12 : ∀ a b, ρ₁ a ≠ γ₂ b) (h22 : ∀ a b, ρ₂ a ≠ γ₂ b)
    (hT11 : ∀ a b, pairDiff T (ρ₁ a) (γ₁ b) = if a = b then 1 else 0)
    (hT12 : ∀ a b, pairDiff T (ρ₁ a) (γ₂ b) = 0)
    (hT22 : ∀ a b, pairDiff T (ρ₂ a) (γ₂ b) = if a = b then 1 else 0) :
    PairRankGe T (2 * Fintype.card ι) := by
  intro M hM
  have hsub : M.submatrix (Sum.elim ρ₁ ρ₂) (Sum.elim γ₁ γ₂) =
      Matrix.fromBlocks (1 : Matrix ι ι F) 0 (M.submatrix ρ₂ γ₁) (1 : Matrix ι ι F) := by
    ext a b
    rcases a with a | a <;> rcases b with b | b
    · rw [Matrix.submatrix_apply, Sum.elim_inl, Sum.elim_inl, Matrix.fromBlocks_apply₁₁, hM _ _ (h11 a b), hT11,
        Matrix.one_apply]
    · rw [Matrix.submatrix_apply, Sum.elim_inl, Sum.elim_inr, Matrix.fromBlocks_apply₁₂, hM _ _ (h12 a b), hT12,
        Matrix.zero_apply]
    · rw [Matrix.submatrix_apply, Sum.elim_inr, Sum.elim_inl, Matrix.fromBlocks_apply₂₁, Matrix.submatrix_apply]
    · rw [Matrix.submatrix_apply, Sum.elim_inr, Sum.elim_inr, Matrix.fromBlocks_apply₂₂, hM _ _ (h22 a b), hT22,
        Matrix.one_apply]
  have hunit : IsUnit (Matrix.fromBlocks (1 : Matrix ι ι F) 0 (M.submatrix ρ₂ γ₁) (1 : Matrix ι ι F)) := by
    rw [Matrix.isUnit_iff_isUnit_det, Matrix.det_fromBlocks_zero₁₂]
    simp
  calc 2 * Fintype.card ι = Fintype.card (ι ⊕ ι) := by rw [Fintype.card_sum]; ring
    _ = _ := (Matrix.rank_of_isUnit _ hunit).symm
    _ ≤ M.rank := by rw [← hsub]; exact Matrix.rank_submatrix_le M _ _

section DesignRank

variable {n L np B n' : ℕ}

theorem pairDiff_dIncr {R : Type} [CommRing R] (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) (i : Fin n) (l : Fin n') (t : Fin B) {u v : DBits n L np B n'} (huv : u ≠ v) :
    pairDiff (dIncr pc vq w i l t) u v =
      ∑ j, ((dP pc w i t j).α u * (dX vq t j l).α v + (dP pc w i t j).α v * (dX vq t j l).α u) := by
  unfold dIncr
  rw [pairDiff_sum univ (fun j z => (dP pc w i t j).eval z * (dX vq t j l).eval z)]
  exact Finset.sum_congr rfl fun j _ => pairDiff_mul _ _ huv

/-- A cell-block's target pairs `β_p(i, j)` with `b(t, j, l)` at weight `w_p`, and nothing else. -/
theorem pairDiff_dIncr_inl_inr {R : Type} [CommRing R] (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) (i : Fin n) (l : Fin n') (t : Fin B)
    (a : Fin n × Fin L × Fin np) (b : Fin B × Fin L × Fin n') :
    pairDiff (dIncr pc vq w i l t) (.inl a) (.inr b) = if a.1 = i ∧ b = (t, a.2.1, l) then w a.2.2 else 0 := by
  rw [pairDiff_dIncr pc vq w i l t Sum.inl_ne_inr]
  simp only [dP_α_inl, dP_α_inr, dX_α_inl, dX_α_inr, mul_zero, add_zero, ite_mul, zero_mul]
  rw [Finset.sum_eq_single a.2.1]
  · by_cases h1 : a.1 = i <;> by_cases h2 : b = (t, a.2.1, l) <;> simp [h1, h2]
  · intro j _ hj
    simp [Ne.symm hj]
  · simp

theorem pairDiff_dIncr_inl_inl {R : Type} [CommRing R] (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) (i : Fin n) (l : Fin n') (t : Fin B)
    {a a' : Fin n × Fin L × Fin np} (h : a ≠ a') :
    pairDiff (dIncr pc vq w i l t) (.inl a) (.inl a' : DBits n L np B n') = 0 := by
  rw [pairDiff_dIncr pc vq w i l t (fun h' => h (Sum.inl_injective h'))]
  simp [dX_α_inl]

theorem pairDiff_dIncr_inr_inr {R : Type} [CommRing R] (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) (i : Fin n) (l : Fin n') (t : Fin B)
    {b b' : Fin B × Fin L × Fin n'} (h : b ≠ b') :
    pairDiff (dIncr pc vq w i l t) (.inr b) (.inr b' : DBits n L np B n') = 0 := by
  rw [pairDiff_dIncr pc vq w i l t (fun h' => h (Sum.inr_injective h'))]
  simp [dP_α_inr]

/-- **A cell-block's free-diagonal rank floor** in exact arithmetic: `L`, from plane `p₀`'s weight 1. -/
theorem pairRankGe_dIncr (pc : Fin n → Fin B → Fin L → F) (vq : Fin B → Fin L → Fin n' → F) (w : Fin np → F)
    {p₀ : Fin np} (hw : w p₀ = 1) (i : Fin n) (l : Fin n') (t : Fin B) :
    PairRankGe (dIncr pc vq w i l t) L := by
  have := pairRankGe_of_one (dIncr pc vq w i l t) (fun j : Fin L => (.inl (i, j, p₀) : DBits n L np B n'))
    (fun j => .inr (t, j, l)) (fun _ _ => Sum.inl_ne_inr) (fun a b => by
      rw [pairDiff_dIncr_inl_inr]
      rcases eq_or_ne a b with rfl | hab
      · simp [hw]
      · simp [hab, Ne.symm hab])
  simpa using this

/-- **Two cell-blocks' free-diagonal rank floor** (Theorem J's step 3 on the design): `2L` for distinct cell-blocks.
Sharing a row `i` makes each `j` a path centred at `β_p₀(i, j)`, sharing `(t, l)` one centred at `b(t, j, l)`, and
otherwise the two edges are disjoint. -/
theorem pairRankGe_dIncr_two (pc : Fin n → Fin B → Fin L → F) (vq : Fin B → Fin L → Fin n' → F) (w : Fin np → F)
    {p₀ : Fin np} (hw : w p₀ = 1) (c c' : DCells n n' B) (hne : c ≠ c') :
    PairRankGe (fun z => dIncr pc vq w c.1 c.2.1 c.2.2 z + dIncr pc vq w c'.1 c'.2.1 c'.2.2 z) (2 * L) := by
  obtain ⟨i, l, t⟩ := c
  obtain ⟨i', l', t'⟩ := c'
  set T := fun z => dIncr pc vq w i l t z + dIncr pc vq w i' l' t' z with hTdef
  have hadd : ∀ u v, pairDiff T u v =
      pairDiff (dIncr pc vq w i l t) u v + pairDiff (dIncr pc vq w i' l' t') u v := fun u v =>
    pairDiff_add (dIncr pc vq w i l t) (dIncr pc vq w i' l' t') u v
  suffices h : PairRankGe T (2 * Fintype.card (Fin L)) by simpa using h
  by_cases hi : i = i'
  · subst hi
    have htl : ¬(t = t' ∧ l = l') := fun h => hne (by rw [h.1, h.2])
    refine pairRankGe_of_blocks T (fun j => .inr (t, j, l)) (fun j => .inl (i, j, p₀)) (fun j => .inl (i, j, p₀))
      (fun j => .inr (t', j, l')) (fun _ _ => Sum.inr_ne_inl) (fun a b h => htl ?_) (fun _ _ => Sum.inl_ne_inr)
      (fun a b => ?_) (fun a b => ?_) (fun a b => ?_)
    · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      exact ⟨h.1, h.2.2⟩
    · rw [hadd, pairDiff_comm, pairDiff_dIncr_inl_inr, pairDiff_comm, pairDiff_dIncr_inl_inr]
      by_cases hab : a = b
      · subst hab
        have : ¬((t, a, l) = (t', a, l')) := fun h => htl (by simp only [Prod.mk.injEq] at h; exact ⟨h.1, h.2.2⟩)
        simp [hw, this]
      · have : ¬((t, a, l) = (t', b, l')) := fun h => htl (by simp only [Prod.mk.injEq] at h; exact ⟨h.1, h.2.2⟩)
        simp [hab, this]
    · have hne' : (t, a, l) ≠ (t', b, l') := fun h => htl (by simp only [Prod.mk.injEq] at h; exact ⟨h.1, h.2.2⟩)
      rw [hadd, pairDiff_dIncr_inr_inr _ _ _ _ _ _ hne', pairDiff_dIncr_inr_inr _ _ _ _ _ _ hne', add_zero]
    · rw [hadd, pairDiff_dIncr_inl_inr, pairDiff_dIncr_inl_inr]
      by_cases hab : a = b
      · subst hab
        have : ¬((t', a, l') = (t, a, l)) := fun h => htl (by simp only [Prod.mk.injEq] at h; exact ⟨h.1.symm, h.2.2.symm⟩)
        simp [hw, this]
      · have : ¬((t', b, l') = (t, a, l)) := fun h => htl (by simp only [Prod.mk.injEq] at h; exact ⟨h.1.symm, h.2.2.symm⟩)
        simp [hab, this, Ne.symm hab]
  · by_cases htl : t = t' ∧ l = l'
    · obtain ⟨rfl, rfl⟩ := htl
      refine pairRankGe_of_blocks T (fun j => .inl (i, j, p₀)) (fun j => .inr (t, j, l)) (fun j => .inr (t, j, l))
        (fun j => .inl (i', j, p₀)) (fun _ _ => Sum.inl_ne_inr) (fun a b h => hi ?_) (fun _ _ => Sum.inr_ne_inl)
        (fun a b => ?_) (fun a b => ?_) (fun a b => ?_)
      · simp only [Sum.inl.injEq, Prod.mk.injEq] at h
        exact h.1
      · rw [hadd, pairDiff_dIncr_inl_inr, pairDiff_dIncr_inl_inr]
        rcases eq_or_ne a b with rfl | hab
        · simp [hi, hw]
        · simp [hab, hi, Ne.symm hab]
      · have hne' : (i, a, p₀) ≠ (i', b, p₀) := fun h => hi (by simp only [Prod.mk.injEq] at h; exact h.1)
        rw [hadd, pairDiff_dIncr_inl_inl _ _ _ _ _ _ hne', pairDiff_dIncr_inl_inl _ _ _ _ _ _ hne', add_zero]
      · rw [hadd, pairDiff_comm, pairDiff_dIncr_inl_inr, pairDiff_comm, pairDiff_dIncr_inl_inr]
        rcases eq_or_ne a b with rfl | hab
        · simp [Ne.symm hi, hw]
        · simp [hab, Ne.symm hi]
    · refine pairRankGe_of_blocks T (fun j => .inl (i, j, p₀)) (fun j => .inl (i', j, p₀)) (fun j => .inr (t, j, l))
        (fun j => .inr (t', j, l')) (fun _ _ => Sum.inl_ne_inr) (fun _ _ => Sum.inl_ne_inr)
        (fun _ _ => Sum.inl_ne_inr) (fun a b => ?_) (fun a b => ?_) (fun a b => ?_)
      · rw [hadd, pairDiff_dIncr_inl_inr, pairDiff_dIncr_inl_inr]
        rcases eq_or_ne a b with rfl | hab
        · simp [hi, hw]
        · simp [hab, hi, Ne.symm hab]
      · rw [hadd, pairDiff_dIncr_inl_inr, pairDiff_dIncr_inl_inr]
        have : ¬((t', b, l') = (t, a, l)) := fun h => htl (by simp only [Prod.mk.injEq] at h; exact ⟨h.1.symm, h.2.2.symm⟩)
        simp [hi, this]
      · rw [hadd, pairDiff_dIncr_inl_inr, pairDiff_dIncr_inl_inr]
        rcases eq_or_ne a b with rfl | hab
        · simp [Ne.symm hi, hw]
        · simp [hab, Ne.symm hi, Ne.symm hab]

end DesignRank

/-- **A quad run** in exact arithmetic over a field `F` (round 5 §3, the persistent quads of Theorem J″): quad `q` runs
`s q` slices, and slice `j` has one operand `A` for the whole quad and an operand `B` for each lane, so lane `ℓ`'s word
is `Σ_j A_j B_jℓ` plus a bit-affine word. Each lane is a full cell, a piece of a merged cell, or junk (`lane`), and
merged cell `m`'s word is a combination of its pieces' words plus a bit-affine word. -/
structure QuadRun (Q M V F : Type) where
  s : Q → ℕ
  A : (q : Q) → Fin (s q) → BitAff V F
  B : (q : Q) → Fin (s q) → Fin 4 → BitAff V F
  off : Q → Fin 4 → BitAff V F
  lane : Q → Fin 4 → QLane M
  lam : Q → Fin 4 → F
  offM : M → BitAff V F

namespace QuadRun

variable {Q M C : Type} [Fintype Q] [Fintype M] [DecidableEq M]

/-- Lane `ℓ`'s word. -/
def word (P : QuadRun Q M V F) (q : Q) (ℓ : Fin 4) (z : V → F) : F :=
  ∑ j, (P.A q j).eval z * (P.B q j ℓ).eval z + (P.off q ℓ).eval z

/-- Merged cell `m`'s word. -/
def mword (P : QuadRun Q M V F) (m : M) (z : V → F) : F :=
  ∑ q, ∑ ℓ, (if P.lane q ℓ = .piece m then P.lam q ℓ else 0) * P.word q ℓ z + (P.offM m).eval z

/-- The run computes cells `tgt` on its full lanes (`cell q ℓ`, distinct within a quad) and `tgtM` on its merged
cells, at every bit point with at most two ones. -/
structure Computes (P : QuadRun Q M V F) (cell : Q → Fin 4 → C) (tgt : C → (V → F) → F)
    (tgtM : M → (V → F) → F) : Prop where
  full : ∀ q ℓ, P.lane q ℓ = .full → ∀ U : Finset V, U.card ≤ 2 →
    P.word q ℓ (bitInd U) = tgt (cell q ℓ) (bitInd U)
  merged : ∀ m (U : Finset V), U.card ≤ 2 → P.mword m (bitInd U) = tgtM m (bitInd U)
  distinct : ∀ q ℓ ℓ', ℓ ≠ ℓ' → P.lane q ℓ = .full → P.lane q ℓ' = .full → cell q ℓ ≠ cell q ℓ'

variable {P : QuadRun Q M V F} {cell : Q → Fin 4 → C} {tgt : C → (V → F) → F} {tgtM : M → (V → F) → F}

omit [Fintype M] in
/-- **The rank bound on one full lane**: a quad holding a cell with free-diagonal floor `K` has `2 s ≥ K`. -/
theorem full_ge (hC : P.Computes cell tgt tgtM) {K : ℕ} (hone : ∀ c, PairRankGe (tgt c) K) (q : Q) (ℓ : Fin 4)
    (hℓ : P.lane q ℓ = .full) : K ≤ 2 * P.s q := by
  have := products_ge_of_pairRankGe univ (fun _ => (1 : F)) (P.A q) (fun j => P.B q j ℓ)
    (fun z => (P.off q ℓ).eval z) (fun u v huv => pairDiff_aff _ huv) (tgt (cell q ℓ)) (hone _)
    (fun U hU => by rw [← hC.full q ℓ hℓ U hU, word]; simp)
  simpa using this

omit [Fintype M] in
/-- **Theorem J** (P2-FF round 5 §3) on a quad run: a quad with two full lanes, holding distinct cells whose summed
targets have free-diagonal floor `2K`, has `s ≥ K`. The two lanes' words share each slice's `A`, so their sum is
`Σ_j A_j (B_jℓ + B_jℓ')` plus a bit-affine word: `s` products. -/
theorem theoremJ (hC : P.Computes cell tgt tgtM) {K : ℕ}
    (hpair : ∀ c c', c ≠ c' → PairRankGe (fun z => tgt c z + tgt c' z) (2 * K)) (q : Q)
    (h2 : 2 ≤ (univ.filter fun ℓ => P.lane q ℓ = .full).card) : K ≤ P.s q := by
  obtain ⟨ℓ, hℓ, ℓ', hℓ', hne⟩ := Finset.one_lt_card.1 h2
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hℓ hℓ'
  have := products_ge_of_pairRankGe univ (fun _ => (1 : F)) (P.A q) (fun j => (P.B q j ℓ).add (P.B q j ℓ'))
    (fun z => (P.off q ℓ).eval z + (P.off q ℓ').eval z)
    (fun u v huv => by
      rw [pairDiff_add (fun z => (P.off q ℓ).eval z) (fun z => (P.off q ℓ').eval z), pairDiff_aff _ huv,
        pairDiff_aff _ huv, add_zero])
    _ (hpair _ _ (hC.distinct q ℓ ℓ' hne hℓ hℓ'))
    (fun U hU => by
      rw [← hC.full q ℓ hℓ U hU, ← hC.full q ℓ' hℓ' U hU, word, word]
      simp only [BitAff.eval_add, one_mul, mul_add, Finset.sum_add_distrib]
      ring)
  simp only [Finset.card_univ, Fintype.card_fin] at this
  omega

omit [Fintype M] in
/-- **The rank bound on a merged cell**: its pieces carry `2 Σ s ≥ K` slices, for a target of floor `K`. -/
theorem merged_ge (hC : P.Computes cell tgt tgtM) {K : ℕ} (hM : ∀ m, PairRankGe (tgtM m) K) (m : M) :
    K ≤ 2 * ∑ q, ∑ ℓ, if P.lane q ℓ = .piece m then P.s q else 0 := by
  classical
  let S : Finset (Σ q : Q, Fin 4 × Fin (P.s q)) := univ.filter fun x => P.lane x.1 x.2.1 = .piece m
  have hcard : S.card = ∑ q, ∑ ℓ, if P.lane q ℓ = .piece m then P.s q else 0 := by
    simp only [S, Finset.card_filter, Fintype.sum_sigma, Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun q _ => Finset.sum_congr rfl fun ℓ _ => ?_
    split_ifs <;> simp
  rw [← hcard]
  refine products_ge_of_pairRankGe S (fun x => P.lam x.1 x.2.1) (fun x => P.A x.1 x.2.2)
    (fun x => P.B x.1 x.2.2 x.2.1)
    (fun z => ∑ q, ∑ ℓ, (if P.lane q ℓ = .piece m then P.lam q ℓ else 0) * (P.off q ℓ).eval z + (P.offM m).eval z)
    (fun u v huv => ?_) (tgtM m) (hM m) (fun U hU => ?_)
  · rw [pairDiff_add (fun z => ∑ q, ∑ ℓ, (if P.lane q ℓ = .piece m then P.lam q ℓ else 0) * (P.off q ℓ).eval z)
      (fun z => (P.offM m).eval z), pairDiff_aff _ huv, add_zero,
      pairDiff_sum univ (fun q z => ∑ ℓ, (if P.lane q ℓ = .piece m then P.lam q ℓ else 0) * (P.off q ℓ).eval z)]
    refine Finset.sum_eq_zero fun q _ => ?_
    rw [pairDiff_sum univ (fun ℓ z => (if P.lane q ℓ = .piece m then P.lam q ℓ else 0) * (P.off q ℓ).eval z)]
    refine Finset.sum_eq_zero fun ℓ _ => ?_
    rw [pairDiff_mul_left _ (fun z => (P.off q ℓ).eval z), pairDiff_aff _ huv, mul_zero]
  · rw [← hC.merged m U hU, mword]
    have hS : ∀ g : (Σ q : Q, Fin 4 × Fin (P.s q)) → F, ∑ x ∈ S, g x =
        ∑ q, ∑ ℓ, ∑ j, if P.lane q ℓ = .piece m then g ⟨q, (ℓ, j)⟩ else 0 := fun g => by
      simp only [S, Finset.sum_filter, Fintype.sum_sigma, Fintype.sum_prod_type]
    rw [hS]
    simp only [word]
    have h : ∀ q ℓ, (if P.lane q ℓ = .piece m then P.lam q ℓ else 0) *
        (∑ j, (P.A q j).eval (bitInd U) * (P.B q j ℓ).eval (bitInd U) + (P.off q ℓ).eval (bitInd U)) =
        (∑ j, if P.lane q ℓ = .piece m then
          P.lam q ℓ * ((P.A q j).eval (bitInd U) * (P.B q j ℓ).eval (bitInd U)) else 0) +
        (if P.lane q ℓ = .piece m then P.lam q ℓ else 0) * (P.off q ℓ).eval (bitInd U) := fun q ℓ => by
      split_ifs <;> simp [mul_add, Finset.mul_sum]
    simp only [h, Finset.sum_add_distrib]
    ring

/-- **Theorem H on a quad run** (round 5 §3 and round 6 §6, in exact arithmetic): with merges at `h ≥ 2K/3` slots, a
run whose cells have free-diagonal floor `K`, and pairs of distinct cells `2K`, pays at least `K` slots a useful cell
(a full lane or a merged cell). Theorem J and the rank bounds are `theoremJ`, `full_ge` and `merged_ge`. -/
theorem cost_ge (hC : P.Computes cell tgt tgtM) {K : ℕ} (hone : ∀ c, PairRankGe (tgt c) K)
    (hpair : ∀ c c', c ≠ c' → PairRankGe (fun z => tgt c z + tgt c' z) (2 * K))
    (hM : ∀ m, PairRankGe (tgtM m) K) (h : ℝ) (hh : 2 * (K : ℝ) / 3 ≤ h) :
    (K : ℝ) * (∑ q, ((univ.filter fun ℓ => P.lane q ℓ = .full).card : ℝ) + Fintype.card M) ≤
      ∑ q, 4 * (P.s q : ℝ) + h * Fintype.card M := by
  refine theoremH P.lane (fun q => (P.s q : ℝ)) K h (fun q => Nat.cast_nonneg _) (Nat.cast_nonneg _) hh
    (fun q h2 => by exact_mod_cast QuadRun.theoremJ hC hpair q h2) (fun q h1 => ?_) (fun m => ?_)
  · obtain ⟨ℓ, hℓ⟩ := Finset.card_eq_one.1 h1
    have hmem : ℓ ∈ univ.filter fun ℓ => P.lane q ℓ = .full := by rw [hℓ]; exact Finset.mem_singleton_self ℓ
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hmem
    have h1' : (K : ℝ) ≤ 2 * (P.s q : ℝ) := by exact_mod_cast QuadRun.full_ge hC hone q ℓ hmem
    linarith
  · have h1' : (K : ℝ) ≤ 2 * ∑ q, ∑ ℓ, if P.lane q ℓ = .piece m then (P.s q : ℝ) else 0 := by
      exact_mod_cast QuadRun.merged_ge hC hM m
    linarith

end QuadRun

/-- **Theorem H on the re-encoded design**, in exact arithmetic over a field (FP8 operands and scaled reads
included): a quad run whose full lanes and merged cells hold cell-blocks pays `L` slots a useful cell-block once a
merged cell costs `h ≥ 2L/3` slots. Theorem J and the rank bounds come from the design's free-diagonal floors
(`pairRankGe_dIncr`, `pairRankGe_dIncr_two`). -/
theorem QuadRun.cost_ge_reEncoded {Q M : Type} [Fintype Q] [Fintype M] [DecidableEq M] {n L np B n' : ℕ}
    {P : QuadRun Q M (DBits n L np B n') F} (pc : Fin n → Fin B → Fin L → F) (vq : Fin B → Fin L → Fin n' → F)
    (w : Fin np → F) {p₀ : Fin np} (hw : w p₀ = 1) (cell : Q → Fin 4 → DCells n n' B) (cellM : M → DCells n n' B)
    (hC : P.Computes cell (fun c => dIncr pc vq w c.1 c.2.1 c.2.2)
      (fun m => dIncr pc vq w (cellM m).1 (cellM m).2.1 (cellM m).2.2))
    (h : ℝ) (hh : 2 * (L : ℝ) / 3 ≤ h) :
    (L : ℝ) * (∑ q, ((univ.filter fun ℓ => P.lane q ℓ = .full).card : ℝ) + Fintype.card M) ≤
      ∑ q, 4 * (P.s q : ℝ) + h * Fintype.card M :=
  QuadRun.cost_ge hC (fun _ => pairRankGe_dIncr pc vq w hw _ _ _) (fun c c' hne => pairRankGe_dIncr_two pc vq w hw c c' hne)
    (fun _ => pairRankGe_dIncr pc vq w hw _ _ _) h hh

/-- At VF2's merge price, 32 MAC-units a merged word on int8-rate kinds (round 6 §6), every `L ≤ 48` qualifies:
`c = 1` at the adopted `L = 32`. -/
theorem QuadRun.cost_ge_reEncoded_32 {Q M : Type} [Fintype Q] [Fintype M] [DecidableEq M] {n L np B n' : ℕ}
    {P : QuadRun Q M (DBits n L np B n') F} (pc : Fin n → Fin B → Fin L → F) (vq : Fin B → Fin L → Fin n' → F)
    (w : Fin np → F) {p₀ : Fin np} (hw : w p₀ = 1) (cell : Q → Fin 4 → DCells n n' B) (cellM : M → DCells n n' B)
    (hC : P.Computes cell (fun c => dIncr pc vq w c.1 c.2.1 c.2.2)
      (fun m => dIncr pc vq w (cellM m).1 (cellM m).2.1 (cellM m).2.2))
    (hL : L ≤ 48) :
    (L : ℝ) * (∑ q, ((univ.filter fun ℓ => P.lane q ℓ = .full).card : ℝ) + Fintype.card M) ≤
      ∑ q, 4 * (P.s q : ℝ) + 32 * Fintype.card M :=
  QuadRun.cost_ge_reEncoded pc vq w hw cell cellM hC 32 (by
    have : (L : ℝ) ≤ 48 := by exact_mod_cast hL
    linarith)

end QuadRuns

/-! ### The chain lemma (RS-E-REF r4 IV.2)

With stacked identities every k-block reads the same U bits, so the last committed prefix word has `F₂` rank `2L`
only, and Theorem K on increments does not add up when one product changes parity at several blocks. The chain lemma
reads the cross block instead. The bits split into U's side and the coins' side (`Vl ⊕ Vr`). A product `g · h` of a
function of U's bits and a function of the coins, of any form (table lookups, OR, wrapped arithmetic: `OneSided`), has
a `(β, b)` block of rank 1 (`OneSided.crossRankLe`), and an affine product, mixed or not, of rank at most 2
(`AffProd.crossRankLe`). Stack `T` committed prefix words of one row at distinct (column, block) pairs and flatten
along `b`: over any field the ring maps to (`F₂` for wrapped arithmetic, the field itself for exact) the products give
rank at most their count (`rank_cross_le`), and the targets give rank `T L`, since the column of `b(t′, j, l)` is
`Σ_{t ≥ t′} e_t ⊗ e_j`, triangular in `t` within a column (`rank_cross_dRow`). So `T` committed prefix words of a row
need `T L` one-sided products (`chainBound_oneSided`), and `T L / 2` once mixed affine ones are allowed by the cross
block alone (`chainBound_two`). Only the cross block is read there, so the bound holds whatever a product's `β–β` or
`b–b` part. Mixed products at factor 1 are Lemma G (RS-E-REF r5 V.1): for operands any function of U's bits plus
affine in the bits (`GProd`), the words' cross blocks are the products' alternating forms (`pairDiff_mul_flat`), and
over characteristic 2, with the targets' coin–coin block zero (`pairDiff_dPrefix_inr_inr`), removing one product
with a coin in its support changes the cross block by rank one (`altForm_subst`, `lemmaG`), so the products number
at least `T L` (`chainBound_gProd`). A round's run is a straight-line program with products of any form (`SlpG`),
and the count adds up over rows that share no product, with partners, merges and sharing among a row's cells free
(`SlpG.cells_card_le`). That gives the window's cost line (`RoundsHypK`, `roundsHyp_of_K`). -/

section Chain

variable {Vl Vr R : Type} [Fintype Vl] [Fintype Vr] [DecidableEq Vl] [DecidableEq Vr] [CommRing R]

/-- A product's cross block (its second differences on the pairs `(U bit, coin)`) is a sum of `m` rank-one terms. -/
def CrossRankLe (π : (Vl ⊕ Vr → R) → R) (m : ℕ) : Prop :=
  ∃ (a : Fin m → Vl → R) (b : Fin m → Vr → R), ∀ u v, pairDiff π (.inl u) (.inr v) = ∑ s, a s u * b s v

omit [Fintype Vl] [Fintype Vr] in
/-- A one-sided product's cross block is the outer product of its factors' first differences: rank 1. -/
theorem OneSided.crossRankLe {π : (Vl ⊕ Vr → R) → R} (hπ : OneSided π) : CrossRankLe π 1 := by
  obtain ⟨g, h, hg, hh, hgh⟩ := hπ
  refine ⟨fun _ u => g (bitInd {.inl u}) - g (bitInd ∅), fun _ v => h (bitInd {.inr v}) - h (bitInd ∅),
    fun u v => ?_⟩
  have e1 : g (bitInd {.inl u, .inr v}) = g (bitInd {.inl u}) := hg _ _ fun a => by simp [bitInd]
  have e2 : g (bitInd {.inr v}) = g (bitInd ∅) := hg _ _ fun a => by simp [bitInd]
  have e3 : h (bitInd {.inl u, .inr v}) = h (bitInd {.inr v}) := hh _ _ fun b => by simp [bitInd]
  have e4 : h (bitInd {.inl u}) = h (bitInd ∅) := hh _ _ fun b => by simp [bitInd]
  simp only [pairDiff, hgh, e1, e2, e3, e4, Fin.sum_univ_one]
  ring

/-- An affine product's cross block has rank at most 2. -/
theorem AffProd.crossRankLe {π : (Vl ⊕ Vr → R) → R} (hπ : AffProd π) : CrossRankLe π 2 := by
  obtain ⟨A, B, hAB⟩ := hπ
  refine ⟨![fun u => A.α (.inl u), fun u => B.α (.inl u)], ![fun v => B.α (.inr v), fun v => A.α (.inr v)],
    fun u v => ?_⟩
  have : pairDiff π (.inl u) (.inr v) = pairDiff (fun z => A.eval z * B.eval z) (.inl u) (.inr v) := by
    simp only [pairDiff, hAB]
  rw [this, pairDiff_mul A B Sum.inl_ne_inr, Fin.sum_univ_two]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
  ring

/-- An operand that is any function of U's bits plus a bit-affine function (RS-E-REF r5 V.1's class: recodings,
tables, any per-entry function of U, plus a byte's affine dependence on the coins). -/
def UAffB (o : (Vl ⊕ Vr → R) → R) : Prop :=
  ∃ f, LeftOnly f ∧ ∃ A : BitAff (Vl ⊕ Vr) R, ∀ z, o z = f z + A.eval z

/-- **A product of two such operands**: mixed bytes, any function of U's bits plus affine in the coins. -/
def GProd (π : (Vl ⊕ Vr → R) → R) : Prop :=
  ∃ o o', UAffB o ∧ UAffB o' ∧ ∀ z, π z = o z * o' z

omit [DecidableEq Vl] [DecidableEq Vr] in
theorem AffProd.gProd {π : (Vl ⊕ Vr → R) → R} (hπ : AffProd π) : GProd π := by
  obtain ⟨A, B, hAB⟩ := hπ
  exact ⟨_, _, ⟨fun _ => 0, fun _ _ _ => rfl, A, fun z => (zero_add _).symm⟩,
    ⟨fun _ => 0, fun _ _ _ => rfl, B, fun z => (zero_add _).symm⟩, hAB⟩

/-- An operand's first difference at `0` along bit `u`. -/
def fdiff (o : (Vl ⊕ Vr → R) → R) (u : Vl ⊕ Vr) : R := o (bitInd {u}) - o (bitInd ∅)

omit [Fintype Vl] [Fintype Vr] in
/-- A product of two operands with zero second difference on the pair `(u, v)` has second difference
`δo(u) δo′(v) + δo′(u) δo(v)` there. -/
theorem pairDiff_mul_flat {o o' : (Vl ⊕ Vr → R) → R} {u v : Vl ⊕ Vr} (ho : pairDiff o u v = 0)
    (ho' : pairDiff o' u v = 0) :
    pairDiff (fun z => o z * o' z) u v = fdiff o u * fdiff o' v + fdiff o' u * fdiff o v := by
  simp only [pairDiff, fdiff] at *
  linear_combination o' (bitInd {u, v}) * ho +
    (o (bitInd {u}) + o (bitInd {v}) - o (bitInd ∅)) * ho'

theorem UAffB.flat_inl_inr {o : (Vl ⊕ Vr → R) → R} (ho : UAffB o) (x : Vl) (y : Vr) :
    pairDiff o (.inl x) (.inr y) = 0 := by
  obtain ⟨f, hf, A, hA⟩ := ho
  have e : pairDiff o (.inl x) (.inr y) = pairDiff (fun z => f z + A.eval z) (.inl x) (.inr y) := by
    simp only [pairDiff, hA]
  have e1 : f (bitInd {.inl x, .inr y}) = f (bitInd {.inl x}) := hf _ _ fun a => by simp [bitInd]
  have e2 : f (bitInd {.inr y}) = f (bitInd ∅) := hf _ _ fun a => by simp [bitInd]
  rw [e, pairDiff_add, pairDiff_aff _ Sum.inl_ne_inr, add_zero]
  simp only [pairDiff, e1, e2]
  ring

theorem UAffB.flat_inr_inr {o : (Vl ⊕ Vr → R) → R} (ho : UAffB o) {y y' : Vr} (h : y ≠ y') :
    pairDiff o (.inr y) (.inr y') = 0 := by
  obtain ⟨f, hf, A, hA⟩ := ho
  have e : pairDiff o (.inr y) (.inr y') = pairDiff (fun z => f z + A.eval z) (.inr y) (.inr y') := by
    simp only [pairDiff, hA]
  have e1 : f (bitInd {.inr y, .inr y'}) = f (bitInd ∅) := hf _ _ fun a => by simp [bitInd]
  have e2 : f (bitInd {.inr y}) = f (bitInd ∅) := hf _ _ fun a => by simp [bitInd]
  have e3 : f (bitInd {.inr y'}) = f (bitInd ∅) := hf _ _ fun a => by simp [bitInd]
  rw [e, pairDiff_add, pairDiff_aff _ (fun h' => h (Sum.inr_injective h')), add_zero]
  simp only [pairDiff, e1, e2, e3]
  ring

/-- **The flattened cross block of a stack of words**: words `T_t`, each a combination of the products in `S` plus a
bit-affine word at every bit point with at most two ones, have a stacked cross block (rows `(t, U bit)`, columns
coins) of rank at most the products' summed cross ranks, over any field the ring maps to. -/
theorem rank_cross_le {F : Type} [Field F] (φ : R →+* F) {ι τ : Type} [Fintype τ] (S : Finset ι)
    (π : ι → (Vl ⊕ Vr → R) → R) (m : ι → ℕ) (hπ : ∀ k ∈ S, CrossRankLe (π k) (m k)) (c : τ → ι → R)
    (r : τ → BitAff (Vl ⊕ Vr) R) (T : τ → (Vl ⊕ Vr → R) → R)
    (hT : ∀ t (U : Finset (Vl ⊕ Vr)), U.card ≤ 2 →
      T t (bitInd U) = ∑ k ∈ S, c t k * π k (bitInd U) + (r t).eval (bitInd U)) :
    (Matrix.of fun (x : τ × Vl) (v : Vr) => φ (pairDiff (T x.1) (.inl x.2) (.inr v))).rank ≤ ∑ k ∈ S, m k := by
  classical
  choose a b hab using hπ
  let U : Matrix (τ × Vl) (Σ k : S, Fin (m k)) F := fun x s => φ (c x.1 s.1) * φ (a s.1 s.1.2 s.2 x.2)
  let W : Matrix (Σ k : S, Fin (m k)) Vr F := fun s v => φ (b s.1 s.1.2 s.2 v)
  have hUW : (Matrix.of fun (x : τ × Vl) (v : Vr) => φ (pairDiff (T x.1) (.inl x.2) (.inr v))) = U * W := by
    ext x v
    have h1 : pairDiff (T x.1) (.inl x.2) (.inr v) =
        pairDiff (fun z => ∑ k ∈ S, c x.1 k * π k z + (r x.1).eval z) (.inl x.2) (.inr v) := by
      simp only [pairDiff]
      rw [hT x.1 {.inl x.2, .inr v} ((Finset.card_insert_le _ _).trans (by simp)), hT x.1 {.inl x.2} (by simp),
        hT x.1 {.inr v} (by simp), hT x.1 ∅ (by simp)]
    have hpd : pairDiff (T x.1) (.inl x.2) (.inr v) = ∑ k ∈ S, c x.1 k * pairDiff (π k) (.inl x.2) (.inr v) := by
      rw [h1, pairDiff_add (fun z => ∑ k ∈ S, c x.1 k * π k z) (fun z => (r x.1).eval z),
        pairDiff_aff _ Sum.inl_ne_inr, add_zero, pairDiff_sum S (fun k z => c x.1 k * π k z)]
      exact Finset.sum_congr rfl fun k _ => pairDiff_mul_left _ _ _ _
    rw [Matrix.of_apply, hpd, Matrix.mul_apply, Fintype.sum_sigma, ← Finset.sum_coe_sort S, map_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [hab k.1 k.2, Finset.mul_sum, map_sum]
    refine Finset.sum_congr rfl fun s _ => ?_
    simp only [U, W, map_mul]
    ring
  rw [hUW]
  refine (Matrix.rank_mul_le_left U W).trans ((Matrix.rank_le_card_width U).trans ?_)
  rw [Fintype.card_sigma]
  simp only [Fintype.card_fin]
  exact (Finset.sum_coe_sort S m).le

end Chain

section LemmaG

variable {F X Y ι τ : Type} [Field F] [Fintype X] [Fintype Y] [Fintype τ] [DecidableEq Y]

/-- A family of products' summed alternating form on the bit pair `(u, v)` at word `t`:
`Σ_r c_tr (a_r(u) b_r(v) + b_r(u) a_r(v))`, each product given by its operands' first differences `a_r`, `b_r`. -/
def altForm (S : Finset ι) (a b : ι → X ⊕ Y → F) (c : τ → ι → F) (t : τ) (u v : X ⊕ Y) : F :=
  ∑ r ∈ S, c t r * (a r u * b r v + b r u * a r v)

omit [Fintype X] [Fintype Y] [Fintype τ] [DecidableEq Y] in
theorem altForm_self (h2 : (2 : F) = 0) (S : Finset ι) (a b : ι → X ⊕ Y → F) (c : τ → ι → F) (t : τ)
    (u : X ⊕ Y) : altForm S a b c t u u = 0 :=
  Finset.sum_eq_zero fun r _ => by linear_combination (c t r * a r u * b r u) * h2

omit [Fintype X] [Fintype Y] [Fintype τ] [DecidableEq Y] in
theorem altForm_swap (S : Finset ι) (a b : ι → X ⊕ Y → F) (c : τ → ι → F) (t : τ) (u v : X ⊕ Y) :
    altForm S b a c t u v = altForm S a b c t u v :=
  Finset.sum_congr rfl fun r _ => by ring

omit [Fintype X] [Fintype Y] [Fintype τ] [DecidableEq Y] in
/-- **Winograd's substitution, one product** (RS-E-REF r5 V.1): on the hyperplane where product `r₀`'s first
operand vanishes, solved for its coin `y₀` (`s = a_r₀(y₀) ≠ 0`), `r₀` drops out and every operand `ℓ` becomes
`ℓ − ℓ(y₀)/s · a_r₀`; over characteristic 2 the forms change by `a_r₀ ∧ (column y₀)` only. -/
theorem altForm_subst [DecidableEq ι] (h2 : (2 : F) = 0) (S : Finset ι) (a b : ι → X ⊕ Y → F) (c : τ → ι → F)
    {r₀ : ι} (y₀ : Y) (hs : a r₀ (.inr y₀) ≠ 0) (t : τ) (u v : X ⊕ Y) :
    altForm (S.erase r₀) (fun r w => a r w - a r (.inr y₀) / a r₀ (.inr y₀) * a r₀ w)
        (fun r w => b r w - b r (.inr y₀) / a r₀ (.inr y₀) * a r₀ w) c t u v =
      altForm S a b c t u v - (a r₀ v * altForm S a b c t u (.inr y₀) + a r₀ u * altForm S a b c t v (.inr y₀)) /
        a r₀ (.inr y₀) := by
  unfold altForm
  have h0 : ∀ w, a r₀ w - a r₀ (.inr y₀) / a r₀ (.inr y₀) * a r₀ w = 0 := fun w => by
    rw [div_self hs, one_mul, sub_self]
  rw [Finset.sum_erase _ (by simp only [h0, zero_mul, mul_zero, add_zero])]
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib, Finset.sum_div, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun r _ => ?_
  field_simp
  linear_combination (c t r * a r (.inr y₀) * b r (.inr y₀) * a r₀ u * a r₀ v) * h2

/-- **Lemma G** (RS-E-REF r5 V.1, over characteristic 2): products whose operands have first differences `a_r`,
`b_r`, combined into words whose coin–coin block vanishes, give a stacked cross block (rows `(word, U bit)`, columns
coins) of rank at most the number of products. U–U junk is free. By induction, removing one product with a coin in
its support (`altForm_subst`): the coin–coin blocks stay zero and the cross block changes by a rank-one term. -/
theorem lemmaG (h2 : (2 : F) = 0) (S : Finset ι) (a b : ι → X ⊕ Y → F) (c : τ → ι → F)
    (hYY : ∀ t y y', y ≠ y' → altForm S a b c t (.inr y) (.inr y') = 0) :
    (Matrix.of fun (x : τ × X) (y : Y) => altForm S a b c x.1 (.inl x.2) (.inr y)).rank ≤ S.card := by
  classical
  induction S using Finset.strongInduction generalizing a b with
  | H S ih =>
  have hYY' : ∀ t y y', altForm S a b c t (.inr y) (.inr y') = 0 := fun t y y' => by
    by_cases hyy : y = y'
    · subst hyy
      exact altForm_self h2 S a b c t _
    · exact hYY t y y' hyy
  by_cases h0 : ∀ r ∈ S, ∀ y, a r (.inr y) = 0 ∧ b r (.inr y) = 0
  · have : (Matrix.of fun (x : τ × X) (y : Y) => altForm S a b c x.1 (.inl x.2) (.inr y)) = 0 := by
      ext x y
      simp only [Matrix.of_apply, Matrix.zero_apply, altForm]
      refine Finset.sum_eq_zero fun r hr => ?_
      rw [(h0 r hr y).1, (h0 r hr y).2]
      ring
    rw [this, Matrix.rank_zero]
    exact Nat.zero_le _
  push Not at h0
  obtain ⟨r₀, hr₀, y₀, hor⟩ := h0
  wlog hs : a r₀ (.inr y₀) ≠ 0 generalizing a b
  · have hb : b r₀ (.inr y₀) ≠ 0 := hor (not_not.mp hs)
    have := this b a (fun t y y' hyy => by rw [altForm_swap]; exact hYY t y y' hyy)
      (fun t y y' => by rw [altForm_swap]; exact hYY' t y y') (fun h => absurd h hb) hb
    simpa only [altForm_swap] using this
  have hsub := altForm_subst h2 S a b c y₀ hs
  have h' := ih (S.erase r₀) (Finset.erase_ssubset hr₀)
    (fun r w => a r w - a r (.inr y₀) / a r₀ (.inr y₀) * a r₀ w)
    (fun r w => b r w - b r (.inr y₀) / a r₀ (.inr y₀) * a r₀ w) fun t y y' _ => by
      rw [hsub, hYY' t y y', hYY' t y y₀, hYY' t y' y₀]
      simp
  have hsplit : (Matrix.of fun (x : τ × X) (y : Y) => altForm S a b c x.1 (.inl x.2) (.inr y)) =
      (Matrix.of fun (x : τ × X) (y : Y) => altForm (S.erase r₀)
          (fun r w => a r w - a r (.inr y₀) / a r₀ (.inr y₀) * a r₀ w)
          (fun r w => b r w - b r (.inr y₀) / a r₀ (.inr y₀) * a r₀ w) c x.1 (.inl x.2) (.inr y)) +
        Matrix.vecMulVec (fun x : τ × X => altForm S a b c x.1 (.inl x.2) (.inr y₀))
          (fun y => a r₀ (.inr y) / a r₀ (.inr y₀)) := by
    ext x y
    rw [Matrix.add_apply, Matrix.of_apply, Matrix.of_apply, hsub, hYY' x.1 y y₀, Matrix.vecMulVec_apply]
    field_simp
    ring
  rw [hsplit]
  calc _ ≤ _ + _ := Pouw.SteinerRigidity.rank_add_le _ _
    _ ≤ (S.erase r₀).card + 1 := add_le_add h' (Matrix.rank_vecMulVec_le _ _)
    _ = S.card := Finset.card_erase_add_one hr₀

end LemmaG

section ChainDesign

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

/-- A committed prefix word pairs `β_p(i, j)` with every `b(t′, j, l)`, `t′ ≤ t`, at weight `w_p`. -/
theorem pairDiff_dPrefix_inl_inr (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) (i : Fin n) (l : Fin n') (t : ℕ) (a : Fin n × Fin L × Fin np) (b : Fin B × Fin L × Fin n') :
    pairDiff (dPrefix pc vq w i l t) (.inl a) (.inr b) =
      if a.1 = i ∧ b.2 = (a.2.1, l) ∧ b.1.val ≤ t then w a.2.2 else 0 := by
  have h : dPrefix pc vq w i l t = fun z => ∑ t' : Fin B, (if t'.val ≤ t then 1 else 0) * dIncr pc vq w i l t' z := by
    funext z
    simp only [dPrefix, ite_mul, one_mul, zero_mul]
  rw [h, pairDiff_sum univ (fun t' z => (if t'.val ≤ t then 1 else 0) * dIncr pc vq w i l t' z)]
  simp_rw [pairDiff_mul_left, pairDiff_dIncr_inl_inr]
  rw [Finset.sum_eq_single b.1]
  · obtain ⟨t₁, j₁, l₁⟩ := b
    simp only [Prod.mk.injEq, true_and]
    by_cases h3 : t₁.val ≤ t
    · simp [h3]
    · simp [h3]
  · intro t' _ ht'
    have : ¬(b = (t', a.2.1, l)) := fun h => ht' (by rw [h])
    simp [this]
  · simp

/-- A committed prefix word has no coin–coin terms. -/
theorem pairDiff_dPrefix_inr_inr (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R)
    (w : Fin np → R) (i : Fin n) (l : Fin n') (t : ℕ) {b b' : Fin B × Fin L × Fin n'} (h : b ≠ b') :
    pairDiff (dPrefix pc vq w i l t) (.inr b) (.inr b' : DBits n L np B n') = 0 := by
  have hd : dPrefix pc vq w i l t = fun z => ∑ t' : Fin B, (if t'.val ≤ t then 1 else 0) * dIncr pc vq w i l t' z := by
    funext z
    simp only [dPrefix, ite_mul, one_mul, zero_mul]
  rw [hd, pairDiff_sum univ (fun t' z => (if t'.val ≤ t then 1 else 0) * dIncr pc vq w i l t' z)]
  refine Finset.sum_eq_zero fun t' _ => ?_
  rw [pairDiff_mul_left, pairDiff_dIncr_inr_inr pc vq w i l t' h, mul_zero]

/-- Differencing consecutive coin blocks undoes the prefix sum. -/
theorem sum_prefix_diff {F : Type} [CommRing F] (T S : Fin B) :
    ∑ t' : Fin B, (if t'.val ≤ T.val then (1 : F) else 0) *
      ((if t' = S then 1 else 0) - (if t'.val = S.val + 1 then 1 else 0)) = if S = T then 1 else 0 := by
  simp only [mul_sub, Finset.sum_sub_distrib, mul_ite, mul_one, mul_zero]
  rw [Finset.sum_ite_eq']
  have h2 : (∑ t' : Fin B, if t'.val = S.val + 1 then (if t'.val ≤ T.val then (1 : F) else 0) else 0) =
      if S.val + 1 ≤ T.val then 1 else 0 := by
    by_cases hS : S.val + 1 ≤ T.val
    · rw [Finset.sum_eq_single ⟨S.val + 1, by omega⟩]
      · simp [hS]
      · intro t' _ ht'
        have : t'.val ≠ S.val + 1 := fun h => ht' (Fin.ext h)
        simp [this]
      · simp
    · simp only [hS, ↓reduceIte]
      refine Finset.sum_eq_zero fun t' _ => ?_
      by_cases h : t'.val = S.val + 1
      · have : ¬t'.val ≤ T.val := by omega
        simp only [h, hS, ↓reduceIte]
      · simp only [h, ↓reduceIte]
  rw [h2]
  simp only [Finset.mem_univ, ↓reduceIte]
  by_cases hST : S = T
  · subst hST
    simp
  · have : S.val ≠ T.val := fun h => hST (Fin.ext h)
    by_cases h1 : S.val ≤ T.val
    · have h1' : S.val + 1 ≤ T.val := by omega
      simp [h1, h1', hST]
    · have h1' : ¬S.val + 1 ≤ T.val := by omega
      simp [h1, h1', hST]

/-- **A row's prefix words' stacked cross block** (RS-E-REF r4 IV.2, r5 V.1's row scope): with a plane whose weight
maps to 1, prefix words of row `i` at distinct (column, block) pairs `cl` have a flattened cross block of rank at
least `L` a word. On the rows `β_p₀(i, j)` it is `[j = j′][l = l′][t′ ≤ t]`, and differencing consecutive coin blocks
per column (`Q`) turns it into the identity. -/
theorem rank_cross_dRow {F : Type} [Field F] (φ : R →+* F) (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) {p₀ : Fin np} (hw : φ (w p₀) = 1) (i : Fin n) {τ : Type}
    [Fintype τ] [DecidableEq τ] (cl : τ → Fin n' × Fin B) (hcl : Function.Injective cl) :
    L * Fintype.card τ ≤ (Matrix.of fun (x : τ × (Fin n × Fin L × Fin np)) (v : Fin B × Fin L × Fin n') =>
      φ (pairDiff (dPrefix pc vq w i (cl x.1).1 (cl x.1).2.val) (.inl x.2) (.inr v))).rank := by
  set M := Matrix.of fun (x : τ × (Fin n × Fin L × Fin np)) (v : Fin B × Fin L × Fin n') =>
    φ (pairDiff (dPrefix pc vq w i (cl x.1).1 (cl x.1).2.val) (.inl x.2) (.inr v)) with hMdef
  let ρ : τ × Fin L → τ × (Fin n × Fin L × Fin np) := fun y => (y.1, (i, y.2, p₀))
  let Q : Matrix (Fin B × Fin L × Fin n') (τ × Fin L) F := fun v y =>
    if v.2 = (y.2, (cl y.1).1) then
      ((if v.1 = (cl y.1).2 then 1 else 0) - (if v.1.val = (cl y.1).2.val + 1 then 1 else 0)) else 0
  have hM : ∀ y v, M.submatrix ρ id y v =
      if v.2 = (y.2, (cl y.1).1) ∧ v.1.val ≤ (cl y.1).2.val then 1 else 0 := by
    intro y v
    simp only [Matrix.submatrix_apply, id, hMdef, Matrix.of_apply, ρ, pairDiff_dPrefix_inl_inr, true_and]
    split_ifs <;> simp [hw]
  have hMQ : M.submatrix ρ id * Q = 1 := by
    ext y y'
    rw [Matrix.mul_apply, Matrix.one_apply]
    simp_rw [hM]
    rw [Fintype.sum_prod_type]
    have hin : ∀ t' : Fin B, (∑ v2 : Fin L × Fin n',
        (if (v2 = (y.2, (cl y.1).1) ∧ t'.val ≤ (cl y.1).2.val) then (1 : F) else 0) * Q (t', v2) y') =
          (if (y.2, (cl y.1).1) = (y'.2, (cl y'.1).1) then 1 else 0) *
            ((if t'.val ≤ (cl y.1).2.val then (1 : F) else 0) *
              ((if t' = (cl y'.1).2 then 1 else 0) - (if t'.val = (cl y'.1).2.val + 1 then 1 else 0))) := by
      intro t'
      rw [Finset.sum_eq_single (y.2, (cl y.1).1)]
      · simp only [Q, true_and]
        by_cases hj : (y.2, (cl y.1).1) = (y'.2, (cl y'.1).1) <;>
          by_cases ht : t'.val ≤ (cl y.1).2.val <;> simp [hj, ht]
      · intro v2 _ hv2
        simp [hv2]
      · simp
    simp only [hin]
    rw [← Finset.mul_sum, sum_prefix_diff]
    by_cases hyy : y = y'
    · subst hyy
      simp
    · simp only [hyy, ↓reduceIte]
      by_cases hj : (y.2, (cl y.1).1) = (y'.2, (cl y'.1).1)
      · have h2 : (cl y'.1).2 ≠ (cl y.1).2 := fun h =>
          hyy (Prod.ext (hcl (Prod.ext (congrArg Prod.snd hj) h.symm)) (congrArg Prod.fst hj))
        simp [hj, h2]
      · simp [hj]
  calc L * Fintype.card τ = (1 : Matrix (τ × Fin L) (τ × Fin L) F).rank := by
        rw [Matrix.rank_one, Fintype.card_prod, Fintype.card_fin, mul_comm]
    _ = (M.submatrix ρ id * Q).rank := by rw [hMQ]
    _ ≤ (M.submatrix ρ id).rank := Matrix.rank_mul_le_left _ _
    _ ≤ M.rank := Matrix.rank_submatrix_le M ρ id

/-- **The prefix words' stacked cross block** (RS-E-REF r4 IV.2), one cell: `T` prefix words of cell `(i, l)` at
distinct blocks `tt` have a flattened cross block of rank at least `T L` (`rank_cross_dRow` in one column). -/
theorem rank_cross_dPrefix {F : Type} [Field F] (φ : R →+* F) (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) {p₀ : Fin np} (hw : φ (w p₀) = 1) (i : Fin n) (l : Fin n') {τ : Type} [Fintype τ]
    [DecidableEq τ] (tt : τ → Fin B) (htt : Function.Injective tt) :
    L * Fintype.card τ ≤ (Matrix.of fun (x : τ × (Fin n × Fin L × Fin np)) (v : Fin B × Fin L × Fin n') =>
      φ (pairDiff (dPrefix pc vq w i l (tt x.1).val) (.inl x.2) (.inr v))).rank :=
  rank_cross_dRow φ pc vq w hw i (fun t => (l, tt t)) fun _ _ h => htt (congrArg Prod.snd h)

/-- **The chain lemma** (RS-E-REF r4 IV.2, r5 V.1's row scope): `T` committed prefix words of one row at distinct
(column, block) pairs, each a combination of the products in `S` plus a bit-affine word and right at every bit point
with at most two ones, have `T L` at most the products' summed cross ranks. -/
theorem chain_products_ge {F : Type} [Field F] (φ : R →+* F) (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) {p₀ : Fin np} (hw : φ (w p₀) = 1) (i : Fin n) {ι τ : Type}
    [Fintype τ] [DecidableEq τ] (cl : τ → Fin n' × Fin B) (hcl : Function.Injective cl) (S : Finset ι)
    (π : ι → (DBits n L np B n' → R) → R) (m : ι → ℕ) (hπ : ∀ k ∈ S, CrossRankLe (π k) (m k)) (c : τ → ι → R)
    (r : τ → BitAff (DBits n L np B n') R)
    (hT : ∀ t (U : Finset (DBits n L np B n')), U.card ≤ 2 →
      ∑ k ∈ S, c t k * π k (bitInd U) + (r t).eval (bitInd U) = dPrefix pc vq w i (cl t).1 (cl t).2.val (bitInd U)) :
    L * Fintype.card τ ≤ ∑ k ∈ S, m k :=
  (rank_cross_dRow φ pc vq w hw i cl hcl).trans
    (rank_cross_le (Vl := Fin n × Fin L × Fin np) (Vr := Fin B × Fin L × Fin n') φ S π m hπ c r
      (fun t => dPrefix pc vq w i (cl t).1 (cl t).2.val) fun t U hU => (hT t U hU).symm)

/-- **The chain bound** for a class `Ok` of products at factor `c`, on the re-encoded design: `T` committed prefix
words of one row at distinct (column, block) pairs, each a combination of products in `Ok` plus a bit-affine word,
and each right at every bit point with at most two ones, use at least `T L / c` products. -/
def ChainBound (R : Type) [CommRing R] (n L np B n' : ℕ) (w : Fin np → R)
    (Ok : ((DBits n L np B n' → R) → R) → Prop) (c : ℕ) : Prop :=
  ∀ (ι τ : Type) [Fintype τ] [DecidableEq τ] (S : Finset ι) (π : ι → (DBits n L np B n' → R) → R),
    (∀ k ∈ S, Ok (π k)) → ∀ (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (i : Fin n)
    (cl : τ → Fin n' × Fin B), Function.Injective cl → ∀ (coef : τ → ι → R)
    (r : τ → BitAff (DBits n L np B n') R),
    (∀ t (U : Finset (DBits n L np B n')), U.card ≤ 2 →
      ∑ k ∈ S, coef t k * π k (bitInd U) + (r t).eval (bitInd U) =
        dPrefix pc vq w i (cl t).1 (cl t).2.val (bitInd U)) →
    L * Fintype.card τ ≤ c * S.card

/-- **The chain lemma for one-sided products** (RS-E-REF r4 IV.2): with a plane whose weight maps to 1, `T`
committed prefix words of a row need `T L` one-sided products of any form, in any arithmetic with a map to a field:
`F₂` for s32 wrap (`toF2`) or exact integer sums, the field itself for exact arithmetic over a field. -/
theorem chainBound_oneSided {F : Type} [Field F] (φ : R →+* F) {w : Fin np → R} {p₀ : Fin np}
    (hw : φ (w p₀) = 1) : ChainBound R n L np B n' w OneSided 1 := by
  intro ι τ _ _ S π hπ pc vq i cl hcl coef r hT
  have := chain_products_ge φ pc vq w hw i cl hcl S π (fun _ => 1) (fun k hk => (hπ k hk).crossRankLe) coef r hT
  simpa using this

/-- With affine mixed products allowed, the cross block alone gives factor 2: each counts twice. -/
theorem chainBound_two {F : Type} [Field F] (φ : R →+* F) {w : Fin np → R} {p₀ : Fin np}
    (hw : φ (w p₀) = 1) : ChainBound R n L np B n' w (fun π => OneSided π ∨ AffProd π) 2 := by
  classical
  intro ι τ _ _ S π hπ pc vq i cl hcl coef r hT
  have h := chain_products_ge φ pc vq w hw i cl hcl S π (fun k => if OneSided (π k) then 1 else 2)
    (fun k hk => by
      by_cases h1 : OneSided (π k)
      · simpa [h1] using h1.crossRankLe
      · simpa [h1] using ((hπ k hk).resolve_left h1).crossRankLe) coef r hT
  refine h.trans ?_
  calc ∑ k ∈ S, (if OneSided (π k) then 1 else 2) ≤ ∑ _k ∈ S, 2 :=
        Finset.sum_le_sum fun k _ => by split_ifs <;> omega
    _ = 2 * S.card := by rw [Finset.sum_const, smul_eq_mul, mul_comm]

/-- A combination of products plus a bit-affine word has, on a pair of distinct bits, the products' combined
second differences. -/
theorem pairDiff_comb {ι : Type} (S : Finset ι) (π : ι → (DBits n L np B n' → R) → R) (c : ι → R)
    (A : BitAff (DBits n L np B n') R) (T : (DBits n L np B n' → R) → R)
    (hT : ∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → ∑ k ∈ S, c k * π k (bitInd U) + A.eval (bitInd U) = T (bitInd U))
    {u v : DBits n L np B n'} (huv : u ≠ v) : pairDiff T u v = ∑ k ∈ S, c k * pairDiff (π k) u v := by
  have h1 : pairDiff T u v = pairDiff (fun z => ∑ k ∈ S, c k * π k z + A.eval z) u v := by
    simp only [pairDiff]
    rw [← hT {u, v} ((Finset.card_insert_le _ _).trans (by simp)), ← hT {u} (by simp), ← hT {v} (by simp),
      ← hT ∅ (by simp)]
  rw [h1, pairDiff_add (fun z => ∑ k ∈ S, c k * π k z) (fun z => A.eval z), pairDiff_aff _ huv, add_zero,
    pairDiff_sum S (fun k z => c k * π k z)]
  exact Finset.sum_congr rfl fun k _ => pairDiff_mul_left _ _ _ _

/-- **The chain lemma for mixed products** (RS-E-REF r5 V.1, Lemma G), over any field of characteristic 2 the ring
maps to (`F₂` for s32 wrap, `toF2`): `T` committed prefix words of a row need `T L` products whose operands are any
function of U's bits plus an affine function of the bits (`GProd`), mixed bytes included. The words' cross blocks are
the products' alternating forms (`pairDiff_mul_flat`), whose coin–coin blocks vanish with the targets'
(`pairDiff_dPrefix_inr_inr`), so Lemma G bounds their rank by the products' count, and `rank_cross_dRow` gives
`T L`. Over `ℚ` the factor fails (`not_chainBound_affProd_rat`). -/
theorem chainBound_gProd {F : Type} [Field F] (φ : R →+* F) (h2 : (2 : F) = 0) {w : Fin np → R} {p₀ : Fin np}
    (hw : φ (w p₀) = 1) : ChainBound R n L np B n' w GProd 1 := by
  classical
  intro ι τ _ _ S π hπ pc vq i cl hcl coef r hT
  have hπ' : ∀ k, ∃ o o' : (DBits n L np B n' → R) → R, k ∈ S → UAffB o ∧ UAffB o' ∧ ∀ z, π k z = o z * o' z :=
    fun k => by
      by_cases hk : k ∈ S
      · obtain ⟨o, o', h⟩ := hπ k hk
        exact ⟨o, o', fun _ => h⟩
      · exact ⟨fun _ => 0, fun _ => 0, fun h => absurd h hk⟩
  choose o o' hoo using hπ'
  let a : ι → DBits n L np B n' → F := fun k u => φ (fdiff (o k) u)
  let b : ι → DBits n L np B n' → F := fun k u => φ (fdiff (o' k) u)
  let c : τ → ι → F := fun t k => φ (coef t k)
  have hflat : ∀ t {u v : DBits n L np B n'}, u ≠ v → (∀ k ∈ S, pairDiff (o k) u v = 0) →
      (∀ k ∈ S, pairDiff (o' k) u v = 0) →
      φ (pairDiff (dPrefix pc vq w i (cl t).1 (cl t).2.val) u v) = altForm S a b c t u v := by
    intro t u v huv h1 h1'
    rw [pairDiff_comb S π (coef t) (r t) _ (hT t) huv, map_sum]
    refine Finset.sum_congr rfl fun k hk => ?_
    have e : pairDiff (π k) u v = pairDiff (fun z => o k z * o' k z) u v := by
      simp only [pairDiff, (hoo k hk).2.2]
    rw [e, pairDiff_mul_flat (h1 k hk) (h1' k hk)]
    simp only [a, b, c, map_mul, map_add]
  have hXY : ∀ t (x : Fin n × Fin L × Fin np) (y : Fin B × Fin L × Fin n'),
      φ (pairDiff (dPrefix pc vq w i (cl t).1 (cl t).2.val) (.inl x) (.inr y)) =
        altForm S a b c t (.inl x) (.inr y) := fun t x y =>
    hflat t Sum.inl_ne_inr (fun k hk => (hoo k hk).1.flat_inl_inr x y) fun k hk => (hoo k hk).2.1.flat_inl_inr x y
  have hYY : ∀ t (y y' : Fin B × Fin L × Fin n'), y ≠ y' →
      altForm S a b c t (.inr y) (.inr y' : DBits n L np B n') = 0 := fun t y y' hyy => by
    rw [← hflat t (fun h => hyy (Sum.inr_injective h)) (fun k hk => (hoo k hk).1.flat_inr_inr hyy)
      fun k hk => (hoo k hk).2.1.flat_inr_inr hyy, pairDiff_dPrefix_inr_inr pc vq w i _ _ hyy, map_zero]
  have hrank := rank_cross_dRow φ pc vq w hw i cl hcl
  simp_rw [hXY] at hrank
  rw [one_mul]
  exact hrank.trans (lemmaG h2 S a b c hYY)

end ChainDesign

/-- Winograd's mixed product over `ℚ` at `L = 2`, first operand: `(x₀ + x₁)/2 − y₀ + y₁`. -/
def winogradA : BitAff (DBits 1 2 1 1 1) ℚ :=
  ⟨0, Sum.elim (fun a => ![1 / 2, 1 / 2] a.2.1) fun b => ![-1, 1] b.2.1⟩

/-- Winograd's mixed product over `ℚ` at `L = 2`, second operand: `(x₁ − x₀)/2 + y₀ + y₁`. -/
def winogradB : BitAff (DBits 1 2 1 1 1) ℚ :=
  ⟨0, Sum.elim (fun a => ![-1 / 2, 1 / 2] a.2.1) fun b => ![1, 1] b.2.1⟩

/-- The affine correction to Winograd's product: `(x₀ − x₁)/4 + y₀ − y₁`. -/
def winogradR : BitAff (DBits 1 2 1 1 1) ℚ :=
  ⟨0, Sum.elim (fun a => ![1 / 4, -1 / 4] a.2.1) fun b => ![1, -1] b.2.1⟩

/-- Over `ℚ`, one mixed affine product plus an affine word is the prefix word `x₀ y₀ + x₁ y₁` of `L = 2` at every
bit point. -/
theorem winograd_rat_word (z : DBits 1 2 1 1 1 → ℚ) (hz : ∀ v, z v * z v = z v) :
    winogradA.eval z * winogradB.eval z + winogradR.eval z = dPrefix 0 0 (fun _ => 1) 0 0 0 z := by
  have hx0 := hz (.inl (0, 0, 0))
  have hx1 := hz (.inl (0, 1, 0))
  have hy0 := hz (.inr (0, 0, 0))
  have hy1 := hz (.inr (0, 1, 0))
  simp only [winogradA, winogradB, winogradR, BitAff.eval, dPrefix, dIncr, dP, dX, Fintype.sum_sum_type,
    Fintype.sum_prod_type, Fin.sum_univ_two, Fin.sum_univ_one, Sum.elim_inl, Sum.elim_inr, Matrix.cons_val_zero,
    Matrix.cons_val_one, Fin.isValue, Pi.zero_apply]
  simp
  linear_combination (1 / 4 : ℚ) * hx1 - (1 / 4 : ℚ) * hx0 + hy1 - hy0

/-- **Lemma G needs characteristic 2**: over `ℚ` the chain bound fails at factor 1 for affine products, since one
mixed product commits a word of `L = 2` (`winograd_rat_word`). -/
theorem not_chainBound_affProd_rat : ¬ChainBound ℚ 1 2 1 1 1 (fun _ => 1) AffProd 1 := by
  intro h
  have := h Unit Unit {()} (fun _ z => winogradA.eval z * winogradB.eval z)
    (fun _ _ => ⟨winogradA, winogradB, fun _ => rfl⟩) 0 0 0 (fun _ => (0, 0)) (fun _ _ _ => rfl) (fun _ _ => 1)
    (fun _ => winogradR) fun _ U _ => by
      rw [Finset.sum_singleton, one_mul]
      exact winograd_rat_word _ fun v => by unfold bitInd; split_ifs <;> norm_num
  simp at this

namespace SlpG

variable {V R : Type} [Fintype V] [CommRing R] {N : ℕ}

/-- Each slot's bit-affine part. -/
def aff (P : SlpG V R N) : Fin N → BitAff V R := P.toSlp.aff

/-- **The closed form**: every slot is a combination of the program's products plus a bit-affine word. -/
theorem val_eq (P : SlpG V R N) (z : V → R) (j : Fin N) :
    P.val z j = ∑ i, P.coef j i * P.prod i z + (P.aff j).eval z := by
  obtain ⟨n, hn⟩ := j
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    rw [val, coef, aff, Slp.coef, Slp.aff_eval]
    have hP : P.toSlp.isProd = P.isProd := rfl
    have hL : P.toSlp.lin = P.lin := rfl
    have hO : P.toSlp.off = P.off := rfl
    rw [hP, hL, hO]
    by_cases hp : P.isProd ⟨n, hn⟩ = true
    · simp only [hp, ↓reduceIte, Pi.single_apply, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq',
        Finset.mem_univ, add_zero]
    · simp only [hp, Bool.false_eq_true, ↓reduceIte]
      have hk : ∀ k : Fin n, P.val z ⟨k.val, k.isLt.trans hn⟩ =
          ∑ i, P.toSlp.coef ⟨k.val, k.isLt.trans hn⟩ i * P.prod i z +
            (P.toSlp.aff ⟨k.val, k.isLt.trans hn⟩).eval z := fun k => ih k.val k.isLt _
      simp only [hk, mul_add, Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_apply, Pi.smul_apply,
        smul_eq_mul, Finset.sum_mul, mul_assoc]
      rw [Finset.sum_comm (s := (Finset.univ : Finset (Fin n))) (t := (Finset.univ : Finset (Fin N)))]
      ring

omit [Fintype V] in
/-- Only products carry coefficients. -/
theorem isProd_of_coef_ne_zero (P : SlpG V R N) (j i : Fin N) (h : P.coef j i ≠ 0) : P.isProd i = true :=
  P.toSlp.isProd_of_coef_ne_zero j i h

end SlpG

section ChainRound

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

open Classical in
/-- **Row-pure runs add up** (the chain lemma, per round; RS-E-REF r5 V.1's row scope): if certified prefix words
of distinct rows share no product, and the products are in a class with chain factor `c`, the program has at least
`L / c` products per certified cell-block. Sharing among a row's cells, along their prefix chains included, is
free. -/
theorem SlpG.cells_card_le {w : Fin np → R} {Ok : ((DBits n L np B n' → R) → R) → Prop} {c : ℕ}
    (hc : ChainBound R n L np B n' w Ok c) {N : ℕ} (P : SlpG (DBits n L np B n') R N)
    (hOk : ∀ j, P.isProd j = true → Ok (P.prod j)) (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (word : DCells n n' B → Fin N) (cor : Finset (DCells n n' B))
    (hT : ∀ e ∈ cor, ∀ U : Finset (DBits n L np B n'), U.card ≤ 2 →
      P.val (bitInd U) (word e) = dPrefix pc vq w e.1 e.2.1 e.2.2.val (bitInd U))
    (hpure : ∀ e ∈ cor, ∀ e' ∈ cor, e.1 ≠ e'.1 → ∀ k,
      P.coef (word e) k ≠ 0 → P.coef (word e') k = 0) :
    L * cor.card ≤ c * (univ.filter fun j => P.isProd j = true).card := by
  let row : DCells n n' B → Fin n := fun e => e.1
  let fib : Fin n → Finset (DCells n n' B) := fun q => cor.filter fun e => row e = q
  let used : Fin n → Finset (Fin N) := fun q =>
    univ.filter fun k => ∃ e ∈ fib q, P.coef (word e) k ≠ 0
  have hq : ∀ q, L * (fib q).card ≤ c * (used q).card := by
    intro q
    have := hc (Fin N) (fib q) (used q) P.prod
      (fun k hk => by
        obtain ⟨e, -, he⟩ := (Finset.mem_filter.mp hk).2
        exact hOk k (P.isProd_of_coef_ne_zero _ _ he))
      pc vq q (fun e => e.1.2)
      (fun e e' h => by
        have h1 := (Finset.mem_filter.mp e.2).2
        have h2 := (Finset.mem_filter.mp e'.2).2
        apply Subtype.ext
        exact Prod.ext (h1.trans h2.symm) h)
      (fun e k => P.coef (word e.1) k) (fun e => P.aff (word e.1))
      (fun e U hU => by
        have h1 := Finset.mem_filter.mp e.2
        have hq1 : e.1.1 = q := h1.2
        have hTe := hT e.1 h1.1 U hU
        rw [hq1] at hTe
        rw [← hTe, P.val_eq]
        congr 1
        refine (Finset.sum_subset (Finset.subset_univ (used q))
          (f := fun k => P.coef (word e.1) k * P.prod k (bitInd U)) fun k _ hk => ?_)
        have : P.coef (word e.1) k = 0 := by
          by_contra hne
          exact hk (Finset.mem_filter.mpr ⟨Finset.mem_univ _, e.1, e.2, hne⟩)
        rw [this, zero_mul])
    simpa [Fintype.card_coe] using this
  have hdisj : ((cor.image row : Finset _) : Set (Fin n)).PairwiseDisjoint used := by
    intro q _ q' _ hqq'
    refine Finset.disjoint_left.mpr fun k hk hk' => ?_
    obtain ⟨e, he, hek⟩ := (Finset.mem_filter.mp hk).2
    obtain ⟨e', he', hek'⟩ := (Finset.mem_filter.mp hk').2
    have h1 := Finset.mem_filter.mp he
    have h2 := Finset.mem_filter.mp he'
    exact hek' (hpure e h1.1 e' h2.1 (fun h => hqq' (h1.2.symm.trans (h.trans h2.2))) k hek)
  calc L * cor.card = L * ∑ q ∈ cor.image row, (fib q).card := by
        rw [Finset.card_eq_sum_card_image row cor]
    _ = ∑ q ∈ cor.image row, L * (fib q).card := by rw [Finset.mul_sum]
    _ ≤ ∑ q ∈ cor.image row, c * (used q).card := Finset.sum_le_sum fun q _ => hq q
    _ = c * ((cor.image row).biUnion used).card := by rw [Finset.card_biUnion hdisj, Finset.mul_sum]
    _ ≤ c * (univ.filter fun j => P.isProd j = true).card := by
        refine Nat.mul_le_mul_left _ (Finset.card_le_card fun k hk => ?_)
        obtain ⟨q, -, hk⟩ := Finset.mem_biUnion.mp hk
        obtain ⟨e, -, hek⟩ := (Finset.mem_filter.mp hk).2
        exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, P.isProd_of_coef_ne_zero _ _ hek⟩

end ChainRound

/-! ### The luck lemma (RS-E-REF r4 IV.4)

Theorem S_G's per-round hypothesis `PathBound p` asks that a round of deficiency `d` pass with probability at most
`1 − (1 − p) d`. In a spot check the verifier draws a cell and the round's bits uniformly, and the check passes where
the drawn cell's error (its committed word minus the target) vanishes. An error of degree at most 2 on the cube
(`CubeQuad`) that is not zero is nonzero on at least a quarter of the cube, in any ring (`luck_quad`): if a pair
difference is nonzero, one corner of each of its squares is; else the error is affine, and if a first difference is
nonzero, one end of each of its edges is; else it is constant. So spot windows with degree-2 errors satisfy
`PathBound (3/4)`, the deficiency the share of uncertified cells (`RoundGame.Spot`, `RoundGame.passBound_spot`,
`RoundGame.pathBound_spot`). The design's targets have degree 2 (`cubeQuad_dPrefix`), and so have the errors of runs
whose products have affine operands, one-sided or mixed (`SlpG.cubeQuad_err`), on U's 7 uniform planes: the
re-encoded alphabet is covered at ¾. A prover's play is a tree of per-round error tables (`SpotPlay`), each fixed
before its round's coins; its game (`SpotPlay.game`) is a spot-check window whenever its errors have degree 2
(`SpotPlay.game_spot`), and `RoundsHypKS`, the window's rounds over that game, gives `RoundsHypK` at `p = ¾` for
runs with affine operands (`roundsHypK_of_spot`).

The ¾ is tight in that deficiency: dropping one plane's term leaves the error `w_p β_p x`, which vanishes on three
quarters of the cube (`luck_quad_tight`). It fails for one-sided operands of any form, whose errors can be wrong at
one point only (`luck_oneSided_point`). On the integer alphabet (`U ∈ ±1, …, ±64`, never 0) dropping a whole term is
right with probability about ½: an error affine in the coins vanishes with probability at most
`½ + ½ Pr[its coin coefficients vanish]` (`luck_coins`), and a dropped term's coefficient `c + c′ U`, `c′` a unit,
vanishes on at most one letter, so `½ + 1/(2|α|)` (`luck_dropped`). The design's letters `U = −64 + Σ_p w_p β_p` are
128 distinct values, none 0 (`uRe_injective`, `uRe_ne_zero`), so a dropped term is right with probability at most
`½ + 2⁻⁸ = 0.5039` (`luck_dropped_reEncoded`), and `½` if dropped outright (`luck_dropped_reEncoded_zero`). Those
are per-term bounds: a window-level ½ is false at `|S| ≥ 4` (RS-E-REF r5 V.2), and the luck premise is stated given
the drawn pair indices at `p = ¾` (decision 39), which the spot check gives. For one-sided operands of any form the
luck is the parity lemma's `0.854`, in the product deficit rather than this one (`roundsHyp_of_stagedP`,
`roundsHyp_of_overlapP`). -/

section Luck

variable {V R : Type} [Fintype V] [DecidableEq V] [CommRing R]

omit [Fintype V] in
theorem CubeQuad.add {f g : (V → R) → R} (hf : CubeQuad f) (hg : CubeQuad g) :
    CubeQuad fun z => f z + g z := by
  intro S u v huv hu hv
  rw [pairDiff_add, ← hf S u v huv hu hv, ← hg S u v huv hu hv]
  ring

omit [Fintype V] in
theorem CubeQuad.sub {f g : (V → R) → R} (hf : CubeQuad f) (hg : CubeQuad g) :
    CubeQuad fun z => f z - g z := by
  intro S u v huv hu hv
  have h1 := hf S u v huv hu hv
  have h2 := hg S u v huv hu hv
  simp only [pairDiff] at h1 h2 ⊢
  linear_combination h1 - h2

omit [Fintype V] in
theorem CubeQuad.mul_left (k : R) {f : (V → R) → R} (hf : CubeQuad f) : CubeQuad fun z => k * f z := by
  intro S u v huv hu hv
  have h := hf S u v huv hu hv
  simp only [pairDiff] at h ⊢
  linear_combination k * h

omit [Fintype V] in
theorem CubeQuad.sum {ι : Type} (s : Finset ι) {f : ι → (V → R) → R} (hf : ∀ i ∈ s, CubeQuad (f i)) :
    CubeQuad fun z => ∑ i ∈ s, f i z := by
  intro S u v huv hu hv
  rw [pairDiff_sum]
  simp only [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i hi => hf i hi S u v huv hu hv

omit [Fintype V] in
theorem CubeQuad.congr {f g : (V → R) → R} (h : ∀ z, f z = g z) (hf : CubeQuad f) : CubeQuad g := by
  have : f = g := funext h
  exact this ▸ hf

omit [Fintype V] in
theorem cubeQuad_zero : CubeQuad fun _ : V → R => (0 : R) := by
  intro S u v _ _ _
  simp [pairDiff]

theorem cubeQuad_aff (A : BitAff V R) : CubeQuad fun z => A.eval z := by
  intro S u v huv hu hv
  have hu' : u ∉ insert v S := by simp [huv, hu]
  rw [pairDiff_aff A huv]
  simp only [BitAff.eval_bitInd, Finset.sum_insert hu', Finset.sum_insert hu, Finset.sum_insert hv]
  ring

theorem cubeQuad_mul (A B : BitAff V R) : CubeQuad fun z => A.eval z * B.eval z := by
  intro S u v huv hu hv
  have hu' : u ∉ insert v S := by simp [huv, hu]
  rw [pairDiff_mul A B huv]
  simp only [BitAff.eval_bitInd, Finset.sum_insert hu', Finset.sum_insert hu, Finset.sum_insert hv]
  ring

omit [Fintype V] in
/-- A degree-2 function's first difference along `u` at `S` is its first difference at `0` plus `u`'s pair
differences with `S`. -/
theorem CubeQuad.firstDiff {f : (V → R) → R} (hf : CubeQuad f) (u : V) (S : Finset V) (hu : u ∉ S) :
    f (bitInd (insert u S)) - f (bitInd S) = f (bitInd {u}) - f (bitInd ∅) + ∑ v ∈ S, pairDiff f u v := by
  induction S using Finset.induction_on with
  | empty => simp
  | insert v S hvS ih =>
    have huv : u ≠ v := fun h => hu (h ▸ Finset.mem_insert_self _ _)
    have huS : u ∉ S := fun h => hu (Finset.mem_insert_of_mem h)
    have h := hf S u v huv huS hvS
    rw [Finset.sum_insert hvS]
    linear_combination h + ih huS

open Classical in
/-- **The luck lemma for degree 2**, in any ring: a function of degree at most 2 on the cube that is not zero is
nonzero on at least a quarter of the cube. -/
theorem luck_quad {f : (V → R) → R} (hf : CubeQuad f) (hne : ∃ S : Finset V, f (bitInd S) ≠ 0) :
    Fintype.card (Finset V) ≤ 4 * (univ.filter fun S : Finset V => f (bitInd S) ≠ 0).card := by
  set Nz := univ.filter fun S : Finset V => f (bitInd S) ≠ 0 with hNz
  rw [Fintype.card_finset]
  by_cases hA : ∃ u v, u ≠ v ∧ pairDiff f u v ≠ 0
  · obtain ⟨u, v, huv, hpd⟩ := hA
    have hsurj : ((univ \ {u, v}).powerset).card ≤ Nz.card := by
      refine Finset.card_le_card_of_surjOn (fun T => (T.erase u).erase v) fun S hS => ?_
      have hS' : S ⊆ univ \ {u, v} := Finset.mem_powerset.mp hS
      have hu : u ∉ S := fun h => by simpa using hS' h
      have hv : v ∉ S := fun h => by simpa using hS' h
      have hq := hf S u v huv hu hv
      by_contra hno
      have hz : ∀ T, (T.erase u).erase v = S → f (bitInd T) = 0 := fun T hT => by
        by_contra hT'
        exact hno ⟨T, by simpa [hNz] using hT', hT⟩
      have e1 := hz S (by rw [Finset.erase_eq_of_notMem hu, Finset.erase_eq_of_notMem hv])
      have e2 := hz (insert u S) (by rw [Finset.erase_insert hu, Finset.erase_eq_of_notMem hv])
      have e3 := hz (insert v S) (by rw [Finset.erase_insert_of_ne (Ne.symm huv), Finset.erase_eq_of_notMem hu,
        Finset.erase_insert hv])
      have e4 := hz (insert u (insert v S)) (by
        rw [Finset.erase_insert (by simp [huv, hu]), Finset.erase_insert hv])
      rw [e1, e2, e3, e4] at hq
      exact hpd (by rw [← hq]; ring)
    rw [Finset.card_powerset, Finset.card_sdiff, Finset.inter_univ, Finset.card_univ, Finset.card_pair huv]
      at hsurj
    have h2 : 2 ≤ Fintype.card V := by
      have := Finset.card_le_univ ({u, v} : Finset V)
      rwa [Finset.card_pair huv] at this
    calc 2 ^ Fintype.card V = 4 * 2 ^ (Fintype.card V - 2) := by
          rw [show (4 : ℕ) = 2 ^ 2 by norm_num, ← pow_add]
          congr 1
          omega
      _ ≤ 4 * Nz.card := Nat.mul_le_mul_left _ hsurj
  · push Not at hA
    by_cases hB : ∃ u, f (bitInd {u}) - f (bitInd ∅) ≠ 0
    · obtain ⟨u, hu1⟩ := hB
      have hsurj : ((univ.erase u).powerset).card ≤ Nz.card := by
        refine Finset.card_le_card_of_surjOn (fun T => T.erase u) fun S hS => ?_
        have hu : u ∉ S := fun h => by simpa using Finset.mem_powerset.mp hS h
        have hd := hf.firstDiff u S hu
        rw [Finset.sum_eq_zero fun v hv => hA u v (fun h => hu (h ▸ hv)), add_zero] at hd
        by_cases h0 : f (bitInd S) = 0
        · refine ⟨insert u S, ?_, Finset.erase_insert hu⟩
          simp only [hNz, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq]
          intro h
          rw [h, h0, sub_zero] at hd
          exact hu1 hd.symm
        · exact ⟨S, by simpa [hNz] using h0, Finset.erase_eq_of_notMem hu⟩
      rw [Finset.card_powerset, Finset.card_erase_of_mem (Finset.mem_univ u), Finset.card_univ] at hsurj
      have h1 : 1 ≤ Fintype.card V := Fintype.card_pos_iff.mpr ⟨u⟩
      calc 2 ^ Fintype.card V = 2 * 2 ^ (Fintype.card V - 1) := by
            rw [← pow_succ']
            congr 1
            omega
        _ ≤ 4 * Nz.card := by omega
    · push Not at hB
      have hc : ∀ S : Finset V, f (bitInd S) = f (bitInd ∅) := by
        intro S
        induction S using Finset.induction_on with
        | empty => rfl
        | insert a S haS ih =>
          have hd := hf.firstDiff a S haS
          rw [Finset.sum_eq_zero fun v hv => hA a v (fun h => haS (h ▸ hv)), add_zero, hB a] at hd
          rw [← ih]
          exact sub_eq_zero.mp hd
      obtain ⟨S₀, hS₀⟩ := hne
      have hall : Nz = univ := by
        refine Finset.eq_univ_of_forall fun S => ?_
        simp only [hNz, Finset.mem_filter, Finset.mem_univ, true_and]
        rw [hc S]
        rw [hc S₀] at hS₀
        exact hS₀
      rw [hall, Finset.card_univ, Fintype.card_finset]
      omega

/-- ... as a probability: a nonzero degree-2 error vanishes at uniform bits with probability at most `¾`. -/
theorem prob_zero_le_of_cubeQuad {f : (V → R) → R} (hf : CubeQuad f) (hne : ∃ S : Finset V, f (bitInd S) ≠ 0) :
    prob (fun S : Finset V => f (bitInd S) = 0) ≤ 3 / 4 := by
  classical
  have h := luck_quad hf hne
  have hsum := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Finset V)))
    (fun S : Finset V => f (bitInd S) = 0)
  rw [prob_eq_expect, ← prob_eq_expect]
  unfold prob
  have hpos : (0 : ℝ) < Fintype.card (Finset V) := by exact_mod_cast Fintype.card_pos
  rw [div_le_iff₀ hpos]
  have h' : ((univ.filter fun S : Finset V => f (bitInd S) = 0).card : ℝ) +
      ((univ.filter fun S : Finset V => ¬f (bitInd S) = 0).card : ℝ) = Fintype.card (Finset V) := by
    rw [← Finset.card_univ]; exact_mod_cast hsum
  have h'' : (Fintype.card (Finset V) : ℝ) ≤
      4 * ((univ.filter fun S : Finset V => ¬f (bitInd S) = 0).card : ℝ) := by
    exact_mod_cast h
  convert (by linarith : ((univ.filter fun S : Finset V => f (bitInd S) = 0).card : ℝ) ≤
    3 / 4 * (Fintype.card (Finset V) : ℝ)) using 2

end Luck

theorem expect_affine_le {Ω : Type*} [Fintype Ω] {c : ℝ} (hc : 0 ≤ c) (k : ℝ) (f : Ω → ℝ) :
    expect (fun ω => c + k * f ω) ≤ c + k * expect f := by
  unfold expect
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
  · simp [h0, hc]
  · have hC : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast h0
    apply le_of_eq
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, add_div,
      mul_div_cancel_left₀ _ hC.ne', mul_div_assoc]

section Spot

variable {C V R : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing R]

/-- **The spot check's pass bound**: with degree-2 errors, a uniform cell and uniform bits find a zero with
probability at most `1 − ¼ d`, `d` the share of uncertified cells. -/
theorem prob_spot_le (e : C → (V → R) → R) (he : ∀ c, CubeQuad (e c)) :
    prob (fun a : C × Finset V => e a.1 (bitInd a.2) = 0) ≤ 1 - (1 - 3 / 4) * spotDef e := by
  classical
  rw [prob_eq_expect, expect_prod, spotDef, prob_eq_expect]
  have hpt : ∀ c : C, expect (fun S : Finset V => if e (c, S).1 (bitInd (c, S).2) = 0 then (1 : ℝ) else 0) ≤
      1 + -(1 - 3 / 4) * (if ∃ U : Finset V, U.card ≤ 2 ∧ e c (bitInd U) ≠ 0 then 1 else 0) := by
    intro c
    split_ifs with hb
    · obtain ⟨U, -, hU⟩ := hb
      rw [← prob_eq_expect (fun S : Finset V => e c (bitInd S) = 0)]
      have := prob_zero_le_of_cubeQuad (he c) ⟨U, hU⟩
      linarith
    · rw [mul_zero, add_zero]
      exact expect_le_of_le zero_le_one fun S => by split_ifs <;> norm_num
  have := (expect_mono hpt).trans (expect_affine_le zero_le_one _ _)
  linarith

/-- **A spot-check window** (the shape of the re-encoded design's rounds): each round the prover's errors `e c`
(its committed word minus the target, per cell, as functions of the round's bits) are fixed before the round's
coins and of degree 2 (`CubeQuad`); the round's deficiency is the share of uncertified cells (`spotDef`); the
verifier draws a cell and the bits uniformly, and the check passes only where the drawn cell's error vanishes. -/
def RoundGame.Spot (R : Type) [CommRing R] {C V : Type} [Fintype C] [Fintype V] [DecidableEq V] :
    {n : ℕ} → RoundGame (C × Finset V) n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ e : C → (V → R) → R, (∀ c, CubeQuad (e c)) ∧ (∀ a, d a = spotDef e) ∧
      ∀ a, pass a → e a.1 (bitInd a.2) = 0) ∧ ∀ a, (next a).Spot R

/-- **Spot checks with degree-2 errors satisfy Theorem S's pass bound at `p = ¾`.** -/
theorem RoundGame.passBound_spot : ∀ {n : ℕ} (G : RoundGame (C × Finset V) n), G.Spot R → G.PassBound (3 / 4)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    obtain ⟨⟨e, he, hd, hpass⟩, h2⟩ := h
    exact ⟨⟨spotDef e, hd, (prob_mono hpass).trans (prob_spot_le e he)⟩,
      fun a => RoundGame.passBound_spot (next a) (h2 a)⟩

/-- **`PathBound` at `p = ¾` for spot checks with degree-2 errors** (RS-E-REF r4 IV.4: the ¾ needs affine
operands). -/
theorem RoundGame.pathBound_spot {n : ℕ} (G : RoundGame (C × Finset V) n) (h : G.Spot R) : G.PathBound (3 / 4) :=
  G.pathBound_of_passBound (G.passBound_spot h)

end Spot

section LuckDesign

variable {R : Type} [CommRing R] {n L np B n' : ℕ}

theorem cubeQuad_dIncr (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : Fin B) : CubeQuad (dIncr pc vq w i l t) := by
  unfold dIncr
  exact CubeQuad.sum univ fun j _ => cubeQuad_mul _ _

/-- The design's targets, the committed prefix words, are of degree 2 on the round's bits. -/
theorem cubeQuad_dPrefix (pc : Fin n → Fin B → Fin L → R) (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R)
    (i : Fin n) (l : Fin n') (t : ℕ) : CubeQuad (dPrefix pc vq w i l t) := by
  unfold dPrefix
  refine CubeQuad.sum univ fun t' _ => ?_
  by_cases h : t'.val ≤ t
  · simp only [h, ↓reduceIte]
    exact cubeQuad_dIncr pc vq w i l t'
  · simp only [h, ↓reduceIte]
    exact cubeQuad_zero

/-- A program whose products have affine operands computes words of degree 2. -/
theorem SlpG.cubeQuad_val {Vl Vr : Type} [Fintype Vl] [Fintype Vr] [DecidableEq Vl] [DecidableEq Vr] {N : ℕ}
    (P : SlpG (Vl ⊕ Vr) R N) (hOk : ∀ j, P.isProd j = true → AffProd (P.prod j)) (j : Fin N) :
    CubeQuad fun z => P.val z j := by
  refine CubeQuad.congr (fun z => (P.val_eq z j).symm)
    ((CubeQuad.sum univ fun i _ => ?_).add (cubeQuad_aff _))
  by_cases hc : P.coef j i = 0
  · exact cubeQuad_zero.congr fun z => by rw [hc, zero_mul]
  · obtain ⟨A, B, hAB⟩ := hOk i (P.isProd_of_coef_ne_zero j i hc)
    exact ((cubeQuad_mul A B).congr fun z => (hAB z).symm).mul_left _

/-- **The re-encoded design's errors are of degree 2** when the products have affine operands (one-sided or
mixed): a word minus its committed prefix target. With `RoundGame.pathBound_spot`, a window of spot checks of such
runs satisfies `PathBound (3/4)` on the design's alphabet, U's 7 uniform planes. -/
theorem SlpG.cubeQuad_err {N : ℕ} (P : SlpG (DBits n L np B n') R N)
    (hOk : ∀ j, P.isProd j = true → AffProd (P.prod j)) (j : Fin N) (pc : Fin n → Fin B → Fin L → R)
    (vq : Fin B → Fin L → Fin n' → R) (w : Fin np → R) (i : Fin n) (l : Fin n') (t : ℕ) :
    CubeQuad fun z => P.val z j - dPrefix pc vq w i l t z :=
  (P.cubeQuad_val hOk j).sub (cubeQuad_dPrefix pc vq w i l t)

/-- **The ¾ is tight for degree 2**: on one U bit and one coin, the error `β x` (one plane's term dropped)
vanishes on three quarters of the cube. So in the uncertified-word deficiency no `p < ¾` holds, on the
re-encoded alphabet too: dropping one plane's term leaves the error `w_p β_p x`. -/
theorem luck_quad_tight :
    CubeQuad (fun z : Unit ⊕ Unit → ℤ => z (.inl ()) * z (.inr ())) ∧
      prob (fun S : Finset (Unit ⊕ Unit) => (bitInd S (.inl ()) : ℤ) * bitInd S (.inr ()) = 0) = 3 / 4 := by
  refine ⟨(cubeQuad_mul (⟨0, Sum.elim 1 0⟩ : BitAff (Unit ⊕ Unit) ℤ) ⟨0, Sum.elim 0 1⟩).congr fun z => ?_, ?_⟩
  · simp [BitAff.eval, Fintype.sum_sum_type]
  · have h3 : (univ.filter fun S : Finset (Unit ⊕ Unit) =>
        (bitInd S (.inl ()) : ℤ) * bitInd S (.inr ()) = 0).card = 3 := by decide
    have h4 : Fintype.card (Finset (Unit ⊕ Unit)) = 4 := by decide
    unfold prob
    rw [h4]
    convert (show ((3 : ℕ) : ℝ) / ((4 : ℕ) : ℝ) = 3 / 4 by norm_num) using 3
    convert h3

/-- **One-sided products of any form are not of degree 2, and the ¾ fails for them**: the product of the
indicators that no U bit and no coin is set is one-sided, is wrong at the all-zero point (so its word is
uncertified) and right everywhere else, so it passes with probability `1 − 2^{−|V|}`. For such runs the deficiency
is the product deficit (`parDef`), where the parity luck holds at `0.854` (`StagedPlay.game_stagedP`). -/
theorem luck_oneSided_point {Vl Vr : Type} [Fintype Vl] [Fintype Vr] [DecidableEq Vl] [DecidableEq Vr] :
    OneSided (fun z : Vl ⊕ Vr → ℤ =>
        (if ∀ a, z (.inl a) = 0 then 1 else 0) * (if ∀ b, z (.inr b) = 0 then 1 else 0)) ∧
      ((if ∀ a, (bitInd ∅ : Vl ⊕ Vr → ℤ) (.inl a) = 0 then 1 else 0) *
        (if ∀ b, (bitInd ∅ : Vl ⊕ Vr → ℤ) (.inr b) = 0 then 1 else 0) : ℤ) ≠ 0 ∧
      ∀ S : Finset (Vl ⊕ Vr), S.Nonempty →
        ((if ∀ a, (bitInd S : Vl ⊕ Vr → ℤ) (.inl a) = 0 then 1 else 0) *
          (if ∀ b, (bitInd S : Vl ⊕ Vr → ℤ) (.inr b) = 0 then 1 else 0) : ℤ) = 0 := by
  refine ⟨⟨fun z => if ∀ a, z (.inl a) = 0 then 1 else 0, fun z => if ∀ b, z (.inr b) = 0 then 1 else 0,
    fun z z' h => by simp only [h], fun z z' h => by simp only [h], fun z => rfl⟩, by simp [bitInd], ?_⟩
  rintro S ⟨x, hx⟩
  cases x with
  | inl a =>
    have : ¬∀ a', (bitInd S : Vl ⊕ Vr → ℤ) (.inl a') = 0 := fun h => by simpa [bitInd, hx] using h a
    simp only [this, ↓reduceIte, zero_mul]
  | inr b =>
    have : ¬∀ b', (bitInd S : Vl ⊕ Vr → ℤ) (.inr b') = 0 := fun h => by simpa [bitInd, hx] using h b
    simp only [this, ↓reduceIte, mul_zero]

/-- **The luck of a coin-affine error** (RS-E-REF r4 IV.4, the integer alphabet): if, given the U bits `a`, the
error is affine in the coins, `f₀ a + Σ_{v ∈ S} f v a`, it vanishes at uniform `(a, S)` with probability at most
`½ + ½ Pr_a[f v a = 0 for every coin v]`. -/
theorem luck_coins {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] (f₀ : α → R) (f : V → α → R) :
    prob (fun x : α × Finset V => f₀ x.1 + ∑ v ∈ x.2, f v x.1 = 0) ≤
      1 / 2 + 1 / 2 * prob fun a => ∀ v, f v a = 0 := by
  classical
  rw [prob_eq_expect, expect_prod, prob_eq_expect (fun a => ∀ v, f v a = 0)]
  have hpt : ∀ a : α, expect (fun S : Finset V => if f₀ (a, S).1 + ∑ v ∈ (a, S).2, f v (a, S).1 = 0 then (1 : ℝ)
      else 0) ≤ 1 / 2 + 1 / 2 * (if ∀ v, f v a = 0 then 1 else 0) := by
    intro a
    split_ifs with hall
    · exact expect_le_of_le (by norm_num) fun S => by split_ifs <;> norm_num
    · push Not at hall
      obtain ⟨v, hv⟩ := hall
      rw [mul_zero, add_zero, ← prob_eq_expect (fun S : Finset V => f₀ a + ∑ u ∈ S, f u a = 0)]
      let tg : Finset V → Finset V := fun S => if v ∈ S then S.erase v else insert v S
      have hinv : Function.Involutive tg := fun S => by
        by_cases h : v ∈ S
        · simp [tg, h, Finset.insert_erase h]
        · simp [tg, h, Finset.erase_insert h]
      have hmaps : ∀ S, f₀ a + ∑ u ∈ S, f u a = 0 → f₀ a + ∑ u ∈ tg S, f u a ≠ 0 := fun S hS => by
        by_cases h : v ∈ S
        · have e := Finset.add_sum_erase S (fun u => f u a) h
          simp only [tg, h, ↓reduceIte]
          intro h0
          exact hv (by linear_combination e + hS - h0)
        · simp only [tg, h, ↓reduceIte, Finset.sum_insert h]
          intro h0
          exact hv (by linear_combination h0 - hS)
      have hle : (univ.filter fun S : Finset V => f₀ a + ∑ u ∈ S, f u a = 0).card ≤
          (univ.filter fun S : Finset V => ¬f₀ a + ∑ u ∈ S, f u a = 0).card :=
        Finset.card_le_card_of_injOn tg (fun S hS => by
          simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hS ⊢
          exact hmaps S hS) (hinv.injective.injOn)
      have hsum := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Finset V)))
        (fun S : Finset V => f₀ a + ∑ u ∈ S, f u a = 0)
      unfold prob
      have hpos : (0 : ℝ) < Fintype.card (Finset V) := by exact_mod_cast Fintype.card_pos
      rw [div_le_iff₀ hpos]
      have h' : ((univ.filter fun S : Finset V => f₀ a + ∑ u ∈ S, f u a = 0).card : ℝ) +
          ((univ.filter fun S : Finset V => ¬f₀ a + ∑ u ∈ S, f u a = 0).card : ℝ) = Fintype.card (Finset V) := by
        rw [← Finset.card_univ]; exact_mod_cast hsum
      have hle' : ((univ.filter fun S : Finset V => f₀ a + ∑ u ∈ S, f u a = 0).card : ℝ) ≤
          ((univ.filter fun S : Finset V => ¬f₀ a + ∑ u ∈ S, f u a = 0).card : ℝ) := by exact_mod_cast hle
      convert (by linarith : ((univ.filter fun S : Finset V => f₀ a + ∑ u ∈ S, f u a = 0).card : ℝ) ≤
        1 / 2 * (Fintype.card (Finset V) : ℝ)) using 2
  exact (expect_mono hpt).trans (expect_affine_le (by norm_num) _ _)

/-- A dropped term's coin coefficient `c + c' U(a)`, with `c'` a unit and the alphabet's `U` injective, vanishes for
at most one letter: probability at most `1 / |α|`. -/
theorem prob_dropped_le {α : Type} [Fintype α] (U : α → R) (hU : Function.Injective U) (c c' : R)
    (hc' : IsUnit c') : prob (fun a => c + c' * U a = 0) ≤ 1 / Fintype.card α := by
  classical
  unfold prob
  have h1 : (univ.filter fun a => c + c' * U a = 0).card ≤ 1 :=
    Finset.card_le_one.mpr fun a ha b hb => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
      exact hU (hc'.mul_left_cancel (by linear_combination ha - hb))
  exact div_le_div_of_nonneg_right (by exact_mod_cast h1) (by positivity)

/-- **The luck on the re-encoded alphabet** (RS-E-REF r4 IV.4): a coin-affine error one of whose coin
coefficients is a dropped term `c + c' U(a)` (`c'` a unit, `U` injective on the alphabet `α`) vanishes with
probability at most `½ + 1 / (2 |α|)`. On U's 128 letters `±1, …, ±64` that is `½ + 2⁻⁸ = 0.5039`, RS-E-REF r4's
best found (`luck_dropped_reEncoded`). -/
theorem luck_dropped {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] (f₀ : α → R) (f : V → α → R)
    (U : α → R) (hU : Function.Injective U) (v₀ : V) (c c' : R) (hc' : IsUnit c') (hv₀ : ∀ a, f v₀ a = c + c' * U a) :
    prob (fun x : α × Finset V => f₀ x.1 + ∑ v ∈ x.2, f v x.1 = 0) ≤ 1 / 2 + 1 / (2 * Fintype.card α) := by
  have h1 := luck_coins f₀ f
  have h2 : prob (fun a => ∀ v, f v a = 0) ≤ prob fun a => c + c' * U a = 0 :=
    prob_mono fun a ha => (hv₀ a) ▸ ha v₀
  have h3 := prob_dropped_le U hU c c' hc'
  have e : (1 : ℝ) / (2 * Fintype.card α) = 1 / 2 * (1 / Fintype.card α) := by
    rw [one_div_mul_one_div]
  rw [e]
  linarith

/-- The re-encoded design's plane weights over `ℕ`. -/
def wReN : Fin 7 → ℕ := ![1, 2, 4, 8, 16, 32, 65]

/-- An entry's letter from its 7 planes, over `ℕ` and shifted by 64: `Σ_{p ∈ a} w_p ∈ {0, …, 63} ∪ {65, …, 128}`. -/
def uReN (a : Finset (Fin 7)) : ℕ := ∑ p ∈ a, wReN p

/-- Reading the planes back from `uReN`: plane 6 is set iff the value is at least 65, the rest are binary. -/
def uReDec (m : ℕ) : Finset (Fin 7) :=
  if 65 ≤ m then insert 6 (univ.filter fun p : Fin 7 => p.val < 6 ∧ (m - 65).testBit p.val)
  else univ.filter fun p : Fin 7 => p.val < 6 ∧ m.testBit p.val

theorem uReDec_uReN : ∀ a : Finset (Fin 7), uReDec (uReN a) = a := by decide +kernel

theorem uReN_le (a : Finset (Fin 7)) : uReN a ≤ 128 := by
  calc uReN a ≤ ∑ p : Fin 7, wReN p := Finset.sum_le_sum_of_subset (Finset.subset_univ a)
    _ = 128 := by simp [wReN, Fin.sum_univ_succ]

/-- **The re-encoded design's letter** `U = −64 + Σ_p w_p β_p` of an entry's 7 planes, over `Z/2³²`. -/
def uRe (a : Finset (Fin 7)) : ZMod (2 ^ 32) := -64 + ∑ p ∈ a, wRe p

theorem uRe_eq (a : Finset (Fin 7)) : uRe a = -64 + (uReN a : ZMod (2 ^ 32)) := by
  rw [uRe, uReN, Nat.cast_sum]
  congr 1
  refine Finset.sum_congr rfl fun p _ => ?_
  fin_cases p <;> rfl

/-- The 128 plane patterns give 128 distinct letters. -/
theorem uRe_injective : Function.Injective uRe := by
  intro a b h
  rw [uRe_eq, uRe_eq, add_right_inj] at h
  have hN : uReN a = uReN b := by
    have := congrArg ZMod.val h
    rwa [ZMod.val_natCast, ZMod.val_natCast, Nat.mod_eq_of_lt ((uReN_le a).trans_lt (by norm_num)),
      Nat.mod_eq_of_lt ((uReN_le b).trans_lt (by norm_num))] at this
  rw [← uReDec_uReN a, ← uReDec_uReN b, hN]

/-- `U` is never 0 (`uReN a ≠ 64`). -/
theorem uRe_ne_zero (a : Finset (Fin 7)) : uRe a ≠ 0 := by
  have h64 : uReN a ≠ 64 := by
    have : ∀ a : Finset (Fin 7), uReN a ≠ 64 := by decide +kernel
    exact this a
  rw [uRe_eq]
  intro h
  have h' : (uReN a : ZMod (2 ^ 32)) = ((64 : ℕ) : ZMod (2 ^ 32)) := by
    rw [Nat.cast_ofNat]
    linear_combination h
  have := congrArg ZMod.val h'
  rw [ZMod.val_natCast, ZMod.val_natCast, Nat.mod_eq_of_lt ((uReN_le a).trans_lt (by norm_num)),
    Nat.mod_eq_of_lt (by norm_num)] at this
  exact h64 this

/-- **The luck of a dropped term on the re-encoded alphabet** (RS-E-REF r4 IV.4): an error affine in the coins, given
the dropped entry's planes `a` and the other bits `b`, one of whose coin coefficients is `c b + c′ U(a)` with `c′` a
unit (a whole term `U x` dropped, partly compensated by `c b`), vanishes with probability at most `½ + 2⁻⁸`. -/
theorem luck_dropped_reEncoded {β V : Type} [Fintype β] [Fintype V] [DecidableEq V]
    (f₀ : Finset (Fin 7) × β → ZMod (2 ^ 32)) (f : V → Finset (Fin 7) × β → ZMod (2 ^ 32)) (v₀ : V)
    (c : β → ZMod (2 ^ 32)) (c' : ZMod (2 ^ 32)) (hc' : IsUnit c') (hv₀ : ∀ a b, f v₀ (a, b) = c b + c' * uRe a) :
    prob (fun x : (Finset (Fin 7) × β) × Finset V => f₀ x.1 + ∑ v ∈ x.2, f v x.1 = 0) ≤ 1 / 2 + 1 / 256 := by
  have h1 := luck_coins f₀ f
  have h2 : prob (fun ab : Finset (Fin 7) × β => ∀ v, f v ab = 0) ≤ 1 / 128 := by
    refine (prob_mono fun ab hab => (hv₀ ab.1 ab.2) ▸ hab v₀).trans ?_
    refine prob_le_of_slices (fun ab : Finset (Fin 7) × β => c ab.2 + c' * uRe ab.1 = 0) (by norm_num)
      fun b => ?_
    have := prob_dropped_le uRe uRe_injective (c b) c' hc'
    rwa [Fintype.card_finset, Fintype.card_fin] at this
  linarith

/-- ... and `½` when the term is dropped outright (`c = 0`): `c′ U` is never 0, since `U` never is. -/
theorem luck_dropped_reEncoded_zero {β V : Type} [Fintype β] [Fintype V] [DecidableEq V]
    (f₀ : Finset (Fin 7) × β → ZMod (2 ^ 32)) (f : V → Finset (Fin 7) × β → ZMod (2 ^ 32)) (v₀ : V)
    (c' : ZMod (2 ^ 32)) (hc' : IsUnit c') (hv₀ : ∀ a b, f v₀ (a, b) = c' * uRe a) :
    prob (fun x : (Finset (Fin 7) × β) × Finset V => f₀ x.1 + ∑ v ∈ x.2, f v x.1 = 0) ≤ 1 / 2 := by
  have h1 := luck_coins f₀ f
  have hF : ∀ ab : Finset (Fin 7) × β, ¬∀ v, f v ab = 0 := fun ab hab =>
    (hc'.mul_right_eq_zero.not.mpr (uRe_ne_zero ab.1)) ((hv₀ ab.1 ab.2) ▸ hab v₀)
  have h2 : prob (fun ab : Finset (Fin 7) × β => ∀ v, f v ab = 0) ≤ 0 := by
    unfold prob
    simp [hF]
  linarith

end LuckDesign

theorem prob_eq_card {Ω : Type*} [Fintype Ω] (p : Ω → Prop) [DecidablePred p] :
    prob p = ((univ.filter p).card : ℝ) / Fintype.card Ω := by
  rw [prob_eq_expect]
  unfold expect
  rw [Finset.sum_boole]

/-- The path that plays `a` every round. -/
def Paths.const {Ω : Type} (a : Ω) : ∀ n, Paths Ω n
  | 0 => ()
  | n + 1 => (a, Paths.const a n)

/-- **A prover's spot-check play** over `n` rounds: per round, its error table (its committed word minus the target,
per cell, as a function of the round's bits), fixed before the round's coins, then the rest of the play, which may
depend on them. -/
inductive SpotPlay (C V A : Type) : ℕ → Type
  | done : SpotPlay C V A 0
  | round {n : ℕ} (e : C → (V → A) → A) (next : C × Finset V → SpotPlay C V A n) : SpotPlay C V A (n + 1)

namespace SpotPlay

variable {C V A : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing A]

/-- **The window's game** of a play: round `w`'s deficiency is the share of uncertified cells (`spotDef`), and its
check passes where the drawn cell's error vanishes at the drawn bits. -/
noncomputable def game : {n : ℕ} → SpotPlay C V A n → RoundGame (C × Finset V) n
  | _, .done => .done
  | _, .round e next => .round (fun _ => spotDef e) (fun a => e a.1 (bitInd a.2) = 0) fun a => (next a).game

omit [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] in
/-- Round `w`'s error table along the realized coins. -/
def errAt : {n : ℕ} → SpotPlay C V A n → Paths (C × Finset V) n → Fin n → C → (V → A) → A
  | _, .done, _, w => w.elim0
  | _, .round e next, x, w =>
    Fin.cases (motive := fun _ => C → (V → A) → A) e (fun w' => (next x.1).errAt x.2 w') w

omit [Fintype V] in
theorem dAt_game : ∀ {n : ℕ} (P : SpotPlay C V A n) (x : Paths (C × Finset V) n) (w : Fin n),
    P.game.dAt x w = spotDef (P.errAt x w)
  | _, .done, _, w => w.elim0
  | _, .round e next, x, w => by
    induction w using Fin.cases with
    | zero => simp [game, errAt, RoundGame.dAt]
    | succ w' =>
      simp only [game, errAt, RoundGame.dAt, Fin.cases_succ]
      exact dAt_game (next x.1) x.2 w'

/-- Degree-2 errors along every play make the game a spot-check window. -/
theorem game_spot : ∀ {n : ℕ} (P : SpotPlay C V A n), (∀ x w c, CubeQuad (P.errAt x w c)) → P.game.Spot A
  | _, .done, _ => trivial
  | _, .round e next, h => by
    simp only [game]
    refine ⟨⟨e, fun c => ?_, fun _ => rfl, fun a ha => ha⟩, fun a => game_spot (next a) fun x w c => ?_⟩
    · simpa [errAt] using h (Paths.const (c, ∅) _) 0 c
    · simpa [errAt] using h (a, x) w.succ c

end SpotPlay

/-- A round's spot coins on the re-encoded design: a cell-block and the round's bits. -/
abbrev DSpot (n L np B n' : ℕ) : Type := DCells n n' B × Finset (DBits n L np B n')

open Classical in
/-- The certified cells, those whose error is zero at every bit point with at most two ones, are at least the
share `1 − d` of the cells. -/
theorem card_certified_ge {C V A : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing A]
    (e : C → (V → A) → A) :
    (Fintype.card C : ℝ) * (1 - spotDef e) ≤
      (univ.filter fun c => ∀ U : Finset V, U.card ≤ 2 → e c (bitInd U) = 0).card := by
  classical
  rw [spotDef, prob_eq_card]
  have hsplit := Finset.card_filter_add_card_filter_not (s := (univ : Finset C))
    (fun c => ∃ U : Finset V, U.card ≤ 2 ∧ e c (bitInd U) ≠ 0)
  have hnot : (univ.filter fun c => ¬∃ U : Finset V, U.card ≤ 2 ∧ e c (bitInd U) ≠ 0) =
      univ.filter fun c => ∀ U : Finset V, U.card ≤ 2 → e c (bitInd U) = 0 :=
    Finset.filter_congr fun c _ => by simp only [not_exists, not_and, not_not]
  rw [hnot, Finset.card_univ] at hsplit
  rcases Nat.eq_zero_or_pos (Fintype.card C) with h0 | h0
  · rw [h0]
    simp
  · have hpos : (0 : ℝ) < Fintype.card C := by exact_mod_cast h0
    have hs : ((univ.filter fun c => ∃ U : Finset V, U.card ≤ 2 ∧ e c (bitInd U) ≠ 0).card : ℝ) +
        ((univ.filter fun c => ∀ U : Finset V, U.card ≤ 2 → e c (bitInd U) = 0).card : ℝ) = Fintype.card C := by
      exact_mod_cast hsplit
    rw [mul_sub, mul_one, mul_div_cancel₀ _ hpos.ne']
    linarith

namespace StagePlay

variable {C V A : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing A]

omit [Fintype V] in
theorem defi_sgame (M : ℕ) : ∀ {m : ℕ} (P : StagePlay C V A m) (x : Paths (Finset V) m),
    (P.sgame M).defi x = ∑ s, (M : ℝ)⁻¹ * spotDef (P.errAt x s)
  | _, .done, _ => by simp [sgame, StageGame.defi]
  | _, .stage e next, x => by
    rw [Fin.sum_univ_succ]
    simp only [sgame, StageGame.defi, errAt, Fin.cases_zero, Fin.cases_succ]
    rw [defi_sgame M (next x.1) x.2]

omit [Fintype V] in
theorem wrong_sgame (M : ℕ) : ∀ {m : ℕ} (P : StagePlay C V A m) (x : Paths (Finset V) m),
    (P.sgame M).wrong x = ∑ s, (M : ℝ)⁻¹ * prob fun c => ¬P.ok x s c
  | _, .done, _ => by simp [sgame, StageGame.wrong]
  | _, .stage e next, x => by
    rw [Fin.sum_univ_succ]
    simp only [sgame, StageGame.wrong, ok, Fin.cases_zero, Fin.cases_succ]
    rw [wrong_sgame M (next x.1) x.2]

omit [Fintype V] in
/-- **The round's check passes with probability at most `1 − U`**: the check's stage and cell are uniform. -/
theorem prob_ok_le {m : ℕ} (P : StagePlay C V A m) (x : Paths (Finset V) m) :
    prob (fun j : Fin m × C => P.ok x j.1 j.2) ≤ 1 - (P.sgame m).wrong x := by
  classical
  rw [prob_eq_expect, expect_prod, wrong_sgame]
  calc expect (fun s => expect fun c => if P.ok x (s, c).1 (s, c).2 then (1 : ℝ) else 0)
      ≤ expect (fun s => 1 + -1 * prob fun c => ¬P.ok x s c) := expect_mono fun s => by
        rw [← prob_eq_expect (fun c => P.ok x s c)]
        linarith [prob_add_not_le (fun c => P.ok x s c)]
    _ ≤ 1 + -1 * expect (fun s => prob fun c => ¬P.ok x s c) := expect_affine_le zero_le_one _ _
    _ = 1 - ∑ s, (m : ℝ)⁻¹ * prob fun c => ¬P.ok x s c := by
        rw [← Finset.mul_sum]
        unfold expect
        rw [Fintype.card_fin]
        ring

/-- **The stage game is valid at `p = ¾`** when every stage's errors have degree 2 (`prob_spot_le`, per stage). -/
theorem valid_sgame (M : ℕ) : ∀ {m : ℕ} (P : StagePlay C V A m), (∀ x s c, CubeQuad (P.errAt x s c)) →
    (P.sgame M).Valid (3 / 4)
  | _, .done, _ => trivial
  | _, .stage e next, h => by
    classical
    have he : ∀ c, CubeQuad (e c) := fun c => by simpa [errAt] using h (Paths.const ∅ _) 0 c
    refine ⟨mul_nonneg (inv_nonneg.mpr (Nat.cast_nonneg _)) (prob_nonneg _),
      fun S => mul_nonneg (inv_nonneg.mpr (Nat.cast_nonneg _)) (prob_nonneg _), ?_,
      fun S => valid_sgame M (next S) fun x s c => by simpa [errAt] using h (S, x) s.succ c⟩
    rw [expect_mul_left, mul_left_comm]
    refine mul_le_mul_of_nonneg_left ?_ (inv_nonneg.mpr (Nat.cast_nonneg _))
    rcases isEmpty_or_nonempty C with hC | hC
    · have h0 : spotDef e = 0 := by
        unfold spotDef prob
        simp
      rw [h0, mul_zero]
      unfold expect
      exact div_nonneg (Finset.sum_nonneg fun S _ => prob_nonneg _) (Nat.cast_nonneg _)
    · have h1 : expect (fun S : Finset V => prob fun c => e c (bitInd S) ≠ 0) =
          prob fun a : C × Finset V => e a.1 (bitInd a.2) ≠ 0 := by
        rw [prob_eq_expect (fun a : C × Finset V => e a.1 (bitInd a.2) ≠ 0), expect_prod_swap]
        congr 1
        funext S
        exact prob_eq_expect _
      rw [h1]
      have h2 := prob_add_not (fun a : C × Finset V => e a.1 (bitInd a.2) = 0)
      have h3 := prob_spot_le e he
      linarith

end StagePlay

namespace StagedPlay

variable {C V A : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] {M : ℕ}

omit [Fintype V] in
theorem dAt_game : ∀ {n : ℕ} (P : StagedPlay C V A M n) (x : Paths (Paths (Finset V) M × (Fin M × C)) n)
    (w : Fin n), P.game.dAt x w = ∑ s, (M : ℝ)⁻¹ * spotDef (P.errAt x w s)
  | _, .done, _, w => w.elim0
  | _, .round P next, x, w => by
    induction w using Fin.cases with
    | zero =>
      simp only [game, errAt, RoundGame.dAt, Fin.cases_zero]
      exact P.defi_sgame M x.1.1
    | succ w' =>
      simp only [game, errAt, RoundGame.dAt, Fin.cases_succ]
      exact dAt_game (next x.1) x.2 w'

/-- **Degree-2 errors along every play make the game a staged window at `p = ¾`.** -/
theorem game_staged : ∀ {n : ℕ} (P : StagedPlay C V A M n), (∀ x w s c, CubeQuad (P.errAt x w s c)) →
    P.game.Staged (3 / 4) M
  | _, .done, _ => trivial
  | _, .round P next, h => by
    simp only [game]
    refine ⟨⟨P.sgame M, P.valid_sgame M fun x s c => ?_, fun x j => rfl, fun x => P.prob_ok_le x⟩,
      fun a => game_staged (next a) fun x w s c => ?_⟩
    · simpa [errAt] using h ((x, (s, c)), Paths.const (x, (s, c)) _) 0 s c
    · simpa [errAt] using h (a, x) w.succ s c

end StagedPlay

theorem ChainBound.mono {R : Type} [CommRing R] {n L np B n' : ℕ} {w : Fin np → R}
    {Ok Ok' : ((DBits n L np B n' → R) → R) → Prop} {c : ℕ} (h : ChainBound R n L np B n' w Ok c)
    (hOk : ∀ π, Ok' π → Ok π) : ChainBound R n L np B n' w Ok' c :=
  fun ι τ _ _ S π hπ => h ι τ S π fun k hk => hOk _ (hπ k hk)

/-- **The window's rounds with Theorem K's cost line** (the re-encoded design, chain form), a named hypothesis. It is
`RoundsHyp` with the cost line derived rather than assumed. Along every play `x`:
- round `w`'s run is a straight-line program over a ring `Ar` on the round's bits (`prog w`; `Z/2³²` for s32 wrap
  with unscaled reads, `ℚ` for exact arithmetic) whose products are in the class `Ok` (`OneSided`, of any form; or also `AffProd`, mixed operands), on committed
  constants `pc w` (`P − 128`) and `vq w` (`⌊x/s⌋`);
- each cell-block `(i, l, t)` it certifies (`cor w`) has its committed prefix word `T_t(i, l)` (`word w`) right at
  every bit point with at most two ones;
- certified words of distinct rows share no product (row-pure); partners, merges and sharing among a row's cells,
  along their prefix chains included, are free;
- it certifies at least `(1 − d_w) C` of the round's `C = n n' B` cell-blocks, `d_w` the game's round-`w` deficiency;
- the run is charged at least `κ` a product, less the noise reads `V`.

The game's `PathBound p` is derived at `p = ¾` when the game is the spot-check game of runs with affine operands
(`RoundsHypKS`, `roundsHypK_of_spot`). For one-sided operands of any form it fails in this deficiency
(`luck_oneSided_point`) and holds at `p = 0.854` in the product deficit, with charging purity in place of row purity
(`RoundsHypKStP`, `roundsHyp_of_stagedP`; `RoundsHypKOvP`, `roundsHyp_of_overlapP`). -/
def RoundsHypK (Mc : Machine) (S : Service) (Ω : Type) [Fintype Ω] (R : ℕ) (p Vmax δb δz κ : ℝ)
    (n L np B n' : ℕ) {Ar : Type} [CommRing Ar] (wt : Fin np → Ar)
    (Ok : ((DBits n L np B n' → Ar) → Ar) → Prop) : Prop :=
  ∃ e : S.Seed ≃ Paths Ω R, ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
    ∃ (G : RoundGame Ω R) (Bind ZK : Fin R → Paths Ω R → Prop) (V : Paths Ω R → ℝ),
      G.PathBound p ∧ (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (DBits n L np B n') Ar (N w))
        (pc : Fin R → Fin n → Fin B → Fin L → Ar)
        (vq : Fin R → Fin B → Fin L → Fin n' → Ar)
        (word : ∀ w, DCells n n' B → Fin (N w)) (cor : Fin R → Finset (DCells n n' B)),
        (∀ w j, (prog w).isProd j = true → Ok ((prog w).prod j)) ∧
        (∀ w, ∀ c ∈ cor w, ∀ U : Finset (DBits n L np B n'), U.card ≤ 2 →
          (prog w).val (bitInd U) (word w c) = dPrefix (pc w) (vq w) wt c.1 c.2.1 c.2.2.val (bitInd U)) ∧
        (∀ w, ∀ c ∈ cor w, ∀ c' ∈ cor w, c.1 ≠ c'.1 → ∀ k,
          (prog w).coef (word w c) k ≠ 0 → (prog w).coef (word w c') k = 0) ∧
        (∀ w, ((n * n' * B : ℕ) : ℝ) * (1 - G.dAt x w) ≤ (cor w).card) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

/-- **The chain lemma gives the window's cost line**: under `RoundsHypK` for a product class with chain factor
`c`, `RoundsHyp` holds with `F₀ = κ L n n' B / c` and no slack. -/
theorem roundsHyp_of_K {Mc : Machine} {S : Service} {Ω : Type} [Fintype Ω] {R : ℕ} {p Vmax δb δz κ : ℝ}
    {n L np B n' : ℕ} {Ar : Type} [CommRing Ar] {wt : Fin np → Ar}
    {Ok : ((DBits n L np B n' → Ar) → Ar) → Prop} {c : ℕ}
    (h : RoundsHypK Mc S Ω R p Vmax δb δz κ n L np B n' wt Ok) (hc : ChainBound Ar n L np B n' wt Ok c)
    (hc0 : 0 < c) (hκ : 0 ≤ κ) :
    RoundsHyp Mc S Ω R p (κ * L * ((n * n' * B : ℕ) : ℝ) / c) 0 Vmax δb δz := by
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨G, Bind, ZK, V, hG, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  refine ⟨G, Bind, ZK, V, G.expBound_of_pathBound hG, hB, hZ, hacc, hV, fun x => ?_⟩
  obtain ⟨N, prog, pc, vq, word, cor, hOk, hT, hpure, hcor, hcharge⟩ := hK x
  have hcR : (0 : ℝ) < c := by exact_mod_cast hc0
  have hprod : ∀ w, (L : ℝ) / c * (cor w).card ≤ ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) :=
    fun w => by
      have := (prog w).cells_card_le hc (hOk w) (pc w) (vq w) (word w) (cor w) (hT w) (hpure w)
      rw [div_mul_eq_mul_div, div_le_iff₀ hcR, mul_comm _ (c : ℝ)]
      exact_mod_cast this
  set C : ℝ := ((n * n' * B : ℕ) : ℝ)
  have hsum : ((R : ℝ) - G.deficiency x) * (κ * L * C / c) = κ * ∑ w, (L : ℝ) / c * (C * (1 - G.dAt x w)) := by
    rw [G.deficiency_eq_sum, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
    field_simp
  calc ((R : ℝ) - G.deficiency x) * (κ * L * C / c) - V x - R * 0
      = κ * ∑ w, (L : ℝ) / c * (C * (1 - G.dAt x w)) - V x := by rw [hsum]; ring
    _ ≤ κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x := by
      gcongr with w
      exact (mul_le_mul_of_nonneg_left (hcor w) (by positivity)).trans (hprod w)
    _ ≤ Mc.charge M (e.symm x, r) := hcharge

/-- At chain factor 1 (one-sided products, `chainBound_oneSided`; mixed ones, `chainBound_gProd`): `F₀ = κ L n n' B`,
the round's `n × LB × n'` GEMM at `κ` a product (`c = 1`). -/
theorem roundsHyp_of_K_one {Mc : Machine} {S : Service} {Ω : Type} [Fintype Ω] {R : ℕ} {p Vmax δb δz κ : ℝ}
    {n L np B n' : ℕ} {Ar : Type} [CommRing Ar] {wt : Fin np → Ar}
    {Ok : ((DBits n L np B n' → Ar) → Ar) → Prop}
    (h : RoundsHypK Mc S Ω R p Vmax δb δz κ n L np B n' wt Ok) (hc : ChainBound Ar n L np B n' wt Ok 1)
    (hκ : 0 ≤ κ) : RoundsHyp Mc S Ω R p (κ * L * ((n * n' * B : ℕ) : ℝ)) 0 Vmax δb δz := by
  simpa using roundsHyp_of_K h hc one_pos hκ

/-- **The window's rounds as spot checks** (the re-encoded design, chain form, with the luck derived), a named
hypothesis. It is `RoundsHypK` with its game the spot-check game of the prover's play (`SpotPlay.game`: the round's
deficiency is the share of uncertified cell-blocks, and the check passes where the drawn cell-block's error vanishes
at the drawn bits), and the certified cell-blocks those whose error is zero at every bit point with at most two
ones. Along every play `x`, round `w`'s error table is its run's committed prefix words minus their targets. -/
def RoundsHypKS (Mc : Machine) (S : Service) (R : ℕ) (Vmax δb δz κ : ℝ) (n L np B n' : ℕ) {Ar : Type}
    [CommRing Ar] (wt : Fin np → Ar) (Ok : ((DBits n L np B n' → Ar) → Ar) → Prop) : Prop :=
  ∃ e : S.Seed ≃ Paths (DSpot n L np B n') R, ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
    ∃ (P : SpotPlay (DCells n n' B) (DBits n L np B n') Ar R) (Bind ZK : Fin R → Paths (DSpot n L np B n') R → Prop)
      (V : Paths (DSpot n L np B n') R → ℝ),
      (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (DBits n L np B n') Ar (N w))
        (pc : Fin R → Fin n → Fin B → Fin L → Ar)
        (vq : Fin R → Fin B → Fin L → Fin n' → Ar)
        (word : ∀ w, DCells n n' B → Fin (N w)),
        (∀ w j, (prog w).isProd j = true → Ok ((prog w).prod j)) ∧
        (∀ w c, P.errAt x w c =
          fun z => (prog w).val z (word w c) - dPrefix (pc w) (vq w) wt c.1 c.2.1 c.2.2.val z) ∧
        (∀ w c c', (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w c (bitInd U) = 0) →
          (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w c' (bitInd U) = 0) →
          c.1 ≠ c'.1 → ∀ k,
          (prog w).coef (word w c) k ≠ 0 → (prog w).coef (word w c') k = 0) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

/-- **The luck is derived for affine operands**: under `RoundsHypKS` with products of affine operands (one-sided or
mixed), `RoundsHypK` holds at `p = ¾`, its `PathBound` from the spot check's degree-2 errors
(`RoundGame.pathBound_spot`, `SlpG.cubeQuad_err`) and its certified count from the game's deficiency. -/
theorem roundsHypK_of_spot {Mc : Machine} {S : Service} {R : ℕ} {Vmax δb δz κ : ℝ} {n L np B n' : ℕ}
    {Ar : Type} [CommRing Ar] {wt : Fin np → Ar} {Ok : ((DBits n L np B n' → Ar) → Ar) → Prop}
    (h : RoundsHypKS Mc S R Vmax δb δz κ n L np B n' wt Ok) (hOk : ∀ π, Ok π → AffProd π) :
    RoundsHypK Mc S (DSpot n L np B n') R (3 / 4) Vmax δb δz κ n L np B n' wt Ok := by
  classical
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨P, Bind, ZK, V, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  refine ⟨P.game, Bind, ZK, V, P.game.pathBound_spot (P.game_spot fun x w c => ?_), hB, hZ, hacc, hV,
    fun x => ?_⟩
  · obtain ⟨N, prog, pc, vq, word, hOk', hE, -, -⟩ := hK x
    rw [hE w c]
    exact (prog w).cubeQuad_err (fun j hj => hOk _ (hOk' w j hj)) (word w c) (pc w) (vq w) wt c.1 c.2.1 c.2.2.val
  · obtain ⟨N, prog, pc, vq, word, hOk', hE, hpure, hcharge⟩ := hK x
    let cor : Fin R → Finset (DCells n n' B) := fun w =>
      univ.filter fun c => ∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w c (bitInd U) = 0
    refine ⟨N, prog, pc, vq, word, cor, hOk', fun w c hc U hU => ?_, fun w c hc c' hc' hne k hk => ?_,
      fun w => ?_, hcharge⟩
    · have := (Finset.mem_filter.mp hc).2 U hU
      rw [hE w c] at this
      exact sub_eq_zero.mp this
    · exact hpure w c c' (Finset.mem_filter.mp hc).2 (Finset.mem_filter.mp hc').2 hne k hk
    · rw [P.dAt_game x w, spotDef, prob_eq_card]
      have hC : ((n * n' * B : ℕ) : ℝ) = (Fintype.card (DCells n n' B) : ℝ) := by
        simp [Fintype.card_prod, mul_assoc]
      have hsplit := Finset.card_filter_add_card_filter_not (s := (univ : Finset (DCells n n' B)))
        (fun c => ∃ U : Finset (DBits n L np B n'), U.card ≤ 2 ∧ P.errAt x w c (bitInd U) ≠ 0)
      have hnot : (univ.filter fun c => ¬∃ U : Finset (DBits n L np B n'), U.card ≤ 2 ∧
          P.errAt x w c (bitInd U) ≠ 0) = cor w :=
        Finset.filter_congr fun c _ => by simp only [not_exists, not_and, not_not]
      rw [hnot, Finset.card_univ] at hsplit
      rw [hC]
      rcases Nat.eq_zero_or_pos (Fintype.card (DCells n n' B)) with h0 | h0
      · rw [h0]
        simp
      · have hpos : (0 : ℝ) < Fintype.card (DCells n n' B) := by exact_mod_cast h0
        have hs : ((univ.filter fun c => ∃ U : Finset (DBits n L np B n'), U.card ≤ 2 ∧
            P.errAt x w c (bitInd U) ≠ 0).card : ℝ) + ((cor w).card : ℝ) = Fintype.card (DCells n n' B) := by
          exact_mod_cast hsplit
        rw [mul_sub, mul_one, mul_div_cancel₀ _ hpos.ne']
        linarith

/-- **The window's staged rounds** (per-stage draws: decision 36, the stage-coins draft §2,
as RS-E-REF r5 V.3 repaired it; the re-encoded design, chain form, the luck derived), a named hypothesis. Each round
is `St` stages: stage `s` commits its operands, then draws both noise blocks of its products together (keyed by the
draw: the stage's bits), then commits its outputs; one check at the round's end draws a stage and a cell-block. The
window's game is the staged spot-check game of the prover's play (`StagedPlay.game`). Along every play `x`:
- round `w`'s stage `s` is a straight-line program over a ring `Ar` on the stage's bits (`prog w s`) whose products
  are in `Ok`, on committed constants `pc w s` and `vq w s`, and its error table is its committed prefix words
  (`word w s`) minus their targets;
- certified words of distinct rows of a stage share no product (row-pure);
- the run is charged at least `κ` a product over every stage of every round (cost additivity: a stage's interval,
  from its draw to its outputs' commitment, holds no other draw), less the noise reads `V`.

Decode's deferred edit, every operand of the round's products committed before one draw, is the case `St = 1`.
Overlapping stages, and stages of different shapes, are `RoundsHypKOv`. -/
def RoundsHypKSt (Mc : Machine) (S : Service) (R St : ℕ) (Vmax δb δz κ : ℝ) (n L np B n' : ℕ) {Ar : Type}
    [CommRing Ar] (wt : Fin np → Ar) (Ok : ((DBits n L np B n' → Ar) → Ar) → Prop) : Prop :=
  ∃ e : S.Seed ≃ Paths (DStage St n L np B n') R, ∀ (Ω' : Type) [Fintype Ω']
    (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)), (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    ∀ r : Ω', ∃ (P : StagedPlay (DCells n n' B) (DBits n L np B n') Ar St R)
      (Bind ZK : Fin R → Paths (DStage St n L np B n') R → Prop) (V : Paths (DStage St n L np B n') R → ℝ),
      (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → Fin St → ℕ) (prog : ∀ w s, SlpG (DBits n L np B n') Ar (N w s))
        (pc : Fin R → Fin St → Fin n → Fin B → Fin L → Ar)
        (vq : Fin R → Fin St → Fin B → Fin L → Fin n' → Ar)
        (word : ∀ w s, DCells n n' B → Fin (N w s)),
        (∀ w s j, (prog w s).isProd j = true → Ok ((prog w s).prod j)) ∧
        (∀ w s c, P.errAt x w s c =
          fun z => (prog w s).val z (word w s c) - dPrefix (pc w s) (vq w s) wt c.1 c.2.1 c.2.2.val z) ∧
        (∀ w s c c', (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w s c (bitInd U) = 0) →
          (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w s c' (bitInd U) = 0) →
          c.1 ≠ c'.1 → ∀ k, (prog w s).coef (word w s c) k ≠ 0 → (prog w s).coef (word w s c') k = 0) ∧
        κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

/-- `roundsHyp_of_staged` on one slice (`RoundsSlice`): one adversary at one value of its coins, its acceptance
`acc`, charge `cost` and breaks named, and `RoundsHypKSt`'s cost line at that slice. -/
theorem roundsSlice_of_staged {R St : ℕ} {Vmax κ : ℝ} {n L np B n' : ℕ} {Ar : Type} [CommRing Ar]
    {wt : Fin np → Ar} {Ok : ((DBits n L np B n' → Ar) → Ar) → Prop}
    (P : StagedPlay (DCells n n' B) (DBits n L np B n') Ar St R) (acc : Paths (DStage St n L np B n') R → Prop)
    (cost : Paths (DStage St n L np B n') R → ℝ) (Bind ZK : Fin R → Paths (DStage St n L np B n') R → Prop)
    (V : Paths (DStage St n L np B n') R → ℝ) (hacc : ∀ x, acc x → P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x)
    (hV : ∀ x, V x ≤ Vmax)
    (hK : ∀ x, ∃ (N : Fin R → Fin St → ℕ) (prog : ∀ w s, SlpG (DBits n L np B n') Ar (N w s))
        (pc : Fin R → Fin St → Fin n → Fin B → Fin L → Ar)
        (vq : Fin R → Fin St → Fin B → Fin L → Fin n' → Ar)
        (word : ∀ w s, DCells n n' B → Fin (N w s)),
        (∀ w s j, (prog w s).isProd j = true → Ok ((prog w s).prod j)) ∧
        (∀ w s c, P.errAt x w s c =
          fun z => (prog w s).val z (word w s c) - dPrefix (pc w s) (vq w s) wt c.1 c.2.1 c.2.2.val z) ∧
        (∀ w s c c', (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w s c (bitInd U) = 0) →
          (∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w s c' (bitInd U) = 0) →
          c.1 ≠ c'.1 → ∀ k, (prog w s).coef (word w s c) k ≠ 0 → (prog w s).coef (word w s c') k = 0) ∧
        κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x ≤ cost x)
    (hc : ChainBound Ar n L np B n' wt Ok 1) (hOk : ∀ π, Ok π → AffProd π) (hSt : 0 < St) (hκ : 0 ≤ κ) :
    RoundsSlice (3 / 4) (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) 0 Vmax acc cost Bind ZK := by
  classical
  refine ⟨P.game, V, P.game.expBound_of_staged (by norm_num) (P.game_staged fun x w s c => ?_), hacc, hV,
    fun x => ?_⟩
  · obtain ⟨N, prog, pc, vq, word, hOk', hE, -, -⟩ := hK x
    rw [hE w s c]
    exact (prog w s).cubeQuad_err (fun j hj => hOk _ (hOk' w s j hj)) (word w s c) (pc w s) (vq w s) wt c.1 c.2.1
      c.2.2.val
  · obtain ⟨N, prog, pc, vq, word, hOk', hE, hpure, hcharge⟩ := hK x
    let cor : Fin R → Fin St → Finset (DCells n n' B) := fun w s =>
      univ.filter fun c => ∀ U : Finset (DBits n L np B n'), U.card ≤ 2 → P.errAt x w s c (bitInd U) = 0
    have hprod : ∀ w s, (L : ℝ) * (cor w s).card ≤
        ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) := fun w s => by
      have := (prog w s).cells_card_le hc (hOk' w s) (pc w s) (vq w s) (word w s) (cor w s)
        (fun c hc U hU => by
          have := (Finset.mem_filter.mp hc).2 U hU
          rw [hE w s c] at this
          exact sub_eq_zero.mp this)
        fun c hc c' hc' hne k hk =>
          hpure w s c c' (Finset.mem_filter.mp hc).2 (Finset.mem_filter.mp hc').2 hne k hk
      rw [one_mul] at this
      exact_mod_cast this
    have hC : ((n * n' * B : ℕ) : ℝ) = (Fintype.card (DCells n n' B) : ℝ) := by
      simp [Fintype.card_prod, mul_assoc]
    have hcor : ∀ w s, ((n * n' * B : ℕ) : ℝ) * (1 - spotDef (P.errAt x w s)) ≤ (cor w s).card := fun w s => by
      rw [hC]
      convert card_certified_ge (P.errAt x w s)
    set C : ℝ := ((n * n' * B : ℕ) : ℝ)
    have hSt' : (St : ℝ) ≠ 0 := by exact_mod_cast hSt.ne'
    have hdef : P.game.deficiency x = (St : ℝ)⁻¹ * ∑ w, ∑ s, spotDef (P.errAt x w s) := by
      rw [P.game.deficiency_eq_sum, Finset.mul_sum]
      refine Finset.sum_congr rfl fun w _ => ?_
      rw [P.dAt_game x w, Finset.mul_sum]
    have hsum : ((R : ℝ) - P.game.deficiency x) * (St * (κ * L * C)) =
        κ * ∑ w, ∑ s, (L : ℝ) * (C * (1 - spotDef (P.errAt x w s))) := by
      have e2 : ∑ w : Fin R, ∑ s : Fin St, (L : ℝ) * (C * (1 - spotDef (P.errAt x w s))) =
          L * C * (R * St - ∑ w, ∑ s, spotDef (P.errAt x w s)) := by
        simp only [mul_sub, mul_one, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul, ← Finset.mul_sum]
        ring
      rw [hdef, e2]
      field_simp
    calc ((R : ℝ) - P.game.deficiency x) * (St * (κ * L * C)) - V x - R * 0
        = κ * ∑ w, ∑ s, (L : ℝ) * (C * (1 - spotDef (P.errAt x w s))) - V x := by rw [hsum]; ring
      _ ≤ κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x := by
        gcongr with w _ s
        exact (mul_le_mul_of_nonneg_left (hcor w s) (Nat.cast_nonneg L)).trans (hprod w s)
      _ ≤ cost x := hcharge

/-- **Staged rounds give the window's cost line with stage-wise floors summed**: under `RoundsHypKSt` with products
of affine operands and chain factor 1, `RoundsHyp` holds at `p = ¾` with `F₀ = St κ L n n' B`, the round's `St`
stage floors summed, and no slack. The round's deficiency is the mean of its stages' (`StagedPlay.dAt_game`), its
exponential bound comes from the stages' supermartingale (`RoundGame.expBound_of_staged`, `StagedPlay.game_staged`),
and each stage's certified count from its own errors (`roundsSlice_of_staged`, slice by slice). -/
theorem roundsHyp_of_staged {Mc : Machine} {S : Service} {R St : ℕ} {Vmax δb δz κ : ℝ} {n L np B n' : ℕ}
    {Ar : Type} [CommRing Ar] {wt : Fin np → Ar} {Ok : ((DBits n L np B n' → Ar) → Ar) → Prop}
    (h : RoundsHypKSt Mc S R St Vmax δb δz κ n L np B n' wt Ok) (hc : ChainBound Ar n L np B n' wt Ok 1)
    (hOk : ∀ π, Ok π → AffProd π) (hSt : 0 < St) (hκ : 0 ≤ κ) :
    RoundsHyp Mc S (DStage St n L np B n') R (3 / 4) (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) 0 Vmax δb δz := by
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨P, Bind, ZK, V, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  obtain ⟨G, V', hG, hacc', hV', hcost⟩ := roundsSlice_of_staged P
    (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
    (fun x => Mc.charge M (e.symm x, r)) Bind ZK V hacc hV hK hc hOk hSt hκ
  exact ⟨G, Bind, ZK, V', hG, hB, hZ, hacc', hV', hcost⟩

section Overlap

variable {St : ℕ} {ι : Fin St → Type} [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] {R : Type} [CommRing R]

/-- Draw `t`'s bits `z`, every other draw's bits at 0. -/
noncomputable def extZ (t : Fin St) (z : ι t → R) : (Σ t, ι t) → R := Function.extend (Sigma.mk t) z 0

omit [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] in
theorem extZ_mk (t : Fin St) (z : ι t → R) (v : ι t) : extZ t z ⟨t, v⟩ = z v :=
  sigma_mk_injective.extend_apply z 0 v

omit [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] in
theorem extZ_ne (t : Fin St) (z : ι t → R) {t' : Fin St} (h : t' ≠ t) (v : ι t') : extZ t z ⟨t', v⟩ = 0 := by
  rw [extZ, Function.extend_apply']
  · rfl
  · rintro ⟨a, ha⟩
    exact h (congrArg Sigma.fst ha).symm

omit [∀ t, DecidableEq (ι t)] in
theorem sum_extZ_mul (t : Fin St) (z : ι t → R) (g : (Σ t, ι t) → R) :
    ∑ b, g b * extZ t z b = ∑ v, g ⟨t, v⟩ * z v := by
  rw [Fintype.sum_sigma, Finset.sum_eq_single t]
  · exact Finset.sum_congr rfl fun v _ => by rw [extZ_mk]
  · intro t' _ ht
    exact Finset.sum_eq_zero fun v _ => by rw [extZ_ne t z ht, mul_zero]
  · simp

omit [∀ t, Fintype (ι t)] in
theorem mergeAt_mk (U : ∀ s, Finset (ι s)) (t : Fin St) (z : ι t → R) (v : ι t) : mergeAt U t z ⟨t, v⟩ = z v := by
  simp [mergeAt]

omit [∀ t, Fintype (ι t)] in
theorem mergeAt_ne (U : ∀ s, Finset (ι s)) (t : Fin St) (z : ι t → R) {s : Fin St} (h : s ≠ t) (v : ι s) :
    mergeAt U t z ⟨s, v⟩ = bitInd (U s) v := by
  simp [mergeAt, Function.update_of_ne h]

omit [∀ t, Fintype (ι t)] in
theorem mergeAt_self (U : ∀ s, Finset (ι s)) (t : Fin St) :
    mergeAt U t (bitInd (U t)) = (drawInd U : (Σ s, ι s) → R) := by
  funext ⟨s, v⟩
  by_cases h : s = t
  · subst h
    rw [mergeAt_mk]
    rfl
  · rw [mergeAt_ne U t _ h]
    rfl

omit [∀ t, Fintype (ι t)] in
theorem mergeAt_update (U : ∀ s, Finset (ι s)) (t : Fin St) (S : Finset (ι t)) (z : ι t → R) :
    mergeAt (Function.update U t S) t z = mergeAt U t z := by
  funext ⟨s, v⟩
  by_cases h : s = t
  · subst h
    rw [mergeAt_mk, mergeAt_mk]
  · rw [mergeAt_ne _ t _ h, mergeAt_ne U t _ h, Function.update_of_ne h]

omit [∀ t, Fintype (ι t)] in
theorem mergeAt_eq_add (U : ∀ s, Finset (ι s)) (t : Fin St) (z : ι t → R) (b : Σ s, ι s) :
    mergeAt U t z b = mergeAt U t 0 b + extZ t z b := by
  obtain ⟨s, v⟩ := b
  by_cases h : s = t
  · subst h
    rw [mergeAt_mk, mergeAt_mk, extZ_mk, Pi.zero_apply, zero_add]
  · rw [mergeAt_ne U t _ h, mergeAt_ne U t _ h, extZ_ne t z h, add_zero]

/-- A joint bit-affine operand with every other draw's bits at theirs in `U`: an operand on draw `t`'s bits. -/
def BitAff.restrictAt (U : ∀ s, Finset (ι s)) (t : Fin St) (A : BitAff (Σ s, ι s) R) : BitAff (ι t) R :=
  ⟨A.eval (mergeAt U t 0), fun v => A.α ⟨t, v⟩⟩

theorem BitAff.eval_mergeAt (U : ∀ s, Finset (ι s)) (t : Fin St) (A : BitAff (Σ s, ι s) R) (z : ι t → R) :
    A.eval (mergeAt U t z) = (A.restrictAt U t).eval z := by
  simp only [BitAff.eval, BitAff.restrictAt]
  rw [add_assoc, ← sum_extZ_mul t z A.α, ← Finset.sum_add_distrib]
  exact congrArg _ (Finset.sum_congr rfl fun b _ => by rw [mergeAt_eq_add U t z b, mul_add])

/-- A program whose products are K∥'s computes, on draw `t`'s bits at any bit point of the other draws', words of
degree 2. -/
theorem SlpG.cubeQuad_val_mergeAt {N : ℕ} (P : SlpG (Σ s, ι s) R N) (hOk : ∀ j, P.isProd j = true → AffMulAt (P.prod j))
    (U : ∀ s, Finset (ι s)) (t : Fin St) (j : Fin N) : CubeQuad fun z => P.val (mergeAt U t z) j := by
  refine CubeQuad.congr (fun z => (P.val_eq (mergeAt U t z) j).symm)
    ((CubeQuad.sum univ fun i _ => ?_).add ((cubeQuad_aff ((P.aff j).restrictAt U t)).congr fun z =>
      (BitAff.eval_mergeAt U t _ z).symm))
  by_cases hc : P.coef j i = 0
  · exact cubeQuad_zero.congr fun z => by rw [hc, zero_mul]
  · obtain ⟨A, B, hAB⟩ := hOk i (P.isProd_of_coef_ne_zero j i hc) U t
    exact ((cubeQuad_mul A B).congr fun z => (hAB z).symm).mul_left _

end Overlap

section OverlapLuck

variable {St : ℕ} {κ ι : Fin St → Type} [∀ t, Fintype (κ t)] [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)]
  {R : Type} [CommRing R]

omit [∀ t, DecidableEq (ι t)] in
theorem prob_fst {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (p : α → Prop) :
    prob (fun b : α × β => p b.1) = prob p := by
  classical
  rw [prob_eq_expect, expect_prod, prob_eq_expect]
  congr 1
  funext a
  simpa using expect_const_add_mul (Ω := β) (if p a then (1 : ℝ) else 0) 0 fun _ => 0

omit [∀ t, Fintype (κ t)] in
/-- One cell's pass probability with overlapping draws equals the spot check's on its own draw with the other
draws' bits as the cell: swapping draw `t`'s bits for fresh ones is an involution. -/
theorem prob_cell_eq (e : ((Σ s, ι s) → R) → R) (t : Fin St) :
    prob (fun U : ∀ s, Finset (ι s) => e (drawInd U) = 0) =
      prob fun b : (∀ s, Finset (ι s)) × Finset (ι t) => e (mergeAt b.1 t (bitInd b.2)) = 0 := by
  classical
  let φ : (∀ s, Finset (ι s)) × Finset (ι t) → (∀ s, Finset (ι s)) × Finset (ι t) :=
    fun b => (Function.update b.1 t b.2, b.1 t)
  have hφ : Function.Involutive φ := fun b => by
    simp only [φ, Function.update_idem, Function.update_self, Function.update_eq_self]
  rw [prob_equiv hφ.toPerm (fun b : (∀ s, Finset (ι s)) × Finset (ι t) => e (mergeAt b.1 t (bitInd b.2)) = 0),
    ← prob_fst (β := Finset (ι t)) fun U : ∀ s, Finset (ι s) => e (drawInd U) = 0]
  congr 1
  funext b
  have hs : hφ.toPerm.symm b = φ b := rfl
  rw [hs]
  simp only [φ]
  rw [mergeAt_update, ← mergeAt_self b.1 t]

/-- **The spot check's pass bound with overlapping draws** (RS-D1E r2 §2.5): with every cell's error of degree 2 on
its own draw's bits at every bit point of the other draws', a uniform cell and uniform draws find a zero with
probability at most `1 − ¼ d`, `d` the share of cells and draws at which the cell is uncertified. -/
theorem prob_ov_le (e : (Σ t, κ t) → ((Σ s, ι s) → R) → R) (he : ∀ c U, CubeQuad fun z => e c (mergeAt U c.1 z)) :
    prob (fun a : (Σ t, κ t) × (∀ s, Finset (ι s)) => e a.1 (drawInd a.2) = 0) ≤ 1 - (1 - 3 / 4) * ovDef e := by
  classical
  rw [prob_eq_expect, expect_prod, ovDef, prob_eq_expect, expect_prod]
  have hpt : ∀ c : Σ t, κ t,
      expect (fun U : ∀ s, Finset (ι s) => if e (c, U).1 (drawInd (c, U).2) = 0 then (1 : ℝ) else 0) ≤
        1 + -(1 - 3 / 4) * expect fun U : ∀ s, Finset (ι s) =>
          if ∃ S : Finset (ι (c, U).1.1), S.card ≤ 2 ∧ e (c, U).1 (mergeAt (c, U).2 (c, U).1.1 (bitInd S)) ≠ 0
          then (1 : ℝ) else 0 := by
    intro c
    rw [← prob_eq_expect (fun U : ∀ s, Finset (ι s) => e c (drawInd U) = 0), prob_cell_eq (e c) c.1,
      ← prob_eq_expect (fun U : ∀ s, Finset (ι s) =>
        ∃ S : Finset (ι c.1), S.card ≤ 2 ∧ e c (mergeAt U c.1 (bitInd S)) ≠ 0)]
    have := prob_spot_le (fun (U : ∀ s, Finset (ι s)) (z : ι c.1 → R) => e c (mergeAt U c.1 z)) (he c)
    rw [spotDef] at this
    linarith
  have := (expect_mono hpt).trans (expect_affine_le zero_le_one _ _)
  linarith

namespace OvPlay

variable {A : Type} [CommRing A]

theorem dAt_game : ∀ {n : ℕ} (P : OvPlay κ ι A n) (x : Paths ((Σ t, κ t) × (∀ s, Finset (ι s))) n) (w : Fin n),
    P.game.dAt x w = ovDef (P.errAt x w)
  | _, .done, _, w => w.elim0
  | _, .round e next, x, w => by
    induction w using Fin.cases with
    | zero => simp [game, errAt, RoundGame.dAt]
    | succ w' =>
      simp only [game, errAt, RoundGame.dAt, Fin.cases_succ]
      exact dAt_game (next x.1) x.2 w'

/-- Errors of degree 2 on each cell's own draw, at every bit point of the other draws', along every play make the
game meet Theorem S's pass bound at `p = ¾`. -/
theorem passBound_game : ∀ {n : ℕ} (P : OvPlay κ ι A n),
    (∀ x w c U, CubeQuad fun z => P.errAt x w c (mergeAt U c.1 z)) → P.game.PassBound (3 / 4)
  | _, .done, _ => trivial
  | _, .round e next, h => by
    simp only [game]
    refine ⟨⟨ovDef e, fun _ => rfl, prob_ov_le e fun c U => ?_⟩, fun a => passBound_game (next a) fun x w c U => ?_⟩
    · simpa [errAt] using h (Paths.const (c, fun _ => ∅) _) 0 c U
    · simpa [errAt] using h (a, x) w.succ c U

end OvPlay

end OverlapLuck

/-! ### The parity luck (RS-E-REF r6 VI.2)

For one-sided operands of any form the luck is a parity lemma. Mod 2, a cell-block's target is `⟨x, y⟩` plus a
function of `x` and a function of `y`, with `x` the cell's row of U at one odd-weighted plane and `y` the cell-block's
coins. A run whose word for the cell-block has `R` two-sided products is, mod 2 at the cell's frame (every other bit
fixed), `⟨F x, G y⟩ + a x + b y` with `F`, `G`, `a`, `b` of any form. Its agreement `p` has `(2p − 1)² ≤ 2^R / 2^K`
(`parity_corr_sq`, `parity_prob_sq`: Cauchy–Schwarz over `y`, Parseval over `x`). So a cell-block one product short
passes with probability at most `½ + ½ √½ < 0.854` (`parity_luck_short`), and two short at most `¾`
(`parity_luck_two`), over `Z` or `Z/2³²` alike (any ring with a map onto `F₂`). At a cell's frame in the bits
(`ParityForm`) the same bounds hold with every other bit uniform (`prob_zero_le_of_parityForm`).

That luck can't sit in `RoundsHypK`'s `PathBound`, whose deficiency is the share of uncertified cells: no `p < 1`
bounds it for one-sided operands (`luck_oneSided_point`). It sits in the product deficit (RS-E-REF r6 VI.2.1): a
cell-block is short of products (`ParShort`) when its error has a parity form with fewer than `L` terms at its frame,
and a round's product deficit is the share of its cell-blocks short (`parDef`; per draw, `ovDefP`). The spot, staged
and overlapped games in that deficiency (`SpotPlay.gameP`, `StagedPlay.gameP`, `OvPlay.gameP`) accept as their games
do and meet Theorem S's bounds at `0.854` whatever the errors (`passBound_gameP`, `StagedPlay.game_stagedP`). The
floor reads the design: at cell-block `(i, l, t)`'s frame (`dFrame`: row `i` of U at plane `p₀`, the block's coins)
the target is `⟨x, y⟩` plus junk (`split_dPrefix`), one-sided products split (`OneSided.splitProd`), so a one-sided
run's error is a parity form whose terms are the products its word reads that are two-sided there
(`SlpG.parityForm_err`; per draw, `SlpG.parityForm_err_ov`). A cell-block not short is charged at least `L` products,
and with no product charged to two cell-blocks (charging purity, `SlpG.ChargePure`, `SlpG.ChargePureAt`, in place of
row purity) the charges add up (`SlpG.card_full_le`, `SlpG.card_full_le_ov`). So `RoundsHypKStP` and `RoundsHypKOvP`
give `RoundsHyp` at `p = 0.854` with Theorem K's `F₀` (`roundsHyp_of_stagedP`, `roundsHyp_of_overlapP`), and
`D₀ = 274 ln 2` reaches `2⁻⁴⁰` (`certifiedFraction120θ_stagedP_40`, `certifiedFraction120θ_overlapP_40`). -/

section Parity

/-- `(−1)^z` on `F₂`. -/
def sgn (z : ZMod 2) : ℝ := if z = 0 then 1 else -1

theorem sgn_add (a b : ZMod 2) : sgn (a + b) = sgn a * sgn b := by
  have h : ∀ a b : ZMod 2, (a + b = 0 ↔ (a = 0 ↔ b = 0)) := by decide
  unfold sgn
  by_cases ha : a = 0 <;> by_cases hb : b = 0 <;> simp [h, ha, hb]

theorem sgn_mul_self (a : ZMod 2) : sgn a * sgn a = 1 := by
  unfold sgn
  split_ifs <;> norm_num

theorem sgn_sum {ι : Type*} (s : Finset ι) (f : ι → ZMod 2) : sgn (∑ i ∈ s, f i) = ∏ i ∈ s, sgn (f i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [sgn]
  | insert i s hi ih => rw [sum_insert hi, prod_insert hi, sgn_add, ih]

theorem sgn_eq (u v : ZMod 2) : sgn (u + v) = 2 * (if u = v then 1 else 0) - 1 := by
  have h : ∀ u v : ZMod 2, (u + v = 0 ↔ u = v) := by decide
  unfold sgn
  by_cases huv : u = v <;> norm_num [h, huv]

theorem sum_sgn_mul (g : ZMod 2) : ∑ t : ZMod 2, sgn (g * t) = if g = 0 then 2 else 0 := by
  have h2 : ∀ f : ZMod 2 → ℝ, ∑ t : ZMod 2, f t = f 0 + f 1 := fun f => Fin.sum_univ_two f
  rw [h2, mul_zero, mul_one]
  unfold sgn
  by_cases hg : g = 0 <;> norm_num [hg]

theorem sum_sgn_dot {K : ℕ} (γ : Fin K → ZMod 2) :
    ∑ y : Fin K → ZMod 2, sgn (γ ⬝ᵥ y) = if γ = 0 then 2 ^ K else 0 := by
  classical
  simp only [dotProduct, sgn_sum]
  rw [← Fintype.prod_sum (fun i t => sgn (γ i * t))]
  simp only [sum_sgn_mul]
  split_ifs with h
  · simp [h]
  · obtain ⟨i, hi⟩ : ∃ i, γ i ≠ 0 := by
      by_contra hc
      push Not at hc
      exact h (funext hc)
    exact prod_eq_zero (mem_univ i) (by simp [hi])

theorem card_vec (K : ℕ) : Fintype.card (Fin K → ZMod 2) = 2 ^ K := by
  simp

theorem zmod2_add_eq_zero (s t : ZMod 2) : s + t = 0 ↔ t = s := by
  revert s t
  decide

/-- Parseval on `F₂^K`. -/
theorem sum_sq_hat {K : ℕ} (u : (Fin K → ZMod 2) → ℝ) :
    ∑ y : Fin K → ZMod 2, (∑ x, u x * sgn (x ⬝ᵥ y)) ^ 2 = 2 ^ K * ∑ x, u x ^ 2 := by
  classical
  have h1 : ∀ y : Fin K → ZMod 2, (∑ x, u x * sgn (x ⬝ᵥ y)) ^ 2 =
      ∑ x, ∑ x', u x * u x' * sgn ((x + x') ⬝ᵥ y) := fun y => by
    rw [sq, sum_mul_sum]
    refine sum_congr rfl fun x _ => sum_congr rfl fun x' _ => ?_
    rw [add_dotProduct, sgn_add]
    ring
  simp only [h1]
  rw [sum_comm]
  simp only [fun x => sum_comm (s := (univ : Finset (Fin K → ZMod 2))) (t := univ)
    (f := fun y x' => u x * u x' * sgn ((x + x') ⬝ᵥ y))]
  simp only [← mul_sum, sum_sgn_dot]
  rw [mul_sum]
  refine sum_congr rfl fun x _ => ?_
  have hx : ∀ x' : Fin K → ZMod 2, (x + x' = 0) ↔ x' = x := fun x' => by
    constructor
    · intro h
      funext i
      exact (zmod2_add_eq_zero _ _).mp (congrFun h i)
    · rintro rfl
      funext i
      exact (zmod2_add_eq_zero _ _).mpr rfl
  simp only [hx, mul_ite, mul_zero, sum_ite_eq', mem_univ, ↓reduceIte]
  ring

/-- **The parity correlation** (RS-E-REF r6, Lemma P2): for any `F, G : F₂^K → F₂^R` and any `a, b`, the sum of
`(−1)^(⟨F x, G y⟩ + a x + b y + ⟨x, y⟩)` over `x, y` has square at most `2^R (2^K)³`. -/
theorem parity_corr_sq {K R : ℕ} (F G : (Fin K → ZMod 2) → Fin R → ZMod 2) (a b : (Fin K → ZMod 2) → ZMod 2) :
    (∑ x, ∑ y, sgn (F x ⬝ᵥ G y + a x + b y + x ⬝ᵥ y)) ^ 2 ≤ 2 ^ R * (2 ^ K) ^ 3 := by
  classical
  set N : ℝ := 2 ^ K with hN
  let T : (Fin R → ZMod 2) → (Fin K → ZMod 2) → ℝ := fun β y => ∑ x, sgn (a x + F x ⬝ᵥ β) * sgn (x ⬝ᵥ y)
  have hs : ∑ x, ∑ y, sgn (F x ⬝ᵥ G y + a x + b y + x ⬝ᵥ y) = ∑ y, sgn (b y) * T (G y) y := by
    rw [sum_comm]
    refine sum_congr rfl fun y _ => ?_
    simp only [T, mul_sum]
    refine sum_congr rfl fun x _ => ?_
    simp only [sgn_add]
    ring
  have hcs := sum_mul_sq_le_sq_mul_sq univ (fun y => sgn (b y)) (fun y => T (G y) y)
  have hb : ∑ y : Fin K → ZMod 2, sgn (b y) ^ 2 = N := by
    simp only [sq, sgn_mul_self, sum_const, card_univ, card_vec, nsmul_eq_mul, mul_one, hN]
    push_cast
    ring
  have hT : ∑ y : Fin K → ZMod 2, T (G y) y ^ 2 ≤ ∑ y : Fin K → ZMod 2, ∑ β, T β y ^ 2 :=
    sum_le_sum fun y _ => single_le_sum (f := fun β => T β y ^ 2) (fun β _ => sq_nonneg _) (mem_univ (G y))
  have hP : ∑ y : Fin K → ZMod 2, ∑ β, T β y ^ 2 = 2 ^ R * N * N := by
    have hβ : ∀ β, ∑ y, T β y ^ 2 = N * N := fun β => by
      simp only [T]
      rw [sum_sq_hat (fun x => sgn (a x + F x ⬝ᵥ β))]
      simp only [sq, sgn_mul_self, sum_const, card_univ, card_vec, nsmul_eq_mul, mul_one, hN]
      push_cast
      ring
    rw [sum_comm]
    simp only [hβ, sum_const, card_univ, card_vec, nsmul_eq_mul]
    push_cast
    ring
  rw [hs]
  calc (∑ y, sgn (b y) * T (G y) y) ^ 2 ≤ (∑ y, sgn (b y) ^ 2) * ∑ y, T (G y) y ^ 2 := hcs
    _ ≤ N * (2 ^ R * N * N) := by
      rw [hb, ← hP]
      exact mul_le_mul_of_nonneg_left hT (by positivity)
    _ = 2 ^ R * N ^ 3 := by ring

/-- **The parity lemma** (RS-E-REF r6, Lemma P2) in probability: `x, y` uniform on `F₂^K`, any `F, G : F₂^K → F₂^R`
and any `a, b`; the agreement `p` of `⟨F x, G y⟩ + a x + b y` with `⟨x, y⟩` has `(2p − 1)² ≤ 2^R / 2^K`. -/
theorem parity_prob_sq {K R : ℕ} (F G : (Fin K → ZMod 2) → Fin R → ZMod 2) (a b : (Fin K → ZMod 2) → ZMod 2) :
    (2 * prob (fun xy : (Fin K → ZMod 2) × (Fin K → ZMod 2) =>
      F xy.1 ⬝ᵥ G xy.2 + a xy.1 + b xy.2 = xy.1 ⬝ᵥ xy.2) - 1) ^ 2 ≤ 2 ^ R / 2 ^ K := by
  classical
  have hN : (0 : ℝ) < 2 ^ K := by positivity
  have hS : ∑ x, ∑ y, sgn (F x ⬝ᵥ G y + a x + b y + x ⬝ᵥ y) =
      2 * ∑ xy : (Fin K → ZMod 2) × (Fin K → ZMod 2),
        (if F xy.1 ⬝ᵥ G xy.2 + a xy.1 + b xy.2 = xy.1 ⬝ᵥ xy.2 then (1 : ℝ) else 0) - (2 ^ K) ^ 2 := by
    rw [Fintype.sum_prod_type]
    simp only [sgn_eq, sum_sub_distrib, ← mul_sum, sum_const, card_univ, card_vec, nsmul_eq_mul, mul_one]
    push_cast
    ring
  have hp : 2 * prob (fun xy : (Fin K → ZMod 2) × (Fin K → ZMod 2) =>
      F xy.1 ⬝ᵥ G xy.2 + a xy.1 + b xy.2 = xy.1 ⬝ᵥ xy.2) - 1 =
      (∑ x, ∑ y, sgn (F x ⬝ᵥ G y + a x + b y + x ⬝ᵥ y)) / (2 ^ K) ^ 2 := by
    rw [hS, prob_eq_expect]
    unfold expect
    rw [Fintype.card_prod, card_vec]
    push_cast
    field_simp
  rw [hp, div_pow, div_le_div_iff₀ (by positivity) hN]
  calc (∑ x, ∑ y, sgn (F x ⬝ᵥ G y + a x + b y + x ⬝ᵥ y)) ^ 2 * 2 ^ K ≤ 2 ^ R * (2 ^ K) ^ 3 * 2 ^ K :=
        mul_le_mul_of_nonneg_right (parity_corr_sq F G a b) hN.le
    _ = 2 ^ R * ((2 ^ K) ^ 2) ^ 2 := by ring

/-- **The parity luck, one product short**: with `R < K` the agreement is at most `½ + ½ √½ < 0.854`. -/
theorem parity_luck_short {K R : ℕ} (hR : R < K) (F G : (Fin K → ZMod 2) → Fin R → ZMod 2)
    (a b : (Fin K → ZMod 2) → ZMod 2) :
    prob (fun xy : (Fin K → ZMod 2) × (Fin K → ZMod 2) => F xy.1 ⬝ᵥ G xy.2 + a xy.1 + b xy.2 = xy.1 ⬝ᵥ xy.2) ≤
      427 / 500 := by
  have h := parity_prob_sq F G a b
  have h2 : (2 : ℝ) ^ R / 2 ^ K ≤ 1 / 2 := by
    rw [div_le_div_iff₀ (by positivity) (by positivity), one_mul]
    calc (2 : ℝ) ^ R * 2 = 2 ^ (R + 1) := by ring
      _ ≤ 2 ^ K := pow_le_pow_right₀ (by norm_num) hR
  nlinarith

/-- **The parity luck, two products short**: with `R + 2 ≤ K` the agreement is at most `¾`. -/
theorem parity_luck_two {K R : ℕ} (hR : R + 2 ≤ K) (F G : (Fin K → ZMod 2) → Fin R → ZMod 2)
    (a b : (Fin K → ZMod 2) → ZMod 2) :
    prob (fun xy : (Fin K → ZMod 2) × (Fin K → ZMod 2) => F xy.1 ⬝ᵥ G xy.2 + a xy.1 + b xy.2 = xy.1 ⬝ᵥ xy.2) ≤
      3 / 4 := by
  have h := parity_prob_sq F G a b
  have h2 : (2 : ℝ) ^ R / 2 ^ K ≤ 1 / 4 := by
    rw [div_le_div_iff₀ (by positivity) (by positivity), one_mul]
    calc (2 : ℝ) ^ R * 4 = 2 ^ (R + 2) := by ring
      _ ≤ 2 ^ K := pow_le_pow_right₀ (by norm_num) hR
  nlinarith

end Parity

section ParityFrame

variable {V A : Type} [Fintype V] [DecidableEq V] [CommRing A] {K : ℕ}

/-- A cell's frame bits mod 2 at the bit point `S`: its `x` bits (`.inl j`) and its `y` bits (`.inr j`). -/
def frameBits (fr : Fin K ⊕ Fin K → V) (S : Finset V) : Fin K ⊕ Fin K → ZMod 2 := fun s => if fr s ∈ S then 1 else 0

/-- The bit point `S` with its frame bits replaced by `z`. -/
def reframe (fr : Fin K ⊕ Fin K → V) (S : Finset V) (z : Fin K ⊕ Fin K → ZMod 2) : Finset V :=
  univ.filter fun v => (∃ s, fr s = v ∧ z s = 1) ∨ ((∀ s, fr s ≠ v) ∧ v ∈ S)

theorem mem_reframe_fr {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (S : Finset V)
    (z : Fin K ⊕ Fin K → ZMod 2) (s : Fin K ⊕ Fin K) : fr s ∈ reframe fr S z ↔ z s = 1 := by
  simp only [reframe, mem_filter, mem_univ, true_and]
  constructor
  · rintro (⟨s', hs', hz⟩ | ⟨h, -⟩)
    · rwa [← hfr hs']
    · exact absurd rfl (h s)
  · exact fun h => Or.inl ⟨s, rfl, h⟩

omit [Fintype V] in
theorem mem_reframe_off {fr : Fin K ⊕ Fin K → V} [Fintype V] (S : Finset V) (z : Fin K ⊕ Fin K → ZMod 2) {v : V}
    (hv : ∀ s, fr s ≠ v) : v ∈ reframe fr S z ↔ v ∈ S := by
  simp only [reframe, mem_filter, mem_univ, true_and]
  constructor
  · rintro (⟨s, hs, -⟩ | ⟨-, h⟩)
    · exact absurd hs (hv s)
    · exact h
  · exact fun h => Or.inr ⟨hv, h⟩

theorem zmod2_ite_one (t : ZMod 2) : (if t = 1 then (1 : ZMod 2) else 0) = t := by
  revert t
  decide

theorem frameBits_reframe {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (S : Finset V)
    (z : Fin K ⊕ Fin K → ZMod 2) : frameBits fr (reframe fr S z) = z := by
  funext s
  simp only [frameBits, mem_reframe_fr hfr]
  exact zmod2_ite_one (z s)

theorem reframe_reframe {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (S : Finset V)
    (z : Fin K ⊕ Fin K → ZMod 2) : reframe fr (reframe fr S z) (frameBits fr S) = S := by
  ext v
  by_cases hv : ∃ s, fr s = v
  · obtain ⟨s, rfl⟩ := hv
    rw [mem_reframe_fr hfr]
    simp only [frameBits]
    split_ifs with h <;> simp [h]
  · push Not at hv
    rw [mem_reframe_off _ _ hv, mem_reframe_off _ _ hv]

/-- Two frame replacements that agree off the frame bits in `P` agree at every bit outside `fr '' P`. -/
theorem mem_reframe_congr {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (S : Finset V)
    {z z' : Fin K ⊕ Fin K → ZMod 2} (P : Fin K ⊕ Fin K → Prop) (hz : ∀ s, ¬P s → z s = z' s) {v : V}
    (hv : ∀ s, P s → fr s ≠ v) : v ∈ reframe fr S z ↔ v ∈ reframe fr S z' := by
  by_cases h : ∃ s, fr s = v
  · obtain ⟨s, rfl⟩ := h
    rw [mem_reframe_fr hfr, mem_reframe_fr hfr, hz s fun hp => hv s hp rfl]
  · push Not at h
    rw [mem_reframe_off _ _ h, mem_reframe_off _ _ h]

/-- Replacing a uniform bit point's frame bits by uniform ones keeps it uniform: `(S, z) ↦ (S with z, S's frame
bits)` is an involution. -/
theorem prob_reframe {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (P : Finset V → Prop) :
    prob P = prob fun b : Finset V × (Fin K ⊕ Fin K → ZMod 2) => P (reframe fr b.1 b.2) := by
  classical
  let φ : Finset V × (Fin K ⊕ Fin K → ZMod 2) → Finset V × (Fin K ⊕ Fin K → ZMod 2) :=
    fun b => (reframe fr b.1 b.2, frameBits fr b.1)
  have hφ : Function.Involutive φ := fun b => by
    simp only [φ, reframe_reframe hfr, frameBits_reframe hfr]
  rw [prob_equiv hφ.toPerm (fun b : Finset V × (Fin K ⊕ Fin K → ZMod 2) => P (reframe fr b.1 b.2)),
    ← prob_fst (β := Fin K ⊕ Fin K → ZMod 2) P]
  congr 1
  funext b
  have hs : hφ.toPerm.symm b = φ b := rfl
  rw [hs]
  simp only [φ, reframe_reframe hfr]

/-- **A parity form** of an error at a cell's frame `fr` (its `x` bits, then its `y` bits) with `R` terms: mod 2
(`φ`) the error at every bit point is `⟨F, G⟩ + a + b + ⟨x, y⟩`, with `F` and `a` blind to the `y` bits and `G` and
`b` blind to the `x` bits (RS-E-REF r6 VI.2.1's cell projection: every other bit is fixed). -/
def ParityForm (fr : Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) (R : ℕ) (e : (V → A) → A) : Prop :=
  ∃ (F G : Finset V → Fin R → ZMod 2) (a b : Finset V → ZMod 2),
    (∀ S S' : Finset V, (∀ v, (∀ j, fr (.inr j) ≠ v) → (v ∈ S ↔ v ∈ S')) → F S = F S' ∧ a S = a S') ∧
    (∀ S S' : Finset V, (∀ v, (∀ j, fr (.inl j) ≠ v) → (v ∈ S ↔ v ∈ S')) → G S = G S' ∧ b S = b S') ∧
    ∀ S, φ (e (bitInd S)) =
      F S ⬝ᵥ G S + a S + b S + (fun j => frameBits fr S (.inl j)) ⬝ᵥ fun j => frameBits fr S (.inr j)

/-- **The parity lemma at a cell's frame**: an error with a parity form of `R` terms vanishes at a uniform bit point
with probability at most any bound `q` of the parity lemma on `F₂^K` at `R` terms. -/
theorem prob_zero_le_of_parityForm {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) {φ : A →+* ZMod 2}
    {R : ℕ} {e : (V → A) → A} (h : ParityForm fr φ R e) {q : ℝ} (hq : 0 ≤ q)
    (hcore : ∀ (F G : (Fin K → ZMod 2) → Fin R → ZMod 2) (a b : (Fin K → ZMod 2) → ZMod 2),
      prob (fun xy : (Fin K → ZMod 2) × (Fin K → ZMod 2) => F xy.1 ⬝ᵥ G xy.2 + a xy.1 + b xy.2 = xy.1 ⬝ᵥ xy.2) ≤
        q) :
    prob (fun S : Finset V => e (bitInd S) = 0) ≤ q := by
  classical
  obtain ⟨F, G, a, b, hF, hG, he⟩ := h
  have hmono : prob (fun S : Finset V => e (bitInd S) = 0) ≤ prob fun S : Finset V =>
      F S ⬝ᵥ G S + a S + b S = (fun j => frameBits fr S (.inl j)) ⬝ᵥ fun j => frameBits fr S (.inr j) :=
    prob_mono fun S hS => by
      have := he S
      rw [hS, map_zero] at this
      exact ((zmod2_add_eq_zero _ _).mp this.symm).symm
  refine hmono.trans ?_
  rw [prob_reframe hfr, prob_equiv (Equiv.prodComm _ _)]
  refine prob_le_of_slices _ hq fun S => ?_
  rw [prob_equiv (Equiv.sumArrowEquivProdArrow (Fin K) (Fin K) (ZMod 2))]
  refine le_of_eq_of_le ?_ (hcore (fun x => F (reframe fr S (Sum.elim x 0))) (fun y => G (reframe fr S (Sum.elim 0 y)))
    (fun x => a (reframe fr S (Sum.elim x 0))) fun y => b (reframe fr S (Sum.elim 0 y)))
  congr 1
  funext xy
  have hsy : (Equiv.sumArrowEquivProdArrow (Fin K) (Fin K) (ZMod 2)).symm xy = Sum.elim xy.1 xy.2 := by
    funext s
    cases s <;> rfl
  simp only [Equiv.prodComm_symm, Equiv.prodComm_apply, Prod.fst_swap, Prod.snd_swap, hsy,
    frameBits_reframe hfr, Sum.elim_inl, Sum.elim_inr]
  have hx := hF (reframe fr S (Sum.elim xy.1 xy.2)) (reframe fr S (Sum.elim xy.1 0)) fun v hv =>
    mem_reframe_congr hfr S (fun s => ∃ j, s = .inr j) (fun s hs => by
      cases s with
      | inl j => rfl
      | inr j => exact absurd ⟨j, rfl⟩ hs) fun s ⟨j, hj⟩ => hj ▸ hv j
  have hy := hG (reframe fr S (Sum.elim xy.1 xy.2)) (reframe fr S (Sum.elim 0 xy.2)) fun v hv =>
    mem_reframe_congr hfr S (fun s => ∃ j, s = .inl j) (fun s hs => by
      cases s with
      | inl j => exact absurd ⟨j, rfl⟩ hs
      | inr j => rfl) fun s ⟨j, hj⟩ => hj ▸ hv j
  rw [hx.1, hx.2, hy.1, hy.2]

end ParityFrame

section ParitySplit

variable {V : Type} {K : ℕ}

/-- One-sided junk at a frame: a function blind to the `y` bits plus one blind to the `x` bits. -/
def Split (fr : Fin K ⊕ Fin K → V) (h : Finset V → ZMod 2) : Prop :=
  ∃ a b : Finset V → ZMod 2, YBlind fr a ∧ XBlind fr b ∧ ∀ S, h S = a S + b S

theorem XBlind.const (fr : Fin K ⊕ Fin K → V) (u : ZMod 2) : XBlind fr fun _ => u := fun _ _ _ => rfl

theorem YBlind.const (fr : Fin K ⊕ Fin K → V) (u : ZMod 2) : YBlind fr fun _ => u := fun _ _ _ => rfl

theorem XBlind.add {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : XBlind fr f) (hg : XBlind fr g) :
    XBlind fr fun S => f S + g S := fun S S' h => congrArg₂ (· + ·) (hf S S' h) (hg S S' h)

theorem YBlind.add {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : YBlind fr f) (hg : YBlind fr g) :
    YBlind fr fun S => f S + g S := fun S S' h => congrArg₂ (· + ·) (hf S S' h) (hg S S' h)

theorem XBlind.mul {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : XBlind fr f) (hg : XBlind fr g) :
    XBlind fr fun S => f S * g S := fun S S' h => congrArg₂ (· * ·) (hf S S' h) (hg S S' h)

theorem YBlind.mul {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : YBlind fr f) (hg : YBlind fr g) :
    YBlind fr fun S => f S * g S := fun S S' h => congrArg₂ (· * ·) (hf S S' h) (hg S S' h)

theorem XBlind.sum {fr : Fin K ⊕ Fin K → V} {ι : Type} (s : Finset ι) {f : ι → Finset V → ZMod 2}
    (hf : ∀ i ∈ s, XBlind fr (f i)) : XBlind fr fun S => ∑ i ∈ s, f i S := fun S S' h =>
  sum_congr rfl fun i hi => hf i hi S S' h

theorem YBlind.sum {fr : Fin K ⊕ Fin K → V} {ι : Type} (s : Finset ι) {f : ι → Finset V → ZMod 2}
    (hf : ∀ i ∈ s, YBlind fr (f i)) : YBlind fr fun S => ∑ i ∈ s, f i S := fun S S' h =>
  sum_congr rfl fun i hi => hf i hi S S' h

theorem YBlind.congr {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : YBlind fr f) (h : ∀ S, f S = g S) :
    YBlind fr g := fun S S' hS => by rw [← h, ← h]; exact hf S S' hS

theorem Split.of_y {fr : Fin K ⊕ Fin K → V} {f : Finset V → ZMod 2} (hf : YBlind fr f) : Split fr f :=
  ⟨f, fun _ => 0, hf, XBlind.const fr 0, fun _ => (add_zero _).symm⟩

theorem Split.of_x {fr : Fin K ⊕ Fin K → V} {f : Finset V → ZMod 2} (hf : XBlind fr f) : Split fr f :=
  ⟨fun _ => 0, f, YBlind.const fr 0, hf, fun _ => (zero_add _).symm⟩

theorem Split.add {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : Split fr f) (hg : Split fr g) :
    Split fr fun S => f S + g S := by
  obtain ⟨a, b, ha, hb, he⟩ := hf
  obtain ⟨a', b', ha', hb', he'⟩ := hg
  exact ⟨fun S => a S + a' S, fun S => b S + b' S, ha.add ha', hb.add hb', fun S => by simp only [he, he']; ring⟩

theorem Split.neg {fr : Fin K ⊕ Fin K → V} {f : Finset V → ZMod 2} (hf : Split fr f) : Split fr fun S => -f S := by
  obtain ⟨a, b, ha, hb, he⟩ := hf
  exact ⟨fun S => -a S, fun S => -b S, fun S S' h => congrArg (- ·) (ha S S' h),
    fun S S' h => congrArg (- ·) (hb S S' h),
    fun S => by simp only [he]; ring⟩

theorem Split.sub {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : Split fr f) (hg : Split fr g) :
    Split fr fun S => f S - g S := by
  simpa [sub_eq_add_neg] using hf.add hg.neg

theorem Split.sum {fr : Fin K ⊕ Fin K → V} {ι : Type} (s : Finset ι) {f : ι → Finset V → ZMod 2}
    (hf : ∀ i ∈ s, Split fr (f i)) : Split fr fun S => ∑ i ∈ s, f i S := by
  classical
  induction s using Finset.induction_on with
  | empty => exact Split.of_y (by simpa using YBlind.const fr 0)
  | insert i s hi ih =>
    simp only [sum_insert hi]
    exact (hf i (mem_insert_self i s)).add (ih fun i' hi' => hf i' (mem_insert_of_mem hi'))

theorem Split.congr {fr : Fin K ⊕ Fin K → V} {f g : Finset V → ZMod 2} (hf : Split fr f) (h : ∀ S, f S = g S) :
    Split fr g := by
  obtain ⟨a, b, ha, hb, he⟩ := hf
  exact ⟨a, b, ha, hb, fun S => by rw [← h, he]⟩

variable [DecidableEq V] {A : Type} [CommRing A]

/-- A product split at a frame, mod 2: a factor blind to the `y` bits times one blind to the `x` bits. -/
def SplitProd (fr : Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) (π : (V → A) → A) : Prop :=
  ∃ g h : Finset V → ZMod 2, YBlind fr g ∧ XBlind fr h ∧ ∀ S, φ (π (bitInd S)) = g S * h S

omit [CommRing A] in
/-- A bit off the frame's `x` bits. -/
theorem XBlind.bit {fr : Fin K ⊕ Fin K → V} {v : V} (hv : ∀ j, fr (.inl j) ≠ v) :
    XBlind fr fun S => if v ∈ S then 1 else 0 := fun S S' h => by simp only [h v hv]

omit [CommRing A] in
theorem YBlind.bit {fr : Fin K ⊕ Fin K → V} {v : V} (hv : ∀ j, fr (.inr j) ≠ v) :
    YBlind fr fun S => if v ∈ S then 1 else 0 := fun S S' h => by simp only [h v hv]

/-- A ring map onto `F₂` sends `bitInd` to the bit's indicator. -/
theorem map_bitInd (φ : A →+* ZMod 2) (S : Finset V) (v : V) :
    φ (bitInd S v) = if v ∈ S then 1 else 0 := by
  unfold bitInd
  split_ifs <;> simp

omit [CommRing A] in
/-- A bit times a constant is junk at any frame: blind to the `x` bits if it is a `y` bit, else to the `y` bits. -/
theorem Split.const_mul_bit {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (u : ZMod 2) (v : V) :
    Split fr fun S => u * if v ∈ S then 1 else 0 := by
  by_cases hv : ∃ j, fr (.inr j) = v
  · obtain ⟨j, rfl⟩ := hv
    exact Split.of_x ((XBlind.const fr u).mul (XBlind.bit fun j' h => Sum.inl_ne_inr (hfr h)))
  · push Not at hv
    exact Split.of_y ((YBlind.const fr u).mul (YBlind.bit hv))

/-- **A bit-affine word is junk at any frame**, mod 2. -/
theorem Split.bitAff [Fintype V] {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (φ : A →+* ZMod 2)
    (Af : BitAff V A) :
    Split fr fun S => φ (Af.eval (bitInd S)) :=
  ((Split.of_y (YBlind.const fr (φ Af.c))).add (Split.sum univ fun v _ =>
    Split.const_mul_bit hfr (φ (Af.α v)) v)).congr fun S => by
      simp only [BitAff.eval, map_add, map_sum, map_mul, map_bitInd]

theorem zmod2_add_self (u : ZMod 2) : u + u = 0 := by
  revert u
  decide

/-- **A parity form from junk**: two-sided factors `F` blind to the `y` bits and `G` blind to the `x` bits, and the
rest of the error mod 2, past `⟨F, G⟩ + ⟨x, y⟩`, junk at the frame. -/
theorem parityForm_of_split {fr : Fin K ⊕ Fin K → V} {φ : A →+* ZMod 2} {R : ℕ} {e : (V → A) → A}
    (F G : Finset V → Fin R → ZMod 2) (hF : ∀ k, YBlind fr fun S => F S k) (hG : ∀ k, XBlind fr fun S => G S k)
    (h : Split fr fun S => φ (e (bitInd S)) - F S ⬝ᵥ G S -
      (fun j => frameBits fr S (.inl j)) ⬝ᵥ fun j => frameBits fr S (.inr j)) :
    ParityForm fr φ R e := by
  obtain ⟨a, b, ha, hb, he⟩ := h
  refine ⟨F, G, a, b, fun S S' hS => ⟨funext fun k => hF k S S' hS, ha S S' hS⟩,
    fun S S' hS => ⟨funext fun k => hG k S S' hS, hb S S' hS⟩, fun S => ?_⟩
  linear_combination he S

open Classical in
/-- **The products charged to a word at a frame**: those it reads with an odd coefficient (`φ coef ≠ 0`; mod 2 an
even one reads nothing) that are two-sided there. -/
noncomputable def charged (fr : Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) {N : ℕ} (coef : Fin N → A)
    (π : Fin N → (V → A) → A) : Finset (Fin N) :=
  univ.filter fun i => φ (coef i) ≠ 0 ∧ TwoSidedAt fr φ (π i)

/-- **A sum of split products plus a bit-affine word, minus a target of the form `⟨x, y⟩` plus junk, has a parity
form with its charged products as terms**: the uncharged products it reads are blind to one side, so junk. -/
theorem parityForm_sum [Fintype V] {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) (φ : A →+* ZMod 2)
    {N : ℕ} (coef : Fin N → A) (π : Fin N → (V → A) → A) (Af : BitAff V A)
    (hπ : ∀ i, coef i ≠ 0 → SplitProd fr φ (π i)) (T : (V → A) → A)
    (hT : Split fr fun S => φ (T (bitInd S)) -
      (fun j => frameBits fr S (.inl j)) ⬝ᵥ fun j => frameBits fr S (.inr j)) :
    ParityForm fr φ (charged fr φ coef π).card fun z => ∑ i, coef i * π i z + Af.eval z - T z := by
  classical
  have hπ' : ∀ i, ∃ g h : Finset V → ZMod 2,
      coef i ≠ 0 → YBlind fr g ∧ XBlind fr h ∧ ∀ S, φ (π i (bitInd S)) = g S * h S := fun i => by
    by_cases hc : coef i = 0
    · exact ⟨0, 0, fun h => absurd hc h⟩
    · obtain ⟨g, h, hg, hh, hgh⟩ := hπ i hc
      exact ⟨g, h, fun _ => ⟨hg, hh, hgh⟩⟩
  choose g h hgh using hπ'
  set ch := charged fr φ coef π with hch
  have hmem : ∀ i ∈ ch, coef i ≠ 0 := fun i hi => ne_zero_of_map ((mem_filter.mp hi).2).1
  let en := ch.equivFin
  refine parityForm_of_split (fun S k => φ (coef (en.symm k)) * g (en.symm k) S) (fun S k => h (en.symm k) S)
    (fun k => (YBlind.const fr _).mul (hgh _ (hmem _ (en.symm k).2)).1)
    (fun k => (hgh _ (hmem _ (en.symm k).2)).2.1) ?_
  have hjunk : Split fr fun S => ∑ i ∈ univ.filter (fun i => i ∉ ch), φ (coef i) * φ (π i (bitInd S)) := by
    refine Split.sum _ fun i hi => ?_
    have hi' : i ∉ ch := (mem_filter.mp hi).2
    by_cases hc : φ (coef i) = 0
    · exact Split.of_y ((YBlind.const fr 0).congr fun S => by rw [hc, zero_mul])
    · have hts : ¬TwoSidedAt fr φ (π i) := fun ht => hi' (mem_filter.mpr ⟨mem_univ _, hc, ht⟩)
      unfold TwoSidedAt at hts
      rcases not_and_or.mp hts with hx | hy
      · exact Split.of_x ((XBlind.const fr _).mul (not_not.mp hx))
      · exact Split.of_y ((YBlind.const fr _).mul (not_not.mp hy))
  refine ((hjunk.add (Split.bitAff hfr φ Af)).sub hT).congr fun S => ?_
  have hsplit := sum_filter_add_sum_filter_not (s := (univ : Finset (Fin N))) (fun i => i ∈ ch)
    (fun i => φ (coef i) * φ (π i (bitInd S)))
  have hfilt : univ.filter (fun i => i ∈ ch) = ch := by ext i; simp
  rw [hfilt] at hsplit
  have hch' : ∑ i ∈ ch, φ (coef i) * φ (π i (bitInd S)) =
      ∑ k, φ (coef (en.symm k)) * g (en.symm k) S * h (en.symm k) S := by
    rw [← sum_coe_sort ch, ← Equiv.sum_comp en.symm]
    refine sum_congr rfl fun k _ => ?_
    rw [(hgh _ (hmem _ (en.symm k).2)).2.2 S, mul_assoc]
  simp only [map_sub, map_add, map_sum, map_mul]
  rw [← hsplit, hch']
  simp only [dotProduct]
  linear_combination zmod2_add_self (∑ x, frameBits fr S (Sum.inl x) * frameBits fr S (Sum.inr x))

end ParitySplit

section ParityDesign

variable {Ar : Type} [CommRing Ar] {n L np B n' : ℕ}

theorem dFrame_injective (p₀ : Fin np) (c : DCells n n' B) : Function.Injective (dFrame (L := L) p₀ c) := by
  rintro (j | j) (j' | j') h <;> simp_all [dFrame]

/-- A one-sided product of any form is split at every cell-block's frame: its factor on U's bits is blind to the
coins, and its factor on the coins to U's bits. -/
theorem OneSided.splitProd (φ : Ar →+* ZMod 2) (p₀ : Fin np) (c : DCells n n' B)
    {π : (DBits n L np B n' → Ar) → Ar} (hπ : OneSided π) : SplitProd (dFrame p₀ c) φ π := by
  obtain ⟨g, h, hg, hh, hgh⟩ := hπ
  refine ⟨fun S => φ (g (bitInd S)), fun S => φ (h (bitInd S)), fun S S' hS => ?_, fun S S' hS => ?_,
    fun S => by rw [hgh, map_mul]⟩
  · exact congrArg φ (hg (bitInd S) (bitInd S') fun a => by
      unfold bitInd
      rw [if_congr (hS (.inl a) fun j => by simp [dFrame]) rfl rfl])
  · exact congrArg φ (hh (bitInd S) (bitInd S') fun b => by
      unfold bitInd
      rw [if_congr (hS (.inr b) fun j => by simp [dFrame]) rfl rfl])

theorem dX_eval (vq : Fin B → Fin L → Fin n' → Ar) (t : Fin B) (j : Fin L) (l : Fin n')
    (z : DBits n L np B n' → Ar) : (dX vq t j l).eval z = vq t j l + z (.inr (t, j, l)) := by
  simp [BitAff.eval, dX, Fintype.sum_sum_type]

theorem dP_eval (pc : Fin n → Fin B → Fin L → Ar) (w : Fin np → Ar) (i : Fin n) (t : Fin B) (j : Fin L)
    (z : DBits n L np B n' → Ar) : (dP pc w i t j).eval z = pc i t j + ∑ p, w p * z (.inl (i, j, p)) := by
  simp [BitAff.eval, dP, Fintype.sum_sum_type, Fintype.sum_prod_type, ite_and]

/-- **The design's target is `⟨x, y⟩` plus junk at its cell-block's frame**, mod 2 with `φ (w p₀) = 1`: block
`t`'s increment is `Σ_j (x_j + Q_j)(c_j + y_j)`, `Q_j` U's other planes and the constants, and the earlier blocks'
increments read no coin of the frame. -/
theorem split_dPrefix (φ : Ar →+* ZMod 2) {p₀ : Fin np} {w : Fin np → Ar} (hw : φ (w p₀) = 1)
    (pc : Fin n → Fin B → Fin L → Ar) (vq : Fin B → Fin L → Fin n' → Ar) (c : DCells n n' B) :
    Split (dFrame p₀ c) fun S => φ (dPrefix pc vq w c.1 c.2.1 c.2.2.val (bitInd S)) -
      (fun j => frameBits (dFrame p₀ c) S (.inl j)) ⬝ᵥ fun j => frameBits (dFrame p₀ c) S (.inr j) := by
  classical
  obtain ⟨i, l, t⟩ := c
  set fr := dFrame (L := L) p₀ (i, l, t) with hfr
  let xb : Fin L → Finset (DBits n L np B n') → ZMod 2 := fun j S =>
    if (.inl (i, j, p₀) : DBits n L np B n') ∈ S then 1 else 0
  let yb : Fin B → Fin L → Finset (DBits n L np B n') → ZMod 2 := fun t' j S =>
    if (.inr (t', j, l) : DBits n L np B n') ∈ S then 1 else 0
  let Q : Fin B → Fin L → Finset (DBits n L np B n') → ZMod 2 := fun t' j S =>
    φ (pc i t' j) + ∑ p ∈ univ.erase p₀, φ (w p) * if (.inl (i, j, p) : DBits n L np B n') ∈ S then 1 else 0
  have hP : ∀ t' j S, φ ((dP pc w i t' j).eval (bitInd S)) = xb j S + Q t' j S := fun t' j S => by
    rw [dP_eval, map_add, map_sum, ← add_sum_erase _ _ (mem_univ p₀)]
    simp only [map_mul, map_bitInd, hw, one_mul, xb, Q]
    ring
  have hX : ∀ t' j S, φ ((dX vq t' j l).eval (bitInd S)) = φ (vq t' j l) + yb t' j S := fun t' j S => by
    rw [dX_eval, map_add, map_bitInd]
  have hQx : ∀ t' j, XBlind fr (Q t' j) := fun t' j =>
    (XBlind.const fr _).add (XBlind.sum _ fun p hp => (XBlind.const fr _).mul (XBlind.bit fun j' h => by
      simp only [hfr, dFrame, Sum.elim_inl, Sum.inl.injEq, Prod.mk.injEq] at h
      exact (mem_erase.mp hp).1 h.2.2.symm))
  have hQy : ∀ t' j, YBlind fr (Q t' j) := fun t' j =>
    (YBlind.const fr _).add (YBlind.sum _ fun p _ => (YBlind.const fr _).mul (YBlind.bit fun j' h => by
      simp [hfr, dFrame] at h))
  have hxy : ∀ j, YBlind fr (xb j) := fun j => YBlind.bit fun j' h => by simp [hfr, dFrame] at h
  have hyx : ∀ t' j, XBlind fr (yb t' j) := fun t' j => XBlind.bit fun j' h => by simp [hfr, dFrame] at h
  have hyy : ∀ t' j, t' ≠ t → YBlind fr (yb t' j) := fun t' j ht => YBlind.bit fun j' h => by
    simp only [hfr, dFrame, Sum.elim_inr, Sum.inr.injEq, Prod.mk.injEq] at h
    exact ht h.1.symm
  have hpiece : ∀ t' j, Split fr fun S =>
      (if t'.val ≤ t.val then (xb j S + Q t' j S) * (φ (vq t' j l) + yb t' j S) else 0) -
        if t' = t then xb j S * yb t j S else 0 := fun t' j => by
    by_cases ht : t' = t
    · subst ht
      refine ((Split.of_y (((hxy j).mul (YBlind.const fr (φ (vq t' j l)))).add
        ((hQy t' j).mul (YBlind.const fr (φ (vq t' j l)))))).add
        (Split.of_x ((hQx t' j).mul (hyx t' j)))).congr fun S => ?_
      simp only [le_refl, ↓reduceIte]
      ring
    · by_cases hle : t'.val ≤ t.val
      · refine (Split.of_y (((hxy j).add (hQy t' j)).mul ((YBlind.const fr (φ (vq t' j l))).add
          (hyy t' j ht)))).congr
          fun S => ?_
        simp only [hle, ht, ↓reduceIte, sub_zero]
      · exact (Split.of_y (YBlind.const fr 0)).congr fun S => by simp [hle, ht]
  refine (Split.sum univ fun t' _ => Split.sum univ fun j _ => hpiece t' j).congr fun S => ?_
  simp only [sum_sub_distrib]
  have hdot : ((fun j => frameBits fr S (.inl j)) ⬝ᵥ fun j => frameBits fr S (.inr j)) =
      ∑ t' : Fin B, ∑ j, if t' = t then xb j S * yb t j S else 0 := by
    rw [sum_comm]
    simp only [sum_ite_eq', mem_univ, ↓reduceIte, dotProduct]
    rfl
  rw [hdot]
  congr 1
  simp only [dPrefix, dIncr, map_sum, apply_ite φ, map_zero, map_mul]
  refine sum_congr rfl fun t' _ => ?_
  split_ifs with h
  · exact sum_congr rfl fun j _ => by rw [hP, hX]
  · simp

/-- **A one-sided run's error has a parity form at every cell-block's frame**, its terms the products charged to the
cell-block's word (`charged`): any straight-line program over a ring with a map `φ` onto `F₂`, its products one-sided
of any form, and `φ (w p₀) = 1`. -/
theorem SlpG.parityForm_err (φ : Ar →+* ZMod 2) {p₀ : Fin np} {w : Fin np → Ar} (hw : φ (w p₀) = 1) {N : ℕ}
    (P : SlpG (DBits n L np B n') Ar N) (hOk : ∀ j, P.isProd j = true → OneSided (P.prod j))
    (pc : Fin n → Fin B → Fin L → Ar) (vq : Fin B → Fin L → Fin n' → Ar) (j : Fin N) (c : DCells n n' B) :
    ParityForm (dFrame p₀ c) φ (charged (dFrame p₀ c) φ (P.coef j) P.prod).card
      fun z => P.val z j - dPrefix pc vq w c.1 c.2.1 c.2.2.val z := by
  have h := parityForm_sum (dFrame_injective p₀ c) φ (P.coef j) P.prod (P.aff j)
    (fun i hi => OneSided.splitProd φ p₀ c (hOk i (P.isProd_of_coef_ne_zero j i hi))) _ (split_dPrefix φ hw pc vq c)
  have e : (fun z => P.val z j - dPrefix pc vq w c.1 c.2.1 c.2.2.val z) =
      fun z => ∑ i, P.coef j i * P.prod i z + (P.aff j).eval z - dPrefix pc vq w c.1 c.2.1 c.2.2.val z :=
    funext fun z => by rw [P.val_eq]
  rw [e]
  exact h

end ParityDesign

section ParityCount

variable {V A : Type} [DecidableEq V] [CommRing A] {K : ℕ}

/-- **A cell-block short of products**: its error has a parity form with fewer than `K` terms at its frame. -/
def ParShort (fr : Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) (e : (V → A) → A) : Prop :=
  ∃ R < K, ParityForm fr φ R e

/-- **Disjoint charges add up**: at least `L` products charged to each full cell, no product charged twice. -/
theorem card_full_le {C : Type} [Fintype C] {N L : ℕ} (ch : C → Finset (Fin N)) (prods : Finset (Fin N))
    (hsub : ∀ c, ch c ⊆ prods) (hdisj : ∀ c c', c ≠ c' → Disjoint (ch c) (ch c')) (full : C → Prop)
    [DecidablePred full] (hfull : ∀ c, full c → L ≤ (ch c).card) : L * (univ.filter full).card ≤ prods.card := by
  calc L * (univ.filter full).card = ∑ c ∈ univ.filter full, L := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ c ∈ univ.filter full, (ch c).card := sum_le_sum fun c hc => hfull c (mem_filter.mp hc).2
    _ = ((univ.filter full).biUnion ch).card := by
        classical
        exact (card_biUnion fun c _ c' _ hne => hdisj c c' hne).symm
    _ ≤ prods.card := card_le_card (biUnion_subset.mpr fun c _ => hsub c)

end ParityCount

section ParityRound

variable {Ar : Type} [CommRing Ar] {n L np B n' : ℕ}

open Classical in
/-- **Charge-pure one-sided runs add up**: the program has at least `L` products per cell-block not short of
products. -/
theorem SlpG.card_full_le (φ : Ar →+* ZMod 2) {p₀ : Fin np} {w : Fin np → Ar} (hw : φ (w p₀) = 1) {N : ℕ}
    (P : SlpG (DBits n L np B n') Ar N) (hOk : ∀ j, P.isProd j = true → OneSided (P.prod j))
    (pc : Fin n → Fin B → Fin L → Ar) (vq : Fin B → Fin L → Fin n' → Ar) (word : DCells n n' B → Fin N)
    (hpure : P.ChargePure φ p₀ word) :
    L * (univ.filter fun c => ¬ParShort (dFrame p₀ c) φ
      fun z => P.val z (word c) - dPrefix pc vq w c.1 c.2.1 c.2.2.val z).card ≤
      (univ.filter fun j => P.isProd j = true).card := by
  refine _root_.Pouw.RelativeSecurity.card_full_le (fun c => charged (dFrame p₀ c) φ (P.coef (word c)) P.prod) _
    (fun c i hi => mem_filter.mpr ⟨mem_univ _, P.isProd_of_coef_ne_zero _ _ (ne_zero_of_map (mem_filter.mp hi).2.1)⟩)
    (fun c c' hne => disjoint_left.mpr fun i hi hi' =>
      hpure c c' i hne (mem_filter.mp hi).2.1 (mem_filter.mp hi).2.2 (mem_filter.mp hi').2.1
        (mem_filter.mp hi').2.2) _ fun c hc => ?_
  by_contra hlt
  exact hc ⟨_, not_le.mp hlt, P.parityForm_err φ hw hOk pc vq (word c) c⟩

end ParityRound


section ParityGame

variable {C V A : Type} [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] {K : ℕ}

/-- **A round's product deficit** (RS-E-REF r6 VI.2.1): the share of cells short of products at their frames
(`ParShort`). -/
noncomputable def parDef (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) (e : C → (V → A) → A) : ℝ :=
  prob fun c => ParShort (fr c) φ (e c)

omit [Fintype C] in
/-- A cell short of products passes with probability at most `0.854`. -/
theorem prob_parShort_le {fr : Fin K ⊕ Fin K → V} (hfr : Function.Injective fr) {φ : A →+* ZMod 2}
    {e : (V → A) → A} (h : ParShort fr φ e) : prob (fun S : Finset V => e (bitInd S) = 0) ≤ 427 / 500 := by
  obtain ⟨R, hR, hF⟩ := h
  exact prob_zero_le_of_parityForm hfr hF (by norm_num) fun F G a b => parity_luck_short hR F G a b

/-- **The spot check's pass bound for one-sided operands of any form**: a uniform cell and uniform bits find a zero
with probability at most `1 − (1 − 0.854) d`, `d` the share of cells short of products. -/
theorem prob_spot_parity_le (fr : C → Fin K ⊕ Fin K → V) (hfr : ∀ c, Function.Injective (fr c))
    (φ : A →+* ZMod 2) (e : C → (V → A) → A) :
    prob (fun a : C × Finset V => e a.1 (bitInd a.2) = 0) ≤ 1 - (1 - 427 / 500) * parDef fr φ e := by
  classical
  rw [prob_eq_expect, expect_prod, parDef, prob_eq_expect]
  have hpt : ∀ c : C, expect (fun S : Finset V => if e (c, S).1 (bitInd (c, S).2) = 0 then (1 : ℝ) else 0) ≤
      1 + -(1 - 427 / 500) * (if ParShort (fr c) φ (e c) then 1 else 0) := by
    intro c
    split_ifs with hb
    · rw [← prob_eq_expect (fun S : Finset V => e c (bitInd S) = 0)]
      have := prob_parShort_le (hfr c) hb
      linarith
    · rw [mul_zero, add_zero]
      exact expect_le_of_le zero_le_one fun S => by split_ifs <;> norm_num
  have := (expect_mono hpt).trans (expect_affine_le zero_le_one _ _)
  linarith

namespace SpotPlay

/-- **The window's product-deficit game** of a spot play: as `game`, its deficiency the share of cells short of
products (`parDef`). -/
noncomputable def gameP (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    {n : ℕ} → SpotPlay C V A n → RoundGame (C × Finset V) n
  | _, .done => .done
  | _, .round e next => .round (fun _ => parDef fr φ e) (fun a => e a.1 (bitInd a.2) = 0) fun a => (next a).gameP fr φ

omit [Fintype V] in
theorem accepts_gameP (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : SpotPlay C V A n) (x : Paths (C × Finset V) n), (P.gameP fr φ).accepts x ↔ P.game.accepts x
  | _, .done, _ => Iff.rfl
  | _, .round _ next, x => and_congr Iff.rfl (accepts_gameP fr φ (next x.1) x.2)

omit [Fintype V] in
theorem dAt_gameP (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : SpotPlay C V A n) (x : Paths (C × Finset V) n) (w : Fin n),
    (P.gameP fr φ).dAt x w = parDef fr φ (P.errAt x w)
  | _, .done, _, w => w.elim0
  | _, .round e next, x, w => by
    induction w using Fin.cases with
    | zero => simp [gameP, errAt, RoundGame.dAt]
    | succ w' =>
      simp only [gameP, errAt, RoundGame.dAt, Fin.cases_succ]
      exact dAt_gameP fr φ (next x.1) x.2 w'

/-- **Spot plays meet Theorem S's pass bound at `p = 0.854` in the product deficit**, whatever their errors. -/
theorem passBound_gameP {fr : C → Fin K ⊕ Fin K → V} (hfr : ∀ c, Function.Injective (fr c)) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : SpotPlay C V A n), (P.gameP fr φ).PassBound (427 / 500)
  | _, .done => trivial
  | _, .round e next =>
    ⟨⟨parDef fr φ e, fun _ => rfl, prob_spot_parity_le fr hfr φ e⟩, fun a => passBound_gameP hfr φ (next a)⟩

end SpotPlay

namespace StagePlay

/-- **The round's product-deficit stage game**: as `sgame`, each stage's step `1/M` of its share of cells short of
products. -/
noncomputable def sgameP (M : ℕ) (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    {m : ℕ} → StagePlay C V A m → StageGame (Finset V) m
  | _, .done => .done
  | _, .stage e next =>
    .stage ((M : ℝ)⁻¹ * parDef fr φ e) (fun S => (M : ℝ)⁻¹ * prob fun c => e c (bitInd S) ≠ 0)
      fun S => (next S).sgameP M fr φ

omit [Fintype V] in
theorem defi_sgameP (M : ℕ) (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    ∀ {m : ℕ} (P : StagePlay C V A m) (x : Paths (Finset V) m),
    (P.sgameP M fr φ).defi x = ∑ s, (M : ℝ)⁻¹ * parDef fr φ (P.errAt x s)
  | _, .done, _ => by simp [sgameP, StageGame.defi]
  | _, .stage e next, x => by
    rw [Fin.sum_univ_succ]
    simp only [sgameP, StageGame.defi, errAt, Fin.cases_zero, Fin.cases_succ]
    rw [defi_sgameP M fr φ (next x.1) x.2]

omit [Fintype V] in
theorem wrong_sgameP (M : ℕ) (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    ∀ {m : ℕ} (P : StagePlay C V A m) (x : Paths (Finset V) m), (P.sgameP M fr φ).wrong x = (P.sgame M).wrong x
  | _, .done, _ => rfl
  | _, .stage e next, x => by
    simp only [sgameP, sgame, StageGame.wrong]
    rw [wrong_sgameP M fr φ (next x.1) x.2]

/-- **The product-deficit stage game is valid at `p = 0.854`**, whatever the stages' errors. -/
theorem valid_sgameP (M : ℕ) {fr : C → Fin K ⊕ Fin K → V} (hfr : ∀ c, Function.Injective (fr c))
    (φ : A →+* ZMod 2) : ∀ {m : ℕ} (P : StagePlay C V A m), (P.sgameP M fr φ).Valid (427 / 500)
  | _, .done => trivial
  | _, .stage e next => by
    classical
    refine ⟨mul_nonneg (inv_nonneg.mpr (Nat.cast_nonneg _)) (prob_nonneg _),
      fun S => mul_nonneg (inv_nonneg.mpr (Nat.cast_nonneg _)) (prob_nonneg _), ?_,
      fun S => valid_sgameP M hfr φ (next S)⟩
    rw [expect_mul_left, mul_left_comm]
    refine mul_le_mul_of_nonneg_left ?_ (inv_nonneg.mpr (Nat.cast_nonneg _))
    rcases isEmpty_or_nonempty C with hC | hC
    · have h0 : parDef fr φ e = 0 := by
        unfold parDef prob
        simp
      rw [h0, mul_zero]
      unfold expect
      exact div_nonneg (Finset.sum_nonneg fun S _ => prob_nonneg _) (Nat.cast_nonneg _)
    · have h1 : expect (fun S : Finset V => prob fun c => e c (bitInd S) ≠ 0) =
          prob fun a : C × Finset V => e a.1 (bitInd a.2) ≠ 0 := by
        rw [prob_eq_expect (fun a : C × Finset V => e a.1 (bitInd a.2) ≠ 0), expect_prod_swap]
        congr 1
        funext S
        exact prob_eq_expect _
      rw [h1]
      have h2 := prob_add_not (fun a : C × Finset V => e a.1 (bitInd a.2) = 0)
      have h3 := prob_spot_parity_le fr hfr φ e
      linarith

end StagePlay

namespace StagedPlay

variable {M : ℕ}

/-- **The window's product-deficit game** of a staged play: as `game`, round `w`'s deficiency its product-deficit
stage game's. -/
noncomputable def gameP (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    {n : ℕ} → StagedPlay C V A M n → RoundGame (Paths (Finset V) M × (Fin M × C)) n
  | _, .done => .done
  | _, .round P next =>
    .round (fun a => (P.sgameP M fr φ).defi a.1) (fun a => P.ok a.1 a.2.1 a.2.2) fun a => (next a).gameP fr φ

omit [Fintype V] in
theorem accepts_gameP (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : StagedPlay C V A M n) (x : Paths (Paths (Finset V) M × (Fin M × C)) n),
    (P.gameP fr φ).accepts x ↔ P.game.accepts x
  | _, .done, _ => Iff.rfl
  | _, .round _ next, x => and_congr Iff.rfl (accepts_gameP fr φ (next x.1) x.2)

omit [Fintype V] in
theorem dAt_gameP (fr : C → Fin K ⊕ Fin K → V) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : StagedPlay C V A M n) (x : Paths (Paths (Finset V) M × (Fin M × C)) n) (w : Fin n),
    (P.gameP fr φ).dAt x w = ∑ s, (M : ℝ)⁻¹ * parDef fr φ (P.errAt x w s)
  | _, .done, _, w => w.elim0
  | _, .round P next, x, w => by
    induction w using Fin.cases with
    | zero =>
      simp only [gameP, errAt, RoundGame.dAt, Fin.cases_zero]
      exact P.defi_sgameP M fr φ x.1.1
    | succ w' =>
      simp only [gameP, errAt, RoundGame.dAt, Fin.cases_succ]
      exact dAt_gameP fr φ (next x.1) x.2 w'

/-- **Staged plays make a staged window at `p = 0.854` in the product deficit**, whatever their errors. -/
theorem game_stagedP {fr : C → Fin K ⊕ Fin K → V} (hfr : ∀ c, Function.Injective (fr c)) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : StagedPlay C V A M n), (P.gameP fr φ).Staged (427 / 500) M
  | _, .done => trivial
  | _, .round P next => by
    simp only [gameP]
    exact ⟨⟨P.sgameP M fr φ, P.valid_sgameP M hfr φ, fun x j => rfl, fun x => by
      rw [P.wrong_sgameP M fr φ x]
      exact P.prob_ok_le x⟩, fun a => game_stagedP hfr φ (next a)⟩

end StagedPlay

end ParityGame

section ParityOverlap

variable {St : ℕ} {κ ι : Fin St → Type} [∀ t, Fintype (κ t)] [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)]
  {A : Type} [CommRing A] {K : ℕ}

/-- **A round's product deficit with overlapping draws**: the share of cells and draws at which the cell, on its own
draw's bits with every other draw's at theirs, is short of products at its frame. -/
noncomputable def ovDefP (fr : (c : Σ t, κ t) → Fin K ⊕ Fin K → ι c.1) (φ : A →+* ZMod 2)
    (e : (Σ t, κ t) → ((Σ s, ι s) → A) → A) : ℝ :=
  prob fun a : (Σ t, κ t) × (∀ s, Finset (ι s)) => ParShort (fr a.1) φ fun z => e a.1 (mergeAt a.2 a.1.1 z)

/-- **The overlapped spot check's pass bound for one-sided operands of any form**: per draw, with the other draws'
bits given, as `prob_ov_le`. -/
theorem prob_ovP_le (fr : (c : Σ t, κ t) → Fin K ⊕ Fin K → ι c.1) (hfr : ∀ c, Function.Injective (fr c))
    (φ : A →+* ZMod 2) (e : (Σ t, κ t) → ((Σ s, ι s) → A) → A) :
    prob (fun a : (Σ t, κ t) × (∀ s, Finset (ι s)) => e a.1 (drawInd a.2) = 0) ≤
      1 - (1 - 427 / 500) * ovDefP fr φ e := by
  classical
  rw [prob_eq_expect, expect_prod, ovDefP, prob_eq_expect, expect_prod]
  have hpt : ∀ c : Σ t, κ t,
      expect (fun U : ∀ s, Finset (ι s) => if e (c, U).1 (drawInd (c, U).2) = 0 then (1 : ℝ) else 0) ≤
        1 + -(1 - 427 / 500) * expect fun U : ∀ s, Finset (ι s) =>
          if ParShort (fr (c, U).1) φ (fun z => e (c, U).1 (mergeAt (c, U).2 (c, U).1.1 z)) then (1 : ℝ) else 0 := by
    intro c
    rw [← prob_eq_expect (fun U : ∀ s, Finset (ι s) => e c (drawInd U) = 0), prob_cell_eq (e c) c.1,
      ← prob_eq_expect (fun U : ∀ s, Finset (ι s) => ParShort (fr c) φ fun z => e c (mergeAt U c.1 z))]
    have := prob_spot_parity_le (fun _ : ∀ s, Finset (ι s) => fr c) (fun _ => hfr c) φ
      fun (U : ∀ s, Finset (ι s)) (z : ι c.1 → A) => e c (mergeAt U c.1 z)
    rw [parDef] at this
    linarith
  have := (expect_mono hpt).trans (expect_affine_le zero_le_one _ _)
  linarith

namespace OvPlay

/-- **The window's product-deficit game** of a play with overlapping draws: as `game`, its deficiency `ovDefP`. -/
noncomputable def gameP (fr : (c : Σ t, κ t) → Fin K ⊕ Fin K → ι c.1) (φ : A →+* ZMod 2) :
    {n : ℕ} → OvPlay κ ι A n → RoundGame ((Σ t, κ t) × (∀ s, Finset (ι s))) n
  | _, .done => .done
  | _, .round e next => .round (fun _ => ovDefP fr φ e) (fun a => e a.1 (drawInd a.2) = 0) fun a => (next a).gameP fr φ

theorem accepts_gameP (fr : (c : Σ t, κ t) → Fin K ⊕ Fin K → ι c.1) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : OvPlay κ ι A n) (x : Paths ((Σ t, κ t) × (∀ s, Finset (ι s))) n),
    (P.gameP fr φ).accepts x ↔ P.game.accepts x
  | _, .done, _ => Iff.rfl
  | _, .round _ next, x => and_congr Iff.rfl (accepts_gameP fr φ (next x.1) x.2)

theorem dAt_gameP (fr : (c : Σ t, κ t) → Fin K ⊕ Fin K → ι c.1) (φ : A →+* ZMod 2) :
    ∀ {n : ℕ} (P : OvPlay κ ι A n) (x : Paths ((Σ t, κ t) × (∀ s, Finset (ι s))) n) (w : Fin n),
    (P.gameP fr φ).dAt x w = ovDefP fr φ (P.errAt x w)
  | _, .done, _, w => w.elim0
  | _, .round e next, x, w => by
    induction w using Fin.cases with
    | zero => simp [gameP, errAt, RoundGame.dAt]
    | succ w' =>
      simp only [gameP, errAt, RoundGame.dAt, Fin.cases_succ]
      exact dAt_gameP fr φ (next x.1) x.2 w'

/-- **Overlapped plays meet Theorem S's pass bound at `p = 0.854` in the product deficit**, whatever their
errors. -/
theorem passBound_gameP {fr : (c : Σ t, κ t) → Fin K ⊕ Fin K → ι c.1} (hfr : ∀ c, Function.Injective (fr c))
    (φ : A →+* ZMod 2) : ∀ {n : ℕ} (P : OvPlay κ ι A n), (P.gameP fr φ).PassBound (427 / 500)
  | _, .done => trivial
  | _, .round e next =>
    ⟨⟨ovDefP fr φ e, fun _ => rfl, prob_ovP_le fr hfr φ e⟩, fun a => passBound_gameP hfr φ (next a)⟩

end OvPlay

end ParityOverlap

section OverlapRound

variable {St : ℕ} {R : Type} [CommRing R] {n B n' : Fin St → ℕ} {L np : ℕ}

open Classical in
/-- **Overlapping draws add up** (Theorem K∥, RS-D1E r2 §2.2): a round's run over every draw's bits, with K∥'s
products, row-pure across the rows of every draw, has at least `L / c` products per certified cell-block of every
stage, at any bit point `W` of the draws. Per stage `t` and row: with every other draw's bits at theirs in `W`
(`mergeAt`), the products are affine products on draw `t`'s bits, their coefficients unchanged, so the chain bound
counts the products feeding the row's certified words; row purity makes the counts add over every row of every draw,
whatever intervals the products lie in. -/
theorem SlpG.cells_card_le_ov {w : Fin np → R} {c : ℕ}
    (hc : ∀ t, ChainBound R (n t) L np (B t) (n' t) w (fun π => AffProd π) c) {N : ℕ}
    (P : SlpG (Σ t, DBits (n t) L np (B t) (n' t)) R N) (hOk : ∀ j, P.isProd j = true → AffMulAt (P.prod j))
    (W : ∀ t, Finset (DBits (n t) L np (B t) (n' t)))
    (pc : ∀ t, Fin (n t) → Fin (B t) → Fin L → R) (vq : ∀ t, Fin (B t) → Fin L → Fin (n' t) → R)
    (word : (Σ t, DCells (n t) (n' t) (B t)) → Fin N) (cor : ∀ t, Finset (DCells (n t) (n' t) (B t)))
    (hT : ∀ t, ∀ e ∈ cor t, ∀ U : Finset (DBits (n t) L np (B t) (n' t)), U.card ≤ 2 →
      P.val (mergeAt W t (bitInd U)) (word ⟨t, e⟩) = dPrefix (pc t) (vq t) w e.1 e.2.1 e.2.2.val (bitInd U))
    (hpure : ∀ t, ∀ e ∈ cor t, ∀ t', ∀ e' ∈ cor t', (⟨t, e.1⟩ : Σ t, Fin (n t)) ≠ ⟨t', e'.1⟩ → ∀ k,
      P.coef (word ⟨t, e⟩) k ≠ 0 → P.coef (word ⟨t', e'⟩) k = 0) :
    L * ∑ t, (cor t).card ≤ c * (univ.filter fun j => P.isProd j = true).card := by
  let fib : ∀ ρ : Σ t, Fin (n t), Finset (DCells (n ρ.1) (n' ρ.1) (B ρ.1)) := fun ρ =>
    (cor ρ.1).filter fun e => e.1 = ρ.2
  let used : (Σ t, Fin (n t)) → Finset (Fin N) := fun ρ =>
    univ.filter fun k => ∃ e ∈ fib ρ, P.coef (word ⟨ρ.1, e⟩) k ≠ 0
  have hq : ∀ ρ, L * (fib ρ).card ≤ c * (used ρ).card := by
    rintro ⟨t, q⟩
    have := hc t (Fin N) (fib ⟨t, q⟩) (used ⟨t, q⟩) (fun k z => P.prod k (mergeAt W t z))
      (fun k hk => by
        obtain ⟨e, -, he⟩ := (Finset.mem_filter.mp hk).2
        exact hOk k (P.isProd_of_coef_ne_zero _ _ he) W t)
      (pc t) (vq t) q (fun e => e.1.2)
      (fun e e' h => Subtype.ext (Prod.ext ((Finset.mem_filter.mp e.2).2.trans (Finset.mem_filter.mp e'.2).2.symm) h))
      (fun e k => P.coef (word ⟨t, e.1⟩) k) (fun e => (P.aff (word ⟨t, e.1⟩)).restrictAt W t)
      (fun e U hU => by
        have h1 := Finset.mem_filter.mp e.2
        have hq1 : e.1.1 = q := h1.2
        have hTe := hT t e.1 h1.1 U hU
        rw [hq1] at hTe
        rw [← hTe, P.val_eq, ← BitAff.eval_mergeAt]
        congr 1
        refine (Finset.sum_subset (Finset.subset_univ (used ⟨t, q⟩))
          (f := fun k => P.coef (word ⟨t, e.1⟩) k * P.prod k (mergeAt W t (bitInd U))) fun k _ hk => ?_)
        have : P.coef (word ⟨t, e.1⟩) k = 0 := by
          by_contra hne
          exact hk (Finset.mem_filter.mpr ⟨Finset.mem_univ _, e.1, e.2, hne⟩)
        rw [this, zero_mul])
    simpa [Fintype.card_coe] using this
  have hdisj : ((univ : Finset (Σ t, Fin (n t))) : Set (Σ t, Fin (n t))).PairwiseDisjoint used := by
    rintro ⟨t, q⟩ - ⟨t', q'⟩ - hne
    refine Finset.disjoint_left.mpr fun k hk hk' => ?_
    obtain ⟨e, he, hek⟩ := (Finset.mem_filter.mp hk).2
    obtain ⟨e', he', hek'⟩ := (Finset.mem_filter.mp hk').2
    have h1 := Finset.mem_filter.mp he
    have h2 := Finset.mem_filter.mp he'
    refine hek' (hpure t e h1.1 t' e' h2.1 ?_ k hek)
    have e1 : e.1 = q := h1.2
    have e2 : e'.1 = q' := h2.2
    rw [e1, e2]
    exact hne
  have hcard : ∀ t, (cor t).card = ∑ q, (fib ⟨t, q⟩).card := fun t =>
    Finset.card_eq_sum_card_fiberwise (f := fun e : DCells (n t) (n' t) (B t) => e.1) (s := cor t) (t := univ)
      fun _ _ => Finset.mem_univ _
  calc L * ∑ t, (cor t).card = ∑ ρ : Σ t, Fin (n t), L * (fib ρ).card := by
        rw [Fintype.sum_sigma, Finset.mul_sum]
        exact Finset.sum_congr rfl fun t _ => by rw [hcard t, Finset.mul_sum]
    _ ≤ ∑ ρ, c * (used ρ).card := Finset.sum_le_sum fun ρ _ => hq ρ
    _ = c * (univ.biUnion used).card := by rw [Finset.card_biUnion hdisj, Finset.mul_sum]
    _ ≤ c * (univ.filter fun j => P.isProd j = true).card := by
        refine Nat.mul_le_mul_left _ (Finset.card_le_card fun k hk => ?_)
        obtain ⟨ρ, -, hk⟩ := Finset.mem_biUnion.mp hk
        obtain ⟨e, -, hek⟩ := (Finset.mem_filter.mp hk).2
        exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, P.isProd_of_coef_ne_zero _ _ hek⟩

end OverlapRound

/-- **The window's rounds with overlapping stages** (Theorem K∥, RS-D1E r2 §2.2 and §2.5), a named hypothesis. Each
round is `St` stages of any shapes (stage `t`: `n t` rows, `B t` blocks of `L`, `n' t` columns), each with its own
draw of its design bits; the draws may overlap at any depth, since the round's run is one straight-line program over
every draw's bits (`prog w`), fixed before the round's first draw. The window's game is the spot-check game of the
prover's play over the round's draws (`OvPlay.game`): the check draws a cell-block among every stage's, and the
round's deficiency is the share of cell-blocks and draws at which the cell-block, on its own draw's bits with the
other draws' at theirs, is uncertified (`ovDef`). Along every play `x`:
- round `w`'s products are K∥'s: at every bit point of the other draws, products of two operands affine in draw
  `t`'s bits (`AffMulAt`), so a stage's operand may depend on other draws in any way;
- stage `t`'s error table is its committed prefix words (`word w`) minus their targets on draw `t`'s bits;
- at every bit point of the draws, certified words of distinct rows, of any stages, share no product (row-pure);
- the run is charged at least `κ` a product, each product once, less the noise reads `V`.

`RoundsHypKSt` is the sequential case with stages of one shape, where a stage's program may also branch on earlier
draws. -/
def RoundsHypKOv (Mc : Machine) (S : Service) (R St : ℕ) (Vmax δb δz κ : ℝ) (n : Fin St → ℕ) (L np : ℕ)
    (B n' : Fin St → ℕ) {Ar : Type} [CommRing Ar] (wt : Fin np → Ar) : Prop :=
  ∃ e : S.Seed ≃ Paths (OSpot St n L np B n') R, ∀ (Ω' : Type) [Fintype Ω']
    (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)), (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    ∀ r : Ω', ∃ (P : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => DBits (n t) L np (B t) (n' t)) Ar R)
      (Bind ZK : Fin R → Paths (OSpot St n L np B n') R → Prop) (V : Paths (OSpot St n L np B n') R → ℝ),
      (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (OBits St n L np B n') Ar (N w))
        (pc : Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar)
        (vq : Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar)
        (word : ∀ w, OCells St n n' B → Fin (N w)),
        (∀ w j, (prog w).isProd j = true → AffMulAt ((prog w).prod j)) ∧
        (∀ w c, P.errAt x w c = fun z => (prog w).val z (word w c) -
          dPrefix (pc w c.1) (vq w c.1) wt c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
        (∀ w (W : ODraws St n L np B n') c c',
          (∀ U : Finset (DBits (n c.1) L np (B c.1) (n' c.1)), U.card ≤ 2 →
            P.errAt x w c (mergeAt W c.1 (bitInd U)) = 0) →
          (∀ U : Finset (DBits (n c'.1) L np (B c'.1) (n' c'.1)), U.card ≤ 2 →
            P.errAt x w c' (mergeAt W c'.1 (bitInd U)) = 0) →
          (⟨c.1, c.2.1⟩ : Σ t, Fin (n t)) ≠ ⟨c'.1, c'.2.1⟩ → ∀ k,
          (prog w).coef (word w c) k ≠ 0 → (prog w).coef (word w c') k = 0) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

theorem card_oCells (St : ℕ) (n n' B : Fin St → ℕ) :
    Fintype.card (OCells St n n' B) = ∑ t, n t * n' t * B t := by
  simp [OCells, Fintype.card_sigma, Fintype.card_prod, mul_assoc]

/-- `roundsHyp_of_overlap` on one slice (`RoundsSlice`): one adversary at one value of its coins, its acceptance
`acc`, charge `cost` and breaks named, and `RoundsHypKOv`'s cost line at that slice. -/
theorem roundsSlice_of_overlap {R St : ℕ} {Vmax κ : ℝ} {n B n' : Fin St → ℕ} {L np : ℕ} {Ar : Type} [CommRing Ar]
    {wt : Fin np → Ar}
    (P : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => DBits (n t) L np (B t) (n' t)) Ar R)
    (acc : Paths (OSpot St n L np B n') R → Prop) (cost : Paths (OSpot St n L np B n') R → ℝ)
    (Bind ZK : Fin R → Paths (OSpot St n L np B n') R → Prop) (V : Paths (OSpot St n L np B n') R → ℝ)
    (hacc : ∀ x, acc x → P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (hV : ∀ x, V x ≤ Vmax)
    (hK : ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (OBits St n L np B n') Ar (N w))
        (pc : Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar)
        (vq : Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar)
        (word : ∀ w, OCells St n n' B → Fin (N w)),
        (∀ w j, (prog w).isProd j = true → AffMulAt ((prog w).prod j)) ∧
        (∀ w c, P.errAt x w c = fun z => (prog w).val z (word w c) -
          dPrefix (pc w c.1) (vq w c.1) wt c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
        (∀ w (W : ODraws St n L np B n') c c',
          (∀ U : Finset (DBits (n c.1) L np (B c.1) (n' c.1)), U.card ≤ 2 →
            P.errAt x w c (mergeAt W c.1 (bitInd U)) = 0) →
          (∀ U : Finset (DBits (n c'.1) L np (B c'.1) (n' c'.1)), U.card ≤ 2 →
            P.errAt x w c' (mergeAt W c'.1 (bitInd U)) = 0) →
          (⟨c.1, c.2.1⟩ : Σ t, Fin (n t)) ≠ ⟨c'.1, c'.2.1⟩ → ∀ k,
          (prog w).coef (word w c) k ≠ 0 → (prog w).coef (word w c') k = 0) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ cost x)
    (hc : ∀ t, ChainBound Ar (n t) L np (B t) (n' t) wt (fun π => AffProd π) 1) (hκ : 0 ≤ κ) :
    RoundsSlice (3 / 4) (κ * L * (Fintype.card (OCells St n n' B) : ℝ)) 0 Vmax acc cost Bind ZK := by
  classical
  refine ⟨P.game, V, RoundGame.expBound_of_pathBound _
    (P.game.pathBound_of_passBound (P.passBound_game fun x w c U => ?_)), hacc, hV, fun x => ?_⟩
  · obtain ⟨N, prog, pc, vq, word, hOk, hE, -, -⟩ := hK x
    rw [hE w c]
    refine ((prog w).cubeQuad_val_mergeAt (hOk w) U c.1 (word w c)).sub
      ((cubeQuad_dPrefix (pc w c.1) (vq w c.1) wt c.2.1 c.2.2.1 c.2.2.2.val).congr fun z => ?_)
    congr 1
    funext v
    rw [mergeAt_mk]
  · obtain ⟨N, prog, pc, vq, word, hOk, hE, hpure, hcharge⟩ := hK x
    set C : ℝ := (Fintype.card (OCells St n n' B) : ℝ)
    let cert : Fin R → ODraws St n L np B n' → OCells St n n' B → Prop := fun w W c =>
      ∀ U : Finset (DBits (n c.1) L np (B c.1) (n' c.1)), U.card ≤ 2 → P.errAt x w c (mergeAt W c.1 (bitInd U)) = 0
    have hprod : ∀ w W, (L : ℝ) * ((univ.filter fun c => cert w W c).card : ℝ) ≤
        ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) := fun w W => by
      let cor : ∀ t, Finset (DCells (n t) (n' t) (B t)) := fun t => univ.filter fun c => cert w W ⟨t, c⟩
      have := (prog w).cells_card_le_ov hc (hOk w) W (pc w) (vq w) (word w) cor
        (fun t c hc U hU => by
          have h1 := (Finset.mem_filter.mp hc).2 U hU
          rw [hE w ⟨t, c⟩] at h1
          have h2 : (fun v => mergeAt W t (bitInd U : DBits (n t) L np (B t) (n' t) → Ar) ⟨t, v⟩) = bitInd U :=
            funext fun v => mergeAt_mk W t _ v
          simp only at h1
          rw [h2] at h1
          exact sub_eq_zero.mp h1)
        fun t c hc t' c' hc' hne k hk =>
          hpure w W ⟨t, c⟩ ⟨t', c'⟩ (Finset.mem_filter.mp hc).2 (Finset.mem_filter.mp hc').2 hne k hk
      have hs : (univ.filter fun c => cert w W c) = univ.sigma cor := by
        ext ⟨t, c⟩
        simp [cor]
      rw [one_mul] at this
      rw [hs, Finset.card_sigma]
      exact_mod_cast this
    have hcor : ∀ w, C * (1 - ovDef (P.errAt x w)) =
        expect fun W : ODraws St n L np B n' => ((univ.filter fun c => cert w W c).card : ℝ) := fun w => by
      have hd : ovDef (P.errAt x w) = expect fun W : ODraws St n L np B n' =>
          prob fun c : OCells St n n' B => ¬cert w W c := by
        rw [ovDef, prob_eq_expect, expect_prod_swap]
        congr 1
        funext W
        rw [prob_eq_expect]
        congr 1
        funext c
        simp only [cert, not_forall, exists_prop]
      have hW : ∀ W : ODraws St n L np B n', ((univ.filter fun c => cert w W c).card : ℝ) =
          C + -C * prob fun c : OCells St n n' B => ¬cert w W c := fun W => by
        rw [prob_eq_card]
        have hsplit := Finset.card_filter_add_card_filter_not (s := (univ : Finset (OCells St n n' B)))
          fun c => cert w W c
        rw [Finset.card_univ] at hsplit
        rcases Nat.eq_zero_or_pos (Fintype.card (OCells St n n' B)) with h0 | h0
        · have : (univ.filter fun c => cert w W c).card = 0 := by omega
          simp [C, h0, this]
        · have hpos : (0 : ℝ) < Fintype.card (OCells St n n' B) := by exact_mod_cast h0
          have hs' : ((univ.filter fun c => cert w W c).card : ℝ) + ((univ.filter fun c => ¬cert w W c).card : ℝ) =
              Fintype.card (OCells St n n' B) := by exact_mod_cast hsplit
          simp only [C]
          field_simp
          linarith
      rw [hd, funext hW, expect_const_add_mul]
      ring
    have hdef : P.game.deficiency x = ∑ w, ovDef (P.errAt x w) := by
      rw [P.game.deficiency_eq_sum]
      exact Finset.sum_congr rfl fun w _ => P.dAt_game x w
    calc ((R : ℝ) - P.game.deficiency x) * (κ * L * C) - V x - R * 0
        = κ * ∑ w, (L : ℝ) * (C * (1 - ovDef (P.errAt x w))) - V x := by
          rw [hdef]
          simp only [mul_sub, mul_one, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul, ← Finset.mul_sum]
          ring
      _ ≤ κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x := by
        gcongr with w
        rw [hcor w, ← expect_mul_left]
        exact expect_le_of_le (Nat.cast_nonneg _) fun W => hprod w W
      _ ≤ cost x := hcharge

/-- **Overlapping stages give the window's cost line** (Theorem K∥): under `RoundsHypKOv` with chain factor 1 on
every stage's shape, `RoundsHyp` holds at `p = ¾` with `F₀ = κ L C`, `C` the round's cell-blocks over all its
stages, and no slack. The luck is per draw, with the other draws' bits given (`OvPlay.passBound_game`), and the
certified count is `SlpG.cells_card_le_ov`'s at every bit point of the draws, so at least its mean
(`roundsSlice_of_overlap`, slice by slice). -/
theorem roundsHyp_of_overlap {Mc : Machine} {S : Service} {R St : ℕ} {Vmax δb δz κ : ℝ} {n B n' : Fin St → ℕ}
    {L np : ℕ} {Ar : Type} [CommRing Ar] {wt : Fin np → Ar} (h : RoundsHypKOv Mc S R St Vmax δb δz κ n L np B n' wt)
    (hc : ∀ t, ChainBound Ar (n t) L np (B t) (n' t) wt (fun π => AffProd π) 1) (hκ : 0 ≤ κ) :
    RoundsHyp Mc S (OSpot St n L np B n') R (3 / 4) (κ * L * (Fintype.card (OCells St n n' B) : ℝ)) 0 Vmax δb δz := by
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨P, Bind, ZK, V, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  obtain ⟨G, V', hG, hacc', hV', hcost⟩ := roundsSlice_of_overlap P
    (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
    (fun x => Mc.charge M (e.symm x, r)) Bind ZK V hacc hV hK hc hκ
  exact ⟨G, Bind, ZK, V', hG, hB, hZ, hacc', hV', hcost⟩

section ParityOverlapRound

variable {St : ℕ} {Ar : Type} [CommRing Ar] {n B n' : Fin St → ℕ} {L np : ℕ}

/-- **An overlapped one-sided run's error has a parity form at every cell-block's frame**, at every bit point `W` of
the other draws: its terms the products charged to the word there. -/
theorem SlpG.parityForm_err_ov (φ : Ar →+* ZMod 2) {p₀ : Fin np} {w : Fin np → Ar} (hw : φ (w p₀) = 1) {N : ℕ}
    (P : SlpG (OBits St n L np B n') Ar N) (hOk : ∀ j, P.isProd j = true → OneSidedAt (P.prod j))
    (W : ODraws St n L np B n') (pc : ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar)
    (vq : ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar) (j : Fin N) (c : OCells St n n' B) :
    ParityForm (dFrame p₀ c.2) φ (charged (dFrame p₀ c.2) φ (P.coef j) fun i z => P.prod i (mergeAt W c.1 z)).card
      fun z => P.val (mergeAt W c.1 z) j - dPrefix (pc c.1) (vq c.1) w c.2.1 c.2.2.1 c.2.2.2.val z := by
  have h := parityForm_sum (dFrame_injective p₀ c.2) φ (P.coef j) (fun i z => P.prod i (mergeAt W c.1 z))
    ((P.aff j).restrictAt W c.1)
    (fun i hi => OneSided.splitProd φ p₀ c.2 (hOk i (P.isProd_of_coef_ne_zero j i hi) W c.1)) _
    (split_dPrefix φ hw (pc c.1) (vq c.1) c.2)
  have e : (fun z => P.val (mergeAt W c.1 z) j - dPrefix (pc c.1) (vq c.1) w c.2.1 c.2.2.1 c.2.2.2.val z) =
      fun z => ∑ i, P.coef j i * P.prod i (mergeAt W c.1 z) + ((P.aff j).restrictAt W c.1).eval z -
        dPrefix (pc c.1) (vq c.1) w c.2.1 c.2.2.1 c.2.2.2.val z :=
    funext fun z => by rw [P.val_eq, BitAff.eval_mergeAt]
  rw [e]
  exact h

open Classical in
/-- **Charge-pure one-sided overlapped runs add up**: at every bit point of the draws, the round's run has at least
`L` products per cell-block, of any stage, not short of products. -/
theorem SlpG.card_full_le_ov (φ : Ar →+* ZMod 2) {p₀ : Fin np} {w : Fin np → Ar} (hw : φ (w p₀) = 1) {N : ℕ}
    (P : SlpG (OBits St n L np B n') Ar N) (hOk : ∀ j, P.isProd j = true → OneSidedAt (P.prod j))
    (W : ODraws St n L np B n') (pc : ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar)
    (vq : ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar) (word : OCells St n n' B → Fin N)
    (hpure : P.ChargePureAt φ p₀ word) :
    L * (univ.filter fun c : OCells St n n' B => ¬ParShort (dFrame p₀ c.2) φ
      fun z => P.val (mergeAt W c.1 z) (word c) - dPrefix (pc c.1) (vq c.1) w c.2.1 c.2.2.1 c.2.2.2.val z).card ≤
      (univ.filter fun j => P.isProd j = true).card := by
  refine _root_.Pouw.RelativeSecurity.card_full_le
    (fun c => charged (dFrame p₀ c.2) φ (P.coef (word c)) fun i z => P.prod i (mergeAt W c.1 z)) _
    (fun c i hi => mem_filter.mpr ⟨mem_univ _, P.isProd_of_coef_ne_zero _ _ (ne_zero_of_map (mem_filter.mp hi).2.1)⟩)
    (fun c c' hne => disjoint_left.mpr fun i hi hi' =>
      hpure W c c' i hne (mem_filter.mp hi).2.1 (mem_filter.mp hi).2.2 (mem_filter.mp hi').2.1
        (mem_filter.mp hi').2.2) _ fun c hc => ?_
  by_contra hlt
  exact hc ⟨_, not_le.mp hlt, P.parityForm_err_ov φ hw hOk W pc vq (word c) c⟩

end ParityOverlapRound

/-- A share's complement counts the cells outside it. -/
theorem card_not_ge {C : Type} [Fintype C] (Q : C → Prop) [DecidablePred Q] :
    (Fintype.card C : ℝ) * (1 - prob Q) ≤ (univ.filter fun c => ¬Q c).card := by
  rw [prob_eq_card]
  have hsplit := Finset.card_filter_add_card_filter_not (s := (univ : Finset C)) Q
  rw [Finset.card_univ] at hsplit
  rcases Nat.eq_zero_or_pos (Fintype.card C) with h0 | h0
  · rw [h0]
    simp
  · have hpos : (0 : ℝ) < Fintype.card C := by exact_mod_cast h0
    have hs : ((univ.filter Q).card : ℝ) + ((univ.filter fun c => ¬Q c).card : ℝ) = Fintype.card C := by
      exact_mod_cast hsplit
    rw [mul_sub, mul_one, mul_div_cancel₀ _ hpos.ne']
    linarith

/-- **The window's staged rounds, one-sided operands of any form** (RS-E-REF r6 VI.2), a named hypothesis:
`RoundsHypKSt` with its products one-sided of any form (`OneSided`) in place of affine, and charging purity
(`SlpG.ChargePure`, through a ring map `φ` onto `F₂` at plane `p₀`) in place of row purity. -/
def RoundsHypKStP (Mc : Machine) (S : Service) (R St : ℕ) (Vmax δb δz κ : ℝ) (n L np B n' : ℕ) {Ar : Type}
    [CommRing Ar] (wt : Fin np → Ar) (φ : Ar →+* ZMod 2) (p₀ : Fin np) : Prop :=
  ∃ e : S.Seed ≃ Paths (DStage St n L np B n') R, ∀ (Ω' : Type) [Fintype Ω']
    (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)), (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    ∀ r : Ω', ∃ (P : StagedPlay (DCells n n' B) (DBits n L np B n') Ar St R)
      (Bind ZK : Fin R → Paths (DStage St n L np B n') R → Prop) (V : Paths (DStage St n L np B n') R → ℝ),
      (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → Fin St → ℕ) (prog : ∀ w s, SlpG (DBits n L np B n') Ar (N w s))
        (pc : Fin R → Fin St → Fin n → Fin B → Fin L → Ar)
        (vq : Fin R → Fin St → Fin B → Fin L → Fin n' → Ar)
        (word : ∀ w s, DCells n n' B → Fin (N w s)),
        (∀ w s j, (prog w s).isProd j = true → OneSided ((prog w s).prod j)) ∧
        (∀ w s c, P.errAt x w s c =
          fun z => (prog w s).val z (word w s c) - dPrefix (pc w s) (vq w s) wt c.1 c.2.1 c.2.2.val z) ∧
        (∀ w s, (prog w s).ChargePure φ p₀ (word w s)) ∧
        κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

/-- `roundsHyp_of_stagedP` on one slice (`RoundsSlice`): one adversary at one value of its coins, its acceptance
`acc`, charge `cost` and breaks named, and `RoundsHypKStP`'s cost line at that slice. -/
theorem roundsSlice_of_stagedP {R St : ℕ} {Vmax κ : ℝ} {n L np B n' : ℕ} {Ar : Type} [CommRing Ar]
    {wt : Fin np → Ar} {φ : Ar →+* ZMod 2} {p₀ : Fin np}
    (P : StagedPlay (DCells n n' B) (DBits n L np B n') Ar St R) (acc : Paths (DStage St n L np B n') R → Prop)
    (cost : Paths (DStage St n L np B n') R → ℝ) (Bind ZK : Fin R → Paths (DStage St n L np B n') R → Prop)
    (V : Paths (DStage St n L np B n') R → ℝ) (hacc : ∀ x, acc x → P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x)
    (hV : ∀ x, V x ≤ Vmax)
    (hK : ∀ x, ∃ (N : Fin R → Fin St → ℕ) (prog : ∀ w s, SlpG (DBits n L np B n') Ar (N w s))
        (pc : Fin R → Fin St → Fin n → Fin B → Fin L → Ar)
        (vq : Fin R → Fin St → Fin B → Fin L → Fin n' → Ar)
        (word : ∀ w s, DCells n n' B → Fin (N w s)),
        (∀ w s j, (prog w s).isProd j = true → OneSided ((prog w s).prod j)) ∧
        (∀ w s c, P.errAt x w s c =
          fun z => (prog w s).val z (word w s c) - dPrefix (pc w s) (vq w s) wt c.1 c.2.1 c.2.2.val z) ∧
        (∀ w s, (prog w s).ChargePure φ p₀ (word w s)) ∧
        κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x ≤ cost x)
    (hw : φ (wt p₀) = 1) (hSt : 0 < St) (hκ : 0 ≤ κ) :
    RoundsSlice (427 / 500) (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) 0 Vmax acc cost Bind ZK := by
  classical
  refine ⟨P.gameP (dFrame p₀) φ, V, (P.gameP (dFrame p₀) φ).expBound_of_staged (by norm_num)
    (P.game_stagedP (fun c => dFrame_injective p₀ c) φ),
    fun x hx => (hacc x hx).imp_left (P.accepts_gameP (dFrame p₀) φ x).mpr, hV, fun x => ?_⟩
  obtain ⟨N, prog, pc, vq, word, hOk, hE, hpure, hcharge⟩ := hK x
  set C : ℝ := ((n * n' * B : ℕ) : ℝ)
  have hC : C = (Fintype.card (DCells n n' B) : ℝ) := by
    simp [C, Fintype.card_prod, mul_assoc]
  have hprod : ∀ w s, (L : ℝ) * (C * (1 - parDef (dFrame p₀) φ (P.errAt x w s))) ≤
      ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) := fun w s => by
    have h1 := (prog w s).card_full_le φ hw (hOk w s) (pc w s) (vq w s) (word w s) (hpure w s)
    have h2 := card_not_ge fun c => ParShort (dFrame p₀ c) φ (P.errAt x w s c)
    rw [hC, parDef]
    refine (mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg L)).trans ?_
    simp only [hE w s] at h1 ⊢
    exact_mod_cast h1
  have hSt' : (St : ℝ) ≠ 0 := by exact_mod_cast hSt.ne'
  have hdef : (P.gameP (dFrame p₀) φ).deficiency x =
      (St : ℝ)⁻¹ * ∑ w, ∑ s, parDef (dFrame p₀) φ (P.errAt x w s) := by
    rw [RoundGame.deficiency_eq_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun w _ => ?_
    rw [P.dAt_gameP (dFrame p₀) φ x w, Finset.mul_sum]
  have hsum : ((R : ℝ) - (P.gameP (dFrame p₀) φ).deficiency x) * (St * (κ * L * C)) =
      κ * ∑ w, ∑ s, (L : ℝ) * (C * (1 - parDef (dFrame p₀) φ (P.errAt x w s))) := by
    have e2 : ∑ w : Fin R, ∑ s : Fin St, (L : ℝ) * (C * (1 - parDef (dFrame p₀) φ (P.errAt x w s))) =
        L * C * (R * St - ∑ w, ∑ s, parDef (dFrame p₀) φ (P.errAt x w s)) := by
      simp only [mul_sub, mul_one, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, ← Finset.mul_sum]
      ring
    rw [hdef, e2]
    field_simp
  calc ((R : ℝ) - (P.gameP (dFrame p₀) φ).deficiency x) * (St * (κ * L * C)) - V x - R * 0
      = κ * ∑ w, ∑ s, (L : ℝ) * (C * (1 - parDef (dFrame p₀) φ (P.errAt x w s))) - V x := by rw [hsum]; ring
    _ ≤ κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x := by
      gcongr with w _ s
      exact hprod w s
    _ ≤ cost x := hcharge

/-- **Staged rounds of one-sided operands of any form give the window's cost line** (RS-E-REF r6 VI.2): under
`RoundsHypKStP` with `φ (wt p₀) = 1`, `RoundsHyp` holds at `p = 0.854` with `F₀ = St κ L n n' B` and no slack. The
game is the product-deficit game (`StagedPlay.gameP`, accepting as `game` does), whose rounds are staged at `0.854`
whatever the errors (`StagedPlay.game_stagedP`); each stage's cell-blocks not short of products are counted by
`SlpG.card_full_le` (`roundsSlice_of_stagedP`, slice by slice). -/
theorem roundsHyp_of_stagedP {Mc : Machine} {S : Service} {R St : ℕ} {Vmax δb δz κ : ℝ} {n L np B n' : ℕ}
    {Ar : Type} [CommRing Ar] {wt : Fin np → Ar} {φ : Ar →+* ZMod 2} {p₀ : Fin np}
    (h : RoundsHypKStP Mc S R St Vmax δb δz κ n L np B n' wt φ p₀) (hw : φ (wt p₀) = 1) (hSt : 0 < St)
    (hκ : 0 ≤ κ) :
    RoundsHyp Mc S (DStage St n L np B n') R (427 / 500) (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) 0 Vmax δb δz := by
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨P, Bind, ZK, V, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  obtain ⟨G, V', hG, hacc', hV', hcost⟩ := roundsSlice_of_stagedP P
    (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
    (fun x => Mc.charge M (e.symm x, r)) Bind ZK V hacc hV hK hw hSt hκ
  exact ⟨G, Bind, ZK, V', hG, hB, hZ, hacc', hV', hcost⟩

/-- **The window's rounds with overlapping stages, one-sided operands of any form** (RS-E-REF r6 VI.2), a named
hypothesis: `RoundsHypKOv` with its products one-sided per draw at every bit point of the other draws (`OneSidedAt`)
in place of `AffMulAt`, and charging purity across draws (`SlpG.ChargePureAt`) in place of row purity. -/
def RoundsHypKOvP (Mc : Machine) (S : Service) (R St : ℕ) (Vmax δb δz κ : ℝ) (n : Fin St → ℕ) (L np : ℕ)
    (B n' : Fin St → ℕ) {Ar : Type} [CommRing Ar] (wt : Fin np → Ar) (φ : Ar →+* ZMod 2) (p₀ : Fin np) : Prop :=
  ∃ e : S.Seed ≃ Paths (OSpot St n L np B n') R, ∀ (Ω' : Type) [Fintype Ω']
    (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)), (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    ∀ r : Ω', ∃ (P : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => DBits (n t) L np (B t) (n' t)) Ar R)
      (Bind ZK : Fin R → Paths (OSpot St n L np B n') R → Prop) (V : Paths (OSpot St n L np B n') R → ℝ),
      (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (OBits St n L np B n') Ar (N w))
        (pc : Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar)
        (vq : Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar)
        (word : ∀ w, OCells St n n' B → Fin (N w)),
        (∀ w j, (prog w).isProd j = true → OneSidedAt ((prog w).prod j)) ∧
        (∀ w c, P.errAt x w c = fun z => (prog w).val z (word w c) -
          dPrefix (pc w c.1) (vq w c.1) wt c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
        (∀ w, (prog w).ChargePureAt φ p₀ (word w)) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

/-- `roundsHyp_of_overlapP` on one slice (`RoundsSlice`): one adversary at one value of its coins, its acceptance
`acc`, charge `cost` and breaks named, and `RoundsHypKOvP`'s cost line at that slice. -/
theorem roundsSlice_of_overlapP {R St : ℕ} {Vmax κ : ℝ} {n B n' : Fin St → ℕ} {L np : ℕ} {Ar : Type}
    [CommRing Ar] {wt : Fin np → Ar} {φ : Ar →+* ZMod 2} {p₀ : Fin np}
    (P : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => DBits (n t) L np (B t) (n' t)) Ar R)
    (acc : Paths (OSpot St n L np B n') R → Prop) (cost : Paths (OSpot St n L np B n') R → ℝ)
    (Bind ZK : Fin R → Paths (OSpot St n L np B n') R → Prop) (V : Paths (OSpot St n L np B n') R → ℝ)
    (hacc : ∀ x, acc x → P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (hV : ∀ x, V x ≤ Vmax)
    (hK : ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (OBits St n L np B n') Ar (N w))
        (pc : Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar)
        (vq : Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar)
        (word : ∀ w, OCells St n n' B → Fin (N w)),
        (∀ w j, (prog w).isProd j = true → OneSidedAt ((prog w).prod j)) ∧
        (∀ w c, P.errAt x w c = fun z => (prog w).val z (word w c) -
          dPrefix (pc w c.1) (vq w c.1) wt c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
        (∀ w, (prog w).ChargePureAt φ p₀ (word w)) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ cost x)
    (hw : φ (wt p₀) = 1) (hκ : 0 ≤ κ) :
    RoundsSlice (427 / 500) (κ * L * (Fintype.card (OCells St n n' B) : ℝ)) 0 Vmax acc cost Bind ZK := by
  classical
  let fr : (c : OCells St n n' B) → Fin L ⊕ Fin L → DBits (n c.1) L np (B c.1) (n' c.1) := fun c => dFrame p₀ c.2
  refine ⟨P.gameP fr φ, V, RoundGame.expBound_of_pathBound _
    ((P.gameP fr φ).pathBound_of_passBound (P.passBound_gameP (fun c => dFrame_injective p₀ c.2) φ)),
    fun x hx => (hacc x hx).imp_left (P.accepts_gameP fr φ x).mpr, hV, fun x => ?_⟩
  obtain ⟨N, prog, pc, vq, word, hOk, hE, hpure, hcharge⟩ := hK x
  set C : ℝ := (Fintype.card (OCells St n n' B) : ℝ)
  let full : Fin R → ODraws St n L np B n' → OCells St n n' B → Prop := fun w W c =>
    ¬ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z)
  have hprod : ∀ w W, (L : ℝ) * ((univ.filter fun c => full w W c).card : ℝ) ≤
      ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) := fun w W => by
    have h1 := (prog w).card_full_le_ov φ hw (hOk w) W (pc w) (vq w) (word w) (hpure w)
    simp only [full, fr, hE, mergeAt_mk]
    exact_mod_cast h1
  have hcor : ∀ w, C * (1 - ovDefP fr φ (P.errAt x w)) ≤
      expect fun W : ODraws St n L np B n' => ((univ.filter fun c => full w W c).card : ℝ) := fun w => by
    have hd : ovDefP fr φ (P.errAt x w) = expect fun W : ODraws St n L np B n' =>
        prob fun c : OCells St n n' B => ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z) := by
      rw [ovDefP, prob_eq_expect, expect_prod_swap]
      congr 1
      funext W
      rw [prob_eq_expect]
    have ha : C * (1 - ovDefP fr φ (P.errAt x w)) = expect fun W : ODraws St n L np B n' =>
        C + -C * prob fun c : OCells St n n' B => ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z) := by
      rw [hd, expect_const_add_mul]
      ring
    rw [ha]
    refine expect_mono fun W => ?_
    have := card_not_ge fun c : OCells St n n' B => ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z)
    simp only [full]
    linarith
  have hdef : (P.gameP fr φ).deficiency x = ∑ w, ovDefP fr φ (P.errAt x w) := by
    rw [RoundGame.deficiency_eq_sum]
    exact Finset.sum_congr rfl fun w _ => P.dAt_gameP fr φ x w
  calc ((R : ℝ) - (P.gameP fr φ).deficiency x) * (κ * L * C) - V x - R * 0
      = κ * ∑ w, (L : ℝ) * (C * (1 - ovDefP fr φ (P.errAt x w))) - V x := by
        rw [hdef]
        simp only [mul_sub, mul_one, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul, ← Finset.mul_sum]
        ring
    _ ≤ κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x := by
      gcongr with w
      refine (mul_le_mul_of_nonneg_left (hcor w) (Nat.cast_nonneg L)).trans ?_
      rw [← expect_mul_left]
      exact expect_le_of_le (Nat.cast_nonneg _) fun W => hprod w W
    _ ≤ cost x := hcharge

/-- **Overlapping stages of one-sided operands of any form give the window's cost line** (RS-E-REF r6 VI.2): under
`RoundsHypKOvP` with `φ (wt p₀) = 1`, `RoundsHyp` holds at `p = 0.854` with `F₀ = κ L C`, `C` the round's
cell-blocks over all its stages, and no slack. The game is the product-deficit game (`OvPlay.gameP`), whose rounds
pass at most `1 − 0.146 d` whatever the errors (`OvPlay.passBound_gameP`); the count is `SlpG.card_full_le_ov`'s at
every bit point of the draws, so at least its mean (`roundsSlice_of_overlapP`, slice by slice). -/
theorem roundsHyp_of_overlapP {Mc : Machine} {S : Service} {R St : ℕ} {Vmax δb δz κ : ℝ} {n B n' : Fin St → ℕ}
    {L np : ℕ} {Ar : Type} [CommRing Ar] {wt : Fin np → Ar} {φ : Ar →+* ZMod 2} {p₀ : Fin np}
    (h : RoundsHypKOvP Mc S R St Vmax δb δz κ n L np B n' wt φ p₀) (hw : φ (wt p₀) = 1) (hκ : 0 ≤ κ) :
    RoundsHyp Mc S (OSpot St n L np B n') R (427 / 500) (κ * L * (Fintype.card (OCells St n n' B) : ℝ)) 0 Vmax δb
      δz := by
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨P, Bind, ZK, V, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  obtain ⟨G, V', hG, hacc', hV', hcost⟩ := roundsSlice_of_overlapP P
    (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
    (fun x => Mc.charge M (e.symm x, r)) Bind ZK V hacc hV hK hw hκ
  exact ⟨G, Bind, ZK, V', hG, hB, hZ, hacc', hV', hcost⟩

/-- **The window's rounds with Theorem H's cost line** (the re-encoded design in exact arithmetic over a field `F`:
FP8 operands or scaled reads, with exact sums), a named hypothesis. It is `RoundsHyp` with the cost line derived
rather than assumed. Along every play `x`:
- round `w`'s run is a quad run over `F` on the round's bits (`P w`: Theorem J″'s persistent quads, cell-pure), on
  committed constants `pc w` and `vq w`, whose full lanes and merged cells hold cell-blocks (`cell w`, `cellM w`);
- its full lanes and merged cells number at least `(1 − d_w) C`, `C = n n' B` the round's cell-blocks;
- the run is charged at least `κ` a slot, at 4 slots a quad's slice and `h` a merged cell, less the noise reads.

Not needed for the re-encoded design: `RoundsHypK` over `ℚ` gives the same floor with merges free
(`certifiedFraction120θ_reEncodedExactK_40`); this is the alternative with merges priced. -/
def RoundsHypH (Mc : Machine) (S : Service) (Ω : Type) [Fintype Ω] (R : ℕ) (p Vmax δb δz κ h : ℝ)
    (F : Type) [Field F] (n L np B n' : ℕ) (wt : Fin np → F) : Prop :=
  ∃ e : S.Seed ≃ Paths Ω R, ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
    ∃ (G : RoundGame Ω R) (Bind ZK : Fin R → Paths Ω R → Prop) (V : Paths Ω R → ℝ),
      G.PathBound p ∧ (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (nq nm : Fin R → ℕ) (P : ∀ w, QuadRun (Fin (nq w)) (Fin (nm w)) (DBits n L np B n') F)
        (pc : Fin R → Fin n → Fin B → Fin L → F) (vq : Fin R → Fin B → Fin L → Fin n' → F)
        (cell : ∀ w, Fin (nq w) → Fin 4 → DCells n n' B) (cellM : ∀ w, Fin (nm w) → DCells n n' B),
        (∀ w, (P w).Computes (cell w) (fun c => dIncr (pc w) (vq w) wt c.1 c.2.1 c.2.2)
          (fun m => dIncr (pc w) (vq w) wt (cellM w m).1 (cellM w m).2.1 (cellM w m).2.2)) ∧
        (∀ w, ((n * n' * B : ℕ) : ℝ) * (1 - G.dAt x w) ≤
          ∑ q, ((univ.filter fun ℓ => (P w).lane q ℓ = .full).card : ℝ) + nm w) ∧
        κ * ∑ w, (∑ q, 4 * ((P w).s q : ℝ) + h * nm w) - V x ≤ Mc.charge M (e.symm x, r)

/-- **Theorem H gives the window's cost line** in exact arithmetic: under `RoundsHypH` with plane `p₀` at weight 1
and merges at `h ≥ 2L/3` slots, `RoundsHyp` holds with `F₀ = κ L n n' B` and no slack, as from Theorem K. -/
theorem roundsHyp_of_H {Mc : Machine} {S : Service} {Ω : Type} [Fintype Ω] {R : ℕ} {p Vmax δb δz κ h : ℝ}
    {F : Type} [Field F] {n L np B n' : ℕ} {wt : Fin np → F} {p₀ : Fin np}
    (hH : RoundsHypH Mc S Ω R p Vmax δb δz κ h F n L np B n' wt) (hwt : wt p₀ = 1) (hκ : 0 ≤ κ)
    (hh : 2 * (L : ℝ) / 3 ≤ h) :
    RoundsHyp Mc S Ω R p (κ * L * ((n * n' * B : ℕ) : ℝ)) 0 Vmax δb δz := by
  obtain ⟨e, hH⟩ := hH
  refine ⟨e, fun Ω' _ M hc r => ?_⟩
  obtain ⟨G, Bind, ZK, V, hG, hB, hZ, hacc, hV, hK⟩ := hH Ω' M hc r
  refine ⟨G, Bind, ZK, V, G.expBound_of_pathBound hG, hB, hZ, hacc, hV, fun x => ?_⟩
  obtain ⟨nq, nm, P, pc, vq, cell, cellM, hC, hcor, hcharge⟩ := hK x
  have hcost : ∀ w, (L : ℝ) * (∑ q, ((univ.filter fun ℓ => (P w).lane q ℓ = .full).card : ℝ) + nm w) ≤
      ∑ q, 4 * ((P w).s q : ℝ) + h * nm w := fun w => by
    have := QuadRun.cost_ge_reEncoded (pc w) (vq w) wt hwt (cell w) (cellM w) (hC w) h hh
    rwa [Fintype.card_fin] at this
  set C : ℝ := ((n * n' * B : ℕ) : ℝ)
  have hsum : ((R : ℝ) - G.deficiency x) * (κ * L * C) = κ * ∑ w, (L : ℝ) * (C * (1 - G.dAt x w)) := by
    rw [G.deficiency_eq_sum, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
    ring
  calc ((R : ℝ) - G.deficiency x) * (κ * L * C) - V x - R * 0
      = κ * ∑ w, (L : ℝ) * (C * (1 - G.dAt x w)) - V x := by rw [hsum]; ring
    _ ≤ κ * ∑ w, (∑ q, 4 * ((P w).s q : ℝ) + h * nm w) - V x := by
      gcongr with w
      exact (mul_le_mul_of_nonneg_left (hcor w) (Nat.cast_nonneg L)).trans (hcost w)
    _ ≤ Mc.charge M (e.symm x, r) := hcharge

/-- The re-encoded design's plane weights over `ℚ` (exact arithmetic). -/
def wReQ : Fin 7 → ℚ := ![1, 2, 4, 8, 16, 32, 65]

theorem wReQ_zero : wReQ 0 = 1 := rfl

section ModelB

/-! ## The Model B bridge (RS-MODEL, `gpu-model/rs-model.lean`)

Model B prices a straight-line gate program by the sum of its gates' prices (`timeB`, additive) and never reads its
wiring (`timeB_rewire`). A Model B algorithm `I → O` is a typed circuit (`Circ`) of gates and routing: its semantics
is gate evaluation and its time is `timeB` of its gate list. `machine` is the `Machine`; `matMachine` is the
`MatMachine` over ℤ given circuits for the products, sums and differences at their prices (`MatGates`), which a gate
family with one GEMM kind per shape provides (`gemmGates`). On sm_120 (`price120`) the worst case is `κ = 16` u per
product (s32 operands as 4 × 4 int8 limb products) and `κadd = 1` u per entry; on the int8 grid and the tile shapes the
product is `κG = 1` u (`gridMatMachine120`). -/

namespace ModelB

/-- RS-MODEL's gate kind: arity, outputs, the operand range on which it is well-formed (`ok`), and its exact semantics
there. -/
structure Kind where
  arity : ℕ
  outs : ℕ
  ok : (Fin arity → ℤ) → Bool
  sem : (Fin arity → ℤ) → Fin outs → ℤ

/-- RS-MODEL's gate: a kind, the wires it reads, its round. -/
structure GateApp (K : Type) where
  kind : K
  args : List ℕ
  round : ℕ := 0

/-- RS-MODEL's straight-line program. -/
abbrev StraightLine (K : Type) : Type := List (GateApp K)

variable {K : Type}

/-- RS-MODEL's `time_B`: the sum of the gates' prices (on sm_120 in u = 1/R* SMSP-cycles). -/
def timeB (price : K → ℕ) (P : StraightLine K) : ℕ := (P.map fun g => price g.kind).sum

theorem timeB_append (price : K → ℕ) (P Q : StraightLine K) : timeB price (P ++ Q) = timeB price P + timeB price Q := by
  simp [timeB]

/-- RS-MODEL's `timeB_rewire`: rewiring never changes time. -/
theorem timeB_rewire (price : K → ℕ) (P : StraightLine K) (f : GateApp K → List ℕ) :
    timeB price (P.map fun g => { g with args := f g }) = timeB price P := by
  simp [timeB, Function.comp_def]

/-- The program of a gate list, with its wiring left out (Model B's time doesn't read it, `timeB_rewire`). -/
def progOf (l : List K) : StraightLine K := l.map fun k => { kind := k, args := [] }

theorem timeB_progOf_append (price : K → ℕ) (l m : List K) :
    timeB price (progOf (l ++ m)) = timeB price (progOf l) + timeB price (progOf m) := by
  rw [progOf, List.map_append, timeB_append]
  rfl

theorem timeB_progOf (price : K → ℕ) (l : List K) : timeB price (progOf l) = (l.map price).sum := by
  simp [timeB, progOf, Function.comp_def]

/-- An additive price on gate lists, nonnegative and additive under concatenation: RS-MODEL's `timeB` at a per-kind
price in ℕ (`Price.ofNat`), or the sum of a rational per-kind price (`Price.ofRat`, for `B_θ`). -/
structure Price (K : Type) where
  t : List K → ℝ
  nonneg : ∀ l, 0 ≤ t l
  append : ∀ l m, t (l ++ m) = t l + t m

theorem Price.t_nil (p : Price K) : p.t [] = 0 := by
  have h := p.append [] []
  rw [List.nil_append] at h
  linarith

/-- RS-MODEL's `timeB` at a per-kind price in ℕ. -/
def Price.ofNat (price : K → ℕ) : Price K where
  t l := (timeB price (progOf l) : ℝ)
  nonneg _ := Nat.cast_nonneg _
  append l m := by rw [timeB_progOf_append, Nat.cast_add]

/-- The sum of a nonnegative rational per-kind price. -/
def Price.ofRat (price : K → ℚ) (h : ∀ k, 0 ≤ price k) : Price K where
  t l := (((l.map price).sum : ℚ) : ℝ)
  nonneg l := Rat.cast_nonneg.mpr (List.sum_nonneg fun x hx => by
    obtain ⟨k, -, rfl⟩ := List.mem_map.mp hx
    exact h k)
  append l m := by rw [List.map_append, List.sum_append, Rat.cast_add]

variable (kd : K → Kind)

/-- A typed Model B circuit. `gate k` is one gate of kind `k` (off its range `ok` it outputs 0, which only adds
programs); the rest is routing, which moves wires and computes nothing: `reindex` (read wires in any order, with
fan-out), `curry` and `uncurry` (reshape a block), `join` (two bundles side by side), `ofBits` (bit wires, such as σ and
the coins, read as 0/1 integer wires, as RS-MODEL's `extra` inputs are), and the combinators. A protocol in Model B
has bit bundles (`α → Bool`) for σ and the coins, and integer bundles for its matrices and messages. -/
inductive Circ : Type → Type → Type 1 where
  | gate (k : K) : Circ (Fin (kd k).arity → ℤ) (Fin (kd k).outs → ℤ)
  | reindex {α β V : Type} (f : β → α) : Circ (α → V) (β → V)
  | curry {α β V : Type} : Circ (α × β → V) (α → β → V)
  | uncurry {α β V : Type} : Circ (α → β → V) (α × β → V)
  | join {α β V : Type} : Circ ((α → V) × (β → V)) (α ⊕ β → V)
  | ofBits {α : Type} : Circ (α → Bool) (α → ℤ)
  | seq {I J O : Type} : Circ I J → Circ J O → Circ I O
  | pair {I J O : Type} : Circ I J → Circ I O → Circ I (J × O)
  | fst {I J : Type} : Circ (I × J) I
  | snd {I J : Type} : Circ (I × J) J

variable {kd}

/-- What a circuit computes. -/
def Circ.eval : {I O : Type} → Circ kd I O → I → O
  | _, _, .gate k, w => fun o => if (kd k).ok w then (kd k).sem w o else 0
  | _, _, .reindex f, w => w ∘ f
  | _, _, .curry, w => fun a b => w (a, b)
  | _, _, .uncurry, w => fun p => w p.1 p.2
  | _, _, .join, w => Sum.elim w.1 w.2
  | _, _, .ofBits, w => fun a => if w a then 1 else 0
  | _, _, .seq f g, x => g.eval (f.eval x)
  | _, _, .pair f g, x => (f.eval x, g.eval x)
  | _, _, .fst, x => x.1
  | _, _, .snd, x => x.2

/-- Its gates, in order. -/
def Circ.gates : {I O : Type} → Circ kd I O → List K
  | _, _, .gate k => [k]
  | _, _, .reindex _ => []
  | _, _, .curry => []
  | _, _, .uncurry => []
  | _, _, .join => []
  | _, _, .ofBits => []
  | _, _, .seq f g => f.gates ++ g.gates
  | _, _, .pair f g => f.gates ++ g.gates
  | _, _, .fst => []
  | _, _, .snd => []

/-- Its Model B program (wiring left out). -/
def Circ.prog {I O : Type} (c : Circ kd I O) : StraightLine K := progOf c.gates

variable (kd) in
/-- A Model B algorithm `I → O`: what it computes and its gate list, as some circuit's. (A subtype, so that
algorithms are small types; the circuit itself lives in `Type 1`.) -/
def Alg (I O : Type) : Type := {p : (I → O) × List K // ∃ c : Circ kd I O, c.eval = p.1 ∧ c.gates = p.2}

/-- A circuit's algorithm. -/
def Circ.alg {I O : Type} (c : Circ kd I O) : Alg kd I O := ⟨(c.eval, c.gates), c, rfl, rfl⟩

variable (kd) in
/-- **Model B as a `Machine`**, at an additive price `p` (`timeB` for RS-MODEL's B, `Price.ofNat`): time is `p` of the
gate list and a run costs its time (Model B is oblivious); composition concatenates gate lists, so time is additive
(RS-DEF's T1, `timeB_append`), and routing is free. -/
def machine (p : Price K) : Machine where
  Alg := Alg kd
  eval f := f.1.1
  time f := p.t f.1.2
  cost f _ := p.t f.1.2
  size f := p.t f.1.2
  seq f g := ⟨(g.1.1 ∘ f.1.1, f.1.2 ++ g.1.2), by
    obtain ⟨c, hc, hc'⟩ := f.2
    obtain ⟨d, hd, hd'⟩ := g.2
    exact ⟨.seq c d, by rw [← hc, ← hd]; rfl, by rw [← hc', ← hd']; rfl⟩⟩
  pair f g := ⟨(fun x => (f.1.1 x, g.1.1 x), f.1.2 ++ g.1.2), by
    obtain ⟨c, hc, hc'⟩ := f.2
    obtain ⟨d, hd, hd'⟩ := g.2
    exact ⟨.pair c d, by rw [← hc, ← hd]; rfl, by rw [← hc', ← hd']; rfl⟩⟩
  fst := Circ.fst.alg
  snd := Circ.snd.alg
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg f := p.nonneg f.1.2
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg f := p.nonneg f.1.2
  time_seq_le f g := (p.append f.1.2 g.1.2).le
  le_time_seq f g := by
    show _ ≤ p.t (f.1.2 ++ g.1.2)
    rw [p.append]
    linarith [p.nonneg g.1.2]
  time_pair_le f g := (p.append f.1.2 g.1.2).le
  le_time_pair_left f g := by
    show _ ≤ p.t (f.1.2 ++ g.1.2)
    rw [p.append]
    linarith [p.nonneg g.1.2]
  le_time_pair_right f g := by
    show _ ≤ p.t (f.1.2 ++ g.1.2)
    rw [p.append]
    linarith [p.nonneg f.1.2]
  time_fst := p.t_nil
  time_snd := p.t_nil

/-- A circuit's time in `machine` is the price of its gates. -/
theorem time_alg (p : Price K) {I O : Type} (c : Circ kd I O) : (machine kd p).time c.alg = p.t c.gates := rfl

/-- At `Price.ofNat`, it is RS-MODEL's `time_B` of its program. -/
theorem time_alg_ofNat (price : K → ℕ) {I O : Type} (c : Circ kd I O) :
    (machine kd (.ofNat price)).time c.alg = (timeB price c.prog : ℝ) := rfl

theorem eval_alg (p : Price K) {I O : Type} (c : Circ kd I O) : (machine kd p).eval c.alg = c.eval := rfl

variable (kd) in
/-- Circuits for the matrix operations over ℤ at their prices: an `a × b` by `b × c` product at `κ` per
multiply-accumulate, an entrywise sum or difference at `κadd` per entry. -/
structure MatGates (p : Price K) (κ κadd : ℝ) where
  mul : ∀ a b c : ℕ, Circ kd (Mat ℤ a b × Mat ℤ b c) (Mat ℤ a c)
  add : ∀ a b : ℕ, Circ kd (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b)
  sub : ∀ a b : ℕ, Circ kd (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b)
  eval_mul : ∀ (a b c : ℕ) (x : Mat ℤ a b × Mat ℤ b c), (mul a b c).eval x = x.1 * x.2
  eval_add : ∀ (a b : ℕ) (x : Mat ℤ a b × Mat ℤ a b), (add a b).eval x = x.1 + x.2
  eval_sub : ∀ (a b : ℕ) (x : Mat ℤ a b × Mat ℤ a b), (sub a b).eval x = x.1 - x.2
  time_mul_le : ∀ a b c : ℕ, p.t (mul a b c).gates ≤ κ * ((a : ℝ) * b * c)
  time_add_le : ∀ a b : ℕ, p.t (add a b).gates ≤ κadd * ((a : ℝ) * b)
  time_sub_le : ∀ a b : ℕ, p.t (sub a b).gates ≤ κadd * ((a : ℝ) * b)

/-- **Model B as a `MatMachine`** over ℤ, from circuits for the matrix operations. -/
def matMachine {p : Price K} {κ κadd : ℝ} (G : MatGates kd p κ κadd) : MatMachine ℤ where
  toMachine := machine kd p
  κ := κ
  κadd := κadd
  mul a b c := (G.mul a b c).alg
  add a b := (G.add a b).alg
  sub a b := (G.sub a b).alg
  eval_mul := G.eval_mul
  eval_add := G.eval_add
  eval_sub := G.eval_sub
  time_mul_le := G.time_mul_le
  time_add_le := G.time_add_le
  time_sub_le := G.time_sub_le

variable (kd) in
/-- Circuits for a grid product: exact on the shapes `shape` with operand entries in the grid `Gr`, at `κG` per
multiply-accumulate. Off the grid or the shapes they may compute anything. -/
structure GridGates (p : Price K) (Gr : ℤ → Prop) (shape : ℕ → ℕ → ℕ → Prop) (κG : ℝ) where
  mul : ∀ a b c : ℕ, Circ kd (Mat ℤ a b × Mat ℤ b c) (Mat ℤ a c)
  eval_mul : ∀ (a b c : ℕ) (x : Mat ℤ a b × Mat ℤ b c), shape a b c → (∀ i k, Gr (x.1 i k)) →
    (∀ k j, Gr (x.2 k j)) → (mul a b c).eval x = x.1 * x.2
  time_mul_le : ∀ a b c : ℕ, shape a b c → p.t (mul a b c).gates ≤ κG * ((a : ℝ) * b * c)

/-- **Model B as a `GridMatMachine`** over ℤ: `matMachine G` with the grid product from `I`. -/
def gridMatMachine {p : Price K} {κ κadd κG : ℝ} {Gr : ℤ → Prop} {shape : ℕ → ℕ → ℕ → Prop}
    (G : MatGates kd p κ κadd) (I : GridGates kd p Gr shape κG) : GridMatMachine ℤ where
  toMatMachine := matMachine G
  G := Gr
  shape := shape
  κG := κG
  mulG a b c := (I.mul a b c).alg
  eval_mulG := I.eval_mul
  time_mulG_le := I.time_mul_le

/-! ### The circuits exist: one GEMM kind per shape -/

/-- A pair of matrices on the wires `Fin (a b + c d)`, row-major (routing only). -/
def flatPair (a b c d : ℕ) : Circ kd (Mat ℤ a b × Mat ℤ c d) (Fin (a * b + c * d) → ℤ) :=
  .seq (.pair (.seq .fst (.seq (Circ.uncurry (α := Fin a) (β := Fin b) (V := ℤ)) (.reindex finProdFinEquiv.symm)))
      (.seq .snd (.seq (Circ.uncurry (α := Fin c) (β := Fin d) (V := ℤ)) (.reindex finProdFinEquiv.symm))))
    (.seq .join (.reindex finSumFinEquiv.symm))

/-- The wires `Fin (a c)` as an `a × c` matrix (routing only). -/
def unflat (a c : ℕ) : Circ kd (Fin (a * c) → ℤ) (Mat ℤ a c) :=
  .seq (.reindex finProdFinEquiv) (Circ.curry (α := Fin a) (β := Fin c) (V := ℤ))

/-- The first matrix read off `flatPair`'s wires. -/
def leftOf (a b c d : ℕ) (w : Fin (a * b + c * d) → ℤ) : Mat ℤ a b :=
  fun i k => w (finSumFinEquiv (.inl (finProdFinEquiv (i, k))))

/-- The second matrix read off `flatPair`'s wires. -/
def rightOf (a b c d : ℕ) (w : Fin (a * b + c * d) → ℤ) : Mat ℤ c d :=
  fun k j => w (finSumFinEquiv (.inr (finProdFinEquiv (k, j))))

theorem leftOf_flatPair (a b c d : ℕ) (x : Mat ℤ a b × Mat ℤ c d) :
    leftOf a b c d ((flatPair (kd := kd) a b c d).eval x) = x.1 := by
  funext i k
  simp [leftOf, flatPair, Circ.eval]

theorem rightOf_flatPair (a b c d : ℕ) (x : Mat ℤ a b × Mat ℤ c d) :
    rightOf a b c d ((flatPair (kd := kd) a b c d).eval x) = x.2 := by
  funext i k
  simp [rightOf, flatPair, Circ.eval]

/-- The GEMM gate family: one kind per shape for the product, the sum and the difference. -/
inductive GemmK
  | mm (a b c : ℕ)
  | add (a b : ℕ)
  | sub (a b : ℕ)

/-- Their exact semantics on `flatPair`'s wires, every operand in range. -/
def gemmKind : GemmK → Kind
  | .mm a b c => ⟨a * b + b * c, a * c, fun _ => true, fun w o =>
      (leftOf a b b c w * rightOf a b b c w) (finProdFinEquiv.symm o).1 (finProdFinEquiv.symm o).2⟩
  | .add a b => ⟨a * b + a * b, a * b, fun _ => true, fun w o =>
      (leftOf a b a b w + rightOf a b a b w) (finProdFinEquiv.symm o).1 (finProdFinEquiv.symm o).2⟩
  | .sub a b => ⟨a * b + a * b, a * b, fun _ => true, fun w o =>
      (leftOf a b a b w - rightOf a b a b w) (finProdFinEquiv.symm o).1 (finProdFinEquiv.symm o).2⟩

/-- Their prices: `κ₀` per multiply-accumulate, `κa` per entry. -/
def gemmPrice (κ₀ κa : ℕ) : GemmK → ℕ
  | .mm a b c => κ₀ * (a * b * c)
  | .add a b => κa * (a * b)
  | .sub a b => κa * (a * b)

/-- The product circuit: `flatPair`, one `mm` gate, `unflat`. -/
def gemmMul (a b c : ℕ) : Circ gemmKind (Mat ℤ a b × Mat ℤ b c) (Mat ℤ a c) :=
  .seq (flatPair a b b c) (.seq (Circ.gate (kd := gemmKind) (.mm a b c)) (unflat a c))

/-- The sum circuit. -/
def gemmAdd (a b : ℕ) : Circ gemmKind (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b) :=
  .seq (flatPair a b a b) (.seq (Circ.gate (kd := gemmKind) (.add a b)) (unflat a b))

/-- The difference circuit. -/
def gemmSub (a b : ℕ) : Circ gemmKind (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b) :=
  .seq (flatPair a b a b) (.seq (Circ.gate (kd := gemmKind) (.sub a b)) (unflat a b))

/-- **The circuits exist**: the GEMM gate family realizes `MatGates` with `κ = κ₀` and `κadd = κa`. -/
def gemmGates (κ₀ κa : ℕ) : MatGates gemmKind (.ofNat (gemmPrice κ₀ κa)) κ₀ κa where
  mul := gemmMul
  add := gemmAdd
  sub := gemmSub
  eval_mul a b c x := by
    funext i j
    simp only [gemmMul, Circ.eval, unflat, gemmKind, Function.comp_apply, Equiv.symm_apply_apply,
      leftOf_flatPair, rightOf_flatPair, ite_true]
  eval_add a b x := by
    funext i j
    simp only [gemmAdd, Circ.eval, unflat, gemmKind, Function.comp_apply, Equiv.symm_apply_apply,
      leftOf_flatPair, rightOf_flatPair, ite_true]
  eval_sub a b x := by
    funext i j
    simp only [gemmSub, Circ.eval, unflat, gemmKind, Function.comp_apply, Equiv.symm_apply_apply,
      leftOf_flatPair, rightOf_flatPair, ite_true]
  time_mul_le a b c := by
    simp [gemmMul, Price.ofNat, Circ.gates, flatPair, unflat, progOf, timeB, gemmPrice]
  time_add_le a b := by
    simp [gemmAdd, Price.ofNat, Circ.gates, flatPair, unflat, progOf, timeB, gemmPrice]
  time_sub_le a b := by
    simp [gemmSub, Price.ofNat, Circ.gates, flatPair, unflat, progOf, timeB, gemmPrice]

/-! ### sm_120 (RS-MODEL §4) -/

/-- RS-MODEL's sm_120 gate kinds, per warp instruction. -/
inductive K120
  | mma | mmaTf32 | imad | dp4a | hfma2 | ffma | alu | ldst

/-- RS-MODEL's `price120`, the LP dual of Model A's census rows, in u per warp instruction. -/
def price120 : K120 → ℕ
  | .mma => 4096 | .mmaTf32 => 4096 | .imad => 128 | .dp4a => 128 | .hfma2 => 64 | .ffma => 32 | .alu => 32
  | .ldst => 32

/-- **Model B on sm_120 as a `MatMachine`**, with one worst-case constant: `κ = 16` u per product (an s32 product as
4 × 4 int8 limb products on the tensor pipe at 1 u each) and `κadd = 1` u per entry (one ALU lane). It reads the
kinds' silicon semantics `kd` and the circuits `G` over them (named hypothesis `Gemm120`). -/
def matMachine120 (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) : MatMachine ℤ := matMachine G

theorem matMachine120_κ (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) : (matMachine120 kd G).κ = 16 := rfl

theorem matMachine120_κadd (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) : (matMachine120 kd G).κadd = 1 :=
  rfl

/-- **The sm_120 circuits** (a named hypothesis): over the kinds' silicon semantics `kd`, exact circuits for the
matrix operations over ℤ at 16 u per product and 1 u per entry. RS-MODEL prices sm_120's kinds and gives them no
semantics, so this is where `mma`'s exact semantics and the limb decomposition enter. -/
def Gemm120 (kd : K120 → Kind) : Prop := Nonempty (MatGates kd (.ofNat price120) 16 1)

/-- The int8 grid, `G_op`. -/
def Int8 (z : ℤ) : Prop := -128 ≤ z ∧ z ≤ 127

/-- The `m16n8k32` tile shapes, with the s32 accumulator exact (`b ≤ 2^17`, `int8_product_exact`). -/
def Tile120 (a b c : ℕ) : Prop := 16 ∣ a ∧ 32 ∣ b ∧ 8 ∣ c ∧ b ≤ 2 ^ 17

theorem tile120_square {n : ℕ} (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17) : Tile120 n n n :=
  ⟨(by norm_num : (16 : ℕ) ∣ 32).trans h32, h32, (by norm_num : (8 : ℕ) ∣ 32).trans h32, h17⟩

theorem int8_of_abs_le {a e s : ℤ} (ha : |a| ≤ 127 - s) (he : |e| ≤ s) : Int8 (a + e) := by
  have h := abs_le.mp ((abs_add_le a e).trans (by linarith : |a| + |e| ≤ 127))
  exact ⟨by linarith [h.1], h.2⟩

/-- **The edited operands are int8** (RS-RT's V6): an instance in `G_use = [−(127 − s), 127 − s]` edited by noise
bounded by `s` stays in int8. -/
theorem edit_int8 {Mc : MatMachine ℤ} {Seed : Type} {n : ℕ} {E : Mc.Edit Seed n} {s : ℤ} (hE : E.Bounded s)
    (σ : Seed) {x : Mat ℤ n n × Mat ℤ n n} (hx : InGrid (127 - s) x) :
    (∀ i k, Int8 ((E.edit σ x).1 i k)) ∧ ∀ k j, Int8 ((E.edit σ x).2 k j) :=
  ⟨fun i k => int8_of_abs_le (hx i k).1 (hE σ i k).1, fun k j => int8_of_abs_le (hx k j).2 (hE σ k j).2⟩

/-- **An accepted transcript's opened edited entries are int8**: the range check puts the committed instance's opened
entries in `G_use`, so under every σ' their edits are int8 operands, the grid the edit floor prices at 1 u. -/
theorem accepted_edit_int8 {Mc : MatMachine ℤ} {n : ℕ} {s : ℤ} (S : GridService n (127 - s))
    {E : Mc.Edit S.Seed n} (hE : E.Bounded s) {c : S.Com} {σ : S.Seed} {a : S.Ans} (h : S.verify c σ a = true)
    (σ' : S.Seed) {i j : Fin n} (hij : S.opened σ i j) :
    Int8 ((E.edit σ' (S.committed c)).1 i j) ∧ Int8 ((E.edit σ' (S.committed c)).2 i j) :=
  ⟨int8_of_abs_le (S.range_check c σ a h i j hij).1 (hE σ' i j).1,
    int8_of_abs_le (S.range_check c σ a h i j hij).2 (hE σ' i j).2⟩

/-- With every entry opened, an accepted transcript's whole edited instance is int8. -/
theorem accepted_edit_int8_all {Mc : MatMachine ℤ} {n : ℕ} {s : ℤ} (S : GridService n (127 - s))
    (hall : ∀ σ i j, S.opened σ i j) {E : Mc.Edit S.Seed n} (hE : E.Bounded s) {c : S.Com} {σ : S.Seed}
    {a : S.Ans} (h : S.verify c σ a = true) (σ' : S.Seed) :
    (∀ i k, Int8 ((E.edit σ' (S.committed c)).1 i k)) ∧ ∀ k j, Int8 ((E.edit σ' (S.committed c)).2 k j) :=
  edit_int8 hE σ' (S.inGrid_of_accepts hall h)

/-- Circuits for the int8 product on sm_120's tile shapes: exact on int8 operands at 1 u per multiply-accumulate
(RS-MODEL's `gemm120`: `a b c` u for the `m16n8k32` tile GEMM). -/
abbrev Int8Gates120 (kd : K120 → Kind) : Type 1 := GridGates kd (.ofNat price120) Int8 Tile120 1

/-- **The int8 circuits** (a named hypothesis, the width-typed refinement): over the kinds' silicon semantics `kd`,
exact int8 × int8 products on tile-multiple shapes at 1 u per multiply-accumulate, s32 exact. -/
def Int8Gemm120 (kd : K120 → Kind) : Prop := Nonempty (Int8Gates120 kd)

/-- **Model B on sm_120 with the operand grid typed**: the worst-case `matMachine120` (16 u per product on all
operands, 1 u per entry for sums) with the int8 product at `κG = 1` u on `G_op = int8` and the tile shapes. -/
def gridMatMachine120 (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    GridMatMachine ℤ :=
  gridMatMachine G I

theorem gridMatMachine120_κG (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120 kd G I).κG = 1 := rfl

theorem gridMatMachine120_κadd (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120 kd G I).κadd = 1 := rfl

/-- **Route (c) on sm_120, on the int8 grid** (`κ = 1` u on `G_op = int8`): on a range-checking service with
`G_use = [−(127 − s), 127 − s]` (the promise), noise bounded by `s` and `32 ∣ n ≤ 2^17`, the edited protocol served by
the int8 tile GEMM is relatively secure against `N = n³` u, with `ε = U / F − 1`, `U = n³ + 2 n² + time gen +
time prove`, and `γ = (2 n² + time gen + time prove) / n³`. The edits stay int8 by `edit_int8`. -/
theorem relativelySecure_edited120 (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120 kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120 kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η F : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120 kd G I).eval prove ((σ, y), C)).1 = ((gridMatMachine120 kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120 kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120 kd G I).eval ((gridMatMachine120 kd G I).mulG n n n) (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120 kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120 kd G I).eval ((gridMatMachine120 kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (hfloor : (gridMatMachine120 kd G I).EditFloor S.toService E ((gridMatMachine120 kd G I).mulG n n n) prove
      (InGrid (127 - s)) F κf)
    (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ (n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (gridMatMachine120 kd G I).time E.gen +
      (gridMatMachine120 kd G I).time prove) :
    RelativelySecure ((gridMatMachine120 kd G I).edited S.toService E ((gridMatMachine120 kd G I).mulG n n n) prove
      (InGrid (127 - s)))
      (((n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (gridMatMachine120 kd G I).time E.gen +
        (gridMatMachine120 kd G I).time prove) / F - 1) δ
      ((2 * (n : ℝ) ^ 2 + (gridMatMachine120 kd G I).time E.gen + (gridMatMachine120 kd G I).time prove) /
        (n : ℝ) ^ 3) η 0 ((n : ℝ) ^ 3) := by
  have h := (gridMatMachine120 kd G I).relativelySecure_of_editFloor_grid S.toService E prove (InGrid (127 - s))
    (tile120_square h32 h17) (fun σ _ hx => edit_int8 hE σ hx) hcom hacc hfloor hκδ hF
    (by rw [gridMatMachine120_κG, gridMatMachine120_κadd, one_mul, mul_one]; exact hFU)
    (by rw [gridMatMachine120_κG, one_mul]; exact pow_pos (Nat.cast_pos.mpr hn) 3)
  rwa [gridMatMachine120_κG, gridMatMachine120_κadd, one_mul, mul_one] at h

/-- **Route (c) on sm_120, unrestricted operands** (the worst case, `κ = 16` u): reference `N = 16 n³` u, `ε = U / F − 1`
with `U = 16 n³ + 2 n² + time gen + time prove`, and `γ = (2 n² + time gen + time prove) / (16 n³)`. -/
theorem relativelySecure_edited120_wide (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (S : Service) {n : ℕ}
    (hn : 0 < n) (E : (matMachine120 kd G).Edit S.Seed n)
    (prove : (matMachine120 kd G).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n) (S.Com × S.Ans))
    {κf δ η F : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((matMachine120 kd G).eval prove ((σ, y), C)).1 = ((matMachine120 kd G).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((matMachine120 kd G).eval prove ((ω.1, x, ω.2), E.product ω.1 x)).1 ω.1
        ((matMachine120 kd G).eval prove ((ω.1, x, ω.2), E.product ω.1 x)).2 = true)
    (hfloor : (matMachine120 kd G).EditFloor S E ((matMachine120 kd G).mul n n n) prove (fun _ => True) F κf)
    (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ 16 * (n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (matMachine120 kd G).time E.gen +
      (matMachine120 kd G).time prove) :
    RelativelySecure ((matMachine120 kd G).edited S E ((matMachine120 kd G).mul n n n) prove fun _ => True)
      ((16 * (n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (matMachine120 kd G).time E.gen + (matMachine120 kd G).time prove) / F -
        1) δ
      ((2 * (n : ℝ) ^ 2 + (matMachine120 kd G).time E.gen + (matMachine120 kd G).time prove) / (16 * (n : ℝ) ^ 3)) η 0
      (16 * (n : ℝ) ^ 3) := by
  have h := (matMachine120 kd G).relativelySecure_of_editFloor_wide S E prove hcom hacc hfloor hκδ hF
    (by rw [matMachine120_κ, matMachine120_κadd, mul_one]; exact hFU)
    (by rw [matMachine120_κ]; exact mul_pos (by norm_num) (pow_pos (Nat.cast_pos.mpr hn) 3))
  rwa [matMachine120_κ, matMachine120_κadd, mul_one] at h

/-! ### B_θ (RS-MODEL §6.2): Model B with shared memory's row mixed in

`B_θ = (1 − θ)·B + θ·R6`, with R6 the shared-memory data path (bytes / 128 per SM per cycle), at θ = 1/5. An instruction
carries its kind and the 4-byte words it moves through shared memory or L1; a gate reading its wires freely moves none,
so a B circuit is a B_θ circuit at `4/5` of its B time (`Circ.liftθ`, `MatGates.liftθ`), and every B_θ adversary has a
B shadow at most `5/4` of its B_θ time (`Circ.projθ`), which carries B's floor to `4/5` of it (`serviceFloor_θ`). -/

/-- An sm_120 warp instruction for `B_θ`: its kind and the 4-byte words it moves through shared memory or L1 (a gather
or a scatter is an `ldst`). -/
structure Instr120 where
  kind : K120
  words : ℕ

/-- One 4-byte word on the shared-memory row R6 alone: `W = R*/8 = 37.51` u (`R* = 300.08`, `gpu-model/rs-gather.py`). -/
def wordPrice : ℚ := 3751 / 100

/-- **RS-MODEL §6.2's `B_θ` at θ = 1/5**, `(1 − θ)·price120 + θ·R6`: MMA 3276.8 u, IMAD and DP4A 102.4, HFMA2 51.2,
every lane or ld/st instruction 25.6, plus `θ W = 7.502` u per word through shared memory or L1. -/
def price120θ (i : Instr120) : ℚ := 4 / 5 * price120 i.kind + 1 / 5 * wordPrice * i.words

theorem price120θ_nonneg (i : Instr120) : 0 ≤ price120θ i := by
  unfold price120θ wordPrice
  positivity

/-- `B_θ` as an additive price. -/
def priceθ : Price Instr120 := .ofRat price120θ price120θ_nonneg

/-- An instruction's semantics is its kind's (the words it moves are routing). -/
abbrev kdθ (kd : K120 → Kind) : Instr120 → Kind := fun i => kd i.kind

/-- B_θ prices a list at least `4/5` of B's price of its kinds. -/
theorem timeB_le_priceθ (l : List Instr120) :
    4 / 5 * (timeB price120 (progOf (l.map Instr120.kind)) : ℝ) ≤ priceθ.t l := by
  show _ ≤ (((l.map price120θ).sum : ℚ) : ℝ)
  rw [timeB_progOf, List.map_map, Nat.cast_list_sum, Rat.cast_list_sum, List.map_map, List.map_map,
    ← List.sum_map_mul_left]
  refine List.sum_le_sum fun i _ => ?_
  simp only [Function.comp_apply, price120θ, wordPrice]
  push_cast
  have : (0 : ℝ) ≤ 1 / 5 * (3751 / 100) * i.words := by positivity
  linarith

/-- A B gate list read in B_θ, each gate moving no words, costs `4/5` of its B time. -/
theorem priceθ_map_lift (l : List K120) :
    priceθ.t (l.map fun k => (⟨k, 0⟩ : Instr120)) = 4 / 5 * (timeB price120 (progOf l) : ℝ) := by
  show (((l.map fun k => (⟨k, 0⟩ : Instr120)).map price120θ).sum : ℝ) = _
  have h : (l.map fun k => (⟨k, 0⟩ : Instr120)).map price120θ = l.map fun k => (4 / 5 : ℚ) * price120 k := by
    simp [price120θ, Function.comp_def]
  rw [h, List.sum_map_mul_left, timeB_progOf]
  push_cast
  simp [Function.comp_def]

/-- A B circuit as a B_θ circuit: its gates read their wires freely, so they move no words. -/
def Circ.liftθ {kd : K120 → Kind} : {I O : Type} → Circ kd I O → Circ (kdθ kd) I O
  | _, _, .gate k => Circ.gate (kd := kdθ kd) ⟨k, 0⟩
  | _, _, .reindex f => .reindex f
  | _, _, .curry => .curry
  | _, _, .uncurry => .uncurry
  | _, _, .join => .join
  | _, _, .ofBits => .ofBits
  | _, _, .seq f g => .seq f.liftθ g.liftθ
  | _, _, .pair f g => .pair f.liftθ g.liftθ
  | _, _, .fst => .fst
  | _, _, .snd => .snd

/-- A B_θ circuit's B shadow: each instruction as its kind. -/
def Circ.projθ {kd : K120 → Kind} : {I O : Type} → Circ (kdθ kd) I O → Circ kd I O
  | _, _, .gate i => Circ.gate (kd := kd) i.kind
  | _, _, .reindex f => .reindex f
  | _, _, .curry => .curry
  | _, _, .uncurry => .uncurry
  | _, _, .join => .join
  | _, _, .ofBits => .ofBits
  | _, _, .seq f g => .seq f.projθ g.projθ
  | _, _, .pair f g => .pair f.projθ g.projθ
  | _, _, .fst => .fst
  | _, _, .snd => .snd

theorem Circ.eval_liftθ {kd : K120 → Kind} {I O : Type} (c : Circ kd I O) : c.liftθ.eval = c.eval := by
  induction c with
  | seq f g ihf ihg => funext x; simp only [Circ.liftθ, Circ.eval, ihf, ihg]
  | pair f g ihf ihg => funext x; simp only [Circ.liftθ, Circ.eval, ihf, ihg]
  | _ => rfl

theorem Circ.gates_liftθ {kd : K120 → Kind} {I O : Type} (c : Circ kd I O) :
    c.liftθ.gates = c.gates.map fun k => (⟨k, 0⟩ : Instr120) := by
  induction c with
  | seq f g ihf ihg => simp only [Circ.liftθ, Circ.gates, ihf, ihg, List.map_append]
  | pair f g ihf ihg => simp only [Circ.liftθ, Circ.gates, ihf, ihg, List.map_append]
  | _ => rfl

theorem Circ.eval_projθ {kd : K120 → Kind} {I O : Type} (c : Circ (kdθ kd) I O) : c.projθ.eval = c.eval := by
  induction c with
  | seq f g ihf ihg => funext x; simp only [Circ.projθ, Circ.eval, ihf, ihg]
  | pair f g ihf ihg => funext x; simp only [Circ.projθ, Circ.eval, ihf, ihg]
  | _ => rfl

theorem Circ.gates_projθ {kd : K120 → Kind} {I O : Type} (c : Circ (kdθ kd) I O) :
    c.projθ.gates = c.gates.map Instr120.kind := by
  induction c with
  | seq f g ihf ihg => simp only [Circ.projθ, Circ.gates, ihf, ihg, List.map_append]
  | pair f g ihf ihg => simp only [Circ.projθ, Circ.gates, ihf, ihg, List.map_append]
  | _ => rfl

/-- A B_θ algorithm's B shadow: the same function, each instruction as its kind. -/
def Alg.projθ {kd : K120 → Kind} {I O : Type} (f : Alg (kdθ kd) I O) : Alg kd I O :=
  ⟨(f.1.1, f.1.2.map Instr120.kind), by
    obtain ⟨c, hc, hc'⟩ := f.2
    exact ⟨c.projθ, by rw [Circ.eval_projθ, hc], by rw [Circ.gates_projθ, hc']⟩⟩

/-- **B's floor carries to B_θ at `1 − θ = 4/5` of it**: an adversary's B shadow computes the same messages, and B_θ
prices it at least `4/5` of its B time (`timeB_le_priceθ`). Superseded by `serviceFloorPerProver_θ`. -/
theorem serviceFloor_θ (kd : K120 → Kind) (S : Service) {F κ : ℝ}
    (h : ServiceFloor (machine kd (.ofNat price120)) S F κ) : ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * F) κ := by
  intro Ω _ M hc
  refine (prob_mono fun ω hω => ⟨hω.1, ?_⟩).trans (h Ω M.projθ hc)
  have h1 := timeB_le_priceθ M.1.2
  have h2 : priceθ.t M.1.2 < 4 / 5 * F := (max_lt_iff.mp hω.2).1
  have h3 : (timeB price120 (progOf (M.1.2.map Instr120.kind)) : ℝ) < F := by linarith
  exact max_lt h3 h3

/-- **B's floor per adversary carries to B_θ at `4/5` of it** (`serviceFloor_θ` adversary by adversary, which it
supersedes): a B_θ adversary's bound is its B shadow's (`Alg.projθ`), so the floors per prover's `breakBound` reads the
shadow's collision finders, at budgets computed from the shadow's B time: its gates priced in B, which no choice of
finder sets. -/
theorem serviceFloorPerProver_θ (kd : K120 → Kind) (S : Service) {F : ℝ}
    {κ : ∀ (Ω : Type) [Fintype Ω], (machine kd (.ofNat price120)).Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal}
    (h : ServiceFloorPerProver (machine kd (.ofNat price120)) S F κ) :
    ServiceFloorPerProver (machine (kdθ kd) priceθ) S (4 / 5 * F) (fun Ω _ M => κ Ω M.projθ) := by
  intro Ω _ M hc
  refine (ENNReal.ofReal_le_ofReal (prob_mono fun ω hω => ⟨hω.1, ?_⟩)).trans (h Ω M.projθ hc)
  have h1 := timeB_le_priceθ M.1.2
  have h2 : priceθ.t M.1.2 < 4 / 5 * F := (max_lt_iff.mp hω.2).1
  have h3 : (timeB price120 (progOf (M.1.2.map Instr120.kind)) : ℝ) < F := by linarith
  exact max_lt h3 h3

/-- B's matrix circuits in B_θ, at `4/5` of their prices. -/
noncomputable def MatGates.liftθ {kd : K120 → Kind} {κ κadd : ℝ} (G : MatGates kd (.ofNat price120) κ κadd) :
    MatGates (kdθ kd) priceθ (4 / 5 * κ) (4 / 5 * κadd) where
  mul a b c := (G.mul a b c).liftθ
  add a b := (G.add a b).liftθ
  sub a b := (G.sub a b).liftθ
  eval_mul a b c x := by rw [Circ.eval_liftθ]; exact G.eval_mul a b c x
  eval_add a b x := by rw [Circ.eval_liftθ]; exact G.eval_add a b x
  eval_sub a b x := by rw [Circ.eval_liftθ]; exact G.eval_sub a b x
  time_mul_le a b c := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (G.time_mul_le a b c) (by norm_num)
  time_add_le a b := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (G.time_add_le a b) (by norm_num)
  time_sub_le a b := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (G.time_sub_le a b) (by norm_num)

/-- B's grid circuits in B_θ, at `4/5` of their price. -/
noncomputable def GridGates.liftθ {kd : K120 → Kind} {Gr : ℤ → Prop} {shape : ℕ → ℕ → ℕ → Prop} {κG : ℝ}
    (I : GridGates kd (.ofNat price120) Gr shape κG) : GridGates (kdθ kd) priceθ Gr shape (4 / 5 * κG) where
  mul a b c := (I.mul a b c).liftθ
  eval_mul a b c x hs hx hy := by rw [Circ.eval_liftθ]; exact I.eval_mul a b c x hs hx hy
  time_mul_le a b c hs := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (I.time_mul_le a b c hs) (by norm_num)

/-- **Model B_θ on sm_120 with the operand grid typed**: B's circuits read in B_θ, the int8 tile GEMM at
`κG = 4/5` u per multiply-accumulate (the in-model `0.8 n³`, gates reading their wires freely), sums at `κadd = 4/5`
u per entry, and the worst case at `κ = 64/5` u. -/
noncomputable def gridMatMachine120θ (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    GridMatMachine ℤ :=
  gridMatMachine G.liftθ I.liftθ

theorem gridMatMachine120θ_κG (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120θ kd G I).κG = 4 / 5 := mul_one _

theorem gridMatMachine120θ_κadd (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120θ kd G I).κadd = 4 / 5 := mul_one _

/-- **Route (c) on sm_120 in B_θ, on the int8 grid, from any B floor**: B's service floor `F_B ≤ n³` u (a named
hypothesis, `ServiceFloor`) carries to `0.8 F_B` u in B_θ (`serviceFloor_θ`), and on a range-checking service with
`G_use = [−(127 − s), 127 − s]` and noise bounded by `s` (so the edits stay int8, `edit_int8`) the edited protocol served
by the int8 tile GEMM is relatively secure against `N = 0.8 n³` u with `ε = U / (0.8 F_B) − 1`,
`U = 0.8 n³ + 1.6 n² + time gen + time prove` (no commitment cost credited), and
`γ = (1.6 n² + time gen + time prove) / (0.8 n³)`. Superseded by `relativelySecurePerProver_edited120θ_of_floor`:
its `κf` is one bound for every adversary, so over SHA-512 commitments `hfloorB` holds only at `κf` about
1. -/
theorem relativelySecure_edited120θ_of_floor (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    {FB : ℝ} (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService FB κf) (hκδ : κf < δ) (hFB : 0 < FB)
    (hFBn : FB ≤ (n : ℝ) ^ 3) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * FB) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hN : (0 : ℝ) < 4 / 5 * (n : ℝ) ^ 3 := mul_pos (by norm_num) (pow_pos (Nat.cast_pos.mpr hn) 3)
  have hg := (gridMatMachine120θ kd G I).time_nonneg E.gen
  have hp := (gridMatMachine120θ kd G I).time_nonneg prove
  have h := (gridMatMachine120θ kd G I).relativelySecure_of_editFloor_grid S.toService E prove (InGrid (127 - s))
    (tile120_square h32 h17) (fun σ _ hx => edit_int8 hE σ hx) hcom hacc (serviceFloor_θ kd S.toService hfloorB) hκδ
    (by positivity) (by rw [gridMatMachine120θ_κG, gridMatMachine120θ_κadd]; nlinarith [sq_nonneg (n : ℝ)])
    (by rw [gridMatMachine120θ_κG]; exact hN)
  rwa [gridMatMachine120θ_κG, gridMatMachine120θ_κadd] at h

/-- **Route (c) on sm_120 in B_θ, on the int8 grid, from any B floor per adversary** (proposed, a claim change;
`relativelySecure_edited120θ_of_floor` adversary by adversary, which it supersedes): B's service floor per adversary
`F_B ≤ n³` u, at each adversary's own `κf` (a named hypothesis, `ServiceFloorPerProver`; the floors per prover of
`Definitions.Pouw.Guarantees.RelativeSecurity` give it at `κf = breakBound sha fb fz cb db cz dz 2^{−40} δb βb δz βz`,
`relativelySecurePerProver_edited120θ_at_breakBound`), carries to `0.8 F_B` u in B_θ
(`serviceFloorPerProver_θ`), and the edited protocol is relatively secure per adversary with
`relativelySecure_edited120θ_of_floor`'s `ε`, `γ` and `N`, at each B_θ adversary's threshold `κf(M.projθ) + δ'`: its B
shadow's bound, read through the shadow's collision finders, plus one `δ' > 0`. -/
theorem relativelySecurePerProver_edited120θ_of_floor (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1)
    (I : Int8Gates120 kd) {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n)
    (h17 : n ≤ 2 ^ 17) (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {δ' η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    {FB : ℝ}
    {κf : ∀ (Ω : Type) [Fintype Ω], (machine kd (.ofNat price120)).Alg (S.Seed × Ω) (S.Com × S.Ans) → ENNReal}
    (hfloorB : ServiceFloorPerProver (machine kd (.ofNat price120)) S.toService FB κf) (hδ' : 0 < δ')
    (hFB : 0 < FB) (hFBn : FB ≤ (n : ℝ) ^ 3) :
    RelativelySecurePerProver
      ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove
        (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen +
        (gridMatMachine120θ kd G I).time prove) / (4 / 5 * FB) - 1)
      (fun Ω _ M => κf Ω (Alg.projθ (kd := kd) M) + ENNReal.ofReal δ')
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hN : (0 : ℝ) < 4 / 5 * (n : ℝ) ^ 3 := mul_pos (by norm_num) (pow_pos (Nat.cast_pos.mpr hn) 3)
  have hg := (gridMatMachine120θ kd G I).time_nonneg E.gen
  have hp := (gridMatMachine120θ kd G I).time_nonneg prove
  have h := (gridMatMachine120θ kd G I).relativelySecurePerProver_of_editFloor_grid S.toService E prove
    (InGrid (127 - s)) (tile120_square h32 h17) (fun σ _ hx => edit_int8 hE σ hx) hcom hacc
    (serviceFloorPerProver_θ kd S.toService hfloorB) hδ'
    (by positivity) (by rw [gridMatMachine120θ_κG, gridMatMachine120θ_κadd]; nlinarith [sq_nonneg (n : ℝ)])
    (by rw [gridMatMachine120θ_κG]; exact hN)
  rwa [gridMatMachine120θ_κG, gridMatMachine120θ_κadd] at h

/-- **Route (c) on sm_120 in B_θ at the floors' error per prover** (proposed, a claim change):
`relativelySecurePerProver_edited120θ_of_floor` at `κf = breakBound sha fb fz cb db cz dz 2^{−40} δb βb δz βz`, the
error the floors per prover of `Definitions.Pouw.Guarantees.RelativeSecurity` conclude. Each B_θ
adversary's threshold is its B shadow's error plus `δ'`: the shadow's finders read through `ecr/sha-512` at budgets of
`cb · time + db` and `cz · time + dz` SHA-512 calls, `time` the shadow's B time, which the adversary's gates fix and no
choice of finder lowers. -/
theorem relativelySecurePerProver_edited120θ_at_breakBound (kd : K120 → Kind)
    (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) {n : ℕ} {s : ℤ} (S : GridService n (127 - s))
    (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17) (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {δ' η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    {FB : ℝ} {R : ℕ} (sha : Claims.Msg → Claims.Msg) (fb fz : Finders (machine kd (.ofNat price120)) S.toService R)
    (cb db cz dz δb βb δz βz : ℝ)
    (hfloorB : ServiceFloorPerProver (machine kd (.ofNat price120)) S.toService FB
      (breakBound sha fb fz cb db cz dz ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz)) (hδ' : 0 < δ')
    (hFB : 0 < FB) (hFBn : FB ≤ (n : ℝ) ^ 3) :
    RelativelySecurePerProver
      ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove
        (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen +
        (gridMatMachine120θ kd G I).time prove) / (4 / 5 * FB) - 1)
      (fun Ω _ M => breakBound sha fb fz cb db cz dz ((2 : ℝ) ^ 40)⁻¹ δb βb δz βz Ω (Alg.projθ (kd := kd) M) +
        ENNReal.ofReal δ')
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) :=
  relativelySecurePerProver_edited120θ_of_floor kd G I S hn h32 h17 E hE prove hcom hacc hfloorB hδ' hFB hFBn

/-- **Route (c) in B_θ at Conjecture A's floor** (`c = 1`, `F_B = n³` u, open): `ε = γ`. -/
theorem relativelySecure_edited120θ (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService ((n : ℝ) ^ 3) κf) (hκδ : κf < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hN : (0 : ℝ) < 4 / 5 * (n : ℝ) ^ 3 := mul_pos (by norm_num) (pow_pos (Nat.cast_pos.mpr hn) 3)
  have h := relativelySecure_edited120θ_of_floor kd G I S hn h32 h17 E hE prove hcom hacc hfloorB hκδ
    (pow_pos (Nat.cast_pos.mpr hn) 3) le_rfl
  rw [div_sub_one hN.ne'] at h
  convert h using 2
  ring

/-- **Route (c) in B_θ at a floor of Theorem E's form** (RS-D1E): B's floor `F_B = (c / m_max) (φ n³ − sl)` u for
rationals `0 < c ≤ m_max` and `φ ≤ 1` and a slack `0 ≤ sl < φ n³` (`L (M + K)`, and RS-W′'s excess while it stands), so
`ε = U / (0.8 F_B) − 1`. -/
theorem relativelySecure_edited120θ_rat (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (c mmax φ : ℚ) {sl : ℝ} (hc : 0 < c) (hcm : c ≤ mmax) (hφ : φ ≤ 1) (hsl : 0 ≤ sl) (hslφ : sl < φ * (n : ℝ) ^ 3)
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService
      (((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl)) κf) (hκδ : κf < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * (((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl))) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hq0 : (0 : ℝ) < ((c / mmax : ℚ) : ℝ) := by exact_mod_cast div_pos hc (hc.trans_le hcm)
  have hq1 : ((c / mmax : ℚ) : ℝ) ≤ 1 := by exact_mod_cast (div_le_one (hc.trans_le hcm)).mpr hcm
  have hφ' : (φ : ℝ) ≤ 1 := by exact_mod_cast hφ
  have hn3 : (0 : ℝ) ≤ (n : ℝ) ^ 3 := by positivity
  have hφn : (φ : ℝ) * (n : ℝ) ^ 3 ≤ (n : ℝ) ^ 3 := by nlinarith
  refine relativelySecure_edited120θ_of_floor kd G I S hn h32 h17 E hE prove hcom hacc hfloorB hκδ
    (mul_pos hq0 (by linarith)) ?_
  calc ((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl) ≤ 1 * ((φ : ℝ) * (n : ℝ) ^ 3 - sl) :=
        mul_le_mul_of_nonneg_right hq1 (by linarith)
    _ ≤ (n : ℝ) ^ 3 := by linarith

/-- Theorem E's `φ = 1 − f − m / N_B` (RS-D1E). -/
def phiE (f m NB : ℚ) : ℚ := 1 - f - m / NB

/-- **Route (c) in B_θ at Theorem E's proved floor** (`c = ½`, `m_max = 2`, `φ = 1 − f − m / N_B`): B's floor
`F_B = (φ n³ − sl) / 4` u gives `ε = U / ((φ n³ − sl) / 5) − 1`, about `4 / φ − 1` for large `n`. -/
theorem relativelySecure_edited120θ_half (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (f m NB : ℚ) {sl : ℝ} (hf : 0 ≤ f) (hm : 0 ≤ m / NB) (hsl : 0 ≤ sl)
    (hslφ : sl < (phiE f m NB : ℝ) * (n : ℝ) ^ 3)
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService
      (1 / 4 * ((phiE f m NB : ℝ) * (n : ℝ) ^ 3 - sl)) κf) (hκδ : κf < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * (1 / 4 * ((phiE f m NB : ℝ) * (n : ℝ) ^ 3 - sl))) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hq : ((1 / 2 / 2 : ℚ) : ℝ) = 1 / 4 := by norm_num
  have h := relativelySecure_edited120θ_rat kd G I S hn h32 h17 E hE prove hcom hacc (1 / 2) 2 (phiE f m NB)
    (by norm_num) (by norm_num) (by unfold phiE; linarith) hsl hslφ (by rw [hq]; exact hfloorB) hκδ
  rwa [hq] at h

/-! ### The certified fraction

Superseded per adversary. Every theorem of the `certifiedFraction120θ` family reads one `κf`, or one `δb` and
`δz` for every adversary through its `RoundsHyp*`, so over SHA-512 commitments it holds only at a bound about 1: an
adversary that hard-codes a collision breaks binding in every round. Its form per adversary is
`serviceFloorPerProver_θ` at the floors per prover of `Definitions.Pouw.Guarantees.RelativeSecurity` (the same floor, each
adversary's error its `breakBound`, read at its B shadow, its finders at budgets computed from the shadow's B time),
proved in `Proofs.Pouw.RelativeSecurityGuarantees` and `Proofs.Pouw.RelativeSecurityJ10R`; their reader is
`relativelySecurePerProver_edited120θ_at_breakBound`.
-/

/-- **The certified fraction** (the headline): from B's floor of Theorem E's form, every adversary in B_θ that commits
first has its accepted runs charged (`max(|M|, cost)`, the program included) below `(c φ / m_max) N − 0.8 (c / m_max) sl`
with probability at most `κ`, for `N = 0.8 n³` u the reference matmul. No commitment cost is credited. -/
theorem certifiedFraction120θ (kd : K120 → Kind) (S : Service) (n : ℕ) (c mmax φ : ℚ) {sl κf : ℝ}
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S (((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl)) κf) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (((c * φ / mmax : ℚ) : ℝ) * (4 / 5 * (n : ℝ) ^ 3) - 4 / 5 * ((c / mmax : ℚ) : ℝ) * sl) κf := by
  convert serviceFloor_θ kd S hfloorB using 1
  push_cast
  ring

/-- **The certified fraction at Theorem E's proved floor** (`c = ½`, `m_max = 2`): accepted runs are charged at least
`(φ / 4) N − sl / 5` except with probability `κ`, `N = 0.8 n³` u. -/
theorem certifiedFraction120θ_half (kd : K120 → Kind) (S : Service) (n : ℕ) (f m NB : ℚ) {sl κf : ℝ}
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S (1 / 4 * ((phiE f m NB : ℝ) * (n : ℝ) ^ 3 - sl)) κf) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      ((phiE f m NB : ℝ) / 4 * (4 / 5 * (n : ℝ) ^ 3) - sl / 5) κf := by
  convert serviceFloor_θ kd S hfloorB using 1
  ring

/-- **The certified fraction from Theorem S** (in place of Theorem M): under the window's rounds in Model B
(`RoundsHyp`: per round, Theorem E's floor `F₀` of the round's GEMMs, the slack `sl = L(M + K)`, and the pass bound
along guard paths), every adversary in B_θ that commits first has its accepted runs charged below
`0.8 ((R − D) F₀ − Vmax − R sl)` with probability at most `e^{−(1−p) D} + R (δb + δz)`. Of the window's `0.8 R F₀`
that is the share `1 − D / R − (Vmax + R sl) / (R F₀)`. -/
theorem certifiedFraction120θ_rounds (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb δz : ℝ} (h : RoundsHyp (machine kd (.ofNat price120)) S Ω R p F₀ sl Vmax δb δz) (hp : p ≤ 1)
    (hF₀ : 0 < F₀) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) (D : ℝ) :
    ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * ((R - D) * F₀ - Vmax - R * sl))
      (Real.exp (-((1 - p) * D)) + R * (δb + δz)) :=
  serviceFloor_θ kd S (serviceFloor_of_rounds h hp hF₀ hδb hδz D)

/-- **The certified fraction at `κ = 2⁻⁴⁰ + R (δb + δz)`**: `D₀ = 40 ln 2 / (1 − p)`, so the luck loss is
`27.73 / ((1 − p) R)` of the window, detection included. -/
theorem certifiedFraction120θ_rounds_40 (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb δz : ℝ} (h : RoundsHyp (machine kd (.ofNat price120)) S Ω R p F₀ sl Vmax δb δz) (hp : p < 1)
    (hF₀ : 0 < F₀) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * ((R - 40 * Real.log 2 / (1 - p)) * F₀ - Vmax - R * sl))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have hc : (1 - p) * (40 * Real.log 2 / (1 - p)) = ((40 : ℕ) : ℝ) * Real.log 2 := by
    field_simp [(sub_pos.mpr hp).ne']
    norm_num
  have h40 : Real.exp (-((1 - p) * (40 * Real.log 2 / (1 - p)))) = ((2 : ℝ) ^ 40)⁻¹ := by
    rw [hc, Real.exp_neg, Real.exp_nat_mul, Real.exp_log two_pos]
  rw [← h40]
  exact certifiedFraction120θ_rounds kd S h hp.le hF₀ hδb hδz _

/-- **The certified fraction from Theorem K** (the re-encoded design, in place of the paper's cost line): under
`RoundsHypK` in Model B for a product class with chain factor 1, every adversary in B_θ that commits first has its
accepted runs charged below `0.8 ((R − D) κ L n n' B − Vmax)` with probability at most `e^{−(1−p) D} + R (δb + δz)`.
That is `c = 1` of the window's `R` GEMMs of `n × LB × n'` at `κ` a product, less `D` rounds of luck and the noise
reads. -/
theorem certifiedFraction120θ_K (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p Vmax δb δz κ : ℝ} {n L np B n' : ℕ} {Ar : Type} [CommRing Ar] {wt : Fin np → Ar}
    {Ok : ((DBits n L np B n' → Ar) → Ar) → Prop}
    (h : RoundsHypK (machine kd (.ofNat price120)) S Ω R p Vmax δb δz κ n L np B n' wt Ok)
    (hc : ChainBound Ar n L np B n' wt Ok 1) (hp : p ≤ 1) (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) (D : ℝ) :
    ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * ((R - D) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (Real.exp (-((1 - p) * D)) + R * (δb + δz)) := by
  have hF₀ : 0 < κ * L * ((n * n' * B : ℕ) : ℝ) := by
    rw [mul_assoc]
    exact mul_pos hκ (by exact_mod_cast hC)
  simpa only [mul_zero, sub_zero] using
    certifiedFraction120θ_rounds kd S (roundsHyp_of_K_one h hc hκ.le) hp hF₀ hδb hδz D

/-- ... at `κ = 2⁻⁴⁰ + R (δb + δz)` (`D₀ = 40 ln 2 / (1 − p)`). -/
theorem certifiedFraction120θ_K_40 (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p Vmax δb δz κ : ℝ} {n L np B n' : ℕ} {Ar : Type} [CommRing Ar] {wt : Fin np → Ar}
    {Ok : ((DBits n L np B n' → Ar) → Ar) → Prop}
    (h : RoundsHypK (machine kd (.ofNat price120)) S Ω R p Vmax δb δz κ n L np B n' wt Ok)
    (hc : ChainBound Ar n L np B n' wt Ok 1) (hp : p < 1) (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 40 * Real.log 2 / (1 - p)) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have hF₀ : 0 < κ * L * ((n * n' * B : ℕ) : ℝ) := by
    rw [mul_assoc]
    exact mul_pos hκ (by exact_mod_cast hC)
  simpa only [mul_zero, sub_zero] using
    certifiedFraction120θ_rounds_40 kd S (roundsHyp_of_K_one h hc hκ.le) hp hF₀ hδb hδz

/-- **The re-encoded design's certified fraction** (`X = ⌊x/s⌋ + b`, U's planes at `w = (1, 2, 4, 8, 16, 32, 65)`),
at `κ = 2⁻⁴⁰ + R (δb + δz)`, for runs whose products are one-sided, of any form: the chain lemma
(`chainBound_oneSided`) discharges the chain bound. -/
theorem certifiedFraction120θ_reEncoded_40 (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p Vmax δb δz κ : ℝ} {n L B n' : ℕ}
    (h : RoundsHypK (machine kd (.ofNat price120)) S Ω R p Vmax δb δz κ n L 7 B n' wRe OneSided) (hp : p < 1)
    (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 40 * Real.log 2 / (1 - p)) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) :=
  certifiedFraction120θ_K_40 kd S h (chainBound_oneSided toF2 toF2_wRe_zero) hp hκ hC hδb hδz

/-- ... for runs with mixed products (`GProd`: operands any function of U's bits plus affine in the bits, mixed
affine bytes included): Lemma G over `F₂` (`chainBound_gProd`) discharges the chain bound. -/
theorem certifiedFraction120θ_reEncodedMixed_40 (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p Vmax δb δz κ : ℝ} {n L B n' : ℕ}
    (h : RoundsHypK (machine kd (.ofNat price120)) S Ω R p Vmax δb δz κ n L 7 B n' wRe GProd) (hp : p < 1)
    (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 40 * Real.log 2 / (1 - p)) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) :=
  certifiedFraction120θ_K_40 kd S h (chainBound_gProd toF2 rfl toF2_wRe_zero) hp hκ hC hδb hδz

/-- **The re-encoded design's certified fraction in exact arithmetic, from the chain lemma** (RS-E-REF r4 IV.3, so
Theorem H is not needed): over `ℚ` (FP8 operands or scaled reads, exact sums) the chain lemma holds at `φ = id`, so
runs with one-sided products of any form, merges free, have the floor they have under s32 wrap. -/
theorem certifiedFraction120θ_reEncodedExactK_40 (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p Vmax δb δz κ : ℝ} {n L B n' : ℕ}
    (h : RoundsHypK (machine kd (.ofNat price120)) S Ω R p Vmax δb δz κ n L 7 B n' wReQ OneSided) (hp : p < 1)
    (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 40 * Real.log 2 / (1 - p)) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) :=
  certifiedFraction120θ_K_40 kd S h
    (chainBound_oneSided (RingHom.id ℚ) (p₀ := 0) (by rw [RingHom.id_apply]; exact wReQ_zero)) hp hκ hC hδb hδz

/-- **The re-encoded design's certified fraction with the luck derived** (one-sided affine products: `U`'s planes
times the coins' word, or any product of an operand affine in U's bits and one affine in the coins): under
`RoundsHypKS`, neither `PathBound` nor a chain bound is a hypothesis; the spot check gives `p = ¾`, so `D₀ = 160 ln 2`
rounds of luck at `2⁻⁴⁰`. -/
theorem certifiedFraction120θ_reEncodedSpot_40 (kd : K120 → Kind) (S : Service) {R : ℕ} {Vmax δb δz κ : ℝ}
    {n L B n' : ℕ}
    (h : RoundsHypKS (machine kd (.ofNat price120)) S R Vmax δb δz κ n L 7 B n' wRe
      fun π => OneSided π ∧ AffProd π)
    (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 160 * Real.log 2) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have e : (40 * Real.log 2 / (1 - 3 / 4) : ℝ) = 160 * Real.log 2 := by
    rw [show (1 : ℝ) - 3 / 4 = 1 / 4 by norm_num]
    ring
  have := certifiedFraction120θ_K_40 kd S (roundsHypK_of_spot h fun _ hπ => hπ.2)
    ((chainBound_oneSided toF2 toF2_wRe_zero).mono fun _ hπ => hπ.1) (by norm_num) hκ hC hδb hδz
  rwa [e] at this

/-- ... with mixed affine products allowed too (any product of two affine operands): Lemma G
(`chainBound_gProd`) discharges the chain bound, so neither the luck nor the chain is a hypothesis. -/
theorem certifiedFraction120θ_reEncodedSpotMixed_40 (kd : K120 → Kind) (S : Service) {R : ℕ} {Vmax δb δz κ : ℝ}
    {n L B n' : ℕ}
    (h : RoundsHypKS (machine kd (.ofNat price120)) S R Vmax δb δz κ n L 7 B n' wRe fun π => AffProd π)
    (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 160 * Real.log 2) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have e : (40 * Real.log 2 / (1 - 3 / 4) : ℝ) = 160 * Real.log 2 := by
    rw [show (1 : ℝ) - 3 / 4 = 1 / 4 by norm_num]
    ring
  have := certifiedFraction120θ_K_40 kd S (roundsHypK_of_spot h fun _ hπ => hπ)
    ((chainBound_gProd toF2 rfl toF2_wRe_zero).mono fun _ hπ => hπ.gProd) (by norm_num) hκ hC hδb hδz
  rwa [e] at this

/-- **The re-encoded design's certified fraction with per-stage draws** (decision 36, RS-E-REF r5 V.3's repair):
under `RoundsHypKSt` with products of affine operands (one-sided or mixed), rounds of `St` stages, each drawing its
products' noise after committing their operands, have the floor with the `St` stage floors summed,
`F₀ = St κ L n n' B`, at `D₀ = 160 ln 2` rounds of luck at `2⁻⁴⁰`, with neither a luck nor a chain hypothesis. -/
theorem certifiedFraction120θ_staged_40 (kd : K120 → Kind) (S : Service) {R St : ℕ} {Vmax δb δz κ : ℝ}
    {n L B n' : ℕ}
    (h : RoundsHypKSt (machine kd (.ofNat price120)) S R St Vmax δb δz κ n L 7 B n' wRe fun π => AffProd π)
    (hSt : 0 < St) (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 160 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have e : (40 * Real.log 2 / (1 - 3 / 4) : ℝ) = 160 * Real.log 2 := by
    rw [show (1 : ℝ) - 3 / 4 = 1 / 4 by norm_num]
    ring
  have hF₀ : 0 < (St : ℝ) * (κ * L * ((n * n' * B : ℕ) : ℝ)) := by
    rw [mul_assoc]
    exact mul_pos (by exact_mod_cast hSt) (mul_pos hκ (by exact_mod_cast hC))
  have := certifiedFraction120θ_rounds_40 kd S
    (roundsHyp_of_staged h ((chainBound_gProd toF2 rfl toF2_wRe_zero).mono fun _ hπ => hπ.gProd) (fun _ hπ => hπ)
      hSt hκ.le) (by norm_num) hF₀ hδb hδz
  rwa [e, mul_zero, sub_zero] at this

/-- **Decode's deferred edit as an instance** (RS-D1E (b), RS-E-REF r5 V.3: every operand of the round's products
committed before the round's one draw): rounds of one stage, with the floor `F₀ = κ L n n' B`. -/
theorem certifiedFraction120θ_decode_40 (kd : K120 → Kind) (S : Service) {R : ℕ} {Vmax δb δz κ : ℝ}
    {n L B n' : ℕ}
    (h : RoundsHypKSt (machine kd (.ofNat price120)) S R 1 Vmax δb δz κ n L 7 B n' wRe fun π => AffProd π)
    (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 160 * Real.log 2) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  simpa only [Nat.cast_one, one_mul] using certifiedFraction120θ_staged_40 kd S h one_pos hκ hC hδb hδz

/-- **The certified fraction with overlapping stages** (Theorem K∥): under `RoundsHypKOv` with K∥'s products,
rounds of `St` stages of any shapes, their draws overlapping at any depth, have the floor `F₀ = κ L Σ_t n_t n'_t B_t`,
every stage's cell-blocks counted once, at `D₀ = 160 ln 2` rounds of luck at `2⁻⁴⁰`, with neither a luck nor a chain
hypothesis. -/
theorem certifiedFraction120θ_overlap_40 (kd : K120 → Kind) (S : Service) {R St : ℕ} {Vmax δb δz κ : ℝ}
    {n B n' : Fin St → ℕ} {L : ℕ}
    (h : RoundsHypKOv (machine kd (.ofNat price120)) S R St Vmax δb δz κ n L 7 B n' wRe) (hκ : 0 < κ)
    (hC : 0 < L * ∑ t, n t * n' t * B t) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 160 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have e : (40 * Real.log 2 / (1 - 3 / 4) : ℝ) = 160 * Real.log 2 := by
    rw [show (1 : ℝ) - 3 / 4 = 1 / 4 by norm_num]
    ring
  have hF₀ : 0 < κ * L * (Fintype.card (OCells St n n' B) : ℝ) := by
    rw [mul_assoc, card_oCells]
    exact mul_pos hκ (by exact_mod_cast hC)
  have := certifiedFraction120θ_rounds_40 kd S
    (roundsHyp_of_overlap h (fun _ => (chainBound_gProd toF2 rfl toF2_wRe_zero).mono fun _ hπ => hπ.gProd) hκ.le)
    (by norm_num) hF₀ hδb hδz
  rwa [e, mul_zero, sub_zero, card_oCells] at this

theorem ServiceFloor.mono {Mc : Machine} {S : Service} {F κ κ' : ℝ} (h : ServiceFloor Mc S F κ) (hκ : κ ≤ κ') :
    ServiceFloor Mc S F κ' := fun Ω _ M hc => (h Ω M hc).trans hκ

/-- At the parity luck `0.854`, `D₀ = 274 ln 2` rounds of luck reach `2⁻⁴⁰`: `0.146 · 274 ≥ 40`. -/
theorem exp_parity_274 : Real.exp (-((1 - 427 / 500) * (274 * Real.log 2))) ≤ ((2 : ℝ) ^ 40)⁻¹ := by
  have h40 : Real.exp (-(((40 : ℕ) : ℝ) * Real.log 2)) = ((2 : ℝ) ^ 40)⁻¹ := by
    rw [Real.exp_neg, Real.exp_nat_mul, Real.exp_log two_pos]
  rw [← h40]
  refine Real.exp_le_exp.mpr ?_
  have := Real.log_pos (by norm_num : (1 : ℝ) < 2)
  push_cast
  nlinarith

/-- **The certified fraction with per-stage draws, one-sided operands of any form** (RS-E-REF r6 VI.2): under
`RoundsHypKStP` on `Z/2³²` read mod 2 (`toF2`, plane 0 at an odd weight), the staged floor `F₀ = St κ L n n' B` at
`D₀ = 274 ln 2` rounds of luck at `2⁻⁴⁰` (the parity luck `0.854`), with neither a luck nor a chain hypothesis. -/
theorem certifiedFraction120θ_stagedP_40 (kd : K120 → Kind) (S : Service) {R St : ℕ} {Vmax δb δz κ : ℝ}
    {n L B n' : ℕ} (h : RoundsHypKStP (machine kd (.ofNat price120)) S R St Vmax δb δz κ n L 7 B n' wRe toF2 0)
    (hSt : 0 < St) (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 274 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have hF₀ : 0 < (St : ℝ) * (κ * L * ((n * n' * B : ℕ) : ℝ)) := by
    rw [mul_assoc]
    exact mul_pos (by exact_mod_cast hSt) (mul_pos hκ (by exact_mod_cast hC))
  have := certifiedFraction120θ_rounds kd S (roundsHyp_of_stagedP h toF2_wRe_zero hSt hκ.le) (by norm_num) hF₀ hδb
    hδz (274 * Real.log 2)
  rw [mul_zero, sub_zero] at this
  exact ServiceFloor.mono this (by linarith [exp_parity_274])

/-- **The certified fraction with overlapping stages, one-sided operands of any form** (RS-E-REF r6 VI.2): under
`RoundsHypKOvP` on `Z/2³²` read mod 2, the floor `F₀ = κ L Σ_t n_t n'_t B_t` at `D₀ = 274 ln 2` rounds of luck at
`2⁻⁴⁰`, with neither a luck nor a chain hypothesis. -/
theorem certifiedFraction120θ_overlapP_40 (kd : K120 → Kind) (S : Service) {R St : ℕ} {Vmax δb δz κ : ℝ}
    {n B n' : Fin St → ℕ} {L : ℕ}
    (h : RoundsHypKOvP (machine kd (.ofNat price120)) S R St Vmax δb δz κ n L 7 B n' wRe toF2 0) (hκ : 0 < κ)
    (hC : 0 < L * ∑ t, n t * n' t * B t) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 274 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have hF₀ : 0 < κ * L * (Fintype.card (OCells St n n' B) : ℝ) := by
    rw [mul_assoc, card_oCells]
    exact mul_pos hκ (by exact_mod_cast hC)
  have := certifiedFraction120θ_rounds kd S (roundsHyp_of_overlapP h toF2_wRe_zero hκ.le) (by norm_num) hF₀ hδb hδz
    (274 * Real.log 2)
  rw [mul_zero, sub_zero, card_oCells] at this
  exact ServiceFloor.mono this (by linarith [exp_parity_274])

/-- **The certified fraction from Theorem H** (the re-encoded design in exact arithmetic, merges priced at
`h ≥ 2L/3` slots): under `RoundsHypH` in Model B, the same floor as from Theorem K. -/
theorem certifiedFraction120θ_H (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p Vmax δb δz κ h : ℝ} {F : Type} [Field F] {n L np B n' : ℕ} {wt : Fin np → F} {p₀ : Fin np}
    (hH : RoundsHypH (machine kd (.ofNat price120)) S Ω R p Vmax δb δz κ h F n L np B n' wt) (hwt : wt p₀ = 1)
    (hh : 2 * (L : ℝ) / 3 ≤ h) (hp : p ≤ 1) (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb)
    (hδz : 0 ≤ δz) (D : ℝ) :
    ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * ((R - D) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (Real.exp (-((1 - p) * D)) + R * (δb + δz)) := by
  have hF₀ : 0 < κ * L * ((n * n' * B : ℕ) : ℝ) := by
    rw [mul_assoc]
    exact mul_pos hκ (by exact_mod_cast hC)
  simpa only [mul_zero, sub_zero] using
    certifiedFraction120θ_rounds kd S (roundsHyp_of_H hH hwt hκ.le hh) hp hF₀ hδb hδz D

/-- **The re-encoded design's certified fraction in exact arithmetic** over `ℚ`, at VF2's merge price (32 slots a
merged word, so any `L ≤ 48`) and `κ = 2⁻⁴⁰ + R (δb + δz)`. -/
theorem certifiedFraction120θ_reEncodedExact_40 (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p Vmax δb δz κ : ℝ} {n L B n' : ℕ}
    (hH : RoundsHypH (machine kd (.ofNat price120)) S Ω R p Vmax δb δz κ 32 ℚ n L 7 B n' wReQ) (hL : L ≤ 48)
    (hp : p < 1) (hκ : 0 < κ) (hC : 0 < L * (n * n' * B)) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (4 / 5 * ((R - 40 * Real.log 2 / (1 - p)) * (κ * L * ((n * n' * B : ℕ) : ℝ)) - Vmax))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have hF₀ : 0 < κ * L * ((n * n' * B : ℕ) : ℝ) := by
    rw [mul_assoc]
    exact mul_pos hκ (by exact_mod_cast hC)
  have hh : 2 * (L : ℝ) / 3 ≤ 32 := by
    have : (L : ℝ) ≤ 48 := by exact_mod_cast hL
    linarith
  simpa only [mul_zero, sub_zero] using
    certifiedFraction120θ_rounds_40 kd S (roundsHyp_of_H hH wReQ_zero hκ.le hh) hp hF₀ hδb hδz

/-- **Route (c) in B_θ from Theorem K**, with no edit-floor hypothesis: on the int8 grid, if the window's rounds
satisfy `RoundsHypK` on B for the grid service (the re-encoded design, a product class with chain factor 1:
one-sided products by `chainBound_oneSided`, mixed ones by `chainBound_gProd`), then the window's floor
`F_B = (R − D) κ L m n' B − Vmax` (positive, at most `n³`) is B's service floor at `κf = e^{−(1−p) D} + R (δb + δz)`,
and the edited protocol is relatively secure with `ε = U / (0.8 F_B) − 1` for every `δ > κf`. Over `Z/2³²`, Theorem K
takes the place of the field-only bit-pair projection behind `EditFloor`. -/
theorem relativelySecure_edited120θ_K (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1)
    (I : Int8Gates120 kd) {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n)
    (h17 : n ≤ 2 ^ 17) (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    {Ω : Type} [Fintype Ω] {R : ℕ} {p Vmax δb δz κ : ℝ} {m L np B n' : ℕ} {Ar : Type} [CommRing Ar] {wt : Fin np → Ar}
    {Ok : ((DBits m L np B n' → Ar) → Ar) → Prop} (hK : RoundsHypK (machine kd (.ofNat price120)) S.toService Ω R p Vmax δb δz κ m L np B n' wt Ok)
    (hc : ChainBound Ar m L np B n' wt Ok 1) (hp : p ≤ 1) (hκ : 0 < κ) (hC : 0 < L * (m * n' * B)) (hδb : 0 ≤ δb)
    (hδz : 0 ≤ δz) (D : ℝ) {FB : ℝ} (hFBdef : FB = (R - D) * (κ * L * ((m * n' * B : ℕ) : ℝ)) - Vmax)
    (hFB : 0 < FB) (hFBn : FB ≤ (n : ℝ) ^ 3) (hκδ : Real.exp (-((1 - p) * D)) + R * (δb + δz) < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * FB) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hF₀ : 0 < κ * L * ((m * n' * B : ℕ) : ℝ) := by
    rw [mul_assoc]
    exact mul_pos hκ (by exact_mod_cast hC)
  have hfl := serviceFloor_of_rounds (roundsHyp_of_K_one hK hc hκ.le) hp hF₀ hδb hδz D
  rw [mul_zero, sub_zero, ← hFBdef] at hfl
  exact relativelySecure_edited120θ_of_floor kd G I S hn h32 h17 E hE prove hcom hacc hfl hκδ hFB hFBn

/-- **Route (c) in B_θ from Theorem H**, with no edit-floor hypothesis: the same conclusion under `RoundsHypH`, the
re-encoded design in exact arithmetic over a field with merges at `h ≥ 2L/3` slots. -/
theorem relativelySecure_edited120θ_H (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1)
    (I : Int8Gates120 kd) {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n)
    (h17 : n ≤ 2 ^ 17) (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    {Ω : Type} [Fintype Ω] {R : ℕ} {p Vmax δb δz κ h : ℝ} {F : Type} [Field F] {m L np B n' : ℕ}
    {wt : Fin np → F} {p₀ : Fin np}
    (hH : RoundsHypH (machine kd (.ofNat price120)) S.toService Ω R p Vmax δb δz κ h F m L np B n' wt)
    (hwt : wt p₀ = 1) (hh : 2 * (L : ℝ) / 3 ≤ h) (hp : p ≤ 1) (hκ : 0 < κ) (hC : 0 < L * (m * n' * B))
    (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) (D : ℝ) {FB : ℝ}
    (hFBdef : FB = (R - D) * (κ * L * ((m * n' * B : ℕ) : ℝ)) - Vmax) (hFB : 0 < FB) (hFBn : FB ≤ (n : ℝ) ^ 3)
    (hκδ : Real.exp (-((1 - p) * D)) + R * (δb + δz) < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * FB) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hF₀ : 0 < κ * L * ((m * n' * B : ℕ) : ℝ) := by
    rw [mul_assoc]
    exact mul_pos hκ (by exact_mod_cast hC)
  have hfl := serviceFloor_of_rounds (roundsHyp_of_H hH hwt hκ.le hh) hp hF₀ hδb hδz D
  rw [mul_zero, sub_zero, ← hFBdef] at hfl
  exact relativelySecure_edited120θ_of_floor kd G I S hn h32 h17 E hE prove hcom hacc hfl hκδ hFB hFBn

/-! ### Model A (RS-MODEL §3, over ℚ) and `B_θ ≤ A` -/

section ModelA

variable {Res : Type} (rs : List Res) (cost : K → Res → ℚ)

/-- RS-MODEL's `use`: per resource, the sum of a gate list's cost vectors. -/
def useA (l : List K) (r : Res) : ℚ := (l.map fun k => cost k r).sum

/-- RS-MODEL's `dot`: a row's load on a use vector, over the resources `rs`. -/
def dotA (row u : Res → ℚ) : ℚ := (rs.map fun r => row r * u r).sum

/-- RS-MODEL's `time_A`: the largest certified row (rows are normalized: a real run of length `T` has
`dot row (use P) ≤ T`). -/
def timeA (rows : List (Res → ℚ)) (l : List K) : ℚ := rows.foldr (fun row m => max (dotA rs row (useA cost l)) m) 0

theorem timeA_nonneg (rows : List (Res → ℚ)) (l : List K) : 0 ≤ timeA rs cost rows l := by
  induction rows with
  | nil => exact le_refl 0
  | cons a rows ih =>
    simp only [timeA, List.foldr_cons] at *
    exact le_max_of_le_right ih

theorem dotA_le_timeA {rows : List (Res → ℚ)} {row : Res → ℚ} (h : row ∈ rows) (l : List K) :
    dotA rs row (useA cost l) ≤ timeA rs cost rows l := by
  induction rows with
  | nil => cases h
  | cons a rows ih =>
    simp only [timeA, List.foldr_cons] at *
    rcases List.mem_cons.mp h with rfl | h
    · exact le_max_left _ _
    · exact (ih h).trans (le_max_right _ _)

/-- **RS-MODEL's `timeB_le_timeA`** (B as a dual combination of A's rows), over ℚ and for a price bounded by the
combination: if `price k ≤ Σ_p y_p (row_p · cost k)` with weights `y_p ≥ 0` on rows of A, then
`Σ price ≤ (Σ y) · time_A`. -/
theorem sum_le_timeA (rows : List (Res → ℚ)) (ys : List (ℚ × (Res → ℚ))) (hys : ∀ p ∈ ys, 0 ≤ p.1 ∧ p.2 ∈ rows)
    (price : K → ℚ) (hprice : ∀ k, price k ≤ (ys.map fun p => p.1 * dotA rs p.2 (cost k)).sum) (l : List K) :
    (l.map price).sum ≤ (ys.map Prod.fst).sum * timeA rs cost rows l := by
  have key : (l.map price).sum ≤ (ys.map fun p => p.1 * dotA rs p.2 (useA cost l)).sum := by
    induction l with
    | nil => simp [useA, dotA]
    | cons k l ih =>
      have e : ∀ p : ℚ × (Res → ℚ), p.1 * dotA rs p.2 (useA cost (k :: l)) =
          p.1 * dotA rs p.2 (cost k) + p.1 * dotA rs p.2 (useA cost l) := by
        intro p
        rw [← mul_add]
        congr 1
        simp [dotA, useA, mul_add, List.sum_map_add]
      rw [List.map_cons, List.sum_cons, List.map_congr_left (fun p _ => e p), List.sum_map_add]
      linarith [hprice k]
  calc _ ≤ _ := key
    _ ≤ (ys.map fun p => p.1 * timeA rs cost rows l).sum :=
        List.sum_le_sum fun p hp => mul_le_mul_of_nonneg_left (dotA_le_timeA rs cost (hys p hp).2 l) (hys p hp).1
    _ = _ := List.sum_map_mul_right _ _ _

end ModelA

/-- **`B_θ ≤ A`** (RS-MODEL §6.2's lemma, as an instance of `timeB_le_timeA`): if B's prices are a dual combination of
A's rows with total weight at most 1 (`hB`, `hsum`) and R6 is a row of A that prices each word at `W` (`hW`), then
`B_θ = (4/5)·B + (1/5)·R6` is the combination `(4/5)·ys + (1/5)·R6`, of total weight at most 1, so B_θ's time of
every gate list is at most A's. -/
theorem timeθ_le_timeA {Res : Type} (rs : List Res) (cost : Instr120 → Res → ℚ) (rows : List (Res → ℚ))
    (ys : List (ℚ × (Res → ℚ))) (hys : ∀ p ∈ ys, 0 ≤ p.1 ∧ p.2 ∈ rows) (hsum : (ys.map Prod.fst).sum ≤ 1)
    (hB : ∀ i, (price120 i.kind : ℚ) ≤ (ys.map fun p => p.1 * dotA rs p.2 (cost i)).sum)
    {R6 : Res → ℚ} (hR6 : R6 ∈ rows) (hW : ∀ i : Instr120, wordPrice * i.words ≤ dotA rs R6 (cost i))
    (l : List Instr120) : (l.map price120θ).sum ≤ timeA rs cost rows l := by
  have h := sum_le_timeA rs cost rows (ys.map (fun p : ℚ × (Res → ℚ) => (4 / 5 * p.1, p.2)) ++ [(1 / 5, R6)]) ?_ price120θ ?_ l
  · have hs : ((ys.map (fun p : ℚ × (Res → ℚ) => (4 / 5 * p.1, p.2)) ++ [(1 / 5, R6)]).map Prod.fst).sum =
        4 / 5 * (ys.map Prod.fst).sum + 1 / 5 := by
      simp only [List.map_append, List.map_map, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
        List.sum_nil, add_zero]
      rw [← List.sum_map_mul_left]
      rfl
    rw [hs] at h
    have hA := timeA_nonneg rs cost rows l
    nlinarith
  · intro p hp
    rcases List.mem_append.mp hp with hp | hp
    · obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hp
      have := (hys q hq).1
      exact ⟨by positivity, (hys q hq).2⟩
    · rw [List.mem_singleton] at hp
      subst hp
      exact ⟨by norm_num, hR6⟩
  · intro i
    have e : ((ys.map (fun p : ℚ × (Res → ℚ) => (4 / 5 * p.1, p.2)) ++ [(1 / 5, R6)]).map fun p : ℚ × (Res → ℚ) => p.1 * dotA rs p.2 (cost i)).sum =
        4 / 5 * (ys.map fun p => p.1 * dotA rs p.2 (cost i)).sum + 1 / 5 * dotA rs R6 (cost i) := by
      simp only [List.map_append, List.map_map, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
        List.sum_nil, add_zero]
      rw [← List.sum_map_mul_left]
      congr 2
      exact List.map_congr_left fun p _ => by simp [mul_assoc]
    rw [e]
    unfold price120θ
    linarith [hB i, hW i]

end ModelB

end ModelB

end Pouw.RelativeSecurity
