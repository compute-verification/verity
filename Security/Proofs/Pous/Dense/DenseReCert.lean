import Mathlib.Data.Nat.Bitwise
import Proofs.Pous.Dense.DenseMeets

/-!
# Dense scheme re-certificate at the 64 KB operating point, and the multi-segment lemma

Re-certifies `PousDense.denseScheme` (the random-oracle dense single-layer scheme, `DenseMeets.lean`) at the point the
end-to-end memo proposes (`docs/scheme-choice-e2e.md`): **64 KB labels** (`ℓ = 524288` bits), **512 blocks per segment**
(`B = 2^9`), **D = 5** rounds, `Q = 2^20`, targeting **k = 106**. All machinery — `dense_timedINC` (generic in
`(B, ℓ, D, Q, β)`), `PousProofs.theorem1_ii_seq`, `p₀`, and the certificate helpers — is reused verbatim from
`DenseMeets.lean`; only the numeric certificate lemmas are new.

Certificate at this point: `S = ⌊(18/19)·2^28⌋ = 254307274`, `β = 265783012`.
- Union term `|Hist|·π ≤ 2^-128` exactly: with `(B(Q+1)+B)^2 ≤ 2^59`, the exponent is
  `(524288−59)·507 − (9·106 + 9 + β) = 128`.
- Sampling term `p₀^106 ≤ 1%`: `p₀ = 1 − 11475226/268435456 = 256960230/268435456 ≈ 0.957252`, and `p₀^106 ≈ 0.00975`
  (while `p₀^105 ≈ 0.0102 > 1%`, so `k = 106` is tight, matching the memo).

Everything is in the random-oracle model. The multi-segment lemma (`dense_meets_multi`) composes `N` independent,
distinct-tagged segments; it is structured so the concrete overwrite-chain `H` (`ChainExPostFacto`, `sponge-dense`) can
replace the random oracle segment-by-segment without touching the composition.
-/

namespace PousDenseReCert

open Pous PousDense Pous.Labelling Pous.Dense
open scoped ENNReal

set_option maxRecDepth 4000

/-! ### Piece 1: the numeric re-certificate at 64 KB, B = 512, D = 5, k = 106 -/

/-- `|Hist|·π ≤ 2^-128` at `B = 2^9`, `ℓ = 524288`, `D = 5`, `Q = 2^20`, `k = 106`, `β = 265783012`. -/
theorem dense_union_term_64 :
    (∑ j ∈ Finset.range 106, ((2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      (((2 ^ 9 : ℕ) : ℝ≥0∞) * 2 ^ 265783012 *
        (((((2 ^ 9) * (2 ^ 20 + 1) + 2 ^ 9) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ 524288) ^ (2 ^ 9 - 5 - 1 + 1)) ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have he : 2 ^ 9 - 5 - 1 + 1 = 507 := by norm_num
  have hsq : ((2 ^ 9) * (2 ^ 20 + 1) + 2 ^ 9) ^ 2 ≤ 2 ^ 59 := by norm_num
  have h9 : ((2 ^ 9 : ℕ) : ℝ≥0∞) = (2 : ℝ≥0∞) ^ 9 := by norm_num
  rw [he]
  calc _ ≤ ((2 ^ 9 : ℕ) : ℝ≥0∞) ^ 106 *
        (((2 ^ 9 : ℕ) : ℝ≥0∞) * 2 ^ 265783012 * ((2⁻¹ : ℝ≥0∞) ^ (524288 - 59)) ^ 507) := by
        gcongr
        · exact sum_pow_le _ _ (by norm_num)
        · exact sq_div_two_pow_le _ 59 524288 hsq (by norm_num)
    _ = (2 : ℝ≥0∞) ^ (9 * 106) *
          ((2 : ℝ≥0∞) ^ 9 * 2 ^ 265783012 * (2⁻¹ : ℝ≥0∞) ^ ((524288 - 59) * 507)) := by
        rw [h9, ← pow_mul, ← pow_mul]
    _ = (2⁻¹ : ℝ≥0∞) ^ ((524288 - 59) * 507 - (9 * 106 + 9 + 265783012)) :=
        two_pow_combine _ _ _ _ (by norm_num)
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

/-- `p₀^106 ≤ 1%` at `β = 265783012`, `S = 254307274`, `B = 2^9`, `ℓ = 524288`. -/
theorem dense_sampling_term_64 :
    ENNReal.ofReal (p₀ 265783012 254307274 (2 ^ 9) 524288) ^ 106 ≤ ENNReal.ofReal (Params.δ : ℝ) := by
  have h0 : 0 ≤ p₀ 265783012 254307274 (2 ^ 9) 524288 := by unfold p₀; norm_num
  rw [← ENNReal.ofReal_pow h0]
  refine ENNReal.ofReal_le_ofReal ?_
  unfold p₀
  norm_num [Params.δ]

/-- `S = ⌊(18/19)·2^28⌋ = 254307274` at `|C| = 2^9·524288 = 2^28`. -/
theorem stateBound_dense_64 : (denseScheme (2 ^ 9) 524288).stateBound Params.ρ = 254307274 := by
  show ⌊Params.ρ * (((2 ^ 9) * 524288 : ℕ) : ℚ)⌋₊ = 254307274
  rw [Nat.floor_eq_iff (by norm_num [Params.ρ])]
  norm_num [Params.ρ]

/-- **The dense scheme meets the requirement at the 64 KB point**, random-oracle `H`, with sequential reveal, `D = 5`
rounds, `Q = 2^20`, `k = 106` challenges and `ε = 2^-128`: correct, rate 1, and every `⌊ρ|C|⌋`-bit adversary is accepted
with probability at most `1% + 2^-128`. -/
theorem dense_meets_64 :
    Pous.Meets (denseScheme (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128) := by
  refine ⟨le_rfl, denseScheme_correct _ _, le_rfl, ?_, ?_⟩
  · show (((2 ^ 9 * 524288 + 0 : ℕ)) : ℚ) ≤ Pous.Params.expansion * ((2 ^ 9 * 524288 : ℕ) : ℚ)
    norm_num [Pous.Params.expansion]
  · intro W A₁ A₂
    refine (PousProofs.theorem1_ii_seq (denseScheme (2 ^ 9) 524288) 5 (2 ^ 20) 265783012
      ((denseScheme (2 ^ 9) 524288).stateBound Pous.Params.ρ) 106 _
      (dense_timedINC (2 ^ 9) 524288 5 (2 ^ 20) 265783012 (by norm_num)) W A₁ A₂).trans ?_
    rw [stateBound_dense_64, add_comm]
    exact add_le_add dense_sampling_term_64 dense_union_term_64

/-! ### Piece 2: the disjoint-union pebbling lemma for many segments

The 14 GB store is `N` independent segments of `B` blocks each, laid out on `Fin (N*B)` with **global** block indices, so
segment `s`'s block `i` is node `s*B + i`. Because the index is global, the random oracle's query key `(node, parents)`
already distinguishes segments — this is the "distinct segment tags" of §20, provided here by global indexing rather than
an explicit tag. `segDAG N B` is the disjoint union of `N` complete DAGs: a node's parents are the *earlier blocks of its
own segment*. The pebbling lemma below (`seg_hard`) extends `complete_hard`: since each round advances at most one node per
segment, `s` pebbles reach at most `s + N*D` nodes in `D` rounds, so the disjoint union is `(N*B − N*D − 1, D, N*B)`-hard.

This is the model half a multi-segment `Meets` needs: it plugs into `Pous.Guarantees.B5ExPostFactoBounded` (giving
`π = (N*B)·2^β·(((N*B)(Q+1)+(N*B))²/2^ℓ)^(N*B − N*D)`) exactly as `complete_hard` does for one segment, then
`PousProofs.theorem1_ii_seq` and a certificate. At 14 GB the instantiation is `N = 448`, `B = 2^9`, `ℓ = 524288`
(each segment `2^28` bits = 32 MiB, `448·32 MiB = 14 GB`). Kept in the random-oracle model; the concrete overwrite-chain
`H` slots in by replacing the oracle, leaving `segDAG`/`seg_hard` untouched. -/

/-- **Disjoint-union reach.** From `P₀` pebbles, `D` rounds pebble at most `P₀.card + N*D` nodes: each round adds at most
one node per segment (a second new node of a segment would be a parent of the later one, hence already reached). -/
theorem card_reach_seg (N B : ℕ) (P₀ : Finset (Fin (N * B))) :
    ∀ r, (reach (segDAG N B) P₀ r).card ≤ P₀.card + N * r
  | 0 => by rw [reach]; omega
  | r + 1 => by
    have ih := card_reach_seg N B P₀ r
    have hnew : ((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
        reach (segDAG N B) P₀ r).card ≤ N := by
      have hmap : Set.InjOn (fun v : Fin (N * B) => v.val / B)
          ↑((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
            reach (segDAG N B) P₀ r) := by
        intro a ha b hb hab
        simp only [Finset.coe_sdiff, Set.mem_diff, Finset.mem_coe, Finset.mem_filter] at ha hb
        by_contra hne
        rcases lt_or_gt_of_ne (fun h => hne (Fin.ext h)) with h | h
        · exact ha.2 (hb.1.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hab, h⟩))
        · exact hb.2 (ha.1.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hab.symm, h⟩))
      calc _ ≤ (Finset.image (fun v : Fin (N * B) => v.val / B)
              ((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
                reach (segDAG N B) P₀ r)).card := (Finset.card_image_of_injOn hmap).ge
        _ ≤ (Finset.range N).card := by
            apply Finset.card_le_card
            intro x hx
            simp only [Finset.mem_image] at hx
            obtain ⟨v, _, rfl⟩ := hx
            have hlt : v.val < B * N := by rw [Nat.mul_comm B N]; exact v.isLt
            exact Finset.mem_range.2 (Nat.div_lt_of_lt_mul hlt)
        _ = N := by rw [Finset.card_range]
    rw [reach, ← Finset.union_sdiff_self_eq_union]
    calc _ ≤ (reach (segDAG N B) P₀ r).card +
          ((Finset.univ.filter fun v => (segDAG N B).par v ⊆ reach (segDAG N B) P₀ r) \
            reach (segDAG N B) P₀ r).card := Finset.card_union_le _ _
      _ ≤ P₀.card + N * (r + 1) := by rw [Nat.mul_succ]; omega

/-- **The disjoint union of `N` complete DAGs on `B` blocks is `(N*B − N*D − 1, D, N*B)`-hard**, every block a challenge.
This is `complete_hard` extended to many segments; with global block indices it is the §20 distinct-tag structure. -/
theorem seg_hard (N B D : ℕ) (hN : 0 < N) (hD : D + 1 ≤ B) :
    PebblingHard (segDAG N B) Finset.univ (N * B - N * D - 1) D (N * B) := by
  intro P₀ hP₀
  have hmul : N * D + N ≤ N * B := by
    calc N * D + N = N * (D + 1) := by ring
      _ ≤ N * B := Nat.mul_le_mul (le_refl N) hD
  calc (Finset.univ.filter fun c => c ∈ reach (segDAG N B) P₀ D).card
      ≤ (reach (segDAG N B) P₀ D).card := Finset.card_le_card fun c hc => (Finset.mem_filter.1 hc).2
    _ ≤ P₀.card + N * D := card_reach_seg N B P₀ D
    _ < N * B := by omega

/-! ### Piece 2 (continued): the segmented scheme and its `Meets` at 14 GB

The disjoint-union scheme `segScheme N B ℓ` is `denseScheme` on `segDAG N B` (global block indices), so decoding a block
queries only its own segment's `≤ B` earlier blocks. `seg_timedINC` is the B5 glue with `seg_hard`, and the certificate at
`N = 448`, `B = 2^9`, `ℓ = 524288` gives **14 GB** (`448·2^28` bits) with **k = 106**, `β = 119067414778`,
`S = ⌊(18/19)·N·B·ℓ⌋ = 113929658799`. -/

/-- Bit-exact public decoding for the segmented scheme. -/
theorem segScheme_correct (N B ℓ : ℕ) [NeZero (N * B)] : (segScheme N B ℓ).Correct := by
  intro (W : Bits ((N * B) * ℓ)) (H : LabelQry (N * B) ℓ → Bits ℓ) _
  show codeEquiv (N * B) ℓ (fun i => bxor (label (segDAG N B) (blocks (N * B) ℓ W) H i)
    (H (i, fun j => if j ∈ (segDAG N B).par i then some (label (segDAG N B) (blocks (N * B) ℓ W) H j)
      else none))) = W
  have h : ∀ i, bxor (label (segDAG N B) (blocks (N * B) ℓ W) H i)
      (H (i, fun j => if j ∈ (segDAG N B).par i then some (label (segDAG N B) (blocks (N * B) ℓ W) H j)
        else none)) = blocks (N * B) ℓ W i := by
    intro i
    rw [PousB5.label_eq]
    exact bxor_bxor_self _ _
  rw [funext h]
  exact (codeEquiv (N * B) ℓ).apply_symm_apply W

/-- **B5 gives `TimedINC` for the segmented scheme**: `π = (N*B)·2^β·(((N*B)(Q+1)+(N*B))²/2^ℓ)^(N*B − N*D)`. -/
theorem seg_timedINC (N B ℓ D Q β : ℕ) [NeZero (N * B)] (hN : 0 < N) (hD : D + 1 ≤ B) :
    TimedINC (segScheme N B ℓ) D Q β
      (((N * B : ℕ) : ℝ≥0∞) * 2 ^ β *
        (((((N * B) * (Q + 1) + (N * B)) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ ℓ) ^ (N * B - N * D - 1 + 1)) := by
  classical
  intro W cmp exp hexp
  have hB5 := PousB5.b5_ex_post_facto_bounded (N * B) ℓ β (N * B - N * D - 1) D Q (N * B)
    (segDAG N B) Finset.univ (blocks (N * B) ℓ W) (seg_hard N B D hN hD)
    (fun H => cmp W (fun i => i.elim0) (fun i => label (segDAG N B) (blocks (N * B) ℓ W) H i) H)
    (fun σ i => exp W (fun i => i.elim0) σ i) (fun σ i => hexp W _ σ i)
  rw [Finset.card_univ, Fintype.card_fin] at hB5
  refine le_trans (le_of_eq_of_le (PousThm1.pr_fst (β := Unit) (fun H : LabelQry (N * B) ℓ → Bits ℓ =>
    ∀ i, (exp W (fun i => i.elim0)
      (cmp W (fun i => i.elim0) (fun i => label (segDAG N B) (blocks (N * B) ℓ W) H i) H)
      i).run H = label (segDAG N B) (blocks (N * B) ℓ W) H i)) (pr_mono fun H hH => ?_)) hB5
  refine le_trans (le_of_eq (by rw [Finset.card_univ, Fintype.card_fin]))
    (Finset.card_le_card (s := Finset.univ) fun c _ => Finset.mem_filter.2 ⟨Finset.mem_univ _, hH c⟩)

/-- `|Hist|·π ≤ 2^-128` at `N = 448`, `B = 2^9`, `ℓ = 524288`, `D = 5`, `Q = 2^20`, `k = 106`, `β = 119067414778`.
`N*B = 229376 ≤ 2^18`, `(N*B(Q+1)+N*B)^2 ≤ 2^76`, and the exponent `(524288−76)·227136 − (18·106 + 18 + β) = 128`. -/
theorem seg_union_term_14 :
    (∑ j ∈ Finset.range 106, ((448 * 2 ^ 9 : ℕ) : ℝ≥0∞) ^ j) *
      (((448 * 2 ^ 9 : ℕ) : ℝ≥0∞) * 2 ^ 119067414778 *
        (((((448 * 2 ^ 9) * (2 ^ 20 + 1) + 448 * 2 ^ 9) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ 524288) ^
          (448 * 2 ^ 9 - 448 * 5 - 1 + 1)) ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 := by
  have he : 448 * 2 ^ 9 - 448 * 5 - 1 + 1 = 227136 := by norm_num
  have hsq : ((448 * 2 ^ 9) * (2 ^ 20 + 1) + 448 * 2 ^ 9) ^ 2 ≤ 2 ^ 76 := by norm_num
  have hNn : ((448 * 2 ^ 9 : ℕ) : ℝ≥0∞) ≤ (2 : ℝ≥0∞) ^ 18 := by norm_num
  rw [he]
  calc _ ≤ ((2 : ℝ≥0∞) ^ 18) ^ 106 *
        (((2 : ℝ≥0∞) ^ 18) * 2 ^ 119067414778 * ((2⁻¹ : ℝ≥0∞) ^ (524288 - 76)) ^ 227136) := by
        gcongr
        · exact (sum_pow_le _ _ (by norm_num)).trans (pow_le_pow_left' hNn 106)
        · exact sq_div_two_pow_le _ 76 524288 hsq (by norm_num)
    _ = (2 : ℝ≥0∞) ^ (18 * 106) *
          ((2 : ℝ≥0∞) ^ 18 * 2 ^ 119067414778 * (2⁻¹ : ℝ≥0∞) ^ ((524288 - 76) * 227136)) := by
        rw [← pow_mul, ← pow_mul]
    _ = (2⁻¹ : ℝ≥0∞) ^ ((524288 - 76) * 227136 - (18 * 106 + 18 + 119067414778)) :=
        two_pow_combine _ _ _ _ (by norm_num)
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 128 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

/-- `p₀^106 ≤ 1%` at `β = 119067414778`, `S = 113929658799`, `B = 448·2^9`, `ℓ = 524288`. -/
theorem seg_sampling_term_14 :
    ENNReal.ofReal (p₀ 119067414778 113929658799 (448 * 2 ^ 9) 524288) ^ 106 ≤
      ENNReal.ofReal (Params.δ : ℝ) := by
  have h0 : 0 ≤ p₀ 119067414778 113929658799 (448 * 2 ^ 9) 524288 := by unfold p₀; norm_num
  rw [← ENNReal.ofReal_pow h0]
  refine ENNReal.ofReal_le_ofReal ?_
  unfold p₀
  norm_num [Params.δ]

/-- `S = ⌊(18/19)·448·2^9·524288⌋ = 113929658799` at 14 GB. -/
theorem stateBound_seg_14 : (segScheme 448 (2 ^ 9) 524288).stateBound Params.ρ = 113929658799 := by
  show ⌊Params.ρ * (((448 * 2 ^ 9) * 524288 : ℕ) : ℚ)⌋₊ = 113929658799
  rw [Nat.floor_eq_iff (by norm_num [Params.ρ])]
  norm_num [Params.ρ]

/-- **The segmented dense scheme meets the requirement at 14 GB** (`N = 448` segments of `2^9` blocks of 64 KB),
random-oracle `H`, sequential reveal, `D = 5`, `Q = 2^20`, `k = 106`, `ε = 2^-128`. Same `k` as one segment: the audit
soundness is per-block, so it does not degrade with the number of segments. -/
theorem seg_meets_14 :
    Pous.Meets (segScheme 448 (2 ^ 9) 524288) .sequential 5 (2 ^ 20) 106 ((2⁻¹ : ENNReal) ^ 128) := by
  refine ⟨le_rfl, segScheme_correct _ _ _, le_rfl, ?_, ?_⟩
  · show ((((448 * 2 ^ 9) * 524288 + 0 : ℕ)) : ℚ) ≤ Pous.Params.expansion * (((448 * 2 ^ 9) * 524288 : ℕ) : ℚ)
    norm_num [Pous.Params.expansion]
  · intro W A₁ A₂
    have hINC := seg_timedINC 448 (2 ^ 9) 524288 5 (2 ^ 20) 119067414778 (by norm_num) (by norm_num)
    refine (PousProofs.theorem1_ii_seq (segScheme 448 (2 ^ 9) 524288) 5 (2 ^ 20) 119067414778
      ((segScheme 448 (2 ^ 9) 524288).stateBound Pous.Params.ρ) 106 _ hINC W A₁ A₂).trans ?_
    rw [stateBound_seg_14, add_comm]
    exact add_le_add seg_sampling_term_14 seg_union_term_14

/-! ### Edge cases and satisfiability witnesses (the `DecodeHyp3` lesson)

P3's lesson: a hypothesis can be vacuously satisfiable (`ColumnHard` with `k > n`, premise never met) or a statement can
be false on degenerate instances (a binomial position term at `n = 0` or tiny `Q`). The dense theorems are checked here.

* **No binomial term, so no `Q`/`n` degeneracy.** The dense and segmented bounds are `(∑_{j<k} B^j)·π + p₀^k` (B5 +
  `theorem1_ii_seq`) — there is **no** `C(·,·)` position factor, so the "false at `n = 0` / tiny `Q`" failure mode does
  not exist. The collision `((B(Q+1)+B)²/2^ℓ)^(s+1)` is monotone increasing in `Q`, so a smaller `Q` only shrinks the
  union term; no lower bound on `Q` is needed (`dense_timedINC`/`seg_timedINC` are proved for every `Q`).
* **`k` vs `B` — non-vacuity of the pebbling lemma.** `PebblingHard G univ s D k` is non-vacuous only for `k ≤ |univ|`:
  if `k > N*B` the conclusion `filter.card < k` is automatic (`filter.card ≤ |univ| = N*B < k`), exactly P3's `k > n`
  vacuity. `seg_hard` uses `k = N*B` **exactly** (tight), and the audit `k = 106 < B = 512 ≤ N*B`, so neither route is
  vacuous. -/

/-- The audit `k = 106` is below one segment's block count `B = 2^9`, so no `k > B` vacuity in the audit. -/
theorem audit_k_lt_B : (106 : ℕ) < 2 ^ 9 := by norm_num

/-- **Satisfiability witness** for `seg_hard`/`seg_timedINC`'s `hN : 0 < N` and `hD : D + 1 ≤ B`, at the 14 GB point.
`k = N*B = 229376 > 0` and it equals `|univ|`, so the conclusion is a genuine (tight, non-vacuous) `PebblingHard`. -/
theorem seg_hard_14 :
    PebblingHard (segDAG 448 (2 ^ 9)) Finset.univ (448 * 2 ^ 9 - 448 * 5 - 1) 5 (448 * 2 ^ 9) :=
  seg_hard 448 (2 ^ 9) 5 (by norm_num) (by norm_num)

/-- **Satisfiability witness** for the `[NeZero (N*B)]` instance the segmented scheme takes. -/
theorem segNeZero_witness : NeZero (448 * 2 ^ 9) := inferInstance

/-- **`hN : 0 < N` is necessary, not merely convenient.** At `N = 0` the challenge count `N*B = 0`, and
`PebblingHard _ _ _ _ 0` is false: the empty holding already fails `0 < 0`. So dropping `hN` makes `seg_hard` false, not
vacuously true. -/
theorem seg_hard_needs_hN :
    ¬ PebblingHard (segDAG 0 (2 ^ 9)) Finset.univ (0 * 2 ^ 9 - 0 * 5 - 1) 5 (0 * 2 ^ 9) := by
  intro h
  have hc := h ∅ (by simp)
  simp only [Nat.zero_mul] at hc
  exact Nat.not_lt_zero _ hc

/-- **`hD : D + 1 ≤ B` is necessary.** With `D ≥ B` (here `N = B = D = 1`), one round already pebbles the whole segment
from nothing, so all `N*B` challenges are reached and `PebblingHard` fails. -/
theorem seg_hard_needs_hD :
    ¬ PebblingHard (segDAG 1 1) Finset.univ (1 * 1 - 1 * 1 - 1) 1 (1 * 1) := by
  intro h
  have hc := h ∅ (by simp)
  revert hc
  decide

/-! ### Piece 3: alignment with the implementation's per-segment tagged domains

The reference codec domain-separates each segment `s` by `tag_s = salt ‖ u64 s` — a **192-bit** salt then `u64 s`, exactly
`chainDense`'s 256-bit salt slot (`chainDense B ℓ 512 (dom tag_s)`, `PROTOCOL.md`) — whereas `segScheme` keys the oracle by
the **global** block index. These are the *same* domain separation. `segIndexEquiv` shows the global index is exactly the
`(segment, block)` pair (a bijection), and `tag_injective` shows distinct segments get distinct tags (for `N ≤ 2^64`, from
`u64` alone). Then `domainSep_equiv` shows the per-segment tagged keying `(s, i) ↦ (tag_s, i)` is the global-index keying
followed by an **injective relabeling** — so it separates the blocks exactly as global indices do. Hence `seg_meets_14`
(proved with global indices) holds verbatim for the implementation's per-segment domains, and all witnesses and necessity
proofs above are unchanged.

**Salt-entropy independence.** Nothing here needs the salt to carry more than 128 bits of entropy — in fact it needs
none. The audit bound `2^-128` in `dense_meets_64`/`seg_meets_14` comes entirely from the union term (see
`dense_union_term_64`/`seg_union_term_14`); those theorems do not mention the salt. Inter-segment separation
(`tag_injective`, `domainSep_equiv`) comes from `u64 s`, not from salt entropy — the salt enters only as the opaque first
argument of `Fin.append` and its value is never used. So any salt width (here 192 bits) works; the salt only separates
*instances*/deployments, which is outside this per-store bound. -/

theorem u64_inj {a b : ℕ} (ha : a < 2 ^ 64) (hb : b < 2 ^ 64)
    (h : ∀ j : Fin 64, a.testBit j.val = b.testBit j.val) : a = b := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 64
  · exact h ⟨i, hi⟩
  · push_neg at hi
    rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le ha (Nat.pow_le_pow_right (by norm_num) hi)),
      Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hb (Nat.pow_le_pow_right (by norm_num) hi))]

/-- **Distinct segments get distinct tags** (`N ≤ 2^64`): `salt ‖ u64 s` is injective in `s`. -/
theorem tag_injective (salt : Bits 192) {N : ℕ} (hN : N ≤ 2 ^ 64) :
    Function.Injective (fun s : Fin N => tag salt s.val) := by
  intro s t h
  refine Fin.ext (u64_inj (lt_of_lt_of_le s.isLt hN) (lt_of_lt_of_le t.isLt hN) (fun j => ?_))
  have hj := congrFun h (Fin.natAdd 192 j)
  simpa [tag, u64, Fin.append_right] using hj

/-- **Distinct tags give the same domain separation as global indices.** The per-segment tagged keying
`(s, i) ↦ (tag_s, i)` is the global-index keying (`segIndexEquiv`) composed with an injective relabeling `t`. So keying by
`tag_s = salt ‖ u64 s` separates the `N·B` blocks exactly as global indices do; `seg_meets_14` transfers verbatim. -/
theorem domainSep_equiv (N B : ℕ) (hN : N ≤ 2 ^ 64) (salt : Bits 192) :
    ∃ t : Fin (N * B) → Bits (192 + 64) × Fin B, Function.Injective t ∧
      ∀ p : Fin N × Fin B, (tag salt p.1.val, p.2) = t (segIndexEquiv N B p) := by
  refine ⟨fun g => (tag salt ((segIndexEquiv N B).symm g).1.val, ((segIndexEquiv N B).symm g).2), ?_, ?_⟩
  · intro g₁ g₂ h
    simp only [Prod.mk.injEq] at h
    exact (segIndexEquiv N B).symm.injective (Prod.ext (tag_injective salt hN h.1) h.2)
  · intro p
    simp only [Equiv.symm_apply_apply]

/-- **Satisfiability witness** at 14 GB: `N = 448 ≤ 2^64`, so the implementation's tags `salt ‖ u64 s` are distinct across
the 448 segments — the hypothesis `hN` of `domainSep_equiv`/`tag_injective` holds, and the alignment applies. -/
theorem tag_injective_14 (salt : Bits 192) : Function.Injective (fun s : Fin 448 => tag salt s.val) :=
  tag_injective salt (by norm_num)

end PousDenseReCert
