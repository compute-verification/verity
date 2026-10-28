import Definitions.Pouw.Dimension.Defs
import Definitions.Pouw.Assumptions.Dimension.Pinned
import Proofs.Pouw.NCP.RelationProofs
import Proofs.Pouw.NCP.RelationWitness
import Proofs.Pouw.NCP.ChainProofs

/-!
# Theorem D: proofs

* `dimLePriced`: one dimension per priced value; `count_le`, the generic count.
* `secondDifference`: the four-point evaluation lemma, jointly over units and entries. The mixed difference kills
  `V₀` (`sd_V0`), and `A = g(F₁)` drops out because the words have no `A·E` term (`mixdiff2`).
* `pricedCount`, `algBound` (Theorem D), `ttNCPAlg` (TT_NCP in AlgM_16 at `γ₀ = 16/(3k)`).
* `algWitness`, `algSharp`: the hypotheses hold jointly, the bound is attained, and `16/(3k)` is sharp.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.Dimension.Assumptions

variable {G : Type}

/-! ## Programs -/

theorem slots_nil (L : List (G → ℚ)) : slots [] L = L := rfl

theorem slots_cons (ins : Instr G) (P : List (Instr G)) (L : List (G → ℚ)) :
    slots (ins :: P) L = slots P (L ++ [ins.out L]) := rfl

theorem mem_slots_of_mem (P : List (Instr G)) {L : List (G → ℚ)} {f : G → ℚ} (h : f ∈ L) : f ∈ slots P L := by
  induction P generalizing L with
  | nil => exact h
  | cons ins P ih => rw [slots_cons]; exact ih (List.mem_append_left _ h)

theorem mem_slots_of_priced {P : List (Instr G)} {c : ℕ} {f : G → ℚ} (h : Instr.priced c f ∈ P)
    (L : List (G → ℚ)) : f ∈ slots P L := by
  induction P generalizing L with
  | nil => simp at h
  | cons ins P ih =>
    rw [slots_cons]
    rcases List.mem_cons.1 h with h | h
    · subst h
      exact mem_slots_of_mem P (List.mem_append_right _ (List.mem_singleton_self _))
    · exact ih h _

theorem span1_append (L : List (G → ℚ)) (x : G → ℚ) :
    span1 (L ++ [x]) = Submodule.span ℚ {x} ⊔ span1 L := by
  have : (insert 1 {f | f ∈ L ++ [x]} : Set (G → ℚ)) = insert x (insert 1 {f | f ∈ L}) := by
    ext f
    simp only [Set.mem_insert_iff, Set.mem_ofPred_eq, List.mem_append, List.mem_singleton]
    tauto
  rw [span1, this, Submodule.span_insert]
  rfl

theorem out_free_mem (c : ℚ) (w : ℕ → ℚ) (L : List (G → ℚ)) : (Instr.free c w).out L ∈ span1 L := by
  have h : (Instr.free c w).out L = c • (1 : G → ℚ) + ∑ s ∈ range L.length, w s • L.getD s 0 := by
    funext g
    simp [Instr.out, Finset.sum_apply]
  rw [h]
  refine Submodule.add_mem _ (Submodule.smul_mem _ _ (Submodule.subset_span (Set.mem_insert _ _)))
    (Submodule.sum_mem _ fun s hs => Submodule.smul_mem _ _ (Submodule.subset_span (Set.mem_insert_of_mem _ ?_)))
  have hs' : s < L.length := mem_range.1 hs
  show L.getD s 0 ∈ L
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hs', Option.getD_some]
  exact List.getElem_mem hs'

theorem finrank_step [Fintype G] (ins : Instr G) (L : List (G → ℚ)) :
    Module.finrank ℚ (span1 (L ++ [ins.out L])) ≤
      Module.finrank ℚ (span1 L) + (if ins.isPriced then 1 else 0) := by
  rw [span1_append]
  cases ins with
  | free c w =>
    rw [sup_eq_right.2 ((Submodule.span_singleton_le_iff_mem _ _).2 (out_free_mem c w L))]
    simp [Instr.isPriced]
  | priced p f =>
    have h1 := Submodule.finrank_add_le_finrank_add_finrank (Submodule.span ℚ {(Instr.priced p f).out L}) (span1 L)
    have h2 : Module.finrank ℚ (Submodule.span ℚ {(Instr.priced p f).out L}) ≤ 1 := by
      have := finrank_span_finset_le_card (R := ℚ) ({(Instr.priced p f).out L} : Finset (G → ℚ))
      rw [Set.finrank, Finset.coe_singleton] at this
      exact this
    simp only [Instr.isPriced, ite_true]
    omega

theorem nPriced_cons (ins : Instr G) (P : List (Instr G)) :
    nPriced (ins :: P) = nPriced P + (if ins.isPriced then 1 else 0) := by
  simp only [nPriced, List.countP_cons]

/-- **One dimension per priced value.** -/
theorem dimLePriced : DimLePriced := by
  intro G _ P
  induction P with
  | nil => intro L; exact Nat.le_add_right _ _
  | cons ins P ih =>
    intro L
    rw [slots_cons, nPriced_cons]
    have := ih (L ++ [ins.out L])
    have := finrank_step ins L
    omega

/-- **The generic count**: if targets are linearly independent modulo `V(inputs)` and every target is a slot, the
program has at least as many priced instructions as targets. -/
theorem count_le {ι : Type} [Fintype G] [Fintype ι] (P : List (Instr G)) (L : List (G → ℚ)) (t : ι → G → ℚ)
    (hind : ∀ c : ι → ℚ, ∑ i, c i • t i ∈ span1 L → ∀ i, c i = 0) (hout : ∀ i, t i ∈ slots P L) :
    Fintype.card ι ≤ nPriced P := by
  classical
  have hli : LinearIndependent ℚ t :=
    Fintype.linearIndependent_iff.2 fun c hc => hind c (hc ▸ Submodule.zero_mem _)
  set T := Submodule.span ℚ (Set.range t)
  have hT : Module.finrank ℚ T = Fintype.card ι := finrank_span_eq_card hli
  have hinf : T ⊓ span1 L = ⊥ := by
    rw [eq_bot_iff]
    rintro x ⟨hxT, hxV⟩
    obtain ⟨c, rfl⟩ := (Submodule.mem_span_range_iff_exists_fun ℚ).1 hxT
    have hc := hind c hxV
    simp [hc]
  have hsup : T ⊔ span1 L ≤ span1 (slots P L) := by
    refine sup_le (Submodule.span_le.2 ?_) (Submodule.span_mono ?_)
    · rintro _ ⟨i, rfl⟩
      exact Submodule.subset_span (Set.mem_insert_of_mem _ (hout i))
    · intro f hf
      rcases hf with hf | hf
      · exact Or.inl hf
      · exact Or.inr (mem_slots_of_mem P hf)
  have h1 := Submodule.finrank_sup_add_finrank_inf_eq T (span1 L)
  rw [hinf, finrank_bot] at h1
  have h2 := Submodule.finrank_mono hsup
  have h3 := dimLePriced G P L
  omega

/-! ## The four-point evaluation lemma -/

/-- The four-point mixed difference at the probe `(u, i, a, b, j)` (`RelationProofs.pt`). -/
def sd (N m n k u i a b j : ℕ) : (Noise N m n k → ℚ) →ₗ[ℚ] ℚ :=
  LinearMap.proj (R := ℚ) (φ := fun _ : Noise N m n k => ℚ) (pt N m n k true true u i a b j) -
    LinearMap.proj (R := ℚ) (φ := fun _ : Noise N m n k => ℚ) (pt N m n k true false u i a b j) -
    LinearMap.proj (R := ℚ) (φ := fun _ : Noise N m n k => ℚ) (pt N m n k false true u i a b j) +
    LinearMap.proj (R := ℚ) (φ := fun _ : Noise N m n k => ℚ) (pt N m n k false false u i a b j)

theorem sd_apply {N m n k u i a b j : ℕ} (φ : Noise N m n k → ℚ) :
    sd N m n k u i a b j φ = φ (pt N m n k true true u i a b j) - φ (pt N m n k true false u i a b j) -
      φ (pt N m n k false true u i a b j) + φ (pt N m n k false false u i a b j) := rfl

/-- The mixed difference kills the affine functions of the noise. -/
theorem sd_coords {N m n k : ℕ} (u i a b j : ℕ) {φ : Noise N m n k → ℚ} (h : φ ∈ span1 (coords N m n k)) :
    sd N m n k u i a b j φ = 0 := by
  have hle : span1 (coords N m n k) ≤ LinearMap.ker (sd N m n k u i a b j) := by
    refine Submodule.span_le.2 ?_
    intro f hf
    rcases hf with rfl | hf
    · simp [sd_apply]
    · simp only [Set.mem_ofPred_eq, coords, List.mem_append, List.mem_flatMap, List.mem_map,
        List.mem_finRange, true_and] at hf
      rcases hf with ⟨u', i', l', rfl⟩ | ⟨l', j', rfl⟩
      · show sd N m n k u i a b j _ = 0
        simp only [sd_apply, pt]
        ring
      · show sd N m n k u i a b j _ = 0
        simp only [sd_apply, pt]
        ring
  exact hle h

/-- The mixed difference kills the functions of `F₁`: the probes `(true, bF)` and `(false, bF)` share `F₁`. -/
theorem sd_fnF {N m n k : ℕ} (u i a b j : ℕ) {φ : Noise N m n k → ℚ} (h : φ ∈ fnF N m n k) :
    sd N m n k u i a b j φ = 0 := by
  have hle : fnF N m n k ≤ LinearMap.ker (sd N m n k u i a b j) := by
    refine Submodule.span_le.2 fun f hf => ?_
    have hf' : ∀ ω ω' : Noise N m n k, ω.2 = ω'.2 → f ω = f ω' := hf
    show sd N m n k u i a b j f = 0
    rw [sd_apply, hf' (pt N m n k false true u i a b j) (pt N m n k true true u i a b j) rfl,
      hf' (pt N m n k false false u i a b j) (pt N m n k true false u i a b j) rfl]
    ring
  exact hle h

/-- The mixed difference kills `V₀`. -/
theorem sd_V0 {N m n k : ℕ} (u i a b j : ℕ) {φ : Noise N m n k → ℚ} (h : φ ∈ V0 N m n k) :
    sd N m n k u i a b j φ = 0 := by
  obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.1 h
  rw [map_add, sd_coords u i a b j hy, sd_fnF u i a b j hz, add_zero]

theorem coords_V0 {N m n k : ℕ} : ∀ f ∈ coords N m n k, f ∈ V0 N m n k := fun _ hf =>
  Submodule.mem_sup_left (Submodule.subset_span (Set.mem_insert_of_mem _ hf))

theorem span1_le_V0 {N m n k : ℕ} {L : List (Noise N m n k → ℚ)} (hL : ∀ f ∈ L, f ∈ V0 N m n k) :
    span1 L ≤ V0 N m n k := by
  refine Submodule.span_le.2 fun f hf => ?_
  rcases hf with rfl | hf
  · exact Submodule.mem_sup_left (Submodule.subset_span (Set.mem_insert _ _))
  · exact hL f hf

/-- The mixed difference with `A` read at `F₁ = F` (`A₁`) and at `F₁ = 0` (`A₀`): no `A·E` term, so `A`
drops out. -/
theorem mixdiff2 (k : ℕ) (A1 A0 B E F R : Mat) (i j τ : ℕ) :
    ck k A1 B E F R i j τ - ck k A0 B E 0 R i j τ - ck k A1 B 0 F R i j τ + ck k A0 B 0 0 R i j τ =
      ck k 0 0 E F R i j τ := by
  simp only [ck, ← sum_sub_distrib, ← sum_add_distrib]
  refine sum_congr rfl fun l _ => ?_
  rw [stackX_split k A1 E, stackX_split k A0 E, stackY_split k B R F]
  ring

theorem e1Of_pt_ne {N m n k : ℕ} (bE bF : Bool) {u0 u : ℕ} (i a b j : ℕ) (h : u ≠ u0) :
    e1Of (pt N m n k bE bF u0 i a b j) u = e1Of (pt N m n k false bF u0 i a b j) u := by
  funext x y
  simp [e1Of, pt, h]

theorem sd_word_ne {k m n N : ℕ} (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) {u0 u : ℕ}
    (i a b j : ℕ) (h : u ≠ u0) (p : ℕ) :
    sd N m n k u0 i a b j (word k m n N R A B u p) = 0 := by
  have hE : ∀ bF, checked k m n N R (A (pt N m n k true bF u0 i a b j).2) B (pt N m n k true bF u0 i a b j) u p =
      checked k m n N R (A (pt N m n k false bF u0 i a b j).2) B (pt N m n k false bF u0 i a b j) u p := by
    intro bF
    unfold checked
    rw [e1Of_pt_ne true bF i a b j h]
    rfl
  simp only [sd_apply, word, hE]
  ring

theorem sd_word_self {k m n N : ℕ} (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {u0 e a b : ℕ} (hu : u0 < N) (he : e < m * n) (ha : a < k) (hb : b < k) (p : ℕ) :
    sd N m n k u0 (e / n) a b (e % n) (word k m n N R A B u0 p) = (coefAt k m n R p e a b : ℚ) := by
  have hn : 0 < n := Nat.pos_of_ne_zero fun h => by simp [h] at he
  have hi : e / n < m := (Nat.div_lt_iff_lt_mul hn).2 he
  have hj : e % n < n := Nat.mod_lt e hn
  have h := fun bE bF => congrFun (checked_pt (N := N) R (A (pt N m n k bE bF u0 (e / n) a b (e % n)).2) B bE bF
    hu hi ha hb hj) p
  have hpt : ∀ bF, (pt N m n k false bF u0 (e / n) a b (e % n)).2 = (pt N m n k true bF u0 (e / n) a b (e % n)).2 :=
    fun _ => rfl
  simp only [sd_apply, word, h, ite_true, Bool.false_eq_true, ite_false]
  rw [hpt true, hpt false, ← coef_eq R e a b p, ← mixdiff2 k (A (pt N m n k true true u0 (e / n) a b (e % n)).2 u0)
    (A (pt N m n k true false u0 (e / n) a b (e % n)).2 u0) (B u0)]
  push_cast
  ring

theorem sum_blocks (M e : ℕ) (he : e < M) (h : ℕ → ℚ) :
    ∀ T, ∑ p ∈ range (T * M), (if p % M = e then h p else 0) = ∑ t ∈ range T, h (t * M + e) := by
  intro T
  induction T with
  | zero => simp
  | succ T ih =>
    rw [Nat.succ_mul, sum_range_add, ih, sum_range_succ]
    congr 1
    rw [sum_congr rfl fun r hr => by rw [add_comm (T * M) r, Nat.add_mul_mod_self_right,
      Nat.mod_eq_of_lt (mem_range.1 hr)]]
    rw [sum_ite_eq' (range M) e, ite_eq_left_iff.2 fun h => absurd (mem_range.2 he) h, add_comm]

theorem div_blocks {M e t : ℕ} (he : e < M) : (t * M + e) / M = t := by
  have hM : 0 < M := by omega
  rw [add_comm, Nat.add_mul_div_right _ _ hM, Nat.div_eq_of_lt he, zero_add]

/-- **The four-point evaluation lemma.** -/
theorem secondDifference : SecondDifference := by
  intro k m n N R A B U c hU hsum u0 hu0 e he a ha b hb
  have hN : u0 < N := mem_range.1 (hU hu0)
  have h0 := sd_V0 u0 (e / n) a b (e % n) hsum
  rw [map_sum, sum_eq_single u0 (fun u _ hne => by
      rw [map_sum]
      exact sum_eq_zero fun p _ => by rw [map_smul, sd_word_ne R A B _ _ _ _ hne, smul_zero])
    (fun h => absurd hu0 h), map_sum] at h0
  simp only [map_smul, smul_eq_mul, sd_word_self R A B hN he ha hb, coefAt] at h0
  push_cast at h0
  have hM : e < m * n := he
  rw [← h0, nonFinal]
  simp only [mul_ite, mul_zero]
  rw [sum_blocks (m * n) e hM (fun p => c u0 p * (ckMat k R (16 * (p / (m * n) + 1)) a b : ℚ))]
  refine sum_congr rfl fun t _ => ?_
  rw [div_blocks hM]

/-! ## Independence over ℚ -/

theorem indepQ {k : ℕ} {R : Mat} (h : CheckpointIndependent k 16 R) (T : ℕ) (hT : 16 * T < 3 * k) (q : ℕ → ℚ)
    (hq : ∀ a < k, ∀ b < k, ∑ t ∈ range T, q t * (ckMat k R (16 * (t + 1)) a b : ℚ) = 0) :
    ∀ t < T, q t = 0 := by
  classical
  set D : ℕ := ∏ s ∈ range T, (q s).den with hDdef
  have hD : (0 : ℚ) < D := Nat.cast_pos.2 (prod_pos fun s _ => (q s).den_pos)
  obtain ⟨z, hz⟩ : ∃ z : ℕ → ℤ, ∀ t < T, (z t : ℚ) = q t * D := by
    refine ⟨fun t => (q t).num * ∏ s ∈ (range T).erase t, ((q s).den : ℤ), fun t ht => ?_⟩
    push_cast
    rw [hDdef, Nat.cast_prod, ← mul_prod_erase (range T) (fun s => ((q s).den : ℚ)) (mem_range.2 ht), ← mul_assoc,
      Rat.mul_den_eq_num]
  have hzero := h T z hT fun a ha b hb => by
    have h1 : ((∑ t ∈ range T, z t * ckMat k R (16 * (t + 1)) a b : ℤ) : ℚ) = 0 := by
      push_cast
      rw [sum_congr rfl fun t ht => by rw [hz t (mem_range.1 ht)]]
      have : ∑ t ∈ range T, q t * (D : ℚ) * (ckMat k R (16 * (t + 1)) a b : ℚ) =
          (D : ℚ) * ∑ t ∈ range T, q t * (ckMat k R (16 * (t + 1)) a b : ℚ) := by
        rw [mul_sum]
        exact sum_congr rfl fun t _ => by ring
      rw [this, hq a ha b hb, mul_zero]
    exact_mod_cast h1
  intro t ht
  have h2 : q t * D = 0 := by rw [← hz t ht, hzero t ht]; simp
  exact (mul_eq_zero.1 h2).resolve_right hD.ne'

/-- Under independence, the non-final checked words are linearly independent modulo `V₀`. -/
theorem indep_mod {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) {U : Finset ℕ}
    (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) (c : ℕ → ℕ → ℚ)
    (hc : (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • word k m n N R A B u p) ∈ V0 N m n k) :
    ∀ u ∈ U, ∀ p < nonFinal k m n, c u p = 0 := by
  intro u hu p hp
  have hmn : 0 < m * n := Nat.pos_of_ne_zero fun h => by simp [nonFinal, h] at hp
  have hp' : p / (m * n) < 3 * k / 16 - 1 := (Nat.div_lt_iff_lt_mul hmn).2 hp
  have hT : 0 < 3 * k / 16 - 1 := lt_of_le_of_lt (Nat.zero_le _) hp'
  have hk : 16 * (3 * k / 16 - 1) < 3 * k := by omega
  have := indepQ hind (3 * k / 16 - 1) hk (fun t => c u (t * (m * n) + p % (m * n)))
    (fun a ha b hb => secondDifference k m n N R A B U c hU hc u hu (p % (m * n)) (Nat.mod_lt _ hmn) a ha b hb)
    (p / (m * n)) hp'
  rwa [Nat.div_add_mod'] at this

/-! ## Theorem D -/

/-- The non-final checked words, indexed by `U ×ˢ range (nonFinal k m n)`, are linearly independent modulo
`V₀`. -/
theorem indep_sub {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) {U : Finset ℕ}
    (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) (c : ↥(U ×ˢ range (nonFinal k m n)) → ℚ)
    (hc : ∑ x, c x • word k m n N R A B x.1.1 x.1.2 ∈ V0 N m n k) : ∀ x, c x = 0 := by
  classical
  let c' : ℕ → ℕ → ℚ := fun u p => if h : (u, p) ∈ U ×ˢ range (nonFinal k m n) then c ⟨(u, p), h⟩ else 0
  have hsum : ∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c' u p • word k m n N R A B u p =
      ∑ x : ↥(U ×ˢ range (nonFinal k m n)), c x • word k m n N R A B x.1.1 x.1.2 := by
    rw [← sum_product (s := U) (t := range (nonFinal k m n)) (f := fun x => c' x.1 x.2 • word k m n N R A B x.1 x.2),
      ← sum_coe_sort (U ×ˢ range (nonFinal k m n))]
    refine sum_congr rfl fun x _ => ?_
    have hx : ((x : ℕ × ℕ).1, (x : ℕ × ℕ).2) ∈ U ×ˢ range (nonFinal k m n) := x.2
    simp only [c', dif_pos hx]
  rintro ⟨⟨u, p⟩, hx⟩
  have := indep_mod A B hind hU c' (hsum ▸ hc) u (mem_product.1 hx).1 p (mem_range.1 (mem_product.1 hx).2)
  rw [show c' u p = c ⟨(u, p), hx⟩ from dif_pos hx] at this
  exact this

/-- **The priced count.** -/
theorem pricedCount : PricedCount := by
  intro k m n N R A B U L P hind hU hL hout
  have hcount := count_le P L
    (fun x : ↥(U ×ˢ range (nonFinal k m n)) => word k m n N R A B x.1.1 x.1.2)
    (fun c hc => indep_sub A B hind hU c (span1_le_V0 hL hc))
    (fun x => hout x.1.1 (mem_product.1 x.2).1 x.1.2 (mem_range.1 (mem_product.1 x.2).2))
  rw [Fintype.card_coe, card_product, card_range, nonFinal] at hcount
  calc U.card * (m * n) * (3 * k / 16 - 1) = U.card * ((3 * k / 16 - 1) * (m * n)) := by ring
    _ ≤ nPriced P := hcount

theorem cost_cons (ins : Instr G) (P : List (Instr G)) : cost (ins :: P) = ins.price + cost P := by
  simp [cost]

theorem h16_cost {P : List (Instr G)} (h : H16 P) : 16 * nPriced P ≤ cost P := by
  induction P with
  | nil => simp [nPriced, cost]
  | cons ins P ih =>
    rw [nPriced_cons, cost_cons]
    have h1 := ih fun x hx => h x (List.mem_cons_of_mem _ hx)
    have h2 := h ins (by simp)
    cases ins with
    | free c w => simp only [Instr.isPriced, Instr.price, Bool.false_eq_true, ite_false] at h2 ⊢; omega
    | priced p f => simp only [Instr.isPriced, Instr.price, ite_true, forall_const] at h2 ⊢; omega

/-- **Theorem D.** -/
theorem algBound : AlgBound := by
  intro k m n N R A B U L P hind hU hL hH hout
  have := pricedCount k m n N R A B U L P hind hU hL hout
  have := h16_cost hH
  omega

/-- **TT_NCP in AlgM_16** at `γ₀ = 16/(3k)`. -/
theorem ttNCPAlg : TTNCPAlg := by
  intro k m n N R A B U L P γ₀ h16 hk hind hU hL hH hout hγ hγ1 hev
  have hb := algBound k m n N R A B U L P hind hU hL hH hout
  obtain ⟨k', rfl⟩ := h16
  have hk' : 1 ≤ k' := by omega
  have h3 : 3 * (16 * k') / 16 - 1 = 3 * k' - 1 := by omega
  rw [h3] at hb
  have hbR : (16 : ℝ) * (U.card * (m * n) * (3 * k' - 1)) ≤ cost P := by
    have : ((16 * (U.card * (m * n) * (3 * k' - 1)) : ℕ) : ℝ) ≤ cost P := by exact_mod_cast hb
    push_cast [Nat.cast_sub (by omega : 1 ≤ 3 * k')] at this
    linarith
  rw [sum_const, nsmul_eq_mul] at hev
  have hpos : 0 < 1 - γ₀ := by linarith
  rw [div_lt_iff₀ hpos] at hev
  push_cast at hev hγ
  have h48 : 16 ≤ γ₀ * (3 * (16 * (k' : ℝ))) := (div_le_iff₀ (by positivity)).1 hγ
  have hX : (0 : ℝ) ≤ U.card * (m * n) := by positivity
  nlinarith [mul_le_mul_of_nonneg_left h48 hX]

/-! ## Witnesses and sharpness -/

/-- One priced instruction at price `c` per non-final checked word of the units in `U`. -/
noncomputable def honest (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) (c : ℕ) : List (Instr (Noise N m n k)) :=
  U.toList.flatMap fun u => (List.range (nonFinal k m n)).map fun p => Instr.priced c (word k m n N R A B u p)

theorem mem_honest {k m n N : ℕ} {R : Mat} {A : (Fin k → Fin n → Fin 64) → ℕ → Mat} {B : ℕ → Mat} {U : Finset ℕ}
    {c : ℕ} {ins : Instr (Noise N m n k)} :
    ins ∈ honest k m n N R A B U c ↔ ∃ u ∈ U, ∃ p < nonFinal k m n, Instr.priced c (word k m n N R A B u p) = ins := by
  simp [honest, List.mem_flatMap, List.mem_map]

theorem honest_h16 {k m n N : ℕ} {R : Mat} {A : (Fin k → Fin n → Fin 64) → ℕ → Mat} {B : ℕ → Mat}
    {U : Finset ℕ} {c : ℕ} (hc : 16 ≤ c) : H16 (honest k m n N R A B U c) := by
  intro ins hins _
  obtain ⟨u, _, p, _, rfl⟩ := mem_honest.1 hins
  exact hc

theorem honest_outputs {k m n N : ℕ} {R : Mat} {A : (Fin k → Fin n → Fin 64) → ℕ → Mat} {B : ℕ → Mat}
    {U : Finset ℕ} {c : ℕ} {L : List (Noise N m n k → ℚ)} :
    Outputs k m n N R A B U (honest k m n N R A B U c) L := fun u hu p hp =>
  mem_slots_of_priced (mem_honest.2 ⟨u, hu, p, hp, rfl⟩) _

theorem cost_append (P Q : List (Instr G)) : cost (P ++ Q) = cost P + cost Q := by
  simp [cost]

theorem cost_flatMap {α : Type} (L : List α) (f : α → List (Instr G)) :
    cost (L.flatMap f) = (L.map fun a => cost (f a)).sum := by
  induction L with
  | nil => simp [cost]
  | cons a L ih => rw [List.flatMap_cons, cost_append, ih, List.map_cons, List.sum_cons]

theorem honest_cost (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ) :
    cost (honest k m n N R A B U c) = c * (U.card * (m * n) * (3 * k / 16 - 1)) := by
  have h1 : ∀ u, cost ((List.range (nonFinal k m n)).map fun p => Instr.priced c (word k m n N R A B u p)) =
      c * nonFinal k m n := by
    intro u
    simp only [cost, List.map_map]
    rw [show (Instr.price ∘ fun p => Instr.priced c (word k m n N R A B u p)) = fun _ => c from rfl,
      List.map_const', List.sum_replicate, List.length_range, smul_eq_mul, mul_comm]
  rw [honest, cost_flatMap]
  simp only [h1]
  rw [List.map_const', List.sum_replicate, length_toList, smul_eq_mul, nonFinal]
  ring

/-- **Theorem D's hypotheses hold jointly, and the bound is attained.** -/
theorem algWitness : AlgWitness := by
  intro k hk h16 m n N A B U _
  obtain ⟨R, hadm, hind⟩ := RelationWitness.independentWitness k hk h16
  exact ⟨R, honest k m n N R A B U 16, hadm, hind, coords_V0, honest_h16 le_rfl, honest_outputs,
    honest_cost _ _ _ _ _ _ _ _ _⟩

/-- **`γ₀ = 16/(3k)` is sharp in AlgM_16.** -/
theorem algSharp : AlgSharp := by
  intro k m n N R A B U γ₀ h16 hk _ hne hmn hγ
  refine ⟨honest k m n N R A B U 16, honest_h16 le_rfl, honest_outputs, ?_⟩
  rw [honest_cost]
  obtain ⟨k', rfl⟩ := h16
  have hk' : 1 ≤ k' := by omega
  have h3 : 3 * (16 * k') / 16 - 1 = 3 * k' - 1 := by omega
  rw [h3, sum_const, nsmul_eq_mul]
  have hX : (0 : ℝ) < U.card * (m * n) := by
    have := Nat.cast_pos (α := ℝ).2 (Nat.mul_pos (card_pos.2 hne) hmn)
    push_cast at this
    exact this
  push_cast [Nat.cast_sub (by omega : 1 ≤ 3 * k')] at hγ ⊢
  have hk'R : (1 : ℝ) ≤ k' := by exact_mod_cast hk'
  have hγ' : γ₀ * (3 * (16 * (k' : ℝ))) < 16 := (lt_div_iff₀ (by positivity)).1 hγ
  have hpos : 0 < 1 - γ₀ := by
    have : (16 : ℝ) / (3 * (16 * k')) ≤ 1 / 3 := by
      rw [div_le_iff₀ (by positivity)]
      nlinarith
    linarith
  rw [div_lt_iff₀ hpos]
  nlinarith [mul_lt_mul_of_pos_left hγ' hX]

/-! ## Sanity: the degenerate cases -/

/-- No units: the empty program outputs every word of no unit, at cost `0`, and the bound is `0`. -/
theorem sanity_no_units (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (L : List (Noise N m n k → ℚ)) :
    Outputs k m n N R A B ∅ ([] : List (Instr (Noise N m n k))) L ∧ cost ([] : List (Instr (Noise N m n k))) = 0 ∧
      16 * ((∅ : Finset ℕ).card * (m * n) * (3 * k / 16 - 1)) = 0 :=
  ⟨fun u hu => absurd hu (Finset.notMem_empty u), rfl, by simp⟩

/-- Why the final checkpoint is not counted: at the degenerate input `A = B = 0`, for every `R` and `16 ∣ k`, every
final checked word is the zero function, the output of one free instruction. -/
theorem sanity_final_free {k m n N : ℕ} (R : Mat) (h16 : 16 ∣ k) (u p : ℕ) (hp : nonFinal k m n ≤ p)
    (hp' : p < positions k m n) :
    word k m n N R (fun _ _ => 0) (fun _ => 0) u p = (Instr.free 0 0).out (coords N m n k) := by
  have hmn : 0 < m * n := Nat.pos_of_ne_zero fun h => by simp [positions, h] at hp'
  have hT : 0 < 3 * k / 16 := Nat.pos_of_ne_zero fun h => by simp [positions, h] at hp'
  have hdiv : p / (m * n) = 3 * k / 16 - 1 :=
    Nat.div_eq_of_lt_le hp (by rw [Nat.sub_add_cancel hT]; exact hp')
  have hdepth : 16 * (p / (m * n) + 1) = 3 * k := by rw [hdiv]; omega
  funext ω
  simp only [word, checked, hdepth, Instr.out]
  have := congrFun (congrFun ((NCP.Proofs.stacked k 0 0 (e1Of ω u) (f1Of ω) R).2) (p % (m * n) / n))
    (p % (m * n) % n)
  simp only [ck]
  rw [show (∑ l ∈ range (3 * k), stackX k 0 (e1Of ω u) R (p % (m * n) / n) l *
      stackY k 0 R (f1Of ω) l (p % (m * n) % n)) = mm (3 * k) (stackX k 0 (e1Of ω u) R)
      (stackY k 0 R (f1Of ω)) (p % (m * n) / n) (p % (m * n) % n) from rfl, this]
  simp [mm]

/-- H_16 is needed: at price 1 per word the same program costs `|U|·m·n·(3k/16 − 1)`, 16 times less than the bound. -/
theorem sanity_h16_needed (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) :
    ∃ P : List (Instr (Noise N m n k)), Outputs k m n N R A B U P (coords N m n k) ∧
      cost P = U.card * (m * n) * (3 * k / 16 - 1) :=
  ⟨honest k m n N R A B U 1, honest_outputs, by rw [honest_cost, one_mul]⟩

/-! ## A1 -/

/-- **A1's witness.** -/
theorem a1Witness : A1Witness := by
  intro k R c γ₁ hind hc m n N A B U hU V hV r p _ hVp
  have hinf : V ⊓ V0 N m n k = ⊥ := by
    rw [eq_bot_iff]
    rintro x ⟨hxV, hxA⟩
    obtain ⟨d, rfl⟩ := (Submodule.mem_span_range_iff_exists_fun ℚ).1 (hV hxV)
    have hd := indep_sub A B hind hU d hxA
    simp [hd]
  have h1 := Submodule.finrank_sup_add_finrank_inf_eq V (V0 N m n k)
  rw [hinf, finrank_bot] at h1
  have h2 := Submodule.finrank_mono (sup_le hVp (le_sup_right : V0 N m n k ≤ _))
  have h3 := Submodule.finrank_add_le_finrank_add_finrank (Submodule.span ℚ (Set.range p)) (V0 N m n k)
  have h4 := finrank_range_le_card (R := ℚ) p
  rw [Set.finrank, Fintype.card_fin] at h4
  have hdim : Module.finrank ℚ V ≤ r := by omega
  have hd : (Module.finrank ℚ V : ℝ) ≤ r := by exact_mod_cast hdim
  have h0 : (0 : ℝ) ≤ Module.finrank ℚ V := Nat.cast_nonneg _
  nlinarith

/-- **Theorem D for the protocol's checked values.** -/
theorem algBoundWrapped : AlgBoundWrapped := by
  intro k m n N R A B U L P hind hR hk hU hA hB hL hH hout
  refine algBound k m n N R A B U L P hind hU hL hH fun u hu p hp => ?_
  have hpos : p < positions k m n :=
    lt_of_lt_of_le hp (Nat.mul_le_mul_right _ (Nat.sub_le _ _))
  have heq : (fun ω => (Chain.wrapped k m n N R (A ω.2) B ω u p : ℚ)) = word k m n N R A B u p := by
    funext ω
    simp only [Chain.wrapped, word]
    rw [Pouw.Proofs.wrap32_of_mem
      (ChainProofs.int7Int32 k m n N R (A ω.2) B ω u p hR hk (hA ω.2 u hu) (hB u hu) hpos)]
  rw [← heq]
  exact hout u hu p hp

theorem e1_mem_coords {k : ℕ} {a : ℕ} (ha : a < k) :
    (fun ω : Noise 1 1 1 k => (e1Of ω 0 0 a : ℚ)) ∈ span1 (coords 1 1 1 k) := by
  refine Submodule.subset_span (Set.mem_insert_of_mem _ ?_)
  show _ ∈ coords 1 1 1 k
  refine List.mem_append_left _ (List.mem_flatMap.2 ⟨0, List.mem_finRange _, List.mem_flatMap.2 ⟨0,
    List.mem_finRange _, List.mem_map.2 ⟨⟨a, ha⟩, List.mem_finRange _, ?_⟩⟩⟩)
  funext ω
  simp only [e1Of, dif_pos (show 0 < 1 ∧ 0 < 1 ∧ a < k from ⟨by norm_num, by norm_num, ha⟩)]
  push_cast
  rfl

theorem f1_mem_coords {k : ℕ} {b : ℕ} (hb : b < k) :
    (fun ω : Noise 1 1 1 k => (f1Of ω b 0 : ℚ)) ∈ span1 (coords 1 1 1 k) := by
  refine Submodule.subset_span (Set.mem_insert_of_mem _ ?_)
  show _ ∈ coords 1 1 1 k
  refine List.mem_append_right _ (List.mem_flatMap.2 ⟨⟨b, hb⟩, List.mem_finRange _,
    List.mem_map.2 ⟨0, List.mem_finRange _, ?_⟩⟩)
  funext ω
  simp only [f1Of, dif_pos (show b < k ∧ 0 < 1 from ⟨hb, by norm_num⟩)]
  push_cast
  rfl

/-- **A1 implies MinRank ≥ 2.** -/
theorem a1MinRank : A1MinRank := by
  intro k R c γ₁ hc hA1 q x y hxy
  by_contra hne
  simp only [not_or, not_forall, exists_prop] at hne
  obtain ⟨⟨a0, ha0, hx0⟩, ⟨b0, hb0, hy0⟩⟩ := hne
  set T := 3 * k / 16 - 1 with hT
  let φ : Noise 1 1 1 k → ℚ := ∑ t ∈ range T, q t • word k 1 1 1 R (fun _ _ => 0) (fun _ => 0) 0 t
  let ℓx : Noise 1 1 1 k → ℚ := fun ω => ∑ a ∈ range k, x a * (e1Of ω 0 0 a : ℚ)
  let ℓy : Noise 1 1 1 k → ℚ := fun ω => ∑ b ∈ range k, y b * (f1Of ω b 0 : ℚ)
  have hℓx : ℓx ∈ V0 1 1 1 k := by
    have : ℓx = ∑ a ∈ range k, x a • fun ω : Noise 1 1 1 k => (e1Of ω 0 0 a : ℚ) := by
      funext ω; simp [ℓx, Finset.sum_apply]
    rw [this]
    exact Submodule.mem_sup_left
      (Submodule.sum_mem _ fun a ha => Submodule.smul_mem _ _ (e1_mem_coords (mem_range.1 ha)))
  have hℓy : ℓy ∈ V0 1 1 1 k := by
    have : ℓy = ∑ b ∈ range k, y b • fun ω : Noise 1 1 1 k => (f1Of ω b 0 : ℚ) := by
      funext ω; simp [ℓy, Finset.sum_apply]
    rw [this]
    exact Submodule.mem_sup_left
      (Submodule.sum_mem _ fun b hb => Submodule.smul_mem _ _ (f1_mem_coords (mem_range.1 hb)))
  have hφ : φ = ℓx * ℓy := by
    funext ω
    have hE : ∀ i l, k ≤ l → e1Of ω 0 i l = 0 := fun i l hl => by
      simp only [e1Of]; rw [dif_neg]; omega
    have hF : ∀ l j, k ≤ l → f1Of ω l j = 0 := fun l j hl => by
      simp only [f1Of]; rw [dif_neg]; omega
    simp only [φ, ℓx, ℓy, Finset.sum_apply, Pi.smul_apply, smul_eq_mul, word, checked, Pi.mul_apply,
      Nat.mul_one, ck_expand k _ _ R hE hF]
    simp only [Nat.mod_one, Nat.div_one]
    push_cast
    rw [sum_mul_sum]
    simp only [mul_sum]
    rw [sum_comm]
    refine sum_congr rfl fun a ha => ?_
    rw [sum_comm]
    refine sum_congr rfl fun b hb => ?_
    have := hxy a (mem_range.1 ha) b (mem_range.1 hb)
    calc ∑ t ∈ range T, q t * ((e1Of ω 0 0 a : ℚ) * (f1Of ω b 0 : ℚ) * (ckMat k R (16 * (t + 1)) a b : ℚ))
        = (e1Of ω 0 0 a : ℚ) * (f1Of ω b 0 : ℚ) *
            ∑ t ∈ range T, q t * (ckMat k R (16 * (t + 1)) a b : ℚ) := by
          rw [mul_sum]; exact sum_congr rfl fun t _ => by ring
      _ = x a * (e1Of ω 0 0 a : ℚ) * (y b * (f1Of ω b 0 : ℚ)) := by rw [this]; ring
  have hφT : φ ∈ targetSpan k 1 1 1 R (fun _ _ => 0) (fun _ => 0) {0} := by
    refine Submodule.sum_mem _ fun t ht => Submodule.smul_mem _ _ (Submodule.subset_span ?_)
    have hmem : ((0 : ℕ), t) ∈ ({0} : Finset ℕ) ×ˢ range (nonFinal k 1 1) := by
      simp only [mem_product, mem_singleton, mem_range, nonFinal, mul_one, true_and]
      exact mem_range.1 ht
    exact ⟨⟨(0, t), hmem⟩, rfl⟩
  have hφ0 : φ ≠ 0 := by
    intro h0
    have := congrFun h0 (pt 1 1 1 k true true 0 0 a0 b0 0)
    rw [hφ] at this
    simp only [Pi.mul_apply, Pi.zero_apply, ℓx, ℓy, e1Of_pt true true b0 0 (show 0 < 1 by norm_num)
      (show 0 < 1 by norm_num) ha0, f1Of_pt true true 0 0 a0 hb0 (show 0 < 1 by norm_num), if_true, unitAt]
    at this
    simp only [true_and, and_true, Int.cast_ite, Int.cast_one, Int.cast_zero, mul_ite, mul_one, mul_zero,
      sum_ite_eq', mem_range, ha0, hb0, if_true] at this
    exact mul_ne_zero hx0 hy0 this
  have hV := hA1 1 1 1 (fun _ _ => 0) (fun _ => 0) {0} (by simp) (Submodule.span ℚ {φ})
    ((Submodule.span_singleton_le_iff_mem _ _).2 hφT) 1 (fun _ => ℓx * ℓy)
    (fun _ => ⟨ℓx, hℓx, ℓy, hℓy, rfl⟩)
    ((Submodule.span_singleton_le_iff_mem _ _).2 (Submodule.mem_sup_left (Submodule.subset_span ⟨0, hφ.symm⟩)))
  rw [finrank_span_singleton hφ0] at hV
  push_cast at hV
  linarith

end Pouw.Dimension
