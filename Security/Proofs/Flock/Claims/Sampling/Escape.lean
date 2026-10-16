import Definitions.Claims.Sampling
import Proofs.Flock.Claims.Sampling.Draws

/-!
# The claims block's escape and miss, as the draws' probabilities

`Definitions.Claims.Sampling` states the draw's term in closed form: `Law.escape N L B`, the formula for the
probability that `L`'s draw over regions `0, …, N − 1` misses every region of `B`, and `Law.miss L N t`, the largest
escape of `t` or more regions. Here each well-formed law gets its model draw over `Fin N` (`model`), the audit layer's
law of the same shape, and the formula is that draw's `Claims.Pr`:

* `finOf N B`: the regions of `B`, as units of a draw over `Fin N`.
* `strataOf`, `strataOf_eq_iff`: a stratified law's strata, as the assignment `σ` of the audit layer's
  `Law.stratified`, region `u` in stratum `s` exactly when the registration lists it there.
* `model`, `escape_eq_Pr`: **the escape is the model draw's miss probability**, for every well-formed law and every set
  of regions (`subset_miss_eq`, `bernoulli_miss_eq`, `stratified_miss_eq` of `Draws.lean`).
* `escape_le_miss`, `miss_le_one`, `miss_nonneg`, `Pr_miss_le_miss`: the draw misses `t ≤ |B|` wrong regions with
  probability at most `miss(t) ∈ [0, 1]`.
* `miss_subset_le`, `miss_bernoulli_le`, `miss_stratified_le`: `miss(t)` is at most `C(N − t, k)/C(N, k)`,
  `((den − num)/den)^t`, and `(1 − π)^t` when every stratum draws at least a fraction `π` of its regions.

That the deployed verifier's draw has its law's distribution is not stated by the block: `Live.lean` proves it of the
executable draw on uniform bytes, and binding the sampled verifier's draw to it is the code side's.
-/

namespace Claims.Sampling

open Finset

/-- The regions of `B`, as units of a draw over `Fin N`. -/
def finOf (N : ℕ) (B : Finset ℕ) : Finset (Fin N) := univ.filter fun u => u.val ∈ B

theorem map_finOf {N : ℕ} {B : Finset ℕ} (hB : B ⊆ range N) : (finOf N B).map Fin.valEmbedding = B := by
  ext a
  simp only [finOf, mem_map, mem_filter, mem_univ, true_and, Fin.valEmbedding_apply]
  constructor
  · rintro ⟨u, hu, rfl⟩
    exact hu
  · intro ha
    exact ⟨⟨a, mem_range.1 (hB ha)⟩, ha, rfl⟩

theorem card_finOf {N : ℕ} {B : Finset ℕ} (hB : B ⊆ range N) : (finOf N B).card = B.card := by
  rw [← card_map Fin.valEmbedding, map_finOf hB]

/-! ## The stratified law's strata -/

section Strata

variable {N : ℕ} {ss : List (List ℕ × ℕ)}

theorem mem_strata (hp : (ss.flatMap Prod.fst).Perm (List.range N)) (u : Fin N) : ∃ p ∈ ss, u.val ∈ p.1 :=
  List.mem_flatMap.1 (hp.mem_iff.2 (List.mem_range.2 u.2))

theorem strata_lt (hp : (ss.flatMap Prod.fst).Perm (List.range N)) {p : List ℕ × ℕ} (h : p ∈ ss) {a : ℕ}
    (ha : a ∈ p.1) : a < N :=
  List.mem_range.1 (hp.mem_iff.1 (List.mem_flatMap.2 ⟨p, h, ha⟩))

theorem strata_nodup (hp : (ss.flatMap Prod.fst).Perm (List.range N)) :
    (∀ p ∈ ss, p.1.Nodup) ∧ ss.Pairwise (Function.onFun List.Disjoint Prod.fst) :=
  List.nodup_flatMap.1 (hp.nodup_iff.2 List.nodup_range)

/-- The stratum region `u` is registered in: the first whose regions include it. -/
def strataOf (hp : (ss.flatMap Prod.fst).Perm (List.range N)) (u : Fin N) : Fin ss.length :=
  ⟨ss.findIdx fun p => decide (u.val ∈ p.1), List.findIdx_lt_length_of_exists (by
    obtain ⟨p, h, hu⟩ := mem_strata hp u
    exact ⟨p, h, decide_eq_true hu⟩)⟩

/-- **Region `u` is in stratum `s` exactly when the registration lists it there**: the strata partition the regions. -/
theorem strataOf_eq_iff (hp : (ss.flatMap Prod.fst).Perm (List.range N)) (u : Fin N) (s : Fin ss.length) :
    strataOf hp u = s ↔ u.val ∈ ss[s].1 := by
  have hlt : ss.findIdx (fun p => decide (u.val ∈ p.1)) < ss.length := (strataOf hp u).2
  have hfound : u.val ∈ (ss[ss.findIdx fun p => decide (u.val ∈ p.1)]'hlt).1 :=
    of_decide_eq_true (List.findIdx_getElem (w := hlt))
  constructor
  · rintro rfl
    exact hfound
  · intro hu
    apply Fin.ext
    show ss.findIdx (fun p => decide (u.val ∈ p.1)) = s.val
    rcases lt_trichotomy (ss.findIdx fun p : List ℕ × ℕ => decide (u.val ∈ p.1)) s.val with h | h | h
    · exact absurd hu (List.pairwise_iff_getElem.1 (strata_nodup hp).2 _ _ hlt s.2 h hfound)
    · exact h
    · have := List.not_of_lt_findIdx h
      simp only [decide_eq_false_iff_not] at this
      exact absurd hu this

theorem stratum_strataOf (hp : (ss.flatMap Prod.fst).Perm (List.range N)) (s : Fin ss.length) :
    FlockSoundness.Audit.Law.stratum (strataOf hp) s = finOf N ss[s].1.toFinset := by
  ext u
  simp [FlockSoundness.Audit.Law.mem_stratum, finOf, strataOf_eq_iff]

theorem card_stratum_strataOf (hp : (ss.flatMap Prod.fst).Perm (List.range N)) (s : Fin ss.length) :
    (FlockSoundness.Audit.Law.stratum (strataOf hp) s).card = ss[s].1.length := by
  rw [stratum_strataOf, card_finOf]
  · exact List.toFinset_card_of_nodup ((strata_nodup hp).1 _ (List.getElem_mem _))
  · intro a ha
    exact mem_range.2 (strata_lt hp (List.getElem_mem _) (List.mem_toFinset.1 ha))

theorem card_filter_strataOf (hp : (ss.flatMap Prod.fst).Perm (List.range N)) {B : Finset ℕ}
    (hB : B ⊆ range N) (s : Fin ss.length) :
    ((finOf N B).filter (strataOf hp · = s)).card = (B.filter (· ∈ ss[s].1)).card := by
  have : (finOf N B).filter (strataOf hp · = s) = finOf N (B.filter (· ∈ ss[s].1)) := by
    ext u
    simp [finOf, strataOf_eq_iff]
  rw [this, card_finOf ((filter_subset _ B).trans hB)]

end Strata

/-! ## Each law's model draw -/

/-- A stratified law's model draw: the audit layer's `Law.stratified` at its strata and counts. -/
def stratifiedModel {N : ℕ} {ss : List (List ℕ × ℕ)} (hw : (Law.stratified ss).WellFormed N) :
    FlockSoundness.Audit.Law N :=
  FlockSoundness.Audit.Law.stratified (strataOf hw.1) (fun s => ss[s].2) fun s => by
    rw [card_stratum_strataOf hw.1 s]
    exact (hw.2 _ (List.getElem_mem _)).2

/-- **A well-formed law's model draw** over `Fin N`: a uniform `k`-subset, each region on its own coin in `[0, den)`
below `num`, or independent uniform `k_s`-subsets of the strata. -/
def model {N : ℕ} : (L : Law) → L.WellFormed N → FlockSoundness.Audit.Law N
  | .subset k, hw => FlockSoundness.Audit.Law.subset N k hw.2
  | .bernoulli num den, hw => @FlockSoundness.Audit.Law.bernoulli N num den ⟨hw.1.ne'⟩
  | .stratified _, hw => stratifiedModel hw

theorem escape_stratified_eq {N : ℕ} {ss : List (List ℕ × ℕ)} (hw : (Law.stratified ss).WellFormed N)
    {B : Finset ℕ} (hB : B ⊆ range N) :
    Law.escape N (.stratified ss) B =
      Claims.Pr (fun ω : (stratifiedModel hw).Ω => Disjoint ((stratifiedModel hw).draw ω) (finOf N B)) := by
  unfold stratifiedModel
  rw [stratified_miss_eq, Law.escape, ← List.ofFn_getElem_eq_map, List.prod_ofFn]
  refine prod_congr rfl fun s _ => ?_
  rw [card_stratum_strataOf, card_filter_strataOf hw.1 hB]
  rfl

/-- **The escape is the model draw's miss probability**: `Law.escape N L B` is the probability that `L`'s draw over
`Fin N` misses every region of `B`. -/
theorem escape_eq_Pr {N : ℕ} (L : Law) (hw : L.WellFormed N) {B : Finset ℕ} (hB : B ⊆ range N) :
    L.escape N B = Claims.Pr (fun ω : (model L hw).Ω => Disjoint ((model L hw).draw ω) (finOf N B)) := by
  match L, hw with
  | .subset k, hw =>
    show _ = Claims.Pr (fun S : {S : Finset (Fin N) // S.card = k} => Disjoint S.1 (finOf N B))
    rw [subset_miss_eq hw.2, card_finOf hB]
    rfl
  | .bernoulli num den, hw =>
    have : NeZero den := ⟨hw.1.ne'⟩
    show _ = Claims.Pr (fun ω : Fin N → Fin den => Disjoint (univ.filter fun u => (ω u).val < num) (finOf N B))
    rw [bernoulli_miss_eq hw.2, card_finOf hB]
    rfl
  | .stratified ss, hw => exact escape_stratified_eq hw hB

theorem escape_nonneg {N : ℕ} (L : Law) (hw : L.WellFormed N) {B : Finset ℕ} (hB : B ⊆ range N) :
    0 ≤ L.escape N B := by
  rw [escape_eq_Pr L hw hB]
  exact Pr_nonneg _

theorem escape_le_one {N : ℕ} (L : Law) (hw : L.WellFormed N) {B : Finset ℕ} (hB : B ⊆ range N) :
    L.escape N B ≤ 1 := by
  rw [escape_eq_Pr L hw hB]
  exact Pr_le_one _

/-! ## The miss at `t` -/

theorem miss_le_one {N : ℕ} (L : Law) (hw : L.WellFormed N) (t : ℕ) : L.miss N t ≤ 1 :=
  Real.iSup_le (fun B => escape_le_one L hw B.2.1) zero_le_one

theorem miss_nonneg {N : ℕ} (L : Law) (hw : L.WellFormed N) (t : ℕ) : 0 ≤ L.miss N t :=
  Real.iSup_nonneg fun B => escape_nonneg L hw B.2.1

theorem escape_le_miss {N t : ℕ} (L : Law) (hw : L.WellFormed N) {B : Finset ℕ} (hB : B ⊆ range N)
    (ht : t ≤ B.card) : L.escape N B ≤ L.miss N t :=
  le_ciSup (f := fun B : {B : Finset ℕ // B ⊆ range N ∧ t ≤ B.card} => L.escape N B.1)
    ⟨1, by rintro _ ⟨B, rfl⟩; exact escape_le_one L hw B.2.1⟩ ⟨B, hB, ht⟩

/-- **The draw misses `t` wrong regions** with probability at most `miss(t)`: the term `κ_t` carries. -/
theorem Pr_miss_le_miss {N t : ℕ} (L : Law) (hw : L.WellFormed N) {B : Finset ℕ} (hB : B ⊆ range N)
    (ht : t ≤ B.card) :
    Claims.Pr (fun ω : (model L hw).Ω => Disjoint ((model L hw).draw ω) (finOf N B)) ≤ L.miss N t := by
  rw [← escape_eq_Pr L hw hB]
  exact escape_le_miss L hw hB ht

/-- **The uniform `k`-subset's miss at `t`** is at most `C(N − t, k)/C(N, k)`. -/
theorem miss_subset_le {N k : ℕ} (t : ℕ) :
    (Law.subset k).miss N t ≤ ((N - t).choose k : ℝ) / (N.choose k : ℝ) :=
  Real.iSup_le (fun B => by
    show ((N - B.1.card).choose k : ℝ) / (N.choose k : ℝ) ≤ _
    gcongr
    exact B.2.2)
    (by positivity)

/-- **The Bernoulli law's miss at `t`** is at most `((den − num)/den)^t`. -/
theorem miss_bernoulli_le {N num den : ℕ} (h : num ≤ den) (t : ℕ) :
    (Law.bernoulli num den).miss N t ≤ (((den - num : ℕ) : ℝ) / den) ^ t := by
  have h0 : (0 : ℝ) ≤ ((den - num : ℕ) : ℝ) / den := by positivity
  have h1 : ((den - num : ℕ) : ℝ) / den ≤ 1 := by
    rcases Nat.eq_zero_or_pos den with rfl | hd
    · simp
    · rw [div_le_one (by exact_mod_cast hd)]
      exact_mod_cast Nat.sub_le den num
  exact Real.iSup_le (fun B => pow_le_pow_of_le_one h0 h1 B.2.2) (pow_nonneg h0 t)

/-- **The stratified law's miss at `t`** is at most `(1 − π)^t` when every stratum draws at least a fraction `π ≤ 1`
of its regions. -/
theorem miss_stratified_le {N : ℕ} {ss : List (List ℕ × ℕ)} (hw : (Law.stratified ss).WellFormed N) {π : ℝ}
    (hπ0 : 0 ≤ π) (hπ1 : π ≤ 1) (hπ : ∀ p ∈ ss, π * p.1.length ≤ p.2) (t : ℕ) :
    (Law.stratified ss).miss N t ≤ (1 - π) ^ t := by
  refine Real.iSup_le (fun B => ?_) (pow_nonneg (by linarith) t)
  rw [escape_stratified_eq hw B.2.1]
  refine stratified_miss_le _ _ _ hπ0 hπ1 (fun s _ => ?_) (by rw [card_finOf B.2.1]; exact B.2.2)
  rw [card_stratum_strataOf]
  exact hπ _ (List.getElem_mem _)

end Claims.Sampling
