import Proofs.Population.Determinacy
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
# Skippable work: what the faulty proof blocks' cones and luck bound

A run commits `σ` while computing only the units `K`; the units it skips are `univ \ K`. A skipped unit can be correct
for three reasons: it is downstream of a faulty proof block (its inputs are garbage the prover chose, which may make it
easy, as `A = −b` zeroes a product), it was copied from a unit computing the same thing on the same inputs, or luck.

* `skipped_sub`, **the population's part**: in a checked population, every skipped unit is downstream of a faulty
  proof block or lucky (`SkipsLucky`: a skipped unit correct on honest inputs is lucky). Units outside the faulty proof
  blocks' cone read exactly their honest inputs (`eq_honest_off_down`), which is where the per-unit premise applies.
* `skipped_work_le`: so the skipped work is at most `|wrong| · F` plus the lucky work, `F` bounding the work of any
  unit's downstream cone, which the verifier computes from the public dataflow.
* `skipped_work_le_of_luck`: if the lucky work is at most a `p = num/den` share of the skipped work plus a slack, the
  skipped work `W` satisfies `(den − num) · W ≤ den · (m F + slack)`, that is `W ≤ (m F + slack) / (1 − p)`:
  compute-accounting's `F m / (1 − p) + slack`.
* `skipped_work_le_wt`, `skipped_work_le_wt_of_luck`: the same with every cone at most `F` times its root's weight, so
  the bound reads the faulty proof blocks' weight, which weighted sampling bounds: `W ≤ (F · wt(B) + slack) / (1 − p)`.
* `twin_copy`, **why luck needs distinct units**: two units computing the same function of the same input: the run
  that computes one and copies it has no faulty proof block, and every hardness premise must count the copy as luck. The
  premise holds only when no two credited units share (function, inputs, coin), which is the compile pass's job
  (a coin per unit), not the population's.
-/

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

/-- **Every skipped unit is downstream of a faulty proof block, or lucky.** -/
theorem skipped_sub [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest σ : S → V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = a s) {K L : Finset U} (hl : P.SkipsLucky f honest σ K L) :
    univ \ K ⊆ P.down (P.wrong f σ) ∪ L := by
  intro u hu
  have huK : u ∉ K := by simpa using hu
  rw [mem_union]
  by_cases hd : u ∈ P.down (P.wrong f σ)
  · exact Or.inl hd
  · right
    have hcor : P.Correct f σ u := P.correct_of_not_mem_wrong fun hw =>
      hd ((P.mem_down).2 ⟨u, hw, Relation.ReflTransGen.refl⟩)
    refine hl u huK (fun r hr => ?_) hcor
    by_cases hri : r ∈ P.input
    · exact (hσ r hri).trans (hh r hri).symm
    · obtain ⟨v, hv⟩ := P.covered_of_singleAssignment h.1 r hri
      refine P.eq_honest_off_down h f a hh hc hσ v (fun hvd => hd ?_) r hv
      obtain ⟨w, hw, hwv⟩ := (P.mem_down).1 hvd
      exact (P.mem_down).2 ⟨w, hw, hwv.tail ⟨r, hv, hr⟩⟩

theorem sum_union_le {α : Type} [DecidableEq α] (s t : Finset α) (g : α → ℕ) :
    ∑ x ∈ s ∪ t, g x ≤ ∑ x ∈ s, g x + ∑ x ∈ t, g x := by
  rw [← sum_union_inter]
  exact Nat.le_add_right _ _

theorem sum_biUnion_le {α ι : Type} [DecidableEq α] [DecidableEq ι] (B : Finset ι) (t : ι → Finset α)
    (g : α → ℕ) : ∑ x ∈ B.biUnion t, g x ≤ ∑ i ∈ B, ∑ x ∈ t i, g x := by
  induction B using Finset.induction_on with
  | empty => simp
  | insert i B hi ih =>
    rw [biUnion_insert, sum_insert hi]
    exact (sum_union_le _ _ g).trans (Nat.add_le_add_left ih _)

theorem down_subset_biUnion [Fintype U] [DecidableEq U] (B : Finset U) :
    P.down B ⊆ B.biUnion fun u => P.down {u} := by
  intro v hv
  obtain ⟨u, hu, huv⟩ := (P.mem_down).1 hv
  exact mem_biUnion.2 ⟨u, hu, (P.mem_down).2 ⟨u, mem_singleton_self u, huv⟩⟩

/-- **The skipped work** is at most the faulty proof blocks' cones and the lucky work. -/
theorem skipped_work_le [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest σ : S → V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = a s) {K L : Finset U} (hl : P.SkipsLucky f honest σ K L) (wt : U → ℕ) (F : ℕ)
    (hF : ∀ u, ∑ v ∈ P.down {u}, wt v ≤ F) :
    ∑ u ∈ univ \ K, wt u ≤ (P.wrong f σ).card * F + ∑ u ∈ L, wt u := by
  calc ∑ u ∈ univ \ K, wt u ≤ ∑ u ∈ P.down (P.wrong f σ) ∪ L, wt u :=
        sum_le_sum_of_subset (P.skipped_sub h f a hh hc hσ hl)
    _ ≤ ∑ u ∈ P.down (P.wrong f σ), wt u + ∑ u ∈ L, wt u := sum_union_le _ _ _
    _ ≤ (P.wrong f σ).card * F + ∑ u ∈ L, wt u := by
        refine Nat.add_le_add_right ?_ _
        calc ∑ u ∈ P.down (P.wrong f σ), wt u
            ≤ ∑ u ∈ (P.wrong f σ).biUnion fun u => P.down {u}, wt u :=
              sum_le_sum_of_subset (P.down_subset_biUnion _)
          _ ≤ ∑ u ∈ P.wrong f σ, ∑ v ∈ P.down {u}, wt v := sum_biUnion_le _ _ _
          _ ≤ ∑ _u ∈ P.wrong f σ, F := sum_le_sum fun u _ => hF u
          _ = (P.wrong f σ).card * F := by simp

/-- **The skipped work under a luck rate** `p = num/den`: `(den − num) · W ≤ den · (m F + slack)`. -/
theorem skipped_work_le_of_luck [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest σ : S → V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = a s) {K L : Finset U} (hl : P.SkipsLucky f honest σ K L) (wt : U → ℕ) (F : ℕ)
    (hF : ∀ u, ∑ v ∈ P.down {u}, wt v ≤ F) (m : ℕ) (hm : (P.wrong f σ).card ≤ m) (num den slack : ℕ)
    (hluck : den * ∑ u ∈ L, wt u ≤ num * ∑ u ∈ univ \ K, wt u + den * slack) :
    (den - num) * ∑ u ∈ univ \ K, wt u ≤ den * (m * F + slack) := by
  have h1 := P.skipped_work_le h f a hh hc hσ hl wt F hF
  have h2 : (P.wrong f σ).card * F ≤ m * F := Nat.mul_le_mul_right _ hm
  have h3 : den * ∑ u ∈ univ \ K, wt u ≤ den * (m * F) + den * ∑ u ∈ L, wt u := by
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (h1.trans (Nat.add_le_add_right h2 _))
  rw [Nat.sub_mul, Nat.mul_add]
  omega

/-- **The skipped work, with every cone at most `F` times its root's weight**: at most `F · wt(wrong)` plus the lucky
work. -/
theorem skipped_work_le_wt [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest σ : S → V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = a s) {K L : Finset U} (hl : P.SkipsLucky f honest σ K L) (wt : U → ℕ) (F : ℕ)
    (hF : ∀ u, ∑ v ∈ P.down {u}, wt v ≤ F * wt u) :
    ∑ u ∈ univ \ K, wt u ≤ F * ∑ u ∈ P.wrong f σ, wt u + ∑ u ∈ L, wt u := by
  calc ∑ u ∈ univ \ K, wt u ≤ ∑ u ∈ P.down (P.wrong f σ) ∪ L, wt u :=
        sum_le_sum_of_subset (P.skipped_sub h f a hh hc hσ hl)
    _ ≤ ∑ u ∈ P.down (P.wrong f σ), wt u + ∑ u ∈ L, wt u := sum_union_le _ _ _
    _ ≤ F * ∑ u ∈ P.wrong f σ, wt u + ∑ u ∈ L, wt u := by
        refine Nat.add_le_add_right ?_ _
        calc ∑ u ∈ P.down (P.wrong f σ), wt u
            ≤ ∑ u ∈ (P.wrong f σ).biUnion fun u => P.down {u}, wt u :=
              sum_le_sum_of_subset (P.down_subset_biUnion _)
          _ ≤ ∑ u ∈ P.wrong f σ, ∑ v ∈ P.down {u}, wt v := sum_biUnion_le _ _ _
          _ ≤ ∑ u ∈ P.wrong f σ, F * wt u := sum_le_sum fun u _ => hF u
          _ = F * ∑ u ∈ P.wrong f σ, wt u := (mul_sum _ _ _).symm

/-- **The skipped work under weighted sampling and a luck rate** `p = num/den`: if sampling bounds the faulty proof
blocks' weight by `M`, then `(den − num) · W ≤ den · (F M + slack)`, that is `W ≤ (F · wt(B) + slack) / (1 − p)`. -/
theorem skipped_work_le_wt_of_luck [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest σ : S → V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = a s) {K L : Finset U} (hl : P.SkipsLucky f honest σ K L) (wt : U → ℕ) (F : ℕ)
    (hF : ∀ u, ∑ v ∈ P.down {u}, wt v ≤ F * wt u) (M : ℕ) (hM : ∑ u ∈ P.wrong f σ, wt u ≤ M)
    (num den slack : ℕ) (hluck : den * ∑ u ∈ L, wt u ≤ num * ∑ u ∈ univ \ K, wt u + den * slack) :
    (den - num) * ∑ u ∈ univ \ K, wt u ≤ den * (F * M + slack) := by
  have h1 := P.skipped_work_le_wt h f a hh hc hσ hl wt F hF
  have h2 : F * ∑ u ∈ P.wrong f σ, wt u ≤ F * M := Nat.mul_le_mul_left _ hM
  have h3 : den * ∑ u ∈ univ \ K, wt u ≤ den * (F * M) + den * ∑ u ∈ L, wt u := by
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (h1.trans (Nat.add_le_add_right h2 _))
  rw [Nat.sub_mul, Nat.mul_add]
  omega

/-! ## Copies -/

/-- Two units reading the one input slot `0` and writing slots `1` and `2`. -/
def twin : Dataflow (Fin 2) (Fin 3) where
  input := {0}
  reads := fun _ => {0}
  writes := fun u => {u.succ}

/-- Both units compute `g` of the input. -/
def twinFns {V : Type} (g : V → V) : twin.Fns V := fun _ r _ => g (r ⟨0, by simp [twin]⟩)

/-- **A copy is correct work that was never done**: the run that computes unit `0` and copies its output to unit `1`
has no faulty proof block, so any hardness premise for it must count unit `1` as lucky. -/
theorem twin_copy {V : Type} (g : V → V) (a : Fin 3 → V) :
    let σ : Fin 3 → V := fun s => if s = 0 then a 0 else g (a 0)
    (∀ u, twin.Correct (twinFns g) σ u) ∧
      ∀ L : Finset (Fin 2), twin.SkipsLucky (twinFns g) σ σ {0} L → 1 ∈ L := by
  intro σ
  have hcor : ∀ u, twin.Correct (twinFns g) σ u := by
    intro u ⟨s, hs⟩
    have hs' : s = u.succ := by simpa [twin] using hs
    subst hs'
    simp [σ, twinFns, Fin.succ_ne_zero]
  exact ⟨hcor, fun L hl => hl 1 (by decide) (fun _ _ => rfl) (hcor 1)⟩

end Population.Dataflow
