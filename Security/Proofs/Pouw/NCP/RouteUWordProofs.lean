import Models.Pouw.Assumptions.NCP.RouteUWord
import Proofs.Pouw.NCP.ChainProofs
import Proofs.Pouw.NCP.RouteUProofs
import Models.Pouw.Guarantees.NCP

/-!
# The word layer on route U: proofs

* `checkedUPos_sub`: route U's value minus the unbiased one is `shiftU`, a function of `F₁` alone.
* `reductionOnlyIfU`: at the four noise points `RelationProofs.pt` the shift is `shiftU` at `F₁ = 0` or at
  `F₁ = unitAt b j`, the same for both values of `E₁`, so the mixed difference cancels it (`mixdiff_U`) and the
  unbiased argument goes through.
* `ncpFinalOnlyIndU`, `ncpWordFloorIndU`: `RelationProofs.not_identity`, `final_of_identity`, `ncpFinalOnly` and
  `ncpWordFloor` with `reductionOnlyIfU` in place of `reductionOnlyIf`.
* `int7Int32UPos`, `wrapFinalOnlyIndU`, `wrapWordFloorIndU`: `ChainProofs`' wrap argument on route U.
-/

namespace Pouw.NCP.RouteUWordProofs

open Finset Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.NCP.RouteU

/-- Route U's shift at position `p`, a function of `F₁` (for the fixed `B u` and `R`). -/
def shiftU (β : ℤ) (k m n : ℕ) (R : Mat) (B : ℕ → Mat) (F : Mat) (u p : ℕ) : ℤ :=
  β * (∑ l ∈ range (16 * (p / (m * n) + 1)), stackY k (B u) R F l (p % (m * n) % n) -
    ∑ l ∈ range (3 * k), stackY k (B u) R F l (p % (m * n) % n))

theorem checkedUPos_sub (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) :
    checkedUPos β k m n N R A B ω u p - checked k m n N R A B ω u p = shiftU β k m n R B (f1Of ω) u p := by
  rw [checkedUPos, checkedU, NCP.RouteUProofs.runU_eq, shiftU, checked, ck]
  simp only [mm]
  ring

/-- Route U's values at the four noise points: the unbiased ones plus the shift at `F₁ ∈ {0, unitAt b j}`. -/
theorem checkedUPos_pt (β : ℤ) {k m n N : ℕ} (R : Mat) (A B : ℕ → Mat) (bE bF : Bool) {u i a b j : ℕ}
    (hu : u < N) (hi : i < m) (ha : a < k) (hb : b < k) (hj : j < n) (q : ℕ) :
    checkedUPos β k m n N R A B (pt N m n k bE bF u i a b j) u q =
      ck k (A u) (B u) (if bE then unitAt i a else 0) (if bF then unitAt b j else 0) R
        (q % (m * n) / n) (q % (m * n) % n) (16 * (q / (m * n) + 1)) +
      shiftU β k m n R B (if bF then unitAt b j else 0) u q := by
  have h := checkedUPos_sub β k m n N R A B (pt N m n k bE bF u i a b j) u q
  have h2 := congrFun (checked_pt R A B bE bF hu hi ha hb hj) q
  rw [f1Of_pt bE bF u i a hb hj] at h
  linarith

/-- **The reduction on route U.** -/
theorem reductionOnlyIfU : ReductionOnlyIfU := by
  intro β k m n N R A B u p r hu hid e he a ha b hb
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with h | h
    · simp [h] at he
    · exact h
  have hi : e / n < m := (Nat.div_lt_iff_lt_mul hn).2 he
  have hj : e % n < n := Nat.mod_lt e hn
  set X : Bool → Bool → ℕ → ℤ :=
    fun bE bF => checkedUPos β k m n N R A B (pt N m n k bE bF u (e / n) a b (e % n)) u with hX
  have hx : ∀ q, X true true q - X true false q - X false true q + X false false q = coefAt k m n R q e a b := by
    intro q
    simp only [hX, checkedUPos_pt β R A B _ _ hu hi ha hb hj, ite_true, Bool.false_eq_true, ite_false]
    rw [← coef_eq R e a b q, ← mixdiff k (A u) (B u)]
    ring
  have key := target_affine r (X true true) (X true false) (X false true) (X false false)
  rw [show (fun q => X true true q - X true false q - X false true q + X false false q) =
    fun q => coefAt k m n R q e a b from funext hx] at key
  have h1 : X true true p = r.target (X true true) := hid _
  have h2 : X true false p = r.target (X true false) := hid _
  have h3 : X false true p = r.target (X false true) := hid _
  have h4 : X false false p = r.target (X false false) := hid _
  rw [← hx p, h1, h2, h3, h4, key]

/-- No non-final position of a relation-free instance is an identity of route U's values. -/
theorem not_identityU (β : ℤ) {k m n N : ℕ} {R : Mat} {A B : ℕ → Mat} {adv : Word.Adversary}
    (hRF : RelationFree k 16 R) (hV : Word.Valid (fun _ => positions k m n) adv) {u p : ℕ} (hu : u < N)
    (hp : p < positions k m n) (hnf : 16 * (p / (m * n) + 1) < 3 * k) :
    ¬ Word.Identity (checkedUPos β k m n N R A B) adv u p := by
  rintro ⟨hne, hid⟩
  have hred := reductionOnlyIfU β k m n N R A B u p (adv.rule u p) hu hid
  have hmn : 0 < m * n := by
    rcases Nat.eq_zero_or_pos (m * n) with h | h
    · simp [positions, h] at hp
    · exact h
  have he : p % (m * n) < m * n := Nat.mod_lt p hmn
  have hRd := hV u p hp
  have hq : ∀ (x : ℕ) (s : Bool), x < p →
      (if x % (m * n) = p % (m * n) then Word.sgn s else 0) ≠ 0 → x / (m * n) < p / (m * n) := by
    intro x s hx h
    split_ifs at h with hxe
    · exact div_lt_of_mod_eq hx hxe
    · exact absurd rfl h
  generalize hr : adv.rule u p = r at hne hred hRd
  cases r with
  | fresh => exact hne rfl
  | copy s x =>
    refine hRF (p / (m * n)) hnf _ 0 (x / (m * n)) 0 (sgn_opt _ s) (Or.inl rfl) (hq x s hRd)
      (fun h => absurd rfl h) fun a ha b hb => ?_
    have := hred (p % (m * n)) he a ha b hb
    simp only [coefAt, Word.Rule.target, ite_true, mul_zero, sub_zero] at this
    rw [this]
    split_ifs <;> ring
  | sum s x t y =>
    refine hRF (p / (m * n)) hnf _ _ (x / (m * n)) (y / (m * n)) (sgn_opt _ s) (sgn_opt _ t)
      (hq x s hRd.1) (hq y t hRd.2) fun a ha b hb => ?_
    have := hred (p % (m * n)) he a ha b hb
    simp only [coefAt, Word.Rule.target, ite_true, mul_zero, add_zero, sub_zero] at this
    rw [this]
    split_ifs <;> ring
  | const c =>
    refine hRF (p / (m * n)) hnf 0 0 0 0 (Or.inl rfl) (Or.inl rfl) (fun h => absurd rfl h)
      (fun h => absurd rfl h) fun a ha b hb => ?_
    have := hred (p % (m * n)) he a ha b hb
    simp only [coefAt, Word.Rule.target, ite_true, sub_self] at this
    rw [this]
    ring

/-- An identity position of route U's values, for a relation-free instance, is in the final checkpoint. -/
theorem final_of_identityU (β : ℤ) {k m n N : ℕ} {R : Mat} {A B : ℕ → Mat} {adv : Word.Adversary} (h16 : 16 ∣ k)
    (hRF : RelationFree k 16 R) (hV : Word.Valid (fun _ => positions k m n) adv) {u p : ℕ} (hu : u < N)
    (hp : p < positions k m n) (hid : Word.Identity (checkedUPos β k m n N R A B) adv u p) :
    p / (m * n) = 3 * k / 16 - 1 := by
  have hmn : 0 < m * n := by
    rcases Nat.eq_zero_or_pos (m * n) with h | h
    · simp [positions, h] at hp
    · exact h
  have hlt : p / (m * n) < 3 * k / 16 := (Nat.div_lt_iff_lt_mul hmn).2 hp
  have hf : ¬ 16 * (p / (m * n) + 1) < 3 * k := fun h => not_identityU β hRF hV hu hp h hid
  obtain ⟨c, rfl⟩ := h16
  generalize p / (m * n) = q at hlt hf ⊢
  omega

/-- **FinalOnly for route U under independence.** -/
theorem ncpFinalOnlyIndU : NCPFinalOnlyIndU := by
  intro β k m n N R A B adv h16 hI hV u hu
  have hRF := RelationWitness.independentRelationFree k R hI
  classical
  unfold Word.identCount
  refine (card_le_card_of_injOn (· % (m * n)) (t := range (m * n)) ?_ ?_).trans (by simp)
  · intro p hp
    have hp := mem_filter.1 (mem_coe.1 hp)
    rw [mem_range] at hp
    rw [mem_coe, mem_range]
    have hmn : 0 < m * n := by
      rcases Nat.eq_zero_or_pos (m * n) with h | h
      · simp [positions, h] at hp
      · exact h
    exact Nat.mod_lt p hmn
  · intro p hp p' hp' h
    have hp := mem_filter.1 (mem_coe.1 hp)
    have hp' := mem_filter.1 (mem_coe.1 hp')
    rw [mem_range] at hp hp'
    have e1 := final_of_identityU β h16 hRF hV hu hp.1 hp.2
    have e2 := final_of_identityU β h16 hRF hV hu hp'.1 hp'.2
    simp only at h
    rw [← Nat.div_add_mod p (m * n), ← Nat.div_add_mod p' (m * n), e1, e2, h]

/-- `wordBound`'s sum at `identCount ≤ m·n` per unit. -/
theorem floor_sum {Ω : Type} {k m n N : ℕ} (X : Ω → ℕ → ℕ → ℤ) (adv : Word.Adversary) (γ₀ : ℝ)
    (hF : ∀ u < N, Word.identCount (fun _ => positions k m n) X adv u ≤ m * n) :
    ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * ((fun _ => positions k m n) u : ℕ) -
        Word.identCount (fun _ => positions k m n) X adv u)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1))) := by
  calc ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * ((fun _ => positions k m n) u : ℕ) -
        Word.identCount (fun _ => positions k m n) X adv u))
      ≤ ∑ _u ∈ range N, (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1))) := sum_le_sum fun u hu => by
        apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
        have h1 : (Word.identCount (fun _ => positions k m n) X adv u : ℝ) ≤ (m * n : ℕ) := by
          exact_mod_cast hF u (mem_range.1 hu)
        generalize Word.identCount (fun _ => positions k m n) X adv u = ic at h1 ⊢
        simp only [positions]
        push_cast at h1 ⊢
        nlinarith
    _ = N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1))) := by
        rw [sum_const, card_range, nsmul_eq_mul]

/-- **The floor for route U under independence.** -/
theorem ncpWordFloorIndU : Pouw.Guarantees.NCP.NCPWordFloorIndU := by
  intro β k m n N R A B adv γ₀ h16 hI hγ0 hγ1 hV hNz
  exact (WordProofs.wordBound (Noise N m n k) (fun _ => positions k m n) (checkedUPos β k m n N R A B) N adv γ₀
    hγ0 hγ1 hV hNz).trans
    (floor_sum _ adv γ₀ fun u hu => ncpFinalOnlyIndU β k m n N R A B adv h16 hI hV u hu)

/-- **The int32 range of route U's checked values.** -/
theorem int7Int32UPos : Pouw.Guarantees.NCP.Int7Int32UPos := by
  intro β k m n N R A B ω u p hβ1 hβ2 hR hA hB hp
  simp only [checkedUPos, checkedU]
  have hE : ∀ i, ∀ l < k, InNoise6 (e1Of ω u i l) := fun i l _ => ChainProofs.e1Of_noise ω u i l
  have hF : ∀ l < k, ∀ j, InNoise6 (f1Of ω l j) := fun l _ j => ChainProofs.f1Of_noise ω l j
  rcases hR with ⟨hR, hk⟩ | ⟨hR, hk⟩
  · exact NCP.RouteUProofs.int7Int32U β k _ _ _ _ R _ _ _ hβ1 hβ2 hR hE hF hA hB hk (ChainProofs.depth_le hp)
  · exact NCP.RouteUProofs.int7Int32UUnsigned β k _ _ _ _ R _ _ _ hβ1 hβ2 hR hE hF hA hB hk
      (ChainProofs.depth_le hp)

theorem identCount_wrappedU (β : ℤ) {k m n N : ℕ} {R : Mat} {A B : ℕ → Mat} {adv : Word.Adversary}
    (hV : Word.Valid (fun _ => positions k m n) adv)
    (hr : ∀ (ω : Noise N m n k) (u p : ℕ), u < N → p < positions k m n →
      InInt32 (checkedUPos β k m n N R A B ω u p))
    {u : ℕ} (hu : u < N) :
    Word.identCount (fun _ => positions k m n) (wrappedU β k m n N R A B) adv u =
      Word.identCount (fun _ => positions k m n) (checkedUPos β k m n N R A B) adv u := by
  classical
  unfold Word.identCount
  refine congrArg card (filter_congr fun p hp => ?_)
  have hp := mem_range.1 hp
  have heq : ∀ ω q, q ≤ p → wrappedU β k m n N R A B ω u q = checkedUPos β k m n N R A B ω u q :=
    fun ω q hq => Pouw.Proofs.wrap32_of_mem (hr ω u q hu (lt_of_le_of_lt hq hp))
  have ht : ∀ ω, (adv.rule u p).target (wrappedU β k m n N R A B ω u) =
      (adv.rule u p).target (checkedUPos β k m n N R A B ω u) :=
    fun ω => WordProofs.target_congr _ p (hV u p hp) _ _ fun q hq => heq ω q hq.le
  constructor
  · rintro ⟨hne, h⟩
    exact ⟨hne, fun ω => by rw [← heq ω p le_rfl, ← ht ω]; exact h ω⟩
  · rintro ⟨hne, h⟩
    exact ⟨hne, fun ω => by rw [heq ω p le_rfl, ht ω]; exact h ω⟩

/-- **FinalOnly for route U's protocol checked values.** -/
theorem wrapFinalOnlyIndU : Pouw.Guarantees.NCP.WrapFinalOnlyIndU := by
  intro β k m n N R A B adv h16 hI hV hr u hu
  rw [identCount_wrappedU β hV hr hu]
  exact ncpFinalOnlyIndU β k m n N R A B adv h16 hI hV u hu

/-- **The floor for route U's protocol checked values.** -/
theorem wrapWordFloorIndU : Pouw.Guarantees.NCP.WrapWordFloorIndU := by
  intro β k m n N R A B adv γ₀ h16 hI hγ0 hγ1 hV hr hNz
  exact (WordProofs.wordBound (Noise N m n k) (fun _ => positions k m n) (wrappedU β k m n N R A B) N adv γ₀
    hγ0 hγ1 hV hNz).trans
    (floor_sum _ adv γ₀ fun u hu => wrapFinalOnlyIndU β k m n N R A B adv h16 hI hV hr u hu)

end Pouw.NCP.RouteUWordProofs
