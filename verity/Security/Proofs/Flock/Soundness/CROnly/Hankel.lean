import Proofs.Flock.Soundness.CROnly.Hiding

/-!
# hm96's Hankel family is universal

For a nonzero `d`, the map `κ ↦ M(κ)·d` is onto (`hankel_surjective`): on the keys supported on `j₀ … j₀ + n − 1`, for
`j₀` the first index where `d` is nonzero, it is injective, since the row of the last nonzero symbol reads it times
`d_{j₀}` alone, and so onto. It is additive in the key, so each of its values is taken by as many keys as its kernel
holds, and two distinct salts `y ≠ y'` collide (`M·(y − y') = 0`) under a `1/|B|` fraction of the keys
(`hankel_universal`).
-/

namespace FlockSoundness.CROnly

open Finset

variable {F : Type} [Field F] {n l : ℕ}

theorem hankel_add_key (κ κ' : Fin (n + l - 1) → F) (y : Fin l → F) :
    hankel (κ + κ') y = hankel κ y + hankel κ' y := by
  funext i
  simp only [hankel, Pi.add_apply, add_mul, sum_add_distrib]

theorem hankel_sub_key (κ κ' : Fin (n + l - 1) → F) (y : Fin l → F) :
    hankel (κ - κ') y = hankel κ y - hankel κ' y := by
  funext i
  simp only [hankel, Pi.sub_apply, sub_mul, sum_sub_distrib]

theorem hankel_sub_salt (κ : Fin (n + l - 1) → F) (y y' : Fin l → F) :
    hankel κ (y - y') = hankel κ y - hankel κ y' := by
  funext i
  simp only [hankel, Pi.sub_apply, mul_sub, sum_sub_distrib]

/-- The key that holds `v` at `j₀ … j₀ + n − 1` and 0 elsewhere. -/
def shiftKey (j₀ : Fin l) (v : Fin n → F) (t : Fin (n + l - 1)) : F :=
  if h : j₀.val ≤ t.val ∧ t.val - j₀.val < n then v ⟨t.val - j₀.val, h.2⟩ else 0

theorem shiftKey_sub (j₀ : Fin l) (v v' : Fin n → F) :
    shiftKey j₀ (v - v') = shiftKey j₀ v - shiftKey j₀ v' := by
  funext t
  simp only [shiftKey, Pi.sub_apply]
  split_ifs <;> simp

theorem shiftKey_at (j₀ : Fin l) (v : Fin n → F) (a : Fin n) (h : a.val + j₀.val < n + l - 1) :
    shiftKey j₀ v ⟨a.val + j₀.val, h⟩ = v a := by
  have h1 : j₀.val ≤ a.val + j₀.val ∧ a.val + j₀.val - j₀.val < n :=
    ⟨Nat.le_add_left _ _, by rw [Nat.add_sub_cancel]; exact a.isLt⟩
  unfold shiftKey
  split_ifs with hc
  · exact congrArg v (Fin.ext (by simp))
  · exact absurd h1 hc

variable [Fintype F] [DecidableEq F]

omit [Fintype F] in
/-- On the keys `shiftKey j₀ v`, `M·d` vanishes only at `v = 0`, for `j₀` the first index where `d` is nonzero. -/
theorem hankel_shiftKey_eq_zero {d : Fin l → F} {j₀ : Fin l} (hj₀ : d j₀ ≠ 0) (hmin : ∀ j, j < j₀ → d j = 0)
    {v : Fin n → F} (hv : hankel (shiftKey j₀ v) d = 0) : v = 0 := by
  classical
  by_contra hne
  have hS : (univ.filter fun a => v a ≠ 0).Nonempty := by
    obtain ⟨a, ha⟩ := Function.ne_iff.1 hne
    exact ⟨a, mem_filter.2 ⟨mem_univ a, ha⟩⟩
  obtain ⟨a, hva, hmax⟩ : ∃ a, v a ≠ 0 ∧ ∀ b, a < b → v b = 0 :=
    ⟨_, (mem_filter.1 (max'_mem _ hS)).2, fun b hb => by_contra fun h =>
      (not_le.2 hb) (le_max' _ b (mem_filter.2 ⟨mem_univ b, h⟩))⟩
  have h0 := congrFun hv a
  simp only [hankel, Pi.zero_apply] at h0
  rw [sum_eq_single j₀] at h0
  · rw [shiftKey_at] at h0
    exact mul_ne_zero hva hj₀ h0
  · intro j _ hj
    rcases lt_or_gt_of_ne hj with hlt | hgt
    · rw [hmin j hlt, mul_zero]
    · have hgt' : j₀.val < j.val := hgt
      unfold shiftKey
      split_ifs with hc
      · rw [hmax, zero_mul]
        show a.val < a.val + j.val - j₀.val
        omega
      · exact zero_mul _
  · intro h
    exact absurd (mem_univ j₀) h

/-- **`κ ↦ M(κ)·d` is onto** for every nonzero `d`. -/
theorem hankel_surjective {d : Fin l → F} (hd : d ≠ 0) :
    Function.Surjective fun κ : Fin (n + l - 1) → F => hankel κ d := by
  classical
  have hS : (univ.filter fun j => d j ≠ 0).Nonempty := by
    obtain ⟨j, hj⟩ := Function.ne_iff.1 hd
    exact ⟨j, mem_filter.2 ⟨mem_univ j, hj⟩⟩
  obtain ⟨j₀, hj₀, hmin⟩ : ∃ j₀, d j₀ ≠ 0 ∧ ∀ j, j < j₀ → d j = 0 :=
    ⟨_, (mem_filter.1 (min'_mem _ hS)).2, fun j hj => by_contra fun h =>
      (not_le.2 hj) (min'_le _ j (mem_filter.2 ⟨mem_univ j, h⟩))⟩
  have hinj : Function.Injective fun v : Fin n → F => hankel (shiftKey j₀ v) d := by
    intro v v' h
    have h2 : hankel (shiftKey j₀ v) d = hankel (shiftKey j₀ v') d := h
    have h' : hankel (shiftKey j₀ (v - v')) d = 0 := by
      rw [shiftKey_sub, hankel_sub_key, h2, sub_self]
    exact sub_eq_zero.1 (hankel_shiftKey_eq_zero hj₀ hmin h')
  intro u
  obtain ⟨v, hv⟩ := Finite.injective_iff_surjective.1 hinj u
  exact ⟨shiftKey j₀ v, hv⟩

/-- **hm96's Hankel family is universal** (`verity/commitments/hm96/PROTOCOL.md` §2): for `y ≠ y'`,
`M(κ)·y = M(κ)·y'` for at most a `1/|F|^n` fraction of the keys `κ`. -/
theorem hankel_universal : Universal (hankel : (Fin (n + l - 1) → F) → (Fin l → F) → Fin n → F) := by
  classical
  intro y y' hyy
  have hd : y - y' ≠ 0 := sub_ne_zero.2 hyy
  obtain ⟨s, hs⟩ : ∃ s : (Fin n → F) → Fin (n + l - 1) → F, ∀ u, hankel (s u) (y - y') = u :=
    ⟨Function.surjInv (hankel_surjective hd), Function.surjInv_eq (hankel_surjective hd)⟩
  have hcard : #{κ : Fin (n + l - 1) → F | (hankel κ y : Fin n → F) = hankel κ y'} = Fintype.card {κ : Fin (n + l - 1) → F // hankel κ (y - y') = 0} := by
    rw [Fintype.card_subtype]
    congr 1
    ext κ
    simp only [mem_filter, mem_univ, true_and, hankel_sub_salt, sub_eq_zero]
  rw [hcard, mul_comm, ← Fintype.card_prod]
  refine Fintype.card_le_of_injective (fun p : (Fin n → F) × {κ : Fin (n + l - 1) → F // hankel κ (y - y') = 0} => p.2.1 + s p.1) ?_
  rintro ⟨u, κ, hκ⟩ ⟨u', κ', hκ'⟩ h
  have h1 : κ + s u = κ' + s u' := h
  have hu : u = u' := by
    have h2 := congrArg (fun κ => hankel κ (y - y')) h1
    simp only [hankel_add_key, hκ, hκ', hs, zero_add] at h2
    exact h2
  subst hu
  have : κ = κ' := add_right_cancel h1
  subst this
  rfl

end FlockSoundness.CROnly
