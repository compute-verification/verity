import Mathlib.Data.Fintype.BigOperators
import Proofs.Pouw.Dimension.H100WProofs
import Models.Pouw.Assumptions.Fp8Atom.H1TGamma

/-!
# Proofs of H-1T's γ corollary

A credited word of unit `u` reads only coordinate `u` of the joint salt. So placing a salt `s` of unit `u` in an
otherwise fixed joint salt (`place`) recovers unit `u`'s word on `s`, and leaves every other unit's words alone. Two
credited words of different units that are equal are then both constant, and a word equal to a free value is constant
or a raw XOF word of its own unit: `DistinctLiveH1T` excludes all three. The bound is `distinctLiveMH100` at `ε = 0`.
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.H1T Pouw.Dimension.H100

variable {N : ℕ} {Us : Fin N → Unit}

/-- The joint salt with every raw word 0. -/
def zeroSalt (Us : Fin N → Unit) : MSalt Us := fun _ _ _ _ => ⟨0, by norm_num⟩

/-- Unit `u`'s salt `s`, the others' zero. -/
def place (Us : Fin N → Unit) (u : Fin N) (s : (Us u).Salt) : MSalt Us := Function.update (zeroSalt Us) u s

theorem mword_place_self (u : Fin N) (c : Fin (Us u).m × Fin (Us u).n × Pos (Us u).T) (s : (Us u).Salt) :
    mword Us ⟨u, c⟩ (place Us u s) = (Us u).word c s := by
  simp [mword, place, Function.update_self]

theorem mword_place_ne {u u' : Fin N} (h : u' ≠ u) (c : Fin (Us u').m × Fin (Us u').n × Pos (Us u').T)
    (s : (Us u).Salt) : mword Us ⟨u', c⟩ (place Us u s) = mword Us ⟨u', c⟩ (zeroSalt Us) := by
  simp [mword, place, Function.update_of_ne h]

theorem h1tUnitsDistinct (hD : DistinctLiveH1T) : H1TUnitsDistinct := by
  intro N Us hL
  have hconst : ∀ u c (q : ℚ), (Us u).word c ≠ fun _ => q := fun u c q e =>
    (hD (Us u) (hL u)).2 c (e ▸ Or.inl ⟨q, rfl⟩)
  refine ⟨?_, ?_⟩
  · rintro ⟨u, c⟩ ⟨u', c'⟩ e
    by_cases hu : u' = u
    · subst hu
      have : (Us u').word c = (Us u').word c' := by
        funext s
        rw [← mword_place_self, ← mword_place_self, e]
      rw [(hD (Us u') (hL u')).1 this]
    · refine absurd ?_ (hconst u c (mword Us ⟨u', c'⟩ (zeroSalt Us)))
      funext s
      rw [← mword_place_self, e, mword_place_ne hu]
  · rintro ⟨u, c⟩ (⟨q, e⟩ | ⟨u', i, τ, ℓ, e⟩)
    · refine hconst u c q (funext fun s => ?_)
      rw [← mword_place_self, e]
    · by_cases hu : u' = u
      · subst hu
        refine (hD (Us u') (hL u')).2 c (Or.inr ⟨i, τ, ℓ, funext fun s => ?_⟩)
        rw [← mword_place_self, e]
        simp [place, Function.update_self]
      · refine hconst u c 0 (funext fun s => ?_)
        rw [← mword_place_self, e]
        simp [place, Function.update_of_ne hu, zeroSalt]

theorem h1tGamma (hD : DistinctLiveH1T) : H1TGamma := by
  intro pr hpr N Us hL fpSem L P hLf hP hout
  obtain ⟨hinj, hfree⟩ := h1tUnitsDistinct hD N Us hL
  let ts : List (MSalt Us → ℚ) := (Finset.univ : Finset (Idx Us)).toList.map (mword Us)
  have hlen : ts.length = ∑ u, (Us u).m * (Us u).n * Fintype.card (Pos (Us u).T) := by
    simp only [ts, List.length_map, Finset.length_toList, Finset.card_univ, Fintype.card_sigma, Fintype.card_prod,
      Fintype.card_fin, mul_assoc]
  have hDL : DistinctLiveAt 0 Set.univ (mfree Us) ts := by
    refine ⟨Finset.univ.image (mword Us), ?_, ⟨?_, ?_⟩, ?_⟩
    · intro t ht
      obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 ht
      simp [ts]
    · intro t _ t' _ h
      exact funext fun g => h g trivial
    · intro t ht f hf h
      obtain ⟨x, -, rfl⟩ := Finset.mem_image.1 ht
      exact hfree x ((funext fun g => h g trivial : mword Us x = f) ▸ hf)
    · rw [Finset.card_image_of_injective _ hinj, Finset.card_univ]
      have : ts.length = Fintype.card (Idx Us) := by simp [ts]
      rw [this]; simp
  have hts : ∀ t ∈ ts, ∃ r ∈ regs100 fpSem P L, ∀ g ∈ (Set.univ : Set (MSalt Us)), r g = t g := by
    intro t ht
    obtain ⟨x, -, rfl⟩ := List.mem_map.1 ht
    obtain ⟨r, hr, e⟩ := hout x
    exact ⟨r, hr, fun g _ => e g⟩
  have h := distinctLiveMH100 pr hpr (MSalt Us) (mfree Us) (fun q => Or.inl ⟨q, rfl⟩) 0 Set.univ ts fpSem L P hDL
    hLf hP hts
  rw [hlen] at h
  simpa using h

end Pouw.Fp8Atom.H1T.Proofs
