module

public import Proofs.Pouw.PearlPlus.Numerics
public import Proofs.Pouw.PearlPlus.Rules

/-!
# Pearl+'s bridge, part (c): `s5_stats`' ρ is a finite word

`Words.s5Stats` refuses a non-finite `t = s + 4ρ` (`finite (f32AddV2 s (f32MulV2 S5_FOUR rho))`). Where the stats are a
value, `t` is finite, so both of its addends are (`fp32AddIsTheSpec`), among them `f32MulV2 S5_FOUR ρ`. Core's product of
any word with a NaN or an infinity is `F32_NAN_PTX` or an infinity, so ρ decodes finite, and a word that decodes finite
is `wordQ`-finite.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

namespace Rho

theorem bind_ok {ε α β : Type} {m : Except ε α} {f : α → Except ε β} {b : β} (h : (m >>= f) = .ok b) :
    ∃ a, m = .ok a ∧ f a = .ok b := by
  cases m with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem s5Stats_t {x : Array ℕ} {s ρw αw βw : ℕ} (h : Words.s5Stats x = .ok (s, ρw, αw, βw)) :
    ∃ s₀ t, Words.finite (Evaluate.Fp32.f32AddV2 s₀ (Evaluate.Fp32.f32MulV2 Words.S5_FOUR ρw)) = .ok t := by
  unfold Words.s5Stats at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨t, ht, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  cases h
  exact ⟨_, t, ht⟩

theorem wordQ_of_finite {w t : ℕ} (h : Words.finite w = .ok t) : (wordQ w).isSome := by
  have h1 : (w >>> 23) &&& 0xFF = (w >>> 23) % 256 := Nat.and_two_pow_sub_one_eq_mod _ 8
  unfold Words.finite at h
  rw [h1] at h
  by_cases he : (w >>> 23) % 256 = 255
  · rw [he] at h
    cases h
  · rw [PeelExactProofs.wordQ_fields he]
    rfl

theorem wordQ_mul_nonfin (a b : ℕ) (hb : ∀ s m, Verity.Fp32.decode b ≠ .fin s m) :
    wordQ (Verity.Fp32.f32MulV2 a b) = none := by
  have hd : ∀ s m, (Verity.Fp32.decode a).mul (Verity.Fp32.decode b) ≠ .fin s m := by
    intro s m
    cases Verity.Fp32.decode a <;> cases hdb : Verity.Fp32.decode b
    all_goals first
      | exact absurd hdb (hb _ _)
      | (simp only [Verity.Fp32.Decoded.mul]
         try split
         all_goals intro h; cases h)
  unfold Verity.Fp32.f32MulV2
  generalize (Verity.Fp32.decode a).mul (Verity.Fp32.decode b) = d at hd ⊢
  cases d with
  | nan => exact Numerics.wordQ_nanPtx
  | inf s => exact Numerics.wordQ_infWord s
  | fin s m => exact absurd rfl (hd s m)

end Rho

open Rho Numerics in
theorem statsRhoIsFinite : StatsRhoIsFinite := by
  intro x s ρw αw βw h
  obtain ⟨s₀, t, ht⟩ := s5Stats_t h
  obtain ⟨-, hm, -⟩ := fp32AddIsTheSpec _ _ (wordQ_of_finite ht)
  rw [show Evaluate.Fp32.f32MulV2 = Verity.Fp32.f32MulV2 from Properties.Evaluator.Fp32.f32MulV2] at hm
  by_cases hf : ∃ s m, Verity.Fp32.decode ρw = .fin s m
  · obtain ⟨_, _, hd⟩ := hf
    obtain ⟨y, hy⟩ := wordQ_of_decode hd
    rw [hy]
    rfl
  · rw [wordQ_mul_nonfin _ _ fun s m hd => hf ⟨s, m, hd⟩] at hm
    exact absurd hm (by decide)

end Pouw.PearlPlus.Bridge
