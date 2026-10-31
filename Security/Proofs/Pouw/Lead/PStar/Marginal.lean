import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Card
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring

/-!
# A part of the draw caps the whole (proofs)

A draw `ω` drawn uniformly from a finite type `Ω` whose part `get ω ∈ D` can be overwritten (`set ω d`) splits as the
part and the rest, `|Ω| = |D|·|rest|`. So if a value `F ω` moves with the part only through a term `g`,
`F (set ω d) = F ω − g (get ω) + g d`, then `F` hits a value fixed before the draw on at most the share of the draws
that `g` hits it on (`part_fiber_le`): no more than `P/|D|` when no value has more than `P` preimages under `g`. The
coin lanes' per-word `p*` over the whole of a call's coins is this with `D` one step's row bit and coin lanes.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset

variable {Ω D G : Type*} [Fintype Ω] [DecidableEq Ω] [Fintype D] [AddCommGroup G] [DecidableEq G]

/-- **A part of the draw caps the whole**: with `set` overwriting the part `get` reads, and `F` moving with the part only
through `g`, `F` takes any value on at most `P·|Ω|/|D|` draws when `g` takes none on more than `P` parts. The values
are in any additive group, a word's or a whole tile's. -/
theorem part_fiber_le (get : Ω → D) (set : Ω → D → Ω) (get_set : ∀ ω d, get (set ω d) = d)
    (set_set : ∀ ω d d', set (set ω d) d' = set ω d') (set_get : ∀ ω, set ω (get ω) = ω) (F : Ω → G) (g : D → G)
    (hF : ∀ ω d, F (set ω d) = F ω - g (get ω) + g d) {P : ℕ}
    (hP : ∀ v, (univ.filter fun d => g d = v).card ≤ P) (v : G) :
    Fintype.card D * (univ.filter fun ω => F ω = v).card ≤ P * Fintype.card Ω := by
  rcases isEmpty_or_nonempty D with hD | ⟨⟨d₀⟩⟩
  · have : IsEmpty Ω := ⟨fun ω => hD.false (get ω)⟩
    simp
  have hmaps : Set.MapsTo (fun ω => set ω d₀) (univ : Finset Ω) (univ.image fun ω => set ω d₀) :=
    fun ω _ => mem_image_of_mem _ (mem_univ ω)
  have hfib : ∀ y ∈ univ.image (fun ω => set ω d₀),
      (univ.filter fun ω => set ω d₀ = y).card = Fintype.card D := by
    intro y hy
    obtain ⟨ω₀, -, rfl⟩ := mem_image.mp hy
    rw [← card_univ (α := D)]
    refine card_nbij' get (set (set ω₀ d₀)) (fun ω _ => mem_coe.mpr (mem_univ _)) (fun d _ => ?_)
      (fun ω hω => ?_) (fun d _ => get_set _ _)
    · rw [mem_coe, mem_filter]
      exact ⟨mem_univ _, by rw [set_set, set_set]⟩
    · rw [mem_coe, mem_filter] at hω
      rw [← hω.2, set_set, set_get]
  have hcard : Fintype.card Ω = (univ.image fun ω => set ω d₀).card * Fintype.card D := by
    rw [← card_univ, card_eq_sum_card_fiberwise hmaps, sum_congr rfl hfib, sum_const, nsmul_eq_mul, Nat.cast_id]
  have hhit : (univ.filter fun ω => F ω = v).card ≤ P * (univ.image fun ω => set ω d₀).card := by
    have hmaps' : Set.MapsTo (fun ω => set ω d₀) (univ.filter fun ω => F ω = v : Finset Ω)
        (univ.image fun ω => set ω d₀) := fun ω _ => mem_image_of_mem _ (mem_univ ω)
    rw [card_eq_sum_card_fiberwise hmaps']
    calc _ ≤ ∑ _y ∈ univ.image (fun ω => set ω d₀), P := sum_le_sum fun y hy => ?_
      _ = P * _ := by rw [sum_const, nsmul_eq_mul, Nat.cast_id, mul_comm]
    obtain ⟨ω₀, -, rfl⟩ := mem_image.mp hy
    refine (card_le_card_of_injOn get (fun ω hω => ?_) (fun ω hω ω' hω' he => ?_)).trans
      (hP (v - F (set ω₀ d₀) + g d₀))
    · rw [mem_coe, mem_filter, mem_filter] at hω
      rw [mem_coe, mem_filter]
      refine ⟨mem_univ _, ?_⟩
      have h := hF ω d₀
      rw [hω.2, hω.1.2] at h
      rw [h]
      abel
    · rw [mem_coe, mem_filter] at hω hω'
      rw [← set_get ω, ← set_get ω', ← set_set ω d₀, ← set_set ω' d₀, hω.2, hω'.2, he]
  calc Fintype.card D * _ ≤ Fintype.card D * (P * (univ.image fun ω => set ω d₀).card) :=
        Nat.mul_le_mul_left _ hhit
    _ = P * Fintype.card Ω := by rw [hcard]; ring

end Pouw.SecurityProofs.Lead.PStar
