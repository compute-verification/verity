module

public import Proofs.Properties.InferenceOnly.RecordJson

/-!
# The digits of a wire

SPOT's reader (`Spot.Wires.Wire.ofJson`) refuses a wire that has no digit, a digit with no step, or a digit of radix
zero (`wire_digits`). The record's wiring is `Spot.Edges.edgesOf`'s, and that each of its wires has a step for each
digit and no digit of radix zero is `EdgeDigits` (proved: `edgeDigits`), which is what the record check's test that
a wire stays inside a slot (`Record.sameSlot`) is sound on.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Lean (Json)
open Spot.Wires
open InferenceOnlyVerifier.Record.SecurityProofs.Json (bind_ok mapM_ok)

/-- **A wire SPOT's reader reads has a step for each digit, and no digit of radix zero.** -/
theorem wire_digits {j : Json} {w : Wire} (h : Wire.ofJson j = .ok w) :
    w.step.length = w.radix.length ∧ ∀ r ∈ w.radix, 1 ≤ r := by
  have hc : (w.radix.isEmpty || w.radix.length != w.step.length || w.radix.any (· < 1)) = false := by
    unfold Wire.ofJson at h
    dsimp only at h
    repeat' (first
      | (replace h := bind_ok h
         obtain ⟨_, hx, h⟩ := h
         first
           | (cases hx; done)
           | (try dsimp only at h))
      | (split at h))
    all_goals (cases h; exact Bool.eq_false_iff.2 (by assumption))
  simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq, List.any_eq_false, decide_eq_true_eq] at hc
  obtain ⟨⟨-, hlen⟩, hrad⟩ := hc
  exact ⟨hlen.symm, fun r hr => by have := hrad r hr; omega⟩

/-- **Every wire of a wiring SPOT's reader reads is a wire it reads.** -/
theorem wiring_wires {j : Json} {W : Wiring} (h : Wiring.ofJson j = .ok W) :
    ∀ w ∈ W.wires, ∃ jw, Wire.ofJson jw = .ok w := by
  unfold Wiring.ofJson at h
  replace h := bind_ok h
  obtain ⟨jt, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lt, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨ts', -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨jc, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lc, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨cs, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨jw, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨lw, -, h⟩ := h
  replace h := bind_ok h
  obtain ⟨wires, hwires, h⟩ := h
  have hW : W.wires = wires := by
    split at h
    · replace h := bind_ok h
      obtain ⟨a1, -, h⟩ := h
      replace h := bind_ok h
      obtain ⟨a2, -, h⟩ := h
      replace h := bind_ok h
      obtain ⟨a3, -, h⟩ := h
      cases h
      rfl
    · replace h := bind_ok h
      obtain ⟨a3, -, h⟩ := h
      cases h
      rfl
  intro w hw
  rw [hW] at hw
  obtain ⟨x, -, hx⟩ := (mapM_ok _ _ _ hwires w).1 hw
  exact ⟨x, hx⟩

/-- **Every wire of the record's wiring has a step for each digit, and no digit of radix zero.** The wires are
`Spot.Edges.edgesOf`'s, each of one digit; proving it of `edgesOf` replaces this hypothesis. -/
def EdgeDigits : Prop :=
  ∀ (P : Flock.Program.Prog) (doc : Json) (text : Wiring), Spot.RecordText.wiring P doc = .ok text →
    ∀ w ∈ text.wires, w.step.length = w.radix.length ∧ ∀ r ∈ w.radix, 1 ≤ r

end InferenceOnlyVerifier.Record.SecurityProofs
