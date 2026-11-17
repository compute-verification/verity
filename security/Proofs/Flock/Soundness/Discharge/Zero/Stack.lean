module

public import Proofs.Flock.Soundness.Discharge.Zero.Delta

/-! A VU's stacked inputs that read the zero (`ZeroIn`), the same at every VU. -/

@[expose] public section

open Flock

namespace FlockSoundness.Discharge.Zero

def InPort (c : Circuit) (off : ℕ) : Prop := off ∈ portBits c.unit.inGroups

def ZeroIn (c : Circuit) (x : ℕ) : Prop :=
  InPort c (x % (c.unit.inWords * WORD)) ∧ ZeroAt c (x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD))

instance instDecidablePredNatZeroIn (c : Circuit) : DecidablePred (ZeroIn c) := fun _ => by
  unfold ZeroIn InPort ZeroAt ZeroPort
  infer_instance

theorem zeroAt_mod (c : Circuit) (s₀ off : ℕ) : ZeroAt c (s₀ % c.upv) off ↔ ZeroAt c s₀ off := by
  simp only [ZeroAt, ZeroPort, Nat.mod_mod]

theorem zeroAt_vu (c : Circuit) (g i off : ℕ) : ZeroAt c (g * c.upv + i) off ↔ ZeroAt c i off := by
  rw [← zeroAt_mod c (g * c.upv + i), ← zeroAt_mod c i, Nat.add_comm, Nat.add_mul_mod_self_right]

theorem zeroIn_vu {c : Circuit} {x : ℕ} (h : ZeroIn c x) (g : ℕ) :
    InPort c (x % (c.unit.inWords * WORD)) ∧
      ZeroAt c (g * c.upv + x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD)) :=
  ⟨h.1, (zeroAt_vu c g _ _).2 h.2⟩

theorem flatSrcAt_zeroIn {c : Circuit} {x : ℕ} (h : ZeroIn c x) (g : ℕ) :
    flatSrcAt c (g * c.upv + x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD)) =
        c.unitRange.slot (g * c.upv + x / (c.unit.inWords * WORD)) + c.unit.useful ∨
      flatSrcAt c (g * c.upv + x / (c.unit.inWords * WORD)) (x % (c.unit.inWords * WORD)) =
        c.unitRange.slot 0 + c.unit.useful :=
  flatSrcAt_zero (zeroIn_vu h g).2

def withZero (c : Circuit) {n m k : ℕ} (zer : Fin k) (rest : Fin n → Fin m → Fin k) (u : Fin n) (x : Fin m) : Fin k :=
  if ZeroIn c x.val then zer else rest u x

theorem withZero_eq_zer (c : Circuit) {n m k : ℕ} {zer : Fin k} {rest : Fin n → Fin m → Fin k}
    (hrest : ∀ u x, rest u x ≠ zer) (u : Fin n) (x : Fin m) : withZero c zer rest u x = zer ↔ ZeroIn c x.val := by
  unfold withZero
  split
  · exact iff_of_true rfl ‹_›
  · exact iff_of_false (hrest u x) ‹_›

end FlockSoundness.Discharge.Zero
