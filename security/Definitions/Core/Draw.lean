module

public import Mathlib.Data.Fintype.Pi
public import Definitions.Core.Game
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! The sampling designs every area reads (`SamplingDesign`, `Draw`, `Draw.Subset`, `Coins`). -/

@[expose] public section

namespace Verity

open scoped ENNReal

/-- A sampling design over the units `U`: the verifier's uniform coins, and the unit set each samples. -/
structure SamplingDesign (U : Type) where
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  draw : Ω → Finset U

attribute [instance] SamplingDesign.fintype SamplingDesign.nonempty

namespace SamplingDesign

variable {U : Type}

/-- The probability that a sample misses every unit of `B`. -/
noncomputable def missProb (L : SamplingDesign U) (B : Finset U) : ℝ≥0∞ :=
  Game.prCoin fun ω => Disjoint (L.draw ω) B

/-- The count curve's sampling term: the largest missProb over sets of at least `K` units. -/
noncomputable def miss (L : SamplingDesign U) (K : ℕ) : ℝ≥0∞ :=
  ⨆ (B : Finset U) (_ : K ≤ B.card), L.missProb B

noncomputable def incl (L : SamplingDesign U) (u : U) : ℝ≥0∞ :=
  Game.prCoin fun ω => u ∈ L.draw ω

end SamplingDesign

/-- **An index sample**: `s` indices into `N` units, each uniform and independent of the others. -/
abbrev Draw (s N : ℕ) : Type := Fin s → Fin N

def Draw.samplingDesign (s N : ℕ) [NeZero N] : SamplingDesign (Fin N) where
  Ω := Draw s N
  nonempty := ⟨fun _ => ⟨0, Nat.pos_of_ne_zero (NeZero.ne N)⟩⟩
  draw ω := Finset.univ.image ω

abbrev Draw.Subset (N k : ℕ) : Type := {s : Finset (Fin N) // s.card = k}

/-- **A window's coins**: the public randomness `Pub` the window's checks read, and an index sample of `s` from `N`. -/
abbrev Coins (Pub : Type) (s N : ℕ) : Type := Pub × Draw s N

end Verity
