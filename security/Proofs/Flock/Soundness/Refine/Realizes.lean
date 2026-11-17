module

public import Proofs.Flock.Soundness.Refine.Tapes
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.CategoryTheory.Category.Init
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import VCVio.Prelude.Core

/-! `FoldRealizes`: the executable's circuit fold computes a model statement's; a fact about the statement builder
that the refinement takes as a hypothesis, not a verifier check. -/

@[expose] public section

namespace FlockSoundness.Refine

open Flock

local notation "𝔽" => FlockLevel3.GF128

structure FoldRealizes (c : CircuitFold) (S : Model.Statement) : Prop where
  kLog : c.kLog = S.kLog
  pin : c.pin = S.pin.val
  fold : ∀ (α : F128) (lw ρ v : Array F128), lw.size = 64 → ρ.size = S.kLog - 6 → c.fold α lw ρ = .ok v →
    ∀ col : Fin (2 ^ S.kLog), fOf v[col.val]! = ∑ i : Fin (2 ^ S.kLog),
      fOf lw[i.val % 64]! * Model.eqAt (vecOf (S.kLog - 6) ρ) (i.val / 64) *
        (fOf α * ((S.A₀ i col).val : 𝔽) + ((S.B₀ i col).val : 𝔽))
  folded : ∀ (α β : F128) (lw ρ ts w : Array F128), c.folded α β lw ρ ts = .ok w →
    ∃ v, c.fold α lw ρ = .ok v ∧ v.size = 2 ^ S.kLog ∧
      w = ts.foldl (fun acc t => halve t acc) (v.modify c.pin (· + β))

end FlockSoundness.Refine
