module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
public import Proofs.Flock.SpotFlock.RunsAs.Round2

/-! `RunsAs.RoundTwo` at every held copy (`stmtOk_of_check`), with `RegistersNone` for the inputs premise. -/

@[expose] public section

namespace SpotFlock.RunsAs

open Claims
open Lean (Json)

theorem stmtOk_of_check (h : Held) : RoundTwo h := by
  intro own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hRN hT hN hRec hR hSc hP hc
  have hi : own.interior = rec := congrArg Spot.SecondStage.Own.interior hA
  have hIN : Round2.InputsNone own := by
    apply Round2.inputsNone_of_registersNone
    rw [hi]
    exact hRN
  exact Round2.stmtOk_of_check h own ss x m₁ m₂ c₀ c₁ st rec draw hx hm₁ hd hA hS hIN hT hN hRec hR hSc hP hc

end SpotFlock.RunsAs
