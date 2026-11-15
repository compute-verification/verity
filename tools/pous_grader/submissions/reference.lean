import Proofs.Pous.Statements

/-! The grader's positive control: a proof of `Pous.Guarantees.KFloor`. -/

theorem solution : Pous.Guarantees.KFloor := by
  refine ⟨by norm_num [Pous.Params.ρ, Pous.Params.δ], fun k hk => ?_⟩
  by_contra h
  have hmono : Pous.Params.ρ ^ 85 ≤ Pous.Params.ρ ^ k :=
    pow_le_pow_of_le_one (by norm_num [Pous.Params.ρ]) (by norm_num [Pous.Params.ρ]) (by omega)
  have h85 : Pous.Params.δ < Pous.Params.ρ ^ 85 := by norm_num [Pous.Params.ρ, Pous.Params.δ]
  exact absurd (h85.trans_le (hmono.trans hk)) (lt_irrefl _)
