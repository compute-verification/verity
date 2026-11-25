module

public import Definitions.Hash.Assumptions
public import Proofs.Core.Game.Draw

/-! A forger's probability is the value of the run of its one uniform coin. -/

@[expose] public section

namespace Hash.Assumptions

open Verity

/-- `LmsForger.Pr` is the value, as a real, of the run in which the forger's coin is drawn. -/
theorem LmsForger.pr_eq_toReal_value (verify : Msg → Msg → Msg → Bool) (F : LmsForger) :
    F.Pr verify = (Game.value (F.Forges verify) (Game.draw F.S)).toReal :=
  Game.pr_eq_toReal_value (F.Forges verify)

end Hash.Assumptions
