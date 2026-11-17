module

public import Definitions.Poub.Remote.Run
public import Definitions.Poub.RateAuditDeployed
import Mathlib.Basic.ENNReal.Real

/-! The deployed game's session read as a run of PoUB (`Run.ofDeployed`); what the game does not model is `0`, and no
route proved for the run reads it. -/

@[expose] public section

namespace ResourceAccounting.Remote

open ResourceAccounting.Rate

variable {n P N b h ρ S L q r s d G τ : ℕ}

/-- A deployed session as a run of PoUB on node `Nd` with parameters `a`, with `X` bits completed a pass. -/
def Run.ofDeployed (A : DeployedRun n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (X : ℕ) (Nd : Node (Fin τ))
    (a : Audit) : Run Nd a (Store N b × (Fin n → WindowCoins P N b h ρ r s d)) where
  accepts x := ∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll x.1 (x.2 w).1
  committed _ := n * P * N
  correct x := ∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card
  read x m := ∑ w, ∑ j,
    ((A.win w x.2).dev j x.1 (x.2 w).1).tierReads tier m x.1 ((x.2 w).1 j).1 ((A.win w x.2).oracle j x.1 (x.2 w).1)
  readOut _ := n * P * (L - ρ)
  state _ := n * P * S
  stateOut _ := 0
  rebuilt _ := n * P * X
  rebuiltOut _ := 0
  rebuildWork _ := 0
  pouwWork _ := 0
  inBits _ := 0
  outBits _ := 0
  zOut _ := 0
  workOut _ := 0
  other _ := 0

/-- Window `w` of a deployed session as a run of PoUB on node `Nd` with parameters `a`, over the session's
coins: `Run.ofDeployed`'s fields at window `w`'s `P` passes, accepting when every window's sample passes. -/
def Run.ofDeployedWindow (A : DeployedRun n P N b h ρ S L q r s d G) (w : Fin n) (tier : ℕ → Fin τ) (X : ℕ)
    (Nd : Node (Fin τ)) (a : Audit) : Run Nd a (Store N b × (Fin n → WindowCoins P N b h ρ r s d)) where
  accepts x := ∀ v t, (x.2 v).2 t ∈ (A.win v x.2).correctAll x.1 (x.2 v).1
  committed _ := P * N
  correct x := ((A.win w x.2).correctAll x.1 (x.2 w).1).card
  read x m := ∑ j,
    ((A.win w x.2).dev j x.1 (x.2 w).1).tierReads tier m x.1 ((x.2 w).1 j).1 ((A.win w x.2).oracle j x.1 (x.2 w).1)
  readOut _ := P * (L - ρ)
  state _ := P * S
  stateOut _ := 0
  rebuilt _ := P * X
  rebuiltOut _ := 0
  rebuildWork _ := 0
  pouwWork _ := 0
  inBits _ := 0
  outBits _ := 0
  zOut _ := 0
  workOut _ := 0
  other _ := 0

end ResourceAccounting.Remote
