import Proofs.Flock.Soundness.Discharge.Composed.Coined

/-!
# The level-0 padding geometry of `CoinedOK`

`geometry_of_padOK`: at `cols` level 0's message columns (`logCols`) and `t_pad` its padding, the padded level 0's
conditions (`Accounting.PadOK`) and rate `1/2` (`logLen = logCols + 1`) give C-Flock ZK's geometry: `cols ≤ 40`,
`t_pad ≤ 2^cols`, and opened positions inside the codeword are inside `2^(cols+1)`.
-/

namespace FlockSoundness.Discharge.Composed

open Accounting

theorem geometry_of_padOK (l : Level) (t nS : ℕ) (pos : Fin nS → ℕ) (hp : PadOK l t)
    (hrate : l.logLen = l.logCols + 1) (hS : nS ≤ t) (hinj : Function.Injective pos)
    (hlt : ∀ q, pos q < 2 ^ l.logLen) :
    l.logCols ≤ 40 ∧ t ≤ 2 ^ l.logCols ∧ nS ≤ t ∧ Function.Injective pos ∧ ∀ q, pos q < 2 ^ (l.logCols + 1) :=
  ⟨by have := hp.hd; omega, hp.ht, hS, hinj, fun q => hrate ▸ hlt q⟩

end FlockSoundness.Discharge.Composed
