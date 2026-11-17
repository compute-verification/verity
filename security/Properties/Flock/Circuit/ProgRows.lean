module

public import Flock.Desc
public import Properties.Flock.Circuit.Keyed

/-! Every unit of the circuit's program has the circuit's rows. -/

@[expose] public section

namespace FlockSoundness.Discharge.Placed

theorem rows_uniform (k : ℕ) (R : Rows) (ins : ℕ → Fin R.nIn → Fin k) (one : Fin k) (hone : ∀ u c, ins u c ≠ one) :
    ∀ n (u : Fin n), (uniform k R ins one hone n).rows u = R
  | 0, u => u.elim0
  | n + 1, u => by
      simp only [uniform, Prog.rows]
      split
      · exact rows_uniform k R ins one hone n _
      · rfl

theorem rows_circuitProg (c : Flock.Circuit) (hU : Built c.unit) (nU B : ℕ)
    (src : Fin (circuitRows c hU).nIn → Option (Fin B)) (u : Fin nU) :
    (circuitProg c hU nU B src).rows u = circuitRows c hU :=
  rows_uniform _ _ _ _ _ nU u

end FlockSoundness.Discharge.Placed

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)

theorem rows_keyProg (u : Fin n) : (Placed.keyProg D.unroll hU n).rows u = Placed.circuitRows D.unroll hU :=
  Placed.rows_circuitProg _ _ _ _ _ u

end Flock.Bridge
