module

public import Flock.Desc
public import Properties.Flock.Circuit.Keyed

/-!
# Every unit of the circuit's program has the circuit's rows

Unit `u` of `uniform … R` has rows `R` (`rows_uniform`), so every unit of the circuit's program (`circuitProg`,
`rows_circuitProg`) and of the key program (`keyProg`, `rows_keyProg`) has the circuit's template instance rows,
which `Flock.Bridge.unitInst` casts along. Moved unchanged from `Proofs.Flock.Soundness.Discharge.Placed.Setup`,
`Proofs.Flock.Soundness.Discharge.Placed.Wiring` and `Proofs.Flock.Bridge.UnitRows`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Placed

/-- Every unit of `uniform` has rows `R`. -/
theorem rows_uniform (k : ℕ) (R : Rows) (ins : ℕ → Fin R.nIn → Fin k) (one : Fin k) (hone : ∀ u c, ins u c ≠ one) :
    ∀ n (u : Fin n), (uniform k R ins one hone n).rows u = R
  | 0, u => u.elim0
  | n + 1, u => by
      simp only [uniform, Prog.rows]
      split
      · exact rows_uniform k R ins one hone n _
      · rfl

/-- Every unit of the circuit's program has the circuit's template instance rows. -/
theorem rows_circuitProg (c : Flock.Circuit) (hU : Built c.unit) (nU B : ℕ)
    (src : Fin (circuitRows c hU).nIn → Option (Fin B)) (u : Fin nU) :
    (circuitProg c hU nU B src).rows u = circuitRows c hU :=
  rows_uniform _ _ _ _ _ nU u

end FlockSoundness.Discharge.Placed

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)

/-- Every unit of the key program has the circuit's template instance rows. -/
theorem rows_keyProg (u : Fin n) : (Placed.keyProg D.unroll hU n).rows u = Placed.circuitRows D.unroll hU :=
  Placed.rows_circuitProg _ _ _ _ _ u

end Flock.Bridge
