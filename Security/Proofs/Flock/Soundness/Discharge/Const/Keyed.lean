import Proofs.Flock.Soundness.Discharge.Placed.Keyed
import Proofs.Flock.Soundness.Discharge.Zero.Uniform
import Proofs.Flock.Soundness.Discharge.Const.ConstCols

/-!
# `hConst` at the production program's places

The production (untyped) statement's program is `Placed.keyProg c hU n`: `n` units of the circuit's template instance
rows, each input on the gate of its Δ source key, each constant on the one gate `Placed.oneGate`
(`Zero.oneGate_uniform`). `Placed.Sites.progPlaces_keyProg` places it from the sites and the parse alone, with the
program's own instance (`Placed.Sites.progPlaces_keyProg_inst`). So only a unit's constant column sits on the one gate:
`constCols_keyProg`, the twin of `Zero.zeroCols_keyProg`, at the same places and from the same arguments.
-/

namespace FlockSoundness.Discharge.Const

open FlockSoundness Audit Flock

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {c : Flock.Circuit}

/-- **`hConst` at `Placed.Sites.progPlaces_keyProg`**, the places of the circuit's program on Δ's source keys that take
only the sites and the parse: only a unit's constant column sits on the one gate. -/
theorem constCols_keyProg (hU : Built c.unit) {si : Placed.Sites plan tab}
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n))) :
    ((Placed.keyProg c hU n).partition outs).ConstCols plan tab
      (Placed.Sites.progPlaces_keyProg hU hparse outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  constCols_of_instC _ (Zero.oneGate_uniform n) (Placed.Sites.progPlaces_keyProg_inst hU hparse outs)

end FlockSoundness.Discharge.Const
