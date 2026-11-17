module

public import Proofs.Flock.Soundness.Discharge.Const.ConstCols
public import Proofs.Flock.Soundness.Discharge.Placed.Keyed
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Discharge.Zero.Uniform

/-! `hConst` at the proving program's places (`constCols_keyProg`), the twin of `Zero.zeroCols_keyProg`. -/

@[expose] public section

namespace FlockSoundness.Discharge.Const

open FlockSoundness Audit Flock

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length} {c : Flock.Circuit}

theorem constCols_keyProg (hU : Built c.unit) {si : Placed.Sites plan tab}
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n))) :
    ((Placed.keyProg c hU n).partition outs).ConstCols plan tab
      (Placed.Sites.progPlaces_keyProg hU hparse outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  constCols_of_instC _ (Zero.oneGate_uniform n) (Placed.Sites.progPlaces_keyProg_inst hU hparse outs)

end FlockSoundness.Discharge.Const
