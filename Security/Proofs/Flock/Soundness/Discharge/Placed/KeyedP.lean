import Proofs.Flock.Soundness.Discharge.Placed.Keyed
import Proofs.Flock.Soundness.Discharge.Aliased.KeysP

/-!
# The circuit's program on Δ's source keys with public inputs

`keyProgP c pi hU n` is `keyProg` at the keys with public inputs (`Aliased.keyIdxP`): each unit's inputs at one bit of
one public word share one gate, its other inputs are on `keyProg`'s gates. Two inputs of one unit are on one gate only
when they have one key (`keyProgP_wire_key`), which `HiddenExec.aliased_of_keyP` reads; without a public leaf map it is
`keyProg` (`keyProgP_empty`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Placed

open Model Audit

/-- **The circuit's program on Δ's source keys with public inputs**: `n` units of the circuit's template instance rows,
input `x` of each on the gate of its key (`Aliased.keyIdxP`), the zero's gate when it reads the zero. -/
noncomputable abbrev keyProgP (c : Flock.Circuit) (pi : HmIn.Ports) (hU : Built c.unit) (n : ℕ) :
    Prog (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n) n :=
  circuitProg c hU n (circuitRows c hU).nIn (Aliased.keyIdxP c pi (circuitRows c hU).nIn)

/-- **Without a public leaf map, `keyProgP` is `keyProg`.** -/
theorem keyProgP_empty {c : Flock.Circuit} {pi : HmIn.Ports} (h : pi.leaves = #[]) (hU : Built c.unit) (n : ℕ) :
    keyProgP c pi hU n = keyProg c hU n := by
  simp only [keyProgP, keyProg, Aliased.keyIdxP_empty h]

/-- **Two inputs of one unit of `keyProgP` on one gate have one key** (`Aliased.srcKeyP`), in the program's own
instance. -/
theorem keyProgP_wire_key {c : Flock.Circuit} {pi : HmIn.Ports} (hU : Built c.unit) {n : ℕ}
    (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n))) (u : Fin n)
    (x x' : Fin ((keyProgP c pi hU n).rows u).w) (hx : x.val < ((keyProgP c pi hU n).rows u).nIn)
    (hx' : x'.val < ((keyProgP c pi hU n).rows u).nIn)
    (hw : (Const.Prog.instC (keyProgP c pi hU n) outs u).wire x =
      (Const.Prog.instC (keyProgP c pi hU n) outs u).wire x') :
    Aliased.srcKeyP c pi x.val = Aliased.srcKeyP c pi x'.val := by
  have e := congrArg Rows.nIn (rows_circuitProg c hU n (circuitRows c hU).nIn
    (Aliased.keyIdxP c pi (circuitRows c hU).nIn) u)
  have h1 : x.val < (circuitRows c hU).nIn := lt_of_lt_of_eq hx e
  have h2 : x'.val < (circuitRows c hU).nIn := lt_of_lt_of_eq hx' e
  rw [Const.Prog.instC_wire_input _ _ _ x hx, Const.Prog.instC_wire_input _ _ _ x' hx'] at hw
  have hv : (Const.Prog.insAt (keyProgP c pi hU n) u x.val).val =
      (Const.Prog.insAt (keyProgP c pi hU n) u x'.val).val :=
    congrArg Fin.val hw
  have e1 : (Const.Prog.insAt (keyProgP c pi hU n) u x.val).val =
      (insOf n (circuitRows c hU).nIn (Aliased.keyIdxP c pi (circuitRows c hU).nIn) u.val ⟨x.val, h1⟩).val :=
    insAt_uniform n u x.val h1
  have e2 : (Const.Prog.insAt (keyProgP c pi hU n) u x'.val).val =
      (insOf n (circuitRows c hU).nIn (Aliased.keyIdxP c pi (circuitRows c hU).nIn) u.val ⟨x'.val, h2⟩).val :=
    insAt_uniform n u x'.val h2
  exact Aliased.keyIdxP_eq (n := (circuitRows c hU).nIn) (x := ⟨x.val, h1⟩) (x' := ⟨x'.val, h2⟩)
    ((insOf_eq_iff u.isLt ⟨x.val, h1⟩ ⟨x'.val, h2⟩).1 (Fin.ext (e1.symm.trans (hv.trans e2))))

end FlockSoundness.Discharge.Placed
