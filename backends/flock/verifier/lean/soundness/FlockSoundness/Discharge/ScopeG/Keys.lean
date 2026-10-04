import FlockSoundness.Discharge.Placed.Keyed
import FlockSoundness.Discharge.ScopeG.Scope

set_option Elab.async false

/-!
# The VU program's input gates on Δ's source keys

`vuProg` puts VU input `b` of every unit on the gate of its key (`vuKeyIdx`, `Aliased.srcKey` at the VU-stacked unit
input `vuIns c b`), as `Placed.keyProg` does at `Aliased.keyIdx`. So, as there, two inputs of one unit are on one gate
only when they have one key (`vuProg_wire_key`): Δ copies one column into both, or both read the zero.
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3 Compose

variable {c : Circuit}

/-- **A VU input's code is `none` exactly at the zero** (`Aliased.keyIdx_eq_none` at `vuKey`). -/
theorem vuKeyIdx_eq_none {B : ℕ} (x : Fin B) : vuKeyIdx c B x = none ↔ vuKey c x.val = .zero := by
  unfold vuKeyIdx; split
  · exact iff_of_true rfl ‹_›
  · refine iff_of_false (fun h => ?_) ‹_›
    rw [List.find?_eq_none] at h
    exact h x (List.mem_finRange x) (by simp)

/-- **VU inputs with one code have one key** (`Aliased.keyIdx_eq` at `vuKey`). -/
theorem vuKeyIdx_eq {B : ℕ} {x x' : Fin B} (h : vuKeyIdx c B x = vuKeyIdx c B x') : vuKey c x.val = vuKey c x'.val := by
  by_cases hz : vuKey c x.val = .zero
  · have := (vuKeyIdx_eq_none x).2 hz
    rw [h] at this
    rw [hz, (vuKeyIdx_eq_none x').1 this]
  · have hz' : vuKey c x'.val ≠ .zero := fun hz' =>
      hz ((vuKeyIdx_eq_none x).1 (h.trans ((vuKeyIdx_eq_none x').2 hz')))
    unfold vuKeyIdx at h
    rw [ite_eq_right hz, ite_eq_right hz'] at h
    rcases hf : (List.finRange B).find? (fun i => vuKey c i.val = vuKey c x.val) with _ | i
    · rw [List.find?_eq_none] at hf
      exact absurd (by simp) (hf x (List.mem_finRange x))
    · rw [hf] at h
      have h1 := List.find?_some hf
      have h2 := List.find?_some h.symm
      simp only [decide_eq_true_eq] at h1 h2
      exact h1.symm.trans h2

/-- **VU inputs with one key have one code** (the converse of `vuKeyIdx_eq`): the first VU input with that key. -/
theorem vuKeyIdx_eq_of_key {B : ℕ} {x x' : Fin B} (h : vuKey c x.val = vuKey c x'.val) :
    vuKeyIdx c B x = vuKeyIdx c B x' := by
  unfold vuKeyIdx
  simp only [h]

/-- **Two inputs of one unit of `vuProg` on one gate have one key** (`vuKey`), in the program's own instance
(`Placed.keyProg_wire_key` at the VU's program). -/
theorem vuProg_wire_key (hs : scopeOkGT c = true) {n : ℕ} (outs : Finset (Fin (vuN c hs n))) (u : Fin n)
    (x x' : Fin ((vuProg c hs n).rows u).w) (hx : x.val < ((vuProg c hs n).rows u).nIn)
    (hx' : x'.val < ((vuProg c hs n).rows u).nIn)
    (hw : (Const.Prog.instC (vuProg c hs n) outs u).wire x = (Const.Prog.instC (vuProg c hs n) outs u).wire x') :
    vuKey c x.val = vuKey c x'.val := by
  have e := congrArg Rows.nIn (rows_vuProg c hs n u)
  have h1 : x.val < vuB c hs := lt_of_lt_of_eq hx e
  have h2 : x'.val < vuB c hs := lt_of_lt_of_eq hx' e
  rw [Const.Prog.instC_wire_input _ _ _ x hx, Const.Prog.instC_wire_input _ _ _ x' hx'] at hw
  have hv : (Const.Prog.insAt (vuProg c hs n) u x.val).val = (Const.Prog.insAt (vuProg c hs n) u x'.val).val :=
    congrArg Fin.val hw
  have e1 : (Const.Prog.insAt (vuProg c hs n) u x.val).val =
      (Placed.insOf n (vuB c hs) (vuKeyIdx c (vuB c hs)) u.val ⟨x.val, h1⟩).val :=
    Placed.insAt_uniform n u x.val h1
  have e2 : (Const.Prog.insAt (vuProg c hs n) u x'.val).val =
      (Placed.insOf n (vuB c hs) (vuKeyIdx c (vuB c hs)) u.val ⟨x'.val, h2⟩).val :=
    Placed.insAt_uniform n u x'.val h2
  exact vuKeyIdx_eq (B := vuB c hs) (x := ⟨x.val, h1⟩) (x' := ⟨x'.val, h2⟩)
    ((Placed.insOf_eq_iff u.isLt ⟨x.val, h1⟩ ⟨x'.val, h2⟩).1 (Fin.ext (e1.symm.trans (hv.trans e2))))

/-- **Inputs of one unit of `vuProg` with one key are on one gate**: the converse of `vuProg_wire_key`. -/
theorem vuProg_wire_of_key (hs : scopeOkGT c = true) {n : ℕ} (outs : Finset (Fin (vuN c hs n))) (u : Fin n)
    (x x' : Fin ((vuProg c hs n).rows u).w) (hx : x.val < ((vuProg c hs n).rows u).nIn)
    (hx' : x'.val < ((vuProg c hs n).rows u).nIn) (hk : vuKey c x.val = vuKey c x'.val) :
    (Const.Prog.instC (vuProg c hs n) outs u).wire x = (Const.Prog.instC (vuProg c hs n) outs u).wire x' := by
  have e := congrArg Rows.nIn (rows_vuProg c hs n u)
  have h1 : x.val < vuB c hs := lt_of_lt_of_eq hx e
  have h2 : x'.val < vuB c hs := lt_of_lt_of_eq hx' e
  rw [Const.Prog.instC_wire_input _ _ _ x hx, Const.Prog.instC_wire_input _ _ _ x' hx']
  have e1 : (Const.Prog.insAt (vuProg c hs n) u x.val).val =
      (Placed.insOf n (vuB c hs) (vuKeyIdx c (vuB c hs)) u.val ⟨x.val, h1⟩).val :=
    Placed.insAt_uniform n u x.val h1
  have e2 : (Const.Prog.insAt (vuProg c hs n) u x'.val).val =
      (Placed.insOf n (vuB c hs) (vuKeyIdx c (vuB c hs)) u.val ⟨x'.val, h2⟩).val :=
    Placed.insAt_uniform n u x'.val h2
  have hi := (Placed.insOf_eq_iff (B := vuB c hs) (src := vuKeyIdx c (vuB c hs)) u.isLt ⟨x.val, h1⟩ ⟨x'.val, h2⟩).2
    (vuKeyIdx_eq_of_key (B := vuB c hs) (x := ⟨x.val, h1⟩) (x' := ⟨x'.val, h2⟩) hk)
  exact Fin.ext (e1.trans ((congrArg Fin.val hi).trans e2.symm))

end FlockSoundness.Discharge.ScopeG
