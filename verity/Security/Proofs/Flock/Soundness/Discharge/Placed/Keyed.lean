import Proofs.Flock.Soundness.Discharge.Placed.Wiring
import Proofs.Flock.Soundness.Discharge.Aliased.Keys
import Proofs.Flock.Soundness.Discharge.Const.Inst

/-!
# `ProgPlaces` for the circuit's program on Δ's source keys

`keyProg c hU n` is the circuit's program (`circuitProg`) with each input gate indexed by Δ's source key
(`Aliased.keyIdx`): two inputs of one unit are on one gate exactly when Δ copies one column into them, or both read the
zero (`keyProg_wire_key`). At every drawn unit's site in an accepted untyped statement:

* W6 holds from the parse (`Sites.rowsOf_circuitProg`): every drawn table's circuit file parses to `c`.
* The program's own instance of the unit (`Const.Prog.instC`) is aliased as the accepted layout requires
  (`Sites.aliased_keyProg`, from `Aliased.setupH_aliased_of_key`).

So `Sites.progPlaces_keyProg` is a `ProgPlaces` that takes only the sites and the parse.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Placed

open Model Audit

/-- An instance carried along equal rows wires each column where the original wires it. -/
theorem wire_cast {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R R' : Rows} (e : R = R')
    (inst : P.IsRowsUnit u R) (x : Fin R'.w) :
    (e ▸ inst).wire x = inst.wire (Fin.cast (congrArg Rows.w e).symm x) := by
  subst e; rfl

/-- An accepted statement's circuit is the one its circuit file parses to. -/
theorem Accepted.c_eq {T : Model.Statement} (a : Accepted T) {c : Flock.Circuit}
    (hp : HmRow.parse a.tags a.cf a.tables = .ok c) : a.st.c = c := by
  have := (setupH_spec a.ht a.h).1
  rw [hp] at this
  exact (Except.ok.inj this).symm

/-- `uniform` wires unit `u`'s input `i` to the input gate `ins u i`. -/
theorem insAt_uniform {k : ℕ} {R : Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one} :
    ∀ (n : ℕ) (u : Fin n) (i : ℕ) (hi : i < R.nIn),
      (Const.Prog.insAt (uniform k R ins one hone n) u i).val = (ins u.val ⟨i, hi⟩).val
  | 0, u, _, _ => u.elim0
  | n + 1, u, i, hi => by
    induction u using Fin.lastCases with
    | last =>
      change (Const.Prog.insAt (Prog.snoc (uniform k R ins one hone n) R _ _ _) (Fin.last n) i).val = _
      rw [Const.Prog.insAt_last _ _ _ _ _ i hi]
      simp
    | cast v =>
      change (Const.Prog.insAt (Prog.snoc (uniform k R ins one hone n) R _ _ _) v.castSucc i).val = _
      rw [Const.Prog.insAt_castSucc]
      simp [insAt_uniform n v i hi]

/-- **The circuit's program on Δ's source keys**: `n` units of the circuit's template instance rows, input `x` of each
on the gate of its key (`Aliased.keyIdx`), the zero's gate when it reads the zero. -/
noncomputable abbrev keyProg (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) :
    Prog (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n) n :=
  circuitProg c hU n (circuitRows c hU).nIn (Aliased.keyIdx c (circuitRows c hU).nIn)

/-- **Two inputs of one unit of `keyProg` on one gate have one key** (`Aliased.srcKey`), in the program's own instance. -/
theorem keyProg_wire_key {c : Flock.Circuit} (hU : Built c.unit) {n : ℕ}
    (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n))) (u : Fin n)
    (x x' : Fin ((keyProg c hU n).rows u).w) (hx : x.val < ((keyProg c hU n).rows u).nIn)
    (hx' : x'.val < ((keyProg c hU n).rows u).nIn)
    (hw : (Const.Prog.instC (keyProg c hU n) outs u).wire x = (Const.Prog.instC (keyProg c hU n) outs u).wire x') :
    Aliased.srcKey c x.val = Aliased.srcKey c x'.val := by
  have e := congrArg Rows.nIn (rows_circuitProg c hU n (circuitRows c hU).nIn
    (Aliased.keyIdx c (circuitRows c hU).nIn) u)
  have h1 : x.val < (circuitRows c hU).nIn := lt_of_lt_of_eq hx e
  have h2 : x'.val < (circuitRows c hU).nIn := lt_of_lt_of_eq hx' e
  rw [Const.Prog.instC_wire_input _ _ _ x hx, Const.Prog.instC_wire_input _ _ _ x' hx'] at hw
  have hv : (Const.Prog.insAt (keyProg c hU n) u x.val).val = (Const.Prog.insAt (keyProg c hU n) u x'.val).val :=
    congrArg Fin.val hw
  have e1 : (Const.Prog.insAt (keyProg c hU n) u x.val).val =
      (insOf n (circuitRows c hU).nIn (Aliased.keyIdx c (circuitRows c hU).nIn) u.val ⟨x.val, h1⟩).val :=
    insAt_uniform n u x.val h1
  have e2 : (Const.Prog.insAt (keyProg c hU n) u x'.val).val =
      (insOf n (circuitRows c hU).nIn (Aliased.keyIdx c (circuitRows c hU).nIn) u.val ⟨x'.val, h2⟩).val :=
    insAt_uniform n u x'.val h2
  exact Aliased.keyIdx_eq (n := (circuitRows c hU).nIn) (x := ⟨x.val, h1⟩) (x' := ⟨x'.val, h2⟩)
    ((insOf_eq_iff u.isLt ⟨x.val, h1⟩ ⟨x'.val, h2⟩).1 (Fin.ext (e1.symm.trans (hv.trans e2))))

section keyed

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {c : Flock.Circuit} (hU : Built c.unit) {si : Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)

include hparse in
/-- **W6 for the circuit's program, from the parse**: on any input gates, when every drawn unit's table accepted a
circuit file that parses to `c`. -/
theorem Sites.rowsOf_circuitProg (B : ℕ) (src : Fin (circuitRows c hU).nIn → Option (Fin B)) :
    si.RowsOf (circuitProg c hU n B src) :=
  Sites.rowsOf_uniform hU (insOf n B src) (oneGate n B) (insOf_ne_one n B src) si hparse

include hparse in
/-- W6 for `keyProg`, from the parse. -/
theorem Sites.rowsOf_keyProg : si.RowsOf (keyProg c hU n) :=
  Sites.rowsOf_circuitProg hU hparse _ _

variable (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n)))

/-- **`aliased` at each drawn unit's site, from Δ's copies**: in the accepted layout of the table the unit is drawn
into, the program's own instance of the unit (`Const.Prog.instC`) puts on one gate only inputs Δ copies one column
into, or the zero (`Aliased.setupH_aliased_of_key`). -/
theorem Sites.aliased_keyProg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (si.acc S R u h).layout.Aliased (si.acc S R u h).regs
      (Sites.rowsOf_keyProg hU hparse S R u h ▸ Const.Prog.instC (keyProg c hU n) outs u)
      (Fin.cast (si.acc S R u h).nbl_eq.symm (si.o S R u h)) (si.hvu S R u h) := by
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have e := congrArg Rows.nIn (Sites.rowsOf_keyProg hU hparse S R u h)
  refine Aliased.setupH_aliased_of_key (si.acc S R u h).ht (si.acc S R u h).h (si.acc S R u h).regs _ _
    (si.hvu S R u h) ?_
  intro x x' hw hx hx'
  have hw' := ((wire_cast (Sites.rowsOf_keyProg hU hparse S R u h) (Const.Prog.instC (keyProg c hU n) outs u) x).symm.trans
    hw).trans (wire_cast (Sites.rowsOf_keyProg hU hparse S R u h) (Const.Prog.instC (keyProg c hU n) outs u) x')
  have hk := keyProg_wire_key hU outs u _ _ (lt_of_lt_of_eq hx e.symm) (lt_of_lt_of_eq hx' e.symm) hw'
  exact (congrArg (fun d => Aliased.srcKey d x.val = Aliased.srcKey d x'.val) hc).mpr hk

/-- **`ProgPlaces` for the circuit's program on Δ's source keys**, from the sites and the parse alone: each drawn unit
placed at its template instance in its table's accepted statement (`placed`, W6 from the parse), its instance the
program's own (`Const.Prog.instC`), aliased as Δ copies (`Sites.aliased_keyProg`). -/
noncomputable def Sites.progPlaces_keyProg : ProgPlaces (keyProg c hU n) outs plan tab :=
  progPlaces_of_layout si (Sites.rowsOf_keyProg hU hparse) outs (Const.Prog.instC _ outs)
    (Sites.aliased_keyProg hU hparse outs)

/-- `progPlaces_keyProg`'s instance is the program's own. -/
theorem Sites.progPlaces_keyProg_inst (u : Fin n) :
    (Sites.progPlaces_keyProg hU hparse outs).inst u = Const.Prog.instC (keyProg c hU n) outs u :=
  rfl

end keyed

end FlockSoundness.Discharge.Placed
