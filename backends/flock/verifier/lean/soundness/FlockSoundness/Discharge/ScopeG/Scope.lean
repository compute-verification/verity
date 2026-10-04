import FlockSoundness.Discharge.ScopeG.Vu
import FlockSoundness.Discharge.ScopeW.Scope

set_option Elab.async false

/-!
# The scope widened to cut inputs and tails, and the VU's program there

`ScopeW.scopeOkW` (and `Layout.scopeOk`) admit a circuit whose unit has one input group (`inGroups.size == 1`) and
returns the outputs (`outNet == unitNet`), because `Placed.keyProg` runs the unit net alone. The VU's program
(`Vu.lean`) runs every column of the VU's units, stages and lookup slots, so neither is needed:

* `scopeOkGT`: the unit has its leaf group, perhaps cut inputs after it (`1 ≤ inGroups.size`; `HmRow.parse` refuses more
  than two), the outputs come from the unit or the tail's last stage, the rows have the four shapes `rowShapeW` admits,
  the circuit is untyped, and the VU's columns have an order (`orderedB`, `Nodup`) in which every input a row reads is a
  forced zero or a row's message bit (`vuReadsB`);
* `scopeOkG`: `scopeOkGT` with the unit returning the outputs (`outNet == unitNet`): no tail out net.

`vuProg c hs n` is `n` units of the VU's composed rows (`vuRows`), each VU input on the input gate of its Δ source
(`vuKeyIdx`, `Aliased.srcKey`'s keys, as `Placed.keyProg`'s), and `vuOutsOf` the gates on the returned outputs' columns.
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3 Compose

/-- **The circuits the VU's program covers, tails that return the outputs included.** -/
def scopeOkGT (c : Circuit) : Bool :=
  decide (1 ≤ c.unit.inGroups.size) && decide (c.unit.unitLog ≤ 32) && c.typed.isNone &&
    c.ports.all ScopeW.rowShapeW && orderedB (vuLogical c) && decide (vuOrder c).Nodup && vuReadsB c

/-- **The circuits the VU's program covers whose unit returns the outputs.** -/
def scopeOkG (c : Circuit) : Bool := scopeOkGT c && c.outNet == c.unitNet

theorem scopeOkG_GT {c : Circuit} (h : scopeOkG c = true) : scopeOkGT c = true := by
  simp only [scopeOkG, Bool.and_eq_true] at h
  exact h.1

theorem scopeOkG_outNet {c : Circuit} (h : scopeOkG c = true) : c.outNet = c.unitNet := by
  simp only [scopeOkG, Bool.and_eq_true, beq_iff_eq] at h
  exact h.2

/-- **What `scopeOkGT` gives**: the leaf group, the unit's slot size, an untyped circuit, the four row shapes, the VU's
order, its columns once each, and the inputs its rows read. -/
theorem scopeOkGT_spec {c : Circuit} (h : scopeOkGT c = true) :
    1 ≤ c.unit.inGroups.size ∧ c.unit.unitLog ≤ 32 ∧ c.typed = none ∧ (∀ p ∈ c.ports.toList, ScopeW.rowShapeW p = true) ∧
      Ordered (vuLogical c) ∧ (vuOrder c).Nodup ∧ vuReadsB c = true := by
  simp only [scopeOkGT, Bool.and_eq_true, decide_eq_true_eq, Option.isNone_iff_eq_none, Array.all_eq_true_iff_forall_mem,
    ← Array.mem_toList_iff] at h
  obtain ⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩ := h
  exact ⟨h1, h2, h3, h4, ordered_of_orderedB h5, h6, h7⟩

theorem scopeOkGT_ordered {c : Circuit} (h : scopeOkGT c = true) : Ordered (vuLogical c) := (scopeOkGT_spec h).2.2.2.2.1

/-! ## The VU's program -/

variable (c : Circuit) (hs : scopeOkGT c = true) (n : ℕ)

/-- **A VU's rows**: the VU's columns in `vuOrder`, its inputs first, each computed row its block row
(`Rows.compose`). -/
abbrev vuRows : Rows := Rows.compose (vuLogical c) (scopeOkGT_ordered hs)

/-- The VU's inputs. -/
abbrev vuB : ℕ := (vuRows c hs).nIn

/-- VU input `s`'s Δ source (`Aliased.srcKey` at its VU-stacked unit input). -/
def vuKey (s : ℕ) : Aliased.Src := Aliased.srcKey c ((vuIns c).getD s 0)

/-- **The VU inputs' keys as codes**: none at the zero, else the first VU input with the same key
(`Aliased.keyIdx` at `vuKey`). -/
def vuKeyIdx (B : ℕ) (b : Fin B) : Option (Fin B) :=
  if vuKey c b.val = .zero then none else (List.finRange B).find? fun i => vuKey c i.val = vuKey c b.val

/-- The gates of the VU's program. -/
abbrev vuN : ℕ := Placed.gates (Placed.inGates n (vuB c hs)) (vuRows c hs) n

/-- **The program of a circuit's VUs**: `n` units of the VU's rows, VU input `b` of each on its Δ source's gate
(`Placed.insOf` at `vuKeyIdx`), the constant on `Placed.oneGate`: `Placed.circuitProg` at `vuRows`. -/
def vuProg : Prog (vuN c hs n) n :=
  Placed.uniform (Placed.inGates n (vuB c hs)) (vuRows c hs) (Placed.insOf n (vuB c hs) (vuKeyIdx c (vuB c hs)))
    (Placed.oneGate n (vuB c hs)) (Placed.insOf_ne_one n (vuB c hs) (vuKeyIdx c (vuB c hs))) n

/-- Every unit of the VU's program has the VU's rows. -/
theorem rows_vuProg (u : Fin n) : (vuProg c hs n).rows u = vuRows c hs :=
  Placed.rows_uniform _ _ _ _ _ n u

/-- The constant's gate of the VU's program. -/
def vuOneG : Fin (vuN c hs n) := Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (vuB c hs))

/-- The zero's gate of the VU's program. -/
def vuZeroG : Fin (vuN c hs n) := Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (vuB c hs))

/-- **The returned outputs' columns**, at VU 0: output word `o`'s 16 bits from `HmOut.outCol c 0 o`, the unit's
returned output or the tail's last stage's output port. -/
def vuOutCols : List ℕ :=
  (List.range c.outWords).flatMap fun o => (List.range 16).map (HmOut.outCol c 0 o + ·)

/-- **The returned outputs' bits**: the gates the units' instances put on returned outputs' columns
(`Layout.outsOf` at the VU's program). -/
noncomputable def vuOutsOf : Finset (Fin (vuN c hs n)) :=
  @Finset.filter _ (fun g => ∃ u x, (Const.Prog.rowsWireC (vuProg c hs n) u).wire x = g ∧
    x.val < (vuLogical c).order.length ∧ (vuLogical c).order.getD x.val 0 ∈ vuOutCols c) (Classical.decPred _)
    Finset.univ

end FlockSoundness.Discharge.ScopeG
