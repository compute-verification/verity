import Proofs.Flock.Recursive.UniversalPack
import Proofs.Flock.Soundness.Types.ProgramPlaces

/-! # Recursion: a table class's slot computes its unit's type, and the universal unit's on 16-bit leaves

A `TableClass` places one unit's composed rows in every block, at each slot's columns. A witness that satisfies the
statement satisfies those rows (`rowsSat_of_satisfies`) with the block's `one` at 1 (`Placement.one_eq`), and composed
rows that hold are the unit's `blockRow`s held as `Types.Dag.SatAt`, so they carry the unit type's outputs
(`Rows.compose_eval_unit`). Hence:
- `placed_evalT`: wherever a placement puts a unit's composed rows, each output's logical column holds the unit type's
  output (`evalT`) on the input columns' values;
- `TableClass.slot_evalT`, `TableClass.slot_eval`: at a table class's slot, with each input read at its copy position,
  and for a flat unit type as `eval`;
- `TableClass.slot_uuW`: at `uuTypeW p`, output word `i`'s logical column holds `Desc.eval` of the description bits on
  the data bits, each read at its copy position.
-/

namespace FlockSoundness.Discharge.Recursion.UniversalSlot

open Flock Flock.Derive Flock.DeriveAll Audit Model FlockSoundness.Compose FlockSoundness.Types
  FlockSoundness.Discharge.Recursion.UniversalClass FlockSoundness.Discharge.Recursion.UniversalLower
  FlockSoundness.Discharge.Recursion.UniversalPack

/-- **Placed composed rows compute the unit's type.** If a witness satisfies a statement that places a unit's composed
rows in block `o` at the columns `col`, output `k`'s logical column there holds the unit type's output `k` on any `x`
that agrees with the input columns. -/
theorem placed_evalT {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}
    (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) {S : Statement}
    {col : Fin (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)).w → Fin (2 ^ S.kLog)}
    (hpl : Placement S (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)) col)
    {z : Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) {x : ℕ → Bool}
    (hx : ∀ i : Fin (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)).w,
      i.val < (done.getD unit default).inCols.size → x i.val = bit (S.block z o (col i)))
    {k : ℕ} (hk : k < t.outputs.size)
    (ℓ : Fin (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)).w)
    (hℓ : ℓ.val = lidx (ofBlock words done (done.getD unit default)) ((done.getD unit default).outCols.getD k 0)) :
    bit (S.block z o (col ℓ)) = (Dag.evalT types words (unit + 1) t x).getD k false :=
  Rows.compose_eval_unit hd ht (z := fun c => bit (S.block z o (col c))) (rowsSat_of_satisfies hpl hz o)
    (by show bit (S.block z o (col _)) = true; rw [hpl.one_eq hz o]; rfl) hx (unit + 1) (Nat.lt_succ_self unit) hk ℓ
    hℓ

namespace TableClass

variable {St : Statement} (c : FlockSoundness.TableClass St)

/-- **A slot computes the class's unit type**: in every block of a satisfying witness, output `k`'s logical column at
slot `g` holds the unit type's output `k` on the inputs read at their copy positions. -/
theorem slot_evalT {t : CircuitType} (ht : (c.ls.getD c.unit default).type.bind c.types = some t) (g : Fin c.slots)
    {z : Witness St.m} (hz : St.Satisfies z) (o : Fin (2 ^ St.nbl)) {k : ℕ} (hk : k < t.outputs.size)
    (ℓ : Fin c.rows.w)
    (hℓ : ℓ.val = lidx (ofBlock c.words c.done (c.done.getD c.unit default))
      ((c.done.getD c.unit default).outCols.getD k 0)) :
    bit (St.block z o (c.col g ℓ)) =
      (Dag.evalT c.types c.words (c.unit + 1) t fun i => bit (St.block z o (c.copyPos g i))).getD k false :=
  placed_evalT c.hd ht (c.placed g) hz o (fun i hi => congrArg bit (c.copy g i hi z hz o).symm) hk ℓ hℓ

/-- `slot_evalT` for a flat unit type, as `eval`. -/
theorem slot_eval {t : CircuitType} (ht : (c.ls.getD c.unit default).type.bind c.types = some t)
    (hflat : flatOk t = true) (g : Fin c.slots) {z : Witness St.m} (hz : St.Satisfies z) (o : Fin (2 ^ St.nbl))
    {k : ℕ} (hk : k < t.outputs.size) (ℓ : Fin c.rows.w)
    (hℓ : ℓ.val = lidx (ofBlock c.words c.done (c.done.getD c.unit default))
      ((c.done.getD c.unit default).outCols.getD k 0)) :
    bit (St.block z o (c.col g ℓ)) = (eval t fun i => bit (St.block z o (c.copyPos g i))).getD k false := by
  rw [slot_evalT c ht g hz o hk ℓ hℓ, Dag.evalT_flat c.types c.words hflat]

/-- **A slot of the universal unit's class computes the class's interpretation.** When the class's unit type is
`uuTypeW p`, in every block of a satisfying witness, output word `i`'s logical column at slot `g` holds `Desc.eval` of
description bit `j` read at copy position `j` on data bit `a` read at copy position `16·blockWords width + a`. -/
theorem slot_uuW {p : Params} (ht : (c.ls.getD c.unit default).type.bind c.types = some (uuTypeW p))
    (g : Fin c.slots) {z : Witness St.m} (hz : St.Satisfies z) (o : Fin (2 ^ St.nbl)) (i : Fin p.nOut)
    (ℓ : Fin c.rows.w)
    (hℓ : ℓ.val = lidx (ofBlock c.words c.done (c.done.getD c.unit default))
      ((c.done.getD c.unit default).outCols.getD (16 * i.val) 0)) :
    bit (St.block z o (c.col g ℓ)) =
      Desc.eval (fun j : Fin p.width => bit (St.block z o (c.copyPos g j)))
        (fun a => bit (St.block z o (c.copyPos g (16 * blockWords p.width + a)))) i := by
  rw [slot_evalT c ht g hz o (by rw [size_outputs_uuTypeW]; have := i.isLt; omega) ℓ hℓ]
  exact evalT_uuTypeW p c.types c.words c.unit _ i

end TableClass

end FlockSoundness.Discharge.Recursion.UniversalSlot
