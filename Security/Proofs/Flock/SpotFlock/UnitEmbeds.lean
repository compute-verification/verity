module

public import Proofs.Flock.SpotFlock.Rows
public import Proofs.Flock.SpotFlock.UnitRows

@[expose] public section

/-!
# A placed unit, read off its session's block circuit

`valueAt_unitPlace`: at a column of a unit placed in table `j` of a parsed statement at block `o` (`unitPlace`), under
a witness of the deployed relation, the block circuit's value at that column's input gate (`valueAt_relation`) is the
value the unit's block decodes at the column's wire.

`embeds_unitPlace`: so at any wire map that sends each wire the unit computes or reads to the input gate of one of its
columns, the block circuit's values hold each of the unit's gates (`UnitPlace.apply_decode`) and carry the lowering's
constants (`decode_consts`). That is what `Embeds` asks of a checked unit (`unitVals`), where the layout's wires are
such a map.
-/

namespace SpotFlock

open FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

variable {c : _root_.Flock.Circuit} (hU : Built c.unit) {n : ℕ}
  (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))
  {T : Model.Statement} (a : HiddenExec.AcceptedH T) (hparse : HmOut.unrolled a.st.c = c)

include hparse in
/-- **The session's value at a placed unit's column is the unit's decoded value**: for a unit placed in table `j` of
a parsed statement at block `o` (`unitPlace`), under a witness of the deployed relation whose table-`j` share the
accepted statement's block R1CS holds, the block circuit's value at the input gate of the unit's column `cc` is the
value the unit's block decodes at `cc`'s wire. -/
theorem valueAt_unitPlace {p : Parsed} {j : ℕ} (hj : j < p.sts.size) (hk : T.kLog = p.sts[j].c.kLog) (u : Fin n)
    {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) {w : Claims.Flock.Wit}
    (hR : Claims.Flock.RelS (toReadable relation p) w) (hz : T.Satisfies (zOf p.sts.size w j T.m))
    (cc : Fin (unitPlace hU outs a hparse u hg o).R.w) :
    valueAt (toReadable relation p) o.val w
        (j * 2 ^ (toReadable relation p).kLog + ((unitPlace hU outs a hparse u hg o).col cc).val) =
      (unitPlace hU outs a hparse u hg o).decode (zOf p.sts.size w j T.m)
        ((unitPlace hU outs a hparse u hg o).inst.wire cc) := by
  obtain ⟨hfit, -⟩ := (relS_relation p w).1 hR j hj
  have hk0 : p.sts[j]!.c.kLog = (toReadable relation p).kLog := hfit.2.1
  have hk' : T.kLog = p.sts[j]!.c.kLog := by rw [hk, getElem!_pos p.sts j hj]
  have hcol : ((unitPlace hU outs a hparse u hg o).col cc).val < 2 ^ (toReadable relation p).kLog :=
    lt_of_lt_of_eq ((unitPlace hU outs a hparse u hg o).col cc).isLt (by rw [hk', hk0])
  rw [valueAt_relation o.val w hj hcol]
  show _ = (unitPlace hU outs a hparse u hg o).inst.decode _ ((unitPlace hU outs a hparse u hg o).inst.wire cc)
  rw [(unitPlace hU outs a hparse u hg o).inst.decode_wire ((unitPlace hU outs a hparse u hg o).agree hz) cc]
  have hb := bit_block_zOf p w j hk' o ((unitPlace hU outs a hparse u hg o).col cc)
  unfold Flock.InnerRows.bitAt at hb
  rw [hk0] at hb
  exact hb.symm

include hparse in
/-- **A placed unit's gates hold at its session's values**: for a unit placed in table `j` of a parsed statement at
block `o`, under a witness of the deployed relation whose table-`j` share the accepted statement's block R1CS holds,
and a wire map `W` that sends each wire the unit computes or reads to the input gate of one of its columns, the block
circuit's values at `W` hold each of the unit's gates and carry the lowering's constants (`Bridge.constsOf`) wherever
the unit reads them. -/
theorem embeds_unitPlace {p : Parsed} {j : ℕ} (hj : j < p.sts.size) (hk : T.kLog = p.sts[j].c.kLog) (u : Fin n)
    {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl)) {w : Claims.Flock.Wit}
    (hR : Claims.Flock.RelS (toReadable relation p) w) (hz : T.Satisfies (zOf p.sts.size w j T.m))
    (W : Fin (Discharge.Layout.keyN c hU n) → ℕ)
    (hW : ∀ x, (((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit x = some u ∨
        ∃ g', ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit g' = some u ∧
          x ∈ (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op g').args) →
      ∃ cc, (unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).inst.wire cc = x ∧
        W x = j * 2 ^ (toReadable relation p).kLog +
          ((unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).col cc).val) :
    (∀ x, ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit x = some u →
      valueAt (toReadable relation p) o.val w (W x) =
        (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op x).apply
          fun y => valueAt (toReadable relation p) o.val w (W y)) ∧
    ∀ x b, Bridge.constsOf c hU n x = some b →
      (∃ g', ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit g' = some u ∧
        x ∈ (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op g').args) →
      valueAt (toReadable relation p) o.val w (W x) = b := by
  have hval : ∀ x, (((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit x = some u ∨
      ∃ g', ((Placed.keyProg c hU n).partition (Discharge.Layout.outsOf c hU n)).unit g' = some u ∧
        x ∈ (((Placed.keyProg c hU n).circuit (Discharge.Layout.outsOf c hU n)).op g').args) →
      valueAt (toReadable relation p) o.val w (W x) =
        (unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).decode (zOf p.sts.size w j T.m) x := by
    intro x hx
    obtain ⟨cc, rfl, hWx⟩ := hW x hx
    rw [hWx]
    exact valueAt_unitPlace hU _ a hparse hj hk u hg o hR hz cc
  refine ⟨fun x hu => ?_, fun x b hb hr => ?_⟩
  · rw [hval x (Or.inl hu), (unitPlace hU (Discharge.Layout.outsOf c hU n) a hparse u hg o).apply_decode hz x hu]
    exact Audit.Op.apply_congr _ fun y hy => (hval y (Or.inr ⟨x, hu, hy⟩)).symm
  · rw [hval x (Or.inr hr)]
    exact decode_consts hU a hparse u hg o hz x b hb hr

end SpotFlock
