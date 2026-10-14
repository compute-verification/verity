import Proofs.Flock.VBridge.Family.Names
import Proofs.Flock.Soundness.Discharge.Hm.Site
import Proofs.Flock.Soundness.Discharge.Layout.OutPin

/-!
# A part's `out` columns, from the verifier's own checks

`PartOutCols` (`Family/VCoins.lean`) reads a part circuit's returned-output columns, its units' output leaves and its
computed columns. None of them is in META: the columns are the unit's, which `RecResiduals.check` holds to the
verifier's `unitLaid` (`check_part`), and the leaves are `HmRow.check`'s leaf maps at a unit that returns the outputs
(`CheckFacts.leavesOut`, at `scopeOk`'s `outNet = unitNet`). So it needs no round trip of `circuitFile` through
`HmRow.parse`:

* `outCols_of_unitLaid`: the verifier's `rec-residuals` unit returns its `RES_OUT_WORDS` 16-bit outputs from
  `partBase`, past its inputs, in its computed columns (`outCols_of_laid`'s, at `unitLaid`);
* `partOutCols_of`: `PartOutCols`, stated in full, at any family.

The parts' `v` and row columns (`PartVCols`, `PartRowCols`, `PartRowBits`) are another matter: their keys
(`Aliased.srcKey`) read copy 0's leaf map and the ports' names and words, which only META's `leaves_in` and `ports`
give, and META is read by `Lean.Json.parse`, a `partial def` no proof unfolds.
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge

/-- **The verifier's `rec-residuals` unit returns its outputs from `partBase`**: its output group's first column, of
`RES_OUT_WORDS` ports, past the inputs, its bits within the computed columns. -/
theorem outCols_of_unitLaid {c : Flock.Circuit} (hU : Built c.unit) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit) :
    (c.unit.outGroups.getD 0 default).col * Flock.WORD = partBase S ns nv ∧
      (c.unit.outGroups.getD 0 default).bits.size = RecResiduals.RES_OUT_WORDS ∧
      (Rows.ofNet hU).nIn ≤ partBase S ns nv ∧
      ∀ t < 16 * RecResiduals.RES_OUT_WORDS, partBase S ns nv + t < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp := by
  obtain ⟨-, -, -, -, hcp, -, hog, -⟩ := Flock.Net.ofRows_ok hn
  have e : RecResiduals.unitLaid S ns nv = layout ((RecResiduals.unitNet' S ns nv).run St.empty).2 (partIn ns nv)
      (Array.replicate RecResiduals.RES_OUT_WORDS 16) ((RecResiduals.unitNet' S ns nv).run St.empty).1.toArray := rfl
  have hst : ((RecResiduals.unitNet' S ns nv).run St.empty).2 = (partRun S ns nv).2 := by rw [unitNet'_run]
  refine ⟨?_, ?_, ?_, fun t ht => ?_⟩
  · rw [hog, e, hst, layout_outGroups]
    exact outStart_word _ _ _
  · rw [hog, e, layout_outGroups]
    show (Array.replicate RecResiduals.RES_OUT_WORDS 16).size = _
    simp
  · rw [Rows.ofNet_nIn, inWords_unitLaid hn]
    exact (min_le_left _ _).trans (Decode.inRows_le_outStart _ _ _)
  · rw [show (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp = c.unit.constPos from Rows.ofNet_one hU, hcp, e, hst]
    exact outStart_add_lt _ _ _ _ (by rw [foldl_replicate]; omega)

end FlockVBridge.OfVStar

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockVBridge.OfVStar (partBase)

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **`PartOutCols`, from the verifier's checks** (`Family/VCoins.lean`'s statement, in full): part `p`'s unit is the
verifier's `unitLaid` (`check_part`, at `HUnitName`), which returns its outputs from `partBase` in its computed columns
(`outCols_of_unitLaid`), and each copy's output leaves are its output group's `RES_OUT_WORDS` ports
(`CheckFacts.leavesOut`, where `scopeOk` has the unit return the outputs). -/
theorem partOutCols_of (hne : Nonempty (Reg × L.Ω × Reg₂)) :
    ∀ (p : ℕ) (j : Fin F.stmts.size) (nm : String) (pt : VStar.Part), F.stmts[j] = .inner p nm pt →
      ∀ S ns nv, RecResiduals.part F.sh p = .ok (S, ns, nv) → ∀ cpy < (F.Vs j).c.upv,
        ∀ e < 16 * RecResiduals.RES_OUT_WORDS,
          partBase S ns nv + e = Layout.outCol (F.Vs j).c (e / 16) (e % 16) ∧
            e / 16 < ((F.Vs j).c.leavesOut.getD cpy #[]).size ∧ (Rows.ofNet (F.Vs j).hU).nIn ≤ partBase S ns nv ∧
            partBase S ns nv + e < (Rows.ofNet (F.Vs j).hU).nIn + (Rows.ofNet (F.Vs j).hU).nComp := by
  have hname := F.hUnitName_of hne
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  intro p j nm pt hj S ns nv hpart cpy hcpy e he
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname j p nm pt hj).trans (statements_name F.hstmts _ hj)
  have P := parse_facts (F.zs R ω [] R₂ j).hc
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := P.recResiduals
  obtain ⟨S', ns', nv', hpart', hn⟩ :=
    FlockVBridge.OfVStar.check_part hchk (resParams_of_check hchk hnm) F.hsh (F.hplain j)
  have hs := hpart.symm.trans hpart'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hs
  obtain ⟨rfl, rfl, rfl⟩ := hs
  obtain ⟨hcol, hsz, hlo, hhi⟩ := FlockVBridge.OfVStar.outCols_of_unitLaid (F.Vs j).hU hn
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have ho : (F.Vs j).c.outNet = (F.Vs j).c.unitNet := by
    have hs := (F.Vs j).hscope
    unfold Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  have hlt : cpy < (F.Vs j).c.leavesOut.size := by rw [C.leavesOut.1]; exact hcpy
  have hl := C.leavesOut.2 cpy hcpy ho
  rw [getElem!_pos (F.Vs j).c.leavesOut cpy hlt] at hl
  have hlv : ((F.Vs j).c.leavesOut.getD cpy #[]).size = RecResiduals.RES_OUT_WORDS := by
    rw [Array.getD_eq_getD_getElem?, getElem?_pos (F.Vs j).c.leavesOut cpy hlt, Option.getD_some, hl, hsz]
  refine ⟨?_, by rw [hlv]; omega, hlo, hhi e he⟩
  unfold Layout.outCol
  omega

end VStarFamily

end family

end FlockVBridge.Family

#print axioms FlockVBridge.OfVStar.outCols_of_unitLaid
#print axioms FlockVBridge.Family.VStarFamily.partOutCols_of
