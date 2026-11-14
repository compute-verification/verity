module

public import Proofs.Accounting.ReadUnit.Deployed
public import Proofs.Accounting.ReadUnit.Guarantees
public import Proofs.Core.Game.Expect

/-!
# A table read the read's template check accepts binds

`Accounting.SecurityProofs.TableReadBinding` proves `Accounting.Guarantees.TableReadBinding` from `readUnitDrawn_check`:
the check's run puts the read's input `x` on column `x` and output `j` on `Layout.outCol`'s column, so a read at the
template's columns (`portsAt`) is a read at the run's ports (`portsAt_eq`, `readPorts`). SHA-512's collision resistance
at the reads' finder is the named assumption at its one coin (`expectedCR_of_draw`).

`gatesRefine_check` is the gate fact under it on its own: at every copy of a unit of a parsed templated statement whose
template the check accepts and whose unit returns the circuit's outputs, with the unit's constant registered as `true`,
the gates at the template's columns refine `Hash.EmbeddingRead` (`ReadUnit.GatesRefine`). A statement that takes
`GatesRefine` as a premise has it from this theorem where its premise is at that file's partition
(`Flock.Bridge.partsOf` of `.tmpl ct`), at those ports (`portsAt`), at SHA-512 and at the pinned hm96 key.
-/

@[expose] public section

namespace Accounting.ReadUnit

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Gates

theorem colAt_of_val {R : FlockSoundness.Rows} {x : ℕ} {c : Fin R.w} (h : c.val = x) : colAt R x = c := by
  subst h
  simp [colAt, c.isLt]

/-- **The template's columns are the run's**: where the run puts input `x` on column `x` (`hφ`) and output `j` on
`Layout.outCol`'s column (`hout`), a read at copy `j`'s template columns is the read at the run's ports. -/
theorem portsAt_eq (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (u : Fin n) {j : ℕ}
    (hj : j < D.unroll.upv) (V H : ℕ) {st : St} {outs : List Lin}
    (hr : HoldsRead (FlockSoundness.Rows.ofNet hU) (readIn H (Hash.Tree.depth V)) st outs)
    (hφ : ∀ x < readIn H (Hash.Tree.depth V), (hr.φ x).val = x)
    (hout : ∀ k < 16 * H + 1,
      (hr.out k).val = FlockSoundness.Discharge.Layout.outCol D.unroll (k / 16) (k % 16)) :
    portsAt D hU n u hj V H = readPorts (Flock.Bridge.copyGate D hU n u hj) hr V H := by
  have hin : ∀ x < readIn H (Hash.Tree.depth V),
      colAt (FlockSoundness.Rows.ofNet hU) x = hr.φ x := fun x hx => colAt_of_val (hφ x hx)
  unfold portsAt readPorts
  simp only [Flock.ReadNet.ROOT_AT, Flock.ReadNet.DOMAIN_AT, Flock.ReadNet.ROW_AT, Flock.ReadNet.saltAt,
    Flock.ReadNet.pathAt]
  congr 1
  · funext i
    rw [hin i (by have := i.isLt; unfold readIn; omega)]
  · funext i
    rw [hin (32 + i) (by have := i.isLt; unfold readIn; omega)]
  · funext i
    rw [hin (544 + i) (by have := i.isLt; unfold readIn; omega)]
  · funext i
    rw [hin (1056 + i) (by have := i.isLt; unfold readIn; omega)]
  · funext i
    rw [hin (1056 + 16 * H + i) (by have : i.val < 1536 := i.isLt; unfold readIn; omega)]
  · funext k i
    rw [hin (2592 + 16 * H + 512 * k + i) (by have := k.isLt; have := i.isLt; unfold readIn; omega)]
  · funext i
    rw [colAt_of_val (hout i (by have := i.isLt; omega))]
  · have h := hout (16 * H) (by omega)
    rw [Nat.mul_div_cancel_left H (by norm_num : 0 < 16), Nat.mul_mod_right] at h
    rw [colAt_of_val h]

/-- **The read's gates refine its reference at every copy of a unit the read's check accepts.** Let the templated
statement parse into `ct` (`hparse`, `Flock.HmOut.parseTmpl`), with its unit template passing the read's check
(`hcheck`, `Flock.ReadNet.check V H`) and its unit returning the circuit's outputs (`ho`). Then in the key program of
`n` units of `ct`'s circuit, at unit `u` with its constant registered as `true` (`hone`) and at each copy `j` of its
stack, the gates at the read template's columns (`portsAt`) refine `Hash.EmbeddingRead` for a table of `V` rows of `H`
words, at SHA-512 under the pinned hm96 key (`GatesRefine`). That is: at every transcript where `u` is correct and the
registered constants among `u`'s inputs and outputs hold, in `u`'s local evaluation the row the gates return is the
reference's on their values and their verdict is set only where the reference accepts the opening; and the verdict's
gate is committed. The root's and the domain id's gates are the copy's inputs, outside `u`, so `GatesRefine` asks
nothing of them here.

It names no probability, no acceptance and no hash's collision resistance. It is the gate premise
`Accounting.SecurityProofs.readUnitDrawnCR` takes of each read, which `readUnitDrawn_check` discharges inside its own
proof, here on its own and from the same parts (`unrolled_leaves_of_parse`, `read_of_check`, `gatesRefine_layout`),
stated at the template's columns through `portsAt_eq`. -/
theorem gatesRefine_check {V H : ℕ} {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {ct : Flock.Circuit} {tu : Flock.TmplUnit}
    (hparse : Flock.HmOut.parseTmpl tags bytes tables = .ok ct) (ht : ct.tmpl = some tu)
    (hcheck : Flock.ReadNet.check V H tu.t = true) (ho : ct.outNet = ct.unitNet)
    (hB : FlockSoundness.Built (Flock.HmOut.unrolled ct).unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (hone : consts (Flock.Bridge.constGate (.tmpl ct) hB n u) = some true) {j : ℕ}
    (hj : j < (Flock.HmOut.unrolled ct).upv) :
    GatesRefine Verity.sha512 Verity.sha512 Flock.Hm96.Default512.key.data.toList consts
      (Flock.Bridge.partsOf (.tmpl ct) hB n) u (portsAt (.tmpl ct) hB n u hj V H) := by
  have hsz : ct.unitNet < ct.nets.size := by
    obtain ⟨_, _, P, -, rfl, -, -⟩ := FlockSoundness.TmplValue.parseTmpl_facts hparse
    exact P.unitNet
  obtain ⟨hupv, hleaves⟩ := unrolled_leaves_of_parse hparse ho
  obtain ⟨run, hr, hsound, hrow, hφ, hout⟩ := read_of_check ht hsz hcheck hB
  have hlen : 16 * H < (run.1.1 ++ [run.1.2]).length := by
    simp only [List.length_append, List.length_singleton, hrow]
    omega
  have hH : H < ((Flock.HmOut.unrolled ct).leavesOut.getD j #[]).size := by
    rw [hleaves j hj, Gates.unrolled_outPorts ht hsz hcheck]
    omega
  have hv : (hr.out (16 * H)).val = FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) H 0 := by
    have h := hout (16 * H) (by omega)
    rwa [Nat.mul_div_cancel_left H (by norm_num : 0 < 16), Nat.mul_mod_right] at h
  rw [portsAt_eq (.tmpl ct) hB n u hj V H hr hφ hout]
  exact gatesRefine_layout pinnedKey (.tmpl ct) hB n hr hsound hupv hlen hv consts u hone hj hH

end Accounting.ReadUnit

namespace Claims

/-- **The named assumption at a finder's one coin is `ExpectedCR` there**: a finder is the game of one uniform coin over
its coins, played by its one strategy. -/
theorem expectedCR_of_draw {H : Msg → Msg} (F : Finder)
    (h : FlockSoundness.Assumptions.SHA512CRExpected H (Verity.Game.draw F.S) (fun _ => PUnit.unit) F.out
      fun s => (F.cost s : ℝ)) : ExpectedCR H F := by
  classical
  unfold ExpectedCR Finder.Pr
  have e : Claims.Pr (fun s : F.S => ∃ u v, F.out s = some (u, v) ∧ Collision H u v) =
      Verity.Game.expect (Verity.Game.ind fun s => ∃ x y, F.out s = some (x, y) ∧ x ≠ y ∧ H x = H y)
        (Verity.Game.draw F.S) (fun _ => PUnit.unit) := by
    show ((Finset.univ.filter _).card : ℝ) / Fintype.card F.S = (∑ c : F.S, Verity.Game.ind _ c) / _
    congr 1
    simp only [Verity.Game.ind]
    rw [Finset.sum_boole]
    rfl
  exact (e.trans (Verity.Game.prob_toReal _ (Verity.Game.draw F.S) (fun _ => PUnit.unit)).symm).trans_le h

end Claims

namespace Accounting.SecurityProofs

open Accounting.ReadUnit

theorem TableReadBinding : Accounting.Guarantees.TableReadBinding := by
  intro V H t hcheck tags bytes tables ct hparse tu ht htu ho
  subst htu
  intro n consts hone ι R hR Ω _ _ acc drawn tr cost ε hcom below hcr
  obtain ⟨run, hr, hφ, hout, hbind⟩ :=
    Gates.readUnitDrawn_check (Ω := Ω) hparse ht hcheck ho (FlockSoundness.unrolled_built hparse) n
  refine hbind consts hone R ?_ acc drawn tr cost ε hcom below (Claims.expectedCR_of_draw _ hcr)
  intro i hi
  obtain ⟨j, hj, hp⟩ := hR i hi
  exact ⟨j, hj, hp.trans
    (portsAt_eq (.tmpl ct) (FlockSoundness.unrolled_built hparse) n (R.unit i) hj V H hr hφ hout)⟩

end Accounting.SecurityProofs
