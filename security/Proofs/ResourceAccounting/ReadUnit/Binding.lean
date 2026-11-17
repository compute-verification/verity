module

public import Proofs.ResourceAccounting.ReadUnit.Deployed
public import Proofs.ResourceAccounting.ReadUnit.Guarantees
public import Proofs.Core.Game.Expect

/-! `TableReadBinding` from `readUnitSampled_check`, at the read template's columns. -/

@[expose] public section

namespace ResourceAccounting.ReadUnit

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Gates

theorem colAt_of_val {R : FlockSoundness.Rows} {x : ℕ} {c : Fin R.w} (h : c.val = x) : colAt R x = c := by
  subst h
  simp [colAt, c.isLt]

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

/-- The root's and the domain id's gates are the copy's inputs, outside `u`, so `GatesRefine` asks nothing of them
here. It names no probability, no acceptance and no hash's collision resistance. -/
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

end ResourceAccounting.ReadUnit

namespace Claims

theorem expectedCR_of_sample {H : Msg → Msg} (F : Finder)
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

namespace ResourceAccounting.SecurityProofs

open ResourceAccounting.ReadUnit

theorem TableReadBinding : ResourceAccounting.Guarantees.TableReadBinding := by
  intro V H t hcheck tags bytes tables ct hparse tu ht htu ho
  subst htu
  intro n consts hone ι R hR Ω _ _ acc sampled tr cost ε hcom below hcr
  obtain ⟨run, hr, hφ, hout, hbind⟩ :=
    Gates.readUnitSampled_check (Ω := Ω) hparse ht hcheck ho (FlockSoundness.unrolled_built hparse) n
  refine hbind consts hone R ?_ acc sampled tr cost ε hcom below (Claims.expectedCR_of_sample _ hcr)
  intro i hi
  obtain ⟨j, hj, hp⟩ := hR i hi
  exact ⟨j, hj, hp.trans
    (portsAt_eq (.tmpl ct) (FlockSoundness.unrolled_built hparse) n (R.unit i) hj V H hr hφ hout)⟩

end ResourceAccounting.SecurityProofs
