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
