import Proofs.Flock.Soundness.Discharge.Layout.Outs
import Proofs.Flock.Soundness.Discharge.Layout.KeyProgP

set_option Elab.async false

/-!
# The returned outputs' bits as the program's outputs, on the keys with public inputs

`Outs.lean` at `Placed.keyProgP c pi hU n`, whose units read the gates of their keys with public inputs:

* `OutsOutP`: `OutsOut` at `keyProgP`, every committed gate a unit computes on a returned output's column;
* `keyProgP_reads`: `keyProgP`'s units read only input gates besides their own;
* `outsOfP`, `outsOutP_outsOfP`: the gates the units' instances put on returned outputs' columns, at which `OutsOutP`
  holds (`computed_committed_output`).
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Audit Placed

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports)

/-- **The committed gates a unit of `keyProgP` computes are returned outputs' bits** (`OutsOut` at `keyProgP`). -/
def OutsOutP (outs : Finset (Fin (keyN c hU n))) : Prop :=
  ∀ u g, g ∈ ((keyProgP c pi hU n).partition outs).io u → (keyProgP c pi hU n).unit g = some u →
    ∀ x : Fin ((keyProgP c pi hU n).rows u).w, (Const.Prog.instC (keyProgP c pi hU n) outs u).wire x = g →
      ∃ u' j t, OutColAt c hU x.val u' j t

/-- **`keyProgP`'s units read only input gates besides their own**, as `keyProg_reads`. -/
theorem keyProgP_reads (outs : Finset (Fin (keyN c hU n))) (u : Fin n) (y : Fin ((keyProgP c pi hU n).rows u).w)
    (hy : ∀ i, y ≠ ((keyProgP c pi hU n).rows u).comp i) :
    ((keyProgP c pi hU n).partition outs).unit ((Const.Prog.instC (keyProgP c pi hU n) outs u).wire y) = none := by
  show (keyProgP c pi hU n).unit _ = none
  refine unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
    (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) n _ ?_
  by_cases hx : y.val < ((keyProgP c pi hU n).rows u).nIn
  · rw [Const.Prog.instC_wire_input _ _ _ y hx]
    have e := congrArg FlockSoundness.Rows.nIn
      (rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u)
    have hxB : y.val < keyB c hU := lt_of_lt_of_eq hx e
    have hv : (Const.Prog.insAt (keyProgP c pi hU n) u y.val).val =
        (Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u.val ⟨y.val, hxB⟩).val :=
      Placed.insAt_uniform n u y.val hxB
    rw [hv]
    exact Fin.isLt _
  · have hone : y = ((keyProgP c pi hU n).rows u).one := by
      apply Fin.ext
      have hw : y.val < ((keyProgP c pi hU n).rows u).nIn + ((keyProgP c pi hU n).rows u).nComp + 1 := y.isLt
      by_contra hne
      have hi : y.val - ((keyProgP c pi hU n).rows u).nIn < ((keyProgP c pi hU n).rows u).nComp := by
        simp only [FlockSoundness.Rows.one] at hne
        omega
      exact hy ⟨_, hi⟩ (Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega))
    rw [hone, Const.Prog.instC_wire_one]
    have h0 : (Const.Prog.oneAt (keyProgP c pi hU n) u).val = (Placed.oneGate n (keyB c hU)).val :=
      oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
        (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU))) n u
    rw [h0]
    exact Fin.isLt _

/-- **The returned outputs' bits of `keyProgP`**: the gates the units' instances put on returned outputs' columns
(`OutColAt`). -/
noncomputable def outsOfP : Finset (Fin (keyN c hU n)) :=
  @Finset.filter _ (fun g => ∃ u x, (Const.Prog.rowsWireC (keyProgP c pi hU n) u).wire x = g ∧
    ∃ u' j t, OutColAt c hU x.val u' j t) (Classical.decPred _) Finset.univ

/-- **`OutsOutP` at the returned outputs' bits**, as `outsOut_outsOf`. -/
theorem outsOutP_outsOfP : OutsOutP c hU n pi (outsOfP c hU n pi) := by
  intro u g hg hgu x hx
  have hmem : g ∈ outsOfP c hU n pi :=
    computed_committed_output (fun u => Const.Prog.instC (keyProgP c pi hU n) (outsOfP c hU n pi) u)
      (fun u y hy => keyProgP_reads c hU n pi _ u y hy) hg hgu
  obtain ⟨-, u₁, x₁, hw₁, u', j, t, hoc⟩ := (@Finset.mem_filter _ _ (Classical.decPred _) _ _).1 hmem
  have eR := rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdxP c pi (keyB c hU)) u₁
  have eN : ((keyProgP c pi hU n).rows u₁).nIn = c.upv * (Rows.ofNet hU).nIn := by
    rw [congrArg FlockSoundness.Rows.nIn eR]; rfl
  have eC : ((keyProgP c pi hU n).rows u₁).nComp = c.upv * ((Rows.ofNet hU).nComp + 1) := by
    rw [congrArg FlockSoundness.Rows.nComp eR]; rfl
  obtain ⟨hu', hj, ht, h1, h2, hxv⟩ := hoc
  have hlt := Rows.mul_add_lt hu' (show 1 + (outCol c j t - (Rows.ofNet hU).nIn) < (Rows.ofNet hU).nComp + 1 by omega)
  have hi : x₁.val - ((keyProgP c pi hU n).rows u₁).nIn < ((keyProgP c pi hU n).rows u₁).nComp := by
    rw [eN, eC]; omega
  have hx₁ : x₁ = ((keyProgP c pi hU n).rows u₁).comp ⟨_, hi⟩ :=
    Fin.ext (by simp only [FlockSoundness.Rows.comp]; rw [eN] at hi ⊢; omega)
  have hu₁ : u₁ = u := by
    have hm := (Const.Prog.instC (keyProgP c pi hU n) (outsOfP c hU n pi) u₁).mem ⟨_, hi⟩
    rw [← hx₁] at hm
    have hm' : (keyProgP c pi hU n).unit g = some u₁ := hw₁ ▸ hm
    rw [hgu] at hm'
    exact (Option.some.inj hm').symm
  subst hu₁
  rcases (Const.Prog.instC (keyProgP c pi hU n) (outsOfP c hU n pi) u₁).shared x x₁ (hx.trans hw₁.symm) with
    he | ⟨-, h2'⟩
  · exact ⟨u', j, t, hu', hj, ht, h1, h2, by rw [he]; exact hxv⟩
  · rw [eN] at h2'
    omega

end FlockSoundness.Discharge.Layout
