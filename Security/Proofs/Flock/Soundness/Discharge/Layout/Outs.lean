import Proofs.Flock.Soundness.Discharge.Layout.OutPin
import Properties.Flock.Circuit.Outs

set_option Elab.async false

/-!
# The returned outputs' bits as the program's outputs

`keyProg_hout` takes `OutsOut`, a condition on the program's outputs `outs`: every committed gate a unit computes is on a
returned output's column. `outsOf` is the outputs that make it hold: the gates the units' instances put on returned
outputs' columns (`outsOut_outsOf`). It holds because a committed gate a unit computes is an output when the units read
only input gates besides their own (`computed_committed_output`), which `keyProg`'s units do (`keyProg_reads`), and a
unit's instance puts distinct computed columns on distinct gates (`IsRowsUnit.shared`).
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Audit Placed

/-- **A committed gate a unit computes is an output**, when every unit reads, besides its own computed gates, only input
gates. -/
theorem computed_committed_output {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {R : Fin n → Rows}
    (I : ∀ u, P.IsRowsUnit u (R u))
    (hread : ∀ u (y : Fin (R u).w), (∀ i, y ≠ (R u).comp i) → P.unit ((I u).wire y) = none)
    {u : Fin n} {g : Fin C.N} (hg : g ∈ P.io u) (hgu : P.unit g = some u) : g ∈ C.outputs := by
  classical
  simp only [Audit.Partition.io, Finset.mem_filter] at hg
  have hc := hg.1
  simp only [Audit.Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and] at hc
  rcases hc with hin | hout | ⟨h, hgh, hne⟩
  · rw [← P.input_iff] at hin
    rw [hin] at hgu
    cases hgu
  · exact hout
  · exfalso
    rcases hh : P.unit h with _ | u'
    · have hi := (P.input_iff h).1 hh
      cases hop : C.op h <;> simp only [hop, Audit.Op.isInput, Audit.Op.args] at hi hgh <;>
        first | exact absurd hi (by decide) | exact absurd hgh (List.not_mem_nil)
    · obtain ⟨i, rfl⟩ := (I u').onto h hh
      rw [(I u').op i] at hgh
      simp only [Audit.Op.args, List.mem_append, List.mem_map] at hgh
      obtain ⟨y, -, rfl⟩ : ∃ y, y ∈ (R u').a i ++ (R u').b i ∧ (I u').wire y = g := by
        rcases hgh with ⟨y, hy, e⟩ | ⟨y, hy, e⟩
        · exact ⟨y, List.mem_append_left _ hy, e⟩
        · exact ⟨y, List.mem_append_right _ hy, e⟩
      by_cases hcomp : ∃ j, y = (R u').comp j
      · obtain ⟨j, rfl⟩ := hcomp
        exact hne (by rw [hh, (I u').mem j])
      · rw [hread u' y fun j e => hcomp ⟨j, e⟩] at hgu
        cases hgu

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- **`keyProg`'s units read only input gates besides their own**: a column of a unit's instance that is no computed
column, an input or the constant, is on a program input. -/
theorem keyProg_reads (outs : Finset (Fin (keyN c hU n))) (u : Fin n) (y : Fin ((keyProg c hU n).rows u).w)
    (hy : ∀ i, y ≠ ((keyProg c hU n).rows u).comp i) :
    ((keyProg c hU n).partition outs).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire y) = none := by
  show (keyProg c hU n).unit _ = none
  refine unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
    (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n _ ?_
  by_cases hx : y.val < ((keyProg c hU n).rows u).nIn
  · rw [Const.Prog.instC_wire_input _ _ _ y hx]
    have e := congrArg FlockSoundness.Rows.nIn
      (rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u)
    have hxB : y.val < keyB c hU := lt_of_lt_of_eq hx e
    have hv : (Const.Prog.insAt (keyProg c hU n) u y.val).val =
        (Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u.val ⟨y.val, hxB⟩).val :=
      Placed.insAt_uniform n u y.val hxB
    rw [hv]
    exact Fin.isLt _
  · have hone : y = ((keyProg c hU n).rows u).one := by
      apply Fin.ext
      have hw : y.val < ((keyProg c hU n).rows u).nIn + ((keyProg c hU n).rows u).nComp + 1 := y.isLt
      by_contra hne
      have hi : y.val - ((keyProg c hU n).rows u).nIn < ((keyProg c hU n).rows u).nComp := by
        simp only [FlockSoundness.Rows.one] at hne
        omega
      exact hy ⟨_, hi⟩ (Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega))
    rw [hone, Const.Prog.instC_wire_one]
    have h0 : (Const.Prog.oneAt (keyProg c hU n) u).val = (Placed.oneGate n (keyB c hU)).val :=
      oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
        (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
        (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n u
    rw [h0]
    exact Fin.isLt _

/-- **`OutsOut` at the returned outputs' bits**: every committed gate a unit computes is on a returned output's column of
the unit's instance. -/
theorem outsOut_outsOf : OutsOut c hU n (outsOf c hU n) := by
  intro u g hg hgu x hx
  have hmem : g ∈ outsOf c hU n :=
    computed_committed_output (fun u => Const.Prog.instC (keyProg c hU n) (outsOf c hU n) u)
      (fun u y hy => keyProg_reads c hU n _ u y hy) hg hgu
  obtain ⟨-, u₁, x₁, hw₁, u', j, t, hoc⟩ := (@Finset.mem_filter _ _ (Classical.decPred _) _ _).1 hmem
  have eR := rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u₁
  have eN : ((keyProg c hU n).rows u₁).nIn = c.upv * (Rows.ofNet hU).nIn := by
    rw [congrArg FlockSoundness.Rows.nIn eR]; rfl
  have eC : ((keyProg c hU n).rows u₁).nComp = c.upv * ((Rows.ofNet hU).nComp + 1) := by
    rw [congrArg FlockSoundness.Rows.nComp eR]; rfl
  obtain ⟨hu', hj, ht, h1, h2, hxv⟩ := hoc
  have hlt := Rows.mul_add_lt hu' (show 1 + (outCol c j t - (Rows.ofNet hU).nIn) < (Rows.ofNet hU).nComp + 1 by omega)
  have hi : x₁.val - ((keyProg c hU n).rows u₁).nIn < ((keyProg c hU n).rows u₁).nComp := by
    rw [eN, eC]; omega
  have hx₁ : x₁ = ((keyProg c hU n).rows u₁).comp ⟨_, hi⟩ :=
    Fin.ext (by simp only [FlockSoundness.Rows.comp]; rw [eN] at hi ⊢; omega)
  have hu₁ : u₁ = u := by
    have hm := (Const.Prog.instC (keyProg c hU n) (outsOf c hU n) u₁).mem ⟨_, hi⟩
    rw [← hx₁] at hm
    have hm' : (keyProg c hU n).unit g = some u₁ := hw₁ ▸ hm
    rw [hgu] at hm'
    exact (Option.some.inj hm').symm
  subst hu₁
  rcases (Const.Prog.instC (keyProg c hU n) (outsOf c hU n) u₁).shared x x₁ (hx.trans hw₁.symm) with he | ⟨-, h2'⟩
  · exact ⟨u', j, t, hu', hj, ht, h1, h2, by rw [he]; exact hxv⟩
  · rw [eN] at h2'
    omega

end FlockSoundness.Discharge.Layout
