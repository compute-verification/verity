import Proofs.Flock.Bridge.Partitions
import Proofs.Flock.Bridge.Statement
import Proofs.Flock.Soundness.Discharge.Layout.Outs

/-!
# The key program's proof blocks partition its lowered program

At a circuit file's key program (`keyProg c hU n`, partitioned into its units, its outputs the returned outputs' bits
`outsOf`, at the circuit's constants `constsOf`), `partOf_partitions`'s premises on the circuit hold:

* `unit_of_mem_outsOf`, `outsOf_not_input`: a returned output's bit is a gate some unit computes, so no output is an
  input.
* `keyProg_unit_ne`: where the circuit file has a template instance (`1 ≤ upv`), every unit computes a gate.

So `partOf_partitions_key`: at strata that sum to `n`, the key program's proof blocks (`partOf`) partition its lowered
program (`progOf`) when every proof block commits at most `unitClassMax` bits.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Layout FlockSoundness.Discharge.Placed
  Claims.Sampling

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- **A returned output's bit is a gate some unit computes**: the column it sits on is a computed column of the unit
instance that puts it there. -/
theorem unit_of_mem_outsOf {g : Fin (keyN c hU n)} (hg : g ∈ outsOf c hU n) :
    ∃ u, (keyProg c hU n).unit g = some u := by
  obtain ⟨-, u₁, x₁, hw₁, u', j, t, hoc⟩ := (@Finset.mem_filter _ _ (Classical.decPred _) _ _).1 hg
  have eR := rows_circuitProg c hU n (keyB c hU) (FlockSoundness.Discharge.Aliased.keyIdx c (keyB c hU)) u₁
  have eN : ((keyProg c hU n).rows u₁).nIn = c.upv * (FlockSoundness.Rows.ofNet hU).nIn := by
    rw [congrArg FlockSoundness.Rows.nIn eR]; rfl
  have eC : ((keyProg c hU n).rows u₁).nComp = c.upv * ((FlockSoundness.Rows.ofNet hU).nComp + 1) := by
    rw [congrArg FlockSoundness.Rows.nComp eR]; rfl
  obtain ⟨hu', -, -, h1, h2, hxv⟩ := hoc
  have hlt := FlockSoundness.Rows.mul_add_lt hu'
    (show 1 + (outCol c j t - (FlockSoundness.Rows.ofNet hU).nIn) < (FlockSoundness.Rows.ofNet hU).nComp + 1 by
      omega)
  have hi : x₁.val - ((keyProg c hU n).rows u₁).nIn < ((keyProg c hU n).rows u₁).nComp := by
    rw [eN, eC]; omega
  have hx₁ : x₁ = ((keyProg c hU n).rows u₁).comp ⟨_, hi⟩ :=
    Fin.ext (by simp only [FlockSoundness.Rows.comp]; rw [eN] at hi ⊢; omega)
  have hm := (FlockSoundness.Discharge.Const.Prog.instC (keyProg c hU n) (outsOf c hU n) u₁).mem ⟨_, hi⟩
  rw [← hx₁] at hm
  have hm' : (keyProg c hU n).unit g = some u₁ := hw₁ ▸ hm
  exact ⟨u₁, hm'⟩

/-- **No output of the key program's circuit is an input**: each is a gate some unit computes. -/
theorem outsOf_not_input (g : Fin (keyN c hU n)) (hg : g ∈ ((keyProg c hU n).circuit (outsOf c hU n)).outputs) :
    (((keyProg c hU n).circuit (outsOf c hU n)).op g).isInput = false := by
  obtain ⟨u, hu⟩ := unit_of_mem_outsOf c hU n hg
  have h : ¬ ((keyProg c hU n).op g).isInput = true := fun h => by
    rw [((keyProg c hU n).input_iff g).2 h] at hu
    cases hu
  show ((keyProg c hU n).op g).isInput = false
  simpa using h

/-- **Every unit of the key program computes a gate** where the circuit file has a template instance. -/
theorem keyProg_unit_ne (hupv : 1 ≤ c.upv) (outs : Finset (Fin (keyN c hU n))) (u : Fin n) :
    ∃ g, ((keyProg c hU n).partition outs).unit g = some u := by
  have eR := rows_circuitProg c hU n (keyB c hU) (FlockSoundness.Discharge.Aliased.keyIdx c (keyB c hU)) u
  have hc : 0 < ((keyProg c hU n).rows u).nComp := by
    rw [congrArg FlockSoundness.Rows.nComp eR]
    show 0 < c.upv * ((FlockSoundness.Rows.ofNet hU).nComp + 1)
    exact Nat.mul_pos hupv (by omega)
  exact ⟨_, (FlockSoundness.Discharge.Const.Prog.instC (keyProg c hU n) outs u).mem ⟨0, hc⟩⟩

/-- **The key program's proof blocks partition its lowered program**, at strata that sum to `n` and the circuit's
constants, where the circuit file has a template instance and every proof block commits at most `unitClassMax` bits. -/
theorem partOf_partitions_key (hupv : 1 ≤ c.upv) {ns : List ℕ} (hns : ns.sum = n)
    (hcls : ∀ u, committed (progOf ((keyProg c hU n).circuit (outsOf c hU n)) fun g => constsOf c hU n g)
      (blockOf ((keyProg c hU n).partition (outsOf c hU n)) (fun g => constsOf c hU n g) u) ≤ unitClassMax) :
    (partOf ((keyProg c hU n).partition (outsOf c hU n)) (fun g => constsOf c hU n g) ns).Partitions
      (progOf ((keyProg c hU n).circuit (outsOf c hU n)) fun g => constsOf c hU n g) :=
  partOf_partitions _ _ hns (keyProg_unit_ne c hU n hupv _) (outsOf_not_input c hU n) hcls

end Flock.Bridge
