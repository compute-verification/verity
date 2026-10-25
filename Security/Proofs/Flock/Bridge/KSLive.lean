import Proofs.Flock.Bridge.Assembly
import Proofs.Flock.Bridge.Count
import Proofs.Flock.Bridge.FaultyRightC

/-!
# `SampledKSLiveV2` from the two open targets, and where the second one fails

* `Bridge.sampledKSLiveV2_of_transports_e2e : TransportsLiveC → EndToEndHiddenC → SampledKSLiveV2`: `AssemblyTargetC`
  (`assemblyTargetC`) with M1's core (`faultyRightTargetC`), the slot count (`partRus`) and the count lemma (`count`)
  plugged in.
* `Bridge.faulty_of_read_noRow`: under `commitDep`, a unit that reads a position outside it whose wire has no row is
  faulty for every witness. No opening passes there (`atOf` is `none`, so `commitDep_val_noRow`), and `progOf` has no
  constant gates (`progOf_ne_const`), so `Prog.Correct`'s read clause fails. The constant `oneG` and the zero `zeroG`
  are input gates with no row (`rowOfH = none`) that a unit reads through its constant column
  (`Layout.single_gate`), so every unit that reads them is faulty, honest or not, and `EndToEndHiddenC` at
  `canonOpen` counts those units as wrong. That is why the second hypothesis needs the constants to be public in
  `progOf` (`.const`), not a new premise.
-/

namespace Flock.Bridge

open FlockSoundness.Audit Claims Claims.Sampling

variable {C : FlockSoundness.Audit.Circuit}

/-- A wire with no row has no row at its position. -/
theorem rowOfPos_posOf_none (rows : Fin C.N → Option RowAt) {g : Fin C.N} (h : rows g = none) :
    rowOfPos C rows (posOf C g) = none := by
  unfold rowOfPos
  cases hw : wireOf C (posOf C g) with
  | none => rfl
  | some g' =>
    have he : posOf C g' = posOf C g := by simpa using List.find?_some hw
    simp [posOf_injective he, h]

/-- **No opening passes under `commitDep` at a position with no row**, so every witness's value there is `none`. -/
theorem commitDep_val_noRow (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (nl : ℕ) (roots : List Msg)
    (H : Msg → Msg) (s : Claims.Sampling.Stmt) (hs : s.roots = roots) (w : Wit) {i : ℕ} (hr : rowOfPos C rows i = none) :
    w.val H (commitDep C rows c nl roots) s i = none := by
  unfold Wit.val
  rw [List.find?_eq_none.2, Option.map_none]
  intro o _ ho
  simp only [Bool.and_eq_true, beq_iff_eq] at ho
  obtain ⟨hp, hO⟩ := ho
  subst hs
  simp [Commitment.Opens, commitDep, pathCommit, atOf, hp, hr, OC.eval] at hO

/-- **A unit that reads a position with no row is faulty** under `commitDep`, at every witness: the position is outside
the unit, no opening passes there (`commitDep_val_noRow`), and `progOf` has no constant to read instead
(`progOf_ne_const`). -/
theorem faulty_of_read_noRow {n : ℕ} (P : Partition C n) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (nl : ℕ) (roots : List Msg) (H : Msg → Msg) (kd : ℕ) (desc : Msg) (q : Query) (w : Wit) (u : Fin n) {p i : ℕ}
    (hp : p ∈ unitGates P u) {x : BGate} (hx : (progOf C).gates[p]? = some x) (hi : i ∈ x.reads)
    (hiu : i ∉ unitGates P u) (hr : rowOfPos C rows i = none) :
    u.val ∈ faulty H (commitDep C rows c nl roots) (stmtOf P kd desc q roots) (partOf P) w := by
  classical
  unfold faulty
  refine Finset.mem_filter.2 ⟨by rw [Finset.mem_range, rus_length_partOf]; have := u.isLt; omega, ?_⟩
  rintro ⟨r, hr', ι, hc⟩
  rw [rus_partOf_getElem?] at hr'
  cases hr'
  have hread := (hc (unitGates P u) (List.mem_singleton_self _) p hp).1 x hx i hi hiu
  apply hread
  have hgates : RU.gates [unitGates P u] = unitGates P u := by simp [RU.gates]
  have hnc : ∀ b, (progOf C).gates[i]? ≠ some (.const b) := progOf_ne_const i
  unfold Prog.read
  split
  · rename_i b hb
    exact absurd hb (hnc b)
  · change RU.ref (progOf C) [unitGates P u]
      (w.val H (commitDep C rows c nl roots) (stmtOf P kd desc q roots)) ι i = none
    unfold RU.ref
    have hv := commitDep_val_noRow rows c nl roots H (stmtOf P kd desc q roots) rfl w hr
    simp [hgates, hiu, hv]

end Flock.Bridge

namespace Flock.Guarantees

/-- **`SampledKSLiveV2` from canonical transport and `EndToEndHiddenC`**: `AssemblyTargetC` with its three proved
targets plugged in. -/
theorem _root_.Flock.Bridge.sampledKSLiveV2_of_transports_e2e (hT : Bridge.TransportsLiveC)
    (hE : Bridge.EndToEndHiddenC) : SampledKSLiveV2 :=
  Bridge.assemblyTargetC hT Bridge.faultyRightTargetC Bridge.partRus Bridge.count hE

end Flock.Guarantees
