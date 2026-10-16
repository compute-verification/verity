import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify

/-!
# The tables' sizes: `ProvedScope.sizesOk`

* `sizesOk_iff`: `sizesOk` passes exactly when every table's `m` is at least `mMin` (`25` under `--zk`, else `22`).
* `inScope_iff_scope`: for every statement the verifier parses, `Claims.Flock.InScope zk (toReadable rel p)` holds exactly
  when `verify`'s scope refusal (`Parsed.scope`: `ProvedScope.check`, then `sizesOk`) passes. Every table has the first
  table's `m` (`tables_fit`), so `sizesOk` is `InScope`'s `m` bound.
* `sizesOk_of_verify`, `sizesOk_of_zkVerify`: where `Flock.verify` (`Flock.Zk.verify`) accepts, `sizesOk` passes on its
  tables (`fast100` refuses `m < 22`, `Zk.queries0` `m ≤ 24`). So `verdict_sizesOk`: `verifyCmd`'s verdict is the same
  with `sizesOk` as without it, for every session, refusal reasons included.
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge

theorem sizesOk_iff (zk : Bool) (ms : Array Nat) :
    ProvedScope.sizesOk zk ms = .ok () ↔ ∀ m ∈ ms, ProvedScope.mMin zk ≤ m := by
  unfold ProvedScope.sizesOk
  split
  · rename_i m hm
    simp only [reduceCtorEq, false_iff, not_forall, not_le]
    exact ⟨m, Array.mem_of_find?_eq_some hm, by simpa using Array.find?_some hm⟩
  · rename_i hm
    simp only [true_iff]
    intro m hmem
    simpa using Array.find?_eq_none.1 hm m hmem

theorem scope_ok_iff (p : Parsed) :
    p.scope = .ok () ↔ p.check = .ok () ∧ ProvedScope.sizesOk p.zk ((p.sts.map Setup.ofCircuit).map Setup.m) = .ok () := by
  unfold Parsed.scope
  cases p.check <;> simp [bind, Except.bind]

theorem mMin_ge (zk : Bool) : 22 ≤ ProvedScope.mMin zk := by
  cases zk <;> simp [ProvedScope.mMin]

/-- **The readable scope is `verify`'s scope refusal**, for every statement the verifier parses: `ProvedScope.check`, then
every table's `m` at least `mMin` (`ProvedScope.sizesOk`). -/
theorem inScope_iff_scope (rel : Parsed → Relation) {p : Parsed} (hp : Parses p) :
    Claims.Flock.InScope p.zk (toReadable rel p) ↔ p.scope = .ok () := by
  rw [inScope_iff rel hp, scope_ok_iff, sizesOk_iff, Array.map_map]
  refine and_congr_right fun hc => ?_
  have hs := tablesOf_of_parses hp
  have hR := ((check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  obtain ⟨st, h0, -, -, -, -, -, -, hzk⟩ := table0 hs hR.2.1 hpi
  obtain ⟨hpos, hst⟩ := Array.getElem?_eq_some_iff.1 h0
  have em : (toReadable rel p).m = st.m := by simp [toReadable, h0]
  rw [em]
  constructor
  · intro h22 m hm
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hm
    have hj' : j < p.sts.size := by simpa using hj
    have hfit : p.sts[j].m = p.sts[0].m := (tables_fit hs hR.2.1 hpi hj').1
    simp only [Array.getElem_map, Function.comp_apply]
    show ProvedScope.mMin p.zk ≤ p.sts[j].m
    rw [hfit, hst]
    cases hz : p.zk
    · simpa [ProvedScope.mMin] using h22
    · simpa [ProvedScope.mMin] using hzk hz
  · intro h
    have := h _ (Array.mem_map_of_mem (f := Setup.m ∘ Setup.ofCircuit) (Array.getElem_mem hpos))
    simp only [Function.comp_apply, hst] at this
    exact (mMin_ge _).trans this

/-- **Where `Flock.verify` accepts, `sizesOk` passes on its tables**: its schedules are `fast100`'s, which refuse
`m < 22`. -/
theorem sizesOk_of_verify {spec : SessionSpec} {tabs : Array Setup} {record : String} {proofs : Array ByteArray}
    (h : Flock.verify spec tabs record proofs = .ok ()) : ProvedScope.sizesOk false (tabs.map Setup.m) = .ok () := by
  unfold Flock.verify at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := Refine.ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨_, -, h⟩ := Refine.ebind_ok h
  obtain ⟨_, -, h⟩ := Refine.ebind_ok h
  obtain ⟨schs, hschs, -⟩ := Refine.ebind_ok h
  obtain ⟨hsz, hsch⟩ := Exec.array_mapM_ok_getElem hschs
  refine (sizesOk_iff _ _).2 fun m hm => ?_
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hm
  have hj' : j < tabs.size := by simpa using hj
  have hf := hsch j hj' (hsz ▸ hj')
  simp only [Array.getElem_map]
  cases hm : fast100 tabs[j].m with
  | none => rw [hm] at hf; cases hf
  | some _ =>
    unfold fast100 at hm
    by_contra hlt
    simp only [ProvedScope.mMin, Bool.false_eq_true, ↓reduceIte, not_le] at hlt
    simp [hlt] at hm

/-- **Where `Flock.Zk.verify` accepts, `sizesOk` passes on its tables**: its schedules are `Zk.schedule`'s, whose
`queries0` refuses `m ≤ 24`. -/
theorem sizesOk_of_zkVerify {spec : SessionSpec} {tabs : Array Setup} {record : String} {proofs : Array ByteArray}
    (h : Flock.Zk.verify spec tabs record proofs = .ok ()) : ProvedScope.sizesOk true (tabs.map Setup.m) = .ok () := by
  obtain ⟨-, _, _, -, -, -, -, -, hall⟩ := ZkExec.zk_verify_ok _ _ _ _ h
  refine (sizesOk_iff _ _).2 fun m hm => ?_
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hm
  have hj' : j < tabs.size := by simpa using hj
  obtain ⟨_, _, hz, -⟩ := hall j hj'
  simp only [Array.getElem_map]
  unfold Zk.schedule at hz
  obtain ⟨_, hq, -⟩ := except_bind_ok hz
  unfold Zk.queries0 at hq
  by_contra hlt
  simp only [ProvedScope.mMin, ↓reduceIte, not_le] at hlt
  simp only [show tabs[j].m ≤ 24 by omega, ↓reduceIte] at hq
  cases hq

/-- **`sizesOk` changes no verdict**: `verifyCmd`'s verdict tail (the verify step, the form's refusal `scope`, then
`sizesOk` on the tables, then the rest) is the same without `sizesOk`, for every session, its refusal reason included. -/
theorem verdict_sizesOk (zk : Bool) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) (scope rest : Except String Unit) :
    (do
      (if zk then Flock.Zk.verify spec tabs record proofs else Flock.verify spec tabs record proofs)
      scope
      ProvedScope.sizesOk zk (tabs.map Setup.m)
      rest) =
    (do
      (if zk then Flock.Zk.verify spec tabs record proofs else Flock.verify spec tabs record proofs)
      scope
      rest) := by
  have hv : (if zk then Flock.Zk.verify spec tabs record proofs else Flock.verify spec tabs record proofs) = .ok () →
      ProvedScope.sizesOk zk (tabs.map Setup.m) = .ok () := by
    cases zk
    · exact sizesOk_of_verify
    · exact sizesOk_of_zkVerify
  cases h : (if zk then Flock.Zk.verify spec tabs record proofs else Flock.verify spec tabs record proofs) with
  | error e => rfl
  | ok u =>
    cases u
    cases scope with
    | error e => rfl
    | ok u =>
      cases u
      simp only [bind, Except.bind, hv h]

end Flock.Readable
