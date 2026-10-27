import Proofs.Flock.Claims.KS.PlainAccPrR
import Proofs.Flock.Claims.KS.PlainRTable

/-!
# The level columns an accepted run of `VR rd` opens

At the level columns `plainRawLevelColsRV rd` (`plainRawLevelCols` at the run `runOfRV rd` reads):

* `plainRawLevelColsRV_top`: an accepted run has none at or past its schedule's last level (`plainRawLevelCols_topK`'s
  proof): the run's shape is table 0's, whose schedule is the statement's `fast100` one. So `levelTopAt_R`.
-/

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **`runOfRV rd`'s run at an accepted run is table 0's of the statement's setup**, at its `m`. -/
theorem runOfRV_m {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω c).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} (hs : setupOf false x = some (s, spec, sts))
    (hst : sts[0]? = some st) {rr : Claims.Flock.ZkRun}
    (hro : runOfRV rd (Claims.run sha512 P (VR rd) x ω c).seen 0 = some rr) : rr.m = st.m := by
  obtain ⟨fl, hfl, hx⟩ := accR_last h
  unfold runOfRV at hro
  rw [show (Claims.run sha512 P (VR rd) x ω c).seen.msgs = (Claims.run sha512 P (VR rd) x ω c).msgs from rfl, hfl]
    at hro
  dsimp only at hro
  rw [hx] at hro
  cases hd : decR rd x ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst)
      (((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.snd).map RecDeploy.outCoin) with
  | none => rw [hd] at hro; cases hro
  | some d =>
    rw [hd] at hro
    obtain ⟨st', hst', hm⟩ := runOfDecR_m hro
    obtain ⟨⟨s', spec', sts'⟩, _, _, _, _, _, _, hv, -, -, -, -, -, hdd⟩ := decR_some hd
    rw [hdd] at hst'
    rw [hs] at hv
    obtain ⟨-, -, rfl⟩ : s = s' ∧ spec = spec' ∧ sts = sts' := by simpa using hv
    have e : st' = st := Option.some.inj (hst'.symm.trans hst)
    rw [hm, e]

/-- **An accepted run of `VR rd` has no level columns at or past its schedule's last level**
(`plainRawLevelCols_topK`'s proof, at `runOfRV_m`). -/
theorem plainRawLevelColsRV_top {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω}
    {x : Claims.Msg} {ω : Ω} {c : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω c).out = .acc)
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule}
    (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch)
    (r : ℕ) {ℓ : ℕ} (hℓ : sch.levels.size ≤ ℓ) :
    plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω c).seen r ℓ = [] := by
  unfold plainRawLevelColsRV
  rcases hro : runOfRV rd (Claims.run sha512 P (VR rd) x ω c).seen 0 with _ | rr
  · rfl
  · have hm := runOfRV_m h hs hst hro
    have e : rr.plainShape.sched = sch := by
      show Claims.Flock.plainSched rr.m = sch
      unfold Claims.Flock.plainSched
      rw [hm, hsch]
      rfl
    have htop : ∀ (a : Array Round) (b : ByteArray), plainLevelColsOf rr.plainShape a b ℓ = [] := fun a b => by
      unfold plainLevelColsOf
      rw [if_neg]
      rintro ⟨-, hl⟩
      rw [e] at hl
      omega
    show (match rr.reps[r]? with
      | some y => plainLevelColsOf rr.plainShape y.1 y.2 ℓ
      | none => []) = []
    split <;> simp [htop]

/-- **`LevelTopAt` at `VR rd` and its level columns.** -/
theorem levelTopAt_R (rd : Claims.Msg → Option FinalP) : LevelTopAt (VR rd) (fun _ => plainRawLevelColsRV rd) :=
  fun _ _ _ _ h _ _ _ _ _ hs hst hsch r _ hℓ => plainRawLevelColsRV_top h hs hst hsch r hℓ

end FlockDeploy.KS
