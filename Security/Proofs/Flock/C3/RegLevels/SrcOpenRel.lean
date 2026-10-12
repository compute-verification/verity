import Proofs.Flock.C3.RegLevels.SrcRun
import Proofs.Flock.Soundness.Refine.OpeningConverse
import Proofs.Flock.VBridge.Algebra.RingSwitch
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins
import Proofs.Flock.Soundness.Discharge.Exec.Shape
import Proofs.Flock.Recursive.ForkRuns

/-!
# n7's levels: ring switching's output is the opening's (`OpenRel`)

`levelTie_of_run`'s `hR` asks that ring switching's `T` and switched claims be the output R's opening plays to
(`OpenRel`). `openRel_of_plays` gives it where the ring-switch residuals are zero: each claim passes the verifier's
value check and `T` is the batched sum on the transposes the messages hold (`ringSwitch_checks`), at the coins
`c[k₀]`, `c[k₀ + 1]` and `c[k₀ + 2]` (`ringSwitch_spec`). So the model's opening of the run's claims, played with the
ring switches and those coins, returns that `T` and those claims (`opening_of_checks`, on `twoRsRecord`, a record of
the three rounds retaining the bytes each answers). That play is one on a tape (`toT_exact`) of the opening's exact
shape (`opening_exact`), so it is the source's opening steps when their messages and coins are the play's, and its
output is `u.opened` (`PlaysT.det`).
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr toT_exact)

/-- A record of ring switching's rounds for two claims with ring switches `s`: claim `k`'s seven coins `c[k₀ + k]`,
retaining the bytes framed before them, then the two batching coins `c[k₀ + 2]`. -/
def twoRsRecord (s c : Array (Array Flock.F128)) (k₀ : ℕ) : Flock.Transcript :=
  ⟨#[⟨7, ByteArray.empty, c[k₀ + 0]!, some (rsFramed (rsPending ByteArray.empty 0) s[0]!)⟩,
      ⟨7, ByteArray.empty, c[k₀ + 1]!, some (rsFramed (rsPending ByteArray.empty 1) s[1]!)⟩,
      ⟨2, ByteArray.empty, c[k₀ + 2]!, some (ByteArray.empty ++ Flock.Transcript.header 4 2 2)⟩], 0, ByteArray.empty⟩

theorem twoRsRecord_rounds {s c : Array (Array Flock.F128)} {k₀ : ℕ} (h7 : ∀ k < 2, c[k₀ + k]!.size = 7)
    (h2 : c[k₀ + 2]!.size = 2) :
    RoundsWF (twoRsRecord s c k₀).rounds ∧ RsRounds (twoRsRecord s c k₀) s 2 (fun k => c[k₀ + k]!) c[k₀ + 2]! := by
  refine ⟨fun i hi => ?_, ⟨(show 0 + 2 < 3 by omega), fun k hk => ?_, fun k hk => ?_, fun k hk => ?_, rfl, rfl, rfl⟩⟩
  · have hi' : i < 3 := hi
    rcases (by omega : i = 0 ∨ i = 1 ∨ i = 2) with rfl | rfl | rfl
    · exact h7 0 (by omega)
    · exact h7 1 (by omega)
    · exact h2
  all_goals rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl <;> exact rfl

/-- **Ring switching's output is the opening's** (`levelTie_of_run`'s `hR`). Where ring switching over F returns zero
residuals on two claims that are the run's (`ClaimRel`), at messages whose transposes are their ring switches' (`hT`),
on rounds `c` none of which is empty, and the source's opening steps are the opening's on those ring switches
(`OpenTie`'s form) with `c`'s rounds from `k₀` as their coins, ring switching's `T` and switched claims are the run's
opening output. The claims are read as the verifier's (`a.map (·.toClaim)`): `AClaim F128` has no default. -/
theorem openRel_of_plays {e : ExecHolo} {u : C3Run e}
    {M : Flock.RecAlgebra.Messages Flock.F128} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {k₀ k₂ : ℕ} {rsr : Array Flock.F128} {rs : Flock.RecAlgebra.RsPts Flock.F128}
    (hrs : Flock.RecAlgebra.ringSwitch M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂))
    (hz : ∀ x ∈ rsr, x = 0) (hne : Flock.RecAlgebra.NoEmpty c) (ha : a.size = 2)
    (hshape : ∀ k < 2, (M.rs[k]!).size = 128 ∧ ((a.map (·.toClaim))[k]!).skip.size = 64 ∧
      ((a.map (·.toClaim))[k]!).x.size = e.cls.shape.mReg - 6)
    (hT : ∀ k < 2, M.rsT[k]! = tOf M.rs[k]!)
    (hcl : ∀ i : Fin 2, ClaimRel e.cls.shape.mReg (a.map (·.toClaim))[i.val]! (u.claim i).toClaim)
    (hpl : PlaysT (FlockC3.Opening.openingAt u) u.op (some u.opened))
    (hmsg : ∃ ρ γ, u.op = FlockC3.Opening.openSteps (fun i => vecOf 128 M.rs[i.val]!) ρ γ)
    (hcoin : u.op.filterMap Sum.getRight? = ((List.range' k₀ (k₂ - k₀)).map fun i => c[i]!).map (coinArr fOf)) :
    ∃ nc, OpenRel e.cls.shape.mReg nc rs.T rs.sws u.opened := by
  obtain ⟨ρ, γ, hop⟩ := hmsg
  have hn : a.size ≠ 0 := by omega
  have hA : (a.map (·.toClaim)).size = 2 := by simp [ha]
  have hcl_eq : ∀ k (hk : k < a.size), (a.map (·.toClaim))[k]! = a[k].toClaim := fun k hk => by
    rw [getElem!_pos (a.map (·.toClaim)) k (by simpa using hk), Array.getElem_map]
  obtain ⟨_betas, -, -, hgam, hk₂, -, hsws, hj⟩ := Flock.RecAlgebra.ringSwitch_spec hn hrs
  obtain ⟨hval, hTr⟩ := Flock.RecAlgebra.ringSwitch_checks hn hrs hz
  obtain ⟨hgc, hgs⟩ := Flock.RecAlgebra.ofRounds_get? hne hgam
  rw [ha] at hgc hgs
  obtain ⟨hlt, hgeq⟩ := Array.getElem?_eq_some_iff.mp hgc
  have hg : c[k₀ + 2]! = rs.gamma := (getElem!_pos c _ hlt).trans hgeq
  have hcj : ∀ j < 2, c[k₀ + j]!.size = 7 ∧ rs.sws[j]! =
      ({ gamma := rs.gamma[j]!, rpp := c[k₀ + j]!, z := ((a.map (·.toClaim))[j]!).x.extract 1 } : Flock.Switched) := by
    intro j hj2
    obtain ⟨c', hc', -, -, hsw⟩ := hj j (by omega)
    obtain ⟨hget, hsz⟩ := Flock.RecAlgebra.ofRounds_get? hne hc'
    obtain ⟨hlt', hceq⟩ := Array.getElem?_eq_some_iff.mp hget
    rw [(getElem!_pos c _ hlt').trans hceq, getElem!_pos rs.sws j (by omega), hcl_eq j (by omega)]
    exact ⟨hsz, Option.some.inj ((Array.getElem?_eq_getElem (by omega)).symm.trans hsw)⟩
  obtain ⟨hwf, hR⟩ := twoRsRecord_rounds (s := #[M.rs[0]!, M.rs[1]!]) (fun k hk => (hcj k hk).1)
    (by rw [hg]; exact hgs)
  have hR' : RsRounds (twoRsRecord #[M.rs[0]!, M.rs[1]!] c k₀) #[M.rs[0]!, M.rs[1]!] (a.map (·.toClaim)).size
      (fun k => c[k₀ + k]!) c[k₀ + 2]! := by
    rw [hA]
    exact hR
  obtain ⟨op, hP, hO⟩ := opening_of_checks e.cls.shape.mReg (seven_le_mReg e.cls.shape)
    (a.map (·.toClaim)) [(u.claim 0).toClaim, (u.claim 1).toClaim] (by simp [ha])
    (fun k hk => by
      have hk' : k < 2 := hk
      rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
      · exact hcl 0
      · exact hcl 1)
    (by simp [ha]) (by simp [ha]) #[M.rs[0]!, M.rs[1]!] _ hwf _ _ (by simp [ha])
    (fun k hk => by
      have hk' : k < 2 := by simpa [ha] using hk
      rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
      · exact hshape 0 (by omega)
      · exact hshape 1 (by omega))
    (fun k hk => by
      have hk' : k < 2 := by simpa [ha] using hk
      have hv := hval k (by omega)
      rw [hcl_eq k (by omega)]
      rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
      · exact hv
      · exact hv)
    hR'
  rw [hA] at hP hO
  obtain ⟨ts, hts, hl, hr⟩ := toT_exact fOf hP
  have hl' : ts.filterMap Sum.getLeft? = u.op.filterMap Sum.getLeft? := by
    rw [hl, hop]
    rfl
  have hc3 : (List.range' k₀ (k₂ - k₀)).map (fun i => c[i]!) =
      (List.range 2).map (fun k => c[k₀ + k]!) ++ [c[k₀ + 2]!] := by
    rw [show k₂ - k₀ = 3 by omega]
    rfl
  have hr' : ts.filterMap Sum.getRight? = u.op.filterMap Sum.getRight? :=
    hr.trans ((congrArg (List.map (coinArr fOf)) hc3.symm).trans hcoin.symm)
  have hto : ts = u.op := by
    have h1 := FlockSoundness.Discharge.Exec.weave_self ts
    have h2 := FlockSoundness.Discharge.Exec.weave_self u.op
    rw [FlockSoundness.Discharge.Exec.Exact.kinds hts (FlockSoundness.Discharge.Exec.opening_exact execArith _ _),
      hl', hr'] at h1
    rw [FlockSoundness.Discharge.Exec.Exact.kinds hpl (FlockSoundness.Discharge.Exec.opening_exact execArith _ _)]
      at h2
    exact h1.symm.trans h2
  rw [hto] at hts
  obtain rfl : op = u.opened := Option.some.inj (hts.det hpl)
  have hTeq : rs.T = Flock.dot c[k₀ + 2]! ((List.range 2).map fun k =>
      Flock.dot (Flock.eqTable c[k₀ + k]!) (tOf #[M.rs[0]!, M.rs[1]!][k]!)).toArray := by
    rw [hTr, ha, hg]
    refine congrArg (fun l => Flock.dot rs.gamma (List.toArray l)) (List.map_congr_left fun j hj => ?_)
    have hj2 := List.mem_range.mp hj
    show Flock.dot (Flock.eqTable (rs.sws[j]!).rpp) M.rsT[j]! =
      Flock.dot (Flock.eqTable c[k₀ + j]!) (tOf #[M.rs[0]!, M.rs[1]!][j]!)
    rw [(hcj j hj2).2, hT j hj2]
    rcases (by omega : j = 0 ∨ j = 1) with rfl | rfl <;> rfl
  have hsw : rs.sws = (Array.range 2).map fun k => ({ gamma := (c[k₀ + 2]!)[k]!, rpp := c[k₀ + k]!,
      z := ((a.map (·.toClaim))[k]!).x.extract 1 } : Flock.Switched) := by
    apply Array.ext
    · rw [hsws, ha, Array.size_map, Array.size_range]
    · intro k h1 h2
      have hk : k < 2 := by simpa using h2
      rw [Array.getElem_map, Array.getElem_range, ← getElem!_pos rs.sws k h1, (hcj k hk).2, ← hg]
  refine ⟨2, ?_⟩
  rw [hTeq, hsw]
  exact hO

end FlockC3.RegLevels
