import Proofs.Flock.C3.RegLevels.SrcTie
import Proofs.Flock.VBridge.Runs.Lg

/-!
# n7's levels: `LevelTie` from R's opening's algebra

R's `rec-opening` unit computes, after the claims' ring switches, Ligerito's algebra over F from the absorbed `T`
(`RecOpening.run`, `opening_parts`), and its last residual makes that `T` the batched sum. Where its residuals are zero,
V*'s Ligerito passes the final check (`ligerito_checks`); where the proof also carries V*'s messages, OOD values and
`y_r`, a nonce per query and claim and its cap, and every level's queries pass against that cap (`Opened`), the
verifier's Ligerito accepts (`ligerito_of_run`) on a record of V*'s rounds built here (`levelTie_of_lig`). At the shape
`RecOpening.shape?` gives at `mReg` the schedule is `fast100`'s and, where `ℓ` is in the class's range, R's registered
one (`schedOf_holoAt`), so `levelTie_of_run` gives `LevelTie` from R's opening's run and what V*'s other R units and
the source hold: the openings (`Opened`), ring switching's output (`OpenRel`), the cap's digests, the proof's fields,
and the source's Ligerito steps.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)

/-- `RecOpening.shape?`'s shape is at its `m`, on `fast100`'s schedule there. -/
theorem shape_spec {m c : ℕ} {sh : Flock.Shape} (h : Flock.RecOpening.shape? m c = some sh) :
    sh.m = m ∧ Flock.fast100 m = some sh.sched := by
  unfold Flock.RecOpening.shape? at h
  cases hf : Flock.fast100 m with
  | none => simp [hf] at h
  | some s =>
    by_cases hc : c < 1
    · simp [hf, hc] at h
    · simp [hf, hc] at h
      subst h
      exact ⟨rfl, rfl⟩

/-- **R's shape is on R's registered schedule** where `ℓ` is in the class's range: `holoAt`'s is `schedAt mReg`, the
model schedule of `fast100`'s at `mReg`, where `RecOpening.shape?` found R's. -/
theorem schedOf_holoAt {st : Flock.Stmt} {hp : HoloParams} {cl : ℕ} {sh : Flock.Shape}
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (h : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg cl = some sh) :
    Flock.fast100 sh.m = some sh.sched ∧ Refine.schedOf sh.sched = (holoAt st hp).rsch.sch := by
  obtain ⟨hm, hf⟩ := shape_spec h
  refine ⟨by rw [hm]; exact hf, ?_⟩
  have e : (holoAt st hp).rsch.sch =
      if hp.EllOk (execClassAt st).cls.kLog then schedAt (holoAt st hp).cls.shape.mReg else default := rfl
  rw [e, ite_eq_left hell]
  unfold FlockSoundness.Discharge.Recursion.schedAt
  rw [hf]

/-- **R's opening's algebra** is its claims' ring switches, then Ligerito from the absorbed `T` on their output, then
the absorbed `T` minus the batched sum. -/
theorem opening_parts {sh : Flock.Shape} {M : Flock.RecAlgebra.Messages Flock.F128}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {k₀ k₁ : ℕ} {src : Flock.RecAlgebra.Coins}
    {res : Array Flock.F128} (h : Flock.RecOpening.run sh M a k₀ src = .ok (res, k₁)) :
    ∃ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch M a k₀ src = .ok ((rsr, rs), k₂) ∧
      Flock.RecAlgebra.ligerito sh.sched M (M.tAtom.getD rs.T) rs.sws k₂ src = .ok ((lgr, lg), k₁) ∧
      res = rsr ++ lgr ++ #[Flock.RecAlgebra.Val.add (M.tAtom.getD rs.T) rs.T] := by
  unfold Flock.RecOpening.run at h
  obtain ⟨⟨rsr, rs⟩, k₂, h₁, h⟩ := Flock.RecAlgebra.cm_bind h
  obtain ⟨⟨lgr, lg⟩, k₃, h₂, h⟩ := Flock.RecAlgebra.cm_bind h
  cases Flock.RecAlgebra.cm_pure h
  exact ⟨rsr, rs, k₂, lgr, lg, h₁, h₂, rfl⟩

/-- **V*'s Ligerito accepts on a record of `c`'s rounds**: where V*'s Ligerito over F returns zero residuals from `tIn`
on the rounds `c` from round `k₂`, at a `fast100` schedule, the proof carries V*'s messages, OOD values and `y_r`, a
nonce per query and claim and the initial cap `cap`, and every level's queries pass against `cap` (`Opened`), the
verifier's Ligerito accepts from round `k₂` to the end of a well-formed record of `c`'s rounds that retain the bytes it
frames, consuming `c`'s rounds from `k₂` on. -/
theorem ligRun_of_lig {m : ℕ} {sch : Flock.Schedule} {M : Flock.RecAlgebra.Messages Flock.F128} {tIn : Flock.F128}
    {sws : Array Flock.Switched} {c : Array (Array Flock.F128)} {k₂ k₃ : ℕ} {lgr : Array Flock.F128}
    {lg : Flock.RecAlgebra.LgPts} (hf : Flock.fast100 m = some sch)
    (hlg : Flock.RecAlgebra.ligerito sch M tIn sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₃))
    (hz : ∀ x ∈ lgr, x = 0) (hne : Flock.RecAlgebra.NoEmpty c)
    {msId : String} {cap : Array ByteArray} {p : Flock.LigeritoProof}
    (hpm : p.messages = M.msgs.map fun mm => (Flock.RecAlgebra.kv mm.1, Flock.RecAlgebra.kv mm.2))
    (hpo : p.oodValues = M.ood) (hyr : p.finalProof.yr = M.yr) (hic : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = sch.levels.size) (hcn : p.consistencyNonces.size = sch.levels.size)
    (hcbn : p.claimNonces.size = M.ood.size + sch.levels.size)
    (hopen : ∀ cs ∈ lg.conss.toList, Flock.RecAlgebra.Opened (Flock.MerkleScheme.sha512 msId) sch cap p M cs) :
    ∃ (rs : Array Flock.Round) (P' : ByteArray), RoundsWF rs ∧
      Flock.ligerito (Flock.MerkleScheme.sha512 msId) sch tIn cap p (Flock.basis sws) ⟨rs, k₂, ByteArray.empty⟩ =
        .ok ((), ⟨rs, c.size, P'⟩) ∧
      consumed rs k₂ c.size = (List.range' k₂ (c.size - k₂)).map (fun i => c[i]!) := by
  obtain ⟨ext, stV, hrun, hd, hmsg, hood, -, hlc, hfin, -⟩ := Flock.RecAlgebra.ligerito_checks hlg hz
  have : Fact (Flock.RecAlgebra.NoEmpty c) := ⟨hne⟩
  have hk₃ : k₃ = c.size := by
    have he : (k₃ == (c.filter (·.size ≠ 0)).size) = true := hd
    rw [Flock.RecAlgebra.filter_noEmpty hne, beq_iff_eq] at he
    exact he
  subst hk₃
  have hdim := (FlockSoundness.Refine.fast100_dims m sch hf).1
  have hlv : ∀ i < sch.levels.size, FlockSoundness.Refine.LevelSeam (sch.levels.getD i default) := fun i hi => by
    have he : sch.levels.getD i default = sch.levels[i] := by simp [Array.getD, hi]
    rw [he]
    exact FlockSoundness.Refine.fast100_level m sch hf _ (Array.getElem_mem hi)
  have h2 : 2 ≤ sch.levels.size := by
    obtain ⟨-, h, -⟩ := hdim
    exact h
  have hcs := FlockVBridge.Runs.ligeritoRun_conss hdim hlv hrun
  obtain ⟨-, H⟩ := Flock.RecAlgebra.ligerito_of_run (ms := Flock.MerkleScheme.sha512 msId) (cap := cap) (p := p)
    hdim hlv hrun hfin hmsg hood hpm hpo hyr hic (by omega) (by omega) (by omega) (by rw [← hlc]; exact hopen)
  obtain ⟨P', F, hF⟩ := H ByteArray.empty
  obtain ⟨rs, hrs⟩ : ∃ rs : Array Flock.Round, rs = c.mapIdx fun j cj =>
      ({ n := cj.size, sha := ByteArray.empty, coins := cj, bytes := some (F j) } : Flock.Round) := ⟨_, rfl⟩
  have hsz : rs.size = c.size := by rw [hrs, Array.size_mapIdx]
  have hget : ∀ j < c.size, (rs[j]!).n = (c[j]!).size ∧ (rs[j]!).coins = c[j]! ∧ (rs[j]!).bytes = some (F j) :=
    fun j hj => by
      rw [getElem!_pos rs j (by omega), getElem!_pos c j hj]
      subst hrs
      refine ⟨?_, ?_, ?_⟩ <;> simp [Array.getElem_mapIdx]
  have hwf : RoundsWF rs := fun j hj => by rw [(hget j (by omega)).2.1, (hget j (by omega)).1]
  have hfit := FlockVBridge.Runs.fits_of (c := c) (F := F) (rs := rs) (k₀ := k₂)
    (fun j hj => (hget j (by omega)).2.1.symm) hwf (fun j _ hj => (hget j (by omega)).2.2)
  rw [hsz] at hfit
  have hcons : consumed rs k₂ c.size = (List.range' k₂ (c.size - k₂)).map (fun i => c[i]!) := by
    unfold consumed
    refine List.map_congr_left fun j hj => ?_
    obtain ⟨hj1, hj2⟩ := List.mem_range'_1.mp hj
    exact (hget j (by omega)).2.1
  exact ⟨rs, P', hwf, hF rs hfit, hcons⟩

/-- **`LevelTie` from V*'s Ligerito**: where V*'s Ligerito accepts on a record of `c`'s rounds (`ligRun_of_lig`), at a
`fast100` schedule that is R's registered one, and `tIn` and `sws` are R's opening's output (`OpenRel`). The source's
Ligerito steps take its kinds, the proof's messages, `c`'s rounds from `k₂` as coins, and the proof's rows. -/
theorem levelTie_of_lig {cap₀ : Cap Dig} {e : ExecHolo} {u : C3Run e} {m : ℕ} {sch : Flock.Schedule}
    {M : Flock.RecAlgebra.Messages Flock.F128} {tIn : Flock.F128} {sws : Array Flock.Switched}
    {c : Array (Array Flock.F128)} {k₂ k₃ : ℕ} {lgr : Array Flock.F128} {lg : Flock.RecAlgebra.LgPts}
    (hf : Flock.fast100 m = some sch) (hsch : Refine.schedOf sch = e.rsch.sch)
    (hlg : Flock.RecAlgebra.ligerito sch M tIn sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₃))
    (hz : ∀ x ∈ lgr, x = 0) (hne : Flock.RecAlgebra.NoEmpty c)
    {msId : String} {cap : Array ByteArray} {p : Flock.LigeritoProof} (hcap : capDig cap = cap₀)
    (hpm : p.messages = M.msgs.map fun mm => (Flock.RecAlgebra.kv mm.1, Flock.RecAlgebra.kv mm.2))
    (hpo : p.oodValues = M.ood) (hyr : p.finalProof.yr = M.yr) (hic : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = sch.levels.size) (hcn : p.consistencyNonces.size = sch.levels.size)
    (hcbn : p.claimNonces.size = M.ood.size + sch.levels.size)
    (hopen : ∀ cs ∈ lg.conss.toList, Flock.RecAlgebra.Opened (Flock.MerkleScheme.sha512 msId) sch cap p M cs)
    {nc : ℕ} (hop : OpenRel sch.m nc tIn sws u.opened)
    (hk : u.lg.map kindOf <+: ligShape GF128 GF256 Dig e.rsch.sch)
    (hl : u.lg.filterMap Sum.getLeft? = ligMsgs p capDig sch)
    (hr : u.lg.filterMap Sum.getRight? = ((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf))
    (ho : ∀ ℓ j, u.os.1 2 ℓ j = openRow p sch ℓ j) :
    LevelTie cap₀ e u := by
  obtain ⟨rs, P', hwf, hlig, hcons⟩ := ligRun_of_lig hf hlg hz hne hpm hpo hyr hic hqn hcn hcbn hopen
  exact ⟨msId, m, sch, tIn, sws, nc, cap, p, ⟨rs, k₂, ByteArray.empty⟩, ⟨rs, c.size, P'⟩, hf, hsch, hcap, hop, hwf,
    hlig, hk, hl, hr.trans (congrArg (List.map (coinArr fOf)) hcons.symm), ho⟩

/-- **R's Ligerito steps woven from the proof's messages and `c`'s rounds are a play's tape** (`levelTie_of_lig`'s
`hk`, `hl` and `hr`, `repStepsWoven_of_tie`'s analog at R): where V*'s Ligerito accepts on a record of `c`'s rounds
(`ligRun_of_lig`), the model's compiled Ligerito plays on the proof's messages and the rounds the verifier's consumed
before its final check (`ligerito_refines`), which consumes none (`finalProg_ok`), so those are `c`'s rounds from `k₂`
on; steps woven from them along its shape are that play's tape (`weave_of_prefix`). -/
theorem lgSteps_of_lig {e : ExecHolo} {u : C3Run e} {m : ℕ} {sch : Flock.Schedule}
    {M : Flock.RecAlgebra.Messages Flock.F128} {tIn : Flock.F128} {sws : Array Flock.Switched}
    {c : Array (Array Flock.F128)} {k₂ k₃ : ℕ} {lgr : Array Flock.F128} {lg : Flock.RecAlgebra.LgPts}
    (hf : Flock.fast100 m = some sch) (hsch : Refine.schedOf sch = e.rsch.sch)
    (hlg : Flock.RecAlgebra.ligerito sch M tIn sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₃))
    (hz : ∀ x ∈ lgr, x = 0) (hne : Flock.RecAlgebra.NoEmpty c)
    {msId : String} {cap : Array ByteArray} {p : Flock.LigeritoProof}
    (hpm : p.messages = M.msgs.map fun mm => (Flock.RecAlgebra.kv mm.1, Flock.RecAlgebra.kv mm.2))
    (hpo : p.oodValues = M.ood) (hyr : p.finalProof.yr = M.yr) (hic : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = sch.levels.size) (hcn : p.consistencyNonces.size = sch.levels.size)
    (hcbn : p.claimNonces.size = M.ood.size + sch.levels.size)
    (hopen : ∀ cs ∈ lg.conss.toList, Flock.RecAlgebra.Opened (Flock.MerkleScheme.sha512 msId) sch cap p M cs)
    (hu : u.lg = FlockSoundness.Discharge.Exec.weave (ligShape GF128 GF256 Dig e.rsch.sch) (ligMsgs p capDig sch)
      (((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf))) :
    u.lg.map kindOf <+: ligShape GF128 GF256 Dig e.rsch.sch ∧ u.lg.filterMap Sum.getLeft? = ligMsgs p capDig sch ∧
      u.lg.filterMap Sum.getRight? = ((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf) := by
  obtain ⟨rs, P', hwf, hlig, hcons⟩ := ligRun_of_lig hf hlg hz hne hpm hpo hyr hic hqn hcn hcbn hopen
  have hdim := (FlockSoundness.Refine.fast100_dims m sch hf).1
  have hlv : ∀ i < sch.levels.size, FlockSoundness.Refine.LevelSeam (sch.levels.getD i default) := fun i hi => by
    have he : sch.levels.getD i default = sch.levels[i] := by simp [Array.getD, hi]
    rw [he]
    exact FlockSoundness.Refine.fast100_level m sch hf _ (Array.getElem_mem hi)
  obtain ⟨_pend, _initCh, _levels, _stF, _stF', tF, -, -, hpl, -, -, -, -, -, -, -, hfp, -⟩ :=
    FlockSoundness.Refine.ligerito_refines (Flock.MerkleScheme.sha512 msId) sch tIn cap p (Flock.basis sws) 2
      (fun _ => 0) capDig (fun _ ℓ j => openRow p sch ℓ j) hdim hlv
      (fun i hi => by
        rw [show sch.levels.getD i default = sch.levels[i] by simp [Array.getD, hi]]
        exact FlockSoundness.Refine.fast100_alpha m sch hf _ (Array.getElem_mem hi))
      (fun _ _ _ _ => rfl) (t := ⟨rs, k₂, ByteArray.empty⟩) (t' := ⟨rs, c.size, P'⟩) hwf hlig
  obtain ⟨-, htF, -⟩ := FlockSoundness.Refine.finalProg_ok hfp
  subst htF
  obtain ⟨ts, hts, h1, h2⟩ := FlockSoundness.Discharge.PlaysCoin.toT_exact fOf hpl
  have hk := FlockSoundness.Discharge.Exec.Spine.prefix hts
    (FlockSoundness.Discharge.Exec.ligeritoC_spine (D := Dig) FlockSoundness.execArith 2 _ _ _)
  rw [hsch] at hk
  have h2' : ts.filterMap Sum.getRight? =
      ((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf) :=
    h2.trans (congrArg (List.map (coinArr fOf)) hcons)
  have he : u.lg = ts := by
    rw [hu, ← h1, ← h2']
    exact FlockVBridge.Family.weave_of_prefix hk
  rw [he]
  exact ⟨hk, h1, h2'⟩

/-- **(n7's levels) `LevelTie` from R's opening's run**, where `ℓ` is in the class's range: R's opening's algebra at
the shape `RecOpening.shape?` gives at `mReg` returns zero residuals on V*'s rounds `c`, and at its ring switches'
output and its Ligerito's points (`hR`) the levels' queries pass against the proof's cap (`Opened`), ring switching's
`T` and switched claims are R's opening's output (`OpenRel`), and the source's coin steps are `c`'s rounds from
Ligerito's first; the proof carries V*'s messages (`hpm hpo hyr hic`, a nonce per query and claim), its cap has the
digests `cap₀`, and the source's Ligerito steps have the shape's kinds, the proof's messages and its rows. -/
theorem levelTie_of_run {st : Flock.Stmt} {hp : HoloParams} {cap₀ : Cap Dig} {u : C3Run (holoAt st hp)} {cl : ℕ}
    {sh : Flock.Shape} {M : Flock.RecAlgebra.Messages Flock.F128} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {k₀ k₁ : ℕ} {res : Array Flock.F128}
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg cl = some sh)
    (hrun : Flock.RecOpening.run sh M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c)
    {msId : String} {cap : Array ByteArray} {p : Flock.LigeritoProof} (hcap : capDig cap = cap₀)
    (hpm : p.messages = M.msgs.map fun mm => (Flock.RecAlgebra.kv mm.1, Flock.RecAlgebra.kv mm.2))
    (hpo : p.oodValues = M.ood) (hyr : p.finalProof.yr = M.yr) (hic : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = sh.sched.levels.size) (hcn : p.consistencyNonces.size = sh.sched.levels.size)
    (hcbn : p.claimNonces.size = M.ood.size + sh.sched.levels.size)
    (hR : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch M a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched M rs.T rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₁) →
      (∀ cs ∈ lg.conss.toList, Flock.RecAlgebra.Opened (Flock.MerkleScheme.sha512 msId) sh.sched cap p M cs) ∧
      (∃ nc, OpenRel sh.sched.m nc rs.T rs.sws u.opened) ∧
      u.lg.filterMap Sum.getRight? = ((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf))
    (hk : u.lg.map kindOf <+: ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch)
    (hl : u.lg.filterMap Sum.getLeft? = ligMsgs p capDig sh.sched)
    (ho : ∀ ℓ j, u.os.1 2 ℓ j = openRow p sh.sched ℓ j) :
    LevelTie cap₀ (holoAt st hp) u := by
  obtain ⟨hf, hsch⟩ := schedOf_holoAt hell hsh
  obtain ⟨rsr, rs, k₂, lgr, lg, h₁, h₂, rfl⟩ := opening_parts hrun
  have hlast : Flock.RecAlgebra.Val.add (M.tAtom.getD rs.T) rs.T = 0 := hz _ (by simp)
  rw [Flock.RecAlgebra.f128_eq_of_add_eq_zero hlast] at h₂
  obtain ⟨hopen, ⟨nc, hop⟩, hr⟩ := hR rsr rs k₂ lgr lg h₁ h₂
  exact levelTie_of_lig hf hsch h₂ (fun x hx => hz x (by simp [hx])) hne hcap hpm hpo hyr hic hqn hcn hcbn hopen
    hop hk hl hr ho

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.ligRun_of_lig
#print axioms FlockC3.RegLevels.levelTie_of_lig
#print axioms FlockC3.RegLevels.lgSteps_of_lig
