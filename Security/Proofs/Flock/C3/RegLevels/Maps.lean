import Proofs.Flock.C3.Reps.Maps
import Proofs.Flock.C3.RegLevels.SrcUnits

/-!
# n7's levels at C3's decoder of R's openings

* `runAlpha_of_run`: the α of R's run on the messages w holds is `runAlpha`, at any consistency sums: the run maps to
  `skel`'s, values forgotten (`ringSwitch_sim`, `ligerito_sim` at `homUnit`, `map_skel`), with the same coin points.
* `levelTie_of_maps`: `levelTie_of_units` at `StackedMaps.ops`, its `hrs`, `hq` and `hR`'s α supplied.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Refine (openAt)

/-- **(`hR`'s α) R's run's α is `runAlpha`**, whatever consistency sums the messages hold. -/
theorem runAlpha_of_run {sh : Flock.Shape} {w : Array Flock.F128} {cons : ℕ → ℕ → Flock.F128}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {k₀ k₁ k₂ : ℕ} {c : Array (Array Flock.F128)}
    {rsr : Array Flock.F128} {rs : Flock.RecAlgebra.RsPts Flock.F128} {lgr : Array Flock.F128}
    {lg : Flock.RecAlgebra.LgPts}
    (h₁ : Flock.RecAlgebra.ringSwitch (FlockC3.Opening.decodedO sh w cons) a k₀ (Flock.RecAlgebra.Coins.ofRounds c) =
      .ok ((rsr, rs), k₂))
    (h₂ : Flock.RecAlgebra.ligerito sh.sched (FlockC3.Opening.decodedO sh w cons) rs.T rs.sws k₂
      (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((lgr, lg), k₁)) (ℓ : ℕ) :
    FlockC3.Reps.runAlpha sh (a.map (Flock.RecAlgebra.AClaim.map fun _ => ())) k₀ c ℓ = (lg.conss[ℓ]!).alpha := by
  have e : (FlockC3.Opening.decodedO sh w cons).map (fun _ : Flock.F128 => ()) = FlockC3.Reps.skel sh :=
    FlockC3.Reps.map_skel sh _
  have h₁' := Flock.RecAlgebra.ringSwitch_sim FlockC3.Reps.homUnit _ a k₀ _ _ _ h₁
  have h₂' := Flock.RecAlgebra.ligerito_sim FlockC3.Reps.homUnit sh.sched _ rs.T rs.sws k₂ _ _ _ h₂
  rw [e] at h₁' h₂'
  simp only [FlockC3.Reps.runAlpha, h₁', Flock.RecAlgebra.RsPts.map, h₂']

/-- **(n7's levels) `LevelTie` at C3's decoder of R's openings**: `levelTie_of_units` at `ops := M.ops`, its rows' count
(`rows_size`), its `q` and, at the run's points, its α (`runAlpha_of_run`) by construction. -/
theorem levelTie_of_maps {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} {Reg' : Type}
    (M : FlockC3.Reps.StackedMaps stmt Vs) (zs : (s : Fin S) → (Vs s).Outer Reg') (Lb : (s : Fin S) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (Lb s)) {st : Flock.Stmt} {hp : HoloParams} {cap₀ : Cap Dig} {u : C3Run (holoAt st hp)}
    {cl : ℕ} {sh : Flock.Shape} {w : Array Flock.F128} {kap : Array (Array (Array Flock.F128))}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {c : Array (Array Flock.F128)} {k₀ k₁ : ℕ}
    {res : Array Flock.F128} {ops : FlockVBridge.AccChain.Openings}
    (hops : ops = M.ops sh Lb kap (a.map (Flock.RecAlgebra.AClaim.map fun _ => ())) k₀ c)
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg cl = some sh)
    (hrun : Flock.RecOpening.run sh (msgsO sh w ops) a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c)
    {cap : Array ByteArray} {p : Flock.LigeritoProof} (hcap : capDig cap = cap₀)
    (hpm : p.messages = (msgsO sh w ops).msgs.map fun mm => (Flock.RecAlgebra.kv mm.1, Flock.RecAlgebra.kv mm.2))
    (hpo : p.oodValues = (msgsO sh w ops).ood) (hyr : p.finalProof.yr = (msgsO sh w ops).yr)
    (hic : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = sh.sched.levels.size) (hcn : p.consistencyNonces.size = sh.sched.levels.size)
    (hcbn : p.claimNonces.size = (msgsO sh w ops).ood.size + sh.sched.levels.size)
    (hcaps : FlockVBridge.Family.CapsAt sh.sched cap p) (sibs : ℕ → Array ByteArray)
    (hO : ∀ ℓ < sh.sched.levels.size, openAt p (sh.sched.levels.size - 1) ℓ = (ops.rows ℓ, sibs ℓ, #[]))
    (hss : ∀ ℓ < sh.sched.levels.size,
      (sibs ℓ).size = (sh.sched.levels[ℓ]!).q * ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
    (h64 : ∀ ℓ < sh.sched.levels.size, ∀ x ∈ sibs ℓ, x.size = 64)
    (hR : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.Coins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) rs.T rs.sws k₂ (Flock.RecAlgebra.Coins.ofRounds c) =
        .ok ((lgr, lg), k₁) →
      OpensUnits zs Lb sh ops sibs cap p lg ∧ (∃ nc, OpenRel sh.sched.m nc rs.T rs.sws u.opened) ∧
      u.lg.filterMap Sum.getRight? = ((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf))
    (hk : u.lg.map kindOf <+: ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch)
    (hl : u.lg.filterMap Sum.getLeft? = ligMsgs p capDig sh.sched)
    (ho : ∀ ℓ j, u.os.1 2 ℓ j = openRow p sh.sched ℓ j) :
    LevelTie cap₀ (holoAt st hp) u := by
  subst hops
  exact levelTie_of_units zs Lb hG (msId := "") hell hsh hrun hz hne hcap hpm hpo hyr hic hqn hcn hcbn hcaps sibs hO
    (fun ℓ _ => M.rows_size sh Lb kap _ k₀ c ℓ) hss h64 (fun _ _ => rfl)
    (fun rsr rs k₂ lgr lg h₁ h₂ => ⟨fun ℓ _ => runAlpha_of_run h₁ h₂ ℓ, hR rsr rs k₂ lgr lg h₁ h₂⟩) hk hl ho

end FlockC3.RegLevels
