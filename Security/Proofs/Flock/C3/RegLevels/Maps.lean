import Proofs.Flock.C3.Reps.Maps
import Proofs.Flock.C3.RegLevels.RsT

/-!
# n7's levels at C3's decoder of R's openings

* `runAlpha_of_run`: the α of R's run on the messages w holds is `runAlpha`, at any consistency sums: the run maps to
  `skel`'s, values forgotten (`ringSwitch_sim`, `ligerito_sim` at `homUnit`, `map_skel`), with the same coin points.
* `opensUnits_of_maps`: `OpensUnits` at `StackedMaps.ops`, at the map's sessions and units and copy 0 (`OpensAtMaps`).
* `levelTie_of_maps`: `levelTie_of_reads` at `StackedMaps.ops`: its `hrs`, `hwd` (from the map's rows' widths) and `hq`,
  and `hR` by `hR_of_units`, its α by `runAlpha_of_run` and its units by `opensUnits_of_maps`. Left: the opening
  steps' coins (custody), the Ligerito steps' weave (the source's definition), each opening at its map's unit, and
  the run's and the proof's ties.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Refine (capAt)
open Flock (sibsOf)

/-- **(`hR`'s α) R's run's α is `runAlpha`**, whatever consistency sums the messages hold. -/
theorem runAlpha_of_run {sh : Flock.Shape} {w : Array Flock.F128} {cons : ℕ → ℕ → Flock.F128}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {k₀ k₁ k₂ : ℕ} {c : Array (Array Flock.F128)}
    {rsr : Array Flock.F128} {rs : Flock.RecAlgebra.RsPts Flock.F128} {lgr : Array Flock.F128}
    {lg : Flock.RecAlgebra.LgPts}
    (h₁ : Flock.RecAlgebra.ringSwitch (FlockC3.Opening.decodedO sh w cons) a k₀ (Flock.RecAlgebra.CoinSource.ofRounds c) =
      .ok ((rsr, rs), k₂))
    (h₂ : Flock.RecAlgebra.ligerito sh.sched (FlockC3.Opening.decodedO sh w cons) rs.T rs.sws k₂
      (Flock.RecAlgebra.CoinSource.ofRounds c) = .ok ((lgr, lg), k₁)) (ℓ : ℕ) :
    FlockC3.Reps.runAlpha sh (a.map (Flock.RecAlgebra.AClaim.map fun _ => ())) k₀ c ℓ = (lg.conss[ℓ]!).alpha := by
  have e : (FlockC3.Opening.decodedO sh w cons).map (fun _ : Flock.F128 => ()) = FlockC3.Reps.skel sh :=
    FlockC3.Reps.map_skel sh _
  have h₁' := Flock.RecAlgebra.ringSwitch_sim FlockC3.Reps.homUnit _ a k₀ _ _ _ h₁
  have h₂' := Flock.RecAlgebra.ligerito_sim FlockC3.Reps.homUnit sh.sched _ rs.T rs.sws k₂ _ _ _ h₂
  rw [e] at h₁' h₂'
  simp only [FlockC3.Reps.runAlpha, h₁', Flock.RecAlgebra.RsPts.map, h₂']

section Maps

variable {S : ℕ} {stmt : Fin S → VStar.Stmt} {Vs : Fin S → VStmt} {Reg' : Type}

/-- **Each opening of R's levels is its map's unit's**: opening `j` of level `ℓ` is copy 0 of unit `M.un ℓ j` in session
`M.sess ℓ`, opening `M.row` at the run's position through its siblings to its cap entry (`OpensUnits` at the map). -/
def OpensAtMaps (M : FlockC3.Reps.StackedMaps stmt Vs) (zs : (s : Fin S) → (Vs s).Outer Reg')
    (Lb : (s : Fin S) → (Vs s).Bits) (sh : Flock.Shape) (sibs : ℕ → Array ByteArray) (cap : Array ByteArray)
    (p : Flock.LigeritoProof) (lg : Flock.RecAlgebra.LgPts) : Prop :=
  ∀ ℓ < sh.sched.levels.size, ∀ j < (sh.sched.levels[ℓ]!).q, ∃ a cf A,
    OpensTopAt (zs (M.sess ℓ)) (Lb (M.sess ℓ)) (M.un ℓ j) 0 (FlockVBridge.AccChain.levelLanes sh ℓ) (M.row Lb ℓ j) a cf
      A ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) ((lg.conss[ℓ]!).pos[j]!)
      (sibsOf (sibs ℓ) ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) j)
      ((capAt cap p ℓ).getD ((lg.conss[ℓ]!).pos[j]! >>> ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
        ByteArray.empty)

theorem opensUnits_of_maps (M : FlockC3.Reps.StackedMaps stmt Vs) {zs : (s : Fin S) → (Vs s).Outer Reg'}
    {Lb : (s : Fin S) → (Vs s).Bits} {sh : Flock.Shape} {sibs : ℕ → Array ByteArray} {cap : Array ByteArray}
    {p : Flock.LigeritoProof} {lg : Flock.RecAlgebra.LgPts} {kap : Array (Array (Array Flock.F128))}
    {cl : Array (Flock.RecAlgebra.AClaim Unit)} {k₀ : ℕ} {c : Array (Array Flock.F128)}
    (h : OpensAtMaps M zs Lb sh sibs cap p lg) : OpensUnits zs Lb sh (M.ops sh Lb kap cl k₀ c) sibs cap p lg :=
  ⟨M.sess, M.un, fun _ _ => 0, fun ℓ hℓ j hj => by rw [M.rows_get sh Lb kap cl k₀ c hj]; exact h ℓ hℓ j hj⟩

/-- **(n7's levels) `LevelTie` at C3's decoder of R's openings**: `levelTie_of_reads` at `ops := M.ops`, its rows'
count and widths (`rows_size`, `rows_get`, `hwd`), its `q` and, at the run's points, its α (`runAlpha_of_run`) by
construction, `OpenRel` by `hR_of_units` and the openings' units by `opensUnits_of_maps`. Left: the opening steps'
coins (`hcoin`), the Ligerito steps woven from the proof's messages and `c`'s rounds from the ring switches' `k₂`
(`hR`'s second conjunct, `lgSteps_of_lig`'s), each opening at its map's unit (`OpensAtMaps`), and the run's, the
claims' and the proof's ties. -/
theorem levelTie_of_maps (M : FlockC3.Reps.StackedMaps stmt Vs) (zs : (s : Fin S) → (Vs s).Outer Reg')
    (Lb : (s : Fin S) → (Vs s).Bits) (hG : ∀ s, (zs s).Good (Lb s)) {st : Flock.Stmt} {hp : HoloParams}
    {cap₀ : Cap Dig} {u : C3Run (holoAt st hp)} {sh : Flock.Shape} {w : Array Flock.F128}
    {kap : Array (Array (Array Flock.F128))} {a : Array (Flock.RecAlgebra.AClaim Flock.F128)}
    {c : Array (Array Flock.F128)} {k₀ k₁ : ℕ} {res : Array Flock.F128} {ops : FlockVBridge.AccChain.Openings}
    (hops : ops = M.ops sh Lb kap (a.map (Flock.RecAlgebra.AClaim.map fun _ => ())) k₀ c)
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some sh)
    (hrun : Flock.RecOpening.run sh (msgsO sh w ops) a k₀ (Flock.RecAlgebra.CoinSource.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c) (e0 : ℕ → ByteArray) (sb : ℕ → ℕ → ℕ → ByteArray)
    (e : ℕ → ℕ → ByteArray) (hcap : capDig (capR sh e0) = cap₀)
    (hwd : ∀ ℓ < sh.sched.levels.size, ∀ j < (sh.sched.levels[ℓ]!).q,
      (M.row Lb ℓ j).size = FlockVBridge.AccChain.levelLanes sh ℓ) (ha : a.size = 2)
    (hshape : ∀ k < 2, ((msgsO sh w ops).rs[k]!).size = 128 ∧
      ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).skip.size = 64 ∧
      ((a.map Flock.RecAlgebra.AClaim.toClaim)[k]!).x.size = (holoAt st hp).cls.shape.mReg - 6)
    (hcl : ∀ i : Fin 2,
      ClaimRel (holoAt st hp).cls.shape.mReg (a.map Flock.RecAlgebra.AClaim.toClaim)[i.val]! (u.claim i).toClaim)
    (hpl : PlaysT (FlockC3.Opening.openingAt u) u.op (some u.opened)) (htie : FlockC3.Opening.OpenTie sh w ops u)
    (hcoin : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.CoinSource.ofRounds c) = .ok ((rsr, rs), k₂) →
      u.op.filterMap Sum.getRight? = ((List.range' k₀ (k₂ - k₀)).map fun i => c[i]!).map (coinArr fOf))
    (hR : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.CoinSource.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) rs.T rs.sws k₂ (Flock.RecAlgebra.CoinSource.ofRounds c) =
        .ok ((lgr, lg), k₁) →
      OpensAtMaps M zs Lb sh (sibsR sh ops sb e) (capR sh e0) (pR sh e0 w ops sb e) lg ∧
      u.lg = FlockSoundness.Discharge.Exec.weave (ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch)
        (ligMsgs (pR sh e0 w ops sb e) capDig sh.sched)
        (((List.range' k₂ (c.size - k₂)).map (fun i => c[i]!)).map (coinArr fOf)))
    (ho : ∀ ℓ j, u.os.1 2 ℓ j = openRow (pR sh e0 w ops sb e) sh.sched ℓ j) :
    LevelTie cap₀ (holoAt st hp) u := by
  subst hops
  exact levelTie_of_reads zs Lb hG hell hsh hrun hz hne "" e0 sb e hcap (fun ℓ _ => M.rows_size sh Lb kap _ k₀ c ℓ)
    (fun ℓ hℓ j hj => (congrArg Array.size (M.rows_get sh Lb kap _ k₀ c hj)).trans (hwd ℓ hℓ j hj)) (fun _ _ => rfl)
    (hR_of_units hsh hrun hz hne ha hshape hcl hpl htie hcoin fun rsr rs k₂ lgr lg h₁ h₂ =>
      ⟨fun ℓ _ => runAlpha_of_run h₁ h₂ ℓ, opensUnits_of_maps M (hR rsr rs k₂ lgr lg h₁ h₂).1,
        (hR rsr rs k₂ lgr lg h₁ h₂).2⟩) ho

end Maps

end FlockC3.RegLevels
