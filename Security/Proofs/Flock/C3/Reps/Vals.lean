import Proofs.Flock.C3.Reps.Tie
import Proofs.Flock.VBridge.Family.Runs

/-!
# The reps' value stubs `ProofOfVals` and `SetupOfVals`, from G's facts at holo

Two of the four parts `repUnitOfAlg_of_parts` takes at values `vals` of V*'s layers are G's facts at a rep, which read
no frame and so hold at a holo shape:

* `unitProof_of_from`, `proofOfVals_of_from`: the proof's part (`UnitProof`) from `ProofFrom`, `RunsFrom`'s fields at
  one rep that read no frame, and the claims the record answers, `claimsF` at the rep's values (as `run_of_frames`
  checks them). The ring switches' count, sizes and values are E's (`ringSwitch_of_residuals`, `decoded_rs_size`), and
  the claims' shapes the verifier run's points, sized without `NoEmpty` (`ofRounds_size`), which a holo rep's `h`
  round, drawing no coins, breaks. G's proof file carries the messages at any shape (`msgs_proofOf`).
* `unitSetup_of_from`, `setupOfVals_of_from`: the setup's part (`UnitSetup`) from `SetupRealizes` and `ShapeIs` at the
  run's statement and the values' setup and schedule (`SetupFrom`), with the schedule's model, the plain leaf's name
  and the extras at the link points (G's `ExtrasAre`).
-/

namespace FlockC3.Reps

noncomputable section

section parts

local notation "𝔽" => FlockLevel3.GF128

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Compiled
open FlockVBridge.Family (CombFolds proofOf proofOf_zc le_of_fast100)
open FlockVBridge.Runs (pair_append_get)

/-- `Coins.ofRounds` hands `n` coins wherever it reads a round of `n` (RecAlgebra.lean:40–45), empty rounds or not. -/
theorem ofRounds_size {rs : Array (Array F128)} {k n : ℕ} {c : Array F128}
    (h : (Flock.RecAlgebra.Coins.ofRounds rs).round k n = .ok c) : c.size = n := by
  simp only [Flock.RecAlgebra.Coins.ofRounds] at h
  split at h
  · split at h
    · cases h; assumption
    · cases h
  · cases h

/-- **G's `RunsFrom` at one rep, past its frames**: the setup's dimensions are the shape's and its schedule
`fast100`'s, the extras the rep's and well formed, the record's coins the rep's, the proof carries the messages w
holds, the comb port is the lincheck's fold (`CombFolds`), and the claims are `claimsF` at the verifier's rows. -/
structure ProofFrom (sh : Flock.Shape) (d : RepData) (v : UnitVals) : Prop where
  m : v.st.m = sh.m
  kLog : v.st.kLog = sh.kLog
  kLog_le : v.st.kLog ≤ v.st.m
  fast : Flock.fast100 v.st.m = some v.sch
  extras : d.extras = v.extras
  extras_wf : ∀ j < v.extras.size, v.extras[j]!.skip.size = 64 ∧ v.extras[j]!.x.size = v.st.m - 6
  coins : d.rounds = v.t.rounds.map (·.coins)
  zerocheck : ∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
    ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) v.p.zerocheck
  lincheck : LcProofOf (decoded sh d.w d.comb d.cons) v.p.lincheck
  ringSwitches : v.p.ringSwitches = (decoded sh d.w d.comb d.cons).rs
  foldPts : CombFolds v.st sh d
  claims : ∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
    v.claims = claimsF sh (valOf sh d.w d.comb d.cons) info d.extras

/-- **G's proof file carries the messages w holds** at any shape: `ProofFrom`'s `zerocheck`, `lincheck` and
`ringSwitches` at `proofOf`. -/
theorem msgs_proofOf (sh : Flock.Shape) (params : Flock.PcsParams) (cap : Array ByteArray)
    (lig : Flock.LigeritoProof) (d : RepData) :
    (∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons))
        (proofOf sh params cap lig d).zerocheck) ∧
    LcProofOf (decoded sh d.w d.comb d.cons) (proofOf sh params cap lig d).lincheck ∧
    (proofOf sh params cap lig d).ringSwitches = (decoded sh d.w d.comb d.cons).rs :=
  ⟨fun _ _ _ h => proofOf_zc h, ⟨rfl, rfl⟩, rfl⟩

/-- **The proof's part of a rep unit, from G's facts at the rep**, at values meeting E's preconditions. -/
theorem unitProof_of_from {sh : Flock.Shape} {d : RepData} {v : UnitVals} (halg : AlgebraHolds sh d)
    (hacc : AccHolds sh d) (h : ProofFrom sh d v) : UnitProof sh d v := by
  obtain ⟨vr, info, kappa, S, res, hrows, hS, hw, hc, hres, hzero⟩ := halg
  have hacc' := hacc vr info kappa hrows
  obtain ⟨hcl0, hval, -⟩ := ringSwitch_of_residuals hrows hS hw hc hacc' hres hzero
  obtain ⟨_, _, hrun, hres'⟩ := residuals_eq_run d.w d.comb d.cons hrows hS hw hc hacc'
  rw [hres] at hres'
  cases hres'
  obtain ⟨-, _, _, _, _, _, _, _, h0, h1, -, -, -⟩ := run_stages hrun
  obtain ⟨-, -, hszc⟩ := decoded_sizes sh d.w d.comb d.cons
  have hslc := decoded_lc_size sh d.w d.comb d.cons
  obtain ⟨hsrs, hsrs'⟩ := decoded_rs_size sh d.w d.comb d.cons
  have hmS := h.m
  have hkS := h.kLog
  have hkm := h.kLog_le
  have h22 := le_of_fast100 h.fast
  obtain ⟨_, c₁, _, -, hc₁, -, hr, -, hrho, -, -⟩ := zerocheck_coins (by omega : 14 ≤ sh.m) h0
  obtain ⟨_, _, _, -, -, -, -, -, -, -, -, hrIn, -, hts, -, -⟩ := lincheck_coins h1
  simp only [Info.map, ZcPts.map, LcPts.map] at hr hrho hrIn hts
  have hc₁s := ofRounds_size hc₁
  have hcl := h.claims vr info kappa hrows
  have hFs := claimsF_size sh (valOf sh d.w d.comb d.cons) info d.extras
  refine ⟨h.coins, h.zerocheck, h.lincheck, h.foldPts, ?_, fun j hj => ?_, fun j hj => ?_⟩
  · rw [h.ringSwitches, hsrs, hcl]
    exact hcl0
  · rw [hcl, hFs] at hj
    refine ⟨by rw [h.ringSwitches]; exact hsrs' j (by omega), ?_⟩
    have hex : (d.extras.map fun e => (⟨Val.cst e.value, e.skip, e.x⟩ : AClaim F128)).map AClaim.toClaim =
        d.extras := by
      rw [Array.map_map]
      exact Array.map_id'' (fun e => rfl) _
    rw [hcl]
    unfold claimsF claimsOf
    rw [Array.map_append, hex]
    simp only [ZcPts.map, LcPts.map, List.map_toArray, List.map_cons, List.map_nil, AClaim.toClaim]
    refine pair_append_get (P := fun c => c.skip.size = 64 ∧ c.x.size = v.st.m - 6)
      ⟨skipWeights_size _, ?_⟩ ⟨skipWeights_size _, ?_⟩ (by rw [h.extras]; exact h.extras_wf) j hj
    · show (info.lc.rIn ++ info.zc.rho.extract (sh.kLog - 6) (sh.m - 6)).size = v.st.m - 6
      simp only [Array.size_append, Array.size_extract, Array.size_reverse, hrIn, hts, hrho, hslc, hszc]
      omega
    · show info.zc.r.size = v.st.m - 6
      rw [hr, Array.size_append, FlockLevel3.pinned_size, hc₁s]
      omega
  · rw [hcl] at hj
    rw [h.ringSwitches, hcl]
    exact hval j (by omega)

/-- **G's setup facts at one rep's values**: the setup realizes `S` (`SetupRealizes`) at the values' schedule, which is
V*'s shape's (`ShapeIs`) and models `sch₀`; the values' leaf is the plain one `SetupRealizes` names; and the extras are
the setup's at every well-formed link point arrays at `pts` (G's `ExtrasAre`). -/
structure SetupFrom (S : FlockSoundness.Model.Statement) (sch₀ : FlockSoundness.Accounting.Schedule)
    (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → 𝔽) (sh : Flock.Shape) (v : UnitVals) : Prop where
  realizes : FlockVBridge.Family.SetupRealizes v.st S v.sch ptLocal mPts
  shape : FlockVBridge.Family.ShapeIs sh v.st v.sch
  sched : schedOf v.sch = sch₀
  leaf : v.msId = "flock-leaf/sha512-unsalted"
  extras : ∀ P, FlockVBridge.Family.PointsOK v.st S ptLocal mPts pts P → v.st.extra P = .ok v.extras

/-- **The setup's and schedule's part of a rep unit, from G's setup facts.** -/
theorem unitSetup_of_from {S : FlockSoundness.Model.Statement} {sch₀ : FlockSoundness.Accounting.Schedule}
    {ptLocal mPts : ℕ} {pts : Fin 2 → Fin mPts → 𝔽} {sh : Flock.Shape} {v : UnitVals}
    (h : SetupFrom S sch₀ ptLocal mPts pts sh v) :
    UnitSetup S sch₀ (FlockSoundness.Model.extraClaims S ptLocal mPts pts) sh v := by
  obtain ⟨hmS, hkS, hfold, hmk, hfast, -, -, hkL, hpts⟩ := h.realizes
  obtain ⟨P, e, he, hext, hes⟩ := hpts pts
  obtain rfl : e = v.extras := Except.ok.inj (he.symm.trans (h.extras P ⟨e, he, hext, hes⟩))
  obtain ⟨hlen, hrel⟩ := hext _ he
  exact ⟨h.shape.1.symm, hkL.trans h.shape.2.1.symm, hmS, hkS, hfold, hfast, h.sched, by rw [hmk, h.leaf], hlen,
    hrel⟩

end parts

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

section source

variable {S : ℕ} {Vs : Fin S → VStmt}

/-- **`ProofOfVals` from G's facts at each rep** (`ProofFrom`). -/
theorem proofOfVals_of_from {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData} {vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (h : AtRep zs dg N k Rw sh rep fun _ _ _ Lb i => ProofFrom sh (rep Lb i) (vals Lb i)) :
    ProofOfVals zs dg N k Rw sh rep vals := fun R ω t R₂ τ o hA i halg hacc =>
  unitProof_of_from halg hacc (h R ω t R₂ τ o hA i halg hacc)

/-- **`SetupOfVals` from G's setup facts at each rep** (`SetupFrom`) at the run's statement, schedule and link
points. -/
theorem setupOfVals_of_from {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (h : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      SetupFrom ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (src R ω t Lb).pts sh (vals Lb i)) :
    SetupOfVals zs dg N k Rw src sh rep vals := fun R ω t R₂ τ o hA i halg hacc =>
  unitSetup_of_from (h R ω t R₂ τ o hA i halg hacc)

end source

end

end FlockC3.Reps
