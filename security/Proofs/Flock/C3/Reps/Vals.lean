module

public import Proofs.Flock.C3.Reps.Tie
public import Proofs.Flock.VBridge.Family.Runs
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Refine.Arith
import Proofs.Flock.VBridge.Runs.Rs

/-! The reps' value stubs `ProofOfVals` and `SetupOfVals`, from G's facts at holo. -/

@[expose] public section

namespace FlockC3.Reps

noncomputable section

section parts

local notation "𝔽" => FlockLevel3.GF128

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Compiled
open FlockVBridge.Family (CombFolds proofOf proofOf_zc le_of_fast100)
open FlockVBridge.Runs (pair_append_get)

/-- It holds without `NoEmpty`, which a holo rep's `h` round, flipping no coins, breaks. -/
theorem ofRounds_size {rs : Array (Array F128)} {k n : ℕ} {c : Array F128}
    (h : (Flock.RecAlgebra.RoundCoins.ofRounds rs).round k n = .ok c) : c.size = n := by
  simp only [Flock.RecAlgebra.RoundCoins.ofRounds] at h
  split at h
  · split at h
    · cases h; assumption
    · cases h
  · cases h

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

theorem msgs_proofOf (sh : Flock.Shape) (params : Flock.PcsParams) (cap : Array ByteArray)
    (lig : Flock.LigeritoProof) (d : RepData) :
    (∀ w info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons))
        (proofOf sh params cap lig d).zerocheck) ∧
    LcProofOf (decoded sh d.w d.comb d.cons) (proofOf sh params cap lig d).lincheck ∧
    (proofOf sh params cap lig d).ringSwitches = (decoded sh d.w d.comb d.cons).rs :=
  ⟨fun _ _ _ h => proofOf_zc h, ⟨rfl, rfl⟩, rfl⟩

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

structure SetupFrom (S : FlockSoundness.Model.Statement) (sch₀ : FlockSoundness.Accounting.Schedule)
    (ptLocal mPts : ℕ) (pts : Fin 2 → Fin mPts → 𝔽) (sh : Flock.Shape) (v : UnitVals) : Prop where
  realizes : FlockVBridge.Family.SetupRealizes v.st S v.sch ptLocal mPts
  shape : FlockVBridge.Family.ShapeIs sh v.st v.sch
  sched : schedOf v.sch = sch₀
  leaf : v.msId = "flock-leaf/sha512-unsalted"
  extras : ∀ P, FlockVBridge.Family.PointsOK v.st S ptLocal mPts pts P → v.st.extra P = .ok v.extras

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

def claimsAt (sh : Flock.Shape) (d : RepData) : Array Flock.Claim :=
  match verifierRows sh d.rounds d.extras d.comb with
  | .ok (_, info, _) => claimsF sh (valOf sh d.w d.comb d.cons) info d.extras
  | .error _ => #[]

theorem claimsAt_eq {sh : Flock.Shape} {d : RepData} {w info kappa}
    (h : verifierRows sh d.rounds d.extras d.comb = .ok (w, info, kappa)) :
    claimsAt sh d = claimsF sh (valOf sh d.w d.comb d.cons) info d.extras := by
  unfold claimsAt
  rw [h]

structure ValsSrc where
  st : Flock.Setup
  sch : Flock.Schedule
  params : Flock.PcsParams
  cap : Array ByteArray
  lig : Flock.LigeritoProof
  dom : ByteArray
  fr : ℕ → List UInt8
  rpp : ℕ → Array Flock.F128
  gamma : Array Flock.F128
  t₄ : Flock.Transcript

def ValsSrc.vals (b : ValsSrc) (sh : Flock.Shape) (d : RepData) : UnitVals :=
  ⟨b.st, "flock-leaf/sha512-unsalted", b.sch, proofOf sh b.params b.cap b.lig d, d.extras,
    Flock.Transcript.init (FlockVBridge.Family.streamRounds d.rounds b.fr) b.dom, claimsAt sh d, b.rpp, b.gamma,
    b.t₄⟩

def RepExtrasAt (st : Flock.Setup) (S : FlockSoundness.Model.Statement) (ptLocal mPts : ℕ)
    (pts : Fin 2 → Fin mPts → 𝔽) (d : RepData) : Prop :=
  ∃ P, FlockVBridge.Family.PointsOK st S ptLocal mPts pts P ∧ st.extra P = .ok d.extras

theorem extras_of_at {st : Flock.Setup} {S : FlockSoundness.Model.Statement} {ptLocal mPts : ℕ}
    {pts : Fin 2 → Fin mPts → 𝔽} {d : RepData} (h : RepExtrasAt st S ptLocal mPts pts d) :
    ∀ P, FlockVBridge.Family.PointsOK st S ptLocal mPts pts P → st.extra P = .ok d.extras := by
  obtain ⟨P₀, h₀, he⟩ := h
  intro P hP
  rw [FlockVBridge.Family.extras_unique hP h₀, he]

theorem ValsSrc.proofFrom (b : ValsSrc) {S : FlockSoundness.Model.Statement} {ptLocal mPts : ℕ}
    {pts : Fin 2 → Fin mPts → 𝔽} {sh : Flock.Shape} {d : RepData}
    (hset : FlockVBridge.Family.SetupRealizes b.st S b.sch ptLocal mPts)
    (hsh : FlockVBridge.Family.ShapeIs sh b.st b.sch) (hx : RepExtrasAt b.st S ptLocal mPts pts d)
    (hfold : CombFolds b.st sh d) : ProofFrom sh d (b.vals sh d) := by
  obtain ⟨-, -, -, -, hfast, -, hkm, -, -⟩ := hset
  obtain ⟨_, ⟨e, he, -, hes⟩, he'⟩ := hx
  obtain rfl : e = d.extras := Except.ok.inj (he.symm.trans he')
  have hm := msgs_proofOf sh b.params b.cap b.lig d
  exact
    { m := hsh.1.symm
      kLog := hsh.2.1.symm
      kLog_le := hkm
      fast := hfast
      extras := rfl
      extras_wf := hes
      coins := (FlockVBridge.Family.streamRounds_coins d.rounds b.fr).symm
      zerocheck := hm.1
      lincheck := hm.2.1
      ringSwitches := hm.2.2
      foldPts := hfold
      claims := fun _ _ _ h => claimsAt_eq h }

theorem ValsSrc.setupFrom (b : ValsSrc) {S : FlockSoundness.Model.Statement}
    {sch₀ : FlockSoundness.Accounting.Schedule} {ptLocal mPts : ℕ} {pts : Fin 2 → Fin mPts → 𝔽} {sh : Flock.Shape}
    {d : RepData} (hset : FlockVBridge.Family.SetupRealizes b.st S b.sch ptLocal mPts)
    (hsh : FlockVBridge.Family.ShapeIs sh b.st b.sch) (hsched : schedOf b.sch = sch₀)
    (hx : RepExtrasAt b.st S ptLocal mPts pts d) : SetupFrom S sch₀ ptLocal mPts pts sh (b.vals sh d) :=
  ⟨hset, hsh, hsched, rfl, extras_of_at hx⟩

end parts

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

theorem sched_holoAt
    (h : FlockSoundness.Discharge.Recursion.ClassOk (FlockSoundness.Discharge.Recursion.shapeOf st))
    {s : Flock.Schedule} (hs : Flock.fast100 (holoAt st hp).cls.m = some s) : schedOf s = (holoAt st hp).sch := by
  have e : FlockSoundness.Discharge.Recursion.execClassAt st =
      FlockSoundness.Discharge.Recursion.ExecClass.ofStmt st h := dite_eq_left h
  have hm : (holoAt st hp).cls.m = st.m := by
    show (FlockSoundness.Discharge.Recursion.execClassAt st).cls.m = st.m
    exact congrArg (fun c : FlockSoundness.Discharge.Recursion.ExecClass => c.cls.m) e
  rw [hm] at hs
  exact (FlockSoundness.Discharge.Recursion.execClassAt_sch_of_fast100 h hs).symm

theorem ValsSrc.setupFrom_holo (b : ValsSrc)
    (hcls : FlockSoundness.Discharge.Recursion.ClassOk (FlockSoundness.Discharge.Recursion.shapeOf st))
    {pts : Fin 2 → Fin (holoAt st hp).cls.mPts → GF128} {sh : Flock.Shape} {d : RepData}
    (hset : FlockVBridge.Family.SetupRealizes b.st ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) b.sch
      (holoAt st hp).cls.ptLocal (holoAt st hp).cls.mPts)
    (hsh : FlockVBridge.Family.ShapeIs sh b.st b.sch)
    (hx : RepExtrasAt b.st ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
      (holoAt st hp).cls.mPts pts d) :
    SetupFrom ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (holoAt st hp).cls.ptLocal
      (holoAt st hp).cls.mPts pts sh (b.vals sh d) := by
  refine b.setupFrom hset hsh (sched_holoAt hcls ?_) hx
  obtain ⟨hmS, -, -, -, hfast, -⟩ := hset
  rw [← show b.st.m = (holoAt st hp).cls.m from hmS]
  exact hfast

section source

variable {S : ℕ} {Vs : Fin S → VStmt}

theorem proofOfVals_of_from {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (h : AtRep zs dg N k Rw sh rep fun R ω t Lb i => ProofFrom sh (rep R ω t Lb i) (vals R ω t Lb i)) :
    ProofOfVals zs dg N k Rw sh rep vals := fun R ω t R₂ τ o hA i halg hacc =>
  unitProof_of_from halg hacc (h R ω t R₂ τ o hA i halg hacc)

theorem setupOfVals_of_from {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals}
    (h : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      SetupFrom ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).sch (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (src R ω t Lb).pts sh (vals R ω t Lb i)) :
    SetupOfVals zs dg N k Rw src sh rep vals := fun R ω t R₂ τ o hA i halg hacc =>
  unitSetup_of_from (h R ω t R₂ τ o hA i halg hacc)

theorem proofOfVals_of_src {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (hset : AtRep zs dg N k Rw sh rep fun _ _ _ Lb i =>
      FlockVBridge.Family.SetupRealizes (b Lb i).st ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
        (b Lb i).sch (holoAt st hp).cls.ptLocal (holoAt st hp).cls.mPts)
    (hsh : AtRep zs dg N k Rw sh rep fun _ _ _ Lb i => FlockVBridge.Family.ShapeIs sh (b Lb i).st (b Lb i).sch)
    (hx : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      RepExtrasAt (b Lb i).st ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (src R ω t Lb).pts (rep R ω t Lb i))
    (hfold : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      FlockVBridge.Family.CombFolds (b Lb i).st sh (rep R ω t Lb i)) :
    ProofOfVals zs dg N k Rw sh rep fun R ω t Lb i => (b Lb i).vals sh (rep R ω t Lb i) :=
  fun R ω t R₂ τ o hA i halg hacc => unitProof_of_from halg hacc
    (ValsSrc.proofFrom _ (hset R ω t R₂ τ o hA i halg hacc) (hsh R ω t R₂ τ o hA i halg hacc)
      (hx R ω t R₂ τ o hA i halg hacc) (hfold R ω t R₂ τ o hA i halg hacc))

theorem setupOfVals_of_src {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape} {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData}
    {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (hcls : FlockSoundness.Discharge.Recursion.ClassOk (FlockSoundness.Discharge.Recursion.shapeOf st))
    (hset : AtRep zs dg N k Rw sh rep fun _ _ _ Lb i =>
      FlockVBridge.Family.SetupRealizes (b Lb i).st ((holoAt st hp).cls.pubStatement (holoAt st hp).regs)
        (b Lb i).sch (holoAt st hp).cls.ptLocal (holoAt st hp).cls.mPts)
    (hsh : AtRep zs dg N k Rw sh rep fun _ _ _ Lb i => FlockVBridge.Family.ShapeIs sh (b Lb i).st (b Lb i).sch)
    (hx : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      RepExtrasAt (b Lb i).st ((holoAt st hp).cls.pubStatement (holoAt st hp).regs) (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts (src R ω t Lb).pts (rep R ω t Lb i)) :
    SetupOfVals zs dg N k Rw src sh rep fun R ω t Lb i => (b Lb i).vals sh (rep R ω t Lb i) :=
  fun R ω t R₂ τ o hA i halg hacc => unitSetup_of_from
    (ValsSrc.setupFrom_holo _ hcls (hset R ω t R₂ τ o hA i halg hacc) (hsh R ω t R₂ τ o hA i halg hacc)
      (hx R ω t R₂ τ o hA i halg hacc))

end source

end

end FlockC3.Reps
