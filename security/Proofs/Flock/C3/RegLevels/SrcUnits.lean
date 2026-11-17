module

public import Proofs.Flock.C3.RegLevels.SrcConss
public import Proofs.Flock.C3.RegLevels.SrcOpened
public import Proofs.Flock.C3.RegLevels.SrcRun
public import Proofs.Flock.C3.RegLevels.SrcTie
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
import Proofs.Flock.Soundness.Refine.Seam

/-! `levelTie_of_run`'s `Opened` from R's units. -/

@[expose] public section

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose
open FlockSoundness.Discharge.PlaysCoin (coinArr)
open FlockSoundness.Refine (capAt openAt)
open Flock (sibsOf)

abbrev msgsO (sh : Flock.Shape) (w : Array Flock.F128) (ops : FlockVBridge.AccChain.Openings) :
    Flock.RecAlgebra.Messages Flock.F128 :=
  FlockC3.Opening.decodedO sh w (FlockC3.Opening.consO ops)

section Units

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

def OpensUnits (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits) (sh : Flock.Shape)
    (ops : FlockVBridge.AccChain.Openings) (sibs : ℕ → Array ByteArray) (cap : Array ByteArray)
    (p : Flock.LigeritoProof) (lg : Flock.RecAlgebra.LgPts) : Prop :=
  ∃ (sess : ℕ → Fin nS) (un : (ℓ : ℕ) → ℕ → Fin (Vs (sess ℓ)).nv) (cp : ℕ → ℕ → ℕ),
    ∀ ℓ < sh.sched.levels.size, ∀ j < (sh.sched.levels[ℓ]!).q, ∃ a cf A,
      OpensTopAt (zs (sess ℓ)) (Lb (sess ℓ)) (un ℓ j) (cp ℓ j) (FlockVBridge.AccChain.levelLanes sh ℓ)
        (ops.rows ℓ)[j]! a cf A ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) ((lg.conss[ℓ]!).pos[j]!)
        (sibsOf (sibs ℓ) ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap) j)
        ((capAt cap p ℓ).getD ((lg.conss[ℓ]!).pos[j]! >>> ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
          ByteArray.empty)

theorem levelTie_of_units (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (Lb s)) {st : Flock.Stmt} {hp : HoloParams} {cap₀ : Cap Dig} {u : C3Run (holoAt st hp)}
    {cl : ℕ} {sh : Flock.Shape} {w : Array Flock.F128} {ops : FlockVBridge.AccChain.Openings}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {c : Array (Array Flock.F128)} {k₀ k₁ : ℕ}
    {res : Array Flock.F128} (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg cl = some sh)
    (hrun : Flock.RecOpening.run sh (msgsO sh w ops) a k₀ (Flock.RecAlgebra.RoundCoins.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c)
    {msId : String} {cap : Array ByteArray} {p : Flock.LigeritoProof} (hcap : capDig cap = cap₀)
    (hpm : p.messages = (msgsO sh w ops).msgs.map fun mm => (Flock.RecAlgebra.kv mm.1, Flock.RecAlgebra.kv mm.2))
    (hpo : p.oodValues = (msgsO sh w ops).ood) (hyr : p.finalProof.yr = (msgsO sh w ops).yr)
    (hic : (p.initialCap == cap) = true)
    (hqn : p.queryNonces.size = sh.sched.levels.size) (hcn : p.consistencyNonces.size = sh.sched.levels.size)
    (hcbn : p.claimNonces.size = (msgsO sh w ops).ood.size + sh.sched.levels.size)
    (hcaps : FlockVBridge.Family.CapsAt sh.sched cap p) (sibs : ℕ → Array ByteArray)
    (hO : ∀ ℓ < sh.sched.levels.size, openAt p (sh.sched.levels.size - 1) ℓ = (ops.rows ℓ, sibs ℓ, #[]))
    (hrs : ∀ ℓ < sh.sched.levels.size, (ops.rows ℓ).size = (sh.sched.levels[ℓ]!).q)
    (hss : ∀ ℓ < sh.sched.levels.size,
      (sibs ℓ).size = (sh.sched.levels[ℓ]!).q * ((sh.sched.levels[ℓ]!).d - (sh.sched.levels[ℓ]!).cap))
    (h64 : ∀ ℓ < sh.sched.levels.size, ∀ x ∈ sibs ℓ, x.size = 64)
    (hq : ∀ ℓ < sh.sched.levels.size, ops.q ℓ = (sh.sched.levels[ℓ]!).q)
    (hR : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.RoundCoins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) rs.T rs.sws k₂ (Flock.RecAlgebra.RoundCoins.ofRounds c) =
        .ok ((lgr, lg), k₁) →
      (∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha) ∧ OpensUnits zs Lb sh ops sibs cap p lg ∧
      (∃ nc, OpenRel sh.sched.m nc rs.T rs.sws u.opened) ∧
      u.lg = FlockSoundness.Discharge.Exec.weave (ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch)
        (ligMsgs p capDig sh.sched) (((List.range' k₂ (c.size - k₂)).map (fun i => (c)[i]!)).map (coinArr fOf)))
    (ho : ∀ ℓ j, u.os.1 2 ℓ j = openRow p sh.sched ℓ j) :
    LevelTie cap₀ (holoAt st hp) u := by
  obtain ⟨hf, hsch⟩ := schedOf_holoAt hell hsh
  obtain ⟨rsr, rs, k₂, lgr, lg, h₁, h₂, rfl⟩ := opening_parts hrun
  have hlast : Flock.RecAlgebra.Val.add ((msgsO sh w ops).tAtom.getD rs.T) rs.T = 0 := hz _ (by simp)
  rw [Flock.RecAlgebra.f128_eq_of_add_eq_zero hlast] at h₂
  have hzl : ∀ x ∈ lgr, x = 0 := fun x hx => hz x (by simp [hx])
  obtain ⟨halpha, ⟨sess, un, cp, hu⟩, ⟨nc, hop⟩, hw⟩ := hR rsr rs k₂ lgr lg h₁ h₂
  have hopen := opened_of_units zs Lb hG (sh := sh) (w := w) (ops := ops) (msId := msId)
    (conssAt_of_lig hf h₂ hzl hne) hcaps sibs hO hrs hss h64 hq halpha sess un cp hu
  obtain ⟨hk, hl, hr⟩ := lgSteps_of_lig hf hsch h₂ hzl hne hpm hpo hyr hic hqn hcn hcbn hopen hw
  exact levelTie_of_lig hf hsch h₂ hzl hne hcap hpm hpo hyr hic hqn hcn hcbn hopen hop hk hl hr ho

end Units

local instance lawfulBEqByteArray : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

def readR (ops : FlockVBridge.AccChain.Openings) (sb : ℕ → ℕ → ℕ → ByteArray) (e : ℕ → ℕ → ByteArray) :
    FlockVBridge.Family.LigRead :=
  ⟨fun ℓ j => (ops.rows ℓ)[j]!, sb, fun _ _ => ByteArray.empty, e⟩

def proofR (sch : Flock.Schedule) (cap : Array ByteArray) (M : Flock.RecAlgebra.Messages Flock.F128)
    (x : FlockVBridge.Family.LigRead) : Flock.LigeritoProof :=
  { FlockVBridge.Family.ligOfRead 64 none sch x with
    initialCap := cap
    messages := M.msgs.map fun mm => (Flock.RecAlgebra.kv mm.1, Flock.RecAlgebra.kv mm.2)
    oodValues := M.ood
    finalProof := { (FlockVBridge.Family.ligOfRead 64 none sch x).finalProof with yr := M.yr }
    queryNonces := Array.replicate sch.levels.size 0
    consistencyNonces := Array.replicate sch.levels.size 0
    claimNonces := Array.replicate (M.ood.size + sch.levels.size) 0 }

theorem openAt_proofR (sch : Flock.Schedule) (cap : Array ByteArray) (M : Flock.RecAlgebra.Messages Flock.F128)
    (x : FlockVBridge.Family.LigRead) (R ℓ : ℕ) :
    openAt (proofR sch cap M x) R ℓ = openAt (FlockVBridge.Family.ligOfRead 64 none sch x) R ℓ := rfl

theorem capAt_proofR (sch : Flock.Schedule) (cap : Array ByteArray) (M : Flock.RecAlgebra.Messages Flock.F128)
    (x : FlockVBridge.Family.LigRead) (cap' : Array ByteArray) (ℓ : ℕ) :
    capAt cap' (proofR sch cap M x) ℓ = capAt cap' (FlockVBridge.Family.ligOfRead 64 none sch x) ℓ := rfl

theorem capsAt_proofR {sch : Flock.Schedule} {cap : Array ByteArray} {M : Flock.RecAlgebra.Messages Flock.F128}
    (x : FlockVBridge.Family.LigRead) (e0 : ℕ → ByteArray) :
    FlockVBridge.Family.CapsAt sch (FlockVBridge.Family.capArr 64 (sch.levels.getD 0 default).cap e0)
      (proofR sch cap M x) := by
  intro ℓ hℓ
  rw [capAt_proofR]
  exact FlockVBridge.Family.capsAt_ligOfRead none x e0 ℓ hℓ

theorem openAt_proofR_eq {sh : Flock.Shape} {cap : Array ByteArray} {M : Flock.RecAlgebra.Messages Flock.F128}
    {ops : FlockVBridge.AccChain.Openings} {sb : ℕ → ℕ → ℕ → ByteArray} {e : ℕ → ℕ → ByteArray}
    (hl4 : ∀ i < sh.sched.levels.size, 1 ≤ i → (sh.sched.levels.getD i default).lanes = 4)
    (hrs : ∀ ℓ < sh.sched.levels.size, (ops.rows ℓ).size = (sh.sched.levels[ℓ]!).q)
    (hwd : ∀ ℓ < sh.sched.levels.size, ∀ j < (sh.sched.levels[ℓ]!).q,
      ((ops.rows ℓ)[j]!).size = FlockVBridge.AccChain.levelLanes sh ℓ)
    {ℓ : ℕ} (hℓ : ℓ < sh.sched.levels.size) :
    openAt (proofR sh.sched cap M (readR ops sb e)) (sh.sched.levels.size - 1) ℓ =
      (ops.rows ℓ, (FlockVBridge.Family.levAt none sh.sched (readR ops sb e) ℓ 0).siblings, #[]) := by
  rw [openAt_proofR]
  obtain ⟨w', hw, h⟩ := FlockVBridge.Family.openAt_ligOfRead (sch := sh.sched) none (readR ops sb e) hℓ
  rw [h]
  have hlw : FlockVBridge.AccChain.levelLanes sh ℓ = w' := by
    rcases hw with rfl | ⟨h0, rfl⟩
    · rw [FlockVBridge.Family.Conss.getD_default]
      rfl
    · rw [FlockVBridge.AccChain.levelLanes, ← FlockVBridge.Family.Conss.getD_default, hl4 ℓ hℓ h0]
      rfl
  have hrows : (FlockVBridge.Family.levAt none sh.sched (readR ops sb e) ℓ w').rows = ops.rows ℓ := by
    apply Array.ext
    · simp only [FlockVBridge.Family.levAt, FlockVBridge.Family.levelOf, Array.size_ofFn,
        FlockVBridge.Family.Conss.getD_default, hrs ℓ hℓ]
    · intro j _ hj2
      simp only [FlockVBridge.Family.levAt, FlockVBridge.Family.levelOf, Array.getElem_ofFn, readR]
      rw [← getElem!_pos (ops.rows ℓ) j hj2]
      exact FlockVBridge.Family.fitRow_of_size ((hwd ℓ hℓ j (by rw [← hrs ℓ hℓ]; exact hj2)).trans hlw)
  rw [hrows]
  rfl

abbrev capR (sh : Flock.Shape) (e0 : ℕ → ByteArray) : Array ByteArray :=
  FlockVBridge.Family.capArr 64 (sh.sched.levels.getD 0 default).cap e0

abbrev pR (sh : Flock.Shape) (e0 : ℕ → ByteArray) (w : Array Flock.F128) (ops : FlockVBridge.AccChain.Openings)
    (sb : ℕ → ℕ → ℕ → ByteArray) (e : ℕ → ℕ → ByteArray) : Flock.LigeritoProof :=
  proofR sh.sched (capR sh e0) (msgsO sh w ops) (readR ops sb e)

abbrev sibsR (sh : Flock.Shape) (ops : FlockVBridge.AccChain.Openings) (sb : ℕ → ℕ → ℕ → ByteArray)
    (e : ℕ → ℕ → ByteArray) (ℓ : ℕ) : Array ByteArray :=
  (FlockVBridge.Family.levAt none sh.sched (readR ops sb e) ℓ 0).siblings

section Reads

variable {nS : ℕ} {Vs : Fin nS → VStmt} {Reg' : Type}

/-- It leaves as hypotheses R's openings' read (`hrs hwd hq`, and the run's `α`), each opening's units, the
cap's digests, and `levelTie_of_run`'s other ties. -/
theorem levelTie_of_reads (zs : (s : Fin nS) → (Vs s).Outer Reg') (Lb : (s : Fin nS) → (Vs s).Bits)
    (hG : ∀ s, (zs s).Good (Lb s)) {st : Flock.Stmt} {hp : HoloParams} {cap₀ : Cap Dig} {u : C3Run (holoAt st hp)}
    {cl : ℕ} {sh : Flock.Shape} {w : Array Flock.F128} {ops : FlockVBridge.AccChain.Openings}
    {a : Array (Flock.RecAlgebra.AClaim Flock.F128)} {c : Array (Array Flock.F128)} {k₀ k₁ : ℕ}
    {res : Array Flock.F128} (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hsh : Flock.RecOpening.shape? (holoAt st hp).cls.shape.mReg cl = some sh)
    (hrun : Flock.RecOpening.run sh (msgsO sh w ops) a k₀ (Flock.RecAlgebra.RoundCoins.ofRounds c) = .ok (res, k₁))
    (hz : ∀ x ∈ res, x = 0) (hne : Flock.RecAlgebra.NoEmpty c) (msId : String) (e0 : ℕ → ByteArray)
    (sb : ℕ → ℕ → ℕ → ByteArray) (e : ℕ → ℕ → ByteArray) (hcap : capDig (capR sh e0) = cap₀)
    (hrs : ∀ ℓ < sh.sched.levels.size, (ops.rows ℓ).size = (sh.sched.levels[ℓ]!).q)
    (hwd : ∀ ℓ < sh.sched.levels.size, ∀ j < (sh.sched.levels[ℓ]!).q,
      ((ops.rows ℓ)[j]!).size = FlockVBridge.AccChain.levelLanes sh ℓ)
    (hq : ∀ ℓ < sh.sched.levels.size, ops.q ℓ = (sh.sched.levels[ℓ]!).q)
    (hR : ∀ (rsr : Array Flock.F128) (rs : Flock.RecAlgebra.RsPts Flock.F128) (k₂ : ℕ) (lgr : Array Flock.F128)
      (lg : Flock.RecAlgebra.LgPts),
      Flock.RecAlgebra.ringSwitch (msgsO sh w ops) a k₀ (Flock.RecAlgebra.RoundCoins.ofRounds c) = .ok ((rsr, rs), k₂) →
      Flock.RecAlgebra.ligerito sh.sched (msgsO sh w ops) rs.T rs.sws k₂ (Flock.RecAlgebra.RoundCoins.ofRounds c) =
        .ok ((lgr, lg), k₁) →
      (∀ ℓ < sh.sched.levels.size, ops.alpha ℓ = (lg.conss[ℓ]!).alpha) ∧
      OpensUnits zs Lb sh ops (sibsR sh ops sb e) (capR sh e0) (pR sh e0 w ops sb e) lg ∧
      (∃ nc, OpenRel sh.sched.m nc rs.T rs.sws u.opened) ∧
      u.lg = FlockSoundness.Discharge.Exec.weave (ligShape GF128 GF256 Dig (holoAt st hp).rsch.sch)
        (ligMsgs (pR sh e0 w ops sb e) capDig sh.sched)
        (((List.range' k₂ (c.size - k₂)).map (fun i => (c)[i]!)).map (coinArr fOf)))
    (ho : ∀ ℓ j, u.os.1 2 ℓ j = openRow (pR sh e0 w ops sb e) sh.sched ℓ j) :
    LevelTie cap₀ (holoAt st hp) u := by
  obtain ⟨hf, -⟩ := schedOf_holoAt hell hsh
  obtain ⟨-, -, -, -, hl4, -, -⟩ := (FlockSoundness.Refine.fast100_dims _ _ hf).1
  exact levelTie_of_units zs Lb hG (sh := sh) (w := w) (ops := ops) (msId := msId) (cap := capR sh e0)
    (p := pR sh e0 w ops sb e) hell hsh hrun hz hne hcap rfl rfl rfl (beq_self_eq_true _)
    (by show (Array.replicate _ _).size = _; simp) (by show (Array.replicate _ _).size = _; simp)
    (by show (Array.replicate _ _).size = _; simp) (capsAt_proofR _ e0) (sibsR sh ops sb e)
    (fun ℓ hℓ => openAt_proofR_eq hl4 hrs hwd hℓ) hrs
    (fun ℓ _ => by
      simp only [sibsR, FlockVBridge.Family.levAt, FlockVBridge.Family.levelOf, Array.size_ofFn,
        FlockVBridge.Family.Conss.getD_default])
    (fun ℓ _ y hy => by
      simp only [sibsR, FlockVBridge.Family.levAt, FlockVBridge.Family.levelOf] at hy
      obtain ⟨k, rfl⟩ := Array.mem_ofFn.1 hy
      exact FlockVBridge.Family.size_fitB _ _)
    hq hR ho

end Reads

end FlockC3.RegLevels
