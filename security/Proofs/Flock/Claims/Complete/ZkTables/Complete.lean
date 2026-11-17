module

public import Proofs.Flock.Claims.Complete.Json.Spec
public import Proofs.Flock.Claims.Complete.Zk.Complete
public import Proofs.Flock.Claims.Complete.Zk.Json.Files
public import Proofs.Flock.Claims.Complete.Zk.Json.Runs
public import Proofs.Flock.Claims.Complete.ZkTables.Discharge
public import Proofs.Flock.Claims.Complete.ZkTables.Link
public import Proofs.Flock.Claims.Complete.ZkTables.Runs
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Zk.Consumed
import Proofs.Flock.Claims.Complete.Zk.Lanes
import Proofs.Flock.Claims.Complete.Zk.RepRounds
import Proofs.Flock.Claims.Complete.Zk.Replay
import Proofs.Flock.Claims.Complete.ZkTables.Count
import Proofs.Flock.Claims.Complete.ZkTables.Scope

/-! Masked completeness at any number of tables, from named premises (`complete`). -/

@[expose] public section

open Flock FlockDeploy Transcript

namespace FlockComplete.ZkTables

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

theorem readyZk_of_ready {x : Claims.Msg} (h : ready H x = true) {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) : readyZk H x = true := by
  unfold ready at h
  unfold readyZk
  rw [hs] at h ⊢
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  exact ⟨⟨⟨⟨⟨⟨h1, h.1.1.1.1.1⟩, h.1.1.1.1.2⟩, h.1.1.1.2⟩, h.1.1.2⟩, h.1.2⟩, (recordFitsAt_one h1).1 h.2⟩

def Case.toZk (c : Case H rel) (h1 : c.sts.size = 1) : CaseZk H rel :=
  ⟨c.x, c.w, c.ω, c.r, c.s, c.spec, c.sts, c.scope, c.rel_w, readyZk_of_ready c.ready c.setup h1, c.good, c.setup⟩

theorem sessionS_one {resp : ℕ → Claims.Msg} {spec : SessionSpec} {sts : Array Flock.Stmt} (h1 : sts.size = 1)
    (w : Claims.Flock.Wit) (ω : TapeZk) : sessionS H resp spec sts w ω = sessionSZk H resp spec sts w ω := by
  unfold sessionS sessionSZk
  rw [traces_one h1]

def Run.toZk (o : Run H rel) (h1 : o.sts.size = 1) : RunZk H rel :=
  ⟨o.toCase.toZk h1, o.record, o.proofs, o.bodies, (sessionS_one h1 o.w o.ω).symm.trans o.ran⟩

section Several

variable (H rel)

structure RecordAt (o : Run H rel) : Prop where
  finalFits : o.record.toUTF8.size < 2 ^ 64 ∧ o.proofs.size < 2 ^ 64 ∧ ∀ b ∈ o.proofs, b.size < 2 ^ 64
  draw : sampleEq (FlockSoundness.Discharge.Exec.recordDraw o.record) o.s.read.draw = true
  liveReads : liveOf o.spec o.record = .ok o.live
  canonical : Flock.RecordText.check o.record = .ok ()
  rootB : (o.live.rootB.toList == o.link.1) = true
  publicsLink : (o.live.publics.toList == o.link.2) = true
  points : (o.live.points == pointsOf o.spec (Claims.coinSeq o.r 0)) = true
  count : ((o.live.streams.toList.flatMap (·.toList)).length == o.mids.length) = true
  rounds : ((o.live.streams.toList.flatMap (·.toList)).zip o.mids).all
    (fun (r, a, c) => r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n) = true
  decodes : Record.decodeWith o.hs o.spec o.record = .ok o.sess
  publics : ∀ q ∈ (o.tabs.zip o.sess.publics).toList, q.1.publics q.2 = .ok ()
  proofList : (o.sess.proofSha.size != o.spec.repStreams.size ||
    !o.spec.repStreams.all fun a => o.sess.proofSha.any fun x => x.1 == a.2.2) = false
  proofDigests : ∀ q ∈ (o.spec.repStreams.zip o.proofs).toList,
    ((o.sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (dgOf o.spec (Claims.hOf H) q.2))) = false
  repStream : ∀ i < o.tabs.size, ∀ r < 2,
    (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).isSome = true
  repExtras : ∀ i < o.tabs.size, o.tabs[i]!.extra o.sess.points = .ok (o.extras i)
  repDecodes : ∀ i < o.tabs.size, ∀ r < 2,
    Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok (o.zp i r)
  repParams : ∀ i < o.tabs.size, ∀ r < 2, ((o.zp i r).proof.params != pcsParams o.tabs[i]!.merkle (o.sch i)) = false
  padsRoot : ∀ i < o.tabs.size, ∀ r < 2, ((o.zp i r).inner.root.size != 64) = false
  capCommitted : ∀ i < o.tabs.size, ∀ r < 2, (flat (o.zp i r).proof.cap != o.sess.rootB[i]!) = false

def RecordJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → RecordAt H rel o

structure OpenAt (o : Run H rel) : Prop where
  replay : ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r) = .ok (o.rp i r)
  ringSwitch : ∀ i < o.tabs.size, ∀ r < 2,
    (ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
      (o.zp i r).proof.ringSwitches).run (o.rp i r).2 = .ok (o.sw i r)
  ligerito : ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap (o.zp i r).proof.ligerito
      (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 = .ok (o.lg i r)
  lanes : ∀ i < o.tabs.size, ∀ r < 2,
    Zk.ligeritoConstraints (o.lay i) (o.lg i r).1 (o.zp i r).proof.ligerito (o.zp i r).level0 = .ok (o.lanes i r)
  inner : ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
      (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r)
  consumed : ∀ i < o.tabs.size, ∀ r < 2, ((o.inner i r).2.k != (o.stream i r).rounds.size) = false

def OpenJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → OpenAt H rel o

end Several

theorem Case.one_or_several (c : Case H rel) : c.sts.size = 1 ∨ 1 < c.sts.size := by
  have := c.tables.1
  omega

theorem complete_of_at (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h64 : JsonRT.Out64 H) (hSR : SessionRunsZk H rel) (hRec : ∀ o : Run H rel, RecordAt H rel o)
    (hOp : ∀ o : Run H rel, OpenAt H rel o) (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) :
    Claims.Flock.Complete H (deployZk rel) :=
  complete_of_goals H rel hMR hMF
    { session := hSR
      sends := sends_goal H rel
      linkFirst := linkFirst_goal H h64 rel
      finalFits := fun o => (hRec o).finalFits
      draw := fun o => (hRec o).draw
      liveReads := fun o => (hRec o).liveReads
      canonical := fun o => (hRec o).canonical
      rootB := fun o => (hRec o).rootB
      publicsLink := fun o => (hRec o).publicsLink
      points := fun o => (hRec o).points
      count := fun o => (hRec o).count
      rounds := fun o => (hRec o).rounds
      tables := tables_goal H rel
      decodes := fun o => (hRec o).decodes
      publics := fun o => (hRec o).publics
      schedules := schedules_goal H rel
      proofCount := proofCount_goal H rel
      proofList := fun o => (hRec o).proofList
      proofDigests := fun o => (hRec o).proofDigests
      repStream := fun o => (hRec o).repStream
      repExtras := fun o => (hRec o).repExtras
      repDecodes := fun o => (hRec o).repDecodes
      repParams := fun o => (hRec o).repParams
      padsRoot := fun o => (hRec o).padsRoot
      replay := fun o => (hOp o).replay
      ringSwitch := fun o => (hOp o).ringSwitch
      ligerito := fun o => (hOp o).ligerito
      lanes := fun o => (hOp o).lanes
      inner := fun o => (hOp o).inner
      consumed := fun o => (hOp o).consumed
      capCommitted := fun o => (hRec o).capCommitted }

theorem complete (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel) (hR : JsonRT.RepShapedZk H rel)
    (hPR : JsonRT.ProverRunsZk H rel) (hRS : FlockComplete.RingSwitchZk H rel) (hLg : FlockComplete.LigeritoZk H rel)
    (hIn : FlockComplete.Inner H rel) (hSJ : StmtOkJ H rel)
    (hPRJ : ProverRunsJ H rel) (hRec : RecordJ H rel) (hOp : OpenJ H rel) (hMR : MaskRank.MaskRankBound)
    (hMF : MaskFree rel) : Claims.Flock.Complete H (deployZk rel) := by
  have hP : JsonRT.ProverOkZk H rel := JsonRT.proverOkZk h64 (JsonRT.repRoundsZk_goal H rel)
  have hD : FlockComplete.RepDecodesZk H rel := JsonRT.repDecodesZk_goal h64 hS hP hR
  have hRp : FlockComplete.Replay H rel := replay_goal h64 hS hP hR
  have hSR : FlockComplete.SessionRunsZk H rel := JsonRT.sessionRunsZk_goal hS hPR
  have rec1 : ∀ o : Run H rel, (h1 : o.sts.size = 1) → RecordAt H rel o := fun o h1 =>
    { finalFits := by
        obtain ⟨-, -, -, -, -, -, hsz, hb, -⟩ := JsonRT.RunZk.rep h64 hS hP hR (o.toZk h1)
        exact ⟨o.recordFits, by change o.proofs.size = 2 at hsz; omega, hb⟩
      draw := JsonRT.sampleMatchesZk_goal h64 hS hP (o.toZk h1)
      liveReads := JsonRT.liveReadsZk_goal h64 hS hP (o.toZk h1)
      canonical := JsonRT.canonicalZk_goal h64 hS hP (o.toZk h1)
      rootB := JsonRT.custodyRootBZk_goal h64 hS hP (o.toZk h1)
      publicsLink := JsonRT.custodyPublicsZk_goal h64 hS hP (o.toZk h1)
      points := JsonRT.custodyPointsZk_goal h64 hS hP (o.toZk h1)
      count := JsonRT.custodyCountZk_goal h64 hS hP (o.toZk h1)
      rounds := JsonRT.custodyRoundsZk_goal h64 hS hP (o.toZk h1)
      decodes := JsonRT.decodesZk_goal h64 hS hP (o.toZk h1)
      publics := JsonRT.publicsZk_goal h64 hS hP (o.toZk h1)
      proofList := JsonRT.proofListZk_goal h64 hS hP (o.toZk h1)
      proofDigests := JsonRT.proofDigestsZk_goal h64 hS hP (o.toZk h1)
      repStream := JsonRT.repStreamZk_goal h64 hS hP (o.toZk h1)
      repExtras := JsonRT.repExtrasZk_goal h64 hS hP (o.toZk h1)
      repDecodes := JsonRT.repDecodesZk_goal h64 hS hP hR (o.toZk h1)
      repParams := JsonRT.repParamsZk_goal h64 hS hP hR (o.toZk h1)
      padsRoot := JsonRT.padsRoot_goal h64 hS hP hR (o.toZk h1)
      capCommitted := JsonRT.capCommittedZk_goal h64 hS hP hR (o.toZk h1) }
  have open1 : ∀ o : Run H rel, (h1 : o.sts.size = 1) → OpenAt H rel o := fun o h1 =>
    { replay := hRp (o.toZk h1)
      ringSwitch := hRS (o.toZk h1)
      ligerito := hLg (o.toZk h1)
      lanes := laneConstraints_goal hLg hD (o.toZk h1)
      inner := hIn (o.toZk h1)
      consumed := consumedZk_goal h64 hS hP hRp hRS hLg hIn (o.toZk h1) }
  refine complete_of_at H rel h64 (fun c => (c.one_or_several).elim (fun h1 => ?_) (sessionRunsJ_goal hSJ hPRJ c))
    (fun o => (o.one_or_several).elim (rec1 o) (hRec o)) (fun o => (o.one_or_several).elim (open1 o) (hOp o)) hMR hMF
  obtain ⟨record, proofs, bodies, hran⟩ := hSR (c.toZk h1)
  exact ⟨record, proofs, bodies, (sessionS_one h1 c.w c.ω).trans hran⟩

end FlockComplete.ZkTables
