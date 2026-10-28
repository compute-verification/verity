import Proofs.Flock.Claims.Complete.ZkTables.Discharge
import Proofs.Flock.Claims.Complete.ZkTables.Link
import Proofs.Flock.Claims.Complete.ZkTables.Runs
import Proofs.Flock.Claims.Complete.Zk.Complete

/-!
# Masked completeness at any number of tables, from named premises

* **One table is `Zk/`'s**: a good case of one table is a `CaseZk` (`Case.toZk`: `ready` gives `readyZk`), its session
  `sessionSZk`'s (`sessionS_one`), and a run of one table a `RunZk` (`Run.toZk`), at which every method is the same. So
  every goal `complete_zk` rests on, or proves, holds at a run of one table, the record's size from `ready`
  (`Run.recordFits`), as `complete_zk`'s from `readyZk` (`RunZk.recordFits`).
* **At every number of tables**: the scope's goals (`tables_goal`, `schedules_goal`, `proofCount_goal`), the link
  message first (`linkFirst_goal`) and the sends (`sends_goal`).
* **Several tables**: the session runs from `StmtOkJ` and `ProverRunsJ` (`sessionRunsJ_goal`, `complete_zk`'s
  `StmtOkZk` and `ProverRunsZk` at several tables, the latter per table); and what `complete_zk` asks or proves at one
  table, at a run of several, as named `Prop`s: the record checks that `Zk/Json/` proves at one table from the record's
  premises (`RecordAt`, `RecordJ`), and the rep's checks (`OpenAt`, `OpenJ`: per rep the masked replay, the ring
  switch, the masked Ligerito, level 0's lanes, the inner proof and every round consumed), of which `complete_zk`
  takes the ring switch, the masked Ligerito and the inner proof at one table and proves the rest.
* `complete`: Flock's masked verifier is complete for `honestP` with error `δc H`
  (`Claims.Flock.Complete H (deployZk rel)`), from `complete_zk`'s premises and
  these four.
-/

open Flock FlockDeploy Transcript

namespace FlockComplete.ZkTables

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-! ## One table -/

/-- **`ready` at a setup of one table is `readyZk`.** -/
theorem readyZk_of_ready {x : Claims.Msg} (h : ready H x = true) {s : Ins} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hs : setupOf true x = some (s, spec, sts)) (h1 : sts.size = 1) : readyZk H x = true := by
  unfold ready at h
  unfold readyZk
  rw [hs] at h ⊢
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  exact ⟨⟨⟨⟨⟨⟨h1, h.1.1.1.1.1.1⟩, h.1.1.1.1.2⟩, h.1.1.1.2⟩, h.1.1.2⟩, h.1.2⟩, (recordFitsAt_one h1).1 h.2⟩

/-- **A good case of one table is a good masked case.** -/
def Case.toZk (c : Case H rel) (h1 : c.sts.size = 1) : CaseZk H rel :=
  ⟨c.x, c.w, c.ω, c.r, c.s, c.spec, c.sts, c.scope, c.rel_w, readyZk_of_ready c.ready c.setup h1, c.good, c.setup⟩

/-- **At one table the session is `sessionSZk`'s.** -/
theorem sessionS_one {resp : ℕ → Claims.Msg} {spec : SessionSpec} {sts : Array Flock.Stmt} (h1 : sts.size = 1)
    (w : Claims.Flock.Wit) (ω : TapeZk) : sessionS H resp spec sts w ω = sessionSZk H resp spec sts w ω := by
  unfold sessionS sessionSZk
  rw [traces_one h1]

/-- **A run of one table is a masked run.** -/
def Run.toZk (o : Run H rel) (h1 : o.sts.size = 1) : RunZk H rel :=
  ⟨o.toCase.toZk h1, o.record, o.proofs, o.bodies, (sessionS_one h1 o.w o.ω).symm.trans o.ran⟩

/-! ## Several tables -/

section Several

variable (H rel)

/-- **The record's checks at a run**: `Goals`' checks on the record and the proof files, which `Zk/Json/` proves at one
table from the record's premises. -/
structure RecordAt (o : Run H rel) : Prop where
  finalFits : o.record.toUTF8.size < 2 ^ 64 ∧ o.proofs.size < 2 ^ 64 ∧ ∀ b ∈ o.proofs, b.size < 2 ^ 64
  draw : drawEq (FlockSoundness.Discharge.Exec.recordDraw o.record) o.s.read.draw = true
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

/-- **The record's checks at every run of several tables.** -/
def RecordJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → RecordAt H rel o

/-- **The rep's checks, at a run**: per rep the masked replay, the ring switch, the masked Ligerito, level 0's lane
phase, the inner proof and every round consumed. -/
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

/-- **The rep's checks, at every run of several tables.** -/
def OpenJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → OpenAt H rel o

end Several

/-! ## The composition -/

theorem Case.one_or_several (c : Case H rel) : c.sts.size = 1 ∨ 1 < c.sts.size := by
  have := c.tables.1
  omega

/-- **Masked completeness at any number of tables** from the session running at every good case, and the record's
checks and the rep's checks at every run. -/
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

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables: from `complete_zk`'s premises, which give every check at one
table, and at several tables the statement's and the prover's pieces' (`StmtOkJ`, `ProverRunsJ`), the record's checks
(`RecordJ`) and the rep's checks (`OpenJ`). -/
theorem complete (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel) (hR : JsonRT.RepShapedZk H rel)
    (hPR : JsonRT.ProverRunsZk H rel) (hRS : FlockComplete.RingSwitchZk H rel) (hLg : FlockComplete.LigeritoZk H rel)
    (hIn : FlockComplete.Inner H rel) (hSJ : StmtOkJ H rel)
    (hPRJ : ProverRunsJ H rel) (hRec : RecordJ H rel) (hOp : OpenJ H rel) (hMR : MaskRank.MaskRankBound)
    (hMF : MaskFree rel) : Claims.Flock.Complete H (deployZk rel) := by
  have hP : JsonRT.ProverOkZk H rel := JsonRT.proverOkZk h64 (JsonRT.repRoundsZk_goal H rel)
  have hD : FlockComplete.RepDecodesZk H rel := JsonRT.repDecodesZk_goal S h64 hS hP hR
  have hRp : FlockComplete.Replay H rel := replay_goal S h64 hS hP hR
  have hSR : FlockComplete.SessionRunsZk H rel := JsonRT.sessionRunsZk_goal S hS hPR
  have rec1 : ∀ o : Run H rel, (h1 : o.sts.size = 1) → RecordAt H rel o := fun o h1 =>
    { finalFits := by
        obtain ⟨-, -, -, -, -, -, hsz, hb, -⟩ := JsonRT.RunZk.rep S h64 hS hP hR (o.toZk h1)
        exact ⟨o.recordFits, by change o.proofs.size = 2 at hsz; omega, hb⟩
      draw := JsonRT.drawMatchesZk_goal S h64 hS hP (o.toZk h1)
      liveReads := JsonRT.liveReadsZk_goal S h64 hS hP (o.toZk h1)
      canonical := JsonRT.canonicalZk_goal S h64 hS hP (o.toZk h1)
      rootB := JsonRT.custodyRootBZk_goal S h64 hS hP (o.toZk h1)
      publicsLink := JsonRT.custodyPublicsZk_goal S h64 hS hP (o.toZk h1)
      points := JsonRT.custodyPointsZk_goal S h64 hS hP (o.toZk h1)
      count := JsonRT.custodyCountZk_goal S h64 hS hP (o.toZk h1)
      rounds := JsonRT.custodyRoundsZk_goal S h64 hS hP (o.toZk h1)
      decodes := JsonRT.decodesZk_goal S h64 hS hP (o.toZk h1)
      publics := JsonRT.publicsZk_goal S h64 hS hP (o.toZk h1)
      proofList := JsonRT.proofListZk_goal S h64 hS hP (o.toZk h1)
      proofDigests := JsonRT.proofDigestsZk_goal S h64 hS hP (o.toZk h1)
      repStream := JsonRT.repStreamZk_goal S h64 hS hP (o.toZk h1)
      repExtras := JsonRT.repExtrasZk_goal S h64 hS hP (o.toZk h1)
      repDecodes := JsonRT.repDecodesZk_goal S h64 hS hP hR (o.toZk h1)
      repParams := JsonRT.repParamsZk_goal S h64 hS hP hR (o.toZk h1)
      padsRoot := JsonRT.padsRoot_goal S h64 hS hP hR (o.toZk h1)
      capCommitted := JsonRT.capCommittedZk_goal S h64 hS hP hR (o.toZk h1) }
  have open1 : ∀ o : Run H rel, (h1 : o.sts.size = 1) → OpenAt H rel o := fun o h1 =>
    { replay := hRp (o.toZk h1)
      ringSwitch := hRS (o.toZk h1)
      ligerito := hLg (o.toZk h1)
      lanes := laneConstraints_goal hLg hD (o.toZk h1)
      inner := hIn (o.toZk h1)
      consumed := consumedZk_goal S h64 hS hP hRp hRS hLg hIn (o.toZk h1) }
  refine complete_of_at H rel h64 (fun c => (c.one_or_several).elim (fun h1 => ?_) (sessionRunsJ_goal S hSJ hPRJ c))
    (fun o => (o.one_or_several).elim (rec1 o) (hRec o)) (fun o => (o.one_or_several).elim (open1 o) (hOp o)) hMR hMF
  obtain ⟨record, proofs, bodies, hran⟩ := hSR (c.toZk h1)
  exact ⟨record, proofs, bodies, (sessionS_one h1 c.w c.ω).trans hran⟩

end FlockComplete.ZkTables
