import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.Complete.ZkTables.MaskFree
import Proofs.Flock.Claims.Complete.ZkTables.Trace
import Proofs.Flock.Claims.Complete.Zk.Extras

/-!
# The masked ring switch at several tables

`Zk/RingSwitch.lean` and `Zk/Extras.lean` at every table of a run of any number of tables.

* `ringSwitchJ`: at every table and rep, the verifier's ring switch on the replay's claims, each valued by its check,
  and the extra claims returns, given the extra claims' algebra (`RsExtrasAt`): the rep ran zerocheck, lincheck and
  the ring switch in closed form (`repPZk_rs`), and the replay's claims have the prover's points (`rs_of`), on the
  table's stream (`Run.walkJ`).
* `Run.rsExtras`: at the deployed relation the extra claims' algebra holds at every table from `RelS` alone: table `i`
  is the statement the soundness proof plays, satisfied by its share of the witness's bits (`Case.model`), each
  table's link points have the session's `m_pts` (`Case.mPts`, `Run.sess_points`), and `claims_switch`.
* `completeR`, `completeRel`: completeness at any number of tables with the ring switch dropped from the rep's checks
  (`LgInnerAt`: the masked Ligerito and the inner proof), given `RsExtrasAt`, which the deployed relation gives.
  `completeJRel`: the same from `complete_zk_relation`'s premises at one table, the ring switch there from
  `ringSwitchZk_goal` and `rsExtrasZk`.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The extra claims' ring-switch algebra at a run**: at every table, each extra claim's switched vector of the
table's trace meets its value. -/
def RsExtrasAt (o : Run H rel) : Prop :=
  ∀ i < o.tabs.size, ∀ k < (o.extras i).size,
    dot (wOf (o.extras i)[k]!) (switchVec (traces o.sts o.w)[i]! (o.extras i)[k]!) = (o.extras i)[k]!.value

/-- **The masked ring switch replays** at every table and rep of a run, given the extra claims' algebra there. -/
theorem ringSwitchJ (S : JsonSpec) (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase)
    (hR : RepShapedAt o.toCase) (hX : RsExtrasAt o) :
    ∀ i < o.tabs.size, ∀ r < 2,
      (ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
        (o.zp i r).proof.ringSwitches).run (o.rp i r).2 = .ok (o.sw i r) := by
  intro i hi r hr
  have hRp := replayJ S h64 o hS hR i hi r hr
  have hi' := Run.lt_sts hi
  obtain ⟨schs, cms, streams, hss, hbytes, -, -, hrep⟩ := Run.walkJ S h64 o hS hR
  obtain ⟨htabi, -, -, hreps⟩ := hrep i hi'
  obtain ⟨hstream, p, ic, l, t', l', prior, ws, -, hzp, hst', -, -, hrun⟩ := hreps r hr
  have h25 := (o.toCase.tables.2.2.2 i hi').1
  have hm : 14 ≤ (Setup.ofCircuit o.sts[i]!).m := by
    show 14 ≤ o.sts[i]!.m
    rw [getElem!_pos o.sts i hi']
    omega
  obtain ⟨full, hsw, hpre⟩ := repPZk_rs (hm := hm) (h := hrun)
  have hk : 2 * i + r < streams.size := by omega
  have hAg : Agrees (o.stream i r).rounds t'.rounds 0 := by
    unfold Agrees
    intro j hj _
    have hmem : t'.rounds[j] ∈ streams[2 * i + r]! := by
      rw [hst']
      exact Array.getElem_mem hj
    obtain ⟨c, hc⟩ := hbytes _ (by rw [getElem!_pos streams _ hk]; exact Array.getElem_mem _) _ hmem
    refine ⟨roundRec o.spec (Claims.hOf H) t'.rounds[j], ?_, rfl, rfl, by simp [roundRec, hc]⟩
    rw [hstream, hst']
    simp [hj]
  refine Replay.ok_val ?_
  have hm' : 14 ≤ o.tabs[i]!.m := by rw [htabi]; exact hm
  generalize o.rp i r = q at hRp ⊢
  obtain ⟨ht, hcl⟩ := replayWith_ret o.hs o.tabs[i]! hm' (o.lay i) (o.zp i r).proof (o.zp i r).inner.root
    (o.stream i r).rounds 0 _ q.1 q.2 hRp
  rw [ht, hzp, htabi]
  rw [htabi] at hcl
  exact rs_of (hs := o.hs) (st := Setup.ofCircuit o.sts[i]!) (hk := rfl) (h₀ := rfl) (hs₀ := rfl)
    (hR := hAg.of_prefix hpre) (hcl := hcl) (hsw := hsw) (hX := fun k hk => hX i hi k hk)

/-- **Table `i` of a good case of the deployed relation, satisfied**: it is well formed, and its share of the
witness's bits (`Readable.zOf J w i`) satisfies `stmtOf` at its regions. -/
theorem Case.model (c : Case H Readable.relation) {i : ℕ} (hi : i < c.sts.size) :
    ∃ hwf : StmtWF c.sts[i], RegionsWF c.sts[i] ∧
      (stmtOf c.sts[i] hwf (regionsOf c.sts[i])).Satisfies (Readable.zOf c.sts.size c.w i c.sts[i].m) := by
  obtain ⟨p, hps, hty, hpi, hpe, hR⟩ := c.parsed
  have hi' : i < p.sts.size := by rw [hpe]; exact hi
  obtain ⟨hf, hsat⟩ := (Readable.relS_relation p c.w).1 hR i hi'
  have key := Readable.tableAt_model (Readable.tablesOf_of_parses hps) hty hpi hi'
  have e : p.sts[i]! = p.sts[i] := getElem!_pos p.sts i hi'
  revert key hsat hf
  rw [e]
  rintro hf hsat ⟨hwf, hrw, hmat⟩
  obtain ⟨h1, h2, h3⟩ := hsat
  have hA : ((Readable.Tab.ofStmt p.sts[i]).toModel hf.2.2).A₀ = (stmtOf p.sts[i] hwf (regionsOf p.sts[i])).A₀ :=
    funext fun a => funext fun b => (hmat a b).1
  have hB : ((Readable.Tab.ofStmt p.sts[i]).toModel hf.2.2).B₀ = (stmtOf p.sts[i] hwf (regionsOf p.sts[i])).B₀ :=
    funext fun a => funext fun b => (hmat a b).2
  have hsat' : (stmtOf p.sts[i] hwf (regionsOf p.sts[i])).Satisfies (Readable.zOf p.sts.size c.w i p.sts[i].m) := by
    refine ⟨fun o j => ?_, h2, h3⟩
    have := h1 o j
    rw [hA, hB] at this
    exact this
  have he : p.sts[i] = c.sts[i] := by simp only [hpe]
  clear hA hB h1 h2 h3 hmat hf
  revert hwf hrw hsat'
  rw [he, hpe]
  intro hwf hrw hsat'
  exact ⟨hwf, hrw, hsat'⟩

/-- **Every table's link points have the session's length**: `setupTables` refuses tables whose `m_pts` differ. -/
theorem Case.mPts (c : Case H rel) {i : ℕ} (hi : i < c.sts.size) : c.sts[i].mPts = c.spec.mPts :=
  (FlockSoundness.Discharge.HiddenExec.setupTables_pointsAny c.setupTables).2 i hi

/-- The record's link points have the session's `m_pts`: `Record.decodeWith` checks it (S9). -/
theorem Run.sess_points (o : Run H rel) : ∀ p ∈ o.sess.points.toList, p.size = o.spec.mPts := by
  unfold Run.sess
  cases h : Record.decodeWith o.hs o.spec o.record with
  | error e =>
    intro p hp
    rw [show (val (Except.error e : Except String RunRecord)).points = #[] from rfl] at hp
    simp at hp
  | ok s => exact decodeWith_points o.hs h

/-- **The extra claims' ring-switch algebra at the deployed relation**, at every table of a run, from `RelS` alone. -/
theorem Run.rsExtras (o : Run H Readable.relation) : RsExtrasAt o := by
  intro i hi k hk
  have hi' := Run.lt_sts hi
  obtain ⟨hwf, hrw, hsat⟩ := o.toCase.model hi'
  have h25 := (o.toCase.tables.2.2.2 i hi').1
  have ht : o.tabs[i]! = Setup.ofCircuit o.sts[i] := by
    rw [getElem!_pos o.tabs i hi]
    simp [Run.tabs, tabsOf]
  have hzb : ∀ a : Fin (2 ^ (stmtOf o.sts[i] hwf (regionsOf o.sts[i])).m),
      Zc.bitF (traces o.sts o.w)[i]!.z a.val = ((Readable.zOf o.sts.size o.w i o.sts[i].m a).val : 𝔽) := by
    intro a
    rw [traces_getElem o.sts o.w hi']
    unfold Zc.bitF
    rw [traceOf_z o.sts[i] _ a.isLt, Lc.bitF_toZ, Readable.tableWit_getD hi' o.w a.val]
    rfl
  have hex : o.extras i = val (o.sts[i].extraClaims o.sess.points) := by
    simp only [Run.extras, ht, Setup.ofCircuit]
  have hpts : ∀ p ∈ o.sess.points.toList, p.size = o.sts[i].mPts := by
    rw [o.toCase.mPts hi']
    exact o.sess_points
  rw [hex] at hk ⊢
  have key := claims_switch o.sts[i] hwf hrw (by omega) _ hsat (traces o.sts o.w)[i]! hzb (Zc.coinsAt o.resp 0)
    (Lc.coinsAt o.resp 0 0) o.sess.points hpts
  rw [getElem!_pos _ k hk]
  exact key _ (Array.mem_append.2 (Or.inr (Array.getElem_mem hk)))

variable (H rel) in
/-- **The rep's checks but the ring switch, at a run**: per rep the masked Ligerito and the inner proof. -/
structure LgInnerAt (o : Run H rel) : Prop where
  ligerito : ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap (o.zp i r).proof.ligerito
      (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 = .ok (o.lg i r)
  inner : ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
      (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r)

/-- **The rep's checks at a run**, from the extra claims' algebra, the masked Ligerito and the inner proof there. -/
theorem repChecksAt_of (S : JsonSpec) (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase)
    (hR : RepShapedAt o.toCase) (hX : RsExtrasAt o) (hC : LgInnerAt H rel o) : RepChecksAt H rel o :=
  { ringSwitch := ringSwitchJ S h64 o hS hR hX
    ligerito := hC.ligerito
    inner := hC.inner }

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables, from the per-table premises at every good case (`StmtAt`, `ProverRunsAt`, `RepShapedAt`), the
extra claims' algebra (`RsExtrasAt`), and per rep the masked Ligerito and the inner proof (`LgInnerAt`). -/
theorem completeR (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) (S : JsonRT.JsonSpec)
    (h64 : JsonRT.Out64 H) (hS : ∀ c : Case H rel, StmtAt c) (hPR : ∀ c : Case H rel, ProverRunsAt c)
    (hR : ∀ c : Case H rel, RepShapedAt c) (hX : ∀ o : Run H rel, RsExtrasAt o)
    (hC : ∀ o : Run H rel, LgInnerAt H rel o) (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) :
    Claims.Flock.Complete H (deployZk rel) :=
  completeAll H rel S h64 hS hPR hR (fun o => repChecksAt_of S h64 o (hS o.toCase) (hR o.toCase) (hX o) (hC o)) hMR
    hMF

/-- **Flock's masked verifier is complete at the deployed relation** for the honest masked prover on each table's
share, with error `δc H`, at any number of tables: the statement's premise at one table (`StmtOkZk`), the prover's
pieces and the rep proofs' shapes per table (`ProverRunsAt`, `RepShapedAt`), and per rep the masked Ligerito and the
inner proof (`LgInnerAt`). -/
theorem completeRel (H : Claims.Msg → Claims.Msg) (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H)
    (hS : JsonRT.StmtOkZk H Readable.relation) (hPR : ∀ c : Case H Readable.relation, ProverRunsAt c)
    (hR : ∀ c : Case H Readable.relation, RepShapedAt c) (hC : ∀ o : Run H Readable.relation, LgInnerAt H _ o)
    (hMR : MaskRank.MaskRankBound) : Claims.Flock.Complete H (deployZk Readable.relation) :=
  completeR H Readable.relation S h64 (stmtAt_of S hS) hPR hR Run.rsExtras hC hMR maskFree_relation

variable (H rel) in
/-- **The rep's checks but the ring switch, at every run of several tables.** -/
def LgInnerJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → LgInnerAt H rel o

/-- **Flock's masked verifier is complete at the deployed relation** for the honest masked prover on each table's
share, with error `δc H`, at any number of tables: from `complete_zk_relation`'s premises, and at
several tables the prover's pieces (`ProverRunsJ`), the proof files' (`RepShapedJ`), and per rep the masked Ligerito
and the inner proof (`LgInnerJ`). -/
theorem completeJRel (H : Claims.Msg → Claims.Msg) (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H)
    (hS : JsonRT.StmtOkZk H Readable.relation) (hR : JsonRT.RepShapedZk H Readable.relation)
    (hPR : JsonRT.ProverRunsZk H Readable.relation) (hLg : FlockComplete.LigeritoZk H Readable.relation)
    (hIn : FlockComplete.Inner H Readable.relation) (hPRJ : ProverRunsJ H Readable.relation)
    (hRJ : RepShapedJ H Readable.relation) (hCJ : LgInnerJ H Readable.relation) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  have hP : JsonRT.ProverOkZk H Readable.relation :=
    JsonRT.proverOkZk h64 (JsonRT.repRoundsZk_goal H Readable.relation)
  completeJS H Readable.relation S h64 hS hR hPR (ringSwitchZk_goal S h64 hS hP hR (rsExtrasZk H)) hLg hIn hPRJ hRJ
    (fun o hc => repChecksAt_of S h64 o (stmtOkJ S o.toCase hc) (hRJ o.toCase hc) o.rsExtras (hCJ o hc)) hMR
    maskFree_relation

end FlockComplete.ZkTables
