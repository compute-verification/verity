import Proofs.Flock.Claims.KS.ZkPlanHJ
import Proofs.Flock.Claims.KS.ZkEntries

/-!
# The deployed `--zk` decoders at the statement's draw

`Deploy.V` sets a masked session up at the statement's draw and runs `verify --zk` on the record against that setup;
its checks give the record's own draw only up to key order (`checks_draw`). `verify --zk` itself sets the session up at
the record's draw (`AcceptsZK`), which keeps the draw: it is the session's `unitDraw` and is written into the public
header, so the two setups agree only where the draws are equal as values. Here the old discharge reads the setup at the
statement's draw instead, so no draw in the record is read:

* `AcceptsAt`, `RecordsLiveAtS`: `AcceptsZK` and `RecordsLiveAt` at the setup at a given draw (`setupAt`).
* `RunsOnZCJV`, `decodes_of_runsOnZCJV`: `RunsOnZCJ` and `decodesOnZKJ_of_runsOnZCJ` at any verdict on the live
  outcomes; `RunsOnZCJW` is `RunsOnZCJV` at any verdict on the outputs.
* `runsAtZCJ_live`, `runsAtZCJH_live`: `runsZCJ_live` and `runsZCJH_liveI` at `AcceptsAt`, public and hidden outputs.
* `OpensEntries`, `runsAtZCJ_live_entries`, `runsAtZCJH_live_entries`: at public and hidden outputs, each accepted
  output's level-0 openings verify at its cap and are, on the witness lanes, the masked reader's entries
  (`zk_table_opens₀Z`).
* `setupAt_of_setup`, `recordsLiveAtS_of_checks`: `Deploy.V`'s setup is the setup at the statement's draw, and where
  its checks pass the last message records the run's live outcome there (`recordsLiveAt_of_checks` without `hdraw`).
* `wrS`, `not_acceptsAt_nil`, `recordsLiveAtS_wrS`: the record writer at a draw, which records the live session.
* `acceptsAt_of_decide`, `acceptsAt_wrS`: an accepting decision of `Deploy.V --zk` is accepted at the setup at its
  statement's draw, at its last message and at `wrS` (`accepts_of_decide`, `acceptsZK_wrD` without `hdraw`).
* `decodesTable_zkTabs`, `decodesTableH_zkTabs`: the deployed `--zk` decoders `opnD` and `decD` are right at
  `zkTabs` on the outcomes `verify --zk` accepts at the setup at the empty draw at `wrS`, at public and hidden
  outputs (`ZkPlanJ`'s and `ZkPlanHJ`'s plans at the law of that one draw).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable repTapeJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`verify --zk` accepts at the session it sets up at the draw `d`**: `AcceptsZK` with the setup at `d`
(`FlockDeploy.KS.setupAt`) for the setup at the record's draw. -/
def AcceptsAt (I : Inputs) (d : Lean.Json) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, FlockDeploy.KS.setupAt (zkInputs I) (some d) = .ok (spec, sts) ∧ sts.forM Zk.shapeOk = .ok () ∧
    Zk.verify spec (sts.map Setup.ofCircuit) t.record t.proofs = .ok ()

/-- **`RecordsLiveAt` at the setup at the draw `d`**: where `verify --zk` accepts the transcript `t` at the session it
sets up at `d`, its proof files are the outcome `o`'s, and the session its record decodes to there has `o`'s link points
and, at each of `o`'s tables, `o`'s `root_B` and rep streams' rounds. -/
def RecordsLiveAtS (I : Inputs) (d : Lean.Json) {J : ℕ} (o : LiveOutJ J mPts) (t : Exec.Transcript) : Prop :=
  AcceptsAt I d t →
    t.proofs = o.proofs ∧
    ∀ spec sts sess, FlockDeploy.KS.setupAt (zkInputs I) (some d) = .ok (spec, sts) →
      Record.decode spec t.record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin J, sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

section Runs

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts))
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (L : Law n)
  (acc : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Prop)

/-- **`RunsOnZCJ` at the verdict `acc`** on the live outcomes, for flock-verify `--zk`'s acceptance of what a writer
writes. -/
def RunsOnZCJV : Prop :=
  ∀ ω R (o : LiveOutJ (planZ (L.draw ω) R).length mPts), acc (L.draw ω) R o →
    ∀ j, ∃ pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl,
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 0)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 0) (fin (L.draw ω) R j 0 o.proofs) pend.1 ∧
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 1)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 1) (fin (L.draw ω) R j 1 o.proofs) pend.2 ∧
      (outZC Hs ((planZ (L.draw ω) R).get j) (capOf (L.draw ω) R o.rootB j) pend
        (opn (L.draw ω) R o.proofs j)).accepted = true

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
/-- **The decoder at the law's draws, table by table, from `RunsOnZCJV`** (`decodesOnZKJ_of_runsOnZCJ` at `acc`). -/
theorem decodes_of_runsOnZCJV (h : RunsOnZCJV A Hs planZ opn dec fin capOf L acc) (ω : L.Ω) (R : Reg)
    (j : Fin (planZ (L.draw ω) R).length) :
    DecodesTable (outJ Hs (planZ (L.draw ω) R)) (fun _ o => o.accepted = true) (acc (L.draw ω) R)
      (opn (L.draw ω) R) (dec (L.draw ω) R) (fun _ _ => true) (fin (L.draw ω) R) (capOf (L.draw ω) R)
      (repJ A Hs (planZ (L.draw ω) R)) j := by
  intro o ho
  obtain ⟨pend, h₀, h₁, hacc⟩ := h ω R o ho j
  obtain ⟨hl₀, hl₁⟩ := outZC_lig Hs hacc
  exact ⟨pend, PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 0 hl₀) h₀,
    PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 1 hl₁) h₁, hacc⟩

/-- **`RunsOnZCJV` at an output verdict `ok`** of the live outcome, the table and the output, for `accepted`. -/
def RunsOnZCJW (ok : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Fin (planZ S R).length →
    OutZC 𝔽 D Sl → Prop) : Prop :=
  ∀ ω R (o : LiveOutJ (planZ (L.draw ω) R).length mPts), acc (L.draw ω) R o →
    ∀ j, ∃ pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl,
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 0)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 0) (fin (L.draw ω) R j 0 o.proofs) pend.1 ∧
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 1)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 1) (fin (L.draw ω) R j 1 o.proofs) pend.2 ∧
      ok (L.draw ω) R o j (outZC Hs ((planZ (L.draw ω) R).get j) (capOf (L.draw ω) R o.rootB j) pend
        (opn (L.draw ω) R o.proofs j))

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
theorem RunsOnZCJW.runsOnZCJV {ok : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts →
      Fin (planZ S R).length → OutZC 𝔽 D Sl → Prop}
    (h : RunsOnZCJW A Hs planZ opn dec fin capOf L acc ok)
    (hok : ∀ S R o j out, ok S R o j out → out.accepted = true) : RunsOnZCJV A Hs planZ opn dec fin capOf L acc := by
  intro ω R o ho j
  obtain ⟨pend, h₀, h₁, hk⟩ := h ω R o ho j
  exact ⟨pend, h₀, h₁, hok _ _ _ _ _ hk⟩

end Runs

/-- **An output's level-0 openings are the masked reader's entries at table `j`**: the setup at the draw `d` has a
table `j`, whose schedule is `Zk.schedule`'s, the record decodes to a session with table `j`'s two rep streams, and the
output's level-0 openings verify against `cap` at `width₀` below level 0's `2^d` and are, on the witness lanes, the
reps' `zkRepColsOf` entries at `2·|regions|` extra claims on the streams' rounds and table `j`'s proof files. -/
def OpensEntries (I : Inputs) (d : Lean.Json) (t : Exec.Transcript) (j : ℕ) (T : TabZK 𝔽 mPts) (cap : Cap Dig)
    (out : OutZC 𝔽 Dig (List UInt8)) : Prop :=
  ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord) (sch : Flock.Schedule) (tPad : ℕ)
    (s₀ s₁ : StreamRec) (h : j < sts.size),
    FlockDeploy.KS.setupAt (zkInputs I) (some d) = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
    Zk.schedule sts[j].m = .ok (sch, tPad) ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
    sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
    (∀ x ∈ opens₀Z T.sch out, VerifiedZ H512 enc512 leaf512 cap (width₀ T) (pathLen (lvOf T.sch 0)) x ∧
      x.1 < 2 ^ (sch.levels.getD 0 default).d) ∧
    (opens₀Z T.sch out).map (fun x => (x.1, x.2.1.take (2 ^ T.sch.k0))) =
      Claims.Flock.zkRepColsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₀.rounds
          t.proofs[2 * j]! ++
        Claims.Flock.zkRepColsOf (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) s₁.rounds
          t.proofs[2 * j + 1]!

/-- **`runsZCJ_live` at the setup at each draw, with the entries**: `RunsOnZCJV` at `AcceptsAt` of the law's draw,
where `wr` records the live session at the setup at that draw (`RecordsLiveAtS`), so the record's own draw is never
read, with each accepted output's level-0 openings the masked reader's entries (`OpensEntries`, `zk_table_opens₀Z`). -/
theorem runsAtZCJ_live_entries (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZJ dj y₀) (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o))
      (fun S R o j out => out.accepted = true ∧
        OpensEntries I (dj S) (wr S R o) j.val ((planZJ dj y₀ S R).get j) (capOfRoot o.rootB[j.val]!) out) := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  have hpts2 := setupTables_points hh' hs
  have hsr : spec.retainRounds = true := ((setupTables_retain hh' hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := FlockDeploy.KS.setupAt_tableJ hh hs
  have hjn : j.val < nTab I := planZJ_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, c, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨-, -, sch₀, hf⟩ := schedule_spec hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ _ht j.val
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZJ_of ht' x (sch := schs[j.val].1) (tPad := schs[j.val].2)
    (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZJ dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZJ
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZJ.table_eq _ _
  rw [← hysch, ← hytp, ← hyst] at hv0 hv1
  rw [← hyst] at hst
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hs hdecS
  obtain ⟨hroot, hstreams⟩ := hrj j
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : sts[j.val].mPts = mPts := by rw [← hyst]; exact y.x.hmPts
  have hsz' : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp j.val hj).symm.trans hmP)
  have hwf := setupH_wf y.x.h
  have hExt : ExtraRealizes (Setup.ofCircuit y.x.st) y.table.S sess.points
      (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts) := by
    have := ofCircuit_extra y.x.st hwf.1 hwf.2 sess.points (hpd.1.trans hp2) (by rw [y.x.hmPts]; exact hsz')
    rw [y.x.hmPts, hpts] at this
    exact this
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupH_tagsAt_hm96 ((zkTags_merkleLeaf I).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam := y.table_seam
  have hlay := y.table_lay
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold y.table.S :=
    ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀
    s₀.rounds _ tr₀ y.hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀ (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁
    s₁.rounds _ tr₁ y.hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁ (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess j.val _ _ c y.hz hms hseam hlay hfold hExt hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZJ y.table = ex₀.size := by
    unfold nXZJ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZJ (planZJ (Reg := Reg) dj y₀) (L.draw ω) R j =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZJ
    rw [hTy, hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZJ (planZJ (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZJ
    rw [hst]
    dsimp only
    rw [y.hz]
    dsimp only
    rw [hTy, hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · rw [hTy, hdecEq, ← hr₀]
    exact hrr₀
  · rw [hTy, hdecEq, ← hr₁]
    exact hrr₁
  · dsimp only
    rw [hTy, hopn, ← hroot]
    have hok := outZC_of_accZ hacc'
    obtain ⟨hver, hent⟩ := zk_table_opens₀Z (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts
      spec (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ c _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀ zp₁
      hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁ hok
    have hN : (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length = 2 * sts[j.val].regions.size := by
      rw [DecodeZL.extraClaims_length, ← hyst]
      show 2 * (regionsOf y.x.st).length = _
      simp [regionsOf]
    refine ⟨hok, spec, sts, sess, y.sch, y.tPad, s₀, s₁, hj, hs, hdecS, by rw [← hyst]; exact y.hz, hf₀, hf₁, hver,
      hent.trans ?_⟩
    rw [hN, hyst]

/-- **`runsZCJ_live` at the setup at each draw**: `RunsOnZCJV` at `AcceptsAt` of the law's draw, where `wr` records the
live session at the setup at that draw (`RecordsLiveAtS`), so the record's own draw is never read
(`runsAtZCJ_live_entries`). -/
theorem runsAtZCJ_live (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) :
    RunsOnZCJV (K := 𝕂) execArith hs512 (planZJ dj y₀) (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o)) :=
  RunsOnZCJW.runsOnZCJV execArith hs512 _ _ _ _ _ _ _ (runsAtZCJ_live_entries I _ht dj y₀ wr L hT hRec hDraw)
    fun _ _ _ _ _ h => h.1

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`runsZCJH_liveI` at the setup at each draw, with the entries**: `RunsOnZCJV` at `AcceptsAt` of the law's draw
under hidden outputs, where `wr` records the live session at the setup at that draw (`RecordsLiveAtS`), with each
accepted output's level-0 openings the masked reader's entries (`OpensEntries`, `zk_table_opens₀Z`). -/
theorem runsAtZCJH_live_entries (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZHJ dj y₀) (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o))
      (fun S R o j out => out.accepted = true ∧
        OpensEntries I (dj S) (wr S R o) j.val ((planZHJ dj y₀ S R).get j) (capOfRoot o.rootB[j.val]!) out) := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, hshape, hv⟩ := hacc
  have hpts2 := setupTables_pointsAny hs
  have hsr : spec.retainRounds = true := ((setupTables_retainAny hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := FlockDeploy.KS.setupAt_tableJH hh hs
  have hjn : j.val < nTab I := planZHJ_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, cap, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨sch₀, hf⟩ := fast100_of_schedule hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ ht j.val
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := HiddenExec.drawSetupH_of_setupHiddenU (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape j.val hj)
    (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZHJ dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZHJ
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZHJ.table_eq _ _
  rw [← hysch, ← hytp, ← hyst] at hv0 hv1
  rw [← hyst] at hst
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hs hdecS
  obtain ⟨hroot, hstreams⟩ := hrj j
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : sts[j.val].mPts = mPts := by rw [← hyst]; exact y.x.hmPts
  have hsz' : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp j.val hj).symm.trans hmP)
  have hExt : ExtraRealizes (Setup.ofCircuit y.x.st) y.table.S sess.points
      (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts) := by
    have := ofCircuit_extra y.x.st y.x.wf.1 y.x.wf.2 sess.points (hpd.1.trans hp2) (by rw [y.x.hmPts]; exact hsz')
    rw [y.x.hmPts, hpts] at this
    exact this
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupHidden_hm96 (((tagsAt_merkleLeaf _ _ _).trans (zkTags_merkleLeaf I)).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam := y.table_seam
  have hlay := y.table_lay
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold y.table.S :=
    ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀
    s₀.rounds _ tr₀ y.hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀ (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁
    s₁.rounds _ tr₁ y.hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁ (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess j.val _ _ cap y.hz hms hseam hlay hfold hExt hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZJ y.table = ex₀.size := by
    unfold nXZJ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZJ (planZHJ (Reg := Reg) dj y₀) (L.draw ω) R j =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZJ
    rw [hTy, hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZHJ (planZHJ (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZHJ
    rw [hst]
    dsimp only
    rw [y.hz]
    dsimp only
    rw [hTy, hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · rw [hTy, hdecEq, ← hr₀]
    exact hrr₀
  · rw [hTy, hdecEq, ← hr₁]
    exact hrr₁
  · dsimp only
    rw [hTy, hopn, ← hroot]
    have hok := outZC_of_accZ hacc'
    obtain ⟨hver, hent⟩ := zk_table_opens₀Z (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts
      spec (Setup.ofCircuit y.x.st) y.sch y.tPad sess j.val _ _ cap _ rfl y.hz hseam hExt hwfA hv0 hv1 s₀ s₁ zp₀
      zp₁ hf₀ hf₁ hdz₀ hdz₁ (p₀, p₁) _ hpl₀ hpl₁ hok
    have hN : (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length = 2 * sts[j.val].regions.size := by
      rw [DecodeZL.extraClaims_length, ← hyst]
      show 2 * (regionsOf y.x.st).length = _
      simp [regionsOf]
    refine ⟨hok, spec, sts, sess, y.sch, y.tPad, s₀, s₁, hj, hs, hdecS, by rw [← hyst]; exact y.hz, hf₀, hf₁, hver,
      hent.trans ?_⟩
    rw [hN, hyst]

/-- **`runsZCJH_liveI` at the setup at each draw**: `RunsOnZCJV` at `AcceptsAt` of the law's draw under hidden
outputs, where `wr` records the live session at the setup at that draw (`RecordsLiveAtS`; `runsAtZCJH_live_entries`). -/
theorem runsAtZCJH_live (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJV (K := 𝕂) execArith hs512 (planZHJ dj y₀) (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o)) :=
  RunsOnZCJW.runsOnZCJV execArith hs512 _ _ _ _ _ _ _ (runsAtZCJH_live_entries I ht dj y₀ wr L hT hRec hDraw)
    fun _ _ _ _ _ h => h.1

end FlockSoundness.Discharge.ZkHidden

namespace Claims.Bridge.Custody

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge
open Lean (Json)

local instance lawfulBEqByteArrayDraw : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-- **`Deploy.V`'s setup is `verify --zk`'s at the statement's draw** (`setupOf_zkInputs` with no record), where the
statement carries no `--public-inputs`. -/
theorem setupAt_of_setup {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (hs : s.setup = .ok (spec, sts))
    (hzk : s.read.zk = true) (hpi : s.read.publicInputs = none) :
    KS.setupAt (ZkExec.zkInputs s.read.I) s.read.draw = .ok (spec, sts) := by
  rw [← Ins.setup_read hs]
  simp only [KS.setupAt, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, ZkExec.zkInputs, hzk, hpi,
    ite_true]

/-- **Custody gives `RecordsLiveAtS`** (`recordsLiveAt_of_checks` at the statement's draw `d`, so with no premise on
the record's draw): where `Deploy.V --zk`'s checks pass, the last message's record and proof files are a record of the
run's live outcome (`viewOf`) at the setup at `d`, at any `J` of the session's tables. -/
theorem recordsLiveAtS_of_checks {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {J mPts : ℕ} {d : Json} (hs : s.setup = .ok (spec, sts)) (hzk : s.read.zk = true) (hd : s.read.draw = some d)
    (hJ : J ≤ spec.tables.size) (hc : checks zk s spec x link c₁ mids f = .ok ()) :
    Composed.RecordsLiveAtS s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  intro _
  refine ⟨rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [← hd, setupAt_of_setup hs hzk (publicInputs_none_of_checks hc)] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decode_live hdec hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  refine ⟨by simp only [viewOf]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList = ((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)
    rw [hD]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine forall₂_map_eq hyr fun rd rd' hmem ⟨hn, hcn, hb, hsha⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hsha hbytes
    subst hn hcn hb hbytes
    simp [withSha, hsha _ rfl]

open Classical in
/-- **The record writer at the statement's draw**: at the live outcome `o`, a transcript `verify --zk` accepts at the
setup at the draw `d` that records `o` there (`RecordsLiveAtS`), where one exists, else a session without proof files,
which `verify --zk` refuses (`not_acceptsAt_nil`). A decided run's last message is such a transcript for its outcome
(`acceptsAt_wrS`). -/
noncomputable def wrS {J mPts : ℕ} (I : Exec.Inputs) (d : Json) (o : Tables.LiveOutJ J mPts) : Exec.Transcript :=
  if h : ∃ t, Composed.AcceptsAt I d t ∧ Composed.RecordsLiveAtS I d o t then h.choose else ⟨"", #[]⟩

theorem not_acceptsAt_nil (I : Exec.Inputs) (d : Json) (record : String) :
    ¬ Composed.AcceptsAt I d ⟨record, #[]⟩ := by
  rintro ⟨spec, sts, -, -, h⟩
  exact verify_nil _ _ _ h

/-- **`wrS` records the live session at the setup at `d`** (`RecordsLiveAtS`). -/
theorem recordsLiveAtS_wrS {J mPts : ℕ} (I : Exec.Inputs) (d : Json) (o : Tables.LiveOutJ J mPts) :
    Composed.RecordsLiveAtS I d o (wrS I d o) := by
  unfold wrS
  split
  · rename_i h
    exact (Classical.choose_spec h).2
  · exact fun h => absurd h (not_acceptsAt_nil _ _ _)

/-- **An accepting decision of `Deploy.V --zk` at SHA-512 is an accepting `verify --zk` at the statement's draw**
(`accepts_of_decide` with no premise on the record's draw): on a statement of draw `d`, the last message's record and
proof files are accepted at the setup at `d` (`AcceptsAt`), and record the run's live outcome `viewOf` there. -/
theorem acceptsAt_of_decide {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} {J mPts : ℕ} {d : Json} (ha : act true x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next true x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    Composed.AcceptsAt s.read.I d ⟨f.record, f.proofs⟩ ∧
      Composed.RecordsLiveAtS s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  obtain ⟨hv, hc⟩ := (next_decide_sha512 ha).1 hacc
  simp only [ite_true] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide ha)
  have hzk : s.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hsA := setupAt_of_setup hs hzk (publicInputs_none_of_checks hc)
  rw [hd] at hsA
  exact ⟨⟨spec, sts, hsA, setup_shapeOk hs hzk, hv⟩, recordsLiveAtS_of_checks hs hzk hd hJ hc⟩

/-- **`Deploy.V --zk`'s accepting runs are accepted at `wrS`**: where its decision accepts at SHA-512 on a statement of
draw `d`, `verify --zk` accepts at the setup at `d` what `wrS` writes for the run's live outcome at `J ≤` the session's
tables. -/
theorem acceptsAt_wrS {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} {J mPts : ℕ} {d : Json} (ha : act true x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next true x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    Composed.AcceptsAt s.read.I d (wrS s.read.I d (viewOf J mPts spec link c₁ f)) := by
  have hex : ∃ t, Composed.AcceptsAt s.read.I d t ∧
      Composed.RecordsLiveAtS s.read.I d (viewOf J mPts spec link c₁ f) t :=
    ⟨_, acceptsAt_of_decide ha hacc hd hJ⟩
  unfold wrS
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

end Claims.Bridge.Custody

namespace FlockDeploy.KS.Gen

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge _root_.Claims.Bridge.Custody
open Lean (Json)

attribute [local instance] _root_.Claims.Bridge.Custody.lawfulBEqByteArrayDraw

variable [S : ZkSetup]

/-- **`Deploy.V`'s setup is `verify --zk`'s at the statement's draw** (`setupOf_zkInputs` with no record), where the
statement carries no `--public-inputs`. -/
theorem Claims.Bridge.Custody.setupAt_of_setup {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (hs : S.Ok zk s spec sts)
    (hzk : s.read.zk = true) (hpi : s.read.publicInputs = none) :
    KS.setupAt (ZkExec.zkInputs s.read.I) s.read.draw = .ok (spec, sts) := by
  rw [← Ins.setup_read hs]
  simp only [KS.setupAt, Readable.Read.setup, Readable.Read.tags, Readable.Read.coins, ZkExec.zkInputs, hzk, hpi,
    ite_true]

/-- **Custody gives `RecordsLiveAtS`** (`recordsLiveAt_of_checks` at the statement's draw `d`, so with no premise on
the record's draw): where `Deploy.V --zk`'s checks pass, the last message's record and proof files are a record of the
run's live outcome (`viewOf`) at the setup at `d`, at any `J` of the session's tables. -/
theorem Claims.Bridge.Custody.recordsLiveAtS_of_checks {zk : Bool} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {x : Claims.Msg}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {J mPts : ℕ} {d : Json} (hs : S.Ok zk s spec sts) (hzk : s.read.zk = true) (hd : s.read.draw = some d)
    (hJ : J ≤ spec.tables.size) (hc : checks zk s spec x link c₁ mids f = .ok ()) :
    Composed.RecordsLiveAtS s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  intro _
  refine ⟨rfl, fun spec' sts' sess hset hdec => ?_⟩
  rw [← hd, setupAt_of_setup hs hzk (publicInputs_none_of_checks hc)] at hset
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hset)
  obtain ⟨-, lv, hlv, hcus⟩ := checks_ok hc
  obtain ⟨hroot, -, hpts, hrounds⟩ := custody_ok hcus
  obtain ⟨eroot, epts, esize, estr⟩ := decode_live hdec hlv
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  refine ⟨by simp only [viewOf]; rw [epts, hpts], fun j => ⟨?_, fun r => ?_⟩⟩
  · show sess.rootB[j.val]! = link.1.toArray[j.val]!
    rw [eroot, ← hroot]
  · obtain ⟨hq, hw⟩ := _root_.Claims.Bridge.Custody.repStreams_getElem spec (i := j.val) (r := r.val) (by omega) r.isLt
    obtain ⟨y, hy, hyr⟩ := estr _ hq
    rw [hw] at hy
    refine ⟨y, hy, ?_⟩
    show y.rounds.toList = ((liveD spec f.record).streams[2 * j.val + r.val]!).toList.map (withSha spec)
    rw [hD]
    have hq' : 2 * j.val + r.val < lv.streams.size := esize ▸ hq
    refine forall₂_map_eq hyr fun rd rd' hmem ⟨hn, hcn, hb, hsha⟩ => ?_
    obtain ⟨am, -, hbytes, -⟩ := forall₂_mem_left hrounds rd' (List.mem_flatMap.2
      ⟨lv.streams[2 * j.val + r.val]!, by rw [getElem!_pos lv.streams _ hq']; exact Array.getElem_mem_toList hq',
        hmem⟩)
    rcases rd with ⟨n, sha, coins, bytes⟩
    rcases rd' with ⟨n', sha', coins', bytes'⟩
    simp only at hn hcn hb hsha hbytes
    subst hn hcn hb hbytes
    simp [withSha, hsha _ rfl]

/-- **An accepting decision of `Deploy.V --zk` at SHA-512 is an accepting `verify --zk` at the statement's draw**
(`accepts_of_decide` with no premise on the record's draw): on a statement of draw `d`, the last message's record and
proof files are accepted at the setup at `d` (`AcceptsAt`), and record the run's live outcome `viewOf` there. -/
theorem Claims.Bridge.Custody.acceptsAt_of_decide {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} {J mPts : ℕ} {d : Json} (ha : act true x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next true x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    Composed.AcceptsAt s.read.I d ⟨f.record, f.proofs⟩ ∧
      Composed.RecordsLiveAtS s.read.I d (viewOf J mPts spec link c₁ f) ⟨f.record, f.proofs⟩ := by
  obtain ⟨hv, hc⟩ := (next_decide_sha512 ha).1 hacc
  simp only [ite_true] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide ha)
  have hzk : s.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hsA := setupAt_of_setup hs hzk (publicInputs_none_of_checks hc)
  rw [hd] at hsA
  exact ⟨⟨spec, sts, hsA, setup_shapeOk hs hzk, hv⟩, recordsLiveAtS_of_checks hs hzk hd hJ hc⟩

/-- **`Deploy.V --zk`'s accepting runs are accepted at `wrS`**: where its decision accepts at SHA-512 on a statement of
draw `d`, `verify --zk` accepts at the setup at `d` what `wrS` writes for the run's live outcome at `J ≤` the session's
tables. -/
theorem Claims.Bridge.Custody.acceptsAt_wrS {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f : Final} {J mPts : ℕ} {d : Json} (ha : act true x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((next true x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d) (hJ : J ≤ spec.tables.size) :
    Composed.AcceptsAt s.read.I d (wrS s.read.I d (viewOf J mPts spec link c₁ f)) := by
  have hex : ∃ t, Composed.AcceptsAt s.read.I d t ∧
      Composed.RecordsLiveAtS s.read.I d (viewOf J mPts spec link c₁ f) t :=
    ⟨_, acceptsAt_of_decide ha hacc hd hJ⟩
  unfold wrS
  rw [dite_eq_left_of_eq_true (eq_true hex)]
  exact (Classical.choose_spec hex).1

export Claims.Bridge.Custody (setupAt_of_setup recordsLiveAtS_of_checks acceptsAt_of_decide acceptsAt_wrS)

end FlockDeploy.KS.Gen


namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (planZJ outJ repJ)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ planZHJ)
open FlockSoundness.Discharge.Tables (DecodesTable)

/-- **The deployed `--zk` decoders are right at `zkTabs`**: where `verify --zk` sets its session up at the empty draw
`dj ∅`, on untyped inputs under `TagsOkZJ`, at every layout and link layout at the session's link points, the
openings `opnD` and decoder `decD` at `tabsJ` of `zkTabs` are right on the outcomes `verify --zk` accepts at the setup
at `dj ∅` at `wrS`: they are the old ones at the plan (`opnAtZJ_eq_opnD`), which is `tabsJ` of `zkTabs`
(`planZJ_eq_tabsJ`), and those are right there (`runsAtZCJ_live` at the law of that one draw, `recordsLiveAtS_wrS`,
`drawOkZKJ_empty`, table 0's setup as the reference). -/
theorem decodesTable_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : Composed.TagsOkZJ I) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (fun _ o => o.accepted = true)
      (fun o => Composed.AcceptsAt I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o))
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ := setupZJ_of_setupAt hT.1 ht dj hs hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJV execArith ZkExec.hs512 _ _ _ _ _ _ _
    (Composed.runsAtZCJ_live (Reg := Unit) I ht dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok)) () ()
  have hp := planZJ_eq_tabsJ hT.1 ht dj y₀ hs hok h0 L hz hk hlay ()
  revert i
  rw [← hp, ← opnAtZJ_eq_opnD (planZJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  exact h

/-- **The deployed hidden-output `--zk` decoders are right at `zkTabs`**: where `verify --zk` sets its session up at
the empty draw `dj ∅` and its tables pass the shape check, on untyped inputs under `TagsOkZHJ` whose circuit file
`HmOut.parse` accepts, at every layout and link layout at the session's link points, the openings `opnD` and decoder
`decD` at `tabsJ` of `zkTabs` are right on the outcomes `verify --zk` accepts at the setup at `dj ∅` at `wrS`: they are
the old hidden-output ones at the plan (`opnAtZHJ_eq_opnD`), which is `tabsJ` of `zkTabs` (`planZHJ_eq_tabsJ`), and
those are right there (`runsAtZCJH_live` at the law of that one draw, `recordsLiveAtS_wrS`, `drawOkZKHJ_empty`,
table 0's setup as the reference). -/
theorem decodesTableH_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : TagsOkZHJ I) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts)) (hshape : sts.forM Flock.Zk.shapeOk = .ok ())
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (fun _ o => o.accepted = true)
      (fun o => Composed.AcceptsAt I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o))
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKHJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ :=
    setupZHJ_of_setupAt hT.1 ht dj hs hshape hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJV execArith ZkExec.hs512 _ _ _ _ _ _ _
    (ZkHidden.runsAtZCJH_live (Reg := Unit) I ht dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok)) () ()
  have hp := planZHJ_eq_tabsJ hT.1 ht dj y₀ hs hshape hok h0 L hz hk hlay ()
  revert i
  rw [← hp, ← opnAtZHJ_eq_opnD (planZHJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  exact h

end FlockDeploy.KS
