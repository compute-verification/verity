import Proofs.Flock.Claims.KS.ZkRows

/-!
# The masked reader's recorded caps at a rep's run

`OpensRows` reads an accepted output's openings as the reps' rows; a fork at a cap past level 0 also reads the caps
each rep recorded. `CapsRows` says those are the recursive caps of the rep's proof file: at the setup at the draw and
table `j`'s `Zk.schedule` schedule, rep `r`'s proof file `2j + r` decodes at `2·|regions|` extra claims to a proof whose
`ℓ`-th recursive cap's `capDig` is the rep's recorded cap `ℓ`, for each `ℓ` below the schedule's levels less one.

* `runsAtZCJ_live_caps`, `runsAtZCJH_live_caps`: at public and hidden outputs, each accepted output's reps record their
  proof files' caps (`runsAtZCJ_live_rows`, `runsAtZCJH_live_rows` at `CapsRows`, the played rep's caps from
  `zk_rep_reqsL`).
* `decodesTableC_zkTabs`, `decodesTableHC_zkTabs`, `decodesTableC_deploy`: the deployed `opnD` and `decD` at `zkTabs`
  are right at `CapsRows` at one live outcome `o₀`, on `o₀` where `verify --zk` accepts what `wrS` writes for it
  (`decodesTableR_zkTabs`, `decodesTableHR_zkTabs`, `decodesTableR_deploy` at `CapsRows`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable repTapeJ)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **An output's reps record their proof files' recursive caps at table `j`**: at the setup at the draw `d` and table
`j`'s `Zk.schedule` schedule, rep `r`'s proof file `2j + r` decodes at `2·|regions|` extra claims, and the rep's
recorded caps are the `capDig`s of that proof's recursive caps below the schedule's levels less one. -/
def CapsRows (I : Inputs) (d : Lean.Json) (t : Exec.Transcript) (j : ℕ) (out : OutZC 𝔽 Dig (List UInt8)) : Prop :=
  ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sch : Flock.Schedule) (tPad : ℕ) (h : j < sts.size),
    FlockDeploy.KS.setupAt (zkInputs I) (some d) = .ok (spec, sts) ∧ Zk.schedule sts[j].m = .ok (sch, tPad) ∧
    ∀ r < 2, ∃ zp : Zk.ZkProof,
      Zk.decode (shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size)) t.proofs[2 * j + r]! = .ok zp ∧
      (if r = 0 then out.pend.1 else out.pend.2).lig.caps =
        (List.range (sch.levels.size - 1)).map (fun i => capDig zp.proof.ligerito.recursiveCaps[i]!)

/-- **`runsZCJ_live` at the setup at each draw, with the caps**: `runsAtZCJ_live_rows` at `CapsRows`, each played
rep's caps its proof's (`zk_rep_reqsL`). -/
theorem runsAtZCJ_live_caps (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZJ dj y₀) (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o))
      (fun S R o j out => out.accepted = true ∧ CapsRows I (dj S) (wr S R o) j.val out) := by
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
  obtain ⟨q₀, hq₀, hc₀, -⟩ := zk_rep_reqsL hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ zp₀
    s₀.rounds _ tr₀ y.hz hseam hlen₀ hcr₀ (hwfA s₀ hm₀) hrun₀ hk₀
  obtain ⟨q₁, hq₁, hc₁, -⟩ := zk_rep_reqsL hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ zp₁
    s₁.rounds _ tr₁ y.hz hseam hlen₀ hcr₀ (hwfA s₁ hm₁) hrun₁ hk₁
  rw [← plays_det hpl₀ hq₀] at hc₀
  rw [← plays_det hpl₁ hq₁] at hc₁
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
    have hN : (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length = 2 * sts[j.val].regions.size := by
      rw [DecodeZL.extraClaims_length, ← hyst]
      show 2 * (regionsOf y.x.st).length = _
      simp [regionsOf]
    rw [hN, hyst] at hdz₀ hdz₁
    refine ⟨hok, spec, sts, y.sch, y.tPad, hj, hs, by rw [← hyst]; exact y.hz, fun r hr => ?_⟩
    obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
    · exact ⟨zp₀, by rw [Nat.add_zero]; exact hdz₀, by rw [if_pos rfl]; exact hc₀⟩
    · exact ⟨zp₁, hdz₁, by rw [if_neg Nat.one_ne_zero]; exact hc₁⟩

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`runsZCJH_liveI` at the setup at each draw, with the caps**: `runsAtZCJH_live_rows` at `CapsRows`, each played
rep's caps its proof's (`zk_rep_reqsL`). -/
theorem runsAtZCJH_live_caps (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : ∀ S R o, RecordsLiveAtS I (dj S) o (wr S R o))
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJW (K := 𝕂) execArith hs512 (planZHJ dj y₀) (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L
      (fun S R o => AcceptsAt I (dj S) (wr S R o))
      (fun S R o j out => out.accepted = true ∧ CapsRows I (dj S) (wr S R o) j.val out) := by
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
  have hcJ : HmOut.parse (inputsJ I j.val).tags (inputsJ I j.val).circuitFile (inputsJ I j.val).tables = .ok c :=
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := drawSetupH_of_setupHiddenI (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hcJ hst hf hD
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
  obtain ⟨q₀, hq₀, hc₀, -⟩ := zk_rep_reqsL hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ zp₀
    s₀.rounds _ tr₀ y.hz hseam hlen₀ hcr₀ (hwfA s₀ hm₀) hrun₀ hk₀
  obtain ⟨q₁, hq₁, hc₁, -⟩ := zk_rep_reqsL hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ zp₁
    s₁.rounds _ tr₁ y.hz hseam hlen₀ hcr₀ (hwfA s₁ hm₁) hrun₁ hk₁
  rw [← plays_det hpl₀ hq₀] at hc₀
  rw [← plays_det hpl₁ hq₁] at hc₁
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
    have hN : (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length = 2 * sts[j.val].regions.size := by
      rw [DecodeZL.extraClaims_length, ← hyst]
      show 2 * (regionsOf y.x.st).length = _
      simp [regionsOf]
    rw [hN, hyst] at hdz₀ hdz₁
    refine ⟨hok, spec, sts, y.sch, y.tPad, hj, hs, by rw [← hyst]; exact y.hz, fun r hr => ?_⟩
    obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
    · exact ⟨zp₀, by rw [Nat.add_zero]; exact hdz₀, by rw [if_pos rfl]; exact hc₀⟩
    · exact ⟨zp₁, hdz₁, by rw [if_neg Nat.one_ne_zero]; exact hc₁⟩

end FlockSoundness.Discharge.ZkHidden

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (planZJ outJ repJ)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ planZHJ)
open FlockSoundness.Discharge.Tables (DecodesTable LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128

/-- **The deployed `--zk` decoders are right at `zkTabs` at the caps, at one outcome** (`decodesTableR_zkTabs` with
`runsAtZCJ_live_caps`). -/
theorem decodesTableC_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : Composed.TagsOkZJ I) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (o₀ : LiveOutJ (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length spec.mPts)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun j out => out.accepted = true ∧
        Composed.CapsRows I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o₀) j.val out)
      (fun o => Composed.AcceptsAt I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o) ∧ o = o₀)
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ := setupZJ_of_setupAt hT.1 ht dj hs hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJW
    (Composed.runsAtZCJ_live_caps (Reg := Unit) I ht dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok)) () ()
  have hp := planZJ_eq_tabsJ hT.1 ht dj y₀ hs hok h0 L hz hk hlay ()
  revert i o₀
  rw [← hp, ← opnAtZJ_eq_opnD (planZJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  intro o₀ i
  exact h _ _ (fun o ho => by obtain ⟨ha, rfl⟩ := ho; exact ⟨ha, fun _ _ hw => ⟨hw.1, hw⟩⟩) i

/-- **The deployed hidden-output `--zk` decoders are right at `zkTabs` at the caps, at one outcome**
(`decodesTableHR_zkTabs` with `runsAtZCJH_live_caps`). -/
theorem decodesTableHC_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupAt (ZkExec.zkInputs I) (some (dj ∅)) = .ok (spec, sts)) (hshape : sts.forM Flock.Zk.shapeOk = .ok ())
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (o₀ : LiveOutJ (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length spec.mPts)
    (i : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun j out => out.accepted = true ∧
        Composed.CapsRows I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o₀) j.val out)
      (fun o => Composed.AcceptsAt I (dj ∅) (Claims.Bridge.Custody.wrS I (dj ∅) o) ∧ o = o₀)
      (opnD sts (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (decD (tabsJ sts.size (zkTabs h0 L hz hk hlay)))
      (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKHJ_empty hT.1 dj hs
  obtain ⟨y₀, -, -, -⟩ :=
    setupZHJ_of_setupAt hT.1 ht hc dj hs hshape hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Composed.decodes_of_runsOnZCJW
    (ZkHidden.runsAtZCJH_live_caps (Reg := Unit) I ht hc dj y₀ (fun S _ o => Claims.Bridge.Custody.wrS I (dj S) o)
      { Ω := Unit, draw := fun _ => ∅ } hT (fun S _ o => Claims.Bridge.Custody.recordsLiveAtS_wrS I (dj S) o)
      (fun _ => hok)) () ()
  have hp := planZHJ_eq_tabsJ hT.1 ht hc dj y₀ hs hshape hok h0 L hz hk hlay ()
  revert i o₀
  rw [← hp, ← opnAtZHJ_eq_opnD (planZHJ dj y₀) dj hT.1 hs (by rw [hp, tabsJ_length])]
  intro o₀ i
  exact h _ _ (fun o ho => by obtain ⟨ha, rfl⟩ := ho; exact ⟨ha, fun _ _ hw => ⟨hw.1, hw⟩⟩) i

/-- **The deployed `--zk` decoders are right at `zkTabs` at the caps at a masked statement in scope, at one outcome**:
`decodesTableC_zkTabs` at public outputs, `decodesTableHC_zkTabs` at hidden outputs (`decodesTableR_deploy`). -/
theorem decodesTableC_deploy {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : parse true Readable.relation x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d)
    (o₀ : LiveOutJ (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts)
    (i : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) :
    DecodesTable (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (fun j out => out.accepted = true ∧
        Composed.CapsRows ins.read.I d (Claims.Bridge.Custody.wrS ins.read.I d o₀) j.val out)
      (fun o => Composed.AcceptsAt ins.read.I d (Claims.Bridge.Custody.wrS ins.read.I d o) ∧ o = o₀)
      (opnD sts (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (decD (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay))) i := by
  have hlen : (tables true x).length = sts.size := by
    unfold tables
    rw [hset]
    simp
  revert i o₀
  rw [hlen]
  intro o₀ i
  obtain ⟨hzk, hsA⟩ := setupAt_of_setupOf hs hin hset
  rw [hd] at hsA
  obtain ⟨ht, hrr, hl⟩ := tags_of_inScope hs hin hset
  cases hh : ins.read.I.tags.hiddenOutputs
  · exact decodesTableC_zkTabs ht ⟨hh, hrr, hl⟩ (fun _ : Finset (Fin 0) => d) hsA h0 L hz hk hlay o₀ i
  · have hshape := Claims.Bridge.Custody.setup_shapeOk (setupOf_some hset).2 hzk
    obtain ⟨-, hstJ⟩ := setupAt_tableJH hh hsA
    have hp := (Hidden.setupHidden_spec' (hstJ 0 h0)).parse
    have hc : HmOut.parse (ZkExec.zkTags ins.read.I) ins.read.I.circuitFile ins.read.I.tables = .ok sts[0].c :=
      (ZkHidden.parseH_tagsAt ((ZkExec.zkTags_typed _).trans ht) _ _ _ _).symm.trans hp
    exact decodesTableHC_zkTabs ht ⟨hh, hrr, hl⟩ hc (fun _ : Finset (Fin 0) => d) hsA hshape h0 L hz hk hlay o₀ i

end FlockDeploy.KS
