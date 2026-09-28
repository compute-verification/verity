import Proofs.Flock.Soundness.Discharge.ZkExec.LiveStatements
import Proofs.Flock.Soundness.Discharge.ZkExec.Verdict
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysRun
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.Exec.Merkle
import Proofs.Flock.Soundness.Discharge.Exec.Retained
import Proofs.Flock.Soundness.Discharge.DecodeZL.Runs

/-!
# `RunsZC` at the live `--zk` tables, from `TablesAtZKLive`

`runsZC_live`: at each of the law's draws, on an outcome flock-verify `--zk` accepts, `TablesAtZKLive` puts the draw's
table at `tablesAtZK` (`DrawSetupZK.table`, at the executable's dimensions, pads and inner code: `table_seam`,
`table_lay`), the acceptance peels to the one table's two `verifyRep` successes (`setupTables_ok`, `zk_verify_ok`) and
each rep's run (`zk_verifyRep_ok`), `zk_rep_run_plays` makes each run a `RepRun` for `decAtZ` at the result it plays on
the proof's messages, and `zk_table_verdict` makes the table's verdict accept at the two results.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL ZkSession

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

section Table

variable {n mPts : ℕ} {I : Inputs} {S : Finset (Fin n)} {d : Lean.Json}

/-- A draw's `--zk` table is at the executable's dimensions. -/
theorem table_seam (y : ZkLink.DrawSetupZK I mPts S d) :
    ZkTabSeam y.table (Setup.ofCircuit y.x.st) y.sch y.tPad :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- A draw's `--zk` table has the executable's pads and inner code. -/
theorem table_lay (y : ZkLink.DrawSetupZK I mPts S d) : ZkTabLay y.table (Setup.ofCircuit y.x.st) y.sch := by
  constructor <;> intros <;> rfl

end Table

/-- **A `--zk` rep's `RepRun`, at the result it plays on the proof's messages**: `zk_rep_run`, with the run of the
compiled rep on `repMsgsZ` that the `RepRun` is at. -/
theorem zk_rep_run_plays {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : TabZK 𝔽 mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (extras : Array Flock.Claim) (bytes : ByteArray) (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray)
    (tr : Flock.Transcript) (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hdl : st.merkle.digestLen = 64)
    (hseam : ZkTabSeam T st sch tPad) (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hdec : Zk.decode (shapeZK st sch extras) bytes = .ok zp) (hroot : zp.inner.root.size = 64)
    (hwf : RoundsWF rounds) (hret : AllRetained rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) :
    ∃ pend : ZkLink.PendZ 𝔽 Dig Sl,
      Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend ∧
      PlaysCoin.RepRun (ZkLink.repZCAt execArith Hs T pts r)
        (repShapeZ 𝔽 𝕂 Dig T (Model.extraClaims T.S T.ptLocal mPts pts).length)
        (decZ (wRestZ st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len)) rounds.toList [] pend := by
  obtain ⟨pend, hpl, -⟩ := zk_rep_plays Hs T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf
    hrun hk
  obtain ⟨Tv, hl, hR⟩ := zk_rep_frames st sch tPad r.val extras zp rounds dom tr hsch hrun hk hret
  obtain ⟨hyb, htau, hy⟩ := repProg_sizes _ _ _ _ _ _ hrun
  have hcut := zk_cut st sch tPad extras bytes zp Tv dom rounds hsch hdl hdec hroot hyb htau hy hl hR
  have hσ : repShapeZ 𝔽 𝕂 Dig T (Model.extraClaims T.S T.ptLocal mPts pts).length =
      shapeZE st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len := by
    unfold repShapeZ shapeZE
    rw [← hlen, hseam.hm, hseam.hk, hseam.hsch, hseam.he, hseam.ht, hseam.hnp, hseam.hnP, hseam.hq,
      Nat.add_comm extras.size 2]
  refine ⟨pend, hpl, ⟨fun rd hrd => ?_, _, hpl, ?_⟩⟩
  · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hrd
    have hi' : i < rounds.size := by simpa using hi
    have := hwf i hi'
    rw [getElem!_pos rounds i hi'] at this
    simpa using this
  · rw [hσ]
    exact hcut

/-- **`RunsZC` at the live `--zk` tables, at the law's draws** (`RunsZCLive`): `TablesAtZKLive` at the draw, the two
reps' `verifyRep` successes, `zk_rep_run_plays` for each rep, and `zk_table_verdict` at the two results. -/
theorem runsZC_live : RunsZCLive := by
  intro n mPts Reg I ht dj S₀ y₀ wr L hTab hT hRec hDraw ω R o hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨st, sch, tPad, -, -, hst, hz, y, rfl, rfl, rfl, hTy, hextR⟩ :=
    hTab I ht dj y₀ wr hh hc1 hRec (L.draw ω) R o (hDraw ω) hacc
  obtain ⟨hd, hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  have hpts2 := setupTables_points hh' hs
  have hsr : spec.retainRounds = true := ((setupTables_retain hh' hs).trans (zkTags_retainRounds I)).trans hrr
  rcases setupTables_ok hh' hs with ⟨-, st', hst', rfl, rfl⟩ | ⟨hc', -⟩
  swap
  · change 1 < I.count at hc'
    omega
  have hst'' : Stmt.setupH (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj (L.draw ω)))
      I.partition I.program = .ok st' := by
    rw [← hd]
    exact hst'
  obtain rfl : st' = y.x.st := Except.ok.inj (hst''.symm.trans hst)
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, c, hz0, hv0, hv1⟩ := hall 0 (by simp)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  change Zk.schedule (Setup.ofCircuit y.x.st).m = .ok schs[0] at hz0
  change Zk.verifyRep y.x.st.spec (Setup.ofCircuit y.x.st) schs[0].1 schs[0].2 sess 0 0 _ = .ok c at hv0
  change Zk.verifyRep y.x.st.spec (Setup.ofCircuit y.x.st) schs[0].1 schs[0].2 sess 0 1 _ = .ok c at hv1
  have hsc : schs[0] = (y.sch, y.tPad) := Except.ok.inj (hz0.symm.trans hz)
  rw [hsc] at hv0 hv1
  dsimp only at hv0 hv1
  obtain ⟨hroot, hpts, hstreams⟩ := hrec y.x.st.spec #[y.x.st] sess hs hdecS
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : y.x.st.mPts = mPts := (hDraw ω).2 y.x.st hst
  have hsz : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp 0 (by simp)).symm.trans hmP)
  have hExt := hextR sess.points (hpd.1.trans hp2) hsz
  rw [hpts] at hExt
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupH_hm96 ((zkTags_merkleLeaf I).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam : ZkTabSeam (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R) (Setup.ofCircuit y.x.st) y.sch
      y.tPad := by
    rw [hTy]
    exact table_seam y
  have hlay : ZkTabLay (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R) (Setup.ofCircuit y.x.st) y.sch := by
    rw [hTy]
    exact table_lay y
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).S := by
    rw [hTy]
    exact ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hExt' : ExtraRealizes (Setup.ofCircuit y.x.st) (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).S
      sess.points (Model.extraClaims (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).S
        (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts) := by
    rw [hTy]
    exact hExt
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt' ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R) o.pts 0
    (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀ s₀.rounds _ tr₀ hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀
    (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R) o.pts 1
    (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁ s₁.rounds _ tr₁ hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁
    (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (ZkLink.tablesAtZK (Reg := Reg) dj y₀) (L.draw ω) R o.pts y.x.st.spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess 0 _ _ c hz hms hseam hlay hfold hExt' hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).S
        (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZ (ZkLink.tablesAtZK (Reg := Reg) dj y₀) (L.draw ω) R = ex₀.size := by
    unfold nXZ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZ (ZkLink.tablesAtZK (Reg := Reg) dj y₀) (L.draw ω) R =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZ
    rw [hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZ (ZkLink.tablesAtZK (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).S
        (ZkLink.tablesAtZK (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * 0]!, (wr (L.draw ω) R o).proofs[2 * 0 + 1]!] := by
    rw [← hprf]
    unfold opnAtZ
    rw [hst]
    dsimp only
    rw [hz]
    dsimp only
    rw [hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · show PlaysCoin.RepRun _ _ (decAtZ (ZkLink.tablesAtZK (Reg := Reg) dj y₀) (L.draw ω) R) o.rep0 [] p₀
    rw [hdecEq, ← show s₀.rounds.toList = o.rep0 from hr₀]
    exact hrr₀
  · show PlaysCoin.RepRun _ _ (decAtZ (ZkLink.tablesAtZK (Reg := Reg) dj y₀) (L.draw ω) R) o.rep1 [] p₁
    rw [hdecEq, ← show s₁.rounds.toList = o.rep1 from hr₁]
    exact hrr₁
  · show accZ _ _ _ (outZ hs512 _ _ _ (fun _ => capOfRoot o.rootB) (p₀, p₁) _) = true
    rw [hopn, ← hroot]
    exact hacc'

end FlockSoundness.Discharge.ZkExec
