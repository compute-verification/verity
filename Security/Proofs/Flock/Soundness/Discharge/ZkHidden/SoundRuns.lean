import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDefs
import Proofs.Flock.Soundness.Discharge.ZkHidden.Theorems
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLive

/-!
# `RunsZC` at the live hidden-output `--zk` tables, from `TablesAtZKHLive`

`runsZC_liveH`: `ZkExec.runsZC_live`'s proof under hidden outputs. `TablesAtZKHLive` puts the draw's table at
`tablesAtZKH` (`DrawSetupZKH.table`, at the executable's dimensions, pads and inner code: `table_seamH`, `table_layH`),
the acceptance peels to the one table's two `verifyRep` successes (`setupTables_okH`, `zk_verify_ok`) and each rep's run
(`zk_verifyRep_ok`), and `zk_rep_run_plays` and `zk_table_verdict`, which read the table only through those seams, make
each run a `RepRun` for `decAtZ` and the table's verdict accept at the two results.

`runsZC_liveHU` is the proof, from the session alone (`tablesAtZKH_liveU`). `runsZC_liveHI` and `runsZC_liveH` read it
on every circuit `HmOut.parse` accepts (where the unit returns the outputs, for `runsZC_liveH`), and `runsZC_liveHISt` and
`runsZC_liveHSt` on every statement circuit (`HmOut.parseStmt`, a templated file's template unrolled).
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec HiddenExec

local notation "𝕂" => FlockLevel3.GF256

section Table

variable {n mPts : ℕ} {I : Inputs} {S : Finset (Fin n)} {d : Lean.Json}

/-- A hidden-output draw's `--zk` table is at the executable's dimensions. -/
theorem table_seamH (y : DrawSetupZKH I mPts S d) :
    ZkTabSeam y.table (Setup.ofCircuit y.x.st) y.sch y.tPad :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- A hidden-output draw's `--zk` table has the executable's pads and inner code. -/
theorem table_layH (y : DrawSetupZKH I mPts S d) : ZkTabLay y.table (Setup.ofCircuit y.x.st) y.sch := by
  constructor <;> intros <;> rfl

end Table

/-- **`RunsZC` at the live hidden-output `--zk` tables, from the session alone** (`tablesAtZKH_liveU`). -/
theorem runsZC_liveHU {n mPts : ℕ} {Reg : Type} (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n)
    (hT : HiddenDec.TagsOkH I) (hRec : ZkLink.RecordsLiveZK I dj wr) (hDraw : ∀ ω, DrawOkZKH I dj mPts (L.draw ω)) :
    RunsOnZC (K := 𝕂) execArith hs512 (tablesAtZKH (Reg := Reg) dj y₀) I wr
      (opnAtZH (tablesAtZKH dj y₀) I dj) (decAtZ (tablesAtZKH dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L := by
  intro ω R o hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨st, sch, tPad, -, -, hst, hz, y, rfl, rfl, rfl, hTy, hextR⟩ :=
    tablesAtZKH_liveU (Reg := Reg) I ht dj y₀ wr hh hc1 hRec (L.draw ω) R o (hDraw ω) hacc
  obtain ⟨hd, hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  have hpts2 := setupTables_pointsAny hs
  have hsr : spec.retainRounds = true := ((setupTables_retainAny hs).trans (zkTags_retainRounds I)).trans hrr
  rcases setupTables_okH hh' hs with ⟨-, st', hst', rfl, rfl⟩ | ⟨hc', -⟩
  swap
  · change 1 < I.count at hc'
    omega
  have hst'' : Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj (L.draw ω)))
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
    setupHidden_hm96 ((zkTags_merkleLeaf I).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam : ZkTabSeam (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R) (Setup.ofCircuit y.x.st) y.sch y.tPad := by
    rw [hTy]
    exact table_seamH y
  have hlay : ZkTabLay (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R) (Setup.ofCircuit y.x.st) y.sch := by
    rw [hTy]
    exact table_layH y
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).S := by
    rw [hTy]
    exact ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hExt' : ExtraRealizes (Setup.ofCircuit y.x.st) (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).S
      sess.points (Model.extraClaims (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).S
        (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts) := by
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
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R) o.pts 0
    (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀ s₀.rounds _ tr₀ hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀
    (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R) o.pts 1
    (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁ s₁.rounds _ tr₁ hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁
    (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (tablesAtZKH (Reg := Reg) dj y₀) (L.draw ω) R o.pts y.x.st.spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess 0 _ _ c hz hms hseam hlay hfold hExt' hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).S
        (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZ (tablesAtZKH (Reg := Reg) dj y₀) (L.draw ω) R = ex₀.size := by
    unfold nXZ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZ (tablesAtZKH (Reg := Reg) dj y₀) (L.draw ω) R =
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
  have hopn : opnAtZH (tablesAtZKH (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).S
        (tablesAtZKH (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * 0]!, (wr (L.draw ω) R o).proofs[2 * 0 + 1]!] := by
    rw [← hprf]
    unfold opnAtZH
    rw [hst]
    dsimp only
    rw [hz]
    dsimp only
    rw [hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · show PlaysCoin.RepRun _ _ (decAtZ (tablesAtZKH (Reg := Reg) dj y₀) (L.draw ω) R) o.rep0 [] p₀
    rw [hdecEq, ← show s₀.rounds.toList = o.rep0 from hr₀]
    exact hrr₀
  · show PlaysCoin.RepRun _ _ (decAtZ (tablesAtZKH (Reg := Reg) dj y₀) (L.draw ω) R) o.rep1 [] p₁
    rw [hdecEq, ← show s₁.rounds.toList = o.rep1 from hr₁]
    exact hrr₁
  · show accZ _ _ _ (outZ hs512 _ _ _ (fun _ => capOfRoot o.rootB) (p₀, p₁) _) = true
    rw [hopn, ← hroot]
    exact hacc'

theorem runsZC_liveHI : RunsZCLiveHI := by
  intro n mPts Reg I ht c _ dj S₀ y₀ wr L _ hT hRec hDraw
  exact runsZC_liveHU I ht dj y₀ wr L hT hRec hDraw

theorem runsZC_liveH : RunsZCLiveH := by
  intro n mPts Reg I ht c _ _ dj S₀ y₀ wr L _ hT hRec hDraw
  exact runsZC_liveHU I ht dj y₀ wr L hT hRec hDraw

theorem runsZC_liveHISt : RunsZCLiveHISt := by
  intro n mPts Reg I ht c _ dj S₀ y₀ wr L _ hT hRec hDraw
  exact runsZC_liveHU I ht dj y₀ wr L hT hRec hDraw

theorem runsZC_liveHSt : RunsZCLiveHSt := by
  intro n mPts Reg I ht c _ _ dj S₀ y₀ wr L _ hT hRec hDraw
  exact runsZC_liveHU I ht dj y₀ wr L hT hRec hDraw

end FlockSoundness.Discharge.ZkHidden
