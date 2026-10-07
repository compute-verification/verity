import Proofs.Flock.Soundness.Discharge.ZkPubIn.Composed
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Tables
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Setup
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundRuns
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundDrawOk

/-!
# `RunsZC` at the `--zk` tables with the verifier's public inputs (statements)

ZkHidden's `RecordsLiveZK`, `opnAtZH`, `TablesAtZKHLive`, `RunsZCLiveH` and `drawOkZKH_of_exec` with
`--public-inputs F` (`own` F's bytes): each reads flock-verify's acceptance (`AcceptsZKP I own`) and setupHidden's
statement at `some own` where the original reads `AcceptsZK I` and setupHidden's at `none`.

* `RecordsLiveZKP`: `RecordsLiveZK` at `AcceptsZKP I own` and `setupOfP`.
* `opnAtZHP`: `opnAtZH` at setupHidden's statement with `own`.
* `TablesAtZKHPLive`: `TablesAtZKHLive` at `AcceptsZKP I own`, `DrawOkZKHP`, `DrawSetupZKHP` and `tablesAtZKHP`.
* `RunsZCLiveHP`: `RunsZCLiveH` at those, concluding `RunsOnZCA` at `AcceptsZKP I own`.
* `drawOkZKHP_of_exec`: `drawOkZKH_of_exec` with `own`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock Refine Exec Game Model DecodeZL ZkSession ZkExec ZkLink ZkHidden HiddenDec

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`wr` is the coin server's record of the live `--zk --public-inputs F` session** (`RecordsLiveZK` at
`AcceptsZKP I own`): whenever flock-verify accepts what `wr` writes for the outcome `o` at draw `S`, the record carries
the draw `dj S`, the proof files are `o`'s, and the record decodes to a session whose table's `root_B`, link points and
rep streams' rounds are `o`'s. -/
def RecordsLiveZKP (I : Inputs) (own : ByteArray) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ S R o, AcceptsZKP I own (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOfP (zkInputs I) own (wr S R o).record = .ok (spec, sts) →
      Record.decode spec (wr S R o).record = .ok sess →
      sess.rootB[0]! = o.rootB ∧ ptsOf mPts sess.points = o.pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = pick (o.rep0, o.rep1) r

/-- **The `--zk` openings at draw `S` with the verifier's public inputs** (`opnAtZH` at setupHidden's statement with
`own`). -/
noncomputable def opnAtZHP (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts) (I : Inputs) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray) :
    ZkLink.OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
      I.program (some own) with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZ T S R) #[proofs[0]!, proofs[1]!]
    | .error _ => default
  | .error _ => default

/-- **Wherever flock-verify accepts a session with the verifier's public inputs, the `--zk` table is the
executable's** (`TablesAtZKHLive` at `AcceptsZKP I own`, `DrawOkZKHP`, `DrawSetupZKHP` and `tablesAtZKHP`). -/
def TablesAtZKHPLive : Prop :=
  ∀ (I : Inputs) (own : ByteArray) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = true → I.count ≤ 1 → RecordsLiveZKP I own dj wr →
    ∀ S R o, DrawOkZKHP I own dj mPts S → AcceptsZKP I own (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
          I.program (some own) = .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZKHP I own mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZKHP (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

/-- **`RunsZC` at the live `--zk` tables with the verifier's public inputs, at the law's draws** (`RunsZCLiveH` at
`AcceptsZKP I own`, `tablesAtZKHP`, `opnAtZHP`, `TablesAtZKHPLive`, `RecordsLiveZKP` and `DrawOkZKHP`). -/
def RunsZCLiveHP : Prop :=
  ∀ {n mPts : ℕ} {Reg : Type} (I : Inputs) (own : ByteArray) (_ht : I.tags.typed = false) {c : Flock.Circuit}
    (_hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (_ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (L : Audit.Law n),
    TablesAtZKHPLive (n := n) (mPts := mPts) (Reg := Reg) → TagsOkH I → RecordsLiveZKP I own dj wr →
    (∀ ω, DrawOkZKHP I own dj mPts (L.draw ω)) →
    RunsOnZCA (K := 𝕂) execArith hs512 (tablesAtZKHP (Reg := Reg) dj y₀) (AcceptsZKP I own) wr
      (opnAtZHP (tablesAtZKHP dj y₀) I own dj) (decAtZ (tablesAtZKHP dj y₀)) (fun _ _ _ _ => [])
      (fun _ _ b _ => capOfRoot b) L

/-- **A one-table hidden-output `--zk --public-inputs F` acceptance at the coin server's draw sets the draw up**
(`ZkHidden.drawSetupZKH_of_accepts` with `own`). -/
theorem drawSetupZKHP_of_accepts {I : Inputs} {own : ByteArray} (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    {dj : Finset (Fin n) → Lean.Json} {S : Finset (Fin n)} (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1)
    {t : Exec.Transcript} (h : AcceptsZKP I own t) (hd : recordDraw t.record = some (dj S))
    (hok : DrawOkZKHP I own dj mPts S) : Nonempty (DrawSetupZKHP I own mPts S (dj S)) := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  obtain ⟨st, hst, rfl, rfl, -, -, -⟩ := setupTables_oneHP ((zkTags_hiddenOutputs I).trans hh) hc1 hs
  rw [hd] at hst
  obtain ⟨-, sess, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, cap, hz, -⟩ := hall 0 (by simp)
  have hz' : Flock.Zk.schedule st.m = .ok (schs[0].1, schs[0].2) := by simp only [Array.map_singleton, Array.getElem_singleton] at hz; exact hz
  obtain ⟨sch, hf⟩ := fast100_of_schedule hz'
  obtain ⟨_, pop, -, hdr⟩ := setupHidden_drawnP hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, rfl⟩ := drawSetupHP_of_setupHidden (I := zkInputs I) ((zkTags_typed I).trans ht) hc ho hst hf hD
    (hok.1 D hD) (hok.2 _ hst)
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := zk_schedOK x.st _ _ x.wf.1 x.wf.2 x.hm
    (by simpa using array_forM_ok hshape 0 (by simp)) hz'
  exact ⟨⟨x, _, _, hz', h1, h2, h3, h4, h5, h6 _⟩⟩

/-- **Wherever flock-verify accepts with the verifier's public inputs, the `--zk` table is the executable's.** -/
theorem tablesAtZKHP_live : TablesAtZKHPLive (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I own ht c hc ho dj S₀ y₀ wr hh hc1 hrec S R o hok h
  obtain ⟨hd, -, -⟩ := hrec S R o h
  have hne := drawSetupZKHP_of_accepts ht hc ho hh hc1 h hd hok
  refine ⟨(Classical.choice hne).x.st, (Classical.choice hne).sch, (Classical.choice hne).tPad,
    (Classical.choice hne).x.wf.1, (Classical.choice hne).x.wf.2, (Classical.choice hne).x.h,
    (Classical.choice hne).hz, Classical.choice hne, rfl, rfl, rfl, ?_, fun points hpts hsz => ?_⟩
  · show tableAtZKHP dj y₀ S = _
    unfold tableAtZKHP
    rw [dite_eq_left_of_eq_true (eq_true hne)]
  · have e := ofCircuit_extra (Classical.choice hne).x.st (Classical.choice hne).x.wf.1 (Classical.choice hne).x.wf.2
      points hpts (by rw [(Classical.choice hne).x.hmPts]; exact hsz)
    rw [(Classical.choice hne).x.hmPts] at e
    exact e

section Table

variable {I : Inputs} {own : ByteArray} {S : Finset (Fin n)} {d : Lean.Json}

/-- A draw's `--zk` table with the verifier's public inputs is at the executable's dimensions. -/
theorem table_seamHP (y : DrawSetupZKHP I own mPts S d) :
    ZkTabSeam y.table (Setup.ofCircuit y.x.st) y.sch y.tPad :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- A draw's `--zk` table with the verifier's public inputs has the executable's pads and inner code. -/
theorem table_layHP (y : DrawSetupZKHP I own mPts S d) : ZkTabLay y.table (Setup.ofCircuit y.x.st) y.sch := by
  constructor <;> intros <;> rfl

end Table

/-- **`RunsZC` at the live `--zk` tables with the verifier's public inputs** (`runsZC_liveH`'s proof). -/
theorem runsZC_liveHP : RunsZCLiveHP := by
  intro n mPts Reg I own ht c hc ho dj S₀ y₀ wr L hTab hT hRec hDraw ω R o hacc
  obtain ⟨hh, hrr, hl, hc1⟩ := hT
  obtain ⟨st, sch, tPad, -, -, hst, hz, y, rfl, rfl, rfl, hTy, hextR⟩ :=
    hTab I own ht hc ho dj y₀ wr hh hc1 hRec (L.draw ω) R o (hDraw ω) hacc
  obtain ⟨hd, hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  obtain ⟨st', hst', rfl, rfl, hret, -, hmp⟩ := setupTables_oneHP hh' hc1 hs
  have hsr : st'.spec.retainRounds = true := (hret.trans (zkTags_retainRounds I)).trans hrr
  have hst'' : Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj (L.draw ω)))
      I.partition I.program (some own) = .ok st' := by
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
  have hpd := decode_points hdecS
  have hmP : y.x.st.mPts = mPts := (hDraw ω).2 y.x.st hst
  have hsz : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp 0 (by simp)).symm.trans hmP)
  have hExt := hextR sess.points (hpd.1.trans rfl) hsz
  rw [hpts] at hExt
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupHidden_hm96P ((zkTags_merkleLeaf I).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam : ZkTabSeam (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R) (Setup.ofCircuit y.x.st) y.sch y.tPad := by
    rw [hTy]
    exact table_seamHP y
  have hlay : ZkTabLay (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R) (Setup.ofCircuit y.x.st) y.sch := by
    rw [hTy]
    exact table_layHP y
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).S := by
    rw [hTy]
    exact ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hExt' : ExtraRealizes (Setup.ofCircuit y.x.st) (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).S
      sess.points (Model.extraClaims (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).S
        (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts) := by
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
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R) o.pts 0
    (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀ s₀.rounds _ tr₀ hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀
    (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R) o.pts 1
    (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁ s₁.rounds _ tr₁ hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁
    (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (tablesAtZKHP (Reg := Reg) dj y₀) (L.draw ω) R o.pts y.x.st.spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess 0 _ _ c hz hms hseam hlay hfold hExt' hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).S
        (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZ (tablesAtZKHP (Reg := Reg) dj y₀) (L.draw ω) R = ex₀.size := by
    unfold nXZ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZ (tablesAtZKHP (Reg := Reg) dj y₀) (L.draw ω) R =
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
  have hopn : opnAtZHP (tablesAtZKHP (Reg := Reg) dj y₀) I own dj (L.draw ω) R o.proofs =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).S
        (tablesAtZKHP (Reg := Reg) dj y₀ (L.draw ω) R).ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * 0]!, (wr (L.draw ω) R o).proofs[2 * 0 + 1]!] := by
    rw [← hprf]
    unfold opnAtZHP
    rw [hst]
    dsimp only
    rw [hz]
    dsimp only
    rw [hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · show PlaysCoin.RepRun _ _ (decAtZ (tablesAtZKHP (Reg := Reg) dj y₀) (L.draw ω) R) o.rep0 [] p₀
    rw [hdecEq, ← show s₀.rounds.toList = o.rep0 from hr₀]
    exact hrr₀
  · show PlaysCoin.RepRun _ _ (decAtZ (tablesAtZKHP (Reg := Reg) dj y₀) (L.draw ω) R) o.rep1 [] p₁
    rw [hdecEq, ← show s₁.rounds.toList = o.rep1 from hr₁]
    exact hrr₁
  · show accZ _ _ _ (outZ hs512 _ _ _ (fun _ => capOfRoot o.rootB) (p₀, p₁) _) = true
    rw [hopn, ← hroot]
    exact hacc'

/-- **The coin server's draw is `DrawOkZKHP` at every draw of the live law** (`drawOkZKH_of_exec` with `own`). -/
theorem drawOkZKHP_of_exec (I : Inputs) (own : ByteArray) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Audit.Law.stratum σs s).card)
    (St : Fin m → Draw.Stratum) (hS : Audit.Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.stratified Kd (Array.ofFn St)) pop) (ω : Ω) :
    DrawOkZKHP I own dj (mPtsOf c.g (∑ s, ks s))
      ((Audit.Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) := by
  refine ⟨fun D hD u hu => ?_, fun st hst => ?_⟩
  · rw [hdj _ D hD, drawOf_proved, drawOf_mem]
    exact ⟨u, hu, rfl⟩
  · rw [setupHidden_mPts_of_provedP hc (fun D hD => by rw [hdj _ D hD, drawOf_proved, drawOf_size]) hst]
    rcases execOS_draw_card_or σs ks St hS Kd pop src ω with hcard | huniv
    · rw [hcard]
    · obtain ⟨ud, hud, -⟩ := setupHidden_mPts_drawnP hst
      rw [hdj _ ud hud, huniv] at hud
      rw [huniv, Finset.card_univ, Fintype.card_fin, sum_ks_of_univ σs ks hks St hS Kd pop hud]

end FlockSoundness.Discharge.ZkPubIn
