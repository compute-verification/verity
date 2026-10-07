import Proofs.Flock.Soundness.Discharge.ZkPubIn.ComposedJ
import Proofs.Flock.Soundness.Discharge.ZkPubIn.TablesJ
import Proofs.Flock.Soundness.Discharge.Canonical.Session

set_option Elab.async false

/-!
# `RunsZC` at the canonical `--zk` tables of `J` tables with the verifier's public inputs

`Canonical.runsZCJH_liveC` and the `J`-table record, draw and openings it reads (`RecordsLiveZKJ`, `DrawOkZKHJ`,
`opnAtZHJ`) with `--public-inputs F` (`own` F's bytes): each reads flock-verify's acceptance `AcceptsZKP I own` and
setupHidden's statement at `some own` where the original reads `AcceptsZK I` and setupHidden's at `none`.

* `RecordsLiveZKJP`, `DrawOkZKHJP`, `opnAtZHJP`: the originals at `AcceptsZKP I own`, `setupOfP` and
  `Stmt.setupHidden … (some own)`;
* `drawSetupHP_of_setupHiddenC`: `drawSetupHP_of_setupHidden` at a canonical circuit, without `outNet = unitNet`
  (`formatOk_g_pos`);
* `runsZCJH_liveCP`: `runsZCJH_liveC`'s proof at these, concluding `RunsOnZCJA` at `AcceptsZKP I own` and `planZHJP`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.Canonical (formatOk formatOk_g_pos)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-- **`wr` is the coin server's record of the live `--zk --public-inputs F` session of `J` tables** (`RecordsLiveZKJ`
at `AcceptsZKP I own` and `setupOfP`). -/
def RecordsLiveZKJP (I : Inputs) (own : ByteArray) (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript) : Prop :=
  ∀ S R o, AcceptsZKP I own (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOfP (zkInputs I) own (wr S R o).record = .ok (spec, sts) →
      Record.decode spec (wr S R o).record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin (Jf S R), sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

/-- **The coin server's draw at `S`, at `J` tables with the verifier's public inputs** (`DrawOkZKHJ` with
`setupHidden … (some own)`). -/
def DrawOkZKHJP (I : Inputs) (own : ByteArray) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) :
    Prop :=
  ∀ j : Fin (nTab I),
    (∀ D, Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D → ∀ u ∈ partS I (dj S) S j, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j)) I.partition I.program (some own) = .ok st → st.mPts = mPts

/-- **The `--zk` openings at table `j` with the verifier's public inputs** (`opnAtZHJ` at setupHidden's statement with
`own`). -/
noncomputable def opnAtZHJP (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs) (own : ByteArray)
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray)
    (j : Fin (planZ S R).length) : OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (tagsAt (zkTags I) I.count j.val) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j.val)) I.partition I.program (some own) with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZJ ((planZ S R).get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | .error _ => default
  | .error _ => default

/-- **A `DrawSetupHP` from setupHidden's acceptance with `own` at a canonical circuit** (`drawSetupHP_of_setupHidden`
without `outNet = unitNet`: `0 < c.g` is `formatOk`'s, `formatOk_g_pos`). -/
theorem drawSetupHP_of_setupHiddenC {I : Inputs} {own : ByteArray} {S : Finset (Fin n)} {d : Lean.Json}
    {c : Flock.Circuit} {st : Stmt} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hfmt : formatOk c = true)
    (h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program
      (some own) = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : DrawSetupHP I own mPts S d, x.st = st := by
  have hcst : st.c = c := Except.ok.inj ((setupHidden_parseCP h).symm.trans hc)
  have hg : 0 < st.c.g := hcst ▸ formatOk_g_pos hfmt
  obtain ⟨_, pop, -, hdr⟩ := setupHidden_drawnP h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.drawn_spec hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := setupHidden_nP h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact Integrate.fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := Integrate.model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts, setupHidden_wfP h⟩, rfl⟩

/-- **`RunsZC` at the live canonical `--zk` tables of a session of `J` tables with the verifier's public inputs, at
the law's draws** (`runsZCJH_liveC` at `AcceptsZKP I own`, `planZHJP`, `opnAtZHJP`, `RecordsLiveZKJP` and
`DrawOkZKHJP`). -/
theorem runsZCJH_liveCP (I : Inputs) (own : ByteArray) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hfmt : formatOk c = true)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length mPts →
      Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I)
    (hRec : RecordsLiveZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKHJP I own dj mPts (L.draw ω)) :
    RunsOnZCJA (K := 𝕂) execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj)
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hd, hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, hshape, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  have hpts2 := setupTables_pointsHP hh' hs
  have hsr : spec.retainRounds = true := ((setupTables_retainHP hh' hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := setupOfP_tableJH hh hd hs
  have hjn : j.val < nTab I := planZHJP_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, cap, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨sch₀, hf⟩ := fast100_of_schedule hz'
  have hcJ : HmOut.parse (inputsJ I j.val).tags (inputsJ I j.val).circuitFile (inputsJ I j.val).tables = .ok c :=
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  obtain ⟨_, pop, -, hdr⟩ := setupHidden_drawnP hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetupHP_of_setupHiddenC (I := inputsJ I j.val) (own := own)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) hcJ hfmt hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZHJP_of x (by rw [hx]; exact array_forM_ok hshape j.val hj)
    (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZHJP (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZHJP (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZHJP dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZHJP
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZHJP.table_eq _ _
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
    setupHidden_hm96P (((tagsAt_merkleLeaf _ _ _).trans (zkTags_merkleLeaf I)).trans hl) hst
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
  have hdecEq : decAtZJ (planZHJP (Reg := Reg) dj y₀) (L.draw ω) R j =
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
  have hopn : opnAtZHJP (planZHJP (Reg := Reg) dj y₀) I own dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZHJP
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
    exact outZC_of_accZ hacc'

end FlockSoundness.Discharge.ZkPubIn
