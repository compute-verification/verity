import Proofs.Flock.EndToEndHidden.Statement

/-!
# Proof of C-Flock's end-to-end guarantee under hidden outputs

`Flock.SecurityProofs.EndToEndHiddenModel : Flock.Guarantees.EndToEndHiddenModel` is `ZkLink.zk_flock_countPr_rowsC`
at the hidden-output tables, its knowledge term at the finders' collision probabilities (`ksAvgZC_le_pr`).
`Flock.SecurityProofs.EndToEndHidden : Flock.Guarantees.EndToEndHidden` is its corollary (`EndToEndHidden_of_model`):
the model with no registered read (`EndToEndHiddenModel_noReads`) at the key program's partition and the public file's
rows, through the live session
(`zk_headline_exec_tables`), with its miss term in closed form (`Audit.Law.subset_miss`), and its knowledge and link
terms the finders' collision probabilities (`ksAvgPrZ`, `linkPrZC`).
`Flock.SecurityProofs.EndToEndHiddenDrawn : Flock.Guarantees.EndToEndHiddenDrawn` is
`zk_session_soundHJ_custody_drawn_reg`. Each reads the circuit the descriptor unrolls to, which `HmOut.parse` reads from the file (`Flock.Desc.parse_flat`). `EndToEndHiddenFlat` and
`EndToEndHiddenDrawnFlat` are the two at the flat descriptor `Desc.single c` (`Flock.Desc.parse_single`).

`EndToEndHidden_refSetup` shows the hidden-output statements' reference setup `y₀` (theirs and `EndToEndRegistered`'s)
is no extra condition: every draw flock-verify accepts gives one, at any number of tables.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

open FlockSoundness.Discharge FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec
  FlockSoundness.Discharge.ZkHidden in
/-- **C-Flock's one-stage audit under hidden outputs, at the model level.** -/
theorem EndToEndHiddenModel : Flock.Guarantees.EndToEndHiddenModel := by
  intro n Reg I mPts dj S₀ d₀ y₀ C P decode Dh Hc vb hlow hown canon hcan isRead wo L k Rw M hk hRw hM ρ hρ hr K₀ σ
  exact (zk_flock_countPr_rowsC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb canon isRead wo hlow hown hcan
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) K₀ σ).trans
    (add_le_add (add_le_add le_rfl (ksAvgZC_le_pr execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ))) le_rfl)

open FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden in
open Classical in
/-- **`EndToEndHiddenModel` with no registered read**: with every opening canonical and no registered read, a unit
wrong on the registration (`wrongRegZ`) is wrong in the model (`wrongRegZ_subset_wrongRegZc`) and `δ_rows = 0`
(`rowsPrZC_noReads`), so the count curve at `wrongRegZ` has the bound `miss + ε_ks + δ_link`. -/
theorem EndToEndHiddenModel_noReads (hm : Flock.Guarantees.EndToEndHiddenModel) (n : ℕ) (Reg : Type) (I : Inputs)
    (mPts : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (C : FlockSoundness.Audit.Circuit) (P : FlockSoundness.Audit.Partition C n)
    (decode : ZkSession.DecoderZK (C := C) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀))
    (Dh : Type) (Hc : List UInt8 → Dh) (vb : ValueBindingZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode Hc)
    (hlow : ZkSession.LoweringSoundZK execArith P (planZHJ dj y₀) (tabZHJ dj y₀) decode)
    (hown : ∀ u, P.ReadsOwn u) (L : Law n)
    (k Rw M : ℕ) (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) (ρ : ℝ) (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀)))) :
    prob (fun o => o.1 = true ∧
        K₀ ≤ (wrongRegZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ)
          (XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀))) σ ≤
      L.miss K₀ + ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        linkPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb k Rw M ρ (reg σ) (cont σ) := by
  have h := hm n Reg I mPts dj S₀ d₀ y₀ C P decode Dh Hc vb hlow hown (fun _ _ => True)
    (fun _ _ _ _ _ _ _ _ => trivial) (fun _ => False) (fun _ => default) L k Rw M hk hRw hM ρ hρ hr K₀ σ
  rw [rowsPrZC_noReads, add_zero] at h
  refine le_trans (Game.prob_mono (fun o ho => ⟨ho.1, ho.2.trans (Finset.card_le_card ?_)⟩) _ _) h
  exact (wrongRegZ_subset_wrongRegZc execArith P _ _ vb _ _ _).trans Finset.subset_union_left

open FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden in
/-- **`EndToEndHiddenModel` gives `EndToEndHidden`**: the live audit is the audit game at the simulated prover
(`zk_headline_exec_tables`), where the model applies at the key program's partition, the binding of the public file's
rows and the lowering the placement gives, and the live subset draw misses at most as often as the uniform one. -/
theorem EndToEndHidden_of_model (hm : Flock.Guarantees.EndToEndHiddenModel) : Flock.Guarantees.EndToEndHidden := by
  intro n Reg I ht D hU pub pinC pinP hFiles hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k
    Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ L P vb σ
  have hc := Flock.Desc.parse_flat hFiles.1
  have h := zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
        fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ _ _
    ((EndToEndHiddenModel_noReads hm n Reg I _ dj S₀ d₀ y₀ _ P _ _ _ vb
      (ZkSession.loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJ dj y₀ ht hc hscope hU))
      (Layout.keyProg_readsOwn D.unroll hU n _) L k Rw M hk hRw hM ρ hρ hr K₀ σ).trans
      (add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀) le_rfl) le_rfl))
  rw [Law.subset_miss hkd hK₀] at h
  exact h

/-- **C-Flock's one-stage audit, end to end, under hidden outputs.** -/
theorem EndToEndHidden : Flock.Guarantees.EndToEndHidden := EndToEndHidden_of_model EndToEndHiddenModel

/-- **C-Flock's one-stage audit, end to end, under hidden outputs, at the drawn units.** -/
theorem EndToEndHiddenDrawn : Flock.Guarantees.EndToEndHiddenDrawn := by
  intro n Reg I ht D hD hU pub hpub hpt hscope kd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ
    hρ hr τ L P vb σ
  exact Discharge.ZkHidden.zk_session_soundHJ_custody_drawn_reg I ht (Flock.Desc.parse_flat hD) hU hpub hpt hscope kd
    hdiv dj y₀ wr dg N Lb src hTags hCust hk hRw hM hρ hr τ

/-- **C-Flock's one-stage audit, end to end, under hidden outputs, at a flat circuit file.** -/
theorem EndToEndHiddenFlat : Flock.Guarantees.EndToEndHiddenFlat := by
  intro n Reg I ht c hc hU pub hpub
  exact EndToEndHidden n Reg I ht (.single c) hU pub (Flock.Sha512.hash I.circuitFile) (Flock.Sha512.hash I.publicFile)
    (And.intro (Flock.Desc.parse_single hc) (And.intro hpub (And.intro rfl rfl)))

/-- **C-Flock's one-stage audit, end to end, under hidden outputs, at the drawn units, at a flat circuit file.** -/
theorem EndToEndHiddenDrawnFlat : Flock.Guarantees.EndToEndHiddenDrawnFlat := by
  intro n Reg I ht c hc
  exact EndToEndHiddenDrawn n Reg I ht (.single c) (Flock.Desc.parse_single hc)

/-- `EndToEndHiddenPerTest` gives its flat form, at `Desc.single c`. -/
theorem EndToEndHiddenPerTestFlat_of (h : Flock.Guarantees.EndToEndHiddenPerTest) :
    Flock.Guarantees.EndToEndHiddenPerTestFlat := by
  intro n Reg I ht c hc
  exact h n Reg I ht (.single c) (Flock.Desc.parse_single hc)

open FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden in
/-- **The hidden-output statements' reference setup exists at every draw flock-verify accepts, at any number of
tables.** `EndToEnd_refSetup` under hidden outputs: on untyped inputs under `TagsOkZHJ`, whose circuit file
`HmOut.parse` accepts as `c` (`hU`, `hscope`: its unit returns the outputs), at the record's draw file
(`DrawFileZK dj (subset kd) pop`), the session's tables dividing `kd`, at a transcript flock-verify `--zk` accepts whose
record's draw is `dj S`, table 0 has a `DrawSetupZHJ` at `mPtsOf c.g (kd / nTab I)` link points, at its part of `S` and
its draw: the `y₀` of `EndToEndHidden`, `EndToEndHiddenDrawn`, `EndToEndRegistered` and `EndToEndRegisteredDrawn`, at
`S₀ = partS I (dj S) S 0` and `d₀ = drawJ I (dj S) 0`. -/
theorem EndToEndHidden_refSetup {n : ℕ} (I : Inputs) (ht : I.tags.typed = false) (hT : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : Flock.HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop)
    {S : Finset (Fin n)} {t : Exec.Transcript} (hacc : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
      (drawJ I (dj S) 0)) := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hT.1 hd hs
  have hj : 0 < sts.size := hsz ▸ nTab_pos I
  have hst := hstJ 0 hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨_hs0, _, hz0, -⟩ := hall 0 (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Flock.Zk.schedule sts[0].m = .ok (schs[0].1, schs[0].2) := hz0
  obtain ⟨_, hf⟩ := fast100_of_schedule hz'
  have hcJ : Flock.HmOut.parse (inputsJ I 0).tags (inputsJ I 0).circuitFile (inputsJ I 0).tables = .ok c :=
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  have hDraw := drawOkZKHJ_of_file I ht hc kd pop hdiv dj hdj S ⟨0, nTab_pos I⟩
  obtain ⟨_, -, hdr⟩ := Hidden.setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[0].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := HiddenExec.drawSetupH_of_setupHidden (I := inputsJ I 0) (S := partS I (dj S) S ⟨0, nTab_pos I⟩)
    (mPts := mPtsOf c.g (kd / nTab I)) (typedJ ht 0) hcJ (Layout.scopeOkH_spec hU hscope).2.1 hst hf hD
    (hDraw.1 D hD) (hDraw.2 _ hst)
  obtain ⟨y, -, -, -⟩ := drawSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape 0 hj)
    (sch := schs[0].1) (tPad := schs[0].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

end Flock.SecurityProofs
