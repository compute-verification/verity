import Proofs.Flock.EndToEndHidden.Statement

/-!
# Proof of C-Flock's end-to-end guarantee under hidden outputs

`Flock.SecurityProofs.EndToEndHidden : Flock.Guarantees.EndToEndHidden` is `zk_session_soundHJ_custody_reg` with its
first and last terms in closed form, as `EndToEnd` is `zk_session_soundJ_custody_reg`'s (`Audit.Law.subset_miss`, and
`linkBoundZC` at A2). `Flock.SecurityProofs.EndToEndHiddenDrawn : Flock.Guarantees.EndToEndHiddenDrawn` is
`zk_session_soundHJ_custody_drawn_reg` with its last term in closed form.

`EndToEndHidden_refSetup` shows the hidden-output statements' reference setup `y₀` (theirs and `EndToEndRegistered`'s)
is no extra condition: every draw flock-verify accepts gives one, at any number of tables.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **C-Flock's one-stage audit, end to end, under hidden outputs.** -/
theorem EndToEndHidden : Flock.Guarantees.EndToEndHidden := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h := Discharge.ZkHidden.zk_session_soundHJ_custody_reg I ht hc hU hpub hpt hscope kd hkd hdiv dj y₀ wr dg N Lb
    src hA3 hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **C-Flock's one-stage audit, end to end, under hidden outputs, at the drawn units.** -/
theorem EndToEndHiddenDrawn : Flock.Guarantees.EndToEndHiddenDrawn := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hTags hCust k Rw M hk hRw hM ρ
    hρ hr t' ht' τ qF qS L P vb σ hCR hA2
  have h := Discharge.ZkHidden.zk_session_soundHJ_custody_drawn_reg I ht hc hU hpub hpt hscope kd hdiv dj y₀ wr dg N
    Lb src hTags hCust hk hRw hM hρ hr ht' τ (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

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
