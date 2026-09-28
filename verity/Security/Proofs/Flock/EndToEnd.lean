import Proofs.Flock.EndToEnd.Statement

/-!
# Proof of C-Flock's end-to-end guarantee

`Flock.SecurityProofs.EndToEnd : Flock.Guarantees.EndToEnd` is `zk_session_soundJ_custody_reg` with its first and last
terms in closed form:
* `zk_session_soundJ_custody_reg`: one `--zk` session of `J` tables at the record's custody, at the live subset draw, is
  sound against the public file, `Pr[flock-verify accepts ∧ at least K₀ units wrong on it] ≤ miss(K₀) + ε_ks + δ_link`;
* `Audit.Law.subset_miss`: the uniform `kd`-subset's `miss(K₀)` is `C(n − K₀, kd) / C(n, kd)` at `K₀ ≤ n`;
* `linkBoundZC`: at a prover whose link finders satisfy A2 (`LinkCRZC`), `δ_link` is its closed form.

`EndToEnd_refSetup` shows the statement's reference setup `y₀` is no extra condition: every draw flock-verify accepts
gives one, at any number of tables.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **C-Flock's one-stage audit, end to end.** -/
theorem EndToEnd : Flock.Guarantees.EndToEnd := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h := Discharge.Composed.zk_session_soundJ_custody_reg I ht hc hU hpub hpt hscope kd hkd hdiv dj y₀ wr dg N Lb src hA3
    hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

open FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.Composed in
/-- **`EndToEnd`'s reference setup exists at every draw flock-verify accepts, at any number of tables.** At the record's
draw file (`DrawFileZK dj (subset kd) pop`), the session's tables dividing `kd`, at a transcript flock-verify `--zk`
accepts under `TagsOkZJ` whose record's draw is `dj S`, table 0 has a `DrawSetupZJ` at `mPtsOf c.g (kd / nTab I)` link
points, at its part of `S` and its draw: `EndToEnd`'s `y₀`, at `S₀ = partS I (dj S) S 0` and `d₀ = drawJ I (dj S) 0`.
A setup of the whole draw at those link points, the binder `y₀` had before, exists only where `mPtsOf c.g kd` is the
same number, which fails once `nTab I ≥ 2` and `⌈kd / c.g⌉ > 8`. -/
theorem EndToEnd_refSetup {n : ℕ} (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : Flock.HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) pop) (hT : TagsOkZJ I)
    {S : Finset (Fin n)} {t : Exec.Transcript} (hacc : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
      (drawJ I (dj S) 0)) := by
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJ hT.1 hd hs
  have hj : 0 < sts.size := hsz ▸ nTab_pos I
  have hst := hstJ 0 hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨_hs0, _, hz0, -⟩ := hall 0 (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Flock.Zk.schedule sts[0].m = .ok (schs[0].1, schs[0].2) := hz0
  obtain ⟨-, -, _, hf⟩ := schedule_spec hz'
  have hDraw := drawOkZKJ_of_file I ht hc kd pop hdiv dj hdj S ⟨0, nTab_pos I⟩
  obtain ⟨_, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I 0) (S := partS I (dj S) S ⟨0, nTab_pos I⟩)
    (mPts := mPtsOf c.g (kd / nTab I)) (typedJ ht 0) hst hf hD (hDraw.1 D hD) (hDraw.2 _ hst)
  obtain ⟨y, -⟩ := drawSetupZJ_of (typedJ ht 0) x (sch := schs[0].1) (tPad := schs[0].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

end Flock.SecurityProofs
