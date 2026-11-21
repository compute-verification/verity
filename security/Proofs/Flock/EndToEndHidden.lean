module

public import Proofs.Flock.EndToEndHidden.Statement
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Spot.Sampling.SamplingDesign
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched

/-!
# Flock's end-to-end statements under hidden outputs: the per-test form and the reference setup

`EndToEndHiddenPerTestFlat_of`: `EndToEndHiddenPerTest` gives its flat form, at the flat descriptor `Desc.single c`
(`Flock.Desc.parse_single`).

`EndToEndHidden_refSetup` shows the hidden-output statements' reference setup `y₀` (theirs and `EndToEndRegistered`'s)
is no extra condition: every sample verity-verify-proof accepts gives one, at any number of tables.
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness Spot.Sampling

/-- `EndToEndHiddenPerTest` gives its flat form, at `Desc.single c`. -/
theorem EndToEndHiddenPerTestFlat_of (h : Flock.Guarantees.EndToEndHiddenPerTest) :
    Flock.Guarantees.EndToEndHiddenPerTestFlat := by
  intro n Reg I ht c hc
  exact h n Reg I ht (.single c) (Flock.Desc.parse_single hc)

open FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden in
/-- **The hidden-output statements' reference setup exists at every sample verity-verify-proof accepts, at any number of
tables.** `EndToEnd_refSetup` under hidden outputs: on untyped inputs under `TagsOkZHJ`, whose circuit file
`HmOut.parse` accepts as `c` (`hU`, `hscope`: its unit returns the outputs), at the record's sample file
(`SampleFileZK dj (subset kd) pop`), the session's tables dividing `kd`, at a transcript verity-verify-proof `--zk` accepts whose
record's sample is `dj S`, table 0 has a `SampleSetupZHJ` at `mPtsOf c.g (kd / nTab I)` link points, at its part of `S` and
its sample: the `y₀` of `EndToEndHidden`, `EndToEndRegistered` and `EndToEndRegisteredSampled`, at
`S₀ = partS I (dj S) S 0` and `d₀ = drawJ I (dj S) 0`. -/
theorem EndToEndHidden_refSetup {n : ℕ} (I : Inputs) (ht : I.tags.typed = false) (hT : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : Flock.HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop)
    {S : Finset (Fin n)} {t : Exec.Transcript} (hacc : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
      (drawJ I (dj S) 0)) := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hT.1 hd hs
  have hj : 0 < sts.size := hsz ▸ nTab_pos I
  have hst := hstJ 0 hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨_hs0, _, hz0, -⟩ := hall 0 (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Flock.Zk.schedule sts[0].wideLeaf sts[0].m = .ok (schs[0].1, schs[0].2) := hz0
  obtain ⟨_, hf⟩ := fast100_of_schedule hz'
  have hcJ : Flock.HmOut.parse (inputsJ I 0).tags (inputsJ I 0).circuitFile (inputsJ I 0).tables = .ok c :=
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  have hDraw := sampleOkZKHJ_of_file I ht hc kd pop hdiv dj hdj S ⟨0, nTab_pos I⟩
  obtain ⟨_, -, hdr⟩ := Hidden.setupHidden_sampled hst
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec (c := { sts[0].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := HiddenExec.sampleSetupH_of_setupHidden (I := inputsJ I 0) (S := partS I (dj S) S ⟨0, nTab_pos I⟩)
    (mPts := mPtsOf c.g (kd / nTab I)) (typedJ ht 0) hcJ (Layout.scopeOkH_spec hU hscope).2.1 hst hf hD
    (hDraw.1 D hD) (hDraw.2 _ hst)
  obtain ⟨y, -, -, -⟩ := sampleSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape 0 hj)
    (sch := schs[0].1) (tPad := schs[0].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

end Flock.SecurityProofs
