module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.NonVacuousV2
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import all Lean.Data.Json.Basic

@[expose] public section

set_option Elab.async false

/-! The hidden-output headlines over `sha512/row-seg/v1` rows, alone and mixed with `sha512/row/v2` rows, hold of a
circuit; not covered, as for `zk_session_soundHJ_nonvacuous_v2`: a session over the witnesses, `hRec`, `hpub`, `hpt`,
`hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed

variable {n : ℕ}

def witnessSegPort (name : String) (bits chunks : ℕ) : Flock.Port :=
  { name := name, words := bits / 16, chunks := chunks, nb := 0, bits := bits, v2 := true,
    segs := some (HmRow.segShape bits).1 }

def witnessSeg : Flock.Circuit :=
  { witnessV2 with name := "witness-seg", ports := #[witnessSegPort "w" 1024 2, witnessV2Port "y"] }

def witnessSegOut : Flock.Circuit :=
  { witnessSeg with name := "witness-seg-out", ports := #[witnessSegPort "w" 1024 2, witnessSegPort "y" 1024 2] }

def witnessSegBad : Flock.Circuit :=
  { witnessSeg with ports := #[witnessSegPort "w" 1024 2, witnessSegPort "y" 1024 1] }

def witnessSegLate : Flock.Circuit :=
  { witnessSeg with ports := #[witnessSegPort "w" 896 2, witnessV2Port "y"] }

def witnessMix : Flock.Circuit :=
  { witnessV2 with
    name := "witness-mix", ports := #[witnessSegPort "a" 17408 21, witnessV2Port "b", witnessV2Port "y"] }

theorem witnessSeg_scope :
    witnessSeg.ports.any (·.segs.isSome) = true ∧ Layout.scopeOkH witnessSeg = true ∧
      Layout.scopeOk witnessSeg = false ∧ Layout.scopeOkH witnessSegBad = false ∧
      Layout.scopeOkH witnessSegLate = false := by
  decide +kernel

theorem witnessSegOut_scope :
    (witnessSegOut.ports[1]!.segs.isSome) = true ∧ Layout.scopeOkH witnessSegOut = true ∧
      Layout.scopeOk witnessSegOut = false := by
  decide +kernel

theorem witnessMix_scope :
    witnessMix.ports.any (·.segs.isSome) = true ∧ witnessMix.ports.any (fun p => p.v2 && p.segs.isNone) = true ∧
      (witnessMix.ports[0]!.segs.map (·.size)) = some 3 ∧ Layout.scopeOkH witnessMix = true ∧
      Layout.scopeOk witnessMix = false := by
  decide +kernel

theorem zk_session_soundHJ_nonvacuous_seg (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok witnessSeg) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Built witnessSeg.unit ∧ Layout.scopeOkH witnessSeg = true ∧ Layout.scopeOk witnessSeg = false ∧
      Hidden.Any.RowsFitS witnessSeg ∧
      Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf witnessSeg.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I), Nonempty
        (SampleSetupZHJ (inputsJ I j) (mPtsOf witnessSeg.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf witnessSeg.g (kd / nTab I)) S' := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have P := parse_facts (parse_untyped ht' hc).1
  obtain ⟨ls, hls⟩ := P.check
  have hU : Built witnessSeg.unit := unit_built P (check_facts P.typed hls)
  obtain ⟨-, hH, hS, -, -⟩ := witnessSeg_scope
  obtain ⟨hy, hall, hok⟩ := zk_session_soundHJ_nonvacuous I ht hTags hc hU hH kd hdiv dj hdj h hd
  exact ⟨hU, hH, hS, rowsFit_of_scopeH ht' hc hH, hy, hall, hok⟩

theorem zk_session_soundHJ_nonvacuous_segOut (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok witnessSegOut) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Built witnessSegOut.unit ∧ Layout.scopeOkH witnessSegOut = true ∧ Layout.scopeOk witnessSegOut = false ∧
      Hidden.Any.RowsFitS witnessSegOut ∧
      Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf witnessSegOut.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I), Nonempty
        (SampleSetupZHJ (inputsJ I j) (mPtsOf witnessSegOut.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf witnessSegOut.g (kd / nTab I)) S' := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have P := parse_facts (parse_untyped ht' hc).1
  obtain ⟨ls, hls⟩ := P.check
  have hU : Built witnessSegOut.unit := unit_built P (check_facts P.typed hls)
  obtain ⟨-, hH, hS⟩ := witnessSegOut_scope
  obtain ⟨hy, hall, hok⟩ := zk_session_soundHJ_nonvacuous I ht hTags hc hU hH kd hdiv dj hdj h hd
  exact ⟨hU, hH, hS, rowsFit_of_scopeH ht' hc hH, hy, hall, hok⟩

theorem zk_session_soundHJ_nonvacuous_mix (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok witnessMix) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Built witnessMix.unit ∧ Layout.scopeOkH witnessMix = true ∧ Layout.scopeOk witnessMix = false ∧
      Hidden.Any.RowsFitS witnessMix ∧
      Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf witnessMix.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I), Nonempty
        (SampleSetupZHJ (inputsJ I j) (mPtsOf witnessMix.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf witnessMix.g (kd / nTab I)) S' := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have P := parse_facts (parse_untyped ht' hc).1
  obtain ⟨ls, hls⟩ := P.check
  have hU : Built witnessMix.unit := unit_built P (check_facts P.typed hls)
  obtain ⟨-, -, -, hH, hS⟩ := witnessMix_scope
  obtain ⟨hy, hall, hok⟩ := zk_session_soundHJ_nonvacuous I ht hTags hc hU hH kd hdiv dj hdj h hd
  exact ⟨hU, hH, hS, rowsFit_of_scopeH ht' hc hH, hy, hall, hok⟩

end FlockSoundness.Discharge.ZkHidden
