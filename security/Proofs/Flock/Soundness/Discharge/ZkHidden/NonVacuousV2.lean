module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
public import Proofs.Flock.Soundness.Discharge.ZkHidden.NonVacuousJ
public import Proofs.Flock.Soundness.Discharge.ZkLink.SampleOk
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import all Lean.Data.Json.Basic
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
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup

@[expose] public section

set_option Elab.async false

/-! The hidden-output headlines over `sha512/row/v2` rows hold of a circuit (`witnessV2`). -/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed

variable {n : ℕ}

def witnessV2Rows (col : ℕ) : Sparse where
  rows := 2048
  width := 1026
  data := ⟨#[col.toUInt8, 0, 0, 0]⟩
  start := Array.replicate 1025 0
  len := (Array.replicate 1024 0).push 1

def witnessV2Net : Net where
  name := "witness-v2"
  useful := 1026
  constPos := 1025
  inGroups := #[(⟨0, 8, #[1024]⟩ : Flock.Group)]
  outGroups := #[(⟨8, 1, #[1]⟩ : Flock.Group)]
  inWords := 8
  aT := .pure (witnessV2Rows 0)
  bT := .pure (witnessV2Rows 1)
  unitLog := 11

def witnessV2Port (name : String) : Flock.Port :=
  { name := name, words := 64, chunks := 1, nb := 1, bits := 1024, v2 := true }

def witnessV2 : Flock.Circuit :=
  { (default : Flock.Circuit) with
    name := "witness-v2", nets := #[("unit", witnessV2Net)], unitNet := 0, outNet := 0,
    ports := #[witnessV2Port "w", witnessV2Port "y"], outPorts := #[("y", 1)], outRowPorts := #[1], g := 1, upv := 1 }

theorem witnessV2_scope :
    witnessV2.ports.all (·.v2) = true ∧ Layout.scopeOkH witnessV2 = true ∧ Layout.scopeOk witnessV2 = false := by
  decide +kernel

/-- Not covered: that a session over `witnessV2` exists, `hRec`, `hpub`, `hpt`, `hkd`, `hA3`, the count curve's
parameters and `TableCRZ`. -/
theorem zk_session_soundHJ_nonvacuous_v2 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok witnessV2) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Built witnessV2.unit ∧ Layout.scopeOkH witnessV2 = true ∧ Layout.scopeOk witnessV2 = false ∧
      Hidden.Any.RowsFitS witnessV2 ∧ Hidden.rowV2 witnessV2 = true ∧
      Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf witnessV2.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I), Nonempty
        (SampleSetupZHJ (inputsJ I j) (mPtsOf witnessV2.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf witnessV2.g (kd / nTab I)) S' := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have P := parse_facts (parse_untyped ht' hc).1
  obtain ⟨ls, hls⟩ := P.check
  have hU : Built witnessV2.unit := unit_built P (check_facts P.typed hls)
  obtain ⟨-, hH, hS⟩ := witnessV2_scope
  obtain ⟨hy, hall, hok⟩ := zk_session_soundHJ_nonvacuous I ht hTags hc hU hH kd hdiv dj hdj h hd
  have hv2 : Hidden.rowV2 witnessV2 = true := by decide +kernel
  exact ⟨hU, hH, hS, rowsFit_of_scopeH ht' hc hH, hv2, hy, hall, hok⟩

end FlockSoundness.Discharge.ZkHidden
