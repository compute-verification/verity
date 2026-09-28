import Proofs.Flock.Soundness.Discharge.ZkHidden.NonVacuousJ
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
import Proofs.Flock.Soundness.ExecSetup

set_option Elab.async false

/-!
# The hidden-output headlines over `sha512/row/v2` rows hold of a circuit

`Layout.scopeOkH` widens `Layout.scopeOk` by circuits whose rows are `sha512/row/v2` or `sha512/row-seg/v1` rows. The
pins here say the widening by v2 rows alone is not empty and the widened headlines' premises meet there, so their
guarantees over v2 rows hold of something (row-seg rows and a mix: `NonVacuousSeg.lean`).

* `witnessV2`: a unit of one 1,024-bit input row whose one computed row is the AND of the row's bits 0 and 1, with its
  input row `w` and its output row `y` each a `sha512/row/v2` row of 64 u16 words (1,024 bits) in one compression, one
  output port per output row (the toy bit-row circuit's shape).
* `witnessV2_scope`: every row of `witnessV2` is a `sha512/row/v2` row, `scopeOkH` admits it and `scopeOk` refuses it,
  checked by evaluation.
* `zk_session_soundHJ_nonvacuous_v2`: at a session `verify --zk` accepts whose circuit file parses as `witnessV2`, the
  premises the hidden-output headlines read of the circuit hold together (`Built`, `scopeOkH`, the rows that fit,
  `Hidden.Any.RowsFitS`, with every row a v2 row), with `zk_session_soundHJ_nonvacuous`'s `y₀` and draw hypotheses.

Not covered: that a session over `witnessV2` exists (the verifier accepting one; the `_bit_row_toy` sessions show it at
run time), `hRec`, `hpub`, `hpt`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`, as for
`zk_session_soundHJ_nonvacuous`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed

variable {n : ℕ}

/-- One column per row: row 1024 is `[col]`, every other row empty. -/
def witnessV2Rows (col : ℕ) : Sparse where
  rows := 2048
  width := 1026
  data := ⟨#[col.toUInt8, 0, 0, 0]⟩
  start := Array.replicate 1025 0
  len := (Array.replicate 1024 0).push 1

/-- A unit of one 1,024-bit input row (input words 0 … 7, columns 0 … 1023), one computed row (column 1024, the AND of
the row's bits 0 and 1), the constant (column 1025) and one output bit. -/
def witnessV2Net : Net where
  name := "witness-v2"
  useful := 1026
  constPos := 1025
  inGroups := #[(⟨0, 8, #[1024]⟩ : Flock.Group)]
  outGroups := #[(⟨8, 1, #[1]⟩ : Flock.Group)]
  inWords := 8
  a := witnessV2Rows 0
  b := witnessV2Rows 1
  unitLog := 11

/-- A `sha512/row/v2` row of 64 u16 words (1,024 bits) in one compression. -/
def witnessV2Port (name : String) : Flock.Port :=
  { name := name, words := 64, chunks := 1, nb := 1, bits := 1024, v2 := true }

/-- A circuit whose unit is `witnessV2Net`, whose input row `w` and output row `y` are `sha512/row/v2` rows, one output
port per output row. -/
def witnessV2 : Flock.Circuit :=
  { (default : Flock.Circuit) with
    name := "witness-v2", nets := #[("unit", witnessV2Net)], unitNet := 0, outNet := 0,
    ports := #[witnessV2Port "w", witnessV2Port "y"], outPorts := #[("y", 1)], outRowPorts := #[1], g := 1, upv := 1 }

/-- **`scopeOkH` admits a circuit of `sha512/row/v2` rows that `scopeOk` refuses.** -/
theorem witnessV2_scope :
    witnessV2.ports.all (·.v2) = true ∧ Layout.scopeOkH witnessV2 = true ∧ Layout.scopeOk witnessV2 = false := by
  decide +kernel

/-- **The hidden-output headlines' circuit premises and draw hypotheses hold together at `witnessV2`.** At the coin
server's draw file (`DrawFileZK`, `subset kd`, the tables dividing `kd`) and one session `verify --zk` accepts at the
draw `dj S`, on untyped inputs under `TagsOkZHJ`'s tags whose circuit file parses as `witnessV2`: its unit is built,
`scopeOkH` admits it and `scopeOk` refuses it, its rows fit with every row a `sha512/row/v2` row
(`Hidden.Any.RowsFitS`, `Hidden.rowV2`), and at `mPtsOf witnessV2.g (kd / nTab I)` link points there is a `y₀` (table
0's setup), every table is set up, and `DrawOkZKHJ` holds at every draw (`zk_session_soundHJ_nonvacuous`, which applies
`zk_session_soundHJ` to them). -/
theorem zk_session_soundHJ_nonvacuous_v2 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok witnessV2) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Built witnessV2.unit ∧ Layout.scopeOkH witnessV2 = true ∧ Layout.scopeOk witnessV2 = false ∧
      Hidden.Any.RowsFitS witnessV2 ∧ Hidden.rowV2 witnessV2 = true ∧
      Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf witnessV2.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I), Nonempty
        (DrawSetupZHJ (inputsJ I j) (mPtsOf witnessV2.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf witnessV2.g (kd / nTab I)) S' := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have P := parse_facts (parse_untyped ht' hc).1
  obtain ⟨ls, hls⟩ := P.check
  have hU : Built witnessV2.unit := unit_built P (check_facts P.typed hls)
  obtain ⟨-, hH, hS⟩ := witnessV2_scope
  obtain ⟨hy, hall, hok⟩ := zk_session_soundHJ_nonvacuous I ht hTags hc hU hH kd hdiv dj hdj h hd
  have hv2 : Hidden.rowV2 witnessV2 = true := by decide +kernel
  exact ⟨hU, hH, hS, rowsFit_of_scopeH ht' hc hH, hv2, hy, hall, hok⟩

end FlockSoundness.Discharge.ZkHidden
