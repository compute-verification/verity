import Proofs.Flock.Soundness.Discharge.ZkHidden.NonVacuousV2

set_option Elab.async false

/-!
# The hidden-output headlines over `sha512/row-seg/v1` rows, alone and mixed with `sha512/row/v2` rows, hold of a circuit

`Layout.scopeOkH` admits circuits whose input rows are each a `sha512/row/v2` or a `sha512/row-seg/v1` row, in any mix,
the output rows `sha512/row/v2`. The pins here say both widenings are not empty and the widened headlines' premises meet
there, so their guarantees over row-seg rows, and over a PoUW tile's mix of them, hold of something.

* `witnessSeg`: `witnessV2` with its input row `w` a `sha512/row-seg/v1` row of 1,024 bits (one segment of two
  compressions, the outer message in the last). `witnessSeg_scope`: `scopeOkH` admits it and `scopeOk` refuses it, and
  `scopeOkH` refuses it with its output row segmented (`witnessSegOut`) or with `w` of 896 bits, whose segment's 112
  bytes and SHA-512's 17 bytes of padding spill past its last block (`witnessSegLate`).
* `witnessMix`: a tile's rows: a `sha512/row-seg/v1` row `a` of 17,408 bits (three segments, of 8,192, 8,192 and 1,024
  bits), a `sha512/row/v2` row `b` of 1,024 bits, and the `sha512/row/v2` output row `y`. `witnessMix_scope`: `scopeOkH`
  admits it and `scopeOk` refuses it.
* `zk_session_soundHJ_nonvacuous_seg`, `zk_session_soundHJ_nonvacuous_mix`: `zk_session_soundHJ_nonvacuous_v2` at
  them: at a session `verify --zk` accepts whose circuit file parses as the witness, the premises the hidden-output
  headlines read of the circuit hold together (`Built`, `scopeOkH`, `Hidden.Any.RowsFitS`), with
  `zk_session_soundHJ_nonvacuous`'s `y₀` and draw hypotheses.

Not covered, as for `zk_session_soundHJ_nonvacuous_v2`: that a session over the witnesses exists, `hRec`, `hpub`, `hpt`,
`hkd`, `hA3`, the count curve's parameters and `TableCRZ`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed

variable {n : ℕ}

/-- A `sha512/row-seg/v1` row of `bits` bits in `chunks` compressions, segmented as `HmRow.segShape` cuts it. -/
def witnessSegPort (name : String) (bits chunks : ℕ) : Flock.Port :=
  { name := name, words := bits / 16, chunks := chunks, nb := 0, bits := bits, v2 := true,
    segs := some (HmRow.segShape bits).1 }

/-- `witnessV2` with its input row `w` a `sha512/row-seg/v1` row of 1,024 bits. -/
def witnessSeg : Flock.Circuit :=
  { witnessV2 with name := "witness-seg", ports := #[witnessSegPort "w" 1024 2, witnessV2Port "y"] }

/-- `witnessSeg` with its output row segmented too. -/
def witnessSegOut : Flock.Circuit :=
  { witnessSeg with ports := #[witnessSegPort "w" 1024 2, witnessSegPort "y" 1024 2] }

/-- `witnessSeg` with `w` of 896 bits: its one segment's 112 bytes end past its last block once padded. -/
def witnessSegLate : Flock.Circuit :=
  { witnessSeg with ports := #[witnessSegPort "w" 896 2, witnessV2Port "y"] }

/-- A tile's rows: a `sha512/row-seg/v1` row `a` of three segments, a `sha512/row/v2` row `b`, the output row `y`. -/
def witnessMix : Flock.Circuit :=
  { witnessV2 with
    name := "witness-mix", ports := #[witnessSegPort "a" 17408 21, witnessV2Port "b", witnessV2Port "y"] }

/-- **`scopeOkH` admits a circuit with a `sha512/row-seg/v1` row that `scopeOk` refuses**, and refuses it with its output
row segmented or with a segment that ends past its last block. -/
theorem witnessSeg_scope :
    witnessSeg.ports.any (·.segs.isSome) = true ∧ Layout.scopeOkH witnessSeg = true ∧
      Layout.scopeOk witnessSeg = false ∧ Layout.scopeOkH witnessSegOut = false ∧
      Layout.scopeOkH witnessSegLate = false := by
  decide +kernel

/-- **`scopeOkH` admits a circuit mixing a `sha512/row-seg/v1` row of three segments with `sha512/row/v2` rows**, which
`scopeOk` refuses. -/
theorem witnessMix_scope :
    witnessMix.ports.any (·.segs.isSome) = true ∧ witnessMix.ports.any (fun p => p.v2 && p.segs.isNone) = true ∧
      (witnessMix.ports[0]!.segs.map (·.size)) = some 3 ∧ Layout.scopeOkH witnessMix = true ∧
      Layout.scopeOk witnessMix = false := by
  decide +kernel

/-- **The hidden-output headlines' circuit premises and draw hypotheses hold together at `witnessSeg`**:
`zk_session_soundHJ_nonvacuous_v2` at a circuit file that parses as `witnessSeg`. -/
theorem zk_session_soundHJ_nonvacuous_seg (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok witnessSeg) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Built witnessSeg.unit ∧ Layout.scopeOkH witnessSeg = true ∧ Layout.scopeOk witnessSeg = false ∧
      Hidden.Any.RowsFitS witnessSeg ∧
      Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf witnessSeg.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I), Nonempty
        (DrawSetupZHJ (inputsJ I j) (mPtsOf witnessSeg.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf witnessSeg.g (kd / nTab I)) S' := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have P := parse_facts (parse_untyped ht' hc).1
  obtain ⟨ls, hls⟩ := P.check
  have hU : Built witnessSeg.unit := unit_built P (check_facts P.typed hls)
  obtain ⟨-, hH, hS, -, -⟩ := witnessSeg_scope
  obtain ⟨hy, hall, hok⟩ := zk_session_soundHJ_nonvacuous I ht hTags hc hU hH kd hdiv dj hdj h hd
  exact ⟨hU, hH, hS, rowsFit_of_scopeH ht' hc hH, hy, hall, hok⟩

/-- **The hidden-output headlines' circuit premises and draw hypotheses hold together at `witnessMix`**:
`zk_session_soundHJ_nonvacuous_v2` at a circuit file that parses as `witnessMix`. -/
theorem zk_session_soundHJ_nonvacuous_mix (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok witnessMix) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Built witnessMix.unit ∧ Layout.scopeOkH witnessMix = true ∧ Layout.scopeOk witnessMix = false ∧
      Hidden.Any.RowsFitS witnessMix ∧
      Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf witnessMix.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I), Nonempty
        (DrawSetupZHJ (inputsJ I j) (mPtsOf witnessMix.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf witnessMix.g (kd / nTab I)) S' := by
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have P := parse_facts (parse_untyped ht' hc).1
  obtain ⟨ls, hls⟩ := P.check
  have hU : Built witnessMix.unit := unit_built P (check_facts P.typed hls)
  obtain ⟨-, -, -, hH, hS⟩ := witnessMix_scope
  obtain ⟨hy, hall, hok⟩ := zk_session_soundHJ_nonvacuous I ht hTags hc hU hH kd hdiv dj hdj h hd
  exact ⟨hU, hH, hS, rowsFit_of_scopeH ht' hc hH, hy, hall, hok⟩

end FlockSoundness.Discharge.ZkHidden
