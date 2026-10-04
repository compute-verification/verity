import FlockSoundness.Discharge.ScopeW.Copies
import FlockSoundness.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# Output rows of runs of output ports

An output row is one output port or a run of them (`HmOut.outGroup`): run `j` starts at output word `at_j` and holds
`w_j` words. The first run starts at word 0 and each starts where the one before it ends (`runStart_zero`,
`runStart_succ`), and the runs cover the output words (`runStart_size`), so output word `o` is word `o - at_r` of the run
`r` that starts at or before it while the next starts past it (`run_holds`). Its row, of any widened shape, holds the
run's words: exactly at a bit row, zero-padded to whole SHA-512 blocks at a lone port's row (`checkRows_wordsW`).

* `outRowsS_of`: the output rows `msg_copyW` reads (`OutRowsS`), from `rowShapeW` and `HmOut.checkRows`;
* `out_copyW`: at Δ's context (`RowDelta`), copy `u'`'s output `j`, bit `t` is bit `t` of word `k` of run `r`'s row
  message (`msg_copyW`, `outCol_leaf`), at every widened shape, a segmented output row among them;
* `out_copyHW`: the same at an accepted hidden-output statement's site (`HiddenExec.out_copyH` at runs).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt)
open FlockSoundness.Discharge.HiddenExec

/-! ## Runs of output ports -/

section runs

variable {c : Circuit}

/-- **The first run starts at output word 0.** -/
theorem runStart_zero (c : Circuit) : (HmOut.outGroup c 0).2.2.1 = 0 := by
  sorry

/-- **Each run starts where the one before it ends.** -/
theorem runStart_succ (c : Circuit) (j : ℕ) :
    (HmOut.outGroup c (j + 1)).2.2.1 = (HmOut.outGroup c j).2.2.1 + (HmOut.outGroup c j).2.2.2 := by
  sorry

/-- **The runs cover the output words**: past the last run is output word `outWords`, when the runs cover the output
ports (`Hidden.parse_rowPorts`). -/
theorem runStart_size (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) :
    (HmOut.outGroup c c.outRowPorts.size).2.2.1 = c.outWords := by
  sorry

/-- **There is a run** below an output word, when the runs cover the output ports. -/
theorem runs_pos (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) {o : ℕ} (ho : o < c.outWords) :
    0 < c.outRowPorts.size := by
  sorry

/-- **Run `r` holds output word `o`** when it starts at or before `o` and the next run, if any, starts past it: `o` is
below its end. -/
theorem run_holds (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) {o r : ℕ} (ho : o < c.outWords)
    (hr : r < c.outRowPorts.size) (hle : (HmOut.outGroup c r).2.2.1 ≤ o)
    (hnext : r + 1 < c.outRowPorts.size → o < (HmOut.outGroup c (r + 1)).2.2.1) :
    o - (HmOut.outGroup c r).2.2.1 < (HmOut.outGroup c r).2.2.2 := by
  sorry

/-! ## An output row's words -/

/-- **An output row holds its run's words** (`HmOut.checkRows`), at rows of the widened shapes (`16 · words` bits):
exactly, at a bit row of the run's bits (`sha512/row/v2` or `sha512/row-seg/v1`), or zero-padded to whole SHA-512 blocks
at a lone port's row. -/
theorem checkRows_wordsW (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {j : ℕ} (hj : j < c.outRowPorts.size) :
    (c.ports.getD (HmOut.nIn c + j) default).words = (HmOut.outGroup c j).2.2.2 ∨
      ((HmOut.outGroup c j).2.1 = 1 ∧
        (c.ports.getD (HmOut.nIn c + j) default).words = ((HmOut.outGroup c j).2.2.2 + 63) / 64 * 64) := by
  sorry

/-- **A segmented output row's words are exactly its run's** (`sha512/row-seg/v1`, as row_a's output row). -/
theorem checkRows_words_segW (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {j : ℕ} (hj : j < c.outRowPorts.size) (hs : (c.ports.getD (HmOut.nIn c + j) default).segs.isSome) :
    (c.ports.getD (HmOut.nIn c + j) default).words = (HmOut.outGroup c j).2.2.2 := by
  sorry

/-- **An output row's words hold its run's** (`checkRows_wordsW`). -/
theorem checkRows_le_wordsW (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {j : ℕ} (hj : j < c.outRowPorts.size) :
    (HmOut.outGroup c j).2.2.2 ≤ (c.ports.getD (HmOut.nIn c + j) default).words := by
  sorry

/-- **The output rows `msg_copyW` reads** (`OutRowsS`): every row of a widened shape, and each output row after the
input rows (`Hidden.checkRows_out`). -/
theorem outRowsS_of (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true) (hcr : HmOut.checkRows c = .ok ()) :
    OutRowsS c := by
  sorry

/-- `outRowsS_of` at the widened scope (`scopeOkW_ports`). -/
theorem outRowsS_of_scope (h : scopeOkW c = true) (hcr : HmOut.checkRows c = .ok ()) : OutRowsS c :=
  outRowsS_of (scopeOkW_ports h) hcr

end runs

/-! ## A returned output's copy, at runs -/

section Out

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  {keep : ℕ → Bool} (hΔ : RowDelta st (Hidden.outPairs st.c) keep) (hrows : OutRowsS st.c)
include L F hΔ hrows

/-- **A returned output's column holds its output row's message bit, at runs**: in a satisfying witness, at VU `g` of
block `o`, the unit slot of copy `u'`'s computed column `Layout.outCol j t` (output `j`, bit `t`) holds what the witness
holds at output row `nIn + r`'s message column for word `k`, bit `t`, where the copy's output word is word `k` of run `r`
(`msg_copyW`, `outCol_leaf`). -/
theorem out_copyW (ho : st.c.outNet = st.c.unitNet) (hcr : HmOut.checkRows st.c = .ok ())
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g : ℕ} (hg : g < st.c.g) {u' j t : ℕ}
    (hu' : u' < st.c.upv) (hj : j < (st.c.leavesOut.getD u' #[]).size) (ht : t < 16) {r k : ℕ}
    (hr : r < st.c.outRowPorts.size) (hk : k < (HmOut.outGroup st.c r).2.2.2)
    (hw : (HmOut.outGroup st.c r).2.2.1 + k = (st.c.leavesOut.getD u' #[])[j]) :
    blockBit (L.model regs) z o (Hidden.deltaMsgCol st.c g (HmOut.nIn st.c + r) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t) := by
  sorry

end Out

/-- **`HiddenExec.out_copyH` at runs of output ports and rows of the widened shapes**: in a satisfying witness of an
accepted statement whose unit returns the outputs, at block `o` and VU `g`, copy `u'`'s computed column `outCol j t`
(its output `j`, bit `t`) holds the witness's bit at output row `nIn + r`'s message column for bit `16 k + t`
(`Typed.msgCol`), where the copy's output word `lo[j]` is word `k` of run `r` (`out_copyW` at `rowDelta_hidden`). -/
theorem out_copyHW {T : Model.Statement} (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hW : ∀ p ∈ a.st.c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hu' : u' < a.st.c.upv) (hj : j < (a.st.c.leavesOut.getD u' #[]).size) (ht : t < 16)
    (h1 : (unitRows a.st a.L.unit_built).nIn ≤ Layout.outCol a.st.c j t)
    (h2 : Layout.outCol a.st.c j t < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = a.st.c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (Layout.outCol a.st.c j t -
        (unitRows a.st a.L.unit_built).nIn))))
    {r k : ℕ} (hr : r < a.st.c.outRowPorts.size) (hk : k < (HmOut.outGroup a.st.c r).2.2.2)
    (hw : (HmOut.outGroup a.st.c r).2.2.1 + k = (a.st.c.leavesOut.getD u' #[])[j]) :
    T.block z o (a.col hg x) =
      z (siteAddrH a o (Typed.msgCol a.st.c g (HmOut.nIn a.st.c + r) ((16 * k + t) / 8) ((16 * k + t) % 8))) := by
  sorry

end FlockSoundness.Discharge.ScopeW
