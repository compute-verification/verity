import FlockSoundness.Discharge.ScopeW.Copies
import FlockSoundness.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# Output rows of runs of output ports

An output row is one output port or a run of them (`HmOut.outGroup`): run `j` starts at output word `at_j` and holds
`w_j` words. The first run starts at word 0 and each starts where the one before it ends (`runStart_zero`,
`runStart_succ`), and the runs cover the output words (`runStart_size`), so output word `o` is word `o - at_r` of the
run `r` that starts at or before it while the next starts past it (`run_holds`). Its row, of any widened shape, holds
the run's words: exactly at a bit row, zero-padded to whole SHA-512 blocks at a lone port's row (`checkRows_wordsW`).

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

theorem foldl_add_extract_succ (a : Array ℕ) (j : ℕ) :
    (a.extract 0 (j + 1)).foldl (· + ·) 0 = (a.extract 0 j).foldl (· + ·) 0 + a.getD j 0 := by
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_extract, Array.toList_extract,
    List.extract_eq_take_drop, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero, Nat.sub_zero, List.take_add_one,
    List.foldl_append, Array.getD_eq_getD_getElem?, ← Array.getElem?_toList]
  cases a.toList[j]? <;> simp

theorem foldl_snd_extract_add (a : Array (String × ℕ)) (i k : ℕ) :
    (a.extract 0 (i + k)).foldl (· + ·.2) 0 =
      (a.extract 0 i).foldl (· + ·.2) 0 + (a.extract i (i + k)).foldl (· + ·.2) 0 := by
  simp only [← Array.foldl_toList, Array.toList_extract, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero,
    Nat.add_sub_cancel_left, Hidden.foldl_snd_eq, Nat.zero_add, List.take_add, List.map_append, List.sum_append]

/-- **The first run starts at output word 0.** -/
theorem runStart_zero (c : Circuit) : (HmOut.outGroup c 0).2.2.1 = 0 := by
  simp [HmOut.outGroup]

/-- **Each run starts where the one before it ends.** -/
theorem runStart_succ (c : Circuit) (j : ℕ) :
    (HmOut.outGroup c (j + 1)).2.2.1 = (HmOut.outGroup c j).2.2.1 + (HmOut.outGroup c j).2.2.2 := by
  simp only [HmOut.outGroup]
  rw [foldl_add_extract_succ, foldl_snd_extract_add]

/-- **The runs cover the output words**: past the last run is output word `outWords`, when the runs cover the output
ports (`Hidden.parse_rowPorts`). -/
theorem runStart_size (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) :
    (HmOut.outGroup c c.outRowPorts.size).2.2.1 = c.outWords := by
  simp only [HmOut.outGroup, Array.extract_size, hsum, Circuit.outWords]

/-- **There is a run** below an output word, when the runs cover the output ports. -/
theorem runs_pos (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) {o : ℕ} (ho : o < c.outWords) :
    0 < c.outRowPorts.size := by
  by_contra h0
  have h := runStart_size hsum
  rw [show c.outRowPorts.size = 0 by omega, runStart_zero] at h
  omega

/-- **Run `r` holds output word `o`** when it starts at or before `o` and the next run, if any, starts past it: `o` is
below its end. -/
theorem run_holds (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) {o r : ℕ} (ho : o < c.outWords)
    (hr : r < c.outRowPorts.size) (hle : (HmOut.outGroup c r).2.2.1 ≤ o)
    (hnext : r + 1 < c.outRowPorts.size → o < (HmOut.outGroup c (r + 1)).2.2.1) :
    o - (HmOut.outGroup c r).2.2.1 < (HmOut.outGroup c r).2.2.2 := by
  rcases Nat.lt_or_ge (r + 1) c.outRowPorts.size with h | h
  · have := hnext h
    rw [runStart_succ] at this
    omega
  · have := runStart_size hsum
    rw [show c.outRowPorts.size = r + 1 by omega, runStart_succ] at this
    omega

/-! ## An output row's words -/

theorem rowShapeW_bits {p : Port} (h : rowShapeW p = true) : p.bits = 16 * p.words := by
  unfold rowShapeW at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.1

theorem getD_mem_ports {q : ℕ} (hq : q < c.ports.size) : c.ports.getD q default ∈ c.ports.toList := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
  exact Array.getElem_mem_toList _

/-- **An output row holds its run's words** (`HmOut.checkRows`), at rows of the widened shapes (`16 · words` bits):
exactly, at a bit row of the run's bits (`sha512/row/v2` or `sha512/row-seg/v1`), or zero-padded to whole SHA-512 blocks
at a lone port's row. -/
theorem checkRows_wordsW (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {j : ℕ} (hj : j < c.outRowPorts.size) :
    (c.ports.getD (HmOut.nIn c + j) default).words = (HmOut.outGroup c j).2.2.2 ∨
      ((HmOut.outGroup c j).2.1 = 1 ∧
        (c.ports.getD (HmOut.nIn c + j) default).words = ((HmOut.outGroup c j).2.2.2 + 63) / 64 * 64) := by
  obtain ⟨hlt, hrow⟩ := Hidden.checkRows_out hcr hj
  have hb := rowShapeW_bits (hW _ (getD_mem_ports hlt))
  rcases hrow with ⟨h1, hw, -, -⟩ | ⟨-, hb'⟩
  · exact Or.inr ⟨h1, hw⟩
  · left
    omega

/-- **A segmented output row's words are exactly its run's** (`sha512/row-seg/v1`, as row_a's output row). -/
theorem checkRows_words_segW (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {j : ℕ} (hj : j < c.outRowPorts.size) (hs : (c.ports.getD (HmOut.nIn c + j) default).segs.isSome) :
    (c.ports.getD (HmOut.nIn c + j) default).words = (HmOut.outGroup c j).2.2.2 := by
  obtain ⟨hlt, hrow⟩ := Hidden.checkRows_out hcr hj
  have hb := rowShapeW_bits (hW _ (getD_mem_ports hlt))
  rcases hrow with ⟨-, -, -, hs'⟩ | ⟨-, hb'⟩
  · rw [hs'] at hs
    cases hs
  · omega

/-- **An output row's words hold its run's** (`checkRows_wordsW`). -/
theorem checkRows_le_wordsW (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {j : ℕ} (hj : j < c.outRowPorts.size) :
    (HmOut.outGroup c j).2.2.2 ≤ (c.ports.getD (HmOut.nIn c + j) default).words := by
  rcases checkRows_wordsW hcr hW hj with h | ⟨-, h⟩ <;> omega

/-- **The output rows `msg_copyW` reads** (`OutRowsS`): every row of a widened shape, and each output row after the
input rows (`Hidden.checkRows_out`). -/
theorem outRowsS_of (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true) (hcr : HmOut.checkRows c = .ok ()) :
    OutRowsS c :=
  ⟨hW, fun _ hj => (Hidden.checkRows_out hcr hj).1⟩

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
holds at output row `nIn + r`'s message column for word `k`, bit `t`, where the copy's output word is word `k` of run
`r` (`msg_copyW`, `outCol_leaf`). -/
theorem out_copyW (ho : st.c.outNet = st.c.unitNet) (hcr : HmOut.checkRows st.c = .ok ())
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g : ℕ} (hg : g < st.c.g) {u' j t : ℕ}
    (hu' : u' < st.c.upv) (hj : j < (st.c.leavesOut.getD u' #[]).size) (ht : t < 16) {r k : ℕ}
    (hr : r < st.c.outRowPorts.size) (hk : k < (HmOut.outGroup st.c r).2.2.2)
    (hw : (HmOut.outGroup st.c r).2.2.1 + k = (st.c.leavesOut.getD u' #[])[j]) :
    blockBit (L.model regs) z o (Hidden.deltaMsgCol st.c g (HmOut.nIn st.c + r) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t) := by
  have P : ParseFacts tags st.c := F.parse
  have C : CheckFacts st.c := F.check
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨h16, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hleaf := outCol_leaf P.typed ho (check_returned P.typed ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu' hj
  have hkw : k < (st.c.ports.getD (HmOut.nIn st.c + r) default).words :=
    lt_of_lt_of_le hk (checkRows_le_wordsW hcr hrows.1 hr)
  have hsrc : Hidden.outSrc st.c g r k t = st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t := by
    unfold Hidden.outSrc
    simp only [hk, ↓reduceIte]
    rw [hw, hleaf]
    unfold Layout.outCol
    omega
  rw [msg_copyW L F hΔ hrows hz o hg hr hkw ht, hsrc]

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
  have hcol := col_val_compH a hg x hu' h1 h2 hx
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hW hg x hu' hj h1 h2 hx hr hk hw hcol ⊢
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht' s.parse
  have F0 := flat_unitFacts ht' h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  have hrows := outRowsS_of hW hcr
  have hmc := out_copyW L F0 (rowDelta_hidden L F0 hrows hda) hrows ho hcr hz o hg hu' hj ht hr hk hw
  have hb : (16 * k + t) / 8 = 2 * k + t / 8 := by omega
  have hb8 : (16 * k + t) % 8 = t % 8 := by omega
  have hin := Layout.msgCol_lt_of_parse hpa hg (hrows.2 r hr) (2 * k + t / 8) (t % 8) (Nat.mod_lt _ (by norm_num))
  refine bit_inj _ _ ?_
  rw [hb, hb8, bit_siteAddrH ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, rfl⟩
    z o hin, Hidden.msgCol_delta st.c g (HmOut.nIn st.c + r) (2 * k + t / 8) (t % 8), hmc, ← hcol,
    blockBit_of_lt _ z o (Fin.isLt _)]

end FlockSoundness.Discharge.ScopeW
