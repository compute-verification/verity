import FlockSoundness.Discharge.ScopeW.Copies
import FlockSoundness.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# Output rows of runs of output ports

An output row is one output port or a run of them (`HmOut.outGroup`). The runs are contiguous and cover the output
words, and an output row of a widened shape holds its run's words, exactly at a bit row and zero-padded to whole
SHA-512 blocks at a lone port's row (`outGroup_start_succ`, `outGroup_start_size`, `outRunOf_spec` and
`outRowW_words`, beside `outBitRowW` in `KeyH.lean`, which reads this file):

* `checkRows_le_wordsW`, `checkRows_words_segW`: an output row's words hold its run's, exactly at a segmented row;
* `outRowsS_of`: the output rows `msg_copyW` reads (`OutRowsS`), from `rowShapeW` and `HmOut.checkRows`;
* `out_copyW`: at Δ's context (`RowDelta`), copy `u'`'s output `j`, bit `t` is bit `t` of word `k` of run `r`'s row
  message (`msg_copyW`, `outCol_leaf`), at every widened shape, a segmented output row among them;
* `out_copyHW`: the same at an accepted hidden-output statement's site (`HiddenExec.out_copyH` at runs).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Model
open FlockSoundness.Discharge.Hm (blockBit blockBit_of_lt)
open FlockSoundness.Discharge.HiddenExec

/-! ## An output row's words -/

section runs

variable {c : Circuit}

theorem getD_mem_ports {q : ℕ} (hq : q < c.ports.size) : c.ports.getD q default ∈ c.ports.toList := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
  exact Array.getElem_mem_toList _

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

/-- **An output row's words hold its run's**: exactly, or a lone port's padded to whole SHA-512 blocks
(`Hidden.checkRows_out`, at rows of `16 · words` bits). -/
theorem checkRows_le_wordsW (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {j : ℕ} (hj : j < c.outRowPorts.size) :
    (HmOut.outGroup c j).2.2.2 ≤ (c.ports.getD (HmOut.nIn c + j) default).words := by
  obtain ⟨hlt, hrow⟩ := Hidden.checkRows_out hcr hj
  have hb := rowShapeW_bits (hW _ (getD_mem_ports hlt))
  rcases hrow with ⟨-, hw, -, -⟩ | ⟨-, hb'⟩ <;> omega

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
