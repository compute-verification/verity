import FlockSoundness.Discharge.ScopeW.OutRuns
import FlockSoundness.Discharge.HiddenExec.KeyH
import FlockSoundness.Discharge.ScopeW.RowEnc
import FlockSoundness.Discharge.ScopeW.Rows

set_option Elab.async false

/-!
# `lay` and `rs` at `keyProg`'s places in hidden-output tables, under the widened scope

`HiddenExec.keyReadersH` at rows of the four shapes `rowShapeW` admits. Two things change:

* an output row is a run of output ports (`HmOut.outGroup`), so the gate on an output word `o` is a bit of the output
  row whose run holds `o` (`outRunOf`), at its offset in the run (`outBitRowW`); on a circuit whose output rows are one
  port each it is `HiddenExec.outBitRow`'s bit;
* each row is hashed under its own schema (`encW`: `rowEncOf` of its port), with its extractor (`pairW`), so `rs` is an
  `HmRowsP` (`keyRowsHW`).

The readers' addresses are `keyReadersH`'s: `Typed.msgCol` places a row's byte in its chain's compression
(`HmRow.byteAt`, segment by segment) or, past them, in its `hm96` slot's block, so they hold at every shape.

What is left open, each a named obligation on a `sorry` stub:
* `keyProgHW_copies`: Δ's copies, at input rows (`copy_siteH`, unchanged) and at output rows of runs (`out_copyH` at
  `outBitRowW`);
* `keyProgHW_bcPinned`: `keyProgH_bcPinned` at `rowOfHW` (no row's shape is read);
* `keyProgHW_valPinned`: every computed committed gate is an output row's bit (`outBitRowW`, at runs);
* `keyRowsHW_computes`: `HmRowComputes` at each row's schema: the `b ‖ c` at the row's `hm96` output is the commit
  string of `rowEncOf` of the row's value (`bc_site_commitH` at `sha512/row/v2` and `sha512/row-seg/v1` rows).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Model Audit Binding Placed
open FlockSoundness.Discharge.HiddenExec
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW OutColAt outDecode OutsOut KeysRow BitVal rowEnc
  rowEnc_inj)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

/-! ## Output rows of runs of output ports -/

variable (c) in
/-- **The output row holding output word `o`**: the last output row whose run of output ports starts at or before `o`
(`HmOut.outGroup`'s third component). -/
def outRunOf (o : ℕ) : ℕ := Nat.findGreatest (fun j => (HmOut.outGroup c j).2.2.1 ≤ o) (c.outRowPorts.size - 1)

variable (c) in
/-- **An output row's bit, at runs**: copy `d.1`'s output `d.2.1`, bit `d.2.2`, of unit `u` is bit
`16 (o - at) + d.2.2` of `u`'s output row `nIn + j`, where the copy's output word `o` is in output row `j`'s run
(`outRunOf`), which starts at output word `at`; none past the row. -/
def outBitRowW (u : Fin n) (d : ℕ × ℕ × ℕ) : Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  if hq : HmOut.nIn c + outRunOf c (outWordOf c d) < c.ports.size then
    if hk : 16 * (outWordOf c d - (HmOut.outGroup c (outRunOf c (outWordOf c d))).2.2.1) + d.2.2 <
        portRowW c ⟨_, hq⟩ then
      some (u, ⟨⟨_, hq⟩, ⟨_, hk⟩⟩)
    else none
  else none

variable (c hU n) in
open Classical in
/-- **The output-row bit of a unit's computed gate, at runs**: `HiddenExec.outRowOf` at `outBitRowW`. -/
noncomputable def outRowOfW (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match (keyProg c hU n).unit g with
  | some u =>
    if h : ∃ x, (Const.Prog.instC (keyProg c hU n) outs u).wire x = g then
      outBitRowW c u (outDecode c hU (Classical.choose h).val)
    else none
  | none => none

variable (c hU n) in
/-- **The row bits of `keyProg`, at runs**: an input row's bit (`Layout.keyRowOf`), else an output row's
(`outRowOfW`). -/
noncomputable def rowOfHW (outs : Finset (Fin (keyN c hU n))) (g : Fin (keyN c hU n)) :
    Option (Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :=
  match keyRowOf c hU n (portRowW c) g with
  | some r => some r
  | none => outRowOfW c hU n outs g

theorem outBitRowW_spec {u u' : Fin n} {d : ℕ × ℕ × ℕ} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : outBitRowW c u d = some (u', ⟨q, k⟩)) :
    u' = u ∧ q.val = HmOut.nIn c + outRunOf c (outWordOf c d) ∧
      k.val = 16 * (outWordOf c d - (HmOut.outGroup c (outRunOf c (outWordOf c d))).2.2.1) + d.2.2 := by
  unfold outBitRowW at h
  split at h
  · split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨hu, hs⟩ := h
      subst hu
      cases hs
      exact ⟨rfl, rfl, rfl⟩
    · exact absurd h (by simp)
  · exact absurd h (by simp)

/-- **Output word `o` is word `o - at` of run `outRunOf c o`**, which starts at output word `at`, for `o < outWords`,
when the runs cover the output ports (`run_holds`). -/
theorem outRunOf_spec (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size) {o : ℕ} (ho : o < c.outWords) :
    outRunOf c o < c.outRowPorts.size ∧ (HmOut.outGroup c (outRunOf c o)).2.2.1 ≤ o ∧
      o - (HmOut.outGroup c (outRunOf c o)).2.2.1 < (HmOut.outGroup c (outRunOf c o)).2.2.2 := by
  have hpos := runs_pos hsum ho
  have hP : (HmOut.outGroup c (outRunOf c o)).2.2.1 ≤ o :=
    Nat.findGreatest_spec (P := fun j => (HmOut.outGroup c j).2.2.1 ≤ o) (Nat.zero_le _)
      (by simp only [runStart_zero, Nat.zero_le])
  have hle : outRunOf c o ≤ c.outRowPorts.size - 1 := Nat.findGreatest_le _
  have hlt : outRunOf c o < c.outRowPorts.size := by omega
  refine ⟨hlt, hP, run_holds hsum ho hlt hP fun h1 => ?_⟩
  have hng := Nat.findGreatest_is_greatest (P := fun j => (HmOut.outGroup c j).2.2.1 ≤ o) (k := outRunOf c o + 1)
    (Nat.lt_succ_self _) (by omega)
  simpa only [not_le] using hng

/-- **An output word's bit is an output row's, at runs**: at an output word of the circuit, in a circuit whose runs
cover its output ports and whose output rows, of the widened shapes, hold their runs' words (`HmOut.checkRows`), each of
its 16 bits is a bit of the row of the run holding it. -/
theorem outBitRowW_ne_none (hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size)
    (hcr : HmOut.checkRows c = .ok ()) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true) (u : Fin n)
    {d : ℕ × ℕ × ℕ} (hw : outWordOf c d < c.outWords) (ht : d.2.2 < 16) : outBitRowW c u d ≠ none := by
  obtain ⟨hr, -, hk⟩ := outRunOf_spec hsum hw
  have hle := checkRows_le_wordsW hcr hW hr
  have hq := (Hidden.checkRows_out hcr hr).1
  have hpw : portRowW c ⟨_, hq⟩ =
      16 * (c.ports.getD (HmOut.nIn c + outRunOf c (outWordOf c d)) default).words := by
    unfold portRowW
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
    rfl
  unfold outBitRowW
  rw [dite_eq_left hq, dite_eq_left (by rw [hpw]; omega)]
  exact Option.some_ne_none _

theorem outRowOfW_wire (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    (hgu : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u)
    (hx : ((keyProg c hU n).rows u).nIn ≤ x.val) :
    outRowOfW c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) =
      outBitRowW c u (outDecode c hU x.val) := by
  have hc : ∃ x', (Const.Prog.instC (keyProg c hU n) outs u).wire x' =
      (Const.Prog.instC (keyProg c hU n) outs u).wire x := ⟨x, rfl⟩
  have hch : Classical.choose hc = x := by
    rcases (Const.Prog.instC (keyProg c hU n) outs u).shared _ x (Classical.choose_spec hc) with he | ⟨-, h2⟩
    · exact he
    · omega
  unfold outRowOfW
  rw [hgu]
  dsimp only
  rw [dite_eq_left hc, hch]

theorem outRowOfW_wire_some (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : outRowOfW c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some u ∧
      ((keyProg c hU n).rows u).nIn ≤ x.val ∧ outBitRowW c u (outDecode c hU x.val) = some (u', ⟨q, k⟩) := by
  rcases hgu : (keyProg c hU n).unit ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | u''
  · unfold outRowOfW at h
    rw [hgu] at h
    exact absurd h (by simp)
  · obtain ⟨rfl, hx⟩ := unit_wire outs x hgu
    exact ⟨rfl, hx, (outRowOfW_wire outs x hgu hx).symm.trans h⟩

theorem rowOfHW_of_key (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    {r : Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)} (h : keyRowOf c hU n (portRowW c) g = some r) :
    rowOfHW c hU n outs g = some r := by
  unfold rowOfHW
  rw [h]

theorem rowOfHW_of_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    (h : keyRowOf c hU n (portRowW c) g = none) : rowOfHW c hU n outs g = outRowOfW c hU n outs g := by
  unfold rowOfHW
  rw [h]

theorem rowOfHW_none (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    (h : rowOfHW c hU n outs g = none) :
    keyRowOf c hU n (portRowW c) g = none ∧ outRowOfW c hU n outs g = none := by
  rcases hk : keyRowOf c hU n (portRowW c) g with _ | r
  · exact ⟨rfl, (rowOfHW_of_none outs hk).symm.trans h⟩
  · rw [rowOfHW_of_key outs hk] at h
    cases h

/-- **A row bit of a unit's own instance is that unit's row**, at runs. -/
theorem rowOfHW_wire_unit (outs : Finset (Fin (keyN c hU n))) {u u' : Fin n} (x : Fin ((keyProg c hU n).rows u).w)
    {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (h : rowOfHW c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩)) :
    u' = u := by
  rcases hk : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | r
  · rw [rowOfHW_of_none outs hk] at h
    exact (outBitRowW_spec (outRowOfW_wire_some outs x h).2.2).1
  · rw [rowOfHW_of_key outs hk] at h
    rw [h] at hk
    exact (Layout.keyRowOf_wire outs x hk).1

/-! ## Output-row copies at runs -/

section Out

open FlockSoundness.Discharge.Hm (blockBit)

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  {keep : ℕ → Bool} (hΔ : RowDelta st (Hidden.outPairs st.c) keep) (hrows : OutRowsS st.c)
include L F hΔ hrows

/-- **A returned output's column holds its output-row bit's message bit, at runs** (`out_copyW` at `outBitRowW`): in a
satisfying witness, at VU `g` of block `o`, the unit slot of copy `u'`'s computed column `Layout.outCol j t` holds what
the witness holds at the message column of the output-row bit `outBitRowW` gives copy `u'`'s output `j`, bit `t`. -/
theorem out_copy_bitW (ho : st.c.outNet = st.c.unitNet) (hcr : HmOut.checkRows st.c = .ok ())
    (hsum : st.c.outRowPorts.foldl (· + ·) 0 = st.c.outPorts.size)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g : ℕ} (hg : g < st.c.g) {u' j t : ℕ}
    (hu' : u' < st.c.upv) (hj : j < (st.c.leavesOut.getD u' #[]).size) (ht : t < 16)
    {u v : Fin n} {q : Fin st.c.ports.size} {k : Fin (portRowW st.c q)}
    (hb : outBitRowW st.c u (u', j, t) = some (v, ⟨q, k⟩)) :
    blockBit (L.model regs) z o (Hidden.deltaMsgCol st.c g q.val (k.val / 8) (k.val % 8)) =
      blockBit (L.model regs) z o (st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t) := by
  obtain ⟨-, hq, hk⟩ := outBitRowW_spec hb
  have P : ParseFacts tags st.c := F.parse
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : st.c.leavesOut.getD u' #[] = st.c.leavesOut[u']! := by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hwj : outWordOf st.c (u', j, t) = (st.c.leavesOut.getD u' #[])[j] := getElem!_pos _ j hj
  have hwlt : outWordOf st.c (u', j, t) < st.c.outWords := by
    rw [hwj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  obtain ⟨hr, hle, hkk⟩ := outRunOf_spec hsum hwlt
  have hw : (HmOut.outGroup st.c (outRunOf st.c (outWordOf st.c (u', j, t)))).2.2.1 +
      (outWordOf st.c (u', j, t) - (HmOut.outGroup st.c (outRunOf st.c (outWordOf st.c (u', j, t)))).2.2.1) =
      (st.c.leavesOut.getD u' #[])[j] := by
    rw [← hwj]
    omega
  have := out_copyW L F hΔ hrows ho hcr hz o hg hu' hj ht hr hkk hw
  dsimp only at hk
  rw [hq, hk, show ∀ a, (16 * a + t) / 8 = 2 * a + t / 8 from fun a => by omega,
    show ∀ a, (16 * a + t) % 8 = t % 8 from fun a => by omega]
  exact this

end Out

/-- **`HiddenExec.out_copy_key` at runs and the widened shapes**: in a satisfying witness of an accepted statement on
`c` whose unit returns the outputs, at block `o` and VU `g`, the VU's column on copy `u'`'s output `j`, bit `t`
(`Layout.OutColAt`) holds the witness's bit at the message column of its output-row bit (`outBitRowW`, `out_copyHW`). -/
theorem out_copy_keyW {T : Model.Statement} (a : AcceptedH T) (hce : a.st.c = c) (ho : c.outNet = c.unitNet)
    (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hoc : OutColAt c hU x.val u' j t) {u v : Fin n} {q : Fin c.ports.size} {k : Fin (portRowW c q)}
    (hb : outBitRowW c u (u', j, t) = some (v, ⟨q, k⟩)) :
    T.block z o (a.col hg x) = z (siteAddrH a o (Typed.msgCol a.st.c g q.val (k.val / 8) (k.val % 8))) := by
  subst hce
  obtain ⟨-, hq, hk⟩ := outBitRowW_spec hb
  obtain ⟨hu', hj, ht, h1, h2, hx⟩ := hoc
  have P := parse_facts a.parse
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : a.st.c.leavesOut.getD u' #[] = a.st.c.leavesOut[u']! := by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hwj : outWordOf a.st.c (u', j, t) = (a.st.c.leavesOut.getD u' #[])[j] := getElem!_pos _ j hj
  have hwlt : outWordOf a.st.c (u', j, t) < a.st.c.outWords := by
    rw [hwj]
    refine (hlo u' hu').2 _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  obtain ⟨hr, hle, hkk⟩ := outRunOf_spec (Hidden.parse_rowPorts a.parse).2 hwlt
  have hw : (HmOut.outGroup a.st.c (outRunOf a.st.c (outWordOf a.st.c (u', j, t)))).2.2.1 +
      (outWordOf a.st.c (u', j, t) -
        (HmOut.outGroup a.st.c (outRunOf a.st.c (outWordOf a.st.c (u', j, t)))).2.2.1) =
      (a.st.c.leavesOut.getD u' #[])[j] := by
    rw [← hwj]
    omega
  have := out_copyHW a ho hW hz o hg x hu' hj ht h1 h2 hx hr hkk hw
  rw [hq, hk]
  exact this

/-! ## The readers -/

section places

variable {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (keyN c hU n))) (popOf : Reg → Public)

/-- **The readers at `keyProg`'s sites in hidden-output tables, at runs**: `HiddenExec.keyReadersH` at `rowOfHW`. -/
noncomputable def keyReadersHW : Layout.Readers ((keyProg c hU n).circuit outs) plan tab :=
  Layout.Readers.ofRows (keyPlaceH hparse outs) (Layout.keyN_pos c hU n) (portRowW c)
    (W := Finset.univ.sup (portRowW c) + 1)
    (fun q => (Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _)) (Nat.le_add_left 1 _) (rowOfHW c hU n outs)
    (keyRowAddrH si) (keySaltAddrH si (hU := hU)) (keyBcAddrH si)
    (fun R u q => ((popOf R).digest u.val q.val c.ports.size).data.toList)
    (Layout.keyGate (c := c) (hU := hU) (n := n) fun _ _ => false)

variable (ho : c.outNet = c.unitNet) (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
  (houts : OutsOut c hU n outs)

include ho hW houts in
/-- **Δ's copies at `keyProg`'s sites, at the widened scope**: a column on an input row's bit holds the witness's bit at
its key's message column (`copy_siteH`); a column on an output row's bit (`outBitRowW`) holds the bit at the output
row's message column (`out_copyH` at runs of output ports and at the four shapes). -/
theorem keyProgHW_copies : (keyReadersHW hparse outs popOf).Copies (keyPlaceH hparse outs) := by
  refine Layout.ofRows_copies _ fun S R u hu M hM x hx u' q k hr => ?_
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hrw := SitesH.rowsOf_keyProg hU hparse S R u hu
  rcases hk : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) with _ | r
  · have ho' : outRowOfW c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = some (u', ⟨q, k⟩) :=
      (rowOfHW_of_none outs hk).symm.trans hr
    obtain ⟨hgu, -, hb⟩ := outRowOfW_wire_some outs x ho'
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hx hgu x rfl
    rw [Layout.outDecode_of hoc] at hb
    have := out_copy_keyW (si.acc S R u hu) hc ho hW hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) hoc hb
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this
  · have hr' : keyRowOf c hU n (portRowW c) ((Const.Prog.instC (keyProg c hU n) outs u).wire x) =
        some (u', ⟨q, k⟩) := hk.trans ((rowOfHW_of_key outs hk).symm.trans hr)
    obtain ⟨-, hxn, hkey⟩ := Layout.keyRowOf_wire outs x hr'
    have hk' : Aliased.srcKey (si.acc S R u hu).st.c x.val = .msg q.val (k.val / 8) (k.val % 8) := by
      rw [hc]; exact hkey
    have hin' : Typed.msgCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val (k.val / 8) (k.val % 8) <
        2 ^ (si.acc S R u hu).st.c.kLog := by
      have hv := si.hvu S R u hu
      rw [hc] at hv ⊢
      exact Layout.msgCol_lt_of_parse (hparse S R u hu) hv q.isLt _ _ (Nat.mod_lt _ (by norm_num))
    have := copy_siteH (si.acc S R u hu) hM (si.o S R u hu) (si.hvu S R u hu)
      (Fin.cast (congrArg FlockSoundness.Rows.w hrw) x) (lt_of_lt_of_eq hxn (congrArg FlockSoundness.Rows.nIn hrw))
      hk' hin'
    simp only [keyRowAddrH, hu, ↓reduceDIte]
    exact this

/-- **The rows' `b ‖ c` at `keyProg`'s sites is the registration's**, as `keyProgH_bcPinned` (`bc_site_ofH`). -/
theorem keyProgHW_bcPinned
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val) :
    (keyReadersHW hparse outs popOf).BcPinned (P := (keyProg c hU n).partition outs) := by
  refine Layout.ofRows_bcPinned _ fun S R u hu M hM g hg u' q k hr => ?_
  obtain ⟨x, rfl⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  rw [rowOfHW_wire_unit outs x hr]
  have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hnp : (si.acc S R u hu).st.c.ports.size = c.ports.size := by rw [hc]
  have := bc_site_ofH (hsite S R u hu M hM) hM (si.hvu S R u hu) (p := q.val) (by rw [hnp]; exact q.isLt)
  rw [hnp] at this
  unfold keyBcAddrH
  simp only [hu, ↓reduceDIte]
  exact this

include ho hW houts in
/-- **The single gates at `keyProg`'s sites are the constant and the zero**, as `keyProgH_valPinned`: a computed
committed gate is a returned output's, so a bit of the output row whose run holds its word (`outBitRowW`). -/
theorem keyProgHW_valPinned (hkeys : KeysRow c hU (portRowW c)) :
    (keyReadersHW hparse outs popOf).ValPinned (P := (keyProg c hU n).partition outs) := by
  refine Layout.ofRows_valPinned _ fun S R u hu M hM g hg hr => ?_
  have hpub := Layout.gate_pub (keyPlaceH hparse outs) (constCols_keyProgH hU hparse outs)
    (zeroCols_keyProgH hU hparse outs) hu hM hg
  obtain ⟨x, hx⟩ := Layout.exists_col_of_io ((keyPlaceH hparse outs) S R u hu).inst hg
  change bit _ = Layout.keyGate (c := c) (hU := hU) (n := n) (fun _ _ => false) R g
  obtain ⟨hk, hno⟩ := rowOfHW_none outs hr
  rcases Layout.single_gate outs hkeys ⟨x, hx⟩ hg hk with rfl | rfl | ⟨hm, -, -⟩
  · rw [hpub.2 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, Layout.zeroG_ne_oneG, ↓reduceIte]
  · rw [hpub.1 (Finset.mem_singleton_self _)]
    simp only [Layout.keyGate, ↓reduceIte]
  · exfalso
    subst hx
    obtain ⟨u'', j, t, hoc⟩ := houts u _ hg hm x rfl
    have hno' : outRowOfW c hU n outs ((Const.Prog.instC (keyProg c hU n) outs u).wire x) = none := hno
    rw [outRowOfW_wire outs x hm (unit_wire outs x hm).2, Layout.outDecode_of hoc] at hno'
    have hc := (si.acc S R u hu).c_eq (hparse S R u hu)
    have hcr : HmOut.checkRows c = .ok () := hc ▸ (si.acc S R u hu).checkRows
    have P := parse_facts (hparse S R u hu)
    obtain ⟨ls, hls⟩ := P.check
    obtain ⟨-, -, hlo⟩ := Layout.check_out P.typed ho hls
    obtain ⟨hu', hj, ht, -⟩ := hoc
    have hget : c.leavesOut.getD u'' #[] = c.leavesOut[u'']! := by
      rw [Hidden.getElem!_eq_getD']; rfl
    have hwlt : outWordOf c (u'', j, t) < c.outWords := by
      rw [show outWordOf c (u'', j, t) = (c.leavesOut.getD u'' #[])[j] from getElem!_pos _ j hj]
      refine (hlo u'' hu').2 _ ?_
      rw [← hget]
      exact Array.getElem_mem_toList _
    exact outBitRowW_ne_none (Hidden.parse_rowPorts (hparse S R u hu)).2 hcr hW u hwlt ht hno'

/-- **`lay` at `keyProg`'s places in hidden-output tables, at the widened scope**: `keyReadersHW`'s layout, with
`bitCode` for the single gates and Δ's copies from `keyProgHW_copies`. -/
noncomputable def keyLayoutHW :
    Binding.Layout ((keyProg c hU n).partition outs) plan tab
      (((keyProg c hU n).partition outs).placeDecoder plan tab (keyPlaceH hparse outs)) :=
  (keyReadersHW hparse outs popOf).layout (keyPlaceH hparse outs) Layout.bitCode
    (keyProgHW_copies hparse outs popOf ho hW houts)

/-- **`rs` at `keyProg`'s places in hidden-output tables, at the widened scope**: `keyReadersHW`'s rows, each under its
port's schema (`encW`, `pairW`), from `keyProgHW_copies`, `keyProgHW_bcPinned` and `keyProgHW_valPinned`. -/
noncomputable def keyRowsHW
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    HmRowsP (keyLayoutHW hparse outs popOf ho hW houts) :=
  rowsP (keyReadersHW hparse outs popOf) (keyPlaceH hparse outs) (encW c n (Fin (keyN c hU n)))
    (pairW c n (Fin (keyN c hU n))) pairW_spec Layout.bitCode Layout.bitCode_inj
    (keyProgHW_copies hparse outs popOf ho hW houts)
    (keyProgHW_bcPinned hparse outs popOf hsite) (keyProgHW_valPinned hparse outs popOf ho hW houts hkeys)

/-- **`HmRowComputes` at `keyProg`'s rows, at the widened scope** (`hHm` for `keyRowsHW`): at every drawn unit's
committed wire held by a row, in a satisfying witness of the unit's table, the `b ‖ c` at the row's `hm96` output is the
commit string of `rowEncOf` of the row's value under its port's schema and the salt at `Hm.saltCol`. -/
theorem keyRowsHW_computes
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSiteH (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hkeys : KeysRow c hU (portRowW c)) :
    (keyRowsHW hparse outs popOf ho hW houts hsite hkeys).Computes := sorry

end places

end FlockSoundness.Discharge.ScopeW
