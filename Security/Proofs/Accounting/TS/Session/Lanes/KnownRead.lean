import Proofs.Accounting.TS.Session.Lanes.KnownFields
import Proofs.Flock.C3.Reps.ReadWhole
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.TmplReads
import Proofs.Flock.VBridge.Unit
import Proofs.Flock.Verifier.Registered
import Flock.KnownRead

/-!
# Every bit of a known port is on a wire of the unit, from flock-verify's check

`flock-verify --registered F`, where `F` names known content, refuses a statement whose circuit names a port twice or
has no port of a name the known content names (`Flock.KnownRead.namesOk`), and one of whose ports `F` names at known
content is not a `sha512/row/v2` row of the known rows' words, or has a row bit that no computed row of the unit reads
(an input row) or no copy returns (an output row) (`Flock.KnownRead.readsOk`; `check` is both, run on every statement
beside the complete check). `readsOk` checks the input bits at columns every one of which the unit's computed rows read
(`FlockSoundness.TmplFold.readsAt_of_readsOk`). What that refusal establishes, from the verifier's code:

* `names_of_check`: the circuit names each port once, and has a port of each known name;

* `srcAt_of_srcAtMsg`, `srcKey_of_srcMsg`: the check's `srcMsg` at copy `u`'s input column `y` is Δ's source key at
  stacked input `u · W + y` (`Aliased.srcKey`) where it names a message bit (its port bits a mask, `portMask_getD`);
* `colRead_of_read`: a column `readIns` marks, at a copy of the unit, whose key is bit `k` of port `q`, is a column a
  row of the circuit's template instance reads (`ColRead`), through the stack's rows (`stack_a`, `stack_b`);
* `read_of_check`, `colRead_of_check`: so every bit `k < 16 · words` of an input port (`HmOut.nIn`) the known content
  names is `ColRead`, and `rowBitRead_of_checkIn` on a wire of every unit (`RowBitRead`, `rowBitRead_of_col`);
* `out_of_check`, `outPortOf_eq`, `rowBitRead_of_out`, `rowBitRead_of_checkOut`: every bit of an output port the known
  content names is returned by a copy, at a computed column, whose gate in each unit's instance the hidden-output
  layout places on that bit (`HiddenExec.rowOfH`) and the unit commits as an output: on a wire of every unit;
* `shape_of_check`, `knownPort_of_check`: a known port is a known value's row (`KnownPort`) of its known rows' words;
* `laneKnownHolds_of_check`, `combKnownOk_of_check`: `laneKnownHolds` and `combKnownOk` from the check alone: the
  circuit names each port once, and `admit`, `pub` and `o0` are its ports, known values' rows of 32, 1616 and 16 bits
  (`knownRows`' words), each bit on a wire of unit 0.
-/

namespace Accounting.TS.Lanes

open PouwWindow.Lanes Flock.SoundAtBridge Accounting.TS.Checked FlockSoundness.Discharge
open FlockSoundness.Discharge.Layout (portRowW)

section

variable {c : _root_.Flock.Circuit}

/-- The check's source at a leaf port is `srcAt`'s where it names a message bit. -/
theorem srcAt_of_srcAtMsg {k p t : ℕ} {r : ℕ × ℕ × ℕ} (h : _root_.Flock.KnownRead.srcAtMsg c k p t = some r) :
    Aliased.srcAt c k p t = .msg r.1 r.2.1 r.2.2 := by
  have hs : _root_.Flock.KnownRead.starts c = FlockSoundness.Typed.starts c := rfl
  unfold _root_.Flock.KnownRead.srcAtMsg at h
  rw [hs] at h
  unfold Aliased.srcAt
  dsimp only at h ⊢
  split_ifs at h ⊢ <;> first | (cases h; done) | (cases h; rfl) | simp_all

/-- **The check's `srcMsg` is Δ's source key**: at copy `u`'s input column `y < W`, where it names message bit
`r`, `srcKey` at stacked input `u · W + y` is that bit. -/
theorem srcKey_of_srcMsg {u y : ℕ} (hy : y < c.unit.inWords * _root_.Flock.WORD) {r : ℕ × ℕ × ℕ}
    (h : _root_.Flock.KnownRead.srcMsg c
      (_root_.Flock.ProvedScope.portMask c.unit.inGroups (c.unit.inWords * _root_.Flock.WORD))
      (_root_.Flock.portCols c.unit.inGroups) u y = some r) :
    Aliased.srcKey c (u * (c.unit.inWords * _root_.Flock.WORD) + y) = .msg r.1 r.2.1 r.2.2 := by
  have hW : 0 < c.unit.inWords * _root_.Flock.WORD := by omega
  have hm : (u * (c.unit.inWords * _root_.Flock.WORD) + y) % (c.unit.inWords * _root_.Flock.WORD) = y := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hy]
  have hd : (u * (c.unit.inWords * _root_.Flock.WORD) + y) / (c.unit.inWords * _root_.Flock.WORD) = u := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hW, Nat.div_eq_of_lt hy, Nat.zero_add]
  unfold _root_.Flock.KnownRead.srcMsg at h
  rw [FailClosed.portMask_getD _ hy] at h
  unfold Aliased.srcKey
  rw [hm, hd]
  split at h
  · rename_i hp
    have hp' := of_decide_eq_true hp
    split
    · exact srcAt_of_srcAtMsg h
    · rename_i hn
      exact absurd hp' hn
  · cases h

/-- **A column the unit reads at one of its copies is read by the circuit's rows**: where a computed row of the unit
reads input column `y` (`readIns`), copy `u` is below `upv` and Δ's key at stacked input `u · W + y` is bit `k` of row
port `q`, that bit is `ColRead`: the copy's own computed row in the stack reads the column. -/
theorem colRead_of_read (hU : FlockSoundness.Built c.unit) (hu : c.unit.unitLog ≤ 32) {q : Fin c.ports.size}
    {k u y : ℕ} (hup : u < c.upv) (hrd : (_root_.Flock.RecResiduals.readIns c.unit).getD y false = true)
    (hkey : Aliased.srcKey c (u * (c.unit.inWords * _root_.Flock.WORD) + y) = .msg q.val (k / 8) (k % 8)) :
    ColRead c hU q k := by
  obtain ⟨hy, r, hr, hr', hc⟩ := (FlockC3.Reps.ReadWhole.getD_readIns c.unit y).1 hrd
  obtain ⟨i, x, hx, hxv⟩ := FlockC3.Reps.ReadWhole.readCol_of_row hU hu hr hr' hc
  have hxn : x.val < (FlockSoundness.Rows.ofNet hU).nIn := by
    rw [FlockSoundness.Rows.ofNet_nIn hU, hxv]
    exact hy
  have hcol : ((FlockSoundness.Rows.ofNet hU).stackCol c.upv hup x).val =
      u * (c.unit.inWords * _root_.Flock.WORD) + y := by
    unfold FlockSoundness.Rows.stackCol
    rw [dite_eq_left hxn]
    show u * (FlockSoundness.Rows.ofNet hU).nIn + x.val = _
    rw [Layout.ofNet_nIn_eq, hxv]
  unfold ColRead Placed.circuitRows
  refine ⟨⟨u * ((FlockSoundness.Rows.ofNet hU).nComp + 1) + (1 + i.val),
    FlockSoundness.Rows.mul_add_lt hup (by omega)⟩, (FlockSoundness.Rows.ofNet hU).stackCol c.upv hup x, ?_, ?_, ?_⟩
  · rw [FlockVBridge.stack_a _ _ hup i, FlockVBridge.stack_b _ _ hup i]
    rcases List.mem_append.1 hx with h | h
    · exact List.mem_append_left _ (List.mem_map.2 ⟨x, h, rfl⟩)
    · exact List.mem_append_right _ (List.mem_map.2 ⟨x, h, rfl⟩)
  · rw [hcol]
    show _ < c.upv * (FlockSoundness.Rows.ofNet hU).nIn
    rw [Layout.ofNet_nIn_eq]
    exact FlockSoundness.Rows.mul_add_lt hup (lt_of_lt_of_le hy (Nat.min_le_left _ _))
  · rw [hcol]
    exact hkey

/-- The check is the names' check, then the reads'. -/
theorem parts_of_check {known : Known} (h : _root_.Flock.KnownRead.check c known = .ok ()) :
    _root_.Flock.KnownRead.namesOk c known = .ok () ∧ _root_.Flock.KnownRead.readsOk c known = .ok () := by
  unfold _root_.Flock.KnownRead.check at h
  cases hn : _root_.Flock.KnownRead.namesOk c known with
  | error e => rw [hn] at h; cases h
  | ok v => rw [hn] at h; exact ⟨rfl, h⟩

/-- **The names the check accepted**: where the known content names a port, the circuit names each port once, and has
a port of each name the known content names. -/
theorem names_of_check {known : Known} (h : _root_.Flock.KnownRead.check c known = .ok ()) {nm : String}
    {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) :
    (∀ q q' : Fin c.ports.size, c.ports[q].name = c.ports[q'].name → q = q') ∧
      ∀ nm' ks', (nm', ks') ∈ known → ∃ q : Fin c.ports.size, c.ports[q].name = nm' := by
  have h1 := (parts_of_check h).1
  unfold _root_.Flock.KnownRead.namesOk at h1
  have hne : ¬ known.isEmpty = true := by
    intro he
    rw [Array.isEmpty_iff] at he
    subst he
    simp at hm
  rw [if_neg hne] at h1
  split at h1
  · cases h1
  · rename_i hd
    have hlt : ∀ q q' : Fin c.ports.size, q.val < q'.val → c.ports[q].name ≠ c.ports[q'].name := by
      intro q q' hl he
      have hb : ¬ ((List.range q'.val).any fun q'' => c.ports[q'']!.name == c.ports[q'.val]!.name) = true :=
        List.find?_eq_none.1 hd q'.val (List.mem_range.2 q'.isLt)
      refine hb (List.any_eq_true.2 ⟨q.val, List.mem_range.2 hl, ?_⟩)
      rw [getElem!_pos c.ports q.val q.isLt, getElem!_pos c.ports q'.val q'.isLt, beq_iff_eq]
      exact he
    refine ⟨fun q q' he => ?_, fun nm' ks' hm' => ?_⟩
    · rcases Nat.lt_trichotomy q.val q'.val with hl | hl | hl
      · exact absurd he (hlt q q' hl)
      · exact Fin.ext hl
      · exact absurd he.symm (hlt q' q hl)
    · have h2 := _root_.Flock.Registered.forM_ok h1 (nm', ks') (Array.mem_toList_iff.2 hm')
      dsimp only at h2
      split at h2
      · rename_i ha
        obtain ⟨i, hi, hn⟩ := Array.any_eq_true.1 ha
        exact ⟨⟨i, hi⟩, beq_iff_eq.1 hn⟩
      · cases h2

/-- **What the check accepted of a known input port**: at each bit `k < 16 · words` of an input port `q` the known
content names, a copy `u` of the unit and an input column `y` a computed row reads, where the check's `srcMsg` is bit `k`
of `q`. -/
theorem read_of_check {known : Known} (h : _root_.Flock.KnownRead.check c known = .ok ()) {nm : String}
    {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : ℕ} (hq : q < _root_.Flock.HmOut.nIn c)
    (hn : c.ports[q]!.name = nm) {k : ℕ} (hk : k < 16 * c.ports[q]!.words) :
    ∃ u y, u < c.upv ∧ (_root_.Flock.RecResiduals.readIns c.unit).getD y false = true ∧
      _root_.Flock.KnownRead.srcMsg c
        (_root_.Flock.ProvedScope.portMask c.unit.inGroups (c.unit.inWords * _root_.Flock.WORD))
        (_root_.Flock.portCols c.unit.inGroups) u y = some (q, k / 8, k % 8) := by
  have hqs : q < c.ports.size := by unfold _root_.Flock.HmOut.nIn at hq; omega
  obtain ⟨rd, h, hrd⟩ := FlockSoundness.TmplFold.readsAt_of_readsOk (parts_of_check h).2
  unfold _root_.Flock.KnownRead.readsOkAt at h
  dsimp only at h
  have h1 := _root_.Flock.Registered.forM_ok h (nm, ks) (Array.mem_toList_iff.2 hm)
  dsimp only at h1
  have h2 := _root_.Flock.Registered.forM_ok h1 q (List.mem_range.2 hqs)
  rw [if_pos hn] at h2
  split at h2
  · cases h2
  · try rw [if_pos hq] at h2
    split at h2
    · cases h2
    · rename_i hf
      have hb := List.find?_eq_none.1 hf k (List.mem_range.2 hk)
      first
        | simp only [Bool.not_eq_true', Bool.not_eq_false] at hb
        | simp at hb
      unfold _root_.Flock.KnownRead.bitRead at hb
      obtain ⟨⟨u, p⟩, -, hb⟩ := List.any_eq_true.1 hb
      dsimp only at hb
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
      exact ⟨u, _, hb.1.1, hrd _ hb.1.2, hb.2⟩

/-- **What the check accepted of a known output port**: at each bit `k < 16 · words` of an output port `q` the known
content names, the bit's word is word `k / 16` of output port `q - nIn` (`outAt`), and some copy `u` of the unit returns
it as its output `j`, the bit at a computed column of the unit's output group. -/
theorem out_of_check {known : Known} (h : _root_.Flock.KnownRead.check c known = .ok ()) {nm : String}
    {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : ℕ} (hqs : q < c.ports.size)
    (hq : ¬ q < _root_.Flock.HmOut.nIn c) (hn : c.ports[q]!.name = nm) {k : ℕ} (hk : k < 16 * c.ports[q]!.words) :
    _root_.Flock.KnownRead.outAt c (q - _root_.Flock.HmOut.nIn c) + k / 16 <
        _root_.Flock.KnownRead.outAt c (q - _root_.Flock.HmOut.nIn c + 1) ∧
      ∃ u j, u < c.upv ∧ j < (c.leavesOut.getD u #[]).size ∧
        (c.leavesOut.getD u #[])[j]! = _root_.Flock.KnownRead.outAt c (q - _root_.Flock.HmOut.nIn c) + k / 16 ∧
        c.unit.inWords * _root_.Flock.WORD ≤
          (c.unit.outGroups.getD 0 default).col * _root_.Flock.WORD + (16 * j + k % 16) ∧
        (c.unit.outGroups.getD 0 default).col * _root_.Flock.WORD + (16 * j + k % 16) < c.unit.constPos := by
  obtain ⟨rd, h, hrd⟩ := FlockSoundness.TmplFold.readsAt_of_readsOk (parts_of_check h).2
  unfold _root_.Flock.KnownRead.readsOkAt at h
  dsimp only at h
  have h1 := _root_.Flock.Registered.forM_ok h (nm, ks) (Array.mem_toList_iff.2 hm)
  dsimp only at h1
  have h2 := _root_.Flock.Registered.forM_ok h1 q (List.mem_range.2 hqs)
  rw [if_pos hn] at h2
  split at h2
  · cases h2
  · try rw [if_neg hq] at h2
    split at h2
    · cases h2
    · rename_i hf
      have hb := List.find?_eq_none.1 hf k (List.mem_range.2 hk)
      first
        | simp only [Bool.not_eq_true', Bool.not_eq_false] at hb
        | simp at hb
      unfold _root_.Flock.KnownRead.bitOut at hb
      dsimp only at hb
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hb
      obtain ⟨hw, ha⟩ := hb
      obtain ⟨⟨u, j⟩, -, hc⟩ := List.any_eq_true.1 ha
      dsimp only at hc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      obtain ⟨⟨⟨⟨hu, hj⟩, hw'⟩, h3⟩, h4⟩ := hc
      exact ⟨hw, u, j, hu, hj, hw', h3, h4⟩

/-- **The shape of a known port's row, from the check**: a port the known content names with rows `ks` is a
`sha512/row/v2` row without segments, of each known row's words. -/
theorem shape_of_check {known : Known} (h : _root_.Flock.KnownRead.check c known = .ok ()) {nm : String}
    {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : ℕ} (hqs : q < c.ports.size)
    (hn : c.ports[q]!.name = nm) :
    c.ports[q]!.v2 = true ∧ c.ports[q]!.segs = none ∧ ∀ i (hi : i < ks.size), ks[i].1.size = c.ports[q]!.words := by
  obtain ⟨rd, h, hrd⟩ := FlockSoundness.TmplFold.readsAt_of_readsOk (parts_of_check h).2
  unfold _root_.Flock.KnownRead.readsOkAt at h
  dsimp only at h
  have h1 := _root_.Flock.Registered.forM_ok h (nm, ks) (Array.mem_toList_iff.2 hm)
  dsimp only at h1
  have h2 := _root_.Flock.Registered.forM_ok h1 q (List.mem_range.2 hqs)
  rw [if_pos hn] at h2
  split at h2
  · cases h2
  · rename_i hs
    have hb : _root_.Flock.KnownRead.shapeOk c.ports[q]! ks = true := by
      cases hb' : _root_.Flock.KnownRead.shapeOk c.ports[q]! ks
      · rw [hb'] at hs
        exact absurd rfl hs
      · rfl
    unfold _root_.Flock.KnownRead.shapeOk at hb
    rw [Bool.and_eq_true, Bool.and_eq_true] at hb
    obtain ⟨⟨hv, hsg⟩, ha⟩ := hb
    exact ⟨hv, Option.isNone_iff_eq_none.1 hsg, fun i hi => by simpa using Array.all_eq_true.1 ha i hi⟩

/-- The check's `outAt` is `Hidden.outAt`. -/
theorem outAt_eq (j : ℕ) : _root_.Flock.KnownRead.outAt c j = Hidden.outAt c j := rfl

theorem outAt_le_succ (j : ℕ) : Hidden.outAt c j ≤ Hidden.outAt c (j + 1) := by
  unfold Hidden.outAt
  rw [List.take_succ, List.map_append, List.sum_append]
  omega

theorem outAt_mono {a b : ℕ} (h : a ≤ b) : Hidden.outAt c a ≤ Hidden.outAt c b := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (outAt_le_succ _)

/-- **Output word `o` is of output port `p`** where it is between `p`'s first word and the next port's
(`HiddenExec.outPortOf`, the greatest port starting at or before `o`). -/
theorem outPortOf_eq {p o : ℕ} (h1 : Hidden.outAt c p ≤ o) (h2 : o < Hidden.outAt c (p + 1)) :
    HiddenExec.outPortOf c o = p := by
  have e : ∀ j, c.outPorts.size ≤ j → Hidden.outAt c j = Hidden.outAt c c.outPorts.size := by
    intro j hj
    unfold Hidden.outAt
    rw [List.take_of_length_le (show c.outPorts.toList.length ≤ j by simpa using hj),
      List.take_of_length_le (show c.outPorts.toList.length ≤ c.outPorts.size by simp)]
  have hp : p ≤ c.outPorts.size := by
    by_contra hlt
    have e1 := e (p + 1) (by omega)
    have e2 := e p (by omega)
    omega
  unfold HiddenExec.outPortOf
  rw [Nat.findGreatest_eq_iff]
  exact ⟨hp, fun _ => h1, fun n hn _ hP => (not_le.2 (lt_of_lt_of_le h2 (outAt_mono (Nat.succ_le_of_lt hn)))) hP⟩

theorem outBitRow_of {n : ℕ} (u : Fin n) {d : ℕ × ℕ × ℕ} {q k : ℕ} (hq : q < c.ports.size)
    (hqv : _root_.Flock.HmOut.nIn c + HiddenExec.outPortOf c (HiddenExec.outWordOf c d) = q)
    (hkv : 16 * (HiddenExec.outWordOf c d - Hidden.outAt c (HiddenExec.outPortOf c (HiddenExec.outWordOf c d))) +
      d.2.2 = k) (hk : k < portRowW c ⟨q, hq⟩) :
    HiddenExec.outBitRow c u d = some (u, ⟨⟨q, hq⟩, ⟨k, hk⟩⟩) := by
  subst hqv hkv
  unfold HiddenExec.outBitRow
  rw [dite_eq_left hq, dite_eq_left hk]

/-- **Every bit of a known input port is read by the circuit's rows**, where flock-verify's check accepts. -/
theorem colRead_of_check (hU : FlockSoundness.Built c.unit) (hu : c.unit.unitLog ≤ 32) {known : Known}
    (h : _root_.Flock.KnownRead.check c known = .ok ()) {nm : String} {ks : Array (Array ℕ × ByteArray)}
    (hm : (nm, ks) ∈ known) {q : Fin c.ports.size} (hq : q.val < _root_.Flock.HmOut.nIn c)
    (hn : c.ports[q].name = nm) {k : ℕ} (hk : k < portRowW c q) : ColRead c hU q k := by
  have e : c.ports[q.val]! = c.ports[q] := getElem!_pos c.ports q.val q.isLt
  obtain ⟨u, y, hup, hrd, hs⟩ := read_of_check h hm hq (by rw [e]; exact hn) (by rw [e]; exact hk)
  have hy : y < c.unit.inWords * _root_.Flock.WORD :=
    lt_of_lt_of_le ((FlockC3.Reps.ReadWhole.getD_readIns c.unit y).1 hrd).1 (Nat.min_le_left _ _)
  exact colRead_of_read hU hu hup hrd (srcKey_of_srcMsg hy hs)

end

/-- A statement in the proved scope (`scopeOkH`) has a unit log of at most 32. -/
theorem unitLog_of_scopeOkH {c : _root_.Flock.Circuit} (h : Layout.scopeOkH c = true) : c.unit.unitLog ≤ 32 := by
  unfold Layout.scopeOkH at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1.1.2

/-- An output a unit computes is one of its committed wires. -/
theorem mem_io_of_output {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
    {u : Fin n} {g : Fin C.N} (hg : g ∈ C.outputs) (hu : P.unit g = some u) : g ∈ P.io u := by
  classical
  simp only [FlockSoundness.Audit.Partition.io, Finset.mem_filter]
  exact ⟨P.mem_committed_of_output hg, Or.inl hu⟩

variable {s : RowsSession}

/-- **A returned output's bit is on a wire of every unit**: where copy `j` of the unit returns its output `j'`, bit `t`,
at a computed column of the unit's rows, and that is bit `k` of output row `q` (`outBitRow`), unit `u`'s instance puts
a gate on the column that the hidden-output layout places on bit `k` of `u`'s row of `q` (`rowOfH_of_none`,
`outRowOf_wire`) and that `u` returns, so commits (`outsOf`): a wire of `u`. -/
theorem rowBitRead_of_out (u : Fin s.n) {j j' t : ℕ} {q : Fin s.c.ports.size} {k : Fin (portRowW s.c q)}
    (hj : j < s.c.upv) (hj' : j' < (s.c.leavesOut.getD j #[]).size) (ht : t < 16)
    (h1 : (FlockSoundness.Rows.ofNet s.hU).nIn ≤ Layout.outCol s.c j' t)
    (h2 : Layout.outCol s.c j' t < (FlockSoundness.Rows.ofNet s.hU).nIn + (FlockSoundness.Rows.ofNet s.hU).nComp)
    (hrow : HiddenExec.outBitRow s.c u (j, j', t) = some (u, ⟨q, k⟩)) : RowBitRead s u q k.val := by
  have eR := Placed.rows_circuitProg s.c s.hU s.n (Layout.keyB s.c s.hU) (Aliased.keyIdx s.c (Layout.keyB s.c s.hU)) u
  have eN : ((Placed.keyProg s.c s.hU s.n).rows u).nIn = s.c.upv * (FlockSoundness.Rows.ofNet s.hU).nIn := by
    rw [congrArg FlockSoundness.Rows.nIn eR]; rfl
  have eC : ((Placed.keyProg s.c s.hU s.n).rows u).nComp =
      s.c.upv * ((FlockSoundness.Rows.ofNet s.hU).nComp + 1) := by
    rw [congrArg FlockSoundness.Rows.nComp eR]; rfl
  have hlt := FlockSoundness.Rows.mul_add_lt hj
    (show 1 + (Layout.outCol s.c j' t - (FlockSoundness.Rows.ofNet s.hU).nIn) <
      (FlockSoundness.Rows.ofNet s.hU).nComp + 1 by omega)
  have hi : j * ((FlockSoundness.Rows.ofNet s.hU).nComp + 1) +
      (1 + (Layout.outCol s.c j' t - (FlockSoundness.Rows.ofNet s.hU).nIn)) <
      ((Placed.keyProg s.c s.hU s.n).rows u).nComp := by
    rw [eC]; exact hlt
  obtain ⟨x, hx⟩ : ∃ x, x = ((Placed.keyProg s.c s.hU s.n).rows u).comp ⟨_, hi⟩ := ⟨_, rfl⟩
  have hxv : x.val = s.c.upv * (FlockSoundness.Rows.ofNet s.hU).nIn +
      (j * ((FlockSoundness.Rows.ofNet s.hU).nComp + 1) +
        (1 + (Layout.outCol s.c j' t - (FlockSoundness.Rows.ofNet s.hU).nIn))) := by
    rw [hx]
    simp only [FlockSoundness.Rows.comp]
    rw [eN]
  have hge : ((Placed.keyProg s.c s.hU s.n).rows u).nIn ≤ x.val := by
    rw [eN, hxv]
    omega
  have hgu : (Placed.keyProg s.c s.hU s.n).unit
      ((Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).wire x) = some u := by
    rw [hx]
    exact (Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).mem ⟨_, hi⟩
  have hoc : Layout.OutColAt s.c s.hU x.val j j' t := ⟨hj, hj', ht, h1, h2, hxv⟩
  have hkey : Layout.keyRowOf s.c s.hU s.n (portRowW s.c)
      ((Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).wire x) = none := by
    rcases hk : Layout.keyRowOf s.c s.hU s.n (portRowW s.c)
        ((Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).wire x) with _ | ⟨u', q', k'⟩
    · rfl
    · have := (Layout.keyRowOf_wire (Layout.outsOf s.c s.hU s.n) x hk).2.1
      omega
  have hrowH : HiddenExec.rowOfH s.c s.hU s.n (Layout.outsOf s.c s.hU s.n)
      ((Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).wire x) = some (u, ⟨q, k⟩) := by
    rw [HiddenExec.rowOfH_of_none _ hkey, HiddenExec.outRowOf_wire _ x hgu hge, Layout.outDecode_of hoc, hrow]
  have hmem : (Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).wire x ∈
      Layout.outsOf s.c s.hU s.n :=
    (@Finset.mem_filter _ _ (Classical.decPred _) _ _).2 ⟨Finset.mem_univ _, u, x, rfl, j, j', t, hoc⟩
  refine ⟨_, k.isLt, hrowH, ?_⟩
  exact mem_io_of_output _ hmem hgu

/-- **A known input port's bits are read at every unit**, where flock-verify's check accepts the session's circuit at
its known content. -/
theorem rowBitRead_of_checkIn (u : Fin s.n) {known : Known} (h : _root_.Flock.KnownRead.check s.c known = .ok ())
    {nm : String} {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : Fin s.c.ports.size}
    (hq : q.val < _root_.Flock.HmOut.nIn s.c) (hn : s.c.ports[q].name = nm) {k : ℕ} (hk : k < portRowW s.c q) :
    RowBitRead s u q k :=
  rowBitRead_of_col u hk (colRead_of_check s.hU (unitLog_of_scopeOkH s.hscope) h hm hq hn hk)

/-- **A known output port's bits are returned at every unit**, where flock-verify's check accepts the session's
circuit at its known content: the copy and output the check found (`out_of_check`) return the bit, of output row `q`
(`outPortOf_eq`, `outBitRow_of`), at a computed column (`ofNet_nIn_eq`, `ofNet_one`), so `rowBitRead_of_out`. -/
theorem rowBitRead_of_checkOut (u : Fin s.n) {known : Known} (h : _root_.Flock.KnownRead.check s.c known = .ok ())
    {nm : String} {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : Fin s.c.ports.size}
    (hq : ¬ q.val < _root_.Flock.HmOut.nIn s.c) (hn : s.c.ports[q].name = nm) {k : ℕ} (hk : k < portRowW s.c q) :
    RowBitRead s u q k := by
  have e : s.c.ports[q.val]! = s.c.ports[q] := getElem!_pos s.c.ports q.val q.isLt
  obtain ⟨hw, j, j', hj, hj', hwj, h1, h2⟩ :=
    out_of_check h hm q.isLt hq (by rw [e]; exact hn) (by rw [e]; exact hk)
  rw [outAt_eq, outAt_eq] at hw
  rw [outAt_eq] at hwj
  have hW : HiddenExec.outWordOf s.c (j, j', k % 16) =
      Hidden.outAt s.c (q.val - _root_.Flock.HmOut.nIn s.c) + k / 16 := hwj
  have hP : HiddenExec.outPortOf s.c (HiddenExec.outWordOf s.c (j, j', k % 16)) =
      q.val - _root_.Flock.HmOut.nIn s.c := by
    rw [hW]
    exact outPortOf_eq (Nat.le_add_right _ _) hw
  have hrow := outBitRow_of (c := s.c) u (d := (j, j', k % 16)) q.isLt (by rw [hP]; omega)
    (by rw [hP, hW]; first | omega | (dsimp only; omega)) hk
  have hc : (FlockSoundness.Rows.ofNet s.hU).nIn + (FlockSoundness.Rows.ofNet s.hU).nComp = s.c.unit.constPos :=
    FlockSoundness.Rows.ofNet_one s.hU
  exact rowBitRead_of_out (k := ⟨k, hk⟩) u hj hj' (Nat.mod_lt _ (by norm_num))
    (by rw [Layout.ofNet_nIn_eq]; exact h1) (by rw [hc]; exact h2) hrow

/-- **A known port's bits are on wires of every unit**, where flock-verify's check accepts the session's circuit at
its known content: read at an input row (`rowBitRead_of_checkIn`), returned at an output row
(`rowBitRead_of_checkOut`). -/
theorem rowBitRead_of_check (u : Fin s.n) {known : Known} (h : _root_.Flock.KnownRead.check s.c known = .ok ())
    {nm : String} {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : Fin s.c.ports.size}
    (hn : s.c.ports[q].name = nm) {k : ℕ} (hk : k < portRowW s.c q) : RowBitRead s u q k := by
  by_cases hq : q.val < _root_.Flock.HmOut.nIn s.c
  · exact rowBitRead_of_checkIn u h hm hq hn hk
  · exact rowBitRead_of_checkOut u h hm hq hn hk

/-- **A known port's row, from the check**: a known value's row (`KnownPort`) of each known row's words. -/
theorem knownPort_of_check {known : Known} (h : _root_.Flock.KnownRead.check s.c known = .ok ()) {nm : String}
    {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : Fin s.c.ports.size}
    (hn : s.c.ports[q].name = nm) :
    KnownPort s q ∧ ∀ i (hi : i < ks.size), 16 * ks[i].1.size = portRowW s.c q := by
  have e : s.c.ports[q.val]! = s.c.ports[q] := getElem!_pos s.c.ports q.val q.isLt
  obtain ⟨hv, hsg, hw⟩ := shape_of_check h hm q.isLt (by rw [e]; exact hn)
  refine ⟨⟨hsg, hv⟩, fun i hi => ?_⟩
  show 16 * ks[i].1.size = 16 * s.c.ports[q].words
  rw [hw i hi, e]

theorem size_rowWords (bs : List Bool) : (rowWords bs).size = (bs.length + 15) / 16 := by
  simp [rowWords]

/-- **Lane `l`'s known fields hold at the opened rows, from flock-verify's check**: `laneKnownHolds` where the check
accepts the session's circuit at `knownOf P salts (some l)` and any rows after it (`more`, the ports read in full), which
makes the circuit name each port once, and each of
`admit`, `pub` and `o0` a port of it, a known value's row of its known row's words (32, 1616 and 16 bits) whose every
bit is on a wire of unit 0. -/
theorem laneKnownHolds_of_check (T : Tile)
    {P : LanePub} {salts : String → ByteArray} {l : Fin 32} (hK : KnownHeld s (knownOf P salts (some l)))
    {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {more : Known}
    (hread : _root_.Flock.KnownRead.check s.c (knownOf P salts (some l) ++ more) = .ok ()) :
    LaneKnownHolds T (portBit s (openedReg s)) P l := by
  have hma : ("admit", #[(admitRow (List.ofFn P.admit), salts "admit")]) ∈ knownOf P salts (some l) ++ more :=
    Array.mem_append.2 (Or.inl (by simp [knownOf, knownRows]))
  have hmp : ("pub", #[(pubRow P.salt P.position P.roots l.val, salts "pub")]) ∈ knownOf P salts (some l) ++ more :=
    Array.mem_append.2 (Or.inl (by simp [knownOf, knownRows]))
  have hmo : ("o0", #[(okRow, salts "o0")]) ∈ knownOf P salts (some l) ++ more :=
    Array.mem_append.2 (Or.inl (by simp [knownOf, knownRows]))
  obtain ⟨hN, hex⟩ := names_of_check hread hma
  obtain ⟨qa, ha⟩ := hex _ _ hma
  obtain ⟨qp, hp⟩ := hex _ _ hmp
  obtain ⟨qo, ho⟩ := hex _ _ hmo
  obtain ⟨hPa, hwa⟩ := knownPort_of_check hread hma ha
  obtain ⟨hPp, hwp⟩ := knownPort_of_check hread hmp hp
  obtain ⟨hPo, hwo⟩ := knownPort_of_check hread hmo ho
  have wa : portRowW s.c qa = 32 := by
    rw [← hwa 0 (by simp)]
    show 16 * (admitRow (List.ofFn P.admit)).size = 32
    simp [admitRow, size_rowWords]
  have wp : portRowW s.c qp = 1616 := by
    rw [← hwp 0 (by simp)]
    show 16 * (pubRow P.salt P.position P.roots l.val).size = 1616
    simp [pubRow, size_rowWords, length_pubBits]
  have wo : portRowW s.c qo = 16 := by
    rw [← hwo 0 (by simp)]
    show 16 * okRow.size = 16
    simp [okRow, size_rowWords]
  exact laneKnownHolds T hN hK hu hS hcomp ha hp ho hPa hPp hPo
    (fun k hk => rowBitRead_of_check u hread hma ha (by omega))
    (fun k hk => rowBitRead_of_check u hread hmp hp (by omega))
    (rowBitRead_of_check u hread hmo ho (by omega))

/-- **The combine reads `ok = 1`, from flock-verify's check**: `combKnownOk` where the check accepts the session's
circuit at `knownOf P salts none` and any rows after it (`more`), which makes the circuit name each port once, and `o0` a port of it, a known value's
row of one word whose bits are on wires of unit 0. -/
theorem combKnownOk_of_check
    {P : LanePub} {salts : String → ByteArray} (hK : KnownHeld s (knownOf P salts none)) {u : Fin s.n}
    (hu : u.val = 0) (hS : s.ServedRegistered u) (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ())
    {more : Known} (hread : _root_.Flock.KnownRead.check s.c (knownOf P salts none ++ more) = .ok ()) :
    portBit s (openedReg s) "o0" 0 = some true := by
  have hmo : ("o0", #[(okRow, salts "o0")]) ∈ knownOf P salts none ++ more :=
    Array.mem_append.2 (Or.inl (by simp [knownOf, knownRows]))
  obtain ⟨hN, hex⟩ := names_of_check hread hmo
  obtain ⟨qo, ho⟩ := hex _ _ hmo
  obtain ⟨hPo, hwo⟩ := knownPort_of_check hread hmo ho
  have wo : portRowW s.c qo = 16 := by
    rw [← hwo 0 (by simp)]
    show 16 * okRow.size = 16
    simp [okRow, size_rowWords]
  exact combKnownOk hN hK hu hS hcomp ho hPo (rowBitRead_of_check u hread hmo ho (by omega))

end Accounting.TS.Lanes
