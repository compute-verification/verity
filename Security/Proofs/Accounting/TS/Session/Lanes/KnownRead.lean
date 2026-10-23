import Proofs.Accounting.TS.Session.Lanes.KnownFields
import Proofs.Flock.C3.Reps.ReadWhole
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.VBridge.Unit
import Proofs.Flock.Verifier.Registered
import Flock.KnownRead

/-!
# Every bit of a known input port is read, from flock-verify's check

`flock-verify --registered F` refuses a statement one of whose input ports `F` names at known content has a row bit that
no computed row of the unit reads (`Flock.KnownRead.check`, run on every statement beside the complete check). What that
refusal establishes, from the verifier's code:

* `srcAt_of_srcAtMsg`, `srcKey_of_srcMsg`: the check's `srcMsg` at copy `u`'s input column `y` is Δ's source key at
  stacked input `u · W + y` (`Aliased.srcKey`) where it names a message bit (its port bits a mask, `portMask_getD`);
* `colRead_of_read`: a column `readIns` marks, at a copy of the unit, whose key is bit `k` of port `q`, is a column a
  row of the circuit's template instance reads (`ColRead`), through the stack's rows (`stack_a`, `stack_b`);
* `read_of_check`, `colRead_of_check`: so every bit `k < 16 · words` of an input port (`HmOut.nIn`) the known content
  names is `ColRead`, and `rowBitRead_of_check` read at every unit (`RowBitRead`, `rowBitRead_of_col`);
* `laneKnownHolds_of_check`: `laneKnownHolds` with its `admit` and `pub` bits read by the check, at input ports of
  the widths the lane's layout reads (32 and 1616 bits).
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
  unfold _root_.Flock.KnownRead.check at h
  dsimp only at h
  have h1 := _root_.Flock.Registered.forM_ok h (nm, ks) (Array.mem_toList_iff.2 hm)
  dsimp only at h1
  have h2 := _root_.Flock.Registered.forM_ok h1 q (List.mem_range.2 hq)
  rw [if_pos hn] at h2
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
    exact ⟨u, _, hb.1.1, hb.1.2, hb.2⟩

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

variable {s : RowsSession}

/-- **A known input port's bits are read at every unit**, where flock-verify's check accepts the session's circuit at
its known content. -/
theorem rowBitRead_of_check (u : Fin s.n) {known : Known} (h : _root_.Flock.KnownRead.check s.c known = .ok ())
    {nm : String} {ks : Array (Array ℕ × ByteArray)} (hm : (nm, ks) ∈ known) {q : Fin s.c.ports.size}
    (hq : q.val < _root_.Flock.HmOut.nIn s.c) (hn : s.c.ports[q].name = nm) {k : ℕ} (hk : k < portRowW s.c q) :
    RowBitRead s u q k :=
  rowBitRead_of_col u hk (colRead_of_check s.hU (unitLog_of_scopeOkH s.hscope) h hm hq hn hk)

/-- **Lane `l`'s known fields hold at the opened rows, its admit and pub bits read by flock-verify's check**:
`laneKnownHolds` where the check accepts the session's circuit at `knownOf P salts (some l)`, `admit` and `pub` are
input ports (`HmOut.nIn`) of at least the 32 and 1616 bits the lane reads, and `o0`'s bit 0 is read. -/
theorem laneKnownHolds_of_check (T : Tile)
    (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {P : LanePub} {salts : String → ByteArray} {l : Fin 32} (hK : KnownHeld s (knownOf P salts (some l)))
    {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ())
    (hread : _root_.Flock.KnownRead.check s.c (knownOf P salts (some l)) = .ok ())
    {qa qp qo : Fin s.c.ports.size} (ha : s.c.ports[qa].name = "admit") (hp : s.c.ports[qp].name = "pub")
    (ho : s.c.ports[qo].name = "o0") (hPa : KnownPort s qa) (hPp : KnownPort s qp) (hPo : KnownPort s qo)
    (hqa : qa.val < _root_.Flock.HmOut.nIn s.c) (hqp : qp.val < _root_.Flock.HmOut.nIn s.c)
    (hwa : 32 ≤ portRowW s.c qa) (hwp : 1616 ≤ portRowW s.c qp) (hro : RowBitRead s u qo 0) :
    LaneKnownHolds T (portBit s (openedReg s)) P l := by
  have hma : ("admit", #[(admitRow (List.ofFn P.admit), salts "admit")]) ∈ knownOf P salts (some l) := by
    simp [knownOf, knownRows]
  have hmp : ("pub", #[(pubRow P.salt P.position P.roots l.val, salts "pub")]) ∈ knownOf P salts (some l) := by
    simp [knownOf, knownRows]
  exact laneKnownHolds T hN hK hu hS hcomp ha hp ho hPa hPp hPo
    (fun k hk => rowBitRead_of_check u hread hma hqa ha (by omega))
    (fun k hk => rowBitRead_of_check u hread hmp hqp hp (by omega)) hro

end Accounting.TS.Lanes
