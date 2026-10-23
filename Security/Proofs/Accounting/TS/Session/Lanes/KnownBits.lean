import Proofs.Accounting.TS.Session.Lanes.PortBits
import Proofs.Flock.VBridge.Decode

/-!
# A lane's read row bits from its circuit, and its known rows' bits

`Lanes.PortBits` reads `portBit` at the opened rows wherever the bit is on an input gate its unit reads
(`RowBitRead`), from the row's bytes. Here:

* `ColRead c hU q k`: some row of the circuit's template instance reads an input column whose Δ key is bit `k` of row
  port `q` (`Aliased.srcKey`), a fact of the parsed circuit alone;
* `rowBitRead_of_col`: that gives `RowBitRead` at every unit: the column's gate in the unit's own instance is that row
  bit (`FlockVBridge.Decode.keyRowOf_instC`, an input row's in `HiddenExec.rowOfH`) and, read by a row of the unit and
  no gate of it, is a wire of the unit (`IsRowsUnit.mem_io_of_read`);
* `known_held_at`, `known_bytes`, `known_readAt_zero`: where the known rows hold (`KnownHeld`), under flock-verify's
  complete check, a registrant's row of a port the known content names is read at a position of the known rows and has
  that known row's `sha512/row/v2` bytes; a one-row known value is read at row 0;
* `portBit_known`, `holds_known`: so `portBit` at the port's name reads the known row's bits.
-/

namespace Accounting.TS.Lanes

open Flock.SoundAtBridge Accounting.TS.Checked FlockSoundness.Discharge
open FlockSoundness.Discharge.Layout (keyN keyB keyRowOf portRowW)

/-- **Bit `k` of row port `q` is read by the circuit**: some row of the circuit's template instance reads an input
column whose Δ key is message bit `k % 8` of byte `k / 8` of row port `q`. -/
def ColRead (c : _root_.Flock.Circuit) (hU : FlockSoundness.Built c.unit) (q : Fin c.ports.size) (k : ℕ) : Prop :=
  ∃ (i : Fin (Placed.circuitRows c hU).nComp) (x : Fin (Placed.circuitRows c hU).w),
    x ∈ (Placed.circuitRows c hU).a i ++ (Placed.circuitRows c hU).b i ∧ x.val < (Placed.circuitRows c hU).nIn ∧
      Aliased.srcKey c x.val = .msg q.val (k / 8) (k % 8)

variable {s : RowsSession}

/-- **A bit the circuit reads is read at every unit**: the column's gate in unit `u`'s own instance carries bit `k` of
`u`'s row of `q`, and is a wire of `u`. -/
theorem rowBitRead_of_col (u : Fin s.n) {q : Fin s.c.ports.size} {k : ℕ} (hk : k < portRowW s.c q)
    (h : ColRead s.c s.hU q k) : RowBitRead s u q k := by
  obtain ⟨i, x, hx, hxn, hkey⟩ := h
  have key : ∀ (R : FlockSoundness.Rows) (_ : (Placed.keyProg s.c s.hU s.n).rows u = R) (i : Fin R.nComp)
      (x : Fin R.w), x ∈ R.a i ++ R.b i → x.val < R.nIn → R.nIn = keyB s.c s.hU →
      Aliased.srcKey s.c x.val = .msg q.val (k / 8) (k % 8) → RowBitRead s u q k := by
    intro R hR i x hx hxn hB hkey
    subst hR
    have hxB : x.val < keyB s.c s.hU := lt_of_lt_of_eq hxn hB
    have hr := FlockVBridge.Decode.keyRowOf_instC (Layout.outsOf s.c s.hU s.n) u x hxB hk hkey
    obtain ⟨-, hlt, -, -⟩ := Layout.keyRowOf_spec hr
    have hnone : (Placed.keyProg s.c s.hU s.n).unit
        ((Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).wire x) = none :=
      Layout.unit_uniform_input (k := Placed.inGates s.n (keyB s.c s.hU)) (R := Placed.circuitRows s.c s.hU)
        (ins := Placed.insOf s.n (keyB s.c s.hU) (Aliased.keyIdx s.c (keyB s.c s.hU)))
        (one := Placed.oneGate s.n (keyB s.c s.hU))
        (hone := Placed.insOf_ne_one s.n (keyB s.c s.hU) (Aliased.keyIdx s.c (keyB s.c s.hU))) s.n _
        (by show _ < 2 + s.n * keyB s.c s.hU; exact hlt)
    exact ⟨_, hk, HiddenExec.rowOfH_of_key _ hr,
      (Const.Prog.instC (Placed.keyProg s.c s.hU s.n) (Layout.outsOf s.c s.hU s.n) u).mem_io_of_read
        hx fun he => by cases hnone.symm.trans he⟩
  exact key _ (Placed.rows_circuitProg s.c s.hU s.n (keyB s.c s.hU) (Aliased.keyIdx s.c (keyB s.c s.hU)) u) i x hx
    hxn rfl hkey

/-- **A known port's registered row is a known row**: where the known rows hold (`KnownHeld`) and every port of the
circuit is a port of `own` (`Flock.Registered.checkComplete`), unit `u`'s registrant's row of a port `q` the known
content names, with rows `ks`, is read at a position of `ks` and has the `sha512/row/v2` bytes of the known row there. -/
theorem known_held_at {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) :
    (s.rdU u).readAt (Sum.inl (u, q)) < ks.size ∧ (rsOf s).enc (Sum.inl (u, q)) (rowVal s u q) =
      (_root_.Flock.Registered.rowBytesV2 ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1).data.toList := by
  have hn := _root_.Flock.Registered.checkComplete_ok hcomp _ (Array.getElem_mem q.isLt)
  have hread : (s.rdU u).isRead (Sum.inl (u, q)) := (hS.1 _).2 ⟨q, rfl, hn⟩
  have hname : ((s.rdU u).port (Sum.inl (u, q))).name = s.c.ports[q].name := by
    rw [(hS.2 q hn).1]
    obtain ⟨pt, hpt, hpn⟩ := hn
    exact Held.find_name_eq (l := s.own.toList) (Array.mem_toList_iff.2 hpt) hpn
  exact hK _ _ hm u _ hread hname

/-- A known port's registered row has the known row's bytes at its read's position (`known_held_at`). -/
theorem known_bytes {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) :
    (rsOf s).enc (Sum.inl (u, q)) (rowVal s u q) =
      (_root_.Flock.Registered.rowBytesV2 ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1).data.toList :=
  (known_held_at hK hS hcomp hm).2

/-- **A one-row known value is read at row 0**: a known port whose known content is one row is read at its position 0. -/
theorem known_readAt_zero {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    {r : Array ℕ × ByteArray} (hm : (s.c.ports[q].name, #[r]) ∈ known) :
    (s.rdU u).readAt (Sum.inl (u, q)) = 0 := by
  have h := (known_held_at hK hS hcomp hm).1
  simpa using h

/-- **`portBit` at a known port**: at a read bit `k` of unit 0's row of a `sha512/row/v2` port `q` without segments
that the known content names with rows `ks`, `portBit` at `q`'s name is bit `k` of the known row at the read's
position. -/
theorem portBit_known (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    (hseg : s.c.ports[q.val]!.segs = none) (hv2 : s.c.ports[q.val]!.v2 = true)
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) {k : ℕ}
    (hread : RowBitRead s u q k) :
    portBit s (openedReg s) s.c.ports[q].name k =
      some ((ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1[k / 16]!).testBit (k % 16)) :=
  portBit_bytes hN hu hS hcomp hseg hv2 (known_bytes hK hS hcomp hm) hread

/-- **A field of a known port**: where every bit of the field `[lo, lo + n)` of unit 0's row of a known port `q` is
read and the known row's bits there are `x`'s, `portBit` at `q`'s name holds `x` there. -/
theorem holds_known (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {known : Known} (hK : KnownHeld s known) {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    (hseg : s.c.ports[q.val]!.segs = none) (hv2 : s.c.ports[q.val]!.v2 = true)
    {ks : Array (Array ℕ × ByteArray)} (hm : (s.c.ports[q].name, ks) ∈ known) {lo n x : ℕ}
    (hread : ∀ i < n, RowBitRead s u q (lo + i))
    (hx : ∀ i < n, (ks[(s.rdU u).readAt (Sum.inl (u, q))]!.1[(lo + i) / 16]!).testBit ((lo + i) % 16) = x.testBit i) :
    Holds (portBit s (openedReg s) s.c.ports[q].name) lo n x :=
  holds_of_bytes hN hu hS hcomp hseg hv2 (known_bytes hK hS hcomp hm) hread hx

end Accounting.TS.Lanes
