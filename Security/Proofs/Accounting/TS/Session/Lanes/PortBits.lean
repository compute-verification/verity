import Proofs.Accounting.TS.Session.Lanes.Holds
import Proofs.Accounting.TS.HeldSound
import Proofs.Accounting.TS.Session.Held.Known
import Proofs.Flock.VBridge.Family.Registrants

/-!
# What a lane's decode reads at the opened rows: the registrant's row bits

`decLane` and `decComb` read their ports' row bits through `portBit` at the opened rows (`openedReg`). This file says
what those bits are, so `LaneHolds` and `CombHolds` become statements of the rows' bytes:

* `RowBitRead s u q k`: bit `k` of unit `u`'s row of port `q` is on a gate of the key program that is a wire of `u`
  (`HiddenExec.rowOfH`, `P.io`): an input gate `u` reads, or a gate `u` returns. A fact of the circuit;
* `vb_pos_row`, `vb_bit_row`: the value binding puts such a gate in its row's commit string, at bit `k`
  (`Readers.ofRows`);
* `openedReg_row`: under flock-verify's complete check (`Flock.Registered.checkComplete`) and a registrant that reads
  `u`'s served roots (`ServedRegistered`), the opened rows at such a gate are bit `k` of the value `u`'s registrant
  opened at that row (`rowVal`);
* `portBit_opened`: so in a circuit that names each port once, `portBit` at the port's name is that bit;
* `rowVal_bit`: at a `sha512/row/v2` port without segments, a value whose row bytes are `rowBytesV2 row` has bit `k`
  the row's word `k / 16`'s bit `k % 16` (`FlockC3.RegLevels.bits_of_v2`);
* `portBit_bytes`, `holds_of_bytes`: `portBit` at a read bit, and `Holds` on a field all of whose bits are read, from
  the row's bytes. A known row (`KnownHeld`, flock-verify's `--known` check) and a registered one give those bytes.
-/

namespace Accounting.TS.Lanes

open Flock.SoundAtBridge Accounting.TS.Checked FlockSoundness.Discharge
open FlockSoundness.Discharge.Layout (keyN keyRowOf portRowW)

variable (s : RowsSession)

/-- **The value unit `u`'s registrant opened at its row of port `q`**. -/
noncomputable abbrev rowVal (u : Fin s.n) (q : Fin s.c.ports.size) :
    Layout.BitVal (Finset.univ.sup (portRowW s.c) + 1) :=
  (s.rdU u).wv (Sum.inl (u, q))

/-- **Bit `k` of unit `u`'s row of port `q` is on a wire of `u`**: a gate of the key program carries it (the
hidden-output layout `HiddenExec.rowOfH` at the session's row widths) and is a wire of `u` (`P.io`): an input gate some
gate of `u` reads, at an input row, or a gate `u` returns, at an output row. -/
def RowBitRead (u : Fin s.n) (q : Fin s.c.ports.size) (k : ℕ) : Prop :=
  ∃ (g : Fin (keyN s.c s.hU s.n)) (hk : k < portRowW s.c q),
    HiddenExec.rowOfH s.c s.hU s.n (Layout.outsOf s.c s.hU s.n) g = some (u, ⟨q, ⟨k, hk⟩⟩) ∧
      (g : Fin s.Circ.N) ∈ s.P.io u

variable {s}

theorem rowW_lt (q : Fin s.c.ports.size) {k : ℕ} (hk : k < portRowW s.c q) :
    k < Finset.univ.sup (portRowW s.c) + 1 :=
  Nat.lt_succ_of_le ((Nat.le_of_lt hk).trans (Finset.le_sup (f := portRowW s.c) (Finset.mem_univ q)))

/-- A row bit's wire is in its row's commit string. -/
theorem vb_pos_row {g : Fin s.Circ.N} {u : Fin s.n} {q : Fin s.c.ports.size} {k : Fin (portRowW s.c q)}
    (h : HiddenExec.rowOfH s.c s.hU s.n (Layout.outsOf s.c s.hU s.n) g = some (u, ⟨q, k⟩)) :
    s.vb.pos g = Sum.inl (u, q) := by
  dsimp only [RowsSession.vb, ZkReg.rsZHJS, ZkBind.HmRowsZX.bindingZ, ZkBind.LayoutZ.bindingZ, ZkHidden.layZHJ,
    ZkBind.LayoutZ.ofShadow, ZkHidden.layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout, Layout.ofBits,
    HiddenExec.keyReadersH]
  exact Layout.ofRows_pos_some _ h

/-- A row bit's wire is its bit `k` of its commit string's value. -/
theorem vb_bit_row (v : Layout.BitVal (Finset.univ.sup (portRowW s.c) + 1)) {g : Fin s.Circ.N} {u : Fin s.n}
    {q : Fin s.c.ports.size} {k : Fin (portRowW s.c q)}
    (h : HiddenExec.rowOfH s.c s.hU s.n (Layout.outsOf s.c s.hU s.n) g = some (u, ⟨q, k⟩)) :
    s.vb.bit v g = v.get k.val := by
  rw [show v.get k.val = v.2 ⟨k.val, rowW_lt q k.isLt⟩ by
    unfold Layout.BitVal.get; rw [dite_eq_left (rowW_lt q k.isLt)]]
  dsimp only [RowsSession.vb, ZkReg.rsZHJS, ZkBind.HmRowsZX.bindingZ, ZkBind.LayoutZ.bindingZ, ZkHidden.layZHJ,
    ZkBind.LayoutZ.ofShadow, ZkHidden.layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout, Layout.ofBits,
    HiddenExec.keyReadersH, Layout.Readers.ofRows]
  rw [h]
  rfl

/-- **The opened rows at a read row bit**: where unit `u`'s registrant reads its served roots' rows
(`ServedRegistered`) and every port of the circuit is a port of `own` (`Flock.Registered.checkComplete`), the opened
rows at a gate that carries bit `k` of `u`'s row of port `q` and is a wire of `u` are that bit of the value `u`'s
registrant opened there. -/
theorem openedReg_row {u : Fin s.n} (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {g : Fin s.Circ.N}
    {q : Fin s.c.ports.size} {k : Fin (portRowW s.c q)}
    (hr : HiddenExec.rowOfH s.c s.hU s.n (Layout.outsOf s.c s.hU s.n) g = some (u, ⟨q, k⟩)) (hg : g ∈ s.P.io u)
    {a : ℕ} (ha : u.val = a) :
    openedReg s a g.val = some ((rowVal s u q).get k.val) := by
  subst ha
  have hpos := vb_pos_row hr
  have hread : (s.rdU u).isRead (s.vb.pos g) :=
    (hS.1 _).2 ⟨q, hpos, _root_.Flock.Registered.checkComplete_ok hcomp _ (Array.getElem_mem q.isLt)⟩
  unfold openedReg
  rw [Held.regRead_at s _ hg hread]
  show some (s.vb.bit (s.vb.val (s.wo u (s.vb.pos g))) g) = _
  have e : s.vb.val (s.wo u (s.vb.pos g)) = rowVal s u q := congrArg (fun p => s.vb.val (s.wo u p)) hpos
  exact congrArg some ((vb_bit_row _ hr).trans
    (congrArg (fun v : Layout.BitVal (Finset.univ.sup (portRowW s.c) + 1) => v.get k.val) e))

/-- **`portBit` at the opened rows is the registrant's row bit**: in a circuit that names each port once, at a read
bit `k` of unit 0's row of port `q` (`RowBitRead`), `portBit` at `q`'s name is bit `k` of the value unit 0's registrant
opened at that row. -/
theorem portBit_opened (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size} {k : ℕ}
    (hread : RowBitRead s u q k) :
    portBit s (openedReg s) s.c.ports[q].name k = some ((rowVal s u q).get k) := by
  obtain ⟨g₀, hk, hk₀, hg₀⟩ := hread
  have hex : ∃ (g : Fin (keyN s.c s.hU s.n)) (u' : Fin s.n) (q' : Fin s.c.ports.size) (b : Fin (portRowW s.c q')),
      u'.val = 0 ∧ s.c.ports[q'].name = s.c.ports[q].name ∧ b.val = k ∧
        HiddenExec.rowOfH s.c s.hU s.n (Layout.outsOf s.c s.hU s.n) g = some (u', ⟨q', b⟩) ∧
        (g : Fin s.Circ.N) ∈ s.P.io u' :=
    ⟨g₀, u, q, ⟨k, hk⟩, hu, rfl, rfl, hk₀, hg₀⟩
  unfold portBit
  rw [dite_eq_left hex]
  obtain ⟨u', q', b, hu', hn, hb, hkey, hio⟩ := Classical.choose_spec hex
  obtain rfl : u' = u := Fin.ext (hu'.trans hu.symm)
  obtain rfl : q' = q := hN _ _ hn
  exact (openedReg_row (g := Classical.choose hex) hS hcomp hkey hio hu).trans (by rw [hb])

/-- **A `sha512/row/v2` row's bits from its bytes**: at a port without segments whose rows are `sha512/row/v2`'s, a
registrant's value whose row bytes (`rsOf`'s encoding) are `rowBytesV2 row` has bit `k`, within the row's width, the
row's word `k / 16`'s bit `k % 16`. -/
theorem rowVal_bit {u : Fin s.n} {q : Fin s.c.ports.size} (hseg : s.c.ports[q.val]!.segs = none)
    (hv2 : s.c.ports[q.val]!.v2 = true) {row : Array ℕ}
    (h : (rsOf s).enc (Sum.inl (u, q)) (rowVal s u q) = (_root_.Flock.Registered.rowBytesV2 row).data.toList)
    {k : ℕ} (hk : k < portRowW s.c q) :
    (rowVal s u q).get k = (row[k / 16]!).testBit (k % 16) := by
  have h' : HiddenExec.encPortH s.c.ports[q.val]! (rowVal s u q) =
      (_root_.Flock.Registered.rowBytesV2 row).data.toList := h
  have h2 := (congrFun (FlockVBridge.Family.encPortH_of_none hseg) (rowVal s u q)).symm.trans h'
  rw [hv2] at h2
  unfold Layout.BitVal.get
  rw [dite_eq_left (rowW_lt q hk)]
  exact FlockC3.RegLevels.bits_of_v2 h2 k _

/-- **`portBit` at a read bit, from the row's bytes**: `portBit_opened` and `rowVal_bit`. -/
theorem portBit_bytes (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    (hseg : s.c.ports[q.val]!.segs = none) (hv2 : s.c.ports[q.val]!.v2 = true) {row : Array ℕ}
    (h : (rsOf s).enc (Sum.inl (u, q)) (rowVal s u q) = (_root_.Flock.Registered.rowBytesV2 row).data.toList)
    {k : ℕ} (hread : RowBitRead s u q k) :
    portBit s (openedReg s) s.c.ports[q].name k = some ((row[k / 16]!).testBit (k % 16)) := by
  obtain ⟨g, hk, hrest⟩ := hread
  rw [portBit_opened hN hu hS hcomp ⟨g, hk, hrest⟩, rowVal_bit hseg hv2 h hk]

/-- **A field's bits at the opened rows, from the row's bytes**: where every bit of the field `[lo, lo + n)` of unit 0's
row of port `q` is read (`RowBitRead`), and the row's bits there are `x`'s, `portBit` at `q`'s name holds `x` there
(`Holds`). -/
theorem holds_of_bytes (hN : ∀ q q' : Fin s.c.ports.size, s.c.ports[q].name = s.c.ports[q'].name → q = q')
    {u : Fin s.n} (hu : u.val = 0) (hS : s.ServedRegistered u)
    (hcomp : _root_.Flock.Registered.checkComplete s.c s.own = .ok ()) {q : Fin s.c.ports.size}
    (hseg : s.c.ports[q.val]!.segs = none) (hv2 : s.c.ports[q.val]!.v2 = true) {row : Array ℕ}
    (h : (rsOf s).enc (Sum.inl (u, q)) (rowVal s u q) = (_root_.Flock.Registered.rowBytesV2 row).data.toList)
    {lo n x : ℕ} (hread : ∀ i < n, RowBitRead s u q (lo + i))
    (hx : ∀ i < n, (row[(lo + i) / 16]!).testBit ((lo + i) % 16) = x.testBit i) :
    Holds (portBit s (openedReg s) s.c.ports[q].name) lo n x := fun i hi => by
  rw [portBit_bytes hN hu hS hcomp hseg hv2 h (hread i hi), hx i hi]

end Accounting.TS.Lanes
