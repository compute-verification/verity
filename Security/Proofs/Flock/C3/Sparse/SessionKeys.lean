import Proofs.Flock.C3.Sparse.SrcLayer

/-!
# `sparseSessionAt_of_regs`' message keys, from the circuit, the reads file and the registration

`hkeys` (each message column a row of the sparse unit reads is a registered bit of copy 0 of unit `u`) is three facts,
each with its own producer, composed as `hRowKeys_of` composes `RowColKeys` and `RowsRegistered` at the parts:

* `SparseColKeys`: the unit's Δ key at each message column a row reads is a bit of the row of a port named as a message
  port, within the row. A fact of V*'s `rec-sparse` circuit file (`rec_sparse.ports`, the rows from column 0 in
  `messagePorts` order), `RowColKeys`' circuit half;
* `OwnNamesMsgs`: the session's reads file names every message port (`rec_sparse.write_rows` names each row port of
  the statement), `RowColKeys`' reads-file half;
* `hread`: the session registers each row of a port its reads file names at unit `u`: `RowsRegistered` at one session
  and unit, the form `regBit_of_colKeyed`, `haout_of_laid` and `sparseSessionAt_of_zero` take.

`hkeys_of` composes them; the column is an input of the unit by `rowCol_lt_nInS`.
-/

namespace FlockC3.Sparse

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode (RegBit)
open FlockVBridge.OfVStar (ReadCol rowCol)

/-- **The sparse unit keys its message columns by message rows**: at each message column a row reads, the Δ key of copy
0's input is bit `k'` of the row of port `q'`, within the row, and `q'` is named as a message port. -/
def SparseColKeys (c : Flock.Circuit) (hU : Built c.unit) (sh : Flock.RecSparse.Shape) : Prop :=
  ∀ q < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum,
    ReadCol (Rows.ofNet hU) (rowCol (Flock.RecSparse.messagePorts sh) 0 q) →
      ∃ (q' : Fin c.ports.size) (k' : ℕ) (_ : k' < Layout.portRowW c q'),
        Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + rowCol (Flock.RecSparse.messagePorts sh) 0 q) =
          .msg q'.val (k' / 8) (k' % 8) ∧
        c.ports[q'].name ∈ (Flock.RecSparse.messagePorts sh).map (·.1)

/-- **The reads file names every message port.** -/
def OwnNamesMsgs (own : Array Flock.Registered.Port) (sh : Flock.RecSparse.Shape) : Prop :=
  ∀ nm ∈ (Flock.RecSparse.messagePorts sh).map (·.1), ∃ pt ∈ own, pt.name = nm

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv}

/-- **`hkeys` from the circuit's keys, the reads file and the registration** (`hRowKeys_of`'s twin): the column's key
names a message row, which the reads file names, so a registered read whose bit is the registrant's. -/
theorem hkeys_of (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) {sh : Flock.RecSparse.Shape} {name : String}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok c.unit) {u : Fin nv} (hcol : SparseColKeys c hU sh)
    (hown : OwnNamesMsgs zo.own sh)
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q))) :
    ∀ q < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum,
      ReadCol (Rows.ofNet hU) (rowCol (Flock.RecSparse.messagePorts sh) 0 q) →
        ∃ b, RegBit zo u 0 (rowCol (Flock.RecSparse.messagePorts sh) 0 q) b := by
  intro q hq hrd
  obtain ⟨q', k', hk, hkey, hnm⟩ := hcol q hq hrd
  obtain ⟨pt, hpt, hpn⟩ := hown _ hnm
  exact ⟨_, FlockVBridge.Sparse.rowCol_lt_nInS hU hn hq, q', k', hk, hkey, hread q' ⟨pt, hpt, hpn⟩, rfl⟩

end FlockC3.Sparse
