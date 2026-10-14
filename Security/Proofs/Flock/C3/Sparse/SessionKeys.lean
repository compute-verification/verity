import Proofs.Flock.C3.Sparse.SrcLayer
import Proofs.Flock.C3.Sparse.SessionReg
import Proofs.Flock.C3.Compose.RowsReg

/-!
# `sparseSessionAt_of_regs`' message keys, from the circuit, the reads file and the registration

`hkeys` (each message column a row of the sparse unit reads is a registered bit of copy 0 of unit `u`) is three facts,
each with its own producer, composed as `hRowKeys_of` composes `RowColKeys` and `RowsRegistered` at the parts:

* `SparseColKeys`: the unit's Δ key at each message column a row reads is a bit of the row of a port named as a message
  port, within the row. A fact of V*'s `rec-sparse` circuit file (`rec_sparse.ports`, the rows from column 0 in
  `messagePorts` order), `RowColKeys`' circuit half. It has no producer yet: `RecSparse.check` compares the unit alone,
  not `c.ports` or `c.leavesIn`, so it needs `RecOpen.rowsOk`/`leavesOk`'s refusals at `RecSparse.ports sh`, and then
  `check_rows` and `srcKey_opening`'s twin through `srcKey_inOrder` (`RegLevels/OpensRegs.lean`);
* `OwnNamesMsgs`: the session's reads file names every message port, `RowColKeys`' reads-file half. It is the
  conclusion of S's `sparseOwnNames_of_checkOwn` (`PubRows/Shape.lean`, from `RecVStar.checkOwn (.sparse sh hr) own`);
* `hread`: the session registers each row of a port its reads file names at unit `u`: `RowsRegisteredC3` at one
  session and unit (`rowsRegisteredC3_of_covered`, `rowsCovered_checked`), the form `regReads_of_check`,
  `haout_of_laid` and `sparseSessionAt_of_zero` take.

`hkeys_of` composes them at one session, as `regCols_of_check` does for an R level; the column is an input of the unit
by `rowCol_lt_nInS`. `hkeys_of_rows` takes `hread` from `RowsRegisteredC3` at an accepted outer proof, in
`sparseSessionAt_of_regs`' form.
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

section Family

open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **`sparseSessionAt_of_regs`' `hkeys` at an accepted outer proof** (`hkeys_of`), with `hread` from
`RowsRegisteredC3`. -/
theorem hkeys_of_rows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hrr : F.RowsRegisteredC3 dg N k Rw) {R : Reg} {ω : L.Ω} {t : List (Cm × (holoInnerAt T CT st hp).Coin)} {R₂ : Reg₂}
    {τ : (j : Fin (vstarCount F.ish)) → Strategy ((F.zs R ω t R₂ j).game dg N)}
    {o : (j : Fin (vstarCount F.ish)) → Reg' × Finset (Fin (F.Vs j).nv) × Exec.Transcript}
    (hA : Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o) {s : Fin (vstarCount F.ish)}
    {sh : Flock.RecSparse.Shape} {name : String}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok (F.Vs s).c.unit)
    (hcol : SparseColKeys (F.Vs s).c (F.Vs s).hU sh) (hown : OwnNamesMsgs (F.zs R ω t R₂ s).own sh)
    (u : Fin (F.Vs s).nv) :
    ∀ q < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum,
      ReadCol (Rows.ofNet (F.Vs s).hU) (rowCol (Flock.RecSparse.messagePorts sh) 0 q) →
        ∃ b, RegBit (F.zs R ω t R₂ s) u 0 (rowCol (Flock.RecSparse.messagePorts sh) 0 q) b :=
  hkeys_of (F.zs R ω t R₂ s) hn hcol hown (hrr R ω t R₂ τ o hA s u)

end Family

end FlockC3.Sparse

#print axioms FlockC3.Sparse.hkeys_of
#print axioms FlockC3.Sparse.hkeys_of_rows
