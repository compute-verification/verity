import Proofs.Flock.Soundness.Discharge.Placed.Setup
import Proofs.Flock.Soundness.Types.ProgramPlaces
import Proofs.Flock.Soundness.ExecFlatCopies

/-!
# `ProgPlaces` on the typed path

On a typed flat-class statement (`tags.typed = true`), the verifier derives the unit's rows itself (`deriveChecked`), and
`setupH_flatTableClass` gives the accepted statement's `TableClass`. `UProg.derivedPlaces_of_classes` places each drawn
unit of a program of units (`UProg`) at a slot of its table's class, as `DerivedPlaces`. This file gives the headline's
`ProgPlaces` for the program's rows circuit `p.toRows hk` (whose circuit and partition are `p.C hk` and `p.P hk`):

* `ofRowsAt`: `ProgPlaces` from a place per drawn unit (`RowsAt`) aliased at a program-wide instance;
* `UProg.rows_toRows`: the rows circuit's unit rows are the units' composed rows;
* `UProg.progPlaces_of_classes`: `ProgPlaces (p.toRows hk) p.outsC plan tab` from each drawn unit's table class, slot and
  block, given that the unit reads the class's derivation inputs (`Reads`) and the slot's copies (`Copies`);
* `TypedAccepted`, `UProg.progPlaces_of_setupH`: the same, each class from `setupH_flatTableClass`.
-/

open Flock FlockLevel3 Flock.DeriveAll

namespace FlockSoundness.Discharge.Placed

open Model Audit Types.Dag Compose

/-! ## `ProgPlaces` from places -/

/-- An instance of equal rows. -/
def castInst {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R₁ R₂ : Rows} (e : R₁ = R₂)
    (i : P.IsRowsUnit u R₁) : P.IsRowsUnit u R₂ :=
  e ▸ i

/-- Aliasing moves along equal rows. -/
theorem aliased_cast {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {St : Model.Statement}
    {R₁ R₂ : Rows} (e : R₁ = R₂) (inst : P.IsRowsUnit u R₂) (a : RowsAt St R₂) (h : a.Aliased inst.wire) :
    (a.cast e.symm).Aliased (castInst e.symm inst).wire := by
  subst e; exact h

section OfRowsAt

variable {N n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}

/-- **`ProgPlaces` from a place per drawn unit**: unit `u`'s instance `inst u`, and a block and columns at which its
table's statement holds its rows (`pl`), carrying one value on its inputs that share a gate (`hal`). -/
def ofRowsAt (q : Prog N n) (outs : Finset (Fin N)) (inst : ∀ u, (q.partition outs).IsRowsUnit u (q.rows u))
    (pl : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → RowsAt (Partition.tabOf plan tab S R u).S (q.rows u))
    (hal : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), (pl S R u h).Aliased (inst u).wire) :
    ProgPlaces q outs plan tab where
  inst := inst
  o S R u h := (pl S R u h).o
  col S R u h := (pl S R u h).col
  placed S R u h := (pl S R u h).placed
  aliased := hal

end OfRowsAt

/-! ## A program of units -/

namespace UProg

variable {k one : ℕ}

/-- **The rows circuit's unit rows are the units' composed rows.** -/
theorem rows_toRows (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n),
    (p.toRows hk).rows u = p.rowsOf hk u
  | _, .inputs, u => u.elim0
  | _, .snoc p s src valid, u => by
      refine Fin.lastCases ?_ (fun v => ?_) u
      · rw [UProg.rowsOf, UProg.spec_last]
        simp [UProg.toRows, Prog.rows]
      · rw [UProg.rowsOf, UProg.spec_castSucc]
        simp only [UProg.toRows, Prog.rows, Fin.val_castSucc, v.isLt, ↓reduceDIte]
        exact rows_toRows hk p v

/-- Unit `u`'s instance in the rows circuit, at the rows circuit's unit rows. -/
noncomputable def instRows (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) :
    ((p.toRows hk).partition p.outsC).IsRowsUnit u ((p.toRows hk).rows u) :=
  castInst (rows_toRows hk p u).symm (p.instOf hk u)

variable {Reg : Type} {mPts : ℕ}

/-- **`ProgPlaces` for a program of units, from its tables' classes** (`derivedPlaces_of_classes` in the headline's
form): unit `u`, drawn in `S` under `R`, at slot `slot` of block `o` of its table's class `cls`, whose derivation inputs
it reads (`hr`), where the slot's inputs that read one of `u`'s sources copy one position, or forced-zero rows (`hcp`). -/
noncomputable def progPlaces_of_classes (hk : one < k) {outs : List ℕ} (p : UProg k one outs)
    (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (cls : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S → TableClass (Partition.tabOf plan tab S R u).S)
    (slot : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), Fin (cls S R u h).slots)
    (o : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), (p.spec hk u).Reads (cls S R u h))
    (hcp : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u)) :
    ProgPlaces (p.toRows hk) p.outsC plan tab :=
  ofRowsAt (p.toRows hk) p.outsC (instRows hk p)
    (fun S R u h => (((cls S R u h).atSlot (slot S R u h) (o S R u h)).cast
      (UnitSpec.R_eq_of_reads (hr S R u h)).symm).cast (rows_toRows hk p u).symm)
    fun S R u h => aliased_cast (rows_toRows hk p u) (p.instOf hk u) _
      (p.aliased_of_copies hk u (cls S R u h) (slot S R u h) (o S R u h) (hr S R u h) (hcp S R u h))

end UProg

/-! ## From setupH's accept -/

/-- **An accepted typed flat-class statement whose model statement is `T`**: `Stmt.setupH` accepted, the circuit file's
typed sections read to class `k` and flat unit net `net` (`Typed.read`), and `T` is the statement it places. -/
structure TypedAccepted (T : Model.Statement) where
  tags : Tags
  cf : ByteArray
  pf : ByteArray
  coins : Coins
  tables : List (String × ByteArray)
  draw : Option Lean.Json
  partition : Option Lean.Json
  program : Option ByteArray
  st : Stmt
  ht : tags.typed = true
  h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st
  k : Typed.Class
  net : Net
  hr : Typed.read tags cf tables = .ok (k, .inl net)
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hT : (setupH_flatLayout ht h hr).2.2.model regs = T

/-- A table class of an equal statement. -/
def TableClass.cast {St St' : Model.Statement} (e : St = St') (c : TableClass St) : TableClass St' := e ▸ c

/-- **The accepted statement's table class** (`setupH_flatTableClass`). -/
noncomputable def TypedAccepted.cls {T : Model.Statement} (a : TypedAccepted T) : TableClass T :=
  TableClass.cast a.hT (setupH_flatTableClass a.ht a.h a.hr a.regs)

namespace UProg

variable {k one : ℕ} {Reg : Type} {mPts : ℕ}

/-- **`ProgPlaces` for a program of units, from setupH's accept on the typed path**: each drawn unit at slot `slot` of
block `o` of its table's accepted class, given that it reads the class's derivation inputs (`hr`) and the slot's
copies (`hcp`). -/
noncomputable def progPlaces_of_setupH (hk : one < k) {outs : List ℕ} (p : UProg k one outs)
    (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (acc : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S → TypedAccepted (Partition.tabOf plan tab S R u).S)
    (slot : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), Fin (acc S R u h).cls.slots)
    (o : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S), (p.spec hk u).Reads (acc S R u h).cls)
    (hcp : ∀ (S : Finset (Fin p.n)) (R : Reg) (u : Fin p.n) (h : u ∈ S),
      (acc S R u h).cls.Copies (slot S R u h) (p.srcOf u)) :
    ProgPlaces (p.toRows hk) p.outsC plan tab :=
  progPlaces_of_classes hk p plan tab (fun S R u h => (acc S R u h).cls) slot o hr hcp

end UProg

end FlockSoundness.Discharge.Placed
