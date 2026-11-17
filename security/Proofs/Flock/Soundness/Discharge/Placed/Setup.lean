module

public import Proofs.Flock.Soundness.ExecSetup
public import Proofs.Flock.Soundness.E2E
public import Properties.Flock.Circuit.Keyed
public import Properties.Flock.Circuit.ProgRows
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Lowering

/-! `ProgPlaces` from `Stmt.setupH`'s accept on the untyped path, where W6 holds by definition and the verifier does
not check that a unit's rows compute its template's Definition. -/

@[expose] public section

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Placed

open Model Audit

theorem ofNet_eq {n n' : Flock.Net} (e : n = n') (h : Built n) (h' : Built n') :
    Rows.ofNet h = Rows.ofNet h' := by
  subst e; rfl

theorem lunit_of_parse {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c c' : Flock.Circuit}
    (hp : HmRow.parse tags bytes tables = .ok c) (e : c' = c) : lunit c' = c.unit := by
  subst e; exact lunit_flat (parse_facts hp).tmpl

/-- **An accepted untyped statement whose model statement is `T`**: `Stmt.setupH` accepted the circuit file `cf`, public
file `pf`, coins, tables, sample, partition object and program, returning `st`, and `T` is the statement it places,
`Layout.model` at regions `regs`. -/
structure Accepted (T : Model.Statement) where
  tags : Tags
  cf : ByteArray
  pf : ByteArray
  coins : Coins
  tables : List (String × ByteArray)
  draw : Option Lean.Json
  partition : Option Lean.Json
  program : Option ByteArray
  st : Stmt
  ht : tags.typed = false
  h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hT : (setupH_layout ht h).model regs = T

namespace Accepted

variable {T : Model.Statement} (a : Accepted T)

theorem layout : Layout a.st := setupH_layout a.ht a.h

/-- **The accepted statement's template instance rows**: the unit net's rows, stacked `units_per_vu` times. -/
noncomputable def rows : Rows := (unitRows a.st a.layout.unit_built).stack a.st.c.upv

theorem kLog_eq : 2 ^ ((setupH_layout a.ht a.h).model a.regs).kLog = 2 ^ T.kLog :=
  congrArg (fun S : Model.Statement => 2 ^ S.kLog) a.hT

theorem nbl_eq : 2 ^ ((setupH_layout a.ht a.h).model a.regs).nbl = 2 ^ T.nbl :=
  congrArg (fun S : Model.Statement => 2 ^ S.nbl) a.hT

/-- **Template instance `g`'s columns in the block** (`Layout.col`): the unit slots `ur.slot (g·upv + i)`, the shared
constant at the pin. -/
noncomputable def col {g : ℕ} (hg : g < a.st.c.g) (c : Fin a.rows.w) : Fin (2 ^ T.kLog) :=
  Fin.cast a.kLog_eq ((setupH_layout a.ht a.h).col a.regs hg c)

theorem placement_cast {S S' : Model.Statement} (hS : S = S') {R : Rows} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (h : Placement S R col) :
    Placement S' R (fun c => Fin.cast (congrArg (fun S : Model.Statement => 2 ^ S.kLog) hS) (col c)) := by
  subst hS; exact h

theorem placement {g : ℕ} (hg : g < a.st.c.g) : Placement T a.rows (a.col hg) :=
  placement_cast a.hT (placement_of_setupH a.ht a.h a.regs hg)

theorem rows_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) (hU : Built c.unit) :
    a.rows = circuitRows c hU := by
  have hc : a.st.c = c := by
    have := (setupH_spec a.ht a.h).1
    rw [hp] at this
    exact (Except.ok.inj this).symm
  unfold rows circuitRows unitRows
  rw [ofNet_eq (lunit_of_parse hp hc) a.layout.unit_built hU, hc]

theorem aliased_of_layout {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (hR : R = a.rows) (inst : P.IsRowsUnit u R) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hal : a.layout.Aliased a.regs (hR ▸ inst) (Fin.cast a.nbl_eq.symm o) hg) :
    ∀ c c', inst.wire c = inst.wire c' → c.val < R.nIn → c'.val < R.nIn → ∀ z : Model.Witness T.m, T.Satisfies z →
      T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c)) =
        T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c')) := by
  subst hR
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  exact hal

end Accepted

theorem placement_rows {S : Model.Statement} {R R' : Rows} (hR : R = R') {col : Fin R'.w → Fin (2 ^ S.kLog)}
    (h : Placement S R' col) : Placement S R (fun c => col (Fin.cast (congrArg Rows.w hR) c)) := by
  subst hR; exact h

section Sites

variable {N n : ℕ} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each sampled proof block's site**: the acceptance whose statement is the statement of the table unit `u` is sampled into
(`acc`), and the unit's template instance (`vu`, below the statement's `vus_per_block`) and block (`o`) there. -/
structure Sites where
  acc : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Accepted (Partition.tabOf plan tab S R u).S
  vu : ∀ (S : Finset (Fin n)) (_ : Reg) (u : Fin n), u ∈ S → ℕ
  hvu : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), vu S R u h < (acc S R u h).st.c.g
  o : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl)

variable {plan tab}

/-- **W6 at the sites**: each sampled proof block's rows in the program are the template instance rows its table's acceptance
placed. -/
def Sites.RowsOf (si : Sites plan tab) (p : Prog N n) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), p.rows u = (si.acc S R u h).rows

variable {p : Prog N n} (si : Sites plan tab) (hrows : si.RowsOf p)

/-- A sampled proof block's columns in its table's block: its template instance's (`Accepted.col`). -/
noncomputable def Sites.col (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin (p.rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun c => (si.acc S R u h).col (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) c)

theorem Sites.placed (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Placement (Partition.tabOf plan tab S R u).S (p.rows u) (si.col hrows S R u h) :=
  placement_rows (hrows S R u h) ((si.acc S R u h).placement (si.hvu S R u h))

/-- **`ProgPlaces` from setupH's accept.** Each sampled proof block's place is its template instance in its table's accepted
statement (`si`); `placed` holds given W6 (`hrows`). The unit's instance (`inst`) and its inputs' aliasing there (`hal`)
are taken. -/
noncomputable def progPlaces (outs : Finset (Fin N)) (inst : ∀ u, (p.partition outs).IsRowsUnit u (p.rows u))
    (hal : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), ∀ c c', (inst u).wire c = (inst u).wire c' → c.val < (p.rows u).nIn →
      c'.val < (p.rows u).nIn → ∀ z : Model.Witness (Partition.tabOf plan tab S R u).S.m,
        (Partition.tabOf plan tab S R u).S.Satisfies z →
        (Partition.tabOf plan tab S R u).S.block z (si.o S R u h) (si.col hrows S R u h c) =
          (Partition.tabOf plan tab S R u).S.block z (si.o S R u h) (si.col hrows S R u h c')) :
    ProgPlaces p outs plan tab where
  inst := inst
  o := si.o
  col := si.col hrows
  placed S R u h := (si.placed hrows S R u h).sat
  aliased := hal

/-- **`ProgPlaces` from setupH's accept, its aliasing from `Layout.Aliased`** at each sampled proof block's template instance and
block. -/
noncomputable def progPlaces_of_layout (outs : Finset (Fin N))
    (inst : ∀ u, (p.partition outs).IsRowsUnit u (p.rows u))
    (hal : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), (si.acc S R u h).layout.Aliased (si.acc S R u h).regs
      (hrows S R u h ▸ inst u) (Fin.cast (si.acc S R u h).nbl_eq.symm (si.o S R u h)) (si.hvu S R u h)) :
    ProgPlaces p outs plan tab :=
  progPlaces si hrows outs inst fun S R u h =>
    (si.acc S R u h).aliased_of_layout (hrows S R u h) (inst u) (si.o S R u h) (si.hvu S R u h) (hal S R u h)

end Sites

section Uniform

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {k : ℕ} {c : Flock.Circuit} (hU : Built c.unit) (ins : ℕ → Fin (circuitRows c hU).nIn → Fin k) (one : Fin k)
  (hone : ∀ u c, ins u c ≠ one) (si : Sites plan tab)

theorem Sites.rowsOf_uniform
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c) :
    si.RowsOf (uniform k (circuitRows c hU) ins one hone n) := fun S R u h =>
  (rows_uniform k _ ins one hone n u).trans ((si.acc S R u h).rows_eq (hparse S R u h) hU).symm

end Uniform

end FlockSoundness.Discharge.Placed
