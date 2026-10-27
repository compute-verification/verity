import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.E2E

/-!
# `ProgPlaces` from `Stmt.setupH`'s accept

`Prog.flock_headline` takes each drawn unit's place in its table's statement (`ProgPlaces`). This file builds it from the
executable's accept step on the untyped path (`tags.typed = false`, the statement production the circuit prover runs):

* `Accepted T`: a run of `Stmt.setupH` that accepted, whose statement (`Layout.model`, the level-0 matrices the verifier
  folds) is `T`. Its template instance rows `Accepted.rows` are the unit net's rows stacked `units_per_vu` times, placed
  at every template instance of every block (`Accepted.placement`, from `placement_of_setupH`).
* `progPlaces`: `ProgPlaces p outs plan tab` from an acceptance for each drawn unit's table, the unit's template instance and block, and
  W6 as `hrows` (the program's unit rows are the template instance rows its table placed). `placed` is proved; `inst`
  and `aliased` are taken (`aliased` in `ProgPlaces`' form, or from `Layout.Aliased` through
  `Accepted.aliased_of_layout`).
* `uniform`: the program of `n` units with the same rows, each reading only input gates, and `progPlaces_uniform`, where
  W6 is discharged: every table parsed one circuit file, and the program's rows are that circuit's (`circuitRows`).

On this path W6 holds by definition: setupH's placed rows depend only on the parsed circuit (`unitRows_eq`), and the
program is built from them. That a unit's rows compute its template's Definition is not checked by the untyped verifier.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Placed

open Model Audit

/-! ## The circuit's template instance rows -/

/-- **A template instance's rows for a circuit**: `units_per_vu` copies of the unit net's rows, one shared constant
(`Rows.stack`). -/
noncomputable def circuitRows (c : Flock.Circuit) (hU : Built c.unit) : Rows := (Rows.ofNet hU).stack c.upv

/-- The unit net's rows depend only on the circuit. -/
theorem ofNet_eq {c c' : Flock.Circuit} (e : c = c') (h : Built c.unit) (h' : Built c'.unit) :
    Rows.ofNet h = Rows.ofNet h' := by
  subst e; rfl

/-! ## An accepted statement -/

/-- **An accepted untyped statement whose model statement is `T`**: `Stmt.setupH` accepted the circuit file `cf`, public
file `pf`, coins, tables, draw, partition object and program, returning `st`, and `T` is the statement it places,
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

/-- The accepted statement's layout (`setupH_layout`). -/
theorem layout : Layout a.st := setupH_layout a.ht a.h

/-- **The accepted statement's template instance rows**: the unit net's rows, stacked `units_per_vu` times. -/
noncomputable def rows : Rows := (unitRows a.st a.layout.unit_built).stack a.st.c.upv

theorem kLog_eq : 2 ^ ((setupH_layout a.ht a.h).model a.regs).kLog = 2 ^ T.kLog :=
  congrArg (fun S : Model.Statement => 2 ^ S.kLog) a.hT

theorem nbl_eq : 2 ^ ((setupH_layout a.ht a.h).model a.regs).nbl = 2 ^ T.nbl :=
  congrArg (fun S : Model.Statement => 2 ^ S.nbl) a.hT

/-- **Template instance `g`'s columns in the block**: `stackPlace` at the unit slots `ur.slot (g·upv + i)`, the shared
constant at the pin. -/
noncomputable def col {g : ℕ} (hg : g < a.st.c.g) (c : Fin a.rows.w) : Fin (2 ^ T.kLog) :=
  Fin.cast a.kLog_eq
    (stackPlace ((setupH_layout a.ht a.h).model a.regs) (unitRows a.st a.layout.unit_built) a.st.c.upv
      (fun u => a.st.c.unitRange.slot (g * a.st.c.upv + u)) ((setupH_layout a.ht a.h).hb hg) c)

/-- A placement moves along an equal statement. -/
theorem placement_cast {S S' : Model.Statement} (hS : S = S') {R : Rows} {col : Fin R.w → Fin (2 ^ S.kLog)}
    (h : Placement S R col) :
    Placement S' R (fun c => Fin.cast (congrArg (fun S : Model.Statement => 2 ^ S.kLog) hS) (col c)) := by
  subst hS; exact h

/-- **The accepted statement places its template instance rows at every template instance** (`placement_of_setupH`), in
every block. -/
theorem placement {g : ℕ} (hg : g < a.st.c.g) : Placement T a.rows (a.col hg) :=
  placement_cast a.hT (placement_of_setupH a.ht a.h a.regs hg)

/-- **The template instance rows are the parsed circuit's** (W6 on the untyped path, by definition): they depend on the
circuit file alone, through `HmRow.parse`. -/
theorem rows_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) (hU : Built c.unit) :
    a.rows = circuitRows c hU := by
  have hc : a.st.c = c := by
    have := (setupH_spec a.ht a.h).1
    rw [hp] at this
    exact (Except.ok.inj this).symm
  unfold rows circuitRows unitRows
  rw [ofNet_eq hc a.layout.unit_built hU, hc]

/-- **`Layout.Aliased` in `ProgPlaces`' form**: the accepted layout's aliasing at template instance `g` and block `o`, for an instance of
rows `R` equal to the template instance rows, read at `col`. -/
theorem aliased_of_layout {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
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

/-! ## Each drawn unit's place -/

/-- A placement moves along equal rows. -/
theorem placement_rows {S : Model.Statement} {R R' : Rows} (hR : R = R') {col : Fin R'.w → Fin (2 ^ S.kLog)}
    (h : Placement S R' col) : Placement S R (fun c => col (Fin.cast (congrArg Rows.w hR) c)) := by
  subst hR; exact h

section Sites

variable {N n : ℕ} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each drawn unit's site**: the acceptance whose statement is the statement of the table unit `u` is drawn into
(`acc`), and the unit's template instance (`vu`, below the statement's `vus_per_block`) and block (`o`) there. -/
structure Sites where
  acc : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Accepted (Partition.tabOf plan tab S R u).S
  vu : ∀ (S : Finset (Fin n)) (_ : Reg) (u : Fin n), u ∈ S → ℕ
  hvu : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), vu S R u h < (acc S R u h).st.c.g
  o : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl)

variable {plan tab}

/-- **W6 at the sites**: each drawn unit's rows in the program are the template instance rows its table's acceptance
placed. -/
def Sites.RowsOf (si : Sites plan tab) (p : Prog N n) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), p.rows u = (si.acc S R u h).rows

variable {p : Prog N n} (si : Sites plan tab) (hrows : si.RowsOf p)

/-- A drawn unit's columns in its table's block: its template instance's (`Accepted.col`). -/
noncomputable def Sites.col (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin (p.rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun c => (si.acc S R u h).col (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) c)

/-- **W6's placement**: the statement of the table each drawn unit is drawn into places the unit's rows at its template
instance. -/
theorem Sites.placed (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Placement (Partition.tabOf plan tab S R u).S (p.rows u) (si.col hrows S R u h) :=
  placement_rows (hrows S R u h) ((si.acc S R u h).placement (si.hvu S R u h))

/-- **`ProgPlaces` from setupH's accept.** Each drawn unit's place is its template instance in its table's accepted
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
  placed := si.placed hrows
  aliased := hal

/-- **`ProgPlaces` from setupH's accept, its aliasing from `Layout.Aliased`** at each drawn unit's template instance and
block. -/
noncomputable def progPlaces_of_layout (outs : Finset (Fin N))
    (inst : ∀ u, (p.partition outs).IsRowsUnit u (p.rows u))
    (hal : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), (si.acc S R u h).layout.Aliased (si.acc S R u h).regs
      (hrows S R u h ▸ inst u) (Fin.cast (si.acc S R u h).nbl_eq.symm (si.o S R u h)) (si.hvu S R u h)) :
    ProgPlaces p outs plan tab :=
  progPlaces si hrows outs inst fun S R u h =>
    (si.acc S R u h).aliased_of_layout (hrows S R u h) (inst u) (si.o S R u h) (si.hvu S R u h) (hal S R u h)

end Sites

/-! ## The program of a circuit's units -/

/-- The gates of `n` units with rows `R` after `k` input gates. -/
def gates (k : ℕ) (R : Rows) : ℕ → ℕ
  | 0 => k
  | n + 1 => gates k R n + R.nComp

theorem le_gates (k : ℕ) (R : Rows) : ∀ n, k ≤ gates k R n
  | 0 => le_rfl
  | n + 1 => (le_gates k R n).trans (Nat.le_add_right _ _)

/-- **`n` units of rows `R` on `k` input gates**: unit `u`'s input columns read the input gates `ins u`, its constant the
input gate `one`, which none of them reads. No unit reads another's gates. -/
def uniform (k : ℕ) (R : Rows) (ins : ℕ → Fin R.nIn → Fin k) (one : Fin k) (hone : ∀ u c, ins u c ≠ one) :
    (n : ℕ) → Prog (gates k R n) n
  | 0 => .inputs k
  | n + 1 => .snoc (uniform k R ins one hone n) R (fun c => Fin.castLE (le_gates k R n) (ins n c))
      (Fin.castLE (le_gates k R n) one) (fun c h => hone n c (Fin.castLE_injective _ h))

/-- Every unit of `uniform` has rows `R`. -/
theorem rows_uniform (k : ℕ) (R : Rows) (ins : ℕ → Fin R.nIn → Fin k) (one : Fin k) (hone : ∀ u c, ins u c ≠ one) :
    ∀ n (u : Fin n), (uniform k R ins one hone n).rows u = R
  | 0, u => u.elim0
  | n + 1, u => by
      simp only [uniform, Prog.rows]
      split
      · exact rows_uniform k R ins one hone n _
      · rfl

section Uniform

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {k : ℕ} {c : Flock.Circuit} (hU : Built c.unit) (ins : ℕ → Fin (circuitRows c hU).nIn → Fin k) (one : Fin k)
  (hone : ∀ u c, ins u c ≠ one) (si : Sites plan tab)

/-- **W6 for the circuit's program, by definition**: when every drawn unit's table accepted a circuit file that parses
to `c`, the program of `c`'s template instance rows (`uniform … (circuitRows c hU)`) has, at every drawn unit, the rows its table placed. -/
theorem Sites.rowsOf_uniform
    (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c) :
    si.RowsOf (uniform k (circuitRows c hU) ins one hone n) := fun S R u h =>
  (rows_uniform k _ ins one hone n u).trans ((si.acc S R u h).rows_eq (hparse S R u h) hU).symm

end Uniform

end FlockSoundness.Discharge.Placed
