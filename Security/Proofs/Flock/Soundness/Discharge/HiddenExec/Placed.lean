import Proofs.Flock.Soundness.Discharge.HiddenExec.Copies
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup

/-!
# `ProgPlaces`, `hConst` and `hZero` from `Stmt.setupHidden`'s accept

`Placed/Setup.lean`, `Placed/Keyed.lean`, `Const/Keyed.lean` and `Zero/Uniform.lean`'s circuit section, for untyped
hidden-output statements:

* `AcceptedH T`: a run of `Stmt.setupHidden` that accepted, with a layout `L` of its statement (`Layout`, which
  `Hidden.hidden_layout` gives), whose unit slots' rows are its flat twin's (`UnitRowsFlat`, which `unitRowsFlat` gives),
  and whose block R1CS `L.model regs` is `T`. Its template instance rows are the unit net's rows stacked `units_per_vu` times, placed at
  every template instance (`AcceptedH.placement`); its aliasing comes from Δ's source keys (`AcceptedH.aliased_of_key`,
  through `aliased_of_copiesH`), and a stacked input that reads the zero carries 0 (`AcceptedH.zero_block`).
* `SitesH plan tab`: each drawn unit's acceptance, template instance and block, as `Placed.Sites`.
* `SitesH.progPlaces_keyProg`: `ProgPlaces` for the circuit's program on Δ's source keys (`Placed.keyProg`), from the
  sites and the parse alone; `constCols_keyProgH` and `zeroCols_keyProgH` are its `hConst` and `hZero`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3
open Model Audit
open FlockSoundness.Discharge.Zero (ZeroIn)

/-! ## An accepted hidden-output statement -/

/-- **An accepted untyped hidden-output statement whose block R1CS is `T`**: `Stmt.setupHidden` accepted the circuit
file `cf`, public file `pf`, coins, tables, draw, partition object and program, returning `st`; `L` is a layout of `st`,
whose unit slots' rows are its flat twin's (`hR`), and `T` is `L.model` at regions `regs`. -/
structure AcceptedH (T : Model.Statement) where
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
  h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st
  L : Layout st
  hR : UnitRowsFlat st
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hT : L.model regs = T

namespace AcceptedH

variable {T : Model.Statement} (a : AcceptedH T)

/-- The flat twin's layout (`flat_layout`). -/
theorem flatLayout : Layout (Hidden.flatTwin a.st) := flat_layout a.ht a.h

/-- The flat twin's unit facts (`flat_unitFacts`). -/
theorem unitFacts : Aliased.UnitFacts a.tags (Hidden.flatTwin a.st) := flat_unitFacts a.ht a.h

/-- **The accepted statement's template instance rows**: the unit net's rows, stacked `units_per_vu` times. -/
noncomputable def rows : Rows := (unitRows a.st a.L.unit_built).stack a.st.c.upv

theorem kLog_eq : 2 ^ (a.L.model a.regs).kLog = 2 ^ T.kLog :=
  congrArg (fun S : Model.Statement => 2 ^ S.kLog) a.hT

theorem nbl_eq : 2 ^ (a.L.model a.regs).nbl = 2 ^ T.nbl :=
  congrArg (fun S : Model.Statement => 2 ^ S.nbl) a.hT

/-- **Template instance `g`'s columns in the block** (`Layout.col`): the unit slots `ur.slot (g·upv + i)`, the shared
constant at the pin. -/
noncomputable def col {g : ℕ} (hg : g < a.st.c.g) (c : Fin a.rows.w) : Fin (2 ^ T.kLog) :=
  Fin.cast a.kLog_eq (a.L.col a.regs hg c)

/-- **The accepted statement places its template instance rows at every template instance** (`Layout.placement`), in
every block. -/
theorem placement {g : ℕ} (hg : g < a.st.c.g) : Placement T a.rows (a.col hg) :=
  Placed.Accepted.placement_cast a.hT (a.L.placement a.regs hg)

/-- An accepted statement's circuit is the one its circuit file parses to (`parse_untyped`). -/
theorem c_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) : a.st.c = c := by
  have := (Hidden.parse_untyped a.ht (Hidden.setupHidden_spec' a.h).parse).1
  rw [hp] at this
  exact (Except.ok.inj this).symm

/-- **The template instance rows are the parsed circuit's** (W6 on the untyped path, by definition). -/
theorem rows_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) (hU : Built c.unit) :
    a.rows = Placed.circuitRows c hU := by
  have hc := a.c_eq hp
  unfold rows Placed.circuitRows unitRows
  rw [Placed.ofNet_eq (Placed.lunit_of_parse hp hc) a.L.unit_built hU, hc]

/-- **`Layout.Aliased` in `ProgPlaces`' form**, as `Placed.Accepted.aliased_of_layout`. -/
theorem aliased_of_layout {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (hR : R = a.rows) (inst : P.IsRowsUnit u R) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hal : a.L.Aliased a.regs (hR ▸ inst) (Fin.cast a.nbl_eq.symm o) hg) :
    ∀ c c', inst.wire c = inst.wire c' → c.val < R.nIn → c'.val < R.nIn → ∀ z : Model.Witness T.m, T.Satisfies z →
      T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c)) =
        T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c')) := by
  subst hR
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  exact hal

/-- **Aliasing from keys**: an instance that puts template instance `g`'s stacked unit inputs on one gate only when they
have one key (`Aliased.srcKey`) is aliased in the accepted statement. -/
theorem aliased_of_key {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows a.st a.L.unit_built).stack a.st.c.upv))
    (o : Fin (2 ^ (a.L.model a.regs).nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hk : ∀ c c', inst.wire c = inst.wire c' →
      c.val < ((unitRows a.st a.L.unit_built).stack a.st.c.upv).nIn →
      c'.val < ((unitRows a.st a.L.unit_built).stack a.st.c.upv).nIn →
        Aliased.srcKey a.st.c c.val = Aliased.srcKey a.st.c c'.val) :
    a.L.Aliased a.regs inst o hg :=
  aliased_of_copiesH a.L a.flatLayout a.regs a.unitFacts a.hR inst hg
    (Aliased.copies_of_key a.flatLayout a.regs a.unitFacts inst hg hk) o

/-- **A template instance's stacked input that reads the zero carries 0**, at the template instance's columns, in every
block of a satisfying witness (`stack_zero_blockH`). -/
theorem zero_block {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w) (hx : x.val < a.rows.nIn)
    (hz : ZeroIn a.st.c x.val) {z : Model.Witness T.m} (hsat : T.Satisfies z) (o : Fin (2 ^ T.nbl)) :
    T.block z o (a.col hg x) = 0 := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  exact stack_zero_blockH L (flat_layout ht h) regs (flat_unitFacts ht h) hR hg x hx hz hsat o

end AcceptedH

/-! ## Each drawn unit's place -/

section Sites

variable {N n : ℕ} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each drawn unit's site**, as `Placed.Sites`: the hidden-output acceptance whose statement is the statement of the
table unit `u` is drawn into (`acc`), and the unit's template instance (`vu`, below `vus_per_block`) and block (`o`)
there. -/
structure SitesH where
  acc : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → AcceptedH (Partition.tabOf plan tab S R u).S
  vu : ∀ (S : Finset (Fin n)) (_ : Reg) (u : Fin n), u ∈ S → ℕ
  hvu : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), vu S R u h < (acc S R u h).st.c.g
  o : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl)

variable {plan tab}

/-- **W6 at the sites**: each drawn unit's rows in the program are the template instance rows its table's acceptance
placed. -/
def SitesH.RowsOf (si : SitesH plan tab) (p : Prog N n) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), p.rows u = (si.acc S R u h).rows

variable {p : Prog N n} (si : SitesH plan tab) (hrows : si.RowsOf p)

/-- A drawn unit's columns in its table's block: its template instance's (`AcceptedH.col`). -/
noncomputable def SitesH.col (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin (p.rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun c => (si.acc S R u h).col (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) c)

/-- **W6's placement**: the statement of the table each drawn unit is drawn into places the unit's rows at its template
instance. -/
theorem SitesH.placed (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Placement (Partition.tabOf plan tab S R u).S (p.rows u) (si.col hrows S R u h) :=
  Placed.placement_rows (hrows S R u h) ((si.acc S R u h).placement (si.hvu S R u h))

/-- **`ProgPlaces` from setupHidden's accept**, its aliasing from `Layout.Aliased` at each drawn unit's template
instance and block. -/
noncomputable def SitesH.progPlaces_of_layout (outs : Finset (Fin N))
    (inst : ∀ u, (p.partition outs).IsRowsUnit u (p.rows u))
    (hal : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), (si.acc S R u h).L.Aliased (si.acc S R u h).regs
      (hrows S R u h ▸ inst u) (Fin.cast (si.acc S R u h).nbl_eq.symm (si.o S R u h)) (si.hvu S R u h)) :
    ProgPlaces p outs plan tab where
  inst := inst
  o := si.o
  col := si.col hrows
  placed := si.placed hrows
  aliased S R u h :=
    (si.acc S R u h).aliased_of_layout (hrows S R u h) (inst u) (si.o S R u h) (si.hvu S R u h) (hal S R u h)

end Sites

/-! ## The circuit's program on Δ's source keys -/

section keyed

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {c : Flock.Circuit} (hU : Built c.unit) {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)

include hparse in
/-- **W6 for the circuit's program, by definition**, as `Placed.Sites.rowsOf_uniform`. -/
theorem SitesH.rowsOf_uniform {k : ℕ} (ins : ℕ → Fin (Placed.circuitRows c hU).nIn → Fin k) (one : Fin k)
    (hone : ∀ u c, ins u c ≠ one) : si.RowsOf (Placed.uniform k (Placed.circuitRows c hU) ins one hone n) :=
  fun S R u h => (Placed.rows_uniform k _ ins one hone n u).trans ((si.acc S R u h).rows_eq (hparse S R u h) hU).symm

include hparse in
/-- W6 for the circuit's program on any input gates, from the parse. -/
theorem SitesH.rowsOf_circuitProg (B : ℕ) (src : Fin (Placed.circuitRows c hU).nIn → Option (Fin B)) :
    si.RowsOf (Placed.circuitProg c hU n B src) :=
  SitesH.rowsOf_uniform hU hparse (Placed.insOf n B src) (Placed.oneGate n B) (Placed.insOf_ne_one n B src)

include hparse in
/-- W6 for `keyProg`, from the parse. -/
theorem SitesH.rowsOf_keyProg : si.RowsOf (Placed.keyProg c hU n) :=
  SitesH.rowsOf_circuitProg hU hparse _ _

variable (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

/-- **`aliased` at each drawn unit's site, from Δ's copies** (as `Placed.Sites.aliased_keyProg`): in the layout of the
hidden-output table the unit is drawn into, the program's own instance of the unit puts on one gate only inputs with one
source key (`AcceptedH.aliased_of_key`). -/
theorem SitesH.aliased_keyProg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (si.acc S R u h).L.Aliased (si.acc S R u h).regs
      (SitesH.rowsOf_keyProg hU hparse S R u h ▸ Const.Prog.instC (Placed.keyProg c hU n) outs u)
      (Fin.cast (si.acc S R u h).nbl_eq.symm (si.o S R u h)) (si.hvu S R u h) := by
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have e := congrArg Rows.nIn (SitesH.rowsOf_keyProg hU hparse S R u h)
  refine (si.acc S R u h).aliased_of_key _ _ (si.hvu S R u h) ?_
  intro x x' hw hx hx'
  have hw' := ((Placed.wire_cast (SitesH.rowsOf_keyProg hU hparse S R u h)
    (Const.Prog.instC (Placed.keyProg c hU n) outs u) x).symm.trans hw).trans
    (Placed.wire_cast (SitesH.rowsOf_keyProg hU hparse S R u h) (Const.Prog.instC (Placed.keyProg c hU n) outs u) x')
  have hk := Placed.keyProg_wire_key hU outs u _ _ (lt_of_lt_of_eq hx e.symm) (lt_of_lt_of_eq hx' e.symm) hw'
  exact (congrArg (fun d => Aliased.srcKey d x.val = Aliased.srcKey d x'.val) hc).mpr hk

/-- **`ProgPlaces` for the circuit's program on Δ's source keys at hidden-output sites**, from the sites and the parse
alone: each drawn unit placed at its template instance in its table's accepted statement (W6 from the parse), its
instance the program's own (`Const.Prog.instC`), aliased as Δ copies (`SitesH.aliased_keyProg`). -/
noncomputable def SitesH.progPlaces_keyProg : ProgPlaces (Placed.keyProg c hU n) outs plan tab :=
  si.progPlaces_of_layout (SitesH.rowsOf_keyProg hU hparse) outs (Const.Prog.instC _ outs)
    (SitesH.aliased_keyProg hU hparse outs)

/-- `progPlaces_keyProg`'s instance is the program's own. -/
theorem SitesH.progPlaces_keyProg_inst (u : Fin n) :
    (SitesH.progPlaces_keyProg hU hparse outs).inst u = Const.Prog.instC (Placed.keyProg c hU n) outs u :=
  rfl

/-- **`hConst` at `SitesH.progPlaces_keyProg`**: only a unit's constant column sits on the one gate. -/
theorem constCols_keyProgH :
    ((Placed.keyProg c hU n).partition outs).ConstCols plan tab (SitesH.progPlaces_keyProg hU hparse outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  Const.constCols_of_instC _ (Zero.oneGate_uniform n) (SitesH.progPlaces_keyProg_inst hU hparse outs)

end keyed

/-! ## `hZero` at the circuit's program -/

section circuit

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {k : ℕ} {c : Flock.Circuit} {hU : Built c.unit} {ins : ℕ → Fin (Placed.circuitRows c hU).nIn → Fin k}
  {one : Fin k} {hone : ∀ u x, ins u x ≠ one}

/-- **`hZero` for the circuit's program at hidden-output sites**, as `Zero.zeroCols_uniform`: every column on the zero
gate carries 0 in every satisfying witness, when every input the program wires to the zero gate reads the zero. -/
theorem zeroCols_uniformH (si : SitesH plan tab)
    (hrows : si.RowsOf (Placed.uniform k (Placed.circuitRows c hU) ins one hone n))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    {outs : Finset (Fin (Placed.gates k (Placed.circuitRows c hU) n))}
    (pp : ProgPlaces (Placed.uniform k (Placed.circuitRows c hU) ins one hone n) outs plan tab)
    (hinst : ∀ u, pp.inst u = Const.Prog.instC _ outs u) (ho : ∀ S R u h, pp.o S R u h = si.o S R u h)
    (hcol : ∀ S R u h, pp.col S R u h = si.col hrows S R u h)
    {zer : Fin k} (hz1 : zer ≠ one) (hzer : ∀ (u : Fin n) x, ins u.val x = zer → ZeroIn c x.val) :
    ((Placed.uniform k (Placed.circuitRows c hU) ins one hone n).partition outs).ZeroCols plan tab pp.derived.place
      {Fin.castLE (Placed.le_gates k _ n) zer} := by
  refine Zero.Prog.zeroCols_singleton pp (one := Fin.castLE (Placed.le_gates k _ n) one)
    (Zero.unit_uniform n _ zer.isLt) (fun he => hz1 (Fin.castLE_injective _ he)) (fun u => ?_) ?_
  · rw [hinst u, Const.Prog.instC_wire_one]
    exact Zero.oneGate_uniform n u
  · intro S R u h x hx hw z hsat
    rw [hinst u, Const.Prog.instC_wire_input _ _ _ x hx] at hw
    have hx' : x.val < (Placed.circuitRows c hU).nIn :=
      lt_of_lt_of_eq hx (congrArg Rows.nIn (Placed.rows_uniform _ _ _ _ _ n u))
    have hv := congrArg Fin.val hw
    rw [Zero.insAt_uniform n u x.val hx'] at hv
    have hzi : ZeroIn c x.val := hzer u ⟨x.val, hx'⟩ (Fin.ext hv)
    rw [ho, hcol]
    have hxa : x.val < (si.acc S R u h).rows.nIn := lt_of_lt_of_eq hx (congrArg Rows.nIn (hrows S R u h))
    have hza : ZeroIn (si.acc S R u h).st.c x.val := by
      rw [(si.acc S R u h).c_eq (hparse S R u h)]
      exact hzi
    exact (si.acc S R u h).zero_block (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) x) hxa hza hsat _

/-- **`hZero` for the circuit's program on source-indexed input gates**, as `Zero.zeroCols_circuitProg`. -/
theorem zeroCols_circuitProgH {B : ℕ} {src : Fin (Placed.circuitRows c hU).nIn → Option (Fin B)}
    (si : SitesH plan tab) (hrows : si.RowsOf (Placed.circuitProg c hU n B src))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    {outs : Finset (Fin (Placed.gates (Placed.inGates n B) (Placed.circuitRows c hU) n))}
    (pp : ProgPlaces (Placed.circuitProg c hU n B src) outs plan tab)
    (hinst : ∀ u, pp.inst u = Const.Prog.instC _ outs u) (ho : ∀ S R u h, pp.o S R u h = si.o S R u h)
    (hcol : ∀ S R u h, pp.col S R u h = si.col hrows S R u h) (hsrc : ∀ x, src x = none → ZeroIn c x.val) :
    ((Placed.circuitProg c hU n B src).partition outs).ZeroCols plan tab pp.derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n B)} :=
  zeroCols_uniformH si hrows hparse pp hinst ho hcol
    (fun h => by simp [Placed.zeroGate, Placed.oneGate, Fin.ext_iff] at h)
    fun u x hx => hsrc x ((Placed.insOf_eq_zero_iff u.isLt x).1 hx)

/-- **`hZero` at `SitesH.progPlaces_keyProg`**, as `Zero.zeroCols_keyProg`: every column on the zero gate carries 0 in
every satisfying witness. -/
theorem zeroCols_keyProgH (hU : Built c.unit) {si : SitesH plan tab}
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n))) :
    ((Placed.keyProg c hU n).partition outs).ZeroCols plan tab
      (SitesH.progPlaces_keyProg hU hparse outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_circuitProgH si (SitesH.rowsOf_keyProg hU hparse) hparse _ (fun _ => rfl) (fun _ _ _ _ => rfl)
    (fun _ _ _ _ => rfl) fun x hx => by
      have h1 : x.val < c.upv * min (c.unit.inWords * WORD) c.unit.constPos := x.isLt
      exact (Zero.srcKey_zero_iff (lt_of_lt_of_le h1 (Nat.mul_le_mul_left _ (Nat.min_le_left _ _)))).1
        ((Aliased.keyIdx_eq_none x).1 hx)

end circuit

end FlockSoundness.Discharge.HiddenExec
