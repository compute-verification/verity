import Proofs.Flock.Soundness.Discharge.HiddenExec.Placed
import Proofs.Flock.Soundness.Discharge.HiddenExec.CopiesP
import Proofs.Flock.Soundness.Discharge.Placed.KeyedP

/-!
# `ProgPlaces` and `hConst` from `Stmt.setupHidden`'s accept at public inputs

`Placed.lean` at the verifier's public inputs (`Hidden.SpecP`):

* `AcceptedHP T`: a run of `Stmt.setupHidden` that accepted at public inputs `own`, with the circuit's public input ports
  `pi`, a layout `L` whose kept unit slot rows are its flat twin's (`UnitRowsFlatP`, which `unitRowsFlatP` gives), and
  regions `regs` holding the statement's, its block R1CS `L.model regs` `T`. Its aliasing comes from the keys with
  public inputs (`AcceptedHP.aliased_of_key`, through `aliased_of_keyP`).
* `SitesHP plan tab`: each drawn unit's acceptance, template instance and block, as `SitesH`.
* `SitesHP.progPlaces_keyProgP`: `ProgPlaces` for the circuit's program on the keys with public inputs
  (`Placed.keyProgP`), from the sites and the parse alone; `constCols_keyProgHP` and `zeroCols_keyProgHP` are its
  `hConst` and `hZero` (a stacked input on the zero gate is off a public leaf, `AcceptedHP.zero_block`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3
open Model Audit
open FlockSoundness.Discharge.Zero (ZeroIn)

/-! ## An accepted hidden-output statement at public inputs -/

/-- **A stacked input off a public leaf that reads the zero carries 0** at public inputs, at template instance `g`'s
place, in every block of a satisfying witness (`stack_zero_blockH`, through `copyPos_blockHP`). -/
theorem stack_zero_blockHP {tags : Tags} {st : Stmt} {pi : HmIn.Ports} (L : Layout st)
    (Lf : Layout (Hidden.flatTwin st)) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (F : Aliased.UnitFacts tags (Hidden.flatTwin st)) (hp : HmIn.parse st.c = .ok pi) (hR : UnitRowsFlatP st pi)
    {g : ℕ} (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) (hneg : Aliased.pubLeaf st.c pi x.val < 0)
    (hz : ZeroIn st.c x.val) {z : Model.Witness (L.model regs).m} (hsat : (L.model regs).Satisfies z)
    (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (L.col regs hg x) = 0 := by
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    have : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
    rwa [L.unitRows_nIn] at this
  rw [copyPos_blockHP L Lf regs F hp F.check hR hg x hx hneg hsat o]
  obtain ⟨s, hs, he⟩ := Zero.copyPos_zero Lf regs F hg hx' hz
  rw [he]
  exact zeroPos_blockH L Lf regs F hs hsat o

/-- **An accepted untyped hidden-output statement at public inputs whose block R1CS is `T`**: `Stmt.setupHidden`
accepted at the verifier's public inputs `own`, returning `st` with public input ports `pi` (`s`); `L` is a layout of
`st` whose kept unit slot rows are its flat twin's (`hR`), and `T` is `L.model` at regions `regs` holding `st`'s. -/
structure AcceptedHP (T : Model.Statement) where
  tags : Tags
  cf : ByteArray
  pf : ByteArray
  coins : Coins
  tables : List (String × ByteArray)
  draw : Option Lean.Json
  partition : Option Lean.Json
  program : Option ByteArray
  own : Option ByteArray
  st : Stmt
  ht : tags.typed = false
  h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st
  pi : HmIn.Ports
  s : Hidden.SpecP tags cf pf coins tables draw own pi st
  L : Layout st
  hR : UnitRowsFlatP st pi
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs
  hT : L.model regs = T

namespace AcceptedHP

variable {T : Model.Statement} (a : AcceptedHP T)

/-- The flat twin's layout (`flat_layoutP`). -/
theorem flatLayout : Layout (Hidden.flatTwin a.st) := flat_layoutP a.ht a.s

/-- The flat twin's unit facts (`flat_unitFactsP`). -/
theorem unitFacts : Aliased.UnitFacts a.tags (Hidden.flatTwin a.st) := flat_unitFactsP a.ht a.s

/-- **The accepted statement's template instance rows**: the unit net's rows, stacked `units_per_vu` times. -/
noncomputable def rows : Rows := (unitRows a.st a.L.unit_built).stack a.st.c.upv

theorem kLog_eq : 2 ^ (a.L.model a.regs).kLog = 2 ^ T.kLog :=
  congrArg (fun S : Model.Statement => 2 ^ S.kLog) a.hT

theorem nbl_eq : 2 ^ (a.L.model a.regs).nbl = 2 ^ T.nbl :=
  congrArg (fun S : Model.Statement => 2 ^ S.nbl) a.hT

/-- **Template instance `g`'s columns in the block**, as `AcceptedH.col`. -/
noncomputable def col {g : ℕ} (hg : g < a.st.c.g) (c : Fin a.rows.w) : Fin (2 ^ T.kLog) :=
  Fin.cast a.kLog_eq (a.L.col a.regs hg c)

/-- **The accepted statement places its template instance rows at every template instance**, in every block. -/
theorem placement {g : ℕ} (hg : g < a.st.c.g) : Placement T a.rows (a.col hg) :=
  Placed.Accepted.placement_cast a.hT (a.L.placement a.regs hg)

/-- An accepted statement's circuit is the one its circuit file parses to (`parse_untyped`). -/
theorem c_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) : a.st.c = c := by
  have := (Hidden.parse_untyped a.ht a.s.parse).1
  rw [hp] at this
  exact (Except.ok.inj this).symm

/-- An accepted statement's public input ports are its circuit's (`HmIn.parse`). -/
theorem pi_eq {c : Flock.Circuit} {pi : HmIn.Ports} (hp : HmRow.parse a.tags a.cf a.tables = .ok c)
    (hpi : HmIn.parse c = .ok pi) : a.pi = pi := by
  have := a.s.ports
  rw [a.c_eq hp, hpi] at this
  exact (Except.ok.inj this).symm

/-- **The template instance rows are the parsed circuit's** (W6 on the untyped path, by definition). -/
theorem rows_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) (hU : Built c.unit) :
    a.rows = Placed.circuitRows c hU := by
  have hc := a.c_eq hp
  unfold rows Placed.circuitRows unitRows
  rw [Placed.ofNet_eq (Placed.lunit_of_parse hp hc) a.L.unit_built hU, hc]

/-- **`Layout.Aliased` in `ProgPlaces`' form**, as `AcceptedH.aliased_of_layout`. -/
theorem aliased_of_layout {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (hR : R = a.rows) (inst : P.IsRowsUnit u R) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hal : a.L.Aliased a.regs (hR ▸ inst) (Fin.cast a.nbl_eq.symm o) hg) :
    ∀ c c', inst.wire c = inst.wire c' → c.val < R.nIn → c'.val < R.nIn → ∀ z : Model.Witness T.m, T.Satisfies z →
      T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c)) =
        T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c')) := by
  subst hR
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, pi, s, L, hR, regs, hregs, hT⟩ := a
  subst hT
  exact hal

/-- **Aliasing from keys with public inputs**: an instance that puts template instance `g`'s stacked unit inputs on one
gate only when they have one key (`Aliased.srcKeyP`) is aliased in the accepted statement (`aliased_of_keyP`). -/
theorem aliased_of_key {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows a.st a.L.unit_built).stack a.st.c.upv))
    (o : Fin (2 ^ (a.L.model a.regs).nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hk : ∀ c c', inst.wire c = inst.wire c' →
      c.val < ((unitRows a.st a.L.unit_built).stack a.st.c.upv).nIn →
      c'.val < ((unitRows a.st a.L.unit_built).stack a.st.c.upv).nIn →
        Aliased.srcKeyP a.st.c a.pi c.val = Aliased.srcKeyP a.st.c a.pi c'.val) :
    a.L.Aliased a.regs inst o hg := by
  obtain ⟨hpa, -⟩ := Hidden.parse_untyped a.ht a.s.parse
  have P := parse_facts hpa
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  obtain ⟨ro, ri, -, hRi, hr⟩ := a.s.regions
  exact aliased_of_keyP a.L a.flatLayout a.regs a.unitFacts a.s.ports C a.hR a.hregs hRi
    (fun r hr' => by rw [hr, Array.toList_append]; exact List.mem_append_right _ hr') inst hg hk o

/-- **A template instance's stacked input off a public leaf that reads the zero carries 0**, at the template instance's
columns, in every block of a satisfying witness (`stack_zero_blockHP`). -/
theorem zero_block {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w) (hx : x.val < a.rows.nIn)
    (hneg : Aliased.pubLeaf a.st.c a.pi x.val < 0) (hz : ZeroIn a.st.c x.val) {z : Model.Witness T.m}
    (hsat : T.Satisfies z) (o : Fin (2 ^ T.nbl)) :
    T.block z o (a.col hg x) = 0 := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, pi, s, L, hR, regs, hregs, hT⟩ := a
  subst hT
  exact stack_zero_blockHP L (flat_layoutP ht s) regs (flat_unitFactsP ht s) s.ports hR hg x hx hneg hz hsat o

end AcceptedHP

/-! ## Each drawn unit's place -/

section SitesP

variable {N n : ℕ} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each drawn unit's site at public inputs**, as `SitesH`: the acceptance at public inputs of the statement of the
table unit `u` is drawn into (`acc`), and the unit's template instance (`vu`) and block (`o`) there. -/
structure SitesHP where
  acc : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → AcceptedHP (Partition.tabOf plan tab S R u).S
  vu : ∀ (S : Finset (Fin n)) (_ : Reg) (u : Fin n), u ∈ S → ℕ
  hvu : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), vu S R u h < (acc S R u h).st.c.g
  o : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl)

variable {plan tab}

/-- **W6 at the sites**: each drawn unit's rows in the program are the template instance rows its table's acceptance
placed. -/
def SitesHP.RowsOf (si : SitesHP plan tab) (p : Prog N n) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), p.rows u = (si.acc S R u h).rows

variable {p : Prog N n} (si : SitesHP plan tab) (hrows : si.RowsOf p)

/-- A drawn unit's columns in its table's block: its template instance's (`AcceptedHP.col`). -/
noncomputable def SitesHP.col (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin (p.rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun c => (si.acc S R u h).col (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) c)

/-- **W6's placement** at public inputs. -/
theorem SitesHP.placed (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Placement (Partition.tabOf plan tab S R u).S (p.rows u) (si.col hrows S R u h) :=
  Placed.placement_rows (hrows S R u h) ((si.acc S R u h).placement (si.hvu S R u h))

/-- **`ProgPlaces` from setupHidden's accept at public inputs**, its aliasing from `Layout.Aliased` at each drawn
unit's template instance and block. -/
noncomputable def SitesHP.progPlaces_of_layout (outs : Finset (Fin N))
    (inst : ∀ u, (p.partition outs).IsRowsUnit u (p.rows u))
    (hal : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), (si.acc S R u h).L.Aliased (si.acc S R u h).regs
      (hrows S R u h ▸ inst u) (Fin.cast (si.acc S R u h).nbl_eq.symm (si.o S R u h)) (si.hvu S R u h)) :
    ProgPlaces p outs plan tab where
  inst := inst
  o := si.o
  col := si.col hrows
  placed S R u h := (si.placed hrows S R u h).sat
  aliased S R u h :=
    (si.acc S R u h).aliased_of_layout (hrows S R u h) (inst u) (si.o S R u h) (si.hvu S R u h) (hal S R u h)

end SitesP

/-! ## The circuit's program on the keys with public inputs -/

section keyedP

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {c : Flock.Circuit} (hU : Built c.unit) {pi : HmIn.Ports} {si : SitesHP plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (hpi : HmIn.parse c = .ok pi)

include hparse in
/-- W6 for the circuit's program on any input gates, from the parse. -/
theorem SitesHP.rowsOf_circuitProg (B : ℕ) (src : Fin (Placed.circuitRows c hU).nIn → Option (Fin B)) :
    si.RowsOf (Placed.circuitProg c hU n B src) :=
  fun S R u h => (Placed.rows_uniform _ (Placed.circuitRows c hU) (Placed.insOf n B src) (Placed.oneGate n B)
    (Placed.insOf_ne_one n B src) n u).trans ((si.acc S R u h).rows_eq (hparse S R u h) hU).symm

include hparse in
/-- W6 for `keyProgP`, from the parse. -/
theorem SitesHP.rowsOf_keyProgP : si.RowsOf (Placed.keyProgP c pi hU n) :=
  SitesHP.rowsOf_circuitProg hU hparse _ _

variable (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

include hpi in
/-- **`aliased` at each drawn unit's site at public inputs**: in the layout of the table the unit is drawn into, the
program's own instance of the unit puts on one gate only inputs with one key with public inputs
(`Placed.keyProgP_wire_key`, `AcceptedHP.aliased_of_key`). -/
theorem SitesHP.aliased_keyProgP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (si.acc S R u h).L.Aliased (si.acc S R u h).regs
      (SitesHP.rowsOf_keyProgP hU hparse S R u h ▸ Const.Prog.instC (Placed.keyProgP c pi hU n) outs u)
      (Fin.cast (si.acc S R u h).nbl_eq.symm (si.o S R u h)) (si.hvu S R u h) := by
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have hq := (si.acc S R u h).pi_eq (hparse S R u h) hpi
  have e := congrArg Rows.nIn (SitesHP.rowsOf_keyProgP (pi := pi) hU hparse S R u h)
  refine (si.acc S R u h).aliased_of_key _ _ (si.hvu S R u h) ?_
  intro x x' hw hx hx'
  have hw' := ((Placed.wire_cast (SitesHP.rowsOf_keyProgP (pi := pi) hU hparse S R u h)
    (Const.Prog.instC (Placed.keyProgP c pi hU n) outs u) x).symm.trans hw).trans
    (Placed.wire_cast (SitesHP.rowsOf_keyProgP (pi := pi) hU hparse S R u h)
      (Const.Prog.instC (Placed.keyProgP c pi hU n) outs u) x')
  have hk := Placed.keyProgP_wire_key hU outs u _ _ (lt_of_lt_of_eq hx e.symm) (lt_of_lt_of_eq hx' e.symm) hw'
  exact (congrArg₂ (fun d q => Aliased.srcKeyP d q x.val = Aliased.srcKeyP d q x'.val) hc hq).mpr hk

/-- **`ProgPlaces` for the circuit's program on the keys with public inputs at hidden-output sites**, from the sites
and the parse alone: each drawn unit placed at its template instance in its table's accepted statement, its instance the
program's own (`Const.Prog.instC`), aliased (`SitesHP.aliased_keyProgP`). -/
noncomputable def SitesHP.progPlaces_keyProgP : ProgPlaces (Placed.keyProgP c pi hU n) outs plan tab :=
  si.progPlaces_of_layout (SitesHP.rowsOf_keyProgP (pi := pi) hU hparse) outs (Const.Prog.instC _ outs)
    (SitesHP.aliased_keyProgP hU hparse hpi outs)

/-- `progPlaces_keyProgP`'s instance is the program's own. -/
theorem SitesHP.progPlaces_keyProgP_inst (u : Fin n) :
    (SitesHP.progPlaces_keyProgP hU hparse hpi outs).inst u = Const.Prog.instC (Placed.keyProgP c pi hU n) outs u :=
  rfl

/-- **`hConst` at `SitesHP.progPlaces_keyProgP`**: only a unit's constant column sits on the one gate. -/
theorem constCols_keyProgHP :
    ((Placed.keyProgP c pi hU n).partition outs).ConstCols plan tab
      (SitesHP.progPlaces_keyProgP hU hparse hpi outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.oneGate n (Placed.circuitRows c hU).nIn)} :=
  Const.constCols_of_instC _ (Zero.oneGate_uniform n) (SitesHP.progPlaces_keyProgP_inst hU hparse hpi outs)

end keyedP


/-! ## `hZero` at the circuit's program on the keys with public inputs -/

section circuitP

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {k : ℕ} {c : Flock.Circuit} {hU : Built c.unit} {pi : HmIn.Ports} {ins : ℕ → Fin (Placed.circuitRows c hU).nIn → Fin k}
  {one : Fin k} {hone : ∀ u x, ins u x ≠ one}

/-- **`hZero` for the circuit's program at hidden-output sites at public inputs**, as `zeroCols_uniformH`: every column
on the zero gate carries 0 in every satisfying witness, when every input the program wires to the zero gate is off a
public leaf and reads the zero. -/
theorem zeroCols_uniformHP (si : SitesHP plan tab)
    (hrows : si.RowsOf (Placed.uniform k (Placed.circuitRows c hU) ins one hone n))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (hpi : HmIn.parse c = .ok pi)
    {outs : Finset (Fin (Placed.gates k (Placed.circuitRows c hU) n))}
    (pp : ProgPlaces (Placed.uniform k (Placed.circuitRows c hU) ins one hone n) outs plan tab)
    (hinst : ∀ u, pp.inst u = Const.Prog.instC _ outs u) (ho : ∀ S R u h, pp.o S R u h = si.o S R u h)
    (hcol : ∀ S R u h, pp.col S R u h = si.col hrows S R u h)
    {zer : Fin k} (hz1 : zer ≠ one)
    (hzer : ∀ (u : Fin n) x, ins u.val x = zer → ZeroIn c x.val ∧ Aliased.pubLeaf c pi x.val < 0) :
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
    obtain ⟨hzi, hni⟩ := hzer u ⟨x.val, hx'⟩ (Fin.ext hv)
    rw [ho, hcol]
    have hxa : x.val < (si.acc S R u h).rows.nIn := lt_of_lt_of_eq hx (congrArg Rows.nIn (hrows S R u h))
    have hc := (si.acc S R u h).c_eq (hparse S R u h)
    have hq := (si.acc S R u h).pi_eq (hparse S R u h) hpi
    have hza : ZeroIn (si.acc S R u h).st.c x.val := by rw [hc]; exact hzi
    have hna : Aliased.pubLeaf (si.acc S R u h).st.c (si.acc S R u h).pi x.val < 0 :=
      (congrArg₂ (fun d q => Aliased.pubLeaf d q x.val < 0) hc hq).mpr hni
    exact (si.acc S R u h).zero_block (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) x) hxa hna hza
      hsat _

/-- **`hZero` for the circuit's program on source-indexed input gates at public inputs**, as
`zeroCols_circuitProgH`. -/
theorem zeroCols_circuitProgHP {B : ℕ} {src : Fin (Placed.circuitRows c hU).nIn → Option (Fin B)}
    (si : SitesHP plan tab) (hrows : si.RowsOf (Placed.circuitProg c hU n B src))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (hpi : HmIn.parse c = .ok pi)
    {outs : Finset (Fin (Placed.gates (Placed.inGates n B) (Placed.circuitRows c hU) n))}
    (pp : ProgPlaces (Placed.circuitProg c hU n B src) outs plan tab)
    (hinst : ∀ u, pp.inst u = Const.Prog.instC _ outs u) (ho : ∀ S R u h, pp.o S R u h = si.o S R u h)
    (hcol : ∀ S R u h, pp.col S R u h = si.col hrows S R u h)
    (hsrc : ∀ x, src x = none → ZeroIn c x.val ∧ Aliased.pubLeaf c pi x.val < 0) :
    ((Placed.circuitProg c hU n B src).partition outs).ZeroCols plan tab pp.derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n B)} :=
  zeroCols_uniformHP si hrows hparse hpi pp hinst ho hcol
    (fun h => by simp [Placed.zeroGate, Placed.oneGate, Fin.ext_iff] at h)
    fun u x hx => hsrc x ((Placed.insOf_eq_zero_iff u.isLt x).1 hx)

/-- **`hZero` at `SitesHP.progPlaces_keyProgP`**: every column on the zero gate carries 0 in every satisfying witness,
since an input on it is off a public leaf and reads the zero (`Aliased.srcKeyP_eq_zero`). -/
theorem zeroCols_keyProgHP (hU : Built c.unit) {si : SitesHP plan tab}
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (hpi : HmIn.parse c = .ok pi)
    (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n))) :
    ((Placed.keyProgP c pi hU n).partition outs).ZeroCols plan tab
      (SitesHP.progPlaces_keyProgP hU hparse hpi outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_circuitProgHP si (SitesHP.rowsOf_keyProgP (pi := pi) hU hparse) hparse hpi _ (fun _ => rfl)
    (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl) fun x hx => by
      have h1 : x.val < c.upv * min (c.unit.inWords * WORD) c.unit.constPos := x.isLt
      obtain ⟨hneg, hk⟩ := Aliased.srcKeyP_eq_zero.1 ((Aliased.keyIdxP_eq_none x).1 hx)
      exact ⟨(Zero.srcKey_zero_iff (lt_of_lt_of_le h1 (Nat.mul_le_mul_left _ (Nat.min_le_left _ _)))).1 hk, hneg⟩

end circuitP

end FlockSoundness.Discharge.HiddenExec
