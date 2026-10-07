import Proofs.Flock.Soundness.Discharge.ZkPubIn.Flat
import Proofs.Flock.Soundness.Discharge.ZkPubIn.Tables
import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites

set_option Elab.async false

/-!
# The sites and places of `keyProgP` in tables with the verifier's public inputs

HiddenExec's `AcceptedH`, `SitesH`, `TableH.acc` and `SitesH.progPlaces_keyProg` (`HiddenExec/Placed.lean`,
`HiddenExec/Sites.lean`) with `Stmt.setupHidden … (some own)`, at the program with its public gates (`keyProgP`):

* `AcceptedHP T`: a run of `Stmt.setupHidden … (some own)` that accepted, with a layout of its statement, its unit
  slots' rows its flat twin's off the public leaves (`UnitRowsFlatP`), and its block R1CS `T`; `AcceptedHP.col`,
  `.placement` as `AcceptedH`'s, and its aliasing from keys that put each public leaf's bit alone on its gate
  (`AcceptedHP.aliased_of_keyP`).
* `SitesHP plan tab`: each drawn unit's acceptance, VU and block; `TableHP.acc`: a table's acceptance.
* `SitesHP.progPlaces_keyProgP`: `ProgPlaces` for `keyProgP` at the circuit's public input ports, its aliasing
  `SitesHP.aliased_keyProgP` (a public leaf's bit is on a gate of its own, `keyIdxP_eq`; any other input's gate is a Δ
  key's, whose copies `HmIn.delta` keeps).
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model Refine Audit HiddenExec
open FlockSoundness.Discharge.Exec (Inputs)

/-! ## An accepted hidden-output statement with the verifier's public inputs -/

/-- **An accepted untyped hidden-output statement with the verifier's public inputs whose block R1CS is `T`**:
`AcceptedH` with `Stmt.setupHidden … (some own)`, its unit slots' rows its flat twin's at every row that is no public
leaf's bit (`UnitRowsFlatP`: at a public leaf's bit `HmIn.delta` drops the twin's copy of the forced zero). -/
structure AcceptedHP (T : Model.Statement) where
  tags : Tags
  cf : ByteArray
  pf : ByteArray
  coins : Coins
  tables : List (String × ByteArray)
  draw : Option Lean.Json
  partition : Option Lean.Json
  program : Option ByteArray
  own : ByteArray
  st : Stmt
  ht : tags.typed = false
  h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st
  L : FlockSoundness.Layout st
  hR : UnitRowsFlatP st
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hT : L.model regs = T

namespace AcceptedHP

variable {T : Model.Statement} (a : AcceptedHP T)

/-- The flat twin's layout (`flat_layoutP`). -/
theorem flatLayout : FlockSoundness.Layout (Hidden.flatTwin a.st) := flat_layoutP a.ht a.h

/-- The flat twin's unit facts (`flat_unitFactsP`). -/
theorem unitFacts : Aliased.UnitFacts a.tags (Hidden.flatTwin a.st) := flat_unitFactsP a.ht a.h

/-- The public leaf maps' layout at the circuit's public input ports. -/
theorem leavesPublic {pi : HmIn.Ports} (hpi : HmIn.parse a.st.c = .ok pi) : LeavesPublic a.st.c pi := by
  obtain ⟨pi₀, s⟩ := setupHidden_specP a.h
  obtain rfl : pi₀ = pi := Except.ok.inj (s.inParse.symm.trans hpi)
  exact s.leavesPublic

/-- **The accepted statement's VU rows**: the unit net's rows, stacked `units_per_vu` times. -/
noncomputable def rows : Rows := (unitRows a.st a.L.unit_built).stack a.st.c.upv

theorem kLog_eq : 2 ^ (a.L.model a.regs).kLog = 2 ^ T.kLog :=
  congrArg (fun S : Model.Statement => 2 ^ S.kLog) a.hT

theorem nbl_eq : 2 ^ (a.L.model a.regs).nbl = 2 ^ T.nbl :=
  congrArg (fun S : Model.Statement => 2 ^ S.nbl) a.hT

/-- **VU `g`'s columns in the block** (`AcceptedH.col`). -/
noncomputable def col {g : ℕ} (hg : g < a.st.c.g) (c : Fin a.rows.w) : Fin (2 ^ T.kLog) :=
  Fin.cast a.kLog_eq
    (stackPlace (a.L.model a.regs) (unitRows a.st a.L.unit_built) a.st.c.upv
      (fun u => a.st.c.unitRange.slot (g * a.st.c.upv + u)) (a.L.hb hg) c)

/-- **The accepted statement places its VU rows at every VU** (`Layout.placement`), in every block. -/
theorem placement {g : ℕ} (hg : g < a.st.c.g) : Placement T a.rows (a.col hg) :=
  Placed.Accepted.placement_cast a.hT (a.L.placement a.regs hg)

/-- An accepted statement's circuit is the one its circuit file parses to. -/
theorem c_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) : a.st.c = c := by
  obtain ⟨_, hs⟩ := setupHidden_specP a.h
  have := (Hidden.parse_untyped a.ht hs.parse).1
  rw [hp] at this
  exact (Except.ok.inj this).symm

/-- **The VU rows are the parsed circuit's**. -/
theorem rows_eq {c : Flock.Circuit} (hp : HmRow.parse a.tags a.cf a.tables = .ok c) (hU : Built c.unit) :
    a.rows = Placed.circuitRows c hU := by
  have hc := a.c_eq hp
  unfold rows Placed.circuitRows unitRows
  rw [Placed.ofNet_eq hc a.L.unit_built hU, hc]

/-- **`Layout.Aliased` in `ProgPlaces`' form** (`AcceptedH.aliased_of_layout`). -/
theorem aliased_of_layout {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (hR : R = a.rows) (inst : P.IsRowsUnit u R) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hal : a.L.Aliased a.regs (hR ▸ inst) (Fin.cast a.nbl_eq.symm o) hg) :
    ∀ c c', inst.wire c = inst.wire c' → c.val < R.nIn → c'.val < R.nIn → ∀ z : Model.Witness T.m, T.Satisfies z →
      T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c)) =
        T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c')) := by
  subst hR
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, own, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  exact hal

/-- **Aliasing from keys, with the public inputs**: an instance that puts two of VU `g`'s stacked unit inputs on one
gate only when they are one input, or neither a public leaf's bit and both of one key (`Aliased.srcKey`), is aliased
in the accepted statement (`aliased_of_copiesHP`). -/
theorem aliased_of_keyP {pi : HmIn.Ports} (hpi : HmIn.parse a.st.c = .ok pi) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n} (inst : P.IsRowsUnit u ((unitRows a.st a.L.unit_built).stack a.st.c.upv))
    (o : Fin (2 ^ (a.L.model a.regs).nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hk : ∀ c c', inst.wire c = inst.wire c' →
      c.val < ((unitRows a.st a.L.unit_built).stack a.st.c.upv).nIn →
      c'.val < ((unitRows a.st a.L.unit_built).stack a.st.c.upv).nIn →
        c = c' ∨ (pubLeaf a.st.c pi c.val = none ∧ pubLeaf a.st.c pi c'.val = none ∧
          Aliased.srcKey a.st.c c.val = Aliased.srcKey a.st.c c'.val)) :
    a.L.Aliased a.regs inst o hg :=
  aliased_of_copiesHP a.L a.flatLayout a.regs a.unitFacts a.hR hpi (a.leavesPublic hpi) inst hg
    (Aliased.copies_of_key a.flatLayout a.regs a.unitFacts inst hg fun c c' hw hc hc' => by
      rcases hk c c' hw hc hc' with rfl | ⟨-, -, he⟩
      · rfl
      · exact he)
    (fun c c' hw hc hc' hne => by
      rcases hk c c' hw hc hc' with he | ⟨h1, h2, -⟩
      · exact absurd he hne
      · exact ⟨h1, h2⟩) o

end AcceptedHP

/-! ## Each drawn unit's place -/

section Sites

variable {N n : ℕ} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- **Each drawn unit's site** (`SitesH` with `AcceptedHP`). -/
structure SitesHP where
  acc : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → AcceptedHP (Partition.tabOf plan tab S R u).S
  vu : ∀ (S : Finset (Fin n)) (_ : Reg) (u : Fin n), u ∈ S → ℕ
  hvu : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), vu S R u h < (acc S R u h).st.c.g
  o : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl)

variable {plan tab}

/-- **W6 at the sites**: each drawn unit's rows in the program are the VU rows its table's acceptance placed. -/
def SitesHP.RowsOf (si : SitesHP plan tab) (p : Prog N n) : Prop :=
  ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), p.rows u = (si.acc S R u h).rows

variable {p : Prog N n} (si : SitesHP plan tab) (hrows : si.RowsOf p)

/-- A drawn unit's columns in its table's block: its VU's (`AcceptedHP.col`). -/
noncomputable def SitesHP.col (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin (p.rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun c => (si.acc S R u h).col (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) c)

/-- **W6's placement**: the statement of the table each drawn unit is drawn into places the unit's rows at its VU. -/
theorem SitesHP.placed (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Placement (Partition.tabOf plan tab S R u).S (p.rows u) (si.col hrows S R u h) :=
  Placed.placement_rows (hrows S R u h) ((si.acc S R u h).placement (si.hvu S R u h))

end Sites

/-! ## `keyProgP`'s places -/

section keyed

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {c : Flock.Circuit} (hU : Built c.unit) {si : SitesHP plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (pi : HmIn.Ports)

/-- **`keyIdxP` at an input that is no public leaf's bit**: no gate exactly at the zero key, and otherwise an input
with its key that is no public leaf's bit either. -/
theorem keyIdxP_none (c : Flock.Circuit) (pi : HmIn.Ports) (m : ℕ) (x : Fin m) (hx : pubLeaf c pi x.val = none) :
    (keyIdxP c pi m x = none ↔ Aliased.srcKey c x.val = .zero) ∧
      ∀ i, keyIdxP c pi m x = some i → pubLeaf c pi i.val = none ∧ Aliased.srcKey c i.val = Aliased.srcKey c x.val := by
  unfold keyIdxP
  simp only [hx, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
  by_cases hz : Aliased.srcKey c x.val = .zero
  · simp only [hz, ↓reduceIte, reduceCtorEq, false_implies, implies_true, and_self]
  · simp only [hz, ↓reduceIte, iff_false]
    refine ⟨fun hn => ?_, fun i hi => ?_⟩
    · rw [List.find?_eq_none] at hn
      exact hn x (List.mem_finRange x) (by simp [hx])
    · have := List.find?_some hi
      simp only [Bool.and_eq_true, Option.isNone_iff_eq_none, decide_eq_true_eq] at this
      exact this

/-- **Two inputs on one `keyIdxP` gate are one input, or neither a public leaf's bit and both of one key.** -/
theorem keyIdxP_eq (c : Flock.Circuit) (pi : HmIn.Ports) (m : ℕ) (x x' : Fin m)
    (h : keyIdxP c pi m x = keyIdxP c pi m x') :
    x = x' ∨ (pubLeaf c pi x.val = none ∧ pubLeaf c pi x'.val = none ∧
      Aliased.srcKey c x.val = Aliased.srcKey c x'.val) := by
  have hpub : ∀ y : Fin m, (pubLeaf c pi y.val).isSome → keyIdxP c pi m y = some y := fun y hy => by
    unfold keyIdxP; simp [hy]
  cases hx : pubLeaf c pi x.val with
  | some v =>
    have e := (hpub x (by simp [hx])).symm.trans h
    cases hx' : pubLeaf c pi x'.val with
    | some v' => exact Or.inl (Option.some.inj (e.trans (hpub x' (by simp [hx']))))
    | none =>
      have := ((keyIdxP_none c pi m x' hx').2 x e.symm).1
      rw [hx] at this
      cases this
  | none =>
    cases hx' : pubLeaf c pi x'.val with
    | some v' =>
      have := ((keyIdxP_none c pi m x hx).2 x' (h.trans (hpub x' (by simp [hx'])))).1
      rw [hx'] at this
      cases this
    | none =>
      refine Or.inr ⟨rfl, rfl, ?_⟩
      have k1 := keyIdxP_none c pi m x hx
      have k2 := keyIdxP_none c pi m x' hx'
      cases hk : keyIdxP c pi m x with
      | none => rw [k1.1.1 hk, k2.1.1 (h ▸ hk)]
      | some i => rw [← (k1.2 i hk).2, ← (k2.2 i (h ▸ hk)).2]

/-- **Two inputs of one unit of `keyProgP` on one gate are one input, or neither a public leaf's bit and both of one
key**, in the program's own instance (`Placed.keyProg_wire_key`). -/
theorem keyProgP_wire (c : Flock.Circuit) (hU : Built c.unit) {n : ℕ} (pi : HmIn.Ports)
    (outs : Finset (Fin (Layout.keyN c hU n))) (u : Fin n)
    (x x' : Fin ((keyProgP c hU n pi).rows u).w) (hx : x.val < ((keyProgP c hU n pi).rows u).nIn)
    (hx' : x'.val < ((keyProgP c hU n pi).rows u).nIn)
    (hw : (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x =
      (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x') :
    x.val = x'.val ∨ (pubLeaf c pi x.val = none ∧ pubLeaf c pi x'.val = none ∧
      Aliased.srcKey c x.val = Aliased.srcKey c x'.val) := by
  have e := congrArg Rows.nIn (Placed.rows_circuitProg c hU n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)) u)
  have h1 : x.val < (Placed.circuitRows c hU).nIn := lt_of_lt_of_eq hx e
  have h2 : x'.val < (Placed.circuitRows c hU).nIn := lt_of_lt_of_eq hx' e
  rw [Const.Prog.instC_wire_input _ _ _ x hx, Const.Prog.instC_wire_input _ _ _ x' hx'] at hw
  have hv : (Const.Prog.insAt (keyProgP c hU n pi) u x.val).val =
      (Const.Prog.insAt (keyProgP c hU n pi) u x'.val).val := congrArg Fin.val hw
  have e1 : (Const.Prog.insAt (keyProgP c hU n pi) u x.val).val =
      (Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)) u.val ⟨x.val, h1⟩).val :=
    Placed.insAt_uniform n u x.val h1
  have e2 : (Const.Prog.insAt (keyProgP c hU n pi) u x'.val).val =
      (Placed.insOf n (Layout.keyB c hU) (keyIdxP c pi (Layout.keyB c hU)) u.val ⟨x'.val, h2⟩).val :=
    Placed.insAt_uniform n u x'.val h2
  rcases keyIdxP_eq c pi _ ⟨x.val, h1⟩ ⟨x'.val, h2⟩
    ((Placed.insOf_eq_iff u.isLt ⟨x.val, h1⟩ ⟨x'.val, h2⟩).1 (Fin.ext (e1.symm.trans (hv.trans e2)))) with he | he
  · exact Or.inl (Fin.mk.inj he)
  · exact Or.inr he

include hparse in
/-- W6 for `keyProgP`, from the parse. -/
theorem SitesHP.rowsOf_keyProgP : si.RowsOf (keyProgP c hU n pi) :=
  fun S R u h => (Placed.rows_circuitProg c hU n _ _ u).trans ((si.acc S R u h).rows_eq (hparse S R u h) hU).symm

variable (hpi : HmIn.parse c = .ok pi) (outs : Finset (Fin (Layout.keyN c hU n)))

include hparse hpi in
/-- **`aliased` at each drawn unit's site**: in a satisfying witness of the table a unit is drawn into, the program's
own instance of the unit puts on one gate only inputs that hold one value at the unit's VU and block: a public leaf's
bit is alone on its gate, and inputs with one Δ key, none a public leaf's bit, are one column or the zero
(`HmIn.delta` drops only a public leaf's copies), at the circuit's public input ports `pi`. -/
theorem SitesHP.aliased_keyProgP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    ∀ x x', (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x =
        (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x' →
      x.val < ((keyProgP c hU n pi).rows u).nIn → x'.val < ((keyProgP c hU n pi).rows u).nIn →
      ∀ z : Model.Witness (Partition.tabOf plan tab S R u).S.m, (Partition.tabOf plan tab S R u).S.Satisfies z →
        (Partition.tabOf plan tab S R u).S.block z (si.o S R u h)
            (si.col (SitesHP.rowsOf_keyProgP hU hparse pi) S R u h x) =
          (Partition.tabOf plan tab S R u).S.block z (si.o S R u h)
            (si.col (SitesHP.rowsOf_keyProgP hU hparse pi) S R u h x') := by
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have hpi' : HmIn.parse (si.acc S R u h).st.c = .ok pi := by rw [hc]; exact hpi
  have hrw := SitesHP.rowsOf_keyProgP hU hparse pi S R u h
  have e := congrArg Rows.nIn hrw
  refine (si.acc S R u h).aliased_of_layout hrw (Const.Prog.instC (keyProgP c hU n pi) outs u) (si.o S R u h)
    (si.hvu S R u h) ?_
  refine (si.acc S R u h).aliased_of_keyP hpi' _ _ (si.hvu S R u h) ?_
  intro x x' hw hx hx'
  have hw' := ((Placed.wire_cast hrw (Const.Prog.instC (keyProgP c hU n pi) outs u) x).symm.trans hw).trans
    (Placed.wire_cast hrw (Const.Prog.instC (keyProgP c hU n pi) outs u) x')
  rcases keyProgP_wire c hU pi outs u _ _ (lt_of_lt_of_eq hx e.symm) (lt_of_lt_of_eq hx' e.symm) hw' with he | hk
  · exact Or.inl (Fin.ext he)
  · right
    exact (congrArg (fun d => pubLeaf d pi x.val = none ∧ pubLeaf d pi x'.val = none ∧
      Aliased.srcKey d x.val = Aliased.srcKey d x'.val) hc).mpr hk

/-- **`ProgPlaces` for `keyProgP` at the sites**: each drawn unit placed at its VU in its table's accepted statement
(W6 from the parse), its instance the program's own (`Const.Prog.instC`), aliased as `SitesHP.aliased_keyProgP`. -/
noncomputable def SitesHP.progPlaces_keyProgP : ProgPlaces (keyProgP c hU n pi) outs plan tab where
  inst := Const.Prog.instC _ outs
  o := si.o
  col := si.col (SitesHP.rowsOf_keyProgP hU hparse pi)
  placed := si.placed (SitesHP.rowsOf_keyProgP hU hparse pi)
  aliased := SitesHP.aliased_keyProgP hU hparse pi hpi outs

end keyed

/-! ## A drawn table's acceptance -/

namespace TableHP

variable {n : ℕ} {I : Inputs} {own : ByteArray} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false)
  {S : Finset (Fin n)} (x : TableHP I own mPts S)

include hc in
/-- A drawn table's statement is on the session's one circuit. -/
theorem c_eq : x.st.c = c := by
  obtain ⟨_, hs⟩ := setupHidden_specP x.h
  exact Except.ok.inj (hs.parse.symm.trans hc)

include ht in
/-- **A table's statement has the layout the placement reads** (`layoutP`). -/
theorem layout : FlockSoundness.Layout x.st := layoutP ht x.h

include ht hc hv1 in
/-- **A table's unit slots' rows are its flat twin's off the public leaves** (`rowsFlatP`). -/
theorem rowsFlat : UnitRowsFlatP x.st := rowsFlatP ht x.h (by rw [x.c_eq hc]; exact hv1)

include ht in
/-- **The table's statement is the layout's block R1CS at its regions** (`Integrate.model_eq`, and `TableHP.hT`). -/
theorem model_eq : (x.layout ht).model x.regs = x.T.S := by
  obtain ⟨_, s⟩ := setupHidden_specP x.h
  exact (Integrate.model_eq (parse_facts (Hidden.parse_untyped ht s.parse).1) (x.layout ht) s.leaf x.hwf
    x.regs).trans x.hT.symm

/-- **A drawn table's acceptance**: its `Stmt.setupHidden … (some own)` run (`TableH.acc`). -/
noncomputable def acc : AcceptedHP x.T.S where
  tags := I.tags
  cf := I.circuitFile
  pf := I.publicFile
  coins := I.coins
  tables := I.tables
  draw := some x.d
  partition := I.partition
  program := I.program
  own := own
  st := x.st
  ht := ht
  h := x.h
  L := x.layout ht
  hR := x.rowsFlat ht hc hv1
  regs := x.regs
  hT := x.model_eq ht

end TableHP

end FlockSoundness.Discharge.ZkPubIn
