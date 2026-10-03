import FlockSoundness.Discharge.ZkPubIn.Spec
import FlockSoundness.Discharge.ZkPubIn.Tables
import FlockSoundness.Discharge.HiddenExec.Sites

set_option Elab.async false

/-!
# The sites and places of `keyProgP` in tables with the verifier's public inputs

HiddenExec's `AcceptedH`, `SitesH`, `TableH.acc` and `SitesH.progPlaces_keyProg` (`HiddenExec/Placed.lean`,
`HiddenExec/Sites.lean`) with `Stmt.setupHidden … (some own)`, at the program with its public gates (`keyProgP`):

* `AcceptedHP T`: a run of `Stmt.setupHidden … (some own)` that accepted, with a layout of its statement, its unit
  slots' rows its flat twin's, and its block R1CS `T`; `AcceptedHP.col`, `.placement` as `AcceptedH`'s.
* `SitesHP plan tab`: each drawn unit's acceptance, VU and block; `TableHP.acc`: a table's acceptance.
* `SitesHP.progPlaces_keyProgP`: `ProgPlaces` for `keyProgP`, its aliasing `SitesHP.aliased_keyProgP` (a public
  leaf's bit is on a gate of its own; any other input's gate is a Δ key's, whose copies `HmIn.delta` keeps).
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Model Refine Audit HiddenExec
open FlockSoundness.Discharge.Exec (Inputs)

/-! ## An accepted hidden-output statement with the verifier's public inputs -/

/-- **An accepted untyped hidden-output statement with the verifier's public inputs whose block R1CS is `T`**:
`AcceptedH` with `Stmt.setupHidden … (some own)`. -/
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
  hR : UnitRowsFlat st
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hT : L.model regs = T

namespace AcceptedHP

variable {T : Model.Statement} (a : AcceptedHP T)

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

include hparse in
/-- W6 for `keyProgP`, from the parse. -/
theorem SitesHP.rowsOf_keyProgP : si.RowsOf (keyProgP c hU n pi) :=
  fun S R u h => (Placed.rows_circuitProg c hU n _ _ u).trans ((si.acc S R u h).rows_eq (hparse S R u h) hU).symm

variable (outs : Finset (Fin (Layout.keyN c hU n)))

include hparse in
/-- **`aliased` at each drawn unit's site**: in a satisfying witness of the table a unit is drawn into, the program's
own instance of the unit puts on one gate only inputs that hold one value at the unit's VU and block: a public leaf's
bit is alone on its gate, and inputs with one Δ key, none a public leaf's bit, are one column or the zero
(`HmIn.delta` drops only a public leaf's copies). -/
theorem SitesHP.aliased_keyProgP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    ∀ x x', (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x =
        (Const.Prog.instC (keyProgP c hU n pi) outs u).wire x' →
      x.val < ((keyProgP c hU n pi).rows u).nIn → x'.val < ((keyProgP c hU n pi).rows u).nIn →
      ∀ z : Model.Witness (Partition.tabOf plan tab S R u).S.m, (Partition.tabOf plan tab S R u).S.Satisfies z →
        (Partition.tabOf plan tab S R u).S.block z (si.o S R u h)
            (si.col (SitesHP.rowsOf_keyProgP hU hparse pi) S R u h x) =
          (Partition.tabOf plan tab S R u).S.block z (si.o S R u h)
            (si.col (SitesHP.rowsOf_keyProgP hU hparse pi) S R u h x') := by
  sorry

/-- **`ProgPlaces` for `keyProgP` at the sites**: each drawn unit placed at its VU in its table's accepted statement
(W6 from the parse), its instance the program's own (`Const.Prog.instC`), aliased as `SitesHP.aliased_keyProgP`. -/
noncomputable def SitesHP.progPlaces_keyProgP : ProgPlaces (keyProgP c hU n pi) outs plan tab where
  inst := Const.Prog.instC _ outs
  o := si.o
  col := si.col (SitesHP.rowsOf_keyProgP hU hparse pi)
  placed := si.placed (SitesHP.rowsOf_keyProgP hU hparse pi)
  aliased := SitesHP.aliased_keyProgP hU hparse pi outs

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
/-- **A table's statement has the layout the placement reads** (`Hidden.hidden_layout` with the public inputs). -/
theorem layout : FlockSoundness.Layout x.st := by
  sorry

include ht hc hv1 in
/-- **A table's unit slots' rows are its flat twin's** (`unitRowsFlat` with the public inputs). -/
theorem rowsFlat : UnitRowsFlat x.st := by
  sorry

include ht in
/-- **The table's statement is the layout's block R1CS at its regions** (`Integrate.model_eq`, and `TableHP.hT`). -/
theorem model_eq : (x.layout ht).model x.regs = x.T.S := by
  sorry

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
