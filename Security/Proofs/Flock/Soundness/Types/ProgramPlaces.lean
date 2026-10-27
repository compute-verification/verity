import Proofs.Flock.Soundness.Types.Program

/-!
# Each drawn unit's place, for a program of units (verified-lowering phase 1, S4d)

`E2E.lean`'s `DerivedPlaces` asks, for every draw and every drawn unit, the unit's place in its table: its rows, its
instance in the committed circuit, a block and the columns there at which the table's statement holds the rows. For a
program of units (`Types.Program`) the rows and the instance are the program's own:
- `UProg.spec u` is unit `u`'s spec, `UProg.rowsOf u` its composed rows and `UProg.instOf u` their instance in `p.P`;
- `UProg.derivedPlaces` is `DerivedPlaces` from a block and columns per drawn unit at which its table holds its rows
  (`RowsAt`), and carries one value on the unit's inputs that share a gate (`RowsAt.Aliased`).

`TableClass` is what a table's accepted statement gives (1d and 1e): the verifier's derivation of its unit class, the
class's rows placed at each slot, each slot input's copy position and the forced-zero rows.
`UProg.derivedPlaces_of_classes` places each drawn unit at a slot of its table's class:
- when the unit reads the class's derivation inputs (`Reads`), it has the class's rows, since `deriveChecked` is a function;
- when the slot's inputs that read one of the unit's sources (`UProg.srcOf`) copy one position, or forced-zero rows
  (`TableClass.Copies`), the place is aliased (`aliased_of_copies`). A unit's instance wires each input to its source's
  gate (`instOf_wire`), and inputs on one gate read one source (`srcGate_inj`).
-/

open Flock Flock.DeriveAll

namespace FlockSoundness

open Audit Types.Dag Compose Model

/-- A block, and the columns there at which statement `St` holds rows `R`. -/
structure RowsAt (St : Statement) (R : Rows) where
  o : Fin (2 ^ St.nbl)
  col : Fin R.w → Fin (2 ^ St.kLog)
  placed : Placement St R col

/-- The same place, for equal rows. -/
def RowsAt.cast {St : Statement} {R R' : Rows} (e : R = R') (a : RowsAt St R) : RowsAt St R' := e ▸ a

theorem RowsAt.cast_o {St : Statement} {R R' : Rows} (e : R = R') (a : RowsAt St R) : (a.cast e).o = a.o := by
  subst e; rfl

theorem RowsAt.cast_col {St : Statement} {R R' : Rows} (e : R = R') (a : RowsAt St R) (c : Fin R'.w) :
    (a.cast e).col c = a.col (Fin.cast (congrArg Rows.w e).symm c) := by
  subst e; rfl

/-- **The place carries one value on inputs that share a gate**: in every witness satisfying `St`, the input columns on
one gate of `wire` hold one value. -/
def RowsAt.Aliased {St : Statement} {R : Rows} {N : ℕ} (a : RowsAt St R) (wire : Fin R.w → Fin N) : Prop :=
  ∀ c c', wire c = wire c' → c.val < R.nIn → c'.val < R.nIn → ∀ z : Witness St.m, St.Satisfies z →
    St.block z a.o (a.col c) = St.block z a.o (a.col c')

/-- **A table's unit class, placed at each slot**: the verifier's run of `deriveChecked` on the class's layouts, and
statement `St`'s level-0 matrices holding the class's composed rows at each slot's columns. Every block of a satisfying
witness carries, at each slot input, what it carries at the input's copy position (`copyPos`, from Δ), and 0 at each
forced-zero row (`zeros`, whose `A₀` and `B₀` rows are 0). -/
structure TableClass (St : Statement) where
  types : String → Option CircuitType
  ls : Array Layout
  info : String → Option TableInfo
  words : String → Option (Array ℕ)
  unit : ℕ
  done : Array D
  hd : deriveChecked types ls info words unit = .ok done
  slots : ℕ
  col : Fin slots → Fin (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)).w →
    Fin (2 ^ St.kLog)
  placed : ∀ g, Placement St (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)) (col g)
  copyPos : Fin slots → ℕ → Fin (2 ^ St.kLog)
  copy : ∀ g (c : Fin (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)).w),
    c.val < (Rows.compose (ofBlock words done (done.getD unit default)) (ofBlock_ordered hd)).nIn →
    ∀ z : Witness St.m, St.Satisfies z → ∀ o, St.block z o (col g c) = St.block z o (copyPos g c.val)
  zeros : Finset (Fin (2 ^ St.kLog))
  zero : ∀ q ∈ zeros, ∀ z : Witness St.m, St.Satisfies z → ∀ o, St.block z o q = 0

namespace TableClass

variable {St : Statement} (c : TableClass St)

/-- The class's composed rows. -/
abbrev rows : Rows := Rows.compose (ofBlock c.words c.done (c.done.getD c.unit default)) (ofBlock_ordered c.hd)

/-- Slot `g` of block `o`. -/
def atSlot (g : Fin c.slots) (o : Fin (2 ^ St.nbl)) : RowsAt St c.rows := ⟨o, c.col g, c.placed g⟩

/-- **Slot `g`'s inputs that read one source copy one position, or forced-zero rows**, for the sources `src` of the unit
placed there. -/
def Copies (g : Fin c.slots) (src : ℕ → Src) : Prop :=
  ∀ i < c.rows.nIn, ∀ j < c.rows.nIn, src i = src j →
    c.copyPos g i = c.copyPos g j ∨ c.copyPos g i ∈ c.zeros ∧ c.copyPos g j ∈ c.zeros

end TableClass

/-- **A zero row carries 0**: a position whose `A₀` row is 0 is 0 in every block of a satisfying witness. -/
theorem block_eq_zero_of_row {St : Statement} {q : Fin (2 ^ St.kLog)} (hq : St.A₀ q = 0) {z : Witness St.m}
    (hz : St.Satisfies z) (o : Fin (2 ^ St.nbl)) : St.block z o q = 0 := by
  simpa [Matrix.mulVec, dotProduct, hq] using (hz.1 o q).symm

/-- A unit reads a table class's derivation inputs: its types, layouts, table info and words, and layout. -/
def UnitSpec.Reads {St : Statement} (s : UnitSpec) (c : TableClass St) : Prop :=
  s.types = c.types ∧ s.ls = c.ls ∧ s.info = c.info ∧ s.words = c.words ∧ s.idx = c.unit

theorem compose_congr {L L' : Logical} (e : L = L') (h : Ordered L) (h' : Ordered L') :
    Rows.compose L h = Rows.compose L' h' := by
  subst e; rfl

/-- **A unit that reads a class's inputs has the class's rows**: `deriveChecked` is a function. -/
theorem UnitSpec.R_eq_of_reads {St : Statement} {s : UnitSpec} {c : TableClass St} (h : s.Reads c) : s.R = c.rows := by
  obtain ⟨ht, hl, hi, hw, hu⟩ := h
  have hs := s.hd
  rw [ht, hl, hi, hw, hu, c.hd] at hs
  have hdn : s.done = c.done := (Except.ok.inj hs).symm
  exact compose_congr (by rw [hw, hdn, hu]) _ _

/-- A unit pair's spec. -/
def UnitPair.toSpec {C CB : Audit.Circuit} {n : ℕ} {P : Partition C n} {PB : Partition CB n} {m : Fin CB.N → Fin C.N}
    {u : Fin n} (U : UnitPair P PB m u) : UnitSpec :=
  ⟨U.types, U.ls, U.info, U.words, U.idx, U.done, U.hd, U.t, U.ht, U.loOf, U.F, U.hF⟩

/-- A unit pair's rows instance wires input column `c` to gate `f c`. -/
def UnitPair.WireOn {C CB : Audit.Circuit} {n : ℕ} {P : Partition C n} {PB : Partition CB n} {m : Fin CB.N → Fin C.N}
    {u : Fin n} (U : UnitPair P PB m u) (f : ℕ → ℕ) : Prop :=
  ∀ c : Fin (Rows.compose (ofBlock U.words U.done (U.done.getD U.idx default)) U.ord).w, c.val < U.t.inBits →
    (U.rows.wire c).val = f c.val

namespace UProg

variable {k one : ℕ}

/-- Unit `u`'s spec. -/
noncomputable def spec (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) : UnitSpec :=
  (p.pairs hk u).toSpec

/-- Unit `u`'s composed rows. -/
noncomputable abbrev rowsOf (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) : Rows :=
  (p.spec hk u).R

/-- Their instance in the program's rows circuit. -/
noncomputable def instOf (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) :
    (p.P hk).IsRowsUnit u (p.rowsOf hk u) :=
  (p.pairs hk u).rows

theorem spec_last (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) :
    (snoc p s src valid).spec hk (Fin.last p.n) = s := by
  unfold spec
  rw [pairs, Fin.lastCases_last]
  rfl

theorem spec_castSucc (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) (v : Fin p.n) :
    (snoc p s src valid).spec hk v.castSucc = p.spec hk v := by
  unfold spec
  rw [pairs, Fin.lastCases_castSucc]
  rfl

/-- Unit `u`'s sources. -/
def srcOf : {outs : List ℕ} → (p : UProg k one outs) → Fin p.n → ℕ → Src
  | _, .inputs, u => u.elim0
  | _, .snoc p _ src _, u => Fin.lastCases (motive := fun _ => ℕ → Src) src (fun v => p.srcOf v) u

theorem srcOf_last {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) : (snoc p s src valid).srcOf (Fin.last p.n) = src := by
  rw [srcOf, Fin.lastCases_last]

theorem srcOf_castSucc {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) (v : Fin p.n) :
    (snoc p s src valid).srcOf v.castSucc = p.srcOf v := by
  rw [srcOf, Fin.lastCases_castSucc]

/-- A source of the program before a unit is appended has the same gate after. -/
theorem srcGate_snoc {outs : List ℕ} (p : UProg k one outs) (s : UnitSpec) (src : ℕ → Src)
    (valid : ∀ i < s.t.inBits, (src i).Valid k one outs) {x : Src} (hx : x.Valid k one outs) :
    (snoc p s src valid).srcGate x = p.srcGate x := by
  cases x with
  | input a => rfl
  | out v o =>
    have hv : v ≠ p.n := by rw [n_eq p]; exact Nat.ne_of_lt hx.1
    simp only [srcGate, outGate, hv, ite_false]

/-- A unit's sources are valid in the program. -/
theorem srcOf_valid (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) {i : ℕ},
    i < (p.spec hk u).t.inBits → (p.srcOf u i).Valid k one outs
  | _, .inputs, u, _, _ => u.elim0
  | _, .snoc p s src valid, u, i, hi => by
    induction u using Fin.lastCases with
    | last =>
      rw [spec_last] at hi
      rw [srcOf_last]
      exact valid_snoc (valid i hi)
    | cast v =>
      rw [spec_castSucc] at hi
      rw [srcOf_castSucc]
      exact valid_snoc (srcOf_valid hk p v hi)

/-- **A unit's rows instance wires each input column to its source's gate.** -/
theorem pairs_wire (hk : one < k) : ∀ {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n),
    (p.pairs hk u).WireOn fun i => p.srcGate (p.srcOf u i)
  | _, .inputs, u => u.elim0
  | _, .snoc p s src valid, u => by
    induction u using Fin.lastCases with
    | last =>
      have e : (snoc p s src valid).pairs hk (Fin.last p.n) = lastPair hk p s src valid := by
        unfold pairs; exact Fin.lastCases_last
      rw [e]
      intro c hc
      beta_reduce
      rw [srcOf_last, srcGate_snoc p s src valid (valid _ hc)]
      have key : ∀ c' : Fin s.R.w, c'.val < s.t.inBits →
          (Prog.colWire s.R (p.insOf s src valid) (p.oneOf hk) c').val = p.srcGate (src c'.val) := by
        intro c' hc'
        rw [Prog.colWire_val, dite_eq_left (by rw [s.nIn_eq]; exact hc')]
        rfl
      exact key c hc
    | cast v =>
      have e : (snoc p s src valid).pairs hk v.castSucc = snocPair hk p s src valid (p.pairs hk v) := by
        rw [pairs, Fin.lastCases_castSucc]
      rw [e]
      intro c hc
      beta_reduce
      rw [srcOf_castSucc, srcGate_snoc p s src valid (srcOf_valid hk p v hc)]
      exact pairs_wire hk p v c hc

theorem instOf_wire (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) (c : Fin (p.rowsOf hk u).w)
    (hc : c.val < (p.spec hk u).t.inBits) : ((p.instOf hk u).wire c).val = p.srcGate (p.srcOf u c.val) :=
  p.pairs_wire hk u c hc

variable {Reg : Type} {mPts : ℕ}

/-- **Each drawn unit's place, from its rows' place in its table**, which carries one value on the unit's inputs that
share a gate. -/
noncomputable def derivedPlaces (hk : one < k) {outs : List ℕ} (p : UProg k one outs)
    (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (at_ : ∀ S R u, u ∈ S → RowsAt (Partition.tabOf plan tab S R u).S (p.rowsOf hk u))
    (hal : ∀ S R u (h : u ∈ S), (at_ S R u h).Aliased (p.instOf hk u).wire) :
    (p.P hk).DerivedPlaces plan tab (p.rowsOf hk) where
  place S R u h :=
    ⟨p.rowsOf hk u, p.instOf hk u, (at_ S R u h).o, (at_ S R u h).col, (at_ S R u h).placed.sat, hal S R u h⟩
  rows _ _ _ _ := rfl

/-- **A slot's copies make its place aliased**: unit `u`, at slot `g` of block `o` of a class whose derivation inputs it
reads, carries one value on its inputs that share a gate when the slot's inputs that read one of `u`'s sources copy one
position, or forced-zero rows. Inputs on one gate read one source (`srcGate_inj`). -/
theorem aliased_of_copies (hk : one < k) {outs : List ℕ} (p : UProg k one outs) (u : Fin p.n) {St : Statement}
    (cl : TableClass St) (g : Fin cl.slots) (o : Fin (2 ^ St.nbl)) (hr : (p.spec hk u).Reads cl)
    (hc : cl.Copies g (p.srcOf u)) :
    ((cl.atSlot g o).cast (UnitSpec.R_eq_of_reads hr).symm).Aliased (p.instOf hk u).wire := by
  intro c c' hw h1 h2 z hz
  have hn : (p.rowsOf hk u).nIn = cl.rows.nIn := congrArg Rows.nIn (UnitSpec.R_eq_of_reads hr)
  have hi : (p.rowsOf hk u).nIn = (p.spec hk u).t.inBits := (p.spec hk u).nIn_eq
  rw [RowsAt.cast_o, RowsAt.cast_col, RowsAt.cast_col]
  change St.block z o (cl.col g _) = St.block z o (cl.col g _)
  rw [cl.copy g _ (by rw [Fin.val_cast, ← hn]; exact h1) z hz o, cl.copy g _ (by rw [Fin.val_cast, ← hn]; exact h2) z hz o,
    Fin.val_cast, Fin.val_cast]
  have hv := congrArg Fin.val hw
  rw [p.instOf_wire hk u c (hi ▸ h1), p.instOf_wire hk u c' (hi ▸ h2)] at hv
  have hs := p.srcGate_inj (p.srcOf_valid hk u (hi ▸ h1)) (p.srcOf_valid hk u (hi ▸ h2)) hv
  rcases hc c.val (hn ▸ h1) c'.val (hn ▸ h2) hs with he | ⟨hz1, hz2⟩
  · rw [he]
  · rw [cl.zero _ hz1 z hz o, cl.zero _ hz2 z hz o]

/-- **Each drawn unit's place, from its table's class**: unit `u`, drawn in `S` under `R`, at slot `slot` of block `o`
of its table's class `cls`, whose derivation inputs it reads, where the slot's inputs that read one of `u`'s sources
copy one position, or forced-zero rows. -/
noncomputable def derivedPlaces_of_classes (hk : one < k) {outs : List ℕ} (p : UProg k one outs)
    (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
    (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)
    (cls : ∀ S R u, u ∈ S → TableClass (Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (o : ∀ S R u, u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ S R u (h : u ∈ S), (p.spec hk u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u)) :
    (p.P hk).DerivedPlaces plan tab (p.rowsOf hk) :=
  p.derivedPlaces hk plan tab (fun S R u h =>
      ((cls S R u h).atSlot (slot S R u h) (o S R u h)).cast (UnitSpec.R_eq_of_reads (hr S R u h)).symm)
    fun S R u h => p.aliased_of_copies hk u (cls S R u h) (slot S R u h) (o S R u h) (hr S R u h) (hcp S R u h)

end UProg

end FlockSoundness
