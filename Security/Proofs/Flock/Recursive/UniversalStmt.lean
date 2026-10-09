import Proofs.Flock.Recursive.UniversalLower
import Proofs.Flock.Recursive.Private
import Proofs.Flock.Soundness.Discharge.Hm.Port
import Flock.HmRow

/-!
# Recursion: the universal unit's statement by its rows, and when a session is faulty

The universal unit's private circuit is the description its registered rows commit to, run as the decoded circuit. This
file states that, for `RecursiveAudit`'s (i) and the verifier's refusals, over any bit order of the rows.

* **The decode bridge** (`interpret`, `eval_eq_interpret`): `interpret c x` runs a private circuit of the class
  (`Circ.eval`), and `Desc.eval d x = interpret (decode d) x` for every description and input, with no well-formedness
  hypothesis. `eval_uuType_interpret`, `uu_rows_iff_interpret` and `uu_unit_sound_interpret` are `UniversalLower`'s
  theorems in that form: the unit's outputs are `interpret (decode d) x`.
* **Faulty** (`Faulty d x y`): claimed outputs `y` that differ from `interpret (decode d) x`. On the unit's rows it is
  exactly "no assignment satisfies them" (`faulty_iff_unsat`), and a satisfying assignment of a typed statement's unit
  rows rules it out (`not_faulty_of_unit`).
* **Rows and their openings**: `rowCommit r salt` is a row's hm96 commit string `b ‖ c` (`sha512/row/v1`'s prefix and
  the row's bytes, under `salt`), `Opens q r` says some salt commits `r` to `q`, and `openRow q` is a row `q` opens to
  (`ByteArray.empty` when none does). Row bit `k` is bit `k mod 8` of byte `⌊k/8⌋` (`rowBit`), so bit `j mod 16` of
  u16 word `⌊j/16⌋` is row bit `j` (`rowBit_eq_word16`).
* **A bit order** `π : Fin n → ℕ × ℕ` names each bit's row and its bit position there. `rowBits π rows` reads given
  rows, `bitsAt π q` the rows the commit strings `q` open to, and `descAt π q` is the description.
* **The statement** (`stmt low π q ω`): the description `q` commits to through `π`, lowered by `low`. It reads only the
  rows `π` names (`stmt_congr`, `stmt_of_cover`: when every row below `nr` is read and agrees, the statement is the read
  strings'). At no hidden entries every `Hidden cls` is the empty template (`hidden_eq`), so there `stmt` is one
  statement whatever the rows (`stmt_const`): the description reaches the inner statement through its rows' commit
  strings, never through `Hidden`.
* **The layout functions** (staging (a), 6692c8703), computable over a statement's files. The row ports are `p0` (the
  description: one table row, which every instance refs), `p1` (the instance's inputs) and `out` (its hidden outputs).
  `argOrder p j` is bit `j` of registered row `0`: description bit `j` is bit `j mod 16` of `p0`'s u16 word `⌊j/16⌋`
  (`descAt_word`), and `descAt_op`, `descAt_src`, `descAt_out` read each place of `UniversalClass.layout` there.
  `srcAt st u a` is bit `a` of instance `u`'s `p1` table row, `outAt st u i` bit `16 i` (u16 word `i`) of its `out`
  table row, where `tableRow st u q` is the row `Public.digest` reads (`digestsAt_tableRow`). `paramsAt program` is
  `{G, N_IN, N_OUT}` as the program descriptor binds them (`UniversalUnit_v1`'s bindings), and `unitTypeAt st` the
  unit type a typed statement names, which T1 would compare with the packed universal type at those parameters.
* **A faulty session** (`SessionFaulty π ρx ρy q qs`): some instance faulty at every opening of the registered rows,
  which equals faulty at the developer's opening unless two openings of a read row collide under SHA-512 (cr/sha-512,
  listed). The description is read from the registered strings `q`, the instance's inputs and outputs from the
  statement's strings `qs` through `ρx u` and `ρy u`. Where every row read opens to exactly one row it is the chosen
  openings' `Faulty` (`sessionFaulty_iff`).
* **The layout tie** (T4, `not_faulty_of_rows`, `not_sessionFaulty_of`): an instance whose rows open the statement's
  strings at its table rows, the description's being the registered `q 0` (T5b's equality), and whose outputs read
  there are the decoded circuit's on its description and inputs read there, is no faulty session's witness. A row's bits
  at its columns are `rowBit_bytesOf`.
-/

namespace FlockSoundness.Discharge.Recursion.UniversalStmt

open Flock Flock.Derive Audit FlockSoundness.Types FlockSoundness.Discharge.Recursion.UniversalClass
open FlockSoundness.Discharge.Recursion.UniversalLower

variable {p : Params}

/-! ## The decode bridge -/

/-- **A private circuit of the class run on its inputs**: `Circ.eval`. Python's `universal.interpret` reads a
description; it is `interpret ∘ decode` (`eval_eq_interpret`). -/
abbrev interpret (c : Circ p) (x : Fin p.nIn → Bool) : Fin p.nOut → Bool := c.eval x

/-- **The description's outputs are its decoded circuit's**, for every description and input, with no well-formedness
hypothesis. -/
theorem eval_eq_interpret (d : Desc p) (x : Fin p.nIn → Bool) : d.eval x = interpret (UniversalClass.decode d) x :=
  (eval_decode d x).symm

/-- **What the unit computes, as the decoded circuit**: on any input bits, the type's outputs are
`interpret (decode d) x` of the description `d` and the data `x` those bits hold. -/
theorem eval_uuType_interpret (x : ℕ → Bool) :
    eval (uuType p) x = List.ofFn (interpret (UniversalClass.decode (descOf (p := p) x)) (dataOf x)) := by
  rw [eval_uuType, eval_eq_interpret]

/-- `uu_rows_iff` with the decoded circuit: the unit's rows carry outputs `y` on description `d` and data `x` exactly
when `y = interpret (decode d) x`. -/
theorem uu_rows_iff_interpret {inSplit outSplit : List ℕ} {dd : Derived}
    (hd : derive (uuType p) inSplit outSplit = .ok dd) (d : Desc p) (xs : Fin p.nIn → Bool) (y : Fin p.nOut → Bool) :
    (∃ z : ℕ → Bool, Sat dd.rows z ∧ z dd.geo.const = true ∧ (∀ j : Fin p.width, z (dd.inCol j) = d j) ∧
      (∀ a : Fin p.nIn, z (dd.inCol (p.width + a)) = xs a) ∧ ∀ i : Fin p.nOut, z (dd.outCol i) = y i) ↔
      y = interpret (UniversalClass.decode d) xs := by
  rw [← eval_eq_interpret]
  exact uu_rows_iff hd d xs y

/-- `uu_unit_sound` with the decoded circuit: a typed statement's unit rows, satisfied with `one` at 1, carry on each
output column `interpret (decode d) x` of their description columns `d` and data columns `x`. -/
theorem uu_unit_sound_interpret {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array DeriveAll.D}
    (hd : DeriveAll.deriveChecked types ls info words unit = .ok done)
    (ht : (ls.getD unit default).type.bind types = some (uuType p)) (one : ℕ) (z : ℕ → Bool)
    (hz : Dag.SatAt (DeriveAll.blockRow words (done.getD unit default) one) z) (h1 : z one = true) (i : Fin p.nOut) :
    z ((done.getD unit default).outCols.getD i 0) =
      interpret (UniversalClass.decode fun j : Fin p.width => z ((done.getD unit default).inCols.getD j 0))
        (fun a => z ((done.getD unit default).inCols.getD (p.width + a) 0)) i :=
  (uu_unit_sound hd ht one z hz h1 i).trans (congrFun (eval_eq_interpret _ _) i)

/-! ## Faulty -/

/-- **A faulty instance**: claimed outputs `y` that differ from the decoded circuit's on the inputs `x`. Over a session
(`SessionFaulty`) an instance is faulty at every opening of the registered rows, which equals faulty at the developer's
opening unless two openings of a read row collide under SHA-512 (cr/sha-512, listed). -/
def Faulty (d : Desc p) (x : Fin p.nIn → Bool) (y : Fin p.nOut → Bool) : Prop := y ≠ interpret (UniversalClass.decode d) x

theorem faulty_iff_eval {d : Desc p} {x : Fin p.nIn → Bool} {y : Fin p.nOut → Bool} : Faulty d x y ↔ y ≠ d.eval x := by
  rw [Faulty, eval_eq_interpret]

/-- **On the unit's rows, faulty is unsatisfiable**: claimed outputs `y` on description `d` and data `x` are faulty
exactly when no assignment satisfies the unit's rows with the constant at 1 and those columns. -/
theorem faulty_iff_unsat {inSplit outSplit : List ℕ} {dd : Derived} (hd : derive (uuType p) inSplit outSplit = .ok dd)
    (d : Desc p) (x : Fin p.nIn → Bool) (y : Fin p.nOut → Bool) :
    Faulty d x y ↔ ¬ ∃ z : ℕ → Bool, Sat dd.rows z ∧ z dd.geo.const = true ∧ (∀ j : Fin p.width, z (dd.inCol j) = d j) ∧
      (∀ a : Fin p.nIn, z (dd.inCol (p.width + a)) = x a) ∧ ∀ i : Fin p.nOut, z (dd.outCol i) = y i := by
  rw [uu_rows_iff_interpret hd]
  rfl

/-- **A typed statement's satisfied unit rows are no faulty instance**: when the unit's description, data and output
columns hold `d`, `x` and `y`, then `y = interpret (decode d) x`. -/
theorem not_faulty_of_unit {types : String → Option CircuitType} {ls : Array Flock.Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array DeriveAll.D}
    (hd : DeriveAll.deriveChecked types ls info words unit = .ok done)
    (ht : (ls.getD unit default).type.bind types = some (uuType p)) (one : ℕ) (z : ℕ → Bool)
    (hz : Dag.SatAt (DeriveAll.blockRow words (done.getD unit default) one) z) (h1 : z one = true)
    {d : Desc p} {x : Fin p.nIn → Bool} {y : Fin p.nOut → Bool}
    (hdz : ∀ j : Fin p.width, z ((done.getD unit default).inCols.getD j 0) = d j)
    (hxz : ∀ a : Fin p.nIn, z ((done.getD unit default).inCols.getD (p.width + a) 0) = x a)
    (hyz : ∀ i : Fin p.nOut, z ((done.getD unit default).outCols.getD i 0) = y i) : ¬ Faulty d x y := by
  intro hf
  apply hf
  funext i
  have e := uu_unit_sound_interpret hd ht one z hz h1 i
  rw [show (fun j : Fin p.width => z ((done.getD unit default).inCols.getD j 0)) = d from funext hdz,
    show (fun a : Fin p.nIn => z ((done.getD unit default).inCols.getD (p.width + a) 0)) = x from funext hxz] at e
  rw [← hyz i, e]

/-! ## Rows and their openings -/

/-- **A row's hm96 commit string under `salt`**: `b ‖ c` (`Hm96.Default512.commitString`) of the SHA-512 of
`sha512/row/v1`'s prefix for `|r| / 2` u16 words and the row's bytes, as `Recursive.Private.regRowCommit_opens` and the
circuit's `hm96` slot compute it. -/
def rowCommit (r salt : ByteArray) : ByteArray :=
  Flock.Hm96.Default512.commitString (Flock.Sha512.hash (Flock.HmRow.rowPrefix (r.size / 2) ++ r)) salt

/-- **The commit string `q` opens to the row `r`**: some salt commits `r` to `q`. -/
def Opens (q r : ByteArray) : Prop := ∃ salt, rowCommit r salt = q

open Classical in
/-- **The row a commit string commits to**: a row it opens to, `ByteArray.empty` when it opens to none. Two rows opening
one string are a SHA-512 collision (the binding `regRowCommit_binding` states for registered rows). -/
noncomputable def openRow (q : ByteArray) : ByteArray :=
  if h : ∃ r, Opens q r then Classical.choose h else ByteArray.empty

theorem opens_openRow {q r : ByteArray} (h : Opens q r) : Opens q (openRow q) := by
  have h' : ∃ r, Opens q r := ⟨r, h⟩
  rw [openRow, dite_eq_left h']
  exact Classical.choose_spec h'

theorem openRow_eq {q r : ByteArray} (h : Opens q r) (hu : ∀ r', Opens q r' → r' = r) : openRow q = r :=
  hu _ (opens_openRow h)

/-- **Bit `k` of a row**: bit `k mod 8` of byte `⌊k/8⌋` (`Layout.rowBytes`' order), `false` past the row. -/
def rowBit (r : ByteArray) (k : ℕ) : Bool := (r.get! (k / 8)).toNat.testBit (k % 8)

/-- **u16 word `w` of a row**, little-endian: bytes `2w` and `2w + 1`. -/
def word16 (r : ByteArray) (w : ℕ) : ℕ := (r.get! (2 * w)).toNat + 256 * (r.get! (2 * w + 1)).toNat

/-- Bit `b < 16` of u16 word `w` is row bit `16 w + b`. -/
theorem testBit_word16 (r : ByteArray) (w : ℕ) {b : ℕ} (hb : b < 16) :
    (word16 r w).testBit b = rowBit r (16 * w + b) := by
  have e : word16 r w = 2 ^ 8 * (r.get! (2 * w + 1)).toNat + (r.get! (2 * w)).toNat := by
    unfold word16
    ring
  rw [e, Nat.testBit_two_pow_mul_add _ (UInt8.toNat_lt _), rowBit]
  split_ifs with h8
  · rw [show (16 * w + b) / 8 = 2 * w by omega, show (16 * w + b) % 8 = b by omega]
  · rw [show (16 * w + b) / 8 = 2 * w + 1 by omega, show (16 * w + b) % 8 = b - 8 by omega]

/-- **Row bit `j` is bit `j mod 16` of u16 word `⌊j/16⌋`.** -/
theorem rowBit_eq_word16 (r : ByteArray) (j : ℕ) : rowBit r j = (word16 r (j / 16)).testBit (j % 16) := by
  rw [testBit_word16 r _ (Nat.mod_lt _ (by decide)), Nat.div_add_mod]

/-! ## A bit order, and the statement -/

/-- **Bits read through a bit order** `π` from given rows: bit `j` is bit `(π j).2` of row `(π j).1`. -/
def rowBits {n : ℕ} (π : Fin n → ℕ × ℕ) (rows : ℕ → ByteArray) : Fin n → Bool :=
  fun j => rowBit (rows (π j).1) (π j).2

/-- **Bits through `π` of the rows the commit strings `q` open to.** -/
noncomputable def bitsAt {n : ℕ} (π : Fin n → ℕ × ℕ) (q : ℕ → ByteArray) : Fin n → Bool :=
  rowBits π fun i => openRow (q i)

/-- **The description the commit strings `q` commit to**, read through `π`. -/
noncomputable abbrev descAt (π : Fin p.width → ℕ × ℕ) (q : ℕ → ByteArray) : Desc p := bitsAt π q

/-- **What `π` reads depends only on the rows it names**: on rows below `nr`, strings that agree give the same bits. -/
theorem bitsAt_congr {n nr : ℕ} {π : Fin n → ℕ × ℕ} (hπ : ∀ j, (π j).1 < nr) {q q' : ℕ → ByteArray}
    (h : ∀ i < nr, q i = q' i) : bitsAt π q = bitsAt π q' :=
  funext fun j => by
    show rowBit (openRow (q (π j).1)) (π j).2 = rowBit (openRow (q' (π j).1)) (π j).2
    rw [h _ (hπ j)]

/-- **A private circuit's statement by its rows' commit strings**: the description the strings `q` commit to through
`π`, lowered by `low`. The draw `ω` is unused: the registration fixes the description before any coin. -/
noncomputable def stmt {H Ω : Type} (low : Desc p → H) (π : Fin p.width → ℕ × ℕ) (q : ℕ → ByteArray) (_ω : Ω) : H :=
  low (descAt π q)

/-- **`stmt` reads only the description rows**: when `π` names rows below `nr`, `stmt q ω` depends only on `q i` for
`i < nr`. -/
theorem stmt_congr {H Ω : Type} (low : Desc p → H) {π : Fin p.width → ℕ × ℕ} {nr : ℕ} (hπ : ∀ j, (π j).1 < nr)
    {q q' : ℕ → ByteArray} (h : ∀ i < nr, q i = q' i) (ω : Ω) : stmt low π q ω = stmt low π q' ω := by
  show low (bitsAt π q) = low (bitsAt π q')
  rw [bitsAt_congr hπ h]

/-- **At reads that cover the description rows, the statement is the read strings'**: when every row below `nr` is
read (`ReadsCover`) and `q` agrees with the read strings `row` wherever a row is read, `stmt q ω = stmt row ω`. -/
theorem stmt_of_cover {H Ω : Type} (low : Desc p → H) {π : Fin p.width → ℕ × ℕ} {nr : ℕ} (hπ : ∀ j, (π j).1 < nr)
    (reads : ℕ → Prop) (row : ℕ → ByteArray) (hcov : ∀ i < nr, reads i) {q : ℕ → ByteArray}
    (hq : ∀ i, reads i → q i = row i) (ω : Ω) : stmt low π q ω = stmt low π row ω :=
  stmt_congr low hπ (fun i hi => hq i (hcov i hi)) ω

/-- **At no hidden entries, a hidden template is the empty one.** -/
theorem hidden_eq {cls : HClass} (h : cls.nnz = 0) (M M' : Hidden cls) : M = M' :=
  funext fun e => absurd e.isLt (by omega)

/-- **The universal unit's lowering** to a class with no hidden entries: the empty template. -/
def lowerU {cls : HClass} (h : cls.nnz = 0) : Desc p → Hidden cls :=
  fun _ e => absurd e.isLt (by omega)

/-- **At no hidden entries the statement is one**, whatever the rows and the draw: `Hidden cls` holds no description. -/
theorem stmt_const {cls : HClass} (h : cls.nnz = 0) {Ω : Type} (low : Desc p → Hidden cls) (π : Fin p.width → ℕ × ℕ)
    (q q' : ℕ → ByteArray) (ω ω' : Ω) : stmt low π q ω = stmt low π q' ω' :=
  hidden_eq h _ _

/-! ## The layout functions (staging (a)) -/

/-- **The staging's row ports**, in `c.ports` order: the description `p0`, the instance's inputs `p1`, its hidden
outputs `out`. -/
def descPort : ℕ := 0

/-- The instance's inputs' port, `p1`. -/
def xPort : ℕ := 1

/-- The instance's hidden outputs' port, `out`. -/
def outPort : ℕ := 2

/-- **The argument order** (staging (a)): description bit `j` is bit `j` of registered row `0`, the one `p0` row, which
is bit `j mod 16` of its u16 word `⌊j/16⌋` under `packed=True` (`descAt_word`). The sources `op` (`2G` bits), `src`
(`2G·K`) and `out` (`N_OUT·K`) follow one another there in `UniversalClass.layout`'s order. -/
def argOrder (p : Params) (j : Fin p.width) : ℕ × ℕ := (descPort, j.val)

/-- The argument order reads row `0` only. -/
theorem argOrder_lt (j : Fin p.width) : (argOrder p j).1 < 1 := Nat.zero_lt_one

/-- **At the argument order, description bit `j` is bit `j mod 16` of u16 word `⌊j/16⌋` of row `0`.** -/
theorem descAt_word (q : ℕ → ByteArray) (j : Fin p.width) :
    descAt (argOrder p) q j = (word16 (openRow (q 0)) (j.val / 16)).testBit (j.val % 16) :=
  rowBit_eq_word16 _ _

/-- Op bit `t` of gate `g` is bit `2g + t` of row `0`. -/
theorem descAt_op (q : ℕ → ByteArray) (g : Fin p.g) (t : Fin 2) :
    descAt (argOrder p) q (layout p (.inl (g, t))) = rowBit (openRow (q 0)) (2 * g.val + t.val) := by
  show rowBit (openRow (q 0)) (layout p _).val = _
  rw [layout_op]

/-- Bit `j` of gate `g`'s source `s` is bit `2G + (2g + s)·K + j` of row `0`. -/
theorem descAt_src (q : ℕ → ByteArray) (g : Fin p.g) (s : Fin 2) (j : Fin p.k) :
    descAt (argOrder p) q (layout p (.inr (.inl ((g, s), j)))) =
      rowBit (openRow (q 0)) (2 * p.g + (2 * g.val + s.val) * p.k + j.val) := by
  show rowBit (openRow (q 0)) (layout p _).val = _
  rw [layout_src]

/-- Bit `j` of output `i`'s index is bit `2G + 2G·K + i·K + j` of row `0`. -/
theorem descAt_out (q : ℕ → ByteArray) (i : Fin p.nOut) (j : Fin p.k) :
    descAt (argOrder p) q (layout p (.inr (.inr (i, j)))) =
      rowBit (openRow (q 0)) (2 * p.g + 2 * p.g * p.k + i.val * p.k + j.val) := by
  show rowBit (openRow (q 0)) (layout p _).val = _
  rw [layout_out]

/-- **At the argument order the statement reads row `0` only.** -/
theorem stmt_argOrder_congr {H Ω : Type} (low : Desc p → H) {q q' : ℕ → ByteArray} (h : q 0 = q' 0) (ω : Ω) :
    stmt low (argOrder p) q ω = stmt low (argOrder p) q' ω :=
  stmt_congr low argOrder_lt (fun i hi => by obtain rfl := Nat.lt_one_iff.mp hi; exact h) ω

/-- **The table row instance `u`'s port `q` reads**, as `Public.digest` reads it: `starts[q] + refs[u·np + q]` in a
shared-row file (its tables port-major), `u·np + q` otherwise, `np` the circuit's ports. -/
def tableRow (st : Flock.Stmt) (u q : ℕ) : ℕ :=
  match st.pub.tables with
  | some t => t.starts.getD q 0 + t.refs.getD (u * st.c.ports.size + q) 0
  | none => u * st.c.ports.size + q

/-- **The statement's commit strings by table row**: the public file's `b ‖ c` at row `i`, `ByteArray.empty` past
them. -/
def digestsAt (st : Flock.Stmt) (i : ℕ) : ByteArray := st.pub.digests.getD i ByteArray.empty

/-- **`digestsAt` at an instance's table row is the digest the verifier reads for it.** -/
theorem digestsAt_tableRow (st : Flock.Stmt) (u q : ℕ) :
    digestsAt st (tableRow st u q) = st.pub.digest u q st.c.ports.size := by
  unfold digestsAt tableRow Flock.Public.digest
  rcases st.pub.tables with _ | t <;> rfl

/-- **Instance `u`'s input bit `a`**: bit `a` of its `p1` table row. -/
def srcAt (st : Flock.Stmt) (u : Fin st.n) (a : Fin p.nIn) : ℕ × ℕ := (tableRow st u xPort, a.val)

/-- **Instance `u`'s output `i`**: bit `16 i` of its `out` table row, bit `0` of the row's u16 word `i`. -/
def outAt (st : Flock.Stmt) (u : Fin st.n) (i : Fin p.nOut) : ℕ × ℕ := (tableRow st u outPort, 16 * i.val)

/-- **`{G, N_IN, N_OUT}` a program descriptor binds**: the bindings of its composite `UniversalUnit_v1` (descriptor v1,
PROTOCOL §4.2 and §4.3), `none` when it holds none. -/
def paramsOfDescriptor (d : Lean.Json) : Option Params := do
  let .obj defs ← (d.getObjVal? "definitions").toOption | none
  defs.toArray.findSome? fun (_, f) => do
    guard ((f.getObjValAs? String "form").toOption == some "composite" &&
      (f.getObjValAs? String "id").toOption == some "UniversalUnit_v1")
    let b ← (f.getObjVal? "bindings").toOption
    let nat := fun k => (b.getObjValAs? Nat k).toOption
    return ⟨← nat "G", ← nat "N_IN", ← nat "N_OUT"⟩

/-- **The statement's parameters `{G, N_IN, N_OUT}`**, read off its program file (the descriptor META binds by
`program_sha512`), `none` when the file is no descriptor or binds no `UniversalUnit_v1`. A statement's other files don't
fix them: META and the header name the class only by digest, and a unit's leaf ports are 16-bit words whichever class
it is. -/
def paramsAt (program : ByteArray) : Option Params := do
  let text ← String.fromUTF8? program
  paramsOfDescriptor (← (Lean.Json.parse text).toOption)

/-- **The statement's unit type**: the type its class names for the unit's layout, `none` for an untyped statement.
T1 compares it with the universal unit's packed type at `paramsAt`, which Lean doesn't define yet. -/
def unitTypeAt (st : Flock.Stmt) : Option Flock.CircuitType :=
  st.c.cls.bind fun k => (k.ls.getD k.unit default).type.bind (Flock.Typed.typeIn k.types)

/-! ## A faulty session -/

/-- **`rows` open the commit strings `q` at every row `π` reads.** -/
def RowsOpen {n : ℕ} (π : Fin n → ℕ × ℕ) (q rows : ℕ → ByteArray) : Prop := ∀ j, Opens (q (π j).1) (rows (π j).1)

/-- **A faulty session**: some instance `u` faulty at every opening of the registered rows, which equals faulty at the
developer's opening unless two openings of a read row collide under SHA-512 (cr/sha-512, listed). The rows are those
`u` is read from: the description's from the registered commit strings `q` through `π`, the instance's inputs and
claimed outputs from the statement's commit strings `qs` through `ρx u` and `ρy u`. A string that opens to no row leaves
no opening, so an instance read from one is faulty. -/
def SessionFaulty {ι : Type} (π : Fin p.width → ℕ × ℕ) (ρx : ι → Fin p.nIn → ℕ × ℕ) (ρy : ι → Fin p.nOut → ℕ × ℕ)
    (q qs : ℕ → ByteArray) : Prop :=
  ∃ u, ∀ rows srows : ℕ → ByteArray, RowsOpen π q rows → RowsOpen (ρx u) qs srows → RowsOpen (ρy u) qs srows →
    Faulty (rowBits π rows) (rowBits (ρx u) srows) (rowBits (ρy u) srows)

/-- **Every string `π` reads opens to exactly one row.** -/
def OpensOnce {n : ℕ} (π : Fin n → ℕ × ℕ) (q : ℕ → ByteArray) : Prop :=
  ∀ j, (∃ r, Opens (q (π j).1) r) ∧ ∀ r r', Opens (q (π j).1) r → Opens (q (π j).1) r' → r = r'

theorem rowsOpen_openRow {n : ℕ} {π : Fin n → ℕ × ℕ} {q : ℕ → ByteArray} (h : OpensOnce π q) :
    RowsOpen π q fun i => openRow (q i) :=
  fun j => let ⟨_, hr⟩ := (h j).1; opens_openRow hr

theorem rowBits_eq_bitsAt {n : ℕ} {π : Fin n → ℕ × ℕ} {q rows : ℕ → ByteArray} (h : OpensOnce π q)
    (hr : RowsOpen π q rows) : rowBits π rows = bitsAt π q :=
  funext fun j => by
    show rowBit (rows (π j).1) (π j).2 = rowBit (openRow (q (π j).1)) (π j).2
    rw [(h j).2 _ _ (hr j) (rowsOpen_openRow h j)]

/-- **Where every row read opens once, a faulty session is one whose chosen openings are faulty.** -/
theorem sessionFaulty_iff {ι : Type} {π : Fin p.width → ℕ × ℕ} {ρx : ι → Fin p.nIn → ℕ × ℕ}
    {ρy : ι → Fin p.nOut → ℕ × ℕ} {q qs : ℕ → ByteArray} (hq : OpensOnce π q) (hx : ∀ u, OpensOnce (ρx u) qs)
    (hy : ∀ u, OpensOnce (ρy u) qs) :
    SessionFaulty π ρx ρy q qs ↔ ∃ u, Faulty (descAt π q) (bitsAt (ρx u) qs) (bitsAt (ρy u) qs) := by
  constructor
  · rintro ⟨u, hu⟩
    exact ⟨u, hu _ _ (rowsOpen_openRow hq) (rowsOpen_openRow (hx u)) (rowsOpen_openRow (hy u))⟩
  · rintro ⟨u, hu⟩
    refine ⟨u, fun rows srows hr hsx hsy => ?_⟩
    rw [rowBits_eq_bitsAt hq hr, rowBits_eq_bitsAt (hx u) hsx, rowBits_eq_bitsAt (hy u) hsy]
    exact hu

/-- **An instance correct at some opening is no faulty session's witness**: rows that open the strings it reads and on
which its claimed outputs are `interpret (decode d) x`. -/
theorem not_faulty_at {ι : Type} {π : Fin p.width → ℕ × ℕ} {ρx : ι → Fin p.nIn → ℕ × ℕ}
    {ρy : ι → Fin p.nOut → ℕ × ℕ} {q qs : ℕ → ByteArray} (u : ι) (rows srows : ℕ → ByteArray)
    (hr : RowsOpen π q rows) (hsx : RowsOpen (ρx u) qs srows) (hsy : RowsOpen (ρy u) qs srows)
    (hok : rowBits (ρy u) srows = interpret (UniversalClass.decode (rowBits π rows)) (rowBits (ρx u) srows)) :
    ¬ ∀ rows srows : ℕ → ByteArray, RowsOpen π q rows → RowsOpen (ρx u) qs srows → RowsOpen (ρy u) qs srows →
      Faulty (rowBits π rows) (rowBits (ρx u) srows) (rowBits (ρy u) srows) :=
  fun h => h rows srows hr hsx hsy hok

/-! ## T4: the layout tie -/

/-- **Bit `k` of the row whose bytes are `l`** is bit `k mod 8` of `l`'s byte `⌊k/8⌋` (`Hm.bitL`). -/
theorem rowBit_mk (l : List UInt8) (k : ℕ) : rowBit ⟨l.toArray⟩ k = FlockSoundness.Discharge.Hm.bitL l k := by
  unfold rowBit FlockSoundness.Discharge.Hm.bitL
  congr 2
  show l.toArray[k / 8]! = l.getD (k / 8) 0
  rw [getElem!_def, List.getD_eq_getElem?_getD, List.getElem?_toArray]
  rcases l[k / 8]? <;> rfl

/-- **A row at its columns**: bit `k < 8N` of the row whose byte `j`'s bit `t` is `w (a j t)` is
`w (a ⌊k/8⌋ (k mod 8))`. At `a = Typed.msgCol c g q` and `N = 2 · words` it is port `q`'s row at template instance
`g`, the message bytes `row_bc_flat` commits. -/
theorem rowBit_bytesOf (w : ℕ → Bool) {N : ℕ} (a : ℕ → ℕ → ℕ) {k : ℕ} (hk : k < 8 * N) :
    rowBit ⟨(FlockSoundness.Discharge.Hm.bytesOf w N a).toArray⟩ k = w (a (k / 8) (k % 8)) := by
  rw [rowBit_mk, FlockSoundness.Discharge.Hm.bitL_bytesOf w a hk]

/-- **An instance's rows, as the statement's**: `rx` at its `p1` table row, `ry` at every other. -/
def instRows (st : Flock.Stmt) (u : ℕ) (rx ry : ByteArray) : ℕ → ByteArray :=
  fun i => if i = tableRow st u xPort then rx else ry

theorem rowBits_argOrder (rd : ByteArray) : rowBits (argOrder p) (fun _ => rd) = fun j => rowBit rd j.val := rfl

theorem rowBits_srcAt (st : Flock.Stmt) (u : Fin st.n) (rx ry : ByteArray) :
    rowBits (srcAt (p := p) st u) (instRows st u rx ry) = fun a => rowBit rx a.val := by
  funext a
  simp [rowBits, srcAt, instRows]

theorem rowBits_outAt (st : Flock.Stmt) (u : Fin st.n) (rx ry : ByteArray)
    (hne : tableRow st u outPort ≠ tableRow st u xPort) :
    rowBits (outAt (p := p) st u) (instRows st u rx ry) = fun i => rowBit ry (16 * i.val) := by
  funext i
  simp [rowBits, outAt, instRows, hne]

theorem rowsOpen_argOrder {q : ℕ → ByteArray} {rd : ByteArray} (h : Opens (q 0) rd) :
    RowsOpen (argOrder p) q fun _ => rd :=
  fun _ => h

theorem rowsOpen_srcAt {st : Flock.Stmt} {u : Fin st.n} {rx ry : ByteArray}
    (h : Opens (digestsAt st (tableRow st u xPort)) rx) :
    RowsOpen (srcAt (p := p) st u) (digestsAt st) (instRows st u rx ry) := fun _ => by
  simpa [srcAt, instRows] using h

theorem rowsOpen_outAt {st : Flock.Stmt} {u : Fin st.n} {rx ry : ByteArray}
    (hne : tableRow st u outPort ≠ tableRow st u xPort) (h : Opens (digestsAt st (tableRow st u outPort)) ry) :
    RowsOpen (outAt (p := p) st u) (digestsAt st) (instRows st u rx ry) := fun _ => by
  simpa [outAt, instRows, hne] using h

/-- **T4 at one instance, from its rows.** Instance `u`'s rows `rd`, `rx`, `ry` open the statement's strings at its
`p0`, `p1` and `out` table rows (T3 with the region tie), and the `p0` string is the registered `q 0` (`h5b`, T5b's
equality). Its `p1` and `out` table rows differ (`hne`: the public file's refs lie below their tables' counts, and the
tables are port-major). Its outputs, bit `16 i` of `ry`, are the decoded circuit's on its description, bit `j` of `rd`,
and its inputs, bit `a` of `rx` (`hok`, the unit's soundness at its slot read through T2's copies). Then `u` is no
faulty session's witness. -/
theorem not_faulty_of_rows {st : Flock.Stmt} {q : ℕ → ByteArray} (u : Fin st.n) {rd rx ry : ByteArray}
    (h5b : digestsAt st (tableRow st u descPort) = q 0) (hne : tableRow st u outPort ≠ tableRow st u xPort)
    (hod : Opens (digestsAt st (tableRow st u descPort)) rd) (hox : Opens (digestsAt st (tableRow st u xPort)) rx)
    (hoy : Opens (digestsAt st (tableRow st u outPort)) ry)
    (hok : (fun i : Fin p.nOut => rowBit ry (16 * i.val)) =
      interpret (UniversalClass.decode fun j : Fin p.width => rowBit rd j.val) fun a => rowBit rx a.val) :
    ¬ ∀ rows srows : ℕ → ByteArray, RowsOpen (argOrder p) q rows → RowsOpen (srcAt (p := p) st u) (digestsAt st) srows →
      RowsOpen (outAt (p := p) st u) (digestsAt st) srows →
      Faulty (rowBits (argOrder p) rows) (rowBits (srcAt st u) srows) (rowBits (outAt st u) srows) := by
  intro h
  have hf := h (fun _ => rd) (instRows st u rx ry) (rowsOpen_argOrder (h5b ▸ hod)) (rowsOpen_srcAt hox)
    (rowsOpen_outAt hne hoy)
  rw [rowBits_argOrder, rowBits_srcAt, rowBits_outAt st u rx ry hne] at hf
  exact hf hok

/-- **T4 over the session**: a session none of whose instances is a witness is not faulty. -/
theorem not_sessionFaulty_of {st : Flock.Stmt} {q : ℕ → ByteArray}
    (h : ∀ u : Fin st.n, ¬ ∀ rows srows : ℕ → ByteArray, RowsOpen (argOrder p) q rows →
      RowsOpen (srcAt (p := p) st u) (digestsAt st) srows → RowsOpen (outAt (p := p) st u) (digestsAt st) srows →
      Faulty (rowBits (argOrder p) rows) (rowBits (srcAt st u) srows) (rowBits (outAt st u) srows)) :
    ¬ SessionFaulty (argOrder p) (srcAt st) (outAt st) q (digestsAt st) :=
  fun ⟨u, hu⟩ => h u hu

/-! ## (i) at a faulty session -/

/-- **(i)'s event at a faulty session, from (i)'s event at the statement failing**: when, at every outcome `s` reaches
where the verifier accepts the statement (`acc`) and the sessions (`ver`), the statement holding rules out a faulty
session, the faulty event is at most as likely as the statement failing. -/
theorem prob_faulty_le {α : Type} {acc ver holds faulty : α → Prop} (g : FlockSoundness.Game α)
    (s : FlockSoundness.Game.Strategy g)
    (htie : ∀ o, FlockSoundness.Game.Outcome g s o → acc o → ver o → holds o → ¬ faulty o) :
    FlockSoundness.Game.prob (fun o => acc o ∧ ver o ∧ faulty o) g s ≤
      FlockSoundness.Game.prob (fun o => acc o ∧ ver o ∧ ¬ holds o) g s :=
  Flock.SecurityProofs.prob_mono_outcome g s fun o ho h => ⟨h.1, h.2.1, fun hh => htie o ho h.1 h.2.1 hh h.2.2⟩

end FlockSoundness.Discharge.Recursion.UniversalStmt
