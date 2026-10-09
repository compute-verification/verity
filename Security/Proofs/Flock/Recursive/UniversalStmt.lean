import Proofs.Flock.Recursive.UniversalLower
import Proofs.Flock.Recursive.Private
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
* **The argument order** (`argOrder p`, staging option (a)): the sources `op`, `src`, `out` are rows `0`, `1`, `2`, and
  the bit at offset `j` of a source is row bit `j`: bit `j mod 16` of the row's u16 word `⌊j/16⌋` (`descAt_word`).
  `descAt_op`, `descAt_src`, `descAt_out` read each place of `UniversalClass.layout` there.
* **A faulty session** (`SessionFaulty π ρx ρy q qs`): some instance whose claimed outputs differ from the decoded
  circuit's on its inputs under every opening of the rows they're read from (the description's from the registered
  strings `q`; the instance's inputs and outputs from the statement's strings `qs`, through `ρx u` and `ρy u`). Where
  every row read opens to exactly one row it is the chosen openings' `Faulty` (`sessionFaulty_iff`).
* `prob_faulty_le`: (i)'s event at a faulty session from (i)'s event at the statement failing, once the statement
  holding rules out a faulty session at every outcome reached.
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
theorem eval_eq_interpret (d : Desc p) (x : Fin p.nIn → Bool) : d.eval x = interpret (decode d) x :=
  (eval_decode d x).symm

/-- **What the unit computes, as the decoded circuit**: on any input bits, the type's outputs are
`interpret (decode d) x` of the description `d` and the data `x` those bits hold. -/
theorem eval_uuType_interpret (x : ℕ → Bool) :
    eval (uuType p) x = List.ofFn (interpret (decode (descOf (p := p) x)) (dataOf x)) := by
  rw [eval_uuType, eval_eq_interpret]

/-- `uu_rows_iff` with the decoded circuit: the unit's rows carry outputs `y` on description `d` and data `x` exactly
when `y = interpret (decode d) x`. -/
theorem uu_rows_iff_interpret {inSplit outSplit : List ℕ} {dd : Derived}
    (hd : derive (uuType p) inSplit outSplit = .ok dd) (d : Desc p) (xs : Fin p.nIn → Bool) (y : Fin p.nOut → Bool) :
    (∃ z : ℕ → Bool, Sat dd.rows z ∧ z dd.geo.const = true ∧ (∀ j : Fin p.width, z (dd.inCol j) = d j) ∧
      (∀ a : Fin p.nIn, z (dd.inCol (p.width + a)) = xs a) ∧ ∀ i : Fin p.nOut, z (dd.outCol i) = y i) ↔
      y = interpret (decode d) xs := by
  rw [← eval_eq_interpret]
  exact uu_rows_iff hd d xs y

/-- `uu_unit_sound` with the decoded circuit: a typed statement's unit rows, satisfied with `one` at 1, carry on each
output column `interpret (decode d) x` of their description columns `d` and data columns `x`. -/
theorem uu_unit_sound_interpret {types : String → Option CircuitType} {ls : Array Layout}
    {info : String → Option TableInfo} {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array DeriveAll.D}
    (hd : DeriveAll.deriveChecked types ls info words unit = .ok done)
    (ht : (ls.getD unit default).type.bind types = some (uuType p)) (one : ℕ) (z : ℕ → Bool)
    (hz : Dag.SatAt (DeriveAll.blockRow words (done.getD unit default) one) z) (h1 : z one = true) (i : Fin p.nOut) :
    z ((done.getD unit default).outCols.getD i 0) =
      interpret (decode fun j : Fin p.width => z ((done.getD unit default).inCols.getD j 0))
        (fun a => z ((done.getD unit default).inCols.getD (p.width + a) 0)) i :=
  (uu_unit_sound hd ht one z hz h1 i).trans (congrFun (eval_eq_interpret _ _) i)

/-! ## Faulty -/

/-- **A faulty instance**: claimed outputs `y` that differ from the decoded circuit's on the inputs `x`. -/
def Faulty (d : Desc p) (x : Fin p.nIn → Bool) (y : Fin p.nOut → Bool) : Prop := y ≠ interpret (decode d) x

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
theorem not_faulty_of_unit {types : String → Option CircuitType} {ls : Array Layout}
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

/-! ## The argument order -/

/-- **The argument order** (staging option (a)): the sources `op` (`2G` bits), `src` (`2G·K`) and `out` (`N_OUT·K`)
are rows `0`, `1` and `2`, and the bit at offset `j` of a source is row bit `j`, bit `j mod 16` of the row's u16 word
`⌊j/16⌋` under `packed=True` (`descAt_word`). -/
def argOrder (p : Params) (j : Fin p.width) : ℕ × ℕ :=
  if j.val < 2 * p.g then (0, j.val)
  else if j.val < 2 * p.g + 2 * p.g * p.k then (1, j.val - 2 * p.g)
  else (2, j.val - (2 * p.g + 2 * p.g * p.k))

/-- The argument order reads rows `0`, `1` and `2` only. -/
theorem argOrder_lt (j : Fin p.width) : (argOrder p j).1 < 3 := by
  unfold argOrder
  split_ifs <;> simp

/-- **At the argument order, description bit `j` is bit `j mod 16` of u16 word `⌊j/16⌋` of its source's row**, `j`
its offset in the source. -/
theorem descAt_word (q : ℕ → ByteArray) (j : Fin p.width) :
    descAt (argOrder p) q j =
      (word16 (openRow (q (argOrder p j).1)) ((argOrder p j).2 / 16)).testBit ((argOrder p j).2 % 16) :=
  rowBit_eq_word16 _ _

/-- Op bit `t` of gate `g` is bit `2g + t` of row `0`. -/
theorem descAt_op (q : ℕ → ByteArray) (g : Fin p.g) (t : Fin 2) :
    descAt (argOrder p) q (layout p (.inl (g, t))) = rowBit (openRow (q 0)) (2 * g.val + t.val) := by
  have hg := g.isLt
  have ht := t.isLt
  show rowBit (openRow (q (argOrder p _).1)) (argOrder p _).2 = _
  unfold argOrder
  rw [layout_op, ite_eq_left (by omega)]

/-- Bit `j` of gate `g`'s source `s` is bit `(2g + s)·K + j` of row `1`. -/
theorem descAt_src (q : ℕ → ByteArray) (g : Fin p.g) (s : Fin 2) (j : Fin p.k) :
    descAt (argOrder p) q (layout p (.inr (.inl ((g, s), j)))) =
      rowBit (openRow (q 1)) ((2 * g.val + s.val) * p.k + j.val) := by
  have hg := g.isLt
  have hs := s.isLt
  have hj := j.isLt
  have hle : (2 * g.val + s.val + 1) * p.k ≤ 2 * p.g * p.k := Nat.mul_le_mul_right _ (by omega)
  have hsucc : (2 * g.val + s.val + 1) * p.k = (2 * g.val + s.val) * p.k + p.k := by ring
  show rowBit (openRow (q (argOrder p _).1)) (argOrder p _).2 = _
  unfold argOrder
  rw [layout_src, ite_eq_right (by omega), ite_eq_left (by omega)]
  congr 2
  omega

/-- Bit `j` of output `i`'s index is bit `i·K + j` of row `2`. -/
theorem descAt_out (q : ℕ → ByteArray) (i : Fin p.nOut) (j : Fin p.k) :
    descAt (argOrder p) q (layout p (.inr (.inr (i, j)))) = rowBit (openRow (q 2)) (i.val * p.k + j.val) := by
  show rowBit (openRow (q (argOrder p _).1)) (argOrder p _).2 = _
  unfold argOrder
  rw [layout_out, ite_eq_right (by omega), ite_eq_right (by omega)]
  congr 2
  omega

/-- **At the argument order the statement reads rows `0`, `1`, `2` only.** -/
theorem stmt_argOrder_congr {H Ω : Type} (low : Desc p → H) {q q' : ℕ → ByteArray} (h : ∀ i < 3, q i = q' i) (ω : Ω) :
    stmt low (argOrder p) q ω = stmt low (argOrder p) q' ω :=
  stmt_congr low argOrder_lt h ω

/-! ## A faulty session -/

/-- **`rows` open the commit strings `q` at every row `π` reads.** -/
def RowsOpen {n : ℕ} (π : Fin n → ℕ × ℕ) (q rows : ℕ → ByteArray) : Prop := ∀ j, Opens (q (π j).1) (rows (π j).1)

/-- **A faulty session**: some instance `u` whose claimed outputs differ from the decoded circuit's on its inputs under
every opening of the rows they're read from: the description's from the registered commit strings `q` through `π`, the
instance's inputs and claimed outputs from the statement's commit strings `qs` through `ρx u` and `ρy u`. A string that
opens to no row leaves no opening, so an instance read from one is faulty. -/
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
    (hok : rowBits (ρy u) srows = interpret (decode (rowBits π rows)) (rowBits (ρx u) srows)) :
    ¬ ∀ rows srows : ℕ → ByteArray, RowsOpen π q rows → RowsOpen (ρx u) qs srows → RowsOpen (ρy u) qs srows →
      Faulty (rowBits π rows) (rowBits (ρx u) srows) (rowBits (ρy u) srows) :=
  fun h => h rows srows hr hsx hsy hok

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
