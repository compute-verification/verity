import Proofs.Flock.Soundness.Discharge.PrivateCircuit.InstanceDefs

/-!
# Protocol 2 with hidden reads: a padded bag of unit circuits over a committed row store

The full circuit is cut into proof units, each running its class's unit circuit (a template) at its class's padded size;
the hidden program is a **topology table**, committed under a registered root: for each unit its dimensions and the
store cells its read slots read. A **row store**, committed under a second registered root, holds every cell a unit reads
or writes: the registered cells (the full circuit's public and secret inputs) and each unit's output rows. A proof unit's
circuit is its unit circuit plus a read gadget that opens its table entry at its own index, opens each input cell at the index the entry names,
and opens each of its output rows at that row's public index, every opening checked against the registration's roots in
the circuit (`Flock.Registered.opensLeaf`, hm96 leaves).

* **The full circuit** (`Pop`, `Entry`, `Table`, `evalR`): unit `j`'s outputs end to end are its block on the cells its entry
  names, a registered cell `x m` or an earlier unit's output row (`Entry.Acyclic`, which the gadget checks).
* **Against a store** (`UnitOk`, `bad`, `Reaches`, `cone`, `OwedWrongR`): unit `j` is correct against the store `w` when
  its entry is acyclic and its block on `w` at its entry's cells gives `w` at its output rows; `bad T w` is the set of
  units that are not. The registered circuit's owed outputs are wrong (`OwedWrongR`) when the store's value at an owed
  output row is not `evalR`'s on the store's registered cells.
* **Reads** (`ReadVal`, `ReadOk`, `readPair`, `regCommit`): an opening of index `i` of a registered root is a value, its
  salt and its path; it checks when the hm96 leaf of the value under the salt opens the root at `i`. `readPair` is the
  explicit extractor of two openings of one index: the leaf scheme's pair when the leaves agree, else the registered
  reads' (`TwoOpenings.pair`). `regCommit lf n` commits to `n` values under one root (`Commit`), its extractor `readPair`
  at the first index where two openings' values differ.
* **The class circuit's layout** (`Field`, `ReadLayout`, `ClassRel`, `ClassWrongOn`): each unit's gadget holds its entry
  read, its input reads and its output writes at fixed gates of the session's circuit; `tabOf`, `xOf`, `yOf` read them.
  `ClassRel` is the class circuit's relation at those reads under the registration's roots, and `ClassWrongOn` is
  `UnitWrongOn`'s form for it: at the good assignments a unit of the partition is wrong exactly when `ClassRel` fails.
  `ClassComputes` is the class-net fact it follows from (`classWrongOn_of_computes`): each unit's one committed gate is
  its check, set at the good assignments, which the unit's gates compute as the relation.
* **The public class list** (`BlockClass`, `ClassList`): template, dimension class and padded count per
  block, public. It can tell apart circuits whose class lists differ; within one class list only the committed contents
  (dimensions within a class, read positions) are hidden. At hidden unit circuits the class list is size-only
  (`ReadsHidden.lean`).
* **The session's headlines, as Props** (`ZkSessionDrawn`, `DrawEscapes`): protocol 1's drawn bound (no drawn unit wrong
  except with `ε_ks + δ_link`, `zk_session_drawnR`'s form) and the draw's escape at a set fixed before it.
-/

namespace FlockSoundness.Discharge.PrivateCircuit.Reads

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

/-! ## The full circuit: unit circuits over a row store -/

/-- **A store cell**: registered cell `m` (a public or secret input), or output row `l` of unit `i`. -/
inductive Cell where
  | reg (m : ℕ)
  | out (i l : ℕ)
  deriving DecidableEq

/-- **A padded population of unit circuits**: `N` units, each reading `r` cells and writing `o` output rows of values
`V`; unit `j` runs the block `blk j` (its class's unit circuit, so a function of `j`'s class) at the dimensions `d : Dm`
its topology entry holds. -/
structure Pop (V : Type) where
  N : ℕ
  r : ℕ
  o : ℕ
  Dm : Type
  blk : ℕ → Dm → (Fin r → V) → Fin o → V

variable {V : Type}

/-- **A topology entry**: a unit's dimensions and the cell each of its `r` read slots reads. -/
structure Entry (P : Pop V) where
  dims : P.Dm
  src : Fin P.r → Cell

/-- An entry of unit `j` is **acyclic**: every output row it reads is row `l < o` of a unit `i < j`. -/
def Entry.Acyclic {P : Pop V} (e : Entry P) (j : ℕ) : Prop :=
  ∀ k i l, e.src k = .out i l → i < j ∧ l < P.o

/-- **A topology table**: each unit's entry. -/
abbrev Table (P : Pop V) : Type := Fin P.N → Entry P

variable (P : Pop V)

/-- **The full circuit end to end** on the registered cells `x` under the table `T`: unit `j`'s outputs are its block at its
entry's dimensions on its entry's cells, `x m` for a registered cell and the earlier unit's output row for an output
(`default` for an entry that is not acyclic). -/
def evalR [Inhabited V] (T : Table P) (x : ℕ → V) (j : Fin P.N) : Fin P.o → V :=
  P.blk j (T j).dims fun k => match (T j).src k with
    | .reg m => x m
    | .out i l => if h : i < j.val ∧ l < P.o then evalR T x ⟨i, lt_trans h.1 j.isLt⟩ ⟨l, h.2⟩ else default
termination_by j.val
decreasing_by exact h.1

/-- **Unit `j` is correct against the store `w`**: its entry is acyclic, and its block at its entry's dimensions on `w` at
its entry's cells gives `w` at its output rows. -/
def UnitOk (T : Table P) (w : Cell → V) (j : Fin P.N) : Prop :=
  (T j).Acyclic j ∧ ∀ l : Fin P.o, P.blk j (T j).dims (fun k => w ((T j).src k)) l = w (.out j l)

open Classical in
/-- **The units wrong against the store**. -/
noncomputable def bad (T : Table P) (w : Cell → V) : Finset (Fin P.N) :=
  Finset.univ.filter fun j => ¬ UnitOk P T w j

/-- **Unit `j` feeds unit `i`** under `T`: `j = i`, or `j` feeds a unit whose output row one of `i`'s slots reads. -/
inductive Reaches (T : Table P) : Fin P.N → Fin P.N → Prop
  | refl (i : Fin P.N) : Reaches T i i
  | step {j i : Fin P.N} (k : Fin P.r) (i' : Fin P.N) (l : ℕ) (h : (T i).src k = .out i'.val l)
      (hr : Reaches T j i') : Reaches T j i

open Classical in
/-- **The cone of unit `i`**: the units feeding it under `T`, `i` included. -/
noncomputable def cone (T : Table P) (i : Fin P.N) : Finset (Fin P.N) :=
  Finset.univ.filter fun j => Reaches P T j i

/-- **The registered circuit's owed outputs are wrong**: at some owed output row `(i, l)`, the store's value is not the
circuit's end to end on the store's registered cells. -/
def OwedWrongR [Inhabited V] (T : Table P) (w : Cell → V) (owed : Finset (Fin P.N × Fin P.o)) : Prop :=
  ∃ p ∈ owed, w (.out p.1 p.2) ≠ evalR P T (fun m => w (.reg m)) p.1 p.2

/-! ## Reads of a registered root -/

/-- **An opening of one index of a registered root**: a value, its salt and its path. -/
structure ReadVal (α : Type) where
  val : α
  salt : List UInt8
  path : ByteArray

/-- **The opening checks**: the leaf of its value under its salt opens the root `rt` at index `i`. -/
def ReadOk {α : Type} (lf : ExplicitLeaf α) (rt : Flock.Registered.Port) (i : ℕ) (x : ReadVal α) : Prop :=
  Flock.Registered.opensLeaf rt (lf.leaf x.val x.salt).1 i x.path = true

open Classical in
/-- **The extractor of two openings of index `i`**: the leaf scheme's pair when their leaves agree, else the registered
reads' pair of the two leaves and paths. -/
noncomputable def readPair {α : Type} (lf : ExplicitLeaf α) (rt : Flock.Registered.Port) (i : ℕ) (x y : ReadVal α) :
    List UInt8 × List UInt8 :=
  if lf.leaf x.val x.salt = lf.leaf y.val y.salt then lf.pair x.val x.salt y.val y.salt
  else TwoOpenings.pair ⟨rt, i, lf.leaf x.val x.salt, x.path, lf.leaf y.val y.salt, y.path⟩

open Classical in
/-- **The commitment of `n` values under one registered root**: an opening is the root and an opening of each index;
it opens `rt` to `p` when its root is `rt` and each index's opening has value `p i` and checks. The extractor is
`readPair` at the first index where two openings' values differ. -/
noncomputable def regCommit {α : Type} (lf : ExplicitLeaf α) (n : ℕ) :
    Commit Flock.Registered.Port (Fin n → α) (Flock.Registered.Port × (Fin n → ReadVal α)) where
  Opens rt p op := op.1 = rt ∧ ∀ i, (op.2 i).val = p i ∧ ReadOk lf rt i (op.2 i)
  ext op op' :=
    if h : ∃ i, (op.2 i).val ≠ (op'.2 i).val then
      readPair lf op.1 (Classical.choose h).val (op.2 (Classical.choose h)) (op'.2 (Classical.choose h))
    else ([], [])

/-! ## The class circuit's layout -/

/-- **A field of the committed values**: `w` bits at fixed gates of a circuit of `G` gates, decoded. -/
structure Field (G : ℕ) (α : Type) where
  w : ℕ
  gate : Fin w → Fin G
  dec : (Fin w → Bool) → α

/-- The field's value at the assignment `v`: its decoder on the bits at its gates. -/
def Field.read {G : ℕ} {α : Type} (f : Field G α) (v : Fin G → Bool) : α := f.dec fun b => v (f.gate b)

/-- **The class circuit's layout** in a session circuit of `G` gates: where unit `u`'s gadget holds its entry read
(`ent u`), its input read at slot `k` (`inp u k`) and its write of output row `l` (`out u l`). -/
structure ReadLayout (G : ℕ) where
  ent : Fin P.N → Field G (ReadVal (Entry P))
  inp : Fin P.N → Fin P.r → Field G (ReadVal V)
  out : Fin P.N → Fin P.o → Field G (ReadVal V)

namespace ReadLayout

variable {P} {G : ℕ} (L : ReadLayout P G)

/-- Unit `u`'s topology entry, as its gadget holds it at `v`. -/
def tabOf (v : Fin G → Bool) (u : Fin P.N) : Entry P := ((L.ent u).read v).val

/-- Unit `u`'s input at slot `k`, as its gadget holds it at `v`. -/
def xOf (v : Fin G → Bool) (u : Fin P.N) (k : Fin P.r) : V := ((L.inp u k).read v).val

/-- Unit `u`'s output row `l`, as its gadget holds it at `v`. -/
def yOf (v : Fin G → Bool) (u : Fin P.N) (l : Fin P.o) : V := ((L.out u l).read v).val

end ReadLayout

variable {P}

/-- **The class circuit's relation at unit `u`**, at the assignment `v`, under the table root `rtT` and the store root
`rtS`, with the store's public index map `idx`: the entry read opens `rtT` at `u`, and is acyclic; each input read opens
`rtS` at the index of the cell the entry names for its slot; each output write opens `rtS` at the index of `u`'s output
row; and the block at the entry's dimensions on the inputs gives the outputs. -/
def ClassRel {G M : ℕ} (L : ReadLayout P G) (lfT : ExplicitLeaf (Entry P)) (lfS : ExplicitLeaf V) (idx : Cell → Fin M)
    (rtT rtS : Flock.Registered.Port) (v : Fin G → Bool) (u : Fin P.N) : Prop :=
  ReadOk lfT rtT u ((L.ent u).read v) ∧ (L.tabOf v u).Acyclic u ∧
    (∀ k, ReadOk lfS rtS (idx ((L.tabOf v u).src k)) ((L.inp u k).read v)) ∧
    (∀ l : Fin P.o, ReadOk lfS rtS (idx (.out u l)) ((L.out u l).read v)) ∧
    ∀ l, P.blk u (L.tabOf v u).dims (L.xOf v u) l = L.yOf v u l

/-- **The partition's wrong units are the class relation's**: at the assignments `good R` holds of, for the registration
`R` (whose roots are the class circuit's public inputs), unit `u` is wrong exactly when `rel R v u` fails. `UnitWrongOn`'s
form, with the class circuit's relation in place of the universal unit's. -/
def ClassWrongOn {C : Audit.Circuit} {n : ℕ} {Reg : Type} (Pt : Audit.Partition C n)
    (good : Reg → (Fin C.N → Bool) → Prop) (rel : Reg → (Fin C.N → Bool) → Fin n → Prop) : Prop :=
  ∀ R v, good R v → ∀ u, u ∈ Pt.wrong v ↔ ¬ rel R v u

/-- **The class net computes the class relation**, unit by unit (`UniversalComputes`' role for the class circuit). Each
unit `u` holds one committed gate of its own, its check `ok u`, a circuit output: every other gate of `u` is internal.
At the assignments `good R` holds of, every check is set (`v (ok u) = true`, the verifier's public output) and `u`'s
gates applied to the committed values evaluate its check to `true` exactly when `rel R v u` holds. -/
structure ClassComputes {C : Audit.Circuit} {n : ℕ} {Reg : Type} (Pt : Audit.Partition C n)
    (good : Reg → (Fin C.N → Bool) → Prop) (rel : Reg → (Fin C.N → Bool) → Fin n → Prop) (ok : Fin n → Fin C.N) :
    Prop where
  unit_ok : ∀ u, Pt.unit (ok u) = some u
  out_ok : ∀ u, ok u ∈ C.outputs
  only_ok : ∀ g ∈ Pt.committed, ∀ u, Pt.unit g = some u → g = ok u
  set_ok : ∀ R v, good R v → ∀ u, v (ok u) = true
  computes : ∀ R v, good R v → ∀ u, (Pt.localEval v u (ok u) = true ↔ rel R v u)

/-! ## The public class list -/

/-- **A block's public class**: its template (the unit circuit it runs), its dimension class and its padded count (the units of that
class in the padded population). -/
structure BlockClass where
  template : ℕ
  dimClass : ℕ
  padded : ℕ
  deriving DecidableEq

/-- **The public class list**, one `BlockClass` per block. The verifier sees it, so it can tell apart circuits whose
class lists differ; within one class list only the committed contents of the topology table (dimensions within a class,
read positions) are hidden. `ReadsHidden.lean`'s `SizeList` is the size-only form, at committed unit circuits. -/
abbrev ClassList := List BlockClass

/-- The topology table's length at a class list: the padded population, every block's padded count. -/
def ClassList.units (cl : ClassList) : ℕ := (cl.map BlockClass.padded).sum

/-! ## The session's headlines, as Props -/

/-- **Protocol 1's drawn bound at a session `s`**, as a Prop (`zk_session_drawnR`'s form): under A3 and A6, for every live
prover `τ` whose knowledge finders meet `cr/sha-512`, accepting while a drawn unit is wrong in the committed values has
probability at most `ε_ks + δ_link`. -/
def ZkSessionDrawn {C : Audit.Circuit} {n : ℕ} {Pt : Audit.Partition C n} {Reg : Type} (s : UnitSession Pt Reg)
    (drawn : s.Out → Finset (Fin n)) : Prop :=
  Assumptions.UniformRandomBytes s.Lb s.src → s.A6 → ∀ τ : Strategy s.G, s.CR τ →
    prob (fun o => s.Accepts o ∧ ∃ u ∈ drawn o, u ∈ Pt.wrong (s.vals τ o)) s.G τ ≤ s.εks τ + s.δlink τ

/-- **The draw escapes a set fixed before it with at most the law's escape**: for every prover `τ` and every set `B`
that reads only the registration, accepting while the draw misses `B` of the outcome's registration has probability at
most the largest escape of the law at `B R`. -/
def DrawEscapes {C : Audit.Circuit} {n : ℕ} {Pt : Audit.Partition C n} {Reg : Type} (s : UnitSession Pt Reg)
    (drawn : s.Out → Finset (Fin n)) : Prop :=
  ∀ (τ : Strategy s.G) (B : Reg → Finset (Fin n)),
    prob (fun o => s.Accepts o ∧ Disjoint (drawn o) (B (s.regOf o))) s.G τ ≤ ⨆ R, s.Lm.escape (B R)

end FlockSoundness.Discharge.PrivateCircuit.Reads
