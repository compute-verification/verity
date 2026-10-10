import Mathlib.Data.Fintype.Sigma
import Proofs.Flock.Soundness.Audit.Law
import Proofs.Flock.Soundness.Game.Prob
import Proofs.Flock.Verifier.Registered
import VStar.Stage

/-!
# Sampling with prover-chosen batching, the partition hidden: the model

The trace is committed under one registered root, a record per unit at the unit's position, and the hidden partition
gives each unit its type. The auditor draws units over the padded indices by a law (`FlockSoundness.Audit.Law`). The
prover batches the drawn units into the slots of sessions whose shapes come from a public menu: each session selects its
type's matrix by a hidden index, padded to its shape's nonzero bound, and is one C-Flock session, which F1's recursion
proves. The auditor sees only the sessions' shapes, and checks them against the menu's padded sessions for the draw's
size. That each drawn unit sits in exactly one slot is proved inside the outer proof, against the committed partition,
and each slot is bound to its unit's committed record by its opening of the root.

* `Shape`, `Menu`: a session's block size and nonzero bound; the menu's shapes, the inner shape V* is staged over for
  each (`Menu.vstar`, Lean's `VStar.statements`), and the padded sessions of an audit of `k` draws (`Menu.padded`).
* `Trace`, `Types`, `wrong`: the committed trace, partial (it holds nothing at a position no record opens); each type's
  relation and its matrix's nonzeros; the units wrong in a trace.
* `Batching`, `Hidden`: the prover's sessions, with their hidden types and slots; and those with each slot's opening.
* `CoverCheck`, `Bound`, `Covers`: the outer proof's coverage check; each held slot on its unit's committed type and
  values; each drawn unit in exactly one slot, on them.
* `Commit`, `Commit.Committed`, `Commit.SlotOpens`, `recordPair`, `SlotBreak`: the trace commitment; the trace it
  commits; the slots' openings; the SHA-512 inputs two openings at one position give; slot binding's break.
* `Proves`, `Accepts`, `sampledAudit`, `Reaches`, `Recursion`: the statement the sessions prove; the auditor's verdict;
  F2's audit; an outcome a run reaches; and F1's conclusion as F2 uses it, an interface.
-/

namespace FlockSampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

/-! ## The menu -/

/-- A session's shape, all the auditor sees of it: its block size (the slots it holds) and its nonzero bound (the size
its type's matrix is padded to). -/
structure Shape where
  block : ℕ
  nnz : ℕ
  deriving DecidableEq, Repr

/-- The menu. `shapes`: the shapes a session may take; `inner s`: the inner session's shape V* is staged over for a
session of shape `s`; `padded k`: the sessions an audit of `k` drawn units runs, by shape and in order, a public function
of `k`. -/
structure Menu where
  shapes : List Shape
  inner : Shape → VStar.Algebra.Shape
  padded : ℕ → List Shape

/-- V*'s statements over a session of shape `s`: Lean's `VStar.statements` at its inner shape. -/
def Menu.vstar (M : Menu) (s : Shape) : Except String (Array VStar.Stmt) := VStar.statements (M.inner s)

/-- `true` at a result, `false` at a refusal. -/
def isOk {ε α : Type} : Except ε α → Bool
  | .ok _ => true
  | .error _ => false

/-- V* stages every shape on the menu, as the code computes it. -/
def Menu.staged (M : Menu) : Bool := M.shapes.all fun s => isOk (M.vstar s)

/-! ## The trace and its types -/

/-- The committed trace over `N` padded indices: each unit's type, from the hidden partition, and its values; `none` at
a position no record opens. -/
abbrev Trace (N : ℕ) (T V : Type) := Fin N → Option (T × V)

/-- The unit types: each one's relation on a unit's values, and its matrix's nonzero count. -/
structure Types (T V : Type) where
  holds : T → V → Prop
  nnz : T → ℕ

open Classical in
/-- The units wrong in a trace: those with no record, and those whose values their type's relation refuses. -/
noncomputable def wrong {N : ℕ} {T V : Type} (Ty : Types T V) (X : Trace N T V) : Finset (Fin N) :=
  Finset.univ.filter fun i => ∀ t v, X i = some (t, v) → ¬ Ty.holds t v

/-! ## The prover's batching -/

/-- The prover's sessions: `B` of them, each with its public shape, its hidden type (the matrix it selects) and, in each
slot of its block, the unit it holds (`none`: padding) and the values it carries. -/
structure Batching (N : ℕ) (T V : Type) where
  B : ℕ
  shape : Fin B → Shape
  typ : Fin B → T
  idx : (b : Fin B) → Fin (shape b).block → Option (Fin N)
  val : (b : Fin B) → Fin (shape b).block → V

namespace Batching

variable {N : ℕ} {T V : Type}

/-- A slot: a session, and a position in its block. -/
abbrev Pos (β : Batching N T V) : Type := (b : Fin β.B) × Fin (β.shape b).block

/-- The unit slot `p` holds. -/
def held (β : Batching N T V) (p : β.Pos) : Option (Fin N) := β.idx p.1 p.2

/-- The values slot `p` carries. -/
def slotVal (β : Batching N T V) (p : β.Pos) : V := β.val p.1 p.2

/-- The sessions' shapes, in order: what the auditor sees. -/
def view (β : Batching N T V) : List Shape := (List.finRange β.B).map β.shape

end Batching

/-- What the sessions hold hidden: the batching, and each slot's opening of the root (its path). -/
structure Hidden (N : ℕ) (T V : Type) where
  β : Batching N T V
  path : β.Pos → ByteArray

/-! ## Coverage -/

/-- **The outer proof's coverage check**, made against the committed hidden partition: every held slot holds a drawn
unit, no unit is held twice, and as many slots are held as units are drawn. -/
structure CoverCheck {N : ℕ} {T V : Type} (S : Finset (Fin N)) (β : Batching N T V) : Prop where
  drawn : ∀ p i, β.held p = some i → i ∈ S
  once : ∀ p q i, β.held p = some i → β.held q = some i → p = q
  count : (Finset.univ.filter fun p : β.Pos => (β.held p).isSome = true).card = S.card

/-- Each held slot is on its unit's committed type and values. -/
def Bound {N : ℕ} {T V : Type} (X : Trace N T V) (β : Batching N T V) : Prop :=
  ∀ p i, β.held p = some i → X i = some (β.typ p.1, β.slotVal p)

/-- Each drawn unit sits in exactly one slot, of its committed type, on its committed values. -/
def Covers {N : ℕ} {T V : Type} (X : Trace N T V) (S : Finset (Fin N)) (β : Batching N T V) : Prop :=
  ∀ i ∈ S, ∃! p : β.Pos, β.held p = some i ∧ X i = some (β.typ p.1, β.slotVal p)

/-! ## The trace commitment and slot binding -/

/-- The trace commitment: the registered root `rt`, each unit's record `enc t v` (the commit string whose hm96 tree leaf
the root's tree holds, a unit's type and values), and each unit's position in the tree. -/
structure Commit (N : ℕ) (T V : Type) where
  rt : Flock.Registered.Port
  enc : T → V → ByteArray
  pos : Fin N → ℕ

namespace Commit

variable {N : ℕ} {T V : Type}

/-- A record determines the type and values it encodes. -/
def Injective (C : Commit N T V) : Prop := ∀ t v t' v', C.enc t v = C.enc t' v' → t = t' ∧ v = v'

/-- `X` is a trace committed under `C`'s root, with the openings `path₀`: each unit `X` holds opens the root at its
position, and `X` holds a unit wherever some record opens. It is an analysis device, fixed by the root alone, so before
the draw. -/
structure Committed (C : Commit N T V) (X : Trace N T V) (path₀ : Fin N → ByteArray) : Prop where
  opens : ∀ i t v, X i = some (t, v) → Flock.Registered.opens C.rt (C.enc t v) (C.pos i) (path₀ i) = true
  full : ∀ i t v path, Flock.Registered.opens C.rt (C.enc t v) (C.pos i) path = true → (X i).isSome = true

/-- The outer proof's slot openings: each held slot's record opens the root at its unit's position. -/
def SlotOpens (C : Commit N T V) (h : Hidden N T V) : Prop :=
  ∀ p i, h.β.held p = some i →
    Flock.Registered.opens C.rt (C.enc (h.β.typ p.1) (h.β.slotVal p)) (C.pos i) (h.path p) = true

end Commit

/-- The SHA-512 inputs that two records' openings at one position give: their tree leaves' inputs when the leaves
agree, and the two leaves' openings' pair (`Flock.Registered.openPair`) otherwise. -/
def recordPair (rt : Flock.Registered.Port) (pos : ℕ) (x path y path' : ByteArray) : ByteArray × ByteArray :=
  if Flock.Hm96.Default512.treeLeaf x = Flock.Hm96.Default512.treeLeaf y then
    (Flock.Hm96.Default512.leafPrefix ++ x, Flock.Hm96.Default512.leafPrefix ++ y)
  else
    Flock.Registered.openPair rt pos (Flock.Hm96.Default512.treeLeaf x) path (Flock.Hm96.Default512.treeLeaf y) path'

/-- **Slot binding's break**, tied to the run: a held slot whose opening and the committed trace's at its unit's
position give two different SHA-512 inputs with one digest. -/
def SlotBreak {N : ℕ} {T V : Type} (C : Commit N T V) (X : Trace N T V) (path₀ : Fin N → ByteArray)
    (h : Hidden N T V) : Prop :=
  ∃ p i t v, h.β.held p = some i ∧ X i = some (t, v) ∧
    Flock.Registered.Collides
      (recordPair C.rt (C.pos i) (C.enc (h.β.typ p.1) (h.β.slotVal p)) (h.path p) (C.enc t v) (path₀ i))

/-! ## The audit -/

/-- **The statement the sessions prove**, about what they hold hidden: the coverage check against the draw `S`, each held
slot's opening of the root, each session's type within its shape's nonzero bound, and each held slot's values satisfying
its session's type. -/
structure Proves {N : ℕ} {T V : Type} (Ty : Types T V) (C : Commit N T V) (S : Finset (Fin N)) (h : Hidden N T V) :
    Prop where
  cover : CoverCheck S h.β
  opens : C.SlotOpens h
  fits : ∀ b, Ty.nnz (h.β.typ b) ≤ (h.β.shape b).nnz
  holds : ∀ p i, h.β.held p = some i → Ty.holds (h.β.typ p.1) (h.β.slotVal p)

/-- **The auditor's verdict** on an outcome at the draw `S`: the sessions' shapes are the menu's padded sessions for
`S`'s size, and every session verifies. It reads the outcome only through `view` and `verifies`. -/
def Accepts {N : ℕ} {O : Type} (M : Menu) (view : O → List Shape) (verifies : O → Bool) (S : Finset (Fin N))
    (o : O) : Prop :=
  view o = M.padded S.card ∧ verifies o = true

/-- **F2's audit.** The prover registers `R` (its trace root among it); the auditor draws `S` by `L`; the sessions on the
drawn units run against `R` and end in an outcome. Output: the outcome and the draw. -/
def sampledAudit {N : ℕ} (L : Law N) {Reg O : Type} (session : Finset (Fin N) → Reg → Game O) :
    Game (O × Finset (Fin N)) :=
  .send Reg fun R => .coin L.Ω fun ω => (session (L.draw ω) R).map fun o => (o, L.draw ω)

section Strategies

variable {N : ℕ} {L : Law N} {Reg O : Type} {session : Finset (Fin N) → Reg → Game O}

/-- The registration of an audit strategy. -/
def regOf (σ : Strategy (sampledAudit L session)) : Reg := σ.1

/-- The prover's session strategy at each draw. -/
def contOf (σ : Strategy (sampledAudit L session)) (ω : L.Ω) : Strategy (session (L.draw ω) (regOf σ)) :=
  Strategy.ofMap _ _ (σ.2 ω)

end Strategies

/-- An outcome a run reaches: one of nonzero probability. -/
def Reaches {O : Type} (g : Game O) (τ : Strategy g) (o : O) : Prop := prob (fun o' => o' = o) g τ ≠ 0

/-- **F1's conclusion, as F2 uses it**: an interface, not F1's statement. `verifies`: the sessions' verdict on an outcome;
`hid`: what they hold hidden behind it, as the extractor recovers it (an analysis device). At every draw `S` and
registration `R`, for every prover of the sessions: a break of its runs (`Break`: a fork, or a read commit string whose
opening collides with the registered one's), or the sessions verify while the statement they prove fails with
probability at most `ε`. F1's `RecursiveAudit` (i), at each session's inner statement over V*'s statements for its
shape, is to instantiate it. -/
structure Recursion {N : ℕ} {T V Reg O : Type} (session : Finset (Fin N) → Reg → Game O) (Ty : Types T V)
    (com : Reg → Commit N T V) where
  verifies : O → Bool
  hid : Finset (Fin N) → Reg → O → Hidden N T V
  Break : (S : Finset (Fin N)) → (R : Reg) → Strategy (session S R) → Prop
  ε : (S : Finset (Fin N)) → (R : Reg) → Strategy (session S R) → ℝ≥0∞
  sound : ∀ S R (τ : Strategy (session S R)), Break S R τ ∨
    prob (fun o => verifies o = true ∧ ¬ Proves Ty (com R) S (hid S R o)) (session S R) τ ≤ ε S R τ

end FlockSampled
