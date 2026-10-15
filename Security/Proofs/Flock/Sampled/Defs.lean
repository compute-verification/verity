import Mathlib.Data.Finset.Sort
import Proofs.Flock.Soundness.Audit.Law
import Proofs.Flock.Soundness.Game.Prob
import Flock.Statement
import Proofs.Flock.Verifier.Registered
import VStar.Stage

/-!
# Sampling with prover-chosen batching, the partition hidden: the model

The trace is committed under one registered root, a record per unit at the unit's position holding the unit's type (from
the hidden partition) and its values. The auditor draws units over the padded indices by a law
(`FlockSoundness.Audit.Law`). The prover batches the drawn units into the slots of sessions whose shapes come from a
public menu: each slot selects its unit's type's matrix by a hidden index into its session's type table, which holds at
most the shape's public type bound and is padded up to it, each matrix padded to the shape's nonzero bound; each session
is one C-Flock session, which F1's recursion proves. The auditor sees only the sessions' shapes, and checks them against
the menu's padded sessions for the draw's size. That each drawn unit sits in exactly one slot is checked once, globally,
in the outer proof: the slots' records pass up V*'s levels, and the top statement routes them into the draw's order. Each
slot is bound to its unit's committed record by its opening of the root.

* `Shape`, `Menu`, `Menu.Staged`, `Menu.InRange`, `Menu.Within`: a session's slots and nonzero bound; the menu's shapes,
  the inner shape V* is staged over for each (`Menu.vstar`, Lean's `VStar.statements`), each shape's type bound and the
  padded sessions of an audit of `k` draws (`Menu.padded`); the obligations that V* stages the menu and that the live
  verifier's range check accepts it (`Menu.rangeCheck`); a placement of types at the menu's shapes within its bounds.
* `Trace`, `Types`, `Types.FitsIn`, `wrong`: the committed trace, partial (it holds nothing at a position no record
  opens); each type's relation, its matrix's nonzeros and its unit's template instances; a type within a shape's nonzero
  bound and unit positions; the units wrong in a trace.
* `Batching`, `Hidden`, `Hidden.honest`: the prover's sessions, with their slots' hidden types; those with each slot's
  opening; and the honest prover's openings.
* `drawOrder`, `CoverCheck`, `Bound`, `Covers`: the drawn units in order; the outer proof's coverage check; each held
  slot on its unit's committed type and values; each drawn unit in exactly one slot, on them.
* `Commit`, `Commit.Injective`, `Commit.Committed`, `Commit.SlotOpens`, `recordPair`, `SlotBreak`: the trace commitment;
  the record's format; the trace it commits; the slots' openings; the SHA-512 inputs two openings at one position give;
  slot binding's break.
* `Proves`, `Accepts`, `sampledAudit`, `Reaches`, `Public`, `reveal`, `Recursion`: the statement the sessions prove; the
  auditor's verdict; F2's audit at a registration; an outcome a run reaches; the reveal list and an honest audit's; and
  F1's conclusion as F2 uses it, an interface, its soundness and its zero knowledge.
-/

namespace FlockSampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

/-! ## The menu -/

/-- A session's shape, all the auditor sees of it: its slots (F6's `B`: the units a session proves, each slot holding any
unit within the bounds), its nonzero bound (F6's `Z`: the size each matrix of its type table is padded to), and the unit
positions each slot has (U_σ's, F6's `instances_max`). Every slot of the session has all its positions, whatever its type:
its unit's template instances fill the first of them and dummy copies of the unit, satisfiable and wired to nothing, the
rest. So the layout V* reads per slot is the menu entry's alone, and public. -/
structure Shape where
  slots : ℕ
  nnz : ℕ
  positions : ℕ
  deriving DecidableEq, Repr

/-- The menu, public and fixed per model, never per audit. `shapes`: the shapes a session may take; `inner s`: the inner
session's shape V* is staged over for a session of shape `s` (a pinned menu's is `VStar.Menu.inner`, in
`Menu.ofEntries`); `types s`: the public bound on the types a session of shape `s` selects among
(its type table's size, the real count padded up to it); `padded k`: the sessions an audit of `k` drawn units runs, by
shape and in order, a function of `k` and the menu alone. -/
structure Menu where
  shapes : List Shape
  inner : Shape → VStar.Algebra.Shape
  types : Shape → ℕ
  padded : ℕ → List Shape

/-- V*'s statements over a session of shape `s`: Lean's `VStar.statements` at its inner shape. -/
def Menu.vstar (M : Menu) (s : Shape) : Except String (Array VStar.Stmt) := VStar.statements (M.inner s)

/-- `true` at a result, `false` at a refusal. -/
def isOk {ε α : Type} : Except ε α → Bool
  | .ok _ => true
  | .error _ => false

/-- V* stages every shape on the menu, as the code computes it. -/
def Menu.staged (M : Menu) : Bool := M.shapes.all fun s => isOk (M.vstar s)

/-- Obligation: V* stages every shape on the menu, `VStar.statements` at its inner shape a result and not a refusal. A
fact about our own code (V*'s builder), not about the world: `#guard menu₀.staged` tests it at one menu
(`Proofs.Flock.Sampled.NonVacuity`), and a proof is to come from `VStar.statements`. Not a listed guarantee's premise
yet. -/
def Menu.Staged (M : Menu) : Prop := ∀ s ∈ M.shapes, ∃ vs, M.vstar s = .ok vs

/-- The live verifier's range check (`Flock.checkInRange`, which `setupH` and `setupTables` make on every hm96 statement) on the inner
session of a session of shape `s`, read off its inner shape: the block size `k_log`; the public regions, two claims each
past the two every session opens (`Stmt.nClaims`); and a link point's coordinates, `PT_LOCAL` and the table's block bits
`m − k_log`. -/
def Menu.rangeCheck (M : Menu) (s : Shape) : Except String Unit :=
  Flock.checkInRange (M.inner s).kLog (((M.inner s).claims - 2) / 2) (Flock.PT_LOCAL + ((M.inner s).m - (M.inner s).kLog))

/-- Obligation: the live verifier accepts every shape on the menu, its range check on the inner session a result and not
a refusal (`Flock.checkInRange_ok`: `k_log ≤ 27`, at most 1024 regions, `m_pts ≤ 64`, the range the proved bound
covers). V*'s builder doesn't check it (its refusals are `m`'s schedule, `k_log < 6` and fewer than two claims), and
`setup` refuses every session at a shape outside it, the honest prover's too: it is what limits a shape's nonzero bound,
which the menu leaves a parameter. A fact about the menu and our own code; `menu₀_inRange` proves it at one menu. -/
def Menu.InRange (M : Menu) : Prop := ∀ s ∈ M.shapes, M.rangeCheck s = .ok ()

/-! ## The trace and its types -/

/-- The committed trace over `N` padded indices: each unit's type, from the hidden partition, and its values; `none` at
a position no record opens. -/
abbrev Trace (N : ℕ) (T V : Type) := Fin N → Option (T × V)

/-- The unit types: each one's relation on a unit's values, its matrix's nonzero count, and the template instances a unit
of it fills (F6's `inst`). -/
structure Types (T V : Type) where
  holds : T → V → Prop
  nnz : T → ℕ
  instances : T → ℕ

/-- **A type fits a shape**: its matrix within the shape's nonzero bound, and its unit's instances within the shape's unit
positions, the rest of which hold dummy copies of the unit. -/
structure Types.FitsIn {T V : Type} (Ty : Types T V) (t : T) (s : Shape) : Prop where
  nnz : Ty.nnz t ≤ s.nnz
  instances : Ty.instances t ≤ s.positions

open Classical in
/-- The units wrong in a trace: those with no record, and those whose values their type's relation refuses. -/
noncomputable def wrong {N : ℕ} {T V : Type} (Ty : Types T V) (X : Trace N T V) : Finset (Fin N) :=
  Finset.univ.filter fun i => ∀ t v, X i = some (t, v) → ¬ Ty.holds t v

/-- **A placement within the menu's bounds** of the types `U`: each placed at a shape on the menu it fits, and at most
`M.types s` of them at each shape `s`. The placement is hidden; only the bounds are public. -/
def Menu.Within {T V : Type} (M : Menu) (Ty : Types T V) (U : Finset T) (place : T → Shape) : Prop :=
  (∀ t ∈ U, place t ∈ M.shapes ∧ Ty.FitsIn t (place t)) ∧ ∀ s, (U.filter fun t => place t = s).card ≤ M.types s

/-! ## The prover's batching -/

/-- The prover's sessions, each with its public shape and, in each of its slots, the unit it holds (`none`: padding),
that slot's hidden type (the matrix it selects) and the values it carries. -/
structure Batching (N : ℕ) (T V : Type) where
  sessions : ℕ
  shape : Fin sessions → Shape
  typ : (b : Fin sessions) → Fin (shape b).slots → T
  idx : (b : Fin sessions) → Fin (shape b).slots → Option (Fin N)
  val : (b : Fin sessions) → Fin (shape b).slots → V

namespace Batching

variable {N : ℕ} {T V : Type}

/-- A slot: a session, and a position among its slots. -/
abbrev Pos (β : Batching N T V) : Type := (b : Fin β.sessions) × Fin (β.shape b).slots

/-- The unit slot `p` holds. -/
def held (β : Batching N T V) (p : β.Pos) : Option (Fin N) := β.idx p.1 p.2

/-- The type slot `p` selects. -/
def slotTyp (β : Batching N T V) (p : β.Pos) : T := β.typ p.1 p.2

/-- The values slot `p` carries. -/
def slotVal (β : Batching N T V) (p : β.Pos) : V := β.val p.1 p.2

/-- The sessions' shapes, in order: what the auditor sees. -/
def view (β : Batching N T V) : List Shape := (List.finRange β.sessions).map β.shape

open Classical in
/-- The types session `b`'s held slots select: what its type table must hold. -/
noncomputable def table (β : Batching N T V) (b : Fin β.sessions) : Finset T :=
  (Finset.univ.filter fun j : Fin (β.shape b).slots => (β.idx b j).isSome = true).image (β.typ b)

end Batching

/-- What the sessions hold hidden: the batching, and each slot's opening of the root (its path). -/
structure Hidden (N : ℕ) (T V : Type) where
  β : Batching N T V
  path : β.Pos → ByteArray

/-- The honest prover's openings of a batching: each held slot's at its unit's committed path in `path₀`. -/
def Hidden.honest {N : ℕ} {T V : Type} (β : Batching N T V) (path₀ : Fin N → ByteArray) : Hidden N T V :=
  ⟨β, fun p => (β.held p).elim ByteArray.empty path₀⟩

/-! ## Coverage -/

/-- The drawn units in the draw's order, then dummies (`none`): position `j` holds `S`'s `j`-th unit in increasing
order, the order the auditor fixes from its own draw. -/
def drawOrder {N : ℕ} (S : Finset (Fin N)) (j : ℕ) : Option (Fin N) :=
  if h : j < S.card then some (S.orderEmbOfFin rfl ⟨j, h⟩) else none

/-- **The outer proof's coverage check**, made once, globally, against the committed hidden partition, with no coin:
every session outputs its slots' records, which pass up V*'s levels; the top statement routes them all through a
permutation network whose switches the prover sets (`route`, a Beneš network's permutation of the slots), then checks
that position `j` holds the draw's `j`-th unit for each `j < |S|`, and a dummy after (`drawOrder`). A switch network only
permutes, so each drawn unit is held exactly once, and the prover still picks the batching. -/
def CoverCheck {N : ℕ} {T V : Type} (S : Finset (Fin N)) (β : Batching N T V) : Prop :=
  ∃ (n : ℕ) (route : Fin n ≃ β.Pos), S.card ≤ n ∧ ∀ j : Fin n, β.held (route j) = drawOrder S j

/-- Each held slot is on its unit's committed type and values. -/
def Bound {N : ℕ} {T V : Type} (X : Trace N T V) (β : Batching N T V) : Prop :=
  ∀ p i, β.held p = some i → X i = some (β.slotTyp p, β.slotVal p)

/-- Each drawn unit sits in exactly one slot, of its committed type, on its committed values. -/
def Covers {N : ℕ} {T V : Type} (X : Trace N T V) (S : Finset (Fin N)) (β : Batching N T V) : Prop :=
  ∀ i ∈ S, ∃! p : β.Pos, β.held p = some i ∧ X i = some (β.slotTyp p, β.slotVal p)

/-! ## The trace commitment and slot binding -/

/-- The trace commitment: the registered root `rt`, each unit's record `enc t v` (the commit string whose hm96 tree leaf
the root's tree holds: the unit's type, in a fixed-width field, and its values, salted), and each unit's position in the
tree. -/
structure Commit (N : ℕ) (T V : Type) where
  rt : Flock.Registered.Port
  enc : T → V → ByteArray
  pos : Fin N → ℕ

namespace Commit

variable {N : ℕ} {T V : Type}

/-- **The record's format**, the type in the leaf: a record determines the type and values it encodes, as its
fixed-width type field before the values gives, so one opening yields exactly one `(t, v)`. -/
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
    Flock.Registered.opens C.rt (C.enc (h.β.slotTyp p) (h.β.slotVal p)) (C.pos i) (h.path p) = true

end Commit

/-- The SHA-512 inputs that two records' openings at one position give: their tree leaves' inputs when the leaves
agree, and the two leaves' openings' pair (`Flock.Registered.openPair`) otherwise. F1's `bcPair` restated (same `if`),
which replaces it when F1 lands. -/
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
      (recordPair C.rt (C.pos i) (C.enc (h.β.slotTyp p) (h.β.slotVal p)) (h.path p) (C.enc t v) (path₀ i))

/-! ## The audit -/

/-- **The statement the sessions prove**, about what they hold hidden: the coverage check against the draw `S`, each held
slot's opening of the root, each held slot's type fitting its shape (`Types.FitsIn`), and each held slot's values in its
type's relation (`holds`: the slot check, candidate 3, the bit-committed sumcheck at w = 1, over the shape's unit
positions with dummy copies in those the type doesn't fill; kept abstract here). -/
structure Proves {N : ℕ} {T V : Type} (Ty : Types T V) (C : Commit N T V) (S : Finset (Fin N)) (h : Hidden N T V) :
    Prop where
  cover : CoverCheck S h.β
  opens : C.SlotOpens h
  fits : ∀ p i, h.β.held p = some i → Ty.FitsIn (h.β.slotTyp p) (h.β.shape p.1)
  holds : ∀ p i, h.β.held p = some i → Ty.holds (h.β.slotTyp p) (h.β.slotVal p)

/-- **The auditor's verdict** on an outcome at the draw `S`: the sessions' shapes are the menu's padded sessions for `S`'s
size, and every session verifies. It reads the outcome only through `view` and `verifies`. -/
def Accepts {N : ℕ} {O : Type} (M : Menu) (view : O → List Shape) (verifies : O → Bool) (S : Finset (Fin N))
    (o : O) : Prop :=
  view o = M.padded S.card ∧ verifies o = true

/-- **F2's audit at a registration**, one game per registration with its sessions in the continuation: the prover has
registered `R` (its trace root among it); the auditor draws `S` by `L`; the sessions on the drawn units run against `R`.
Output: the outcome and the draw. -/
def sampledAudit {N : ℕ} (L : Law N) {Reg O : Type} (session : Finset (Fin N) → Reg → Game O) (R : Reg) :
    Game (O × Finset (Fin N)) :=
  .coin L.Ω fun ω => (session (L.draw ω) R).map fun o => (o, L.draw ω)

/-- The prover's session strategy at each draw. -/
def contOf {N : ℕ} {L : Law N} {Reg O : Type} {session : Finset (Fin N) → Reg → Game O} {R : Reg}
    (σ : Strategy (sampledAudit L session R)) (ω : L.Ω) : Strategy (session (L.draw ω) R) :=
  Strategy.ofMap _ _ (σ ω)

/-- **An outcome a run reaches**: `τ` plays `g` to `o` at some coins. F1's `Game.Outcome` restated, which replaces it
when F1 lands. -/
def Reaches {O : Type} : (g : Game O) → Strategy g → O → Prop
  | .ret b, _, o => b = o
  | .send _ k, τ, o => Reaches (k τ.1) τ.2 o
  | @Game.coin _ _ _ _ k, τ, o => ∃ c, Reaches (k c) (τ c) o

/-- **The reveal list** at a draw, all F2's zero-knowledge simulator reads: the menu (its shapes, nonzero bounds and type
bounds), the draw (the auditor's own coins, so `k = |S|`), the sessions' shapes in order, and the owed public outputs. -/
structure Public (N : ℕ) (Out : Type) where
  menu : Menu
  draw : Finset (Fin N)
  shapes : List Shape
  out : Out

/-- An honest audit's reveal list at the draw `S`: its sessions' shapes are the menu's padded sessions for `|S|`. -/
def reveal {N : ℕ} {Out : Type} (M : Menu) (S : Finset (Fin N)) (out : Out) : Public N Out :=
  ⟨M, S, M.padded S.card, out⟩

/-- **F1's conclusion, as F2 uses it**: an interface, not F1's statement (`PublicAudit`, or `RecursiveAudit` at
registered units).

(i), soundness: `verifies`, the sessions' verdict on an outcome; `hid`, what they hold hidden behind it, as the extractor
recovers it (an analysis device). At every registration `R` and draw `S`, for every prover of the sessions: a break of
its runs (`Break`: F1's forks, or a read commit string whose opening collides with the registered one's), or the sessions
verify while the statement they prove fails with probability at most `ε`.

(ii), zero knowledge (`RecursiveFirewallPrivate`, being restated at `hidingGap`; its form is F1's, not pinned here):
`out R`, the owed outputs the registration names; `real x R h`, the auditor's view of the honest sessions at the reveal
list `x`, on the registration `R` (its root among it) and the hidden `h`; `sim x`, the simulated view, reading `x` alone;
`hides`, that the two are close (`Close`, F1's notion and gap) at every honest witness: the owed outputs `R`'s, the
sessions at `x`'s shapes, proving the statement at `x`'s draw. -/
structure Recursion {N : ℕ} {T V Reg O : Type} (session : Finset (Fin N) → Reg → Game O) (Ty : Types T V)
    (com : Reg → Commit N T V) where
  verifies : O → Bool
  hid : Finset (Fin N) → Reg → O → Hidden N T V
  Break : (S : Finset (Fin N)) → (R : Reg) → Strategy (session S R) → Prop
  ε : (S : Finset (Fin N)) → (R : Reg) → Strategy (session S R) → ℝ≥0∞
  sound : ∀ S R (τ : Strategy (session S R)), Break S R τ ∨
    prob (fun o => verifies o = true ∧ ¬ Proves Ty (com R) S (hid S R o)) (session S R) τ ≤ ε S R τ
  Out : Type
  out : Reg → Out
  View : Type
  real : Public N Out → Reg → Hidden N T V → View
  sim : Public N Out → View
  Close : Public N Out → View → View → Prop
  hides : ∀ (x : Public N Out) (R : Reg) (h : Hidden N T V), x.out = out R → x.shapes = h.β.view →
    Proves Ty (com R) x.draw h → Close x (real x R h) (sim x)

end FlockSampled
