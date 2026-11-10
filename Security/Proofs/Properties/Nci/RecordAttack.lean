module

public import Proofs.Properties.Nci.Record

/-!
# Instances against the circuit record's properties

A scratch file of instances, never merged: what the statements of `Properties.Nci.Record` say and leave out.

* **A split leaf is refused.** Under `Aligned` a writer's leaf that meets a leaf its reader passes on lies inside one
  leaf of the row (`aligned_inside_of_passed`), so a wire whose writer has a leaf wider than every leaf of the row that
  reads it, one of them passed on where it meets that leaf, is not `Aligned` (`not_aligned_of_wider`), and no circuit
  is `Served`. Where no unit passes a gate on, `Aligned` asks nothing (`aligned_of_not_passed`): the widths are then
  the count's to hold.
* **A wire from a writer the Program has no unit for.** `Served` holds the wiring's templates to the Program by their
  ids alone, and `Spot.Wires.check` bounds a wire's writer by the units the wiring declares. Where a row has a writer
  and that writer is no unit of the Program, or has no such output row, the row's gates are wired and a read of one
  lands on no gate (`lands_nowhere`): the wired circuit reads nothing for it. So does a read of a wired leaf past
  every bit of its writer's row (`lands_nowhere_past`).
* **A held row that has a writer** puts no gate in θ (`held_wired_names_nothing`): the wired circuit has none of its
  gates. `Good` asks that an opening's root and domain rows be held, and never that they have no writer.
* **The pins and the record's text.** `Carries`, the four premises of the rate rule and `Complies` are the same
  statements whatever pins the check reports (`carries_repin`, `owns_repin`, `complies_repin`), and `Served` is the
  same whatever record text it reports (`served_retext`).
* **The premises on the check hold of a check that reads nothing** (`refuseAll_premises`): they are consistent, and
  that an accepting check meets them is not shown here.
* **A leaf handed on by reference is passed on.** `PassedOn` is by the gate a return's leaf resolves to. Where the
  walk from that leaf goes on to the port's reference (`Reaches`), the two are one gate whatever the fuel
  (`same_gate`), so the unit passes the port's gate on (`passedOn_of_reaches`). At a toy Program (`Toy`) the kernel
  runs the walk: a parameter returned through a callee's return (`Toy.call_out`, `Toy.call_passedOn`) and through a
  batch member (`Toy.batch_out1`, `Toy.batch_passedOn`) is the port's gate, and a copy the unit computes is another
  gate (`Toy.copy_differs`). The steps for any Program are `step_call`, `step_param`, `step_batch`, `step_member`.
* **A computed copy.** A gate that is no input gate is under no wire (`not_wired_of_computed`), and a read of a gate
  under no wire lands on that gate alone (`lands_eq_of_not_wired`): a reader of a copy reads the copy. Through a wire
  whose writer's leaves are under no wire, a read lands only on leaves of the writer's one output row
  (`lands_on_writer_leaf`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn resolve resolveStep)
open Flock.TemplateQuery (TRange)

/-! ### Wires -/

/-- Under `Aligned`, a writer's leaf that meets a leaf its reader passes on lies inside one leaf of the row. -/
theorem aligned_inside_of_passed {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    (h : Aligned P rs W S) {r : Row} {l' g' lo' hi' l g lo hi : ℕ} (hw : WriterLeaf P rs S r l' g' lo' hi')
    (hleaf : RowLeaf P rs W r l g lo hi) (h1 : lo < hi') (h2 : lo' < hi) (hpass : PassedOn P rs r g) :
    ∃ l g lo hi, RowLeaf P rs W r l g lo hi ∧ lo ≤ lo' ∧ hi' ≤ hi :=
  (h r l' g' lo' hi' hw).resolve_right fun hno => hno l g lo hi hleaf h1 h2 hpass

/-- A wire whose writer has a leaf wider than every leaf of the row that reads it, and that meets a leaf the reader
passes on, is not `Aligned`. -/
theorem not_aligned_of_wider {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {r : Row} {l' g' lo' hi' w l g lo hi : ℕ} (hw : WriterLeaf P rs S r l' g' lo' hi')
    (hrow : ∀ l g lo hi, RowLeaf P rs W r l g lo hi → hi - lo ≤ w) (hwide : w < hi' - lo')
    (hleaf : RowLeaf P rs W r l g lo hi) (h1 : lo < hi') (h2 : lo' < hi) (hpass : PassedOn P rs r g) :
    ¬ Aligned P rs W S := by
  intro h
  obtain ⟨l1, g1, lo1, hi1, hleaf1, hle1, hle2⟩ := aligned_inside_of_passed h hw hleaf h1 h2 hpass
  have := hrow l1 g1 lo1 hi1 hleaf1
  omega

/-- Where no unit passes a gate on, `Aligned` asks nothing of the widths. -/
theorem aligned_of_not_passed {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    (h : ∀ r g, ¬ PassedOn P rs r g) : Aligned P rs W S :=
  fun r _ _ _ _ _ => Or.inr fun _ g _ _ _ _ _ => h r g

/-- How a read lands: on the gate itself where no wire covers it, else through a gate that feeds it. -/
theorem lands_inv {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary} {o a : ℕ}
    (h : Lands P rs W S o a) :
    (¬ Wired P rs W S o ∧ a = o) ∨ ∃ g, Feeds P rs W S g o ∧ Lands P rs W S g a := by
  cases h with
  | here _ hnw => exact Or.inl ⟨hnw, rfl⟩
  | wire _ g _ hf hl => exact Or.inr ⟨g, hf, hl⟩

/-- **A wire from a writer the Program has no unit for.** A leaf of a row that has a writer, where no gate is a leaf
of that writer's output row: a read of it lands on no gate. -/
theorem lands_nowhere {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary} {r : Row}
    {l g lo hi : ℕ} (hown : OwnRows P rs W) (hleaf : RowLeaf P rs W r l g lo hi)
    (hw : (S.writerOf r.tid r.unit r.port r.row).isSome)
    (hno : ∀ l' g' lo' hi', ¬ WriterLeaf P rs S r l' g' lo' hi') (a : ℕ) : ¬ Lands P rs W S g a := by
  intro h
  rcases lands_inv h with ⟨hnw, -⟩ | ⟨g', hf, -⟩
  · exact hnw ⟨r, l, lo, hi, hw, hleaf⟩
  · obtain ⟨r', l1, l', lo1, hi1, lo', hi', hleaf', hwl, -, -⟩ := hf
    obtain ⟨hr, -⟩ := hown.2 r l lo hi r' l1 lo1 hi1 g hleaf hleaf'
    rw [← hr] at hwl
    exact hno l' g' lo' hi' hwl

/-- **A wired leaf past its writer's bits.** A leaf of a row that has a writer, every reading of it as a leaf of that
row starting at or after the last bit of each leaf of the writer's row: a read of it lands on no gate. -/
theorem lands_nowhere_past {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary} {r : Row}
    {l g lo hi : ℕ} (hown : OwnRows P rs W) (hleaf : RowLeaf P rs W r l g lo hi)
    (hw : (S.writerOf r.tid r.unit r.port r.row).isSome)
    (hpast : ∀ l' g' lo' hi' l1 lo1 hi1, WriterLeaf P rs S r l' g' lo' hi' → RowLeaf P rs W r l1 g lo1 hi1 →
      hi' ≤ lo1) (a : ℕ) : ¬ Lands P rs W S g a := by
  intro h
  rcases lands_inv h with ⟨hnw, -⟩ | ⟨g', hf, -⟩
  · exact hnw ⟨r, l, lo, hi, hw, hleaf⟩
  · obtain ⟨r', l1, l', lo1, hi1, lo', hi', hleaf', hwl, hmeet, -⟩ := hf
    obtain ⟨hr, -⟩ := hown.2 r l lo hi r' l1 lo1 hi1 g hleaf hleaf'
    rw [← hr] at hwl hleaf'
    have := hpast l' g' lo' hi' l1 lo1 hi1 hwl hleaf'
    omega

/-- **A held row that has a writer puts no gate in θ**: its gates are wired, and the wired circuit has none of
them. -/
theorem held_wired_names_nothing {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {D : Verity.Circuit} {at_ : Fin D.N → ℕ} (hlay : LaysOut P rs W S D at_) {r : Row}
    (hw : (S.writerOf r.tid r.unit r.port r.row).isSome) (x : Fin D.N) (l lo hi : ℕ) :
    ¬ RowLeaf P rs W r l (at_ x) lo hi := fun h =>
  ((hlay.2.1 (at_ x)).1 ⟨x, rfl⟩).2 ⟨r, l, lo, hi, hw, h⟩

/-! ### The pins and the record's text -/

/-- The check `F`, reporting the pins `ps`. -/
def repin (F : Form) (ps : F.Case → List String) : Form :=
  { F with inst := fun c => { F.inst c with pins := ps c } }

/-- The check `F`, reporting the record text `s`. -/
def retext (F : Form) (s : F.Case → String) : Form :=
  { F with inst := fun c => { F.inst c with record := s c } }

/-- `Carries` holds no pin to the instance. -/
theorem carries_repin (F : Form) (ps : F.Case → List String) (j : Json) (c : F.Case) :
    Carries (repin F ps) j c ↔ Carries F j c := Iff.rfl

theorem owns_repin (F : Form) (ps : F.Case → List String) : Owns (repin F ps) ↔ Owns F := Iff.rfl

theorem laidOut_repin (F : Form) (ps : F.Case → List String) : LaidOut (repin F ps) ↔ LaidOut F := Iff.rfl

theorem counts_repin (F : Form) (ps : F.Case → List String) : Counts (repin F ps) ↔ Counts F := Iff.rfl

theorem rates_repin (F : Form) (ps : F.Case → List String) : Rates (repin F ps) ↔ Rates F := Iff.rfl

theorem complies_repin (F : Form) (ps : F.Case → List String) : Complies (repin F ps) ↔ Complies F := Iff.rfl

/-- `Served` reads no record text. -/
theorem served_retext (F : Form) (s : F.Case → String) (c : F.Case) (d : Verity.RegisteredCircuit) :
    Served (retext F s) c d ↔ Served F c d := Iff.rfl

theorem counts_retext (F : Form) (s : F.Case → String) : Counts (retext F s) ↔ Counts F := Iff.rfl

/-! ### A check that reads nothing -/

/-- A check that refuses every instance. -/
def refuseAll : Form where
  Case := Unit
  read := fun _ _ => .error "refused"
  inst := fun _ => { program := "", record := "", gamma := 0, num := 0, den := 1, pins := [] }
  wiring := fun _ => { templates := [], cuts := [], wires := [] }
  slot := fun _ _ _ => none
  held := fun _ _ => none
  summarize := fun _ => { gates := 0, fresh := 0 }
  rate := fun _ _ _ _ => some "refused"
  layout := fun _ => none

theorem refuseAll_read {j tr : Json} {c : refuseAll.Case} (h : refuseAll.read j tr = .ok c) : False := by
  have h' : (Except.error "refused" : Except String Unit) = Except.ok c := h
  cases h'

theorem refuseAll_rate {Γ num den : ℕ} {s : Summary} (h : refuseAll.rate Γ num den s = none) : False := by
  have h' : (some "refused" : Option String) = none := h
  cases h'

/-- **The premises on the check hold of a check that reads nothing.** -/
theorem refuseAll_premises :
    Owns refuseAll ∧ LaidOut refuseAll ∧ Counts refuseAll ∧ Rates refuseAll ∧ Provenance refuseAll ∧
      LayoutReads refuseAll :=
  ⟨fun _ _ _ h => (refuseAll_read h).elim, fun _ _ _ h => (refuseAll_read h).elim,
    fun _ _ _ h => (refuseAll_read h).elim, fun _ _ _ _ h => (refuseAll_rate h).elim,
    fun _ _ _ _ _ h => (refuseAll_read h).elim, fun _ _ _ _ => rfl⟩

/-! ### A computed copy -/

/-- A gate that is no input gate is under no wire: a row's leaves are input gates. -/
theorem not_wired_of_computed {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    (hown : OwnRows P rs W) {g p : ℕ} {ops : Array ℕ} {sc : Scope} (hg : P.gate g = .ok (p, ops, sc))
    (hc : Flock.Program.isInputId (P.defs[p]!).key = false) : ¬ Wired P rs W S g := by
  rintro ⟨r, l, lo, hi, -, hleaf⟩
  obtain ⟨p', ops', sc', hg', hin⟩ := hown.1 r l g lo hi hleaf
  rw [hg] at hg'
  have hp : p = p' := congrArg Prod.fst (Except.ok.inj hg')
  rw [← hp, hc] at hin
  exact Bool.false_ne_true hin

/-- A read of a gate under no wire lands on that gate alone. -/
theorem lands_eq_of_not_wired {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {o a : ℕ} (hnw : ¬ Wired P rs W S o) (h : Lands P rs W S o a) : a = o := by
  rcases lands_inv h with ⟨-, ha⟩ | ⟨g, hf, -⟩
  · exact ha
  · obtain ⟨r, l, l', lo, hi, lo', hi', hleaf, hwl, -, -⟩ := hf
    obtain ⟨wt, wu, out, sc', hwr, -⟩ := hwl
    exact (hnw ⟨r, l, lo, hi, by simp [hwr], hleaf⟩).elim

/-- **What a wire lets a reader read.** A leaf of a row that has a writer, the writer's leaves under no wire (it
computes them, or they are its own unwired inputs): a read of the leaf lands only on leaves of the writer's one
output row. -/
theorem lands_on_writer_leaf {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {r : Row} {l g lo hi a : ℕ} (hown : OwnRows P rs W) (hleaf : RowLeaf P rs W r l g lo hi)
    (hw : (S.writerOf r.tid r.unit r.port r.row).isSome)
    (hcomp : ∀ l' g' lo' hi', WriterLeaf P rs S r l' g' lo' hi' → ¬ Wired P rs W S g')
    (h : Lands P rs W S g a) : ∃ l' lo' hi', WriterLeaf P rs S r l' a lo' hi' := by
  rcases lands_inv h with ⟨hnw, -⟩ | ⟨g', hf, hl⟩
  · exact (hnw ⟨r, l, lo, hi, hw, hleaf⟩).elim
  · obtain ⟨r', l1, l', lo1, hi1, lo', hi', hleaf', hwl, -, -⟩ := hf
    obtain ⟨hr, -⟩ := hown.2 r l lo hi r' l1 lo1 hi1 g hleaf hleaf'
    rw [← hr] at hwl
    have ha := lands_eq_of_not_wired (hcomp l' g' lo' hi' hwl) hl
    rw [ha]
    exact ⟨l', lo', hi', hwl⟩

/-! ### A leaf handed on by reference -/

/-- One step of a walk that goes on costs one unit of fuel. -/
theorem run_inr {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {n : ℕ} {a a' : α}
    (hs : step a = .ok (.inr a')) :
    Flock.Steps.run what step (n + 1) a = Flock.Steps.run what step n a' := by
  rw [Flock.Steps.run.eq_2, hs]
  all_goals rfl

/-- A walk that ends, one step on. -/
theorem run_drop {α β : Type} {what : String} {step : α → Flock.V (β ⊕ α)} {a a' : α} {b : β}
    (hs : step a = .ok (.inr a')) :
    ∀ {n : ℕ}, Flock.Steps.run what step n a = .ok b → ∃ m, Flock.Steps.run what step m a' = .ok b
  | 0, h => by simp [Flock.Steps.run] at h
  | n + 1, h => ⟨n, by rw [run_inr hs] at h; exact h⟩

/-- The walk from `x` ends at gate `g`, in some number of steps. -/
def Ends (P : Prog) (x : Scope × Ref) (g : ℕ) : Prop :=
  ∃ n, Flock.Steps.run "scope resolution" (resolveStep P) n x = .ok g

theorem ends_of_resolve {P : Prog} {sc : Scope} {r : Ref} {g : ℕ} (h : resolve P sc r = .ok g) :
    Ends P (sc, r) g :=
  ⟨Flock.Steps.FUEL, h⟩

/-- A walk ends at one gate, whatever the fuel. -/
theorem ends_unique {P : Prog} {x : Scope × Ref} {g g' : ℕ} (h : Ends P x g) (h' : Ends P x g') : g = g' := by
  obtain ⟨n, hn⟩ := h
  obtain ⟨m, hm⟩ := h'
  have h1 := Flock.Steps.run_mono (Nat.le_max_left n m) hn
  have h2 := Flock.Steps.run_mono (Nat.le_max_right n m) hm
  exact Except.ok.inj (h1.symm.trans h2)

/-- The walk from `x` goes on to `y`: each step follows one reference. -/
inductive Reaches (P : Prog) : Scope × Ref → Scope × Ref → Prop
  | refl (x : Scope × Ref) : Reaches P x x
  | step {x y z : Scope × Ref} : resolveStep P x = .ok (.inr y) → Reaches P y z → Reaches P x z

theorem ends_of_reaches {P : Prog} {x y : Scope × Ref} (hr : Reaches P x y) :
    ∀ {g : ℕ}, Ends P x g → Ends P y g := by
  induction hr with
  | refl _ => intro g h; exact h
  | step hs _ ih =>
    intro g h
    obtain ⟨n, hn⟩ := h
    exact ih (show Ends P _ g from run_drop hs hn)

/-- **A reference followed is the same gate.** Where the walk from `(sc, r)` goes on to `(sc', r')` and both
resolve, they resolve to one gate. -/
theorem same_gate {P : Prog} {sc sc' : Scope} {r r' : Ref} {g g' : ℕ} (hr : Reaches P (sc, r) (sc', r'))
    (h : resolve P sc r = .ok g) (h' : resolve P sc' r' = .ok g') : g = g' :=
  ends_unique (ends_of_reaches hr (ends_of_resolve h)) (ends_of_resolve h')

/-- A unit passes on the gate a leaf of its return resolves to. -/
theorem passedOn_of_out {P : Prog} {rs : Array TRange} {r : Row} {sc : Scope} {d : Defn} {f l g : ℕ}
    (hu : UnitAt P rs r.tid r.unit sc) (hd : P.defs[sc.d]! = d) (hf : f < (fieldsOf d.ret).size)
    (hl : l < (outRow d f).2.size) (hg : resolve P sc (d.body.ret.get ((outRow d f).1 + l)) = .ok g) :
    PassedOn P rs r g := by
  subst hd
  exact ⟨sc, "o" ++ toString f, l, _, _, hu, f, rfl, hf, hl, hg, rfl, rfl⟩

/-- **A leaf handed on by reference is passed on.** A leaf of the unit's return that resolves, and whose walk goes
on to leaf `j` of the unit's parameter `i`: the unit passes on the gate that leaf of the parameter resolves to. -/
theorem passedOn_of_reaches {P : Prog} {rs : Array TRange} {r : Row} {sc : Scope} {d : Defn} {f l i j g g' : ℕ}
    (hu : UnitAt P rs r.tid r.unit sc) (hd : P.defs[sc.d]! = d) (hf : f < (fieldsOf d.ret).size)
    (hl : l < (outRow d f).2.size)
    (hr : Reaches P (sc, d.body.ret.get ((outRow d f).1 + l)) (sc, ⟨true, i, j⟩))
    (hport : resolve P sc ⟨true, i, j⟩ = .ok g)
    (hout : resolve P sc (d.body.ret.get ((outRow d f).1 + l)) = .ok g') : PassedOn P rs r g := by
  have hgg : g' = g := same_gate hr hout hport
  rw [hgg] at hout
  exact passedOn_of_out hu hd hf hl hout

/-- A walk the kernel runs in `n` steps is the verifier's. -/
theorem resolve_of_run {P : Prog} {sc : Scope} {r : Ref} {g : ℕ} (n : ℕ) (hn : n ≤ Flock.Steps.FUEL)
    (h : Flock.Steps.run "scope resolution" (resolveStep P) n (sc, r) = .ok g) : resolve P sc r = .ok g :=
  Flock.Steps.run_mono hn h

theorem le_fuel {n : ℕ} (h : n ≤ 8) : n ≤ Flock.Steps.FUEL :=
  Nat.le_trans h (by decide)

/-! ### A toy Program: a parameter returned as it is, through a callee, through a batch member, and copied -/

namespace Toy

/-- One reference: leaf `j` of parameter `i` (`p`), or of node `i`'s output. -/
def one (p : Bool) (i j : ℕ) : Flock.Program.Seq := { parts := #[.affine p i j 1], starts := #[0], count := 1 }

/-- An input gate of 16 bits. -/
def dIn : Defn :=
  { key := "Input16_v1", id := "Input16_v1", prim := true, params := #[], ret := .value 16, gates := 1,
    paramLeaves := 0, outLeaves := 1 }

/-- A gate of two operands. -/
def dAnd : Defn :=
  { key := "And16_v1", id := "And16_v1", prim := true, params := #[("a", .value 16), ("b", .value 16)],
    ret := .value 16, gates := 1, paramLeaves := 2, outLeaves := 1 }

/-- `Id(x) = x`: no gate; the return is the parameter's leaf. -/
def dId : Defn :=
  { key := "Id_v1", id := "Id_v1", prim := false, params := #[("x", .value 16)], ret := .value 16,
    body := { params := #[("x", .value 16)], retType := .value 16, nodes := #[], ret := one true 0 0,
              offsets := #[0] },
    gates := 0, paramLeaves := 1, outLeaves := 1 }

/-- `ViaCall(x) = Id(x)`: the return is the callee's return. -/
def dCall : Defn :=
  { key := "ViaCall_v1", id := "ViaCall_v1", prim := false, params := #[("x", .value 16)], ret := .value 16,
    body := { params := #[("x", .value 16)], retType := .value 16,
              nodes := #[{ form := .call, fn := 2, args := #[{ ty := .value 16, seq := one true 0 0 }],
                           out := .value 16 }],
              ret := one false 0 0, offsets := #[0, 0] },
    gates := 0, paramLeaves := 1, outLeaves := 1 }

/-- `ViaBatch(x) = Id` over the two elements of `x`: the return is the members' returns. -/
def dBatch : Defn :=
  { key := "ViaBatch_v1", id := "ViaBatch_v1", prim := false, params := #[("x", .array 2 (.value 16))],
    ret := .array 2 (.value 16),
    body := { params := #[("x", .array 2 (.value 16))], retType := .array 2 (.value 16),
              nodes := #[{ form := .batch, fn := 2,
                           args := #[{ ty := .array 2 (.value 16),
                                       seq := { parts := #[.affine true 0 0 2], starts := #[0], count := 2 },
                                       dims := some (2, 1) }],
                           out := .array 2 (.value 16), n := 2 }],
              ret := { parts := #[.affine false 0 0 2], starts := #[0], count := 2 }, offsets := #[0, 0] },
    gates := 0, paramLeaves := 2, outLeaves := 2 }

/-- `Copy(x) = And(x, x)`: one gate of the unit's own. -/
def dCopy : Defn :=
  { key := "Copy_v1", id := "Copy_v1", prim := false, params := #[("x", .value 16)], ret := .value 16,
    body := { params := #[("x", .value 16)], retType := .value 16,
              nodes := #[{ form := .prim, fn := 1,
                           args := #[{ ty := .value 16, seq := one true 0 0 },
                                     { ty := .value 16, seq := one true 0 0 }],
                           out := .value 16 }],
              ret := one false 0 0, offsets := #[0, 1] },
    gates := 1, paramLeaves := 1, outLeaves := 1 }

/-- A root node that is an input gate. -/
def inNode : Flock.Program.Node := { form := .prim, fn := 0, args := #[], out := .value 16 }

/-- A root node that calls Definition `fn` on one argument. -/
def callNode (fn : ℕ) (ty : Ty) (seq : Flock.Program.Seq) : Flock.Program.Node :=
  { form := .call, fn := fn, args := #[{ ty := ty, seq := seq }], out := ty }

/-- The root: each unit's parameter is input gates of its own. Gates 0 to 4 are the inputs, gate 5 is `Copy`'s. -/
def dRoot : Defn :=
  { key := "Root_v1", id := "Root_v1", prim := false, params := #[], ret := .tuple #[],
    body := { params := #[], retType := .tuple #[],
              nodes := #[inNode, callNode 2 (.value 16) (one false 0 0),
                         inNode, callNode 3 (.value 16) (one false 2 0),
                         inNode, inNode,
                         callNode 4 (.array 2 (.value 16))
                           { parts := #[.explicit #[⟨false, 4, 0⟩, ⟨false, 5, 0⟩]], starts := #[0], count := 2 },
                         inNode, callNode 5 (.value 16) (one false 7 0)],
              ret := { parts := #[], starts := #[], count := 0 },
              offsets := #[0, 1, 1, 2, 2, 3, 4, 4, 5, 6] },
    gates := 6, paramLeaves := 0, outLeaves := 0 }

def toyP : Prog := { (default : Prog) with defs := #[dIn, dAnd, dId, dCall, dBatch, dCopy, dRoot], root := 6 }

/-- The activation of root node `k`. -/
def unit (k : ℕ) : Scope := toyP.child toyP.rootScope k 0

/-! A parameter returned as it is: the return's leaf is the port's reference itself. -/

theorem direct_ret : dId.body.ret.get 0 = ⟨true, 0, 0⟩ := rfl

theorem direct_port : resolve toyP (unit 1) ⟨true, 0, 0⟩ = .ok 0 := resolve_of_run 2 (le_fuel (by decide)) rfl

/-! Through a callee's return. -/

/-- The return of `ViaCall`'s unit is its callee's return, in the callee's activation. -/
theorem call_step1 :
    resolveStep toyP (unit 3, ⟨false, 0, 0⟩) = .ok (.inr (.child (unit 3) 2 2 0 0, ⟨true, 0, 0⟩)) := rfl

/-- The callee's return is its parameter, which is the unit's argument: the port's reference. -/
theorem call_step2 :
    resolveStep toyP (.child (unit 3) 2 2 0 0, ⟨true, 0, 0⟩) = .ok (.inr (unit 3, ⟨true, 0, 0⟩)) := rfl

theorem call_reaches : Reaches toyP (unit 3, ⟨false, 0, 0⟩) (unit 3, ⟨true, 0, 0⟩) :=
  .step call_step1 (.step call_step2 (.refl _))

/-- The port of `ViaCall`'s unit is input gate 1. -/
theorem call_port : resolve toyP (unit 3) ⟨true, 0, 0⟩ = .ok 1 := resolve_of_run 2 (le_fuel (by decide)) rfl

/-- **The return of `ViaCall`'s unit is input gate 1 too**: the verifier's walk, run by the kernel. -/
theorem call_out : resolve toyP (unit 3) ⟨false, 0, 0⟩ = .ok 1 := resolve_of_run 4 (le_fuel (by decide)) rfl

/-- The same without running the walk to its end: whatever the two resolve to, it is one gate. -/
theorem call_same {g g' : ℕ} (h : resolve toyP (unit 3) ⟨false, 0, 0⟩ = .ok g)
    (h' : resolve toyP (unit 3) ⟨true, 0, 0⟩ = .ok g') : g = g' :=
  same_gate call_reaches h h'

/-! Through a batch member. -/

/-- Leaf 1 of the return of `ViaBatch`'s unit is member 1's return, in that member's activation. -/
theorem batch_step1 :
    resolveStep toyP (unit 6, ⟨false, 0, 1⟩) = .ok (.inr (.child (unit 6) 2 4 0 1, ⟨true, 0, 0⟩)) := rfl

/-- Member 1's return is its parameter, which is leaf 1 of the unit's argument: the port's reference. -/
theorem batch_step2 :
    resolveStep toyP (.child (unit 6) 2 4 0 1, ⟨true, 0, 0⟩) = .ok (.inr (unit 6, ⟨true, 0, 1⟩)) := rfl

theorem batch_reaches : Reaches toyP (unit 6, ⟨false, 0, 1⟩) (unit 6, ⟨true, 0, 1⟩) :=
  .step batch_step1 (.step batch_step2 (.refl _))

theorem batch_port0 : resolve toyP (unit 6) ⟨true, 0, 0⟩ = .ok 2 := resolve_of_run 2 (le_fuel (by decide)) rfl

theorem batch_port1 : resolve toyP (unit 6) ⟨true, 0, 1⟩ = .ok 3 := resolve_of_run 2 (le_fuel (by decide)) rfl

theorem batch_out0 : resolve toyP (unit 6) ⟨false, 0, 0⟩ = .ok 2 := resolve_of_run 4 (le_fuel (by decide)) rfl

/-- **Leaf 1 of the return of `ViaBatch`'s unit is input gate 3**, the gate of leaf 1 of its port. -/
theorem batch_out1 : resolve toyP (unit 6) ⟨false, 0, 1⟩ = .ok 3 := resolve_of_run 4 (le_fuel (by decide)) rfl

/-! A computed copy. -/

/-- The port of `Copy`'s unit is input gate 4. -/
theorem copy_port : resolve toyP (unit 8) ⟨true, 0, 0⟩ = .ok 4 := resolve_of_run 2 (le_fuel (by decide)) rfl

/-- The return of `Copy`'s unit is gate 5, the unit's own. -/
theorem copy_out : resolve toyP (unit 8) ⟨false, 0, 0⟩ = .ok 5 := resolve_of_run 1 (le_fuel (by decide)) rfl

/-- **A computed copy is another gate**: the unit's return and its port resolve to different gates. -/
theorem copy_differs : resolve toyP (unit 8) ⟨false, 0, 0⟩ ≠ resolve toyP (unit 8) ⟨true, 0, 0⟩ := by
  rw [copy_out, copy_port]
  exact fun h => absurd (Except.ok.inj h) (by decide)

/-! `PassedOn` itself, at the toy. -/

/-- The ranges of a record whose one template is `ViaCall`. -/
def rsCall : Array TRange := #[⟨3, "ViaCall_v1", 0, 1⟩]

theorem call_unit : UnitAt toyP rsCall "ViaCall_v1" 0 (unit 3) :=
  ⟨0, 0, by decide, rfl, by decide, rfl, rfl⟩

/-- **A parameter returned through a callee's return is passed on**: `ViaCall`'s unit passes input gate 1 on. -/
theorem call_passedOn : PassedOn toyP rsCall ⟨"ViaCall_v1", 0, "x", 0⟩ 1 := by
  have h2 : (outRow dCall 0).2 = Flock.Program.Ty.widths (.value 16) := rfl
  have hx : dCall.body.ret.get ((outRow dCall 0).1 + 0) = ⟨false, 0, 0⟩ := rfl
  refine passedOn_of_out (sc := unit 3) (d := dCall) (f := 0) (l := 0) call_unit rfl (by decide) ?_ ?_
  · rw [h2, Flock.Program.Ty.widths]
    decide
  · rw [hx]
    exact call_out

/-- The ranges of a record whose one template is `ViaBatch`. -/
def rsBatch : Array TRange := #[⟨6, "ViaBatch_v1", 0, 1⟩]

theorem batch_unit : UnitAt toyP rsBatch "ViaBatch_v1" 0 (unit 6) :=
  ⟨0, 0, by decide, rfl, by decide, rfl, rfl⟩

/-- **A parameter returned through a batch member is passed on**: `ViaBatch`'s unit passes input gate 3 on. -/
theorem batch_passedOn : PassedOn toyP rsBatch ⟨"ViaBatch_v1", 0, "x", 0⟩ 3 := by
  have h2 : (outRow dBatch 0).2 = Flock.Program.Ty.widths (.array 2 (.value 16)) := rfl
  have hx : dBatch.body.ret.get ((outRow dBatch 0).1 + 1) = ⟨false, 0, 1⟩ := rfl
  refine passedOn_of_out (sc := unit 6) (d := dBatch) (f := 0) (l := 1) batch_unit rfl (by decide) ?_ ?_
  · rw [h2]
    simp [Flock.Program.Ty.widths]
  · rw [hx]
    exact batch_out1

end Toy

/-! ### The steps, for any Program -/

/-- A call node's output leaf is the callee's return leaf, in the callee's activation. -/
theorem step_call (P : Prog) (sc : Scope) (k j : ℕ) (h : ((P.defs[sc.d]!).body.nodes[k]!).form = .call) :
    resolveStep P (sc, ⟨false, k, j⟩) =
      .ok (.inr (.child sc ((P.defs[sc.d]!).body.nodes[k]!).fn (sc.off + (P.defs[sc.d]!).body.offsets[k]!) k 0,
        (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).body.ret.get j)) := by
  rcases sc with ⟨d, off, _ | starts⟩ | ⟨parent, d, off, node, member⟩ <;>
    first
      | (unfold resolveStep; simp only [h]; all_goals rfl)
      | (unfold resolveStep; dsimp only; rw [h]; all_goals rfl)

/-- A parameter's leaf, in the activation of a call node, is the node's argument's leaf in the caller. -/
theorem step_param (P : Prog) (parent : Scope) (d off k m i j : ℕ)
    (h : ((P.defs[parent.d]!).body.nodes[k]!).form = .call) :
    resolveStep P (.child parent d off k m, ⟨true, i, j⟩) =
      .ok (.inr (parent, (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j)) := by
  unfold resolveStep
  simp only [h]
  all_goals rfl

/-- A batch node's output leaf, its callee no primitive, is a member's return leaf in that member's activation. -/
theorem step_batch (P : Prog) (sc : Scope) (k j : ℕ) (h : ((P.defs[sc.d]!).body.nodes[k]!).form = .batch)
    (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = false) :
    resolveStep P (sc, ⟨false, k, j⟩) =
      .ok (.inr (.child sc ((P.defs[sc.d]!).body.nodes[k]!).fn
        (sc.off + (P.defs[sc.d]!).body.offsets[k]! +
          j / (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).outLeaves *
            (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).gates) k
        (j / (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).outLeaves),
        (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).body.ret.get
          (j % (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).outLeaves))) := by
  rcases sc with ⟨d, off, _ | starts⟩ | ⟨parent, d, off, node, member⟩ <;>
    first
      | (unfold resolveStep; simp only [h, hp, Bool.false_eq_true, ↓reduceIte]; all_goals rfl)
      | (unfold resolveStep; simp only [h]; rw [hp]; all_goals rfl)
      | (unfold resolveStep; dsimp only; rw [h]; dsimp only; rw [hp]; all_goals rfl)

/-- A parameter's leaf, in the activation of a batch member, is that member's leaf of the node's argument in the
caller. -/
theorem step_member (P : Prog) (parent : Scope) (d off k m i j : ℕ)
    (h : ((P.defs[parent.d]!).body.nodes[k]!).form = .batch) :
    resolveStep P (.child parent d off k m, ⟨true, i, j⟩) =
      .ok (.inr (parent, (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).memberRef m j)) := by
  unfold resolveStep
  simp only [h]
  all_goals rfl

end NciCheck.Properties.Record.Attack
