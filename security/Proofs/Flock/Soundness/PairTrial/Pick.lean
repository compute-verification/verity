module

public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-!
# Trial: the pickers over a pair function

A scratch, not to merge. `StrictCR`'s pickers (`collOf`, `onePick`, `conflictPick`) take a proof that `H` has a
collision and choose the two strings from that proposition. Here each takes a function that gives the two strings
(`collOfP`, `onePickP`, `conflictPickP`), and the lemma beside it takes the function's spec: at a witness of the event,
the two strings are a collision. The four lemmas every bound passes through follow as before (`expect_one_leP`,
`expect₂_conflict_leP`, `forkE_conflict_leP`, `tree_le_strictP`), and so does `TreeRewind` with the pair as a field
(`TreeRewindP`).
-/

@[expose] public section

namespace FlockSoundness.StrictCR

open Game Model Accounting
open scoped ENNReal

section FindersP

variable {D : Type} (H : List UInt8 → D)

open Classical in
/-- The pair `pr` gives at a witness of `p`, when `p` has one; nothing otherwise. -/
noncomputable def collOfP {W : Type} (p : W → Prop) (pr : W → List UInt8 × List UInt8) : Pair :=
  if h : ∃ w, p w then some (pr h.choose) else none

theorem collOfP_isColl {W : Type} {p : W → Prop} {pr : W → List UInt8 × List UInt8}
    (hpr : ∀ w, p w → Hash.Collision H (pr w).1 (pr w).2) (h : ∃ w, p w) : IsColl H (collOfP p pr) := by
  unfold collOfP
  rw [dite_eq_left h]
  exact ⟨(pr h.choose).1, (pr h.choose).2, rfl, hpr _ h.choose_spec⟩

/-- An event of a run, as a finder's output: the pair at a witness of the event, when it has one. -/
noncomputable def onePickP {α W : Type} (qw : α → W → Prop) (pr : α → W → List UInt8 × List UInt8) (a : α) : Pair :=
  collOfP (qw a) (pr a)

theorem ind_le_onePickP {α W : Type} (q : α → Prop) (qw : α → W → Prop) (pr : α → W → List UInt8 × List UInt8)
    (hq : ∀ a, q a → ∃ w, qw a w) (hpr : ∀ a w, qw a w → Hash.Collision H (pr a w).1 (pr a w).2) (a : α) :
    ind q a ≤ ind (IsColl H) (onePickP qw pr a) := by
  unfold ind
  by_cases h : q a
  · have hc : IsColl H (onePickP qw pr a) := collOfP_isColl H (hpr a) (hq a h)
    rw [ite_eq_left h, ite_eq_left hc]
  · rw [ite_eq_right h]
    split_ifs <;> norm_num

/-- Two runs' conflict at some position, as a finder's output: the pair at a position where they conflict. -/
noncomputable def conflictPickP {α P V : Type} (op : α → P → Option V)
    (pr : α × α → P → List UInt8 × List UInt8) (b b' : α) : Pair :=
  onePickP (fun (x : α × α) (p : P) => Rewinding.ConflictAt op x p) pr (b, b')

theorem expect_one_leP {α W : Type} (g : Game α) (s : Strategy g) (q : α → Prop) (qw : α → W → Prop)
    (pr : α → W → List UInt8 × List UInt8) (hq : ∀ a, q a → ∃ w, qw a w)
    (hpr : ∀ a w, qw a w → Hash.Collision H (pr a w).1 (pr a w).2) :
    expect (ind q) g s ≤ (oneRun g s (onePickP qw pr)).ev (ind (IsColl H)) := by
  rw [oneRun_ev]
  exact expect_mono (ind_le_onePickP H q qw pr hq hpr) g s

theorem expect₂_conflict_leP {α P V : Type} (g : Game α) (s : Strategy g) (op : α → P → Option V)
    (pr : α × α → P → List UInt8 × List UInt8)
    (hpr : ∀ x p, Rewinding.ConflictAt op x p → Hash.Collision H (pr x p).1 (pr x p).2) :
    expect₂ (fun b b' => ind (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p) (b, b')) g s ≤
      (twoRuns g s (conflictPickP op pr)).ev (ind (IsColl H)) := by
  rw [twoRuns_ev]
  exact expect_mono (fun b => expect_mono (fun b' =>
    ind_le_onePickP H (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p)
      (fun (x : α × α) (p : P) => Rewinding.ConflictAt op x p) pr (fun _ h => h) hpr (b, b')) _ _) _ _

section Fork

variable {α β T Dc I γ : Type} {get : I → ℕ → Option (T × Dc)} {R : I → α → β → Prop}

theorem forkE_conflict_leP {P V : Type} (plur : Lock.Chooser T Dc β) (slot : ℕ) (op : Dc → β → P → Option V)
    (pr : Dc → β × β → P → List UInt8 × List UInt8)
    (hpr : ∀ d x p, Rewinding.ConflictAt (op d) x p → Hash.Collision H (pr d x p).1 (pr d x p).2)
    {i : I} {g : Game α} {g' : Game β} (L : Lock T Dc get R i g g') (σ' : Strategy g') :
    Lock.forkE plur slot (fun d b b' => ind (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b')) L σ' ≤
      (forkFinder plur slot (fun d => conflictPickP (op d) (pr d)) none L σ').ev (ind (IsColl H)) :=
  forkE_le_ev plur slot (fun d b b' => ind_le_onePickP H (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p)
    (fun (x : β × β) (p : P) => Rewinding.ConflictAt (op d) x p) (pr d) (fun _ h => h) (hpr d) (b, b'))
    (ind_nonneg _ _) L σ'

end Fork

open Classical in
/-- `tree_le_strict` over a pair function: `hcol` is replaced by `pr` and its spec `hpr`, and the statement is
otherwise the same. -/
theorem tree_le_strictP {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] (op : α → P → Option V)
    (g : Game α) (s : Strategy g) (vstar : P → V) (hv : Rewinding.IsPluralityE op g s vstar) (Q : ℕ)
    (hQ : ∀ a, (Finset.univ.filter fun p => (op a p).isSome).card ≤ Q)
    (pr : α × α → P → List UInt8 × List UInt8)
    (hpr : ∀ x p, Rewinding.ConflictAt op x p → Hash.Collision H (pr x p).1 (pr x p).2) {q : ℝ}
    (hCR : (twoRuns g s (conflictPickP op pr)).CR H q) :
    expect (ind fun a => ∃ p, Rewinding.Off op vstar a p) g s ≤
      Real.sqrt (Fintype.card P * Q * (q ^ 2 / 2 ^ (513 : ℝ))) :=
  (Rewinding.expect_bad_le op g s vstar hv Q hQ).trans (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
    ((expect₂_conflict_leP H g s op pr hpr).trans hCR.le) (by positivity)))

/-- `TreeRewind` with the pair as a field: `pair` gives the two strings of two runs' conflict at a position, and `binds`
says they collide. -/
structure TreeRewindP {β : Type} (G : Game β) (S : Strategy G) (bad : β → Prop) : Type 1 where
  α : Type
  P : Type
  V : Type
  [fintypeP : Fintype P]
  [fintypeV : Fintype V]
  [decEqV : DecidableEq V]
  op : α → P → Option V
  g : Game α
  s : Strategy g
  vstar : P → V
  plur : Rewinding.IsPluralityE op g s vstar
  Q : ℕ
  opens : ∀ a, (Finset.univ.filter fun p => (op a p).isSome).card ≤ Q
  pair : α × α → P → List UInt8 × List UInt8
  binds : ∀ x p, Rewinding.ConflictAt op x p → Hash.Collision H (pair x p).1 (pair x p).2
  covers : prob bad G S ≤ ENNReal.ofReal (expect (ind fun a => ∃ p, Rewinding.Off op vstar a p) g s)

namespace TreeRewindP

attribute [local instance] fintypeP fintypeV decEqV

variable {H} {β : Type} {G : Game β} {S : Strategy G} {bad : β → Prop}

/-- The finder `δ_tree` counts: two runs after the registration, and the pair at a position they open to different
values. -/
noncomputable def finder (tr : TreeRewindP H G S bad) : Finder Pair :=
  twoRuns tr.g tr.s (conflictPickP tr.op tr.pair)

noncomputable def bound (tr : TreeRewindP H G S bad) (q : ℝ) : ℝ :=
  Real.sqrt (Fintype.card tr.P * tr.Q * (q ^ 2 / 2 ^ (513 : ℝ)))

theorem le (tr : TreeRewindP H G S bad) {q : ℝ} (h : tr.finder.CR H q) : prob bad G S ≤ ENNReal.ofReal (tr.bound q) :=
  tr.covers.trans (ENNReal.ofReal_le_ofReal
    (tree_le_strictP H tr.op tr.g tr.s tr.vstar tr.plur tr.Q tr.opens tr.pair tr.binds h))

def ofNever (h : ∀ b, ¬ bad b) : TreeRewindP H G S bad where
  α := Unit
  P := Empty
  V := Unit
  op _ e := e.elim
  g := .ret ()
  s := PUnit.unit
  vstar e := e.elim
  plur e := e.elim
  Q := 0
  opens _ := by simp
  pair _ e := e.elim
  binds _ e := e.elim
  covers := (prob_mono (p := bad) (q := fun _ => False) h G S).trans (by rw [prob_false]; exact zero_le)

theorem ofNever_bound (h : ∀ b, ¬ bad b) (q : ℝ) : (ofNever h : TreeRewindP H G S bad).bound q = 0 := by
  simp [bound, ofNever]

end TreeRewindP

end FindersP

end FlockSoundness.StrictCR
