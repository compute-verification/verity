module

public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-!
# Trial: the pickers as a search of a list, over a pair function

A scratch, not to merge. `PairTrial.Pick` hands the pickers a pair function and still takes the witness of the event
by choice (`h.choose`). Here the witness is searched for: `pickOf ws p pr` is `pr` at the first member of the list
`ws` that passes `p`. A test is decided classically; nothing is chosen. Two runs' conflict is searched over the
positions (`conflictPickS`, over `Finset.univ.toList` for a finite type of positions); a run's own event is searched
over a list the run holds (`onePickS`). The four lemmas every bound passes through follow as before
(`expect_one_leS`, `expect₂_conflict_leS`, `forkE_conflict_leS`, `tree_le_strictS`), and so does `TreeRewind`
(`TreeRewindS`).
-/

@[expose] public section

namespace FlockSoundness.StrictCR

open Game Model Accounting
open scoped ENNReal

section FindersS

variable {D : Type} (H : List UInt8 → D)

open Classical in
/-- The pair `pr` gives at the first member of `ws` that passes `p`; nothing when none does. -/
noncomputable def pickOf {W : Type} (ws : List W) (p : W → Prop) (pr : W → List UInt8 × List UInt8) : Pair :=
  (ws.find? fun w => decide (p w)).map pr

open Classical in
theorem pickOf_isColl {W : Type} {ws : List W} {p : W → Prop} {pr : W → List UInt8 × List UInt8}
    (hpr : ∀ w, p w → Hash.Collision H (pr w).1 (pr w).2) (h : ∃ w ∈ ws, p w) : IsColl H (pickOf ws p pr) := by
  obtain ⟨w, hw, hp⟩ := h
  have hs : (ws.find? fun w => decide (p w)).isSome := List.find?_isSome.2 ⟨w, hw, decide_eq_true hp⟩
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  have hv' : p v := of_decide_eq_true (List.find?_some hv)
  refine ⟨(pr v).1, (pr v).2, ?_, hpr v hv'⟩
  unfold pickOf
  rw [hv]
  rfl

/-- An event of a run, as a finder's output: the pair at the first member of the run's list that passes. -/
noncomputable def onePickS {α W : Type} (ws : α → List W) (qw : α → W → Prop)
    (pr : α → W → List UInt8 × List UInt8) (a : α) : Pair :=
  pickOf (ws a) (qw a) (pr a)

theorem ind_le_onePickS {α W : Type} (q : α → Prop) (ws : α → List W) (qw : α → W → Prop)
    (pr : α → W → List UInt8 × List UInt8) (hq : ∀ a, q a → ∃ w ∈ ws a, qw a w)
    (hpr : ∀ a w, qw a w → Hash.Collision H (pr a w).1 (pr a w).2) (a : α) :
    ind q a ≤ ind (IsColl H) (onePickS ws qw pr a) := by
  unfold ind
  by_cases h : q a
  · have hc : IsColl H (onePickS ws qw pr a) := pickOf_isColl H (hpr a) (hq a h)
    rw [ite_eq_left h, ite_eq_left hc]
  · rw [ite_eq_right h]
    split_ifs <;> norm_num

/-- Two runs' conflict at some position, as a finder's output: the pair at the first position where they conflict. -/
noncomputable def conflictPickS {α P V : Type} [Fintype P] (op : α → P → Option V)
    (pr : α × α → P → List UInt8 × List UInt8) (b b' : α) : Pair :=
  onePickS (fun _ : α × α => (Finset.univ : Finset P).toList)
    (fun (x : α × α) (p : P) => Rewinding.ConflictAt op x p) pr (b, b')

theorem ind_le_conflictPickS {α P V : Type} [Fintype P] (op : α → P → Option V)
    (pr : α × α → P → List UInt8 × List UInt8)
    (hpr : ∀ x p, Rewinding.ConflictAt op x p → Hash.Collision H (pr x p).1 (pr x p).2) (b b' : α) :
    ind (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p) (b, b') ≤ ind (IsColl H) (conflictPickS op pr b b') :=
  ind_le_onePickS H (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p)
    (fun _ : α × α => (Finset.univ : Finset P).toList) (fun (x : α × α) (p : P) => Rewinding.ConflictAt op x p) pr
    (fun _ h => h.elim fun p hp => ⟨p, Finset.mem_toList.2 (Finset.mem_univ p), hp⟩) hpr (b, b')

theorem expect_one_leS {α W : Type} (g : Game α) (s : Strategy g) (q : α → Prop) (ws : α → List W)
    (qw : α → W → Prop) (pr : α → W → List UInt8 × List UInt8) (hq : ∀ a, q a → ∃ w ∈ ws a, qw a w)
    (hpr : ∀ a w, qw a w → Hash.Collision H (pr a w).1 (pr a w).2) :
    expect (ind q) g s ≤ (oneRun g s (onePickS ws qw pr)).ev (ind (IsColl H)) := by
  rw [oneRun_ev]
  exact expect_mono (ind_le_onePickS H q ws qw pr hq hpr) g s

theorem expect₂_conflict_leS {α P V : Type} [Fintype P] (g : Game α) (s : Strategy g) (op : α → P → Option V)
    (pr : α × α → P → List UInt8 × List UInt8)
    (hpr : ∀ x p, Rewinding.ConflictAt op x p → Hash.Collision H (pr x p).1 (pr x p).2) :
    expect₂ (fun b b' => ind (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p) (b, b')) g s ≤
      (twoRuns g s (conflictPickS op pr)).ev (ind (IsColl H)) := by
  rw [twoRuns_ev]
  exact expect_mono (fun b => expect_mono (fun b' => ind_le_conflictPickS H op pr hpr b b') _ _) _ _

section Fork

variable {α β T Dc I γ : Type} {get : I → ℕ → Option (T × Dc)} {R : I → α → β → Prop}

theorem forkE_conflict_leS {P V : Type} [Fintype P] (plur : Lock.Chooser T Dc β) (slot : ℕ)
    (op : Dc → β → P → Option V) (pr : Dc → β × β → P → List UInt8 × List UInt8)
    (hpr : ∀ d x p, Rewinding.ConflictAt (op d) x p → Hash.Collision H (pr d x p).1 (pr d x p).2)
    {i : I} {g : Game α} {g' : Game β} (L : Lock T Dc get R i g g') (σ' : Strategy g') :
    Lock.forkE plur slot (fun d b b' => ind (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b')) L σ' ≤
      (forkFinder plur slot (fun d => conflictPickS (op d) (pr d)) none L σ').ev (ind (IsColl H)) :=
  forkE_le_ev plur slot (fun d b b' => ind_le_conflictPickS H (op d) (pr d) (hpr d) b b') (ind_nonneg _ _) L σ'

end Fork

open Classical in
/-- `tree_le_strict` over a searched pair: `hcol` is replaced by `pr` and its spec `hpr`, and the statement is
otherwise the same. -/
theorem tree_le_strictS {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] (op : α → P → Option V)
    (g : Game α) (s : Strategy g) (vstar : P → V) (hv : Rewinding.IsPluralityE op g s vstar) (Q : ℕ)
    (hQ : ∀ a, (Finset.univ.filter fun p => (op a p).isSome).card ≤ Q)
    (pr : α × α → P → List UInt8 × List UInt8)
    (hpr : ∀ x p, Rewinding.ConflictAt op x p → Hash.Collision H (pr x p).1 (pr x p).2) {q : ℝ}
    (hCR : (twoRuns g s (conflictPickS op pr)).CR H q) :
    expect (ind fun a => ∃ p, Rewinding.Off op vstar a p) g s ≤
      Real.sqrt (Fintype.card P * Q * (q ^ 2 / 2 ^ (513 : ℝ))) :=
  (Rewinding.expect_bad_le op g s vstar hv Q hQ).trans (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
    ((expect₂_conflict_leS H g s op pr hpr).trans hCR.le) (by positivity)))

/-- `TreeRewind` with the pair as a field and the position searched for. -/
structure TreeRewindS {β : Type} (G : Game β) (S : Strategy G) (bad : β → Prop) : Type 1 where
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

namespace TreeRewindS

attribute [local instance] fintypeP fintypeV decEqV

variable {H} {β : Type} {G : Game β} {S : Strategy G} {bad : β → Prop}

noncomputable def finder (tr : TreeRewindS H G S bad) : Finder Pair :=
  twoRuns tr.g tr.s (conflictPickS tr.op tr.pair)

noncomputable def bound (tr : TreeRewindS H G S bad) (q : ℝ) : ℝ :=
  Real.sqrt (Fintype.card tr.P * tr.Q * (q ^ 2 / 2 ^ (513 : ℝ)))

theorem le (tr : TreeRewindS H G S bad) {q : ℝ} (h : tr.finder.CR H q) : prob bad G S ≤ ENNReal.ofReal (tr.bound q) :=
  tr.covers.trans (ENNReal.ofReal_le_ofReal
    (tree_le_strictS H tr.op tr.g tr.s tr.vstar tr.plur tr.Q tr.opens tr.pair tr.binds h))

def ofNever (h : ∀ b, ¬ bad b) : TreeRewindS H G S bad where
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

theorem ofNever_bound (h : ∀ b, ¬ bad b) (q : ℝ) : (ofNever h : TreeRewindS H G S bad).bound q = 0 := by
  simp [bound, ofNever]

end TreeRewindS

end FindersS

end FlockSoundness.StrictCR
