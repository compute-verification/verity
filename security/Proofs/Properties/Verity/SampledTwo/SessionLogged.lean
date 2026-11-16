module

public import Proofs.Properties.Verity.SampledTwo.SessionCoins

/-!
# The interior's log on the play's, from the cut's read

The sessions' coins take one fact from the cut as a premise (`InteriorLogged`): on a play the two-stage form accepts
whose log the check read is the device's, the play's log is the interior's read after releases that neither sample nor
open a coin tree. This module derives it from what the cut's read gives, and names what is left.

What the cut's read gives (`InteriorCut`): on one window both stages read one session, so the interior's read is the
whole of the play's log; on two windows the play's log is the first session's read and then the interior's. So on two
windows the releases before the interior's are the first session's, and `InteriorLogged` asks that they sample and
open no coin tree. It has to: the interior's read numbers its coin trees from its own log's first (`readLog` starts
from no tree), and the device numbers a play's trees from the play's first, so a tree sampled before the interior's
session would put each of the interior's openings on another tree of the device's.

* `readLog_treeless`: a read that shows no coin tree's `Hello` shows no opening (`treeStep` reads an opening only of
  a tree read before it). So the first session's part is one fact: its read shows no `Hello`.
* `interiorLogged_of`: `InteriorLogged`, from the cut's read (`InteriorCut`) and a first session that samples and
  opens no coin tree (`FirstFresh`).
* `readLog_plain`: a read of a log none of whose lines is a coin tree's shows only fresh coins, so no `Hello`.

Not established here: that the first session's read shows no `Hello` on two windows. The form's first stage reads the
first session's log only through `Spot.Challenger.releasedRootsOf`, which passes over a tree's lines; the form refuses
such a log by a step of its own (`VStar.Recursive.SampledTwo.firstTreeless`), which `SessionTreeless` reads.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled.Binding (Read)

/-! ## A read that samples no coin tree opens none -/

/-- **A read that shows no coin tree's `Hello` shows no opening**, after no tree: `treeStep` reads an opening only of a
tree read before it (`Verity.Sampled.treeStep_some`), and the trees read grow only at a `Hello`. -/
theorem readTail_treeless {label : String} {after : ByteArray} {records : List Json} :
    ∀ (lines : List String) (pending : Option String) (ξ : ByteArray) {log : List Release},
      Verity.Sampled.readTail label after records lines [] pending ξ = some log →
      (∀ r ∈ log, ∀ sp, r.frame ≠ .hello sp) → ∀ r ∈ log, openTree r.frame = none
  | [], none, _, log, hR, _, r, hr => by
    simp only [Verity.Sampled.readTail, Option.some.injEq] at hR
    subst hR
    exact absurd hr List.not_mem_nil
  | [], some _, _, _, hR, _, _, _ => by simp [Verity.Sampled.readTail] at hR
  | l₀ :: ls, pending, ξ, log, hR, hH, r, hr => by
    unfold Verity.Sampled.readTail at hR
    by_cases ht₀ : Spot.Challenger.treeLine l₀ = true
    · rw [ite_eq_left ht₀] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r₀, ts'⟩, hX₀, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX₀
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      rcases Verity.Sampled.treeStep_some hX with ⟨_, _, spec, hr₀, -⟩ | ⟨i, s, n, -, -, -, stream, -, hk⟩
      · exact absurd hr₀ (hH r₀ List.mem_cons_self spec)
      · simp at hk
    · rw [ite_eq_right ht₀] at hR
      cases pending with
      | none => exact readTail_treeless ls (some l₀) ξ hR hH r hr
      | some a =>
        simp only [Option.bind_eq_bind] at hR
        obtain ⟨⟨lab, root⟩, -, hR⟩ := Option.bind_eq_some_iff.1 hR
        simp only [Option.pure_def] at hR
        have step : ∀ (q : Release) (ξ' : ByteArray) (rest : List Release),
            Verity.Sampled.readTail label after records ls [] none ξ' = some rest → log = q :: rest →
            (∃ lab', q.frame = .fresh lab') → openTree r.frame = none := by
          intro q ξ' rest hrest hlog hq
          subst hlog
          rcases List.mem_cons.1 hr with rfl | hr'
          · obtain ⟨lab', hq⟩ := hq
            rw [hq]
            rfl
          · exact readTail_treeless ls none ξ' hrest (fun x hx => hH x (List.mem_cons_of_mem _ hx)) r hr'
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact step _ _ rest hrest (Option.some.inj hR).symm ⟨_, rfl⟩
        · rw [ite_eq_right hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact step _ _ rest hrest (Option.some.inj hR).symm ⟨_, rfl⟩

/-- **A read of a log's lines that shows no coin tree's `Hello` shows no opening** (`readTail_treeless`), the head's
line read first. -/
theorem readLog_treeless {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Release}
    (hR : Verity.Sampled.readLog label after records ξ lines = some log)
    (hH : ∀ r ∈ log, ∀ sp, r.frame ≠ .hello sp) : ∀ r ∈ log, openTree r.frame = none := by
  cases lines with
  | nil =>
    simp only [Verity.Sampled.readLog, Option.some.injEq] at hR
    subst hR
    exact fun r hr => absurd hr List.not_mem_nil
  | cons l₀ ls =>
    simp only [Verity.Sampled.readLog] at hR
    by_cases ht₀ : Spot.Challenger.treeLine l₀ = true
    · rw [ite_eq_left ht₀] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r₀, ts⟩, hX₀, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX₀
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      rcases Verity.Sampled.treeStep_some hX with ⟨_, _, spec, hr₀, -⟩ | ⟨i, s, n, -, -, -, stream, -, hk⟩
      · exact absurd hr₀ (hH r₀ List.mem_cons_self spec)
      · simp at hk
    · rw [ite_eq_right ht₀] at hR
      exact readTail_treeless ls none ξ hR hH

/-! ## A log that is two reads, one after the other -/

/-- A list related to two lists one after the other is two lists, each related to one. -/
theorem forall₂_append_left {α β : Type} {R : α → β → Prop} : ∀ {a b : List α} {c : List β},
    List.Forall₂ R (a ++ b) c → ∃ c₁ c₂, c = c₁ ++ c₂ ∧ List.Forall₂ R a c₁ ∧ List.Forall₂ R b c₂
  | [], _, c, h => ⟨[], c, rfl, .nil, h⟩
  | x :: a, b, c, h => by
    rw [List.cons_append] at h
    cases h with
    | cons hx ht =>
      obtain ⟨c₁, c₂, rfl, h₁, h₂⟩ := forall₂_append_left ht
      exact ⟨_ :: c₁, c₂, rfl, .cons hx h₁, h₂⟩

/-- Each entry of a list related to another is related to an entry of it. -/
theorem forall₂_mem_of_right {α β : Type} {R : α → β → Prop} : ∀ {a : List α} {b : List β}, List.Forall₂ R a b →
    ∀ y ∈ b, ∃ x ∈ a, R x y
  | _, _, .nil, y, hy => absurd hy List.not_mem_nil
  | _, _, .cons hx ht, y, hy => by
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨_, List.mem_cons_self, hx⟩
    · obtain ⟨x, hx', hR⟩ := forall₂_mem_of_right ht y hy
      exact ⟨x, List.mem_cons_of_mem _ hx', hR⟩

/-! ## The interior's log on the play's -/

/-- **What the cut's read gives of the interior's**, on a play the form accepts where `Good` holds: the interior's
read is the play's log, release by release (one window: both stages read one session), or the play's log is the first
session's read `r₁` and then the interior's (two windows). -/
def InteriorCut (C : CaseOf) (Good : Exchange → Prop) (first : Exchange → List Release) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ l, interiorRead C o = some l →
    List.Forall₂ Read.Of (l.map Read.ofV2) o.log ∨
      List.Forall₂ Read.Of ((first o ++ l).map Read.ofV2) o.log

/-- **The first session's read samples no coin tree and opens none**, on a play the form accepts where `Good`
holds. -/
def FirstFresh (C : CaseOf) (Good : Exchange → Prop) (first : Exchange → List Release) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ r ∈ first o, openTree r.frame = none ∧ ∀ sp, r.frame ≠ .hello sp

/-- **`InteriorLogged`, from the cut's read and a first session of no coin tree**: on one window the releases before
the interior's are none; on two they are the first session's read's, frame by frame (`of_ofV2`). -/
theorem interiorLogged_of {C : CaseOf} {Good : Exchange → Prop} {first : Exchange → List Release}
    (hcut : InteriorCut C Good first) (hfree : FirstFresh C Good first) : InteriorLogged C Good := by
  intro o hacc hgood l hl
  rcases hcut o hacc hgood l hl with h | h
  · exact ⟨[], o.log, rfl, fun r hr => absurd hr List.not_mem_nil, h⟩
  · rw [List.map_append] at h
    obtain ⟨q, l', hq, h₁, h₂⟩ := forall₂_append_left h
    refine ⟨q, l', hq, fun r hr => ?_, h₂⟩
    obtain ⟨x', hx', hR⟩ := forall₂_mem_of_right h₁ r hr
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hx'
    rw [← (of_ofV2 hR).1]
    exact hfree o hacc hgood x hx

/-! ## A read of no coin tree's line -/

/-- **A read of lines none of which is a coin tree's shows only fresh coins**, after any trees: `readTail` reads a
coin tree's `Hello` or opening only at a coin tree's line (`Spot.Challenger.treeLine`). -/
theorem readTail_plain {label : String} {after : ByteArray} {records : List Json} :
    ∀ (lines : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      {log : List Release}, Verity.Sampled.readTail label after records lines ts pending ξ = some log →
      (∀ l ∈ lines, Spot.Challenger.treeLine l = false) → ∀ r ∈ log, ∃ lab, r.frame = .fresh lab
  | [], _, none, _, log, hR, _, r, hr => by
    simp only [Verity.Sampled.readTail, Option.some.injEq] at hR
    subst hR
    exact absurd hr List.not_mem_nil
  | [], _, some _, _, _, hR, _, _, _ => by simp [Verity.Sampled.readTail] at hR
  | l₀ :: ls, ts, pending, ξ, log, hR, hP, r, hr => by
    unfold Verity.Sampled.readTail at hR
    have ht₀ : ¬ Spot.Challenger.treeLine l₀ = true := by
      rw [hP l₀ List.mem_cons_self]
      exact Bool.false_ne_true
    rw [ite_eq_right ht₀] at hR
    have hP' : ∀ l ∈ ls, Spot.Challenger.treeLine l = false := fun l hl => hP l (List.mem_cons_of_mem _ hl)
    cases pending with
    | none => exact readTail_plain ls ts (some l₀) ξ hR hP' r hr
    | some a =>
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨lab, root⟩, -, hR⟩ := Option.bind_eq_some_iff.1 hR
      simp only [Option.pure_def] at hR
      have step : ∀ (q : Release) (ξ' : ByteArray) (rest : List Release),
          Verity.Sampled.readTail label after records ls ts none ξ' = some rest → log = q :: rest →
          (∃ lab', q.frame = .fresh lab') → ∃ lab', r.frame = .fresh lab' := by
        intro q ξ' rest hrest hlog hq
        subst hlog
        rcases List.mem_cons.1 hr with rfl | hr'
        · exact hq
        · exact readTail_plain ls ts none ξ' hrest hP' r hr'
      by_cases hm : lab = label ∧ root = after
      · rw [ite_eq_left hm] at hR
        obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
        exact step _ _ rest hrest (Option.some.inj hR).symm ⟨_, rfl⟩
      · rw [ite_eq_right hm] at hR
        obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
        exact step _ _ rest hrest (Option.some.inj hR).symm ⟨_, rfl⟩

/-- **A read of a log none of whose lines is a coin tree's shows only fresh coins** (`readTail_plain`): no coin
tree's `Hello` and no opening. -/
theorem readLog_plain {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Release}
    (hR : Verity.Sampled.readLog label after records ξ lines = some log)
    (hP : ∀ l ∈ lines, Spot.Challenger.treeLine l = false) : ∀ r ∈ log, ∃ lab, r.frame = .fresh lab := by
  cases lines with
  | nil =>
    simp only [Verity.Sampled.readLog, Option.some.injEq] at hR
    subst hR
    exact fun r hr => absurd hr List.not_mem_nil
  | cons l₀ ls =>
    simp only [Verity.Sampled.readLog] at hR
    have ht₀ : ¬ Spot.Challenger.treeLine l₀ = true := by
      rw [hP l₀ List.mem_cons_self]
      exact Bool.false_ne_true
    rw [ite_eq_right ht₀] at hR
    exact readTail_plain ls [] none ξ hR fun l hl => hP l (List.mem_cons_of_mem _ hl)

end Verity.SampledTwo.SessionCoins
