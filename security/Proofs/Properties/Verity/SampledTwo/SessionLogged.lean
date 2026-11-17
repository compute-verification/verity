module

public import Proofs.Properties.Verity.SampledTwo.SessionCoins

/-! `InteriorLogged` from the cut's read and a first session of no coin tree (`FirstFresh`); that the first
session's read shows no `Hello` on two windows is not established here. -/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled.Binding (Read)

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

theorem forall₂_append_left {α β : Type} {R : α → β → Prop} : ∀ {a b : List α} {c : List β},
    List.Forall₂ R (a ++ b) c → ∃ c₁ c₂, c = c₁ ++ c₂ ∧ List.Forall₂ R a c₁ ∧ List.Forall₂ R b c₂
  | [], _, c, h => ⟨[], c, rfl, .nil, h⟩
  | x :: a, b, c, h => by
    rw [List.cons_append] at h
    cases h with
    | cons hx ht =>
      obtain ⟨c₁, c₂, rfl, h₁, h₂⟩ := forall₂_append_left ht
      exact ⟨_ :: c₁, c₂, rfl, .cons hx h₁, h₂⟩

theorem forall₂_mem_of_right {α β : Type} {R : α → β → Prop} : ∀ {a : List α} {b : List β}, List.Forall₂ R a b →
    ∀ y ∈ b, ∃ x ∈ a, R x y
  | _, _, .nil, y, hy => absurd hy List.not_mem_nil
  | _, _, .cons hx ht, y, hy => by
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨_, List.mem_cons_self, hx⟩
    · obtain ⟨x, hx', hR⟩ := forall₂_mem_of_right ht y hy
      exact ⟨x, List.mem_cons_of_mem _ hx', hR⟩

def InteriorCut (C : CaseOf) (Good : Exchange → Prop) (first : Exchange → List Release) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ l, interiorRead C o = some l →
    List.Forall₂ Read.Of (l.map Read.ofV2) o.log ∨
      List.Forall₂ Read.Of ((first o ++ l).map Read.ofV2) o.log

/-- The first session's read samples no coin tree and opens none, because the interior's read numbers its
coin trees from its own log's first and the device numbers them from the play's first. -/
def FirstFresh (C : CaseOf) (Good : Exchange → Prop) (first : Exchange → List Release) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ r ∈ first o, openTree r.frame = none ∧ ∀ sp, r.frame ≠ .hello sp

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
