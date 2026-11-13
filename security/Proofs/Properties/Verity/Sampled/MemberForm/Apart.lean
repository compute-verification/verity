module

public import Proofs.Properties.Verity.Sampled.MemberForm.Accept
public import Proofs.Core.QSort
public import Proofs.Core.QSortPerm

/-!
# The parameters' members share no unit

The per-member form's parameters are read by `Spot.OneStageZk.membersOf`, which sorts every member's ranges by
their starts and refuses a range that starts before the one before it ends. This file proves what that check gives:
no unit is in two members' ranges (`Apart`).

* **The loop ran at every place** (`forIn_yields`): a loop that returns ran its body at every entry, where a pass
  of the body yields.
* **Sorted ranges that each end by the next one's start share no unit** (`apart_of_adjacent`): the sorted ranges
  are a permutation of the members' (`Verity.QSort.qsort_perm`), in the order of their starts
  (`Verity.QSort.ByKey.qsort_sorted`), so a range ends by the start of every later one.
* **The parse's members share no unit** (`membersOf_apart`, `membersParamsOfJson_apart`,
  `paramsEitherOfJson_apart`: in either form, as the one statement's parse names no members).
* **The form's accepts' members share no unit** (`FormApart`): of everything v2 accepts at the coins it read
  (`V2Accepts`) on parameters that name members. It holds of the check (`formApart`), which reads them by
  `paramsEitherOfJson`: its accept carries that parse.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Properties.Challenger (Draws)
open FlockSoundness.Refine (ebind_ok)

/-- **A loop over a list that returns ran its body at every entry**, where a pass of the body yields. -/
theorem forIn_yields {f : ℕ → PUnit.{1} → Except String (ForInStep PUnit.{1})}
    (hy : ∀ i s, f i ⟨⟩ = .ok s → s = .yield ⟨⟩) :
    ∀ (l : List ℕ) {u : PUnit.{1}}, forIn l ⟨⟩ f = .ok u → ∀ i ∈ l, f i ⟨⟩ = .ok (.yield ⟨⟩)
  | [], _, _, i, hi => absurd hi List.not_mem_nil
  | a :: l, u, h, i, hi => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := ebind_ok h
    have hs' := hy a s hs
    subst hs'
    rcases List.mem_cons.1 hi with rfl | hi
    · exact hs
    · exact forIn_yields hy l h i hi

/-- The ranges a fold appends are the members' ranges, in the members' order. -/
theorem foldl_units (l : List _root_.Spot.OneStageZk.Member) (init : Array (ℕ × ℕ)) :
    (l.foldl (fun acc m => acc ++ m.units) init).toList = init.toList ++ l.flatMap fun m => m.units.toList := by
  induction l generalizing init with
  | nil => simp
  | cons m l ih => rw [List.foldl_cons, ih, Array.toList_append, List.flatMap_cons, List.append_assoc]

/-- **No unit is in two of the members' ranges.** -/
def Apart (ms : List _root_.Spot.OneStageZk.Member) : Prop :=
  ms.Pairwise fun a b => ∀ x ∈ a.units.toList, ∀ y ∈ b.units.toList,
    ∀ u, x.1 ≤ u → u < x.2 → y.1 ≤ u → u < y.2 → False

/-- The members' ranges, in the order of their starts, as `Spot.OneStageZk.membersOf` sorts them. -/
def sortedUnits (ms : Array _root_.Spot.OneStageZk.Member) : Array (ℕ × ℕ) :=
  (ms.foldl (fun acc m => acc ++ m.units) #[]).qsort (·.1 < ·.1)

/-- **Members whose ranges, sorted by their starts, each end by the next one's start share no unit.** -/
theorem apart_of_adjacent (ms : Array _root_.Spot.OneStageZk.Member)
    (hadj : ∀ i, 1 ≤ i → i < (sortedUnits ms).size → (sortedUnits ms)[i - 1]!.2 ≤ (sortedUnits ms)[i]!.1) :
    Apart ms.toList := by
  have key : ∀ rs : Array (ℕ × ℕ), rs.Perm (ms.foldl (fun acc m => acc ++ m.units) #[]) →
      rs.toList.Pairwise (fun x y => x.1 ≤ y.1) → (∀ i, 1 ≤ i → i < rs.size → rs[i - 1]!.2 ≤ rs[i]!.1) →
      Apart ms.toList := by
    intro rs hperm hsort hadj
    have hD : rs.toList.Pairwise fun x y => ∀ u, x.1 ≤ u → u < x.2 → y.1 ≤ u → u < y.2 → False := by
      rw [List.pairwise_iff_getElem] at hsort ⊢
      intro i j hi hj hij u h1 h2 h3 h4
      have hi' : i < rs.size := by simpa using hi
      have hj' : j < rs.size := by simpa using hj
      have ha := hadj (i + 1) (by omega) (by omega)
      rw [Nat.add_sub_cancel, getElem!_pos rs i hi', getElem!_pos rs (i + 1) (by omega)] at ha
      have hs : rs[i + 1].1 ≤ rs[j].1 := by
        rcases Nat.lt_or_ge (i + 1) j with hlt | hge
        · have := hsort (i + 1) j (by simpa using (by omega : i + 1 < rs.size)) hj hlt
          simpa only [Array.getElem_toList] using this
        · have e : j = i + 1 := by omega
          subst e
          exact Nat.le_refl _
      simp only [Array.getElem_toList] at h2 h3
      omega
    have hD' := (List.Perm.pairwise_iff (fun {x y} h u a b c d => h u c d a b)
      (Array.perm_iff_toList_perm.1 hperm)).1 hD
    rw [← Array.foldl_toList, foldl_units] at hD'
    exact (List.pairwise_flatMap.1 (by simpa using hD')).2
  exact key _ (_root_.Verity.QSort.qsort_perm _ _)
    (_root_.Verity.QSort.ByKey.qsort_sorted (fun x : ℕ × ℕ => x.1) _) hadj

/-- **The members `membersOf` reads share no unit**: it sorts every member's ranges by their starts and refuses a
range that starts before the one before it ends. -/
theorem membersOf_apart {shared : _root_.Spot.OneStageZk.Own} {j : Json}
    {ms : Array _root_.Spot.OneStageZk.Member} (h : _root_.Spot.OneStageZk.membersOf shared j = .ok ms) :
    Apart ms.toList := by
  unfold _root_.Spot.OneStageZk.membersOf at h
  split at h
  · obtain ⟨ms', -, h⟩ := ebind_ok h
    dsimp only at h
    split at h
    · obtain ⟨_, h0, -⟩ := ebind_ok h
      cases h0
    · obtain ⟨u, hfor, h⟩ := ebind_ok h
      obtain rfl : ms' = ms := Except.ok.inj h
      rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hfor
      have hall := forIn_yields (fun i s hs => by
        try dsimp only at hs
        split at hs
        · obtain ⟨_, h0, -⟩ := ebind_ok hs
          cases h0
        · exact (Except.ok.inj hs).symm) _ hfor
      refine apart_of_adjacent ms' fun i h1 h2 => ?_
      unfold sortedUnits at h2 ⊢
      have hi := hall i (by simp only [List.mem_range'_1, Std.Legacy.Range.size]; omega)
      try dsimp only at hi
      split at hi
      · obtain ⟨_, h0, -⟩ := ebind_ok hi
        cases h0
      · rename_i hc
        exact Nat.le_of_not_lt hc
  · cases h

/-- **The per-member form's parameters' members share no unit**: they are the ones `membersOf` reads. -/
theorem membersParamsOfJson_apart {j : Json} {p : _root_.Spot.OneStageZk.Params}
    (h : _root_.Spot.OneStageZk.membersParamsOfJson j = .ok p) {ms : Array _root_.Spot.OneStageZk.Member}
    (hm : p.members = some ms) : Apart ms.toList := by
  unfold _root_.Spot.OneStageZk.membersParamsOfJson at h
  split at h
  · obtain ⟨⟨⟩, -, h⟩ := ebind_ok h
    try dsimp only at h
    obtain ⟨p0, -, h⟩ := ebind_ok h
    try dsimp only at h
    obtain ⟨ms', hms', h⟩ := ebind_ok h
    cases h
    cases hm
    exact membersOf_apart hms'
  · cases h

/-- **Parameters parsed in either form name members that share no unit**: the per-member form's parse reads them by
`membersOf` (`membersParamsOfJson_apart`), and the one statement's parse names none
(`Spot.OneStageZk.paramsOfJson_members`). -/
theorem paramsEitherOfJson_apart {j : Json} {p : _root_.Spot.OneStageZk.Params}
    (h : _root_.Spot.OneStageZk.paramsEitherOfJson j = .ok p) {ms : Array _root_.Spot.OneStageZk.Member}
    (hm : p.members = some ms) : Apart ms.toList := by
  unfold _root_.Spot.OneStageZk.paramsEitherOfJson at h
  split at h
  · exact membersParamsOfJson_apart h hm
  · cases hm.symm.trans (_root_.Spot.OneStageZk.paramsOfJson_members h)

/-- **The per-member form's accepts' members share no unit**: wherever v2 accepts at the coins it read
(`V2Accepts`) on parameters that name members, no unit is in two of their ranges. An accept carries the parse of
its parameters' bytes (`V2Accepts`' first clause), so a check that reads them by `paramsEitherOfJson` has it of
that parse (`paramsEitherOfJson_apart`); the form's run does not give it, which holds each member to one range and
the counts to the population and passes two ranges that share a unit. -/
def FormApart : Prop :=
  ∀ (bs : _root_.Core.Blobs) (a b e : ByteArray) (p : _root_.Spot.OneStageZk.Params)
    (c : _root_.Spot.OneStage.Commitment) (ev : _root_.Spot.OneStageZk.Evidence) (ξ : ByteArray) (dr : Draws)
    (ms : Array _root_.Spot.OneStageZk.Member),
    V2Accepts bs a b e p c ev ξ dr → p.members = some ms → Apart ms.toList

/-- **The check's accepts' members share no unit**: an accept carries the parse of its parameters' bytes, which the
sampled component reads by `paramsEitherOfJson` (`dispatch_params`), and that parse names members that share no unit
(`paramsEitherOfJson_apart`). -/
theorem formApart : FormApart := by
  intro bs a b e p c ev ξ dr ms h hm
  obtain ⟨hp, -⟩ := h
  cases hj : _root_.Core.canonicalJson a with
  | error _ =>
    rw [hj] at hp
    cases hp
  | ok j =>
    rw [hj, Trees.ok_bind] at hp
    obtain ⟨rest, hr⟩ := dispatch_params hp
    exact paramsEitherOfJson_apart hr hm

end Verity.Sampled.MemberForm
