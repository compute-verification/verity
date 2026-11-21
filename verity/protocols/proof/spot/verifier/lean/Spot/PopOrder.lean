/-
`Slots.popOk`, the run's `population-order` check, decided from the ranges. `popOk` is stated on `Slots.layout`'s
population, and `layout` builds the input class too, a proof block per root input gate, which `popOk` never reads.
`popOkRanges` reads only the query's ranges (`Partition.evaluate`) and its units' gates (`Partition.unitGates`): the
strata's segments (`Draw.strataOf`, as `layout` reads them) chain `0, 1, …, n - 1` (`chain_iff`), and each of those units
has gates. `popOk_eq_popOkRanges` is the equality, so `popOk` runs as `popOkRanges` in every module compiled after this
one (`Run.run` and `OneStageZk.popHeld` among them), and `popOk` itself is unchanged. A module that calls `popOk`
without this one among its imports runs it as stated, off the layout.
-/
module

public import Spot.Slots
public import Spot.FilePart

/-!
# The run's `population-order` check, decided from the ranges

`chain` and `segsOf`, the population's segments in slot order; `popOkRanges`, `Slots.popOk` read off the query's ranges
and unit gates with no proof block built; `popCoverOk`, its cover form, which a subset sampling design's check reads
(`OneStageZk.popHeld`); the lemmas up to `popOk_of` that tie `popOk` to `popOkRanges`; and `popOk_eq_popOkRanges`,
the `@[csimp]` equality under which `popOk` runs as `popOkRanges`. `Spot.Run` and `Spot.OneStageZkStatement` import it, so the
run's `population-order` checks and the one-stage check's (`OneStageZk.popHeld`) run `popOkRanges`.
-/

@[expose] public section

namespace Spot.Slots

open Lean (Json)
open Flock (V)

/-- Segments `[lo, hi)`, read in order, are `a, a + 1, …, b - 1`: each nonempty one starts where the last ended, and the
last ends at `b`. -/
def chain : List (Nat × Nat) → Nat → Nat → Bool
  | [], a, b => a == b
  | (lo, hi) :: ss, a, b => if hi ≤ lo then chain ss a b else lo == a && chain ss hi b

/-- The units of segments `[lo, hi)`, in order. -/
def segUnits (ss : List (Nat × Nat)) : List Nat := ss.flatMap fun s => List.range' s.1 (s.2 - s.1)

/-- The population's segments, in slot order: each stratum's ranges (`Draw.strataOf`), stratum by stratum. -/
def segsOf (rs : Array Flock.TemplateQuery.TRange) : List (Nat × Nat) :=
  (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).toList.flatMap (·.2.toList)

/-- `popOk` from the ranges, with no proof block and no input gate built: the query's ranges and unit gates are
defined, its strata's segments chain `0, 1, …, n - 1`, and every one of those units has its gates. -/
def popOkRanges (p : Program) (query : Json) (n : Nat) : Bool :=
  match Partition.evaluate p query with
  | .error _ => false
  | .ok rs =>
    match Partition.unitGates p query with
    | .error _ => false
    | .ok gs => chain (segsOf rs) 0 n && decide (n ≤ gs.size)

/-- **`popOk`'s cover form**, a subset sampling design's population check: the query's ranges and unit gates are
defined, the strata's segments together, in any order, are a disjoint cover of `0, 1, …, n - 1`
(`FilePart.coverRanges` of `segsOf`), and every one of those units has its gates. -/
def popCoverOk (p : Program) (query : Json) (n : Nat) : Bool :=
  match Partition.evaluate p query with
  | .error _ => false
  | .ok rs =>
    match Partition.unitGates p query with
    | .error _ => false
    | .ok gs => FilePart.coverRanges (segsOf rs) == some n && decide (n ≤ gs.size)

theorem chain_le : ∀ (ss : List (Nat × Nat)) (a b : Nat), chain ss a b = true → a ≤ b := by
  intro ss
  induction ss with
  | nil => intro a b h; simp [chain] at h; omega
  | cons s ss ih =>
    intro a b h
    obtain ⟨lo, hi⟩ := s
    simp only [chain] at h
    split at h
    · exact ih a b h
    · simp only [Bool.and_eq_true, beq_iff_eq] at h
      have := ih hi b h.2
      omega

theorem chain_iff : ∀ (ss : List (Nat × Nat)) (a m : Nat),
    chain ss a (a + m) = true ↔ segUnits ss = List.range' a m := by
  intro ss
  induction ss with
  | nil =>
    intro a m
    simp only [chain, segUnits, List.flatMap_nil, beq_iff_eq]
    constructor
    · intro h; have : m = 0 := by omega
      subst this; rfl
    · intro h
      have := congrArg List.length h
      simp at this
      omega
  | cons s ss ih =>
    intro a m
    obtain ⟨lo, hi⟩ := s
    simp only [chain, segUnits, List.flatMap_cons] at ih ⊢
    split
    · rename_i hle
      have : hi - lo = 0 := by omega
      rw [this, List.range'_zero, List.nil_append]
      exact ih a m
    · rename_i hlt
      simp only [Bool.and_eq_true, beq_iff_eq]
      constructor
      · rintro ⟨hlo, hc⟩
        subst hlo
        have hle := chain_le ss hi (lo + m) hc
        have e : lo + m = hi + (lo + m - hi) := by omega
        rw [e] at hc
        rw [(ih hi (lo + m - hi)).1 hc]
        have := @List.range'_append lo (hi - lo) (lo + m - hi) 1
        simp only [Nat.one_mul] at this
        rw [show lo + (hi - lo) = hi by omega, show hi - lo + (lo + m - hi) = m by omega] at this
        exact this
      · intro h
        have hk : 0 < hi - lo := by omega
        have hlen := congrArg List.length h
        simp only [List.length_append, List.length_range'] at hlen
        have hm : hi - lo ≤ m := by omega
        have split := @List.range'_append a (hi - lo) (m - (hi - lo)) 1
        simp only [Nat.one_mul] at split
        rw [show hi - lo + (m - (hi - lo)) = m by omega] at split
        rw [← split] at h
        have hlo : lo = a := by
          obtain ⟨k, hk'⟩ : ∃ k, hi - lo = k + 1 := ⟨hi - lo - 1, by omega⟩
          rw [hk', List.range'_succ, List.range'_succ] at h
          simp only [List.cons_append, List.cons.injEq] at h
          exact h.1
        subst hlo
        refine ⟨rfl, ?_⟩
        have hrest := List.append_cancel_left h
        rw [show lo + (hi - lo) = hi by omega] at hrest
        rw [show lo + m = hi + (m - (hi - lo)) by omega]
        exact (ih hi (m - (hi - lo))).2 hrest

theorem mapM_ok {α β γ ε : Type} (f : α → Except ε β) (g : β → γ) (h : α → γ) :
    ∀ (l : List α), (∀ x ∈ l, ∃ b, f x = .ok b ∧ g b = h x) → ∃ bs, l.mapM f = .ok bs ∧ bs.map g = l.map h := by
  intro l
  induction l with
  | nil => intro _; exact ⟨[], rfl, rfl⟩
  | cons a l ih =>
    intro hl
    obtain ⟨b, hb, hgb⟩ := hl a (by simp)
    obtain ⟨bs, hbs, hgbs⟩ := ih (fun x hx => hl x (by simp [hx]))
    refine ⟨b :: bs, ?_, ?_⟩
    · rw [List.mapM_cons, hb, hbs]; rfl
    · simp [hgb, hgbs]

theorem mapM_err {α β ε : Type} (f : α → Except ε β) :
    ∀ (l : List α), (∃ x ∈ l, ∃ e, f x = .error e) → ∃ e, l.mapM f = .error e := by
  intro l
  induction l with
  | nil => intro ⟨x, hx, _⟩; simp at hx
  | cons a l ih =>
    intro ⟨x, hx, e, he⟩
    rw [List.mapM_cons]
    cases hfa : f a with
    | error e' => exact ⟨e', rfl⟩
    | ok b =>
      have hx' : x ∈ l := by
        rcases List.mem_cons.1 hx with rfl | h
        · rw [hfa] at he; cases he
        · exact h
      obtain ⟨e'', he''⟩ := ih ⟨x, hx', e, he⟩
      refine ⟨e'', ?_⟩
      show (do let bs ← List.mapM f l; pure (b :: bs)) = Except.error e''
      rw [he'']; rfl

theorem flat_toList (ranges : Array (Nat × Nat)) :
    (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) ranges).toList =
      segUnits ranges.toList := by
  simp only [Array.toList_flatMap, Array.toList_map, Array.toList_range, segUnits, List.range'_eq_map_range]

/-- The layout's population, stratum by stratum, where every unit's block is defined and is that unit's. -/
theorem strata_ok {β : Type} (unitOf : β → Option Nat) (G : Nat → V β) (S : Array (String × Array (Nat × Nat)))
    {r : V (Array (Array β))}
    (hr : S.mapM (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd)) = r)
    (hG : ∀ x ∈ S.toList, ∀ u ∈ segUnits x.2.toList, ∃ b, G u = .ok b ∧ unitOf b = some u) :
    ∃ v, r = .ok v ∧ ((v.toList.map Array.toList).flatten).map unitOf =
      (S.toList.flatMap fun x => segUnits x.2.toList).map some := by
  subst hr
  obtain ⟨vs, hvs, hmap⟩ := mapM_ok (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd))
    (fun v => v.toList.map unitOf) (fun x => (segUnits x.2.toList).map some) S.toList (fun x hx => by
      obtain ⟨bs, hbs, hb⟩ := mapM_ok G unitOf some _ (hG x hx)
      refine ⟨bs.toArray, ?_, by simpa using hb⟩
      rw [Array.mapM_eq_mapM_toList, flat_toList, hbs]; rfl)
  refine ⟨vs.toArray, ?_, ?_⟩
  · rw [Array.mapM_eq_mapM_toList, hvs]; rfl
  · rw [List.map_flatten, List.toList_toArray, List.map_map, List.map_flatMap]
    have : (List.map unitOf ∘ Array.toList) = fun v : Array β => v.toList.map unitOf := rfl
    rw [this, hmap, List.flatten_eq_flatMap, List.flatMap_map]
    rfl

theorem strata_err {β : Type} (G : Nat → V β) (S : Array (String × Array (Nat × Nat))) {r : V (Array (Array β))}
    (hr : S.mapM (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd)) = r)
    (hG : ∃ x ∈ S.toList, ∃ u ∈ segUnits x.2.toList, ∃ e, G u = .error e) : ∃ e, r = .error e := by
  subst hr
  obtain ⟨x, hx, u, hu, e, he⟩ := hG
  rw [Array.mapM_eq_mapM_toList]
  obtain ⟨e', he'⟩ := mapM_err G (segUnits x.2.toList) ⟨u, hu, e, he⟩
  obtain ⟨e'', he''⟩ := mapM_err (fun x => Array.mapM G
      (Array.flatMap (fun x => Array.map (fun x_1 => x.fst + x_1) (Array.range (x.snd - x.fst))) x.snd))
    S.toList ⟨x, hx, e', by rw [Array.mapM_eq_mapM_toList, flat_toList, he']; rfl⟩
  exact ⟨e'', by rw [he'']; rfl⟩

theorem segUnits_segsOf (rs : Array Flock.TemplateQuery.TRange) :
    segUnits (segsOf rs) = ((Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).toList.flatMap
      fun x => segUnits x.2.toList) := by
  simp only [segsOf, segUnits, List.flatMap_assoc]

/-- Where every segment unit has gates, the layout is defined and its population is those units, in order. -/
theorem layout_ok (p : Program) (query : Json) (rs : Array Flock.TemplateQuery.TRange) (gs : Array (Nat × Nat))
    (h1 : Partition.evaluate p query = .ok rs) (h2 : Partition.unitGates p query = .ok gs)
    (hall : ∀ u ∈ segUnits (segsOf rs), u < gs.size) :
    ∃ l, layout p query = .ok l ∧ (popOf l).map (·.unit) = (segUnits (segsOf rs)).map some := by
  have hG : ∀ x ∈ (Flock.Draw.strataOf (rs.map fun r => (r.template, r.first, r.count))).toList, ∀ u ∈ segUnits x.2.toList,
      u < gs.size := fun x hx u hu => hall u (by rw [segUnits_segsOf]; exact List.mem_flatMap.2 ⟨x, hx, hu⟩)
  unfold layout
  rw [h1, h2]
  simp only [bind, Except.bind]
  split
  · rename_i err heq
    obtain ⟨v, hv, _⟩ := strata_ok (·.unit) _ _ heq (by
      intro x hx u hu
      have hlt := hG x hx u hu
      simp only [Array.getElem?_eq_getElem hlt]
      exact ⟨_, rfl, rfl⟩)
    cases hv
  · rename_i v heq
    obtain ⟨v', hv, hvu⟩ := strata_ok (·.unit) _ _ heq (by
      intro x hx u hu
      have hlt := hG x hx u hu
      simp only [Array.getElem?_eq_getElem hlt]
      exact ⟨_, rfl, rfl⟩)
    cases hv
    refine ⟨_, rfl, ?_⟩
    rw [segUnits_segsOf, ← hvu]
    rfl

/-- Where a segment unit has no gates, the layout is an error. -/
theorem layout_err (p : Program) (query : Json) (rs : Array Flock.TemplateQuery.TRange) (gs : Array (Nat × Nat))
    (h1 : Partition.evaluate p query = .ok rs) (h2 : Partition.unitGates p query = .ok gs) (u : Nat)
    (hu : u ∈ segUnits (segsOf rs)) (hge : gs.size ≤ u) : ∃ e, layout p query = .error e := by
  rw [segUnits_segsOf] at hu
  obtain ⟨x, hx, hux⟩ := List.mem_flatMap.1 hu
  unfold layout
  rw [h1, h2]
  simp only [bind, Except.bind]
  split
  · exact ⟨_, rfl⟩
  · rename_i v heq
    obtain ⟨e, he⟩ := strata_err _ _ heq ⟨x, hx, u, hux, _, by simp only [Array.getElem?_eq_none hge]; rfl⟩
    cases he

theorem popOk_of (p : Program) (query : Json) (n : Nat) (rs : Array Flock.TemplateQuery.TRange) (gs : Array (Nat × Nat))
    (h1 : Partition.evaluate p query = .ok rs) (h2 : Partition.unitGates p query = .ok gs) :
    popOk p query n = (chain (segsOf rs) 0 n && decide (n ≤ gs.size)) := by
  have hc : chain (segsOf rs) 0 n = decide (segUnits (segsOf rs) = List.range n) := by
    have := chain_iff (segsOf rs) 0 n
    rw [Nat.zero_add, ← List.range_eq_range'] at this
    by_cases h : segUnits (segsOf rs) = List.range n
    · simp [this.2 h, h]
    · have : chain (segsOf rs) 0 n = false := by
        cases hh : chain (segsOf rs) 0 n
        · rfl
        · exact absurd (this.1 hh) h
      rw [this]; simp [h]
  rw [hc]
  by_cases hall : ∀ u ∈ segUnits (segsOf rs), u < gs.size
  · obtain ⟨l, hl, hu⟩ := layout_ok p query rs gs h1 h2 hall
    unfold popOk
    rw [hl]
    simp only [hu]
    rw [Bool.eq_iff_iff]
    simp only [decide_eq_true_eq, Bool.and_eq_true]
    rw [List.map_inj_right (fun _ _ h => Option.some.inj h)]
    constructor
    · intro h
      refine ⟨h, ?_⟩
      cases n with
      | zero => exact Nat.zero_le _
      | succ k => exact hall k (by rw [h]; simp)
    · exact fun h => h.1
  · obtain ⟨u, hu, hge⟩ : ∃ u ∈ segUnits (segsOf rs), gs.size ≤ u := by
      simp only [Classical.not_forall, Nat.not_lt] at hall
      obtain ⟨u, hu, h⟩ := hall
      exact ⟨u, hu, h⟩
    obtain ⟨e, he⟩ := layout_err p query rs gs h1 h2 u hu hge
    unfold popOk
    rw [he]
    symm
    simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
    by_cases h : segUnits (segsOf rs) = List.range n
    · right
      have : u < n := by rw [h] at hu; simpa using hu
      omega
    · left; exact h

/-- `popOk` runs as `popOkRanges`: the layout's population is defined and its slot `u` is unit `u` exactly when the
query's strata chain `0, …, n - 1` and every such unit has gates. -/
@[csimp] theorem popOk_eq_popOkRanges : @popOk = @popOkRanges := by
  funext p query n
  unfold popOkRanges
  cases h1 : Partition.evaluate p query with
  | error e => unfold popOk layout; rw [h1]; rfl
  | ok rs =>
    cases h2 : Partition.unitGates p query with
    | error e => unfold popOk layout; rw [h1, h2]; rfl
    | ok gs => exact popOk_of p query n rs gs h1 h2

theorem foldlM_chain : ∀ (ss : List (Nat × Nat)) (a b : Nat), (∀ s ∈ ss, s.1 < s.2) →
    (ss.foldlM (fun at_ r => if r.1 == at_ then some r.2 else none) a = some b ↔ chain ss a b = true) := by
  intro ss
  induction ss with
  | nil => intro a b _; simp [chain]
  | cons s ss ih =>
    intro a b hs
    obtain ⟨lo, hi⟩ := s
    have hlt : lo < hi := hs (lo, hi) (by simp)
    have hrest : ∀ s ∈ ss, s.1 < s.2 := fun s h => hs s (by simp [h])
    have hc : chain ((lo, hi) :: ss) a b = (lo == a && chain ss hi b) := by
      simp only [chain, show ¬ hi ≤ lo by omega, ↓reduceIte]
    rw [List.foldlM_cons, hc]
    by_cases h : lo = a
    · subst h
      simp only [beq_self_eq_true, ite_true, Option.bind_eq_bind, Option.bind_some, Bool.true_and]
      exact ih hi b hrest
    · simp [h]

theorem segUnits_filter : ∀ (ss : List (Nat × Nat)),
    segUnits (ss.filter fun r => decide (r.1 < r.2)) = segUnits ss := by
  intro ss
  induction ss with
  | nil => rfl
  | cons s ss ih =>
    obtain ⟨lo, hi⟩ := s
    by_cases h : lo < hi
    · simp only [List.filter_cons, h, decide_true, ite_true, segUnits, List.flatMap_cons] at ih ⊢
      rw [ih]
    · simp only [List.filter_cons, h, decide_false, Bool.false_eq_true, ite_false, segUnits, List.flatMap_cons] at ih ⊢
      rw [ih, show hi - lo = 0 by omega, List.range'_zero, List.nil_append]

/-- Nonempty segments sorted by start whose units are a permutation of `a, …, a + m - 1` chain from `a` to `a + m`. -/
theorem chain_of_perm : ∀ (ts : List (Nat × Nat)) (a m : Nat), ts.Pairwise (fun x y => x.1 ≤ y.1) →
    (∀ s ∈ ts, s.1 < s.2) → (segUnits ts).Perm (List.range' a m) → chain ts a (a + m) = true := by
  intro ts
  induction ts with
  | nil =>
    intro a m _ _ hp
    have := hp.length_eq
    simp only [segUnits, List.flatMap_nil, List.length_nil, List.length_range'] at this
    subst this
    simp [chain]
  | cons s ts ih =>
    intro a m hsort hne hp
    obtain ⟨lo, hi⟩ := s
    rw [List.pairwise_cons] at hsort
    have hlt : lo < hi := hne (lo, hi) (by simp)
    have hne' : ∀ s ∈ ts, s.1 < s.2 := fun s h => hne s (by simp [h])
    have hunits : segUnits ((lo, hi) :: ts) = List.range' lo (hi - lo) ++ segUnits ts := by
      simp only [segUnits, List.flatMap_cons]
    rw [hunits] at hp
    have mem : ∀ u, u ∈ List.range' lo (hi - lo) ++ segUnits ts ↔ a ≤ u ∧ u < a + m := fun u => by
      rw [hp.mem_iff, List.mem_range'_1]
    have hlo := (mem lo).1 (List.mem_append_left _ (List.mem_range'_1.2 ⟨Nat.le_refl _, by omega⟩))
    have hhi := (mem (hi - 1)).1 (List.mem_append_left _ (List.mem_range'_1.2 ⟨by omega, by omega⟩))
    have ha : lo ≤ a := by
      rcases List.mem_append.1 ((mem a).2 ⟨Nat.le_refl _, by omega⟩) with h | h
      · exact (List.mem_range'_1.1 h).1
      · obtain ⟨y, hy, hay⟩ := List.mem_flatMap.1 h
        exact Nat.le_trans (hsort.1 y hy) (List.mem_range'_1.1 hay).1
    have e : lo = a := by omega
    subst e
    have split := @List.range'_append lo (hi - lo) (lo + m - hi) 1
    simp only [Nat.one_mul] at split
    rw [show lo + (hi - lo) = hi by omega, show hi - lo + (lo + m - hi) = m by omega] at split
    rw [← split] at hp
    have hc := ih hi (lo + m - hi) hsort.2 hne' (List.perm_append_left_iff _ |>.1 hp)
    rw [show hi + (lo + m - hi) = lo + m by omega] at hc
    simp [chain, show ¬ hi ≤ lo by omega, hc]

/-- **The cover is the units' permutation**: `FilePart.coverRanges ss` is `n` exactly when the units of `ss` are
`0, …, n - 1`, each once, in any order. -/
theorem coverRanges_iff (ss : List (Nat × Nat)) (n : Nat) :
    FilePart.coverRanges ss = some n ↔ (segUnits ss).Perm (List.range n) := by
  let fs := ss.filter fun r => decide (r.1 < r.2)
  let ts := fs.mergeSort fun x y => decide (x.1 ≤ y.1)
  have hperm : ts.Perm fs := List.mergeSort_perm _ _
  have hne : ∀ s ∈ ts, s.1 < s.2 := fun s h => by
    have := (List.mem_filter.1 (hperm.mem_iff.1 h)).2
    simpa using this
  have hsort : ts.Pairwise (fun x y => x.1 ≤ y.1) :=
    (List.pairwise_mergeSort (le := fun x y : Nat × Nat => decide (x.1 ≤ y.1))
      (fun a b c h1 h2 => by simp only [decide_eq_true_eq] at *; omega)
      (fun a b => by simp only [Bool.or_eq_true, decide_eq_true_eq]; omega)
      fs).imp (fun h => by simpa using h)
  have hunits : (segUnits ts).Perm (segUnits ss) := by
    rw [← segUnits_filter ss]
    exact hperm.flatMap_right _
  show ts.foldlM _ 0 = some n ↔ _
  rw [foldlM_chain ts 0 n hne]
  constructor
  · intro h
    have := (chain_iff ts 0 n).1 (by rw [Nat.zero_add]; exact h)
    rw [← List.range_eq_range'] at this
    exact hunits.symm.trans (by rw [this])
  · intro h
    have := chain_of_perm ts 0 n hsort hne (by rw [← List.range_eq_range']; exact hunits.trans h)
    rwa [Nat.zero_add] at this

theorem popCovers_of (p : Program) (query : Json) (n : Nat) (rs : Array Flock.TemplateQuery.TRange)
    (gs : Array (Nat × Nat)) (h1 : Partition.evaluate p query = .ok rs) (h2 : Partition.unitGates p query = .ok gs) :
    popCovers p query n = (FilePart.coverRanges (segsOf rs) == some n && decide (n ≤ gs.size)) := by
  have hc : (FilePart.coverRanges (segsOf rs) == some n) = decide ((segUnits (segsOf rs)).Perm (List.range n)) := by
    rw [Bool.eq_iff_iff, beq_iff_eq, decide_eq_true_eq]
    exact coverRanges_iff _ _
  rw [hc]
  by_cases hall : ∀ u ∈ segUnits (segsOf rs), u < gs.size
  · obtain ⟨l, hl, hu⟩ := layout_ok p query rs gs h1 h2 hall
    unfold popCovers
    rw [hl]
    simp only [hu]
    rw [Bool.eq_iff_iff]
    simp only [decide_eq_true_eq, Bool.and_eq_true]
    have hm : ((segUnits (segsOf rs)).map some).Perm ((List.range n).map some) ↔
        (segUnits (segsOf rs)).Perm (List.range n) :=
      ⟨fun h => by simpa [List.map_map, Function.comp_def] using h.map (·.getD 0), fun h => h.map _⟩
    rw [hm]
    constructor
    · intro h
      refine ⟨h, ?_⟩
      cases n with
      | zero => exact Nat.zero_le _
      | succ k => exact hall k (h.mem_iff.2 (by simp))
    · exact fun h => h.1
  · obtain ⟨u, hu, hge⟩ : ∃ u ∈ segUnits (segsOf rs), gs.size ≤ u := by
      simp only [Classical.not_forall, Nat.not_lt] at hall
      obtain ⟨u, hu, h⟩ := hall
      exact ⟨u, hu, h⟩
    obtain ⟨e, he⟩ := layout_err p query rs gs h1 h2 u hu hge
    unfold popCovers
    rw [he]
    symm
    simp only [Bool.and_eq_false_iff, decide_eq_false_iff_not]
    by_cases h : (segUnits (segsOf rs)).Perm (List.range n)
    · right
      have : u < n := by simpa using h.mem_iff.1 hu
      omega
    · left; exact h

/-- `popCovers` runs as `popCoverOk`: the layout's population is defined and its units are `0, …, n - 1` in any order
exactly when the query's strata's segments cover them (`coverRanges_iff`) and every such unit has gates. -/
@[csimp] theorem popCovers_eq_popCoverOk : @popCovers = @popCoverOk := by
  funext p query n
  unfold popCoverOk
  cases h1 : Partition.evaluate p query with
  | error e => unfold popCovers layout; rw [h1]; rfl
  | ok rs =>
    cases h2 : Partition.unitGates p query with
    | error e => unfold popCovers layout; rw [h1, h2]; rfl
    | ok gs => exact popCovers_of p query n rs gs h1 h2

end Spot.Slots
