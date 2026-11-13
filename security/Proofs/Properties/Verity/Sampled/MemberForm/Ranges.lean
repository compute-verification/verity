module

public import Mathlib.Data.List.Forall2

/-!
# One range of units, and lists in one order

A member of the per-member form is one range of the registration's units, `[lo, hi)`, and the members' statements
come in the order of their first units. This module holds what the form's proofs read of such a range and of lists
in one order, with no verifier and no window in it: arrays and lists of natural numbers alone.

* **A list read entry by entry** (`mapM_ok_at`): a `mapM` whose function returns each entry of a list of the same
  length returns that list.
* **One range** (`one_range`, `first_range`, `count_range`, `indices_range`, `indices_range_at`, `sampled_range`):
  an array of one range is that range; its first unit is `lo`, its count `hi - lo`, its units `lo, …, hi - 1` in
  order, so the unit at slot `u` is `lo + u`; and a list of units holds one of its units exactly when it holds a
  unit from `lo` up to `hi`.
* **Lists in one order** (`eq_of_key_eq`, `eq_of_sorted`): two members of a list strictly ascending under a key
  with one key are one member, and two such lists with the same members are one list.
* **The kept members' places** (`filter_at`, `filter_length`): the member at place `i` that a filter keeps is the
  filter's member after as many as it keeps before place `i`, and the filter keeps as many as the whole list
  counts.
* **Lists related member by member** (`forall₂_countP`, `forall₂_at`, `forall₂_at_left`, `forall₂_trans`,
  `forall₂_pairwise`): they count alike before each place where the relation carries the predicate, are related
  at each place, compose, and carry a pairwise relation from one to the other.
* **Sessions' units over members of one range each** (`UnitsOver`, `UnitsOver.kept`, `UnitsOver.mem_units`): where
  the members' ranges share no unit, the statements' members are members in order, each session's units are in its
  statement's member's range and the units wanted are the sessions', the statements' members are the members with
  a wanted unit, in the members' order, and a unit of a statement's member's range is wanted exactly when that
  statement's session proved it.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

/-! ## A list read entry by entry -/

/-- A `mapM` whose function returns each entry of `l'` at the entry of `l` at its place returns `l'`. -/
theorem mapM_ok_at {α β ε : Type} (g : α → Except ε β) :
    ∀ (l : List α) (l' : List β), l.length = l'.length →
      (∀ i (hi : i < l.length) (hi' : i < l'.length), g l[i] = .ok l'[i]) → l.mapM g = .ok l'
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => nomatch hl
  | _ :: _, [], hl, _ => nomatch hl
  | a :: l, b :: l', hl, hg => by
    have h0 : g a = .ok b := hg 0 (Nat.zero_lt_succ _) (Nat.zero_lt_succ _)
    have ht : l.mapM g = .ok l' := mapM_ok_at g l l' (Nat.succ.inj hl)
      (fun i hi hi' => hg (i + 1) (Nat.succ_lt_succ hi) (Nat.succ_lt_succ hi'))
    rw [List.mapM_cons, h0, ht]
    rfl

/-! ## One range -/

/-- An array of one range is that range. -/
theorem one_range {us : Array (ℕ × ℕ)} (h : us.size = 1) : ∃ lo hi, us = #[(lo, hi)] := by
  obtain ⟨l⟩ := us
  match l, h with
  | [x], _ => exact ⟨x.1, x.2, rfl⟩

/-- One range's first unit is its lower end. -/
theorem first_range (lo hi : ℕ) :
    (#[(lo, hi)] : Array (ℕ × ℕ)).foldl (fun a x => min a x.1)
      (((#[(lo, hi)] : Array (ℕ × ℕ))[0]?.map (·.1)).getD 0) = lo := by
  show min lo lo = lo
  exact Nat.min_self lo

/-- One range's count is its length. -/
theorem count_range (lo hi : ℕ) :
    (#[(lo, hi)] : Array (ℕ × ℕ)).foldl (fun n x => n + (x.2 - x.1)) 0 = hi - lo := by
  show 0 + (hi - lo) = hi - lo
  exact Nat.zero_add _

/-- One range's units are `lo, …, hi - 1`, in order. -/
theorem indices_range (lo hi : ℕ) :
    ((#[(lo, hi)] : Array (ℕ × ℕ)).flatMap fun x => (Array.range (x.2 - x.1)).map (x.1 + ·)) =
      (Array.range (hi - lo)).map (lo + ·) := by
  show #[] ++ (Array.range (hi - lo)).map (lo + ·) = _
  exact Array.empty_append

/-- The unit at slot `u` of one range's units is `lo + u`, and `u` is below the range's length. -/
theorem indices_range_at {lo hi u v : ℕ} (h : ((Array.range (hi - lo)).map (lo + ·))[u]? = some v) :
    u < hi - lo ∧ v = lo + u := by
  rw [Array.getElem?_map, Array.getElem?_range] at h
  split at h
  · rename_i hlt
    exact ⟨hlt, (Option.some.inj h).symm⟩
  · cases h

/-- A list of units holds a unit of one range exactly when it holds a unit from `lo` up to `hi`. -/
theorem sampled_range (lo hi : ℕ) (want : Array ℕ) :
    (want.any fun u => (#[(lo, hi)] : Array (ℕ × ℕ)).any fun x => decide (x.1 ≤ u) && decide (u < x.2)) = true ↔
      ∃ u ∈ want, lo ≤ u ∧ u < hi := by
  have e : ∀ u : ℕ, ((#[(lo, hi)] : Array (ℕ × ℕ)).any fun x => decide (x.1 ≤ u) && decide (u < x.2)) =
      (decide (lo ≤ u) && decide (u < hi)) := fun u => by
    rw [← Array.any_toList]
    show (decide (lo ≤ u) && decide (u < hi) || false) = _
    exact Bool.or_false _
  rw [Array.any_eq_true']
  constructor
  · rintro ⟨u, hu, hx⟩
    rw [e, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq] at hx
    exact ⟨u, hu, hx⟩
  · rintro ⟨u, hu, hx⟩
    refine ⟨u, hu, ?_⟩
    rw [e, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq]
    exact hx

/-! ## Lists in one order -/

/-- Two members of a list strictly ascending under `key` with one key are one member. -/
theorem eq_of_key_eq {α : Type} {key : α → ℕ} {l : List α} (h : l.Pairwise fun x y => key x < key y) {a b : α}
    (ha : a ∈ l) (hb : b ∈ l) (e : key a = key b) : a = b := by
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha
  obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hb
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · exact absurd (List.pairwise_iff_getElem.1 h i j hi hj hij) (by rw [e]; exact Nat.lt_irrefl _)
  · rfl
  · exact absurd (List.pairwise_iff_getElem.1 h j i hj hi hij) (by rw [e]; exact Nat.lt_irrefl _)

/-- Two lists strictly ascending under one key, with the same members, have one head. -/
theorem head_of_sorted {α : Type} {key : α → ℕ} {a b : α} {l l' : List α}
    (h : (a :: l).Pairwise fun x y => key x < key y) (h' : (b :: l').Pairwise fun x y => key x < key y)
    (hm : ∀ x, x ∈ a :: l ↔ x ∈ b :: l') : a = b := by
  have ha : a ∈ b :: l' := (hm a).1 List.mem_cons_self
  have hb : b ∈ a :: l := (hm b).2 List.mem_cons_self
  rcases List.mem_cons.1 ha with rfl | ha'
  · rfl
  · rcases List.mem_cons.1 hb with rfl | hb'
    · rfl
    · have h1 : key b < key a := (List.pairwise_cons.1 h').1 a ha'
      have h2 : key a < key b := (List.pairwise_cons.1 h).1 b hb'
      exact absurd h1 (Nat.lt_asymm h2)

/-- **Two lists strictly ascending under one key, with the same members, are one list.** -/
theorem eq_of_sorted {α : Type} {key : α → ℕ} : ∀ {l l' : List α},
    l.Pairwise (fun x y => key x < key y) → l'.Pairwise (fun x y => key x < key y) →
    (∀ x, x ∈ l ↔ x ∈ l') → l = l'
  | [], [], _, _, _ => rfl
  | [], b :: l', _, _, hm => nomatch (hm b).2 List.mem_cons_self
  | a :: l, [], _, _, hm => nomatch (hm a).1 List.mem_cons_self
  | a :: l, b :: l', h, h', hm => by
    obtain rfl : a = b := head_of_sorted h h' hm
    have hl := List.pairwise_cons.1 h
    have hl' := List.pairwise_cons.1 h'
    have hm' : ∀ x, x ∈ l ↔ x ∈ l' := fun x => by
      constructor
      · intro hx
        rcases List.mem_cons.1 ((hm x).1 (List.mem_cons_of_mem _ hx)) with rfl | hx'
        · exact absurd (hl.1 x hx) (Nat.lt_irrefl _)
        · exact hx'
      · intro hx
        rcases List.mem_cons.1 ((hm x).2 (List.mem_cons_of_mem _ hx)) with rfl | hx'
        · exact absurd (hl'.1 x hx) (Nat.lt_irrefl _)
        · exact hx'
    rw [eq_of_sorted hl.2 hl'.2 hm']

/-! ## The kept members' places -/

/-- **The member at place `i` that a filter keeps is the filter's member after as many as it keeps before `i`.** -/
theorem filter_at {α : Type} (p : α → Bool) : ∀ (l : List α) (i : ℕ) (a : α), l[i]? = some a → p a = true →
    (l.filter p)[(l.take i).countP p]? = some a
  | [], i, a, h, _ => by
    rw [List.getElem?_nil] at h
    cases h
  | b :: l, 0, a, h, hp => by
    rw [List.getElem?_cons_zero] at h
    obtain rfl := Option.some.inj h
    rw [List.take_zero, List.countP_nil, List.filter_cons_of_pos hp, List.getElem?_cons_zero]
  | b :: l, i + 1, a, h, hp => by
    rw [List.getElem?_cons_succ] at h
    have ih := filter_at p l i a h hp
    rw [List.take_succ_cons, List.countP_cons]
    cases hb : p b with
    | true =>
      rw [List.filter_cons_of_pos hb, ite_eq_left rfl, List.getElem?_cons_succ]
      exact ih
    | false =>
      rw [List.filter_cons_of_neg (by rw [hb]; exact Bool.false_ne_true), ite_eq_right Bool.false_ne_true,
        Nat.add_zero]
      exact ih

/-- A filter keeps as many members as the whole list counts. -/
theorem filter_length {α : Type} (p : α → Bool) (l : List α) :
    (l.filter p).length = (l.take l.length).countP p := by
  rw [List.take_length, List.countP_eq_length_filter]

/-! ## Lists related member by member -/

/-- Lists related member by member count alike before each place, where the relation carries the predicate. -/
theorem forall₂_countP {α β : Type} {R : α → β → Prop} {p : α → Bool} {q : β → Bool} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → (∀ a b, R a b → p a = q b) →
      ∀ i, (l.take i).countP p = (l'.take i).countP q
  | _, _, .nil, _, i => by rw [List.take_nil, List.take_nil, List.countP_nil, List.countP_nil]
  | _, _, .cons _ _, _, 0 => by rw [List.take_zero, List.take_zero, List.countP_nil, List.countP_nil]
  | _, _, .cons hab ht, hpq, i + 1 => by
    rw [List.take_succ_cons, List.take_succ_cons, List.countP_cons, List.countP_cons,
      forall₂_countP ht hpq i, hpq _ _ hab]

/-- Lists related member by member are related at each place of the second. -/
theorem forall₂_at {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ (i : ℕ) (b : β), l'[i]? = some b →
      ∃ a, l[i]? = some a ∧ R a b
  | _, _, .nil, i, b, h => by
    rw [List.getElem?_nil] at h
    cases h
  | _, _, .cons hab _, 0, b, hb => by
    rw [List.getElem?_cons_zero] at hb
    obtain rfl := Option.some.inj hb
    exact ⟨_, List.getElem?_cons_zero, hab⟩
  | _, _, .cons _ ht, i + 1, b, hb => by
    rw [List.getElem?_cons_succ] at hb
    obtain ⟨a', ha', hr⟩ := forall₂_at ht i b hb
    exact ⟨a', by rw [List.getElem?_cons_succ]; exact ha', hr⟩

/-- Lists related member by member are related at each place of the first. -/
theorem forall₂_at_left {α β : Type} {R : α → β → Prop} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ (i : ℕ) (a : α), l[i]? = some a →
      ∃ b, l'[i]? = some b ∧ R a b
  | _, _, .nil, i, a, h => by
    rw [List.getElem?_nil] at h
    cases h
  | _, _, .cons hab _, 0, a, ha => by
    rw [List.getElem?_cons_zero] at ha
    obtain rfl := Option.some.inj ha
    exact ⟨_, List.getElem?_cons_zero, hab⟩
  | _, _, .cons _ ht, i + 1, a, ha => by
    rw [List.getElem?_cons_succ] at ha
    obtain ⟨b', hb', hr⟩ := forall₂_at_left ht i a ha
    exact ⟨b', by rw [List.getElem?_cons_succ]; exact hb', hr⟩

/-- Lists related member by member compose. -/
theorem forall₂_trans {α β γ : Type} {R : α → β → Prop} {S : β → γ → Prop} :
    ∀ {l : List α} {l' : List β} {l'' : List γ}, List.Forall₂ R l l' → List.Forall₂ S l' l'' →
      List.Forall₂ (fun a c => ∃ b, R a b ∧ S b c) l l''
  | _, _, _, .nil, .nil => .nil
  | _, _, _, .cons hab ht, .cons hbc ht' => .cons ⟨_, hab, hbc⟩ (forall₂_trans ht ht')

/-- A pairwise relation of a list goes to a list related to it member by member. -/
theorem forall₂_pairwise {α β : Type} {R : α → β → Prop} {P : β → β → Prop} {Q : α → α → Prop}
    (hPQ : ∀ a a' b b', R a b → R a' b' → P b b' → Q a a') :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → l'.Pairwise P → l.Pairwise Q
  | _, _, .nil, _ => List.Pairwise.nil
  | _, _, .cons hab ht, hp => by
    obtain ⟨hb, hp'⟩ := List.pairwise_cons.1 hp
    refine List.pairwise_cons.2 ⟨fun a' ha' => ?_, forall₂_pairwise hPQ ht hp'⟩
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ha'
    obtain ⟨b', hb', hr⟩ := forall₂_at_left ht i _ (List.getElem?_eq_getElem hi)
    exact hPQ _ _ _ _ hab hr (hb b' (List.mem_of_getElem? hb'))

/-! ## The sessions' units over the members' ranges -/

/-- A member of a list strictly ascending under a key is at one place. -/
theorem idx_of_sorted {α : Type} {key : α → ℕ} {l : List α} (h : l.Pairwise fun x y => key x < key y)
    {i j : ℕ} {a : α} (hi : l[i]? = some a) (hj : l[j]? = some a) : i = j := by
  obtain ⟨hi', ei⟩ := List.getElem?_eq_some_iff.1 hi
  obtain ⟨hj', ej⟩ := List.getElem?_eq_some_iff.1 hj
  rcases Nat.lt_trichotomy i j with hij | rfl | hij
  · have hlt := List.pairwise_iff_getElem.1 h i j hi' hj' hij
    rw [ei, ej] at hlt
    exact absurd hlt (Nat.lt_irrefl _)
  · rfl
  · have hlt := List.pairwise_iff_getElem.1 h j i hj' hi' hij
    rw [ei, ej] at hlt
    exact absurd hlt (Nat.lt_irrefl _)

/-- A unit `lo + u` is among the units `lo + ·` of slots exactly when `u` is among the slots. -/
theorem mem_map_add {lo u : ℕ} {us : Array ℕ} : lo + u ∈ us.map (lo + ·) ↔ u ∈ us := by
  rw [Array.mem_map]
  constructor
  · rintro ⟨v, hv, e⟩
    obtain rfl : v = u := Nat.add_left_cancel e
    exact hv
  · intro hu
    exact ⟨u, hu, rfl⟩

/-- **Sessions' units over members of one range each.** The members `vms`, each of the range `[lo, hi)`, are
strictly ascending by `lo` and no unit is in two members' ranges; the statements' members `held` are members of
`vms`, different members in the order of `lo`; the units `U` the sessions proved, a session a statement in order,
are each in its statement's member's range; and the units `want` are exactly the units some session proved. -/
structure UnitsOver {α : Type} (lo hi : α → ℕ) (vms held : List α) (U : List (Array ℕ)) (want : Array ℕ) :
    Prop where
  sorted : vms.Pairwise fun a b => lo a < lo b
  apart : ∀ a ∈ vms, ∀ b ∈ vms, ∀ x, lo a ≤ x → x < hi a → lo b ≤ x → x < hi b → a = b
  order : held.Pairwise fun a b => a ≠ b ∧ lo a ≤ lo b
  among : ∀ a ∈ held, a ∈ vms
  units : ∀ (j : ℕ) (a : α) (us : Array ℕ), held[j]? = some a → U[j]? = some us → ∀ x ∈ us, lo a ≤ x ∧ x < hi a
  each : ∀ (j : ℕ) (us : Array ℕ), U[j]? = some us → ∃ a, held[j]? = some a
  want : ∀ x, x ∈ want ↔ ∃ us ∈ U, x ∈ us

namespace UnitsOver

variable {α : Type} {lo hi : α → ℕ} {vms held : List α} {U : List (Array ℕ)} {want : Array ℕ}

/-- The statements' members are strictly ascending by `lo`: two of them with one `lo` are one member of `vms`. -/
theorem strict (h : UnitsOver lo hi vms held U want) : held.Pairwise fun a b => lo a < lo b :=
  h.order.imp_of_mem fun {a b} ha hb hab =>
    Nat.lt_of_le_of_ne hab.2 fun e => hab.1 (eq_of_key_eq h.sorted (h.among a ha) (h.among b hb) e)

/-- A unit of `want` is one a session proved, in its statement's member's range. -/
theorem want_at (h : UnitsOver lo hi vms held U want) {x : ℕ} (hx : x ∈ want) :
    ∃ (j : ℕ) (a : α) (us : Array ℕ), held[j]? = some a ∧ U[j]? = some us ∧ x ∈ us ∧ lo a ≤ x ∧ x < hi a := by
  obtain ⟨us, hus, hxu⟩ := (h.want x).1 hx
  obtain ⟨j, hj⟩ := List.getElem?_of_mem hus
  obtain ⟨a, ha⟩ := h.each j us hj
  exact ⟨j, a, us, ha, hj, hxu, h.units j a us ha hj x hxu⟩

/-- **The statements' members are the members with a unit in `want`, in the members' order**: where `s` says of a
member of `vms` that `want` holds a unit of its range, and says so of every statement's member. -/
theorem kept (h : UnitsOver lo hi vms held U want) {s : α → Bool}
    (hs : ∀ a ∈ vms, s a = true ↔ ∃ x ∈ want, lo a ≤ x ∧ x < hi a) (hsam : ∀ a ∈ held, s a = true) :
    held = vms.filter s := by
  refine eq_of_sorted (key := lo) h.strict (h.sorted.filter s) fun a => ?_
  rw [List.mem_filter]
  constructor
  · intro ha
    exact ⟨h.among a ha, hsam a ha⟩
  · rintro ⟨ha, hsa⟩
    obtain ⟨x, hx, hlo, hhi⟩ := (hs a ha).1 hsa
    obtain ⟨j, b, us, hb, -, -, hlo', hhi'⟩ := h.want_at hx
    have hbm : b ∈ held := List.mem_of_getElem? hb
    obtain rfl : a = b := h.apart a ha b (h.among b hbm) x hlo hhi hlo' hhi'
    exact hbm

/-- **A unit of a statement's member's range is in `want` exactly when the statement's session proved it.** -/
theorem mem_units (h : UnitsOver lo hi vms held U want) {j : ℕ} {a : α} {us : Array ℕ}
    (ha : held[j]? = some a) (hu : U[j]? = some us) {x : ℕ} (hlo : lo a ≤ x) (hhi : x < hi a) :
    x ∈ want ↔ x ∈ us := by
  constructor
  · intro hx
    obtain ⟨j', b, us', hb, hu', hxu, hlo', hhi'⟩ := h.want_at hx
    have ham : a ∈ held := List.mem_of_getElem? ha
    have hbm : b ∈ held := List.mem_of_getElem? hb
    obtain rfl : a = b := h.apart a (h.among a ham) b (h.among b hbm) x hlo hhi hlo' hhi'
    obtain rfl : j = j' := idx_of_sorted h.strict ha hb
    obtain rfl : us = us' := Option.some.inj (hu.symm.trans hu')
    exact hxu
  · intro hx
    exact (h.want x).2 ⟨us, List.mem_of_getElem? hu, hx⟩

end UnitsOver

end Verity.Sampled.MemberForm
