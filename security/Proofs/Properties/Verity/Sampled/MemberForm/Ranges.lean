module

public import Mathlib.Data.List.Forall2
public import Mathlib.Data.Finset.Basic
public import Mathlib.Algebra.BigOperators.Group.List.Basic

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
  statement's session proved it. Members related to members pairwise apart are apart (`forall₂_apart`).
* **A range's proof blocks in a sample** (`emb_mem`, `range_sampled`): where a range's proof block `u` sits at
  unit `lo + u` and the sample holds the proof blocks at the wanted units, the range has a proof block in the
  sample exactly when a wanted unit is in the range.
* **Ranges that lie one after another** (`sum_le_of_apart`, `start_of_tiles`, `start_at`): ranges that share no
  unit, in the order of their starts, the nonempty ones inside `[0, n)`, with lengths summing to `n`, lie one after
  another from `0`: a nonempty one starts at the sum of the lengths before it. With them, what the sums read
  (`foldl_add_eq_sum`, `forall₂_map_eq`, `sum_perm`), and members at such ranges sit at their bases
  (`emb_of_tiles`).
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

/-- A member of the first of two lists related member by member is related to a member of the second. -/
theorem forall₂_left {α β : Type} {R : α → β → Prop} {l : List α} {l' : List β} (h : List.Forall₂ R l l') (a : α)
    (ha : a ∈ l) : ∃ b ∈ l', R a b := by
  obtain ⟨i, hi⟩ := List.getElem?_of_mem ha
  obtain ⟨b, hb, hr⟩ := forall₂_at_left h i a hi
  exact ⟨b, List.mem_of_getElem? hb, hr⟩

/-- **Members related to members apart are one member**: of two lists related member by member, the second
pairwise apart under a symmetric relation, two members of the first, each related only to members not apart from
what the other is related to, are equal. -/
theorem forall₂_apart {α β : Type} {R : α → β → Prop} {P : β → β → Prop} (hsym : ∀ b b', P b b' → P b' b) :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → l'.Pairwise P →
      ∀ a ∈ l, ∀ a' ∈ l, (∀ b b', R a b → R a' b' → ¬ P b b') → a = a'
  | _, _, .nil, _, _, ha, _, _, _ => nomatch ha
  | _, _, .cons (a := a₀) (b := b₀) (l₁ := l) (l₂ := l') hr t, hp, a, ha, a', ha', hn => by
    obtain ⟨hhead, htail⟩ := List.pairwise_cons.1 hp
    rcases List.mem_cons.1 ha with e | ht
    · rcases List.mem_cons.1 ha' with e' | ht'
      · exact e.trans e'.symm
      · obtain ⟨b', hb', hr'⟩ := forall₂_left t a' ht'
        exact absurd (hhead b' hb') (hn b₀ b' (e ▸ hr) hr')
    · rcases List.mem_cons.1 ha' with e' | ht'
      · obtain ⟨b, hb, hr'⟩ := forall₂_left t a ht
        exact absurd (hsym _ _ (hhead b hb)) (hn b b₀ hr' (e' ▸ hr))
      · exact forall₂_apart hsym t htail a ht a' ht' hn

/-- A member of the second of two lists of one length is zipped with a member of the first, at its place. -/
theorem zip_at {α β : Type} {l : List α} {l' : List β} (h : l.length = l'.length) {j : ℕ} {b : β}
    (hb : l'[j]? = some b) : ∃ a, (l.zip l')[j]? = some (a, b) := by
  obtain ⟨hj, -⟩ := List.getElem?_eq_some_iff.1 hb
  have hj' : j < l.length := h ▸ hj
  exact ⟨l[j], List.getElem?_zip_eq_some.2 ⟨List.getElem?_eq_getElem hj', hb⟩⟩

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

/-! ## A range's proof blocks in a sample -/

/-- **A proof block of a range is in the sample of the units `want` exactly when its unit is wanted**: where the
range's proof block `u` sits at unit `lo + u` of the population and the sample holds the proof blocks at the
units `want`. -/
theorem emb_mem {nJ n lo : ℕ} {emb : Fin n → Fin nJ} (place : ∀ u, (emb u).val = lo + u.val)
    {S : Finset (Fin nJ)} {want : Array ℕ} (hS : ∀ v, v ∈ S ↔ v.val ∈ want) (u : Fin n) :
    emb u ∈ S ↔ lo + u.val ∈ want := by
  rw [hS, place]

/-- **A range has a proof block in the sample exactly when a wanted unit is in the range.** -/
theorem range_sampled {nJ n lo : ℕ} {emb : Fin n → Fin nJ} (place : ∀ u, (emb u).val = lo + u.val)
    {S : Finset (Fin nJ)} {want : Array ℕ} (hS : ∀ v, v ∈ S ↔ v.val ∈ want) :
    (∃ u : Fin n, emb u ∈ S) ↔ ∃ x ∈ want, lo ≤ x ∧ x < lo + n := by
  constructor
  · rintro ⟨u, hu⟩
    exact ⟨lo + u.val, (emb_mem place hS u).1 hu, Nat.le_add_right _ _, Nat.add_lt_add_left u.isLt _⟩
  · rintro ⟨x, hx, hlo, hhi⟩
    have hlt : x - lo < n := by omega
    refine ⟨⟨x - lo, hlt⟩, (emb_mem place hS _).2 ?_⟩
    have e : lo + (x - lo) = x := by omega
    show lo + (x - lo) ∈ want
    rw [e]
    exact hx

/-! ## Ranges that lie one after another -/

/-- **Ranges that share no unit, the nonempty ones inside `[s, e)`, have lengths summing to at most `e - s`**, in
the order of their starts. -/
theorem sum_le_of_apart {α : Type} (lo len : α → ℕ) :
    ∀ (l : List α) (s e : ℕ), l.Pairwise (fun a b => lo a ≤ lo b) →
      l.Pairwise (fun a b => ∀ x, lo a ≤ x → x < lo a + len a → lo b ≤ x → x < lo b + len b → False) →
      (∀ a ∈ l, 0 < len a → s ≤ lo a ∧ lo a + len a ≤ e) → (l.map len).sum ≤ e - s
  | [], _, _, _, _, _ => by simp
  | a :: t, s, e, hs, hd, hin => by
    rw [List.map_cons, List.sum_cons]
    obtain ⟨hsa, hst⟩ := List.pairwise_cons.1 hs
    obtain ⟨hda, hdt⟩ := List.pairwise_cons.1 hd
    by_cases h0 : len a = 0
    · rw [h0, Nat.zero_add]
      exact sum_le_of_apart lo len t s e hst hdt fun b hb => hin b (List.mem_cons_of_mem _ hb)
    · have ha := hin a List.mem_cons_self (Nat.pos_of_ne_zero h0)
      have ht := sum_le_of_apart lo len t (lo a + len a) e hst hdt fun b hb hpos => by
        have hb' := hin b (List.mem_cons_of_mem _ hb) hpos
        refine ⟨?_, hb'.2⟩
        by_contra hlt
        exact hda b hb (lo b) (hsa b hb) (by omega) (Nat.le_refl _) (by omega)
      omega

/-- **Ranges that share no unit, in the order of their starts, the nonempty ones inside `[0, n)`, with lengths
summing to `n`, lie one after another from `0`**: a nonempty one starts at the sum of the lengths before it. The
ranges before it fit below its start and the rest from its start up to `n`, so neither part has room to spare. -/
theorem start_of_tiles {α : Type} (lo len : α → ℕ) {l : List α} {n : ℕ}
    (hs : l.Pairwise fun a b => lo a ≤ lo b)
    (hd : l.Pairwise fun a b => ∀ x, lo a ≤ x → x < lo a + len a → lo b ≤ x → x < lo b + len b → False)
    (hin : ∀ a ∈ l, 0 < len a → lo a + len a ≤ n) (hsum : (l.map len).sum = n) {pre : List α} {a : α}
    {post : List α} (he : l = pre ++ a :: post) (hpos : 0 < len a) : lo a = (pre.map len).sum := by
  subst he
  have hs' := List.pairwise_append.1 hs
  have hd' := List.pairwise_append.1 hd
  have h1 : (pre.map len).sum ≤ lo a - 0 :=
    sum_le_of_apart lo len pre 0 (lo a) hs'.1 hd'.1 fun b hb hb0 => by
      refine ⟨Nat.zero_le _, ?_⟩
      by_contra hlt
      have hle : lo b ≤ lo a := hs'.2.2 b hb a List.mem_cons_self
      exact hd'.2.2 b hb a List.mem_cons_self (lo a) hle (by omega) (Nat.le_refl _) (by omega)
  have h2 : ((a :: post).map len).sum ≤ n - lo a :=
    sum_le_of_apart lo len (a :: post) (lo a) n hs'.2.1 hd'.2.1 fun b hb hb0 => by
      refine ⟨?_, hin b (List.mem_append_right _ hb) hb0⟩
      rcases List.mem_cons.1 hb with e | hb'
      · rw [e]
      · exact (List.pairwise_cons.1 hs'.2.1).1 b hb'
  rw [List.map_append, List.sum_append] at hsum
  have ha := hin a (List.mem_append_right _ List.mem_cons_self) hpos
  omega

/-- `start_of_tiles`, at a place of the list: the nonempty range at place `i` starts at the sum of the first `i`
lengths. -/
theorem start_at {α : Type} (lo len : α → ℕ) {l : List α} {n : ℕ}
    (hs : l.Pairwise fun a b => lo a ≤ lo b)
    (hd : l.Pairwise fun a b => ∀ x, lo a ≤ x → x < lo a + len a → lo b ≤ x → x < lo b + len b → False)
    (hin : ∀ a ∈ l, 0 < len a → lo a + len a ≤ n) (hsum : (l.map len).sum = n) {i : ℕ} {a : α}
    (ha : l[i]? = some a) (hpos : 0 < len a) : lo a = ((l.map len).take i).sum := by
  obtain ⟨hi, rfl⟩ := List.getElem?_eq_some_iff.1 ha
  have he : l = l.take i ++ l[i] :: l.drop (i + 1) := by
    rw [List.getElem_cons_drop hi, List.take_append_drop]
  rw [← List.map_take]
  exact start_of_tiles lo len hs hd hin hsum he hpos

/-- A fold that adds a count of each entry is the sum of the entries' counts. -/
theorem foldl_add_eq_sum {α : Type} (f : α → ℕ) (a : Array α) :
    a.foldl (fun acc m => acc + f m) 0 = (a.toList.map f).sum := by
  rw [← Array.foldl_toList, List.sum_eq_foldl, List.foldl_map]

/-- Lists related member by member map alike under functions the relation equates. -/
theorem forall₂_map_eq {α β γ : Type} {R : α → β → Prop} {f : α → γ} {g : β → γ} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → (∀ a b, R a b → f a = g b) → l.map f = l'.map g
  | _, _, .nil, _ => rfl
  | _, _, .cons h t, hfg => by
    rw [List.map_cons, List.map_cons, hfg _ _ h, forall₂_map_eq t hfg]

/-- Lists with the same members, in any order, have one sum of counts. -/
theorem sum_perm {α : Type} (f : α → ℕ) {l l' : List α} (h : l.Perm l') : (l.map f).sum = (l'.map f).sum :=
  (h.map f).sum_eq

/-- **Members at ranges that share no unit, in the order of their starts, with counts summing to the population,
sit one after another**: where each member `m` of `ms` has as many proof blocks as its range `[first, first + count)`
holds units and its proof block `u` at the population's place `first + u`, that place is `base + u`, the base the
sum of the counts of the members before it. -/
theorem emb_of_tiles {α β : Type} (first count : α → ℕ) {nJ : ℕ} (n : β → ℕ) (emb : (b : β) → Fin (n b) → Fin nJ)
    {vms : List α} {ms : List β}
    (hall : List.Forall₂ (fun vm m => n m = count vm ∧ ∀ u : Fin (n m), (emb m u).val = first vm + u.val) vms ms)
    (hsorted : vms.Pairwise fun a b => first a < first b)
    (hapart : vms.Pairwise fun a b =>
      ∀ x, first a ≤ x → x < first a + count a → first b ≤ x → x < first b + count b → False)
    (htotal : (ms.map n).sum = nJ) (j : Fin ms.length) (u : Fin (n (ms.get j))) :
    (emb (ms.get j) u).val = ((ms.map n).take j.1).sum + u.val := by
  have hcnt : vms.map count = ms.map n := forall₂_map_eq hall fun a b h => h.1.symm
  obtain ⟨vm, hvm, hR⟩ := forall₂_at hall j.1 (ms.get j) (List.getElem?_eq_getElem j.2)
  have hpos : 0 < count vm := by
    rw [← hR.1]
    exact Nat.lt_of_le_of_lt (Nat.zero_le _) u.2
  have hin : ∀ a ∈ vms, 0 < count a → first a + count a ≤ nJ := by
    intro a ha hpa
    obtain ⟨m, -, hRa⟩ := forall₂_left hall a ha
    have hv : (emb m ⟨count a - 1, by rw [hRa.1]; omega⟩).val = first a + (count a - 1) := hRa.2 _
    have hlt := (emb m ⟨count a - 1, by rw [hRa.1]; omega⟩).2
    omega
  have hstart := start_at first count (hsorted.imp fun h => Nat.le_of_lt h) hapart hin
    (by rw [hcnt]; exact htotal) hvm hpos
  rw [hR.2 u, hstart, hcnt]

end Verity.Sampled.MemberForm
