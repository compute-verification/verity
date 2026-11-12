module

public import Proofs.Properties.Verity.Sampled.RoundsRead
public import Proofs.Properties.Verity.Sampled.RootB

/-!
# The members' sessions, one after another, in the read of the release log

The per-member form holds the members' sessions to the challenger's release log by
`Spot.Challenger.membersTreeReason`: each session's coins its own tree's, its tree's last round its closing round,
opened after the SHA-512 of its evidence, no tree two sessions', and the trees' rounds opened one tree after another in
the sessions' order (`membersOrder`). v2's read of the same lines (`readLog`) is the device's log. What the check
gives of that read:

* **The scan** (`orderScan`, `membersOrderFrom_scan`): `membersOrder` over the rounds the lines open, each its tree and
  whether its stream is the one named the session's. Before any round the scan's list holds every earlier round's tree
  (`orderScan_split`), so an earlier round's tree is at or before a round's own in the list (`orderScan_order`); and
  after a tree's second round of its session stream no round of that tree is opened, where the list names no tree twice
  (`orderScan_last`). On the lines: `membersOrder_order`, `membersOrder_last`.
* **In order** (`opens_in_order`): in the read, an opening of session `j`'s tree has only openings of sessions' trees
  at or before `j` before it.
* **A session's closing round** (`closing_read`): where a tree shows round 1 of its session stream opened, the read has
  an opening of that tree's session stream, after one opening of that stream, answering the bytes that round was opened
  after, and no opening of the tree after it.
* **Recorded before the next** (`evidence_before`): that opening is before every opening of a later session's tree.
* **The members' check** (`membersTreeReason_none`, `memberTreeReason_none`, `members_read`): where
  `membersTreeReason` accepts, the three hold of the read at the sessions' trees, each session's closing round
  answering the bytes its evidence's SHA-512 names (`answers_of_hex`), and each session's tree less its closing round
  is the session's as v2's one-statement check holds one (`sessionTreeReason`).
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger

open private strOf from Spot.ChallengerTrees

/-! ## Lists -/

theorem nodup_idx {l : List String} (hn : l.Nodup) {i j : ℕ} {x : String} (hi : l[i]? = some x)
    (hj : l[j]? = some x) : i = j :=
  (hn.getElem?_inj (List.getElem?_eq_some_iff.1 hi).1).1 (hi.trans hj.symm)

/-- A list that repeats an entry is not free of repeats. -/
theorem not_nodup_snoc {l E : List String} {x : String} (hx : x ∈ l) (hn : (l ++ [x] ++ E).Nodup) : False := by
  obtain ⟨k, hk⟩ := List.getElem?_of_mem hx
  have hklt : k < l.length := (List.getElem?_eq_some_iff.1 hk).1
  have h1 : (l ++ [x] ++ E)[k]? = some x := by
    rw [List.append_assoc, List.getElem?_append_left hklt]
    exact hk
  have h2 : (l ++ [x] ++ E)[l.length]? = some x := by
    rw [List.append_assoc, List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.singleton_append,
      List.getElem?_cons_zero]
  have := nodup_idx hn h1 h2
  omega

/-- In a list free of repeats, an entry among those up to `a` is at or before `a`. -/
theorem idx_le {l A B : List String} {a b : String} (hn : l.Nodup) (hl : l = A ++ a :: B) (hb : b ∈ A ++ [a])
    {j j' : ℕ} (hj : l[j]? = some a) (hj' : l[j']? = some b) : j' ≤ j := by
  have ha : l[A.length]? = some a := by
    rw [hl, List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero]
  obtain ⟨k, hk⟩ := List.getElem?_of_mem hb
  have hklt : k < (A ++ [a]).length := (List.getElem?_eq_some_iff.1 hk).1
  have hk' : l[k]? = some b := by
    rw [hl, show A ++ a :: B = (A ++ [a]) ++ B by rw [List.append_assoc, List.singleton_append],
      List.getElem?_append_left hklt]
    exact hk
  rw [List.length_append, List.length_singleton] at hklt
  have e1 := nodup_idx hn hj ha
  have e2 := nodup_idx hn hj' hk'
  omega

theorem mem_of_countP_pos {α : Type} {p : α → Bool} : ∀ {l : List α}, 0 < l.countP p → ∃ a ∈ l, p a = true
  | [], h => absurd h (by simp)
  | x :: l, h => by
    by_cases hx : p x = true
    · exact ⟨x, List.mem_cons_self, hx⟩
    · rw [List.countP_cons, ite_eq_right hx, Nat.add_zero] at h
      obtain ⟨a, ha, hp⟩ := mem_of_countP_pos h
      exact ⟨a, List.mem_cons_of_mem _ ha, hp⟩

theorem countP_pos_of_mem {α : Type} {p : α → Bool} :
    ∀ {l : List α} {a : α}, a ∈ l → p a = true → 0 < l.countP p
  | [], _, h, _ => absurd h List.not_mem_nil
  | x :: l, a, h, hp => by
    rw [List.countP_cons]
    rcases List.mem_cons.1 h with rfl | h
    · rw [ite_eq_left hp]
      omega
    · have := countP_pos_of_mem h hp
      omega

/-- The entry a count is reached at: where more than `k` entries hold `p`, one holds it after exactly `k` that do. -/
theorem split_at_count {α : Type} {p : α → Bool} : ∀ (L : List α) (k : ℕ), k < L.countP p →
    ∃ a q b, L = a ++ q :: b ∧ p q = true ∧ a.countP p = k
  | [], _, h => absurd h (by simp)
  | x :: L, k, h => by
    by_cases hx : p x = true
    · cases k with
      | zero => exact ⟨[], x, L, rfl, hx, rfl⟩
      | succ k =>
        rw [List.countP_cons, ite_eq_left hx] at h
        obtain ⟨a, q, b, hL, hq, ha⟩ := split_at_count (p := p) L k (by omega)
        exact ⟨x :: a, q, b, congrArg (x :: ·) hL, hq, by rw [List.countP_cons, ite_eq_left hx, ha]⟩
    · rw [List.countP_cons, ite_eq_right hx, Nat.add_zero] at h
      obtain ⟨a, q, b, hL, hq, ha⟩ := split_at_count L k h
      exact ⟨x :: a, q, b, congrArg (x :: ·) hL, hq, by rw [List.countP_cons, ite_eq_right hx, Nat.add_zero, ha]⟩

/-- Two entries holding `p`, each after as many that do, are one entry. -/
theorem same_at_count {p : String → Bool} {a b a₂ b₂ : List String} {c c₂ : String}
    (h : a ++ c :: b = a₂ ++ c₂ :: b₂) (hc : p c = true) (hc₂ : p c₂ = true) (hk : a.countP p = a₂.countP p) :
    c = c₂ := by
  rcases List.append_eq_append_iff.1 h with ⟨a', rfl, ha⟩ | ⟨a', rfl, ha⟩
  · cases a' with
    | nil => exact (List.cons.inj ha).1
    | cons x a'' =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      rw [List.countP_append, List.countP_cons, ite_eq_left hc] at hk
      omega
  · cases a' with
    | nil => exact (List.cons.inj ha).1.symm
    | cons x a'' =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      rw [List.countP_append, List.countP_cons, ite_eq_left hc₂] at hk
      omega

/-! ## The scan -/

/-- `Spot.Challenger.membersOrderFrom`'s scan over the rounds the lines open, each its tree and whether its stream
is the one named the session's. -/
def orderScan : Array String → Option String → Bool → List (String × Bool) → Option (Array String)
  | roots, _, _, [] => some roots
  | roots, last, closed, (root, session) :: ops =>
    if last = some root then
      if closed then none else orderScan roots last session ops
    else if closed ∨ last = none then orderScan (roots.push root) (some root) false ops
    else none

theorem membersOrderFrom_scan : ∀ (ls : List String) (roots : Array String) (last : Option String) (closed : Bool),
    membersOrderFrom roots last closed ls = orderScan roots last closed (ls.filterMap openOf)
  | [], _, _, _ => rfl
  | l :: ls, roots, last, closed => by
    cases h : openOf l with
    | none =>
      rw [List.filterMap_cons_none h]
      simp only [membersOrderFrom, h, membersOrderFrom_scan ls]
    | some x =>
      obtain ⟨root, session⟩ := x
      rw [List.filterMap_cons_some h]
      simp only [membersOrderFrom, orderScan, h, membersOrderFrom_scan ls]

/-- One step of the scan: a round of the last tree, not yet closed, or the first round of another. -/
theorem orderScan_cons {roots : Array String} {last : Option String} {closed : Bool} {root : String}
    {session : Bool} {ops : List (String × Bool)} {out : Array String}
    (h : orderScan roots last closed ((root, session) :: ops) = some out) :
    (last = some root ∧ closed = false ∧ orderScan roots last session ops = some out) ∨
      (last ≠ some root ∧ orderScan (roots.push root) (some root) false ops = some out) := by
  rw [orderScan] at h
  by_cases h1 : last = some root
  · rw [ite_eq_left h1] at h
    cases closed with
    | true => simp at h
    | false => exact Or.inl ⟨h1, rfl, by simpa using h⟩
  · rw [ite_eq_right h1] at h
    by_cases h2 : closed = true ∨ last = none
    · rw [ite_eq_left h2] at h
      exact Or.inr ⟨h1, h⟩
    · rw [ite_eq_right h2] at h
      cases h

/-- The scan only adds to its list. -/
theorem orderScan_prefix : ∀ (ops : List (String × Bool)) (roots : Array String) (last : Option String)
    (closed : Bool) (out : Array String), orderScan roots last closed ops = some out →
    ∃ E, out.toList = roots.toList ++ E
  | [], roots, _, _, out, h => by
    rw [orderScan] at h
    cases h
    exact ⟨[], (List.append_nil _).symm⟩
  | (root, session) :: ops, roots, last, closed, out, h => by
    rcases orderScan_cons h with ⟨-, -, h'⟩ | ⟨-, h'⟩
    · exact orderScan_prefix ops roots last session out h'
    · obtain ⟨E, hE⟩ := orderScan_prefix ops (roots.push root) (some root) false out h'
      exact ⟨root :: E, by rw [hE, Array.toList_push, List.append_assoc, List.singleton_append]⟩

/-- The scan's last tree is the last of its list. -/
def LastOf (roots : Array String) (last : Option String) : Prop :=
  ∀ r, last = some r → ∃ A, roots.toList = A ++ [r]

theorem lastOf_none (roots : Array String) : LastOf roots none := by
  intro r hr
  cases hr

theorem lastOf_push (roots : Array String) (root : String) : LastOf (roots.push root) (some root) := by
  intro r hr
  exact ⟨roots.toList, by rw [Array.toList_push, Option.some.inj hr]⟩

/-- **The scan at a split**: the scan of `pre ++ rest` is a scan of `rest` from a state whose list holds the list
it began with and the tree of every round of `pre`. -/
theorem orderScan_split : ∀ (pre rest : List (String × Bool)) (roots : Array String) (last : Option String)
    (closed : Bool) (out : Array String), orderScan roots last closed (pre ++ rest) = some out → LastOf roots last →
    ∃ roots₁ last₁ closed₁, orderScan roots₁ last₁ closed₁ rest = some out ∧ LastOf roots₁ last₁ ∧
      (∀ q ∈ pre, q.1 ∈ roots₁.toList) ∧ ∀ x ∈ roots.toList, x ∈ roots₁.toList
  | [], _, roots, last, closed, _, h, hI =>
    ⟨roots, last, closed, h, hI, fun _ hq => absurd hq List.not_mem_nil, fun _ hx => hx⟩
  | (root, session) :: pre, rest, roots, last, closed, out, h, hI => by
    have h : orderScan roots last closed ((root, session) :: (pre ++ rest)) = some out := h
    rcases orderScan_cons h with ⟨h1, -, h'⟩ | ⟨-, h'⟩
    · obtain ⟨roots₁, last₁, closed₁, hs, hI₁, hq, hx⟩ := orderScan_split pre rest roots last session out h' hI
      obtain ⟨A, hA⟩ := hI root h1
      refine ⟨roots₁, last₁, closed₁, hs, hI₁, fun q hq' => ?_, hx⟩
      rcases List.mem_cons.1 hq' with rfl | hq'
      · exact hx root (by rw [hA]; simp)
      · exact hq q hq'
    · obtain ⟨roots₁, last₁, closed₁, hs, hI₁, hq, hx⟩ :=
        orderScan_split pre rest (roots.push root) (some root) false out h' (lastOf_push roots root)
      refine ⟨roots₁, last₁, closed₁, hs, hI₁, fun q hq' => ?_,
        fun x hx' => hx x (by rw [Array.toList_push]; exact List.mem_append_left _ hx')⟩
      rcases List.mem_cons.1 hq' with rfl | hq'
      · exact hx root (by rw [Array.toList_push]; simp)
      · exact hq q hq'

/-- **An earlier round's tree is at or before a round's own**: the scan's list is the trees up to the round's, then
the rest, and every earlier round's tree is among those up to it. -/
theorem orderScan_order {ops pre post : List (String × Bool)} {p : String × Bool} {roots : Array String}
    {last : Option String} {closed : Bool} {out : Array String} (h : orderScan roots last closed ops = some out)
    (hI : LastOf roots last) (he : ops = pre ++ p :: post) :
    ∃ A B, out.toList = A ++ p.1 :: B ∧ ∀ q ∈ pre, q.1 ∈ A ++ [p.1] := by
  subst he
  obtain ⟨roots₁, last₁, closed₁, hs, hI₁, hq, -⟩ := orderScan_split pre (p :: post) roots last closed out h hI
  obtain ⟨root, session⟩ := p
  show ∃ A B, out.toList = A ++ root :: B ∧ ∀ q ∈ pre, q.1 ∈ A ++ [root]
  rcases orderScan_cons hs with ⟨h1, -, h'⟩ | ⟨-, h'⟩
  · obtain ⟨A, hA⟩ := hI₁ root h1
    obtain ⟨E, hE⟩ := orderScan_prefix post roots₁ last₁ session out h'
    refine ⟨A, E, by rw [hE, hA, List.append_assoc, List.singleton_append], fun q hq' => ?_⟩
    rw [← hA]
    exact hq q hq'
  · obtain ⟨E, hE⟩ := orderScan_prefix post (roots₁.push root) (some root) false out h'
    exact ⟨roots₁.toList, E, by rw [hE, Array.toList_push, List.append_assoc, List.singleton_append],
      fun q hq' => List.mem_append_left _ (hq q hq')⟩

/-- A tree the scan has left is opened no more, where its list names no tree twice. -/
theorem orderScan_gone : ∀ (ops : List (String × Bool)) (roots : Array String) (last : Option String)
    (closed : Bool) (out : Array String), orderScan roots last closed ops = some out → out.toList.Nodup →
    ∀ ρ, ρ ∈ roots.toList → last ≠ some ρ → ∀ p ∈ ops, p.1 ≠ ρ
  | [], _, _, _, _, _, _, _, _, _, _, hp => absurd hp List.not_mem_nil
  | (root, session) :: ops, roots, last, closed, out, h, hn, ρ, hρ, hl, p, hp => by
    rcases orderScan_cons h with ⟨h1, -, h'⟩ | ⟨-, h'⟩
    · rcases List.mem_cons.1 hp with rfl | hp
      · exact fun e => hl (h1.trans (congrArg some e))
      · exact orderScan_gone ops roots last session out h' hn ρ hρ hl p hp
    · have hne : root ≠ ρ := by
        intro e
        obtain ⟨E, hE⟩ := orderScan_prefix ops (roots.push root) (some root) false out h'
        rw [hE, Array.toList_push, e] at hn
        exact not_nodup_snoc hρ hn
      rcases List.mem_cons.1 hp with rfl | hp
      · exact hne
      · exact orderScan_gone ops (roots.push root) (some root) false out h' hn ρ
          (by rw [Array.toList_push]; exact List.mem_append_left _ hρ) (fun e => hne (Option.some.inj e)) p hp

/-- A closed tree is opened no more. -/
theorem orderScan_closed {ops : List (String × Bool)} {roots : Array String} {out : Array String} {ρ : String}
    (h : orderScan roots (some ρ) true ops = some out) (hn : out.toList.Nodup) (hρ : ρ ∈ roots.toList) :
    ∀ p ∈ ops, p.1 ≠ ρ := by
  cases ops with
  | nil => exact fun _ hp => absurd hp List.not_mem_nil
  | cons p₀ ops =>
    obtain ⟨root, session⟩ := p₀
    rcases orderScan_cons h with ⟨-, hc, -⟩ | ⟨h1, h'⟩
    · cases hc
    · have hne : root ≠ ρ := fun e => h1 (congrArg some e.symm)
      intro p hp
      rcases List.mem_cons.1 hp with rfl | hp
      · exact hne
      · exact orderScan_gone ops (roots.push root) (some root) false out h' hn ρ
          (by rw [Array.toList_push]; exact List.mem_append_left _ hρ) (fun e => hne (Option.some.inj e)) p hp

/-- **After a tree's later round of its session stream, no round of that tree**: where the scan's list names no tree
twice, a round of the session stream of a tree an earlier round opened is that tree's last. -/
theorem orderScan_last {ops pre post : List (String × Bool)} {ρ : String} {roots : Array String}
    {last : Option String} {closed : Bool} {out : Array String} (h : orderScan roots last closed ops = some out)
    (hI : LastOf roots last) (hn : out.toList.Nodup) (he : ops = pre ++ (ρ, true) :: post) {q : String × Bool}
    (hq : q ∈ pre) (hqρ : q.1 = ρ) : ∀ p ∈ post, p.1 ≠ ρ := by
  subst he
  obtain ⟨roots₁, last₁, closed₁, hs, -, hpre, -⟩ :=
    orderScan_split pre ((ρ, true) :: post) roots last closed out h hI
  have hρ : ρ ∈ roots₁.toList := by
    rw [← hqρ]
    exact hpre q hq
  rcases orderScan_cons hs with ⟨h1, -, h'⟩ | ⟨-, h'⟩
  · rw [h1] at h'
    exact orderScan_closed h' hn hρ
  · obtain ⟨E, hE⟩ := orderScan_prefix post (roots₁.push ρ) (some ρ) false out h'
    rw [hE, Array.toList_push] at hn
    exact (not_nodup_snoc hρ hn).elim

/-! ## The lines -/

theorem jsonStrIs_self (s : String) : jsonStrIs (Json.str s) s = true := by
  simp [jsonStrIs]

/-- A line opening a stream of a tree is a round `openOf` reads, of that tree. -/
theorem openOf_opensLine {l ρ name : String} (h : opensLine ρ name l = true) :
    openOf l = some (ρ, jsonStrIs (Json.str name) TREE_SESSION) := by
  unfold opensLine at h
  cases he : event l with
  | error e =>
    rw [he] at h
    simp at h
  | ok d =>
    rw [he] at h
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨ht, ⟨hT, hr⟩, hs⟩ := h
    rw [Lines.treeLine_ok he, hT, Bool.false_or] at ht
    unfold openOf
    rw [he]
    simp only [ht, Trees.strIs hr, Trees.strIs hs, ↓reduceIte]

theorem openOf_opensRound {l ρ name : String} {k : ℕ} (h : OpensRound l ρ name k) :
    openOf l = some (ρ, jsonStrIs (Json.str name) TREE_SESSION) :=
  openOf_opensLine h.opensLine

/-- **The lines' rounds in the sessions' order**: where `membersOrder` reads the lines as the trees `used`, a line
opening a round of the tree `ρ` splits `used` at `ρ`, and every earlier line's tree is among those up to it. -/
theorem membersOrder_order {lines preL postL : List String} {l : String} {used : Array String}
    (h : membersOrder lines = some used) (hl : lines = preL ++ l :: postL) {ρ : String} {b : Bool}
    (ho : openOf l = some (ρ, b)) :
    ∃ A B, used.toList = A ++ ρ :: B ∧ ∀ l' ∈ preL, ∀ ρ' b', openOf l' = some (ρ', b') → ρ' ∈ A ++ [ρ] := by
  unfold membersOrder at h
  rw [membersOrderFrom_scan] at h
  have hs : lines.filterMap openOf = preL.filterMap openOf ++ (ρ, b) :: postL.filterMap openOf := by
    rw [hl, List.filterMap_append, List.filterMap_cons_some ho]
  obtain ⟨A, B, hout, hq⟩ := orderScan_order h (lastOf_none _) hs
  exact ⟨A, B, hout, fun l' hl' ρ' b' ho' => hq (ρ', b') (List.mem_filterMap.2 ⟨l', hl', ho'⟩)⟩

/-- **A tree's later round of its session stream is its last line**: where `membersOrder` reads the lines as trees
none named twice, no line after a round of the session stream of a tree an earlier line opened opens that tree. -/
theorem membersOrder_last {lines a b : List String} {c : String} {used : Array String}
    (h : membersOrder lines = some used) (hn : used.toList.Nodup) (hl : lines = a ++ c :: b) {ρ : String}
    (hc : openOf c = some (ρ, true)) {l₀ : String} (hl₀ : l₀ ∈ a) {b₀ : Bool} (h₀ : openOf l₀ = some (ρ, b₀)) :
    ∀ y ∈ b, ∀ b', openOf y ≠ some (ρ, b') := by
  unfold membersOrder at h
  rw [membersOrderFrom_scan] at h
  have hs : lines.filterMap openOf = a.filterMap openOf ++ (ρ, true) :: b.filterMap openOf := by
    rw [hl, List.filterMap_append, List.filterMap_cons_some hc]
  intro y hy b' hy'
  exact orderScan_last h (lastOf_none _) hn hs (List.mem_filterMap.2 ⟨l₀, hl₀, h₀⟩) rfl (ρ, b')
    (List.mem_filterMap.2 ⟨y, hy, hy'⟩) rfl

/-! ## The read -/

theorem opening_of_opensAt {fr : Verity.Frame} {i s : ℕ} (h : fr.opensAt i s = true) :
    ∃ n, fr = .opening i s n := by
  cases fr with
  | opening i' s' n =>
    have h' : (i' == i && s' == s) = true := h
    simp only [Bool.and_eq_true, beq_iff_eq] at h'
    obtain ⟨rfl, rfl⟩ := h'
    exact ⟨n, rfl⟩
  | fresh lab => exact absurd h (by simp [Verity.Frame.opensAt])
  | hello sp => exact absurd h (by simp [Verity.Frame.opensAt])

theorem keyAt_idx {ts : List (String × Json)} {i si : ℕ} {ρ name : String} (h : KeyAt ts i si ρ name) :
    ts.findIdx? (·.1 == ρ) = some i := by
  obtain ⟨hf, -⟩ := h
  exact hf

theorem keyAt_root {ts : List (String × Json)} {i si : ℕ} {ρ name : String} (h : KeyAt ts i si ρ name) :
    ∃ sp, ts[i]? = some (ρ, sp) := by
  obtain ⟨-, sp, hg, -⟩ := h
  exact ⟨sp, hg⟩

/-- The tree first of a root has that root. -/
theorem findIdx_fst {ts : List (String × Json)} {i : ℕ} {ρ : String} {y : String × Json}
    (hf : ts.findIdx? (·.1 == ρ) = some i) (hg : ts[i]? = some y) : y.1 = ρ := by
  obtain ⟨hlt, hp, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hf
  obtain ⟨_, hx⟩ := List.getElem?_eq_some_iff.1 hg
  rw [hx] at hp
  exact beq_iff_eq.1 hp

/-- Sessions of one tree are one session, where no tree is two sessions'. -/
theorem tree_inj {used : Array String} (hn : used.toList.Nodup) {ts : List (String × Json)} {tree : ℕ → ℕ}
    (htree : ∀ j ρ, used[j]? = some ρ → ts.findIdx? (·.1 == ρ) = some (tree j)) {j j' : ℕ} (hj : j < used.size)
    (hj' : j' < used.size) (h : tree j = tree j') : j = j' := by
  have h1 := htree j _ (Array.getElem?_eq_getElem hj)
  have h2 := htree j' _ (Array.getElem?_eq_getElem hj')
  rw [h] at h1
  obtain ⟨hlt, -, -⟩ := List.findIdx?_eq_some_iff_getElem.1 h2
  have hg := List.getElem?_eq_getElem hlt
  have e : used[j] = used[j'] := (findIdx_fst h1 hg).symm.trans (findIdx_fst h2 hg)
  exact nodup_idx hn (x := used[j]) (by rw [Array.getElem?_toList]; exact Array.getElem?_eq_getElem hj)
    (by rw [Array.getElem?_toList, Array.getElem?_eq_getElem hj', e])

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Verity.Release} {trees : Array LoggedTree}

/-- **The sessions' trees are opened in the sessions' order**: where `coinTreesOf` accepts the release log's lines,
`readLog` reads them, `membersOrder` reads them as the trees `used`, none named twice, and `tree` numbers each
session's tree as the read does, an opening of session `j`'s tree has only openings of the trees of sessions at or
before `j` before it. -/
theorem opens_in_order (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    {used : Array String} (hO : membersOrder lines = some used) (hn : used.toList.Nodup) {tree : ℕ → ℕ}
    (htree : ∀ j ρ, used[j]? = some ρ → (heldTrees trees).findIdx? (·.1 == ρ) = some (tree j))
    {pre : List Verity.Release} {r : Verity.Release} {post : List Verity.Release} (he : log = pre ++ r :: post)
    {j s n : ℕ} (hj : j < used.size) (hr : r.frame = .opening (tree j) s n) {q : Verity.Release} (hq : q ∈ pre)
    {j' s' n' : ℕ} (hj' : j' < used.size) (hq' : q.frame = .opening (tree j') s' n') : j' ≤ j := by
  have hF := coinTreesOf_fold recs hT
  obtain ⟨preL, l, postL, ρ, name, k, hls, hro, hkey, -, hcnt⟩ := readLog_opening hR hF he hr
  obtain ⟨a, b, hpre⟩ := List.append_of_mem hq
  have he' : log = a ++ q :: (b ++ r :: post) := by
    rw [he, hpre, List.append_assoc, List.cons_append]
  obtain ⟨_, _, _, ρ', name', _, -, -, hkey', -, -⟩ := readLog_opening hR hF he' hq'
  obtain ⟨sq, hsq⟩ := keyAt_root hkey
  obtain ⟨sq', hsq'⟩ := keyAt_root hkey'
  have eρ : ρ = used[j] := findIdx_fst (htree j _ (Array.getElem?_eq_getElem hj)) hsq
  have eρ' : ρ' = used[j'] := findIdx_fst (htree j' _ (Array.getElem?_eq_getElem hj')) hsq'
  have hpos : 0 < openedOf pre (tree j') s' :=
    countP_pos_of_mem hq (by
      show q.frame.opensAt (tree j') s' = true
      rw [hq']
      simp [Verity.Frame.opensAt])
  rw [hcnt _ _ _ _ hkey'] at hpos
  obtain ⟨l', hl', ho'⟩ := mem_of_countP_pos hpos
  obtain ⟨A, B, hused, hbefore⟩ := membersOrder_order hO hls (openOf_opensRound hro.1)
  have hmem := hbefore l' hl' _ _ (openOf_opensLine ho')
  refine idx_le hn hused hmem ?_ ?_
  · rw [Array.getElem?_toList, Array.getElem?_eq_getElem hj, eρ]
  · rw [Array.getElem?_toList, Array.getElem?_eq_getElem hj', eρ']

/-- **A session's closing round, in the read**: where `coinTreesOf` accepts the release log's lines, `readLog` reads
them and `membersOrder` reads them as trees none named twice, a tree that shows round 1 of its stream named the
session's opened has in the read an opening `x` of that stream of it (`KeyAt`), named so in the specs of the `Hello`s
before it, after one opening of that stream, answering the bytes that round was opened after, with no opening of the
tree after it. -/
theorem closing_read (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    {used : Array String} (hO : membersOrder lines = some used) (hn : used.toList.Nodup) {t : LoggedTree}
    (ht : t ∈ trees) {o : TreeOpening} (ho : o ∈ t.opened) (hs : o.stream = TREE_SESSION) (hk : o.round = 1) :
    ∃ Lc x L₂ i s n, log = Lc ++ x :: L₂ ∧ x.frame = .opening i s n ∧
      KeyAt (heldTrees trees) i s t.root TREE_SESSION ∧
      ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION ∧ openedOf Lc i s = 1 ∧
      Flock.unhex o.after = .ok x.answers ∧ ∀ y ∈ L₂, ∀ s' n', y.frame ≠ .opening i s' n' := by
  have hF := coinTreesOf_fold recs hT
  obtain ⟨c, hc, hro, d, hev, haft, -⟩ := opened_after hT t ht o ho
  rw [hs, hk] at hro
  have hcl := hro.opensLine
  obtain ⟨a, b, hlc⟩ := List.append_of_mem hc
  have hcount := coinTreesOf_round hT hlc hro
  obtain ⟨htl, dc, hevc, hTE, hrc, hsc, -⟩ := hro
  obtain ⟨i, s, hkey⟩ := treeFold_keyAt lines [] _ hF c hc htl dc hevc hTE hrc hsc
  have h2 : 1 < openedOf log i s := by
    rw [readLog_openedOf hR hF hkey, hlc, List.countP_append, List.countP_cons, ite_eq_left hcl, ← hcount]
    omega
  obtain ⟨Lc, x, L₂, he, hx, hLc⟩ := split_at_count log 1 h2
  have hx' : x.frame.opensAt i s = true := hx
  have hLc' : openedOf Lc i s = 1 := hLc
  obtain ⟨n, hxf⟩ := opening_of_opensAt hx'
  obtain ⟨lx, ρx, namex, kx, hlx, hrx, hkx, hnamex, hkxe⟩ := readLog_round hT hR he hxf
  have e' : ρx = t.root ∧ namex = TREE_SESSION := by
    have e := keyAt_beq hkx hkey
    simpa using e.symm
  obtain ⟨rfl, rfl⟩ := e'
  have hk1 : kx = 1 := hkxe.trans hLc'
  subst hk1
  obtain ⟨hox, -, dx, hevx, -, aft, haftx, hunhex⟩ := hrx
  obtain ⟨a₂, b₂, hlx2⟩ := List.append_of_mem hlx
  have hcount₂ := coinTreesOf_round hT hlx2 hox
  have hsame : c = lx := same_at_count (hlc.symm.trans hlx2) hcl hox.opensLine (hcount.symm.trans hcount₂)
  subst hsame
  have hd : d = dx := Except.ok.inj (hev.symm.trans hevx)
  subst hd
  have haft' : o.after = aft := Json.str.inj (haft.symm.trans haftx)
  subst haft'
  refine ⟨Lc, x, L₂, i, s, n, he, hxf, hkey, hnamex, hLc', hunhex, fun y hy s' n' hyf => ?_⟩
  obtain ⟨L₂a, L₂b, hL₂⟩ := List.append_of_mem hy
  have he₂ : log = (Lc ++ x :: L₂a) ++ y :: L₂b := by
    rw [he, hL₂, List.append_assoc, List.cons_append]
  obtain ⟨preL, ly, postL, ρy, namey, ky, hly, hroy, hkeyy, -, hcnty⟩ := readLog_opening hR hF he₂ hyf
  obtain ⟨sq, hsq⟩ := keyAt_root hkey
  obtain ⟨sq', hsq'⟩ := keyAt_root hkeyy
  have eρ : ρy = t.root := congrArg Prod.fst (Option.some.inj (hsq'.symm.trans hsq))
  have h1 : openedOf (Lc ++ x :: L₂a) i s = openedOf Lc i s + (openedOf L₂a i s + 1) := by
    unfold openedOf
    rw [List.countP_append, List.countP_cons, ite_eq_left hx']
  have hc2 : 1 < preL.countP (opensLine t.root TREE_SESSION) := by
    rw [← hcnty i s t.root TREE_SESSION hkey, h1, hLc']
    omega
  obtain ⟨a', c', b', hpreL, hc', ha'⟩ := split_at_count preL 1 hc2
  obtain ⟨l₀, hl₀, h₀⟩ := mem_of_countP_pos (show 0 < a'.countP (opensLine t.root TREE_SESSION) by omega)
  have hlines : lines = a' ++ c' :: (b' ++ ly :: postL) := by
    rw [hly, hpreL, List.append_assoc, List.cons_append]
  have hoc' : openOf c' = some (t.root, true) := by
    rw [openOf_opensLine hc', jsonStrIs_self]
  exact membersOrder_last hO hn hlines hoc' hl₀ (openOf_opensLine h₀) ly (by simp)
    (jsonStrIs (Json.str namey) TREE_SESSION) (by rw [openOf_opensRound hroy.1, eρ])

/-- **A session's closing round is before the next session's tree**: with `closing_read`'s premises and `tree`
numbering each session's tree as the read does, the read's opening that answers the bytes session `j'`'s tree's round
1 of its session stream was opened after is before every opening of the tree of a session `j` after `j'`. -/
theorem evidence_before (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    {used : Array String} (hO : membersOrder lines = some used) (hn : used.toList.Nodup) {tree : ℕ → ℕ}
    (htree : ∀ j ρ, used[j]? = some ρ → (heldTrees trees).findIdx? (·.1 == ρ) = some (tree j)) {j' : ℕ}
    {t : LoggedTree} (ht : t ∈ trees) (hu : used[j']? = some t.root) {o : TreeOpening} (ho : o ∈ t.opened)
    (hs : o.stream = TREE_SESSION) (hk : o.round = 1) {pre : List Verity.Release} {r : Verity.Release}
    {post : List Verity.Release} (he : log = pre ++ r :: post) {j s n : ℕ} (hj : j < used.size)
    (hr : r.frame = .opening (tree j) s n) (hlt : j' < j) : ∃ q ∈ pre, Flock.unhex o.after = .ok q.answers := by
  have hj' : j' < used.size := (Array.getElem?_eq_some_iff.1 hu).1
  obtain ⟨Lc, x, L₂, i, sx, nx, hex, hxf, hkey, -, -, hun, -⟩ := closing_read hT hR hO hn ht ho hs hk
  have hi : i = tree j' := Option.some.inj ((keyAt_idx hkey).symm.trans (htree j' _ hu))
  subst hi
  refine ⟨x, ?_, hun⟩
  have hne : ∀ nx', r.frame ≠ .opening (tree j') sx nx' := by
    intro nx' hfr
    have ht' : tree j' = tree j := (Verity.Frame.opening.inj (hfr.symm.trans hr)).1
    have := tree_inj hn htree hj' hj ht'
    omega
  rcases List.append_eq_append_iff.1 (hex.symm.trans he) with ⟨a', hpre, ha⟩ | ⟨c', hLc, hc⟩
  · cases a' with
    | nil =>
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      exact absurd hxf (hne nx)
    | cons y a'' =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      rw [hpre]
      simp
  · cases c' with
    | nil =>
      obtain ⟨rfl, -⟩ := List.cons.inj hc
      exact absurd hxf (hne nx)
    | cons y c'' =>
      rw [List.cons_append] at hc
      obtain ⟨rfl, -⟩ := List.cons.inj hc
      have := opens_in_order hT hR hO hn htree hex hj' hxf (show r ∈ Lc by rw [hLc]; simp) hj hr
      omega

/-! ## What the members' check gives -/

theorem strOf_str (s : String) : strOf (Json.str s) = s := rfl

/-- The coin trees the sessions' records name, by root, in the sessions' order. -/
def usedOf (records : Array (String × Array (String × Option ℕ) × Json × String)) : Array String :=
  records.map fun x => strOf (getD (getD x.2.2.1 "coin_tree") "root")

theorem usedOf_size (records : Array (String × Array (String × Option ℕ) × Json × String)) :
    (usedOf records).size = records.size := by
  unfold usedOf
  rw [Array.size_map]

theorem usedOf_getElem? {records : Array (String × Array (String × Option ℕ) × Json × String)} {j : ℕ}
    {x : String × Array (String × Option ℕ) × Json × String} (hx : records[j]? = some x) :
    (usedOf records)[j]? = some (strOf (getD (getD x.2.2.1 "coin_tree") "root")) := by
  simp only [usedOf, Array.getElem?_map, hx, Option.map_some]

theorem closes_fields {o : TreeOpening} {evidence : String} (h : o.closes evidence = true) :
    o.stream = TREE_SESSION ∧ o.round = 1 ∧ o.n = 0 ∧ o.after = evidence := by
  unfold TreeOpening.closes at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact ⟨h.1.1.1.1, h.1.1.1.2, h.1.1.2, h.1.2⟩

/-- **What `memberTreeReason` accepts**: the record's tree is a logged tree, the last round the log shows opened of it
closes the session after the evidence, and the tree less that round is the session's as `sessionTreeReason` holds
one. -/
theorem memberTreeReason_none {tables : Array (String × Option ℕ)} {record : Json} {evidence : String}
    (h : memberTreeReason trees tables record evidence = none) :
    ∃ t ∈ trees, getD (getD record "coin_tree") "root" = Json.str t.root ∧ ∃ o, t.opened.back? = some o ∧
      o.closes evidence = true ∧ sessionTreeReason #[t.unclosed] tables record = none := by
  unfold memberTreeReason at h
  split at h
  · cases h
  · rename_i t ht
    split at h
    · rename_i o ho
      split at h
      · rename_i hc
        exact ⟨t, Array.mem_of_find?_eq_some ht, Trees.strIs (Array.find?_some ht :), o, ho, hc, h⟩
      · cases h
    · cases h

/-- **What `membersTreeReason` accepts**: every session's record as `memberTreeReason` holds it, no tree two
sessions', and the lines' rounds one tree after another in the sessions' order. -/
theorem membersTreeReason_none {records : Array (String × Array (String × Option ℕ) × Json × String)}
    (h : membersTreeReason trees lines records = none) :
    (∀ x ∈ records, memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2 = none) ∧ (usedOf records).toList.Nodup ∧
      membersOrder lines = some (usedOf records) := by
  unfold membersTreeReason at h
  dsimp only at h
  split at h
  · rename_i x hx
    have hp : (memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2).isSome = true := (Array.find?_some hx :)
    cases hm : memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2 with
    | none =>
      rw [hm] at hp
      simp at hp
    | some w =>
      rw [hm] at h
      simp at h
  · rename_i hnone
    split at h
    · cases h
    · rename_i h1
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i h3
          refine ⟨fun x hx => ?_, Lines.distinct_nodup (Decidable.of_not_not h1), Decidable.of_not_not h3⟩
          have hno := Array.find?_eq_none.1 hnone x hx
          cases hm : memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2 with
          | none => rfl
          | some w =>
            exact absurd (show (memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2).isSome = true by simp [hm]) hno

/-- The bytes a round opened after the SHA-512 of `b`, in hex, answers. -/
theorem answers_of_hex {ev : String} {a b : ByteArray} (h : Flock.unhex ev = .ok a) (he : ev = sha512Hex b) :
    a = Flock.Sha512.hash b := by
  rw [he] at h
  exact (Except.ok.inj ((FlockComplete.JsonRT.unhex_hexOf (Flock.Sha512.hash b)).symm.trans h)).symm

/-- **The members' sessions on the read**: where `coinTreesOf` accepts the release log's lines, `readLog` reads them
and `membersTreeReason` accepts the sessions `records` against them, there is a numbering `tree` of the sessions'
trees, as the read numbers trees, at which:

* each session's record names a logged tree, the `tree j`-th the read numbers; that tree less its closing round is
  the session's as `sessionTreeReason` holds one; and the read has an opening of its stream named the session's, after
  one opening of that stream, answering the bytes the session's evidence's SHA-512 names in hex, with no opening of the
  tree after it;
* an opening of session `j`'s tree has only openings of the trees of sessions at or before `j` before it;
* before an opening of session `j`'s tree, each earlier session has an opening answering the bytes its evidence's
  SHA-512 names in hex. -/
theorem members_read (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    {records : Array (String × Array (String × Option ℕ) × Json × String)}
    (hM : membersTreeReason trees lines records = none) :
    ∃ tree : ℕ → ℕ,
      (∀ j x, records[j]? = some x → ∃ t ∈ trees, getD (getD x.2.2.1 "coin_tree") "root" = Json.str t.root ∧
        (heldTrees trees).findIdx? (·.1 == t.root) = some (tree j) ∧
        sessionTreeReason #[t.unclosed] x.2.1 x.2.2.1 = none ∧
        ∃ Lc c L₂ s n, log = Lc ++ c :: L₂ ∧ c.frame = .opening (tree j) s n ∧
          ((helloSpecs Lc)[tree j]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION ∧
          openedOf Lc (tree j) s = 1 ∧ Flock.unhex x.2.2.2 = .ok c.answers ∧
          ∀ y ∈ L₂, ∀ s' n', y.frame ≠ .opening (tree j) s' n') ∧
      (∀ pre r post, log = pre ++ r :: post → ∀ j, j < records.size → ∀ s n, r.frame = .opening (tree j) s n →
        ∀ q ∈ pre, ∀ j', j' < records.size → ∀ s' n', q.frame = .opening (tree j') s' n' → j' ≤ j) ∧
      ∀ pre r post, log = pre ++ r :: post → ∀ j, j < records.size → ∀ s n, r.frame = .opening (tree j) s n →
        ∀ j' x', j' < j → records[j']? = some x' → ∃ q ∈ pre, Flock.unhex x'.2.2.2 = .ok q.answers := by
  obtain ⟨hall, hnod, hO⟩ := membersTreeReason_none hM
  have hsess : ∀ (j : ℕ) (x : String × Array (String × Option ℕ) × Json × String), records[j]? = some x → ∃ t ∈ trees,
      getD (getD x.2.2.1 "coin_tree") "root" = Json.str t.root ∧ (usedOf records)[j]? = some t.root ∧
      sessionTreeReason #[t.unclosed] x.2.1 x.2.2.1 = none ∧
      ∃ o ∈ t.opened, o.stream = TREE_SESSION ∧ o.round = 1 ∧ o.after = x.2.2.2 := by
    intro j x hx
    obtain ⟨t, ht, hroot, o, ho, hc, hS⟩ := memberTreeReason_none (hall x (Array.mem_of_getElem? hx))
    obtain ⟨hs, hk, -, ha⟩ := closes_fields hc
    exact ⟨t, ht, hroot, by rw [usedOf_getElem? hx, hroot, strOf_str], hS, o,
      Array.mem_of_getElem? (show t.opened[t.opened.size - 1]? = some o from ho), hs, hk, ha⟩
  have hidx : ∀ (j : ℕ) (ρ : String), (usedOf records)[j]? = some ρ →
      ∃ i, (heldTrees trees).findIdx? (·.1 == ρ) = some i := by
    intro j ρ hρ
    have hjlt : j < records.size := by
      have := (Array.getElem?_eq_some_iff.1 hρ).1
      rwa [usedOf_size] at this
    obtain ⟨t, ht, -, hu, -, o, ho, hs, hk, -⟩ := hsess j _ (Array.getElem?_eq_getElem hjlt)
    obtain rfl : t.root = ρ := Option.some.inj (hu.symm.trans hρ)
    obtain ⟨_, _, _, i, _, _, -, -, hkey, -, -, -, -⟩ := closing_read hT hR hO hnod ht ho hs hk
    exact ⟨i, keyAt_idx hkey⟩
  obtain ⟨tree, htree⟩ : ∃ tree : ℕ → ℕ, ∀ (j : ℕ) (ρ : String), (usedOf records)[j]? = some ρ →
      (heldTrees trees).findIdx? (·.1 == ρ) = some (tree j) := by
    refine ⟨fun j => (((usedOf records)[j]?).bind fun ρ => (heldTrees trees).findIdx? (·.1 == ρ)).getD 0,
      fun j ρ hρ => ?_⟩
    obtain ⟨i, hi⟩ := hidx j ρ hρ
    simp only [hρ, Option.bind_some, hi, Option.getD_some]
  refine ⟨tree, fun j x hx => ?_, ?_, ?_⟩
  · obtain ⟨t, ht, hroot, hu, hS, o, ho, hs, hk, ha⟩ := hsess j x hx
    obtain ⟨Lc, c, L₂, i, s, n, he, hcf, hkey, hname, hcount, hun, hlast⟩ :=
      closing_read hT hR hO hnod ht ho hs hk
    have hi : i = tree j := Option.some.inj ((keyAt_idx hkey).symm.trans (htree j _ hu))
    subst hi
    rw [ha] at hun
    exact ⟨t, ht, hroot, htree j _ hu, hS, Lc, c, L₂, s, n, he, hcf, hname, hcount, hun, hlast⟩
  · intro pre r post he j hj s n hr q hq j' hj' s' n' hq'
    exact opens_in_order hT hR hO hnod htree he (by rw [usedOf_size]; exact hj) hr hq
      (by rw [usedOf_size]; exact hj') hq'
  · intro pre r post he j hj s n hr j' x' hlt hx'
    obtain ⟨t, ht, -, hu, -, o, ho, hs, hk, ha⟩ := hsess j' x' hx'
    obtain ⟨q, hq, hun⟩ :=
      evidence_before hT hR hO hnod htree ht hu ho hs hk he (by rw [usedOf_size]; exact hj) hr hlt
    rw [ha] at hun
    exact ⟨q, hq, hun⟩

end Verity.Sampled.MemberForm
