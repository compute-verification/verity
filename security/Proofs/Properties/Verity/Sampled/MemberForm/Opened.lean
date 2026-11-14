module

public import Proofs.Properties.Verity.Sampled.MemberForm.Tree

/-!
# Each opening the read shows is a round its tree shows opened, and a record's tree opens only its rounds

`Lines.trees_log` and `Trees.opened_in` go from a tree's rounds to the release log's lines. A session of several needs
the way back: of an opening the read shows, the round its tree shows opened, with what the tree holds of it.

* **A record's tree opens only the record's rounds** (`ret_tail`, `opened_recorded`, `tree_recorded`):
  `sessionTreeReason` ends on the test that each round its tree shows opened is below the rounds the record gives
  that round's stream (`recordRounds`), so a record it accepts has it; at a member's tree, of the tree less its last
  opening.
* **Each line opening a round is its tree's opening** (`LinesOpened`, `Opened`, `lines_opened`): where `coinTreesOf`
  accepts the lines, each line that opens a round names a tree read, whose opening of that round of that stream is
  the line's own: its count, what it was opened after and its coins' SHA-512.
* **An opening the read shows** (`read_opening`): so each opening the read shows, of tree `i`'s stream `si`, is the
  opening a tree read shows of the round the read's openings of that stream before it count, of the stream named so,
  with the opening's count, answering the bytes the tree's opening was opened after, with the coins the records give
  that round.
* **Which tree** (`tree_root_inj`, `tree_of_key`, `mem_pop_or_back`, `mem_unclosed`): trees read of one root are one
  tree, so the tree an opening is of is the one first of its root; and an opening a tree shows, other than its last,
  is one the tree less its last opening shows.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Spot Spot.Challenger

open private strOf from Spot.ChallengerTrees

/-! ## A record's tree opens only the record's rounds -/

/-- **A loop that returns at its first reason, then its reason**: where none returns, the reason after the loop is
none. -/
theorem ret_tail {α γ σ : Type}
    {f : α → Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × σ))}
    {g : Option (Option γ) × σ → Id (Option γ)}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u')))
    (hg : ∀ w u, g (some (some w), u) ≠ none) {xs : Array α} {u : σ}
    (h : (forIn (m := Id) xs (none, u) f >>= g) = none) : ∃ u', g (none, u') = none := by
  have hf' : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u') ∧ True) :=
    fun x u s hs => (hf x u s hs).imp id fun ⟨u', hu⟩ => ⟨u', hu, trivial⟩
  rcases Trees.forIn_held_arr (P := fun _ => True) hf' xs u with ⟨w, hw⟩ | ⟨hn, -⟩
  · have e : forIn (m := Id) xs (none, u) f = (some (some w), Prod.snd (forIn (m := Id) xs (none, u) f)) :=
      Prod.ext hw rfl
    rw [e] at h
    exact absurd h (hg w _)
  · have e : forIn (m := Id) xs (none, u) f = (none, Prod.snd (forIn (m := Id) xs (none, u) f)) := Prod.ext hn rfl
    rw [e] at h
    exact ⟨_, h⟩

/-- **A record `sessionTreeReason` accepts, at any tables, records each round its tree shows opened**: its coin
tree's root is that of a tree of the log, and each round that tree shows opened is below the rounds the record gives
that round's stream (`recordRounds`). -/
theorem opened_recorded {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, recordTree record = t.root ∧ ∀ o ∈ t.opened, o.round < recordRounds record o.stream := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  rename_i t ht
  refine ⟨t, Array.mem_of_find?_eq_some ht, ?_, ?_⟩
  · have hp := Array.find?_some ht
    unfold Verity.Sampled.recordTree
    rw [Trees.strIs hp]
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rcases hlo : t.opening TREE_SESSION 0 with _ | o
  · simp only [hlo, Bool.false_eq_true, ↓reduceIte] at h
    all_goals (split at h <;> (try split at h) <;> cases h)
  simp only [hlo] at h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  refine (ret_tail ?_ (fun w _ hw => by cases hw) h).elim fun u' hg => ?_
  · intro s u res hres
    split at hres
    case h_2 => exact .inl ⟨_, _, hres.symm⟩
    rename_i name rounds hsr
    refine Or.imp id ?_ (Trees.step_held (P := fun _ => True) ?_ ?_ ?_ hres)
    · rintro ⟨u', hy, -⟩
      exact ⟨u', hy⟩
    · intro x u res' hres'
      split at hres'
      case h_2 => exact .inl ⟨_, _, hres'.symm⟩
      rename_i coins n hcn
      rcases ho : t.opening name x.2 with _ | o
      · simp only [ho, Bool.false_eq_true, ↓reduceIte] at hres'
        all_goals (split at hres' <;> exact .inl ⟨_, _, hres'.symm⟩)
      simp only [ho] at hres'
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      exact .inr ⟨_, hres'.symm, trivial⟩
    · exact fun _ _ => ⟨(), rfl⟩
    · exact fun _ => ⟨(), rfl⟩
  dsimp only at hg
  split at hg
  · rename_i hall
    exact fun o ho => of_decide_eq_true (Array.all_eq_true'.1 hall o ho)
  · cases hg

variable {t : LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}

/-- **A member's record records each round its tree less its last opening shows opened** (`opened_recorded`, at the
one tree the member's check holds the record to). -/
theorem tree_recorded (h : sessionTreeReason #[t.unclosed] tables record = none) :
    ∀ o ∈ t.unclosed.opened, o.round < recordRounds record o.stream := by
  obtain ⟨t', ht', -, hall⟩ := opened_recorded h
  obtain rfl := mem_one ht'
  exact hall

/-! ## Each line opening a round is its tree's opening -/

/-- **What `coinTreesOf` holds of each line that opens a round**, of the trees `trees` it read from the lines `pre`:
the line names a tree of them, and that tree's opening of the line's round of the line's stream is the line's own,
with its count, what it was opened after and its coins' SHA-512. -/
def LinesOpened (pre : List String) (trees : Array LoggedTree) : Prop :=
  ∀ l ∈ pre, treeLine l = true → ∀ d, event l = .ok d → jsonStrIs (getD d "event") TREE_EVENT = false →
    ∃ t ∈ trees, getD d "tree" = .str t.root ∧ ∃ S k n, getD d "stream" = .str S ∧
      nat? (getD d "round") = some k ∧ nat? (getD d "n") = some n ∧
      t.opening S k = some ⟨S, k, n, strOf (getD d "after"), strOf (getD d "coins")⟩

/-- `coinTreesOf`'s invariant here: each round a tree opened is below the count of its stream's rounds opened, and
`LinesOpened`. -/
def Opened (pre : List String) (trees : Array LoggedTree) : Prop :=
  (∀ t ∈ trees, ∀ o ∈ t.opened, o.round < (t.opened.filter (·.stream == o.stream)).size) ∧ LinesOpened pre trees

/-- A round opened next of its stream is the one `opening` finds. -/
theorem opening_new {ts : Array LoggedTree} {i : ℕ} {t : LoggedTree} {x : TreeOpening} (hget : ts[i]? = some t)
    (hr : ∀ o ∈ t.opened, o.round < (t.opened.filter (·.stream == o.stream)).size)
    (hk : x.round = (t.opened.filter (·.stream == x.stream)).size) :
    ∃ t' ∈ ts.modify i (fun t => { t with opened := t.opened.push x }), t'.root = t.root ∧
      t'.opening x.stream x.round = some x := by
  refine ⟨_, Lines.mem_modify_self hget, rfl, ?_⟩
  show (t.opened.push x).find? (fun o => o.stream == x.stream && o.round == x.round) = some x
  have hfind : t.opened.find? (fun o => o.stream == x.stream && o.round == x.round) = none := by
    rw [Array.find?_eq_none]
    intro o ho e
    simp only [Bool.and_eq_true, beq_iff_eq] at e
    have hlt := hr o ho
    rw [e.1, ← hk, e.2] at hlt
    exact Nat.lt_irrefl _ hlt
  rw [Array.find?_push, hfind]
  simp

/-- A tree's round opened stays opened, of a tree of the same root, when another round of a tree is. -/
theorem opening_kept {ts : Array LoggedTree} {i : ℕ} {x : TreeOpening} {t : LoggedTree} {S : String} {k : ℕ}
    {o : TreeOpening} (ht : t ∈ ts) (ho : t.opening S k = some o) :
    ∃ t' ∈ ts.modify i (fun t => { t with opened := t.opened.push x }), t'.root = t.root ∧
      t'.opening S k = some o := by
  rcases Lines.mem_modify_of_mem (i := i) (g := fun (t : LoggedTree) => { t with opened := t.opened.push x }) ht
    with hm | hm
  · exact ⟨t, hm, rfl, ho⟩
  · exact ⟨_, hm, rfl, Lines.opening_push ho x⟩

/-- **Each line `coinTreesOf` reads keeps `Opened`.** -/
theorem lines_opened {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    Opened log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Trees.forIn_inv_run (I := Opened) ?_
    (by exact ⟨fun _ h => absurd h (by simp), fun _ h => absurd h (by simp)⟩) h
  intro pre line ts r hI hr
  have mem : ∀ {l : String}, l ∈ pre ++ [line] → l ∈ pre ∨ l = line := fun h => by simpa using h
  split at hr
  case isFalse htl =>
    cases hr
    refine ⟨ts, rfl, hI.1, fun l hl htl' d hev hT => ?_⟩
    rcases mem hl with hl | rfl
    · exact hI.2 l hl htl' d hev hT
    · exact absurd htl' htl
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, Trees.ok_bind] at hr
    split at hr
    · rename_i htree
      split at hr
      case isFalse => cases hr
      split at hr
      case isTrue => cases hr
      cases hr
      refine ⟨_, rfl, fun t ht o ho => ?_, fun l hl htl' j hev' hT => ?_⟩
      · rcases Array.mem_push.1 ht with ht | rfl
        · exact hI.1 t ht o ho
        · simp at ho
      · rcases mem hl with hl | rfl
        · obtain ⟨t, ht, h⟩ := hI.2 l hl htl' j hev' hT
          exact ⟨t, Array.mem_push_of_mem _ ht, h⟩
        · obtain rfl : j = d := Except.ok.inj (hev'.symm.trans hev)
          rw [htree] at hT
          cases hT
    · rename_i hopen
      split at hr
      case isFalse => cases hr
      split at hr
      case h_2 => cases hr
      rename_i root htree
      split at hr
      case h_2 => cases hr
      rename_i i hidx
      split at hr
      case h_2 => cases hr
      rename_i t hget
      split at hr
      case h_2 => cases hr
      rename_i s hstream
      split at hr
      case h_2 => cases hr
      rename_i k n hkn
      obtain ⟨hi, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
      have hti : ts[i] = t := by
        rw [Array.getElem?_eq_getElem hi] at hget
        exact Option.some.inj hget
      rw [hti] at hp
      have hroot : t.root = root := beq_iff_eq.1 hp
      have htmem : t ∈ ts := hti ▸ Array.getElem_mem hi
      simp only [Prod.mk.injEq] at hkn
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          have hcount : k = (t.opened.filter (·.stream == s)).size := beq_iff_eq.1 hcond.1.1.1.1.2
          cases hr
          refine ⟨_, rfl, fun t' ht' o ho => ?_, fun l hl htl' j hev' hT => ?_⟩
          · rcases Trees.mem_modify ht' with ht' | ⟨a, ha, rfl⟩
            · exact hI.1 t' ht' o ho
            · have hat : a = t := Option.some.inj (ha.symm.trans hget)
              rw [hat] at ho ⊢
              exact Lines.rounds_push (hI.1 t htmem) (by exact hcount) o ho
          · rcases mem hl with hl | rfl
            · obtain ⟨t', ht', hr', S, k', n', hS, hk', hn', ho⟩ := hI.2 l hl htl' j hev' hT
              obtain ⟨t'', ht'', hroot'', ho''⟩ := opening_kept (i := i)
                (x := ⟨s, k, n, strOf (getD d "after"), strOf (getD d "coins")⟩) ht' ho
              exact ⟨t'', ht'', by rw [hroot'']; exact hr', S, k', n', hS, hk', hn', ho''⟩
            · obtain rfl : j = d := Except.ok.inj (hev'.symm.trans hev)
              obtain ⟨t'', ht'', hroot'', ho''⟩ := opening_new (i := i)
                (x := ⟨s, k, n, strOf (getD j "after"), strOf (getD j "coins")⟩) hget (hI.1 t htmem)
                (by exact hcount)
              exact ⟨t'', ht'', by rw [hroot'', hroot]; exact htree, s, k, n, hstream, hkn.1, hkn.2, ho''⟩
        · cases hr

/-! ## An opening the read shows -/

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Verity.Release} {trees : Array LoggedTree}

/-- **An opening the read shows is the round its tree shows opened**: where `coinTreesOf` accepts the release log's
lines and `readLog` reads them, an opening `x` the read shows of tree `i`'s stream `si`, of `n` coins, is of a tree
read `t` and a stream `name` of it (`KeyAt`), named so in the specs of the `Hello`s before `x`; `t` shows opened the
round of `name` that the read's openings of that stream before `x` count, of `n` coins, after bytes that in hex are
what `x` answers; and `x`'s coins are the ones the records give that round (`Lines.openCoins`). -/
theorem read_opening (hT : coinTreesOf lines = .ok trees) (hR : readLog label after recs ξ lines = some log)
    {pre : List Verity.Release} {x : Verity.Release} {post : List Verity.Release} (he : log = pre ++ x :: post)
    {i si n : ℕ} (hx : x.frame = .opening i si n) :
    ∃ t ∈ trees, ∃ name o, KeyAt (heldTrees trees) i si t.root name ∧
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[si]?) = some name ∧
      t.opening name (openedOf pre i si) = some o ∧ o.n = n ∧ Flock.unhex o.after = .ok x.answers ∧
      x.coins = Lines.openCoins recs t.root name (openedOf pre i si) := by
  obtain ⟨l, ρ, name, k, hl, ⟨hor, hco, d, hev, hn, aft, haft, hans⟩, hkey, hname, hk⟩ := readLog_round hT hR he hx
  obtain ⟨htl, d', hev', hTE, hroot, hs, hround⟩ := hor
  obtain rfl : d' = d := Except.ok.inj (hev'.symm.trans hev)
  obtain ⟨t, ht, htree, S, k', n', hS, hk', hn', ho⟩ := (lines_opened hT).2 l hl htl d' hev' hTE
  subst hk
  have eρ : ρ = t.root := Json.str.inj (hroot.symm.trans htree)
  have eS : name = S := Json.str.inj (hs.symm.trans hS)
  have ek : openedOf pre i si = k' := Option.some.inj (hround.symm.trans hk')
  have en : n = n' := Option.some.inj ((hn i si n hx).symm.trans hn')
  subst eρ eS en ek
  refine ⟨t, ht, name, _, hkey, hname, ho, rfl, ?_, hco⟩
  show Flock.unhex (strOf (getD d' "after")) = .ok x.answers
  rw [haft]
  exact hans

/-! ## Which tree -/

/-- Trees read of one root are one tree. -/
theorem tree_root_inj (hT : coinTreesOf lines = .ok trees) {t t' : LoggedTree} (ht : t ∈ trees) (ht' : t' ∈ trees)
    (h : t'.root = t.root) : t' = t :=
  List.inj_on_of_nodup_map (Lines.trees_log hT).nodup (Array.mem_toList_iff.2 ht') (Array.mem_toList_iff.2 ht) h

/-- **The tree a stream of the trees read is of is the one first of its root**: a stream of tree `i` of the trees
read is of the tree `t'` (`KeyAt`), and `t` is the tree read first of its root at `i`, so `t'` is `t`. -/
theorem tree_of_key (hT : coinTreesOf lines = .ok trees) {t t' : LoggedTree} (ht : t ∈ trees) (ht' : t' ∈ trees)
    {i si : ℕ} {name : String} (hi : (heldTrees trees).findIdx? (·.1 == t.root) = some i)
    (hk : KeyAt (heldTrees trees) i si t'.root name) : t' = t := by
  obtain ⟨sp, hg⟩ := keyAt_root hk
  exact tree_root_inj hT ht ht' (findIdx_fst hi hg)

/-- An entry of an array is one of the array less its last, or its last. -/
theorem mem_pop_or_back {β : Type} {a : Array β} {o : β} (h : o ∈ a) : o ∈ a.pop ∨ a.back? = some o := by
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 h
  by_cases hl : i < a.size - 1
  · exact .inl (Array.mem_iff_getElem.2 ⟨i, by simpa using hl, by simp⟩)
  · refine .inr ?_
    have : i = a.size - 1 := by omega
    subst this
    simp [Array.back?]

/-- **An opening a tree shows, other than its last, is one the tree less its last opening shows.** -/
theorem mem_unclosed {o oc : TreeOpening} (ho : o ∈ t.opened) (hb : t.opened.back? = some oc) (hne : o ≠ oc) :
    o ∈ t.unclosed.opened := by
  rcases mem_pop_or_back ho with h | h
  · exact h
  · exact absurd (Option.some.inj (h.symm.trans hb)) hne

end Verity.Sampled.MemberForm
