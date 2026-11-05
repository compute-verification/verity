module

public import Proofs.Properties.Verity.SampledRead
public import Proofs.Properties.Verity.Sampled.Walk
public import Proofs.Properties.Verity.Sampled.Lines

/-!
# Each opening v2's read shows opens the round its stream's openings before it count

v2's check reads the device's log from the lines of the challenger session's release log (`readLog`), and
`SampledProofs.Challenger.coinTreesOf` reads the same lines as the coin trees the session sampled, accepting a line that
opens a round of a tree's stream only where that round is the count of the stream's rounds the tree shows opened. The
per-round bridge between the two:

* **The lines that open a stream** (`opensLine`, `OpensRound`): a coin tree's line whose event isn't a tree's sample,
  naming the tree's root and the stream, and the round it names.
* **What `coinTreesOf` counts** (`OpensCounted`, `coinTreesOf_counted`): where it accepts the lines, each tree, the first
  of its root, shows as many rounds of a stream opened as the lines open that stream of that root, and each line opening
  a round opens the round the lines before it count (`coinTreesOf_round`).
* **What the read shows** (`readLog_opening`, `readLog_openedOf`): each opening the read shows, of tree `i`'s stream
  `si`, is a line that reads as it (`ReadsOpen`): the round it names, with the coins the sessions' records give that
  round (`Lines.openCoins`), its count of coins and the bytes it answers. Tree `i`'s stream `si` is the stream of the
  tree it names (`KeyAt`), and the read opens each stream as often as the lines before it do.
* **The bridge** (`readLog_round`): so the round each opening the read shows names is the count of its stream's openings
  in the log before it (`Verity.openedOf`), as `Verity.Frame.opening` numbers a stream's rounds. And each round a tree
  shows opened (`LoggedTree.opening`) is below the read's openings of that stream (`opening_lt_openedOf`), which with a
  record's last round opened bounds the record's rounds by the read's.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open SampledProofs SampledProofs.Challenger

/-! ## The lines that open a stream -/

/-- Whether release log line `l` opens a round of stream `name` of the coin tree whose root is `root`: a coin tree's
line (`treeLine`) whose event isn't a tree's sample, naming that tree and that stream, as `coinTreesOf` reads it. -/
def opensLine (root name : String) (l : String) : Bool :=
  treeLine l && match event l with
    | .ok d => !jsonStrIs (getD d "event") TREE_EVENT && jsonStrIs (getD d "tree") root &&
        jsonStrIs (getD d "stream") name
    | .error _ => false

theorem opensLine_skip {ρ name l : String} (h : ¬treeLine l = true) : opensLine ρ name l = false := by
  simp [opensLine, h]

theorem opensLine_hello {ρ name l : String} {d : Json} (he : event l = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = true) : opensLine ρ name l = false := by
  simp [opensLine, he, hT]

theorem opensLine_open {ρ name l root stream : String} {d : Json} (ht : treeLine l = true) (he : event l = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = false) (hr : getD d "tree" = .str root)
    (hs : getD d "stream" = .str stream) : opensLine ρ name l = (root == ρ && stream == name) := by
  unfold opensLine
  rw [ht, he]
  simp only [hT, hr, hs, Bool.true_and, Bool.not_false]
  rfl

/-- **Release log line `l` opens round `k` of stream `name` of the coin tree whose root is `root`**, as
`coinTreesOf` reads it. -/
def OpensRound (l root name : String) (k : ℕ) : Prop :=
  treeLine l = true ∧ ∃ d, event l = .ok d ∧ jsonStrIs (getD d "event") TREE_EVENT = false ∧
    getD d "tree" = .str root ∧ getD d "stream" = .str name ∧ nat? (getD d "round") = some k

theorem OpensRound.opensLine {l root name : String} {k : ℕ} (h : OpensRound l root name k) :
    opensLine root name l = true := by
  obtain ⟨ht, d, he, hT, hr, hs, -⟩ := h
  simp [opensLine_open ht he hT hr hs]

theorem snoc_eq_append_cons {α : Type} {pre a b : List α} {x l : α} (h : pre ++ [x] = a ++ l :: b) :
    (b = [] ∧ a = pre ∧ l = x) ∨ ∃ b', b = b' ++ [x] ∧ pre = a ++ l :: b' := by
  cases b.eq_nil_or_concat with
  | inl hb =>
    subst hb
    obtain ⟨h1, h2⟩ := List.append_inj' h rfl
    exact .inl ⟨rfl, h1.symm, (List.cons.inj h2).1.symm⟩
  | inr hb =>
    obtain ⟨b', y, rfl⟩ := hb
    rw [List.concat_eq_append] at h ⊢
    have h' : pre ++ [x] = (a ++ l :: b') ++ [y] := by rw [h]; simp
    obtain ⟨h1, h2⟩ := List.append_inj' h' rfl
    obtain rfl := (List.cons.inj h2).1
    exact .inr ⟨b', rfl, h1⟩

/-! ## What `coinTreesOf` counts -/

/-- **The trees `coinTreesOf` holds after the release log's lines `pre`, counted**: each tree, the first of its root,
shows as many rounds of a stream opened as `pre` has lines opening that stream of that root (`opensLine`); `pre` opens
no stream of a root none of the trees has; and each line of `pre` opening a round opens the round its stream's lines
before it count. -/
structure OpensCounted (pre : List String) (trees : Array LoggedTree) : Prop where
  held : ∀ i t, trees[i]? = some t → trees.findIdx? (·.root == t.root) = some i →
    ∀ s, (t.opened.filter (·.stream == s)).size = pre.countP (opensLine t.root s)
  none : ∀ ρ, trees.findIdx? (·.root == ρ) = none → ∀ s, pre.countP (opensLine ρ s) = 0
  round : ∀ a l b ρ s k, pre = a ++ l :: b → OpensRound l ρ s k → k = a.countP (opensLine ρ s)

theorem OpensCounted.nil : OpensCounted [] #[] where
  held i t h := by simp at h
  none _ _ _ := rfl
  round a l b _ _ _ h := by simp at h

theorem OpensCounted.skip {pre : List String} {ts : Array LoggedTree} {line : String} (hI : OpensCounted pre ts)
    (h : ¬treeLine line = true) : OpensCounted (pre ++ [line]) ts where
  held i t hi hf s := by
    rw [List.countP_append, hI.held i t hi hf s, List.countP_singleton, opensLine_skip h]
    rfl
  none ρ hf s := by
    rw [List.countP_append, hI.none ρ hf s, List.countP_singleton, opensLine_skip h]
    rfl
  round a l b ρ s k he ho := by
    rcases snoc_eq_append_cons he with ⟨-, -, rfl⟩ | ⟨b', -, he'⟩
    · exact absurd ho.1 h
    · exact hI.round a l b' ρ s k he' ho

theorem OpensCounted.hello {pre : List String} {ts : Array LoggedTree} {line : String} {d : Json}
    {t' : LoggedTree} (hI : OpensCounted pre ts) (he : event line = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = true) (ht' : t'.opened = #[]) :
    OpensCounted (pre ++ [line]) (ts.push t') where
  held i t hi hf s := by
    rw [List.countP_append, List.countP_singleton, opensLine_hello he hT]
    rw [Array.findIdx?_push] at hf
    rw [Array.getElem?_push] at hi
    split at hi
    · rename_i hil
      subst hil
      have e := Option.some.inj hi
      subst e
      cases hfi : ts.findIdx? (·.root == t'.root) with
      | some j =>
        rw [hfi, Option.some_or] at hf
        obtain rfl := Option.some.inj hf
        obtain ⟨hj, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hfi
        exact absurd hj (Nat.lt_irrefl _)
      | none =>
        rw [ht', hI.none _ hfi s]
        rfl
    · cases hfi : ts.findIdx? (·.root == t.root) with
      | some j =>
        rw [hfi, Option.some_or] at hf
        obtain rfl := Option.some.inj hf
        rw [hI.held j t hi hfi s]
        rfl
      | none =>
        rw [hfi, Option.none_or] at hf
        split at hf
        · obtain rfl := Option.some.inj hf
          exact absurd rfl ‹¬_›
        · cases hf
  none ρ hf s := by
    rw [Array.findIdx?_push] at hf
    cases hfi : ts.findIdx? (·.root == ρ) with
    | some j => rw [hfi, Option.some_or] at hf; cases hf
    | none =>
      rw [List.countP_append, List.countP_singleton, opensLine_hello he hT, hI.none ρ hfi s]
      rfl
  round a l b ρ s k hp ho := by
    rcases snoc_eq_append_cons hp with ⟨-, -, rfl⟩ | ⟨b', -, he'⟩
    · obtain ⟨-, d', he', hT', -⟩ := ho
      rw [he] at he'
      cases he'
      rw [hT] at hT'
      cases hT'
    · exact hI.round a l b' ρ s k he' ho

theorem findIdx?_modify_root {ts : Array LoggedTree} {i : ℕ} {f : LoggedTree → LoggedTree}
    (hf : ∀ u, (f u).root = u.root) (ρ : String) :
    (ts.modify i f).findIdx? (·.root == ρ) = ts.findIdx? (·.root == ρ) := by
  rw [← findIdx?_toList, ← findIdx?_toList, Array.toList_modify]
  have h1 := List.findIdx?_map (f := LoggedTree.root) (p := (· == ρ)) (l := ts.toList.modify i f)
  have h2 := List.findIdx?_map (f := LoggedTree.root) (p := (· == ρ)) (l := ts.toList)
  rw [map_modify_eq _ f hf] at h1
  exact h1.symm.trans h2

theorem OpensCounted.opening {pre : List String} {ts : Array LoggedTree} {line : String} {d : Json}
    {root s : String} {i k : ℕ} {t : LoggedTree} {o : TreeOpening} (hI : OpensCounted pre ts)
    (htl : treeLine line = true) (he : event line = .ok d) (hT : jsonStrIs (getD d "event") TREE_EVENT = false)
    (htree : getD d "tree" = .str root) (hidx : ts.findIdx? (·.root == root) = some i) (hget : ts[i]? = some t)
    (hstream : getD d "stream" = .str s) (hk : nat? (getD d "round") = some k)
    (hkc : k = (t.opened.filter (·.stream == s)).size) (ho : o.stream = s) :
    OpensCounted (pre ++ [line]) (ts.modify i fun u => { u with opened := u.opened.push o }) := by
  have hl : ∀ ρ name, opensLine ρ name line = (root == ρ && s == name) := fun ρ name =>
    opensLine_open htl he hT htree hstream
  obtain ⟨hi, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
  have hti : ts[i] = t := by
    rw [Array.getElem?_eq_getElem hi] at hget
    exact Option.some.inj hget
  rw [hti] at hp
  have hroot : t.root = root := beq_iff_eq.1 hp
  have hfm := findIdx?_modify_root (ts := ts) (i := i) (f := fun u => { u with opened := u.opened.push o })
    (fun _ => rfl)
  refine ⟨fun j t' hj hf s' => ?_, fun ρ hf s' => ?_, fun a l b ρ s' k' hp' ho' => ?_⟩
  · rw [hfm] at hf
    rw [List.countP_append, List.countP_singleton, hl]
    rw [Array.getElem?_modify] at hj
    split at hj
    · rename_i hij
      subst hij
      rw [hget] at hj
      obtain rfl := Option.some.inj hj
      dsimp only at hf ⊢
      rw [Array.filter_push, hroot]
      rw [hroot] at hf
      have e := hI.held i t hget (by rw [hroot]; exact hf) s'
      rw [hroot] at e
      simp only [beq_self_eq_true, Bool.true_and]
      split
      · rename_i hs
        rw [ho] at hs
        rw [Array.size_push, e, ite_eq_left hs]
      · rename_i hs
        rw [ho] at hs
        rw [e, ite_eq_right hs]
        rfl
    · rename_i hij
      rw [hI.held j t' hj hf s']
      have hne : (root == t'.root) = false := by
        cases hb : (root == t'.root)
        · rfl
        · rw [beq_iff_eq] at hb
          rw [← hb] at hf
          exact absurd (Option.some.inj (hidx.symm.trans hf)) hij
      rw [hne]
      rfl
  · rw [hfm] at hf
    rw [List.countP_append, List.countP_singleton, hl, hI.none ρ hf s']
    have hne : (root == ρ) = false := by
      cases hb : (root == ρ)
      · rfl
      · rw [beq_iff_eq] at hb
        rw [← hb, hidx] at hf
        cases hf
    rw [hne]
    rfl
  · rcases snoc_eq_append_cons hp' with ⟨-, rfl, rfl⟩ | ⟨b', -, he'⟩
    · obtain ⟨-, d', he', -, hr', hs', hk'⟩ := ho'
      rw [he] at he'
      cases he'
      rw [htree] at hr'
      cases hr'
      rw [hstream] at hs'
      cases hs'
      rw [hk] at hk'
      cases hk'
      rw [hkc, hI.held i t hget (by rw [hroot]; exact hidx) s, hroot]
    · exact hI.round a l b' ρ s' k' he' ho'

/-- **`coinTreesOf` counts each stream's rounds**: where it accepts the release log, its trees and each line opening a
round are as `OpensCounted` says. -/
theorem coinTreesOf_counted {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    OpensCounted log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Trees.forIn_inv_run (I := OpensCounted) ?_ OpensCounted.nil h
  intro pre line ts r hI hr
  split at hr
  case isFalse htl => cases hr; exact ⟨ts, rfl, hI.skip htl⟩
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, Trees.ok_bind] at hr
    split at hr
    · rename_i hT
      split at hr
      case isFalse => cases hr
      split at hr
      case isTrue => cases hr
      cases hr
      exact ⟨_, rfl, hI.hello hev hT rfl⟩
    · rename_i hT
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
      simp only [Prod.mk.injEq] at hkn
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          cases hr
          exact ⟨_, rfl, hI.opening htl hev (Bool.eq_false_iff.2 hT) htree hidx hget hstream hkn.1
            (beq_iff_eq.1 hcond.1.1.1.1.2) rfl⟩
        · cases hr

/-- **Each line opening a round opens the round its stream's lines before it count**: where `coinTreesOf` accepts
the release log `lines`, a line of it opening round `k` of stream `name` of the tree whose root is `ρ` (`OpensRound`)
has `k` lines before it opening that stream of that tree (`opensLine`). -/
theorem coinTreesOf_round {lines pre post : List String} {l : String} {trees : Array LoggedTree} {ρ name : String}
    {k : ℕ} (h : coinTreesOf lines = .ok trees) (hl : lines = pre ++ l :: post) (ho : OpensRound l ρ name k) :
    k = pre.countP (opensLine ρ name) :=
  (coinTreesOf_counted h).round pre l post ρ name k hl ho

/-! ## What the read shows -/

/-- **Tree `i`'s stream `si` among the trees read `ts` is stream `name` of the coin tree whose root is `ρ`**, each tree
read its root and its line's spec: the first tree of root `ρ` is the `i`-th, and `name` is first at place `si` of its
spec's streams (`specStreams`), as `treeStep` finds them. -/
def KeyAt (ts : List (String × Json)) (i si : ℕ) (ρ name : String) : Prop :=
  ts.findIdx? (·.1 == ρ) = some i ∧ ∃ sp, ts[i]? = some (ρ, sp) ∧
    (specStreams sp).findIdx? (jsonStrIs · name) = some si

theorem KeyAt.append {ts : List (String × Json)} {i si : ℕ} {ρ name : String} (h : KeyAt ts i si ρ name)
    (more : List (String × Json)) : KeyAt (ts ++ more) i si ρ name := by
  obtain ⟨hf, sp, hg, hs⟩ := h
  have hlt : i < ts.length := (List.findIdx?_eq_some_iff_getElem.1 hf).1
  refine ⟨?_, sp, ?_, hs⟩
  · rw [List.findIdx?_append, hf, Option.some_or]
  · rw [List.getElem?_append_left hlt, hg]

/-- Two streams of the trees read are one exactly where they are the same stream of the same tree. -/
theorem keyAt_beq {ts : List (String × Json)} {i₁ s₁ i₂ s₂ : ℕ} {ρ₁ n₁ ρ₂ n₂ : String}
    (h₁ : KeyAt ts i₁ s₁ ρ₁ n₁) (h₂ : KeyAt ts i₂ s₂ ρ₂ n₂) :
    (i₁ == i₂ && s₁ == s₂) = (ρ₁ == ρ₂ && n₁ == n₂) := by
  obtain ⟨hf₁, sp₁, hg₁, hs₁⟩ := h₁
  obtain ⟨hf₂, sp₂, hg₂, hs₂⟩ := h₂
  rw [Bool.eq_iff_iff]
  simp only [Bool.and_eq_true, beq_iff_eq]
  constructor
  · rintro ⟨rfl, rfl⟩
    rw [hg₁] at hg₂
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hg₂)
    obtain ⟨hlt₁, hp₁, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs₁
    obtain ⟨hlt₂, hp₂, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs₂
    rw [Trees.strIs hp₁] at hp₂
    exact ⟨rfl, by simpa [jsonStrIs] using hp₂⟩
  · rintro ⟨rfl, rfl⟩
    obtain rfl := Option.some.inj (hf₁.symm.trans hf₂)
    rw [hg₁] at hg₂
    obtain rfl := (Prod.mk.inj (Option.some.inj hg₂)).2
    exact ⟨rfl, Option.some.inj (hs₁.symm.trans hs₂)⟩

private theorem ok_of_toOption {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x with
  | error e => cases h
  | ok b => cases h; rfl

/-- **A coin tree's line that isn't a tree's sample reads as an opening** (`treeStep`): of the round its `round` names
of its stream `stream` of its tree `root`, tree `i`'s stream `si` among the trees read (`KeyAt`), its first `n`
coins after the bytes its `after` gives, with the coins the records give that round (`Lines.openCoins`). -/
theorem opening_of_treeStep {records : List Json} {ts ts' : List (String × Json)} {d : Json} {r : Verity.Release}
    (h : treeStep records ts d = some (r, ts')) (hT : jsonStrIs (getD d "event") TREE_EVENT = false) :
    ∃ root stream k n i si aft, getD d "tree" = .str root ∧ getD d "stream" = .str stream ∧
      nat? (getD d "round") = some k ∧ nat? (getD d "n") = some n ∧ getD d "after" = .str aft ∧
      Flock.unhex aft = .ok r.answers ∧ ts' = ts ∧ r.frame = .opening i si n ∧ KeyAt ts i si root stream ∧
      r.coins = Lines.openCoins records root stream k := by
  unfold treeStep at h
  rw [ite_eq_right (by rw [hT]; exact Bool.false_ne_true)] at h
  simp only [Option.bind_eq_bind] at h
  split at h
  · rename_i root hroot
    split at h
    · rename_i stream hstream
      split at h
      · rename_i aft haft
        obtain ⟨k, hk, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨n, hn, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨i, hi, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨⟨x, spec⟩, hts, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨s, hs, h⟩ := Option.bind_eq_some_iff.1 h
        obtain ⟨b, hb, h⟩ := Option.bind_eq_some_iff.1 h
        simp only [Option.pure_def, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨hlt, hp, -⟩ := List.findIdx?_eq_some_iff_getElem.1 hi
        rw [List.getElem?_eq_getElem hlt] at hts
        have hx : ts[i] = (x, spec) := Option.some.inj hts
        rw [hx] at hp
        have hxr : root = x := (beq_iff_eq.1 hp).symm
        subst hxr
        exact ⟨root, stream, k, n, i, s, aft, hroot, hstream, hk, hn, haft, ok_of_toOption hb, rfl, rfl,
          ⟨hi, spec, by rw [List.getElem?_eq_getElem hlt, hx], hs⟩, rfl⟩
      · simp at h
    · simp at h
  · simp at h

/-- The trees `treeFold` reads after `ts` extend `ts`. -/
theorem treeFold_prefix {records : List Json} :
    ∀ (ls : List String) (ts tsF : List (String × Json)), treeFold records ts ls = some tsF →
      ∃ more, ts ++ more = tsF
  | [], ts, tsF, h => by
    simp only [treeFold, Option.some.injEq] at h
    exact ⟨[], by rw [h, List.append_nil]⟩
  | l :: ls, ts, tsF, h => by
    simp only [treeFold] at h
    split at h
    · obtain ⟨⟨r, ts'⟩, hX, h⟩ := Option.bind_eq_some_iff.1 h
      obtain ⟨j, -, hX⟩ := Option.bind_eq_some_iff.1 hX
      obtain ⟨more, hm⟩ := treeFold_prefix ls ts' tsF h
      rcases treeStep_some hX with ⟨root, sj, -, -, rfl, -⟩ | ⟨-, -, -, -, rfl, -⟩
      · exact ⟨(root, sj) :: more, by rw [← hm, List.append_assoc, List.singleton_append]⟩
      · exact ⟨more, hm⟩
    · exact treeFold_prefix ls ts tsF h

theorem treeFold_tree {records : List Json} {ts ts' : List (String × Json)} {l : String} {ls : List String}
    {j : Json} {r : Verity.Release} (ht : treeLine l = true) (he : event l = .ok j)
    (hX : treeStep records ts j = some (r, ts')) : treeFold records ts (l :: ls) = treeFold records ts' ls := by
  simp only [treeFold, ite_eq_left ht, he, Except.toOption, Option.bind_some, hX]

theorem treeFold_skip {records : List Json} {ts : List (String × Json)} {l : String} {ls : List String}
    (ht : ¬treeLine l = true) : treeFold records ts (l :: ls) = treeFold records ts ls := by
  simp only [treeFold, ite_eq_right ht]

/-- **A release a coin tree's line reads opens a stream exactly where its line opens it** (`opensLine`): tree `i`'s
stream `si` of the trees read by the end, stream `name` of the tree whose root is `ρ` (`KeyAt`), where the trees the
line is read after are the first of them. -/
theorem treeStep_opensAt {records : List Json} {ts ts' more : List (String × Json)} {d : Json} {r : Verity.Release}
    {l : String} (htl : treeLine l = true) (he : event l = .ok d) (h : treeStep records ts d = some (r, ts'))
    {i si : ℕ} {ρ name : String} (hk : KeyAt (ts ++ more) i si ρ name) :
    r.frame.opensAt i si = opensLine ρ name l := by
  by_cases hT : jsonStrIs (getD d "event") TREE_EVENT = true
  · rcases treeStep_some h with ⟨-, -, spec, hr, -⟩ | ⟨-, -, -, -, -, hT', -⟩
    · rw [hr, opensLine_hello he hT]
      rfl
    · rw [hT] at hT'
      cases hT'
  · have hT := Bool.eq_false_iff.2 hT
    obtain ⟨root, stream, k, n, i', si', aft, hroot, hstream, -, -, -, -, -, hr, hk', -⟩ := opening_of_treeStep h hT
    rw [hr, opensLine_open htl he hT hroot hstream]
    exact keyAt_beq (hk'.append more) hk

/-- **Line `l` of the release log reads as release `r`, an opening of round `k` of stream `name` of the coin tree whose
root is `ρ`** (`OpensRound`): with the coins the records give that round (`Lines.openCoins`), as many coins as the
line's `n`, after the bytes the line's `after` gives. -/
def ReadsOpen (records : List Json) (l : String) (r : Verity.Release) (ρ name : String) (k : ℕ) : Prop :=
  OpensRound l ρ name k ∧ r.coins = Lines.openCoins records ρ name k ∧ ∃ d, event l = .ok d ∧
    (∀ i si n, r.frame = .opening i si n → nat? (getD d "n") = some n) ∧
    ∃ aft, getD d "after" = .str aft ∧ Flock.unhex aft = .ok r.answers

theorem event_of_toOption {l : String} {j : Json} (h : (event l).toOption = some j) : event l = .ok j :=
  ok_of_toOption h

theorem helloSpecs_fresh (P : List Verity.Release) {q : Verity.Release} {lab : String} (hq : q.frame = .fresh lab) :
    helloSpecs (P ++ [q]) = helloSpecs P := by
  rw [helloSpecs_append]
  simp [helloSpecs, hq]

/-- **An opening `readTail` reads is a line opening a round, counted**: where `readTail` reads the lines `lines` after
the trees `ts` and the releases `P`, and the trees read by the end are `tsF`, each opening it reads, of tree `i`'s
stream `si`, is a line of `lines` reading as it (`ReadsOpen`) of the stream `KeyAt` names there, named so in the specs
of the `Hello`s before it; and each stream is opened in the releases before it as often as the lines before that line
open it (`opensLine`). -/
theorem readTail_opening {label : String} {after : ByteArray} {records : List Json} {tsF : List (String × Json)} :
    ∀ (lines : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      (P : List Verity.Release) {log : List Verity.Release},
      readTail label after records lines ts pending ξ = some log → treeFold records ts lines = some tsF →
      List.Forall₂ (fun t sp => specOf t.2 = some sp) ts (helloSpecs P) →
      ∀ pre r post, log = pre ++ r :: post → ∀ i si n, r.frame = .opening i si n →
        ∃ preL l postL ρ name k, lines = preL ++ l :: postL ∧ ReadsOpen records l r ρ name k ∧
          KeyAt tsF i si ρ name ∧ ((helloSpecs (P ++ pre))[i]?.bind fun sp => sp.streams[si]?) = some name ∧
          ∀ i' si' ρ' name', KeyAt tsF i' si' ρ' name' → openedOf pre i' si' = preL.countP (opensLine ρ' name')
  | [], _, none, _, _, log, hR, _, _, pre, r, post, he, _, _, _, _ => by
    simp only [readTail, Option.some.injEq] at hR
    subst hR
    simp at he
  | [], _, some _, _, _, _, hR, _, _, _, _, _, _, _, _, _, _ => by simp [readTail] at hR
  | l :: ls, ts, pending, ξ, P, log, hR, hF, hS, pre, r, post, he, i, si, n, hr => by
    unfold readTail at hR
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r₀, ts'⟩, hX, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      have hev : event l = .ok j := event_of_toOption hj
      rw [treeFold_tree ht hev hX] at hF
      obtain ⟨more, hm⟩ := treeFold_prefix ls ts' tsF hF
      cases pre with
      | nil =>
        simp only [List.nil_append, List.cons.injEq] at he
        obtain ⟨rfl, rfl⟩ := he
        have hT : jsonStrIs (getD j "event") TREE_EVENT = false := by
          rcases treeStep_some hX with ⟨_, _, _, hr', _⟩ | ⟨_, _, _, _, _, hT, _⟩
          · rw [hr] at hr'
            cases hr'
          · exact hT
        obtain ⟨root, stream, k, n', i', si', aft, hroot, hstream, hk, hn, haft, hb, rfl, hr', hkey, hc⟩ :=
          opening_of_treeStep hX hT
        rw [hr] at hr'
        obtain ⟨rfl, rfl, rfl⟩ := Verity.Frame.opening.inj hr'
        refine ⟨[], l, ls, root, stream, k, rfl,
          ⟨⟨ht, j, hev, hT, hroot, hstream, hk⟩, hc, j, hev, fun _ _ _ h => ?_, aft, haft, hb⟩, ?_, ?_,
          fun _ _ _ _ _ => rfl⟩
        · rw [hr] at h
          obtain ⟨-, -, rfl⟩ := Verity.Frame.opening.inj h
          exact hn
        · rw [← hm]
          exact hkey.append more
        · rw [List.append_nil]
          obtain ⟨-, sp, hg, hs⟩ := hkey
          obtain ⟨spec, hspec, hsp⟩ := forall₂_getElem? hS hg
          rw [hspec, Option.bind_some]
          obtain ⟨hlt, heq, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs
          exact specOf_streams hsp (by rw [Array.getElem?_eq_getElem hlt, Trees.strIs heq])
      | cons q pre' =>
        simp only [List.cons_append, List.cons.injEq] at he
        obtain ⟨rfl, he⟩ := he
        have hS' : List.Forall₂ (fun t sp => specOf t.2 = some sp) ts' (helloSpecs (P ++ [r₀])) := by
          rcases treeStep_some hX with ⟨root, sj, spec, hr', hts, hsp, -⟩ | ⟨_, _, _, hr', hts, -⟩
          · have h1 : helloSpecs [r₀] = [spec] := by simp [helloSpecs, hr']
            rw [helloSpecs_append, h1, hts]
            exact forall₂_snoc hS hsp
          · have h1 : helloSpecs [r₀] = [] := by simp [helloSpecs, hr']
            rw [helloSpecs_append, h1, List.append_nil, hts]
            exact hS
        obtain ⟨preL, l', postL, ρ, name, k, hls, hro, hkey, hname, hcnt⟩ :=
          readTail_opening ls ts' pending ξ (P ++ [r₀]) hrest hF hS' pre' r post he i si n hr
        refine ⟨l :: preL, l', postL, ρ, name, k, by rw [hls]; rfl, hro, hkey, by simpa using hname,
          fun i' si' ρ' name' hk' => ?_⟩
        have hts : ∃ more', ts ++ more' = tsF := by
          rcases treeStep_some hX with ⟨root, sj, -, -, hts, -⟩ | ⟨-, -, -, -, hts, -⟩
          · exact ⟨(root, sj) :: more, by rw [← hm, hts, List.append_assoc, List.singleton_append]⟩
          · exact ⟨more, by rw [← hm, hts]⟩
        obtain ⟨more', hm'⟩ := hts
        rw [← hm'] at hk'
        have hq := treeStep_opensAt ht hev hX hk'
        rw [hm'] at hk'
        unfold openedOf at hcnt ⊢
        rw [List.countP_cons, List.countP_cons, hcnt i' si' ρ' name' hk', hq]
    · rw [ite_eq_right ht] at hR
      rw [treeFold_skip ht] at hF
      cases pending with
      | none =>
        obtain ⟨preL, l', postL, ρ, name, k, hls, hro, hkey, hname, hcnt⟩ :=
          readTail_opening ls ts (some l) ξ P hR hF hS pre r post he i si n hr
        refine ⟨l :: preL, l', postL, ρ, name, k, by rw [hls]; rfl, hro, hkey, hname,
          fun i' si' ρ' name' hk' => ?_⟩
        rw [List.countP_cons, opensLine_skip ht, ← hcnt i' si' ρ' name' hk']
        rfl
      | some a =>
        simp only [Option.bind_eq_bind] at hR
        obtain ⟨⟨lab, root⟩, -, hR⟩ := Option.bind_eq_some_iff.1 hR
        simp only [Option.pure_def] at hR
        have step : ∀ (q : Verity.Release) (lab' : String) (ξ' : ByteArray) (rest : List Verity.Release),
            q.frame = .fresh lab' → readTail label after records ls ts none ξ' = some rest → log = q :: rest →
            ∃ preL l' postL ρ name k, l :: ls = preL ++ l' :: postL ∧ ReadsOpen records l' r ρ name k ∧
              KeyAt tsF i si ρ name ∧ ((helloSpecs (P ++ pre))[i]?.bind fun sp => sp.streams[si]?) = some name ∧
              ∀ i' si' ρ' name', KeyAt tsF i' si' ρ' name' →
                openedOf pre i' si' = preL.countP (opensLine ρ' name') := by
          intro q lab' ξ' rest hq hrest hlog
          subst hlog
          cases pre with
          | nil =>
            simp only [List.nil_append, List.cons.injEq] at he
            rw [← he.1, hq] at hr
            cases hr
          | cons q' pre' =>
            simp only [List.cons_append, List.cons.injEq] at he
            obtain ⟨rfl, he⟩ := he
            obtain ⟨preL, l', postL, ρ, name, k, hls, hro, hkey, hname, hcnt⟩ :=
              readTail_opening ls ts none ξ' (P ++ [q]) hrest hF (by rw [helloSpecs_fresh P hq]; exact hS)
                pre' r post he i si n hr
            refine ⟨l :: preL, l', postL, ρ, name, k, by rw [hls]; rfl, hro, hkey, by simpa using hname,
              fun i' si' ρ' name' hk' => ?_⟩
            unfold openedOf at hcnt ⊢
            rw [List.countP_cons, List.countP_cons, hcnt i' si' ρ' name' hk', opensLine_skip ht, hq]
            rfl
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact step _ label _ rest rfl hrest (Option.some.inj hR).symm
        · rw [ite_eq_right hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact step _ lab _ rest rfl hrest (Option.some.inj hR).symm

/-- **An opening `readLog` reads is a line opening a round, counted** (`readTail_opening`): the head's line, which
`readLog` reads first, reads at most a tree's `Hello`. -/
theorem readLog_opening {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Verity.Release} {tsF : List (String × Json)}
    (hR : readLog label after records ξ lines = some log) (hF : treeFold records [] lines = some tsF)
    {pre : List Verity.Release} {r : Verity.Release} {post : List Verity.Release} (he : log = pre ++ r :: post)
    {i si n : ℕ} (hr : r.frame = .opening i si n) :
    ∃ preL l postL ρ name k, lines = preL ++ l :: postL ∧ ReadsOpen records l r ρ name k ∧
      KeyAt tsF i si ρ name ∧ ((helloSpecs pre)[i]?.bind fun sp => sp.streams[si]?) = some name ∧
      ∀ i' si' ρ' name', KeyAt tsF i' si' ρ' name' → openedOf pre i' si' = preL.countP (opensLine ρ' name') := by
  cases lines with
  | nil =>
    simp only [readLog, Option.some.injEq] at hR
    subst hR
    simp at he
  | cons l ls =>
    simp only [readLog] at hR
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r₀, ts⟩, hX, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      have hev : event l = .ok j := event_of_toOption hj
      rw [treeFold_tree ht hev hX] at hF
      rcases treeStep_some hX with ⟨root, sj, spec, hr₀, hts, hsp, hT⟩ | ⟨_, _, _, _, _, _, _, _, hst⟩
      · cases pre with
        | nil =>
          simp only [List.nil_append, List.cons.injEq] at he
          rw [← he.1, hr₀] at hr
          cases hr
        | cons q pre' =>
          simp only [List.cons_append, List.cons.injEq] at he
          obtain ⟨rfl, he⟩ := he
          have hS : List.Forall₂ (fun t sp => specOf t.2 = some sp) ts (helloSpecs [r₀]) := by
            have h1 : helloSpecs [r₀] = [spec] := by simp [helloSpecs, hr₀]
            rw [h1, hts, List.nil_append]
            exact .cons hsp .nil
          obtain ⟨preL, l', postL, ρ, name, k, hls, hro, hkey, hname, hcnt⟩ :=
            readTail_opening ls ts none ξ [r₀] hrest hF hS pre' r post he i si n hr
          refine ⟨l :: preL, l', postL, ρ, name, k, by rw [hls]; rfl, hro, hkey, by simpa using hname,
            fun i' si' ρ' name' hk' => ?_⟩
          unfold openedOf at hcnt ⊢
          rw [List.countP_cons, List.countP_cons, hcnt i' si' ρ' name' hk', opensLine_hello hev hT, hr₀]
          rfl
      · simp at hst
    · rw [ite_eq_right ht] at hR
      rw [treeFold_skip ht] at hF
      obtain ⟨preL, l', postL, ρ, name, k, hls, hro, hkey, hname, hcnt⟩ :=
        readTail_opening ls [] none ξ [] hR hF .nil pre r post he i si n hr
      refine ⟨l :: preL, l', postL, ρ, name, k, by rw [hls]; rfl, hro, hkey, by simpa using hname,
        fun i' si' ρ' name' hk' => ?_⟩
      rw [List.countP_cons, opensLine_skip ht, ← hcnt i' si' ρ' name' hk']
      rfl

/-- **The read opens each stream as often as the release log's lines do** (`opensLine`): where `readTail` reads the
lines `lines` after the trees `ts`, and the trees read by the end are `tsF`, tree `i`'s stream `si` among them, stream
`name` of the tree whose root is `ρ` (`KeyAt`). -/
theorem readTail_openedOf {label : String} {after : ByteArray} {records : List Json} {tsF : List (String × Json)}
    {i si : ℕ} {ρ name : String} (hk : KeyAt tsF i si ρ name) :
    ∀ (lines : List String) (ts : List (String × Json)) (pending : Option String) (ξ : ByteArray)
      {log : List Verity.Release}, readTail label after records lines ts pending ξ = some log →
      treeFold records ts lines = some tsF → openedOf log i si = lines.countP (opensLine ρ name)
  | [], _, none, _, log, hR, _ => by
    simp only [readTail, Option.some.injEq] at hR
    subst hR
    rfl
  | [], _, some _, _, _, hR, _ => by simp [readTail] at hR
  | l :: ls, ts, pending, ξ, log, hR, hF => by
    unfold readTail at hR
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r₀, ts'⟩, hX, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      have hev : event l = .ok j := event_of_toOption hj
      rw [treeFold_tree ht hev hX] at hF
      obtain ⟨more, hm⟩ := treeFold_prefix ls ts' tsF hF
      have hts : ∃ more', ts ++ more' = tsF := by
        rcases treeStep_some hX with ⟨root, sj, -, -, hts, -⟩ | ⟨-, -, -, -, hts, -⟩
        · exact ⟨(root, sj) :: more, by rw [← hm, hts, List.append_assoc, List.singleton_append]⟩
        · exact ⟨more, by rw [← hm, hts]⟩
      obtain ⟨more', hm'⟩ := hts
      have hk' := hk
      rw [← hm'] at hk'
      have hq := treeStep_opensAt ht hev hX hk'
      have ih := readTail_openedOf hk ls ts' pending ξ hrest hF
      unfold openedOf at ih ⊢
      rw [List.countP_cons, List.countP_cons, ih, hq]
    · rw [ite_eq_right ht] at hR
      rw [treeFold_skip ht] at hF
      rw [List.countP_cons, opensLine_skip ht]
      cases pending with
      | none => exact readTail_openedOf hk ls ts (some l) ξ hR hF
      | some a =>
        simp only [Option.bind_eq_bind] at hR
        obtain ⟨⟨lab, root⟩, -, hR⟩ := Option.bind_eq_some_iff.1 hR
        simp only [Option.pure_def] at hR
        have step : ∀ (q : Verity.Release) (lab' : String) (ξ' : ByteArray) (rest : List Verity.Release),
            q.frame = .fresh lab' → readTail label after records ls ts none ξ' = some rest → log = q :: rest →
            openedOf log i si = ls.countP (opensLine ρ name) + if false = true then 1 else 0 := by
          intro q lab' ξ' rest hq hrest hlog
          subst hlog
          have ih := readTail_openedOf hk ls ts none ξ' hrest hF
          unfold openedOf at ih ⊢
          rw [List.countP_cons, ih, hq]
          rfl
        by_cases hm : lab = label ∧ root = after
        · rw [ite_eq_left hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact step _ label _ rest rfl hrest (Option.some.inj hR).symm
        · rw [ite_eq_right hm] at hR
          obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
          exact step _ lab _ rest rfl hrest (Option.some.inj hR).symm

/-- **`readLog`'s read opens each stream as often as the release log's lines do** (`readTail_openedOf`), the head's
line read first. -/
theorem readLog_openedOf {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Verity.Release} {tsF : List (String × Json)}
    (hR : readLog label after records ξ lines = some log) (hF : treeFold records [] lines = some tsF) {i si : ℕ}
    {ρ name : String} (hk : KeyAt tsF i si ρ name) : openedOf log i si = lines.countP (opensLine ρ name) := by
  cases lines with
  | nil =>
    simp only [readLog, Option.some.injEq] at hR
    subst hR
    rfl
  | cons l ls =>
    simp only [readLog] at hR
    by_cases ht : treeLine l = true
    · rw [ite_eq_left ht] at hR
      simp only [Option.bind_eq_bind] at hR
      obtain ⟨⟨r₀, ts⟩, hX, hR⟩ := Option.bind_eq_some_iff.1 hR
      obtain ⟨j, hj, hX⟩ := Option.bind_eq_some_iff.1 hX
      simp only [Option.pure_def] at hR
      obtain ⟨rest, hrest, hR⟩ := Option.bind_eq_some_iff.1 hR
      cases hR
      have hev : event l = .ok j := event_of_toOption hj
      rw [treeFold_tree ht hev hX] at hF
      rcases treeStep_some hX with ⟨root, sj, spec, hr₀, hts, hsp, hT⟩ | ⟨_, _, _, _, _, _, _, _, hst⟩
      · have ih := readTail_openedOf hk ls ts none ξ hrest hF
        unfold openedOf at ih ⊢
        rw [List.countP_cons, List.countP_cons, ih, opensLine_hello hev hT, hr₀]
        rfl
      · simp at hst
    · rw [ite_eq_right ht] at hR
      rw [treeFold_skip ht] at hF
      rw [List.countP_cons, opensLine_skip ht, ← readTail_openedOf hk ls [] none ξ hR hF]
      rfl

/-! ## The bridge -/

/-- **The per-round bridge**: where `coinTreesOf` accepts the release log `lines` and `readLog` reads it, each opening
the read shows, of tree `i`'s stream `si`, is a line of the log that reads as it (`ReadsOpen`), opening round `k` of the
stream `name` of the tree whose root is `ρ`: tree `i`'s stream `si` among `coinTreesOf`'s trees (`KeyAt`), named
`name` in the specs of the `Hello`s before it, where `k` is the count of that stream's openings before it
(`openedOf`). -/
theorem readLog_round {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Verity.Release} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    (hR : readLog label after records ξ lines = some log) {pre : List Verity.Release} {r : Verity.Release}
    {post : List Verity.Release} (he : log = pre ++ r :: post) {i si n : ℕ} (hr : r.frame = .opening i si n) :
    ∃ l ρ name k, l ∈ lines ∧ ReadsOpen records l r ρ name k ∧ KeyAt (heldTrees trees) i si ρ name ∧
      ((helloSpecs pre)[i]?.bind fun sp => sp.streams[si]?) = some name ∧ k = openedOf pre i si := by
  obtain ⟨preL, l, postL, ρ, name, k, hls, hro, hkey, hname, hcnt⟩ :=
    readLog_opening hR (coinTreesOf_fold records hT) he hr
  refine ⟨l, ρ, name, k, by rw [hls]; simp, hro, hkey, hname, ?_⟩
  rw [hcnt i si ρ name hkey]
  exact coinTreesOf_round hT hls hro.1

/-- The `i`-th tree is the first of its root, where the trees' roots are distinct. -/
theorem findIdx?_root {trees : Array LoggedTree} (hn : (trees.toList.map (·.root)).Nodup) {i : ℕ} {t : LoggedTree}
    (h : trees[i]? = some t) : trees.findIdx? (·.root == t.root) = some i := by
  have hm : (trees.toList.map (·.root))[i]? = some t.root := by
    rw [List.getElem?_map, Array.getElem?_toList, h]
    rfl
  have e := Lines.findIdx_nodup hn hm
  rw [List.findIdx?_map] at e
  rw [← findIdx?_toList]
  exact e

/-- **A round a tree shows opened is below the read's openings of its stream**: where `coinTreesOf` accepts the
release log `lines` and `readLog` reads it, round `k` of stream `name` that a tree `t` of `coinTreesOf`'s shows opened
(`LoggedTree.opening`) is below the openings the read shows of tree `i`'s stream `si`, that stream of `t` (`KeyAt`). -/
theorem opening_lt_openedOf {label : String} {after : ByteArray} {records : List Json} {ξ : ByteArray}
    {lines : List String} {log : List Verity.Release} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    (hR : readLog label after records ξ lines = some log) {t : LoggedTree} (ht : t ∈ trees) {name : String} {k : ℕ}
    {o : TreeOpening} (ho : t.opening name k = some o) {i si : ℕ} (hk : KeyAt (heldTrees trees) i si t.root name) :
    k < openedOf log i si := by
  unfold LoggedTree.opening at ho
  have hmem : o ∈ t.opened := Array.mem_of_find?_eq_some ho
  have hp := Array.find?_some ho
  simp only [Bool.and_eq_true, beq_iff_eq] at hp
  obtain ⟨hs, rfl⟩ := hp
  have hTr := Lines.trees_log hT
  obtain ⟨j, hj⟩ := Array.mem_iff_getElem?.1 ht
  have hc := (coinTreesOf_counted hT).held j t hj (findIdx?_root hTr.nodup hj) name
  rw [readLog_openedOf hR (coinTreesOf_fold records hT) hk, ← hc, ← hs]
  exact hTr.rounds t ht o hmem

end Verity.Sampled
