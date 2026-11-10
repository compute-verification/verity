module

public import Proofs.Properties.Verity.Sampled.Trees

/-!
# The release log's first opening, and the one coin tree of v2's sessions

v2 holds a `--zk` session's record to the coin trees the release log shows sampled (`coinTreesOf`), one tree a session
(`sessionsTreeReason`):

* **One tree a session** (`sessions_size`, `treesHeld_size`): where `treesHeld` accepts, the log shows as many trees as
  the evidence names sessions, since `sessionsTreeReason` refuses a tree no session records.
* **The first opening** (`FirstRound`, `first_round`): `coinTreesOf` takes a stream's round `k` only as its `k`-th
  opened, so the log's first line opening a round opens round 0, of a tree the log shows sampled.
* **The link's coins** (`link_some`): a record `sessionTreeReason` accepts gives the coins of its session stream's
  round 0 (`Verity.Sampled.recordRound`), its link points.
-/

@[expose] public section

namespace Verity.Sampled.LinkTrees

open Lean (Json)
open Spot Spot.Challenger

/-! ## One tree a session -/

/-- **A loop that returns at its first reason and pushes a member at each step it yields**, from the state `(none, u)`,
returns a reason, or none at a state of `u.size + l.length` members. -/
theorem forIn_count {α γ β : Type}
    {f : α → Option (Option γ) × Array β → Id (ForInStep (Option (Option γ) × Array β))}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ b, s = ForInStep.yield (none, u.push b))) :
    ∀ (l : List α) (u : Array β), (∃ w, Prod.fst (forIn (m := Id) l (none, u) f) = some (some w)) ∨
      (Prod.fst (forIn (m := Id) l (none, u) f) = none ∧
        (Prod.snd (forIn (m := Id) l (none, u) f)).size = u.size + l.length)
  | [], _ => .inr ⟨rfl, rfl⟩
  | a :: l, u => by
    rw [List.forIn_cons]
    rcases hf a u _ rfl with ⟨w, u', hw⟩ | ⟨b, hy⟩
    · exact .inl ⟨w, by rw [hw]; rfl⟩
    · rw [hy]
      rcases forIn_count hf l (u.push b) with ⟨w, hw⟩ | ⟨hn, hs⟩
      · exact .inl ⟨w, hw⟩
      · exact .inr ⟨hn, hs.trans (by rw [Array.size_push, List.length_cons]; omega)⟩

/-- **A loop that returns at its first reason and pushes a member at each step it yields, then its reason**: where none
returns, it returns none at a state of a member for each step. -/
theorem ret_count {α γ β : Type}
    {f : α → Option (Option γ) × Array β → Id (ForInStep (Option (Option γ) × Array β))}
    {g : Option (Option γ) × Array β → Id (Option γ)}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ b, s = ForInStep.yield (none, u.push b)))
    (hg : ∀ w u, g (some (some w), u) ≠ none) {xs : Array α} {u₀ : Array β}
    (h : (forIn (m := Id) xs (none, u₀) f >>= g) = none) :
    ∃ u : Array β, u.size = u₀.size + xs.size ∧ g (none, u) = none := by
  rw [← Array.forIn_toList] at h
  rcases forIn_count hf xs.toList u₀ with ⟨w, hw⟩ | ⟨hn, hs⟩
  · have e : forIn (m := Id) xs.toList (none, u₀) f =
        (some (some w), Prod.snd (forIn (m := Id) xs.toList (none, u₀) f)) := Prod.ext hw rfl
    rw [e] at h
    exact absurd h (hg w _)
  · have e : forIn (m := Id) xs.toList (none, u₀) f =
        (none, Prod.snd (forIn (m := Id) xs.toList (none, u₀) f)) := Prod.ext hn rfl
    rw [e] at h
    exact ⟨Prod.snd (forIn (m := Id) xs.toList (none, u₀) f), by rw [hs, Array.length_toList], h⟩

/-- **Sessions `sessionsTreeReason` accepts use every tree**: as many as the log shows. -/
theorem sessions_size {trees : Array LoggedTree} {records : Array (String × Array (String × Option Nat) × Json)}
    (h : sessionsTreeReason trees records = none) : trees.size = records.size := by
  unfold sessionsTreeReason at h
  dsimp only [Id.run] at h
  have hc := ret_count ?_ (fun w _ hw => by cases hw) h
  · obtain ⟨u, hu, hg⟩ := hc
    have hu' : u.size = records.size := by rw [hu]; simp
    rw [← hu']
    try dsimp only at hg
    split at hg
    · exact (beq_iff_eq.1 ‹_›).symm
    · cases hg
  · rintro ⟨name, tables, record⟩ u s hs
    split at hs
    case h_1 => exact .inl ⟨_, _, hs.symm⟩
    exact .inr ⟨_, hs.symm⟩

/-- **The trees `treesHeld` holds the evidence's sessions to are one a session.** -/
theorem treesHeld_size {p : Spot.OneStageZk.Own} {ts : Array LoggedTree}
    {e : Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    (h : Spot.OneStageZk.treesHeld p ts e bs = .ok ()) : ts.size = e.sessions.size := by
  unfold Spot.OneStageZk.treesHeld at h
  dsimp only at h
  obtain ⟨named, hn, hrest⟩ := Trees.bind_ok h
  rw [← Array.forIn_toList] at hn
  have hI := Trees.forIn_inv
    (I := fun pre (named : Array (String × Array (String × Option Nat) × Json)) => named.size = pre.length) ?_
    e.sessions.toList [] #[] named (by simp) hn
  · have hsess : sessionsTreeReason ts named = none := by
      split at hrest
      case h_1 => cases hrest
      rename_i hne
      cases hw : sessionsTreeReason ts named with
      | none => rfl
      | some w => exact (hne w hw).elim
    rw [sessions_size hsess, hI]
    simp
  · intro pre x named' res hI' hres
    obtain ⟨b', -, hres⟩ := Trees.bind_ok hres
    split at hres
    case h_2 => cases hres
    cases hres
    refine ⟨_, rfl, ?_⟩
    rw [Array.size_push, hI', List.length_append, List.length_singleton]

/-- An array of one member holds no other. -/
theorem eq_of_size_one {β : Type} {xs : Array β} (h : xs.size = 1) {a b : β} (ha : a ∈ xs) (hb : b ∈ xs) :
    a = b := by
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 ha
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 hb
  have e : i = j := by omega
  subst e
  rfl

/-! ## The link's coins -/

/-- **A record `sessionTreeReason` accepts, at any tables, gives its link points' coins**, its session stream's
round 0. -/
theorem link_some {trees : Array LoggedTree} {tables : Array (String × Option Nat)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ cs, Verity.Sampled.recordRound record TREE_SESSION 0 = some cs := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rename_i ps hlink
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rename_i rootF points streams htri
  simp only [Prod.mk.injEq] at htri hlink
  obtain ⟨-, hpts, -⟩ := htri
  obtain ⟨-, -, hps⟩ := hlink
  refine ⟨points, ?_⟩
  unfold Verity.Sampled.recordRound
  rw [ite_eq_left (by simp), hps]
  exact hpts

/-! ## The first opening -/

/-- **A coin tree's line whose event isn't a tree's sample opens a round** (`Verity.Sampled.openLine`). -/
theorem openLine_of_open {l : String} {d : Json} (ht : treeLine l = true) (hev : event l = .ok d)
    (hT : ¬ jsonStrIs (getD d "event") TREE_EVENT = true) : Verity.Sampled.openLine l = true := by
  have htl : (jsonStrIs (getD d "event") TREE_EVENT || jsonStrIs (getD d "event") OPEN_EVENT) = true := by
    simpa only [treeLine, hev] using ht
  have hol : Verity.Sampled.openLine l = jsonStrIs (getD d "event") OPEN_EVENT := by
    simp only [Verity.Sampled.openLine, hev]
  rw [hol]
  simpa [hT] using htl

theorem snoc_split {β : Type} {pre p₀ p₁ : List β} {x l : β} (h : pre ++ [x] = p₀ ++ l :: p₁) :
    (p₀ = pre ∧ l = x ∧ p₁ = []) ∨ ∃ p₁', pre = p₀ ++ l :: p₁' := by
  rcases p₁.eq_nil_or_concat with rfl | ⟨p₁', y, rfl⟩
  · obtain ⟨h1, h2⟩ := List.append_inj' h rfl
    exact .inl ⟨h1.symm, (List.cons.inj h2).1.symm, rfl⟩
  · rw [List.concat_eq_append] at h
    have h' : pre ++ [x] = (p₀ ++ l :: p₁') ++ [y] := by rw [h]; simp
    obtain ⟨h1, -⟩ := List.append_inj' h' rfl
    exact .inr ⟨p₁', h1⟩

theorem root_mem_modify {xs : Array LoggedTree} {i : Nat} {f : LoggedTree → LoggedTree}
    (hf : ∀ t, (f t).root = t.root) {t : LoggedTree} (ht : t ∈ xs) : ∃ t' ∈ xs.modify i f, t'.root = t.root := by
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 ht
  have hj' : j < (xs.modify i f).size := by simpa using hj
  refine ⟨(xs.modify i f)[j], Array.getElem_mem hj', ?_⟩
  rw [Array.getElem_modify]
  split
  · exact hf _
  · rfl

/-- **The trees after the release log's lines `pre`, up to its first opening**: where no line of `pre` opens a round
(`Verity.Sampled.openLine`), none of `trees` shows a round opened; and the first line of `pre` that `coinTreesOf`
reads as an opening (a coin tree's line whose event isn't a tree's sample) opens round 0, of a tree of `trees` by its
root. -/
def FirstRound (pre : List String) (trees : Array LoggedTree) : Prop :=
  ((∀ l ∈ pre, Verity.Sampled.openLine l = false) → ∀ t ∈ trees, t.opened = #[]) ∧
    ∀ p₀ l p₁ d, pre = p₀ ++ l :: p₁ → (∀ l' ∈ p₀, Verity.Sampled.openLine l' = false) → treeLine l = true →
      event l = .ok d → jsonStrIs (getD d "event") TREE_EVENT = false →
      nat? (getD d "round") = some 0 ∧ ∃ root, getD d "tree" = .str root ∧ ∃ t ∈ trees, t.root = root

/-- A line `coinTreesOf` reads as no opening keeps `FirstRound`, at trees of the earlier ones and new ones that open
nothing, where each earlier tree is a later one. -/
theorem first_keep {pre : List String} {line : String} {ts ts' : Array LoggedTree} (hI : FirstRound pre ts)
    (hnot : ∀ d, treeLine line = true → event line = .ok d → jsonStrIs (getD d "event") TREE_EVENT = false → False)
    (hnew : ∀ t ∈ ts', t ∈ ts ∨ t.opened = #[]) (hold : ∀ t ∈ ts, t ∈ ts') : FirstRound (pre ++ [line]) ts' := by
  refine ⟨fun hno t ht => ?_, fun p₀ l p₁ d he hno htl hev hT => ?_⟩
  · rcases hnew t ht with ht | he
    · exact hI.1 (fun l hl => hno l (List.mem_append_left _ hl)) t ht
    · exact he
  · rcases snoc_split he with ⟨rfl, rfl, rfl⟩ | ⟨p₁', hp⟩
    · exact (hnot d htl hev hT).elim
    · obtain ⟨hk, root, hroot, t, ht, htr⟩ := hI.2 p₀ l p₁' d hp hno htl hev hT
      exact ⟨hk, root, hroot, t, hold t ht, htr⟩

/-- A line `coinTreesOf` reads as opening round `k` of stream `s` of the tree `t` of root `root`, its `k`-th opened,
keeps `FirstRound`. -/
theorem first_open {pre : List String} {line : String} {ts : Array LoggedTree} {d : Json} {root s : String}
    {i k : Nat} {t : LoggedTree} {f : LoggedTree → LoggedTree} (hI : FirstRound pre ts) (htl : treeLine line = true)
    (hev : event line = .ok d) (hT : ¬ jsonStrIs (getD d "event") TREE_EVENT = true)
    (hroot : getD d "tree" = .str root) (hidx : ts.findIdx? (·.root == root) = some i) (hget : ts[i]? = some t)
    (hk : nat? (getD d "round") = some k) (hk0 : k = (t.opened.filter (·.stream == s)).size)
    (hf : ∀ t, (f t).root = t.root) : FirstRound (pre ++ [line]) (ts.modify i f) := by
  have hopen := openLine_of_open htl hev hT
  obtain ⟨hlt, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
  have hti : ts[i] = t := by
    rw [Array.getElem?_eq_getElem hlt] at hget
    exact Option.some.inj hget
  rw [hti] at hp
  have hroot' : t.root = root := beq_iff_eq.1 hp
  have htm : t ∈ ts := hti ▸ Array.getElem_mem hlt
  refine ⟨fun hno => absurd (hno line (List.mem_append_right _ (List.mem_singleton_self _))) (by simp [hopen]),
    fun p₀ l p₁ d' he hno htl' hev' hT' => ?_⟩
  rcases snoc_split he with ⟨rfl, rfl, rfl⟩ | ⟨p₁', hp'⟩
  · obtain rfl := Except.ok.inj (hev.symm.trans hev')
    have he0 : t.opened = #[] := hI.1 hno t htm
    refine ⟨?_, root, hroot, ?_⟩
    · rw [hk, hk0, he0, Array.filter_empty, Array.size_empty]
    · obtain ⟨t', ht', htr⟩ := root_mem_modify (i := i) hf htm
      exact ⟨t', ht', htr.trans hroot'⟩
  · obtain ⟨hk', root', hroot'', t', ht', htr'⟩ := hI.2 p₀ l p₁' d' hp' hno htl' hev' hT'
    obtain ⟨t'', ht'', htr''⟩ := root_mem_modify (i := i) hf ht'
    exact ⟨hk', root', hroot'', t'', ht'', htr''.trans htr'⟩

/-- **The trees `coinTreesOf` reads hold `FirstRound` of the whole log.** -/
theorem first_round_inv {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    FirstRound log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Trees.forIn_inv_run (I := FirstRound) ?_
    ⟨fun _ t ht => absurd ht (by simp), fun p₀ l p₁ d he => absurd he (by simp)⟩ h
  intro pre line ts r hI hr
  split at hr
  case isFalse =>
    rename_i htl
    cases hr
    exact ⟨ts, rfl, first_keep hI (fun _ ht _ _ => htl ht) (fun t ht => .inl ht) fun _ ht => ht⟩
  rename_i htl
  cases hev : event line with
  | error e => rw [hev] at hr; cases hr
  | ok d =>
    rw [hev, Trees.ok_bind] at hr
    split at hr
    · rename_i hTr
      split at hr
      case isFalse => cases hr
      split at hr
      case isTrue => cases hr
      cases hr
      refine ⟨_, rfl, first_keep hI (fun d' _ hev' hT' => ?_) (fun t ht => ?_) fun t ht => Array.mem_push.2 (.inl ht)⟩
      · cases hev.symm.trans hev'
        rw [hTr] at hT'
        cases hT'
      · rcases Array.mem_push.1 ht with ht | rfl
        · exact .inl ht
        · exact .inr rfl
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
      simp only [Prod.mk.injEq] at hkn
      split at hr
      all_goals
        try dsimp only at hr
        split at hr
        · rename_i hcond
          simp only [Bool.and_eq_true] at hcond
          cases hr
          exact ⟨_, rfl, first_open hI htl hev hopen htree hidx hget hkn.1 (beq_iff_eq.1 hcond.1.1.1.1.2)
            fun _ => rfl⟩
        · cases hr

/-- **The release log's first opening opens round 0 of a tree the log shows sampled**: where `coinTreesOf` reads the
log `p₀ ++ l :: p₁` to `trees` and no line of `p₀` opens a round, the line `l`, read as an opening, names round 0 and the
root of a tree of `trees`. -/
theorem first_round {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees)
    {p₀ p₁ : List String} {l : String} {d : Json} (hlog : log = p₀ ++ l :: p₁)
    (hno : ∀ l' ∈ p₀, Verity.Sampled.openLine l' = false) (htl : treeLine l = true) (hev : event l = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = false) :
    nat? (getD d "round") = some 0 ∧ ∃ root, getD d "tree" = .str root ∧ ∃ t ∈ trees, t.root = root :=
  (first_round_inv h).2 p₀ l p₁ d hlog hno htl hev hT

end Verity.Sampled.LinkTrees
