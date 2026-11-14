module

public import Proofs.Properties.Verity.SampledTwo.SessionOpened
public import Proofs.Properties.Verity.SampledTwo.MemberChecks
public import Proofs.Properties.Verity.Sampled.SpecTie
import all Init.Data.Repr

/-!
# The releases the session-round locator places come in the order of their rounds

On a play the two-stage form accepts, take the interior challenger session's log as the check read it
(`SessionCoins.interiorRead`). The session-round locator (`SessionCoins.locOf`) puts some of its releases on rounds of
the second stage's sessions. This file proves that it puts them in order: of two releases it places, the one earlier
in the log is on the lower round (`SessionCoins.LocatedRising`, `SessionCoins.located_rising`). So every placed
release before round `n`'s is of a round below `n`.

Two placed releases are of different sessions or of one.

* **Different sessions.** The members' check opens the sessions' trees in the sessions' order
  (`SessionCoins.MembersHeld.before`), and a later session's rounds are higher (`SessionCoins.off_mono`).
* **One session, so one tree.** Four steps.
  * *The log's order is the lines' order.* The read gives each opening its line and, for every stream, as many
    openings before the release as lines before its line (`Verity.Sampled.readLog_opening`), so the earlier
    release's line comes first (`Verity.Sampled.split_before`).
  * *The lines' order is the tree's order.* Each tree `coinTreesOf` holds lists, as the rounds it shows opened, the
    stream and the round of the lines that open a round of its root, in the lines' order (`Verity.Sampled.OpensListed`,
    `Verity.Sampled.coinTreesOf_listed`).
  * *The refusal.* The form refuses a tree whose rounds, the last one aside, are not opened stream by stream
    (`Spot.Challenger.repStreamOrder`), so the earlier line is of an earlier place in the tree's spec, or of the same
    place and an earlier round (`Verity.Sampled.lines_rise`). The later release is not the tree's last opening: the
    members' check holds the last to be the closing round, the session stream's round 1, and the locator never places
    that (`SessionCoins.slotStream_ne_closing`).
  * *Places are session rounds.* The tree's spec lists the session stream, then the table's rep 0 stream, then its
    rep 1 stream (`Verity.Sampled.SessionStreams`): the session's verdict holds the record's `coin_tree.spec` to the
    session's own tree, `Flock.Zk.coinSpec` at the form's one table (`SessionCoins.setupAt_coinTree`,
    `Verity.Sampled.specValue_of_decode`), and the members' check holds the record's `coin_tree` to the logged tree's.
    So a stream's place is the place the session's layout gives its round (`SessionCoins.place_of_streams`,
    `SessionCoins.slotPlace_lt`).

The file also states, of the same read, what one-stage v2's order check gives it (`SessionCoins.ReadLinkFirst`,
`SessionCoins.readLinkFirst`): no release of the second stage's sample follows an opening, and the first opening is of
the session stream.

`SessionCoins.runMembers_sessions` is the per-member form's run as a whole: every statement's every session accepted
(`MemberChecks.SessionAccepted`), which `MemberChecks.stepMember_sessions` says of one statement's step.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Spot Spot.Challenger
open FlockDeploy (JsonEquiv)
open Properties.Challenger.CanonRead (sorted_getD sortedKeys_of_wf)

/-! ## Lists -/

/-- Of a list without its last entry, pairwise related: an entry and a later one that is not the last are related. -/
theorem pairwise_dropLast_mid {α : Type} {R : α → α → Prop} {xs ys zs : List α} {a b : α}
    (h : (xs ++ a :: (ys ++ b :: zs)).dropLast.Pairwise R) (hzs : zs ≠ []) : R a b := by
  rw [List.dropLast_append_of_ne_nil (by simp), List.dropLast_cons_of_ne_nil (by simp),
    List.dropLast_append_of_ne_nil (by simp), List.dropLast_cons_of_ne_nil hzs] at h
  exact (List.pairwise_cons.1 (List.pairwise_append.1 h).2.1).1 b (by simp)

/-- Of two splits of one list, the one with fewer entries of a kind before its point comes first. -/
theorem split_before {α : Type} {p : α → Bool} {A B A' B' : List α} {x x' : α}
    (h : A ++ x :: B = A' ++ x' :: B') (hc : A.countP p < A'.countP p) : ∃ M, A' = A ++ x :: M := by
  rcases List.append_eq_append_iff.1 h with ⟨a', hA, ha⟩ | ⟨c', hA, -⟩
  · cases a' with
    | nil =>
      rw [List.append_nil] at hA
      rw [hA] at hc
      exact absurd hc (Nat.lt_irrefl _)
    | cons y a'' =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      exact ⟨a'', hA⟩
  · rw [hA, List.countP_append] at hc
    omega

theorem filterMap_snoc_none {α β : Type} {f : α → Option β} {pre : List α} {x : α} (h : f x = none) :
    (pre ++ [x]).filterMap f = pre.filterMap f := by
  rw [List.filterMap_append, List.filterMap_cons_none h, List.filterMap_nil, List.append_nil]

theorem filterMap_snoc_some {α β : Type} {f : α → Option β} {pre : List α} {x : α} {b : β} (h : f x = some b) :
    (pre ++ [x]).filterMap f = pre.filterMap f ++ [b] := by
  rw [List.filterMap_append, List.filterMap_cons_some h, List.filterMap_nil]

/-! ## What `coinTreesOf` lists -/

/-- **The stream and the round release log line `l` opens of the coin tree whose root is `ρ`**, as `coinTreesOf`
reads it (`OpensRound`); none where it opens no round of that tree. -/
def lineOpening (ρ : String) (l : String) : Option (String × ℕ) :=
  if treeLine l = true then
    match event l with
    | .ok d =>
      if jsonStrIs (getD d "event") TREE_EVENT = true then none
      else
        match getD d "tree", getD d "stream", nat? (getD d "round") with
        | .str root, .str s, some k => if root = ρ then some (s, k) else none
        | _, _, _ => none
    | .error _ => none
  else none

theorem lineOpening_skip {ρ l : String} (h : ¬treeLine l = true) : lineOpening ρ l = none := by
  simp [lineOpening, h]

theorem lineOpening_hello {ρ l : String} {d : Json} (he : event l = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = true) : lineOpening ρ l = none := by
  simp [lineOpening, he, hT]

theorem lineOpening_open {ρ l root s : String} {k : ℕ} {d : Json} (ht : treeLine l = true) (he : event l = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = false) (hr : getD d "tree" = .str root)
    (hs : getD d "stream" = .str s) (hk : nat? (getD d "round") = some k) :
    lineOpening ρ l = if root = ρ then some (s, k) else none := by
  simp [lineOpening, ht, he, hT, hr, hs, hk]

theorem OpensRound.lineOpening {l root name : String} {k : ℕ} (h : OpensRound l root name k) :
    lineOpening root l = some (name, k) := by
  obtain ⟨ht, d, he, hT, hr, hs, hk⟩ := h
  rw [lineOpening_open ht he hT hr hs hk, ite_eq_left rfl]

/-- **The trees `coinTreesOf` holds after the release log's lines `pre`, listed**: each tree, the first of its root,
shows opened the rounds, each its stream and its number, that the lines of `pre` open of that root, in the lines'
order (`lineOpening`); and `pre` opens no round of a root none of the trees has. -/
structure OpensListed (pre : List String) (trees : Array LoggedTree) : Prop where
  held : ∀ i t, trees[i]? = some t → trees.findIdx? (·.root == t.root) = some i →
    t.opened.toList.map (fun o => (o.stream, o.round)) = pre.filterMap (lineOpening t.root)
  other : ∀ ρ, trees.findIdx? (·.root == ρ) = none → pre.filterMap (lineOpening ρ) = []

theorem OpensListed.nil : OpensListed [] #[] where
  held i t h := by simp at h
  other _ _ := rfl

theorem OpensListed.skip {pre : List String} {ts : Array LoggedTree} {line : String} (hI : OpensListed pre ts)
    (h : ¬treeLine line = true) : OpensListed (pre ++ [line]) ts where
  held i t hi hf := by rw [filterMap_snoc_none (lineOpening_skip h), hI.held i t hi hf]
  other ρ hf := by rw [filterMap_snoc_none (lineOpening_skip h), hI.other ρ hf]

theorem OpensListed.hello {pre : List String} {ts : Array LoggedTree} {line : String} {d : Json}
    {t' : LoggedTree} (hI : OpensListed pre ts) (he : event line = .ok d)
    (hT : jsonStrIs (getD d "event") TREE_EVENT = true) (ht' : t'.opened = #[]) :
    OpensListed (pre ++ [line]) (ts.push t') where
  held i t hi hf := by
    rw [filterMap_snoc_none (lineOpening_hello he hT)]
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
        rw [ht', hI.other _ hfi]
        rfl
    · cases hfi : ts.findIdx? (·.root == t.root) with
      | some j =>
        rw [hfi, Option.some_or] at hf
        obtain rfl := Option.some.inj hf
        exact hI.held j t hi hfi
      | none =>
        rw [hfi, Option.none_or] at hf
        split at hf
        · obtain rfl := Option.some.inj hf
          exact absurd rfl ‹¬_›
        · cases hf
  other ρ hf := by
    rw [Array.findIdx?_push] at hf
    cases hfi : ts.findIdx? (·.root == ρ) with
    | some j => rw [hfi, Option.some_or] at hf; cases hf
    | none => rw [filterMap_snoc_none (lineOpening_hello he hT), hI.other ρ hfi]

theorem OpensListed.opening {pre : List String} {ts : Array LoggedTree} {line : String} {d : Json}
    {root s : String} {i k : ℕ} {t : LoggedTree} {o : TreeOpening} (hI : OpensListed pre ts)
    (htl : treeLine line = true) (he : event line = .ok d) (hT : jsonStrIs (getD d "event") TREE_EVENT = false)
    (htree : getD d "tree" = .str root) (hidx : ts.findIdx? (·.root == root) = some i) (hget : ts[i]? = some t)
    (hstream : getD d "stream" = .str s) (hk : nat? (getD d "round") = some k) (ho : o.stream = s)
    (hor : o.round = k) :
    OpensListed (pre ++ [line]) (ts.modify i fun u => { u with opened := u.opened.push o }) := by
  have hl : ∀ ρ, lineOpening ρ line = if root = ρ then some (s, k) else none := fun ρ =>
    lineOpening_open htl he hT htree hstream hk
  obtain ⟨hi, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hidx
  have hti : ts[i] = t := by
    rw [Array.getElem?_eq_getElem hi] at hget
    exact Option.some.inj hget
  rw [hti] at hp
  have hroot : t.root = root := beq_iff_eq.1 hp
  have hfm := findIdx?_modify_root (ts := ts) (i := i) (f := fun u => { u with opened := u.opened.push o })
    (fun _ => rfl)
  refine ⟨fun j t' hj hf => ?_, fun ρ hf => ?_⟩
  · rw [hfm] at hf
    rw [Array.getElem?_modify] at hj
    split at hj
    · rename_i hij
      subst hij
      rw [hget] at hj
      obtain rfl := Option.some.inj hj
      dsimp only at hf ⊢
      have hat : lineOpening t.root line = some (o.stream, o.round) := by
        rw [hl, ite_eq_left hroot.symm, ho, hor]
      rw [filterMap_snoc_some hat, ← hI.held i t hget hf, Array.toList_push, List.map_append]
      rfl
    · rename_i hij
      have hne : ¬ root = t'.root := by
        intro hb
        rw [← hb] at hf
        exact hij (Option.some.inj (hidx.symm.trans hf))
      have hat : lineOpening t'.root line = none := by rw [hl, ite_eq_right hne]
      rw [filterMap_snoc_none hat]
      exact hI.held j t' hj hf
  · rw [hfm] at hf
    have hne : ¬ root = ρ := by
      intro hb
      rw [← hb, hidx] at hf
      cases hf
    have hat : lineOpening ρ line = none := by rw [hl, ite_eq_right hne]
    rw [filterMap_snoc_none hat]
    exact hI.other ρ hf

/-- **`coinTreesOf` lists each tree's rounds in the lines' order**: where it accepts the release log, its trees are
as `OpensListed` says. -/
theorem coinTreesOf_listed {log : List String} {trees : Array LoggedTree} (h : coinTreesOf log = .ok trees) :
    OpensListed log trees := by
  unfold coinTreesOf at h
  dsimp only at h
  refine Trees.forIn_inv_run (I := OpensListed) ?_ OpensListed.nil h
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
        · cases hr
          exact ⟨_, rfl, hI.opening htl hev (Bool.eq_false_iff.2 hT) htree hidx hget hstream hkn.1 rfl rfl⟩
        · cases hr

/-! ## The refusal on the lines -/

/-- Of the rounds a tree shows opened, an earlier one and a later one that is not the last are in the order the
refusal holds the tree's rounds to, the last one aside. -/
theorem mid_of_listed {t : LoggedTree} {xs ys zs : List (String × ℕ)} {a b : String × ℕ}
    (hW : t.opened.toList.map (fun o => (o.stream, o.round)) = xs ++ a :: (ys ++ b :: zs))
    (hp : (t.opened.pop.toList.map fun o => (streamPlace t.spec o.stream, o.round)).Pairwise
      fun x y => roundBefore x y = true)
    {o : TreeOpening} (hb : t.opened.back? = some o) (hne : (o.stream, o.round) ≠ b) :
    roundBefore (streamPlace t.spec a.1, a.2) (streamPlace t.spec b.1, b.2) = true := by
  have hzs : zs ≠ [] := by
    intro hz
    rw [hz] at hW
    have hlast : (t.opened.toList.map fun o => (o.stream, o.round)).getLast? = some b := by
      have hcat : xs ++ a :: (ys ++ [b]) = (xs ++ a :: ys) ++ [b] := by simp
      rw [hW, hcat, List.getLast?_concat]
    rw [List.getLast?_map, Array.getLast?_toList, hb] at hlast
    exact hne (Option.some.inj hlast)
  have hp' : (((t.opened.toList.map fun o => (o.stream, o.round)).map
      fun p : String × ℕ => (streamPlace t.spec p.1, p.2)).dropLast).Pairwise
      fun x y => roundBefore x y = true := by
    rw [List.map_map, ← List.map_dropLast, ← Array.toList_pop]
    exact hp
  rw [hW, List.map_append, List.map_cons, List.map_append, List.map_cons] at hp'
  exact pairwise_dropLast_mid (R := fun x y => roundBefore x y = true) hp' (fun h => hzs (List.map_eq_nil_iff.1 h))

/-- **Where a tree's rounds are opened stream by stream, an earlier line opens an earlier round**: where `coinTreesOf`
accepts the release log `lines` and `repStreamOrder` passes its tree `t`, of two lines that open rounds of `t`, the
later not the tree's last opening, the earlier is of a stream the tree's spec lists before the later's, or an earlier
round of the same place's stream (`roundBefore`). -/
theorem lines_rise {lines : List String} {trees : Array LoggedTree} {i : ℕ} {t : LoggedTree}
    (hT : coinTreesOf lines = .ok trees) (hi : trees[i]? = some t)
    (hf : trees.findIdx? (·.root == t.root) = some i) (hord : repStreamOrder t = true)
    {L₁ L₂ L₃ : List String} {l l' : String} (hl : lines = L₁ ++ l :: (L₂ ++ l' :: L₃))
    {name name' : String} {k k' : ℕ} (ho : OpensRound l t.root name k) (ho' : OpensRound l' t.root name' k')
    {o : TreeOpening} (hb : t.opened.back? = some o) (hne : (o.stream, o.round) ≠ (name', k')) :
    roundBefore (streamPlace t.spec name, k) (streamPlace t.spec name', k') = true := by
  have hW := (coinTreesOf_listed hT).held i t hi hf
  rw [hl, List.filterMap_append, List.filterMap_cons_some (OpensRound.lineOpening ho), List.filterMap_append,
    List.filterMap_cons_some (OpensRound.lineOpening ho')] at hW
  have hp : (t.opened.pop.toList.map fun o => (streamPlace t.spec o.stream, o.round)).Pairwise
      fun x y => roundBefore x y = true := VStar.Recursive.SampledTwo.roundsOrdered_pairwise hord
  have h := mid_of_listed hW hp hb hne
  exact h

/-! ## A stream's place in a session's tree -/

/-- A stream's place in a spec is where the spec's streams first name it. -/
theorem streamPlace_of_key {spec : Json} {name : String} {s : ℕ}
    (h : (specStreams spec).findIdx? (jsonStrIs · name) = some s) : streamPlace spec name = s := by
  unfold streamPlace
  unfold specStreams at h
  generalize getD spec "streams" = g at h ⊢
  cases g with
  | arr ss =>
    dsimp only at h ⊢
    rw [h]
    rfl
  | _ => simp at h

theorem findIdx_three {a b c name : String} {s : ℕ}
    (h : [Json.str a, Json.str b, Json.str c].findIdx? (jsonStrIs · name) = some s) :
    (s = 0 ∧ name = a) ∨ (s = 1 ∧ name = b) ∨ (s = 2 ∧ name = c) := by
  obtain ⟨hlt, hp, -⟩ := List.findIdx?_eq_some_iff_getElem.1 h
  rcases s with _ | _ | _ | s
  · exact .inl ⟨rfl, (Json.str.inj (Trees.strIs hp)).symm⟩
  · exact .inr (.inl ⟨rfl, (Json.str.inj (Trees.strIs hp)).symm⟩)
  · exact .inr (.inr ⟨rfl, (Json.str.inj (Trees.strIs hp)).symm⟩)
  · exact absurd hlt (by simp only [List.length_cons, List.length_nil]; omega)

theorem repStream_ne_session (t : String) (r : ℕ) : repStreamName t r ≠ TREE_SESSION :=
  ne_of_beq_false (repStreamName_ne t r)

theorem rep_zero_ne_one (t t' : String) : repStreamName t 0 ≠ repStreamName t' 1 := fun h =>
  absurd (repStreamName_inj (r := (0 : Fin 2)) (r' := (1 : Fin 2)) h).2 (by decide)

/-- The name `Flock.Zk.coinSpec` gives a table's rep 0 stream is `repStreamName`'s. -/
theorem rep_zero (t : String) : s!"{t}/rep0" = repStreamName t 0 := by
  unfold repStreamName
  rw [String.append_assoc]
  rfl

/-- The name `Flock.Zk.coinSpec` gives a table's rep 1 stream is `repStreamName`'s. -/
theorem rep_one (t : String) : s!"{t}/rep1" = repStreamName t 1 := by
  unfold repStreamName
  rw [String.append_assoc]
  rfl

/-- **A one-table session's coin tree lists the session stream, then the table's two rep streams.** -/
theorem coinSpec_streams (t : String) :
    (Flock.Zk.coinSpec #[t]).streams = #[TREE_SESSION, repStreamName t 0, repStreamName t 1] := by
  rw [← rep_zero, ← rep_one]
  rfl

/-- **A one-table session's rep streams are its table's two**, rep 0 then rep 1. -/
theorem repStreams_table {spec : Flock.SessionSpec} {T : String} (h : spec.tables = #[T]) :
    spec.repStreams.toList = [(0, 0, repStreamName T 0), (0, 1, repStreamName T 1)] := by
  rw [FlockDeploy.repStreams_toList, h]
  rfl

/-- **A spec that is a one-table session's coin tree lists the session stream, then the session's rep streams**: where
the JSON spec `d` is the coin tree of the table `T` and the session `spec` has that table alone, `d`'s streams are
the session stream and then `spec.repStreams`' names, in `spec.repStreams`' order. -/
theorem specStreams_repStreams {d : Json} {spec : Flock.SessionSpec} {T : String}
    (hv : Lines.SpecValue d (Flock.Zk.coinSpec #[T])) (ht : spec.tables = #[T]) :
    specStreams d = #[Json.str TREE_SESSION] ++ spec.repStreams.map fun x => Json.str x.2.2 := by
  unfold specStreams
  rw [hv.streams, coinSpec_streams]
  apply Array.ext'
  simp only [Array.toList_append, Array.toList_map]
  rw [repStreams_table ht]
  rfl

theorem tableNames_one (t : Flock.Tags) : t.tableNames 1 = #[t.table] := by
  unfold Flock.Tags.tableNames
  rfl

/-- **A spec that lists a one-table session's streams**: the session stream, then some table's rep 0 stream, then
its rep 1 stream. -/
def SessionStreams (spec : Json) : Prop :=
  ∃ T, (specStreams spec).toList = [Json.str TREE_SESSION, Json.str (repStreamName T 0), Json.str (repStreamName T 1)]

theorem sessionStreams_of_spec {spec : Json} {T : String} (h : Lines.SpecValue spec (Flock.Zk.coinSpec #[T])) :
    SessionStreams spec := by
  refine ⟨T, ?_⟩
  unfold specStreams
  rw [h.streams, coinSpec_streams]
  first
    | rfl
    | simp

/-! ## The logged tree's spec is the session's -/

/-- **The logged tree a record's `coin_tree` prints as has the spec `checkRecord` held that `coin_tree` to**, where
`coinTreesOf` accepts the release log and the record holds its keys in order (`helloSpec_of_record`'s first step, of
the tree itself). -/
theorem specValue_of_record {lines : List String} {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees)
    {t : LoggedTree} (ht : t ∈ trees) {record : Json} (hj : jeq (getD record "coin_tree") t.record = true)
    (hrs : sortedKeys record = true) {sp : Flock.CoinTree.Spec} {nonce key : String}
    (hC : Flock.CoinTree.checkRecord sp nonce (getD record "coin_tree") = .ok key) : Lines.SpecValue t.spec sp := by
  have hcs := sorted_getD hrs "coin_tree"
  have hts := trees_sorted hT t ht
  obtain ⟨s, hs1, hs2⟩ := checkRecord_spec hC
  exact specValue_of_equiv hts (Device.printed_spec sp).1
    (Eq.trans (Eq.symm (spec_equiv hcs hts hj))
      (Properties.Challenger.CanonLine.canon_inj (sorted_getD hcs "spec") (Device.printed_spec sp).1 hs1 hs2))
    (specValue_toJson sp)

/-- **The logged tree a record's `coin_tree` prints as has the session's coin tree's spec**, where `Record.decode`
accepts the record's text at a session on that coin tree and `record` is the text's value. -/
theorem specValue_of_decode {spec : Flock.SessionSpec} {text : String} {sess : Flock.RunRecord}
    {sp : Flock.CoinTree.Spec} (hd : Flock.Record.decode spec text = .ok sess) (hc : spec.coinTree = some sp)
    {record : Json} (hJ : Flock.JsonParse.parse text = .ok record) {lines : List String}
    {trees : Array LoggedTree} (hT : coinTreesOf lines = .ok trees) {t : LoggedTree} (ht : t ∈ trees)
    (hj : jeq (getD record "coin_tree") t.record = true) : Lines.SpecValue t.spec sp := by
  obtain ⟨j, c, nonce, key, hp, hcj, hC⟩ := decode_coinTree hd hc
  have hjr : j = record := Except.ok.inj (hp.symm.trans hJ)
  rw [hjr] at hcj
  rw [← getD_of_fld hcj] at hC
  exact specValue_of_record hT ht hj (sortedKeys_of_wf record (FlockComplete.JsonRT.parse_wf text record hJ)) hC

end Verity.Sampled

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled (helloSpecs repStreamName)

/-! ## A session's round's place -/

/-- **The place of a session's round's stream** among a one-table session's streams: the session stream's at round
0, the rep 0 stream's at the first `R` rounds after it, the rep 1 stream's past them (`slotStream`). -/
def slotPlace (R j : ℕ) : ℕ := if j = 0 then 0 else if j - 1 < R then 1 else 2

/-- A stream at a place of a one-table session's streams, named as a session's round's stream is, is at that round's
place. -/
theorem slot_place {T tab : String} {R j s : ℕ} {name : String} (hn : (slotStream tab R j).1 = name)
    (h : (s = 0 ∧ name = Spot.Challenger.TREE_SESSION) ∨ (s = 1 ∧ name = repStreamName T 0) ∨
      (s = 2 ∧ name = repStreamName T 1)) :
    s = slotPlace R j := by
  unfold slotStream at hn
  unfold slotPlace
  by_cases h0 : j = 0
  · simp only [h0, ↓reduceIte] at hn ⊢
    rcases h with ⟨hs, -⟩ | ⟨-, he⟩ | ⟨-, he⟩
    · exact hs
    · exact absurd (hn.trans he).symm (Verity.Sampled.repStream_ne_session T 0)
    · exact absurd (hn.trans he).symm (Verity.Sampled.repStream_ne_session T 1)
  · simp only [h0, ↓reduceIte] at hn ⊢
    by_cases hR : j - 1 < R
    · simp only [hR, ↓reduceIte] at hn ⊢
      rcases h with ⟨-, he⟩ | ⟨hs, -⟩ | ⟨-, he⟩
      · exact absurd (hn.trans he) (Verity.Sampled.repStream_ne_session tab 0)
      · exact hs
      · exact absurd (hn.trans he) (Verity.Sampled.rep_zero_ne_one tab T)
    · simp only [hR, ↓reduceIte] at hn ⊢
      rcases h with ⟨-, he⟩ | ⟨-, he⟩ | ⟨hs, -⟩
      · exact absurd (hn.trans he) (Verity.Sampled.repStream_ne_session tab 1)
      · exact absurd (hn.trans he).symm (Verity.Sampled.rep_zero_ne_one T tab)
      · exact hs

/-- **Two of a session's rounds whose streams' places and rounds are in order are in order.** -/
theorem slotPlace_lt {tab tab' : String} {R j j' : ℕ}
    (h : Spot.Challenger.roundBefore (slotPlace R j, (slotStream tab R j).2)
      (slotPlace R j', (slotStream tab' R j').2) = true) : j < j' := by
  unfold Spot.Challenger.roundBefore slotPlace slotStream at h
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  by_cases h0 : j = 0 <;> by_cases h0' : j' = 0 <;> by_cases hR : j - 1 < R <;> by_cases hR' : j' - 1 < R <;>
    simp only [h0, h0', hR, hR', ↓reduceIte] at h <;> omega

/-- **In a one-table session's tree, a session's round's stream is at the round's place.** -/
theorem place_of_streams {spec : Json} (hS : Verity.Sampled.SessionStreams spec) {name : String} {s : ℕ}
    (hk : (Verity.Sampled.specStreams spec).findIdx? (Spot.jsonStrIs · name) = some s) {tab : String} {R j : ℕ}
    (hn : (slotStream tab R j).1 = name) : Spot.Challenger.streamPlace spec name = slotPlace R j := by
  obtain ⟨T, hT⟩ := hS
  rw [Verity.Sampled.streamPlace_of_key hk]
  rw [← Verity.Sampled.findIdx?_toList, hT] at hk
  exact slot_place hn (Verity.Sampled.findIdx_three hk)

/-- **No session's round is its tree's closing round**, the session stream's round 1. -/
theorem slotStream_ne_closing (tab : String) (R j : ℕ) :
    slotStream tab R j ≠ (Spot.Challenger.TREE_SESSION, 1) := by
  unfold slotStream
  by_cases h0 : j = 0
  · rw [ite_eq_left h0]
    intro h
    exact absurd (Prod.mk.inj h).2 (by decide)
  · rw [ite_eq_right h0]
    by_cases hR : j - 1 < R
    · rw [ite_eq_left hR]
      intro h
      exact Verity.Sampled.repStream_ne_session tab 0 (Prod.mk.inj h).1
    · rw [ite_eq_right hR]
      intro h
      exact Verity.Sampled.repStream_ne_session tab 1 (Prod.mk.inj h).1

theorem heldTrees_getElem? {trees : Array Spot.Challenger.LoggedTree} {i : ℕ} {ρ : String} {sp : Json}
    (h : (Verity.Sampled.heldTrees trees)[i]? = some (ρ, sp)) :
    ∃ t, trees[i]? = some t ∧ t.root = ρ ∧ t.spec = sp := by
  have h' : (trees.toList.map fun t => (t.root, t.spec))[i]? = some (ρ, sp) := h
  rw [List.getElem?_map, Array.getElem?_toList] at h'
  cases hti : trees[i]? with
  | none => simp [hti] at h'
  | some t =>
    rw [hti] at h'
    obtain ⟨h1, h2⟩ := Prod.mk.inj (Option.some.inj h')
    exact ⟨t, rfl, h1, h2⟩

/-! ## The located releases of the read rise -/

section Rise

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Release} {trees : Array Spot.Challenger.LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)} {tree : ℕ → ℕ}

/-- **The session-round locator puts the read's releases in order**: where the members' check accepts the sessions'
records against the read, the refusal `Spot.Challenger.repStreamOrder` passes each tree, and each session's tree's
spec lists a one-table session's streams (`Verity.Sampled.SessionStreams`), of two releases of the read the locator
places, the earlier is on the lower round. -/
theorem MembersHeld.rise (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log)
    (hM : Spot.Challenger.membersTreeReason trees lines records = none)
    (hord : ∀ t ∈ trees, Spot.Challenger.repStreamOrder t = true)
    (hS : ∀ (j : ℕ) (x : String × Array (String × Option ℕ) × Json × String), records[j]? = some x →
      ∀ t ∈ trees, Spot.getD (Spot.getD x.2.2.1 "coin_tree") "root" = Json.str t.root →
        Verity.Sampled.SessionStreams t.spec)
    {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String} (hs : ∀ t, off (t + 1) = off t + K t)
    {a b c : List Release} {r r' : Release} (he : log = a ++ r :: (b ++ r' :: c)) {n n' : ℕ}
    (h1 : sessLoc off K M tab a r.frame = some n) (h2 : sessLoc off K M tab (a ++ r :: b) r'.frame = some n') :
    n < n' := by
  have he2 : log = (a ++ r :: b) ++ r' :: c := by rw [he]; simp
  obtain ⟨t₁, j₁, s₁, c₁, ht₁, hf₁, hj₁, hn₁, hname₁, hk₁⟩ := H.key hT hR he h1
  obtain ⟨t₂, j₂, s₂, c₂, ht₂, hf₂, hj₂, hn₂, hname₂, hk₂⟩ := H.key hT hR he2 h2
  have hle : t₁ ≤ t₂ := H.before _ _ _ he2 t₂ ht₂ s₂ c₂ hf₂ r (by simp) t₁ ht₁ s₁ c₁ hf₁
  rcases Nat.lt_or_ge t₁ t₂ with hlt | hge
  · have hmono := off_mono hs (show t₁ + 1 ≤ t₂ from hlt)
    rw [hs] at hmono
    omega
  · have e12 : t₂ = t₁ := Nat.le_antisymm hge hle
    rw [e12] at hf₂ hj₂ hn₂ hname₂ hk₂
    have hF := Verity.Sampled.coinTreesOf_fold recs hT
    obtain ⟨P, l, Q, ρ, name, k, hls, hro, hkey, hnm, hcnt⟩ := Verity.Sampled.readLog_opening hR hF he hf₁
    obtain ⟨P', l', Q', ρ', name', k', hls', hro', hkey', hnm', hcnt'⟩ :=
      Verity.Sampled.readLog_opening hR hF he2 hf₂
    have en : name = (slotStream (tab t₁) (repsOf (K t₁)) j₁).1 := Option.some.inj (hnm.symm.trans hname₁)
    have en' : name' = (slotStream (tab t₁) (repsOf (K t₁)) j₂).1 := Option.some.inj (hnm'.symm.trans hname₂)
    -- the rounds the two lines open are the rounds the session's layout gives the two releases
    have hc1 : openedOf a (tree t₁) s₁ = P.countP (Verity.Sampled.opensLine ρ name) :=
      hcnt (tree t₁) s₁ ρ name hkey
    have hc1' : openedOf (a ++ r :: b) (tree t₁) s₁ = P'.countP (Verity.Sampled.opensLine ρ name) :=
      hcnt' (tree t₁) s₁ ρ name hkey
    have hc2 : openedOf (a ++ r :: b) (tree t₁) s₂ = P'.countP (Verity.Sampled.opensLine ρ' name') :=
      hcnt' (tree t₁) s₂ ρ' name' hkey'
    have ek : k = (slotStream (tab t₁) (repsOf (K t₁)) j₁).2 :=
      (Verity.Sampled.coinTreesOf_round hT hls hro.1).trans (hc1.symm.trans hk₁)
    have ek' : k' = (slotStream (tab t₁) (repsOf (K t₁)) j₂).2 :=
      (Verity.Sampled.coinTreesOf_round hT hls' hro'.1).trans (hc2.symm.trans hk₂)
    -- the earlier release's line comes first
    have hcount : openedOf (a ++ r :: b) (tree t₁) s₁ = openedOf a (tree t₁) s₁ + (openedOf b (tree t₁) s₁ + 1) := by
      unfold openedOf
      rw [List.countP_append, List.countP_cons, ite_eq_left (by simp [hf₁, Verity.Frame.opensAt])]
    have hlt : P.countP (Verity.Sampled.opensLine ρ name) < P'.countP (Verity.Sampled.opensLine ρ name) := by
      omega
    obtain ⟨Mid, hP'⟩ := Verity.Sampled.split_before (hls.symm.trans hls') hlt
    have hlines : lines = P ++ l :: (Mid ++ l' :: Q') := by rw [hls', hP']; simp
    -- both lines open rounds of the session's one tree
    obtain ⟨-, sp, hsp, hss⟩ := hkey
    obtain ⟨-, sp', hsp', hss'⟩ := hkey'
    obtain ⟨eρ, esp⟩ := Prod.mk.inj (Option.some.inj (hsp.symm.trans hsp'))
    obtain ⟨t, hti, htr, hts⟩ := heldTrees_getElem? hsp
    have htm : t ∈ trees := Array.mem_iff_getElem?.2 ⟨_, hti⟩
    have hfi := Verity.Sampled.findIdx?_root (Verity.Sampled.Lines.trees_log hT).nodup hti
    have ho : Verity.Sampled.OpensRound l t.root name k := by rw [htr]; exact hro.1
    have ho' : Verity.Sampled.OpensRound l' t.root name' k' := by rw [htr, eρ]; exact hro'.1
    have hkS : (Verity.Sampled.specStreams t.spec).findIdx? (Spot.jsonStrIs · name) = some s₁ := by
      rw [hts]; exact hss
    have hkS' : (Verity.Sampled.specStreams t.spec).findIdx? (Spot.jsonStrIs · name') = some s₂ := by
      rw [hts, esp]; exact hss'
    -- the tree's last opening is its closing round, which the later release is not
    obtain ⟨x, hx⟩ : ∃ x, records[t₁]? = some x := ⟨_, Array.getElem?_eq_getElem ht₁⟩
    obtain ⟨hall, -, -⟩ := Verity.Sampled.MemberForm.membersTreeReason_none hM
    obtain ⟨t', ht', hroot', o, hback, hcl, -⟩ :=
      Verity.Sampled.MemberForm.memberTreeReason_none (hall x (Array.mem_of_getElem? hx))
    obtain ⟨ρx, hρx, hux⟩ := H.roots t₁ x hx
    have eρx : ρ = ρx := Verity.Sampled.MemberForm.findIdx_fst (H.idx t₁ ρx hux) hsp
    have ett : t' = t := List.inj_on_of_nodup_map (Verity.Sampled.Lines.trees_log hT).nodup
      (Array.mem_toList_iff.2 ht') (Array.mem_toList_iff.2 htm)
      ((Json.str.inj (hroot'.symm.trans hρx)).trans (eρx.symm.trans htr.symm))
    rw [ett] at hroot' hback
    obtain ⟨hos, hor, -, -⟩ := Verity.Sampled.MemberForm.closes_fields hcl
    have hne : (o.stream, o.round) ≠ (name', k') := by
      intro hEq
      obtain ⟨hq1, hq2⟩ := Prod.mk.inj hEq
      exact slotStream_ne_closing (tab t₁) (repsOf (K t₁)) j₂
        (Prod.ext (en'.symm.trans (hq1.symm.trans hos)) (ek'.symm.trans (hq2.symm.trans hor)))
    -- the refusal, at the session's layout
    have hSS := hS t₁ x hx t htm hroot'
    have hrb := Verity.Sampled.lines_rise hT hti hfi (hord t htm) hlines ho ho' hback hne
    rw [place_of_streams hSS hkS en.symm, place_of_streams hSS hkS' en'.symm, ek, ek'] at hrb
    have hjj := slotPlace_lt hrb
    omega

end Rise

/-! ## The whole run's sessions -/

private theorem mapErrorOk {ε α : Type} {x : Except ε α} {f : ε → ε} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

/-- **One statement's step, where the run still accepts every session after it**: the run before it accepted every
session, and each of the statement's sessions was accepted at its member's statement
(`MemberChecks.SessionAccepted`), where the statement `st` the step reads is the member's own at its slice (`hst`).
It is `MemberChecks.stepMember_sessions` with the run before kept. -/
theorem stepMember_all {slice : Spot.OneStageZk.Member → Json}
    {st : Spot.OneStageZk.Member → Flock.V Spot.OneStageZk.StepAt} {bs : Core.Blobs} {r r' : Spot.OneStageZk.Run}
    {z : Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member}
    (hst : ∀ a, st z.2 = .ok a → z.2.statement (slice z.2) bs = .ok a)
    (h : Spot.OneStageZk.stepMember st bs r z = .ok r') (ha : r'.all = true) :
    r.all = true ∧ ∃ circuitB pubB rd, ∀ s ∈ z.1.sessions.toList, MemberChecks.SessionAccepted
      (Spot.OneStageZk.StepAt.ofOwn z.2.own Flock.Tags.circuit circuitB pubB bs rd
        (z.2.population (slice z.2) pubB)) bs s := by
  unfold Spot.OneStageZk.stepMember at h
  have h := mapErrorOk h
  unfold Spot.OneStageZk.runFrom at h
  obtain ⟨a, ha', h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨circuitB, pubB, rd, -, -, -, -, rfl⟩ := MemberChecks.statement_ok (hst a ha')
  rw [← Array.foldlM_toList] at h
  obtain ⟨har, hs⟩ := MemberChecks.foldl_all _ r r' h ha
  exact ⟨har, circuitB, pubB, rd, hs⟩

/-- **The statements, run in order, accepting every session**: the run before them accepted every session, and each
statement's sessions were accepted at its member's statement, where the statement `st` the steps read is each
member's own at its slice (`hst`). -/
theorem foldl_members {slice : Spot.OneStageZk.Member → Json}
    {st : Spot.OneStageZk.Member → Flock.V Spot.OneStageZk.StepAt} {bs : Core.Blobs} :
    ∀ (l : List (Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member)) (r r' : Spot.OneStageZk.Run),
      (∀ z ∈ l, ∀ a, st z.2 = .ok a → z.2.statement (slice z.2) bs = .ok a) →
      l.foldlM (Spot.OneStageZk.stepMember st bs) r = .ok r' → r'.all = true →
        r.all = true ∧ ∀ z ∈ l, ∃ circuitB pubB rd, ∀ s ∈ z.1.sessions.toList, MemberChecks.SessionAccepted
          (Spot.OneStageZk.StepAt.ofOwn z.2.own Flock.Tags.circuit circuitB pubB bs rd
            (z.2.population (slice z.2) pubB)) bs s
  | [], r, r', _, h, ha => by
    obtain rfl : r = r' := Except.ok.inj h
    exact ⟨ha, fun _ hz => by simp at hz⟩
  | z :: l, r, r', hst, h, ha => by
    rw [List.foldlM_cons] at h
    obtain ⟨r₁, h₁, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
    obtain ⟨ha₁, hl⟩ := foldl_members l r₁ r' (fun z' hz' => hst z' (List.mem_cons_of_mem _ hz')) h ha
    obtain ⟨har, hz⟩ := stepMember_all (hst z (by simp)) h₁ ha₁
    refine ⟨har, fun z' hz' => ?_⟩
    rcases List.mem_cons.1 hz' with rfl | hz'
    · exact hz
    · exact hl z' hz'

/-- **The per-member form's run, accepting every session**: where `runMembers` returns a run that accepts every
session, the statements' structure held, and each statement's each session was accepted at its member's statement
(`MemberChecks.SessionAccepted`), the statements paired with their members in order. -/
theorem runMembers_sessions {slice : Spot.OneStageZk.Member → Json} {ms : Array Spot.OneStageZk.Member}
    {sts : Array Spot.OneStageZk.StatementFiles} {want : Array ℕ} {bs : Core.Blobs} {r : Spot.OneStageZk.Run}
    (h : Spot.OneStageZk.runMembers slice ms sts want bs = .ok r) (ha : r.all = true) :
    ∃ held, Spot.OneStageZk.membersHeld ms sts = .ok held ∧ ∀ z ∈ (sts.zip held).toList, ∃ circuitB pubB rd,
      ∀ s ∈ z.1.sessions.toList, MemberChecks.SessionAccepted
        (Spot.OneStageZk.StepAt.ofOwn z.2.own Flock.Tags.circuit circuitB pubB bs rd
          (z.2.population (slice z.2) pubB)) bs s := by
  unfold Spot.OneStageZk.runMembers at h
  obtain ⟨held, hheld, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨files, hfiles, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨r₁, hfold, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨_, -, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  have e : r₁ = r := Except.ok.inj h
  rw [e] at hfold
  rw [← Array.foldlM_toList] at hfold
  have hst : ∀ z ∈ (sts.zip held).toList, ∀ a, Spot.OneStageZk.statementAt files z.2 = .ok a →
      z.2.statement (slice z.2) bs = .ok a := by
    rintro ⟨sf, m⟩ hz a hza
    rw [Array.toList_zip] at hz
    have hm : m ∈ held := Array.mem_def.2 (List.of_mem_zip hz).2
    exact MemberChecks.statementAt_statement hfiles (MemberChecks.membersHeld_find hheld m hm) hza
  exact ⟨held, hheld,
    (foldl_members (slice := slice) (st := Spot.OneStageZk.statementAt files) _ _ _ hst hfold ha).2⟩

/-- **What an accepted `membersRun` ran**: the statements' members, the challenger's trees, the order check on the
release log, the log's lines, `membersTreesHeld` on the statements paired with their members, and `runMembers` at
each member's slice of the registration (`membersRun_held`, with the order check and the run). -/
theorem membersRun_ran {p : Spot.OneStageZk.Params} {ms : Array Spot.OneStageZk.Member}
    {c : Spot.OneStage.Commitment} {e : Spot.OneStageZk.Evidence} {sts : Array Spot.OneStageZk.StatementFiles}
    {bs : Core.Blobs} {r : Spot.OneStageZk.Run} (h : Spot.OneStageZk.membersRun p ms c e sts bs = .ok r) :
    ∃ held ts L roots, Spot.OneStageZk.membersHeld ms sts = .ok held ∧
      Spot.OneStageZk.treesOf p.base e.base = .ok ts ∧ Spot.OneStageZk.orderHeld p c e = .ok () ∧
      Spot.OneStageZk.linesOf e = .ok L ∧ Spot.OneStageZk.membersTreesHeld ts L (sts.zip held) bs = .ok () ∧
      Spot.OneStageZk.runMembers (fun m => m.slice c.record roots) ms sts e.base.provedUnits bs = .ok r := by
  unfold Spot.OneStageZk.membersRun at h
  simp only [Verity.Sampled.bind_ok] at h
  obtain ⟨_, hs, roots, -, hrm⟩ := h
  unfold Spot.OneStageZk.membersStructure at hs
  simp only [Verity.Sampled.bind_ok] at hs
  obtain ⟨held, hheld, hs⟩ := hs
  split at hs
  · simp only [Verity.Sampled.bind_ok] at hs
    obtain ⟨_, hthrow, -⟩ := hs
    cases hthrow
  · simp only [Verity.Sampled.bind_ok] at hs
    obtain ⟨ts, hts, _, hO, L, hL, _, hM, -⟩ := hs
    exact ⟨held, ts, L, roots, hheld, hts, hO, hL, hM, hrm⟩

/-! ## A member's session's tree lists its streams -/

/-- **The session `--zk` sets up at one table is on that table's coin tree**: where `setupAt` sets a session up at
inputs of one table and hidden outputs, the session's coin tree is `Flock.Zk.coinSpec` at the statement's table. -/
theorem setupAt_coinTree {W : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {dr : Option Json}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (hh : (Flock.Witness.zkTags W).hiddenOutputs = true) (hc : W.count = 1)
    (h : Spot.OneStageZk.setupAt W rd dr = .ok (spec, tabs, x)) :
    spec.coinTree = some (Flock.Zk.coinSpec #[W.tags.table]) := by
  obtain ⟨sts, hs, -⟩ := Verity.Sampled.setupAt_ok h
  obtain ⟨hset, -⟩ := Verity.Sampled.sessionAt_ok hs
  have hn : W.tags.tableNames W.count = #[W.tags.table] := by
    rw [hc]
    exact Verity.Sampled.tableNames_one _
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hset with ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · rw [← hn]
    exact (Verity.Sampled.setupHidden_coinTreeP hst).trans rfl
  · exact absurd h1 (by omega)

/-- **The session `--zk` sets up at one table has that table alone.** -/
theorem setupAt_oneTable {W : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {dr : Option Json}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (hh : (Flock.Witness.zkTags W).hiddenOutputs = true) (hc : W.count = 1)
    (h : Spot.OneStageZk.setupAt W rd dr = .ok (spec, tabs, x)) : spec.tables = #[W.tags.table] := by
  obtain ⟨sts, hs, -⟩ := Verity.Sampled.setupAt_ok h
  obtain ⟨hset, -⟩ := Verity.Sampled.sessionAt_ok hs
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hset with ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · obtain ⟨_, hP⟩ := FlockSoundness.Discharge.Hidden.setupHidden_specP hst
    have e : st.tags.table = W.tags.table := (congrArg Flock.Tags.table hP.tagsEq).trans rfl
    exact congrArg (fun t : String => #[t]) e
  · exact absurd h1 (by omega)

/-- **What a member's accepted session's verdict fixes of its tree**: where the per-member form accepted the session
`s` at its member's statement, `recJ` is its record on the blobs, and `sessionTreeReason` holds that record to the
logged tree `t` less its last opening, the session the verdict verified (`setupAt`'s, the one `Flock.Zk.verify`
accepted the record at) has one table `T` alone, and `t`'s spec is that table's coin tree. -/
theorem accepted_spec {own : Spot.OneStageZk.Own} {circuitB pubB : ByteArray} {bs : Core.Blobs}
    {rd : Spot.OneStageZk.Reads} {pop : Flock.V Unit} {s : Spot.OneStageZk.RunFiles}
    (hacc : MemberChecks.SessionAccepted
      (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop) bs s)
    {recJ : Json} (hrec : Verity.Sampled.recordOn bs s = some recJ) {lines : List String}
    {trees : Array Spot.Challenger.LoggedTree} (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    {t : Spot.Challenger.LoggedTree} (ht : t ∈ trees) {tables : Array (String × Option ℕ)}
    (hS : Spot.Challenger.sessionTreeReason #[t.unclosed] tables recJ = none) :
    ∃ text dr proofs spec tabs rest T,
      Spot.OneStageZk.setupAt (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I
          (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).rd dr = .ok (spec, tabs, rest) ∧
        Flock.Zk.verify spec tabs text proofs = .ok () ∧ spec.tables = #[T] ∧
        Verity.Sampled.Lines.SpecValue t.spec (Flock.Zk.coinSpec #[T]) := by
  obtain ⟨coinKeys, b, text, hg, hu, hv⟩ := hacc
  obtain ⟨proofs, spec, tabs, rest, -, hst, hzk⟩ := Verity.Sampled.verdictOf_ok hv
  obtain ⟨sess, hd⟩ := Verity.Sampled.zkVerify_decode hzk
  have htag : Flock.Tags.find? Spot.OneStageZk.STATEMENT = some Flock.Tags.circuit := rfl
  have hh : (Flock.Witness.zkTags
      (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I).hiddenOutputs = true :=
    (Verity.Sampled.statement_zkTags htag _ _).1
  have hc := setupAt_coinTree (W := (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I)
    hh rfl hst
  have htab := setupAt_oneTable (W := (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I)
    hh rfl hst
  obtain ⟨j, cj, nonce, key, hp, -, -⟩ := Verity.Sampled.decode_coinTree hd hc
  have hrec' : Verity.Sampled.recordOn bs s = some ((Flock.JsonParse.parse text).toOption.getD Json.null) := by
    simp [Verity.Sampled.recordOn, hg, hu, Except.toOption]
  have hj : recJ = j := by
    have e := Option.some.inj (hrec.symm.trans hrec')
    rw [hp] at e
    exact e
  rw [← hj] at hp
  obtain ⟨t', ht', -, hjq⟩ := Verity.Sampled.Trees.coinTree_held hS
  have et' : t' = t.unclosed := Array.mem_singleton.1 ht'
  rw [et'] at hjq
  have hjq' : Spot.jeq (Spot.getD recJ "coin_tree") t.record = true := hjq
  exact ⟨text, _, proofs, spec, tabs, rest, _, hst, hzk, htab,
    Verity.Sampled.specValue_of_decode hd hc hp hT ht hjq'⟩

/-- **A member's accepted session's tree lists a one-table session's streams** (`accepted_spec`): `t`'s spec lists
the session stream and the statement's table's two rep streams. -/
theorem sessionStreams_of_accepted {own : Spot.OneStageZk.Own} {circuitB pubB : ByteArray} {bs : Core.Blobs}
    {rd : Spot.OneStageZk.Reads} {pop : Flock.V Unit} {s : Spot.OneStageZk.RunFiles}
    (hacc : MemberChecks.SessionAccepted
      (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop) bs s)
    {recJ : Json} (hrec : Verity.Sampled.recordOn bs s = some recJ) {lines : List String}
    {trees : Array Spot.Challenger.LoggedTree} (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    {t : Spot.Challenger.LoggedTree} (ht : t ∈ trees) {tables : Array (String × Option ℕ)}
    (hS : Spot.Challenger.sessionTreeReason #[t.unclosed] tables recJ = none) :
    Verity.Sampled.SessionStreams t.spec := by
  obtain ⟨-, -, -, -, -, -, T, -, -, -, hv⟩ := accepted_spec hacc hrec hT ht hS
  exact Verity.Sampled.sessionStreams_of_spec hv

/-- **A member's session's verdict ties its tree's streams to the session's rep streams**: where the per-member form
accepted the session `s` at its member's statement, `recJ` is its record on the blobs, and `sessionTreeReason` holds
that record to the logged tree `t` less its last opening, `t`'s spec lists the session stream and then the rep
streams of the session the verdict verified (`Flock.SessionSpec.repStreams` of `setupAt`'s session, the one
`Flock.Zk.verify` accepted the record at), in that session's order. So the order the form's check holds the tree's
rounds to, the tree's spec's, is the session's own order of its rep streams. -/
theorem treeStreams_repStreams {own : Spot.OneStageZk.Own} {circuitB pubB : ByteArray} {bs : Core.Blobs}
    {rd : Spot.OneStageZk.Reads} {pop : Flock.V Unit} {s : Spot.OneStageZk.RunFiles}
    (hacc : MemberChecks.SessionAccepted
      (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop) bs s)
    {recJ : Json} (hrec : Verity.Sampled.recordOn bs s = some recJ) {lines : List String}
    {trees : Array Spot.Challenger.LoggedTree} (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    {t : Spot.Challenger.LoggedTree} (ht : t ∈ trees) {tables : Array (String × Option ℕ)}
    (hS : Spot.Challenger.sessionTreeReason #[t.unclosed] tables recJ = none) :
    ∃ text dr proofs spec tabs rest,
      Spot.OneStageZk.setupAt (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).I
          (Spot.OneStageZk.StepAt.ofOwn own Flock.Tags.circuit circuitB pubB bs rd pop).rd dr = .ok (spec, tabs, rest) ∧
        Flock.Zk.verify spec tabs text proofs = .ok () ∧
        Verity.Sampled.specStreams t.spec =
          #[Json.str Spot.Challenger.TREE_SESSION] ++ spec.repStreams.map fun x => Json.str x.2.2 := by
  obtain ⟨text, dr, proofs, spec, tabs, rest, T, hst, hzk, htab, hv⟩ := accepted_spec hacc hrec hT ht hS
  exact ⟨text, dr, proofs, spec, tabs, rest, hst, hzk, Verity.Sampled.specStreams_repStreams hv htab⟩

/-! ## The piece at the form's acceptance -/

theorem forall₂_getElem?_right {α β : Type} {R : α → β → Prop} {l₁ : List α} {l₂ : List β}
    (h : List.Forall₂ R l₁ l₂) {i : ℕ} {b : β} (hb : l₂[i]? = some b) : ∃ a, l₁[i]? = some a ∧ R a b := by
  have hi : i < l₁.length := by
    rw [h.length_eq]
    exact (List.getElem?_eq_some_iff.1 hb).1
  obtain ⟨b', hb', hR⟩ := forall₂_getElem? h (List.getElem?_eq_getElem hi)
  have e : b' = b := Option.some.inj (hb'.symm.trans hb)
  rw [e] at hR
  exact ⟨_, List.getElem?_eq_getElem hi, hR⟩

/-- **The located releases of the log the check read come in the order of their rounds**: on a play the form accepts,
of two releases of the interior challenger session's log as the check read it that the session-round locator places,
the earlier in the log is on the lower round. -/
def LocatedRising (A : Live₂) (C : CaseOf) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ a r b r' c, l = a ++ r :: (b ++ r' :: c) →
    ∀ n n', locOf A tab a r.frame = some n → locOf A tab (a ++ r :: b) r'.frame = some n' → n < n'

-- headroom: one proof unpacks the form's steps, the members' check and the run, as `lines_read` does
set_option maxHeartbeats 1000000 in
/-- **`LocatedRising`, at the form's acceptance**: the accepted check ran `membersRun` on the interior's audit
(`accepted_steps`, `inputAt_ok`), whose members' check held the sessions' records to the trees of the lines the read
reads (`membersRun_ran`, `named_exact`), whose run accepted every session (`runMembers_sessions`), so each session's
tree lists its streams (`sessionStreams_of_accepted`); and the form refused a tree whose rounds are not opened stream
by stream (`accepted_repOrdered`). So `MembersHeld.rise` applies. -/
theorem located_rising (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) : LocatedRising A C tab := by
  intro o hacc l hl a r b r' c hsplit n n' h1 h2
  obtain ⟨k, p, e, s, c₂, d, ξ₂, recs, lines, hk, hp, he, hs, hc₂, hd, hξ, hrecs, hlines, hR⟩ := interiorRead_some hl
  obtain ⟨k', hk', p', c', e', pub, prof, ⟨hp', -, he'⟩, hck⟩ := hacc
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain rfl : p = p' := Except.ok.inj (hp.symm.trans hp')
  obtain rfl : e = e' := Except.ok.inj (he.symm.trans he')
  obtain ⟨-, c₂', ξ₂', run, i, rn, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hck
  obtain ⟨⟨s', hs', hc₂'⟩, hξ', hrun, hall, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  obtain rfl : s = s' := Except.ok.inj (hs.symm.trans hs')
  obtain rfl : c₂ = c₂' := Except.ok.inj (hc₂.symm.trans hc₂')
  obtain rfl : ξ₂ = ξ₂' := Except.ok.inj (hξ.symm.trans hξ')
  obtain ⟨held, ts, L, roots, hheld, hts, -, hL, hM, hrm⟩ := membersRun_ran hrun
  rw [Verity.Sampled.linesOf_check, hlines] at hL
  obtain rfl : lines = L := Except.ok.inj hL
  obtain ⟨named, hnone, hN₀⟩ := named_exact hM
  have hN := hN₀
  rw [zip_sessions (hsz _ _ _ hheld)] at hN
  have hrecs' := named_records hN
  rw [hrecs] at hrecs'
  obtain rfl : recs = named.toList.map (·.2.2.1) := Option.some.inj hrecs'
  obtain ⟨key, w, htt⟩ := Verity.Sampled.treesOf_ok hts
  obtain ⟨ls, lines', hlog, hls, hct⟩ := Verity.Sampled.trustedTreesOf_ok htt
  have hl' := Verity.Sampled.linesOf_of hlog hls
  rw [hlines] at hl'
  obtain rfl : lines = lines' := Except.ok.inj hl'
  obtain ⟨ts', hts', hord⟩ := VStar.Recursive.SampledTwo.repStreamsOrdered_trees
    (VStar.Recursive.SampledTwo.accepted_repOrdered hck)
  have ets : ts' = ts := Except.ok.inj (hts'.symm.trans hts)
  rw [ets] at hord
  obtain ⟨held', hheld', hmem⟩ := runMembers_sessions hrm hall
  have eheld : held' = held := Except.ok.inj (hheld'.symm.trans hheld)
  rw [eheld] at hmem
  obtain ⟨tree, H⟩ := membersHeld_of hct hR hnone
  have hS : ∀ (j : ℕ) (x : String × Array (String × Option ℕ) × Json × String), named[j]? = some x →
      ∀ t ∈ ts, Spot.getD (Spot.getD x.2.2.1 "coin_tree") "root" = Json.str t.root →
        Verity.Sampled.SessionStreams t.spec := by
    intro j x hx t ht hroot
    have hx' : named.toList[j]? = some x := by rw [Array.getElem?_toList]; exact hx
    obtain ⟨sn, hsj, hnamed, -⟩ := forall₂_getElem?_right hN₀ hx'
    obtain ⟨z, hz, hsz'⟩ := List.mem_flatMap.1 (List.mem_of_getElem? hsj)
    obtain ⟨circuitB, pubB, rd, hacc'⟩ := hmem z hz
    obtain ⟨hallr, -, -⟩ := Verity.Sampled.MemberForm.membersTreeReason_none hnone
    obtain ⟨t', ht', hroot', o', -, -, hSt⟩ :=
      Verity.Sampled.MemberForm.memberTreeReason_none (hallr x (Array.mem_of_getElem? hx))
    have ett : t' = t := List.inj_on_of_nodup_map (Verity.Sampled.Lines.trees_log hct).nodup
      (Array.mem_toList_iff.2 ht') (Array.mem_toList_iff.2 ht) (Json.str.inj (hroot'.symm.trans hroot))
    rw [ett] at hSt
    exact sessionStreams_of_accepted (hacc' sn hsz') hnamed hct ht hSt
  unfold locOf at h1 h2
  exact H.rise hct hR hnone hord hS (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h) (SpotFlock.off_succ A.h)
    hsplit h1 h2

/-! ## The sample, then the link -/

/-- A source the parameters name is of their challenger's family, after the registration's digest. -/
theorem source_family {p : Spot.OneStageZk.Params} {c : Spot.OneStage.Commitment}
    {d} (h : Properties.Challenger.oneStageZkSource p c = some d) :
    ∃ key, p.base.challenger = some (key, d.label) ∧ d.after = c.digest := by
  unfold Properties.Challenger.oneStageZkSource Properties.Challenger.oneStageSource at h
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h
  all_goals first
    | (have hd := Option.some.inj h
       subst hd
       first
         | exact ⟨_, hch, rfl⟩
         | exact ⟨_, rfl, rfl⟩)
    | cases h

/-- **The log the check read holds the second stage's sample before every opening, and the link's opening first**:
on a play the form accepts, the interior challenger session's log as the check read it is in one-stage v2's order
(`Verity.Sampled.SampleLinkFirst`) at the source the read reads under: no release of the sample follows an opening
of a coin tree's round, and the first round opened is of the session stream. -/
def ReadLinkFirst (C : CaseOf) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l →
    ∃ (k : Case) (p : VStar.Recursive.SampledTwo.Params) (e : VStar.Recursive.SampledTwo.Evidence) (s : String)
      (c₂ : Spot.OneStage.Commitment), ∃ d,
      C o = some k ∧ (Core.canonicalJson k.params >>= VStar.Recursive.SampledTwo.paramsOfJson) = .ok p ∧
        VStar.Recursive.SampledTwo.evidenceOf k.evidence = .ok e ∧ Flock.canon e.record = .ok s ∧
        Spot.OneStage.commitmentOf s.toUTF8 = .ok c₂ ∧
        Properties.Challenger.oneStageZkSource p.audit c₂ = some d ∧
        Verity.Sampled.SampleLinkFirst d.label d.after l

/-- **`ReadLinkFirst`, at the form's acceptance**: the accepted check ran `membersRun` on the interior's audit, whose
order check passed on the lines the read reads (`membersRun_ran`, `Verity.Sampled.orderHeld_ok`), at the family and
the digest the read's source names (`source_family`); the read keeps that order
(`Verity.Sampled.sampleLinkFirst_readLog`). -/
theorem readLinkFirst (C : CaseOf) : ReadLinkFirst C := by
  intro o hacc l hl
  obtain ⟨k, p, e, s, c₂, d, ξ₂, recs, lines, hk, hp, he, hs, hc₂, hd, hξ, hrecs, hlines, hR⟩ := interiorRead_some hl
  obtain ⟨k', hk', p', c', e', pub, prof, ⟨hp', -, he'⟩, hck⟩ := hacc
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain rfl : p = p' := Except.ok.inj (hp.symm.trans hp')
  obtain rfl : e = e' := Except.ok.inj (he.symm.trans he')
  obtain ⟨-, c₂', ξ₂', run, i, rn, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hck
  obtain ⟨⟨s', hs', hc₂'⟩, -, hrun, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  obtain rfl : s = s' := Except.ok.inj (hs.symm.trans hs')
  obtain rfl : c₂ = c₂' := Except.ok.inj (hc₂.symm.trans hc₂')
  obtain ⟨held, ts, L, roots, -, -, hO, -, -, -⟩ := membersRun_ran hrun
  obtain ⟨key, hch, haft⟩ := source_family hd
  have hB := Verity.Sampled.orderHeld_ok hO hch hlines
  rw [← haft] at hB
  exact ⟨k, p, e, s, c₂, d, hk, hp, he, hs, hc₂, hd, Verity.Sampled.sampleLinkFirst_readLog hB hR⟩

end Verity.SampledTwo.SessionCoins
