module

public import Proofs.Properties.Verity.SampledTwo.SessionHeld
public import Proofs.Properties.Verity.Sampled.RoundsRead

/-!
# Every round a session's record gives coins of is opened on that session's round

`SessionCoins.located_own` says a release the session-round locator puts on a session's round is that session's
record's coins for it. This module is the other way: on a play the two-stage form accepts, each round of a second-stage
session whose record gives coins has its opening in the log the check read, on that session's round.

* `roundOf_slotStream`: the stream and round `slotStream` gives a session's round are read back as that round
  (`roundOf`), at the link's round and the two reps' (`j ≤ 2 * R`): no opening is read as a round past them.
* `mem_located_of`: a release the locator puts on a round is among the round's releases.
* `opening_of_unclosed`: a round a tree shows opened without its closing round, it shows opened.
* `MembersHeld.opened`: where the members' check accepts the sessions' records against the read, a round session
  `t`'s record gives coins of (`Verity.Sampled.recordRound`), the link's or a rep's, is one its tree shows opened
  (`Verity.Sampled.Trees.round_held`), so the read opens that stream of that tree more often than the round's number
  (`Verity.Sampled.opening_lt_openedOf`), and the opening after exactly that many is the one the locator puts on
  session `t`'s round: the tree is the `t`-th first opened (`MembersHeld.place`) and the stream is named as the
  record names it in the `Hello`s before it (`Verity.Sampled.readLog_named`).
* `located_opened`: on a play the form accepts, the releases on round `j` of session `t` in the log the check read,
  `j` the link's round or a rep's, are exactly one opening, with the coins session `t`'s record gives the round.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled (helloSpecs repStreamName)

/-- **A session's round's stream and round read back as the round**, at the link's round and the two reps'. -/
theorem roundOf_slotStream (tab : String) (R j : ℕ) (hj : j ≤ 2 * R) :
    roundOf tab R (some (slotStream tab R j).1) (slotStream tab R j).2 = some j := by
  by_cases h0 : j = 0
  · subst h0
    simp [roundOf, slotStream]
  · by_cases hR : j - 1 < R
    · have hs : slotStream tab R j = (repStreamName tab 0, j - 1) := by
        unfold slotStream
        rw [if_neg h0, if_pos hR]
      rw [hs]
      have hne : ¬ (some (repStreamName tab 0) = some Spot.Challenger.TREE_SESSION ∧ j - 1 = 0) := by
        rintro ⟨h, -⟩
        have hb := Verity.Sampled.repStreamName_ne tab 0
        rw [Option.some.inj h] at hb
        simp at hb
      unfold roundOf
      rw [if_neg hne, if_pos ⟨rfl, hR⟩]
      congr 1
      omega
    · have hs : slotStream tab R j = (repStreamName tab 1, j - 1 - R) := by
        unfold slotStream
        rw [if_neg h0, if_neg hR]
      rw [hs]
      have hne : ¬ (some (repStreamName tab 1) = some Spot.Challenger.TREE_SESSION ∧ j - 1 - R = 0) := by
        rintro ⟨h, -⟩
        have hb := Verity.Sampled.repStreamName_ne tab 1
        rw [Option.some.inj h] at hb
        simp at hb
      have hne' : ¬ (some (repStreamName tab 1) = some (repStreamName tab 0) ∧ j - 1 - R < R) := by
        rintro ⟨e, -⟩
        have h2 := (Verity.Sampled.repStreamName_inj (t := tab) (t' := tab) (r := 1) (r' := 0)
          (Option.some.inj e)).2
        exact absurd h2 (by decide)
      unfold roundOf
      rw [if_neg hne, if_neg hne', if_pos ⟨rfl, by omega⟩]
      congr 1
      omega

/-- **A release the locator puts on a round is among the round's releases.** -/
theorem mem_located_of {loc : List Release → Verity.Frame → Option ℕ} {n : ℕ} {a b : List Release} {r : Release}
    (h : loc a r.frame = some n) : (r.frame, r.coins) ∈ located loc n [] (a ++ r :: b) := by
  rw [located_append, List.nil_append]
  refine List.mem_append_right _ ?_
  simp only [located, if_pos h]
  exact List.mem_append_left _ (List.mem_singleton.2 rfl)

/-- A round's releases with one among them and at most one are that one. -/
theorem eq_singleton {α : Type} {l : List α} {x : α} (hx : x ∈ l) (h1 : l.length ≤ 1) : l = [x] := by
  cases l with
  | nil => cases hx
  | cons y rest =>
    cases rest with
    | nil =>
      rcases List.mem_singleton.1 hx with rfl
      rfl
    | cons z rest =>
      simp only [List.length_cons] at h1
      omega

/-- **A round a tree shows opened without its closing round, it shows opened.** -/
theorem opening_of_unclosed {u : Spot.Challenger.LoggedTree} {name : String} {k : ℕ}
    {o : Spot.Challenger.TreeOpening} (h : u.unclosed.opening name k = some o) :
    ∃ o', u.opening name k = some o' ∧ o' ∈ u.opened ∧ o'.stream = name ∧ o'.round = k := by
  have h' : u.opened.pop.find? (fun o => o.stream == name && o.round == k) = some o := h
  have hmem : o ∈ u.opened.pop := Array.mem_of_find?_eq_some h'
  have hp := Array.find?_some h'
  have hmem' : o ∈ u.opened := by
    have h1 : o ∈ u.opened.pop.toList := Array.mem_def.1 hmem
    rw [Array.toList_pop] at h1
    exact Array.mem_def.2 (List.mem_of_mem_dropLast h1)
  have hsome : (u.opened.find? fun o => o.stream == name && o.round == k).isSome = true :=
    Array.find?_isSome.2 ⟨o, hmem', hp⟩
  obtain ⟨o', ho'⟩ := Option.isSome_iff_exists.1 hsome
  have hp' := Array.find?_some ho'
  simp only [Bool.and_eq_true, beq_iff_eq] at hp'
  exact ⟨o', ho', Array.mem_of_find?_eq_some ho', hp'.1, hp'.2⟩

section Opened

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Release} {trees : Array Spot.Challenger.LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)} {tree : ℕ → ℕ}

/-- **Every round a session's record gives coins of is opened in the read, on that session's round**: where the
members' check accepts the sessions' records against the read, and session `t`'s record gives coins of the stream and
round its round `j` is (`slotStream`), `j` the link's round or a rep's (`j ≤ 2 * repsOf (K t)`), the read has a
release the session-round locator puts on session `t`'s round `j`. -/
theorem MembersHeld.opened (H : MembersHeld lines trees log records tree)
    (hT : Spot.Challenger.coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log)
    (hM : Spot.Challenger.membersTreeReason trees lines records = none) {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String}
    {t j : ℕ} {x : String × Array (String × Option ℕ) × Json × String} (hx : records[t]? = some x) (htM : t < M)
    (hj : j < K t) (hjR : j ≤ 2 * repsOf (K t)) {cs : ByteArray}
    (hcs : Verity.Sampled.recordRound x.2.2.1 (slotStream (tab t) (repsOf (K t)) j).1
      (slotStream (tab t) (repsOf (K t)) j).2 = some cs) :
    ∃ pre r post, log = pre ++ r :: post ∧ sessLoc off K M tab pre r.frame = some (off t + j) := by
  have ht : t < records.size := (Array.getElem?_eq_some_iff.1 hx).1
  obtain ⟨tree', h1, -, -⟩ := Verity.Sampled.MemberForm.members_read hT hR hM
  obtain ⟨u, hu, hroot, -, hS, -⟩ := h1 t x hx
  obtain ⟨u', hu', -, o, ho, -⟩ := Verity.Sampled.Trees.round_held hS hcs
  obtain rfl : u' = u.unclosed := Array.mem_singleton.1 hu'
  obtain ⟨o', ho', hmem, hstream, hround⟩ := opening_of_unclosed ho
  obtain ⟨l, hl, htl, d, hev, hTE, htree, hstr, -, -⟩ := Verity.Sampled.Trees.opened_in hT u hu o' hmem
  have hF := Verity.Sampled.coinTreesOf_fold recs hT
  obtain ⟨i, si, hkey⟩ := Verity.Sampled.treeFold_keyAt lines [] _ hF l hl htl d hev hTE htree hstr
  rw [hstream] at hkey
  have hlt := Verity.Sampled.opening_lt_openedOf hT hR hu ho' hkey
  obtain ⟨ρ, hρ, hused⟩ := H.roots t x hx
  obtain rfl : ρ = u.root := Json.str.inj (hρ.symm.trans hroot)
  obtain rfl : tree t = i :=
    Option.some.inj ((H.idx t _ hused).symm.trans (Verity.Sampled.MemberForm.keyAt_idx hkey))
  have hlt' : (slotStream (tab t) (repsOf (K t)) j).2 <
      log.countP (fun r : Release => r.frame.opensAt (tree t) si) := hlt
  obtain ⟨pre, r, post, he, hq, hcount⟩ := Verity.Sampled.MemberForm.split_at_count log _ hlt'
  obtain ⟨n, hfr⟩ := Verity.Sampled.MemberForm.opening_of_opensAt hq
  have hcount' : openedOf pre (tree t) si = (slotStream (tab t) (repsOf (K t)) j).2 := hcount
  have hplace := H.place hT hR he ht hfr
  obtain ⟨l₂, ρ₂, name₂, k₂, -, -, -, hname₂, -⟩ := Verity.Sampled.readLog_round hT hR he hfr
  obtain ⟨sp, hsp, -⟩ := Option.bind_eq_some_iff.1 hname₂
  have hnamed := Verity.Sampled.readLog_named hR hF hkey
  have hsp' : (helloSpecs log)[tree t]? = some sp := by
    rw [he, Verity.Sampled.helloSpecs_append, List.getElem?_append_left (List.getElem?_eq_some_iff.1 hsp).1, hsp]
  rw [hsp', Option.bind_some] at hnamed
  have hname : ((helloSpecs pre)[tree t]?.bind fun sp => sp.streams[si]?) =
      some (slotStream (tab t) (repsOf (K t)) j).1 := by
    rw [hsp, Option.bind_some]
    exact hnamed
  refine ⟨pre, r, post, he, ?_⟩
  rw [hfr]
  simp only [sessLoc]
  rw [hplace, hname, hcount', roundOf_slotStream _ _ _ hjR]
  simp only [Option.bind_some, htM, hj, and_self, ↓reduceIte]

end Opened

/-- **The releases on a session's round are its record's round's opening**: on a play the form accepts, where
session `t`'s record gives coins of the stream and round its round `j` is, `j` the link's round or a rep's, the
releases the session-round locator puts on that round in the log the check read are exactly one opening, with those
coins' bits. -/
theorem located_opened (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) {o : Exchange}
    (hacc : FormAccepts C o) {l : List Release} (hl : interiorRead C o = some l) {recs : List Json}
    (hrecs : sessionRecords C o = some recs) {t j : ℕ} (ht : t < A.h.ms.length) (hj : j < SpotFlock.k₀ A.h t)
    (hjR : j ≤ 2 * repsOf (SpotFlock.k₀ A.h t)) {rec : Json} (hrec : recs[t]? = some rec) {cs : ByteArray}
    (hcs : Verity.Sampled.recordRound rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
      (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2 = some cs) :
    ∃ i s n, located (locOf A tab) (SpotFlock.off A.h t + j) [] l =
      [(.opening i s n, Verity.bitsOfBytes cs)] := by
  obtain ⟨label, after, ξ, lines, trees, records, hT, hR, hMR, hS⟩ :=
    membersAccepted (membersNamed C hsz) o hacc l hl
  obtain rfl : records.toList.map (·.2.2.1) = recs := Option.some.inj (hS.symm.trans hrecs)
  obtain ⟨x, hx, rfl⟩ : ∃ x : String × Array (String × Option ℕ) × Json × String,
      records[t]? = some x ∧ x.2.2.1 = rec := by
    rw [List.getElem?_map, Array.getElem?_toList] at hrec
    cases hy : records[t]? with
    | none => rw [hy] at hrec; simp at hrec
    | some y => rw [hy] at hrec; exact ⟨y, rfl, Option.some.inj hrec⟩
  obtain ⟨tree, H⟩ := membersHeld_of hT hR hMR
  obtain ⟨pre, r, post, rfl, hloc⟩ := H.opened hT hR hMR (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h)
    (M := A.h.ms.length) (tab := tab) hx ht hj hjR hcs
  have hmem : (r.frame, r.coins) ∈ located (locOf A tab) (SpotFlock.off A.h t + j) [] (pre ++ r :: post) :=
    mem_located_of (loc := locOf A tab) hloc
  have hone := located_one_accepted A C tab hsz o hacc _ hl (SpotFlock.off A.h t + j)
  obtain ⟨recs', rec', hrecs', hrec', ⟨i, s, n, hfr⟩, hcoins⟩ :=
    located_own_accepted A C tab hsz o hacc _ hl t j ht hj _ hmem
  obtain rfl : records.toList.map (·.2.2.1) = recs' := Option.some.inj (hrecs.symm.trans hrecs')
  obtain rfl : x.2.2.1 = rec' := Option.some.inj (hrec.symm.trans hrec')
  have hfr' : r.frame = .opening i s n := hfr
  have hcoins' : r.coins = Verity.bitsOfBytes cs := by
    have h2 : r.coins = ((Verity.Sampled.recordRound x.2.2.1 (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
        (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map Verity.bitsOfBytes).getD [] := hcoins
    rw [h2, hcs]
    rfl
  refine ⟨i, s, n, ?_⟩
  rw [eq_singleton hmem hone, hfr', hcoins']

end Verity.SampledTwo.SessionCoins
