module

public import Proofs.Properties.Verity.Sampled.Window.Streams
public import Proofs.Properties.Verity.Sampled.RootB

/-!
# Sampled proofs' coupling over a window: a session's moves are its record's

A member's session's trace answers its rounds at releases of the one play's log (`Window/Answered.lean`), which the
window's readings name as the `u`-th session's link and rep rounds (`Window/Sessions.lean`). This module reads what
that trace decodes against the record the verifier read for that session, clause by clause of `MemberMovesOf`, as
`RootB.lean`, `LinkPts.lean` and `MovesReach.lean` do at one session:

* **The record of each session** (`sessionRecordAt`, `sentRecordAt`, `sessionReveal`): the `u`-th session's record
  in what the prover sent, and what the play reveals for a slot of that session, the link's message or a rep round's
  in that record, or, for its closing round, the session's evidence, the bytes its proof files are read from; at
  session `0` the record is the one session's (`sentRecordAt_zero`).
* **What the verifier's read of a session holds the device's log to** (`SessionRead`): at the `u`-th tree the log
  opens, the tree's first opening is of its stream named the session's, answers the SHA-512 of the record's link
  message and releases the record's link coins; each later opening of it is of a round the record gives its stream,
  answers that round's digest and releases that round's coins; the log opens every round the record gives a stream;
  each stream the record gives rounds of is a stream of the tree; the tree is the session's coin tree; each of
  its openings asks for the count of coins the record gives there, a coin a 16 bytes of the link's points and a
  round's `n`; and the tree's last opening is the session's closing round, the second opening of its stream named the
  session's, answering the SHA-512 of the session's evidence. But for the closing round it is the several-session
  form of what `MovesReach.lean` and `CountsLayout.lean` derive from the one-session read (`verdict_read`, `hans`,
  `OpenedRecorded`, `keyAt_of_record`, `SetsUp.helloSpec_session`, `Trees.session_counted`), stated on the log
  alone.
* **A stop's messages off the misses** (`stop_modes_tape`): at a position that keeps the invariant, on a tape through
  it that misses nowhere, a round fixed there decodes the message the play reveals for it, where the cuts are at the
  bytes the release answers and the message has that SHA-512.
* **The roots** (`session_rootB`, clause (ii)) and **the link points** (`session_pts`, clause (iii)) at a session's
  first stop, and **an answered stop's message and round** (`session_stop_msg`, `session_stop_round`,
  `session_answered`) and **the rep rounds**
  (`session_reps`, clause (iv)): the rounds the session's answered stops submit on a rep of its one table, in order,
  are the record's rounds of that stream (`session_stream_rounds`, `SessionRead.openedOf_le`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger
open FlockSoundness.Refine (roundOf ptsOf)
open FlockSoundness.Discharge.Tables (LiveOutJ SubmitJ addRoundJ)
open Claims.Bridge.CustodyZkP (repRounds)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## A stop's messages, off the misses -/

/-- **A later position's tape is a tape through the earlier one.** -/
theorem _root_.Verity.Game.Pos.Le.through {β W κ : Type} {g : Game β} {σ : Strategy g} {q r : Pos g σ W κ}
    (h : q.Le r) : ∃ y, r.tape = q.back y := by
  obtain ⟨f, hf⟩ := h
  exact ⟨f r.x, hf r.x⟩

/-- **Off the misses, a stop decodes the message the play reveals for its round**, on any tape through it
(`RoundsReach.stop_modes_reveal`, at a tape and no trace): at a position `q` that keeps the invariant at the rounds'
cuts, on a tape `z` through `q` that misses nowhere, where the cuts `q` fixes are at the bytes the release `r` of the
play's log answers (`CutsAt`) and the play reveals for a round fixed there a message `w` whose SHA-512 is those
bytes, `q` decodes `w` for that round. -/
theorem stop_modes_tape {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ}
    {rd : Reading R} {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {pre L : List Release} {r : Release}
    (hc : CutsAt rd.reveal q pre (CoinTape.run (V.game θ F c) σ z).log)
    (hL : (CoinTape.run (V.game θ F c) σ z).log = pre ++ r :: L) {i : Fin R} (hi : (q.Ks i).isHere = true)
    {w : Claims.Msg} (hw : rd.reveal i (CoinTape.run (V.game θ F c) σ z) = some w)
    (hd : sha512Msg w = r.answers.data.toList) : q.modes i = some w := by
  obtain ⟨y, rfl⟩ := hz
  have hrev : (roundCut V θ F c rd i).cut.revealOn σ (q.back y) = some w := by
    refine (Pos.revealOn_here hq (Pos.Le.refl q) (hc i r L hi hL) y).trans ?_
    unfold digestReveal
    rw [hw]
    simp [Option.filter, hd]
  rw [Pos.modes_eq hq (Pos.Le.refl q) y]
  dsimp only
  rw [ite_eq_left hi]
  by_contra hne
  exact hn.1 i ((DigestCut.missOn_iff _ _ _).2 ⟨w, hrev, hne⟩)

/-- **A stop whose fixed rounds all pick one frame decodes a message for one round only**, at a window's readings
of distinct tables, one at least (`windowReadingAt_unique`). -/
theorem windowStop_modes {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup) (hT : 0 < tables.size)
    {per : ℕ} (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g}
    {q : Pos g σ Claims.Msg (Fin R)} {j : Fin R} {pre : List Release} {fr : Frame}
    (hall : ∀ j', (q.Ks j').isHere = true → (windowReadingAt tables per reveal record recordDigest).round j' pre fr)
    (hP : (windowReadingAt tables per reveal record recordDigest).round j pre fr) :
    ∀ j', j' ≠ j → q.modes j' = none := fun j' hj' =>
  Pos.modes_eq_none (Bool.eq_false_iff.2 fun h =>
    hj' (windowReadingAt_unique ht hT hper reveal record recordDigest (hall j' h) hP))

/-- **A session's first stop is at its link's opening**: at a window's readings of distinct tables, where the rounds
the stop `q` fixes pick a release after the log `L₀` (`LoggedAfter`) and each round picking after `L₀` is the link's
of session `u`, that release is an opening of a tree `L₀` didn't open, after `L₀` opened `u` trees, and `q` decodes
a message for the link's round only. -/
theorem LoggedAfter.session_link {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup)
    (hT : 0 < tables.size) {per : ℕ} (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g}
    {q : Pos g σ Claims.Msg (Fin R)} {L₀ Lg : List Release} {bits : List Bool} {u : ℕ}
    (h : LoggedAfter (windowReadingAt tables per reveal record recordDigest).round q L₀ bits Lg)
    (hL₀ : ∀ j : Fin R, PicksIn (windowReadingAt tables per reveal record recordDigest).round Lg j L₀ →
      slotOf tables.size per (j.val % per) = .link ∧ j.val / per = u) :
    ∃ (j : Fin R) (i s n : ℕ) (b : ByteArray) (L : List Release), (q.Ks j).isHere = true ∧
      (∀ j', j' ≠ j → q.modes j' = none) ∧ slotOf tables.size per (j.val % per) = .link ∧ j.val / per = u ∧
      Lg = L₀ ++ ⟨.opening i s n, b, bits⟩ :: L ∧ i ∉ openedTrees L₀ ∧ (openedTrees L₀).length = u := by
  obtain ⟨j, i, s, n, b, L, hjK, hall, hj, hlog⟩ := h
  obtain ⟨hjl, hju⟩ := hL₀ j ⟨_, L, hlog, hj⟩
  have hj' : sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) L₀ (.opening i s n) := hj
  rw [hjl, hju] at hj'
  obtain ⟨i', s', n', hfr, -, hni, hlen⟩ := hj'
  obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
  exact ⟨j, _, _, _, b, L, hjK, windowStop_modes ht hT hper reveal record recordDigest hall hj, hjl, hju, hlog,
    hni, hlen⟩

/-- **An answered stop's slot, at a session of a window** (`LoggedAfter.slot`): at a window's readings of distinct
tables, no fewer than the decoders' `J`, and the decoders at its layout, a stop `q` that decodes the submission `sub`
and whose fixed rounds pick a release after the log `pre` fixes a rep round `j` of the first table, at which it
decodes the message `sub` is made of; the release is an opening the device serves of the tree `pre` opened at `j`'s
session's place, of its stream named that table's rep, after as many openings of it as the round's number. The
session's closing round, past the tables, decodes no submission. -/
theorem LoggedAfter.session_slot {R J : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) {tables : Array String}
    (ht : tables.toList.Nodup) (hJ : J ≤ tables.size) {per : ℕ} (hper : 0 < per)
    (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g} {q : Pos g σ Claims.Msg (Fin R)}
    {sub : SubmitJ J}
    (hsub : (decodersAt reg fun j : Fin R => slotOf tables.size per (j.val % per)).sub J q.modes = some sub)
    {pre : List Release} {bits : List Bool} {log : List Release}
    (h : LoggedAfter (windowReadingAt tables per reveal record recordDigest).round q pre bits log) :
    ∃ (ht' : 0 < J) (r : Fin 2) (k : ℕ) (w : Claims.Msg) (j : Fin R) (i s n : ℕ) (b : ByteArray)
      (L' : List Release), slotOf tables.size per (j.val % per) = .rep 0 r k ∧ (q.Ks j).isHere = true ∧
        q.modes j = some w ∧ sub = ⟨⟨0, ht'⟩, r, ⟨w.toArray⟩, tailCount w⟩ ∧ Served pre i s n ∧
        (openedTrees pre)[j.val / per]? = some i ∧
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = tables[0]?.map (repStreamName · r.val) ∧
        openedOf pre i s = k ∧ log = pre ++ ⟨.opening i s n, b, bits⟩ :: L' ∧
        (windowReadingAt tables per reveal record recordDigest).round j pre (.opening i s n) := by
  obtain ⟨j, i, s, n, b, L', hj, hall, hP, he⟩ := h
  obtain ⟨t, ht', r, k, w, hLj, hw, hs⟩ :=
    sub_of_single reg (fun j : Fin R => slotOf tables.size per (j.val % per)) J (modes := q.modes)
      (windowStop_modes ht (Nat.lt_of_lt_of_le sub.tab.pos hJ) hper reveal record recordDigest hall hP)
      hsub
  have hLj' : slotOf tables.size per (j.val % per) = .rep t r k := hLj
  have htT : t < tables.size := Nat.lt_of_lt_of_le ht' hJ
  obtain ⟨-, hLa⟩ := slotOf_rep hLj' htT
  obtain ⟨rfl, -⟩ := slotAt_eq_rep hLa
  have hP' : sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) pre (.opening i s n) := hP
  rw [hLj'] at hP'
  obtain ⟨i', s', n', hfr, hsv, hu, hname, hk⟩ := hP'
  obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
  rw [slotStream_lt htT r] at hname
  exact ⟨ht', r, k, w, j, _, _, _, b, L', hLj', hj, hw, hs, hsv, hu, hname, hk, he, hP⟩

/-- **A window's rounds have a slot for each of a session's rep streams' first `K` rounds**, wherever each session
has at least `2K + 2` rounds and the window has `u + 1` sessions' rounds (`Live.layout_covers`, at a session of a
window). -/
theorem window_covers (T : ℕ) {R per u K : ℕ} (hper : 2 * K + 2 ≤ per) (hu : (u + 1) * per ≤ R) (ρ : Fin 2) :
    ∀ k < K, ∃ j : Fin R, j.val / per = u ∧ slotOf T per (j.val % per) = .rep 0 ρ k := fun k hk => by
  have hρ := ρ.isLt
  have hlt : 2 * k + ρ.val + 1 < per := by omega
  have hne : 2 * k + ρ.val + 1 + 1 ≠ per := by omega
  have hpos : 0 < per := by omega
  have hR : per * u + (2 * k + ρ.val + 1) < R := by
    have e : (u + 1) * per = per * u + per := by rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    omega
  refine ⟨⟨per * u + (2 * k + ρ.val + 1), hR⟩, ?_, ?_⟩
  · show (per * u + (2 * k + ρ.val + 1)) / per = u
    rw [Nat.mul_add_div hpos, Nat.div_eq_of_lt hlt, Nat.add_zero]
  · show slotOf T per ((per * u + (2 * k + ρ.val + 1)) % per) = .rep 0 ρ k
    rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hlt, slotOf_of_ne T hne]
    exact slotAt_rep k ρ

/-- **A window's rounds have each session's closing round**, at its last place, wherever the window has `u + 1`
sessions' rounds. -/
theorem window_close (T : ℕ) {R per u : ℕ} (hper : 0 < per) (hu : (u + 1) * per ≤ R) :
    ∃ j : Fin R, j.val / per = u ∧ slotOf T per (j.val % per) = closeSlot T := by
  have hlt : per - 1 < per := by omega
  have hR : per * u + (per - 1) < R := by
    have e : (u + 1) * per = per * u + per := by rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    omega
  refine ⟨⟨per * u + (per - 1), hR⟩, ?_, ?_⟩
  · show (per * u + (per - 1)) / per = u
    rw [Nat.mul_add_div hper, Nat.div_eq_of_lt hlt, Nat.add_zero]
  · show slotOf T per ((per * u + (per - 1)) % per) = closeSlot T
    rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
    exact slotOf_last T (by omega)

/-- One table is a table. -/
theorem one_table_pos (tb : String) : 0 < (#[tb] : Array String).size := Nat.zero_lt_one

/-! ## The record of each session -/

/-- **The `u`-th session's record as the verifier reads it** on the blobs `bs`: the blob the evidence's `u`-th
session names as its record, as text (`sessionRecord`, at each session). -/
def sessionRecordAt (u : ℕ) (e : _root_.Spot.OneStageZk.Evidence) (bs : _root_.Core.Blobs) : Option String :=
  e.sessions[u]?.bind fun s => (bs.get s.record s!"session {s.name}'s record").toOption.bind String.fromUTF8?

/-- The first session's record is the one session's. -/
theorem sessionRecordAt_zero : sessionRecordAt 0 = sessionRecord := rfl

/-- **The `u`-th session's record the verifier reads in what the prover sent**, on the blobs sent (`sentRecord`, at
each session). -/
def sentRecordAt (u : ℕ) (s : Verity.Sent) : Option String :=
  (caseOf s).toOption.bind fun x => sessionRecordAt u x.2.2 (_root_.Core.Blobs.ofArray s.blobs)

/-- The first session's record in what the prover sent is the one session's. -/
theorem sentRecordAt_zero : sentRecordAt 0 = sentRecord := rfl

/-- **What the play reveals for a slot of the `u`-th session** (`slotReveal`, at each session): the link's message
of that session's record, the round's of the table's rep stream in it, or, at a slot past the tables, the session's
closing round's, the session's evidence (`evidence`): the bytes its proof files are read from. -/
def sessionReveal (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u : ℕ) :
    Slot → Exchange → Option Claims.Msg
  | .link => fun o => (sentRecordAt u o.sent).bind (linkMsg · tables)
  | .rep t r k => fun o =>
    if t = tables.size then evidence u o
    else tables[t]?.bind fun a => (sentRecordAt u o.sent).bind (roundMsg · (repStreamName a r.val) k)

theorem sessionReveal_link (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u : ℕ)
    (o : Exchange) : sessionReveal tables evidence u .link o = (sentRecordAt u o.sent).bind (linkMsg · tables) :=
  rfl

theorem sessionReveal_rep (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u t : ℕ)
    (r : Fin 2) (k : ℕ) (o : Exchange) {a : String} (ha : tables[t]? = some a) :
    sessionReveal tables evidence u (.rep t r k) o =
      (sentRecordAt u o.sent).bind (roundMsg · (repStreamName a r.val) k) := by
  have ha' : tables.toList[t]? = some a := by
    rw [Array.getElem?_toList]
    exact ha
  have hlt : t < tables.size := (List.getElem?_eq_some_iff.1 ha').1
  have hne : ¬ t = tables.size := Nat.ne_of_lt hlt
  show (if t = tables.size then evidence u o
    else tables[t]?.bind fun a => (sentRecordAt u o.sent).bind (roundMsg · (repStreamName a r.val) k)) = _
  rw [if_neg hne, ha, Option.bind_some]

/-- What the play reveals for a session's closing round is the session's evidence. -/
theorem sessionReveal_close (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u : ℕ)
    (o : Exchange) : sessionReveal tables evidence u (closeSlot tables.size) o = evidence u o :=
  if_pos rfl

/-! ## What the verifier's read of a session holds the device's log to -/

/-- **Where an opening of a session's tree is among the trees a log has opened**: an opening of the `u`-th tree a
log opens is the first of that tree, after the log has opened `u` trees, or follows a log whose `u`-th tree it is. -/
theorem opened_at_or_new {Lg L₁ L₂ : List Release} {x : Release} {i s n u : ℕ} (he : Lg = L₁ ++ x :: L₂)
    (hx : x.frame = .opening i s n) (hτ : (openedTrees Lg)[u]? = some i) :
    (i ∉ openedTrees L₁ ∧ (openedTrees L₁).length = u) ∨ (openedTrees L₁)[u]? = some i := by
  by_cases hmem : i ∈ openedTrees L₁
  · obtain ⟨v, hv⟩ := List.getElem?_of_mem hmem
    have h := openedTrees_getElem?_append hv (x :: L₂)
    rw [← he] at h
    have hvu : v = u := nodup_getElem?_inj (openedTrees_nodup Lg) h hτ
    rw [hvu] at hv
    exact Or.inr hv
  · obtain ⟨e, hE⟩ := openedTrees_new hx hmem L₂
    rw [← he] at hE
    have h : (openedTrees Lg)[(openedTrees L₁).length]? = some i := by
      rw [hE, List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero]
    exact Or.inl ⟨hmem, nodup_getElem?_inj (openedTrees_nodup Lg) h hτ⟩

/-- **What the verifier's read of a session holds the device's log to, less the closing round**, at the `u`-th tree
the log `Lg` opens, the session's record `text`, parsed as `J` and decoded as `sess`, and the session's tables:

* `coins`: each round the record gives a stream gives coins;
* `link`: the tree's first opening is of its stream named the session's, answers the SHA-512 of the record's link
  message, and releases the record's link coins where the record gives any;
* `round`: each later opening of the tree, of a stream other than the session's, is of a round the record gives that
  stream, answers that round's digest, and releases that round's coins where the record gives any;
* `full`: the log opens every round the record gives a stream of the tree;
* `key`: each stream the record gives rounds of is a stream of the tree;
* `tree`: the tree is sampled with the session's coin tree;
* `linkCount`: the tree's first opening asks for a coin a 16 bytes of the record's link coins;
* `roundCount`: each later opening of the tree, of a stream other than the session's, asks for the `n` of the round
  the record gives there.

No field reads an opening of the tree's stream named the session's after its first, so none changes where the log
drops one (`SessionOpens.drop_iff`, `Window/Closing.lean`). -/
structure SessionOpens (Lg : List Release) (tables : Array String) (u : ℕ) (text : String) (J : Json)
    (sess : Flock.RunRecord) : Prop where
  parse : Flock.JsonParse.parse text = .ok J
  coins : ∀ name, ∀ k < recordRounds J name, (recordRound J name k).isSome
  link : ∀ L₁ x L₂ i s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n → i ∉ openedTrees L₁ →
    (openedTrees L₁).length = u →
      ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION ∧
        (∃ w, linkMsg text tables = some w ∧ sha512Msg w = x.answers.data.toList) ∧
        ∃ b, recordRound J TREE_SESSION 0 = some b ∧ (Verity.bitsOfBytes b ≠ [] → x.coins = Verity.bitsOfBytes b)
  round : ∀ L₁ x L₂ i s n name, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n →
    (openedTrees L₁)[u]? = some i → ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some name →
    (name == TREE_SESSION) = false →
      openedOf L₁ i s < recordRounds J name ∧
        (∀ st, sess.streams.find? (·.name == name) = some st → openedOf L₁ i s < st.rounds.size →
          x.answers = st.rounds[openedOf L₁ i s]!.sha) ∧
        ∀ b, recordRound J name (openedOf L₁ i s) = some b → Verity.bitsOfBytes b ≠ [] →
          x.coins = Verity.bitsOfBytes b
  full : ∀ i s name, (openedTrees Lg)[u]? = some i →
    ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = some name → (name == TREE_SESSION) = false →
      recordRounds J name ≤ openedOf Lg i s
  key : ∀ name, 0 < recordRounds J name → ∀ i, (openedTrees Lg)[u]? = some i →
    ∃ s : ℕ, ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = some name
  tree : ∀ i, (openedTrees Lg)[u]? = some i → (helloSpecs Lg)[i]? = some (Flock.Zk.coinSpec tables)
  linkCount : ∀ L₁ x L₂ i s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n → i ∉ openedTrees L₁ →
    (openedTrees L₁).length = u → ∀ b, recordRound J TREE_SESSION 0 = some b → n = b.size / 16
  roundCount : ∀ L₁ x L₂ i s n name, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n →
    (openedTrees L₁)[u]? = some i → ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some name →
    (name == TREE_SESSION) = false → ∀ st, sess.streams.find? (·.name == name) = some st →
      openedOf L₁ i s < st.rounds.size → n = st.rounds[openedOf L₁ i s]!.n

/-- **What the verifier's read of a session holds the device's log to**: `SessionOpens`, and the session's closing
round at its evidence `w`, the bytes its proof files are read from (`close`): the tree has an opening of its stream
named the session's after one opening of that stream, the link's; it answers the SHA-512 of the session's evidence;
and no opening of the tree follows it. -/
structure SessionRead (Lg : List Release) (tables : Array String) (u : ℕ) (text : String) (J : Json)
    (sess : Flock.RunRecord) (w : Claims.Msg) : Prop extends SessionOpens Lg tables u text J sess where
  close : ∀ i, (openedTrees Lg)[u]? = some i → ∃ Lc x L₂ s n, Lg = Lc ++ x :: L₂ ∧ x.frame = .opening i s n ∧
    (openedTrees Lc)[u]? = some i ∧
    ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION ∧ openedOf Lc i s = 1 ∧
    sha512Msg w = x.answers.data.toList ∧
    ∀ L₁ y L₃ s' n', Lg = L₁ ++ y :: L₃ → y.frame = .opening i s' n' → L₁.length ≤ Lc.length

/-- **A log a session's read holds opens each rep stream of the session's tree at most the rounds the record gives
it** (`openedOf_le_recordRounds`, at a session of a window). -/
theorem SessionRead.openedOf_le {Lg : List Release} {tables : Array String} {u : ℕ} {text : String} {J : Json}
    {sess : Flock.RunRecord} {w : Claims.Msg} (hR : SessionRead Lg tables u text J sess w) (hS : LogServed Lg)
    {i s : ℕ}
    {name : String} (hτ : (openedTrees Lg)[u]? = some i)
    (hname : ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = some name)
    (hnT : (name == TREE_SESSION) = false) : openedOf Lg i s ≤ recordRounds J name := by
  suffices h : ∀ pre post, Lg = pre ++ post → openedOf pre i s ≤ recordRounds J name from
    h Lg [] (List.append_nil Lg).symm
  intro pre
  induction pre using List.reverseRecOn with
  | nil => intro _ _; exact Nat.zero_le _
  | append_singleton pre r ih =>
    intro post he
    rw [List.append_assoc, List.singleton_append] at he
    rw [openedOf_append]
    by_cases hr : r.frame.opensAt i s = true
    · obtain ⟨n, hn⟩ := opening_of_opensAt hr
      rw [ite_eq_left hr]
      obtain ⟨sp, hsp, hs, -⟩ := hS pre r post i s n he hn
      have h1 : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some sp.streams[s] := by
        rw [hsp, Option.bind_some, Array.getElem?_eq_getElem hs]
      have h2 := helloSpecs_bind_append h1 (r :: post)
      rw [← he, hname] at h2
      have h1' : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name := h1.trans h2.symm
      rcases opened_at_or_new he hn hτ with ⟨hni, hlen⟩ | hu
      · have h3 := (hR.link pre r post i s n he hn hni hlen).1
        rw [h1'] at h3
        rw [Option.some.inj h3] at hnT
        simp at hnT
      · have hlt := (hR.round pre r post i s n name he hn hu h1' hnT).1
        omega
    · rw [ite_eq_right hr]
      exact ih (r :: post) he

/-! ## The roots and the link points -/

/-- **The roots decoded at a session's first stop are its record's `root_B`** (`MemberMovesOf`'s roots;
`firstStop_rootB`, at a session of a window): at a window's readings at one table and the decoders at its layout, at
a position `q` that keeps the invariant, on a tape through it off the misses, where the rounds `q` fixes pick a
release after the log `L₀` (`LoggedAfter`), each round picking after `L₀` is the link's of session `u`, the cuts
`q` fixes are at the bytes that release answers (`CutsAt`), and the verifier's read holds the play's log to session
`u`'s record (`SessionRead`), the roots the decoders read at `q` are the `root_B` of the session the record decodes
to. -/
theorem session_rootB {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ} {Reg : Type}
    (reg : Option Claims.Msg → Reg) {tb : String} {per : ℕ} (hper : 0 < per)
    (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading R}
    (hrd : rd = windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {L₀ : List Release} {bits : List Bool}
    (hLA : LoggedAfter rd.round q L₀ bits (CoinTape.run (V.game θ F c) σ z).log)
    (hc : CutsAt rd.reveal q L₀ (CoinTape.run (V.game θ F c) σ z).log) {u : ℕ}
    (hL₀ : ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j L₀ →
      slotOf (#[tb] : Array String).size per (j.val % per) = .link ∧ j.val / per = u)
    {text : String} {J : Json} {spec : Flock.SessionSpec} {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (ht : spec.tables = #[tb]) (hσ : spec.sigma.size = 64) :
    (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)).root q.modes =
      sess.rootB := by
  subst hrd
  obtain ⟨j, i, s, n, b, L, hjK, hmodes, hjl, hju, hlog, hni, hlen⟩ :=
    LoggedAfter.session_link (by simp) (one_table_pos tb) hper (sessionReveal #[tb] ev) record₀ recordDigest hLA hL₀
  obtain ⟨-, ⟨w, hw, hwd⟩, -⟩ := hR.link L₀ _ L i s n hlog rfl hni hlen
  have hrev : (windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest).reveal j
      (CoinTape.run (V.game θ F c) σ z) = some w := by
    show sessionReveal #[tb] ev (j.val / per) (slotOf (#[tb] : Array String).size per (j.val % per))
      (CoinTape.run (V.game θ F c) σ z) = some w
    rw [hjl, hju, sessionReveal_link, hrec, Option.bind_some]
    exact hw
  have hmode := stop_modes_tape hq hz hn hc hlog hjK hrev hwd
  refine root_record reg (fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)) (i := j)
    (by show (slotOf (#[tb] : Array String).size per (j.val % per)).isLink = true; rw [hjl]; rfl) hmode
    (fun j' hj' _ => hmodes j' hj') hd ht hσ ?_
  rw [ht]
  exact hw

/-- **The link points the device's log shows at a session's first stop are its record's** (`MemberMovesOf`'s link
points; `link_pts`, at a session of a window): at a window's readings of distinct tables, where the rounds the stop
`q` fixes pick a release after the log `L₀` that released `linkBits fieldEnc pts`, each round picking after `L₀` is
the link's of session `u`, and the verifier's read holds the log to session `u`'s record (`SessionRead`) decoded at
a session of two link points of some coordinates, the session's link points read as `pts`. -/
theorem session_pts {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup) (hT : 0 < tables.size) {per : ℕ}
    (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g} {q : Pos g σ Claims.Msg (Fin R)}
    {L₀ Lg : List Release} {mP : ℕ} {pts : Fin 2 → Fin mP → 𝔽} {u : ℕ}
    (hLA : LoggedAfter (windowReadingAt tables per reveal record recordDigest).round q L₀ (linkBits fieldEnc pts)
      Lg)
    (hL₀ : ∀ j : Fin R, PicksIn (windowReadingAt tables per reveal record recordDigest).round Lg j L₀ →
      slotOf tables.size per (j.val % per) = .link ∧ j.val / per = u)
    {tables' : Array String} {text : String} {J : Json} {spec : Flock.SessionSpec} {sess : Flock.RunRecord}
    {w₀ : Claims.Msg} (hR : SessionRead Lg tables' u text J sess w₀) (hd : Flock.Record.decode spec text = .ok sess)
    (hp2 : spec.points = 2) (hpos : 0 < spec.mPts) : ptsOf mP sess.points = pts := by
  obtain ⟨j, i, s, n, b, L, -, -, -, -, hlog, hni, hlen⟩ :=
    LoggedAfter.session_link ht hT hper reveal record recordDigest hLA hL₀
  obtain ⟨-, -, b', hb', hcoins⟩ := hR.link L₀ _ L i s n hlog rfl hni hlen
  have hbits : Verity.bitsOfBytes b' = linkBits fieldEnc pts :=
    (hcoins (link_bits_ne hd hR.parse hp2 hpos hb')).symm
  exact pts_of_record hd hR.parse hp2 hb' hbits

/-! ## The rep rounds, stop by stop -/

/-- **An answered stop's message is its session's record's** (`stop_round`'s message, at a session of a window): at a
window's readings at one table, at a position `q` that keeps the invariant, on a tape through it off the misses,
where `q` fixes round `j`'s cut, round `k` of rep `ρ` of session `u`, at the most likely message `w`, and the
device's log has an opening of the `u`-th tree's stream `s`, named that rep's, after the log `m` at which the cuts
`q` fixes are at the bytes that opening answers and which opened that stream `k` times: where the verifier's read
holds the log to session `u`'s record (`SessionRead`) and `Flock.Zk.verify` accepts the record under retained rounds
and SHA-512 digests, the record's stream so named has round `k`, whose message is `w`. -/
theorem session_stop_msg {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ}
    {tb : String} (per : ℕ) (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading R}
    (hrd : rd = windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {j : Fin R} {u : ℕ} {ρ : Fin 2} {k : ℕ}
    (hLj : slotOf (#[tb] : Array String).size per (j.val % per) = .rep 0 ρ k) (hju : j.val / per = u)
    (hj : (q.Ks j).isHere = true)
    {w : Claims.Msg} (hw : q.modes j = some w) {bits : List Bool} {m L' : List Release} {i s n : ℕ}
    {b : ByteArray} (hlog : (CoinTape.run (V.game θ F c) σ z).log = m ++ ⟨.opening i s n, b, bits⟩ :: L')
    (hcut : CutsAt rd.reveal q m (CoinTape.run (V.game θ F c) σ z).log)
    (hτ : (openedTrees m)[u]? = some i) (hk : openedOf m i s = k)
    (hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb ρ.val))
    {text : String} {J : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) (htab : spec.tables = #[tb]) :
    ∃ st, sess.streams.find? (·.name == repStreamName tb ρ.val) = some st ∧ k < st.rounds.size ∧
      roundMsg text (repStreamName tb ρ.val) k = some w := by
  subst hrd hk
  have h0 : tb = spec.tables[0]! := by
    rw [htab]
    all_goals rfl
  subst h0
  have hnT := repStreamName_ne spec.tables[0]! ρ.val
  obtain ⟨h0t, sch, tPad, cap, -, hrep⟩ := verifyRep_of_accepted hv hd ρ
  obtain ⟨st, _, _, _, hst, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrep
  have hst' : sess.streams.find? (·.name == repStreamName spec.tables[0]! ρ.val) = some st := hst
  have hrr : recordRounds J (repStreamName spec.tables[0]! ρ.val) = st.rounds.size :=
    recordRounds_decode hd hR.parse hnT hst'
  obtain ⟨hlt, hans, -⟩ := hR.round m _ L' i s n _ hlog rfl hτ hname hnT
  have hkR : openedOf m i s < st.rounds.size := by
    rw [hrr] at hlt
    exact hlt
  obtain ⟨cs', -, -, hsha⟩ := recordRound_decode hd hR.parse hnT hst' hkR
  obtain ⟨bb, hbb, hmsg⟩ := roundMsg_decode hd hret hst' hkR
  have hdg512 : spec.dg bb = Flock.Sha512.hash bb := by simp [Flock.SessionSpec.dg, h512]
  have hyans : b = st.rounds[openedOf m i s].sha := by
    have h : b = st.rounds[openedOf m i s]!.sha := hans st hst' hkR
    rw [getElem!_pos st.rounds (openedOf m i s) hkR] at h
    exact h
  have hrev : (windowReadingAt #[spec.tables[0]!] per (sessionReveal #[spec.tables[0]!] ev) record₀
      recordDigest).reveal j (CoinTape.run (V.game θ F c) σ z) = some bb.data.toList := by
    show sessionReveal #[spec.tables[0]!] ev (j.val / per)
      (slotOf (#[spec.tables[0]!] : Array String).size per (j.val % per))
      (CoinTape.run (V.game θ F c) σ z) = some bb.data.toList
    rw [hLj, hju]
    refine (sessionReveal_rep #[spec.tables[0]!] ev u 0 ρ (openedOf m i s) (CoinTape.run (V.game θ F c) σ z)
      (a := spec.tables[0]!) rfl).trans ?_
    rw [hrec, Option.bind_some]
    exact hmsg
  have hmode := stop_modes_tape hq hz hn hcut hlog hj hrev (by
    show sha512Msg bb.data.toList = b.data.toList
    rw [hyans, hsha bb hbb, hdg512]
    all_goals rfl)
  have hwb : w = bb.data.toList := Option.some.inj (hw.symm.trans hmode)
  rw [hwb]
  exact ⟨st, hst', hkR, hmsg⟩

/-- **An answered stop's round is its session's record's** (`stop_round`, at a session of a window): at
`session_stop_msg`'s stop, where the opening released the coins `cs` of `w`'s count, the record's stream named the
rep's has round `k`, and it is the round `roundOf` makes of `w` and `cs`. -/
theorem session_stop_round {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ}
    {tb : String} (per : ℕ) (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading R}
    (hrd : rd = windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {j : Fin R} {u : ℕ} {ρ : Fin 2} {k : ℕ}
    (hLj : slotOf (#[tb] : Array String).size per (j.val % per) = .rep 0 ρ k) (hju : j.val / per = u)
    (hj : (q.Ks j).isHere = true)
    {w : Claims.Msg} (hw : q.modes j = some w) {cs : Fin (tailCount w) → 𝔽} {m L' : List Release} {i s n : ℕ}
    {b : ByteArray}
    (hlog : (CoinTape.run (V.game θ F c) σ z).log = m ++ ⟨.opening i s n, b, roundBits fieldEnc cs⟩ :: L')
    (hcut : CutsAt rd.reveal q m (CoinTape.run (V.game θ F c) σ z).log)
    (hτ : (openedTrees m)[u]? = some i) (hk : openedOf m i s = k)
    (hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb ρ.val))
    {text : String} {J : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) {dg : ByteArray → ByteArray}
    (hdg : spec.dg = dg) (htab : spec.tables = #[tb]) :
    ∃ st, sess.streams.find? (·.name == repStreamName tb ρ.val) = some st ∧
      ∃ hk : k < st.rounds.size, roundOf dg ⟨w.toArray⟩ (tailCount w) cs = st.rounds[k] := by
  obtain ⟨st, hst', hkR, hmsg⟩ := session_stop_msg per ev record₀ recordDigest hrd hq hz hn hLj hju hj hw hlog hcut
    hτ hk hname hrec hR hd hv hret h512 htab
  subst hk
  have h0 : tb = spec.tables[0]! := by
    rw [htab]
    all_goals rfl
  subst h0
  have hnT := repStreamName_ne spec.tables[0]! ρ.val
  have hrr : recordRounds J (repStreamName spec.tables[0]! ρ.val) = st.rounds.size :=
    recordRounds_decode hd hR.parse hnT hst'
  obtain ⟨-, -, hcoin⟩ := hR.round m _ L' i s n _ hlog rfl hτ hname hnT
  obtain ⟨b', hb'⟩ := Option.isSome_iff_exists.1
    (hR.coins (repStreamName spec.tables[0]! ρ.val) (openedOf m i s) (by rw [hrr]; exact hkR))
  obtain ⟨cs', hrr', hf', -⟩ := recordRound_decode hd hR.parse hnT hst' hkR
  have hbits : Verity.bitsOfBytes b' = roundBits fieldEnc cs := by
    by_cases hc : Verity.bitsOfBytes b' = []
    · have hrc : _root_.Spot.Challenger.recordCoins? (.arr cs') = some b' := hrr'.symm.trans hb'
      have hlen := congrArg List.length (bits_of_coins hf' hrc)
      rw [hc, length_chunks, List.length_nil, Array.length_toList] at hlen
      have hmem : st ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hst')
      have hwf := FlockSoundness.Refine.decode_roundsWF hd st hmem (openedOf m i s) hkR
      rw [getElem!_pos st.rounds (openedOf m i s) hkR] at hwf
      have htc := rep_tailCount_accepted hv hd hret ρ hst' hkR hmsg
      have hzero : (roundBits fieldEnc cs).length = 0 := by
        rw [length_roundBits, htc, ← hwf]
        omega
      rw [hc, List.length_eq_zero_iff.1 hzero]
    · exact (hcoin b' hb' hc).symm
  obtain ⟨hB', hS', hN', hsz, hC'⟩ := answered_of_record hd hR.parse hret hnT hst' hkR hmsg hb' hbits
  have hcoins : (Array.ofFn fun a => (cs a).val) = st.rounds[openedOf m i s].coins := by
    refine Array.ext (by rw [Array.size_ofFn, hsz]) fun a h₁ h₂ => ?_
    have ha : a < tailCount w := by simpa using h₁
    rw [Array.getElem_ofFn, hC' ⟨a, ha⟩, getElem!_pos _ a h₂]
  refine ⟨st, hst', hkR, ?_⟩
  have e : st.rounds[openedOf m i s] =
      ({ n := st.rounds[openedOf m i s].n, sha := st.rounds[openedOf m i s].sha,
         coins := st.rounds[openedOf m i s].coins, bytes := st.rounds[openedOf m i s].bytes } : Flock.Round) := rfl
  rw [e, hB', hS', hN', ← hcoins, ← hdg]
  all_goals rfl

/-- **An answered stop of a session submits its record's round** (`LoggedAfter.session_slot`, `session_stop_round`):
at a window's readings at one table and the decoders at its layout, at a position `q` that keeps the invariant, on a
tape through it off the misses, where `q` decodes the submission `sub`, the rounds it fixes pick a release after the
log `m` that released `roundBits fieldEnc cs`, each round picking after `m` is of session `u`, the cuts `q` fixes are
at the bytes that release answers, and the verifier's read and the record are as `session_stop_round` takes them:
the release is an opening the device serves of the `u`-th tree's stream named `sub`'s rep's, a rep round of session
`u` picks it, and the round `roundOf` makes of `sub` and `cs` is the record's round of that stream at the count of
its openings in `m`. -/
theorem session_answered {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R J : ℕ}
    {Reg : Type} (reg : Option Claims.Msg → Reg) (hJ : J ≤ 1) {tb : String} {per : ℕ} (hper : 0 < per)
    (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg)
    {rd : Reading R} (hrd : rd = windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {sub : SubmitJ J} {cs : Fin sub.n → 𝔽} {m : List Release}
    (hsub : (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)).sub J q.modes =
      some sub)
    (hlog : LoggedAfter rd.round q m (roundBits fieldEnc cs) (CoinTape.run (V.game θ F c) σ z).log)
    (hcut : CutsAt rd.reveal q m (CoinTape.run (V.game θ F c) σ z).log) {u : ℕ}
    (hses : ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j m → j.val / per = u)
    {text : String} {J' : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J' sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) {dg : ByteArray → ByteArray}
    (hdg : spec.dg = dg) (htab : spec.tables = #[tb]) :
    ∃ (j : Fin R) (k i s n : ℕ) (x : Release) (L' : List Release), j.val / per = u ∧
      slotOf (#[tb] : Array String).size per (j.val % per) = .rep sub.tab.val sub.rep k ∧ rd.round j m x.frame ∧
      (CoinTape.run (V.game θ F c) σ z).log = m ++ x :: L' ∧ x.frame = .opening i s n ∧ Served m i s n ∧
      (openedTrees m)[u]? = some i ∧
      ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb sub.rep.val) ∧
      ∃ st, sess.streams.find? (·.name == repStreamName tb sub.rep.val) = some st ∧
        ∃ hk : openedOf m i s < st.rounds.size, roundOf dg sub.bytes sub.n cs = st.rounds[openedOf m i s] := by
  subst hrd
  obtain ⟨ht', r, k, w, j, i, s, n, b, L', hLj, hj, hw, hsa, hsv, hu', hname, hk, he, hP⟩ :=
    LoggedAfter.session_slot reg (by simp) hJ hper (sessionReveal #[tb] ev) record₀ recordDigest hsub hlog
  subst hsa
  have hju := hses j ⟨_, L', he, hP⟩
  rw [hju] at hu'
  have hname' : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb r.val) := hname
  obtain ⟨st, hst, hkR, hround⟩ := session_stop_round per ev record₀ recordDigest rfl hq hz hn hLj hju hj hw he
    hcut hu' hk hname' hrec hR hd hv hret h512 hdg htab
  subst hk
  exact ⟨j, _, i, s, n, _, L', hju, hLj, hP, he, rfl, hsv, hu', hname', st, hst, hkR, hround⟩

/-! ## The rep rounds -/

/-- **A session's trace's rounds on a rep stream of its table are its record's** (`MemberMovesOf`'s rounds;
`rep_rounds`, at a session of a window): at a window's readings at one table, of at least `1026` rounds a session
and `u + 1` sessions' rounds, and the decoders at its layout, where the session's answered stops `as`, each keeping
the invariant with the tape `z` through it, each release their round's coins after the logs `ms`, each longer than
the one before (`MemberReach.order`, at the coin map that writes a round's coins as `roundBits fieldEnc`); `z`
misses nowhere; each round picking after one of `ms` is of session `u`, every release a round of session `u` picks
is after `L₀` or one of `ms`, or is its closing round's, and only the link's round picks after `L₀`; the log opens a
`u`-th tree and its trees
name their streams distinctly; and the verifier's read holds the log to session `u`'s record (`SessionRead`), which
`Flock.Zk.verify` accepts under retained rounds and SHA-512 digests: the rounds the stops submit on the table's rep
`ρ`, in order, are the record's rounds of that stream (`repRounds`). -/
theorem session_reps {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R J : ℕ}
    {Reg : Type} (reg : Option Claims.Msg → Reg) (hJ : J ≤ 1) {tb : String} {per : ℕ} (hper : 1026 ≤ per)
    (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg)
    {rd : Reading R} (hrd : rd = windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {z : CoinTape (V.game θ F c) σ} (hn : NoMiss V θ F c rd σ z)
    {as : List (Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)))}
    {ms : List (List Release)} {L₀ : List Release} {u τ : ℕ} (hu : (u + 1) * per ≤ R)
    (hF : List.Forall₂ (fun a m =>
      (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)).sub J
        a.1.toStop.modes = some a.2.1 ∧
      LoggedAfter rd.round a.1.toStop m (roundBits fieldEnc a.2.2) (CoinTape.run (V.game θ F c) σ z).log ∧
      CutsAt rd.reveal a.1.toStop m (CoinTape.run (V.game θ F c) σ z).log) as ms)
    (hI : ∀ a ∈ as, a.1.toStop.Inv (roundCuts V θ F c rd) ∧ ∃ y, z = a.1.toStop.back y)
    (hpw : ms.Pairwise (fun m m' => m.length < m'.length))
    (hses : ∀ m ∈ ms, ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j m → j.val / per = u)
    (hcov : ∀ (j : Fin R) L₁, j.val / per = u →
      PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j L₁ → L₁ = L₀ ∨ L₁ ∈ ms ∨
        slotOf (#[tb] : Array String).size per (j.val % per) = closeSlot (#[tb] : Array String).size)
    (hL₀ : ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j L₀ →
      slotOf (#[tb] : Array String).size per (j.val % per) = .link)
    (hτ : (openedTrees (CoinTape.run (V.game θ F c) σ z).log)[u]? = some τ)
    (hnd : ∀ sp ∈ helloSpecs (CoinTape.run (V.game θ F c) σ z).log, sp.streams.toList.Nodup)
    (t : Fin J) (htz : t.val = 0) (ρ : Fin 2)
    {text : String} {J' : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J' sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) {dg : ByteArray → ByteArray}
    (hdg : spec.dg = dg) (htab : spec.tables = #[tb]) :
    as.foldl (fun rs a => addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2))
      (fun _ _ => []) t ρ = repRounds spec sess t.val ρ.val := by
  subst hrd
  have h0 : tb = spec.tables[0]! := by
    rw [htab]
    all_goals rfl
  obtain ⟨tv, htv⟩ := t
  obtain rfl : tv = 0 := htz
  show _ = repRounds spec sess 0 ρ.val
  have hS := logServed_run V θ F c σ z
  have hnT := repStreamName_ne tb ρ.val
  obtain ⟨h0t, sch, tPad, cap, -, hrep⟩ := verifyRep_of_accepted hv hd ρ
  obtain ⟨st, _, _, _, hst, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrep
  have hst' : sess.streams.find? (·.name == repStreamName tb ρ.val) = some st := by
    rw [h0]
    exact hst
  have hrr := recordRounds_decode hd hR.parse hnT hst'
  have hRR : repRounds spec sess 0 ρ.val = st.rounds.toList := by
    show ((sess.streams.find? (·.name == repStreamName spec.tables[0]! ρ.val)).map (·.rounds.toList)).getD [] = _
    rw [← h0, hst']
    all_goals rfl
  have hstop : ∀ a m, (a, m) ∈ as.zip ms → ∃ (j : Fin R) (k i s n : ℕ) (x : Release) (L' : List Release),
      j.val / per = u ∧ slotOf (#[tb] : Array String).size per (j.val % per) = .rep a.2.1.tab.val a.2.1.rep k ∧
      (windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest).round j m x.frame ∧
      (CoinTape.run (V.game θ F c) σ z).log = m ++ x :: L' ∧ x.frame = .opening i s n ∧ Served m i s n ∧
      (openedTrees m)[u]? = some i ∧
      ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb a.2.1.rep.val) ∧
      ∃ st', sess.streams.find? (·.name == repStreamName tb a.2.1.rep.val) = some st' ∧
        ∃ hk : openedOf m i s < st'.rounds.size,
          roundOf dg a.2.1.bytes a.2.1.n a.2.2 = st'.rounds[openedOf m i s] := by
    intro a m hx
    obtain ⟨hsub, hlog, hcut⟩ := List.forall₂_zip hF hx
    obtain ⟨hinv, hzq⟩ := hI a (List.of_mem_zip hx).1
    exact session_answered reg hJ (by omega) ev record₀ recordDigest rfl hinv hzq hn hsub hlog hcut
      (hses m (List.of_mem_zip hx).2) hrec hR hd hv hret h512 hdg htab
  by_cases hpos : 0 < st.rounds.size
  · obtain ⟨s, hs0⟩ := hR.key (repStreamName tb ρ.val) (by rw [hrr]; exact hpos) τ hτ
    have hs : ((helloSpecs (CoinTape.run (V.game θ F c) σ z).log)[τ]?.bind fun sp => sp.streams[s]?) =
        (#[tb] : Array String)[(⟨0, htv⟩ : Fin J).val]?.map (repStreamName · ρ.val) := hs0
    have hF' : List.Forall₂ (fun a m => ∃ (j : Fin R) (k : ℕ) (x : Release) (L' : List Release),
        j.val / per = u ∧ slotOf (#[tb] : Array String).size per (j.val % per) = .rep a.2.1.tab.val a.2.1.rep k ∧
          (windowReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest).round j m x.frame ∧
          (CoinTape.run (V.game θ F c) σ z).log = m ++ x :: L') as ms := by
      refine List.forall₂_iff_zip.2 ⟨hF.length_eq, fun {a m} hx => ?_⟩
      obtain ⟨j, k, _, _, _, x, L', hju, hLj, hP, he, -⟩ := hstop a m hx
      exact ⟨j, k, x, L', hju, hLj, hP, he⟩
    have hfirst : ∀ L₁ x L₂ s' n, (CoinTape.run (V.game θ F c) σ z).log = L₁ ++ x :: L₂ →
        x.frame = .opening τ s' n → τ ∉ openedTrees L₁ →
        ((helloSpecs L₁)[τ]?.bind fun sp => sp.streams[s']?) = some _root_.Spot.Challenger.TREE_SESSION := by
      intro L₁ x L₂ s' n he hx hni
      rcases opened_at_or_new he hx hτ with ⟨-, hlen⟩ | hmem
      · exact (hR.link L₁ x L₂ τ s' n he hx hni hlen).1
      · exact absurd (List.mem_of_getElem? hmem) hni
    have hle : openedOf (CoinTape.run (V.game θ F c) σ z).log τ s ≤ 512 := hS.openedOf_le (hR.tree τ hτ) s
    have hcovL : ∀ k < openedOf (CoinTape.run (V.game θ F c) σ z).log τ s, ∃ j : Fin R,
        j.val / per = u ∧ slotOf (#[tb] : Array String).size per (j.val % per) = .rep (⟨0, htv⟩ : Fin J).val ρ k :=
      fun k hk => window_covers (#[tb] : Array String).size (K := 512) (by omega) hu ρ k (by omega)
    have hf : ∀ a m, (a, m) ∈ as.zip ms → a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = ρ →
        roundOf dg a.2.1.bytes a.2.1.n a.2.2 =
          st.rounds[openedOf m τ s]! := by
      intro a m hx htr
      obtain ⟨j, k, i, s', n, x, L', -, -, -, he, -, ⟨sp, hsp, hslt, -⟩, hu', hname, st', hst'', hkR, hround⟩ :=
        hstop a m hx
      rw [htr.2] at hname hst''
      have hi : τ = i := by
        have h := openedTrees_getElem?_append hu' (x :: L')
        rw [← he, hτ] at h
        exact Option.some.inj h
      subst hi
      have hss : s = s' := (stream_eq he hnd ⟨sp, hsp, hslt⟩ hname hs0).symm
      subst hss
      have hE : st = st' := Option.some.inj (hst'.symm.trans hst'')
      subst hE
      exact hround.trans (getElem!_pos st.rounds (openedOf m τ s) hkR).symm
    have hlen : openedOf (CoinTape.run (V.game θ F c) σ z).log τ s = st.rounds.size := by
      have h1 := hR.openedOf_le hS hτ hs0 hnT
      have h2 := hR.full τ s _ hτ hs0 hnT
      rw [hrr] at h1 h2
      omega
    have key := session_stream_rounds
      (γ := Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))) (J := J)
      (fun a => a.2.1.tab) (fun a => a.2.1.rep) (fun a => roundOf dg a.2.1.bytes a.2.1.n a.2.2)
      (tables := #[tb]) (by simp) (per := per) (by omega) (sessionReveal #[tb] ev) record₀ recordDigest hτ hF' hpw
      hcov hL₀ hS hnd hfirst ⟨0, htv⟩ (one_table_pos tb) ρ hs hcovL (fun k => st.rounds[k]!) hf
    refine key.trans ?_
    rw [hlen, map_range_getElem!, hRR]
  · have hzero : st.rounds.size = 0 := by omega
    have hnone : ∀ a ∈ as, ¬ (a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = ρ) := by
      intro a ha htr
      have ha' : a ∈ (as.zip ms).map Prod.fst := by
        rw [List.map_fst_zip hF.length_eq.le]
        exact ha
      obtain ⟨⟨a', m⟩, hx, rfl⟩ := List.mem_map.1 ha'
      have hrep : a'.2.1.rep = ρ := htr.2
      obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, st', hst'', hkR, _⟩ := hstop a' m hx
      rw [hrep] at hst''
      obtain rfl : st' = st := Option.some.inj (hst''.symm.trans hst')
      omega
    have hfil : as.filter (fun a => decide (a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = ρ)) = [] :=
      List.filter_eq_nil_iff.2 fun a ha h => hnone a ha (of_decide_eq_true h)
    have hnil : st.rounds.toList = [] := List.eq_nil_of_length_eq_zero (by simpa using hzero)
    refine (foldl_addRoundJ
      (fun a : Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)) => a.2.1.tab)
      (fun a : Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)) => a.2.1.rep)
      (fun a : Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)) =>
        roundOf dg a.2.1.bytes a.2.1.n a.2.2)
      as (fun _ _ => []) ⟨0, htv⟩ ρ).trans ?_
    rw [hfil, hRR, hnil]
    all_goals rfl

end Verity.Sampled
