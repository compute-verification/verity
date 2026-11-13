module

public import Proofs.Properties.Verity.SampledTwo.SessionRecordRounds
public import Proofs.Properties.Verity.SampledTwo.CutPrints
public import Proofs.Properties.Verity.Sampled.MovesLayout

/-!
# A release put on a session's round answers the digest the session's record gives that round

The session-round locator (`locOf`) puts an opening on round `k` of session `t` where it opens, of the `t`-th coin
tree the log opens, the stream and the round the session's layout gives `k` (`slotStream`). `SessionRecordRounds`
holds that round to be one the session's record gives. This module reads what the opening answered: the bytes the
challenger opened the round after are the digest the session's record gives the round (`recordDigest`).

* `recordDigest`: the digest a session's record gives a round. Of the stream `TREE_SESSION`, the SHA-512 of the
  link's `root_F` and each table's `root_B` in table-name order (`Spot.Challenger.linkAfter` at `linkRootF` and
  `linkRootB`); of another stream, the bytes of the round's `msg_sha512`, in the record's first stream so named.
* `link_after`, `recordDigest_round`: a record `Spot.Challenger.sessionTreeReason` accepts has its tree's
  `TREE_SESSION` round 0 opened after the link's digest in lowercase hex, and each round the record gives a stream
  opened after that round's `msg_sha512`.
* `MembersHeld.answers`: at the read. The release is an opening of session `t`'s tree, of the stream and round
  `slotStream` gives (`MembersHeld.key`). The read gives the opening its line, whose `after` in lowercase hex is the
  bytes the release answers (`Verity.Sampled.readLog_round`), and the line is the one the tree shows that round
  opened after (`Verity.Sampled.opening_after`). The tree is the one session `t`'s record names
  (`MembersHeld.roots`, `MembersHeld.idx`), and the members' check holds the record to that tree less its closing
  round (`Verity.Sampled.MemberForm.memberTreeReason_none`).
* `LocatedAnswers`, `located_answers`: the same of the log the check read, on a play the form accepts, with the
  record the `t`-th of the second stage's sessions' records the check read (`sessionRecords`).
* `LogAnswers`, `logAnswers_of`, `logAnswers_run`: the same of the play's own log, at any locator of `CoinMap` that
  agrees with the session-round locator at the sessions' rounds (`LocatesSessions`), so at the run's (`runLocate`,
  `locates_run`): the log the check read has the device's frames and answered bytes after releases that open no
  coin tree (`InteriorLogged`).

**Not established here**: that the digest the record gives a round is the SHA-512 of the message the session's
verifier reads at that round (the session's verifier's check of each round's bytes, and the link's roots as the
session's statement fixes them), and what the session's closing round answers.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Verity.Sampled.Binding (Read)
open Spot Spot.Challenger

/-! ## The digest a session's record gives a round -/

/-- The bytes a session's record names as its link's `root_F`. -/
def linkRootF (record : Json) : Option ByteArray :=
  match getD (getD record "link") "root_f" with
  | .str s => (Flock.unhex s).toOption
  | _ => none

/-- The tables' `root_B`s a session's record names in its link, each with its table's name. -/
def linkRootB (record : Json) : Array (String × ByteArray) :=
  (entries (getD (getD record "link") "root_b")).filterMap fun (kv : String × Json) => match kv.2 with
    | .str s => (Flock.unhex s).toOption.map (kv.1, ·)
    | _ => none

/-- **The digest a session's record gives round `k` of stream `stream`** of its coin tree: of `TREE_SESSION`, whose
round 0 is the link points, the SHA-512 of the link's `root_F` and each table's `root_B` in table-name order
(`Spot.Challenger.linkAfter`); of another stream, the bytes of round `k`'s `msg_sha512` in the record's first stream so
named, as `Spot.Challenger.recordRounds` counts its rounds; none where the record names no such bytes. -/
def recordDigest (record : Json) (stream : String) (k : ℕ) : Option ByteArray :=
  if stream == TREE_SESSION then
    (linkRootF record).map fun rootF => linkAfter rootF (linkRootB record)
  else
    match getD record "streams" with
    | .arr ss =>
      match ss.find? (fun x => match getD x "stream" with | .str n => n == stream | _ => false) with
      | some x =>
        match getD x "rounds" with
        | .arr rs => rs[k]?.bind fun r => match getD r "msg_sha512" with
          | .str h => (Flock.unhex h).toOption
          | _ => none
        | _ => none
      | none => none
    | _ => none

/-- **A record `sessionTreeReason` accepts, at any tables, has its link points opened after its link's digest**: its
coin tree's root is that of a tree of the log, and the log shows that tree's `TREE_SESSION` round 0 opened after the
SHA-512, in lowercase hex, of the link's `root_F` and each table's `root_B` in table-name order. -/
theorem link_after {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, Verity.Sampled.recordTree record = t.root ∧ ∃ o rootF, t.opening TREE_SESSION 0 = some o ∧
      linkRootF record = some rootF ∧ o.after = Flock.hexOf (linkAfter rootF (linkRootB record)) := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  rename_i t ht
  refine ⟨t, Array.mem_of_find?_eq_some ht, ?_, ?_⟩
  · have hp := Array.find?_some ht
    unfold Verity.Sampled.recordTree
    rw [Verity.Sampled.Trees.strIs hp]
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
  rename_i hlinkOk
  simp only [Bool.and_eq_true, beq_iff_eq] at hlinkOk
  have hF : linkRootF record = some rootF := by
    have h1 := congrArg Prod.fst htri
    unfold linkRootF
    exact h1
  exact ⟨o, rootF, rfl, hF, hlinkOk.1.2⟩

/-- The digest a record gives a round of `TREE_SESSION` is its link's. -/
theorem recordDigest_link {record : Json} {rootF : ByteArray} (h : linkRootF record = some rootF) (k : ℕ) :
    recordDigest record TREE_SESSION k = some (linkAfter rootF (linkRootB record)) := by
  unfold recordDigest
  rw [ite_eq_left (beq_self_eq_true _), h]
  rfl

/-- **A round a record gives a stream other than `TREE_SESSION` has the bytes of its `msg_sha512` for its digest**,
where each round of each of the record's streams is one the tree `t` shows opened after that round's `msg_sha512`
(as `Verity.Sampled.rounds_held` gives it of a record `sessionTreeReason` accepts): the tree shows the round opened,
and the record's digest for it is what that opening's `after` names. -/
theorem recordDigest_round {record : Json} {t : LoggedTree} {streams : Array Json}
    (hstr : getD record "streams" = .arr streams)
    (hall : ∀ s ∈ streams, ∀ name rounds, getD s "stream" = .str name → getD s "rounds" = .arr rounds →
      ∀ x ∈ rounds.zipIdx, (recordCoins? (getD x.1 "coins")).isSome ∧
        ∃ o, t.opening name x.2 = some o ∧ getD x.1 "msg_sha512" = .str o.after)
    {name : String} (hne : name ≠ TREE_SESSION) {k : ℕ} (hk : k < recordRounds record name) :
    ∃ o, t.opening name k = some o ∧ recordDigest record name k = (Flock.unhex o.after).toOption := by
  have hb : (name == TREE_SESSION) = false := beq_eq_false_iff_ne.2 hne
  unfold recordRounds at hk
  unfold recordDigest
  rw [hb] at hk ⊢
  simp only [Bool.false_eq_true, ↓reduceIte, hstr] at hk ⊢
  split at hk
  case h_2 => exact absurd hk (Nat.not_lt_zero _)
  rename_i x hfind
  split at hk
  case h_2 => exact absurd hk (Nat.not_lt_zero _)
  rename_i rs hrs
  have hp := Array.find?_some hfind
  have hx : x ∈ streams := Array.mem_of_find?_eq_some hfind
  split at hp
  case h_2 => cases hp
  rename_i n hn
  obtain rfl : n = name := beq_iff_eq.1 hp
  obtain ⟨-, o, ho, hmsg⟩ :=
    hall x hx n rs hn hrs (rs[k], k) (Array.mem_zipIdx_iff_getElem?.2 (Array.getElem?_eq_getElem hk))
  refine ⟨o, ho, ?_⟩
  simp only [Array.getElem?_eq_getElem hk, Option.bind_some]
  have hmsg' : getD rs[k] "msg_sha512" = .str o.after := hmsg
  rw [hmsg']

/-- A session's round past its first is of a stream other than `TREE_SESSION`. -/
theorem slotStream_ne_session (tab : String) (R : ℕ) {j : ℕ} (h : j ≠ 0) :
    (slotStream tab R j).1 ≠ TREE_SESSION := by
  unfold slotStream
  rw [ite_eq_right h]
  by_cases hR : j - 1 < R
  · rw [ite_eq_left hR]
    exact Verity.Sampled.repStream_ne_session tab 0
  · rw [ite_eq_right hR]
    exact Verity.Sampled.repStream_ne_session tab 1

/-! ## The read -/

section Answers

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Release} {trees : Array LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)} {tree : ℕ → ℕ}

/-- **A release of the read the session-round locator puts on a session's round answers the digest the session's
record gives that round**: where the members' check accepts the sessions' records against the read and the locator
puts a release on round `j` of session `t`, the bytes the release answers are the digest session `t`'s record gives
the stream and round `slotStream` gives `j` (`recordDigest`). -/
theorem MembersHeld.answers (H : MembersHeld lines trees log records tree)
    (hT : coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) {off K : ℕ → ℕ} {M : ℕ} {tab : ℕ → String}
    (hs : ∀ t, off (t + 1) = off t + K t) {pre : List Release} {r : Release} {post : List Release}
    (he : log = pre ++ r :: post) {t j : ℕ} (hj : j < K t)
    (hloc : sessLoc off K M tab pre r.frame = some (off t + j)) :
    ∃ x, records[t]? = some x ∧
      recordDigest x.2.2.1 (slotStream (tab t) (repsOf (K t)) j).1 (slotStream (tab t) (repsOf (K t)) j).2 =
        some r.answers := by
  obtain ⟨t', j', s, c, -, hf, hj', hn, hname, hk⟩ := H.key hT hR he hloc
  obtain ⟨rfl, rfl⟩ := off_sep hs hj hj' hn
  obtain ⟨x, hx, hlt⟩ := H.recorded hT hR hM hs he hj hloc
  refine ⟨x, hx, ?_⟩
  obtain ⟨l, ρ, name, k, hl, hro, hkey, hnm, hkk⟩ := Verity.Sampled.readLog_round hT hR he hf
  obtain rfl : name = (slotStream (tab t) (repsOf (K t)) j).1 := Option.some.inj (hnm.symm.trans hname)
  obtain rfl : k = (slotStream (tab t) (repsOf (K t)) j).2 := hkk.trans hk
  -- the tree the release opens is the one the session's record names
  obtain ⟨-, sp, hsp, -⟩ := hkey
  obtain ⟨u, hui, hur, -⟩ := heldTrees_getElem? hsp
  have hum : u ∈ trees := Array.mem_iff_getElem?.2 ⟨_, hui⟩
  have hnd := (Verity.Sampled.Lines.trees_log hT).nodup
  obtain ⟨ρx, hρx, hux⟩ := H.roots t x hx
  have eρ : ρ = ρx := Verity.Sampled.MemberForm.findIdx_fst (H.idx t ρx hux) hsp
  have hroot : getD (getD x.2.2.1 "coin_tree") "root" = Json.str u.root := by rw [hρx, hur, eρ]
  -- the members' check holds the record to that tree less its closing round
  obtain ⟨u', hu', hroot', _, -, -, hS⟩ := Verity.Sampled.MemberForm.memberTreeReason_none
    ((Verity.Sampled.MemberForm.membersTreeReason_none hM).1 x (Array.mem_iff_getElem?.2 ⟨t, hx⟩))
  have eu : u' = u := List.inj_on_of_nodup_map hnd (Array.mem_toList_iff.2 hu') (Array.mem_toList_iff.2 hum)
    (Json.str.inj (hroot'.symm.trans hroot))
  rw [eu] at hS
  -- the line the read gives the release: its `after` names the bytes the release answers
  obtain ⟨hor, -, d, hd, -, aft, haft, hun⟩ := hro
  rw [← hur] at hor
  by_cases h0 : j = 0
  · have e : slotStream (tab t) (repsOf (K t)) j = (TREE_SESSION, 0) := by
      unfold slotStream
      rw [ite_eq_left h0]
    have hor' : Verity.Sampled.OpensRound l u.root TREE_SESSION 0 := by
      rw [e] at hor
      exact hor
    rw [e]
    change recordDigest x.2.2.1 TREE_SESSION 0 = some r.answers
    obtain ⟨v, hv, -, o, rootF, ho, hF, hafter⟩ := link_after hS
    obtain rfl : v = u.unclosed := Array.mem_singleton.1 hv
    have ha := Verity.Sampled.opening_after hT hum (opening_unclosed ho) hl hor' hd
    obtain rfl : aft = o.after := Json.str.inj (haft.symm.trans ha)
    rw [hafter, Flock.Hex.unhex_hexOf] at hun
    rw [recordDigest_link hF, Except.ok.inj hun]
  · have hne := slotStream_ne_session (tab t) (repsOf (K t)) h0
    obtain ⟨v, hv, -, streams, hstr, hall⟩ := Verity.Sampled.rounds_held hS
    obtain rfl : v = u.unclosed := Array.mem_singleton.1 hv
    obtain ⟨o, ho, hdig⟩ := recordDigest_round hstr hall hne hlt
    have ha := Verity.Sampled.opening_after hT hum (opening_unclosed ho) hl hor hd
    obtain rfl : aft = o.after := Json.str.inj (haft.symm.trans ha)
    rw [hdig, hun]
    rfl

end Answers

/-! ## The log the check read -/

/-- **A release of the log the check read that is put on a session's round answers the digest the session's record
gives that round**: on a play the form accepts, where the session-round locator puts a release of the interior
challenger session's log, as the check read it, on round `k` of session `t`, the bytes the release answers are the
digest the `t`-th of the sessions' records the check read gives the stream and round `slotStream` gives `k`
(`sessionRecords`, `recordDigest`). -/
def LocatedAnswers (A : Live₂) (C : CaseOf) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → ∀ l, interiorRead C o = some l → ∀ a r b, l = a ++ r :: b →
    ∀ t k, k < SpotFlock.k₀ A.h t → locOf A tab a r.frame = some (SpotFlock.off A.h t + k) →
      ∃ recs rec, sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
        recordDigest rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).1
          (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).2 = some r.answers

/-- **`LocatedAnswers`, at the form's acceptance**: the accepted check ran the members' check on the lines the read
reads, at the sessions' records (`membersAccepted`), so `MembersHeld.answers` applies. -/
theorem located_answers (A : Live₂) (C : CaseOf) (tab : ℕ → String) (hsz : HeldSize) : LocatedAnswers A C tab := by
  intro o hacc l hl a r b hsplit t k hk hloc
  obtain ⟨label, after, ξ, lines, trees, records, hT, hR, hMR, hS⟩ :=
    membersAccepted (membersNamed C hsz) o hacc l hl
  obtain ⟨tree, H⟩ := membersHeld_of hT hR hMR
  unfold locOf at hloc
  obtain ⟨x, hx, hd⟩ := H.answers hT hR hMR (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h)
    (SpotFlock.off_succ A.h) hsplit hk hloc
  refine ⟨_, x.2.2.1, hS, ?_, hd⟩
  rw [List.getElem?_map, Array.getElem?_toList, hx]
  rfl

/-! ## The play's log -/

/-- **A release of the play's log that is put on a session's round answers the digest the session's record gives
that round**: on a play the form accepts whose log the check read is the device's (`Good`), where the locator `L`
puts a release of the play's log on round `k` of session `t`, the bytes the release answers are the digest the
`t`-th of the sessions' records the check read gives the stream and round `slotStream` gives `k`. -/
def LogAnswers (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (L : CoinMap.Locate) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ a rl b, o.log = a ++ rl :: b → ∀ t k, k < SpotFlock.k₀ A.h t →
    CoinMap.locAt L a rl.frame rl.answers = some (SpotFlock.off A.h t + k) →
      ∃ recs rec, sessionRecords C o = some recs ∧ recs[t]? = some rec ∧
        recordDigest rec (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).1
          (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) k).2 = some rl.answers

/-- The log the check read has the frames and the answered bytes of the device's releases it reads, one for one. -/
theorem answers_of_read {l l' : List Release} (hR : List.Forall₂ Read.Of (l.map Read.ofV2) l') :
    List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) l l' :=
  (List.forall₂_map_left_iff.1 hR).imp fun _ _ h => by
    obtain ⟨hf, ha, -⟩ := h
    exact ⟨hf, ha⟩

/-- **`LogAnswers`, from the log the check read** (`located_answers`), at a locator that agrees with the
session-round locator at the sessions' rounds: the play's log is the read log after releases of no coin tree, the same
frames and answered bytes one by one (`InteriorLogged`), so the release is on the round its like in the read log is on
(`locOf_logged`) and answers the bytes its like answers. -/
theorem logAnswers_of {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {L : CoinMap.Locate} {tab : ℕ → String}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hsz : HeldSize) (hI : InteriorLogged C Good) :
    LogAnswers A C Good L tab := by
  intro o hacc hgood a rl b hlog t k hk hloc
  obtain ⟨l, hl⟩ := interiorReads C hsz o hacc
  obtain ⟨q, l', hlog', hq, hF⟩ := hI o hacc hgood l hl
  have h1 : locOf A tab a rl.frame = some (SpotFlock.off A.h t + k) :=
    (hL a rl.frame rl.answers _ (groupOf_off A t k)).1 hloc
  obtain ⟨i, hi⟩ := openTree_of_locOf h1
  rw [hlog'] at hlog
  obtain ⟨a', rfl, hl''⟩ := split_past hlog (fun x hx => (hq x hx).1) hi
  subst hl''
  obtain ⟨la, x, lrest, rfl, hla, hx, -⟩ := forall₂_split_right (answers_of_read hF)
  have hxf : x.frame = rl.frame := hx.1
  have hxa : x.answers = rl.answers := hx.2
  have e1 : locOf A tab la x.frame = some (SpotFlock.off A.h t + k) := by
    rw [locOf_logged A tab hq (hla.imp fun _ _ h => h.1) hxf]
    exact h1
  obtain ⟨rs, rec, hrs, hrec, hd⟩ := located_answers A C tab hsz o hacc _ hl la x lrest rfl t k hk e1
  exact ⟨rs, rec, hrs, hrec, hxa ▸ hd⟩

/-- **`LogAnswers` at the run's locator** (`runLocate`, which agrees with the session-round locator at the sessions'
rounds: `locates_run`). -/
theorem logAnswers_run (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hsz : HeldSize) (hI : InteriorLogged C Good) : LogAnswers A C Good (runLocate A Ls tab) tab :=
  logAnswers_of (locates_run A Ls tab) hsz hI

end Verity.SampledTwo.SessionCoins
