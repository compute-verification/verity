module

public import Proofs.Properties.Verity.SampledTwo.SessionMiddle

/-!
# A session's last round

The last round of session `t`, round `SpotFlock.k₀ A.h t - 1`, is the round its verifier gives its verdict on. The
session-round locator puts no release on it (`SessionCoins.locOf_ne_last`). The challenger closed the session's tree
after the SHA-512 of the session's evidence (`Verity.Sampled.MemberForm.members_read`), and that closing is in the
play's log before the round. This module reads the closing's bytes off the log before the round and shows they are
the SHA-512 of the session's evidence.

* `lastOpened`, `closeOf`: the bytes the last opening of a tree in a log answers, and those of the `t`-th tree the
  log opens, in the order first opened.
* `MembersHeld.closing`: at the numbering of the sessions' trees `MembersHeld` holds, session `t`'s tree has an
  opening that answers the bytes its evidence's SHA-512 names in hex, with no opening of the tree after it.
* `MembersHeld.close_before`: that opening is before every release the session-round locator puts on a round at or
  past the session's last.
* `MembersHeld.closeOf_logged`: `closeOf` reads those bytes off a log of the same frames and answered bytes, after
  releases of no coin tree.
* `evidenceAt`, `LogCloses`, `logCloses_of`: on a play the form accepts whose log the check read is the device's,
  `closeOf` of the play's log before session `t`'s last round is the SHA-512 of the evidence of the `t`-th of the
  statements' sessions.
* `sessionMsgs_length`, `last_msg_of`: a session of `2 + 2 · repRounds` rounds reads a message at its last round.
* `showsAt_session_last`: `Messages.ShowsAt` at a session's last round of the form's reading, with the digest
  `closeOf` of the log before the round and the string the session's evidence.

**Not established here**: `Messages.Explains` at a session's last round. The message there is the session's
statement, its record's text and its proof files; the evidence gives back the record and the proof files, and the
statement is the statements' round's. `RecordsCover` and `MsgsOfRecords` are hypotheses, as in `SessionMiddle.lean`.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Claims
open Lean (Json)
open Spot Spot.Challenger

/-! ## Lists -/

theorem forall₂_mem_right {α β : Type} {R : α → β → Prop} {l : List α} {l' : List β} (h : List.Forall₂ R l l') :
    ∀ y ∈ l', ∃ x ∈ l, R x y := by
  induction h with
  | nil =>
    intro y hy
    cases hy
  | cons hab _ ih =>
    intro y hy
    rcases List.mem_cons.1 hy with rfl | hy
    · exact ⟨_, List.mem_cons_self, hab⟩
    · obtain ⟨x, hx, hR⟩ := ih y hy
      exact ⟨x, List.mem_cons_of_mem _ hx, hR⟩

/-! ## The bytes a tree's last opening answers -/

/-- **The bytes the last opening of tree `i` in a log answers**; none where the log doesn't open the tree. -/
def lastOpened (i : ℕ) (p : List Release) : Option ByteArray :=
  ((p.filter fun r => openTree r.frame == some i).getLast?).map (·.answers)

/-- **The bytes a session's closing answers, read off a log**: the bytes the last opening of the `t`-th tree the log
opens, in the order first opened, answers. -/
def closeOf (t : ℕ) (p : List Release) : Option ByteArray :=
  (treesOpened p)[t]?.bind fun i => lastOpened i p

/-- The last opening of tree `i` in a log is an opening of it that no opening of it follows. -/
theorem lastOpened_of {i : ℕ} (a : List Release) {c : Release} {b : List Release}
    (hc : openTree c.frame = some i) (hb : ∀ y ∈ b, openTree y.frame ≠ some i) :
    lastOpened i (a ++ c :: b) = some c.answers := by
  have hnil : (b.filter fun r => openTree r.frame == some i) = [] :=
    List.filter_eq_nil_iff.2 fun y hy h => hb y hy (eq_of_beq h)
  have hpos : (openTree c.frame == some i) = true := by
    rw [hc]
    exact beq_self_eq_true _
  have hcb : (c :: b).filter (fun r => openTree r.frame == some i) = [c] := by
    rw [List.filter_cons, hnil, ite_eq_left hpos]
  unfold lastOpened
  rw [List.filter_append, hcb, List.getLast?_append, List.getLast?_singleton]
  rfl

/-- A frame that is no opening of tree `i` opens another tree, or none. -/
theorem openTree_ne {fr : Verity.Frame} {i : ℕ} (h : ∀ s n, fr ≠ .opening i s n) : openTree fr ≠ some i := by
  intro e
  cases fr with
  | fresh _ => cases e
  | hello _ => cases e
  | opening j s n =>
    have hj : j = i := Option.some.inj e
    exact h s n (by rw [hj])

/-! ## The read -/

section Closing

variable {label : String} {after : ByteArray} {recs : List Json} {ξ : ByteArray} {lines : List String}
  {log : List Release} {trees : Array LoggedTree}
  {records : Array (String × Array (String × Option ℕ) × Json × String)} {tree : ℕ → ℕ}

/-- **A session's tree's last opening answers the bytes the session's evidence's SHA-512 names in hex**: where the
members' check accepts the sessions' records against the read, session `t`'s tree, at the numbering `MembersHeld`
holds, has an opening that answers those bytes, with no opening of the tree after it
(`Verity.Sampled.MemberForm.members_read`, whose numbering is this one at each session: both are the place of the
record's tree among the trees the lines hold). -/
theorem MembersHeld.closing (H : MembersHeld lines trees log records tree)
    (hT : coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log)
    (hM : membersTreeReason trees lines records = none) {t : ℕ}
    {x : String × Array (String × Option ℕ) × Json × String} (hx : records[t]? = some x) :
    ∃ Lc c L₂ s n, log = Lc ++ c :: L₂ ∧ c.frame = .opening (tree t) s n ∧
      Flock.unhex x.2.2.2 = .ok c.answers ∧ ∀ y ∈ L₂, ∀ s' n', y.frame ≠ .opening (tree t) s' n' := by
  obtain ⟨tree', h1, -, -⟩ := Verity.Sampled.MemberForm.members_read hT hR hM
  obtain ⟨u, -, hroot, hfi, -, Lc, c, L₂, s, n, hlog, hc, -, -, hun, hlast⟩ := h1 t x hx
  have hu : (Verity.Sampled.MemberForm.usedOf records)[t]? = some u.root := by
    rw [Verity.Sampled.MemberForm.usedOf_getElem? hx, hroot, Verity.Sampled.MemberForm.strOf_str]
  have e : tree' t = tree t := Option.some.inj (hfi.symm.trans (H.idx t u.root hu))
  rw [e] at hc hlast
  exact ⟨Lc, c, L₂, s, n, hlog, hc, hun, hlast⟩

/-- **A session's closing is before every release put on a round at or past the session's last**: where an opening
`c` of session `t`'s tree is in the read and the session-round locator puts the release `x` on a round at least
session `t`'s last, at a session of at least two rounds, `c` is before `x`. The release `x` opens a later session's
tree: no release is put on the last round itself (`locOf_ne_last`), and an earlier session's rounds are below it. An
opening of a later session's tree has no opening of session `t`'s tree after it (`MembersHeld.before`). -/
theorem MembersHeld.close_before (H : MembersHeld lines trees log records tree)
    (hT : coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log) (A : Live₂) (tab : ℕ → String) {t : ℕ}
    (ht : t < records.size) (hK : 2 ≤ SpotFlock.k₀ A.h t) {Lc : List Release} {c : Release} {L₂ : List Release}
    (hc : log = Lc ++ c :: L₂) {st n : ℕ} (hcf : c.frame = .opening (tree t) st n) {pre : List Release}
    {x : Release} {post : List Release} (he : log = pre ++ x :: post) {n₀ : ℕ}
    (hloc : locOf A tab pre x.frame = some n₀)
    (hn : SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) ≤ n₀) :
    ∃ mid, pre = Lc ++ c :: mid ∧ L₂ = mid ++ x :: post := by
  have hloc' := hloc
  unfold locOf at hloc'
  obtain ⟨t', j', s, cc, ht', hxf, hj', hn₀, -, -⟩ := H.key hT hR he hloc'
  have hlt : t < t' := by
    by_contra hle
    rcases Nat.lt_or_eq_of_le (Nat.le_of_not_lt hle) with h | h
    · have h1 := off_mono (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h) (SpotFlock.off_succ A.h)
        (show t' + 1 ≤ t from h)
      have h2 := SpotFlock.off_succ A.h t'
      omega
    · rw [h] at hn₀ hj'
      have e : n₀ = SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1) := by omega
      rw [e] at hloc
      exact locOf_ne_last A tab pre x.frame hK hloc
  rcases List.append_eq_append_iff.1 (hc.symm.trans he) with ⟨a', hpre, ha⟩ | ⟨c', hLc, hc'⟩
  · cases a' with
    | nil =>
      have e : c = x := (List.cons.inj ha).1
      rw [e, hxf] at hcf
      have := H.inj ht' ht (Verity.Frame.opening.inj hcf).1
      omega
    | cons y a'' =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, hL₂⟩ := List.cons.inj ha
      exact ⟨a'', hpre, hL₂⟩
  · cases c' with
    | nil =>
      have e : x = c := (List.cons.inj hc').1
      rw [← e, hxf] at hcf
      have := H.inj ht' ht (Verity.Frame.opening.inj hcf).1
      omega
    | cons y c'' =>
      rw [List.cons_append] at hc'
      obtain ⟨rfl, -⟩ := List.cons.inj hc'
      have := H.before Lc c L₂ hc t ht st n hcf _ (by rw [hLc]; simp) t' ht' s cc hxf
      omega

/-- **`closeOf` reads a session's closing's bytes off the device's log**: where the read is releases that end at an
opening `c` of session `t`'s tree, then releases `mid`, none an opening of that tree, then any others, and a log is
releases `q` of no coin tree and then releases of the same frames and answered bytes as those up to `mid`'s end,
`closeOf t` of that log is the bytes `c` answers. The trees the log opens are the read's (`same_append_of_frames`),
the sessions' first few in order (`MembersHeld.prefix`), so the `t`-th is session `t`'s. -/
theorem MembersHeld.closeOf_logged (H : MembersHeld lines trees log records tree)
    (hT : coinTreesOf lines = .ok trees)
    (hR : Verity.Sampled.readLog label after recs ξ lines = some log) {t : ℕ} (ht : t < records.size)
    {Lc : List Release} {c : Release} {mid post : List Release} (hlog : log = (Lc ++ c :: mid) ++ post)
    {st n : ℕ} (hcf : c.frame = .opening (tree t) st n)
    (hmid : ∀ y ∈ mid, ∀ s' n', y.frame ≠ .opening (tree t) s' n') {q a' : List Release}
    (hq : ∀ r ∈ q, openTree r.frame = none ∧ ∀ sp, r.frame ≠ .hello sp)
    (hF : List.Forall₂ (fun x y => x.frame = y.frame ∧ x.answers = y.answers) (Lc ++ c :: mid) a') :
    closeOf t (q ++ a') = some c.answers := by
  have hS : Same [] q := by simpa using same_append (pre := []) hq
  have hS' : Same (Lc ++ c :: mid) (q ++ a') := by
    simpa using same_append_of_frames (hF.imp fun _ _ h => h.1) hS
  obtain ⟨hS1, -, -⟩ := hS'
  obtain ⟨m, hm, hpre⟩ := H.prefix hT hR (Lc ++ c :: mid) post hlog
  have hmem : tree t ∈ treesOpened (Lc ++ c :: mid) := mem_treesOpened (q := c) (by simp) hcf
  rw [hpre] at hmem
  obtain ⟨a, ha, hfa⟩ := List.mem_map.1 hmem
  have ha' := List.mem_range.1 ha
  have hat : a = t := H.inj (by omega) ht hfa
  have htm : t < m := by omega
  have hget : (treesOpened (q ++ a'))[t]? = some (tree t) := by
    rw [← hS1, hpre, List.getElem?_map, List.getElem?_range htm]
    rfl
  obtain ⟨Lc', c', mid', rfl, -, hcc, hmm⟩ := forall₂_split_left hF
  have hopen : openTree c'.frame = some (tree t) := congrArg openTree (hcc.1.symm.trans hcf)
  have hno : ∀ y ∈ mid', openTree y.frame ≠ some (tree t) := by
    intro y hy
    obtain ⟨w, hw, hwy⟩ := forall₂_mem_right hmm y hy
    rw [← hwy.1]
    exact openTree_ne (hmid w hw)
  have hlast : lastOpened (tree t) (q ++ (Lc' ++ c' :: mid')) = some c.answers := by
    rw [← List.append_assoc, lastOpened_of (q ++ Lc') hopen hno, hcc.2]
  unfold closeOf
  rw [hget]
  exact hlast

end Closing

/-! ## The play's log -/

/-- **The evidence of the `t`-th of the statements' sessions**, on the case of the play `o`: the bytes
`Spot.OneStageZk.sessionEvidence` gives the session on the case's blobs, the statements' sessions in the order
`sessionRecords` reads their records in. -/
def evidenceAt (C : CaseOf) (o : Exchange) (t : ℕ) : Option ByteArray := do
  let k ← C o
  let e ← (VStar.Recursive.SampledTwo.evidenceOf k.evidence).toOption
  let s ← (e.statements.toList.flatMap (·.sessions.toList))[t]?
  (Spot.OneStageZk.sessionEvidence k.blobs s).toOption

/-- **The log before a session's last round holds the session's closing**: on a play the form accepts whose log the
check read is the device's (`Good`), at a template `t` whose session has at least two rounds, the `t`-th of the
statements' sessions has evidence on the case, and `closeOf t` of the play's log before the session's last round is
that evidence's SHA-512. -/
def LogCloses (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ t, t < A.h.ms.length → 2 ≤ SpotFlock.k₀ A.h t →
    ∃ ev, evidenceAt C o t = some ev ∧
      closeOf t (Messages.logBefore L o (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1))).1 =
        some (Flock.Sha512.hash ev)

/-- **`LogCloses`, from the read** (`MembersHeld.closing`), at a locator that agrees with the session-round locator
at the sessions' rounds, where the check read a session record for each template (`RecordsCover`): the play's log is
the read log after releases of no coin tree, the same frames and answered bytes one by one (`InteriorLogged`). The
play's log before the round is the whole log, or ends before a release put on a round at or past the session's last,
whose like in the read log the session's closing is before (`MembersHeld.close_before`); either way `closeOf` reads
the closing's bytes off it (`MembersHeld.closeOf_logged`). -/
theorem logCloses_of {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {L : CoinMap.Locate} {tab : ℕ → String}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hsz : HeldSize) (hI : InteriorLogged C Good)
    (hcov : RecordsCover A C Good) : LogCloses A C Good L := by
  intro o hacc hgood t ht hK
  obtain ⟨recs, rc, hrecs, hrc⟩ := hcov o hacc hgood t ht
  obtain ⟨l, hl⟩ := interiorReads C hsz o hacc
  obtain ⟨k, e, label, after, ξ, lines, trees, records, hk, he, hT, hR, hM, hN⟩ :=
    membersNamed C hsz o hacc l hl
  have hS : sessionRecords C o = some (records.toList.map (·.2.2.1)) :=
    sessionRecords_of hk he (named_records hN)
  obtain rfl : records.toList.map (·.2.2.1) = recs := Option.some.inj (hS.symm.trans hrecs)
  obtain ⟨x, hx, -⟩ : ∃ x : String × Array (String × Option ℕ) × Json × String,
      records[t]? = some x ∧ x.2.2.1 = rc := by
    rw [List.getElem?_map, Array.getElem?_toList] at hrc
    cases hy : records[t]? with
    | none => rw [hy] at hrc; simp at hrc
    | some y => rw [hy] at hrc; exact ⟨y, rfl, Option.some.inj hrc⟩
  have hx' : records.toList[t]? = some x := by
    rw [Array.getElem?_toList]
    exact hx
  obtain ⟨s, hs, -, ev, hev, hxe⟩ := forall₂_getElem?_right hN hx'
  have htr : t < records.size := (Array.getElem?_eq_some_iff.1 hx).1
  obtain ⟨tree, H⟩ := membersHeld_of hT hR hM
  obtain ⟨Lc, c, L₂, st, n, hlog, hcf, hun, hlast⟩ := H.closing hT hR hM hx
  have hans : c.answers = Flock.Sha512.hash ev := Verity.Sampled.MemberForm.answers_of_hex hun hxe
  refine ⟨ev, ?_, ?_⟩
  · unfold evidenceAt
    simp only [hk, he, hs, hev, Option.bind_eq_bind, Option.bind_some, Except.toOption]
  obtain ⟨q, l', hplay, hq, hF⟩ := hI o hacc hgood l hl
  have hFA := answers_of_read hF
  rw [← hans]
  show closeOf t (CoinMap.before L (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) [] o.log).1 = some c.answers
  rcases CoinMap.before_split L (i := SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) [] o.log with
    ⟨-, h1⟩ | ⟨rl, rest, j, hsplit, -, hj, hij⟩
  · rw [h1, hplay]
    rw [hlog] at hFA
    exact H.closeOf_logged hT hR htr (post := []) (by rw [List.append_nil]; exact hlog) hcf hlast hq hFA
  · rw [List.nil_append] at hj
    have hgi := groupOf_off A t (SpotFlock.k₀ A.h t - 1)
    have hsi := stage_lt_of_sessions hgi
    have hgj : groupOf A j = .sessions := sessions_of_lt (by omega)
    have h2 := (hL _ rl.frame rl.answers j hgj).1 hj
    obtain ⟨i, hi⟩ := openTree_of_locOf h2
    obtain ⟨a', hpre, hl''⟩ := split_past (hplay.symm.trans hsplit) (fun x hx => (hq x hx).1) hi
    subst hl''
    obtain ⟨la, y, lrest, rfl, hla, hy, -⟩ := forall₂_split_right hFA
    have e1 : locOf A tab la y.frame = some j := by
      rw [locOf_logged A tab hq (hla.imp fun _ _ h => h.1) hy.1, ← hpre]
      exact h2
    obtain ⟨mid, rfl, hL₂⟩ := H.close_before hT hR A tab htr hK hlog hcf rfl e1 hij
    rw [hpre]
    refine H.closeOf_logged hT hR htr (post := y :: lrest) rfl hcf (fun w hw => hlast w ?_) hq hla
    rw [hL₂]
    exact List.mem_append_left _ hw

/-- **`LogCloses` at the run's locator** (`runLocate`, which agrees with the session-round locator at the sessions'
rounds: `locates_run`). -/
theorem logCloses_run (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hsz : HeldSize) (hI : InteriorLogged C Good) (hcov : RecordsCover A C Good) :
    LogCloses A C Good (runLocate A Ls tab) :=
  logCloses_of (locates_run A Ls tab) hsz hI hcov

/-! ## The last message -/

/-- **A one-table session the masked verdict accepts reads `2 + 2 · repRounds` messages**: the link's, one a round
of each of the two reps, and the last. -/
theorem sessionMsgs_length {x : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {st : Flock.Stmt}
    {record : String} {proofs : Array ByteArray} {T : String} {ms : List Msg}
    (hs : FlockDeploy.setupOf true x = some (s, spec, #[st]))
    (hv : Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok ()) (hT : spec.tables = #[T])
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms) :
    ms.length = 2 + 2 * FlockDeploy.repRounds true spec st := by
  obtain ⟨lv, mids, hlv, hmids, rfl⟩ := SessionMsgs.sessionMsgs_some hm
  have hl := SpotFlock.ok_of_toOption hlv
  obtain ⟨h2, _, -, hreps⟩ := session_reps hs hv hT hl
  obtain ⟨hs0, -⟩ := hreps 0 (by omega)
  obtain ⟨hs1, -⟩ := hreps 1 (by omega)
  have hlen : mids.length = (lv.streams.toList.flatMap Array.toList).length :=
    (SessionMsgs.mapM_some _ hmids).1
  have hflen : (lv.streams.toList.flatMap Array.toList).length = 2 * FlockDeploy.repRounds true spec st := by
    rw [flat_two h2, List.length_append, Array.length_toList, Array.length_toList, hs0, hs1]
    omega
  rw [List.length_append, List.length_cons, List.length_singleton, hlen, hflen]
  omega

/-- **Each session has at least two rounds and the session reading reads a message at its last**, from
`RecordsCover` and `MsgsOfRecords` (`sessionMsgs_length`). -/
theorem last_msg_of {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Exchange → Prop} {tab : ℕ → String}
    (hcov : RecordsCover A C Good) (hmsgs : MsgsOfRecords A C S Good tab) {t : ℕ} (ht : t < A.h.ms.length)
    {o : Exchange} (hacc : FormAccepts C o) (hgood : Good o) :
    2 ≤ SpotFlock.k₀ A.h t ∧ ∃ m, S.msg o t (SpotFlock.k₀ A.h t - 1) = some m := by
  obtain ⟨recs, rc, hrecs, hrc⟩ := hcov o hacc hgood t ht
  obtain ⟨x, s, spec, st, record, proofs, ms, hs, hv, -, hT, hk, -, hm, hS⟩ :=
    hmsgs o hacc hgood recs rc t ht hrecs hrc
  have hlen := sessionMsgs_length hs hv hT hm
  have hi : SpotFlock.k₀ A.h t - 1 < ms.length := by omega
  refine ⟨by omega, ms[SpotFlock.k₀ A.h t - 1]'hi, ?_⟩
  rw [hS]
  exact List.getElem?_eq_getElem hi

end Verity.SampledTwo.SessionCoins

namespace Verity.SampledTwo

open Claims

/-! ## `ShowsAt` at a session's last round -/

/-- **A session's last round of the form's reading shows the session's evidence**: at the last round of template
`t`'s session, where the digest a play fixes for the round is the bytes the session's closing answers, read off the
log before the round (`SessionCoins.closeOf`), and the string it shows is the evidence of the `t`-th of the
statements' sessions (`SessionCoins.evidenceAt`), an accepted play whose log the check read is the device's shows a
string that has its digest (`SessionCoins.LogCloses`). -/
theorem showsAt_session_last (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hClose : SessionCoins.LogCloses A C Good L) (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) {t : ℕ}
    (ht : t < A.h.ms.length) (hK : 2 ≤ SpotFlock.k₀ A.h t)
    (hmsg : ∀ o, FormAccepts C o → Good o → ∃ m, S.msg o t (SpotFlock.k₀ A.h t - 1) = some m)
    (hd : ∀ o, FormAccepts C o → Good o → Dg o (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) =
      (SessionCoins.closeOf t
        (Messages.logBefore L o (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1))).1).map
          fun b => [b.data.toList])
    (hr : ∀ o, FormAccepts C o → Good o → raw o (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) =
      (SessionCoins.evidenceAt C o t).map fun b => [b.data.toList]) :
    Messages.ShowsAt (formReading A C S dev) (FormAccepts C) Good Dg raw
      (SpotFlock.off A.h t + (SpotFlock.k₀ A.h t - 1)) := by
  intro o hA hG
  obtain ⟨ev, hev, hclose⟩ := hClose o hA hG t ht hK
  obtain ⟨m, hm⟩ := hmsg o hA hG
  have hlt : SpotFlock.k₀ A.h t - 1 < SpotFlock.k₀ A.h t := by omega
  refine ⟨⟨m, ?_⟩, [(Flock.Sha512.hash ev).data.toList], [ev.data.toList], ?_, ?_, ?_⟩
  · rw [Messages.formMsg_session A C S dev o ht (Nat.le_add_right _ _) (Nat.add_lt_add_left hlt _),
      Nat.add_sub_cancel_left]
    exact hm
  · exact (hd o hA hG).trans (congrArg (Option.map fun b : ByteArray => [b.data.toList]) hclose)
  · exact (hr o hA hG).trans (congrArg (Option.map fun b : ByteArray => [b.data.toList]) hev)
  · exact congrArg (fun m : Msg => [m]) (Verity.Sampled.sha512Msg_bytes ev)

end Verity.SampledTwo
