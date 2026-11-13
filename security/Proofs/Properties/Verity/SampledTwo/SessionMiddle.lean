module

public import Proofs.Properties.Verity.SampledTwo.SessionAnswers
public import Proofs.Properties.Verity.SampledTwo.LogStrings
public import Proofs.Properties.Verity.SampledTwo.SessionAsk
public import Proofs.Properties.Verity.SampledTwo.SessionReplay

/-!
# A session's rounds after its link and before its last message

Round `j` of session `t`, at `1 ≤ j` and `j + 1 < SpotFlock.k₀ A.h t`, is a round of one of the session's two reps.
The session's verifier reads there the bytes the session's record retained for that round, after the tag `0`
(`SessionMsgs.sessionMsgs`). The challenger opened the round after the digest the record gives it
(`SessionCoins.LogAnswers`). This module shows that digest is the SHA-512 of the retained bytes, so the round shows a
string that has its digest, and the round's message is that string after the tag.

* `recordDigest_decode`, `recordDigest_retained`: the digest a record gives a round of a stream other than
  `TREE_SESSION` is the one `Flock.Record.decode` reads for it, which under SHA-512 digests is the SHA-512 of the
  bytes the round retained.
* `session_reps`: the record of a one-table session the masked verdict accepts has two rep streams, each of the
  rep's rounds, each round's digest that of its retained bytes.
* `slot_lt_rounds`, `mid_digest`: at such a session, each round before the last is a round its record gives, and the
  digest the record gives a round after the link's is the SHA-512 of the message the session reads there, less its
  tag.
* `RoundsRecorded`, `LogOpened`, `logOpened_of`: where each session's record gives coins of each of its rounds before
  the last, the play's log has a release the locator puts on that round.
* `MidDigests`, `midDigests_of`, `roundsRecorded_of`: both from `RecordsCover` and `MsgsOfRecords`.
* `showsAt_session_mid`, `explains_session_mid`: `Messages.ShowsAt` and `Messages.Explains` at such a round of the
  form's reading, with the digest the bytes the release that ends the log before the round answers
  (`Messages.endAnswers`) and the string the round's message less its tag.

**Not established here**: `RecordsCover`, that the check read a session record for each template, and
`MsgsOfRecords`, that the messages the session reading reads of template `t`'s session are those of the `t`-th
record the check read, at a one-table SHA-512 setup the masked verdict accepts, of `2 + 2 · repRounds` rounds. Both
are hypotheses here. Nor are `Messages.ShowsAt` and `Messages.Explains` at a session's first round or its last.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Claims
open Lean (Json)
open Verity.Sampled (repStreamName)
open Spot Spot.Challenger

/-! ## Lists -/

theorem forall₂_split_left {α β : Type} {R : α → β → Prop} {a : List α} {x : α} {rest : List α} :
    ∀ {l' : List β}, List.Forall₂ R (a ++ x :: rest) l' →
      ∃ a' y rest', l' = a' ++ y :: rest' ∧ List.Forall₂ R a a' ∧ R x y ∧ List.Forall₂ R rest rest' := by
  induction a with
  | nil =>
    intro l' h
    rw [List.nil_append] at h
    cases h with
    | cons hw h₁ => exact ⟨[], _, _, rfl, List.Forall₂.nil, hw, h₁⟩
  | cons w a ih =>
    intro l' h
    rw [List.cons_append] at h
    cases h with
    | cons hw h₁ =>
      obtain ⟨a', y, rest', rfl, ha, hy, hr⟩ := ih h₁
      exact ⟨_ :: a', y, rest', rfl, List.Forall₂.cons hw ha, hy, hr⟩

theorem some_of_getD_cons {α : Type} {x : Option (List α)} {a : α} {l : List α} (h : x.getD [] = a :: l) :
    x = some (a :: l) := by
  cases x with
  | none => cases h
  | some m => exact congrArg some h

theorem flat_two {α : Type} {xs : Array (Array α)} (h : xs.size = 2) :
    xs.toList.flatMap Array.toList = (xs[0]!).toList ++ (xs[1]!).toList := by
  obtain ⟨l⟩ := xs
  match l, h with
  | [a, b], _ =>
    show [a, b].flatMap Array.toList = a.toList ++ b.toList
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]

theorem getElem?_flat_two {α : Type} {xs : Array (Array α)} (h : xs.size = 2) {R : ℕ}
    (h0 : (xs[0]!).size = R) (i : ℕ) :
    (xs.toList.flatMap Array.toList)[i]? = if i < R then (xs[0]!)[i]? else (xs[1]!)[i - R]? := by
  rw [flat_two h]
  by_cases hi : i < R
  · rw [ite_eq_left hi, List.getElem?_append_left (by rw [Array.length_toList, h0]; exact hi),
      Array.getElem?_toList]
  · rw [ite_eq_right hi, List.getElem?_append_right (by rw [Array.length_toList, h0]; omega),
      Array.length_toList, h0, Array.getElem?_toList]

theorem getElem?_mid {α : Type} {a f : α} {mids : List α} {i : ℕ} (hi : i < mids.length) :
    (a :: mids ++ [f])[i + 1]? = mids[i]? := by
  rw [List.getElem?_append_left (by rw [List.length_cons]; omega), List.getElem?_cons_succ]

theorem slotStream_succ (tab : String) (R i : ℕ) :
    slotStream tab R (i + 1) = if i < R then (repStreamName tab 0, i) else (repStreamName tab 1, i - R) := by
  unfold slotStream
  rw [ite_eq_right (Nat.succ_ne_zero i), Nat.add_sub_cancel]

/-! ## The digest a record gives a rep's round -/

/-- **The digest a record gives a round of a stream other than `TREE_SESSION` is the one `Flock.Record.decode` reads
for it**, under SHA-512 digests: the record's first stream so named is the one the decoded record finds by that name,
and the round's `msg_sha512` is the hex of the digest the decoded round has. -/
theorem recordDigest_decode {spec : Flock.SessionSpec} {text : String} {sess : Flock.RunRecord} {J : Json}
    (h512 : spec.sha512Digests = true) (hd : Flock.Record.decode spec text = .ok sess)
    (hJ : Flock.JsonParse.parse text = .ok J) {name : String} (hne : name ≠ TREE_SESSION)
    {y : Flock.StreamRec} (hy : sess.streams.find? (·.name == name) = some y) {k : ℕ}
    (hk : k < y.rounds.size) : recordDigest J name k = some y.rounds[k].sha := by
  obtain ⟨j, recs, hp, hrecs, hstr⟩ := Verity.Sampled.decode_sha hd
  rw [hJ] at hp
  obtain rfl := Except.ok.inj hp
  have hy' : sess.streams.toList.find? (·.name == name) = some y := by rw [Array.find?_toList]; exact hy
  obtain ⟨x, ⟨-, rs, hrs, hrds⟩, hx⟩ := Claims.Bridge.Custody.forall₂_find?
    (p := fun (y : Flock.StreamRec) => y.name == name)
    (q := fun (x : Json) => (Flock.Record.str x "stream").toOption == some name)
    (fun a b hab => by obtain ⟨h1, -⟩ := hab; rw [h1]; rfl) (Verity.Sampled.forall₂_swap hstr) hy'
  have hlen : rs.size = y.rounds.size := by simpa using hrds.length_eq
  have hrk : k < rs.size := by omega
  obtain ⟨s, hs, hu⟩ := Claims.Bridge.Custody.forall₂_getElem hrds hrk
  rw [getElem!_pos y.rounds k hk] at hu
  have hdk : spec.dk = "sha512" := by
    unfold Flock.SessionSpec.dk
    simp [h512]
  rw [show s!"msg_{spec.dk}" = "msg_sha512" by simp [hdk, toString]] at hs
  have hx' : recs.find? (fun x => (Flock.Record.str x "stream").toOption == some name) = some x := by
    rw [← Array.find?_toList]; exact hx
  have hb : (name == TREE_SESSION) = false := beq_eq_false_iff_ne.2 hne
  unfold recordDigest
  rw [hb]
  simp only [Bool.false_eq_true, ↓reduceIte, Verity.Sampled.arr_getD hrecs]
  rw [Verity.Sampled.find?_congr (q' := fun x => (Flock.Record.str x "stream").toOption == some name) ?_, hx']
  · simp only [Verity.Sampled.arr_getD hrs, Array.getElem?_eq_getElem hrk, Option.bind_some,
      Verity.Sampled.str_getD hs]
    exact congrArg Except.toOption hu
  · intro a ha
    obtain ⟨y', -, hy''⟩ := Claims.Bridge.Custody.forall₂_mem_left hstr a (Array.mem_toList_iff.2 ha)
    rw [Verity.Sampled.str_getD hy''.1, hy''.1]
    rfl

/-- **The digest a record gives a round that retained its bytes is the SHA-512 of those bytes**, under SHA-512
digests: of a stream other than `TREE_SESSION` whose decoded rounds are, one for one, the rounds `rounds` with their
bytes, each retained round's digest that of its bytes (as `Claims.Bridge.Custody.decode_live` gives of the live
session's rep streams). -/
theorem recordDigest_retained {spec : Flock.SessionSpec} {text : String} {sess : Flock.RunRecord} {J : Json}
    (h512 : spec.sha512Digests = true) (hd : Flock.Record.decode spec text = .ok sess)
    (hJ : Flock.JsonParse.parse text = .ok J) {name : String} (hne : name ≠ TREE_SESSION)
    {y : Flock.StreamRec} (hy : sess.streams.find? (·.name == name) = some y) {rounds : Array Flock.Round}
    (hrds : List.Forall₂ (fun (rd rd' : Flock.Round) => rd.bytes = rd'.bytes ∧
      ∀ b, rd.bytes = some b → rd.sha = spec.dg b) y.rounds.toList rounds.toList)
    {k : ℕ} {r : Flock.Round} (hr : rounds[k]? = some r) {b : ByteArray} (hb : r.bytes = some b) :
    (recordDigest J name k).map (·.data.toList) = some (Verity.sha512Msg b.data.toList) := by
  rw [← Array.getElem?_toList] at hr
  obtain ⟨rd, hrd, hbytes, hsha⟩ := forall₂_getElem?_right hrds hr
  rw [Array.getElem?_toList] at hrd
  obtain ⟨hky, rfl⟩ := Array.getElem?_eq_some_iff.1 hrd
  have hdg : spec.dg b = Flock.Sha512.hash b := by
    unfold Flock.SessionSpec.dg
    simp [h512]
  rw [recordDigest_decode h512 hd hJ hne hy hky, hsha b (hbytes.trans hb), hdg]
  exact congrArg some (Verity.Sampled.sha512Msg_bytes b).symm

/-! ## A one-table session's record -/

/-- **The record of a one-table session the masked verdict accepts has two rep streams, each of the rep's rounds**:
the live session has the two, each of `FlockDeploy.repRounds` rounds (`SessionReplay.streams_even`), and the decoded
record finds each by the table's rep stream name, with the live rounds' bytes, each retained round's digest that of its
bytes (`Claims.Bridge.Custody.decode_live`). -/
theorem session_reps {x : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {st : Flock.Stmt}
    {record : String} {proofs : Array ByteArray} {lv : FlockDeploy.Live} {T : String}
    (hs : FlockDeploy.setupOf true x = some (s, spec, #[st]))
    (hv : Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok ()) (hT : spec.tables = #[T])
    (hl : FlockDeploy.liveOf spec record = .ok lv) :
    lv.streams.size = 2 ∧ ∃ sess, Flock.Record.decode spec record = .ok sess ∧ ∀ q, q < 2 →
      (lv.streams[q]!).size = FlockDeploy.repRounds true spec st ∧
      ∃ y, sess.streams.find? (·.name == repStreamName T q) = some y ∧
        List.Forall₂ (fun (rd rd' : Flock.Round) => rd.bytes = rd'.bytes ∧
          ∀ b, rd.bytes = some b → rd.sha = spec.dg b) y.rounds.toList (lv.streams[q]!).toList := by
  obtain ⟨sess, hd⟩ := Verity.Sampled.zkVerify_decode hv
  obtain ⟨hsz, hrounds⟩ := SessionReplay.streams_even hs hv hl
  have h2 : lv.streams.size = 2 := hsz
  obtain ⟨-, -, -, hstreams⟩ := Claims.Bridge.Custody.decode_live hd hl
  have htab := Verity.Sampled.repStreams_table hT
  have hrep : spec.repStreams.size = 2 := by
    have h := congrArg List.length htab
    rw [Array.length_toList] at h
    exact h
  refine ⟨h2, sess, hd, fun q hq => ?_⟩
  have hq' : q < spec.repStreams.size := by omega
  obtain ⟨y, hy, hrds⟩ := hstreams q hq'
  have hname : spec.repStreams[q] = (0, q, repStreamName T q) := by
    have h1 : spec.repStreams.toList[q]? = some spec.repStreams[q] := by
      rw [Array.getElem?_toList, Array.getElem?_eq_getElem hq']
    rw [htab] at h1
    obtain rfl | rfl : q = 0 ∨ q = 1 := by omega
    · exact (Option.some.inj h1).symm
    · exact (Option.some.inj h1).symm
  rw [hname] at hy
  refine ⟨?_, y, hy, hrds.imp fun _ _ h => ⟨h.2.2.1, h.2.2.2⟩⟩
  have hql : q < lv.streams.size := by omega
  rw [getElem!_pos lv.streams q hql, hrounds q hql]
  obtain rfl | rfl : q = 0 ∨ q = 1 := by omega
  · rfl
  · rfl

/-- **Each round before the last of a one-table session the masked verdict accepts is a round its record gives**: the
link's is round 0 of `TREE_SESSION`, and round `j ≥ 1` is a round of the rep stream `slotStream` gives it, which has the
rep's rounds (`session_reps`, `Verity.Sampled.recordRounds_decode`). -/
theorem slot_lt_rounds {x : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {st : Flock.Stmt}
    {record : String} {proofs : Array ByteArray} {T : String} {ms : List Msg} {J : Json}
    (hs : FlockDeploy.setupOf true x = some (s, spec, #[st]))
    (hv : Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok ()) (hT : spec.tables = #[T])
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms)
    (hJ : Flock.JsonParse.parse record = .ok J) {j : ℕ} (hj : j ≤ 2 * FlockDeploy.repRounds true spec st) :
    (slotStream T (FlockDeploy.repRounds true spec st) j).2 <
      Verity.Sampled.recordRounds J (slotStream T (FlockDeploy.repRounds true spec st) j).1 := by
  obtain ⟨lv, mids, hlv, -, -⟩ := SessionMsgs.sessionMsgs_some hm
  have hl := SpotFlock.ok_of_toOption hlv
  obtain ⟨-, sess, hd, hreps⟩ := session_reps hs hv hT hl
  have hcount : ∀ q, q < 2 → Verity.Sampled.recordRounds J (repStreamName T q) =
      FlockDeploy.repRounds true spec st := by
    intro q hq
    obtain ⟨hsize, y, hy, hrds⟩ := hreps q hq
    have hlen := hrds.length_eq
    rw [Array.length_toList, Array.length_toList] at hlen
    rw [Verity.Sampled.recordRounds_decode hd hJ (Verity.Sampled.repStreamName_ne T q) hy, hlen, hsize]
  cases j with
  | zero =>
    have e : slotStream T (FlockDeploy.repRounds true spec st) 0 = (TREE_SESSION, 0) := by
      unfold slotStream
      rw [ite_eq_left rfl]
    rw [e]
    show 0 < Verity.Sampled.recordRounds J TREE_SESSION
    rw [Verity.Sampled.recordRounds, ite_eq_left (beq_self_eq_true _)]
    exact Nat.one_pos
  | succ i =>
    rw [slotStream_succ]
    by_cases hi : i < FlockDeploy.repRounds true spec st
    · rw [ite_eq_left hi]
      show i < Verity.Sampled.recordRounds J (repStreamName T 0)
      rw [hcount 0 (by omega)]
      exact hi
    · rw [ite_eq_right hi]
      show i - FlockDeploy.repRounds true spec st < Verity.Sampled.recordRounds J (repStreamName T 1)
      rw [hcount 1 (by omega)]
      omega

/-- **The digest the record gives a rep's round is the SHA-512 of the message the session reads there, less its
tag**, at a one-table session the masked verdict accepts under SHA-512 digests: the session's message at round `j`,
after the link's and before the last, is the tag `0` and the bytes the `(j - 1)`-th of the two rep streams' rounds
retained (`SessionMsgs.sessionMsgs`), and that round is the one `slotStream` gives `j` (`recordDigest_retained`). -/
theorem mid_digest {x : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {st : Flock.Stmt}
    {record : String} {proofs : Array ByteArray} {T : String} {ms : List Msg} {J : Json}
    (hs : FlockDeploy.setupOf true x = some (s, spec, #[st]))
    (hv : Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok ())
    (h512 : spec.sha512Digests = true) (hT : spec.tables = #[T])
    (hm : SessionMsgs.sessionMsgs spec x record proofs = some ms)
    (hJ : Flock.JsonParse.parse record = .ok J) {j : ℕ} (h0 : 0 < j)
    (hj : j ≤ 2 * FlockDeploy.repRounds true spec st) {u : Msg} (hu : ms[j]? = some (0 :: u)) :
    (recordDigest J (slotStream T (FlockDeploy.repRounds true spec st) j).1
      (slotStream T (FlockDeploy.repRounds true spec st) j).2).map (·.data.toList) =
        some (Verity.sha512Msg u) := by
  obtain ⟨lv, mids, hlv, hmids, rfl⟩ := SessionMsgs.sessionMsgs_some hm
  have hl := SpotFlock.ok_of_toOption hlv
  obtain ⟨h2, sess, hd, hreps⟩ := session_reps hs hv hT hl
  obtain ⟨hs0, y0, hy0, hrds0⟩ := hreps 0 (by omega)
  obtain ⟨hs1, y1, hy1, hrds1⟩ := hreps 1 (by omega)
  obtain ⟨i, rfl⟩ : ∃ i, j = i + 1 := ⟨j - 1, by omega⟩
  have hlen : mids.length = (lv.streams.toList.flatMap Array.toList).length :=
    (SessionMsgs.mapM_some _ hmids).1
  have hflen : (lv.streams.toList.flatMap Array.toList).length = 2 * FlockDeploy.repRounds true spec st := by
    rw [flat_two h2, List.length_append, Array.length_toList, Array.length_toList, hs0, hs1]
    omega
  have hi : i < mids.length := by omega
  rw [getElem?_mid hi] at hu
  obtain ⟨r, hr, hfr⟩ :=
    Option.bind_eq_some_iff.1 ((Verity.Sampled.LinkRead.mapM_getElem?' hmids i).symm.trans hu)
  have hfr' : r.bytes.map (fun b => (0 : UInt8) :: b.data.toList) = some (0 :: u) := hfr
  obtain ⟨b, hbytes, hbu⟩ := Option.map_eq_some_iff.1 hfr'
  obtain rfl : b.data.toList = u := (List.cons.inj hbu).2
  rw [getElem?_flat_two h2 hs0] at hr
  rw [slotStream_succ]
  by_cases hiR : i < FlockDeploy.repRounds true spec st
  · rw [ite_eq_left hiR] at hr
    rw [ite_eq_left hiR]
    exact recordDigest_retained h512 hd hJ (Verity.Sampled.repStream_ne_session T 0) hy0 hrds0 hr hbytes
  · rw [ite_eq_right hiR] at hr
    rw [ite_eq_right hiR]
    exact recordDigest_retained h512 hd hJ (Verity.Sampled.repStream_ne_session T 1) hy1 hrds1 hr hbytes

/-! ## The play's log has a release on each round before a session's last -/

/-- **The play's log has a release on each round before a session's last**: on a play the form accepts whose log the
check read is the device's (`Good`), for each round `j` of template `t`'s session before its last, the locator `L` puts
a release of the play's log on that round. -/
def LogOpened (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (L : CoinMap.Locate) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ t j, t < A.h.ms.length → j + 1 < SpotFlock.k₀ A.h t →
    ∃ q rl rest, o.log = q ++ rl :: rest ∧
      CoinMap.locAt L q rl.frame rl.answers = some (SpotFlock.off A.h t + j)

/-- **Each session's record gives coins of each of its rounds before the last**: on a play the form accepts whose log
the check read is the device's, each round `j` of template `t`'s session before its last is the link's or a rep's
(`j ≤ 2 * repsOf (k₀ t)`), and the `t`-th of the sessions' records the check read (`sessionRecords`) gives coins of
the stream and round `slotStream` gives it. -/
def RoundsRecorded (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ t j, t < A.h.ms.length → j + 1 < SpotFlock.k₀ A.h t →
    j ≤ 2 * repsOf (SpotFlock.k₀ A.h t) ∧
      ∃ recs rc cs, sessionRecords C o = some recs ∧ recs[t]? = some rc ∧
        Verity.Sampled.recordRound rc (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
          (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2 = some cs

/-- **`LogOpened`, from the records' rounds** (`RoundsRecorded`), at a locator that agrees with the session-round
locator at the sessions' rounds: the log the check read has a release on the round (`MembersHeld.opened`), and the
play's log is that log after releases of no coin tree, the same frames one by one (`InteriorLogged`), so its like in
the play's log is on the same round (`locOf_logged`). -/
theorem logOpened_of {A : Live₂} {C : CaseOf} {Good : Exchange → Prop} {L : CoinMap.Locate} {tab : ℕ → String}
    (hL : LocatesSessions A tab (CoinMap.locAt L)) (hsz : HeldSize) (hI : InteriorLogged C Good)
    (hG : RoundsRecorded A C Good tab) : LogOpened A C Good L := by
  intro o hacc hgood t j ht hj
  obtain ⟨hjR, recs, rc, cs, hrecs, hrc, hcs⟩ := hG o hacc hgood t j ht hj
  obtain ⟨l, hl⟩ := interiorReads C hsz o hacc
  obtain ⟨label, after, ξ, lines, trees, records, hTr, hRd, hMR, hS⟩ :=
    membersAccepted (membersNamed C hsz) o hacc l hl
  obtain rfl : records.toList.map (·.2.2.1) = recs := Option.some.inj (hS.symm.trans hrecs)
  obtain ⟨x, hx, rfl⟩ : ∃ x : String × Array (String × Option ℕ) × Json × String,
      records[t]? = some x ∧ x.2.2.1 = rc := by
    rw [List.getElem?_map, Array.getElem?_toList] at hrc
    cases hy : records[t]? with
    | none => rw [hy] at hrc; simp at hrc
    | some y => rw [hy] at hrc; exact ⟨y, rfl, Option.some.inj hrc⟩
  obtain ⟨tree, H⟩ := membersHeld_of hTr hRd hMR
  obtain ⟨pre, r, post, rfl, hloc⟩ := H.opened hTr hRd hMR (off := SpotFlock.off A.h) (K := SpotFlock.k₀ A.h)
    (M := A.h.ms.length) (tab := tab) hx ht (show j < SpotFlock.k₀ A.h t by omega) hjR hcs
  obtain ⟨q, l', hlog, hq, hF⟩ := hI o hacc hgood _ hl
  obtain ⟨a', y, rest', rfl, ha, hy, -⟩ := forall₂_split_left (frames_of_read hF)
  refine ⟨q ++ a', y, rest', hlog.trans (List.append_assoc q a' (y :: rest')).symm, ?_⟩
  refine (hL (q ++ a') y.frame y.answers _ (groupOf_off A t j)).2 ?_
  rw [← locOf_logged A tab hq ha hy]
  exact hloc

/-- **`LogOpened` at the run's locator** (`runLocate`, which agrees with the session-round locator at the sessions'
rounds: `locates_run`). -/
theorem logOpened_run (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) (Ls : CoinMap.Locate) (tab : ℕ → String)
    (hsz : HeldSize) (hI : InteriorLogged C Good) (hG : RoundsRecorded A C Good tab) :
    LogOpened A C Good (runLocate A Ls tab) :=
  logOpened_of (locates_run A Ls tab) hsz hI hG

/-! ## The sessions' records and the messages the session reading reads -/

/-- **The check read a session record for each template**: on a play the form accepts whose log the check read is the
device's, the sessions' records the check read have a `t`-th for each template `t`. -/
def RecordsCover (A : Live₂) (C : CaseOf) (Good : Exchange → Prop) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ t, t < A.h.ms.length →
    ∃ recs rc, sessionRecords C o = some recs ∧ recs[t]? = some rc

/-- **The messages the session reading reads of template `t`'s session are those of the `t`-th record the check
read**: on a play the form accepts whose log the check read is the device's, the `t`-th of the sessions' records the
check read is the parse of a record text the masked verdict accepts, with proof files, at the session a statement
sets up at one table, `tab t`, under SHA-512 digests; the session has `2 + 2 · repRounds` rounds; and the messages `S`
reads of template `t`'s session are that record's (`SessionMsgs.sessionMsgs`). -/
def MsgsOfRecords (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Exchange → Prop) (tab : ℕ → String) :
    Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ recs rc t, t < A.h.ms.length → sessionRecords C o = some recs →
    recs[t]? = some rc →
      ∃ (x : Msg) (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (st : Flock.Stmt) (record : String)
        (proofs : Array ByteArray) (ms : List Msg),
        FlockDeploy.setupOf true x = some (s, spec, #[st]) ∧
        Flock.Zk.verify spec (FlockDeploy.tabsOf #[st]) record proofs = .ok () ∧
        spec.sha512Digests = true ∧ spec.tables = #[tab t] ∧
        SpotFlock.k₀ A.h t = 2 + 2 * FlockDeploy.repRounds true spec st ∧
        Flock.JsonParse.parse record = .ok rc ∧
        SessionMsgs.sessionMsgs spec x record proofs = some ms ∧ ∀ j, S.msg o t j = ms[j]?

/-- **The digest a session's record gives a rep's round is the SHA-512 of the message read there, less its tag**: on
a play the form accepts whose log the check read is the device's, where `S` reads a message of tag `0` at round `j` of
template `t`'s session, after the link's round and before the last, the digest the `t`-th of the sessions' records
gives the stream and round `slotStream` gives `j` is the SHA-512 of the rest of that message. -/
def MidDigests (A : Live₂) (C : CaseOf) (S : SessionReading) (Good : Exchange → Prop) (tab : ℕ → String) : Prop :=
  ∀ o, FormAccepts C o → Good o → ∀ recs rc t, t < A.h.ms.length → sessionRecords C o = some recs →
    recs[t]? = some rc → ∀ j u, 0 < j → j + 1 < SpotFlock.k₀ A.h t → S.msg o t j = some (0 :: u) →
      (recordDigest rc (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).1
        (slotStream (tab t) (repsOf (SpotFlock.k₀ A.h t)) j).2).map (·.data.toList) = some (Verity.sha512Msg u)

/-- **`MidDigests`, from `MsgsOfRecords`** (`mid_digest`). -/
theorem midDigests_of {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Exchange → Prop} {tab : ℕ → String}
    (h : MsgsOfRecords A C S Good tab) : MidDigests A C S Good tab := by
  intro o hacc hgood recs rc t ht hrecs hrc j u h0 hj hmsg
  obtain ⟨x, s, spec, st, record, proofs, ms, hs, hv, h512, hT, hk, hJ, hm, hS⟩ :=
    h o hacc hgood recs rc t ht hrecs hrc
  have hR : repsOf (SpotFlock.k₀ A.h t) = FlockDeploy.repRounds true spec st := by
    unfold repsOf
    omega
  rw [hR]
  exact mid_digest hs hv h512 hT hm hJ h0 (by omega) ((hS j).symm.trans hmsg)

/-- **`RoundsRecorded`, from `RecordsCover` and `MsgsOfRecords`**: a session of `2 + 2 · repRounds` rounds has the
link's round and the two reps' before its last, each is a round the record gives (`slot_lt_rounds`), and a record the
members' check accepts gives coins of each round it gives (`Verity.Sampled.recordRound_isSome`). -/
theorem roundsRecorded_of {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Exchange → Prop}
    {tab : ℕ → String} (hsz : HeldSize) (hcov : RecordsCover A C Good) (hmsgs : MsgsOfRecords A C S Good tab) :
    RoundsRecorded A C Good tab := by
  intro o hacc hgood t j ht hj
  obtain ⟨recs, rc, hrecs, hrc⟩ := hcov o hacc hgood t ht
  obtain ⟨x, s, spec, st, record, proofs, ms, hs, hv, -, hT, hk, hJ, hm, -⟩ :=
    hmsgs o hacc hgood recs rc t ht hrecs hrc
  have hR : repsOf (SpotFlock.k₀ A.h t) = FlockDeploy.repRounds true spec st := by
    unfold repsOf
    omega
  have hlt := slot_lt_rounds hs hv hT hm hJ (j := j) (by omega)
  rw [← hR] at hlt
  refine ⟨by omega, ?_⟩
  obtain ⟨l, hl⟩ := interiorReads C hsz o hacc
  obtain ⟨label, after, ξ, lines, trees, records, hTr, hRd, hMR, hS⟩ :=
    membersAccepted (membersNamed C hsz) o hacc l hl
  obtain rfl : records.toList.map (·.2.2.1) = recs := Option.some.inj (hS.symm.trans hrecs)
  obtain ⟨y, hy, rfl⟩ : ∃ y : String × Array (String × Option ℕ) × Json × String,
      records[t]? = some y ∧ y.2.2.1 = rc := by
    rw [List.getElem?_map, Array.getElem?_toList] at hrc
    cases hy : records[t]? with
    | none => rw [hy] at hrc; simp at hrc
    | some y => rw [hy] at hrc; exact ⟨y, rfl, Option.some.inj hrc⟩
  obtain ⟨u', -, -, _, -, -, hSess⟩ := Verity.Sampled.MemberForm.memberTreeReason_none
    ((Verity.Sampled.MemberForm.membersTreeReason_none hMR).1 y (Array.mem_iff_getElem?.2 ⟨t, hy⟩))
  obtain ⟨cs, hcs⟩ := Option.isSome_iff_exists.1 (Verity.Sampled.recordRound_isSome hSess _ _ hlt)
  exact ⟨_, _, cs, hS, hrc, hcs⟩

end Verity.SampledTwo.SessionCoins

namespace Verity.SampledTwo

open Claims

/-! ## `ShowsAt` and `Explains` at a rep's round -/

/-- **A round whose message has the tag `tag` on every accepted play is explained by the message less its tag**, at
any context: two accepted plays that show the same string there read the same message. -/
theorem Messages.explains_tail {κ : Type} {A : Live₂} {Rd : Reading A} {Acc Good : Verity.Exchange → Prop}
    {raw : Verity.Exchange → ℕ → Option (List Msg)} {ctx : Verity.Exchange → ℕ → κ} {i : ℕ} (tag : UInt8)
    (hm : ∀ o, Acc o → Good o → ∃ u, Rd.msg o i = some (tag :: u))
    (hr : ∀ o, Acc o → Good o → raw o i = (Rd.msg o i).map fun m => [m.tail]) :
    Messages.Explains Rd Acc Good raw ctx i := by
  intro o o' hA hG hA' hG' _ hraw
  obtain ⟨u, hu⟩ := hm o hA hG
  obtain ⟨u', hu'⟩ := hm o' hA' hG'
  rw [hr o hA hG, hr o' hA' hG', hu, hu'] at hraw
  have e : u = u' := (List.cons.inj (Option.some.inj hraw)).1
  rw [hu, hu', e]

/-- **A session's message before its last has the tag `0`**, at the session reading and at the form's: the play's log
has a release on the round (`SessionCoins.LogOpened`), and the sessions' messages up to a round a release is put on
are of tag `0` (`Messages.SessionMsgsAsk`). -/
theorem session_msg_tag (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hOpen : SessionCoins.LogOpened A C Good L) (hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) Good L)
    {t j : ℕ} (ht : t < A.h.ms.length) (hj : j + 1 < SpotFlock.k₀ A.h t) (o : Verity.Exchange)
    (hA : FormAccepts C o) (hG : Good o) :
    ∃ u, S.msg o t j = some (0 :: u) ∧
      (formReading A C S dev).msg o (SpotFlock.off A.h t + j) = some (0 :: u) := by
  obtain ⟨q, rl, rest, hlog, hloc⟩ := hOpen o hA hG t j ht hj
  have hjk : j < SpotFlock.k₀ A.h t := by omega
  obtain ⟨-, htag⟩ := hAsk o hA hG q rl rest hlog t j ht hjk hloc
  obtain ⟨u, hu⟩ := htag j (Nat.le_refl j)
  have hmsg : S.msg o t j = some (0 :: u) := SessionCoins.some_of_getD_cons hu
  refine ⟨u, hmsg, ?_⟩
  rw [Messages.formMsg_session A C S dev o ht (Nat.le_add_right _ j) (Nat.add_lt_add_left hjk _),
    Nat.add_sub_cancel_left]
  exact hmsg

/-- **A rep's round of the form's reading shows its message less the tag**: at round `j` of template `t`'s session,
after the link's round and before the last, where the digest a play fixes for the round is the bytes the release that
ends the log before the round answers (`Messages.endAnswers`) and the string it shows is the round's message less its
tag, an accepted play whose log the check read is the device's shows a string that has its digest. The log before the
round ends at the release on the round (`SessionCoins.before_session`), which answers the digest the session's record
gives the round (`SessionCoins.LogAnswers`), the SHA-512 of the message less its tag (`SessionCoins.MidDigests`). -/
theorem showsAt_session_mid (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (tab : ℕ → String) (hL : SessionCoins.LocatesSessions A tab (CoinMap.locAt L))
    (hRise : SessionCoins.LogRising A C Good tab) (hOpen : SessionCoins.LogOpened A C Good L)
    (hAns : SessionCoins.LogAnswers A C Good L tab) (hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) Good L)
    (hMid : SessionCoins.MidDigests A C S Good tab) (Dg raw : Verity.Exchange → ℕ → Option (List Msg)) {t j : ℕ}
    (ht : t < A.h.ms.length) (h0 : 0 < j) (hj : j + 1 < SpotFlock.k₀ A.h t)
    (hd : ∀ o, FormAccepts C o → Good o → Dg o (SpotFlock.off A.h t + j) =
      Messages.endAnswers (Messages.logBefore L o (SpotFlock.off A.h t + j)))
    (hr : ∀ o, FormAccepts C o → Good o → raw o (SpotFlock.off A.h t + j) =
      ((formReading A C S dev).msg o (SpotFlock.off A.h t + j)).map fun m => [m.tail]) :
    Messages.ShowsAt (formReading A C S dev) (FormAccepts C) Good Dg raw (SpotFlock.off A.h t + j) := by
  intro o hA hG
  obtain ⟨q, rl, rest, hlog, hloc⟩ := hOpen o hA hG t j ht hj
  have hjk : j < SpotFlock.k₀ A.h t := by omega
  have hb := SessionCoins.before_session hL hRise hA hG hlog hloc (SessionCoins.groupOf_off A t j)
  obtain ⟨recs, rc, hrecs, hrc, hdig⟩ := hAns o hA hG q rl rest hlog t j hjk hloc
  obtain ⟨u, hmsg, hm⟩ := session_msg_tag A C S dev Good L hOpen hAsk ht hj o hA hG
  have hsha := hMid o hA hG recs rc t ht hrecs hrc j u h0 hj hmsg
  rw [hdig] at hsha
  have e : rl.answers.data.toList = Verity.sha512Msg u := Option.some.inj hsha
  refine ⟨⟨_, hm⟩, [rl.answers.data.toList], [u], ?_, ?_, ?_⟩
  · exact (hd o hA hG).trans (congrArg (Option.map fun fb : Verity.Frame × ByteArray => [fb.2.data.toList])
      (congrArg Prod.snd hb))
  · exact (hr o hA hG).trans (congrArg (Option.map fun m : Msg => [m.tail]) hm)
  · exact congrArg (fun m : Msg => [m]) e.symm

/-- **A session's round before its last, of the form's reading, is explained by its message less the tag**, at any
context: the message there has the tag `0` on every accepted play (`session_msg_tag`). -/
theorem explains_session_mid {κ : Type} (A : Live₂) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (Good : Verity.Exchange → Prop) (L : CoinMap.Locate)
    (hOpen : SessionCoins.LogOpened A C Good L) (hAsk : Messages.SessionMsgsAsk A S (FormAccepts C) Good L)
    (raw : Verity.Exchange → ℕ → Option (List Msg)) (ctx : Verity.Exchange → ℕ → κ) {t j : ℕ}
    (ht : t < A.h.ms.length) (hj : j + 1 < SpotFlock.k₀ A.h t)
    (hr : ∀ o, FormAccepts C o → Good o → raw o (SpotFlock.off A.h t + j) =
      ((formReading A C S dev).msg o (SpotFlock.off A.h t + j)).map fun m => [m.tail]) :
    Messages.Explains (formReading A C S dev) (FormAccepts C) Good raw ctx (SpotFlock.off A.h t + j) :=
  Messages.explains_tail 0 (fun o hA hG =>
    (session_msg_tag A C S dev Good L hOpen hAsk ht hj o hA hG).imp fun _ h => h.2) hr

end Verity.SampledTwo
