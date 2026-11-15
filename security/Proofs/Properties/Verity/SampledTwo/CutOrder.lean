module

public import Proofs.Properties.Verity.SampledTwo.CutRead
public import Proofs.Properties.Challenger.Window
public import Proofs.Properties.Challenger.CanonLine
import all Init.Data.String.Defs
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# The cut where the device's ledger orders the two sessions

`CutRead.cutBound_read` takes `PrintsTwo`: on two windows, the play's log is the first session's releases and then
the interior's, each printed by the runs the device's ledger opens on its window. That speaks of the device and of the
prover's request together: a device that opens each window as the prover's request names it prints the interior's
releases on the first window where the prover swaps the two, and `PrintsTwo` fails at a faithful history.

This module takes the order from the ledger the check read, in three steps.

* **What the check shows** (`OrderShown`): at an accepted play on two windows, the ledger the second stage's coin
  evidence carries holds an open of the first window for a session `a`, an `end` of `a`, an open of the second window
  for a session `b` and a `reserve` of `b`, the end's `seq` below the reserve's. It is a hypothesis of `order_binds`
  and `cutBound_order`, and `CutAccept.orderShown` derives it from the form's acceptance.
* **Each of those lines is the device's** (`line_device`, `entry_device`): a line of a ledger `drawCoins` accepted is
  signed under the key (`Window.verifyLedger_sig`), so the device's own ledger holds a line with the same text apart
  from its signature, or the finder outputs a break from that session's record. `Window.run_opens` is this for the
  one open a sample reads. Two lines of one text apart from their signatures have one kind, one `seq`, one window root
  and, an open, an end or a reserve, one session (`same_fields`; `entry_seq` reads an entry's `seq` and session off its
  JSON).
* **The device's fact** (`PrintsInOrder`): where the device's own ledger opens the first window for `a` and the second
  for `b` and ends `a` at a `seq` below `b`'s reserve's, the play's log is the releases of the runs on the first
  window and then of those on the second. `Properties.Challenger.History` holds a device's runs and its ledger lines
  with no order between a release and a line, so no check gives this: it says the device releases a session's coins
  after it signs the session's reserve and before it signs its end. It names no window the prover chose: under the
  swap its condition fails, and it holds. No theorem of this module proves it of a device, and it is no named
  assumption: `cutBound_order` takes it as a hypothesis on the histories it is given.

`printsTwo_of_order` puts the three together at one play: `PrintsTwo`, or a break in the interior's record.
`cutBound_order` is the cut at `ReadIsDevice` with `PrintsInOrder` in `PrintsTwo`'s place: the same bound, finder and
forger, the interior's record being one the finder reads already.

As in `Cut`, at a history holding one play's signatures (`SignsPlay`) a genuine signature of an earlier period that the
evidence carries is none the history signed, so it is the forger's, and the forger's term bounds nothing there.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

set_option mvcgen.warning false

open private strOf from Spot.ChallengerSession

namespace Verity.SampledTwo

open Lean (Json)
open Std.Do
open Properties.Challenger Spot.Challenger
open FlockDeploy (JsonEquiv)
open Properties.Pouw.LiveSalt.Coin (taggedBytes)
open Properties.Pouw.LiveSalt.Ledger (of_triple of_triple_opt throw_spec none_spec failure_spec toOption_spec
  ledgerTag_spec nat_spec some_pure_spec)
open Properties.Challenger.ChallengePhase (ledgerBytes recordBytes)
open Properties.Challenger.CanonRead (getD_equiv sorted_getD equiv_str)
open scoped ENNReal

/-! ## A ledger entry's `seq`, session and window root -/

/-- **The window root an open names**, as `verifyLedger` reads it (`LiveSalt.openRoot`'s string). -/
def openedRoot (e : LedgerEntry) : String := strOf (Properties.Pouw.LiveSalt.openRoot e)

/-- Equivalent values are one string, or neither is a string. -/
theorem strOf_equiv {x y : Json} (h : JsonEquiv x y) : strOf x = strOf y := by
  by_cases hx : ∃ s, x = .str s
  · obtain ⟨s, rfl⟩ := hx
    rw [equiv_str h.symm]
  · by_cases hy : ∃ s, y = .str s
    · obtain ⟨s, rfl⟩ := hy
      exact absurd ⟨s, equiv_str h⟩ hx
    · have e1 : strOf x = "" := by
        cases x with
        | str s => exact absurd ⟨s, rfl⟩ hx
        | _ => rfl
      have e2 : strOf y = "" := by
        cases y with
        | str s => exact absurd ⟨s, rfl⟩ hy
        | _ => rfl
      rw [e1, e2]

-- `mvcgen` splits `Core.Record.ofJson` into one goal per field it parses
set_option maxHeartbeats 4000000 in
/-- A record entry's `seq` is its JSON's number. -/
theorem recordOfJson_seq {j : Json} {r : _root_.Core.Record} (h : _root_.Core.Record.ofJson j = .ok r) :
    ∃ n : Lean.JsonNumber, Spot.getD j "seq" = .num n ∧ r.seq = n.mantissa.toNat := by
  refine of_triple (Q := fun r => ∃ n : Lean.JsonNumber, Spot.getD j "seq" = .num n ∧ r.seq = n.mantissa.toNat) ?_ h
  clear h
  mvcgen [_root_.Core.Record.ofJson, -Spec.throw_Except, throw_spec]
  all_goals first
    | exact ⟨_, by assumption, rfl⟩
    | (refine ⟨_, ?_, rfl⟩; assumption)

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- A ledger entry's `seq` is its record's, and an open's, an end's or a reserve's session is its record's. -/
theorem ledgerEntry_more {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    ∃ r, (_root_.Core.Record.ofJson e.entry).toOption = some r ∧ r.seq = e.seq ∧
      (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → r.session = some e.session) := by
  refine of_triple_opt (Q := fun e => ∃ r, (_root_.Core.Record.ofJson e.entry).toOption = some r ∧
      r.seq = e.seq ∧ (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → r.session = some e.session)) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]
  all_goals refine ⟨_, by assumption, rfl, fun ho => ?_⟩
  all_goals first | assumption | (exfalso; simp_all)

/-- **A ledger entry's `seq` and session are its JSON's**: its `seq` the number under `seq`, and an open's, an end's
or a reserve's session the string under `session`. -/
theorem entry_seq {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    (∃ n : Lean.JsonNumber, Spot.getD e.entry "seq" = .num n ∧ e.seq = n.mantissa.toNat) ∧
      (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → Spot.getD e.entry "session" = .str e.session) := by
  obtain ⟨r, hr, hq, hses⟩ := ledgerEntry_more h
  have hr' := Properties.Challenger.ChallengePhase.toOption_some hr
  obtain ⟨n, hn, hrn⟩ := recordOfJson_seq hr'
  obtain ⟨-, -, h3⟩ := Properties.Challenger.Window.recordOfJson_fields hr'
  exact ⟨⟨n, hn, hq.symm.trans hrn⟩, fun hk => h3 _ (hses hk)⟩

/-- **Two ledger entries of one text apart from their signatures have one kind, one `seq` and one window root, and
an open's, an end's or a reserve's session is the other's.** -/
theorem same_fields {l l' : Json} {e e' : LedgerEntry} (he : ledgerEntry? l = some e)
    (he' : ledgerEntry? l' = some e') {s : String}
    (hs : Flock.canonUtf8 (_root_.Core.Record.unsigned e.entry) = .ok s)
    (hs' : Flock.canonUtf8 (_root_.Core.Record.unsigned e'.entry) = .ok s) :
    e'.kind = e.kind ∧ e'.seq = e.seq ∧ openedRoot e' = openedRoot e ∧
      (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → e'.session = e.session) := by
  obtain ⟨sO, objO, kindO, schemaO, bodyO, -, -⟩ := Properties.Challenger.Window.ledgerEntry_fields he
  obtain ⟨sE, objE, kindE, schemaE, bodyE, -, -⟩ := Properties.Challenger.Window.ledgerEntry_fields he'
  obtain ⟨⟨n, hn, hq⟩, hses⟩ := entry_seq he
  obtain ⟨⟨n', hn', hq'⟩, hses'⟩ := entry_seq he'
  have F : ∀ k, k ≠ "sig" → JsonEquiv (Spot.getD e.entry k) (Spot.getD e'.entry k) := fun k hk =>
    Properties.Challenger.Window.entry_fields_equiv sO sE objO objE hs hs' hk
  have hK := F "kind" (by decide)
  rw [kindO, kindE] at hK
  have hk : e'.kind = e.kind := Json.str.inj (equiv_str hK.symm)
  have hQ := F "seq" (by decide)
  rw [hn, hn'] at hQ
  have hnn : n' = n := Json.num.inj (Properties.Challenger.CanonLine.equiv_num hQ.symm)
  have hseq : e'.seq = e.seq := by rw [hq, hq', hnn]
  have hS := F "schema" (by decide)
  rw [schemaO, schemaE] at hS
  have hsch : e'.schema = e.schema := Json.str.inj (equiv_str hS.symm)
  have hB := F "body" (by decide)
  have sBo := sorted_getD sO "body"
  have sBe := sorted_getD sE "body"
  rw [bodyO, bodyE] at hB
  rw [bodyO] at sBo
  rw [bodyE] at sBe
  have hW := getD_equiv sBo sBe hB "window"
  have hR : JsonEquiv (Properties.Pouw.LiveSalt.openRoot e) (Properties.Pouw.LiveSalt.openRoot e') := by
    unfold Properties.Pouw.LiveSalt.openRoot
    rw [hsch]
    exact getD_equiv (sorted_getD sBo "window") (sorted_getD sBe "window") hW _
  have hroot : openedRoot e' = openedRoot e := (strOf_equiv hR).symm
  refine ⟨hk, hseq, hroot, fun hkind => ?_⟩
  have hkind' : e'.kind = "open" ∨ e'.kind = "end" ∨ e'.kind = "reserve" := by
    rw [hk]
    exact hkind
  have hI := F "session" (by decide)
  rw [hses hkind, hses' hkind'] at hI
  exact Json.str.inj (equiv_str hI.symm)

/-! ## A line of the ledger the check read is the device's -/

section Lines

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-- **A signed line of the ledger a sample's coin evidence carries is a line of the device's own ledger, apart from its
signature**, or the finder outputs a break: the device signed its message, or the signature is a forgery; what the
device signs that hashes a ledger role is one of its own ledger lines' messages; and equal messages are equal texts
without their signatures, or a collision of two strings the finder hashes. `Window.run_opens`' argument, for any
line. -/
theorem line_device {ls : Array Json} {l : Json} {e : LedgerEntry} {m : ByteArray}
    (hc : Spot.getD coins "ledger" = .arr ls) (hl : l ∈ ls.toList) (hle : ledgerEntry? l = some e)
    (hm : ledgerMessage (ident key.pub) e.entry = .ok m) (hv : LmsVerifies key m e.sig) :
    (∃ line ∈ dev.ledger key.pub, ∃ e' s, ledgerEntry? line = some e' ∧
        Flock.canonUtf8 (_root_.Core.Record.unsigned e.entry) = .ok s ∧
        Flock.canonUtf8 (_root_.Core.Record.unsigned e'.entry) = .ok s) ∨ Break dev key label coins := by
  rcases signed_or_break (dev := dev) (label := label)
      (Properties.Challenger.Window.ledger_mem_signatures hc hl hle hm) hv with hsigned | b
  · obtain ⟨tag, s1, ht1, hc1, hm1, hU1, hT1⟩ := Properties.Challenger.Window.ledger_hashed hle hm
    have hU : ledgerBytes (ident key.pub) (Flock.Sha512.hash (taggedBytes tag s1)) ∈
        readHashed key label coins ++ deviceHashed dev key.pub :=
      List.mem_append_left _ (Properties.Challenger.Window.mem_readHashed_ledger hc hl hU1)
    unfold Signed signedBy at hsigned
    rcases List.mem_append.1 hsigned with hrun | hled
    · obtain ⟨r, hr, hms⟩ := List.mem_flatMap.1 hrun
      cases hcs : r.commitment with
      | none => rw [hcs] at hms; cases hms
      | some cs =>
        rw [hcs] at hms
        rcases List.mem_append.1 hms with hrec | hcl
        · cases hrm : recordMessage (ident key.pub) r.host cs with
          | error _ => rw [hrm] at hrec; cases hrec
          | ok m' =>
            rw [hrm] at hrec
            have hmm : m = m' := List.mem_singleton.1 hrec
            obtain ⟨d', hd', hm''⟩ := Properties.Challenger.ChallengePhase.recordMessage_ok hrm
            have hV : recordBytes (ident key.pub) r.host d' ∈
                readHashed key label coins ++ deviceHashed dev key.pub :=
              List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_run hr hcs
                (.inl (Properties.Challenger.ChallengePhase.record_mem_closeHashed
                  (Properties.Challenger.ChallengePhase.record_mem_recordHashed hd'))))
            rcases eq_or_break hU hV (hm1.symm.trans (hmm.trans hm'')) with hEq | b
            · exact absurd hEq.symm (Properties.Challenger.Window.recordBytes_ne_ledger _ _ _ _ _)
            · exact .inr b
        · cases hcm : closeMessage (ident key.pub) r.host cs r.id r.log.length (logChain r.log) r.ended with
          | error _ => rw [hcm] at hcl; cases hcl
          | ok m' =>
            rw [hcm] at hcl
            have hmm : m = m' := List.mem_singleton.1 hcl
            obtain ⟨rs, hrs, hm''⟩ := Properties.Challenger.ChallengePhase.closeMessage_ok hcm
            have hV : Properties.Challenger.ChallengePhase.closeBytes (ident key.pub) rs r.id r.log.length
                (logChain r.log) r.ended ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
              List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_run hr hcs
                (.inl (Properties.Challenger.ChallengePhase.close_mem_closeHashed hrs)))
            rcases eq_or_break hU hV (hm1.symm.trans (hmm.trans hm'')) with hEq | b
            · exact absurd hEq.symm (Properties.Challenger.ChallengePhase.closeBytes_ne_ledger _ _ _ _ _ _ _ _)
            · exact .inr b
    · obtain ⟨line, hline, hlm⟩ := List.mem_filterMap.1 hled
      obtain ⟨e', he', hem⟩ := Option.bind_eq_some_iff.1 hlm
      obtain ⟨tag', s2, ht2, hc2, hm2, hU2, hT2⟩ :=
        Properties.Challenger.Window.ledger_hashed he' (Properties.Challenger.ChallengePhase.toOption_some hem)
      have hV : ledgerBytes (ident key.pub) (Flock.Sha512.hash (taggedBytes tag' s2)) ∈
          readHashed key label coins ++ deviceHashed dev key.pub :=
        List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_ledger hline hU2)
      rcases eq_or_break hU hV (hm1.symm.trans hm2) with hEq | b
      · have hTr : taggedBytes tag s1 ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_left _ (Properties.Challenger.Window.mem_readHashed_ledger hc hl hT1)
        have hTd : taggedBytes tag' s2 ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_ledger hline hT2)
        rcases eq_or_break hTr hTd (Properties.Challenger.Window.ledgerBytes_inj hEq) with hEq2 | b
        · obtain ⟨rfl, rfl⟩ := Properties.Challenger.Window.taggedBytes_inj2
            (Properties.Challenger.Window.ledgerTag_size ht1 ht2) hEq2
          exact .inl ⟨line, hline, e', s1, he', hc1, hc2⟩
        · exact .inr b
      · exact .inr b
  · exact .inr b

/-- **Every entry of a ledger `drawCoins` accepted is a line of it, signed under the key**: `drawCoins` reads its
session through `checked`, and `verifyLedger` checked each entry's signature over its message
(`Window.verifyLedger_sig`). -/
theorem lines_signed {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ)
    {e : LedgerEntry} (he : e ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger")) :
    ∃ ls l m, Spot.getD coins "ledger" = .arr ls ∧ l ∈ ls.toList ∧ ledgerEntry? l = some e ∧
      ledgerMessage (ident key.pub) e.entry = .ok m ∧ LmsVerifies key m e.sig := by
  obtain ⟨c, roots, ht, -⟩ := Properties.Challenger.ChallengePhase.drawCoins_ok h
  obtain ⟨a, hp, -⟩ := Properties.Challenger.ChallengePhase.trustedChallengePhaseOf_ok ht
  obtain ⟨ob, hao⟩ := Properties.Pouw.LiveSalt.Ledger.pinnedChallengePhase_ok hp
  obtain ⟨wp, win, en, hch, -⟩ := Properties.Pouw.LiveSalt.Ledger.auditedOpen_ok hao
  obtain ⟨es, hes, hvl⟩ := Properties.Pouw.LiveSalt.Ledger.checked_ok hch
  have hmem : e ∈ es.toList := by
    rw [← hes]
    exact he
  obtain ⟨m, hm, hv⟩ := Properties.Challenger.Window.verifyLedger_sig hvl e hmem
  obtain ⟨ls, hc, l, hl, hle⟩ : ∃ ls : Array Json, Spot.getD coins "ledger" = .arr ls ∧
      ∃ l ∈ ls.toList, ledgerEntry? l = some e := by
    unfold Properties.Pouw.LiveSalt.entries at he
    split at he
    · rename_i ls hc
      obtain ⟨l, hl, he⟩ := List.mem_filterMap.1 he
      exact ⟨ls, hc, l, hl, he⟩
    · simp at he
  exact ⟨ls, l, m, hc, hl, hle, hm, Properties.Challenger.ChallengePhase.lms_of_check hv⟩

/-- **An entry of a ledger `drawCoins` accepted is one of the device's own ledger's**, with its kind, its `seq`, its
window root and, an open, an end or a reserve, its session; or the finder outputs a break. -/
theorem entry_device {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ)
    {e : LedgerEntry} (he : e ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger")) :
    (∃ line ∈ dev.ledger key.pub, ∃ e', ledgerEntry? line = some e' ∧ e'.kind = e.kind ∧ e'.seq = e.seq ∧
        openedRoot e' = openedRoot e ∧
        (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → e'.session = e.session)) ∨
      Break dev key label coins := by
  obtain ⟨ls, l, m, hc, hl, hle, hm, hv⟩ := lines_signed h he
  rcases line_device (dev := dev) (label := label) hc hl hle hm hv with ⟨line, hline, e', s, he', hs, hs'⟩ | b
  · obtain ⟨hk, hq, hr, hses⟩ := same_fields hle he' hs hs'
    exact .inl ⟨line, hline, e', he', hk, hq, hr, hses⟩
  · exact .inr b

end Lines

/-! ## The device's ledger's order -/

/-- **The ledger opens the window root `w` for the run `a`**: one of its lines is an open of `a` whose window's root
is `w`, withheld or not, under either schema. -/
def LedgerOpens (ledger : List Json) (w a : String) : Prop :=
  ∃ line ∈ ledger, ∃ e, ledgerEntry? line = some e ∧ e.kind = "open" ∧ e.session = a ∧ openedRoot e = w

/-- **The ledger ends the run `a` before it reserves the run `b`**: it holds an `end` line of `a` and a `reserve` line
of `b`, the end's `seq` below the reserve's. -/
def EndsBefore (ledger : List Json) (a b : String) : Prop :=
  ∃ x ∈ ledger, ∃ y ∈ ledger, ∃ ex ey, ledgerEntry? x = some ex ∧ ledgerEntry? y = some ey ∧
    ex.kind = "end" ∧ ex.session = a ∧ ey.kind = "reserve" ∧ ey.session = b ∧ ex.seq < ey.seq

/-- **The device printed the play's log in its ledger's order**: on one window, the runs on it print the whole log
(`PrintsOn`, as `PrintsTwo`); on two, where the device's own ledger under a key opens the first window for a run
`a` and the second for a run `b` and ends `a` before it reserves `b` (`EndsBefore`), the log is the releases the runs
on the first window print and then those the runs on the second print. A fact of the device alone: it releases a
session's coins after it signs the session's reserve and before it signs its end. -/
def PrintsInOrder (dev : History) (w₁ w₂ : String) (log : List Release) : Prop :=
  if w₁ = w₂ then PrintsOn dev w₁ log
  else ∀ pk a b, LedgerOpens (dev.ledger pk) w₁ a → LedgerOpens (dev.ledger pk) w₂ b → EndsBefore (dev.ledger pk) a b →
    ∃ log₁ log₂, log = log₁ ++ log₂ ∧ PrintsOn dev w₁ log₁ ∧ PrintsOn dev w₂ log₂

/-- **What an accepted play on two windows shows of its two sessions' order** (a hypothesis where it is used here;
`CutAccept.orderShown` derives it from the form's acceptance): the ledger the second stage's coin evidence carries
holds an open of the first window, an `end` of that open's session, an open of the second window and a `reserve` of
that open's session, the end's `seq` below the reserve's. -/
def OrderShown (θ : ByteArray) : Prop :=
  ∀ s : Verity.Sent, s.params = θ → Accepts s → ∀ p c e, caseOf s = .ok (p, c, e) →
    ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      ∃ o₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
      ∃ e₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
      ∃ rv ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
      ∃ o₂ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
        o₁.kind = "open" ∧ openedRoot o₁ = w₁ ∧ e₁.kind = "end" ∧ e₁.session = o₁.session ∧
        o₂.kind = "open" ∧ openedRoot o₂ = w₂ ∧ rv.kind = "reserve" ∧ rv.session = o₂.session ∧ e₁.seq < rv.seq

/-- **The device printed the play's log in its two sessions, or the record carries a break**, on two windows: where
`drawCoins` accepted the coin evidence `coins` under `key`, its ledger shows an open of `w₁`, an `end` of that open's
session, an open of `w₂` and a `reserve` of that open's session, the end's `seq` below the reserve's, and the device
printed the log in its ledger's order (`PrintsInOrder`). Each of the four lines is the device's own (`entry_device`),
so its ledger opens the two windows and ends the first's session before it reserves the second's. -/
theorem printsTwo_of_order {dev : History} {key : ChallengerKey} {label : String} {coins : Json} {w : String}
    {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) {w₁ w₂ : String} (hne : w₁ ≠ w₂)
    {log : List Release} (hP : PrintsInOrder dev w₁ w₂ log) {o₁ e₁ rv o₂ : LedgerEntry}
    (ho₁ : o₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (he₁ : e₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (hrv : rv ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (ho₂ : o₂ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (k₁ : o₁.kind = "open") (r₁ : openedRoot o₁ = w₁) (ke : e₁.kind = "end") (se : e₁.session = o₁.session)
    (k₂ : o₂.kind = "open") (r₂ : openedRoot o₂ = w₂) (kr : rv.kind = "reserve") (sr : rv.session = o₂.session)
    (hlt : e₁.seq < rv.seq) : PrintsTwo dev w₁ w₂ log ∨ Break dev key label coins := by
  rcases entry_device (dev := dev) h ho₁ with ⟨la, hla, oa, hoa, hka, -, hra, hsa⟩ | b
  · rcases entry_device (dev := dev) h he₁ with ⟨lx, hlx, ex, hex, hkx, hqx, -, hsx⟩ | b
    · rcases entry_device (dev := dev) h hrv with ⟨ly, hly, ey, hey, hky, hqy, -, hsy⟩ | b
      · rcases entry_device (dev := dev) h ho₂ with ⟨lb, hlb, ob, hob, hkb, -, hrb, hsb⟩ | b
        · left
          unfold PrintsInOrder at hP
          rw [if_neg hne] at hP
          unfold PrintsTwo
          rw [if_neg hne]
          refine hP key.pub o₁.session o₂.session ?_ ?_ ?_
          · exact ⟨la, hla, oa, hoa, hka.trans k₁, hsa (.inl k₁), hra.trans r₁⟩
          · exact ⟨lb, hlb, ob, hob, hkb.trans k₂, hsb (.inl k₂), hrb.trans r₂⟩
          · refine ⟨lx, hlx, ly, hly, ex, ey, hex, hey, hkx.trans ke, (hsx (.inr (.inl ke))).trans se,
              hky.trans kr, (hsy (.inr (.inr kr))).trans sr, ?_⟩
            rw [hqx, hqy]
            exact hlt
        · exact .inr b
      · exact .inr b
    · exact .inr b
  · exact .inr b

/-! ## The binding at one play -/

open Classical in
/-- The histories that are `h` at the play `o` and hold nothing at every other. -/
noncomputable def onlyAt (o : Exchange) (h : History) : Exchange → History :=
  fun o' => if o' = o then h else ⟨fun _ => [], fun _ => []⟩

theorem onlyAt_self (o : Exchange) (h : History) : onlyAt o h o = h := by
  unfold onlyAt
  exact if_pos rfl

/-- A history that holds no run prints any log on any window. -/
theorem printsOn_nil (w : String) (log : List Release) : PrintsOn ⟨fun _ => [], fun _ => []⟩ w log := by
  intro pk r hr
  first
    | cases hr
    | exact absurd hr List.not_mem_nil

theorem printsTwo_onlyAt {o : Exchange} {h : History} {θ : ByteArray}
    (hP : ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → PrintsTwo h w₁ w₂ o.log) :
    ∀ o' : Exchange, o'.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) →
      PrintsTwo (onlyAt o h o') w₁ w₂ o'.log := by
  intro o' _ w₁ w₂ hw
  by_cases ho : o' = o
  · rw [ho, onlyAt_self]
    exact hP w₁ w₂ hw
  · have hn : onlyAt o h o' = ⟨fun _ => [], fun _ => []⟩ := by
      unfold onlyAt
      exact if_neg ho
    rw [hn]
    unfold PrintsTwo
    by_cases hww : w₁ = w₂
    · rw [if_pos hww]
      exact printsOn_nil _ _
    · rw [if_neg hww]
      exact ⟨o'.log, [], (List.append_nil _).symm, printsOn_nil _ _, printsOn_nil _ _⟩

/-- **The binding at one play**: where the first stage's coins are read under the parameters' key (`FirstReads`) and
the device's history at this play printed its log in its two sessions (`PrintsTwo`, at this play alone), an accepted
play where the check read no log of both sessions, or read one that isn't the device's, carries a break in one of the
two sessions' records, or an opening collision. `CutRead.reads_or_break` and `CutHeld.binds_of_accepts`, at the
histories that are the device's at this play and hold nothing at every other (`onlyAt`). -/
theorem binds_at (dev : Exchange → History) (θ : ByteArray) (hF : FirstReads θ) {o : Exchange} (hθ : o.params = θ)
    (ha : Accepts o.sent) (hP : ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → PrintsTwo (dev o) w₁ w₂ o.log)
    (hg : ¬ ReadIsDevice o) :
    (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
      ∃ l ∈ opensReads o.sent, ∃ u v, Verity.Sampled.Lines.openCollision l o.log = some (u, v) ∧ u ≠ v ∧
        Flock.Sha512.hash u = Flock.Sha512.hash v := by
  have hP' := printsTwo_onlyAt (o := o) (h := dev o) (θ := θ) hP
  have hd := onlyAt_self o (dev o)
  rcases reads_or_break (onlyAt o (dev o)) θ hF hP' hθ ha with hr | hb
  · have hb := binds_of_accepts (onlyAt o (dev o)) θ hP' o hθ ha fun hl => hg ⟨hr, hl⟩
    rw [hd] at hb
    exact hb
  · rw [hd] at hb
    exact .inl hb

/-- **The binding where the device's ledger orders the two sessions**: where the first stage's coins are read under
the parameters' key (`FirstReads`), an accepted play on two windows shows the order in the ledger the second stage's
coin evidence carries (`OrderShown`) and the device's history at each play printed the play's log in its ledger's
order (`PrintsInOrder`), an accepted play where the check read no log of both sessions, or read one that isn't the
device's, carries a break in one of the two sessions' records, or an opening collision. -/
theorem order_binds (dev : Exchange → History) (θ : ByteArray) (hF : FirstReads θ) (hO : OrderShown θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) →
      PrintsInOrder (dev o) w₁ w₂ o.log)
    {o : Exchange} (hθ : o.params = θ) (ha : Accepts o.sent) (hg : ¬ ReadIsDevice o) :
    (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
      ∃ l ∈ opensReads o.sent, ∃ u v, Verity.Sampled.Lines.openCollision l o.log = some (u, v) ∧ u ≠ v ∧
        Flock.Sha512.hash u = Flock.Sha512.hash v := by
  have ha' := ha
  obtain ⟨p, c, e, pub, prof, h1, hacc⟩ := ha'
  obtain ⟨key, d₁, _, hK, h3, _⟩ := hF o.sent hθ ha p c e h1
  obtain ⟨-, c₂, ξ₂, r, i, _run, hin, -, -, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨⟨sj, hcanon, hcom⟩, h7, _, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have h5 : interiorOf e = .ok c₂ := by
    unfold interiorOf
    rw [hcanon]
    first
      | exact hcom
      | simpa [bind, Except.bind] using hcom
  obtain ⟨d₂, hd₂⟩ := oneStageZk_source h7
  have h6 : secondDraws p e = some d₂ := by
    unfold secondDraws
    rw [h5]
    first
      | exact hd₂
      | simpa [Except.toOption] using hd₂
  have h2 : keyOf o.sent.params = some key := by
    have h := hK
    rw [← hθ] at h
    exact h
  have hk₂ : d₂.key = key := key_second h1 h2 h5 h6
  have hm₂ : (d₂, e.interior.base.coins) ∈ records o.sent := by rw [records_eq h1 h2, h3, h6]; simp
  have hw : windowsOf θ = some (d₁.window, d₂.window) := by rw [← hθ]; exact windowsOf_eq h1 h3 h5 h6
  obtain ⟨raw, hraw, -⟩ := oneStageZk_reads p.audit c₂ d₂ hd₂ e.interior ξ₂ h7
  have hPT : PrintsTwo (dev o) d₁.window d₂.window o.log ∨
      Break (dev o) d₂.key d₂.label e.interior.base.coins := by
    have hPO := hP o hθ d₁.window d₂.window hw
    by_cases hww : d₁.window = d₂.window
    · left
      unfold PrintsInOrder at hPO
      rw [if_pos hww] at hPO
      unfold PrintsTwo
      rw [if_pos hww]
      exact hPO
    · obtain ⟨o₁, ho₁, e₁, he₁, rv, hrv, o₂, ho₂, k₁, r₁, ke, se, k₂, r₂, kr, sr, hlt⟩ :=
        hO o.sent hθ ha p c e h1 d₁.window d₂.window hw hww
      exact printsTwo_of_order hraw hww hPO ho₁ he₁ hrv ho₂ k₁ r₁ ke se k₂ r₂ kr sr hlt
  rcases hPT with hPT | b
  · refine binds_at dev θ hF hθ ha (fun w₁ w₂ hw' => ?_) hg
    rw [hw] at hw'
    have hpair := Option.some.inj hw'
    simp only [Prod.mk.injEq] at hpair
    obtain ⟨rfl, rfl⟩ := hpair
    exact hPT
  · exact .inl ⟨(d₂, e.interior.base.coins), hm₂, by rw [hK, ← hk₂], b⟩

/-! ## The cut where the device's ledger orders the two sessions -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The cut where the device's ledger orders the two sessions**: where the sampled verdict accepts only where the
form does (`Wired`), the first stage's coins are read under the parameters' key (`FirstReads`), an accepted play on two
windows shows the order in the ledger the second stage's coin evidence carries (`OrderShown`), the device's history at
each play printed the play's log in its ledger's order (`PrintsInOrder`) and signed what the play's signatures are on
(`SignsPlay`), an accepted play whose logs the check didn't read, or read other than the device's, is at most as
likely as the read finder's collision plus the forger's forgery (`terms`), over both sessions. `cutBound_read` with
`PrintsInOrder`, which names no window the prover chose, in `PrintsTwo`'s place. -/
theorem cutBound_order (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hw : Wired V k θ) (hF : FirstReads θ) (hO : OrderShown θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) →
      PrintsInOrder (dev o) w₁ w₂ o.log)
    (hs : SignsPlay V dev θ F c) : CutBound V k θ F c σ ReadIsDevice (terms V dev θ F c σ) :=
  cutBound_of V k dev θ F c σ ReadIsDevice hw (fun _ hθ ha hg => order_binds dev θ hF hO hP hθ ha hg) hs

end Verity.SampledTwo
