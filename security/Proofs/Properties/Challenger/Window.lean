module

public import Proofs.Properties.Challenger.Session
public import Proofs.Properties.Challenger.CanonRead
import all Init.Data.String.Defs
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# A sample's window is one the device's own ledger opens, or a break

`drawCoins` reads its session from the auditor's ledger: `pinnedSession` takes the v2 open whose root is the window's
(`Ledger.auditedOpen_ok`), `sessionNamed` reads the session that open names (`sessionNamed_sid`), and `withheldOn`
requires every open of the window's root to be withheld (`drawCoins_withheld`, `withheld_of`). `verifyLedger` checked
every ledger entry's LMS signature over its message under the window's key (`verifyLedger_sig`), and that pair is one the
finder checks (`ledger_mem_signatures`). So the device signed the open's message, or the finder outputs a forgery
(`signed_or_break`). What the device signs that hashes a ledger role is one of its own ledger lines' messages, since a
run's record and close begin with roles of other lengths (`recordBytes_ne_ledger`, `closeBytes_ne_ledger`). Equal
messages hash equal tagged digests, and equal digests are equal strings, or a collision of two the finder hashes
(`eq_or_break`); the two ledger tags have one length (`ledgerTag_size`), so the tagged strings name one tag and one
canonical text (`taggedBytes_inj2`). Every entry holds its keys in order (`ledgerEntry?`'s `sortedKeys` guard), so two
entries whose unsigned canonical texts agree have equivalent fields other than `sig` (`entry_fields_equiv`, by
`CanonRead.canon_inj`). The device's line is then an open of the same session, under v2, of the window's root, withheld
(`window_opens`), and `drawCoins_opens` adds it to `Session.sessionOf_bound`'s run.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

set_option mvcgen.warning false

open private verifies isBodyOf isDigest from Spot.ChallengerSession

namespace Properties.Challenger.Window

open Lean (Json)
open Std.Do
open Spot Spot.Challenger
open FlockDeploy (JsonEquiv)
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj be64_size)
open Properties.Pouw.LiveSalt.Coin (enc enc_inj taggedBytes tagged_ok)
open Properties.Pouw.LiveSalt.Ledger (of_triple of_triple_opt throw_spec ok_spec bind_ok recordOfJson_spec mapM_spec
  none_spec failure_spec toOption_spec ledgerTag_spec nat_spec some_pure_spec forIn_array_zipIdx_ok wp_ite restart
  join_spec mem_snoc)
open Properties.Challenger.Session (ledgerBytes ledgerBytes_bs recordBytes recordBytes_bs record_role_size
  ledger_role_size enc_ne)
open Properties.Challenger.CanonRead (getD_equiv sorted_getD equiv_str equiv_bool getD_unsigned sorted_unsigned)

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

/-! ## A ledger entry's fields -/

-- `mvcgen` splits `Core.Record.ofJson` into one goal per field it parses
set_option maxHeartbeats 4000000 in
/-- A record entry's kind, body and session are its JSON's. -/
theorem recordOfJson_fields {j : Json} {r : Core.Record} (h : Core.Record.ofJson j = .ok r) :
    getD j "kind" = .str r.kind ∧ getD j "body" = r.body ∧ (∀ s, r.session = some s → getD j "session" = .str s) := by
  refine of_triple (Q := fun r => getD j "kind" = .str r.kind ∧ getD j "body" = r.body ∧
    (∀ s, r.session = some s → getD j "session" = .str s)) ?_ h
  clear h
  mvcgen [Core.Record.ofJson, -Spec.throw_Except, throw_spec]
  all_goals exact ⟨by assumption, rfl, fun s hs => by cases hs <;> assumption⟩

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- A ledger entry holds its keys in order, its body is of its kind's form, and its record names its kind, its body
and, for an open, its session. -/
theorem ledgerEntry_read {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    sortedKeys e.entry = true ∧ isBodyOf e.kind (e.schema == LEDGER_SCHEMA) e.body = true ∧
      ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.kind = e.kind ∧ r.body = e.body ∧
        (e.kind = "open" → r.session = some e.session) := by
  refine of_triple_opt (Q := fun e => sortedKeys e.entry = true ∧
      isBodyOf e.kind (e.schema == LEDGER_SCHEMA) e.body = true ∧
      ∃ r, (Core.Record.ofJson e.entry).toOption = some r ∧ r.kind = e.kind ∧ r.body = e.body ∧
        (e.kind = "open" → r.session = some e.session)) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]
  all_goals refine ⟨by assumption, by assumption, _, by assumption, rfl, rfl, fun ho => ?_⟩
  all_goals first | assumption | (exfalso; simp_all)

/-- **A ledger entry's fields**: it is a sorted object whose `kind`, `schema` and `body` are the entry's, an open's
`session` its session, and its body of its kind's form. -/
theorem ledgerEntry_fields {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    sortedKeys e.entry = true ∧ (∃ kvs, e.entry = .obj kvs) ∧ getD e.entry "kind" = .str e.kind ∧
      getD e.entry "schema" = .str e.schema ∧ getD e.entry "body" = e.body ∧
      (e.kind = "open" → getD e.entry "session" = .str e.session) ∧
      isBodyOf e.kind (e.schema == LEDGER_SCHEMA) e.body = true := by
  obtain ⟨hs, hb, r, hr, hk, hbd, ho⟩ := ledgerEntry_read h
  obtain ⟨hobj, hsch⟩ := Session.ledgerEntry_obj h
  obtain ⟨h1, h2, h3⟩ := recordOfJson_fields (Session.toOption_some hr)
  exact ⟨hs, hobj, hk ▸ h1, hsch, hbd ▸ h2, fun hko => h3 _ (ho hko), hb⟩

theorem digest_str {j : Json} {n : Nat} (h : isDigest j n = true) : ∃ s, j = .str s := by
  cases j with
  | str s => exact ⟨s, rfl⟩
  | _ => simp [isDigest] at h

/-- A v2 open's window root is a string. -/
theorem open_root {body : Json} (h : isBodyOf "open" true body = true) :
    ∃ r, getD (getD body "window") "root" = .str r := by
  simp only [isBodyOf, isWindow, ite_true, Bool.and_eq_true] at h
  exact digest_str h.1.1.1.1.2

/-! ## What `verifyLedger` and `drawCoins` check -/

/-- Every entry of `pre` is signed under `key`: its message reads, and its signature verifies over it. -/
def SigInv (key : ChallengerKey) (pre : List LedgerEntry) : Prop :=
  ∀ x ∈ pre, ∃ m, ledgerMessage (key.pub.extract 0 16) x.entry = .ok m ∧ verifies key m x.sig = true

theorem sig_step {pre : List LedgerEntry} {e : LedgerEntry} {m : ByteArray}
    (hI : SigInv key pre) (hm : ledgerMessage (key.pub.extract 0 16) e.entry = .ok m)
    (hv : verifies key m e.sig = true) : SigInv key (pre ++ [e]) := by
  intro x hx
  rcases mem_snoc.1 hx with hx | rfl
  · exact hI x hx
  · exact ⟨m, hm, hv⟩

theorem ledgerMessage_eq_spec (I : ByteArray) (j : Json) :
    ⦃⌜True⌝⦄ ledgerMessage I j ⦃⇓? a => ⌜ledgerMessage I j = .ok a⌝⦄ := ok_spec _

-- `mvcgen` splits `verifyLedger`'s loop body into one goal per path through its checks
set_option maxHeartbeats 4000000 in
/-- **`verifyLedger` checked every entry's signature** over its message under the key. -/
theorem verifyLedger_sig {es : Array LedgerEntry} {r : Array Win × Array MonitoringPeriod}
    (h : verifyLedger es key = .ok r) : SigInv key es.toList := by
  unfold verifyLedger at h
  rw [bind_ok] at h
  obtain ⟨s, hs, -⟩ := h
  refine forIn_array_zipIdx_ok es _ _ (fun pre _ => SigInv key pre) (fun _ h => absurd h List.not_mem_nil) ?_ hs
  intro pre e s t hI hb
  obtain ⟨wins, leaves, records, periods, current, reserves, prev, last, v2, twoStep⟩ := s
  refine of_triple (Q := fun t => ∃ s', t = .yield s' ∧ SigInv key (pre ++ [e])) ?_ hb
  clear hb hs
  mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_eq_spec, join_spec]
  all_goals (try intros)
  all_goals (try (refine wp_ite (fun _ => restart ?_) (fun _ => restart ?_) <;>
    mvcgen [-Spec.throw_Except, throw_spec, ledgerMessage_eq_spec, join_spec]))
  all_goals refine ⟨_, rfl, sig_step hI ‹_› ‹_›⟩

-- `mvcgen` splits `sessionNamed` into one goal per check it makes of the session's files
set_option maxHeartbeats 1000000 in
/-- **The session `sessionNamed` reads is the one the audited open names.** -/
theorem sessionNamed_sid {a : Audited} {session : Json} {r : LiveCommitment × Roots}
    (h : sessionNamed a key session = .ok r) :
    ∃ r', sessionOf key (getD session "record") (getD session "log") (getD session "close") = .ok r' ∧
      r'.1 = r.1 ∧ r'.2.1 = r.2 ∧ r'.2.2 = a.session := by
  refine of_triple (Q := fun r => ∃ r',
    sessionOf key (getD session "record") (getD session "log") (getD session "close") = .ok r' ∧
      r'.1 = r.1 ∧ r'.2.1 = r.2 ∧ r'.2.2 = a.session) ?_ h
  clear h
  mvcgen [sessionNamed, -Spec.throw_Except, throw_spec, Session.sessionOf_ok_spec, recordOfJson_spec, mapM_spec]
  all_goals exact ⟨_, ‹_›, rfl, rfl, beq_iff_eq.1 ‹_›⟩

-- `mvcgen` splits `drawCoins` into one goal per check it makes of the record's form, the window and the family
set_option maxHeartbeats 1000000 in
/-- **`drawCoins` accepts only a sample whose window the ledger opens withheld.** -/
theorem drawCoins_withheld {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    withheldOn (getD coins "ledger") w = true := by
  refine of_triple (Q := fun _ => withheldOn (getD coins "ledger") w = true) ?_ h
  clear h
  mvcgen [drawCoins, -Spec.throw_Except, throw_spec, Session.trustedSessionOf_ok_spec, Session.challengerStream_ok_spec]

/-- **Every open of the window's root is withheld** in a ledger `withheldOn` accepts. -/
theorem withheld_of {ledger : Json} {w : String} (h : withheldOn ledger w = true) {ls : Array Json} {l : Json}
    {o : LedgerEntry} (hc : ledger = .arr ls) (hl : l ∈ ls.toList) (he : ledgerEntry? l = some o)
    (hk : o.kind = "open") (hr : getD (getD o.body "window") "root" = .str w) :
    getD o.body "withheld" = .bool true := by
  subst hc
  unfold withheldOn at h
  simp only [Bool.and_eq_true, List.all_eq_true] at h
  have hmem : o ∈ ls.toList.filterMap fun l => (ledgerEntry? l).filter fun e =>
      e.kind == "open" && (match getD (getD e.body "window") "root" with | .str r => r == w | _ => false) :=
    List.mem_filterMap.2 ⟨l, hl, by rw [he]; simp [Option.filter, hk, hr]⟩
  have := h.2 o hmem
  split at this
  · assumption
  · cases this

/-! ## The strings the finder hashes and the signatures it checks -/

theorem ledgerBytes_inj {I d d' : ByteArray} (h : ledgerBytes I d = ledgerBytes I d') : d = d' := by
  have e := congrArg bs h
  rw [ledgerBytes_bs, ledgerBytes_bs] at e
  exact enc_inj (List.append_cancel_left (List.append_cancel_left e))

theorem recordBytes_ne_ledger (I : ByteArray) (host : String) (d I' d' : ByteArray) :
    recordBytes I host d ≠ ledgerBytes I' d' := by
  intro h
  have hb := congrArg bs h
  rw [recordBytes_bs, ledgerBytes_bs] at hb
  exact enc_ne (by rw [record_role_size]; decide) (by rw [ledger_role_size]; decide)
    (by rw [record_role_size, ledger_role_size]; decide) hb

/-- Tagged texts under tags of one length are one text under one tag. -/
theorem taggedBytes_inj2 {tag tag' s s' : String} (hs : tag.toUTF8.size = tag'.toUTF8.size)
    (h : taggedBytes tag s = taggedBytes tag' s') : tag = tag' ∧ s = s' := by
  have e := congrArg bs h
  simp only [taggedBytes, bs_append, List.append_assoc, List.append_cancel_left_eq] at e
  rw [hs] at e
  have e2 := List.append_cancel_left e
  obtain ⟨e3, e4⟩ := List.append_inj e2 (by rw [bs_length, bs_length, hs])
  exact ⟨String.toByteArray_inj.1 (bs_inj e3),
    String.toByteArray_inj.1 (bs_inj (List.append_inj e4 (by rw [bs_length, bs_length, be64_size, be64_size])).2)⟩

theorem ledgerTag_size {s s' t t' : String} (h : ledgerTag? s = some t) (h' : ledgerTag? s' = some t') :
    t.toUTF8.size = t'.toUTF8.size := by
  unfold ledgerTag? at h h'
  split at h <;> split at h' <;> cases h <;> cases h' <;> decide

/-- **What a ledger line's message hashes**: its entry's tagged canonical text without its `sig`, under its schema's
tag, then the role, the identifier and that digest; both strings the line's `ledgerHashed`. -/
theorem ledger_hashed {I : ByteArray} {line : Json} {e : LedgerEntry} {m : ByteArray}
    (he : ledgerEntry? line = some e) (hm : ledgerMessage I e.entry = .ok m) :
    ∃ tag s, ledgerTag? e.schema = some tag ∧ Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s ∧
      m = Flock.Sha512.hash (ledgerBytes I (Flock.Sha512.hash (taggedBytes tag s))) ∧
      ledgerBytes I (Flock.Sha512.hash (taggedBytes tag s)) ∈ ledgerHashed I line ∧
      taggedBytes tag s ∈ ledgerHashed I line := by
  obtain ⟨schema, tag, d, hs, ht, hd, rfl⟩ := Session.ledgerMessage_ok hm
  obtain ⟨⟨kvs, hkvs⟩, hs'⟩ := Session.ledgerEntry_obj he
  obtain rfl : schema = e.schema := Json.str.inj (hs.symm.trans hs')
  obtain ⟨s, hcs, rfl⟩ := tagged_ok hd
  have hb : Json.mkObj ((entries e.entry).toList.filter (·.1 != "sig")) = Core.Record.unsigned e.entry := by
    rw [hkvs]
    rfl
  refine ⟨tag, s, ht, hcs, rfl, ?_, ?_⟩
  · unfold ledgerHashed
    rw [he]
    simp only [ht]
    rw [hb, hd]
    exact List.mem_append_right _ (List.mem_singleton_self _)
  · unfold ledgerHashed
    rw [he]
    simp only [ht]
    rw [hb]
    exact List.mem_append_left _ (Session.tagged_mem hcs)

theorem mem_readHashed_ledger {ls : Array Json} {l : Json} {x : ByteArray} (hc : getD coins "ledger" = .arr ls)
    (hl : l ∈ ls.toList) (hx : x ∈ ledgerHashed (ident key.pub) l) : x ∈ readHashed key label coins := by
  unfold readHashed
  simp only [hc, List.mem_append, List.mem_flatMap]
  exact .inl (.inr ⟨l, hl, hx⟩)

theorem ledger_mem_signatures {ls : Array Json} {l : Json} {e : LedgerEntry} {m : ByteArray}
    (hc : getD coins "ledger" = .arr ls) (hl : l ∈ ls.toList) (he : ledgerEntry? l = some e)
    (hm : ledgerMessage (ident key.pub) e.entry = .ok m) : (m, e.sig) ∈ signatures key coins := by
  unfold signatures
  simp only [hc]
  apply List.mem_append_right
  exact List.mem_filterMap.2 ⟨l, hl, by simp [he, hm, Except.toOption]⟩

/-- **Two entries whose unsigned canonical texts agree have equivalent fields other than `sig`.** -/
theorem entry_fields_equiv {a b : Json} (ha : sortedKeys a = true) (hb : sortedKeys b = true)
    (hao : ∃ kvs, a = .obj kvs) (hbo : ∃ kvs, b = .obj kvs) {s : String}
    (hsa : Flock.canonUtf8 (Core.Record.unsigned a) = .ok s) (hsb : Flock.canonUtf8 (Core.Record.unsigned b) = .ok s)
    {k : String} (hk : k ≠ "sig") : JsonEquiv (getD a k) (getD b k) := by
  obtain ⟨m, rfl⟩ := hao
  obtain ⟨n, rfl⟩ := hbo
  have h := CanonRead.canon_inj (sorted_unsigned ha) (sorted_unsigned hb) hsa hsb
  have := getD_equiv (sorted_unsigned ha) (sorted_unsigned hb) h k
  rwa [getD_unsigned ha hk, getD_unsigned hb hk] at this

/-! ## The window -/

/-- **The device's ledger line that signs an open's message is an open of the same session, under v2, of the same
window, withheld**, when the two lines' entries' unsigned canonical texts agree. -/
theorem opens_of_text {line : Json} {o e : LedgerEntry} {w : String} (ho : ledgerEntry? line = some o)
    {line' : Json} (he : ledgerEntry? line' = some e) {s : String}
    (hso : Flock.canonUtf8 (Core.Record.unsigned o.entry) = .ok s)
    (hse : Flock.canonUtf8 (Core.Record.unsigned e.entry) = .ok s)
    (hk : o.kind = "open") (hv2 : o.schema = LEDGER_SCHEMA) (hr : getD (getD o.body "window") "root" = .str w)
    (hw : getD o.body "withheld" = .bool true) :
    e.kind = "open" ∧ e.schema = LEDGER_SCHEMA ∧ e.session = o.session ∧
      getD (getD e.body "window") "root" = .str w ∧ getD e.body "withheld" = .bool true := by
  obtain ⟨sO, objO, kindO, schemaO, bodyO, sessO, -⟩ := ledgerEntry_fields ho
  obtain ⟨sE, objE, kindE, schemaE, bodyE, sessE, -⟩ := ledgerEntry_fields he
  have F : ∀ k, k ≠ "sig" → JsonEquiv (getD o.entry k) (getD e.entry k) := fun k hk =>
    entry_fields_equiv sO sE objO objE hso hse hk
  have hK := F "kind" (by decide)
  rw [kindO, kindE, hk] at hK
  have hke : e.kind = "open" := Json.str.inj (equiv_str hK.symm)
  have hS := F "schema" (by decide)
  rw [schemaO, schemaE, hv2] at hS
  have hse2 : e.schema = LEDGER_SCHEMA := Json.str.inj (equiv_str hS.symm)
  have hI := F "session" (by decide)
  rw [sessO hk, sessE hke] at hI
  have hsess : e.session = o.session := Json.str.inj (equiv_str hI.symm)
  have hB := F "body" (by decide)
  have sBo := sorted_getD sO "body"
  have sBe := sorted_getD sE "body"
  rw [bodyO, bodyE] at hB
  rw [bodyO] at sBo
  rw [bodyE] at sBe
  have hW := getD_equiv sBo sBe hB "window"
  have hR := getD_equiv (sorted_getD sBo "window") (sorted_getD sBe "window") hW "root"
  rw [hr] at hR
  have hH := getD_equiv sBo sBe hB "withheld"
  rw [hw] at hH
  exact ⟨hke, hse2, hsess, equiv_str hR.symm, equiv_bool hH.symm⟩

/-- **The session the ledger opens for the window is one the device's own ledger opens on it, withheld, under v2**, or
the finder outputs a break. -/
theorem window_opens {w : String} {pins main : Json} {a : Audited}
    (hp : pinnedSession (getD coins "ledger") (Json.mkObj [("root", Json.str w)]) key pins main = .ok a)
    (hwd : withheldOn (getD coins "ledger") w = true) :
    OpensOn dev key.pub w a.session ∨ Break dev key label coins := by
  obtain ⟨ob, hao⟩ := Properties.Pouw.LiveSalt.Ledger.pinnedSession_ok hp
  obtain ⟨wp, win, en, hch, hfw, hsch, -, -, -, hsid, -⟩ := Properties.Pouw.LiveSalt.Ledger.auditedOpen_ok hao
  obtain ⟨es, hes, hvl⟩ := Properties.Pouw.LiveSalt.Ledger.checked_ok hch
  obtain ⟨v2, hI⟩ := Properties.Pouw.LiveSalt.Ledger.verifyLedger_inv hvl
  have hpred := Array.find?_some hfw
  obtain ⟨i, hi, hwi⟩ := Array.mem_iff_getElem.1 (Array.mem_of_find?_eq_some hfw)
  have hW := hI.win i hi
  rw [hwi] at hW
  obtain ⟨hOes, hkO, hroot, -⟩ := hW
  have hOent : win.opened ∈ Properties.Pouw.LiveSalt.entries (getD coins "ledger") := hes ▸ hOes
  obtain ⟨ls, hc, l, hl, hle⟩ : ∃ ls : Array Json, getD coins "ledger" = .arr ls ∧
      ∃ l ∈ ls.toList, ledgerEntry? l = some win.opened := by
    unfold Properties.Pouw.LiveSalt.entries at hOent
    split at hOent
    · rename_i ls hc
      obtain ⟨l, hl, he⟩ := List.mem_filterMap.1 hOent
      exact ⟨ls, hc, l, hl, he⟩
    · simp at hOent
  obtain ⟨-, -, -, -, -, -, hbodyOf⟩ := ledgerEntry_fields hle
  have hv2 : win.opened.schema = LEDGER_SCHEMA := beq_iff_eq.1 hsch
  rw [hkO, hv2, beq_self_eq_true] at hbodyOf
  obtain ⟨r, hr⟩ := open_root hbodyOf
  simp only [Properties.Pouw.LiveSalt.openRoot, hv2, beq_self_eq_true, ite_true, hr] at hroot
  have hrw : r = w := hroot.symm.trans ((beq_iff_eq.1 hpred).trans rfl)
  subst hrw
  have hwh := withheld_of hwd hc hl hle hkO hr
  obtain ⟨m, hm, hv⟩ := verifyLedger_sig hvl _ hOes
  have hm' : ledgerMessage (ident key.pub) win.opened.entry = .ok m := hm
  rcases signed_or_break (dev := dev) (label := label) (ledger_mem_signatures hc hl hle hm')
    (Session.lms_of_check hv) with hsigned | b
  · obtain ⟨tag, s1, ht1, hc1, hm1, hU1, hT1⟩ := ledger_hashed hle hm'
    have hU : ledgerBytes (ident key.pub) (Flock.Sha512.hash (taggedBytes tag s1)) ∈
        readHashed key label coins ++ deviceHashed dev key.pub :=
      List.mem_append_left _ (mem_readHashed_ledger hc hl hU1)
    unfold Signed signedBy at hsigned
    rcases List.mem_append.1 hsigned with hrun | hled
    · obtain ⟨s, hs, hms⟩ := List.mem_flatMap.1 hrun
      cases hcs : s.commitment with
      | none => rw [hcs] at hms; cases hms
      | some cs =>
        rw [hcs] at hms
        rcases List.mem_append.1 hms with hrec | hcl
        · cases hrm : recordMessage (ident key.pub) s.host cs with
          | error _ => rw [hrm] at hrec; cases hrec
          | ok m' =>
            rw [hrm] at hrec
            obtain rfl : m = m' := List.mem_singleton.1 hrec
            obtain ⟨d', hd', hm''⟩ := Session.recordMessage_ok hrm
            have hV : recordBytes (ident key.pub) s.host d' ∈
                readHashed key label coins ++ deviceHashed dev key.pub :=
              List.mem_append_right _ (Session.mem_deviceHashed_run hs hcs (.inl (Session.record_mem_closeHashed
                (Session.record_mem_recordHashed hd'))))
            rcases eq_or_break hU hV (hm1.symm.trans hm'') with e | b
            · exact absurd e.symm (recordBytes_ne_ledger _ _ _ _ _)
            · exact .inr b
        · cases hcm : closeMessage (ident key.pub) s.host cs s.id s.log.length (logChain s.log) s.ended with
          | error _ => rw [hcm] at hcl; cases hcl
          | ok m' =>
            rw [hcm] at hcl
            obtain rfl : m = m' := List.mem_singleton.1 hcl
            obtain ⟨rs, hrs, hm''⟩ := Session.closeMessage_ok hcm
            have hV : Session.closeBytes (ident key.pub) rs s.id s.log.length (logChain s.log) s.ended ∈
                readHashed key label coins ++ deviceHashed dev key.pub :=
              List.mem_append_right _ (Session.mem_deviceHashed_run hs hcs (.inl (Session.close_mem_closeHashed hrs)))
            rcases eq_or_break hU hV (hm1.symm.trans hm'') with e | b
            · exact absurd e.symm (Session.closeBytes_ne_ledger _ _ _ _ _ _ _ _)
            · exact .inr b
    · obtain ⟨line, hline, hlm⟩ := List.mem_filterMap.1 hled
      obtain ⟨e', he', hem⟩ := Option.bind_eq_some_iff.1 hlm
      obtain ⟨tag', s2, ht2, hc2, hm2, hU2, hT2⟩ := ledger_hashed he' (Session.toOption_some hem)
      have hV : ledgerBytes (ident key.pub) (Flock.Sha512.hash (taggedBytes tag' s2)) ∈
          readHashed key label coins ++ deviceHashed dev key.pub :=
        List.mem_append_right _ (Session.mem_deviceHashed_ledger hline hU2)
      rcases eq_or_break hU hV (hm1.symm.trans hm2) with e | b
      · have hTr : taggedBytes tag s1 ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_left _ (mem_readHashed_ledger hc hl hT1)
        have hTd : taggedBytes tag' s2 ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_right _ (Session.mem_deviceHashed_ledger hline hT2)
        rcases eq_or_break hTr hTd (ledgerBytes_inj e) with e2 | b
        · obtain ⟨rfl, rfl⟩ := taggedBytes_inj2 (ledgerTag_size ht1 ht2) e2
          obtain ⟨hke, hse, hsess, hre, hwe⟩ := opens_of_text hle he' hc1 hc2 hkO hv2 hr hwh
          exact .inl ⟨line, hline, e', he', hke, hse, hsess.trans hsid.symm, hre, hwe⟩
        · exact .inr b
      · exact .inr b
  · exact .inr b

/-- **A sample `drawCoins` accepts reads its coins against a run the device closed, a run the device's own ledger opens on
the sample's window**, withheld, under v2, or the finder outputs a break: `Session.drawCoins_session`'s run, with
`OpensOn`. -/
theorem drawCoins_opens {w : String} {after ξ : ByteArray} (h : drawCoins key w label after coins = .ok ξ) :
    ∃ c roots, challengerStream c (getD coins "segment") after
        (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ ∧
      ((∃ s ∈ dev.runs key.pub, OpensOn dev key.pub w s.id ∧ s.commitment = some c ∧ s.ended = none ∧
          releasedRootsOf c s.log = .ok roots) ∨ Break dev key label coins) := by
  obtain ⟨c, roots, ht, hs⟩ := Session.drawCoins_ok h
  refine ⟨c, roots, hs, ?_⟩
  obtain ⟨a, hp, hn⟩ := Session.trustedSessionOf_ok ht
  obtain ⟨⟨c', roots', sid⟩, ho, rfl, rfl, rfl⟩ := sessionNamed_sid hn
  rcases window_opens (dev := dev) (label := label) hp (drawCoins_withheld h) with hopen | b
  · rcases Session.sessionOf_bound (dev := dev) (label := label) ho with ⟨s, hs', hc, hid, he, hr⟩ | b
    · exact .inl ⟨s, hs', hid ▸ hopen, hc, he, hr⟩
    · exact .inr b
  · exact .inr b

end Properties.Challenger.Window
