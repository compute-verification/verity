module

public import Proofs.Properties.Challenger.ChallengePhase
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# The reader accepts where each of its steps does

`ChallengePhase.drawCoins_ok`, `ChallengePhase.challengePhaseNamed_ok` and their like read an acceptance back into what each step of the
reader returned. These are the converses, the steps a proof that the reader accepts a session assembles:

* `drawCoins` reads the stream where the evidence has its five keys, `trustedChallengePhaseOf` returns a commitment and its
  roots, the ledger opens the window withheld, the segment is of the family and `challengerStream` reads it
  (`drawCoins_ok_of_steps`), and `coinsReason` gives no reason where the copy of main's history is the caller's and
  the bytes read are the start of that stream (`coinsReason_none_of_steps`);
* `trustedChallengePhaseOf` returns what `challengePhaseNamed` returns of the session `pinnedChallengePhase` returns
  (`trustedChallengePhaseOf_ok_of_steps`), and `pinnedChallengePhase` the session `auditedOpen` returns where the pins log and the
  copy of main's history give no reason (`pinnedChallengePhase_ok_of_steps`, `pinsReason_none_of` for a pins log of one line,
  `firstParentReason_none_of`);
* `auditedOpen` returns the session of the window the checked ledger opens under v2 and ends closed, in no period
  (`auditedOpen_ok_of_steps`);
* `challengePhaseNamed` returns what `challengePhaseOf` returns where the session's files are the ones the ledger's end names
  (`challengePhaseNamed_ok_of_steps`), and `challengePhaseOf` the record's commitment, the log's roots and the close's session where
  the record and the close are signed and the log is the one the close signs (`challengePhaseOf_ok_of_steps`).

Under them, for the smallest session a challenger closes, one window whose one family holds one coin:

* a ledger line reads as the entry of the record its text parses to (`ledgerEntry_some_of_steps`), the ledger of that
  window's open and its end verifies to the one window, ended (`verifyLedger_two`, `checked_two`), and opens it
  withheld (`withheldOn_two`);
* a record signed under the key gives no reason (`recordReason_none_of_steps`), the release log of that commitment's
  head, one root and its release shows the family's one root (`releasedRootsOf_one`), and `challengerStream` reads the
  one opening's coin (`challengerStream_one`).

Each hypothesis is one step's result, so a proof of an acceptance supplies them one at a time.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private strOf isDigest jsonTrue verifies join pairs isTime escapesSurrogate isBodyOf from Spot.ChallengerSession

namespace Properties.Challenger.Steps

open Lean (Json)
open Flock (V)
open _root_.Spot _root_.Spot.Challenger

/-- A list of strings reads back from its JSON strings. -/
theorem mapM_strs {f : Json → V String} (hf : ∀ s, f (.str s) = .ok s) (lines : List String) :
    (lines.map Json.str).mapM f = .ok lines := by
  induction lines with
  | nil => rfl
  | cons l ls ih =>
    simp only [List.map_cons, List.mapM_cons, ih, hf]
    rfl

theorem bytes_beq (a : ByteArray) : (a == a) = true := by
  show (a.data == a.data) = true
  exact beq_self_eq_true _

/-! ## The session's files -/

/-- `recordReason` gives no reason for a record whose signature decodes and verifies under the key over the record's
message. -/
theorem recordReason_none_of_steps {key : ChallengerKey} {record : Json} {r : ChallengerRecord} {m : ByteArray}
    {sig : Lms.Signature} (hn : record.isNull = false) (hr : ChallengerRecord.ofJson record = .ok r)
    (hm : recordMessage (key.pub.extract 0 16) r.host r.commitment = .ok m)
    (hd : Lms.Signature.decode key.params r.signature = .ok sig)
    (hv : Lms.verify key.params key.pub m sig = .ok true) : recordReason key record = .ok none := by
  unfold recordReason
  simp [hn, hr, hm, hd, hv, Except.toOption, pure, Except.pure, bind, Except.bind]

/-- `releasedRootsOf` shows the one root of a commitment's one family: the release log is the commitment's head, a root
line `a` and its release `b`, each its own event and no coin tree's, the root line naming the family, coin 0 and the
root, and the release naming them back. -/
theorem releasedRootsOf_one {c : LiveCommitment} {a b label root : String} {r s : Json} {bytes : ByteArray} {n : Nat}
    (hf : c.families = [(label, n)]) (hta : treeLine a = false) (htb : treeLine b = false)
    (hr : event a = .ok r) (hs : event b = .ok s)
    (hkr : keys? r = some #["event", "index", "label", "received_at", "root"])
    (hks : keys? s = some #["after", "event", "index", "label", "released_at"])
    (her : jsonStrIs (getD r "event") "root" = true) (hes : jsonStrIs (getD s "event") "release" = true)
    (htr : isTime (getD r "received_at") = true) (hts : isTime (getD s "released_at") = true)
    (hl : getD r "label" = .str label) (hroot : getD r "root" = .str root) (hb : Flock.unhex root = .ok bytes)
    (hir : int? (getD r "index") = some 0) (his : int? (getD s "index") = some 0)
    (hsl : jsonStrIs (getD s "label") label = true) (hsa : jsonStrIs (getD s "after") root = true)
    (hlen : root.length = 128) :
    releasedRootsOf c ["{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}", a, b] =
      .ok [(label, [bytes])] := by
  unfold releasedRootsOf
  simp [hf, hta, htb, hr, hs, hkr, hks, her, hes, htr, hts, hl, hroot, hb, hir, his, hsl, hsa, hlen, pairs,
    Except.toOption, pure, Except.pure, bind, Except.bind, -Array.any_eq_true, -Array.any_eq_false]

/-- `challengePhaseOf` returns the record's commitment, the roots its release log shows and the close's session: the record is
signed under the key, the close's signature decodes and verifies over the close's message, the log has the lines the
close counts and chains to its chain, and the session's client closed it. -/
theorem challengePhaseOf_ok_of_steps {key : ChallengerKey} {record close : Json} {lines : List String}
    {r : ChallengerRecord} {c : ChallengerClose} {m : ByteArray} {sig : Lms.Signature} {roots : Roots}
    (hrr : recordReason key record = .ok none) (hr : ChallengerRecord.ofJson record = .ok r)
    (hn : close.isNull = false) (hc : ChallengerClose.ofJson close = .ok c)
    (hm : closeMessage (key.pub.extract 0 16) r.host r.commitment c.session c.lines c.chain c.ended = .ok m)
    (hd : Lms.Signature.decode key.params c.signature = .ok sig)
    (hv : Lms.verify key.params key.pub m sig = .ok true)
    (hl : lines.length = c.lines) (hch : logChain lines = c.chain) (he : c.ended = none)
    (hroots : releasedRootsOf r.commitment lines = .ok roots) :
    challengePhaseOf key record (.arr (lines.map Json.str).toArray) close = .ok (r.commitment, roots, c.session) := by
  rw [he] at hm
  unfold challengePhaseOf
  simp only [hrr, hr, hn, hc, hm, hd, hv, he, bind, Except.bind, pure, Except.pure, Except.toOption,
    beq_self_eq_true, Bool.false_eq_true, ite_false, ite_true]
  rw [mapM_strs (fun _ => rfl)]
  simp only [hl, hch, hroots, beq_self_eq_true, ite_true]

/-- `challengePhaseNamed` returns what `challengePhaseOf` returns of the session's files: the close names the ledger's session, the
record its host, the record is signed at the leaf the ledger's end names over the commitment it names, and the release
log and the refused hellos are the ones the end sealed. -/
theorem challengePhaseNamed_ok_of_steps {a : Audited} {key : ChallengerKey} {session : Json} {c : LiveCommitment}
    {roots : Roots} {r : ChallengerRecord} {lines hellos : List String}
    (hk : keys? session = some #["close", "hellos", "log", "record"])
    (hs : challengePhaseOf key (getD session "record") (getD session "log") (getD session "close") =
      .ok (c, roots, a.session))
    (hr : ChallengerRecord.ofJson (getD session "record") = .ok r) (hh : r.host = a.host)
    (hq : leafOf r.signature = a.recordLeaf) (hroot : Flock.hexOf c.root = a.commitment)
    (hlog : getD session "log" = .arr (lines.map Json.str).toArray)
    (hld : jsonStrIs (getD a.files LOG_FILE) (logDigest lines) = true)
    (hhel : getD session "hellos" = .arr (hellos.map Json.str).toArray)
    (hhd : jsonStrIs (getD a.files HELLOS_FILE) (logDigest hellos) = true)
    (hhn : hellos.length = a.refusedHellos) : challengePhaseNamed a key session = .ok (c, roots) := by
  unfold challengePhaseNamed
  simp only [hk, hs, hr, hh, hq, hroot, bind, Except.bind, pure, Except.pure, beq_self_eq_true, Bool.and_self,
    ite_true]
  simp only [hlog, hhel]
  rw [mapM_strs (fun _ => rfl)]
  simp only [hld, ite_true]
  rw [mapM_strs (fun _ => rfl)]
  simp only [hhd, hhn, beq_self_eq_true, Bool.and_self, ite_true]

/-! ## The ledger's lines -/

/-- `ledgerEntry?` reads a line as the entry of the record its text parses to: the text is short enough, parses to a
value it is the canonical text of, escapes no surrogate, and the value is a session's record of the ledger's form. -/
theorem ledgerEntry_some_of_steps {text session tag t : String} {e : Json} {r : Core.Record}
    (hlen : text.length ≤ MAX_LEDGER_LINE) (hp : Flock.JsonParse.parse text = .ok e)
    (hpy : PyJson.canonical e = .ok text) (hes : escapesSurrogate text = false) (hc : Flock.canon e = .ok t)
    (hs : sortedKeys e = true) (hr : Core.Record.ofJson e = .ok r) (ht : ledgerTag? r.schema = some tag)
    (h4 : 4 ≤ r.sig.size) (hk : r.kind = "open" ∨ r.kind = "end") (hse : r.session = some session)
    (hid : isChallengePhaseId session = true) (hb : isBodyOf r.kind (r.schema == LEDGER_SCHEMA) r.body = true) :
    ledgerEntry? (.str text) = some ⟨e, text, r.schema, r.seq, r.prev, r.kind, session, r.sig, r.body⟩ := by
  unfold ledgerEntry?
  rcases hk with hk | hk <;> rw [hk] at hb <;>
    simp [hlen, hp, hpy, hes, hc, hs, hr, ht, h4, hk, hse, hid, hb, Except.toOption, guard, bind, Option.bind,
      pure, failure]

/-- No lane has a period open, so a window joins none. -/
theorem join_none (periods : Array MonitoringPeriod) (i : Nat) (lane session : String) :
    join periods #[] i lane session = .ok (periods, none) := by
  unfold join
  simp [pure, Except.pure]

/-- `verifyDays` accepts a ledger of one open and its end: no lane declares a day. -/
theorem verifyDays_two {o e : LedgerEntry} (hok : o.kind = "open") (hek : e.kind = "end") :
    verifyDays #[o, e] = .ok () := by
  unfold verifyDays
  simp [Array.zipIdx, hok, hek, pure, Except.pure, bind, Except.bind, -Array.any_eq_true, -Array.any_eq_false]

/-- **`verifyLedger` reads the ledger of one window**, its v2 open `o` and its end `e`, as that window, ended, in no
period: the open is entry 0 after `ZERO` and names no reserve, the end is entry 1 chained to the open's text and of
its session, each is signed under the key, and the end's record leaf `r` is between the leaves the two were signed
at. -/
theorem verifyLedger_two {key : ChallengerKey} {o e : LedgerEntry} {mo me : ByteArray} {root : String} {r : Nat}
    (hos : o.schema = LEDGER_SCHEMA) (hoq : o.seq = 0) (hop : o.prev = ZERO) (hok : o.kind = "open")
    (hom : ledgerMessage (key.pub.extract 0 16) o.entry = .ok mo) (hov : verifies key mo o.sig = true)
    (hor : strOf (getD (getD o.body "window") "root") = root) (hof : field? o.body "reserve" = none)
    (hes : e.schema = LEDGER_SCHEMA) (heq : e.seq = 1)
    (hep : e.prev = Flock.hexOf (Flock.Sha512.hash o.text.toUTF8)) (hek : e.kind = "end")
    (hem : ledgerMessage (key.pub.extract 0 16) e.entry = .ok me) (hev : verifies key me e.sig = true)
    (hse : e.session = o.session) (hrl : nat? (getD e.body "record_leaf") = some r)
    (hrn : (getD e.body "record_leaf").isNull = false) (h1 : leafOf o.sig < r) (h2 : r < leafOf e.sig) :
    verifyLedger #[o, e] key = .ok (#[{ root, opened := o, leaf := leafOf o.sig, ended := some e }], #[]) := by
  have hl0 : (-1 : Int) < (leafOf o.sig : Int) := by omega
  have hl1 : (leafOf o.sig : Int) < (leafOf e.sig : Int) := by omega
  have hn0 : r ≠ leafOf o.sig := by omega
  have hn1 : r ≠ leafOf e.sig := by omega
  unfold verifyLedger
  simp [Array.zipIdx, hos, hoq, hop, hok, hom, hov, hor, hof, join_none, hes, heq, hep, hek, hem, hev, hse, hrl, hrn,
    h1, h2, verifyDays_two hok hek, hl0, hl1, hn0, hn1, LEDGER_SCHEMA, LEDGER_SCHEMA_V1, bind, Except.bind, pure,
    Except.pure, -Array.any_eq_true, -Array.any_eq_false]

/-- `checked` returns what `verifyLedger` returns of a ledger's two lines, an open under the pinned key and an end,
each read by `ledgerEntry?`. -/
theorem checked_two {key : ChallengerKey} {l0 l1 : Json} {o e : LedgerEntry} {r : Array Win × Array MonitoringPeriod}
    (h0 : ledgerEntry? l0 = some o) (h1 : ledgerEntry? l1 = some e) (hok : o.kind = "open") (hek : e.kind = "end")
    (hkey : strOf (getD o.body "key_sha512") = Flock.hexOf (Flock.Sha512.hash key.pub))
    (hv : verifyLedger #[o, e] key = .ok r) : checked (.arr #[l0, l1]) key = .ok r := by
  unfold checked
  simp [Array.zipIdx, h0, h1, hok, hek, hkey, hv, Json.isNull, pure, Except.pure, bind, Except.bind,
    -Array.any_eq_true, -Array.any_eq_false]

/-- `withheldOn` holds of a ledger of one open and its end whose open names the window and marks its session
withheld. -/
theorem withheldOn_two {l0 l1 : Json} {o e : LedgerEntry} {window : String} (h0 : ledgerEntry? l0 = some o)
    (h1 : ledgerEntry? l1 = some e) (hok : o.kind = "open") (hek : e.kind = "end")
    (hr : getD (getD o.body "window") "root" = .str window) (hw : getD o.body "withheld" = .bool true) :
    withheldOn (.arr #[l0, l1]) window = true := by
  unfold withheldOn
  simp [h0, h1, hok, hek, hr, hw, Option.filter]

/-! ## The ledger's session -/

/-- `auditedOpen` returns the session of the window `w` the checked ledger opens on the window's root under v2 and ends
closed, its end naming its record's leaf, its refused hellos, its commitment and the files it sealed, where the ledger
holds no period. -/
theorem auditedOpen_ok_of_steps {ledger window : Json} {key : ChallengerKey} {wins : Array Win} {w : Win}
    {e : LedgerEntry} {recordLeaf refusedHellos : Nat} (hw : isRun window = true)
    (hc : checked ledger key = .ok (wins, #[]))
    (hf : wins.find? (·.root == strOf (getD window "root")) = some w)
    (hs : w.opened.schema = LEDGER_SCHEMA) (he : w.ended = some e)
    (hcl : jsonTrue (getD e.body "closed") = true)
    (hrl : nat? (getD e.body "record_leaf") = some recordLeaf)
    (hrh : nat? (getD e.body "refused_hellos") = some refusedHellos)
    (hcm : isDigest (getD e.body "commitment") = true)
    (hfl : SEALED_FILES.all (fun f => (field? (getD e.body "files") f).isSome) = true) :
    auditedOpen ledger window key = .ok
      ({ session := w.opened.session, host := strOf (getD w.opened.body "host"), prover := laneOf w.opened.body,
         files := getD e.body "files", recordLeaf, commitment := strOf (getD e.body "commitment"), refusedHellos,
         pinned := strOf (getD w.opened.body "pinned"), period := none, cap := Json.null, circuit := Json.null },
        w.opened.body) := by
  unfold auditedOpen
  simp only [hw, hc, hf, hs, he, hcl, hrl, hrh, hcm, hfl, bind, Except.bind, pure, Except.pure, beq_self_eq_true,
    Bool.and_self, ite_true]
  rfl

/-- `pinsReason` gives no reason for a pins log of one line: the line reads as a pin (`pinOf?`) of the commit the open
names, and the open seals that one line. -/
theorem pinsReason_none_of {session : String} {opened line : Json} {c : String} (hp : pinOf? line = some c)
    (hn : nat? (getD (getD opened "pins") "lines") = some 1)
    (hk : keys? (getD opened "pins") = some #["lines", "sha512"])
    (hd : isDigest (getD (getD opened "pins") "sha512") = true)
    (hs : jsonStrIs (getD (getD opened "pins") "sha512") (logDigest [strOf line]) = true)
    (hc : c = strOf (getD opened "pinned")) : pinsReason session opened (.arr #[line]) = none := by
  subst hc
  unfold pinsReason
  simp only [Id.run, Json.isNull, Bool.false_eq_true, ↓reduceIte, Array.zipIdx, Nat.zero_add,
    List.mapIdx_toArray, List.mapIdx_cons, List.mapIdx_nil, Nat.toString_eq_repr, List.forIn_toArray, List.forIn_cons,
    hp, List.push_toArray, List.nil_append, List.forIn_nil, pure_bind, hn, hk, BEq.rfl, Std.le_refl, decide_true,
    Bool.and_self, hd, List.size_toArray, List.length_cons, List.length_nil, Nat.lt_irrefl, List.extract_toArray,
    List.extract_eq_take_drop, Nat.sub_zero, List.drop_zero, List.take_succ_cons, List.take_nil, List.map_cons,
    List.map_nil, hs, Array.contains_eq_mem, decide_eq_true_eq, List.mem_toArray, List.mem_cons, List.not_mem_nil,
    or_false]
  rfl

/-- `firstParentReason` gives no reason where the copy of main's history is a list of commits that holds the pinned
one. -/
theorem firstParentReason_none_of {session pinned : String} {cs : Array Json}
    (hd : cs.all (isDigest · 40) = true) (hp : cs.any (jsonStrIs · pinned) = true) :
    firstParentReason session pinned (.arr cs) = none := by
  unfold firstParentReason
  simp only [hd, hp, Bool.not_true, Bool.false_eq_true, ite_false, ite_true]

/-- `pinnedChallengePhase` returns the session `auditedOpen` returns where the pins log and the copy of main's history give
no reason. -/
theorem pinnedChallengePhase_ok_of_steps {ledger window : Json} {key : ChallengerKey} {pins main : Json} {a : Audited}
    {opened : Json} (ha : auditedOpen ledger window key = .ok (a, opened))
    (hp : pinsReason a.session opened pins = none)
    (hf : firstParentReason a.session (strOf (getD opened "pinned")) main = none) :
    pinnedChallengePhase ledger window key pins main = .ok a := by
  unfold pinnedChallengePhase
  simp only [ha, hp, hf, bind, Except.bind]
  rfl

/-- `trustedChallengePhaseOf` returns what `challengePhaseNamed` returns of the session `pinnedChallengePhase` returns. -/
theorem trustedChallengePhaseOf_ok_of_steps {ledger window : Json} {key : ChallengerKey} {pins main session : Json}
    {a : Audited} {r : LiveCommitment × Roots} (hp : pinnedChallengePhase ledger window key pins main = .ok a)
    (hs : challengePhaseNamed a key session = .ok r) : trustedChallengePhaseOf ledger window key pins main session = .ok r := by
  unfold trustedChallengePhaseOf
  simp only [hp, hs, bind, Except.bind]

/-! ## The sample -/

/-- `challengerStream` reads the coin of a segment's one opening: the segment is of the family from coin 0, opened
after `after`, its opening is coin 0's under the commitment (`liveReason`), and the family's roots are the one
`after`. -/
theorem challengerStream_one {c : LiveCommitment} {segment doc : Json} {after : ByteArray} {label : String}
    {o : LiveOpening} (hk : keys? segment = some #["after", "label", "openings", "start"])
    (hl : getD segment "label" = .str label) (hst : nat? (getD segment "start") = some 0)
    (hop : getD segment "openings" = .arr #[doc]) (ha : jsonStrIs (getD segment "after") (Flock.hexOf after) = true)
    (ho : LiveOpening.ofJson doc = .ok o) (hr : liveReason c o label 0 after = .ok none) :
    challengerStream c segment after (some [after]) = .ok o.coin := by
  unfold challengerStream
  simp [hk, hl, hst, hop, ha, ho, hr, pure, Except.pure, bind, Except.bind, -Array.any_eq_true,
    -Array.any_eq_false]

/-- **`drawCoins` reads the stream where each of its steps accepts**: the evidence has its five keys,
`trustedChallengePhaseOf` returns the session's commitment and roots, the ledger opens the window withheld, the segment is of
the family, and `challengerStream` reads the segment under that commitment and the family's roots. -/
theorem drawCoins_ok_of_steps {key : ChallengerKey} {window label : String} {after ξ : ByteArray} {coins : Json}
    {c : LiveCommitment} {roots : Roots}
    (hk : keys? coins = some #["ledger", "main", "pins", "segment", "session"])
    (ht : trustedChallengePhaseOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key (getD coins "pins")
      (getD coins "main") (getD coins "session") = .ok (c, roots))
    (hw : withheldOn (getD coins "ledger") window = true)
    (hl : jsonStrIs (getD (getD coins "segment") "label") label = true)
    (hs : challengerStream c (getD coins "segment") after
      (some (((roots.find? (·.1 == label)).map (·.2)).getD [])) = .ok ξ) :
    drawCoins key window label after coins = .ok ξ := by
  unfold drawCoins
  simp only [hk, ht, hw, hl, hs, bind, Except.bind, beq_self_eq_true]
  rfl

/-- **`coinsReason` gives no reason where `drawCoins` reads a stream** that the bytes read are the start of, every
coin of it read, the evidence's copy of main's history being the caller's. -/
theorem coinsReason_none_of_steps {key : ChallengerKey} {main coins : Json} {window label : String}
    {after read ξ : ByteArray} (hd : drawCoins key window label after coins = .ok ξ)
    (hm : sameMain main (getD coins "main") = true) (hr : startReason ξ read = none) :
    coinsReason key main coins window label after read = none := by
  unfold coinsReason
  simp only [hd, hm, hr, Bool.not_true, Bool.false_eq_true, ite_false]

end Properties.Challenger.Steps
