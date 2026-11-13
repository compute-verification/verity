module

public import Proofs.Properties.Challenger.Session

/-!
# The reader accepts where each of its steps does

`Session.drawCoins_ok`, `Session.sessionNamed_ok` and their like read an acceptance back into what each step of the
reader returned. These are the converses, the steps a proof that the reader accepts a session assembles:

* `drawCoins` reads the stream where the evidence has its five keys, `trustedSessionOf` returns a commitment and its
  roots, the ledger opens the window withheld, the segment is of the family and `challengerStream` reads it
  (`drawCoins_ok_of_steps`), and `coinsReason` gives no reason where the copy of main's history is the caller's and
  the bytes read are the start of that stream (`coinsReason_none_of_steps`);
* `trustedSessionOf` returns what `sessionNamed` returns of the session `pinnedSession` returns
  (`trustedSessionOf_ok_of_steps`), and `pinnedSession` the session `auditedOpen` returns where the pins log and the
  copy of main's history give no reason (`pinnedSession_ok_of_steps`, `pinsReason_none_of` for a pins log of one line,
  `firstParentReason_none_of`);
* `auditedOpen` returns the session of the window the checked ledger opens under v2 and ends closed, in no period
  (`auditedOpen_ok_of_steps`);
* `sessionNamed` returns what `sessionOf` returns where the session's files are the ones the ledger's end names
  (`sessionNamed_ok_of_steps`), and `sessionOf` the record's commitment, the log's roots and the close's session where
  the record and the close are signed and the log is the one the close signs (`sessionOf_ok_of_steps`).

Each hypothesis is one step's result, so a proof of an acceptance supplies them one at a time.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private strOf isDigest jsonTrue from Spot.ChallengerSession

namespace Properties.Challenger.Steps

open Lean (Json)
open Flock (V)
open Spot Spot.Challenger

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

/-- `sessionOf` returns the record's commitment, the roots its release log shows and the close's session: the record is
signed under the key, the close's signature decodes and verifies over the close's message, the log has the lines the
close counts and chains to its chain, and the session's client closed it. -/
theorem sessionOf_ok_of_steps {key : ChallengerKey} {record close : Json} {lines : List String}
    {r : ChallengerRecord} {c : ChallengerClose} {m : ByteArray} {sig : Lms.Signature} {roots : Roots}
    (hrr : recordReason key record = .ok none) (hr : ChallengerRecord.ofJson record = .ok r)
    (hn : close.isNull = false) (hc : ChallengerClose.ofJson close = .ok c)
    (hm : closeMessage (key.pub.extract 0 16) r.host r.commitment c.session c.lines c.chain c.ended = .ok m)
    (hd : Lms.Signature.decode key.params c.signature = .ok sig)
    (hv : Lms.verify key.params key.pub m sig = .ok true)
    (hl : lines.length = c.lines) (hch : logChain lines = c.chain) (he : c.ended = none)
    (hroots : releasedRootsOf r.commitment lines = .ok roots) :
    sessionOf key record (.arr (lines.map Json.str).toArray) close = .ok (r.commitment, roots, c.session) := by
  rw [he] at hm
  unfold sessionOf
  simp only [hrr, hr, hn, hc, hm, hd, hv, he, bind, Except.bind, pure, Except.pure, Except.toOption,
    beq_self_eq_true, Bool.false_eq_true, ite_false, ite_true]
  rw [mapM_strs (fun _ => rfl)]
  simp only [hl, hch, hroots, bytes_beq, beq_self_eq_true, ite_true]

/-- `sessionNamed` returns what `sessionOf` returns of the session's files: the close names the ledger's session, the
record its host, the record is signed at the leaf the ledger's end names over the commitment it names, and the release
log and the refused hellos are the ones the end sealed. -/
theorem sessionNamed_ok_of_steps {a : Audited} {key : ChallengerKey} {session : Json} {c : LiveCommitment}
    {roots : Roots} {r : ChallengerRecord} {lines hellos : List String}
    (hk : keys? session = some #["close", "hellos", "log", "record"])
    (hs : sessionOf key (getD session "record") (getD session "log") (getD session "close") =
      .ok (c, roots, a.session))
    (hr : ChallengerRecord.ofJson (getD session "record") = .ok r) (hh : r.host = a.host)
    (hq : leafOf r.signature = a.recordLeaf) (hroot : Flock.hexOf c.root = a.commitment)
    (hlog : getD session "log" = .arr (lines.map Json.str).toArray)
    (hld : jsonStrIs (getD a.files LOG_FILE) (logDigest lines) = true)
    (hhel : getD session "hellos" = .arr (hellos.map Json.str).toArray)
    (hhd : jsonStrIs (getD a.files HELLOS_FILE) (logDigest hellos) = true)
    (hhn : hellos.length = a.refusedHellos) : sessionNamed a key session = .ok (c, roots) := by
  unfold sessionNamed
  simp only [hk, hs, hr, hh, hq, hroot, bind, Except.bind, pure, Except.pure, beq_self_eq_true, Bool.and_self,
    ite_true]
  simp only [hlog, hhel]
  rw [mapM_strs (fun _ => rfl)]
  simp only [hld, ite_true]
  rw [mapM_strs (fun _ => rfl)]
  simp only [hhd, hhn, beq_self_eq_true, Bool.and_self, ite_true]

/-! ## The ledger's session -/

/-- `auditedOpen` returns the session of the window `w` the checked ledger opens on the window's root under v2 and ends
closed, its end naming its record's leaf, its refused hellos, its commitment and the files it sealed, where the ledger
holds no period. -/
theorem auditedOpen_ok_of_steps {ledger window : Json} {key : ChallengerKey} {wins : Array Win} {w : Win}
    {e : LedgerEntry} {recordLeaf refusedHellos : Nat} (hw : isWindow window = true)
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

/-- `pinnedSession` returns the session `auditedOpen` returns where the pins log and the copy of main's history give
no reason. -/
theorem pinnedSession_ok_of_steps {ledger window : Json} {key : ChallengerKey} {pins main : Json} {a : Audited}
    {opened : Json} (ha : auditedOpen ledger window key = .ok (a, opened))
    (hp : pinsReason a.session opened pins = none)
    (hf : firstParentReason a.session (strOf (getD opened "pinned")) main = none) :
    pinnedSession ledger window key pins main = .ok a := by
  unfold pinnedSession
  simp only [ha, hp, hf, bind, Except.bind]
  rfl

/-- `trustedSessionOf` returns what `sessionNamed` returns of the session `pinnedSession` returns. -/
theorem trustedSessionOf_ok_of_steps {ledger window : Json} {key : ChallengerKey} {pins main session : Json}
    {a : Audited} {r : LiveCommitment × Roots} (hp : pinnedSession ledger window key pins main = .ok a)
    (hs : sessionNamed a key session = .ok r) : trustedSessionOf ledger window key pins main session = .ok r := by
  unfold trustedSessionOf
  simp only [hp, hs, bind, Except.bind]

/-! ## The draw -/

/-- **`drawCoins` reads the stream where each of its steps accepts**: the evidence has its five keys,
`trustedSessionOf` returns the session's commitment and roots, the ledger opens the window withheld, the segment is of
the family, and `challengerStream` reads the segment under that commitment and the family's roots. -/
theorem drawCoins_ok_of_steps {key : ChallengerKey} {window label : String} {after ξ : ByteArray} {coins : Json}
    {c : LiveCommitment} {roots : Roots}
    (hk : keys? coins = some #["ledger", "main", "pins", "segment", "session"])
    (ht : trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key (getD coins "pins")
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
