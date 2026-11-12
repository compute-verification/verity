/-
A read across streams: a later stream's registration whose `previous` names a record of an earlier stream, so that
the later stream's blocks read that record's ports as registered values. `across` gives three verdicts on it, each
computed whatever the others say:

* `logged`: the later registration's `previous` is the digest of a record of the earlier stream's log (`loggedAt`), the
  registrations that stream received, in order;
* `order`: in the one session the auditor's ledger opens on the window (`Challenger.trustedSessionOf`), the bytes the later
  stream's sample read are exactly the coins of the later stream's family (`family`, which the caller names and the
  evidence does not) that the challenger's signed log shows released after the later registration
  (`Challenger.coinsReason`), and that log shows a coin released after the earlier registration, the last of those before
  the first released after the later one (`receiptReason`): the record whose ports the later blocks read was received
  first, and no coin that sampled one of them was released before the challenger received what fixes its rows;
* `audit`: the earlier stream's audit record (`Record.record`'s) is the earlier registration's and accepted, with
  `complete`, whether that record says every one of its entries ran (`auditVerdict`); no verdict when none is handed.

The audit record is the caller's, and it is held to nothing: `across` checks its format and its `registration_digest`,
reads its `accepted` and `complete` as it gives them and none of its entries, and does not hold it to the record the
earlier stream's verifier wrote. So `audit`, and with it case 1, says no more than the record handed says.

`case` names the outcome: 1 when all three hold and the earlier stream's audit record is complete; 2 when the earlier
stream's log holds no such record or the order is not the log's; 3 when those hold and the earlier stream's audit record
is missing or did not accept; 4 when those hold and that audit record accepted and is not complete: it accepted with
entries that did not run, and is no audit of those entries, so a later record does not rest on it as on one that is.

The order is read from one session's signed log over every family's lines, in the log's own order (`receivedOf`). A
family's coins follow one commitment (`Challenger.FAMILY_ROOT`), so the earlier registration's coins are another
family's than the later sample's, and no one family's roots show both receipts.
-/
module

public import Spot.Record
public import Spot.ChallengerCoins
meta import Spot.Record
meta import Spot.ChallengerCoins

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Record

open Lean (Json)
open Flock (V)

/-- One verdict of `across`: accepted, refused with its reason, or none. -/
structure AcrossVerdict where
  accepted : Option Bool
  why : Option String := none

/-- `{"accepted": true | false | null}`, with `why` when refused. -/
def AcrossVerdict.toJson (v : AcrossVerdict) : Json :=
  Json.mkObj (("accepted", match v.accepted with | some b => Json.bool b | none => Json.null) ::
    (v.why.map fun w => ("why", Json.str w)).toList)

/-- The verdict of a check that gives its reason for refusing, or none when it holds. -/
def AcrossVerdict.ofReason : Option String → AcrossVerdict
  | none => { accepted := some true }
  | some why => { accepted := some false, why := some why }

/-- The index of the first record of `log` whose digest (`Flock.WindowRecord.digestHex`) is `previous`. -/
def loggedAt (log : Array Json) (previous : String) : Option Nat :=
  log.findIdx? fun r => match Flock.WindowRecord.digestHex r with
    | .ok d => d == previous
    | .error _ => false

def UNLOGGED : String := "the later registration's previous is no record of the earlier stream's log"

/-- Why `received`, the roots a session's log shows (the root the challenger had received when it released each coin, of
whatever family, in the log's order), does not show `earlier` received before every coin released after `later`: no
coin was released after one of them, or the last coin after `earlier` is not before the first after `later`. A coin's
number in the reason is its place among the log's coins, over every family. -/
def receiptReason (received : List ByteArray) (earlier later : ByteArray) : Option String :=
  let coinsAfter (x : ByteArray) := (received.zipIdx.filter (·.1 == x)).map (·.2)
  match (coinsAfter earlier).getLast?, (coinsAfter later).head? with
  | none, _ => some "the challenger's log shows no coin released after the earlier registration: its receipt is not in the log"
  | some _, none => some "the challenger's log shows no coin released after the later registration: its receipt is not in the log"
  | some a, some b =>
    if a < b then none
    else some s!"the challenger's log shows coin {b} released after the later registration and coin {a} after the earlier registration: the earlier registration's coins come first"

/-- The roots of the `root` events among the release log `log`'s lines after its head, in the log's own order: for
each coin the challenger released, of whatever family, the root it had received (`Challenger.releasedRootsOf` reads the
same lines family by family). A line that is no event, and an event that is no `root`, give none. It checks no line's
form: `receivedOf` reads a log only once `Challenger.trustedSessionOf` holds it. -/
def rootLines (log : Json) : List ByteArray :=
  match log with
  | .arr lines => lines.toList.tail.filterMap fun
    | .str line =>
      match Challenger.event line with
      | .ok e =>
        if jsonStrIs (getD e "event") "root" then
          match getD e "root" with
          | .str r => (Flock.unhex r).toOption
          | _ => none
        else none
      | .error _ => none
    | _ => none
  | _ => []

/-- The roots the challenger received, one for each coin it released, over every family's lines in its signed log's own
order (`rootLines`), from the session the auditor's ledger opens on `window`, once that session's files hold
(`Challenger.trustedSessionOf`, as `Challenger.coinsReason` reads them). -/
def receivedOf (coins : Json) (window : String) : V (List ByteArray) := do
  let session := getD coins "session"
  let _ ← Challenger.trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)])
    (← Challenger.ChallengerKey.ofJson (getD coins "key")) (getD coins "pins") (getD coins "main") session
  return rootLines (getD session "log")

def NO_LOG : String := "the challenger's session published no signed log to read the order from"

def NO_FAMILY : String := "the later sample read no coins of a challenger's family"

/-- Why the later stream's sample does not hold the order its registration rests on: it has no coin evidence (`coins`
null), the caller names no family it read (`family` none), its coins are not the ones of that family the challenger's
signed log shows released after `later` (`Challenger.coinsReason` on the bytes `read`), or that log does not show
`earlier` received first (`receiptReason`, over every family's lines). -/
def orderReason (window : String) (family : Option String) (earlier later : ByteArray) (coins : Json)
    (read : ByteArray) : Option String :=
  match coins, family with
  | .null, _ => some NO_LOG
  | _, none => some NO_FAMILY
  | _, some label =>
    match Challenger.coinsReason coins window label later read with
    | some why => some s!"the later sample's coins: {why}"
    | none =>
      match receivedOf coins window with
      | .error why => some why
      | .ok received => receiptReason received earlier later

/-- The verdict on the earlier stream's audit record: whether it is the earlier registration's and accepted, and
whether it is complete, every one of its entries having run; `why` when it did not accept, or accepted and is not
complete. -/
structure AuditVerdict where
  accepted : Option Bool
  complete : Option Bool
  why : Option String := none

/-- `{"accepted": true | false | null, "complete": true | false | null}`, with `why` when there is one. -/
def AuditVerdict.toJson (v : AuditVerdict) : Json :=
  let tri : Option Bool → Json
    | some b => Json.bool b
    | none => Json.null
  Json.mkObj ([("accepted", tri v.accepted), ("complete", tri v.complete)] ++
    (v.why.map fun w => ("why", Json.str w)).toList)

/-- Why `audit` is no audit record of the registration `previous` (its digest, hex): it is no audit record, or it is
another registration's. -/
def auditOther (audit : Json) (previous : String) : Option String :=
  if !jsonStrIs (getD audit "format") FORMAT then some s!"not a {FORMAT} record"
  else if !jsonStrIs (getD audit "registration_digest") previous then
    some "the audit record is another registration's than the one the later registration names"
  else none

def NOT_ACCEPTED : String := "the earlier stream's audit record did not accept"

def NOT_COMPLETE : String :=
  "the earlier stream's audit record accepted with entries that did not run: it is not complete, and is no audit of those entries"

private def saysTrue : Json → Bool
  | .bool true => true
  | _ => false

/-- The verdict on `audit`, the audit record handed for the registration `previous` (its digest, hex): none when none
is handed (null); refused, with no word on `complete`, when it is no audit record of that registration (`auditOther`);
else `accepted` and `complete` as the record's own fields say, each true only where the field is `true`. `audit` is
the caller's record and is held to nothing: no entry of it is read, and nothing holds it to the record the earlier
stream's verifier wrote, so the verdict says no more than the record handed says. -/
def auditVerdict (audit : Json) (previous : String) : AuditVerdict :=
  match audit with
  | .null => { accepted := none, complete := none }
  | a =>
    match auditOther a previous with
    | some why => { accepted := some false, complete := none, why := some why }
    | none =>
      let accepted := saysTrue (getD a "accepted")
      let complete := saysTrue (getD a "complete")
      { accepted := some accepted, complete := some complete,
        why := if !accepted then some NOT_ACCEPTED else if !complete then some NOT_COMPLETE else none }

/-- What `across` reads: the window's open root the verifier holds, the earlier stream's log, its audit record (null:
none), the later stream's registration, and the later stream's sample's coin evidence, the challenger's family the
later stream read those coins from as the caller names it (none: it read none of a family), and the bytes it read. -/
structure Across where
  window : String
  log : Array Json
  audit : Json
  later : Json
  coins : Json
  family : Option String
  read : ByteArray

/-- 2 when `logged` or `order` does not accept; else 3 when `audit` does not; else 4 when it does and is not complete;
else 1: all three accept and the audit record is complete. -/
def acrossCase (logged order : AcrossVerdict) (audit : AuditVerdict) : Nat :=
  if logged.accepted != some true || order.accepted != some true then 2
  else if audit.accepted != some true then 3
  else if audit.complete != some true then 4
  else 1

/-- What `across` finds on one read: its three verdicts, the earlier record's index in the earlier stream's log (none
when it holds none) and the two registrations' digests in hex. -/
structure AcrossFound where
  logged : AcrossVerdict
  received : Option Nat
  order : AcrossVerdict
  audit : AuditVerdict
  earlier : String
  later : String

def AcrossFound.caseOf (f : AcrossFound) : Nat := acrossCase f.logged f.order f.audit

def AcrossFound.fields (f : AcrossFound) : List (String × Json) :=
  [("logged", f.logged.toJson), ("received", match f.received with | some k => Lean.toJson k | none => Json.null),
   ("order", f.order.toJson), ("audit", f.audit.toJson), ("case", Lean.toJson f.caseOf),
   ("earlier", Json.str f.earlier), ("later", Json.str f.later)]

/-- One read across streams (module docstring). Refused: a later registration that names no previous record in hex, or
whose digest is not defined. -/
def acrossFound (i : Across) : V AcrossFound := do
  let .str previous := getD i.later "previous" | throw "the later registration names no previous record"
  let earlier ← Flock.unhex previous
  let later ← Flock.WindowRecord.digest i.later
  let received := loggedAt i.log previous
  return { logged := .ofReason (if received.isSome then none else some UNLOGGED), received,
           order := .ofReason (orderReason i.window i.family earlier later i.coins i.read),
           audit := auditVerdict i.audit previous, earlier := previous, later := Flock.hexOf later }

/-- The verdicts on one read across streams (`acrossFound`): `{"logged", "received", "order", "audit", "case",
"earlier", "later"}`, `received` the earlier record's index in the log (null when it holds none), `earlier` and
`later` the two registrations' digests in hex. -/
def across (i : Across) : V Json := return Json.mkObj (← acrossFound i).fields

/-! ## The cases -/

private def root (n : Nat) : ByteArray := ByteArray.mk #[n.toUInt8]

-- coins 0 and 1 after the earlier registration, coins 2 and 3 after the later one
#guard receiptReason [root 1, root 1, root 2, root 2] (root 1) (root 2) == none
-- a root between them changes nothing
#guard receiptReason [root 1, root 9, root 2] (root 1) (root 2) == none
-- a coin after the earlier registration follows one after the later
#guard (receiptReason [root 1, root 2, root 1, root 2] (root 1) (root 2)).isSome
#guard (receiptReason [root 2, root 1] (root 1) (root 2)).isSome
-- a receipt the log does not hold
#guard (receiptReason [root 2, root 2] (root 1) (root 2)).isSome
#guard (receiptReason [root 1, root 1] (root 1) (root 2)).isSome
#guard (receiptReason [] (root 1) (root 2)).isSome
-- one registration named on both sides is not before itself
#guard (receiptReason [root 1, root 1] (root 1) (root 1)).isSome
-- a sample with no coin evidence holds no order, and neither does one of no family
#guard orderReason "" (some "draw/1") (root 1) (root 2) .null (root 0) == some NO_LOG
#guard orderReason "" none (root 1) (root 2) .null (root 0) == some NO_LOG
#guard orderReason "" none (root 1) (root 2) (Json.mkObj []) (root 0) == some NO_FAMILY

private def wide (n : Nat) : ByteArray := ByteArray.mk (List.replicate 64 n.toUInt8).toArray

private def lineOf (fields : List (String × Json)) : String := (Flock.canon (Json.mkObj fields)).toOption.getD ""

private def rootLine (label : String) (index : Nat) (r : ByteArray) : String :=
  lineOf [("event", Json.str "root"), ("index", Lean.toJson index), ("label", Json.str label),
    ("received_at", Lean.toJson (0 : Nat)), ("root", Json.str (Flock.hexOf r))]

private def releaseLine (label : String) (index : Nat) (r : ByteArray) : String :=
  lineOf [("after", Json.str (Flock.hexOf r)), ("event", Json.str "release"), ("index", Lean.toJson index),
    ("label", Json.str label), ("released_at", Lean.toJson (0 : Nat))]

private def threeFamilies : Challenger.LiveCommitment :=
  { scheme := "", epoch := 0, families := [("window", 1), ("draw/0", 1), ("draw/1", 2)], root := wide 0 }

/-- A release log as a session of three families leaves it: its head, the window's coin, the earlier registration's
coin in one family, a coin tree's line, and the later registration's two coins in another. -/
private def threeLines : List String :=
  [lineOf [("event", Json.str "commitment"), ("root", Json.str (Flock.hexOf (wide 0)))],
   rootLine "window" 0 (wide 9), releaseLine "window" 0 (wide 9),
   rootLine "draw/0" 0 (wide 1), releaseLine "draw/0" 0 (wide 1),
   lineOf [("event", Json.str Challenger.TREE_EVENT)],
   rootLine "draw/1" 0 (wide 2), releaseLine "draw/1" 0 (wide 2),
   rootLine "draw/1" 1 (wide 2), releaseLine "draw/1" 1 (wide 2)]

private def logOf (lines : List String) : Json := Json.arr (lines.map Json.str).toArray

-- the log is one `Challenger.releasedRootsOf` reads, family by family
#guard (Challenger.releasedRootsOf threeFamilies threeLines).toOption ==
  some [("window", [wide 9]), ("draw/0", [wide 1]), ("draw/1", [wide 2, wide 2])]
-- the same lines' roots over every family, in the log's order: the head, the releases and a coin tree's line give none
#guard rootLines (logOf threeLines) == [wide 9, wide 1, wide 2, wide 2]
-- the order across two families: the earlier registration's coin, then the later's two
#guard receiptReason (rootLines (logOf threeLines)) (wide 1) (wide 2) == none
#guard (receiptReason (rootLines (logOf threeLines)) (wide 2) (wide 1)).isSome
-- the head is no coin's line, whatever it says; no log gives no root
#guard rootLines (logOf [rootLine "draw/0" 0 (wide 1)]) == []
#guard rootLines (logOf []) == []
#guard rootLines .null == []
-- a line that is not its own canonical JSON is no event, and neither is one that is no string
#guard rootLines (Json.arr #[Json.str "", Json.str "{ \"event\": \"root\", \"root\": \"01\" }", Json.null]) == []
-- `rootLines` holds a line to no more than that: the log's form is `Challenger.trustedSessionOf`'s to check, before it
#guard rootLines (Json.arr #[Json.str "", Json.str "{\"event\":\"root\",\"root\":\"01\"}"]) == [root 1]

private def audited (digest : String) (accepted : Json) (complete : Json := .bool true) : Json :=
  Json.mkObj [("format", Json.str FORMAT), ("registration_digest", Json.str digest), ("accepted", accepted),
    ("complete", complete)]

private def verdictOn (audit : Json) : Option Bool × Option Bool :=
  let v := auditVerdict audit "ab"
  (v.accepted, v.complete)

#guard verdictOn (audited "ab" (.bool true)) == (some true, some true)
#guard (auditVerdict (audited "ab" (.bool true)) "ab").why == none
-- an accept that is not complete: a `complete` that is `false`, is not `true`, or is not there
#guard verdictOn (audited "ab" (.bool true) (.bool false)) == (some true, some false)
#guard (auditVerdict (audited "ab" (.bool true) (.bool false)) "ab").why == some NOT_COMPLETE
#guard verdictOn (audited "ab" (.bool true) .null) == (some true, some false)
#guard verdictOn (audited "ab" (.bool true) (.str "true")) == (some true, some false)
#guard verdictOn (Json.mkObj [("format", Json.str FORMAT), ("registration_digest", Json.str "ab"),
  ("accepted", .bool true)]) == (some true, some false)
-- a record that did not accept, complete or not
#guard verdictOn (audited "ab" (.bool false)) == (some false, some true)
#guard (auditVerdict (audited "ab" (.bool false) (.bool false)) "ab").why == some NOT_ACCEPTED
#guard verdictOn (audited "ab" .null) == (some false, some true)
#guard verdictOn (audited "ab" (.str "true")) == (some false, some true)
-- no audit record of this registration
#guard verdictOn (audited "cd" (.bool true)) == (some false, none)
#guard verdictOn (Json.mkObj [("registration_digest", Json.str "ab"), ("accepted", .bool true),
  ("complete", .bool true)]) == (some false, none)
#guard verdictOn .null == (none, none)

private def reg (n : Nat) : Json :=
  Json.mkObj [("format", Json.str "verity/registration/v2"), ("population", Lean.toJson n)]

private def digestOf (r : Json) : String := (Flock.WindowRecord.digestHex r).toOption.getD ""

#guard loggedAt #[reg 1, reg 2] (digestOf (reg 2)) == some 1
#guard loggedAt #[reg 1, reg 2] (digestOf (reg 3)) == none
#guard loggedAt #[] (digestOf (reg 1)) == none

#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨some true, some true, none⟩ == 1
#guard acrossCase ⟨some false, none⟩ ⟨some true, none⟩ ⟨some true, some true, none⟩ == 2
#guard acrossCase ⟨some true, none⟩ ⟨some false, none⟩ ⟨none, none, none⟩ == 2
#guard acrossCase ⟨some false, none⟩ ⟨some true, none⟩ ⟨some true, some false, none⟩ == 2
#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨none, none, none⟩ == 3
#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨some false, some true, none⟩ == 3
#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨some false, some false, none⟩ == 3
#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨some true, some false, none⟩ == 4
#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨some true, none, none⟩ == 4

end Spot.Record
