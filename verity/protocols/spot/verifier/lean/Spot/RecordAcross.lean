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

`acrossAll` takes a list of such reads beside the digest of one record, a class's boundary record, and the last
stream's own audit record, and runs `across` on each read. A read reaches the boundary record when its `previous` is it,
or is the later registration of a read of the list that reaches it (`reached`). Each read hands its earlier stream's
audit record, so the reads hand one for every record of the list but its ends, the later records no read of it reads
(`ends`). A chain has one end, its last record, and `last` is that stream's audit record, read as every other is
(`auditVerdict`, on the end's digest); a list with no end or with several has no one last record, and `last` is then
no audit of it (`lastVerdict`). Its `case` is the list's: 1 when every read reaches the boundary record and is case 1
and the last stream's audit record accepted and is complete; 2 when a read does not reach the boundary record or is
case 2; 3 when none is and an audit record, a read's earlier stream's or the last stream's, is missing or not an
accept; 4 when none is either and one of those audit records is an accept that is not complete.
It holds nothing about a record the list leaves out: that the list is every record that reads the class is the
caller's. `last` is the caller's as each read's audit record is, and is held to no more.
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

/-- Whose audit record a read hands, and which registration it is to be of, in a reason's words. -/
def EARLIER : String := "the earlier stream's"

def NAMED : String := "the one the later registration names"

/-- Whose audit record a list is handed for its last record (`lastVerdict`), and which registration it is to be of. -/
def LAST : String := "the last stream's"

def LAST_NAMED : String := "the list's last record"

/-- Why `audit` is no audit record of the registration `previous` (its digest, hex): it is no audit record, or it is
another registration's than that one, `named` in the reason's words. -/
def auditOther (audit : Json) (previous : String) (named : String := NAMED) : Option String :=
  if !jsonStrIs (getD audit "format") FORMAT then some s!"not a {FORMAT} record"
  else if !jsonStrIs (getD audit "registration_digest") previous then
    some s!"the audit record is another registration's than {named}"
  else none

/-- That the audit record of `whose` stream did not accept. -/
def notAccepted (whose : String) : String := s!"{whose} audit record did not accept"

/-- That the audit record of `whose` stream accepted and is not complete. -/
def notComplete (whose : String) : String :=
  s!"{whose} audit record accepted with entries that did not run: it is not complete, and is no audit of those entries"

def NOT_ACCEPTED : String := notAccepted EARLIER

def NOT_COMPLETE : String := notComplete EARLIER

private def saysTrue : Json → Bool
  | .bool true => true
  | _ => false

/-- The verdict on `audit`, the audit record handed for the registration `previous` (its digest, hex): none when none
is handed (null); refused, with no word on `complete`, when it is no audit record of that registration (`auditOther`);
else `accepted` and `complete` as the record's own fields say, each true only where the field is `true`. `audit` is
the caller's record and is held to nothing: no entry of it is read, and nothing holds it to the record the earlier
stream's verifier wrote, so the verdict says no more than the record handed says. `whose` and `named` are the reasons'
words for the stream the record is of and the registration it is to name: a read's earlier stream's, unless given. -/
def auditVerdict (audit : Json) (previous : String) (whose : String := EARLIER) (named : String := NAMED) :
    AuditVerdict :=
  match audit with
  | .null => { accepted := none, complete := none }
  | a =>
    match auditOther a previous named with
    | some why => { accepted := some false, complete := none, why := some why }
    | none =>
      let accepted := saysTrue (getD a "accepted")
      let complete := saysTrue (getD a "complete")
      { accepted := some accepted, complete := some complete,
        why := if !accepted then some (notAccepted whose) else if !complete then some (notComplete whose) else none }

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

/-- `got` with the later record of every read of `links` (each its earlier and later records' digests) whose earlier
record is in it, until a pass adds none or `fuel` passes ran. -/
def reachedFrom (links : Array (String × String)) : Nat → Std.HashMap String Unit → Std.HashMap String Unit
  | 0, got => got
  | fuel + 1, got =>
    let more := links.foldl (init := got) fun got (earlier, later) =>
      if got.contains earlier then got.insert later () else got
    if more.size == got.size then got else reachedFrom links fuel more

/-- The records `links` reach from `boundary`: `boundary`, and the later record of every read whose earlier record is
reached. A pass that adds a record adds one of at most `links.size`, so `links.size + 1` passes end it. -/
def reached (boundary : String) (links : Array (String × String)) : Std.HashMap String Unit :=
  reachedFrom links (links.size + 1) ((∅ : Std.HashMap String Unit).insert boundary ())

/-- The ends of `links` (each read's earlier and later records' digests): the later records no read of the list reads,
each once, in the list's order. A chain has one, its last record; a ring has none. -/
def ends (links : Array (String × String)) : Array String :=
  let read := links.foldl (init := (∅ : Std.HashMap String Unit)) fun got (earlier, _) => got.insert earlier ()
  (links.foldl (init := ((#[] : Array String), read)) fun (out, seen) (_, later) =>
    if seen.contains later then (out, seen) else (out.push later, seen.insert later ())).1

/-- The verdict on `last`, the audit record handed for the last stream of a list whose ends are `ends`: `auditVerdict`'s
on the one end's digest, its reasons in the last stream's words. Refused, with no word on `complete`, where the list
has no end or several: no one record is its last, and one audit record is no audit of each. -/
def lastVerdict (last : Json) (ends : Array String) : AuditVerdict :=
  match ends with
  | #[e] => auditVerdict last e LAST LAST_NAMED
  | _ => { accepted := some false, complete := none,
           why := some s!"the list ends in {ends.size} records, not one: no one record is its last" }

/-- A list's case from each read's own case and whether it reaches the boundary record, and from the verdict on the
last stream's audit record: 2 when a read does not reach the boundary record or is case 2; else 3 when a read is case 3
or `last` is no accept; else 4 when a read is case 4 or `last` is not complete; else 1. -/
def acrossAllCase (links : Array (Nat × Bool)) (last : AuditVerdict) : Nat :=
  if links.any fun (c, reaches) => !reaches || !(c == 1 || c == 3 || c == 4) then 2
  else if last.accepted != some true || links.any (fun (c, _) => c == 3) then 3
  else if last.complete != some true || links.any (fun (c, _) => c == 4) then 4
  else 1

/-- What `acrossAll` reads: the digest (hex) of a class's boundary record, the reads across streams that rest on
it, each as `across` reads one, and the last stream's own audit record (null: none). -/
structure AcrossAll where
  boundary : String
  links : Array Across
  last : Json

/-- The verdicts on a list of reads across streams (module docstring): `{"boundary", "links", "last", "audit",
"case"}`, `links` each read's `across` in the list's order with `reaches`, whether its earlier record is reached from
`boundary` through reads of the list, `last` the digest in hex of the list's last record (null where it has no end or
several), and `audit` the verdict on the last stream's audit record (`lastVerdict`). Refused: a list with no read, a
`boundary` that is not hex, or a read `across` refuses, named by its index. -/
def acrossAll (i : AcrossAll) : V Json := do
  if i.links.isEmpty then throw "the list holds no read across streams"
  if let .error why := Flock.unhex i.boundary then throw s!"boundary: {why}"
  let mut found : Array AcrossFound := #[]
  for l in i.links do
    match acrossFound l with
    | .ok f => found := found.push f
    | .error why => throw s!"link {found.size}: {why}"
  let pairs := found.map fun f => (f.earlier, f.later)
  let got := reached i.boundary pairs
  let links := found.map fun f => (f, got.contains f.earlier)
  let closing := ends pairs
  let last := lastVerdict i.last closing
  return Json.mkObj [("boundary", Json.str i.boundary),
    ("links", Json.arr (links.map fun (f, reaches) => Json.mkObj (f.fields ++ [("reaches", Json.bool reaches)]))),
    ("last", match closing with | #[e] => Json.str e | _ => Json.null),
    ("audit", last.toJson),
    ("case", Lean.toJson (acrossAllCase (links.map fun (f, reaches) => (f.caseOf, reaches)) last))]

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
-- a read's reasons, word for word
#guard NOT_ACCEPTED == "the earlier stream's audit record did not accept"
#guard NOT_COMPLETE
  == "the earlier stream's audit record accepted with entries that did not run: it is not complete, and is no audit of those entries"
#guard (auditVerdict (audited "cd" (.bool true)) "ab").why
  == some "the audit record is another registration's than the one the later registration names"

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

private def reaches (boundary : String) (links : Array (String × String)) : Array Bool :=
  links.map fun l => (reached boundary links).contains l.1

-- a chain of three records, in order and out of order
#guard reaches "a" #[("a", "b"), ("b", "c")] == #[true, true]
#guard reaches "a" #[("b", "c"), ("a", "b")] == #[true, true]
#guard reaches "a" #[("c", "d"), ("b", "c"), ("a", "b")] == #[true, true, true]
-- two records on the boundary record
#guard reaches "a" #[("a", "b"), ("a", "c")] == #[true, true]
-- a read whose earlier record no read of the list gives, and one behind it
#guard reaches "a" #[("a", "b"), ("x", "c"), ("c", "d")] == #[true, false, false]
-- a read of the boundary record's own previous does not reach it
#guard reaches "a" #[("z", "a"), ("a", "b")] == #[false, true]
-- a ring that the boundary record is not on
#guard reaches "a" #[("b", "c"), ("c", "b")] == #[false, false]
#guard reaches "a" #[] == #[]

-- a chain's one end, in order and out of order
#guard ends #[("a", "b"), ("b", "c")] == #["c"]
#guard ends #[("b", "c"), ("a", "b")] == #["c"]
#guard ends #[("a", "b")] == #["b"]
-- two records on the boundary record: two ends, in the list's order
#guard ends #[("a", "c"), ("a", "b")] == #["c", "b"]
-- a read given twice ends the list once
#guard ends #[("a", "b"), ("a", "b")] == #["b"]
-- a ring has no end, and neither has a list with no read
#guard ends #[("b", "c"), ("c", "b")] == #[]
#guard ends #[] == #[]

private def lastOn (last : Json) (ends : Array String) : Option Bool × Option Bool :=
  let v := lastVerdict last ends
  (v.accepted, v.complete)

-- one end: the verdict is the audit record's own on that record
#guard lastOn (audited "ab" (.bool true)) #["ab"] == (some true, some true)
#guard lastOn (audited "ab" (.bool true) (.bool false)) #["ab"] == (some true, some false)
#guard (lastVerdict (audited "ab" (.bool true) (.bool false)) #["ab"]).why == some (notComplete LAST)
#guard lastOn (audited "ab" (.bool false)) #["ab"] == (some false, some true)
#guard (lastVerdict (audited "ab" (.bool false)) #["ab"]).why == some (notAccepted LAST)
#guard lastOn (audited "cd" (.bool true)) #["ab"] == (some false, none)
#guard (lastVerdict (audited "cd" (.bool true)) #["ab"]).why
  == some "the audit record is another registration's than the list's last record"
#guard lastOn .null #["ab"] == (none, none)
-- no end, or several: the record handed is no audit of the list's last record
#guard lastOn (audited "ab" (.bool true)) #[] == (some false, none)
#guard lastOn (audited "ab" (.bool true)) #["ab", "cd"] == (some false, none)
#guard lastOn .null #["ab", "cd"] == (some false, none)
#guard (lastVerdict (audited "ab" (.bool true)) #["ab", "cd"]).why.isSome

private def whole : AuditVerdict := ⟨some true, some true, none⟩

-- the last stream's audit record accepted and complete: the list's case is its reads'
#guard acrossAllCase #[(1, true), (1, true)] whole == 1
#guard acrossAllCase #[(1, true), (3, true)] whole == 3
#guard acrossAllCase #[(1, true), (4, true)] whole == 4
#guard acrossAllCase #[(4, true), (3, true)] whole == 3
#guard acrossAllCase #[(4, true), (2, true)] whole == 2
#guard acrossAllCase #[(3, true), (2, true)] whole == 2
#guard acrossAllCase #[(1, true), (1, false)] whole == 2
#guard acrossAllCase #[(3, false)] whole == 2
#guard acrossAllCase #[(4, false)] whole == 2
#guard acrossAllCase #[(1, true), (0, true)] whole == 2
-- every read case 1: the list's case is the last stream's audit record's
#guard acrossAllCase #[(1, true), (1, true)] ⟨some true, some false, none⟩ == 4
#guard acrossAllCase #[(1, true), (1, true)] ⟨some true, none, none⟩ == 4
#guard acrossAllCase #[(1, true), (1, true)] ⟨some false, some true, none⟩ == 3
#guard acrossAllCase #[(1, true), (1, true)] ⟨some false, none, none⟩ == 3
#guard acrossAllCase #[(1, true), (1, true)] ⟨none, none, none⟩ == 3
-- the worse of the two: a missing audit record before one that is not complete, and a read that fails before both
#guard acrossAllCase #[(1, true), (4, true)] ⟨none, none, none⟩ == 3
#guard acrossAllCase #[(1, true), (3, true)] ⟨some true, some false, none⟩ == 3
#guard acrossAllCase #[(1, true), (2, true)] ⟨none, none, none⟩ == 2
#guard acrossAllCase #[(1, false)] ⟨some true, some false, none⟩ == 2

end Spot.Record
