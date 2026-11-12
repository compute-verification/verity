/-
A read across streams: a later stream's registration whose `previous` names a record of an earlier stream, so that
the later stream's blocks read that record's ports as registered values. `across` gives three verdicts on it, each
computed whatever the others say:

* `logged`: the later registration's `previous` is the digest of a record of the earlier stream's log (`loggedAt`), the
  registrations that stream received, in order;
* `order`: in the one session the auditor's ledger opens on the window (`Challenger.trustedSessionOf`), the bytes the later
  stream's sample read are exactly the coins the challenger's signed log shows released after the later registration
  (`Challenger.coinsReason`), and that log shows a coin released after the earlier registration, the last of those before
  the first released after the later one (`receiptReason`): the record whose ports the later blocks read was received
  first, and no coin that sampled one of them was released before the challenger received what fixes its rows;
* `audit`: the earlier stream's audit record (`Record.record`'s) is the earlier registration's and accepted
  (`auditReason`); no verdict when none is handed.

`case` names the outcome: 1 when all three hold; 2 when the earlier stream's log holds no such record or the order is
not the log's; 3 when those hold and the earlier stream's audit record is missing or did not accept.

The order is read from one coin family of one session, the later sample's own: both registrations' receipts are roots
of that family.

`acrossAll` takes a list of such reads beside the digest of one record, a class's boundary record, and runs `across` on
each. A read reaches that record when its `previous` is it, or is the later registration of a read of the list that
reaches it (`reached`). Its `case` is the list's: 1 when every read reaches the boundary record and is case 1; 2 when one
does not reach it or is case 2; 3 when neither and a read's earlier stream's audit record is missing or did not accept.
It holds nothing about a record the list leaves out: that the list is every record that reads the class is the
caller's.
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

/-- Why `received`, a coin family's roots (the root the challenger had received when it released each coin, in order),
does not show `earlier` received before every coin released after `later`: no coin was released after one of them, or
the last coin after `earlier` is not before the first after `later`. -/
def receiptReason (received : List ByteArray) (earlier later : ByteArray) : Option String :=
  let coinsAfter (x : ByteArray) := (received.zipIdx.filter (·.1 == x)).map (·.2)
  match (coinsAfter earlier).getLast?, (coinsAfter later).head? with
  | none, _ => some "the challenger's log shows no coin released after the earlier registration: its receipt is not in the log"
  | some _, none => some "the challenger's log shows no coin released after the later registration: its receipt is not in the log"
  | some a, some b =>
    if a < b then none
    else some s!"the challenger's log shows coin {b} released after the later registration and coin {a} after the earlier registration: the earlier registration's coins come first"

/-- The roots of the coin family the evidence `coins` names (its segment's label), from the session the auditor's ledger
opens on `window` (`Challenger.trustedSessionOf`, as `Challenger.coinsReason` reads it). -/
def receivedOf (coins : Json) (window : String) : V (List ByteArray) := do
  let (_, roots) ← Challenger.trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)])
    (← Challenger.ChallengerKey.ofJson (getD coins "key")) (getD coins "pins") (getD coins "main") (getD coins "session")
  return match getD (getD coins "segment") "label" with
    | .str l => ((roots.find? (·.1 == l)).map (·.2)).getD []
    | _ => []

def NO_LOG : String := "the challenger's session published no signed log to read the order from"

/-- Why the later stream's sample does not hold the order its registration rests on: it has no coin evidence (`coins`
null), its coins are not the ones the challenger's signed log shows released after `later` (`Challenger.coinsReason` on
the bytes `read`), or that log does not show `earlier` received first (`receiptReason`). -/
def orderReason (window : String) (earlier later : ByteArray) (coins : Json) (read : ByteArray) : Option String :=
  match coins with
  | .null => some NO_LOG
  | _ =>
    match Challenger.coinsReason coins window later read with
    | some why => some s!"the later sample's coins: {why}"
    | none =>
      match receivedOf coins window with
      | .error why => some why
      | .ok received => receiptReason received earlier later

/-- Why `audit` is not an accepted audit record of the registration `previous` (its digest, hex): it is no audit record,
it is another registration's, or it did not accept. -/
def auditReason (audit : Json) (previous : String) : Option String :=
  if !jsonStrIs (getD audit "format") FORMAT then some s!"not a {FORMAT} record"
  else if !jsonStrIs (getD audit "registration_digest") previous then
    some "the audit record is another registration's than the one the later registration names"
  else match getD audit "accepted" with
    | .bool true => none
    | _ => some "the earlier stream's audit record did not accept"

/-- What `across` reads: the window's open root the verifier holds, the earlier stream's log, its audit record (null:
none), the later stream's registration, and the later stream's sample's coin evidence and the bytes it read. -/
structure Across where
  window : String
  log : Array Json
  audit : Json
  later : Json
  coins : Json
  read : ByteArray

/-- 1 when all three verdicts accept, 2 when `logged` or `order` does not, 3 when they do and `audit` does not. -/
def acrossCase (logged order audit : AcrossVerdict) : Nat :=
  if logged.accepted != some true || order.accepted != some true then 2
  else if audit.accepted != some true then 3
  else 1

/-- What `across` finds on one read: its three verdicts, the earlier record's index in the earlier stream's log (none
when it holds none) and the two registrations' digests in hex. -/
structure AcrossFound where
  logged : AcrossVerdict
  received : Option Nat
  order : AcrossVerdict
  audit : AcrossVerdict
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
           order := .ofReason (orderReason i.window earlier later i.coins i.read),
           audit := (match i.audit with
             | .null => { accepted := none }
             | a => .ofReason (auditReason a previous)),
           earlier := previous, later := Flock.hexOf later }

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

/-- A list's case from each read's own case and whether it reaches the boundary record: 2 when a read does not reach it
or is case 2, else 3 when a read is case 3, else 1. -/
def acrossAllCase (links : Array (Nat × Bool)) : Nat :=
  if links.any fun (c, reaches) => !reaches || (c != 1 && c != 3) then 2
  else if links.any fun (c, _) => c != 1 then 3
  else 1

/-- What `acrossAll` reads: the digest (hex) of a class's boundary record, and the reads across streams that rest on
it, each as `across` reads one. -/
structure AcrossAll where
  boundary : String
  links : Array Across

/-- The verdicts on a list of reads across streams (module docstring): `{"boundary", "links", "case"}`, `links` each
read's `across` in the list's order with `reaches`, whether its earlier record is reached from `boundary` through reads
of the list. Refused: a list with no read, a `boundary` that is not hex, or a read `across` refuses, named by its
index. -/
def acrossAll (i : AcrossAll) : V Json := do
  if i.links.isEmpty then throw "the list holds no read across streams"
  if let .error why := Flock.unhex i.boundary then throw s!"boundary: {why}"
  let mut found : Array AcrossFound := #[]
  for l in i.links do
    match acrossFound l with
    | .ok f => found := found.push f
    | .error why => throw s!"link {found.size}: {why}"
  let got := reached i.boundary (found.map fun f => (f.earlier, f.later))
  let links := found.map fun f => (f, got.contains f.earlier)
  return Json.mkObj [("boundary", Json.str i.boundary),
    ("links", Json.arr (links.map fun (f, reaches) => Json.mkObj (f.fields ++ [("reaches", Json.bool reaches)]))),
    ("case", Lean.toJson (acrossAllCase (links.map fun (f, reaches) => (f.caseOf, reaches))))]

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
-- a sample with no coin evidence holds no order
#guard orderReason "" (root 1) (root 2) .null (root 0) == some NO_LOG

private def audited (digest : String) (accepted : Json) : Json :=
  Json.mkObj [("format", Json.str FORMAT), ("registration_digest", Json.str digest), ("accepted", accepted)]

#guard auditReason (audited "ab" (.bool true)) "ab" == none
#guard (auditReason (audited "ab" (.bool false)) "ab").isSome
#guard (auditReason (audited "ab" .null) "ab").isSome
#guard (auditReason (audited "ab" (.str "true")) "ab").isSome
#guard (auditReason (audited "cd" (.bool true)) "ab").isSome
#guard (auditReason (Json.mkObj [("registration_digest", Json.str "ab"), ("accepted", .bool true)]) "ab").isSome

private def reg (n : Nat) : Json :=
  Json.mkObj [("format", Json.str "verity/registration/v2"), ("population", Lean.toJson n)]

private def digestOf (r : Json) : String := (Flock.WindowRecord.digestHex r).toOption.getD ""

#guard loggedAt #[reg 1, reg 2] (digestOf (reg 2)) == some 1
#guard loggedAt #[reg 1, reg 2] (digestOf (reg 3)) == none
#guard loggedAt #[] (digestOf (reg 1)) == none

#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨some true, none⟩ == 1
#guard acrossCase ⟨some false, none⟩ ⟨some true, none⟩ ⟨some true, none⟩ == 2
#guard acrossCase ⟨some true, none⟩ ⟨some false, none⟩ ⟨none, none⟩ == 2
#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨none, none⟩ == 3
#guard acrossCase ⟨some true, none⟩ ⟨some true, none⟩ ⟨some false, none⟩ == 3

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

#guard acrossAllCase #[(1, true), (1, true)] == 1
#guard acrossAllCase #[(1, true), (3, true)] == 3
#guard acrossAllCase #[(3, true), (2, true)] == 2
#guard acrossAllCase #[(1, true), (1, false)] == 2
#guard acrossAllCase #[(3, false)] == 2
#guard acrossAllCase #[(1, true), (0, true)] == 2

end Spot.Record
