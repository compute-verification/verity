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

/-- The verdicts on one read across streams (module docstring): `{"logged", "received", "order", "audit", "case",
"earlier", "later"}`, `received` the earlier record's index in the log (null when it holds none), `earlier` and
`later` the two registrations' digests in hex. Refused: a later registration that names no previous record in hex, or
whose digest is not defined. -/
def across (i : Across) : V Json := do
  let .str previous := getD i.later "previous" | throw "the later registration names no previous record"
  let earlier ← Flock.unhex previous
  let later ← Flock.WindowRecord.digest i.later
  let received := loggedAt i.log previous
  let logged : AcrossVerdict := .ofReason (if received.isSome then none else some UNLOGGED)
  let order : AcrossVerdict := .ofReason (orderReason i.window earlier later i.coins i.read)
  let audit : AcrossVerdict := match i.audit with
    | .null => { accepted := none }
    | a => .ofReason (auditReason a previous)
  return Json.mkObj [("logged", logged.toJson),
    ("received", match received with | some k => Lean.toJson k | none => Json.null),
    ("order", order.toJson), ("audit", audit.toJson), ("case", Lean.toJson (acrossCase logged order audit)),
    ("earlier", Json.str previous), ("later", Json.str (Flock.hexOf later))]

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

end Spot.Record
