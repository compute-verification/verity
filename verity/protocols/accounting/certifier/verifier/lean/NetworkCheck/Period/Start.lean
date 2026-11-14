module

public import Flock.RegKnown
public import CoreProtocol.Protocol
meta import Flock.RegKnown
meta import CoreProtocol.Protocol

/-!
# The scan's start

What the period's check (`NetworkCheck.Period`) holds an execution phase's scan's start to: the registered value the
start unit reads (`startName`), its opening to the empty state (`startEmpty`), and the start finding (`startCheck`).
-/

@[expose] public section

namespace NetworkCheck.Period

open Lean (Json toJson)

/-- The value an execution phase's scan starts from, in its registration: the start unit's `sin`, the first parameter of
the scan's Program, as one-stage names a registered value (`<template's index>.<parameter>`, the scan's driver's
`CARRY`). -/
def startName : String := "0.sin"

/-- A registration's start entries: those of its `registered` whose value is `startName`. -/
def startEntries (reg : Json) : Array Json :=
  let named := fun (e : Json) => match e.getObjValD "value" with | .str x => x == startName | _ => false
  match reg.getObjValD "registered" with | .arr xs => xs.filter named | _ => #[]

/-- The scan's registration starts from the empty state: it has a start entry (`startEntries`), and its `window`'s
`"empty": {"0.sin": {"words": W, "salt": hex}}` opens every one to `W` zero u16 words (`W` at least 1) under a salt of
`HmRow.SALT_BYTES` bytes: the hm96 leaf of that row (`Registered.knownPayload`, as the verifier registers a value at
known content), framed at 0 under the entry's domain, is the entry's root, the root of a tree of that one leaf. The
start unit reads its `sin` there and copies it to row 0 of the scan's state, the state before bucket 0, which the
Programs anchored at bucket 0 read; a row's digest binds its length, so a reader of another width doesn't open it. The
row is public, the empty state, and so is its salt. -/
def startEmpty (reg : Json) : Bool :=
  let o := ((reg.getObjValD "window").getObjValD "empty").getObjValD startName
  let hex := fun (k : String) (j : Json) => match j.getObjValD k with
    | .str x => (Flock.unhex x).toOption
    | _ => none
  let es := startEntries reg
  !es.isEmpty && es.all fun e => match (o.getObjValD "words").getNat?, hex "salt" o, hex "root" e, hex "domain" e with
    | .ok w, some salt, some root, some dom =>
      decide (0 < w) && decide (salt.size = Flock.HmRow.SALT_BYTES) &&
        decide (Flock.FV3.Sha512.leafAs Flock.HmRow.ROW_SCHEMA_V2 dom 0
          (Flock.Registered.knownPayload (Array.replicate w 0) salt) = root)
    | _, _, _, _ => false

/-- A registration whose start entries are at `roots`, each under the domain of 64 bytes `0d`, opened as `words` zero
words under the salt of 192 bytes `07`. -/
def startProbe (roots : List String) (words : Nat) : Json :=
  let entry := fun (root : String) => Json.mkObj
    [("domain", .str (String.join (List.replicate 64 "0d"))), ("root", .str root), ("value", .str startName)]
  Json.mkObj [("registered", .arr (roots.map entry).toArray), ("window", Json.mkObj [("empty", Json.mkObj
    [(startName, Json.mkObj [("salt", .str (String.join (List.replicate 192 "07"))), ("words", toJson words)])])])]

-- the roots of one zero word and of two under that salt and domain (the mirror's `start_empty`), and of the word 1
#guard
  let one := "cc8a8f235c85967255dadcfea4db8f2534aa0a12d0c3e9c289462c14072a92cc\
    3c65bfd17a3df01f6dcb91eec5ff7a6cb18294dc0ad07ab56a1e2fcd341161e0"
  let two := "ea02849e4fcf22320c9818f4da9dcd116bd876c29c37e98672c7d5c65bd98c5c\
    efd29d7568977d2d3396a404ca2e6c7582a7280a71c9acdfd0dd30b8822fd27b"
  let chosen := "ef7087146312d7acc2a957d16a9375942589fb284a8b7d7f0b6f8b4544ef9a90\
    3312d70b3b50a232f4ace84ee521f0601cd99008a52c1b98315ba10d15c81a51"
  startEmpty (startProbe [one] 1) && startEmpty (startProbe [two] 2) && startEmpty (startProbe [one, one] 1) &&
    !startEmpty (startProbe [one] 2) && !startEmpty (startProbe [two] 1) && !startEmpty (startProbe [chosen] 1) &&
    !startEmpty (startProbe [] 1) && !startEmpty (startProbe [one, two] 1) && !startEmpty (startProbe [one] 0) &&
    !startEmpty Json.null

/-- What the check says of an execution phase after its period's first. Its scan starts from the state the execution
phase before it ended in, which the check neither holds nor charges. -/
def startCarriedRefusal : String :=
  "the scan's start is the state the link-window before it ended in, and the check holds or charges a start only at \
    its period's first link-window (startCheck): a period is one link-window"

/-- Execution phase `w`'s start finding in the rows form, the `i`-th of its period: a refusal unless it is the period's
first (`i = 0`) and, where it is the run's first (`w = 0`), its scan starts from the empty state (`startEmpty`). The
start unit copies the registered value `startName` to row 0 of the scan's state, which the Programs anchored at
bucket 0 read. A period's first execution phase that is not the run's first starts from the state the one before it
ended in, which the check holds to nothing: it charges that start once, as free input at the state's full width (the
period's `start` term, `Scan.start`), so an adversary-chosen start carries no more than the bits it counts. A later
execution phase of a period starts from the state the one before it ended in, which the check neither holds nor
charges, so it refuses it (`startCarriedRefusal`). -/
def startCheck (w i : Nat) (reg : Json) : Option (Core.Verdict × String) :=
  if i != 0 then some (.reject, s!"window {w}: {startCarriedRefusal}")
  else if w != 0 || startEmpty reg then none
  else some (.reject, s!"window {w}: the scan's start is not the empty state: its registration has no {startName} \
    entry among its registered values, or its window.empty of {startName} is no opening of one to zero u16 words \
    under its salt at that entry's root and domain (startEmpty), so the Programs anchored at bucket 0 can read a \
    state the prover chose: register the start unit's sin as the empty state's row and give its salt")

-- the run's first execution phase at the empty state and not at another; a period's first that is not the run's
-- first at any start, charged; a period's second refused, at any execution phase
#guard
  let one := "cc8a8f235c85967255dadcfea4db8f2534aa0a12d0c3e9c289462c14072a92cc\
    3c65bfd17a3df01f6dcb91eec5ff7a6cb18294dc0ad07ab56a1e2fcd341161e0"
  (startCheck 0 0 (startProbe [one] 1)).isNone && (startCheck 0 0 (startProbe [one] 2)).isSome &&
    (startCheck 3 0 (startProbe [one] 2)).isNone && (startCheck 3 0 Json.null).isNone &&
    (startCheck 1 1 (startProbe [one] 1)).map (·.2) == some s!"window 1: {startCarriedRefusal}" &&
    (startCheck 4 1 (startProbe [one] 1)).map (·.2) == some s!"window 4: {startCarriedRefusal}"

end NetworkCheck.Period
