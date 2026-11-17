module

public import Properties.Challenger
public import Properties.Spot.MonitoringPeriod

/-! SPOT's period: the ledger its check reads is the auditor's device's own or a break, its stronger forms under
`Numbered` and `Chained`. -/

@[expose] public section

namespace Properties.Spot.MonitoringPeriod

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger Properties.Challenger

/-- What a ledger line says, its signature aside: the canonical text of its entry without its `sig`
(`Core.Record.unsigned`), which is what the device signs (`ledgerMessage` hashes it under the entry's schema's tag).
`none` for a line that is no entry. -/
def unsignedText (line : Json) : Option String := do
  let e ← ledgerEntry? line
  (Flock.canonUtf8 (Core.Record.unsigned e.entry)).toOption

/-- **Each line of the ledger a period's check reads is one the device wrote, under the number it is read at**: its
line at place `i` is an entry numbered `i` that says, signature aside, what a line of the device's ledger under
the key says; or the fixed finder outputs a break. -/
def LedgerLinesOfDevice : Prop :=
  ∀ (dev : History) (p : _root_.Spot.MonitoringPeriod.Params) (ledger : Json) (r : _root_.Spot.MonitoringPeriod.Read),
    _root_.Spot.MonitoringPeriod.read p ledger = .ok r →
      (∃ lines : Array Json, ledger = .arr lines ∧ Reaches p ledger ∧
          ∀ (i : Nat) (line : Json), lines[i]? = some line →
            (∃ e, ledgerEntry? line = some e ∧ e.seq = i) ∧ (unsignedText line).isSome ∧
              ∃ own ∈ dev.ledger p.key.pub, unsignedText own = unsignedText line) ∨
        Break dev p.key p.label (Json.mkObj [("ledger", ledger)])

/-- **The device numbers its ledger's lines**: under the key `pk`, a line of its ledger that is an entry numbered `i`
is its line `i`; it asks nothing of a line that is no entry the reader takes. It is taken as a hypothesis, not
proved. -/
def Numbered (dev : History) (pk : ByteArray) : Prop :=
  ∀ line ∈ dev.ledger pk, ∀ e, ledgerEntry? line = some e → (dev.ledger pk)[e.seq]? = some line

/-- **The ledger a period's check reads is the device's own, whole up to the period's end**: for a device that
numbers its lines (`Numbered`), the presented lines are, signatures aside, the first lines of its ledger in
order, none dropped and none added; or the fixed finder outputs a break. -/
def LedgerOfDevice : Prop :=
  ∀ (dev : History) (p : _root_.Spot.MonitoringPeriod.Params) (ledger : Json) (r : _root_.Spot.MonitoringPeriod.Read),
    _root_.Spot.MonitoringPeriod.read p ledger = .ok r →
    Numbered dev p.key.pub →
      (∃ lines : Array Json, ledger = .arr lines ∧ Reaches p ledger ∧
          lines.size ≤ (dev.ledger p.key.pub).length ∧
          (∀ line ∈ lines.toList, (unsignedText line).isSome) ∧
          lines.toList.map unsignedText = ((dev.ledger p.key.pub).take lines.size).map unsignedText) ∨
        Break dev p.key p.label (Json.mkObj [("ledger", ledger)])

/-- **The device chains its ledger's lines**: under `pk`, where a line and its next are both entries, the next names
as its `prev` the hex SHA-512 of the earlier line's whole text, signature included. Like `Numbered` it is taken
as a hypothesis, not proved. -/
def Chained (dev : History) (pk : ByteArray) : Prop :=
  ∀ (i : Nat) (line next : Json) (e e' : LedgerEntry), (dev.ledger pk)[i]? = some line →
    (dev.ledger pk)[i + 1]? = some next → ledgerEntry? line = some e → ledgerEntry? next = some e' →
      e'.prev = Flock.hexOf (Flock.Sha512.hash e.text.toUTF8)

/-- The whole texts of the lines of `lines` that are entries, signatures included: what the reader hashes to chain a
line to the line after it (`verifyLedger`'s `prev`). -/
def lineTexts (lines : List Json) : List ByteArray :=
  lines.filterMap fun line => (ledgerEntry? line).map (·.text.toUTF8)

/-- **The finder at the lines' whole texts**: `Properties.Challenger.findBreak` on the ledger alone, else the first
collision among the whole texts of the presented lines and the device's own under the key, so a break of
`findBreak` is a break of this one. -/
def findLineBreak (dev : History) (key : ChallengerKey) (label : String) (lines : Array Json) : Option Finding :=
  match findBreak dev key label (Json.mkObj [("ledger", Json.arr lines)]) with
  | some w => some w
  | none => (collisionIn (lineTexts lines.toList ++ lineTexts (dev.ledger key.pub))).map fun (x, y) => .collision x y

/-- **The break at the lines' whole texts**: `findLineBreak`, run on the presented lines and the device, outputs a
valid witness. -/
def LineBreak (dev : History) (key : ChallengerKey) (label : String) (lines : Array Json) : Prop :=
  ∃ w, findLineBreak dev key label lines = some w ∧ w.Valid dev key

/-- **Every line of the ledger a period's check reads but the last is the device's own line, signature included**,
under `Numbered` and `Chained`, or `LineBreak` holds. The last line is not held: no line after it names its
digest, so its signature and the LMS leaf read from it stay free. -/
def LedgerOfDeviceSigned : Prop :=
  ∀ (dev : History) (p : _root_.Spot.MonitoringPeriod.Params) (ledger : Json) (r : _root_.Spot.MonitoringPeriod.Read),
    _root_.Spot.MonitoringPeriod.read p ledger = .ok r →
    Numbered dev p.key.pub → Chained dev p.key.pub →
      ∃ lines : Array Json, ledger = .arr lines ∧
        (lines.toList.take (lines.size - 1) = (dev.ledger p.key.pub).take (lines.size - 1) ∨
          LineBreak dev p.key p.label lines)

end Properties.Spot.MonitoringPeriod
