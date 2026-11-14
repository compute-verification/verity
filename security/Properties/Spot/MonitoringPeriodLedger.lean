module

public import Properties.Challenger
public import Properties.Spot.MonitoringPeriod

/-!
# SPOT's period: the ledger its check reads is the auditor's device's own (`Spot.MonitoringPeriod.read`)

`Properties.Spot.MonitoringPeriod`'s statements are about the ledger the check was presented. These three say whose it is,
against the device model of `Properties.Challenger` (`History`: the device's runs and its ledger's lines under a
key; `Break`: the fixed finder's forgery or SHA-512 collision).

* `LedgerLinesOfDevice` takes nothing of the device. Every line of a ledger the check reads is, its signature aside,
  a line the device wrote, read at the number the device gave it; or a break.
* `LedgerOfDevice` takes that the device numbers its ledger's lines (`Numbered`). Then the ledger the check reads is
  the device's own from its first line, none dropped and none added, up to a line stamped at or after the monitoring
  period's end; or a break.
* `LedgerOfDeviceSigned` takes also that the device chains its ledger's lines (`Chained`). Then every presented line
  but the last is the device's own line whole, its signature included; or a break of the finder that also hashes the
  lines' whole texts (`LineBreak`).

**Why the second takes `Numbered`.** The first form is every presented line's own: a device that wrote two lines
under one number satisfies it while the presented ledger holds only one of them. That the device writes one ledger
under a key, each line numbered by the lines before it, is the device's behaviour on its trusted host, which
`History` does not hold; it is the one fact of the device the second form takes, a hypothesis of that statement
alone. `Numbered` asks it only of the lines that are entries the reader takes, each at the place its number names:
a line of the device's that the reader refuses, wherever it is, leaves the premise as it is.

**Why the third takes `Chained`, and where it stops.** The reader chains the presented lines: each names as its
`prev` the SHA-512 of the whole text of the line before it, signature included, and a line's `prev` is in what the
device signs. So the device's own line that says what a presented line says names the digest of the presented line
before it. That this is the digest of the device's own line before is again the device's behaviour, which `History`
does not hold: the second fact of the device taken, a hypothesis of the third statement alone. No line names the
digest of the last presented line, so its signature stays free.

**What they leave open.** The first two hold no line's signature to the device's: the device's own line and the
presented one say the same and may be signed apart. The third holds every line's but the last's, and so leaves free
the LMS leaf the reader reads from the last line's signature. The order is the ledger's, and a line's stamp is the
device's clock, of which nothing is taken.

**The history is any.** `dev` may hold every run and every ledger line the device made under the key, those of
earlier audits among them: at such a history an earlier line the check reads is the device's own, and no forgery.
The statements say something of a device only at a history that holds its whole ledger under the key, from its
first line: the check reads the ledger from its first line, and a line the history lacks is a forgery to the
finder.
-/

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

/-- **Each line of the ledger a period's check reads is one the device wrote, under the number it is read at.** For
every challenger device and every ledger the ledger's half of the period's check reads under the verifier's key
(`Spot.MonitoringPeriod.read`): the ledger is a list of lines that reaches the monitoring period's end, and the line at each
place `i` is an entry numbered `i` that says, its signature aside, what a line of the device's own ledger under that
key says; or the fixed finder outputs a break from the presented ledger and the device (`Break`, on the ledger
alone). -/
def LedgerLinesOfDevice : Prop :=
  ∀ (dev : History) (p : _root_.Spot.MonitoringPeriod.Params) (ledger : Json) (r : _root_.Spot.MonitoringPeriod.Read),
    _root_.Spot.MonitoringPeriod.read p ledger = .ok r →
      (∃ lines : Array Json, ledger = .arr lines ∧ Reaches p ledger ∧
          ∀ (i : Nat) (line : Json), lines[i]? = some line →
            (∃ e, ledgerEntry? line = some e ∧ e.seq = i) ∧ (unsignedText line).isSome ∧
              ∃ own ∈ dev.ledger p.key.pub, unsignedText own = unsignedText line) ∨
        Break dev p.key p.label (Json.mkObj [("ledger", ledger)])

/-- **The device numbers its ledger's lines**: under the key `pk`, a line of its ledger that is an entry numbered
`i` is its line `i`. So no two different lines of it share a number (module docstring). It asks nothing of a line
that is no entry the reader takes, a later form's among them. It is taken as a hypothesis, not proved: it is a
lemma of the modelled device once the model builds that device's whole ledger line by line, and no statement takes
it then. -/
def Numbered (dev : History) (pk : ByteArray) : Prop :=
  ∀ line ∈ dev.ledger pk, ∀ e, ledgerEntry? line = some e → (dev.ledger pk)[e.seq]? = some line

/-- **The ledger a period's check reads is the device's own, whole up to the period's end.** For every challenger
device that numbers its ledger's lines under the verifier's key (`Numbered`), and every ledger the ledger's half of
the period's check reads under that key (`Spot.MonitoringPeriod.read`): the presented lines are, their signatures aside, the
first lines of the device's own ledger, as many as were presented, in order, none dropped and none added, and one of
them is stamped at or after the monitoring period's end; or the fixed finder outputs a break from the presented
ledger and the device (`Break`, on the ledger alone). So of the lines the device signed up to its first line stamped
at or after the period's end, the check read every one. -/
def LedgerOfDevice : Prop :=
  ∀ (dev : History) (p : _root_.Spot.MonitoringPeriod.Params) (ledger : Json) (r : _root_.Spot.MonitoringPeriod.Read),
    _root_.Spot.MonitoringPeriod.read p ledger = .ok r →
    Numbered dev p.key.pub →
      (∃ lines : Array Json, ledger = .arr lines ∧ Reaches p ledger ∧
          lines.size ≤ (dev.ledger p.key.pub).length ∧
          (∀ line ∈ lines.toList, (unsignedText line).isSome) ∧
          lines.toList.map unsignedText = ((dev.ledger p.key.pub).take lines.size).map unsignedText) ∨
        Break dev p.key p.label (Json.mkObj [("ledger", ledger)])

/-- **The device chains its ledger's lines**: under the key `pk`, where a line of its ledger and its next line are
both entries, the next names as its `prev` the SHA-512 of the earlier line's whole text, its signature included, in
hex. It asks nothing of a line that is no entry the reader takes. Like `Numbered` it is taken as a hypothesis, not
proved: it is a lemma of the modelled device once the model builds that device's whole ledger line by line, and no
statement takes it then. -/
def Chained (dev : History) (pk : ByteArray) : Prop :=
  ∀ (i : Nat) (line next : Json) (e e' : LedgerEntry), (dev.ledger pk)[i]? = some line →
    (dev.ledger pk)[i + 1]? = some next → ledgerEntry? line = some e → ledgerEntry? next = some e' →
      e'.prev = Flock.hexOf (Flock.Sha512.hash e.text.toUTF8)

/-- The whole texts of the lines of `lines` that are entries, signatures included: what the reader hashes to chain a
line to the line after it (`verifyLedger`'s `prev`). -/
def lineTexts (lines : List Json) : List ByteArray :=
  lines.filterMap fun line => (ledgerEntry? line).map (·.text.toUTF8)

/-- **The finder at the lines' whole texts**, one fixed computable function of the presented lines and the device
for every prover: what `Properties.Challenger.findBreak` outputs from the ledger alone, else the first collision
among the whole texts of the presented lines and of the device's own under the key. `findBreak` hashes each line
without its signature and is left as it is; this finder calls it and then hashes what it does not. A break of
`findBreak` on the ledger is a break of this one. -/
def findLineBreak (dev : History) (key : ChallengerKey) (label : String) (lines : Array Json) : Option Finding :=
  match findBreak dev key label (Json.mkObj [("ledger", Json.arr lines)]) with
  | some w => some w
  | none => (collisionIn (lineTexts lines.toList ++ lineTexts (dev.ledger key.pub))).map fun (x, y) => .collision x y

/-- **The break at the lines' whole texts**: `findLineBreak`, run on the presented lines and the device, outputs a
valid witness. -/
def LineBreak (dev : History) (key : ChallengerKey) (label : String) (lines : Array Json) : Prop :=
  ∃ w, findLineBreak dev key label lines = some w ∧ w.Valid dev key

/-- **Every line of the ledger a period's check reads but the last is the device's own line, signature included.**
For every challenger device that numbers and chains its ledger's lines under the verifier's key (`Numbered`,
`Chained`), and every ledger the ledger's half of the period's check reads under that key (`Spot.MonitoringPeriod.read`): the
presented lines but the last are the first lines of the device's own ledger, each whole, in order; or the finder at
the lines' whole texts outputs a break (`LineBreak`). So the LMS leaf the reader reads from each of those lines'
signatures (`leafOf`) is the one the device's own line carries. The last presented line is not held: no line after
it names its digest, so its signature, and the leaf read from it, stay free. -/
def LedgerOfDeviceSigned : Prop :=
  ∀ (dev : History) (p : _root_.Spot.MonitoringPeriod.Params) (ledger : Json) (r : _root_.Spot.MonitoringPeriod.Read),
    _root_.Spot.MonitoringPeriod.read p ledger = .ok r →
    Numbered dev p.key.pub → Chained dev p.key.pub →
      ∃ lines : Array Json, ledger = .arr lines ∧
        (lines.toList.take (lines.size - 1) = (dev.ledger p.key.pub).take (lines.size - 1) ∨
          LineBreak dev p.key p.label lines)

end Properties.Spot.MonitoringPeriod
