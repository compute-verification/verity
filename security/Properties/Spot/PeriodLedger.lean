module

public import Properties.Challenger
public import Properties.Spot.Period

/-!
# Sampled proofs' period: the ledger its check reads is the auditor's device's own (`Spot.Period.read`)

`Properties.Spot.Period`'s statements are about the ledger the check was presented. These two say whose it is,
against the device model of `Properties.Challenger` (`History`: the device's runs and its ledger's lines under a
key; `Break`: the fixed finder's forgery or SHA-512 collision).

* `LedgerLinesOfDevice` takes nothing of the device. Every line of a ledger the check reads is, its signature aside,
  a line the device wrote, read at the number the device gave it; or a break.
* `LedgerOfDevice` takes that the device numbers its ledger's lines (`Numbered`). Then the ledger the check reads is
  the device's own from its first line, none dropped and none added, up to a line stamped at or after the monitoring
  period's end; or a break.

**Why the second takes `Numbered`.** The first form is every presented line's own: a device that wrote two lines
under one number satisfies it while the presented ledger holds only one of them. That the device writes one ledger
under a key, each line numbered by the lines before it, is the device's behaviour on its trusted host, which
`History` does not hold; it is the one fact of the device the second form takes, a hypothesis of that statement
alone.

**What neither says.** A line's signature is not held to the device's: the device's own line and the presented one
say the same and may be signed apart. The order is the ledger's, and a line's stamp is the device's clock, of which
nothing is taken. And the history `dev` is any: the statements say something of a device only at a history that
holds its whole ledger under the key, from its first line, since the check reads the ledger from its first line and
a line the history lacks is a forgery to the finder.
-/

@[expose] public section

namespace Properties.Spot.Period

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger Properties.Challenger

/-- What a ledger line says, its signature aside: the canonical text of its entry's fields other than `sig`, which
is what the device signs (`ledgerMessage` hashes it under the entry's schema's tag). `none` for a line that is no
entry. -/
def unsignedText (line : Json) : Option String := do
  let e ← ledgerEntry? line
  (Flock.canon (Json.mkObj ((entries e.entry).toList.filter (·.1 != "sig")))).toOption

/-- **Each line of the ledger a period's check reads is one the device wrote, under the number it is read at.** For
every challenger device and every ledger the ledger's half of the period's check reads under the verifier's key
(`Spot.Period.read`): the ledger is a list of lines that reaches the monitoring period's end, and the line at each
place `i` is an entry numbered `i` that says, its signature aside, what a line of the device's own ledger under that
key says; or the fixed finder outputs a break from the presented ledger and the device (`Break`, on the ledger
alone). -/
def LedgerLinesOfDevice : Prop :=
  ∀ (dev : History) (p : _root_.Spot.Period.Params) (ledger : Json) (r : _root_.Spot.Period.Read),
    _root_.Spot.Period.read p ledger = .ok r →
      (∃ lines : Array Json, ledger = .arr lines ∧ Reaches p ledger ∧
          ∀ (i : Nat) (line : Json), lines[i]? = some line →
            (∃ e, ledgerEntry? line = some e ∧ e.seq = i) ∧ (unsignedText line).isSome ∧
              ∃ own ∈ dev.ledger p.key.pub, unsignedText own = unsignedText line) ∨
        Break dev p.key p.label (Json.mkObj [("ledger", ledger)])

/-- **The device numbers its ledger's lines**: under the key `pk`, line `i` of its ledger is an entry whose number
is `i`. So it writes one ledger a key, and no two of its lines share a number (module docstring). -/
def Numbered (dev : History) (pk : ByteArray) : Prop :=
  ∀ (i : Nat) (line : Json), (dev.ledger pk)[i]? = some line → ∃ e, ledgerEntry? line = some e ∧ e.seq = i

/-- **The ledger a period's check reads is the device's own, whole up to the period's end.** For every challenger
device that numbers its ledger's lines under the verifier's key (`Numbered`), and every ledger the ledger's half of
the period's check reads under that key (`Spot.Period.read`): the presented lines are, their signatures aside, the
first lines of the device's own ledger, as many as were presented, in order, none dropped and none added, and one of
them is stamped at or after the monitoring period's end; or the fixed finder outputs a break from the presented
ledger and the device (`Break`, on the ledger alone). So of the lines the device signed up to its first line stamped
at or after the period's end, the check read every one. -/
def LedgerOfDevice : Prop :=
  ∀ (dev : History) (p : _root_.Spot.Period.Params) (ledger : Json) (r : _root_.Spot.Period.Read),
    _root_.Spot.Period.read p ledger = .ok r →
    Numbered dev p.key.pub →
      (∃ lines : Array Json, ledger = .arr lines ∧ Reaches p ledger ∧
          lines.size ≤ (dev.ledger p.key.pub).length ∧
          (∀ line ∈ lines.toList, (unsignedText line).isSome) ∧
          lines.toList.map unsignedText = ((dev.ledger p.key.pub).take lines.size).map unsignedText) ∨
        Break dev p.key p.label (Json.mkObj [("ledger", ledger)])

end Properties.Spot.Period
