import Firewall.Protocol
import Flock.Firewall.Released

/-!
# What the firewall's release releases (`verity/firewall-release/v0`)

`Core.Firewall.protocol` (`verity/experimental/core/service/firewall`) is one outer session's release decision as a core
protocol: the verifier's public input, the firewall's record before the session, and the session asked for, the
firewall's clock and the session's transcript. These are its properties, for every input, each a `Prop` that
`Properties.Proofs.Firewall` proves on Lean core:

* `AcceptsHolds`: an accepted session's transcript is one `Flock.Firewall.Contract` lets through (`Contract.Holds`), so
  `holds_commit`, `holds_masked`, `holds_fixed`, `holds_coins` and `holds_finish` say what left; `AnswerHolds` says the
  same of every byte string `Protocol.answer` accepts.
* `CheckAccepts`: an accepted session is the next statement of a recursive session the record opened, that never
  stopped and whose inner session finished, on a stream under its failure budget, released at the end of its window
  (`start_ms + windows_ms[0] + ... + windows_ms[j]`) or, unscheduled, at once, and never after that time.
* `AcceptsComputed`: under D8 (a), the named assumption `Flock.Firewall.Contract.FirewallComputes`, every value an
  accepted session lets out by the firewall's own computation is the specified one.

The record is the one the session's admission read: the record before the session, read, with the session's slot
appended (a dead holder's stop included), read again (`Record.op`). Not stated here: what the session appends to the record
(its refusal or stop, record I/O like the file it is kept in), the send after the decision (`outerSession`), and that
Rust's `flock_live::release` and Python's `verity_flock.firewall` decide as this does, which the agreement tests check.
-/

namespace Properties.Firewall

open Core Core.Firewall
open Flock.Firewall.Contract (PublicInput Transcript Holds)

/-- What `check` accepts. The session's transcript is one the contract lets through. The record before the session reads
(`r₀`), and so does it with the session's slot appended (`r₁`). In `r₁`, the recursive session asked for opened (`o`, its
first open) and never stopped, and its inner session finished. The statement asked for is its `j`-th, the first not
released: every one before it was. Its stream's failures are under its budget, if it has one. What is made public is the
session, its statement, its stream and its release time: under a schedule, `start_ms` plus the windows of statements 0 to
`j`, and the firewall's clock is not past it; without one, none. The profile is the transcript's phase and item count. -/
def CheckAccepts : Prop :=
  ∀ (pub : PublicInput) (r : Record) (e : Evidence) (p : Released) (c : Checked), check pub r e = .accept p c →
    Holds pub e.transcript ∧
    ∃ r₀ r₁ o j, Record.read r.text = .ok r₀ ∧
      Record.read (r₀.extend (slot r₀.events e.session e.part)).text = .ok r₁ ∧
      Event.opened o ∈ r₁.events ∧ o.session = e.session ∧ openOf r₁.events e.session = some o ∧
      (∀ part why, Event.stop e.session part why ∉ r₁.events) ∧
      Event.finish e.session ∈ r₁.events ∧
      (∃ hj : j < o.parts.size, o.parts[j] = e.part ∧ Event.release e.session e.part ∉ r₁.events ∧
        ∀ i (hi : i < j), Event.release e.session o.parts[i] ∈ r₁.events) ∧
      (∀ b, budgetOf r₁.events o.stream = some b → failures r₁.events o.stream < b) ∧
      p = { session := e.session, part := e.part, stream := o.stream,
            releaseMs := o.release.map fun r => r.start + (r.windows.toList.take (j + 1)).sum } ∧
      (∀ t, p.releaseMs = some t → e.nowMs ≤ t) ∧
      c = { phase := e.transcript.phase, items := e.transcript.items.size }

/-- The firewall's release lets out only a transcript `Flock.Firewall.Contract` lets through. -/
def AcceptsHolds : Prop :=
  ∀ (pub : PublicInput) (r : Record) (e : Evidence), protocol.Accepts pub r e → Holds pub e.transcript

/-- For every three byte strings `core-firewall` accepts: the public input, record and evidence they decode to, and the
evidence's transcript is one the contract lets through under that public input. -/
def AnswerHolds : Prop :=
  ∀ params commitment evidence : ByteArray, (protocol.answer params commitment evidence).verdict = .accept →
    ∃ pub r e, (canonicalJson params >>= PublicInput.ofJson) = .ok pub ∧ recordOf commitment = .ok r ∧
      evidenceOf evidence = .ok e ∧ protocol.Accepts pub r e ∧ Holds pub e.transcript

/-- Under D8 (a) (`FirewallComputes`), every value an accepted session lets out by the firewall's own computation (its
recheck, its opening, its own values) is `spec`'s for its name. -/
def AcceptsComputed : Prop :=
  ∀ (spec : String → Option ByteArray) (pub : PublicInput) (r : Record) (e : Evidence), protocol.Accepts pub r e →
    Flock.Firewall.Contract.FirewallComputes spec e.transcript →
    ∀ (what : String) (src : Flock.Firewall.Contract.By) (want sent : ByteArray) (ps : Array Nat),
      Flock.Firewall.Contract.Item.fixed what src want sent ps ∈ e.transcript.items.toList →
      (src = .recheck ∨ src = .shadow ∨ src = .own) → spec what = some sent

end Properties.Firewall
