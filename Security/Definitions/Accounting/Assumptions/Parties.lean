import Mathlib

/-!
# An auditor party's assumptions: `Faithful`, `Secret`, `Confined`, and its witness path's `WitnessPathPrivate`

An auditor party (a network certifier, the memory challenger) is a program the auditor runs on its own device: code, a
state it keeps, coins of its own, and declared channels it sends messages on, each to the auditor in the clear (a public
message) or to the prover only (witness: what the prover's zero-knowledge proofs read). These are what a protocol that
reads a party's messages takes of one party `p`, each a named `Prop` about one run of it (`Run`), in place of
`trusted-host` (`Verity.Assumptions.TrustedHost`) and no-state-rollback. That the party runs on a machine the auditor
owns and the prover doesn't control is why we believe them, not an assumption.

* `Faithful` (`faithful`): `p` runs its code on its true inputs, each step's input being what was delivered to it, its
  own clock's true reading or its own coin source's draw, and each step starts from the state the last one left: its
  state persists and is never rolled back to an earlier copy, so an LMS leaf is never used twice.
* `Secret` (`secret`): the prover never reads or sets `p`'s secret state: its signing keys, its LMS state, and its coins
  not yet released.
* `Confined` (`confined`): all `p` emits, on any wire, is its declared channels' messages.
* `WitnessPathPrivate` (`witness-path-private`): the path between `p` and the prover carries no observer but its two
  endpoints, so nobody else sees a witness message. Witness messages travel unencrypted for now; an encrypted path
  would replace this assumption.

`Covered` (`Accounting.Covered`) stays the site premise. What the auditor and the prover see of a party under these
(`auditorView`, `proverView`) is stated over its code's messages in `Properties.Attest`.
-/

namespace Accounting.Assumptions.Parties

/-- Where a declared channel's messages go: to the auditor in the clear, or to the prover only, as witness. -/
inductive Dest
  | clear
  | witness
  deriving DecidableEq

/-- An auditor party's code: its state from its own coins, and one step: from its state and an input (a message
delivered to it, a reading of its clock, a draw of its coin source), its next state and the messages it sends, each on
one of its declared channels. -/
structure Code (Ω S I Ch M : Type*) where
  init : Ω → S
  step : S → I → S × List (Ch × M)

variable {Ω S I Ch M Wi A : Type*}

/-- The code on the inputs `xs` from the state `s`: the state each step starts from, then the last state, and every
message it sends, in order. -/
def Code.trace (c : Code Ω S I Ch M) : S → List I → List S × List (Ch × M)
  | s, [] => ([s], [])
  | s, x :: xs => let r := c.step s x; let t := c.trace r.1 xs; (s :: t.1, r.2 ++ t.2)

/-- The code's messages on the inputs `xs`, from the state its coins `ω` give it. -/
def Code.messages (c : Code Ω S I Ch M) (ω : Ω) (xs : List I) : List (Ch × M) := (c.trace (c.init ω) xs).2

/-- One run of an auditor party as it happened: its coins; at each step its true input (what was delivered to it, its
clock's true reading, its coin source's draw) and the input it took; the state each step started from, then its last;
the messages it sent on its declared channels; everything it emitted on any wire, each with where it went; the prover's
accesses (reads and writes) to its secret state; and the messages on its path to the prover that someone other than
the party and the prover observed. -/
structure Run (Ω S I Ch M Wi A : Type*) where
  coins : Ω
  truth : List I
  taken : List I
  states : List S
  sent : List (Ch × M)
  emitted : List (Dest × Wi)
  access : List A
  overheard : List (Ch × M)

/-- **`faithful`**: the party runs its code `c` on its true inputs, its clock's and its coin source's included, and
each step starts from the state the last one left (no rollback): what it took is what was true, and its states and
messages are the code's on those inputs from the state its coins give it. -/
def Faithful (c : Code Ω S I Ch M) (r : Run Ω S I Ch M Wi A) : Prop :=
  r.taken = r.truth ∧ (r.states, r.sent) = c.trace (c.init r.coins) r.taken

/-- **`secret`**: the prover never reads or sets the party's secret state (its signing keys, its LMS state, its coins
not yet released). -/
def Secret (r : Run Ω S I Ch M Wi A) : Prop := r.access = []

/-- **`confined`**: all the party emits is its declared channels' messages, each as `wire` puts it on the wire, to
where its channel goes (`dest`). -/
def Confined (dest : Ch → Dest) (wire : Ch × M → Wi) (r : Run Ω S I Ch M Wi A) : Prop :=
  r.emitted = r.sent.map fun m => (dest m.1, wire m)

/-- **`witness-path-private`**: the path between the party and the prover carries no observer but its endpoints:
nobody else observed a message on it. -/
def WitnessPathPrivate (r : Run Ω S I Ch M Wi A) : Prop := r.overheard = []

/-- What the auditor (and anyone but the prover) sees of the party: what it emitted in the clear, and what was overheard
on its path to the prover. -/
def auditorView (wire : Ch × M → Wi) (r : Run Ω S I Ch M Wi A) : List Wi :=
  (r.emitted.filter (·.1 = .clear)).map (·.2) ++ r.overheard.map wire

/-- What the prover sees of the party: all it emitted, and its accesses to the party's secret state. -/
def proverView (r : Run Ω S I Ch M Wi A) : List Wi × List A := (r.emitted.map (·.2), r.access)

end Accounting.Assumptions.Parties
