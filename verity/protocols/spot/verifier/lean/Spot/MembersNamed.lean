/-
The per-member form's naming rule, as a statement: what a port of a member's registered file may name.

A member's registered file (`Own.registered`, `flock-verify --registered`'s object) is the verifier's own, from the
parameters, so every member's is fixed before any coin, a sampled member's and an unsampled one's alike. Each of its
ports names a tree by its `value`, and a session of the member opens its units' reads of the port in the tree of the
port's root and domain. The rule holds that name to the registration: the value is a port of the record's `roots` or a
value of its `registered`, and the port is the tree the record names there (`Named`), for every port of every member
(`Member.PortsNamed`).

It says nothing of a tree no member's file names: a root of the record that is no member's and that no port reads is
fixed by the registration's digest alone, no session opens it, and nothing here speaks of its leaves. It says nothing
of which leaf a unit reads either: a port's reads are the parameters', as the port is.
-/
module

public import Spot.OneStageZk

@[expose] public section

namespace Spot.OneStageZk

open Lean (Json)

/-- The registration's root at the port `port`, as the record check reads it (`Record.byPort`: a port's last). -/
def rootAt? (rec : Json) (port : String) : Option Json :=
  ((Record.byPort rec).find? (·.1 == port)).map (·.2)

/-- The registration's registered entry of the value `value`: its last, the one whose root the record check reads
(`Record.registeredRoots`). -/
def entryAt? (rec : Json) (value : String) : Option Json :=
  (match getD rec "registered" with | .arr es => es | _ => #[]).foldl (fun acc e =>
    match getD e "value" with
    | .str v => if v == value then some e else acc
    | _ => acc) none

/-- The port `q` is the tree at the registration's root `r`: the root's root, its count of leaves and its domain. A
root names no row width, so the width a port reads it at is the port's, bound by the opening. -/
def AtRoot (q : Flock.Registered.Port) (r : Json) : Prop :=
  getD r "root" = .str (Flock.hexOf q.root) ∧ nat? (getD r "leaves") = some q.leaves ∧
    getD r "domain" = .str (Flock.hexOf q.domain)

/-- The leaf width a registered entry names: `word_bits`, or the default by leaving it out. -/
def entryBits (e : Json) : Option Nat :=
  match field? e Flock.Registered.WIDTH_KEY with
  | none => some Flock.Registered.WORD_BITS
  | some j => nat? j

/-- The port `q` is the registration's registered entry `e`: the entry's root, its count of leaves, its row width
where the port names one, its leaf width and its domain. -/
def AtEntry (q : Flock.Registered.Port) (e : Json) : Prop :=
  getD e "root" = .str (Flock.hexOf q.root) ∧ nat? (getD e "leaves") = some q.leaves ∧
    (∀ w, q.words = some w → nat? (getD e "words") = some w) ∧ entryBits e = some q.wordBits ∧
    getD e "domain" = .str (Flock.hexOf q.domain)

/-- **The port `q` names a tree the registration `rec` names, and is that tree**: its `value` is a port of the
record's `roots` (the registration's own, the member's, another member's) or a value of its `registered`, and the port
is each tree the record names by it: the root's (`AtRoot`), the entry's (`AtEntry`). -/
def Named (rec : Json) (q : Flock.Registered.Port) : Prop :=
  ((rootAt? rec q.value).isSome ∨ (entryAt? rec q.value).isSome) ∧
    (∀ r, rootAt? rec q.value = some r → AtRoot q r) ∧ ∀ e, entryAt? rec q.value = some e → AtEntry q e

/-- **The naming rule at a member**: its registered file, where the parameters name one, reads as registered reads
and every port of it is named (`Named`). It reads the parameters and the record alone: no blob, no session and no
sample, so it is of a member the sample missed as of one it held. -/
def Member.PortsNamed (m : Member) (rec : Json) : Prop :=
  ∀ j, m.own.registered = some j → ∃ own, Flock.Registered.ofJson j = .ok own ∧ ∀ q ∈ own, Named rec q

end Spot.OneStageZk
