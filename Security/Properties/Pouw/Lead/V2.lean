import CoreProtocol.Protocol
import Definitions.Pouw.Lead.Model

/-!
# The lead's acceptance at PoUW's `verity/pouw/v2` case (draft, over a stub of the case's readers)

Crediting the lead's guarantee (`Pouw.Guarantees.Lead.G`) to a window needs each call the verifier accepts to be a
call `G` reads as accepted (`Pouw.Lead.Accepted`) at the coins it was served under. `PouwCheck` has no
`verity/pouw/v2` case yet: `PouwCheck.protocol` is `verity/pouw/v1`, a window check that reads no call's shape, coins
or egress. So these properties are stated over any `Core.Protocol` with that id and the readers it must expose
(`Reads`), a stub of the case's own functions of its commitment and evidence. When the case exists, its readers
instantiate `Reads`, and the properties are stated at `PouwCheck`'s v2 protocol itself.

`Accepted` has three conjuncts at a call's coins `ω`, and each reaches the case differently:

* **The admitted call** (`L.Admits`: `1 ≤ K` and at most `2^40` MMA tiles) is the case's own check of the call's
  registered shape, public and fixed before its coins. It refuses a call whose layout the lead doesn't admit
  (`ShapeAdmitted`).
* **Egress** (`(σ.exec ω).egress ≤ L.callBits`) is the case's own check. It refuses a call whose bits, as the network
  certifier counted them, exceed the call's share of its forward's cap (`EgressWithin`). The certifier counts every
  bit the node exchanges (`trusted-host`), so the execution's egress is at most the count.
* **The honest root** (`(σ.exec ω).root = C.root H (word L κ ω)`): the case reads the root the call registered, and
  coins the certifier drew after the call's `A` root (`CoinsAfterA`). A drawn unit's proof establishes its tiles'
  words under that root. That every unit's statement holds is sampling's guarantee, at the window's ε and δ, not the
  case's. The case establishes only that the units cover the call's tiles once each and that every drawn unit's proof
  was accepted (`UnitsCover`).

`TiesToAccepted` is the property the accounting reads. `AcceptedOf` is what it gives: at an accepted window's credited
call, a strategy that registered the call's root and exchanged at most what the certifier counted, and whose root is
its units' statements' honest commitment, is `Accepted` at the case's coins.
-/

namespace Properties.Pouw.Lead.V2

open _root_.Pouw.Lead

/-- The lead's layout at a call's registered shape `(T, m, n, K)`. -/
def layoutAt (s : ℕ × ℕ × ℕ × ℕ) : Layout := lead s.1 s.2.1 s.2.2.1 s.2.2.2

/-- **What the v2 case reads of a window**: a stub of `PouwCheck`'s v2 readers, each a function of the case's
commitment and evidence. They give the calls the case credits; each call's registered shape, `A` root and words root;
the coins the network certifier signed for the call, in the lead's arrangement, and the `A` root its signed record
says it drew them after; the bits the certifier counted for the call; the registered partition's units of the call,
each a set of the call's MMA tiles; the units the window's coin drew; and whether the verifier of record accepted each
drawn unit's proof. -/
structure Reads (P : Core.Protocol) where
  /-- The calls the window credits. -/
  calls : P.Commitment → P.Evidence → List ℕ
  /-- A call's shape `(T, m, n, K)`, registered before its coins. -/
  shape : P.Commitment → ℕ → ℕ × ℕ × ℕ × ℕ
  /-- A call's `A` root, registered before its coins. -/
  aRoot : P.Commitment → ℕ → Hash.Digest
  /-- A call's words root, as registered. -/
  root : P.Commitment → ℕ → Hash.Digest
  /-- A call's coins, from the certifier's signed record, in the lead's arrangement. -/
  coins : (c : P.Commitment) → P.Evidence → (i : ℕ) → Coins (layoutAt (shape c i))
  /-- The `A` root the certifier's signed record says its coins were drawn after. -/
  coinsAfter : P.Evidence → ℕ → Hash.Digest
  /-- The bits the certifier counted for a call. -/
  egress : P.Evidence → ℕ → ℕ
  /-- The registered partition's units of a call, each a set of its MMA tiles (row block, column block, step). -/
  units : (c : P.Commitment) → (i : ℕ) → List (Finset (ℕ × ℕ × Fin (layoutAt (shape c i)).S))
  /-- The units of a call the window's coin drew, as indices into `units`. -/
  drawn : P.Evidence → ℕ → List ℕ
  /-- Whether the verifier of record accepted a drawn unit's proof. -/
  proved : P.Evidence → ℕ → ℕ → Bool

variable (P : Core.Protocol) (r : Reads P)

/-- **The case refuses a call it doesn't admit**: at an accepted window, each credited call's registered shape lays
out as a call the lead admits. -/
def ShapeAdmitted : Prop :=
  ∀ p c e, P.Accepts p c e → ∀ i ∈ r.calls c e, (layoutAt (r.shape c i)).Admits

/-- **The case refuses a call past its bits**: at an accepted window, each credited call's certifier-counted bits are
at most its share of its forward's cap. -/
def EgressWithin : Prop :=
  ∀ p c e, P.Accepts p c e → ∀ i ∈ r.calls c e, (r.egress e i : ℚ) ≤ (layoutAt (r.shape c i)).callBits

/-- **The coins follow `A`**: at an accepted window, each credited call's coins come from a certifier record that was
drawn after the call's registered `A` root. -/
def CoinsAfterA : Prop :=
  ∀ p c e, P.Accepts p c e → ∀ i ∈ r.calls c e, r.coinsAfter e i = r.aRoot c i

/-- **The units cover the call's tiles once each, and every drawn unit was proved**: at an accepted window, each
credited call's units are pairwise disjoint, their union is the call's committed tiles (`tileIdx`), every drawn index
names a unit, and the verifier of record accepted each drawn unit's proof. -/
def UnitsCover : Prop :=
  ∀ p c e, P.Accepts p c e → ∀ i ∈ r.calls c e,
    (r.units c i).Pairwise Disjoint ∧
      (∀ x, x ∈ tileIdx (layoutAt (r.shape c i)) ↔ ∃ U ∈ r.units c i, x ∈ U) ∧
      ∀ u ∈ r.drawn e i, u < (r.units c i).length ∧ r.proved e i u = true

/-- **The v2 case ties to the lead's acceptance**: it is PoUW's `verity/pouw/v2` case, and what it accepts keeps the
shape refusal, the egress refusal, the coins' order and the units' cover. -/
def TiesToAccepted : Prop :=
  P.id = "verity/pouw/v2" ∧ ShapeAdmitted P r ∧ EgressWithin P r ∧ CoinsAfterA P r ∧ UnitsCover P r

/-- **`Accepted` at the case's coins**: at an accepted window's credited call, a strategy that exchanged at most the
certifier's count and registered the call's root is `Accepted` at the call's coins, once that root is the honest
commitment of the words `κ`. That last hypothesis is the call's units' statements, which sampling establishes. -/
def AcceptedOf : Prop :=
  ∀ p c e, P.Accepts p c e → ∀ i ∈ r.calls c e,
    ∀ (H : Hash.Msg → Hash.Digest) (C : Commit (layoutAt (r.shape c i)))
      (κ : Fin (layoutAt (r.shape c i)).m → Fin (layoutAt (r.shape c i)).S → Fin (layoutAt (r.shape c i)).n → ℤ)
      (σ : Strategy (layoutAt (r.shape c i))),
      (σ.exec (r.coins c e i)).egress ≤ r.egress e i →
      (σ.exec (r.coins c e i)).root = r.root c i →
      r.root c i = C.root H (word (layoutAt (r.shape c i)) κ (r.coins c e i)) →
      Accepted (layoutAt (r.shape c i)) H C κ σ (r.coins c e i)

end Properties.Pouw.Lead.V2
