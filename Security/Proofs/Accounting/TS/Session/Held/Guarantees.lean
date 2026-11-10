module

public import Proofs.Accounting.TS.Session.Held.Assumptions
public import Proofs.Accounting.TS.Session.Assumptions
public import Proofs.Flock.SoundAtBridge.Served
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# A checked unit's proof layer with its served checks is sound through its Flock sessions (the statement)

`Accounting.Guarantees.TS.HeldSessionSound`: `Accounting.Guarantees.TS.CheckedSessionSound`'s window, with the layer
`Accounting.TS.Held.heldLayer` in place of `checkedLayer`. Its relation concludes the served checks:

* (a) the `ok` rows are the verifier's, proved from flock-verify's known-content refusal (`Registered.checkKnownAll`)
  and `FlockC3.RegLevels.knownColl_spec`, not a premise. The service names the `ok` port's known content in the
  registered file, each row the word `1` (`okRow`) under its salt;
* (b) every sampled proof block's checks hold in the declared circuit (`Accounting.Served.Held`), under
  `Accounting.Assumptions.TS.OkRealizes`, the premise that the Flock circuit's `ok` wires carry them.

The bound is `CheckedSessionSound`'s `δ`: an accepted proof whose rows meet `CheckedRel` meets `HeldRel` on every path,
so the held layer's unsound event is inside the checked layer's. `CheckedSessionSound`'s statement is unchanged. It sits
in the proofs package because the sessions do (`reads_exempt`).
-/

@[expose] public section

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.TS.Checked Accounting.TS.Held Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A checked unit's proof layer, with its served checks, is sound at `δ` through its Flock sessions, fully
checked.** `CheckedSessionSound`'s window and premises, at the held layer `heldLayer sess out ok known Cd cls unitOf dec
right` (accepted where flock-verify `--zk --registered` accepts, at the registered file's known content `known x w i`;
opening the committed rows at the registered reads; its relation `HeldRel`), and in addition at every session: the known
content names the port `ok` with rows each the word `1` (`okRow`), and its key program's `ok` wires carry the checks of
round `w`'s declared circuit `Cd w` (`Accounting.Assumptions.TS.OkRealizes`). Then the layer accepts only where
flock-verify accepts (`AcceptedLive`), and each proof is accepted with committed rows that fail `HeldRel` on the path
with probability at most `δ`: outside `δ` the rows are one assignment the circuit's evaluation agrees with on every
sampled proof block, at which every sampled proof block's checks of `Cd` hold, or the registrant's `ok` openings exhibit an explicit
SHA-512 collision (`OkColl`). -/
def HeldSessionSound : Prop :=
  ∀ {ι Ω : Type} [Fintype Ω] {S nπ : ℕ} (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (ok : String)
    (known : Paths Ω S → Fin S → Fin nπ → Array (String × Array (Array ℕ × ByteArray)))
    (Cd : Fin S → Accounting.Compose.Declared ι) (cls : Fin S → ℕ → Accounting.Served.Cls)
    (unitOf : ∀ x w i, Fin (sess x w i).n → ℕ) (dec : ∀ x w i, (Fin (sess x w i).Circ.N → Bool) → ι → ℤ)
    (right : Paths Ω S → (Fin S → Fin nπ → ℕ → ℕ → Option Bool) → Prop) (δ : ℝ),
    (∀ x w i, (sess x w i).CR) →
    Accounting.Assumptions.TS.WindowDesign sess out →
    (∀ x w i, (sess x w i).kd = (sess x w i).n) →
    (∀ x w i u, (sess x w i).ServedRegistered u) →
    (∀ x w i, (∃ q : Fin (sess x w i).c.ports.size, (sess x w i).c.ports[q].name = ok) ∧
      ∃ pt ∈ (sess x w i).own, pt.name = ok) →
    (∀ x w i, ∃ ks, (ok, ks) ∈ known x w i ∧ ∀ r ∈ ks, r.1 = okRow) →
    (∀ x w i, Accounting.Assumptions.TS.OkRealizes (sess x w i) ok (Cd w) (cls w) (unitOf x w i) (dec x w i)) →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    Accounting.Assumptions.TS.AcceptedLive (heldLayer sess out ok known Cd cls unitOf dec right) sess out ∧
      (heldLayer sess out ok known Cd cls unitOf dec right).SoundAt δ

end Accounting.Guarantees.TS
