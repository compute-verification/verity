module

public import Proofs.Flock.SoundAtBridge.Session
public import Proofs.Flock.C3.Compose.RowsReg
public import Proofs.Flock.EndToEndRegistered
public import Proofs.Flock.SoundAtBridge.Reduce
public import Proofs.Flock.SoundAtBridge.Served
public import Proofs.Flock.SoundAtBridge.Statement
public import Proofs.Pouw.TS.Session.Held.Known
public import Definitions.ResourceAccounting.Served
public import Batteries.Data.ByteArray
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
# A checked unit's `ok` rows at its Flock sessions

verity-verify-proof `--registered F` registers the `ok` port at known content (`Registered.KNOWN_KEY`: h1_prove's rows,
each the one word `1` under the public `ok_salt`), and refuses a file whose `ok` root is not the root of those rows
(`Registered.checkKnownAll`). On an accepted session the registrant's row at every read of a port named `ok` has the
`sha512/row/v2` bytes of the word `1` (`OkVal`), or the registrant's openings exhibit a SHA-512 collision (`OkColl`, a
fact of the session's commit-time openings, with both preimages in hand: `FlockC3.RegLevels.knownColl`).

These definitions read `RowsSession`, in the proofs package, so they sit here under `reads_exempt`'s
`Proofs.Pouw.TS.Session`.
-/

@[expose] public section

namespace ResourceAccounting.TS.Held

open ResourceAccounting.TS Flock.SoundAtBridge FlockSoundness.Discharge.ZkReg
open Pouw.RelativeSecurity (Paths)
open FlockSoundness.Refine (H512)

/-- The `ok` port's row: the one u16 word `1` (`h1unit.register_ok`). -/
def okRow : Array ℕ := #[1]

/-- The session's row encodings (`rsShHJS`), the ones its registrants' openings are of. -/
noncomputable abbrev rsOf (s : RowsSession) :=
  rsShHJS (Reg := s.Reg) s.dj s.y₀ s.ht s.hc s.hscope s.hU s.hpub

/-- **Wire `g` of unit `u` is an `ok` wire**: a wire of `u` (`P.io`) at a registered read of `rdU u` whose registered
port is named `ok`. -/
def OkWire (s : RowsSession) (ok : String) (u : Fin s.n) (g : Fin s.Circ.N) : Prop :=
  g ∈ s.P.io u ∧ (s.rdU u).isRead (s.vb.pos g) ∧ ((s.rdU u).port (s.vb.pos g)).name = ok

/-- **`v` is the `ok` row at the row position `p`**: its row bytes there (`rsOf`'s encoding, `sha512/row/v2` at a v2
row port) are the `ok` row's (`Registered.rowBytesV2 okRow`). -/
def OkVal (s : RowsSession) (p : s.vb.Pos) (v : s.vb.Val) : Prop :=
  (rsOf s).enc p v = (Flock.Registered.rowBytesV2 okRow).data.toList

/-- **The `ok` tie collides at the session**: at some registered read `p` of some unit's registrant, at a port named
`ok`, the known-content extractor against the `ok` port's known rows `ks` (`knownColl`) outputs two different SHA-512
inputs with one digest. -/
def OkColl (s : RowsSession) (ok : String) (ks : Array (Array ℕ × ByteArray)) : Prop :=
  ∃ (u : Fin s.n) (p : s.vb.Pos), (s.rdU u).isRead p ∧ ((s.rdU u).port p).name = ok ∧ ∃ a b,
    FlockC3.RegLevels.knownColl ((s.rdU u).port p) ks ((s.rdU u).readAt p) ((rsOf s).enc p ((s.rdU u).wv p))
      ((s.rdU u).ws p) ((s.rdU u).wpath p) = some (a, b) ∧ a ≠ b ∧ H512 a = H512 b

end ResourceAccounting.TS.Held
