module

public import Proofs.Pouw.TS.Session.Group.Premises
public import Proofs.Pouw.TS.Session.Lanes.Canon
public import Proofs.Pouw.TS.Session.Group.Layer
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

/-! The statement that a leaf proved as a group of `L n + 1` one-unit Flock sessions is accepted with a failing relation
and no collision in hand with probability at most `(L n + 1) δ`. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.TS.Group Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **A leaf's group of linked Flock sessions realizes one unit, any shape, fully checked.** For any tile `T` whose
`T.lanes` lanes take `n` chunks each, and any window whose proof `i` of round `w` on the path `x` is a leaf proved as its
group's sessions `sess x w i k` (`k : Ix T.lanes n`: lane `l`'s chunk `c` at `some (l, c)`, the combine at `none`), each
a session `sess₀ x w i k` at its registrants' openings `op x w i k` with its registered reads built from `own`
(`RowsSession.canon`), with outcomes `out x w i k`, and the group layer
`groupLayer T n sess out known okSalt full want opened rel right` (accepted where the leaf verifier's check of the tile
and its chunks passes, `PouwWindow.Lanes.checkChunks T n`, and at every one of the `T.lanes · n + 1` sessions
verity-verify-proof `--zk --registered` accepts with its known-content, complete and known-read checks, at the known content
`known x w i k` followed by its `o0` row 1 under `okSalt x w i k` and the ports `full x w i k` it reads in full, and
`pouw-leaf-own`'s checks of its file at the ports `want x w i k`, of its sample, every unit of a non-empty population,
and of its reads, one at each port, pass; opening `opened x w i`, its relation `rel x w i`, or a clash of two of the
sessions' registrants' records, `LeafColl`, or a collision of a session's known-content tie, `LeafKnownColl`): given
the named premises `ResourceAccounting.Premises.TS.SessionsCR` (`RowsSession.CR` at every session),
`ResourceAccounting.Premises.TS.ExecutionPhaseDesigns` (the window's sampling design at every statement) and `ResourceAccounting.Premises.TS.GroupRealizes`
(the group's circuits realize `rel` of `opened` at the rows the sessions open, past the leaf verifier's check of the
tile and its chunks), and
$$n_s \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta,$$
`δ ≥ 0`, at every session (`n_s` its units), each proof is accepted with an opening that fails `rel`, and no clash of its
sessions' registrants' records or of their known-content ties, with probability at most `(T.lanes · n + 1) δ`. -/
def LeafGroupSound : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ n : ℕ} {V : Type} (T : PouwWindow.Lanes.Tile)
    (sess₀ : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → RowsSession)
    (op : ∀ x w i k, (sess₀ x w i k).Openings),
    let sess := fun x w i k => (sess₀ x w i k).canon (op x w i k)
    ∀ (out : ∀ x w i k, (sess x w i k).Out)
    (known : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Lanes.Known)
    (okSalt : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → ByteArray)
    (full : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → Ix T.lanes n → List _root_.Flock.Registered.Port)
    (opened : Paths Ω S → Fin S → Fin nπ → V) (rel : Paths Ω S → Fin S → Fin nπ → V → Prop)
    (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) (δ : ℝ),
    ResourceAccounting.Premises.TS.SessionsCR sess →
    ResourceAccounting.Premises.TS.ExecutionPhaseDesigns sess out →
    ResourceAccounting.Premises.TS.GroupRealizes sess (fun x w i k => okKnown (known x w i k) (okSalt x w i k)) full want
      (fun _ _ _ => PouwWindow.Lanes.checkChunks T n = .ok ()) opened rel →
    (∀ x w i k, ((sess x w i k).n : ℝ≥0∞) * (sess x w i k).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    (groupLayer T n sess out known okSalt full want opened rel right).SoundAt ((T.lanes * n + 1) * δ)

end ResourceAccounting.Guarantees.TS
