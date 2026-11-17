module

public import Proofs.Flock.SoundAtBridge.Union
public import Proofs.Flock.SoundAtBridge.Session
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

/-! `SoundAtBridge`, PoUW's proof premise from Flock's rows link (trusted); importing this module makes `X` a token
(ArkLib's global `X` notation), so a module whose declarations bind `X` can't import it. -/

@[expose] public section

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Discharge.ZkReg ResourceAccounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

variable {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type}

/-- **Gap 1, the window's sampling design** (open): at each round `w` and
proof `i`, the window's coins, read through the session proof `i` runs and its outcome, have the sessions' sampling design
(`SessionSamplingDesign`): the session's coins are the path's, fresh and uniform given its setting and its prover. -/
def ExecutionPhaseSamplingDesign (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out) : Prop :=
  ∀ w i, SessionSamplingDesign (Out := fun s : RowsSession => s.Out) (fun s => s.G) (fun s => s.τ) (fun x => sess x w i)
    (fun x => out x w i)

/-- **Gap 1, acceptance** (open): the layer accepts proof `i` of round `w` only
where verity-verify-proof accepts its session's outcome (`LiveAcceptsZKR`). -/
def AcceptedLive (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) : Prop :=
  ∀ x w i, pf.accepted x w i → LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)

/-- **Gap 2, the opening is the plurality** (open): at each
`(w, i)`, an accepted proof's opening is other than `rows` of its session's value layer (`XplurZC`) with probability
at most `δo`; `δo = 0` when `opened` is defined as that value (`openedPlur_zero`). -/
def OpenedPlurAt (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δo : ℝ) : Prop :=
  ∀ w i, Pouw.RelativeSecurity.prob
    (fun x => pf.accepted x w i ∧ pf.opened x w i ≠ rows (sess x w i) (sess x w i).Xv) ≤ δo

/-- **Gap 1, the relation is what the sampled proof blocks certify** (open): on every path, when no sampled proof block of proof `i`'s
session is wrong (`IncorrectAt`) and every sampled proof block's rows are right at its own owned reads (`RowsAt`), proof `i`'s
relation holds at `rows` of the value layer; it can hold only for a relation that reads no unsampled unit and no input
row at a read the verifier doesn't own. -/
def RelOfRows (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) : Prop :=
  ∀ x w i, (∀ u ∈ (out x w i).2.1, ¬ (sess x w i).IncorrectAt u ∧ (sess x w i).RowsAt (out x w i) u) →
    pf.rel x w i (rows (sess x w i) (sess x w i).Xv)

end Flock.SoundAtBridge

namespace Flock.Guarantees

open Flock.SoundAtBridge ResourceAccounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

/-- **PoUW's proof premise from Flock's rows link, at `δ_ZK = δo + δ`.** For any window of proofs
(`pf : ResourceAccounting.TS.ProofLayer Ω S nπ V`, its relation on the path) whose proof `i` of round `w` on the
path `x` is the Flock session `sess x w i` (`EndToEndRegisteredRows`' setting with one registrant's opening per
unit, `RowsSession`) with outcome `out x w i`, and a reading `rows s` of a session's value layer as a layer value:
given `EndToEndRegisteredRows`' named assumptions at every session and every unit's opening (`RowsSession.CR`), the
window's sampling design (`ExecutionPhaseSamplingDesign`), acceptance (`AcceptedLive`), the opening's agreement with the plurality outside `δo`
(`OpenedPlurAt`) and the relation's reading of the sampled proof blocks (`RelOfRows`), and `δ ≥ 0` with
$$n \cdot (\varepsilon_{ks} + \delta_{link} + \delta_{rows} + q_R^2/2^{513}) \le \delta$$
at every session a path runs (`RowsSession.bound`, `n` its units), each proof is accepted with an opening that fails
its relation on the path with probability at most `δo + δ` (`pf.SoundAt (δo + δ)`). -/
def SoundAtBridge : Prop :=
  ∀ {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)
    (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (δo δ : ℝ),
    -- cr/sha-512 and ecr/sha-512 as EndToEndRegisteredRows takes them, at every session and every unit's opening
    (∀ x w i, (sess x w i).CR) →
    ExecutionPhaseSamplingDesign sess out →
    AcceptedLive pf sess out →
    OpenedPlurAt pf sess rows δo →
    RelOfRows pf sess out rows →
    (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal δ) → 0 ≤ δ →
    pf.SoundAt (δo + δ)

end Flock.Guarantees
