module

public import Proofs.Flock.Claims.Recursion.Deploy
public import Proofs.Flock.Claims.Recursion.DeployScope
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

/-! What the recursion's zero knowledge rests on at `deployRec` (`zk_deployRec_iff`): the inner prover's hm96 commit
strings (`Hm96Hiding`) and Flock's ZK at `deployZkRec`, both premises here. -/

@[expose] public section

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open scoped NNReal

variable {N Ω : Claims.Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
  [∀ x, Nonempty (Ω x)]

noncomputable def deployRecZ (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZkDeploy Coin :=
  { deployRec IV ivScope b with Nonce := N, Rest := Ω, tapeOf := tapeOf, SO := SO }

def ZKBound (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b)
    (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) : Prop :=
  ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
    Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
      Claims.ZKmalRx N Ω H (deployRec IV ivScope b).V.k (deployRecZ IV ivScope b tapeOf SO).prover
        (Claims.Recursion.R (deployRec IV ivScope b) I) (Claims.Recursion.Sim (deployRec IV ivScope b) I (SO t))
        (Claims.Recursion.εzk (deployRec IV ivScope b) I t δ₁) (Claims.Recursion.γzk (deployRec IV ivScope b) t)

theorem zk_deployRec (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (h : ZKBound H IV ivScope b I tapeOf SO) :
    Claims.Recursion.ZK H (deployRecZ IV ivScope b tapeOf SO) I (deployZkRec relO) :=
  ⟨rfl, h⟩

theorem zk_deployRec_iff (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZK H (deployRecZ IV ivScope b tapeOf SO) I (deployZkRec relO) ↔
      ZKBound H IV ivScope b I tapeOf SO :=
  ⟨And.right, zk_deployRec H IV ivScope b I relO tapeOf SO⟩

theorem refP_stage (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (w : Claims.Flock.Wit)
    (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg)) (ω : Tape b) (h₁ : IV.k (xIOf x) ≤ hist.length)
    (h₂ : hist.length < IV.k (xIOf x) + kS x) :
    (refP IV b w).next x hist ω =
      commitThen (fun cs => (deployRec IV ivScope b).stage x (asRounds (hist.take (IV.k (xIOf x)))) cs)
        ((List.range ((deployRec IV ivScope b).nC x)).map fun j =>
          ((b.ref.digests w ω.1 (hist.take (IV.k (xIOf x)))).getD j 0, ω.2.1 j)) [] := by
  simp only [refP, deployRec, Nat.not_lt.2 h₁, h₂, ↓reduceIte]

theorem commitThen_zero (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (k : List Claims.Msg → Claims.OC Claims.Msg) (y : Salt) (ps : List ((Fin 64 → UInt8) × Salt))
    (cs : List Claims.Msg) :
    commitThen k ((0, y) :: ps) cs =
      Claims.OC.bind ((deployRec IV ivScope b).zeroCommit y) fun c => commitThen k ps (cs ++ [c]) := by
  simp only [commitThen, Claims.Recursion.Deploy.zeroCommit, Claims.OC.bind, deployRec, maskOf, Pi.zero_apply,
    UInt8.zero_xor]

theorem nums_εzk (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (x : Claims.Msg) (s : ℕ)
    (t : ℕ) (δ₁ : ℝ) :
    ((deployRec IV ivScope b).nums x s).εzk t δ₁ =
      8 * (b.zl (staged b.st x (pick b.st x s) s)).1 * δ₁ + 8 * t * (b.zl (staged b.st x (pick b.st x s) s)).2 * δ₁ +
        2 * (1 / 3) ^ t :=
  rfl

theorem parse_circ {x : Claims.Msg} {s : Claims.Recursion.CompiledStatement} (h : parse x = some s) : s.circ = none := by
  simp only [parse, RecScope.parseOf, Option.map_eq_some_iff] at h
  obtain ⟨r, -, rfl⟩ := h
  rfl

theorem zkPriv_deployRec (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZKPriv H (deployRecZ IV ivScope b tapeOf SO) I (deployZkRec relO) := by
  refine ⟨rfl, fun t _ δ₁ _ x w hR => ?_⟩
  obtain ⟨s, hs, hc, -⟩ := hR
  rw [parse_circ (x := x) hs] at hc
  cases hc

end RecDeploy
