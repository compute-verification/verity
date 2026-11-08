module

public import Definitions.Core.Interaction
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
# Claims: zero knowledge with a public outcome

The claims core's one zero-knowledge notion for a protocol whose outcome is public (each verdict, and where a session
stopped): per outcome, the honest transcript's probability of the outcome and an event is at most a simulator's,
given the outcome, of the event, plus an error of the statement, the witness, the verifier and the outcome.

* `SimOut`: a simulator given the outcome, fixed before the witness and the verifier.
* `ZKOut`: zero knowledge per outcome (`ZKmalRx`'s one-sided form at each outcome).
* `ZKOutLeak`: the leak form over finitely many outcomes, its corollary: the number of outcomes times the simulator
  that samples the outcome uniformly, plus the sum of the outcomes' errors.
-/

@[expose] public section

namespace Claims

/-- **A simulator given the outcome**: its coins, and its program, which also receives the protocol's public outcome
`o ∈ St` (each verdict, and where a session stopped). -/
structure SimOut (St : Type) where
  C : Msg → Type
  [fin : ∀ x, Fintype (C x)]
  [ne : ∀ x, Nonempty (C x)]
  prog : (x : Msg) → St → C x → SP (List (Msg × Out))

attribute [instance] SimOut.fin SimOut.ne

/-- The simulator given the outcome `o`, as the spec's `Simulator`. -/
def SimOut.given {St : Type} (S : SimOut St) (o : St) : Simulator where
  C := S.C
  prog x c := S.prog x o c

/-- **Zero knowledge with a public outcome**, per outcome (`ZKmalRx`'s one-sided form at each outcome): for `(x, w) ∈ R`,
every verifier `V*` of the protocol's `k(x)` rounds and tape `θ`, every outcome `o` and every event, the honest
transcript's probability of showing the outcome `o` and the event is at most the simulator's, given `o`, of the event,
plus `ε(x, w, V*, θ, o)`. `out` reads the outcome off the transcript, so the outcome is public; the simulator is fixed
before the witness and the verifier; `ε` is a function of the statement, the witness, the verifier and the outcome, such
as a collision finder's success built from them (`Fork`), never an assumed bound. -/
def ZKOut {W St : Type} (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x)) (R : Rel W)
    (out : Msg → List (Msg × Out) → St) (S : SimOut St) (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ) :
    Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (o : St) (Ev : List (Msg × Out) → Prop),
    Pr (fun ω : N x × Ω x =>
        out x (runStar H (P w x) Vs x ω θ).msgs = o ∧ Ev (runStar H (P w x) Vs x ω θ).msgs) ≤
      Pr (fun c : S.C x => Ev ((S.given o).out H Vs θ x c)) + ε x w Θ Vs θ o

/-- **The leak form, from the per-outcome form**: over finitely many outcomes, on every event the honest transcript's
probability is at most the number of outcomes times the simulator's that samples the outcome uniformly, plus the sum of
the outcomes' errors. -/
def ZKOutLeak : Prop :=
  ∀ {W St : Type} [Fintype St] [Nonempty St] (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x))
    (R : Rel W) (out : Msg → List (Msg × Out) → St) (S : SimOut St)
    (ε : (x : Msg) → W → (Θ : Type) → VStar Θ → Θ → St → ℝ),
    ZKOut N Ω H k P R out S ε →
    ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (Ev : List (Msg × Out) → Prop),
      Pr (fun ω : N x × Ω x => Ev (runStar H (P w x) Vs x ω θ).msgs) ≤
        (Fintype.card St : ℝ) * Pr (fun p : St × S.C x => Ev ((S.given p.1).out H Vs θ x p.2)) +
          ∑ o, ε x w Θ Vs θ o

end Claims
