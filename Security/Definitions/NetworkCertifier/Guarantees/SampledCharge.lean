module

public import Definitions.NetworkCertifier.SampledCharge
public import Definitions.NetworkCertifier.SampledReg
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
# The sampled audit's charge: the leak bound when step proof blocks are sampled

Theorem 3's bounds on `constantRate` (`Guarantees`) count runs that the Program accepts exactly. A sampled
audit gives that only at `δ ≥ miss(1)`, where no unit may be faulty. At a smaller `δ` it gives `Accepted` by the
circuit's Program (`circuitProgram`) with advice also held at the incorrect proof blocks' committed gates
(`SampledAuditAccepted`, `Guarantees/SampledAudit`). `EncardAccSeqsSampledLe` counts those runs: the charge adds the
incorrect proof blocks' committed bits and `log2` of the number of possible incorrect sets.

**Composed with `SampledAuditAccepted`.** On the statements in its scope, which include that the layout fits the program
(`BoundaryGates.Fits`), an accepted audit meets `RAcc δ` except with probability `δ + ε_ks + ε_bind` over the selection.
On `RAcc δ`, the incorrect set is one of `faultySets d P δ`. Every observation sequence the trace's transfer gates carry,
with every status complete, is then `Accepted` with advice held only at that set's committed gates. So, with its inputs
other than the ingress among `As`, the run's observations are among the sequences this guarantee counts. The leak is
then at most `c + log2 #faultySets + β + n·log2 D` bits, except with that probability. That the transfer gates carry the
run's observations is the certifier's trust (`trusted-host`), and the statuses are its published record. The theorem is
`NetTiming.sampledLeak` (`Proofs.NetworkCertifier.SampledLeak`), with `SampledAuditAccepted` a hypothesis. `RAcc` pins
the inputs besides the ingress to the trace's committed values, so it takes that inputs among `As` suffice for the
trace's (`AdviceIn`); `sampledLeak_reg` proves that for every circuit at the inputs pinned to the trace at the
registered inputs `reg` and `false` off the free inputs (`AsReg`), with `c` the number of free inputs not in `reg`.
`SampledLeakAccepted` is that bound as a guarantee, with `SampledAuditAccepted` discharged by its proof.
`SampledLeakRegistered` is the same bound with the registered inputs at the registration's values (`regVal`), which it
opens under the statement's roots (`RegOpens`), and binding (`CommitmentScheme.Binds`) in place of the trace's: the values the
count leaves out are then the registration's, and an extracted trace that disagrees with them is a collision `B_reg`
finds (`NetTiming.sampledLeak_registered`).

**What `As` and `c` are.** `As` is what the circuit's inputs other than the ingress may be. Its count is a premise of the
run, not an assumption. Under the decided composition the registered inputs are fixed by the registration, and the
advice is advice v1. Its charge, the coded length counted by the in-circuit encoder, is at most the declared public `c`,
and at most `2^c` advice values are charged `c` or less (`hAs`), as for any prefix-free code (Kraft).
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims.Sampling
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak bound under a sampled audit.** Take the circuit's Program of `C` over a layout `lay` that fits it up to
the horizon `n`, a partition `P` and a design `d`. The observation sequences of length `n`, with jitter below `D`, that
are accepted with some possible incorrect set at `δ` (`AccSeqsSampled`) number at most
`2^c · #faultySets · 2^β · D^n`. Here the inputs other than the ingress range over `As`, at most `2^c` values, and each
possible incorrect set's proof blocks commit at most `β` bits (`faultyBits`). That is
`c + log2 #(faulty sets) + Σ_{v faulty} committed bits + n·log2 D` bits. At `δ ≥ miss(1)` no non-empty incorrect set
is possible, and the bound is `c + n·log2 D`, as when the Program is accepted exactly. -/
def EncardAccSeqsSampledLe : Prop :=
  ∀ {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ) (hfits : lay.Fits C n)
    (δ : ℝ) (As : Set (ℕ → Bool)) (c : ℕ) (hAs : As.encard ≤ ((2 ^ c : ℕ) : ℕ∞)) (β : ℕ)
    (hβ : ∀ F ∈ faultySets d P δ, faultyBits C P F ≤ β) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ),
    (AccSeqsSampled C P d lay n As δ op D).encard ≤
      ((2 ^ c * (faultySets d P δ).card * 2 ^ β * D ^ n : ℕ) : ℕ∞)

/-- **An accepted sampled audit bounds the leak.** At every compiled sampled verifier `D` where
`Claims.SamplingGeneric.Compiled.KS` holds, every public layout `lay` and horizon `k` of its statements, `δ ≥ 0`, and
`β` bounding each possible incorrect set's committed bits (`faultyBits`), on the statements in scope:

* `Pr[acc ∧ ¬RLeak δ] ≤ δ + ε_ks + ε_bind`: except with that probability, every observation sequence of length `k` with
  jitter below `Dj` that the trace's transfer gates carry with every status complete is among the sequences
  `AccSeqsSampled` counts, with the inputs besides the ingress the trace's at the registered inputs `reg` and `false`
  off the free inputs (`AsReg`);
* at whatever values `ρ` the registered inputs hold, those sequences number at most
  `2^#(freeInputs \ reg) · #faultySets(δ) · 2^β · Dj^k`: the leak is at most
  `#(freeInputs \ reg) + log2 #faultySets(δ) + β + k·log2 Dj` bits.

`reg` is any set of each statement's inputs: the count leaves out the values the trace holds there. Those values are
public only where something binds them, as the registration's commitment binds the registered inputs. -/
def SampledLeakAccepted : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (β : AuditStatement → ℕ) (Dj : ℕ)
    (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s),
    KSLinOn H D.V (ScopeAcc D lay k)
        (RLeak H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj)
        D.E (D.κ δ) D.γ (fun _ => 1) (Bext H D.V D.E) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        ∀ (op : ℕ → List (Obs L K S) → List Bool) (ρ : ℕ → Bool),
          (AccSeqsSampled s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) δ op Dj).encard ≤
            ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (faultySets s.design P δ).card * 2 ^ β s *
              Dj ^ k s : ℕ) : ℕ∞)

/-- **An accepted sampled audit bounds the leak at the registration's values.** As `SampledLeakAccepted`, where the
trace's commitment also binds (`CommitmentScheme.Binds`) and each statement's registration opens its registered inputs
`reg` under the statement's roots (`RegOpens`, openings `ops`), on the statements in scope:

* `Pr[acc ∧ ¬RLeak δ] ≤ δ + ε_ks + ε_bind + Pr[B_reg]`, with the inputs besides the ingress pinned at the registration's
  values at `reg` (`AsReg` at `regVal`), a set the statement and the registration fix, and `B_reg` the collision finder
  on the extraction's calls and those that check the registration's openings (`Breg`);
* those sequences number at most `2^#(freeInputs \ reg) · #faultySets(δ) · 2^β · Dj^k`. -/
def SampledLeakRegistered : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (hb : D.commit.Binds H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) (β : AuditStatement → ℕ) (Dj : ℕ)
    (hβ : ∀ s P, D.partition s = some P → ∀ F ∈ faultySets s.design P δ, faultyBits s.prog P F ≤ β s),
    (∀ x, ScopeAcc D lay k x → (∀ s, D.parse x = some s → RegOpens H D.commit s (reg s) (ops s)) →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
        Claims.PrBad H D.V
            (RLeak H D.commit D.parse D.partition L K S lay k δ
              (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj) D.E x P ≤
          D.κ δ x + Real.sqrt (D.γ x * Claims.PrB H D.V P x) + (Bext H D.V D.E Ω P x).Pr H +
            (Breg H D.V D.E D.commit D.parse ops Ω P x).Pr H) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        ∀ op : ℕ → List (Obs L K S) → List Bool,
          (AccSeqsSampled s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) δ op
              Dj).encard ≤
            ((2 ^ (freeInputs s.prog (lay s) (k s) \ reg s).card * (faultySets s.design P δ).card * 2 ^ β s *
              Dj ^ k s : ℕ) : ℕ∞)

end NetTiming.Guarantees
