module

public import Definitions.NetworkCertifier.Checked
public import Definitions.NetworkCertifier.Guarantees.SampledAudit
public import Definitions.Core.Assumptions
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
# The leak bounds with checked inputs

`SampledLeakAccepted` and `PooledLeakAccepted` with each statement's checked inputs `chk` left out of the count: the free
inputs a check on the whole run pins (`Definitions.NetworkCertifier.Checked`). Each statement's check pins its checked
inputs (`CheckedUnique`), and the trace's run is valid except with probability `εchk` (`CheckedWitness`), so a run
whose observations the bound counts is a valid run, whose checked inputs the rest of the run fixes: they cost no advice
bits. The incorrect proof blocks' committed bits are charged at the most any possible incorrect set commits, rather than
at a bound the statement takes.

The monitoring period's statement takes its inputs as `PooledLeakAccepted` does, named (`PeriodChannel`,
`PeriodAdaptive`, `PeriodEscapes`, `PeriodLeaksChecked`), and bounds the escape by `(1 - q)^Fmax` itself, which is at most
`δ` at `Fmax ≥ ln(1/δ)/q`. It is conditional as `PooledLeakAccepted` is: its inputs are `Flock.Guarantees.SampledKSLive`
and the live `SampledLeakChecked` at each window's history, neither discharged yet.
-/

@[expose] public section

namespace NetTiming.Guarantees

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

section Sampled

open Claims.Sampling
open Claims.SamplingGeneric (Arg Compiled)

/-- **An accepted sampled audit bounds the leak with the checked inputs left out.** At every compiled sampled verifier
`D` where `Claims.SamplingGeneric.Compiled.KS` holds, a public layout `lay` and horizon `k` of each statement, `δ ≥ 0`,
registered inputs `reg`, checked inputs `chk` and each statement's check `valid`:

* on the statements in scope whose trace run the check holds on except with probability `εchk` (`CheckedWitness`),
  `Pr[acc ∧ ¬RLeakChecked δ] ≤ δ + ε_ks + ε_bind + εchk`: except with that probability, every observation sequence of
  length `k` with jitter below `Dj` that the trace's transfer gates carry with every status complete is among the
  sequences `AccSeqsChecked` counts at the registration's values;
* on a statement in scope whose check pins `chk` (`CheckedUnique`), at every `ρ`, those sequences number at most
  `2^#(freeInputs \ (reg ∪ chk)) · #faultySets(δ) · 2^β · Dj^k`, `β` the most committed bits of a possible incorrect
  set. -/
def SampledLeakChecked : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (εchk : Claims.Msg → ℝ)
    (Dj : ℕ),
    KSLinOn H D.V (fun x => ScopeAcc D lay k x ∧ CheckedWitness H D valid εchk x)
        (RLeakChecked H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) valid
          Dj)
        D.E (fun x => D.κ δ x + εchk x) D.γ (fun _ => 1) (Bext H D.V D.E) ∧
      ∀ x, ScopeAcc D lay k x → ∀ s P, D.parse x = some s → D.partition s = some P →
        CheckedUnique s.prog P s.design (lay s) (k s) (chk s) (valid s) →
        ∀ (op : ℕ → List (Obs L K S) → List Bool) (ρ : ℕ → Bool),
          (AccSeqsChecked s.prog P s.design (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) (valid s) δ op
              Dj).encard ≤
            ((2 ^ (freeInputs s.prog (lay s) (k s) \ (reg s ∪ chk s)).card * (faultySets s.design P δ).card *
              2 ^ ((faultySets s.design P δ).sup (faultyBits s.prog P)) * Dj ^ k s : ℕ) : ℕ∞)

end Sampled

section Pooled

open Claims Claims.Sampling Finset

/-- **The leak through the network certifier over a monitoring period, with the checked inputs left out** (conditional on
its named inputs). Over `m` windows on independent live coins, window `i` registered as `sw i` (partition `Pw i`, values
`ρ i` at the registered inputs, operator `op i`), with `n i` proof blocks, each window's check pinning its checked inputs
(`CheckedUnique`); with the network certifier running its code by `trusted-host`, its one run giving the cells the client
receives and its record of them, the windows' observations; a covert sender's channel (`PeriodChannel`), each window's
acceptance and incorrect set reading only the earlier windows' coins (`PeriodAdaptive`), the conditional escape
(`PeriodEscapes`) and the leak relation with a valid run at each window's own incorrect set (`PeriodLeaksChecked`):

* for every secret, every audit accepts and the good event fails (`PooledGoodChecked`) with probability at most
  `(1 - q)^Fmax + Σ e + Σ e'`;
* at every sample of the coins, the secrets that meet the good event number at most
  `#Ss · ∏_c (cap c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ (reg_i ∪ chk_i)) · Dj^k_i`,
  `N = Σ_i n i` and `b` the most committed bits of one of the period's proof blocks. -/
def PooledLeakChecked : Prop :=
  ∀ {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg chk : AuditStatement → Finset ℕ)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) {m : ℕ} (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool) (n : Fin m → ℕ)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (Zs : Set Z)
    (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : PeriodChannel f cap Ss Zs schedOf srvOf dec fun p => (run tap p).1)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop) (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (had : PeriodAdaptive n acc W) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i) (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : PeriodEscapes Zs acc W q e)
    (hleak : PeriodLeaksChecked lay k reg valid Dj sw Pw ρ op Zs acc W
      (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) e')
    (Fmax : ℕ),
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬PooledGoodChecked lay k reg valid Dj sw Pw ρ op Fmax (W z ω) (run tap (schedOf z ω, srvOf z ω)).2) ≤
        (1 - q) ^ Fmax + ∑ i, e i + ∑ i, e' i) ∧
      ∀ ω, {z | z ∈ Zs ∧
          PooledGoodChecked lay k reg valid Dj sw Pw ρ op Fmax (W z ω) (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax *
          2 ^ ((univ.sup fun i => (range (n i)).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
          ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card *
            Dj ^ k (sw i)) : ℕ) : ℕ∞)

end Pooled

end NetTiming.Guarantees
