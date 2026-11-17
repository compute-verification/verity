module

public import Definitions.Certifier.PooledLeak
public import Definitions.Core.Assumptions
public import Definitions.Certifier.Tap
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

/-! The leak through the network attester over a monitoring period, pooled: conditional on two inputs per window that
nothing discharges, so it bounds no deployed system. -/

@[expose] public section

namespace NetTiming.Guarantees

open Claims Claims.Sampling Finset

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network attester over a monitoring period of accepted sampled audits** (conditional on its
two per-window inputs). Over `m` windows on independent live coins, window `i` registered as `sw i` (partition `Pw i`,
values `ρ i` at the registered inputs), with `n i` proof blocks, each committing at most `b` bits; with an operator
`op` fixed before the period and the covert sender's secrets, a function of the observations alone: window `i`'s
operator at the period's observations `o` is `op i (runsBefore o i)`, which reads only the windows before `i`, closed
by then (`runsBefore`), and window `i` only through the observations before each step, so no window still open and
none after; with
the network attester running its code by `trusted-device` on schedules in `Ss`, each held by its record format `f`
(`RecordFormat.fits`) and giving connection `c` at most `cap c` cells, its one run giving the cells the client receives
and its record of them in the format `f`,
the windows' observations; and a covert sender in the server whose secrets `z ∈ Zs` a decoder recovers from the
client's cells at the period's coins. For each secret, `acc z i ω x` is window `i`'s audit accepting at its coins `x`
after the earlier windows' coins `ω`, and `W z ω i ⊆ range (n i)` is the incorrect set of the witness computed from window
`i`'s registration; both read only the windows before `i`. Given

* `hesc`, the conditional escape: `Pr_x[acc z i ω x] ≤ (1 - q)^|W z ω i| + e i`, `q` the design's per-window
  detection probability (`Flock.Guarantees.SampledKSLive` at the history, not yet discharged);
* `hleak`, the leak relation at that witness: `Pr[acc_i ∧ ¬LeakAt (W z ω i)] ≤ e' i` on the record's observations,
  at the window's operator at the record's earlier windows (the live `SampledLeakAccepted`, P2, not yet discharged);

for `0 ≤ q ≤ 1` and `Fmax ≥ ln(1/δ)/q`:

* for every secret, every audit accepts and the monitoring period's good event fails (`PooledGood`: the incorrect sets
  hold under `Fmax` units and every window meets its leak relation at its own) with probability at most
  `δ + Σ e + Σ e'`;
* at every sample of the coins, the secrets that meet the good event number at most
  `#Ss · ∏_c (cap c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ reg_i) · Dj^k_i`, `N = Σ_i n i`. -/
def PooledLeakAccepted : Prop :=
  ∀ {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m : ℕ}
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    (n : Fin m → ℕ) (b : ℕ) (hb : ∀ i, ∀ v < n i, faultyBits (sw i).prog (Pw i) {v} ≤ b)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn) (srvOf : Z → (Fin m → Ω) → Srv)
    (hs : ∀ z ∈ Zs, ∀ ω, schedOf z ω ∈ Ss) (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (hdz : ∀ z ∈ Zs, ∀ ω, dec ω (run tap (schedOf z ω, srvOf z ω)).1 = z)
    (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (hacc : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → acc z i ω = acc z i ω')
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ)
    (hW : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → W z ω i = W z ω' i)
    (hWn : ∀ z ω i, W z ω i ⊆ range (n i)) (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : ∀ z ∈ Zs, ∀ i ω, Pr (acc z i ω) ≤ (1 - q) ^ (W z ω i).card + e i)
    (hleak : ∀ z ∈ Zs, ∀ i, Pr (fun ω => acc z i ω (ω i) ∧
      ¬LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (run tap (schedOf z ω, srvOf z ω)).2 i)) (W z ω i)
        ((run tap (schedOf z ω, srvOf z ω)).2 i)) ≤ e' i)
    (δ : ℝ) (hδ : 0 < δ) (Fmax : ℕ) (hF : Real.log (1 / δ) ≤ q * Fmax),
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, acc z i ω (ω i)) ∧
        ¬PooledGood lay k reg Dj sw Pw ρ op Fmax (W z ω) (run tap (schedOf z ω, srvOf z ω)).2) ≤
        δ + ∑ i, e i + ∑ i, e' i) ∧
      ∀ ω, {z | z ∈ Zs ∧ PooledGood lay k reg Dj sw Pw ρ op Fmax (W z ω) (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
        Ss.encard * (((∏ c, (cap c + 1)) * ((∑ i, n i) + 1) ^ Fmax * 2 ^ (b * Fmax) *
          ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ reg (sw i)).card * Dj ^ k (sw i)) : ℕ) : ℕ∞)

end NetTiming.Guarantees
