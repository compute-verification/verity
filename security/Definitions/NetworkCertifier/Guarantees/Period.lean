module

public import Definitions.Core.Assumptions
public import Definitions.NetworkCertifier.Guarantees.SampledCharge
public import Definitions.NetworkCertifier.Period
public import Definitions.NetworkCertifier.Tap
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
# The leak through the network certifier over a monitoring period of sampled audits

A monitoring period is `n` windows, each audited by the sampled verifier at a statement it registered (`sw i`, with its
partition `Pw i` and the values `ρ i` the registration commits at its registered inputs). The network certifier is one
device on the server's uplink, and it functions as an active network certifier (`Definitions.NetworkCertifier.Tap`): one
run of its code gives the cells the client receives and the observations the audits read, its record of those cells, so
on every schedule its record format holds the client's view reads back from the observations. The network certifier
registers its records at the transfer gates: its openings `tops s` under the statement's roots, its signed root for the
window, so they vary with what the server did
(`CertifierRegisters`). `PeriodLeakAccepted` composes the sampled leak bound at the network certifier's rows (`RLeakCertifier`, with the
prover's registered inputs pinned: `AsReg`) with that read-back, so the content term of the bound comes from the
audits, not from a premise:

* each window's audit, at a statement in its scope, fails the leak relation at the network certifier's rows with
  probability at most `δ + ε_ks + ε_bind + Pr[B_tap]` at that statement (`Claims.SamplingGeneric.Compiled.KS`'s errors,
  and `B_tap` the collisions between the extracted trace's openings and the network certifier's, under
  `CommitmentScheme.Binds`);
* where the network certifier registers its observations, the secrets of a covert sender in the server whose every
  window meets its good event (`WindowGood`) number at most
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`: the sender moves at most
  `log2 #Ss + Σ_c log2 (cap_c + 1) + Σ_i (#(freeInputs_i \ reg_i) + log2 #faultySets_i(δ) + β_i + k_i·log2 Dj)` bits
  through the network certifier in the monitoring period.

The union of the windows' failures over the monitoring period's live coins is not stated here. `WindowGood` is the
event the first part bounds, plus one condition no proof here gives: the registration's values at the prover's
registered inputs, which the registration's commitment binds. The network certifier's rows are not pinned: they vary
with the secret. The first part covers an extracted trace that leaves them outside the incorrect proof blocks, where every
transfer gate is opened; the count charges an incorrect proof block's rows through the Program, from the free inputs, the
incorrect sets, their committed bits and the timing. A proof block the sample missed is in the incorrect set.
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of accepted sampled audits.** At every compiled
sampled verifier `D` where `Claims.SamplingGeneric.Compiled.KS` holds and the trace's commitment binds
(`CommitmentScheme.Binds`), over a monitoring period of `n`
windows registered as `sw i` (partition `Pw i`, registered values `ρ i`, and `β i` bounding the committed bits of each
of its possible incorrect sets), with an operator `op` fixed before the period and the covert sender's secrets, a
function of the observations alone: window `i`'s operator at the period's observations `o` is `op i (runsBefore o i)`,
which reads only the windows before `i`, closed by then (`runsBefore`), and window `i` only through the observations
before each step, so no window still open and none after; with the network certifier's openings `tops s` at each
statement,
and a network certifier that runs its code by `trusted-host` on schedules in `Ss`, each held by its record format `f`
(`RecordFormat.fits`) and giving connection `c` at most `cap c` cells, its one run giving the cells the client receives
and the observations of the monitoring period's windows (its record of those cells, in the format `f`):

* each window's audit, at a statement in its scope, has `Pr[acc ∧ ¬RLeakCertifier δ] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`;
* where at every window's statement the network certifier registers its observations at that window's operator at the
  run's earlier windows (`CertifierRegisters`), a covert sender in the server, whose secrets a decoder recovers from the
  client's view of the server side: its secrets whose every window meets `WindowGood`, at that window's statement
  `x z i` and extracted witness `wit z i`, number at most
  `#Ss · ∏_c (cap c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`. -/
def PeriodLeakAccepted : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (hPw : ∀ i, D.partition (sw i) = some (Pw i))
    (ρ : Fin n → ℕ → Bool) (op : Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ)
    (hβ : ∀ i, ∀ F ∈ faultySets (sw i).design (Pw i) δ, faultyBits (sw i).prog (Pw i) F ≤ β i)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool)) (hb : D.commit.Binds H),
    (∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakCertifier H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
          D.E m P ≤
        D.κ δ m + Real.sqrt (D.γ m * Claims.PrB H D.V P m) + (Bext H D.V D.E Ω P m).Pr H +
          (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H) ∧
      ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
          (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s) ((run tap (schedOf z, srvOf z)).2 i)) →
        {z | z ∈ Zs ∧ ∀ i, WindowGood H D L K S lay k δ reg Dj (sw i) (ρ i) tops (x z i) (wit z i)
            ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
          Ss.encard * (((∏ c, (cap c + 1)) * ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞))

end NetTiming.Guarantees
