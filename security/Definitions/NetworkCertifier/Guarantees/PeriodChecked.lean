module

public import Definitions.Core.Assumptions
public import Definitions.NetworkCertifier.Assumptions.RegisteredInputs
public import Definitions.NetworkCertifier.Carried
public import Definitions.NetworkCertifier.Guarantees.SampledCharge
public import Definitions.NetworkCertifier.PeriodChecked
public import Definitions.NetworkCertifier.PeriodConditions
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
# The leak through the network certifier over a monitoring period, with the checked inputs left out

`PeriodLeakAccepted` with each window's checked inputs `chk` left out of the count
(`Definitions.NetworkCertifier.Checked`), and the client's view read back from the windows' records under a registered
parameter (`RecordFormat.ReadsBy`) in place of the whole schedule.

**The check.** Each statement has two checks on its run, `valid₁` and `valid₂`, each with its own miss bound; together
they pin the checked inputs (`CheckedUnique` at their conjunction). The run they are held to is the one the leak
relation at the network certifier's rows accepts at (`certifierRun`): the trace's inputs, the ingress the operator's
moves on the network certifier's records give, and an incorrect proof block's committed gates held at the trace filled
in with the network certifier's openings. `CheckedWitness` at that run is the premise. At the trace's run (`traceRun`)
it would not carry the bound: the two runs differ at a gate the network certifier opens and the trace leaves unopened.

**The count.** With the record read back under a parameter of the registered set `Θ`, a secret whose every window
meets its good event is recovered from one parameter and the windows' records, each a sequence its window's Program
accepts with a valid run. So the secrets number at most `#Θ` times the windows' count (`windowsCount`) with the
registered and the checked inputs pinned; no factor counts the schedules or a connection's cells.

**The registered inputs.** `reg` is as `PeriodLeakAccepted` has it. The good event (`WindowGoodChecked`, stated here:
`WindowGood` with the leak relation at a valid run and the window's check the registration's) reads that the trace
holds the registration's values there as the named premise `RegisteredInputsHeld`, open: the registration's openings
discharge it under the commitment's binding. Where no input is registered it holds of every trace.

The union of the windows' failures over the monitoring period's live coins is not stated here.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

section Period

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **A window's good event with a valid run**: at the window's statement `x` and the audit's extracted witness `w`,
the leak relation at the network certifier's rows holds with a valid run (the network certifier's openings `tops s`),
`x` is in the audit's scope and registered as `s₀`, with `s₀`'s check, the trace holds `ρ` at the registered inputs
(`RegisteredInputsHeld`), and the run's observations `o` have the horizon's length, jitter below `Dj` and every status
complete. -/
def WindowGoodChecked (H : Msg → Msg) (D : Compiled Coin A) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (Dj : ℕ) (s₀ : AuditStatement) (ρ : ℕ → Bool) (tops : AuditStatement → List (Opening Bool)) (x : Msg) (w : Wit)
    (o : List (Obs L K S)) : Prop :=
  RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s))
      valid Dj tops x w ∧
    ScopeAcc D lay k x ∧
    ∃ s, D.parse x = some s ∧ SameRegistration D lay k reg s₀ s ∧ valid s = valid s₀ ∧
      RegisteredInputsHeld H D.commit (reg s) ρ s w ∧ Covered (k s) o ∧ o.length = k s ∧ ∀ y ∈ o, y.jitter < Dj

end Period

end NetTiming

namespace NetTiming.Guarantees

open Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of accepted sampled audits, with the checked
inputs left out.** At every compiled sampled verifier `D` where `Claims.SamplingGeneric.Compiled.KS` holds and the
trace's commitment binds (`CommitmentScheme.Binds`), over a monitoring period of `n` windows registered as `sw i`
(`WindowsPlaced`: partition `Pw i`, and `β i` bounding the committed bits of each of its possible incorrect sets; its
registered values `ρ i`), each window's two checks together pinning its checked inputs (`CheckedUnique`); with an
operator `op` fixed before the period and the covert sender's secrets, window `i`'s reading only the windows before
`i`, closed by then (`runsBefore`); with the network certifier's openings `tops s` at each statement, and a network
certifier that runs its code by `trusted-host`, its one run giving the cells the client receives and the observations
of the monitoring period's windows (its record of those cells, in the format `f`), which reads back under a registered
parameter (`RecordFormat.ReadsBy`); and a covert sender in the server over the registered parameters `Θ`, whose secrets
a decoder recovers from the client's view of the server side (`PeriodChannelReg`):

* each window's audit, at a statement in its scope where the two checks miss on the run at the network certifier's rows
  with probability at most `ε₁` and `ε₂` (`CheckedWitness` at `certifierRun`), has
  `Pr[acc ∧ ¬RLeakCertifierChecked δ] ≤ δ + ε_ks + ε_bind + ε₁ + ε₂ + Pr[B_tap]`;
* where at every window's statement the network certifier registers its observations at that window's operator at the
  run's observations (`CertifierRegisters`), for the secrets whose every window meets `WindowGoodChecked`, at that
  window's statement `x z i` and extracted witness `wit z i`:
  * **held**: the cells the client receives are read from the windows' records under a registered parameter of `Θ`,
    and each window's record is a sequence its registered Program accepts with a valid run at the registration's
    values (`AccSeqsChecked`);
  * **counted**: they number at most
    `#Θ · ∏_i 2^#(freeInputs_i \ (reg_i ∪ chk_i)) · #faultySets_i(δ) · 2^β_i · Dj^k_i` (`windowsCount`). -/
def PeriodLeakChecked : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg chk : AuditStatement → Finset ℕ)
    (valid₁ valid₂ : AuditStatement → (ℕ → Bool) → Prop) (ε₁ ε₂ : Claims.Msg → ℝ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D.partition δ sw Pw β)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r))
    (ρ : Fin n → ℕ → Bool) (op : Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    {Conn Cell Srv Dev Z Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin n → List (Obs L K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows)
    (Θ : Set Par) (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : PeriodChannelReg f allows Θ Zs schedOf srvOf dec fun p => (run tap p).1)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (hb : D.commit.Binds H),
    (∀ m, ScopeAcc D lay k m →
        CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₁ ε₁ m →
        CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₂ ε₂ m →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ
            (fun s => AsReg s.prog (lay s) (k s) (reg s)) (fun s r => valid₁ s r ∧ valid₂ s r) Dj tops)
          D.E m P ≤
        D.κ δ m + ε₁ m + ε₂ m + Real.sqrt (D.γ m * Claims.PrB H D.V P m) + (Bext H D.V D.E Ω P m).Pr H +
          (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H) ∧
      ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
          (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s) ((run tap (schedOf z, srvOf z)).2 i)) →
        (∀ z ∈ Zs, (∀ i, WindowGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
              (ρ i) tops (x z i) (wit z i) ((run tap (schedOf z, srvOf z)).2 i)) →
            ∃ θ ∈ Θ, (run tap (schedOf z, srvOf z)).1 = rd θ (run tap (schedOf z, srvOf z)).2 ∧
              ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
                AccSeqsChecked (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
                  (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i)) (ρ i))
                  (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r) δ
                  (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) Dj) ∧
          {z | z ∈ Zs ∧ ∀ i, WindowGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
              (ρ i) tops (x z i) (wit z i) ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
            Θ.encard * ((windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞))

end NetTiming.Guarantees
