module

public import Definitions.NetworkCertifier.SampledCharge
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
# Checked inputs: free inputs a check on the whole run pins

The leak bounds charge every free input outside the registered ones (`SampledLeakAccepted`, `PooledLeakAccepted`). A
statement's *checked inputs* `chk` are free inputs whose values a check on the run fixes, given everything else the run
reads: a memory-checked pool's reads, each the latest write before it, say. They cost no advice bits once the check holds
on the whole run, since the run's other values then fix every egress row.

**The run** is what `C` computes over the whole horizon `k` on the ingress the operator's moves give at each step before
`k`, the inputs `a` and the values `free` held at the incorrect proof blocks' committed gates (`valOver`). An incorrect
proof block's committed values reach the check (they feed what the checked inputs are checked against), so the check is a
property `valid` of the whole run, not of the inputs alone.

* `AcceptedChecked`, `AccSeqsChecked` and `RLeakChecked` are `AcceptedFaulty`, `AccSeqsSampled` and `RLeak` with a `valid`
  run; `LeakAtChecked` and `PooledGoodChecked` are `LeakAt` and `PooledGood` with it, the incorrect set holding no proof
  block of a class sampled `all`.
* `CheckedUnique`: the check pins `chk`. Two `valid` runs whose ingress agrees through step `t`, whose inputs agree off
  `chk`, and which hold one set of values, only at committed gates of proof blocks of no class sampled `all`, give step `t`
  one egress row. The check's own proof blocks are sampled `all`, so no incorrect set holds their gates.
* `CheckedWitness`, a scope: an accepted run whose extracted witness has no proof block sampled `all` incorrect has a
  `valid` trace run, the incorrect proof blocks' committed gates held at the trace's values (`heldAdv`), except with
  probability `εchk` (`PrCheckMiss`).
* `PeriodChannel`, `PeriodAdaptive`, `PeriodEscapes` and `PeriodLeaksChecked`: a monitoring period's inputs, as
  `PooledLeakAccepted` takes them, the leak relation at `LeakAtChecked`.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- **Accepted with the incorrect set `F` and a valid run**: some inputs `a ∈ As` and values `free` held only at `F`'s
committed gates make the first `k` steps `Accepted` by the circuit's Program, and the run over the horizon on the ingress
the operator's moves give is `valid`. -/
def AcceptedChecked {L K S : ℕ} (C : Prog) (P : Partition) (lay : BoundaryGates L S) (k : ℕ) (As : Set (ℕ → Bool))
    (valid : (ℕ → Bool) → Prop) (F : Finset ℕ) (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)) :
    Prop :=
  ∃ a ∈ As, ∃ free : ℕ → Option Bool, (∀ g b, free g = some b → FaultyGate C P F g) ∧
    valid (valOver C (lay.inputOf k ((List.range k).map fun i => op i (o.take i)) a) free) ∧
    Accepted (circuitProgram C lay k) op k o (a, free)

/-- **The sequences a sampled audit accepts at `δ` with a valid run**: `AccSeqsSampled` with `AcceptedChecked`. -/
def AccSeqsChecked {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (As : Set (ℕ → Bool)) (valid : (ℕ → Bool) → Prop) (δ : ℝ) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) :
    Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ ∃ F ∈ faultySets d P δ, AcceptedChecked C P lay n As valid F op o}

/-- **The check pins the checked inputs `chk`**: at each step `t < n`, two runs over the horizon `n` whose ingress agrees
through step `t`, whose inputs agree off `chk`, and which hold one set of values, only at committed gates of proof blocks
of no class sampled `all`, both `valid`, give step `t` one egress row. -/
def CheckedUnique {L S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : BoundaryGates L S) (n : ℕ)
    (chk : Finset ℕ) (valid : (ℕ → Bool) → Prop) : Prop :=
  ∀ t < n, ∀ (ins ins' : List (List Bool)) (a a' : ℕ → Bool) (free : ℕ → Option Bool),
    ins.length = n → ins'.length = n → (∀ j ≤ t, ins.getD j [] = ins'.getD j []) → (∀ g ∉ chk, a g = a' g) →
    (∀ g b, free g = some b → ∃ v ∉ allSlots d P, FaultyGate C P {v} g) →
    valid (valOver C (lay.inputOf n ins a) free) → valid (valOver C (lay.inputOf n ins' a') free) →
    ∀ ℓ, rowOf (valOver C (lay.inputOf n ins a) free) (lay.egress t ℓ) =
      rowOf (valOver C (lay.inputOf n ins' a') free) (lay.egress t ℓ)

/-- **The leak relation with a valid run** at `δ`: `RLeak`, with the sequences `AccSeqsChecked` counts at each statement's
check `valid s`. -/
def RLeakChecked (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool))
    (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧
    ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
      Carries (lay s) s.prog (w.val H Cm s) op (k s) o → Covered (k s) o → o.length = k s → (∀ y ∈ o, y.jitter < Dj) →
        o ∈ AccSeqsChecked s.prog P s.design (lay s) (k s) (As s (traceVal s.prog (w.val H Cm s))) (valid s) δ op Dj

section Claim

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The check's miss**: `Pr[run = acc ∧ E^{P*}(x) = w]` at a witness `w` whose statement parses and has a partition,
whose incorrect set holds no proof block of a class sampled `all`, and whose trace run (`C` on the trace's committed
values, the incorrect proof blocks' committed gates held at them, `heldAdv`) is not `valid`. -/
noncomputable def PrCheckMiss (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) : ℝ :=
  Pr fun r : Ω × (Fin (D.V.k x) → Coin) × (E D decode).C x =>
    (run H P D.V x r.1 r.2.1).out = .acc ∧ ∃ w s Pt, ((E D decode).eval H P x r.1 r.2.1 r.2.2).1 = some w ∧
      D.parse x = some s ∧ D.partition s = some Pt ∧ Disjoint (faulty H D.commit s Pt w) (allSlots s.design Pt) ∧
      ¬valid s (valOver s.prog (traceVal s.prog (w.val H D.commit s))
        (heldAdv s.prog Pt (faulty H D.commit s Pt w) (w.val H D.commit s)))

/-- **The check holds on the trace's run at `x`**, a scope: at every prover, the check misses with probability at most
`εchk x` (`PrCheckMiss`). -/
def CheckedWitness (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (valid : AuditStatement → (ℕ → Bool) → Prop) (εchk : Msg → ℝ) (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), PrCheckMiss H D decode valid x P ≤ εchk x

end Claim

/-- **A window's leak relation at the incorrect set `F` with a valid run**: the layout fits, `F` holds no proof block of a
class sampled `all`, and the observations, of length `k s` with jitter below `Dj`, are accepted with a valid run at the
inputs pinned at the registered ones (`AcceptedChecked` at `AsReg`). -/
def LeakAtChecked {L K S : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) (s : AuditStatement)
    (P : Partition) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool) (F : Finset ℕ) (o : List (Obs L K S)) :
    Prop :=
  (lay s).Fits s.prog (k s) ∧ Disjoint F (allSlots s.design P) ∧ o.length = k s ∧ (∀ y ∈ o, y.jitter < Dj) ∧
    AcceptedChecked s.prog P (lay s) (k s) (AsReg s.prog (lay s) (k s) (reg s) ρ) (valid s) F op o

/-- **A monitoring period's good event with valid runs**: the incorrect sets `W i` hold fewer than `Fmax` units in total,
and every window's observations meet its leak relation at its own set (`LeakAtChecked`). -/
def PooledGoodChecked {L K S m : ℕ} (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) (sw : Fin m → AuditStatement)
    (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool) (op : Fin m → ℕ → List (Obs L K S) → List Bool) (Fmax : ℕ)
    (W : Fin m → Finset ℕ) (o : Fin m → List (Obs L K S)) : Prop :=
  ∑ i, (W i).card < Fmax ∧ ∀ i, LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i) (W i) (o i)

/-- **The covert sender's channel over a monitoring period**: the schedules in `Ss` give connection `c` at most `cap c`
cells and the record format holds each (`RecordFormat.fits`); each secret `z ∈ Zs` runs on schedules in `Ss` and is
decoded from the cells the client receives (`cells`). -/
def PeriodChannel {m : ℕ} {Ω Conn Cell Srv Ob Z : Type} (f : RecordFormat Conn Cell Ob) (cap : Conn → ℕ)
    (Ss : Set (CertifierSchedule Conn)) (Zs : Set Z) (schedOf : Z → (Fin m → Ω) → CertifierSchedule Conn)
    (srvOf : Z → (Fin m → Ω) → Srv) (dec : (Fin m → Ω) → (Conn → List (ℕ × CellView Cell)) → Z)
    (cells : CertifierSchedule Conn × Srv → Conn → List (ℕ × CellView Cell)) : Prop :=
  (∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c) ∧ (∀ s ∈ Ss, f.fits s) ∧ (∀ z ∈ Zs, ∀ ω, schedOf z ω ∈ Ss) ∧
    ∀ z ∈ Zs, ∀ ω, dec ω (cells (schedOf z ω, srvOf z ω)) = z

/-- **Each window reads only the earlier windows' coins**: window `i`'s acceptance event `acc z i ω` and incorrect set
`W z ω i` are one at any two histories that agree before `i`, and the set is among the window's `n i` proof blocks. -/
def PeriodAdaptive {m : ℕ} {Ω Z : Type} (n : Fin m → ℕ) (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ) : Prop :=
  (∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → acc z i ω = acc z i ω') ∧
    (∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → W z ω i = W z ω' i) ∧ ∀ z ω i, W z ω i ⊆ Finset.range (n i)

/-- **The conditional escape**: given the earlier windows' coins, window `i`'s audit accepts with probability at most
`(1 - q)^|W_i| + e_i`, `q` the design's per-window detection probability. -/
def PeriodEscapes {m : ℕ} {Ω Z : Type} [Fintype Ω] (Zs : Set Z) (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ) (q : ℝ) (e : Fin m → ℝ) : Prop :=
  ∀ z ∈ Zs, ∀ i ω, Pr (acc z i ω) ≤ (1 - q) ^ (W z ω i).card + e i

/-- **The leak relation with a valid run at each window's own incorrect set**: window `i`'s audit accepts while its
observations `obs z ω i` are not accepted with a valid run at `W z ω i` (`LeakAtChecked`) with probability at most
`e' i`. -/
def PeriodLeaksChecked {L K S m : ℕ} {Ω Z : Type} [Fintype Ω] (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ)
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → ℕ → List (Obs L K S) → List Bool) (Zs : Set Z) (acc : Z → Fin m → (Fin m → Ω) → Ω → Prop)
    (W : Z → (Fin m → Ω) → Fin m → Finset ℕ) (obs : Z → (Fin m → Ω) → Fin m → List (Obs L K S)) (e' : Fin m → ℝ) :
    Prop :=
  ∀ z ∈ Zs, ∀ i, Pr (fun ω => acc z i ω (ω i) ∧
    ¬LeakAtChecked lay k reg valid Dj (sw i) (Pw i) (ρ i) (op i) (W z ω i) (obs z ω i)) ≤ e' i

end NetTiming
