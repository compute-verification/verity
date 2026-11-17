module

public import Proofs.WeightSecurity.TwoReg

/-! A period of two-stage verification phases with one failure probability for the whole period. -/

@[expose] public section

namespace Claims.Sampling

def BestInteriorsAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) (val : ℕ → Option Bool)
    (ι : ℕ → ℕ → Bool) : Prop :=
  ∀ ρ < d.part.blocks.length, ∀ ι' : ℕ → ℕ → Bool,
    d.esc Pt (footprintAt s Pt d.part val ι') ρ ≤ d.esc Pt (footprintAt s Pt d.part val ι) ρ

/-- Every best interiors give one value: `P̄_a` reads a footprint through its replay blocks' escapes only. -/
noncomputable def Design.Two.PaBestAt (d : Design.Two) (s : AuditStatement) (Pt : Partition)
    (val : ℕ → Option Bool) : ℝ :=
  d.Pa Pt (footprintAt s Pt d.part val (Classical.epsilon (BestInteriorsAt s Pt d val)))

end Claims.Sampling

namespace Claims.SamplingGeneric.CompiledTwo

open Claims
open Claims.Sampling (incorrectAllAt)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

noncomputable def FregTwoAt (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (val : ℕ → Option Bool) (r : Fin (L.V.k x) → Coin) : Finset ℕ :=
  match L.parse x with
  | none => ∅
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d => incorrectAllAt s Pt d.part val (L.iregVal H P x ω r)
    | _, _ => ∅

end Claims.SamplingGeneric.CompiledTwo

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game hiding Msg
open scoped ENNReal

structure ServedEsc (O V Tick Adv : Type) extends Served O V Tick Adv where
  missProb : ℝ≥0∞

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

namespace ServedEsc

open Classical in
noncomputable def leadMissProb : List (ServedEsc O V Tick Adv) → ℝ≥0∞
  | [] => 1
  | w :: l => if w.sampledIncorrect then 1 else w.missProb * leadMissProb l

def UndetectedBelow (δ : ℝ) (l : List (ServedEsc O V Tick Adv)) : Prop :=
  (∃ w ∈ l, w.accepted ∧ w.sampledIncorrect) ∨ leadMissProb l < ENNReal.ofReal δ

end ServedEsc

noncomputable def acceptedSampledAt (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (P : Prover Unit)
    (val : ℕ → Option Bool) : ℝ :=
  Pr fun r : Fin (L.V.k x) → Coin =>
    (run H P L.V x () r).out = .acc ∧ ∃ u ∈ L.checked x r, u ∈ L.FregTwoAt H P x () val r

open Classical in
noncomputable def regCountAt (L : CompiledTwo Coin A) (x : Msg) (val : ℕ → Option Bool) : ℕ :=
  match L.parse x with
  | none => 0
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d => if InScopeTwo Pt d s then (incorrectAtBoundaryAt s Pt d.part val).card else 0
    | _, _ => 0

open Classical in
/-- It reads the statement and `val` only, and no coin of the run. -/
noncomputable def escAt (L : CompiledTwo Coin A) (x : Msg) (val : ℕ → Option Bool) : ℝ≥0∞ :=
  match L.parse x with
  | none => 1
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d => if InScopeTwo Pt d s then ENNReal.ofReal (d.PaBestAt s Pt val) else 1
    | _, _ => 1

noncomputable def twoAtRec (H : Msg → Msg) (L : CompiledTwo Coin A) (val : ℕ → Option Bool) (n : ℕ) (e : ℝ≥0∞)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) : ServedEsc O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ∃ u ∈ L.checked b.2.1 b.2.2.2, u ∈ L.FregTwoAt H b.2.2.1 b.2.1 () val b.2.2.2
  incorrect := n
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2
  missProb := e

noncomputable def twoAtRun (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) (T : ℕ)
    (s : Strategy (monitoringPeriodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (ServedEsc O V Tick Adv)) :=
  monitoringPeriodRun (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoAtRec H (Lv t) (val t τ.1 τ.2.1.1 τ.2.1.2) (regCountAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))
      (escAt (Lv t) τ.2.1.1 (val t τ.1 τ.2.1.1 τ.2.1.2))) 0 T s

/-- The sampling term is `δ` however many verification phases the period has. -/
def MonitoringPeriodTwoProd (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (monitoringPeriodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, MonitoringPeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ → (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t) →
    ∀ δ : ℝ, Verity.Game.value
        (fun l : List (ServedEsc O V Tick Adv) =>
          (∀ r ∈ l, r.accepted) ∧ (l.map ServedEsc.missProb).prod < ENNReal.ofReal δ)
        (twoAtRun H Lv val T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t)

def MonitoringPeriodTwoLead (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (monitoringPeriodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, MonitoringPeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ → (Lv t).Scope τ.2.1.1 →
      acceptedSampledAt H (Lv t) τ.2.1.1 τ.2.1.2 (val t τ.1 τ.2.1.1 τ.2.1.2) ≤ ε t) →
    ∀ δ : ℝ, Verity.Game.value (ServedEsc.UndetectedBelow δ) (twoAtRun H Lv val T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t)

/-- It takes no knowledge soundness of the argument: a caller meets their hypothesis with its own bound on the event
at the trace it reads. -/
def CompilerMonitoringPeriodTwoProd : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (O V Tick Adv : Type)
    (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A)
    (val : (t : ℕ) → (O → V) × Tick × Adv → Msg → Prover Unit → ℕ → Option Bool),
    (∀ t, (Lv t).Sessions H) →
    MonitoringPeriodTwoLead (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv val ∧
      MonitoringPeriodTwoProd (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv val

end WeightSecurity.SecurityProofs
