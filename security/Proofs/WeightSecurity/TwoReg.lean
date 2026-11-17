module

public import Definitions.Claims.SamplingTwoReg
public import Proofs.WeightSecurity.Two

/-! A period of two-stage windows at the committed traces. -/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game hiding Msg
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

open Classical in
noncomputable def regCount (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (P : Prover Unit) : ℕ :=
  match L.parse x with
  | none => 0
  | some s =>
    match L.partition s, L.two s with
    | some Pt, some d =>
      if InScopeTwo Pt d s then (incorrectAtBoundaryAt s Pt d.part (L.regVal H P x ())).card else 0
    | _, _ => 0

noncomputable def twoRegRec (H : Msg → Msg) (L : CompiledTwo Coin A) (n : ℕ)
    (b : ((O → V) × Tick × Adv) × TwoPlay L) : Served O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ∃ u ∈ L.checked b.2.1 b.2.2.2, u ∈ L.FregTwo H b.2.2.1 b.2.1 () b.2.2.2
  incorrect := n
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2

noncomputable def twoRegRun (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (T : ℕ)
    (s : Strategy (monitoringPeriodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (Served O V Tick Adv)) :=
  monitoringPeriodRun (fun t => twoGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => twoRegRec H (Lv t) (regCount H (Lv t) τ.2.1.1 τ.2.1.2)) 0 T s

def MonitoringPeriodTwoReg (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (r₁ r₂ : ℝ) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (monitoringPeriodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, MonitoringPeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      (Lv t).εregTwo H τ.2.1.1 τ.2.1.2 ≤ ε t) →
    ∀ M : ℕ, Verity.Game.value (Served.Undetected M) (twoRegRun H Lv T s) ≤
      ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) + ENNReal.ofReal ((1 - r₁ * r₂) ^ M)

def CompilerMonitoringPeriodTwoReg : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (O V Tick Adv : Type)
    (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (r₁ r₂ : ℝ), A.KS H →
    (∀ t, (Lv t).commit.Binds H ∧ (Lv t).Sessions H) → 0 ≤ r₁ → r₁ ≤ 1 → 0 ≤ r₂ → r₂ ≤ 1 →
    (∀ t x s Pt d, (Lv t).parse x = some s → (Lv t).partition s = some Pt → (Lv t).two s = some d →
      InScopeTwo Pt d s → d.FirstRate r₁ ∧ d.SecondRate Pt r₂) →
    MonitoringPeriodTwoReg (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv r₁ r₂

end WeightSecurity.SecurityProofs
