module

public import Definitions.Claims.SamplingGeneric
public import Proofs.WeightSecurity.Stop

/-! A period of one-stage windows at the committed traces. -/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric
open FlockSoundness.Game
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

abbrev OnePlay (L : Compiled Coin A) := (x : Msg) × Prover Unit × (Fin (L.V.k x) → Coin)

noncomputable def oneGame (L : Compiled Coin A) (Rls : Type) : Verity.Game (Rls × OnePlay L) :=
  .send Rls fun m => .send (Msg × Prover Unit) fun xp =>
    .coin (Fin (L.V.k xp.1) → Coin) fun r => .ret (m, ⟨xp.1, xp.2, r⟩)

def OnePlay.accepted (H : Msg → Msg) {L : Compiled Coin A} (p : OnePlay L) : Prop :=
  (run H p.2.1 L.V p.1 () p.2.2).out = .acc

open Classical in
noncomputable def oneRegCount (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (P : Prover Unit) : ℕ :=
  if L.Scope x then (L.Freg H P x ()).card else 0

noncomputable def oneRegRec (H : Msg → Msg) (L : Compiled Coin A) (n : ℕ)
    (b : ((O → V) × Tick × Adv) × OnePlay L) : Served O V Tick Adv where
  accepted := b.2.accepted H
  sampledIncorrect := ∃ u ∈ L.checked b.2.1 b.2.2.2, u ∈ L.Freg H b.2.2.1 b.2.1 ()
  incorrect := n
  egress := b.1.1
  tick := b.1.2.1
  adv := b.1.2.2

noncomputable def oneRegRun (H : Msg → Msg) (Lv : ℕ → Compiled Coin A) (T : ℕ)
    (s : Strategy (monitoringPeriodG (fun t => oneGame (Lv t) ((O → V) × Tick × Adv)) 0 T)) :
    Verity.Game (List (Served O V Tick Adv)) :=
  monitoringPeriodRun (fun t => oneGame (Lv t) ((O → V) × Tick × Adv))
    (fun t τ => oneRegRec H (Lv t) (oneRegCount H (Lv t) τ.2.1.1 τ.2.1.2)) 0 T s

def MonitoringPeriodReg (H : Msg → Msg) (Lv : ℕ → Compiled Coin A) (x : ℝ) : Prop :=
  ∀ (T : ℕ) (ε : ℕ → ℝ) (s : Strategy (monitoringPeriodG (fun t => oneGame (Lv t) ((O → V) × Tick × Adv)) 0 T)),
    (∀ t τ, MonitoringPeriodPlays (fun t => oneGame (Lv t) ((O → V) × Tick × Adv)) 0 T s t τ →
      (Lv t).εreg H τ.2.1.1 τ.2.1.2 ≤ ε t) →
    ∀ M : ℕ, Verity.Game.value (Served.Undetected M) (oneRegRun H Lv T s) ≤
      ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) + ENNReal.ofReal (x ^ M)

def CompilerMonitoringPeriodReg : Prop :=
  ∀ (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin) (O V Tick Adv : Type)
    (H : Msg → Msg) (Lv : ℕ → Compiled Coin A) (x : ℝ), A.KS H →
    (∀ t, (Lv t).commit.Binds H ∧ (Lv t).Sessions H) → 0 ≤ x → x ≤ 1 →
    (∀ t y s P, (Lv t).parse y = some s → (Lv t).partition s = some P → InScope P s →
      ∀ B ⊆ Finset.range P.blocks.length, s.design.Pa P B ≤ x ^ B.card) →
    MonitoringPeriodReg (O := O) (V := V) (Tick := Tick) (Adv := Adv) H Lv x

end WeightSecurity.SecurityProofs
