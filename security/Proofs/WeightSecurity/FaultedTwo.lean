module

public import Proofs.WeightSecurity.Faulted
public import Proofs.WeightSecurity.Two

/-!
# The restated theft theorem at a two-stage audit

`TheftFaulted` over windows each audited by SPOT's two-stage verifier (`CompiledTwo`), at the budget
`M = T·(K − 1) + 1` that `two_undetected` leaves undetected with probability at most `Σ_t ε_t`. A two-stage window's
incorrect count (`twoRec`) is the fewest replay blocks an extraction in `RTwo` marks, so here a fault pattern names
replay blocks: `size φ` counts replay blocks, and a fault overrides a replay block's outputs. It takes `TheftFaulted`'s
cut `κ`: `FaultedUseHard` is taken only at the replay-block patterns `κ` doesn't cut (`Cut.Interiors`), and a pattern
`κ` cuts costs its hint.
-/

@[expose] public section

namespace WeightSecurity.SecurityProofs

open Claims Claims.Sampling Claims.SamplingGeneric WeightSecurity.Assumptions
open FlockSoundness.Game hiding Msg
open scoped ENNReal

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {O V Tick Adv : Type}

/-- **`TheftFaulted` at a two-stage audit**: with each window's verifier knowledge sound at the footprint, refusing
what is out of scope, and capped at `K` (`WindowCap`), the outsider steals the weights with probability at most the
atypical weights' probability, plus the fault channel's `|S|·N^(λ−1)·p` at the budget `M = T·(K − 1) + 1` incorrect
replay blocks, plus `Σ_t ε_t`, each `ε_t` bounding window `t`'s `KSLinOn` error at every strategy the run plays it with,
plus the probability that an accepted release is not the faulted circuit's run at a pattern of at most its window's
incorrect replay blocks. `N` counts only the patterns the cut `κ` doesn't cut, at which alone `FaultedUseHard` is taken;
a pattern `κ` cuts adds its hint to `|S|`. -/
theorem TheftFaultedTwo {Ω R Θ Q Θ' F : Type} [Fintype Ω] [Fintype R] [Fintype Adv] [Inhabited Adv]
    [Fintype O] [DecidableEq O] [Fintype V] [Fintype Tick] [Inhabited Tick] [Fintype F]
    (H : Msg → Msg) (Lv : ℕ → CompiledTwo Coin A) (T L K : ℕ) {δ : ℝ} (hδ : 0 ≤ δ) (ε : ℕ → ℝ)
    (C : Θ → Q × Adv → O → V) (Cf : Θ → (Q × Adv) × F → O → V) (size : F → ℕ) (φ₀ : F)
    (κ : Cut (Q × Adv) (O → V)) (gen : Ω → Θ)
    (Typ : Set Θ) (Stolen : Θ → Θ' → Prop) (Aset : Set (Outsider R (Q × Adv) (O → V) Θ'))
    (Af : Set (Outsider R ((Q × Adv) × κ.Interiors Cf size φ₀) (O → V) Θ')) (p lam : ℝ)
    (Ou : Outsider R (Q × Adv) (Option ((O → V) × Tick)) Θ')
    (s : Θ → R → Strategy (periodG (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T))
    (hφ₀ : NoFault size φ₀)
    (hhon : HonestUseHard C gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T Aset p)
    (hfh : FaultedUseHard C (fun θ (x : (Q × Adv) × κ.Interiors Cf size φ₀) => Cf θ (x.1, x.2.1))
      (fun φ : κ.Interiors Cf size φ₀ => size φ.1) gen (fun θ θ' => θ ∈ Typ ∧ Stolen θ θ') T (T * (K - 1) + 1)
      Aset Af lam)
    (hsim : Simulable Af (Ou.withFaults (⟨φ₀, Or.inl rfl⟩ : κ.Interiors Cf size φ₀))
      (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T
        (T * (K - 1) + 1) ((L + 1) * Fintype.card O)))
    (hks : ∀ t < T, (Lv t).KS H)
    (href : ∀ t < T, ∀ x (P : Prover Unit) r, (run H P (Lv t).V x () r).out = .acc → (Lv t).Scope x)
    (hcap : ∀ t < T, WindowCap (Lv t) δ K)
    (herr : ∀ θ r t τ, PeriodPlays (fun t => twoGame (Lv t) ((O → V) × Tick × Adv)) 0 T (s θ r) t τ →
      twoErr H (Lv t) δ τ ≤ ε t) :
    Verity.Game.avg (fun z : Ω × R => Verity.Game.value
        (fun l => Stolen (gen z.1) (Ou.out z.2 (Served.view Ou L z.2 l T))) (twoRun H Lv δ T (s (gen z.1) z.2))) ≤
      ENNReal.ofReal (Claims.Pr fun z : Ω × R => gen z.1 ∉ Typ) +
        ENNReal.ofReal (Fintype.card
          (faultChannel R Q Adv O V Tick (κ.Interiors Cf size φ₀) κ (fun φ => size φ.1) ⟨φ₀, Or.inl rfl⟩ T
            (T * (K - 1) + 1) ((L + 1) * Fintype.card O)).S *
            ((patternCount (fun φ : κ.Interiors Cf size φ₀ => size φ.1) T (T * (K - 1) + 1) : ℝ) ^ (lam - 1) *
              p)) +
        ∑ t ∈ Finset.range T, ENNReal.ofReal (ε t) +
        Verity.Game.avg (fun z : Ω × R => Verity.Game.value
          (fun l => ¬ Served.Faithful Cf size Ou L (gen z.1) z.2 l) (twoRun H Lv δ T (s (gen z.1) z.2))) :=
  TheftFaulted C Cf size φ₀ κ gen Typ Stolen Aset Af p lam Ou T L (T * (K - 1) + 1) _
    (fun θ r => twoRun H Lv δ T (s θ r)) hφ₀ hhon hfh hsim
    fun θ r => two_undetected H Lv hδ T K hks href hcap ε (s θ r) (herr θ r)

end WeightSecurity.SecurityProofs
