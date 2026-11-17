module

public import Definitions.Claims

/-! `ZKmalRx` at one tape. -/

@[expose] public section

namespace Claims.ZKMal

theorem zkmalRx_const {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (P : W → Prover (N × Ω)) (R : Rel W) (S : Simulator) (εzk γ : Msg → ℝ) :
    ZKmalRx (fun _ => N) (fun _ => Ω) H k (fun w _ => P w) R S εzk γ ↔ ZKmalR H k P R S εzk γ :=
  Iff.rfl

theorem zkmalRx_const_of {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N × Ω)) (x₀ : Msg) (hP : ∀ w x, P w x = P w x₀) (R : Rel W)
    (S : Simulator) (εzk γ : Msg → ℝ) :
    ZKmalRx (fun _ => N) (fun _ => Ω) H k P R S εzk γ ↔ ZKmalR H k (fun w => P w x₀) R S εzk γ := by
  rw [show P = fun w _ => P w x₀ from funext fun w => funext fun x => hP w x]
  exact Iff.rfl

theorem _root_.Claims.ZKmalRx.at {W : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} (h : ZKmalRx N Ω H k P R S εzk γ) (x : Msg) :
    ZKmalR H k (fun w => P w x) (fun x' w => x' = x ∧ R x' w) S εzk γ := by
  rintro x' w ⟨rfl, hw⟩
  exact h x' w hw

end Claims.ZKMal
