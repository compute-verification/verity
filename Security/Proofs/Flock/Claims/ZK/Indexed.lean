import Definitions.Claims.IndexedTapes

/-!
# `ZKmalRx` at one tape

The spec's zero knowledge on statement-indexed tapes (`ZKmalRx`) at one tape at every statement is `ZKmalR`
(`zkmalRx_const`), also at a prover that takes the statement and is the same at each (`zkmalRx_const_of`); at one
statement it is `ZKmalR` on that statement's tape, at the relation there (`ZKmalRx.at`).
-/

namespace Claims.ZKMal

/-- `ZKmalR` is `ZKmalRx` at one tape at every statement. -/
theorem zkmalRx_const {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (P : W → Prover (N × Ω)) (R : Rel W) (S : Simulator) (εzk γ : Msg → ℝ) :
    ZKmalRx (fun _ => N) (fun _ => Ω) H k (fun w _ => P w) R S εzk γ ↔ ZKmalR H k P R S εzk γ :=
  Iff.rfl

/-- **`ZKmalRx` at a prover the same at every statement** is `ZKmalR` at that prover (at any one statement `x₀`). -/
theorem zkmalRx_const_of {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N × Ω)) (x₀ : Msg) (hP : ∀ w x, P w x = P w x₀) (R : Rel W)
    (S : Simulator) (εzk γ : Msg → ℝ) :
    ZKmalRx (fun _ => N) (fun _ => Ω) H k P R S εzk γ ↔ ZKmalR H k (fun w => P w x₀) R S εzk γ := by
  rw [show P = fun w _ => P w x₀ from funext fun w => funext fun x => hP w x]
  exact Iff.rfl

/-- **`ZKmalRx` at one statement `x`** is `ZKmalR` on `x`'s tape, at the relation restricted to `x`. -/
theorem _root_.Claims.ZKmalRx.at {W : Type} {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)]
    [∀ x, Fintype (Ω x)] [∀ x, Nonempty (Ω x)] {H : Msg → Msg} {k : Msg → ℕ} {P : W → (x : Msg) → Prover (N x × Ω x)}
    {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} (h : ZKmalRx N Ω H k P R S εzk γ) (x : Msg) :
    ZKmalR H k (fun w => P w x) (fun x' w => x' = x ∧ R x' w) S εzk γ := by
  rintro x' w ⟨rfl, hw⟩
  exact h x' w hw

end Claims.ZKMal
