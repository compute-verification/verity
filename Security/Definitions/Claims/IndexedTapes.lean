import Definitions.Claims

/-!
# Claims: zero knowledge on statement-indexed tapes

Zero knowledge of a protocol of sessions in turn is on statement-indexed tapes (`ZKmalRx`): the prover at `x` runs on
a nonce and a rest of `x`'s own types, since each session runs on a tape of
its own and the number of sessions grows with the statement. `ZKmalR` is its case of one tape at every statement. The
recursion's and sampling's claims read it (`Recursion.ZK`, `Recursion.ZKPriv`, `Sampling.ZK`), each masked session with
its own `Hello` round; Flock's session stays at one tape (`Flock.ZK`), proved on a class of statements of bounded table
count (`Flock.zk_deployZk_ref_on`).
-/

namespace Claims

/-- **`ZKmalR` on statement-indexed tapes**: the prover at `x` runs on a nonce `N x` and a rest `Ω x`, so its tape may
grow with the statement, as a protocol of a session per unit's does. At one tape at every statement it is `ZKmalR`. -/
def ZKmalRx {W : Type} (N Ω : Msg → Type) [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (H : Msg → Msg) (k : Msg → ℕ) (P : W → (x : Msg) → Prover (N x × Ω x)) (R : Rel W)
    (S : Simulator) (εzk γ : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = k x → ∀ (Ev : List (Msg × Out) → Prop),
    |Pr (fun ω : N x × Ω x => Ev (runStar H (P w x) Vs x ω θ).msgs) - Pr (fun c : S.C x => Ev (S.out H Vs θ x c))|
      ≤ εzk x + γ x * (Fork H (P w x) Vs θ x).Pr H

end Claims
