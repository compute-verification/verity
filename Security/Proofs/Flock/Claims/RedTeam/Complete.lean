import Proofs.Flock.Claims.Prob

/-!
# Red team: `Complete` holds at an empty tape

`Complete` quantifies over the honest prover's tapes (`∀ ω : Ω`) and asks nothing of `Ω`, so an honest prover with no
tape (`Ω = Empty`) meets it at `δ_c = 0`, whatever the verifier, relation and scope. With a nonempty tape the form has
content: a verifier that rejects every run fails it wherever the scope and the relation meet (`not_complete_rej`).

* `complete_empty`: `Complete H V P R Scope 0` at every `P : W → Prover Empty`.
* `flockComplete_empty`: Flock's `Complete` at every record; at the deployed records, `RedTeam.Deployed`.
* `not_complete_rej`: at a nonempty tape, the always-rejecting verifier fails `Complete` at `δ_c < 1`.
-/

namespace Claims.RedTeam

/-- An honest prover with no tape. -/
def noTape {W : Type} : W → Prover Empty := fun _ => ⟨fun _ _ ω => ω.elim⟩

/-- **`Complete` at an empty tape**, at `δ_c = 0` and every verifier, relation and scope. -/
theorem complete_empty {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (V : Verifier Coin) (P : W → Prover Empty) (R : Rel W) (Scope : Msg → Prop) : Complete H V P R Scope 0 :=
  fun _ _ _ _ ω => ω.elim

/-- **Flock's `Complete` at an empty tape**, at every record. -/
theorem flockComplete_empty {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    (H : Msg → Msg) (D : Flock.Deploy Coin Col T) : Flock.Complete H D noTape 0 :=
  complete_empty H D.V noTape _ _

/-- The verifier that rejects every run. -/
def rejV (Coin : Type) : Verifier Coin := ⟨fun _ => 1, fun _ _ _ => .ret .rej⟩

/-- **At a nonempty tape the form has content**: the always-rejecting verifier fails `Complete` at a statement in scope
with a witness, at `δ_c < 1`. -/
theorem not_complete_rej {Coin W Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Nonempty Ω]
    (H : Msg → Msg) (P : W → Prover Ω) (R : Rel W) (Scope : Msg → Prop) (δc : Msg → ℝ) {x : Msg} {w : W}
    (hs : Scope x) (hr : R x w) (hδ : δc x < 1) : ¬ Complete H (rejV Coin) P R Scope δc := by
  intro h
  have h1 := h x w hs hr (Classical.arbitrary Ω)
  have h0 : Pr (fun r : Fin ((rejV Coin).k x) → Coin => (run H (P w) (rejV Coin) x (Classical.arbitrary Ω) r).out = .acc)
      = 0 := by
    rw [Pr_congr (q := fun _ => False) fun r => by simp [run, stepWith, rejV, OC.eval]]
    exact Pr_false
  linarith

end Claims.RedTeam
