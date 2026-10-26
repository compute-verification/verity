import Proofs.Core.Game.BatchInduced
import Proofs.Pouw.PearlC.FlockTile.Bridge
import Proofs.Flock.Soundness.Discharge.Exec.Event

/-!
# A causal strategy's projection onto the live audit

R5's game is causal: the prover's strategy is a tree over the coins released so far, and the sampled proofs session
(`FlockSoundness.Discharge.Exec.auditLive`, C-Flock's game) runs inside it beside the other subprotocols' rounds. A
bound on the session's event at every strategy of `auditLive` holds at every strategy of the game it runs in:

* `prob_toVerity_le`: in core's image of a C-Flock game (`toVerity`).
* `project_interleave_fst`, `project_interleave_snd`, `project_batch`: the session interleaved with another game, on
  either side, or one game of a batch: the period strategy induces one of the session's (core's
  `prob_interleave_fst_le_induced` and its siblings) whose event is at least as likely.
* `prob_bind_le_of_le`: the session, or a game holding it, after a commit phase: a bound at every continuation is one on
  the whole.
* `auditAfter`: the live audit once the prover has registered `R` (`auditLive_eq`); `prob_auditAfter_le` turns a
  bound on `auditLive` into one on every `auditAfter R`, so the registration may be sent in the period's commit phase.
-/

namespace Flock.Bridge

open scoped ENNReal
open Pouw.PearlC.FlockTile (toVerity stratOfVerity prob_ofVerity)

variable {α β : Type}

/-- **A bound at every strategy holds in core's image.** -/
theorem prob_toVerity_le {Q : α → Prop} {δ : ℝ≥0∞} {g : FlockSoundness.Game α}
    (hg : ∀ τ, FlockSoundness.Game.prob Q g τ ≤ δ) (s : Verity.Game.Strategy (toVerity g)) :
    Verity.Game.prob Q (toVerity g) s ≤ δ := by
  rw [← prob_ofVerity]
  exact hg _

/-- **The session on the left of an interleaving**: a strategy of the interleaving projects onto one of the session
whose event is at least as likely. -/
theorem project_interleave_fst (Q : α → Prop) (g : FlockSoundness.Game α) (h : Verity.Game β)
    (σ : Verity.Game.Strategy (Verity.Game.interleave (toVerity g) h)) :
    ∃ τ : FlockSoundness.Game.Strategy g,
      Verity.Game.prob (fun x => Q x.1) (Verity.Game.interleave (toVerity g) h) σ ≤ FlockSoundness.Game.prob Q g τ := by
  obtain ⟨s, -, hs⟩ := Verity.Game.prob_interleave_fst_le_induced Q (toVerity g) h σ
  exact ⟨stratOfVerity g s, by rw [prob_ofVerity]; exact hs⟩

/-- **The session on the right of an interleaving.** -/
theorem project_interleave_snd (Q : β → Prop) (h : Verity.Game α) (g : FlockSoundness.Game β)
    (σ : Verity.Game.Strategy (Verity.Game.interleave h (toVerity g))) :
    ∃ τ : FlockSoundness.Game.Strategy g,
      Verity.Game.prob (fun x => Q x.2) (Verity.Game.interleave h (toVerity g)) σ ≤ FlockSoundness.Game.prob Q g τ := by
  obtain ⟨s, -, hs⟩ := Verity.Game.prob_interleave_snd_le_induced Q h (toVerity g) σ
  exact ⟨stratOfVerity g s, by rw [prob_ofVerity]; exact hs⟩

/-- **The session as one game of a batch**, at position `pre.length`. -/
theorem project_batch (Q : α → Prop) (d : α) (g : FlockSoundness.Game α) (pre post : List (Verity.Game α))
    (σ : Verity.Game.Strategy (Verity.Game.batch (pre ++ toVerity g :: post))) :
    ∃ τ : FlockSoundness.Game.Strategy g,
      Verity.Game.prob (fun outs => Q (outs.getD pre.length d)) (Verity.Game.batch (pre ++ toVerity g :: post)) σ ≤
        FlockSoundness.Game.prob Q g τ := by
  obtain ⟨s, -, hs⟩ := Verity.Game.prob_batch_getD_le_induced Q d (toVerity g) post pre σ
  exact ⟨stratOfVerity g s, by rw [prob_ofVerity]; exact hs⟩

/-- **A bound at every continuation is one on the whole**: after any commit phase `g`, a continuation whose event is
at most `δ` at every strategy leaves it at most `δ`. -/
theorem prob_bind_le_of_le {Q : β → Prop} {δ : ℝ≥0∞} (g : Verity.Game α) (f : α → Verity.Game β)
    (hf : ∀ a t, Verity.Game.prob Q (f a) t ≤ δ) (σ : Verity.Game.Strategy (g.bind f)) :
    Verity.Game.prob Q (g.bind f) σ ≤ δ := by
  refine (Verity.Game.prob_bind_le f Q (fun _ => True) δ (fun a _ t => hf a t) g σ).trans ?_
  have h0 : Verity.Game.prob (fun _ => ¬ True) g (Verity.Game.Strategy.ofBind f g σ) = 0 :=
    le_antisymm ((Verity.Game.prob_mono (q := fun _ => False) (fun _ h => h trivial) g _).trans
      (Verity.Game.prob_false g _).le) bot_le
  rw [h0, zero_add]

/-- The session interleaved after a commit phase: the bound on the session at every strategy bounds its event there. -/
theorem prob_bind_interleave_fst_le {γ : Type} {Q : α → Prop} {δ : ℝ≥0∞} (c : Verity.Game γ)
    (g : γ → FlockSoundness.Game α) (h : γ → Verity.Game β) (hg : ∀ a τ, FlockSoundness.Game.prob Q (g a) τ ≤ δ)
    (σ : Verity.Game.Strategy (c.bind fun a => Verity.Game.interleave (toVerity (g a)) (h a))) :
    Verity.Game.prob (fun x => Q x.1) (c.bind fun a => Verity.Game.interleave (toVerity (g a)) (h a)) σ ≤ δ :=
  prob_bind_le_of_le c _ (fun a t => by
    obtain ⟨τ, hτ⟩ := project_interleave_fst Q (g a) (h a) t
    exact hτ.trans (hg a τ)) σ

section Audit

open FlockSoundness.Discharge.Exec (auditLive)
open FlockSoundness.Audit (Law)

variable {n : ℕ} {Reg : Type}

/-- **The live audit once the prover has registered `R`**: the verifier draws, then the live session. -/
def auditAfter (L : Law n) (Reg : Type)
    (live : Finset (Fin n) → Reg → FlockSoundness.Game FlockSoundness.Discharge.Exec.Transcript) (R : Reg) :
    FlockSoundness.Game (Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript) :=
  .coin L.Ω fun ω => (live (L.draw ω) R).map fun t => (R, L.draw ω, t)

theorem auditLive_eq (L : Law n) (Reg : Type)
    (live : Finset (Fin n) → Reg → FlockSoundness.Game FlockSoundness.Discharge.Exec.Transcript) :
    auditLive L Reg live = .send Reg (auditAfter L Reg live) := rfl

/-- **A bound on the live audit bounds it after every registration**, so the registration may come from the period's
commit phase. -/
theorem prob_auditAfter_le {Q : Reg × Finset (Fin n) × FlockSoundness.Discharge.Exec.Transcript → Prop} {δ : ℝ≥0∞}
    (L : Law n)
    (live : Finset (Fin n) → Reg → FlockSoundness.Game FlockSoundness.Discharge.Exec.Transcript)
    (h : ∀ τ, FlockSoundness.Game.prob Q (auditLive L Reg live) τ ≤ δ) (R : Reg)
    (τ : FlockSoundness.Game.Strategy (auditAfter L Reg live R)) :
    FlockSoundness.Game.prob Q (auditAfter L Reg live R) τ ≤ δ :=
  h ⟨R, τ⟩

end Audit

end Flock.Bridge
