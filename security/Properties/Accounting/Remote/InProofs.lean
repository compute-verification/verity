module

public import RemoteAudit.Window
public import Definitions.Core.Interaction

/-!
# PoUB's read session proved in zero knowledge, as deployed: stated, unproved

The memory-read session once its openings stay inside proofs. The auditor's session decides window `w` by a span check
on the span the prover sends: the sampled proof system's accepted one-stage statements, one opening unit a draw.
`SpanInProofsDeployed` says that a fixed extractor, rerunning any prover, recovers openings that today's check
`checkWindow` accepts, except `n · (ε_ks + ε_bind)` over the span's `n` statements and the collision term `Claims.KS`
carries.

The span check is SPANIN's `RemoteAudit.Span.checkSpan` (`RemoteAudit/Span.lean`, memory-accounting/poub-span-from-spot),
not on this branch yet, so the statement takes it and its `Span` as parameters, at the signature its issue gives:
`checkSpan (a : Auditor) (reg : Registration) (w : WindowRecord) (drawn : Array (Nat × Nat)) (sp : Span) : Verdict`. The
deployed instance is `SpanInProofsDeployed H Span.Span (·.statements.size) (fun reg w drawn sp => checkSpan a reg w drawn
sp == .accept)` at the auditor's `a`.

It is stated and not proved. What closes it is proofs' period form of V5 with every unit proved:
`Flock.Guarantees.IntegrityAllProvedCarried` at `I` the window's statements (each proves all its units, `kd = n`, and no
leaf is carried), whose conclusion is `Verity.InProofs.Gives` at `K₀ = 1` for the run whose failing units are the game's
wrong ones. `Properties.RemoteAudit.InProofs.SecurityProofs.spanGives` (`Proofs.Properties.Accounting.Remote.InProofs`)
then carries it from the opening unit's circuit to `checkWindow`. No carried form is needed: every value
`checkOpening` reads outside the opening is the verifier's own.
-/

@[expose] public section

namespace Properties.RemoteAudit.SpanDeployed

open _root_.RemoteAudit

/-- **PoUB's read session proved in zero knowledge, as deployed: stated and unproved**; proofs'
`IntegrityAllProvedCarried` (the period form of V5, every unit proved) closes it through `spanGives`. For a span check
`checkSpan` over spans of `statements sp` statements (SPANIN's, not on this branch yet), the registration `reg`, window
`w` and its draws `drawn`, and a session verifier `V` that accepts only a span (`spanOf`) of at most `n` statements that
`checkSpan` accepts: a fixed extractor `E` recovers, by rerunning any prover, openings that `checkWindow` accepts, except
`n · (εKs + εBind)` and `Claims.KS`'s collision term at `γ`. -/
def SpanInProofsDeployed (H : Claims.Msg → Claims.Msg) (Span : Type) (statements : Span → ℕ)
    (checkSpan : Registration → WindowRecord → Array (Nat × Nat) → Span → Bool) (εKs εBind γ : ℝ) : Prop :=
  ∀ (reg : Registration) (w : WindowRecord) (drawn : Array (Nat × Nat)) (n : ℕ)
    {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (V : Claims.Verifier Coin)
    (spanOf : List Claims.Msg → Option Span),
    (∀ (x : Claims.Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin),
      (Claims.run H P V x ω r).out = .acc →
        ∃ sp, spanOf ((Claims.run H P V x ω r).msgs.map Prod.fst) = some sp ∧ statements sp ≤ n ∧
          checkSpan reg w drawn sp = true) →
    ∃ E : Claims.Extractor Coin (Array Opening) V,
      Claims.KS H V (fun _ os => checkWindow reg reg.digest #[] w drawn os = .accept) E
        (fun _ => n * (εKs + εBind)) fun _ => γ

end Properties.RemoteAudit.SpanDeployed
