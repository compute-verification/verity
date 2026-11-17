module

public import Proofs.Flock.Claims.Sampling.Session

/-! `SessionRunsTo`: a unit's session runs as Flock runs, with the run's acceptance implying the session's only, since a later session may still refuse. -/

@[expose] public section

namespace Claims.Sampling

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

def SessionRunsTo (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin) (u : ℕ),
    (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    ∃ (xS : Msg) (emb : Fin (D.flock.V.k xS) → Fin (D.V.k x)) (P' : Prover Unit),
      StrictMono emb ∧ (∀ t, t ∈ D.rounds x u ↔ ∃ i, emb i = t) ∧
      (∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, r' t = r t) →
        (run H P D.V x ω r').out = .acc → (run H P' D.flock.V xS () fun i => r' (emb i)).out = .acc) ∧
      ∀ r' : Fin (D.V.k x) → Coin, (∀ t ∉ D.rounds x u, (∀ i, t < emb i) → r' t = r t) →
        D.session x u (run H P D.V x ω r').seen = some (xS, (run H P' D.flock.V xS () fun i => r' (emb i)).seen) ∧
          ∀ q ∈ (run H P' D.flock.V xS () fun i => r' (emb i)).vLog, q ∈ (run H P D.V x ω r').vLog

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
theorem SessionRuns.to {H : Msg → Msg} {D : Deploy Coin Col T} (h : SessionRuns H D) : SessionRunsTo H D := by
  intro x Ω P ω r u hu
  obtain ⟨xS, emb, P', hmono, hrange, hacc, hsess⟩ := h x Ω P ω r u hu
  exact ⟨xS, emb, P', hmono, hrange, fun r' hr' => (hacc r' hr').mp, hsess⟩

end Claims.Sampling
