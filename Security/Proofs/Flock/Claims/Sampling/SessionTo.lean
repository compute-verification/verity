module

public import Proofs.Flock.Claims.Sampling.Session

/-!
# `SessionRunsTo`: a unit's session runs as Flock runs, with the run's acceptance implying the session's

`SessionRunsTo H D` is `SessionRuns H D` with the acceptance clause one way: where coins off a unit's session rounds are
fixed, the sampled run accepts only where the session's Flock run accepts. It is what a sampled verifier that runs several
sessions in sequence, and accepts only where every session accepted, gives: a session that accepts leaves the run free to
refuse in a later session, so the run's acceptance is not the session's. `SessionRuns.to` reads it off `SessionRuns`.
-/

@[expose] public section

namespace Claims.Sampling

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **Every checked unit's session runs as Flock runs, the run's acceptance implying the session's**: at the coins `r`
that check unit `u`, its session is a Flock run at a statement `xS` on its rounds `emb` (all of `u`'s session rounds, in
order) of some prover `P'`, such that at any coins `r'` that agree with `r` off those rounds the run accepts only where
that Flock run accepts, and at any coins that agree with `r` before them the session the run shows is that Flock run's
transcript, whose `H` calls the run makes. -/
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
/-- **`SessionRuns` gives `SessionRunsTo`**: an acceptance that agrees both ways agrees one way. -/
theorem SessionRuns.to {H : Msg → Msg} {D : Deploy Coin Col T} (h : SessionRuns H D) : SessionRunsTo H D := by
  intro x Ω P ω r u hu
  obtain ⟨xS, emb, P', hmono, hrange, hacc, hsess⟩ := h x Ω P ω r u hu
  exact ⟨xS, emb, P', hmono, hrange, fun r' hr' => (hacc r' hr').mp, hsess⟩

end Claims.Sampling
