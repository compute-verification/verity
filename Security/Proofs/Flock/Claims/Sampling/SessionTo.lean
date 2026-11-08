module

public import Proofs.Flock.Claims.Sampling.Session

/-!
# A checked unit's session, one way

`SessionRunsTo` is `SessionRuns` (`Proofs.Flock.Claims.Sampling.Session`) with the session deciding the run one way: at
coins that agree with `r` off `u`'s rounds, an accepting sampled run makes the Flock prover's run on the session's coins
accept. The generic compiler's sessions read only that half (`Claims.SamplingGeneric.runs_ofDeploy`). Where several
sessions run in one sampled run, the other half can fail: a later session's messages may read an earlier session's
coins, so one session's run needn't decide whether the sampled run accepts.

* `SessionRuns.to`: `SessionRuns` gives `SessionRunsTo`.
-/

@[expose] public section

namespace Claims.Sampling

variable {Coin Col T : Type} [Inhabited Coin]

/-- **A checked unit's session is a Flock run that the sampled run's acceptance carries**: `SessionRuns` with, at coins
`r'` agreeing with `r` off `u`'s rounds, the sampled run accepting only when `P'`'s run on `r'`'s session coins does. -/
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

/-- **`SessionRuns` gives `SessionRunsTo`**: a session that decides the run both ways decides it one way. -/
theorem SessionRuns.to {H : Msg → Msg} {D : Deploy Coin Col T} (h : SessionRuns H D) : SessionRunsTo H D :=
  fun x Ω P ω r u hu =>
    let ⟨xS, emb, P', hmono, hrange, hacc, hsess⟩ := h x Ω P ω r u hu
    ⟨xS, emb, P', hmono, hrange, fun r' hr' => (hacc r' hr').mp, hsess⟩

end Claims.Sampling
