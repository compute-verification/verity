module

public import Definitions.Claims.SamplingTwo

/-!
# A checked unit's openings at its replay block

`Claims.SamplingGeneric.SessionOpensTwo`, the two-stage compiler's `opens` clause at the sampled record
(`Claims.Sampling.Deploy`). It reads only the definitions tree, so a deployment's proof of it reads none of the
compiler's proofs; `SamplingGeneric.TwoDeployed` feeds it to the two-stage contract (`sessionsTwo_compiled`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims.Sampling (InScopeTwo replayOf)

/-- **A checked unit's Flock witness makes it correct on its replay block**: on a statement in the two-stage scope,
at an accepted run that checks slot `u` and shows its session, a Flock witness in the session's relation reveals
openings such that any values `val` and interior values `ι` agreeing with those that pass (under the statement's roots
and the interior roots the run showed for `u`'s replay block `ρ`) make the proof block at `u` correct on `ρ`'s values
(`Prog.refIn` at `ρ`). -/
def SessionOpensTwo {Coin Col T : Type} [Inhabited Coin] (H : Msg → Msg) (D : Claims.Sampling.Deploy Coin Col T)
    (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two)
    (iroots : Msg → Seen → ℕ → List Msg) : Prop :=
  ∀ x s Pt d (Ω : Type) (P : Prover Ω) ω r u xS vS wF,
    D.parse x = some s → D.partition s = some Pt → two s = some d → InScopeTwo Pt d s →
    (run H P D.V x ω r).out = .acc → (∃ p : Fin (Claims.Sampling.nPU D x), (D.checked x r)[p.val]? = some u) →
    D.session x u (run H P D.V x ω r).seen = some (xS, vS) → Claims.Flock.check D.flock.parse xS wF = true →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ replayOf Pt d.part u = some ρ ∧ d.part.blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ D.reveal s u (run H P D.V x ω r).seen wF, D.commit.Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (run H P D.V x ω r).seen ρ → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn rb val ι) b

end Claims.SamplingGeneric
