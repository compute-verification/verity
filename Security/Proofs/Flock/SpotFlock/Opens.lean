module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Claims.Sampling.OpensTwo
public import Proofs.Flock.Claims.Sampling.Deployed

/-!
# SPOT-Flock's session opens each checked proof block at its replay block

`sessionOpensTwo h`: `Claims.SamplingGeneric.SessionOpensTwo` at the deployed two-stage code (`code h`), Flock's
masked record and SHA-512, from what one session statement establishes (`StmtOpens h`).

**The run.** An accepted run of `V h` answered rounds 0 and 1 with their coins and round 2 with nothing, so the
verifier's check of the session's statement passed there (`stmtOk h x m₁ xS c₀ c₁`, at the round-1 message `m₁`, the
round-2 message `xS` and the coins of rounds 0 and 1). The session is `xS` and the transcript from round 3 on, the
interior roots are those `m₁` states, and the checked units are the second stage's on the coins of rounds 0 and 1.

**The statement** (`StmtOpens h`): with no run and no coin beyond those two, a session statement that the verifier's
check passes, at a Flock witness in its relation, reveals for each unit the second stage checks openings under which
its proof block is correct on its replay block's values. That is what the session statement's relation, the
witness's encoding of its openings and the registered reads establish, and the run only selects it.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **A run's transcript through its session's statement**: rounds 0 and 1 answered with their coins `c₀` and `c₁`,
round 2's message `xS` answered with nothing, then `rest` and the decision `out`. -/
def seenAt (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) (out : Out) : Seen :=
  ⟨(m₀, .msg c₀.bytes) :: (m₁, .msg c₁.bytes) :: (xS, .msg []) :: rest, out⟩

/-- **A session statement opens every unit it checks**: at a run's statement in the two-stage scope, a session
statement `xS` that the verifier's check passes at the round-1 message `m₁` and the coins `c₀` and `c₁`, and a Flock
witness in `xS`'s relation, each unit `u` the two stages check on those coins has a proof block `b` in a replay block
`ρ` with gates `rb`, and any values `val` and interior values `ι` that agree with the openings the witness reveals and
that pass (under the run's roots, and the interior roots `m₁` states for `ρ`) make `b` correct on `ρ`'s values. -/
def StmtOpens : Prop :=
  ∀ x s Pt d (m₀ m₁ xS : Msg) (c₀ c₁ : FlockDeploy.Coin) (rest : List (Msg × Out)) (out : Out) wF u,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ xS c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (src c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ replayOf Pt d.part u = some ρ ∧ d.part.blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ (code h).reveal s u (seenAt m₀ m₁ xS c₀ c₁ rest out) wF,
          (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots h x (seenAt m₀ m₁ xS c₀ c₁ rest out) ρ → ι q.2.pos = q.2.val)) →
        s.prog.Correct (s.prog.refIn rb val ι) b

/-- **SPOT-Flock's session opens each checked proof block at its replay block**: `SessionOpensTwo` at the deployed
two-stage code, from `StmtOpens`. -/
theorem sessionOpensTwo (hs : StmtOpens h) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) (iroots h) := by
  sorry

end SpotFlock
