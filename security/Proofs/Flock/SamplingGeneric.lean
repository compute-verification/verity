module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Fails
public import Proofs.Flock.SamplingGeneric.Split
public import Proofs.Flock.SamplingGeneric.SessCount
public import Proofs.Flock.SamplingGeneric.Assemble
public import Proofs.Flock.SamplingGeneric.Coins
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.Flock.SamplingGeneric.Count
public import Proofs.Flock.SamplingGeneric.Bits
public import Proofs.Flock.SamplingGeneric.KSAt
public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Flock.SamplingGeneric.RegAt
public import Proofs.Flock.SamplingGeneric.RegX
public import Proofs.Flock.SamplingGeneric.RegCanon
public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.Counter
public import Proofs.Flock.SamplingGeneric.Deployed
public import Proofs.Flock.SamplingGeneric.ZK
public import Proofs.Flock.SamplingGeneric.FlockZk
public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.SamplingGeneric.TwoCount
public import Proofs.Flock.SamplingGeneric.TwoKSAt
public import Proofs.Flock.SamplingGeneric.TwoServed
public import Proofs.Flock.SamplingGeneric.TwoTable
public import Proofs.Flock.SamplingGeneric.TwoBernoulli
public import Proofs.Flock.SamplingGeneric.TwoCap
public import Proofs.Flock.SamplingGeneric.Complete
public import Proofs.Flock.SamplingGeneric.TwoRankin
public import Proofs.Flock.SamplingGeneric.TwoReg
public import Proofs.Flock.SamplingGeneric.TwoRegMiss
public import Proofs.Flock.SamplingGeneric.TwoRegX
public import Proofs.Flock.SamplingGeneric.TwoRegFirstHit
public import Proofs.Flock.SamplingGeneric.TwoRegFirstHitClass
public import Proofs.Flock.SamplingGeneric.RunExtAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassNumbers
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassCount

/-!
# The sampled compiler in the claims core, at Flock

The compiled extractor's failure event, its probability and `Compiled.KS` from the split are in
`SamplingGeneric/Fails.lean`; the sessions' weights at the gates their blocks need, counted by block
(`Compiled.sum_sessions_le`), in `SamplingGeneric/SessCount.lean`.

`Arg.ofFlock` is Flock's `E^rep` at the deployed decoder with Flock's errors, so its knowledge soundness is `Flock.KS`'s
at that decoder (`flockArgKS`, by unfolding: its coins, extractor, `κ` and `γ` are `Flock.E D D.decode`'s, `Flock.κ`'s
and `Flock.γ`'s). So the compiler's knowledge soundness gives Flock's (`flockKS_of_compilerKS`);
it is two steps (`SamplingGeneric/Split.lean`: `KSOf`, `SessionKSOf`, composed by `compilerKS_of`), the second proved
from the session transfer and the session's fork (`SamplingGeneric/Session.lean`: `sessionKSOf`), the first from
`KSOf`'s deterministic half, a failed extraction giving the failure event or a `Bext` collision (`split`,
`SamplingGeneric/Assemble.lean`: `ksOf`). So **`CompilerKS` is proved** (`compilerKS`), and with it Flock's instance
(`flockKS`) and the count form (`compilerKSCount`, `SamplingGeneric/Count.lean`). **`CompilerEfficient` is proved**
(`compilerEfficient`, `SamplingGeneric/Efficient.lean`): every proof system's extractor is `E^rep` at numbers its
shape fixes, so its strict and expected reruns and its own calls are bounded at every proof system (`Arg.reruns_le`,
`Arg.rerunsExp_le`, `Arg.own_le`), and the opening checks' count is its only premise beyond the contract.
`CompilerBits` follows from `CompilerKS` and `CompilerEfficient`'s expected-reruns and own-calls bounds
(`SamplingGeneric/Bits.lean`), so **`CompilerBits` is proved** (`compilerBits`). Their own-call bounds read the
opening checks' count, `Compiled.OpenCallsLe`, without which they fail (`Counter.not_ownLe_openCalls`,
`SamplingGeneric/Counter.lean`). A session's coins in a compiled run, `coinsOf` and `place`, invert each other
(`SamplingGeneric/Coins.lean`). The committed form, against the incorrect set of the plurality trace the prover's tape
fixes, follows from the sessions' bound and the rewinding lemma (`SamplingGeneric/Reg.lean`: `compilerKSReg_of`), so
**`CompilerKSReg` is proved** (`compilerKSReg`), and with it Flock's (`flockKSReg`). The same form against the incorrect
set of the trace read at the statement, the plurality over the prover's tape and the coins together, which no tape
moves, follows from the same bound and the rewinding lemma over the tapes and the runs
(`SamplingGeneric/RegX.lean`: `compilerKSRegX_of`), so **`CompilerKSRegX` is proved** (`compilerKSRegX`). At two stages
the committed form's event, against the committed boundary trace and each replay block's committed interior, follows the
same way, the interior's rewinding at the first stage's coins and its conflicts a second binding finder's collisions
(`SamplingGeneric/TwoReg.lean`: `CompiledTwo.eventTwo_le`).

`CompilerKSAt` reads `KS`'s two hash terms at a budget on the prover's whole SHA-512 work rather than at the calls the
oracle logs: `KS` gives it at every budget, and `CompilerEfficient`'s expected reruns and own calls bound its binding
finder's cost (`SamplingGeneric/KSAt.lean`), so **`CompilerKSAt` is proved** (`compilerKSAt`). The committed form's
binding terms read at a budget too: `B_conf` declared at `T` a run costs at most
`2T·(1 + n_sess·e_s) + 2·n_PU·openCalls` in expectation at the contract (`SamplingGeneric/RegAt.lean`), so
**`CompilerKSRegAt` is proved** (`compilerKSRegAt`); on every coin it costs at most
`2T·(1 + n_sess·ρ_s) + 2·n_PU·openCalls`, so **`CompilerKSRegAtSB` is proved** too (`compilerKSRegAtSB`), at
`StrictBirthday` in place of `ExpectedCR`. At two stages the
extractor is `Compiled.E` at the two-stage verifier's own waits' counts, and the two-stage contract gives the facts
its cost reads, so **`CompilerEfficientTwo` and `CompilerKSAtTwo` are proved** (`compilerEfficientTwo`,
`compilerKSAtTwo`, `SamplingGeneric/TwoKSAt.lean`).

A frozen sampled record is a compiled verifier over Flock (`ofDeploy`, `SamplingGeneric/Deployed.lean`): its opening
count is the frozen `Sampling.OpenCallsLe` (`openCallsLe_of_deploy`), and it meets the contract where the record meets
the frozen hypotheses (`sessions_ofDeploy`). Its knowledge soundness is the one guarantees read at the deployed sampled
verifier; `Claims.Sampling` holds no extractor or knowledge-soundness claim of its own. At the deployed
record `deployOf F c` (`SamplingDeploy.compiled`), four of those are facts about the sampled verifier's code
(`Code.DrawLaw`, `Code.Refuses`, `Code.InteriorFixed`, `Code.DrawFixed`), and the compiler's KS and committed form
there take exactly them, Flock's KS at `F`, the commitment's binding and the sessions' runs, numbers and openings
(`SamplingDeploy.ks_compiled`, `SamplingDeploy.ksReg_compiled`). At Flock's masked record and SHA-512, Flock's KS,
the decoder's list bound and the commitment's binding are proved (`SamplingGeneric/Masked.lean`: `ks_masked`,
`ksReg_masked`, and `SamplingDeploy.ks_masked` at the deployed record).

**`CompilerZK` is proved** (`compilerZK`, `SamplingGeneric/ZK.lean`): a hybrid over the sessions, each step `A`'s zero
knowledge at the session's statement against `V*` after the run's messages before it (`residual`). Flock's ZK server is
zero knowledge per outcome (`FlockZk.server_zkOut`, `SamplingGeneric/FlockZk.lean`), at the simulator given the outcome
and the server's error, under the premises `Claims.ZKServer.Flock.zk_server_lean_loop` takes.

**`CompilerComplete` is proved** (`compilerComplete`, `SamplingGeneric/Complete.lean`): each sampled slot rejects with
at most its session's error, the session's coins being uniform once the coins off its rounds are fixed, and a union
bound over the first `K` slots gives the compiled error.

At a two-stage design whose first stage samples a uniform `Q`-subset of the population's `N` replay blocks, its `P̄_a` is
at most the product a Bernoulli(Q/N) replay level gives, `∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ)` (`SamplingGeneric/TwoBernoulli.lean`:
`Design.Two.Pa_le_bernoulli_pop`), the integrity profile's acceptance. At a first stage below independent catches at
rates `q_ρ`, the admissible sets' count and their patterns' count are at most `δ^{−θ}` times a product over the replay
blocks (`SamplingGeneric/TwoRankin.lean`: **`AdmissibleTwoRankin`** and **`AdmissibleTwoRankinPatterns`** are proved).
At a first stage that catches each replay block at rate `r₁` and a second stage that samples each cell at rate `r₂`,
`P̄_a ≤ (1 − r₁·r₂)^m`, `m` the replay blocks the footprint marks incorrect, so a footprint in `M̄_δ` marks fewer than
every `K` with `(1 − r₁·r₂)^K ≤ δ` (`SamplingGeneric/TwoCap.lean`: **`CapTwo`** is proved). At two stages, the
committed form's incorrect set is the grid's incorrect proof blocks at the committed traces, and reads no second-stage
coin (`SamplingGeneric/TwoCap.lean`: `CompiledTwo.FregTwo_eq`, `CompiledTwo.FregTwo_forkOn`). So at rates `r₁` and
`r₂` the run's sample misses it with probability at most `(1 − r₁·r₂)^{|W|}`, `W` the replay blocks wrong at the
committed boundary trace, and the committed form's event gives its threshold form and the headline
(`SamplingGeneric/TwoRegMiss.lean`: **`CompilerKSRegTwo`**, **`CompilerMissTwo`** and **`CompilerHeadlineTwo`** are
proved, `CompiledTwo.ksRegTwo_of_event`). Those lemmas read the boundary trace as a parameter, so the same three hold
with the boundary read at the statement, where the replay blocks wrong at the boundary are one set at every tape
(`SamplingGeneric/TwoRegX.lean`: **`CompilerKSRegTwoX`**, **`CompilerMissTwoX`** and **`CompilerHeadlineTwoX`** are
proved).
-/

@[expose] public section

namespace Claims.SamplingGeneric

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

theorem flockArgKS : FlockArgKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ hKS hdec
  exact hKS D.decode hdec

theorem flockKS_of_compilerKS (h : CompilerKS) : FlockKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D) L H (flockArgKS H D hKS hdec) hB hS

/-- **The sampled compiler is knowledge sound**: `KSOf` and `SessionKSOf` (`compilerKS_of`). -/
theorem compilerKS : CompilerKS := compilerKS_of ksOf sessionKSOf

/-- **Flock is an instance**: the compiler at `Arg.ofFlock`, from `Flock.KS` and its decoder's list bound. -/
theorem flockKS : FlockKS (Coin := Coin) (Col := Col) (T := T) := flockKS_of_compilerKS compilerKS

/-- The count form, at the miss of a count. -/
theorem compilerKSCount : CompilerKSCount := compilerKSCount_of compilerKS

theorem flockKSReg_of_compilerKSReg (h : CompilerKSReg) : FlockKSReg (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D) L H (flockArgKS H D hKS hdec) hB hS

/-- **The committed form**: `CompilerKSReg` from the sessions' bound (`compilerKSReg_of`). -/
theorem compilerKSReg : CompilerKSReg := compilerKSReg_of sessionKSOf

/-- **Flock's committed form**: the compiler's at `Arg.ofFlock`. -/
theorem flockKSReg : FlockKSReg (Coin := Coin) (Col := Col) (T := T) := flockKSReg_of_compilerKSReg compilerKSReg

/-- **The committed form at the trace read at the statement**: `CompilerKSRegX` from the sessions' bound
(`compilerKSRegX_of`). -/
theorem compilerKSRegX : CompilerKSRegX := compilerKSRegX_of sessionKSOf

/-- **Bits**: `CompilerKS` and `CompilerEfficient` (`compilerBits_of`). -/
theorem compilerBits : CompilerBits := compilerBits_of compilerKS compilerEfficient

/-- **KS read at a time budget**: `CompilerKS` and `CompilerEfficient` (`compilerKSAt_of`). -/
theorem compilerKSAt : CompilerKSAt := compilerKSAt_of compilerKS compilerEfficient

/-- **The committed form read at a time budget**: `CompilerKSReg` and `B_conf`'s expected cost
(`compilerKSRegAt_of`). -/
theorem compilerKSRegAt : CompilerKSRegAt := compilerKSRegAt_of compilerKSReg

/-- **The committed form read at a time budget and the strict birthday bound**: `CompilerKSReg` and `B_conf`'s cost on
every coin (`compilerKSRegAtSB_of`). -/
theorem compilerKSRegAtSB : CompilerKSRegAtSB := compilerKSRegAtSB_of compilerKSReg

end Claims.SamplingGeneric

namespace SamplingDeploy

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code)
  (iroots : Claims.Msg → Claims.Seen → ℕ → List Claims.Msg)

/-- **The compiler at the deployed record**: `Compiled.KS` of the deployed sampled verifier over Flock (`compiled`),
from Flock's KS at `F` and its deployed decoder's list bound, the commitment's binding, the code's four facts and the
sessions' runs, numbers and openings (`flockKS`, `sessions_compiled`). -/
theorem ks_compiled (H : Claims.Msg → Claims.Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode) (hB : (commit c).Binds H) (hd : c.DrawLaw)
    (hr : c.Refuses H) (hi : c.InteriorFixed H iroots) (hf : c.DrawFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : (compiled F c iroots).KS H :=
  Claims.SamplingGeneric.flockKS H F _ hKS hlist hB (sessions_compiled F c iroots H hd hr hi hf hruns hnum hopens)

/-- **The committed form at the deployed record**: as `ks_compiled`, `KSReg` (`flockKSReg`). -/
theorem ksReg_compiled (H : Claims.Msg → Claims.Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode) (hB : (commit c).Binds H) (hd : c.DrawLaw)
    (hr : c.Refuses H) (hi : c.InteriorFixed H iroots) (hf : c.DrawFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : (compiled F c iroots).KSReg H :=
  Claims.SamplingGeneric.flockKSReg H F _ hKS hlist hB (sessions_compiled F c iroots H hd hr hi hf hruns hnum hopens)

end SamplingDeploy
