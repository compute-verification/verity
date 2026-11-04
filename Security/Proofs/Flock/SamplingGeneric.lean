module

public import Definitions.Claims.SamplingGeneric
public import Proofs.Flock.Claims.ERepCost
public import Proofs.Flock.SamplingGeneric.Fails
public import Proofs.Flock.SamplingGeneric.Split
public import Proofs.Flock.SamplingGeneric.Assemble
public import Proofs.Flock.SamplingGeneric.Coins
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.Flock.SamplingGeneric.Count
public import Proofs.Flock.SamplingGeneric.Bits
public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.Counter
public import Proofs.Flock.SamplingGeneric.Deployed
public import Proofs.Flock.SamplingGeneric.Instance

/-!
# The sampled compiler in the claims core, at Flock

The compiled extractor's failure event, its probability and `Compiled.KS` from the split are in
`SamplingGeneric/Fails.lean`.

`Arg.ofFlock` is Flock's `E^rep` at a decoder with Flock's errors, so its knowledge soundness is `Flock.KS`'s
(`flockArgKS`, by unfolding: its coins, extractor, `κ` and `γ` are `Flock.E`'s, `Flock.κ`'s and `Flock.γ`'s), and its
rounds are its shape's (`arg_ofFlock_wf`). Its shape carries `N₀`, so its reruns are `E^rep`'s at the shape: at most
`1 + m′·K`, and `1 + m′·k` in expectation (`ofFlock_reruns`). So the compiler's knowledge soundness gives Flock's
(`flockKS_of_compilerKS`); it is two steps (`SamplingGeneric/Split.lean`: `KSOf`, `SessionKSOf`, composed by
`compilerKS_of`), the second proved
from the session transfer and the session's fork (`SamplingGeneric/Session.lean`: `sessionKSOf`), the first from
`KSOf`'s deterministic half, a failed extraction giving the failure event or a `Bext` collision (`split`,
`SamplingGeneric/Assemble.lean`: `ksOf`). So **`CompilerKS` is proved** (`compilerKS`), and with it Flock's instance
(`flockKS`) and the count form (`compilerKSCount`, `SamplingGeneric/Count.lean`). `CompilerBits` follows from it and
the extractor's expected reruns and own calls (`SamplingGeneric/Bits.lean`, `ExpOwnOf`), so **`CompilerEfficient`,
`ExpOwnOf` and `CompilerBits` are proved** (`compilerEfficient`, `expOwnOf`, `SamplingGeneric/Efficient.lean`;
`compilerBits`). Their own-call bounds read `Compiled.OpenCallsLe`, without which they fail
(`Counter.not_ownLe_openCalls`, `SamplingGeneric/Counter.lean`). A
session's coins in a compiled run, `coinsOf` and `place`, invert each other (`SamplingGeneric/Coins.lean`). The
committed form, against the incorrect set of the plurality trace the prover's tape fixes, follows from the sessions'
bound and the rewinding lemma (`SamplingGeneric/Reg.lean`: `compilerKSReg_of`), so **`CompilerKSReg` is proved**
(`compilerKSReg`), and with it Flock's (`flockKSReg`).

A frozen sampled record is a compiled verifier over Flock (`ofDeploy`, `SamplingGeneric/Deployed.lean`): its opening
count is the frozen `Sampling.OpenCallsLe` (`openCallsLe_of_deploy`), its errors are within `Claims.Sampling`'s, and it
meets the contract where the record meets the frozen hypotheses `KS_of` reads (`sessions_ofDeploy`). At the deployed
record `deployOf F c` (`SamplingDeploy.compiled`), four of those are facts about the sampled verifier's code
(`Code.DrawLaw`, `Code.Refuses`, `Code.InteriorFixed`, `Code.DrawFixed`), and the compiler's KS and committed form
there take exactly them, Flock's KS at `F`, the commitment's binding and the sessions' runs, numbers and openings
(`SamplingDeploy.ks_compiled`, `SamplingDeploy.ksReg_compiled`).

`Claims.Sampling` is the compiler's instance there: at coupled coins `Sampling.E` and the compiled extractor at
`ofDeploy` fail alike, with the same failure probability and `B_ext` (`SamplingGeneric/Instance.lean`:
`samplingKS_of_flockKS`), so `Sampling.KS` follows from `Flock.KS`, the commitment's binding and the frozen hypotheses
`sessions_ofDeploy` reads, with `ε_table ≥ 0` (`samplingKS`), and at the deployed record from the code's four facts
with the sessions' runs, numbers and openings (`SamplingDeploy.samplingKS_deployOf`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

omit [Inhabited Coin] [DecidableEq Col] in
theorem arg_ofFlock_wf (D : Claims.Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    (Arg.ofFlock D decode).WF := fun _ => rfl

omit [DecidableEq Col] in
/-- **`E^rep`'s reruns at its shape**: at most `1 + m′·K` on every coin, and `1 + m′·k` in expectation, with
`k = ⌈4·N₀/e⌉` read from the shape (`ERep_cost`). -/
theorem ofFlock_reruns (H : Msg → Msg) (D : Claims.Flock.Deploy Coin Col T) (decode : List (ℕ × Col) → List T) :
    (Arg.ofFlock D decode).RerunsLe H (fun s => 1 + s.1 * s.2.1) ∧
      (Arg.ofFlock D decode).RerunsExpLe H (fun s => 1 + s.1 * ⌈4 * s.2.2.2 / Real.exp 1⌉₊) :=
  Claims.ERep_cost H D.V D.waits D.k D.cap _ _ _

theorem flockArgKS : FlockArgKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ decode hKS hdec
  exact hKS decode hdec

theorem flockKS_of_compilerKS (h : CompilerKS) : FlockKS (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ decode L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D decode) L H (arg_ofFlock_wf D decode) (flockArgKS H D decode hKS hdec) hB hS

/-- **The sampled compiler is knowledge sound**: `KSOf` and `SessionKSOf` (`compilerKS_of`). -/
theorem compilerKS : CompilerKS := compilerKS_of ksOf sessionKSOf

/-- **Flock is an instance**: the compiler at `Arg.ofFlock`, from `Flock.KS` and its decoder's list bound. -/
theorem flockKS : FlockKS (Coin := Coin) (Col := Col) (T := T) := flockKS_of_compilerKS compilerKS

/-- The count form, at the miss of a count. -/
theorem compilerKSCount : CompilerKSCount := compilerKSCount_of compilerKS

theorem flockKSReg_of_compilerKSReg (h : CompilerKSReg) : FlockKSReg (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ decode L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D decode) L H (arg_ofFlock_wf D decode) (flockArgKS H D decode hKS hdec) hB hS

/-- **The committed form**: `CompilerKSReg` from the sessions' bound (`compilerKSReg_of`). -/
theorem compilerKSReg : CompilerKSReg := compilerKSReg_of sessionKSOf

/-- **Flock's committed form**: the compiler's at `Arg.ofFlock`. -/
theorem flockKSReg : FlockKSReg (Coin := Coin) (Col := Col) (T := T) := flockKSReg_of_compilerKSReg compilerKSReg

/-- **Bits**: `CompilerKS` and `ExpOwnOf` (`compilerBits_of`). -/
theorem compilerBits : CompilerBits := compilerBits_of compilerKS expOwnOf

/-- **`Sampling.KS` from the compiler**: `flockKS` at `ofDeploy` (`samplingKS_of_flockKS`). -/
theorem samplingKS [Zero Col] {H : Msg → Msg} {D : Claims.Sampling.Deploy Coin Col T}
    (iroots : Msg → Seen → ℕ → List Msg) (hKS : Claims.Flock.KS H D.flock) (hbind : D.commit.Binds H)
    (hopens : Claims.Sampling.SessionOpens H D iroots) (hfix : Claims.Sampling.InteriorFixed H D iroots)
    (hdraw : Claims.Sampling.DrawLaw D) (hrefuse : Claims.Sampling.Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x)
    (hdfix : Claims.Sampling.DrawFixed D) (hruns : Claims.Sampling.SessionRuns H D)
    (hnum : Claims.Sampling.InnerNumbers D) : Claims.Sampling.KS H D :=
  samplingKS_of_flockKS flockKS iroots hKS hbind hopens hfix hdraw hrefuse hε hdfix hruns hnum

end Claims.SamplingGeneric

namespace SamplingDeploy

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T) (c : Code)
  (decode : List (ℕ × List FlockLevel3.GF128) → List T) (iroots : Claims.Msg → Claims.Seen → ℕ → List Claims.Msg)

/-- **The compiler at the deployed record**: `Compiled.KS` of the deployed sampled verifier over Flock (`compiled`),
from Flock's KS at `F` and its decoder's list bound, the commitment's binding, the code's four facts and the sessions'
runs, numbers and openings (`flockKS`, `sessions_compiled`). -/
theorem ks_compiled (H : Claims.Msg → Claims.Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ decode) (hB : (commit c).Binds H) (hd : c.DrawLaw)
    (hr : c.Refuses H) (hi : c.InteriorFixed H iroots) (hf : c.DrawFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : (compiled F c decode iroots).KS H :=
  Claims.SamplingGeneric.flockKS H F decode _ hKS hlist hB
    (sessions_compiled F c decode iroots H hd hr hi hf hruns hnum hopens)

/-- **The committed form at the deployed record**: as `ks_compiled`, `KSReg` (`flockKSReg`). -/
theorem ksReg_compiled (H : Claims.Msg → Claims.Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ decode) (hB : (commit c).Binds H) (hd : c.DrawLaw)
    (hr : c.Refuses H) (hi : c.InteriorFixed H iroots) (hf : c.DrawFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : (compiled F c decode iroots).KSReg H :=
  Claims.SamplingGeneric.flockKSReg H F decode _ hKS hlist hB
    (sessions_compiled F c decode iroots H hd hr hi hf hruns hnum hopens)

/-- **`Sampling.KS` at the deployed record**: from Flock's KS at `F`, the commitment's binding, the code's four facts
and the sessions' runs, numbers and openings (`samplingKS`). -/
theorem samplingKS_deployOf (H : Claims.Msg → Claims.Msg) (hKS : Claims.Flock.KS H F) (hB : (commit c).Binds H)
    (hd : c.DrawLaw) (hr : c.Refuses H) (hi : c.InteriorFixed H iroots) (hf : c.DrawFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : Claims.Sampling.KS H (deployOf F c) :=
  Claims.SamplingGeneric.samplingKS iroots hKS hB hopens hi hd hr (deployOf_εtable_nonneg F c) hf hruns hnum

end SamplingDeploy
