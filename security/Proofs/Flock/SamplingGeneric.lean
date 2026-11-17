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
public import Proofs.Flock.SamplingGeneric.RegTrace
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
public import Proofs.Flock.SamplingGeneric.TwoRankinThreshold
public import Proofs.Flock.SamplingGeneric.TwoReg
public import Proofs.Flock.SamplingGeneric.TwoRegMiss
public import Proofs.Flock.SamplingGeneric.TwoRegX
public import Proofs.Flock.SamplingGeneric.TwoRegFirstHit
public import Proofs.Flock.SamplingGeneric.TwoRegFirstHitClass
public import Proofs.Flock.SamplingGeneric.TwoFirstHitKS
public import Proofs.Flock.SamplingGeneric.RunExtAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassNumbers
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassCount

/-! The sampled compiler in the claims core, with Flock as its instance at the deployed decoder. -/

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

theorem flockKS : FlockKS (Coin := Coin) (Col := Col) (T := T) := flockKS_of_compilerKS compilerKS

/-- The count form, at the miss of a count. -/
theorem compilerKSCount : CompilerKSCount := compilerKSCount_of compilerKS

theorem flockKSReg_of_compilerKSReg (h : CompilerKSReg) : FlockKSReg (Coin := Coin) (Col := Col) (T := T) := by
  intro H D _ L hKS hdec hB hS
  exact h Coin (Arg.ofFlock D) L H (flockArgKS H D hKS hdec) hB hS

/-- **The committed form**: `CompilerKSReg` from the sessions' bound (`compilerKSReg_of`). -/
theorem compilerKSReg : CompilerKSReg := compilerKSReg_of sessionKSOf

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

theorem ks_compiled (H : Claims.Msg → Claims.Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode) (hB : (commit c).Binds H) (hd : c.SampleDesign)
    (hr : c.Refuses H) (hi : c.InteriorFixed H iroots) (hf : c.SampleFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : (compiled F c iroots).KS H :=
  Claims.SamplingGeneric.flockKS H F _ hKS hlist hB (sessions_compiled F c iroots H hd hr hi hf hruns hnum hopens)

theorem ksReg_compiled (H : Claims.Msg → Claims.Msg) (hKS : Claims.Flock.KS H F)
    (hlist : Claims.Flock.ListWithin F.len F.enc F.θ F.decode) (hB : (commit c).Binds H) (hd : c.SampleDesign)
    (hr : c.Refuses H) (hi : c.InteriorFixed H iroots) (hf : c.SampleFixed)
    (hruns : Claims.Sampling.SessionRuns H (deployOf F c)) (hnum : Claims.Sampling.InnerNumbers (deployOf F c))
    (hopens : Claims.Sampling.SessionOpens H (deployOf F c) iroots) : (compiled F c iroots).KSReg H :=
  Claims.SamplingGeneric.flockKSReg H F _ hKS hlist hB (sessions_compiled F c iroots H hd hr hi hf hruns hnum hopens)

end SamplingDeploy
