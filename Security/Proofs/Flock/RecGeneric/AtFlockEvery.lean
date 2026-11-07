module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.AtFlockProof
public import Proofs.Flock.RecGeneric.StatementProof
public import Proofs.Flock.Claims.Recursion.DeployFacts
public import Proofs.Flock.Claims.Recursion.OuterKS
public import Proofs.Flock.Claims.Recursion.KSFlockLinR
public import Proofs.Flock.Claims.Recursion.BindsDeployed
public import Proofs.Flock.Claims.Recursion.DeploySessions
public import Proofs.Flock.Claims.Recursion.BoundRoute
public import Proofs.Flock.Claims.Recursion.FlockKSRecord
public import Proofs.Flock.Claims.Recursion.DeployedR
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.KS.PlainSimR
public import Proofs.Flock.Claims.Assembled
public import Proofs.Flock.Claims.Sampling.Assembled

/-!
# `KSLinkOn` at every statement, from the statements a run accepts

`KSLinkOn` at the statements `G` admits holds at every statement wherever the composite's verifier accepts only at
statements `G` admits: elsewhere no run accepts, so `Pr[accept ∧ ¬extract] = 0`, and the bound is nonnegative where the
tables' errors are (`ksLinkOn_every`). The deployed compiled verifier accepts only dense statements
(`RecDeploy.run_acc_denseAt`, from `RecDeploy.acc_dense`), so `KSLinkOn` at `RecDeploy.DenseAt` gives
`Flock.Guarantees.RecursiveKSEveryAtRec` at every compiled record `recBoundAt` builds (`recursiveKSEveryAtRec_of_dense`).
At the deployment every premise of `RecoversLinkOn` but F7's contract holds, so `RecoversLinkOn` at the dense
statements and `ContractDense` give `Flock.Guarantees.RecursiveKSEvery` (`recursiveKSEvery_of_recovers`).
-/

@[expose] public section

namespace RecDeploy

open FlockDeploy (Coin)

/-- **The deployed compiled verifier accepts only dense statements**: a run accepts only where a step does
(`stepWith_acc`), which parses `x` at a dense form (`acc_dense`). -/
theorem run_acc_denseAt (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {Ω : Type} {P : Claims.Prover Ω} {ω : Ω}
    {r : Fin ((deployRec IV ivScope b).V.k x) → Coin}
    (h : (Claims.run H P (deployRec IV ivScope b).V x ω r).out = .acc) : DenseAt x := by
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ h
  exact fun s hs => acc_dense (IV := IV) (st := b.st) H ha s hs

end RecDeploy

namespace Claims.RecGeneric

open Claims Claims.Recursion

section Instance

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  {Col' T' : Type} [DecidableEq Col']

/-- **`KSLinkOn` at every statement from `KSLinkOn` where runs accept**: where every run the composite's verifier
accepts is of a statement `G` admits, the sessions' and `DE`'s table errors are nonnegative, `KSLinkOn` at `G` holds
at every statement: elsewhere no run accepts, so the failure is `0`, at most the bound. -/
theorem ksLinkOn_every [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (hK : KSLinkOn H D Bd DO Cp G)
    (hacc : ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin ((compDep D Bd DO).V.k x) → Coin),
      (run H P (compDep D Bd DO).V x ω r).out = .acc → G x)
    (hT : TableErrNonneg (D.record Bd.IVStar)) (hε : ∀ x, 0 ≤ Bd.DE.εtable x) :
    KSLinkOn H D Bd DO Cp fun _ => True := by
  intro x _ Ω _ _ P M R
  by_cases hG : G x
  · exact hK x hG Ω P M R
  have hF : ∀ F : Finder, 0 ≤ F.Pr H := fun F => Pr_nonneg _
  have h0 : PrBad H (compDep D Bd DO).V ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
      ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤ 0 := by
    unfold PrBad
    calc _ ≤ Pr (fun _ : Ω × (Fin ((compDep D Bd DO).V.k x) → Coin) ×
          ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)).C x => False) :=
          Pr_mono fun s hs => hG (hacc x Ω P s.1 s.2.1 hs.1)
      _ = 0 := Pr_false
  refine h0.trans ?_
  have hκ : 0 ≤ Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x := by
    unfold Claims.Recursion.κ
    refine add_nonneg ?_ (mul_nonneg (Nat.cast_nonneg _)
      (Finset.sum_nonneg fun s _ => Nums.κ_nonneg _ (hT x s)))
    show 0 ≤ Claims.Flock.κ Bd.deploy _
    unfold Claims.Flock.κ
    have h1 : (0 : ℝ) ≤ 4 * Bd.deploy.εtable (xI (D.record Bd.IVStar) x) :=
      mul_nonneg (by norm_num) (hε (xI (D.record Bd.IVStar) x))
    have h2 : (0 : ℝ) ≤ 2 ^ (-(Bd.deploy.waits (xI (D.record Bd.IVStar) x) : ℝ)) :=
      Real.rpow_nonneg (by norm_num) _
    have h3 : (0 : ℝ) ≤ 2 * (Bd.deploy.k (xI (D.record Bd.IVStar) x) : ℝ) /
        Bd.deploy.cap (xI (D.record Bd.IVStar) x) := by positivity
    have h4 := Real.exp_pos (-(Bd.deploy.k (xI (D.record Bd.IVStar) x) : ℝ) / 8)
    linarith
  refine add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg hκ (Real.sqrt_nonneg _))
    (mul_nonneg (by positivity) ?_)) (mul_nonneg (by positivity) (add_nonneg ?_ (by positivity))))
    (mul_nonneg (by norm_num) (add_nonneg (hF _) (by positivity))))
    (mul_nonneg ?_ (add_nonneg ?_ (by positivity)))
  all_goals first
    | (unfold βdev2; positivity)
    | (unfold RunFinder.PrAt; split_ifs <;> first | exact hF _ | exact le_rfl)
    | (unfold PrBinnerV; split_ifs <;> first | exact hF _ | exact le_rfl)
    | (unfold PrFCI; split_ifs <;> first | exact hF _ | exact le_rfl)

end Instance

open FlockDeploy (Coin FinalP)

/-- **`RecursiveKSEveryAtRec` from `KSLinkOn` at the dense statements**, at every compiled record `recBoundAt` builds:
its verifier accepts only dense statements (`RecDeploy.run_acc_denseAt`) and its sessions' table errors are
nonnegative (`RecDeploy.tableErrNonneg`), so `ksLinkOn_every` lifts `KSLinkOn` at `RecDeploy.DenseAt` (what
`RecoversLinkOn` gives from `ContractDense`) to every statement, where `DE`'s table errors are nonnegative. -/
theorem recursiveKSEveryAtRec_of_dense (b : RecDeploy.Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (rd : Msg → Option FinalP) (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ)
    (hK : KSLinkOn FlockDeploy.sha512 (RecDeploy.recBoundAt _root_.Flock.Readable.relation b L)
      (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
        RecDeploy.rowInDeployed final w)
      (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt)
    (hε : ∀ x, 0 ≤ (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
        RecDeploy.rowInDeployed final w).DE.εtable x) :
    _root_.Flock.Guarantees.RecursiveKSEveryAtRec (RecDeploy.recBoundAt _root_.Flock.Readable.relation b L) rd final
      wE w :=
  ksLinkOn_every _ _ _ _ _ _ hK
    (fun _ _ _ _ _ h => RecDeploy.run_acc_denseAt _ (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck
      { b with rd := L.readers } FlockDeploy.sha512 h)
    (RecDeploy.tableErrNonneg _ _ _) hε

/-- **`RecursiveKSEveryAtRec` from `RecoversLinkOn` at the dense statements and F7's contract there**, at every
compiled record `recBoundAt` builds and every reader whose plain samples are in range (`hIn`): `RecoversLinkOn`'s other
premises hold at the deployment, namely L1 (`compositionKS`), `KSFlockLin` at `DE`
(`RecDeploy.ksFlockLin_R`), each outer session's KS (`outerKS_of_flockKS` at `FlockDeploy.KS.zk_KS_outer` and the
sessions' facts), the sessions' and `DE`'s nonnegative table errors, the record's inner statement, answers that decode
to their coins (`decCoin`), F8 (`RecDeploy.lastCoinFree_ofFlock_R`), `rowIn` injective (`RecDeploy.binds_deployed`) and
the deployed decoders listing within the radius (`listWithin_deployPlainTrim`, `listWithin_deployZk`). -/
theorem recursiveKSEveryAtRec_of_recovers (b : RecDeploy.Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (rd : Msg → Option FinalP) (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hR : RecoversLinkOn FlockDeploy.sha512 (RecDeploy.recBoundAt _root_.Flock.Readable.relation b L)
      (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
        RecDeploy.rowInDeployed final w)
      (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt)
    (hC : ContractDenseAtRec (RecDeploy.recBoundAt _root_.Flock.Readable.relation b L) rd final wE w) :
    _root_.Flock.Guarantees.RecursiveKSEveryAtRec (RecDeploy.recBoundAt _root_.Flock.Readable.relation b L) rd final
      wE w := by
  have hε : ∀ x, 0 ≤ (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w).DE.εtable x :=
    fun x => FlockDeploy.Assembled.εtable_nonneg false x
  refine recursiveKSEveryAtRec_of_dense b L rd final wE w (hR compositionKS
    (RecDeploy.ksFlockLin_R rd wE FlockDeploy.joinR final w hIn) ?_ (RecDeploy.tableErrNonneg _ _ _) (fun _ => rfl)
    ⟨FlockDeploy.decCoin [], fun c => ?_⟩ hε (RecDeploy.lastCoinFree_ofFlock_R _ rd _ (by rfl) _ _)
    (RecDeploy.binds_deployed _ _ _ wE FlockDeploy.joinR final w)
    (Claims.Sampling.listWithin_deployPlainTrim _root_.Flock.Readable.relation)
    (Claims.Sampling.listWithin_deployZk _root_.Flock.Readable.relation) hC) hε
  · let Bd := RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w
    exact outerKS_of_flockKS FlockDeploy.KS.zk_KS_outer (RecDeploy.sessionNums _ _ _ _)
      (RecDeploy.tableErrNonneg _ _ _) (RecDeploy.sessionRounds (Inner.ofFlock Bd.deploy Bd.deploy.decode) _ _ _)
      (RecDeploy.sessionAccepts _ _ _) (RecDeploy.sessionRuns_sha512 (Inner.ofFlock Bd.deploy Bd.deploy.decode) _ _ _)
  · exact (congrArg (FlockDeploy.decCoin []) (RecDeploy.bdDeployed_ans _root_.Flock.Readable.relation
      (FlockDeploy.nextR rd) (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w c)).trans (Function.leftInverse_invFun RecDeploy.coin_bytes_injective c)

/-- **`Flock.Guarantees.RecursiveKSEvery` from `RecoversLinkOn` at the dense statements and F7**
(`ContractDense`), at the deployment's records and values: `recursiveKSEveryAtRec_of_recovers` at the deployed
reader, whose plain samples are in range (`FlockDeploy.KS.plainRdInR`). `RecoversLinkOn` there is F9a from F6's three
term comparisons. -/
theorem recursiveKSEvery_of_recovers
    (hR : ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → _root_.Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → _root_.Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      RecoversLinkOn FlockDeploy.sha512
        (RecDeploy.recBoundAt _root_.Flock.Readable.relation
          { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR))
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
          RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
          RecDeploy.DeployedR.wR)
        (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt)
    (hC : ContractDense) : _root_.Flock.Guarantees.RecursiveKSEvery :=
  fun b aux nOf hd recRest => recursiveKSEveryAtRec_of_recovers _ _ _ _ _ _
    (fun x _ => FlockDeploy.KS.plainRdInR RecDeploy.DeployedR.rdR x) (hR b aux nOf hd recRest)
    (hC b aux nOf hd recRest)

end Claims.RecGeneric
