module

public import Proofs.Flock.RecGeneric.AtFlockCompiledSessions
public import Proofs.Flock.RecGeneric.AtFlockCompiledContract
public import Proofs.Flock.RecGeneric.AtFlockEvery
public import Proofs.Flock.RecGeneric.AtFlockServer
public import Proofs.Flock.RecGeneric.OuterProof

/-!
# L2 at Flock: the compiled recursive verifier's knowledge soundness at every statement

L2 (`compilerKS`, `Definitions.Claims.RecursionGeneric`'s `CompilerKS`) at `Proto.ofFlock`, `Server.ofBound`,
`Arg.ofFlock DO` and
`recCompiler` gives `KSCompiledOn` at the statements where the compiler's contract holds (`ksCompiledOn_of`), from F4
(`protoWFOfFlock`, `kSAbortLinOfFlockLin`), F8, F2 (`bindsOfBound`), Flock's KS as the outer argument's and the
compiler's contract for `E_O` (`SessionsOfRec`). `Comp.err` is nonnegative where the inner protocol is well formed and
`κ_O` is (`Comp.err_nonneg`), and the deployed compiled verifier accepts only dense statements, so `KSCompiledOn` at
`RecDeploy.DenseAt` holds at every statement (`ksCompiledOn_every`). At the deployment every premise but the
compiler's contract holds, so `ContractCompiledDense` gives `Flock.Guarantees.RecursiveKSCompiledEvery`
(`recursiveKSCompiledEvery_of_contract`), and A1d proves it from F1 (`contractCompiledDense`, `ivStarOfBound`,
`openViewOfBound`): `Flock.SecurityProofs.RecursiveKSCompiledEvery`, with no hypothesis.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion
open Claims.SamplingGeneric (Arg)

/-- **`Comp.err` is nonnegative** where the inner protocol is well formed and `κ_O` is nonnegative: every other term is
a probability, a square root, or a nonnegative weight times one. -/
theorem Comp.err_nonneg {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (K : Comp Coin)
    (I : Proto Coin W) (S : Server) (H : Msg → Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (m : Fin I.nT → ℕ) (M : ℕ) (hI : I.WF) (hκO : 0 ≤ K.κO x) : 0 ≤ K.err I S H P x m M := by
  obtain ⟨-, hκ, hw, htail, -⟩ := hI
  have hF : ∀ F : Finder, 0 ≤ F.Pr H := fun F => Pr_nonneg _
  have hPO : 0 ≤ K.PrOut H P x := Pr_nonneg _
  have hreads : 0 ≤ K.reads I x := Finset.sum_nonneg fun i _ => mul_nonneg (hw i _) (Nat.cast_nonneg _)
  have hβ : 0 ≤ K.β I x := mul_nonneg (add_nonneg (Nat.cast_nonneg _) hreads) (Nat.cast_nonneg _)
  have hκK : 0 ≤ K.κ I x := add_nonneg (hκ _) (mul_nonneg (Nat.cast_nonneg _) hκO)
  unfold Comp.err
  refine add_nonneg (add_nonneg (add_nonneg (add_nonneg (add_nonneg hκK (Real.sqrt_nonneg _))
    (Finset.sum_nonneg fun i _ => mul_nonneg (hw i _) (add_nonneg ?_ (htail i _ _)))) (mul_nonneg hreads hPO)) ?_)
    (mul_nonneg hβ (add_nonneg (add_nonneg ?_ (by positivity)) hPO))
  all_goals (unfold RunFinder.PrAt; split_ifs <;> first | exact hF _ | exact le_rfl)

section Instance

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  {Col' T' : Type} [DecidableEq Col']

omit [DecidableEq Col'] in
/-- **L2 at Flock gives `KSCompiledOn` at the statements `G` admits**, the instance of `CompilerKS` at Π_in the
element-level Flock (`Proto.ofFlock` at the deployed decoder), the hm96 server (`Server.ofBound`), Π_out Flock
(`Arg.ofFlock DO`) and today's compiler (`recCompiler`): from L2 (`hL2`), the element-level Flock's KS with no root
(`KSFlockLin` at `DE`), Flock's KS as Π_out's, the compiler's contract for `E_O`, answers that decode to their coins,
`DE`'s nonnegative table errors, F8, `rowIn` injective (`Bound.Binds`), the decoder meeting the list-decoding spec and
the compiler's contract at each statement `G` admits. -/
theorem ksCompiledOn_of [Zero Col] (hL2 : CompilerKS) (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (hKS : KSFlockLin H Bd.DE Cp)
    (hA : (Arg.ofFlock DO).KS H) (hS : SessionsOfRec H D DO ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))
    (hdec : ∃ dec : Msg → Coin, ∀ c, dec (Bd.ans c) = c) (hε : ∀ x, 0 ≤ Bd.DE.εtable x)
    (hLC : (Proto.ofFlock Bd Bd.deploy.decode Cp).LastCoinFree H) (hB : Bd.Binds)
    (hdI : decOK Bd.DE Bd.deploy.decode) (hC : ContractCompiledOn H D Bd DO Cp G) :
    KSCompiledOn H D Bd DO Cp G := by
  intro x hx Ω _ _ P m M
  exact hL2 Coin _ H (Proto.ofFlock Bd Bd.deploy.decode Cp)
    (Server.ofBound Bd) (Arg.ofFlock DO) (recCompiler D DO) (protoWFOfFlock Bd Bd.deploy.decode Cp hdec hε)
    (kSAbortLinOfFlockLin H Bd Cp hKS Bd.deploy.decode hdI) hLC (bindsOfBound Bd hB) hA hS x (hC x hx) Ω P m M

omit [DecidableEq Col] [DecidableEq Col'] in
/-- **`KSCompiledOn` at every statement from `KSCompiledOn` where runs accept**: where every run the compiled verifier
accepts is of a statement `G` admits, the inner protocol is well formed and the sessions' table errors are
nonnegative, elsewhere no run accepts, so the failure is `0`, at most the bound (`Comp.err_nonneg`). -/
theorem ksCompiledOn_every (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (hK : KSCompiledOn H D Bd DO Cp G)
    (hacc : ∀ (x : Msg) (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin ((compiledAt D Bd DO Cp).V.k x) → Coin),
      (run H P (compiledAt D Bd DO Cp).V x ω r).out = .acc → G x)
    (hI : (Proto.ofFlock Bd Bd.deploy.decode Cp).WF)
    (hT : TableErrNonneg (D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans))) :
    KSCompiledOn H D Bd DO Cp fun _ => True := by
  intro x _ Ω _ _ P m M
  by_cases hG : G x
  · exact hK x hG Ω P m M
  refine le_trans ?_ (Comp.err_nonneg _ _ _ H P x m M hI ?_)
  · unfold PrBad
    calc _ ≤ Pr (fun _ : Ω × (Fin ((compiledAt D Bd DO Cp).V.k x) → Coin) ×
          ((compiledAt D Bd DO Cp).toComp.E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)).C x =>
            False) :=
          Pr_mono fun s hs => hG (hacc x Ω P s.1 s.2.1 hs.1)
      _ = 0 := Pr_false
  · show (0 : ℝ) ≤ ∑ s : Fin ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nS x),
      ((D.record ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans)).nums x s).κ
    exact Finset.sum_nonneg fun s _ => Nums.κ_nonneg _ (hT x s)

end Instance

open FlockDeploy (Coin FinalP)

/-- The deployed bound record's answers decode to their coins. -/
theorem ans_dec_deployed (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ) :
    ∃ dec : Msg → Coin, ∀ c, dec ((RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w).ans c) = c :=
  ⟨FlockDeploy.decCoin [], fun c => (congrArg (FlockDeploy.decCoin []) (RecDeploy.bdDeployed_ans
    _root_.Flock.Readable.relation (FlockDeploy.nextR rd) (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation
      rd).cols wE FlockDeploy.joinR RecDeploy.rowInDeployed final w c)).trans
    (Function.leftInverse_invFun RecDeploy.coin_bytes_injective c)⟩

/-- **`RecursiveKSCompiledAtRec` from the compiler's contract at the dense statements**, at every compiled record
`recBoundAt` builds and every reader whose plain samples are in range (`hIn`): `ksCompiledOn_of` at `RecDeploy.DenseAt`
with its other premises at the deployment (`RecDeploy.ksFlockLin_R`, `flockArgKS_deployed`, `sessionsOfRec_deployed`,
`ans_dec_deployed`, `DE`'s nonnegative table errors, F8, `RecDeploy.binds_deployed`, `listWithin_deployPlainTrim`),
lifted to every statement by `ksCompiledOn_every`. -/
theorem recursiveKSCompiledAtRec_of_contract (b : RecDeploy.Brings) (L : Claims.Recursion.VStarB.LayoutB)
    (rd : Msg → Option FinalP) (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) (wE w : Msg → ℝ)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc FlockDeploy.sha512 (FlockDeploy.VR rd) x → FlockDeploy.KS.PlainRdInR rd x)
    (hC : ContractCompiledDenseAtRec (RecDeploy.recBoundAt _root_.Flock.Readable.relation b L) rd final wE w) :
    _root_.Flock.Guarantees.RecursiveKSCompiledAtRec (RecDeploy.recBoundAt _root_.Flock.Readable.relation b L) rd
      final wE w := by
  have hε : ∀ x, 0 ≤ (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w).DE.εtable x :=
    fun x => FlockDeploy.Assembled.εtable_nonneg false x
  have hK := ksCompiledOn_of compilerKS FlockDeploy.sha512 _ _ _ FlockDeploy.KS.CpDep RecDeploy.DenseAt
    (RecDeploy.ksFlockLin_R rd wE FlockDeploy.joinR final w hIn) flockArgKS_deployed
    (sessionsOfRec_deployed _root_.Flock.Readable.relation b L _) (ans_dec_deployed rd final wE w) hε
    (RecDeploy.lastCoinFree_ofFlock_R _ rd _ (by rfl) _ _)
    (RecDeploy.binds_deployed _ _ _ wE FlockDeploy.joinR final w)
    (Claims.Sampling.listWithin_deployPlainTrim _root_.Flock.Readable.relation) hC
  exact ksCompiledOn_every _ _ _ _ _ _ hK
    (fun _ _ _ _ _ h => RecDeploy.run_acc_denseAt _ (FlockDeploy.deployPlain _root_.Flock.Readable.relation).scopeCheck
      { b with rd := L.readers } FlockDeploy.sha512 h)
    (protoWFOfFlock _ _ FlockDeploy.KS.CpDep (ans_dec_deployed rd final wE w) hε) (RecDeploy.tableErrNonneg _ _ _)

/-- **`Flock.Guarantees.RecursiveKSCompiledEvery` from the compiler's contract at the deployment's dense statements**
(`ContractCompiledDense`, A1d): `recursiveKSCompiledAtRec_of_contract` at the deployed reader, whose plain samples are in
range (`FlockDeploy.KS.plainRdInR`). -/
theorem recursiveKSCompiledEvery_of_contract (hC : ContractCompiledDense) :
    _root_.Flock.Guarantees.RecursiveKSCompiledEvery :=
  fun b aux nOf hd recRest => recursiveKSCompiledAtRec_of_contract _ _ _ _ _ _
    (fun x _ => FlockDeploy.KS.plainRdInR RecDeploy.DeployedR.rdR x) (hC b aux nOf hd recRest)

end Claims.RecGeneric

namespace Flock.SecurityProofs

open Claims.RecGeneric

/-- **Knowledge soundness of the recursive verifier compiled over Flock `--zk`, at every statement, at the deployed
reader**: `CompilerKS`'s Flock instance (`ksCompiledOn_of compilerKS`) at the dense statements, lifted to every
statement. -/
theorem RecursiveKSCompiledEvery : Flock.Guarantees.RecursiveKSCompiledEvery :=
  recursiveKSCompiledEvery_of_contract (contractCompiledDense (ivStarOfBound _) (openViewOfBound _))

end Flock.SecurityProofs
