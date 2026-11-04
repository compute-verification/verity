module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Claims.Recursion.DeployFacts
public import Proofs.Flock.Claims.Recursion.OuterKS

/-!
# `KSLinkOn` at every statement, from the statements a run accepts

`KSLinkOn` at the statements `G` admits holds at every statement wherever the composite's verifier accepts only at
statements `G` admits: elsewhere no run accepts, so `Pr[accept ∧ ¬extract] = 0`, and the bound is nonnegative where the
tables' errors are (`ksLinkOn_every`). The deployed compiled verifier accepts only dense statements
(`RecDeploy.run_acc_denseAt`, from `RecDeploy.acc_dense`), so `KSLinkOn` at `RecDeploy.DenseAt` gives
`Flock.Guarantees.RecursiveKSEveryAtRec` at every compiled record `recBoundAt` builds (`recursiveKSEveryAtRec_of_dense`).
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

end Claims.RecGeneric
