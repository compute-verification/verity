import Proofs.Flock.Recursive.KSNontrivial
import Proofs.Flock.Claims.Recursion.HresAsm

/-!
# The recursive verifier's knowledge soundness at the linked relation

`RecursiveKSLinkAt` is `RecursiveKSAt` with the linked relation `R_L` (the deployed relation, and every listed link's
reader row equal to its writer row), the extractor that also reads the link table's rows off its judged draw, and one
more term, the link finder's `Pr[B_L]`. `recursiveKSLink_of` proves it at the deployed values from `RecursiveKS`, the
deployed chain at `R` (`RecDeploy.ResLower.recursiveKS_deployed`), and the links' binding at the deployment
(`LinksBindDeployed`: on an accepted run whose extracted witness is in `R` and fails the link clause, two of the inputs
the link finder reads collide), by the union bound `Claims.Recursion.ksLin2LinkOn_of`. `recursiveKSLink_nontrivial` is
its conclusion's nontrivial lemma, at any link clause, from `RecDeploy.KSFixture`, as `recursiveKS_nontrivial`'s.
-/

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **The links bind at the deployment**: at every brings, coin-server record, pair of decoders meeting the
list-decoding spec and dense statement, `LinkBindsAt` at the deployed record, bound and outer Flock. -/
def LinksBindDeployed {Tb : Type} (Q : Msg → Claims.Flock.Wit → Tb → Prop) (rows : Msg → List Claims.Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Claims.Flock.Wit) → Claims.Flock.Wit → Tb →
      List Msg) : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) decI,
    Claims.Recursion.decOK (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols RecDeploy.DeployedR.wER
      FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wR).deploy decI →
    ∀ decO, Claims.Recursion.decOK (RecDeploy.deployZkRec _root_.Flock.Readable.relation) decO → ∀ x, RecDeploy.DenseAt x →
    Claims.Recursion.LinkBindsAt FlockDeploy.sha512
      ((RecDeploy.recBoundAt _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)).record
          (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
            (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
            RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
            RecDeploy.DeployedR.wR).IVStar)
      (Claims.Recursion.Inner.ofFlock
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
          RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
          RecDeploy.DeployedR.wR).deploy decI)
      (RecDeploy.deployZkRec _root_.Flock.Readable.relation) decO Q rows ins x

/-- **Knowledge soundness at the linked relation, from `RecursiveKS` and the links' binding**: `RecursiveKS` (the
deployed chain at `R`, reused as it stands) and `LinksBindDeployed` give `RecursiveKSLinkAt` at the deployed values,
by the union bound (`Claims.Recursion.ksLin2LinkOn_of`). -/
theorem recursiveKSLink_of {Tb : Type} {Q : Msg → Claims.Flock.Wit → Tb → Prop} {rows : Msg → List Claims.Flock.Wit → Tb}
    {ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Claims.Flock.Wit) → Claims.Flock.Wit → Tb →
      List Msg}
    (hL : LinksBindDeployed Q rows ins) :
    ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
      (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
      RecursiveKSLinkAt RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.OpenRows.elemR
        RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR (RecDeploy.deployedB b) aux nOf hd recRest Q rows ins :=
  fun b aux nOf hd recRest =>
    Claims.Recursion.ksLin2LinkOn_of (RecDeploy.ResLower.recursiveKS_deployed b aux nOf hd recRest)
      (hL b aux nOf hd recRest)

/-- **`RecursiveKSLinkAt`'s nontrivial lemma**, at any link clause, table reader and finder inputs: its conclusion,
`RecursiveKSLinkAtRec` at the deployed values, is false at the verifier that accepts everything, at every brings, from
`RecDeploy.KSFixture`. At the fixture's statement `E*` extracts nothing (the accepting record stages no V\* statement),
so the link finder finds nothing (`Claims.Recursion.prBL_of_E_none`), `KSLin2On` holds there
(`Claims.Recursion.ksLin2On_of_linkOn`), and that is false as `recursiveKS_nontrivial` shows. -/
theorem recursiveKSLink_nontrivial {Tb : Type} (Q : Msg → Claims.Flock.Wit → Tb → Prop)
    (rows : Msg → List Claims.Flock.Wit → Tb)
    (ins : (Msg → Msg) → Msg → List ((ℕ → Coin) × List (Msg × Out) × List Claims.Flock.Wit) → Claims.Flock.Wit → Tb →
      List Msg)
    (h : RecDeploy.KSFixture) :
    ∃ (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
      (recRest : Msg → List (Msg × Out) → List UInt8), ∀ b : RecDeploy.Brings,
      ¬ RecursiveKSLinkAtRec
        (RecDeploy.recBoundAt _root_.Flock.Readable.relation
          { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)).acceptAll
        RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR Q rows ins := by
  obtain ⟨x, aux, nOf, hd, recRest, hP, hO, hn, hS⟩ := h
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_parses hP
  refine ⟨aux, nOf, hd, recRest, fun b hL => ?_⟩
  have hO' : ∀ j < RecDeploy.nS x, ∃ i spec' sts', RecDeploy.PartIn
      (RecDeploy.staged ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } : RecDeploy.Brings).st
        x (RecDeploy.pick ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } :
          RecDeploy.Brings).st x j) j) i spec' sts' := hO
  refine RecDeploy.not_ksLin2On_acceptAll _root_.Flock.Readable.relation _root_.Flock.Readable.relation
    { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
    (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols RecDeploy.DeployedR.wER
    FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wR FlockDeploy.KS.CpDep
    (fun y => y = x) x rfl hP.2.1
    (Claims.Recursion.κ_lt_half_of_lamLin2
      (RecDeploy.lamLin2_deployed_ge_dep _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
        RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
        RecDeploy.DeployedR.wR Claims.GS.plainDecodeRD hin hO' hn hS (n := 0) (by norm_num) (v := 0) (by norm_num)))
    (Claims.Recursion.ksLin2On_of_linkOn hL (fun y (hy : y = x) => hy ▸ denseAt_of_parses hP) ?_)
  intro decI _ decO _ y hy Ω _ _ P
  change y = x at hy
  subst hy
  refine Claims.Recursion.prBL_of_E_none _ _ _ _ _ _ _ P y fun ω r₀ c => ?_
  refine Claims.Recursion.E_none_of_vstar_nil _ _ _ _ decO y ?_ ?_ ?_ P ω r₀ c
  · exact fun _ => rfl
  · exact hP.2.1
  · exact RecDeploy.zkRec_parse_nil _

end Flock.Guarantees
