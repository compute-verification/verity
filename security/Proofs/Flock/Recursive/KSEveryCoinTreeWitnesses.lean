module

public import Proofs.Flock.Recursive.KSEveryCoinTreeStatement
public import Proofs.Flock.Recursive.KSCoinTreeWitnesses
public import Proofs.Flock.RecGeneric.AtFlockNontrivial
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# `RecursiveKSEveryDeployed`'s witnesses (the Lean check's `honest` and `nontrivial`)

`RecursiveKSEveryDeployed` is `RecursiveKSEvery`'s bound at Hello-committed coins (`KSLinkOnC`), so its witnesses are
`RecursiveKSEvery`'s with what the committed run adds, as `RecursiveKSDeployed`'s are `RecursiveKS`'s:

* **The honest lemma** (`recursiveKSEveryDeployed_honest`), from `RecDeploy.HonestFixture`: every V\* part accepts at
  its run, the reference prover is accepted at the committed coins of every coin tree, and hm96 leaves under the pinned
  key hide at a finite `δ₁` (the conjuncts of `recursiveKSDeployed_honest` that `KSLinkOnC` reads).
* **The nontrivial lemma** (`recursiveKSEveryDeployed_nontrivial`), from `RecDeploy.KSFixture`: `KSLinkOnC` at `δ₁ = 0`
  is false at the verifier that accepts everything (`not_ksLinkOnC_acceptAll`). At the one-salt tree that hides at `0`
  the silent prover is accepted on every committed run, `E*` run on it with any simulator's root and openings extracts
  nothing, and the rest of the bound is `not_ksLinkOn_acceptAll`'s, below `1`.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- Where `E_O` gives no witnesses at any run of `P`, `E*` run on `P` recovers no witness. -/
theorem Pr_compE_none {W : Type} (K : Comp Coin) (I : Proto Coin W) (S : Server) (H : Msg → Msg) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (R : Rel W)
    (hO : ∀ ω d c, ((K.EO.ext x d c).eval H (run H P K.V x ω)).1 = none) :
    Pr (fun s : Ω × (Fin (K.V.k x) → Coin) × (K.E I S).C x => R.ext x ((K.E I S).eval H P x s.1 s.2.1 s.2.2).1) =
      0 := by
  refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
  rw [Comp.E_none_of_EO_none K I S H P x s.1 s.2.1 s.2.2 (hO s.1)]
  exact id

variable {Col T Col' T' : Type} [DecidableEq Col] [DecidableEq Col']

omit [DecidableEq Col] in
/-- **`KSLinkOnC` fails at the verifier that accepts everything**, at `δ₁ = 0`, under `not_ksLinkOn_acceptAll`'s
premises: at the one-salt tree whose leaf is its salt, which hides at `0` (`Witness.Hm96Hiding_holds`), the silent
prover is accepted on every committed run (`Pr_runCM_acc`), and `E*`, run on it with any simulator's root and openings,
recovers nothing (`Pr_compE_none`), so the bound's left side is `1` and its first term `0`; the rest is `KSLinkOn`'s
bound at that prover, below `1` as in `not_ksLinkOn_acceptAll`, and `2 · k(x) · δ₁ = 0`. -/
theorem not_ksLinkOnC_acceptAll [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (x : Msg) (hx : G x)
    (hk : 0 < (D.record Bd.IVStar).V.k x) (hS : 0 < (D.record Bd.IVStar).nS x) (hP : DO.parse [] = none)
    (hκ : Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x < 1 / 2) :
    ¬ KSLinkOnC H D.acceptAll Bd DO Cp G 0 := by
  intro h
  have hc : ∀ y (r : Fin ((D.acceptAll.record Bd.IVStar).V.k y) → Coin) ws,
      (D.acceptAll.record Bd.IVStar).callsAt H y r ws = [] := fun _ _ _ => rfl
  have hs : ∀ y ps, (D.acceptAll.record Bd.IVStar).vstar y ps = [] := fun _ _ => rfl
  have hv : ∀ as cs, (D.acceptAll.record Bd.IVStar).V.next x as cs = .ret .acc := fun _ _ => rfl
  have hvC : ∀ as cs, (compDep D.acceptAll Bd DO).V.next x as cs = .ret .acc := fun _ _ => rfl
  have hkC : 0 < (compDep D.acceptAll Bd DO).V.k x := hk
  have hS' : 0 < (D.acceptAll.record Bd.IVStar).nS x := hS
  have hκ' : Claims.Recursion.κ (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x < 1 / 2 := hκ
  have hO : ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (ω : Ω) d c,
      (((compDep D.acceptAll Bd DO).EO.ext x d c).eval H (run H P (compDep D.acceptAll Bd DO).V x ω)).1 = none :=
    fun _ _ d c => EO_none_of_vstar_nil H _ (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode x (hs x) hS' hP _ d c
  obtain ⟨S, hSim⟩ := h ENNReal.zero_ne_top x hx Unit Unit Unit Unit
    ⟨fun _ _ => (), id, fun _ => (), fun _ _ => [], fun _ _ _ _ _ => []⟩ Witness.Hm96Hiding_holds
  have h1 := hSim Unit ⟨fun _ _ _ _ _ => .ret []⟩
    ⌈4 * (4 * (Cp.width (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x)⌉₊
    (16 * ((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x) + 1))
  have hL : ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (M : ℕ),
      (linkPlan (Coin := Coin) Bd.DE.V.k M).PrAt H
        ((compDep D.acceptAll Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x)
        ((compDep D.acceptAll Bd DO).xI x) = 0 := fun P M =>
    linkPlan_PrAt_noAcc H _ M _ _ fun ω u hu => by
      simp [Comp.verModel, Comp.wits, Extractor.eval, hO P] at hu
  rw [Claims.Recursion.Pr_runCM_acc H _ _ x _ hvC hkC,
    Pr_compE_none _ _ _ H _ x _ fun ω d c => hO _ ω d c] at h1
  unfold linkRhs at h1
  rw [prB_acc H _ _ x hv, Claims.Witness.Recursion.prBinnerV_noCalls H (D.acceptAll.record Bd.IVStar) hc hs,
    prFCI_noCalls H (D.acceptAll.record Bd.IVStar) hc,
    Claims.Witness.Recursion.prF0I_noCalls H (D.acceptAll.record Bd.IVStar) hc, hL, ENNReal.toReal_zero] at h1
  simp only [mul_zero, Real.sqrt_zero, zero_add, add_zero] at h1
  have hT := Claims.Witness.Recursion.tail_lt_quarter
    (4 * (Cp.width (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x)
  have hF := Claims.Witness.Recursion.f0_tail_lt
    ((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (Claims.Recursion.xI (D.acceptAll.record Bd.IVStar) x))
  linarith

end Claims.RecGeneric

namespace RecDeploy.HonestHalfFixture

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest
open RecDeploy.HonestHalf

/-- **`RecursiveKSEveryDeployed`'s honest lemma**: at `HonestFixture`'s run, every V\* part accepts, the reference
prover, reading neither the root nor the openings, is accepted at the run's tape and coins committed in every coin
tree, and hm96 leaves under the pinned key hide at its hiding gap, a finite `δ₁`: `recursiveKSDeployed_honest` without
the statement's density and the decoders' spec, which `RecursiveKSEveryDeployed` does not read. -/
theorem recursiveKSEveryDeployed_honest (h : HonestFixture) :
    ∃ (mask : ℕ → ℕ → Bool) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
      (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
      (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
      (w : Claims.Flock.Wit) (ω : Tape (bLAt mask (IrDep x) b0 x aux nOf hd recRest))
      (r : Fin ((DAt mask (IrDep x) b0 x aux nOf hd recRest).V.k x) → Coin),
      (∀ s < nS x,
        Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 (DAt mask (IrDep x) b0 x aux nOf hd recRest)
          (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR OpenRows.elemR) Claims.Recursion.Form.Dense
          (showsPart recRead
            ({ bringsAt mask (IrDep x) (deployedB b0) x with tr := recTr aux nOf hd recRest } : Brings))
          (cmtTr recRead) s x (Claims.coinSeq r)
          ((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask (IrDep x) b0 x aux nOf hd recRest) w)
            (DAt mask (IrDep x) b0 x aux nOf hd recRest).V x ω r).msgs.map Prod.fst)) ∧
      (∀ (T : Type) (C : Claims.Bridge.CoinTree Coin T ((DAt mask (IrDep x) b0 x aux nOf hd recRest).V.k x))
        (τ : T),
        (Claims.Bridge.runCM FlockDeploy.sha512
          (Claims.Bridge.CProver.ofProver (refP (IAt decI).V (bLAt mask (IrDep x) b0 x aux nOf hd recRest) w))
          (DAt mask (IrDep x) b0 x aux nOf hd recRest).V x C ω r τ).out = .acc) ∧
      Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash
        (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
          FlockSoundness.Discharge.ZkView.saltHash)) ∧
      ENNReal.ofReal (FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
        FlockSoundness.Discharge.ZkView.saltHash) ≠ ⊤ := by
  obtain ⟨mask, b0, x, aux, nOf, hd, recRest, decI, w, ω, r, hA, -, hC, -, -, hH, hT⟩ :=
    recursiveKSDeployed_honest h
  exact ⟨mask, b0, x, aux, nOf, hd, recRest, decI, w, ω, r, hA, hC, hH, hT⟩

end RecDeploy.HonestHalfFixture

namespace Flock.Guarantees

open Claims (Msg Out)

/-- **`RecursiveKSEveryDeployed`'s nontrivial lemma**: its conclusion, `KSLinkOnC` at the deployed values, is false at
the verifier that accepts everything (`RecAt.acceptAll` of the deployment's compiled record), at `δ₁ = 0` and every
brings, from the named assumption `RecDeploy.KSFixture`, as `recursiveKSEvery_nontrivial` is for `RecursiveKSEvery`:
the statement has a V\* session whose inner statement is in the plain scope, and `λ₂ ≥ 100` at the deployed decoders
gives `κ < 1/2`; at the one-salt tree that hides at `0`, the silent prover is accepted on every committed run and `E*`
extracts nothing (`Claims.RecGeneric.not_ksLinkOnC_acceptAll`). -/
theorem recursiveKSEveryDeployed_nontrivial (h : RecDeploy.KSFixture) :
    ∃ (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
      (recRest : Msg → List (Msg × Out) → List UInt8), ∀ b : RecDeploy.Brings,
      ¬ Claims.RecGeneric.KSLinkOnC FlockDeploy.sha512
        (RecDeploy.recBoundAt _root_.Flock.Readable.relation
          { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)).acceptAll
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
          RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
          RecDeploy.DeployedR.wR)
        (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep (fun _ => True) 0 := by
  obtain ⟨x, aux, nOf, hd, recRest, hP, hO, hn, hS⟩ := h
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_parses hP
  refine ⟨aux, nOf, hd, recRest, fun b => ?_⟩
  have hO' : ∀ j < RecDeploy.nS x, ∃ i spec' sts', RecDeploy.PartIn
      (RecDeploy.staged ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } : RecDeploy.Brings).st
        x (RecDeploy.pick ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } :
          RecDeploy.Brings).st x j) j) i spec' sts' := hO
  let Bd := RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols RecDeploy.DeployedR.wER
    FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wR
  let D := RecDeploy.recBoundAt _root_.Flock.Readable.relation
    { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
    (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)
  exact Claims.RecGeneric.not_ksLinkOnC_acceptAll FlockDeploy.sha512 D Bd
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep (fun _ => True) x trivial
    (RecDeploy.recBound_k_pos _ _ _ _ _ _ _ _ _ _ x) hP.2.1 (RecDeploy.zkRec_parse_nil _)
    (Claims.Recursion.κ_lt_half_of_lamLin2
      (RecDeploy.lamLin2_deployed_ge_dep _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
        RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
        RecDeploy.DeployedR.wR Bd.deploy.decode hin hO' hn hS (n := 0) (by norm_num) (v := 0) (by norm_num)))

end Flock.Guarantees
