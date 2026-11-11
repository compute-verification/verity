module

public import Proofs.Flock.Recursive.KSNontrivial
public import Proofs.Flock.Claims.Recursion.Assumptions.HonestFixture
public import Proofs.Flock.Claims.Recursion.KSLin2Committed
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Ring.BooleanRing
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
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Recursion.HonestHalfTops
import Proofs.Flock.Claims.Recursion.PlainKSAbortFails
import Proofs.Flock.Claims.Witnesses.RecursionLin2
import Proofs.Flock.Soundness.CROnly.Hiding

/-!
# `RecursiveKSDeployed`'s witnesses (the Lean check's `honest` and `nontrivial`)

`RecursiveKSDeployed` is `RecursiveKS`'s bound at Hello-committed coins (`KSLin2OnC`), so its witnesses are
`RecursiveKS`'s with what the committed run adds: a coin tree whose leaves hide at `δ₁ ≠ ∞`, and a prover that reads
the tree's root and openings.

* **The honest lemma** (`recursiveKSDeployed_honest`), from the one named assumption `RecDeploy.HonestFixture`: at its
  run every V\* part accepts (`honest_half_fixture`'s conclusion), the statement is dense (`DenseAt`, from the run's
  parse), the reference prover, reading neither the root nor the openings (`Claims.Bridge.CProver.ofProver`), is accepted
  at the committed coins of every coin tree (`Claims.Bridge.runCM_ofProver`), the deployed inner and outer decoders meet
  the list-decoding spec (`decOK`), and hm96 leaves under the pinned key hide at its hiding gap, a finite `δ₁`, with no
  premise (`CROnly.hm96Hiding_gap`).
* **The nontrivial lemma** (`recursiveKSDeployed_nontrivial`), from `RecDeploy.KSFixture`: `KSLin2OnC` at `δ₁ = 0` is
  false at the verifier that accepts everything (`RecAt.acceptAll`), at every brings. At the one-salt tree whose leaf is
  its salt, which hides at `0` (`Witness.Hm96Hiding_holds`), the silent prover is accepted on every committed run
  (`Pr_runCM_acc`); the accepting record stages no V\* statement, so the extractor, run on it with any simulator's root
  and openings, recovers nothing (`Pr_E_none`); and the rest of the bound is `not_ksLin2On_acceptAll`'s, below `1`
  (`not_ksLin2OnC_acceptAll`).
-/

@[expose] public section

namespace Claims.Bridge

/-- **A prover against committed coins that reads neither the root nor the openings**: `P`'s rounds. -/
def CProver.ofProver {Ω : Type} (P : Prover Ω) : CProver Ω := ⟨fun x _ hist _ ω => P.next x hist ω⟩

/-- **It runs as `P`**: at every coin tree and its randomness, the committed run of `CProver.ofProver P` is `P`'s run
at the same tape and coins. -/
theorem runCM_ofProver {Coin Ω T : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin)
    (x : Msg) (C : CoinTree Coin T (V.k x)) (ω : Ω) (r : Fin (V.k x) → Coin) (τ : T) :
    runCM H (CProver.ofProver P) V x C ω r τ = run H P V x ω r := rfl

end Claims.Bridge

namespace Claims.Recursion

section Committed

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- At a verifier that accepts at its first round, every committed run accepts. -/
theorem Pr_runCM_acc {Ω T₀ : Type} [Fintype Ω] [Nonempty Ω] [Fintype T₀] [Nonempty T₀] (H : Msg → Msg)
    (P : Bridge.CProver Ω) (V : Verifier Coin) (x : Msg) (C : Bridge.CoinTree Coin T₀ (V.k x))
    (hv : ∀ as cs, V.next x as cs = .ret .acc) (hk : 0 < V.k x) :
    Pr (fun p : Ω × (Fin (V.k x) → Coin) × T₀ => (Bridge.runCM H P V x C p.1 p.2.1 p.2.2).out = .acc) = 1 := by
  have h : ∀ p : Ω × (Fin (V.k x) → Coin) × T₀, (Bridge.runCM H P V x C p.1 p.2.1 p.2.2).out = .acc := fun p =>
    run_acc H (P.atM (C.hello p.2.1 p.2.2) (Bridge.openSeq (C.opn p.2.1 p.2.2))) V x hv hk p.1 p.2.1
  exact (Pr_congr fun p => iff_true_intro (h p)).trans (Pr_const_of trivial)

/-- Where the record stages no V\* statement and the outer parser refuses the empty one, `E*` recovers no witness on
any run of any prover (`E_none_of_vstar_nil`). -/
theorem Pr_E_none {W : Type} (H : Msg → Msg) (A : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (x : Msg)
    (hs : ∀ ps, A.vstar x ps = []) (hS : 0 < A.nS x) (hP : DO.parse [] = none) {Ω : Type} [Fintype Ω]
    (P : Prover Ω) :
    Pr (fun s : Ω × (Fin (A.V.k x) → Coin) × (E H A I DO decO).C x =>
      (Recursion.R A I).ext x ((E H A I DO decO).eval H P x s.1 s.2.1 s.2.2).1) = 0 := by
  refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
  rw [E_none_of_vstar_nil H A I DO decO x hs hS hP P s.1 s.2.1 s.2.2]
  exact id

/-- **`KSLin2OnC` fails at the verifier that accepts everything**, at `δ₁ = 0`, under `not_ksLin2On_acceptAll`'s
premises: at the one-salt tree whose leaf is its salt, which hides at `0` (`Witness.Hm96Hiding_holds`), the silent
prover is accepted on every committed run (`Pr_runCM_acc`), and the extractor, run on it with any simulator's root and
openings, recovers nothing (`Pr_E_none`), so the bound's left side is `1` and its first term `0`; the rest is
`KSLin2On`'s bound at that prover, below `1` as in `not_ksLin2On_acceptAll`, and `2 · k(x) · δ₁ = 0`. -/
theorem not_ksLin2OnC_acceptAll [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (decI : List (ℕ × Col) → List T)
    (hdI : decOK Bd.deploy decI) (decO : List (ℕ × Col') → List T') (hdO : decOK DO decO) (x : Msg) (hx : G x)
    (hk : 0 < (D.record Bd.IVStar).V.k x) (hS : 0 < (D.record Bd.IVStar).nS x) (hP : DO.parse [] = none)
    (hκ : κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x < 1 / 2) :
    ¬ KSLin2OnC H D.acceptAll Bd DO Cp G 0 := by
  intro h
  have hc : ∀ y (r : Fin ((D.acceptAll.record Bd.IVStar).V.k y) → Coin) ws,
      (D.acceptAll.record Bd.IVStar).callsAt H y r ws = [] := fun _ _ _ => rfl
  have hs : ∀ y ps, (D.acceptAll.record Bd.IVStar).vstar y ps = [] := fun _ _ => rfl
  have hv : ∀ as cs, (D.acceptAll.record Bd.IVStar).V.next x as cs = .ret .acc := fun _ _ => rfl
  have hk' : 0 < (D.acceptAll.record Bd.IVStar).V.k x := hk
  have hS' : 0 < (D.acceptAll.record Bd.IVStar).nS x := hS
  have hκ' : κ (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x < 1 / 2 := hκ
  obtain ⟨S, hSim⟩ := h ENNReal.zero_ne_top decI hdI decO hdO x hx Unit Unit Unit Unit
    ⟨fun _ _ => (), id, fun _ => (), fun _ _ => [], fun _ _ _ _ _ => []⟩ Witness.Hm96Hiding_holds
  have h1 := hSim Unit ⟨fun _ _ _ _ _ => .ret []⟩
    ⌈4 * (4 * (Cp.width (xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x)⌉₊
    (16 * ((Inner.ofFlock Bd.deploy decI).k (xI (D.acceptAll.record Bd.IVStar) x) + 1))
  rw [Pr_runCM_acc H _ _ x _ hv hk', Pr_E_none H _ _ DO decO x (hs x) hS' hP] at h1
  unfold lin2Rhs at h1
  rw [prB_acc H _ _ x hv, Witness.Recursion.prBinnerV_noCalls H (D.acceptAll.record Bd.IVStar) hc hs,
    prFCI_noCalls H (D.acceptAll.record Bd.IVStar) hc,
    Witness.Recursion.prF0I_noCalls H (D.acceptAll.record Bd.IVStar) hc,
    Witness.Recursion.prFdevV_noCalls H (D.acceptAll.record Bd.IVStar) hc hs, ENNReal.toReal_zero] at h1
  simp only [mul_zero, Real.sqrt_zero, zero_add, add_zero] at h1
  have hT := Witness.Recursion.tail_lt_quarter
    (4 * (Cp.width (xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x)
  have hF := Witness.Recursion.f0_tail_lt
    ((Inner.ofFlock Bd.deploy decI).k (xI (D.acceptAll.record Bd.IVStar) x))
  linarith

end Committed

end Claims.Recursion

namespace RecDeploy

open Claims (Msg)

/-- **`KSLin2OnC` at the verifier that accepts everything fails** at `δ₁ = 0`, at any statement of `G` with a V\*
session where `κ < 1/2`, at the deployed decoders: `not_ksLin2On_acceptAll`'s instance of
`Claims.Recursion.not_ksLin2OnC_acceptAll`. -/
theorem not_ksLin2OnC_acceptAll (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (Cp : Claims.Recursion.Caps FlockDeploy.Coin) (G : Msg → Prop) (x : Msg) (hx : G x) (hS : 0 < nS x)
    (hκ : Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy Claims.GS.plainDecodeRD)
      x < 1 / 2) :
    ¬ Claims.Recursion.KSLin2OnC FlockDeploy.sha512 (recBoundAt rel b L).acceptAll
      (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO) Cp G 0 :=
  Claims.Recursion.not_ksLin2OnC_acceptAll _ _ _ _ _ _ Claims.GS.plainDecodeRD (FlockDeploy.listWithin_plainTrim rel)
    Claims.GS.zkDecodeRD (Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl) x hx
    (recBound_k_pos rel b L nextE colsE wE join rowIn final w x) hS (zkRec_parse_nil relO) hκ

end RecDeploy

namespace RecDeploy.HonestHalfFixture

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest
open RecDeploy.HonestHalf

/-- **`RecursiveKSDeployed`'s honest lemma**: at `HonestFixture`'s run, the deployed recursive run at the deployed
brings at `x` under `deployedB`, every V\* part accepts (`honest_half_tops`, as `honest_half_fixture`); the statement is
dense, where `RecursiveKSDeployed` reads it (`DenseAt`); the reference prover, reading neither the root nor the openings,
is accepted at the run's tape and coins committed in every coin tree (`Claims.Bridge.runCM_ofProver`); the deployed
inner and outer decoders meet the list-decoding spec; and hm96 leaves under the pinned key hide at its hiding gap, a
finite `δ₁`, with no premise (`CROnly.hm96Hiding_gap`). -/
theorem recursiveKSDeployed_honest (h : HonestFixture) :
    ∃ (mask : ℕ → ℕ → Bool) (Ir : InnerRec) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
      (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
      (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
      (w : Claims.Flock.Wit) (ω : Tape (bLAt mask Ir b0 x aux nOf hd recRest))
      (r : Fin ((DAt mask Ir b0 x aux nOf hd recRest).V.k x) → Coin),
      (∀ s < nS x,
        Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 (DAt mask Ir b0 x aux nOf hd recRest)
          (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR OpenRows.elemR) Claims.Recursion.Form.Dense
          (showsPart recRead ({ bringsAt mask Ir (deployedB b0) x with tr := recTr aux nOf hd recRest } : Brings))
          (cmtTr recRead) s x (Claims.coinSeq r)
          ((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w)
            (DAt mask Ir b0 x aux nOf hd recRest).V x ω r).msgs.map Prod.fst)) ∧
      DenseAt x ∧
      (∀ (T : Type) (C : Claims.Bridge.CoinTree Coin T ((DAt mask Ir b0 x aux nOf hd recRest).V.k x)) (τ : T),
        (Claims.Bridge.runCM FlockDeploy.sha512
          (Claims.Bridge.CProver.ofProver (refP (IAt decI).V (bLAt mask Ir b0 x aux nOf hd recRest) w))
          (DAt mask Ir b0 x aux nOf hd recRest).V x C ω r τ).out = .acc) ∧
      Claims.Recursion.decOK DeployedR.bdR.deploy Claims.GS.plainDecodeRD ∧
      Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) Claims.GS.zkDecodeRD ∧
      Claims.Hm96Hiding FlockSoundness.Discharge.ZkView.My FlockSoundness.Discharge.ZkView.saltHash
        (ENNReal.ofReal (FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
          FlockSoundness.Discharge.ZkView.saltHash)) ∧
      ENNReal.ofReal (FlockSoundness.CROnly.hidingGap FlockSoundness.Discharge.ZkView.My
        FlockSoundness.Discharge.ZkView.saltHash) ≠ ⊤ := by
  obtain ⟨mask, Ir, b0, x, aux, nOf, hd, recRest, decI, w, st, ω, r, hR, hW, hN, hI, hT⟩ := h
  refine ⟨mask, Ir, b0, x, aux, nOf, hd, recRest, decI, w, ω, r,
    HonestHalfTops.honest_half_tops mask Ir b0 x aux nOf hd recRest decI ω r hR hW hN hI hT, fun s hs => ?_,
    fun _ C τ => ?_, FlockDeploy.listWithin_plainTrim _root_.Flock.Readable.relation,
    Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl, FlockSoundness.CROnly.hm96Hiding_gap _ _,
    ENNReal.ofReal_ne_top⟩
  · obtain rfl : st = s := Option.some.inj (hR.2.1.symm.trans hs)
    exact hR.2.2.1
  · rw [Claims.Bridge.runCM_ofProver]
    exact hR.2.2.2

end RecDeploy.HonestHalfFixture

namespace Flock.Guarantees

open Claims (Msg Out)

/-- **`RecursiveKSDeployed`'s nontrivial lemma**: its conclusion, `KSLin2OnC` at the deployed values, is false at the
verifier that accepts everything (`RecAt.acceptAll` of the deployment's compiled record), at `δ₁ = 0` and every brings,
from the named assumption `RecDeploy.KSFixture`, as `recursiveKS_nontrivial` is for `RecursiveKS`: the statement is
dense (`denseAt_of_parses`), in the plain scope (`plainIn_of_parses`) and has a V\* session, and `λ₂ ≥ 100` there
(`RecDeploy.lamLin2_deployed_ge_dep`) gives `κ < 1/2`; at the one-salt tree that hides at `0`, the silent prover is
accepted on every committed run and the extractor recovers nothing (`RecDeploy.not_ksLin2OnC_acceptAll`). -/
theorem recursiveKSDeployed_nontrivial (h : RecDeploy.KSFixture) :
    ∃ (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
      (recRest : Msg → List (Msg × Out) → List UInt8), ∀ b : RecDeploy.Brings,
      ¬ Claims.Recursion.KSLin2OnC FlockDeploy.sha512
        (RecDeploy.recBoundAt _root_.Flock.Readable.relation
          { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)).acceptAll
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
          RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
          RecDeploy.DeployedR.wR)
        (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt 0 := by
  obtain ⟨x, aux, nOf, hd, recRest, hP, hO, hn, hS⟩ := h
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_parses hP
  refine ⟨aux, nOf, hd, recRest, fun b => ?_⟩
  have hO' : ∀ j < RecDeploy.nS x, ∃ i spec' sts', RecDeploy.PartIn
      (RecDeploy.staged ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } : RecDeploy.Brings).st
        x (RecDeploy.pick ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } :
          RecDeploy.Brings).st x j) j) i spec' sts' := hO
  exact RecDeploy.not_ksLin2OnC_acceptAll _root_.Flock.Readable.relation _root_.Flock.Readable.relation
    { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
    (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols RecDeploy.DeployedR.wER
    FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wR FlockDeploy.KS.CpDep
    RecDeploy.DenseAt x (denseAt_of_parses hP) hP.2.1
    (Claims.Recursion.κ_lt_half_of_lamLin2
      (RecDeploy.lamLin2_deployed_ge_dep _root_.Flock.Readable.relation
        { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR) (FlockDeploy.nextR RecDeploy.DeployedR.rdR)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation RecDeploy.DeployedR.rdR).cols
        RecDeploy.DeployedR.wER FlockDeploy.joinR RecDeploy.rowInDeployed RecDeploy.DeployedR.finalR
        RecDeploy.DeployedR.wR Claims.GS.plainDecodeRD hin hO' hn hS (n := 0) (by norm_num) (v := 0) (by norm_num)))

end Flock.Guarantees
