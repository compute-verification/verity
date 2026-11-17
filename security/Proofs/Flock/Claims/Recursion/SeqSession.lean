module

public import Proofs.Flock.Claims.Recursion.SeqSim
public import Proofs.Flock.Claims.ZK.Deployed
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
import Proofs.Flock.Claims.Recursion.DeployZK
import Proofs.Flock.Claims.ZK.Flock

/-! The V\* sessions of the masked recursion, from Flock's zero knowledge (`zkBoundM_of_tables`). -/

@[expose] public section

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open Claims.ZKMal (Layout ZKmalAt SeqTape progOf progOf_of Proto)
open scoped NNReal

variable (IV : Claims.Verifier Coin) (b : Brings)

theorem VO_k_xsOf (x : Claims.Msg) (j : ℕ) (u : List (Claims.Msg × Claims.Msg)) (hj : j < nS x) :
    VO.k (xsOf IV b x j u) = (nums b.st x j).rounds := by
  simp only [xsOf, vstar_getD _ _ _ hj]
  exact congrArg Claims.Recursion.Nums.rounds (numsOf_vstarAt _ _ _ _)

noncomputable def soOf (SF : Claims.Simulator) (CB : Claims.Msg → ℕ → Type) [∀ x j, Fintype (CB x j)]
    [∀ x j, Nonempty (CB x j)] : Claims.Msg → ℕ → Claims.Recursion.OSim :=
  fun x j => ⟨CB x j, fun xS c => progOf SF (CB x j) xS c⟩

def Shaped (T₀ : SeqTape) (SF : Claims.Simulator) (εF γF : Claims.Msg → ℝ) (CB : Claims.Msg → ℕ → Type)
    (εB : Claims.Msg → ℕ → ℝ) (g : ℝ) (x : Claims.Msg) : Prop :=
  ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
    Nonempty (CB x j ≃ SF.C (xsOf IV b x j u)) ∧ εF (xsOf IV b x j u) ≤ εB x j ∧ γF (xsOf IV b x j u) ≤ g

theorem sessions_of_zkAt {Col TF : Type} (H : Claims.Msg → Claims.Msg) (T₀ : SeqTape)
    (DF : Claims.Flock.Deploy Coin Col TF) {NB ΩB : Type} [Fintype NB] [Nonempty NB] [Fintype ΩB] [Nonempty ΩB]
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (SF : Claims.Simulator) (εF γF : Claims.Msg → ℝ)
    (hF : Claims.Flock.ZKAt H DF PS SF εF γF) (hk : ∀ xS, DF.V.k xS = VO.k xS) (CB : Claims.Msg → ℕ → Type)
    [∀ x j, Fintype (CB x j)] [∀ x j, Nonempty (CB x j)] (εB : Claims.Msg → ℕ → ℝ) (g : ℝ) (x : Claims.Msg)
    (hsh : Shaped IV b T₀ SF εF γF CB εB g x) :
    ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg), u.length = (parts IV b (T₀.N × T₀.Ω)).start x j →
      ZKmalAt H ((nums b.st x j).rounds + 1) PS (Claims.Flock.R DF.parse (xsOf IV b x j u)) (xsOf IV b x j u)
        (soOf SF CB x j).C ((soOf SF CB x j).prog (xsOf IV b x j u)) (εB x j) g := by
  intro j hj u hu
  obtain ⟨hC, hε, hγ⟩ := hsh j hj u hu
  have e := hF.2.at_equiv (xsOf IV b x j u) hC.some
  simp only [hk, VO_k_xsOf IV b x j u hj] at e
  show ZKmalAt H _ PS _ _ (CB x j) (progOf SF (CB x j) (xsOf IV b x j u)) _ _
  rw [progOf_of hC]
  exact e.mono hε hγ

theorem zkBoundM_of_zkAt {Col TF : Type} (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : Claims.Msg → SeqTape) (NB ΩB : Type) [Fintype NB]
    [Nonempty NB] [Fintype ΩB] [Nonempty ΩB] (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (PS : Claims.Flock.Wit → Claims.Prover (NB × ΩB)) (DF : Claims.Flock.Deploy Coin Col TF)
    (SF : ℕ → Claims.Simulator) (εF : ℕ → ℝ≥0 → Claims.Msg → ℝ) (γF : ℕ → Claims.Msg → ℝ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (h₀ : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        ZKmalAt H (kPre IV x) (preP IV b (pre x)) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
          ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
          (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0)
    (hF : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
        Claims.Flock.ZKAt H DF PS (SF t) (εF t δ₁) (γF t))
    (hk : ∀ xS, DF.V.k xS = VO.k xS)
    (hsh : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        Shaped IV b (T₀ x) (SF t) (εF t δ₁) (γF t) (CB t) (fun x j => ((deployRec IV ivScope b).nums x j).εzk t δ₁)
          (4 * t * 3 ^ t) x)
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
      (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Claims.Flock.R DF.parse (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)) :
    ZKBoundM IV b H ivScope I T₀ NB ΩB pre wit PS fun t => soOf (SF t) (CB t) :=
  zkBoundM_of IV b H ivScope I T₀ NB ΩB pre wit PS _ (Claims.Flock.R DF.parse) hI h₀
    (fun t ht δ₁ hδ x => sessions_of_zkAt IV b H (T₀ x) DF PS (SF t) (εF t δ₁) (γF t) (hF t ht δ₁ hδ) hk (CB t) _ _
      x (hsh t ht δ₁ hδ x)) hW

end RecDeploy

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open Claims.ZKMal (Layout ZKmalAt SeqTape progOf progOf_of Proto)
open FlockSoundness.Discharge
open scoped NNReal

theorem VO_k_pos (x : Claims.Msg) : 0 < VO.k x := by
  simp only [VO]
  split <;> omega

theorem deployZkRec_proto_k (rel : Readable.Parsed → Readable.Relation) (G : Proto Claims.Flock.Wit)
    (hK : ∀ x, G.K x = VO.k x - 1) (x : Claims.Msg) : (deployZkRec rel).V.k x + 1 = G.k x := by
  have := VO_k_pos x
  have := hK x
  show VO.k x + 1 = G.K x + 2
  omega

theorem Flock.zkAt_deployZkRec_of_tables_bind (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (t : ℕ) (ht : 0 < t) (hK : ∀ x, G.K x = VO.k x - 1)
    {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤)
    (Mp : ∀ x, G.Nonce → G.TPreModel x) (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x)
    (nL nB : Claims.Msg → ℕ) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Claims.Flock.R (deployZkRec rel).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding H) :
    Claims.Flock.ZKAt H (deployZkRec rel) G.prover ((G.withSim Mc).simulatorE t (t * 3 ^ t))
      (fun x => 8 * nL x * δ₁.toReal + 8 * t * nB x * δ₁.toReal + 2 * (1 / 3) ^ t) fun _ => 4 * t * 3 ^ t :=
  Claims.ZKMal.Flock.zkAt_of_tables_bind H (deployZkRec rel) G t ht rfl (deployZkRec_proto_k rel G hK) hT1 hδ₁ Mp Mc
    nL nB hnL hnB hL hOK hC

variable (IV : Claims.Verifier Coin) (b : Brings)

theorem zkBoundM_of_tables (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (T₀ : Claims.Msg → SeqTape) (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (h₀ : ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        ZKmalAt H (kPre IV x) (preP IV b (pre x)) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
          ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
          (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0)
    (hkey : ∀ δ₁ : ℝ≥0, Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
      Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding H)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
      (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Claims.Flock.R (deployZkRec relO).parse (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)) :
    ZKBoundM IV b H ivScope I T₀ G.Nonce G.Rand pre wit G.prover
      fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t) := by
  refine zkBoundM_of_zkAt IV b H ivScope I T₀ G.Nonce G.Rand pre wit G.prover (deployZkRec relO)
    (fun t => (G.withSim Mc).simulatorE t (t * 3 ^ t))
    (fun t δ₁ x => 8 * nL x * (δ₁ : ℝ) + 8 * t * nB x * (δ₁ : ℝ) + 2 * (1 / 3) ^ t) (fun t _ => 4 * t * 3 ^ t) CB
    hI h₀ (fun t ht δ₁ hδ => ?_) (fun _ => rfl) (fun t ht δ₁ hδ x j hj u hu => ?_) hW
  · have h := Flock.zkAt_deployZkRec_of_tables_bind H relO G t ht hK (hkey δ₁ hδ) ENNReal.coe_ne_top Mp Mc nL nB hnL
      hnB hL hOK hC
    simpa only [ENNReal.coe_toReal] using h
  · obtain ⟨h1, h2⟩ := hzl x j hj u hu
    refine ⟨hCB t ht x j hj u hu, ?_, le_rfl⟩
    beta_reduce
    rw [nums_εzk]
    have hδ0 : (0 : ℝ) ≤ δ₁ := δ₁.2
    have e1 : (nL (xsOf IV b x j u) : ℝ) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 := by exact_mod_cast h1
    have e2 : (nB (xsOf IV b x j u) : ℝ) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2 := by exact_mod_cast h2
    gcongr

end RecDeploy
