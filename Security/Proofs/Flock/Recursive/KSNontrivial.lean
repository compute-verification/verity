import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.VCallsAt

/-!
# `RecursiveKS` at the verifier that accepts everything (LEAN-66's `nontrivial`)

LEAN-66 asks of each guarantee that states what an acceptance means a lemma that its conclusion is false at the
verifier that accepts everything. For `RecursiveKS` that verifier is the compiled recursive verifier of the compiled
record, `(D.record IV).V`, whose acceptance `PrBad` reads, replaced by one that keeps its rounds and accepts at its first
round with no SHA-512 call (`RecAt.acceptAll`); the parser, the inner view, `IVStar`, the bound record, the outer
record, the caps and the dense statements stay deployed. `RecursiveKSAtAll` is `RecursiveKSAt` there.

The bound's right-hand side is at least the recursion's knowledge error `κ`, whatever the verifier
(`ksLin2On_of_one_le_κ`). At a statement whose bytes don't read (`RecDeploy.readX x = none`), `κ` is the inner Flock's at
the empty inner statement, which sets up no table, so `κ ≥ e⁰ = 1` (`κ_acceptAll_of_readX_none`) and the bound holds at
every verifier there, the one that accepts everything included (`ksLin2On_acceptAll_readX_none`). The deployed parser
reads a statement only through `jsonOf` (`Lean.Json.Parser.anyCore`, a `partial def`, which the kernel can't unfold), so
no statement is provably read, and wherever none is, `RecursiveKSAtAll` holds (`recursiveKSAtAll_of_readX_none`).
-/

namespace Claims.Recursion

/-- **The compiled records with the verifier that accepts everything**: each record's verifier keeps its rounds and
accepts at its first round with no SHA-512 call. The parser, the inner view and every other field are `D`'s. -/
def RecAt.acceptAll {Coin : Type} [Inhabited Coin] (D : RecAt Coin) : RecAt Coin where
  record IV := { D.record IV with V := ⟨(D.record IV).V.k, fun _ _ _ => .ret .acc⟩ }
  msgs := D.msgs
  xI := D.xI
  runs := D.runs

section Bound

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

theorem PrFCI_nonneg (H : Msg → Msg) (D : Deploy Coin) {W : Type} (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg)
    (M : ℕ) : 0 ≤ PrFCI H D I DO decO Cp P x M := by
  unfold PrFCI; split
  · exact Pr_nonneg _
  · exact le_rfl

theorem βdev2_nonneg (D : Deploy Coin) {W : Type} (I : Inner Coin W) (Cp : Caps Coin) (x : Msg) :
    0 ≤ βdev2 D I Cp x := by
  unfold βdev2; positivity

/-- **`KSLin2On` holds wherever `κ ≥ 1`**, at every compiled record: `Pr[accept ∧ ¬extract] ≤ 1 ≤ κ`, and every other
term of the bound is nonnegative. -/
theorem ksLin2On_of_one_le_κ [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop)
    (hκ : ∀ decI, decOK Bd.deploy decI → ∀ x, G x → 1 ≤ κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x) :
    KSLin2On H D Bd DO Cp G := by
  intro decI hdI decO _ x hx Ω _ _ P M R
  have h1 : PrBad H (D.record Bd.IVStar).V (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
      (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤ 1 := Pr_le_one _
  have hκ1 := hκ decI hdI x hx
  have hs := Real.sqrt_nonneg
    (γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x * PrB H (D.record Bd.IVStar).V P x)
  have hB := mul_nonneg (by positivity : (0 : ℝ) ≤ 4 * ((Inner.ofFlock Bd.deploy decI).V.k
      (xI (D.record Bd.IVStar) x) : ℝ) + 1)
    (PrBinnerV_nonneg H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x)
  have hC := mul_nonneg (by positivity : (0 : ℝ) ≤ 4 * (Cp.width (xI (D.record Bd.IVStar) x) : ℝ))
    (add_nonneg (PrFCI_nonneg H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M)
      (by positivity : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1)))
  have hF := mul_nonneg (by norm_num : (0 : ℝ) ≤ 4)
    (add_nonneg (Pr_nonneg _ : 0 ≤ PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R)
      (by positivity : (0 : ℝ) ≤ (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) : ℝ) + 1) /
        ((R : ℝ) + 1)))
  have hD := mul_nonneg (βdev2_nonneg (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x)
    (add_nonneg (PrFdevV_nonneg H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M)
      (by positivity : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1)))
  linarith

end Bound

end Claims.Recursion

namespace RecDeploy

open Claims (Msg)

theorem parse_of_readX_none {x : Msg} (h : readX x = none) : parse x = none := by
  simp only [parse, RecScope.parseOf, RecScope.parseBytes, decodeR, h, Option.map_none, Option.filter_none]

theorem nS_of_readX_none {x : Msg} (h : readX x = none) : nS x = 0 := by
  simp only [nS, h]

/-- The empty statement sets up no plain table. -/
theorem tables_nil : FlockDeploy.tables false [] = [] := rfl

/-- **The inner Flock's `κ` at the empty statement is at least one**: it sets up no table, so `E^rep` waits for no
accepted rerun (`k = 0`) and `e^{-k/8} = 1`. -/
theorem one_le_κ_nil (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    1 ≤ Claims.Flock.κ (bdDeployed rel nextE colsE wE join rowIn final w).deploy [] := by
  have hN : (bdDeployed rel nextE colsE wE join rowIn final w).deploy.N₀ [] = 0 := by
    show FlockDeploy.N₀ false [] = 0
    simp only [FlockDeploy.N₀, tables_nil, List.map_nil, List.sum_nil]
  have hε : (bdDeployed rel nextE colsE wE join rowIn final w).deploy.εtable [] = 0 := by
    show FlockDeploy.εtable false [] = 0
    simp only [FlockDeploy.εtable, tables_nil, List.map_nil, List.sum_nil]
  have hk : (bdDeployed rel nextE colsE wE join rowIn final w).deploy.k [] = 0 := by
    unfold Claims.Flock.Deploy.k
    rw [hN]
    simp
  unfold Claims.Flock.κ
  rw [hε, hk]
  have := Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 2)
    (-((bdDeployed rel nextE colsE wE join rowIn final w).deploy.waits [] : ℝ))
  simp only [Nat.cast_zero, mul_zero, zero_div, neg_zero, Real.exp_zero, add_zero, zero_add]
  linarith

/-- **`κ ≥ 1` where the statement's bytes don't read**, at the compiled records with the verifier that accepts
everything: no V\* part, and the inner statement is the empty one. -/
theorem one_le_κ_acceptAll_of_readX_none (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) {x : Msg} (h : readX x = none) :
    1 ≤ Claims.Recursion.κ ((recBoundAt rel b L).acceptAll.record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) x := by
  have hx : Claims.Recursion.xI ((recBoundAt rel b L).acceptAll.record
      (bdDeployed rel nextE colsE wE join rowIn final w).IVStar) x = [] := by
    show ((parse x).map Claims.Recursion.Stmt.inner).getD [] = []
    rw [parse_of_readX_none h]
    rfl
  have : IsEmpty (Fin (((recBoundAt rel b L).acceptAll.record
      (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).nS x)) := by
    show IsEmpty (Fin (nS x))
    rw [nS_of_readX_none h]
    exact Fin.isEmpty'
  unfold Claims.Recursion.κ
  rw [Finset.univ_eq_empty, Finset.sum_empty, mul_zero, add_zero, hx]
  exact one_le_κ_nil rel nextE colsE wE join rowIn final w

/-- **`KSLin2` at the verifier that accepts everything holds wherever the statement's bytes don't read.** -/
theorem ksLin2On_acceptAll_readX_none (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (Cp : Claims.Recursion.Caps FlockDeploy.Coin) :
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L).acceptAll
      (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO) Cp (fun x => readX x = none) :=
  Claims.Recursion.ksLin2On_of_one_le_κ _ _ _ _ _ _ fun decI _ _ hx =>
    one_le_κ_acceptAll_of_readX_none rel b L nextE colsE wE join rowIn final w decI hx

end RecDeploy

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

/-- **`RecursiveKSAt` at the verifier that accepts everything**: the compiled records' verifier keeps its rounds and
accepts at its first round with no SHA-512 call (`RecAt.acceptAll`); every other record and the dense statements are
`RecursiveKSAt`'s. -/
def RecursiveKSAtAll (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  Claims.Recursion.KSLin2On FlockDeploy.sha512
    (RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
      (RecDeploy.DeployedR.layR rest)).acceptAll
    (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
      RecDeploy.rowInDeployed final w)
    (RecDeploy.deployZkRec _root_.Flock.Readable.relation) FlockDeploy.KS.CpDep RecDeploy.DenseAt

/-- **`RecursiveKSAt` at the verifier that accepts everything holds wherever no statement's bytes read**: every
statement is then dense, and at each `κ ≥ 1`. The deployed parser reads a statement only through `jsonOf`, whose
`Lean.Json.Parser.anyCore` is a `partial def` the kernel can't unfold, so no proof exhibits a statement that reads, and
`¬ RecursiveKSAtAll` is not provable at the deployed parser. -/
theorem recursiveKSAtAll_of_readX_none (hJ : ∀ x, RecDeploy.readX x = none) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveKSAtAll rd final rest wE w b aux nOf hd recRest := by
  intro decI hdI decO hdO x _ Ω _ _ P M R
  exact RecDeploy.ksLin2On_acceptAll_readX_none _ _ _ _ _ _ _ _ _ _ _ _ _ decI hdI decO hdO x (hJ x) Ω P M R

end Flock.Guarantees
