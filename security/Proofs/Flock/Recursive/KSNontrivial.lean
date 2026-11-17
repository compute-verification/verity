module

public import Proofs.Flock.Claims.Recursion.Assumptions.KSFixture
public import Proofs.Flock.Claims.Recursion.PlainKSAbortFails
public import Proofs.Flock.Claims.Witnesses.RecursionLin2
public import Proofs.Flock.Recursive.KSStatement
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.DeployScope

/-! `RecursiveKS`'s nontrivial lemma at the verifier that accepts everything, from the named assumption `RecDeploy.KSFixture`, an assumption because no Lean proof can build a statement the deployed parser's `partial def` reads. -/

@[expose] public section

namespace Claims.Recursion

/-- **The compiled records with the verifier that accepts everything**: each record's verifier keeps its rounds and
accepts at its first round with no SHA-512 call, so it computes nothing for its witnesses to record (`calls`,
`innerSetup`, `innerCalls`) and stages no V\* statement (`vstar`). The parser, the inner view, the numbers and every
other field are `D`'s. -/
def RecAt.acceptAll {Coin : Type} [Inhabited Coin] (D : RecAt Coin) : RecAt Coin where
  record IV := { D.record IV with
    V := ⟨(D.record IV).V.k, fun _ _ _ => .ret .acc⟩
    vstar := fun _ _ => []
    calls := fun _ _ => []
    innerSetup := fun _ => []
    innerCalls := fun _ _ _ _ => [] }
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
  have hF0 : 0 ≤ PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R := Pr_nonneg _
  have hF : 0 ≤ 4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
      (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) : ℝ) + 1) / ((R : ℝ) + 1)) :=
    mul_nonneg (by norm_num) (add_nonneg hF0 (by positivity))
  have hD := mul_nonneg (βdev2_nonneg (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x)
    (add_nonneg (PrFdevV_nonneg H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M)
      (by positivity : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1)))
  linarith

theorem prBad_acc {W Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (R : Rel W)
    (E : Extractor Coin W V) (x : Msg) (P : Prover Ω) (hv : ∀ as cs, V.next x as cs = .ret .acc) (hk : 0 < V.k x)
    (hE : ∀ ω r₀ c, (E.eval H P x ω r₀ c).1 = none) : PrBad H V R E x P = 1 := by
  unfold PrBad
  refine (Pr_congr (q := fun _ => True) fun s => iff_true_intro ⟨run_acc H P V x hv hk s.1 s.2.1, ?_⟩).trans
    (Pr_const_of trivial)
  rw [hE]
  exact id

theorem eval_bind_fst {R α β : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R α) (f : α → RP R β) :
    ((p.bind f).eval H run).1 = ((f (p.eval H run).1).eval H run).1 :=
  (Claims.RP.eval_bind H run p f).1

theorem eval_serve_firstWait_none {R₁ R₂ W' Op : Type} (q : R₁ → RP R₂ Seen) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W') (check : W' → Bool) (h : ∀ os, assemble os = none) (H : Msg → Msg)
    (run : R₂ → Transcript) : ∀ rss : List (List R₁),
    ((serve q (firstWait k opens assemble check rss)).eval H run).1 = none
  | [] => rfl
  | rs :: rss => by
    have e : firstWait k opens assemble check (rs :: rss) =
        (waitAcc k rs []).bind fun _ => firstWait k opens assemble check rss := by
      simp only [firstWait, h]
    rw [e, Witness.Recursion.serve_bind, eval_bind_fst]
    exact eval_serve_firstWait_none q k opens assemble check h H run rss

theorem eval_collect_none {R α : Type} (H : Msg → Msg) (run : R → Transcript) (p : RP R (Option α))
    (ps : List (RP R (Option α))) (hp : (p.eval H run).1 = none) : ((collect (p :: ps)).eval H run).1 = none := by
  rw [collect, eval_bind_fst, hp]
  rfl

theorem EO_none_of_vstar_nil {W : Type} (H : Msg → Msg) (A : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (x : Msg)
    (hs : ∀ ps, A.vstar x ps = []) (hS : 0 < A.nS x) (hP : DO.parse [] = none)
    (run : (Fin (A.V.k x) → Coin) → Transcript) (d : Fin (A.V.k x) → Coin) (c : (EO A I DO decO).C x) :
    (((EO A I DO decO).ext x d c).eval H run).1 = none := by
  have hA : ∀ (xs : List Msg) (s : ℕ) os, xs = [] →
      Flock.assemble DO.parse DO.table decO DO.read (xs.getD s []) os = none := by
    intro xs s os h
    subst h
    simp only [List.getD_nil, Flock.assemble, hP]
  have hl : List.finRange (A.nS x) ≠ [] := by
    intro h
    have := congrArg List.length h
    simp only [List.length_finRange, List.length_nil] at this
    omega
  simp only [EO, RP.eval]
  split
  · obtain ⟨a, l, hal⟩ := List.exists_cons_of_ne_nil hl
    rw [hal, List.map_cons]
    apply eval_collect_none
    exact eval_serve_firstWait_none _ _ _ _ _ (fun os => hA _ _ os (hs _)) H run _
  · rfl

theorem E_none_of_vstar_nil {W : Type} (H : Msg → Msg) (A : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (x : Msg)
    (hs : ∀ ps, A.vstar x ps = []) (hS : 0 < A.nS x) (hP : DO.parse [] = none) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r₀ : Fin (A.V.k x) → Coin) (c : (E H A I DO decO).C x) :
    ((E H A I DO decO).eval H P x ω r₀ c).1 = none := by
  have hO := EO_none_of_vstar_nil H A I DO decO x hs hS hP (run H P A.V x ω)
  unfold Extractor.eval
  simp only [E, served, erep, serve, outerStage, eval_bind_fst, hO, dec, RP.eval, reduceCtorEq, ↓reduceIte]

theorem prB_acc {Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (hv : ∀ as cs, V.next x as cs = .ret .acc) : PrB H V P x = 0 := by
  unfold PrB
  split_ifs with hk
  · unfold Finder.Pr
    refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
    rintro ⟨u, v, h, -⟩
    simp only [Claims.B, run_vLog_acc H P V x hv hk, List.append_nil, findColl] at h
    cases h
  · rfl

theorem prFCI_noCalls (H : Msg → Msg) (D : Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) {W : Type} (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) : PrFCI H D I DO decO Cp P x M = 0 := by
  unfold PrFCI
  split_ifs with hW
  · unfold Finder.Pr
    refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
    rintro ⟨u, v, h, -⟩
    simp only [FCI, hc, List.append_nil, findColl, ite_self, reduceCtorEq] at h
  · rfl

theorem not_ksLin2On_acceptAll [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (decI : List (ℕ × Col) → List T)
    (hdI : decOK Bd.deploy decI) (decO : List (ℕ × Col') → List T') (hdO : decOK DO decO) (x : Msg) (hx : G x)
    (hk : 0 < (D.record Bd.IVStar).V.k x) (hS : 0 < (D.record Bd.IVStar).nS x) (hP : DO.parse [] = none)
    (hκ : κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x < 1 / 2) :
    ¬ KSLin2On H D.acceptAll Bd DO Cp G := by
  intro h
  have hc : ∀ y (r : Fin ((D.acceptAll.record Bd.IVStar).V.k y) → Coin) ws,
      (D.acceptAll.record Bd.IVStar).callsAt H y r ws = [] := fun _ _ _ => rfl
  have hs : ∀ y ps, (D.acceptAll.record Bd.IVStar).vstar y ps = [] := fun _ _ => rfl
  have hv : ∀ as cs, (D.acceptAll.record Bd.IVStar).V.next x as cs = .ret .acc := fun _ _ => rfl
  have hk' : 0 < (D.acceptAll.record Bd.IVStar).V.k x := hk
  have hS' : 0 < (D.acceptAll.record Bd.IVStar).nS x := hS
  have hκ' : κ (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x < 1 / 2 := hκ
  have h1 := h decI hdI decO hdO x hx Unit Witness.silentP
    ⌈4 * (4 * (Cp.width (xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x)⌉₊
    (16 * ((Inner.ofFlock Bd.deploy decI).k (xI (D.acceptAll.record Bd.IVStar) x) + 1))
  have hBad : PrBad H (D.acceptAll.record Bd.IVStar).V
      (Recursion.R (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
      (E H (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x Witness.silentP = 1 :=
    prBad_acc H _ _ _ x _ hv hk'
      (E_none_of_vstar_nil H _ _ DO decO x (hs x) hS' hP Witness.silentP)
  have hB : PrB H (D.acceptAll.record Bd.IVStar).V Witness.silentP x = 0 := prB_acc H _ _ x hv
  rw [hBad, hB, Witness.Recursion.prBinnerV_noCalls H (D.acceptAll.record Bd.IVStar) hc hs,
    prFCI_noCalls H (D.acceptAll.record Bd.IVStar) hc,
    Witness.Recursion.prF0I_noCalls H (D.acceptAll.record Bd.IVStar) hc,
    Witness.Recursion.prFdevV_noCalls H (D.acceptAll.record Bd.IVStar) hc hs] at h1
  simp only [mul_zero, Real.sqrt_zero, zero_add, add_zero] at h1
  have hT := Witness.Recursion.tail_lt_quarter
    (4 * (Cp.width (xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x)
  have hF := Witness.Recursion.f0_tail_lt
    ((Inner.ofFlock Bd.deploy decI).k (xI (D.acceptAll.record Bd.IVStar) x))
  linarith

end Bound

theorem κ_lt_half_of_lamLin2 {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {D : Deploy Coin}
    {I : Inner Coin W} {Cp : Caps Coin} {x : Msg} {n v : ℕ} (h : 100 ≤ lamLin2 D I Cp x n v) : κ D I x < 1 / 2 := by
  unfold lamLin2 at h
  simp only [le_sub_iff_add_le, le_min_iff] at h
  have h6 : 0 ≤ Real.logb 2 6 := Real.logb_nonneg (by norm_num) (by norm_num)
  have ha : (100 : ℝ) ≤ Real.logb 2 (1 / κ D I x) := by linarith [h.1.1.1.1.1]
  rcases le_or_gt (κ D I x) 0 with h0 | h0
  · linarith
  have e : (2 : ℝ) ^ (-Real.logb 2 (1 / κ D I x)) = κ D I x := by
    rw [one_div, Real.logb_inv, neg_neg, Real.rpow_logb (by norm_num) (by norm_num) h0]
  calc κ D I x = (2 : ℝ) ^ (-Real.logb 2 (1 / κ D I x)) := e.symm
    _ ≤ (2 : ℝ) ^ (-(100 : ℝ)) := Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
    _ < (2 : ℝ) ^ (-(1 : ℝ)) := Real.rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num)
    _ = 1 / 2 := by rw [Real.rpow_neg_one]; norm_num

end Claims.Recursion

namespace RecDeploy

open Claims (Msg)

theorem parse_of_readX_none {x : Msg} (h : readX x = none) : parse x = none := by
  simp only [parse, Flock.RecScope.parseOf, Flock.RecScope.parseBytes, decodeR, h, Option.map_none,
    Option.filter_none]

theorem nS_of_readX_none {x : Msg} (h : readX x = none) : nS x = 0 := by
  simp only [nS, h]

theorem tables_nil : FlockDeploy.tables false [] = [] := rfl

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

theorem one_le_κ_acceptAll_of_readX_none (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) {x : Msg} (h : readX x = none) :
    1 ≤ Claims.Recursion.κ ((recBoundAt rel b L).acceptAll.record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI) x := by
  have hx : Claims.Recursion.xI ((recBoundAt rel b L).acceptAll.record
      (bdDeployed rel nextE colsE wE join rowIn final w).IVStar) x = [] := by
    show ((parse x).map Claims.Recursion.CompiledStatement.inner).getD [] = []
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

theorem ksLin2On_acceptAll_readX_none (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (Cp : Claims.Recursion.Caps FlockDeploy.Coin) :
    Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L).acceptAll
      (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO) Cp (fun x => readX x = none) :=
  Claims.Recursion.ksLin2On_of_one_le_κ _ _ _ _ _ _ fun decI _ _ hx =>
    one_le_κ_acceptAll_of_readX_none rel b L nextE colsE wE join rowIn final w decI hx

theorem recBound_k_pos (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (x : Msg) :
    0 < ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).V.k x := by
  show 0 < (FlockDeploy.V false).k (xIOf x) + kS x + _
  have := V_k_pos_flock false (xIOf x)
  omega

theorem zkRec_parse_nil (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    (deployZkRec rel).parse [] = none := by
  have h : readOfO [] = none := by
    unfold readOfO
    rw [insO_nil]
    rfl
  show (_root_.Flock.Readable.parseBytes readOfO []).map (_root_.Flock.Readable.toReadable rel) = none
  unfold _root_.Flock.Readable.parseBytes
  rw [h]
  rfl

theorem not_ksLin2On_acceptAll (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (Cp : Claims.Recursion.Caps FlockDeploy.Coin) (G : Msg → Prop) (x : Msg) (hx : G x) (hS : 0 < nS x)
    (hκ : Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy Claims.GS.plainDecodeRD)
      x < 1 / 2) :
    ¬ Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L).acceptAll
      (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO) Cp G :=
  Claims.Recursion.not_ksLin2On_acceptAll _ _ _ _ _ _ Claims.GS.plainDecodeRD (FlockDeploy.listWithin_plainTrim rel)
    Claims.GS.zkDecodeRD (Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl) x hx
    (recBound_k_pos rel b L nextE colsE wE join rowIn final w x) hS (zkRec_parse_nil relO) hκ

end RecDeploy

namespace Flock.Guarantees

open Claims (Msg Out)
open FlockDeploy (Coin FinalP)

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

/-- The deployed parser reads a statement only through `jsonOf`, whose `Lean.Json.Parser.anyCore` is a `partial def` the
kernel can't unfold, so no proof exhibits a statement that reads, and `¬ RecursiveKSAtAll` is not provable there. -/
theorem recursiveKSAtAll_of_readX_none (hJ : ∀ x, RecDeploy.readX x = none) (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    RecursiveKSAtAll rd final rest wE w b aux nOf hd recRest := by
  have h := RecDeploy.ksLin2On_acceptAll_readX_none _root_.Flock.Readable.relation _root_.Flock.Readable.relation
    { b with tr := RecDeploy.recTr aux nOf hd recRest } (RecDeploy.DeployedR.layR rest) (FlockDeploy.nextR rd)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR RecDeploy.rowInDeployed
    final w FlockDeploy.KS.CpDep
  intro decI hdI decO hdO x _
  exact h decI hdI decO hdO x (hJ x)

theorem denseAt_of_parses {x : Msg} (h : RecDeploy.DeployedParses x) : RecDeploy.DenseAt x := by
  intro s hs
  have h1 := h.1
  rw [hs] at h1
  have h2 := Option.some.inj (h1 : some (s.form.holo, s.form.sparse) = some (false, none))
  show s.form.holo = false ∧ s.form.sparse = none
  exact ⟨congrArg Prod.fst h2, congrArg Prod.snd h2⟩

theorem plainIn_of_parses {x : Msg} (h : RecDeploy.DeployedParses x) :
    ∃ s spec sts, FlockDeploy.PlainIn (RecDeploy.xIOf x) s spec sts := by
  have h3 := h.2.2
  cases hs : FlockDeploy.setupOf false (RecDeploy.xIOf x) with
  | none => rw [hs] at h3; simp at h3
  | some t =>
    obtain ⟨s, spec, sts⟩ := t
    rw [hs] at h3
    have hv : RecDeploy.setupView (s, spec, sts) = true := Option.some.inj h3
    simp only [RecDeploy.setupView, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, and_assoc] at hv
    obtain ⟨h1, h2, hp, ht⟩ := hv
    refine ⟨s, spec, sts, ⟨hs, h1, h2, hp, fun st hst => ?_⟩⟩
    obtain ⟨a, b, c, d, e⟩ := ht st hst
    exact ⟨a, b, show st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1026 ∧ spec.mPts ≤ 64 from ⟨c, d, e⟩⟩

/-- **`RecursiveKS`'s nontrivial lemma**: `RecursiveKS`'s conclusion, `RecursiveKSAtRec` at the deployed
values, is false at the verifier that accepts everything (`RecAt.acceptAll` of the deployment's compiled record), at
every brings, from the named assumption
`RecDeploy.KSFixture`: some bytes the deployed readers read as a dense compiled statement with a V\* session whose
inner statement is in the plain scope (`DeployedParses`), with a live-coin record at whose transcript its V\* parts
are in scope (`PartsInScope`). Everything else is proved here: the statement is dense (`denseAt_of_parses`) and in the
plain scope (`plainIn_of_parses`); the silent prover is accepted on every run, and the accepting record stages no V\*
statement, which the outer Flock's parser refuses (`RecDeploy.zkRec_parse_nil`), so `E*` extracts nothing and
`Pr[accept ∧ ¬extract] = 1`; every finder term is `0`; the tails are below `1/4` each; and `λ₂ ≥ 100` there
(`RecDeploy.lamLin2_deployed_ge_dep`) gives `κ < 1/2`. `E*` is the one `KSLin2On` names at that record, and the
collision term `√(γ·Pr[B])` is a summand of the bound, not a disjunct, and is `0` there (`prB_acc`). -/
theorem recursiveKS_nontrivial (h : RecDeploy.KSFixture) :
    ∃ (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
      (recRest : Msg → List (Msg × Out) → List UInt8), ∀ b : RecDeploy.Brings,
      ¬ RecursiveKSAtRec
        (RecDeploy.recBoundAt _root_.Flock.Readable.relation
          { RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest }
          (RecDeploy.DeployedR.layR RecDeploy.OpenRows.elemR)).acceptAll
        RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.DeployedR.wER RecDeploy.DeployedR.wR := by
  obtain ⟨x, aux, nOf, hd, recRest, hP, hO, hn, hS⟩ := h
  obtain ⟨s, spec, sts, hin⟩ := plainIn_of_parses hP
  refine ⟨aux, nOf, hd, recRest, fun b => ?_⟩
  have hO' : ∀ j < RecDeploy.nS x, ∃ i spec' sts', RecDeploy.PartIn
      (RecDeploy.staged ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } : RecDeploy.Brings).st
        x (RecDeploy.pick ({ RecDeploy.deployedB b with tr := RecDeploy.recTr aux nOf hd recRest } :
          RecDeploy.Brings).st x j) j) i spec' sts' := hO
  exact RecDeploy.not_ksLin2On_acceptAll _root_.Flock.Readable.relation _root_.Flock.Readable.relation
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
