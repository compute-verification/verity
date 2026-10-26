import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.VCallsAt
import Proofs.Flock.Claims.Recursion.PlainKSAbortFails
import Proofs.Flock.Claims.Recursion.LamDeployed
import Proofs.Flock.Claims.Witnesses.RecursionLin2

/-!
# `RecursiveKS` at the verifier that accepts everything (LEAN-66's `nontrivial`)

LEAN-66 asks of each guarantee that states what an acceptance means a lemma that its conclusion is false at the
verifier that accepts everything. For `RecursiveKS` that verifier is the compiled recursive verifier of the compiled
record, `(D.record IV).V`, whose acceptance `PrBad` reads, replaced by one that keeps its rounds and accepts at its first
round with no SHA-512 call; it computes nothing, so its record holds no call and stages no V\* statement
(`RecAt.acceptAll`). The parser, the inner view, the numbers, `IVStar`, the bound record, the outer record, the caps and
the dense statements stay deployed. `RecursiveKSAtAll` is `RecursiveKSAt` there.

* **The nontrivial lemma** (`recursiveKSAt_nontrivial`): `RecursiveKSAtAll` is false wherever a dense statement in the
  deployed scope has no witness. The silent prover is accepted on every run and nothing is extracted, so
  `Pr[accept ∧ ¬extract] = 1`; every finder term is `0`; the tails are below `1/4` each at a large enough `M` and `R`;
  and `λ₂ ≥ 100` there (`RecDeploy.lamLin2_deployed_ge_dep`) gives `κ < 1/2` (`κ_lt_half_of_lamLin2`). The general
  form is `not_ksLin2On_acceptAll`, at any compiled record and statement with no witness and `κ < 1/2`.
* **Why its premises stay premises** (`recursiveKSAtAll_of_readX_none`): the bound's right-hand side is at least `κ`,
  whatever the verifier (`ksLin2On_of_one_le_κ`). At a statement whose bytes don't read (`RecDeploy.readX x = none`),
  `κ` is the inner Flock's at the empty inner statement, which sets up no table, so `κ ≥ e⁰ = 1` and the bound holds
  at every verifier there (`ksLin2On_acceptAll_readX_none`). The deployed parser reads a statement only through
  `jsonOf` (`Lean.Json.Parser.anyCore`, a `partial def`, which the kernel can't unfold), so no proof exhibits a
  statement that reads, and the nontrivial lemma takes one as its premise.
-/

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

/-- At a statement with no witness, every extractor fails on every run of a verifier that accepts at its first round. -/
theorem prBad_acc {W Ω : Type} [Fintype Ω] [Nonempty Ω] (H : Msg → Msg) (V : Verifier Coin) (R : Rel W)
    (E : Extractor Coin W V) (x : Msg) (P : Prover Ω) (hv : ∀ as cs, V.next x as cs = .ret .acc) (hk : 0 < V.k x)
    (hR : ∀ w, ¬ R x w) : PrBad H V R E x P = 1 := by
  unfold PrBad
  refine (Pr_congr (q := fun _ => True) fun s => iff_true_intro ⟨run_acc H P V x hv hk s.1 s.2.1, ?_⟩).trans
    (Pr_const_of trivial)
  cases (E.eval H P x s.1 s.2.1 s.2.2).1 with
  | none => exact id
  | some w => exact hR w

/-- A verifier that accepts at its first round calls `H` nowhere, so `B` finds nothing. -/
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

/-- `FCI` finds nothing where the outer witnesses record no call. -/
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

/-- **`KSLin2On` fails at the verifier that accepts everything** at any statement of `G` where the compiled verifier
takes a round, no witness is in the relation and `κ < 1/2`: the silent prover is accepted on every run and no
extractor finds a witness, so `Pr[accept ∧ ¬extract] = 1`. The accepting verifier and its record call `H` nowhere, so
every finder term is `0`, and at `M = ⌈4·(4W + β₂)⌉` and `R = 16·(k_I + 1)` each of the two tails is below `1/4`. -/
theorem not_ksLin2On_acceptAll [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) (decI : List (ℕ × Col) → List T)
    (hdI : decOK Bd.deploy decI) (decO : List (ℕ × Col') → List T') (hdO : decOK DO decO) (x : Msg) (hx : G x)
    (hk : 0 < (D.record Bd.IVStar).V.k x)
    (hR : ∀ w, ¬ Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x w)
    (hκ : κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x < 1 / 2) :
    ¬ KSLin2On H D.acceptAll Bd DO Cp G := by
  intro h
  have hc : ∀ y (r : Fin ((D.acceptAll.record Bd.IVStar).V.k y) → Coin) ws,
      (D.acceptAll.record Bd.IVStar).callsAt H y r ws = [] := fun _ _ _ => rfl
  have hs : ∀ y ps, (D.acceptAll.record Bd.IVStar).vstar y ps = [] := fun _ _ => rfl
  have hv : ∀ as cs, (D.acceptAll.record Bd.IVStar).V.next x as cs = .ret .acc := fun _ _ => rfl
  have hk' : 0 < (D.acceptAll.record Bd.IVStar).V.k x := hk
  have hR' : ∀ w, ¬ Recursion.R (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x w := hR
  have hκ' : κ (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x < 1 / 2 := hκ
  have h1 := h decI hdI decO hdO x hx Unit Witness.silentP
    ⌈4 * (4 * (Cp.width (xI (D.acceptAll.record Bd.IVStar) x) : ℝ) +
      βdev2 (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x)⌉₊
    (16 * ((Inner.ofFlock Bd.deploy decI).k (xI (D.acceptAll.record Bd.IVStar) x) + 1))
  have hBad : PrBad H (D.acceptAll.record Bd.IVStar).V
      (Recursion.R (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
      (E H (D.acceptAll.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x Witness.silentP = 1 :=
    prBad_acc H _ _ _ x _ hv hk' hR'
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

/-- **`λ₂ ≥ 100` gives `κ < 1/2`**: `λ₂`'s first term is `log₂(1/κ)`. -/
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

/-- The compiled verifier takes at least one round at every statement: the inner verifier `IVStar` takes `DE`'s, Flock's
deployed plain verifier's (`V_k_pos_flock`). -/
theorem recBound_k_pos (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (x : Msg) :
    0 < ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).V.k x := by
  show 0 < (FlockDeploy.V false).k (xIOf x) + kS x + _
  have := V_k_pos_flock false (xIOf x)
  omega

/-- **`KSLin2` at the verifier that accepts everything fails** at any statement of `G` with no witness in the
relation where `κ < 1/2`, at the deployed decoders (`FlockDeploy.listWithin_plainTrim`,
`Claims.GS.listWithin_zkDecodeRD`). -/
theorem not_ksLin2On_acceptAll (rel relO : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (Cp : Claims.Recursion.Caps FlockDeploy.Coin) (G : Msg → Prop) (x : Msg) (hx : G x)
    (hR : ∀ v, ¬ Claims.Recursion.R ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy Claims.GS.plainDecodeRD)
      x v)
    (hκ : Claims.Recursion.κ ((recBoundAt rel b L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy Claims.GS.plainDecodeRD)
      x < 1 / 2) :
    ¬ Claims.Recursion.KSLin2On FlockDeploy.sha512 (recBoundAt rel b L).acceptAll
      (bdDeployed rel nextE colsE wE join rowIn final w) (deployZkRec relO) Cp G :=
  Claims.Recursion.not_ksLin2On_acceptAll _ _ _ _ _ _ Claims.GS.plainDecodeRD (FlockDeploy.listWithin_plainTrim rel)
    Claims.GS.zkDecodeRD (Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl) x hx
    (recBound_k_pos rel b L nextE colsE wE join rowIn final w x) hR hκ

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

/-- **`RecursiveKS`'s nontrivial lemma** (LEAN-66): `RecursiveKSAt` is false at the verifier that accepts everything
wherever a dense statement has no witness and is in the deployed scope, `PlainIn` at its inner statement and `PartIn` at
each staged V\* part, with at most `2^16` parts and `2^56` inner setup inputs. There `λ₂ ≥ 100`
(`RecDeploy.lamLin2_deployed_ge_dep`), so `κ < 1/2`, while the accepting verifier's `Pr[accept ∧ ¬extract]` is `1`.
The premises are all of the statement: no proof exhibits one at the deployed parser, which reads only through `jsonOf`
(`recursiveKSAtAll_of_readX_none`). -/
theorem recursiveKSAt_nontrivial (rd : Msg → Option FinalP)
    (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg))
    (rest : (r : RecDeploy.XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (wE w : Msg → ℝ)
    (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) {x : Msg}
    (hD : RecDeploy.DenseAt x)
    (hR : ∀ v, ¬ Claims.Recursion.R
      ((RecDeploy.recBoundAt _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR rest)).record
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
          RecDeploy.rowInDeployed final w).IVStar)
      (Claims.Recursion.Inner.ofFlock
        (RecDeploy.bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR rd)
          (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
          RecDeploy.rowInDeployed final w).deploy Claims.GS.plainDecodeRD) x v)
    {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (RecDeploy.xIOf x) s spec sts)
    (hO : ∀ j < RecDeploy.nS x, ∃ i spec' sts',
      RecDeploy.PartIn (RecDeploy.staged b.st x (RecDeploy.pick b.st x j) j) i spec' sts')
    (hn : RecDeploy.nS x ≤ 2 ^ 16) (hS : (RecDeploy.innerSetupIns (RecDeploy.xIOf x)).length ≤ 2 ^ 56) :
    ¬ RecursiveKSAtAll rd final rest wE w b aux nOf hd recRest :=
  RecDeploy.not_ksLin2On_acceptAll _ _ _ _ _ _ _ _ _ _ _ _ _ x hD hR
    (Claims.Recursion.κ_lt_half_of_lamLin2
      (RecDeploy.lamLin2_deployed_ge_dep _root_.Flock.Readable.relation { b with tr := RecDeploy.recTr aux nOf hd recRest }
        (RecDeploy.DeployedR.layR rest) (FlockDeploy.nextR rd)
        (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation rd).cols wE FlockDeploy.joinR
        RecDeploy.rowInDeployed final w Claims.GS.plainDecodeRD hin hO hn hS (n := 0) (by norm_num) (v := 0)
        (by norm_num)))

/-- **`RecursiveKS` at the verifier that accepts everything**: `RecursiveKSAtAll` at the deployed reader, for every
other field of the brings and every record the coin server keeps. -/
def RecursiveKSAll : Prop :=
  ∀ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    RecursiveKSAtAll RecDeploy.DeployedR.rdR RecDeploy.DeployedR.finalR RecDeploy.OpenRows.elemR RecDeploy.DeployedR.wER
      RecDeploy.DeployedR.wR (RecDeploy.deployedB b) aux nOf hd recRest

/-- **A false statement in the deployed scope**, `recursiveKS_nontrivial`'s premise: brings, a coin-server record and
a dense statement with no witness in the relation, in the deployed scope as `RecDeploy.lamLin2_deployed_ge_dep` takes
it (`PlainIn` at its inner statement, `PartIn` at each staged V\* part, at most `2^16` parts and `2^56` inner setup
inputs). A statement of the deployment, not a claim: no proof exhibits one at the deployed parser, which reads only
through `jsonOf` (`recursiveKSAtAll_of_readX_none`). -/
def FalseInScope : Prop :=
  ∃ (b : RecDeploy.Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) (x : Msg)
    (s : FlockDeploy.Ins) (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt),
    RecDeploy.DenseAt x ∧
    (∀ v, ¬ Claims.Recursion.R
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
          RecDeploy.DeployedR.wR).deploy Claims.GS.plainDecodeRD) x v) ∧
    FlockDeploy.PlainIn (RecDeploy.xIOf x) s spec sts ∧
    (∀ j < RecDeploy.nS x, ∃ i spec' sts',
      RecDeploy.PartIn (RecDeploy.staged (RecDeploy.deployedB b).st x (RecDeploy.pick (RecDeploy.deployedB b).st x j) j)
        i spec' sts') ∧
    RecDeploy.nS x ≤ 2 ^ 16 ∧ (RecDeploy.innerSetupIns (RecDeploy.xIOf x)).length ≤ 2 ^ 56

/-- **`RecursiveKS`'s nontrivial lemma, at the guarantee** (LEAN-66): `RecursiveKS` is false at the verifier that
accepts everything wherever the deployment has a false statement in scope (`FalseInScope`). -/
theorem recursiveKS_nontrivial (h : FalseInScope) : ¬ RecursiveKSAll := by
  obtain ⟨b, aux, nOf, hd, recRest, x, s, spec, sts, hD, hR, hin, hO, hn, hS⟩ := h
  exact fun hK => recursiveKSAt_nontrivial _ _ _ _ _ _ aux nOf hd recRest hD hR hin hO hn hS (hK b aux nOf hd recRest)

end Flock.Guarantees
