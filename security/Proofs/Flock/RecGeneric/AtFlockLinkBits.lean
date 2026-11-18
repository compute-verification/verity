module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.LinkCost
public import Proofs.Flock.Claims.Recursion.BitsLin2
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ERepCost

/-!
# The gated link's binding term at Flock, per SHA-512 call

`KSLinkOn`'s binding term is `β₂·(Pr[linkPlan finds a collision at verModel] + 1/(M+1))`. A rerun of today's compiled
record (`Comp.ofRec`) costs one sample (`sampleCost`, at most `t·c_Be/2` in expectation, `Ex_sampleCost_le`) and the
evaluations its finder searches and its inner view's run makes: at most `|S(x_I)| + 2n + v + w` at call bounds `n`
(the inner verifier's, on the inner messages and on the inner view), `v` (the compiled verifier's own at the V\*
statements) and `w` (the calls `E_O`'s witnesses record) (`rerunCost_ofRec_le`). The link names at most two reruns in
expectation (`linkPlan_ExCost_le`), so under `ExpectedCR` at its finder the term at `M = 2^256` is at most `t·ε_L`,
`ε_L = β₂·c_L/2^256` at `c_L = c_Be + 2(|S(x_I)| + 2n + v + w) + 1` (`epsLink`, `linkTerm_le`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-- **The link's calls per prover call**: `c_L = c_Be + 2(|S(x_I)| + 2n + v + w) + 1`. -/
noncomputable def linkCalls {Coin : Type} [Inhabited Coin] (D : Deploy Coin) (x : Msg) (n v w : ℕ) : ℕ :=
  cBe D x + 2 * (setupLen D x + 2 * n + v + w) + 1

/-- **The link's binding term per SHA-512 call of the prover's run**: `ε_L = β₂·c_L/2^256`. -/
noncomputable def epsLink {Coin W : Type} [Inhabited Coin] (D : Deploy Coin) (I : Inner Coin W) (Cp : Caps Coin)
    (x : Msg) (n v w : ℕ) : ℝ :=
  βdev2 D I Cp x * linkCalls D x n v w / 2 ^ (256 : ℝ)

section Instance

variable {Coin Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col']

/-- **A rerun costs a sample and its searched evaluations**: at most `sampleCost + |S(x_I)| + 2n + v + w`. -/
theorem rerunCost_ofRec_le (H : Msg → Msg) (D : RecAt Coin) (IV : Verifier Coin) {W : Type} (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} (P : Prover Ω) (x : Msg)
    {n v w : ℕ} (hn : ∀ cs ws, ((D.record IV).innerCalls H x cs ws).length ≤ n)
    (hview : ∀ r ws, ((Comp.ofRec D IV I DO decO).innerLog IV H x r ws).length ≤ n)
    (hv : ∀ ps, (setupCalls H (D.record IV) ((D.record IV).vstar x ps)).length ≤ v)
    (hw : ∀ ω u ws, (Comp.ofRec D IV I DO decO).wits H P x ω u = some ws → ((D.record IV).calls x ws).length ≤ w)
    (ω : Ω) (u : (Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x) :
    (Comp.ofRec D IV I DO decO).rerunCost IV H P x ω u ≤
      sampleCost H (D.record IV) I DO decO P x ω u + (setupLen (D.record IV) x + 2 * n + v + w) := by
  have hsplit : (Comp.ofRec D IV I DO decO).rerunCost IV H P x ω u =
      sampleCost H (D.record IV) I DO decO P x ω u + ((Comp.ofRec D IV I DO decO).viewCalls IV H P x ω u).length +
        ((Comp.ofRec D IV I DO decO).rerunCalls H P x ω u).length +
        ((Comp.ofRec D IV I DO decO).ownAt H P x ω u).length := rfl
  have hV : ((Comp.ofRec D IV I DO decO).viewCalls IV H P x ω u).length ≤ n := by
    unfold Comp.viewCalls
    cases (Comp.ofRec D IV I DO decO).wits H P x ω u with
    | none => simp
    | some ws => exact hview u.1 ws
  have hR : ((Comp.ofRec D IV I DO decO).rerunCalls H P x ω u).length ≤ w + setupLen (D.record IV) x + n := by
    unfold Comp.rerunCalls
    cases hW : (Comp.ofRec D IV I DO decO).wits H P x ω u with
    | none => simp
    | some ws =>
      show ((D.record IV).callsAt H x u.1 ws).length ≤ _
      have h2 := hw ω u ws hW
      have h3 := hn (coinSeq u.1) ws
      simp only [Deploy.callsAt, List.length_append, List.length_map, setupLen]
      omega
  have hO : ((Comp.ofRec D IV I DO decO).ownAt H P x ω u).length ≤ v := hv _
  rw [hsplit]
  omega

/-- **A rerun costs at most `t·c_Be/2 + |S(x_I)| + 2n + v + w` in expectation**, at a prover whose runs make at most `t`
calls. -/
theorem Ex_rerunCost_ofRec_le (H : Msg → Msg) (D : RecAt Coin) (IV : Verifier Coin) {W : Type}
    (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T')
    (hS : SessionAccepts (D.record IV)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (t : ℕ)
    (ht : ∀ ω r, (run H P (D.record IV).V x ω r).pLog.length + (run H P (D.record IV).V x ω r).vLog.length ≤ t)
    {n v w : ℕ} (hn : ∀ cs ws, ((D.record IV).innerCalls H x cs ws).length ≤ n)
    (hview : ∀ r ws, ((Comp.ofRec D IV I DO decO).innerLog IV H x r ws).length ≤ n)
    (hv : ∀ ps, (setupCalls H (D.record IV) ((D.record IV).vstar x ps)).length ≤ v)
    (hw : ∀ ω u ws, (Comp.ofRec D IV I DO decO).wits H P x ω u = some ws → ((D.record IV).calls x ws).length ≤ w) :
    Ex (fun q : Ω × ((Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x) =>
      ((Comp.ofRec D IV I DO decO).rerunCost IV H P x q.1 q.2 : ℝ)) ≤
      t * cBe (D.record IV) x / 2 + ((setupLen (D.record IV) x + 2 * n + v + w : ℕ) : ℝ) := by
  calc _ ≤ Ex (fun q : Ω × ((Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x) =>
          (sampleCost H (D.record IV) I DO decO P x q.1 q.2 : ℝ) +
            ((setupLen (D.record IV) x + 2 * n + v + w : ℕ) : ℝ)) :=
        Ex_mono fun q => by exact_mod_cast rerunCost_ofRec_le H D IV I DO decO P x hn hview hv hw q.1 q.2
    _ = Ex (fun q : Ω × ((Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x) =>
          (sampleCost H (D.record IV) I DO decO P x q.1 q.2 : ℝ)) +
            ((setupLen (D.record IV) x + 2 * n + v + w : ℕ) : ℝ) := by
        rw [Ex_add, Ex_const]
    _ ≤ _ := add_le_add (Ex_samples_le H (D.record IV) I DO decO x P
        (fun ω => Ex_sampleCost_le H (D.record IV) I DO decO hS x P t ht ω)) le_rfl

/-- **The link at today's compiled record under `ExpectedCR`**: at most `(t·c_Be + 2(|S(x_I)| + 2n + v + w))/2^256`. -/
theorem linkPlan_PrAt_ofRec_le (H : Msg → Msg) (D : RecAt Coin) (IV : Verifier Coin) {W : Type}
    (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T')
    (hS : SessionAccepts (D.record IV)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (t : ℕ)
    (ht : ∀ ω r, (run H P (D.record IV).V x ω r).pLog.length + (run H P (D.record IV).V x ω r).vLog.length ≤ t)
    {n v w : ℕ} (hn : ∀ cs ws, ((D.record IV).innerCalls H x cs ws).length ≤ n)
    (hview : ∀ r ws, ((Comp.ofRec D IV I DO decO).innerLog IV H x r ws).length ≤ n)
    (hv : ∀ ps, (setupCalls H (D.record IV) ((D.record IV).vstar x ps)).length ≤ v)
    (hw : ∀ ω u ws, (Comp.ofRec D IV I DO decO).wits H P x ω u = some ws → ((D.record IV).calls x ws).length ≤ w)
    (k : Msg → ℕ) (M : ℕ) (y : Msg)
    (hcr : ∀ h : 0 < (linkPlan (Coin := Coin) k M).sel y,
      ExpectedCR H ((linkPlan (Coin := Coin) k M).finder ((Comp.ofRec D IV I DO decO).verModel IV H P x) y h)) :
    (linkPlan (Coin := Coin) k M).PrAt H ((Comp.ofRec D IV I DO decO).verModel IV H P x) y ≤
      ((t : ℝ) * cBe (D.record IV) x + 2 * ((setupLen (D.record IV) x + 2 * n + v + w : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) := by
  refine (linkPlan_PrAt_le_ecr H k M _ y hcr).trans (div_le_div_of_nonneg_right ?_ (by positivity))
  have h := Ex_rerunCost_ofRec_le H D IV I DO decO hS P x t ht hn hview hv hw
  calc 2 * Ex (fun q : Ω × ((Fin ((D.record IV).V.k x) → Coin) × (EO (D.record IV) I DO decO).C x) =>
        (((Comp.ofRec D IV I DO decO).verModel IV H P x).cost q.1 q.2 : ℝ))
      ≤ 2 * (t * cBe (D.record IV) x / 2 + ((setupLen (D.record IV) x + 2 * n + v + w : ℕ) : ℝ)) :=
        mul_le_mul_of_nonneg_left h (by norm_num)
    _ = _ := by ring

/-- **The link's binding term at `M = 2^256` is at most `t·ε_L`** (`epsLink`), at a prover whose runs make between `1`
and `t` calls, under `ExpectedCR` at the link's finder. -/
theorem linkTerm_le (H : Msg → Msg) (D : RecAt Coin) (IV : Verifier Coin) {W : Type}
    (I : Inner Coin W) (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (Cp : Caps Coin)
    (hS : SessionAccepts (D.record IV)) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (t : ℕ)
    (ht1 : 1 ≤ t)
    (ht : ∀ ω r, (run H P (D.record IV).V x ω r).pLog.length + (run H P (D.record IV).V x ω r).vLog.length ≤ t)
    {n v w : ℕ} (hn : ∀ cs ws, ((D.record IV).innerCalls H x cs ws).length ≤ n)
    (hview : ∀ r ws, ((Comp.ofRec D IV I DO decO).innerLog IV H x r ws).length ≤ n)
    (hv : ∀ ps, (setupCalls H (D.record IV) ((D.record IV).vstar x ps)).length ≤ v)
    (hw : ∀ ω u ws, (Comp.ofRec D IV I DO decO).wits H P x ω u = some ws → ((D.record IV).calls x ws).length ≤ w)
    (k : Msg → ℕ) (y : Msg)
    (hcr : ∀ h : 0 < (linkPlan (Coin := Coin) k (2 ^ 256)).sel y,
      ExpectedCR H ((linkPlan (Coin := Coin) k (2 ^ 256)).finder ((Comp.ofRec D IV I DO decO).verModel IV H P x) y h)) :
    βdev2 (D.record IV) I Cp x *
        ((linkPlan (Coin := Coin) k (2 ^ 256)).PrAt H ((Comp.ofRec D IV I DO decO).verModel IV H P x) y +
          1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) ≤
      t * epsLink (D.record IV) I Cp x n v w := by
  have hP := linkPlan_PrAt_ofRec_le H D IV I DO decO hS P x t ht hn hview hv hw k (2 ^ 256) y hcr
  have h2 : (2 : ℝ) ^ (256 : ℝ) = ((2 ^ 256 : ℕ) : ℝ) := by
    rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast, Nat.cast_pow, Nat.cast_ofNat]
  have hpos : (0 : ℝ) < ((2 ^ 256 : ℕ) : ℝ) := by positivity
  have htail : 1 / (((2 ^ 256 : ℕ) : ℝ) + 1) ≤ 1 / ((2 ^ 256 : ℕ) : ℝ) :=
    one_div_le_one_div_of_le hpos (by linarith)
  have hβ : 0 ≤ βdev2 (D.record IV) I Cp x := by unfold βdev2; positivity
  have htR : (1 : ℝ) ≤ t := by exact_mod_cast ht1
  rw [h2] at hP
  have key : ((t : ℝ) * cBe (D.record IV) x + 2 * ((setupLen (D.record IV) x + 2 * n + v + w : ℕ) : ℝ)) /
        ((2 ^ 256 : ℕ) : ℝ) + 1 / ((2 ^ 256 : ℕ) : ℝ) ≤
      t * ((linkCalls (D.record IV) x n v w : ℕ) / ((2 ^ 256 : ℕ) : ℝ)) := by
    rw [div_add_div_same, mul_div_assoc']
    refine div_le_div_of_nonneg_right ?_ hpos.le
    unfold linkCalls
    push_cast
    have hX : (0 : ℝ) ≤ 2 * ((setupLen (D.record IV) x : ℝ) + 2 * n + v + w) + 1 := by positivity
    have hb : (0 : ℝ) ≤ cBe (D.record IV) x := Nat.cast_nonneg _
    nlinarith [mul_le_mul_of_nonneg_right htR hX]
  unfold epsLink
  rw [h2]
  calc βdev2 (D.record IV) I Cp x *
        ((linkPlan (Coin := Coin) k (2 ^ 256)).PrAt H ((Comp.ofRec D IV I DO decO).verModel IV H P x) y +
          1 / (((2 ^ 256 : ℕ) : ℝ) + 1))
      ≤ βdev2 (D.record IV) I Cp x * (t * ((linkCalls (D.record IV) x n v w : ℕ) / ((2 ^ 256 : ℕ) : ℝ))) :=
        mul_le_mul_of_nonneg_left ((add_le_add hP htail).trans key) hβ
    _ = _ := by ring

end Instance

end Claims.RecGeneric
