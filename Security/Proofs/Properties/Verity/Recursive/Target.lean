import Proofs.Properties.Verity.Recursive
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Recursion.BitsLin2
import Proofs.Flock.Claims.Recursion.DeployFacts
import Proofs.Core.Circuit

/-!
# The target, proved: sampled proofs at the recursive configuration (piece H)

`sampledProofsRecursive_of` proves `sampledProofsRecursive`'s bound from the property G states (`AccOfAccepts`, the
statement of `acc_of_accepts` at the deployment), and `sampledProofsRecursive` is it at `acc_of_accepts`.

The proof, at a strategy `σ` whose finders meet `FinderCR` (elsewhere the error is `⊤`), where the files pass the
commit phase's check (elsewhere no play is accepted) and sampled proofs' claim fails on the extracted transcript `X`
(elsewhere the event is empty), so that the unbound units `U` number more than `ε n`:
* each accepted play shows, through the coupling (`SessionIn.names_shows`) and G, an accepted run of the recursive
  verifier at a dense statement (`verdict_run`);
* at each draw `ξ` such a play has, the draw misses `U`, or the run accepts while the extractor returns no witness in
  the recursive relation (`PrBad`, at most `t' · 2^{-λ}` by `RecursiveKSAt` through `bitsLin2On_of_ksLin2On`), or the
  extractor's witness, correct and opened on every drawn unit (`hread`), differs from `X` at some bit of a wire of a drawn
  unit of `U` (`unbound_drawn`), the bit `X`'s plurality didn't pick (`pr_verdict_le`);
* averaged over the draws, the miss is at most `δs` (`hdraw`) and each bit's minority weight is at most
  `√(E[bindBudget] / 2^256)`, since the two weights' product is the probability that two independent plays read the
  wire differently, which `hcoll` turns into the binding finder's collision (`weight_mul_le`).
-/

namespace Verity.Recursive

open scoped ENNReal
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy

namespace Target

open Claims

/-! ## Uniform averages -/

theorem ex_ind {β : Type} [Fintype β] (p : β → Prop) [DecidablePred p] :
    Ex (fun a => if p a then (1 : ℝ) else 0) = Pr p := by
  rw [Pr_eq_sum]; rfl

theorem pr_or_le {β : Type} [Fintype β] (p q : β → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  rw [← ex_ind, ← ex_ind, ← ex_ind, ← Ex_add]
  refine Ex_mono fun a => ?_
  by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq]

/-- The union bound over a finite set. -/
theorem pr_exists_finset_le {β I : Type} [Fintype β] (s : Finset I) (E : I → β → Prop) :
    Pr (fun a => ∃ i ∈ s, E i a) ≤ ∑ i ∈ s, Pr (E i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Pr_false]
  | insert j s hj ih =>
      rw [Finset.sum_insert hj]
      refine le_trans (Pr_mono fun a ha => ?_) ((pr_or_le _ _).trans (add_le_add le_rfl ih))
      obtain ⟨i, hi, he⟩ := ha
      rcases Finset.mem_insert.mp hi with rfl | hi
      · exact Or.inl he
      · exact Or.inr ⟨i, hi, he⟩

/-- Independent events on the two coordinates of a product. -/
theorem pr_mul {β γ : Type} [Fintype β] [Fintype γ] (p : β → Prop) (q : γ → Prop) :
    Pr (fun s : β × γ => p s.1 ∧ q s.2) = Pr p * Pr q := by
  classical
  rw [Pr_eq_sum (fun s : β × γ => p s.1 ∧ q s.2), Pr_eq_sum p, Pr_eq_sum q, div_mul_div_comm,
    Finset.sum_mul_sum, Fintype.sum_prod_type, Fintype.card_prod]
  push_cast
  congr 1
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
  by_cases hp : p a <;> by_cases hq : q b <;> simp [hp, hq]

theorem ex_mul {β γ : Type} [Fintype β] [Fintype γ] (f : β → ℝ) (g : γ → ℝ) :
    Ex f * Ex g = Ex (fun s : β × γ => f s.1 * g s.2) := by
  unfold Ex
  rw [div_mul_div_comm, Finset.sum_mul_sum, Fintype.sum_prod_type, Fintype.card_prod]
  push_cast
  rfl

theorem ex_add3 {β : Type} [Fintype β] [Nonempty β] (f g : β → ℝ) (c : ℝ) :
    Ex (fun x => f x + c + g x) = Ex f + c + Ex g := by
  rw [Ex_add (fun x => f x + c) g, Ex_add f (fun _ => c), Ex_const]

theorem ex_sum {β I : Type} [Fintype β] (s : Finset I) (f : I → β → ℝ) :
    Ex (fun x => ∑ i ∈ s, f i x) = ∑ i ∈ s, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

/-! ## The plurality word -/

theorem testBit_sum_pow (p : ℕ → Prop) [DecidablePred p] :
    ∀ n, (∑ j ∈ Finset.range n, if p j then 2 ^ j else 0) < 2 ^ n ∧
      ∀ i < n, ((∑ j ∈ Finset.range n, if p j then 2 ^ j else 0).testBit i = true ↔ p i)
  | 0 => ⟨by simp, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
  | n + 1 => by
    obtain ⟨hlt, hbit⟩ := testBit_sum_pow p n
    rw [Finset.sum_range_succ]
    by_cases hn : p n
    · rw [if_pos hn, Nat.add_comm]
      refine ⟨by rw [Nat.pow_succ]; omega, fun i hi => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
      · rw [Nat.testBit_two_pow_add_gt hi]; exact hbit i hi
      · rw [Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow hlt]; simp [hn]
    · rw [if_neg hn, Nat.add_zero]
      refine ⟨lt_of_lt_of_le hlt (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ n)), fun i hi => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
      · exact hbit i hi
      · rw [Nat.testBit_lt_two_pow hlt]; simp [hn]

/-- Bit `i < w` of the plurality word is set exactly where `p i`. -/
theorem getLsbD_ofNat_sum (w : ℕ) (p : ℕ → Prop) [DecidablePred p] {i : ℕ} (hi : i < w) :
    (BitVec.ofNat w (∑ j ∈ Finset.range w, if p j then 2 ^ j else 0)).getLsbD i = true ↔ p i := by
  rw [BitVec.getLsbD_ofNat, decide_eq_true hi, Bool.true_and]
  exact (testBit_sum_pow p w).2 i hi

theorem le_sqrt_mul {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) : a ≤ Real.sqrt (a * b) := by
  calc a = Real.sqrt (a * a) := (Real.sqrt_mul_self ha).symm
    _ ≤ Real.sqrt (a * b) := Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hab ha)

/-- **The plurality's minority**: where `bit` is set exactly when `a < b`, the weight of `!bit` (`b` at `true`, `a` at
`false`) is at most `√K` wherever the two weights' product is at most `K`. -/
theorem minority_le {a b K : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hK : b * a ≤ K) (bit : Bool) (hbit : bit = true ↔ a < b) :
    (if !bit then b else a) ≤ Real.sqrt K := by
  by_cases h : a < b
  · have : bit = true := hbit.2 h
    subst this
    calc (if !true then b else a) = a := rfl
      _ ≤ Real.sqrt (a * b) := le_sqrt_mul ha h.le
      _ ≤ Real.sqrt K := Real.sqrt_le_sqrt (by linarith [mul_comm a b])
  · have : bit = false := by
      cases bit
      · rfl
      · exact absurd (hbit.1 rfl) h
    subst this
    calc (if !false then b else a) = b := rfl
      _ ≤ Real.sqrt (b * a) := le_sqrt_mul hb (not_lt.1 h)
      _ ≤ Real.sqrt K := Real.sqrt_le_sqrt hK

end Target

open Target
open Claims (Ex Pr Ex_nonneg Ex_mono Ex_div Pr_nonneg Pr_mono Pr_le_one Pr_false)

/-- An event no play after `θ`, `F` and `c` meets has probability zero. -/
theorem prob_play_eq_zero (ns : List ℕ) (θ F c : ByteArray) (p : Ending → Prop)
    (hp : ∀ log e, ¬ p ⟨θ, F, c, log, e⟩) (σ : Game.Strategy (play ns θ F c)) : Game.prob p (play ns θ F c) σ = 0 := by
  refine (Game.prob_bind_of_const (fun log => Game.send ByteArray fun e => Game.ret (⟨θ, F, c, log, e⟩ : Ending)) p
    (fun _ => False) (fun log t => ?_) (releases ns) σ).trans (Game.prob_false _ _)
  simp [Game.prob, hp]

/-- **G's property at the deployment `q`** (`acc_of_accepts`' statement): where the compiled recursive component accepts
blobs that a statement names, that show a run of the deployed recursive verifier with the component's staged rows, and
whose records hold the run's live coins, the statement is dense and the run accepts. -/
def AccOfAccepts (q : Dep) : Prop :=
  ∀ (p : VStar.Recursive.Params) (c : SampledProofs.OneStage.Commitment) (e : VStar.Recursive.Evidence)
    (bs : Core.Blobs) (pub : SampledProofs.OneStage.Public) (prof : Core.Integrity),
    VStar.Recursive.checkBlobs p c e bs = .accept pub prof → ∀ x, Names p e bs x →
    ∀ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (q.D.V.k x) → Coin),
      Shows q.Bd.IVStar q.st x e.proofs bs (Claims.run FlockDeploy.sha512 P q.D.V x ω r).seen →
      RowsShow q.Bd.IVStar q.st x e.proofs bs (Claims.run FlockDeploy.sha512 P q.D.V x ω r).seen →
      LiveCoins q.Bd.IVStar q.st x (Claims.run FlockDeploy.sha512 P q.D.V x ω r).seen →
      RecDeploy.DenseAt x ∧ (Claims.run FlockDeploy.sha512 P q.D.V x ω r).out = .acc

/-- `AccOfAccepts` is `acc_of_accepts`. -/
theorem accOfAccepts (q : Dep) : AccOfAccepts q :=
  fun p c e bs _ _ hacc x hnames _ P ω r hshow hrows hlive =>
    acc_of_accepts q.wE q.w q.b q.aux q.nOf q.hd q.recRest p c e bs hacc x hnames P ω r hshow hrows hlive

open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **An accepted play shows an accepted run at a dense statement**: where the recursive configuration's verdict holds of
the play's outcome at the draw `ξ` and the session's tape and coins `s`, the coupling's statement is dense and the
deployed recursive verifier accepts the session's run at `s`. -/
theorem verdict_run (q : Dep) (hG : AccOfAccepts q) (V : Vocab d α) {θ F c : ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q θ F c σ)
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen)
    (ξ : S.Ξ) (s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin)) (h : RecVerdict V (S.out ξ s)) :
    RecDeploy.DenseAt (S.x ξ) ∧ (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2).out = .acc := by
  obtain ⟨-, -, ⟨j, p, hj, hp⟩, bs, hacc⟩ := h
  have hacc' : ((VStar.Recursive.Sampled.protocol.on bs).answer (S.out ξ s).params (S.out ξ s).commitment
      (S.out ξ s).evidence).verdict = .accept := hacc
  obtain ⟨p', c', ev, hp', -, he, pub, prof, hch⟩ :=
    Properties.Composition.answerAccepts (VStar.Recursive.Sampled.protocol.on bs) _ _ _ hacc'
  rw [hj] at hp'
  change VStar.Recursive.Sampled.paramsOfJson j = .ok p' at hp'
  rw [hp] at hp'
  cases hp'
  cases ev with
  | inl e₀ =>
      change VStar.Recursive.Sampled.checkBlobs (.inr p) c' (.inl e₀) (Core.Blobs.ofArray bs) = .accept pub prof
        at hch
      simp [VStar.Recursive.Sampled.checkBlobs] at hch
  | inr e =>
      have hch' : VStar.Recursive.checkBlobs p c' e (Core.Blobs.ofArray bs) = .accept pub prof := hch
      obtain ⟨hN, hS, hR⟩ := S.names_shows ξ s bs j p e hacc hj hp he
      exact hG p c' e _ pub prof hch' (S.x ξ) hN S.Ω (S.P ξ) s.1 s.2 hS hR (hlive ξ s.1 s.2)

/-- **The event a bit's weight counts** (`Dep.weight`'s): at the draw `ξ` and a run with the extractor's seed `s`, the
run accepts, `g` is a wire of a unit `ξ` draws, and the extractor returns a witness in the recursive relation whose
reading has `bit` at bit `i` of `g`. -/
def wEvent (q : Dep) (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (S : SessionIn V q θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (g : Fin (V.circuitOf F).N) (i : ℕ) (bit : Bool) (ξ : S.Ξ) (s : q.Sample V S ξ) : Prop :=
  (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
    (∃ u ∈ S.drawOf ξ, g ∈ (V.partsOf F).io u) ∧
    ∃ wt, (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt ∧ q.R (S.x ξ) wt ∧
      (readX (S.x ξ) wt g).getLsbD i = bit

theorem weight_eq (q : Dep) (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (S : SessionIn V q θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (g : Fin (V.circuitOf F).N) (i : ℕ) (bit : Bool) :
    q.weight V S readX g i bit = Ex fun ξ => Pr (wEvent q V S readX g i bit ξ) := rfl

theorem weight_nonneg (q : Dep) (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (S : SessionIn V q θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (g : Fin (V.circuitOf F).N) (i : ℕ) (bit : Bool) : 0 ≤ q.weight V S readX g i bit :=
  Ex_nonneg fun _ => Pr_nonneg _

/-- **The extracted transcript's bit is the plurality's**: bit `i` of gate `g` of `Dep.extract` is set exactly where its
weight at `true` exceeds its weight at `false`. -/
theorem extract_bit (q : Dep) (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (S : SessionIn V q θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (g : Fin (V.circuitOf F).N) {i : ℕ} (hi : i < (V.circuitOf F).w g) :
    (q.extract V S readX g).getLsbD i = true ↔
      q.weight V S readX g i false < q.weight V S readX g i true :=
  getLsbD_ofNat_sum _ (fun j => q.weight V S readX g j false < q.weight V S readX g j true) hi

/-- **Two plays that read a drawn wire's bit differently collide**: the product of a bit's two weights is at most the
binding finder's collision probability averaged over pairs of draws, so at most `E[bindBudget] / 2^256` under
`FinderCR`. -/
theorem weight_mul_le (q : Dep) (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}
    (S : SessionIn V q θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hbind : ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ))
    (g : Fin (V.circuitOf F).N) (i : ℕ) :
    q.weight V S readX g i true * q.weight V S readX g i false ≤
      Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ) := by
  rw [weight_eq, weight_eq, ex_mul, ← Ex_div]
  refine Ex_mono fun p => ?_
  rw [← pr_mul]
  refine le_trans (Pr_mono fun s hs => ?_) (hbind p.1 p.2)
  obtain ⟨⟨-, hd₁, w₁, he₁, hR₁, hb₁⟩, ⟨-, hd₂, w₂, he₂, hR₂, hb₂⟩⟩ := hs
  have hne : readX (S.x p.1) w₁ g ≠ readX (S.x p.2) w₂ g := fun h => by
    rw [h] at hb₁
    rw [hb₁] at hb₂
    exact Bool.noConfusion hb₂
  obtain ⟨m, m', hm, hcol⟩ := hcoll p.1 p.2 w₁ w₂ g hR₁ hR₂ hd₁ hd₂ hne
  refine ⟨m, m', ?_, hcol⟩
  first
    | (simp only [Dep.bindAt, he₁, he₂]; exact hm)
    | simp [Dep.bindAt, he₁, he₂, hm]

/-- **A drawn unbound unit gives a wire bit off the plurality**: where the witness's reading is correct on, and opened
at, every unit the draw holds (`hread`), a drawn unit unbound in `X` has a wire where `X` differs from the reading, at
some bit. -/
theorem unbound_drawn (V : Vocab d α) {θ F c : ByteArray} {Xw Xe : (V.circuitOf F).Transcript} {u : Fin (V.units F)}
    (hc : (V.partsOf F).Correct Xw u) (ho : ∀ g ∈ (V.partsOf F).io u, V.Opens θ F c g (Xw g))
    (hu : u ∈ Accounting.unbound (V.partsOf F) (V.Opens θ F c) Xe) :
    ∃ g ∈ (V.partsOf F).io u, ∃ i < (V.circuitOf F).w g, (Xw g).getLsbD i = !(Xe g).getLsbD i := by
  classical
  have hg : ∃ g ∈ (V.partsOf F).io u, Xe g ≠ Xw g := by
    by_contra hall
    push_neg at hall
    have hu' := (Finset.mem_filter.1 hu).2
    rcases hu' with hw | ⟨g, hg, hng⟩
    · exact hw ((V.partsOf F).correct_congr hall hc)
    · exact hng (by rw [hall g hg]; exact ho g hg)
  obtain ⟨g, hg, hne⟩ := hg
  refine ⟨g, hg, ?_⟩
  by_contra hall
  push_neg at hall
  refine hne (BitVec.eq_of_getLsbD_eq fun i hi => ?_)
  have := hall i hi
  cases h₁ : (Xe g).getLsbD i <;> cases h₂ : (Xw g).getLsbD i <;> simp_all

/-- **One draw's share**: at the draw `ξ`, the probability over the session's tape and coins that the recursive
configuration's verdict holds is at most the draw's miss of the unbound units (one where `ξ` misses them), plus
`t' · 2^{-λ}` (`RecursiveKSAt`'s `BitsLin2On`), plus, at each bit of each gate, the probability of the event the weight of
the bit the plurality didn't pick counts. -/
theorem pr_verdict_le (q : Dep) (hG : AccOfAccepts q)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, RecDeploy.DenseAt x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, RecDeploy.DenseAt x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, RecDeploy.DenseAt x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (hcr : ∀ ξ, q.SessionCR (nI (S.x ξ)) (vS (S.x ξ)) (S.x ξ) S.Ω (S.P ξ))
    (hread : ∀ ξ wt, q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf F).Correct (readX (S.x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen)
    (ξ : S.Ξ) :
    Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V (S.out ξ s)) ≤
      (if Disjoint (S.drawOf ξ) (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (q.extract V S readX)) then 1
        else 0) + q.tCost V S * (2 : ℝ) ^ (-lam) +
      ∑ g, ∑ i ∈ Finset.range ((V.circuitOf F).w g),
        Pr (wEvent q V S readX g i (!(q.extract V S readX g).getLsbD i) ξ) := by
  classical
  set Xe := q.extract V S readX with hXdef
  set U := Accounting.unbound (V.partsOf F) (V.Opens θ F c) Xe with hUdef
  have hsum0 : 0 ≤ ∑ g, ∑ i ∈ Finset.range ((V.circuitOf F).w g),
      Pr (wEvent q V S readX g i (!(Xe g).getLsbD i) ξ) :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => Pr_nonneg _
  have hc0 : 0 ≤ (q.tCost V S : ℝ) * (2 : ℝ) ^ (-lam) := by positivity
  by_cases hex : ∃ s, RecVerdict V (S.out ξ s)
  swap
  · push_neg at hex
    have h0 : Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V (S.out ξ s)) = 0 :=
      (le_antisymm (le_trans (Pr_mono fun s hs => hex s hs) (le_of_eq Pr_false)) (Pr_nonneg _))
    rw [h0]
    have : (0 : ℝ) ≤ if Disjoint (S.drawOf ξ) U then 1 else 0 := by split_ifs <;> norm_num
    linarith
  obtain ⟨s₀, hs₀⟩ := hex
  have hdense := (verdict_run q hG V S hlive ξ s₀ hs₀).1
  by_cases hD : Disjoint (S.drawOf ξ) U
  · rw [if_pos hD]
    have := Pr_le_one (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V (S.out ξ s))
    linarith
  rw [if_neg hD, zero_add]
  -- `RecursiveKSAt` at the session's prover: `PrBad ≤ t' · 2^{-λ}`
  have hB := Claims.Recursion.bitsLin2On_of_ksLin2On ?hS hKS
  case hS => exact RecDeploy.sessionAccepts _ _ _
  have hcrξ := hcr ξ
  have hbad : Pr (fun s : q.Sample V S ξ =>
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
        ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ≤
      q.tCost V S * (2 : ℝ) ^ (-lam) := by
    have h1 := hB q.decI hdecI q.decO hdecO (S.x ξ) hdense S.Ω (S.P ξ) (q.tCost V S) (le_max_left _ _)
      (fun ω r => le_trans (S.calls_le ξ ω r) (le_max_right _ _)) (nI (S.x ξ)) (hn _ hdense) (vS (S.x ξ))
      (hv _ hdense) hcrξ.1 hcrξ.2.1 hcrξ.2.2.1 hcrξ.2.2.2.1 hcrξ.2.2.2.2
    refine le_trans h1 (mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _))
    exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (neg_le_neg (hlam _ hdense))
  -- the verdict's event, over the run and the extractor's seed
  have hlift : Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V (S.out ξ s)) =
      Pr (fun s : q.Sample V S ξ => RecVerdict V (S.out ξ (s.1, s.2.1))) :=
    (Claims.Pr_drop (γ := q.E.C (S.x ξ)) (fun ω r => RecVerdict V (S.out ξ (ω, r)))).symm
  rw [hlift]
  obtain ⟨u₀, hu₀d, hu₀U⟩ := Finset.not_disjoint_iff.1 hD
  calc Pr (fun s : q.Sample V S ξ => RecVerdict V (S.out ξ (s.1, s.2.1)))
      ≤ Pr (fun s : q.Sample V S ξ =>
          ((Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
            ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ∨
          ∃ g ∈ Finset.univ, ∃ i ∈ Finset.range ((V.circuitOf F).w g),
            wEvent q V S readX g i (!(Xe g).getLsbD i) ξ s) := by
        refine Pr_mono fun s hs => ?_
        have hacc := (verdict_run q hG V S hlive ξ (s.1, s.2.1) hs).2
        by_cases hR : q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1
        · right
          cases hw : (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 with
          | none => rw [hw] at hR; exact (show False from hR).elim
          | some wt =>
              rw [hw] at hR
              obtain ⟨hcu, hou⟩ := hread ξ wt hR u₀ hu₀d
              obtain ⟨g, hg, i, hi, hbit⟩ := unbound_drawn V hcu hou hu₀U
              exact ⟨g, Finset.mem_univ _, i, Finset.mem_range.2 hi, hacc, ⟨u₀, hu₀d, hg⟩, wt, hw, hR, hbit⟩
        · exact Or.inl ⟨hacc, hR⟩
    _ ≤ Pr (fun s : q.Sample V S ξ =>
          (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
            ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) +
        Pr (fun s : q.Sample V S ξ => ∃ g ∈ Finset.univ, ∃ i ∈ Finset.range ((V.circuitOf F).w g),
            wEvent q V S readX g i (!(Xe g).getLsbD i) ξ s) := pr_or_le _ _
    _ ≤ q.tCost V S * (2 : ℝ) ^ (-lam) + ∑ g, ∑ i ∈ Finset.range ((V.circuitOf F).w g),
          Pr (wEvent q V S readX g i (!(Xe g).getLsbD i) ξ) := by
        refine add_le_add hbad ?_
        refine le_trans (pr_exists_finset_le Finset.univ _) (Finset.sum_le_sum fun g _ => ?_)
        exact pr_exists_finset_le _ _

/-- **The bound at one coupling**: averaged over the draws, the probability that the recursive configuration's verdict
holds is at most the draw's miss `δs`, plus `t' · 2^{-λ}`, plus the file's wire bits times `√(E[bindBudget] / 2^256)`. -/
theorem ex_verdict_le (q : Dep) (hG : AccOfAccepts q)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, RecDeploy.DenseAt x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, RecDeploy.DenseAt x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, RecDeploy.DenseAt x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcr : q.FinderCR V S collOf dColl nI vS) (δs : ℝ)
    (hdraw : Pr (fun ξ : S.Ξ =>
      Disjoint (S.drawOf ξ) (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (q.extract V S readX))) ≤ δs)
    (hread : ∀ ξ wt, q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf F).Correct (readX (S.x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) :
    Ex (fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V (S.out ξ s)) ≤
      δs + q.tCost V S * (2 : ℝ) ^ (-lam) +
        (∑ g, (V.circuitOf F).w g : ℕ) *
          Real.sqrt (Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ)) := by
  classical
  set K := Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ) with hKdef
  refine le_trans (Ex_mono fun ξ => pr_verdict_le q hG hKS hdecI hdecO nI vS lam hn hv hlam V S readX hcr.1
    hread hlive ξ) ?_
  rw [ex_add3, ex_ind, ex_sum]
  refine add_le_add (add_le_add hdraw le_rfl) ?_
  have hbit : ∀ g, ∀ i ∈ Finset.range ((V.circuitOf F).w g),
      Ex (fun ξ => Pr (wEvent q V S readX g i (!(q.extract V S readX g).getLsbD i) ξ)) ≤ Real.sqrt K := by
    intro g i hi
    rw [← weight_eq]
    have := minority_le (weight_nonneg q V S readX g i false) (weight_nonneg q V S readX g i true)
      (weight_mul_le q V S readX collOf dColl hcoll hcr.2 g i) _
      (extract_bit q V S readX g (Finset.mem_range.1 hi))
    cases h : (q.extract V S readX g).getLsbD i <;> rw [h] at this <;> first | exact this | simpa using this
  calc ∑ g, Ex (fun ξ => ∑ i ∈ Finset.range ((V.circuitOf F).w g),
          Pr (wEvent q V S readX g i (!(q.extract V S readX g).getLsbD i) ξ))
      = ∑ g, ∑ i ∈ Finset.range ((V.circuitOf F).w g),
          Ex (fun ξ => Pr (wEvent q V S readX g i (!(q.extract V S readX g).getLsbD i) ξ)) :=
        Finset.sum_congr rfl fun g _ => ex_sum _ _
    _ ≤ ∑ g, ∑ _i ∈ Finset.range ((V.circuitOf F).w g), Real.sqrt K :=
        Finset.sum_le_sum fun g _ => Finset.sum_le_sum fun i hi => hbit g i hi
    _ = (∑ g, (V.circuitOf F).w g : ℕ) * Real.sqrt K := by
        simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, Nat.cast_sum, Finset.sum_mul]

/-- **The target from G's property**: `sampledProofsRecursive`'s bound, at every strategy, from `AccOfAccepts` and
`sampledProofsRecursive`'s hypotheses. -/
theorem sampledProofsRecursive_of (q : Dep) (hG : AccOfAccepts q)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, RecDeploy.DenseAt x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, RecDeploy.DenseAt x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, RecDeploy.DenseAt x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (V : Vocab d α)
    (K₀ : ByteArray → ByteArray → ℕ) (δs : ℝ)
    (hK₀ : ∀ θ F (m : ℕ), (V.certOf θ).ε * V.units F < m → K₀ θ F ≤ m)
    (Sof : ∀ θ F c (σ : Game.Strategy (V.game θ F c)), SessionIn V q θ F c σ)
    (hdraw : ∀ θ F c σ (U : Finset (Fin (V.units F))), K₀ θ F ≤ U.card →
      Claims.Pr (fun ξ : (Sof θ F c σ).Ξ => Disjoint ((Sof θ F c σ).drawOf ξ) U) ≤ δs)
    (readX : (F : ByteArray) → Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (hX : ∀ θ F c σ, V.extract θ F c σ = q.extract V (Sof θ F c σ) (readX F))
    (hread : ∀ θ F c σ ξ wt, V.Registered θ F → q.R ((Sof θ F c σ).x ξ) wt → ∀ u ∈ (Sof θ F c σ).drawOf ξ,
      (V.partsOf F).Correct (readX F ((Sof θ F c σ).x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf F).io u, V.Opens θ F c g (readX F ((Sof θ F c σ).x ξ) wt g))
    (collOf : ByteArray → Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ θ F c σ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf F).N),
      q.R ((Sof θ F c σ).x ξ₁) w₁ → q.R ((Sof θ F c σ).x ξ₂) w₂ →
      (∃ u ∈ (Sof θ F c σ).drawOf ξ₁, g ∈ (V.partsOf F).io u) →
      (∃ u ∈ (Sof θ F c σ).drawOf ξ₂, g ∈ (V.partsOf F).io u) →
      readX F ((Sof θ F c σ).x ξ₁) w₁ g ≠ readX F ((Sof θ F c σ).x ξ₂) w₂ g →
      ∃ m m', collOf F ((Sof θ F c σ).x ξ₁) w₁ ((Sof θ F c σ).x ξ₂) w₂ = some (m, m') ∧
        Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ θ F c σ ξ ω r, LiveCoins q.Bd.IVStar q.st ((Sof θ F c σ).x ξ)
      (Claims.run FlockDeploy.sha512 ((Sof θ F c σ).P ξ) q.D.V ((Sof θ F c σ).x ξ) ω r).seen) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
      Game.prob (fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
        q.err V (Sof θ F c) (collOf F) dColl nI vS δs lam σ := by
  classical
  intro θ F c σ
  unfold Dep.err
  split_ifs with hcr
  swap
  · exact le_top
  set S := Sof θ F c σ with hSdef
  -- where the claim holds, the event is empty
  by_cases hH : V.Holds θ F c σ .sampled ⟨θ, F, c, [], ByteArray.empty⟩
  · refine le_trans (Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hH) _ σ) ?_
    rw [Game.prob_false]
    exact zero_le
  -- where the files fail the commit phase's check, no play is accepted
  by_cases hreg : V.Registered θ F
  swap
  · rw [show Game.prob (fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ = 0 from
      prob_play_eq_zero (V.rounds θ) θ F c _ (fun _ _ h => hreg h.1.1) σ]
    exact zero_le
  -- more than an `ε` share of the units is unbound in the extracted transcript
  have hU : K₀ θ F ≤ (Accounting.unbound (V.partsOf F) (V.Opens θ F c) (q.extract V S (readX F))).card := by
    refine hK₀ θ F _ ?_
    have hH' : ¬ ((Accounting.unbound (V.partsOf F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
        (V.certOf θ).ε * V.units F := hH
    rw [hX θ F c σ] at hH'
    exact not_le.mp hH'
  calc Game.prob (fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ
      ≤ ENNReal.ofReal (Ex fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) =>
          (fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (S.out ξ s)) := S.prob_le _
    _ ≤ ENNReal.ofReal (Ex fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) =>
          RecVerdict V (S.out ξ s)) :=
        ENNReal.ofReal_le_ofReal (Ex_mono fun ξ => Pr_mono fun s hs => hs.1)
    _ ≤ q.delta V S dColl δs lam := by
        unfold Dep.delta
        refine ENNReal.ofReal_le_ofReal ?_
        exact ex_verdict_le q hG hKS hdecI hdecO nI vS lam hn hv hlam V S (readX F) (collOf F) dColl hcr δs
          (hdraw θ F c σ _ hU) (fun ξ wt hR u hu => hread θ F c σ ξ wt hreg hR u hu)
          (fun ξ₁ ξ₂ w₁ w₂ g => hcoll θ F c σ ξ₁ ξ₂ w₁ w₂ g) (hlive θ F c σ)

/-- **The target: sampled proofs at the recursive configuration**, in the per-prover form over the
causal game (R4′, R5, R6), with the recursive configuration's verdict (`RecVerdict`). At all parameters `θ`, circuit
files `F` and commitments `c`, and for every strategy `σ`, the probability that the recursive component accepts while
sampled proofs' claim fails on the extractor's transcript (`¬ Holds .sampled`) is at most `Dep.err` at `σ`, the shared
`condError FinderCR δ σ`: where `FinderCR` holds at the finders built from `σ`,
`δs + t'(σ) · 2^{-λ} + bits · √(E[bindBudget(σ)] / 2^256)`, and `⊤` elsewhere, with `t'(σ)` the strategy's SHA-512
calls in one run (`Vocab.cost`) and every finder's budget defined from it. There is no hypothesis on the strategy but
`live-verifier`.

It takes these hypotheses:
* `RecursiveKSAt` at the deployed brings (`hKS`), and the honest half (`HonestHalf`);
* the decoders meet the list-decoding spec (`hdecI`, `hdecO`);
* the inner and setup call bounds `nI`, `vS` and the level `λ` (`hn`, `hv`, `hlam`), at the dense statements;
* `K₀` covers the claim's `ε` share (`hK₀`);
* `Sof` is the coupling at each strategy (`SessionIn`), and its draw misses any `K₀` units with
  probability at most `δs` (`hdraw`);
* the vocabulary's extractor is the recursive configuration's (`hX`, `Dep.extract` at `Sof`);
* the deployment's reading `readX` of a witness in the recursive relation is correct at, and opened by the commitment
  on, every unit the statement draws (`hread`); and `collOf`, from two such witnesses that read a wire of units drawn
  in both differently, outputs a SHA-512 collision (`hcoll`);
* the records hold the run's live coins (`LiveCoins`, `live-verifier`). `live-verifier` is admissible until the
  component reads the coins through the coin server's signed device log; then this hypothesis goes. -/
theorem sampledProofsRecursive (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hHon : HonestHalf q.wE q.w q.b q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, RecDeploy.DenseAt x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, RecDeploy.DenseAt x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, RecDeploy.DenseAt x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (V : Vocab d α)
    (K₀ : ByteArray → ByteArray → ℕ) (δs : ℝ)
    (hK₀ : ∀ θ F (m : ℕ), (V.certOf θ).ε * V.units F < m → K₀ θ F ≤ m)
    (Sof : ∀ θ F c (σ : Game.Strategy (V.game θ F c)), SessionIn V q θ F c σ)
    (hdraw : ∀ θ F c σ (U : Finset (Fin (V.units F))), K₀ θ F ≤ U.card →
      Claims.Pr (fun ξ : (Sof θ F c σ).Ξ => Disjoint ((Sof θ F c σ).drawOf ξ) U) ≤ δs)
    (readX : (F : ByteArray) → Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (hX : ∀ θ F c σ, V.extract θ F c σ = q.extract V (Sof θ F c σ) (readX F))
    (hread : ∀ θ F c σ ξ wt, V.Registered θ F → q.R ((Sof θ F c σ).x ξ) wt → ∀ u ∈ (Sof θ F c σ).drawOf ξ,
      (V.partsOf F).Correct (readX F ((Sof θ F c σ).x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf F).io u, V.Opens θ F c g (readX F ((Sof θ F c σ).x ξ) wt g))
    (collOf : ByteArray → Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ θ F c σ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf F).N),
      q.R ((Sof θ F c σ).x ξ₁) w₁ → q.R ((Sof θ F c σ).x ξ₂) w₂ →
      (∃ u ∈ (Sof θ F c σ).drawOf ξ₁, g ∈ (V.partsOf F).io u) →
      (∃ u ∈ (Sof θ F c σ).drawOf ξ₂, g ∈ (V.partsOf F).io u) →
      readX F ((Sof θ F c σ).x ξ₁) w₁ g ≠ readX F ((Sof θ F c σ).x ξ₂) w₂ g →
      ∃ m m', collOf F ((Sof θ F c σ).x ξ₁) w₁ ((Sof θ F c σ).x ξ₂) w₂ = some (m, m') ∧
        Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ θ F c σ ξ ω r, LiveCoins q.Bd.IVStar q.st ((Sof θ F c σ).x ξ)
      (Claims.run FlockDeploy.sha512 ((Sof θ F c σ).P ξ) q.D.V ((Sof θ F c σ).x ξ) ω r).seen) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
      Game.prob (fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
        q.err V (Sof θ F c) (collOf F) dColl nI vS δs lam σ :=
  fun θ F c σ =>
  sampledProofsRecursive_of q (accOfAccepts q) hKS hdecI hdecO nI vS lam hn hv hlam V K₀ δs hK₀ Sof hdraw readX hX
    hread collOf dColl hcoll hlive θ F c σ

end Verity.Recursive
