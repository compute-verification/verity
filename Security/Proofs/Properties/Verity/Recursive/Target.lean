module

public import Proofs.Properties.Verity.Recursive
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Recursion.BitsLin2
public import Proofs.Flock.Claims.Recursion.DeployFacts
public import Proofs.Core.Circuit

/-!
# The target, proved: sampled proofs at the recursive configuration

`spotRecursive` proves the target's bound at each strategy from the property `acc_of_accepts` states
(`AccOfAccepts`, the statement of `acc_of_accepts` at the deployment), at `acc_of_accepts`.

The proof, at a strategy `σ` whose finders meet `FinderCR` (elsewhere the error is `⊤`), where the files pass the
commit phase's check (elsewhere no play is accepted) and sampled proofs' claim fails on the extracted transcript `X`
(elsewhere the event is empty), so that the unbound units `U` number more than `ε n`:
* each accepted play shows, through the coupling (`SessionIn.names_shows`) and G, an accepted run of the recursive
  verifier at a dense statement (`verdict_run`);
* at each sample `ξ` such a play has, the sample misses `U`, or the run accepts while the extractor returns no witness in
  the recursive relation (`PrBad`, at most `Dep.ksAt`: `t' · 2^{-λ}` by `RecursiveKSAt` through
  `bitsLin2On_of_ksLin2On`, and at a statement at the deployment's tables also `2^{-167.7} + t' · 2^{-174.95}` through
  `bitsLin2KOn_of_ksLin2On` and `Dep.kappa_eps_oneTable`), or the
  extractor's witness, correct and opened on every sampled proof block (`hread`), differs from `X` at some bit of a wire of a sampled
  proof block of `U` (`unbound_drawn`), the bit `X`'s plurality didn't pick (`offEvent`, `pr_verdict_le`);
* averaged over the samples, the miss is at most `δs` (`hdraw`), and the last event is at most `√(min(W, 2n) · K)`,
  `K = E[bindBudget] / 2^256`, with `W` the file's wire bits and `n` its units. Both bounds pick one piece per play
  (`units_bound`), giving each piece a share `a` at most `p`, and rest on `hcoll`, which turns two independent plays
  that read a sampled wire differently into the binding finder's collision, so that a play off `X` at a bit collides with
  every play that reads `X`'s bit there (`majority_le_coll`):
  - at the bits (`ex_off_le_bits`), `p` is the bit's minority weight, at most its majority's
    (`minority_le_majority`), so `∑ a p ≤ K` and `(∑ a)² ≤ W ∑ a² ≤ WK`;
  - at the units (`ex_off_le_units`), `p_u` is the probability of sampling `u`, and the majority at any bit of `u`'s
    wires weighs at least `p_u / 2` (`drawn_le_weight`), so `∑ a p ≤ 2K` and `(∑ a)² ≤ n ∑ a² ≤ 2nK`.
-/

@[expose] public section

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

theorem ex_add3 {β : Type} [Fintype β] (f h g : β → ℝ) :
    Ex (fun x => f x + h x + g x) = Ex f + Ex h + Ex g := by
  rw [Ex_add (fun x => f x + h x) g, Ex_add f h]

theorem ex_sum {β I : Type} [Fintype β] (s : Finset I) (f : I → β → ℝ) :
    Ex (fun x => ∑ i ∈ s, f i x) = ∑ i ∈ s, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

theorem ex_congr {β : Type} [Fintype β] {f g : β → ℝ} (h : ∀ a, f a = g a) : Ex f = Ex g := by
  rw [funext h]

theorem ex_mul_const {β : Type} [Fintype β] (f : β → ℝ) (c : ℝ) : Ex (fun a => f a * c) = Ex f * c := by
  unfold Ex; rw [← Finset.sum_mul]; ring

theorem ex_const_mul {β : Type} [Fintype β] (c : ℝ) (f : β → ℝ) : Ex (fun a => c * f a) = c * Ex f := by
  unfold Ex; rw [← Finset.mul_sum]; ring

/-- An average over a product is the iterated average. -/
theorem ex_prod {β γ : Type} [Fintype β] [Fintype γ] (f : β → γ → ℝ) :
    Ex (fun p : β × γ => f p.1 p.2) = Ex (fun a => Ex (fun b => f a b)) := by
  unfold Ex
  rw [Fintype.sum_prod_type, Fintype.card_prod, ← Finset.sum_div]
  push_cast
  ring

theorem ex_comm {β γ : Type} [Fintype β] [Fintype γ] (f : β → γ → ℝ) :
    Ex (fun a => Ex (fun b => f a b)) = Ex (fun b => Ex (fun a => f a b)) := by
  rw [← ex_prod, ← ex_prod (fun b a => f a b)]
  unfold Ex
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type, Finset.sum_comm, Fintype.card_prod, Fintype.card_prod, mul_comm]

/-- An event of a product's two coordinates, averaged over the first. -/
theorem pr_prod {β γ : Type} [Fintype β] [Fintype γ] (P : β → γ → Prop) :
    Ex (fun b => Pr (fun c => P b c)) = Pr (fun s : β × γ => P s.1 s.2) := by
  classical
  rw [← ex_ind (fun s : β × γ => P s.1 s.2), ex_prod (fun b c => if P b c then (1 : ℝ) else 0)]
  exact ex_congr fun b => (ex_ind (P b)).symm

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

/-! ## Counting pieces -/

/-- **The pieces' share**: over plays `(ξ, s)`, `s` uniform on `Smp ξ` given `ξ`, let `off u` be events of the pieces
`u` of a finite `ι`, each within `drawn u`. Where a play in `off u` makes `c` times the probability that a second,
independent play collides with it (`coll`) at least that of `drawn u`, and two independent plays collide with
probability at most `K`, the plays in some `off u` have probability at most `√(c |ι| K)`. One piece picked per play
gives each piece a share `a u ≤ p u = Pr[drawn u]`; the pairs give `∑ a u · p u ≤ cK`, so
`(∑ a u)² ≤ |ι| ∑ a u² ≤ c |ι| K`. -/
theorem units_bound {Ξ ι : Type} [Fintype Ξ] [Fintype ι] {Smp : Ξ → Type} [∀ ξ, Fintype (Smp ξ)]
    (off drawn : ι → (ξ : Ξ) → Smp ξ → Prop) (coll : (ξ₁ : Ξ) → Smp ξ₁ → (ξ₂ : Ξ) → Smp ξ₂ → Prop) {c K : ℝ}
    (hc : 0 ≤ c) (hdrawn : ∀ u ξ s, off u ξ s → drawn u ξ s)
    (hlow : ∀ u ξ₁ (s₁ : Smp ξ₁), off u ξ₁ s₁ →
      Ex (fun ξ => Pr (drawn u ξ)) ≤ c * Ex (fun ξ₂ => Pr (coll ξ₁ s₁ ξ₂)))
    (hK : Ex (fun p : Ξ × Ξ => Pr (fun s : Smp p.1 × Smp p.2 => coll p.1 s.1 p.2 s.2)) ≤ K) :
    Ex (fun ξ => Pr (fun s : Smp ξ => ∃ u, off u ξ s)) ≤ Real.sqrt (c * Fintype.card ι * K) := by
  classical
  obtain ⟨sel, hsel, hsel'⟩ : ∃ sel : (ξ : Ξ) → Smp ξ → Option ι,
      (∀ ξ s u, sel ξ s = some u → off u ξ s) ∧ ∀ ξ s, (∃ u, off u ξ s) → ∃ u, sel ξ s = some u := by
    refine ⟨fun ξ s => if h : ∃ u, off u ξ s then some h.choose else none, fun ξ s u h => ?_,
      fun ξ s hex => ⟨hex.choose, dite_eq_left hex⟩⟩
    change (if h : ∃ u, off u ξ s then some h.choose else none) = some u at h
    split_ifs at h with hex
    rw [← Option.some.inj h]
    exact hex.choose_spec
  obtain ⟨a, ha⟩ : ∃ a : ι → ℝ, a = fun u => Ex fun ξ => Pr fun s : Smp ξ => sel ξ s = some u := ⟨_, rfl⟩
  obtain ⟨p, hp⟩ : ∃ p : ι → ℝ, p = fun u => Ex fun ξ => Pr (drawn u ξ) := ⟨_, rfl⟩
  have ha0 : ∀ u, 0 ≤ a u := fun u => by rw [ha]; exact Ex_nonneg fun _ => Pr_nonneg _
  have hap : ∀ u, a u ≤ p u := fun u => by
    rw [ha, hp]; exact Ex_mono fun ξ => Pr_mono fun s hs => hdrawn u ξ s (hsel ξ s u hs)
  have h1 : Ex (fun ξ => Pr (fun s : Smp ξ => ∃ u, off u ξ s)) ≤ ∑ u, a u := by
    rw [ha, ← ex_sum Finset.univ (fun u ξ => Pr fun s : Smp ξ => sel ξ s = some u)]
    refine Ex_mono fun ξ => le_trans (Pr_mono fun s hs => ?_) (pr_exists_finset_le Finset.univ _)
    obtain ⟨u, hu⟩ := hsel' ξ s hs
    exact ⟨u, Finset.mem_univ _, hu⟩
  have hpt : ∀ ξ (s : Smp ξ), (∑ u, (if sel ξ s = some u then (1 : ℝ) else 0) * p u) ≤
      c * Ex (fun ξ₂ => Pr (coll ξ s ξ₂)) := by
    intro ξ s
    cases h : sel ξ s with
    | none =>
        simp only [reduceCtorEq, ite_false, zero_mul, Finset.sum_const_zero]
        exact mul_nonneg hc (Ex_nonneg fun _ => Pr_nonneg _)
    | some u₀ =>
        rw [Finset.sum_eq_single u₀ (fun u _ hu => by simp [Ne.symm hu]) (fun h' => absurd (Finset.mem_univ u₀) h'),
          show (if some u₀ = some u₀ then (1 : ℝ) else 0) = 1 by simp, one_mul, hp]
        exact hlow u₀ ξ s (hsel ξ s u₀ h)
  have h3 : ∑ u, a u * p u ≤ c * K := by
    have e1 : ∑ u, a u * p u =
        Ex (fun ξ => Ex (fun s : Smp ξ => ∑ u, (if sel ξ s = some u then (1 : ℝ) else 0) * p u)) := by
      rw [ex_congr fun ξ => ex_sum Finset.univ (fun u (s : Smp ξ) => (if sel ξ s = some u then (1 : ℝ) else 0) * p u),
        ex_sum]
      refine Finset.sum_congr rfl fun u _ => ?_
      rw [ha, ← ex_mul_const]
      exact ex_congr fun ξ => by rw [ex_mul_const, ex_ind]
    have e2 : Ex (fun ξ => Ex (fun s : Smp ξ => c * Ex (fun ξ₂ => Pr (coll ξ s ξ₂)))) =
        c * Ex (fun p : Ξ × Ξ => Pr (fun s : Smp p.1 × Smp p.2 => coll p.1 s.1 p.2 s.2)) := by
      rw [ex_congr fun ξ => ex_const_mul c _, ex_const_mul,
        ex_prod (fun ξ₁ ξ₂ => Pr (fun s : Smp ξ₁ × Smp ξ₂ => coll ξ₁ s.1 ξ₂ s.2))]
      congr 1
      refine ex_congr fun ξ => ?_
      rw [ex_comm]
      exact ex_congr fun ξ₂ => pr_prod (fun (s : Smp ξ) (s₂ : Smp ξ₂) => coll ξ s ξ₂ s₂)
    rw [e1]
    calc _ ≤ Ex (fun ξ => Ex (fun s : Smp ξ => c * Ex (fun ξ₂ => Pr (coll ξ s ξ₂)))) :=
          Ex_mono fun ξ => Ex_mono fun s => hpt ξ s
      _ = _ := e2
      _ ≤ c * K := mul_le_mul_of_nonneg_left hK hc
  have hsq : ∑ u, a u ^ 2 ≤ ∑ u, a u * p u :=
    Finset.sum_le_sum fun u _ => by rw [sq]; exact mul_le_mul_of_nonneg_left (hap u) (ha0 u)
  have hcs := sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset ι)) (f := a)
  rw [Finset.card_univ] at hcs
  calc Ex (fun ξ => Pr (fun s : Smp ξ => ∃ u, off u ξ s)) ≤ ∑ u, a u := h1
    _ = Real.sqrt ((∑ u, a u) ^ 2) := (Real.sqrt_sq (Finset.sum_nonneg fun u _ => ha0 u)).symm
    _ ≤ Real.sqrt (c * Fintype.card ι * K) := Real.sqrt_le_sqrt <| by
        calc (∑ u, a u) ^ 2 ≤ (Fintype.card ι : ℝ) * ∑ u, a u ^ 2 := hcs
          _ ≤ (Fintype.card ι : ℝ) * (c * K) := mul_le_mul_of_nonneg_left (hsq.trans h3) (Nat.cast_nonneg _)
          _ = c * Fintype.card ι * K := by ring

end Target

open Target
open Claims (Ex Pr Ex_nonneg Ex_mono Ex_div Pr_nonneg Pr_mono Pr_le_one Pr_false)

/-- **`acc_of_accepts`' property at the deployment `q`** (`acc_of_accepts`' statement): where the compiled recursive component accepts
blobs that a statement names, that show a run of the deployed recursive verifier with the component's staged rows, and
whose records hold the run's live coins, the statement is dense and the run accepts. -/
def AccOfAccepts (q : Dep) : Prop :=
  ∀ (p : VStar.Recursive.Params) (c : Spot.OneStage.Commitment) (e : VStar.Recursive.Evidence)
    (bs : Core.Blobs) (pub : Spot.OneStage.Public) (prof : Core.Integrity),
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

/-- **Some run of the deployed recursive verifier accepts at `x`**, at some prover, tape and coins: the statements the
call bounds and the level are read at. -/
def Dep.RunAccepts (q : Dep) (x : Msg) : Prop :=
  ∃ (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (q.D.V.k x) → Coin),
    (Claims.run FlockDeploy.sha512 P q.D.V x ω r).out = .acc

/-- **`κ ≤ 2^{-167.7}` and `ε ≤ 2^{-174.95}` at a statement at the deployment's tables**, at an inner setup of at most
`2^32` calls and call bounds `n ≤ 2^32` and `v ≤ nS x · 138543104` (`setupCalls_deployedB_le_cap`'s, at most
`16 · 138543104 ≤ 2^32` at the verifier's 16-part cap): `RecDeploy.kappa_eps_oneTable_cap` at the deployed recursive
record. -/
theorem Dep.kappa_eps_oneTable (q : Dep) {x : Msg} (h : q.OneTable x) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 32)
    {n v : ℕ} (hn : n ≤ 2 ^ 32) (hv : v ≤ nS x * 138543104) :
    Claims.Recursion.κ q.D q.I x ≤ 2 ^ (-(167.7 : ℝ)) ∧
      Claims.Recursion.epsLin2 q.D q.I FlockDeploy.KS.CpDep x n v ≤ 2 ^ (-(174.95 : ℝ)) :=
  kappa_eps_oneTable_cap _root_.Flock.Readable.relation { deployedB q.b with tr := recTr q.aux q.nOf q.hd q.recRest }
    (DeployedR.layR OpenRows.elemR) (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols q.wE FlockDeploy.joinR
    rowInDeployed DeployedR.finalR q.w q.decI h hS hn hv

open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **An accepted play shows an accepted run at a dense statement**: where the recursive configuration's verdict holds of
the play's outcome at the sample `ξ` and the session's tape and coins `s`, the coupling's statement is dense and the
deployed recursive verifier accepts the session's run at `s`. -/
theorem verdict_run (q : Dep) (hG : AccOfAccepts q) (V : Vocab d α) (k : Verity.Registration.Composite)
    {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q k θ F c σ)
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen)
    (ξ : S.Ξ) (s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin)) (h : RecVerdict V k (S.out ξ s)) :
    RecDeploy.DenseAt (S.x ξ) ∧ (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2).out = .acc := by
  obtain ⟨-, -, a, cb, eb, hcase, hda, j, p, hj, hp⟩ := h
  obtain ⟨p', c', ev, hp', -, he, pub, prof, hch⟩ := id hda
  rw [hj] at hp'
  change VStar.Recursive.Sampled.paramsOfJson j = .ok p' at hp'
  rw [hp] at hp'
  cases hp'
  cases ev with
  | inl e₀ =>
      change VStar.Recursive.Sampled.checkBlobs (.inr p) c' (.inl e₀)
        (Core.Blobs.ofArray (S.out ξ s).sent.blobs) = .accept pub prof at hch
      simp [VStar.Recursive.Sampled.checkBlobs] at hch
  | inr e =>
      have hch' : VStar.Recursive.checkBlobs p c' e (Core.Blobs.ofArray (S.out ξ s).sent.blobs) = .accept pub prof :=
        hch
      obtain ⟨hN, hS, hR⟩ := S.names_shows ξ s a cb eb j p e hcase hda hj hp he
      exact hG p c' e _ pub prof hch' (S.x ξ) hN S.Ω (S.P ξ) s.1 s.2 hS hR (hlive ξ s.1 s.2)

/-- **The event a bit's weight counts** (`Dep.weight`'s): at the sample `ξ` and a run with the extractor's seed `s`, the
run accepts, `g` is a wire of a unit `ξ` samples, and the extractor returns a witness in the recursive relation whose
reading has `bit` at bit `i` of `g`. -/
def wEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) (i : ℕ) (bit : Bool) (ξ : S.Ξ) (s : q.Sample V S ξ) : Prop :=
  (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
    (∃ u ∈ S.drawOf ξ, g ∈ (V.partsOf θ F).io u) ∧
    ∃ wt, (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt ∧ q.R (S.x ξ) wt ∧
      (readX (S.x ξ) wt g).getLsbD i = bit

theorem weight_eq (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) (i : ℕ) (bit : Bool) :
    q.weight V S readX g i bit = Ex fun ξ => Pr (wEvent q V S readX g i bit ξ) := rfl

/-- **The extracted transcript's bit is the plurality's**: bit `i` of gate `g` of `Dep.extract` is set exactly where its
weight at `true` exceeds its weight at `false`. -/
theorem extract_bit (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) {i : ℕ} (hi : i < (V.circuitOf θ F).w g) :
    (q.extract V S readX g).getLsbD i = true ↔
      q.weight V S readX g i false < q.weight V S readX g i true :=
  getLsbD_ofNat_sum _ (fun j => q.weight V S readX g j false < q.weight V S readX g j true) hi

/-- **The plurality's minority weighs at most its majority**: at a bit `i` of a wire `g`, the weight of the bit
`Dep.extract` didn't pick is at most that of the bit it picked. -/
theorem minority_le_majority (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (g : Fin (V.circuitOf θ F).N) {i : ℕ} (hi : i < (V.circuitOf θ F).w g) :
    q.weight V S readX g i (!(q.extract V S readX g).getLsbD i) ≤
      q.weight V S readX g i ((q.extract V S readX g).getLsbD i) := by
  have hx := extract_bit q V S readX g hi
  cases hb : (q.extract V S readX g).getLsbD i
  · exact not_lt.1 fun h => by rw [hb] at hx; exact Bool.false_ne_true (hx.2 h)
  · rw [hb] at hx; exact (hx.1 rfl).le

/-- **The event the binding term counts at the unit `u`**: at the sample `ξ` and a run with the extractor's seed `s`, the
run accepts, `ξ` samples `u`, and the extractor returns a witness in the recursive relation whose reading differs from
`Xt` at a bit of a wire of `u`. -/
def offEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace) (Xt : (V.circuitOf θ F).Trace)
    (u : Fin (V.units θ F)) (ξ : S.Ξ) (s : q.Sample V S ξ) : Prop :=
  (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧ u ∈ S.drawOf ξ ∧
    ∃ wt, (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt ∧ q.R (S.x ξ) wt ∧
      ∃ g ∈ (V.partsOf θ F).io u, ∃ i < (V.circuitOf θ F).w g, (readX (S.x ξ) wt g).getLsbD i = !(Xt g).getLsbD i

/-- **A run's sample of the unit `u`**: at the sample `ξ` and a run with the extractor's seed `s`, the run accepts, `ξ`
samples `u`, and the extractor returns a witness in the recursive relation. -/
def drawEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ) (u : Fin (V.units θ F)) (ξ : S.Ξ)
    (s : q.Sample V S ξ) : Prop :=
  (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧ u ∈ S.drawOf ξ ∧
    ∃ wt, (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt ∧ q.R (S.x ξ) wt

/-- **The binding finder's collision** (`bindAt`'s event) at the samples `ξ₁`, `ξ₂` and the runs with seeds `s₁`, `s₂`. -/
def collEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (ξ₁ : S.Ξ) (s₁ : q.Sample V S ξ₁) (ξ₂ : S.Ξ) (s₂ : q.Sample V S ξ₂) : Prop :=
  ∃ m m', (q.bindAt V S collOf dColl ξ₁ ξ₂).out (s₁, s₂) = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m'

/-- **The event the binding term counts at the wire bit `b`**: `wEvent` at bit `b.2` of the wire `b.1`, reading the
bit `Dep.extract` didn't pick there. -/
def bitEvent (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {σ : Game.Strategy (V.game θ F c)}
    {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) (ξ : S.Ξ) (s : q.Sample V S ξ) : Prop :=
  wEvent q V S readX b.1 b.2 (!(q.extract V S readX b.1).getLsbD b.2) ξ s

/-- **A run off the plurality collides with every run on it**: where a run at the sample `ξ₁` with seed `s₁` returns a
witness in the recursive relation whose reading, at a bit `i` of a wire `g` that `ξ₁` samples, is not the bit
`Dep.extract` picked, the weight of the picked bit is at most the probability, over a second, independent sample and
run, that the binding finder at the two collides (`hcoll`). -/
theorem majority_le_coll (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    {ξ₁ : S.Ξ} {s₁ : q.Sample V S ξ₁} {w₁ : Claims.Flock.Wit} {g : Fin (V.circuitOf θ F).N} {i : ℕ}
    (hd₁ : ∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u)
    (he₁ : (q.E.eval FlockDeploy.sha512 (S.P ξ₁) (S.x ξ₁) s₁.1 s₁.2.1 s₁.2.2).1 = some w₁) (hR₁ : q.R (S.x ξ₁) w₁)
    (hb₁ : (readX (S.x ξ₁) w₁ g).getLsbD i = !(q.extract V S readX g).getLsbD i) :
    q.weight V S readX g i ((q.extract V S readX g).getLsbD i) ≤
      Ex (fun ξ₂ => Pr (collEvent q V S collOf dColl ξ₁ s₁ ξ₂)) := by
  rw [weight_eq]
  refine Ex_mono fun ξ₂ => Pr_mono fun s₂ hs₂ => ?_
  obtain ⟨-, hd₂, w₂, he₂, hR₂, hb₂⟩ := hs₂
  have hne : readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g := fun h => by
    rw [h, hb₂] at hb₁
    cases hx : (q.extract V S readX g).getLsbD i <;> simp [hx] at hb₁
  obtain ⟨m, m', hm, hcol⟩ := hcoll ξ₁ ξ₂ w₁ w₂ g hR₁ hR₂ hd₁ hd₂ hne
  refine ⟨m, m', ?_, hcol⟩
  simp only [Dep.bindAt, he₁, he₂]
  exact hm

/-- **Two independent plays collide at most `E[bindBudget] / 2^256`** under `FinderCR`'s binding premise (`hbind`):
the binding finder at each pair of samples is the pair of plays. -/
theorem ex_coll_le (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hbind : ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ)) :
    Ex (fun p : S.Ξ × S.Ξ => Pr (fun s : q.Sample V S p.1 × q.Sample V S p.2 =>
      collEvent q V S collOf dColl p.1 s.1 p.2 s.2)) ≤
      Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ) := by
  rw [← Ex_div]
  exact Ex_mono fun p => hbind p.1 p.2

/-- **The plurality's bit weighs at least half a sampled proof block**: at a bit `i` of a wire `g` of the unit `u`, the weight of
the bit `Dep.extract` picked there is at least half the probability that the run accepts, samples `u` and the extractor
returns a witness in the recursive relation. -/
theorem drawn_le_weight (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace) {u : Fin (V.units θ F)}
    {g : Fin (V.circuitOf θ F).N} (hg : g ∈ (V.partsOf θ F).io u) {i : ℕ} (hi : i < (V.circuitOf θ F).w g) :
    Ex (fun ξ => Pr (drawEvent q V S u ξ)) ≤ 2 * q.weight V S readX g i ((q.extract V S readX g).getLsbD i) := by
  have hsum : Ex (fun ξ => Pr (drawEvent q V S u ξ)) ≤
      q.weight V S readX g i true + q.weight V S readX g i false := by
    rw [weight_eq, weight_eq, ← Claims.Ex_add]
    refine Ex_mono fun ξ => le_trans (Pr_mono fun s hs => ?_) (pr_or_le _ _)
    obtain ⟨hacc, hu, wt, hw, hR⟩ := hs
    cases hb : (readX (S.x ξ) wt g).getLsbD i
    · exact Or.inr ⟨hacc, ⟨u, hu, hg⟩, wt, hw, hR, hb⟩
    · exact Or.inl ⟨hacc, ⟨u, hu, hg⟩, wt, hw, hR, hb⟩
  have hx := extract_bit q V S readX g hi
  cases hb : (q.extract V S readX g).getLsbD i
  · have : q.weight V S readX g i true ≤ q.weight V S readX g i false :=
      not_lt.1 fun h => by rw [hb] at hx; exact Bool.false_ne_true (hx.2 h)
    linarith
  · have : q.weight V S readX g i false < q.weight V S readX g i true := by rw [hb] at hx; exact hx.1 rfl
    linarith

/-- **The binding term at the bits**: averaged over the samples, the probability that the run accepts while the
extractor's witness differs from the plurality `Dep.extract` at a bit of a wire of a sampled proof block is at most
`√(W · E[bindBudget] / 2^256)`, `W` the file's wire bits (`units_bound` over the bits): a run off the plurality at a
bit collides, through `hcoll`, with every run that reads the plurality's bit there (`majority_le_coll`), which weighs
at least the bit's minority (`minority_le_majority`). -/
theorem ex_off_le_bits (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hbind : ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ)) :
    Ex (fun ξ => Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s)) ≤
      Real.sqrt ((∑ g, (V.circuitOf θ F).w g : ℕ) *
        (Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ))) := by
  classical
  have hlow : ∀ b ξ₁ (s₁ : q.Sample V S ξ₁), bitEvent q V S readX b ξ₁ s₁ →
      Ex (fun ξ => Pr (bitEvent q V S readX b ξ)) ≤
        1 * Ex (fun ξ₂ => Pr (collEvent q V S collOf dColl ξ₁ s₁ ξ₂)) := by
    intro b ξ₁ s₁ hs
    obtain ⟨-, hd₁, w₁, he₁, hR₁, hb₁⟩ := hs
    rw [one_mul, show Ex (fun ξ => Pr (bitEvent q V S readX b ξ)) =
      q.weight V S readX b.1 b.2 (!(q.extract V S readX b.1).getLsbD b.2) from rfl]
    exact (minority_le_majority q V S readX b.1 b.2.isLt).trans
      (majority_le_coll q V S readX collOf dColl hcoll hd₁ he₁ hR₁ hb₁)
  have h := units_bound (Smp := q.Sample V S) (bitEvent q V S readX) (bitEvent q V S readX)
    (collEvent q V S collOf dColl) zero_le_one (fun _ _ _ hs => hs) hlow (ex_coll_le q V S collOf dColl hbind)
  simp only [Fintype.card_sigma, Fintype.card_fin, one_mul] at h
  refine le_trans (Ex_mono fun ξ => Pr_mono fun s hs => ?_) h
  obtain ⟨u, hacc, hu, wt, hw, hR, g, hg, i, hi, hbit⟩ := hs
  exact ⟨⟨g, i, hi⟩, hacc, ⟨u, hu, hg⟩, wt, hw, hR, hbit⟩

/-- **The binding term at the units**: averaged over the samples, the probability that the run accepts while the
extractor's witness differs from the plurality `Dep.extract` at a bit of a wire of a sampled proof block is at most
`√(2n · E[bindBudget] / 2^256)`, `n` the file's units (`units_bound`): a run off the plurality at a bit of a wire of `u`
collides, through `hcoll`, with every run that reads the plurality's bit there, which weighs at least half `u`'s
sample (`drawn_le_weight`). -/
theorem ex_off_le_units (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hbind : ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ)) :
    Ex (fun ξ => Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s)) ≤
      Real.sqrt (2 * V.units θ F *
        (Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ))) := by
  classical
  have hlow : ∀ u ξ₁ (s₁ : q.Sample V S ξ₁), offEvent q V S readX (q.extract V S readX) u ξ₁ s₁ →
      Ex (fun ξ => Pr (drawEvent q V S u ξ)) ≤
        2 * Ex (fun ξ₂ => Pr (collEvent q V S collOf dColl ξ₁ s₁ ξ₂)) := by
    intro u ξ₁ s₁ hs
    obtain ⟨-, hu₁, w₁, he₁, hR₁, g, hg, i, hi, hb₁⟩ := hs
    exact le_trans (drawn_le_weight q V S readX hg hi) (mul_le_mul_of_nonneg_left
      (majority_le_coll q V S readX collOf dColl hcoll ⟨u, hu₁, hg⟩ he₁ hR₁ hb₁) (by norm_num))
  have h := units_bound (Smp := q.Sample V S) (offEvent q V S readX (q.extract V S readX)) (drawEvent q V S)
    (collEvent q V S collOf dColl) (by norm_num : (0 : ℝ) ≤ 2) (fun u ξ s hs => by
      obtain ⟨hacc, hu, wt, hw, hR, -⟩ := hs
      exact ⟨hacc, hu, wt, hw, hR⟩) hlow (ex_coll_le q V S collOf dColl hbind)
  rwa [Fintype.card_fin] at h

/-- **A sampled unbound unit gives a wire bit off the plurality**: where the witness's reading is correct on, and opened
at, every unit the sample holds (`hread`), a sampled proof block unbound in `X` has a wire where `X` differs from the reading, at
some bit. -/
theorem unbound_drawn (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray} {Xw Xe : (V.circuitOf θ F).Trace}
    {u : Fin (V.units θ F)}
    (hc : (V.partsOf θ F).Correct Xw u) (ho : ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (Xw g))
    (hu : u ∈ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) Xe) :
    ∃ g ∈ (V.partsOf θ F).io u, ∃ i < (V.circuitOf θ F).w g, (Xw g).getLsbD i = !(Xe g).getLsbD i := by
  classical
  have hg : ∃ g ∈ (V.partsOf θ F).io u, Xe g ≠ Xw g := by
    by_contra hall
    push_neg at hall
    have hu' := (Finset.mem_filter.1 hu).2
    rcases hu' with hw | ⟨g, hg, hng⟩
    · exact hw ((V.partsOf θ F).correct_congr hall hc)
    · exact hng (by rw [hall g hg]; exact ho g hg)
  obtain ⟨g, hg, hne⟩ := hg
  refine ⟨g, hg, ?_⟩
  by_contra hall
  push_neg at hall
  refine hne (BitVec.eq_of_getLsbD_eq fun i hi => ?_)
  have := hall i hi
  cases h₁ : (Xe g).getLsbD i <;> cases h₂ : (Xw g).getLsbD i <;> simp_all

/-- **One sample's share**: at the sample `ξ`, the probability over the session's tape and coins that the recursive
configuration's verdict holds is at most the sample's miss of the unbound units (one where `ξ` misses them), plus
the knowledge-soundness term at its statement (`Dep.ksAt`: `RecursiveKSAt`'s `BitsLin2On`, and at the deployment's
tables the least of it and its `BitsLin2KOn` with `Dep.kappa_eps_oneTable`), plus the probability that the run accepts
while the extractor's witness differs from the plurality at a bit of a wire of a sampled proof block (`offEvent`). -/
theorem pr_verdict_le (q : Dep) (hG : AccOfAccepts q)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (hcap : ∀ x, q.RunAccepts x → q.OneTable x →
      (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI x ≤ 2 ^ 32 ∧ vS x ≤ nS x * 138543104)
    (V : Vocab d α) (k : Verity.Registration.Composite) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (hcr : ∀ ξ, q.SessionCR (nI (S.x ξ)) (vS (S.x ξ)) (S.x ξ) S.Ω (S.P ξ))
    (hread : ∀ ξ wt, q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen)
    (ξ : S.Ξ) :
    Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) ≤
      (if Disjoint (S.drawOf ξ) (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (q.extract V S readX)) then 1
        else 0) + q.ksAt V S lam (S.x ξ) +
      Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s) := by
  classical
  set Xe := q.extract V S readX with hXdef
  set U := Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) Xe with hUdef
  have hsum0 : 0 ≤ Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX Xe u ξ s) := Pr_nonneg _
  have hc0 : 0 ≤ q.ksAt V S lam (S.x ξ) := by
    unfold Dep.ksAt
    split_ifs
    · exact le_min (by positivity) (by positivity)
    · positivity
  by_cases hex : ∃ s, RecVerdict V k (S.out ξ s)
  swap
  · push_neg at hex
    have h0 : Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) = 0 :=
      (le_antisymm (le_trans (Pr_mono fun s hs => hex s hs) (le_of_eq Pr_false)) (Pr_nonneg _))
    rw [h0]
    have : (0 : ℝ) ≤ if Disjoint (S.drawOf ξ) U then 1 else 0 := by split_ifs <;> norm_num
    linarith
  obtain ⟨s₀, hs₀⟩ := hex
  have hdense := (verdict_run q hG V k S hlive ξ s₀ hs₀).1
  have hrun : q.RunAccepts (S.x ξ) := ⟨S.Ω, S.P ξ, s₀.1, s₀.2, (verdict_run q hG V k S hlive ξ s₀ hs₀).2⟩
  by_cases hD : Disjoint (S.drawOf ξ) U
  · rw [if_pos hD]
    have := Pr_le_one (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s))
    linarith
  rw [if_neg hD, zero_add]
  -- `RecursiveKSAt` at the session's prover: `PrBad ≤ t' · 2^{-λ}`, and `≤ κ + t' · ε` at the deployment's tables
  have hcrξ := hcr ξ
  have hlamB : Pr (fun s : q.Sample V S ξ =>
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
        ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ≤
      q.tCost V S * (2 : ℝ) ^ (-lam) := by
    have hB := Claims.Recursion.bitsLin2On_of_ksLin2On ?hS hKS
    case hS => exact RecDeploy.sessionAccepts _ _ _
    have h1 := hB q.decI hdecI q.decO hdecO (S.x ξ) hdense S.Ω (S.P ξ) (q.tCost V S) (le_max_left _ _)
      (fun ω r => le_trans (S.calls_le ξ ω r) (le_max_right _ _)) (nI (S.x ξ)) (hn _ hrun) (vS (S.x ξ))
      (hv _ hrun) hcrξ.1 hcrξ.2.1 hcrξ.2.2.1 hcrξ.2.2.2.1 hcrξ.2.2.2.2
    refine le_trans h1 (mul_le_mul_of_nonneg_left ?_ (Nat.cast_nonneg _))
    exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (neg_le_neg (hlam _ hrun))
  have hbad : Pr (fun s : q.Sample V S ξ =>
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
        ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ≤
      q.ksAt V S lam (S.x ξ) := by
    unfold Dep.ksAt
    split_ifs with h1t
    · refine le_min ?_ hlamB
      have hB := Claims.Recursion.bitsLin2KOn_of_ksLin2On ?hS hKS
      case hS => exact RecDeploy.sessionAccepts _ _ _
      have h1 := hB q.decI hdecI q.decO hdecO (S.x ξ) hdense S.Ω (S.P ξ) (q.tCost V S) (le_max_left _ _)
        (fun ω r => le_trans (S.calls_le ξ ω r) (le_max_right _ _)) (nI (S.x ξ)) (hn _ hrun) (vS (S.x ξ))
        (hv _ hrun) hcrξ.1 hcrξ.2.1 hcrξ.2.2.1 hcrξ.2.2.2.1 hcrξ.2.2.2.2
      obtain ⟨hSc, hnc, hvc⟩ := hcap _ hrun h1t
      obtain ⟨hκ, hε⟩ := q.kappa_eps_oneTable h1t hSc hnc hvc
      exact le_trans h1 (add_le_add hκ (mul_le_mul_of_nonneg_left hε (Nat.cast_nonneg _)))
    · exact hlamB
  -- the verdict's event, over the run and the extractor's seed
  have hlift : Pr (fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) =
      Pr (fun s : q.Sample V S ξ => RecVerdict V k (S.out ξ (s.1, s.2.1))) :=
    (Claims.Pr_drop (γ := q.E.C (S.x ξ)) (fun ω r => RecVerdict V k (S.out ξ (ω, r)))).symm
  rw [hlift]
  obtain ⟨u₀, hu₀d, hu₀U⟩ := Finset.not_disjoint_iff.1 hD
  calc Pr (fun s : q.Sample V S ξ => RecVerdict V k (S.out ξ (s.1, s.2.1)))
      ≤ Pr (fun s : q.Sample V S ξ =>
          ((Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
            ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) ∨
          ∃ u, offEvent q V S readX Xe u ξ s) := by
        refine Pr_mono fun s hs => ?_
        have hacc := (verdict_run q hG V k S hlive ξ (s.1, s.2.1) hs).2
        by_cases hR : q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1
        · right
          cases hw : (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 with
          | none => rw [hw] at hR; exact (show False from hR).elim
          | some wt =>
              rw [hw] at hR
              obtain ⟨hcu, hou⟩ := hread ξ wt hR u₀ hu₀d
              obtain ⟨g, hg, i, hi, hbit⟩ := unbound_drawn V hcu hou hu₀U
              exact ⟨u₀, hacc, hu₀d, wt, hw, hR, g, hg, i, hi, hbit⟩
        · exact Or.inl ⟨hacc, hR⟩
    _ ≤ Pr (fun s : q.Sample V S ξ =>
          (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
            ¬ q.R.ext (S.x ξ) (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1) +
        Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX Xe u ξ s) := pr_or_le _ _
    _ ≤ q.ksAt V S lam (S.x ξ) + Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX Xe u ξ s) :=
        add_le_add hbad le_rfl

/-- **The bound at one coupling**: averaged over the samples, the probability that the recursive configuration's verdict
holds is at most the sample's miss `δs`, plus the knowledge-soundness term averaged over the samples (`Dep.ksAt`), plus
`√(min(W, 2n) · E[bindBudget] / 2^256)`, with `W` the file's wire bits and `n` its units (`ex_off_le_bits`,
`ex_off_le_units`). -/
theorem ex_verdict_le (q : Dep) (hG : AccOfAccepts q)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (hcap : ∀ x, q.RunAccepts x → q.OneTable x →
      (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI x ≤ 2 ^ 32 ∧ vS x ≤ nS x * 138543104)
    (V : Vocab d α) (k : Verity.Registration.Composite) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcr : q.FinderCR V S collOf dColl nI vS) (δs : ℝ)
    (hdraw : Pr (fun ξ : S.Ξ =>
      Disjoint (S.drawOf ξ) (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (q.extract V S readX))) ≤ δs)
    (hread : ∀ ξ wt, q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) :
    Ex (fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) => RecVerdict V k (S.out ξ s)) ≤
      δs + Ex (fun ξ => q.ksAt V S lam (S.x ξ)) +
        Real.sqrt (min ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) *
          (Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ))) := by
  classical
  refine le_trans (Ex_mono fun ξ => pr_verdict_le q hG hKS hdecI hdecO nI vS lam hn hv hlam hcap V k S readX hcr.1
    hread hlive ξ) ?_
  rw [ex_add3, ex_ind]
  refine add_le_add (add_le_add hdraw le_rfl) ?_
  rcases le_total ((∑ g, (V.circuitOf θ F).w g : ℕ) : ℝ) (2 * V.units θ F) with h | h
  · rw [min_eq_left h]
    exact ex_off_le_bits q V S readX collOf dColl hcoll hcr.2
  · rw [min_eq_right h]
    exact ex_off_le_units q V S readX collOf dColl hcoll hcr.2

/-- **The target: sampled proofs at the recursive configuration**, in the per-prover form over the causal game (R4′,
R5, R6), with the recursive configuration's verdict (`RecVerdict`). At composite `k`, the parameters `θ`, the
committed files `F`, the commitment `c` and a strategy `σ`, with `S` its recursive session (`SessionIn`), the
probability that the recursive component accepts while sampled proofs' claim fails on the extractor's transcript
(`¬ Holds k .sampled`) is at most `Dep.errAt` at `S`, the shared `condError FinderCR δ`: where `FinderCR` holds at
the finders built from `σ`, `δs + E_ξ[ksAt] + √(min(bits, 2n) · E[bindBudget(σ)] / 2^256)` with `n` the units,
and `⊤` elsewhere, where `ksAt` is `t'(σ) · 2^{-λ}`, and at a sample whose statement is at the deployment's tables the
least of it and `2^{-167.7} + t'(σ) · 2^{-174.95}` (`Dep.ksAt`), with `t'(σ)` the strategy's SHA-512 calls in one run
(`Vocab.cost`) and every finder's budget defined from it. There is no hypothesis on the strategy but `live-verifier`.

It takes these hypotheses:
* `RecursiveKSAt` at the deployed brings (`hKS`);
* the decoders meet the list-decoding spec (`hdecI`, `hdecO`);
* the inner and setup call bounds `nI`, `vS` and the level `λ` (`hn`, `hv`, `hlam`), and at the deployment's tables
  the inner setup's calls and `nI` at most `2^32` and `vS` at most `nS · 138543104`, `setupCalls_deployedB_le_cap`'s
  bound (`hcap`), at the statements some run of the deployed recursive verifier accepts (`Dep.RunAccepts`);
* `K₀` covers the claim's `ε` share (`hK₀`), and the session's sample misses any `K₀` units with probability at most `δs`
  (`hdraw`);
* the vocabulary's extractor is the recursive configuration's (`hX`, `Dep.extract` at `S`);
* the deployment's reading `readX` of a witness in the recursive relation is correct at, and opened by the commitment
  on, every unit the statement samples (`hread`); and `collOf`, from two such witnesses that read a wire of units sampled
  in both differently, outputs a SHA-512 collision (`hcoll`);
* the records hold the run's live coins (`LiveCoins`, `live-verifier`). `live-verifier` is admissible until the
  component reads the coins through the challenger's live-coin part's signed device log; then this hypothesis goes.

The proof, at a strategy whose finders meet `FinderCR` (elsewhere the error is `⊤`), where the files pass the commit
phase's check (elsewhere no play is accepted) and the claim fails (elsewhere the event is empty), is `ex_verdict_le`
through the coupling's `prob_le`, at `acc_of_accepts`' property. -/
theorem spotRecursive (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (hcap : ∀ x, q.RunAccepts x → q.OneTable x →
      (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 ∧ nI x ≤ 2 ^ 32 ∧ vS x ≤ nS x * 138543104)
    (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Trace)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl K₀ : ℕ) (δs : ℝ)
    (hK₀ : ∀ m : ℕ, (V.certOf θ).ε * V.units θ F < m → K₀ ≤ m)
    (hdraw : ∀ U : Finset (Fin (V.units θ F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs)
    (hX : V.extract θ F c σ = q.extract V S readX)
    (hread : ∀ ξ wt, V.Registered θ F → q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧ ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂ = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) :
    Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      q.errAt V S collOf dColl nI vS δs lam := by
  classical
  unfold Dep.errAt
  split_ifs with hcr
  swap
  · exact le_top
  -- where the claim holds, the event is empty
  by_cases hH : V.Holds k θ F c σ .sampled ⟨θ, F, c, .empty, [], .empty, [], [], .empty, #[]⟩
  · refine le_trans (Game.prob_mono (q := fun _ => False) (fun o ho => ho.2 hH) _ σ) ?_
    rw [Game.prob_false]
    exact zero_le
  -- where the files fail the commit phase's check, no play is accepted
  by_cases hreg : V.Registered θ F
  swap
  · rw [show Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ = 0 from
      prob_play_eq_zero (V.families θ) (D := V.signer θ) θ F c _ (fun _ h3 h => hreg (h3.1 ▸ h3.2.1 ▸ h.1.1)) σ]
    exact zero_le
  -- more than an `ε` share of the units is unbound in the extracted transcript
  have hU : K₀ ≤ (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (q.extract V S readX)).card := by
    refine hK₀ _ ?_
    have hH' : ¬ ((Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ)).card : ℝ) ≤
        (V.certOf θ).ε * V.units θ F := hH
    rw [hX] at hH'
    exact not_le.mp hH'
  calc Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ
      ≤ ENNReal.ofReal (Ex fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) =>
          (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (S.out ξ s)) := S.prob_le _
    _ ≤ ENNReal.ofReal (Ex fun ξ => Pr fun s : S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) =>
          RecVerdict V k (S.out ξ s)) :=
        ENNReal.ofReal_le_ofReal (Ex_mono fun ξ => Pr_mono fun s hs => hs.1)
    _ ≤ q.delta V S dColl δs lam := by
        unfold Dep.delta
        refine ENNReal.ofReal_le_ofReal ?_
        exact ex_verdict_le q (accOfAccepts q) hKS hdecI hdecO nI vS lam hn hv hlam hcap V k S readX collOf dColl hcr
          δs (hdraw _ hU) (fun ξ wt hR u hu => hread ξ wt hreg hR u hu) hcoll hlive

end Verity.Recursive
