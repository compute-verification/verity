module

public import Definitions.Claims.SamplingTwoRegX
public import Definitions.NetworkCertifier.PooledReg
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Sampling.EndToEndDraw
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Claims.Sampling.Wait
public import Proofs.Flock.SamplingGeneric.Fails
public import Proofs.Flock.SamplingGeneric.Reg
public import Proofs.Flock.SamplingGeneric.ZK
public import Proofs.NetworkCertifier.Pooled

/-!
# The pooled bound at the trace with the registered openings: the sample's term

What the sample gives at one execution phase, and the form the pooled bound takes it in over a monitoring period.

* The rate. `rateAt_subset`: a uniform sample of `kd` of the `n` proof blocks catches at every rate `q` with
  `q · n ≤ kd`. `rateAt_all`: a sample of every proof block catches at rate `1`.
* One execution phase. `pr_acc_of_event`: at a statement in scope and any set `F` of its proof blocks, a run that
  accepts with a sampled block of `F` with probability at most `c` accepts with probability at most
  `(1 - q)^{|F|} + c`. It takes no knowledge soundness and no binding. `pr_acc_regX_le` and `pr_acc_regTwoX_le` are
  that bound at the trace read at the statement, at one stage and at two.
* The monitoring period. `phasesMiss_of_scope` turns a bound at each statement in scope into the bound on each
  phase's verdict that `NetTiming.pr_pooled_ex_le` takes, the verifier refusing what is out of scope and reading the
  first of the phase's coins (`pr_castLE`); the bound's error `E z ω i` may read the secret and the period's coins, as
  an error with a charge does. `phasesMiss_of_event` is it at the trace with the registered openings from any bound on
  each phase's event, and `phasesMiss_regX` at the trace read at the statement. `phaseAcc_prefix` and
  `wreg_prefix`: a phase's verdict and its incorrect set read only the earlier phases' coins where its statement, its
  prover's strategy and its registered openings do.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled CompiledTwo)
open Claims.Sampling.EndToEndDraw (SubsetDesign blocks_length Pa_subset)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The design's rate -/

/-- **A uniform sample of `kd` of the `n` proof blocks catches at every rate `q` with `q · n ≤ kd`**, at a design of
one class sampled by a uniform `kd`-subset of its proof blocks. -/
theorem rateAt_subset {s : AuditStatement} {P : Partition} {c : List (List (List ℕ))} {kd : ℕ}
    (hd : SubsetDesign s.design kd) (hP : P.classes = [c]) (hkd : kd ≤ P.blocks.length) {q : ℝ}
    (hq : q * P.blocks.length ≤ kd) : RateAt s P q := by
  intro B hB
  have hC : (0 : ℝ) < P.blocks.length.choose kd := by exact_mod_cast Nat.choose_pos hkd
  have hw : window 0 P.blocks.length B = B := window_zero hB
  rw [Pa_subset hd hP, hw, div_le_iff₀ hC]
  refine (choose_sub_le P.blocks.length kd hkd B.card).trans ?_
  rw [mul_comm]
  refine mul_le_mul_of_nonneg_right ?_ hC.le
  rcases Nat.eq_zero_or_pos P.blocks.length with hn | hn
  · have hB0 : B.card = 0 := by
      have h := Finset.card_le_card hB
      rw [Finset.card_range] at h
      omega
    rw [hB0, pow_zero, pow_zero]
  · have hn' : (0 : ℝ) < P.blocks.length := by exact_mod_cast hn
    have hkn : (kd : ℝ) / P.blocks.length ≤ 1 := (div_le_one hn').2 (by exact_mod_cast hkd)
    have hqk : q ≤ kd / P.blocks.length := (le_div_iff₀ hn').2 hq
    exact pow_le_pow_left₀ (by linarith) (by linarith) _

/-- **A sample of every proof block catches at rate `1`**, at a design of one class sampled whole: it holds a block of
every set that is not empty. -/
theorem rateAt_all {s : AuditStatement} {P : Partition} {c : List (List (List ℕ))} (hd : s.design.rules = [.all])
    (hP : P.classes = [c]) : RateAt s P 1 := by
  intro B hB
  have hw : window 0 c.flatten.length B = B := window_zero (by rw [← blocks_length hP]; exact hB)
  have hPa : s.design.Pa P B = (OneStage.all).Pa (strataSizes c) (window 0 c.flatten.length B) * 1 := by
    unfold Design.Pa
    rw [hd, hP]
    rfl
  rw [hPa, mul_one, hw, sub_self]
  by_cases h0 : B = ∅
  · subst h0
    simp [OneStage.Pa]
  · have hc : B.card ≠ 0 := fun h => h0 (Finset.card_eq_zero.1 h)
    simp [OneStage.Pa, h0, zero_pow hc]

/-! ## One execution phase -/

/-- **Acceptance from any bound on the event at a set of proof blocks**, at one execution phase's statement in scope:
where the run accepts with a sampled proof block of `F` with probability at most `c`, it accepts with probability at
most `(1 - q)^{|F|} + c`. It takes no knowledge soundness and no binding. -/
theorem pr_acc_of_event {H : Msg → Msg} {D : Compiled Coin A} (hS : D.Sessions H) {x : Msg} (hsc : D.Scope x)
    {q : ℝ} (hq1 : q ≤ 1) (hq : ∀ s Pt, D.parse x = some s → D.partition s = some Pt → RateAt s Pt q)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (F : Finset ℕ) (hF : ∀ v ∈ F, v < D.nPU x) {c : ℝ}
    (h1 : Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ D.checked x p.2, u ∈ F) ≤ c) :
    Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc) ≤ (1 - q) ^ F.card + c := by
  obtain ⟨s, Pt, hs, hPt, hin⟩ := hsc
  have hpart : D.part s = Pt := by simp only [Compiled.part, hPt, Option.getD_some]
  have hsub : F ⊆ range Pt.blocks.length := fun v hv =>
    mem_range.2 (by rw [← Compiled.nPU_eq hs hPt]; exact hF v hv)
  have hpow0 : 0 ≤ (1 - q) ^ F.card := pow_nonneg (by linarith) _
  have hPa : s.design.Pa (D.part s) F ≤ (1 - q) ^ F.card := by
    rw [hpart]
    exact hq s Pt hs hPt _ hsub
  have hmiss : Pr (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F) ≤ (1 - q) ^ F.card := by
    have h := Compiled.pr_miss_le hS hs hPt hin F hF (δ := max 0 (s.design.Pa (D.part s) F)) (le_max_left _ _)
    refine (Claims.Pr_mono fun r hr => ⟨fun hM => ?_, hr⟩).trans (h.trans (max_le hpow0 hPa))
    simp only [Design.M, Set.mem_ofPred_eq] at hM
    exact absurd hM (not_lt.2 (le_max_right _ _))
  have hsnd : Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => ∀ u ∈ D.checked x p.2, u ∉ F) ≤ (1 - q) ^ F.card :=
    (Claims.SamplingGeneric.Pr_snd (α := Ω)
      (fun r : Fin (D.V.k x) → Coin => ∀ u ∈ D.checked x r, u ∉ F)).trans_le hmiss
  have h2 := Claims.Sampling.Pr_or_le
    (p := fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc)
    (fun p hp => by
      by_cases h : ∃ u ∈ D.checked x p.2, u ∈ F
      · exact Or.inl ⟨hp, h⟩
      · exact Or.inr fun u hu hu' => h ⟨u, hu, hu'⟩) h1 hsnd
  linarith

/-- **Acceptance at the trace read at the statement**: a run accepts with probability at most
`(1 − q)^{|FregX|} + ε_reg,X`. -/
theorem pr_acc_regX_le {H : Msg → Msg} {D : Compiled Coin A} (hS : D.Sessions H) (hK : D.KSRegX H) {x : Msg}
    (hx : D.Scope x) {q : ℝ} (hq1 : q ≤ 1)
    (hq : ∀ s Pt, D.parse x = some s → D.partition s = some Pt → RateAt s Pt q)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc) ≤
      (1 - q) ^ (D.FregX H P x).card + D.εregX H x P := by
  have ⟨s, _, hs, _, _⟩ := hx
  refine pr_acc_of_event hS hx hq1 hq Ω P (D.FregX H P x) (fun v hv => ?_) (hK x hx Ω P).1
  simp only [Compiled.FregX, hs, mem_filter, mem_range] at hv
  simp only [Compiled.nPU, hs]
  exact hv.1

/-- **Acceptance at the boundary trace read at the statement, two stages**: a run accepts with probability at most
`(1 − r₁·r₂)^{|W|} + ε_reg,2,X`, `W` the replay blocks incorrect at that trace. -/
theorem pr_acc_regTwoX_le {H : Msg → Msg} {D : CompiledTwo Coin A} (hK : D.KSRegTwoX H) {r₁ r₂ : ℝ}
    (hM : D.MissTwoX H r₁ r₂) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : D.Scope x)
    (hs : D.parse x = some s) (hPt : D.partition s = some Pt) (hd : D.two s = some d) (h1 : d.FirstRate r₁)
    (h2 : d.SecondRate Pt r₂) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc) ≤
      (1 - r₁ * r₂) ^ (wrongAtBoundaryAt s Pt d.part (D.regValX H P x)).card + D.εregTwoX H x P := by
  have hev := (hK x hx Ω P).1
  have hmiss : Pr (fun p : Ω × (Fin (D.V.k x) → Coin) =>
      ∀ u ∈ D.checked x p.2, u ∉ D.FregTwoX H P x p.1 p.2) ≤
      (1 - r₁ * r₂) ^ (wrongAtBoundaryAt s Pt d.part (D.regValX H P x)).card := by
    rw [Claims.Sampling.Pr_prod_eq_Ex]
    exact (Claims.Ex_mono fun ω => hM x s Pt d hx hs hPt hd h1 h2 Ω P ω).trans_eq (Claims.Ex_const _)
  have h3 := Claims.Sampling.Pr_or_le
    (p := fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc)
    (fun p hp => by
      by_cases h : ∃ u ∈ D.checked x p.2, u ∈ D.FregTwoX H P x p.1 p.2
      · exact Or.inl ⟨hp, h⟩
      · exact Or.inr fun u hu hu' => h ⟨u, hu, hu'⟩) hev hmiss
  linarith

/-- `FregR` holds only the statement's proof blocks. -/
theorem fregR_lt (H : Msg → Msg) (D : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (ops : AuditStatement → List (Opening Bool)) : ∀ v ∈ FregR H D P x ops, v < D.nPU x := by
  intro v hv
  cases hs : D.parse x with
  | none => simp [FregR, hs] at hv
  | some s =>
    simp only [FregR, hs, mem_filter, mem_range] at hv
    simp only [Compiled.nPU, hs]
    exact hv.1

/-! ## The monitoring period -/

section Phases

variable {m kV : ℕ} {Tp Sec : Type}

/-- A phase's incorrect set reads only the earlier phases' coins where its statement, its prover's strategy and its
registered openings do. -/
theorem wreg_prefix [Fintype Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A)
    {x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg}
    {Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp}
    {ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)}
    (had : PhasesAdaptive x Pv ops) :
    ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → WregR H D x Pv ops z ω i = WregR H D x Pv ops z ω' i := by
  intro z i ω ω' h
  unfold WregR
  rw [had.1 z i ω ω' h, had.2.1 z i ω ω' h, had.2.2 z i ω ω' h]

omit [Fintype Coin] [Nonempty Coin] in
/-- A phase's verdict reads only the earlier phases' coins where its statement and its prover's strategy do. -/
theorem phaseAcc_prefix (H : Msg → Msg) (D : Msg → Compiled Coin A)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (hx : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → x z ω i = x z ω' i)
    (hP : ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → Pv z ω i = Pv z ω' i) :
    ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → phaseAcc H D x Pv hk z i ω = phaseAcc H D x Pv hk z i ω' := by
  intro z i ω ω' h
  have key : ∀ (x₁ x₂ : Msg) (P₁ P₂ : Prover Tp) (h₁ : (D x₁).V.k x₁ ≤ kV) (h₂ : (D x₂).V.k x₂ ≤ kV), x₁ = x₂ →
      P₁ = P₂ →
      (fun c : Tp × (Fin kV → Coin) => (run H P₁ (D x₁).V x₁ c.1 (fun j => c.2 (Fin.castLE h₁ j))).out = .acc) =
        fun c : Tp × (Fin kV → Coin) => (run H P₂ (D x₂).V x₂ c.1 (fun j => c.2 (Fin.castLE h₂ j))).out = .acc := by
    intro x₁ x₂ P₁ P₂ h₁ h₂ hxx hPP
    subst hxx
    subst hPP
    rfl
  exact key _ _ _ _ _ _ (hx z i ω ω' h) (hP z i ω ω' h)

omit [Inhabited Coin] in
/-- The verifier's coins at the statement's own number of rounds, as the first of `kV` coins: the coins past them
are uniform and read by nothing. -/
theorem pr_castLE [Fintype Tp] {n : ℕ} (h : n ≤ kV) (f : Tp → (Fin n → Coin) → Prop) :
    Pr (fun c : Tp × (Fin kV → Coin) => f c.1 (fun j => c.2 (Fin.castLE h j))) =
      Pr (fun p : Tp × (Fin n → Coin) => f p.1 p.2) := by
  rw [Claims.Sampling.Pr_prod_eq_Ex, Claims.Sampling.Pr_prod_eq_Ex]
  exact congrArg Ex (funext fun t => Claims.Sampling.Pr_comp_inj (Fin.castLE h) (Fin.castLE_injective h) (f t))

/-- **The bound on each execution phase's verdict from a bound at each statement in scope**: the verifier refuses
what is out of scope, so the bound is needed only in scope. The error `E z ω i` is any that is not negative; it may
read the secret and the period's coins. Its conclusion is the form `NetTiming.pr_pooled_ex_le` takes each phase's
acceptance in. -/
theorem phasesMiss_of_scope [Fintype Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A} (hS : ∀ y, (D y).Sessions H)
    (Zs : Set Sec) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (W : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Finset ℕ) {q : ℝ} (hq1 : q ≤ 1)
    (E : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → ℝ) (hE0 : ∀ z ω i, 0 ≤ E z ω i)
    (hphase : ∀ z ∈ Zs, ∀ i ω, (D (x z ω i)).Scope (x z ω i) →
      Pr (fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) p.1 p.2).out = .acc) ≤ (1 - q) ^ (W z ω i).card + E z ω i) :
    ∀ z ∈ Zs, ∀ i ω, Pr (phaseAcc H D x Pv hk z i ω) ≤ (1 - q) ^ (W z ω i).card + E z ω i := by
  intro z hz i ω
  by_cases hsc : (D (x z ω i)).Scope (x z ω i)
  · have hc : Pr (phaseAcc H D x Pv hk z i ω) =
        Pr (fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
          (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) p.1 p.2).out = .acc) :=
      pr_castLE (hk z ω i) (fun (t : Tp) (r : Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) t r).out = .acc)
    rw [hc]
    exact hphase z hz i ω hsc
  · have h0 : Pr (phaseAcc H D x Pv hk z i ω) ≤ 0 := by
      refine (Claims.Pr_mono (q := fun _ => False) fun c hc => hsc ?_).trans_eq Claims.Pr_false
      have hc' : (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) c.1
          (fun j => c.2 (Fin.castLE (hk z ω i) j))).out = .acc := hc
      exact (hS (x z ω i)).refuses _ _ _ _ _ hc'
    have h1 := pow_nonneg (sub_nonneg.2 hq1) (W z ω i).card
    have h2 := hE0 z ω i
    linarith

/-- **The bound on each execution phase's verdict at the trace read at the statement**: at `KSRegX` and the sessions'
contract, over a monitoring period whose statements and provers the secret and the period's coins fix, where every
phase's design catches at rate `q` and `e i` bounds phase `i`'s `ε_reg,X`. -/
theorem phasesMiss_regX [Fintype Tp] [Nonempty Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A}
    (hS : ∀ y, (D y).Sessions H) (hK : ∀ y, (D y).KSRegX H) (Zs : Set Sec)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x) {q : ℝ}
    (hq1 : q ≤ 1)
    (hrate : ∀ z ∈ Zs, ∀ ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
    {e : Fin m → ℝ} (he0 : ∀ i, 0 ≤ e i)
    (he : ∀ z ∈ Zs, ∀ ω i, (D (x z ω i)).εregX H (x z ω i) (Pv z ω i) ≤ e i) :
    ∀ z ∈ Zs, ∀ i ω, Pr (phaseAcc H D x Pv hk z i ω) ≤
      (1 - q) ^ ((D (x z ω i)).FregX H (Pv z ω i) (x z ω i)).card + e i :=
  phasesMiss_of_scope hS Zs x Pv hk (fun z ω i => (D (x z ω i)).FregX H (Pv z ω i) (x z ω i)) hq1 (fun _ _ i => e i)
    (fun _ _ i => he0 i) fun z hz i ω hsc =>
      (pr_acc_regX_le (hS (x z ω i)) (hK (x z ω i)) hsc hq1 (fun s Pt hs hPt => hrate z hz ω i s Pt hs hPt) Tp
        (Pv z ω i)).trans (add_le_add le_rfl (he z hz ω i))

/-- **The bound on each execution phase's verdict at the trace with the registered openings, from any bound on each
phase's event**: `E z ω i`, not negative, bounds the probability that phase `i`'s run accepts with a sampled proof
block of its `FregR`, at every statement in scope the period plays it with. -/
theorem phasesMiss_of_event [Fintype Tp] [Nonempty Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A}
    (hS : ∀ y, (D y).Sessions H) (Zs : Set Sec) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    {q : ℝ} (hq1 : q ≤ 1)
    (hrate : ∀ z ∈ Zs, ∀ ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
    (E : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → ℝ) (hE0 : ∀ z ω i, 0 ≤ E z ω i)
    (hev : ∀ z ∈ Zs, ∀ ω i, (D (x z ω i)).Scope (x z ω i) →
      Pr (fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) p.1 p.2).out = .acc ∧
          ∃ u ∈ (D (x z ω i)).checked (x z ω i) p.2,
            u ∈ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) ≤ E z ω i) :
    ∀ z ∈ Zs, ∀ i ω, Pr (phaseAcc H D x Pv hk z i ω) ≤ (1 - q) ^ (WregR H D x Pv ops z ω i).card + E z ω i :=
  phasesMiss_of_scope hS Zs x Pv hk (WregR H D x Pv ops) hq1 E hE0 fun z hz i ω hsc =>
    pr_acc_of_event (hS (x z ω i)) hsc hq1 (fun s Pt hs hPt => hrate z hz ω i s Pt hs hPt) Tp (Pv z ω i)
      (FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i))
      (fregR_lt H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) (hev z hz ω i hsc)

end Phases

end NetTiming
