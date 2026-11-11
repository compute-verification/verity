module

public import Proofs.Flock.SamplingGeneric.RegAt
public import Proofs.Flock.SamplingGeneric.TwoKSAt
public import Proofs.Flock.SamplingGeneric.TwoRegMiss

/-!
# The two-stage committed form read at a time budget

`B_conf,2` declared at a budget `T` a run (`CompiledTwo.BconfTwoAt`) has `B_conf,2`'s coins and outcome, so its
probability (`BconfTwoAt_pr`). Its two runs share the coins off the second stage, and each takes fresh second-stage
coins and its own extractor's, so each is one run on uniform coins (`Ex_forkOn`, `Ex_two_forks`), and its expected
cost is `B_conf`'s bound at `T`, `c_T = 2T·(1 + n_sess·e_s) + 2·n_PU·openCalls` (`BconfTwoAt_cost_le`, at the two-stage
contract's cost facts, `CompiledTwo.Sessions.toCostContract`). So where `ExpectedCR` holds at `B_conf` and `B_conf,2`
declared at `T`, both probabilities are at most `c_T/2^256`, and `KSRegTwo`'s two bounds hold at `ε_reg,2` read at `c_T`
(`CompiledTwo.ksRegTwoAt_of_ksRegTwo`): `CompilerKSRegTwo` gives `CompilerKSRegTwoAt` (`compilerKSRegTwoAt_of`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling

/-- Over two runs' coins sharing a first coordinate, the first run's term has the run's own expectation. -/
theorem Ex_pair_fst {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (f : α × β → ℝ) :
    Ex (fun p : α × β × β => f (p.1, p.2.1)) = Ex f :=
  ((Ex_prod fun p : α × β × β => f (p.1, p.2.1)).trans (congrArg Ex (funext fun r =>
    (Ex_prod fun y : β × β => f (r, y.1)).trans (congrArg Ex (funext fun a => Ex_const (α := β) (f (r, a))))))).trans
    (Ex_prod f).symm

/-- Over two runs' coins sharing a first coordinate, the second run's term has the run's own expectation. -/
theorem Ex_pair_snd {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (f : α × β → ℝ) :
    Ex (fun p : α × β × β => f (p.1, p.2.2)) = Ex f :=
  ((Ex_prod fun p : α × β × β => f (p.1, p.2.2)).trans (congrArg Ex (funext fun r =>
    (Ex_prod fun y : β × β => f (r, y.2)).trans (Ex_const (α := β) (Ex fun b : β => f (r, b)))))).trans
    (Ex_prod f).symm

/-- Two runs at one tape sharing their coins off the rounds `I`, each with fresh coins on `I` and its own extractor's
coins: a cost `c` plus each run's, whose expectation over its coins is at most `b` at every tape, is at most `c + 2b`
in expectation, each run's coins being uniform (`Ex_forkOn`). -/
theorem Ex_two_forks {Coin Ω C : Type} [Fintype Coin] [Nonempty Coin] [Fintype Ω] [Nonempty Ω] [Fintype C]
    [Nonempty C] {k : ℕ} (I : Finset (Fin k)) (G : Ω → (Fin k → Coin) × C → ℝ) (c b : ℝ) (hG : ∀ ω, Ex (G ω) ≤ b) :
    Ex (fun q : Ω × (Fin k → Coin) × ((Fin k → Coin) × C) × ((Fin k → Coin) × C) =>
      c + G q.1 (forkOn I q.2.1 q.2.2.1.1, q.2.2.1.2) + G q.1 (forkOn I q.2.1 q.2.2.2.1, q.2.2.2.2)) ≤ c + 2 * b := by
  have hrun : ∀ ω, Ex (fun y : (Fin k → Coin) × ((Fin k → Coin) × C) => G ω (forkOn I y.1 y.2.1, y.2.2)) ≤ b :=
    fun ω => by
      calc Ex (fun y : (Fin k → Coin) × ((Fin k → Coin) × C) => G ω (forkOn I y.1 y.2.1, y.2.2))
          = Ex (fun r : Fin k → Coin => Ex fun y : (Fin k → Coin) × C => G ω (forkOn I r y.1, y.2)) :=
            Ex_prod fun y : (Fin k → Coin) × ((Fin k → Coin) × C) => G ω (forkOn I y.1 y.2.1, y.2.2)
        _ = Ex (fun r : Fin k → Coin => Ex fun e : Fin k → Coin => Ex fun c' : C => G ω (forkOn I r e, c')) :=
            congrArg Ex (funext fun r => Ex_prod fun y : (Fin k → Coin) × C => G ω (forkOn I r y.1, y.2))
        _ = Ex (fun y : (Fin k → Coin) × (Fin k → Coin) => Ex fun c' : C => G ω (forkOn I y.1 y.2, c')) :=
            (Ex_prod fun y : (Fin k → Coin) × (Fin k → Coin) => Ex fun c' : C => G ω (forkOn I y.1 y.2, c')).symm
        _ = Ex (fun r : Fin k → Coin => Ex fun c' : C => G ω (r, c')) :=
            Ex_forkOn I fun r : Fin k → Coin => Ex fun c' : C => G ω (r, c')
        _ = Ex (G ω) := (Ex_prod (G ω)).symm
        _ ≤ b := hG ω
  have h₁ : Ex (fun q : Ω × (Fin k → Coin) × ((Fin k → Coin) × C) × ((Fin k → Coin) × C) =>
      G q.1 (forkOn I q.2.1 q.2.2.1.1, q.2.2.1.2)) ≤ b :=
    ((Ex_prod fun q : Ω × (Fin k → Coin) × ((Fin k → Coin) × C) × ((Fin k → Coin) × C) =>
      G q.1 (forkOn I q.2.1 q.2.2.1.1, q.2.2.1.2)).trans (congrArg Ex (funext fun ω =>
        Ex_pair_fst fun y : (Fin k → Coin) × ((Fin k → Coin) × C) => G ω (forkOn I y.1 y.2.1, y.2.2)))).trans_le
      ((Ex_mono hrun).trans_eq (Ex_const b))
  have h₂ : Ex (fun q : Ω × (Fin k → Coin) × ((Fin k → Coin) × C) × ((Fin k → Coin) × C) =>
      G q.1 (forkOn I q.2.1 q.2.2.2.1, q.2.2.2.2)) ≤ b :=
    ((Ex_prod fun q : Ω × (Fin k → Coin) × ((Fin k → Coin) × C) × ((Fin k → Coin) × C) =>
      G q.1 (forkOn I q.2.1 q.2.2.2.1, q.2.2.2.2)).trans (congrArg Ex (funext fun ω =>
        Ex_pair_snd fun y : (Fin k → Coin) × ((Fin k → Coin) × C) => G ω (forkOn I y.1 y.2.1, y.2.2)))).trans_le
      ((Ex_mono hrun).trans_eq (Ex_const b))
  rw [Ex_add, Ex_add, Ex_const]
  linarith

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`B_conf,2` declared at a budget has `B_conf,2`'s probability**: the same coins and outcome. -/
theorem BconfTwoAt_pr (H : Msg → Msg) (L : CompiledTwo Coin A) (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) : (BconfTwoAt H L T Ω P x).Pr H = (BconfTwo H L Ω P x).Pr H := by
  simp only [BconfTwoAt, BconfTwo]
  rcases L.parse x with _ | s <;> rfl

/-- **`B_conf,2` declared at a budget, its expected cost**: at most `c_T` (`Compiled.bconfCost`) at the contract's cost
facts and the opening checks' count: `T` for each of its two runs, and each run's extraction's cost at `T`
(`Compiled.runCostAt_exp`), over the shared coins and its own (`Ex_two_forks`). -/
theorem BconfTwoAt_cost_le {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.toCompiled.CostContract H)
    (hopen : L.OpenCallsLe H) (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    Ex (fun q => ((BconfTwoAt H L T Ω P x).cost q : ℝ)) ≤ L.bconfCost x T := by
  delta BconfTwoAt
  rcases hs : L.parse x with _ | s
  · show Ex (fun _ : Unit => ((0 : ℕ) : ℝ)) ≤ _
    rw [Nat.cast_zero, Ex_const]
    exact L.bconfCost_nonneg x T
  · refine (Ex_mono (g := fun q : Ω × (Fin (L.V.k x) → Coin) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) ×
        ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (2 * T : ℝ) + L.runCostAt H T x s P q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.1.1, q.2.2.1.2) +
          L.runCostAt H T x s P q.1 (forkOn (L.stage2 x) q.2.1 q.2.2.2.1, q.2.2.2.2)) fun q => le_of_eq ?_).trans
      ((Ex_two_forks (L.stage2 x) (fun ω rc => L.runCostAt H T x s P ω rc) _ _ fun ω =>
        Compiled.runCostAt_exp hS hopen T hs P ω).trans_eq ?_)
    · simp only [Compiled.runCostAt]
      push_cast
      ring
    · unfold Compiled.bconfCost
      ring

/-- **`KSRegTwo` read at a budget**: where `ExpectedCR` holds at `B_conf` and `B_conf,2` declared at `T`, both
probabilities are at most `c_T/2^256` (`Compiled.BconfAt_pr`, `BconfTwoAt_pr`, `Compiled.BconfAt_cost_le`,
`BconfTwoAt_cost_le`), so `ε_reg,2` is at most its reading at `T` and `KSRegTwo`'s two bounds hold there. -/
theorem ksRegTwoAt_of_ksRegTwo {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.toCompiled.CostContract H)
    (hopen : L.OpenCallsLe H) (h : L.KSRegTwo H) : L.KSRegTwoAt H := by
  intro x hx Ω _ _ P T hE hE₂
  unfold ExpectedCR at hE hE₂
  have hpr : (Compiled.Bconf H L.toCompiled Ω P x).Pr H ≤ L.bconfCost x T / 2 ^ (256 : ℝ) := by
    rw [← Compiled.BconfAt_pr H L.toCompiled T Ω P x]
    exact hE.trans (div_le_div_of_nonneg_right (Compiled.BconfAt_cost_le hS hopen T Ω P x) (by positivity))
  have hpr₂ : (BconfTwo H L Ω P x).Pr H ≤ L.bconfCost x T / 2 ^ (256 : ℝ) := by
    rw [← BconfTwoAt_pr H L T Ω P x]
    exact hE₂.trans (div_le_div_of_nonneg_right (BconfTwoAt_cost_le hS hopen T Ω P x) (by positivity))
  have hε : L.εregTwo H x P ≤ L.εregTwoAt H x P T := by
    unfold CompiledTwo.εregTwo CompiledTwo.εregTwoAt Compiled.εreg Compiled.εregAt
    have h₁ := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hpr) (Nat.cast_nonneg (α := ℝ) (L.nG x))
    have h₂ := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hpr₂) (Nat.cast_nonneg (α := ℝ) (L.nG x))
    linarith
  obtain ⟨h₁, h₂⟩ := h x hx Ω P
  exact ⟨h₁.trans hε, fun s Pt d hs hPt hd δ hδ => (h₂ s Pt d hs hPt hd δ hδ).trans (by linarith)⟩

end CompiledTwo

/-- **`CompilerKSRegTwoAt` from `CompilerKSRegTwo`**: both finders' expected costs at each budget are bounded at the
two-stage contract's cost facts and the opening checks' count (`CompiledTwo.ksRegTwoAt_of_ksRegTwo`). -/
theorem compilerKSRegTwoAt_of (h : CompilerKSRegTwo) : CompilerKSRegTwoAt := by
  intro Coin _ _ _ A L H hA hB hS hopen
  exact CompiledTwo.ksRegTwoAt_of_ksRegTwo hS.toCostContract hopen (h Coin A L H hA hB hS)

/-- **`CompilerKSRegTwoAt` is proved**: `compilerKSRegTwoAt_of` at `compilerKSRegTwo`. -/
theorem compilerKSRegTwoAt : CompilerKSRegTwoAt := compilerKSRegTwoAt_of compilerKSRegTwo

end Claims.SamplingGeneric
