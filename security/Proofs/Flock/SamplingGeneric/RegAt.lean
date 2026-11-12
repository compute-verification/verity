module

public import Proofs.Flock.SamplingGeneric.Efficient
public import Proofs.Flock.SamplingGeneric.Reg

/-!
# The committed form read at a time budget

`B_conf` declared at a budget `T` a run (`Compiled.BconfAt`) has `B_conf`'s coins and outcome, so its probability
(`BconfAt_pr`). At the contract's cost facts and the opening checks' count its expected cost is at most
`c_T = 2T·(1 + n_sess·e_s) + 2·n_PU·openCalls` (`BconfAt_cost_le`): `T` for each of its two runs, and each run's
extraction's own calls, at most `n_PU·openCalls` (`runExt_own_le`), and `T` for each of its reruns, at most
`n_sess·e_s` expected over the run's coins and the extractor's (`runExt_runs_exp`). On every coin it is at most
`2T·(1 + n_sess·ρ_s) + 2·n_PU·openCalls` (`BconfAt_cost_strict`, `runExt_runs_le`). So where `ExpectedCR` holds at
`B_conf` declared at `T`, `Pr[B_conf] ≤ c_T/2^256`, and `KSReg`'s two bounds hold at `ε_reg` read at `c_T`
(`Compiled.ksRegAt_of_ksReg`): `CompilerKSReg` gives `CompilerKSRegAt` (`compilerKSRegAt_of`). Where `StrictBirthday`
holds there instead, `Pr[B_conf] ≤ q_T²/2^513` at that cost on every coin, `q_T`, and `KSReg`'s two bounds hold at
`ε_reg` read at `q_T` (`Compiled.ksRegAtSB_of_ksReg`, `compilerKSRegAtSB_of`).
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Claims.Sampling

/-- Two runs at one tape, each on its own coins: a cost `c` plus each run's, whose expectation over its coins is at most
`b` at every tape, is at most `c + 2b` in expectation. -/
theorem Ex_two_runs {Ω X : Type} [Fintype Ω] [Nonempty Ω] [Fintype X] [Nonempty X] (G : Ω → X → ℝ) (c b : ℝ)
    (hG : ∀ ω, Ex (G ω) ≤ b) : Ex (fun q : Ω × X × X => c + G q.1 q.2.1 + G q.1 q.2.2) ≤ c + 2 * b := by
  have hfst : ∀ ω, Ex (fun y : X × X => G ω y.1) = Ex (G ω) := fun ω =>
    (Ex_prod fun y : X × X => G ω y.1).trans (congrArg Ex (funext fun a => Ex_const (α := X) (G ω a)))
  have hsnd : ∀ ω, Ex (fun y : X × X => G ω y.2) = Ex (G ω) := fun ω =>
    (Ex_prod fun y : X × X => G ω y.2).trans (Ex_const (α := X) (Ex (G ω)))
  have hb : Ex (fun ω => Ex (G ω)) ≤ b := (Ex_mono hG).trans_eq (Ex_const b)
  have h1 : Ex (fun q : Ω × X × X => G q.1 q.2.1) ≤ b :=
    ((Ex_prod fun q : Ω × X × X => G q.1 q.2.1).trans (congrArg Ex (funext hfst))).trans_le hb
  have h2 : Ex (fun q : Ω × X × X => G q.1 q.2.2) ≤ b :=
    ((Ex_prod fun q : Ω × X × X => G q.1 q.2.2).trans (congrArg Ex (funext hsnd))).trans_le hb
  rw [Ex_add, Ex_add, Ex_const]
  linarith

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`B_conf` declared at a budget has `B_conf`'s probability**: the same coins and outcome. -/
theorem BconfAt_pr (H : Msg → Msg) (L : Compiled Coin A) (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) : (BconfAt H L T Ω P x).Pr H = (Bconf H L Ω P x).Pr H := by
  simp only [BconfAt, Bconf]
  -- The finders' coins are the match's, so the parse is generalized everywhere at once.
  rcases L.parse x with _ | s <;> rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem bconfCost_nonneg (L : Compiled Coin A) (x : Msg) (T : ℕ) : 0 ≤ L.bconfCost x T := by
  have he0 : 0 ≤ L.es x := maxBelow_nonneg _ _
  unfold bconfCost
  exact add_nonneg (mul_nonneg (mul_nonneg zero_le_two (Nat.cast_nonneg _))
    (add_nonneg zero_le_one (mul_nonneg (Nat.cast_nonneg _) he0)))
    (mul_nonneg zero_le_two (mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)))

/-- One run's extraction's cost at the budget `T`: its own SHA-512 calls, and `T` for each rerun of the prover. -/
noncomputable def runCostAt (H : Msg → Msg) (L : Compiled Coin A) (T : ℕ) (x : Msg) (s : AuditStatement) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (rc : (Fin (L.V.k x) → Coin) × L.SlotEC x) : ℝ :=
  (ownLog H (run H P L.V x ω) (L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2)).length +
    T * ((L.runExt x s rc.1 (run H P L.V x ω rc.1).seen rc.2).eval H (run H P L.V x ω)).2.2

/-- **One run's extraction's expected cost at the budget `T`**, over the run's coins and the extractor's: at most
`n_PU·openCalls + T·n_sess·e_s` (`runExt_own_le`, `runExt_runs_exp`). -/
theorem runCostAt_exp {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (T : ℕ) {x : Msg} {s : AuditStatement} (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω) (ω : Ω) :
    Ex (fun rc => L.runCostAt H T x s P ω rc) ≤ L.nPU x * L.openCalls x + T * (L.nSess x * L.es x) := by
  unfold runCostAt
  rw [Ex_add, Ex_mul_left]
  refine add_le_add ((Ex_mono (g := fun _ => ((L.nPU x * L.openCalls x : ℕ) : ℝ)) fun rc => ?_).trans_eq ?_)
    (mul_le_mul_of_nonneg_left (runExt_runs_exp hS P ω s) (Nat.cast_nonneg _))
  · exact Nat.cast_le.2 (runExt_own_le hS hopen hs P ω rc.1 rc.2)
  · rw [Ex_const, Nat.cast_mul]

/-- **`B_conf` declared at a budget, its expected cost**: at most `c_T` (`bconfCost`) at the contract's cost facts and
the opening checks' count: `T` for each of its two runs, and each run's extraction's cost at `T` (`runCostAt_exp`),
over its own coins (`Ex_two_runs`). -/
theorem BconfAt_cost_le {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    Ex (fun q => ((BconfAt H L T Ω P x).cost q : ℝ)) ≤ L.bconfCost x T := by
  delta BconfAt
  rcases hs : L.parse x with _ | s
  · show Ex (fun _ : Unit => ((0 : ℕ) : ℝ)) ≤ _
    rw [Nat.cast_zero, Ex_const]
    exact L.bconfCost_nonneg x T
  · refine (Ex_mono (g := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
        (2 * T : ℝ) + L.runCostAt H T x s P q.1 q.2.1 + L.runCostAt H T x s P q.1 q.2.2) fun q => le_of_eq ?_).trans
      ((Ex_two_runs (fun ω rc => L.runCostAt H T x s P ω rc) _ _ fun ω =>
        runCostAt_exp hS hopen T hs P ω).trans_eq ?_)
    · simp only [runCostAt]
      push_cast
      ring
    · unfold bconfCost
      ring

/-- **`B_conf` declared at a budget, its cost on every coin**: at most `2T·(1 + n_sess·ρ_s) + 2·n_PU·openCalls`, each
run's extraction rerunning at most `n_sess·ρ_s` (`runExt_runs_le`) and calling at most `n_PU·openCalls` times itself
(`runExt_own_le`). -/
theorem BconfAt_cost_strict {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (T : ℕ) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    ∀ q, (BconfAt H L T Ω P x).cost q ≤ 2 * T * (1 + L.nSess x * L.ρs x) + 2 * (L.nPU x * L.openCalls x) := by
  delta BconfAt
  rcases hs : L.parse x with _ | s
  · exact fun _ => Nat.zero_le _
  · rintro ⟨ω, ⟨r₁, c₁⟩, ⟨r₂, c₂⟩⟩
    dsimp only
    have h₁ := runExt_runs_le hS P ω s r₁ c₁
    have h₂ := runExt_runs_le hS P ω s r₂ c₂
    have h₃ := runExt_own_le hS hopen hs P ω r₁ c₁
    have h₄ := runExt_own_le hS hopen hs P ω r₂ c₂
    have h₅ := Nat.mul_le_mul_left T (Nat.add_le_add h₁ h₂)
    nlinarith [h₃, h₄, h₅]

/-- **`KSReg` read at a budget**: where `ExpectedCR` holds at `B_conf` declared at `T`, `Pr[B_conf] ≤ c_T/2^256`
(`BconfAt_pr`, `BconfAt_cost_le`), so `ε_reg` is at most its reading at `T` and `KSReg`'s two bounds hold there. -/
theorem ksRegAt_of_ksReg {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (h : L.KSReg H) : L.KSRegAt H := by
  intro x hx Ω _ _ P T hE
  unfold ExpectedCR at hE
  have hpr : (Bconf H L Ω P x).Pr H ≤ L.bconfCost x T / 2 ^ (256 : ℝ) := by
    rw [← BconfAt_pr H L T Ω P x]
    exact hE.trans (div_le_div_of_nonneg_right (BconfAt_cost_le hS hopen T Ω P x) (by positivity))
  have hε : L.εreg H x P ≤ L.εregAt H x P T := by
    unfold Compiled.εreg Compiled.εregAt
    have := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hpr) (Nat.cast_nonneg (α := ℝ) (L.nG x))
    linarith
  obtain ⟨h₁, h₂⟩ := h x hx Ω P
  exact ⟨h₁.trans hε, fun s hs δ hδ => (h₂ s hs δ hδ).trans (by linarith)⟩

/-- **`KSReg` read at a budget and the strict birthday bound**: where `StrictBirthday` holds at `B_conf` declared at
`T`, `Pr[B_conf] ≤ q_T²/2^513`, `q_T` that finder's cost on every coin (`BconfAt_pr`, `BconfAt_cost_strict`), so
`ε_reg` is at most its reading there and `KSReg`'s two bounds hold. -/
theorem ksRegAtSB_of_ksReg {H : Msg → Msg} {L : Compiled Coin A} (hS : L.CostContract H) (hopen : L.OpenCallsLe H)
    (h : L.KSReg H) : L.KSRegAtSB H := by
  intro x hx Ω _ _ P T hB
  have hq : ∀ q, ((BconfAt H L T Ω P x).cost q : ℝ) ≤ L.bconfCostStrict x T := fun q => by
    unfold Compiled.bconfCostStrict
    exact_mod_cast BconfAt_cost_strict hS hopen T Ω P x q
  have hpr : (Bconf H L Ω P x).Pr H ≤ L.bconfCostStrict x T ^ 2 / 2 ^ (513 : ℝ) := by
    rw [← BconfAt_pr H L T Ω P x]
    exact hB _ hq
  have hε : L.εreg H x P ≤ L.εregSB H x P T := by
    unfold Compiled.εreg Compiled.εregSB
    have := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hpr) (Nat.cast_nonneg (α := ℝ) (L.nG x))
    linarith
  obtain ⟨h₁, h₂⟩ := h x hx Ω P
  exact ⟨h₁.trans hε, fun s hs δ hδ => (h₂ s hs δ hδ).trans (by linarith)⟩

end Compiled

/-- **`CompilerKSRegAt` from `CompilerKSReg`**: `B_conf`'s expected cost at each budget is bounded at the contract's
cost facts and the opening checks' count (`Compiled.ksRegAt_of_ksReg`). -/
theorem compilerKSRegAt_of (h : CompilerKSReg) : CompilerKSRegAt := by
  intro Coin _ _ _ A L H hA hB hS hopen
  exact Compiled.ksRegAt_of_ksReg hS.toCostContract hopen (h Coin A L H hA hB hS)

/-- **`CompilerKSRegAtSB` from `CompilerKSReg`**: `B_conf`'s cost on every coin at each budget is bounded at the
contract's cost facts and the opening checks' count (`Compiled.ksRegAtSB_of_ksReg`). -/
theorem compilerKSRegAtSB_of (h : CompilerKSReg) : CompilerKSRegAtSB := by
  intro Coin _ _ _ A L H hA hB hS hopen
  exact Compiled.ksRegAtSB_of_ksReg hS.toCostContract hopen (h Coin A L H hA hB hS)

end Claims.SamplingGeneric
