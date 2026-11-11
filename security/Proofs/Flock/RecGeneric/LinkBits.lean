module

public import Proofs.Flock.RecGeneric.PlansProof
public import Proofs.Flock.Claims.ERepCost

/-!
# The binding error's bits (F9b)

A run finder's collision bound reads its expected cost (`ExpectedCR`), and `RunFinder.MadeLe F e` bounds that cost by
`e(x) · c` in every model whose runs, and forks of them, cost at most `c` calls on average. So a finder that meets
`ExpectedCR` in a model finds a collision with probability at most `e(x) · c / 2^256` there
(`RunFinder.PrAt_le_madeLe`).

The link's plan meets `MadeLe` at `M + 1` (`linkPlanMade`): at its round guess `j` it names only the forks
`fork_j(u₀, u_{i+1})`, `i < M + 1`, of the shared first run (`linkPlan_names`), its cost counts each once
(`linkPlan_cost_le`), and each fork's slot `u_{i+1}` is uniform and independent of `u₀`, so each costs at most `c` on
average (`Ex_forks_le`), whether or not the plan reaches it.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims
open Claims.Recursion (fork)

/-- **An average over a selector, a tape and samples** is at most `b` where each selector value and tape's is. -/
theorem Ex_sel_le {S Ω V : Type} [Fintype S] [Nonempty S] [Fintype Ω] [Nonempty Ω] [Fintype V]
    (F : S → Ω → V → ℝ) (b : ℝ) (h : ∀ q ω, Ex (F q ω) ≤ b) :
    Ex (fun s : S × Ω × V => F s.1 s.2.1 s.2.2) ≤ b :=
  calc Ex (fun s : S × Ω × V => F s.1 s.2.1 s.2.2)
      = Ex fun q : S => Ex fun ω : Ω => Ex (F q ω) := by
        rw [Claims.Ex_prod]
        congr 1
        funext q
        exact Claims.Ex_prod (fun b : Ω × V => F q b.1 b.2)
    _ ≤ Ex fun _ : S => Ex fun _ : Ω => b := Ex_mono fun q => Ex_mono fun ω => h q ω
    _ = b := by rw [Claims.Ex_const, Claims.Ex_const]

variable {Coin : Type}

section Names

variable {Ω C : Type} {n : ℕ} (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C)

/-- **The link's loop names only `first` and its forks** `fork_j(u₀, u_t)`, `i ≤ t < i + N`. -/
theorem linkLoop_names (j : ℕ) (first : RunExp) (P : RunExp → Prop) (hf : P first) :
    ∀ (N i : ℕ), (∀ t, i ≤ t → t < i + N → P (.fork j (.fresh 0) (.fresh t))) →
      ∀ e ∈ ((linkLoop (Coin := Coin) j first N i).exec m ω us).1 ++
        ((linkLoop (Coin := Coin) j first N i).exec m ω us).2.1 ++
        ((linkLoop (Coin := Coin) j first N i).exec m ω us).2.2, P e
  | 0, _, _ => by
    intro e he
    simp [linkLoop, Plan.exec] at he
    exact he ▸ hf
  | N + 1, i, hP => by
    classical
    intro e he
    have hi : P (.fork j (.fresh 0) (.fresh i)) := hP i le_rfl (by omega)
    have ih := linkLoop_names j first P hf N (i + 1) fun t h1 h2 => hP t (by omega) (by omega)
    by_cases ha : m.acc ω (m.named us (.fork j (.fresh 0) (.fresh i)))
    · simp [linkLoop, Plan.exec, ha] at he
      rcases he with rfl | rfl | rfl <;> assumption
    · simp [linkLoop, Plan.exec, ha] at he
      rcases he with rfl | he
      · exact hi
      · exact ih e (by simpa using he)

/-- **The link's plan names only `fork_j(u₀, u_{i+1})`**, `i < M + 1`, at its round guess `j`. -/
theorem linkPlan_names (k : Msg → ℕ) (M : ℕ) (x : Msg) (j : Fin (k x)) :
    ∀ e ∈ (((linkPlan (Coin := Coin) k M).plan x j).exec m ω us).1 ++
      (((linkPlan (Coin := Coin) k M).plan x j).exec m ω us).2.1 ++
      (((linkPlan (Coin := Coin) k M).plan x j).exec m ω us).2.2,
        ∃ i : Fin (M + 1), e = .fork j.val (.fresh 0) (.fresh (i.val + 1)) := by
  classical
  have hf : ∃ i : Fin (M + 1), RunExp.fork j.val (.fresh 0) (.fresh 1) =
      .fork j.val (.fresh 0) (.fresh (i.val + 1)) := ⟨0, rfl⟩
  have hl := linkLoop_names (Coin := Coin) m ω us j.val _
    (fun e => ∃ i : Fin (M + 1), e = .fork j.val (.fresh 0) (.fresh (i.val + 1))) hf M 2 fun t h1 h2 =>
    ⟨⟨t - 1, by omega⟩, by simp only [RunExp.fork.injEq, RunExp.fresh.injEq, true_and]; omega⟩
  intro e he
  by_cases ha : m.acc ω (m.named us (.fork j.val (.fresh 0) (.fresh 1)))
  · simp [linkPlan, Plan.exec, ha] at he
    rcases he with rfl | he
    · exact hf
    · exact hl e (by simpa using he)
  · simp [linkPlan, Plan.exec, ha] at he
    exact he ▸ hf

end Names

variable [Fintype Coin] [Nonempty Coin]

/-- **`linkPlan`'s cost at a round guess, tape and slots** is at most its `M + 1` forks' `fork_j(u₀, u_{i+1})`, each
named once. -/
theorem linkPlan_cost_le {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}
    (k : Msg → ℕ) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg) (h : 0 < (linkPlan (Coin := Coin) k M).sel x)
    (j : Fin (k x)) (ω : Ω) (us : Fin (M + 1 + 1) → (Fin n → Coin) × C) :
    ((linkPlan (Coin := Coin) k M).finder m x h).cost (j, ω, us) ≤
      ∑ i : Fin (M + 1), m.cost ω (fork (m.cut j) (us 0).1 (us i.succ).1, (us i.succ).2) := by
  classical
  let g : Fin (M + 1) → RunExp := fun i => .fork j.val (.fresh 0) (.fresh (i.val + 1))
  let f : RunExp → ℕ := fun e => m.cost ω (m.named (slotRuns us) e)
  let L := (((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us)).1 ++
      (((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us)).2.1 ++
      (((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us)).2.2
  have hg : Function.Injective g := fun a b hab => by
    simp only [g, RunExp.fork.injEq, RunExp.fresh.injEq, true_and] at hab
    exact Fin.ext (by omega)
  have hsub : L.dedup.toFinset ⊆ Finset.univ.image g := fun e he => by
    obtain ⟨i, rfl⟩ := linkPlan_names m ω (slotRuns us) k M x j e
      (List.mem_dedup.1 (List.mem_toFinset.1 he))
    exact Finset.mem_image_of_mem g (Finset.mem_univ i)
  have h0 : slotRuns us 0 = us 0 := congrArg us (Fin.ext (by simp))
  have hs : ∀ i : Fin (M + 1), slotRuns us (i.val + 1) = us i.succ := fun i =>
    congrArg us (Fin.ext (by rw [Fin.val_succ]; exact min_eq_left (by omega)))
  calc ((linkPlan (Coin := Coin) k M).finder m x h).cost (j, ω, us)
      = (L.dedup.map f).sum := rfl
    _ = ∑ e ∈ L.dedup.toFinset, f e := (List.sum_toFinset f (List.nodup_dedup L)).symm
    _ ≤ ∑ e ∈ Finset.univ.image g, f e := Finset.sum_le_sum_of_subset hsub
    _ = ∑ i, f (g i) := Finset.sum_image fun a _ b _ hab => hg hab
    _ = _ := Finset.sum_congr rfl fun i _ => by
        simp only [f, g, RunModel.named, h0, hs]

/-- At a tape and round, `M + 1` forks of a shared first run, each with its own uniform slot, average at most
`(M + 1) · c` where every fork averages at most `c`. -/
theorem Ex_forks_le {Ω C : Type} [Fintype C] [Nonempty C] {n : ℕ} (M : ℕ) (m : RunModel Coin Ω n C) (ω : Ω)
    (j : ℕ) (c : ℝ) (hfork : ∀ u₀ : (Fin n → Coin) × C,
      Ex (fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j) u₀.1 v.1, v.2) : ℝ)) ≤ c) :
    Ex (fun us : Fin (M + 1 + 1) → (Fin n → Coin) × C =>
      ∑ i : Fin (M + 1), (m.cost ω (fork (m.cut j) (us 0).1 (us i.succ).1, (us i.succ).2) : ℝ)) ≤
        ((M + 1 : ℕ) : ℝ) * c := by
  let G : (Fin n → Coin) × C → (Fin (M + 1) → (Fin n → Coin) × C) → ℝ := fun u₀ σ =>
    ∑ i, (m.cost ω (fork (m.cut j) u₀.1 (σ i).1, (σ i).2) : ℝ)
  have h1 := Claims.Ex_cons (D := (Fin n → Coin) × C) (M + 1) G
  calc _ = Ex fun u₀ => Ex fun σ => G u₀ σ := h1
    _ = Ex fun u₀ : (Fin n → Coin) × C => ((M + 1 : ℕ) : ℝ) *
          Ex fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j) u₀.1 v.1, v.2) : ℝ) := by
        congr 1
        funext u₀
        exact Claims.Ex_sum_pi (fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j) u₀.1 v.1, v.2) : ℝ))
          (M + 1)
    _ ≤ Ex fun _ : (Fin n → Coin) × C => ((M + 1 : ℕ) : ℝ) * c :=
        Ex_mono fun u₀ => mul_le_mul_of_nonneg_left (hfork u₀) (Nat.cast_nonneg _)
    _ = ((M + 1 : ℕ) : ℝ) * c := Claims.Ex_const _

/-- **`linkPlan`'s expected cost** at a nonempty round guess is at most `(M + 1) · c`, where every fork averages at
most `c`. -/
theorem linkPlan_ExCost_le {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}
    (k : Msg → ℕ) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg) (h : 0 < (linkPlan (Coin := Coin) k M).sel x)
    [Nonempty (Fin (k x))] (c : ℝ)
    (hfork : ∀ (ω : Ω) (j : ℕ) (u₀ : (Fin n → Coin) × C),
      Ex (fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j) u₀.1 v.1, v.2) : ℝ)) ≤ c) :
    Ex (fun s => (((linkPlan (Coin := Coin) k M).finder m x h).cost s : ℝ)) ≤ ((M + 1 : ℕ) : ℝ) * c := by
  let G : Fin (k x) → Ω → (Fin (M + 1 + 1) → (Fin n → Coin) × C) → ℝ := fun j ω us =>
    ∑ i : Fin (M + 1), (m.cost ω (fork (m.cut j) (us 0).1 (us i.succ).1, (us i.succ).2) : ℝ)
  have h1 : Ex (fun s : Fin (k x) × Ω × (Fin (M + 1 + 1) → (Fin n → Coin) × C) =>
      (((linkPlan (Coin := Coin) k M).finder m x h).cost s : ℝ)) ≤
      Ex (fun s : Fin (k x) × Ω × (Fin (M + 1 + 1) → (Fin n → Coin) × C) => G s.1 s.2.1 s.2.2) :=
    Ex_mono fun s => by
      obtain ⟨j, ω, us⟩ := s
      have := linkPlan_cost_le k M m x h j ω us
      simp only [G]
      exact_mod_cast this
  exact h1.trans (Ex_sel_le G _ fun j ω => Ex_forks_le M m ω j c (hfork ω j))

/-- **F9b: the link's plan names at most `M + 1` runs, each of cost at most `c` on average**: every run it names is a
fork `fork_j(u₀, u_{i+1})` of the shared first run at its round guess, `i < M + 1`, and the slot `u_{i+1}` it forks
with is uniform whatever the plan saw before reaching it. -/
theorem linkPlanMade (k : Msg → ℕ) (M : ℕ) : (linkPlan (Coin := Coin) k M).MadeLe fun _ => M + 1 := by
  intro Ω _ _ n C _ _ m x c hfork hrun
  have hc : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrun (Classical.arbitrary Ω))
  unfold RunFinder.ExCost
  split_ifs with h
  · have : Nonempty (Fin (k x)) := ⟨⟨0, h⟩⟩
    exact linkPlan_ExCost_le k M m x h c hfork
  · exact mul_nonneg (Nat.cast_nonneg _) hc

/-- **A run finder's collision bound from its expected cost**: where `F` names at most `e` runs of cost at most `c` on
average (`MadeLe`) and meets `ExpectedCR` in the model `m`, it finds a collision with probability at most
`e(x) · c / 2^256` there. -/
theorem RunFinder.PrAt_le_madeLe {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}
    (H : Msg → Msg) (F : RunFinder Coin) (e : Msg → ℕ) (hF : F.MadeLe e) (m : RunModel Coin Ω n C) (x : Msg)
    (c : ℝ) (hfork : ∀ (ω : Ω) (j : ℕ) (u₀ : (Fin n → Coin) × C),
      Ex (fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j) u₀.1 v.1, v.2) : ℝ)) ≤ c)
    (hrun : ∀ ω, Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) ≤ c)
    (hCR : ∀ h : 0 < F.sel x, ExpectedCR H (F.finder m x h)) :
    F.PrAt H m x ≤ e x * c / 2 ^ (256 : ℝ) := by
  have hc : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrun (Classical.arbitrary Ω))
  have hE := hF Ω n C m x c hfork hrun
  unfold RunFinder.PrAt
  split_ifs with h
  · refine (hCR h).trans (div_le_div_of_nonneg_right ?_ (by positivity))
    have hx : F.ExCost m x = Ex fun s => ((F.finder m x h).cost s : ℝ) := dif_pos h
    exact hx ▸ hE
  · exact div_nonneg (mul_nonneg (Nat.cast_nonneg _) hc) (by positivity)

end Claims.RecGeneric
