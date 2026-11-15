module

public import Proofs.Flock.RecGeneric.Cost
public import Proofs.Flock.RecGeneric.PlansProof
public import Proofs.Flock.RecGeneric.AtFlockBits

/-!
# The plans' expected costs from one run's

Each of Flock's plans meets `RunFinder.CostLe` at its count (`Cost.lean`): in every model whose runs cost at most `c`
calls on average, its expected cost is at most

* `forkPlan`'s `2c` (`forkPlanCost`): `u₀`, and its fork at the round guess, which averages a run's cost since the
  round is fixed before either run (`Ex_fork_slot`);
* `capPlan`'s `2c` (`capPlanCost`): `u₀`, and its caps' loop where `u₀` passes the loop's own gate, which Wald's bound
  over forks holds to one run (`Claims.Recursion.Ex_gate_loop_fork_le`);
* `conflictPlan`'s `(k + 1)c` (`conflictPlanCost`): `u₀`, and each of its `k` waits where `u₀` is accepted, each on
  its own fresh runs (`waitEmb`) and held to one run by Wald's bound (`Claims.Recursion.Ex_gate_loopCost_le`);
* `linkPlan`'s `2c` (`linkPlanCost`, `linkPlan_ExCost_two`).

At a selector value, tape and fresh runs, a plan's cost is at most its asks' (`dedup_append_sum_le`), since it looks at
and searches only runs it asked about or `u₀`: the caps' loop (`capLoop_cost`) and the waits (`waitLoop_cost`,
`waitsPlan_cost`) ask their runs up to the first that stops them, at `loopCost`. A finder whose plan meets `CostLe e`
and `ExpectedCR` finds a collision with probability at most `e(x)·c/2^256` (`RunFinder.PrAt_le_costLe`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims
open Claims.Recursion (fork loopCost)

variable {Coin : Type}

/-- A deduplicated list costs at most the list. -/
theorem dedup_sum_le {α : Type} [DecidableEq α] (f : α → ℕ) (l : List α) :
    ((l.dedup).map f).sum ≤ (l.map f).sum := by
  have h := dedup_append_sum_le f l [] (by simp)
  rwa [List.append_nil] at h

/-- A plan's cost is at most its asks' where every run it looks at or searches is one it asked about. -/
theorem cost_le_asks {α : Type} [DecidableEq α] (f : α → ℕ) (A B S : List α) (h : ∀ e ∈ B ++ S, e ∈ A) :
    (((A ++ B ++ S).dedup).map f).sum ≤ (A.map f).sum := by
  rw [List.append_assoc]
  exact dedup_append_sum_le f A (B ++ S) h

/-- An average over runs of a function of the first is the function's average. -/
theorem Ex_first {U : Type} [Fintype U] [Nonempty U] (K : ℕ) (f : U → ℝ) :
    Ex (fun us : Fin (K + 1) → U => f (us 0)) = Ex f :=
  calc Ex (fun us : Fin (K + 1) → U => f (us 0)) = Ex fun u : U => Ex fun _ : Fin K → U => f u :=
        Claims.Ex_cons K fun u _ => f u
    _ = Ex f := by
        congr 1
        funext u
        exact Ex_const _

/-- `K + 1` uniform runs are a uniform pair of the first and the rest. -/
theorem Ex_head_tail {U : Type} [Fintype U] (K : ℕ) (F : U → (Fin K → U) → ℝ) :
    Ex (fun us : Fin (K + 1) → U => F (us 0) fun i => us i.succ) = Ex (fun p : U × (Fin K → U) => F p.1 p.2) :=
  (Claims.Ex_cons K F).trans (Claims.Ex_prod fun p : U × (Fin K → U) => F p.1 p.2).symm

theorem Ex_finsum {α I : Type} [Fintype α] [Fintype I] (f : I → α → ℝ) :
    Ex (fun a => ∑ i, f i a) = ∑ i, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

section Loops

variable {Ω C : Type} {n : ℕ} (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C)

/-- **The caps' loop looks only at runs it asked about and searches `u₀` and those**; the runs it asks cost
`loopCost` over the slots it reads, the loop stopping at the first fork that passes the caps' gate. -/
theorem capLoop_cost (j : ℕ) (op : (ℕ → Coin) → Bool) :
    ∀ (N i : ℕ) (vs : Fin N → (Fin n → Coin) × C), (∀ t : Fin N, us (i + t.val) = vs t) →
      (∀ e ∈ ((capLoop (Coin := Coin) j op N i).exec m ω us).2.1,
        e ∈ ((capLoop (Coin := Coin) j op N i).exec m ω us).1) ∧
      (∀ e ∈ ((capLoop (Coin := Coin) j op N i).exec m ω us).2.2,
        e ∈ RunExp.fresh 0 :: ((capLoop (Coin := Coin) j op N i).exec m ω us).1) ∧
      (((capLoop (Coin := Coin) j op N i).exec m ω us).1.map fun e => m.cost ω (m.named us e)).sum =
        loopCost (fun v => capGate m ω op (fork (m.cut j) (us 0).1 v.1, v.2))
          (fun v => m.cost ω (fork (m.cut j) (us 0).1 v.1, v.2)) N vs
  | 0, _, _, _ => by simp [capLoop, Plan.exec, loopCost]
  | N + 1, i, vs, hvs => by
    classical
    have hv0 : us i = vs 0 := by simpa using hvs 0
    have hvs' : ∀ t : Fin N, us (i + 1 + t.val) = Fin.tail vs t := fun t =>
      (congrArg us (by simp only [Fin.val_succ]; omega)).trans (hvs t.succ)
    obtain ⟨h1, h2, h3⟩ := capLoop_cost j op N (i + 1) (Fin.tail vs) hvs'
    have hf : m.named us (.fork j (.fresh 0) (.fresh i)) = (fork (m.cut j) (us 0).1 (vs 0).1, (vs 0).2) := by
      simp only [RunModel.named, hv0]
    have hc : m.cost ω (m.named us (.fork j (.fresh 0) (.fresh i))) =
        m.cost ω (fork (m.cut j) (us 0).1 (vs 0).1, (vs 0).2) := by rw [hf]
    have hsrch : ∀ e ∈ ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).2.2,
        e ∈ RunExp.fresh 0 :: RunExp.fork j (.fresh 0) (.fresh i) ::
          ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).1 := fun e he => by
      rcases List.mem_cons.1 (h2 e he) with h | h
      · exact h ▸ List.mem_cons_self
      · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h)
    by_cases ha : m.acc ω (m.named us (.fork j (.fresh 0) (.fresh i)))
    · by_cases ho : op (m.coins (m.named us (.fork j (.fresh 0) (.fresh i)))) = true
      · have hg : capGate m ω op (fork (m.cut j) (us 0).1 (vs 0).1, (vs 0).2) := by
          rw [← hf]
          exact And.intro ha ho
        have hex : (capLoop (Coin := Coin) j op (N + 1) i).exec m ω us =
            ([.fork j (.fresh 0) (.fresh i)], [.fork j (.fresh 0) (.fresh i)],
              [.fresh 0, .fork j (.fresh 0) (.fresh i)]) := by
          simp [capLoop, Plan.exec, ha, ho]
        rw [hex]
        dsimp only
        refine ⟨fun e he => he, fun e he => he, ?_⟩
        rw [List.map_singleton, List.sum_singleton, hc]
        simp only [loopCost, hg, ↓reduceIte, add_zero]
      · have hg : ¬ capGate m ω op (fork (m.cut j) (us 0).1 (vs 0).1, (vs 0).2) := by
          rw [← hf]
          exact fun h => ho (And.right h)
        have hex : (capLoop (Coin := Coin) j op (N + 1) i).exec m ω us =
            (.fork j (.fresh 0) (.fresh i) :: ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).1,
              .fork j (.fresh 0) (.fresh i) :: ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).2.1,
              ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).2.2) := by
          simp [capLoop, Plan.exec, ha, ho]
        rw [hex]
        dsimp only
        refine ⟨fun e he => ?_, hsrch, ?_⟩
        · rcases List.mem_cons.1 he with h | h
          · exact h ▸ List.mem_cons_self
          · exact List.mem_cons_of_mem _ (h1 e h)
        · rw [List.map_cons, List.sum_cons, h3, hc]
          simp only [loopCost, hg, ↓reduceIte]
    · have hg : ¬ capGate m ω op (fork (m.cut j) (us 0).1 (vs 0).1, (vs 0).2) := by
        rw [← hf]
        exact fun h => ha (And.left h)
      have hex : (capLoop (Coin := Coin) j op (N + 1) i).exec m ω us =
          (.fork j (.fresh 0) (.fresh i) :: ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).1,
            ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).2.1,
            ((capLoop (Coin := Coin) j op N (i + 1)).exec m ω us).2.2) := by
        simp [capLoop, Plan.exec, ha]
      rw [hex]
      dsimp only
      refine ⟨fun e he => List.mem_cons_of_mem _ (h1 e he), hsrch, ?_⟩
      rw [List.map_cons, List.sum_cons, h3, hc]
      simp only [loopCost, hg, ↓reduceIte]

/-- **A wait asks its runs up to the first accepted one and then goes on as `cont` with it**: the runs it asks cost
`loopCost` over the slots it reads, and the run it hands on is one it asked. -/
theorem waitLoop_cost (cont : Option RunExp → Plan Coin) :
    ∀ (N i : ℕ) (vs : Fin N → (Fin n → Coin) × C), (∀ t : Fin N, us (i + t.val) = vs t) →
      ∃ (A : List RunExp) (o : Option RunExp),
        (waitLoop cont N i).exec m ω us =
          (A ++ ((cont o).exec m ω us).1, ((cont o).exec m ω us).2.1, ((cont o).exec m ω us).2.2) ∧
        (∀ e ∈ o.toList, e ∈ A) ∧
        (A.map fun e => m.cost ω (m.named us e)).sum = loopCost (m.acc ω) (m.cost ω) N vs
  | 0, _, _, _ => ⟨[], none, by first | rfl | simp [waitLoop], by simp, by simp [loopCost]⟩
  | N + 1, i, vs, hvs => by
    classical
    have hv0 : us i = vs 0 := by simpa using hvs 0
    have hvs' : ∀ t : Fin N, us (i + 1 + t.val) = Fin.tail vs t := fun t =>
      (congrArg us (by simp only [Fin.val_succ]; omega)).trans (hvs t.succ)
    by_cases ha : m.acc ω (us i)
    · refine ⟨[.fresh i], some (.fresh i), ?_, by simp, ?_⟩
      · simp [waitLoop, Plan.exec, RunModel.named, ha]
      · rw [List.map_singleton, List.sum_singleton]
        simp only [RunModel.named, loopCost, ← hv0, ha, ↓reduceIte, add_zero]
    · obtain ⟨A, o, hex, ho, hs⟩ := waitLoop_cost cont N (i + 1) (Fin.tail vs) hvs'
      refine ⟨.fresh i :: A, o, ?_, fun e he => List.mem_cons_of_mem _ (ho e he), ?_⟩
      · have h1 : (waitLoop cont (N + 1) i).exec m ω us =
            (.fresh i :: ((waitLoop cont N (i + 1)).exec m ω us).1, ((waitLoop cont N (i + 1)).exec m ω us).2.1,
              ((waitLoop cont N (i + 1)).exec m ω us).2.2) := by
          simp [waitLoop, Plan.exec, RunModel.named, ha]
        rw [h1, hex]
        try rfl
      · rw [List.map_cons, List.sum_cons, hs]
        simp only [RunModel.named, loopCost, ← hv0, ha, ↓reduceIte]

/-- **`w` waits of `R` runs look at nothing, search `found` and runs they asked**, and the runs they ask cost each
wait's `loopCost` over its own block of fresh runs, `base + t·R, …, base + t·R + R - 1`. -/
theorem waitsPlan_cost (R : ℕ) :
    ∀ (w base : ℕ) (found : List RunExp),
      ((waitsPlan (Coin := Coin) R w base found).exec m ω us).2.1 = [] ∧
      (∀ e ∈ ((waitsPlan (Coin := Coin) R w base found).exec m ω us).2.2,
        e ∈ found ∨ e ∈ ((waitsPlan (Coin := Coin) R w base found).exec m ω us).1) ∧
      (((waitsPlan (Coin := Coin) R w base found).exec m ω us).1.map fun e => m.cost ω (m.named us e)).sum =
        ∑ t : Fin w, loopCost (m.acc ω) (m.cost ω) R (fun k : Fin R => us (base + t.val * R + k.val))
  | 0, _, found => by simp [waitsPlan, Plan.exec]
  | w + 1, base, found => by
    obtain ⟨A, o, hex, ho, hs⟩ := waitLoop_cost m ω us
      (fun o => waitsPlan (Coin := Coin) R w (base + R) (found ++ o.toList)) R base
      (fun k : Fin R => us (base + k.val)) (fun _ => rfl)
    obtain ⟨h1, h2, h3⟩ := waitsPlan_cost R w (base + R) (found ++ o.toList)
    have hw : waitsPlan (Coin := Coin) R (w + 1) base found =
        waitLoop (fun o => waitsPlan (Coin := Coin) R w (base + R) (found ++ o.toList)) R base := rfl
    rw [hw, hex]
    dsimp only
    refine ⟨h1, fun e he => ?_, ?_⟩
    · rcases h2 e he with h | h
      · rcases List.mem_append.1 h with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_append_left _ (ho e h))
      · exact Or.inr (List.mem_append_right _ h)
    · rw [List.map_append, List.sum_append, hs, h3, Fin.sum_univ_succ]
      congr 1
      · simp
      · refine Finset.sum_congr rfl fun t _ => ?_
        congr 1
        funext k
        congr 1
        simp only [Fin.val_succ]
        ring

end Loops

section Plans

variable [Fintype Coin] [Nonempty Coin] {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}

/-- **`forkPlan`'s cost at a round guess, tape and runs** is at most `u₀`'s and its fork's. -/
theorem forkPlan_cost_le (k : Msg → ℕ) (m : RunModel Coin Ω n C) (x : Msg)
    (h : 0 < (forkPlan (Coin := Coin) k).sel x) (j : Fin (k x)) (ω : Ω) (us : Fin (1 + 1) → (Fin n → Coin) × C) :
    ((forkPlan (Coin := Coin) k).finder m x h).cost (j, ω, us) ≤
      m.cost ω (us 0) + m.cost ω (fork (m.cut j.val) (us 0).1 (us (0 : Fin 1).succ).1, (us (0 : Fin 1).succ).2) := by
  have hs0 : slotRuns us 0 = us 0 := congrArg us (Fin.ext (by simp))
  have hs1 : slotRuns us 1 = us (0 : Fin 1).succ := congrArg us (Fin.ext (by simp))
  have hcost : ((forkPlan (Coin := Coin) k).finder m x h).cost (j, ω, us) =
      ((([RunExp.fresh 0, .fork j.val (.fresh 0) (.fresh 1)]).dedup).map
        fun e => m.cost ω (m.named (slotRuns us) e)).sum := rfl
  rw [hcost]
  refine (dedup_sum_le _ _).trans (le_of_eq ?_)
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, RunModel.named, hs0, hs1]

open Classical in
/-- **`capPlan`'s cost at a cap position, tape and runs** is at most `u₀`'s and, where `u₀` passes the caps' gate,
the caps' loop's over the forks of `u₀`. -/
theorem capPlan_cost_le (Cp : Claims.Recursion.Caps Coin) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg)
    (h : 0 < (capPlan Cp M).sel x) (q : Fin ((capPlan Cp M).sel x)) (ω : Ω)
    (us : Fin (M + 1) → (Fin n → Coin) × C) :
    (((capPlan Cp M).finder m x h).cost (q, ω, us) : ℝ) ≤
      (m.cost ω (us 0) : ℝ) + if capGate m ω (Cp.opens x q) (us 0) then
        (loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2))
          (fun v => m.cost ω (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2)) M (fun i => us i.succ) : ℝ)
        else 0 := by
  have hs0 : slotRuns us 0 = us 0 := congrArg us (Fin.ext (by simp))
  have hplan : (capPlan Cp M).plan x q = .ask (.fresh 0) fun b =>
      if b then .look (.fresh 0) fun cs =>
        if Cp.opens x q cs then capLoop (Cp.round x q) (Cp.opens x q) M 1 else .done []
      else .done [] := rfl
  have hcost : ((capPlan Cp M).finder m x h).cost (q, ω, us) =
      (((((capPlan Cp M).plan x q).exec m ω (slotRuns us)).1 ++
        (((capPlan Cp M).plan x q).exec m ω (slotRuns us)).2.1 ++
        (((capPlan Cp M).plan x q).exec m ω (slotRuns us)).2.2).dedup.map
          fun e => m.cost ω (m.named (slotRuns us) e)).sum := rfl
  have hn0 : m.named (slotRuns us) (.fresh 0) = us 0 := hs0
  obtain ⟨h1, h2, h3⟩ := capLoop_cost m ω (slotRuns us) (Cp.round x q) (Cp.opens x q) M 1 (fun i => us i.succ)
    (fun t => congrArg us (Fin.ext (by have := t.isLt; simp only [Fin.val_succ]; omega)))
  rw [hs0] at h3
  by_cases ha : m.acc ω (us 0)
  · by_cases ho : Cp.opens x q (m.coins (us 0)) = true
    · have hg : capGate m ω (Cp.opens x q) (us 0) := And.intro ha ho
      have hex : ((capPlan Cp M).plan x q).exec m ω (slotRuns us) =
          (.fresh 0 :: ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω (slotRuns us)).1,
            .fresh 0 :: ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω (slotRuns us)).2.1,
            ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω (slotRuns us)).2.2) := by
        rw [hplan]
        simp [Plan.exec, RunModel.named, hs0, ha, ho]
      rw [hcost, hex, if_pos hg]
      have hle := cost_le_asks (fun e => m.cost ω (m.named (slotRuns us) e))
        (.fresh 0 :: ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω (slotRuns us)).1)
        (.fresh 0 :: ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω (slotRuns us)).2.1)
        ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω (slotRuns us)).2.2 fun e he => by
          rcases List.mem_append.1 he with he | he
          · rcases List.mem_cons.1 he with he | he
            · exact he ▸ List.mem_cons_self
            · exact List.mem_cons_of_mem _ (h1 e he)
          · exact h2 e he
      rw [List.map_cons, List.sum_cons, h3, hn0] at hle
      exact_mod_cast hle
    · have hg : ¬ capGate m ω (Cp.opens x q) (us 0) := fun h => ho (And.right h)
      have hex : ((capPlan Cp M).plan x q).exec m ω (slotRuns us) = ([.fresh 0], [.fresh 0], []) := by
        rw [hplan]
        simp [Plan.exec, RunModel.named, hs0, ha, ho]
      rw [hcost, hex, if_neg hg, add_zero]
      have hle := cost_le_asks (fun e => m.cost ω (m.named (slotRuns us) e)) [.fresh 0] [.fresh 0] []
        fun e he => by simpa using he
      rw [List.map_singleton, List.sum_singleton, hn0] at hle
      exact_mod_cast hle
  · have hg : ¬ capGate m ω (Cp.opens x q) (us 0) := fun h => ha (And.left h)
    have hex : ((capPlan Cp M).plan x q).exec m ω (slotRuns us) = ([.fresh 0], [], []) := by
      rw [hplan]
      simp [Plan.exec, RunModel.named, hs0, ha]
    rw [hcost, hex, if_neg hg, add_zero]
    have hle := cost_le_asks (fun e => m.cost ω (m.named (slotRuns us) e)) [.fresh 0] [] [] fun e he => by
      simp at he
    rw [List.map_singleton, List.sum_singleton, hn0] at hle
    exact_mod_cast hle

open Classical in
/-- **`conflictPlan`'s cost at a tape and runs** is at most `u₀`'s and, where `u₀` is accepted, each wait's over its
own block of runs. -/
theorem conflictPlan_cost_le (k : Msg → ℕ) (R : ℕ) (m : RunModel Coin Ω n C) (x : Msg)
    (h : 0 < (conflictPlan (Coin := Coin) k R).sel x) (q : Fin ((conflictPlan (Coin := Coin) k R).sel x)) (ω : Ω)
    (us : Fin (k x * R + 1) → (Fin n → Coin) × C) :
    (((conflictPlan (Coin := Coin) k R).finder m x h).cost (q, ω, us) : ℝ) ≤
      (m.cost ω (us 0) : ℝ) + ∑ t : Fin (k x), if m.acc ω (us 0) then
        (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => slotRuns us (1 + t.val * R + i.val)) : ℝ) else 0 := by
  have hs0 : slotRuns us 0 = us 0 := congrArg us (Fin.ext (by simp))
  have hn0 : m.named (slotRuns us) (.fresh 0) = us 0 := hs0
  have hplan : (conflictPlan (Coin := Coin) k R).plan x q = .ask (.fresh 0) fun b =>
      if b then waitsPlan R (k x) 1 [.fresh 0] else .done [] := rfl
  have hcost : ((conflictPlan (Coin := Coin) k R).finder m x h).cost (q, ω, us) =
      (((((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us)).1 ++
        (((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us)).2.1 ++
        (((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us)).2.2).dedup.map
          fun e => m.cost ω (m.named (slotRuns us) e)).sum := rfl
  obtain ⟨h1, h2, h3⟩ := waitsPlan_cost m ω (slotRuns us) R (k x) 1 [.fresh 0]
  by_cases ha : m.acc ω (us 0)
  · have hex : ((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us) =
        (.fresh 0 :: ((waitsPlan (Coin := Coin) R (k x) 1 [.fresh 0]).exec m ω (slotRuns us)).1,
          ((waitsPlan (Coin := Coin) R (k x) 1 [.fresh 0]).exec m ω (slotRuns us)).2.1,
          ((waitsPlan (Coin := Coin) R (k x) 1 [.fresh 0]).exec m ω (slotRuns us)).2.2) := by
      rw [hplan]
      simp [Plan.exec, RunModel.named, hs0, ha]
    rw [hcost, hex]
    simp only [ha, ↓reduceIte]
    have hle := cost_le_asks (fun e => m.cost ω (m.named (slotRuns us) e))
      (.fresh 0 :: ((waitsPlan (Coin := Coin) R (k x) 1 [.fresh 0]).exec m ω (slotRuns us)).1)
      ((waitsPlan (Coin := Coin) R (k x) 1 [.fresh 0]).exec m ω (slotRuns us)).2.1
      ((waitsPlan (Coin := Coin) R (k x) 1 [.fresh 0]).exec m ω (slotRuns us)).2.2 fun e he => by
        rw [h1, List.nil_append] at he
        rcases h2 e he with he | he
        · rw [List.mem_singleton] at he
          exact he ▸ List.mem_cons_self
        · exact List.mem_cons_of_mem _ he
    rw [List.map_cons, List.sum_cons, h3, hn0] at hle
    exact_mod_cast hle
  · have hex : ((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us) = ([.fresh 0], [], []) := by
      rw [hplan]
      simp [Plan.exec, RunModel.named, hs0, ha]
    rw [hcost, hex]
    simp only [ha, ↓reduceIte, Finset.sum_const_zero, add_zero]
    have hle := cost_le_asks (fun e => m.cost ω (m.named (slotRuns us) e)) [.fresh 0] [] [] fun e he => by
      simp at he
    rw [List.map_singleton, List.sum_singleton, hn0] at hle
    exact_mod_cast hle

end Plans

/-! ## The plans' costs -/

section Costs

variable [Fintype Coin] [Nonempty Coin]

/-- **`forkPlan` names two runs**: `u₀`, and its fork at the round guess, which averages a run's cost. -/
theorem forkPlanCost : ForkPlanCost Coin := by
  intro k Ω _ _ n C _ _ m x c hrun
  have hc : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrun (Classical.arbitrary Ω))
  unfold RunFinder.ExCost
  split_ifs with h
  · haveI : Nonempty (Fin ((forkPlan (Coin := Coin) k).sel x)) := ⟨⟨0, h⟩⟩
    let G : Fin ((forkPlan (Coin := Coin) k).sel x) → Ω → (Fin (1 + 1) → (Fin n → Coin) × C) → ℝ :=
      fun j ω us => (m.cost ω (us 0) : ℝ) +
        (m.cost ω (fork (m.cut j.val) (us 0).1 (us (0 : Fin 1).succ).1, (us (0 : Fin 1).succ).2) : ℝ)
    have h1 : Ex (fun s : Fin ((forkPlan (Coin := Coin) k).sel x) × Ω × (Fin (1 + 1) → (Fin n → Coin) × C) =>
        (((forkPlan (Coin := Coin) k).finder m x h).cost s : ℝ)) ≤
        Ex (fun s : Fin ((forkPlan (Coin := Coin) k).sel x) × Ω × (Fin (1 + 1) → (Fin n → Coin) × C) =>
          G s.1 s.2.1 s.2.2) :=
      Ex_mono fun s => by
        obtain ⟨j, ω, us⟩ := s
        have := forkPlan_cost_le k m x h j ω us
        simp only [G]
        exact_mod_cast this
    refine (h1.trans (Ex_sel_le G (2 * c) fun j ω => ?_)).trans_eq (by norm_num)
    show Ex (fun us : Fin (1 + 1) → (Fin n → Coin) × C => (m.cost ω (us 0) : ℝ) +
        (m.cost ω (fork (m.cut j.val) (us 0).1 (us (0 : Fin 1).succ).1, (us (0 : Fin 1).succ).2) : ℝ)) ≤ 2 * c
    calc Ex (fun us : Fin (1 + 1) → (Fin n → Coin) × C => (m.cost ω (us 0) : ℝ) +
          (m.cost ω (fork (m.cut j.val) (us 0).1 (us (0 : Fin 1).succ).1, (us (0 : Fin 1).succ).2) : ℝ))
        = Ex fun u₀ : (Fin n → Coin) × C => Ex fun σ : Fin 1 → (Fin n → Coin) × C =>
            (m.cost ω u₀ : ℝ) + (m.cost ω (fork (m.cut j.val) u₀.1 (σ 0).1, (σ 0).2) : ℝ) :=
          Claims.Ex_cons 1 fun (u₀ : (Fin n → Coin) × C) (σ : Fin 1 → (Fin n → Coin) × C) =>
            (m.cost ω u₀ : ℝ) + (m.cost ω (fork (m.cut j.val) u₀.1 (σ 0).1, (σ 0).2) : ℝ)
      _ = Ex fun u₀ : (Fin n → Coin) × C => (m.cost ω u₀ : ℝ) + Ex fun v : (Fin n → Coin) × C =>
            (m.cost ω (fork (m.cut j.val) u₀.1 v.1, v.2) : ℝ) := by
          congr 1
          funext u₀
          rw [Ex_add (fun _ : Fin 1 → (Fin n → Coin) × C => (m.cost ω u₀ : ℝ))
            (fun σ : Fin 1 → (Fin n → Coin) × C => (m.cost ω (fork (m.cut j.val) u₀.1 (σ 0).1, (σ 0).2) : ℝ)),
            Ex_const]
          congr 1
          exact Ex_first 0 fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j.val) u₀.1 v.1, v.2) : ℝ)
      _ = Ex (fun u₀ : (Fin n → Coin) × C => (m.cost ω u₀ : ℝ)) + Ex fun u₀ : (Fin n → Coin) × C =>
            Ex fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j.val) u₀.1 v.1, v.2) : ℝ) := Ex_add _ _
      _ ≤ 2 * c := by
          rw [Ex_fork_slot m ω j.val]
          linarith [hrun ω]
  · exact mul_nonneg (Nat.cast_nonneg _) hc

open Classical in
/-- **`capPlan` names two runs in expectation**: `u₀`, and its caps' loop gated by `u₀`'s passing the loop's own
gate, which Wald's bound over forks counts once. -/
theorem capPlanCost : CapPlanCost Coin := by
  intro Cp M Ω _ _ n C _ _ m x c hrun
  have hc : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrun (Classical.arbitrary Ω))
  unfold RunFinder.ExCost
  split_ifs with h
  · haveI : Nonempty (Fin ((capPlan Cp M).sel x)) := ⟨⟨0, h⟩⟩
    let G : Fin ((capPlan Cp M).sel x) → Ω → (Fin (M + 1) → (Fin n → Coin) × C) → ℝ := fun q ω us =>
      (m.cost ω (us 0) : ℝ) + if capGate m ω (Cp.opens x q) (us 0) then
        (loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2))
          (fun v => m.cost ω (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2)) M (fun i => us i.succ) : ℝ)
        else 0
    have h1 : Ex (fun s : Fin ((capPlan Cp M).sel x) × Ω × (Fin (M + 1) → (Fin n → Coin) × C) =>
        (((capPlan Cp M).finder m x h).cost s : ℝ)) ≤
        Ex (fun s : Fin ((capPlan Cp M).sel x) × Ω × (Fin (M + 1) → (Fin n → Coin) × C) => G s.1 s.2.1 s.2.2) :=
      Ex_mono fun s => by
        obtain ⟨q, ω, us⟩ := s
        exact capPlan_cost_le Cp M m x h q ω us
    refine (h1.trans (Ex_sel_le G (2 * c) fun q ω => ?_)).trans_eq (by norm_num)
    show Ex (fun us : Fin (M + 1) → (Fin n → Coin) × C => (m.cost ω (us 0) : ℝ) +
        if capGate m ω (Cp.opens x q) (us 0) then
          (loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2))
            (fun v => m.cost ω (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2)) M (fun i => us i.succ) : ℝ)
          else 0) ≤ 2 * c
    rw [Ex_add, Ex_first M fun u => (m.cost ω u : ℝ),
      Ex_head_tail M fun u (vs : Fin M → (Fin n → Coin) × C) => if capGate m ω (Cp.opens x q) u then
        (loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) u.1 v.1, v.2))
          (fun v => m.cost ω (fork (m.cut (Cp.round x q)) u.1 v.1, v.2)) M vs : ℝ)
        else 0]
    exact (add_le_add (hrun ω) ((Claims.Recursion.Ex_gate_loop_fork_le (m.cut (Cp.round x q))
      (capGate m ω (Cp.opens x q)) (m.cost ω) M).trans (hrun ω))).trans_eq (two_mul c).symm
  · exact mul_nonneg (Nat.cast_nonneg _) hc

open Classical in
/-- **`conflictPlan` names `k + 1` runs in expectation**: `u₀`, and each of its `k` waits gated by `u₀`'s acceptance,
each on its own fresh runs, which Wald's bound counts once. -/
theorem conflictPlanCost : ConflictPlanCost Coin := by
  intro k R Ω _ _ n C _ _ m x c hrun
  have hc : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrun (Classical.arbitrary Ω))
  unfold RunFinder.ExCost
  split_ifs with h
  · haveI : Nonempty (Fin ((conflictPlan (Coin := Coin) k R).sel x)) := ⟨⟨0, h⟩⟩
    let G : Fin ((conflictPlan (Coin := Coin) k R).sel x) → Ω →
        (Fin (k x * R + 1) → (Fin n → Coin) × C) → ℝ := fun _ ω us =>
      (m.cost ω (us 0) : ℝ) + ∑ t : Fin (k x), if m.acc ω (us 0) then
        (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => slotRuns us (1 + t.val * R + i.val)) : ℝ) else 0
    have h1 : Ex (fun s : Fin ((conflictPlan (Coin := Coin) k R).sel x) × Ω ×
          (Fin (k x * R + 1) → (Fin n → Coin) × C) =>
        (((conflictPlan (Coin := Coin) k R).finder m x h).cost s : ℝ)) ≤
        Ex (fun s : Fin ((conflictPlan (Coin := Coin) k R).sel x) × Ω ×
          (Fin (k x * R + 1) → (Fin n → Coin) × C) => G s.1 s.2.1 s.2.2) :=
      Ex_mono fun s => by
        obtain ⟨q, ω, us⟩ := s
        exact conflictPlan_cost_le k R m x h q ω us
    refine (h1.trans (Ex_sel_le G ((((k x : ℕ) : ℝ) + 1) * c) fun q ω => ?_)).trans_eq
      (by first | (push_cast; ring) | push_cast | norm_num | simp)
    show Ex (fun us : Fin (k x * R + 1) → (Fin n → Coin) × C => (m.cost ω (us 0) : ℝ) +
        ∑ t : Fin (k x), if m.acc ω (us 0) then
          (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => slotRuns us (1 + t.val * R + i.val)) : ℝ)
          else 0) ≤ (((k x : ℕ) : ℝ) + 1) * c
    have hwait : ∀ t : Fin (k x), Ex (fun us : Fin (k x * R + 1) → (Fin n → Coin) × C =>
        if m.acc ω (us 0) then
          (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => slotRuns us (1 + t.val * R + i.val)) : ℝ)
          else 0) ≤ c := by
        intro t
        have h0v : ((waitEmb (k x) R t 0 : Fin (k x * R + 1)) : ℕ) = ((0 : Fin (k x * R + 1)) : ℕ) := by
          first | simp [waitEmb] | rfl
        have hsv : ∀ i : Fin R, ((waitEmb (k x) R t i.succ : Fin (k x * R + 1)) : ℕ) =
            min (1 + (t : ℕ) * R + (i : ℕ)) (k x * R) := by
          intro i
          have h1 := block_end_le t R
          have h2 := i.isLt
          have hle : 1 + (t : ℕ) * R + (i : ℕ) ≤ k x * R := by omega
          have hne : ((i.succ : Fin (R + 1)) : ℕ) ≠ 0 := by
            rw [Fin.val_succ]
            omega
          show (if ((i.succ : Fin (R + 1)) : ℕ) = 0 then 0 else (t : ℕ) * R + ((i.succ : Fin (R + 1)) : ℕ)) = _
          rw [if_neg hne, min_eq_left hle, Fin.val_succ]
          omega
        let F : (Fin (R + 1) → (Fin n → Coin) × C) → ℝ := fun f =>
          if m.acc ω (f 0) then (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => f i.succ) : ℝ) else 0
        have hF : (fun us : Fin (k x * R + 1) → (Fin n → Coin) × C =>
            if m.acc ω (us 0) then
              (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => slotRuns us (1 + t.val * R + i.val)) : ℝ)
              else 0) = fun us => F (us ∘ waitEmb (k x) R t) := by
          funext us
          have e0 : us (waitEmb (k x) R t 0) = us 0 := congrArg us (Fin.ext h0v)
          have es : (fun i : Fin R => us (waitEmb (k x) R t i.succ)) =
              fun i : Fin R => slotRuns us (1 + (t : ℕ) * R + (i : ℕ)) :=
            funext fun i => congrArg us (Fin.ext (hsv i))
          show _ = if m.acc ω (us (waitEmb (k x) R t 0)) then
            (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => us (waitEmb (k x) R t i.succ)) : ℝ) else 0
          rw [e0, es]
        rw [hF, Claims.Recursion.UnifMap.comp_emb (γ := (Fin n → Coin) × C) (waitEmb (k x) R t) F]
        have hW := Claims.Recursion.Ex_gate_loopCost_le (U := (Fin n → Coin) × C) (m.acc ω) (m.cost ω) R
        have hsplit : Ex F = Ex (fun p : ((Fin n → Coin) × C) × (Fin R → (Fin n → Coin) × C) =>
            if m.acc ω p.1 then (loopCost (m.acc ω) (m.cost ω) R p.2 : ℝ) else 0) :=
          Ex_head_tail R fun u vs => if m.acc ω u then (loopCost (m.acc ω) (m.cost ω) R vs : ℝ) else 0
        rw [hsplit]
        exact hW.trans (hrun ω)
    rw [Ex_add, Ex_first (k x * R) fun u => (m.cost ω u : ℝ),
      Ex_finsum (fun (t : Fin (k x)) (us : Fin (k x * R + 1) → (Fin n → Coin) × C) =>
        if m.acc ω (us 0) then
          (loopCost (m.acc ω) (m.cost ω) R (fun i : Fin R => slotRuns us (1 + t.val * R + i.val)) : ℝ)
          else 0)]
    calc _ ≤ c + ∑ _t : Fin (k x), c := add_le_add (hrun ω) (Finset.sum_le_sum fun t _ => hwait t)
      _ = (((k x : ℕ) : ℝ) + 1) * c := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring
  · exact mul_nonneg (Nat.cast_nonneg _) hc

/-- **The link's plan names two runs in expectation** (`linkPlan_ExCost_two`). -/
theorem linkPlanCost : LinkPlanCost Coin := by
  intro k M Ω _ _ n C _ _ m x c hrun
  refine (linkPlan_ExCost_two k M m x c hrun).trans_eq ?_
  norm_num

end Costs

/-- **A finder whose plan names at most `e` runs in expectation** finds a collision with probability at most
`e(x)·c/2^256` in a model whose runs cost at most `c` calls on average, where it meets `ExpectedCR`. -/
theorem RunFinder.PrAt_le_costLe [Fintype Coin] [Nonempty Coin] {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C]
    [Nonempty C] {n : ℕ} (H : Msg → Msg) (F : RunFinder Coin) (e : Msg → ℕ) (hF : F.CostLe e)
    (m : RunModel Coin Ω n C) (x : Msg) (c : ℝ) (hrun : ∀ ω, Ex (fun u => (m.cost ω u : ℝ)) ≤ c)
    (hCR : ∀ h : 0 < F.sel x, ExpectedCR H (F.finder m x h)) : F.PrAt H m x ≤ e x * c / 2 ^ (256 : ℝ) := by
  have hp : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  unfold RunFinder.PrAt
  split_ifs with h
  · refine (hCR h).trans (div_le_div_of_nonneg_right ?_ hp.le)
    have hx : F.ExCost m x = Ex fun s => ((F.finder m x h).cost s : ℝ) := dif_pos h
    have hE := hF Ω n C m x c hrun
    rw [hx] at hE
    exact hE
  · have hc : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrun (Classical.arbitrary Ω))
    exact div_nonneg (mul_nonneg (Nat.cast_nonneg _) hc) hp.le

end Claims.RecGeneric
