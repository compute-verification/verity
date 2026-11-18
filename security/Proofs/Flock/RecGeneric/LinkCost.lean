module

public import Proofs.Flock.RecGeneric.Plans
public import Proofs.Flock.Claims.Recursion.KSLin2
public import Proofs.Flock.Claims.Recursion.PlurLin
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ERepCost

/-!
# The gated link's expected cost

`linkPlan k M` asks whether `fork_j(u₀, u₁)` is accepted and only then loops over `fork_j(u₀, u₂), …` to the first that
is. Every run it names is one it asked about, so its cost is at most `Claims.Recursion.linkCost` at the forks of `u₀`
(`linkPlan_cost_le`), and Wald's bound on that gated loop (`Claims.Recursion.Ex_linkCost_le`) with the forks of a
uniform prefix being uniform (`Claims.Recursion.unifMap_forkPair`) gives, in every model, at most two runs in
expectation (`linkPlan_ExCost_le`). Under `ExpectedCR` at its finder it finds a collision with probability at most
`2·E[a run's cost]/2^256` (`linkPlan_PrAt_le_ecr`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

section Exec

variable {Coin Ω C : Type} {n : ℕ} (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C)

/-- The fork at `j` of the run `us 0` with the run `v`. -/
def forkOf (j : ℕ) (v : (Fin n → Coin) × C) : (Fin n → Coin) × C := (fork (m.cut j) (us 0).1 v.1, v.2)

open Classical in
/-- **The link's loop asks only about its forks and searches `first` or one of them**; its asks cost `loopCost` at the
forks of `us 0`. -/
theorem linkLoop_exec (j : ℕ) (first : RunExp) : ∀ (N i : ℕ),
    ((linkLoop (Coin := Coin) j first N i).exec m ω us).2.1 = [] ∧
    (∀ e ∈ ((linkLoop (Coin := Coin) j first N i).exec m ω us).2.2,
      e = first ∨ e ∈ ((linkLoop (Coin := Coin) j first N i).exec m ω us).1) ∧
    (((linkLoop (Coin := Coin) j first N i).exec m ω us).1.map fun e => m.cost ω (m.named us e)).sum =
      loopCost (fun v => m.acc ω (forkOf m us j v)) (fun v => m.cost ω (forkOf m us j v)) N
        (fun t : Fin N => us (i + t.val))
  | 0, i => by simp [linkLoop, Plan.exec, loopCost]
  | N + 1, i => by
    have hn : m.named us (.fork j (.fresh 0) (.fresh i)) = forkOf m us j (us i) := rfl
    have ht : (Fin.tail fun t : Fin (N + 1) => us (i + t.val)) = fun t : Fin N => us (i + 1 + t.val) := by
      funext t
      simp only [Fin.tail, Fin.val_succ]
      congr 1
      omega
    obtain ⟨h1, h2, h3⟩ := linkLoop_exec j first N (i + 1)
    by_cases ha : m.acc ω (forkOf m us j (us i))
    · have hex : (linkLoop (Coin := Coin) j first (N + 1) i).exec m ω us =
          ([.fork j (.fresh 0) (.fresh i)], [], [first, .fork j (.fresh 0) (.fresh i)]) := by
        simp only [linkLoop, Plan.exec, hn, decide_eq_true_eq, ha, ↓reduceIte]
      rw [hex]
      refine ⟨rfl, fun e he => ?_, ?_⟩
      · simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he ⊢
        exact he
      · simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, hn, loopCost,
          Fin.val_zero, ha, ↓reduceIte]
    · have hex : (linkLoop (Coin := Coin) j first (N + 1) i).exec m ω us =
          (.fork j (.fresh 0) (.fresh i) :: ((linkLoop (Coin := Coin) j first N (i + 1)).exec m ω us).1,
            ((linkLoop (Coin := Coin) j first N (i + 1)).exec m ω us).2.1,
            ((linkLoop (Coin := Coin) j first N (i + 1)).exec m ω us).2.2) := by
        simp only [linkLoop, Plan.exec, hn, decide_eq_true_eq, ha, ↓reduceIte]
      rw [hex]
      refine ⟨h1, fun e he => ?_, ?_⟩
      · rcases h2 e he with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_cons_of_mem _ h)
      · simp only [List.map_cons, List.sum_cons, hn, h3, loopCost, Fin.val_zero, Nat.add_zero, ha, ↓reduceIte, ht]

/-- A sum over a list without repeats is at most the sum over a list holding all its entries. -/
theorem sum_dedup_le {α : Type} [DecidableEq α] (f : α → ℕ) (l l' : List α) (h : ∀ e ∈ l, e ∈ l') :
    (l.dedup.map f).sum ≤ (l'.map f).sum := by
  rw [← List.sum_toFinset f (List.nodup_dedup l)]
  calc ∑ e ∈ l.dedup.toFinset, f e ≤ ∑ e ∈ l'.toFinset, f e :=
        Finset.sum_le_sum_of_subset fun e he => by
          rw [List.mem_toFinset, List.mem_dedup] at he
          exact List.mem_toFinset.2 (h e he)
    _ ≤ (l'.map f).sum := by
        induction l' with
        | nil => simp
        | cons a l' ih =>
          rw [List.toFinset_cons, List.map_cons, List.sum_cons]
          calc ∑ e ∈ insert a l'.toFinset, f e ≤ f a + ∑ e ∈ l'.toFinset, f e := by
                by_cases ha : a ∈ l'.toFinset
                · rw [Finset.insert_eq_of_mem ha]; exact Nat.le_add_left _ _
                · rw [Finset.sum_insert ha]
            _ ≤ f a + (l'.map f).sum := Nat.add_le_add_left ih _

end Exec

section Finder

variable {Coin Ω C : Type} [Fintype Coin] [Nonempty Coin] [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C]
  {n : ℕ}

open Classical in
/-- **The link's cost at its coins is at most `linkCost` at the forks of the first fresh run**: the gate's fork, and
the loop's forks where the gate's is accepted. -/
theorem linkPlan_cost_le (k : Msg → ℕ) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg) (h : 0 < k x) (j : Fin (k x))
    (ω : Ω) (sl : Fin (M + 1 + 1) → (Fin n → Coin) × C) :
    ((linkPlan (Coin := Coin) k M).finder m x h).cost (j, ω, sl) ≤
      linkCost (fun v => m.acc ω (forkOf m (slotRuns sl) j.val v))
        (fun v => m.cost ω (forkOf m (slotRuns sl) j.val v)) M
        (slotRuns sl 1, fun t : Fin M => slotRuns sl (2 + t.val)) := by
  have hn : m.named (slotRuns sl) (.fork j.val (.fresh 0) (.fresh 1)) = forkOf m (slotRuns sl) j.val (slotRuns sl 1) :=
    rfl
  obtain ⟨h1, h2, h3⟩ := linkLoop_exec m ω (slotRuns sl) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2
  show ((((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns sl)).1 ++
      (((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns sl)).2.1 ++
      (((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns sl)).2.2).dedup.map
        (fun e => m.cost ω (m.named (slotRuns sl) e)) |>.sum ≤ _
  by_cases ha : m.acc ω (forkOf m (slotRuns sl) j.val (slotRuns sl 1))
  · have hex : ((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns sl) =
        (.fork j.val (.fresh 0) (.fresh 1) ::
            ((linkLoop (Coin := Coin) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec m ω (slotRuns sl)).1,
          ((linkLoop (Coin := Coin) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec m ω (slotRuns sl)).2.1,
          ((linkLoop (Coin := Coin) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec m ω (slotRuns sl)).2.2) := by
      simp only [linkPlan, Plan.exec, hn, decide_eq_true_eq, ha, ↓reduceIte]
    rw [hex]
    refine (sum_dedup_le _ _ (.fork j.val (.fresh 0) (.fresh 1) ::
      ((linkLoop (Coin := Coin) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec m ω (slotRuns sl)).1)
      fun e he => ?_).trans (le_of_eq ?_)
    · rcases List.mem_append.1 he with he | he
      · rcases List.mem_append.1 he with he | he
        · exact he
        · rw [h1] at he
          simp at he
      · rcases h2 e he with h | h
        · rw [h]
          simp
        · exact List.mem_cons_of_mem _ h
    · simp only [List.map_cons, List.sum_cons, hn, h3, linkCost, ha, ↓reduceIte]
  · have hex : ((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns sl) =
        ([.fork j.val (.fresh 0) (.fresh 1)], [], []) := by
      simp only [linkPlan, Plan.exec, hn, decide_eq_true_eq, ha, ↓reduceIte]
    rw [hex]
    refine (sum_dedup_le _ _ [.fork j.val (.fresh 0) (.fresh 1)] fun e he => by simpa using he).trans (le_of_eq ?_)
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, hn, linkCost, ha, ↓reduceIte]

/-- The finder's fresh runs as the first, the gate's continuation and the loop's `M`. -/
theorem unifMap_slots (M : ℕ) :
    UnifMap (fun p : ((Fin n → Coin) × C) × ((Fin n → Coin) × C) × (Fin M → (Fin n → Coin) × C) =>
      (Fin.cons p.1 (Fin.cons p.2.1 p.2.2 : Fin (M + 1) → (Fin n → Coin) × C) :
        Fin (M + 1 + 1) → (Fin n → Coin) × C)) :=
  UnifMap.equiv (((Equiv.refl _).prodCongr (Fin.consEquiv fun _ : Fin (M + 1) => (Fin n → Coin) × C)).trans
    (Fin.consEquiv fun _ : Fin (M + 1 + 1) => (Fin n → Coin) × C))

theorem slotRuns_cons {M : ℕ} (a b : (Fin n → Coin) × C) (vs : Fin M → (Fin n → Coin) × C) :
    slotRuns (Fin.cons a (Fin.cons b vs : Fin (M + 1) → (Fin n → Coin) × C) : Fin (M + 1 + 1) → _) 0 = a ∧
    slotRuns (Fin.cons a (Fin.cons b vs : Fin (M + 1) → (Fin n → Coin) × C) : Fin (M + 1 + 1) → _) 1 = b ∧
    ∀ t : Fin M,
      slotRuns (Fin.cons a (Fin.cons b vs : Fin (M + 1) → (Fin n → Coin) × C) : Fin (M + 1 + 1) → _) (2 + t.val) =
        vs t := by
  refine ⟨?_, ?_, fun t => ?_⟩
  · have : (⟨min 0 (M + 1), Nat.lt_succ_of_le (Nat.min_le_right 0 (M + 1))⟩ : Fin (M + 1 + 1)) = 0 := by
      ext
      simp only [Fin.val_zero] <;> omega
    simp only [slotRuns, this, Fin.cons_zero]
  · have : (⟨min 1 (M + 1), Nat.lt_succ_of_le (Nat.min_le_right 1 (M + 1))⟩ : Fin (M + 1 + 1)) = Fin.succ 0 := by
      ext
      simp only [Fin.val_succ, Fin.val_zero] <;> omega
    simp only [slotRuns, this, Fin.cons_succ, Fin.cons_zero]
  · have := t.isLt
    have : (⟨min (2 + t.val) (M + 1), Nat.lt_succ_of_le (Nat.min_le_right _ (M + 1))⟩ : Fin (M + 1 + 1)) =
        Fin.succ (Fin.succ t) := by
      ext
      simp only [Fin.val_succ] <;> omega
    simp only [slotRuns, this, Fin.cons_succ]

/-- **The link names at most two runs in expectation**, in every model: `E[cost] ≤ 2·E_{ω,u}[cost ω u]`. -/
theorem linkPlan_ExCost_le (k : Msg → ℕ) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg) :
    (linkPlan (Coin := Coin) k M).ExCost m x ≤ 2 * Ex (fun t : Ω × ((Fin n → Coin) × C) => (m.cost t.1 t.2 : ℝ)) := by
  unfold RunFinder.ExCost
  have hc0 : 0 ≤ Ex (fun t : Ω × ((Fin n → Coin) × C) => (m.cost t.1 t.2 : ℝ)) :=
    Ex_nonneg fun _ => Nat.cast_nonneg _
  by_cases h : 0 < (linkPlan (Coin := Coin) k M).sel x
  swap
  · rw [dif_neg h]; linarith
  rw [dif_pos h]
  have hω : ∀ (j : Fin (k x)) (ω : Ω),
      (Ex fun sl : Fin (M + 1 + 1) → (Fin n → Coin) × C =>
        (((linkPlan (Coin := Coin) k M).finder m x h).cost (j, ω, sl) : ℝ)) ≤
        2 * Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) := by
    intro j ω
    calc _ ≤ Ex fun sl : Fin (M + 1 + 1) → (Fin n → Coin) × C =>
            (linkCost (fun v => m.acc ω (forkOf m (slotRuns sl) j.val v))
              (fun v => m.cost ω (forkOf m (slotRuns sl) j.val v)) M
              (slotRuns sl 1, fun t : Fin M => slotRuns sl (2 + t.val)) : ℝ) :=
          Ex_mono fun sl => by exact_mod_cast linkPlan_cost_le k M m x h j ω sl
      _ = Ex fun p : ((Fin n → Coin) × C) × ((Fin n → Coin) × C) × (Fin M → (Fin n → Coin) × C) =>
            (linkCost (fun v => m.acc ω (forkOf m (slotRuns (Fin.cons p.1 (Fin.cons p.2.1 p.2.2 :
                Fin (M + 1) → (Fin n → Coin) × C) : Fin (M + 1 + 1) → _)) j.val v))
              (fun v => m.cost ω (forkOf m (slotRuns (Fin.cons p.1 (Fin.cons p.2.1 p.2.2 :
                Fin (M + 1) → (Fin n → Coin) × C) : Fin (M + 1 + 1) → _)) j.val v)) M
              (slotRuns (Fin.cons p.1 (Fin.cons p.2.1 p.2.2 : Fin (M + 1) → (Fin n → Coin) × C) :
                  Fin (M + 1 + 1) → _) 1,
                fun t : Fin M => slotRuns (Fin.cons p.1 (Fin.cons p.2.1 p.2.2 :
                  Fin (M + 1) → (Fin n → Coin) × C) : Fin (M + 1 + 1) → _) (2 + t.val)) : ℝ) :=
          (unifMap_slots (Coin := Coin) (C := C) M (fun sl : Fin (M + 1 + 1) → (Fin n → Coin) × C =>
            (linkCost (fun v => m.acc ω (forkOf m (slotRuns sl) j.val v))
              (fun v => m.cost ω (forkOf m (slotRuns sl) j.val v)) M
              (slotRuns sl 1, fun t : Fin M => slotRuns sl (2 + t.val)) : ℝ))).symm
      _ = Ex fun a : (Fin n → Coin) × C => Ex fun p : ((Fin n → Coin) × C) × (Fin M → (Fin n → Coin) × C) =>
            (linkCost (fun v => m.acc ω (fork (m.cut j.val) a.1 v.1, v.2))
              (fun v => m.cost ω (fork (m.cut j.val) a.1 v.1, v.2)) M p : ℝ) := by
          refine (Claims.Ex_prod _).trans (congrArg Ex (funext fun a => congrArg Ex (funext fun p => ?_)))
          obtain ⟨h0, h1, h2⟩ := slotRuns_cons a p.1 p.2
          simp only [forkOf, h0, h1, h2]
      _ ≤ Ex fun a : (Fin n → Coin) × C =>
            2 * Ex fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j.val) a.1 v.1, v.2) : ℝ) :=
          Ex_mono fun a => Ex_linkCost_le _ _ M
      _ = 2 * Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) := by
          rw [Claims.Ex_mul_left]
          congr 1
          calc Ex (fun a : (Fin n → Coin) × C =>
                Ex fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j.val) a.1 v.1, v.2) : ℝ))
              = Ex fun r : Fin n → Coin =>
                  Ex fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j.val) r v.1, v.2) : ℝ) :=
                (Claims.Ex_prod _).trans (congrArg Ex (funext fun r => by simp only [Ex_const]))
            _ = _ := (Claims.Ex_prod (fun q : (Fin n → Coin) × ((Fin n → Coin) × C) =>
                  (m.cost ω (fork (m.cut j.val) q.1 q.2.1, q.2.2) : ℝ))).symm.trans
                (unifMap_forkPair (m.cut j.val) (fun u => (m.cost ω u : ℝ)))
  calc Ex (fun s : ((linkPlan (Coin := Coin) k M).finder m x h).S =>
        (((linkPlan (Coin := Coin) k M).finder m x h).cost s : ℝ))
      = Ex fun j : Fin (k x) => Ex fun ω : Ω => Ex fun sl : Fin (M + 1 + 1) → (Fin n → Coin) × C =>
          (((linkPlan (Coin := Coin) k M).finder m x h).cost (j, ω, sl) : ℝ) :=
        (Claims.Ex_prod (fun s : Fin (k x) × Ω × (Fin (M + 1 + 1) → (Fin n → Coin) × C) =>
          (((linkPlan (Coin := Coin) k M).finder m x h).cost s : ℝ))).trans
          (congrArg Ex (funext fun j => Claims.Ex_prod _))
    _ ≤ Ex fun _ : Fin (k x) => Ex fun ω : Ω => 2 * Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) :=
        Ex_mono fun j => Ex_mono fun ω => hω j ω
    _ = 2 * Ex (fun t : Ω × ((Fin n → Coin) × C) => (m.cost t.1 t.2 : ℝ)) := by
        have : Nonempty (Fin (k x)) := ⟨⟨0, h⟩⟩
        rw [Ex_const, Claims.Ex_mul_left, Claims.Ex_prod (fun t : Ω × ((Fin n → Coin) × C) => (m.cost t.1 t.2 : ℝ))]

/-- **A run finder under `ExpectedCR`** finds a collision with probability at most its expected cost over `2^256`. -/
theorem RunFinder.PrAt_le_ecr (H : Msg → Msg) (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg)
    (hcr : ∀ h : 0 < F.sel x, ExpectedCR H (F.finder m x h)) :
    F.PrAt H m x ≤ F.ExCost m x / 2 ^ (256 : ℝ) := by
  unfold RunFinder.PrAt RunFinder.ExCost
  by_cases h : 0 < F.sel x
  · rw [dif_pos h, dif_pos h]; exact hcr h
  · rw [dif_neg h, dif_neg h, zero_div]

/-- **The link under `ExpectedCR`** finds a collision with probability at most `2·E[a run's cost]/2^256`. -/
theorem linkPlan_PrAt_le_ecr (H : Msg → Msg) (k : Msg → ℕ) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg)
    (hcr : ∀ h : 0 < (linkPlan (Coin := Coin) k M).sel x, ExpectedCR H ((linkPlan (Coin := Coin) k M).finder m x h)) :
    (linkPlan (Coin := Coin) k M).PrAt H m x ≤
      2 * Ex (fun t : Ω × ((Fin n → Coin) × C) => (m.cost t.1 t.2 : ℝ)) / 2 ^ (256 : ℝ) :=
  (RunFinder.PrAt_le_ecr H _ m x hcr).trans
    (div_le_div_of_nonneg_right (linkPlan_ExCost_le k M m x) (by positivity))

end Finder

end Claims.RecGeneric
