module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.RecGeneric.LinkBits
public import Proofs.Flock.Claims.Recursion.BitsLin2
public import Proofs.Flock.Claims.Recursion.PlurLin

/-!
# `KSLinkOn`'s bits at the Flock instance

`KSLinkOn`'s bound at `M = 2^256` and `R = 2^257` is at most `κ + t·ε`, `ε` route 2C's five per-call terms
(`Claims.Recursion.lamLin2`'s after `κ`, each as `2^{−T}`), with the gated link's term in place of the deviation
finder's (`ksLinkOn_bits`). Where every statement a run accepts has `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}`, it is
at most `2^{−167.7} + t·2^{−174.95}` (`ksLinkOn_tight`); where no run accepts, `Pr[acc ∧ ¬R] = 0`.

* The link's plan names at most two runs in expectation (`linkPlanMadeTwo`, Wald's bound): its first fork, and where
  that one is accepted, the loop's forks up to the first accepted, which `Claims.Recursion.Ex_linkCost_le` holds to
  one more on average. The `M + 1` of `linkPlanMade` would leave the link's term near `β₂·c` at `M = 2^256`; at two it
  is `β₂·(2c + 1)/2^256 ≤ t·β₂·c_S/2^255` where `2c ≤ t·c_S` (`link_term_le`).
* The other four terms are as `Claims.Recursion.lin2_bound_le` charges them (`lin2_four_le`), and
  `link_bound_split_le` adds the five.

The premises are SHA-512's named collision assumptions (`StrictBirthday` at `B`; `ExpectedCR` at `B_I`, `FCI` and
`F0I` charged as route 2C charges them, and at the link's finder at the reruns, `verModel`), the prover's calls
`t ≥ 1` per run, and the reruns' cost: every rerun of `E_O`'s model, and every fork of one at each first run, costs
at most `c` calls on average, with `2c ≤ t·c_S`, `c_S = c_B + |S(x_I)| + 2(n + v)`.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims
open Claims.Recursion (fork loopCost linkCost)

variable {Coin : Type}

/-- **A deduplicated list costs at most the list it repeats**: where every run of `B` is one of `A`'s, the runs of
`A ++ B`, each counted once, cost at most `A`'s. -/
theorem dedup_append_sum_le {α : Type} [DecidableEq α] (f : α → ℕ) (A B : List α) (hB : ∀ e ∈ B, e ∈ A) :
    (((A ++ B).dedup).map f).sum ≤ (A.map f).sum := by
  rw [← List.sum_toFinset f (List.nodup_dedup _), Finset.sum_list_map_count]
  calc ∑ e ∈ (A ++ B).dedup.toFinset, f e ≤ ∑ e ∈ A.toFinset, f e :=
        Finset.sum_le_sum_of_subset fun e he => by
          rcases List.mem_append.1 (List.mem_dedup.1 (List.mem_toFinset.1 he)) with h | h
          · exact List.mem_toFinset.2 h
          · exact List.mem_toFinset.2 (hB e h)
    _ ≤ ∑ e ∈ A.toFinset, A.count e • f e :=
        Finset.sum_le_sum fun e he => by
          rw [smul_eq_mul]
          exact Nat.le_mul_of_pos_left _ (List.count_pos_iff.2 (List.mem_toFinset.1 he))

section Loop

variable {Ω C : Type} {n : ℕ}

/-- **The link's loop asks only forks of `u₀`, looks at nothing, and searches `first` and what it asked**; the runs
it asks cost `loopCost` over the slots it reads, the loop stopping at the first accepted fork. -/
theorem linkLoop_cost (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C) (j : ℕ) (first : RunExp)
    (u₀ : (Fin n → Coin) × C) (hu₀ : us 0 = u₀) :
    ∀ (N i : ℕ) (vs : Fin N → (Fin n → Coin) × C), (∀ t : Fin N, us (i + t.val) = vs t) →
      ((linkLoop (Coin := Coin) j first N i).exec m ω us).2.1 = [] ∧
      (∀ e ∈ ((linkLoop (Coin := Coin) j first N i).exec m ω us).2.2,
        e = first ∨ e ∈ ((linkLoop (Coin := Coin) j first N i).exec m ω us).1) ∧
      (((linkLoop (Coin := Coin) j first N i).exec m ω us).1.map fun e => m.cost ω (m.named us e)).sum =
        loopCost (fun v => m.acc ω (fork (m.cut j) u₀.1 v.1, v.2))
          (fun v => m.cost ω (fork (m.cut j) u₀.1 v.1, v.2)) N vs
  | 0, _, _, _ => by simp [linkLoop, Plan.exec, loopCost]
  | N + 1, i, vs, hvs => by
    classical
    have hv0 : us i = vs 0 := by simpa using hvs 0
    have hvs' : ∀ t : Fin N, us (i + 1 + t.val) = Fin.tail vs t := fun t =>
      (congrArg us (by simp only [Fin.val_succ]; omega)).trans (hvs t.succ)
    obtain ⟨h1, h2, h3⟩ := linkLoop_cost m ω us j first u₀ hu₀ N (i + 1) (Fin.tail vs) hvs'
    by_cases ha : m.acc ω (fork (m.cut j) u₀.1 (vs 0).1, (vs 0).2)
    · simp only [linkLoop, Plan.exec, RunModel.named, hu₀, hv0, decide_eq_true_eq, ha, ↓reduceIte, loopCost]
      simp [RunModel.named, hu₀, hv0]
    · simp only [linkLoop, Plan.exec, RunModel.named, hu₀, hv0, decide_eq_true_eq, ha, ↓reduceIte, loopCost]
      refine ⟨h1, fun e he => ?_, ?_⟩
      · rcases h2 e he with h | h
        · exact Or.inl h
        · exact Or.inr (List.mem_cons_of_mem _ h)
      · rw [List.map_cons, List.sum_cons, h3]
        simp only [RunModel.named, hu₀, hv0]

end Loop

section Wald

variable [Fintype Coin] [Nonempty Coin]

/-- **`linkPlan`'s cost at a round guess, tape and slots is at most `linkCost`**: the first fork's, and where that
fork is accepted, the loop's forks up to the first accepted, each named once. -/
theorem linkPlan_cost_le_linkCost {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ}
    (k : Msg → ℕ) (M : ℕ) (m : RunModel Coin Ω n C) (x : Msg) (h : 0 < (linkPlan (Coin := Coin) k M).sel x)
    (j : Fin (k x)) (ω : Ω) (us : Fin (M + 1 + 1) → (Fin n → Coin) × C) :
    ((linkPlan (Coin := Coin) k M).finder m x h).cost (j, ω, us) ≤
      linkCost (fun v => m.acc ω (fork (m.cut j) (us 0).1 v.1, v.2))
        (fun v => m.cost ω (fork (m.cut j) (us 0).1 v.1, v.2)) M
        (us (0 : Fin (M + 1)).succ, fun t : Fin M => us t.succ.succ) := by
  classical
  have h0 : slotRuns us 0 = us 0 := congrArg us (Fin.ext (by simp))
  have h1 : slotRuns us 1 = us (0 : Fin (M + 1)).succ := congrArg us (Fin.ext (by simp))
  have hs : ∀ t : Fin M, slotRuns us (2 + t.val) = us t.succ.succ := fun t =>
    congrArg us (Fin.ext (by simp only [Fin.val_succ]; omega))
  obtain ⟨hl1, hl2, hl3⟩ := linkLoop_cost m ω (slotRuns us) j.val (.fork j.val (.fresh 0) (.fresh 1)) (us 0) h0 M 2
    (fun t => us t.succ.succ) hs
  have hfirst : m.named (slotRuns us) (.fork j.val (.fresh 0) (.fresh 1)) =
      (fork (m.cut j) (us 0).1 (us (0 : Fin (M + 1)).succ).1, (us (0 : Fin (M + 1)).succ).2) := by
    simp only [RunModel.named, h0, h1]
  have hcost : ((linkPlan (Coin := Coin) k M).finder m x h).cost (j, ω, us) =
      (((((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us)).1 ++
        (((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us)).2.1 ++
        (((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us)).2.2).dedup.map
          fun e => m.cost ω (m.named (slotRuns us) e)).sum := rfl
  by_cases ha : m.acc ω (fork (m.cut j) (us 0).1 (us (0 : Fin (M + 1)).succ).1, (us (0 : Fin (M + 1)).succ).2)
  · have hex : ((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us) =
        (.fork j.val (.fresh 0) (.fresh 1) ::
          ((linkLoop (Coin := Coin) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec m ω (slotRuns us)).1,
          ((linkLoop (Coin := Coin) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec m ω (slotRuns us)).2.1,
          ((linkLoop (Coin := Coin) j.val (.fork j.val (.fresh 0) (.fresh 1)) M 2).exec m ω (slotRuns us)).2.2) := by
      simp only [linkPlan, Plan.exec, hfirst, decide_eq_true_eq, ha, ↓reduceIte]
    rw [hcost, hex, hl1, List.append_nil]
    refine (dedup_append_sum_le _ _ _ fun e he => ?_).trans (le_of_eq ?_)
    · rcases hl2 e he with h | h
      · exact h ▸ List.mem_cons_self
      · exact List.mem_cons_of_mem _ h
    · rw [List.map_cons, List.sum_cons, hl3, hfirst]
      unfold linkCost
      simp only [ha, ↓reduceIte]
  · have hex : ((linkPlan (Coin := Coin) k M).plan x j).exec m ω (slotRuns us) =
        ([.fork j.val (.fresh 0) (.fresh 1)], [], []) := by
      simp only [linkPlan, Plan.exec, hfirst, decide_eq_true_eq, ha, ↓reduceIte]
    rw [hcost, hex]
    unfold linkCost
    simp [hfirst]

/-- **The link's plan names at most two runs in expectation, each of cost at most `c`** (Wald's bound): its first
fork, and where that one is accepted, the loop's forks up to the first accepted, which `Ex_linkCost_le` holds to one
more fork on average. Every run it names is a fork of the shared first run with a fresh uniform slot. -/
theorem linkPlanMadeTwo (k : Msg → ℕ) (M : ℕ) : (linkPlan (Coin := Coin) k M).MadeLe fun _ => 2 := by
  intro Ω _ _ n C _ _ m x c hfork hrun
  have hc : 0 ≤ c := (Ex_nonneg fun _ => Nat.cast_nonneg _).trans (hrun (Classical.arbitrary Ω))
  unfold RunFinder.ExCost
  split_ifs with h
  · have : Nonempty (Fin (k x)) := ⟨⟨0, h⟩⟩
    let F : Fin (k x) → Ω → (Fin n → Coin) × C → (Fin (M + 1) → (Fin n → Coin) × C) → ℝ := fun j ω u₀ σ =>
      (linkCost (fun v => m.acc ω (fork (m.cut j) u₀.1 v.1, v.2))
        (fun v => m.cost ω (fork (m.cut j) u₀.1 v.1, v.2)) M (σ 0, fun t => σ t.succ) : ℝ)
    let G : Fin (k x) → Ω → (Fin (M + 1 + 1) → (Fin n → Coin) × C) → ℝ := fun j ω us =>
      F j ω (us 0) fun i => us i.succ
    have h1 : Ex (fun s : Fin (k x) × Ω × (Fin (M + 1 + 1) → (Fin n → Coin) × C) =>
        (((linkPlan (Coin := Coin) k M).finder m x h).cost s : ℝ)) ≤
        Ex (fun s : Fin (k x) × Ω × (Fin (M + 1 + 1) → (Fin n → Coin) × C) => G s.1 s.2.1 s.2.2) :=
      Ex_mono fun s => by
        obtain ⟨j, ω, us⟩ := s
        have := linkPlan_cost_le_linkCost k M m x h j ω us
        simp only [G, F]
        exact_mod_cast this
    refine h1.trans (Ex_sel_le G _ fun j ω => ?_)
    show Ex (fun us : Fin (M + 1 + 1) → (Fin n → Coin) × C => F j ω (us 0) fun i => us i.succ) ≤ _
    rw [Claims.Ex_cons (M + 1) (F j ω)]
    refine (Ex_mono fun u₀ => ?_).trans_eq (Ex_const _)
    show Ex (fun σ : Fin (M + 1) → (Fin n → Coin) × C => (linkCost
        (fun v => m.acc ω (fork (m.cut j) u₀.1 v.1, v.2)) (fun v => m.cost ω (fork (m.cut j) u₀.1 v.1, v.2)) M
        (σ 0, fun t => σ t.succ) : ℝ)) ≤ _
    rw [Claims.Ex_cons M (fun (u : (Fin n → Coin) × C) (rest : Fin M → (Fin n → Coin) × C) =>
        (linkCost (fun v => m.acc ω (fork (m.cut j) u₀.1 v.1, v.2))
          (fun v => m.cost ω (fork (m.cut j) u₀.1 v.1, v.2)) M (u, rest) : ℝ)),
      ← Claims.Ex_prod (fun p : ((Fin n → Coin) × C) × (Fin M → (Fin n → Coin) × C) =>
        (linkCost (fun v : (Fin n → Coin) × C => m.acc ω (fork (m.cut j) u₀.1 v.1, v.2))
          (fun v : (Fin n → Coin) × C => m.cost ω (fork (m.cut j) u₀.1 v.1, v.2)) M p : ℝ))]
    have := Claims.Recursion.Ex_linkCost_le (fun v : (Fin n → Coin) × C => m.acc ω (fork (m.cut j) u₀.1 v.1, v.2))
      (fun v : (Fin n → Coin) × C => m.cost ω (fork (m.cut j) u₀.1 v.1, v.2)) M
    have hf := hfork ω j u₀
    push_cast
    linarith
  · push_cast
    linarith

end Wald

section Bits

open Claims.Recursion (βdev2 cBe setupLen xI PrBinnerV PrFCI PrF0I BinnerV FCI F0I SessionAccepts Caps Deploy
  le_two_rpow_of_div)

variable [Fintype Coin] [Nonempty Coin]

/-- **The gated link's term at `M = 2^256` is at most `t·2^{−(255 − log₂(β·c_S))}`**, in a model whose runs, and
forks of them, cost at most `c` calls on average, at `2c ≤ t·c_S` and `1 ≤ t·c_S`, where the link's finder meets
`ExpectedCR`: it names at most two runs in expectation (`linkPlanMadeTwo`), so it finds a collision with probability
at most `2c/2^256 ≤ t·c_S/2^256`, and its tail `1/(2^256 + 1)` is below `t·c_S/2^256`. -/
theorem link_term_le {Ω C : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] {n : ℕ} (H : Msg → Msg)
    (k : Msg → ℕ) (m : RunModel Coin Ω n C) (y : Msg) {β c cS t : ℝ} (hβ : 0 ≤ β) (ht : 0 ≤ t)
    (hfork : ∀ (ω : Ω) (j : ℕ) (u₀ : (Fin n → Coin) × C),
      Ex (fun v : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j) u₀.1 v.1, v.2) : ℝ)) ≤ c)
    (hrun : ∀ ω, Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) ≤ c) (hc : 2 * c ≤ t * cS)
    (htc : 1 ≤ t * cS)
    (hCR : ∀ h : 0 < (linkPlan (Coin := Coin) k (2 ^ 256)).sel y,
      ExpectedCR H ((linkPlan (Coin := Coin) k (2 ^ 256)).finder m y h)) :
    β * ((linkPlan (Coin := Coin) k (2 ^ 256)).PrAt H m y + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) ≤
      t * (2 : ℝ) ^ (-(255 - Real.logb 2 (β * cS))) := by
  have hp256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have h256 : (2 : ℝ) ^ (256 : ℝ) = 2 * 2 ^ (255 : ℝ) := by
    rw [show (256 : ℝ) = 1 + 255 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have hMc : (((2 ^ 256 : ℕ) : ℝ)) = (2 : ℝ) ^ (256 : ℝ) := by
    rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hM : 1 / (((2 ^ 256 : ℕ) : ℝ) + 1) ≤ t * cS / 2 ^ (256 : ℝ) := by
    rw [hMc]
    calc 1 / ((2 : ℝ) ^ (256 : ℝ) + 1) ≤ 1 / (2 : ℝ) ^ (256 : ℝ) :=
          one_div_le_one_div_of_le hp256 (by linarith)
      _ ≤ t * cS / 2 ^ (256 : ℝ) := div_le_div_of_nonneg_right htc hp256.le
  have hP : (linkPlan (Coin := Coin) k (2 ^ 256)).PrAt H m y ≤ t * cS / 2 ^ (256 : ℝ) := by
    refine (RunFinder.PrAt_le_madeLe H _ _ (linkPlanMadeTwo k (2 ^ 256)) m y c hfork hrun hCR).trans
      (div_le_div_of_nonneg_right ?_ hp256.le)
    push_cast
    linarith
  have hcS : 0 ≤ cS := by nlinarith
  calc β * ((linkPlan (Coin := Coin) k (2 ^ 256)).PrAt H m y + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1))
      ≤ β * (t * cS / 2 ^ (256 : ℝ) + t * cS / 2 ^ (256 : ℝ)) :=
        mul_le_mul_of_nonneg_left (add_le_add hP hM) hβ
    _ = t * (β * cS) / 2 ^ (255 : ℝ) := by rw [h256]; ring
    _ ≤ _ := le_two_rpow_of_div ht (mul_nonneg hβ hcS)

variable [Inhabited Coin]

/-- **Route 2C's terms after `κ` other than the deviation finder's, at `M = 2^256` and `R = 2^257`, are each at most
`t·2^{−T}`**, `T` the term's bits in `lamLin2`: the proofs of `Claims.Recursion.lin2_bound_le`'s first four, which read
neither `Fdev` nor its `ExpectedCR`. -/
theorem lin2_four_le {W Col T : Type} [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin)
    (I : Claims.Recursion.Inner Coin W) (DO : Claims.Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T)
    (Cp : Caps Coin) (hS : SessionAccepts D) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ)
    (ht1 : 1 ≤ t) (hrun : ∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ t)
    (n₀ v : ℕ) (hSB : ∀ hk : 0 < D.V.k x, StrictBirthday H (B H D.V P x hk))
    (hBI : ∀ hk : 0 < I.V.k (xI D x),
      ExpectedCR H ((BinnerV H D I DO decO P x hk).charge (setupLen D x + 2 * (n₀ + v))))
    (hFCI : ∀ (M : ℕ) (hW : 0 < Cp.width (xI D x)),
      ExpectedCR H ((FCI H D I DO decO Cp P x M hW).charge (setupLen D x + 2 * (n₀ + v))))
    (hF0I : ∀ R : ℕ,
      ExpectedCR H ((F0I H D I DO decO P x R).charge
        (setupLen D x + (I.k (xI D x) + 1) * (n₀ + v)))) :
    Real.sqrt (Claims.Recursion.γ D I x * PrB H D.V P x) ≤
        t * (2 : ℝ) ^ (-(255.5 - Real.logb 2 (Claims.Recursion.γ D I x) / 2)) ∧
      (4 * I.V.k (xI D x) + 1) * PrBinnerV H D I DO decO P x ≤
        t * (2 : ℝ) ^ (-(256 - Real.logb 2 ((4 * I.V.k (xI D x) + 1) *
          (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ))))) ∧
      4 * Cp.width (xI D x) * (PrFCI H D I DO decO Cp P x (2 ^ 256) + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) ≤
        t * (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * Cp.width (xI D x) *
          (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ))))) ∧
      4 * (PrF0I H D I DO decO P x (2 ^ 257) + ((I.k (xI D x) : ℝ) + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1)) ≤
        t * (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * ((I.k (xI D x) : ℝ) + 1) *
          (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ))))) := by
  generalize n₀ + v = n at hBI hFCI hF0I ⊢
  have ht1' : (1 : ℝ) ≤ t := by exact_mod_cast ht1
  have ht0 : (0 : ℝ) ≤ t := by linarith
  have hc4 : (4 : ℝ) ≤ cBe D x := by
    have : 4 ≤ cBe D x := by unfold cBe; omega
    exact_mod_cast this
  have hnn : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  have hn0 : (0 : ℝ) ≤ setupLen D x + 2 * n := by positivity
  have htcB : (1 : ℝ) ≤ t * cBe D x := by nlinarith
  have htc : (1 : ℝ) ≤ t * (cBe D x + setupLen D x + 2 * n) := by nlinarith
  have htn : (setupLen D x : ℝ) + 2 * n ≤ t * (setupLen D x + 2 * n) := le_mul_of_one_le_left hn0 ht1'
  have hd : ∀ ω, Ex (fun d : (Fin (D.V.k x) → Coin) × (Claims.Recursion.EO D I DO decO).C x =>
      (Claims.Recursion.drawCost H D I DO decO P x ω d : ℝ)) ≤ t * cBe D x / 2 :=
    Claims.Recursion.Ex_drawCost_le H D I DO decO hS x P t hrun
  have hp256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have h256 : (2 : ℝ) ^ (256 : ℝ) = 2 * 2 ^ (255 : ℝ) := by
    rw [show (256 : ℝ) = 1 + 255 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have h257 : (2 : ℝ) ^ (257 : ℝ) = 2 * 2 ^ (256 : ℝ) := by
    rw [show (257 : ℝ) = 1 + 256 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have hMc : (((2 ^ 256 : ℕ) : ℝ)) = (2 : ℝ) ^ (256 : ℝ) := by
    rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hRc : (((2 ^ 257 : ℕ) : ℝ)) = (2 : ℝ) ^ (257 : ℝ) := by
    rw [show (257 : ℝ) = ((257 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hM : 1 / (((2 ^ 256 : ℕ) : ℝ) + 1) ≤ t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := by
    rw [hMc]
    calc 1 / ((2 : ℝ) ^ (256 : ℝ) + 1) ≤ 1 / (2 : ℝ) ^ (256 : ℝ) :=
          one_div_le_one_div_of_le hp256 (by linarith)
      _ ≤ t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := div_le_div_of_nonneg_right htc hp256.le
  have hB : PrB H D.V P x ≤ (2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ) := by
    unfold PrB
    split
    · rename_i hk
      refine hSB hk (2 * t) fun s => ?_
      exact_mod_cast B_cost_le H D.V P x hk t hrun s
    · positivity
  have hγ0 : 0 ≤ Claims.Recursion.γ D I x := by unfold Claims.Recursion.γ; positivity
  refine ⟨?_, ?_, ?_, ?_⟩
  · calc Real.sqrt (Claims.Recursion.γ D I x * PrB H D.V P x)
        ≤ Real.sqrt (Claims.Recursion.γ D I x * ((2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ))) :=
          Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hB hγ0)
      _ = t * (Real.sqrt (Claims.Recursion.γ D I x) * 2 ^ (-(255.5 : ℝ))) :=
          Claims.Sampling.sqrt_birthday hγ0 ht0
      _ ≤ _ := mul_le_mul_of_nonneg_left (Claims.Recursion.sqrt_mul_two_rpow_le hγ0) ht0
  · have hP : PrBinnerV H D I DO decO P x ≤ t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := by
      unfold PrBinnerV
      split_ifs with hk
      · refine Claims.Recursion.ecr_charge_le (hBI hk) ?_
        have := Claims.Recursion.Ex_BinnerV_cost_le H D I DO decO x P hk hd
        push_cast
        linarith
      · positivity
    calc (4 * (I.V.k (xI D x) : ℝ) + 1) * PrBinnerV H D I DO decO P x
        ≤ (4 * (I.V.k (xI D x) : ℝ) + 1) * (t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ)) :=
          mul_le_mul_of_nonneg_left hP (by positivity)
      _ = t * ((4 * (I.V.k (xI D x) : ℝ) + 1) * (cBe D x + setupLen D x + 2 * n)) / 2 ^ (256 : ℝ) := by ring
      _ ≤ _ := le_two_rpow_of_div ht0 (by positivity)
  · have hP : PrFCI H D I DO decO Cp P x (2 ^ 256) ≤
        t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) := by
      unfold PrFCI
      split_ifs with hW
      · refine Claims.Recursion.ecr_charge_le (hFCI (2 ^ 256) hW) ?_
        have := Claims.Recursion.Ex_FCI_cost_le H D I DO decO Cp x P (2 ^ 256) hW hd
        linarith [show ((setupLen D x + 2 * n : ℕ) : ℝ) = setupLen D x + 2 * n by
          simp only [Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]]
      · positivity
    calc 4 * (Cp.width (xI D x) : ℝ) *
          (PrFCI H D I DO decO Cp P x (2 ^ 256) + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1))
        ≤ 4 * (Cp.width (xI D x) : ℝ) * (t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ) +
            t * (cBe D x + setupLen D x + 2 * n) / 2 ^ (256 : ℝ)) :=
          mul_le_mul_of_nonneg_left (add_le_add hP hM) (by positivity)
      _ = t * (4 * (Cp.width (xI D x) : ℝ) * (cBe D x + setupLen D x + 2 * n)) / 2 ^ (255 : ℝ) := by
          rw [h256]; ring
      _ ≤ _ := le_two_rpow_of_div ht0 (by positivity)
  · have hP : PrF0I H D I DO decO P x (2 ^ 257) ≤
        (((I.k (xI D x) : ℝ) + 1) * (t * cBe D x / 2) + (setupLen D x + ((I.k (xI D x) : ℝ) + 1) * n)) /
          2 ^ (256 : ℝ) := by
      refine Claims.Recursion.ecr_charge_le (hF0I (2 ^ 257)) ?_
      have : Ex (fun s => ((F0I H D I DO decO P x (2 ^ 257)).cost s : ℝ)) ≤
          ((I.k (xI D x) : ℝ) + 1) * (t * cBe D x / 2) :=
        Claims.Recursion.Ex_F0I_cost_le H D I DO decO x P (2 ^ 257) hd
      push_cast
      linarith
    have hk0 : (0 : ℝ) ≤ (I.k (xI D x) : ℝ) + 1 := by positivity
    have hk1 : (1 : ℝ) ≤ (I.k (xI D x) : ℝ) + 1 := by
      have : (0 : ℝ) ≤ I.k (xI D x) := Nat.cast_nonneg _
      linarith
    have hs1 : (setupLen D x : ℝ) ≤ ((I.k (xI D x) : ℝ) + 1) * (t * setupLen D x) :=
      (le_mul_of_one_le_left (Nat.cast_nonneg _) ht1').trans (le_mul_of_one_le_left (by positivity) hk1)
    have hn2 : ((I.k (xI D x) : ℝ) + 1) * n ≤ ((I.k (xI D x) : ℝ) + 1) * (t * (2 * n)) :=
      mul_le_mul_of_nonneg_left (by nlinarith [mul_nonneg (sub_nonneg.mpr ht1') hnn]) hk0
    have hR : ((I.k (xI D x) : ℝ) + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1) ≤
        ((I.k (xI D x) : ℝ) + 1) * (t * cBe D x) / 2 ^ (257 : ℝ) := by
      rw [hRc]
      calc ((I.k (xI D x) : ℝ) + 1) / ((2 : ℝ) ^ (257 : ℝ) + 1) ≤ ((I.k (xI D x) : ℝ) + 1) / (2 : ℝ) ^ (257 : ℝ) :=
            div_le_div_of_nonneg_left hk0 (by positivity) (by linarith)
        _ ≤ _ := div_le_div_of_nonneg_right (le_mul_of_one_le_right hk0 htcB) (by positivity)
    calc 4 * (PrF0I H D I DO decO P x (2 ^ 257) +
          ((I.k (xI D x) : ℝ) + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1))
        ≤ 4 * ((((I.k (xI D x) : ℝ) + 1) * (t * cBe D x / 2) + (setupLen D x + ((I.k (xI D x) : ℝ) + 1) * n)) /
              2 ^ (256 : ℝ) + ((I.k (xI D x) : ℝ) + 1) * (t * cBe D x) / 2 ^ (257 : ℝ)) :=
          mul_le_mul_of_nonneg_left (add_le_add hP hR) (by norm_num)
      _ = (t * (4 * ((I.k (xI D x) : ℝ) + 1) * cBe D x) + 4 * (setupLen D x + ((I.k (xI D x) : ℝ) + 1) * n)) /
            2 ^ (256 : ℝ) := by
          rw [h257]; ring
      _ ≤ t * (4 * ((I.k (xI D x) : ℝ) + 1) * (cBe D x + setupLen D x + 2 * n)) / 2 ^ (256 : ℝ) := by
          refine div_le_div_of_nonneg_right ?_ hp256.le
          nlinarith
      _ ≤ _ := le_two_rpow_of_div ht0 (by positivity)

/-- **Route 2C's bound with the gated link, at `M = 2^256` and `R = 2^257`, is at most `κ + t·ε`**, `ε` route 2C's
five terms after `κ`, written out: the first four as route 2C charges them (`lin2_four_le`), and the gated link's in
place of the deviation finder's (`link_term_le`), in a model whose reruns, and forks of them, cost at most `c` calls
on average with `2c ≤ t·c_S`. -/
theorem link_bound_split_le {W Col T : Type} [DecidableEq Col] (H : Msg → Msg) (D : Deploy Coin)
    (I : Claims.Recursion.Inner Coin W) (DO : Claims.Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T)
    (Cp : Caps Coin) (hS : SessionAccepts D) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (t : ℕ)
    (ht1 : 1 ≤ t) (hrun : ∀ ω r, (run H P D.V x ω r).pLog.length + (run H P D.V x ω r).vLog.length ≤ t)
    (n₀ v : ℕ) (hSB : ∀ hk : 0 < D.V.k x, StrictBirthday H (B H D.V P x hk))
    (hBI : ∀ hk : 0 < I.V.k (xI D x),
      ExpectedCR H ((BinnerV H D I DO decO P x hk).charge (setupLen D x + 2 * (n₀ + v))))
    (hFCI : ∀ (M : ℕ) (hW : 0 < Cp.width (xI D x)),
      ExpectedCR H ((FCI H D I DO decO Cp P x M hW).charge (setupLen D x + 2 * (n₀ + v))))
    (hF0I : ∀ R : ℕ,
      ExpectedCR H ((F0I H D I DO decO P x R).charge
        (setupLen D x + (I.k (xI D x) + 1) * (n₀ + v))))
    {Ω' C : Type} [Fintype Ω'] [Nonempty Ω'] [Fintype C] [Nonempty C] {n : ℕ} (k : Msg → ℕ)
    (m : RunModel Coin Ω' n C) (y : Msg) (c : ℝ)
    (hfork : ∀ (ω : Ω') (j : ℕ) (u₀ : (Fin n → Coin) × C),
      Ex (fun u : (Fin n → Coin) × C => (m.cost ω (fork (m.cut j) u₀.1 u.1, u.2) : ℝ)) ≤ c)
    (hrr : ∀ ω, Ex (fun u : (Fin n → Coin) × C => (m.cost ω u : ℝ)) ≤ c)
    (hc : 2 * c ≤ t * (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ)))
    (hLink : ∀ h : 0 < (linkPlan (Coin := Coin) k (2 ^ 256)).sel y,
      ExpectedCR H ((linkPlan (Coin := Coin) k (2 ^ 256)).finder m y h)) :
    Claims.Recursion.κ D I x + Real.sqrt (Claims.Recursion.γ D I x * PrB H D.V P x) +
        (4 * I.V.k (xI D x) + 1) * PrBinnerV H D I DO decO P x +
        4 * Cp.width (xI D x) * (PrFCI H D I DO decO Cp P x (2 ^ 256) + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) +
        4 * (PrF0I H D I DO decO P x (2 ^ 257) + ((I.k (xI D x) : ℝ) + 1) / (((2 ^ 257 : ℕ) : ℝ) + 1)) +
        βdev2 D I Cp x * ((linkPlan (Coin := Coin) k (2 ^ 256)).PrAt H m y + 1 / (((2 ^ 256 : ℕ) : ℝ) + 1)) ≤
      Claims.Recursion.κ D I x + t * ((2 : ℝ) ^ (-(255.5 - Real.logb 2 (Claims.Recursion.γ D I x) / 2)) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 ((4 * I.V.k (xI D x) + 1) *
          (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ))))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * Cp.width (xI D x) * (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ))))) +
        (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * ((I.k (xI D x) : ℝ) + 1) *
          (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ))))) +
        (2 : ℝ) ^ (-(255 - Real.logb 2 (βdev2 D I Cp x * (cBe D x + setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ)))))) := by
  obtain ⟨h₁, h₂, h₃, h₄⟩ := lin2_four_le H D I DO decO Cp hS x P t ht1 hrun n₀ v hSB hBI hFCI hF0I
  have ht1' : (1 : ℝ) ≤ t := by exact_mod_cast ht1
  have hc4 : (4 : ℝ) ≤ cBe D x := by
    have : 4 ≤ cBe D x := by unfold cBe; omega
    exact_mod_cast this
  have hn0 : (0 : ℝ) ≤ setupLen D x + 2 * ((n₀ + v : ℕ) : ℝ) := by positivity
  have h₅ := link_term_le (Coin := Coin) H k m y (β := βdev2 D I Cp x) (by unfold βdev2; positivity)
    (by linarith) hfork hrr hc (by nlinarith) hLink
  linarith

end Bits

section Instance

open Claims.Recursion

variable [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {Col T : Type} [DecidableEq Col] {Col' T' : Type}
  [DecidableEq Col']

/-- **`KSLinkOn`'s bound at `M = 2^256` and `R = 2^257` is at most `κ + t·ε`** at a statement `G` admits, `ε` route
2C's five terms with the gated link's in place of the deviation finder's (`link_bound_split_le`), where every V\*
session's verifier accepts on an accepted run, under SHA-512's named collision assumptions at `B`, `B_I`, `FCI`,
`F0I` and the link's finder at the reruns, and where every rerun of `E_O`'s model, and every fork of one at each first
run, costs at most `c` calls on average, with `2c ≤ t·c_S`. -/
theorem ksLinkOn_bits [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Claims.Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} (hK : KSLinkOn H D Bd DO Cp G)
    (hS : SessionAccepts (D.record Bd.IVStar)) {x : Msg} (hx : G x) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (t : ℕ) (ht1 : 1 ≤ t)
    (hrun : ∀ ω r, (run H P (D.record Bd.IVStar).V x ω r).pLog.length +
      (run H P (D.record Bd.IVStar).V x ω r).vLog.length ≤ t) (n v : ℕ)
    (hSB : ∀ hk : 0 < (D.record Bd.IVStar).V.k x, StrictBirthday H (B H (D.record Bd.IVStar).V P x hk))
    (hBI : ∀ hk : 0 < (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k (xI (D.record Bd.IVStar) x),
      ExpectedCR H ((BinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x
        hk).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v))))
    (hFCI : ∀ (M : ℕ) (hW : 0 < Cp.width (xI (D.record Bd.IVStar) x)),
      ExpectedCR H ((FCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode Cp P x M
        hW).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v))))
    (hF0I : ∀ R : ℕ,
      ExpectedCR H ((F0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x
        R).charge (setupLen (D.record Bd.IVStar) x +
          ((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (xI (D.record Bd.IVStar) x) + 1) * (n + v))))
    (c : ℝ)
    (hfork : ∀ (ω : Ω) (j : ℕ) (u₀ : (Fin ((compDep D Bd DO).V.k x) → Coin) × (compDep D Bd DO).EO.C x),
      Ex (fun u : (Fin ((compDep D Bd DO).V.k x) → Coin) × (compDep D Bd DO).EO.C x =>
        (((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x).cost ω
          (fork (((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x).cut j) u₀.1 u.1,
            u.2) : ℝ)) ≤ c)
    (hrr : ∀ ω, Ex (fun u : (Fin ((compDep D Bd DO).V.k x) → Coin) × (compDep D Bd DO).EO.C x =>
      (((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x).cost ω u : ℝ)) ≤ c)
    (hc : 2 * c ≤ t * (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ)))
    (hLink : ∀ h : 0 < (linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).sel ((compDep D Bd DO).xI x),
      ExpectedCR H ((linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).finder
        ((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x) ((compDep D Bd DO).xI x) h)) :
    PrBad H (compDep D Bd DO).V ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
        ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤
      Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x +
        t * ((2 : ℝ) ^ (-(255.5 - Real.logb 2
              (Claims.Recursion.γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x) / 2)) +
          (2 : ℝ) ^ (-(256 - Real.logb 2 ((4 * (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k
              (xI (D.record Bd.IVStar) x) + 1) *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * Cp.width (xI (D.record Bd.IVStar) x) *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k
              (xI (D.record Bd.IVStar) x) : ℝ) + 1) *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ)))))) :=
  (hK x hx Ω P (2 ^ 256) (2 ^ 257)).trans
    (link_bound_split_le H _ _ DO DO.decode Cp hS x P t ht1 hrun n v hSB hBI hFCI hF0I Bd.DE.V.k _ _ c hfork hrr hc
      hLink)

/-- **`Pr[acc ∧ ¬R] ≤ 2^{−167.7} + t·2^{−174.95}` at every statement**, where every statement a run of the
composite's verifier accepts is one `G` admits with `κ ≤ 2^{−167.7}` and `ε ≤ 2^{−174.95}` (`hb`): `ksLinkOn_bits`
there, and where no run accepts, `Pr[acc ∧ ¬R] = 0`. -/
theorem ksLinkOn_tight [Zero Col] [Zero Col'] {H : Msg → Msg} {D : RecAt Coin} {Bd : Bound Coin Col T}
    {DO : Claims.Flock.Deploy Coin Col' T'} {Cp : Caps Coin} {G : Msg → Prop} (hK : KSLinkOn H D Bd DO Cp G)
    (hS : SessionAccepts (D.record Bd.IVStar)) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (t : ℕ) (ht1 : 1 ≤ t)
    (hrun : ∀ ω r, (run H P (D.record Bd.IVStar).V x ω r).pLog.length +
      (run H P (D.record Bd.IVStar).V x ω r).vLog.length ≤ t) (n v : ℕ)
    (hb : (∃ ω r, (run H P (compDep D Bd DO).V x ω r).out = .acc) → G x ∧
      Claims.Recursion.κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x ≤ 2 ^ (-(167.7 : ℝ)) ∧
      (2 : ℝ) ^ (-(255.5 - Real.logb 2
              (Claims.Recursion.γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) x) / 2)) +
          (2 : ℝ) ^ (-(256 - Real.logb 2 ((4 * (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k
              (xI (D.record Bd.IVStar) x) + 1) *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (4 * Cp.width (xI (D.record Bd.IVStar) x) *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(256 - Real.logb 2 (4 * (((Inner.ofFlock Bd.deploy Bd.deploy.decode).k
              (xI (D.record Bd.IVStar) x) : ℝ) + 1) *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ))))) +
          (2 : ℝ) ^ (-(255 - Real.logb 2 (βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) Cp x *
            (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ))))) ≤
        2 ^ (-(174.95 : ℝ)))
    (hSB : ∀ hk : 0 < (D.record Bd.IVStar).V.k x, StrictBirthday H (B H (D.record Bd.IVStar).V P x hk))
    (hBI : ∀ hk : 0 < (Inner.ofFlock Bd.deploy Bd.deploy.decode).V.k (xI (D.record Bd.IVStar) x),
      ExpectedCR H ((BinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x
        hk).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v))))
    (hFCI : ∀ (M : ℕ) (hW : 0 < Cp.width (xI (D.record Bd.IVStar) x)),
      ExpectedCR H ((FCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode Cp P x M
        hW).charge (setupLen (D.record Bd.IVStar) x + 2 * (n + v))))
    (hF0I : ∀ R : ℕ,
      ExpectedCR H ((F0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy Bd.deploy.decode) DO DO.decode P x
        R).charge (setupLen (D.record Bd.IVStar) x +
          ((Inner.ofFlock Bd.deploy Bd.deploy.decode).k (xI (D.record Bd.IVStar) x) + 1) * (n + v))))
    (c : ℝ)
    (hfork : ∀ (ω : Ω) (j : ℕ) (u₀ : (Fin ((compDep D Bd DO).V.k x) → Coin) × (compDep D Bd DO).EO.C x),
      Ex (fun u : (Fin ((compDep D Bd DO).V.k x) → Coin) × (compDep D Bd DO).EO.C x =>
        (((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x).cost ω
          (fork (((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x).cut j) u₀.1 u.1,
            u.2) : ℝ)) ≤ c)
    (hrr : ∀ ω, Ex (fun u : (Fin ((compDep D Bd DO).V.k x) → Coin) × (compDep D Bd DO).EO.C x =>
      (((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x).cost ω u : ℝ)) ≤ c)
    (hc : 2 * c ≤ t * (cBe (D.record Bd.IVStar) x + setupLen (D.record Bd.IVStar) x + 2 * ((n + v : ℕ) : ℝ)))
    (hLink : ∀ h : 0 < (linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).sel ((compDep D Bd DO).xI x),
      ExpectedCR H ((linkPlan (Coin := Coin) Bd.DE.V.k (2 ^ 256)).finder
        ((compDep D Bd DO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x) ((compDep D Bd DO).xI x) h)) :
    PrBad H (compDep D Bd DO).V ((compDep D Bd DO).R (Proto.ofFlock Bd Bd.deploy.decode Cp))
        ((compDep D Bd DO).E (Proto.ofFlock Bd Bd.deploy.decode Cp) (Server.ofBound Bd)) x P ≤
      2 ^ (-(167.7 : ℝ)) + t * 2 ^ (-(174.95 : ℝ)) := by
  have hpos : (0 : ℝ) ≤ 2 ^ (-(167.7 : ℝ)) + t * 2 ^ (-(174.95 : ℝ)) := by positivity
  by_cases hex : ∃ ω r, (run H P (compDep D Bd DO).V x ω r).out = .acc
  · obtain ⟨hx, hκ, hε⟩ := hb hex
    exact (ksLinkOn_bits hK hS hx P t ht1 hrun n v hSB hBI hFCI hF0I c hfork hrr hc hLink).trans
      (add_le_add hκ (mul_le_mul_of_nonneg_left hε (Nat.cast_nonneg t)))
  · unfold PrBad
    refine le_trans (Pr_mono (q := fun _ => False) fun q hq => ?_) (Pr_false.le.trans hpos)
    exact hex ⟨q.1, q.2.1, hq.1⟩

end Instance

end Claims.RecGeneric
