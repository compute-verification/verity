module

public import Proofs.Flock.RecGeneric.PlansProof

/-!
# Flock's plans at its own runs are its finders (F3)

At the inner protocol's runs as they may abort (`runModel`), each of Flock's three plans finds a collision exactly as
often as the finder it stands for:

* `forkPlan` is `B` (`forkPlanIsBAbort`, `ForkPlanIsBAbort`): it searches its first run and that run's fork at the
  guessed round;
* `capPlan` is `FC` (`capPlanIsFC`, `CapPlanIsFC`): where its first run is accepted and opens the cap position, it
  searches that run and the caps' loop's first fork that is accepted and opens it, or the run twice where none is
  (`capPlan_search`);
* `conflictPlan` is `F0` (`conflictPlanIsF0`, `ConflictPlanIsF0`): where its first run is accepted, it searches that run
  and each wait's first accepted run (`conflictPlan_search`), wait `t`'s run `i` being the plan's fresh run
  `1 + t·R + i` (`waitsEquiv`, `waitsEquiv_cons`).

Each equality is a bijection of coins (`bCoinsEquiv`, `capCoinsEquiv`, `conflictCoinsEquiv`), which keeps them uniform,
under which the plan's searched calls are the finder's, coin by coin.
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

section Searches

variable {Coin Ω C : Type} {n : ℕ}

/-- A fork of a run with itself is the run. -/
theorem fork_self {k : ℕ} (j : ℕ) (r : Fin k → Coin) : fork j r r = r :=
  funext fun i => by simp only [fork, ite_self]

theorem slotRuns_cons_zero {s : ℕ} (u₀ : (Fin n → Coin) × C) (us : Fin s → (Fin n → Coin) × C) :
    slotRuns (Fin.cons u₀ us : Fin (s + 1) → (Fin n → Coin) × C) 0 = u₀ :=
  (congrArg (Fin.cons u₀ us : Fin (s + 1) → (Fin n → Coin) × C) (Fin.ext (by simp))).trans (Fin.cons_zero _ _)

theorem slotRuns_cons_succ {s : ℕ} (u₀ : (Fin n → Coin) × C) (us : Fin s → (Fin n → Coin) × C) (t : ℕ)
    (ht : t < s) : slotRuns (Fin.cons u₀ us : Fin (s + 1) → (Fin n → Coin) × C) (t + 1) = us ⟨t, ht⟩ := by
  have h : (⟨min (t + 1) s, Nat.lt_succ_of_le (Nat.min_le_right (t + 1) s)⟩ : Fin (s + 1)) =
      (⟨t, ht⟩ : Fin s).succ :=
    Fin.ext (show min (t + 1) s = t + 1 by omega)
  show (Fin.cons u₀ us : Fin (s + 1) → (Fin n → Coin) × C) ⟨min (t + 1) s, _⟩ = us ⟨t, ht⟩
  rw [h, Fin.cons_succ]

theorem slotRuns_two_zero (us : Fin (1 + 1) → (Fin n → Coin) × C) : slotRuns us 0 = us 0 :=
  congrArg us (Fin.ext (by simp))

theorem slotRuns_two_one (us : Fin (1 + 1) → (Fin n → Coin) × C) : slotRuns us 1 = us 1 :=
  congrArg us (Fin.ext (by simp))

/-- **The caps' loop searches its first run and the first fork that is accepted and opens**, `firstPick`'s, or its
first run twice where none is. -/
theorem capLoop_exec (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C) (j : ℕ)
    (op : (ℕ → Coin) → Bool) (u₀ : (Fin n → Coin) × C) (hu₀ : us 0 = u₀) :
    ∀ (N i : ℕ) (vs : Fin N → (Fin n → Coin) × C), (∀ t : Fin N, us (i + t.val) = vs t) →
      ∃ e, ((capLoop j op N i).exec m ω us).2.2 = [.fresh 0, e] ∧
        m.named us e = (firstPick (fun v => m.acc ω (fork (m.cut j) u₀.1 v.1, v.2) ∧
          op (m.coins (fork (m.cut j) u₀.1 v.1, v.2)) = true) N vs).elim u₀
            fun v => (fork (m.cut j) u₀.1 v.1, v.2)
  | 0, _, _, _ => ⟨.fresh 0, rfl, hu₀⟩
  | N + 1, i, vs, hvs => by
    classical
    have hv0 : us i = vs 0 := by simpa using hvs 0
    have hvs' : ∀ t : Fin N, us (i + 1 + t.val) = Fin.tail vs t := fun t =>
      (congrArg us (by simp only [Fin.val_succ]; omega)).trans (hvs t.succ)
    obtain ⟨e', he', hn'⟩ := capLoop_exec m ω us j op u₀ hu₀ N (i + 1) (Fin.tail vs) hvs'
    by_cases ha : m.acc ω (fork (m.cut j) u₀.1 (vs 0).1, (vs 0).2)
    · by_cases ho : op (m.coins (fork (m.cut j) u₀.1 (vs 0).1, (vs 0).2)) = true
      · refine ⟨.fork j (.fresh 0) (.fresh i), ?_, ?_⟩
        · simp only [capLoop, Plan.exec, RunModel.named, hu₀, hv0, decide_eq_true_eq, ha, ho, ↓reduceIte]
        · simp only [firstPick, ha, ho, and_self, ↓reduceIte, Option.elim, RunModel.named, hu₀, hv0]
      · refine ⟨e', ?_, ?_⟩
        · simp only [capLoop, Plan.exec, RunModel.named, hu₀, hv0, decide_eq_true_eq, ha, ho, ↓reduceIte]
          exact he'
        · simp only [firstPick, ha, ho, ↓reduceIte]
          exact hn'
    · refine ⟨e', ?_, ?_⟩
      · simp only [capLoop, Plan.exec, RunModel.named, hu₀, hv0, decide_eq_true_eq, ha, ↓reduceIte]
        exact he'
      · simp only [firstPick, ha, false_and, ↓reduceIte]
        exact hn'

/-- **A wait goes on with its first accepted run**, `firstPick`'s, or with none. -/
theorem waitLoop_exec (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C)
    (cont : Option RunExp → Plan Coin) :
    ∀ (N i : ℕ) (vs : Fin N → (Fin n → Coin) × C), (∀ t : Fin N, us (i + t.val) = vs t) →
      ∃ o : Option RunExp, ((waitLoop cont N i).exec m ω us).2.2 = ((cont o).exec m ω us).2.2 ∧
        o.map (m.named us) = firstPick (m.acc ω) N vs
  | 0, _, _, _ => ⟨none, rfl, rfl⟩
  | N + 1, i, vs, hvs => by
    classical
    have hv0 : us i = vs 0 := by simpa using hvs 0
    have hvs' : ∀ t : Fin N, us (i + 1 + t.val) = Fin.tail vs t := fun t =>
      (congrArg us (by simp only [Fin.val_succ]; omega)).trans (hvs t.succ)
    by_cases ha : m.acc ω (vs 0)
    · refine ⟨some (.fresh i), ?_, ?_⟩
      · simp only [waitLoop, Plan.exec, RunModel.named, hv0, decide_eq_true_eq, ha, ↓reduceIte]
      · simp only [firstPick, ha, ↓reduceIte, Option.map, RunModel.named, hv0]
    · obtain ⟨o, ho, hn⟩ := waitLoop_exec m ω us cont N (i + 1) (Fin.tail vs) hvs'
      refine ⟨o, ?_, ?_⟩
      · simp only [waitLoop, Plan.exec, RunModel.named, hv0, decide_eq_true_eq, ha, ↓reduceIte]
        exact ho
      · simp only [firstPick, ha, ↓reduceIte]
        exact hn

/-- **`w` waits search what they found and each wait's first accepted run.** -/
theorem waitsPlan_exec (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C) (R : ℕ) :
    ∀ (w base : ℕ) (found : List RunExp) (ws : Fin w → Fin R → (Fin n → Coin) × C),
      (∀ (t : Fin w) (i : Fin R), us (base + t.val * R + i.val) = ws t i) →
      (((waitsPlan R w base found).exec m ω us).2.2.flatMap fun e => m.log ω (m.named us e)) =
        (found.flatMap fun e => m.log ω (m.named us e)) ++
          (List.finRange w).flatMap fun t => ((firstPick (m.acc ω) R (ws t)).map (m.log ω)).getD []
  | 0, base, found, ws, _ => by simp [waitsPlan, Plan.exec]
  | w + 1, base, found, ws, hws => by
    obtain ⟨o, ho, hn⟩ := waitLoop_exec m ω us (fun o => waitsPlan R w (base + R) (found ++ o.toList)) R base
      (ws 0) (fun i => by simpa using hws 0 i)
    have hws' : ∀ (t : Fin w) (i : Fin R), us (base + R + t.val * R + i.val) = ws t.succ i := fun t i =>
      (congrArg us (by simp only [Fin.val_succ]; ring)).trans (hws t.succ i)
    have ih := waitsPlan_exec m ω us R w (base + R) (found ++ o.toList) (fun t => ws t.succ) hws'
    have hB : (o.toList.flatMap fun e => m.log ω (m.named us e)) =
        ((firstPick (m.acc ω) R (ws 0)).map (m.log ω)).getD [] := by
      rw [← hn]; cases o <;> simp
    have hw : (waitsPlan R (w + 1) base found : Plan Coin) =
        waitLoop (fun o => waitsPlan R w (base + R) (found ++ o.toList)) R base := rfl
    rw [hw, ho, ih, List.flatMap_append, hB, List.finRange_succ, List.flatMap_cons, List.flatMap_map,
      List.append_assoc]

open Classical in
/-- **`FC`'s plan searches its first run and the caps' loop's**, where its first run is accepted and opens `q`. -/
theorem capPlan_search (Cp : Caps Coin) (M : ℕ) (x : Msg) (q : Fin (Cp.width x)) (m : RunModel Coin Ω n C) (ω : Ω)
    (u₀ : (Fin n → Coin) × C) (us : Fin M → (Fin n → Coin) × C) :
    ((((capPlan Cp M).plan x q).exec m ω (slotRuns (Fin.cons u₀ us : Fin (M + 1) → (Fin n → Coin) × C))).2.2.flatMap
        fun e => m.log ω (m.named (slotRuns (Fin.cons u₀ us : Fin (M + 1) → (Fin n → Coin) × C)) e)) =
      if m.acc ω u₀ ∧ Cp.opens x q (m.coins u₀) = true then
        m.log ω u₀ ++ m.log ω ((firstPick (fun v => m.acc ω (fork (m.cut (Cp.round x q)) u₀.1 v.1, v.2) ∧
          Cp.opens x q (m.coins (fork (m.cut (Cp.round x q)) u₀.1 v.1, v.2)) = true) M us).elim u₀
            fun v => (fork (m.cut (Cp.round x q)) u₀.1 v.1, v.2))
      else [] := by
  have hU0 := slotRuns_cons_zero u₀ us
  have hus : ∀ t : Fin M, slotRuns (Fin.cons u₀ us : Fin (M + 1) → (Fin n → Coin) × C) (1 + t.val) = us t :=
    fun t => by rw [Nat.add_comm 1 t.val]; exact slotRuns_cons_succ u₀ us t.val t.isLt
  by_cases ha : m.acc ω u₀
  · by_cases ho : Cp.opens x q (m.coins u₀) = true
    · obtain ⟨e, he, hn⟩ := capLoop_exec m ω _ (Cp.round x q) (Cp.opens x q) u₀ hU0 M 1 us hus
      have hexec : (((capPlan Cp M).plan x q).exec m ω
          (slotRuns (Fin.cons u₀ us : Fin (M + 1) → (Fin n → Coin) × C))).2.2 = [.fresh 0, e] := by
        simp only [capPlan, Plan.exec, RunModel.named, hU0, decide_eq_true_eq, ha, ho, ↓reduceIte]
        exact he
      rw [hexec, if_pos ⟨ha, ho⟩, ← hn]
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, RunModel.named, hU0]
    · have hexec : (((capPlan Cp M).plan x q).exec m ω
          (slotRuns (Fin.cons u₀ us : Fin (M + 1) → (Fin n → Coin) × C))).2.2 = [] := by
        simp only [capPlan, Plan.exec, RunModel.named, hU0, decide_eq_true_eq, ha, ho, Bool.false_eq_true,
          ↓reduceIte]
      rw [hexec, if_neg (fun h => ho h.2), List.flatMap_nil]
  · have hexec : (((capPlan Cp M).plan x q).exec m ω
        (slotRuns (Fin.cons u₀ us : Fin (M + 1) → (Fin n → Coin) × C))).2.2 = [] := by
      simp only [capPlan, Plan.exec, RunModel.named, hU0, decide_eq_true_eq, ha, ↓reduceIte]
    rw [hexec, if_neg (fun h => ha h.1), List.flatMap_nil]

open Classical in
/-- **`F0`'s plan searches its first run and each wait's first accepted run**, where its first run is accepted. -/
theorem conflictPlan_search (k : Msg → ℕ) (R : ℕ) (x : Msg) (j : Fin 1) (m : RunModel Coin Ω n C) (ω : Ω)
    (u₀ : (Fin n → Coin) × C) (W : Fin (k x * R) → (Fin n → Coin) × C)
    (ws : Fin (k x) → Fin R → (Fin n → Coin) × C)
    (hW : ∀ (t : Fin (k x)) (i : Fin R),
      slotRuns (Fin.cons u₀ W : Fin (k x * R + 1) → (Fin n → Coin) × C) (1 + t.val * R + i.val) = ws t i) :
    ((((conflictPlan k R).plan x j).exec m ω
        (slotRuns (Fin.cons u₀ W : Fin (k x * R + 1) → (Fin n → Coin) × C))).2.2.flatMap fun e =>
          m.log ω (m.named (slotRuns (Fin.cons u₀ W : Fin (k x * R + 1) → (Fin n → Coin) × C)) e)) =
      if m.acc ω u₀ then
        m.log ω u₀ ++ (List.finRange (k x)).flatMap fun t => ((firstPick (m.acc ω) R (ws t)).map (m.log ω)).getD []
      else [] := by
  have hU0 := slotRuns_cons_zero u₀ W
  by_cases ha : m.acc ω u₀
  · have hexec : (((conflictPlan k R).plan x j).exec m ω
        (slotRuns (Fin.cons u₀ W : Fin (k x * R + 1) → (Fin n → Coin) × C))).2.2 =
        ((waitsPlan R (k x) 1 [.fresh 0]).exec m ω
          (slotRuns (Fin.cons u₀ W : Fin (k x * R + 1) → (Fin n → Coin) × C))).2.2 := by
      simp only [conflictPlan, Plan.exec, RunModel.named, hU0, decide_eq_true_eq, ha, ↓reduceIte]
    rw [hexec, waitsPlan_exec m ω _ R (k x) 1 [.fresh 0] ws hW, if_pos ha]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, RunModel.named, hU0]
  · have hexec : (((conflictPlan k R).plan x j).exec m ω
        (slotRuns (Fin.cons u₀ W : Fin (k x * R + 1) → (Fin n → Coin) × C))).2.2 = [] := by
      simp only [conflictPlan, Plan.exec, RunModel.named, hU0, decide_eq_true_eq, ha, ↓reduceIte]
    rw [hexec, if_neg ha, List.flatMap_nil]

/-- **`B`'s plan searches its first run and the fork at its round.** -/
theorem forkPlan_search (k : Msg → ℕ) (x : Msg) (j : Fin (k x)) (m : RunModel Coin Ω n C) (ω : Ω)
    (us : Fin (1 + 1) → (Fin n → Coin) × C) :
    ((((forkPlan (Coin := Coin) k).plan x j).exec m ω (slotRuns us)).2.2.flatMap fun e =>
        m.log ω (m.named (slotRuns us) e)) =
      m.log ω (us 0) ++ m.log ω (fork (m.cut j.val) (us 0).1 (us 1).1, (us 1).2) := by
  show ([RunExp.fresh 0, .fork j.val (.fresh 0) (.fresh 1)].flatMap fun e => m.log ω (m.named (slotRuns us) e)) = _
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, RunModel.named, slotRuns_two_zero,
    slotRuns_two_one]

end Searches

/-! ## The coin maps -/

/-- `F0`'s waits as the plan's slots: wait `t`'s run `i` is slot `t·R + i` (`finProdFinEquiv`, curried). -/
def waitsEquiv (K R : ℕ) (U : Type) : (Fin K → Fin R → U) ≃ (Fin (K * R) → U) :=
  (Equiv.curry (Fin K) (Fin R) U).symm.trans (Equiv.arrowCongr finProdFinEquiv (Equiv.refl U))

/-- Wait `t`'s run `i` is the plan's fresh run `1 + t·R + i`. -/
theorem waitsEquiv_cons {K R : ℕ} {Coin C : Type} {n : ℕ} (u₀ : (Fin n → Coin) × C)
    (w : Fin K → Fin R → (Fin n → Coin) × C) (t : Fin K) (i : Fin R) :
    slotRuns (Fin.cons u₀ (waitsEquiv K R _ w) : Fin (K * R + 1) → (Fin n → Coin) × C) (1 + t.val * R + i.val) =
      w t i := by
  have hlt : t.val * R + i.val < K * R := by have := block_end_le t R; have := i.isLt; omega
  rw [show 1 + t.val * R + i.val = t.val * R + i.val + 1 by omega, slotRuns_cons_succ u₀ _ _ hlt]
  show w (finProdFinEquiv.symm ⟨t.val * R + i.val, hlt⟩).1 (finProdFinEquiv.symm ⟨t.val * R + i.val, hlt⟩).2 = w t i
  rw [show (⟨t.val * R + i.val, hlt⟩ : Fin (K * R)) = finProdFinEquiv (t, i) from
    Fin.ext (show t.val * R + i.val = i.val + R * t.val by ring), Equiv.symm_apply_apply]

/-- `F0`'s coins as the plan's: the selector is `Fin 1`'s one value, and the first run and the waits are the slots. -/
def conflictCoinsEquiv (Ω U : Type) (K R : ℕ) : Ω × U × (Fin K → Fin R → U) ≃ Fin 1 × Ω × (Fin (K * R + 1) → U) :=
  ((Equiv.refl Ω).prodCongr (((Equiv.refl U).prodCongr (waitsEquiv K R U)).trans (Fin.consEquiv fun _ => U))).trans
    (Equiv.uniqueProd _ (Fin 1)).symm

/-- `B`'s coins as the plan's: the round, the tape and the two runs, each with its abort coins. -/
def bCoinsEquiv (Ω Coin A : Type) (k : ℕ) :
    (Fin k × Ω × (Fin k → Coin) × (Fin k → Coin)) × A × A ≃ Fin k × Ω × (Fin (1 + 1) → (Fin k → Coin) × A) where
  toFun s := (s.1.1, s.1.2.1, ![(s.1.2.2.1, s.2.1), (s.1.2.2.2, s.2.2)])
  invFun t := ((t.1, t.2.1, (t.2.2 0).1, (t.2.2 1).1), (t.2.2 0).2, (t.2.2 1).2)
  left_inv _ := rfl
  right_inv t := Prod.ext rfl (Prod.ext rfl (funext fun i => by fin_cases i <;> rfl))

/-- `FC`'s coins as the plan's: the first run is the first slot. -/
def capCoinsEquiv (W : ℕ) (Ω U : Type) (M : ℕ) : Fin W × Ω × U × (Fin M → U) ≃ Fin W × Ω × (Fin (M + 1) → U) :=
  (Equiv.refl _).prodCongr ((Equiv.refl Ω).prodCongr (Fin.consEquiv fun _ => U))

/-! ## The plans are the finders -/

/-- **`B`'s plan at a protocol's runs that may abort is `B` there.** -/
theorem forkPlanIsBAbort (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : ForkPlanIsBAbort Coin := by
  intro H V x Ω _ _ P A _ _ keep
  unfold RunFinder.PrAt PrBAbort
  show (if h : 0 < V.k x then ((forkPlan (Coin := Coin) V.k).finder (runModel H V P x keep) x h).Pr H else 0) = _
  split_ifs with hk
  · unfold Finder.Pr
    refine ((UnifMap.equiv (bCoinsEquiv Ω Coin A (V.k x))).Pr _).symm.trans (Pr_congr fun s => ?_)
    obtain ⟨⟨j, ω, r, r'⟩, a, a'⟩ := s
    refine Iff.of_eq (congrArg (fun o => ∃ u v, o = some (u, v) ∧ Collision H u v) ?_)
    exact congrArg findColl (forkPlan_search V.k x j (runModel H V P x keep) ω ![(r, a), (r', a')])
  · rfl

open Classical in
/-- **`FC`'s plan at a protocol's runs that may abort is `FC` there.** -/
theorem capPlanIsFC (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : CapPlanIsFC Coin := by
  intro H V Cp x Ω _ _ P A _ _ keep M
  unfold RunFinder.PrAt PrFC
  show (if h : 0 < Cp.width x then ((capPlan Cp M).finder (runModel H V P x keep) x h).Pr H else 0) = _
  split_ifs with hW
  · unfold Finder.Pr
    refine ((UnifMap.equiv (capCoinsEquiv (Cp.width x) Ω ((Fin (V.k x) → Coin) × A) M)).Pr _).symm.trans
      (Pr_congr fun s => ?_)
    obtain ⟨q, ω, u₀, us⟩ := s
    refine Iff.of_eq (congrArg (fun o => ∃ u v, o = some (u, v) ∧ Collision H u v) ?_)
    refine (congrArg findColl (capPlan_search Cp M x q (runModel H V P x keep) ω u₀ us)).trans ?_
    have hsel : ∀ o : Option ((Fin (V.k x) → Coin) × A),
        o.elim u₀ (capFork V Cp x q u₀) = capFork V Cp x q u₀ (o.getD u₀) := by
      intro o
      cases o with
      | none => exact (congrArg (fun r => (r, u₀.2)) (fork_self (Cp.round x q) u₀.1)).symm
      | some v => rfl
    by_cases ho : opensAt H V Cp P x keep q ω u₀
    · rw [if_pos (show (runModel H V P x keep).acc ω u₀ ∧
        Cp.opens x q ((runModel H V P x keep).coins u₀) = true from ho)]
      show findColl ((abortRun H P V x keep ω u₀).vLog ++ (abortRun H P V x keep ω
          ((firstPick (fun v => opensAt H V Cp P x keep q ω (capFork V Cp x q u₀ v)) M us).elim u₀
            (capFork V Cp x q u₀))).vLog) =
        if opensAt H V Cp P x keep q ω u₀ then
          findColl ((abortRun H P V x keep ω u₀).vLog ++ (abortRun H P V x keep ω (capFork V Cp x q u₀
            ((firstPick (fun v => opensAt H V Cp P x keep q ω (capFork V Cp x q u₀ v)) M us).getD u₀))).vLog)
        else none
      rw [hsel]
      exact (if_pos ho).symm
    · rw [if_neg (show ¬((runModel H V P x keep).acc ω u₀ ∧
        Cp.opens x q ((runModel H V P x keep).coins u₀) = true) from ho)]
      show _ = if opensAt H V Cp P x keep q ω u₀ then _ else none
      exact (if_neg ho).symm
  · rfl

open Classical in
/-- **`F0`'s plan at a protocol's runs that may abort is `F0` there.** -/
theorem conflictPlanIsF0 (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] : ConflictPlanIsF0 Coin := by
  intro H V k x Ω _ _ P A _ _ keep R
  unfold RunFinder.PrAt PrF0
  rw [dif_pos (show 0 < (conflictPlan (Coin := Coin) k R).sel x from Nat.one_pos)]
  unfold Finder.Pr
  refine ((UnifMap.equiv (conflictCoinsEquiv Ω ((Fin (V.k x) → Coin) × A) (k x) R)).Pr _).symm.trans
    (Pr_congr fun s => ?_)
  obtain ⟨ω, u₀, w⟩ := s
  refine Iff.of_eq (congrArg (fun o => ∃ u v, o = some (u, v) ∧ Collision H u v) ?_)
  refine (congrArg findColl (conflictPlan_search k R x 0 (runModel H V P x keep) ω u₀ (waitsEquiv (k x) R _ w) w
    (waitsEquiv_cons u₀ w))).trans ?_
  by_cases ha : accAt H V P x keep ω u₀
  · rw [if_pos (show (runModel H V P x keep).acc ω u₀ from ha)]
    show _ = if accAt H V P x keep ω u₀ then _ else none
    exact (if_pos ha).symm
  · rw [if_neg (show ¬(runModel H V P x keep).acc ω u₀ from ha)]
    show _ = if accAt H V P x keep ω u₀ then _ else none
    exact (if_neg ha).symm

end Claims.RecGeneric
