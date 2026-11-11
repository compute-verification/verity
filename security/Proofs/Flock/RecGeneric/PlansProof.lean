module

public import Proofs.Flock.RecGeneric.AtFlock
public import Proofs.Flock.Claims.Recursion.KSLin2
public import Proofs.Flock.Claims.Recursion.Hybrid
public import Proofs.Flock.Claims.Recursion.OuterKS

/-!
# The plans' reads (P1–P3)

How often each of Flock's plans asks about a bad accepted run, against how often one run is bad (`RunFinder.ReadsLe`):

* `forkPlan` asks about no run (`forkPlanReads`, `ForkPlanReads`);
* `capPlan` asks about its first run, and about the forks of its caps' loop only where that run passes the caps' gate:
  at most twice as often as one run is bad (`capPlanReads`, `CapPlanReads`), the loop counted once by Wald's bound
  (`Pr_gate_loop_fork_le`);
* `conflictPlan` asks about its first run, and about each of its `k` waits only where that run accepts: at most `k + 1`
  times as often (`conflictPlanReads`, `ConflictPlanReads`), each wait counted once by Wald's bound (`Pr_gate_loop_le`)
  on its own fresh runs (`waitEmb`).

Each bound holds at every selector value and tape (`capPlan_asks_le`, `conflictPlan_asks_le`) and averages over them
(`Pr_sel_le`). A plan asks about a bad accepted run of a loop only where the loop's count of bad accepted runs up to its
first stop is nonzero (`capLoop_asks`, `waitLoop_asks`, `waitsPlan_asks`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

/-! ## Uniform coins -/

/-- Splitting `M + 1` uniform samples into the first and the rest keeps them uniform. -/
theorem Pr_cons_split {U : Type} [Fintype U] (M : ℕ) (Q : U → (Fin M → U) → Prop) :
    Pr (fun us : Fin (M + 1) → U => Q (us 0) (Fin.tail us)) = Pr (fun p : U × (Fin M → U) => Q p.1 p.2) := by
  refine ((UnifMap.equiv (A := U × (Fin M → U)) (B := Fin (M + 1) → U) (Fin.consEquiv fun _ => U)).Pr
    (fun us : Fin (M + 1) → U => Q (us 0) (Fin.tail us))).symm.trans (Pr_congr fun p => ?_)
  show Q ((Fin.cons p.1 p.2 : Fin (M + 1) → U) 0) (Fin.tail (Fin.cons p.1 p.2 : Fin (M + 1) → U)) ↔ Q p.1 p.2
  rw [Fin.cons_zero, Fin.tail_cons]

/-- `R + 1` distinct samples among `S + 1` uniform ones are uniform. -/
theorem Pr_emb_split {U : Type} [Fintype U] [Nonempty U] {R S : ℕ} (e : Fin (R + 1) ↪ Fin (S + 1))
    (Q : U → (Fin R → U) → Prop) :
    Pr (fun us : Fin (S + 1) → U => Q (us (e 0)) (fun i => us (e i.succ))) =
      Pr (fun p : U × (Fin R → U) => Q p.1 p.2) :=
  ((UnifMap.comp_emb (γ := U) e).Pr fun f : Fin (R + 1) → U => Q (f 0) (Fin.tail f)).trans (Pr_cons_split R Q)

/-- **A bound at every selector value and tape averages**: if each `(q, ω)` meets `E` at most `d · Pr[bad ω]`, a
uniform `(q, ω, v)` meets it at most `d · Pr[bad]`. -/
theorem Pr_sel_le {S Ω V U : Type} [Fintype S] [Nonempty S] [Fintype Ω] [Fintype V] [Fintype U]
    (E : S → Ω → V → Prop) (bad : Ω → U → Prop) (d : ℝ) (h : ∀ q ω, Pr (E q ω) ≤ d * Pr (bad ω)) :
    Pr (fun s : S × Ω × V => E s.1 s.2.1 s.2.2) ≤ d * Pr (fun t : Ω × U => bad t.1 t.2) :=
  calc Pr (fun s : S × Ω × V => E s.1 s.2.1 s.2.2)
      = Ex fun q : S => Ex fun ω : Ω => Pr (E q ω) := by
        rw [Claims.Recursion.Pr_prod]
        congr 1
        funext q
        exact Claims.Recursion.Pr_prod (fun b : Ω × V => E q b.1 b.2)
    _ ≤ Ex fun _ : S => Ex fun ω : Ω => d * Pr (bad ω) := Ex_mono fun q => Ex_mono fun ω => h q ω
    _ = d * Ex fun ω : Ω => Pr (bad ω) := by rw [Claims.Ex_const, Claims.Ex_mul_left]
    _ = d * Pr (fun t : Ω × U => bad t.1 t.2) :=
        congrArg (d * ·) (Claims.Recursion.Pr_prod (fun t : Ω × U => bad t.1 t.2)).symm

/-- The `t`-th of `K` blocks of `R` runs ends by the last of `K · R`. -/
theorem block_end_le {K : ℕ} (t : Fin K) (R : ℕ) : (t : ℕ) * R + R ≤ K * R :=
  calc (t : ℕ) * R + R = ((t : ℕ) + 1) * R := by ring
    _ ≤ K * R := Nat.mul_le_mul (Nat.succ_le_of_lt t.isLt) le_rfl

/-- **The `t`-th wait's runs** among `conflictPlan`'s: its first run `0`, then the wait's `t·R + 1, …, t·R + R`. -/
def waitEmb (K R : ℕ) (t : Fin K) : Fin (R + 1) ↪ Fin (K * R + 1) where
  toFun i := ⟨if (i : ℕ) = 0 then 0 else (t : ℕ) * R + (i : ℕ), by
    have := block_end_le t R
    have := i.isLt
    split_ifs <;> omega⟩
  inj' i j h := by
    simp only [Fin.mk.injEq] at h
    exact Fin.ext (by split_ifs at h <;> omega)

/-! ## A plan's asks, as loops' counts -/

section Runs

variable {Coin Ω C : Type} {n : ℕ}

/-- **The caps' gate** at a run: it accepts and its coins query the cap position (`op`). -/
def capGate (m : RunModel Coin Ω n C) (ω : Ω) (op : (ℕ → Coin) → Bool) (u : (Fin n → Coin) × C) : Prop :=
  m.acc ω u ∧ op (m.coins u) = true

open Classical in
/-- **A bad accepted run, counted**: `1` where `bad` holds and the run accepts, `0` elsewhere. -/
noncomputable def badAcc (m : RunModel Coin Ω n C) (ω : Ω) (bad : Ω → (Fin n → Coin) × C → Prop)
    (u : (Fin n → Coin) × C) : ℕ :=
  if bad ω u ∧ m.acc ω u then 1 else 0

theorem Ex_badAcc [Fintype Coin] [Fintype C] (m : RunModel Coin Ω n C) (bad : Ω → (Fin n → Coin) × C → Prop)
    (ω : Ω) : Ex (fun u : (Fin n → Coin) × C => (badAcc m ω bad u : ℝ)) = Pr (fun u => bad ω u ∧ m.acc ω u) := by
  classical
  rw [← Claims.Ex_ind]
  congr 1
  funext u
  unfold badAcc
  first
    | (split_ifs <;> simp)
    | (by_cases h : bad ω u ∧ m.acc ω u <;> simp [h])

end Runs

theorem loopCost_succ_ne_zero {U : Type} {g : U → Prop} {c : U → ℕ} {N : ℕ} {vs : Fin (N + 1) → U}
    (h : c (vs 0) ≠ 0 ∨ (¬ g (vs 0) ∧ loopCost g c N (Fin.tail vs) ≠ 0)) : loopCost g c (N + 1) vs ≠ 0 := by
  simp only [loopCost]
  rcases h with h | ⟨hg, h⟩
  · omega
  · rw [if_neg hg]
    omega

section Loops

variable {Coin Ω C : Type} {n : ℕ} (m : RunModel Coin Ω n C) (bad : Ω → (Fin n → Coin) × C → Prop) (ω : Ω)
  (us : ℕ → (Fin n → Coin) × C)

/-- **The caps' loop asks about a bad accepted run only where its count is nonzero**: the loop over the runs `vs`
(the fresh runs from `i` on) forked at `j` from `u₀`, gated by the caps' gate, counting bad accepted forks. -/
theorem capLoop_asks (j : ℕ) (op : (ℕ → Coin) → Bool) :
    ∀ (N i : ℕ) (vs : Fin N → (Fin n → Coin) × C), (∀ k : Fin N, vs k = us (i + k)) →
      (∃ e ∈ ((capLoop j op N i).exec m ω us).1, bad ω (m.named us e) ∧ m.acc ω (m.named us e)) →
      loopCost (fun v => capGate m ω op (fork (m.cut j) (us 0).1 v.1, v.2))
        (fun v => badAcc m ω bad (fork (m.cut j) (us 0).1 v.1, v.2)) N vs ≠ 0
  | 0, _, _, _, ⟨e, he, _⟩ => by simp [capLoop, Plan.exec] at he
  | N + 1, i, vs, hvs, ⟨e, he, hb⟩ => by
    classical
    have h0 : vs 0 = us i := (hvs 0).trans (congrArg us (by simp))
    have hexec : ((capLoop j op (N + 1) i).exec m ω us).1 =
        RunExp.fork j (.fresh 0) (.fresh i) ::
          (if m.acc ω (m.named us (.fork j (.fresh 0) (.fresh i))) ∧
              op (m.coins (m.named us (.fork j (.fresh 0) (.fresh i)))) = true then []
            else ((capLoop j op N (i + 1)).exec m ω us).1) := by
      by_cases ha : m.acc ω (m.named us (.fork j (.fresh 0) (.fresh i))) <;>
        by_cases ho : op (m.coins (m.named us (.fork j (.fresh 0) (.fresh i)))) = true <;>
        simp [capLoop, Plan.exec, ha, ho]
    rw [hexec, List.mem_cons] at he
    apply loopCost_succ_ne_zero
    rw [h0]
    rcases he with rfl | he
    · left
      have hb' : bad ω (fork (m.cut j) (us 0).1 (us i).1, (us i).2) ∧
          m.acc ω (fork (m.cut j) (us 0).1 (us i).1, (us i).2) := hb
      unfold badAcc
      rw [if_pos hb']
      exact one_ne_zero
    · right
      split_ifs at he with hg
      · simp at he
      · exact ⟨hg, capLoop_asks j op N (i + 1) (Fin.tail vs)
          (fun k => (hvs k.succ).trans (congrArg us (by rw [Fin.val_succ]; omega))) ⟨e, he, hb⟩⟩

/-- **A wait asks about a bad accepted run only where its count is nonzero, or its continuation asks**: the wait over
the runs `vs` (the fresh runs from `i` on), stopping at the first that accepts. -/
theorem waitLoop_asks (cont : Option RunExp → Plan Coin) :
    ∀ (N i : ℕ) (vs : Fin N → (Fin n → Coin) × C), (∀ k : Fin N, vs k = us (i + k)) →
      (∃ e ∈ ((waitLoop cont N i).exec m ω us).1, bad ω (m.named us e) ∧ m.acc ω (m.named us e)) →
      loopCost (m.acc ω) (badAcc m ω bad) N vs ≠ 0 ∨
        ∃ o, ∃ e ∈ ((cont o).exec m ω us).1, bad ω (m.named us e) ∧ m.acc ω (m.named us e)
  | 0, _, _, _, h => Or.inr ⟨none, by first | exact h | simpa [waitLoop] using h⟩
  | N + 1, i, vs, hvs, ⟨e, he, hb⟩ => by
    classical
    have h0 : vs 0 = us i := (hvs 0).trans (congrArg us (by simp))
    have hexec : ((waitLoop cont (N + 1) i).exec m ω us).1 = RunExp.fresh i ::
        (if m.acc ω (us i) then ((cont (some (.fresh i))).exec m ω us).1
          else ((waitLoop cont N (i + 1)).exec m ω us).1) := by
      by_cases ha : m.acc ω (us i) <;> simp [waitLoop, Plan.exec, RunModel.named, ha]
    rw [hexec, List.mem_cons] at he
    rcases he with rfl | he
    · left
      apply loopCost_succ_ne_zero
      left
      rw [h0]
      have hb' : bad ω (us i) ∧ m.acc ω (us i) := hb
      unfold badAcc
      rw [if_pos hb']
      exact one_ne_zero
    · split_ifs at he with ha
      · exact Or.inr ⟨_, e, he, hb⟩
      · rcases waitLoop_asks cont N (i + 1) (Fin.tail vs)
          (fun k => (hvs k.succ).trans (congrArg us (by rw [Fin.val_succ]; omega))) ⟨e, he, hb⟩ with h | h
        · left
          apply loopCost_succ_ne_zero
          right
          rw [h0]
          exact ⟨ha, h⟩
        · exact Or.inr h

/-- **`w` waits ask about a bad accepted run only where some wait's count is nonzero**: the `t`-th wait runs over the
fresh runs `base + t·R, …, base + t·R + R - 1`. -/
theorem waitsPlan_asks (R : ℕ) :
    ∀ (w base : ℕ) (found : List RunExp),
      (∃ e ∈ ((waitsPlan R w base found).exec m ω us).1, bad ω (m.named us e) ∧ m.acc ω (m.named us e)) →
      ∃ t : Fin w, loopCost (m.acc ω) (badAcc m ω bad) R
        (fun k : Fin R => us (base + (t : ℕ) * R + (k : ℕ))) ≠ 0
  | 0, _, _, ⟨e, he, _⟩ => by simp [waitsPlan, Plan.exec] at he
  | w + 1, base, found, h => by
    rcases waitLoop_asks m bad ω us (fun o => waitsPlan R w (base + R) (found ++ o.toList)) R base
        (fun k : Fin R => us (base + k)) (fun _ => rfl)
        (by first | exact h | simpa only [waitsPlan] using h) with h | ⟨o, h⟩
    · exact ⟨0, by simpa using h⟩
    · obtain ⟨t, ht⟩ := waitsPlan_asks R w (base + R) (found ++ o.toList) h
      refine ⟨t.succ, ?_⟩
      have e : (fun k : Fin R => us (base + ((t.succ : Fin (w + 1)) : ℕ) * R + (k : ℕ))) =
          fun k : Fin R => us (base + R + (t : ℕ) * R + (k : ℕ)) := by
        funext k
        congr 1
        rw [Fin.val_succ]
        ring
      rw [e]
      exact ht

end Loops

/-! ## Each plan's asks at a selector value and tape -/

section Bounds

variable {Coin Ω C : Type} {n : ℕ} [Fintype Coin] [Nonempty Coin] [Fintype C] [Nonempty C]
  (m : RunModel Coin Ω n C) (bad : Ω → (Fin n → Coin) × C → Prop)

/-- A plan's first run is bad and accepted at most as often as one run is bad. -/
theorem Pr_first_le (M : ℕ) (ω : Ω) :
    Pr (fun us : Fin (M + 1) → (Fin n → Coin) × C => bad ω (us 0) ∧ m.acc ω (us 0)) ≤ Pr (bad ω) :=
  calc _ = Pr (fun p : ((Fin n → Coin) × C) × (Fin M → (Fin n → Coin) × C) => bad ω p.1 ∧ m.acc ω p.1) :=
        Pr_cons_split M (fun u _ => bad ω u ∧ m.acc ω u)
    _ = Pr (fun u : (Fin n → Coin) × C => bad ω u ∧ m.acc ω u) := Claims.Pr_fst (fun u => bad ω u ∧ m.acc ω u)
    _ ≤ Pr (bad ω) := Pr_mono fun _ h => h.1

/-- **`FC`'s plan at a cap position and tape** asks about a bad accepted run at most twice as often as one run is
bad. -/
theorem capPlan_asks_le (Cp : Caps Coin) (M : ℕ) (x : Msg) (q : Fin (Cp.width x)) (ω : Ω) :
    Pr (fun us : Fin (M + 1) → (Fin n → Coin) × C =>
      ∃ e ∈ (((capPlan Cp M).plan x q).exec m ω (slotRuns us)).1,
        bad ω (m.named (slotRuns us) e) ∧ m.acc ω (m.named (slotRuns us) e)) ≤ 2 * Pr (bad ω) := by
  classical
  have hpt : ∀ us : Fin (M + 1) → (Fin n → Coin) × C,
      (∃ e ∈ (((capPlan Cp M).plan x q).exec m ω (slotRuns us)).1,
        bad ω (m.named (slotRuns us) e) ∧ m.acc ω (m.named (slotRuns us) e)) →
      (bad ω (us 0) ∧ m.acc ω (us 0)) ∨
        (capGate m ω (Cp.opens x q) (us 0) ∧
          loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2))
            (fun v => badAcc m ω bad (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2)) M (Fin.tail us) ≠ 0) := by
    intro us ⟨e, he, hb⟩
    have hs0 : slotRuns us 0 = us 0 := congrArg us (Fin.ext (by simp))
    have hplan : (capPlan Cp M).plan x q = .ask (.fresh 0) fun b =>
        if b then .look (.fresh 0) fun cs =>
          if Cp.opens x q cs then capLoop (Cp.round x q) (Cp.opens x q) M 1 else .done []
        else .done [] := rfl
    have hexec : (((capPlan Cp M).plan x q).exec m ω (slotRuns us)).1 = RunExp.fresh 0 ::
        (if capGate m ω (Cp.opens x q) (us 0) then
          ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω (slotRuns us)).1 else []) := by
      rw [hplan]
      by_cases ha : m.acc ω (us 0) <;> by_cases ho : Cp.opens x q (m.coins (us 0)) = true <;>
        simp [Plan.exec, RunModel.named, hs0, capGate, ha, ho]
    rw [hexec, List.mem_cons] at he
    rcases he with rfl | he
    · left
      have h0 : m.named (slotRuns us) (.fresh 0) = us 0 := hs0
      rw [h0] at hb
      exact hb
    · right
      split_ifs at he with hg
      · refine ⟨hg, ?_⟩
        have := capLoop_asks m bad ω (slotRuns us) (Cp.round x q) (Cp.opens x q) M 1 (Fin.tail us)
          (fun k => congrArg us (Fin.ext (by have := k.isLt; simp only [Fin.val_succ]; omega))) ⟨e, he, hb⟩
        rwa [hs0] at this
      · simp at he
  have hB : Pr (fun us : Fin (M + 1) → (Fin n → Coin) × C =>
      capGate m ω (Cp.opens x q) (us 0) ∧
        loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2))
          (fun v => badAcc m ω bad (fork (m.cut (Cp.round x q)) (us 0).1 v.1, v.2)) M (Fin.tail us) ≠ 0) ≤
      Pr (bad ω) :=
    calc _ = Pr (fun p : ((Fin n → Coin) × C) × (Fin M → (Fin n → Coin) × C) =>
          capGate m ω (Cp.opens x q) p.1 ∧
            loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) p.1.1 v.1, v.2))
              (fun v => badAcc m ω bad (fork (m.cut (Cp.round x q)) p.1.1 v.1, v.2)) M p.2 ≠ 0) :=
          Pr_cons_split M (fun u vs => capGate m ω (Cp.opens x q) u ∧
            loopCost (fun v => capGate m ω (Cp.opens x q) (fork (m.cut (Cp.round x q)) u.1 v.1, v.2))
              (fun v => badAcc m ω bad (fork (m.cut (Cp.round x q)) u.1 v.1, v.2)) M vs ≠ 0)
      _ ≤ Ex (fun u => (badAcc m ω bad u : ℝ)) :=
          Pr_gate_loop_fork_le (m.cut (Cp.round x q)) (capGate m ω (Cp.opens x q)) (badAcc m ω bad) M
      _ = Pr (fun u => bad ω u ∧ m.acc ω u) := Ex_badAcc m bad ω
      _ ≤ Pr (bad ω) := Pr_mono fun _ h => h.1
  refine (Pr_mono hpt).trans ((Claims.Pr_or_le _ _).trans ?_)
  have hA := Pr_first_le m bad M ω
  linarith

/-- **`F0`'s plan at a tape** asks about a bad accepted run at most `k + 1` times as often as one run is bad. -/
theorem conflictPlan_asks_le (k : Msg → ℕ) (R : ℕ) (x : Msg) (q : Fin 1) (ω : Ω) :
    Pr (fun us : Fin (k x * R + 1) → (Fin n → Coin) × C =>
      ∃ e ∈ (((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us)).1,
        bad ω (m.named (slotRuns us) e) ∧ m.acc ω (m.named (slotRuns us) e)) ≤ ((k x : ℝ) + 1) * Pr (bad ω) := by
  classical
  have hpt : ∀ us : Fin (k x * R + 1) → (Fin n → Coin) × C,
      (∃ e ∈ (((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us)).1,
        bad ω (m.named (slotRuns us) e) ∧ m.acc ω (m.named (slotRuns us) e)) →
      (bad ω (us 0) ∧ m.acc ω (us 0)) ∨
        ∃ t : Fin (k x), m.acc ω (us 0) ∧
          loopCost (m.acc ω) (badAcc m ω bad) R (fun i : Fin R => slotRuns us (1 + (t : ℕ) * R + (i : ℕ))) ≠ 0 := by
    intro us ⟨e, he, hb⟩
    have hs0 : slotRuns us 0 = us 0 := congrArg us (Fin.ext (by simp))
    have hplan : (conflictPlan (Coin := Coin) k R).plan x q = .ask (.fresh 0) fun b =>
        if b then waitsPlan R (k x) 1 [.fresh 0] else .done [] := rfl
    have hexec : (((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us)).1 = RunExp.fresh 0 ::
        (if m.acc ω (us 0) then ((waitsPlan R (k x) 1 [.fresh 0]).exec m ω (slotRuns us)).1 else []) := by
      rw [hplan]
      by_cases ha : m.acc ω (us 0) <;> simp [Plan.exec, RunModel.named, hs0, ha]
    rw [hexec, List.mem_cons] at he
    rcases he with rfl | he
    · left
      have h0 : m.named (slotRuns us) (.fresh 0) = us 0 := hs0
      rw [h0] at hb
      exact hb
    · right
      split_ifs at he with ha
      · obtain ⟨t, ht⟩ := waitsPlan_asks m bad ω (slotRuns us) R (k x) 1 [.fresh 0] ⟨e, he, hb⟩
        exact ⟨t, ha, ht⟩
      · simp at he
  have hB : ∀ t : Fin (k x), Pr (fun us : Fin (k x * R + 1) → (Fin n → Coin) × C => m.acc ω (us 0) ∧
      loopCost (m.acc ω) (badAcc m ω bad) R (fun i : Fin R => slotRuns us (1 + (t : ℕ) * R + (i : ℕ))) ≠ 0) ≤
      Pr (bad ω) := by
    intro t
    have h0v : ((waitEmb (k x) R t 0 : Fin (k x * R + 1)) : ℕ) = ((0 : Fin (k x * R + 1)) : ℕ) := by
      first | simp [waitEmb] | rfl
    have hsv : ∀ i : Fin R, ((waitEmb (k x) R t i.succ : Fin (k x * R + 1)) : ℕ) =
        min (1 + (t : ℕ) * R + (i : ℕ)) (k x * R) := by
      intro i
      have h1 := block_end_le t R
      have h2 := i.isLt
      have hle : 1 + (t : ℕ) * R + (i : ℕ) ≤ k x * R := by omega
      show (if ((i.succ : Fin (R + 1)) : ℕ) = 0 then 0 else (t : ℕ) * R + ((i.succ : Fin (R + 1)) : ℕ)) = _
      rw [Fin.val_succ, if_neg (Nat.succ_ne_zero _), min_eq_left hle]
      omega
    have hcongr : ∀ us : Fin (k x * R + 1) → (Fin n → Coin) × C,
        (m.acc ω (us 0) ∧ loopCost (m.acc ω) (badAcc m ω bad) R
            (fun i : Fin R => slotRuns us (1 + (t : ℕ) * R + (i : ℕ))) ≠ 0) ↔
          (m.acc ω (us (waitEmb (k x) R t 0)) ∧ loopCost (m.acc ω) (badAcc m ω bad) R
            (fun i : Fin R => us (waitEmb (k x) R t i.succ)) ≠ 0) := by
      intro us
      have h0 : us (waitEmb (k x) R t 0) = us 0 := congrArg us (Fin.ext h0v)
      have hs : (fun i : Fin R => us (waitEmb (k x) R t i.succ)) =
          fun i : Fin R => slotRuns us (1 + (t : ℕ) * R + (i : ℕ)) :=
        funext fun i => congrArg us (Fin.ext (hsv i))
      rw [h0, hs]
    calc _ = Pr (fun us : Fin (k x * R + 1) → (Fin n → Coin) × C =>
          m.acc ω (us (waitEmb (k x) R t 0)) ∧
            loopCost (m.acc ω) (badAcc m ω bad) R (fun i : Fin R => us (waitEmb (k x) R t i.succ)) ≠ 0) :=
          Pr_congr hcongr
      _ = Pr (fun p : ((Fin n → Coin) × C) × (Fin R → (Fin n → Coin) × C) =>
          m.acc ω p.1 ∧ loopCost (m.acc ω) (badAcc m ω bad) R p.2 ≠ 0) :=
          Pr_emb_split (waitEmb (k x) R t) fun u vs => m.acc ω u ∧ loopCost (m.acc ω) (badAcc m ω bad) R vs ≠ 0
      _ ≤ Ex (fun u => (badAcc m ω bad u : ℝ)) := Pr_gate_loop_le (m.acc ω) (badAcc m ω bad) R
      _ = Pr (fun u => bad ω u ∧ m.acc ω u) := Ex_badAcc m bad ω
      _ ≤ Pr (bad ω) := Pr_mono fun _ h => h.1
  refine (Pr_mono hpt).trans ((Claims.Pr_or_le _ _).trans ?_)
  have hsum := (Claims.Pr_exists_le (fun (t : Fin (k x)) (us : Fin (k x * R + 1) → (Fin n → Coin) × C) =>
      m.acc ω (us 0) ∧ loopCost (m.acc ω) (badAcc m ω bad) R
        (fun i : Fin R => slotRuns us (1 + (t : ℕ) * R + (i : ℕ))) ≠ 0)).trans
    (Finset.sum_le_sum fun t _ => hB t)
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
  have hA := Pr_first_le m bad (k x * R) ω
  linarith

end Bounds

/-! ## The plans' reads -/

section Reads

variable (Coin : Type) [Fintype Coin] [Nonempty Coin]

/-- **P1: `B`'s plan asks about no run**, so it meets a bad run with probability `0`. -/
theorem forkPlanReads : ForkPlanReads Coin := by
  intro k Ω _ _ n C _ _ m x bad
  unfold RunFinder.PrAsk
  split_ifs
  · calc Pr _ ≤ Pr (fun _ : Fin ((forkPlan (Coin := Coin) k).sel x) × Ω ×
            (Fin ((forkPlan (Coin := Coin) k).slots x + 1) → (Fin n → Coin) × C) => False) :=
          Pr_mono fun s ⟨e, he, _⟩ => by simp [forkPlan, Plan.exec] at he
      _ = 0 := Pr_false
      _ ≤ _ := by simp
  · simp

/-- **P2: `FC`'s plan meets a bad accepted run at most twice as often as one run is bad**: its first run, and its
caps' loop gated by that run, counted once by Wald's bound. -/
theorem capPlanReads : CapPlanReads Coin := by
  intro Cp M Ω _ _ n C _ _ m x bad
  unfold RunFinder.PrAsk
  split_ifs with hW
  · haveI : Nonempty (Fin ((capPlan Cp M).sel x)) := ⟨⟨0, hW⟩⟩
    refine (Pr_sel_le (fun q ω (us : Fin ((capPlan Cp M).slots x + 1) → (Fin n → Coin) × C) =>
        ∃ e ∈ (((capPlan Cp M).plan x q).exec m ω (slotRuns us)).1,
          bad ω (m.named (slotRuns us) e) ∧ m.acc ω (m.named (slotRuns us) e))
      bad 2 fun q ω => capPlan_asks_le m bad Cp M x q ω).trans (le_of_eq ?_)
    first | simp | norm_num
  · exact mul_nonneg (Nat.cast_nonneg _) (Pr_nonneg _)

/-- **P3: `F0`'s plan meets a bad accepted run at most `k + 1` times as often as one run is bad**: its first run, and
each of its `k` waits gated by that run, each counted once by Wald's bound. -/
theorem conflictPlanReads : ConflictPlanReads Coin := by
  intro k R Ω _ _ n C _ _ m x bad
  unfold RunFinder.PrAsk
  split_ifs with hW
  · haveI : Nonempty (Fin ((conflictPlan (Coin := Coin) k R).sel x)) := ⟨⟨0, hW⟩⟩
    refine (Pr_sel_le (fun q ω (us : Fin ((conflictPlan (Coin := Coin) k R).slots x + 1) → (Fin n → Coin) × C) =>
        ∃ e ∈ (((conflictPlan (Coin := Coin) k R).plan x q).exec m ω (slotRuns us)).1,
          bad ω (m.named (slotRuns us) e) ∧ m.acc ω (m.named (slotRuns us) e))
      bad ((k x : ℝ) + 1) fun q ω => conflictPlan_asks_le m bad k R x q ω).trans (le_of_eq ?_)
    first | simp | (push_cast; ring)
  · exact mul_nonneg (Nat.cast_nonneg _) (Pr_nonneg _)

end Reads

end Claims.RecGeneric
