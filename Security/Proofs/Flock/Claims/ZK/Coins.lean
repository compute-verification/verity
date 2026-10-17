import Proofs.Flock.Claims.ZK.Interact

/-!
# What a run opens

* `allOk_eval`, `strat_eval_cons`: the committed-coin prover's next message, evaluated.
* `cls_eq_some_iff`, `fcls_eq_succ_iff`, `fcls_eq_mism`: a transcript opens `c` exactly when every coin's opening is
  answered, passes the check and reads as `c`'s coin (`TCons`); a rewind at `c` is a binding failure at the first
  opening that passes the check and reads as another coin. `cls_log`, `fcls_eq_mism`: those checks' calls are in the
  classification's.
* `allOk_log`, `strat_log_cons`: the check of an opening after a passing prefix is among the committed-coin prover's
  calls for that round.
-/

namespace Claims.ZKMal

open Claims

section Eval

variable (H : Msg → Msg)

theorem allOk_eval (acc : ℕ → Msg → OC Bool) : ∀ (ms : List Msg) (j : ℕ),
    ((allOk acc j ms).eval H).1 = true ↔ ∀ i (hi : i < ms.length), ((acc (j + i) ms[i]).eval H).1 = true
  | [], j => by simp [allOk, OC.eval]
  | m :: ms, j => by
      simp only [allOk, ocBind_eval]
      cases hb : ((acc j m).eval H).1
      · simp only [Bool.false_eq_true, ite_false, OC.eval, false_iff, not_forall]
        exact ⟨0, by simp, by simpa using hb⟩
      · simp only [ite_true]
        rw [allOk_eval acc ms (j + 1)]
        constructor
        · intro h i hi
          cases i with
          | zero => simpa using hb
          | succ i =>
              have := h i (by simpa using hi)
              simpa [Nat.add_assoc, Nat.add_comm 1] using this
        · intro h i hi
          have := h (i + 1) (by simpa using hi)
          simpa [Nat.add_assoc, Nat.add_comm 1] using this

/-- When the checks before position `l` pass, the check at `l` is among `allOk`'s calls. -/
theorem allOk_log (acc : ℕ → Msg → OC Bool) : ∀ (ms : List Msg) (j l : ℕ) (hl : l < ms.length),
    (∀ i (hi : i < l), ((acc (j + i) (ms[i]'(by omega))).eval H).1 = true) →
      ((acc (j + l) ms[l]).eval H).2 ⊆ ((allOk acc j ms).eval H).2
  | [], _, _, hl, _ => absurd hl (by simp)
  | m :: ms, j, 0, _, _ => by
      simp only [allOk, ocBind_eval, Nat.add_zero, List.getElem_cons_zero]
      exact List.subset_append_left _ _
  | m :: ms, j, l + 1, hl, h => by
      have h0 := h 0 (by omega)
      simp only [Nat.add_zero, List.getElem_cons_zero] at h0
      simp only [allOk, ocBind_eval, h0, ite_true]
      refine List.Subset.trans ?_ (List.subset_append_right _ _)
      have := allOk_log acc ms (j + 1) l (by simpa using hl) fun i hi => by
        have := h (i + 1) (by omega)
        simpa [Nat.add_assoc, Nat.add_comm 1] using this
      simpa [Nat.add_assoc, Nat.add_comm 1] using this

/-- A committed-coin prover's checks of the openings so far are among its next message's calls. -/
theorem strat_log_cons {Coin : Type} (hello abort : Msg) (dec : Msg → Coin) (acc : Msg → ℕ → Msg → OC Bool)
    (gen : List Coin → OC Msg) (a ρ : Msg) (ops : List (Msg × Msg)) :
    ((allOk (acc ρ) 0 (ops.map Prod.snd)).eval H).2 ⊆ ((strat hello abort dec acc gen ((a, ρ) :: ops)).eval H).2 := by
  simp only [strat, ocBind_eval]
  exact List.subset_append_left _ _

theorem strat_eval_nil {Coin : Type} (hello abort : Msg) (dec : Msg → Coin) (acc : Msg → ℕ → Msg → OC Bool)
    (gen : List Coin → OC Msg) : ((strat hello abort dec acc gen []).eval H).1 = hello := by
  simp [strat, OC.eval]

theorem strat_eval_cons {Coin : Type} (hello abort : Msg) (dec : Msg → Coin) (acc : Msg → ℕ → Msg → OC Bool)
    (gen : List Coin → OC Msg) (a ρ : Msg) (ops : List (Msg × Msg)) :
    ((strat hello abort dec acc gen ((a, ρ) :: ops)).eval H).1 =
      if ((allOk (acc ρ) 0 (ops.map Prod.snd)).eval H).1 then ((gen (ops.map fun p => dec p.2)).eval H).1
      else abort := by
  simp only [strat, ocBind_eval]
  split <;> simp [OC.eval]

theorem facc_eval {Coin : Type} [DecidableEq Coin] {K : ℕ} (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin)
    (c : Fin K → Coin) (ρ : Msg) (j : ℕ) (m : Msg) :
    ((facc chk dec c ρ j m).eval H).1 = (((chk ρ j m).eval H).1 && decide (coinAt c j = some (dec m))) := by
  simp [facc, ocBind_eval, OC.eval]

end Eval

/-! ## Answers -/

theorem resp_cons_succ (p : Msg × Out) (τ : List (Msg × Out)) (i : ℕ) : resp (p :: τ) (i + 1) = resp τ i := by
  simp [resp]

theorem resp_eq_some {τ : List (Msg × Out)} {i : ℕ} {m : Msg} :
    resp τ i = some m ↔ ∃ (h : i < τ.length), τ[i].2 = .msg m := by
  unfold resp
  by_cases hi : i < τ.length
  · rw [List.getElem?_eq_getElem hi]
    simp only [hi, exists_true_left]
    generalize τ[i] = p
    obtain ⟨a, b⟩ := p
    cases b <;> simp
  · rw [List.getElem?_eq_none (by omega)]
    simp [hi]

theorem resp_take {τ : List (Msg × Out)} {n i : ℕ} (h : i < n) : resp (τ.take n) i = resp τ i := by
  simp [resp, h]

/-- A round answered with a message is followed by another while rounds are left. -/
theorem interact_length_of_resp (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (i : ℕ) (m : Msg), resp (interact ask pm n hist) i = some m → i + 1 < n →
      i + 1 < (interact ask pm n hist).length
  | 0, _, _, _, _, hn => absurd hn (by omega)
  | n + 1, hist, i, m, h, hn => by
      simp only [interact] at h ⊢
      cases hb : ask (hist.map Prod.fst ++ [pm hist]) with
      | msg m' =>
          rw [hb] at h
          simp only [List.length_cons]
          cases i with
          | zero =>
              obtain ⟨n', rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
              have := interact_head ask pm n' (hist ++ [(pm hist, m')])
              have hpos : 0 < (interact ask pm (n' + 1) (hist ++ [(pm hist, m')])).length := by
                rcases hl : interact ask pm (n' + 1) (hist ++ [(pm hist, m')]) with _ | ⟨p, l⟩
                · rw [hl] at this; simp at this
                · simp
              omega
          | succ i =>
              rw [resp_cons_succ] at h
              have := interact_length_of_resp ask pm n _ i m h (by omega)
              omega
      | acc => rw [hb] at h; cases i <;> simp [resp] at h
      | rej => rw [hb] at h; cases i <;> simp [resp] at h

theorem coinAt_eq_some {Coin : Type} {K : ℕ} (c : Fin K → Coin) (j : ℕ) (b : Coin) :
    coinAt c j = some b ↔ ∃ h : j < K, c ⟨j, h⟩ = b := by
  unfold coinAt
  split <;> simp_all

/-! ## What a transcript opens -/

/-- **The transcript opens `c`'s first `n` coins**: `Hello` is answered with a commitment `ρ`, and each of the first `n`
coins' openings is answered, passes the check against `ρ` and reads as `c`'s coin. -/
def TCons {Coin : Type} {K : ℕ} (chkP : Msg → ℕ → Msg → Bool) (dec : Msg → Coin) (c : Fin K → Coin)
    (τ : List (Msg × Out)) (n : ℕ) : Prop :=
  ∃ ρ, resp τ 0 = some ρ ∧ ∀ l < n, ∃ m, resp τ (l + 1) = some m ∧ chkP ρ l m = true ∧ coinAt c l = some (dec m)

section Cls

variable (H : Msg → Msg)

theorem clsGo_eval (chk : Msg → ℕ → Msg → OC Bool) (τ : List (Msg × Out)) (ρ : Msg) : ∀ (n j : ℕ),
    ((clsGo chk τ ρ j n).eval H).1 = true ↔
      ∀ i < n, ∃ m, resp τ (j + i + 1) = some m ∧ ((chk ρ (j + i) m).eval H).1 = true
  | 0, j => by simp [clsGo, OC.eval]
  | n + 1, j => by
      simp only [clsGo]
      cases hr : resp τ (j + 1) with
      | none =>
          simp only [OC.eval, Bool.false_eq_true, false_iff, not_forall]
          exact ⟨0, by omega, by simp [hr]⟩
      | some m =>
          simp only [ocBind_eval]
          cases hb : ((chk ρ j m).eval H).1
          · simp only [Bool.false_eq_true, ite_false, OC.eval, false_iff, not_forall]
            exact ⟨0, by omega, by simp [hr, hb]⟩
          · simp only [ite_true]
            rw [clsGo_eval chk τ ρ n (j + 1)]
            constructor
            · intro h i hi
              cases i with
              | zero => exact ⟨m, by simpa using hr, by simpa using hb⟩
              | succ i =>
                  obtain ⟨m', h1, h2⟩ := h i (by omega)
                  rw [show j + 1 + i = j + (i + 1) by omega] at h1 h2
                  exact ⟨m', h1, h2⟩
            · intro h i hi
              obtain ⟨m', h1, h2⟩ := h (i + 1) (by omega)
              rw [show j + (i + 1) = j + 1 + i by omega] at h1 h2
              exact ⟨m', h1, h2⟩

/-- When every check passes, each check's calls are among the walk's. -/
theorem clsGo_log (chk : Msg → ℕ → Msg → OC Bool) (τ : List (Msg × Out)) (ρ : Msg) : ∀ (n j : ℕ),
    ((clsGo chk τ ρ j n).eval H).1 = true → ∀ i < n, ∀ m, resp τ (j + i + 1) = some m →
      ((chk ρ (j + i) m).eval H).2 ⊆ ((clsGo chk τ ρ j n).eval H).2
  | 0, _, _, i, hi, _, _ => absurd hi (Nat.not_lt_zero i)
  | n + 1, j, hok, i, hi, m, hm => by
      simp only [clsGo] at hok ⊢
      cases hr : resp τ (j + 1) with
      | none => rw [hr] at hok; simp [OC.eval] at hok
      | some m₀ =>
          rw [hr] at hok
          simp only [ocBind_eval] at hok ⊢
          cases hb : ((chk ρ j m₀).eval H).1
          · rw [hb] at hok; simp [OC.eval] at hok
          · rw [hb] at hok
            simp only [ite_true] at hok ⊢
            cases i with
            | zero =>
                have : m = m₀ := by simp at hm; rw [hr] at hm; exact (Option.some.inj hm).symm
                subst this
                exact List.subset_append_left _ _
            | succ i =>
                have := clsGo_log chk τ ρ n (j + 1) hok i (by omega) m
                  (by rw [show j + 1 + i = j + (i + 1) by omega]; exact hm)
                refine List.Subset.trans ?_ (List.subset_append_right _ _)
                rwa [show j + 1 + i = j + (i + 1) by omega] at this

theorem fclsGo_eval_succ {Coin : Type} [DecidableEq Coin] (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin)
    (cc : ℕ → Option Coin) (τ : List (Msg × Out)) (ρ : Msg) : ∀ (n j : ℕ),
    ((fclsGo chk dec cc τ ρ j n).eval H).1 = .succ ↔
      ∀ i < n, ∃ m, resp τ (j + i + 1) = some m ∧ ((chk ρ (j + i) m).eval H).1 = true ∧ cc (j + i) = some (dec m)
  | 0, j => by simp [fclsGo, OC.eval]
  | n + 1, j => by
      simp only [fclsGo]
      cases hr : resp τ (j + 1) with
      | none =>
          simp only [OC.eval, reduceCtorEq, false_iff, not_forall]
          exact ⟨0, by omega, by simp [hr]⟩
      | some m =>
          simp only [ocBind_eval]
          cases hb : ((chk ρ j m).eval H).1
          · simp only [Bool.false_eq_true, ite_false, OC.eval, reduceCtorEq, false_iff, not_forall]
            exact ⟨0, by omega, by simp [hr, hb]⟩
          · simp only [ite_true]
            by_cases hc : cc j = some (dec m)
            · simp only [hc, ite_true]
              rw [fclsGo_eval_succ chk dec cc τ ρ n (j + 1)]
              constructor
              · intro h i hi
                cases i with
                | zero => exact ⟨m, by simpa using hr, by simpa using hb, by simpa using hc⟩
                | succ i =>
                    obtain ⟨m', h1, h2, h3⟩ := h i (by omega)
                    rw [show j + 1 + i = j + (i + 1) by omega] at h1 h2 h3
                    exact ⟨m', h1, h2, h3⟩
              · intro h i hi
                obtain ⟨m', h1, h2, h3⟩ := h (i + 1) (by omega)
                rw [show j + (i + 1) = j + 1 + i by omega] at h1 h2 h3
                exact ⟨m', h1, h2, h3⟩
            · simp only [hc, ite_false, OC.eval, reduceCtorEq, false_iff, not_forall]
              refine ⟨0, by omega, ?_⟩
              rintro ⟨m', h1, -, h3⟩
              simp only [Nat.add_zero] at h1 h3
              rw [hr] at h1
              cases h1
              exact hc h3

/-- The forced walk stops at the first opening that passes the check and reads as another coin. -/
theorem fclsGo_eval_mism {Coin : Type} [DecidableEq Coin] (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin)
    (cc : ℕ → Option Coin) (τ : List (Msg × Out)) (ρ : Msg) (m : Msg) : ∀ (t n j : ℕ), t < n →
    (∀ i < t, ∃ m, resp τ (j + i + 1) = some m ∧ ((chk ρ (j + i) m).eval H).1 = true ∧ cc (j + i) = some (dec m)) →
    resp τ (j + t + 1) = some m → ((chk ρ (j + t) m).eval H).1 = true → cc (j + t) ≠ some (dec m) →
      ((fclsGo chk dec cc τ ρ j n).eval H).1 = .mism ∧
        ((chk ρ (j + t) m).eval H).2 ⊆ ((fclsGo chk dec cc τ ρ j n).eval H).2
  | _, 0, _, ht, _, _, _, _ => absurd ht (Nat.not_lt_zero _)
  | 0, n + 1, j, _, _, hr, hb, hc => by
      simp only [Nat.add_zero] at hr hb hc
      simp only [fclsGo, hr, ocBind_eval, hb, ite_true, hc, ite_false, OC.eval, List.append_nil, Nat.add_zero]
      exact ⟨trivial, List.Subset.refl _⟩
  | t + 1, n + 1, j, ht, hpre, hr, hb, hc => by
      obtain ⟨m₀, h1, h2, h3⟩ := hpre 0 (by omega)
      simp only [Nat.add_zero] at h1 h2 h3
      simp only [fclsGo, h1, ocBind_eval, h2, ite_true, h3]
      have := fclsGo_eval_mism chk dec cc τ ρ m t n (j + 1) (by omega)
        (fun i hi => by
          obtain ⟨m', e1, e2, e3⟩ := hpre (i + 1) (by omega)
          rw [show j + (i + 1) = j + 1 + i by omega] at e1 e2 e3
          exact ⟨m', e1, e2, e3⟩)
        (by rw [show j + 1 + t = j + (t + 1) by omega]; exact hr)
        (by rw [show j + 1 + t = j + (t + 1) by omega]; exact hb)
        (by rw [show j + 1 + t = j + (t + 1) by omega]; exact hc)
      refine ⟨this.1, List.Subset.trans ?_ (List.subset_append_right _ _)⟩
      simpa [show j + 1 + t = j + (t + 1) by omega] using this.2

end Cls

/-! ## The protocol's classification -/

namespace Proto

variable {W : Type} (G : Proto W) (H : Msg → Msg)

/-- The check, evaluated. -/
def chkP (x : Msg) (ν : G.Nonce) (ρ : Msg) (j : ℕ) (m : Msg) : Bool := ((G.chk x ν ρ j m).eval H).1

/-- How a rewind at `c` ended, evaluated. -/
def fcls (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (τ : List (Msg × Out)) : St :=
  ((fclsOC (G.K x) (G.chk x ν) G.dec c τ).eval H).1

theorem cls_eq_some_iff (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out)) (c : Fin (G.K x) → G.Coin) :
    G.cls H x ν τ = some c ↔ TCons (G.chkP H x ν) G.dec c τ (G.K x) := by
  unfold cls clsOC TCons
  cases hr : resp τ 0 with
  | none => simp [OC.eval]
  | some ρ =>
      simp only [ocBind_eval, Option.some.injEq, exists_eq_left']
      have hg := clsGo_eval H (G.chk x ν) τ ρ (G.K x) 0
      simp only [Nat.zero_add] at hg
      cases hb : ((clsGo (G.chk x ν) τ ρ 0 (G.K x)).eval H).1
      · simp only [Bool.false_eq_true, ite_false, OC.eval, reduceCtorEq, false_iff, not_forall]
        rw [hb] at hg
        simp only [Bool.false_eq_true, false_iff, not_forall] at hg
        obtain ⟨l, hl, hn⟩ := hg
        exact ⟨l, hl, fun ⟨m, h1, h2, _⟩ => hn ⟨m, h1, h2⟩⟩
      · simp only [ite_true, OC.eval, Option.some.injEq]
        rw [hb] at hg
        have hg' := hg.mp rfl
        constructor
        · rintro rfl l hl
          obtain ⟨m, h1, h2⟩ := hg' l hl
          refine ⟨m, h1, h2, ?_⟩
          rw [coinAt_eq_some]
          exact ⟨hl, by rw [h1]; rfl⟩
        · intro h
          funext j
          obtain ⟨m, h1, -, h3⟩ := h j.val j.isLt
          obtain ⟨_, h3⟩ := (coinAt_eq_some c j.val _).mp h3
          rw [h1, Option.getD_some, ← h3]

theorem fcls_eq_succ_iff (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (τ : List (Msg × Out)) :
    G.fcls H x ν c τ = .succ ↔ TCons (G.chkP H x ν) G.dec c τ (G.K x) := by
  unfold fcls fclsOC TCons
  cases hr : resp τ 0 with
  | none => simp [OC.eval]
  | some ρ =>
      simp only [Option.some.injEq, exists_eq_left']
      rw [fclsGo_eval_succ H (G.chk x ν) G.dec (coinAt c) τ ρ (G.K x) 0]
      simp only [Nat.zero_add]
      rfl

/-- A transcript opens `c` exactly when a rewind at `c` with that transcript would succeed. -/
theorem cls_eq_some_iff_fcls (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (τ : List (Msg × Out)) :
    G.cls H x ν τ = some c ↔ G.fcls H x ν c τ = .succ := by
  rw [cls_eq_some_iff, fcls_eq_succ_iff]

/-- **A binding failure at coin `t`**: the rewind opened `c`'s coins before `t`, and coin `t`'s opening passes the
check and reads as another coin. -/
theorem fcls_eq_mism (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (τ : List (Msg × Out)) (ρ m : Msg) (t : ℕ)
    (ht : t < G.K x) (h0 : resp τ 0 = some ρ)
    (hpre : ∀ l < t, ∃ m, resp τ (l + 1) = some m ∧ G.chkP H x ν ρ l m = true ∧ coinAt c l = some (G.dec m))
    (hr : resp τ (t + 1) = some m) (hb : G.chkP H x ν ρ t m = true) (hc : coinAt c t ≠ some (G.dec m)) :
    G.fcls H x ν c τ = .mism ∧
      ((G.chk x ν ρ t m).eval H).2 ⊆ ((fclsOC (G.K x) (G.chk x ν) G.dec c τ).eval H).2 := by
  unfold fcls fclsOC
  rw [h0]
  have := fclsGo_eval_mism H (G.chk x ν) G.dec (coinAt c) τ ρ m t (G.K x) 0 ht
    (fun i hi => by simpa [Proto.chkP] using hpre i hi) (by simpa using hr) (by simpa [Proto.chkP] using hb)
    (by simpa using hc)
  simpa using this

/-- When a transcript opens `c`, each coin's check is among the classification's calls. -/
theorem cls_log (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out)) (c : Fin (G.K x) → G.Coin)
    (h : G.cls H x ν τ = some c) (ρ : Msg) (h0 : resp τ 0 = some ρ) (l : ℕ) (hl : l < G.K x) (m : Msg)
    (hm : resp τ (l + 1) = some m) :
    ((G.chk x ν ρ l m).eval H).2 ⊆ ((clsOC (G.K x) (G.chk x ν) G.dec τ).eval H).2 := by
  have hT := (G.cls_eq_some_iff H x ν τ c).mp h
  obtain ⟨ρ', h0', hall⟩ := hT
  rw [h0] at h0'
  cases h0'
  have hok : ((clsGo (G.chk x ν) τ ρ 0 (G.K x)).eval H).1 = true := by
    rw [clsGo_eval]
    intro i hi
    obtain ⟨m', h1, h2, -⟩ := hall i hi
    exact ⟨m', by simpa using h1, by simpa [Proto.chkP] using h2⟩
  unfold clsOC
  rw [h0]
  simp only [ocBind_eval, OC.eval, List.append_nil]
  have := clsGo_log H (G.chk x ν) τ ρ (G.K x) 0 hok l hl m (by simpa using hm)
  simpa using this

end Proto

end Claims.ZKMal
