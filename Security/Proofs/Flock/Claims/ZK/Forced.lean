import Proofs.Flock.Claims.ZK.Coins

/-!
# The real run and the forced run

At a nonce `ν`, tape `r` and coin vector `c`, the forced prover at `c` sending the honest prover's messages at `c`
(`realVec`) answers every history whose openings pass the check and read as `c`'s coins (`Good`) as the honest prover
does (`agree`). So against one verifier:

* `forced_eq_of_cls`: if the real run opens `c`, the forced run is the real run;
* `real_eq_of_fcls`: if the forced run opens all of `c`, the real run is the forced run;
* `cls_none_of_abort`: if the forced run stops before a binding failure, the real run opens no coin vector: it opened
  another `c'`, the two runs agree up to the first coin where `c'` and `c` differ, and there the forced run meets an
  opening that passes the check and reads as another coin.
* `resp_zero_eq`: every committed-coin prover sends `Hello` first, so every run's commitment is the verifier's answer to
  it.
-/

namespace Claims.ZKMal

open Claims

namespace Proto

variable {W : Type}

/-- **A history consistent with `c`**: every opening in it passes the check against the commitment and reads as `c`'s
coin. -/
def Good (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) :
    List (Msg × Msg) → Prop
  | [] => True
  | (_, ρ) :: ops => ∀ l (hl : l < ops.length),
      G.chkP H x ν ρ l ops[l].2 = true ∧ coinAt c l = some (G.dec ops[l].2)

/-- The honest prover's next message, evaluated. -/
def pmR (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (r : G.Rand) (h : List (Msg × Msg)) : Msg :=
  (((G.prover w).next x h (ν, r)).eval H).1

/-- The dummy-witness prover's next message, evaluated. -/
def pmD (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (r : G.Rand) (h : List (Msg × Msg)) : Msg :=
  ((G.dprover.next x h (ν, r)).eval H).1

/-- The forced prover's next message, evaluated. -/
def pmF (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (v : List Msg)
    (h : List (Msg × Msg)) : Msg :=
  ((G.forced x ν c v h).eval H).1

/-- **The forced prover at `c` with the real messages at `c` answers a history consistent with `c` as the honest prover
does.** -/
theorem agree (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (r : G.Rand) (c : Fin (G.K x) → G.Coin) :
    ∀ h, G.Good H x ν c h → G.pmR H x w ν r h = G.pmF H x ν c (G.realVec H x w ν r c) h
  | [], _ => by simp [pmR, pmF, prover, forced, strat_eval_nil]
  | (a, ρ) :: ops, hg => by
      have hK : ∀ l (hl : l < ops.length), l < G.K x := fun l hl => by
        obtain ⟨_, h2⟩ := hg l hl
        obtain ⟨hK, _⟩ := (coinAt_eq_some c l _).mp h2
        exact hK
      have hlen : ops.length ≤ G.K x := by
        by_contra hn
        exact absurd (hK (G.K x) (by omega)) (lt_irrefl _)
      have hR : ((allOk (G.chk x ν ρ) 0 (ops.map Prod.snd)).eval H).1 = true := by
        rw [allOk_eval]
        intro i hi
        simp only [List.length_map] at hi
        simpa [Proto.chkP] using (hg i hi).1
      have hF : ((allOk (facc (G.chk x ν) G.dec c ρ) 0 (ops.map Prod.snd)).eval H).1 = true := by
        rw [allOk_eval]
        intro i hi
        simp only [List.length_map] at hi
        rw [facc_eval]
        have := hg i hi
        simp only [Proto.chkP] at this
        simp [this.1, this.2]
      have hcs : ops.map (fun p => G.dec p.2) = (List.ofFn c).take ops.length := by
        refine List.ext_getElem (by simp [hlen]) fun l h1 h2 => ?_
        simp only [List.length_map] at h1
        obtain ⟨_, h3⟩ := (coinAt_eq_some c l _).mp (hg l h1).2
        simp [← h3]
      simp only [pmR, pmF, prover, forced]
      rw [strat_eval_cons, strat_eval_cons, hR, hF]
      simp only [ite_true, OC.eval, List.length_map]
      rw [hcs]
      unfold realVec
      rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega)]
      simp

/-- **A transcript that opens `c`'s first `n` coins has histories consistent with `c`** up to round `n + 1`. -/
theorem good_of_TCons (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (τ : List (Msg × Out)) (n : ℕ) (hT : TCons (G.chkP H x ν) G.dec c τ n) :
    ∀ i ≤ n + 1, G.Good H x ν c (histOf (τ.take i)) := by
  obtain ⟨ρ, h0, hall⟩ := hT
  intro i hi
  rcases τ with _ | ⟨⟨a, b⟩, τ'⟩
  · simp [resp] at h0
  cases i with
  | zero => simp [histOf, Good]
  | succ i =>
      have hb : b = .msg ρ := by
        obtain ⟨_, hb⟩ := resp_eq_some.mp h0
        simpa using hb
      subst hb
      simp only [List.take_succ_cons, histOf, List.map_cons, msgOf, Good]
      intro l hl
      simp only [List.length_map, List.length_take] at hl
      obtain ⟨m, h1, h2, h3⟩ := hall l (by omega)
      rw [resp_cons_succ] at h1
      obtain ⟨hl', hm⟩ := resp_eq_some.mp h1
      simp only [List.getElem_map, List.getElem_take]
      rw [hm]
      exact ⟨h2, h3⟩

section Runs

variable (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (w : W) (ν : G.Nonce) (r : G.Rand)

/-- **If the real run opens `c`, the forced run at `c` with the real messages is the real run.** -/
theorem forced_eq_of_cls (c : Fin (G.K x) → G.Coin)
    (h : G.cls H x ν (interact ask (G.pmR H x w ν r) (G.k x) []) = some c) :
    interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) [] = interact ask (G.pmR H x w ν r) (G.k x) [] := by
  have hT := (G.cls_eq_some_iff H x ν _ c).mp h
  refine interact_congr ask _ _ (G.k x) [] fun i hi => ?_
  have hlen : (interact ask (G.pmR H x w ν r) (G.k x) []).length ≤ G.K x + 2 :=
    interact_length_le ask _ (G.k x) []
  simp only [List.nil_append]
  exact (G.agree H x w ν r c _ (G.good_of_TCons H x ν c _ (G.K x) hT i (by omega))).symm

/-- **If the forced run at `c` with the real messages opens all of `c`, the real run is the forced run.** -/
theorem real_eq_of_fcls (c : Fin (G.K x) → G.Coin)
    (h : G.fcls H x ν c (interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) []) = .succ) :
    interact ask (G.pmR H x w ν r) (G.k x) [] = interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) [] := by
  have hT := (G.fcls_eq_succ_iff H x ν c _).mp h
  refine interact_congr ask _ _ (G.k x) [] fun i hi => ?_
  have hlen : (interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) []).length ≤ G.K x + 2 :=
    interact_length_le ask _ (G.k x) []
  simp only [List.nil_append]
  exact G.agree H x w ν r c _ (G.good_of_TCons H x ν c _ (G.K x) hT i (by omega))

/-- The forced run opens all of `c` exactly when the real run opens `c`. -/
theorem fcls_succ_iff (c : Fin (G.K x) → G.Coin) :
    G.fcls H x ν c (interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) []) = .succ ↔
      G.cls H x ν (interact ask (G.pmR H x w ν r) (G.k x) []) = some c := by
  constructor
  · intro h
    rw [G.real_eq_of_fcls H ask x w ν r c h, G.cls_eq_some_iff_fcls]
    exact h
  · intro h
    rw [G.forced_eq_of_cls H ask x w ν r c h, ← G.cls_eq_some_iff_fcls]
    exact h

/-- **A forced run that stops before a binding failure is a real run that opens no coin vector.** -/
theorem cls_none_of_abort (c : Fin (G.K x) → G.Coin)
    (h : G.fcls H x ν c (interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) []) = .abort) :
    G.cls H x ν (interact ask (G.pmR H x w ν r) (G.k x) []) = none := by
  set τR := interact ask (G.pmR H x w ν r) (G.k x) [] with hτR
  set τF := interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) [] with hτF
  cases hc' : G.cls H x ν τR with
  | none => rfl
  | some c' =>
      exfalso
      by_cases hcc : c' = c
      · subst hcc
        have := (G.fcls_succ_iff H ask x w ν r c').mpr hc'
        rw [← hτF, h] at this
        cases this
      · have hex : ∃ t, ∃ ht : t < G.K x, c' ⟨t, ht⟩ ≠ c ⟨t, ht⟩ := by
          by_contra hn
          simp only [not_exists, not_not] at hn
          exact hcc (funext fun j => hn j.val j.isLt)
        classical
        let t := Nat.find hex
        obtain ⟨ht, hne⟩ : ∃ ht : t < G.K x, c' ⟨t, ht⟩ ≠ c ⟨t, ht⟩ := Nat.find_spec hex
        have hbelow : ∀ l < t, coinAt c' l = coinAt c l := by
          intro l hl
          have hmin := Nat.find_min hex hl
          simp only [not_exists, not_not] at hmin
          unfold coinAt
          by_cases hlK : l < G.K x
          · simp [hlK, hmin hlK]
          · simp [hlK]
        have hT' := (G.cls_eq_some_iff H x ν τR c').mp hc'
        obtain ⟨ρ, h0, hall⟩ := hT'
        have hTc : TCons (G.chkP H x ν) G.dec c τR t := ⟨ρ, h0, fun l hl => by
          obtain ⟨m, h1, h2, h3⟩ := hall l (by omega)
          exact ⟨m, h1, h2, by rw [← hbelow l hl]; exact h3⟩⟩
        have htake : τF.take (t + 1 + 1) = τR.take (t + 1 + 1) :=
          interact_take_congr ask _ _ (G.k x) [] (t + 1) fun i hi _ => by
            simp only [List.nil_append]
            exact (G.agree H x w ν r c _ (G.good_of_TCons H x ν c τR t hTc i hi)).symm
        have hresp : ∀ l ≤ t + 1, resp τF l = resp τR l := fun l hl => by
          rw [← resp_take (n := t + 1 + 1) (by omega), htake, resp_take (by omega)]
        obtain ⟨m, h1, h2, h3⟩ := hall t ht
        have hmis := G.fcls_eq_mism H x ν c τF ρ m t ht (by rw [hresp 0 (by omega)]; exact h0)
          (fun l hl => by
            obtain ⟨m', e1, e2, e3⟩ := hall l (by omega)
            exact ⟨m', by rw [hresp (l + 1) (by omega)]; exact e1, e2, by rw [← hbelow l hl]; exact e3⟩)
          (by rw [hresp (t + 1) le_rfl]; exact h1) h2
          (by
            intro hc
            obtain ⟨_, e⟩ := (coinAt_eq_some c t _).mp hc
            obtain ⟨_, e'⟩ := (coinAt_eq_some c' t _).mp h3
            exact hne (e'.trans e.symm))
        rw [h] at hmis
        cases hmis.1

/-- **Every run's commitment is the verifier's answer to the prover's first message.** -/
theorem resp_zero_eq (pm₁ pm₂ : List (Msg × Msg) → Msg) (n : ℕ) (h : pm₁ [] = pm₂ []) :
    resp (interact ask pm₁ (n + 1) []) 0 = resp (interact ask pm₂ (n + 1) []) 0 := by
  unfold resp
  rw [interact_head, interact_head, h]

theorem pmD_nil : G.pmD H x ν r [] = G.hello x ν := by simp [pmD, dprover, strat_eval_nil]

theorem pmF_nil (c : Fin (G.K x) → G.Coin) (v : List Msg) : G.pmF H x ν c v [] = G.hello x ν := by
  simp [pmF, forced, strat_eval_nil]

end Runs

end Proto

end Claims.ZKMal
