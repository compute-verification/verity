import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.ZK.GKCost
import Proofs.Flock.Soundness.ZK.GK.Theorem

/-!
# The Goldreich–Kahan simulator with estimation, in expected polynomial time

`Proto.simulatorE t M` is the session analysis's simulator (`FlockSoundness.ZK.GK.gk`, the deployed `gk_simulate`) as a
`Claims.Simulator`, with threshold `t` and cap `M`, using `V*` only through `SP.ask`:

1. one dummy-witness run; if it does not open every coin, its transcript is the output;
2. **estimation**: fresh dummy-witness runs on the same nonce until `t` of them open every coin, at most `M`; `N` runs;
3. **rewinds**: at most `4N` rewinds at the coins the first run opened, each the SHVZK simulator's messages sent by the
   forced prover; the first that opens them all is the output.

A rewind that opens one of those coins as another (a binding failure), the cap, and running out of rewinds output the
empty transcript.

* `Proto.gkE_zk`: it meets `ZKmalR` at `ε_zk = 3·δ_A + 8t·δ_B + 3^{-t} + t/M`, plus the probability of a collision among
  its own SHA-512 calls: the old `gk_close` on the old `Model`, built from the protocol at each nonce (`Proto.model`),
  whose `bind_real` is `real_eq_of_not_mism`, `PrefinalClose` is `PrefixHides` and `RewindClose` is `SHVZK`;
* `Proto.gkE_costExp`: at most `(1 + 5t)·runBound` calls in expectation over its coins against every verifier, for every
  cap `M` (`SimCostExp`, the old `sim_runs_ev`); `Proto.gkE_cost`: at most `(1 + 5M)·runBound` on every coin
  (`SimCost`, the old `sim_runs_le`);
* `Proto.gkE_zk_pow`: at the cap `M = t·3^t`, `ε_zk = 3·δ_A + 8t·δ_B + 2·3^{-t}`.

**Why a cap.** A `Claims.Simulator` has finitely many coins and stops on every one, so the uncapped simulator is not
one. The cap costs `t/M` in `ε_zk` and nothing in the expected cost, which is uniform in `M`; at `M = t·3^t` its term is
the rewinds' `3^{-t}`, and only the worst case grows with it.

**The coins.** `gk` draws its coins as it goes (a `Rand` program); the simulator reads them from tapes fixed in advance,
the nonce and the first run's tape, `M` dummy-witness tapes and `4M` SHVZK tapes, each read at most once (`gkT`). The
two have one law (`ev_gkT`, from `ev_estimateT` and `ev_rewindsT`).
-/

namespace Claims.ZKMal

open Claims

local notation "GOut" => FlockSoundness.ZK.GK.Out

/-! ## The simulator's program on tapes -/

section Tape

open FlockSoundness.ZK.GK

/-- **The estimation on a tape**: `estimate`, reading its at most `b` coins from `ω` in order. -/
def estimateT {C : Type} (comp : C → Bool) : (b k : ℕ) → (Fin b → C) → ℕ × Bool
  | _, 0, _ => (0, true)
  | 0, _ + 1, _ => (0, false)
  | b + 1, k + 1, ω =>
      ((estimateT comp b (if comp (ω 0) then k else k + 1) fun i => ω i.succ).1 + 1,
        (estimateT comp b (if comp (ω 0) then k else k + 1) fun i => ω i.succ).2)

theorem estimateT_le {C : Type} (comp : C → Bool) : ∀ (b k : ℕ) (ω : Fin b → C), (estimateT comp b k ω).1 ≤ b
  | b, 0, _ => by cases b <;> simp [estimateT]
  | 0, _ + 1, _ => by simp [estimateT]
  | b + 1, k + 1, ω => by
      simp only [estimateT]
      have := estimateT_le comp b (if comp (ω 0) then k else k + 1) fun i => ω i.succ
      omega

theorem ev_estimateT {C : Type} [Fintype C] [Nonempty C] (comp : C → Bool) :
    ∀ (b k : ℕ) (f : ℕ × Bool → ℝ),
      Ex (fun ω : Fin b → C => f (estimateT comp b k ω)) = Rand.ev f (estimate C comp b k)
  | b, 0, f => by cases b <;> simp [estimateT, estimate, Ex_const]
  | 0, k + 1, f => by simp [estimateT, estimate, Ex_const]
  | b + 1, k + 1, f => by
      rw [ev_estimate_succ]
      refine (Ex_cons b fun d σ => f ((estimateT comp b (if comp d then k else k + 1) σ).1 + 1,
        (estimateT comp b (if comp d then k else k + 1) σ).2)).trans ?_
      show Ex _ = Ex _
      congr 1
      funext d
      exact ev_estimateT comp b _ fun r => f (r.1 + 1, r.2)

/-- **The rewinds on a tape**: `rewinds`, reading its at most `n` coins from `σ` in order (`n ≤ L`). -/
def rewindsT {V C : Type} (v₀ : V) (rw : C → V) (comp bad : V → Bool) : (n L : ℕ) → (Fin L → C) → GOut V × ℕ
  | 0, _, _ => (.fail, 0)
  | _ + 1, 0, _ => (.fail, 0)
  | n + 1, L + 1, σ =>
      if bad (rw (σ 0)) then (.bind v₀ (rw (σ 0)), 1)
      else if comp (rw (σ 0)) then (.view (rw (σ 0)), 1)
      else ((rewindsT v₀ rw comp bad n L fun i => σ i.succ).1, (rewindsT v₀ rw comp bad n L fun i => σ i.succ).2 + 1)

theorem rewindsT_le {V C : Type} (v₀ : V) (rw : C → V) (comp bad : V → Bool) :
    ∀ (n L : ℕ) (σ : Fin L → C), (rewindsT v₀ rw comp bad n L σ).2 ≤ n
  | 0, _, _ => by simp [rewindsT]
  | _ + 1, 0, _ => by simp [rewindsT]
  | n + 1, L + 1, σ => by
      have := rewindsT_le v₀ rw comp bad n L fun i => σ i.succ
      simp only [rewindsT]
      split_ifs <;> simp only <;> omega

theorem ev_rewindsT {V C : Type} [Fintype C] [Nonempty C] (v₀ : V) (rw : C → V) (comp bad : V → Bool) :
    ∀ (n L : ℕ), n ≤ L → ∀ f : GOut V × ℕ → ℝ,
      Ex (fun σ : Fin L → C => f (rewindsT v₀ rw comp bad n L σ)) = Rand.ev f (rewinds C v₀ rw comp bad n)
  | 0, L, _, f => by simp [rewindsT, rewinds, Ex_const]
  | n + 1, 0, h, _ => absurd h (by omega)
  | n + 1, L + 1, h, f => by
      rw [ev_rewinds_succ]
      obtain ⟨F, hF⟩ : ∃ F : C → (Fin L → C) → ℝ, F = fun d σ' => if bad (rw d) = true then f (.bind v₀ (rw d), 1)
          else if comp (rw d) = true then f (.view (rw d), 1)
          else f ((rewindsT v₀ rw comp bad n L σ').1, (rewindsT v₀ rw comp bad n L σ').2 + 1) := ⟨_, rfl⟩
      have e : (fun σ : Fin (L + 1) → C => f (rewindsT v₀ rw comp bad (n + 1) (L + 1) σ)) =
          fun σ => F (σ 0) fun i => σ i.succ := by
        funext σ
        simp only [rewindsT, hF]
        split_ifs <;> rfl
      rw [e, Ex_cons L F]
      show Ex _ = Ex _
      congr 1
      funext d
      subst hF
      by_cases hb : bad (rw d) = true
      · simp only [hb, ite_true, Ex_const]
      · by_cases hc : comp (rw d) = true
        · simp only [hb, hc, ite_true, Bool.false_eq_true, ite_false, Ex_const]
        · simp only [hb, hc, Bool.false_eq_true, ite_false]
          exact ev_rewindsT v₀ rw comp bad n L (by omega) fun o => f (o.1, o.2 + 1)

/-- What follows the estimation's result `r`: at most `4·r.1` rewinds from `v₀` if it succeeded, else a failure; the
count adds the first run and the estimation's. -/
def afterT {V C : Type} (v₀ : V) (rw : C → V) (comp bad : V → Bool) (L : ℕ) (r : ℕ × Bool) (σ : Fin L → C) :
    GOut V × ℕ :=
  if r.2 then ((rewindsT v₀ rw comp bad (4 * r.1) L σ).1, (rewindsT v₀ rw comp bad (4 * r.1) L σ).2 + 1 + r.1)
  else (.fail, 1 + r.1)

/-- **`gk` on tapes**: the first run's coin `ω₀`, the estimation's `M` coins `ωs` and the rewinds' `4M` coins `σs`. -/
def gkT {V Cd Cr : Type} (dummy : Cd → V) (rw : V → Cr → V) (comp : V → Bool) (bad : V → V → Bool) (t M : ℕ)
    (ω₀ : Cd) (ωs : Fin M → Cd) (σs : Fin (4 * M) → Cr) : GOut V × ℕ :=
  if comp (dummy ω₀) then
    afterT (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀)) (4 * M) (estimateT (fun ω => comp (dummy ω)) M t ωs) σs
  else (.view (dummy ω₀), 1)

/-- **`gk` and `gkT` have one law.** -/
theorem ev_gkT {V Cd Cr : Type} [Fintype Cd] [Nonempty Cd] [Fintype Cr] [Nonempty Cr] (dummy : Cd → V)
    (rw : V → Cr → V) (comp : V → Bool) (bad : V → V → Bool) (t M : ℕ) (f : GOut V × ℕ → ℝ) :
    Ex (fun q : Cd × (Fin M → Cd) × (Fin (4 * M) → Cr) => f (gkT dummy rw comp bad t M q.1 q.2.1 q.2.2)) =
      Rand.ev f (gk Cd Cr dummy rw comp bad t M) := by
  rw [ev_gk, Ex_prod]
  show Ex _ = Ex _
  congr 1
  funext ω₀
  rw [Ex_prod]
  by_cases hc : comp (dummy ω₀) = true
  · simp only [gkT, hc, ite_true]
    set g : ℕ × Bool → ℝ := fun r => if r.2 = true then
        Rand.ev (fun o => f (o.1, o.2 + 1 + r.1)) (rewinds Cr (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀)) (4 * r.1))
      else f (.fail, 1 + r.1) with hg
    have hpt : ∀ r : ℕ × Bool, r.1 ≤ M →
        Ex (fun σ : Fin (4 * M) → Cr => f (afterT (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀)) (4 * M) r σ)) =
          g r := by
      intro r hr
      simp only [afterT, hg]
      by_cases h2 : r.2 = true
      · simp only [h2, ite_true]
        exact ev_rewindsT (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀)) (4 * r.1) (4 * M) (by omega)
          fun o => f (o.1, o.2 + 1 + r.1)
      · simp only [h2, Bool.false_eq_true, ite_false, Ex_const]
    rw [show (fun ωs : Fin M → Cd => Ex fun σ : Fin (4 * M) → Cr =>
        f (afterT (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀)) (4 * M)
          (estimateT (fun ω => comp (dummy ω)) M t ωs) σ)) =
        fun ωs => g (estimateT (fun ω => comp (dummy ω)) M t ωs) from
      funext fun ωs => hpt _ (estimateT_le _ M t ωs)]
    rw [ev_estimateT]
  · simp only [gkT, hc, Bool.false_eq_true, ite_false, Ex_const]

theorem gkT_runs_le {V Cd Cr : Type} (dummy : Cd → V) (rw : V → Cr → V) (comp : V → Bool) (bad : V → V → Bool)
    (t M : ℕ) (ω₀ : Cd) (ωs : Fin M → Cd) (σs : Fin (4 * M) → Cr) :
    (gkT dummy rw comp bad t M ω₀ ωs σs).2 ≤ 1 + 5 * M := by
  have h1 := estimateT_le (fun ω => comp (dummy ω)) M t ωs
  unfold gkT afterT
  split_ifs
  · have := rewindsT_le (dummy ω₀) (rw (dummy ω₀)) comp (bad (dummy ω₀))
      (4 * (estimateT (fun ω => comp (dummy ω)) M t ωs).1) (4 * M) σs
    simp only
    omega
  · simp only; omega
  · simp only; omega

end Tape

/-! ## Indicators -/

theorem ind_comp {α β : Type} (U : β → Prop) (f : α → β) :
    (fun a => FlockSoundness.Game.ind U (f a)) = FlockSoundness.Game.ind fun a => U (f a) := rfl

theorem Ex_ind_eq {α : Type} [Fintype α] (p : α → Prop) : Ex (FlockSoundness.Game.ind p) = Pr p := by
  classical
  exact Ex_ind p

theorem Ex_ind_comp {α β : Type} [Fintype α] (U : β → Prop) (f : α → β) :
    Ex (fun a => FlockSoundness.Game.ind U (f a)) = Pr fun a => U (f a) :=
  Ex_ind_eq _

theorem avgR_eq_Ex {α : Type} [Fintype α] (f : α → ℝ) : FlockSoundness.ZK.GK.avgR f = Ex f := rfl

/-! ## The forced run away from a binding failure -/

/-- Every round of a transcript but the last is answered with a message. -/
theorem interact_msg (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (j : ℕ) (hj : j + 1 < (interact ask pm n hist).length),
      ∃ m, ((interact ask pm n hist)[j]'(by omega)).2 = .msg m
  | 0, _, _, hj => by simp [interact] at hj
  | n + 1, hist, j, hj => by
      rcases h : ask (hist.map Prod.fst ++ [pm hist]) with m | _ | _
      · cases j with
        | zero => simp [interact, h]
        | succ j =>
            have hj' : j + 1 < (interact ask pm n (hist ++ [(pm hist, m)])).length := by
              simp only [interact, h, List.length_cons] at hj
              omega
            obtain ⟨m', hm'⟩ := interact_msg ask pm n _ j hj'
            exact ⟨m', by simp only [interact, h, List.getElem_cons_succ]; exact hm'⟩
      · simp [interact, h] at hj
      · simp [interact, h] at hj

namespace Proto

variable {W : Type}

/-- **Away from a binding failure, the forced prover with the real messages answers as the honest prover**: on every
history of a transcript whose rounds but the last are answered with messages and whose rewind at `c` is not a binding
failure. Where a history is not consistent with `c`, its first opening that is not passes no check, and both provers
abort. -/
theorem pm_eq_of_not_mism (G : Proto W) (H : Msg → Msg) (x : Msg) (w : W) (ν : G.Nonce) (r : G.Rand)
    (c : Fin (G.K x) → G.Coin) (τ : List (Msg × Out))
    (hmsg : ∀ (j : ℕ) (hj : j + 1 < τ.length), ∃ m, (τ[j]'(by omega)).2 = .msg m) (hlen : τ.length ≤ G.K x + 2)
    (hnm : G.fcls H x ν c τ ≠ .mism) (i : ℕ) (hi : i < τ.length) :
    G.pmR H x w ν r (histOf (τ.take i)) = G.pmF H x ν c (G.realVec H x w ν r c) (histOf (τ.take i)) := by
  classical
  by_cases hg : G.Good H x ν c (histOf (τ.take i))
  · exact G.agree H x w ν r c _ hg
  rcases τ with _ | ⟨⟨a, b⟩, τ'⟩
  · simp at hi
  cases i with
  | zero => exact absurd (by simp [histOf, Good]) hg
  | succ i =>
    simp only [List.length_cons] at hi hlen
    obtain ⟨ρ, hb⟩ := hmsg 0 (by simp only [List.length_cons]; omega)
    simp only [List.getElem_cons_zero] at hb
    subst hb
    simp only [List.take_succ_cons, histOf, List.map_cons, show msgOf (.msg ρ) = ρ from rfl] at hg ⊢
    set ops := List.map (fun p => (p.1, msgOf p.2)) (List.take i τ') with hops
    have hopsLen : ops.length = i := by simp only [hops, List.length_map, List.length_take]; omega
    have hopsGet : ∀ (l : ℕ) (hl : l < ops.length), resp ((a, .msg ρ) :: τ') (l + 1) = some ops[l].2 := by
      intro l hl
      obtain ⟨m, hm⟩ := hmsg (l + 1) (by simp only [List.length_cons]; omega)
      simp only [List.getElem_cons_succ] at hm
      rw [resp_cons_succ, resp_eq_some.mpr ⟨by omega, hm⟩]
      simp [hops, hm, msgOf]
    simp only [Good, not_forall] at hg
    obtain ⟨l₀, hl₀, hbad⟩ := hg
    have hex : ∃ l, ∃ hl : l < ops.length,
        ¬ (G.chkP H x ν ρ l ops[l].2 = true ∧ coinAt c l = some (G.dec ops[l].2)) := ⟨l₀, hl₀, hbad⟩
    obtain ⟨ht, hnot⟩ := Nat.find_spec hex
    have hmin : ∀ l < Nat.find hex, ∀ hl : l < ops.length,
        G.chkP H x ν ρ l ops[l].2 = true ∧ coinAt c l = some (G.dec ops[l].2) := fun l hl hl' => by
      by_contra hn
      exact Nat.find_min hex hl ⟨hl', hn⟩
    by_cases hchk : G.chkP H x ν ρ (Nat.find hex) ops[Nat.find hex].2 = true
    · exfalso
      apply hnm
      refine (G.fcls_eq_mism H x ν c _ ρ ops[Nat.find hex].2 (Nat.find hex) (by omega) (by simp [resp]) ?_
        (hopsGet _ ht) hchk fun h => hnot ⟨hchk, h⟩).1
      intro l hl
      exact ⟨ops[l].2, hopsGet l (by omega), (hmin l hl (by omega)).1, (hmin l hl (by omega)).2⟩
    · have hR : ((allOk (G.chk x ν ρ) 0 (ops.map Prod.snd)).eval H).1 = false := by
        refine Bool.eq_false_iff.mpr fun h => hchk ?_
        have := (allOk_eval H _ _ 0).mp h (Nat.find hex) (by simpa using ht)
        simpa [Proto.chkP] using this
      have hF : ((allOk (facc (G.chk x ν) G.dec c ρ) 0 (ops.map Prod.snd)).eval H).1 = false := by
        refine Bool.eq_false_iff.mpr fun h => hchk ?_
        have := (allOk_eval H _ _ 0).mp h (Nat.find hex) (by simpa using ht)
        rw [facc_eval] at this
        simp only [Bool.and_eq_true] at this
        simpa [Proto.chkP] using this.1
      simp only [pmR, pmF, prover, forced]
      rw [strat_eval_cons, strat_eval_cons, hR, hF]
      rfl

section Runs

variable (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (w : W) (ν : G.Nonce) (r : G.Rand)

/-- **A forced run at `c` with the real messages that is not a binding failure is the real run.** -/
theorem real_eq_of_not_mism (c : Fin (G.K x) → G.Coin)
    (h : G.fcls H x ν c (interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) []) ≠ .mism) :
    interact ask (G.pmR H x w ν r) (G.k x) [] = interact ask (G.pmF H x ν c (G.realVec H x w ν r c)) (G.k x) [] :=
  interact_congr ask _ _ (G.k x) [] fun i hi => by
    simp only [List.nil_append]
    exact G.pm_eq_of_not_mism H x w ν r c _ (interact_msg ask _ (G.k x) [])
      (interact_length_le ask _ (G.k x) []) h i hi

end Runs

/-- A transcript that opens some coin vector is, at every `c`, a rewind that opens all of `c` or a binding failure. -/
theorem fcls_ne_abort_of_cls (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c c' : Fin (G.K x) → G.Coin)
    (τ : List (Msg × Out)) (h : G.cls H x ν τ = some c') : G.fcls H x ν c τ ≠ .abort := by
  classical
  obtain ⟨ρ, h0, hall⟩ := (G.cls_eq_some_iff H x ν τ c').mp h
  by_cases hex : ∃ t, t < G.K x ∧ ∀ m, resp τ (t + 1) = some m → coinAt c t ≠ some (G.dec m)
  · obtain ⟨ht, hne⟩ := Nat.find_spec hex
    obtain ⟨m, h1, h2, -⟩ := hall _ ht
    have := G.fcls_eq_mism H x ν c τ ρ m (Nat.find hex) ht h0 (fun l hl => by
        obtain ⟨m', e1, e2, -⟩ := hall l (by omega)
        refine ⟨m', e1, e2, ?_⟩
        by_contra hn
        exact Nat.find_min hex hl ⟨by omega, fun m'' hm'' => by rw [e1] at hm''; cases hm''; exact hn⟩)
      h1 h2 (hne m h1)
    rw [this.1]
    decide
  · push Not at hex
    have : G.fcls H x ν c τ = .succ := (G.fcls_eq_succ_iff H x ν c τ).mpr ⟨ρ, h0, fun l hl => by
      obtain ⟨m, e1, e2, -⟩ := hall l hl
      obtain ⟨m', e1', e3⟩ := hex l hl
      rw [e1] at e1'
      cases e1'
      exact ⟨m, e1, e2, e3⟩⟩
    rw [this]
    decide

/-! ## The session analysis's model of the protocol at a nonce -/

/-- The dummy-witness run on the nonce `ν` and tape `r`. -/
def dumT (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce) (r : G.Rand) :
    List (Msg × Out) :=
  interact ask (G.pmD H x ν r) (G.k x) []

/-- The run opened every coin. -/
def compT (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out)) : Bool :=
  (G.cls H x ν τ).isSome

/-- The simulator's rewind from the completed run `τ₀`: the SHVZK simulator's messages at its coins, sent by the forced
prover. -/
def rwST (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce) (τ₀ : List (Msg × Out))
    (s : G.SimRand x) : List (Msg × Out) :=
  match G.cls H x ν τ₀ with
  | some c => (G.fRun H ask x ν c ((G.sim x ν c s).eval H).1).2
  | none => τ₀

/-- Hybrid 1's rewind from `τ₀`: the honest prover's messages at its coins on a fresh tape, sent by the forced prover. -/
def rwRT (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (w : W) (ν : G.Nonce)
    (τ₀ : List (Msg × Out)) (r : G.Rand) : List (Msg × Out) :=
  match G.cls H x ν τ₀ with
  | some c => (G.fRun H ask x ν c (G.realVec H x w ν r c)).2
  | none => interact ask (G.pmR H x w ν r) (G.k x) []

/-- A binding failure: the rewind `τ` opened one of the coins `τ₀` opened as another. -/
def badT (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (τ₀ τ : List (Msg × Out)) : Bool :=
  match G.cls H x ν τ₀ with
  | some c => decide (G.fcls H x ν c τ = .mism)
  | none => false

theorem rwST_of_cls (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    {τ₀ : List (Msg × Out)} {c : Fin (G.K x) → G.Coin} (h : G.cls H x ν τ₀ = some c) (s : G.SimRand x) :
    G.rwST H ask x ν τ₀ s = (G.fRun H ask x ν c ((G.sim x ν c s).eval H).1).2 := by
  simp [rwST, h]

theorem rwRT_of_cls (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (w : W) (ν : G.Nonce)
    {τ₀ : List (Msg × Out)} {c : Fin (G.K x) → G.Coin} (h : G.cls H x ν τ₀ = some c) (r : G.Rand) :
    G.rwRT H ask x w ν τ₀ r = (G.fRun H ask x ν c (G.realVec H x w ν r c)).2 := by
  simp [rwRT, h]

theorem badT_of_cls (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) {τ₀ : List (Msg × Out)}
    {c : Fin (G.K x) → G.Coin} (h : G.cls H x ν τ₀ = some c) (τ : List (Msg × Out)) :
    G.badT H x ν τ₀ τ = decide (G.fcls H x ν c τ = .mism) := by
  simp [badT, h]

/-- Away from a binding failure, a rewind at `c` opens every coin exactly when it opens all of `c`. -/
theorem compT_iff (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (τ : List (Msg × Out)) (h : G.fcls H x ν c τ ≠ .mism) : G.compT H x ν τ = true ↔ G.fcls H x ν c τ = .succ := by
  unfold compT
  constructor
  · intro hs
    obtain ⟨c', hc'⟩ := Option.isSome_iff_exists.mp hs
    have := G.fcls_ne_abort_of_cls H x ν c c' τ hc'
    cases hf : G.fcls H x ν c τ <;> simp_all
  · intro hs
    rw [(G.cls_eq_some_iff_fcls H x ν c τ).mpr hs]
    rfl

theorem inc_view0 (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out))
    (h : G.compT H x ν τ = false) : Sum.elim id (fun _ => []) (G.view0 H x ν τ) = τ := by
  unfold compT at h
  unfold view0
  cases hc : G.cls H x ν τ with
  | none => rfl
  | some _ => rw [hc] at h; cases h

theorem compT_view0 (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (τ : List (Msg × Out)) :
    G.compT H x ν τ = (G.view0 H x ν τ).isRight := by
  unfold compT view0
  cases G.cls H x ν τ <;> rfl

/-- **The session analysis's `Model` of the protocol at the nonce `ν`**, against the verifier's answers `ask`: the
real and dummy-witness runs; the simulator's rewinds and hybrid 1's, at the coins the first run opened; a binding
failure; and the extraction run's view (`view0`). `bind_real` is `real_eq_of_not_mism`. -/
def model (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (w : W) (ν : G.Nonce) :
    FlockSoundness.ZK.GK.Model (List (Msg × Out)) (List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin)) G.Rand G.Rand
      (G.SimRand x) where
  real r := interact ask (G.pmR H x w ν r) (G.k x) []
  dummy := G.dumT H ask x ν
  rwS := G.rwST H ask x ν
  rwR := G.rwRT H ask x w ν
  comp := G.compT H x ν
  bad := G.badT H x ν
  pre := G.view0 H x ν
  compP := Sum.isRight
  inc := Sum.elim id fun _ => []
  comp_eq := G.compT_view0 H x ν
  inc_real _ h := (G.inc_view0 H x ν _ h).symm
  inc_dummy _ h := (G.inc_view0 H x ν _ h).symm
  bind_real ω₀ h ω hb := by
    obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp h
    rw [G.rwRT_of_cls H ask x w ν hc] at hb ⊢
    rw [G.badT_of_cls H x ν hc] at hb
    exact (G.real_eq_of_not_mism H ask x w ν ω c fun e => by simp [fRun, e] at hb).symm

/-- The same model with the dummy-witness run in place of the real one: what the simulator's cost reads, without a
witness. -/
def modelD (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce) :
    FlockSoundness.ZK.GK.Model (List (Msg × Out)) (List (Msg × Out) ⊕ (Fin (G.K x) → G.Coin)) G.Rand G.Rand
      (G.SimRand x) where
  real := G.dumT H ask x ν
  dummy := G.dumT H ask x ν
  rwS := G.rwST H ask x ν
  rwR _ := G.dumT H ask x ν
  comp := G.compT H x ν
  bad := G.badT H x ν
  pre := G.view0 H x ν
  compP := Sum.isRight
  inc := Sum.elim id fun _ => []
  comp_eq := G.compT_view0 H x ν
  inc_real _ h := (G.inc_view0 H x ν _ h).symm
  inc_dummy _ h := (G.inc_view0 H x ν _ h).symm
  bind_real _ _ _ _ := rfl

/-- **Lemma C's distance at the model** is `PrefixHides`. -/
theorem prefinalClose_model (G : Proto W) (H : Msg → Msg) (R : Rel W) (δA : Msg → ℝ) (hA : G.PrefixHides H R δA)
    (x : Msg) (w : W) (hR : R x w) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (hk : Vs.k x = G.k x) (ν : G.Nonce) :
    (G.model H (fun as => ((Vs.next x as θ).eval H).1) x w ν).PrefinalClose (δA x) := by
  intro U
  have := hA x w hR ν Θ Vs θ hk U
  simp only [runStar_msgs, hk] at this
  dsimp only [model, dumT]
  rw [avgR_eq_Ex, avgR_eq_Ex, Ex_ind_comp, Ex_ind_comp]
  exact this

/-- **Lemma B's distance at the model, one rewind** is `SHVZK` at the coins the first run opened. -/
theorem rewindClose_model (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (R : Rel W) (δB : Msg → ℝ)
    (hB : G.SHVZK H R δB) (x : Msg) (w : W) (hR : R x w) (hB0 : 0 ≤ δB x) (ν : G.Nonce) :
    (G.model H ask x w ν).RewindClose (δB x) := by
  refine ⟨hB0, fun ω₀ h U => ?_⟩
  obtain ⟨c, hc⟩ : ∃ c, G.cls H x ν (G.dumT H ask x ν ω₀) = some c := Option.isSome_iff_exists.mp h
  dsimp only [model]
  simp only [G.rwST_of_cls H ask x ν hc, G.rwRT_of_cls H ask x w ν hc]
  rw [avgR_eq_Ex, avgR_eq_Ex, Ex_ind_comp, Ex_ind_comp, abs_sub_comm]
  exact hB x w hR ν c fun v => U (G.fRun H ask x ν c v).2

end Proto

/-! ## The simulator -/

/-- The transcript an outcome outputs: a view's, else the empty transcript. -/
def outT : GOut (List (Msg × Out)) → List (Msg × Out)
  | .view τ => τ
  | _ => []

/-- **An event of the output, from two tests of the outcome**: the outcome's value at `Ev` and at its complement, each
within `e` of `P` and `1 − P`, put the output's probability of `Ev` within `e` of `P`, whichever `Ev` says of the empty
transcript. -/
theorem pr_outT_close {Q : Type} [Fintype Q] [Nonempty Q] (g : Q → GOut (List (Msg × Out)))
    (Ev : List (Msg × Out) → Prop) (P e : ℝ)
    (h1 : |Ex (fun q => FlockSoundness.ZK.GK.Out.val (FlockSoundness.Game.ind Ev) (g q)) - P| ≤ e)
    (h2 : |Ex (fun q => FlockSoundness.ZK.GK.Out.val (FlockSoundness.Game.ind fun τ => ¬ Ev τ) (g q)) - (1 - P)| ≤ e) :
    |Pr (fun q => Ev (outT (g q))) - P| ≤ e := by
  classical
  by_cases hE : Ev []
  · have e1 : (fun q => FlockSoundness.ZK.GK.Out.val (FlockSoundness.Game.ind fun τ => ¬ Ev τ) (g q)) =
        FlockSoundness.Game.ind fun q => ¬ Ev (outT (g q)) := by
      funext q
      cases h : g q <;> simp [FlockSoundness.ZK.GK.Out.val, outT, FlockSoundness.Game.ind, hE, h]
    rw [e1, Ex_ind_eq, Pr_compl] at h2
    rw [abs_le] at h2 ⊢
    constructor <;> linarith [h2.1, h2.2]
  · have e1 : (fun q => FlockSoundness.ZK.GK.Out.val (FlockSoundness.Game.ind Ev) (g q)) =
        FlockSoundness.Game.ind fun q => Ev (outT (g q)) := by
      funext q
      cases h : g q <;> simp [FlockSoundness.ZK.GK.Out.val, outT, FlockSoundness.Game.ind, hE, h]
    rwa [e1, Ex_ind_eq] at h1

namespace Proto

variable {W : Type}

/-- One dummy-witness run on the nonce `ν` and tape `r`: its transcript and the coins it opened. -/
def dRunSP (G : Proto W) (x : Msg) (ν : G.Nonce) (r : G.Rand) :
    SP (List (Msg × Out) × Option (Fin (G.K x) → G.Coin)) :=
  spBind (spRun (fun hist => G.dprover.next x hist (ν, r)) (G.k x) []) fun τ =>
    spBind (ocSP (clsOC (G.K x) (G.chk x ν) G.dec τ)) fun o => .ret (τ, o)

/-- **The estimation**: fresh dummy-witness runs on the nonce `ν`, on the tapes `rs` in order, until `k` of them open
every coin, at most `b`; the number of runs and whether `k` opened every coin. -/
def estSP (G : Proto W) (x : Msg) (ν : G.Nonce) : (b k : ℕ) → (Fin b → G.Rand) → SP (ℕ × Bool)
  | _, 0, _ => .ret (0, true)
  | 0, _ + 1, _ => .ret (0, false)
  | b + 1, k + 1, rs => spBind (G.dRunSP x ν (rs 0)) fun p =>
      spBind (G.estSP x ν b (if p.2.isSome then k else k + 1) fun i => rs i.succ) fun r => .ret (r.1 + 1, r.2)

/-- **The rewinds** at the coins `c` the first run `τ₀` opened, on the SHVZK tapes `ss` in order, at most `n`: the first
that opens all of `c` is a view, one that first opens another coin a binding failure, and none a failure. -/
def rewSP (G : Proto W) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (τ₀ : List (Msg × Out)) :
    (n L : ℕ) → (Fin L → G.SimRand x) → SP (GOut (List (Msg × Out)))
  | 0, _, _ => .ret .fail
  | _ + 1, 0, _ => .ret .fail
  | n + 1, L + 1, ss => spBind (G.rewindSP x ν c (ss 0)) fun p =>
      if p.1 = .mism then .ret (.bind τ₀ p.2) else if p.1 = .succ then .ret (.view p.2)
      else G.rewSP x ν c τ₀ n L fun i => ss i.succ

/-- **The Goldreich–Kahan simulator with threshold `t` and cap `M`** (`S_GK`): its coins are the nonce, the first run's
tape, `M` dummy-witness tapes and `4M` SHVZK tapes. -/
def simulatorE (G : Proto W) (t M : ℕ) : Simulator where
  C x := G.Nonce × G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x)
  prog x c :=
    spBind (G.dRunSP x c.1 c.2.1) fun p =>
    match p.2 with
    | none => .ret p.1
    | some cs => spBind (G.estSP x c.1 M t c.2.2.1) fun r =>
        if r.2 then spBind (G.rewSP x c.1 cs p.1 (4 * r.1) (4 * M) c.2.2.2) fun o => .ret (outT o)
        else .ret []

/-- One run's calls: a simulation, the classification's `K` checks, and `k` rounds of one question to `V*`, the checks of
every opening so far and a message. -/
def runBound (G : Proto W) (cChk cMsg cSim : Msg → ℕ) (x : Msg) : ℕ :=
  cSim x + G.K x * cChk x + G.k x * (1 + G.k x * cChk x + cMsg x)

/-! ## What the simulator computes -/

section Eval

variable (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)

theorem dRunSP_eval (r : G.Rand) :
    (SP.eval H ask (G.dRunSP x ν r)).1 = (G.dumT H ask x ν r, G.cls H x ν (G.dumT H ask x ν r)) := by
  simp only [dRunSP, spBind_eval, ocSP_eval, spRun_eval, SP.eval]
  rfl

theorem estSP_eval : ∀ (b k : ℕ) (rs : Fin b → G.Rand),
    (SP.eval H ask (G.estSP x ν b k rs)).1 = estimateT (fun r => G.compT H x ν (G.dumT H ask x ν r)) b k rs
  | b, 0, _ => by cases b <;> simp [estSP, estimateT, SP.eval]
  | 0, _ + 1, _ => by simp [estSP, estimateT, SP.eval]
  | b + 1, k + 1, rs => by
      simp only [estSP, estimateT, spBind_eval, SP.eval, dRunSP_eval]
      rw [estSP_eval b]
      rfl

theorem rewSP_eval {c : Fin (G.K x) → G.Coin} {τ₀ : List (Msg × Out)} (hc : G.cls H x ν τ₀ = some c) :
    ∀ (n L : ℕ) (ss : Fin L → G.SimRand x), (SP.eval H ask (G.rewSP x ν c τ₀ n L ss)).1 =
      (rewindsT τ₀ (G.rwST H ask x ν τ₀) (G.compT H x ν) (G.badT H x ν τ₀) n L ss).1
  | 0, _, _ => rfl
  | _ + 1, 0, _ => rfl
  | n + 1, L + 1, ss => by
      simp only [rewSP, rewindsT, spBind_eval, rewindSP_eval, G.rwST_of_cls H ask x ν hc, G.badT_of_cls H x ν hc]
      have e : (G.fRun H ask x ν c ((G.sim x ν c (ss 0)).eval H).1).1 =
          G.fcls H x ν c (G.fRun H ask x ν c ((G.sim x ν c (ss 0)).eval H).1).2 := rfl
      rw [e]
      generalize (G.fRun H ask x ν c ((G.sim x ν c (ss 0)).eval H).1).2 = τ
      by_cases hm : G.fcls H x ν c τ = .mism
      · simp [hm, SP.eval]
      · by_cases hs : G.fcls H x ν c τ = .succ
        · simp [hs, (G.compT_iff H x ν c τ hm).mpr hs, SP.eval]
        · have hf : G.compT H x ν τ = false := by
            cases h' : G.compT H x ν τ
            · rfl
            · exact absurd ((G.compT_iff H x ν c τ hm).mp h') hs
          simp only [hm, hs, hf, ite_false, decide_false, Bool.false_eq_true]
          exact rewSP_eval hc n L _

/-- **The simulator's output** is the outcome of `gk` on its tapes, with the protocol's runs. -/
theorem simE_eval (t M : ℕ) (r₀ : G.Rand) (rs : Fin M → G.Rand) (ss : Fin (4 * M) → G.SimRand x) :
    (SP.eval H ask ((G.simulatorE t M).prog x (ν, r₀, rs, ss))).1 =
      outT (gkT (G.dumT H ask x ν) (G.rwST H ask x ν) (G.compT H x ν) (G.badT H x ν) t M r₀ rs ss).1 := by
  simp only [simulatorE, spBind_eval, dRunSP_eval]
  unfold gkT
  cases hc : G.cls H x ν (G.dumT H ask x ν r₀) with
  | none =>
      have h : G.compT H x ν (G.dumT H ask x ν r₀) = false := by simp [compT, hc]
      simp [h, SP.eval, outT]
  | some cs =>
      have h : G.compT H x ν (G.dumT H ask x ν r₀) = true := by simp [compT, hc]
      simp only [h, ite_true, afterT, spBind_eval, estSP_eval]
      split_ifs with h2
      · simp [spBind_eval, SP.eval, G.rewSP_eval H ask x ν hc]
      · simp [SP.eval, outT]

end Eval

/-! ## The simulator's cost -/

section Cost

variable (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)

theorem dRunSP_cost (r : G.Rand) (cChk cMsg cSim : Msg → ℕ) (hc : G.CostLe H cChk cMsg cSim) :
    (SP.eval H ask (G.dRunSP x ν r)).2 ≤ G.runBound cChk cMsg cSim x := by
  obtain ⟨hchk, hmsg, -⟩ := hc
  simp only [dRunSP, spBind_eval, ocSP_eval, SP.eval]
  have hrun := spRun_cost H ask (fun hist => G.dprover.next x hist (ν, r)) (G.k x) (G.k x * cChk x + cMsg x)
    (fun h hl => by
      have := strat_log_le H (G.hello x ν) G.abort G.dec (G.chk x ν) (G.dmsg x ν r) (cChk x) (cMsg x)
        (fun ρ j m => hchk x ν ρ j m) (fun cs => hmsg x ν r cs) h
      have := Nat.mul_le_mul_right (cChk x) hl.le
      show ((strat (G.hello x ν) G.abort G.dec (G.chk x ν) (G.dmsg x ν r) h).eval H).2.length ≤ _
      omega)
    (G.k x) [] (by simp)
  have hcls := G.clsOC_log_le H x ν (cChk x) (hchk x ν)
    (SP.eval H ask (spRun (fun hist => G.dprover.next x hist (ν, r)) (G.k x) [])).1
  have e : G.k x * (1 + (G.k x * cChk x + cMsg x)) = G.k x * (1 + G.k x * cChk x + cMsg x) := by ring
  unfold runBound
  omega

theorem rewindSP_cost' (c : Fin (G.K x) → G.Coin) (s : G.SimRand x) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) : (SP.eval H ask (G.rewindSP x ν c s)).2 ≤ G.runBound cChk cMsg cSim x := by
  obtain ⟨hchk, -, hsim⟩ := hc
  have h := G.rewindSP_cost H ask x ν c s (cChk x) (cSim x) (hchk x ν) (hsim x ν c s)
  have := Nat.mul_le_mul_left (G.k x) (show 1 + G.k x * cChk x ≤ 1 + G.k x * cChk x + cMsg x by omega)
  unfold runBound
  omega

theorem estSP_cost (B : ℕ) (hB : ∀ r, (SP.eval H ask (G.dRunSP x ν r)).2 ≤ B) : ∀ (b k : ℕ) (rs : Fin b → G.Rand),
    (SP.eval H ask (G.estSP x ν b k rs)).2 ≤ (estimateT (fun r => G.compT H x ν (G.dumT H ask x ν r)) b k rs).1 * B
  | b, 0, _ => by cases b <;> simp [estSP, estimateT, SP.eval]
  | 0, _ + 1, _ => by simp [estSP, estimateT, SP.eval]
  | b + 1, k + 1, rs => by
      simp only [estSP, estimateT, spBind_eval, SP.eval, dRunSP_eval]
      have h1 := hB (rs 0)
      rw [Nat.add_mul, one_mul]
      cases hc : (G.cls H x ν (G.dumT H ask x ν (rs 0))).isSome
      · have h2 := estSP_cost B hB b (k + 1) fun i => rs i.succ
        simp only [compT, hc, Bool.false_eq_true, ite_false] at h2 ⊢
        omega
      · have h2 := estSP_cost B hB b k fun i => rs i.succ
        simp only [compT, hc, ite_true] at h2 ⊢
        omega

theorem rewSP_cost {c : Fin (G.K x) → G.Coin} {τ₀ : List (Msg × Out)} (hc : G.cls H x ν τ₀ = some c) (B : ℕ)
    (hB : ∀ s, (SP.eval H ask (G.rewindSP x ν c s)).2 ≤ B) : ∀ (n L : ℕ) (ss : Fin L → G.SimRand x),
    (SP.eval H ask (G.rewSP x ν c τ₀ n L ss)).2 ≤
      (rewindsT τ₀ (G.rwST H ask x ν τ₀) (G.compT H x ν) (G.badT H x ν τ₀) n L ss).2 * B
  | 0, _, _ => by simp [rewSP, rewindsT, SP.eval]
  | _ + 1, 0, _ => by simp [rewSP, rewindsT, SP.eval]
  | n + 1, L + 1, ss => by
      have h1 := hB (ss 0)
      simp only [rewSP, rewindsT, spBind_eval, rewindSP_eval, G.rwST_of_cls H ask x ν hc, G.badT_of_cls H x ν hc]
      have e : (G.fRun H ask x ν c ((G.sim x ν c (ss 0)).eval H).1).1 =
          G.fcls H x ν c (G.fRun H ask x ν c ((G.sim x ν c (ss 0)).eval H).1).2 := rfl
      rw [e]
      generalize (G.fRun H ask x ν c ((G.sim x ν c (ss 0)).eval H).1).2 = τ
      by_cases hm : G.fcls H x ν c τ = .mism
      · simp [hm, SP.eval]
        omega
      · by_cases hs : G.fcls H x ν c τ = .succ
        · simp [hs, (G.compT_iff H x ν c τ hm).mpr hs, SP.eval]
          omega
        · have hf : G.compT H x ν τ = false := by
            cases h' : G.compT H x ν τ
            · rfl
            · exact absurd ((G.compT_iff H x ν c τ hm).mp h') hs
          simp only [hm, hs, hf, ite_false, decide_false, Bool.false_eq_true]
          have := rewSP_cost hc B hB n L fun i => ss i.succ
          rw [Nat.add_mul, one_mul]
          omega

/-- **The simulator's calls are at most its runs' count times one run's.** -/
theorem simE_cost (t M : ℕ) (r₀ : G.Rand) (rs : Fin M → G.Rand) (ss : Fin (4 * M) → G.SimRand x)
    (cChk cMsg cSim : Msg → ℕ) (hc : G.CostLe H cChk cMsg cSim) :
    (SP.eval H ask ((G.simulatorE t M).prog x (ν, r₀, rs, ss))).2 ≤
      (gkT (G.dumT H ask x ν) (G.rwST H ask x ν) (G.compT H x ν) (G.badT H x ν) t M r₀ rs ss).2 *
        G.runBound cChk cMsg cSim x := by
  have h0 := G.dRunSP_cost H ask x ν r₀ cChk cMsg cSim hc
  simp only [simulatorE, spBind_eval, dRunSP_eval]
  unfold gkT
  cases hcl : G.cls H x ν (G.dumT H ask x ν r₀) with
  | none =>
      have h : G.compT H x ν (G.dumT H ask x ν r₀) = false := by simp [compT, hcl]
      simp [h, SP.eval]
      omega
  | some cs =>
      have h : G.compT H x ν (G.dumT H ask x ν r₀) = true := by simp [compT, hcl]
      have he := G.estSP_cost H ask x ν _ (fun r => G.dRunSP_cost H ask x ν r cChk cMsg cSim hc) M t rs
      simp only [h, ite_true, afterT, spBind_eval, estSP_eval]
      set e := estimateT (fun r => G.compT H x ν (G.dumT H ask x ν r)) M t rs
      split_ifs with h2
      · have hr := G.rewSP_cost H ask x ν hcl _ (fun s => G.rewindSP_cost' H ask x ν cs s cChk cMsg cSim hc)
          (4 * e.1) (4 * M) ss
        simp only [spBind_eval, SP.eval, Nat.add_zero]
        rw [Nat.add_mul, Nat.add_mul, one_mul]
        omega
      · simp only [SP.eval, Nat.add_zero]
        rw [Nat.add_mul, one_mul]
        omega

end Cost

/-! ## A binding failure in the simulator is a collision among its calls -/

theorem spLog_bind_left {α β : Type} (H : Msg → Msg) (ask : List Msg → Out) (p : SP α) (f : α → SP β) :
    spLog H ask p ⊆ spLog H ask (spBind p f) := by
  rw [spBind_log]
  exact List.subset_append_left _ _

theorem spLog_bind_right {α β : Type} (H : Msg → Msg) (ask : List Msg → Out) (p : SP α) (f : α → SP β) :
    spLog H ask (f (SP.eval H ask p).1) ⊆ spLog H ask (spBind p f) := by
  rw [spBind_log]
  exact List.subset_append_right _ _

theorem rewSP_bind (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (τ₀ : List (Msg × Out)) : ∀ (n L : ℕ) (ss : Fin L → G.SimRand x) (v₀ v : List (Msg × Out)),
    (SP.eval H ask (G.rewSP x ν c τ₀ n L ss)).1 = .bind v₀ v →
      ∃ i : Fin L, (SP.eval H ask (G.rewindSP x ν c (ss i))).1 = (.mism, v) ∧
        spLog H ask (G.rewindSP x ν c (ss i)) ⊆ spLog H ask (G.rewSP x ν c τ₀ n L ss)
  | 0, _, _, _, _, h => by simp [rewSP, SP.eval] at h
  | _ + 1, 0, _, _, _, h => by simp [rewSP, SP.eval] at h
  | n + 1, L + 1, ss, v₀, v, h => by
      simp only [rewSP, spBind_eval, spBind_log] at h ⊢
      generalize hp : (SP.eval H ask (G.rewindSP x ν c (ss 0))).1 = p at h ⊢
      by_cases hm : p.1 = .mism
      · simp only [hm, ite_true, SP.eval, FlockSoundness.ZK.GK.Out.bind.injEq] at h
        obtain ⟨-, rfl⟩ := h
        exact ⟨0, hp.trans (Prod.ext hm rfl), List.subset_append_left _ _⟩
      · by_cases hs : p.1 = .succ
        · simp [hs, SP.eval] at h
        · simp only [hm, hs, ite_false] at h ⊢
          obtain ⟨i, h1, h2⟩ := rewSP_bind G H ask x ν c τ₀ n L _ v₀ v h
          exact ⟨i.succ, h1, List.Subset.trans h2 (List.subset_append_right _ _)⟩

/-- **A binding failure in the simulator is a collision among its own calls**: the first run's check of a coin's
opening and the failing rewind's check of the other opening, against one commitment. -/
theorem simE_coll (G : Proto W) (H : Msg → Msg) (t M : ℕ) {Θ : Type} (Vs : VStar Θ) (θ : Θ) (x : Msg) (ν : G.Nonce)
    (r₀ : G.Rand) (rs : Fin M → G.Rand) (ss : Fin (4 * M) → G.SimRand x) (hC : G.Binding H) (v₀ v : List (Msg × Out))
    (hb : (gkT (G.dumT H (fun as => ((Vs.next x as θ).eval H).1) x ν)
      (G.rwST H (fun as => ((Vs.next x as θ).eval H).1) x ν) (G.compT H x ν) (G.badT H x ν) t M r₀ rs ss).1 =
        .bind v₀ v) :
    findColl (simLog H (G.simulatorE t M) Vs θ x (ν, r₀, rs, ss)) ≠ none := by
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  unfold gkT at hb
  set τD := G.dumT H ask x ν r₀ with hτD
  cases hcl : G.cls H x ν τD with
  | none =>
      have h : G.compT H x ν τD = false := by simp [compT, hcl]
      rw [ite_eq_right (by simp [h])] at hb
      cases hb
  | some c =>
      have h : G.compT H x ν τD = true := by simp [compT, hcl]
      rw [ite_eq_left h] at hb
      unfold afterT at hb
      set e := estimateT (fun r => G.compT H x ν (G.dumT H ask x ν r)) M t rs with he
      by_cases h2 : e.2 = true
      swap
      · rw [ite_eq_right h2] at hb
        cases hb
      rw [ite_eq_left h2] at hb
      rw [← G.rewSP_eval H ask x ν hcl] at hb
      obtain ⟨i, hi, hlogi⟩ := G.rewSP_bind H ask x ν c τD _ _ ss v₀ v hb
      have hlog : ((clsOC (G.K x) (G.chk x ν) G.dec τD).eval H).2 ++
          spLog H ask (G.rewSP x ν c τD (4 * e.1) (4 * M) ss) ⊆ simLog H (G.simulatorE t M) Vs θ x (ν, r₀, rs, ss) := by
        have hd : (SP.eval H ask (G.dRunSP x ν r₀)).1 = (τD, some c) := by
          rw [G.dRunSP_eval H ask x ν r₀, ← hτD, hcl]
        have he' : (SP.eval H ask (G.estSP x ν M t rs)).1 = e := G.estSP_eval H ask x ν M t rs
        unfold simLog
        rw [← hask]
        refine List.append_subset.mpr ⟨?_, ?_⟩
        · refine List.Subset.trans ?_ (spLog_bind_left H ask (G.dRunSP x ν r₀) _)
          unfold dRunSP
          refine List.Subset.trans ?_ (spLog_bind_right H ask _ _)
          refine List.Subset.trans ?_ (spLog_bind_left H ask _ _)
          rw [ocSP_log, spRun_eval]
          exact List.Subset.refl _
        · refine List.Subset.trans ?_ (spLog_bind_right H ask (G.dRunSP x ν r₀) _)
          rw [hd]
          refine List.Subset.trans ?_ (spLog_bind_right H ask (G.estSP x ν M t rs) _)
          rw [he', ite_eq_left h2]
          exact spLog_bind_left H ask _ _
      rw [rewindSP_eval] at hi
      set sv := ((G.sim x ν c (ss i)).eval H).1 with hsv
      set τF := interact ask (G.pmF H x ν c sv) (G.k x) [] with hτF
      have hmis : G.fcls H x ν c τF = .mism := congrArg Prod.fst hi
      obtain ⟨ρ', t', m, ht, h0', hr, hb', hc'⟩ := G.fcls_mism_inv H x ν c τF hmis
      have hlogF : ((fclsOC (G.K x) (G.chk x ν) G.dec c τF).eval H).2 ⊆ spLog H ask (G.rewindSP x ν c (ss i)) := by
        simp only [rewindSP, spBind_log, ocSP_eval, ocSP_log, spRun_eval]
        intro p hp
        simp only [List.mem_append, spLog, List.append_nil]
        exact Or.inr (Or.inr hp)
      obtain ⟨ρ, h0, hall⟩ := (G.cls_eq_some_iff H x ν τD c).mp hcl
      obtain ⟨m₀, e1, e2, e3⟩ := hall t' ht
      have hρ : ρ' = ρ := by
        have := resp_zero_eq ask (G.pmF H x ν c sv) (G.pmD H x ν r₀) (G.K x + 1) (by rw [pmF_nil, pmD_nil])
        have e : resp τF 0 = resp τD 0 := this
        rw [h0', h0] at e
        exact Option.some.inj e
      subst hρ
      have hlog₀ := G.cls_log H x ν τD c hcl ρ' h0 t' ht m₀ e1
      have hdec : G.dec m₀ ≠ G.dec m := fun e => hc'.1 (by rw [e3, e])
      have hcoll := hC x ν ρ' t' m₀ m e2 hb' hdec
      rw [findColl_ne_none_iff] at hcoll ⊢
      obtain ⟨p, hp, q, hq, hne, heq⟩ := hcoll
      have sub : ((G.chk x ν ρ' t' m₀).eval H).2 ++ ((G.chk x ν ρ' t' m).eval H).2 ⊆
          simLog H (G.simulatorE t M) Vs θ x (ν, r₀, rs, ss) := by
        refine List.append_subset.mpr ⟨?_, ?_⟩
        · exact List.Subset.trans hlog₀ (List.Subset.trans (List.subset_append_left _ _) hlog)
        · exact List.Subset.trans hc'.2 (List.Subset.trans hlogF
            (List.Subset.trans hlogi (List.Subset.trans (List.subset_append_right _ _) hlog)))
      exact ⟨p, sub hp, q, sub hq, hne, heq⟩

/-! ## The theorems -/

/-- **The Goldreich–Kahan simulator with estimation meets `ZKmalR`** at
`ε_zk(x) = 3·δ_A(x) + 8t·δ_B(x) + 3^{-t} + t/M`, for a protocol whose dummy-witness run hides the witness until the
last coin, whose messages at a fixed coin vector are simulated and whose opening check binds. -/
theorem gkE_zk (G : Proto W) (H : Msg → Msg) (R : Rel W) (t M : ℕ) (hM : 0 < M) (δA δB : Msg → ℝ)
    (hA : G.PrefixHides H R δA) (hB : G.SHVZK H R δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    ZKmalR H G.k G.prover R (G.simulatorE t M) fun x => 3 * δA x + 8 * t * δB x + (1 / 3) ^ t + t / M := by
  intro x w hR Θ Vs θ hk Ev
  beta_reduce
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  set ε : ℝ := 3 * δA x + 8 * t * δB x + (1 / 3) ^ t + t / M with hε
  have key : ∀ ν : G.Nonce,
      |Pr (fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) -
          Pr (fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
            Ev ((G.simulatorE t M).out H Vs θ x (ν, q)))| ≤
        ε + Pr (fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
          findColl (simLog H (G.simulatorE t M) Vs θ x (ν, q)) ≠ none) := by
    intro ν
    set g : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) → GOut (List (Msg × Out)) × ℕ :=
      fun q => gkT (G.dumT H ask x ν) (G.rwST H ask x ν) (G.compT H x ν) (G.badT H x ν) t M q.1 q.2.1 q.2.2 with hg
    have hBm := G.rewindClose_model H ask R δB hB x w hR (hB0 x) ν
    have hCm := G.prefinalClose_model H R δA hA x w hR Vs θ hk ν
    have hev : ∀ f : GOut (List (Msg × Out)) × ℕ → ℝ,
        FlockSoundness.ZK.GK.Rand.ev f ((G.model H ask x w ν).sim t M) = Ex fun q => f (g q) :=
      fun f => (ev_gkT _ _ _ _ t M f).symm
    have g1 := (G.model H ask x w ν).gk_close hBm hCm t M hM (FlockSoundness.Game.ind Ev)
      (FlockSoundness.Game.ind_nonneg _) (FlockSoundness.Game.ind_le_one _)
    have g2 := (G.model H ask x w ν).gk_close hBm hCm t M hM (FlockSoundness.Game.ind fun τ => ¬ Ev τ)
      (FlockSoundness.Game.ind_nonneg _) (FlockSoundness.Game.ind_le_one _)
    unfold FlockSoundness.ZK.GK.Rand.pr at g1 g2
    rw [hev, hev] at g1 g2
    have eR1 : FlockSoundness.ZK.GK.avgR (fun r => FlockSoundness.Game.ind Ev ((G.model H ask x w ν).real r)) =
        Pr (fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) := by
      show Ex _ = _
      rw [ind_comp, Ex_ind_eq]
      rfl
    have eR2 : FlockSoundness.ZK.GK.avgR
        (fun r => FlockSoundness.Game.ind (fun τ => ¬ Ev τ) ((G.model H ask x w ν).real r)) =
        1 - Pr (fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) := by
      show Ex _ = _
      rw [ind_comp, Ex_ind_eq, Pr_compl]
      rfl
    rw [eR1] at g1
    rw [eR2] at g2
    have hbind : Ex (fun q => FlockSoundness.Game.ind
          (fun o => FlockSoundness.ZK.GK.Out.bindBy (G.model H ask x w ν).bad o.1) (g q)) ≤
        Pr (fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
          findColl (simLog H (G.simulatorE t M) Vs θ x (ν, q)) ≠ none) := by
      rw [ind_comp, Ex_ind_eq]
      refine Pr_mono fun q hq => ?_
      obtain ⟨v₀, v, hb⟩ : ∃ v₀ v, (g q).1 = .bind v₀ v := by
        revert hq
        cases (g q).1 with
        | view _ => intro hq; exact hq.elim
        | bind v₀ v => intro _; exact ⟨v₀, v, rfl⟩
        | fail => intro hq; exact hq.elim
      exact G.simE_coll H t M Vs θ x ν q.1 q.2.1 q.2.2 hC v₀ v hb
    have eS : Pr (fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
        Ev ((G.simulatorE t M).out H Vs θ x (ν, q))) = Pr fun q => Ev (outT (g q).1) :=
      Pr_congr fun q => Iff.of_eq (congrArg Ev (G.simE_eval H ask x ν t M q.1 q.2.1 q.2.2))
    have hout := pr_outT_close (fun q => (g q).1) Ev
      (Pr (fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) [])))
      (ε + Ex (fun q => FlockSoundness.Game.ind
          (fun o => FlockSoundness.ZK.GK.Out.bindBy (G.model H ask x w ν).bad o.1) (g q))) g1 g2
    rw [eS, abs_sub_comm]
    linarith
  have eR : Pr (fun ω : G.Nonce × G.Rand => Ev (runStar H (G.prover w) Vs x ω θ).msgs) =
      Ex fun ν => Pr fun r : G.Rand => Ev (interact ask (G.pmR H x w ν r) (G.k x) []) := by
    rw [Pr_prod_left]
    congr 1
    funext ν
    congr 1
    funext r
    rw [runStar_msgs, hk]
    rfl
  have eS : Pr (fun c : (G.simulatorE t M).C x => Ev ((G.simulatorE t M).out H Vs θ x c)) =
      Ex fun ν => Pr fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
        Ev ((G.simulatorE t M).out H Vs θ x (ν, q)) :=
    Pr_prod_left (α := G.Nonce) (β := G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x))
      fun c => Ev ((G.simulatorE t M).out H Vs θ x c)
  have eC : Pr (fun c : (G.simulatorE t M).C x => findColl (simLog H (G.simulatorE t M) Vs θ x c) ≠ none) =
      Ex fun ν => Pr fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
        findColl (simLog H (G.simulatorE t M) Vs θ x (ν, q)) ≠ none :=
    Pr_prod_left (α := G.Nonce) (β := G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x))
      fun c => findColl (simLog H (G.simulatorE t M) Vs θ x c) ≠ none
  rw [eR, eS, eC, ← Ex_const_add]
  exact Ex_abs_sub_le key

/-- **At the cap `M = t·3^t`**: `ε_zk(x) = 3·δ_A(x) + 8t·δ_B(x) + 2·3^{-t}`. -/
theorem gkE_zk_pow (G : Proto W) (H : Msg → Msg) (R : Rel W) (t : ℕ) (ht : 0 < t) (δA δB : Msg → ℝ)
    (hA : G.PrefixHides H R δA) (hB : G.SHVZK H R δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    ZKmalR H G.k G.prover R (G.simulatorE t (t * 3 ^ t)) fun x => 3 * δA x + 8 * t * δB x + 2 * (1 / 3) ^ t := by
  have e : ((t : ℝ) / ((t * 3 ^ t : ℕ) : ℝ)) = (1 / 3) ^ t := by
    have : (0 : ℝ) < t := by exact_mod_cast ht
    push_cast
    rw [div_mul_eq_div_div, div_self this.ne', one_div_pow]
  intro x w hR Θ Vs θ hk Ev
  have := G.gkE_zk H R t (t * 3 ^ t) (by positivity) δA δB hA hB hB0 hC x w hR Θ Vs θ hk Ev
  simp only [e] at this
  linarith

/-- **The simulator's expected cost**: at most `(1 + 5t)·runBound` calls in expectation over its coins, against every
verifier, at every cap `M`. -/
theorem gkE_costExp (G : Proto W) (H : Msg → Msg) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCostExp H (G.simulatorE t M) fun x => (1 + 5 * t) * G.runBound cChk cMsg cSim x := by
  intro x ask
  show Ex (fun c : G.Nonce × (G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x)) =>
    (((SP.eval H ask ((G.simulatorE t M).prog x c)).2 : ℕ) : ℝ)) ≤ _
  rw [Ex_prod]
  refine le_trans (Ex_mono fun ν => ?_) (Ex_const _).le
  calc Ex (fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
        (((SP.eval H ask ((G.simulatorE t M).prog x (ν, q))).2 : ℕ) : ℝ))
      ≤ Ex fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
          (((gkT (G.dumT H ask x ν) (G.rwST H ask x ν) (G.compT H x ν) (G.badT H x ν) t M q.1 q.2.1 q.2.2).2 : ℕ) :
            ℝ) * (G.runBound cChk cMsg cSim x : ℝ) :=
        Ex_mono fun q => by exact_mod_cast G.simE_cost H ask x ν t M q.1 q.2.1 q.2.2 cChk cMsg cSim hc
    _ = FlockSoundness.ZK.GK.Rand.ev (fun o => (o.2 : ℝ)) ((G.modelD H ask x ν).sim t M) *
          (G.runBound cChk cMsg cSim x : ℝ) := by
        rw [Ex_mul_right]
        congr 1
        exact ev_gkT _ _ _ _ t M fun o => (o.2 : ℝ)
    _ ≤ (1 + 5 * t) * (G.runBound cChk cMsg cSim x : ℝ) :=
        mul_le_mul_of_nonneg_right ((G.modelD H ask x ν).sim_runs_ev t M) (Nat.cast_nonneg _)

/-- **The simulator's worst-case cost**: at most `(1 + 5M)·runBound` calls on every coin, against every verifier. -/
theorem gkE_cost (G : Proto W) (H : Msg → Msg) (t M : ℕ) (cChk cMsg cSim : Msg → ℕ)
    (hc : G.CostLe H cChk cMsg cSim) :
    SimCost H (G.simulatorE t M) fun x => (1 + 5 * M) * G.runBound cChk cMsg cSim x := by
  rintro x ⟨ν, r₀, rs, ss⟩ ask
  exact (G.simE_cost H ask x ν t M r₀ rs ss cChk cMsg cSim hc).trans
    (Nat.mul_le_mul_right _ (gkT_runs_le _ _ _ _ t M r₀ rs ss))

end Proto

end Claims.ZKMal
