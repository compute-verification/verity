module

public import Proofs.Flock.Claims.ZK.Cut
public import Proofs.Flock.Claims.ZK.Prefix

/-!
# The Goldreich–Kahan simulator, one-sided: where the prover goes on at its last message

`Proto.gkE_zk` bounds every event of the real run against the simulator's output from SHVZK, which holds both ways.
A prover that sends the `abort` as its last message where its values fail a check meets SHVZK only one way: its
messages at a coin vector are at most the simulator's on every test that does not grow when the last message becomes
`abort` (`Proto.SHVZKLe`). That is enough for the event that the run goes on at its last round (`goesOnAt`) and does `Ev`:

* `rewinds_le`, `Model.hyb₁_le_sim`: one rewind's value bounded above bounds hybrid 1's against the simulator's, at
  `4t·δ_B`, as `sim_hyb₁` bounds it both ways;
* `Proto.rewindLe_model`: at the protocol's model, one rewind's value is such a test, since a forced run whose last
  message is `abort` reads the same coins (`fcls_take`, `cls_take`) and does not go on (`goesOnAt_set`);
* `Proto.gkE_go`: `Pr[goesOnAt ∧ Ev]` of the real run is at most `Pr[Ev]` of the simulator's output plus
  `4·δ_A + 4t·δ_B + 3^{-t} + t/M` and `4M` times `Fork`'s probability: hybrid 2 (`hyb₂_real`) and hybrid 1's binding
  failure (`hyb₁_hyb₂`, `Proto.hyb₁_bind_le`) as in `gkE_zk`, then `hyb₁_le_sim`; `Proto.gkE_go_pow` at `M = t·3^t`.
-/

@[expose] public section

namespace FlockSoundness.ZK.GK

open Finset Game

section OneSided

variable {V : Type}

/-- **One rewind's value**, given the value `a` of the rewinds after it: `0` at a binding failure, `φ` at a completed
view, `a` otherwise. -/
def rwVal (comp bad : V → Bool) (φ : V → ℝ) (a : ℝ) (v : V) : ℝ :=
  if bad v = true then 0 else if comp v = true then φ v else a

theorem rwVal_nonneg {comp bad : V → Bool} {φ : V → ℝ} (h0 : ∀ v, 0 ≤ φ v) {a : ℝ} (ha : 0 ≤ a) (v : V) :
    0 ≤ rwVal comp bad φ a v := by
  unfold rwVal
  split_ifs
  exacts [le_rfl, h0 v, ha]

theorem rwVal_le_one {comp bad : V → Bool} {φ : V → ℝ} (h1 : ∀ v, φ v ≤ 1) {a : ℝ} (ha : a ≤ 1) (v : V) :
    rwVal comp bad φ a v ≤ 1 := by
  unfold rwVal
  split_ifs
  exacts [zero_le_one, h1 v, ha]

/-- **One run's value bounded above by `δ` bounds `n` rewinds' by `n·δ`**, for the value of `φ` at the outcome. -/
theorem rewinds_le {C₁ C₂ : Type} [Fintype C₁] [Nonempty C₁] [Fintype C₂] [Nonempty C₂] (v₀ : V) (rw₁ : C₁ → V)
    (rw₂ : C₂ → V) (comp bad : V → Bool) {φ : V → ℝ} (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) {δ : ℝ} (hδ : 0 ≤ δ)
    (h : ∀ a, 0 ≤ a → a ≤ 1 →
      avgR (fun σ => rwVal comp bad φ a (rw₁ σ)) ≤ avgR (fun σ => rwVal comp bad φ a (rw₂ σ)) + δ) :
    ∀ n : ℕ, Rand.ev (fun o => Out.val φ o.1) (rewinds C₁ v₀ rw₁ comp bad n) ≤
      Rand.ev (fun o => Out.val φ o.1) (rewinds C₂ v₀ rw₂ comp bad n) + n * δ
  | 0 => by simp [rewinds, Out.val]
  | n + 1 => by
      rw [ev_rewinds_succ, ev_rewinds_succ]
      have ih := rewinds_le v₀ rw₁ rw₂ comp bad h0 h1 hδ h n
      set c₁ := Rand.ev (fun o : Out V × ℕ => Out.val φ o.1) (rewinds C₁ v₀ rw₁ comp bad n) with hc₁
      set c₂ := Rand.ev (fun o : Out V × ℕ => Out.val φ o.1) (rewinds C₂ v₀ rw₂ comp bad n) with hc₂
      have hc0 : 0 ≤ c₁ := Rand.ev_nonneg (fun o : Out V × ℕ => Out.val_nonneg h0 o.1) _
      have hc1 : c₁ ≤ 1 := Rand.ev_le_one (fun o : Out V × ℕ => Out.val_le_one h1 o.1) _
      have hn : (0 : ℝ) ≤ n * δ := mul_nonneg (Nat.cast_nonneg _) hδ
      show avgR (fun σ => rwVal comp bad φ c₁ (rw₁ σ)) ≤
        avgR (fun σ => rwVal comp bad φ c₂ (rw₂ σ)) + ((n + 1 : ℕ) : ℝ) * δ
      calc avgR (fun σ => rwVal comp bad φ c₁ (rw₁ σ))
          ≤ avgR (fun σ => rwVal comp bad φ c₁ (rw₂ σ)) + δ := h c₁ hc0 hc1
        _ ≤ avgR (fun σ => rwVal comp bad φ c₂ (rw₂ σ) + n * δ) + δ := by
            refine add_le_add (avgR_mono fun σ => ?_) le_rfl
            unfold rwVal
            split_ifs <;> linarith
        _ = _ := by
            rw [avgR_add, avgR_const]
            push_cast
            ring

end OneSided

namespace Model

variable {V P Ω Ωd Ωs : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ωd] [Nonempty Ωd] [Fintype Ωs] [Nonempty Ωs]
  (m : Model V P Ω Ωd Ωs)

/-- **Lemma B one-sided, one rewind**: from every completed dummy view, hybrid 1's rewind is at most the simulator's
plus `δ_B` on one rewind's value of `φ`, whatever the value in `[0, 1]` of the rewinds after it. -/
def RewindLe (φ : V → ℝ) (δB : ℝ) : Prop :=
  0 ≤ δB ∧ ∀ ω₀, m.comp (m.dummy ω₀) = true → ∀ a, 0 ≤ a → a ≤ 1 →
    avgR (fun ω => rwVal m.comp (m.bad (m.dummy ω₀)) φ a (m.rwR (m.dummy ω₀) ω)) ≤
      avgR (fun σ => rwVal m.comp (m.bad (m.dummy ω₀)) φ a (m.rwS (m.dummy ω₀) σ)) + δB

/-- **Hybrid 1 against the simulator, one-sided**: hybrid 1's value of `φ` is at most the simulator's plus `4t·δ_B`. -/
theorem hyb₁_le_sim {φ : V → ℝ} (h0 : ∀ v, 0 ≤ φ v) (h1 : ∀ v, φ v ≤ 1) {δB : ℝ} (hB : m.RewindLe φ δB)
    (t M : ℕ) :
    Rand.ev (fun o => Out.val φ o.1) (m.hyb₁ t M) ≤ Rand.ev (fun o => Out.val φ o.1) (m.sim t M) + 4 * t * δB := by
  obtain ⟨hδ, hB⟩ := hB
  unfold sim hyb₁
  rw [ev_gk, ev_gk]
  try dsimp only
  set est := estimate Ωd (fun ω => m.comp (m.dummy ω)) M t
  have hL : m.pD * Rand.ev (fun r => (r.1 : ℝ)) est ≤ t := estimate_runs Ωd _ M t
  rw [← sub_le_iff_le_add', ← avgR_sub]
  refine (avgR_mono (g := fun ω₀ => if m.comp (m.dummy ω₀) = true then
    4 * δB * Rand.ev (fun r => (r.1 : ℝ)) est else 0) fun ω₀ => ?_).trans ?_
  · try dsimp only
    split_ifs with hc
    · rw [← Rand.ev_sub, ← Rand.ev_const_mul]
      refine Rand.ev_mono (fun r => ?_) est
      try dsimp only
      split_ifs
      · have := rewinds_le (m.dummy ω₀) (m.rwR (m.dummy ω₀)) (m.rwS (m.dummy ω₀)) m.comp (m.bad (m.dummy ω₀)) h0 h1
          hδ (hB ω₀ hc) (4 * r.1)
        push_cast at this ⊢
        linarith
      · rw [sub_self]
        exact mul_nonneg (mul_nonneg (by norm_num) hδ) (Nat.cast_nonneg _)
    · simp
  · rw [avgR_split Ωd (fun ω => m.comp (m.dummy ω))]
    have := mul_le_mul_of_nonneg_left hL (by linarith : (0 : ℝ) ≤ 4 * δB)
    show m.pD * (4 * δB * _) + (1 - m.pD) * 0 ≤ _
    nlinarith

end Model

end FlockSoundness.ZK.GK

namespace Claims.ZKMal

open Claims
open Claims.ZKServer (length_histOf_take interact_getElem_fst)
open FlockSoundness.ZK.GK (rwVal)

local notation "GOut" => FlockSoundness.ZK.GK.Out

/-- **The `Finish` verdict read off a transcript**: whether it reaches round `t` and its message there is not the
stop. -/
def goesOnAt (stop : Msg) (t : ℕ) (τ : List (Msg × Out)) : Bool := (τ[t]?).any fun p => decide (p.1 ≠ stop)

/-- How a rewind at `c` ends reads only the transcript's first `L` rounds once its openings end before `L`. -/
theorem fclsGo_take {Coin : Type} [DecidableEq Coin] (chk : Msg → ℕ → Msg → OC Bool) (dec : Msg → Coin)
    (c : ℕ → Option Coin) (τ : List (Msg × Out)) (ρ : Msg) (L : ℕ) :
    ∀ n j, j + n < L → fclsGo chk dec c (τ.take L) ρ j n = fclsGo chk dec c τ ρ j n
  | 0, _, _ => rfl
  | n + 1, j, h => by
      have ih := fclsGo_take chk dec c τ ρ L n (j + 1) (by omega)
      simp only [fclsGo, resp_take (show j + 1 < L by omega), ih]

namespace Proto

variable {W : Type}

/-- **(HB) one-sided**: at every nonce and coin vector `c`, the honest prover's messages at `c` are at most the
simulator's plus `δ_B(x)` on every test with values in `[0, 1]` that does not grow when the last message becomes
`abort`. -/
def SHVZKLe (G : Proto W) (H : Msg → Msg) (R : Rel W) (δB : Msg → ℝ) : Prop :=
  ∀ x w, R x w → ∀ (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (Ψ : List Msg → ℝ), (∀ v, 0 ≤ Ψ v) → (∀ v, Ψ v ≤ 1) →
    (∀ v, Ψ (v.set (G.K x) G.abort) ≤ Ψ v) →
    Ex (fun r : G.Rand => Ψ (G.realVec H x w ν r c)) ≤ Ex (fun s : G.SimRand x => Ψ ((G.sim x ν c s).eval H).1) + δB x

/-- How a rewind at `c` ends reads only the transcript's first `K + 1` rounds. -/
theorem fcls_take (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (τ : List (Msg × Out)) : G.fcls H x ν c (τ.take (G.K x + 1)) = G.fcls H x ν c τ := by
  simp only [fcls, fclsOC, resp_take (show 0 < G.K x + 1 by omega),
    fclsGo_take _ _ _ τ _ (G.K x + 1) (G.K x) 0 (by omega)]

/-- The forced prover's messages before its last read none of `v`'s last. -/
theorem pmF_set (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin) (v : List Msg)
    (m : Msg) (h : List (Msg × Msg)) (hh : h.length ≤ G.K x) :
    G.pmF H x ν c (v.set (G.K x) m) h = G.pmF H x ν c v h := by
  rcases h with _ | ⟨⟨a, ρ⟩, ops⟩
  · rw [G.pmF_nil H x ν c, G.pmF_nil H x ν c]
  · simp only [List.length_cons] at hh
    simp only [pmF, forced]
    rw [strat_eval_cons, strat_eval_cons]
    simp only [OC.eval, List.length_map, List.getD_eq_getElem?_getD,
      List.getElem?_set_ne (show G.K x ≠ ops.length by omega)]

/-- With `abort` as `v`'s last, the forced prover's last message is `abort`. -/
theorem pmF_set_last (G : Proto W) (H : Msg → Msg) (x : Msg) (ν : G.Nonce) (c : Fin (G.K x) → G.Coin)
    (v : List Msg) (h : List (Msg × Msg)) (hh : h.length = G.K x + 1) :
    G.pmF H x ν c (v.set (G.K x) G.abort) h = G.abort := by
  rcases h with _ | ⟨⟨a, ρ⟩, ops⟩
  · simp at hh
  · simp only [List.length_cons, Nat.add_right_cancel_iff] at hh
    simp only [pmF, forced]
    rw [strat_eval_cons]
    by_cases hK : G.K x < v.length <;>
      simp [OC.eval, hh, List.getD_eq_getElem?_getD, List.getElem?_set, hK]

/-- The forced runs with `v` and with `v`'s last message replaced agree on their first `K + 1` rounds. -/
theorem interact_set_take (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (v : List Msg) (m : Msg) :
    (interact ask (G.pmF H x ν c (v.set (G.K x) m)) (G.k x) []).take (G.K x + 1) =
      (interact ask (G.pmF H x ν c v) (G.k x) []).take (G.K x + 1) :=
  interact_take_congr ask _ _ (G.k x) [] (G.K x) fun i hi _ => by
    simp only [List.nil_append]
    exact G.pmF_set H x ν c v m _ (by rw [length_histOf_take]; exact (min_le_left _ _).trans hi)

/-- **A forced run whose last message is `abort` does not go on at its last round.** -/
theorem goesOnAt_set (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    (c : Fin (G.K x) → G.Coin) (v : List Msg) :
    goesOnAt G.abort (G.K x + 1) (interact ask (G.pmF H x ν c (v.set (G.K x) G.abort)) (G.k x) []) = false := by
  unfold goesOnAt
  by_cases hl : G.K x + 1 < (interact ask (G.pmF H x ν c (v.set (G.K x) G.abort)) (G.k x) []).length
  · have h1 := interact_getElem_fst ask (G.pmF H x ν c (v.set (G.K x) G.abort)) (G.k x) [] (G.K x + 1) hl
    have hlen : (histOf ((interact ask (G.pmF H x ν c (v.set (G.K x) G.abort)) (G.k x) []).take
        (G.K x + 1))).length = G.K x + 1 := by
      rw [length_histOf_take, Nat.min_eq_left hl.le]
    rw [List.nil_append, G.pmF_set_last H x ν c v _ hlen] at h1
    obtain ⟨p, hp, hp1⟩ := Option.map_eq_some_iff.mp h1
    rw [hp]
    simp [hp1]
  · rw [List.getElem?_eq_none (by omega)]
    rfl

/-- **One rewind's value does not grow when the last message becomes `abort`**, for a `φ` that vanishes where the run
does not go on at its last round: the rewind reads the same coins and does not go on. -/
theorem rwVal_set (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (x : Msg) (ν : G.Nonce)
    {τ₀ : List (Msg × Out)} {c : Fin (G.K x) → G.Coin} (hc : G.cls H x ν τ₀ = some c) {φ : List (Msg × Out) → ℝ}
    (h0 : ∀ τ, 0 ≤ φ τ) (hφ : ∀ τ, goesOnAt G.abort (G.K x + 1) τ = false → φ τ = 0) (a : ℝ) (v : List Msg) :
    rwVal (G.compT H x ν) (G.badT H x ν τ₀) φ a (G.fRun H ask x ν c (v.set (G.K x) G.abort)).2 ≤
      rwVal (G.compT H x ν) (G.badT H x ν τ₀) φ a (G.fRun H ask x ν c v).2 := by
  set τ' := interact ask (G.pmF H x ν c (v.set (G.K x) G.abort)) (G.k x) []
  set τ := interact ask (G.pmF H x ν c v) (G.k x) []
  have ht : τ'.take (G.K x + 1) = τ.take (G.K x + 1) := G.interact_set_take H ask x ν c v G.abort
  have hb : G.badT H x ν τ₀ τ' = G.badT H x ν τ₀ τ := by
    rw [G.badT_of_cls H x ν hc, G.badT_of_cls H x ν hc, ← G.fcls_take H x ν c τ', ht, G.fcls_take]
  have hcm : G.compT H x ν τ' = G.compT H x ν τ := by
    unfold compT
    rw [← G.cls_take H x ν τ', ht, G.cls_take]
  have hz : φ τ' = 0 := hφ τ' (G.goesOnAt_set H ask x ν c v)
  show rwVal (G.compT H x ν) (G.badT H x ν τ₀) φ a τ' ≤ rwVal (G.compT H x ν) (G.badT H x ν τ₀) φ a τ
  unfold rwVal
  rw [hb, hcm, hz]
  split_ifs
  exacts [le_rfl, h0 τ, le_rfl]

/-- **Lemma B one-sided at the model, one rewind**, from `SHVZKLe` at the coins the first run opened, for a `φ` that
vanishes where the run does not go on at its last round. -/
theorem rewindLe_model (G : Proto W) (H : Msg → Msg) (ask : List Msg → Out) (R : Rel W) (δB : Msg → ℝ)
    (hB : G.SHVZKLe H R δB) (x : Msg) (w : W) (hR : R x w) (hB0 : 0 ≤ δB x) (ν : G.Nonce)
    {φ : List (Msg × Out) → ℝ} (h0 : ∀ τ, 0 ≤ φ τ) (h1 : ∀ τ, φ τ ≤ 1)
    (hφ : ∀ τ, goesOnAt G.abort (G.K x + 1) τ = false → φ τ = 0) :
    (G.model H ask x w ν).RewindLe φ (δB x) := by
  refine ⟨hB0, fun ω₀ h a ha0 ha1 => ?_⟩
  obtain ⟨c, hc⟩ : ∃ c, G.cls H x ν (G.dumT H ask x ν ω₀) = some c := Option.isSome_iff_exists.mp h
  dsimp only [model]
  simp only [G.rwST_of_cls H ask x ν hc, G.rwRT_of_cls H ask x w ν hc]
  rw [avgR_eq_Ex, avgR_eq_Ex]
  exact hB x w hR ν c
    (fun v => rwVal (G.compT H x ν) (G.badT H x ν (G.dumT H ask x ν ω₀)) φ a (G.fRun H ask x ν c v).2)
    (fun v => FlockSoundness.ZK.GK.rwVal_nonneg h0 ha0 _) (fun v => FlockSoundness.ZK.GK.rwVal_le_one h1 ha1 _)
    (fun v => G.rwVal_set H ask x ν hc h0 hφ a v)

/-- **The Goldreich–Kahan simulator with estimation, one-sided**: for a protocol whose dummy-witness run hides the
witness until the last coin, whose messages at a fixed coin vector are at most the simulator's on every test that does
not grow when the last message becomes `abort`, and whose opening check binds, the real run's probability of going on at
its last round and doing `Ev` is at most the simulator's of `Ev`, plus `4·δ_A + 4t·δ_B + 3^{-t} + t/M` and `4M` times
`Fork`'s probability. -/
theorem gkE_go (G : Proto W) (H : Msg → Msg) (R : Rel W) (t M : ℕ) (hM : 0 < M) (δA δB : Msg → ℝ)
    (hA : G.PrefixHides H R δA) (hB : G.SHVZKLe H R δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = G.k x → ∀ Ev : List (Msg × Out) → Prop,
      Pr (fun ω : G.Nonce × G.Rand => goesOnAt G.abort (G.k x - 1) (runStar H (G.prover w) Vs x ω θ).msgs = true ∧
          Ev (runStar H (G.prover w) Vs x ω θ).msgs) ≤
        Pr (fun c : (G.simulatorE t M).C x => Ev ((G.simulatorE t M).out H Vs θ x c)) +
          (4 * δA x + 4 * t * δB x + (1 / 3) ^ t + t / M + 4 * M * (Fork H (G.prover w) Vs θ x).Pr H) := by
  intro x w hR Θ Vs θ hk Ev
  set ask : List Msg → Out := fun as => ((Vs.next x as θ).eval H).1 with hask
  set ε : ℝ := 4 * δA x + 4 * t * δB x + (1 / 3) ^ t + t / M with hε
  have hK1 : G.k x - 1 = G.K x + 1 := by unfold Proto.k; omega
  set φ : List (Msg × Out) → ℝ :=
    FlockSoundness.Game.ind fun τ => goesOnAt G.abort (G.K x + 1) τ = true ∧ Ev τ with hφ
  have hφ0 : ∀ τ, 0 ≤ φ τ := FlockSoundness.Game.ind_nonneg _
  have hφ1 : ∀ τ, φ τ ≤ 1 := FlockSoundness.Game.ind_le_one _
  have hφz : ∀ τ, goesOnAt G.abort (G.K x + 1) τ = false → φ τ = 0 := fun τ h => by
    simp [hφ, FlockSoundness.Game.ind, h]
  have key : ∀ ν : G.Nonce,
      Pr (fun r : G.Rand => goesOnAt G.abort (G.K x + 1) (interact ask (G.pmR H x w ν r) (G.k x) []) = true ∧
          Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) ≤
        Pr (fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
            Ev ((G.simulatorE t M).out H Vs θ x (ν, q))) +
          (ε + Pr (fun p : G.Rand × G.Rand => G.collR H Vs θ x w ν p.1 p.2) * (4 * M)) := by
    intro ν
    set PC := Pr (fun p : G.Rand × G.Rand => G.collR H Vs θ x w ν p.1 p.2) with hPC
    set g : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) → GOut (List (Msg × Out)) × ℕ :=
      fun q => gkT (G.dumT H ask x ν) (G.rwST H ask x ν) (G.compT H x ν) (G.badT H x ν) t M q.1 q.2.1 q.2.2 with hg
    have hBm := G.rewindLe_model H ask R δB hB x w hR (hB0 x) ν hφ0 hφ1 hφz
    have hCm := G.prefinalClose_model H R δA hA x w hR Vs θ hk ν
    have hev : ∀ f : GOut (List (Msg × Out)) × ℕ → ℝ,
        FlockSoundness.ZK.GK.Rand.ev f ((G.model H ask x w ν).sim t M) = Ex fun q => f (g q) :=
      fun f => (ev_gkT _ _ _ _ t M f).symm
    have hbind : FlockSoundness.ZK.GK.Rand.pr (fun o => FlockSoundness.ZK.GK.Out.bindBy (G.model H ask x w ν).bad o.1)
        ((G.model H ask x w ν).hyb₁ t M) ≤ δA x + PC * (4 * M) :=
      G.hyb₁_bind_le H R δA hA hC x w hR Vs θ hk ν t M
    have s1 := (G.model H ask x w ν).hyb₁_le_sim hφ0 hφ1 hBm t M
    have s2 := (G.model H ask x w ν).hyb₁_hyb₂ t M (fun o => FlockSoundness.ZK.GK.Out.val φ o.1)
      (fun o => FlockSoundness.ZK.GK.Out.val_nonneg hφ0 o.1) (fun o => FlockSoundness.ZK.GK.Out.val_le_one hφ1 o.1)
    have s3 := (G.model H ask x w ν).hyb₂_real hCm t M hM φ hφ0 hφ1
    rw [hev] at s1
    have s4 : Ex (fun q => FlockSoundness.ZK.GK.Out.val φ (g q).1) ≤ Pr fun q => Ev (outT (g q).1) := by
      rw [← Ex_ind_eq]
      refine Ex_mono fun q => ?_
      show FlockSoundness.ZK.GK.Out.val φ (g q).1 ≤ FlockSoundness.Game.ind (fun o => Ev (outT o)) (g q).1
      generalize (g q).1 = o
      cases o with
      | view τ =>
          exact FlockSoundness.Game.ind_mono (p := fun τ => goesOnAt G.abort (G.K x + 1) τ = true ∧ Ev τ)
            (q := Ev) (fun _ h => h.2) τ
      | bind _ _ => exact FlockSoundness.Game.ind_nonneg _ _
      | fail => exact FlockSoundness.Game.ind_nonneg _ _
    have eR1 : FlockSoundness.ZK.GK.avgR (fun r => φ ((G.model H ask x w ν).real r)) =
        Pr (fun r : G.Rand => goesOnAt G.abort (G.K x + 1) (interact ask (G.pmR H x w ν r) (G.k x) []) = true ∧
          Ev (interact ask (G.pmR H x w ν r) (G.k x) [])) := by
      rw [avgR_eq_Ex, hφ, ind_comp, Ex_ind_eq]
      rfl
    have eS : Pr (fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
        Ev ((G.simulatorE t M).out H Vs θ x (ν, q))) = Pr fun q => Ev (outT (g q).1) :=
      Pr_congr fun q => Iff.of_eq (congrArg Ev (G.simE_eval H ask x ν t M q.1 q.2.1 q.2.2))
    rw [eR1] at s3
    rw [eS]
    rw [abs_le] at s2 s3
    linarith [s1, s2.1, s2.2, s3.1, s3.2, hbind, s4]
  have eR : Pr (fun ω : G.Nonce × G.Rand => goesOnAt G.abort (G.k x - 1) (runStar H (G.prover w) Vs x ω θ).msgs = true ∧
        Ev (runStar H (G.prover w) Vs x ω θ).msgs) =
      Ex fun ν => Pr fun r : G.Rand => goesOnAt G.abort (G.K x + 1) (interact ask (G.pmR H x w ν r) (G.k x) []) = true ∧
        Ev (interact ask (G.pmR H x w ν r) (G.k x) []) := by
    rw [Pr_prod_left]
    congr 1
    funext ν
    congr 1
    funext r
    rw [runStar_msgs, hk, hK1]
    rfl
  have eS : Pr (fun c : (G.simulatorE t M).C x => Ev ((G.simulatorE t M).out H Vs θ x c)) =
      Ex fun ν => Pr fun q : G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x) =>
        Ev ((G.simulatorE t M).out H Vs θ x (ν, q)) :=
    Pr_prod_left (α := G.Nonce) (β := G.Rand × (Fin M → G.Rand) × (Fin (4 * M) → G.SimRand x))
      fun c => Ev ((G.simulatorE t M).out H Vs θ x c)
  rw [eR, eS]
  refine (Ex_mono key).trans ?_
  rw [Ex_add, Ex_const_add, Ex_mul_const]
  have := mul_le_mul_of_nonneg_right (G.pr_collR_le H Vs θ x w) (by positivity : (0 : ℝ) ≤ 4 * M)
  linarith

/-- **At the cap `M = t·3^t`**: `4·δ_A + 4t·δ_B + 2·3^{-t}` and `4t·3^t` times `Fork`'s probability. -/
theorem gkE_go_pow (G : Proto W) (H : Msg → Msg) (R : Rel W) (t : ℕ) (ht : 0 < t) (δA δB : Msg → ℝ)
    (hA : G.PrefixHides H R δA) (hB : G.SHVZKLe H R δB) (hB0 : ∀ x, 0 ≤ δB x) (hC : G.Binding H) :
    ∀ x w, R x w → ∀ (Θ : Type) (Vs : VStar Θ) (θ : Θ), Vs.k x = G.k x → ∀ Ev : List (Msg × Out) → Prop,
      Pr (fun ω : G.Nonce × G.Rand => goesOnAt G.abort (G.k x - 1) (runStar H (G.prover w) Vs x ω θ).msgs = true ∧
          Ev (runStar H (G.prover w) Vs x ω θ).msgs) ≤
        Pr (fun c : (G.simulatorE t (t * 3 ^ t)).C x => Ev ((G.simulatorE t (t * 3 ^ t)).out H Vs θ x c)) +
          (4 * δA x + 4 * t * δB x + 2 * (1 / 3) ^ t + 4 * t * 3 ^ t * (Fork H (G.prover w) Vs θ x).Pr H) := by
  have e : ((t : ℝ) / ((t * 3 ^ t : ℕ) : ℝ)) = (1 / 3) ^ t := by
    have : (0 : ℝ) < t := by exact_mod_cast ht
    push_cast
    rw [div_mul_eq_div_div, div_self this.ne', one_div_pow]
  have e2 : (4 : ℝ) * ((t * 3 ^ t : ℕ) : ℝ) = 4 * t * 3 ^ t := by push_cast; ring
  intro x w hR Θ Vs θ hk Ev
  have := G.gkE_go H R t (t * 3 ^ t) (by positivity) δA δB hA hB hB0 hC x w hR Θ Vs θ hk Ev
  simp only [e, e2] at this
  linarith

end Proto

end Claims.ZKMal
