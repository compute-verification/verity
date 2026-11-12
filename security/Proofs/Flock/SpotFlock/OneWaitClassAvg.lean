module

public import Proofs.Flock.SpotFlock.OneWaitClassAt
public import Proofs.Flock.SpotFlock.ClassCounts
public import Proofs.Flock.SamplingGeneric.TwoOneWaitAvg
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Masked
import Proofs.Flock.Claims.KS.FindsFlock
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.SpotFlock.RevealsLe

/-!
# The one-wait by-class first-hit bound at the prover's expected calls, at SPOT[Flock]'s code

`CompiledTwo.eventTwoX_oneWait_le_cls_avg` reads, besides a run's expected calls `T`, three averages of a run's
calls times a count of its sample (`Compiled.CallsAvg`): the sessions in the sample (`A_s`), the classes of the
gates it needs (`A_k`) and the classes' representatives its sessions need, session by session (`A_m`). At
SPOT[Flock]'s compiled verifier each count is at most a weight summed over the checked proof blocks. Where the second
stage accepts, the checked blocks are the always-checked ones and those it proves; where it is refused, every block
below `N`. So a count times a nonnegative weight `p` of the run averages at most

`C · E[p] + S · E[p on the refused runs]`,

with `C` the always-checked blocks' weights plus `K · Σ_t k_t · M_t`, and `S` every block's weight below `N`: the
whole partition's count is paid only on the calls spent in refused runs.

* `RegTwo.ex_checked_mul_le`: `RegTwo.ex_checked_le` with the function multiplied by a nonnegative weight.
* `code_ex_mul_le_strata`: a function of the checked proof blocks times a weight, at the code, by template.
* `code_hAs`, `code_hAk_cls`, `code_hAm_cls`: the three averages at the code.
* `code_callsAvg`: `Compiled.CallsAvg` at the code, from a run's expected calls `T`, its expected calls in refused
  runs `T_R` and the expected number of classes its sample needs: `A_s ≤ C_s·T + N·T_R` and
  `A_k, A_m ≤ C_k·T + S_k·T_R`.
* `code_eventTwoX_oneWait_le_cls_avg`: `CompiledTwo.eventTwoX_oneWait_le_cls_avg` at SPOT[Flock]'s compiled
  two-stage verifier, with `code_eventTwoX_oneWait_le_cls_at_budget`'s gates and classes.
-/

@[expose] public section

namespace SamplingDeploy

open Claims

/-- **The refused second stage, in expectation, with a weight**: over coins `ω` with first-stage bytes `src₁ ω` and
second-stage bytes `src₂ ω`, a function `F` of the checked blocks at most `B ≥ 0` wherever the second stage
accepts, times a nonnegative weight `p`, averages at most `B` times the weight's average plus `F` at every block
below `N` times the weight's average on the refused coins. -/
theorem RegTwo.ex_checked_mul_le {Ω : Type} [Fintype Ω] (g : RegTwo) (N : ℕ) (src₁ src₂ : Ω → ByteArray)
    (F : List ℕ → ℝ) (p : Ω → ℝ) {B : ℝ} (hB : 0 ≤ B) (hp : ∀ ω, 0 ≤ p ω)
    (hok : ∀ ω sel n, Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω) = .ok (sel, n) →
      F (g.checked N (src₁ ω) (src₂ ω)) ≤ B) :
    Ex (fun ω => F (g.checked N (src₁ ω) (src₂ ω)) * p ω) ≤
      B * Ex p + F (List.range N) * Ex (fun ω =>
        if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
        else 0) := by
  have hpt : ∀ ω, F (g.checked N (src₁ ω) (src₂ ω)) * p ω ≤ B * p ω + F (List.range N) *
      (if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
        else 0) := by
    intro ω
    by_cases hR : Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false
    · rw [if_pos hR, g.checked_of_refused N _ _ hR]
      have h0 := mul_nonneg hB (hp ω)
      linarith
    · rw [if_neg hR, mul_zero, add_zero]
      refine mul_le_mul_of_nonneg_right ?_ (hp ω)
      cases hs : Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω) with
      | ok d => exact hok ω d.1 d.2 hs
      | error e => exact absurd (by rw [hs]; rfl) hR
  calc Ex (fun ω => F (g.checked N (src₁ ω) (src₂ ω)) * p ω)
      ≤ Ex (fun ω => B * p ω + F (List.range N) *
          (if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
            else 0)) := Ex_mono hpt
    _ = Ex (fun ω => B * p ω) + Ex (fun ω => F (List.range N) *
          (if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
            else 0)) := Ex_add _ _
    _ = B * Ex p + F (List.range N) * Ex (fun ω =>
          if Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false then p ω
          else 0) := by rw [Ex_mul_left, Ex_mul_left]

end SamplingDeploy

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **A function of the checked proof blocks times a weight of the run, at SPOT[Flock]'s code, by template**: a
function `G` at most a weight `w` summed over its blocks, with `w` at most `M t` on template `t`'s blocks, times a
nonnegative `p` of the prover's tape and the coins, averages at most the always-checked blocks' weights below `N`
plus `K · Σ_t k_t · M t`, times `p`'s average, plus every block's weight below `N` times `p`'s average on the coins
whose second stage is refused. -/
theorem code_ex_mul_le_strata (x : Msg) (G : List ℕ → ℝ) (w M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 → w (h.reg.blockOf ρ t i) ≤ M t)
    (hG : ∀ l, G l ≤ ((l.map w).sum : ℝ)) {K : ℕ} (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) → ℝ) (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => G ((code h).checked x q.2) * p q) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            w).sum + K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * Ex p +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map w).sum : ℝ) *
          Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) := by
  have hc : ∀ r : Fin ((code h).V.k x) → FlockDeploy.Coin, (code h).checked x r =
      ((reg h x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r))).getD [] := fun _ => rfl
  cases hd : decode x with
  | none =>
    have h0 : ∀ r, (code h).checked x r = [] := fun r => by rw [hc]; simp [reg, hd]
    have hG0 : G [] ≤ 0 := by simpa using hG []
    calc Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => G ((code h).checked x q.2) * p q)
        ≤ Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => (0 : ℝ) * p q) :=
          Ex_mono fun q => by
            rw [h0]
            exact mul_le_mul_of_nonneg_right hG0 (hp q)
      _ = 0 := by rw [Ex_mul_left, zero_mul]
      _ ≤ _ := add_nonneg (mul_nonneg (Nat.cast_nonneg _) (Ex_nonneg hp))
          (mul_nonneg (Nat.cast_nonneg _) (Ex_nonneg fun q => by
            split_ifs
            · exact hp q
            · exact le_rfl))
  | some _ =>
    have h1 : ∀ r, (code h).checked x r = h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r)) := fun r => by rw [hc]; simp [reg, hd]
    calc Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) => G ((code h).checked x q.2) * p q)
        ≤ Ex (fun q : Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin) =>
            (((h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
              (src (q.2 (order h x).first)) (stageSrc (stageCoins h x q.2))).map w).sum : ℝ) * p q) :=
          Ex_mono fun q => by
            rw [h1]
            exact mul_le_mul_of_nonneg_right (hG _) (hp q)
      _ ≤ _ := h.reg.ex_checked_mul_le (Ω := Ω × (Fin ((code h).V.k x) → FlockDeploy.Coin))
          (SamplingDeploy.units ((base h).atOrder (order h)) x) (fun q => src (q.2 (order h x).first))
          (fun q => stageSrc (stageCoins h x q.2)) (fun l => ((l.map w).sum : ℝ)) p (Nat.cast_nonneg _) hp
          fun q _ _ hsel => by
            exact_mod_cast h.reg.sum_checked_le_strata _ _ _ hsel w M hM (h.reg.sample_size_le hK _)

section Counts

variable {Tr : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) Tr)

/-- **`A_s` at SPOT[Flock]'s code**: the sessions of a run's sample times a nonnegative weight `p` of the run
average at most the always-checked blocks below `N` plus `K · Σ_t k_t`, times `p`'s average, plus `N` times `p`'s
average on the coins whose second stage is refused. -/
theorem code_hAs (x : Msg) {K : ℕ} (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) → ℝ)
    (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
      (((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.firstSlots x
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x q.2)).card : ℝ) * p q) ≤
      (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).length +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0).sum : ℕ) * Ex p +
        ((SamplingDeploy.units ((base h).atOrder (order h)) x) : ℝ) *
          Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) := by
  have := code_ex_mul_le_strata h x
    (fun l => (((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.firstSlots x
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).card : ℝ)) (fun _ => 1) (fun _ => 1)
    (fun _ _ _ _ _ => le_rfl)
    (fun l => by
      rw [sum_map_one]
      exact_mod_cast ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.card_firstSlots_le x
        (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nPU x)).trans
        (List.length_take_le' _ _)) hK p hp
  rw [sum_map_one, sum_map_one, List.length_range] at this
  simp only [mul_one] at this
  exact this

/-- **`A_k` by class at SPOT[Flock]'s code**: with every block of template `t` needing at most `M t` classes under
`cls`, the classes of the gates a run's sample needs times a nonnegative weight `p` of the run average at most the
always-checked blocks' classes below `N` plus `K · Σ_t k_t · M t`, times `p`'s average, plus every block's classes
below `N` times `p`'s average on the coins whose second stage is refused. -/
theorem code_hAk_cls (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) →
      Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x)) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls
        needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) → ℝ)
    (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
      ((((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x q.2)).image cls).card : ℝ) * p q) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * Ex p +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum : ℝ) *
          Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) :=
  code_ex_mul_le_strata h x
    (fun l => ((((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).image cls).card : ℝ))
    (fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card) M hM
    (fun l => by
      have h1 : ((((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
          (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).image cls).card : ℝ) ≤
          (((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x
            (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs)
            (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).card : ℝ) :=
        Nat.cast_le.2 (Finset.card_le_card
          ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.image_sampleNeeds_subset x needs cls
            (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))))
      have hc := Claims.SamplingGeneric.Compiled.card_filter_needs_le
        (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
        (Claims.SamplingGeneric.Compiled.clsNeeds
          ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs)
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nG x)
      refine h1.trans ?_
      exact_mod_cast hc.trans (sum_map_take_le l _ fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card)) hK p hp

/-- **`A_m` by class at SPOT[Flock]'s code**: with a representative representing itself under `cls` and every block
of template `t` needing at most `M t` classes, the classes' representatives a run's sessions need, session by
session, times a nonnegative weight `p` of the run average at most the always-checked blocks' classes below `N`
plus `K · Σ_t k_t · M t`, times `p`'s average, plus every block's classes below `N` times `p`'s average on the
coins whose second stage is refused. -/
theorem code_hAm_cls (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) →
      Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x)) (hidem : ∀ g, cls (cls g) = cls g)
    (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls
        needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω]
    (p : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) → ℝ)
    (hp : ∀ q, 0 ≤ p q) :
    Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
      ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn x needs
        (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs l).image cls)
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x q.2) : ℝ) * p q) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * Ex p +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum : ℝ) *
          Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
            if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
              h.reg.ks (stageSrc (stageCoins h x q.2))) = false then p q else 0) :=
  code_ex_mul_le_strata h x
    (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn x needs
      (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs l).image cls)
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ))
    (fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card) M hM
    (fun l => by
      have h1 : ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn x needs
          (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs l).image
            cls) (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ) ≤
          ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCount x
            (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs)
            (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ) :=
        Nat.cast_le.2
          ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCountIn_cls_le x needs cls hidem
            (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)))
      have hc := (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sum_first_sessNeeds_le x
        (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
        (Claims.SamplingGeneric.Compiled.clsNeeds
          ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs)
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nPU x)
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nG x)
      refine h1.trans ?_
      exact_mod_cast hc.trans (sum_map_take_le l _ fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card)) hK p hp

/-- **The averages of `Compiled.CallsAvg` at SPOT[Flock]'s code, by class**: with a representative representing
itself under `cls` and every block of template `t` needing at most `M t` classes, where a run calls SHA-512 at most
`T` times on average, both parties' calls counted, at most `T_R` of them on average in the runs whose second stage
is refused, and its sample needs gates of at most `K_s` classes on average, the three products average at most

`A_s ≤ C_s · T + N · T_R` and `A_k, A_m ≤ C_k · T + S_k · T_R`,

with `C_s` the always-checked blocks below `N` plus `K · Σ_t k_t`, `C_k` their classes plus `K · Σ_t k_t · M t`,
and `S_k` every block's classes below `N`. -/
theorem code_callsAvg (x : Msg) (needs : ℕ → Finset ℕ)
    (cls : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) →
      Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x)) (hidem : ∀ g, cls (cls g) = cls g)
    (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 →
      (Claims.SamplingGeneric.Compiled.clsNeeds ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls
        needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) {Ω : Type} [Fintype Ω] (P : Prover Ω) {T TR Ks : ℝ}
    (hT : Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
      ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.partiesCalls FlockDeploy.sha512 P x
        q.1 q.2 : ℝ)) ≤ T)
    (hR : Ex (fun q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin) =>
        if Except.toBool (Spot.Select.selectSecond (h.reg.sample (src (q.2 (order h x).first))) h.reg.ns
          h.reg.ks (stageSrc (stageCoins h x q.2))) = false then
          ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.partiesCalls FlockDeploy.sha512 P x
            q.1 q.2 : ℝ)
        else 0) ≤ TR)
    (hKs : Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      ((((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r)).image cls).card : ℝ)) ≤ Ks) :
    (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.CallsAvg FlockDeploy.sha512 P x needs cls T
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).length +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0).sum : ℕ) * T +
        ((SamplingDeploy.units ((base h).atOrder (order h)) x) : ℝ) * TR)
      (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * T +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum : ℝ) * TR)
      (((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum +
          K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) * T +
        (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map
            fun u => (Claims.SamplingGeneric.Compiled.clsNeeds
              ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).nG x) cls needs u).card).sum : ℝ) * TR) Ks := by
  have hp : ∀ q : Ω × (Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin),
      (0 : ℝ) ≤
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.partiesCalls FlockDeploy.sha512 P x
          q.1 q.2 : ℝ) :=
    fun q => Nat.cast_nonneg _
  exact ⟨hT,
    (code_hAs h F x hK _ hp).trans (add_le_add (mul_le_mul_of_nonneg_left hT (Nat.cast_nonneg _))
      (mul_le_mul_of_nonneg_left hR (Nat.cast_nonneg _))),
    (code_hAk_cls h F x needs cls M hM hK _ hp).trans (add_le_add (mul_le_mul_of_nonneg_left hT (Nat.cast_nonneg _))
      (mul_le_mul_of_nonneg_left hR (Nat.cast_nonneg _))),
    (code_hAm_cls h F x needs cls hidem M hM hK _ hp).trans
      (add_le_add (mul_le_mul_of_nonneg_left hT (Nat.cast_nonneg _))
        (mul_le_mul_of_nonneg_left hR (Nat.cast_nonneg _))), hKs⟩

end Counts

/-- **The by-class first-hit bound at the prover's expected calls, the finders run on one wait a session, at the
code and its key**: `CompiledTwo.eventTwoX_oneWait_le_cls_avg` at SPOT[Flock]'s compiled two-stage verifier `L`,
with `needs` the gates each proof block's reveal opens and `cls` the classes of `gateKey`, under `Built` and
`Addresses` at a statement in the two-stage scope. Where a run's calls and their products with its sample's three
counts average at most `T`, `A_s`, `A_k` and `A_m`, the sample needs gates of at most `K_s` classes on average
(`Compiled.CallsAvg`), and the one-wait compiled verifier's `B_conf,X` and its two by-class finders at every `K`
meet the linear expected-time bound, an accepted run samples a proof block of the one-wait compiled verifier's
`FregTwoX` with probability at most `εclsOneWaitAvg`. No budget is held on every run. -/
theorem code_eventTwoX_oneWait_le_cls_avg (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ}
    (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) {T As Ak Am Ks : ℝ}
    (hC : L.toCompiled.CallsAvg FlockDeploy.sha512 P x needs cls T As Ak Am Ks)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.oneWait.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512
      (L.oneWait.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.oneWait.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      L.εclsOneWaitAvg FlockDeploy.sha512 x P T As Ak Am Ks := by
  have hA : (Claims.SamplingGeneric.Arg.ofFlock
      (FlockDeploy.deployZk _root_.Flock.Readable.relation)).KS FlockDeploy.sha512 :=
    Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)
  have hbind : L.commit.Binds FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.commit_binds (code h) _
  have hSess : L.Sessions FlockDeploy.sha512 := by
    subst hL
    exact sessionsTwo_spot h
      (sessionOpensTwo_of_frameIs h (layoutFor h) (embeds_layoutFor h) (frameIs_built h hB hS))
  have hopen : L.toCompiled.OpenCallsLe FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h)
  have hs : L.parse x = some s := by
    subst hL
    exact hx
  have hPt : L.partition s = some Pt := by
    subst hL
    exact hP
  have hd' : L.two s = some d := by
    subst hL
    exact hd
  have hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u := by
    subst hL hneeds
    exact fun u v w => code_hpos h s u v w
  have hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val := by
    subst hcls
    intro g g' hgg
    exact gateKey_block h ((Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg)
  have hkept : ∀ (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.oneWait.SlotEC x),
      L.oneWait.toCompiled.ClassWhole FlockDeploy.sha512 x cls (L.oneWait.kept FlockDeploy.sha512 P x s ω r c) := by
    subst hcls
    exact fun ω r c => L.oneWait.toCompiled.kept_classWhole_of_wit FlockDeploy.sha512 P x s ω _ r c
      (code_hwit_oneWait h hB hS hL hx hP hd hin P ω r c)
  exact Claims.SamplingGeneric.CompiledTwo.eventTwoX_oneWait_le_cls_avg oneWaitFinds_zk hA hbind hSess hopen
    hs hPt hd' hin P needs hpos cls
    (fun a g g' hgg hg =>
      L.oneWait.toCompiled.opensAt_isSome_cls FlockDeploy.sha512 P x a.1 hs cls a.2.1 a.2.2
        (hkept a.1 a.2.1 a.2.2) g g' hgg hg)
    (fun t ec g g' hgg hg =>
      L.oneWait.iopTable_isSome_cls FlockDeploy.sha512 P x t.1 hs d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g'
        hgg hg)
    hC hBc hX hI

end SpotFlock
