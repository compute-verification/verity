module

public import Proofs.Flock.SpotFlock.Drawn
public import Proofs.Flock.SamplingGeneric.TwoFirstHitAt
public import Proofs.Flock.SamplingGeneric.SessCount
public import Proofs.Flock.Claims.Sampling.Refused
public import Proofs.Flock.Claims.Sampling.Select.SecondStrata

/-! The first-hit bound's three counts `N_s`, `K_s` and `K_m` at SPOT[Flock]'s code, each at most a weight summed over
the checked proof blocks. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

theorem sum_map_take_le (l : List ℕ) (n : ℕ) (w : ℕ → ℕ) : ((l.take n).map w).sum ≤ (l.map w).sum := by
  have h := congrArg (fun l => (l.map w).sum) (List.take_append_drop n l)
  simp only [List.map_append, List.sum_append] at h
  omega

theorem sum_map_one (l : List ℕ) : (l.map fun _ => (1 : ℕ)).sum = l.length := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [List.map_cons, List.sum_cons, List.length_cons, ih]; omega

variable (h : Held)

theorem code_ex_le (x : Msg) (G : List ℕ → ℝ) (w : ℕ → ℕ) (M : ℕ) (hM : ∀ u, w u ≤ M)
    (hG : ∀ l, G l ≤ ((l.map w).sum : ℝ)) {K : ℕ} (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => G ((code h).checked x r)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            w).sum + K * h.reg.ks.toList.sum * M : ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map w).sum : ℝ) := by
  have hc : ∀ r : Fin ((code h).V.k x) → FlockDeploy.Coin, (code h).checked x r =
      ((reg h x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r))).getD [] := fun _ => rfl
  cases hd : decode x with
  | none =>
    have h0 : ∀ r, (code h).checked x r = [] := fun r => by rw [hc]; simp [reg, hd]
    calc Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => G ((code h).checked x r))
        ≤ Ex (fun _ : Fin ((code h).V.k x) → FlockDeploy.Coin => (0 : ℝ)) :=
          Ex_mono fun r => by rw [h0]; simpa using hG []
      _ = 0 := Ex_const 0
      _ ≤ _ := add_nonneg (Nat.cast_nonneg _) (mul_nonneg (Pr_nonneg _) (Nat.cast_nonneg _))
  | some _ =>
    have h1 : ∀ r, (code h).checked x r = h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r)) := fun r => by rw [hc]; simp [reg, hd]
    calc Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => G ((code h).checked x r))
        ≤ Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin =>
            (((h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
              (src (r (order h x).first)) (stageSrc (stageCoins h x r))).map w).sum : ℝ)) :=
          Ex_mono fun r => by rw [h1]; exact hG _
      _ ≤ _ := h.reg.ex_sum_checked_le (Ω := Fin ((code h).V.k x) → FlockDeploy.Coin)
          (SamplingDeploy.units ((base h).atOrder (order h)) x) (fun r => src (r (order h x).first))
          (fun r => stageSrc (stageCoins h x r)) w M (K * h.reg.ks.toList.sum) hM fun _ _ _ hsel =>
            h.reg.proved_size_le _ _ hsel hK

theorem code_ex_le_strata (x : Msg) (G : List ℕ → ℝ) (w M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 → w (h.reg.blockOf ρ t i) ≤ M t)
    (hG : ∀ l, G l ≤ ((l.map w).sum : ℝ)) {K : ℕ} (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => G ((code h).checked x r)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            w).sum + K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum : ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map w).sum : ℝ) := by
  have hc : ∀ r : Fin ((code h).V.k x) → FlockDeploy.Coin, (code h).checked x r =
      ((reg h x).map fun g => g.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r))).getD [] := fun _ => rfl
  cases hd : decode x with
  | none =>
    have h0 : ∀ r, (code h).checked x r = [] := fun r => by rw [hc]; simp [reg, hd]
    calc Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => G ((code h).checked x r))
        ≤ Ex (fun _ : Fin ((code h).V.k x) → FlockDeploy.Coin => (0 : ℝ)) :=
          Ex_mono fun r => by rw [h0]; simpa using hG []
      _ = 0 := Ex_const 0
      _ ≤ _ := add_nonneg (Nat.cast_nonneg _) (mul_nonneg (Pr_nonneg _) (Nat.cast_nonneg _))
  | some _ =>
    have h1 : ∀ r, (code h).checked x r = h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
        (src (r (order h x).first)) (stageSrc (stageCoins h x r)) := fun r => by rw [hc]; simp [reg, hd]
    calc Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => G ((code h).checked x r))
        ≤ Ex (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin =>
            (((h.reg.checked (SamplingDeploy.units ((base h).atOrder (order h)) x)
              (src (r (order h x).first)) (stageSrc (stageCoins h x r))).map w).sum : ℝ)) :=
          Ex_mono fun r => by rw [h1]; exact hG _
      _ ≤ _ := h.reg.ex_checked_le (Ω := Fin ((code h).V.k x) → FlockDeploy.Coin)
          (SamplingDeploy.units ((base h).atOrder (order h)) x) (fun r => src (r (order h x).first))
          (fun r => stageSrc (stageCoins h x r)) (fun l => ((l.map w).sum : ℝ)) (Nat.cast_nonneg _) fun r _ _ hsel => by
            exact_mod_cast h.reg.sum_checked_le_strata _ _ _ hsel w M hM (h.reg.sample_size_le hK _)

variable {T : Type} (F : Claims.Flock.Deploy FlockDeploy.Coin (List FlockLevel3.GF128) T)

theorem code_hNs (x : Msg) {K : ℕ} (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      (((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.firstSlots x
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r)).card : ℝ)) ≤
      (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).length +
          K * h.reg.ks.toList.sum : ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
          (SamplingDeploy.units ((base h).atOrder (order h)) x : ℝ) := by
  have := code_ex_le h x (fun l => (((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.firstSlots x
      (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).card : ℝ)) (fun _ => 1) 1 (fun _ => le_rfl)
    (fun l => by
      rw [sum_map_one]
      exact_mod_cast ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.card_firstSlots_le x
        (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nPU x)).trans
        (List.length_take_le' _ _)) hK
  rw [sum_map_one, sum_map_one, List.length_range, mul_one] at this
  exact this

theorem code_hKs (x : Msg) (needs : ℕ → Finset ℕ) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 → (needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      (((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r)).card : ℝ)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (needs u).card).sum + K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum :
            ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map fun u => (needs u).card).sum : ℝ) :=
  code_ex_le_strata h x (fun l => (((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sampleNeeds x
      needs (l.take (Claims.Sampling.nPU (deployOf F (code h)) x))).card : ℝ)) (fun u => (needs u).card) M hM
    (fun l => by
      have hc := Claims.SamplingGeneric.Compiled.card_filter_needs_le
        (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nG x)
      exact_mod_cast hc.trans (sum_map_take_le l _ fun u => (needs u).card)) hK

theorem code_hKm (x : Msg) (needs : ℕ → Finset ℕ) (M : ℕ → ℕ)
    (hM : ∀ ρ t i, t < h.reg.ns.size → i < h.reg.ns.getD t 0 → (needs (h.reg.blockOf ρ t i)).card ≤ M t) {K : ℕ}
    (hK : firstK h.reg.first = some K) :
    Ex (fun r : Fin ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).V.k x) → FlockDeploy.Coin =>
      ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCount x needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).checked x r) : ℝ)) ≤
      ((((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter fun v => h.reg.always v).map
            fun u => (needs u).card).sum + K * ((List.range h.reg.ns.size).map fun t => h.reg.ks.getD t 0 * M t).sum :
            ℕ) +
        Pr (fun r : Fin ((code h).V.k x) → FlockDeploy.Coin => Except.toBool (Spot.Select.selectSecond
          (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks (stageSrc (stageCoins h x r))) = false) *
          (((List.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).map fun u => (needs u).card).sum : ℝ) :=
  code_ex_le_strata h x (fun l => ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sessCount x
      needs (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) : ℝ)) (fun u => (needs u).card) M hM
    (fun l => by
      have hc := (compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.sum_first_sessNeeds_le x
        (l.take (Claims.Sampling.nPU (deployOf F (code h)) x)) needs
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nPU x)
        ((compiledTwo F (code h) iroots (two h) (rs h) (cap h)).toCompiled.nG x)
      exact_mod_cast hc.trans (sum_map_take_le l _ fun u => (needs u).card)) hK

end SpotFlock
