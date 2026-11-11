module

public import Proofs.Flock.SpotFlock.RevealsOpen
public import Proofs.Flock.Claims.Sampling.Select.Second
public import Proofs.Flock.Claims.Sampling.Select.Size
import all Spot.Draw
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# How many proof blocks SPOT[Flock]'s two stages prove

`K`, the proof blocks a two-stage audit checks, is the always-checked blocks and the second stage's draw. The draw has
`Kfirst · Σ_t k_t` of them at most: `k_t` of each of template `t`'s cells, one cell for each replay block the first
stage proves, `Kfirst` of those.

* `drawUnits_work_size`: a work law's draw holds its strata's counts.
* `selectSecond_proved_size`: an accepted second stage proves `|sample| · Σ_t k_t` proof blocks, its draw's units and
  no closure.
* `RegTwo.sample_size_le`: a first stage of count `K` (`SpotFlock.firstK`) hands the second stage at most `K` replay
  blocks; `RegTwo.proved_size_le`: so the second stage proves at most `K · Σ_t k_t`.
* `RegTwo.card_biUnion_checked_le_count`, `SpotFlock.code_card_biUnion_le_count`: the gates the checked proof blocks
  need number at most the always-checked blocks' needs plus `K · Σ_t k_t · M`, with `M` the most gates a block needs,
  at the registration and at the deployed code.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open Lean (Json)

/-- **A work law's draw holds its strata's counts.** -/
theorem drawUnits_work_size {K : ℕ} {ws : Array Flock.Draw.WorkStratum} {n : ℕ} {src : ByteArray}
    {units : Array ℕ} {read : ℕ} (h : Draw.drawUnits (.work K ws) n src = .ok (units, read)) :
    units.size = ((ws.map (·.toStratum)).toList.map (·.k)).sum := by
  unfold Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Draw.take src need) (.work K ws) n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok x =>
    obtain ⟨r, read'⟩ := x
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hu, -⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hu
    unfold Flock.Draw.drawWith at hw
    dsimp only at hw
    rw [StateT.run_bind] at hw
    cases ho : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) (ws.map (·.toStratum))).run 0 with
    | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨v, p⟩ := y
      rw [ho, bind_ok, StateT.run_pure] at hw
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
      cases v with
      | none => exact absurd h1 (by simp)
      | some D =>
        obtain rfl := (Option.some.inj h1).symm
        exact stratifiedWith_size src _ 0 p D ho

/-- The cells' counts sum to `q` times the first `T` templates' counts. -/
theorem snd_sum_cells (q : ℕ) (ns ks : Array ℕ) : ∀ T : ℕ,
    (((List.range T).flatMap fun t => List.replicate q (ns.getD t 0, ks.getD t 0)).map Prod.snd).sum =
      q * (ks.toList.take T).sum
  | 0 => by simp
  | T + 1 => by
    rw [List.range_succ, List.flatMap_append, List.map_append, List.sum_append, snd_sum_cells q ns ks T,
      List.flatMap_singleton, List.map_replicate, List.sum_replicate_nat, sum_take_succ, getD_toList ks T,
      Nat.mul_add]

/-- A second stage `secondProblem` passes has a count for each template's cell. -/
theorem secondProblem_ks_size {sample ns ks : Array ℕ} (h : Select.secondProblem sample ns ks = none) :
    ks.size = ns.size := by
  unfold Select.secondProblem at h
  split at h
  · simp at h
  split at h
  · simp at h
  rename_i hns
  by_contra hne
  exact hns (by simp [hne])

theorem getD_draw_closure (sample ns ks : Array ℕ) (m : ℕ) (units : Array ℕ) :
    getD (Json.mkObj ((entries (Select.secondLaw sample ns ks)).toList ++
      [("population", Lean.toJson m), ("units", natsJson units)])) "closure" = Json.null := (rfl)

theorem size_natsJson_int (units : Array ℕ) :
    ((units.map fun x => Lean.toJson x).filterMap int?).size = units.size := by
  rw [Array.filterMap_map, show (int? ∘ fun x : ℕ => Lean.toJson x) = some ∘ fun x : ℕ => (x : Int) from
    funext int_toJson, Array.filterMap_eq_map rfl, Array.size_map]

/-- The second stage's sample object proves its units and nothing else: `secondLaw` has no closure. -/
theorem proved_second_draw (sample ns ks : Array ℕ) (m : ℕ) (units : Array ℕ) :
    (Draw.proved (Json.mkObj ((entries (Select.secondLaw sample ns ks)).toList ++
      [("population", Lean.toJson m), ("units", natsJson units)]))).size = units.size := by
  unfold Draw.proved
  dsimp only
  rw [Verity.QSort.size_qsort, Array.size_append, getD_draw_units, getD_draw_closure]
  exact (Nat.add_zero _).trans (size_natsJson_int units)

/-- **An accepted second stage proves `|sample| · Σ_t k_t` proof blocks**: `k_t` of each of template `t`'s cells, one
cell for each sampled replay block, and no closure. -/
theorem selectSecond_proved_size {sample ns ks : Array ℕ} {src : ByteArray} {draw : Json} {read : ℕ}
    (h : Select.selectSecond sample ns ks src = .ok (draw, read)) :
    (Draw.proved draw).size = sample.size * ks.toList.sum := by
  unfold Select.selectSecond at h
  cases hq : Select.secondProblem sample ns ks with
  | some e => rw [hq] at h; dsimp only at h; exact absurd h (throw_bind_ne _ _ _)
  | none =>
    rw [hq] at h
    dsimp only at h
    obtain ⟨⟨draw', read'⟩, hd, h⟩ := bind_eq_ok.1 h
    split at h
    · exact absurd h (throw_bind_ne _ _ _)
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    unfold Draw.drawFirstRead at hd
    obtain ⟨parsed, hp, hd⟩ := bind_eq_ok.1 hd
    obtain ⟨⟨units, read''⟩, hu, hd⟩ := bind_eq_ok.1 hd
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hd
    obtain ⟨rfl, rfl⟩ := hd
    obtain ⟨ws, rfl, hmap⟩ := flockLaw_second sample ns ks _ parsed hp
    have hk := congrArg (List.map Prod.snd) (sizes_second hmap)
    rw [List.map_map] at hk
    refine (proved_second_draw sample ns ks (sample.size * ns.toList.sum) units).trans ?_
    rw [drawUnits_work_size hu, show (ws.map (·.toStratum)).toList.map (·.k) = _ from hk, snd_sum_cells,
      List.take_of_length_le (by simp [secondProblem_ks_size hq])]

end Properties.Spot.SecurityProofs

namespace SamplingDeploy

open SpotFlock (firstK select_subset_size)

/-- **A first stage of count `K` hands the second stage at most `K` replay blocks**: `K` where it accepts, none where
it is refused. -/
theorem RegTwo.sample_size_le (g : RegTwo) {K : ℕ} (hK : firstK g.first = some K) (src : ByteArray) :
    (g.sample src).size ≤ K := by
  unfold RegTwo.sample
  cases hs : Spot.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure src with
  | error e => exact Nat.zero_le _
  | ok r =>
    obtain ⟨draw, proved, read⟩ := r
    exact (select_subset_size hK hs).le

/-- **The count**: where the second stage draws on the replay sample of a first stage of count `K`, it proves at most
`K · Σ_t k_t` proof blocks. -/
theorem RegTwo.proved_size_le (g : RegTwo) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) {K : ℕ}
    (hK : firstK g.first = some K) : (Spot.Draw.proved draw).size ≤ K * g.ks.toList.sum := by
  rw [Properties.Spot.SecurityProofs.selectSecond_proved_size hsel]
  exact Nat.mul_le_mul_right _ (g.sample_size_le hK src₁)

/-- **K by the count**: where the second stage draws on the replay sample of a first stage of count `K`, the gates the
checked proof blocks need number at most the always-checked blocks' needs plus `K · Σ_t k_t · M`, with `M` the most
gates a block needs. -/
theorem RegTwo.card_biUnion_checked_le_count (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) {K : ℕ}
    (hK : firstK g.first = some K) (needs : ℕ → Finset ℕ) (M : ℕ) (hM : ∀ u, (needs u).card ≤ M) :
    ((g.checked N src₁ src₂).toFinset.biUnion needs).card ≤
      ∑ v ∈ (Finset.range N).filter (fun v => g.always v = true), (needs v).card + K * g.ks.toList.sum * M := by
  refine (g.card_biUnion_checked_le N src₁ src₂ hsel needs).trans (Nat.add_le_add_left ?_ _)
  have hs := List.sum_le_length_nsmul
    ((Spot.Draw.proved draw).toList.map fun j => (needs (g.blockAt (g.sample src₁) j.toNat)).card) M
    (by
      intro y hy
      obtain ⟨j, _, rfl⟩ := List.mem_map.1 hy
      exact hM _)
  rw [List.length_map, Array.length_toList, smul_eq_mul] at hs
  exact hs.trans (Nat.mul_le_mul_right _ (g.proved_size_le src₁ src₂ hsel hK))

end SamplingDeploy

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **K at the deployed code by the count**: where the second stage draws on the replay sample of a first stage of
count `K`, the gates the checked proof blocks need number at most the always-checked blocks' needs plus
`K · Σ_t k_t · M`, with `M` the most gates a block needs. -/
theorem code_card_biUnion_le_count (x : Msg) (r : Fin ((code h).V.k x) → FlockDeploy.Coin) (needs : ℕ → Finset ℕ)
    {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (h.reg.sample (src (r (order h x).first))) h.reg.ns h.reg.ks
      (src (r (order h x).second)) = .ok (draw, n)) {K : ℕ} (hK : firstK h.reg.first = some K)
    (M : ℕ) (hM : ∀ u, (needs u).card ≤ M) :
    (((code h).checked x r).toFinset.biUnion needs).card ≤
      ∑ v ∈ (Finset.range (SamplingDeploy.units ((base h).atOrder (order h)) x)).filter
          (fun v => h.reg.always v = true), (needs v).card + K * h.reg.ks.toList.sum * M :=
  (Finset.card_le_card (Finset.biUnion_subset_biUnion_of_subset_left needs fun u hu =>
    List.mem_toFinset.2 (code_checked_sub h x r u (List.mem_toFinset.1 hu)))).trans
    (h.reg.card_biUnion_checked_le_count _ _ _ hsel hK needs M hM)

end SpotFlock
