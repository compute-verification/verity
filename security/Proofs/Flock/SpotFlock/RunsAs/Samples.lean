module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
public import Proofs.Flock.Claims.Sampling.Select.Prefix
public import Proofs.Flock.Claims.Sampling.Select.SecondCount
import Proofs.Flock.Claims.Sampling.Select.Take
import Proofs.Flock.Claims.Sampling.Select.Steps

/-!
# The run's two samples are the play's (parts 1 and 2)

A sample reads the bytes it is given in order (`Draw.take`) and stops when its law's draw ends, so a stage `Run.run`
accepted on the challenger's bytes `ξ` samples the same on `ξ` followed by anything (`select_append`,
`selectSecond_append`). The play's coins are the run's bytes followed by zeros (`PlayCoins`), so the model's replay sample
on round 0's coin is the units the run's first stage proves (`sample_pad`), and the model's second draw on the second
stage's coins is the run's second draw (`secondDraw_pad`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

theorem drawFirstRead_append {law : Json} {n : ℕ} {ξ : ByteArray} (t : ByteArray) {r : Json × ℕ}
    (h : Spot.Draw.drawFirstRead law n ξ = .ok r) : Spot.Draw.drawFirstRead law n (ξ ++ t) = .ok r := by
  unfold Spot.Draw.drawFirstRead at h ⊢
  obtain ⟨parsed, hp, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  obtain ⟨ur, hu, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  rw [hp, Properties.Spot.SecurityProofs.bind_ok, Properties.Spot.SecurityProofs.drawUnits_append hu,
    Properties.Spot.SecurityProofs.bind_ok]
  exact h

/-- **The second stage reads only what it needs**: its sample on `ξ` is its sample on `ξ ++ t`. -/
theorem selectSecond_append {sample ns ks : Array ℕ} {ξ : ByteArray} (t : ByteArray) {r : Json × ℕ}
    (h : Spot.Select.selectSecond sample ns ks ξ = .ok r) :
    Spot.Select.selectSecond sample ns ks (ξ ++ t) = .ok r := by
  unfold Spot.Select.selectSecond at h ⊢
  cases hq : Spot.Select.secondProblem sample ns ks with
  | some e =>
    try rw [hq] at h
    dsimp only at h
    exact absurd h (Properties.Spot.SecurityProofs.throw_bind_ne _ _ _)
  | none =>
    try rw [hq] at h
    try rw [hq]
    dsimp only at h ⊢
    obtain ⟨dr, hd, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
    rw [drawFirstRead_append t hd, Properties.Spot.SecurityProofs.bind_ok]
    exact h

/-- **The first stage reads only what it needs**: with no closure map, its sample on `ξ` is its sample on `ξ ++ t`. -/
theorem select_append {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {ξ : ByteArray} (t : ByteArray) {r : Json × Array ℕ × ℕ}
    (h : Spot.Select.select rule p query table none ξ = .ok r) :
    Spot.Select.select rule p query table none (ξ ++ t) = .ok r := by
  unfold Spot.Select.select at h ⊢
  obtain ⟨n, hn, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  obtain ⟨derived, hderived, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
  rw [hn, Properties.Spot.SecurityProofs.bind_ok, hnamed, Properties.Spot.SecurityProofs.bind_ok, hderived,
    Properties.Spot.SecurityProofs.bind_ok]
  dsimp only at h ⊢
  cases rule with
  | all => exact h
  | law text =>
    dsimp only at h ⊢
    obtain ⟨parsed, hp, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
    obtain ⟨ur, hu, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
    rw [hp, Properties.Spot.SecurityProofs.bind_ok, Properties.Spot.SecurityProofs.drawUnits_append hu,
      Properties.Spot.SecurityProofs.bind_ok]
    exact h
  | twoStage grain replay k =>
    dsimp only at h ⊢
    obtain ⟨parsed, hp, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
    obtain ⟨ur, hu, h⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 h
    rw [hp, Properties.Spot.SecurityProofs.bind_ok, Properties.Spot.SecurityProofs.drawUnits_append hu,
      Properties.Spot.SecurityProofs.bind_ok]
    exact h

/-- **Part 1: the model's replay sample is the first stage's**: where the registration has no closure map and the
first stage accepted on `ξ`, proving `proved`, the replay sample on `ξ ++ t` is `proved`. -/
theorem sample_pad {g : SamplingDeploy.RegTwo} {ξ : ByteArray} (t : ByteArray) {draw : Json} {proved : Array ℕ} {read : ℕ}
    (hc : g.first.closure = none)
    (h : Spot.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure ξ =
      .ok (draw, proved, read)) :
    g.sample (ξ ++ t) = proved := by
  rw [hc] at h
  simp only [SamplingDeploy.RegTwo.sample, hc, select_append t h]

/-- **Part 2: the model's second draw is the second stage's**: where the second stage accepted on `ξ` over the model's
replay sample, and the second stage's coins are `ξ` followed by `t`, the model's second draw is the run's. -/
theorem secondDraw_pad (h : Held) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) {ξ : ByteArray} (t : ByteArray)
    {draw : Json} {read : ℕ}
    (hs : Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks ξ = .ok (draw, read))
    (hsrc : stageSrc c₁ = ξ ++ t) :
    secondDraw h c₀ c₁ = some draw := by
  simp only [secondDraw, hsrc, selectSecond_append t hs]

end SpotFlock.RunsAs
