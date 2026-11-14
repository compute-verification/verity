module

public import Proofs.Flock.SpotFlock.AddressFacts.Row

/-!
# Where round 2 puts a checked unit: the second sample's interior slot

The first step of `InstanceFact`: with no proof block always checked, a checked unit of an accepted run is the proof
block `blockAt sample j` of an interior slot `j` the second sample proves (`proved_of_checked`), the sample being
`secondDraw`'s (`select_of_secondDraw`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

/-- **A checked unit is a proved slot's proof block**, where no proof block is always checked and the second stage's
sample is accepted. -/
theorem proved_of_checked {g : RegTwo} {N u : ℕ} {src₁ src₂ : ByteArray} {draw : Json} {read : ℕ}
    (hna : ∀ v, g.always v = false) (hu : u ∈ g.checked N src₁ src₂)
    (hd : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, read)) :
    ∃ j ∈ Spot.Draw.proved draw, g.blockAt (g.sample src₁) j.toNat = u := by
  unfold RegTwo.checked at hu
  rw [hd] at hu
  obtain ⟨-, hv⟩ := List.mem_filter.1 hu
  rw [hna u, Bool.false_or, Array.any_eq_true'] at hv
  obtain ⟨j, hj, he⟩ := hv
  exact ⟨j, hj, beq_iff_eq.1 he⟩

variable (h : Held)

/-- **The run's second sample is the second stage's selection's.** -/
theorem select_of_secondDraw {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {draw : Json}
    (hd : secondDraw h c₀ c₁ = some draw) :
    ∃ read, Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (stageSrc c₁) = .ok (draw, read) := by
  unfold secondDraw at hd
  cases hs : Spot.Select.selectSecond (h.reg.sample (src c₀)) h.reg.ns h.reg.ks (stageSrc c₁) with
  | error e =>
    rw [hs] at hd
    exact absurd hd (by simp)
  | ok p =>
    obtain ⟨d, rd⟩ := p
    rw [hs] at hd
    refine ⟨rd, ?_⟩
    have hd' : d = draw := Option.some.inj hd
    rw [hd']

end AddressRow

end SpotFlock
