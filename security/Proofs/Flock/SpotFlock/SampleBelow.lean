module

public import Proofs.Flock.SpotFlock.Cap

/-!
# A first stage of count `K` samples below its population

A first stage that samples a uniform `K`-subset (`firstK`) proves exactly the units it samples (`select_proved`), and a
`subset` sample that ends is distinct indices below the population it samples over (`Discharge.Exec.subset_spec`). So
every unit it proves is below the registration's population, `Partition.units` (`select_subset_lt`), and every block of
the replay sample is (`RegTwo.sample_lt`).
-/

@[expose] public section

namespace Flock.Bridge

open Properties.Spot.SecurityProofs (bind_ok bind_error onWith_take) in
/-- **A `subset` sample that ends holds units below the population**: `drawUnits`' units are `Draw.subset`'s, the
values of a permutation that keeps `[0, n)` on `[0, k)`, and `k ≤ n` (`subset_spec`). -/
theorem drawUnits_subset_lt {k n : ℕ} {src : ByteArray} {units : Array ℕ} {read : ℕ}
    (h : Spot.Draw.drawUnits (.subset k) n src = .ok (units, read)) : ∀ x ∈ units, x < n := by
  unfold Spot.Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Spot.Draw.take src need) (.subset k) n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨r, read'⟩ := y
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hd, -⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hd
    unfold Flock.Draw.drawWith at hw
    dsimp only at hw
    rw [StateT.run_bind] at hw
    cases ho : (Flock.Draw.onWith (fun need => Spot.Draw.take src need)
        (2 * k * Flock.Draw.width n + 64) (Flock.Draw.subset k n)).run 0 with
    | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
    | ok z =>
      obtain ⟨v, p⟩ := z
      obtain ⟨j, D, rfl, -, hst⟩ := onWith_take src _ (by omega) _ 0 _ _ ho
      obtain ⟨-, -, hrun, -⟩ := hst
      rw [ho, bind_ok, StateT.run_pure] at hw
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
      obtain rfl := (Option.some.inj h1).symm
      obtain ⟨π, hk, hπ, hmem⟩ := FlockSoundness.Discharge.Exec.subset_spec hrun
      intro x hx
      obtain ⟨i, hi, rfl⟩ := (hmem x).1 hx
      exact hπ i (Nat.lt_of_lt_of_le hi hk)

end Flock.Bridge

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot.SecurityProofs (flockLaw_subset deriveLaw_subset)
open Lean (Json)

/-- **A first stage of count `K` proves units below its population**: with no closure map it proves the units it
drew (`select_proved`), a `subset K` sample over the population `Partition.units` gives. -/
theorem select_subset_lt {g : Reg} {K : ℕ} (hK : firstK g = some K) {src : ByteArray} {draw : Json}
    {proved : Array ℕ} {read : ℕ}
    (h : Spot.Select.select g.rule g.program g.query g.table g.closure src = .ok (draw, proved, read)) :
    ∃ n, Spot.Partition.units g.program g.query = .ok n ∧ ∀ x ∈ proved, x < n := by
  obtain ⟨hc, hne, n, named, hn, hl, hf⟩ := firstK_some hK
  refine ⟨n, hn, ?_⟩
  obtain ⟨n', named', derived, hn', hl', hder, units, hpu, hcase⟩ := select_proved _ _ _ _ _ _ _ _ _ h
  rw [hn] at hn'
  obtain rfl := Except.ok.inj hn'
  rw [hl] at hl'
  obtain rfl := Except.ok.inj hl'
  obtain rfl : derived = none := deriveLaw_subset _ _ _ _ _ (flockLaw_subset _ _ _ hf).1 hder
  rw [hpu hc]
  rcases hcase with ⟨hall, -⟩ | ⟨-, parsed, read', hp, hu⟩
  · exact absurd hall hne
  · have hp' : Spot.Draw.flockLaw named n = .ok parsed := hp
    rw [hf] at hp'
    obtain rfl : parsed = .subset K := (Except.ok.inj hp').symm
    have hu' : Spot.Draw.drawUnits (.subset K) n src = .ok (units, read') := hu
    exact _root_.Flock.Bridge.drawUnits_subset_lt hu'

end SpotFlock

namespace SamplingDeploy

open SpotFlock (firstK firstK_some select_subset_lt)

/-- **Every block of the replay sample is below the first stage's population**, where the first stage samples a
uniform `K`-subset: the sample is the units the first stage proves, and none where it is refused. -/
theorem RegTwo.sample_lt (g : RegTwo) {K : ℕ} (hK : firstK g.first = some K) (src : ByteArray) :
    ∃ n, Spot.Partition.units g.first.program g.first.query = .ok n ∧ ∀ ρ ∈ g.sample src, ρ < n := by
  obtain ⟨-, -, n, -, hn, -, -⟩ := firstK_some hK
  refine ⟨n, hn, ?_⟩
  unfold RegTwo.sample
  cases hs : Spot.Select.select g.first.rule g.first.program g.first.query g.first.table g.first.closure src with
  | error e => exact fun ρ hρ => absurd hρ (by simp)
  | ok r =>
    obtain ⟨draw, proved, read⟩ := r
    obtain ⟨n', hn', hlt⟩ := select_subset_lt hK hs
    rw [hn] at hn'
    obtain rfl := Except.ok.inj hn'
    exact hlt

end SamplingDeploy
