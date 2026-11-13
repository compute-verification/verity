module

public import Proofs.Flock.SpotFlock.AddressFacts.RowPlace
public import Proofs.Flock.SpotFlock.HeldOfRun
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.Claims.Sampling.Select.Misses
public import Proofs.Flock.Claims.Sampling.Select.Layout

/-!
# The sampled outer blocks are slots of the layout, from the population check

`AddressRow.SampleSlots h` (`RowPlace`) is the premise `sits_of_slots` takes of the registration: each outer block the
first stage samples is a slot of the layout that names itself. This file reduces it to two facts a run decides.

* `units_of_popOk`: a program and query that pass the population check at `n` (`Slots.popOk`, which the run refuses
  a boundary registration without) have a population of `n` units (`Partition.units`). The layout's population is
  the query's strata laid out unit by unit (`layout_ok`), the strata hold each unit of the query's ranges once
  (`strataOf_perm`, `partition_evaluate_tiles`), and the check holds the layout's population to `n` proof blocks.
* `sampleSlots_of_popOk`: under the population check at `n`, every index below `n` is a slot naming itself
  (`HeldOf.block_of_popOk`), so `SampleSlots h` holds where every sampled outer block is below `n`.
* `sampleSlots_of_units`: the same where every sampled outer block is below the population `Partition.units`
  returns, which is the form a first stage that draws its units from that population gives.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

/-- **The population check fixes the population**: a program and query that pass `Slots.popOk` at `n` have `n`
units. -/
theorem units_of_popOk {p : _root_.Spot.Program} {q : Json} {n : ℕ}
    (hp : _root_.Spot.Slots.popOk p q n = true) : _root_.Spot.Partition.units p q = .ok n := by
  obtain ⟨l, hl⟩ : ∃ l, _root_.Spot.Slots.layout p q = .ok l := by
    cases hl : _root_.Spot.Slots.layout p q with
    | ok l => exact ⟨l, rfl⟩
    | error e =>
      unfold _root_.Spot.Slots.popOk at hp
      rw [hl] at hp
      first
        | cases hp
        | simp at hp
  have hn : (_root_.Spot.Slots.popOf l).length = n := HeldOf.length_popOf hl hp
  obtain ⟨rs, he, pop, hlpop, hlays⟩ := Properties.Spot.SecurityProofs.layout_ok p q l hl
  have hperm := Properties.Spot.SecurityProofs.strataOf_perm _ _
    (Properties.Spot.SecurityProofs.partition_evaluate_tiles p q rs he)
  have hunits := congrArg List.length (Properties.Spot.SecurityProofs.laysOut_units _ _ hlays).1
  simp only [List.length_map] at hunits
  have hlen : (pop.toList.flatMap (·.toList)).length = _root_.Flock.TemplateQuery.population rs :=
    hunits.trans (hperm.length_eq.trans List.length_range')
  have hpopOf : _root_.Spot.Slots.popOf l = pop.toList.flatMap (·.toList) := by
    subst hlpop
    rfl
  have hcount : _root_.Flock.TemplateQuery.population rs = n :=
    (hlen.symm.trans (congrArg List.length hpopOf).symm).trans hn
  exact (Properties.Spot.SecurityProofs.units_eq p q rs he).trans (congrArg Except.ok hcount)

/-- **The sampled outer blocks are slots naming themselves**, where the registration passes the population check at
`n` and every outer block the first stage samples is below `n`. -/
theorem sampleSlots_of_popOk {h : Held} {n : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hlt : ∀ c : ByteArray, ∀ ρ ∈ h.reg.sample c, ρ < n) : SampleSlots h := by
  intro l hl c ρ hρ
  exact HeldOf.block_of_popOk hl hp (hlt c ρ hρ)

/-- **The sampled outer blocks are slots naming themselves**, where the registration passes the population check and
every outer block the first stage samples is below the population. -/
theorem sampleSlots_of_units {h : Held} {n : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hlt : ∀ c : ByteArray, ∃ m, _root_.Spot.Partition.units h.reg.first.program h.reg.first.query = .ok m ∧
      ∀ ρ ∈ h.reg.sample c, ρ < m) : SampleSlots h := by
  refine sampleSlots_of_popOk hp fun c ρ hρ => ?_
  obtain ⟨m, hm, hlt'⟩ := hlt c
  have hmn : m = n := Except.ok.inj (hm.symm.trans (units_of_popOk hp))
  exact Nat.lt_of_lt_of_eq (hlt' ρ hρ) hmn

end AddressRow

end SpotFlock
