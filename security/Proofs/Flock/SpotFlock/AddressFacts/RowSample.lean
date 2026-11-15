module

public import Proofs.Flock.SpotFlock.AddressFacts.RowPlace
public import Proofs.Flock.Claims.Sampling.Select.Misses

/-!
# The sampled outer blocks are slots of the layout, from the population check

`AddressRow.SampleSlots h` (`RowPlace`) says each outer block the first stage's sample holds is a slot of the
registration's layout whose unit is the outer block itself. The run decides half of it: it refuses a registration
whose layout's population is not its `n` units in order (`Slots.popOk`, `population-order`), so every index below `n`
is such a slot. What is left is that the first stage samples below `n`.

* `units_of_layout`: where the layout is defined, the population `Partition.units` returns is the count of the
  layout's population slots. The layout lays the query's strata out unit by unit (`layout_ok`), and the strata are
  the units `0` to the population less one, each once (`strataOf_perm`).
* `popOf_units_of_popOk`, `popOf_length_of_popOk`, `slot_of_popOk`: under the population check at `n`, the layout's
  population slots name the units `0` to `n - 1` in order, there are `n` of them, and every index below `n` is a slot
  naming itself.
* `units_of_popOk`: under the population check at `n`, `Partition.units` returns `n`. So "the first stage samples
  below the population `Partition.units` returns" is "below `n`".
* `sampleSlots_of_popOk`: `SampleSlots h` from the population check at `n` and a sample below `n`.
* `sampleSlots_of_units`: the same from a sample below the population `Partition.units` returns, the form a first
  stage's sample is bounded in.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

/-- **The population is the count of the layout's population slots**: where the layout of `p` under `q` is defined,
`Partition.units` returns the length of its population. -/
theorem units_of_layout {p : _root_.Spot.Program} {q : Json} {l : _root_.Spot.Slots.SlotOrder}
    (hl : _root_.Spot.Slots.layout p q = .ok l) :
    _root_.Spot.Partition.units p q = .ok (_root_.Spot.Slots.popOf l).length := by
  obtain ⟨rs, he, pop, hlpop, hlays⟩ := _root_.Properties.Spot.SecurityProofs.layout_ok p q l hl
  have hperm := _root_.Properties.Spot.SecurityProofs.strataOf_perm _ _
    (_root_.Properties.Spot.SecurityProofs.partition_evaluate_tiles p q rs he)
  have hlenU := hperm.length_eq.trans List.length_range'
  have hu := (_root_.Properties.Spot.SecurityProofs.laysOut_units _ _ hlays).1
  have hlenB : (pop.toList.flatMap (·.toList)).length = _root_.Flock.TemplateQuery.population rs := by
    have h1 := congrArg List.length hu
    simp only [List.length_map] at h1
    exact h1.trans hlenU
  have hpopOf : _root_.Spot.Slots.popOf l = pop.toList.flatMap (·.toList) := by
    subst hlpop
    rfl
  have hcount : _root_.Flock.TemplateQuery.population rs = (_root_.Spot.Slots.popOf l).length :=
    ((congrArg List.length hpopOf).trans hlenB).symm
  rw [_root_.Properties.Spot.SecurityProofs.units_eq p q rs he, hcount]

/-- Under the population check at `n`, the layout is defined. -/
theorem layout_of_popOk {p : _root_.Spot.Program} {q : Json} {n : ℕ}
    (hp : _root_.Spot.Slots.popOk p q n = true) : ∃ l, _root_.Spot.Slots.layout p q = .ok l := by
  cases hl : _root_.Spot.Slots.layout p q with
  | ok l => exact ⟨l, rfl⟩
  | error e =>
    unfold _root_.Spot.Slots.popOk at hp
    rw [hl] at hp
    cases hp

/-- Under the population check at `n`, the layout's population slots name the units `0` to `n - 1`, in order. -/
theorem popOf_units_of_popOk {p : _root_.Spot.Program} {q : Json} {n : ℕ} {l : _root_.Spot.Slots.SlotOrder}
    (hl : _root_.Spot.Slots.layout p q = .ok l) (hp : _root_.Spot.Slots.popOk p q n = true) :
    (_root_.Spot.Slots.popOf l).map (·.unit) = (List.range n).map some := by
  unfold _root_.Spot.Slots.popOk at hp
  rw [hl] at hp
  exact of_decide_eq_true hp

/-- Under the population check at `n`, the layout's population has `n` slots. -/
theorem popOf_length_of_popOk {p : _root_.Spot.Program} {q : Json} {n : ℕ} {l : _root_.Spot.Slots.SlotOrder}
    (hl : _root_.Spot.Slots.layout p q = .ok l) (hp : _root_.Spot.Slots.popOk p q n = true) :
    (_root_.Spot.Slots.popOf l).length = n := by
  have h1 := congrArg List.length (popOf_units_of_popOk hl hp)
  simp only [List.length_map, List.length_range] at h1
  exact h1

/-- **The population check at `n` fixes the population**: `Partition.units` returns `n`. -/
theorem units_of_popOk {p : _root_.Spot.Program} {q : Json} {n : ℕ}
    (hp : _root_.Spot.Slots.popOk p q n = true) : _root_.Spot.Partition.units p q = .ok n := by
  obtain ⟨l, hl⟩ := layout_of_popOk hp
  rw [units_of_layout hl, popOf_length_of_popOk hl hp]

/-- **Every index below `n` is a slot naming itself**, under the population check at `n`. -/
theorem slot_of_popOk {p : _root_.Spot.Program} {q : Json} {n : ℕ} {l : _root_.Spot.Slots.SlotOrder}
    (hl : _root_.Spot.Slots.layout p q = .ok l) (hp : _root_.Spot.Slots.popOk p q n = true) {ρ : ℕ} (hρ : ρ < n) :
    ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ := by
  have hu := popOf_units_of_popOk hl hp
  obtain ⟨rs, -, pop, hlpop, -⟩ := _root_.Properties.Spot.SecurityProofs.layout_ok p q l hl
  have hpopOf : _root_.Spot.Slots.popOf l = _root_.Properties.Spot.SecurityProofs.popBlocks pop := by
    subst hlpop
    rfl
  have hbl : l.blocks.toList = _root_.Properties.Spot.SecurityProofs.popBlocks pop ++
      _root_.Properties.Spot.SecurityProofs.popBlocks (_root_.Properties.Spot.SecurityProofs.inputClass p) := by
    subst hlpop
    exact _root_.Properties.Spot.SecurityProofs.blocks_toList pop _
  rw [hpopOf] at hu
  have hlen : (_root_.Properties.Spot.SecurityProofs.popBlocks pop).length = n := by
    have h1 := congrArg List.length hu
    simp only [List.length_map, List.length_range] at h1
    exact h1
  have hρ' : ρ < (_root_.Properties.Spot.SecurityProofs.popBlocks pop).length := by
    rw [hlen]
    exact hρ
  refine ⟨(_root_.Properties.Spot.SecurityProofs.popBlocks pop)[ρ], ?_, ?_⟩
  · rw [← Array.getElem?_toList, hbl, List.getElem?_append_left hρ', List.getElem?_eq_getElem hρ']
  · have h1 := congrArg (fun L : List (Option ℕ) => L[ρ]?) hu
    simp only [List.getElem?_map, List.getElem?_eq_getElem hρ', List.getElem?_range hρ, Option.map_some,
      Option.some.injEq] at h1
    exact h1

/-- **The first stage samples slots that name themselves** (`SampleSlots`), from the run's population check at `n`
and a first stage that samples below `n`. -/
theorem sampleSlots_of_popOk {h : Held} {n : ℕ}
    (hp : _root_.Spot.Slots.popOk h.reg.first.program h.reg.first.query n = true)
    (hlt : ∀ c : ByteArray, ∀ ρ ∈ h.reg.sample c, ρ < n) : SampleSlots h := by
  intro l hl c ρ hρ
  exact slot_of_popOk hl hp (hlt c ρ hρ)

/-- **`SampleSlots`, from the population check and a sample below the population**: the same, with the first
stage's sample below the population `Partition.units` returns, which the population check holds to `n`. -/
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
