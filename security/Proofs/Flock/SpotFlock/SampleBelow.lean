module

public import Proofs.Flock.SpotFlock.HeldOfRun
public import Proofs.Flock.SpotFlock.RunsAs.Proved
public import Proofs.Flock.SpotFlock.Cap
public import Proofs.Flock.Claims.Sampling.Select.Misses
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF

/-!
# Every unit the first stage proves is below the population

`Select.select` checks the draw it builds (`Draw.checkDraw`) against the population it read (`Partition.units`), and a
checked draw's units ascend with the last below its population. So, with no closure map, every unit a first stage
proves is below the population; and the population `Slots.popOk` holds the layout to is the one `select` reads. Hence
every unit of the replay sample is a slot of the layout that names itself.

* `getD_mkObj_prev`: the entry before the last of a `Json.mkObj`, at a key the last does not have.
* `units_of_popOk`: under `Slots.popOk p query n`, `Partition.units p query` is `n`.
* `select_drawn`: a select under a law, with no closure map, read its population `n` and checked the draw it built
  against it.
* `select_below`: the units a select with no closure map proves are below the population it read.
* `sample_below`, `sample_slots`: at a registration whose first stage has no closure map and whose population check
  passes at `n`, every unit of the replay sample is below `n`, so a slot of the layout whose unit is itself
  (`block_of_popOk`).
* `sampleSlots_of`: the same at a held copy `heldOf` builds for an accepted two-stage run, which ran with no closure map
  (`accepted_steps`) and passed its population check (`population_of_accepted`).

With a closure map a first stage also proves the closure's units; `Draw.checkDraw` holds those below the population
too, and nothing here reads it.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs SpotFlock.RunsAs.Round2
open Lean (Json)

/-- `getD` of `Json.mkObj (l ++ [(k, v), (k', v')])` at `k` is `v` where `k'` is another key: the entry before the last
is the one the object holds at its key. -/
theorem getD_mkObj_prev (l : List (String × Json)) (k k' : String) (v v' : Json) (hne : k' ≠ k) :
    Spot.getD (Json.mkObj (l ++ [(k, v), (k', v')])) k = v := by
  have hl : l ++ [(k, v), (k', v')] = (l ++ [(k, v)]) ++ [(k', v')] := by simp
  have hc : ¬ compare k' k = .eq := fun h => hne (Std.LawfulEqCmp.compare_eq_iff_eq.1 h)
  rw [hl, Json.mkObj, SpotFlock.RunsAs.getD_obj, Std.TreeMap.Raw.ofList_eq_insertMany_empty,
    Std.TreeMap.Raw.insertMany_append,
    Std.TreeMap.Raw.insertMany_list_singleton,
    Std.TreeMap.Raw.getElem?_insert (Std.TreeMap.Raw.WF.insertMany Std.TreeMap.Raw.WF.emptyc)]
  simp only [hc, ↓reduceIte]
  rw [Std.TreeMap.Raw.insertMany_append, Std.TreeMap.Raw.insertMany_list_singleton,
    Std.TreeMap.Raw.getElem?_insert_self (Std.TreeMap.Raw.WF.insertMany Std.TreeMap.Raw.WF.emptyc)]
  rfl

/-- **The population `popOk` holds the layout to is the partition's count**: the layout's population is the evaluated
query's strata laid out unit by unit (`layout_ok`), which hold each of `[0, population)` once (`strataOf_perm`), so `n`
slots are `n` units. -/
theorem units_of_popOk {p : Spot.Program} {query : Json} {n : ℕ} (hp : Spot.Slots.popOk p query n = true) :
    Spot.Partition.units p query = .ok n := by
  unfold Spot.Slots.popOk at hp
  cases hl : Spot.Slots.layout p query with
  | error e =>
    rw [hl] at hp
    first | cases hp | simp at hp
  | ok l =>
    rw [hl] at hp
    have hpop : (Spot.Slots.popOf l).map (·.unit) = (List.range n).map some := of_decide_eq_true hp
    obtain ⟨rs, he, pop, rfl, hlays⟩ := Properties.Spot.SecurityProofs.layout_ok p query l hl
    have h1 := congrArg List.length (Properties.Spot.SecurityProofs.laysOut_units _ _ hlays).1
    have h2 := (Properties.Spot.SecurityProofs.strataOf_perm _ _
      (Properties.Spot.SecurityProofs.partition_evaluate_tiles p query rs he)).length_eq
    have h3 := congrArg List.length hpop
    change ((pop.toList.flatMap (·.toList)).map (·.unit)).length = _ at h3
    simp only [List.length_map, List.length_range, List.length_range'] at h1 h2 h3
    have hn : Flock.TemplateQuery.population rs = n := by omega
    rw [Properties.Spot.SecurityProofs.units_eq p query rs he, hn]

/-- **A select under a law, with no closure map, accepted**: it read the population `n`, its draw object is the law's
entries with `n` and the units it drew, and `checkDraw` passes that draw at `n`. -/
theorem select_drawn {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {src : ByteArray} {draw : Json} {units : Array ℕ} {read : ℕ} (hr : rule ≠ .all)
    (h : Spot.Select.select rule p query table none src = .ok (draw, units, read)) :
    ∃ (n : ℕ) (law : Json), Spot.Partition.units p query = .ok n ∧
      draw = Json.mkObj ((Spot.entries law).toList ++ [("population", Lean.toJson n), ("units", Spot.natsJson units)]) ∧
      ¬(!(Spot.Draw.checkDraw draw (Lean.toJson n) law).isEmpty) = true := by
  unfold Spot.Select.select at h
  cases rule with
  | all => exact absurd rfl hr
  | law t =>
    simp only [bind, Except.bind, pure, Except.pure, Array.isEmpty_empty, ↓reduceIte, Array.append_empty] at h
    repeat'
      first
      | split at h
      | focus (cases h; done)
      | focus (rename_i hthrow; cases hthrow; done)
      | (simp only [Except.ok.injEq, Prod.mk.injEq] at h
         obtain ⟨rfl, rfl, -⟩ := h
         exact ⟨_, _, by assumption, rfl, by assumption⟩)
  | twoStage g t k =>
    simp only [bind, Except.bind, pure, Except.pure, Array.isEmpty_empty, ↓reduceIte, Array.append_empty] at h
    repeat'
      first
      | split at h
      | focus (cases h; done)
      | focus (rename_i hthrow; cases hthrow; done)
      | (simp only [Except.ok.injEq, Prod.mk.injEq] at h
         obtain ⟨rfl, rfl, -⟩ := h
         exact ⟨_, _, by assumption, rfl, by assumption⟩)

/-- Naturals read as ascending integers whose last is below `n` are all below `n`. -/
theorem below_of_last {units : Array ℕ} {xs : Array Int} {n : ℕ} (hx : xs = units.map fun m : ℕ => (m : Int))
    (hasc : Properties.Spot.Ascending xs)
    (hlast : ∀ (h : 0 < xs.size), 0 ≤ xs[0] ∧ ∃ n', some (n : Int) = some n' ∧ xs[xs.size - 1] < n') :
    ∀ ρ ∈ units, ρ < n := by
  intro ρ hρ
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hρ
  have hsz : xs.size = units.size := by rw [hx, Array.size_map]
  obtain ⟨-, n', hn', hlt⟩ := hlast (by omega)
  obtain rfl := Option.some.inj hn'
  have hle : xs[i]'(by omega) ≤ xs[xs.size - 1]'(by omega) := by
    rcases Nat.lt_or_ge i (xs.size - 1) with h1 | h1
    · exact Int.le_of_lt (Verity.Sampled.ascending_lt hasc i (xs.size - 1) (by omega) (by omega) h1)
    · have h2 : i = xs.size - 1 := by omega
      simp [h2]
  have hxi : xs[i]'(by omega) = (units[i] : Int) := by simp [hx]
  omega

/-- **The units a select with no closure map proves are below the population it read**: under `All` they are
`[0, n)`; under a law the draw it built passes `Draw.checkDraw` at `n`, whose units ascend with the last below the
draw's population, `n`. -/
theorem select_below {rule : Spot.Select.Rule} {p : Spot.Program} {query : Json} {table : Option Json}
    {src : ByteArray} {draw : Json} {units : Array ℕ} {read : ℕ}
    (h : Spot.Select.select rule p query table none src = .ok (draw, units, read)) :
    ∃ n, Spot.Partition.units p query = .ok n ∧ ∀ ρ ∈ units, ρ < n := by
  by_cases hr : rule = .all
  · obtain ⟨n, named, derived, hn, -, -, units', hu, hcase⟩ :=
      SpotFlock.select_proved rule p query table none src draw units read h
    rcases hcase with ⟨-, hall⟩ | ⟨hne, -⟩
    · refine ⟨n, hn, fun ρ hρ => ?_⟩
      rw [hu rfl, hall] at hρ
      exact Array.mem_range.1 hρ
    · exact absurd hr hne
  · obtain ⟨n, law, hn, rfl, hcd⟩ := select_drawn hr h
    refine ⟨n, hn, ?_⟩
    have hcd' : Spot.Draw.checkDraw (Json.mkObj ((Spot.entries law).toList ++
        [("population", Lean.toJson n), ("units", Spot.natsJson units)])) (Lean.toJson n) law = #[] := by
      simpa using hcd
    obtain ⟨-, -, -, -, -, us, xs, hus, hxs, hasc, hlast, -⟩ := (Properties.Spot.checkDraw_nil _ _ _).1 hcd'
    have hu : Spot.getD (Json.mkObj ((Spot.entries law).toList ++
        [("population", Lean.toJson n), ("units", Spot.natsJson units)])) "units" = Spot.natsJson units := by
      rw [show (Spot.entries law).toList ++ [("population", Lean.toJson n), ("units", Spot.natsJson units)] =
        ((Spot.entries law).toList ++ [("population", Lean.toJson n)]) ++ [("units", Spot.natsJson units)] by simp]
      exact SpotFlock.RunsAs.getD_mkObj_last _ _ _
    have hpv : Spot.getD (Json.mkObj ((Spot.entries law).toList ++
        [("population", Lean.toJson n), ("units", Spot.natsJson units)])) "population" = Lean.toJson n :=
      getD_mkObj_prev _ _ _ _ _ (by decide)
    rw [hu] at hus
    simp only [Spot.natsJson, Json.arr.injEq] at hus
    subst hus
    have hx : xs = units.map fun m : ℕ => (m : Int) := by
      have hf : (Spot.int? ∘ fun m : ℕ => Lean.toJson m) = some ∘ fun m : ℕ => (m : Int) := by
        funext m
        exact Properties.Spot.toJson_nat_int m
      rw [← Properties.Spot.Credited.filterMap_int_arr hxs]
      apply Array.toList_inj.1
      rw [Array.toList_filterMap, Array.toList_map, Array.toList_map, List.filterMap_map, hf,
        List.filterMap_eq_map]
    rw [hpv, Properties.Spot.toJson_nat_int] at hlast
    exact below_of_last hx hasc hlast

/-- **Every unit of the replay sample is below the population**, at a registration whose first stage has no closure map
and whose population check (`Slots.popOk`) passes at `n`, on any bytes. -/
theorem sample_below {g : RegTwo} {n : ℕ} (hc : g.first.closure = none)
    (hp : Spot.Slots.popOk g.first.program g.first.query n = true) (src : ByteArray) :
    ∀ ρ ∈ g.sample src, ρ < n := by
  intro ρ hρ
  unfold SamplingDeploy.RegTwo.sample at hρ
  rw [hc] at hρ
  cases hs : Spot.Select.select g.first.rule g.first.program g.first.query g.first.table none src with
  | error e =>
    rw [hs] at hρ
    replace hρ : ρ ∈ (#[] : Array ℕ) := hρ
    simp at hρ
  | ok r =>
    obtain ⟨draw, proved, read⟩ := r
    rw [hs] at hρ
    replace hρ : ρ ∈ proved := hρ
    obtain ⟨m, hm, hlt⟩ := select_below hs
    rw [units_of_popOk hp] at hm
    obtain rfl := Except.ok.inj hm
    exact hlt ρ hρ

/-- **Every unit of the replay sample is a slot of the layout that names itself**, at a registration whose first stage
has no closure map and whose population check passes at `n`. -/
theorem sample_slots {g : RegTwo} {n : ℕ} (hc : g.first.closure = none)
    (hp : Spot.Slots.popOk g.first.program g.first.query n = true) :
    ∀ l, Spot.Slots.layout g.first.program g.first.query = .ok l →
      ∀ c : ByteArray, ∀ ρ ∈ g.sample c, ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ :=
  fun _ hl c ρ hρ => block_of_popOk hl hp (sample_below hc hp c ρ hρ)

/-- The first stage `heldOf` reads has the run's closure map. -/
theorem firstOf_closure {i : Spot.Run.Input} {first : Reg} (hf : firstOf i = some first) :
    first.closure = i.closure := by
  unfold firstOf at hf
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hf
  obtain ⟨n, hn, rule, hrule, law, hlaw, workTable, hwt, resolved, hres, rfl⟩ := hf
  rfl

/-- **Every unit of the replay sample is a slot of the layout that names itself** at a held copy `heldOf` builds for an
accepted two-stage run, which ran with no closure map and passed its population check. -/
theorem sampleSlots_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held} {j : Json}
    (hh : heldOf i it u = some h) (ha : AcceptedTwo i it j) :
    ∀ l, Spot.Slots.layout h.reg.first.program h.reg.first.query = .ok l →
      ∀ c : ByteArray, ∀ ρ ∈ h.reg.sample c, ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ := by
  obtain ⟨n, -, hp⟩ := population_of_accepted ha
  obtain ⟨first, ns, ks, l₀, hz, hf, -, -, rfl, -⟩ := heldOf_spec hh
  obtain ⟨hpr, hq⟩ := firstOf_program hf
  rw [← hpr, ← hq] at hp
  have hcl : first.closure = none := (firstOf_closure hf).trans (accepted_steps ha).1
  exact sample_slots (g := (withFrame (held₀ i it u first ns ks (requests l₀) hz) u.c₀).reg) hcl hp

end SpotFlock.HeldOf
