module

public import Proofs.Flock.SpotFlock.HeldOfCheck
public import Proofs.Flock.SpotFlock.ScopeDesc

/-!
# What the run's population check gives the held copies

`descOk` (`ScopeDesc`) decides the two-stage scope at the members from the descriptor; two of its checks,
`filesOk` and `fillOk`, take a step per population slot and unit. At the copies `heldOf` builds, neither is a new check:
the run refuses a boundary registration whose population is not its `n` units in order (`Slots.popOk`,
`population-order`), and given that, both hold by construction, since `heldOf` gives template `t` the units
`[0, n·n_t)` and request `ρ`'s block of it the slots `filesBlockOf … ρ t`.

* `pop_facts`: under `popOk`, the layout's blocks are its population, units `0` to `n - 1` in slot order, then input
  slots, which name no unit (`layout_shape`).
* `unit_of_popOk`, `block_of_popOk`, `requests_of_popOk`: a slot naming a unit names its own index, below `n`; every
  index below `n` is such a slot; and `n` slots name a unit.
* `filesOk_held₀`, `fillOk_held₀`: `filesOk` and `fillOk` at the copies, from `popOk` and, for `filesOk`, the cells
  naming one count per template.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs
open Lean (Json)

/-- **A layout's classes**: its population, then a class of input slots, which name no unit. -/
theorem layout_shape {p : Spot.Program} {query : Json} {l : Spot.Slots.Layout}
    (h : Spot.Slots.layout p query = .ok l) :
    ∃ pop ins, l.classes = #[pop, ins] ∧ ∀ s ∈ ins, ∀ b ∈ s, b.unit = none := by
  unfold Spot.Slots.layout at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h
       refine ⟨_, _, rfl, ?_⟩
       simp)

theorem pop_facts {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.Layout}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) :
    ∃ ins : Array (Array Spot.Slots.ProofBlock), (∀ s ∈ ins, ∀ b ∈ s, b.unit = none) ∧
      l.blocks.toList = Spot.Slots.popOf l ++ (ins.toList.map Array.toList).flatten ∧
      (Spot.Slots.popOf l).map (·.unit) = (List.range n).map some := by
  obtain ⟨pop, ins, hc, hins⟩ := layout_shape hl
  refine ⟨ins, hins, ?_, ?_⟩
  · unfold Spot.Slots.Layout.blocks Spot.Slots.popOf
    rw [hc]
    simp
  · unfold Spot.Slots.popOk at hp
    rw [hl] at hp
    exact of_decide_eq_true hp

theorem length_popOf {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.Layout}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) :
    (Spot.Slots.popOf l).length = n := by
  obtain ⟨-, -, -, hpop⟩ := pop_facts hl hp
  simpa using congrArg List.length hpop

/-- **A slot naming a unit names its own index**, below `n`, under `popOk`. -/
theorem unit_of_popOk {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.Layout}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true)
    {ρ : ℕ} {b : Spot.Slots.ProofBlock} (hb : l.blocks[ρ]? = some b) {v : ℕ} (hv : b.unit = some v) :
    v = ρ ∧ ρ < n := by
  obtain ⟨ins, hins, hbl, hpop⟩ := pop_facts hl hp
  have hlen := length_popOf hl hp
  rw [← Array.getElem?_toList, hbl] at hb
  rcases Nat.lt_or_ge ρ (Spot.Slots.popOf l).length with hρ | hρ
  · rw [List.getElem?_append_left hρ] at hb
    have h1 : ((Spot.Slots.popOf l).map (·.unit))[ρ]? = some (some v) := by
      rw [List.getElem?_map, hb]
      simp [hv]
    rw [hpop, List.getElem?_map, List.getElem?_range (by omega)] at h1
    simp only [Option.map_some, Option.some.injEq] at h1
    exact ⟨h1.symm, by omega⟩
  · rw [List.getElem?_append_right hρ] at hb
    have hmem := List.mem_of_getElem? hb
    simp only [List.mem_flatten, List.mem_map] at hmem
    obtain ⟨_, ⟨s, hs, rfl⟩, hbs⟩ := hmem
    have hn := hins s (by simpa using hs) b (by simpa using hbs)
    rw [hn] at hv
    cases hv

/-- **Every index below `n` is a slot naming itself**, under `popOk`. -/
theorem block_of_popOk {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.Layout}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) {ρ : ℕ} (hρ : ρ < n) :
    ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ := by
  obtain ⟨ins, -, hbl, hpop⟩ := pop_facts hl hp
  have hρ' : ρ < (Spot.Slots.popOf l).length := by rw [length_popOf hl hp]; exact hρ
  refine ⟨(Spot.Slots.popOf l)[ρ], ?_, ?_⟩
  · rw [← Array.getElem?_toList, hbl, List.getElem?_append_left hρ', List.getElem?_eq_getElem hρ']
  · have h1 := congrArg (·[ρ]?) hpop
    simp only [List.getElem?_map, List.getElem?_eq_getElem hρ', List.getElem?_range hρ, Option.map_some,
      Option.some.injEq] at h1
    exact h1

/-- **`n` slots name a unit**, under `popOk`: `heldOf`'s request count is the population. -/
theorem requests_of_popOk {p : Spot.Program} {query : Json} {n : ℕ} {l : Spot.Slots.Layout}
    (hl : Spot.Slots.layout p query = .ok l) (hp : Spot.Slots.popOk p query n = true) : requests l = n := by
  obtain ⟨ins, hins, hbl, hpop⟩ := pop_facts hl hp
  have h1 : (Spot.Slots.popOf l).filter (fun b => b.unit.isSome) = Spot.Slots.popOf l := by
    refine List.filter_eq_self.2 fun b hb => ?_
    have hm : b.unit ∈ (List.range n).map some := hpop ▸ List.mem_map_of_mem hb
    obtain ⟨k, -, hk⟩ := List.mem_map.1 hm
    simp [← hk]
  have h2 : ((ins.toList.map Array.toList).flatten).filter (fun b => b.unit.isSome) = [] := by
    refine List.filter_eq_nil_iff.2 fun b hb => ?_
    simp only [List.mem_flatten, List.mem_map] at hb
    obtain ⟨_, ⟨s, hs, rfl⟩, hbs⟩ := hb
    simp [hins s (by simpa using hs) b (by simpa using hbs)]
  unfold requests
  rw [← Array.length_toList, Array.toList_filter, hbl, List.filter_append, h1, h2, List.append_nil,
    length_popOf hl hp]

theorem getElem?_ms_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg}
    {ns ks : Array ℕ} {r : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true} {t : ℕ} (ht : t < it.circuits.size) :
    ((held₀ i it u first ns ks r hz).ms[t]?).map (·.2.size) = some (r * ns.getD t 0) := by
  simp [Held.ms, held₀, unitsOf, ht, List.getD_eq_getElem?_getD]

/-- **The layout fits the members' files** (`filesOk`) at the copies `heldOf` builds, from the population check and
the cells naming one count per template. -/
theorem filesOk_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg} {ns ks : Array ℕ}
    {l : Spot.Slots.Layout} {n : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true}
    (hl : Spot.Slots.layout first.program first.query = .ok l)
    (hp : Spot.Slots.popOk first.program first.query n = true) (hT : ns.size = it.circuits.size) :
    filesOk (held₀ i it u first ns ks (requests l) hz).reg (held₀ i it u first ns ks (requests l) hz).ms l
      (held₀ i it u first ns ks (requests l) hz).k = true := by
  have hr := requests_of_popOk hl hp
  have hlen := ms_length_held₀ i it u first ns ks (requests l) hz
  unfold filesOk
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range]
  refine ⟨⟨?_, fun t _ => ?_⟩, fun ρ _ => ?_⟩
  · rw [hlen]
    exact hT
  · show ks.toList.getD t 0 = ks.getD t 0
    by_cases h : t < ks.size <;> simp [List.getD_eq_getElem?_getD, Array.getD, h]
  · rcases hb : l.blocks[ρ]? with _ | b
    · rfl
    · dsimp only
      rcases hv : b.unit with _ | v
      · rfl
      · dsimp only
        obtain ⟨rfl, hρ⟩ := unit_of_popOk hl hp hb hv
        simp only [beq_self_eq_true, Bool.true_and, List.all_eq_true, List.mem_range, Bool.and_eq_true,
          decide_eq_true_eq, beq_iff_eq]
        intro t ht
        rw [hlen] at ht
        refine ⟨?_, fun _ _ => rfl⟩
        rw [getElem?_ms_held₀ ht, Option.getD_some, hr]
        exact Nat.mul_le_mul_right _ hρ

/-- **The layout fills the members' files** (`fillOk`) at the copies `heldOf` builds, from the population check. -/
theorem fillOk_held₀ {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {first : Reg} {ns ks : Array ℕ}
    {l : Spot.Slots.Layout} {n : ℕ} {hz : ∀ t, zkOk (refStmt u.xs t) = true}
    (hl : Spot.Slots.layout first.program first.query = .ok l)
    (hp : Spot.Slots.popOk first.program first.query n = true) :
    fillOk (held₀ i it u first ns ks (requests l) hz).ms ns l = true := by
  have hr := requests_of_popOk hl hp
  unfold fillOk
  simp only [List.all_eq_true, List.mem_range, Bool.and_eq_true, decide_eq_true_eq]
  intro t ht x hx
  rw [ms_length_held₀] at ht
  rw [getElem?_ms_held₀ ht, Option.getD_some, hr] at hx
  have hn : 0 < ns.getD t 0 := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hx; omega
  obtain ⟨b, hb, hu⟩ := block_of_popOk hl hp (Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hx))
  refine ⟨hn, ?_⟩
  rw [hb]
  simp [hu]

end SpotFlock.HeldOf
