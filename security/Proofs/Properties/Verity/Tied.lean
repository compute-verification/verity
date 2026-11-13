module

public import Properties.Verity.Tied
public import Proofs.Nci.Untied

/-!
# The tied circuit's blocks: the placement from below, lean's form of the policy on it, and the period's pooling

* `blocks_unit_emb`: a tied cell's y gate is in its home block.
* `fromBelow`: the placement is from below (`Verity.ValueTie.Tying.FromBelow`) at every blocks, homes and untied
  cells.
* `novelReads_asInputs_eq`: a block of the tied circuit with the untied gates `T` as inputs
  (`Verity.RegisteredCircuit.asInputs`, `Verity.Partition.asInputs`) reads, outside `θ`, what the block but `T` reads
  in the tied circuit, so `Tying.CompliesBut` is the per-block budget `counts_asInputs` gives a block: its own `k`, and
  the gates of `T` it holds.
* `periodOutsideG`: `Verity.ValueTie.PeriodOutsideG`.
-/

@[expose] public section

namespace Verity.ValueTie

open Finset

namespace Layout

variable {C : Verity.Circuit} (L : Layout C)

theorem mem_yAt {c : Cell} {g : Fin C.N} : g ∈ L.yAt c ↔ g ∈ L.y ∧ L.cellOf g = c := mem_filter

theorem mem_yOf {U : Finset Cell} {g : Fin C.N} : g ∈ L.yOf U ↔ g ∈ L.y ∧ L.cellOf g ∈ U := mem_filter

end Layout

namespace Tying

variable {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Layout C} (T : Tying C θ L)

/-- **A tied cell's y gate is in its home block.** -/
theorem blocks_unit_emb {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) {g : Fin C.N}
    (hg : g ∈ L.yOf T.cells) : (T.blocks Q home).unit (T.emb g) = some (home (L.cellOf g)) := by
  have hs : ∃ g', T.emb g' = T.emb g := ⟨g, rfl⟩
  have hc : hs.choose = g := T.emb_inj hs.choose_spec
  unfold Tying.blocks
  simp only []
  rw [dif_pos hs, hc, if_pos hg]

/-- **The placement is from below**, at any blocks `Q`, homes `home` and untied cells `U`: a placed cell's gates are
its 32 bits in its home block, an untied cell's are untied gates, and distinct cells sit at disjoint gates. -/
theorem fromBelow {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) (U : Finset Cell) :
    T.FromBelow Q home U := by
  refine ⟨fun c u O h => ?_, fun c hcU u O h => ?_, fun c c' u u' O O' hne h h' => ?_⟩
  · unfold Tying.place at h
    by_cases hc : c ∈ T.cells
    · rw [if_pos hc] at h
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      refine ⟨fun x hx => ?_, ?_⟩
      · obtain ⟨g, hg, rfl⟩ := mem_map.1 hx
        obtain ⟨hgy, hgc⟩ := L.mem_yAt.1 hg
        have hgo : g ∈ L.yOf T.cells := L.mem_yOf.2 ⟨hgy, by rw [hgc]; exact hc⟩
        rw [Partition.mem_gates, T.blocks_unit_emb Q home hgo, hgc]
      · show ∑ x ∈ (L.yAt c).map ⟨T.emb, T.emb_inj⟩, T.tied.D.w x = 32
        rw [sum_map]
        simp only [Function.Embedding.coeFn_mk, T.emb_w]
        exact L.width_word c (T.cells_sub hc)
    · rw [if_neg hc] at h
      cases h
  · unfold Tying.place at h
    by_cases hc : c ∈ T.cells
    · rw [if_pos hc] at h
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      intro x hx
      obtain ⟨g, hg, rfl⟩ := mem_map.1 hx
      obtain ⟨hgy, hgc⟩ := L.mem_yAt.1 hg
      unfold Tying.untiedGates
      exact mem_map_of_mem _ (L.mem_yOf.2 ⟨hgy, by rw [hgc]; exact hcU⟩)
    · rw [if_neg hc] at h
      cases h
  · unfold Tying.place at h h'
    by_cases hc : c ∈ T.cells
    · by_cases hc' : c' ∈ T.cells
      · rw [if_pos hc] at h
        rw [if_pos hc'] at h'
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h')
        rw [disjoint_map]
        exact disjoint_left.2 fun g hg hg' => hne ((L.mem_yAt.1 hg).2.symm.trans (L.mem_yAt.1 hg').2)
      · rw [if_neg hc'] at h'
        cases h'
    · rw [if_neg hc] at h
      cases h

end Tying

/-- **With computed gates `T` as inputs, a block reads, outside `θ`, what the block but `T` reads.** On any registered
circuit `d`, any partition `P` and computed gates `T`, block `u` of `P` with `T` as inputs reads in
`d.asInputs T` the fresh input that `P`'s block `u` but `T` reads in `d`. So `Tying.CompliesBut`'s bound on a block
but its untied gates is the budget `counts_asInputs` takes for the block on the circuit with them as inputs. -/
theorem novelReads_asInputs_eq {m : ℕ} (d : Verity.RegisteredCircuit) (P : Verity.Partition d.D m)
    (T : Finset (Fin d.D.N)) (hT : ∀ g ∈ T, d.D.isInput g = false) (u : Fin m) :
    Nci.novelReads (d.asInputs T) ((P.asInputs T).gates u) = Nci.novelReads d (P.gates u \ T) := by
  have hg : (P.asInputs T).gates u = P.gates u \ T := P.asInputs_gates T u
  have hθ : ∀ a, a ∉ d.θ \ T ↔ a ∉ d.θ := fun a => by
    constructor
    · intro h hθa
      by_cases haT : a ∈ T
      · have h' := hT a haT
        rw [d.θ_input a hθa] at h'
        cases h'
      · exact h (mem_sdiff.2 ⟨hθa, haT⟩)
    · intro h hθa
      exact h (mem_sdiff.1 hθa).1
  have hset : (d.D.asInputs T).into (P.gates u \ T) \ (d.θ \ T) = d.D.into (P.gates u \ T) \ d.θ := by
    ext a
    rw [mem_sdiff, mem_sdiff, Nci.Cut.SecurityProofs.mem_into, Nci.Cut.SecurityProofs.mem_into]
    constructor
    · rintro ⟨⟨ha, g, hgu, hag⟩, hθa⟩
      have h : a ∈ (if g ∈ T then ∅ else d.D.reads g) := hag
      rw [if_neg (mem_sdiff.1 hgu).2] at h
      exact ⟨⟨ha, g, hgu, h⟩, (hθ a).1 hθa⟩
    · rintro ⟨⟨ha, g, hgu, hag⟩, hθa⟩
      refine ⟨⟨ha, g, hgu, ?_⟩, (hθ a).2 hθa⟩
      show a ∈ (if g ∈ T then ∅ else d.D.reads g)
      rw [if_neg (mem_sdiff.1 hgu).2]
      exact hag
  show (d.D.asInputs T).width ((d.D.asInputs T).into ((P.asInputs T).gates u) \ (d.θ \ T)) =
    d.D.width (d.D.into (P.gates u \ T) \ d.θ)
  rw [hg, hset]
  rfl

/-- **One window's blocks outside its `G`, charged whole**: `w + 1` times their charge is at most `c` times the
window's untied words, when no block is charged more than `c`. -/
theorem Window.outside_le (W : Window) (w c : ℕ) (hc : ∀ u, W.charge u ≤ c) :
    (w + 1) * W.outside w ≤ c * W.untied.card := by
  have h1 : W.outside w ≤ (W.heavy w).card * c := by
    unfold Window.outside
    simpa using Finset.sum_le_card_nsmul (W.heavy w) W.charge c fun u _ => hc u
  have h2 : (W.heavy w).card * (w + 1) ≤ ∑ u ∈ W.heavy w, (W.untied.filter fun x => W.home x = u).card := by
    have := Finset.card_nsmul_le_sum (W.heavy w) (fun u => (W.untied.filter fun x => W.home x = u).card) (w + 1)
      fun u hu => by
        unfold Window.heavy at hu
        exact (mem_filter.1 hu).2
    simpa using this
  have h3 : ∑ u ∈ W.heavy w, (W.untied.filter fun x => W.home x = u).card ≤ W.untied.card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := W.home) (t := univ) fun _ _ => mem_univ _]
    exact Finset.sum_le_sum_of_subset (subset_univ _)
  calc (w + 1) * W.outside w ≤ (w + 1) * ((W.heavy w).card * c) := Nat.mul_le_mul_left _ h1
    _ = c * ((W.heavy w).card * (w + 1)) := by ring
    _ ≤ c * W.untied.card := Nat.mul_le_mul_left _ (h2.trans h3)

/-- **`PeriodOutsideG` holds**: the windows' bounds (`Window.outside_le`), summed over the period. -/
theorem periodOutsideG : PeriodOutsideG := by
  intro n W w c hc
  unfold periodOutside
  rw [Finset.mul_sum, Finset.mul_sum]
  exact Finset.sum_le_sum fun k _ => Window.outside_le (W k) w c (hc k)

end Verity.ValueTie
