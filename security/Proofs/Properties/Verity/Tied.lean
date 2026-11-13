module

public import Properties.Verity.Tied
public import Proofs.Nci.Untied

/-!
# The tied circuit's blocks: the placement from below, lean's form of the policy on it, and the period's pooling

* `blocks_unit_emb`: a tied cell's y gate is in its home block.
* `into_blocks_sub`, `compliesBut_of_file`: where the homes hold the tied cells' A rows (`Tying.Homes`), a block of
  the tied circuit reads no fresh input its registered block doesn't, so a registered file whose blocks comply makes
  the tied circuit's comply (`Tying.CompliesBut`) at the same `Γ` and `α`, whatever cells are untied: the honest
  prover passes.
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

namespace Cut

variable {C : Verity.Circuit} (L : Cut C)

theorem mem_yAt {c : Cell} {g : Fin C.N} : g ∈ L.yAt c ↔ g ∈ L.y ∧ L.cellOf g = c := mem_filter

theorem mem_yOf {U : Finset Cell} {g : Fin C.N} : g ∈ L.yOf U ↔ g ∈ L.y ∧ L.cellOf g ∈ U := mem_filter

end Cut

namespace Tying

variable {C : Verity.Circuit} {θ : Finset (Fin C.N)} {L : Cut C} (T : Tying C θ L)

/-- **A tied cell's y gate is in its home block.** -/
theorem blocks_unit_emb {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) {g : Fin C.N}
    (hg : g ∈ L.yOf T.cells) : (T.blocks Q home).unit (T.emb g) = some (home (L.cellOf g)) := by
  have hs : ∃ g', T.emb g' = T.emb g := ⟨g, rfl⟩
  have hc : hs.choose = g := T.emb_inj hs.choose_spec
  unfold Tying.blocks
  simp only []
  rw [dite_eq_left hs, hc, ite_eq_left hg]

/-- **The placement is from below**, at any blocks `Q`, homes `home` and untied cells `U`: a placed cell's gates are
its 32 bits in its home block, an untied cell's are untied gates, and distinct cells sit at disjoint gates. -/
theorem fromBelow {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) (U : Finset Cell) :
    T.FromBelow Q home U := by
  refine ⟨fun c u O h => ?_, fun c hcU u O h => ?_, fun c c' u u' O O' hne h h' => ?_⟩
  · unfold Tying.place at h
    by_cases hc : c ∈ T.cells
    · rw [ite_eq_left hc] at h
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      refine ⟨fun x hx => ?_, ?_⟩
      · obtain ⟨g, hg, rfl⟩ := mem_map.1 hx
        obtain ⟨hgy, hgc⟩ := L.mem_yAt.1 hg
        have hgo : g ∈ L.yOf T.cells := L.mem_yOf.2 ⟨hgy, by rw [hgc]; exact hc⟩
        rw [Partition.mem_gates]
        show (T.blocks Q home).unit (T.emb g) = some (home c)
        rw [T.blocks_unit_emb Q home hgo, hgc]
      · show ∑ x ∈ (L.yAt c).map ⟨T.emb, T.emb_inj⟩, T.tied.D.w x = 32
        rw [sum_map]
        simp only [Function.Embedding.coeFn_mk, T.emb_w]
        exact L.width_word c (T.cells_sub hc)
    · rw [ite_eq_right hc] at h
      cases h
  · unfold Tying.place at h
    by_cases hc : c ∈ T.cells
    · rw [ite_eq_left hc] at h
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      intro x hx
      obtain ⟨g, hg, rfl⟩ := mem_map.1 hx
      obtain ⟨hgy, hgc⟩ := L.mem_yAt.1 hg
      unfold Tying.untiedGates
      exact mem_map_of_mem _ (L.mem_yOf.2 ⟨hgy, by rw [hgc]; exact hcU⟩)
    · rw [ite_eq_right hc] at h
      cases h
  · unfold Tying.place at h h'
    by_cases hc : c ∈ T.cells
    · by_cases hc' : c' ∈ T.cells
      · rw [ite_eq_left hc] at h
        rw [ite_eq_left hc'] at h'
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h')
        rw [disjoint_map]
        exact disjoint_left.2 fun g hg hg' => hne ((L.mem_yAt.1 hg).2.symm.trans (L.mem_yAt.1 hg').2)
      · rw [ite_eq_right hc'] at h'
        cases h'
    · rw [ite_eq_right hc] at h
      cases h

/-- **A served gate but a tied y keeps its registered block.** -/
theorem blocks_unit_served {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) {g : Fin C.N}
    (hg : g ∉ L.yOf T.cells) : (T.blocks Q home).unit (T.emb g) = Q.unit g := by
  have hs : ∃ g', T.emb g' = T.emb g := ⟨g, rfl⟩
  have hc : hs.choose = g := T.emb_inj hs.choose_spec
  unfold Tying.blocks
  simp only []
  rw [dite_eq_left hs, hc, ite_eq_right hg]

/-- **A computed gate the tied circuit adds is in its cell's home block.** -/
theorem blocks_unit_added {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) {h : Fin T.tied.D.N}
    (hna : ∀ g, T.emb g ≠ h) (hi : T.tied.D.isInput h = false) :
    (T.blocks Q home).unit h = some (home (T.owner h)) := by
  have hs : ¬ ∃ g, T.emb g = h := fun ⟨g, e⟩ => hna g e
  have hn : ¬ T.tied.D.isInput h = true := by simp [hi]
  unfold Tying.blocks
  simp only []
  rw [dite_eq_right hs, ite_eq_right hn]

private theorem width_map (S : Finset (Fin C.N)) :
    T.tied.D.width (S.map ⟨T.emb, T.emb_inj⟩) = C.width S := by
  show ∑ x ∈ S.map ⟨T.emb, T.emb_inj⟩, T.tied.D.w x = ∑ g ∈ S, C.w g
  rw [sum_map]
  simp only [Function.Embedding.coeFn_mk, T.emb_w]

/-- **A block of the tied circuit reads no fresh input its registered block doesn't**, where the homes hold the A
rows: what it reads outside itself and `θ` is the registered block's fresh input, embedded. The tie reads its A row in
its own block and the weights, registered, and a y crosses to a block reading it as the registered file's block read
it, an input outside `θ`. -/
theorem into_blocks_sub {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m) (hH : T.Homes Q home)
    (u : Fin m) :
    T.tied.D.into ((T.blocks Q home).gates u) \ T.tied.θ ⊆
      (C.into (Q.gates u) \ θ).map ⟨T.emb, T.emb_inj⟩ := by
  intro a ha
  obtain ⟨haI, haθ⟩ := mem_sdiff.1 ha
  obtain ⟨haB, h, hhB, hah⟩ := Nci.Cut.SecurityProofs.mem_into.1 haI
  have hhu : (T.blocks Q home).unit h = some u := ((T.blocks Q home).mem_gates).1 hhB
  have haB' : (T.blocks Q home).unit a ≠ some u := fun e => haB (((T.blocks Q home).mem_gates).2 e)
  have hθa : ∀ g', T.emb g' = a → g' ∉ θ := fun g' hg' hθ => haθ (hg' ▸ (T.theta_emb g').2 hθ)
  have hrow : ∀ c, home c = u → c ∈ T.cells → (∃ g' ∈ T.row c, T.emb g' = a) → False := by
    rintro c hcu hc ⟨g', hg', rfl⟩
    rcases hH c hc g' hg' with hθ' | hq
    · exact hθa g' rfl hθ'
    · have hgi : C.isInput g' = false := by
        cases hi : C.isInput g'
        · rfl
        · rw [(Q.input_iff g').2 hi] at hq
          cases hq
      have hgy : g' ∉ L.yOf T.cells := fun hy => by
        rw [L.y_input g' (L.mem_yOf.1 hy).1] at hgi
        cases hgi
      exact haB' (by rw [T.blocks_unit_served Q home hgy, hq, hcu])
  have hadd : ∀ c, home c = u → (∀ g', T.emb g' ≠ a) →
      (T.tied.D.isInput a = false → T.owner a = c) → False := by
    intro c hcu hna hown
    cases hi : T.tied.D.isInput a
    · exact haB' (by rw [T.blocks_unit_added Q home hna hi, hown hi, hcu])
    · exact haθ (T.added_registered a hna hi)
  by_cases hs : ∃ g, T.emb g = h
  · obtain ⟨g, rfl⟩ := hs
    by_cases hg : g ∈ L.yOf T.cells
    · rw [T.blocks_unit_emb Q home hg] at hhu
      have hcu : home (L.cellOf g) = u := Option.some.inj hhu
      have hc : L.cellOf g ∈ T.cells := (L.mem_yOf.1 hg).2
      rcases T.reads_y g hg a hah with hr | ⟨hna, hown⟩
      · exact (hrow _ hcu hc hr).elim
      · exact (hadd _ hcu hna hown).elim
    · rw [T.blocks_unit_served Q home hg] at hhu
      have hgc : C.isInput g = false := by
        cases hi : C.isInput g
        · rfl
        · rw [(Q.input_iff g).2 hi] at hhu
          cases hhu
      rw [T.reads_emb g hgc] at hah
      obtain ⟨g', hg', hga⟩ := mem_map.1 hah
      have hga' : T.emb g' = a := hga
      subst hga'
      refine mem_map.2 ⟨g', mem_sdiff.2 ⟨Nci.Cut.SecurityProofs.mem_into.2 ⟨fun hg'Q => ?_, g,
        (Q.mem_gates).2 hhu, hg'⟩, hθa g' rfl⟩, rfl⟩
      have hq := (Q.mem_gates).1 hg'Q
      have hgy : g' ∉ L.yOf T.cells := fun hy => by
        rw [(Q.input_iff g').2 (L.y_input g' (L.mem_yOf.1 hy).1)] at hq
        cases hq
      exact haB' (by rw [T.blocks_unit_served Q home hgy, hq])
  · have hna : ∀ g, T.emb g ≠ h := fun g e => hs ⟨g, e⟩
    have hhi : T.tied.D.isInput h = false := Nci.Cut.SecurityProofs.reader_computed hah
    rw [T.blocks_unit_added Q home hna hhi] at hhu
    have hcu : home (T.owner h) = u := Option.some.inj hhu
    rcases T.reads_tie h hna hhi a hah with hr | ⟨hna', hown⟩
    · exact (hrow _ hcu (T.owner_cell h hna hhi) hr).elim
    · exact (hadd _ hcu hna' hown).elim

/-- **The honest prover passes the policy on the tied circuit**: where the registered file's circuit divides into
blocks `Q` of at most `Γ` rows, each reading at most `α` fresh bits a row, and the homes hold the tied cells' A rows,
the tied circuit's blocks comply at the same `Γ` and `α` whatever cells are untied. -/
theorem compliesBut_of_file {Γ : ℕ} {α : ℝ} {m : ℕ} (Q : Verity.Partition C m) (home : Cell → Fin m)
    (hH : T.Homes Q home)
    (hQ : ∀ u, C.size (Q.gates u) ≤ Γ ∧ (C.width (C.into (Q.gates u) \ θ) : ℝ) ≤ α * C.size (Q.gates u))
    (U : Finset Cell) : T.CompliesBut Γ α U := by
  refine ⟨m, Q, home, hH, fun u => ⟨(hQ u).1, ?_⟩⟩
  have hsub : T.tied.D.into ((T.blocks Q home).gates u \ T.untiedGates U) \ T.tied.θ ⊆
      (T.tied.D.into ((T.blocks Q home).gates u) \ T.tied.θ) ∪ (T.untiedGates U ∩ (T.blocks Q home).gates u) := by
    intro a ha
    obtain ⟨haI, haθ⟩ := mem_sdiff.1 ha
    obtain ⟨haB, h, hhB, hah⟩ := Nci.Cut.SecurityProofs.mem_into.1 haI
    by_cases hB : a ∈ (T.blocks Q home).gates u
    · have hT : a ∈ T.untiedGates U := by
        by_contra hT
        exact haB (mem_sdiff.2 ⟨hB, hT⟩)
      exact mem_union_right _ (mem_inter.2 ⟨hT, hB⟩)
    · exact mem_union_left _ (mem_sdiff.2 ⟨Nci.Cut.SecurityProofs.mem_into.2
        ⟨hB, h, (mem_sdiff.1 hhB).1, hah⟩, haθ⟩)
  have hle : Nci.novelReads T.tied ((T.blocks Q home).gates u \ T.untiedGates U) ≤
      C.width (C.into (Q.gates u) \ θ) + T.tied.D.width (T.untiedGates U ∩ (T.blocks Q home).gates u) := by
    show T.tied.D.width _ ≤ _
    calc T.tied.D.width (T.tied.D.into ((T.blocks Q home).gates u \ T.untiedGates U) \ T.tied.θ)
        ≤ T.tied.D.width ((T.tied.D.into ((T.blocks Q home).gates u) \ T.tied.θ) ∪
            (T.untiedGates U ∩ (T.blocks Q home).gates u)) := by
          unfold Verity.Circuit.width
          exact sum_le_sum_of_subset hsub
      _ ≤ T.tied.D.width (T.tied.D.into ((T.blocks Q home).gates u) \ T.tied.θ) +
            T.tied.D.width (T.untiedGates U ∩ (T.blocks Q home).gates u) := by
          unfold Verity.Circuit.width
          exact le_of_le_of_eq (Nat.le_add_right _ _) sum_union_inter
      _ ≤ T.tied.D.width ((C.into (Q.gates u) \ θ).map ⟨T.emb, T.emb_inj⟩) +
            T.tied.D.width (T.untiedGates U ∩ (T.blocks Q home).gates u) :=
          Nat.add_le_add_right (by
            unfold Verity.Circuit.width
            exact sum_le_sum_of_subset (T.into_blocks_sub Q home hH u)) _
      _ = C.width (C.into (Q.gates u) \ θ) + T.tied.D.width (T.untiedGates U ∩ (T.blocks Q home).gates u) := by
          rw [T.width_map]
  have hle' : (Nci.novelReads T.tied ((T.blocks Q home).gates u \ T.untiedGates U) : ℝ) ≤
      C.width (C.into (Q.gates u) \ θ) + T.tied.D.width (T.untiedGates U ∩ (T.blocks Q home).gates u) := by
    exact_mod_cast hle
  linarith [(hQ u).2]

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
    constructor
    · intro h
      obtain ⟨hin, hθa⟩ := mem_sdiff.1 h
      obtain ⟨ha, g, hgu, hag⟩ := Nci.Cut.SecurityProofs.mem_into.1 hin
      have h' : a ∈ (if g ∈ T then ∅ else d.D.reads g) := hag
      rw [ite_eq_right (mem_sdiff.1 hgu).2] at h'
      exact mem_sdiff.2 ⟨Nci.Cut.SecurityProofs.mem_into.2 ⟨ha, g, hgu, h'⟩, (hθ a).1 hθa⟩
    · intro h
      obtain ⟨hin, hθa⟩ := mem_sdiff.1 h
      obtain ⟨ha, g, hgu, hag⟩ := Nci.Cut.SecurityProofs.mem_into.1 hin
      refine mem_sdiff.2 ⟨Nci.Cut.SecurityProofs.mem_into.2 ⟨ha, g, hgu, ?_⟩, (hθ a).2 hθa⟩
      show a ∈ (if g ∈ T then ∅ else d.D.reads g)
      rw [ite_eq_right (mem_sdiff.1 hgu).2]
      exact hag
  show (d.D.asInputs T).width ((d.D.asInputs T).into ((P.asInputs T).gates u) \ (d.θ \ T)) =
    d.D.width (d.D.into (P.gates u \ T) \ d.θ)
  rw [hg, hset]
  rfl

/-- **One window's blocks outside its `G`, charged whole**: `w + 1` times their charge is at most `c` times the
window's untied words, when no block is charged more than `c`. -/
theorem Charges.outside_le (W : Charges) (w c : ℕ) (hc : ∀ u, W.charge u ≤ c) :
    (w + 1) * W.outside w ≤ c * W.untied.card := by
  have h1 : W.outside w ≤ (W.heavy w).card * c := by
    unfold Charges.outside
    simpa using Finset.sum_le_card_nsmul (W.heavy w) W.charge c fun u _ => hc u
  have h2 : (W.heavy w).card * (w + 1) ≤ ∑ u ∈ W.heavy w, (W.untied.filter fun x => W.home x = u).card := by
    have := Finset.card_nsmul_le_sum (W.heavy w) (fun u => (W.untied.filter fun x => W.home x = u).card) (w + 1)
      fun u hu => by
        unfold Charges.heavy at hu
        exact (mem_filter.1 hu).2
    simpa using this
  have h3 : ∑ u ∈ W.heavy w, (W.untied.filter fun x => W.home x = u).card ≤ W.untied.card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := W.home) (t := univ) fun _ _ => mem_univ _]
    exact Finset.sum_le_sum_of_subset (subset_univ _)
  calc (w + 1) * W.outside w ≤ (w + 1) * ((W.heavy w).card * c) := Nat.mul_le_mul_left _ h1
    _ = c * ((W.heavy w).card * (w + 1)) := by ring
    _ ≤ c * W.untied.card := Nat.mul_le_mul_left _ (h2.trans h3)

/-- **`PeriodOutsideG` holds**: the windows' bounds (`Charges.outside_le`), summed over the period. -/
theorem periodOutsideG : PeriodOutsideG := by
  intro n W w c hc
  unfold periodOutside
  rw [Finset.mul_sum, Finset.mul_sum]
  exact Finset.sum_le_sum fun k _ => Charges.outside_le (W k) w c (hc k)

end Verity.ValueTie
