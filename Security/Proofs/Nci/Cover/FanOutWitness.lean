import Proofs.Nci.Cover.FanOutWalk
import Proofs.Nci.Cover.Witness

/-!
# The fan-out bounds: their hypotheses can hold together

`fanOut_witness`: one small circuit with a counted product and a keyed token structure, a root with two legs and two
lookups, satisfying `UpdateHardness`, `ValueReadHardness` at `f = 2`, `ColumnsIncompressible` with
`k < bcol · V · (E + 1)` at `E = 1 < d`, and covered by inference blocks, of which one holds a lookup, one the root, and
one spends a value.

The circuit (`FanOutWitness.fC`, 15 gates, `θ = {0}`) is the proof red team's, widened so that every value is two bits
and a block can afford one lookup. Gate 0 is the registered token, gates 1–3 and 4–6 the two lookups' columns (`V = 3`)
and gate 7 the product's updated parameter. The lookups are gates 8–9, each reading the token and column 1–3, and 10–11,
reading the token and column 4–6. The root is gates 12–13, each reading both lookups, its two legs (`d = 2`). The
counted product is gate 14, reading the root, its activation, and gate 7.

A value's gates read the same gates, so a set of gates that contains a value and is handed one of its gates is handed
both: a spent value costs `f = 2` bits. Holding a lookup costs its column's 3 bits, so `ColumnsIncompressible` holds at
`bcol = 1`, and `k = 5 < bcol · V · (E + 1) = 6` at `E = 1`. Four blocks of `s = 6` gates cover the circuit: one for each
lookup, handed its column, one for the root, handed both lookups (4 bits), and one for the product, handed the root,
gate 7 and a column bit.
-/

namespace Nci.Cover.SecurityProofs

namespace FanOutWitness

local instance decInferenceBlock (C : Circuit) (s k : ℕ) (I : Finset (Fin C.N)) :
    Decidable (C.InferenceBlock s k I) :=
  inferInstanceAs (Decidable (I.card = s ∧ (C.inputs I \ C.θ).card ≤ k))

local instance decCovers (C : Circuit) (𝓘 : Finset (Finset (Fin C.N))) : Decidable (C.Covers 𝓘) :=
  inferInstanceAs (Decidable (∀ g, (C.pred g).Nonempty → ∃ I ∈ 𝓘, g ∈ C.computed I))

theorem reachesIn_refl {C : Circuit} {X : Finset (Fin C.N)} {g : Fin C.N} (hg : g ∈ X) : C.ReachesIn X g g :=
  ⟨[g], ⟨rfl, rfl, .singleton g⟩, fun _ hv => List.mem_singleton.1 hv ▸ hg⟩

/-- If the gates of `G` all read the same gates, a set that contains `G` and is handed one of them is handed all. -/
theorem subset_inputs_of_pred_eq {C : Circuit} {S G : Finset (Fin C.N)}
    (hG : ∀ x ∈ G, ∀ y ∈ G, C.pred x = C.pred y) (hGS : G ⊆ S) (hne : (G ∩ C.inputs S).Nonempty) :
    G ⊆ C.inputs S := by
  obtain ⟨y, hy⟩ := hne
  obtain ⟨hyG, hyS⟩ := Finset.mem_inter.1 hy
  intro x hx
  rw [mem_inputs] at hyS ⊢
  rw [hG x hx y hyG]
  exact ⟨hGS hx, hyS.2⟩

/-- The witness's wiring: the token 0, the columns 1–3 and 4–6 and the parameter 7 are input gates; 8–9 and 10–11 are
the lookups, 12–13 the root and 14 the counted product. -/
def fPred : Fin 15 → Finset (Fin 15) :=
  ![∅, ∅, ∅, ∅, ∅, ∅, ∅, ∅, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 4, 5, 6}, {0, 4, 5, 6}, {8, 9, 10, 11}, {8, 9, 10, 11},
    {7, 12, 13}]

/-- The witness circuit, with the token registered. -/
abbrev fC : Circuit where
  N := 15
  pred := fPred
  pred_lt := by decide
  θ := {0}
  θ_input := by decide

/-- The counted product `(1, 1)` is gate 14. -/
def fm : ℕ × ℕ → Finset (Fin 15) := fun p => if p = (1, 1) then {14} else ∅

/-- The witness's one product, counted. -/
abbrev fM : MulBlocks fC ℕ ℕ where
  Pall := {(1, 1)}
  P := {(1, 1)}
  P_subset := subset_rfl
  m := fm
  m_nonempty := by decide
  m_computed := by decide
  m_disjoint := by
    rw [Finset.coe_singleton]
    exact Set.pairwiseDisjoint_singleton _ _
  o := fun _ => 14
  m_outputs := by decide

/-- The values: the lookups 0 and 1 and the root 2. -/
def fgates : Fin 3 → Finset (Fin 15) := ![{8, 9}, {10, 11}, {12, 13}]

/-- The lookups' columns, as updated parameters. -/
def fcol : Fin 3 → Finset ℕ := ![{11, 12, 13}, {21, 22, 23}, ∅]

/-- The gates of the lookups' columns. -/
def fcolGates : Fin 3 → Finset (Fin 15) := ![{1, 2, 3}, {4, 5, 6}, ∅]

theorem fgates_disjoint : ∀ u ∈ (Finset.univ : Finset (Fin 3)), ∀ v ∈ (Finset.univ : Finset (Fin 3)), u ≠ v →
    Disjoint (fgates u) (fgates v) := by
  decide

theorem fgates_legs : ∀ g ∈ fgates 2, ∀ q ∈ ({0, 1} : Finset (Fin 3)), ∃ h ∈ fC.exits (fgates q), h ∈ fC.pred g := by
  decide

/-- The keyed token structure: one token, no keys, the root 2 with legs 0 and 1, which are the lookups. -/
def fK : Keyed fM Unit (Fin 3) 2 1 1 3 where
  vals := Finset.univ
  gates := fgates
  tok := fun _ => ()
  gates_disjoint := fun u hu v hv => fgates_disjoint u hu v hv
  gates_θ := by decide
  act := fun _ => 2
  act_mem := fun _ _ => Finset.mem_univ _
  act_link := by
    intro p hp
    obtain rfl : p = (1, 1) := Finset.mem_singleton.1 hp
    exact ⟨12, by decide, 14, by decide, by decide, reachesIn_refl (by decide)⟩
  mult_le := fun _ _ X hX _ => (Finset.card_le_card hX).trans (by decide)
  roots := {2}
  lookups := {0, 1}
  key := fun _ => none
  keyed := by
    intro v _ hv
    exact absurd (show v ∈ ({2} : Finset (Fin 3)) ∪ {0, 1} by revert v; decide) hv
  legs := fun _ => {0, 1}
  legs_mem := by decide
  legs_card := by decide
  legs_link := by
    intro r hr g hg q hq
    obtain rfl : r = 2 := Finset.mem_singleton.1 hr
    have hgX := mem_of_mem_exits hg
    obtain ⟨h, hh, hhg⟩ := fgates_legs g hgX q hq
    exact ⟨h, hh, g, hgX, hhg, reachesIn_refl hgX⟩
  legs_disjoint := by
    intro r _ q _ q' _ hne u hu hu'
    exact hne ((Follows.eq_of_key_none (fun _ => rfl) hu).symm.trans (Follows.eq_of_key_none (fun _ => rfl) hu'))
  col := fcol
  col_card := by decide
  col_disjoint := by decide
  reach := fun _ X hX _ => (Finset.card_le_card hX).trans (by decide)
  reach_lookups := by
    intro p _ u hu hf
    obtain rfl : u = 2 := Follows.eq_of_key_none (fun _ => rfl) hf
    exact absurd hu (by decide)

theorem fM_update (s : ℕ) : Assumptions.UpdateHardness fM s 1 :=
  updateHardness_of_conditional fM (fun _ => ∅)
    (Witness.conditionalUpdateHardness_of_feeds fM (fun _ => ∅) (fun _ => {7}) s (by decide) (by decide)
      (by decide) fun _ _ => by simp)

/-- A spent value's gates all lie in a gate the block computes: the product, for the root, or a gate of the held root,
for a leg. -/
theorem fK_spent_pred {S : Finset (Fin 15)} {u : Fin 3} (hu : fK.Spent S u) :
    ∃ g ∈ fC.computed S, fgates u ⊆ fC.pred g := by
  obtain ⟨-, v, hv, hf⟩ := hu
  obtain rfl : u = v := Follows.eq_of_key_none (fun _ => rfl) hf
  rcases hv with ⟨p, hp, hpu⟩ | ⟨r, hr, hur⟩
  · obtain rfl : u = 2 := hpu.symm
    obtain ⟨hpP, hpS⟩ := Finset.mem_filter.1 hp
    obtain rfl : p = (1, 1) := Finset.mem_singleton.1 hpP
    exact ⟨14, hpS (by decide), by decide⟩
  · obtain ⟨hr2, hrh⟩ := Finset.mem_inter.1 hr
    obtain rfl : r = 2 := Finset.mem_singleton.1 hr2
    obtain ⟨g, hg⟩ := (mem_heldSet.1 hrh).2.2
    obtain ⟨hge, hgc⟩ := Finset.mem_inter.1 hg
    have h : ∀ u ∈ ({0, 1} : Finset (Fin 3)), ∀ g ∈ fgates 2, fgates u ⊆ fC.pred g := by decide
    exact ⟨g, hgc, h u hur g (mem_of_mem_exits hge)⟩

theorem fK_valueRead (s : ℕ) : Assumptions.ValueReadHardness fK s 2 1 := by
  intro S _ X hX
  have hin : ∀ u ∈ X, fgates u ⊆ fC.inputs S := by
    intro u hu
    obtain ⟨g, hg, hug⟩ := fK_spent_pred (hX u hu)
    have hpred : ∀ u : Fin 3, ∀ x ∈ fgates u, ∀ y ∈ fgates u, fC.pred x = fC.pred y := by decide
    exact subset_inputs_of_pred_eq (hpred u) (hug.trans (pred_subset_of_mem_computed hg))
      (mem_readSet.1 (hX u hu).1).2
  have hθ : ∀ u : Fin 3, Disjoint (fgates u) fC.θ := by decide
  have hT : X.biUnion fgates ⊆ fC.inputs S \ fC.θ := by
    intro x hx
    obtain ⟨u, hu, hxu⟩ := Finset.mem_biUnion.1 hx
    exact Finset.mem_sdiff.2 ⟨hin u hu hxu, Finset.disjoint_left.1 (hθ u) hxu⟩
  have hTcard : (X.biUnion fgates).card = 2 * X.card := by
    rw [Finset.card_biUnion fun u _ v _ huv => fgates_disjoint u (Finset.mem_univ _) v (Finset.mem_univ _) huv]
    have h2 : ∀ u : Fin 3, (fgates u).card = 2 := by decide
    simp only [h2, Finset.sum_const, smul_eq_mul, mul_comm]
  have hnat : 2 * X.card + ((fM.whole S).image Prod.snd).card ≤ (fC.inputs S \ fC.θ).card := by
    have hWs : fM.whole S ⊆ {(1, 1)} := Finset.filter_subset _ _
    rcases Finset.subset_singleton_iff.1 hWs with hW | hW
    · rw [hW, Finset.image_empty, Finset.card_empty, add_zero, ← hTcard]
      exact Finset.card_le_card hT
    · have h14 : (14 : Fin 15) ∈ fC.computed S :=
        (Finset.mem_filter.1 (hW ▸ Finset.mem_singleton_self _ : ((1, 1) : ℕ × ℕ) ∈ fM.whole S)).2 (by decide)
      have h7 : (7 : Fin 15) ∈ fC.inputs S \ fC.θ :=
        Finset.mem_sdiff.2 ⟨mem_inputs.2 ⟨pred_subset_of_mem_computed h14 (by decide), Or.inl (by decide)⟩, by decide⟩
      have h7T : (7 : Fin 15) ∉ X.biUnion fgates := by
        intro h
        obtain ⟨u, -, hu⟩ := Finset.mem_biUnion.1 h
        revert hu
        revert u
        decide
      rw [hW, Finset.image_singleton, Finset.card_singleton, ← hTcard, ← Finset.card_insert_of_notMem h7T]
      exact Finset.card_le_card (Finset.insert_subset h7 hT)
  rw [one_mul]
  exact_mod_cast hnat

theorem fK_columns (s : ℕ) : Assumptions.ColumnsIncompressible fK s 1 := by
  intro S _ X hX _
  have hcolpred : ∀ v ∈ ({0, 1} : Finset (Fin 3)), ∀ g ∈ fgates v, ∀ x ∈ fcolGates v, x ∈ fC.pred g := by decide
  have hcol0 : ∀ v ∈ ({0, 1} : Finset (Fin 3)), ∀ x ∈ fcolGates v, fC.pred x = ∅ ∧ x ∉ fC.θ := by decide
  have hcolcard : ∀ Y : Finset (Fin 3), Y ⊆ {0, 1} → (Y.biUnion fcol).card ≤ (Y.biUnion fcolGates).card := by decide
  have hG : X.biUnion fcolGates ⊆ fC.inputs S \ fC.θ := by
    intro x hx
    obtain ⟨v, hv, hxv⟩ := Finset.mem_biUnion.1 hx
    obtain ⟨hvl, hvh⟩ := Finset.mem_inter.1 (hX hv)
    obtain ⟨g, hg⟩ := (mem_heldSet.1 hvh).2.2
    obtain ⟨hge, hgc⟩ := Finset.mem_inter.1 hg
    have hxS := pred_subset_of_mem_computed hgc (hcolpred v hvl g (mem_of_mem_exits hge) x hxv)
    exact Finset.mem_sdiff.2 ⟨mem_inputs.2 ⟨hxS, Or.inl (hcol0 v hvl x hxv).1⟩, (hcol0 v hvl x hxv).2⟩
  have hnat := (hcolcard X fun v hv => (Finset.mem_inter.1 (hX hv)).1).trans (Finset.card_le_card hG)
  rw [one_mul]
  exact_mod_cast hnat

/-- The cover: a block for each lookup, one for the root and one for the product, each of six gates. -/
def fI : Finset (Finset (Fin 15)) :=
  {{0, 1, 2, 3, 8, 9}, {0, 4, 5, 6, 10, 11}, {8, 9, 10, 11, 12, 13}, {0, 1, 7, 12, 13, 14}}

theorem fI_blocks : ∀ I ∈ fI, fC.InferenceBlock 6 5 I := by
  intro I hI
  simp only [fI, Finset.mem_insert, Finset.mem_singleton] at hI
  rcases hI with rfl | rfl | rfl | rfl <;> decide

theorem fI_covers : fC.Covers fI := by decide

end FanOutWitness

open FanOutWitness in
/-- **Non-vacuity of the fan-out bounds**: some circuit with a counted product and a keyed token structure satisfies
`UpdateHardness`, `ValueReadHardness` at `f = 2` and `ColumnsIncompressible` with `k < bcol · V · (E + 1)` and
`1 ≤ E < d`, and has a cover by inference blocks, one of which holds a lookup, one a root, and one spends a value
(`FanOutWitness.fC`: `s = 6`, `k = 5`, `E = 1`, `d = 2`, `V = 3`, `b = bcol = 1`, four blocks). -/
theorem fanOut_witness :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (d aOut c V : ℕ) (K : Keyed M Unit (Fin 3) d aOut c V) (s k E f : ℕ)
      (b bcol : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
      M.P.Nonempty ∧ 0 < b ∧ 2 ≤ f ∧ 1 ≤ E ∧ Assumptions.UpdateHardness M s b ∧
      Assumptions.ValueReadHardness K s f b ∧ Assumptions.ColumnsIncompressible K s bcol ∧
      (k : ℝ) < bcol * V * (E + 1) ∧ E < d ∧ (∀ I ∈ 𝓘, C.InferenceBlock s k I) ∧ C.Covers 𝓘 ∧
      (∃ I ∈ 𝓘, (K.lookups ∩ K.heldSet I).Nonempty) ∧ (∃ I ∈ 𝓘, (K.roots ∩ K.heldSet I).Nonempty) ∧
      (∃ I ∈ 𝓘, ∃ u, K.Spent I u) :=
  ⟨fC, fM, 2, 1, 1, 3, fK, 6, 5, 1, 2, 1, 1, fI, ⟨(1, 1), by decide⟩, one_pos, le_rfl, le_rfl, fM_update 6,
    fK_valueRead 6, fK_columns 6, by norm_num, by decide, fI_blocks, fI_covers,
    ⟨{0, 1, 2, 3, 8, 9}, by decide, 0, by decide⟩, ⟨{8, 9, 10, 11, 12, 13}, by decide, 2, by decide⟩,
    ⟨{0, 1, 7, 12, 13, 14}, by decide, 2, by decide, 2, Or.inl ⟨(1, 1), by decide, rfl⟩, Follows.refl 2⟩⟩

end Nci.Cover.SecurityProofs
