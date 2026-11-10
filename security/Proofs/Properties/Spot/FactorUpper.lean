module

public import Proofs.Properties.Spot.FactorUpper.Capped

/-!
# Rankin's factor is below `factorUpper`

`Properties.Spot.FactorUpperSound` (the sets' form, `FactorUpper/Sets.lean`), `FactorUpperPatternsSound`
(the patterns' form over requests, `FactorUpper/Patterns.lean`) and `FactorUpperCappedSound` (each set's bits capped,
`FactorUpper/Capped.lean`), on the arithmetic of `FactorUpper/Bounds.lean`.

Each honest witness (`_honest`) concludes its guarantee's premises at a replay block of two one-bit proof blocks
(`pairStmt`, `pairDesign`), where the bound returns, and the kernel's evaluation returns the bound at a deployed-shaped
replay block too. Each conclusion fails for a bound that returns `−1` whatever it reads: at a replay block with no proof
blocks each factor is `1` (`_nontrivial`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs

open Claims.Sampling
open _root_.Spot.FactorUpper

/-- **`lnFactorUpper` is above Rankin's factor.** -/
theorem factorUpperSound : FactorUpperSound := FactorUpperProof.factorUpper_sound

/-- **`lnFactorUpperPatterns` is above Rankin's patterns' factor.** -/
theorem factorUpperPatternsSound : FactorUpperPatternsSound := FactorUpperProof.factorUpperPatterns_sound

/-- **`lnFactorUpperCapped` is above Rankin's capped factor.** -/
theorem factorUpperCappedSound : FactorUpperCappedSound := FactorUpperProof.factorUpperCapped_sound

section Witnesses

open FactorUpperWitness

namespace FactorUpperProof

theorem emptyPart_covers (ρ : ℕ) : Covers emptyStmt emptyPart emptyDesign ρ [] :=
  ⟨rfl, fun t c h => by simp at h⟩

theorem emptyPart_rankinFactor (q θ : ℝ) (ρ : ℕ) : rankinFactor emptyStmt emptyPart emptyDesign q θ ρ = 1 := by
  simp [rankinFactor, blocksIn, emptyPart, Partition.strata, Design.Two.passes, Design.Two.esc, bitsOf]

theorem iSup_one_of_mem {S : Finset (Finset ℕ)} {x : Finset ℕ} (hx : x ∈ S) : (⨆ _U : S, (1 : ℝ)) = 1 :=
  haveI : Nonempty S := ⟨⟨x, hx⟩⟩
  ciSup_const

theorem emptyPart_rankinFactorPatterns (q θ : ℝ) (ρ : ℕ) :
    rankinFactorPatterns emptyStmt emptyPart emptyDesign (fun _ => ()) q θ ρ = 1 := by
  have hB : blocksIn emptyPart emptyDesign.part ρ = ∅ := by simp [blocksIn, emptyPart, Partition.strata]
  have hE : ∀ r, blocksTo emptyStmt.prog emptyPart (fun _ => ()) ∅ r = ∅ := fun r =>
    Finset.subset_empty.1 (blocksTo_sub_self ∅ r)
  have hP : ∀ fp, emptyDesign.passes emptyPart q fp ρ = 1 := fun fp => by
    simp [Design.Two.passes, Design.Two.esc, emptyPart, Partition.strata]
  unfold rankinFactorPatterns
  rw [hB]
  simp only [Finset.powerset_empty, Finset.image_singleton, Finset.sum_singleton, hE, hP, Real.one_rpow]
  rw [iSup_one_of_mem (x := ∅) ?_]
  · simp [lightOf, lightBits]
  · simp [hE]

theorem emptyPart_rankinFactorCapped (S : ℕ) (q θ : ℝ) (ρ : ℕ) :
    rankinFactorCapped emptyStmt emptyPart emptyDesign S q θ ρ = 1 := by
  simp [rankinFactorCapped, blocksIn, emptyPart, Partition.strata, Design.Two.passes, Design.Two.esc, bitsOf]

theorem exp_neg_one_lt_one : Real.exp (-1) < 1 :=
  calc Real.exp (-1) < Real.exp 0 := Real.exp_lt_exp.2 (by norm_num)
    _ = 1 := Real.exp_zero

theorem pair_cell : cell pairPart pairDesign.part 0 0 = {0, 1} := by decide

theorem pair_cell_ne {t : ℕ} (ht : t ≠ 0) : cell pairPart pairDesign.part 0 t = ∅ := by
  rw [Finset.eq_empty_iff_forall_notMem]
  intro v hv
  simp only [cell, Finset.mem_filter, Finset.mem_range] at hv
  obtain ⟨hv2, -, hs⟩ := hv
  have h0 : pairPart.stratumOf v = 0 := by
    have hl : pairPart.blocks.length = 2 := by decide
    have : v < 2 := hl ▸ hv2
    interval_cases v <;> decide
  exact ht (hs ▸ h0)

/-- `pairDesign`'s cell holds two proof blocks of one bit each, and none of another width. -/
theorem pair_count (b : ℕ) :
    ((cell pairPart pairDesign.part 0 0).filter fun v => committed pairStmt.prog (pairPart.blocks.getD v []) = b).card ≤
      countIn [(2, 1)] b := by
  rw [pair_cell]
  by_cases hb : b = 1
  · subst hb; exact (Finset.card_filter_le _ _).trans (by decide)
  · rw [Finset.filter_false_of_mem fun v hv => ?_]
    · simp
    · have h1 : committed pairStmt.prog (pairPart.blocks.getD v []) = 1 := by
        simp only [Finset.mem_insert, Finset.mem_singleton] at hv
        rcases hv with rfl | rfl <;> decide
      omega

theorem pair_covers : Covers pairStmt pairPart pairDesign 0 pairCells := by
  refine ⟨by decide, fun t c h => ?_⟩
  match t, h with
  | 0, h =>
    simp only [pairCells, List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    exact ⟨by decide, pair_count⟩
  | _ + 1, h => simp [pairCells] at h

theorem pair_readsCover : ReadsCover pairStmt pairPart pairDesign (fun _ => ()) 0 [[(2, 1)]] () := by
  intro t b
  refine (Finset.card_le_card (Finset.filter_subset_filter _ (blocksTo_sub_self _ _))).trans ?_
  rcases Nat.eq_zero_or_pos t with rfl | ht
  · exact pair_count b
  · rw [pair_cell_ne ht.ne']; simp

end FactorUpperProof

open FactorUpperProof

/-- **`FactorUpperSound`'s honest witness**: `pairCells` covers `pairDesign`'s replay block, and the bound returns there
and at a deployed-shaped replay block. -/
theorem factorUpperSound_honest :
    Covers pairStmt pairPart pairDesign 0 pairCells ∧ (lnFactorUpper pairCells honestParams).isSome = true ∧
      (lnFactorUpper honestCells honestParams).isSome = true :=
  ⟨pair_covers, by decide +kernel, by decide +kernel⟩

/-- **`FactorUpperSound`'s conclusion fails** for a bound of `−1`: Rankin's factor of a replay block with no proof
blocks, which `[]` covers (`emptyPart_covers`), is `1`. -/
theorem factorUpperSound_nontrivial : ¬ rankinFactor emptyStmt emptyPart emptyDesign (1 / 2) 1 0 ≤ Real.exp (-1) := by
  rw [emptyPart_rankinFactor, not_le]
  exact exp_neg_one_lt_one

/-- **`FactorUpperPatternsSound`'s honest witness**: `pairCells` covers `pairDesign`'s replay block, the one request
reading both outputs reads at most `[[(2, 1)]]` of it and serves 2 bits, and the bound returns there; and it returns at
the deployed-shaped replay block with each of at most 4 requests reading 8 of its proof blocks and serving 64 bits. -/
theorem factorUpperPatternsSound_honest :
    Covers pairStmt pairPart pairDesign 0 pairCells ∧
      ReadsCover pairStmt pairPart pairDesign (fun _ => ()) 0 [[(2, 1)]] () ∧
      servedBits (fun _ : Fin pairStmt.prog.outs.length => ()) () = 2 ∧
      (lnFactorUpperPatterns pairCells [[(2, 1)]] 1 2 honestParams).isSome = true ∧
      (lnFactorUpperPatterns honestCells [[(8, 16)]] 4 64 honestParams).isSome = true :=
  ⟨pair_covers, pair_readsCover, by decide, by decide +kernel, by decide +kernel⟩

/-- **`FactorUpperPatternsSound`'s conclusion fails** for a bound of `−1`: the patterns' factor of a replay block with
no proof blocks is `1`. -/
theorem factorUpperPatternsSound_nontrivial :
    ¬ rankinFactorPatterns emptyStmt emptyPart emptyDesign (fun _ => ()) (1 / 2) 1 0 ≤ Real.exp (-1) := by
  rw [emptyPart_rankinFactorPatterns, not_le]
  exact exp_neg_one_lt_one

/-- **`FactorUpperCappedSound`'s honest witness**: `pairCells` covers `pairDesign`'s replay block, and the bound returns
there with each set's bits capped at 1, below its 2, and at the deployed-shaped replay block capped at 64, below its
1,024. -/
theorem factorUpperCappedSound_honest :
    Covers pairStmt pairPart pairDesign 0 pairCells ∧ (lnFactorUpperCapped pairCells 1 honestParams).isSome = true ∧
      (lnFactorUpperCapped honestCells 64 honestParams).isSome = true :=
  ⟨pair_covers, by decide +kernel, by decide +kernel⟩

/-- **`FactorUpperCappedSound`'s conclusion fails** for a bound of `−1`: the capped factor of a replay block with no
proof blocks is `1`. -/
theorem factorUpperCappedSound_nontrivial :
    ¬ rankinFactorCapped emptyStmt emptyPart emptyDesign 64 (1 / 2) 1 0 ≤ Real.exp (-1) := by
  rw [emptyPart_rankinFactorCapped, not_le]
  exact exp_neg_one_lt_one

end Witnesses

end Properties.Spot.SecurityProofs
