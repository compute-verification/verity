import Mathlib
import Definitions.Accounting.Carrier
import Proofs.Accounting.Requirement

/-!
# A pass's credit against its tiles of the declared circuit

The piece of `Accounting.SecurityProofs.DeclaredCircuitRequirement` that reads a pass's credit pass by pass. `W`'s owned
blocks at the pass's step are planned, at distinct positions of the pass's store, so the pass's tiles of `W` are as many
as its owned blocks (`credit_le_tiles`). Its held credit is `W`'s credited live bits in those blocks, packed, so at most a
block's bits each; its moved credit is the bits it reads from appended blocks, those in owned blocks at most a block's
bits each and the rest in one block a group. So the credit is at most a tile's bits for each of its tiles of `W`, and
those are appended tiles (`tiles_subset_appended`).
-/

namespace Accounting.Requirement

variable {G : Type} [DecidableEq G]

/-- **A pass's credit is at most a tile's bits for each of its tiles of `W`**, at a step whose memory map and plan hold
`W`'s credited blocks, whose reads from incomplete blocks lie in one block a group, and whose plan's blocks are at
distinct positions of the pass's layout `lay`. -/
theorem credit_le_tiles {K : Type} [DecidableEq K] {N T M : ℕ} (W : Wiring G) (e : Execution G)
    (ℓ : Store.Placement G) (σ : Store) (plan : Finset ℕ) (grp : G → K) (t : ℕ) (addr : ℕ → Fin T × Fin M)
    (lay : Fin N ↪ Fin T × Fin M)
    (hS : StoreMatches W e ℓ σ 32768 t) (hP : PlanMatches W e ℓ plan 32768 t)
    (hO : OneBlockEach ℓ grp (reads W e t \ creditedReads W e ℓ t))
    (hinj : Set.InjOn addr ↑plan) (hrange : ∀ b ∈ plan, addr b ∈ Set.range lay) :
    max (bitsAt W ℓ (credited W e ℓ t)
          ((blocks ℓ (creditedReads W e ℓ t)).filter fun b => ((addr b).1 : ℕ) ≠ 0))
        (bits W ((reads W e t).filter fun v => ((addr (ℓ.block v)).1 : ℕ) ≠ 0) -
          ((reads W e t \ creditedReads W e ℓ t).image grp).card * 32768) ≤
      (Finset.univ.filter fun i : Fin N =>
        lay i ∈ ((blocks ℓ (creditedReads W e ℓ t)).filter fun b => ((addr b).1 : ℕ) ≠ 0).image addr).card *
        32768 := by
  set owned := (blocks ℓ (creditedReads W e ℓ t)).filter fun b => ((addr b).1 : ℕ) ≠ 0 with howned
  have hown_plan : owned ⊆ plan := fun b hb => hP.1 (Finset.mem_filter.mp hb).1
  have htiles : (Finset.univ.filter fun i : Fin N => lay i ∈ owned.image addr).card = owned.card := by
    have e1 : (Finset.univ.filter fun i : Fin N => lay i ∈ owned.image addr).map lay = owned.image addr := by
      ext z
      simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · rintro ⟨i, hi, rfl⟩
        exact hi
      · intro hz
        obtain ⟨b, hb, rfl⟩ := Finset.mem_image.mp hz
        obtain ⟨i, hi⟩ := hrange b (hown_plan hb)
        exact ⟨i, hi ▸ hz, hi⟩
    calc (Finset.univ.filter fun i : Fin N => lay i ∈ owned.image addr).card
        = ((Finset.univ.filter fun i : Fin N => lay i ∈ owned.image addr).map lay).card :=
          (Finset.card_map lay).symm
      _ = (owned.image addr).card := congrArg Finset.card e1
      _ = owned.card := Finset.card_image_of_injOn (hinj.mono (Finset.coe_subset.mpr hown_plan))
  rw [htiles]
  have hheld : bitsAt W ℓ (credited W e ℓ t) owned ≤ owned.card * 32768 := by
    have := bits_le_bitsAt_add W ℓ (S := (credited W e ℓ t).filter fun v => ℓ.block v ∈ owned) (R := owned) ∅
      (fun v hv => (Finset.mem_filter.mp hv).2)
      (packs_mono W ℓ hS.2.1 ((Finset.filter_subset _ _).trans (Finset.filter_subset _ _)))
    have h0 : bitsAt W ℓ ((credited W e ℓ t).filter fun v => ℓ.block v ∈ owned) ∅ = 0 := by
      simp [bitsAt, bits]
    rw [h0, zero_add, Finset.sdiff_empty] at this
    exact this
  have hmoved : bits W ((reads W e t).filter fun v => ((addr (ℓ.block v)).1 : ℕ) ≠ 0) ≤
      owned.card * 32768 + ((reads W e t \ creditedReads W e ℓ t).image grp).card * 32768 := by
    set Rd := (reads W e t).filter fun v => ((addr (ℓ.block v)).1 : ℕ) ≠ 0 with hRd
    have hsplit : bits W Rd = bits W (Rd.filter fun v => v ∈ creditedReads W e ℓ t) +
        bits W (Rd.filter fun v => v ∉ creditedReads W e ℓ t) :=
      (Finset.sum_filter_add_sum_filter_not Rd _ _).symm
    have h1 : bits W (Rd.filter fun v => v ∈ creditedReads W e ℓ t) ≤ owned.card * 32768 := by
      have := bits_le_bitsAt_add W ℓ (S := Rd.filter fun v => v ∈ creditedReads W e ℓ t) (R := owned) ∅
        (fun v hv => by
          have hv' := Finset.mem_filter.mp hv
          have hv'' := Finset.mem_filter.mp hv'.1
          exact Finset.mem_filter.mpr ⟨Finset.mem_image_of_mem _ hv'.2, hv''.2⟩)
        (packs_mono W ℓ hP.2 ((Finset.filter_subset _ _).trans (Finset.filter_subset _ _)))
      have h0 : bitsAt W ℓ (Rd.filter fun v => v ∈ creditedReads W e ℓ t) ∅ = 0 := by
        simp [bitsAt, bits]
      rw [h0, zero_add, Finset.sdiff_empty] at this
      exact this
    have h2 : bits W (Rd.filter fun v => v ∉ creditedReads W e ℓ t) ≤
        ((reads W e t \ creditedReads W e ℓ t).image grp).card * 32768 := by
      have hsub : Rd.filter (fun v => v ∉ creditedReads W e ℓ t) ⊆ reads W e t \ creditedReads W e ℓ t := by
        intro v hv
        have hv' := Finset.mem_filter.mp hv
        exact Finset.mem_sdiff.mpr ⟨(Finset.mem_filter.mp hv'.1).1, hv'.2⟩
      have h3 : bits W (Rd.filter fun v => v ∉ creditedReads W e ℓ t) ≤
          bits W (reads W e t \ creditedReads W e ℓ t) :=
        Finset.sum_le_sum_of_subset hsub
      exact h3.trans (bits_le_groups W ℓ Finset.sdiff_subset hP.2 grp hO)
    omega
  exact max_le hheld (by omega)

/-- **A pass's tiles of `W` are appended tiles**: the tiles at positions of `W`'s owned blocks, each at an append after
the registration. -/
theorem tiles_subset_appended {Pfx Y : Type} {N T M : ℕ} (layout : Pfx → Y → Fin N ↪ Fin T × Fin M) (c : Pfx) (y : Y)
    (addr : ℕ → Fin T × Fin M) (O : Finset ℕ) :
    (Finset.univ.filter fun i : Fin N => layout c y i ∈ (O.filter fun b => ((addr b).1 : ℕ) ≠ 0).image addr) ⊆
      RateSketch.appendedTiles layout c y := by
  intro i hi
  obtain ⟨b, hb, he⟩ := Finset.mem_image.mp (Finset.mem_filter.mp hi).2
  have h0 := (Finset.mem_filter.mp hb).2
  simp only [RateSketch.appendedTiles, Finset.mem_filter, Finset.mem_univ, true_and]
  rw [← he]
  exact h0

end Accounting.Requirement
