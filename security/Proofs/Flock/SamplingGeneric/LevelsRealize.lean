module

public import Proofs.Flock.SamplingGeneric.Levels
public import Proofs.Flock.Claims.Sampling.Escape
public import Proofs.Core.Game.Tape
public import Proofs.Core.LevelsTwo
public import Proofs.Flock.SamplingGeneric.LevelsTwoKS

/-!
# Levels realize the claims designs

* `OneStage.realized`, `Design.realized`: a well-formed rule is the miss of a sample of its class's window on coins of
  its own (`all`, `subset`, `stratified` and `work` as strata samples, `bernoulli` as one coin per slot), and a
  well-formed design the miss of its classes' samples drawn independently.
* `levelsRealizeOne`: **one level realizes every well-formed claims design** (`LevelsRealizeOne`): its blocks the proof
  blocks one by one, its design `Design.realized`'s sample.
* `levelsRealizeTwo`: **two levels realize every two-stage design in scope** (`LevelsRealizeTwo`): the first level's
  blocks the nonempty replay blocks, its design the replay design's sample (`Design.realized`) read as blocks; the
  second's the proof blocks one by one, its design in each replay block the uniform subsets of its cells. A replay
  block holding no proof block is no block of the first level and escapes surely (`missProb_two_toReal_cells_on`).
* `compilerKS_lev_one`, `compilerKSTwo_lev_two`: `CompilerKS`, and `CompilerKSTwo`'s conclusion where `nPU ≤ nOut`, as
  the levels statement at one and two levels (`CompiledLev.ks_one_lev`, `CompiledLev.ks_two_lev`) through the bridges.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

theorem OneStage.strata_eq_none {ns : List ℕ} {r : OneStage} (h : r.strata ns = none) :
    ∃ num den, r = .bernoulli num den := by
  cases r <;> simp [OneStage.strata] at h ⊢

theorem strataSample_subset : ∀ (ss : List (ℕ × ℕ)) (o : ℕ) (ω : StrataΩ ss),
    strataSample ss o ω ⊆ Ico o (o + (ss.map Prod.fst).sum)
  | [], o, _ => by simp [strataSample]
  | (n, k) :: ss, o, ω => by
    intro v hv
    simp only [strataSample, mem_union, mem_image] at hv
    have ih := strataSample_subset ss (o + n) ω.2
    simp only [List.map_cons, List.sum_cons, mem_Ico]
    rcases hv with ⟨u, -, rfl⟩ | hv
    · exact ⟨by omega, by have := u.2; omega⟩
    · have := mem_Ico.1 (ih hv)
      omega

theorem nonempty_strataΩ : ∀ (ss : List (ℕ × ℕ)), (∀ p ∈ ss, p.2 ≤ p.1) → Nonempty (StrataΩ ss)
  | [], _ => ⟨()⟩
  | (n, k) :: ss, h => by
    have hk : k ≤ n := h _ List.mem_cons_self
    obtain ⟨S, -, hS⟩ := Finset.exists_subset_card_eq (s := (univ : Finset (Fin n))) (by simpa using hk)
    obtain ⟨ω⟩ := nonempty_strataΩ ss fun p hp => h p (List.mem_cons_of_mem _ hp)
    show Nonempty ({S : Finset (Fin n) // S.card = k} × StrataΩ ss)
    exact ⟨(⟨S, hS⟩, ω)⟩

/-- **A rule as a sample of its window**: a well-formed rule over strata of sizes `ns` misses `B`'s members in
`[o, o + Σ ns)` with the probability that a sample of slots in that window, on coins of its own, misses `B`. -/
theorem OneStage.realized {ns : List ℕ} {r : OneStage} (hw : r.WellFormed ns) (o : ℕ) :
    ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (S : Ω → Finset ℕ), (∀ ω, S ω ⊆ Ico o (o + ns.sum)) ∧
      ∀ B, r.Pa ns (window o ns.sum B) = Pr (fun ω => Disjoint (S ω) B) := by
  classical
  cases hs : r.strata ns with
  | some ss =>
    have hsum := OneStage.strata_sum hs hw
    haveI := nonempty_strataΩ ss (OneStage.strata_le hs hw)
    refine ⟨StrataΩ ss, inferInstance, inferInstance, strataSample ss o, fun ω => ?_, fun B => ?_⟩
    · rw [← hsum]
      exact strataSample_subset ss o ω
    · rw [OneStage.Pa_strata hs (window_subset o ns.sum B),
        strataPa_window ss (o := o) (o' := 0) (n := ns.sum) B (by rw [hsum]; omega), add_zero, strataPa_eq_Pr]
  | none =>
    obtain ⟨num, den, rfl⟩ := OneStage.strata_eq_none hs
    obtain ⟨hden, hnum⟩ := hw
    haveI : NeZero den := ⟨by omega⟩
    refine ⟨Fin ns.sum → Fin den, inferInstance, inferInstance,
      fun ω => (univ.filter fun u => (ω u).val < num).image fun u => o + u.val, fun ω v hv => ?_, fun B => ?_⟩
    · obtain ⟨u, -, rfl⟩ := mem_image.1 hv
      exact mem_Ico.2 ⟨by omega, by have := u.2; omega⟩
    · rw [OneStage.Pa_bernoulli ns hnum (window_subset o ns.sum B)]
      exact Pr_congr fun ω =>
        (disjoint_image_val_iff _ (window_subset o ns.sum B)).trans (disjoint_shift_iff o _ B).symm

theorem flatten_length_eq_strataSizes_sum (c : List (List (List ℕ))) : c.flatten.length = (strataSizes c).sum := by
  simp [strataSizes, List.length_flatten]

theorem classOff_add_le : ∀ (cs : List (List (List (List ℕ)))) (i : ℕ), i < cs.length →
    classOff cs i + (cs.getD i []).flatten.length ≤ cs.flatten.flatten.length
  | [], _, h => absurd h (by simp)
  | c :: cs, 0, _ => by simp [classOff, List.flatten_cons, List.flatten_append]
  | c :: cs, i + 1, h => by
    have := classOff_add_le cs i (by simpa using h)
    simp only [classOff, List.take_succ_cons, List.map_cons, List.sum_cons, List.getD_cons_succ, List.flatten_cons,
      List.flatten_append, List.length_append] at this ⊢
    omega

/-- **A design as its classes' independent samples**: a well-formed design on `P` misses each `B` with the probability
that one sample of slots below `|P.blocks|`, each class's on its window and coins, misses `B`. -/
theorem Design.realized {d : Design} {P : Partition} (hw : d.WellFormed P) :
    ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (S : Ω → Finset ℕ), (∀ ω, S ω ⊆ range P.blocks.length) ∧
      ∀ B, d.Pa P B = Pr (fun ω => Disjoint (S ω) B) := by
  classical
  have hcls : ∀ i : Fin d.rules.length, ∃ (Ω : Type) (_ : Fintype Ω) (_ : Nonempty Ω) (S : Ω → Finset ℕ),
      (∀ ω, S ω ⊆ range P.blocks.length) ∧
        ∀ B, classPa d.rules P.classes 0 B i = Pr (fun ω => Disjoint (S ω) B) := by
    intro i
    have hi : i.val < P.classes.length := hw.1 ▸ i.2
    have hmem : (d.rules.getD i .all, P.classes.getD i []) ∈ d.rules.zip P.classes := by
      rw [List.getD_eq_getElem _ _ i.2, List.getD_eq_getElem _ _ hi]
      exact List.mem_iff_getElem.2 ⟨i, by simp [hw.1, hi], by simp⟩
    obtain ⟨Ω, hF, hN, S, hS, hPa⟩ := OneStage.realized (hw.2 _ hmem) (classOff P.classes i)
    refine ⟨Ω, hF, hN, S, fun ω v hv => ?_, fun B => ?_⟩
    · have h1 := classOff_add_le P.classes i hi
      have h2 := mem_Ico.1 (hS ω hv)
      rw [← flatten_length_eq_strataSizes_sum] at h2
      dsimp only at h2
      exact mem_range.2 (by simp only [Partition.blocks]; omega)
    · unfold classPa
      rw [zero_add, flatten_length_eq_strataSizes_sum]
      exact hPa B
  choose Ωc hF hN Sc hSc hPa using hcls
  letI : ∀ i, Fintype (Ωc i) := hF
  haveI : ∀ i, Nonempty (Ωc i) := hN
  refine ⟨(i : Fin d.rules.length) → Ωc i, inferInstance, inferInstance,
    fun ω => univ.biUnion fun i => Sc i (ω i), fun ω => biUnion_subset.2 fun i _ => hSc i (ω i), fun B => ?_⟩
  rw [Design.Pa_eq_Pr d P B hw.1 (fun i ω => Disjoint (Sc i ω) B) fun i => hPa i B]
  exact Pr_congr fun ω => by simp [disjoint_biUnion_left]

theorem slot_of_lt {Pt : Partition} {v : ℕ} (hv : v < Pt.blocks.length) : slot Pt v = {⟨v, hv⟩} := by
  ext i
  simp [slot, Fin.ext_iff]

theorem slot_injOn (Pt : Partition) : Set.InjOn (slot Pt) (range Pt.blocks.length : Set ℕ) := fun a ha b hb h => by
  rw [slot_of_lt (mem_range.1 ha), slot_of_lt (mem_range.1 hb)] at h
  simpa using h

theorem disjoint_image_slot {Pt : Partition} {S B : Finset ℕ} (hS : S ⊆ range Pt.blocks.length)
    (hB : B ⊆ range Pt.blocks.length) : Disjoint (S.image (slot Pt)) (B.image (slot Pt)) ↔ Disjoint S B := by
  constructor
  · intro h
    exact disjoint_left.2 fun v hvS hvB => disjoint_left.1 h (mem_image_of_mem _ hvS) (mem_image_of_mem _ hvB)
  · intro h
    refine disjoint_left.2 fun X hX hX' => ?_
    obtain ⟨a, ha, rfl⟩ := mem_image.1 hX
    obtain ⟨b, hb, hab⟩ := mem_image.1 hX'
    have hba := slot_injOn Pt (mem_coe.2 (hB hb)) (mem_coe.2 (hS ha)) hab
    exact disjoint_left.1 h ha (hba ▸ hb)

/-- **One level realizes every well-formed claims design**: the proof blocks one by one as its blocks, and as its
design the classes' independent samples (`Design.realized`) read as blocks. -/
theorem levelsRealizeOne : LevelsRealizeOne := by
  classical
  intro s Pt hs
  obtain ⟨Ω, hF, hN, S, hS, hPa⟩ := Design.realized hs.2.2
  let l : Verity.Level (Fin Pt.blocks.length) :=
    { blocks := ⊥, design := fun _ => { Ω := Ω, draw := fun ω => (S ω).image (slot Pt) } }
  have hdraw : ∀ ω, ∀ b ∈ (S ω).image (slot Pt), b ∈ l.blocks.parts := fun ω b hb => by
    obtain ⟨v, hv, rfl⟩ := mem_image.1 hb
    rw [slot_of_lt (mem_range.1 (hS ω hv))]
    exact Finpartition.mem_bot_iff.2 ⟨_, mem_univ _, rfl⟩
  let Lv : Verity.Levels (Fin Pt.blocks.length) :=
    { levels := [l]
      nested := ⟨fun _ _ => ⟨univ, mem_singleton_self _, subset_univ _⟩, trivial⟩
      inside := ⟨fun X hX ω b hb => ⟨hdraw ω b hb, (mem_singleton.1 hX) ▸ subset_univ _⟩, trivial⟩ }
  have hblocks : Lv.blocks = l.blocks.parts := rfl
  refine ⟨Lv, rfl, ⟨slot_injOn Pt, fun v hv => ?_, fun X hX => ?_⟩, ?_⟩
  · rw [hblocks, slot_of_lt hv]
    exact Finpartition.mem_bot_iff.2 ⟨_, mem_univ _, rfl⟩
  · rw [hblocks] at hX
    obtain ⟨a, -, rfl⟩ := Finpartition.mem_bot_iff.1 hX
    exact ⟨a.val, a.2, slot_of_lt a.2⟩
  · rw [Verity.Levels.missesAsDesign_one (L := Lv) rfl]
    intro B hB
    show (Verity.Game.prCoin fun ω => Disjoint ((S ω).image (slot Pt)) (B.image (slot Pt))).toReal = _
    rw [hPa B, Verity.Game.pr_eq_toReal_prCoin]
    exact congrArg (fun p => (Verity.Game.prCoin p).toReal)
      (funext fun ω => propext (disjoint_image_slot (hS ω) hB))

theorem mem_topBlock {Pt Q : Partition} {ρ : ℕ} {v : Fin Pt.blocks.length} :
    v ∈ topBlock Pt Q ρ ↔ replayOf Pt Q v = some ρ := by
  simp [topBlock]

theorem mem_cell {Pt Q : Partition} {ρ t v : ℕ} :
    v ∈ cell Pt Q ρ t ↔ v < Pt.blocks.length ∧ replayOf Pt Q v = some ρ ∧ Pt.stratumOf v = t := by
  simp [cell]

theorem cell_subset_range (Pt Q : Partition) (ρ t : ℕ) : cell Pt Q ρ t ⊆ range Pt.blocks.length :=
  fun _ hv => mem_range.2 (mem_cell.1 hv).1

theorem image_slot_inter {Pt : Partition} (X : Finset ℕ) {Y : Finset ℕ} (hY : Y ⊆ range Pt.blocks.length) :
    X.image (slot Pt) ∩ Y.image (slot Pt) = (X ∩ Y).image (slot Pt) := by
  ext Z
  simp only [mem_inter, mem_image]
  constructor
  · rintro ⟨⟨a, ha, rfl⟩, ⟨c, hc, hca⟩⟩
    have hcN := mem_range.1 (hY hc)
    have hmem : (⟨c, hcN⟩ : Fin Pt.blocks.length) ∈ slot Pt a := by
      rw [← hca, slot_of_lt hcN]
      exact mem_singleton_self _
    have hac : c = a := by simpa [slot] using hmem
    exact ⟨a, ⟨ha, hac ▸ hc⟩, rfl⟩
  · rintro ⟨a, ⟨ha, hb⟩, rfl⟩
    exact ⟨⟨a, ha, rfl⟩, ⟨a, hb, rfl⟩⟩

theorem card_image_slot {Pt : Partition} {X : Finset ℕ} (hX : X ⊆ range Pt.blocks.length) :
    (X.image (slot Pt)).card = X.card :=
  card_image_of_injOn ((slot_injOn Pt).mono (coe_subset.2 hX))

/-- **Two levels realize every two-stage design in scope**: the nonempty replay blocks as the first level's blocks,
the replay design's sample (`Design.realized`) read as blocks as its design; the proof blocks one by one as the
second's, and in each replay block the uniform subsets of its cells as its design. -/
theorem levelsRealizeTwo : LevelsRealizeTwo := by
  classical
  intro s Pt d hs
  obtain ⟨-, -, hnd, hsome, hwf⟩ := hs
  obtain ⟨Ω, hF, hN, S, hS, hPa⟩ := Design.realized hwf.1
  have hlt : ∀ {v ρ : ℕ}, replayOf Pt d.part v = some ρ → ρ < d.part.blocks.length := fun h => by
    unfold replayOf at h
    exact List.mem_range.1 (List.mem_of_find?_eq_some h)
  let NE : Finset ℕ := (range d.part.blocks.length).filter fun ρ => (topBlock Pt d.part ρ).Nonempty
  let P₁ : Finpartition (univ : Finset (Fin Pt.blocks.length)) :=
    Finpartition.ofSetoid (Setoid.ker fun v : Fin Pt.blocks.length => replayOf Pt d.part v.val)
  have hrel : ∀ v w : Fin Pt.blocks.length,
      w ∈ P₁.part v ↔ replayOf Pt d.part v = replayOf Pt d.part w := fun _ _ =>
    Finpartition.mem_part_ofSetoid_iff_rel
  have hpart : ∀ (v : Fin Pt.blocks.length) ρ, replayOf Pt d.part v = some ρ → P₁.part v = topBlock Pt d.part ρ := by
    intro v ρ hv
    ext w
    rw [hrel, mem_topBlock, hv]
    exact eq_comm
  have hparts : ∀ X, X ∈ P₁.parts ↔ ∃ ρ ∈ NE, topBlock Pt d.part ρ = X := by
    intro X
    constructor
    · intro hX
      obtain ⟨v, hv⟩ := P₁.nonempty_of_mem_parts hX
      obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hsome v.val v.2)
      refine ⟨ρ, mem_filter.2 ⟨mem_range.2 (hlt hρ), v, mem_topBlock.2 hρ⟩, ?_⟩
      rw [← hpart v ρ hρ]
      exact P₁.part_eq_of_mem hX hv
    · rintro ⟨ρ, hρ, rfl⟩
      obtain ⟨v, hv⟩ := (mem_filter.1 hρ).2
      rw [← hpart v ρ (mem_topBlock.1 hv)]
      exact P₁.part_mem.2 (mem_univ v)
  let ρOf : Finset (Fin Pt.blocks.length) → ℕ := fun X =>
    if h : ∃ ρ ∈ NE, topBlock Pt d.part ρ = X then h.choose else 0
  have hρOf : ∀ ρ ∈ NE, ρOf (topBlock Pt d.part ρ) = ρ := by
    intro ρ hρ
    have h : ∃ ρ' ∈ NE, topBlock Pt d.part ρ' = topBlock Pt d.part ρ := ⟨ρ, hρ, rfl⟩
    simp only [ρOf, dif_pos h]
    exact (topBlock_injOn (mem_filter.1 hρ).2 h.choose_spec.2.symm).symm
  let C : Finset (Fin Pt.blocks.length) → ℕ → Finset (Finset (Fin Pt.blocks.length)) := fun X t =>
    (cell Pt d.part (ρOf X) t).image (slot Pt)
  let l₁ : Verity.Level (Fin Pt.blocks.length) :=
    { blocks := P₁
      design := fun _ => { Ω := Ω, draw := fun ω =>
        ((S ω).filter fun ρ => (topBlock Pt d.part ρ).Nonempty).image (topBlock Pt d.part) } }
  let l₂ : Verity.Level (Fin Pt.blocks.length) :=
    { blocks := ⊥
      design := fun X => Verity.SamplingDesign.cellSubsets Pt.strata.length (C X) fun t => d.k.getD t 0 }
  have hbot : ∀ v (hv : v < Pt.blocks.length), slot Pt v ∈ l₂.blocks.parts := fun v hv => by
    rw [slot_of_lt hv]
    exact Finpartition.mem_bot_iff.2 ⟨_, mem_univ _, rfl⟩
  have hC : ∀ ρ ∈ NE, ∀ t, ∀ b ∈ C (topBlock Pt d.part ρ) t, b ∈ l₂.blocks.parts ∧ b ⊆ topBlock Pt d.part ρ := by
    intro ρ hρ t b hb
    obtain ⟨v, hv, rfl⟩ := mem_image.1 hb
    rw [hρOf ρ hρ] at hv
    obtain ⟨hvN, hvρ, -⟩ := mem_cell.1 hv
    refine ⟨hbot v hvN, ?_⟩
    rw [slot_of_lt hvN, singleton_subset_iff]
    exact mem_topBlock.2 hvρ
  have hin₁ : ∀ ω, ∀ b ∈ (l₁.design univ).draw ω, b ∈ l₁.blocks.parts ∧ b ⊆ univ := by
    intro ω b hb
    obtain ⟨ρ, hρ, rfl⟩ := mem_image.1 hb
    obtain ⟨hρS, hne⟩ := mem_filter.1 hρ
    exact ⟨(hparts _).2 ⟨ρ, mem_filter.2 ⟨hS ω hρS, hne⟩, rfl⟩, subset_univ _⟩
  have hin₂ : ∀ X ∈ l₁.blocks.parts, ∀ ω, ∀ b ∈ (l₂.design X).draw ω, b ∈ l₂.blocks.parts ∧ b ⊆ X := by
    intro X hX ω b hb
    obtain ⟨ρ, hρ, rfl⟩ := (hparts X).1 hX
    obtain ⟨t, -, hbt⟩ := mem_biUnion.1 hb
    exact hC ρ hρ t b ((mem_powersetCard.1 (ω t).2).1 hbt)
  have hnest₂ : ∀ b ∈ l₂.blocks.parts, ∃ X ∈ l₁.blocks.parts, b ⊆ X := by
    intro b hb
    obtain ⟨v, -, rfl⟩ := Finpartition.mem_bot_iff.1 hb
    exact ⟨P₁.part v, P₁.part_mem.2 (mem_univ v), singleton_subset_iff.2 (P₁.mem_part_self.2 (mem_univ v))⟩
  let Lv : Verity.Levels (Fin Pt.blocks.length) :=
    { levels := [l₁, l₂]
      nested := ⟨fun _ _ => ⟨univ, mem_singleton_self _, subset_univ _⟩, hnest₂, trivial⟩
      inside := ⟨fun X hX ω b hb => ⟨(hin₁ ω b hb).1, (mem_singleton.1 hX) ▸ subset_univ _⟩, hin₂, trivial⟩ }
  have hblocks : Lv.blocks = l₂.blocks.parts := rfl
  have hmiss₁ : ∀ T, ((l₁.design univ).missProb T).toReal = d.replay.Pa d.part
      ((range d.part.blocks.length).filter fun ρ => (topBlock Pt d.part ρ).Nonempty ∧ topBlock Pt d.part ρ ∈ T) := by
    intro T
    rw [hPa, Verity.Game.pr_eq_toReal_prCoin]
    show (Verity.Game.prCoin fun ω => Disjoint
      (((S ω).filter fun ρ => (topBlock Pt d.part ρ).Nonempty).image (topBlock Pt d.part)) T).toReal = _
    refine congrArg (fun p => (Verity.Game.prCoin p).toReal) (funext fun ω => propext ?_)
    constructor
    · intro h
      refine disjoint_left.2 fun ρ hρS hρR => ?_
      obtain ⟨-, hne, hT⟩ := mem_filter.1 hρR
      exact disjoint_left.1 h (mem_image_of_mem _ (mem_filter.2 ⟨hρS, hne⟩)) hT
    · intro h
      refine disjoint_left.2 fun X hX hT => ?_
      obtain ⟨ρ, hρ, rfl⟩ := mem_image.1 hX
      obtain ⟨hρS, hne⟩ := mem_filter.1 hρ
      exact disjoint_left.1 h hρS (mem_filter.2 ⟨hS ω hρS, hne, hT⟩)
  have hesc : ∀ ρ ∈ NE, ∀ X : Finset ℕ, ((l₂.design (topBlock Pt d.part ρ)).missProb (X.image (slot Pt))).toReal =
      ∏ t ∈ range Pt.strata.length,
        escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (X ∩ cell Pt d.part ρ t).card := by
    intro ρ hρ X
    show ((Verity.SamplingDesign.cellSubsets Pt.strata.length (C (topBlock Pt d.part ρ)) fun t => d.k.getD t 0).missProb
      (X.image (slot Pt))).toReal = _
    rw [Verity.SamplingDesign.missProb_cellSubsets, ENNReal.toReal_prod, ← Fin.prod_univ_eq_prod_range]
    refine prod_congr rfl fun t _ => ?_
    simp only [C, hρOf ρ hρ]
    rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast,
      card_image_slot (cell_subset_range _ _ _ _), image_slot_inter X (cell_subset_range _ _ _ _),
      card_image_slot (inter_subset_right.trans (cell_subset_range _ _ _ _))]
    rfl
  have hbij : Set.BijOn ρOf (l₁.blocks.parts : Set (Finset (Fin Pt.blocks.length))) (NE : Set ℕ) := by
    refine ⟨fun X hX => ?_, fun X hX Y hY hXY => ?_, fun ρ hρ => ?_⟩
    · obtain ⟨ρ, hρ, rfl⟩ := (hparts X).1 hX
      rw [mem_coe, hρOf ρ hρ]
      exact hρ
    · obtain ⟨ρ, hρ, rfl⟩ := (hparts X).1 hX
      obtain ⟨ρ', hρ', rfl⟩ := (hparts Y).1 hY
      rw [hρOf ρ hρ, hρOf ρ' hρ'] at hXY
      rw [hXY]
    · exact ⟨topBlock Pt d.part ρ, (hparts _).2 ⟨ρ, hρ, rfl⟩, hρOf ρ hρ⟩
  refine ⟨Lv, ⟨hnd, hsome, l₁, [l₂], rfl, List.cons_ne_nil _ _, fun v w => ?_⟩,
    ⟨slot_injOn Pt, fun v hv => hbot v hv, fun X hX => ?_⟩, fun B hB => ?_,
    l₁, l₂, rfl, fun T => (hmiss₁ T).symm.le, fun ρ hρ hne X => (hesc ρ (mem_filter.2 ⟨mem_range.2 hρ, hne⟩) X).symm.le⟩
  · constructor
    · intro h
      exact ⟨P₁.part v, P₁.part_mem.2 (mem_univ v), P₁.mem_part_self.2 (mem_univ v), (hrel v w).2 h⟩
    · rintro ⟨X, hX, hvX, hwX⟩
      have hv := P₁.part_eq_of_mem hX hvX
      exact (hrel v w).1 (by rw [hv]; exact hwX)
  · rw [hblocks] at hX
    obtain ⟨a, -, rfl⟩ := Finpartition.mem_bot_iff.1 hX
    exact ⟨a.val, a.2, slot_of_lt a.2⟩
  · refine Lv.missProb_two_toReal_cells_on rfl d Pt ρOf NE (filter_subset _ _) hbij (B.image (slot Pt))
      (counts Pt d.part B) (fun R hR => ?_) C (fun _ _ => rfl) (fun X hX t _ => ?_) (fun X hX t _ => ?_) ?_
    · rw [hmiss₁]
      congr 1
      ext ρ
      simp only [mem_filter, mem_range, mem_image]
      constructor
      · rintro ⟨hρ, hne, hR'⟩
        exact ⟨_, hR', hρOf ρ (mem_filter.2 ⟨mem_range.2 hρ, hne⟩)⟩
      · rintro ⟨X, hXR, rfl⟩
        obtain ⟨ρ, hρ, rfl⟩ := (hparts X).1 (hR hXR)
        rw [hρOf ρ hρ]
        exact ⟨mem_range.1 (mem_filter.1 hρ).1, (mem_filter.1 hρ).2, hXR⟩
    · exact card_image_slot (cell_subset_range _ _ _ _)
    · obtain ⟨ρ, hρ, rfl⟩ := (hparts X).1 hX
      have hsub : (B.image (slot Pt)).filter (· ⊆ topBlock Pt d.part ρ) ∩ C (topBlock Pt d.part ρ) t =
          B.image (slot Pt) ∩ C (topBlock Pt d.part ρ) t := by
        ext Z
        simp only [mem_inter, mem_filter]
        exact ⟨fun h => ⟨h.1.1, h.2⟩, fun h => ⟨⟨h.1, (hC ρ hρ t Z h.2).2⟩, h.2⟩⟩
      rw [hsub]
      simp only [C, counts, hρOf ρ hρ]
      rw [image_slot_inter B (cell_subset_range _ _ _ _),
        card_image_slot (inter_subset_right.trans (cell_subset_range _ _ _ _))]
    · intro i hi
      have hne : ¬ (topBlock Pt d.part i).Nonempty := fun h =>
        (mem_sdiff.1 hi).2 (mem_filter.2 ⟨(mem_sdiff.1 hi).1, h⟩)
      unfold Design.Two.esc
      refine prod_eq_one fun t _ => ?_
      simp only [counts, cell_eq_empty hne]
      exact Claims.SamplingGeneric.CompiledTwo.escCell_empty _ _

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims.Sampling

/-- **`CompilerKS` is the levels statement at one level**: `CompiledLev.ks_one_lev` at `Compiled.liftLev`, read back
through `levelsRealizeOne`. -/
theorem compilerKS_lev_one : CompilerKS :=
  fun _ _ _ _ _ _ _ hA hB hS => Compiled.ks_of_liftLev levelsRealizeOne fun δ hδ x hx =>
    CompiledLev.ks_one_lev hA hB (Compiled.sessions_liftLev levelsRealizeOne hS) δ hδ x ⟨hx, fun _ _ => rfl⟩

/-- **`CompilerKSTwo`'s conclusion is the levels statement at two levels** where the proof blocks fit in the outer
budget (`nPU ≤ nOut`): `CompiledLev.ks_two_lev` at `CompiledTwo.liftLev`, read back through `levelsRealizeTwo`. -/
theorem compilerKSTwo_lev_two (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin)
    (L : CompiledTwo Coin A) (H : Msg → Msg) (hA : A.KS H) (hB : L.commit.Binds H) (hS : L.Sessions H)
    (hpu : ∀ x, L.nPU x ≤ L.nOut x) : L.KS H :=
  CompiledTwo.ks_of_liftLev levelsRealizeTwo fun δ hδ x hx =>
    CompiledLev.ks_two_lev hA hB (CompiledTwo.sessions_liftLev levelsRealizeTwo hS hpu) δ hδ x
      ⟨hx, fun _ _ _ _ _ hLv => by
        obtain ⟨_, -, -, -, -, -, l₁, l₂, hL, -⟩ := CompiledTwo.liftLev_levels hLv
        exact ⟨l₁, l₂, hL⟩⟩

end Claims.SamplingGeneric
