import Proofs.Pous.Chain.Flat

/-!
# The cumulative chain clock, and the band's hardness from one held label

`Pous.BandChain.clock_invariant` bounds one row: a label that is not a seed has its row's calls within the rounds.
`cum_clock` adds where the row started: its first call's block was already known `ovCalls G v` rounds earlier. In the
band, that block is label `v − 1` (`ovBlock_band_one`), so a run of labels with no seed costs the sum of its rows'
calls (`band_run`), and a holding priced at one label leaves a run of at least 251 labels of 6 calls or more.
-/

namespace PousRebuild

open Pous Pous.Labelling Pous.Sponge Pous.BandChain

section Clock

variable {N : ℕ}

/-- **The cumulative clock invariant.** After `t` rounds from `P₀`, on a row with no held chaining value:
1. a known chaining value `h_{v,j}` has `j ≤ t`, and the block of the row's first call was known `j` rounds earlier;
2. so does a known dropped `r_{v,j}` that was not held;
and every known label is a seed of `P₀`, or its row has at most `t` calls and its first call's block was known
`ovCalls G v` rounds earlier. -/
theorem cum_clock (G : TopoDAG N) (P₀ : ChainKnow N) : ∀ t,
    (∀ p ∈ (ChainKnow.reach G P₀ t).hs, p.2 ≠ 0 →
        p.1 ∉ (P₀.hs.filter fun q => q.2 ≠ 0).image Prod.fst →
        p.2 ≤ t ∧ ∀ u, ovBlock G p.1 1 = some u → u ∈ (ChainKnow.reach G P₀ (t - p.2)).labs) ∧
    (∀ p ∈ (ChainKnow.reach G P₀ t).rs, p ∉ P₀.rs →
        p.1 ∉ (P₀.hs.filter fun q => q.2 ≠ 0).image Prod.fst →
        p.2 ≤ t ∧ ∀ u, ovBlock G p.1 1 = some u → u ∈ (ChainKnow.reach G P₀ (t - p.2)).labs) ∧
    (∀ v ∈ (ChainKnow.reach G P₀ t).labs, v ∈ P₀.seeds G ∨
        (ovCalls G v ≤ t ∧
          ∀ u, ovBlock G v 1 = some u → u ∈ (ChainKnow.reach G P₀ (t - ovCalls G v)).labs)) := by
  intro t
  induction t with
  | zero =>
    refine ⟨?_, ?_, ?_⟩
    · intro p hp h0 hH
      exact absurd (Finset.mem_image.2 ⟨p, Finset.mem_filter.2 ⟨hp, h0⟩, rfl⟩) hH
    · intro p hp hn; exact absurd hp hn
    · intro v hv
      rw [ChainKnow.reach, mem_read_labs] at hv
      left
      unfold ChainKnow.seeds
      rcases hv with hv | hv
      · exact Finset.mem_union_left _ (Finset.mem_union_left _ hv)
      · exact Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨_, hv, by simp⟩)
  | succ t ih =>
    obtain ⟨ih1, ih2, ih3⟩ := ih
    set K := ChainKnow.reach G P₀ t with hK
    have hKs : ChainKnow.reach G P₀ (t + 1) = K.round G := rfl
    have hmono : ∀ {a b : ℕ}, a ≤ b → (ChainKnow.reach G P₀ a).labs ⊆ (ChainKnow.reach G P₀ b).labs :=
      fun h => reach_labs_mono G P₀ h
    -- a call made in round `t + 1`
    have hcall : ∀ p : Fin N × ℕ, K.canCall G p.1 p.2 →
        p.1 ∉ (P₀.hs.filter fun q => q.2 ≠ 0).image Prod.fst →
        p.2 ≤ t + 1 ∧ ∀ u, ovBlock G p.1 1 = some u → u ∈ (ChainKnow.reach G P₀ (t + 1 - p.2)).labs := by
      rintro p ⟨hj1, -, hblk, hprev⟩ hH
      by_cases hp1 : p.2 = 1
      · refine ⟨by omega, fun u hu => ?_⟩
        rw [show t + 1 - p.2 = t by omega]
        exact hblk u (by rw [hp1]; exact hu)
      · rcases hprev with h1 | hprev
        · exact absurd h1 hp1
        · obtain ⟨hle, hlab⟩ := ih1 _ hprev (by simp only; omega) hH
          simp only at hle hlab
          refine ⟨by omega, fun u hu => ?_⟩
          rw [show t + 1 - p.2 = t - (p.2 - 1) by omega]
          exact hlab u hu
    have h2 : ∀ p ∈ (K.round G).rs, p ∉ P₀.rs →
        p.1 ∉ (P₀.hs.filter fun q => q.2 ≠ 0).image Prod.fst →
        p.2 ≤ t + 1 ∧ ∀ u, ovBlock G p.1 1 = some u → u ∈ (ChainKnow.reach G P₀ (t + 1 - p.2)).labs := by
      intro p hp hn hH
      rcases mem_round_rs G K p hp with hp' | hc
      · obtain ⟨hle, hlab⟩ := ih2 p hp' hn hH
        exact ⟨by omega, fun u hu => hmono (by omega) (hlab u hu)⟩
      · exact hcall p hc hH
    refine ⟨?_, ?_, ?_⟩
    · intro p hp h0 hH
      rw [hKs] at hp
      rcases mem_round_hs G K p hp with hp' | hc | ⟨-, hnext, -⟩
      · obtain ⟨hle, hlab⟩ := ih1 p hp' h0 hH
        exact ⟨by omega, fun u hu => hmono (by omega) (hlab u hu)⟩
      · exact hcall p hc hH
      · obtain ⟨hle, hlab⟩ := ih1 _ hnext (by simp) hH
        simp only at hle hlab
        exact ⟨by omega, fun u hu => hmono (by omega) (hlab u hu)⟩
    · rw [hKs]; exact h2
    · intro v hv
      rw [hKs, mem_round_labs] at hv
      rcases hv with (hv | ⟨p, hph, hpr, hb⟩) | hread
      · rcases ih3 v hv with h | ⟨hle, hlab⟩
        · exact Or.inl h
        · exact Or.inr ⟨by omega, fun u hu => hmono (by omega) (hlab u hu)⟩
      · by_cases hn : p ∈ P₀.rs
        · left
          unfold ChainKnow.seeds
          refine Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨p, hn, ?_⟩)
          have hne : p.2 ≠ ovCalls G p.1 := fun h => by rw [h, ovBlock_pad] at hb; simp at hb
          simp [hne, hb]
        · rcases ih3 v ((chain_invariant G P₀ t).2.2.1 p hpr hn v hb) with h | ⟨hle, hlab⟩
          · exact Or.inl h
          · exact Or.inr ⟨by omega, fun u hu => hmono (by omega) (hlab u hu)⟩
      · by_cases hn : (v, ovCalls G v) ∈ P₀.rs
        · left
          unfold ChainKnow.seeds
          exact Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨_, hn, by simp⟩)
        · by_cases hH : v ∈ (P₀.hs.filter fun q => q.2 ≠ 0).image Prod.fst
          · left
            unfold ChainKnow.seeds
            exact Finset.mem_union_left _ (Finset.mem_union_right _ hH)
          · exact Or.inr (h2 _ hread hn hH)

end Clock

/-- In the band, label `v` costs `min(v, d) + 1` calls. -/
theorem ovCalls_band (B d : ℕ) (v : Fin B) : ovCalls (bandDAG B d) v = min v.val d + 1 := by
  rw [ovCalls_eq, card_par_band]

/-- **In the band with `d ≥ 1`, the first call of label `v ≥ 1` absorbs label `v − 1`**: the parents are absorbed
newest first. -/
theorem ovBlock_band_one (B d : ℕ) (hd : 1 ≤ d) (v : Fin B) (hv : 1 ≤ v.val) :
    ovBlock (bandDAG B d) v 1 = some ⟨v.val - 1, by omega⟩ := by
  have hmem : (⟨v.val - 1, by omega⟩ : Fin B) ∈ parentsDesc (bandDAG B d) v :=
    (Finset.mem_sort _).2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, Fin.lt_def.2 (by simp only; omega),
      by simp only; omega⟩)
  have hsort : (parentsDesc (bandDAG B d) v).Pairwise (fun a b : Fin B => b ≤ a) := Finset.pairwise_sort _ _
  have hpar : ∀ u ∈ parentsDesc (bandDAG B d) v, u < v := fun u hu =>
    (bandDAG B d).par_lt v u ((Finset.mem_sort _).1 hu)
  unfold ovBlock
  revert hmem hsort hpar
  generalize parentsDesc (bandDAG B d) v = L
  intro hmem hsort hpar
  cases L with
  | nil => simp at hmem
  | cons h tl =>
    have hlt := hpar h List.mem_cons_self
    rw [Fin.lt_def] at hlt
    simp only [Nat.sub_self, List.getElem?_cons_zero, Option.some.injEq]
    rcases List.mem_cons.1 hmem with heq | htl
    · exact heq.symm
    · have hle := List.rel_of_pairwise_cons hsort htl
      rw [Fin.le_def] at hle
      simp only at hle
      exact Fin.ext (by simp only; omega)

/-- **A run of labels with no seed costs at least 6 rounds per label**, in the band with `d ≥ 5` from index 5 on: if
label `lo + n` is known after `t` rounds and none of labels `lo, …, lo + n` is a seed, then `6·(n + 1) ≤ t`. -/
theorem band_run (B d : ℕ) (hd : 5 ≤ d) (P₀ : ChainKnow B) (lo : ℕ) (hlo : 5 ≤ lo) :
    ∀ n t (h : lo + n < B), (⟨lo + n, h⟩ : Fin B) ∈ (ChainKnow.reach (bandDAG B d) P₀ t).labs →
      (∀ w : Fin B, lo ≤ w.val → w.val ≤ lo + n → w ∉ P₀.seeds (bandDAG B d)) → 6 * (n + 1) ≤ t := by
  intro n
  induction n with
  | zero =>
    intro t h hv hS
    rcases (cum_clock _ P₀ t).2.2 _ hv with hs | ⟨hle, -⟩
    · exact absurd hs (hS _ (by simp only; omega) (by simp only; omega))
    · rw [ovCalls_band] at hle
      simp only at hle
      have := le_min (by omega : 5 ≤ lo + 0) hd
      omega
  | succ n ih =>
    intro t h hv hS
    rcases (cum_clock _ P₀ t).2.2 _ hv with hs | ⟨hle, hlab⟩
    · exact absurd hs (hS _ (by simp only; omega) (by simp only; omega))
    · have hb := ovBlock_band_one B d (by omega) ⟨lo + (n + 1), h⟩ (by simp only; omega)
      have hprev := hlab _ hb
      have heq : (⟨(⟨lo + (n + 1), h⟩ : Fin B).val - 1, by simp only; omega⟩ : Fin B) = ⟨lo + n, by omega⟩ :=
        Fin.ext (by simp only; omega)
      rw [heq] at hprev
      have hrec := ih _ (by omega) hprev (fun w h1 h2 => hS w h1 (by omega))
      rw [ovCalls_band] at hle hrec
      simp only at hle hrec
      have := le_min (by omega : 5 ≤ lo + (n + 1)) hd
      omega

/-- **The band with `d ≥ 5` is `(1, D, 2^9)`-hard for `D ≤ 1505`**: a holding priced at one label never knows all
`2^9` labels within 1505 rounds. It has at most one seed, so labels `5 … 255` (251 labels) or labels `256 … 511`
(256 labels) hold none, and each of those labels costs at least 6 calls. -/
theorem band_hard_one (d : ℕ) (hd : 5 ≤ d) (c ℓ D : ℕ) (hD : D ≤ 1505) :
    ChainHard (bandDAG (2 ^ 9) d) Finset.univ c ℓ ((1 : ℕ) : ℚ) D (2 ^ 9) := by
  intro P₀ hsize
  have hcard : (P₀.seeds (bandDAG (2 ^ 9) d)).card ≤ 1 := by
    have := (card_seeds_le_size (bandDAG (2 ^ 9) d) c ℓ P₀).trans hsize
    exact_mod_cast this
  by_contra hcon
  have hall : ∀ v : Fin (2 ^ 9), v ∈ (ChainKnow.reach (bandDAG (2 ^ 9) d) P₀ D).labs := by
    intro v
    by_contra hv
    have hlt : (Finset.univ.filter fun v => v ∈ (ChainKnow.reach (bandDAG (2 ^ 9) d) P₀ D).labs).card <
        (Finset.univ : Finset (Fin (2 ^ 9))).card :=
      Finset.card_lt_card (Finset.filter_ssubset.2 ⟨v, Finset.mem_univ _, hv⟩)
    rw [Finset.card_univ, Fintype.card_fin] at hlt
    exact hcon hlt
  by_cases hhi : ∀ w : Fin (2 ^ 9), 256 ≤ w.val → w.val ≤ 256 + 255 → w ∉ P₀.seeds (bandDAG (2 ^ 9) d)
  · have := band_run (2 ^ 9) d hd P₀ 256 (by norm_num) 255 D (by norm_num) (hall _) hhi
    omega
  · push Not at hhi
    obtain ⟨w, hw1, -, hw⟩ := hhi
    have hlo : ∀ w' : Fin (2 ^ 9), 5 ≤ w'.val → w'.val ≤ 5 + 250 → w' ∉ P₀.seeds (bandDAG (2 ^ 9) d) := by
      intro w' h1 h2 hw'
      have hww : w' = w := Finset.card_le_one.1 hcard w' hw' w hw
      rw [hww] at h2
      omega
    have := band_run (2 ^ 9) d hd P₀ 5 le_rfl 250 D (by norm_num) (hall _) hlo
    omega

end PousRebuild
