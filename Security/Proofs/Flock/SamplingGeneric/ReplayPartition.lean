module

public import Definitions.Claims.SamplingTwo
import Mathlib.Data.List.Nodup

/-!
# Replay partitions

- **What reads only a partition's shape**, its classes' strata sizes: a design's acceptance probability
  (`Design.Pa_shape`), its well-formedness (`Design.wellFormed_shape`) and the number of blocks
  (`Partition.length_blocks_shape`), so a replay partition may be any partition of the replay layout's shape.
- **Replay blocks that are unions of proof blocks**: a replay partition `Pρ` whose block `ρ` is the proof blocks of `Pt`
  at the indices `I ρ` together, the index sets disjoint, over a proof-block partition `Pt` whose gates are distinct and
  whose proof blocks are not empty: proof block `v` lies in replay block `ρ` (`replayOf`) exactly when `v ∈ I ρ`
  (`replayOf_eq_some_iff_union`), so cell `(ρ, t)` is the indices of `I ρ` in stratum `t` (`cell_eq_union`).
-/

@[expose] public section

namespace Claims.Sampling

/-! ## What reads only a partition's shape -/

theorem length_flatten_strata (c : List (List (List ℕ))) : c.flatten.length = (strataSizes c).sum := by
  simp [strataSizes, List.length_flatten]

theorem classesPa_shape : ∀ (rs : List OneStage) {cs cs' : List (List (List (List ℕ)))} (o : ℕ) (B : Finset ℕ),
    cs.map strataSizes = cs'.map strataSizes → classesPa rs cs o B = classesPa rs cs' o B
  | [], _, _, _, _, _ => rfl
  | _ :: _, [], [], _, _, _ => rfl
  | _ :: _, [], _ :: _, _, _, h => by simp at h
  | _ :: _, _ :: _, [], _, _, h => by simp at h
  | r :: rs, c :: cs, c' :: cs', o, B, h => by
    simp only [List.map_cons, List.cons.injEq] at h
    have hl : c.flatten.length = c'.flatten.length := by rw [length_flatten_strata, length_flatten_strata, h.1]
    simp only [classesPa, h.1, hl, classesPa_shape rs (o + c'.flatten.length) B h.2]

/-- **A design's acceptance probability reads only the partition's shape.** -/
theorem Design.Pa_shape (d : Design) {P Q : Partition} (h : P.classes.map strataSizes = Q.classes.map strataSizes)
    (B : Finset ℕ) : d.Pa P B = d.Pa Q B :=
  classesPa_shape d.rules 0 B h

theorem forall_zip_strataSizes (rs : List OneStage) (cs : List (List (List (List ℕ)))) :
    (∀ p ∈ rs.zip cs, p.1.WellFormed (strataSizes p.2)) ↔ ∀ q ∈ rs.zip (cs.map strataSizes), q.1.WellFormed q.2 := by
  rw [List.zip_map_right]
  constructor
  · intro h q hq
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hq
    exact h p hp
  · intro h p hp
    exact h _ (List.mem_map.2 ⟨p, hp, rfl⟩)

/-- **A design's well-formedness reads only the partition's shape.** -/
theorem Design.wellFormed_shape (d : Design) {P Q : Partition}
    (h : P.classes.map strataSizes = Q.classes.map strataSizes) : d.WellFormed P ↔ d.WellFormed Q := by
  have hlen : P.classes.length = Q.classes.length := by simpa using congrArg List.length h
  unfold Design.WellFormed
  rw [forall_zip_strataSizes, forall_zip_strataSizes, h, hlen]

/-- **The number of blocks reads only the partition's shape.** -/
theorem Partition.length_blocks_shape {P Q : Partition} (h : P.classes.map strataSizes = Q.classes.map strataSizes) :
    P.blocks.length = Q.blocks.length := by
  suffices e : ∀ cs : List (List (List (List ℕ))), cs.flatten.flatten.length = ((cs.map strataSizes).map List.sum).sum by
    simp only [Partition.blocks, e, h]
  intro cs
  induction cs with
  | nil => rfl
  | cons c cs ih =>
    rw [List.flatten_cons, List.flatten_append, List.length_append, ih, length_flatten_strata]
    simp only [List.map_cons, List.sum_cons]

/-! ## Replay blocks that are unions of proof blocks -/

variable {Pt Pρ : Partition} {I : ℕ → List ℕ}

/-- Two proof blocks of a partition whose gates are distinct share no gate. -/
theorem Partition.disjoint_blocks (hnd : Pt.gates.Nodup) {v w : ℕ} (hvw : v ≠ w) {a b : List ℕ}
    (ha : Pt.blocks[v]? = some a) (hb : Pt.blocks[w]? = some b) : a.Disjoint b := by
  have hp := (List.nodup_flatten.1 (show Pt.blocks.flatten.Nodup from hnd)).2
  obtain ⟨hv, rfl⟩ := List.getElem?_eq_some_iff.1 ha
  obtain ⟨hw, rfl⟩ := List.getElem?_eq_some_iff.1 hb
  rcases Nat.lt_or_gt_of_ne hvw with h | h
  · exact List.pairwise_iff_getElem.1 hp v w hv hw h
  · exact (List.pairwise_iff_getElem.1 hp w v hw hv h).symm

section Union

variable (hnd : Pt.gates.Nodup) (hne : ∀ u ∈ Pt.blocks, u ≠ [])
include hnd hne

/-- A proof block not at the indices `I ρ` has a gate outside replay block `ρ`. -/
theorem exists_not_mem_union {v ρ : ℕ} (hv : v < Pt.blocks.length) (hvI : v ∉ I ρ) :
    ∃ g ∈ Pt.blocks[v], g ∉ (I ρ).flatMap fun w => Pt.blocks.getD w [] := by
  obtain ⟨g, hg⟩ := List.exists_mem_of_ne_nil _ (hne _ (List.getElem_mem hv))
  refine ⟨g, hg, fun hmem => ?_⟩
  obtain ⟨w, hw, hgw⟩ := List.mem_flatMap.1 hmem
  have hwv : v ≠ w := fun h => hvI (h ▸ hw)
  rcases Nat.lt_or_ge w Pt.blocks.length with hwl | hwl
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hwl, Option.getD_some] at hgw
    exact Partition.disjoint_blocks hnd hwv (List.getElem?_eq_getElem hv) (List.getElem?_eq_getElem hwl) hg hgw
  · rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hwl, Option.getD_none] at hgw
    simp at hgw

variable (hρ : ∀ ρ, ρ < Pρ.blocks.length → Pρ.blocks[ρ]? = some ((I ρ).flatMap fun w => Pt.blocks.getD w []))
include hρ

/-- **A proof block lies in the replay block whose indices hold it**, the index sets disjoint. -/
theorem replayOf_eq_some_iff_union
    (hdisj : ∀ ρ ρ', ρ < Pρ.blocks.length → ρ' < Pρ.blocks.length → ρ ≠ ρ' → ∀ w ∈ I ρ, w ∉ I ρ')
    {v ρ : ℕ} (hv : v < Pt.blocks.length) (hρl : ρ < Pρ.blocks.length) :
    replayOf Pt Pρ v = some ρ ↔ v ∈ I ρ := by
  have hb : Pt.blocks[v]? = some Pt.blocks[v] := List.getElem?_eq_getElem hv
  unfold replayOf
  rw [List.find?_range_eq_some]
  constructor
  · rintro ⟨hp, -, -⟩
    by_contra hvI
    obtain ⟨g, hg, hgn⟩ := exists_not_mem_union hnd hne hv hvI
    simp only [hb, hρ ρ hρl, List.all_eq_true, List.contains_iff_mem] at hp
    exact hgn (hp g hg)
  · intro hvI
    refine ⟨?_, List.mem_range.2 hρl, fun j hj => ?_⟩
    · simp only [hb, hρ ρ hρl, List.all_eq_true, List.contains_iff_mem]
      intro g hg
      refine List.mem_flatMap.2 ⟨v, hvI, ?_⟩
      rwa [List.getD_eq_getElem?_getD, hb, Option.getD_some]
    · have hjl : j < Pρ.blocks.length := by omega
      obtain ⟨g, hg, hgn⟩ := exists_not_mem_union hnd hne hv (hdisj ρ j hρl hjl (by omega) v hvI)
      simp only [hb, hρ j hjl, Bool.not_eq_eq_eq_not, Bool.not_true, List.all_eq_false, List.contains_iff_mem]
      exact ⟨g, hg, hgn⟩

/-- **Cell `(ρ, t)` is the indices of `I ρ` in stratum `t`**, every index a proof block. -/
theorem cell_eq_union
    (hdisj : ∀ ρ ρ', ρ < Pρ.blocks.length → ρ' < Pρ.blocks.length → ρ ≠ ρ' → ∀ w ∈ I ρ, w ∉ I ρ')
    (hI : ∀ ρ, ρ < Pρ.blocks.length → ∀ w ∈ I ρ, w < Pt.blocks.length) {ρ t : ℕ} (hρl : ρ < Pρ.blocks.length) :
    cell Pt Pρ ρ t = ((I ρ).filter fun v => Pt.stratumOf v = t).toFinset := by
  ext v
  simp only [cell, Finset.mem_filter, Finset.mem_range, List.mem_toFinset, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨hv, hr, hs⟩
    exact ⟨(replayOf_eq_some_iff_union hnd hne hρ hdisj hv hρl).1 hr, hs⟩
  · rintro ⟨hvI, hs⟩
    have hv := hI ρ hρl v hvI
    exact ⟨hv, (replayOf_eq_some_iff_union hnd hne hρ hdisj hv hρl).2 hvI, hs⟩

end Union

end Claims.Sampling
