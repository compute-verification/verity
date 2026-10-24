import Mathlib
import Definitions.Accounting.Guarantees.RateAuditSketch

/-!
# The honest device: the sketch rate audit is not vacuous

A device that holds nothing on chip takes the certified stream's `ρ` sectors in over the link, reads every sector of
the store from device memory once, and queries the oracle at each tile's digest under the stream. It is accepted at
every signs, oracle, data and sample; its footprint is the store, `N·m·w` bits, its link carries exactly the stream, and
its compute is `N` tags (`honest_accepts`; the guarantee `Accounting.SecurityProofs.RateSketchHonest`).
-/

namespace Accounting.RateSketch.Completeness

open Pous (Bits)

variable {N m ρ h : ℕ} {X Dg α : Type}

/-- Read the listed positions into `acc`, a store sector from device memory and a stream sector over the link, then
continue with it. -/
def readAll : List (Pos N m ρ) → Data N m ρ X → (Data N m ρ X → Chip N m ρ h X Dg α) → Chip N m ρ h X Dg α
  | [], acc, k => k acc
  | .inl p :: ps, acc, k => .mem p (fun x => readAll ps (Function.update acc (.inl p) x) k)
  | .inr r :: ps, acc, k => .link (.inr r) (fun x => readAll ps (Function.update acc (.inr r) x) k)

open Classical in
theorem readAll_eq (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (ps : List (Pos N m ρ)) (acc : Data N m ρ X)
    (k : Data N m ρ X → Chip N m ρ h X Dg α) :
    let V : Data N m ρ X := fun p => if p ∈ ps then D p else acc p
    (readAll ps acc k).run O D = (k V).run O D ∧ (readAll ps acc k).queries O D = (k V).queries O D ∧
    (readAll ps acc k).mems O D = (ps.filterMap Sum.getLeft?).toFinset ∪ (k V).mems O D ∧
    (readAll ps acc k).linkSteps O D = (k V).linkSteps O D + ps.countP Sum.isRight ∧
    (readAll ps acc k).rebuilds O D = (k V).rebuilds O D := by
  induction ps generalizing acc with
  | nil => simp [readAll]
  | cons p ps ih =>
    have hR : (fun q => if q ∈ ps then D q else Function.update acc p (D p) q) =
        fun q => if q ∈ p :: ps then D q else acc q := by
      funext q
      by_cases hq : q = p
      · subst hq; simp
      · simp [hq]
    rcases p with p | r
    · obtain ⟨h1, h2, h3, h4, h5⟩ := ih (Function.update acc (.inl p) (D (.inl p)))
      simp only [readAll, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds]
      rw [h1, h2, h3, h4, h5, hR]
      refine ⟨rfl, rfl, ?_, by simp, rfl⟩
      simp [Finset.insert_union]
    · obtain ⟨h1, h2, h3, h4, h5⟩ := ih (Function.update acc (.inr r) (D (.inr r)))
      simp only [readAll, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds]
      rw [h1, h2, h3, h4, h5, hR]
      refine ⟨rfl, rfl, by simp [List.filterMap_cons], by simp [List.countP_cons]; omega, rfl⟩

/-- Query the oracle at each listed tile's digest under the stream of `V`, recording the tag. -/
def tagAll {Sig : Type} (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (V : Data N m ρ X) :
    List (Fin N) → (Fin N → Bits h) → Chip N m ρ h X Dg (Fin N → Bits h)
  | [], acc => .done acc
  | t :: ts, acc => .tag (V.stream, t, dig σ (V.tile t)) (fun y => tagAll dig σ V ts (Function.update acc t y))

theorem tagAll_eq {Sig : Type} (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D V : Data N m ρ X) (ts : List (Fin N)) (acc : Fin N → Bits h) :
    (tagAll dig σ V ts acc).run O D = (fun t => if t ∈ ts then O (V.stream, t, dig σ (V.tile t)) else acc t) ∧
    (tagAll dig σ V ts acc).queries O D = ts.map (fun t => (V.stream, t, dig σ (V.tile t))) ∧
    (tagAll dig σ V ts acc).mems O D = ∅ ∧ (tagAll dig σ V ts acc).linkSteps O D = 0 ∧
    (tagAll dig σ V ts acc).rebuilds O D = 0 := by
  induction ts generalizing acc with
  | nil => simp [tagAll, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds]
  | cons t ts ih =>
    obtain ⟨i1, i2, i3, i4, i5⟩ := ih (Function.update acc t (O (V.stream, t, dig σ (V.tile t))))
    simp only [tagAll, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds]
    refine ⟨?_, by rw [i2, List.map_cons], i3, i4, i5⟩
    rw [i1]
    funext t'
    by_cases ht : t' = t
    · subst ht; simp
    · by_cases hm : t' ∈ ts <;> simp [hm, ht]

/-- Every position: the stream's sectors, then the store's tile by tile. -/
def allPos (N m ρ : ℕ) : List (Pos N m ρ) :=
  (List.finRange ρ).map Sum.inr ++ (List.finRange N).flatMap fun t => (List.finRange m).map fun u => Sum.inl (t, u)

theorem mem_allPos (p : Pos N m ρ) : p ∈ allPos N m ρ := by
  rcases p with ⟨t, u⟩ | r <;> simp [allPos]

/-- The honest chip: take in the stream and read the store, then tag every tile under the stream. -/
def chip [Inhabited X] {Sig : Type} (dig : Sig → (Fin m → X) → Dg) (σ : Sig) : Chip N m ρ h X Dg (Fin N → Bits h) :=
  readAll (allPos N m ρ) (fun _ => default) (fun V => tagAll dig σ V (List.finRange N) (fun _ _ => false))

open Classical in
theorem chip_eq [Inhabited X] {Sig : Type} (dig : Sig → (Fin m → X) → Dg) (σ : Sig) (O : Oracle N ρ h X Dg)
    (D : Data N m ρ X) :
    (chip (h := h) dig σ).run O D = (fun t => O (D.stream, t, dig σ (D.tile t))) ∧
    ((chip (h := h) dig σ).queries O D).length = N ∧
    (chip (h := h) dig σ).mems O D = Finset.univ ∧ (chip (h := h) dig σ).linkSteps O D = ρ ∧
    (chip (h := h) dig σ).rebuilds O D = 0 := by
  have hV : (fun p => if p ∈ allPos N m ρ then D p else (fun _ => default) p) = D := by
    funext p; simp [mem_allPos]
  obtain ⟨h1, h2, h3, h4, h5⟩ := readAll_eq (h := h) O D (allPos N m ρ) (fun _ => default)
    (fun V => tagAll dig σ V (List.finRange N) (fun _ _ => false))
  simp only [hV] at h1 h2 h3 h4 h5
  obtain ⟨i1, i2, i3, i4, i5⟩ := tagAll_eq (h := h) dig σ O D D (List.finRange N) (fun _ _ => false)
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [chip, h1, i1]
    simp
  · rw [chip, h2, i2]
    simp
  · rw [chip, h3, i3, Finset.union_empty]
    ext ⟨t, u⟩
    simp [allPos]
  · rw [chip, h4, i4, zero_add]
    simp [allPos, List.countP_append, List.countP_flatMap, List.countP_map, Function.comp_def]
  · rw [chip, h5, i5]

/-- The honest device: nothing on chip; it takes in the stream, reads every sector once, and tags every tile. -/
def honest (N m ρ h w : ℕ) (X Sig Dg : Type) [Inhabited X] (dig : Sig → (Fin m → X) → Dg) :
    Device N m ρ h 0 (ρ * w) N w X Sig Dg where
  held := ∅
  held_le := by simp
  chip σ _ := chip dig σ
  bounded σ _ O D := by
    obtain ⟨-, h2, -, h4, -⟩ := chip_eq (h := h) dig σ O D
    exact ⟨by rw [h4], h2.le⟩

/-- **Completeness.** The honest device is accepted at every signs, oracle, data and sample; its footprint is the
store, its compute `N` tags. -/
theorem honest_accepts (w : ℕ) {Sig : Type} [Inhabited X] (dig : Sig → (Fin m → X) → Dg) (cTag cEnc : ℕ) (σ : Sig)
    (O : Oracle N ρ h X Dg) (D : Data N m ρ X) {s : ℕ} (I : Fin s → Fin N) :
    (honest N m ρ h w X Sig Dg dig).Accepts dig σ O D I ∧
      (honest N m ρ h w X Sig Dg dig).footprint σ O D = N * m * w ∧
      (honest N m ρ h w X Sig Dg dig).compute cTag cEnc σ O D = N * cTag := by
  obtain ⟨h1, h2, h3, -, h5⟩ := chip_eq (h := h) dig σ O D
  refine ⟨fun i => ?_, ?_, ?_⟩
  · simp only [Device.tags, Device.prog, honest]
    rw [h1]
  · simp only [Device.footprint, Device.prog, honest]
    rw [h3, Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
  · simp only [Device.compute, Device.prog, honest]
    rw [h2, h5, zero_mul, add_zero]

end Accounting.RateSketch.Completeness

/-- **Completeness of the sketch rate audit** (`Accounting.Guarantees.RateSketchHonest`). -/
theorem Accounting.SecurityProofs.RateSketchHonest : Accounting.Guarantees.RateSketchHonest :=
  fun N m ρ h w X Sig Dg _ dig => ⟨Accounting.RateSketch.Completeness.honest N m ρ h w X Sig Dg dig,
    fun cTag cEnc σ O D _ I => Accounting.RateSketch.Completeness.honest_accepts w dig cTag cEnc σ O D I⟩
