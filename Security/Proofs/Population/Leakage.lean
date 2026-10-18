import Proofs.Population.Egress
import Proofs.Population.Pointer

/-!
# Leakage is control plus amplification

The prover knows the secrets `s` (the weights), and commits an assignment `σ s` with at most `m` wrong units; the
observer sees its egress. Over a set `Cl` of secrets (for instance those the honest egress cannot tell apart), the
number of egress values the observer can tell apart is the leakage, in bits its logarithm.

* `card_leak_le`, **control times amplification**: at most `K · α` egress values, `K = (∑_{j ≤ m} C(n, j)) |V| ^ (m w)`
  the control count of `Egress.card_egressAt_le` and `α` a bound on every amplification set (`amp`): the egress values
  the rest of the population makes of the secrets once the wrong units and their writes are fixed. The count needs no
  structure; the structure is what bounds `α`.
* `card_amp_le_one`: in a population with single assignment and acyclicity, when no secret is upstream of the egress,
  every amplification set has at most one value, so `card_leak_le_of_not_up`: the leakage is at most the control.
* `scale_amp`, **fixed arithmetic amplifies**: a checked population whose unit `i` multiplies secret word `i` by a
  scalar, reading both at fixed slots. Its honest egress is zero whatever the secrets (`scale_honest`), yet with the
  one-bit scalar unit wrong the amplification set over all secrets has `2 ^ N` values: one wrong bit of control, `N`
  secret bits out. Reading weights only at fixed indices by fixed arithmetic does not bound amplification.
-/

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

theorem card_writesOf_le (B : Finset U) : (P.writesOf B).card ≤ ∑ u ∈ B, (P.writes u).card := by
  classical
  exact card_biUnion_le

/-- **Leakage is at most control times amplification**: whatever the prover commits for each secret, with at most `m`
wrong units, the egress takes at most `K · α` values over the secrets `Cl`. -/
theorem card_leak_le [Fintype U] [Fintype S] [DecidableEq S] {V : Type} [Fintype V] [DecidableEq V] [Nonempty V]
    (f : P.Fns V) (a : S → V) (X O : Finset S) (Cl : Finset (S → V)) (m w α : ℕ) (hw : ∀ u, (P.writes u).card ≤ w)
    (hα : ∀ B : Finset U, B.card ≤ m → ∀ b, (P.amp f a X O Cl B b).card ≤ α)
    (σ : (S → V) → S → V) (hσ : ∀ s, ∀ t ∈ P.input, σ s t = if t ∈ X then s t else a t)
    (hm : ∀ s, (P.wrong f (σ s)).card ≤ m) :
    (Cl.image fun s (o : O) => σ s o).card ≤
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ (m * w) * α := by
  classical
  let small : Finset (Finset U) := (range (m + 1)).biUnion fun j => powersetCard j univ
  have hsub : Cl.image (fun s (o : O) => σ s o) ⊆
      small.biUnion fun B => (univ : Finset (P.writesOf B → V)).biUnion fun b => P.amp f a X O Cl B b := by
    intro e he
    obtain ⟨s, hs, rfl⟩ := mem_image.1 he
    refine mem_biUnion.2 ⟨P.wrong f (σ s), ?_, mem_biUnion.2 ⟨fun t => σ s t, mem_univ _, ?_⟩⟩
    · simp only [small, mem_biUnion, mem_range, mem_powersetCard]
      exact ⟨_, Nat.lt_succ_of_le (hm s), subset_univ _, rfl⟩
    · rw [amp]
      refine mem_image.2 ⟨(s, σ s), mem_filter.2 ⟨mem_product.2 ⟨hs, mem_univ _⟩, ?_⟩, rfl⟩
      exact ⟨fun t ht => (hσ s t ht).trans (if_congr Iff.rfl rfl rfl), Subset.refl _, fun t => rfl⟩
  have hB : ∀ B ∈ small, ((univ : Finset (P.writesOf B → V)).biUnion fun b => P.amp f a X O Cl B b).card ≤
      Fintype.card V ^ (m * w) * α := by
    intro B hB
    simp only [small, mem_biUnion, mem_range, mem_powersetCard] at hB
    obtain ⟨j, hj, -, hBj⟩ := hB
    have hBm : B.card ≤ m := by omega
    calc _ ≤ ∑ b : P.writesOf B → V, (P.amp f a X O Cl B b).card := card_biUnion_le
      _ ≤ ∑ _b : P.writesOf B → V, α := sum_le_sum fun b _ => hα B hBm b
      _ = Fintype.card V ^ (P.writesOf B).card * α := by simp
      _ ≤ Fintype.card V ^ (m * w) * α := by
        refine Nat.mul_le_mul_right _ (Nat.pow_le_pow_right Fintype.card_pos ?_)
        calc (P.writesOf B).card ≤ ∑ u ∈ B, (P.writes u).card := P.card_writesOf_le B
          _ ≤ ∑ _u ∈ B, w := sum_le_sum fun u _ => hw u
          _ = B.card * w := by simp
          _ ≤ m * w := Nat.mul_le_mul_right _ hBm
  have hsmall : small.card ≤ ∑ j ∈ range (m + 1), (Fintype.card U).choose j := by
    refine card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => ?_))
    rw [card_powersetCard, card_univ]
  calc _ ≤ _ := card_le_card hsub
    _ ≤ ∑ B ∈ small, ((univ : Finset (P.writesOf B → V)).biUnion fun b => P.amp f a X O Cl B b).card :=
        card_biUnion_le
    _ ≤ ∑ _B ∈ small, Fintype.card V ^ (m * w) * α := sum_le_sum hB
    _ = small.card * (Fintype.card V ^ (m * w) * α) := by simp
    _ ≤ (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ (m * w) * α := by
        rw [← Nat.mul_assoc]
        exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hsmall)

/-- **No amplification outside the egress's cone**: when no secret is upstream of the egress, fixing the wrong units
and their writes fixes the egress. -/
theorem card_amp_le_one [Fintype U] [Fintype S] [DecidableEq S] (hc : P.Covered) (ha : P.Acyclic) {V : Type}
    [Fintype V] [DecidableEq V] (f : P.Fns V) (a : S → V) {X O : Finset S} (hX : ∀ s ∈ X, s ∉ P.up O)
    (Cl : Finset (S → V)) (B : Finset U) (b : P.writesOf B → V) : (P.amp f a X O Cl B b).card ≤ 1 := by
  classical
  refine card_le_one.2 fun e he e' he' => ?_
  rw [amp] at he he'
  obtain ⟨p, hp, rfl⟩ := mem_image.1 he
  obtain ⟨q, hq, rfl⟩ := mem_image.1 he'
  obtain ⟨-, hpi, hpw, hpb⟩ := mem_filter.1 hp
  obtain ⟨-, hqi, hqw, hqb⟩ := mem_filter.1 hq
  funext o
  have hold : ∀ r : (S → V) × (S → V), (∀ t ∈ P.input, r.2 t = if t ∈ X then r.1 t else a t) →
      P.Holds a X r.2 := fun r hr t ht htX => by simp [hr t ht, htX]
  refine P.eq_on_up hc ha f hX (hold p hpi) (hold q hqi) hpw hqw (fun u hu t ht => ?_) o (P.mem_up_self o.2)
  have htB : t ∈ P.writesOf B := by
    simp only [writesOf, mem_biUnion]
    exact ⟨u, hu, ht⟩
  exact (hpb ⟨t, htB⟩).trans (hqb ⟨t, htB⟩).symm

/-- **With the secrets outside the egress's cone, leakage is at most the control.** -/
theorem card_leak_le_of_not_up [Fintype U] [Fintype S] [DecidableEq S] (hc : P.Covered) (ha : P.Acyclic) {V : Type}
    [Fintype V] [DecidableEq V] [Nonempty V] (f : P.Fns V) (a : S → V) {X O : Finset S} (hX : ∀ s ∈ X, s ∉ P.up O)
    (Cl : Finset (S → V)) (m w : ℕ) (hw : ∀ u, (P.writes u).card ≤ w) (σ : (S → V) → S → V)
    (hσ : ∀ s, ∀ t ∈ P.input, σ s t = if t ∈ X then s t else a t) (hm : ∀ s, (P.wrong f (σ s)).card ≤ m) :
    (Cl.image fun s (o : O) => σ s o).card ≤
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ (m * w) := by
  simpa using P.card_leak_le f a X O Cl m w 1 hw (fun B _ b => P.card_amp_le_one hc ha f a hX Cl B b) σ hσ hm

end Population.Dataflow

namespace Population

open Finset

/-! ## Fixed arithmetic amplifies -/

/-- Secret word `i` times a scalar: unit `none` writes the scalar (honestly zero) and unit `some i` reads the scalar and
secret word `i`, at fixed slots, and writes their product to output word `i`. -/
def scale (N : ℕ) : Dataflow (Option (Fin N)) (Slot N) where
  input := univ.filter fun s => s.isInput
  reads
    | none => ∅
    | some i => {.ptr, .secret i}
  writes
    | none => {.ptr}
    | some i => {.out i}

/-- The units' functions: the scalar is zero, and output word `i` is secret word `i` times the scalar. -/
def scaleFns (N : ℕ) : (scale N).Fns Bool
  | none => fun _ _ => false
  | some i => fun r _ => r ⟨.secret i, by simp [scale]⟩ && r ⟨.ptr, by simp [scale]⟩

theorem scale_checked (N : ℕ) : (scale N).Checked := by
  refine ⟨fun s => ?_, fun u h => ?_⟩
  · cases s with
    | ptr => exact ⟨fun h => by simp [scale] at h, fun _ => ⟨none, by simp [scale],
        fun u hu => by cases u <;> simp_all [scale]⟩⟩
    | zero i => exact ⟨fun _ u => by cases u <;> simp [scale], fun h => by simp [scale] at h⟩
    | secret i => exact ⟨fun _ u => by cases u <;> simp [scale], fun h => by simp [scale] at h⟩
    | out i => exact ⟨fun h => by simp [scale] at h, fun _ => ⟨some i, by simp [scale],
        fun u hu => by cases u <;> simp_all [scale]⟩⟩
  · obtain ⟨b, -, hbu⟩ := Relation.TransGen.tail'_iff.1 h
    obtain ⟨c, huc, -⟩ := Relation.TransGen.head'_iff.1 h
    obtain ⟨s, hs, hsr⟩ := hbu
    obtain ⟨t, ht, htr⟩ := huc
    cases u with
    | none => simp [scale] at hsr
    | some i =>
      have : t = .out i := by simpa [scale] using ht
      subst this
      cases c <;> simp [scale] at htr

/-- **The honest egress is zero, whatever the secrets.** -/
theorem scale_honest (N : ℕ) {σ : Slot N → Bool} (hσ : ∀ u, (scale N).Correct (scaleFns N) σ u) (i : Fin N) :
    σ (.out i) = false := by
  have hp : σ .ptr = false := hσ none ⟨.ptr, by simp [scale]⟩
  have := hσ (some i) ⟨.out i, by simp [scale]⟩
  simpa [scaleFns, hp] using this

/-- **One wrong bit of control, `N` secret bits out**: with the scalar unit wrong and the scalar one, the egress over
all secrets takes `2 ^ N` values. -/
theorem scale_amp (N : ℕ) :
    2 ^ N ≤ ((scale N).amp (scaleFns N) (fun _ => false) (univ.filter fun s => s.isSecret)
      (univ.filter fun s => s.isOut) univ {none} fun _ => true).card := by
  let O : Finset (Slot N) := univ.filter fun s => s.isOut
  let sec : (Fin N → Bool) → Slot N → Bool := fun x s => match s with
    | .secret i => x i
    | _ => false
  let σ : (Fin N → Bool) → Slot N → Bool := fun x s => match s with
    | .ptr => true
    | .zero _ => false
    | .secret i => x i
    | .out i => x i
  have hmem : ∀ x, (fun o : O => σ x o) ∈ (scale N).amp (scaleFns N) (fun _ => false)
      (univ.filter fun s => s.isSecret) O univ {none} fun _ => true := by
    intro x
    simp only [Dataflow.amp, mem_image, mem_filter, mem_product, mem_univ, and_true, true_and, Prod.exists]
    refine ⟨sec x, σ x, ⟨?_, ?_, ?_⟩, rfl⟩
    · intro t ht
      cases t <;> simp_all [scale, σ, sec]
    · intro u hu
      simp only [Dataflow.wrong, mem_filter, mem_univ, true_and] at hu
      cases u with
      | none => simp
      | some i =>
        exfalso
        refine hu fun ⟨t, ht⟩ => ?_
        have : t = .out i := by simpa [scale] using ht
        subst this
        simp [scaleFns, σ]
    · rintro ⟨t, ht⟩
      have : t = .ptr := by simpa [Dataflow.writesOf, scale] using ht
      subst this
      rfl
  have hinj : Function.Injective fun x (o : O) => σ x o := by
    intro x y h
    funext i
    have := congrFun h ⟨.out i, by simp [O]⟩
    simpa [σ] using this
  have := card_le_card_of_injOn _ (s := univ) (fun x _ => hmem x) hinj.injOn
  simpa using this

end Population
