import Proofs.Population.Egress
import Proofs.Population.Pointer
import Proofs.Population.Work

/-!
# Leakage is the faulty proof blocks' cone egress

The prover knows the secrets `s` (the weights), and commits an assignment `σ s` with at most `m` faulty proof blocks;
the observer sees its egress. Over a set `Cl` of secrets (for instance those the honest egress cannot tell apart), the
number of egress values the observer can tell apart is the leakage, in bits its logarithm.

* `card_leak_le_cone`, **leakage beyond the honest egress is the faulty proof blocks' cone egress**: in a population
  with single assignment and acyclicity, the egress off the cone of the wrong set `B` is the honest egress, and which
  egress slots lie in the cone is fixed by `B` and the wiring. So the egress takes at most `#honest · Σ_B |V| ^ E(B)`
  values, `E(B)` the egress slots in `B`'s cone (`coneEgress`), and `E(B) ≤ Σ_{u ∈ B} E(u)` (`card_coneEgress_le`).
  `card_leak_le_coneEgress` is the uniform form: at most `m` faulty proof blocks and `E(u) ≤ E_max` give
  `#honest · C(|U|, m) · |V| ^ (m E_max)`, that is `log₂ C(|U|, m) + m · E_max · log₂ |V|` bits beyond the honest egress.
  `card_leak_le_coneEgress_sum` is the weighted form: a draw that bounds `Σ_{u ∈ B} E(u) ≤ M` gives `|V| ^ M` times the
  number of wrong sets within that budget.
  Amplification needs no hypothesis: a correct unit that routes secrets to the egress does so inside the cone, which
  `E` prices. `scale_coneEgress` and `scale_amp` show the bound is attained, and `vmStatic_coneEgress` that a machine's
  step reaches the whole egress, where the bound says nothing.
* **Bounded reach.** With every input fixed, the adversary's choice over the egress at the wrong set `B` (its control
  reach) is at most `min(|W(B)|, |O ∩ cone(B)|)` slots (`card_egress_le_reach`), so with at most `m` faulty proof blocks
  `Reach_ctl(O) ≤ min(m w, Reach_inf(O)) + log C(|U|, ≤ m)` (`card_egressAt_le_reach`). Leakage about the secrets
  beyond the honest egress is at most `Reach_inf(O) + log C(|U|, ≤ m)` (`card_leak_le_reach`), with no `m w` term, since
  inside the cone a value may be any function of the secrets.
* `card_leak_le`, **control times amplification**: at most `K · α` egress values, `K = (∑_{j ≤ m} C(n, j)) |V| ^ (m w)`
  the control count of `Egress.card_egressAt_le` and `α` a bound on every amplification set (`amp`): the egress values
  the rest of the population makes of the secrets once the faulty proof blocks and their writes are fixed. The count
  needs no structure; the structure is what bounds `α`.
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
faulty proof blocks, the egress takes at most `K · α` values over the secrets `Cl`. -/
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

/-- **No amplification outside the egress's cone**: when no secret is upstream of the egress, fixing the faulty proof
blocks and their writes fixes the egress. -/
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

/-! ## Leakage is the faulty proof blocks' cone egress -/

/-- Assignments that hold the same inputs and whose faulty proof blocks lie in `B` agree on every slot outside the
writes of `B`'s cone. -/
theorem eq_off_cone [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} (f : P.Fns V) {B : Finset U}
    {σ τ : S → V} (hin : ∀ t ∈ P.input, σ t = τ t) (hσ : P.wrong f σ ⊆ B) (hτ : P.wrong f τ ⊆ B) {s : S}
    (hs : s ∉ P.writesOf (P.down B)) : σ s = τ s := by
  by_cases hsi : s ∈ P.input
  · exact hin s hsi
  · obtain ⟨v, hv⟩ := hc s hsi
    refine P.eq_off_down hc ha f hin hσ hτ v (fun hvd => hs ?_) s hv
    classical
    simp only [writesOf, mem_biUnion]
    exact ⟨v, hvd, hv⟩

theorem mem_coneEgress [Fintype U] {O : Finset S} {B : Finset U} {s : S} :
    s ∈ P.coneEgress O B ↔ s ∈ O ∧ s ∈ P.writesOf (P.down B) := by
  classical
  exact mem_inter

/-- **`E(B) ≤ Σ_{u ∈ B} E(u)`**: the egress in a set's cone is at most the sum of its units' cone egress. -/
theorem card_coneEgress_le [Fintype U] (O : Finset S) (B : Finset U) :
    (P.coneEgress O B).card ≤ ∑ u ∈ B, (P.coneEgress O {u}).card := by
  classical
  refine (card_le_card fun s hs => ?_).trans card_biUnion_le
  simp only [coneEgress, mem_inter, writesOf, mem_biUnion] at hs ⊢
  obtain ⟨hsO, v, hv, hsv⟩ := hs
  obtain ⟨u, hu, -⟩ := mem_biUnion.1 (P.down_subset_biUnion B hv)
  obtain ⟨u', hu', huv⟩ := (P.mem_down).1 hv
  exact ⟨u', hu', hsO, v, (P.mem_down).2 ⟨u', mem_singleton_self u', huv⟩, hsv⟩

/-- **Leakage is at most the honest egress times the faulty proof blocks' cone egress**: for every index `i ∈ Cl` (a
secret, or a secret with the prover's coins), let the prover commit `σ i`, holding the inputs of the honest assignment
`honest i`, with its faulty proof blocks inside some `B ∈ 𝓑`. The egress then takes at most
`#honest · Σ_{B ∈ 𝓑} |V| ^ E(B)` values: off the cone of `B` it is the honest egress, and the cone's egress slots are
fixed by `B` and the wiring. -/
theorem card_leak_le_cone [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V] [DecidableEq V]
    (f : P.Fns V) (O : Finset S) (𝓑 : Finset (Finset U)) {ι : Type} (Cl : Finset ι) (σ honest : ι → S → V)
    (hin : ∀ i ∈ Cl, ∀ t ∈ P.input, σ i t = honest i t) (hh : ∀ i ∈ Cl, ∀ u, P.Correct f (honest i) u)
    (hB : ∀ i ∈ Cl, ∃ B ∈ 𝓑, P.wrong f (σ i) ⊆ B) :
    (Cl.image fun i (o : O) => σ i o).card ≤
      (Cl.image fun i (o : O) => honest i o).card * ∑ B ∈ 𝓑, Fintype.card V ^ (P.coneEgress O B).card := by
  classical
  let H : Finset (O → V) := Cl.image fun i (o : O) => honest i o
  let φ : (B : Finset U) → (O → V) × (P.coneEgress O B → V) → O → V := fun B p o =>
    if ho : o.1 ∈ P.coneEgress O B then p.2 ⟨o.1, ho⟩ else p.1 o
  have hsub : Cl.image (fun i (o : O) => σ i o) ⊆ 𝓑.biUnion fun B => (H ×ˢ univ).image (φ B) := by
    intro e he
    obtain ⟨i, hi, rfl⟩ := mem_image.1 he
    obtain ⟨B, hB𝓑, hwB⟩ := hB i hi
    refine mem_biUnion.2 ⟨B, hB𝓑, mem_image.2 ⟨((fun o => honest i o), fun t => σ i t),
      mem_product.2 ⟨mem_image_of_mem _ hi, mem_univ _⟩, funext fun o => ?_⟩⟩
    by_cases ho : o.1 ∈ P.coneEgress O B
    · simp only [φ, ho, ↓reduceDIte]
    · simp only [φ, ho, ↓reduceDIte]
      have hoc : o.1 ∉ P.writesOf (P.down B) := fun h => ho (by simp [coneEgress, o.2, h])
      exact (P.eq_off_cone hc ha f (hin i hi) hwB (by simp [P.wrong_eq_empty (hh i hi)]) hoc).symm
  calc _ ≤ _ := card_le_card hsub
    _ ≤ ∑ B ∈ 𝓑, ((H ×ˢ univ).image (φ B)).card := card_biUnion_le
    _ ≤ ∑ B ∈ 𝓑, H.card * Fintype.card V ^ (P.coneEgress O B).card := by
        refine sum_le_sum fun B _ => card_image_le.trans (le_of_eq ?_)
        rw [card_product, card_univ, Fintype.card_fun, Fintype.card_coe]
    _ = _ := by rw [mul_sum]

/-- **The uniform form**: with at most `m` faulty proof blocks and every unit's cone egress `E(u) ≤ E`, the egress takes
at most `#honest · C(|U|, m) · |V| ^ (m E)` values, so the leakage beyond the honest egress is at most
`log₂ C(|U|, m) + m · E · log₂ |V|` bits. -/
theorem card_leak_le_coneEgress [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    [DecidableEq V] [Nonempty V] (f : P.Fns V) (O : Finset S) {m E : ℕ} (hmU : m ≤ Fintype.card U)
    (hE : ∀ u, (P.coneEgress O {u}).card ≤ E) {ι : Type} (Cl : Finset ι) (σ honest : ι → S → V)
    (hin : ∀ i ∈ Cl, ∀ t ∈ P.input, σ i t = honest i t) (hh : ∀ i ∈ Cl, ∀ u, P.Correct f (honest i) u)
    (hm : ∀ i ∈ Cl, (P.wrong f (σ i)).card ≤ m) :
    (Cl.image fun i (o : O) => σ i o).card ≤
      (Cl.image fun i (o : O) => honest i o).card * (Fintype.card U).choose m * Fintype.card V ^ (m * E) := by
  classical
  have hB : ∀ i ∈ Cl, ∃ B ∈ powersetCard m (univ : Finset U), P.wrong f (σ i) ⊆ B := by
    intro i hi
    obtain ⟨B, hsub, hcard⟩ := Finset.exists_superset_card_eq (hm i hi) hmU
    exact ⟨B, mem_powersetCard.2 ⟨subset_univ _, hcard⟩, hsub⟩
  refine (P.card_leak_le_cone hc ha f O _ Cl σ honest hin hh hB).trans ?_
  rw [Nat.mul_assoc]
  refine Nat.mul_le_mul_left _ ?_
  calc ∑ B ∈ powersetCard m (univ : Finset U), Fintype.card V ^ (P.coneEgress O B).card
      ≤ ∑ _B ∈ powersetCard m (univ : Finset U), Fintype.card V ^ (m * E) := by
        refine sum_le_sum fun B hB => Nat.pow_le_pow_right Fintype.card_pos ?_
        obtain ⟨-, hBm⟩ := mem_powersetCard.1 hB
        calc (P.coneEgress O B).card ≤ ∑ u ∈ B, (P.coneEgress O {u}).card := P.card_coneEgress_le O B
          _ ≤ ∑ _u ∈ B, E := sum_le_sum fun u _ => hE u
          _ = m * E := by simp [hBm]
    _ = (Fintype.card U).choose m * Fintype.card V ^ (m * E) := by
        rw [sum_const, card_powersetCard, card_univ]
        simp

/-- **Weighted by cone egress**: when the draw bounds the faulty proof blocks' total cone egress `Σ_{u ∈ B} E(u) ≤ M`
instead of their number, the egress takes at most `#honest · #𝓑 · |V| ^ M` values, `𝓑` the wrong sets within that
budget. -/
theorem card_leak_le_coneEgress_sum [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    [DecidableEq V] [Nonempty V] (f : P.Fns V) (O : Finset S) (M : ℕ) {ι : Type} (Cl : Finset ι)
    (σ honest : ι → S → V) (hin : ∀ i ∈ Cl, ∀ t ∈ P.input, σ i t = honest i t)
    (hh : ∀ i ∈ Cl, ∀ u, P.Correct f (honest i) u)
    (hM : ∀ i ∈ Cl, ∑ u ∈ P.wrong f (σ i), (P.coneEgress O {u}).card ≤ M) :
    (Cl.image fun i (o : O) => σ i o).card ≤ (Cl.image fun i (o : O) => honest i o).card *
      ((univ : Finset (Finset U)).filter fun B => ∑ u ∈ B, (P.coneEgress O {u}).card ≤ M).card *
        Fintype.card V ^ M := by
  classical
  refine (P.card_leak_le_cone hc ha f O ((univ : Finset (Finset U)).filter fun B =>
    ∑ u ∈ B, (P.coneEgress O {u}).card ≤ M) Cl σ honest hin hh fun i hi =>
      ⟨_, mem_filter.2 ⟨mem_univ _, hM i hi⟩, Subset.refl _⟩).trans ?_
  rw [Nat.mul_assoc]
  refine Nat.mul_le_mul_left _ ((sum_le_sum fun B hB => Nat.pow_le_pow_right Fintype.card_pos
    ((P.card_coneEgress_le O B).trans (mem_filter.1 hB).2)).trans (le_of_eq ?_))
  rw [sum_const]
  simp

/-! ## Bounded reach

With every input fixed (the weights included), `egress f a ∅ O B` is what the adversary can choose over `O` at the wrong
set `B`: its control reach is the logarithm of its size. The influence reach is `|O ∩ cone(B)|`, the slots of `O` the
faulty proof blocks can touch at all (`coneEgress`). -/

/-- **Control reach is at most influence reach**: with the inputs fixed, the egress at the wrong set `B` takes at most
`|V| ^ |O ∩ cone(B)|` values. -/
theorem card_egress_le_coneEgress [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    (f : P.Fns V) (a : S → V) (O : Finset S) (B : Finset U) :
    (P.egress f a ∅ O B).card ≤ Fintype.card V ^ (P.coneEgress O B).card := by
  classical
  have key : ∀ e ∈ P.egress f a ∅ O B,
      ∃ σ : S → V, (P.Holds a ∅ σ ∧ P.wrong f σ ⊆ B) ∧ (fun o : O => σ o) = e := by
    intro e he
    simpa only [egress, mem_image, mem_filter, mem_univ, true_and] using he
  let g : P.egress f a ∅ O B → (P.coneEgress O B → V) := fun e t =>
    e.1 ⟨t.1, (P.mem_coneEgress.1 t.2).1⟩
  have hg : Function.Injective g := by
    intro e e' h
    obtain ⟨⟨hσa, hσw⟩, hσe⟩ := Classical.choose_spec (key e.1 e.2)
    obtain ⟨⟨hτa, hτw⟩, hτe⟩ := Classical.choose_spec (key e'.1 e'.2)
    apply Subtype.ext
    funext o
    by_cases ho : o.1 ∈ P.coneEgress O B
    · exact congrFun h ⟨o.1, ho⟩
    · have hoc : o.1 ∉ P.writesOf (P.down B) := fun h' => ho (P.mem_coneEgress.2 ⟨o.2, h'⟩)
      rw [← hσe, ← hτe]
      exact P.eq_off_cone hc ha f (fun t ht => (hσa t ht (by simp)).trans (hτa t ht (by simp)).symm) hσw hτw hoc
  calc (P.egress f a ∅ O B).card = Fintype.card (P.egress f a ∅ O B) := (Fintype.card_coe _).symm
    _ ≤ Fintype.card (P.coneEgress O B → V) := Fintype.card_le_of_injective g hg
    _ = Fintype.card V ^ (P.coneEgress O B).card := by rw [Fintype.card_fun, Fintype.card_coe]

/-- **Control reach at a wrong set**: `min(|W(B)|, |O ∩ cone(B)|)` slots' worth of choice. -/
theorem card_egress_le_reach [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    (f : P.Fns V) (a : S → V) (O : Finset S) (B : Finset U) :
    (P.egress f a ∅ O B).card ≤ Fintype.card V ^ min (P.writesOf B).card (P.coneEgress O B).card := by
  rcases le_total (P.writesOf B).card (P.coneEgress O B).card with h | h
  · rw [min_eq_left h]
    exact P.card_egress_le hc ha f a (by simp)
  · rw [min_eq_right h]
    exact P.card_egress_le_coneEgress hc ha f a O B

/-- **Control reach with at most `m` faulty proof blocks**: with the inputs fixed, every unit writing at most `w` slots
and every wrong set of at most `m` units touching at most `R` slots of `O`, the egress takes at most
`(∑_{j ≤ m} C(|U|, j)) · |V| ^ min(m w, R)` values: `Reach_ctl(O) ≤ min(m w, Reach_inf(O)) + log C(|U|, ≤ m)`. This
bounds what any information, not only the secrets, can put on the egress. -/
theorem card_egressAt_le_reach [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    [Nonempty V] (f : P.Fns V) (a : S → V) (O : Finset S) (m w R : ℕ) (hw : ∀ u, (P.writes u).card ≤ w)
    (hR : ∀ B : Finset U, B.card ≤ m → (P.coneEgress O B).card ≤ R) :
    (P.egressAt f a ∅ O m).card ≤
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ min (m * w) R := by
  classical
  let small : Finset (Finset U) := (range (m + 1)).biUnion fun j => powersetCard j univ
  have hsub : P.egressAt f a ∅ O m ⊆ small.biUnion fun B => P.egress f a ∅ O B := by
    intro e he
    simp only [egressAt, mem_image, mem_filter, mem_univ, true_and] at he
    obtain ⟨σ, ⟨hσa, hσm⟩, rfl⟩ := he
    simp only [mem_biUnion]
    refine ⟨P.wrong f σ, ?_, ?_⟩
    · simp only [small, mem_biUnion, mem_range, mem_powersetCard]
      exact ⟨_, Nat.lt_succ_of_le hσm, subset_univ _, rfl⟩
    · simp only [egress, mem_image, mem_filter, mem_univ, true_and]
      exact ⟨σ, ⟨hσa, le_rfl⟩, rfl⟩
  have hB : ∀ B ∈ small, (P.egress f a ∅ O B).card ≤ Fintype.card V ^ min (m * w) R := by
    intro B hB
    simp only [small, mem_biUnion, mem_range, mem_powersetCard] at hB
    obtain ⟨j, hj, -, hBj⟩ := hB
    have hBm : B.card ≤ m := by omega
    refine (P.card_egress_le_reach hc ha f a O B).trans (Nat.pow_le_pow_right Fintype.card_pos ?_)
    refine min_le_min ?_ (hR B hBm)
    calc (P.writesOf B).card ≤ ∑ u ∈ B, (P.writes u).card := P.card_writesOf_le B
      _ ≤ ∑ _u ∈ B, w := sum_le_sum fun u _ => hw u
      _ = B.card * w := by simp
      _ ≤ m * w := Nat.mul_le_mul_right _ hBm
  have hsmall : small.card ≤ ∑ j ∈ range (m + 1), (Fintype.card U).choose j := by
    refine card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => ?_))
    rw [card_powersetCard, card_univ]
  calc (P.egressAt f a ∅ O m).card ≤ (small.biUnion fun B => P.egress f a ∅ O B).card := card_le_card hsub
    _ ≤ ∑ B ∈ small, (P.egress f a ∅ O B).card := card_biUnion_le
    _ ≤ ∑ _B ∈ small, Fintype.card V ^ min (m * w) R := sum_le_sum hB
    _ = small.card * Fintype.card V ^ min (m * w) R := by simp
    _ ≤ _ := Nat.mul_le_mul_right _ hsmall

/-- **Leakage about the secrets is at most influence reach**: with at most `m` faulty proof blocks, each wrong set
touching at most `R` slots of `O`, the egress takes at most `#honest · (∑_{j ≤ m} C(|U|, j)) · |V| ^ R` values, so the
leakage beyond the honest egress is at most `Reach_inf(O) + log C(|U|, ≤ m)`. Inside the cone a value may be any
function of the secrets (`scale_amp`), so the faulty proof blocks' own width `m w` does not bound it. -/
theorem card_leak_le_reach [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V] [DecidableEq V]
    [Nonempty V] (f : P.Fns V) (O : Finset S) (m R : ℕ)
    (hR : ∀ B : Finset U, B.card ≤ m → (P.coneEgress O B).card ≤ R) {ι : Type} (Cl : Finset ι)
    (σ honest : ι → S → V) (hin : ∀ i ∈ Cl, ∀ t ∈ P.input, σ i t = honest i t)
    (hh : ∀ i ∈ Cl, ∀ u, P.Correct f (honest i) u) (hm : ∀ i ∈ Cl, (P.wrong f (σ i)).card ≤ m) :
    (Cl.image fun i (o : O) => σ i o).card ≤ (Cl.image fun i (o : O) => honest i o).card *
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ R := by
  classical
  let small : Finset (Finset U) := (range (m + 1)).biUnion fun j => powersetCard j univ
  have hB : ∀ i ∈ Cl, ∃ B ∈ small, P.wrong f (σ i) ⊆ B := fun i hi => ⟨_, by
    simp only [small, mem_biUnion, mem_range, mem_powersetCard]
    exact ⟨_, Nat.lt_succ_of_le (hm i hi), subset_univ _, rfl⟩, Subset.refl _⟩
  refine (P.card_leak_le_cone hc ha f O small Cl σ honest hin hh hB).trans ?_
  rw [Nat.mul_assoc]
  refine Nat.mul_le_mul_left _ ?_
  calc ∑ B ∈ small, Fintype.card V ^ (P.coneEgress O B).card ≤ ∑ _B ∈ small, Fintype.card V ^ R := by
        refine sum_le_sum fun B hB => Nat.pow_le_pow_right Fintype.card_pos (hR B ?_)
        simp only [small, mem_biUnion, mem_range, mem_powersetCard] at hB
        obtain ⟨j, hj, -, hBj⟩ := hB
        omega
    _ = small.card * Fintype.card V ^ R := by simp
    _ ≤ _ := Nat.mul_le_mul_right _ (card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => by
        rw [card_powersetCard, card_univ])))

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

/-- **The cone egress prices the amplification**: the scalar unit's cone holds the whole egress, so
`card_leak_le_cone` allows `2 ^ N` egress values at the wrong set `{none}`, which `scale_amp` attains. -/
theorem scale_coneEgress (N : ℕ) :
    (scale N).coneEgress (univ.filter fun s => s.isOut) {none} = univ.filter fun s => s.isOut := by
  ext s
  rw [Dataflow.mem_coneEgress]
  refine ⟨And.left, fun hs => ⟨hs, ?_⟩⟩
  cases s with
  | ptr => simp at hs
  | zero i => simp at hs
  | secret i => simp at hs
  | out i =>
    simp only [Dataflow.writesOf, mem_biUnion]
    exact ⟨some i, (scale N).mem_down.2 ⟨none, mem_singleton_self _,
      .single ⟨.ptr, by simp [scale], by simp [scale]⟩⟩, by simp [scale]⟩

/-- **A machine's step reaches the whole egress**: in the pointer machine declared statically, the pointer step's
cone holds every output word, so `E_max = |E|` and the cone bound says nothing. -/
theorem vmStatic_coneEgress (N : ℕ) :
    (vmStatic N).coneEgress (univ.filter fun s => s.isOut) {none} = univ.filter fun s => s.isOut := by
  ext s
  rw [Dataflow.mem_coneEgress]
  refine ⟨And.left, fun hs => ⟨hs, ?_⟩⟩
  cases s with
  | ptr => simp at hs
  | zero i => simp at hs
  | secret i => simp at hs
  | out i =>
    simp only [Dataflow.writesOf, mem_biUnion]
    exact ⟨some i, (vmStatic N).mem_down.2 ⟨none, mem_singleton_self _,
      .single ⟨.ptr, by simp [vmStatic], by simp [vmStatic]⟩⟩, by simp [vmStatic]⟩

end Population
