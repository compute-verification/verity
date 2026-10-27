import Proofs.Population.Determinacy
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Finset.Powerset

/-!
# Non-exfiltration: what the egress can show with at most `m` faulty proof blocks

The egress slots `O` are what leaves; the secret slots `X` are inputs the egress must not reveal beyond what the
faulty proof blocks write. The count is over every assignment that holds the public inputs, the secrets free, and whose
faulty proof blocks lie in `B` (or number at most `m`): the prover's whole freedom, whatever it knows.

* `eq_on_up`, **the egress is a function of the faulty proof blocks' writes**: in a covered acyclic population whose
  secret slots lie outside the egress's upstream cone, two such assignments that agree on what `B` writes agree on
  everything upstream of the egress, the egress included. With `X = ∅` this is robust determinacy, read on the cone.
* `card_egress_le`: so at the wrong set `B` the egress takes at most `|V| ^ |writes B|` values, over all secrets.
* `card_egressAt_le`: with at most `m` faulty proof blocks, each writing at most `w` slots, at most
  `(∑_{j ≤ m} C(|U|, j)) · |V| ^ (m w)` values: `m w log₂|V|` bits of value, plus the location term, the choice of which
  units are wrong.
* `flags_egressAt`, **the location term is needed**: `n` checked units that each write one bit, honestly `false`,
  show `n + 1` egress values with one faulty proof block, more than `2 ^ (m w) = 2` once `n ≥ 2`.
* `loop_egress`, **acyclicity is needed**: one unit that copies `k` slots onto themselves (single assignment, one
  cycle) shows `2 ^ k` egress values with no faulty proof block at all.
-/

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

theorem mem_up_self {O : Finset S} {o : S} (h : o ∈ O) : o ∈ P.up O := ⟨o, h, .refl⟩

theorem mem_up_of_feeds {O : Finset S} {r t : S} (h : P.Feeds r t) (ht : t ∈ P.up O) : r ∈ P.up O := by
  obtain ⟨o, ho, hto⟩ := ht
  exact ⟨o, ho, Relation.ReflTransGen.head h hto⟩

/-- **The egress is a function of the faulty proof blocks' writes**, when the secrets lie outside the egress's upstream
cone. -/
theorem eq_on_up [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} (f : P.Fns V) {a : S → V}
    {X O : Finset S} {B : Finset U} (hX : ∀ s ∈ X, s ∉ P.up O) {σ τ : S → V} (hσa : P.Holds a X σ)
    (hτa : P.Holds a X τ) (hσ : P.wrong f σ ⊆ B) (hτ : P.wrong f τ ⊆ B)
    (hB : ∀ u ∈ B, ∀ s ∈ P.writes u, σ s = τ s) : ∀ s ∈ P.up O, σ s = τ s := by
  have hin : ∀ s ∈ P.up O, s ∈ P.input → σ s = τ s := fun s hs hsi => by
    have hsX : s ∉ X := fun h => hX s h hs
    exact (hσa s hsi hsX).trans (hτa s hsi hsX).symm
  have key : ∀ u, ∀ t ∈ P.writes u, t ∈ P.up O → σ t = τ t := by
    intro u
    induction u using (P.wellFounded_dep ha).induction with
    | _ u ih =>
      intro t ht htO
      by_cases huB : u ∈ B
      · exact hB u huB t ht
      · have cσ := P.correct_of_not_mem_wrong (f := f) (σ := σ) fun h => huB (hσ h)
        have cτ := P.correct_of_not_mem_wrong (f := f) (σ := τ) fun h => huB (hτ h)
        calc σ t = f u (fun r => σ r) ⟨t, ht⟩ := cσ ⟨t, ht⟩
          _ = f u (fun r => τ r) ⟨t, ht⟩ := by
              congr 1
              funext ⟨r, hr⟩
              have hrO : r ∈ P.up O := P.mem_up_of_feeds ⟨u, hr, ht⟩ htO
              by_cases hri : r ∈ P.input
              · exact hin r hrO hri
              · obtain ⟨v, hv⟩ := hc r hri
                exact ih v ⟨r, hv, hr⟩ r hv hrO
          _ = τ t := (cτ ⟨t, ht⟩).symm
  intro s hs
  by_cases hsi : s ∈ P.input
  · exact hin s hs hsi
  · obtain ⟨v, hv⟩ := hc s hsi
    exact key v s hv hs

/-- **The capacity at a wrong set**: at most `|V| ^ |writes B|` egress values, over all secrets outside the egress's
upstream cone. -/
theorem card_egress_le [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    (f : P.Fns V) (a : S → V) {X O : Finset S} {B : Finset U} (hX : ∀ s ∈ X, s ∉ P.up O) :
    (P.egress f a X O B).card ≤ Fintype.card V ^ (P.writesOf B).card := by
  classical
  have key : ∀ e ∈ P.egress f a X O B,
      ∃ σ : S → V, (P.Holds a X σ ∧ P.wrong f σ ⊆ B) ∧ (fun o : O => σ o) = e := by
    intro e he
    simpa only [egress, mem_image, mem_filter, mem_univ, true_and] using he
  let g : P.egress f a X O B → (P.writesOf B → V) := fun e s => Classical.choose (key e.1 e.2) s
  have hg : Function.Injective g := by
    intro e e' h
    obtain ⟨⟨hσa, hσw⟩, hσe⟩ := Classical.choose_spec (key e.1 e.2)
    obtain ⟨⟨hτa, hτw⟩, hτe⟩ := Classical.choose_spec (key e'.1 e'.2)
    apply Subtype.ext
    rw [← hσe, ← hτe]
    funext o
    refine P.eq_on_up hc ha f hX hσa hτa hσw hτw (fun u hu s hs => ?_) o (P.mem_up_self o.2)
    have hsB : s ∈ P.writesOf B := by
      simp only [writesOf, mem_biUnion]
      exact ⟨u, hu, hs⟩
    exact congrFun h ⟨s, hsB⟩
  calc (P.egress f a X O B).card = Fintype.card (P.egress f a X O B) := (Fintype.card_coe _).symm
    _ ≤ Fintype.card (P.writesOf B → V) := Fintype.card_le_of_injective g hg
    _ = Fintype.card V ^ (P.writesOf B).card := by rw [Fintype.card_fun, Fintype.card_coe]

/-- **The capacity with at most `m` faulty proof blocks**: `m w log₂|V|` bits of value and the location term. -/
theorem card_egressAt_le [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    [Nonempty V] (f : P.Fns V) (a : S → V) {X O : Finset S} (hX : ∀ s ∈ X, s ∉ P.up O) (m w : ℕ)
    (hw : ∀ u, (P.writes u).card ≤ w) :
    (P.egressAt f a X O m).card ≤ (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ (m * w) := by
  classical
  let small : Finset (Finset U) := (range (m + 1)).biUnion fun j => powersetCard j univ
  have hsub : P.egressAt f a X O m ⊆ small.biUnion fun B => P.egress f a X O B := by
    intro e he
    simp only [egressAt, mem_image, mem_filter, mem_univ, true_and] at he
    obtain ⟨σ, ⟨hσa, hσm⟩, rfl⟩ := he
    simp only [mem_biUnion]
    refine ⟨P.wrong f σ, ?_, ?_⟩
    · simp only [small, mem_biUnion, mem_range, mem_powersetCard]
      exact ⟨_, Nat.lt_succ_of_le hσm, subset_univ _, rfl⟩
    · simp only [egress, mem_image, mem_filter, mem_univ, true_and]
      exact ⟨σ, ⟨hσa, le_rfl⟩, rfl⟩
  have hB : ∀ B ∈ small, (P.egress f a X O B).card ≤ Fintype.card V ^ (m * w) := by
    intro B hB
    simp only [small, mem_biUnion, mem_range, mem_powersetCard] at hB
    obtain ⟨j, hj, -, hBj⟩ := hB
    refine (P.card_egress_le hc ha f a hX).trans (Nat.pow_le_pow_right Fintype.card_pos ?_)
    calc (P.writesOf B).card ≤ ∑ u ∈ B, (P.writes u).card := card_biUnion_le
      _ ≤ ∑ _u ∈ B, w := sum_le_sum fun u _ => hw u
      _ = B.card * w := by simp
      _ ≤ m * w := Nat.mul_le_mul_right _ (by omega)
  have hsmall : small.card ≤ ∑ j ∈ range (m + 1), (Fintype.card U).choose j := by
    refine card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => ?_))
    rw [card_powersetCard, card_univ]
  calc (P.egressAt f a X O m).card ≤ (small.biUnion fun B => P.egress f a X O B).card := card_le_card hsub
    _ ≤ ∑ B ∈ small, (P.egress f a X O B).card := card_biUnion_le
    _ ≤ ∑ _B ∈ small, Fintype.card V ^ (m * w) := sum_le_sum hB
    _ = small.card * Fintype.card V ^ (m * w) := by simp
    _ ≤ _ := Nat.mul_le_mul_right _ hsmall

/-! ## The location term is needed -/

/-- `n` units, unit `i` writing slot `i` and reading nothing. -/
def flags (n : ℕ) : Dataflow (Fin n) (Fin n) where
  input := ∅
  reads := fun _ => ∅
  writes := fun i => {i}

/-- Every unit honestly writes `false`. -/
def flagsFns (n : ℕ) : (flags n).Fns Bool := fun _ _ _ => false

theorem flags_checked (n : ℕ) : (flags n).Checked := by
  refine ⟨fun s => ⟨fun h => by simp [flags] at h, fun _ => ⟨s, by simp [flags], fun u hu => ?_⟩⟩, ?_⟩
  · simpa [flags, eq_comm] using hu
  · intro u h
    obtain ⟨b, hb⟩ := Relation.TransGen.head'_iff.1 h
    obtain ⟨s, -, hs⟩ := hb.1
    simp [flags] at hs

/-- With one faulty proof block (writing one bit) a checked population shows `n + 1` egress values. -/
theorem flags_egressAt (n : ℕ) :
    n + 1 ≤ ((flags n).egressAt (flagsFns n) (fun _ => false) ∅ univ 1).card := by
  classical
  let φ : Option (Fin n) → (↥(univ : Finset (Fin n)) → Bool) := fun x o => decide (some o.1 = x)
  have hmem : ∀ x, φ x ∈ (flags n).egressAt (flagsFns n) (fun _ => false) ∅ univ 1 := by
    intro x
    simp only [egressAt, mem_image, mem_filter, mem_univ, true_and]
    refine ⟨fun s => decide (some s = x), ⟨fun s hs => by simp [flags] at hs, ?_⟩, rfl⟩
    have hsub : (flags n).wrong (flagsFns n) (fun s => decide (some s = x)) ⊆ x.toFinset := by
      intro u hu
      simp only [wrong, mem_filter, mem_univ, true_and] at hu
      by_contra hux
      refine hu fun ⟨s, hs⟩ => ?_
      have hsu : s = u := by simpa [flags] using hs
      subst hsu
      simpa [flagsFns, Option.mem_toFinset, eq_comm] using hux
    refine (card_le_card hsub).trans ?_
    cases x <;> simp
  have hinj : Function.Injective φ := by
    intro x y h
    cases x with
    | none =>
      cases y with
      | none => rfl
      | some j =>
        have := congrFun h ⟨j, mem_univ _⟩
        simp [φ] at this
    | some i =>
      have := congrFun h ⟨i, mem_univ _⟩
      simpa [φ, eq_comm] using this
  have := card_le_card_of_injOn φ (s := univ) (fun x _ => hmem x) hinj.injOn
  simpa using this

/-! ## Acyclicity is needed -/

/-- One unit that reads and writes the same `k` slots. -/
def loop (k : ℕ) : Dataflow Unit (Fin k) where
  input := ∅
  reads := fun _ => univ
  writes := fun _ => univ

/-- It copies each slot onto itself. -/
def loopFns (k : ℕ) : (loop k).Fns Bool := fun _ r s => r ⟨s.1, mem_univ _⟩

theorem loop_singleAssignment (k : ℕ) : (loop k).SingleAssignment :=
  fun _ => ⟨fun h => by simp [loop] at h, fun _ => ⟨(), by simp [loop], fun _ _ => rfl⟩⟩

theorem loop_not_acyclic {k : ℕ} (hk : 0 < k) : ¬ (loop k).Acyclic :=
  fun h => h () (.single ⟨⟨0, hk⟩, mem_univ _, mem_univ _⟩)

/-- With no faulty proof block at all, the loop shows every one of the `2 ^ k` egress values. -/
theorem loop_egress (k : ℕ) : 2 ^ k ≤ ((loop k).egress (loopFns k) (fun _ => false) ∅ univ ∅).card := by
  classical
  have hmem : ∀ σ : Fin k → Bool, (fun o : ↥(univ : Finset (Fin k)) => σ o) ∈
      (loop k).egress (loopFns k) (fun _ => false) ∅ univ ∅ := by
    intro σ
    simp only [egress, mem_image, mem_filter, mem_univ, true_and]
    refine ⟨σ, ⟨fun s hs => by simp [loop] at hs, ?_⟩, rfl⟩
    rw [(loop k).wrong_eq_empty fun _ _ => rfl]
  have hinj : Function.Injective fun (σ : Fin k → Bool) (o : ↥(univ : Finset (Fin k))) => σ o := by
    intro σ τ h
    funext s
    exact congrFun h ⟨s, mem_univ _⟩
  have := card_le_card_of_injOn _ (s := univ) (fun σ _ => hmem σ) hinj.injOn
  simpa using this

end Population.Dataflow
