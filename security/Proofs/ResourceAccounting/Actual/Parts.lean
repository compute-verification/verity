module

public import Definitions.ResourceAccounting.Guarantees.RegisteredSite
public import Proofs.ResourceAccounting.Actual.Bridges
import Mathlib.Data.Set.Card
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

/-! `B4`'s leak limit as a sum over the trusted devices' record limits. -/

@[expose] public section

namespace ResourceAccounting

open Verity Finset

theorem seen_record_single_Ico {L : Type*} (R : Actual.Run L) {cert : Finset L} (cells : L → Set ℕ) {y : L}
    (hy : y ∈ cert) (a b : ℕ) :
    R.seen '' ↑((R.record {y} cells).filter fun g => a ≤ R.A.t g ∧ R.A.t g < b) =
      R.seen '' ↑((R.record cert cells).filter fun g => a ≤ R.A.t g ∧ R.A.t g < b) ∩ {p | p.2.1 = y} := by
  ext p
  simp only [Set.mem_image, Finset.coe_filter, Actual.Run.record, Finset.mem_filter, Finset.mem_univ, true_and,
    Set.mem_ofPred_eq, Set.mem_inter_iff, Finset.mem_singleton]
  constructor
  · rintro ⟨g, ⟨⟨hx, hc⟩, ht⟩, rfl⟩
    exact ⟨⟨g, ⟨⟨hx ▸ hy, hc⟩, ht⟩, rfl⟩, hx⟩
  · rintro ⟨⟨g, ⟨⟨-, hc⟩, ht⟩, rfl⟩, hgy⟩
    exact ⟨g, ⟨⟨hgy, hc⟩, ht⟩, rfl⟩

theorem encard_le_two_pow_sum {Z ι : Type*} [Fintype ι] {V : ι → Type*} (Zs : Set Z) (view : ∀ j, Z → V j)
    (dec : (∀ j, V j) → Z) (hdec : ∀ z ∈ Zs, dec (fun j => view j z) = z) (K : ι → ℕ)
    (hK : ∀ j, (view j '' Zs).encard ≤ ((2 ^ K j : ℕ) : ℕ∞)) :
    Zs.encard ≤ ((2 ^ ∑ j, K j : ℕ) : ℕ∞) := by
  have hsub : Zs ⊆ dec '' (Set.pi Set.univ fun j => view j '' Zs) := by
    intro z hz
    exact ⟨fun j => view j z, fun j _ => ⟨z, hz, rfl⟩, hdec z hz⟩
  calc Zs.encard ≤ (dec '' (Set.pi Set.univ fun j => view j '' Zs)).encard := Set.encard_le_encard hsub
    _ ≤ (Set.pi Set.univ fun j => view j '' Zs).encard := Set.encard_image_le _ _
    _ = ∏ j, (view j '' Zs).encard := Set.encard_pi_eq_prod_encard
    _ ≤ ∏ j, ((2 ^ K j : ℕ) : ℕ∞) := Finset.prod_le_prod fun j _ => hK j
    _ = ((∏ j, 2 ^ K j : ℕ) : ℕ∞) := (Nat.cast_prod _ _).symm
    _ = ((2 ^ ∑ j, K j : ℕ) : ℕ∞) := by rw [Finset.prod_pow_eq_pow_sum]

theorem records_encard_le {Ω L : Type*} (S : Ω → Actual.Run L) (cert : Finset L) (record : L → Set ℕ) (a b : ℕ)
    (K : L → ℕ) (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r =>
      (S r).seen '' ↑(((S r).record cert record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard ≤
        ((2 ^ ∑ y ∈ cert, K y : ℕ) : ℕ∞) := by
  refine (encard_le_two_pow_sum (ι := ↥cert) (V := fun _ => Set (ℕ × L × ℕ × ℕ × ℕ))
    (Set.range fun r =>
      (S r).seen '' ↑(((S r).record cert record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b))
    (fun j z => z ∩ {p | p.2.1 = j.1}) (fun f => ⋃ j, f j) ?_ (fun j => K j.1) ?_).trans_eq ?_
  · rintro _ ⟨r, rfl⟩
    ext p
    constructor
    · intro hp
      obtain ⟨j, hpj⟩ := Set.mem_iUnion.1 hp
      exact ((Set.mem_inter_iff _ _ _).1 hpj).1
    · intro hp
      obtain ⟨g, hg, rfl⟩ := (Set.mem_image _ _ _).1 hp
      have hx : (S r).A.x g ∈ cert := by
        have hrec := (Finset.mem_filter.1 (Finset.mem_coe.1 hg)).1
        simp only [Actual.Run.record, Finset.mem_filter, Finset.mem_univ, true_and] at hrec
        exact hrec.1
      exact Set.mem_iUnion.2 ⟨⟨(S r).A.x g, hx⟩, (Set.mem_inter_iff _ _ _).2 ⟨Set.mem_image_of_mem _ hg, rfl⟩⟩
  · intro j
    refine le_of_eq_of_le (congrArg Set.encard ?_) (hK j.1 j.2)
    ext z
    constructor
    · rintro ⟨_, ⟨r, rfl⟩, rfl⟩
      exact ⟨r, seen_record_single_Ico (S r) record j.2 a b⟩
    · rintro ⟨r, rfl⟩
      exact ⟨_, ⟨r, rfl⟩, (seen_record_single_Ico (S r) record j.2 a b).symm⟩
  · rw [Finset.sum_coe_sort cert K]

theorem leak_le_sum_K {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hsec : Secrets S cert X a)
    (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        uncertifiedLinkCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞) := by
  refine (SecurityProofs.B4 S ℓ ends β link 𝒱 cert X spec record bucket jitter a b hcs hdl hT hsec).trans ?_
  refine (mul_le_mul' (records_encard_le S cert record a b K hK) le_rfl).trans_eq ?_
  rw [← Nat.cast_mul, ← pow_add, Nat.add_assoc]

section Sinks

variable {L E : Type*} [Fintype E]

theorem uncertifiedLinkCapacity_union_le [DecidableEq L] (ends : E → L × L) (β : E → ℕ) (cert X Y : Finset L)
    (hXY : Disjoint X Y) :
    uncertifiedLinkCapacity ends β cert (X ∪ Y) ≤
      uncertifiedLinkCapacity ends β (cert ∪ Y) X + uncertifiedLinkCapacity ends β (cert ∪ X) Y := by
  classical
  have hd : ∀ z, z ∈ X → z ∉ Y := fun z h => Finset.disjoint_left.1 hXY h
  unfold uncertifiedLinkCapacity
  refine le_trans ?_ (Nat.le.intro Finset.sum_union_inter)
  refine Finset.sum_le_sum_of_subset ?_
  intro e he
  have ha := hd (ends e).1
  have hb := hd (ends e).2
  simp only [boundary, mem_filter, mem_univ, true_and, mem_union] at he ⊢
  by_cases a1 : (ends e).1 ∈ X <;> by_cases a2 : (ends e).2 ∈ X <;>
    by_cases b1 : (ends e).1 ∈ Y <;> by_cases b2 : (ends e).2 ∈ Y <;> simp_all

end Sinks

/-- A sink adds no `K` and no jitter share, and nothing is asked of what it does. -/
theorem leak_le_sum_K_sinks {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X sink : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hXs : Disjoint X sink)
    (hsec : Secrets S cert (X ∪ sink) a) (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ sink ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        (uncertifiedLinkCapacity ends β (cert ∪ sink) X + uncertifiedLinkCapacity ends β (cert ∪ X) sink) * (b - a)) : ℕ) :
          ℕ∞) := by
  refine (leak_le_sum_K S ℓ ends β link 𝒱 cert (X ∪ sink) spec record bucket jitter a b K hcs hdl hT hsec
    hK).trans ?_
  exact Nat.mono_cast (Nat.pow_le_pow_right Nat.two_pos (Nat.add_le_add_left
    (Nat.mul_le_mul_right _ (uncertifiedLinkCapacity_union_le ends β cert X sink hXs)) _))

theorem leak_le_sum_K_sealed {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X sink : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hXs : Disjoint X sink)
    (hsec : Secrets S cert (X ∪ sink) a) (hseal : Covered ends β (cert ∪ X) sink)
    (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ sink ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        uncertifiedLinkCapacity ends β (cert ∪ sink) X * (b - a)) : ℕ) : ℕ∞) := by
  have h := leak_le_sum_K_sinks S ℓ ends β link 𝒱 cert X sink spec record bucket jitter a b K hcs hdl hT hXs hsec hK
  rw [show uncertifiedLinkCapacity ends β (cert ∪ X) sink = 0 from hseal, Nat.add_zero] at h
  exact h

end ResourceAccounting

namespace ResourceAccounting.SecurityProofs

open Verity Finset

theorem LeakParts : Guarantees.LeakParts := by
  intro L E Ω _ _ _ S ℓ ends β link 𝒱 cert chal X sink spec record bucket jitter a b K c hcs hdl hd hT hXs hsec hseal
    hK hc
  have hK' : RecordsWithin S (cert ∪ chal) record a b (fun y => if y ∈ cert then K y else c y) := by
    intro y hy
    by_cases hyc : y ∈ cert
    · have h1 := hK y hyc
      simp only [hyc, ↓reduceIte]
      exact h1
    · have h1 := hc y ((Finset.mem_union.1 hy).resolve_left hyc)
      simp only [hyc, ↓reduceIte]
      exact h1
  have hs : ∑ y ∈ cert ∪ chal, (if y ∈ cert then K y else c y) = (∑ y ∈ cert, K y) + ∑ y ∈ chal, c y := by
    have h1 : ∑ y ∈ cert, (if y ∈ cert then K y else c y) = ∑ y ∈ cert, K y :=
      Finset.sum_congr rfl fun y hy => ite_eq_left hy
    have h2 : ∑ y ∈ chal, (if y ∈ cert then K y else c y) = ∑ y ∈ chal, c y :=
      Finset.sum_congr rfl fun y hy => ite_eq_right (Finset.disjoint_right.1 hd hy)
    rw [Finset.sum_union hd, h1, h2]
  refine (leak_le_sum_K_sealed S ℓ ends β link 𝒱 (cert ∪ chal) X sink spec record bucket jitter a b
    (fun y => if y ∈ cert then K y else c y) hcs hdl hT hXs hsec hseal hK').trans ?_
  refine Nat.mono_cast (Nat.pow_le_pow_right Nat.two_pos (le_of_eq ?_))
  rw [hs, Finset.card_union_eq_card_add_card.2 hd, Nat.mul_add]
  unfold challengerTerm
  omega

end ResourceAccounting.SecurityProofs
