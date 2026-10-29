import Definitions.Pouw.Assumptions.Dimension.H100W
import Mathlib.Combinatorics.Enumerative.DoubleCounting

/-!
# Grinding: target words lost on a success set (scratch; not in a lock)

**Draft, pending review.** A prover that can't retry the coins may still choose its inputs before them and retry the
inputs until the coins fall in a set `S` it likes (grinding). On `S` alone some target words can coincide or equal a
free value, and `DistinctWritesFloor` charges only the rest. This file bounds how many are lost against `S`'s measure.

* **The grid** is every `q`-ary digit vector `ι → Fin q` (`q = 2`: coin bits). Targets and free values are affine in
  the digits (`aff`).
* **Triangular relations shrink the set** (`card_mul_pow_le`): if every point of `S` satisfies `r` affine relations,
  each with a nonzero coefficient at its own pivot digit and a zero coefficient at every later relation's pivot, then
  `|S|·q^r ≤ q^|ι|`. So on a set of measure at least `q^−g` (`q^|ι| ≤ q^g·|S|`) at most `g` such relations hold
  (`card_le_of_measure`).
* **Words lost** (`lost_le`). The digits are grouped into blocks (`blk`) under an order (`ord`), and each target word
  has a key block. Suppose each word reads only blocks up to its key, two words differ somewhere in the key block of the
  one whose key is not earlier, and each word reads its key block. The free values are the constants and a finite
  family of affine coin words `Ec`. Then on every `S` of measure at least `q^−g`, all but `g·M + |Ec|` target words are
  pairwise distinct on `S` and none is free there, where `M` bounds the words a key. A lost word either repeats an
  earlier word or equals a free value; each lost word's key contributes one triangular relation unless its only partners
  are coin words read beyond its key, and at most one word per coin word is lost that way.
-/

namespace Pouw.Dimension.Grind

open Finset Pouw.Dimension.H100

variable {ι : Type} [Fintype ι] {q : ℕ}

/-- An affine function of `q`-ary digits: `c + Σ_k a k · x k`. -/
def aff (c : ℚ) (a : ι → ℚ) : (ι → Fin q) → ℚ := fun x => c + ∑ k, a k * ((x k : ℕ) : ℚ)

/-- **Triangular relations shrink the set**: if every point of `S` satisfies the relations `R`, each with a nonzero
coefficient at its pivot and a zero coefficient at every later relation's pivot, then `|S|·q^|R| ≤ q^|ι|`. -/
theorem card_mul_pow_le [DecidableEq ι] {κ : Type} (S : Finset (ι → Fin q)) (R : Finset κ) (a : κ → ι → ℚ) (c : κ → ℚ)
    (p : κ → ι) (ord : κ → ℕ) (hord : Set.InjOn ord R) (hp : ∀ r ∈ R, a r (p r) ≠ 0)
    (htri : ∀ r ∈ R, ∀ r' ∈ R, ord r < ord r' → a r (p r') = 0)
    (hS : ∀ x ∈ S, ∀ r ∈ R, ∑ k, a r k * ((x k : ℕ) : ℚ) = c r) :
    S.card * q ^ R.card ≤ q ^ Fintype.card ι := by
  classical
  have hpinj : Set.InjOn p R := by
    intro r hr r' hr' h
    by_contra hne
    rcases lt_or_gt_of_ne (fun h' => hne (hord (Finset.mem_coe.2 hr) (Finset.mem_coe.2 hr') h')) with hlt | hlt
    · exact hp r hr (by rw [h]; exact htri r hr r' hr' hlt)
    · exact hp r' hr' (by rw [← h]; exact htri r' hr' r hr hlt)
  set P := R.image p with hP
  have hPc : P.card = R.card := card_image_of_injOn hpinj
  have hdet : ∀ x ∈ S, ∀ y ∈ S, (∀ k, k ∉ P → x k = y k) → x = y := by
    intro x hx y hy hxy
    have hpiv : ∀ n, ∀ r ∈ R, ord r = n → x (p r) = y (p r) := by
      intro n
      induction n using Nat.strong_induction_on with
      | _ n ih =>
        intro r hr hrn
        have hsum : ∑ k, a r k * (((x k : ℕ) : ℚ) - ((y k : ℕ) : ℚ)) = 0 := by
          simp only [mul_sub, Finset.sum_sub_distrib, hS x hx r hr, hS y hy r hr, sub_self]
        rw [Finset.sum_eq_single (p r)] at hsum
        · have h0 := (mul_eq_zero.1 hsum).resolve_left (hp r hr)
          exact Fin.ext (by exact_mod_cast sub_eq_zero.1 h0)
        · intro k _ hk
          by_cases hkP : k ∈ P
          · obtain ⟨r', hr', rfl⟩ := Finset.mem_image.1 hkP
            have hne : ord r' ≠ ord r := fun h =>
              hk (by rw [hord (Finset.mem_coe.2 hr') (Finset.mem_coe.2 hr) h])
            rcases lt_or_gt_of_ne hne with hlt | hlt
            · rw [ih (ord r') (by omega) r' hr' rfl, sub_self, mul_zero]
            · rw [htri r hr r' hr' hlt, zero_mul]
          · rw [hxy k hkP, sub_self, mul_zero]
        · intro h
          exact absurd (Finset.mem_univ _) h
    funext k
    by_cases hkP : k ∈ P
    · obtain ⟨r, hr, rfl⟩ := Finset.mem_image.1 hkP
      exact hpiv _ r hr rfl
    · exact hxy k hkP
  have hcard : S.card ≤ q ^ (Fintype.card ι - R.card) := by
    have hle := Finset.card_le_card_of_injOn (fun x : ι → Fin q => fun k : {k // k ∉ P} => x k.1)
      (s := S) (t := Finset.univ) (fun _ _ => Finset.mem_coe.2 (Finset.mem_univ _))
      (fun x hx y hy h => hdet x hx y hy fun k hk => congrFun h ⟨k, hk⟩)
    simpa only [Finset.card_univ, Fintype.card_fun, Fintype.card_fin, Fintype.card_subtype_compl,
      Fintype.card_coe, hPc] using hle
  have hR : R.card ≤ Fintype.card ι := hPc ▸ Finset.card_le_univ P
  calc S.card * q ^ R.card ≤ q ^ (Fintype.card ι - R.card) * q ^ R.card := Nat.mul_le_mul_right _ hcard
    _ = q ^ Fintype.card ι := by rw [← pow_add, Nat.sub_add_cancel hR]

/-- On a set of measure at least `q^−g`, at most `g` triangular relations hold. -/
theorem card_le_of_measure {S : Finset (ι → Fin q)} {g r : ℕ} (hq : 1 < q)
    (hS : q ^ Fintype.card ι ≤ q ^ g * S.card) (hr : S.card * q ^ r ≤ q ^ Fintype.card ι) : r ≤ g := by
  have hpos : 0 < S.card := by
    rcases Nat.eq_zero_or_pos S.card with h | h
    · rw [h, mul_zero] at hS
      exact absurd hS (Nat.not_le.2 (pow_pos (by omega) _))
    · exact h
  have hle : q ^ r * S.card ≤ q ^ g * S.card := by rw [mul_comm]; exact hr.trans hS
  exact (pow_le_pow_iff_right₀ (by exact_mod_cast hq)).1 (Nat.le_of_mul_le_mul_right hle hpos)

/-- Affine functions of the digits that agree everywhere have the same coefficients. -/
theorem coef_eq_of_aff_eq [DecidableEq ι] (hq : 1 < q) {c c' : ℚ} {a a' : ι → ℚ} (h : aff (q := q) c a = aff c' a') (k : ι) :
    a k = a' k := by
  have h0 := congrFun h fun _ => ⟨0, by omega⟩
  have h1 := congrFun h fun k' => ⟨if k' = k then 1 else 0, by split_ifs <;> omega⟩
  simp [aff] at h0 h1
  linarith

/-- **Words lost on a success set.** Digits sit in blocks `blk` ordered by `ord`; each target word `u ∈ Ω` (the affine
function `aff (wc u) (wa u)`) has a key block `key u` and reads only blocks up to it (`hsupp`). Two words differ in the
key block of the one whose key is not earlier (`hsep`), and every word reads its key block (`hconst`). The free values
are constants and the coin words `Ec` (`hF`), and at most `M` words share a key (`hM`). Then on every set `S` of measure
at least `q^−g`, some `D ⊆ Ω` with `|Ω| ≤ |D| + g·M + |Ec|` has its words pairwise distinct on `S`, none free there, and
one function a word. -/
theorem lost_le [DecidableEq ι] {K U E : Type} [DecidableEq K] [DecidableEq U] [DecidableEq E] (hq : 1 < q)
    (S : Finset (ι → Fin q)) (g : ℕ) (hS : q ^ Fintype.card ι ≤ q ^ g * S.card)
    (blk : ι → K) (ord : K → ℕ) (hord : Function.Injective ord)
    (Ω : Finset U) (key : U → K) (ν : U → ℕ) (hν : ∀ u ∈ Ω, ∀ u' ∈ Ω, key u = key u' → ν u = ν u' → u = u')
    (wc : U → ℚ) (wa : U → ι → ℚ)
    (hsupp : ∀ u ∈ Ω, ∀ k, wa u k ≠ 0 → ord (blk k) ≤ ord (key u))
    (hsep : ∀ u ∈ Ω, ∀ u' ∈ Ω, u ≠ u' → ord (key u') ≤ ord (key u) → ∃ k, blk k = key u ∧ wa u k ≠ wa u' k)
    (hconst : ∀ u ∈ Ω, ∃ k, blk k = key u ∧ wa u k ≠ 0)
    (Ec : Finset E) (ec : E → ℚ) (ea : E → ι → ℚ)
    (F : Set ((ι → Fin q) → ℚ)) (hF : ∀ f ∈ F, (∃ c : ℚ, f = fun _ => c) ∨ ∃ e ∈ Ec, f = aff (ec e) (ea e))
    (M : ℕ) (hM : ∀ k, (Ω.filter fun u => key u = k).card ≤ M) :
    ∃ D ⊆ Ω, DistinctOn (↑S) F (D.image fun u => aff (q := q) (wc u) (wa u)) ∧
      (D.image fun u => aff (q := q) (wc u) (wa u)).card = D.card ∧ Ω.card ≤ D.card + g * M + Ec.card := by
  classical
  set w : U → (ι → Fin q) → ℚ := fun u => aff (wc u) (wa u) with hw
  let before : U → U → Prop := fun u' u => ord (key u') < ord (key u) ∨ (key u' = key u ∧ ν u' < ν u)
  have htot : ∀ u ∈ Ω, ∀ u' ∈ Ω, u ≠ u' → before u u' ∨ before u' u := by
    intro u hu u' hu' hne
    rcases lt_trichotomy (ord (key u)) (ord (key u')) with h | h | h
    · exact Or.inl (Or.inl h)
    · have hk := hord h
      rcases lt_trichotomy (ν u) (ν u') with h' | h' | h'
      · exact Or.inl (Or.inr ⟨hk, h'⟩)
      · exact absurd (hν u hu u' hu' hk h') hne
      · exact Or.inr (Or.inr ⟨hk.symm, h'⟩)
    · exact Or.inr (Or.inl h)
  have hbefore_le : ∀ u u', before u' u → ord (key u') ≤ ord (key u) := by
    rintro u u' (h | ⟨h, _⟩)
    · exact h.le
    · rw [h]
  have hbefore_ne : ∀ u u', before u' u → u' ≠ u := by
    rintro u u' (h | ⟨_, h⟩) rfl <;> exact lt_irrefl _ h
  let inA : U → E → Prop := fun u e =>
    (∀ k, ea e k ≠ 0 → ord (blk k) ≤ ord (key u)) ∧ ∃ k, blk k = key u ∧ wa u k ≠ ea e k
  let Alost : U → Prop := fun u =>
    (∃ u' ∈ Ω, before u' u ∧ ∀ x ∈ S, w u x = w u' x) ∨ (∃ c : ℚ, ∀ x ∈ S, w u x = c) ∨
      ∃ e ∈ Ec, inA u e ∧ ∀ x ∈ S, w u x = aff (ec e) (ea e) x
  let Blost : U → Prop := fun u => ∃ e ∈ Ec, ∀ x ∈ S, w u x = aff (ec e) (ea e) x
  let D := Ω.filter fun u => ¬ Alost u ∧ ¬ Blost u
  let A := Ω.filter Alost
  let B := Ω.filter fun u => ¬ Alost u ∧ Blost u
  have hrel : ∀ u ∈ Ω, Alost u → ∃ c' : ℚ, ∃ a' : ι → ℚ, (∀ x ∈ S, w u x = aff c' a' x) ∧
      (∀ k, a' k ≠ 0 → ord (blk k) ≤ ord (key u)) ∧ ∃ k, blk k = key u ∧ wa u k ≠ a' k := by
    intro u hu hA
    rcases hA with ⟨u', hu', hb, heq⟩ | ⟨c, heq⟩ | ⟨e, -, ⟨hsup, hpiv⟩, heq⟩
    · exact ⟨wc u', wa u', heq, fun k hk => (hsupp u' hu' k hk).trans (hbefore_le u u' hb),
        hsep u hu u' hu' (hbefore_ne u u' hb).symm (hbefore_le u u' hb)⟩
    · refine ⟨c, fun _ => 0, fun x hx => by rw [heq x hx]; simp [aff], fun k hk => absurd rfl hk, ?_⟩
      exact hconst u hu
    · exact ⟨ec e, ea e, heq, hsup, hpiv⟩
  have hA : A.card ≤ g * M := by
    have hfib : A.card ≤ M * (A.image key).card := by
      refine Finset.card_le_mul_card_image A M fun k _ => le_trans (Finset.card_le_card ?_) (hM k)
      exact Finset.filter_subset_filter _ (Finset.filter_subset _ _)
    have hKA : (A.image key).card ≤ g := by
      by_cases hne : (A.image key).Nonempty
      · obtain ⟨k₀, hk₀⟩ := hne
        obtain ⟨u₀, hu₀, -⟩ := Finset.mem_image.1 hk₀
        obtain ⟨-, -, -, -, p₀, -, -⟩ := hrel u₀ (Finset.mem_filter.1 hu₀).1 (Finset.mem_filter.1 hu₀).2
        have : Nonempty ι := ⟨p₀⟩
        have hkey : ∀ k ∈ A.image key, ∃ ρ : ι → ℚ, ∃ c : ℚ, ∃ p : ι, ρ p ≠ 0 ∧ blk p = k ∧
            (∀ k', ρ k' ≠ 0 → ord (blk k') ≤ ord k) ∧ ∀ x ∈ S, ∑ k', ρ k' * ((x k' : ℕ) : ℚ) = c := by
          intro k hk
          obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hk
          have hu0 := Finset.mem_filter.1 hu
          obtain ⟨c', a', heq, hsup, p, hp, hpne⟩ := hrel u hu0.1 hu0.2
          refine ⟨fun k' => wa u k' - a' k', c' - wc u, p, sub_ne_zero.2 hpne, hp, fun k' hk' => ?_,
            fun x hx => ?_⟩
          · by_cases h1 : wa u k' = 0
            · have h2 : a' k' ≠ 0 := fun h2 => hk' (by simp only [h1, h2, sub_zero])
              exact hsup k' h2
            · exact hsupp u hu0.1 k' h1
          · have h3 := heq x hx
            simp only [hw, aff] at h3
            simp only [sub_mul, Finset.sum_sub_distrib]
            linarith
        choose! ρ c p hp hpb hsup hS' using hkey
        refine card_le_of_measure hq hS (card_mul_pow_le S (A.image key) ρ c p ord
          (fun _ _ _ _ h => hord h) hp ?_ fun x hx r hr => hS' r hr x hx)
        intro r hr r' hr' hlt
        by_contra h
        have := hsup r hr (p r') h
        rw [hpb r' hr'] at this
        omega
      · rw [Finset.not_nonempty_iff_eq_empty.1 hne, Finset.card_empty]
        exact Nat.zero_le _
    calc A.card ≤ M * (A.image key).card := hfib
      _ ≤ M * g := Nat.mul_le_mul_left _ hKA
      _ = g * M := Nat.mul_comm _ _
  have hB : B.card ≤ Ec.card := by
    refine Finset.card_le_card_of_forall_subsingleton (fun u e => ∀ x ∈ S, w u x = aff (ec e) (ea e) x)
      (fun u hu => (Finset.mem_filter.1 hu).2.2) fun e _ u hu u' hu' => ?_
    by_contra hne
    have hu0 := Finset.mem_filter.1 hu.1
    have hu0' := Finset.mem_filter.1 hu'.1
    rcases htot u hu0.1 u' hu0'.1 hne with hb | hb
    · exact hu0'.2.1 (Or.inl ⟨u, hu0.1, hb, fun x hx => by rw [hu'.2 x hx, hu.2 x hx]⟩)
    · exact hu0.2.1 (Or.inl ⟨u', hu0'.1, hb, fun x hx => by rw [hu.2 x hx, hu'.2 x hx]⟩)
  have hsub : Ω ⊆ D ∪ A ∪ B := by
    intro u hu
    by_cases ha : Alost u
    · exact Finset.mem_union_left _ (Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hu, ha⟩))
    · by_cases hb : Blost u
      · exact Finset.mem_union_right _ (Finset.mem_filter.2 ⟨hu, ha, hb⟩)
      · exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hu, ha, hb⟩))
  refine ⟨D, Finset.filter_subset _ _, ⟨?_, ?_⟩, ?_, ?_⟩
  · intro t ht t' ht' heq
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 ht
    obtain ⟨u', hu', rfl⟩ := Finset.mem_image.1 ht'
    by_cases huu : u = u'
    · rw [huu]
    · have hu0 := Finset.mem_filter.1 hu
      have hu0' := Finset.mem_filter.1 hu'
      rcases htot u hu0.1 u' hu0'.1 huu with hb | hb
      · exact absurd (Or.inl ⟨u, hu0.1, hb, fun x hx => (heq x (Finset.mem_coe.2 hx)).symm⟩) hu0'.2.1
      · exact absurd (Or.inl ⟨u', hu0'.1, hb, fun x hx => heq x (Finset.mem_coe.2 hx)⟩) hu0.2.1
  · intro t ht f hf heq
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 ht
    have hu0 := Finset.mem_filter.1 hu
    rcases hF f hf with ⟨c, rfl⟩ | ⟨e, he, rfl⟩
    · exact hu0.2.1 (Or.inr (Or.inl ⟨c, fun x hx => heq x (Finset.mem_coe.2 hx)⟩))
    · exact hu0.2.2 ⟨e, he, fun x hx => heq x (Finset.mem_coe.2 hx)⟩
  · refine Finset.card_image_of_injOn fun u hu u' hu' h => ?_
    by_contra hne
    have hu0 := (Finset.mem_filter.1 (Finset.mem_coe.1 hu)).1
    have hu0' := (Finset.mem_filter.1 (Finset.mem_coe.1 hu')).1
    rcases htot u hu0 u' hu0' hne with hb | hb
    · obtain ⟨k, -, hk⟩ := hsep u' hu0' u hu0 (Ne.symm hne) (hbefore_le u' u hb)
      exact hk (coef_eq_of_aff_eq (c := wc u') (a := wa u') (c' := wc u) (a' := wa u) hq h.symm k)
    · obtain ⟨k, -, hk⟩ := hsep u hu0 u' hu0' hne (hbefore_le u u' hb)
      exact hk (coef_eq_of_aff_eq (c := wc u) (a := wa u) (c' := wc u') (a' := wa u') hq h k)
  · calc Ω.card ≤ (D ∪ A ∪ B).card := Finset.card_le_card hsub
      _ ≤ (D ∪ A).card + B.card := Finset.card_union_le _ _
      _ ≤ D.card + A.card + B.card := Nat.add_le_add_right (Finset.card_union_le _ _) _
      _ ≤ D.card + g * M + Ec.card := Nat.add_le_add (Nat.add_le_add_left hA _) hB

end Pouw.Dimension.Grind
