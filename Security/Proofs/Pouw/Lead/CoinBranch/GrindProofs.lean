import Proofs.Pouw.Lead.Tags.SsnProofs
import Mathlib.Combinatorics.Enumerative.DoubleCounting

/-!
# Target words lost on a set of coins: grinding and reading (proofs)

**Draft semantics, pending review.** A prover that can't retry the coins may still choose its inputs before them and
retry the inputs until the coins fall in a set `S` it likes (grinding). A program that reads coin digits and branches
runs each of its straight-line programs on a cell of the coins
(`Pouw.SecurityProofs.Lead.CoinBranch.adaptive_floor`), of measure `q^−g` after `g` digits. On such a set alone some target words can coincide or equal a free value, and
`DistinctWritesMH100Floor` charges only the rest. This file bounds how many are lost against the set's measure.

* **The grid** is every `q`-ary digit vector `ι → Fin q` (`q = 2`: coin bits). Targets and free values are affine in
  the digits (`aff`).
* **Triangular relations shrink the set** (`card_mul_pow_le`): if every point of `S` satisfies `r` affine relations,
  each with a nonzero coefficient at its own pivot digit and a zero coefficient at every later relation's pivot, then
  `|S|·q^r ≤ q^|ι|`. So on a set of measure at least `q^−g` (`q^|ι| ≤ q^g·|S|`) at most `g` such relations hold
  (`card_le_of_measure`).
* **Words lost** (`lost_le`). The digits are grouped into blocks (`blk`) under an order (`ord`); the target words are
  ranked, and each has a window of blocks. Suppose a word and every earlier-ranked word, or any constant, differ at a
  digit in its window past whose block they agree. The free values are the constants and a finite family of affine coin
  words `Ec`. Then on every `S` of measure at least `q^−g`, all but `g·M + |Ec|` target words are pairwise distinct on
  `S` and none is free there, where `M` bounds the words whose window holds a given block. A lost word repeats an
  earlier word or equals a free value. Its relation's last differing block is a pivot block; one relation a pivot
  block is triangular, so at most `g` blocks occur. A word whose only partners are coin words with no such pivot is
  lost at most once a coin word.
-/

namespace Pouw.Dimension.Grind

open Finset Pouw.Dimension.H100 Pouw.Dimension.Ssn

variable {ι : Type} [Fintype ι] {q : ℕ}

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

/-- **Words lost on a success set.** Digits sit in blocks `blk` ordered by `ord`. The target words `u ∈ Ω` (the
affine functions `aff (wc u) (wa u)`) are ranked by `ν`, and each has a window `win u` of blocks. A word and any word
ranked before it differ at some digit in the later one's window, past whose block they agree (`hwin`), and likewise for
a word and every constant (`hwin0`). The free values are constants and the coin words `Ec` (`hF`), and at most `M`
words have a given block in their window (`hM`). Then on every set `S` of measure at least `q^−g`, some `D ⊆ Ω` with
`|Ω| ≤ |D| + g·M + |Ec|` has its words pairwise distinct on `S`, none free there, and one function a word. -/
theorem lost_le [DecidableEq ι] {K U E : Type} [DecidableEq K] [DecidableEq U] [DecidableEq E] (hq : 1 < q)
    (S : Finset (ι → Fin q)) (g : ℕ) (hS : q ^ Fintype.card ι ≤ q ^ g * S.card)
    (blk : ι → K) (ord : K → ℕ) (hord : Function.Injective ord)
    (Ω : Finset U) (ν : U → ℕ) (hν : Set.InjOn ν Ω) (wc : U → ℚ) (wa : U → ι → ℚ) (win : U → Finset K)
    (hwin : ∀ u ∈ Ω, ∀ u' ∈ Ω, ν u' < ν u → ∃ k, blk k ∈ win u ∧ wa u k ≠ wa u' k ∧
      ∀ k', ord (blk k) < ord (blk k') → wa u k' = wa u' k')
    (hwin0 : ∀ u ∈ Ω, ∃ k, blk k ∈ win u ∧ wa u k ≠ 0 ∧ ∀ k', ord (blk k) < ord (blk k') → wa u k' = 0)
    (Ec : Finset E) (ec : E → ℚ) (ea : E → ι → ℚ)
    (F : Set ((ι → Fin q) → ℚ)) (hF : ∀ f ∈ F, (∃ c : ℚ, f = fun _ => c) ∨ ∃ e ∈ Ec, f = aff (ec e) (ea e))
    (M : ℕ) (hM : ∀ b, (Ω.filter fun u => b ∈ win u).card ≤ M) :
    ∃ D ⊆ Ω, DistinctOn (↑S) F (D.image fun u => aff (q := q) (wc u) (wa u)) ∧
      (D.image fun u => aff (q := q) (wc u) (wa u)).card = D.card ∧ Ω.card ≤ D.card + g * M + Ec.card := by
  classical
  set w : U → (ι → Fin q) → ℚ := fun u => aff (wc u) (wa u) with hw
  have htot : ∀ u ∈ Ω, ∀ u' ∈ Ω, u ≠ u' → ν u < ν u' ∨ ν u' < ν u := fun u hu u' hu' hne =>
    lt_or_gt_of_ne fun h => hne (hν (Finset.mem_coe.2 hu) (Finset.mem_coe.2 hu') h)
  let inA : U → E → Prop := fun u e =>
    ∃ k, blk k ∈ win u ∧ wa u k ≠ ea e k ∧ ∀ k', ord (blk k) < ord (blk k') → wa u k' = ea e k'
  let Alost : U → Prop := fun u =>
    (∃ u' ∈ Ω, ν u' < ν u ∧ ∀ x ∈ S, w u x = w u' x) ∨ (∃ c : ℚ, ∀ x ∈ S, w u x = c) ∨
      ∃ e ∈ Ec, inA u e ∧ ∀ x ∈ S, w u x = aff (ec e) (ea e) x
  let Blost : U → Prop := fun u => ∃ e ∈ Ec, ∀ x ∈ S, w u x = aff (ec e) (ea e) x
  let D := Ω.filter fun u => ¬ Alost u ∧ ¬ Blost u
  let A := Ω.filter Alost
  let B := Ω.filter fun u => ¬ Alost u ∧ Blost u
  have hrel : ∀ u ∈ Ω, Alost u → ∃ c' : ℚ, ∃ a' : ι → ℚ, (∀ x ∈ S, w u x = aff c' a' x) ∧
      ∃ k, blk k ∈ win u ∧ wa u k ≠ a' k ∧ ∀ k', ord (blk k) < ord (blk k') → wa u k' = a' k' := by
    intro u hu hA
    rcases hA with ⟨u', hu', hb, heq⟩ | ⟨c, heq⟩ | ⟨e, -, hpiv, heq⟩
    · exact ⟨wc u', wa u', heq, hwin u hu u' hu' hb⟩
    · exact ⟨c, fun _ => 0, fun x hx => by rw [heq x hx]; simp [aff], hwin0 u hu⟩
    · exact ⟨ec e, ea e, heq, hpiv⟩
  have hA : A.card ≤ g * M := by
    by_cases hne : A.Nonempty
    · obtain ⟨u₀, hu₀⟩ := hne
      obtain ⟨-, -, -, p₀, -⟩ := hrel u₀ (Finset.mem_filter.1 hu₀).1 (Finset.mem_filter.1 hu₀).2
      have : Nonempty ι := ⟨p₀⟩
      have : Nonempty U := ⟨u₀⟩
      have hrelA : ∀ u ∈ A, ∃ c' : ℚ, ∃ a' : ι → ℚ, (∀ x ∈ S, w u x = aff c' a' x) ∧
          ∃ k, blk k ∈ win u ∧ wa u k ≠ a' k ∧ ∀ k', ord (blk k) < ord (blk k') → wa u k' = a' k' :=
        fun u hu => hrel u (Finset.mem_filter.1 hu).1 (Finset.mem_filter.1 hu).2
      choose! c' a' heq piv hpwin hpne hptri using hrelA
      set PB := A.image fun u => blk (piv u) with hPB
      have hfib : A.card ≤ M * PB.card := by
        refine Finset.card_le_mul_card_image A M fun b _ => le_trans (Finset.card_le_card ?_) (hM b)
        intro u hu
        obtain ⟨huA, hub⟩ := Finset.mem_filter.1 hu
        exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 huA).1, by rw [← hub]; exact hpwin u huA⟩
      have hrep : ∀ b ∈ PB, ∃ u ∈ A, blk (piv u) = b := fun b hb => Finset.mem_image.1 hb
      choose! rep hrepA hrepb using hrep
      have hPBc : PB.card ≤ g := by
        refine card_le_of_measure hq hS (card_mul_pow_le S PB (fun b k => wa (rep b) k - a' (rep b) k)
          (fun b => c' (rep b) - wc (rep b)) (fun b => piv (rep b)) ord (fun _ _ _ _ h => hord h) ?_ ?_ ?_)
        · intro b hb
          exact sub_ne_zero.2 (hpne _ (hrepA b hb))
        · intro b hb b' hb' hlt
          have h1 := hptri (rep b) (hrepA b hb) (piv (rep b'))
          rw [hrepb b hb, hrepb b' hb'] at h1
          simp only [h1 hlt, sub_self]
        · intro x hx b hb
          have h3 := heq (rep b) (hrepA b hb) x hx
          simp only [hw, aff] at h3
          simp only [sub_mul, Finset.sum_sub_distrib]
          linarith
      calc A.card ≤ M * PB.card := hfib
        _ ≤ M * g := Nat.mul_le_mul_left _ hPBc
        _ = g * M := Nat.mul_comm _ _
    · rw [Finset.not_nonempty_iff_eq_empty.1 hne, Finset.card_empty]
      exact Nat.zero_le _
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
    · obtain ⟨k, -, hk, -⟩ := hwin u' hu0' u hu0 hb
      exact hk (coef_eq_of_aff_eq (c := wc u') (a := wa u') (c' := wc u) (a' := wa u) hq h.symm k)
    · obtain ⟨k, -, hk, -⟩ := hwin u hu0 u' hu0' hb
      exact hk (coef_eq_of_aff_eq (c := wc u) (a := wa u) (c' := wc u') (a' := wa u') hq h k)
  · calc Ω.card ≤ (D ∪ A ∪ B).card := Finset.card_le_card hsub
      _ ≤ (D ∪ A).card + B.card := Finset.card_union_le _ _
      _ ≤ D.card + A.card + B.card := Nat.add_le_add_right (Finset.card_union_le _ _) _
      _ ≤ D.card + g * M + Ec.card := Nat.add_le_add (Nat.add_le_add_left hA _) hB

end Pouw.Dimension.Grind
