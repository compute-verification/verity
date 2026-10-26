import Definitions.Pouw.MerkleExtract

/-!
# Merkle extraction: the deterministic core and the fresh-answer count

* `extract_sound` (Lemma 1, for every hash function): an opening that verifies against `d` equals the extractor's
  word, or the hash collides on a queried point and a verified point, or a verified point outside `L` hashes to a
  target (`d` or a child digest `L` names).
* `card_hit_le`: a uniformly random answer stream puts coordinate `t` in a set of at most `m` values that doesn't
  read `t` on at most a `m / |D|` share of streams. Both random-oracle bad events of `extract_of_ro` are of this form.
-/

namespace Pouw.MerkleExtract

variable {X D W : Type}

/-- **Lemma 1 (deterministic).** For every hash `h` and every set `L` of points queried before the commitment, an
opening of `w` that verifies against `d` is the extractor's word, or `h` collides on a point of `L` and a point the
verification hashes, or a point the verification hashes lies outside `L` and hashes to `d` or a child digest `L`
names. -/
theorem extract_sound (h : X → D) (E : Enc X D W) (L : Set X) :
    ∀ (p : List (Bool × D × D)) (d : D) (w : W), Opens h E d p w →
      ext h E L d (p.map Prod.fst) = some w ∨
      (∃ x ∈ L, ∃ y ∈ pts E p w, x ≠ y ∧ h x = h y) ∨
      (∃ y ∈ pts E p w, y ∉ L ∧ h y ∈ insert d (kids E L))
  | [], d, w, hO => by
      simp only [Opens] at hO
      simp only [List.map_nil, ext, pts]
      by_cases hw : ∃ w', E.leaf w' ∈ L ∧ h (E.leaf w') = d
      · rw [dite_cond_eq_true (eq_true hw)]
        obtain ⟨hmem, hh⟩ := Classical.choose_spec hw
        by_cases heq : Classical.choose hw = w
        · left; rw [heq]
        · right; left
          refine ⟨E.leaf (Classical.choose hw), hmem, E.leaf w, by simp, ?_, ?_⟩
          · exact fun e => heq (E.leaf_inj e)
          · rw [hh, hO]
      · rw [dite_cond_eq_false (eq_false hw)]
        right; right
        refine ⟨E.leaf w, by simp, ?_, ?_⟩
        · intro hm; exact hw ⟨w, hm, hO⟩
        · rw [hO]; exact Set.mem_insert _ _
  | (b, s, c) :: p, d, w, hO => by
      obtain ⟨hd, hrest⟩ := hO
      simp only [List.map_cons, ext]
      by_cases hn : ∃ lr : D × D, E.node lr.1 lr.2 ∈ L ∧ h (E.node lr.1 lr.2) = d
      · rw [dite_cond_eq_true (eq_true hn)]
        obtain ⟨hmem, hh⟩ := Classical.choose_spec hn
        by_cases heq : E.node (Classical.choose hn).1 (Classical.choose hn).2 = E.pt b s c
        · have hc : (if b then (Classical.choose hn).2 else (Classical.choose hn).1) = c := by
            cases b
            · simp only [Enc.pt, Bool.false_eq_true, ↓reduceIte] at heq ⊢; exact (E.node_inj heq).1
            · simp only [Enc.pt, ↓reduceIte] at heq ⊢; exact (E.node_inj heq).2
          have hck : c ∈ kids E L := by
            refine ⟨(Classical.choose hn).1, (Classical.choose hn).2, hmem, ?_⟩
            cases b
            · simp only [Bool.false_eq_true, ↓reduceIte] at hc; exact Or.inl hc.symm
            · simp only [↓reduceIte] at hc; exact Or.inr hc.symm
          rw [hc]
          rcases extract_sound h E L p c w hrest with h1 | ⟨x, hx, y, hy, hxy, hxy'⟩ | ⟨y, hy, hyL, hyT⟩
          · exact Or.inl h1
          · exact Or.inr (Or.inl ⟨x, hx, y, List.mem_cons_of_mem _ hy, hxy, hxy'⟩)
          · refine Or.inr (Or.inr ⟨y, List.mem_cons_of_mem _ hy, hyL, ?_⟩)
            rcases hyT with h' | h'
            · exact Set.mem_insert_of_mem _ (h' ▸ hck)
            · exact Set.mem_insert_of_mem _ h'
        · exact Or.inr (Or.inl ⟨_, hmem, E.pt b s c, List.mem_cons_self .., heq, by rw [hh, hd]⟩)
      · rw [dite_cond_eq_false (eq_false hn)]
        refine Or.inr (Or.inr ⟨E.pt b s c, List.mem_cons_self .., ?_, ?_⟩)
        · intro hm
          apply hn
          cases b
          · exact ⟨(c, s), by simpa [Enc.pt] using hm, by simpa [Enc.pt] using hd⟩
          · exact ⟨(s, c), by simpa [Enc.pt] using hm, by simpa [Enc.pt] using hd⟩
        · rw [hd]; exact Set.mem_insert _ _

/-- **Fresh-answer hits**: if the set `S u` doesn't read coordinate `t` and has at most `m` elements, then at most a
`m / |D|` share of the answer streams `u` put `u t` in `S u`. -/
theorem card_hit_le {q : ℕ} {D : Type} [Fintype D] [DecidableEq D] (t : Fin q)
    (S : (Fin q → D) → Finset D) (m : ℕ)
    (hS : ∀ u a, S (Function.update u t a) = S u) (hm : ∀ u, (S u).card ≤ m) :
    (Finset.univ.filter fun u : Fin q → D => u t ∈ S u).card * Fintype.card D ≤
      m * Fintype.card (Fin q → D) := by
  classical
  rcases isEmpty_or_nonempty D with hD | ⟨⟨a0⟩⟩
  · simp
  let e := Equiv.piSplitAt t (fun _ => D)
  have hsplit : ∀ a r, e.symm (a, r) = Function.update (e.symm (a0, r)) t a := by
    intro a r; funext j
    by_cases hj : j = t
    · subst hj; simp [e, Equiv.piSplitAt_symm_apply]
    · simp [e, Equiv.piSplitAt_symm_apply, hj]
  have h1 : ∀ a r, (e.symm (a, r)) t = a := by intro a r; simp [e, Equiv.piSplitAt_symm_apply]
  have hcount : (Finset.univ.filter fun u : Fin q → D => u t ∈ S u).card =
      ∑ r : {j // j ≠ t} → D, (S (e.symm (a0, r))).card := by
    rw [Finset.card_filter, ← e.symm.sum_comp, Fintype.sum_prod_type_right]
    refine Finset.sum_congr rfl fun r _ => ?_
    have hS' : ∀ a, S (e.symm (a, r)) = S (e.symm (a0, r)) := fun a => by rw [hsplit a r, hS]
    simp only [h1, hS']
    rw [Finset.sum_boole, Finset.filter_mem_eq_inter, Finset.univ_inter]
    simp
  have hall : Fintype.card (Fin q → D) = Fintype.card D * Fintype.card ({j // j ≠ t} → D) := by
    rw [Fintype.card_congr e, Fintype.card_prod]
  rw [hcount, hall]
  calc (∑ r : {j // j ≠ t} → D, (S (e.symm (a0, r))).card) * Fintype.card D
      ≤ (∑ _r : {j // j ≠ t} → D, m) * Fintype.card D := by
        gcongr with r; exact hm _
    _ = m * (Fintype.card D * Fintype.card ({j // j ≠ t} → D)) := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul]; ring

end Pouw.MerkleExtract
