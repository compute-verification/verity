import Definitions.Pouw.MerkleExtract

/-!
# Merkle extraction: the deterministic core and the fresh-answer count

* `extract_sound` (Lemma 1, for every hash function): an opening that verifies against `d` equals the extractor's
  word, or the hash collides on a queried point and a verified point, or a verified point outside `L` hashes to a
  target (`d` or a child digest `L` names).
* `card_hit_le`: a uniformly random answer stream puts coordinate `t` in a set of at most `m` values that doesn't
  read `t` on at most a `m / |D|` share of streams. Both random-oracle bad events of `extract_of_ro` are of this form.
* `extract_of_ro` (Lemma 2): against a `q`-query prover that commits after `τ` queries, extraction from the
  pre-commitment log fails on at most a `(q² + q (2τ + 1)) / |D|` share of the oracle's answer streams: a collision
  among the answers (`q²`), or a post-commitment answer hitting the root or a child digest the log names
  (`q (2τ + 1)`).
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

/-! ### `extract_of_ro` -/

namespace Prover

variable {q τ : ℕ} (A : Prover X D q τ)

/-- A prover's queries are distinct. -/
theorem qry_inj (u : Fin q → D) {i j : Fin q} (h : A.qry i u = A.qry j u) : i = j := by
  rcases lt_trichotomy i j with hij | hij | hij
  · exact absurd h.symm (A.fresh u j i hij)
  · exact hij
  · exact absurd h (A.fresh u i j hij)

/-- The oracle answers the prover's `i`-th query with `u i`. -/
theorem oracle_qry [Inhabited D] (u : Fin q → D) (i : Fin q) : A.oracle u (A.qry i u) = u i := by
  have hx : ∃ j, A.qry j u = A.qry i u := ⟨i, rfl⟩
  simp only [oracle, dif_pos hx]
  rw [A.qry_inj u (Classical.choose_spec hx)]

end Prover

open Classical in
/-- **`extract_of_ro`**: over the uniform answer streams, extraction fails on at most a `(q² + q (2τ + 1)) / |D|`
share. -/
theorem extract_of_ro [Fintype D] [DecidableEq D] [Inhabited D] {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) :
    (Finset.univ.filter fun u => ExtractFails E A u).card * Fintype.card D ≤
      (q * q + q * (2 * τ + 1)) * Fintype.card (Fin q → D) := by
  classical
  -- the collision sets and the target sets
  let S : Fin q → (Fin q → D) → Finset D := fun t u => (Finset.univ.filter fun j : Fin q => j < t).image u
  let T : (Fin q → D) → Finset D := fun u =>
    insert (A.root u) ((Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).biUnion fun i => children E (A.qry i u))
  let S' : Fin q → (Fin q → D) → Finset D := fun b u => if τ ≤ (b : ℕ) then T u else ∅
  have hsub : (Finset.univ.filter fun u => ExtractFails E A u) ⊆
      (Finset.univ.biUnion fun t => Finset.univ.filter fun u => u t ∈ S t u) ∪
        (Finset.univ.biUnion fun b => Finset.univ.filter fun u => u b ∈ S' b u) := by
    intro u hu
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hu
    obtain ⟨p, w, hO, hcov, hne⟩ := hu
    rcases extract_sound (A.oracle u) E (A.pre u) p (A.root u) w hO with
      h1 | ⟨x, hx, y, hy, hxy, hh⟩ | ⟨y, hy, hyL, hyT⟩
    · exact absurd h1 hne
    · obtain ⟨a, -, rfl⟩ := hx
      obtain ⟨b, rfl⟩ := hcov y hy
      rw [A.oracle_qry, A.oracle_qry] at hh
      have hab : a ≠ b := fun e => hxy (by rw [e])
      apply Finset.mem_union_left
      rw [Finset.mem_biUnion]
      rcases lt_or_gt_of_ne hab with hlt | hlt
      · refine ⟨b, Finset.mem_univ _, ?_⟩
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, S, Finset.mem_image]
        exact ⟨a, hlt, hh⟩
      · refine ⟨a, Finset.mem_univ _, ?_⟩
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, S, Finset.mem_image]
        exact ⟨b, hlt, hh.symm⟩
    · obtain ⟨b, rfl⟩ := hcov y hy
      have hb : τ ≤ (b : ℕ) := by
        by_contra hlt; exact hyL ⟨b, by omega, rfl⟩
      rw [A.oracle_qry] at hyT
      apply Finset.mem_union_right
      rw [Finset.mem_biUnion]
      refine ⟨b, Finset.mem_univ _, ?_⟩
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, S', if_pos hb, T, Finset.mem_insert,
        Finset.mem_biUnion]
      rcases hyT with h | ⟨l, r, ⟨i, hi, hq⟩, hc⟩
      · exact Or.inl h
      · right
        refine ⟨i, by simpa using hi, ?_⟩
        have hx : ∃ lr : D × D, E.node lr.1 lr.2 = A.qry i u := ⟨(l, r), hq.symm⟩
        have hlr := E.node_inj ((Classical.choose_spec hx).trans hq)
        unfold children
        rw [dif_pos hx]
        rcases hc with hc | hc
        · simp [hc, hlr.1]
        · simp [hc, hlr.2]
  have hS : ∀ t u a, S t (Function.update u t a) = S t u := by
    intro t u a
    refine Finset.image_congr fun j hj => ?_
    have : j ≠ t := ne_of_lt (by simpa using hj)
    simp [Function.update_of_ne this]
  have hSc : ∀ t u, (S t u).card ≤ q := fun t u =>
    (Finset.card_image_le).trans ((Finset.card_filter_le _ _).trans (by simp))
  have hT : ∀ (b : Fin q) u a, τ ≤ (b : ℕ) → T (Function.update u b a) = T u := by
    intro b u a hb
    have hagree : ∀ j : Fin q, (j : ℕ) < τ → Function.update u b a j = u j := fun j hj =>
      Function.update_of_ne (fun e => by subst e; omega) _ _
    simp only [T]
    rw [A.root_dep _ _ hagree]
    congr 1
    refine Finset.biUnion_congr rfl fun i hi => ?_
    have hi' : (i : ℕ) < τ := by simpa using hi
    rw [A.qry_dep i _ u fun j hj => hagree j (lt_trans (Fin.lt_def.mp hj) hi')]
  have hS' : ∀ b u a, S' b (Function.update u b a) = S' b u := by
    intro b u a
    by_cases hb : τ ≤ (b : ℕ)
    · simp only [S', if_pos hb]; exact hT b u a hb
    · simp only [S', if_neg hb]
  have hTc : ∀ u, (T u).card ≤ 2 * τ + 1 := by
    intro u
    have hF : (Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).card ≤ τ := by
      calc (Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).card ≤ (Finset.range τ).card :=
            Finset.card_le_card_of_injOn (fun i => (i : ℕ))
              (fun i hi => by simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hi
                              simpa using hi) (fun i _ j _ h => Fin.ext h)
        _ = τ := Finset.card_range τ
    have hch : ∀ x, (children E x).card ≤ 2 := by
      intro x; unfold children; split_ifs
      · exact Finset.card_le_two
      · simp
    calc (T u).card
        ≤ ((Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).biUnion fun i => children E (A.qry i u)).card + 1 :=
          Finset.card_insert_le _ _
      _ ≤ (∑ i ∈ Finset.univ.filter fun i : Fin q => (i : ℕ) < τ, (children E (A.qry i u)).card) + 1 := by
          gcongr; exact Finset.card_biUnion_le
      _ ≤ (∑ _i ∈ Finset.univ.filter fun i : Fin q => (i : ℕ) < τ, 2) + 1 := by
          gcongr with i _; exact hch _
      _ ≤ 2 * τ + 1 := by rw [Finset.sum_const, smul_eq_mul]; omega
  have hS'c : ∀ b u, (S' b u).card ≤ 2 * τ + 1 := by
    intro b u
    by_cases hb : τ ≤ (b : ℕ)
    · simp only [S', if_pos hb]; exact hTc u
    · simp only [S', if_neg hb]; simp
  calc (Finset.univ.filter fun u => ExtractFails E A u).card * Fintype.card D
      ≤ ((Finset.univ.biUnion fun t => Finset.univ.filter fun u => u t ∈ S t u) ∪
          (Finset.univ.biUnion fun b => Finset.univ.filter fun u => u b ∈ S' b u)).card * Fintype.card D :=
        Nat.mul_le_mul_right _ (Finset.card_le_card hsub)
    _ ≤ (∑ t : Fin q, (Finset.univ.filter fun u => u t ∈ S t u).card +
          ∑ b : Fin q, (Finset.univ.filter fun u => u b ∈ S' b u).card) * Fintype.card D := by
        refine Nat.mul_le_mul_right _ ((Finset.card_union_le _ _).trans (Nat.add_le_add ?_ ?_))
        · exact Finset.card_biUnion_le
        · exact Finset.card_biUnion_le
    _ = ∑ t : Fin q, (Finset.univ.filter fun u => u t ∈ S t u).card * Fintype.card D +
          ∑ b : Fin q, (Finset.univ.filter fun u => u b ∈ S' b u).card * Fintype.card D := by
        rw [add_mul, Finset.sum_mul, Finset.sum_mul]
    _ ≤ ∑ _t : Fin q, q * Fintype.card (Fin q → D) +
          ∑ _b : Fin q, (2 * τ + 1) * Fintype.card (Fin q → D) := by
        exact add_le_add (Finset.sum_le_sum fun t _ => card_hit_le t (S t) q (hS t) (hSc t))
          (Finset.sum_le_sum fun b _ => card_hit_le b (S' b) (2 * τ + 1) (hS' b) (hS'c b))
    _ = (q * q + q * (2 * τ + 1)) * Fintype.card (Fin q → D) := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]; ring

end Pouw.MerkleExtract
