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
* `extract_of_ro_roots`: the same for `R` roots over one stream, `(q² + ∑_w q (2τ_w + 1)) / |D|`, with one
  repeated-answer event for all the roots, which the bound also counts on its own (`mem_repeats_of_not_injective`).
* Lazy sampling (`exists_consistent`, `card_consistent_le`): every oracle `h : X → D` gives an adaptive querier an
  answer stream, and a querier with distinct queries sees each stream on a `1/|D|^q` share of the oracles.
* `card_fails_oracle_le`: `extract_of_ro_roots` over a uniform oracle `X → D` instead of the stream.
* `opens_congr`: an opening's check reads the hash only at the points it hashes.
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

/-! ### The two bad events, `extract_of_ro` and its `R`-root form -/

section RandomOracle

variable [Fintype D] [DecidableEq D]

/-- The answer streams on which answer `t` repeats an earlier answer. -/
def repeats (q : ℕ) (t : Fin q) : Finset (Fin q → D) :=
  Finset.univ.filter fun u => u t ∈ (Finset.univ.filter fun j : Fin q => j < t).image u

/-- The targets the commitment fixes: the root and the child digests of the queries before it. -/
noncomputable def targets {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) (u : Fin q → D) : Finset D :=
  insert (A.root u) ((Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).biUnion fun i => children E (A.qry i u))

/-- The targets answer `b` can hit: the commitment's, if `b` is answered after it. -/
noncomputable def targetsAt {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) (b : Fin q) (u : Fin q → D) :
    Finset D :=
  if τ ≤ (b : ℕ) then targets E A u else ∅

/-- The answer streams on which an answer after the commitment hits one of its targets. -/
noncomputable def hits {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) : Finset (Fin q → D) :=
  Finset.univ.biUnion fun b => Finset.univ.filter fun u => u b ∈ targetsAt E A b u

/-- A repeated answer: on at most a `q² / |D|` share of the streams. -/
theorem card_repeats_le (q : ℕ) :
    (Finset.univ.biUnion fun t : Fin q => (repeats q t : Finset (Fin q → D))).card * Fintype.card D ≤
      q * q * Fintype.card (Fin q → D) := by
  have hS : ∀ (t : Fin q) (u : Fin q → D) (a : D),
      (Finset.univ.filter fun j : Fin q => j < t).image (Function.update u t a) =
        (Finset.univ.filter fun j : Fin q => j < t).image u := by
    intro t u a
    refine Finset.image_congr fun j hj => ?_
    have : j ≠ t := ne_of_lt (by simpa using hj)
    simp [Function.update_of_ne this]
  have hSc : ∀ (t : Fin q) (u : Fin q → D), ((Finset.univ.filter fun j : Fin q => j < t).image u).card ≤ q :=
    fun t u => (Finset.card_image_le).trans ((Finset.card_filter_le _ _).trans (by simp))
  calc (Finset.univ.biUnion fun t : Fin q => (repeats q t : Finset (Fin q → D))).card * Fintype.card D
      ≤ (∑ t : Fin q, (repeats q t : Finset (Fin q → D)).card) * Fintype.card D :=
        Nat.mul_le_mul_right _ Finset.card_biUnion_le
    _ = ∑ t : Fin q, (repeats q t : Finset (Fin q → D)).card * Fintype.card D := by rw [Finset.sum_mul]
    _ ≤ ∑ _t : Fin q, q * Fintype.card (Fin q → D) :=
        Finset.sum_le_sum fun t _ =>
          card_hit_le t (fun u => (Finset.univ.filter fun j : Fin q => j < t).image u) q (hS t) (hSc t)
    _ = q * q * Fintype.card (Fin q → D) := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]; ring

/-- A stream that gives two queries one answer repeats an answer. -/
theorem mem_repeats_of_not_injective {q : ℕ} {u : Fin q → D} (hu : ¬ Function.Injective u) :
    u ∈ Finset.univ.biUnion fun t : Fin q => (repeats q t : Finset (Fin q → D)) := by
  simp only [Function.Injective, not_forall] at hu
  obtain ⟨a, b, hab, hne⟩ := hu
  rw [Finset.mem_biUnion]
  rcases lt_or_gt_of_ne hne with hlt | hlt
  · refine ⟨b, Finset.mem_univ _, ?_⟩
    simp only [repeats, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
    exact ⟨a, hlt, hab⟩
  · refine ⟨a, Finset.mem_univ _, ?_⟩
    simp only [repeats, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
    exact ⟨b, hlt, hab.symm⟩

omit [Fintype D] in
/-- The targets answer `b` can hit don't read answer `b`. -/
theorem targetsAt_update {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) (b : Fin q) (u : Fin q → D) (a : D) :
    targetsAt E A b (Function.update u b a) = targetsAt E A b u := by
  unfold targetsAt
  by_cases hb : τ ≤ (b : ℕ)
  · rw [if_pos hb, if_pos hb]
    have hagree : ∀ j : Fin q, (j : ℕ) < τ → Function.update u b a j = u j := fun j hj =>
      Function.update_of_ne (fun e => by subst e; omega) _ _
    unfold targets
    rw [A.root_dep _ _ hagree]
    congr 1
    refine Finset.biUnion_congr rfl fun i hi => ?_
    have hi' : (i : ℕ) < τ := by simpa using hi
    rw [A.qry_dep i _ u fun j hj => hagree j (lt_trans (Fin.lt_def.mp hj) hi')]
  · rw [if_neg hb, if_neg hb]

/-- At most `2τ + 1` targets: the root and two children of each query before the commitment. -/
theorem card_targetsAt_le {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) (b : Fin q) (u : Fin q → D) :
    (targetsAt E A b u).card ≤ 2 * τ + 1 := by
  unfold targetsAt
  split_ifs
  · have hF : (Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).card ≤ τ := by
      calc (Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).card ≤ (Finset.range τ).card :=
            Finset.card_le_card_of_injOn (fun i => (i : ℕ))
              (fun i hi => by simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hi
                              simpa using hi) (fun i _ j _ h => Fin.ext h)
        _ = τ := Finset.card_range τ
    have hch : ∀ x, (children E x).card ≤ 2 := by
      intro x; unfold children; split_ifs
      · exact Finset.card_le_two
      · simp
    unfold targets
    calc (insert (A.root u) ((Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).biUnion
            fun i => children E (A.qry i u))).card
        ≤ ((Finset.univ.filter fun i : Fin q => (i : ℕ) < τ).biUnion fun i => children E (A.qry i u)).card + 1 :=
          Finset.card_insert_le _ _
      _ ≤ (∑ i ∈ Finset.univ.filter fun i : Fin q => (i : ℕ) < τ, (children E (A.qry i u)).card) + 1 := by
          gcongr; exact Finset.card_biUnion_le
      _ ≤ (∑ _i ∈ Finset.univ.filter fun i : Fin q => (i : ℕ) < τ, 2) + 1 := by
          gcongr with i _; exact hch _
      _ ≤ 2 * τ + 1 := by rw [Finset.sum_const, smul_eq_mul]; omega
  · simp

/-- An answer after the commitment hitting a target: on at most a `q (2τ + 1) / |D|` share of the streams. -/
theorem card_hits_le {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) :
    (hits E A).card * Fintype.card D ≤ q * (2 * τ + 1) * Fintype.card (Fin q → D) := by
  calc (hits E A).card * Fintype.card D
      ≤ (∑ b : Fin q, (Finset.univ.filter fun u => u b ∈ targetsAt E A b u).card) * Fintype.card D :=
        Nat.mul_le_mul_right _ Finset.card_biUnion_le
    _ = ∑ b : Fin q, (Finset.univ.filter fun u => u b ∈ targetsAt E A b u).card * Fintype.card D := by
        rw [Finset.sum_mul]
    _ ≤ ∑ _b : Fin q, (2 * τ + 1) * Fintype.card (Fin q → D) :=
        Finset.sum_le_sum fun b _ =>
          card_hit_le b (targetsAt E A b) (2 * τ + 1) (targetsAt_update E A b) (card_targetsAt_le E A b)
    _ = q * (2 * τ + 1) * Fintype.card (Fin q → D) := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]; ring

open Classical in
/-- Every stream on which extraction fails repeats an answer or hits a target (Lemma 1 at the prover's log). -/
theorem fails_sub [Inhabited D] {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) :
    (Finset.univ.filter fun u => ExtractFails E A u) ⊆
      (Finset.univ.biUnion fun t => repeats q t) ∪ hits E A := by
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
      simp only [repeats, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
      exact ⟨a, hlt, hh⟩
    · refine ⟨a, Finset.mem_univ _, ?_⟩
      simp only [repeats, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
      exact ⟨b, hlt, hh.symm⟩
  · obtain ⟨b, rfl⟩ := hcov y hy
    have hb : τ ≤ (b : ℕ) := by
      by_contra hlt; exact hyL ⟨b, by omega, rfl⟩
    rw [A.oracle_qry] at hyT
    apply Finset.mem_union_right
    unfold hits
    rw [Finset.mem_biUnion]
    refine ⟨b, Finset.mem_univ _, ?_⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, targetsAt, if_pos hb, targets, Finset.mem_insert,
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

open Classical in
/-- **`extract_of_ro`**: over the uniform answer streams, extraction fails on at most a `(q² + q (2τ + 1)) / |D|`
share. -/
theorem extract_of_ro [Inhabited D] {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) :
    (Finset.univ.filter fun u => ExtractFails E A u).card * Fintype.card D ≤
      (q * q + q * (2 * τ + 1)) * Fintype.card (Fin q → D) := by
  calc (Finset.univ.filter fun u => ExtractFails E A u).card * Fintype.card D
      ≤ ((Finset.univ.biUnion fun t => repeats q t) ∪ hits E A).card * Fintype.card D :=
        Nat.mul_le_mul_right _ (Finset.card_le_card (fails_sub E A))
    _ ≤ ((Finset.univ.biUnion fun t => (repeats q t : Finset (Fin q → D))).card + (hits E A).card) *
          Fintype.card D := Nat.mul_le_mul_right _ (Finset.card_union_le _ _)
    _ = (Finset.univ.biUnion fun t => (repeats q t : Finset (Fin q → D))).card * Fintype.card D +
          (hits E A).card * Fintype.card D := add_mul _ _ _
    _ ≤ q * q * Fintype.card (Fin q → D) + q * (2 * τ + 1) * Fintype.card (Fin q → D) :=
        Nat.add_le_add (card_repeats_le q) (card_hits_le E A)
    _ = (q * q + q * (2 * τ + 1)) * Fintype.card (Fin q → D) := (add_mul _ _ _).symm

open Classical in
/-- **`extract_of_ro_roots`, the `R`-root form**: `R` roots over one answer stream, root `w` sent after the first
`τ w` answers (`A w` is the prover with that root; in a period the `A w` share their queries). Extraction fails at some
root, or two answers repeat, on at most a `(q² + ∑_w q (2 τ_w + 1)) / |D|` share of the streams: the repeated-answer
event is one for all the roots, and each root adds its own targets. -/
theorem extract_of_ro_roots [Inhabited D] {q R : ℕ} {τ : Fin R → ℕ} (E : Enc X D W)
    (A : ∀ w, Prover X D q (τ w)) :
    (Finset.univ.filter fun u => (∃ w, ExtractFails E (A w) u) ∨ ¬ Function.Injective u).card * Fintype.card D ≤
      (q * q + ∑ w, q * (2 * τ w + 1)) * Fintype.card (Fin q → D) := by
  have hsub : (Finset.univ.filter fun u => (∃ w, ExtractFails E (A w) u) ∨ ¬ Function.Injective u) ⊆
      (Finset.univ.biUnion fun t => repeats q t) ∪ Finset.univ.biUnion fun w => hits E (A w) := by
    intro u hu
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hu
    rcases hu with ⟨w, hw⟩ | hu
    · rcases Finset.mem_union.mp (fails_sub E (A w) (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hw⟩)) with h | h
      · exact Finset.mem_union_left _ h
      · exact Finset.mem_union_right _ (Finset.mem_biUnion.mpr ⟨w, Finset.mem_univ _, h⟩)
    · exact Finset.mem_union_left _ (mem_repeats_of_not_injective hu)
  calc (Finset.univ.filter fun u => (∃ w, ExtractFails E (A w) u) ∨ ¬ Function.Injective u).card *
        Fintype.card D
      ≤ ((Finset.univ.biUnion fun t => repeats q t) ∪ Finset.univ.biUnion fun w => hits E (A w)).card *
          Fintype.card D := Nat.mul_le_mul_right _ (Finset.card_le_card hsub)
    _ ≤ ((Finset.univ.biUnion fun t => (repeats q t : Finset (Fin q → D))).card + ∑ w, (hits E (A w)).card) *
          Fintype.card D :=
        Nat.mul_le_mul_right _ ((Finset.card_union_le _ _).trans (Nat.add_le_add_left Finset.card_biUnion_le _))
    _ = (Finset.univ.biUnion fun t => (repeats q t : Finset (Fin q → D))).card * Fintype.card D +
          ∑ w, (hits E (A w)).card * Fintype.card D := by rw [add_mul, Finset.sum_mul]
    _ ≤ q * q * Fintype.card (Fin q → D) + ∑ w, q * (2 * τ w + 1) * Fintype.card (Fin q → D) :=
        Nat.add_le_add (card_repeats_le q) (Finset.sum_le_sum fun w _ => card_hits_le E (A w))
    _ = (q * q + ∑ w, q * (2 * τ w + 1)) * Fintype.card (Fin q → D) := by rw [add_mul, Finset.sum_mul]

end RandomOracle

/-! ### Lazy sampling: from a uniform oracle `X → D` to the uniform answer stream -/

section LazySampling

/-- The oracle's answers to the querier, iterated `n` times from `default`. -/
def iterAns [Inhabited D] {q : ℕ} (qry : Fin q → (Fin q → D) → X) (h : X → D) : ℕ → Fin q → D
  | 0 => fun _ => default
  | n + 1 => fun i => h (qry i (iterAns qry h n))

/-- Iterating once more changes none of the first `n` answers, as query `j` reads only the answers before it. -/
theorem iterAns_succ [Inhabited D] {q : ℕ} (qry : Fin q → (Fin q → D) → X)
    (hdep : ∀ i u u', (∀ j : Fin q, j < i → u j = u' j) → qry i u = qry i u') (h : X → D) :
    ∀ n (j : Fin q), (j : ℕ) < n → iterAns qry h (n + 1) j = iterAns qry h n j := by
  intro n
  induction n with
  | zero => intro j hj; exact absurd hj (Nat.not_lt_zero _)
  | succ n ih =>
    intro j hj
    show h (qry j (iterAns qry h (n + 1))) = h (qry j (iterAns qry h n))
    rw [hdep j (iterAns qry h (n + 1)) (iterAns qry h n) fun l hl => ih l (by rw [Fin.lt_def] at hl; omega)]

/-- **Every oracle gives an adaptive querier an answer stream.** -/
theorem exists_consistent [Inhabited D] {q : ℕ} (qry : Fin q → (Fin q → D) → X)
    (hdep : ∀ i u u', (∀ j : Fin q, j < i → u j = u' j) → qry i u = qry i u') (h : X → D) :
    ∃ u, Consistent qry h u :=
  ⟨iterAns qry h q, fun i => (iterAns_succ qry hdep h q i i.isLt).symm⟩

open Classical in
/-- `h` with its values at the points `p i` replaced by `v i`. -/
noncomputable def overwrite {q : ℕ} (p : Fin q → X) (h : X → D) (v : Fin q → D) : X → D :=
  fun x => if hx : ∃ i, p i = x then v (Classical.choose hx) else h x

theorem overwrite_at {q : ℕ} {p : Fin q → X} (hp : Function.Injective p) (h : X → D) (v : Fin q → D) (i : Fin q) :
    overwrite p h v (p i) = v i := by
  have hx : ∃ j, p j = p i := ⟨i, rfl⟩
  simp only [overwrite, dif_pos hx]
  rw [hp (Classical.choose_spec hx)]

theorem overwrite_off {q : ℕ} (p : Fin q → X) (h : X → D) (v : Fin q → D) {x : X} (hx : ¬ ∃ i, p i = x) :
    overwrite p h v x = h x := by
  simp only [overwrite, dif_neg hx]

theorem overwrite_overwrite {q : ℕ} {p : Fin q → X} (hp : Function.Injective p) (h : X → D) (v v' : Fin q → D) :
    overwrite p (overwrite p h v) v' = overwrite p h v' := by
  funext x
  by_cases hx : ∃ i, p i = x
  · obtain ⟨i, rfl⟩ := hx; rw [overwrite_at hp, overwrite_at hp]
  · rw [overwrite_off p _ _ hx, overwrite_off p _ _ hx, overwrite_off p _ _ hx]

theorem overwrite_self {q : ℕ} {p : Fin q → X} (hp : Function.Injective p) (h : X → D) :
    overwrite p h (fun i => h (p i)) = h := by
  funext x
  by_cases hx : ∃ i, p i = x
  · obtain ⟨i, rfl⟩ := hx; rw [overwrite_at hp]
  · rw [overwrite_off p _ _ hx]

variable [Fintype D]

open Classical in
/-- The oracles that answer `u i` at `q` distinct points `p i` are a `1/|D|^q` share of all oracles. -/
theorem card_agree [Fintype X] [DecidableEq X] {q : ℕ} {p : Fin q → X} (hp : Function.Injective p)
    (u : Fin q → D) :
    (Finset.univ.filter fun h : X → D => ∀ i, h (p i) = u i).card * Fintype.card D ^ q = Fintype.card (X → D) := by
  let e : {h : X → D // ∀ i, h (p i) = u i} × (Fin q → D) ≃ (X → D) :=
    { toFun := fun hv => overwrite p hv.1.1 hv.2
      invFun := fun g => (⟨overwrite p g u, overwrite_at hp g u⟩, fun i => g (p i))
      left_inv := fun ⟨⟨h, hh⟩, v⟩ => by
        have hu : overwrite p h u = h := by
          conv_lhs => rw [show u = fun i => h (p i) from funext fun i => (hh i).symm]
          exact overwrite_self hp h
        simp only [overwrite_overwrite hp, hu, overwrite_at hp]
      right_inv := fun g => by
        simp only
        rw [overwrite_overwrite hp, overwrite_self hp] }
  rw [← Fintype.card_congr e, Fintype.card_prod, Fintype.card_subtype, Fintype.card_fun, Fintype.card_fin]

open Classical in
/-- **Lazy sampling.** Over a uniform oracle `h : X → D`, a querier with adaptive, distinct queries sees each answer
stream on a `1/|D|^q` share of the oracles, so it sees a stream in `S` on at most an `|S|/|D|^q` share. -/
theorem card_consistent_le [Fintype X] [DecidableEq X] {q : ℕ} (qry : Fin q → (Fin q → D) → X)
    (hfresh : ∀ u (i j : Fin q), j < i → qry i u ≠ qry j u) (S : Finset (Fin q → D)) :
    (Finset.univ.filter fun h : X → D => ∃ u ∈ S, Consistent qry h u).card * Fintype.card D ^ q ≤
      S.card * Fintype.card (X → D) := by
  have hinj : ∀ u, Function.Injective fun i => qry i u := fun u i j h => by
    rcases lt_trichotomy i j with hij | hij | hij
    · exact absurd h.symm (hfresh u j i hij)
    · exact hij
    · exact absurd h (hfresh u i j hij)
  have hsub : (Finset.univ.filter fun h : X → D => ∃ u ∈ S, Consistent qry h u) ⊆
      S.biUnion fun u => Finset.univ.filter fun h : X → D => ∀ i, h (qry i u) = u i := by
    intro h hh
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hh
    obtain ⟨u, hu, hc⟩ := hh
    exact Finset.mem_biUnion.mpr ⟨u, hu, Finset.mem_filter.mpr ⟨Finset.mem_univ _, fun i => (hc i).symm⟩⟩
  calc (Finset.univ.filter fun h : X → D => ∃ u ∈ S, Consistent qry h u).card * Fintype.card D ^ q
      ≤ (∑ u ∈ S, (Finset.univ.filter fun h : X → D => ∀ i, h (qry i u) = u i).card) * Fintype.card D ^ q :=
        Nat.mul_le_mul_right _ ((Finset.card_le_card hsub).trans Finset.card_biUnion_le)
    _ = ∑ u ∈ S, (Finset.univ.filter fun h : X → D => ∀ i, h (qry i u) = u i).card * Fintype.card D ^ q :=
        Finset.sum_mul _ _ _
    _ = ∑ _u ∈ S, Fintype.card (X → D) := Finset.sum_congr rfl fun u _ => card_agree (hinj u) u
    _ = S.card * Fintype.card (X → D) := by rw [Finset.sum_const, smul_eq_mul]

open Classical in
/-- **`extract_of_ro_roots` over a uniform oracle.** `R` roots over one querier `qry` (adaptive, distinct queries),
root `w` sent after the first `τ w` answers: the oracles `h : X → D` on whose answer stream extraction fails at some
root, or two answers repeat, are at most a `(q² + ∑_w q (2 τ_w + 1)) / |D|` share of all oracles. -/
theorem card_fails_oracle_le [DecidableEq D] [Fintype X] [DecidableEq X] [Inhabited D] {q R : ℕ} {τ : Fin R → ℕ}
    (E : Enc X D W) (qry : Fin q → (Fin q → D) → X) (hfresh : ∀ u (i j : Fin q), j < i → qry i u ≠ qry j u)
    (A : ∀ w, Prover X D q (τ w)) :
    (Finset.univ.filter fun h : X → D =>
        ∃ u, Consistent qry h u ∧ ((∃ w, ExtractFails E (A w) u) ∨ ¬ Function.Injective u)).card * Fintype.card D ≤
      (q * q + ∑ w, q * (2 * τ w + 1)) * Fintype.card (X → D) := by
  set S := Finset.univ.filter fun u => (∃ w, ExtractFails E (A w) u) ∨ ¬ Function.Injective u
  have hsub : (Finset.univ.filter fun h : X → D =>
        ∃ u, Consistent qry h u ∧ ((∃ w, ExtractFails E (A w) u) ∨ ¬ Function.Injective u)) ⊆
      Finset.univ.filter fun h : X → D => ∃ u ∈ S, Consistent qry h u := by
    intro h hh
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hh ⊢
    obtain ⟨u, hc, hf⟩ := hh
    exact ⟨u, by simpa [S] using hf, hc⟩
  have hpos : 0 < Fintype.card D ^ q := pow_pos Fintype.card_pos q
  have h1 := card_consistent_le qry hfresh S
  have h2 := extract_of_ro_roots E A
  rw [Fintype.card_fun, Fintype.card_fin] at h2
  refine Nat.le_of_mul_le_mul_right ?_ hpos
  calc (Finset.univ.filter fun h : X → D =>
          ∃ u, Consistent qry h u ∧ ((∃ w, ExtractFails E (A w) u) ∨ ¬ Function.Injective u)).card *
            Fintype.card D * Fintype.card D ^ q
      ≤ (Finset.univ.filter fun h : X → D => ∃ u ∈ S, Consistent qry h u).card * Fintype.card D ^ q *
          Fintype.card D := by
        rw [mul_right_comm]; exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (Finset.card_le_card hsub))
    _ ≤ S.card * Fintype.card (X → D) * Fintype.card D := Nat.mul_le_mul_right _ h1
    _ = S.card * Fintype.card D * Fintype.card (X → D) := by ring
    _ ≤ (q * q + ∑ w, q * (2 * τ w + 1)) * Fintype.card D ^ q * Fintype.card (X → D) :=
        Nat.mul_le_mul_right _ h2
    _ = (q * q + ∑ w, q * (2 * τ w + 1)) * Fintype.card (X → D) * Fintype.card D ^ q := by ring

end LazySampling

/-- An opening that verifies under `h` verifies under every `h'` that agrees with `h` on the points it hashes. -/
theorem opens_congr {h h' : X → D} (E : Enc X D W) :
    ∀ {d : D} {p : List (Bool × D × D)} {w : W}, (∀ y ∈ pts E p w, h y = h' y) → Opens h E d p w → Opens h' E d p w
  | d, [], w, hag, hO => by
    simp only [pts, List.mem_singleton, forall_eq] at hag
    simp only [Opens] at hO ⊢
    rw [← hag]; exact hO
  | d, (b, s, c) :: p, w, hag, hO => by
    simp only [pts, List.mem_cons, forall_eq_or_imp] at hag
    simp only [Opens] at hO ⊢
    exact ⟨hag.1 ▸ hO.1, opens_congr E hag.2 hO.2⟩

end Pouw.MerkleExtract
