import Definitions.Pouw.Assumptions.Dimension.H100Sem

/-!
# What each `sm_90` instruction class computes: proofs

* **The documented `k`s** (`TcKind.minK_mem`, `TcKind.minK_le`): each type's least `k` is one of its `k`s.
* **`MH100w` credits a word 32, whatever `L`** (`anyFunctionCeiling`): one `other` instruction a target writes every
  finite set of targets at `32` each. **The tensor-core premise alone buys nothing** (`tcOnlyCeiling`): that program
  has no tensor-core instruction, so it meets `Sm90TcOnly`.
* **The rank fact** (`innerProductRank`): an `L`-term inner product of fresh variables needs `L` rank-one terms.
* **The counting lemma's arithmetic** (`bamFloorArith`): `BamFloor`'s minimisation over the number of words a
  tensor-core register writes.
* **Range-forced purity** (`Lane.pure_of_fitsOn`, `pureLanes_of_wfOn`, `pureLanes_s8_rfp`): on noise spanning more than
  half a type's range, its lanes are pure; s8 lanes on noise in `[−64, 64]` are.
* **The covering count** (`coverAll`): on pure lanes a program costs at least the noise pairs its forms hold, since each
  is one product of a tensor-core register, and a type whose `k` exceeds its price is narrower than the noise
  (`Lane.noiseFree_of_fitsOn`). So the running words cost `N·L` on pure lanes (`coverCount`), and s8 programs on noise
  in `[−64, 64]` pay it with no purity hypothesis (`coverCountS8`).
-/

namespace Pouw.Dimension.H100

open Finset

variable {G : Type}

theorem TcKind.minK_mem (κ : TcKind) : κ.minK ∈ κ.ks := by cases κ <;> decide

theorem TcKind.minK_le (κ : TcKind) : ∀ k ∈ κ.ks, κ.minK ≤ k := by cases κ <;> decide

theorem regs100_map_other (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (l L : List (G → ℚ)) :
    regs100 fpSem (l.map Op100.other) L = L ++ l := by
  induction l generalizing L with
  | nil => simp [regs100]
  | cons f l ih =>
    have h := ih (L ++ [f])
    simp only [regs100, List.map_cons, List.foldl_cons, Op100.outs] at h ⊢
    rw [h, List.append_assoc, List.singleton_append]

theorem cost100_map_other (pr : Prices100) (l : List (G → ℚ)) :
    cost100 pr (l.map Op100.other) = pr.other * l.length := by
  induction l with
  | nil => simp [cost100]
  | cons f l ih =>
    simp only [cost100, List.map_map, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    rw [ih]; simp [Op100.price]; ring

/-- **`MH100w` credits a word 32, whatever `L`**: for every finite set of targets, writing each with one `other`
instruction is a well-formed program at `h100Floor` that outputs them all at exactly `32·|T|`. -/
theorem anyFunctionCeiling (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ))
    (L : List (G → ℚ)) :
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Floor F) ∧ (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧
      cost100 h100Floor P = 32 * T.card := by
  refine ⟨T.toList.map Op100.other, ?_, ?_, ?_⟩
  · intro o ho
    obtain ⟨f, -, rfl⟩ := List.mem_map.1 ho
    trivial
  · intro t ht
    rw [regs100_map_other]
    exact List.mem_append_right _ (Finset.mem_toList.2 ht)
  · rw [cost100_map_other, Finset.length_toList]
    rfl

/-- **The tensor-core premise alone buys nothing**: the same program meets it (it has no tensor-core instruction). -/
theorem tcOnlyCeiling (I : Sm90Isa) (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ))
    (L : List (G → ℚ)) :
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Floor F) ∧ Sm90TcOnly I fpSem F L P ∧
      (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧ cost100 h100Floor P = 32 * T.card := by
  refine ⟨T.toList.map Op100.other, ?_, ?_, ?_, ?_⟩
  · intro o ho
    obtain ⟨f, -, rfl⟩ := List.mem_map.1 ho
    trivial
  · intro i hi
    simp [Op100.TcOnly]
  · intro t ht
    rw [regs100_map_other]
    exact List.mem_append_right _ (Finset.mem_toList.2 ht)
  · rw [cost100_map_other, Finset.length_toList]
    rfl

/-- **The rank fact**: if `r` rank-one terms `α p ⊗ β p` sum to the identity on `L` coordinates, then `L ≤ r`. So an
`L`-term inner product of fresh variables needs `L` products, at most 32 to an accumulator register. -/
theorem innerProductRank (L r : ℕ) (α β : Fin r → Fin L → ℚ)
    (h : ∀ k l, ∑ p, α p k * β p l = if k = l then 1 else 0) : L ≤ r := by
  let A : Matrix (Fin L) (Fin r) ℚ := Matrix.of fun k p => α p k
  let B : Matrix (Fin r) (Fin L) ℚ := Matrix.of fun p l => β p l
  have hAB : A * B = 1 := by
    ext k l
    simp only [A, B, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
    exact h k l
  have h1 : (1 : Matrix (Fin L) (Fin L) ℚ).rank = L := by simp
  have h2 : (A * B).rank ≤ A.rank := Matrix.rank_mul_le_left A B
  have h3 : A.rank ≤ r := by simpa using Matrix.rank_le_width A
  rw [hAB, h1] at h2
  omega

/-- **The counting lemma's arithmetic**: with `p` of the `N` words written by a tensor-core register, `N` words and at least
`max(p, N − p)` other forms (`max(p/2, N − p)` when two words may share a pre-image) cost at least `48` a word
(`128/3`). -/
theorem bamFloorArith (N p : ℚ) :
    N * 48 ≤ 32 * (N + max p (N - p)) ∧ N * (128 / 3) ≤ 32 * (N + max (p / 2) (N - p)) := by
  have h1 := le_max_left p (N - p)
  have h2 := le_max_right p (N - p)
  have h3 := le_max_left (p / 2) (N - p)
  have h4 := le_max_right (p / 2) (N - p)
  constructor <;> linarith

/-- **Range-forced purity**: a lane that fits a type on noise in `[nlo, nhi]` carries at most one noise variable once
the type's range is narrower than `2·(nhi − nlo)`. Two variables with nonzero integer coefficients move the lane by at
least `2·(nhi − nlo)` between two corners of the noise box. -/
theorem Lane.pure_of_fitsOn {α : Type} [Fintype α] {nlo nhi : ℤ} {τ : BamType} {u : Lane α}
    (hle : nlo ≤ nhi) (hw : τ.hi - τ.lo < 2 * (nhi - nlo)) (hu : u.FitsOn nlo nhi τ) : u.Pure := by
  classical
  intro i j hi hj
  by_contra hij
  let lo1 : α → ℤ := fun k => if 0 < u (some k) then nlo else nhi
  let hi1 : α → ℤ := fun k => if 0 < u (some k) then nhi else nlo
  let e₁ : α → ℤ := fun k => if k = i ∨ k = j then lo1 k else nlo
  let e₂ : α → ℤ := fun k => if k = i ∨ k = j then hi1 k else nlo
  have hr₁ : ∀ k, nlo ≤ e₁ k ∧ e₁ k ≤ nhi := by
    intro k; simp only [e₁, lo1]; split_ifs <;> omega
  have hr₂ : ∀ k, nlo ≤ e₂ k ∧ e₂ k ≤ nhi := by
    intro k; simp only [e₂, hi1]; split_ifs <;> omega
  have gain : ∀ k, (k = i ∨ k = j) → nhi - nlo ≤ u (some k) * (e₂ k - e₁ k) := by
    intro k hk
    have hne : u (some k) ≠ 0 := by rcases hk with rfl | rfl <;> assumption
    simp only [e₁, e₂, lo1, hi1, hk, ↓reduceIte]
    split_ifs with hpos
    · nlinarith
    · have : u (some k) ≤ -1 := by omega
      nlinarith
  have hdiff : u.eval e₂ - u.eval e₁ = u (some i) * (e₂ i - e₁ i) + u (some j) * (e₂ j - e₁ j) := by
    have : u.eval e₂ - u.eval e₁ = ∑ k, u (some k) * (e₂ k - e₁ k) := by
      simp only [Lane.eval, mul_sub, sum_sub_distrib]; ring
    rw [this, Fintype.sum_eq_add i j hij]
    intro k ⟨hki, hkj⟩
    simp [e₁, e₂, hki, hkj]
  have h₁ := hu e₁ hr₁
  have h₂ := hu e₂ hr₂
  have gi := gain i (Or.inl rfl)
  have gj := gain j (Or.inr rfl)
  omega

/-- **Range-forced purity of a program**: on noise in `[nlo, nhi]`, a program whose every type is narrower than
`2·(nhi − nlo)` has pure lanes, so the covering count (`coverAll`) applies to it. -/
theorem pureLanes_of_wfOn {α β : Type} [Fintype α] [Fintype β] {nlo nhi : ℤ} {Ts : List BamType}
    (hle : nlo ≤ nhi) (hTs : ∀ τ ∈ Ts, τ.hi - τ.lo < 2 * (nhi - nlo)) (P : List (BamOp α β))
    (hP : ∀ o ∈ P, o.WFOn nlo nhi Ts) : PureLanes P := by
  intro o ho
  have h := hP o ho
  cases o with
  | tc τ src u v =>
    obtain ⟨hτ, hu, hv⟩ := h
    exact ⟨fun t => Lane.pure_of_fitsOn hle (hTs τ hτ) (hu t),
      fun t => Lane.pure_of_fitsOn hle (hTs τ hτ) (hv t)⟩
  | merge c => trivial
  | copy i => trivial
  | const q => trivial

/-- **s8 lanes on noise in `[−64, 64]` are pure**: the noise spans 129 values, so int7 data plus noise is an s8 value
and no s8 lane sums two noise variables. -/
theorem pureLanes_s8_rfp {α β : Type} [Fintype α] [Fintype β] (P : List (BamOp α β))
    (hP : ∀ o ∈ P, o.WFOn (-64) 64 [⟨32, -128, 127, 32⟩]) : PureLanes P :=
  pureLanes_of_wfOn (by norm_num) (by decide) P hP

section Cover

variable {α β : Type} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]

/-- The noise pairs a tensor-core instruction's products hold. -/
def BamOp.pairs : BamOp α β → Finset (α × β)
  | .tc _ _ u v => univ.filter fun p => ∃ t, (u t) (some p.1) ≠ 0 ∧ (v t) (some p.2) ≠ 0
  | _ => ∅

/-- The noise pairs a program's products hold. -/
def progPairs (P : List (BamOp α β)) : Finset (α × β) := P.foldr (fun o s => o.pairs ∪ s) ∅

omit [DecidableEq α] [DecidableEq β] in
theorem noiseSupp_getD {Rs : List (BForm α β)} {T : Finset (α × β)} (hRs : ∀ R ∈ Rs, noiseSupp R ⊆ T) (i : ℕ) :
    noiseSupp (Rs.getD i 0) ⊆ T := by
  rcases lt_or_ge i Rs.length with h | h
  · rw [List.getD_eq_getElem _ _ h]; exact hRs _ (List.getElem_mem h)
  · rw [List.getD_eq_default _ _ h]; intro p hp; simp [noiseSupp] at hp

omit [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β] in
theorem list_sum_entry_ne {l : List (BForm α β)} {x : Option α} {y : Option β} (h : l.sum x y ≠ 0) :
    ∃ M ∈ l, M x y ≠ 0 := by
  induction l with
  | nil => simp at h
  | cons M l ih =>
    rw [List.sum_cons, Matrix.add_apply] at h
    by_cases hM : M x y = 0
    · rw [hM, zero_add] at h
      obtain ⟨M', hM', h'⟩ := ih h
      exact ⟨M', List.mem_cons_of_mem _ hM', h'⟩
    · exact ⟨M, List.mem_cons_self .., hM⟩

theorem noiseSupp_out {Rs : List (BForm α β)} {T : Finset (α × β)} (hRs : ∀ R ∈ Rs, noiseSupp R ⊆ T)
    (o : BamOp α β) (ho : o.pairs ⊆ T) : noiseSupp (o.out Rs) ⊆ T := by
  intro p hp
  simp only [noiseSupp, mem_filter, mem_univ, true_and] at hp
  cases o with
  | tc τ src u v =>
    simp only [BamOp.out, Matrix.add_apply, Matrix.sum_apply, Matrix.vecMulVec_apply, Lane.vec] at hp
    by_cases h1 : (Rs.getD src 0) (some p.1) (some p.2) = 0
    · have hex : ∃ t, (u t) (some p.1) ≠ 0 ∧ (v t) (some p.2) ≠ 0 := by
        by_contra hne
        push Not at hne
        apply hp
        rw [h1, zero_add]
        apply Finset.sum_eq_zero
        intro t _
        by_cases hu : (u t) (some p.1) = 0
        · simp [hu]
        · simp [hne t hu]
      exact ho (by simp only [BamOp.pairs, mem_filter, mem_univ, true_and]; exact hex)
    · exact noiseSupp_getD hRs src (by simp only [noiseSupp, mem_filter, mem_univ, true_and]; exact h1)
  | merge c =>
    simp only [BamOp.out] at hp
    obtain ⟨M, hM, hne⟩ := list_sum_entry_ne hp
    obtain ⟨q, -, rfl⟩ := List.mem_map.1 hM
    rw [Matrix.smul_apply, smul_eq_mul] at hne
    exact noiseSupp_getD hRs q.2 (by
      simp only [noiseSupp, mem_filter, mem_univ, true_and]; exact right_ne_zero_of_mul hne)
  | copy i =>
    exact noiseSupp_getD hRs i (by simp only [noiseSupp, mem_filter, mem_univ, true_and]; exact hp)
  | const q =>
    simp [BamOp.out] at hp

theorem noiseSupp_foldl {T : Finset (α × β)} (P : List (BamOp α β)) (hP : ∀ o ∈ P, o.pairs ⊆ T) :
    ∀ Rs : List (BForm α β), (∀ R ∈ Rs, noiseSupp R ⊆ T) →
      ∀ R ∈ P.foldl (fun Rs o => Rs ++ [o.out Rs]) Rs, noiseSupp R ⊆ T := by
  induction P with
  | nil => intro Rs hRs; simpa using hRs
  | cons o P ih =>
    intro Rs hRs
    rw [List.foldl_cons]
    apply ih (fun o' h => hP o' (List.mem_cons_of_mem _ h))
    intro R hR
    rcases List.mem_append.1 hR with h | h
    · exact hRs R h
    · rw [List.mem_singleton] at h
      subst h
      exact noiseSupp_out hRs o (hP o List.mem_cons_self ..)

theorem pairs_sub_progPairs {P : List (BamOp α β)} {o : BamOp α β} (h : o ∈ P) : o.pairs ⊆ progPairs P := by
  induction P with
  | nil => simp at h
  | cons o' P ih =>
    simp only [progPairs, List.foldr_cons] at ih ⊢
    rcases List.mem_cons.1 h with rfl | h
    · exact subset_union_left
    · exact (ih h).trans subset_union_right

theorem card_progPairs_le (P : List (BamOp α β)) :
    (progPairs P).card ≤ (P.map fun o => o.pairs.card).sum := by
  induction P with
  | nil => simp [progPairs]
  | cons o P ih =>
    simp only [progPairs, List.foldr_cons, List.map_cons, List.sum_cons] at ih ⊢
    exact (card_union_le _ _).trans (by omega)

/-- A lane that fits a type narrower than the noise's span carries no noise variable. -/
theorem Lane.noiseFree_of_fitsOn {γ : Type} [Fintype γ] {nlo nhi : ℤ} {τ : BamType} {u : Lane γ}
    (hle : nlo ≤ nhi) (hw : τ.hi - τ.lo < nhi - nlo) (hu : u.FitsOn nlo nhi τ) (a : γ) : u (some a) = 0 := by
  classical
  by_contra ha
  let e₁ : γ → ℤ := fun k => if k = a then (if 0 < u (some a) then nlo else nhi) else nlo
  let e₂ : γ → ℤ := fun k => if k = a then (if 0 < u (some a) then nhi else nlo) else nlo
  have hr₁ : ∀ k, nlo ≤ e₁ k ∧ e₁ k ≤ nhi := by intro k; simp only [e₁]; split_ifs <;> omega
  have hr₂ : ∀ k, nlo ≤ e₂ k ∧ e₂ k ≤ nhi := by intro k; simp only [e₂]; split_ifs <;> omega
  have hdiff : u.eval e₂ - u.eval e₁ = u (some a) * (e₂ a - e₁ a) := by
    have : u.eval e₂ - u.eval e₁ = ∑ k, u (some k) * (e₂ k - e₁ k) := by
      simp only [Lane.eval, mul_sub, sum_sub_distrib]; ring
    rw [this, Finset.sum_eq_single a]
    · intro k _ hk; simp [e₁, e₂, hk]
    · simp
  have hg : nhi - nlo ≤ u (some a) * (e₂ a - e₁ a) := by
    simp only [e₁, e₂, ↓reduceIte]
    split_ifs with hpos
    · nlinarith
    · have : u (some a) ≤ -1 := by omega
      nlinarith
  have h₁ := hu e₁ hr₁
  have h₂ := hu e₂ hr₂
  omega

theorem h200Types_k_or_narrow : ∀ τ ∈ h200Types, τ.k ≤ τ.price ∨ τ.hi - τ.lo ≤ 15 := by decide

theorem card_pairs_le {nlo nhi : ℤ} (hle : nlo ≤ nhi) (hspan : 15 < nhi - nlo) (o : BamOp α β)
    (hWF : o.WFOn nlo nhi h200Types) (hP : PureLanes [o]) : o.pairs.card ≤ o.price := by
  cases o with
  | tc τ src u v =>
    obtain ⟨hτ, hu, hv⟩ := hWF
    obtain ⟨hpu, hpv⟩ := hP _ List.mem_cons_self
    rcases h200Types_k_or_narrow τ hτ with hk | hn
    · have hsub : (BamOp.tc τ src u v).pairs ⊆ (univ : Finset (Fin τ.k)).biUnion
          (fun t => univ.filter fun p : α × β => (u t) (some p.1) ≠ 0 ∧ (v t) (some p.2) ≠ 0) := by
        intro p hp
        simp only [BamOp.pairs, mem_filter, mem_univ, true_and] at hp
        obtain ⟨t, ht⟩ := hp
        exact mem_biUnion.2 ⟨t, mem_univ _, by simp only [mem_filter, mem_univ, true_and]; exact ht⟩
      have hone : ∀ t : Fin τ.k, (univ.filter fun p : α × β => (u t) (some p.1) ≠ 0 ∧
          (v t) (some p.2) ≠ 0).card ≤ 1 := by
        intro t
        apply Finset.card_le_one.2
        intro p hp q hq
        simp only [mem_filter, mem_univ, true_and] at hp hq
        exact Prod.ext (hpu t _ _ hp.1 hq.1) (hpv t _ _ hp.2 hq.2)
      calc (BamOp.tc τ src u v).pairs.card ≤ _ := card_le_card hsub
        _ ≤ ∑ t : Fin τ.k, 1 := card_biUnion_le.trans (sum_le_sum fun t _ => hone t)
        _ = τ.k := by simp
        _ ≤ τ.price := hk
    · have hemp : (BamOp.tc τ src u v).pairs = ∅ := by
        apply Finset.eq_empty_of_forall_notMem
        intro p hp
        simp only [BamOp.pairs, mem_filter, mem_univ, true_and] at hp
        obtain ⟨t, ht, -⟩ := hp
        exact ht (Lane.noiseFree_of_fitsOn hle (by omega) (hu t) p.1)
      simp [hemp]
  | merge c => simp [BamOp.pairs]
  | copy i => simp [BamOp.pairs]
  | const q => simp [BamOp.pairs]

theorem cover_card_le {nlo nhi : ℤ} (hle : nlo ≤ nhi) (hspan : 15 < nhi - nlo) (P : List (BamOp α β))
    (hWF : ∀ o ∈ P, o.WFOn nlo nhi h200Types) (hP : PureLanes P) (S : Finset (BForm α β))
    (hS : ∀ f ∈ S, f ∈ bamRegs P) : (S.biUnion noiseSupp).card ≤ bamCost P := by
  have hsub : S.biUnion noiseSupp ⊆ progPairs P := by
    intro p hp
    obtain ⟨f, hf, hpf⟩ := mem_biUnion.1 hp
    exact noiseSupp_foldl P (fun o h => pairs_sub_progPairs h) [] (by simp) f (hS f hf) hpf
  calc (S.biUnion noiseSupp).card ≤ (progPairs P).card := card_le_card hsub
    _ ≤ (P.map fun o => o.pairs.card).sum := card_progPairs_le P
    _ ≤ (P.map BamOp.price).sum := List.sum_le_sum fun o ho =>
        card_pairs_le hle hspan o (hWF o ho) (fun o' h => hP o' (by simp at h; exact h ▸ ho))
    _ = bamCost P := rfl

end Cover

/-- **The covering count** (`CoverAll`): on noise spanning more than 15 and pure lanes, a well-formed program at the
H200's types costs at least the number of distinct noise pairs the forms it writes hold. -/
theorem coverAll : CoverAll := fun _ _ _ _ _ _ _ _ P S hle hspan hWF hP hS => cover_card_le hle hspan P hWF hP S hS

/-- The running words' forms hold all `m·n·K` noise pairs (the cells' last words do), so a program costing at least
their number costs at least `N·L` for `N = m·n·(K/L)` words. -/
theorem runForms_count {m K n L : ℕ} (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (c : ℕ) (hLK : L ∣ K)
    (hcard : ((runForms a b L).biUnion noiseSupp).card ≤ c) : (m * n * (K / L) : ℚ) * L ≤ c := by
  rcases Nat.eq_zero_or_pos L with rfl | hL
  · simp
  rcases Nat.eq_zero_or_pos K with rfl | hK
  · simp
  have hJ : 0 < K / L := Nat.div_pos (Nat.le_of_dvd hK hLK) hL
  have hlast : (K / L - 1 + 1) * L = K := by rw [Nat.sub_add_cancel hJ, Nat.div_mul_cancel hLK]
  let g : Fin m × Fin K × Fin n → (Fin m × Fin K) × (Fin K × Fin n) := fun x => ((x.1, x.2.1), (x.2.1, x.2.2))
  have hinj : Function.Injective g := by
    intro x y h
    simp only [g, Prod.mk.injEq] at h
    obtain ⟨⟨h1, h2⟩, -, h3⟩ := h
    exact Prod.ext h1 (Prod.ext h2 h3)
  have hsub : (univ : Finset (Fin m × Fin K × Fin n)).image g ⊆ (runForms a b L).biUnion noiseSupp := by
    intro p hp
    obtain ⟨⟨r, i, c⟩, -, rfl⟩ := mem_image.1 hp
    refine mem_biUnion.2 ⟨runForm a b L r c (K / L - 1), ?_, ?_⟩
    · refine mem_image.2 ⟨(r, c, K / L - 1), ?_, rfl⟩
      simp only [mem_product, mem_univ, mem_range, true_and]
      omega
    · unfold noiseSupp
      refine mem_filter.2 ⟨mem_univ _, ?_⟩
      show (runForm a b L r c (K / L - 1)) (some (r, i)) (some (i, c)) ≠ 0
      have hi : i.val < (K / L - 1 + 1) * L := by rw [hlast]; exact i.isLt
      simp only [runForm, Matrix.of_apply, true_and, hi, ↓reduceIte]
      exact one_ne_zero
  have hmnK : m * K * n ≤ c := by
    have := card_le_card hsub
    rw [card_image_of_injective _ hinj] at this
    simpa [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, mul_assoc] using this.trans hcard
  have hL0 : (L : ℚ) ≠ 0 := by exact_mod_cast hL.ne'
  calc (m * n * (K / L) : ℚ) * L = (m * K * n : ℕ) := by
        rw [mul_assoc, div_mul_cancel₀ _ hL0]; push_cast; ring
    _ ≤ c := by exact_mod_cast hmnK

/-- **`CoverCount` holds**: on pure lanes the running words cost `N·L`, by the covering count. -/
theorem coverCount (L : ℕ) : CoverCount L := fun _ _ _ a b P hLK hWF hP hS =>
  runForms_count a b _ hLK (cover_card_le (nlo := -32) (nhi := 31) (by norm_num) (by norm_num) P hWF hP _ hS)

theorem BamOp.wfOn_mono {α β : Type} [Fintype α] [Fintype β] {nlo nhi : ℤ} {Ts Ts' : List BamType}
    (h : Ts ⊆ Ts') : ∀ o : BamOp α β, o.WFOn nlo nhi Ts → o.WFOn nlo nhi Ts'
  | .tc .., ⟨hτ, hu, hv⟩ => ⟨h hτ, hu, hv⟩
  | .merge _, hc => hc
  | .copy _, _ => trivial
  | .const _, _ => trivial

/-- **`CoverCountS8` holds**: on noise in `[−64, 64]` s8 lanes are pure (`pureLanes_s8_rfp`), so the covering count
gives s8 programs the full `N·L` with no purity hypothesis. -/
theorem coverCountS8 (L : ℕ) : CoverCountS8 L := fun _ _ _ a b P hLK hWF hS =>
  runForms_count a b _ hLK (cover_card_le (nlo := -64) (nhi := 64) (by norm_num) (by norm_num) P
    (fun o ho => BamOp.wfOn_mono (fun τ hτ => by rw [List.mem_singleton.1 hτ]; exact List.mem_cons_self) o
      (hWF o ho))
    (pureLanes_s8_rfp P hWF) _ hS)

end Pouw.Dimension.H100
