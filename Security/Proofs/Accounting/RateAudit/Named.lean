import Mathlib
import Definitions.Accounting.Guarantees.RateAuditNamed
import Proofs.Accounting.RateAudit.Conservation
import Proofs.Accounting.RateAudit.Horizon

/-!
# Conservation over tiers with named words: the proof

A run with at least `k` correct receipts falls in one of three events:

- a correct receipt for a word it never queried: `N/2^h` (`RelayDevice.pr_guess_le`);
- two of its queries with one oracle value (`Collides`): `C(q, 2)/2^h`. At each pair of positions `i < j` with query
  `j` fresh, changing the oracle at query `j`'s point changes none of the queries up to `j` (`Chip.queries_take_congr`),
  so the oracles where `i` and `j` collide meet each line through that point at most once (`card_collidesAt_le`);
- otherwise every correct receipt's word was queried, and only once with that oracle value. Replaying the chip from
  its state and its trace recomputes its queries and its receipts (`Chip.replayRun_trace`), so naming `k` correct words
  locates their contents: at the first replayed query at the word whose oracle value is its receipt (`namedAt`). The
  code is the state, the trace and `k` of the `N` words: `C(N, k)/2^λ` (`card_compress_named_le`).

At every count at once (`RatePaidNamedRO`), the compression is taken at each `k ≤ N` and `∑ₖ C(N, k) = 2^N`
(`RelayDevice.pr_compress_any_named_le`); the route `Sampled` with it (`RateChainSampledNamedRO`) is `RateHorizon` and
that bound at each pass (`Chain.pr_unpaid_le`).
-/

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (card_filter_prod_le card_filter_prod_le' pr_le_of_card pr_mono pr_or_le pr_exists_le)

theorem not_mem_take_idxOf {α : Type} [BEq α] [LawfulBEq α] (y : α) : ∀ l : List α, y ∉ l.take (l.idxOf y)
  | [] => by simp
  | a :: l => by
    by_cases ha : a = y
    · subst ha
      simp
    · have := not_mem_take_idxOf y l
      simp [ha, this, Ne.symm ha]

theorem card_pairs (q : ℕ) : Fintype.card (Σ j : Fin q, Fin j.val) = q.choose 2 := by
  rw [Fintype.card_sigma]
  simp only [Fintype.card_fin]
  rw [Fin.sum_univ_eq_sum_range (fun k => k) q, Nat.choose_two_right, ← Finset.sum_range_id_mul_two,
    Nat.mul_div_cancel _ (by norm_num)]

namespace Chip

variable {N b h ρ : ℕ} {α : Type}

/-- A run's value, recomputed from the oracle and a stream of the bits it was answered with otherwise, as `replay`
recomputes its queries. -/
def replayRun (O : Oracle N b h ρ) : Chip N b h ρ α → (ℕ → Option Bool) → (ℕ → Bool) → α
  | done a, _, _ => a
  | tag x k, m, f => (k (O x)).replayRun O m f
  | ingress a k, m, f =>
    match m a with
    | some v => (k v).replayRun O m f
    | none => (k (f 0)).replayRun O (Function.update m a (some (f 0))) (fun n => f (n + 1))
  | side _ k, m, f => (k (f 0)).replayRun O m (fun n => f (n + 1))
  | stream _ k, m, f => (k (f 0)).replayRun O m (fun n => f (n + 1))
  | rebuild _ k, m, f => (k (fun i => f i)).replayRun O m (fun n => f (n + b))

theorem replayRun_trace (w : World N b h ρ) (c : Chip N b h ρ α) (seen : Finset ℕ) (m : ℕ → Option Bool)
    (hm : ∀ a, m a = if a ∈ seen then some (w.mem a) else none) (l : List Bool) :
    c.replayRun w.oracle m (fun n => (c.trace w seen ++ l).getD n false) = c.run w := by
  induction c generalizing seen m l with
  | done a => simp [replayRun, run]
  | tag x k ih => simp only [replayRun, run, trace, ih _ seen m hm l]
  | ingress a k ih =>
    by_cases ha : a ∈ seen
    · have hma : m a = some (w.mem a) := by rw [hm]; simp [ha]
      simp only [replayRun, trace, run, ha, ↓reduceIte, hma]
      exact ih _ seen m hm l
    · have hma : m a = none := by rw [hm]; simp [ha]
      simp only [replayRun, trace, run, ha, ↓reduceIte, hma, List.cons_append, List.getD_cons_zero,
        List.getD_cons_succ]
      refine ih _ (insert a seen) _ (fun a' => ?_) l
      by_cases h : a' = a
      · subst h; simp
      · simp [hm, h]
  | side a k ih => simpa [replayRun, run, trace] using ih _ seen m hm l
  | stream i k ih => simpa [replayRun, run, trace] using ih _ seen m hm l
  | rebuild j k ih =>
    simp only [replayRun, run, trace, List.append_assoc]
    have hb : (fun i : Fin b => (List.ofFn (w.store j) ++ ((k (w.store j)).trace w seen ++ l)).getD i false) =
        w.store j := by
      funext i
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_left (by simp)]
      simp
    have hs : (fun n => (List.ofFn (w.store j) ++ ((k (w.store j)).trace w seen ++ l)).getD (n + b) false) =
        fun n => ((k (w.store j)).trace w seen ++ l).getD n false := by
      funext n
      rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by simp), List.getD_eq_getElem?_getD]
      simp
    rw [hb, hs, ih _ seen m hm l]

/-- Changing the oracle off a run's first `n` queries changes none of its first `n + 1`. -/
theorem queries_take_congr (w : World N b h ρ) (O' : Oracle N b h ρ) (c : Chip N b h ρ α) (n : ℕ)
    (ho : ∀ x ∈ (c.queries w).take n, O' x = w.oracle x) :
    (c.queries { w with oracle := O' }).take (n + 1) = (c.queries w).take (n + 1) := by
  induction c generalizing n with
  | done a => simp [queries]
  | tag x k ih =>
    cases n with
    | zero => simp [queries]
    | succ n =>
      simp only [queries, List.take_succ_cons, List.mem_cons, forall_eq_or_imp] at ho ⊢
      obtain ⟨hx, ho⟩ := ho
      rw [hx, ih _ n ho]
  | ingress a k ih => exact ih _ n ho
  | side a k ih => exact ih _ n ho
  | stream i k ih => exact ih _ n ho
  | rebuild j k ih => exact ih _ n ho

end Chip

variable {N b h ρ S L q : ℕ}

/-- Two distinct points among a run's queries with one oracle value. -/
def Collides (O : Oracle N b h ρ) (qs : List (Bits ρ × Fin N × Bits b)) : Prop :=
  ∃ x ∈ qs, ∃ y ∈ qs, x ≠ y ∧ O x = O y

/-- Queries `i` and `j` have one oracle value at distinct points, query `j`'s point a first query. -/
def CollidesAt (O : Oracle N b h ρ) (qs : List (Bits ρ × Fin N × Bits b)) (i j : ℕ) : Prop :=
  ∃ x z, qs[j]? = some x ∧ qs[i]? = some z ∧ x ∉ qs.take j ∧ z ≠ x ∧ O z = O x

/-- A collision among at most `q` queries is one at a pair of positions `i < j < q`, at the points' first queries. -/
theorem collidesAt_of_collides {O : Oracle N b h ρ} {qs : List (Bits ρ × Fin N × Bits b)} (hq : qs.length ≤ q)
    (h : Collides O qs) : ∃ p : Σ j : Fin q, Fin j.val, CollidesAt O qs p.2.val p.1.val := by
  classical
  obtain ⟨x, hx, y, hy, hne, hO⟩ := h
  have key : ∀ x y, x ∈ qs → y ∈ qs → x ≠ y → O x = O y → qs.idxOf x < qs.idxOf y →
      ∃ p : Σ j : Fin q, Fin j.val, CollidesAt O qs p.2.val p.1.val := by
    intro x y hx hy hne hO hlt
    have hj : qs.idxOf y < q := (List.idxOf_lt_length_of_mem hy).trans_le hq
    exact ⟨⟨⟨qs.idxOf y, hj⟩, ⟨qs.idxOf x, hlt⟩⟩, y, x, List.getElem?_idxOf hy, List.getElem?_idxOf hx,
      not_mem_take_idxOf y qs, hne, hO⟩
  rcases lt_trichotomy (qs.idxOf x) (qs.idxOf y) with hlt | heq | hgt
  · exact key x y hx hy hne hO hlt
  · have : some x = some y := by rw [← List.getElem?_idxOf hx, ← List.getElem?_idxOf hy, heq]
    exact absurd (Option.some.inj this) hne
  · exact key y x hy hx hne.symm hO.symm hgt

/-- The oracles under which a run's queries `i < j` collide, query `j` fresh, are at most a `2^-h` fraction. -/
theorem card_collidesAt_le {α : Type} (c : Chip N b h ρ α) (w : World N b h ρ) (i j : ℕ) (hij : i < j)
    [DecidablePred (fun O : Oracle N b h ρ => CollidesAt O (c.queries { w with oracle := O }) i j)] :
    (Finset.univ.filter (fun O : Oracle N b h ρ => CollidesAt O (c.queries { w with oracle := O }) i j)).card *
        2 ^ h ≤ Fintype.card (Oracle N b h ρ) := by
  classical
  have hY : Fintype.card (Bits h) = 2 ^ h := by simp
  let Φ : Oracle N b h ρ × Bits h → Oracle N b h ρ := fun p =>
    ((c.queries { w with oracle := p.1 })[j]?).elim p.1 (fun x => Function.update p.1 x p.2)
  rw [← hY, ← Finset.card_univ (α := Bits h), ← Finset.card_product, ← Finset.card_univ (α := Oracle N b h ρ)]
  refine Finset.card_le_card_of_injOn Φ (fun _ _ => Finset.mem_univ _) ?_
  rintro ⟨O₁, v₁⟩ h₁ ⟨O₂, v₂⟩ h₂ he
  simp only [Finset.coe_product, Set.mem_prod, Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and] at h₁ h₂
  obtain ⟨⟨x₁, z₁, hx₁, hz₁, hf₁, hne₁, hO₁⟩, -⟩ := h₁
  obtain ⟨⟨x₂, z₂, hx₂, hz₂, hf₂, hne₂, hO₂⟩, -⟩ := h₂
  simp only [Φ, hx₁, hx₂, Option.elim] at he
  have hagree : ∀ (O : Oracle N b h ρ) x v, x ∉ (c.queries { w with oracle := O }).take j →
      (c.queries { w with oracle := Function.update O x v }).take (j + 1) =
        (c.queries { w with oracle := O }).take (j + 1) := fun O x v hx =>
    Chip.queries_take_congr { w with oracle := O } (Function.update O x v) c j
      (fun y hy => Function.update_of_ne (by rintro rfl; exact hx hy) _ _)
  have t₁ := hagree O₁ x₁ v₁ hf₁
  have t₂ := hagree O₂ x₂ v₂ hf₂
  rw [← he] at t₂
  have at_ : ∀ k, k < j + 1 →
      (c.queries { w with oracle := O₁ })[k]? = (c.queries { w with oracle := O₂ })[k]? := by
    intro k hk
    calc (c.queries { w with oracle := O₁ })[k]? = ((c.queries { w with oracle := O₁ }).take (j + 1))[k]? :=
          (List.getElem?_take_of_lt hk).symm
      _ = ((c.queries { w with oracle := Function.update O₁ x₁ v₁ }).take (j + 1))[k]? := by rw [t₁]
      _ = ((c.queries { w with oracle := O₂ }).take (j + 1))[k]? := by rw [t₂]
      _ = (c.queries { w with oracle := O₂ })[k]? := List.getElem?_take_of_lt hk
  have hx : x₁ = x₂ := Option.some.inj (hx₁.symm.trans ((at_ j (by omega)).trans hx₂))
  have hz : z₁ = z₂ := Option.some.inj (hz₁.symm.trans ((at_ i (by omega)).trans hz₂))
  subst hx
  subst hz
  have hv : v₁ = v₂ := by simpa using congrFun he x₁
  subst hv
  refine Prod.ext (funext fun y => ?_) rfl
  show O₁ y = O₂ y
  by_cases hy : y = x₁
  · rw [hy, ← hO₁, ← hO₂]
    calc O₁ z₁ = Function.update O₁ x₁ v₁ z₁ := (Function.update_of_ne hne₁ _ _).symm
      _ = Function.update O₂ x₁ v₁ z₁ := by rw [he]
      _ = O₂ z₁ := Function.update_of_ne hne₂ _ _
  · simpa [Function.update_of_ne hy] using congrFun he y

/-- The oracles under which a run of at most `q` queries has a collision are at most a `C(q, 2)·2^-h` fraction. -/
theorem card_collides_le {α : Type} (c : Chip N b h ρ α) (w : World N b h ρ)
    (hq : ∀ O, (c.queries { w with oracle := O }).length ≤ q)
    [DecidablePred (fun O : Oracle N b h ρ => Collides O (c.queries { w with oracle := O }))] :
    (Finset.univ.filter (fun O : Oracle N b h ρ => Collides O (c.queries { w with oracle := O }))).card * 2 ^ h ≤
      q.choose 2 * Fintype.card (Oracle N b h ρ) := by
  classical
  have hsub : Finset.univ.filter (fun O : Oracle N b h ρ => Collides O (c.queries { w with oracle := O })) ⊆
      (Finset.univ : Finset (Σ j : Fin q, Fin j.val)).biUnion (fun p =>
        Finset.univ.filter (fun O : Oracle N b h ρ =>
          CollidesAt O (c.queries { w with oracle := O }) p.2.val p.1.val)) := by
    intro O hO
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hO
    obtain ⟨p, hp⟩ := collidesAt_of_collides (hq O) hO
    exact Finset.mem_biUnion.2 ⟨p, Finset.mem_univ _, by simpa using hp⟩
  calc (Finset.univ.filter (fun O : Oracle N b h ρ => Collides O (c.queries { w with oracle := O }))).card * 2 ^ h
      ≤ (∑ p : Σ j : Fin q, Fin j.val, (Finset.univ.filter (fun O : Oracle N b h ρ =>
          CollidesAt O (c.queries { w with oracle := O }) p.2.val p.1.val)).card) * 2 ^ h := by
        gcongr
        rw [Finset.filter_congr_decidable]
        exact (Finset.card_le_card hsub).trans Finset.card_biUnion_le
    _ = ∑ p : Σ j : Fin q, Fin j.val, (Finset.univ.filter (fun O : Oracle N b h ρ =>
          CollidesAt O (c.queries { w with oracle := O }) p.2.val p.1.val)).card * 2 ^ h := Finset.sum_mul _ _ _
    _ ≤ ∑ _p : Σ j : Fin q, Fin j.val, Fintype.card (Oracle N b h ρ) :=
        Finset.sum_le_sum (fun p _ => card_collidesAt_le c w _ _ p.2.isLt)
    _ = q.choose 2 * Fintype.card (Oracle N b h ρ) := by
        rw [Finset.sum_const, Finset.card_univ, card_pairs, smul_eq_mul]

namespace RelayDevice

/-- With side sources that ignore the oracle, a collision among the run's queries has probability at most
`C(q, 2)/2^h`. -/
theorem pr_collides_le (A : RelayDevice N b h ρ S L q) (hA : ∀ E R O O', A.side E R O = A.side E R O') :
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        Collides x.2.2 ((A.chip (A.state x.1)).queries (A.world x.1 x.2.1 x.2.2)))
      ≤ (q.choose 2 : ℝ≥0∞) / 2 ^ h := by
  classical
  refine (pr_le_of_card _ (2 ^ h) (q.choose 2) (by positivity) ?_).trans (le_of_eq (by push_cast; rfl))
  calc _ ≤ Fintype.card (Store N b) * (Fintype.card (Bits ρ) * (q.choose 2 * Fintype.card (Oracle N b h ρ))) := by
        refine card_filter_prod_le _ _ _ (fun E => card_filter_prod_le _ _ _ (fun R => ?_))
        let O₀ : Oracle N b h ρ := fun _ _ => false
        have hw : ∀ O, A.world E R O = { A.world E R O₀ with oracle := O } := fun O => by
          simp only [RelayDevice.world, hA E R O O₀]
        have hq : ∀ O, ((A.chip (A.state E)).queries { A.world E R O₀ with oracle := O }).length ≤ q :=
          fun O => (A.bounded _ _).2
        refine le_trans (le_of_eq ?_) (card_collides_le (A.chip (A.state E)) (A.world E R O₀) hq)
        congr 2
        exact Finset.filter_congr (fun O _ => by simp only [hw O])
    _ = q.choose 2 * Fintype.card (Store N b × Bits ρ × Oracle N b h ρ) := by
        simp only [Fintype.card_prod]
        ring

/-- The queries and receipts a code (the chip's state, a padded trace and a set of named words) replays to. -/
def decodedRun (A : RelayDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ)
    (code : Bits S × Bits r × Finset (Fin N)) : List (Bits ρ × Fin N × Bits b) × (Fin N → Bits h) :=
  ((A.chip code.1).replay O (fun _ => none) (Bandwidth.AuditRO.stream r code.2.1),
    (A.chip code.1).replayRun O (fun _ => none) (Bandwidth.AuditRO.stream r code.2.1))

/-- The query a code locates word `j` at: the first replayed query at `j` whose oracle value is `j`'s replayed
receipt. -/
def namedAt (A : RelayDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ) (code : Bits S × Bits r × Finset (Fin N))
    (j : Fin N) : Option (Bits ρ × Fin N × Bits b) :=
  (decodedRun A O r code).1.find? (fun y => decide (y.2.1 = j ∧ O y = (decodedRun A O r code).2 j))

open Classical in
/-- The stores and streams a code describes: at each named word, the stream and the contents of the query it is
located at. -/
noncomputable def fibNamed (A : RelayDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ)
    (code : Bits S × Bits r × Finset (Fin N)) : Finset (Store N b × Bits ρ) :=
  Finset.univ.filter (fun x => ∀ j ∈ code.2.2, ∃ y, namedAt A O r code j = some y ∧ y.1 = x.2 ∧ x.1 j = y.2.2)

/-- A code that names a word fixes the stream, and the words it names. -/
theorem card_fibNamed (A : RelayDevice N b h ρ S L q) (O : Oracle N b h ρ) (r : ℕ)
    (code : Bits S × Bits r × Finset (Fin N)) (hn : 0 < code.2.2.card) :
    (fibNamed A O r code).card ≤ (2 ^ b) ^ (N - code.2.2.card) := by
  classical
  obtain ⟨j₀, hj₀⟩ := Finset.card_pos.1 hn
  calc (fibNamed A O r code).card ≤ (Finset.univ : Finset (↥(code.2.2)ᶜ → Bits b)).card := by
        refine Finset.card_le_card_of_injOn (fun x j => x.1 j) (fun _ _ => Finset.mem_univ _) ?_
        intro x₁ h₁ x₂ h₂ he
        simp only [fibNamed, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at h₁ h₂
        obtain ⟨y₀, hy₀, hs₁, -⟩ := h₁ j₀ hj₀
        obtain ⟨y₀', hy₀', hs₂, -⟩ := h₂ j₀ hj₀
        rw [hy₀] at hy₀'
        cases hy₀'
        refine Prod.ext (funext fun j => ?_) (hs₁.symm.trans hs₂)
        by_cases hj : j ∈ code.2.2
        · obtain ⟨y, hy, -, e₁⟩ := h₁ j hj
          obtain ⟨y', hy', -, e₂⟩ := h₂ j hj
          rw [hy] at hy'
          cases hy'
          rw [e₁, e₂]
        · exact congrFun he ⟨j, by simpa using hj⟩
    _ = (2 ^ b) ^ (N - code.2.2.card) := by simp

/-- `k` correct receipts whose words were queried, no collision, and too little else: a `C(N, k)·2^-λ` fraction of
stores and streams, by compression with the named words. -/
theorem card_compress_named_le (A : RelayDevice N b h ρ S L q) (O : Oracle N b h ρ) (k lam : ℕ) (hk : 0 < k)
    [DecidablePred (fun x : Store N b × Bits ρ =>
      k ≤ (A.correct x.1 x.2 O ∩ queried A x.1 x.2 O).card ∧ taken A x.1 x.2 O + lam < k * b + ρ ∧
        ¬ Collides O ((A.chip (A.state x.1)).queries (A.world x.1 x.2 O)))] :
    (Finset.univ.filter (fun x : Store N b × Bits ρ =>
        k ≤ (A.correct x.1 x.2 O ∩ queried A x.1 x.2 O).card ∧ taken A x.1 x.2 O + lam < k * b + ρ ∧
          ¬ Collides O ((A.chip (A.state x.1)).queries (A.world x.1 x.2 O)))).card * 2 ^ lam ≤
      N.choose k * Fintype.card (Store N b × Bits ρ) := by
  classical
  by_cases hpar : S + lam ≤ k * b + ρ ∧ k ≤ N
  swap
  · rw [Finset.filter_false_of_mem (fun x _ ⟨hk', hlt, _⟩ =>
      hpar ⟨by unfold taken at hlt; omega, hk'.trans ((Finset.card_le_univ _).trans (by simp))⟩)]
    simp
  obtain ⟨hSl, hkN⟩ := hpar
  set r := k * b + ρ - S - lam with hr
  let codes : Finset (Bits S × Bits r × Finset (Fin N)) :=
    Finset.univ ×ˢ Finset.univ ×ˢ (Finset.univ : Finset (Fin N)).powersetCard k
  have hsub : Finset.univ.filter (fun x : Store N b × Bits ρ =>
        k ≤ (A.correct x.1 x.2 O ∩ queried A x.1 x.2 O).card ∧ taken A x.1 x.2 O + lam < k * b + ρ ∧
          ¬ Collides O ((A.chip (A.state x.1)).queries (A.world x.1 x.2 O))) ⊆
      codes.biUnion (fibNamed A O r) := by
    rintro ⟨E, R⟩ hx
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
    obtain ⟨hkq, hlow, hcol⟩ := hx
    have hT : ((A.chip (A.state E)).trace (A.world E R O) ∅).length < r := by
      have h1 := Chip.trace_length (A.world E R O) (A.chip (A.state E)) ∅
      simp only [Finset.sdiff_empty] at h1
      simp only [taken, RelayDevice.footprint] at hlow
      omega
    obtain ⟨Kk, hKk, hKkcard⟩ := Finset.exists_subset_card_eq hkq
    let bits : Bits r := fun i => ((A.chip (A.state E)).trace (A.world E R O) ∅).getD i false
    have hstream : Bandwidth.AuditRO.stream r bits =
        fun n => ((A.chip (A.state E)).trace (A.world E R O) ∅ ++ []).getD n false := by
      funext n
      simp only [Bandwidth.AuditRO.stream, bits, List.append_nil]
      split_ifs with hn
      · rfl
      · rw [List.getD_eq_default _ _ (by omega)]
    have hdec : decodedRun A O r (A.state E, bits, Kk) =
        ((A.chip (A.state E)).queries (A.world E R O), A.tags E R O) := by
      simp only [decodedRun, hstream]
      exact Prod.ext (Chip.replay_trace (A.world E R O) (A.chip (A.state E)) ∅ (fun _ => none) (fun a => by simp) [])
        (Chip.replayRun_trace (A.world E R O) (A.chip (A.state E)) ∅ (fun _ => none) (fun a => by simp) [])
    refine Finset.mem_biUnion.2 ⟨(A.state E, bits, Kk), ?_, ?_⟩
    · simp only [codes, Finset.mem_product, Finset.mem_univ, true_and, Finset.mem_powersetCard]
      exact ⟨Finset.subset_univ _, hKkcard⟩
    · simp only [fibNamed, Finset.mem_filter, Finset.mem_univ, true_and]
      intro j hj
      obtain ⟨hc, hq⟩ := Finset.mem_inter.1 (hKk hj)
      simp only [RelayDevice.correct, Finset.mem_filter, Finset.mem_univ, true_and] at hc
      simp only [queried, Finset.mem_filter, Finset.mem_univ, true_and] at hq
      simp only [namedAt, hdec]
      cases hf : ((A.chip (A.state E)).queries (A.world E R O)).find?
          (fun y => decide (y.2.1 = j ∧ O y = A.tags E R O j)) with
      | none => exact absurd (List.find?_eq_none.1 hf _ hq) (by simp [hc])
      | some y =>
        have hy := List.mem_of_find?_eq_some hf
        have hp := List.find?_some hf
        simp only [decide_eq_true_eq] at hp
        have hyeq : y = (R, j, E j) := by
          by_contra hne
          exact hcol ⟨y, hy, (R, j, E j), hq, hne, hp.2.trans hc⟩
        subst hyeq
        exact ⟨_, rfl, rfl, rfl⟩
  have hcodes : codes.card = 2 ^ S * (2 ^ r * N.choose k) := by
    simp [codes, Finset.card_product, Finset.card_powersetCard]
  have hΩ : Fintype.card (Store N b × Bits ρ) = (2 ^ b) ^ N * 2 ^ ρ := by simp
  calc (Finset.univ.filter (fun x : Store N b × Bits ρ =>
          k ≤ (A.correct x.1 x.2 O ∩ queried A x.1 x.2 O).card ∧ taken A x.1 x.2 O + lam < k * b + ρ ∧
            ¬ Collides O ((A.chip (A.state x.1)).queries (A.world x.1 x.2 O)))).card * 2 ^ lam
      ≤ (codes.card * (2 ^ b) ^ (N - k)) * 2 ^ lam := by
        gcongr
        rw [Finset.filter_congr_decidable]
        refine (Finset.card_le_card hsub).trans (Finset.card_biUnion_le.trans ?_)
        refine Finset.sum_le_card_nsmul _ _ ((2 ^ b) ^ (N - k)) (fun code hc => ?_) |>.trans (le_of_eq ?_)
        · have hk' : code.2.2.card = k := by
            simp only [codes, Finset.mem_product, Finset.mem_univ, true_and, Finset.mem_powersetCard] at hc
            exact hc.2
          exact (card_fibNamed A O r code (by omega)).trans (le_of_eq (by rw [hk']))
        · rw [smul_eq_mul]
    _ = N.choose k * Fintype.card (Store N b × Bits ρ) := by
        have e1 : (2 : ℕ) ^ S * 2 ^ r * 2 ^ lam = 2 ^ (k * b) * 2 ^ ρ := by
          rw [← pow_add, ← pow_add, ← pow_add]
          congr 1
          omega
        have e2 : (2 : ℕ) ^ (k * b) * (2 ^ b) ^ (N - k) = (2 ^ b) ^ N := by
          rw [mul_comm k b, pow_mul, ← pow_add, Nat.add_sub_cancel' hkN]
        rw [hcodes, hΩ, ← e2]
        calc 2 ^ S * (2 ^ r * N.choose k) * (2 ^ b) ^ (N - k) * 2 ^ lam
            = N.choose k * (2 ^ S * 2 ^ r * 2 ^ lam) * (2 ^ b) ^ (N - k) := by ring
          _ = N.choose k * (2 ^ (k * b) * (2 ^ b) ^ (N - k) * 2 ^ ρ) := by rw [e1]; ring

/-- `k` correct receipts whose words were queried, no collision, and too little else: probability at most
`C(N, k)/2^λ`. -/
theorem pr_compress_named_le (A : RelayDevice N b h ρ S L q) (k lam : ℕ) (hk : 0 < k) :
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
        k ≤ (A.correct x.1 x.2.1 x.2.2 ∩ queried A x.1 x.2.1 x.2.2).card ∧
          taken A x.1 x.2.1 x.2.2 + lam < k * b + ρ ∧
          ¬ Collides x.2.2 ((A.chip (A.state x.1)).queries (A.world x.1 x.2.1 x.2.2)))
      ≤ (N.choose k : ℝ≥0∞) / 2 ^ lam := by
  classical
  refine (pr_le_of_card _ (2 ^ lam) (N.choose k) (by positivity) (by
    rw [card_filter_equiv (Equiv.prodAssoc (Store N b) (Bits ρ) (Oracle N b h ρ)).symm]
    refine (card_filter_prod_le' _ _ (N.choose k * Fintype.card (Store N b × Bits ρ)) (fun O => ?_)).trans
      (le_of_eq (by simp only [Fintype.card_prod]; ring))
    exact card_compress_named_le A O k lam hk)).trans (le_of_eq ?_)
  push_cast
  rfl

/-- At some count `k ≤ N`, `k` correct receipts whose words were queried, no collision, and too little else:
probability at most `∑ₖ C(N, k)/2^λ = 2^N/2^λ`. -/
theorem pr_compress_any_named_le (A : RelayDevice N b h ρ S L q) (lam : ℕ) :
    pr (fun x : Store N b × Bits ρ × Oracle N b h ρ => ∃ k : Fin (N + 1),
        0 < k.1 ∧ k.1 ≤ (A.correct x.1 x.2.1 x.2.2 ∩ queried A x.1 x.2.1 x.2.2).card ∧
          taken A x.1 x.2.1 x.2.2 + lam < k.1 * b + ρ ∧
          ¬ Collides x.2.2 ((A.chip (A.state x.1)).queries (A.world x.1 x.2.1 x.2.2)))
      ≤ (2 ^ N : ℝ≥0∞) / 2 ^ lam := by
  classical
  refine (pr_exists_le _).trans ?_
  calc ∑ k : Fin (N + 1), pr (fun x : Store N b × Bits ρ × Oracle N b h ρ =>
          0 < k.1 ∧ k.1 ≤ (A.correct x.1 x.2.1 x.2.2 ∩ queried A x.1 x.2.1 x.2.2).card ∧
            taken A x.1 x.2.1 x.2.2 + lam < k.1 * b + ρ ∧
            ¬ Collides x.2.2 ((A.chip (A.state x.1)).queries (A.world x.1 x.2.1 x.2.2)))
      ≤ ∑ k : Fin (N + 1), (N.choose k.1 : ℝ≥0∞) / 2 ^ lam := by
        refine Finset.sum_le_sum (fun k _ => ?_)
        rcases Nat.eq_zero_or_pos k.1 with hk | hk
        · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
          rw [Finset.filter_false_of_mem (fun x _ hx => absurd hx.1 (by omega)), Finset.card_empty, zero_mul,
            zero_mul]
        · exact (pr_mono (fun x hx => hx.2)).trans (pr_compress_named_le A k.1 lam hk)
    _ = (2 ^ N : ℝ≥0∞) / 2 ^ lam := by
        rw [← ENNReal.sum_div]
        congr 1
        rw [← Nat.cast_sum, Fin.sum_univ_eq_sum_range (fun k => N.choose k), Nat.sum_range_choose]
        push_cast
        rfl

end RelayDevice

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal
open Bandwidth (pr_le_of_card pr_mono pr_or_le)

/-- **Conservation over tiers with named words, with side sources that ignore the oracle**
(`Accounting.Guarantees.RateConservationNamedRO`). -/
theorem RateConservationNamedRO : Accounting.Guarantees.RateConservationNamedRO := by
  classical
  intro N b h ρ S L q k lam τ tier A hA
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · refine (pr_le_of_card _ 1 0 one_ne_zero ?_).trans (by simp)
    rw [Finset.filter_false_of_mem (fun x _ hx => by simp at hx), Finset.card_empty, zero_mul, zero_mul]
  have hA' := InteractiveDevice.toRelay_side A hA
  have h₁ := A.toRelay.pr_guess_le hA'
  have h₂ := A.toRelay.pr_collides_le hA'
  have h₃ := A.toRelay.pr_compress_named_le k lam hk
  refine (pr_mono (fun x hx => ?_)).trans
    ((pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans (add_le_add h₁ h₂)) h₃))
  obtain ⟨hc, hlt⟩ := hx
  by_cases hcol : Collides x.2.2 ((A.toRelay.chip (A.toRelay.state x.1)).queries (A.toRelay.world x.1 x.2.1 x.2.2))
  · exact Or.inl (Or.inr hcol)
  by_cases hg : ∃ j, j ∉ queried A.toRelay x.1 x.2.1 x.2.2 ∧
      A.toRelay.tags x.1 x.2.1 x.2.2 j = x.2.2 (x.2.1, j, x.1 j)
  · exact Or.inl (Or.inl hg)
  have hsub : A.toRelay.correct x.1 x.2.1 x.2.2 ⊆ queried A.toRelay x.1 x.2.1 x.2.2 := fun j hj => by
    by_contra hq
    exact hg ⟨j, hq, (Finset.mem_filter.1 hj).2⟩
  have hk' : k ≤ (A.toRelay.correct x.1 x.2.1 x.2.2 ∩ queried A.toRelay x.1 x.2.1 x.2.2).card := by
    rw [Finset.inter_eq_left.2 hsub, InteractiveDevice.toRelay_correct]
    exact hc
  have := InteractiveDevice.taken_le A tier x.1 x.2.1 x.2.2
  exact Or.inr ⟨hk', by omega, hcol⟩

/-- **Correct receipts are paid for, at every count, with named words** (`Accounting.Guarantees.RatePaidNamedRO`). -/
theorem RatePaidNamedRO : Accounting.Guarantees.RatePaidNamedRO := by
  classical
  intro N b h ρ S L q lam τ tier A hA
  have hA' := InteractiveDevice.toRelay_side A hA
  refine (pr_mono (fun x hx => ?_)).trans ((pr_or_le _ _).trans (add_le_add
    ((pr_or_le _ _).trans (add_le_add (A.toRelay.pr_guess_le hA') (A.toRelay.pr_collides_le hA')))
    (A.toRelay.pr_compress_any_named_le lam)))
  by_cases hcol : Collides x.2.2 ((A.toRelay.chip (A.toRelay.state x.1)).queries (A.toRelay.world x.1 x.2.1 x.2.2))
  · exact Or.inl (Or.inr hcol)
  by_cases hg : ∃ j, j ∉ queried A.toRelay x.1 x.2.1 x.2.2 ∧
      A.toRelay.tags x.1 x.2.1 x.2.2 j = x.2.2 (x.2.1, j, x.1 j)
  · exact Or.inl (Or.inl hg)
  have hsub : A.toRelay.correct x.1 x.2.1 x.2.2 ⊆ queried A.toRelay x.1 x.2.1 x.2.2 := fun j hj => by
    by_contra hq
    exact hg ⟨j, hq, (Finset.mem_filter.1 hj).2⟩
  have hk' : (A.toRelay.correct x.1 x.2.1 x.2.2 ∩ queried A.toRelay x.1 x.2.1 x.2.2).card =
      (A.correct x.1 x.2.1 x.2.2).card := by
    rw [Finset.inter_eq_left.2 hsub, InteractiveDevice.toRelay_correct]
  have hN : (A.correct x.1 x.2.1 x.2.2).card ≤ N := (Finset.card_le_univ _).trans (by simp)
  have ht := InteractiveDevice.taken_le A tier x.1 x.2.1 x.2.2
  have hk : 0 < (A.correct x.1 x.2.1 x.2.2).card :=
    Nat.pos_of_ne_zero (fun h0 => by rw [h0, zero_mul] at hx; omega)
  refine Or.inr ⟨⟨(A.correct x.1 x.2.1 x.2.2).card, by omega⟩, hk, hk'.ge, ?_, hcol⟩
  dsimp only
  omega

/-- **The route `Sampled` with named words, with side sources that ignore the oracle**
(`Accounting.Guarantees.RateChainSampledNamedRO`). -/
theorem RateChainSampledNamedRO : Accounting.Guarantees.RateChainSampledNamedRO := by
  intro P N b h ρ S L q r s K lam τ tier A hA
  have h₁ := Chain.pr_horizon_le A K
  have h₂ := Chain.pr_unpaid_le A tier (S + (L - ρ) + lam) _ (fun j η =>
    (pr_mono (fun _ hx => by omega)).trans (RatePaidNamedRO N b h ρ S L q lam τ tier (A.device j η) (hA j η)))
  refine (pr_mono (fun x hx => ?_)).trans ((pr_or_le _ _).trans (add_le_add h₁ h₂))
  by_cases hK : ∑ j, ((A.dev j x.2).correct x.1 (x.2 j).1 (A.oracle j x.1 x.2)).card ≤ K
  · exact Or.inl ⟨hx.1, hK⟩
  · refine Or.inr (hx.2.trans_le ?_)
    rw [← Finset.sum_mul]
    exact Nat.mul_le_mul_right _ (by omega)

end Accounting.SecurityProofs
