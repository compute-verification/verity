import Definitions.Pouw.Assumptions.PearlC.RowSeedAssumptions
import Proofs.Pouw.PearlC.RowSeedFragment
import Proofs.Pouw.Prob
import Proofs.Pouw.QTree

/-!
# The fragment corollary: the union bound and `ttOutRowSeed_skipClass` (staging, nothing pinned)

Option (a) of the fragment-draw ruling; the definitions are in
`TTOutRowSeed.lean`, and the named assumption `Assumptions.FragDraw` is in `RowSeedAssumptions.lean`. **P2 rests on
`ttOutRowSeed_of_ttOut` (and `ttOutTileRowSeed_of_ttOutTile`) alone, with no named assumption.** This file is a
supporting line: no statement those proofs read is proved here. It takes no TT_OUT, and its one named assumption is
`FragDraw`.

* **`qtree_union_bound`** (the ruling's item 3): an adaptive program with at most `q` oracle calls, whose result names
  `m` final checks, gets an answer in its input's bad set, each of probability at most `ε`, at some call or final check
  with probability at most `(q + m)·ε`. The proof is by induction on the calls: splitting a uniform oracle at the first
  query (`card_split`) leaves the other answers uniform, and the rest of the program with that answer fixed
  (`QTree.fixAt`) makes one call fewer, a repeated query included. `qtree_union_bound_at` lets the bad sets depend on
  the answer at one input that no bad set contains (the per-job draw), by conditioning on it.
* **`ttOutRowSeed_skipClass`** (the ruling's item 5): under the named assumption `FragDraw` at `Λ = 100`, (C4) and
  (C6), an adversary program with at most `q` calls outputs a skip-class program on unit `u` that saves more than
  `(4096/11)·100 ≈ 37,236` units per row with probability at most `(q + m)/2^128`, `m` the unit's rows. Outside the
  union's bad event every final row has `fragCount ≤ 100`, and `fragSkip_count_le` with (C4) bounds the saving.
* **`rowDrawn_satisfiable`**: (C6) holds on a finite oracle domain, so the corollary's `[Fintype Q]` doesn't make it
  vacuous. (C6) asks for an injective key only on the unit's rows and their 32-bit words, the rows the corollary reads.
-/

namespace Pouw.QTree

variable {Q R α β : Type}

/-- Query the points of `l`, in order. -/
def queryList : List Q → QTree Q R Unit
  | [] => done ()
  | x :: l => query x fun _ => queryList l

theorem hits_bind (Bad : Q → R → Prop) (H : Q → R) (t : QTree Q R α) (f : α → QTree Q R β) :
    (t.bind f).hits Bad H ↔ t.hits Bad H ∨ (f (t.eval H)).hits Bad H := by
  induction t with
  | done a => simp [bind, hits, eval]
  | query x k ih => simp only [bind, hits, eval, ih, or_assoc]

theorem hits_queryList (Bad : Q → R → Prop) (H : Q → R) (l : List Q) :
    (queryList l : QTree Q R Unit).hits Bad H ↔ ∃ x ∈ l, Bad x (H x) := by
  induction l with
  | nil => simp [queryList, hits]
  | cons x l ih => simp [queryList, hits, ih]

theorem calls_queryList (H : Q → R) (l : List Q) : (queryList l : QTree Q R Unit).calls H = l.length := by
  induction l with
  | nil => simp [queryList, calls]
  | cons x l ih => simp [queryList, calls, ih]

theorem calls_pos_of_hits {Bad : Q → R → Prop} {H : Q → R} {t : QTree Q R α} (h : t.hits Bad H) : 0 < t.calls H := by
  cases t with
  | done a => exact h.elim
  | query x k => simp [calls]

variable [DecidableEq Q]

/-- `t` with the answer at `x` fixed to `r`: its queries at `x` are answered `r` without a call. -/
def fixAt (x : Q) (r : R) : QTree Q R α → QTree Q R α
  | done a => done a
  | query y k => if y = x then fixAt x r (k r) else query y fun r' => fixAt x r (k r')

theorem hits_fixAt {Bad : Q → R → Prop} {H : Q → R} {x : Q} {r : R} (t : QTree Q R α)
    (h : t.hits Bad (Function.update H x r)) : Bad x r ∨ (t.fixAt x r).hits Bad H := by
  induction t with
  | done a => exact h.elim
  | query y k ih =>
    by_cases hy : y = x
    · subst hy
      simp only [hits, Function.update_self] at h
      simp only [fixAt, ↓reduceIte]
      rcases h with h | h
      · exact Or.inl h
      · exact ih r h
    · simp only [hits, Function.update_of_ne hy] at h
      simp only [fixAt, hy, ↓reduceIte, hits]
      rcases h with h | h
      · exact Or.inr (Or.inl h)
      · rcases ih (H y) h with h | h
        · exact Or.inl h
        · exact Or.inr (Or.inr h)

theorem calls_fixAt (H : Q → R) (x : Q) (r : R) (t : QTree Q R α) :
    (t.fixAt x r).calls H ≤ t.calls (Function.update H x r) := by
  induction t with
  | done a => simp [fixAt, calls]
  | query y k ih =>
    by_cases hy : y = x
    · subst hy
      simp only [fixAt, ↓reduceIte, calls, Function.update_self]
      exact (ih r).trans (Nat.le_succ _)
    · simp only [fixAt, hy, ↓reduceIte, calls, Function.update_of_ne hy]
      exact Nat.succ_le_succ (ih (H y))

theorem eval_fixAt (H : Q → R) (x : Q) (r : R) (t : QTree Q R α) :
    (t.fixAt x r).eval H = t.eval (Function.update H x r) := by
  induction t with
  | done a => simp [fixAt, eval]
  | query y k ih =>
    by_cases hy : y = x
    · subst hy
      simp only [fixAt, ↓reduceIte, eval, Function.update_self]
      exact ih r
    · simp only [fixAt, hy, ↓reduceIte, eval, Function.update_of_ne hy]
      exact ih (H y)

end Pouw.QTree

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

/-- Splitting a uniform oracle at one point: `|R|·#{H | F (H x) H} = Σ_r #{H | F r (H with x ↦ r)}`. -/
theorem card_split [Fintype Q] [DecidableEq Q] [Fintype R] (x : Q) (F : R → (Q → R) → Prop)
    [∀ r H, Decidable (F r H)] :
    Fintype.card R * (univ.filter fun H : Q → R => F (H x) H).card =
      ∑ r : R, (univ.filter fun H : Q → R => F r (Function.update H x r)).card := by
  let Φ : R × (Q → R) → R × (Q → R) := fun p => (p.2 x, Function.update p.2 x p.1)
  have hΦ : ∀ p, Φ (Φ p) = p := fun p => by
    simp only [Φ, Function.update_self, Function.update_idem, Function.update_eq_self]
  have h1 : Fintype.card R * (univ.filter fun H : Q → R => F (H x) H).card =
      (univ.filter fun p : R × (Q → R) => F (p.2 x) p.2).card := by
    rw [← card_univ, ← card_product, ← filter_product_right, univ_product_univ]
  have h2 : (univ.filter fun p : R × (Q → R) => F p.1 (Function.update p.2 x p.1)).card =
      (univ.filter fun p : R × (Q → R) => F (p.2 x) p.2).card := by
    refine card_nbij' Φ Φ (fun p hp => ?_) (fun p hp => ?_) (fun p _ => hΦ p) (fun p _ => hΦ p)
    · simp only [mem_coe, mem_filter, mem_univ, true_and, Φ, Function.update_self] at hp ⊢
      exact hp
    · simp only [mem_coe, mem_filter, mem_univ, true_and, Φ, Function.update_idem, Function.update_eq_self] at hp ⊢
      exact hp
  rw [h1, ← h2, card_filter, Fintype.sum_prod_type]
  simp only [card_filter]

theorem card_le_of_pr_le {α : Type} [Fintype α] {P : α → Prop} [DecidablePred P] {ε : ℝ} (h : pr P ≤ ε) :
    ((univ.filter P).card : ℝ) ≤ ε * Fintype.card α := by
  rw [pr_eq_card P (univ.filter P) (by simp)] at h
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h0 | h0
  · have : (univ.filter P).card = 0 :=
      Nat.eq_zero_of_le_zero ((card_filter_le _ _).trans (by rw [card_univ, h0]))
    simp [this, h0]
  · rwa [div_le_iff₀ (by exact_mod_cast h0)] at h

theorem pr_le_of_card_le {α : Type} [Fintype α] {P : α → Prop} [DecidablePred P] {b : ℝ} (hb : 0 ≤ b)
    (h : ((univ.filter P).card : ℝ) ≤ b * Fintype.card α) : pr P ≤ b := by
  rw [pr_eq_card P (univ.filter P) (by simp)]
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h0 | h0
  · simp [h0, hb]
  · rwa [div_le_iff₀ (by exact_mod_cast h0)]

/-- A uniform oracle has no values when `R` is empty and `Q` has a point. -/
theorem card_fun_eq_zero [Fintype Q] [DecidableEq Q] [Fintype R] (x : Q) (hc : Fintype.card R = 0) :
    Fintype.card (Q → R) = 0 :=
  Fintype.card_eq_zero_iff.mpr ⟨fun H => (Fintype.card_eq_zero_iff.mp hc).elim (H x)⟩

open Classical in
/-- **The union bound over an adaptive program's queries, counted:** if every bad set has at most `ε·|R|` answers,
at most `n·ε·|Q → R|` oracles make a program with at most `n` calls hit one. -/
theorem card_hits_le [Fintype Q] [DecidableEq Q] [Fintype R] {α : Type} (Bad : Q → R → Prop) {ε : ℝ} (hε : 0 ≤ ε)
    (hB : ∀ x, ((univ.filter fun r : R => Bad x r).card : ℝ) ≤ ε * Fintype.card R) :
    ∀ (n : ℕ) (t : QTree Q R α), (∀ H, t.calls H ≤ n) →
      ((univ.filter fun H : Q → R => t.hits Bad H).card : ℝ) ≤ n * ε * Fintype.card (Q → R) := by
  intro n
  induction n with
  | zero =>
    intro t ht
    have : (univ.filter fun H : Q → R => t.hits Bad H) = ∅ :=
      filter_false_of_mem fun H _ h => by have := QTree.calls_pos_of_hits h; have := ht H; omega
    simp [this]
  | succ n ih =>
    intro t ht
    cases t with
    | done a =>
      have : (univ.filter fun H : Q → R => (QTree.done a : QTree Q R α).hits Bad H) = ∅ :=
        filter_false_of_mem fun H _ h => h
      rw [this, card_empty, Nat.cast_zero]
      positivity
    | query x k =>
      have hN0 : (0 : ℝ) ≤ Fintype.card (Q → R) := Nat.cast_nonneg _
      have hnεN : (0 : ℝ) ≤ n * ε * Fintype.card (Q → R) := by positivity
      have hr : ∀ r : R,
          ((univ.filter fun H : Q → R => Bad x r ∨ (k r).hits Bad (Function.update H x r)).card : ℝ) ≤
            (if Bad x r then (Fintype.card (Q → R) : ℝ) else 0) + n * ε * Fintype.card (Q → R) := by
        intro r
        split_ifs with hb
        · have : ((univ.filter fun H : Q → R => Bad x r ∨ (k r).hits Bad (Function.update H x r)).card : ℝ) ≤
              Fintype.card (Q → R) := by exact_mod_cast card_filter_le _ _
          linarith
        · have hsub : (univ.filter fun H : Q → R => Bad x r ∨ (k r).hits Bad (Function.update H x r)) ⊆
              univ.filter fun H => ((k r).fixAt x r).hits Bad H := by
            intro H hH
            simp only [mem_filter, mem_univ, true_and] at hH ⊢
            rcases hH with h | h
            · exact absurd h hb
            · exact (QTree.hits_fixAt _ h).resolve_left hb
          have hcalls : ∀ H, ((k r).fixAt x r).calls H ≤ n := fun H => by
            have h1 := QTree.calls_fixAt H x r (k r)
            have h2 := ht (Function.update H x r)
            simp only [QTree.calls, Function.update_self] at h2
            omega
          have h3 : ((univ.filter fun H : Q → R => Bad x r ∨ (k r).hits Bad (Function.update H x r)).card : ℝ) ≤
              ((univ.filter fun H => ((k r).fixAt x r).hits Bad H).card : ℝ) := by
            exact_mod_cast card_le_card hsub
          rw [zero_add]
          exact h3.trans (ih _ hcalls)
      have hsplit := card_split x (fun r H => Bad x r ∨ (k r).hits Bad H)
      have e1 : (univ.filter fun H : Q → R => (QTree.query x k).hits Bad H) =
          univ.filter fun H : Q → R => Bad x (H x) ∨ (k (H x)).hits Bad H := by
        ext H; simp only [mem_filter, mem_univ, true_and]; rfl
      rw [e1]
      have hsum : (Fintype.card R : ℝ) * ((univ.filter fun H : Q → R => Bad x (H x) ∨ (k (H x)).hits Bad H).card : ℝ)
          ≤ Fintype.card R * ((n + 1 : ℕ) * ε * Fintype.card (Q → R)) := by
        have hs : (Fintype.card R : ℝ) * ((univ.filter fun H : Q → R => Bad x (H x) ∨ (k (H x)).hits Bad H).card : ℝ) =
            ∑ r : R, ((univ.filter fun H : Q → R => Bad x r ∨ (k r).hits Bad (Function.update H x r)).card : ℝ) := by
          exact_mod_cast hsplit
        rw [hs]
        calc _ ≤ ∑ r : R, ((if Bad x r then (Fintype.card (Q → R) : ℝ) else 0) + n * ε * Fintype.card (Q → R)) :=
              sum_le_sum fun r _ => hr r
          _ = ((univ.filter fun r : R => Bad x r).card : ℝ) * Fintype.card (Q → R) +
                Fintype.card R * (n * ε * Fintype.card (Q → R)) := by
              rw [sum_add_distrib, sum_ite, sum_const_zero, add_zero, sum_const, sum_const, card_univ,
                nsmul_eq_mul, nsmul_eq_mul]
          _ ≤ ε * Fintype.card R * Fintype.card (Q → R) + Fintype.card R * (n * ε * Fintype.card (Q → R)) := by
              gcongr; exact hB x
          _ = _ := by push_cast; ring
      rcases Nat.eq_zero_or_pos (Fintype.card R) with hc | hc
      · have hN := card_fun_eq_zero x hc
        have : (univ.filter fun H : Q → R => Bad x (H x) ∨ (k (H x)).hits Bad H).card = 0 :=
          Nat.eq_zero_of_le_zero ((card_filter_le _ _).trans (by rw [card_univ, hN]))
        simp [this, hN]
      · exact le_of_mul_le_mul_left hsum (by exact_mod_cast hc)

open Classical in
/-- The counting form of `qtree_union_bound`. -/
theorem card_union_bound [Fintype Q] [DecidableEq Q] [Fintype R] {α : Type} (t : QTree Q R α) (q : ℕ)
    (hq : ∀ H, t.calls H ≤ q) (m : ℕ) (fin : α → ℕ → Q) (Bad : Q → R → Prop) {ε : ℝ} (hε : 0 ≤ ε)
    (hB : ∀ x, ((univ.filter fun r : R => Bad x r).card : ℝ) ≤ ε * Fintype.card R) :
    ((univ.filter fun H : Q → R => t.hits Bad H ∨ ∃ i < m, Bad (fin (t.eval H) i) (H (fin (t.eval H) i))).card : ℝ)
      ≤ (q + m) * ε * Fintype.card (Q → R) := by
  let t' := t.bind fun a => QTree.queryList ((List.range m).map (fin a))
  have hcalls : ∀ H, t'.calls H ≤ q + m := fun H => by
    simp only [t', QTree.calls_bind, QTree.calls_queryList, List.length_map, List.length_range]
    exact Nat.add_le_add_right (hq H) m
  have e : (univ.filter fun H : Q → R => t.hits Bad H ∨ ∃ i < m, Bad (fin (t.eval H) i) (H (fin (t.eval H) i))) =
      univ.filter fun H => t'.hits Bad H := by
    ext H
    simp only [mem_filter, mem_univ, true_and, t', QTree.hits_bind, QTree.hits_queryList, List.mem_map,
      List.mem_range]
    constructor
    · rintro (h | ⟨i, hi, h⟩)
      · exact Or.inl h
      · exact Or.inr ⟨_, ⟨i, hi, rfl⟩, h⟩
    · rintro (h | ⟨_, ⟨i, hi, rfl⟩, h⟩)
      · exact Or.inl h
      · exact Or.inr ⟨i, hi, h⟩
  rw [e]
  have := card_hits_le Bad hε hB (q + m) t' hcalls
  push_cast at this
  exact this

/-- **The union bound over an adaptive program's oracle draws** (the fragment-draw ruling's item 3). A program `t` makes
at most `q` oracle calls, and its result `a` names `m` final checks, the points `fin a i`, `i < m`. Each point `x` has a
bad set `Bad x` of answers, of probability at most `ε`. Then the probability over a uniform oracle that some call's
answer, or some final check's, lies in its point's bad set is at most `(q + m)·ε`. A point queried again is answered as
before, so its bad event is the first query's. -/
theorem qtree_union_bound [Fintype Q] [DecidableEq Q] [Fintype R] {α : Type} (t : QTree Q R α) (q : ℕ)
    (hq : ∀ H, t.calls H ≤ q) (m : ℕ) (fin : α → ℕ → Q) (Bad : Q → R → Prop) {ε : ℝ} (hε : 0 ≤ ε)
    (hB : ∀ x, pr (fun r : R => Bad x r) ≤ ε) :
    pr (fun H : Q → R => t.hits Bad H ∨ ∃ i < m, Bad (fin (t.eval H) i) (H (fin (t.eval H) i))) ≤ (q + m) * ε := by
  classical
  refine pr_le_of_card_le (by positivity) ?_
  convert card_union_bound t q hq m fin Bad hε (fun x => card_le_of_pr_le (hB x)) using 1

open Classical in
/-- **The union bound with the bad sets read at a fixed point** `xJ` (the per-job draw): the bad sets `Bad z` may depend
on the answer `z = H xJ`, provided no answer at `xJ` itself is bad. Conditioning on `H xJ` leaves the other answers
uniform, and the program with `xJ` answered `z` has no more calls. -/
theorem qtree_union_bound_at [Fintype Q] [DecidableEq Q] [Fintype R] {α : Type} (t : QTree Q R α) (q : ℕ)
    (hq : ∀ H, t.calls H ≤ q) (m : ℕ) (fin : α → ℕ → Q) (xJ : Q) (Bad : R → Q → R → Prop) {ε : ℝ} (hε : 0 ≤ ε)
    (hJ : ∀ z r, ¬ Bad z xJ r) (hB : ∀ z x, pr (fun r : R => Bad z x r) ≤ ε) :
    pr (fun H : Q → R => t.hits (Bad (H xJ)) H ∨ ∃ i < m, Bad (H xJ) (fin (t.eval H) i) (H (fin (t.eval H) i)))
      ≤ (q + m) * ε := by
  refine pr_le_of_card_le (by positivity) ?_
  let G : R → (Q → R) → Prop := fun z H =>
    t.hits (Bad z) H ∨ ∃ i < m, Bad z (fin (t.eval H) i) (H (fin (t.eval H) i))
  have hz : ∀ z : R, ((univ.filter fun H : Q → R => G z (Function.update H xJ z)).card : ℝ) ≤
      (q + m) * ε * Fintype.card (Q → R) := by
    intro z
    have hsub : (univ.filter fun H : Q → R => G z (Function.update H xJ z)) ⊆
        univ.filter fun H : Q → R => (t.fixAt xJ z).hits (Bad z) H ∨
          ∃ i < m, Bad z (fin ((t.fixAt xJ z).eval H) i) (H (fin ((t.fixAt xJ z).eval H) i)) := by
      intro H hH
      simp only [mem_filter, mem_univ, true_and, G] at hH ⊢
      rw [QTree.eval_fixAt]
      rcases hH with h | ⟨i, hi, h⟩
      · exact Or.inl ((QTree.hits_fixAt _ h).resolve_left (hJ z z))
      · refine Or.inr ⟨i, hi, ?_⟩
        by_cases hx : fin (t.eval (Function.update H xJ z)) i = xJ
        · rw [hx] at h; exact absurd h (hJ z _)
        · rwa [Function.update_of_ne hx] at h
    refine (Nat.cast_le.mpr (card_le_card hsub)).trans ?_
    exact card_union_bound (t.fixAt xJ z) q (fun H => (QTree.calls_fixAt H xJ z t).trans (hq _)) m fin (Bad z) hε
      (fun x => card_le_of_pr_le (hB z x))
  have hsplit := card_split xJ G
  have hsum : (Fintype.card R : ℝ) * ((univ.filter fun H : Q → R => G (H xJ) H).card : ℝ) ≤
      Fintype.card R * ((q + m) * ε * Fintype.card (Q → R)) := by
    have hs : (Fintype.card R : ℝ) * ((univ.filter fun H : Q → R => G (H xJ) H).card : ℝ) =
        ∑ z : R, ((univ.filter fun H : Q → R => G z (Function.update H xJ z)).card : ℝ) := by
      exact_mod_cast hsplit
    rw [hs]
    calc _ ≤ ∑ _z : R, (q + m) * ε * (Fintype.card (Q → R) : ℝ) := sum_le_sum fun z _ => hz z
      _ = _ := by rw [sum_const, card_univ, nsmul_eq_mul]
  rcases Nat.eq_zero_or_pos (Fintype.card R) with hc | hc
  · have hN := card_fun_eq_zero xJ hc
    have : (univ.filter fun H : Q → R => G (H xJ) H).card = 0 :=
      Nat.eq_zero_of_le_zero ((card_filter_le _ _).trans (by rw [card_univ, hN]))
    simp only [G] at this
    simp [this, hN]
  · exact le_of_mul_le_mul_left hsum (by exact_mod_cast hc)

theorem fragCount_congr (p : Pipeline) (G : ℕ) {m n k : ℕ} {A A' : Matrix (Fin m) (Fin k) Code}
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (h : A i = A' i) : fragCount p G A B i = fragCount p G A' B i := by
  unfold fragCount
  congr 1
  ext Jt
  simp only [mem_filter, h]

theorem operandOK_rowCodes {sem : PearlCSem Q R S} {m k : ℕ} {act : Codes} (h : sem.OperandOK m k act) (i : Fin m) :
    sem.OperandOK 1 k (rowCodes act i) :=
  ⟨fun _ _ l hl => h.1 i i.isLt l hl, fun _ => h.2 i⟩

/-- Under (C6), row `i` of `A′` reads the row's words, the job's answer and the row's answer only. -/
theorem unitChain_row_eq {sem : PearlCSem Q R S} {s : S} {U : Workload} {u : ℕ} {xJ : Q}
    {key : ℕ → (Fin (U.layout.shape u).k → ℕ) → Q} (hD : sem.RowDrawn s U u xJ key) {H H' : Q → R}
    {act act' : Codes} (i : Fin (U.layout.shape u).m)
    (hrow : rowWords (U.layout.shape u).k act i = rowWords (U.layout.shape u).k act' i) (hJ : H xJ = H' xJ)
    (hk : H (key i (rowWords (U.layout.shape u).k act i)) = H' (key i (rowWords (U.layout.shape u).k act i))) :
    (sem.unitChain H s U u act).1 i = (sem.unitChain H' s U u act').1 i := by
  have hX : wordsVal (U.layout.shape u).m (U.layout.shape u).k act i =
      wordsVal (U.layout.shape u).m (U.layout.shape u).k act' i := by
    funext l
    have := congrFun hrow l
    simp only [rowWords] at this
    simp only [wordsVal, this]
  have hEA := hD.2.2.2 H H' act act' i hrow hJ hk
  have hFA := (hD.2.2.1 H H' act act' hJ).1
  funext l
  simp only [PearlCSem.unitChain, FormChain.quantA, FormChain.quant, hX, hEA, hFA]

/-- Under (C6), `B̃` reads the job's answer only. -/
theorem unitChain_job_eq {sem : PearlCSem Q R S} {s : S} {U : Workload} {u : ℕ} {xJ : Q}
    {key : ℕ → (Fin (U.layout.shape u).k → ℕ) → Q} (hD : sem.RowDrawn s U u xJ key) {H H' : Q → R}
    (act act' : Codes) (hJ : H xJ = H' xJ) :
    (sem.unitChain H s U u act).2 = (sem.unitChain H' s U u act').2 := by
  obtain ⟨-, hEB, hFB⟩ := hD.2.2.1 H H' act act' hJ
  simp only [PearlCSem.unitChain, FormChain.quantB, hEB, hFB]

/-- **The fragment corollary** (the fragment-draw ruling's item 5), a supporting line: P2 rests on
`ttOutRowSeed_of_ttOut` alone, and this takes no TT_OUT. Under the named assumption `FragDraw` at `Λ = 100`, (C4) and
(C6): for every adversary program `T` with at most `q` oracle calls whose result is a skip program on unit `u`, the
probability over the oracle that the skip program is in the skip class and saves more than `(4096/11)·100` units per row
of the unit, against the honest program on some activation input and state, is at most `(q + m)/2^128`, `m` the unit's
rows. -/
theorem ttOutRowSeed_skipClass [Fintype Q] [DecidableEq Q] [Fintype R] {CM : CostModel Q R S}
    {sem : PearlCSem Q R S} {p : Pipeline} {G : ℕ} {s : S} {U : Workload} {u : ℕ} {xJ : Q}
    {key : ℕ → (Fin (U.layout.shape u).k → ℕ) → Q} (e : SkipEmbed CM sem p G U u) (hD : sem.RowDrawn s U u xJ key)
    (hF : Assumptions.FragDraw sem p G s U u key 100) (hC4 : CM.FragmentPriced e) (T : QTree Q R SkipProgram)
    (q : ℕ) (hq : ∀ H, T.calls H ≤ q) :
    pr (fun H : Q → R => (T.eval H).InClass sem p G s U u H ∧ ∃ (act : ℕ → Codes) (σ : State),
        4096 / 11 * 100 * ((U.layout.shape u).m : ℝ) <
          CM.cost e.honest H U act σ s - CM.cost (e.embed (T.eval H)) H U act σ s)
      ≤ (q + (U.layout.shape u).m) / 2 ^ 128 := by
  classical
  let fin : SkipProgram → ℕ → Q := fun sp i => key i (rowWords (U.layout.shape u).k sp.act i)
  let Bad : R → Q → R → Prop := fun z x r => ∃ (i : Fin (U.layout.shape u).m) (act' : Codes) (H' : Q → R),
    H' xJ = z ∧ H' x = r ∧ key i (rowWords (U.layout.shape u).k act' i) = x ∧
      (∀ l, rowWords (U.layout.shape u).k act' i l < 2 ^ 32) ∧
      sem.OperandOK 1 (U.layout.shape u).k (rowCodes act' i) ∧
      100 < fragCount p G (sem.unitChain H' s U u act').1 (sem.unitChain H' s U u act').2 i
  have hJ : ∀ z r, ¬ Bad z xJ r := by
    rintro z r ⟨i, act', H', -, -, hx, hw, -⟩
    exact hD.2.1 i _ hw hx
  have hB : ∀ z x, pr (fun r : R => Bad z x r) ≤ 1 / 2 ^ 128 := by
    intro z x
    by_cases h0 : ∃ r0, Bad z x r0
    · obtain ⟨r0, i0, act0, H0, hz0, -, hx0, hw0, hok0, -⟩ := h0
      refine (pr_mono fun r hr => ?_).trans (hF H0 act0 i0 hok0)
      obtain ⟨i, act', H', hz, hr', hx, hw, -, hfc⟩ := hr
      obtain ⟨hii, ha⟩ := hD.1 i i0 _ _ hw hw0 (hx.trans hx0.symm)
      subst hii
      have hxJ : xJ ≠ x := fun h => hD.2.1 i _ hw0 (hx0.trans h.symm)
      rw [hx0]
      have hA := unitChain_row_eq hD (H := H') (H' := Function.update H0 x r) (act := act') (act' := act0) i ha
        (by rw [Function.update_of_ne hxJ, hz, hz0]) (by rw [hx, hr', Function.update_self])
      rw [← fragCount_congr p G _ i hA, ← unitChain_job_eq hD act' act0 (hz.trans hz0.symm)]
      exact hfc
    · push Not at h0
      rw [pr_eq_zero h0]
      positivity
  have hstep : ∀ H : Q → R, ((T.eval H).InClass sem p G s U u H ∧ ∃ (act : ℕ → Codes) (σ : State),
      4096 / 11 * 100 * ((U.layout.shape u).m : ℝ) <
        CM.cost e.honest H U act σ s - CM.cost (e.embed (T.eval H)) H U act σ s) →
      T.hits (Bad (H xJ)) H ∨
        ∃ i < (U.layout.shape u).m, Bad (H xJ) (fin (T.eval H) i) (H (fin (T.eval H) i)) := by
    rintro H ⟨⟨hwf, hok, h32, hcor, hund⟩, act, σ, hsave⟩
    right
    have hc4 := hC4 (T.eval H) H act σ s
    have hslots : 100 * (U.layout.shape u).m < 11 * (T.eval H).slots.card := by
      have : (100 * (U.layout.shape u).m : ℝ) < 11 * (T.eval H).slots.card := by linarith
      exact_mod_cast this
    have hcount := fragSkip_count_le p G (sem.unitChain H s U u (T.eval H).act).1
      (sem.unitChain H s U u (T.eval H).act).2 (T.eval H).slots (T.eval H).patch
      (fun s hs => (hwf s hs).1) (fun s hs => (hwf s hs).2.1) (fun s hs => (hwf s hs).2.2.1)
      (fun s hs => (hwf s hs).2.2.2) hcor hund
    have hex : ∃ i : Fin (U.layout.shape u).m, 100 < fragCount p G (sem.unitChain H s U u (T.eval H).act).1
        (sem.unitChain H s U u (T.eval H).act).2 i := by
      by_contra hne
      push Not at hne
      have := sum_le_sum fun i (_ : i ∈ univ) => hne i
      rw [sum_const, card_univ, Fintype.card_fin, smul_eq_mul] at this
      omega
    obtain ⟨i, hi⟩ := hex
    exact ⟨i, i.isLt, i, (T.eval H).act, H, rfl, rfl, rfl, fun l => h32 i i.isLt l l.isLt, operandOK_rowCodes hok i, hi⟩
  calc _ ≤ pr (fun H : Q → R => T.hits (Bad (H xJ)) H ∨
          ∃ i < (U.layout.shape u).m, Bad (H xJ) (fin (T.eval H) i) (H (fin (T.eval H) i))) := pr_mono hstep
    _ ≤ (q + (U.layout.shape u).m) * (1 / 2 ^ 128) :=
        qtree_union_bound_at T q hq _ fin xJ Bad (by positivity) hJ hB
    _ = _ := by ring

/-- **(C6) can hold** with a finite oracle domain: `Q` holds the unit's rows with their 32-bit words, and one more
point, `none`, for the job's input. The key that tags a row with its words satisfies (C6) for a `sem` that reads no
oracle answer (zero noise, the exact forming chain, and unit scales and floor, as `witnessSem`). -/
theorem rowDrawn_satisfiable {R S : Type} (s : S) (U : Workload) (u : ℕ) :
    Nonempty (Fintype (Option (Fin (U.layout.shape u).m × (Fin (U.layout.shape u).k → Fin (2 ^ 32))))) ∧
      ∃ (sem : PearlCSem (Option (Fin (U.layout.shape u).m × (Fin (U.layout.shape u).k → Fin (2 ^ 32)))) R S)
        (key : ℕ → (Fin (U.layout.shape u).k → ℕ) →
          Option (Fin (U.layout.shape u).m × (Fin (U.layout.shape u).k → Fin (2 ^ 32)))),
        sem.RowDrawn s U u none key := by
  classical
  refine ⟨⟨inferInstance⟩, ⟨fun _ _ _ _ _ => ⟨fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ => 0⟩,
    FormChain.exact _, fun _ _ => 1, fun _ _ => 1, fun _ _ => 1⟩,
    fun i a => if h : i < (U.layout.shape u).m ∧ ∀ l, a l < 2 ^ 32 then some (⟨i, h.1⟩, fun l => ⟨a l, h.2 l⟩) else none,
    ?_, ?_, ?_, ?_⟩
  · intro i i' a a' ha ha' h
    dsimp only at h
    rw [dite_eq_left ⟨i.isLt, ha⟩, dite_eq_left ⟨i'.isLt, ha'⟩] at h
    have h1 := congrArg (fun o => o.map fun x => (x.1 : ℕ)) h
    have h2 : ∀ l, a l = a' l := fun l => by
      have := congrArg (fun o => o.map fun x => ((x.2 l : Fin (2 ^ 32)) : ℕ)) h
      simpa using this
    exact ⟨Fin.ext (by simpa using h1), funext h2⟩
  · intro i a ha
    dsimp only
    rw [dite_eq_left ⟨i.isLt, ha⟩]
    exact Option.some_ne_none _
  · intro _ _ _ _ _
    exact ⟨rfl, rfl, rfl⟩
  · intro _ _ _ _ _ _ _ _
    rfl

end Pouw.PearlC
