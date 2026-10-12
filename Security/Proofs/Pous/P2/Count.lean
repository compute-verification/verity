import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Fintype.Perm
import Mathlib.Data.Fintype.Powerset
import Mathlib.Tactic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# Lemma A over permutations of any finite type (generalizing `a2/A2.lean`)

`a2/A2.lean`'s lazy-sampling count with `Bits ℓ` replaced by any finite type `X` with `N = |X|` points, and the
responder given directly as query trees over `Fin B × X`. `count_main` is the counting form of Lemma A with `N^g` in
place of `2^(ℓ·g)`.
-/

namespace PousGen

open Pous Finset
open scoped ENNReal

/-! ## Query trees -/

/-- A query tree: one query per node. -/
inductive Tree (Q A α : Type) : Type where
  | ret : α → Tree Q A α
  | ask : Q → (A → Tree Q A α) → Tree Q A α

namespace Tree

variable {Q A α β : Type}

def run (f : Q → A) : Tree Q A α → α
  | ret a => a
  | ask q k => run f (k (f q))

def work (f : Q → A) : Tree Q A α → ℕ
  | ret _ => 0
  | ask q k => work f (k (f q)) + 1

def bind : Tree Q A α → (α → Tree Q A β) → Tree Q A β
  | ret a, g => g a
  | ask q k, g => ask q (fun a => bind (k a) g)

def askList : List Q → (List A → Tree Q A α) → Tree Q A α
  | [], c => c []
  | q :: qs, c => ask q (fun a => askList qs (fun as => c (a :: as)))

theorem run_bind (f : Q → A) (t : Tree Q A α) (g : α → Tree Q A β) :
    (t.bind g).run f = (g (t.run f)).run f := by
  induction t with
  | ret a => rfl
  | ask q k ih => exact ih (f q)

theorem work_bind (f : Q → A) (t : Tree Q A α) (g : α → Tree Q A β) :
    (t.bind g).work f = t.work f + (g (t.run f)).work f := by
  induction t with
  | ret a => simp [bind, work, run]
  | ask q k ih => simp only [bind, work, run, ih (f q)]; omega

theorem run_askList (f : Q → A) (qs : List Q) (c : List A → Tree Q A α) :
    (askList qs c).run f = (c (qs.map f)).run f := by
  induction qs generalizing c with
  | nil => rfl
  | cons q qs ih => exact ih _

theorem work_askList (f : Q → A) (qs : List Q) (c : List A → Tree Q A α) :
    (askList qs c).work f = (c (qs.map f)).work f + qs.length := by
  induction qs generalizing c with
  | nil => rfl
  | cons q qs ih => simp only [askList, work, List.map_cons, List.length_cons]; rw [ih]; omega

end Tree

/-- A `Prog` as a query tree, one query per node. -/
def toTree {M : IdealModel} {α : Type} : Prog M α → Tree M.Qry M.Ans α
  | .ret a => .ret a
  | .round qs k => Tree.askList qs (fun as => toTree (k as))

theorem run_toTree {M : IdealModel} {α : Type} (f : M.Qry → M.Ans) (p : Prog M α) :
    (toTree p).run f = p.run f := by
  induction p with
  | ret a => rfl
  | round qs k ih => simp only [toTree, Tree.run_askList, Prog.run]; exact ih _

theorem work_toTree {M : IdealModel} {α : Type} (f : M.Qry → M.Ans) (p : Prog M α) :
    (toTree p).work f = p.work f := by
  induction p with
  | ret a => rfl
  | round qs k ih => simp only [toTree, Tree.work_askList, Prog.work]; rw [ih]; omega

/-- The trees `p i`, `i ∈ l`, one after another; block `i` of the output is `p i`'s output. -/
def comb {Q A ι β : Type} [DecidableEq ι] (p : ι → Tree Q A β) (d : ι → β) : List ι → Tree Q A (ι → β)
  | [] => .ret d
  | i :: l => (p i).bind fun b => (comb p d l).bind fun F => .ret (Function.update F i b)

theorem run_comb {Q A ι β : Type} [DecidableEq ι] (p : ι → Tree Q A β) (d : ι → β) (f : Q → A) :
    ∀ (l : List ι) (i : ι), i ∈ l → (comb p d l).run f i = (p i).run f
  | [], i, h => by simp at h
  | j :: l, i, h => by
      simp only [comb, Tree.run_bind, Tree.run]
      by_cases hij : i = j
      · subst hij; simp
      · rw [Function.update_of_ne hij]
        exact run_comb p d f l i (by simpa [hij] using h)

theorem work_comb {Q A ι β : Type} [DecidableEq ι] (p : ι → Tree Q A β) (d : ι → β) (f : Q → A) (T : ℕ)
    (hp : ∀ i, (p i).work f ≤ T) : ∀ l : List ι, (comb p d l).work f ≤ l.length * T
  | [] => by simp [comb, Tree.work]
  | j :: l => by
      simp only [comb, Tree.work_bind, Tree.work, List.length_cons]
      have := work_comb p d f T hp l
      have := hp j
      nlinarith

/-! ## Counting permutation tuples along a transcript -/

section Count

variable {B : ℕ} {X : Type} [Fintype X] [DecidableEq X]

/-- The sample space of M1. -/
abbrev Om (B : ℕ) (X : Type) : Type := Fin B → Equiv.Perm (X)

/-- An answered query: `((j, x), y)` records `Π_j(x) = y`. -/
abbrev Ent (B : ℕ) (X : Type) : Type := (Fin B × X) × X

/-- The query tree the responder runs. -/
abbrev T0 (B : ℕ) (X : Type) : Type := Tree (Fin B × X) (X) (Fin B → X)

def ans (ω : Om B X) (q : Fin B × X) : X := ω q.1 q.2

open Classical in
/-- The permutation tuples consistent with the transcript `L`. -/
noncomputable def cons (L : List (Ent B X)) : Finset (Om B X) :=
  univ.filter fun ω => ∀ e ∈ L, ω e.1.1 e.1.2 = e.2

open Classical in
/-- The answers of block `j` recorded in `L`. -/
noncomputable def R (L : List (Ent B X)) (j : Fin B) : Finset (X) :=
  univ.filter fun y => ∃ x, ((j, x), y) ∈ L

open Classical in
/-- The blocks of `G` whose target has not been hit in `L`. -/
noncomputable def Gp (G : Finset (Fin B)) (Wv : Fin B → X) (L : List (Ent B X)) : Finset (Fin B) :=
  G.filter fun i => Wv i ∉ R L i

noncomputable def Pi (G : Finset (Fin B)) (Wv : Fin B → X) (L : List (Ent B X)) : ℕ :=
  ∏ i ∈ Gp G Wv L, (Fintype.card (X) - (R L i).card)

def Ssum (T g : ℕ) : ℕ := ∑ h ∈ range (g + 1), T.choose h

open Classical in
noncomputable def cnt (G : Finset (Fin B)) (Wv : Fin B → X) (t : T0 B X) (L : List (Ent B X)) : ℕ :=
  ((cons L).filter fun ω => ∀ i ∈ G, ω i (t.run (ans ω) i) = Wv i).card

theorem mem_cons {L : List (Ent B X)} {ω : Om B X} : ω ∈ cons L ↔ ∀ e ∈ L, ω e.1.1 e.1.2 = e.2 := by
  classical
  simp [cons]

theorem mem_R {L : List (Ent B X)} {j : Fin B} {y : X} : y ∈ R L j ↔ ∃ x, ((j, x), y) ∈ L := by
  classical
  simp [R]

theorem cons_cons (L : List (Ent B X)) (j : Fin B) (x y : X) :
    cons (((j, x), y) :: L) = (cons L).filter fun ω => ω j x = y := by
  classical
  ext ω
  simp only [mem_cons, mem_filter, List.mem_cons, forall_eq_or_imp]
  tauto

theorem cons_nil : cons ([] : List (Ent B X)) = univ := by
  classical
  ext ω; simp [mem_cons]

theorem R_cons_self (L : List (Ent B X)) (j : Fin B) (x y : X) :
    R (((j, x), y) :: L) j = insert y (R L j) := by
  classical
  ext v
  simp only [mem_R, List.mem_cons, Prod.mk.injEq, mem_insert]
  constructor
  · rintro ⟨x', (⟨⟨-, rfl⟩, rfl⟩ | h)⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨x', h⟩
  · rintro (rfl | ⟨x', h⟩)
    · exact ⟨x, Or.inl (by simp)⟩
    · exact ⟨x', Or.inr h⟩

theorem R_cons_ne (L : List (Ent B X)) {i j : Fin B} (h : i ≠ j) (x y : X) :
    R (((j, x), y) :: L) i = R L i := by
  classical
  ext v
  simp only [mem_R, List.mem_cons, Prod.mk.injEq]
  constructor
  · rintro ⟨x', (⟨⟨rfl, -⟩, -⟩ | h')⟩
    · exact absurd rfl h
    · exact ⟨x', h'⟩
  · rintro ⟨x', h'⟩; exact ⟨x', Or.inr h'⟩

/-- Splitting the consistent tuples by the answer to `(j, x)`. -/
theorem card_split (L : List (Ent B X)) (j : Fin B) (x : X) (P : Om B X → Prop) [DecidablePred P] :
    ((cons L).filter P).card = ∑ y, ((cons (((j, x), y) :: L)).filter P).card := by
  classical
  rw [card_eq_sum_card_fiberwise (f := fun ω => ω j x) (t := univ) (fun _ _ => mem_univ _)]
  refine sum_congr rfl fun y _ => ?_
  rw [cons_cons, filter_filter, filter_filter]
  congr 1
  ext ω; simp [and_comm]

/-- Swapping two answers unused by block `j` maps the tuples answering `y` at `(j, x)` into those answering `y'`. -/
theorem card_cons_le (L : List (Ent B X)) (j : Fin B) (x : X) {y y' : X} (hy : y ∉ R L j)
    (hy' : y' ∉ R L j) : (cons (((j, x), y) :: L)).card ≤ (cons (((j, x), y') :: L)).card := by
  classical
  let φ : Om B X → Om B X := fun ω => Function.update ω j (Equiv.swap y y' * ω j)
  refine card_le_card_of_injOn φ ?_ ?_
  · intro ω hω
    simp only [Finset.mem_coe] at hω ⊢
    rw [mem_cons] at hω ⊢
    intro e he
    rcases List.mem_cons.1 he with rfl | he
    · have := hω _ (List.mem_cons_self ..)
      simp only at this ⊢
      simp [φ, this]
    · have h1 := hω e (List.mem_cons_of_mem _ he)
      by_cases hej : e.1.1 = j
      · have hv : e.2 ∈ R L j := mem_R.2 ⟨e.1.2, by rw [← hej]; exact he⟩
        have hne : e.2 ≠ y := fun h => hy (h ▸ hv)
        have hne' : e.2 ≠ y' := fun h => hy' (h ▸ hv)
        simp only [φ, hej, Function.update_self, Equiv.Perm.coe_mul, Function.comp_apply]
        rw [← hej, h1, Equiv.swap_apply_of_ne_of_ne hne hne']
      · simp only [φ, Function.update_of_ne hej, h1]
  · intro ω _ ω' _ h
    have h2 := congrArg (fun ω => Function.update ω j (Equiv.swap y y' * ω j)) h
    simpa [φ, ← mul_assoc] using h2

/-- Each answer unused by block `j` is taken by at most a `1/(M − r)` fraction of the consistent tuples. -/
theorem cons_step_le (L : List (Ent B X)) (j : Fin B) (x : X) {y : X} (hy : y ∉ R L j) :
    (cons (((j, x), y) :: L)).card * (Fintype.card (X) - (R L j).card) ≤ (cons L).card := by
  classical
  have h := card_split L j x (fun _ => True)
  simp only [filter_true_of_mem (fun _ _ => trivial)] at h
  rw [h]
  calc (cons (((j, x), y) :: L)).card * (Fintype.card (X) - (R L j).card)
      = ∑ _ ∈ (R L j)ᶜ, (cons (((j, x), y) :: L)).card := by
        rw [sum_const, card_compl, smul_eq_mul, mul_comm]
    _ ≤ ∑ y' ∈ (R L j)ᶜ, (cons (((j, x), y') :: L)).card :=
        sum_le_sum fun y' hy' => card_cons_le L j x hy (mem_compl.1 hy')
    _ ≤ ∑ y', (cons (((j, x), y') :: L)).card := sum_le_sum_of_subset (subset_univ _)

theorem card_cons_split (L : List (Ent B X)) (j : Fin B) (x : X) :
    (cons L).card = ∑ y, (cons (((j, x), y) :: L)).card := by
  classical
  have h := card_split L j x (fun _ => True)
  simpa only [filter_true_of_mem (fun _ _ => trivial)] using h

theorem Ssum_pos (T g : ℕ) : 1 ≤ Ssum T g := by
  unfold Ssum
  calc 1 = T.choose 0 := (Nat.choose_zero_right T).symm
    _ ≤ _ := single_le_sum (f := fun h => T.choose h) (fun _ _ => Nat.zero_le _) (mem_range.2 (Nat.succ_pos g))

theorem Ssum_mono {T T' : ℕ} (h : T ≤ T') (g : ℕ) : Ssum T g ≤ Ssum T' g :=
  sum_le_sum fun _ _ => Nat.choose_le_choose _ h

theorem Ssum_succ (T g : ℕ) : Ssum (T + 1) (g + 1) = Ssum T (g + 1) + Ssum T g := by
  unfold Ssum
  rw [sum_range_succ' _ (g + 1), sum_range_succ' (fun h => T.choose h) (g + 1)]
  simp only [Nat.choose_succ_succ', sum_add_distrib, Nat.choose_zero_right]
  ring

variable (G : Finset (Fin B)) (Wv : Fin B → X)

/-- Guessing: the targets of the unhit blocks `s` at fixed points `o i` hold on at most a
`∏_{i ∈ s} 1/(M − r_i)` fraction of the consistent tuples. -/
theorem base (o : Fin B → X) : ∀ (s : Finset (Fin B)) (L : List (Ent B X)), (∀ i ∈ s, Wv i ∉ R L i) →
    ((cons L).filter fun ω => ∀ i ∈ s, ω i (o i) = Wv i).card *
      ∏ i ∈ s, (Fintype.card (X) - (R L i).card) ≤ (cons L).card := by
  classical
  intro s
  induction s using Finset.induction_on with
  | empty => intro L _; simp
  | @insert i s hi ih =>
    intro L hL
    have hR : ∀ i' ∈ s, R (((i, o i), Wv i) :: L) i' = R L i' := fun i' hi' =>
      R_cons_ne L (by rintro rfl; exact hi hi') _ _
    have h1 := ih (((i, o i), Wv i) :: L) (fun i' hi' => by rw [hR i' hi']; exact hL i' (mem_insert_of_mem hi'))
    have hp : ∏ i' ∈ s, (Fintype.card (X) - (R (((i, o i), Wv i) :: L) i').card) =
        ∏ i' ∈ s, (Fintype.card (X) - (R L i').card) := prod_congr rfl fun i' hi' => by rw [hR i' hi']
    rw [hp] at h1
    have h2 := cons_step_le L i (o i) (hL i (mem_insert_self i s))
    have heq : ((cons L).filter fun ω => ∀ i' ∈ insert i s, ω i' (o i') = Wv i') =
        (cons (((i, o i), Wv i) :: L)).filter fun ω => ∀ i' ∈ s, ω i' (o i') = Wv i' := by
      ext ω
      simp only [mem_filter, cons_cons, forall_mem_insert]
      tauto
    rw [heq, prod_insert hi]
    calc _ = (((cons (((i, o i), Wv i) :: L)).filter fun ω => ∀ i' ∈ s, ω i' (o i') = Wv i').card *
          ∏ i' ∈ s, (Fintype.card (X) - (R L i').card)) * (Fintype.card (X) - (R L i).card) := by ring
      _ ≤ (cons (((i, o i), Wv i) :: L)).card * (Fintype.card (X) - (R L i).card) :=
          Nat.mul_le_mul_right _ h1
      _ ≤ _ := h2

/-- **The invariant.** Along a transcript `L`, a tree of at most `T` further queries gets every block of `G` right
on at most a `Σ_{h ≤ |G'|} C(T, h) / ∏_{i ∈ G'} (M − r_i)` fraction of the consistent tuples. -/
theorem inv (t : T0 B X) : ∀ (L : List (Ent B X)) (T : ℕ), (∀ ω ∈ cons L, t.work (ans ω) ≤ T) →
    cnt G Wv t L * Pi G Wv L ≤ Ssum T (Gp G Wv L).card * (cons L).card := by
  classical
  induction t with
  | ret o =>
    intro L T _
    have hsub : cnt G Wv (.ret o) L ≤
        ((cons L).filter fun ω => ∀ i ∈ Gp G Wv L, ω i (o i) = Wv i).card := by
      apply card_le_card
      intro ω hω
      rw [mem_filter] at hω ⊢
      exact ⟨hω.1, fun i hi => hω.2 i (filter_subset _ _ hi)⟩
    have hb := base Wv o (Gp G Wv L) L (fun i hi => (mem_filter.1 hi).2)
    calc cnt G Wv (.ret o) L * Pi G Wv L
        ≤ ((cons L).filter fun ω => ∀ i ∈ Gp G Wv L, ω i (o i) = Wv i).card * Pi G Wv L :=
          Nat.mul_le_mul_right _ hsub
      _ ≤ (cons L).card := hb
      _ ≤ _ := Nat.le_mul_of_pos_left _ (Ssum_pos _ _)
  | ask q k ih =>
    obtain ⟨j, x⟩ := q
    intro L T hT
    rcases (cons L).eq_empty_or_nonempty with h0 | ⟨ω0, hω0⟩
    · simp [cnt, h0]
    obtain ⟨T', rfl⟩ : ∃ T', T = T' + 1 := by
      have := hT ω0 hω0
      simp only [Tree.work] at this
      exact ⟨T - 1, by omega⟩
    have hmem : ∀ y ω, ω ∈ cons (((j, x), y) :: L) → ω ∈ cons L ∧ ω j x = y := fun y ω hω => by
      rw [cons_cons, mem_filter] at hω; exact hω
    have hTy : ∀ y, ∀ ω ∈ cons (((j, x), y) :: L), (k y).work (ans ω) ≤ T' := by
      intro y ω hω
      obtain ⟨hω', hωy⟩ := hmem y ω hω
      have := hT ω hω'
      simp only [Tree.work, ans] at this
      rw [hωy] at this
      simpa [ans] using (by omega : (k y).work (ans ω) ≤ T')
    have hsplit : cnt G Wv (.ask (j, x) k) L = ∑ y, cnt G Wv (k y) (((j, x), y) :: L) := by
      unfold cnt
      rw [card_split L j x]
      refine sum_congr rfl fun y _ => ?_
      congr 1
      apply filter_congr
      intro ω hω
      have hωy := (hmem y ω hω).2
      simp only [Tree.run, ans, hωy]
    by_cases hj : j ∉ Gp G Wv L
    · -- a query to a block outside `G'`: nothing to hit
      have hGp : ∀ y, Gp G Wv (((j, x), y) :: L) = Gp G Wv L := by
        intro y; ext i
        simp only [Gp, mem_filter]
        by_cases hij : i = j
        · subst hij
          rw [R_cons_self]
          simp only [Gp, mem_filter, not_and, not_not] at hj
          constructor
          · rintro ⟨hG, hn⟩; exact absurd (mem_insert_of_mem (hj hG)) hn
          · rintro ⟨hG, hn⟩; exact absurd (hj hG) hn
        · rw [R_cons_ne L hij]
      have hPi : ∀ y, Pi G Wv (((j, x), y) :: L) = Pi G Wv L := by
        intro y; unfold Pi; rw [hGp y]
        refine prod_congr rfl fun i hi => ?_
        have : i ≠ j := fun h => hj (h ▸ hi)
        rw [R_cons_ne L this]
      rw [hsplit, sum_mul]
      calc ∑ y, cnt G Wv (k y) (((j, x), y) :: L) * Pi G Wv L
          = ∑ y, cnt G Wv (k y) (((j, x), y) :: L) * Pi G Wv (((j, x), y) :: L) := by simp only [hPi]
        _ ≤ ∑ y, Ssum T' (Gp G Wv (((j, x), y) :: L)).card * (cons (((j, x), y) :: L)).card :=
            sum_le_sum fun y _ => ih y _ _ (hTy y)
        _ = Ssum T' (Gp G Wv L).card * (cons L).card := by
            simp only [hGp]; rw [← mul_sum, ← card_cons_split]
        _ ≤ _ := Nat.mul_le_mul_right _ (Ssum_mono (by omega) _)
    push Not at hj
    have hWj : Wv j ∉ R L j := (mem_filter.1 hj).2
    by_cases hx : ∃ v, ((j, x), v) ∈ L
    · -- a repeated query: its answer is known
      obtain ⟨v, hv⟩ := hx
      have hconsv : cons (((j, x), v) :: L) = cons L := by
        rw [cons_cons]
        exact filter_true_of_mem fun ω hω => mem_cons.1 hω _ hv
      have hRv : ∀ i, R (((j, x), v) :: L) i = R L i := by
        intro i
        by_cases hij : i = j
        · subst hij; rw [R_cons_self, insert_eq_of_mem (mem_R.2 ⟨x, hv⟩)]
        · rw [R_cons_ne L hij]
      have hGp : Gp G Wv (((j, x), v) :: L) = Gp G Wv L := by simp only [Gp, hRv]
      have hPi : Pi G Wv (((j, x), v) :: L) = Pi G Wv L := by simp only [Pi, hGp, hRv]
      have hc : cnt G Wv (.ask (j, x) k) L = cnt G Wv (k v) (((j, x), v) :: L) := by
        unfold cnt
        rw [hconsv]
        congr 1
        apply filter_congr
        intro ω hω
        have : ω j x = v := mem_cons.1 hω _ hv
        simp only [Tree.run, ans, this]
      have := ih v _ _ (hTy v)
      rw [hGp, hPi, hconsv] at this
      rw [hc]
      exact this.trans (Nat.mul_le_mul_right _ (Ssum_mono (by omega) _))
    push Not at hx
    -- a fresh query to a block of `G'`: a hit, or a miss that shrinks the free range
    obtain ⟨m, hm⟩ : ∃ m, Fintype.card (X) - (R L j).card = m := ⟨_, rfl⟩
    obtain ⟨P0, hP0⟩ : ∃ P0, ∏ i ∈ (Gp G Wv L).erase j, (Fintype.card (X) - (R L i).card) = P0 :=
      ⟨_, rfl⟩
    have hPiL : Pi G Wv L = m * P0 := by rw [← hm, ← hP0]; exact (mul_prod_erase _ _ hj).symm
    have hempty : ∀ y ∈ R L j, cnt G Wv (k y) (((j, x), y) :: L) = 0 := by
      intro y hy
      obtain ⟨x', hx'⟩ := mem_R.1 hy
      unfold cnt
      rw [card_eq_zero, filter_eq_empty_iff]
      intro ω hω
      obtain ⟨hω', hωy⟩ := hmem y ω hω
      have h1 : ω j x' = y := mem_cons.1 hω' _ hx'
      have : x' = x := (ω j).injective (h1.trans hωy.symm)
      exact absurd (this ▸ hx') (hx y)
    have hRne : ∀ i ∈ (Gp G Wv L).erase j, ∀ y, R (((j, x), y) :: L) i = R L i :=
      fun i hi y => R_cons_ne L (ne_of_mem_erase hi) _ _
    -- the hit
    have hGpW : Gp G Wv (((j, x), Wv j) :: L) = (Gp G Wv L).erase j := by
      ext i
      simp only [Gp, mem_filter, mem_erase]
      by_cases hij : i = j
      · subst hij; simp [R_cons_self]
      · rw [R_cons_ne L hij]; tauto
    have hPiW : Pi G Wv (((j, x), Wv j) :: L) = P0 := by
      unfold Pi; rw [hGpW, ← hP0]
      exact prod_congr rfl fun i hi => by rw [hRne i hi]
    have hcardE : ((Gp G Wv L).erase j).card + 1 = (Gp G Wv L).card := card_erase_add_one hj
    have hW : cnt G Wv (k (Wv j)) (((j, x), Wv j) :: L) * (m * P0) ≤
        Ssum T' ((Gp G Wv L).erase j).card * (cons L).card := by
      have h := ih (Wv j) _ _ (hTy (Wv j))
      rw [hGpW, hPiW] at h
      calc _ = (cnt G Wv (k (Wv j)) (((j, x), Wv j) :: L) * P0) * m := by ring
        _ ≤ (Ssum T' ((Gp G Wv L).erase j).card * (cons (((j, x), Wv j) :: L)).card) * m :=
            Nat.mul_le_mul_right _ h
        _ = Ssum T' ((Gp G Wv L).erase j).card * ((cons (((j, x), Wv j) :: L)).card * m) := by ring
        _ ≤ _ := Nat.mul_le_mul_left _ (hm ▸ cons_step_le L j x hWj)
    -- the misses
    obtain ⟨rest, hrest0⟩ : ∃ rest, (insert (Wv j) (R L j))ᶜ = rest := ⟨_, rfl⟩
    have hrest : rest.card = m - 1 := by
      rw [← hrest0, card_compl, card_insert_of_notMem hWj]; omega
    have hmiss : ∀ y ∈ rest, cnt G Wv (k y) (((j, x), y) :: L) * (m * P0) * (m - 1) ≤
        Ssum T' (Gp G Wv L).card * (cons L).card := by
      intro y hy
      rw [← hrest0] at hy
      simp only [mem_compl, mem_insert, not_or] at hy
      have hGpy : Gp G Wv (((j, x), y) :: L) = Gp G Wv L := by
        ext i
        simp only [Gp, mem_filter]
        by_cases hij : i = j
        · subst hij; rw [R_cons_self]; simp only [mem_insert, not_or]
          exact ⟨fun h => ⟨h.1, h.2.2⟩, fun h => ⟨h.1, Ne.symm hy.1, h.2⟩⟩
        · rw [R_cons_ne L hij]
      have hPiy : Pi G Wv (((j, x), y) :: L) = (m - 1) * P0 := by
        unfold Pi; rw [hGpy, ← mul_prod_erase _ _ hj, R_cons_self, card_insert_of_notMem hy.2, ← hP0]
        have e1 : Fintype.card (X) - ((R L j).card + 1) = m - 1 := by omega
        exact congrArg₂ (· * ·) e1 (prod_congr rfl fun i hi => by rw [hRne i hi])
      have h := ih y _ _ (hTy y)
      rw [hGpy, hPiy] at h
      have hs := hm ▸ cons_step_le L j x hy.2
      generalize cnt G Wv (k y) (((j, x), y) :: L) = c at h ⊢
      generalize (cons (((j, x), y) :: L)).card = d at h hs ⊢
      calc _ = (c * ((m - 1) * P0)) * m := by ring
        _ ≤ (Ssum T' (Gp G Wv L).card * d) * m := Nat.mul_le_mul_right _ h
        _ = Ssum T' (Gp G Wv L).card * (d * m) := by ring
        _ ≤ _ := Nat.mul_le_mul_left _ hs
    have hmiss' : ∑ y ∈ rest, cnt G Wv (k y) (((j, x), y) :: L) * (m * P0) ≤
        Ssum T' (Gp G Wv L).card * (cons L).card := by
      rcases Nat.eq_zero_or_pos (m - 1) with h0 | hpos
      · rw [card_eq_zero.1 (hrest.trans h0), sum_empty]; exact Nat.zero_le _
      · refine Nat.le_of_mul_le_mul_right ?_ hpos
        rw [sum_mul]
        calc _ ≤ ∑ _y ∈ rest, Ssum T' (Gp G Wv L).card * (cons L).card := sum_le_sum hmiss
          _ = _ := by rw [sum_const, hrest, smul_eq_mul, mul_comm]
    have hsum : ∑ y, cnt G Wv (k y) (((j, x), y) :: L) * (m * P0) =
        cnt G Wv (k (Wv j)) (((j, x), Wv j) :: L) * (m * P0) +
          ∑ y ∈ rest, cnt G Wv (k y) (((j, x), y) :: L) * (m * P0) := by
      rw [← sum_add_sum_compl (insert (Wv j) (R L j)), hrest0, sum_insert hWj,
        sum_eq_zero (fun y hy => by rw [hempty y hy, zero_mul]), add_zero]
    rw [hsplit, hPiL, sum_mul, hsum, ← hcardE, Ssum_succ, add_mul, add_comm (Ssum T' _ * _)]
    exact add_le_add hW (hcardE ▸ hmiss')

end Count

/-! ## One state, one set of blocks, and the union -/

section Main

variable {B : ℕ} {X : Type} [Fintype X] [DecidableEq X]

theorem Gp_nil (G : Finset (Fin B)) (Wv : Fin B → X) : Gp G Wv [] = G := by
  classical
  ext i; simp [Gp, mem_R]

theorem Pi_nil (G : Finset (Fin B)) (Wv : Fin B → X) :
    Pi G Wv [] = Fintype.card X ^ G.card := by
  classical
  have hR : ∀ i, R ([] : List (Ent B X)) i = ∅ := fun i => by ext y; simp [mem_R]
  simp [Pi, Gp_nil, hR]

/-- The `|G|` trees of one state, run one after another: all of `G` correct on at most a
`Σ_{h ≤ |G|} C(|G|·Q, h) / N^|G|` fraction of the permutation tuples, `N = |X|`. -/
theorem count_G (p : Fin B → Tree (Fin B × X) X X) (Q : ℕ) (hp : ∀ i (ω : Om B X), (p i).work (ans ω) ≤ Q)
    (d : Fin B → X) (Wv : Fin B → X) (G : Finset (Fin B)) :
    (univ.filter fun ω : Om B X => ∀ i ∈ G, ω i ((p i).run (ans ω)) = Wv i).card *
        Fintype.card X ^ G.card ≤ Ssum (G.card * Q) G.card * Fintype.card (Om B X) := by
  classical
  have h := inv G Wv (comb p d G.toList) [] (G.card * Q) (fun ω _ => by
    have := work_comb p d (ans ω) Q (fun i => hp i ω) G.toList
    rwa [length_toList] at this)
  rw [Gp_nil, Pi_nil, cons_nil, card_univ] at h
  have e : cnt G Wv (comb p d G.toList) [] =
      (univ.filter fun ω : Om B X => ∀ i ∈ G, ω i ((p i).run (ans ω)) = Wv i).card := by
    unfold cnt
    rw [cons_nil]
    congr 1
    apply filter_congr
    intro ω _
    refine forall₂_congr fun i hi => ?_
    rw [run_comb _ _ _ _ i (mem_toList.2 hi)]
  rw [← e]
  exact h

/-- **Lemma A over any finite domain, as a count.** The union over the `2^S` states and the `C(B, g)` sets of `g`
blocks. -/
theorem count_main (S Q g : ℕ) (p : Bits S → Fin B → Tree (Fin B × X) X X)
    (hp : ∀ σ i (ω : Om B X), (p σ i).work (ans ω) ≤ Q) (d : Fin B → X) (Wv : Fin B → X)
    (σf : Om B X → Bits S) :
    (univ.filter fun ω : Om B X =>
        g ≤ (univ.filter fun i => ω i ((p (σf ω) i).run (ans ω)) = Wv i).card).card *
      Fintype.card X ^ g ≤ 2 ^ S * B.choose g * Ssum ((g + 1) * Q) g * Fintype.card (Om B X) := by
  classical
  let EG : Bits S → Finset (Fin B) → Finset (Om B X) := fun σ G =>
    univ.filter fun ω : Om B X => ∀ i ∈ G, ω i ((p σ i).run (ans ω)) = Wv i
  have hsub : (univ.filter fun ω : Om B X =>
      g ≤ (univ.filter fun i => ω i ((p (σf ω) i).run (ans ω)) = Wv i).card) ⊆
      (univ : Finset (Bits S)).biUnion fun σ => (powersetCard g (univ : Finset (Fin B))).biUnion (EG σ) := by
    intro ω hω
    rw [mem_filter] at hω
    obtain ⟨G, hG, hGc⟩ := exists_subset_card_eq hω.2
    rw [mem_biUnion]
    refine ⟨σf ω, mem_univ _, ?_⟩
    rw [mem_biUnion]
    refine ⟨G, mem_powersetCard.2 ⟨subset_univ _, hGc⟩, ?_⟩
    rw [mem_filter]
    exact ⟨mem_univ _, fun i hi => (mem_filter.1 (hG hi)).2⟩
  have hcount : ∀ σ, ∀ G ∈ powersetCard g (univ : Finset (Fin B)),
      (EG σ G).card * Fintype.card X ^ g ≤ Ssum (g * Q) g * Fintype.card (Om B X) := by
    intro σ G hG
    rw [mem_powersetCard] at hG
    have := count_G (p σ) Q (hp σ) d Wv G
    rwa [hG.2] at this
  refine (Nat.mul_le_mul_right _ ((card_le_card hsub).trans card_biUnion_le)).trans ?_
  rw [sum_mul]
  calc ∑ σ : Bits S, ((powersetCard g (univ : Finset (Fin B))).biUnion (EG σ)).card * Fintype.card X ^ g
      ≤ ∑ σ : Bits S, ∑ G ∈ powersetCard g (univ : Finset (Fin B)), (EG σ G).card * Fintype.card X ^ g :=
        sum_le_sum fun σ _ => by rw [← sum_mul]; exact Nat.mul_le_mul_right _ card_biUnion_le
    _ ≤ ∑ σ : Bits S, ∑ G ∈ powersetCard g (univ : Finset (Fin B)), Ssum (g * Q) g * Fintype.card (Om B X) :=
        sum_le_sum fun σ _ => sum_le_sum fun G hG => hcount σ G hG
    _ = 2 ^ S * B.choose g * Ssum (g * Q) g * Fintype.card (Om B X) := by
        rw [sum_const, sum_const, card_powersetCard, card_univ, card_univ, Fintype.card_fin, smul_eq_mul,
          smul_eq_mul]
        have hS : Fintype.card (Bits S) = 2 ^ S := by simp
        rw [hS]; ring
    _ ≤ _ := Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ (Ssum_mono (Nat.mul_le_mul_right _ (by omega)) g))

end Main

/-- Filters by equivalent predicates, whatever their decidability instances, have the same size. -/
theorem card_filter_congr {α : Type} [Fintype α] (P P' : α → Prop) (d : DecidablePred P) (d' : DecidablePred P')
    (h : ∀ x, P x ↔ P' x) : (@Finset.filter α P d univ).card = (@Finset.filter α P' d' univ).card := by
  congr 1
  ext x
  simp [h]

theorem frac_le {N K X P : ℕ} (hK : 0 < K) (hP : 0 < P) (h : N * P ≤ X * K) :
    (N : ℝ≥0∞) / K ≤ (X : ℝ≥0∞) / P := by
  rw [ENNReal.div_le_iff (by exact_mod_cast hK.ne') (ENNReal.natCast_ne_top _), div_eq_mul_inv, mul_right_comm,
    ← div_eq_mul_inv, ENNReal.le_div_iff_mul_le (Or.inl (by exact_mod_cast hP.ne'))
      (Or.inl (ENNReal.natCast_ne_top _))]
  exact_mod_cast h

end PousGen
