module

public import Definitions.Core.Guarantees.MemCheck
public import Mathlib.Algebra.MvPolynomial.Degrees
public import Mathlib.Algebra.Polynomial.Degree.Defs
public import Mathlib.Algebra.Polynomial.Eval.Defs
public import Mathlib.FieldTheory.Finite.GaloisField
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.MvPolynomial.CommRing
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Data.Fin.VecNotation
import Mathlib.Logic.Equiv.Fin.Basic
import Proofs.Core.Game.Basic
import Proofs.Core.Game.Poly
import Proofs.Core.Game.Prob
import Proofs.Core.Game.Tape
import Proofs.Core.ValueSalts

/-!
# What a memory check establishes, and what a fingerprint collision costs it: proofs

`MemoryChecked`. Times increase along the log and only the initial contents are at time `0`, so the writes with the
initial contents are distinct records; the check makes them the reads with the final contents, so no two accesses read
one record and no access reads a final content. By strong induction on the access, each read finds its address's last
write before it, or its initial value: the record it reads is a write's or an initial content, earlier than it by the
time test, and a first access to the address in between would, by induction, have read that same record. The final
contents follow the same way, and an address the log accesses is touched since its first access reads an initial
content. `MemoryChecked_honest`, a log of three accesses to two addresses passes the check;
`MemoryChecked_nontrivial`, an access that reads its own write did not read its memory correctly, though its records
pass the multiset test.

`FingerprintCollision`. A multiset's fingerprint is the value at `(α, γ)` of the polynomial
`∏ c, (X 0 + Σ_i C(c_i) X 1^i)` (`fingerprint2`), of total degree at most `|A| · max 1 (m - 1)`. Two different multisets
give different polynomials (`fingerprint2_injective`): sent to `F[γ][α]`, each is `∏ c, (α - g c)` with
`g c = -Σ_i C(c_i) γ^i`, injective in `c`, and the roots of that product are the multiset of the `g c`. So the difference
is a nonzero polynomial of total degree at most `max |A| |B| · max 1 (m - 1)`, and Schwartz–Zippel
(`Game.prCoin_mvEval_eq_zero_le`) bounds the coins at which it vanishes.

`PrintCollision`. Two different keys' prints differ by a nonzero polynomial of degree at most `m - 1` in `δ`
(`chunkPoly`), which vanishes at a uniform `δ` with probability at most `(m - 1) / |F|`; a union bound over the
`C(|S|, 2)` pairs, by induction on `S`, gives the rest.

Witnesses, at GF(2^128) (Mathlib's `GaloisField 2 128`): `FingerprintCollision_honest`, an honest log's two sides (one
multiset of two-chunk records listed in two orders) have equal fingerprints at every coin, and against a different
multiset the bound at two records of two chunks holds, `2 / 2^128`; `FingerprintCollision_nontrivial`, a check that
accepts every pair of multisets accepts two different ones with probability 1, above the bound `1 / 2^128`.
`PrintCollision_honest`, two keys of four chunks share a print with probability at most `3 / 2^128`;
`PrintCollision_nontrivial`, a check that takes every two keys to share a print does so with probability 1, above
`1 / 2^128`.
-/

@[expose] public section

namespace Verity.MemCheck

open Polynomial
open scoped ENNReal

section

variable {F : Type} [Field F]

/-- A record's chunks as a polynomial in `γ`: `Σ_i C(c_i) X^i`. -/
noncomputable def chunkPoly {m : ℕ} (c : Fin m → F) : F[X] :=
  ∑ i : Fin m, C (c i) * X ^ (i : ℕ)

theorem eval_chunkPoly {m : ℕ} (c : Fin m → F) (γ : F) : (chunkPoly c).eval γ = element γ c := by
  simp only [chunkPoly, element, eval_finsetSum, eval_mul, eval_C, eval_pow, eval_X]
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

theorem coeff_chunkPoly {m : ℕ} (c : Fin m → F) (j : Fin m) : (chunkPoly c).coeff j = c j := by
  classical
  rw [chunkPoly, finsetSum_coeff, Finset.sum_eq_single j]
  · simp
  · intro i _ hij
    simp [Fin.val_inj, Ne.symm hij]
  · simp

theorem chunkPoly_injective {m : ℕ} : Function.Injective (chunkPoly (F := F) (m := m)) := by
  intro c c' h
  funext j
  rw [← coeff_chunkPoly c j, ← coeff_chunkPoly c' j, h]

theorem natDegree_chunkPoly_le {m : ℕ} (c : Fin m → F) : (chunkPoly c).natDegree ≤ m - 1 :=
  natDegree_sum_le_of_forall_le _ _ fun i _ =>
    (natDegree_C_mul_X_pow_le _ _).trans (by have := i.isLt; omega)

/-- A record's factor `α + Σ_i γ^i c_i` as a polynomial in `(α, γ) = (X 0, X 1)`. -/
noncomputable def factor2 {m : ℕ} (c : Fin m → F) : MvPolynomial (Fin 2) F :=
  MvPolynomial.X 0 + ∑ i : Fin m, MvPolynomial.C (c i) * MvPolynomial.X 1 ^ (i : ℕ)

/-- A multiset's fingerprint as a polynomial in `(α, γ) = (X 0, X 1)`. -/
noncomputable def fingerprint2 {m : ℕ} (A : Multiset (Fin m → F)) : MvPolynomial (Fin 2) F :=
  (A.map factor2).prod

theorem eval_factor2 {m : ℕ} (c : Fin m → F) (f : Fin 2 → F) :
    MvPolynomial.eval f (factor2 c) = f 0 + element (f 1) c := by
  simp only [factor2, element, map_add, map_sum, map_mul, map_pow, MvPolynomial.eval_X, MvPolynomial.eval_C]
  congr 1
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

theorem eval_fingerprint2 {m : ℕ} (A : Multiset (Fin m → F)) (f : Fin 2 → F) :
    MvPolynomial.eval f (fingerprint2 A) = fingerprint (f 0) (f 1) A := by
  simp only [fingerprint2, fingerprint, map_multiset_prod, Multiset.map_map, Function.comp_def, eval_factor2]

theorem totalDegree_factor2_le {m : ℕ} (c : Fin m → F) : (factor2 c).totalDegree ≤ max 1 (m - 1) := by
  refine (MvPolynomial.totalDegree_add _ _).trans (max_le_max ?_ ?_)
  · exact (MvPolynomial.totalDegree_X _).le
  · refine (MvPolynomial.totalDegree_finsetSum _ _).trans (Finset.sup_le fun i _ => ?_)
    refine (MvPolynomial.totalDegree_mul _ _).trans ?_
    rw [MvPolynomial.totalDegree_C, MvPolynomial.totalDegree_X_pow, zero_add]
    have := i.isLt
    omega

theorem totalDegree_fingerprint2_le {m : ℕ} (A : Multiset (Fin m → F)) :
    (fingerprint2 A).totalDegree ≤ A.card * max 1 (m - 1) := by
  refine (MvPolynomial.totalDegree_multiset_prod _).trans ?_
  have h := Multiset.sum_le_card_nsmul ((A.map factor2).map MvPolynomial.totalDegree) (max 1 (m - 1)) (by
    simp only [Multiset.map_map, Function.comp_def, Multiset.mem_map]
    rintro _ ⟨c, -, rfl⟩
    exact totalDegree_factor2_le _)
  simpa using h

/-- Two different multisets have different fingerprint polynomials: sent to `F[γ][α]`, each is the product of
`α - (-Σ_i C(c_i) γ^i)` over its records, whose roots are the multiset of the `-Σ_i C(c_i) γ^i`. -/
theorem fingerprint2_injective {m : ℕ} : Function.Injective (fingerprint2 (F := F) (m := m)) := by
  intro A B h
  let φ : MvPolynomial (Fin 2) F →+* Polynomial (Polynomial F) :=
    MvPolynomial.eval₂Hom (Polynomial.C.comp Polynomial.C) ![Polynomial.X, Polynomial.C Polynomial.X]
  let g : (Fin m → F) → Polynomial F := fun c => -chunkPoly c
  have hg : Function.Injective g := fun c c' hcc' => chunkPoly_injective (neg_injective hcc')
  have hφ : ∀ A : Multiset (Fin m → F),
      φ (fingerprint2 A) = ((A.map g).map fun a => Polynomial.X - Polynomial.C a).prod := by
    intro A
    simp only [fingerprint2, map_multiset_prod, Multiset.map_map, Function.comp_def]
    congr 1
    refine Multiset.map_congr rfl fun c _ => ?_
    simp only [φ, g, factor2, chunkPoly, map_add, map_sum, map_mul, map_pow, MvPolynomial.eval₂Hom_X',
      MvPolynomial.eval₂Hom_C, RingHom.coe_comp, Function.comp_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
      sub_neg_eq_add, map_neg]
  have := congrArg (fun p => (φ p).roots) h
  simp only [hφ, Polynomial.roots_multiset_prod_X_sub_C] at this
  exact Multiset.map_injective hg this

end

/-- Relabelling a uniform coin by an equivalence keeps an event's probability. -/
theorem prCoin_equiv {C D : Type} [Fintype C] [Fintype D] (e : C ≃ D) (p : D → Prop) :
    Game.prCoin (fun c => p (e c)) = Game.prCoin p := by
  classical
  unfold Game.prCoin
  exact Game.Tape.avg_equiv e fun d => if p d then 1 else 0

variable {F : Type} [Field F] [Fintype F]

/-- Two different packed keys share a print at a uniform `δ` with probability at most `(m - 1) / |F|`. -/
theorem prCoin_element_eq_le {m : ℕ} (c c' : Fin m → F) (h : c ≠ c') :
    Game.prCoin (fun δ : F => element δ c = element δ c') ≤ ((m - 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
  classical
  have hne : chunkPoly c - chunkPoly c' ≠ 0 := sub_ne_zero.2 fun e => h (chunkPoly_injective e)
  calc Game.prCoin (fun δ : F => element δ c = element δ c')
      = Game.prCoin (fun δ : F => (chunkPoly c - chunkPoly c').eval δ = 0) := by
        congr 1
        funext δ
        simp [eval_chunkPoly, sub_eq_zero]
    _ ≤ ((chunkPoly c - chunkPoly c').natDegree : ℝ≥0∞) / Fintype.card F :=
        Game.prCoin_eval_eq_zero_le _ hne
    _ ≤ ((m - 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
        gcongr
        exact (natDegree_sub_le _ _).trans (max_le (natDegree_chunkPoly_le c) (natDegree_chunkPoly_le c'))

theorem card_gf128 : Fintype.card (GaloisField 2 128) = 2 ^ 128 := by
  rw [Fintype.card_eq_nat_card, GaloisField.card 2 128 (by norm_num)]

section

variable {A V : Type} [DecidableEq A]

/-- The memory after access `i` is the memory before it with `i`'s write at its address. -/
theorem ram_take_succ (v0 : A → V) (log : List (Access A V)) (i : ℕ) (h : i < log.length) :
    ram v0 (log.take (i + 1)) = Function.update (ram v0 (log.take i)) log[i].a (log[i].w, log[i].now) := by
  simp only [ram, List.take_add_one, List.getElem?_eq_getElem h, Option.toList_some, List.foldl_append,
    List.foldl_cons, List.foldl_nil]

/-- Before any access to `a`, the memory holds `a`'s initial value at time `0`. -/
theorem ram_take_of_none (v0 : A → V) (log : List (Access A V)) (a : A) :
    ∀ i, i ≤ log.length → (∀ k (hk : k < log.length), k < i → log[k].a ≠ a) →
      ram v0 (log.take i) a = (v0 a, 0)
  | 0, _, _ => by simp [ram]
  | i + 1, hi, hn => by
    rw [ram_take_succ v0 log i (by omega), Function.update_of_ne (hn i (by omega) (by omega)).symm]
    exact ram_take_of_none v0 log a i (by omega) fun k hk hki => hn k hk (by omega)

/-- After an access `j` to `a`, and no other access to `a` before `i`, the memory holds `j`'s write at `a`. -/
theorem ram_take_of_last (v0 : A → V) (log : List (Access A V)) (a : A) (j : ℕ) (hj : j < log.length)
    (hja : log[j].a = a) :
    ∀ i, j < i → i ≤ log.length → (∀ k (hk : k < log.length), j < k → k < i → log[k].a ≠ a) →
      ram v0 (log.take i) a = (log[j].w, log[j].now)
  | 0, h, _, _ => absurd h (Nat.not_lt_zero _)
  | i + 1, hji, hi, hn => by
    rw [ram_take_succ v0 log i (by omega)]
    rcases Nat.lt_or_ge j i with hlt | hge
    · rw [Function.update_of_ne (hn i (by omega) hlt (by omega)).symm]
      exact ram_take_of_last v0 log a j hj hja i hlt (by omega) fun k hk hjk hki => hn k hk hjk (by omega)
    · obtain rfl : j = i := by omega
      rw [← hja, Function.update_self]

end

end Verity.MemCheck

namespace Verity.SecurityProofs

open Verity.MemCheck Game
open scoped ENNReal

theorem MemoryChecked : Guarantees.MemoryChecked := by
  intro A V _ v0 log touched fin ⟨hinc, hlt, heq⟩
  classical
  set rd : Access A V → A × V × ℕ := fun x => (x.a, x.v, x.t)
  set wr : Access A V → A × V × ℕ := fun x => (x.a, x.w, x.now)
  have hord' : ∀ j i (hj : j < log.length) (hi : i < log.length), log[j].now < log[i].now → j < i := by
    intro j i hj hi h
    by_contra hc
    rcases Nat.lt_or_ge i j with h' | h'
    · exact absurd (List.pairwise_iff_getElem.1 hinc i j hi hj h') (by omega)
    · obtain rfl : j = i := by omega
      exact lt_irrefl _ h
  -- the writes with the initial contents are distinct (times increase, and only the initial ones are at `0`), so the
  -- reads with the final contents are too
  have hW : (writes log + touched.val.map fun a => (a, v0 a, 0)).Nodup := by
    refine Multiset.nodup_add.2 ⟨?_, touched.nodup.map fun a b h => by simpa using congrArg Prod.fst h, ?_⟩
    · refine Multiset.coe_nodup.2 (List.Pairwise.imp ?_
        ((List.pairwise_map (R := fun p q : A × V × ℕ => p.2.2 < q.2.2)).2 hinc))
      intro x y h e
      exact absurd (congrArg (fun p => p.2.2) e) (ne_of_lt h)
    · refine Multiset.disjoint_left.2 fun {p} hp hq => ?_
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 (Multiset.mem_coe.1 hp)
      obtain ⟨b, -, hb⟩ := Multiset.mem_map.1 hq
      have := hlt x hx
      have h2 := congrArg (fun p => p.2.2) hb
      simp at h2
      omega
  rw [← heq] at hW
  obtain ⟨hR, -, hRF⟩ := Multiset.nodup_add.1 hW
  have hR' : (log.map rd).Nodup := Multiset.coe_nodup.1 hR
  have hrdinj : ∀ k i (hk : k < log.length) (hi : i < log.length), rd log[k] = rd log[i] → k = i :=
    fun k i hk hi h => (List.Nodup.getElem_inj_iff hR' (i := k) (j := i) (hi := by simpa using hk)
      (hj := by simpa using hi)).1 (by simpa using h)
  have hrdfin : ∀ k (hk : k < log.length), ∀ a ∈ touched, rd log[k] ≠ (a, fin a) := by
    intro k hk a ha h
    refine Multiset.disjoint_left.1 hRF (Multiset.mem_coe.2 (List.mem_map.2 ⟨log[k], List.getElem_mem hk, h⟩)) ?_
    exact Multiset.mem_map.2 ⟨a, ha, rfl⟩
  -- each record of the reads and final contents is a write's or an initial content
  have hsrc : ∀ p ∈ (reads log + touched.val.map fun a => (a, fin a)),
      (∃ j, ∃ hj : j < log.length, wr log[j] = p) ∨ ∃ a ∈ touched, (a, v0 a, 0) = p := by
    intro p hp
    rw [heq] at hp
    rcases Multiset.mem_add.1 hp with hp | hp
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 (Multiset.mem_coe.1 hp)
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
      exact Or.inl ⟨j, hj, rfl⟩
    · obtain ⟨a, ha, rfl⟩ := Multiset.mem_map.1 hp
      exact Or.inr ⟨a, ha, rfl⟩
  have hread_mem : ∀ i (hi : i < log.length), rd log[i] ∈ (reads log + touched.val.map fun a => (a, fin a)) :=
    fun i hi => Multiset.mem_add.2 (Or.inl (Multiset.mem_coe.2 (List.mem_map.2 ⟨log[i], List.getElem_mem hi, rfl⟩)))
  have hfin_mem : ∀ a ∈ touched, (a, fin a) ∈ (reads log + touched.val.map fun a => (a, fin a)) :=
    fun a ha => Multiset.mem_add.2 (Or.inr (Multiset.mem_map.2 ⟨a, ha, rfl⟩))
  -- the reads, by strong induction on the access: an access reading another's write, or an initial content, while a
  -- first access to its address in between read the same record, would be two accesses reading one record
  have hreads : ∀ i (h : i < log.length), (log[i].v, log[i].t) = ram v0 (log.take i) log[i].a := by
    intro i
    induction i using Nat.strong_induction_on with
    | _ i ih =>
    intro hi
    rcases hsrc _ (hread_mem i hi) with ⟨j, hj, hji⟩ | ⟨a, -, hai⟩
    · simp only [wr, rd, Prod.mk.injEq] at hji
      obtain ⟨ha, hv, ht⟩ := hji
      have hj_lt : j < i := hord' j i hj hi (ht ▸ hlt _ (List.getElem_mem hi))
      have hnone : ∀ k (hk : k < log.length), j < k → k < i → log[k].a ≠ log[i].a := by
        intro k0 hk0 h1 h2 e0
        have hex : ∃ k, ∃ hk : k < log.length, j < k ∧ k < i ∧ log[k].a = log[i].a := ⟨k0, hk0, h1, h2, e0⟩
        obtain ⟨hk, hjk, hki, hka⟩ := Nat.find_spec hex
        have hmin : ∀ k' (hk' : k' < log.length), j < k' → k' < Nat.find hex → log[k'].a ≠ log[i].a :=
          fun k' hk' hjk' hk'k e => Nat.find_min hex hk'k ⟨hk', hjk', by omega, e⟩
        have h1 := ih _ hki hk
        rw [hka, ram_take_of_last v0 log _ j hj ha _ hjk (by omega) hmin] at h1
        have : rd log[Nat.find hex] = rd log[i] := by
          simp only [rd, Prod.mk.injEq] at h1 ⊢
          exact ⟨hka, h1.1.trans hv, h1.2.trans ht⟩
        exact absurd (hrdinj _ i hk hi this) (by omega)
      rw [ram_take_of_last v0 log _ j hj ha i hj_lt (by omega) hnone, hv, ht]
    · simp only [rd, Prod.mk.injEq] at hai
      obtain ⟨ha, hv, ht⟩ := hai
      have hnone : ∀ k (hk : k < log.length), k < i → log[k].a ≠ log[i].a := by
        intro k0 hk0 h1 e0
        have hex : ∃ k, ∃ hk : k < log.length, k < i ∧ log[k].a = log[i].a := ⟨k0, hk0, h1, e0⟩
        obtain ⟨hk, hki, hka⟩ := Nat.find_spec hex
        have hmin : ∀ k' (hk' : k' < log.length), k' < Nat.find hex → log[k'].a ≠ log[i].a :=
          fun k' hk' hk'k e => Nat.find_min hex hk'k ⟨hk', by omega, e⟩
        have h1 := ih _ hki hk
        rw [hka, ram_take_of_none v0 log _ _ (by omega) hmin] at h1
        have : rd log[Nat.find hex] = rd log[i] := by
          simp only [rd, Prod.mk.injEq] at h1 ⊢
          exact ⟨hka, h1.1.trans (ha ▸ hv), h1.2.trans ht⟩
        exact absurd (hrdinj _ i hk hi this) (by omega)
      rw [ram_take_of_none v0 log _ i (by omega) hnone, ← ha, hv, ht]
  refine ⟨hreads, ?_, ?_⟩
  · -- the final contents: an access to the address after the record's source would read that record too
    intro a ha
    rcases hsrc _ (hfin_mem a ha) with ⟨j, hj, hja⟩ | ⟨a', -, ha'⟩
    · simp only [wr, Prod.mk.injEq] at hja
      obtain ⟨hja, hfin⟩ := hja
      have hnone : ∀ k (hk : k < log.length), j < k → k < log.length → log[k].a ≠ a := by
        intro k0 hk0 h1 _ e0
        have hex : ∃ k, ∃ hk : k < log.length, j < k ∧ log[k].a = a := ⟨k0, hk0, h1, e0⟩
        obtain ⟨hk, hjk, hka⟩ := Nat.find_spec hex
        have hmin : ∀ k' (hk' : k' < log.length), j < k' → k' < Nat.find hex → log[k'].a ≠ a :=
          fun k' hk' hjk' hk'k e => Nat.find_min hex hk'k ⟨hk', hjk', e⟩
        have h1 := hreads _ hk
        rw [hka, ram_take_of_last v0 log a j hj hja _ hjk (by omega) hmin] at h1
        refine hrdfin _ hk a ha ?_
        simp only [rd]
        rw [hka, ← hfin, h1]
      have := ram_take_of_last v0 log a j hj hja log.length hj le_rfl hnone
      rw [List.take_length] at this
      rw [this, hfin]
    · simp only [Prod.mk.injEq] at ha'
      obtain ⟨rfl, hfin⟩ := ha'
      have hnone : ∀ k (hk : k < log.length), k < log.length → log[k].a ≠ a' := by
        intro k0 hk0 _ e0
        have hex : ∃ k, ∃ hk : k < log.length, log[k].a = a' := ⟨k0, hk0, e0⟩
        obtain ⟨hk, hka⟩ := Nat.find_spec hex
        have hmin : ∀ k' (hk' : k' < log.length), k' < Nat.find hex → log[k'].a ≠ a' :=
          fun k' hk' hk'k e => Nat.find_min hex hk'k ⟨hk', e⟩
        have h1 := hreads _ hk
        rw [hka, ram_take_of_none v0 log a' _ (by omega) hmin] at h1
        refine hrdfin _ hk a' ha ?_
        simp only [rd]
        rw [hka, h1, hfin]
      have := ram_take_of_none v0 log a' log.length le_rfl hnone
      rw [List.take_length] at this
      rw [this, hfin]
  · -- every accessed address is touched: the first access to it reads a record no write before it made
    intro x hx
    obtain ⟨j0, hj0, rfl⟩ := List.getElem_of_mem hx
    have hex : ∃ k, ∃ hk : k < log.length, log[k].a = log[j0].a := ⟨j0, hj0, rfl⟩
    obtain ⟨hk, hka⟩ := Nat.find_spec hex
    rcases hsrc _ (hread_mem _ hk) with ⟨j, hj, hjk⟩ | ⟨a, ha, hak⟩
    · simp only [wr, rd, Prod.mk.injEq] at hjk
      have hjlt : j < Nat.find hex :=
        hord' j _ hj hk (by rw [hjk.2.2]; exact hlt _ (List.getElem_mem hk))
      exact absurd ⟨hj, hjk.1.trans hka⟩ (Nat.find_min hex hjlt)
    · simp only [rd, Prod.mk.injEq] at hak
      rw [← hka, ← hak.1]
      exact ha

/-- **`MemoryChecked`'s honest lemma.** The log of a memory that holds `0` everywhere, writing `5` and then `7` at
address `true` and `3` at `false`, each access reading what the memory held, passes the check with its final contents. -/
theorem MemoryChecked_honest :
    Checked (fun _ : Bool => (0 : ℕ)) [⟨true, 0, 0, 5, 1⟩, ⟨true, 5, 1, 7, 2⟩, ⟨false, 0, 0, 3, 3⟩] {true, false}
      (fun a => if a then (7, 2) else (3, 3)) := by
  unfold Checked reads writes
  decide

/-- **`MemoryChecked`'s nontrivial lemma: its conclusion fails at a check that accepts everything.** One access that reads
its own write (`9` at address `true`, at time `1`) did not read its memory correctly; it passes the multiset test, with the
initial content as the final one, and only the time test refuses it. -/
theorem MemoryChecked_nontrivial :
    ¬ ReadsCorrect (fun _ : Bool => (0 : ℕ)) [⟨true, 9, 1, 9, 1⟩] {true} fun _ => (0, 0) := by
  unfold ReadsCorrect ram
  decide

theorem FingerprintCollision : Guarantees.FingerprintCollision := by
  intro F _ _ m A B
  classical
  by_cases hAB : A = B
  · refine le_trans (prCoin_mono (q := fun _ => False) fun c hc => hc.1 hAB) ?_
    rw [prCoin_false]
    exact zero_le
  have hne : fingerprint2 A - fingerprint2 B ≠ 0 := sub_ne_zero.2 fun h => hAB (fingerprint2_injective h)
  calc prCoin (fun c : F × F => A ≠ B ∧ fingerprint c.1 c.2 A = fingerprint c.1 c.2 B)
      = prCoin (fun f : Fin 2 → F => MvPolynomial.eval f (fingerprint2 A - fingerprint2 B) = 0) := by
        rw [← prCoin_equiv (finTwoArrowEquiv F)]
        congr 1
        funext f
        simp [eval_fingerprint2, sub_eq_zero, hAB, finTwoArrowEquiv]
    _ ≤ ((fingerprint2 A - fingerprint2 B).totalDegree : ℝ≥0∞) / Fintype.card F :=
        prCoin_mvEval_eq_zero_le _ hne
    _ ≤ ((max A.card B.card * max 1 (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F := by
        gcongr
        refine (MvPolynomial.totalDegree_sub _ _).trans (max_le ?_ ?_)
        · exact (totalDegree_fingerprint2_le A).trans (Nat.mul_le_mul_right _ (le_max_left _ _))
        · exact (totalDegree_fingerprint2_le B).trans (Nat.mul_le_mul_right _ (le_max_right _ _))

theorem PrintCollision : Guarantees.PrintCollision := by
  intro F _ _ m S
  classical
  induction S using Finset.induction_on with
  | empty => simp [prCoin_false]
  | insert a S ha ih =>
    calc prCoin (fun δ : F => ∃ k ∈ insert a S, ∃ k' ∈ insert a S, k ≠ k' ∧ element δ k = element δ k')
        ≤ prCoin (fun δ : F => (∃ k ∈ S, ∃ k' ∈ S, k ≠ k' ∧ element δ k = element δ k') ∨
            ∃ k' ∈ S, element δ a = element δ k') := by
          refine prCoin_mono fun δ => ?_
          rintro ⟨k, hk, k', hk', hkk', he⟩
          rw [Finset.mem_insert] at hk hk'
          rcases hk with rfl | hk <;> rcases hk' with rfl | hk'
          · exact absurd rfl hkk'
          · exact Or.inr ⟨k', hk', he⟩
          · exact Or.inr ⟨k, hk, he.symm⟩
          · exact Or.inl ⟨k, hk, k', hk', hkk', he⟩
      _ ≤ ((S.card.choose 2 * (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F +
            ∑ k' ∈ S, ((m - 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
          refine (prCoin_or_le _ _).trans (add_le_add ih ((prCoin_exists_le _ _).trans ?_))
          exact Finset.sum_le_sum fun k' hk' => prCoin_element_eq_le _ _ fun h => ha (h ▸ hk')
      _ = (((insert a S).card.choose 2 * (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F := by
          rw [Finset.sum_const, nsmul_eq_mul, ← mul_div_assoc, ENNReal.div_add_div_same,
            Finset.card_insert_of_notMem ha, Nat.choose_succ_succ, Nat.choose_one_right]
          push_cast
          ring_nf

/-- **`FingerprintCollision`'s honest lemma.** At GF(2^128), with records of two chunks: the two sides of an honest log,
one multiset listed in two orders, have equal fingerprints at every coin, so the check accepts it with probability 1;
and against a different multiset of two records the collision probability is at most the bound at two records of two
chunks, `2 / 2^128`. -/
theorem FingerprintCollision_honest :
    prCoin (fun c : GaloisField 2 128 × GaloisField 2 128 =>
        fingerprint (m := 2) c.1 c.2 {1, 0} = fingerprint (m := 2) c.1 c.2 {0, 1}) = 1 ∧
      prCoin (fun c : GaloisField 2 128 × GaloisField 2 128 =>
        ({1, 0} : Multiset (Fin 2 → GaloisField 2 128)) ≠ {1, 1} ∧
          fingerprint (m := 2) c.1 c.2 {1, 0} = fingerprint (m := 2) c.1 c.2 {1, 1}) ≤ 2 / 2 ^ 128 := by
  refine ⟨ValueSalts.prCoin_of_forall fun c => by rw [Multiset.pair_comm], ?_⟩
  have := FingerprintCollision (F := GaloisField 2 128) 2 {1, 0} {1, 1}
  rw [card_gf128] at this
  exact this.trans_eq (by norm_num)

/-- **`FingerprintCollision`'s nontrivial lemma: its bound fails at a check that accepts everything.** At GF(2^128), a
check that accepts every pair of multisets accepts the different multisets `{1}` and `{0}` of one-chunk records with
probability 1, above the bound at one record of one chunk, `1 / 2^128`. -/
theorem FingerprintCollision_nontrivial :
    ¬ prCoin (fun _ : GaloisField 2 128 × GaloisField 2 128 =>
        ({1} : Multiset (Fin 1 → GaloisField 2 128)) ≠ {0} ∧ True) ≤ 1 / 2 ^ 128 := by
  rw [ValueSalts.prCoin_of_forall fun _ => ⟨by simp, trivial⟩, one_div, not_le]
  exact ENNReal.inv_lt_one.2 (by norm_num)

open Classical in
/-- **`PrintCollision`'s honest lemma.** At GF(2^128), the two keys `0` and `1` of four chunks share a print at a
uniform `δ` with probability at most the bound at two keys of four chunks, `3 / 2^128`. -/
theorem PrintCollision_honest :
    prCoin (fun δ : GaloisField 2 128 => ∃ k ∈ ({0, 1} : Finset (Fin 4 → GaloisField 2 128)),
        ∃ k' ∈ ({0, 1} : Finset (Fin 4 → GaloisField 2 128)), k ≠ k' ∧ element δ k = element δ k') ≤ 3 / 2 ^ 128 := by
  have := PrintCollision (F := GaloisField 2 128) 4 {0, 1}
  rw [card_gf128, Finset.card_pair zero_ne_one] at this
  exact this.trans_eq (by norm_num)

open Classical in
/-- **`PrintCollision`'s nontrivial lemma: its bound fails at a check that takes every two keys to share a print.** At
GF(2^128), such a check finds the two keys `0` and `1` of two chunks sharing a print with probability 1, above the
bound at two keys of two chunks, `1 / 2^128`. -/
theorem PrintCollision_nontrivial :
    ¬ prCoin (fun _ : GaloisField 2 128 => ∃ k ∈ ({0, 1} : Finset (Fin 2 → GaloisField 2 128)),
        ∃ k' ∈ ({0, 1} : Finset (Fin 2 → GaloisField 2 128)), k ≠ k' ∧ True) ≤ 1 / 2 ^ 128 := by
  rw [ValueSalts.prCoin_of_forall fun _ => ⟨0, by simp, 1, by simp, zero_ne_one, trivial⟩, one_div, not_le]
  exact ENNReal.inv_lt_one.2 (by norm_num)

end Verity.SecurityProofs
