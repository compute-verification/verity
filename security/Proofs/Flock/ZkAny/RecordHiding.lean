module

public import Definitions.Core.ValueSalts
public import Proofs.Core.Game.Basic
public import Proofs.Flock.Soundness.ZK.Hiding
public import Proofs.Flock.Soundness.CROnly.Hiding

/-!
# A registration's ideal record hides its values, at hm96 commitments

`Verity.ValueSalts.Service.ideal` is the probability of an event on one record of a registration whose session, salts,
tags and record tag are uniform. Where the value service's commitments are a function of `N` hm96 leaves, leaf `i` of
the values `v` being `H(dig v i + M·y_i, c(y_i))` under its own uniform salt `y_i`, an ideal record of `v` is within
`N·2δ₁` of one of any other values `v₀` on every event, `δ₁` hm96's hiding gap (`CROnly.hidingGap`), with no premise:
each leaf is within `δ₁` of an ideal one `H(U, c(y))` both ways, and the `N` leaves are swapped one at a time. This is
the `δ` a sampled audit's view at any values (`Flock.Guarantees.SampledAuditAny`) takes, at hm96's leaves.

* `hm96_rev_gap`: `CROnly.hm96Hiding_gap`'s converse, `(U, c y)` within the gap of `(M·y, c y)`: the gap is a statistical
  distance.
* `ideal_leaf_swap_rev`: T1's converse (`ideal_leaf_swap`), an ideal leaf replaced by an hm96 leaf of any digest.
* `leaves_hybrid`: a bound on replacing one leaf of its own salt, in every experiment, is `n` times it over `n` leaves.
* `leaves_swap_values`: `n` hm96 leaves of one digest each replaced by those of another, at `n·2δ₁`.
* `ideal_hm96`: the ideal record's hiding at hm96 commitments.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game Finset
open scoped ENNReal

/-- **hm96 hiding's converse at its gap**: `(U, c y)` for `(U, y)` uniform on `B × Y` is within the hiding gap of
`(M·y, c y)` for `y` uniform on `Y`, on every event. -/
theorem hm96_rev_gap {Y B C : Type} [Fintype Y] [Fintype B] [DecidableEq B] [DecidableEq C] [Nonempty Y] [Nonempty B]
    (My : Y → B) (c : Y → C) (E : B × C → Prop) :
    prCoin (fun p : B × Y => E (p.1, c p.2)) ≤
      prCoin (fun y : Y => E (My y, c y)) + ENNReal.ofReal (CROnly.hidingGap My c) := by
  refine (CROnly.prCoin_comp_le (fun p : B × Y => (p.1, c p.2)) (fun y => (My y, c y))
    ((univ : Finset B) ×ˢ univ.image c) (fun p => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ p.2)⟩)
    (fun y => mem_product.2 ⟨mem_univ _, mem_image_of_mem c (mem_univ y)⟩) E).trans
    (add_le_add le_rfl (ENNReal.ofReal_le_ofReal (le_of_eq ?_)))
  unfold CROnly.hidingGap
  congr 1
  exact Finset.sum_congr rfl fun v _ => abs_sub_comm _ _

/-- **T1's converse**: in an experiment on uniform `(ω, (U, y))` whose leaf is the ideal `H(U, c(y))`, replacing it by the
hm96 leaf `H(x(ω) + M·y, c(y))` changes the probability of any event `G` by at most the hiding gap. The digest `x` may
depend on anything in `ω`; the salt `y` is used only by this leaf. -/
theorem ideal_leaf_swap_rev {Ω Y B C D : Type} [Fintype Ω] [Nonempty Ω] [Fintype Y] [Nonempty Y] [AddCommGroup B]
    [Fintype B] [Nonempty B] [DecidableEq B] [DecidableEq C] (My : Y → B) (c : Y → C) (H : B × C → D) (x : Ω → B)
    (G : Ω → D → Prop) :
    prCoin (fun p : Ω × (B × Y) => G p.1 (H (p.2.1, c p.2.2)))
      ≤ prCoin (fun p : Ω × Y => G p.1 (H (x p.1 + My p.2, c p.2))) + ENNReal.ofReal (CROnly.hidingGap My c) := by
  rw [prCoin_prod_split (fun ω (u : B × Y) => G ω (H (u.1, c u.2))),
    prCoin_prod_split (fun ω y => G ω (H (x ω + My y, c y)))]
  calc avg (fun ω => prCoin fun u : B × Y => G ω (H (u.1, c u.2)))
      ≤ avg (fun ω => prCoin (fun y => G ω (H (x ω + My y, c y))) + ENNReal.ofReal (CROnly.hidingGap My c)) := by
        refine avg_mono fun ω => ?_
        have e : prCoin (fun u : B × Y => G ω (H (x ω + u.1, c u.2))) = prCoin fun u : B × Y => G ω (H (u.1, c u.2)) :=
          prCoin_equiv ((Equiv.addLeft (x ω)).prodCongr (Equiv.refl Y)) (fun u : B × Y => G ω (H (u.1, c u.2)))
        exact (le_of_eq e.symm).trans (hm96_rev_gap My c fun bc => G ω (H (x ω + bc.1, bc.2)))
    _ = avg (fun ω => prCoin fun y => G ω (H (x ω + My y, c y))) + ENNReal.ofReal (CROnly.hidingGap My c) := by
        rw [avg_add, avg_const]

/-- **The hybrid over `n` leaves**: where replacing a leaf `L₁ x y` by `L₂ x y`, both of a digest `x` and the leaf's own
uniform salt `y`, changes the probability of any event by at most `δ` in every experiment whose digest may depend on
anything else it samples, replacing all `n` leaves of an experiment, each of its own salt, changes it by at most `n·δ`.
One leaf at a time: the first, with the other salts part of the experiment, then the rest by induction, with the first
leaf's salt part of it. -/
theorem leaves_hybrid {Y X D : Type} [Fintype Y] [Nonempty Y] (L₁ L₂ : X → Y → D) {δ : ℝ≥0∞}
    (h1 : ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → X) (G : Ω → D → Prop),
      prCoin (fun p : Ω × Y => G p.1 (L₁ (x p.1) p.2)) ≤ prCoin (fun p : Ω × Y => G p.1 (L₂ (x p.1) p.2)) + δ) :
    ∀ (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → Fin n → X) (G : Ω → (Fin n → D) → Prop),
      prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        ≤ prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => L₂ (x p.1 i) (p.2 i)) + n * δ := by
  intro n
  induction n with
  | zero =>
    intro Ω _ _ x G
    have e : (fun p : Ω × (Fin 0 → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        = fun p => G p.1 fun i => L₂ (x p.1 i) (p.2 i) := by
      funext p
      congr 1
      funext i
      exact i.elim0
    rw [e]
    simp
  | succ n ih =>
    intro Ω _ _ x G
    let e₁ : Ω × (Fin (n + 1) → Y) ≃ (Ω × (Fin n → Y)) × Y :=
      { toFun := fun p => ((p.1, Fin.tail p.2), p.2 0)
        invFun := fun q => (q.1.1, Fin.cons q.2 q.1.2)
        left_inv := fun p => by simp [Fin.cons_self_tail]
        right_inv := fun q => by simp }
    let eB : (Ω × (Fin n → Y)) × Y ≃ (Ω × Y) × (Fin n → Y) :=
      { toFun := fun p => ((p.1.1, p.2), p.1.2)
        invFun := fun p => ((p.1.1, p.2), p.1.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    let eC : (Ω × Y) × (Fin n → Y) ≃ Ω × (Fin (n + 1) → Y) :=
      { toFun := fun p => (p.1.1, Fin.cons p.1.2 p.2)
        invFun := fun q => ((q.1, q.2 0), Fin.tail q.2)
        left_inv := fun p => by simp
        right_inv := fun q => by simp [Fin.cons_self_tail] }
    -- the first leaf, with the other `n` salts part of the experiment
    have step₁ := h1 (fun q : Ω × (Fin n → Y) => x q.1 0)
      (fun q d => G q.1 (Fin.cons d fun i => L₁ (x q.1 i.succ) (q.2 i)))
    -- the other `n`, with the first leaf's salt part of the experiment
    have step₂ := ih (Ω := Ω × Y) (fun q i => x q.1 i.succ)
      (fun q ds => G q.1 (Fin.cons (L₂ (x q.1 0) q.2) ds))
    have hA : prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.cons (L₁ (x q.1.1 0) q.2) fun i => L₁ (x q.1.1 i.succ) (q.1.2 i))) := by
      rw [← prCoin_congr_equiv e₁]
      congr 1
      funext p
      congr 1
      funext i
      refine Fin.cases ?_ (fun i => ?_) i <;> rfl
    have hB : prCoin (fun p : (Ω × (Fin n → Y)) × Y =>
          G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.2) fun i => L₁ (x p.1.1 i.succ) (p.1.2 i)))
        = prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
          G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₁ (x p.1.1 i.succ) (p.2 i))) :=
      prCoin_congr_equiv eB (fun p : (Ω × Y) × (Fin n → Y) =>
        G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₁ (x p.1.1 i.succ) (p.2 i)))
    have hC : prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
          G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₂ (x p.1.1 i.succ) (p.2 i)))
        = prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₂ (x p.1 i) (p.2 i)) := by
      rw [← prCoin_congr_equiv eC]
      congr 1
      funext p
      congr 1
      funext i
      refine Fin.cases ?_ (fun i => ?_) i <;> rfl
    calc prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₁ (x p.1 i) (p.2 i))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.cons (L₁ (x q.1.1 0) q.2) fun i => L₁ (x q.1.1 i.succ) (q.1.2 i))) := hA
      _ ≤ prCoin (fun p : (Ω × (Fin n → Y)) × Y =>
            G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.2) fun i => L₁ (x p.1.1 i.succ) (p.1.2 i))) + δ := step₁
      _ = prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
            G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₁ (x p.1.1 i.succ) (p.2 i))) + δ := by
          rw [hB]
      _ ≤ (prCoin (fun p : (Ω × Y) × (Fin n → Y) =>
            G p.1.1 (Fin.cons (L₂ (x p.1.1 0) p.1.2) fun i => L₂ (x p.1.1 i.succ) (p.2 i))) + n * δ) + δ :=
          add_le_add step₂ le_rfl
      _ = prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => L₂ (x p.1 i) (p.2 i))
            + ((n + 1 : ℕ) : ℝ≥0∞) * δ := by
          rw [hC]
          push_cast
          ring

/-- **`n` hm96 leaves of one digest each replaced by those of another**: in an experiment on uniform
`(ω, y₀ … y_{n−1})` whose leaves are `H(x_i(ω) + M·y_i, c(y_i))`, replacing them by `H(x'_i(ω) + M·y_i, c(y_i))`
changes the probability of any event by at most `n·2δ₁`, `δ₁` the hiding gap: each leaf through an ideal one
(`ideal_leaf_swap` at `CROnly.hm96Hiding_gap`, then `ideal_leaf_swap_rev`), one at a time (`leaves_hybrid`). -/
theorem leaves_swap_values {Y B C D : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] [Nonempty B]
    [DecidableEq B] [DecidableEq C] (My : Y → B) (c : Y → C) (H : B × C → D) (n : ℕ) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (x x' : Ω → Fin n → B) (G : Ω → (Fin n → D) → Prop) :
    prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
      ≤ prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => H (x' p.1 i + My (p.2 i), c (p.2 i))) +
        n * (2 * ENNReal.ofReal (CROnly.hidingGap My c)) :=
  leaves_hybrid (δ := 2 * ENNReal.ofReal (CROnly.hidingGap My c)) (fun (a : B × B) y => H (a.1 + My y, c y))
    (fun (a : B × B) y => H (a.2 + My y, c y))
    (fun z G' => ((ideal_leaf_swap My c (CROnly.hm96Hiding_gap My c) H (fun ω => (z ω).1) G').trans
      (add_le_add (ideal_leaf_swap_rev My c H (fun ω => (z ω).2) G') le_rfl)).trans_eq
        (by rw [add_assoc, ← two_mul]))
    n (fun ω i => (x ω i, x' ω i)) G

open Verity.ValueSalts in
/-- **A registration's ideal record hides its values, at hm96 commitments.** Where the value service's commitments of
the values `v` under the salts `y` are a function `g` of `N` hm96 leaves, leaf `i` being `H(dig v i + M·y_i, c(y_i))`
under its own uniform salt `y_i`, an ideal record of `v` is within `N·2δ₁` of one of `v₀` on every event, `δ₁` the
hiding gap. -/
theorem ideal_hm96 {K S Ys B C D V O T R : Type} [Fintype S] [Nonempty S] [Fintype Ys] [Nonempty Ys] [Fintype T]
    [Nonempty T] [Fintype R] [Nonempty R] [AddCommGroup B] [Fintype B] [Nonempty B] [DecidableEq B] [DecidableEq C]
    {N : ℕ} (sv : Service K S (Fin N → Ys) V O T R) (My : Ys → B) (c : Ys → C) (H : B × C → D)
    (dig : V → Fin N → B) (g : (Fin N → D) → O)
    (hout : ∀ v y, sv.out v y = g fun i => H (dig v i + My (y i), c (y i))) (n m : ℕ) (v v₀ : V)
    (A : RegistrationRecord S O T R → Prop) :
    sv.ideal n m v A ≤ sv.ideal n m v₀ A + N * (2 * ENNReal.ofReal (CROnly.hidingGap My c)) := by
  let e : (S × T × R) × (Fin N → Ys) ≃ S × (Fin N → Ys) × T × R :=
    { toFun := fun q => (q.1.1, q.2, q.1.2.1, q.1.2.2)
      invFun := fun p => ((p.1, p.2.2.1, p.2.2.2), p.2.1)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hv : ∀ u : V, sv.ideal n m u A = prCoin (fun q : (S × T × R) × (Fin N → Ys) =>
      A (n, m, (q.1.1, g fun i => H (dig u i + My (q.2 i), c (q.2 i)), q.1.2.1), q.1.2.2)) := fun u => by
    unfold Service.ideal
    simp only [hout]
    exact (prCoin_congr_equiv e _).symm
  rw [hv v, hv v₀]
  exact leaves_swap_values My c H N (fun (_ : S × T × R) i => dig v i) (fun _ i => dig v₀ i)
    (fun ω ls => A (n, m, (ω.1, g ls, ω.2.1), ω.2.2))

end FlockSoundness.ZK
