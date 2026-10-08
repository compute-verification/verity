import Proofs.Flock.Soundness.Assumptions

/-!
# Zero knowledge: T1, an hm96 leaf swapped for an ideal one

`docs/zk-proof-public.md` §4.1's T1: take any experiment in which a leaf's salt `y` is fresh, used for nothing else and
never revealed, while its digest `x` may depend on everything else the experiment samples. Replacing the leaf by an ideal
one changes the probability of any event by at most `δ₁`.

An hm96 leaf is a function of the pair `(b, c) = (x ⊕ M·y, H(sp ‖ y))`, and an ideal one of `(U, H(sp ‖ y))` with `U` fresh
and uniform. `Assumptions.Hm96Hiding` bounds the distance of `(M·y, c)` from `(U, c)` by `δ₁`. XOR by `x` is a bijection
that maps uniform to uniform, so for each value of the other randomness the two leaves are `δ₁` apart, and averaging keeps
the bound. Here XOR is the addition of the group `B`, so the proof holds for any group of digests.

`ideal_leaves_swap` is T1 applied once per leaf, the hybrid of Lemma B's proof: `n` leaves cost `n·δ₁`. `Table.view` has
every leaf the final message doesn't open ideal (world `W₁`); with `|Hid|` such leaves on each side, the hybrid is Lemma B's
bound `2·|Hid|·δ₁`, at most the paper's `2·δ₁·N_hid` (`RealLeaves.table_shvzk_hm96`).
-/

namespace FlockSoundness.ZK

open Game
open scoped ENNReal

theorem avg_prod_split {C D : Type} [Fintype C] [Fintype D] (f : C × D → ℝ≥0∞) :
    avg f = avg fun c => avg fun d => f (c, d) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun d _ => ?_
  ring

theorem prCoin_prod_split {C D : Type} [Fintype C] [Fintype D] (P : C → D → Prop) :
    prCoin (fun p : C × D => P p.1 p.2) = avg fun c => prCoin fun d => P c d := by
  unfold prCoin
  exact avg_prod_split _

theorem prCoin_equiv {C : Type} [Fintype C] (e : C ≃ C) (P : C → Prop) : prCoin (fun c => P (e c)) = prCoin P := by
  unfold prCoin avg
  exact Fintype.sum_equiv e _ _ (fun _ => rfl)

/-- **T1** (§4.1): in an experiment on uniform `(ω, y)` whose leaf is `H(x(ω) + M·y, c(y))`, replacing the leaf by the ideal
`H(U, c(y))` with a fresh uniform `U` changes the probability of any event `G` by at most `δ₁`. The digest `x` may depend on
anything in `ω`; the salt `y` is used only by this leaf. -/
theorem ideal_leaf_swap {Ω Y B C D : Type} [Fintype Ω] [Nonempty Ω] [Fintype Y] [AddCommGroup B] [Fintype B]
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My c δ₁) (H : B × C → D) (x : Ω → B)
    (G : Ω → D → Prop) :
    prCoin (fun p : Ω × Y => G p.1 (H (x p.1 + My p.2, c p.2)))
      ≤ prCoin (fun p : Ω × (B × Y) => G p.1 (H (p.2.1, c p.2.2))) + δ₁ := by
  rw [prCoin_prod_split (fun ω y => G ω (H (x ω + My y, c y))),
    prCoin_prod_split (fun ω (u : B × Y) => G ω (H (u.1, c u.2)))]
  calc avg (fun ω => prCoin fun y => G ω (H (x ω + My y, c y)))
      ≤ avg (fun ω => prCoin (fun u : B × Y => G ω (H (u.1, c u.2))) + δ₁) := by
        refine avg_mono fun ω => ?_
        have h := hT1 fun bc => G ω (H (x ω + bc.1, bc.2))
        have e : prCoin (fun u : B × Y => G ω (H (x ω + u.1, c u.2))) = prCoin fun u : B × Y => G ω (H (u.1, c u.2)) :=
          prCoin_equiv ((Equiv.addLeft (x ω)).prodCongr (Equiv.refl Y)) (fun u : B × Y => G ω (H (u.1, c u.2)))
        exact h.trans (le_of_eq (by rw [e]))
    _ = avg (fun ω => prCoin fun u : B × Y => G ω (H (u.1, c u.2))) + δ₁ := by
        rw [avg_add, avg_const]

theorem prCoin_congr_equiv {C D : Type} [Fintype C] [Fintype D] (e : C ≃ D) (P : D → Prop) :
    prCoin (fun c => P (e c)) = prCoin P := by
  unfold prCoin avg
  rw [Fintype.card_congr e]
  exact Fintype.sum_equiv e _ _ (fun _ => rfl)

/-- **T1 over `n` leaves** (the hybrid of Lemma B's proof): in an experiment on uniform `(ω, y₀ … y_{n−1})` whose `n` leaves
are `H(x_i(ω) + M·y_i, c(y_i))`, replacing every one of them by an ideal leaf `H(U_i, c(y_i))` with fresh uniform `U_i`
changes the probability of any event by at most `n·δ₁`. Each digest may depend on anything in `ω`; each salt is used only
by its leaf. One leaf at a time: the first by `ideal_leaf_swap`, with the other salts part of the experiment, then the
rest by induction, with the first leaf's ideal seed part of it. -/
theorem ideal_leaves_swap {Y B C D : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My c δ₁) (H : B × C → D) :
    ∀ (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → Fin n → B) (G : Ω → (Fin n → D) → Prop),
      prCoin (fun p : Ω × (Fin n → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
        ≤ prCoin (fun p : Ω × (Fin n → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + n * δ₁ := by
  intro n
  induction n with
  | zero =>
    intro Ω _ _ x G
    let e : Ω × (Fin 0 → Y) ≃ Ω × (Fin 0 → B × Y) := (Equiv.refl Ω).prodCongr
      ⟨fun _ => Fin.elim0, fun _ => Fin.elim0, fun _ => funext fun i => i.elim0, fun _ => funext fun i => i.elim0⟩
    have h := prCoin_congr_equiv e (fun p : Ω × (Fin 0 → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
    have e1 : (fun p : Ω × (Fin 0 → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
        = fun p => G (e p).1 fun i => H (((e p).2 i).1, c ((e p).2 i).2) := by
      funext p
      congr 1
      funext i
      exact i.elim0
    rw [e1, h]
    simp
  | succ n ih =>
    intro Ω _ _ x G
    let e₁ : Ω × (Fin (n + 1) → Y) ≃ (Ω × (Fin n → Y)) × Y :=
      { toFun := fun p => ((p.1, Fin.tail p.2), p.2 0)
        invFun := fun q => (q.1.1, Fin.cons q.2 q.1.2)
        left_inv := fun p => by simp [Fin.cons_self_tail]
        right_inv := fun q => by simp }
    let eB : (Ω × (Fin n → Y)) × (B × Y) ≃ (Ω × (B × Y)) × (Fin n → Y) :=
      { toFun := fun p => ((p.1.1, p.2), p.1.2)
        invFun := fun p => ((p.1.1, p.2), p.1.2)
        left_inv := fun _ => rfl
        right_inv := fun _ => rfl }
    let eC : (Ω × (B × Y)) × (Fin n → B × Y) ≃ Ω × (Fin (n + 1) → B × Y) :=
      { toFun := fun p => (p.1.1, Fin.cons p.1.2 p.2)
        invFun := fun q => ((q.1, q.2 0), Fin.tail q.2)
        left_inv := fun p => by simp
        right_inv := fun q => by simp [Fin.cons_self_tail] }
    -- the first leaf, with the other `n` salts part of the experiment
    have step₁ := ideal_leaf_swap My c hT1 H (fun q : Ω × (Fin n → Y) => x q.1 0)
      (fun q d => G q.1 (Fin.cons d fun i => H (x q.1 i.succ + My (q.2 i), c (q.2 i))))
    -- the other `n`, with the first leaf's ideal seed part of the experiment
    have step₂ := ih (Ω := Ω × (B × Y)) (fun q i => x q.1 i.succ)
      (fun q ds => G q.1 (Fin.cons (H (q.2.1, c q.2.2)) ds))
    have hA : prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.cons (H (x q.1.1 0 + My q.2, c q.2)) fun i => H (x q.1.1 i.succ + My (q.1.2 i), c (q.1.2 i)))) := by
      rw [← prCoin_congr_equiv e₁]
      congr 1
      funext p
      congr 1
      funext i
      refine Fin.cases ?_ (fun i => ?_) i <;> rfl
    have hB : prCoin (fun p : (Ω × (Fin n → Y)) × (B × Y) =>
          G p.1.1 (Fin.cons (H (p.2.1, c p.2.2)) fun i => H (x p.1.1 i.succ + My (p.1.2 i), c (p.1.2 i))))
        = prCoin (fun p : (Ω × (B × Y)) × (Fin n → Y) =>
          G p.1.1 (Fin.cons (H (p.1.2.1, c p.1.2.2)) fun i => H (x p.1.1 i.succ + My (p.2 i), c (p.2 i)))) :=
      prCoin_congr_equiv eB (fun p : (Ω × (B × Y)) × (Fin n → Y) =>
        G p.1.1 (Fin.cons (H (p.1.2.1, c p.1.2.2)) fun i => H (x p.1.1 i.succ + My (p.2 i), c (p.2 i))))
    have hC : prCoin (fun p : (Ω × (B × Y)) × (Fin n → B × Y) =>
          G p.1.1 (Fin.cons (H (p.1.2.1, c p.1.2.2)) fun i => H ((p.2 i).1, c (p.2 i).2)))
        = prCoin (fun p : Ω × (Fin (n + 1) → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) := by
      rw [← prCoin_congr_equiv eC]
      congr 1
      funext p
      congr 1
      funext i
      refine Fin.cases ?_ (fun i => ?_) i <;> rfl
    calc prCoin (fun p : Ω × (Fin (n + 1) → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
        = prCoin (fun q : (Ω × (Fin n → Y)) × Y =>
            G q.1.1 (Fin.cons (H (x q.1.1 0 + My q.2, c q.2)) fun i => H (x q.1.1 i.succ + My (q.1.2 i), c (q.1.2 i)))) :=
          hA
      _ ≤ prCoin (fun p : (Ω × (Fin n → Y)) × (B × Y) =>
            G p.1.1 (Fin.cons (H (p.2.1, c p.2.2)) fun i => H (x p.1.1 i.succ + My (p.1.2 i), c (p.1.2 i)))) + δ₁ :=
          step₁
      _ = prCoin (fun p : (Ω × (B × Y)) × (Fin n → Y) =>
            G p.1.1 (Fin.cons (H (p.1.2.1, c p.1.2.2)) fun i => H (x p.1.1 i.succ + My (p.2 i), c (p.2 i)))) + δ₁ := by
          rw [hB]
      _ ≤ (prCoin (fun p : (Ω × (B × Y)) × (Fin n → B × Y) =>
            G p.1.1 (Fin.cons (H (p.1.2.1, c p.1.2.2)) fun i => H ((p.2 i).1, c (p.2 i).2))) + n * δ₁) + δ₁ :=
          add_le_add step₂ le_rfl
      _ = prCoin (fun p : Ω × (Fin (n + 1) → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
            + ((n + 1 : ℕ) : ℝ≥0∞) * δ₁ := by
          rw [hC]
          push_cast
          ring

end FlockSoundness.ZK
