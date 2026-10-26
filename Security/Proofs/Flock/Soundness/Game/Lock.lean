import Definitions.Flock.Game.Lock
import Proofs.Flock.Soundness.Game.Expect
import Proofs.Flock.Soundness.Game.Interleave

/-!
# Lockstep coupling of an oracle game with its compiled game

The compiled protocol replaces each committed table by a cap and opens the queried columns with
paths at the end. `Lock get R i g g'` exhibits the compiled game `g'` running in lockstep with the
oracle game `g`: the same coins and the same messages, except
at a **table** node, where `g` receives a table `t : T` and `g'` a cap `d : D`. The node carries a
slot, and the index `i : I` records the tables fixed so far (`get i slot`). At the leaves the two
outputs satisfy `R i a b`. The compiled game's last message, the openings, has no counterpart in
the oracle game; the oracle game gets a dummy send for it (`withDummy`), which changes no
probability (`expect_withDummy`).

For a compiled strategy `σ'` and a choice of table at every table node (`plur`, the plurality of
the slot's openings over the continuation), `oracleStrat` is the oracle strategy `P̃` that sends
the chosen tables and otherwise plays `σ'`. `expect_le_coupled`: the compiled event has
probability at most the oracle event's under `P̃`, plus the probability of `Bad`, the runs whose
leaf fails the coupling. `leafE_off_le`: per slot, the probability that a run opens a position
off the table chosen at that slot is at most `√(N · Q · Pr[fork conflict])`: the rewinding lemma
(`Rewinding.expect_bad_le`) at the slot's node, averaged over prefixes by Jensen.
-/

namespace FlockSoundness

namespace Game

universe u
variable {α β : Type}

namespace Lock

variable {T D I : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}

/-- A leaf function that ignores the index has the plain expectation. -/
theorem leafE_const (plur : Chooser T D β) (f : β → ℝ) : ∀ {i : I} {g : Game α} {g' : Game β}
    (L : Lock T D get R i g g') (σ' : Strategy g'),
    leafE plur (fun _ b => f b) L σ' = expect f g' σ'
  | _, _, _, .ret _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [leafE, expect]
      congr 1
      exact Finset.sum_congr rfl fun c _ => leafE_const plur f (L c) (σ' c)
  | _, _, _, .send _ L, σ' => leafE_const plur f (L σ'.1) σ'.2
  | _, _, _, .table _ _ _ _ _ L, σ' => leafE_const plur f _ σ'.2

theorem leafE_mono (plur : Chooser T D β) {F F' : I → β → ℝ} (h : ∀ i b, F i b ≤ F' i b) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      leafE plur F L σ' ≤ leafE plur F' L σ'
  | _, _, _, .ret _, _ => h _ _
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [leafE]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => leafE_mono plur h (L c) (σ' c))
        (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ' => leafE_mono plur h (L σ'.1) σ'.2
  | _, _, _, .table _ _ _ _ _ L, σ' => leafE_mono plur h _ σ'.2

/-- Monotonicity where the leaf relation holds. -/
theorem leafE_mono_rel (plur : Chooser T D β) {F F' : I → β → ℝ}
    (h : ∀ i a b, R i a b → F i b ≤ F' i b) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      leafE plur F L σ' ≤ leafE plur F' L σ'
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i a b hr, _ => h i a b hr
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [leafE]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun c _ => leafE_mono_rel plur h (L c) (σ' c)) (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ' => leafE_mono_rel plur h (L σ'.1) σ'.2
  | _, _, _, .table _ _ _ _ _ L, σ' => leafE_mono_rel plur h _ σ'.2

theorem leafE_add (plur : Chooser T D β) (F F' : I → β → ℝ) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      leafE plur (fun i b => F i b + F' i b) L σ' = leafE plur F L σ' + leafE plur F' L σ'
  | _, _, _, .ret _, _ => rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [leafE, leafE_add plur F F' (L _) (σ' _), Finset.sum_add_distrib, add_div]
  | _, _, _, .send _ L, σ' => leafE_add plur F F' (L σ'.1) σ'.2
  | _, _, _, .table _ _ _ _ _ L, σ' => leafE_add plur F F' _ σ'.2

theorem leafE_sum (plur : Chooser T D β) {ι : Type} (s : Finset ι) (F : ι → I → β → ℝ)
    {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g') :
    leafE plur (fun i b => ∑ x ∈ s, F x i b) L σ' = ∑ x ∈ s, leafE plur (F x) L σ' := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp only [Finset.sum_empty]
      rw [leafE_const plur (fun _ => 0) L σ', expect_const]
  | insert x s hx ih =>
      simp only [Finset.sum_insert hx]
      rw [leafE_add plur (F x) (fun i b => ∑ x ∈ s, F x i b), ih]

/-- **The coupling.** If every leaf where the compiled event holds and the run is not `Bad`
satisfies the oracle event, the compiled event's probability is at most the oracle event's under
`P̃` plus the probability of `Bad`. -/
theorem expect_le_coupled (plur : Chooser T D β) (Ec : β → Prop) (Eo : α → Prop)
    (Bad : I → β → Prop) (hR : ∀ i a b, R i a b → Ec b → ¬ Bad i b → Eo a) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      expect (ind Ec) g' σ' ≤
        expect (ind Eo) g (oracleStrat plur L σ') + leafE plur (fun i b => ind (Bad i) b) L σ'
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i a b h, _ => by
      simp only [expect, leafE, oracleStrat]
      unfold ind
      by_cases hE : Ec b
      · by_cases hB : Bad i b
        · rw [if_pos hE, if_pos hB]; split_ifs <;> norm_num
        · rw [if_pos hE, if_pos (hR i a b h hE hB), if_neg hB]; norm_num
      · rw [if_neg hE]; split_ifs <;> norm_num
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [expect, leafE, oracleStrat]
      rw [← add_div]
      refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_le_sum fun c _ => expect_le_coupled plur Ec Eo Bad hR (L c) (σ' c)
  | _, _, _, .send _ L, σ' => expect_le_coupled plur Ec Eo Bad hR (L σ'.1) σ'.2
  | _, _, _, .table _ _ _ _ _ L, σ' => expect_le_coupled plur Ec Eo Bad hR _ σ'.2

/-! ## Building locksteps -/

/-- A dummy last message changes no expectation. -/
theorem expect_withDummy (O : Type) (f : α → ℝ) : ∀ (g : Game α) (σ : Strategy (withDummy O g)),
    expect f (withDummy O g) σ = expect f g (dropDummy O g σ)
  | .ret _, _ => rfl
  | @Game.coin _ C _ _ k, σ => by
      simp only [withDummy, bind_coin, expect]
      exact congrArg (· / _) (Finset.sum_congr rfl fun c _ => expect_withDummy O f (k c) (σ c))
  | .send _ k, σ => expect_withDummy O f (k σ.1) σ.2

/-! ## Two locksteps interleaved -/

/-! ## Rewinding at every table node -/

theorem forkE_nonneg (plur : Chooser T D β) (slot : ℕ) {Φ : D → β → β → ℝ}
    (hΦ : ∀ d b b', 0 ≤ Φ d b b') :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      0 ≤ forkE plur slot Φ L σ'
  | _, _, _, .ret _, _ => le_rfl
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [forkE]
      exact div_nonneg (Finset.sum_nonneg fun c _ => forkE_nonneg plur slot hΦ (L c) (σ' c))
        (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ' => forkE_nonneg plur slot hΦ (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' => by
      simp only [forkE]
      split_ifs
      · exact expect_nonneg (fun b => expect_nonneg (hΦ _ b) _ _) _ _
      · exact forkE_nonneg plur slot hΦ _ σ'.2

/-- Below a slot's node, a leaf function reading only that slot's table is the plain expectation
with the table fixed. -/
theorem leafE_fixed (plur : Chooser T D β) (slot : ℕ) (td : T × D) (f : β → ℝ) (F : I → β → ℝ)
    (hF : ∀ i b, get i slot = some td → F i b = f b) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      get i slot = some td → leafE plur F L σ' = expect f g' σ'
  | _, _, _, .ret _, _, hi => hF _ _ hi
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [leafE, expect]
      congr 1
      exact Finset.sum_congr rfl fun c _ => leafE_fixed plur slot td f F hF (L c) (σ' c) hi
  | _, _, _, .send _ L, σ', hi => leafE_fixed plur slot td f F hF (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s upd _ k' hfresh _ hkeep L, σ', hi => by
      simp only [leafE, expect]
      have hs : s ≠ slot := fun h => by subst h; rw [hfresh] at hi; cases hi
      exact leafE_fixed plur slot td f F hF _ σ'.2 (by rw [hkeep _ _ slot hs.symm, hi])

/-- Jensen for `√` over a uniform average. -/
theorem avg_sqrt_le' {C : Type} [Fintype C] [Nonempty C] (x : C → ℝ) (hx : ∀ c, 0 ≤ x c) :
    (∑ c, Real.sqrt (x c)) / Fintype.card C ≤ Real.sqrt ((∑ c, x c) / Fintype.card C) := by
  have hn : (0 : ℝ) < Fintype.card C := by exact_mod_cast Fintype.card_pos
  have hcs : (∑ c, Real.sqrt (x c)) ^ 2 ≤ Fintype.card C * ∑ c, x c := by
    have := sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset C)) (f := fun c => Real.sqrt (x c))
    simpa [Real.sq_sqrt (hx _)] using this
  apply Real.le_sqrt_of_sq_le
  rw [div_pow, sq (Fintype.card C : ℝ), div_le_div_iff₀ (by positivity) hn]
  nlinarith [hcs]

open Classical in
/-- **Rewinding at a slot.** If the table chosen at `slot` (with cap `d`) always reads, through
`row`, as a plurality of the slot's openings `op d` over the continuation, then the probability
that a run opens a position off the table fixed at `slot` is at most `√(N · Q · Pr[fork conflict])`,
where the fork is at the slot's node. -/
theorem leafE_off_le {P V : Type} [Fintype P] [Fintype V] [DecidableEq V]
    (op : D → β → P → Option V) (row : T → P → V) (slot : ℕ) (Q : ℕ)
    (hQ : ∀ d b, (Finset.univ.filter fun p => (op d b p).isSome).card ≤ Q) (plur : Chooser T D β)
    (hplur : ∀ (d : D) (g'' : Game β) (σ'' : Strategy g''),
      Rewinding.IsPluralityE (op d) g'' σ'' (row (plur slot d g'' σ''))) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      get i slot = none →
        leafE plur (fun i b => ind (fun b => ∃ t d, get i slot = some (t, d) ∧
            ∃ p, Rewinding.Off (op d) (row t) b p) b) L σ' ≤
          Real.sqrt (Fintype.card P * Q *
            forkE plur slot (fun d b b' =>
              ind (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b')) L σ')
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i _ b _, _, hi => by
      simp only [leafE, forkE, hi]
      simp [ind]
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [leafE, forkE]
      have hΦ : ∀ d b b', 0 ≤ ind (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b') :=
        fun d b b' => ind_nonneg _ _
      calc (∑ c : C, leafE plur _ (L c) (σ' c)) / (Fintype.card C : ℝ)
          ≤ (∑ c : C, Real.sqrt (Fintype.card P * Q *
              forkE plur slot _ (L c) (σ' c))) / (Fintype.card C : ℝ) :=
            div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ =>
              leafE_off_le op row slot Q hQ plur hplur (L c) (σ' c) hi) (Nat.cast_nonneg _)
        _ ≤ Real.sqrt ((∑ c : C, Fintype.card P * Q * forkE plur slot _ (L c) (σ' c)) /
              (Fintype.card C : ℝ)) :=
            avg_sqrt_le' _ fun c => mul_nonneg (by positivity) (forkE_nonneg plur slot hΦ _ _)
        _ = _ := by rw [← Finset.mul_sum, mul_div_assoc]
  | _, _, _, .send _ L, σ', hi => leafE_off_le op row slot Q hQ plur hplur (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s upd _ k' hfresh hset hkeep L, σ', hi => by
      simp only [leafE, forkE]
      by_cases hs : s = slot
      · subst hs
        rw [if_pos rfl]
        set t := plur s σ'.1 (k' σ'.1) σ'.2
        rw [leafE_fixed plur s (t, σ'.1) (ind fun b => ∃ p, Rewinding.Off (op σ'.1) (row t) b p) _
          (fun i b h => by
            simp only [h, Option.some.injEq, Prod.mk.injEq]
            congr 1
            funext b
            refine propext ⟨fun ⟨t₁, d₁, ⟨h1, h2⟩, h3⟩ => ?_, fun h3 => ⟨_, _, ⟨rfl, rfl⟩, h3⟩⟩
            subst h1; subst h2; exact h3) _ σ'.2 (hset t σ'.1)]
        exact Rewinding.expect_bad_le (op σ'.1) (k' σ'.1) σ'.2 (row t) (hplur _ _ _) Q (hQ _)
      · rw [if_neg hs]
        exact leafE_off_le op row slot Q hQ plur hplur _ σ'.2
          (by rw [hkeep _ _ slot (Ne.symm hs), hi])

/-! ## Two locksteps of one compiled game with the same slots -/

/-- A lockstep is aligned with itself. -/
theorem Aligned.refl : ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g'),
    Aligned L L
  | _, _, _, .ret h => .ret h h
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L => .coin C fun c => Aligned.refl (L c)
  | _, _, _, .send M L => .send M fun m => Aligned.refl (L m)
  | _, _, _, .table slot _ _ _ _ L => .table slot fun t d => Aligned.refl (L t d)

/-- **Aligned locksteps have the same fork probability.** -/
theorem forkE_congr (plur : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ)
    {R₁ R₂ : I → α → β → Prop} {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : Aligned L₁ L₂) :
    ∀ σ' : Strategy g', forkE plur slot Φ L₁ σ' = forkE plur slot Φ L₂ σ' := by
  induction h with
  | ret => intro σ'; rfl
  | coin C _ ih =>
      intro σ'
      simp only [forkE]
      exact congrArg (· / _) (Finset.sum_congr rfl fun c _ => ih c (σ' c))
  | send M _ ih => intro σ'; exact ih σ'.1 σ'.2
  | table s _ ih =>
      intro σ'
      simp only [forkE]
      split_ifs
      · rfl
      · exact ih _ σ'.1 σ'.2

/-- **Sequencing preserves alignment.** -/
theorem Aligned.bind {α₂ β₂ : Type} {R₁ R₂ : I → α → β → Prop} {R₁' R₂' : I → α₂ → β₂ → Prop}
    {f₁ f₂ : α → Game α₂} {f' : β → Game β₂}
    {h₁ : ∀ i a b, R₁ i a b → Lock T D get R₁' i (f₁ a) (f' b)}
    {h₂ : ∀ i a b, R₂ i a b → Lock T D get R₂' i (f₂ a) (f' b)}
    (hh : ∀ i₁ a₁ i₂ a₂ b (r₁ : R₁ i₁ a₁ b) (r₂ : R₂ i₂ a₂ b),
      Aligned (h₁ i₁ a₁ b r₁) (h₂ i₂ a₂ b r₂))
    {i₁ i₂ : I} {g₁ g₂ : Game α} {g' : Game β}
    {L₁ : Lock T D get R₁ i₁ g₁ g'} {L₂ : Lock T D get R₂ i₂ g₂ g'} (h : Aligned L₁ L₂) :
    Aligned (Lock.bind h₁ L₁) (Lock.bind h₂ L₂) := by
  induction h with
  | ret r₁ r₂ => exact hh _ _ _ _ _ r₁ r₂
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

section InterleaveAligned

variable {α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} [Inhabited T] [Inhabited D]
  {R₁ R₁' : I₁ → α₁ → β₁ → Prop} {R₂ R₂' : I₂ → α₂ → β₂ → Prop}

theorem Aligned.embedLeft (hd : Disjoint get₁ get₂) {i₂ i₂' : I₂} {a₂ a₂' : α₂} {b₂ : β₂}
    (h₂ : R₂ i₂ a₂ b₂) (h₂' : R₂' i₂' a₂' b₂) {i₁ i₁' : I₁} {g₁ g₁'' : Game α₁} {g₁' : Game β₁}
    {L : Lock T D get₁ R₁ i₁ g₁ g₁'} {L' : Lock T D get₁ R₁' i₁' g₁'' g₁'} (h : Aligned L L') :
    Aligned (embedLeft hd h₂ L) (embedLeft hd h₂' L') := by
  induction h with
  | ret r r' => exact .ret ⟨r, h₂⟩ ⟨r', h₂'⟩
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem Aligned.embedRight (hd : Disjoint get₁ get₂) {i₁ i₁' : I₁} {a₁ a₁' : α₁} {b₁ : β₁}
    (h₁ : R₁ i₁ a₁ b₁) (h₁' : R₁' i₁' a₁' b₁) {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂}
    {M : Lock T D get₂ R₂ i₂ g₂ g₂'} {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'} (h : Aligned M M') :
    Aligned (embedRight hd h₁ M) (embedRight hd h₁' M') := by
  induction h with
  | ret r r' => exact .ret ⟨h₁, r⟩ ⟨h₁', r'⟩
  | coin C _ ih => exact .coin C ih
  | send M _ ih => exact .send M ih
  | table s _ ih => exact .table s ih

theorem Aligned.lockWith (hd : Disjoint get₁ get₂) {i₁ i₁' : I₁}
    {adv₁ adv₁' : Game α₂ → Game (α₁ × α₂)} {advC : Game β₂ → Game (β₁ × β₂)}
    {fin₁ fin₁' : α₂ → Game (α₁ × α₂)} {finC : β₂ → Game (β₁ × β₂)}
    {hadv : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}, Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (adv₁ g₂) (advC g₂')}
    {hadv' : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}, Lock T D get₂ R₂' i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁' R₂') (i₁', i₂) (adv₁' g₂) (advC g₂')}
    {hfin : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}, R₂ i₂ a₂ b₂ →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (fin₁ a₂) (finC b₂)}
    {hfin' : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}, R₂' i₂ a₂ b₂ →
      Lock T D (getPair get₁ get₂) (RPair R₁' R₂') (i₁', i₂) (fin₁' a₂) (finC b₂)}
    (hA : ∀ {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂} {M : Lock T D get₂ R₂ i₂ g₂ g₂'}
      {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'}, Aligned M M' → Aligned (hadv M) (hadv' M'))
    (hF : ∀ {i₂ i₂' : I₂} {a₂ a₂' : α₂} {b₂ : β₂} (r : R₂ i₂ a₂ b₂) (r' : R₂' i₂' a₂' b₂),
      Aligned (hfin r) (hfin' r')) :
    ∀ {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂} {M : Lock T D get₂ R₂ i₂ g₂ g₂'}
      {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'}, Aligned M M' →
        Aligned (lockWith hd hadv hfin M) (lockWith hd hadv' hfin' M')
  | _, _, _, _, _, _, _, .ret r r' => hF r r'
  | _, _, _, _, _, _, _, @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL => by
      simp only [Lock.lockWith]
      refine .send Bool fun first => ?_
      cases first
      · exact @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          Aligned.lockWith hd hA hF (hL c)
      · exact hA (@Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL)
  | _, _, _, _, _, _, _, .send M hL => by
      simp only [Lock.lockWith]
      refine .send Bool fun first => ?_
      cases first
      · exact .send M fun m => Aligned.lockWith hd hA hF (hL m)
      · exact hA (.send M hL)
  | _, _, _, _, _, _, _, .table s hL => by
      simp only [Lock.lockWith]
      refine .send Bool fun first => ?_
      cases first
      · exact .table s fun t d => Aligned.lockWith hd hA hF (hL t d)
      · exact hA (.table s hL)

theorem Aligned.interleaveLock (hd : Disjoint get₁ get₂) :
    ∀ {i₁ i₁' : I₁} {g₁ g₁'' : Game α₁} {g₁' : Game β₁} {L : Lock T D get₁ R₁ i₁ g₁ g₁'}
      {L' : Lock T D get₁ R₁' i₁' g₁'' g₁'}, Aligned L L' →
    ∀ {i₂ i₂' : I₂} {g₂ g₂'' : Game α₂} {g₂' : Game β₂} {M : Lock T D get₂ R₂ i₂ g₂ g₂'}
      {M' : Lock T D get₂ R₂' i₂' g₂'' g₂'}, Aligned M M' →
        Aligned (interleaveLock hd L M) (interleaveLock hd L' M')
  | _, _, _, _, _, _, _, .ret r r', _, _, _, _, _, _, _, hM => Aligned.embedRight hd r r' hM
  | _, _, _, _, _, _, _, @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ L₁ L₂ hL,
      _, _, _, _, _, _, _, hM =>
      Aligned.lockWith hd
        (hadv := fun M => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c =>
          Lock.interleaveLock hd (L₁ c) M)
        (hadv' := fun M => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c =>
          Lock.interleaveLock hd (L₂ c) M)
        (hfin := fun r => Lock.embedLeft hd r (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₁))
        (hfin' := fun r => Lock.embedLeft hd r (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₂))
        (fun hM' => @Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ fun c =>
          Aligned.interleaveLock hd (hL c) hM')
        (fun r r' => Aligned.embedLeft hd r r' (@Aligned.coin _ _ _ _ _ _ _ _ _ _ C iF iN _ _ _ _ _ hL))
        hM
  | _, _, _, _, _, _, _, @Aligned.send _ _ _ _ _ _ _ _ _ _ M _ _ _ L₁ L₂ hL,
      _, _, _, _, _, _, _, hM =>
      Aligned.lockWith hd
        (hadv := fun M' => Lock.send M fun m => Lock.interleaveLock hd (L₁ m) M')
        (hadv' := fun M' => Lock.send M fun m => Lock.interleaveLock hd (L₂ m) M')
        (hfin := fun r => Lock.embedLeft hd r (Lock.send M L₁))
        (hfin' := fun r => Lock.embedLeft hd r (Lock.send M L₂))
        (fun hM' => .send M fun m => Aligned.interleaveLock hd (hL m) hM')
        (fun r r' => Aligned.embedLeft hd r r' (.send M hL)) hM
  | _, _, _, _, _, _, _,
      @Aligned.table _ _ _ _ _ _ _ _ _ _ s upd₁ upd₂ _ _ _ hf₁ hs₁ hk₁ hf₂ hs₂ hk₂ L₁ L₂ hL,
      _, _, _, _, _, _, _, hM =>
      Aligned.lockWith hd
        (hadv := fun {i₂ _ _} M => Lock.table s (fun t d => (upd₁ t d, i₂))
          (getPair_fresh₁ hd hf₁ hs₁ i₂) (getPair_set₁ hs₁ i₂)
          (fun t d s' h => getPair_keep₁ hk₁ i₂ t d s' h) fun t d => Lock.interleaveLock hd (L₁ t d) M)
        (hadv' := fun {i₂ _ _} M => Lock.table s (fun t d => (upd₂ t d, i₂))
          (getPair_fresh₁ hd hf₂ hs₂ i₂) (getPair_set₁ hs₂ i₂)
          (fun t d s' h => getPair_keep₁ hk₂ i₂ t d s' h) fun t d => Lock.interleaveLock hd (L₂ t d) M)
        (hfin := fun r => Lock.embedLeft hd r (Lock.table s upd₁ hf₁ hs₁ hk₁ L₁))
        (hfin' := fun r => Lock.embedLeft hd r (Lock.table s upd₂ hf₂ hs₂ hk₂ L₂))
        (fun hM' => .table s fun t d => Aligned.interleaveLock hd (hL t d) hM')
        (fun r r' => Aligned.embedLeft hd r r' (.table s hL)) hM

end InterleaveAligned

end Lock

end Game

end FlockSoundness
