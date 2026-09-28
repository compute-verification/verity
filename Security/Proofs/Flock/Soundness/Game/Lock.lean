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

/-- The compiled game `g'` in lockstep with the oracle game `g` (see the module doc). `get i s`
is the table fixed at slot `s`, `none` before the slot's node. -/
inductive Lock (T D : Type) {I : Type} (get : I → ℕ → Option (T × D)) (R : I → α → β → Prop) :
    I → Game α → Game β → Type 1
  | ret {i : I} {a : α} {b : β} : R i a b → Lock T D get R i (.ret a) (.ret b)
  | coin {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β} :
      ((c : C) → Lock T D get R i (k c) (k' c)) → Lock T D get R i (.coin C k) (.coin C k')
  | send {i : I} (M : Type) {k : M → Game α} {k' : M → Game β} :
      ((m : M) → Lock T D get R i (k m) (k' m)) → Lock T D get R i (.send M k) (.send M k')
  | table {i : I} (slot : ℕ) (upd : T → D → I) {k : T → Game α} {k' : D → Game β}
      (hfresh : get i slot = none) (hset : ∀ t d, get (upd t d) slot = some (t, d))
      (hkeep : ∀ t d s, s ≠ slot → get (upd t d) s = get i s) :
      ((t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)) →
        Lock T D get R i (.send T k) (.send D k')

namespace Lock

variable {T D I : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}

/-- The table chosen at a slot, from the cap and the continuation's strategy. -/
abbrev Chooser (T D : Type) (β : Type) := ℕ → D → (g : Game β) → Strategy g → T

/-- **The oracle-model prover `P̃`**: the chosen table at every table node, `σ'` elsewhere. -/
def oracleStrat (plur : Chooser T D β) : {i : I} → {g : Game α} → {g' : Game β} →
    Lock T D get R i g g' → Strategy g' → Strategy g
  | _, _, _, .ret _, _ => PUnit.unit
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => fun c : C => oracleStrat plur (L c) (σ' c)
  | _, _, _, .send _ L, σ' => ⟨σ'.1, oracleStrat plur (L σ'.1) σ'.2⟩
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ slot _ _ k' _ _ _ L, σ' =>
      ⟨plur slot σ'.1 (k' σ'.1) σ'.2,
        oracleStrat plur (L (plur slot σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2⟩

/-- The expectation of a leaf function `F i b` along the compiled runs, with the index following
the chosen tables. -/
noncomputable def leafE (plur : Chooser T D β) (F : I → β → ℝ) : {i : I} → {g : Game α} →
    {g' : Game β} → Lock T D get R i g g' → Strategy g' → ℝ
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i _ b _, _ => F i b
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF _ _ _ L, σ' =>
      (∑ c : C, leafE plur F (L c) (σ' c)) / (@Fintype.card C iF)
  | _, _, _, .send _ L, σ' => leafE plur F (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ slot _ _ k' _ _ _ L, σ' =>
      leafE plur F (L (plur slot σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2

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

/-- Weaken the leaf relation. -/
def mono {R' : I → α → β → Prop} (h : ∀ i a b, R i a b → R' i a b) : {i : I} → {g : Game α} →
    {g' : Game β} → Lock T D get R i g g' → Lock T D get R' i g g'
  | _, _, _, .ret hr => .ret (h _ _ _ hr)
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => mono h (L c)
  | _, _, _, .send M L => .send M fun m => mono h (L m)
  | _, _, _, .table slot upd hf hs hk L => .table slot upd hf hs hk fun t d => mono h (L t d)

/-- **Sequencing** locksteps. -/
def bind {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} {f : α → Game α₂} {f' : β → Game β₂}
    (h : ∀ i a b, R i a b → Lock T D get R₂ i (f a) (f' b)) : {i : I} → {g : Game α} →
    {g' : Game β} → Lock T D get R i g g' → Lock T D get R₂ i (g.bind f) (g'.bind f')
  | _, _, _, .ret hr => h _ _ _ hr
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => bind h (L c)
  | _, _, _, .send M L => .send M fun m => bind h (L m)
  | _, _, _, .table slot upd hf hs hk L => .table slot upd hf hs hk fun t d => bind h (L t d)

/-- Relabel both outputs. -/
def map {α₂ β₂ : Type} {R₂ : I → α₂ → β₂ → Prop} (f : α → α₂) (f' : β → β₂)
    (h : ∀ i a b, R i a b → R₂ i (f a) (f' b)) {i : I} {g : Game α} {g' : Game β}
    (L : Lock T D get R i g g') : Lock T D get R₂ i (g.map f) (g'.map f') :=
  bind (fun i a b hr => .ret (h i a b hr)) L

/-- The same game on both sides, with no table nodes. -/
def same {i : I} : (g : Game α) → Lock T D get (fun i' a b => i' = i ∧ a = b) i g g
  | .ret _ => .ret ⟨rfl, rfl⟩
  | @Game.coin _ C _ _ k => .coin C fun c => same (k c)
  | .send M k => .send M fun m => same (k m)

/-- The oracle game with a dummy last message, the counterpart of the compiled game's openings. -/
def withDummy (O : Type) (g : Game α) : Game α := g.bind fun a => .send O fun _ => .ret a

/-- The strategy of `g` a strategy of `withDummy O g` plays. -/
def dropDummy (O : Type) : (g : Game α) → Strategy (withDummy O g) → Strategy g
  | .ret _, _ => PUnit.unit
  | @Game.coin _ C _ _ k, σ => fun c : C => dropDummy O (k c) (σ c)
  | .send _ k, σ => ⟨σ.1, dropDummy O (k σ.1) σ.2⟩

/-- A dummy last message changes no expectation. -/
theorem expect_withDummy (O : Type) (f : α → ℝ) : ∀ (g : Game α) (σ : Strategy (withDummy O g)),
    expect f (withDummy O g) σ = expect f g (dropDummy O g σ)
  | .ret _, _ => rfl
  | @Game.coin _ C _ _ k, σ => by
      simp only [withDummy, bind_coin, expect]
      exact congrArg (· / _) (Finset.sum_congr rfl fun c _ => expect_withDummy O f (k c) (σ c))
  | .send _ k, σ => expect_withDummy O f (k σ.1) σ.2


/-! ## Two locksteps interleaved -/

section Interleave

variable {α₁ β₁ α₂ β₂ I₁ I₂ : Type} {get₁ : I₁ → ℕ → Option (T × D)}
  {get₂ : I₂ → ℕ → Option (T × D)} {R₁ : I₁ → α₁ → β₁ → Prop} {R₂ : I₂ → α₂ → β₂ → Prop}

/-- The tables two locksteps have fixed. -/
def getPair (get₁ : I₁ → ℕ → Option (T × D)) (get₂ : I₂ → ℕ → Option (T × D)) (i : I₁ × I₂)
    (s : ℕ) : Option (T × D) :=
  (get₁ i.1 s).orElse fun _ => get₂ i.2 s

/-- Both leaf relations. -/
def RPair (R₁ : I₁ → α₁ → β₁ → Prop) (R₂ : I₂ → α₂ → β₂ → Prop) (i : I₁ × I₂) (a : α₁ × α₂)
    (b : β₁ × β₂) : Prop :=
  R₁ i.1 a.1 b.1 ∧ R₂ i.2 a.2 b.2

/-- The two locksteps' slots are disjoint. -/
structure Disjoint (get₁ : I₁ → ℕ → Option (T × D)) (get₂ : I₂ → ℕ → Option (T × D)) : Prop where
  left : ∀ i₁ i₂ s, (get₁ i₁ s).isSome → get₂ i₂ s = none
  right : ∀ i₁ i₂ s, (get₂ i₂ s).isSome → get₁ i₁ s = none

variable [Inhabited T] [Inhabited D]

theorem getPair_fresh₁ (hd : Disjoint get₁ get₂) {i₁ : I₁} {slot : ℕ} {upd : T → D → I₁}
    (hf : get₁ i₁ slot = none) (hs : ∀ t d, get₁ (upd t d) slot = some (t, d)) (i₂ : I₂) :
    getPair get₁ get₂ (i₁, i₂) slot = none := by
  simp only [getPair, hf, Option.orElse_none]
  exact hd.left (upd default default) i₂ slot (by rw [hs]; rfl)

theorem getPair_set₁ {slot : ℕ} {upd : T → D → I₁} (hs : ∀ t d, get₁ (upd t d) slot = some (t, d))
    (i₂ : I₂) (t : T) (d : D) : getPair get₁ get₂ (upd t d, i₂) slot = some (t, d) := by
  simp only [getPair, hs t d]; rfl

theorem getPair_keep₁ {i₁ : I₁} {slot : ℕ} {upd : T → D → I₁}
    (hk : ∀ t d s, s ≠ slot → get₁ (upd t d) s = get₁ i₁ s) (i₂ : I₂) (t : T) (d : D) (s : ℕ)
    (hs : s ≠ slot) : getPair get₁ get₂ (upd t d, i₂) s = getPair get₁ get₂ (i₁, i₂) s := by
  simp only [getPair, hk t d s hs]

theorem getPair_fresh₂ (hd : Disjoint get₁ get₂) {i₂ : I₂} {slot : ℕ} {upd : T → D → I₂}
    (hf : get₂ i₂ slot = none) (hs : ∀ t d, get₂ (upd t d) slot = some (t, d)) (i₁ : I₁) :
    getPair get₁ get₂ (i₁, i₂) slot = none := by
  simp only [getPair, hd.right i₁ (upd default default) slot (by rw [hs]; rfl),
    Option.orElse_none, hf]

theorem getPair_set₂ (hd : Disjoint get₁ get₂) {slot : ℕ} {upd : T → D → I₂}
    (hs : ∀ t d, get₂ (upd t d) slot = some (t, d)) (i₁ : I₁) (t : T) (d : D) :
    getPair get₁ get₂ (i₁, upd t d) slot = some (t, d) := by
  simp only [getPair, hd.right i₁ (upd t d) slot (by rw [hs]; rfl), Option.orElse_none, hs t d]

theorem getPair_keep₂ {i₂ : I₂} {slot : ℕ} {upd : T → D → I₂}
    (hk : ∀ t d s, s ≠ slot → get₂ (upd t d) s = get₂ i₂ s) (i₁ : I₁) (t : T) (d : D) (s : ℕ)
    (hs : s ≠ slot) : getPair get₁ get₂ (i₁, upd t d) s = getPair get₁ get₂ (i₁, i₂) s := by
  simp only [getPair, hk t d s hs]

/-- Rep 2 ended with related outputs; rep 1 continues alone. -/
def embedLeft (hd : Disjoint get₁ get₂) {i₂ : I₂} {a₂ : α₂} {b₂ : β₂} (h₂ : R₂ i₂ a₂ b₂) :
    {i₁ : I₁} → {g₁ : Game α₁} → {g₁' : Game β₁} → Lock T D get₁ R₁ i₁ g₁ g₁' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (g₁.map fun a => (a, a₂))
        (g₁'.map fun b => (b, b₂))
  | _, _, _, .ret h₁ => .ret ⟨h₁, h₂⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => embedLeft hd h₂ (L c)
  | _, _, _, .send M L => .send M fun m => embedLeft hd h₂ (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .table slot (fun t d => (upd t d, i₂)) (getPair_fresh₁ hd hf hs i₂) (getPair_set₁ hs i₂)
        (fun t d s h => getPair_keep₁ hk i₂ t d s h) fun t d => embedLeft hd h₂ (L t d)

/-- Rep 1 ended with related outputs; rep 2 continues alone. -/
def embedRight (hd : Disjoint get₁ get₂) {i₁ : I₁} {a₁ : α₁} {b₁ : β₁} (h₁ : R₁ i₁ a₁ b₁) :
    {i₂ : I₂} → {g₂ : Game α₂} → {g₂' : Game β₂} → Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (g₂.map fun a => (a₁, a))
        (g₂'.map fun b => (b₁, b))
  | _, _, _, .ret h₂ => .ret ⟨h₁, h₂⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => embedRight hd h₁ (L c)
  | _, _, _, .send M L => .send M fun m => embedRight hd h₁ (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .table slot (fun t d => (i₁, upd t d)) (getPair_fresh₂ hd hf hs i₁) (getPair_set₂ hd hs i₁)
        (fun t d s h => getPair_keep₂ hk i₁ t d s h) fun t d => embedRight hd h₁ (L t d)

/-- `interleaveWith` in lockstep: rep 2 advances, or rep 1 takes a step (`hadv`), or rep 2 has
ended (`hfin`). -/
def lockWith (hd : Disjoint get₁ get₂) {i₁ : I₁} {adv₁ : Game α₂ → Game (α₁ × α₂)}
    {adv₁' : Game β₂ → Game (β₁ × β₂)} {fin₁ : α₂ → Game (α₁ × α₂)} {fin₁' : β₂ → Game (β₁ × β₂)}
    (hadv : ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂}, Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (adv₁ g₂) (adv₁' g₂'))
    (hfin : ∀ {i₂ : I₂} {a₂ : α₂} {b₂ : β₂}, R₂ i₂ a₂ b₂ →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (fin₁ a₂) (fin₁' b₂)) :
    {i₂ : I₂} → {g₂ : Game α₂} → {g₂' : Game β₂} → Lock T D get₂ R₂ i₂ g₂ g₂' →
      Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (interleaveWith adv₁ fin₁ g₂)
        (interleaveWith adv₁' fin₁' g₂')
  | _, _, _, .ret h₂ => hfin h₂
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L =>
      .send Bool fun first => match first with
        | true => hadv (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L)
        | false => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => lockWith hd hadv hfin (L c)
  | _, _, _, .send M L =>
      .send Bool fun first => match first with
        | true => hadv (.send M L)
        | false => .send M fun m => lockWith hd hadv hfin (L m)
  | _, _, _, .table slot upd hf hs hk L =>
      .send Bool fun first => match first with
        | true => hadv (.table slot upd hf hs hk L)
        | false => .table slot (fun t d => (i₁, upd t d)) (getPair_fresh₂ hd hf hs i₁)
            (getPair_set₂ hd hs i₁) (fun t d s h => getPair_keep₂ hk i₁ t d s h)
            fun t d => lockWith hd hadv hfin (L t d)

/-- **Two locksteps interleaved**, whatever schedule the prover chooses. -/
def interleaveLock (hd : Disjoint get₁ get₂) : {i₁ : I₁} → {g₁ : Game α₁} → {g₁' : Game β₁} →
    Lock T D get₁ R₁ i₁ g₁ g₁' → ∀ {i₂ : I₂} {g₂ : Game α₂} {g₂' : Game β₂},
      Lock T D get₂ R₂ i₂ g₂ g₂' →
        Lock T D (getPair get₁ get₂) (RPair R₁ R₂) (i₁, i₂) (interleave g₁ g₂) (interleave g₁' g₂')
  | _, _, _, .ret h₁, _, _, _, L₂ => embedRight hd h₁ L₂
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₁, _, _, _, L₂ =>
      lockWith hd (fun L₂ => @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ fun c => interleaveLock hd (L₁ c) L₂)
        (fun h₂ => embedLeft hd h₂ (@Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L₁)) L₂
  | _, _, _, .send M L₁, _, _, _, L₂ =>
      lockWith hd (fun L₂ => .send M fun m => interleaveLock hd (L₁ m) L₂)
        (fun h₂ => embedLeft hd h₂ (.send M L₁)) L₂
  | _, _, _, .table slot upd hf hs hk L₁, _, _, i₂, L₂ =>
      lockWith hd (fun {i₂ _ _} L₂ => .table slot (fun t d => (upd t d, i₂))
          (getPair_fresh₁ hd hf hs i₂) (getPair_set₁ hs i₂) (fun t d s h => getPair_keep₁ hk i₂ t d s h)
          fun t d => interleaveLock hd (L₁ t d) L₂)
        (fun h₂ => embedLeft hd h₂ (.table slot upd hf hs hk L₁)) L₂

end Interleave

/-! ## Rewinding at every table node -/

/-- The expectation, at the node of `slot`, of `Φ` over two independent continuations: the
success probability of the collision finder that runs the prover to the slot's commitment and
continues it twice. `0` on runs that do not reach the slot. -/
noncomputable def forkE (plur : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ) : {i : I} →
    {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Strategy g' → ℝ
  | _, _, _, .ret _, _ => 0
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF _ _ _ L, σ' =>
      (∑ c : C, forkE plur slot Φ (L c) (σ' c)) / (@Fintype.card C iF)
  | _, _, _, .send _ L, σ' => forkE plur slot Φ (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' =>
      if s = slot then expect₂ (Φ σ'.1) (k' σ'.1) σ'.2
      else forkE plur slot Φ (L (plur s σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2

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

/-- Two locksteps of one compiled game with the same table slots. Their oracle games and leaf
relations may differ; `forkE` reads only the compiled side and the slots (`forkE_congr`). -/
inductive Aligned {I : Type} {get : I → ℕ → Option (T × D)} {R₁ R₂ : I → α → β → Prop} :
    {i₁ i₂ : I} → {g₁ g₂ : Game α} → {g' : Game β} →
      Lock T D get R₁ i₁ g₁ g' → Lock T D get R₂ i₂ g₂ g' → Prop
  | ret {i₁ i₂ : I} {a₁ a₂ : α} {b : β} (h₁ : R₁ i₁ a₁ b) (h₂ : R₂ i₂ a₂ b) :
      Aligned (Lock.ret h₁) (Lock.ret h₂)
  | coin {i₁ i₂ : I} (C : Type) [Fintype C] [Nonempty C] {k₁ k₂ : C → Game α} {k' : C → Game β}
      {L₁ : (c : C) → Lock T D get R₁ i₁ (k₁ c) (k' c)}
      {L₂ : (c : C) → Lock T D get R₂ i₂ (k₂ c) (k' c)} :
      (∀ c, Aligned (L₁ c) (L₂ c)) → Aligned (Lock.coin C L₁) (Lock.coin C L₂)
  | send {i₁ i₂ : I} (M : Type) {k₁ k₂ : M → Game α} {k' : M → Game β}
      {L₁ : (m : M) → Lock T D get R₁ i₁ (k₁ m) (k' m)}
      {L₂ : (m : M) → Lock T D get R₂ i₂ (k₂ m) (k' m)} :
      (∀ m, Aligned (L₁ m) (L₂ m)) → Aligned (Lock.send M L₁) (Lock.send M L₂)
  | table {i₁ i₂ : I} (slot : ℕ) {upd₁ upd₂ : T → D → I} {k₁ k₂ : T → Game α} {k' : D → Game β}
      {hf₁ : get i₁ slot = none} {hs₁ : ∀ t d, get (upd₁ t d) slot = some (t, d)}
      {hk₁ : ∀ t d s, s ≠ slot → get (upd₁ t d) s = get i₁ s}
      {hf₂ : get i₂ slot = none} {hs₂ : ∀ t d, get (upd₂ t d) slot = some (t, d)}
      {hk₂ : ∀ t d s, s ≠ slot → get (upd₂ t d) s = get i₂ s}
      {L₁ : (t : T) → (d : D) → Lock T D get R₁ (upd₁ t d) (k₁ t) (k' d)}
      {L₂ : (t : T) → (d : D) → Lock T D get R₂ (upd₂ t d) (k₂ t) (k' d)} :
      (∀ t d, Aligned (L₁ t d) (L₂ t d)) →
        Aligned (Lock.table slot upd₁ hf₁ hs₁ hk₁ L₁) (Lock.table slot upd₂ hf₂ hs₂ hk₂ L₂)

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
