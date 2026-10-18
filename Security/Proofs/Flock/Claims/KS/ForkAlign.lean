import Proofs.Flock.Claims.KS.ForkSim

/-!
# A fork at a table node, aligned with a fork after a number of coins

`Lock.forkE` forks two runs at the table node of `slot`; `forkC F j` forks them after `j` coins. Where the strategy
reaches that node after exactly `j` coins on every path, or forks to nothing on the paths that don't (`SlotAt`), and
`Φ` at the node's cap is at most `F` on the outcomes the leaves below relate (`hΦ`), the first is at most the second:

* `expect_le_of_get`, `expect₂_le_of_get`: below a node whose index holds a slot's entry, an expectation, and one over
  two runs, are monotone in a bound on the outcomes the leaves relate at that entry, since every table node below has
  a fresh slot and keeps the entry;
* `SlotAt`, `forkE_le_forkC`: the alignment, and the bound it gives.
-/

namespace FlockSoundness.Game.Lock

/-- **Below an entry, a bound on the related outcomes bounds the expectation**: where `i` holds `td` at `slot`, so does
every leaf below (each table node's slot is fresh, so not `slot`), and `f ≤ f'` on the outcomes related there. -/
theorem expect_le_of_get {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) (td : T × D) {f f' : β → ℝ}
    (h : ∀ i a b, R i a b → get i slot = some td → f b ≤ f' b) :
    ∀ {i : I} {g : Game α} {g' : Game β}, Lock T D get R i g g' → ∀ σ' : Strategy g',
      get i slot = some td → expect f g' σ' ≤ expect f' g' σ'
  | _, _, _, .ret hr, _, hi => h _ _ _ hr hi
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [expect]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun c _ => expect_le_of_get plur slot td h (L c) (σ' c) hi) (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ', hi => expect_le_of_get plur slot td h (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' hf _ hk L, σ', hi => by
      have hs : slot ≠ s := fun e => by rw [← e, hi] at hf; exact Option.some_ne_none _ hf
      exact expect_le_of_get plur slot td h (L (plur s σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2 (by rw [hk _ _ slot hs, hi])

/-- **Below an entry, a bound on the related pairs bounds the expectation over two runs.** -/
theorem expect₂_le_of_get {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) (td : T × D) {F F' : β → β → ℝ}
    (h : ∀ i₁ a₁ b₁ i₂ a₂ b₂, R i₁ a₁ b₁ → R i₂ a₂ b₂ → get i₁ slot = some td → get i₂ slot = some td →
      F b₁ b₂ ≤ F' b₁ b₂)
    {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g') (hi : get i slot = some td) :
    expect₂ F g' σ' ≤ expect₂ F' g' σ' := by
  unfold expect₂
  refine expect_le_of_get (get := get) (R := R) plur slot td (fun i₁ a₁ b₁ h₁ hi₁ => ?_) L σ' hi
  exact expect_le_of_get (get := get) (R := R) plur slot td
    (fun i₂ a₂ b₂ h₂ hi₂ => h i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hi₁ hi₂) L σ' hi

/-- **The strategy reaches `slot`'s table node after exactly `j` coins** through `L`: each path ends, or passes sends
and other table nodes and meets that node with no coin left, or is a subtree whose fork there is `0`. -/
inductive SlotAt {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) (Φ : D → β → β → ℝ) :
    ℕ → {i : I} → {g : Game α} → {g' : Game β} → Lock T D get R i g g' → Strategy g' → Prop
  | ret {j : ℕ} {i : I} {a : α} {b : β} (h : R i a b) (σ' : Strategy (.ret b)) :
      SlotAt plur slot Φ j (Lock.ret h) σ'
  | coin {j : ℕ} {i : I} (C : Type) [Fintype C] [Nonempty C] {k : C → Game α} {k' : C → Game β}
      {L : (c : C) → Lock T D get R i (k c) (k' c)} {σ' : Strategy (.coin C k')} :
      (∀ c, SlotAt plur slot Φ j (L c) (σ' c)) → SlotAt plur slot Φ (j + 1) (Lock.coin C L) σ'
  | send {j : ℕ} {i : I} (M : Type) {k : M → Game α} {k' : M → Game β}
      {L : (m : M) → Lock T D get R i (k m) (k' m)} {σ' : Strategy (.send M k')} :
      SlotAt plur slot Φ j (L σ'.1) σ'.2 → SlotAt plur slot Φ j (Lock.send M L) σ'
  | other {j : ℕ} {i : I} (s : ℕ) {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i s = none} {hs : ∀ t d, get (upd t d) s = some (t, d)}
      {hk : ∀ t d s', s' ≠ s → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} {σ' : Strategy (.send D k')}
      (hne : s ≠ slot) :
      SlotAt plur slot Φ j (L (plur s σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2 →
        SlotAt plur slot Φ j (Lock.table s upd hf hs hk L) σ'
  | here {i : I} {upd : T → D → I} {k : T → Game α} {k' : D → Game β}
      {hf : get i slot = none} {hs : ∀ t d, get (upd t d) slot = some (t, d)}
      {hk : ∀ t d s', s' ≠ slot → get (upd t d) s' = get i s'}
      {L : (t : T) → (d : D) → Lock T D get R (upd t d) (k t) (k' d)} (σ' : Strategy (.send D k')) :
      SlotAt plur slot Φ 0 (Lock.table slot upd hf hs hk L) σ'
  | nul {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'} :
      forkE plur slot Φ L σ' = 0 → SlotAt plur slot Φ j L σ'

/-- **A fork at the slot's node is at most the fork after `j` coins** where the strategy reaches the node after `j`
coins (`SlotAt`) and `Φ` at the node's cap is at most `F` on the pairs the leaves relate below it. -/
theorem forkE_le_forkC {T D I α β : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}
    (plur : Chooser T D β) (slot : ℕ) {Φ : D → β → β → ℝ} {F : β → β → ℝ} (hF : ∀ b b', 0 ≤ F b b')
    (hΦ : ∀ t d i₁ a₁ b₁ i₂ a₂ b₂, R i₁ a₁ b₁ → R i₂ a₂ b₂ → get i₁ slot = some (t, d) →
      get i₂ slot = some (t, d) → Φ d b₁ b₂ ≤ F b₁ b₂)
    {j : ℕ} {i : I} {g : Game α} {g' : Game β} {L : Lock T D get R i g g'} {σ' : Strategy g'}
    (h : SlotAt plur slot Φ j L σ') : forkE plur slot Φ L σ' ≤ forkC F j g' σ' := by
  induction h with
  | ret _ _ => exact forkC_nonneg hF _ _ _
  | coin C _ ih =>
      rw [forkC_coin]
      simp only [forkE]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ => ih c) (Nat.cast_nonneg _)
  | send M _ ih => rw [forkC_send]; exact ih
  | other s hne _ ih =>
      simp only [forkE, hne, ↓reduceIte]
      rw [forkC_send]
      exact ih
  | @here i upd k k' hf hs hk L σ' =>
      simp only [forkE, ↓reduceIte]
      rw [forkC_zero]
      exact expect₂_le_of_get (get := get) (R := R) plur slot (plur slot σ'.1 (k' σ'.1) σ'.2, σ'.1)
        (fun i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hi₁ hi₂ => hΦ _ _ i₁ a₁ b₁ i₂ a₂ b₂ h₁ h₂ hi₁ hi₂)
        (L (plur slot σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2 (hs _ _)
  | nul h0 => rw [h0]; exact forkC_nonneg hF _ _ _

end FlockSoundness.Game.Lock
