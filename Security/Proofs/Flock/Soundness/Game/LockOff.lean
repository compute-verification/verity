import Proofs.Flock.Soundness.Game.LockLin

/-!
# Off the chosen table, per position, at the slot's node

`Lock.leafE_off_le` and `Lock.leafE_off_le_link` bound a slot's off-table mass by a fork or link finder at the slot's
node. Here it is left raw, as a functional of the node's continuation, so a caller can compare it, node by node, with
the mass off any other table:

* `Rewinding.expect_off_le_sum`: a run opens some position off a table at most as often as the sum over the positions
  (the union bound);
* `Rewinding.expect_off_le_of_pluralityE`: at each position, a plurality table's off mass is at most any table's;
* `Lock.leafE_off_le_nodeOff`: per slot, the mass off the table chosen there is at most `nodeE` of the per-position
  off mass at that table (`nodeOff`).
-/

namespace FlockSoundness

namespace Rewinding

open Game

variable {α P V : Type} [Fintype P] (op : α → P → Option V)

/-- **The union bound over positions.** -/
theorem expect_off_le_sum (g : Game α) (s : Strategy g) (vstar : P → V) :
    expect (ind fun a => ∃ p, Off op vstar a p) g s ≤ ∑ p, expect (ind fun a => Off op vstar a p) g s := by
  rw [← expect_sum]
  refine expect_mono (fun a => ?_) g s
  by_cases h : ∃ p, Off op vstar a p
  · obtain ⟨p, hp⟩ := h
    calc ind (fun a => ∃ p, Off op vstar a p) a = ind (fun a => Off op vstar a p) a := by
          simp only [ind]; rw [ite_eq_left ⟨p, hp⟩, ite_eq_left hp]
      _ ≤ ∑ p, ind (fun a => Off op vstar a p) a :=
          Finset.single_le_sum (f := fun p => ind (fun a => Off op vstar a p) a)
            (fun p _ => ind_nonneg _ _) (Finset.mem_univ p)
  · rw [show ind (fun a => ∃ p, Off op vstar a p) a = 0 by simp [ind, h]]
    exact Finset.sum_nonneg fun p _ => ind_nonneg _ _

omit [Fintype P] in
/-- **A plurality table is off least**: at each position, the mass a run opens off a plurality table is at most the
mass it opens off any table `w`. -/
theorem expect_off_le_of_pluralityE [Fintype V] [DecidableEq V] (g : Game α) (s : Strategy g) (vstar : P → V)
    (hv : IsPluralityE op g s vstar) (w : P → V) (p : P) :
    expect (ind fun a => Off op vstar a p) g s ≤ expect (ind fun a => Off op w a p) g s := by
  rw [expect_off_eq, expect_off_eq]
  linarith [hv p (w p)]

end Rewinding

namespace Game

namespace Lock

variable {α β : Type} {T D I : Type} {get : I → ℕ → Option (T × D)} {R : I → α → β → Prop}

/-- **The per-position off mass at the slot's node**: with the cap `d`, the sum over the positions of the mass the
continuation opens off the table chosen there. -/
noncomputable def nodeOff {P V : Type} [Fintype P] (op : D → β → P → Option V) (row : T → P → V)
    (plur : Chooser T D β) (slot : ℕ) (d : D) (g'' : Game β) (σ'' : Strategy g'') : ℝ :=
  ∑ p, expect (ind fun b => Rewinding.Off (op d) (row (plur slot d g'' σ'')) b p) g'' σ''

/-- **Off the chosen table, raw**: the probability that a run opens a position off the table fixed at `slot` is at
most the per-position off mass at the slot's node, over the prefixes. -/
theorem leafE_off_le_nodeOff {P V : Type} [Fintype P] (op : D → β → P → Option V) (row : T → P → V) (slot : ℕ)
    (plur : Chooser T D β) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T D get R i g g') (σ' : Strategy g'),
      get i slot = none →
        leafE plur (fun i b => ind (fun b => ∃ t d, get i slot = some (t, d) ∧
            ∃ p, Rewinding.Off (op d) (row t) b p) b) L σ' ≤
          nodeE plur slot (nodeOff op row plur slot) L σ'
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i _ b _, _, hi => by
      simp only [leafE, nodeE, hi]
      simp [ind]
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [leafE, nodeE]
      exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun c _ =>
        leafE_off_le_nodeOff op row slot plur (L c) (σ' c) hi) (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ', hi => leafE_off_le_nodeOff op row slot plur (L σ'.1) σ'.2 hi
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s upd _ k' hfresh hset hkeep L, σ', hi => by
      simp only [leafE, nodeE]
      by_cases hs : s = slot
      · subst hs
        rw [ite_eq_left rfl]
        set t := plur s σ'.1 (k' σ'.1) σ'.2
        rw [leafE_fixed plur s (t, σ'.1) (ind fun b => ∃ p, Rewinding.Off (op σ'.1) (row t) b p) _
          (fun i b h => by
            simp only [h, Option.some.injEq, Prod.mk.injEq]
            congr 1
            funext b
            refine propext ⟨fun ⟨t₁, d₁, ⟨h1, h2⟩, h3⟩ => ?_, fun h3 => ⟨_, _, ⟨rfl, rfl⟩, h3⟩⟩
            subst h1; subst h2; exact h3) _ σ'.2 (hset t σ'.1)]
        exact Rewinding.expect_off_le_sum (op σ'.1) (k' σ'.1) σ'.2 (row t)
      · rw [ite_eq_right hs]
        exact leafE_off_le_nodeOff op row slot plur _ σ'.2 (by rw [hkeep _ _ slot (Ne.symm hs), hi])

/-- **At a plurality chooser, the node's off mass is at most any table's**: where the table chosen at `slot` reads as
a plurality of the slot's openings, the per-position off mass at the node is at most that off any `w`. -/
theorem nodeOff_le_of_plurality {P V : Type} [Fintype P] [Fintype V] [DecidableEq V] (op : D → β → P → Option V)
    (row : T → P → V) (plur : Chooser T D β) (slot : ℕ) (d : D) (g'' : Game β) (σ'' : Strategy g'')
    (hplur : Rewinding.IsPluralityE (op d) g'' σ'' (row (plur slot d g'' σ''))) (w : P → V) :
    nodeOff op row plur slot d g'' σ'' ≤ ∑ p, expect (ind fun b => Rewinding.Off (op d) w b p) g'' σ'' :=
  Finset.sum_le_sum fun p _ => Rewinding.expect_off_le_of_pluralityE (op d) g'' σ'' _ hplur w p

end Lock

end Game

end FlockSoundness
