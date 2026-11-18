module

public import Proofs.Core.LevelsCircuit
public import Proofs.Core.Game.InterleaveBridge

/-!
# Committing late does not help, at any number of levels

The coins drawn level by level (`Levels.Coins`): at level `i`, one draw of `D_i^X` for every block `X`, and `Q_k`
(`Levels.drawTo`) the blocks drawn through level `k`. A commitment made after `Q_k` is drawn is a function of the coins
of the levels below `k` (`Levels.ReadsBelow`), of any of them, the other blocks' included.

`Levels.LateAtLevels`: at the levels of a circuit's partitions (`Levels.ofPartitions`), the prover commits a trace
`τ k` after `Q_k`, by any rule, each agreeing with the one before on that level's committed gates. Every sample then misses the
last trace's incorrect last-level blocks with probability at most `missProb` of one trace's, fixed in advance and
agreeing with the first on the first level's committed gates.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Coins drawn level by level**: at level `i`, a draw of `D_i^X` for every block `X`. -/
def Coins (L : Levels G) : Type :=
  (i : Fin L.levels.length) → (X : Finset G) → ((L.levels.get i).design X).Ω

instance (L : Levels G) : Fintype L.Coins := by unfold Coins; infer_instance

instance (L : Levels G) : Nonempty L.Coins := by unfold Coins; infer_instance

/-- `Q_k`: the blocks drawn through level `k`, `Q₀ = {univ}`. -/
def drawTo (L : Levels G) (ω : L.Coins) : ℕ → Finset (Finset G)
  | 0 => {Finset.univ}
  | k + 1 => if h : k < L.levels.length then
      (L.drawTo ω k).biUnion fun X => ((L.levels.get ⟨k, h⟩).design X).draw (ω ⟨k, h⟩ X)
    else L.drawTo ω k

/-- `f` reads only the coins of the levels below `k`. -/
def ReadsBelow {α : Type} (L : Levels G) (k : ℕ) (f : L.Coins → α) : Prop :=
  ∀ ω ω' : L.Coins, (∀ i : Fin L.levels.length, i.val < k → ω i = ω' i) → f ω = f ω'

/-- A product of per-coordinate functions over the coordinates in `S`, averaged over independent coordinates, is the
product of their averages. -/
theorem _root_.Verity.Game.avg_pi_prod_finset {ι : Type} [Fintype ι] [DecidableEq ι] {Ω : ι → Type}
    [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)] (S : Finset ι) (f : ∀ i, Ω i → ℝ≥0∞) :
    avg (fun ω : ∀ i, Ω i => ∏ i ∈ S, f i (ω i)) = ∏ i ∈ S, avg (f i) := by
  classical
  have key : (fun ω : ∀ i, Ω i => ∏ i ∈ S, f i (ω i)) =
      fun ω => ∏ i, (if i ∈ S then f i (ω i) else 1) := by
    funext ω
    rw [← Finset.prod_filter, Finset.filter_mem_eq_inter, Finset.univ_inter]
  rw [key, avg_pi_prod (fun i c => if i ∈ S then f i c else 1)]
  rw [← Finset.prod_filter_mul_prod_filter_not Finset.univ (· ∈ S), Finset.filter_mem_eq_inter,
    Finset.univ_inter]
  rw [Finset.prod_eq_one (s := Finset.univ.filter (· ∉ S)) fun i hi => by
    rw [Finset.mem_filter] at hi
    simp only [hi.2, ite_false]
    exact avg_const 1, mul_one]
  exact Finset.prod_congr rfl fun i hi => by simp only [hi, ite_true]

/-- The blocks level `k`'s designs draw inside: `P` at the first, then each level's blocks. -/
def _root_.Verity.Level.above (P : Finset (Finset G)) : List (Level G) → ℕ → Finset (Finset G)
  | _, 0 => P
  | [], _ + 1 => P
  | l :: ls, k + 1 => Level.above l.blocks.parts ls k

theorem _root_.Verity.Level.above_succ {P : Finset (Finset G)} :
    ∀ {ls : List (Level G)} {k : ℕ} (h : k < ls.length),
      Level.above P ls (k + 1) = (ls.get ⟨k, h⟩).blocks.parts
  | [], _, h => absurd h (Nat.not_lt_zero _)
  | l :: ls, 0, _ => by cases ls <;> rfl
  | l :: ls, k + 1, h => by
    rw [show Level.above P (l :: ls) (k + 1 + 1) = Level.above l.blocks.parts ls (k + 1) from rfl,
      Level.above_succ (P := l.blocks.parts) (Nat.lt_of_succ_lt_succ h)]
    rfl

theorem _root_.Verity.Level.inside_get :
    ∀ {P : Finset (Finset G)} {ls : List (Level G)}, Level.InsideBelow P ls →
      ∀ (k : ℕ) (h : k < ls.length), ∀ X ∈ Level.above P ls k, ∀ ω, ∀ b ∈ ((ls.get ⟨k, h⟩).design X).draw ω,
        b ∈ (ls.get ⟨k, h⟩).blocks.parts ∧ b ⊆ X
  | _, [], _, _, h => absurd h (Nat.not_lt_zero _)
  | _, l :: ls, hi, 0, _ => fun X hX ω b hb => hi.1 X hX ω b hb
  | _, l :: ls, hi, k + 1, h => fun X hX ω b hb =>
    Level.inside_get hi.2 k (Nat.lt_of_succ_lt_succ h) X hX ω b hb

/-- Blocks of one level meet only when equal. -/
def _root_.Verity.Level.Separate (P : Finset (Finset G)) : Prop :=
  ∀ X ∈ P, ∀ Y ∈ P, (X ∩ Y).Nonempty → X = Y

theorem _root_.Verity.Level.separate_parts (l : Level G) : Level.Separate l.blocks.parts := by
  intro X hX Y hY ⟨g, hg⟩
  exact l.blocks.eq_of_mem_parts hX hY (Finset.mem_inter.1 hg).1 (Finset.mem_inter.1 hg).2

theorem _root_.Verity.Level.separate_above {P : Finset (Finset G)} (hP : Level.Separate P) :
    ∀ (ls : List (Level G)) (k : ℕ), Level.Separate (Level.above P ls k)
  | ls, 0 => by cases ls <;> exact hP
  | [], _ + 1 => hP
  | l :: ls, k + 1 => Level.separate_above (Level.separate_parts l) ls k

theorem separate_univ : Level.Separate ({Finset.univ} : Finset (Finset G)) := by
  intro X hX Y hY _
  rw [Finset.mem_singleton.1 hX, Finset.mem_singleton.1 hY]

theorem drawTo_succ (L : Levels G) (ω : L.Coins) {k : ℕ} (h : k < L.levels.length) :
    L.drawTo ω (k + 1) =
      (L.drawTo ω k).biUnion fun X => ((L.levels.get ⟨k, h⟩).design X).draw (ω ⟨k, h⟩ X) := by
  rw [drawTo]
  split
  · rfl
  · contradiction

/-- `Q_k`'s blocks are blocks of the level above level `k`. -/
theorem drawTo_above (L : Levels G) (ω : L.Coins) :
    ∀ k, k ≤ L.levels.length → L.drawTo ω k ⊆ Level.above {Finset.univ} L.levels k
  | 0, _ => fun X hX => by cases hl : L.levels <;> simpa [drawTo, Level.above, hl] using hX
  | k + 1, hk => by
    have h : k < L.levels.length := hk
    rw [L.drawTo_succ ω h, Level.above_succ h]
    intro X hX
    obtain ⟨Y, hY, hX⟩ := Finset.mem_biUnion.1 hX
    exact (Level.inside_get L.inside k h Y (L.drawTo_above ω k h.le hY) _ X hX).1

/-- `Q_k` reads only the coins of the levels below `k`. -/
theorem drawTo_congr (L : Levels G) {ω ω' : L.Coins} :
    ∀ k, (∀ i : Fin L.levels.length, i.val < k → ω i = ω' i) → L.drawTo ω k = L.drawTo ω' k
  | 0, _ => rfl
  | k + 1, h => by
    by_cases hk : k < L.levels.length
    · rw [L.drawTo_succ ω hk, L.drawTo_succ ω' hk, L.drawTo_congr k fun i hi => h i (Nat.lt_succ_of_lt hi),
        h ⟨k, hk⟩ (Nat.lt_succ_self k)]
    · simp only [drawTo, hk, dite_false]
      exact L.drawTo_congr k fun i hi => h i (Nat.lt_succ_of_lt hi)

/-- **One level's coins, split off**: averaged over the coins, a product over `Q_{k+1}` of a reading of the coins below
`k` is the product over `Q_k` of each block's average over its own draw at level `k`. -/
theorem avg_prod_drawTo_succ (L : Levels G) {k : ℕ} (hk : k < L.levels.length)
    (F : L.Coins → Finset G → ℝ≥0∞) (hF : L.ReadsBelow k F) :
    avg (fun ω => ∏ X ∈ L.drawTo ω (k + 1), F ω X) =
      avg (fun ω => ∏ Y ∈ L.drawTo ω k,
        avg fun c : ((L.levels.get ⟨k, hk⟩).design Y).Ω =>
          ∏ X ∈ ((L.levels.get ⟨k, hk⟩).design Y).draw c, F ω X) := by
  classical
  let i : Fin L.levels.length := ⟨k, hk⟩
  have hsep := Level.separate_above separate_univ L.levels k
  refine Eq.trans (avg_update_pi (T := fun j : Fin L.levels.length => (X : Finset G) → ((L.levels.get j).design X).Ω)
    i (fun ω => ∏ X ∈ L.drawTo ω (k + 1), F ω X)).symm (congrArg avg (funext fun ω => ?_))
  have hbelow : ∀ c, ∀ j : Fin L.levels.length, j.val < k → Function.update ω i c j = ω j := fun c j hj =>
    Function.update_of_ne (fun h => by rw [h] at hj; exact lt_irrefl _ hj) _ _
  have hstep : ∀ c : (X : Finset G) → ((L.levels.get i).design X).Ω,
      ∏ X ∈ L.drawTo (Function.update ω i c) (k + 1), F (Function.update ω i c) X =
        ∏ Y ∈ L.drawTo ω k, ∏ X ∈ ((L.levels.get i).design Y).draw (c Y), F ω X := by
    intro c
    erw [L.drawTo_succ (Function.update ω i c) hk, L.drawTo_congr k (hbelow c), hF _ _ (hbelow c)]
    simp only [show Function.update ω i c ⟨k, hk⟩ = c from Function.update_self _ _ _]
    refine Finset.prod_biUnion fun Y hY Y' hY' hne => Finset.disjoint_left.2 fun X hX hX' => hne ?_
    have hab := L.drawTo_above ω k hk.le
    obtain ⟨hXp, hXY⟩ := Level.inside_get L.inside k hk Y (hab hY) _ X hX
    obtain ⟨-, hXY'⟩ := Level.inside_get L.inside k hk Y' (hab hY') _ X hX'
    obtain ⟨g, hg⟩ := (L.levels.get i).blocks.nonempty_of_mem_parts hXp
    exact hsep Y (hab hY) Y' (hab hY') ⟨g, Finset.mem_inter.2 ⟨hXY hg, hXY' hg⟩⟩
  simp only [hstep]
  exact avg_pi_prod_finset (L.drawTo ω k) fun Y c => ∏ X ∈ ((L.levels.get i).design Y).draw c, F ω X

end Levels

/-- The chain's partition at level `i`, the first below the root at `0`, and the last past its end. -/
def Partition.Chain.part {C : Circuit} : {nc : ℕ} → {Pc : Partition C nc} → Partition.Chain C Pc → ℕ →
    Σ n, Partition C n
  | nc, Pc, .nil, _ => ⟨nc, Pc⟩
  | _, _, .cons Pf _ _ .nil, _ => ⟨_, Pf⟩
  | _, _, .cons Pf _ _ (.cons _ _ _ _), 0 => ⟨_, Pf⟩
  | _, _, .cons _ _ _ (.cons Pg Rg rg rest), i + 1 => (Partition.Chain.cons Pg Rg rg rest).part i

/-- **Committing late does not help, at any number of levels.** At the levels of a circuit's partitions, the trace
`τ k` is committed after `Q_k` is drawn, by any rule, and agrees with `τ (k - 1)` on the committed gates of level
`k - 1`. Every sample misses the incorrect last-level blocks of the last trace with probability at most the miss of
those of one trace fixed in advance that agrees with `τ 0` on the first level's committed gates. -/
def LateAtLevels : Prop :=
  ∀ (C : Circuit) (ch : Partition.Chain C (Partition.whole C)) (τ : ℕ → (Levels.ofPartitions ch).Coins → C.Trace),
    (∀ k, (Levels.ofPartitions ch).ReadsBelow k (τ k)) →
    (∀ k ω, ∀ g ∈ (ch.part k).2.committed, τ (k + 1) ω g = τ k ω g) →
    ∃ τ' : C.Trace, (∀ g ∈ (ch.part 0).2.committed, τ' g = τ 0 (Classical.arbitrary _) g) ∧
      prCoin (fun ω => Disjoint ((Levels.ofPartitions ch).drawTo ω ch.levels.length)
          (((ch.part (ch.levels.length - 1)).2.incorrect (τ (ch.levels.length - 1) ω)).image
            (ch.part (ch.levels.length - 1)).2.block)) ≤
        (Levels.ofPartitions ch).design.missProb
          (((ch.part (ch.levels.length - 1)).2.incorrect τ').image (ch.part (ch.levels.length - 1)).2.block)

end Verity
