module

public import Proofs.Core.LevelsCircuit

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
