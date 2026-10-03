import FlockSoundness.Discharge.ZkTree.Lazy

/-!
# The tree-coin hybrid

`treeCoins_le_model_of`: a committed prover of `g` (shown every leaf before its first message, and each round's block
and salt at its opening) is, on every event of its output and of the leaves it was shown, within `|I|·δ` of a mixture
of strategies of the model `G` (`ErD g G`), when no path opens a round twice and the leaf is two-vector `δ`-hiding
(`Hm96TwoVector`). The mixture draws for itself the leaves of independent blocks and salts, which is its `hello`, and,
at each opening, the rest of the block and a salt consistent with the leaf it showed. The mixture does not depend on
the event.

`treeCoins_le_model`: the same at hm96-sha512 under a uniform session key that the prover sees, with `δ = 2^-256` a
leaf and no hypothesis: the key and the leaves are the mixture's `hello`.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins
open scoped ENNReal

/-- **The tree-coin hybrid.** -/
theorem treeCoins_le_model_of {I Blk Salt B Cs D α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk]
    [DecidableEq Blk] [Fintype Salt] [Nonempty Salt] [AddCommGroup B] [Fintype B]
    {g : TGame I Blk α} {G : Game α} (hgG : ErD g G) (hg : g.Distinct)
    (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) {δ : ℝ≥0∞}
    (hT : Hm96TwoVector My cs δ) (τ : (I → D) → TGame.CStrat Salt g) :
    ∃ M : Mix G (I → D), ∀ E : (I → D) → α → Prop,
      avg (fun ω : I → Blk × Salt =>
        TGame.pr (E fun i => leaf dig My cs H i (ω i)) g (τ fun i => leaf dig My cs H i (ω i)) ω) ≤
        M.pr E + Fintype.card I * δ := by
  sorry

/-- **The tree-coin hybrid at hm96-sha512.** -/
theorem treeCoins_le_model {E₂ I Blk Cs D α : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂]
    (hE : Fintype.card E₂ = 2) [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk]
    [DecidableEq Cs] {g : TGame I Blk α} {G : Game α} (hgG : ErD g G) (hg : g.Distinct)
    (dig : I → Blk → (Fin 512 → E₂)) (cs : Salt E₂ → Cs) (hcs : (Finset.univ.image cs).card ≤ 2 ^ 512)
    (H : Key E₂ → (Fin 512 → E₂) × Cs → D) (τ : Key E₂ → (I → D) → TGame.CStrat (Salt E₂) g) :
    ∃ M : Mix G (Key E₂ × (I → D)), ∀ E : Key E₂ × (I → D) → α → Prop,
      avg (fun κ : Key E₂ => avg fun ω : I → Blk × Salt E₂ =>
        TGame.pr (E (κ, fun i => leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) i (ω i))) g
          (τ κ fun i => leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) i (ω i)) ω) ≤
        M.pr E + Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256) := by
  sorry

end FlockSoundness.Discharge.ZkTree
