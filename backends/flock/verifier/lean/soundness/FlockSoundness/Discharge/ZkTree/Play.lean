import FlockSoundness.Discharge.ZkTree.Audit

/-!
# The mixture's provers are the tree-coin prover at a view

`treeAudit_le_model`'s mixture is explicit: each of its provers is the tree-coin prover `τ` itself, played on the model
audit at a fixed view (`TreeProver.play`): at each draw `ω`, a key and leaves shown at `Hello`, and at each committed
round the rest of a block and a salt for each block the round may open. Played so, `τ` registers `τ.1`, sends `τ`'s
messages on that view, and its committed values (the model's knowledge extractors rewind it on the model's coins) are
`τ`'s at that view.

* `treeCoins_le_mixOf`, `treeCoins_le_avgMixOf`: `treeCoins_le_model_of` and `treeCoins_le_model` with the mixture
  named: `mixOf` (fake leaves `q`, and the posterior salt of `q`'s leaf given each block), averaged over the key.
* `treeAudit_le_play`: `treeAudit_le_model` with a view `vw x` for every prover `x` of the mixture, and that prover
  `τ.play (vw x)`.

So a hypothesis on `τ.play v` at every view `v` (the finders built from `τ` with `v` as advice) covers every prover of
the mixture. The view's law (`Mix.w`) is not efficiently samplable (the salt is drawn from the posterior of a leaf, a
preimage of SHA-512), but no hypothesis reads it: each `τ.play v` is `τ` with fixed advice, as efficient as `τ`.
-/

namespace FlockSoundness.Discharge.ZkTree

open FlockSoundness Game Discharge.ZkCoins Discharge.Exec Refine Finset Function
open scoped ENNReal

/-! ## The hybrid with its mixture named -/

section Hybrid

variable {I Blk Salt B Cs D α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk]
  [Fintype Salt] [Nonempty Salt] [DecidableEq Cs] {g : TGame I Blk α} {G : Game α}
  [AddCommGroup B] [Fintype B]

/-- **The tree-coin hybrid at `mixOf`.** -/
theorem treeCoins_le_mixOf (hgG : ErD g G) (hg : g.Distinct)
    (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) {δ : ℝ≥0∞}
    (hT : Hm96TwoVector My cs δ) (τ : (I → D) → TGame.CStrat Salt g) (E : (I → D) → α → Prop) :
    avg (fun ω : I → Blk × Salt =>
      TGame.pr (E fun i => leaf dig My cs H i (ω i)) g (τ fun i => leaf dig My cs H i (ω i)) ω) ≤
      (mixOf hgG (leaf dig My cs H) τ).pr E + Fintype.card I * δ := by
  sorry

end Hybrid

set_option exponentiation.threshold 300 in
/-- **The tree-coin hybrid at hm96-sha512, at `mixOf` averaged over the key.** -/
theorem treeCoins_le_avgMixOf {E₂ I Blk Cs D α : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂]
    (hE : Fintype.card E₂ = 2) [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk]
    [DecidableEq Cs] {g : TGame I Blk α} {G : Game α} (hgG : ErD g G) (hg : g.Distinct)
    (dig : I → Blk → (Fin 512 → E₂)) (cs : Salt E₂ → Cs) (hcs : (Finset.univ.image cs).card ≤ 2 ^ 512)
    (H : Key E₂ → (Fin 512 → E₂) × Cs → D) (τ : Key E₂ → (I → D) → TGame.CStrat (Salt E₂) g)
    (E : Key E₂ × (I → D) → α → Prop) :
    avg (fun κ : Key E₂ => avg fun ω : I → Blk × Salt E₂ =>
      TGame.pr (E (κ, fun i => leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) i (ω i))) g
        (τ κ fun i => leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) i (ω i)) ω) ≤
      (Mix.avgOver fun κ => mixOf hgG (leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ)) (τ κ)).pr E +
        Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256) := by
  sorry

/-! ## The tree-coin prover at a view -/

section Play

variable {n : ℕ} {E₂ I Blk D Nonce Reg β : Type} [Field E₂] [Fintype E₂] [Fintype I] [DecidableEq I] [Fintype Blk]
  [Nonempty Blk]

/-- **A view of the tree-coin prover at each draw**: the key and the leaves shown at `Hello`, and at each committed
round the rest of a block and a salt for each block. -/
abbrev TreeView (L : Audit.Law n) (E₂ I Blk D : Type) : Type :=
  (ω : L.Ω) → Key E₂ × (I → D) × (I → Blk × (Blk → Salt E₂))

/-- **The tree-coin prover played on the model audit at the view `v`**: it registers `τ.1`, and at each draw `ω` plays
its committed strategy at `v ω`'s key and leaves, each committed round's block the model's coin with `v ω`'s rest of
the block, and its salt `v ω`'s for that block (`toStrat`). -/
def TreeProver.play (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (out : Finset (Fin n) → Reg → β → Transcript) (τ : TreeProver L Reg Nonce E₂ D g) (v : TreeView L E₂ I Blk D) :
    Strategy (auditLive L Reg fun S R => G.map (out S R)) :=
  (⟨τ.1, fun ω => Strategy.map _ _ (Strategy.map (out (L.draw ω) τ.1) G
      (toStrat hgG ((τ.2 ω).2 (v ω).1 (v ω).2.1) (v ω).2.2))⟩ :
    (R' : Reg) × Strategy (Game.coin L.Ω fun ω => (G.map (out (L.draw ω) R')).map fun t => (R', L.draw ω, t)))

end Play

/-- **The tree-coin run of a one-stage audit against the tree-coin prover at a view**: `treeAudit_le_model`, with each
prover of the mixture `τ` played at a view `vw x`. -/
theorem treeAudit_le_play {n : ℕ} {E₂ I Blk Cs D Nonce Reg β : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂]
    (hE : Fintype.card E₂ = 2) [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Blk]
    [DecidableEq Cs] (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (hg : g.Distinct) (out : Finset (Fin n) → Reg → β → Transcript)
    (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (hcs : ∀ ν, (Finset.univ.image (cs ν)).card ≤ 2 ^ 512) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    (τ : TreeProver L Reg Nonce E₂ D g) :
    ∃ M : Mix (auditLive L Reg fun S R => G.map (out S R)) PUnit, ∃ vw : M.Ρ → TreeView L E₂ I Blk D,
      (∀ x, M.σ x = τ.play L hgG out (vw x)) ∧
      ∀ E, prTree L g out dig cs H τ E ≤ M.pr (fun _ => E) + Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256) := by
  sorry

end FlockSoundness.Discharge.ZkTree
