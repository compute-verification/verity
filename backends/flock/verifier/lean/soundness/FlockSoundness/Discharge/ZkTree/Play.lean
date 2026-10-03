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
  [Fintype Salt] [Nonempty Salt] {g : TGame I Blk α} {G : Game α}
  [AddCommGroup B] [Fintype B]

/-- **The tree-coin hybrid at `mixOf`.** -/
theorem treeCoins_le_mixOf (hgG : ErD g G) (hg : g.Distinct)
    (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) {δ : ℝ≥0∞}
    (hT : Hm96TwoVector My cs δ) (τ : (I → D) → TGame.CStrat Salt g) (E : (I → D) → α → Prop) :
    avg (fun ω : I → Blk × Salt =>
      TGame.pr (E fun i => leaf dig My cs H i (ω i)) g (τ fun i => leaf dig My cs H i (ω i)) ω) ≤
      (mixOf hgG (leaf dig My cs H) τ).pr E + Fintype.card I * δ := by
  set ℓ := leaf dig My cs H with hℓ
  let g' : TGame I Blk ((I → D) × α) := .send (I → D) fun v => g.map (Prod.mk v)
  let τ' : (I → D) → TGame.CStrat Salt g' := fun v => ⟨v, TGame.CStrat.mapS (Prod.mk v) g (τ v)⟩
  let p' : (I → D) × α → Prop := fun x => E x.1 x.2
  have hexp : ∀ m, Exp p' g' ℓ τ' m = ∑ ω, W (wt ℓ m) ω * TGame.pr (E (view ℓ m ω)) g (τ (view ℓ m ω)) ω :=
    fun m => Finset.sum_congr rfl fun ω _ => congrArg _ (TGame.pr_mapS p' (Prod.mk (view ℓ m ω)) g _ ω)
  have hnone : wt ℓ (fun _ => none) = fun _ _ => (Fintype.card (Blk × Salt) : ℝ≥0∞)⁻¹ :=
    funext fun _ => funext fun z => wt_none rfl z
  have hstart : avg (fun ω : I → Blk × Salt => TGame.pr (E fun i => ℓ i (ω i)) g (τ fun i => ℓ i (ω i)) ω) =
      Exp p' g' ℓ τ' (fun _ => none) := by
    rw [hexp, hnone, avg_eq_W]
    rfl
  have hend : ∀ q : I → Blk × Salt, Exp p' g' ℓ τ' (fillM ℓ (fun _ => none) q) =
      ∑ r, W (wtR (postK ℓ q)) r * prob (E fun j => ℓ j (q j)) G (toStrat hgG (τ fun j => ℓ j (q j)) r) := by
    intro q
    rw [hexp]
    exact lazy_mix (E fun j => ℓ j (q j)) (sum_postK ℓ q) hgG hg (τ fun j => ℓ j (q j))
  have hcard : (univ.filter fun j : I => (fun _ : I => (none : Option D)) j = none).card = Fintype.card I := by
    simp
  calc avg (fun ω : I → Blk × Salt => TGame.pr (E fun i => ℓ i (ω i)) g (τ fun i => ℓ i (ω i)) ω)
      = Exp p' g' ℓ τ' (fun _ => none) := hstart
    _ ≤ avg (fun q : I → Blk × Salt => Exp p' g' ℓ τ' (fillM ℓ (fun _ => none) q)) + Fintype.card I * δ :=
        fill_le p' g' ℓ τ' (fun k b Γ hΓ => hide_of_twoVector dig My cs H hT k b Γ hΓ) _ _ hcard
    _ = (mixOf hgG ℓ τ).pr E + Fintype.card I * δ := by
        rw [mixOf_pr]
        simp only [hend]

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
  have hT : ∀ κ : Key E₂, Hm96TwoVector (CROnly.hankel (n := 512) (l := 1536) κ) cs
      (ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
        ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) :=
    fun κ => twoVector_of_hiding _ _ (CROnly.hm96Hiding_gap _ _)
  have hMk := fun κ => treeCoins_le_mixOf hgG hg dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) (hT κ) (τ κ)
  have h257 := CROnly.avg_ofReal_le _ (fun κ => CROnly.hidingGap_nonneg _ _) (CROnly.hm96_sha512_sum hE cs hcs)
  have hδ : avg (fun κ : Key E₂ => ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
      ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) ≤
      ENNReal.ofReal (1 / 2 ^ 256) := by
    rw [avg_add]
    calc _ ≤ ENNReal.ofReal (1 / 2 ^ 257) + ENNReal.ofReal (1 / 2 ^ 257) := add_le_add h257 h257
      _ = ENNReal.ofReal (1 / 2 ^ 256) := by
          rw [← ENNReal.ofReal_add (by norm_num) (by norm_num)]
          norm_num
  rw [Mix.avgOver_pr]
  calc _ ≤ avg (fun κ : Key E₂ =>
          (mixOf hgG (leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ)) (τ κ)).pr (fun v => E (κ, v)) +
            Fintype.card I * (ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
              ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs))) :=
        avg_mono fun κ => hMk κ fun v => E (κ, v)
    _ = avg (fun κ =>
          (mixOf hgG (leaf dig (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ)) (τ κ)).pr fun v => E (κ, v)) +
          Fintype.card I * avg (fun κ : Key E₂ =>
            ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs) +
              ENNReal.ofReal (CROnly.hidingGap (CROnly.hankel (n := 512) (l := 1536) κ) cs)) := by
        rw [avg_add, avg_const_mul]
    _ ≤ _ := add_le_add le_rfl (mul_le_mul_right hδ _)

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
  have hMω := fun ω : L.Ω => treeCoins_le_avgMixOf hE hgG hg (dig (τ.2 ω).1) (cs (τ.2 ω).1) (hcs _)
    (H (τ.2 ω).1) (τ.2 ω).2
  refine ⟨Mix.piOver L out τ.1 fun ω => Mix.avgOver fun κ =>
      mixOf hgG (leaf (dig (τ.2 ω).1) (CROnly.hankel (n := 512) (l := 1536) κ) (cs (τ.2 ω).1) (H (τ.2 ω).1 κ))
        ((τ.2 ω).2 κ),
    fun x ω => ((x ω).1, fun j => leaf (dig (τ.2 ω).1) (CROnly.hankel (n := 512) (l := 1536) (x ω).1) (cs (τ.2 ω).1)
      (H (τ.2 ω).1 (x ω).1) j ((x ω).2.1 j), (x ω).2.2), fun _ => rfl, fun E => ?_⟩
  rw [Mix.piOver_pr]
  calc prTree L g out dig cs H τ E
      ≤ avg (fun ω => (Mix.avgOver fun κ =>
          mixOf hgG (leaf (dig (τ.2 ω).1) (CROnly.hankel (n := 512) (l := 1536) κ) (cs (τ.2 ω).1) (H (τ.2 ω).1 κ))
            ((τ.2 ω).2 κ)).pr (fun _ o => E (τ.1, L.draw ω, out (L.draw ω) τ.1 o)) +
          Fintype.card I * ENNReal.ofReal (1 / 2 ^ 256)) :=
        avg_mono fun ω => hMω ω fun _ o => E (τ.1, L.draw ω, out (L.draw ω) τ.1 o)
    _ = _ := by rw [avg_add, avg_const]

end FlockSoundness.Discharge.ZkTree
