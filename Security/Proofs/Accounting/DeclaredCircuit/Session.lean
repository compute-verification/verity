import Mathlib
import Proofs.Accounting.DeclaredCircuit
import Proofs.Accounting.DeclaredCircuit.Slack

/-!
# The session's passes, each priced at its prefix

The pieces of `Accounting.SecurityProofs.DeclaredCircuitRequirement` that price the passes of the stacked session. The
codec and the append coins and pass `j`'s own coins are uniform and independent of the prefix before the pass, so an
event of the three is bounded at every prefix (`pr_at_pass_le`). At every store some candidate holds a correct tag of a
tile it never queried with probability at most `G·N/2^h` (`pr_guess_family_le`). At signs of rank 12, some candidate
made `k + 1` correct digest queries behind unobtained positions with probability at most
`2^a·(q_tot/2^96)^(k+1) + δ`: key binding at the steered layout, with the union over the `2^a` summaries inside the
uniform blocks (`pr_badK_pass_le`). With the rank tail once a window, a session in which some window has rank-poor signs
or some pass one of the two events costs `n·ε + n·P·G·N/2^512 + n·P·(2^a·(q_tot/2^96)^(k+1) + δ)` (`pr_session_le`).
-/

namespace Accounting.RateSketch

open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_exists_le pr_le_of_card pr_mono pr_or_le)

/-- The oracle family of a pass's `G` grinding candidates is finite. -/
local instance instFintypeOracleFamSession (N ρ h ks G : ℕ) :
    Fintype (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := by
  letI : Fintype (Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  exact inferInstance

/-- An event of a pair bounded at every first coordinate is bounded. -/
theorem pr_le_of_fiber_fst {X Y : Type} [Fintype X] [Fintype Y] (P : X × Y → Prop) (ε : ℝ≥0∞)
    (h : ∀ x, pr (fun y => P (x, y)) ≤ ε) : pr P ≤ ε := by
  rw [Rate.pr_equiv (Equiv.prodComm X Y)]
  exact Rate.pr_le_of_fiber _ _ fun x => (pr_mono fun y hy => by simpa using hy).trans (h x)

/-- An event of one member of a uniform family has that member's probability. -/
theorem pr_family_coord_le {β : Type} [Fintype β] {G : ℕ} (g : Fin G) (Q : β → Prop) {ε : ℝ≥0∞}
    (hQ : pr Q ≤ ε) : pr (fun Os : Fin G → β => Q (Os g)) ≤ ε := by
  classical
  rw [Rate.pr_equiv (Equiv.funSplitAt g β)]
  refine Rate.pr_le_of_fiber _ _ fun rest => (pr_mono fun b hb => ?_).trans hQ
  have e1 : (Equiv.funSplitAt g β).symm (b, rest) g = b := by simp
  simp only [e1] at hb
  exact hb

/-- The blocks of a pass's tiles and its own coins, read as the pass's data and its candidates' oracles: an event of
the data and the oracles bounded on uniform data at every oracle family is bounded on uniform blocks. -/
theorem pr_blocks_le {N m ρ h ks G : ℕ}
    (B : Data N m ρ Sector → (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) → Prop) {p : ℝ≥0∞}
    (hB : ∀ Os, pr (fun D => B D Os) ≤ p) :
    pr (fun y : (Fin N → Fin m → Sector) × SketchPassCoins N ρ h ks G =>
      B (Data.ofParts (fun u => y.1 u.1 u.2) y.2.1) y.2.2) ≤ p := by
  classical
  let eD : Data N m ρ Sector ≃ (Fin N → Fin m → Sector) × SketchStream ρ :=
    (Equiv.sumArrowEquivProdArrow _ _ _).trans ((Equiv.curry _ _ _).prodCongr (Equiv.refl _))
  let e : (Fin N → Fin m → Sector) × SketchPassCoins N ρ h ks G ≃
      Data N m ρ Sector × (Fin G → Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) :=
    (Equiv.prodAssoc _ _ _).symm.trans (eD.symm.prodCongr (Equiv.refl _))
  rw [Rate.pr_equiv e]
  refine Rate.pr_le_of_fiber _ _ fun Os => (pr_mono fun D hD => ?_).trans (hB Os)
  have hdata : Data.ofParts (fun u => (e.symm (D, Os)).1 u.1 u.2) (e.symm (D, Os)).2.1 = D := by
    funext p
    rcases p with p | p <;> rfl
  change B (Data.ofParts (fun u => (e.symm (D, Os)).1 u.1 u.2) (e.symm (D, Os)).2.1) (e.symm (D, Os)).2.2 at hD
  rw [hdata] at hD
  exact hD

namespace Device

variable {N m ρ h S L q ks : ℕ}

/-- A correct tag of a tile the run never queried has probability at most `1/2^h` over the oracle, at every data. -/
theorem pr_guess_at_le (A : Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (D : Data N m ρ Sector) (t : Fin N) :
    pr (fun O : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
      t ∈ A.correct sketch σ O D ∧ t ∉ queried A sketch σ O D) ≤ (1 : ℝ≥0∞) / 2 ^ h := by
  classical
  have hbound := pr_le_of_card
    (fun O : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32)) =>
      t ∈ A.correct sketch σ O D ∧ t ∉ queried A sketch σ O D)
    (2 ^ h) 1 (by positivity) (by simpa [Device.correct, and_comm] using card_guess_le A sketch σ D t)
  push_cast at hbound
  simpa using hbound

end Device

section Passes

variable {Y : Type} [Fintype Y]

/-- **A guessed tag at every store.** Over a pass's stream and its candidates' oracles, beside coins `y` that the
devices and the store may read, some candidate holds a correct tag of a tile it never queried with probability at most
`G·N/2^h`. -/
theorem pr_guess_family_le {N m ρ h S L q ks G : ℕ}
    (dv : Y → Fin G → Device N m ρ h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (st : Y → SketchStore N m) :
    pr (fun a : Y × SketchPassCoins N ρ h ks G => ∃ g t,
      t ∈ (dv a.1 g).correct sketch σ (a.2.2 g) (Data.ofParts (st a.1) a.2.1) ∧
        t ∉ queried (dv a.1 g) sketch σ (a.2.2 g) (Data.ofParts (st a.1) a.2.1)) ≤
      (G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ h) := by
  classical
  refine (pr_exists_le _).trans ?_
  have hg : ∀ g : Fin G, pr (fun a : Y × SketchPassCoins N ρ h ks G => ∃ t,
      t ∈ (dv a.1 g).correct sketch σ (a.2.2 g) (Data.ofParts (st a.1) a.2.1) ∧
        t ∉ queried (dv a.1 g) sketch σ (a.2.2 g) (Data.ofParts (st a.1) a.2.1)) ≤
      (N : ℝ≥0∞) / 2 ^ h := by
    intro g
    refine (pr_exists_le _).trans ?_
    have ht : ∀ t : Fin N, pr (fun a : Y × SketchPassCoins N ρ h ks G =>
        t ∈ (dv a.1 g).correct sketch σ (a.2.2 g) (Data.ofParts (st a.1) a.2.1) ∧
          t ∉ queried (dv a.1 g) sketch σ (a.2.2 g) (Data.ofParts (st a.1) a.2.1)) ≤
        (1 : ℝ≥0∞) / 2 ^ h :=
      fun t => pr_le_of_fiber_fst _ _ fun y => pr_le_of_fiber_fst _ _ fun R =>
        pr_family_coord_le g (fun O => t ∈ (dv y g).correct sketch σ O (Data.ofParts (st y) R) ∧
          t ∉ queried (dv y g) sketch σ O (Data.ofParts (st y) R)) (Device.pr_guess_at_le (dv y g) σ _ t)
    refine (Finset.sum_le_sum fun t _ => ht t).trans (le_of_eq ?_)
    simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, div_eq_mul_inv]
  refine (Finset.sum_le_sum fun g _ => hg g).trans (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- **A pass's coins are fresh at its prefix.** The coins `y` are drawn beside the session and pass `j` of window `w`
draws its own coins after the prefix before it, so an event of the prefix, `y` and the pass's coins that is bounded at
every prefix is bounded. -/
theorem pr_at_pass_le {n P N m ρ h ks G d : ℕ} (w : Fin n) (j : Fin P)
    (F : SketchPrefix n P N m ρ h ks G d → Y → SketchPassCoins N ρ h ks G → Prop) (p : ℝ≥0∞)
    (hF : ∀ c, pr (fun a : Y × SketchPassCoins N ρ h ks G => F c a.1 a.2) ≤ p) :
    pr (fun x : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) =>
      F (SketchPrefix.of x.2 w j) x.1 ((x.2 w).2.1 j)) ≤ p := by
  classical
  let e : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) ≃
      (Y × SketchPassCoins N ρ h ks G) × (({v : Fin n // v ≠ w} → SketchWindowCoins P N m ρ h ks G d) ×
        (SketchSigns ks m × ({i : Fin P // i ≠ j} → SketchPassCoins N ρ h ks G) × (Fin d → Fin P × Fin N))) :=
    { toFun := fun x => ((x.1, (Equiv.funSplitAt j _ (Equiv.funSplitAt w _ x.2).1.2.1).1),
        ((Equiv.funSplitAt w _ x.2).2, (Equiv.funSplitAt w _ x.2).1.1,
          (Equiv.funSplitAt j _ (Equiv.funSplitAt w _ x.2).1.2.1).2, (Equiv.funSplitAt w _ x.2).1.2.2))
      invFun := fun z => (z.1.1, (Equiv.funSplitAt w _).symm
        ((z.2.2.1, (Equiv.funSplitAt j _).symm (z.1.2, z.2.2.2.1), z.2.2.2.2), z.2.1))
      left_inv := fun x => by simp only [Prod.mk.eta, Equiv.symm_apply_apply]
      right_inv := fun z => by simp only [Equiv.apply_symm_apply, Prod.mk.eta] }
  rw [Rate.pr_equiv e]
  refine Rate.pr_le_of_fiber _ _ fun rest => ?_
  obtain ⟨ws, σ, rj, pd⟩ := rest
  refine (pr_mono fun a ha => ?_).trans (hF ⟨w, fun v => ws ⟨v.1, v.2.ne⟩, σ, j, fun i => rj ⟨i.1, i.2.ne⟩⟩)
  have hself : (Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm
      ((σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd), ws) w =
        (σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd) := by
    simp
  have hwin : (fun v : {v : Fin n // v < w} => (Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm
      ((σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd), ws) v.1) =
        fun v => ws ⟨v.1, v.2.ne⟩ := by
    funext v
    simp [v.2.ne]
  have hpass : (fun i : {i : Fin P // i < j} =>
      ((σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd) :
        SketchWindowCoins P N m ρ h ks G d).2.1 i.1) = fun i => rj ⟨i.1, i.2.ne⟩ := by
    funext i
    simp [i.2.ne]
  have hj : (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj) j = a.2 := by simp
  have h1 : SketchPrefix.of ((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm
      ((σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd), ws)) w j =
        ⟨w, fun v => ws ⟨v.1, v.2.ne⟩, σ, j, fun i => rj ⟨i.1, i.2.ne⟩⟩ := by
    unfold SketchPrefix.of
    rw [hwin, hself, hpass]
  have h2 : ((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm
      ((σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd), ws) w).2.1 j = a.2 := by
    rw [hself]
    exact hj
  change F (SketchPrefix.of ((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm
      ((σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd), ws)) w j) a.1
    (((Equiv.funSplitAt w (SketchWindowCoins P N m ρ h ks G d)).symm
      ((σ, (Equiv.funSplitAt j (SketchPassCoins N ρ h ks G)).symm (a.2, rj), pd), ws) w).2.1 j) at ha
  rw [h1, h2] at ha
  exact ha

end Passes

section Session

variable {n P N ρ S L q G T M a : ℕ} {Ω Ca Ch : Type}

/-- Candidate `g` of pass `c.j` of window `c.w`, at the prefix `c` before the pass and the codec and the append coins
`y`: the device of the session of the summary `steer c y` the pass is told. -/
abbrev passDev (A : SketchSummarySession a n P N 128 ρ 512 S L q 16 G 2773)
    (steer : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a)
    (c : SketchPrefix n P N 128 ρ 512 16 G 2773) (y : Ω × (Fin T → Ca)) (g : Fin G) :
    Device N 128 ρ 512 S L q 256 Sector (SketchSigns 16 128) (Fin 16 → ZMod (2 ^ 32)) :=
  ((A.session (steer c y)).window c.w c.windows).device c.j c.passes g

/-- **A guessed tag** at pass `c.j` of window `c.w`, on its store and its coins `pc`: some candidate holds a correct
tag of a tile it never queried. -/
abbrev PassGuess (A : SketchSummarySession a n P N 128 ρ 512 S L q 16 G 2773)
    (steer : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a)
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → Sector))
    (choose : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (b : Fin T) → ({b' : Fin T // b' < b} → Ca) → Ch)
    (layout : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (c : SketchPrefix n P N 128 ρ 512 16 G 2773) (y : Ω × (Fin T → Ca)) (pc : SketchPassCoins N ρ 512 16 G) :
    Prop :=
  ∃ g t, t ∈ (passDev A steer c y g).correct sketch c.signs (pc.2 g)
      (Data.ofParts (steeredStore enc choose layout c y) pc.1) ∧
    t ∉ queried (passDev A steer c y g) sketch c.signs (pc.2 g) (Data.ofParts (steeredStore enc choose layout c y) pc.1)

/-- **`k + 1` correct digests behind unobtained positions** at pass `c.j` of window `c.w`, on its store and its coins
`pc`, at signs of rank 12. -/
abbrev PassBadK (k : ℕ) (A : SketchSummarySession a n P N 128 ρ 512 S L q 16 G 2773)
    (steer : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a)
    (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → Sector))
    (choose : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (b : Fin T) → ({b' : Fin T // b' < b} → Ca) → Ch)
    (layout : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (c : SketchPrefix n P N 128 ρ 512 16 G 2773) (y : Ω × (Fin T → Ca)) (pc : SketchPassCoins N ρ 512 16 G) :
    Prop :=
  RankGood 12 c.signs ∧ ∃ g, ∃ ι : Fin (k + 1) → Fin q,
    BadK (passDev A steer c y g) sketch c.signs (pc.2 g) (Data.ofParts (steeredStore enc choose layout c y) pc.1)
      fun i => (ι i : ℕ)

/-- **The slack's price at a prefix, on the steered layout.** At the prefix `c`, over the codec, the append coins and
the pass's own coins, some candidate of the pass made `k + 1` correct digest queries behind unobtained positions, at
signs of rank 12, with probability at most `2^a·(q_tot/2^96)^(k+1) + δ`: key binding at the layout `slot c` of the
summary, the union over the `2^a` summaries inside the uniform blocks (`KeyBinding.pr_steered_le`), and on uniform
blocks `Device.pr_badK_family_le` at each summary's devices. -/
theorem pr_badK_pass_le [Fintype Ω] [Fintype Ca] {k qtot : ℕ} {δ : ℝ≥0∞}
    {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → Sector)}
    (hKB : Assumptions.KeyBinding T M δ enc)
    (choose : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (b : Fin T) → ({b' : Fin T // b' < b} → Ca) → Ch)
    (layout : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (steer : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a)
    (hslot : Assumptions.SteeredLayout layout steer)
    (A : SketchSummarySession a n P N 128 ρ 512 S L q 16 G 2773)
    (hq : ∀ s w η j pre, Assumptions.SketchHashBudget (((A.session s).window w η).device j pre) qtot)
    (c : SketchPrefix n P N 128 ρ 512 16 G 2773) :
    pr (fun z : (Ω × (Fin T → Ca)) × SketchPassCoins N ρ 512 16 G =>
      PassBadK k A steer enc choose layout c z.1 z.2) ≤ 2 ^ a * ((qtot : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ := by
  classical
  obtain ⟨slot, hslot⟩ := hslot
  have hE : ∀ y, steeredStore enc choose layout c y =
      fun u => appendCoinBlocks enc (choose c) (slot c (steer c y)) y u.1 u.2 := fun y => by
    unfold steeredStore
    rw [hslot c y]
  refine (pr_mono fun z hz => ?_).trans (hKB.pr_steered_le (choose c) (slot c) (steer c)
    (C := SketchPassCoins N ρ 512 16 G)
    (fun s Eb pc => RankGood 12 c.signs ∧ ∃ g, ∃ ι : Fin (k + 1) → Fin q,
      BadK (((A.session s).window c.w c.windows).device c.j c.passes g) sketch c.signs (pc.2 g)
        (Data.ofParts (fun u => Eb u.1 u.2) pc.1) fun i => (ι i : ℕ)) fun s => ?_)
  · obtain ⟨hr, g, ι, hb⟩ := hz
    refine ⟨hr, g, ι, ?_⟩
    rw [hE] at hb
    exact hb
  · by_cases hr : RankGood 12 c.signs
    · have h2 := pr_blocks_le (N := N) (m := 128) (ρ := ρ) (h := 512) (ks := 16) (G := G)
        (fun D Os => ∃ g, ∃ ι : Fin (k + 1) → Fin q,
          BadK (((A.session s).window c.w c.windows).device c.j c.passes g) sketch c.signs (Os g) D
            fun i => (ι i : ℕ))
        fun Os => Device.pr_badK_family_le (k := k)
          (fun g => ((A.session s).window c.w c.windows).device c.j c.passes g) c.signs Os 12 qtot (by norm_num) hr
          fun D => hq s c.w c.windows c.j c.passes c.signs Os D
      rw [show 8 * 12 = 96 from rfl] at h2
      refine le_trans (pr_mono fun y hy => ?_) h2
      exact hy.2
    · calc _ ≤ pr (fun _ : (Fin N → Fin 128 → Sector) × SketchPassCoins N ρ 512 16 G => False) :=
            pr_mono fun _ hy => hr hy.1
        _ ≤ _ := by
          rw [Rate.Pooled.pr_false]
          exact bot_le

/-- **The session's passes priced.** Some window has signs of rank below 12, or some pass a guessed tag or `k + 1`
correct digests behind unobtained positions, with probability at most
`n·ε + n·P·G·N/2^512 + n·P·(2^a·(q_tot/2^96)^(k+1) + δ)`: the rank tail once a window, and at each pass the guess at
every store and the slack's price on the steered layout, each at the prefix before the pass. -/
theorem pr_session_le [Fintype Ω] [Fintype Ca] {k qtot : ℕ} {ε δ : ℝ≥0∞}
    {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → Sector)}
    (hRank : Assumptions.SketchRank 16 128 12 ε) (hKB : Assumptions.KeyBinding T M δ enc)
    (choose : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (b : Fin T) → ({b' : Fin T // b' < b} → Ca) → Ch)
    (layout : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M)
    (steer : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a)
    (hslot : Assumptions.SteeredLayout layout steer)
    (A : SketchSummarySession a n P N 128 ρ 512 S L q 16 G 2773)
    (hq : ∀ s w η j pre, Assumptions.SketchHashBudget (((A.session s).window w η).device j pre) qtot) :
    pr (fun x : StackedCoins Ω Ca T n P N ρ G => ∃ w, ¬ RankGood 12 (x.2 w).1 ∨
      ∃ j, PassGuess A steer enc choose layout (SketchPrefix.of x.2 w j) x.1 ((x.2 w).2.1 j) ∨
        PassBadK k A steer enc choose layout (SketchPrefix.of x.2 w j) x.1 ((x.2 w).2.1 j)) ≤
      (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * (2 ^ a * ((qtot : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ) := by
  classical
  have hw : ∀ w : Fin n, pr (fun x : StackedCoins Ω Ca T n P N ρ G => ¬ RankGood 12 (x.2 w).1 ∨
      ∃ j, PassGuess A steer enc choose layout (SketchPrefix.of x.2 w j) x.1 ((x.2 w).2.1 j) ∨
        PassBadK k A steer enc choose layout (SketchPrefix.of x.2 w j) x.1 ((x.2 w).2.1 j)) ≤
      ε + ∑ _j : Fin P, ((G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (2 ^ a * ((qtot : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ)) :=
    fun w => (pr_or_le _ _).trans (add_le_add (SketchDeployedSession.pr_rank_le 12 ε hRank w)
      ((pr_exists_le _).trans (Finset.sum_le_sum fun j _ => (pr_or_le _ _).trans (add_le_add
        (pr_at_pass_le w j (PassGuess A steer enc choose layout) _ fun c =>
          pr_guess_family_le (fun y g => passDev A steer c y g) c.signs
            fun y => steeredStore enc choose layout c y)
        (pr_at_pass_le w j (PassBadK k A steer enc choose layout) _ fun c =>
          pr_badK_pass_le hKB choose layout steer hslot A hq c)))))
  refine (pr_exists_le _).trans ((Finset.sum_le_sum fun w _ => hw w).trans (le_of_eq ?_))
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

end Session

end Accounting.RateSketch
