import Models.Accounting.Assumptions.Compose

/-!
# RT2-INT-1's witness: the guess claim is false on a tile whose checked values are honest whenever it is skipped whole (scratch, not for `main`)

Checked against #2677's tree (`cursor/int-tie-ea24` @ f3569b633b, which merges Compose's b2899a6bf0); `main` has no
`Definitions.Accounting.Compose` yet.

`CircuitWork` and `CircuitWorkPeriod` take `GuessLuckAt (Sc.guess k)` (`p0 = ½` on the live line) for every tile of a
credited kind (`PerTile`), on the tile's whole game `tileGame mode right same .whole t`, whose round passes when the
tile's checked values are the honest ones (`same`). A dead tile of `PouwH1Tile_v2` (#2639) has the canonical empty
receipt at every coin, so a cheater who skips it whole and commits that receipt has `same` whenever it skips:
`guess_false_of_same` shows the claim then fails at every `p0 < 1` as soon as the skip has positive probability.
-/

namespace RtInt2Dead

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

/-- **The guess claim fails on a tile that passes whenever it is skipped whole**: if the checked values are the
honest ones on every coin where tile `t` is skipped whole, and that skip has positive probability, then
`GuessLuckAt p0` of the tile's whole game is false at every `p0 < 1`. -/
theorem guess_false_of_same {C V : Type} [Fintype C] (mode : C → V → Mode) (right same : C → V → Prop) (t : V)
    (hs : ∀ c, mode c t = .whole → same c t) (hpos : 0 < prob (fun c : C => mode c t = .whole))
    {p0 : ℝ} (hp : p0 < 1) :
    ¬ Accounting.Assumptions.Compose.GuessLuckAt p0 (tileGame mode right same .whole t) := by
  intro h
  have h2 : prob (fun c : C => same c t ∧ (0 : ℝ) < if mode c t = Mode.whole then 1 else 0) ≤
      p0 * prob (fun c : C => (0 : ℝ) < if mode c t = Mode.whole then 1 else 0) := by
    simp only [Accounting.Assumptions.Compose.GuessLuckAt, tileGame, ReceiptGame.game, RoundGame.LuckAt] at h
    exact h.2.1
  have e : ∀ c : C, ((0 : ℝ) < if mode c t = Mode.whole then 1 else 0) ↔ mode c t = Mode.whole := by
    intro c
    split_ifs with hc <;> simp [hc]
  have e1 : prob (fun c : C => same c t ∧ (0 : ℝ) < if mode c t = Mode.whole then 1 else 0) =
      prob (fun c : C => mode c t = Mode.whole) := by
    congr 1
    funext c
    apply propext
    rw [e c]
    exact ⟨fun h => h.2, fun h => ⟨hs c h, h⟩⟩
  have e2 : prob (fun c : C => (0 : ℝ) < if mode c t = Mode.whole then 1 else 0) =
      prob (fun c : C => mode c t = Mode.whole) := by
    congr 1
    funext c
    exact propext (e c)
  rw [e1, e2] at h2
  nlinarith [mul_pos (sub_pos.2 hp) hpos]

/-- **At the live line's `p0 = ½`**, for a tile skipped whole on every coin and passing on every coin (a dead
`PouwH1Tile_v2` tile skipped with the canonical empty receipt committed). -/
theorem guess_false_dead {C V : Type} [Fintype C] [Nonempty C] (mode : C → V → Mode) (right same : C → V → Prop)
    (t : V) (hw : ∀ c, mode c t = .whole) (hs : ∀ c, same c t) :
    ¬ Accounting.Assumptions.Compose.GuessLuckAt (1 / 2) (tileGame mode right same .whole t) := by
  have hpos : 0 < prob (fun c : C => mode c t = .whole) := by
    unfold prob
    obtain ⟨c0⟩ := ‹Nonempty C›
    have hc : (0 : ℝ) < Fintype.card C := by exact_mod_cast Fintype.card_pos
    refine div_pos ?_ hc
    first
      | exact_mod_cast Finset.card_pos.2 ⟨c0, by simp [hw c0]⟩
      | exact Nat.cast_pos.2 (Finset.card_pos.2 ⟨c0, by simp [hw c0]⟩)
  exact guess_false_of_same mode right same t (fun c _ => hs c) hpos (by norm_num)

end RtInt2Dead

#print axioms RtInt2Dead.guess_false_of_same
#print axioms RtInt2Dead.guess_false_dead
