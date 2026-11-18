module

public import Definitions.Poub.Carrier

/-!
# Refresh's cut: every payload a pass reads was committed before every key it reads was drawn

Under refresh every tile of a pass is at an append `a` that joined within `L` windows of the pass
(`LifetimeCappedAll`: `joined a ≤ w < joined a + L`), and its payload is seasoned (`Seasoned`: the origin `seen a` is
the registration, or `joined (seen a) + L ≤ w`). With the appends joining in order (`JoinsInOrder`), two tiles `i`,
`i'` of one pass at appends `a`, `a'` have `seen a ≤ a'` (`cut`), strictly unless `seen a = 0` (`cut_strict`):
`joined (seen a) + L ≤ w < joined a' + L` gives `joined (seen a) < joined a'`, which a monotone `joined` turns into
`seen a < a'`. With each append's choices reading only the coins before its origin (`ChoosesBefore`) at origins fixed
before every coin (`OriginsFixed`), no tile's choices read the coin of any tile of the same pass (`cut_choices`).

Exactly one of the two inequalities is strict, and either must be: with slack in the probation
(`joined (seen a) + L ≤ w + 1`, `not_seen_le_of_slack_probation`) or in the cap (`w ≤ joined a' + L`,
`not_seen_le_of_slack_cap`), two appends joining one window break the order. A fresh append's own tiles are never read
under refresh (`fresh_not_read`): `seen a = a ≠ 0` puts `joined a + L ≤ w < joined a + L`.

`cut`, `cut_strict` and `fresh_not_read` read the origins at the coins and need nothing of how they were chosen.
`cut_choices` needs them fixed before every coin: origins read only from the coins of the appends before each append
let the prover choose, after a tile's key coin, which committed payload another tile of the pass carries
(`prefix_origin_reads_key`).
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Refresh

open ResourceAccounting.RateSketch

/-- The order arithmetic: a seasoned origin `s` precedes an append `a'` capped at the same window, or is the
registration. -/
theorem seen_eq_zero_or_lt_of_order {T L w : ℕ} {joined : Fin T → ℕ} {s a' : Fin T} (hord : Monotone joined)
    (hs : (s : ℕ) = 0 ∨ joined s + L ≤ w) (hcap : w < joined a' + L) : (s : ℕ) = 0 ∨ s < a' := by
  rcases hs with h | h
  · exact Or.inl h
  · refine Or.inr (lt_of_not_ge fun hle => ?_)
    have := hord hle
    omega

/-- The order arithmetic, in the cut's form: `s ≤ a'`. -/
theorem seen_le_of_order {T L w : ℕ} {joined : Fin T → ℕ} {s a' : Fin T} (hord : Monotone joined)
    (hs : (s : ℕ) = 0 ∨ joined s + L ≤ w) (hcap : w < joined a' + L) : s ≤ a' := by
  rcases seen_eq_zero_or_lt_of_order hord hs hcap with h | h
  · exact Fin.le_def.mpr (by omega)
  · exact h.le

/-- **Slack in the probation breaks the order**: with `joined (seen a) + L ≤ w + 1` in place of `≤ w`, three appends
joining window `0` at `L = 1` give `seen a = 2` after `a' = 1` in a pass of window `0`. -/
theorem not_seen_le_of_slack_probation :
    ¬ ∀ (joined : Fin 3 → ℕ) (s a' : Fin 3) (L w : ℕ), Monotone joined →
      ((s : ℕ) = 0 ∨ joined s + L ≤ w + 1) → w < joined a' + L → s ≤ a' := by
  intro h
  have := h (fun _ => 0) 2 1 1 0 (fun _ _ _ => le_rfl) (Or.inr le_rfl) Nat.one_pos
  exact absurd this (by decide)

/-- **Slack in the cap breaks the order**: with `w ≤ joined a' + L` in place of `<`, three appends joining window `0`
at `L = 1` give `seen a = 2` after `a' = 1` in a pass of window `1`. -/
theorem not_seen_le_of_slack_cap :
    ¬ ∀ (joined : Fin 3 → ℕ) (s a' : Fin 3) (L w : ℕ), Monotone joined →
      ((s : ℕ) = 0 ∨ joined s + L ≤ w) → w ≤ joined a' + L → s ≤ a' := by
  intro h
  have := h (fun _ => 0) 2 1 1 1 (fun _ _ _ => le_rfl) (Or.inr le_rfl) le_rfl
  exact absurd this (by decide)

/-- **The cut, strict form**: under `LifetimeCappedAll` and `Seasoned`, with the appends joining in order, the origin
of every tile `i` of a pass is the registration or an append strictly before that of every tile `i'` of the pass. -/
theorem cut_strict {n P N m ρ h ks G d T M L : ℕ} {Y : Type}
    {layout : SketchPrefix n P N m ρ h ks G d → Y → Fin N ↪ Fin T × Fin M}
    {joined : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → ℕ}
    {seen : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → Fin T}
    (hord : JoinsInOrder joined) (hcap : LifetimeCappedAll layout joined L) (hs : Seasoned layout joined seen L)
    (x : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d)) (w : Fin n) (j : Fin P) (i i' : Fin N) :
    (seen x (layout (SketchPrefix.of x.2 w j) x.1 i).1 : ℕ) = 0 ∨
      seen x (layout (SketchPrefix.of x.2 w j) x.1 i).1 < (layout (SketchPrefix.of x.2 w j) x.1 i').1 :=
  seen_eq_zero_or_lt_of_order (hord x) (hs x w j i) (hcap x w j i').2

/-- **The cut**: under `LifetimeCappedAll` and `Seasoned`, with the appends joining in order, every two tiles `i`,
`i'` of a pass have `seen (append of i) ≤ append of i'`: the payload of each was committed no later than the append
whose coin keys the other. -/
theorem cut {n P N m ρ h ks G d T M L : ℕ} {Y : Type}
    {layout : SketchPrefix n P N m ρ h ks G d → Y → Fin N ↪ Fin T × Fin M}
    {joined : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → ℕ}
    {seen : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → Fin T}
    (hord : JoinsInOrder joined) (hcap : LifetimeCappedAll layout joined L) (hs : Seasoned layout joined seen L)
    (x : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d)) (w : Fin n) (j : Fin P) (i i' : Fin N) :
    seen x (layout (SketchPrefix.of x.2 w j) x.1 i).1 ≤ (layout (SketchPrefix.of x.2 w j) x.1 i').1 :=
  seen_le_of_order (hord x) (hs x w j i) (hcap x w j i').2

/-- **No tile's choices read the coin of a tile of the same pass**: under the cut's conditions, with origins fixed
before every coin at `s` and each append's choices reading only the coins before its origin, the choices of tile `i`'s
append are the same at any two coin assignments that differ only at the append of tile `i'`. -/
theorem cut_choices {n P N m ρ h ks G d T M L : ℕ} {Y Pfx Ω Ca Ch : Type}
    {layout : SketchPrefix n P N m ρ h ks G d → Y → Fin N ↪ Fin T × Fin M}
    {joined : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → ℕ}
    {seen : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → Fin T} {s : Fin T → Fin T}
    {choose : Pfx → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch}
    (hord : JoinsInOrder joined) (hcap : LifetimeCappedAll layout joined L) (hs : Seasoned layout joined seen L)
    (hfix : OriginsFixed seen s) (hc : ChoosesBefore choose s)
    (x : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d)) (w : Fin n) (j : Fin P) (i i' : Fin N) (c : Pfx) (ω : Ω)
    (κ κ' : {b : Fin T // b < (layout (SketchPrefix.of x.2 w j) x.1 i).1} → Ca)
    (hκ : ∀ b, b.1 ≠ (layout (SketchPrefix.of x.2 w j) x.1 i').1 → κ b = κ' b) :
    choose c ω _ κ = choose c ω _ κ' := by
  refine hc c ω _ κ κ' fun b hb => hκ b fun heq => ?_
  have hle := cut hord hcap hs x w j i i'
  rw [hfix x, ← heq] at hle
  exact absurd hb (not_lt.mpr hle)

/-- **A fresh append's own tiles are never read under refresh**: a tile of a pass whose append declares itself as its
origin is at the registration. -/
theorem fresh_not_read {n P N m ρ h ks G d T M L : ℕ} {Y : Type}
    {layout : SketchPrefix n P N m ρ h ks G d → Y → Fin N ↪ Fin T × Fin M}
    {joined : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → ℕ}
    {seen : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d) → Fin T → Fin T}
    (hcap : LifetimeCappedAll layout joined L) (hs : Seasoned layout joined seen L)
    (x : Y × (Fin n → SketchWindowCoins P N m ρ h ks G d)) (w : Fin n) (j : Fin P) (i : Fin N)
    (hfresh : seen x (layout (SketchPrefix.of x.2 w j) x.1 i).1 = (layout (SketchPrefix.of x.2 w j) x.1 i).1) :
    ((layout (SketchPrefix.of x.2 w j) x.1 i).1 : ℕ) = 0 := by
  have hc := (hcap x w j i).2
  rcases hs x w j i with h | h
  · rwa [hfresh] at h
  · rw [hfresh] at h
    omega

/-- **Origins read from the coins before the append let a payload read a key of its pass**: `cut_choices` needs the
origins fixed before every coin, not only read from the coins of the appends before each append. Appends `1` and `2`
are fresh, joining window `0`, with payloads `false` and `true`; appends `3` and `4` are re-keys joining window `1`,
`3` of append `1` and `4` of append `1` or `2` as append `3`'s coin falls. At `L = 1` a pass of window `1` reading both
re-keys meets every condition of the cut, append `4`'s origin reads only the coins before it, and the cut's order
holds, yet the payload read at append `4` is append `3`'s coin, the key of the other tile of the pass. -/
theorem prefix_origin_reads_key :
    ∃ (seen : (Fin 5 → Bool) → Fin 5 → Fin 5) (pay : Fin 5 → Bool) (joined : Fin 5 → ℕ),
      (∀ (a : Fin 5) (κ κ' : Fin 5 → Bool), (∀ b < a, κ b = κ' b) → seen κ a = seen κ' a) ∧
      Monotone joined ∧
      (∀ a : Fin 5, 3 ≤ a → joined a ≤ 1 ∧ 1 < joined a + 1) ∧
      (∀ κ, ∀ a : Fin 5, 3 ≤ a → (seen κ a : ℕ) = 0 ∨ joined (seen κ a) + 1 ≤ 1) ∧
      (∀ κ, seen κ 4 ≤ 3) ∧
      ∀ κ, pay (seen κ 4) = κ 3 := by
  refine ⟨fun κ a => if a = 4 then (if κ 3 then 2 else 1) else if a = 3 then 1 else a,
    fun a => decide (a = 2), fun a => if (a : ℕ) < 3 then 0 else 1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro a κ κ' h
    by_cases ha : a = 4
    · subst ha
      simp only [if_true, h 3 (by decide)]
    · simp only [ha, if_false]
  · intro a b hab
    have : (a : ℕ) ≤ b := hab
    dsimp only
    split_ifs <;> omega
  · intro a ha
    have : 3 ≤ (a : ℕ) := ha
    dsimp only
    split_ifs <;> constructor <;> omega
  · intro κ a ha
    have h3 : 3 ≤ (a : ℕ) := ha
    right
    by_cases h4 : a = 4
    · subst h4
      cases h : κ 3 <;> simp [h]
    · have h4' : (a : ℕ) ≠ 4 := fun h => h4 (Fin.ext h)
      have : a = 3 := Fin.ext (by have := a.isLt; omega)
      subst this
      simp
  · intro κ
    cases h : κ 3 <;> simp [h]
  · intro κ
    cases h : κ 3 <;> simp [h]

end ResourceAccounting.SecurityProofs.Refresh
