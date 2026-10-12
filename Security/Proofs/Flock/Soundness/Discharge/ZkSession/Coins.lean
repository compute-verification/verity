import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.CoinBinding

/-!
# The coin step: a committed coin is fresh when the prover moves (live/PROTOCOL.md §9, C-Flock ZK)

Under C-Flock ZK the verifier commits to every coin of the session before the prover's first message
(`coin-tree/hm96-sha512/v1`): round `j`'s block of coins is chained into a digest `x_j`, and its leaf is the hm96 leaf
`H(lp ‖ x_j ⊕ M·y_j ‖ H(sp ‖ y_j))` under a fresh salt `y_j`; the root of the leaves is sent first, and round `j`'s
coins, `T_n`, the salt and the path are opened only after the prover's round-`j` message. So the prover's message may
depend on round `j`'s leaf (through the root) and on everything else it has seen, but not on the salt.

`coin_fresh`: whatever the prover sends before the opening, as a function `σ` of the leaf and of any other randomness
`ω` independent of the coin and the salt (the other rounds' coins and salts, hence their leaves and the root; the coins
already opened; the prover's own randomness), the probability of any event `bad` of the message and the coin exceeds
that of the coin drawn fresh after a worst-case message by at most `δ₁`, hm96's hiding distance
(`Assumptions.Hm96Hiding`). It is T1 (`ZK.ideal_leaf_swap`): with an ideal leaf the message is independent of the coin.
`coin_fresh_key`: the same at a uniform key drawn first, with the leaf hash and the event chosen after it and `δ₁`
averaged over the key. `coinLeaf_fresh`: the coin tree's own leaf (`ZK.coinLeaf`), for a block of `n` coins.

So each round of a session whose verifier commits its coins costs its fresh-coin bound plus `δ₁`. `Hm96Hiding` is the
assumption as main states it; worker zk-lean-cr-only proves it for every salt map (`hm96Hiding_gap`, at the distance
`hidingGap`) and bounds `hidingGap`'s mean over hm96-sha512's keys by `2^-257` (`hm96_sha512_sum`), on branch
`cursor/zk-lean-cr-only-95d4`; until that lands, it is a hypothesis here.
-/

namespace FlockSoundness.Discharge.ZkSession

open Game ZK
open scoped ENNReal

/-- **A committed coin is fresh when the prover moves** (T1). The coin `c`, the salt `y` and the rest `ω` are
independent and uniform; the round's digest `x ω c` may depend on the coin and on `ω`; the prover's message is any
function `σ ω ℓ` of `ω` and the leaf `ℓ = H(x + M·y, cs y)`. Then any event of the state, the message and the coin has
probability at most its worst case over states and messages with the coin drawn fresh, plus `δ₁`. -/
theorem coin_fresh {Ω C Y B Cs D M : Type} [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C] [Fintype Y]
    [Nonempty Y] [AddCommGroup B] [Fintype B] (My : Y → B) (cs : Y → Cs) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding My cs δ₁) (H : B × Cs → D) (x : Ω → C → B) (σ : Ω → D → M)
    (bad : Ω → M → C → Prop) :
    prCoin (fun p : (Ω × C) × Y => bad p.1.1 (σ p.1.1 (H (x p.1.1 p.1.2 + My p.2, cs p.2))) p.1.2) ≤
      (⨆ (ω : Ω) (m : M), prCoin (bad ω m)) + δ₁ := by
  refine (ideal_leaf_swap (Ω := Ω × C) My cs hT1 H (fun p => x p.1 p.2)
    (fun p ℓ => bad p.1 (σ p.1 ℓ) p.2)).trans (add_le_add_left ?_ _)
  let e : (Ω × C) × (B × Y) ≃ (Ω × (B × Y)) × C :=
    { toFun := fun p => ((p.1.1, p.2), p.1.2)
      invFun := fun p => ((p.1.1, p.2), p.1.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [← prCoin_congr_equiv e.symm]
  change prCoin (fun p : (Ω × (B × Y)) × C => bad p.1.1 (σ p.1.1 (H (p.1.2.1, cs p.1.2.2))) p.2) ≤ _
  rw [prCoin_prod_split (fun (q : Ω × (B × Y)) (c : C) => bad q.1 (σ q.1 (H (q.2.1, cs q.2.2))) c)]
  calc avg (fun q : Ω × (B × Y) => prCoin fun c => bad q.1 (σ q.1 (H (q.2.1, cs q.2.2))) c)
      ≤ avg (fun _ : Ω × (B × Y) => ⨆ (ω : Ω) (m : M), prCoin (bad ω m)) :=
        avg_mono fun q => le_iSup_of_le q.1 (le_iSup_of_le (σ q.1 (H (q.2.1, cs q.2.2))) le_rfl)
    _ = _ := by
        exact avg_const _

/-- **At a uniform key** (`coin_fresh` averaged): the key `κ` is uniform, drawn first and independent of the rest; the
hiding distance `δ κ` may depend on it, and so may the leaf hash `H κ`, the digest, the prover's message and the event
`bad κ`. The bound is each key's fresh-coin worst case and `δ`, both averaged over the key. -/
theorem coin_fresh_key {Kk Ω C Y B Cs D M : Type} [Fintype Kk] [Nonempty Kk] [Fintype Ω] [Nonempty Ω] [Fintype C]
    [Nonempty C] [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] (My : Kk → Y → B) (cs : Y → Cs) (δ : Kk → ℝ≥0∞)
    (hT1 : ∀ κ, Assumptions.Hm96Hiding (My κ) cs (δ κ)) (H : Kk → B × Cs → D) (x : Kk → Ω → C → B)
    (σ : Kk → Ω → D → M) (bad : Kk → Ω → M → C → Prop) :
    prCoin (fun p : Kk × ((Ω × C) × Y) =>
        bad p.1 p.2.1.1 (σ p.1 p.2.1.1 (H p.1 (x p.1 p.2.1.1 p.2.1.2 + My p.1 p.2.2, cs p.2.2))) p.2.1.2) ≤
      avg (fun κ => ⨆ (ω : Ω) (m : M), prCoin (bad κ ω m)) + avg δ := by
  rw [prCoin_prod_split (fun (κ : Kk) (q : (Ω × C) × Y) =>
    bad κ q.1.1 (σ κ q.1.1 (H κ (x κ q.1.1 q.1.2 + My κ q.2, cs q.2))) q.1.2)]
  calc avg (fun κ => prCoin fun q : (Ω × C) × Y =>
          bad κ q.1.1 (σ κ q.1.1 (H κ (x κ q.1.1 q.1.2 + My κ q.2, cs q.2))) q.1.2)
      ≤ avg (fun κ => (⨆ (ω : Ω) (m : M), prCoin (bad κ ω m)) + δ κ) :=
        avg_mono fun κ => coin_fresh (My κ) cs (hT1 κ) (H κ) (x κ) (σ κ) (bad κ)
    _ = _ := by rw [avg_add]

/-- **The coin tree's leaf** (`ZK.coinLeaf`, `coin-tree/hm96-sha512/v1`): round `pos`'s block of `n` coins, chained from
`T_n`, under hm96's mix `x ⊕ M·y` (written `x + M·y`, the digests' group). Whatever the prover sends before the opening,
as a function of the leaf and of `ω`, any event of the state, the message and the block has probability at most its
fresh-coin worst case plus `δ₁`. `T_n` may depend on `ω`. -/
theorem coinLeaf_fresh {Ω Coin Salt D M : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin]
    [Fintype Salt] [Nonempty Salt] [AddCommGroup D] [Fintype D] (H : List UInt8 → D) (E : CoinEnc D Coin Salt) (My : Salt → D)
    (chainTag roundTag pos lp sp : List UInt8) {δ₁ : ℝ≥0∞}
    (hT1 : Assumptions.Hm96Hiding My (fun y => H (sp ++ E.salt y)) δ₁) (n : ℕ) (Tn : Ω → D)
    (σ : Ω → D → M) (bad : Ω → M → (Fin n → Coin) → Prop) :
    prCoin (fun p : (Ω × (Fin n → Coin)) × Salt =>
        bad p.1.1 (σ p.1.1 (coinLeaf H E (fun x y => x + My y) chainTag roundTag pos lp sp (Tn p.1.1)
          (List.ofFn p.1.2) p.2)) p.1.2) ≤
      (⨆ (ω : Ω) (m : M), prCoin (bad ω m)) + δ₁ :=
  coin_fresh My (fun y => H (sp ++ E.salt y)) hT1 (fun bc => H (lp ++ E.dig bc.1 ++ E.dig bc.2))
    (fun ω c => H (roundTag ++ pos ++ E.dig (chainTop H E chainTag (Tn ω) (List.ofFn c)))) σ bad

end FlockSoundness.Discharge.ZkSession
