import Definitions.Pouw.RelativeSecurity

/-!
# The whole-unit guess, one checkpoint at a time (trusted)

The premise route R-CP (note:20261019T0534Z-draft-fpx-luck) takes instead of the whole-unit guess bound (Compose's
`GuessLuckAt`, the headlines' `WholeLuckAt`). CP-W's check pins every one of a unit's `KS + m` checkpoints
(`PouwWarpCheck_v2`: the XOR over the warp's lanes and chains of `y` and of `z` after each set), and a unit skipped whole
(its product and its fold) is deficient at each of them. So a bound per checkpoint, given the earlier ones, compounds
over them (`Accounting.Guarantees.Checkpoint.WholeGuessOfCheckpoints`).
-/

namespace Accounting.Assumptions.Checkpoint

open Pouw.RelativeSecurity

/-- `accounting-checkpoint-guess` (the per-checkpoint guess bound; a claim, Daniel's to rule on, no card decided).
**A unit skipped whole keeps each checkpoint, given the earlier ones, only by guessing it**: over a tile's coins `C`,
for every checkpoint `s < n`, `Pr[E_s ∧ E_r for every r < s ∧ whole] ≤ q·Pr[E_r for every r < s ∧ whole]`, where `E_s` is
"the committed checkpoint after set `s` is the honest one" and `whole` is the tile's being skipped whole. It is a
condition at the `q` it is given: the live line would read it at `½` (ASK-9's value, per 64-bit checkpoint instead of
per unit) and the seeded whole branch at `e^-4`, per seed query over its answer. In the scope `GuessLuckAt` states (no
computed word, and no coin read before the skip, determines a fold state) a whole-skipped unit's committed checkpoints
read none of its coins, and the sets' fresh coins are independent, so the bound is the honest checkpoint's point mass
over set `s`'s fresh coins at the state the earlier sets leave: a property of the fold and the noise, not of the
cheater. It fails where `GuessLuckAt` fails: under a coin event or an instance on which a checkpoint is cheap to compute
without the product and the fold. -/
def CheckpointGuessAt {C : Type} [Fintype C] (q : ℝ) (n : ℕ) (E : ℕ → C → Prop) (whole : C → Prop) : Prop :=
  ∀ s < n, prob (fun c => E s c ∧ (∀ r < s, E r c) ∧ whole c) ≤ q * prob (fun c => (∀ r < s, E r c) ∧ whole c)

end Accounting.Assumptions.Checkpoint
