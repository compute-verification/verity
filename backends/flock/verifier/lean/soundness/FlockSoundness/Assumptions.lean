import Mathlib.Analysis.SpecialFunctions.Pow.Real
import FlockSoundness.Game.Expect

/-!
# Named assumptions

The results the soundness proof takes without proof. Each is a `Prop`, and each theorem that
uses one takes it as a hypothesis: the development declares no `axiom`, so `#print axioms`
lists only Lean's three standard axioms, and a mis-stated assumption can make a theorem
vacuous but can never make Lean prove something false.

There is one, Assumption 2, a hypothesis about one explicit finder, for the audit's link term.
Assumption 1, BCHKS25's mutual correlated agreement up to the Johnson bound, is proved
(`mcaError_le_bchks25`, from DKT26). Merkle binding is proved too (VCVio's binding theorem,
`Merkle`), and SHA-256's collision resistance enters the compiled theorem as an explicit
probability term (the probability that an explicit, efficient reduction finds a collision), not
as a hypothesis.
-/

namespace FlockSoundness.Assumptions

open Game

/-- **Assumption 2: expected-time collision resistance of SHA-512** (Chiesa, Dall'Agnol, Guan,
Spooner and Yogev, ePrint 2024/1434, Theorem 2), for the audit's value layer: a finder whose
expected number of SHA-512 evaluations is `T` outputs a collision with probability at most
`T / 2^256.5`. It is SHA-512's generic bound, since `min(1, q²/2^513) ≤ q/2^256.5`, and it is
linear in `T` because a finder can gamble on a long run.

A finder is a game `g` played by a fixed strategy `s`, so a randomized algorithm with finitely
many coin outcomes. On outcome `b` it outputs `out b` after `cost b` evaluations of `H`
(SHA-512), counting `t` for each run of the prover it plays.

It is a predicate on one finder, not a statement about all of them: some finder outputs a
collision of `H` after no evaluations, since collisions exist. The link theorem takes it for the
finder it builds from the prover, as the compiled theorem charges strict-time collision
resistance to explicit reductions. -/
def SHA512ExpectedTimeCR {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) : Prop :=
  (prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s).toReal ≤
    expect cost g s / 2 ^ (256.5 : ℝ)

end FlockSoundness.Assumptions
