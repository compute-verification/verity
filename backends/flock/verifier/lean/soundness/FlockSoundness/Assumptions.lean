import Mathlib.Analysis.SpecialFunctions.Pow.Real
import FlockSoundness.Game.Expect

/-!
# Named assumptions

The results the soundness proof takes without proof. Each is a `Prop`, and each theorem that
uses one takes it as a hypothesis: the development declares no `axiom`, so `#print axioms`
lists only Lean's three standard axioms, and a mis-stated assumption can make a theorem
vacuous but can never make Lean prove something false.

There are two. Assumption 2 is a hypothesis about one explicit finder, for the audit's link term. Assumption 3 is a
hypothesis about the byte source the verifier draws its units from.
Assumption 1, BCHKS25's mutual correlated agreement up to the Johnson bound, is proved
(`mcaError_le_bchks25`, from DKT26). Merkle binding is proved too (VCVio's binding theorem,
`Merkle`), and SHA-256's collision resistance enters the compiled theorem as an explicit
probability term (the probability that an explicit, efficient reduction finds a collision), not
as a hypothesis.
-/

namespace FlockSoundness.Assumptions

open Game

/-- **Assumption 2: expected-time collision resistance of SHA-512**, for the audit's value layer, in
the expected-time sense of Chiesa, Dall'Agnol, Guan, Spooner and Yogev (ePrint 2024/1434): a finder
whose expected number of SHA-512 evaluations is `T` outputs a collision with probability at most
`T / 2^256`.

For a random oracle with `N = 2^512` outputs the sharp bound is `T / E[τ_N]`, with `τ_N` the number
of distinct inputs hashed until the first collision. Hashing until then succeeds with probability 1
at expected cost `E[τ_N] = √(πN/2) + 2/3 + o(1) ≈ 2^256.33`, so `2^256` holds there with 0.33 bits
to spare. The fixed-budget bound `min(1, q²/2^513) ≤ q/2^256.5` does not extend to finders that stop
adaptively: that attack beats `T/2^256.5` by a factor 1.13. The bound is linear in `T` because a
finder can gamble on a long run.

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
    expect cost g s / 2 ^ (256 : ℝ)

/-- **Assumption 3 (`uniform/io-getrandombytes`): the verifier's operating-system randomness is independent uniform
bytes**, for the unit draw.

The verifier draws the units it audits from `IO.getRandomBytes`: `Flock.Draw.drawOS` is `Flock.Draw.drawWith` with every
stream's bytes from it. To Lean's logic an `IO` action is a function of the world, and `IO.getRandomBytes` an opaque
constant, so a `Prop` saying its bytes are random would be refutable, and every theorem taking it vacuous. A3 is stated
instead of a byte source: a finite probability space `Ω` (`prCoin` weighs its outcomes evenly) and, on outcome `ω`, the
bytes `t ω i` that stream `i` returns, its first `L i`. The source is uniform when every event about those bytes has the
probability it has for independent uniform bytes.

What A3 assumes of the operating system is that the running verifier's calls to `IO.getRandomBytes` are such a source,
every call's bytes independent of every earlier call's, for every `L`. -/
def UniformRandomBytes {Ω : Type} [Fintype Ω] {m : ℕ} (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) :
    Prop :=
  ∀ E : ((i : Fin m) → Fin (L i) → Fin 256) → Prop, prCoin (fun ω => E (t ω)) = prCoin E

end FlockSoundness.Assumptions
