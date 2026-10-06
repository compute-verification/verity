import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Proofs.Flock.Soundness.Game.Expect
import Proofs.Flock.Soundness.Randomness
import Definitions.Flock.Model.Basic
import Specs.Flock.Assumptions.Soundness

/-!
# Assumption 4, beside the verifier's SHA-256

C-Flock's named assumptions are in `Specs.Flock.Assumptions.Soundness`, on Mathlib alone, and this module re-exports
them. Assumption 4 (`KeyedStreamsUniform`, `prf/sha-256`) stays here: it is stated over `verity.randomness`'s keyed
streams (`Randomness.stream`), which run on the verifier's SHA-256 (`Flock.Sha256.hash`), and the spec imports no
program. No guarantee takes it: it is a hypothesis of lemmas only (the PoUW window draw's, `KeyedDraw`), and a guarantee
that took it would fail the audit until it is a constant of an `assumptions` module.
-/

namespace FlockSoundness.Assumptions

open Game
open scoped ENNReal

/-- **Assumption 4 (`prf/sha-256`): `verity.randomness`'s keyed streams at distinct indices pass a test as independent
uniform bytes do, up to `η`**, for the PoUW window draw.

A verifier secret `source` of `S` bytes gives the key `Randomness.derive source domain context`. The key's stream at
index `idx i` is `Randomness.stream key (idx i) (L i)`, its first `L i` bytes, the spec of `Key.stream` that
`backends/flock/tests/lean/RandomnessVectors.lean` ties to the Python. The streams pass the test `E` with at most the
probability that independent uniform streams of those lengths do, plus `η`, when the source is uniform and the indices'
framings (`Randomness.frames`, the bytes each stream hashes after the key) are pairwise distinct.

It is a predicate on one test, as `SHA512CRExpected` is on one finder. A statement for every test would be refutable, since `2^(8S)`
sources give at most that many stream tuples. The theorems take it for the test they need, whether the draw computed
from the streams misses a given set, and `η` enters their bound.

What A4 assumes of SHA-256 is that its compression function `f` is a pseudorandom function in two ways. Keyed by its
chaining value, it makes the cascade a PRF on prefix-free inputs (Bellare, Canetti and Krawczyk 1996). Keyed by the secret
bytes in its first message block, after a fixed public tag (28 bytes for `stream`, 37 for `derive` with the source's frame
header), it is a dual PRF, as the HMAC analyses use. The inputs are prefix-free: `frames` is length-framed and the counter
is 8 bytes, so length extension doesn't apply. The secret's uniformity is not A4's: it is A5 (`UniformSecret`). -/
def KeyedStreamsUniform (S : ℕ) (domain : String) (context : List (String × Randomness.Part)) {m : ℕ}
    (idx : Fin m → List Randomness.Part) (L : Fin m → ℕ) (E : (Fin m → ByteArray) → Prop) (η : ℝ≥0∞) : Prop :=
  (∀ i j, Randomness.frames (idx i) = Randomness.frames (idx j) → i = j) →
    prCoin (fun src : Fin S → Fin 256 => E fun i =>
        Randomness.stream (Randomness.derive ⟨Array.ofFn fun b => UInt8.ofFin (src b)⟩ domain context) (idx i) (L i)) ≤
      prCoin (fun f : (i : Fin m) → Fin (L i) → Fin 256 => E fun i => ⟨Array.ofFn fun b => UInt8.ofFin (f i b)⟩) + η

end FlockSoundness.Assumptions
