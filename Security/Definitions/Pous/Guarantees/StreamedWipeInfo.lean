import Mathlib.Analysis.SpecialFunctions.Log.Base
import Definitions.Core.Information
import Definitions.Pous.Model.StreamedWipeInfo

/-!
# The streamed-label wipe as conditional mutual information (experimental)

`StreamedWipe` bounds the probability that an accepted prover recovers the whole prior content `z`. This file states
the same retention in Shannon's form, which composes with other accounts of information: how much an accepted run's
state tells about `z`, in bits.
The game is `Security/Definitions/Pous/Model/StreamedWipe.lean`, the vocabulary
`Security/Definitions/Pous/Model/StreamedWipeInfo.lean`, and the information theory
`Security/Definitions/Core/Information.lean`.

**What it rests on.** No named assumption, as for `StreamedWipe`: the PRF enters as an advantage term in the
conclusion, and the physical premises (no unaccounted tier, C5, C6) enter as the game's shape.

The statement is not proved yet. `StreamedWipeMeetsInfoStated` holds its text for the lock. Two lemmas are due with the
proof: the honest one (a prover whose fill ignores `z` has `I = 0 ≤ 1`) and the nontrivial one (a prover that keeps `z`
whole, against a challenger that accepts everything, has `I = q` above the bound).
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **An accepted streamed-label wipe tells little about the prior content.** Take any prover of the sequential wipe
of `N` labels of `b` bits under `k` challenges, with `M` bits inside the boundary when the fill ends, `E` bits of
egress and `q` bits of uniform prior content `z`. Draw `z`, the challenges `c` and the key `K` uniformly, with labels
`L = prf K`. Let `σ'` be the fill's `(σ, ξ)` when the challenger accepts and ⊥ when it rejects (`StreamedWipe.held`).
Then for every `g`:

`I(z ; σ' | c) ≤ max(M + E − (g+1)·b + log₂(k·C(N, g+1)), 0) + q·((g/N)^k + Adv_list) + 3`,

in bits. Here `Adv_list` is the PRF's advantage against every fixed-size list of guesses of `z` read from `(σ', c)`
(`StreamedWipe.listAdvantage`).

The side information is `c` only: the labels and the key are hidden, and an accepted run's analysis sees what the
server and the outside hold, and the challenges.

Each term is a term of `StreamedWipe`'s bound:

- the lucky challenges `(g/N)^k`, on which a prover may keep all of `z`, are charged at `q` bits;
- the compression term `k·C(N, g+1)·2^(M+E) / 2^(q+(g+1)·b)` becomes the `log₂` of its `2^q` multiple, the bits
  of `z` the held state can carry beside `g + 1` labels.

The derivation:

1. The rank of `z` in its posterior given `(σ', c)` is `R`. The `m` most likely values form a list of size `m`.
   `StreamedWipe`'s counts, with the keep replaced by a list of `m` (`compression_count` once per guess), and a
   rejection's `m / 2^q`, bound `Pr[R ≤ m] ≤ (g/N)^k + Adv_list + m · 2^(T'' − q)`.
   Here `2^T'' = k·C(N, g+1)·2^(M+E−(g+1)·b) + 1`, so `T'' ≤ max(…, 0) + 1`.
2. Then `H(z | σ', c) ≥ E[log₂ R] ≥ (1 − a)·(q − T'') − log₂ e − 1/(e·ln 2)`, with `a = (g/N)^k + Adv_list`.
3. `I = q − H(z | σ', c)`, since `z` is uniform and independent of `c`. With the `+1` of `T''`, the constants total
   `2.98 < 3`.

The guessing list in `Adv_list` has to range over every fixed-size list read from `(σ', c)`. The advantage against
`wipeDistinguisher P`, which reads `P.keep`, is not enough. `I` doesn't read `keep`, and a prover with a constant
`keep` can hold all of `z` on real labels and none on uniform ones with that advantage 0. Shannon information is not
a computational quantity, so the term is an advantage of unbounded tests. It is 0 for a prover whose fill ignores `z`.

It is a bound in the game of `Pous.StreamedWipe`, for any `N`, `k`, `b`, `q`, `M`, `E`, `κ`, `g` and `prf`, with no
named assumption. Not proved yet: `StreamedWipeMeetsInfoStated` holds its text. -/
def StreamedWipeMeetsInfo : Prop :=
  ∀ (N k b q M E κ g : ℕ) (prf : Bits κ → Fin N → Bits b) (P : StreamedWipe.Prover N k b q M E),
    Verity.Information.condMutualInfo
        (fun x : (Bits q × (Fin k → Fin N)) × Bits κ => x.1.1)
        (fun x => StreamedWipe.held P x.1.1 (prf x.2) x.1.2)
        (fun x => x.1.2) ≤
      max ((M + E : ℝ) - (g + 1) * b + Real.logb 2 (k * N.choose (g + 1))) 0 +
        q * (((g : ℝ) / N) ^ k + (StreamedWipe.listAdvantage prf P).toReal) + 3

/-- **The streamed wipe's information bound is stated**: `StreamedWipeMeetsInfo`, equal to itself, so the lock records
its text and every definition it reads until it is proved. -/
def StreamedWipeMeetsInfoStated : Prop :=
  let s := StreamedWipeMeetsInfo
  s = s

end Pous.Guarantees
