import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe: secure erasure from streamed PRF labels (experimental)

A proof of secure erasure for memory wipes, in place of the permutation-based
retention for wipes: the ideal-permutation premise (C8) is false for the reference Feistel10-SHAKE256 `Π₂`, where
1.53% of the fill rebuilds any label. The game is `Security/Definitions/Pous/Model/StreamedWipe.lean`,
the protocol `verity/core/protocols/pous/PROTOCOL.md` ("The streamed-label wipe"). Proofs:
`Security/Proofs/Pous/StreamedWipe/`.

**What they rest on.** No named assumption. The labels' PRF (C3, claim `prf/sha-512`) enters as an advantage term in
the conclusion (`prfAdvantage`), which C3 bounds for an efficient prover; the guarantee holds for every prover. The
physical premises enter as the game's shape, and the challenger's deployment must make them true:

- *no unaccounted tier*: `M` is every bit the server can hold when the fill ends, all its memory tiers;
- C6, *egress certified over the whole wipe*: `E` bounds the bits that left the boundary from the fill's start to the
  last challenge, as the network certifier counts them, and the deadline keeps an answer from being fetched from
  outside in any other way;
- C5, *no side channel*: no label reaches the outside except through that egress, and nothing reaches the server during
  the challenges but the indices and what the outside sends, which is a function of `ξ` and the indices.

The game is not yet tied to the actual-circuit model of `Definitions.Accounting.Physical`, whose `Memory` and `Links`
constraints are these premises' forms there.

**What it covers.** The sequential reveal: round `j`'s answer reads the indices before it, not those after. A batched
reveal, where an answer can read its whole batch, is not covered.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **The streamed-label wipe is a proof of secure erasure.** For every prover of the sequential wipe of `N` labels of
`b` bits under `k` challenges, holding `M` bits inside the boundary when the fill ends, with `E` bits of egress, and
`q` bits of uniform prior content, and for every `g`:

`Pr[accepted ∧ the prior content survives] ≤ (g/N)^k + k · C(N, g+1) · 2^(M+E) / 2^(q+(g+1)·b) + Adv_PRF`,

where the labels are `prf K` under a uniform key, and `Adv_PRF` is the PRF's advantage against the distinguisher that
runs the wipe game with this prover. With `M = N·b` (the fill is the whole memory), the compression term is small once
`q` exceeds the missing labels' bits `(N − g − 1)·b` plus `E` plus `log₂ C(N, g+1) + log₂ k`, so an accepted prover
keeps at most about that much of the prior content: the labels it didn't hold, the egress and a counting term.

It is a bound in the game of `Pous.StreamedWipe`, for any `N`, `k`, `b`, `q`, `M`, `E`, `κ` and `prf`, with no
named assumption. Proof: `PousStreamedWipe.streamed_wipe`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipe : Prop :=
  ∀ (N k b q M E κ g : ℕ) (prf : Bits κ → Fin N → Bits b) (P : StreamedWipe.Prover N k b q M E),
    StreamedWipe.winReal prf P ≤
      StreamedWipe.bound N k b q M E g + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

/-- **The deployed wipe: 2.6 TB, 74,000 challenges.** `StreamedWipe` at `N = 40,625,000,000` labels of 512 bits (the
2.6 TB fill, every tier of one server, so `M = N · 512`) and `k = 74,000` sequential challenges: a prover with `E` bits
of egress that keeps at least `E + 8,600,000,000` bits (1.075 GB plus the egress) of uniform prior content through the
wipe is accepted with probability at most `2^-40` plus the PRF's advantage.

The error splits as `2^-41` for sampling (`g = N − 15,610,000`: the prover answers at most `ρ = 0.999616` of the
labels) and `2^-41` for compression, with the count `C(N, r) ≤ N^r ≤ 2^(36·r)`. The exact count gives about 1.023 GB
(`streamed_wipe.retained_bits`), and the explicit number here spends the difference on the cruder count.

Same footing as `StreamedWipe`: no named assumption; C3, the tiers and C5/C6 as there. Proof:
`PousStreamedWipe.streamed_wipe_meets_2600GB`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeets2600GB : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 40625000000 → Bits 512)
    (P : StreamedWipe.Prover 40625000000 74000 512 q (40625000000 * 512) E),
    E + 8600000000 ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 40 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

end Pous.Guarantees
