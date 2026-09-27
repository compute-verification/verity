# Proof of useful space (POUS)

A GPU keeps an encoding `C` of model weights `W` in its memory and decodes `W` from it on every use. A timed audit
must certify that the GPU still holds more than a `ρ` fraction of `C`, so the memory it claims is really spent on the
weights. This directory holds the problem's Lean formalization (`lean/`): the definitions, the pinned statements, and
the proofs so far. A Python distribution (`verity_pous`: reference encoder, decoder and audit simulator) comes when that
code lands.

## The problem: timed audit, perfect isolation

- **Syntax.** A trusted encoder `Enc(W) → (pp, C, vk)` runs once and erases its secrets. `W` is arbitrary and fixed
  before any setup randomness (all zeros included). Decoding `Dec(pp, C) = W` is public and bit-exact. `C` is `N`
  blocks of `b` bits.
- **Parameters.** `λ = 128`; `|C| + |pp| ≤ 1.05·|W|`; retention `ρ = 18/19`; soundness `δ = 1/100` plus the
  construction's `ε_crypto`.
- **Time.** A responder is `(D, Q)`-bounded: `D` sequential steps and `Q` operations in total. In ideal-model proofs a
  step is one round of parallel queries to the primitive, and `Q` counts queries.
- **Game.** The adversary picks `W`; then `Enc(W)`. Preprocessing, with no time limit, leaves a state `σ` and nothing
  else. The verifier issues `k` uniform block indices, all at once or one at a time. Each answer is due within the
  deadline, so the responder is `(D, Q)`-bounded, with no outside help. Whenever a challenge arrives the adversary's
  state is at most `ρ|C|` bits; `W` and `pp` are free.
- **Requirement.** `Pr[every answer correct and on time] ≤ δ + ε_crypto`. The honest prover stores `C` and answers by
  lookup.
- **Cost** (not modeled in Lean): encoded matmuls at most 2× the tuned plaintext sequence, decoding on every use.

The floor is `k = 86` challenges (`ρ^k ≤ δ`). Locality (bandwidth-limited off-device storage) is a later relaxation.

## In Lean (`lean/`)

- **Trusted, reviewed once:** `Pous/Game` (bits, uniform probability, parallel oracle programs `Prog`), `Pous/Model`
  (schemes, the audit game `AuditSecure`, `INC`/`TimedINC`, and the models below), `Pous/Accounting/{Params,LemmaA}`
  (`ρ`, `δ`, `Meets`), `Pous/Assumptions.lean` and `Pous/Pinned.lean`.
- **`Meets sch reveal D Q k ε`:** `ε ≤ 2^-128`, `sch` is correct, `|C| ≥ |W|`, the space bound holds, and
  `AuditSecure` holds at a state of `⌊ρ|C|⌋` bits with bound `δ + ε`.
- **Proofs:** `PousProofs/`, checked by Verity's `tools/lean/audit.py` against `lean-audit.json`.
  `PousProofs/Pinned.lean` proves `Pous.Pinned.X` as `PousProofs.X`.
- **Untrusted submissions:** `grade.sh FILE Pous.Pinned.X` grades one file against one pinned statement (`lean/README.md`).

## Pinned statements (`lean/Pous/Pinned.lean`)

| Statements | What | Status |
|---|---|---|
| `KFloor` | `k = 86` is the least `k` with `ρ^k ≤ δ` | proved |
| `Theorem1ii`, `Theorem1iiTimed` | report Theorem 1(ii): `INC(β, π)` (timed: per block) gives `π + p₀^k` for product-form responders | proved |
| `Theorem1iTimed`, `Theorem1iiSeq`, `Theorem1iiiTimed` | Theorem 1(i) timed, 1(ii) sequential, 1(iii) timed | proved |
| `FibreCount`, `A1CompressionCard`, `A1CompressionPr`, `A6RederiveUntimed`, `A6RederiveTimed` | fibre counting, compression, cheap re-derivation | proved |
| `A3Prime`, `A3PrimeLam` | the sequential audit from Lemma A, taken as a hypothesis | proved |
| `A2LemmaAM1`, `A5RabinGcd`, `A7Params`, `M1Meets` | Lemma A in model M1, the Rabin gcd step, A7′'s parameter rows, `Meets` in M1 given B1′ | open |
| `B1IndepSet`, `B2Stacking`, `B3Chain` | independent sets, Fisch's stacking lemma (Claim 12), the tweaked hash chain | proved |
| `FischClaim13`, `FischClaim14` | Fisch ePrint 2018/702, Claims 13–14 | proved |
| `B5ExPostFactoBounded`, `B5ExPostFacto` | Pietrzak's ex post facto bound for one-way labels | proved |
| `TransposeMixing`, `ReverseNeedsAllChildren` | N10's two core lemmas for P3's invertible labels | proved |
| `N10Refuted` | `¬ InvExPostFactoTwoLayer`: N10 as first stated (PIEs Conjecture 1 at `L = 2`) is false | proved |
| `SampledAuditProd`, `SampledAuditSeq`, `SampledAuditSim` | "fewer than `k` answerable" gives a sampled-audit bound | proved |
| `StackedAuditSeq`, `StackedAuditOnePercent` | the stacked one-way scheme's audit clause; the second is conditional on a depth-robust graph | proved |
| `DenseMeets` | the dense scheme S1 meets `Meets` with no named assumption, in the ideal-hash model only (security, correctness and rate, not decode cost) | proved |

Also proved and pinned in `lean-audit.json`, with no `Pous.Pinned` statement: the one-way bound on Fisch's stacked
graph (`PousBridge.b5_stacked`), and the band certificate for the two-directional column game (`Pous.ColumnGame`): the
band graph `B(220, 12)` at game depths 10 and 11, and `B(220, 16)` at depth 12, save at most `X = 3` labels
(`PousP3Column.band_twoWay_*`).

**Why N10 fails.** Top label `v` reads only column `v` of the lower layer, so an adversary can pay for a lower label a
column at a time, where whole-label pebbling charges it a whole label. N10′, the column-aware restatement over
`Pous.ColumnGame`, is being drafted.

## Named assumptions (`lean/Pous/Assumptions.lean`)

Each is a `Prop` taken as a hypothesis, never an `axiom`.

- `LemmaA sch S D Q g Γ G`: on the good setups `G`, an `S`-bit state and a per-block `(D, Q)` responder are right on at
  least `g` blocks with probability at most `Γ`. For M1 (`G = True`) it is `A2LemmaAM1`.
- `B1Prime sch Mdom S D Q k` (Conjecture B1′): `sch`'s sequential audit obeys model M1's completion bound. Used by
  `M1Meets`.
- Retracted: `InvertibleLabels`, which was N10's statement. It is false (`N10Refuted`).

## Provisional, waiting on Daniel

- `|C| ≥ |W|` in `Meets` (so `|pp| ≤ 0.05·|W|`): otherwise a scheme can keep `W` in the free `pp`.
- `ε_crypto ≤ 2^-128`: an unbounded `ε` makes the requirement vacuous.
- Which reveal a graded `Meets` may use, which waits on the per-answer versus shared deadline.

## Choices that could be wrong

- **H is a monolithic random oracle (narrow-state H).** Every result that models the hash H as a random oracle with
  `m`-bit outputs holds only in that model: B3's chain, B5 and the stacked audits, `DenseMeets`, and P3's labels (the
  band certificate is pure combinatorics, so it is unaffected). A real hash whose `m`-bit output comes from a narrower
  internal state breaks the instantiation. The adversary stores each label's pre-squeeze state and regenerates the
  labels from those states: with SHAKE's 1600-bit state, Dense in one round from about 20% of `C`, and P3 in three
  rounds from about 39%. This is the Ristenpart–Shacham–Shrimpton limit: indifferentiability does not carry over to
  storage games, so no ideal-hash result transfers to a concrete hash by citation. So "no named assumption" for
  `DenseMeets` means within the ideal-hash model only. The planned fix is H as a wide sponge (state at least `m` bits)
  over an ideal permutation, re-proved in the ideal-permutation model; a pilot has started. The Lean statements are
  unchanged. Source: the POUS store's `docs/p3-cryptanalysis.md`, "Narrow-state H".
- **One segment.** Every model plays one segment against a fresh primitive. A deployment must put a salt and the
  segment index in every oracle query.

## Next

N10′ (`ColumnAwareDraft`, `ForwardCoreDraft`, `n10prime`) and `p3-meets` are still changing, and come in the next PR.
