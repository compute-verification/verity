module

public import Definitions.Pouw.PearlC.Tile
public import Definitions.Core.Game
public import Definitions.Core.Interaction

/-!
# The hidden audit (trusted)

The sampled audit (`GGapSampled`) opens each sampled tile. The hidden audit makes the same sample and opens nothing: the
verifier holds the transcript's commitment, and for each sampled tile it plays a proof game with the prover, a public-coin
interaction with live coins (core's `Verity.Game`), accepting when every verdict is accept. The proof is Flock's `--zk`
proof of the tile-check Program. `GGapHidden` is `GGapSampled`'s bound for that audit, for every opening of the commitment
(its salts, the prover's like the transcripts) and every prover of the proof phase in the class its proof system's
soundness speaks of (`HiddenAudit.Bounded`), at `t·δzk(q)` more error.
-/

@[expose] public section

open Claims renaming Ex → ex

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **The hidden audit**: the tile check as a proof. The verifier never opens a sampled tile. It holds the transcript's
commitment `root U τ o`, where the opening `o` is the commitment's randomness (for hm96-SHA-512, the leaves' salts),
and for sampled tile `g`, on the oracle `H` and salt `s` (from which it recomputes the tile's anchors), it plays the proof
game `game H s U (root U τ o) g`, ending in its verdict.

`Bounded τ o g σ q` says the prover `σ` of the proof phase `g`, on the transcript `τ` committed under the opening `o`,
is in the class the proof system's soundness speaks of, at query budget `q`. The class sees `τ` and `o` because the
proof system's finders are built from the committed data. For Flock that class is query-bounded in its own sense:
every collision finder its theorems build from `σ` (`TableCR`, `LinkCR`, and the tree and registered-path finders of
`flock_e2e_drawn_hm96_reads`, the last from the registrant's rows, salts and paths, here `τ`'s under `o`) meets
`SHA512CRStrict` (`cr/sha-512`) or `SHA512CRExpected` (`ecr/sha-512`) at budget `q`. A bare count of SHA-512 evaluations
would not do: some strategy has a SHA-512 collision built in and opens a commitment two ways after no evaluations, and
no soundness bound holds for it (`SHA512CRExpected`'s docstring).

An empty class, or any `Bounded` that excludes the honest prover, makes `TileProofSoundAll` and `GGapHidden` vacuous:
the first then holds at any `δzk ≥ 0` and says nothing about the proof system, so the second follows from `GGapSampled`
alone. So a gap under the hidden audit is cited
only beside its instance's pinned theorem that the honest prover, at its stated `q` (the batch's whole work), satisfies
`ProverBounded`. That theorem is not enough on its own: `Bounded` must be the proof system's own class, stated without
reading which of `τ`'s tiles are good, as Flock's per-finder class is. A class that reads goodness can hold the honest
prover and still make `TileProofSoundAll` hold at `δzk = 0` with no proof system (`TileGame`'s docstring). -/
structure HiddenAudit (Q R S : Type) where
  Root : Type
  Opening : Type
  root : Workload → Transcript → Opening → Root
  game : (Q → R) → S → Workload → Root → ℕ → Verity.Game Bool
  Bounded : Transcript → Opening → (g : Verity.Game (List Bool)) → Verity.Game.Strategy g → ℕ → Prop

/-- The proof phase after the sample `c`: the sampled tiles' proof games run side by side, the prover scheduling them
(`Verity.Game.batch`), their verdicts in sample order. -/
def HiddenAudit.batch (HA : HiddenAudit Q R S) (H : Q → R) (s : S) (U : Workload) (r : HA.Root) {w : ℕ → ℕ}
    {NT t : ℕ} (c : Fin t → Ticket w NT) : Verity.Game (List Bool) :=
  Verity.Game.batch (List.ofFn fun i => HA.game H s U r (c i).1)

/-- A prover of the proof phase against the transcripts `τ` committed under the openings `op`: on the oracle, the salt
and the sample `x = (H, s, c)`, a strategy in the batch of `c`'s tiles' proof games on the commitment of `τ H s` under
`op H s`. The opening is a function of `H` and `s` only, so it is fixed before the sample. -/
abbrev HiddenAudit.Prover (HA : HiddenAudit Q R S) (TR : TileRules Q R S) (U : Workload)
    (τ : (Q → R) → S → Transcript) (op : (Q → R) → S → HA.Opening) (w : ℕ → ℕ) (t : ℕ) : Type :=
  (x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT)) →
    Verity.Game.Strategy (HA.batch x.1 x.2.1 U (HA.root U (τ x.1 x.2.1) (op x.1 x.2.1)) x.2.2)

/-- `σ` is in the proof system's class at budget `q` after every sample, on the transcript and the opening it proves
against. -/
def HiddenAudit.ProverBounded (HA : HiddenAudit Q R S) {TR : TileRules Q R S} {U : Workload}
    {τ : (Q → R) → S → Transcript} {op : (Q → R) → S → HA.Opening} {w : ℕ → ℕ} {t : ℕ}
    (σ : HA.Prover TR U τ op w t) (q : ℕ) : Prop :=
  ∀ x, HA.Bounded (τ x.1 x.2.1) (op x.1 x.2.1) _ (σ x) q

open Classical in
/-- The probability, over the oracle, the salt, the sample `x = (H, s, c)` (uniform) and the proof games' coins, that the
hidden audit accepts (every verdict is accept) while `T` is below `1 − g` of `τ`'s claimed work less `ε_s` of the
layout's: `GGapSampled`'s event, with the verdicts in place of the opened tiles. -/
noncomputable def HiddenAudit.winProb [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (HA : HiddenAudit Q R S)
    (TR : TileRules Q R S) (U : Workload) (τ : (Q → R) → S → Transcript) (op : (Q → R) → S → HA.Opening)
    {w : ℕ → ℕ} {t : ℕ} (σ : HA.Prover TR U τ op w t) (g T εs : ℝ) : ℝ :=
  ex fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
    if T < (1 - g) * (claimedWref TR U (τ x.1 x.2.1) - εs * tileWrefTotal TR U.layout) then
      (Verity.Game.prob (fun v => ∀ i : Fin t, v.getD i false = true)
        (HA.batch x.1 x.2.1 U (HA.root U (τ x.1 x.2.1) (op x.1 x.2.1)) x.2.2) (σ x)).toReal
    else 0

/-- **The gap property under the hidden audit.** `GGapSampled` with the verifier running, after the sample, each sampled
tile's proof game on the transcript's commitment, and accepting when every verdict is accept. For every in-domain
workload, budget `T`, query bound `q`, adversary within them, ticket count, `t` and `ε_s ≤ 1` as in `GGapSampled`, every
opening `op` of the commitment and every prover `σ` of the proof phase in the proof system's class at `q` on the
adversary's transcripts under `op` (`HiddenAudit.Bounded`): the probability, over the oracle, the salt, the sample and the
proof games' coins, that the audit accepts while `T` is below `1 − g` of the claimed work less `ε_s` of the layout's is
at most `(1 − ε_s)^t + δ(q, N) + t·δzk(q)`.

The sample is `GGapSampled`'s, the same random variable and not a fresh one: `x = (H, s, c)`, uniform on the same type
`(Q → R) × S × (Fin t → Ticket w NT)`, and the batch runs the proof games of the tickets `c = x.2.2`, the ones
`AuditAccepts` reads in `GGapSampled`. The commitment's opening is the adversary's, quantified like its transcripts, and a
function of `H` and `s` only, so it is fixed before the sample. The proof games' coins are live coins, flipped inside each
game after it. The prover sees the whole sample. -/
def GGapHidden [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (HA : HiddenAudit Q R S) (g : ℝ) (δ : ℕ → ℕ → ℝ) (δzk : ℕ → ℝ) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ i < (TR.tiling U.layout).NT, (w i : ℝ) = κ * TR.Wref U.layout i) →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 → ∀ op : (Q → R) → S → HA.Opening,
    ∀ σ : HA.Prover TR U (A.transcript U) op w t, HA.ProverBounded σ q →
      HA.winProb TR U (A.transcript U) op σ g T εs ≤ (1 - εs) ^ t + δ q U.layout.N + t * δzk q

end Pouw.PearlC
