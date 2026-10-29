import Definitions.Accounting.Carrier
import Definitions.Accounting.Assumptions.RateAudit
import Definitions.Accounting.Assumptions.RateAuditDeployed
import Definitions.Accounting.Assumptions.Stacking
import Definitions.Accounting.Assumptions.SquaringChain
import Definitions.Accounting.Guarantees.Requirement

/-!
# Stacking: carriers are a source of bounded capacity in every pass (trusted)

Stacking needs a carrier the audit counts: device memory whose contents were chosen after the seed (a credited KV page)
and hold other blocks' encodings, so that one read serves several receipts. In the ideal encoding model a carrier's
contents are a circuit's values, which depend on the encodings only through the circuit's inputs and its own encodes
(`Definitions/Accounting/Carrier.lean`). Five guarantees, whose proofs are in
`Security/Proofs/Accounting/Stacking.lean`:

* `CarrierBits`, the carrier lemma: memory holding a circuit's values holds at most its input and encoded bits about
  the store, `I + q·b` for `I` input bits and `q` blocks encoded.
* `StackedSessionConservationSketchDeployed`: the remote audit's conservation at the deployed parameters
  (`RemoteSessionConservationSketchDeployed`) with a store per pass, made by the append-coin rule: each append's blocks
  are keyed by a coin drawn after its root has committed their digests, so the writers' choices for it are fixed
  before (`Accounting.appendCoinBlocks`), and pass `j` reads the blocks its layout names at the session's coins drawn
  before it (`RateSketch.coinStore`, `RateSketch.programmed`), for devices whose rebuilds beyond `X` are their
  carriers' sectors. The stacked stock is charged in every pass's rebuilds: `X` grows by the carriers' bits under the
  property, `8·R_in·life` input bits and `1024·(c_ρ·reads·life)/E` encoded bits over a carrier's life of `life`
  windows (the property's `L`; `L` here is the side link's bits, as in every rate-audit statement).
* `StackedSessionConservationRelay`: the relay form, in the general class (`RemoteSessionConservationRelay`), where a
  carrier's inputs may depend on the pass's oracle and so relay receipts: a carrier is a side source of the same bits.
  Its store is one for the session, made by the same rule.
* `StackedRequirementSketchDeployed` and `StackedRequirementRelay`: the same sessions, with the shortfall read against a
  declared circuit `W`'s requirement (`Definitions/Accounting/Requirement.lean`) instead of the store: when the memory
  map and every pass's plan hold `W`'s credited blocks at each step the passes are bound to (`Requirement.StoreMatches`,
  `Requirement.PlanMatches`), an accepted session read at least `(1 − γ)` of the live bits the audit credits over
  those steps (`Requirement.creditedBits`: `W`'s live values in complete blocks) and of the bits `W` reads from
  complete blocks (`Requirement.creditedReadBits`).

The stacked sessions charge an encode at the gate model's cost (`encodeBitsCost`, nonlinear units of
`Definitions/Accounting/Gates`: budgeted units and data products, `⊤` where the model has no circuit computing the
words).
They rest on the remote audit's random oracle and named assumptions, and on more: the gate model's two conjectures
`Accounting.Assumptions.SquaringFloor` (a squaring mod `p` costs `e` nonlinear units) and
`Accounting.Assumptions.ChainSum` (a codec word's `k · S` chained squarings cost their sum, so a word costs
`E ≤ k · S · e`), and `Accounting.Assumptions.KeyBinding` (under the append-coin rule, the blocks a pass reads are
uniform but for `δ`). The
game orders every append's choices before its coin, as remote_bandwidth's protocol does, so no choice the writers make
reads the encoding it is keyed under, and the sessions pay for no advice.

**Idioms.** The carrier lemma is the counting step of incompressible encodings (Moran–Wichs, CRYPTO 2020: an
adversary's state that recovers the encoding is at least its entropy) and of proofs of replication (Fisch, SealStack,
2019: a replica stored as a function of other data is worth at most that data's bits), here with the circuit's
inputs and encodes as the only channels, as red-blue pebbling (Hong–Kung 1981; Ren–Devadas, bandwidth-hard functions,
TCC 2017) counts only what crosses the fast-memory boundary. The encode's price in nonlinear units is a circuit lower
bound, assumed per squaring and summed over the chain, as `WorkHardness` assumes the codec's price. The append coin is commit-then-challenge: a
writer commits to an append's digests before the verifier's live coin keys it, so what it chose can't aim at the key.
-/

namespace Accounting.Guarantees

open Accounting
open Pous (pr)
open scoped ENNReal

/-- **The carrier lemma: memory holding a circuit's values holds at most its input and encoded bits about the store.**
For every circuit over the store, seen through its inputs and encodes (`RateSketch.Provenance`), whose inputs total at
most `I` bits and whose encodes compute at most `X` bits at every store, its values take at most `2^(I + X)` contents
as the store ranges over every store:

  `|{run E : E}| ≤ 2^(I + X)`.

**The reading.** In the ideal encoding model the store is uniform, so whatever memory holds the circuit's values (a
carrier) holds at most `I + X` bits about the store: about the other blocks' encodings, which is what stacking needs.
`X ≤ q·b` for `q` encodes of at most a block's `b` bits each, and under `Accounting.Assumptions.SquaringFloor` and
`Accounting.Assumptions.ChainSum` at `E` nonlinear units a word of `8·bytes` bits a budget of `R` nonlinear units gives
`X ≤ 8·bytes·R/E`. The count is over
the prover's choices too: an input is any function of the store, so a carrier is a side source of `I + X` bits, whatever
the prover feeds.

Proof: `Accounting.SecurityProofs.CarrierBits`, by induction on the provenance: a step of `x` bits splits the stores
into at most `2^x` classes, in each of which the rest has `x` bits fewer.

Frozen (no-function): a counting bound over abstract provenances, about no function of the code. -/
def CarrierBits : Prop :=
  ∀ (N m I X : ℕ) {α : Type} (c : RateSketch.Provenance N m α),
    (∀ E, c.inputs E ≤ I ∧ c.encoded E ≤ X) →
    Nat.card (Set.range fun E => c.run E) ≤ 2 ^ (I + X)

/-- **The remote audit at the deployed parameters with a store per pass, with stacking: carriers are rebuilds of
bounded size in every pass, and every appended block is keyed after its writers chose it.**
`RemoteSessionConservationSketchDeployed`'s game and bound, with each pass on its own store, made by the append-coin
rule (`Accounting.appendCoinBlocks`; remote_bandwidth's PROTOCOL.md §3–§4). The store is `T` appends of `M` block
positions each, the registration first, whose coin is the seed. Append `a`'s choices `choose c ω a κ_<a` (its
payloads, which may be a circuit's values over other blocks' encodings, their padding, the run inputs behind them, how
many blocks, the schedule up to it) read the codec `ω`, the coins of the appends before it and the prefix `c`; its own
coin `κ a`, drawn once its root has committed every digest, keys its blocks (`enc`). Pass `j` of window `w` certifies
the blocks at the layout `slot c` (`RateSketch.coinStore`, `RateSketch.programmed`), at the prefix `c` of the
session's coins drawn before it (`RateSketch.SketchPrefix`: earlier windows', the window's signs, its earlier
passes'). `ω` and the append coins are uniform and independent of the session's coins, and under the append-coin rule
the blocks a pass reads are uniform but for `δ` (`Accounting.Assumptions.KeyBinding`). Every pass of the session
rebuilds at most `X` bits
plus one sector for each sector of a carrier (`RateSketch.Carrier`: memory whose contents are a circuit's values, which
decode to those sectors at every store), the carrier chosen at the pass's prefix. Each carrier's circuit, over its life
of `life` windows, takes in at most `R_in` input bytes a window and spends at most `c_ρ` nonlinear units a byte of
certified read, at `reads` bytes a window (the property's parameters; its lifetime cap is `life`, since `L` is the side
link's bits here), an encode of `x` bits costing `encodeBitsCost` at the deployed codec (p1024, 3 rounds) in the gate
model's restricted class (at most `B` operand bits a budgeted gate, on more than `εnum/εden` of the draws), `⊤` where
the model has no circuit computing the words. Under `Accounting.Assumptions.SquaringFloor` at `es` nonlinear units a
squaring and `Accounting.Assumptions.ChainSum`, at `E ≤ 3 · S · es` nonlinear units a word (`S = 1,021` squarings a
root) and `ε ≥ 2^−64` (p1024 and 3 rounds are inside the rest of their floor), the shortfall terms are the deployed
guarantee's at rebuilt bits `X + C` a pass,

  `C = 8·R_in·life + 1024·(c_ρ·reads·life)/E`,

the carriers' bits: inputs, and the bits of the 1024-bit words the circuit can encode with its nonlinear units. The
error is the deployed guarantee's draws and rank tail, and its per-pass hash terms plus `δ` a pass:

  `2·2^-40 + n·ε + (n·P·G)·N/2^512 + (n·P)·qtot/2^96 + (n·P)·δ`.

**The reading.** A carrier serves every pass of every window, so its stock is charged in each pass: `C` adds to the
side capacity `S + (L − ρ·256) + X` as one more source, a carrier being a source of `I + q·b` bits (`CarrierBits`),
with `I = 8·R_in·life` and `q·b ≤ 1024·c_ρ·reads·life/E`. In the verbatim class a sector the device decodes from a
carrier is one it computes rather than reads at its position, a rebuild.

* **Writes during the session.** A pass's store is the one resident going into the step it is bound to (a
  step's record takes effect after its reads, and the pass's plan is that step's reads, `Requirement.PlanMatches`). It
  may depend on every coin drawn before the pass: a page written from an earlier pass's receipts is in the view, and
  what it carries about other blocks is its carrier, chosen at the prefix, so charged in `C`. It may not depend on the
  pass's own coins, its grinding candidates among them: that is the audit rule that a block appended after a pass's seed
  is in no plan of that pass, which the game's store states and which the audit checks.
* **The append coin.** The game is the protocol's order: an append's choices are fixed when its root commits the
  digests, and its coin is drawn after, so no choice the writers make reads the encoding it keys, and none is advice.
  What a writer could once choose by looking at encodings it now chooses before the coin: a payload's padding and
  dead values, the run inputs behind a block, how many blocks an append holds. What the memory map fixes is no choice
  at all: append order and dummy appends (the memory-map unit proves the appends are the map's writes, in order, with
  none extra) and re-keys, each a release and an append of the same payload, at the lifetime cap only. After an
  append's coin its encoding is fixed: a block decodes only under its digest and coin, a coin is drawn once, and a
  re-sent root is refused. What its payload carries about other blocks is its carrier, charged in `C`.
* **Outside the game.** Two choices can still follow a coin and matter. The layout `slot c` reads the prefix only, the
  protocol stating frees, writes and plans as functions of the served steps, fixed before the coins (PROTOCOL.md §8):
  a layout selected by encodings after their coins (which blocks are freed or planned, or a re-key rather than a free
  at the cap, steered through run inputs or other inputs to the served steps) is a choice among the layouts the
  writers can steer to, which this game doesn't pay. And registrations: the game's seed is one draw, so a node that
  registers `R` times (`Registry.drawn`) keeps one of `R` seeds and fails at most `R` times this bound.
* **Life.** `life` is the life of a carrier's whole provenance: the circuit is each carrier's provenance back to values
  that depend on no store sector, and the property's isolation is what bounds it by the inputs and units of `life`
  windows. A carrier copied forward from older ones is one carrier whose life is the chain's, from its first write. A
  re-key keeps `d_j` (the memory-map unit checks it): it re-encodes the same payload under a fresh coin and leaves the
  provenance as it was, so a lifetime cap that restarts at a re-key still bounds `life`. A write that changes a
  payload is an append whose provenance includes the old block's, so its life counts from the chain's first write.

Proof: `Accounting.SecurityProofs.StackedSessionConservationSketchDeployed`: each carrier's sectors are counted by
`CarrierBits` with an encode's cost bounded by `SquaringFloor` and `ChainSum`, so every pass rebuilds at most `X + C`
bits; an
unpaid window has a rank-poor sign matrix or an unpaid pass; an unpaid pass is moved to uniform blocks by `KeyBinding`
at its prefix, where the writers' choices and the layout are fixed, and bounded there by the deployed guarantee's
one-pass bound; and the draws are causal in the passes' correct receipts.

Frozen (no-function, hash): `RemoteSessionConservationSketchDeployed`'s game with carriers on a store per pass made by
the append-coin rule, in the random-oracle and ideal encoding models, under the per-pass hash budget,
`SquaringFloor`, `ChainSum` and `KeyBinding`. -/
def StackedSessionConservationSketchDeployed : Prop :=
  ∀ (n P N ρ S L q G X qtot Rin cρ reads life E es B εnum εden T M : ℕ) (ε δ : ℝ≥0∞) {Ω Ca Ch : Type} [Fintype Ω]
    [Fintype Ca] (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (slot : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Fin N ↪ Fin T × Fin M)
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S L q 16 G 2773),
    7 ≤ n → 0 < P → 0 < N → 0 < E →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.SquaringFloor Gates.p1024 es εnum εden → Assumptions.ChainSum Gates.p1024 3 B es εnum εden →
    E ≤ 3 * Gates.p1024.squarings * es → εden ≤ 2 ^ 64 * εnum →
    (∀ w η j ι, Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot) →
    (∀ w η j ι c σ, ∃ K : RateSketch.Carrier N 128,
      (∀ Es R O,
        let dv := (A.window w η).device j ι c
        let D := RateSketch.Data.ofParts Es R
        (dv.prog σ D).rebuilds O D * 256 ≤ X + K.carried.card * 256) ∧
      ∀ Es, K.provenance.inputs Es ≤ 8 * Rin * life ∧
        K.provenance.gates (encodeBitsCost Gates.p1024 3 B εnum εden) Es ≤ (cρ * reads * life : ℕ)) →
    let C := 8 * Rin * life + 1024 * (cρ * reads * life) / E
    pr (fun x : (Ω × (Fin T → Ca)) × (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctOn
        (RateSketch.programmed (RateSketch.coinStore enc choose slot) x.1 x.2 w) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).readsOn (RateSketch.programmed (RateSketch.coinStore enc choose slot) x.1 x.2 w)
              (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - (0.0015 + RateSketch.gammaSketch N 128 ρ S L (X + C))) * (n * P * N * 32768) ∨
          ∃ w, (((A.win w x.2).readsOn (RateSketch.programmed (RateSketch.coinStore enc choose slot) x.1 x.2 w)
              (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ) <
            (1 - (0.01 + RateSketch.gammaSketch N 128 ρ S L (X + C))) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * ((qtot : ℝ≥0∞) / 2 ^ 96) + (n * P : ℝ≥0∞) * δ

/-- **The remote audit with relaying side sources on a store made by the append-coin rule, with stacking: a carrier is
a side source of its circuit's bits.** `RemoteSessionConservationRelay`'s game and bound on a store made by the
append-coin rule (`Accounting.appendCoinBlocks`), one for the session: `T` appends of `M` block positions, each
append's choices `choose ω a κ_<a` reading the codec and the coins of the appends before it, its own coin keying its
blocks, and the session's blocks at the layout `slot`; under the rule the blocks are uniform but for `δ`
(`Accounting.Assumptions.KeyBinding`). The bound holds for a session whose every pass device also reads a carrier:
its side reads at odd addresses answer a carrier's memory (`Rate.PassProvenance`: a circuit's values in the pass, whose
inputs may depend on the pass's certified stream and oracle, so they may be receipts computed elsewhere), and its other
side reads and the certified stream take in at most `L` bits in every world (`PIChip.linkBitsEven`; the session's own
link bound `L_all` counts the carrier's reads too, and is any). With each carrier's circuit within the property's
budgets, an encode costing `encodeBitsCost` over the codec's field `Fc` at `k` rounds, and
`Accounting.Assumptions.SquaringFloor` at `es` nonlinear units a squaring and `Accounting.Assumptions.ChainSum`, at
`E ≤ k · S · es` nonlinear units a word of `8·bytes` bits (`S = Field.squarings Fc`), inside their floor (`ε ≥ 2^−64`,
`k ≥ 1`, `p ≥ 2^66`, `p ≡ 3 mod 4`), as in `StackedSessionConservationSketchDeployed`, the bound is
`RemoteSessionConservationRelay`'s at side link `L + C`,

  `C = 8·R_in·life + 8·bytes·(c_ρ·reads·life)/E`,

and `δ` once in its error: for every `m`, the probability that every opening of every window passes while the
session's tier reads fall short of `(1 − γ̄)` of its bits or some window's fall short of `(1 − γ_w)` of its own, with
`γ` read at side link `L + C` and slack `λ + m·b`, is at most

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·P·G·(C(N, m)·2^(L + C)/2^(h·m) + 2^q/2^λ) + δ`.

**The reading.** Reading a carrier costs no link, but its contents are its circuit's values, so the device could
rerun the circuit, taking its inputs and the bits it encodes over the link: `C` bits a pass. A carrier is thus a side
source of `C` bits, the one place a receipt relayed through the circuit's inputs can come from, at `h` bits a relayed
receipt (`RemoteSessionConservationRelay`'s term). The writers' choices are fixed before each append's coin, so they
add nothing to the side link, as in `StackedSessionConservationSketchDeployed`, and what stays outside the game is
the same: a layout steered by encodings after their coins, and a node's re-registrations. The general class's receipt
hashes the whole word, not the deployed sketch (as in `RemoteSessionConservationRelay`). The store is one for the
session, made before the session's coins: this form covers an audit whose blocks written during a session are planned
only from the next session on, the stricter of the two audit rules; the deployed sketch form
(`StackedSessionConservationSketchDeployed`) takes a store per pass. `life` is a carrier's whole provenance's, as
there.

Proof: `Accounting.SecurityProofs.StackedSessionConservationRelay`: each device is simulated by one that reruns its
carrier's circuit over the side link and answers the carrier's reads from its values, within `L + C` link bits, with
the same receipts, tier reads and completions; `RemoteSessionConservationRelay`'s argument bounds that session on any
store drawn before the coins, and `KeyBinding` moves its unpaid words' event to uniform blocks.

Frozen (no-function, hash): `RemoteSessionConservationRelay`'s game with carriers on a store made by the append-coin
rule, in the random-oracle and ideal encoding models, under `SquaringFloor`, `ChainSum` and `KeyBinding`. -/
def StackedSessionConservationRelay : Prop :=
  ∀ (n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E T M : ℕ) (cost : ℕ → ℕ)
    (Fc : Gates.Field) (k B es εnum εden : ℕ) (tier : ℕ → Fin τ) (δ : ℝ≥0∞) {Ω Ca Ch : Type} [Fintype Ω]
    [Fintype Ca] (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → Pous.Bits b)
    (choose : Ω → (a : Fin T) → ({c : Fin T // c < a} → Ca) → Ch) (slot : Fin N ↪ Fin T × Fin M)
    (A : Rate.DeployedSession n P N b h ρ S Lall q r s d G) (dbar dw : ℝ),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 → 0 < E → 0 < Fc.bytes →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    Assumptions.SquaringFloor Fc es εnum εden → Assumptions.ChainSum Fc k B es εnum εden →
    E ≤ k * Fc.squarings * es → εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ Fc.p → Fc.p % 4 = 3 →
    (∀ w η j ι c, ∃ K : Rate.PassProvenance N b h ρ (ℕ → Bool),
      (∀ Es R O v i, ((A.window w η).device j ι c).side Es R O v (2 * i + 1) = K.run Es R O i) ∧
      (∀ st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L) ∧
      ∀ Es R O, K.inputs Es R O ≤ 8 * Rin * life ∧
        K.gates (encodeBitsCost Fc k B εnum εden) Es R O ≤ (cρ * reads * life : ℕ)) →
    let C := 8 * Rin * life + 8 * Fc.bytes * (cρ * reads * life) / E
    pr (fun x : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (appendCoinBlocks enc choose slot x.1) (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier (appendCoinBlocks enc choose slot x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (dbar + Rate.Deployed.gammaPass N b ρ S (L + C) (lam + m * b) rounds cRoot F)) * (n * P * N * b) ∨
            ∃ w, (((A.win w x.2).reads tier (appendCoinBlocks enc choose slot x.1) (x.2 w).1 : ℕ) : ℝ) <
              (1 - (dw + Rate.Deployed.gammaPass N b ρ S (L + C) (lam + m * b) rounds cRoot F)) * (P * N * b)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ (L + C) / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam) + δ

/-- **The stacked remote audit at the deployed parameters with a store per pass, read against a declared circuit's
requirement.** `StackedSessionConservationSketchDeployed`'s session, hypotheses and bound, with the session's shortfall
measured against a declared circuit `W`'s requirement (`Definitions/Accounting/Requirement.lean`) over a window `win`
of engine steps. The window has at most `n·P` steps, one for each pass the session's passes are bound to (a pass names
the step whose reads it covers). At each of its steps the memory map `mm` holds `W`'s credited blocks
(`Requirement.StoreMatches`: every block holding a live value the audit credits, `W`'s live values in complete blocks,
is resident, the live values are packed in 4 KiB tiles, and each credited value is in its block's payload) with at most
`N` blocks resident, and the plan of the pass bound to it covers `W`'s reads from complete blocks
(`Requirement.PlanMatches`), of at most `N` blocks. Other blocks may be resident or planned (filler, dead values,
another workload's), and they earn `W` nothing. The probability that
every opening passes while the session's reads fall short of `(1 − γ)` of the credited live bits over the window's
steps (`Requirement.creditedBits`, summed: bit-steps) or of the bits `W` reads from complete blocks over them
(`Requirement.creditedReadBits`), at

  `γ = 0.0015 + γ_sketch(X + C)`,

is at most the stacked guarantee's bound:

  `2·2^-40 + n·ε + (n·P·G)·N/2^512 + (n·P)·qtot/2^96 + (n·P)·δ`.

**The reading.** The audit's shortfall is stated against the store's `n·P·N·32768` bits. When the store and the plans
hold `W`'s credited blocks, `W`'s credited requirement over the bound steps is at most that, since a resident block
holds at most its 32768 bits of `W`'s credited live values (`RequirementHeld`, with no block held) and a planned block
at most its bits of `W`'s reads (`RequirementMoved`). So an accepted session read at least `(1 − γ)` of the live bits
the audit credits and of the bits `W` reads from complete blocks, counted in bits; `RequirementUncredited` and
`RequirementUnplanned` bound what a block still being written leaves out. Each pass reads its own store, the plan's
blocks resident going into its step, made from the coins before it; that its tiles are the memory map's blocks, so
that the bits read are `W`'s, is the binding's (`remote_bandwidth`'s Replay and each pass's bound step), and is not in this game.

**The memory map and the append coin.** With `mm`, the execution `e` and the layout `ℓ` fixed before the seed, as the
statement takes them, which blocks are resident at each step, their indices and their order are fixed with `mm`: frees,
re-key indices, append order and filler blocks are no choice after the seed (a filler or dead block may be resident,
and earns `W` nothing, since `StoreMatches` credits only the blocks of credited values). What `StoreMatches` leaves
free, the payload bits outside credited values (padding, dead values, the blocks holding none) and the run inputs, is
an append's choice, fixed when its root commits the digests and before its coin, so it pays for no advice. What stays
outside the game is `StackedSessionConservationSketchDeployed`'s: a layout steered by encodings after their coins, and
a node's re-registrations.

Proof: `Accounting.SecurityProofs.StackedRequirementSketchDeployed`, from
`Accounting.SecurityProofs.StackedSessionConservationSketchDeployed`, with `W`'s credited live and read bits at most the
store's by `Accounting.SecurityProofs.RequirementHeld` and `Accounting.SecurityProofs.RequirementMoved`.

Frozen (no-function, hash): `StackedSessionConservationSketchDeployed`'s game, with its shortfall read against a
declared circuit's requirement by counting bounds over its wiring, execution and store, in the random-oracle and ideal
encoding models, under the per-pass hash budget, `SquaringFloor`, `ChainSum` and `KeyBinding`. -/
def StackedRequirementSketchDeployed : Prop :=
  ∀ (n P N ρ S L q G X qtot Rin cρ reads life E es B εnum εden T M : ℕ) (ε δ : ℝ≥0∞) {Ω Ca Ch : Type} [Fintype Ω]
    [Fintype Ca] (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → RateSketch.Sector))
    (choose : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (a : Fin T) → ({b : Fin T // b < a} → Ca) → Ch)
    (slot : RateSketch.SketchPrefix n P N 128 ρ 512 16 G 2773 → Fin N ↪ Fin T × Fin M)
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S L q 16 G 2773)
    {V : Type} [DecidableEq V] (W : Requirement.Wiring V) (e : Requirement.Execution V)
    (ℓ : Requirement.Store.Placement V) (mm : Requirement.Store) (plan : ℕ → Finset ℕ) (win : Window),
    7 ≤ n → 0 < P → 0 < N → 0 < E →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.SquaringFloor Gates.p1024 es εnum εden → Assumptions.ChainSum Gates.p1024 3 B es εnum εden →
    E ≤ 3 * Gates.p1024.squarings * es → εden ≤ 2 ^ 64 * εnum →
    (∀ w η j ι, Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot) →
    (∀ w η j ι c σ, ∃ K : RateSketch.Carrier N 128,
      (∀ Es R O,
        let dv := (A.window w η).device j ι c
        let D := RateSketch.Data.ofParts Es R
        (dv.prog σ D).rebuilds O D * 256 ≤ X + K.carried.card * 256) ∧
      ∀ Es, K.provenance.inputs Es ≤ 8 * Rin * life ∧
        K.provenance.gates (encodeBitsCost Gates.p1024 3 B εnum εden) Es ≤ (cρ * reads * life : ℕ)) →
    win.len ≤ n * P →
    (∀ t ∈ win.steps, Requirement.StoreMatches W e ℓ mm 32768 t ∧ (mm.resident t).card ≤ N ∧
      Requirement.PlanMatches W e ℓ (plan t) 32768 t ∧ (plan t).card ≤ N) →
    let C := 8 * Rin * life + 1024 * (cρ * reads * life) / E
    let γ := 0.0015 + RateSketch.gammaSketch N 128 ρ S L (X + C)
    pr (fun x : (Ω × (Fin T → Ca)) × (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      (∀ w t, (x.2 w).2.2 t ∈ (A.win w x.2).correctOn
        (RateSketch.programmed (RateSketch.coinStore enc choose slot) x.1 x.2 w) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).readsOn (RateSketch.programmed (RateSketch.coinStore enc choose slot) x.1 x.2 w)
              (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) < (1 - γ) * (∑ t ∈ win.steps, Requirement.creditedBits W e ℓ t : ℕ) ∨
          (((∑ w, (A.win w x.2).readsOn (RateSketch.programmed (RateSketch.coinStore enc choose slot) x.1 x.2 w)
              (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - γ) * (∑ t ∈ win.steps, Requirement.creditedReadBits W e ℓ t : ℕ)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * ((qtot : ℝ≥0∞) / 2 ^ 96) + (n * P : ℝ≥0∞) * δ

/-- **The stacked remote audit with relaying side sources, read against a declared circuit's requirement.**
`StackedSessionConservationRelay`'s session, hypotheses and bound, with the session's tier reads measured against a
declared circuit `W`'s requirement over a window `win` of at most `n·P` engine steps, as in
`StackedRequirementSketchDeployed`: at each step the memory map `mm` holds `W`'s credited blocks
(`Requirement.StoreMatches`, in blocks of `b` bits) with at most `N` blocks resident, and the plan of the pass bound to
it covers `W`'s reads from complete blocks (`Requirement.PlanMatches`), of at most `N` blocks. For every `m`, the
probability that every opening passes while the
session's tier reads fall short of `(1 − γ)` of the live bits the audit credits over the window's steps
(`Requirement.creditedBits`, summed) or of the bits `W` reads from complete blocks over them
(`Requirement.creditedReadBits`), at

  `γ = d̄ + γ_pass(L + C, λ + m·b)`,

is at most the stacked relay bound:

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·P·G·(C(N, m)·2^(L + C)/2^(h·m) + 2^q/2^λ) + δ`.

**The reading.** As in `StackedRequirementSketchDeployed`: `W`'s credited requirement over the bound steps is at most
the store's `n·P·N·b` bits when the store and the plans hold `W`'s credited blocks (`RequirementHeld`,
`RequirementMoved`), so an
accepted session's tier reads are at least `(1 − γ)` of the live bits the audit credits and of the bits `W` reads from
complete blocks. The store is one for the session, as in
`StackedSessionConservationRelay`: blocks written during a session are planned only from the next. What
`StoreMatches` leaves free is fixed before each append's coin, as there. That the session's store is the memory map's
resident blocks at each pass's step is the binding's, and is not in this game.

Proof: `Accounting.SecurityProofs.StackedRequirementRelay`, from
`Accounting.SecurityProofs.StackedSessionConservationRelay`, with `W`'s credited live and read bits at most the store's
by `Accounting.SecurityProofs.RequirementHeld` and `Accounting.SecurityProofs.RequirementMoved`.

Frozen (no-function, hash): `StackedSessionConservationRelay`'s game, with its shortfall read against a declared
circuit's requirement by counting bounds over its wiring, execution and store, in the random-oracle and ideal encoding
models, under `SquaringFloor`, `ChainSum` and `KeyBinding`. -/
def StackedRequirementRelay : Prop :=
  ∀ (n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E T M : ℕ) (cost : ℕ → ℕ)
    (Fc : Gates.Field) (k B es εnum εden : ℕ) (tier : ℕ → Fin τ) (δ : ℝ≥0∞) {Ω Ca Ch : Type} [Fintype Ω]
    [Fintype Ca] (enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → Pous.Bits b)
    (choose : Ω → (a : Fin T) → ({c : Fin T // c < a} → Ca) → Ch) (slot : Fin N ↪ Fin T × Fin M)
    (A : Rate.DeployedSession n P N b h ρ S Lall q r s d G) (dbar dw : ℝ)
    {V : Type} [DecidableEq V] (W : Requirement.Wiring V) (e : Requirement.Execution V)
    (ℓ : Requirement.Store.Placement V) (mm : Requirement.Store) (plan : ℕ → Finset ℕ) (win : Window),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 → 0 < E → 0 < Fc.bytes →
    Assumptions.KeyBinding T M δ enc →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    Assumptions.SquaringFloor Fc es εnum εden → Assumptions.ChainSum Fc k B es εnum εden →
    E ≤ k * Fc.squarings * es → εden ≤ 2 ^ 64 * εnum → 0 < k → 2 ^ 66 ≤ Fc.p → Fc.p % 4 = 3 →
    (∀ w η j ι c, ∃ K : Rate.PassProvenance N b h ρ (ℕ → Bool),
      (∀ Es R O v i, ((A.window w η).device j ι c).side Es R O v (2 * i + 1) = K.run Es R O i) ∧
      (∀ st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L) ∧
      ∀ Es R O, K.inputs Es R O ≤ 8 * Rin * life ∧
        K.gates (encodeBitsCost Fc k B εnum εden) Es R O ≤ (cρ * reads * life : ℕ)) →
    win.len ≤ n * P →
    (∀ t ∈ win.steps, Requirement.StoreMatches W e ℓ mm b t ∧ (mm.resident t).card ≤ N ∧
      Requirement.PlanMatches W e ℓ (plan t) b t ∧ (plan t).card ≤ N) →
    let C := 8 * Rin * life + 8 * Fc.bytes * (cρ * reads * life) / E
    let γ := dbar + Rate.Deployed.gammaPass N b ρ S (L + C) (lam + m * b) rounds cRoot F
    pr (fun x : (Ω × (Fin T → Ca)) × (Fin n → Rate.WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (appendCoinBlocks enc choose slot x.1) (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier (appendCoinBlocks enc choose slot x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - γ) * (∑ t ∈ win.steps, Requirement.creditedBits W e ℓ t : ℕ) ∨
            (((∑ w, (A.win w x.2).reads tier (appendCoinBlocks enc choose slot x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - γ) * (∑ t ∈ win.steps, Requirement.creditedReadBits W e ℓ t : ℕ)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ (L + C) / 2 ^ (h * m) + (2 ^ q : ℝ≥0∞) / 2 ^ lam) + δ

end Accounting.Guarantees
