import Definitions.Accounting.Carrier
import Models.Accounting.Assumptions.RateAudit
import Models.Accounting.Assumptions.RateAuditDeployed
import Models.Accounting.Assumptions.Stacking
import Models.Accounting.Assumptions.EncodeChain
import Models.Accounting.Guarantees.Requirement

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
  (`RemoteSessionConservationSketchDeployed`) on a programmed store, whose blocks written after the seed are a
  circuit's values under bound keys (`Accounting.Assumptions.KeyBinding`), for devices whose rebuilds beyond `X` are
  their carriers' sectors. The stacked stock is charged in every pass's rebuilds: `X` grows by the carriers' bits under
  the property, `8·R_in·life` input bits and `1024·(c_ρ·reads·life)/E` encoded bits over a carrier's life of `life`
  windows (the property's `L`; `L` here is the side link's bits, as in every rate-audit statement).
* `StackedSessionConservationRelay`: the relay form, in the general class (`RemoteSessionConservationRelay`), where a
  carrier's inputs may depend on the pass's oracle and so relay receipts: a carrier is a side source of the same bits.
* `StackedRequirementSketchDeployed` and `StackedRequirementRelay`: the same sessions, with the shortfall read against a
  declared circuit `W`'s requirement (`Definitions/Accounting/Requirement.lean`) instead of the store: when the memory
  map and every pass's plan are `W`'s at each step the passes are bound to (`Requirement.StoreMatches`,
  `Requirement.PlanMatches`), an accepted session read at least `(1 − γ)` of the bits `W` holds over those steps
  (`Requirement.held`) and of the bits it reads (`Requirement.read`).

The stacked sessions charge an encode at the gate model's cost (`encodeBitsCost`, budgeted gates of
`Definitions/Accounting/Gates`, `⊤` where the model has no circuit computing the words). They rest on the remote
audit's random oracle and named assumptions, and on two more: the gate model's conjecture
`Accounting.Assumptions.EncodeChainHardness` (computing a codec word costs `E` budgeted gates) and
`Accounting.Assumptions.KeyBinding` (for each choice of its writers, the programmed store is a uniform store but for
`δ`). What the writers choose by looking at encodings is advice, `a` bits, which the sessions pay for.

**Idioms.** The carrier lemma is the counting step of incompressible encodings (Moran–Wichs, CRYPTO 2020: an
adversary's state that recovers the encoding is at least its entropy) and of proofs of replication (Fisch, SealStack,
2019: a replica stored as a function of other data is worth at most that data's bits), here with the circuit's
inputs and encodes as the only channels, as red-blue pebbling (Hong–Kung 1981; Ren–Devadas, bandwidth-hard functions,
TCC 2017) counts only what crosses the fast-memory boundary. The encode's price in budgeted gates is a circuit lower
bound, assumed per word as `WorkHardness` assumes the codec's price. The advice is the bit-fixing model's (Unruh,
CRYPTO 2007; Coretti–Dodis–Guo–Steinberger, EUROCRYPT 2018): an adversary whose choices depend on the oracle through
`a` bits gains at most a factor `2^a`.
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
`X ≤ q·b` for `q` encodes of at most a block's `b` bits each, and under `Accounting.Assumptions.EncodeChainHardness` at
`E` budgeted gates a word of `8·bytes` bits a budget of `R` budgeted gates gives `X ≤ 8·bytes·R/E`. The count is over
the prover's choices too: an input is any function of the store, so a carrier is a side source of `I + X` bits, whatever
the prover feeds.

Proof: `Accounting.SecurityProofs.CarrierBits`, by induction on the provenance: a step of `x` bits splits the stores
into at most `2^x` classes, in each of which the rest has `x` bits fewer.

Frozen (no-function): a counting bound over abstract provenances, about no function of the code. -/
def CarrierBits : Prop :=
  ∀ (N m I X : ℕ) {α : Type} (c : RateSketch.Provenance N m α),
    (∀ E, c.inputs E ≤ I ∧ c.encoded E ≤ X) →
    Nat.card (Set.range fun E => c.run E) ≤ 2 ^ (I + X)

/-- **The remote audit at the deployed parameters on a programmed store, with stacking: carriers are rebuilds of
bounded size in every pass, and the writers' choices are advice.** `RemoteSessionConservationSketchDeployed`'s game
and bound on a programmed store `view (choice ω) ω` (`ω` uniform, before the session's coins): the store the writers
make with their choices `choice ω`, its blocks registered before the seed and its blocks written after it, whose
payloads are a circuit's values, under bound keys. There are at most `2^a` choices, and for each the store is a uniform
store but for `δ` (`Accounting.Assumptions.KeyBinding`). Every pass of the session rebuilds at most `X` bits plus one
sector for each sector of a carrier (`RateSketch.Carrier`: memory whose contents are a circuit's values, which decode to
those sectors at every store). Each carrier's circuit, over its life of `life` windows, takes in at most `R_in` input
bytes a window and spends at most `c_ρ` budgeted gates a byte of certified read, at `reads` bytes a window (the
property's parameters; its lifetime cap is `life`, since `L` is the side link's bits here), an encode of `x` bits
costing `encodeBitsCost` at the deployed codec (p1024, 3 rounds) in the gate model's restricted class (budgeted fan-in
at most `fan`, values within `wid` bits, on more than `εnum/εden` of the draws), `⊤` where the model has no circuit
computing the words. Under `Accounting.Assumptions.EncodeChainHardness` at `E` budgeted gates a word, the shortfall
terms are the deployed guarantee's at rebuilt bits `X + C` a pass,

  `C = 8·R_in·life + 1024·(c_ρ·reads·life)/E`,

the carriers' bits: inputs, and the bits of the 1024-bit words the circuit can encode with its budgeted gates. The
error is the deployed guarantee's draws, and its other terms plus `δ` for each choice:

  `2·2^-40 + 2^a·(n·ε + (n·P·G)·N/2^512 + (n·P)·qtot/2^96 + δ)`.

**The reading.** A carrier serves every pass of every window, so its stock is charged in each pass: `C` adds to the
side capacity `S + (L − ρ·256) + X` as one more source, a carrier being a source of `I + q·b` bits (`CarrierBits`),
with `I = 8·R_in·life` and `q·b ≤ 1024·c_ρ·reads·life/E`. In the verbatim class a sector the device decodes from a
carrier is one it computes rather than reads at its position, a rebuild. The circuit is each carrier's whole
provenance: the property's isolation is what bounds it by the inputs and gates of `life` windows, and a carrier
copied forward from older ones is one carrier whose life is the chain's. Key binding is what keeps a written block's
encoding from being chosen: what its payload carries about other blocks is its carrier, and what its writers choose by
looking at encodings (grinding, or a write that adapts to them) is the advice, at most the `log₂` of the candidates
encoded for each block, so within the encode budget. Here the advice multiplies the terms that read the store's
distribution: `a` bits take `a` bits off the 96-bit hash margin and multiply `ε`, where the relay form
(`StackedSessionConservationRelay`) charges them as side capacity. The choices are made before the session's coins; a
write that adapts to the coins is not in this game, whose store is fixed for the session.

Proof: `Accounting.SecurityProofs.StackedSessionConservationSketchDeployed`, from the deployed guarantee's argument at
rebuilt bits `X + C` on any store drawn before the coins, each carrier's sectors counted by `CarrierBits` with an
encode's cost bounded by `EncodeChainHardness`, and the unpaid words' event moved to a uniform store for each choice by
`KeyBinding` and summed over the choices.

Frozen (no-function, hash): `RemoteSessionConservationSketchDeployed`'s game with carriers on a programmed store, in
the random-oracle and ideal encoding models, under the per-pass hash budget, `EncodeChainHardness` and
`KeyBinding`. -/
def StackedSessionConservationSketchDeployed : Prop :=
  ∀ (n P N ρ S L q G X qtot Rin cρ reads life E fan wid εnum εden a : ℕ) (ε δ : ℝ≥0∞) {Ω Ch : Type} [Fintype Ω]
    [Fintype Ch] (view : Ch → Ω → RateSketch.SketchStore N 128) (choice : Ω → Ch)
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S L q 16 G 2773),
    7 ≤ n → 0 < P → 0 < N → 0 < E → Fintype.card Ch ≤ 2 ^ a →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.KeyBinding δ view →
    Assumptions.EncodeChainHardness Gates.p1024 3 fan wid E εnum εden →
    (∀ w η j ι, Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot) →
    (∀ w η j ι c, ∃ K : RateSketch.Carrier N 128,
      (∀ Es σ R O,
        let B := (A.window w η).device j ι c
        let D := RateSketch.Data.ofParts Es R
        (B.prog σ D).rebuilds O D * 256 ≤ X + K.carried.card * 256) ∧
      ∀ Es, K.provenance.inputs Es ≤ 8 * Rin * life ∧
        K.provenance.gates (encodeBitsCost Gates.p1024 3 fan wid εnum εden) Es ≤ (cρ * reads * life : ℕ)) →
    let C := 8 * Rin * life + 1024 * (cρ * reads * life) / E
    pr (fun x : Ω × (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      (∀ w t, (x.2 w).2.2 t ∈
        (A.win w x.2).correctAll (view (choice x.1) x.1) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).reads (view (choice x.1) x.1) (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - (0.0015 + RateSketch.gammaSketch N 128 ρ S L (X + C))) * (n * P * N * 32768) ∨
          ∃ w, (((A.win w x.2).reads (view (choice x.1) x.1) (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ) <
            (1 - (0.01 + RateSketch.gammaSketch N 128 ρ S L (X + C))) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + 2 ^ a * ((n : ℝ≥0∞) * ε +
        (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) + (n * P : ℝ≥0∞) * ((qtot : ℝ≥0∞) / 2 ^ 96) + δ)

/-- **The remote audit with relaying side sources on a programmed store, with stacking: a carrier is a side source of
its circuit's bits, and the writers' choices are side capacity.** `RemoteSessionConservationRelay`'s game and bound on
a programmed store `view (choice ω) ω`, made with one of at most `2^a` choices, for each a uniform store but for `δ`
(`Accounting.Assumptions.KeyBinding`), for a session whose every pass device also reads a carrier: its side reads at
odd addresses answer a carrier's memory (`Rate.PassProvenance`: a circuit's values in the pass, whose inputs may depend
on the pass's certified stream and oracle, so they may be receipts computed elsewhere), and its other side reads and
the certified stream take in at most `L` bits in every world (`PIChip.linkBitsEven`; the session's own link bound
`L_all` counts the carrier's reads too, and is any). With each carrier's circuit within the property's budgets, an
encode costing `encodeBitsCost` over the codec's field `Fc` at `k` rounds, and
`Accounting.Assumptions.EncodeChainHardness` at `E` budgeted gates a word of `8·bytes` bits, as in
`StackedSessionConservationSketchDeployed`, the bound is `RemoteSessionConservationRelay`'s at side link `L + C`,

  `C = 8·R_in·life + 8·bytes·(c_ρ·reads·life)/E`,

with the advice's `a` bits added to the side link and the queries in its error: for every `m`, the probability that
every opening of every window passes while the session's tier reads fall short of `(1 − γ̄)` of its bits or some
window's fall short of `(1 − γ_w)` of its own, with `γ` read at side link `L + C` and slack `λ + m·b`, is at most

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·P·G·(C(N, m)·2^(L + C + a)/2^(h·m) + 2^(q + a)/2^λ) + 2^a·δ`.

**The reading.** Reading a carrier costs no link, but its contents are its circuit's values, so the device could
rerun the circuit, taking its inputs and the bits it encodes over the link: `C` bits a pass. A carrier is thus a side
source of `C` bits, the one place a receipt relayed through the circuit's inputs can come from, at `h` bits a relayed
receipt (`RemoteSessionConservationRelay`'s term). The writers' choices are `a` bits more of it: a store chosen among
`2^a` by its encodings relays as if `a` more bits crossed the side link, in the relayed receipts' term and in the
guessed tags'. The general class's receipt hashes the whole word, not the deployed sketch (as in
`RemoteSessionConservationRelay`). The choices are made before the session's coins.

Proof: `Accounting.SecurityProofs.StackedSessionConservationRelay`: each device is simulated by one that reruns its
carrier's circuit over the side link and answers the carrier's reads from its values, within `L + C` link bits, with
the same receipts, tier reads and completions; `RemoteSessionConservationRelay`'s argument bounds that session on any
store drawn before the coins, and `KeyBinding` moves its unpaid words' event to a uniform store for each choice.

Frozen (no-function, hash): `RemoteSessionConservationRelay`'s game with carriers on a programmed store, in the
random-oracle and ideal encoding models, under `EncodeChainHardness` and `KeyBinding`. -/
def StackedSessionConservationRelay : Prop :=
  ∀ (n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E a : ℕ) (cost : ℕ → ℕ)
    (Fc : Gates.Field) (k fan wid εnum εden : ℕ) (tier : ℕ → Fin τ) (δ : ℝ≥0∞) {Ω Ch : Type} [Fintype Ω]
    [Fintype Ch] (view : Ch → Ω → Rate.Store N b) (choice : Ω → Ch)
    (A : Rate.DeployedSession n P N b h ρ S Lall q r s d G) (dbar dw : ℝ),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 → 0 < E → 0 < Fc.bytes →
    Fintype.card Ch ≤ 2 ^ a →
    Assumptions.KeyBinding δ view →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    Assumptions.EncodeChainHardness Fc k fan wid E εnum εden →
    (∀ w η j ι c, ∃ K : Rate.PassProvenance N b h ρ (ℕ → Bool),
      (∀ Es R O v i, ((A.window w η).device j ι c).side Es R O v (2 * i + 1) = K.run Es R O i) ∧
      (∀ st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L) ∧
      ∀ Es R O, K.inputs Es R O ≤ 8 * Rin * life ∧
        K.gates (encodeBitsCost Fc k fan wid εnum εden) Es R O ≤ (cρ * reads * life : ℕ)) →
    let C := 8 * Rin * life + 8 * Fc.bytes * (cρ * reads * life) / E
    pr (fun x : Ω × (Fin n → Rate.WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (view (choice x.1) x.1) (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier (view (choice x.1) x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - (dbar + Rate.Deployed.gammaPass N b ρ S (L + C) (lam + m * b) rounds cRoot F)) * (n * P * N * b) ∨
            ∃ w, (((A.win w x.2).reads tier (view (choice x.1) x.1) (x.2 w).1 : ℕ) : ℝ) <
              (1 - (dw + Rate.Deployed.gammaPass N b ρ S (L + C) (lam + m * b) rounds cRoot F)) * (P * N * b)))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ (L + C + a) / 2 ^ (h * m) + (2 ^ (q + a) : ℝ≥0∞) / 2 ^ lam) +
          2 ^ a * δ

/-- **The stacked remote audit at the deployed parameters, read against a declared circuit's requirement.**
`StackedSessionConservationSketchDeployed`'s session, hypotheses and bound, with the session's shortfall measured
against a declared circuit `W`'s requirement (`Definitions/Accounting/Requirement.lean`) over a window `win` of engine
steps. The window has at most `n·P` steps, one for each pass the session's passes are bound to (a pass names the step
whose reads it covers). At each of its steps the memory map `mm` is `W`'s (`Requirement.StoreMatches`: its resident
blocks are `W`'s live blocks, packed in 4 KiB tiles, each live value in its block's payload) with at most `N` blocks
resident, and the plan of the pass bound to it is `W`'s reads (`Requirement.PlanMatches`), of at most `N` blocks. The
probability that every opening passes while the session's reads fall short of `(1 − γ)` of the bits `W` keeps live over
the window's steps (`Requirement.held`, in bit-steps) or of the bits it reads over them (`Requirement.read`), at

  `γ = 0.0015 + γ_sketch(X + C)`,

is at most the stacked guarantee's bound:

  `2·2^-40 + 2^a·(n·ε + (n·P·G)·N/2^512 + (n·P)·qtot/2^96 + δ)`.

**The reading.** The audit's shortfall is stated against the store's `n·P·N·32768` bits. When the store and the plans
are `W`'s, `W`'s requirement over the bound steps is at most that, since a resident block holds at most its 32768 bits
of `W`'s live values (`RequirementHeld`, with no block held) and a planned block at most its bits of `W`'s reads
(`RequirementMoved`). So an accepted session read at least `(1 − γ)` of the bits `W` holds and of the bits it reads,
counted in bits. The reads are of the session's store `view (choice ω) ω`, the registered one throughout. That it is
the memory map's resident blocks at each pass's step, so that the bits read are `W`'s, is the pass-to-step binding's
(`remote_bandwidth`'s Replay), and is not in this game.

Proof: `Accounting.SecurityProofs.StackedRequirementSketchDeployed`, from
`Accounting.SecurityProofs.StackedSessionConservationSketchDeployed`, with `W`'s held and read bits at most the store's
by `Accounting.SecurityProofs.RequirementHeld` and `Accounting.SecurityProofs.RequirementMoved`.

Frozen (no-function, hash): `StackedSessionConservationSketchDeployed`'s game, with its shortfall read against a
declared circuit's requirement by counting bounds over its wiring, execution and store, in the random-oracle and ideal
encoding models, under the per-pass hash budget, `EncodeChainHardness` and `KeyBinding`. -/
def StackedRequirementSketchDeployed : Prop :=
  ∀ (n P N ρ S L q G X qtot Rin cρ reads life E fan wid εnum εden a : ℕ) (ε δ : ℝ≥0∞) {Ω Ch : Type} [Fintype Ω]
    [Fintype Ch] (view : Ch → Ω → RateSketch.SketchStore N 128) (choice : Ω → Ch)
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 S L q 16 G 2773)
    {V : Type} [DecidableEq V] (W : Requirement.Wiring V) (e : Requirement.Execution V) (ℓ : Requirement.Layout V)
    (mm : Requirement.Store) (plan : ℕ → Finset ℕ) (win : Window),
    7 ≤ n → 0 < P → 0 < N → 0 < E → Fintype.card Ch ≤ 2 ^ a →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.KeyBinding δ view →
    Assumptions.EncodeChainHardness Gates.p1024 3 fan wid E εnum εden →
    (∀ w η j ι, Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot) →
    (∀ w η j ι c, ∃ K : RateSketch.Carrier N 128,
      (∀ Es σ R O,
        let B := (A.window w η).device j ι c
        let D := RateSketch.Data.ofParts Es R
        (B.prog σ D).rebuilds O D * 256 ≤ X + K.carried.card * 256) ∧
      ∀ Es, K.provenance.inputs Es ≤ 8 * Rin * life ∧
        K.provenance.gates (encodeBitsCost Gates.p1024 3 fan wid εnum εden) Es ≤ (cρ * reads * life : ℕ)) →
    win.len ≤ n * P →
    (∀ t ∈ win.ticks, Requirement.StoreMatches W e ℓ mm 32768 t ∧ (mm.resident t).card ≤ N ∧
      Requirement.PlanMatches W e ℓ (plan t) 32768 t ∧ (plan t).card ≤ N) →
    let C := 8 * Rin * life + 1024 * (cρ * reads * life) / E
    let γ := 0.0015 + RateSketch.gammaSketch N 128 ρ S L (X + C)
    pr (fun x : Ω × (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      (∀ w t, (x.2 w).2.2 t ∈
        (A.win w x.2).correctAll (view (choice x.1) x.1) (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).reads (view (choice x.1) x.1) (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - γ) * Requirement.held W e win ∨
          (((∑ w, (A.win w x.2).reads (view (choice x.1) x.1) (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - γ) * Requirement.read W e win))
      ≤ 2 / 2 ^ 40 + 2 ^ a * ((n : ℝ≥0∞) * ε +
        (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) + (n * P : ℝ≥0∞) * ((qtot : ℝ≥0∞) / 2 ^ 96) + δ)

/-- **The stacked remote audit with relaying side sources, read against a declared circuit's requirement.**
`StackedSessionConservationRelay`'s session, hypotheses and bound, with the session's tier reads measured against a
declared circuit `W`'s requirement over a window `win` of at most `n·P` engine steps, as in
`StackedRequirementSketchDeployed`: at each step the memory map `mm` is `W`'s (`Requirement.StoreMatches`, in blocks of
`b` bits) with at most `N` blocks resident, and the plan of the pass bound to it is `W`'s reads
(`Requirement.PlanMatches`), of at most `N` blocks. For every `m`, the probability that every opening passes while the
session's tier reads fall short of `(1 − γ)` of the bits `W` holds over the window's steps (`Requirement.held`) or of
the bits it reads over them (`Requirement.read`), at

  `γ = d̄ + γ_pass(L + C, λ + m·b)`,

is at most the stacked relay bound:

  `(1 − d̄)^(d·n) + (1 − d_w/n)^(d·n) + n·P·G·(C(N, m)·2^(L + C + a)/2^(h·m) + 2^(q + a)/2^λ) + 2^a·δ`.

**The reading.** As in `StackedRequirementSketchDeployed`: `W`'s requirement over the bound steps is at most the
store's `n·P·N·b` bits when the store and the plans are `W`'s (`RequirementHeld`, `RequirementMoved`), so an accepted
session's tier reads are at least `(1 − γ)` of the bits `W` holds and reads. That the session's store is the memory
map's resident blocks at each pass's step is the pass-to-step binding's, and is not in this game.

Proof: `Accounting.SecurityProofs.StackedRequirementRelay`, from
`Accounting.SecurityProofs.StackedSessionConservationRelay`, with `W`'s held and read bits at most the store's by
`Accounting.SecurityProofs.RequirementHeld` and `Accounting.SecurityProofs.RequirementMoved`.

Frozen (no-function, hash): `StackedSessionConservationRelay`'s game, with its shortfall read against a declared
circuit's requirement by counting bounds over its wiring, execution and store, in the random-oracle and ideal encoding
models, under `EncodeChainHardness` and `KeyBinding`. -/
def StackedRequirementRelay : Prop :=
  ∀ (n P N b h ρ S Lall L q r s d G lam m τ rounds cRoot F Rin cρ reads life E a : ℕ) (cost : ℕ → ℕ)
    (Fc : Gates.Field) (k fan wid εnum εden : ℕ) (tier : ℕ → Fin τ) (δ : ℝ≥0∞) {Ω Ch : Type} [Fintype Ω]
    [Fintype Ch] (view : Ch → Ω → Rate.Store N b) (choice : Ω → Ch)
    (A : Rate.DeployedSession n P N b h ρ S Lall q r s d G) (dbar dw : ℝ)
    {V : Type} [DecidableEq V] (W : Requirement.Wiring V) (e : Requirement.Execution V) (ℓ : Requirement.Layout V)
    (mm : Requirement.Store) (plan : ℕ → Finset ℕ) (win : Window),
    0 < n → 0 < P → 0 < N → 0 < b → 0 < rounds * cRoot → dw ≤ 1 → 0 < E → 0 < Fc.bytes →
    Fintype.card Ch ≤ 2 ^ a →
    Assumptions.KeyBinding δ view →
    Assumptions.WorkHardness b rounds cRoot cost →
    (∀ w η j ι c, Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c)) →
    Assumptions.EncodeChainHardness Fc k fan wid E εnum εden →
    (∀ w η j ι c, ∃ K : Rate.PassProvenance N b h ρ (ℕ → Bool),
      (∀ Es R O v i, ((A.window w η).device j ι c).side Es R O v (2 * i + 1) = K.run Es R O i) ∧
      (∀ st wd, (((A.window w η).device j ι c).chip st).linkBitsEven wd [] ≤ L) ∧
      ∀ Es R O, K.inputs Es R O ≤ 8 * Rin * life ∧
        K.gates (encodeBitsCost Fc k fan wid εnum εden) Es R O ≤ (cρ * reads * life : ℕ)) →
    win.len ≤ n * P →
    (∀ t ∈ win.ticks, Requirement.StoreMatches W e ℓ mm b t ∧ (mm.resident t).card ≤ N ∧
      Requirement.PlanMatches W e ℓ (plan t) b t ∧ (plan t).card ≤ N) →
    let C := 8 * Rin * life + 8 * Fc.bytes * (cρ * reads * life) / E
    let γ := dbar + Rate.Deployed.gammaPass N b ρ S (L + C) (lam + m * b) rounds cRoot F
    pr (fun x : Ω × (Fin n → Rate.WindowCoins P N b h ρ r s d) =>
        (∀ w t, (x.2 w).2 t ∈ (A.win w x.2).correctAll (view (choice x.1) x.1) (x.2 w).1) ∧
          ((((∑ w, (A.win w x.2).reads tier (view (choice x.1) x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - γ) * Requirement.held W e win ∨
            (((∑ w, (A.win w x.2).reads tier (view (choice x.1) x.1) (x.2 w).1 : ℕ)) : ℝ) <
              (1 - γ) * Requirement.read W e win))
      ≤ ENNReal.ofReal (1 - dbar) ^ (d * n) + ENNReal.ofReal (1 - dw / n) ^ (d * n) +
        (n * P * G : ℝ≥0∞) * ((N.choose m : ℝ≥0∞) * 2 ^ (L + C + a) / 2 ^ (h * m) + (2 ^ (q + a) : ℝ≥0∞) / 2 ^ lam) +
          2 ^ a * δ

end Accounting.Guarantees
