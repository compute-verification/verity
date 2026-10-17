import Definitions.Accounting.Carrier
import Models.Accounting.Assumptions.RateAuditDeployed
import Models.Accounting.Assumptions.Stacking

/-!
# Stacking: carriers are a side source of bounded capacity (trusted)

Stacking needs a carrier the audit counts: device memory whose contents were chosen after the seed (a credited KV page)
and hold other blocks' encodings, so that one read serves several receipts. In the ideal encoding model a carrier's
contents are a circuit's values, which depend on the encodings only through the circuit's inputs and its own encodes
(`Definitions/Accounting/Carrier.lean`). Two guarantees, whose proofs are in `Security/Proofs/Accounting/Stacking.lean`:

* `CarrierBits`, the carrier lemma: memory holding a circuit's values holds at most its input and encoded bits about
  the store, `I + q·b` for `I` input bits and `q` blocks encoded.
* `StackedSessionConservationSketchDeployed`: the remote audit's conservation at the deployed parameters
  (`RemoteSessionConservationSketchDeployed`) for devices whose sectors beyond the on-chip state come from carriers,
  with the on-chip state `S` (equivalently the side link's free capacity) raised by the carriers' bits under the
  property: `8·R_in·life` input bits and `b·(c_ρ·reads·life)/rEnc` encoded bits over a carrier's life of `life`
  windows (the property's `L`; `L` here is the side link's bits, as in every rate-audit statement).

They rest on the remote audit's random oracle and named assumptions, and on one more:
`Accounting.Assumptions.CircuitEncodeHardness` (encoding `x` of a block's `b` bits costs `x/b` of `rEnc` roundings).

**Idioms.** The carrier lemma is the counting step of incompressible encodings (Moran–Wichs, CRYPTO 2020: an
adversary's state that recovers the encoding is at least its entropy) and of proofs of replication (Fisch, SealStack,
2019: a replica stored as a function of other data is worth at most that data's bits), here with the circuit's
inputs and encodes as the only channels, as red-blue pebbling (Hong–Kung 1981; Ren–Devadas, bandwidth-hard functions,
TCC 2017) counts only what crosses the fast-memory boundary. The encode's price in roundings is an arithmetic-circuit
lower bound, assumed per block as `WorkHardness` assumes the codec's price.
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
`X ≤ q·b` for `q` encodes of at most a block's `b` bits each, and under
`Accounting.Assumptions.CircuitEncodeHardness` a budget of `R` roundings gives `X ≤ b·R/rEnc`. The count is over the
prover's choices too: an input is any function of the store, so a carrier is a side source of `I + X` bits, whatever
the prover feeds.

Proof: `Accounting.SecurityProofs.CarrierBits`, by induction on the provenance: a step of `x` bits splits the stores
into at most `2^x` classes, in each of which the rest has `x` bits fewer.

Frozen (no-function): a counting bound over abstract provenances, about no function of the code. -/
def CarrierBits : Prop :=
  ∀ (N m I X : ℕ) {α : Type} (c : RateSketch.Provenance N m α),
    (∀ E, c.inputs E ≤ I ∧ c.encoded E ≤ X) →
    Nat.card (Set.range fun E => c.run E) ≤ 2 ^ (I + X)

/-- **The remote audit at the deployed parameters, with stacking: carriers are on-chip state of bounded size.**
`RemoteSessionConservationSketchDeployed`'s game and bound for a session whose devices hold any sectors (at any on-chip
budget `T` of the session's type), provided each device's held sectors beyond `S` bits are carried by a carrier
(`RateSketch.Carrier`: memory whose contents are a circuit's values, which decode to those sectors at every store)
whose circuit, over the carrier's life of `life` windows, takes in at most `R_in` input bytes a window and spends at
most `c_ρ` roundings a byte of certified read, at `reads` bytes a window (the property's parameters; its lifetime
cap is `life`, since `L` is the side link's bits here). Under
`Accounting.Assumptions.CircuitEncodeHardness 32768 rEnc cost` the shortfall terms are the deployed guarantee's at
on-chip state `S + C`,

  `C = 8·R_in·life + 32768·(c_ρ·reads·life)/rEnc`,

the carriers' bits: inputs, and the bits of 4 KiB tiles the circuit can encode with its roundings. The error is the
deployed guarantee's:

  `2·2^-40 + n·ε + (n·P·G)·N/2^512 + (n·P)·qtot/2^96`.

**The reading.** `C` adds to the side capacity `S + (L − ρ·256)` as one more source: a carrier is a side source of
`I + q·b` bits (`CarrierBits`), with `I = 8·R_in·life` and `q = c_ρ·reads·life/rEnc` blocks. In the verbatim class a
carrier's sectors reach the chip whole and at no read, as held sectors do, so a stacked device is a device of the
deployed game at on-chip state `S + C`. The circuit is each carrier's whole provenance: the property's isolation is
what bounds it by the inputs and roundings of `life` windows, and a carrier copied forward from older ones is one
carrier whose life is the chain's.

Proof: `Accounting.SecurityProofs.StackedSessionConservationSketchDeployed`, from
`Accounting.SecurityProofs.RemoteSessionConservationSketchDeployed` at on-chip state `S + C`, each carrier's sectors
counted by `CarrierBits`.

Frozen (no-function, hash): `RemoteSessionConservationSketchDeployed`'s game with carriers, in the random-oracle and
ideal encoding models, under the per-pass hash budget and `CircuitEncodeHardness`. -/
def StackedSessionConservationSketchDeployed : Prop :=
  ∀ (n P N ρ T S L q G X qtot Rin cρ reads life rEnc : ℕ) (cost : ℕ → ℕ) (ε : ℝ≥0∞)
    (A : RateSketch.SketchDeployedSession n P N 128 ρ 512 T L q 16 G 2773),
    7 ≤ n → 0 < P → 0 < N → 0 < rEnc →
    Assumptions.SketchRank 16 128 12 ε →
    Assumptions.CircuitEncodeHardness 32768 rEnc cost →
    (∀ w η j ι c E σ R O,
      let B := (A.window w η).device j ι c
      let D := RateSketch.Data.ofParts E R
      (B.prog σ D).rebuilds O D * 256 ≤ X) →
    (∀ w η j ι, Assumptions.SketchHashBudget ((A.window w η).device j ι) qtot) →
    (∀ w η j ι c, ∃ K : RateSketch.Carrier N 128,
      (((A.window w η).device j ι c).held \ K.carried).card * 256 ≤ S ∧
        ∀ E, K.provenance.inputs E ≤ 8 * Rin * life ∧ K.provenance.roundings cost E ≤ cρ * reads * life) →
    let C := 8 * Rin * life + 32768 * (cρ * reads * life) / rEnc
    pr (fun x : RateSketch.SketchStore N 128 ×
        (Fin n → RateSketch.SketchWindowCoins P N 128 ρ 512 16 G 2773) =>
      (∀ w t, (x.2 w).2.2 t ∈
        (A.win w x.2).correctAll x.1 (x.2 w).1 (x.2 w).2.1) ∧
        ((((∑ w, (A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ)) : ℝ) <
            (1 - (0.0015 + RateSketch.gammaSketch N 128 ρ (S + C) L X)) * (n * P * N * 32768) ∨
          ∃ w, (((A.win w x.2).reads x.1 (x.2 w).1 (x.2 w).2.1 : ℕ) : ℝ) <
            (1 - (0.01 + RateSketch.gammaSketch N 128 ρ (S + C) L X)) * (P * N * 32768)))
      ≤ 2 / 2 ^ 40 + (n : ℝ≥0∞) * ε +
        (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
          (n * P : ℝ≥0∞) * ((qtot : ℝ≥0∞) / 2 ^ 96)

end Accounting.Guarantees
