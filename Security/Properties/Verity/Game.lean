import CoreOnsite
import Properties.Composition
import Definitions.Accounting.JointPeriod
import Definitions.Core.Game
import Definitions.Hash.RowLeaf
import Flock.Hash

/-!
# Soundness as a causal game at onsite's compiled verifier, and each subprotocol's target

The verifier of record is one compiled Lean function, `Core.Onsite.protocol`'s `answer`, the composition
(`Core.Composition.product`) of the subprotocols that decide a period. Five of the six are its components today, one per
`Accounting.Audit`:

| subprotocol | `Audit` | component |
| --- | --- | --- |
| sampled proofs | `.sampled` | `SampledProofs.OneStage.protocol` |
| PoUW | `.pouw` | `PouwCheck.protocol` |
| PoUS | `.pous` | `PousCheck.RunProtocol.protocol` |
| the bandwidth audit | `.poub` | `RemoteAudit.Protocol.protocol` |
| the network certifier | `.network` | `NetworkCheck.Period.protocol` |

The sixth, the wipe (`MemoryChallenger.WipeProtocol.protocol`), is composed with them in `Properties.Verity.Wipe`.

The statement's form is soundness's rulings R1, R4 and R5 (26 Oct):

* **A causal game (R5).** The prover sends the parameters `θ` and the commitment `c` first, so every statement is
  `∀ θ c`, and then plays `play (rounds θ) θ c`, a `Verity.Game`: in each of the device's rounds the prover sends the
  bytes (a root, a transcript digest) that the round's release answers, and then the device draws and releases that
  round's coins (`releases`); the evidence comes last. A strategy (`Verity.Game.Strategy`) is a function of the coins
  released so far and never of later ones, as in C-Flock's live rounds and the game `SampledKSLive*` is stated over.
* **The claim read off the commitment (R4).** A value is opened at its registered position (`place`: the hm96 row leaf
  in a frame-v3-sha512 tree against the commitment's root, its rank and its bit offset in the row) when an opening
  exists that the row-leaf verifier accepts under the compiled SHA-512 (`Opens`, over `Hash.RowLeaf.accepts`). Sampled
  proofs' claim (`CommitHolds`) is that every transcript the commitment opens to has at most an `ε` share of
  `circuitOf`'s units wrong, where a unit with a committed input or output that has no valid opening counts as wrong
  (`unbound`). An honest commitment opens every committed value to the evaluation, and then no unit is unbound.
* **The break.** Two different values opened at one position (`Collides`: by `Hash.Guarantees.RowBinding`, a SHA-512
  collision among the messages the two checks hashed), or a component that accepts on a release missing from the
  device's log (`Forged`: a signature under the device's key on a message it never signed, an LMS forgery).
* **Each component's coins from the device's log.** Subprotocol `a`'s verdict (`Verdict`) is its compiled component's
  acceptance of what the prover sent, with every release its check read (`coinsRead`) in the device's log.
* **The claims read `circuitOf`.** The circuit, its partition and its layout are the parameters' (`circuitOf`, `partsOf`,
  `place`), never the prover's world.

Each subprotocol's target (`SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`) is `Target a`: at every `θ`, `c` and strategy, the probability that `a`'s verdict accepts while
its claim fails, with no break, is at most its published error. `Verity.Soundness` (`Properties.Verity.Soundness`)
composes them.

The names a subprotocol defines are `Vocab`'s fields: `circuitOf`, `units`, `partsOf` and `place` (ONE-7), `rounds` and
`coinsRead` (ONE-4), `world` (the runs PoUW, PoUS, the bandwidth audit and the network certifier realize: ONE-16,
ONE-10, ONE-11 and ONE-12), and `certOf`, `priceOf` and `delta` (what the product publishes, functions of the
parameters alone: ONE-19 for sampled proofs).
-/

namespace Verity

open Accounting Core
open scoped ENNReal

/-- The compiled component of onsite that decides `Audit` `a`. -/
def component : Audit → Protocol
  | .sampled => SampledProofs.OneStage.protocol
  | .pouw => PouwCheck.protocol
  | .pous => PousCheck.RunProtocol.protocol
  | .poub => RemoteAudit.Protocol.protocol
  | .network => NetworkCheck.Period.protocol

/-- What the prover sends onsite's verifier of record: the parameters, the commitment and the evidence, as bytes. -/
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray

/-- **The verifier of record accepts what the prover sent.** -/
def Accepts (s : Sent) : Prop :=
  (Onsite.protocol.answer s.params s.commitment s.evidence).verdict = .accept

/-- The protocol `P` accepts its own case in what the prover sent: onsite's three byte strings decode, and one of their
cases is `P`'s, as one of onsite's components, which decodes under the parameters onsite names for it, and `P`'s check
accepts it. -/
def CaseAccepts (P : Protocol) (s : Sent) : Prop :=
  ∃ (ps : List Composition.Named) (c : ByteArray) (ev : List Composition.Case),
    (canonicalJson s.params >>= Composition.decodeParams) = .ok ps ∧ Composition.decodeCommitment s.commitment = .ok c ∧
      Composition.decodeEvidence s.evidence = .ok ev ∧
      ∃ x ∈ Properties.Composition.triples Onsite.components ps ev, x.1 = P ∧
        Properties.Composition.DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence

/-- `Audit` `a`'s component accepts its own case in what the prover sent. -/
def ComponentAccepts (a : Audit) (s : Sent) : Prop := CaseAccepts (component a) s

/-- One release of the challenger device: the bytes it answers, and the coins it released after them. -/
structure Release where
  answers : ByteArray
  coins : List Bool

/-- **The device's releases, in order**: for a round of `n` coin bits, the prover sends the bytes the release answers,
then the device draws `n` uniform bits and releases them. The output is the device's log. -/
def releases : List ℕ → Game (List Release)
  | [] => .ret []
  | n :: ns => .send ByteArray fun b => .coin (Fin n → Bool) fun x =>
      (releases ns).map fun log => ⟨b, List.ofFn x⟩ :: log

/-- How a play ends: the parameters and commitment sent first, the device's log, and the evidence sent last. -/
structure Outcome where
  params : ByteArray
  commitment : ByteArray
  log : List Release
  evidence : ByteArray

/-- What the prover sent onsite in the play that ended with `o`. -/
def Outcome.sent (o : Outcome) : Sent := ⟨o.params, o.commitment, o.evidence⟩

/-- **The causal game after the parameters `θ` and the commitment `c`**: the device's rounds `ns`, each release after
the bytes it answers, then the evidence. -/
def play (ns : List ℕ) (θ c : ByteArray) : Game Outcome :=
  (releases ns).bind fun log => .send ByteArray fun e => .ret ⟨θ, c, log, e⟩

/-- SHA-512, the compiled one (`Flock.Sha512.hash`), as a 64-byte digest. -/
def sha512 (m : Hash.Msg) : Hash.Digest := fun i => (Flock.Sha512.hash ⟨m.toArray⟩).data.getD i.val 0

/-- A value's registered position: the row-leaf verifier (the frame-v3 domain, the hm96 key and the row format), the
commitment's root it opens against, the leaf's rank, and the value's first bit in the row. -/
structure Place where
  verifier : Hash.RowLeaf.Verifier
  root : Hash.Digest
  rank : ℕ
  offset : ℕ

/-- The fields of a period's run that the prover's world realizes in a play: whose account each resource is charged to,
the window, the device schedule, and what the network certifiers saw. -/
structure World {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  acct : Resource → α
  win : Window
  schedule : Schedule ι α
  fits : schedule.Fits d
  commWithin : ℝ → Prop

/-- **The names the subprotocols define.** The circuit, its units, partition and registered positions from the
parameters (ONE-7); the device's rounds, and the releases each component's check read, decoded from the device's signed
log entries in what the prover sent (ONE-4); the world each play realizes (ONE-10, ONE-11, ONE-12, ONE-16); and the
certificates, price and errors the product publishes, from the parameters alone (ONE-19). -/
structure Vocab {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  circuitOf : ByteArray → Verity.Circuit
  units : ByteArray → ℕ
  partsOf : (θ : ByteArray) → Verity.Partition (circuitOf θ) (units θ)
  place : (θ c : ByteArray) → Fin (circuitOf θ).N → Option Place
  rounds : ByteArray → List ℕ
  coinsRead : Audit → Sent → List Release
  world : (θ c : ByteArray) → Game.Strategy (play (rounds θ) θ c) → Outcome → World d α
  certOf : ByteArray → Certificates
  priceOf : ByteArray → Price
  delta : ByteArray → Audit → ℝ≥0∞

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- The game after `θ` and `c`. -/
def game (θ c : ByteArray) : Game Outcome := play (V.rounds θ) θ c

/-- **The commitment `c` opens gate `g` to `v`**: an opening of the row leaf at `g`'s registered position that the
row-leaf verifier accepts under SHA-512, whose row holds `v`'s bits from the position's offset. -/
def Opens (θ c : ByteArray) (g : Fin (V.circuitOf θ).N) (v : BitVec ((V.circuitOf θ).w g)) : Prop :=
  ∃ p, V.place θ c g = some p ∧ ∃ o : Hash.RowLeaf.Opening,
    Hash.RowLeaf.accepts sha512 sha512 p.verifier p.root p.rank o ∧
      ∀ i : Fin ((V.circuitOf θ).w g), o.row[p.offset + i.val]? = some (v.getLsbD i.val)

open Classical in
/-- The units the commitment doesn't bind correctly in the transcript `X`: those not correct in `X`, and those with a
committed input or output that `c` does not open to its value in `X`. -/
noncomputable def unbound (θ c : ByteArray) (X : (V.circuitOf θ).Transcript) : Finset (Fin (V.units θ)) :=
  Finset.univ.filter fun u =>
    ¬ (V.partsOf θ).Correct X u ∨ ∃ g ∈ (V.partsOf θ).io u, ¬ V.Opens θ c g (X g)

/-- **Sampled proofs' claim, read off the commitment (R4)**: every transcript the commitment opens to has at most an
`ε` share of the circuit's units unbound. -/
def CommitHolds (θ c : ByteArray) : Prop :=
  ∀ X : (V.circuitOf θ).Transcript, (∀ g v, V.Opens θ c g v → X g = v) →
    ((V.unbound θ c X).card : ℝ) ≤ (V.certOf θ).ε * V.units θ

/-- **The claim subprotocol `a` certifies** in the play of `σ` that ended with `o`: sampled proofs' on the commitment,
and the others' on the world the play realized (at least `W` of work at the published price, at least `S` bits of device
memory held on average, at least `Q` bits read, at most `K` bits through the network certifiers). -/
def Holds (θ c : ByteArray) (σ : Game.Strategy (V.game θ c)) : Audit → Outcome → Prop
  | .sampled, _ => V.CommitHolds θ c
  | .pouw, o => (V.certOf θ).W ≤
      ((V.world θ c σ o).schedule.work (V.world θ c σ o).win ((V.world θ c σ o).acct .compute) (V.priceOf θ) : ℝ)
  | .pous, o => (V.certOf θ).S * (V.world θ c σ o).win.len ≤
      ((V.world θ c σ o).schedule.held (V.world θ c σ o).win ((V.world θ c σ o).acct .capacity) : ℝ)
  | .poub, o => (V.certOf θ).Q ≤
      ((V.world θ c σ o).schedule.read (V.world θ c σ o).win ((V.world θ c σ o).acct .bandwidth) : ℝ)
  | .network, o => (V.world θ c σ o).commWithin (V.certOf θ).K

/-- **Subprotocol `a`'s verdict**: its compiled component accepts what the prover sent, and every release its check
read is in the device's log. -/
def Verdict (a : Audit) (o : Outcome) : Prop :=
  ComponentAccepts a o.sent ∧ ∀ r ∈ V.coinsRead a o.sent, r ∈ o.log

/-- A component accepts on a release missing from the device's log: a signature under the device's key on a message
the device never signed. -/
def Forged (o : Outcome) : Prop :=
  ∃ a, ComponentAccepts a o.sent ∧ ∃ r ∈ V.coinsRead a o.sent, r ∉ o.log

/-- Two different values opened at one registered position. -/
def Collides (θ c : ByteArray) : Prop :=
  ∃ g v v', v ≠ v' ∧ V.Opens θ c g v ∧ V.Opens θ c g v'

/-- **The break**: a forged release, or two openings of one position. -/
def Break (θ c : ByteArray) (o : Outcome) : Prop := V.Forged o ∨ V.Collides θ c

/-- **Soundness, in the causal game.** At all parameters `θ` and commitments `c`, and for every strategy, the
probability that onsite's compiled verifier accepts while one of the period's claims fails (more than an `ε` share of
the units unbound in some transcript the commitment opens to, less than `W` of work, fewer than `S` bits held on average,
fewer than `Q` bits read, or more than `K` bits through the network certifiers), with no break, is at most the published
errors summed. -/
def Soundness : Prop :=
  ∀ (θ c : ByteArray) (σ : Game.Strategy (V.game θ c)),
    Game.prob (fun o => Accepts o.sent ∧ (∃ a, ¬ V.Holds θ c σ a o) ∧ ¬ V.Break θ c o) (V.game θ c) σ ≤
      ∑ a, V.delta θ a

/-- **Subprotocol `a`'s target**: at all `θ` and `c`, and for every strategy, the probability that `a`'s verdict accepts
while its claim fails, with no break, is at most its published error. -/
def Target (a : Audit) : Prop :=
  ∀ (θ c : ByteArray) (σ : Game.Strategy (V.game θ c)),
    Game.prob (fun o => V.Verdict a o ∧ ¬ V.Holds θ c σ a o ∧ ¬ V.Break θ c o) (V.game θ c) σ ≤ V.delta θ a

/-- **Sampled proofs' target (ONE-8)**: at most an `ε` share of the units unbound in every transcript the commitment
opens to. -/
def SampledProofsSound : Prop := V.Target .sampled

/-- **PoUW's target (ONE-16)**: at least `W` of work at the published price. -/
def PouwSound : Prop := V.Target .pouw

/-- **PoUS's target (ONE-10)**: at least `S` bits of device memory held on average. -/
def PousSound : Prop := V.Target .pous

/-- **The bandwidth audit's target (ONE-11)**: at least `Q` bits read from device memory. -/
def BandwidthAuditSound : Prop := V.Target .poub

/-- **The network certifier's target (ONE-12)**: at most `K` bits through the network certifiers. -/
def NetworkCertifierSound : Prop := V.Target .network

end Vocab

end Verity
